import Mathlib

namespace power_expression_l666_66639

variable {x : ℂ} -- Define x as a complex number

theorem power_expression (
  h : x - 1/x = 2 * Complex.I * Real.sqrt 2
) : x^(2187:ℕ) - 1/x^(2187:ℕ) = -22 * Complex.I * Real.sqrt 2 :=
by sorry

end power_expression_l666_66639


namespace men_build_walls_l666_66654

-- Define the variables
variables (a b d y : ℕ)

-- Define the work rate based on given conditions
def rate := d / (a * b)

-- Theorem to prove that y equals (a * a) / d given the conditions
theorem men_build_walls (h : a * b * y = a * a * d / a) : 
  y = a * a / d :=
by sorry

end men_build_walls_l666_66654


namespace not_constant_expression_l666_66686

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

noncomputable def squared_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem not_constant_expression (A B C P G : ℝ × ℝ)
  (hG : is_centroid A B C G)
  (hP_on_AB : ∃ x, P = (x, A.2) ∧ A.2 = B.2) :
  ∃ dPA dPB dPC dPG : ℝ,
    dPA = squared_distance P A ∧
    dPB = squared_distance P B ∧
    dPC = squared_distance P C ∧
    dPG = squared_distance P G ∧
    (dPA + dPB + dPC - dPG) ≠ dPA + dPB + dPC - dPG := by
  sorry

end not_constant_expression_l666_66686


namespace maximize_angle_l666_66607

structure Point where
  x : ℝ
  y : ℝ

def A (a : ℝ) : Point := ⟨0, a⟩
def B (b : ℝ) : Point := ⟨0, b⟩

theorem maximize_angle
  (a b : ℝ)
  (h : a > b)
  (h₁ : b > 0)
  : ∃ (C : Point), C = ⟨Real.sqrt (a * b), 0⟩ :=
sorry

end maximize_angle_l666_66607


namespace problem_statement_l666_66661

-- Definitions based on the conditions
def P : Prop := ∀ x : ℝ, (0 < x ∧ x < 1) ↔ (x / (x - 1) < 0)
def Q : Prop := ∀ (A B : ℝ), (A > B) → (A > 90 ∨ B < 90)

-- The proof problem statement
theorem problem_statement : P ∧ ¬Q := 
by
  sorry

end problem_statement_l666_66661


namespace find_other_number_l666_66688

theorem find_other_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 14) (lcm_ab : Nat.lcm a b = 396) (h : a = 36) : b = 154 :=
by
  sorry

end find_other_number_l666_66688


namespace parallel_perpendicular_trans_l666_66613

variables {Plane Line : Type}

-- Definitions in terms of lines and planes
variables (α β γ : Plane) (a b : Line)

-- Definitions of parallel and perpendicular
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- The mathematical statement to prove
theorem parallel_perpendicular_trans :
  (parallel a b) → (perpendicular b α) → (perpendicular a α) :=
by sorry

end parallel_perpendicular_trans_l666_66613


namespace find_monic_polynomial_l666_66653

-- Define the original polynomial
def polynomial_1 (x : ℝ) := x^3 - 4 * x^2 + 9

-- Define the monic polynomial we are seeking
def polynomial_2 (x : ℝ) := x^3 - 12 * x^2 + 243

theorem find_monic_polynomial :
  ∀ (r1 r2 r3 : ℝ), 
    polynomial_1 r1 = 0 → 
    polynomial_1 r2 = 0 → 
    polynomial_1 r3 = 0 → 
    polynomial_2 (3 * r1) = 0 ∧ polynomial_2 (3 * r2) = 0 ∧ polynomial_2 (3 * r3) = 0 :=
by
  intros r1 r2 r3 h1 h2 h3
  sorry

end find_monic_polynomial_l666_66653


namespace mean_of_other_two_l666_66625

theorem mean_of_other_two (a b c d e f : ℕ) (h : a = 1867 ∧ b = 1993 ∧ c = 2019 ∧ d = 2025 ∧ e = 2109 ∧ f = 2121):
  ((a + b + c + d + e + f) - (4 * 2008)) / 2 = 2051 := by
  sorry

end mean_of_other_two_l666_66625


namespace even_perfect_square_factors_l666_66692

theorem even_perfect_square_factors : 
  (∃ count : ℕ, count = 3 * 2 * 3 ∧ 
    (∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ b ≤ 3 ∧ c % 2 = 0 ∧ c ≤ 4) → 
      (2^a * 7^b * 3^c ∣ 2^6 * 7^3 * 3^4))) :=
sorry

end even_perfect_square_factors_l666_66692


namespace sum_of_remainders_l666_66679

theorem sum_of_remainders
  (a b c : ℕ)
  (h₁ : a % 36 = 15)
  (h₂ : b % 36 = 22)
  (h₃ : c % 36 = 9) :
  (a + b + c) % 36 = 10 :=
by
  sorry

end sum_of_remainders_l666_66679


namespace lcm_9_14_l666_66609

/-- Given the definition of the least common multiple (LCM) and the prime factorizations,
    prove that the LCM of 9 and 14 is 126. -/
theorem lcm_9_14 : Int.lcm 9 14 = 126 := by
  sorry

end lcm_9_14_l666_66609


namespace no_positive_n_l666_66649

theorem no_positive_n :
  ¬ ∃ (n : ℕ) (n_pos : n > 0) (a b : ℕ) (a_sd : a < 10) (b_sd : b < 10), 
    (1234 - n) * b = (6789 - n) * a :=
by 
  sorry

end no_positive_n_l666_66649


namespace eugene_boxes_needed_l666_66670

-- Define the number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards not used
def unused_cards : ℕ := 16

-- Define the number of toothpicks per card
def toothpicks_per_card : ℕ := 75

-- Define the number of toothpicks in a box
def toothpicks_per_box : ℕ := 450

-- Calculate the number of cards used
def cards_used : ℕ := total_cards - unused_cards

-- Calculate the number of cards a single box can support
def cards_per_box : ℕ := toothpicks_per_box / toothpicks_per_card

-- Theorem statement
theorem eugene_boxes_needed : cards_used / cards_per_box = 6 := by
  -- The proof steps are not provided as per the instructions. 
  sorry

end eugene_boxes_needed_l666_66670


namespace average_speed_problem_l666_66676

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_problem :
  average_speed 30 40 37.5 7 (30 / 35) (40 / 55) 0.5 (10 / 60) = 51 :=
by
  -- skip the proof
  sorry

end average_speed_problem_l666_66676


namespace petya_prevents_vasya_l666_66618

-- Define the nature of fractions and the players' turns
def is_natural_sum (fractions : List ℚ) : Prop :=
  (fractions.sum = ⌊fractions.sum⌋)

def petya_vasya_game_prevent (fractions : List ℚ) : Prop :=
  ∀ k : ℕ, ∀ additional_fractions : List ℚ, 
  (additional_fractions.length = k) →
  ¬ is_natural_sum (fractions ++ additional_fractions)

theorem petya_prevents_vasya : ∀ fractions : List ℚ, petya_vasya_game_prevent fractions :=
by
  sorry

end petya_prevents_vasya_l666_66618


namespace driver_net_pay_rate_l666_66683

theorem driver_net_pay_rate
    (hours : ℕ) (distance_per_hour : ℕ) (distance_per_gallon : ℕ) 
    (pay_per_mile : ℝ) (gas_cost_per_gallon : ℝ) :
    hours = 3 →
    distance_per_hour = 50 →
    distance_per_gallon = 25 →
    pay_per_mile = 0.75 →
    gas_cost_per_gallon = 2.50 →
    (pay_per_mile * (distance_per_hour * hours) - gas_cost_per_gallon * ((distance_per_hour * hours) / distance_per_gallon)) / hours = 32.5 :=
by
  intros h_hours h_dph h_dpg h_ppm h_gcpg
  sorry

end driver_net_pay_rate_l666_66683


namespace chicken_rabbit_problem_l666_66602

theorem chicken_rabbit_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chicken_rabbit_problem_l666_66602


namespace scientific_notation_of_845_billion_l666_66617

/-- Express 845 billion yuan in scientific notation. -/
theorem scientific_notation_of_845_billion :
  (845 * (10^9 : ℝ)) / (10^9 : ℝ) = 8.45 * 10^3 :=
by
  sorry

end scientific_notation_of_845_billion_l666_66617


namespace DE_eq_DF_l666_66682

variable {Point : Type}
variable {E A B C D F : Point}
variable (square : Π (A B C D : Point), Prop ) 
variable (is_parallel : Π (A B : Point), Prop) 
variable (E_outside_square : Prop)
variable (BE_eq_BD : Prop)
variable (BE_intersects_AD_at_F : Prop)

theorem DE_eq_DF
  (H1 : square A B C D)
  (H2 : is_parallel AE BD)
  (H3 : BE_eq_BD)
  (H4 : BE_intersects_AD_at_F) :
  DE = DF := 
sorry

end DE_eq_DF_l666_66682


namespace calculate_value_l666_66650

theorem calculate_value (a b c x : ℕ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) (h_x : x = 3) :
  x^(a * (b + c)) - (x^a + x^b + x^c) = 204 := by
  sorry

end calculate_value_l666_66650


namespace find_decreased_value_l666_66675

theorem find_decreased_value (x v : ℝ) (hx : x = 7)
  (h : x - v = 21 * (1 / x)) : v = 4 :=
by
  sorry

end find_decreased_value_l666_66675


namespace sum_of_solutions_l666_66623

theorem sum_of_solutions (a : ℝ) (h : 0 < a ∧ a < 1) :
  let x1 := 3 + a
  let x2 := 3 - a
  let x3 := 1 + a
  let x4 := 1 - a
  x1 + x2 + x3 + x4 = 8 :=
by
  intros
  sorry

end sum_of_solutions_l666_66623


namespace tiffany_daily_miles_l666_66681

-- Definitions for running schedule
def billy_sunday_miles := 1
def billy_monday_miles := 1
def billy_tuesday_miles := 1
def billy_wednesday_miles := 1
def billy_thursday_miles := 1
def billy_friday_miles := 1
def billy_saturday_miles := 1

def tiffany_wednesday_miles := 1 / 3
def tiffany_thursday_miles := 1 / 3
def tiffany_friday_miles := 1 / 3

-- Total miles is the sum of miles for the week
def billy_total_miles := billy_sunday_miles + billy_monday_miles + billy_tuesday_miles +
                         billy_wednesday_miles + billy_thursday_miles + billy_friday_miles +
                         billy_saturday_miles

def tiffany_total_miles (T : ℝ) := T * 3 + 
                                   tiffany_wednesday_miles + tiffany_thursday_miles + tiffany_friday_miles

-- Proof problem: show that Tiffany runs 2 miles each day on Sunday, Monday, and Tuesday
theorem tiffany_daily_miles : ∃ T : ℝ, (tiffany_total_miles T = billy_total_miles) ∧ T = 2 :=
by
  sorry

end tiffany_daily_miles_l666_66681


namespace cheryl_bill_cost_correct_l666_66665

def cheryl_electricity_bill_cost : Prop :=
  ∃ (E : ℝ), 
    (E + 400) + 0.20 * (E + 400) = 1440 ∧ 
    E = 800

theorem cheryl_bill_cost_correct : cheryl_electricity_bill_cost :=
by
  sorry

end cheryl_bill_cost_correct_l666_66665


namespace solve_for_a_l666_66604

-- Define the lines
def l1 (x y : ℝ) := x + y - 2 = 0
def l2 (x y a : ℝ) := 2 * x + a * y - 3 = 0

-- Define orthogonality condition
def perpendicular (m₁ m₂ : ℝ) := m₁ * m₂ = -1

-- The theorem to prove
theorem solve_for_a (a : ℝ) :
  (∀ x y : ℝ, l1 x y → ∀ x y : ℝ, l2 x y a → perpendicular (-1) (-2 / a)) → a = 2 := 
sorry

end solve_for_a_l666_66604


namespace least_adjacent_probability_l666_66629

theorem least_adjacent_probability (n : ℕ) 
    (h₀ : 0 < n)
    (h₁ : (∀ m : ℕ, 0 < m ∧ m < n → (4 * m^2 - 4 * m + 8) / (m^2 * (m^2 - 1)) ≥ 1 / 2015)) : 
    (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1)) < 1 / 2015 := by
  sorry

end least_adjacent_probability_l666_66629


namespace brendan_taxes_l666_66666

def total_hours (num_8hr_shifts : ℕ) (num_12hr_shifts : ℕ) : ℕ :=
  (num_8hr_shifts * 8) + (num_12hr_shifts * 12)

def total_wage (hourly_wage : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_wage * hours_worked

def total_tips (hourly_tips : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_tips * hours_worked

def reported_tips (total_tips : ℕ) (report_fraction : ℕ) : ℕ :=
  total_tips / report_fraction

def reported_income (wage : ℕ) (tips : ℕ) : ℕ :=
  wage + tips

def taxes (income : ℕ) (tax_rate : ℚ) : ℚ :=
  income * tax_rate

theorem brendan_taxes (num_8hr_shifts num_12hr_shifts : ℕ)
    (hourly_wage hourly_tips report_fraction : ℕ) (tax_rate : ℚ) :
    (hourly_wage = 6) →
    (hourly_tips = 12) →
    (report_fraction = 3) →
    (tax_rate = 0.2) →
    (num_8hr_shifts = 2) →
    (num_12hr_shifts = 1) →
    taxes (reported_income (total_wage hourly_wage (total_hours num_8hr_shifts num_12hr_shifts))
            (reported_tips (total_tips hourly_tips (total_hours num_8hr_shifts num_12hr_shifts))
            report_fraction))
          tax_rate = 56 :=
by
  intros
  sorry

end brendan_taxes_l666_66666


namespace angle_sum_straight_line_l666_66671

theorem angle_sum_straight_line (x : ℝ) (h : 4 * x + x = 180) : x = 36 :=
sorry

end angle_sum_straight_line_l666_66671


namespace min_value_x_y_l666_66694

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 / (x + 1) + 1 / (y + 1) = 1) :
  x + y ≥ 14 :=
sorry

end min_value_x_y_l666_66694


namespace distinct_real_roots_l666_66645

open Real

theorem distinct_real_roots (n : ℕ) (hn : n > 0) (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (2 * n - 1 < x1 ∧ x1 ≤ 2 * n + 1) ∧ 
  (2 * n - 1 < x2 ∧ x2 ≤ 2 * n + 1) ∧ |x1 - 2 * n| = k ∧ |x2 - 2 * n| = k) ↔ (0 < k ∧ k ≤ 1) :=
by
  sorry

end distinct_real_roots_l666_66645


namespace find_angle_C_range_of_a_plus_b_l666_66647

variables {A B C a b c : ℝ}

-- Define the conditions
def conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a + c) * (Real.sin A - Real.sin C) = Real.sin B * (a - b)

-- Proof problem 1: show angle C is π/3
theorem find_angle_C (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b c A B C) : 
  C = π / 3 :=
sorry

-- Proof problem 2: if c = 2, then show the range of a + b
theorem range_of_a_plus_b (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b 2 A B C) :
  2 < a + b ∧ a + b ≤ 4 :=
sorry

end find_angle_C_range_of_a_plus_b_l666_66647


namespace smallest_palindrome_divisible_by_6_l666_66631

def is_palindrome (x : Nat) : Prop :=
  let d1 := x / 1000
  let d2 := (x / 100) % 10
  let d3 := (x / 10) % 10
  let d4 := x % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by (x n : Nat) : Prop :=
  x % n = 0

theorem smallest_palindrome_divisible_by_6 : ∃ n : Nat, is_palindrome n ∧ is_divisible_by n 6 ∧ 1000 ≤ n ∧ n < 10000 ∧ ∀ m : Nat, (is_palindrome m ∧ is_divisible_by m 6 ∧ 1000 ≤ m ∧ m < 10000) → n ≤ m := 
  by
    exists 2112
    sorry

end smallest_palindrome_divisible_by_6_l666_66631


namespace counties_rained_on_monday_l666_66632

theorem counties_rained_on_monday : 
  ∀ (M T R_no_both R_both : ℝ),
    T = 0.55 → 
    R_no_both = 0.35 →
    R_both = 0.60 →
    (M + T - R_both = 1 - R_no_both) →
    M = 0.70 :=
by
  intros M T R_no_both R_both hT hR_no_both hR_both hInclusionExclusion
  sorry

end counties_rained_on_monday_l666_66632


namespace people_per_column_in_second_scenario_l666_66693

def total_people (num_people_per_column_1 : ℕ) (num_columns_1 : ℕ) : ℕ :=
  num_people_per_column_1 * num_columns_1

def people_per_column_second_scenario (P: ℕ) (num_columns_2 : ℕ) : ℕ :=
  P / num_columns_2

theorem people_per_column_in_second_scenario
  (num_people_per_column_1 : ℕ)
  (num_columns_1 : ℕ)
  (num_columns_2 : ℕ)
  (P : ℕ)
  (h1 : total_people num_people_per_column_1 num_columns_1 = P) :
  people_per_column_second_scenario P num_columns_2 = 48 :=
by
  -- the proof would go here
  sorry

end people_per_column_in_second_scenario_l666_66693


namespace sum_of_x_values_l666_66627

theorem sum_of_x_values (y x : ℝ) (h1 : y = 6) (h2 : x^2 + y^2 = 144) : x + (-x) = 0 :=
by
  sorry

end sum_of_x_values_l666_66627


namespace greatest_integer_lesser_200_gcd_45_eq_9_l666_66669

theorem greatest_integer_lesser_200_gcd_45_eq_9 :
  ∃ n : ℕ, n < 200 ∧ Int.gcd n 45 = 9 ∧ ∀ m : ℕ, (m < 200 ∧ Int.gcd m 45 = 9) → m ≤ n :=
by
  sorry

end greatest_integer_lesser_200_gcd_45_eq_9_l666_66669


namespace ella_emma_hotdogs_l666_66680

-- Definitions based on the problem conditions
def hotdogs_each_sister_wants (E : ℕ) :=
  let luke := 2 * E
  let hunter := 3 * E
  E + E + luke + hunter = 14

-- Statement we need to prove
theorem ella_emma_hotdogs (E : ℕ) (h : hotdogs_each_sister_wants E) : E = 2 :=
by
  sorry

end ella_emma_hotdogs_l666_66680


namespace option_c_incorrect_l666_66663

theorem option_c_incorrect (a : ℝ) : a + a^2 ≠ a^3 :=
sorry

end option_c_incorrect_l666_66663


namespace factorize_first_poly_factorize_second_poly_l666_66610

variable (x m n : ℝ)

-- Proof statement for the first polynomial
theorem factorize_first_poly : x^2 + 14*x + 49 = (x + 7)^2 := 
by sorry

-- Proof statement for the second polynomial
theorem factorize_second_poly : (m - 1) + n^2 * (1 - m) = (m - 1) * (1 - n) * (1 + n) := 
by sorry

end factorize_first_poly_factorize_second_poly_l666_66610


namespace sum_adjacent_angles_pentagon_l666_66606

theorem sum_adjacent_angles_pentagon (n : ℕ) (θ : ℕ) (hn : n = 5) (hθ : θ = 40) :
  let exterior_angle := 360 / n
  let new_adjacent_angle := 180 - (exterior_angle + θ)
  let sum_adjacent_angles := n * new_adjacent_angle
  sum_adjacent_angles = 340 := by
  sorry

end sum_adjacent_angles_pentagon_l666_66606


namespace max_min_values_monotonocity_l666_66646

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 - (1 / 2) * x ^ 2

theorem max_min_values (a : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (ha : a = 1) : 
  f a 0 = 0 ∧ f a 1 = 1 / 2 ∧ f a (1 / 3) = -1 / 54 :=
sorry

theorem monotonocity (a : ℝ) (hx : 0 < x ∧ x < (1 / (6 * a))) (ha : 0 < a) : 
  (3 * a * x ^ 2 - x) < 0 → (f a x) < (f a 0) :=
sorry

end max_min_values_monotonocity_l666_66646


namespace correct_operation_l666_66612

variables (a b : ℝ)

theorem correct_operation : (3 * a + b) * (3 * a - b) = 9 * a^2 - b^2 :=
by sorry

end correct_operation_l666_66612


namespace value_of_a_l666_66619

theorem value_of_a (a : ℕ) : (∃ (x1 x2 x3 : ℤ),
  abs (abs (x1 - 3) - 1) = a ∧
  abs (abs (x2 - 3) - 1) = a ∧
  abs (abs (x3 - 3) - 1) = a ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)
  → a = 1 :=
by
  sorry

end value_of_a_l666_66619


namespace prob_green_is_correct_l666_66667

-- Define the probability of picking any container
def prob_pick_container : ℚ := 1 / 4

-- Define the probability of drawing a green ball from each container
def prob_green_A : ℚ := 6 / 10
def prob_green_B : ℚ := 3 / 10
def prob_green_C : ℚ := 3 / 10
def prob_green_D : ℚ := 5 / 10

-- Define the individual probabilities for a green ball, accounting for container selection
def prob_green_given_A : ℚ := prob_pick_container * prob_green_A
def prob_green_given_B : ℚ := prob_pick_container * prob_green_B
def prob_green_given_C : ℚ := prob_pick_container * prob_green_C
def prob_green_given_D : ℚ := prob_pick_container * prob_green_D

-- Calculate the total probability of selecting a green ball
def prob_green_total : ℚ := prob_green_given_A + prob_green_given_B + prob_green_given_C + prob_green_given_D

-- Theorem statement: The probability of selecting a green ball is 17/40
theorem prob_green_is_correct : prob_green_total = 17 / 40 :=
by
  -- Proof will be provided here.
  sorry

end prob_green_is_correct_l666_66667


namespace sin_480_eq_sqrt3_div_2_l666_66630

theorem sin_480_eq_sqrt3_div_2 : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_eq_sqrt3_div_2_l666_66630


namespace solve_for_x_l666_66636

theorem solve_for_x : ∃ x : ℤ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end solve_for_x_l666_66636


namespace set_intersection_l666_66620

def U : Set ℝ := Set.univ
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x ≥ 2}
def C_U_B : Set ℝ := {x | x < 2}

theorem set_intersection :
  A ∩ C_U_B = {-1, 0, 1} :=
sorry

end set_intersection_l666_66620


namespace compute_paths_in_grid_l666_66659

def grid : List (List Char) := [
  [' ', ' ', ' ', ' ', ' ', ' ', 'C', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', 'C', 'O', 'C', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', 'C', 'O', 'M', 'O', 'C', ' ', ' ', ' '],
  [' ', ' ', ' ', 'C', 'O', 'M', 'P', 'M', 'O', 'C', ' ', ' '],
  [' ', ' ', 'C', 'O', 'M', 'P', 'U', 'P', 'M', 'O', 'C', ' '],
  [' ', 'C', 'O', 'M', 'P', 'U', 'T', 'U', 'P', 'M', 'O', 'C'],
  ['C', 'O', 'M', 'P', 'U', 'T', 'E', 'T', 'U', 'P', 'M', 'O', 'C']
]

def is_valid_path (path : List (Nat × Nat)) : Bool :=
  -- This function checks if a given path is valid according to the problem's grid and rules.
  sorry

def count_paths_from_C_to_E (grid: List (List Char)) : Nat :=
  -- This function would count the number of valid paths from a 'C' in the leftmost column to an 'E' in the rightmost column.
  sorry

theorem compute_paths_in_grid : count_paths_from_C_to_E grid = 64 :=
by
  sorry

end compute_paths_in_grid_l666_66659


namespace range_of_2a_plus_3b_l666_66626

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b ∧ a + b ≤ 1) (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l666_66626


namespace chapters_page_difference_l666_66600

def chapter1_pages : ℕ := 37
def chapter2_pages : ℕ := 80

theorem chapters_page_difference : chapter2_pages - chapter1_pages = 43 := by
  -- Proof goes here
  sorry

end chapters_page_difference_l666_66600


namespace smallest_x_for_1980_power4_l666_66696

theorem smallest_x_for_1980_power4 (M : ℤ) (x : ℕ) (hx : x > 0) :
  (1980 * (x : ℤ)) = M^4 → x = 6006250 :=
by
  -- The proof goes here
  sorry

end smallest_x_for_1980_power4_l666_66696


namespace percentage_passed_l666_66657

-- Definitions corresponding to the conditions
def F_H : ℝ := 25
def F_E : ℝ := 35
def F_B : ℝ := 40

-- Main theorem stating the question's proof.
theorem percentage_passed :
  (100 - (F_H + F_E - F_B)) = 80 :=
by
  -- we can transcribe the remaining process here if needed.
  sorry

end percentage_passed_l666_66657


namespace smallest_pieces_left_l666_66697

theorem smallest_pieces_left (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) : 
    ∃ k, (k = 2 ∧ (m * n) % 3 = 0) ∨ (k = 1 ∧ (m * n) % 3 ≠ 0) :=
by
    sorry

end smallest_pieces_left_l666_66697


namespace bus_driver_total_compensation_l666_66672

theorem bus_driver_total_compensation :
  let regular_rate := 16
  let regular_hours := 40
  let overtime_hours := 60 - regular_hours
  let overtime_rate := regular_rate + 0.75 * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1200 := by
  sorry

end bus_driver_total_compensation_l666_66672


namespace floor_div_eq_floor_div_floor_l666_66637

theorem floor_div_eq_floor_div_floor {α : ℝ} {d : ℕ} (h₁ : 0 < α) : 
  (⌊α / d⌋ = ⌊⌊α⌋ / d⌋) := 
sorry

end floor_div_eq_floor_div_floor_l666_66637


namespace class_tree_total_l666_66660

theorem class_tree_total
  (trees_A : ℕ)
  (trees_B : ℕ)
  (hA : trees_A = 8)
  (hB : trees_B = 7)
  : trees_A + trees_B = 15 := 
by
  sorry

end class_tree_total_l666_66660


namespace boys_contributions_l666_66605

theorem boys_contributions (x y z : ℝ) (h1 : z = x + 6.4) (h2 : (1 / 2) * x = (1 / 3) * y) (h3 : (1 / 2) * x = (1 / 4) * z) :
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  -- This is where the proof would go
  sorry

end boys_contributions_l666_66605


namespace unit_digit_of_fourth_number_l666_66652

theorem unit_digit_of_fourth_number
  (n1 n2 n3 n4 : ℕ)
  (h1 : n1 % 10 = 4)
  (h2 : n2 % 10 = 8)
  (h3 : n3 % 10 = 3)
  (h4 : (n1 * n2 * n3 * n4) % 10 = 8) : 
  n4 % 10 = 3 :=
sorry

end unit_digit_of_fourth_number_l666_66652


namespace problem1_problem2_l666_66622

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -2 * x - 1
  else if 0 < x ∧ x ≤ 1 then -2 * x + 1
  else 0 -- considering the function is not defined outside the given range

-- Statement to prove that f(f(-1)) = -1
theorem problem1 : f (f (-1)) = -1 :=
by
  sorry

-- Statements to prove the solution set for |f(x)| < 1/2
theorem problem2 : { x : ℝ | |f x| < 1 / 2 } = { x : ℝ | -3/4 < x ∧ x < -1/4 } ∪ { x : ℝ | 1/4 < x ∧ x < 3/4 } :=
by
  sorry

end problem1_problem2_l666_66622


namespace john_increased_bench_press_factor_l666_66642

theorem john_increased_bench_press_factor (initial current : ℝ) (decrease_percent : ℝ) 
  (h_initial : initial = 500) 
  (h_current : current = 300) 
  (h_decrease : decrease_percent = 0.80) : 
  current / (initial * (1 - decrease_percent)) = 3 := 
by
  -- We'll provide the proof here later
  sorry

end john_increased_bench_press_factor_l666_66642


namespace proof_problem_l666_66624

variables (p q : Prop)

theorem proof_problem (hpq : p ∨ q) (hnp : ¬p) : q :=
by
  sorry

end proof_problem_l666_66624


namespace hyperbola_asymptote_eccentricity_l666_66655

-- Problem statement: We need to prove that the eccentricity of hyperbola 
-- given the specific asymptote is sqrt(5).

noncomputable def calc_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptote_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : b = 2 * a) :
  calc_eccentricity a b = Real.sqrt 5 := 
by
  -- Insert the proof step here
  sorry

end hyperbola_asymptote_eccentricity_l666_66655


namespace minimal_abs_diff_l666_66689

theorem minimal_abs_diff (a b : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b - 8 * a + 7 * b = 569) : abs (a - b) = 23 :=
sorry

end minimal_abs_diff_l666_66689


namespace max_sum_two_digit_primes_l666_66638

theorem max_sum_two_digit_primes : (89 + 97) = 186 := 
by
  sorry

end max_sum_two_digit_primes_l666_66638


namespace negation_of_proposition_l666_66615

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l666_66615


namespace irene_total_income_l666_66662

noncomputable def irene_income (weekly_hours : ℕ) (base_pay : ℕ) (overtime_pay : ℕ) (hours_worked : ℕ) : ℕ :=
  base_pay + (if hours_worked > weekly_hours then (hours_worked - weekly_hours) * overtime_pay else 0)

theorem irene_total_income :
  irene_income 40 500 20 50 = 700 :=
by
  sorry

end irene_total_income_l666_66662


namespace simplify_P_eq_l666_66633

noncomputable def P (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y) - (x * y - y^2) / (x * y - x^2)

theorem simplify_P_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy: x ≠ y) : P x y = x / y := 
by
  -- Insert proof here
  sorry

end simplify_P_eq_l666_66633


namespace prime_condition_l666_66690

theorem prime_condition (p : ℕ) (h_prime: Nat.Prime p) :
  (∃ m n : ℤ, p = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) ↔ p = 2 ∨ p = 5 :=
by
  sorry

end prime_condition_l666_66690


namespace difference_twice_cecil_and_catherine_l666_66614

theorem difference_twice_cecil_and_catherine
  (Cecil Catherine Carmela : ℕ)
  (h1 : Cecil = 600)
  (h2 : Carmela = 2 * 600 + 50)
  (h3 : 600 + (2 * 600 - Catherine) + Carmela = 2800) :
  2 * 600 - Catherine = 250 := by
  sorry

end difference_twice_cecil_and_catherine_l666_66614


namespace find_x0_l666_66621

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x0 : ℝ) (h : f' x0 = 2) : x0 = Real.exp 1 :=
by {
  sorry
}

end find_x0_l666_66621


namespace find_integer_l666_66684

theorem find_integer (x : ℕ) (h1 : (4 * x)^2 + 2 * x = 3528) : x = 14 := by
  sorry

end find_integer_l666_66684


namespace range_of_y_when_x_3_l666_66603

variable (a c : ℝ)

theorem range_of_y_when_x_3 (h1 : -4 ≤ a + c ∧ a + c ≤ -1) (h2 : -1 ≤ 4 * a + c ∧ 4 * a + c ≤ 5) :
  -1 ≤ 9 * a + c ∧ 9 * a + c ≤ 20 :=
sorry

end range_of_y_when_x_3_l666_66603


namespace sum_of_squares_eq_l666_66643

theorem sum_of_squares_eq :
  ∀ (M G D : ℝ), 
  (M = G / 3) → 
  (G = 450) → 
  (D = 2 * G) → 
  (M^2 + G^2 + D^2 = 1035000) :=
by
  intros M G D hM hG hD
  sorry

end sum_of_squares_eq_l666_66643


namespace number_of_hexagonal_faces_geq_2_l666_66656

noncomputable def polyhedron_condition (P H : ℕ) : Prop :=
  ∃ V E : ℕ, 
    V - E + (P + H) = 2 ∧ 
    3 * V = 2 * E ∧ 
    E = (5 * P + 6 * H) / 2 ∧
    P > 0 ∧ H > 0

theorem number_of_hexagonal_faces_geq_2 (P H : ℕ) (h : polyhedron_condition P H) : H ≥ 2 :=
sorry

end number_of_hexagonal_faces_geq_2_l666_66656


namespace total_balls_l666_66685

theorem total_balls (jungkook_balls : ℕ) (yoongi_balls : ℕ) (h1 : jungkook_balls = 3) (h2 : yoongi_balls = 4) : 
  jungkook_balls + yoongi_balls = 7 :=
by
  -- This is a placeholder for the proof
  sorry

end total_balls_l666_66685


namespace fraction_identity_l666_66648

theorem fraction_identity (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x / y := 
by {
  sorry
}

end fraction_identity_l666_66648


namespace max_cos_a_l666_66673

theorem max_cos_a (a b c : ℝ) 
  (h1 : Real.sin a = Real.cos b) 
  (h2 : Real.sin b = Real.cos c) 
  (h3 : Real.sin c = Real.cos a) : 
  Real.cos a = Real.sqrt 2 / 2 := by
sorry

end max_cos_a_l666_66673


namespace num_mittens_per_box_eq_six_l666_66674

theorem num_mittens_per_box_eq_six 
    (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ)
    (h1 : num_boxes = 4) (h2 : scarves_per_box = 2) (h3 : total_clothing = 32) :
    (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 :=
by
  sorry

end num_mittens_per_box_eq_six_l666_66674


namespace circle_properties_l666_66601

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x = 0

theorem circle_properties (x y : ℝ) :
  circle_center x y ↔ ((x - 2)^2 + y^2 = 2^2) ∧ ((2, 0) = (2, 0)) :=
by
  sorry

end circle_properties_l666_66601


namespace license_plates_count_l666_66616

/--
Define the conditions and constants.
-/
def num_letters := 26
def num_first_digit := 5  -- Odd digits
def num_second_digit := 5 -- Even digits

theorem license_plates_count : num_letters ^ 3 * num_first_digit * num_second_digit = 439400 := by
  sorry

end license_plates_count_l666_66616


namespace tiles_with_no_gaps_l666_66634

-- Define the condition that the tiling consists of regular octagons
def regular_octagon_internal_angle := 135

-- Define the other regular polygons
def regular_triangle_internal_angle := 60
def regular_square_internal_angle := 90
def regular_pentagon_internal_angle := 108
def regular_hexagon_internal_angle := 120

-- The proposition to be proved: A flat surface without gaps
-- can be achieved using regular squares and regular octagons.
theorem tiles_with_no_gaps :
  ∃ (m n : ℕ), regular_octagon_internal_angle * m + regular_square_internal_angle * n = 360 :=
sorry

end tiles_with_no_gaps_l666_66634


namespace number_of_77s_l666_66640

theorem number_of_77s (a b : ℕ) :
  (∃ a : ℕ, 1015 = a + 3 * 77 ∧ a + 21 = 10)
  ∧ (∃ b : ℕ, 2023 = b + 6 * 77 + 2 * 777 ∧ b = 7)
  → 6 = 6 := 
by
    sorry

end number_of_77s_l666_66640


namespace min_value_of_sum_l666_66611

theorem min_value_of_sum (a b : ℝ) (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 2 = 6) :
  a + b ≥ 16 :=
sorry

end min_value_of_sum_l666_66611


namespace find_S6_l666_66635

-- sum of the first n terms of an arithmetic sequence
variable (S : ℕ → ℕ)

-- Given conditions
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- Theorem statement
theorem find_S6 : S 6 = 36 := sorry

end find_S6_l666_66635


namespace sum_of_coordinates_of_X_l666_66608

theorem sum_of_coordinates_of_X 
  (X Y Z : ℝ × ℝ)
  (h1 : dist X Z / dist X Y = 1 / 2)
  (h2 : dist Z Y / dist X Y = 1 / 2)
  (hY : Y = (1, 7))
  (hZ : Z = (-1, -7)) :
  (X.1 + X.2) = -24 :=
sorry

end sum_of_coordinates_of_X_l666_66608


namespace students_not_in_any_activity_l666_66695

def total_students : ℕ := 1500
def students_chorus : ℕ := 420
def students_band : ℕ := 780
def students_chorus_and_band : ℕ := 150
def students_drama : ℕ := 300
def students_drama_and_other : ℕ := 50

theorem students_not_in_any_activity :
  total_students - ((students_chorus + students_band - students_chorus_and_band) + (students_drama - students_drama_and_other)) = 200 :=
by
  sorry

end students_not_in_any_activity_l666_66695


namespace cara_neighbors_l666_66641

def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem cara_neighbors : number_of_pairs 7 = 21 :=
by
  sorry

end cara_neighbors_l666_66641


namespace price_increase_percentage_l666_66658

theorem price_increase_percentage (c : ℝ) (r : ℝ) (p : ℝ) 
  (h1 : r = 1.4 * c) 
  (h2 : p = 1.15 * r) : 
  (p - c) / c * 100 = 61 := 
sorry

end price_increase_percentage_l666_66658


namespace intersection_A_B_l666_66644

def setA : Set ℝ := {x | x^2 - 1 > 0}
def setB : Set ℝ := {x | Real.log x / Real.log 2 < 1}

theorem intersection_A_B :
  {x | x ∈ setA ∧ x ∈ setB} = {x | 1 < x ∧ x < 2} :=
sorry

end intersection_A_B_l666_66644


namespace calculate_expression_l666_66678

theorem calculate_expression : 
  (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := 
by sorry

end calculate_expression_l666_66678


namespace outer_boundary_diameter_l666_66677

theorem outer_boundary_diameter (d_pond : ℝ) (w_picnic : ℝ) (w_track : ℝ)
  (h_pond_diam : d_pond = 16) (h_picnic_width : w_picnic = 10) (h_track_width : w_track = 4) :
  2 * (d_pond / 2 + w_picnic + w_track) = 44 :=
by
  -- We avoid the entire proof, we only assert the statement in Lean
  sorry

end outer_boundary_diameter_l666_66677


namespace hyperbola_foci_coordinates_l666_66668

theorem hyperbola_foci_coordinates :
  ∀ (x y : ℝ), x^2 - (y^2 / 3) = 1 → (∃ c : ℝ, c = 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_coordinates_l666_66668


namespace total_bouquets_sold_l666_66664

-- defining the sale conditions
def monday_bouquets := 12
def tuesday_bouquets := 3 * monday_bouquets
def wednesday_bouquets := tuesday_bouquets / 3

-- defining the total sale
def total_bouquets := monday_bouquets + tuesday_bouquets + wednesday_bouquets

-- stating the theorem
theorem total_bouquets_sold : total_bouquets = 60 := by
  -- the proof would go here
  sorry

end total_bouquets_sold_l666_66664


namespace total_sticks_needed_l666_66698

theorem total_sticks_needed (simon_sticks gerry_sticks micky_sticks darryl_sticks : ℕ):
  simon_sticks = 36 →
  gerry_sticks = (2 * simon_sticks) / 3 →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  darryl_sticks = simon_sticks + gerry_sticks + micky_sticks + 1 →
  simon_sticks + gerry_sticks + micky_sticks + darryl_sticks = 259 :=
by
  intros h_simon h_gerry h_micky h_darryl
  rw [h_simon, h_gerry, h_micky, h_darryl]
  norm_num
  sorry

end total_sticks_needed_l666_66698


namespace triangle_AX_length_l666_66687

noncomputable def length_AX (AB AC BC : ℝ) (h1 : AB = 60) (h2 : AC = 34) (h3 : BC = 52) : ℝ :=
  1020 / 43

theorem triangle_AX_length 
  (AB AC BC AX : ℝ)
  (h1 : AB = 60)
  (h2 : AC = 34)
  (h3 : BC = 52)
  (h4 : AX + (AB - AX) = AB)
  (h5 : AX / (AB - AX) = AC / BC) :
  AX = 1020 / 43 := 
sorry

end triangle_AX_length_l666_66687


namespace both_firms_participate_number_of_firms_participate_social_optimality_l666_66691

-- Definitions for general conditions
variable (α V IC : ℝ)
variable (hα : 0 < α ∧ α < 1)

-- Condition for both firms to participate
def condition_to_participate (V : ℝ) (α : ℝ) (IC : ℝ) : Prop :=
  V * α * (1 - 0.5 * α) ≥ IC

-- Part (a): Under what conditions will both firms participate?
theorem both_firms_participate (α V IC : ℝ) (hα : 0 < α ∧ α < 1) :
  condition_to_participate V α IC → (V * α * (1 - 0.5 * α) ≥ IC) :=
by sorry

-- Part (b): Given V=16, α=0.5, and IC=5, determine the number of firms participating
theorem number_of_firms_participate :
  (condition_to_participate 16 0.5 5) :=
by sorry

-- Part (c): To determine if the number of participating firms is socially optimal
def total_profit (α V IC : ℝ) (both : Bool) :=
  if both then 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
  else α * V - IC

theorem social_optimality :
   (total_profit 0.5 16 5 true ≠ max (total_profit 0.5 16 5 true) (total_profit 0.5 16 5 false)) :=
by sorry

end both_firms_participate_number_of_firms_participate_social_optimality_l666_66691


namespace f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l666_66651

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x / (1 + x) else 1 / (1 + x)

theorem f_property (x : ℝ) (hx : 0 < x) : 
  f x = f (1 / x) :=
by
  sorry

theorem f_equals_when_x_lt_1 (x : ℝ) (hx0 : 0 < x) (hx1 : x < 1) : 
  f x = 1 / (1 + x) :=
by
  sorry

theorem f_equals_when_x_gt_1 (x : ℝ) (hx : 1 < x) : 
  f x = x / (1 + x) :=
by
  sorry

end f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l666_66651


namespace intersection_complement_l666_66628

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by {
  -- To ensure the validity of the theorem, the proof goes here
  sorry
}

end intersection_complement_l666_66628


namespace population_after_10_years_l666_66699

def initial_population : ℕ := 100000
def birth_increase_percent : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

theorem population_after_10_years :
  let birth_increase := initial_population * birth_increase_percent
  let total_emigration := emigration_per_year * years
  let total_immigration := immigration_per_year * years
  let net_movement := total_immigration - total_emigration
  let final_population := initial_population + birth_increase + net_movement
  final_population = 165000 :=
by
  sorry

end population_after_10_years_l666_66699
