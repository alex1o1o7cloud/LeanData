import Mathlib

namespace monotonic_increasing_interval_l211_21188

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (x^2 - 4)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 2 < x → (f x < f (x + 1)) :=
by
  intros x h
  sorry

end monotonic_increasing_interval_l211_21188


namespace range_of_a_l211_21120

noncomputable section

def f (a x : ℝ) := a * x^2 + 2 * a * x - Real.log (x + 1)
def g (x : ℝ) := (Real.exp x - x - 1) / (Real.exp x * (x + 1))

theorem range_of_a
  (a : ℝ)
  (h : ∀ x > 0, f a x + Real.exp (-a) > 1 / (x + 1)) : a ∈ Set.Ici (1 / 2) := 
sorry

end range_of_a_l211_21120


namespace irreducible_fraction_l211_21110

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end irreducible_fraction_l211_21110


namespace trip_drop_probability_l211_21147

-- Definitions
def P_Trip : ℝ := 0.4
def P_Drop_not : ℝ := 0.9

-- Main theorem
theorem trip_drop_probability : ∀ (P_Trip P_Drop_not : ℝ), P_Trip = 0.4 → P_Drop_not = 0.9 → 1 - P_Drop_not = 0.1 :=
by
  intros P_Trip P_Drop_not h1 h2
  rw [h2]
  norm_num

end trip_drop_probability_l211_21147


namespace triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l211_21130

theorem triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero
  (A B C : ℝ) (h : A + B + C = 180): 
    (A = 60 ∨ B = 60 ∨ C = 60) ↔ (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 0) := 
by
  sorry

end triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l211_21130


namespace dobarulho_problem_l211_21105

def is_divisible_by (x d : ℕ) : Prop := d ∣ x

def valid_quadruple (A B C D : ℕ) : Prop :=
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (A ≤ 8) ∧ (D > 1) ∧
  is_divisible_by (100 * A + 10 * B + C) D ∧
  is_divisible_by (100 * B + 10 * C + A) D ∧
  is_divisible_by (100 * C + 10 * A + B) D ∧
  is_divisible_by (100 * (A + 1) + 10 * C + B) D ∧
  is_divisible_by (100 * C + 10 * B + (A + 1)) D ∧
  is_divisible_by (100 * B + 10 * (A + 1) + C) D 

theorem dobarulho_problem :
  ∀ (A B C D : ℕ), valid_quadruple A B C D → 
  (A = 3 ∧ B = 7 ∧ C = 0 ∧ D = 37) ∨ 
  (A = 4 ∧ B = 8 ∧ C = 1 ∧ D = 37) ∨
  (A = 5 ∧ B = 9 ∧ C = 2 ∧ D = 37) :=
by sorry

end dobarulho_problem_l211_21105


namespace perpendicular_vectors_l211_21140

-- Definitions based on the conditions
def vector_a (x : ℝ) := (x, 3)
def vector_b := (3, 1)

-- Statement to prove
theorem perpendicular_vectors (x : ℝ) :
  (vector_a x).1 * (vector_b).1 + (vector_a x).2 * (vector_b).2 = 0 → x = -1 := by
  -- Proof goes here
  sorry

end perpendicular_vectors_l211_21140


namespace ellipse_equation_l211_21181

theorem ellipse_equation (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : ∃ (P : ℝ × ℝ), P = (0, -1) ∧ P.2^2 = b^2) 
  (h4 : ∃ (C2 : ℝ → ℝ → Prop), (∀ x y : ℝ, C2 x y ↔ x^2 + y^2 = 4) ∧ 2 * a = 4) :
  (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + y^2 = 1) :=
by
  sorry

end ellipse_equation_l211_21181


namespace calculate_expression_l211_21163

variable (a : ℝ)

theorem calculate_expression : 2 * a - 7 * a + 4 * a = -a := by
  sorry

end calculate_expression_l211_21163


namespace possible_values_of_r_l211_21143

noncomputable def r : ℝ := sorry

def is_four_place_decimal (x : ℝ) : Prop := 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ x = a / 10 + b / 100 + c / 1000 + d / 10000

def is_closest_fraction (x : ℝ) : Prop := 
  abs (x - 3 / 11) < abs (x - 3 / 10) ∧ abs (x - 3 / 11) < abs (x - 1 / 4)

theorem possible_values_of_r :
  (0.2614 <= r ∧ r <= 0.2864) ∧ is_four_place_decimal r ∧ is_closest_fraction r →
  ∃ n : ℕ, n = 251 := 
sorry

end possible_values_of_r_l211_21143


namespace thomas_total_blocks_l211_21170

-- Definitions according to the conditions
def a1 : Nat := 7
def a2 : Nat := a1 + 3
def a3 : Nat := a2 - 6
def a4 : Nat := a3 + 10
def a5 : Nat := 2 * a2

-- The total number of blocks
def total_blocks : Nat := a1 + a2 + a3 + a4 + a5

-- The proof statement
theorem thomas_total_blocks :
  total_blocks = 55 := 
sorry

end thomas_total_blocks_l211_21170


namespace average_words_per_page_l211_21198

theorem average_words_per_page
  (sheets_to_pages : ℕ := 16)
  (total_sheets : ℕ := 12)
  (total_word_count : ℕ := 240000) :
  (total_word_count / (total_sheets * sheets_to_pages)) = 1250 :=
by
  sorry

end average_words_per_page_l211_21198


namespace total_amount_divided_l211_21161

theorem total_amount_divided (P1 : ℝ) (r1 : ℝ) (r2 : ℝ) (interest : ℝ) (T : ℝ) :
  P1 = 1550 →
  r1 = 0.03 →
  r2 = 0.05 →
  interest = 144 →
  (P1 * r1 + (T - P1) * r2 = interest) → T = 3500 :=
by
  intros hP1 hr1 hr2 hint htotal
  sorry

end total_amount_divided_l211_21161


namespace sum_of_circle_center_coordinates_l211_21109

open Real

theorem sum_of_circle_center_coordinates :
  let x1 := 5
  let y1 := 3
  let x2 := -7
  let y2 := 9
  let x_m := (x1 + x2) / 2
  let y_m := (y1 + y2) / 2
  x_m + y_m = 5 := by
  sorry

end sum_of_circle_center_coordinates_l211_21109


namespace prob_red_blue_calc_l211_21114

noncomputable def prob_red_blue : ℚ :=
  let p_yellow := (6 : ℚ) / 13
  let p_red_blue_given_yellow := (7 : ℚ) / 12
  let p_red_blue_given_not_yellow := (7 : ℚ) / 13
  p_red_blue_given_yellow * p_yellow + p_red_blue_given_not_yellow * (1 - p_yellow)

/-- The probability of drawing a red or blue marble from the updated bag contents is 91/169. -/
theorem prob_red_blue_calc : prob_red_blue = 91 / 169 :=
by
  -- This proof is omitted as per instructions.
  sorry

end prob_red_blue_calc_l211_21114


namespace range_of_m_l211_21129

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : 
  (x + y ≥ m) → m ≤ 18 :=
sorry

end range_of_m_l211_21129


namespace relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l211_21108

-- Let a, b, and c be the sides of a right triangle with c as the hypotenuse.
variables (a b c : ℝ) (n : ℕ)

-- Assume the triangle is a right triangle
-- and assume n is a positive integer.
axiom right_triangle : a^2 + b^2 = c^2
axiom positive_integer : n > 0 

-- The relationships we need to prove:
theorem relationship_a_plus_b_greater_c : n = 1 → a + b > c := sorry
theorem relationship_a_squared_plus_b_squared_equals_c_squared : n = 2 → a^2 + b^2 = c^2 := sorry
theorem relationship_a_n_plus_b_n_less_than_c_n : n ≥ 3 → a^n + b^n < c^n := sorry

end relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l211_21108


namespace least_number_added_to_divide_l211_21113

-- Definitions of conditions
def lcm_three_five_seven_eight : ℕ := Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 8
def remainder_28523_lcm := 28523 % lcm_three_five_seven_eight

-- Lean statement to prove the correct answer
theorem least_number_added_to_divide (n : ℕ) :
  n = lcm_three_five_seven_eight - remainder_28523_lcm :=
sorry

end least_number_added_to_divide_l211_21113


namespace percentage_less_than_l211_21136

theorem percentage_less_than (x y : ℝ) (h1 : y = x * 1.8181818181818181) : (∃ P : ℝ, P = 45) :=
by
  sorry

end percentage_less_than_l211_21136


namespace evaluate_expression_l211_21131

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (1/3 : ℚ)
  let z := (-12 : ℚ)
  let w := (5 : ℚ)
  x^2 * y^3 * z + w = (179/36 : ℚ) :=
by
  sorry

end evaluate_expression_l211_21131


namespace investment_at_6_percent_l211_21100

theorem investment_at_6_percent
  (x y : ℝ) 
  (total_investment : x + y = 15000)
  (total_interest : 0.06 * x + 0.075 * y = 1023) :
  x = 6800 :=
sorry

end investment_at_6_percent_l211_21100


namespace christen_potatoes_peeled_l211_21103

-- Define the initial conditions and setup
def initial_potatoes := 50
def homer_rate := 4
def christen_rate := 6
def time_homer_alone := 5
def combined_rate := homer_rate + christen_rate

-- Calculate the number of potatoes peeled by Homer alone in the first 5 minutes
def potatoes_peeled_by_homer_alone := time_homer_alone * homer_rate

-- Calculate the remaining potatoes after Homer peeled alone
def remaining_potatoes := initial_potatoes - potatoes_peeled_by_homer_alone

-- Calculate the time taken for Homer and Christen to peel the remaining potatoes together
def time_to_finish_together := remaining_potatoes / combined_rate

-- Calculate the number of potatoes peeled by Christen during the shared work period
def potatoes_peeled_by_christen := christen_rate * time_to_finish_together

-- The final theorem we need to prove
theorem christen_potatoes_peeled : potatoes_peeled_by_christen = 18 := by
  sorry

end christen_potatoes_peeled_l211_21103


namespace subscription_total_eq_14036_l211_21153

noncomputable def total_subscription (x : ℕ) : ℕ :=
  3 * x + 14000

theorem subscription_total_eq_14036 (c : ℕ) (profit_b : ℕ) (total_profit : ℕ) 
  (h1 : profit_b = 10200)
  (h2 : total_profit = 30000) 
  (h3 : (profit_b : ℝ) / (total_profit : ℝ) = (c + 5000 : ℝ) / (total_subscription c : ℝ)) :
  total_subscription c = 14036 :=
by
  sorry

end subscription_total_eq_14036_l211_21153


namespace h_f_equals_h_g_l211_21184

def f (x : ℝ) := x^2 - x + 1

def g (x : ℝ) := -x^2 + x + 1

def h (x : ℝ) := (x - 1)^2

theorem h_f_equals_h_g : ∀ x : ℝ, h (f x) = h (g x) :=
by
  intro x
  unfold f g h
  sorry

end h_f_equals_h_g_l211_21184


namespace angle_bisectors_geq_nine_times_inradius_l211_21132

theorem angle_bisectors_geq_nine_times_inradius 
  (r : ℝ) (f_a f_b f_c : ℝ) 
  (h_triangle : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ r = (1 / 2) * (a + b + c) * r ∧ 
      f_a ≥ (2 * a * b / (a + b) + 2 * a * c / (a + c)) / 2 ∧ 
      f_b ≥ (2 * b * a / (b + a) + 2 * b * c / (b + c)) / 2 ∧ 
      f_c ≥ (2 * c * a / (c + a) + 2 * c * b / (c + b)) / 2)
  : f_a + f_b + f_c ≥ 9 * r :=
sorry

end angle_bisectors_geq_nine_times_inradius_l211_21132


namespace problem1_problem2_l211_21123

theorem problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x + y ≥ 6 :=
sorry

theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x * y ≥ 9 :=
sorry

end problem1_problem2_l211_21123


namespace number_of_red_balls_eq_47_l211_21194

theorem number_of_red_balls_eq_47
  (T : ℕ) (white green yellow purple : ℕ)
  (neither_red_nor_purple_prob : ℚ)
  (hT : T = 100)
  (hWhite : white = 10)
  (hGreen : green = 30)
  (hYellow : yellow = 10)
  (hPurple : purple = 3)
  (hProb : neither_red_nor_purple_prob = 0.5)
  : T - (white + green + yellow + purple) = 47 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end number_of_red_balls_eq_47_l211_21194


namespace students_math_inequality_l211_21138

variables {n x a b c : ℕ}

theorem students_math_inequality (h1 : x + a ≥ 8 * n / 10) 
                                (h2 : x + b ≥ 8 * n / 10) 
                                (h3 : n ≥ a + b + c + x) : 
                                x * 5 ≥ 4 * (x + c) :=
by
  sorry

end students_math_inequality_l211_21138


namespace find_positive_integers_l211_21134

theorem find_positive_integers
  (a b c : ℕ) 
  (h : a ≥ b ∧ b ≥ c ∧ a ≥ c)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) :
  (1 + 1 / (a : ℚ)) * (1 + 1 / (b : ℚ)) * (1 + 1 / (c : ℚ)) = 2 →
  (a, b, c) ∈ [(15, 4, 2), (9, 5, 2), (7, 6, 2), (8, 3, 3), (5, 4, 3)] :=
by
  sorry

end find_positive_integers_l211_21134


namespace friction_coefficient_example_l211_21164

variable (α : ℝ) (mg : ℝ) (μ : ℝ)

theorem friction_coefficient_example
    (hα : α = 85 * Real.pi / 180) -- converting degrees to radians
    (hN : ∀ (N : ℝ), N = 6 * mg) -- Normal force in the vertical position
    (F : ℝ) -- Force applied horizontally by boy
    (hvert : F * Real.sin α - mg + (6 * mg) * Real.cos α = 0) -- vertical equilibrium
    (hhor : F * Real.cos α - μ * (6 * mg) - (6 * mg) * Real.sin α = 0) -- horizontal equilibrium
    : μ = 0.08 :=
by
  sorry

end friction_coefficient_example_l211_21164


namespace maria_total_earnings_l211_21125

-- Definitions of the conditions
def day1_tulips := 30
def day1_roses := 20
def day2_tulips := 2 * day1_tulips
def day2_roses := 2 * day1_roses
def day3_tulips := day2_tulips / 10
def day3_roses := 16
def tulip_price := 2
def rose_price := 3

-- Definition of the total earnings calculation
noncomputable def total_earnings : ℤ :=
  let total_tulips := day1_tulips + day2_tulips + day3_tulips
  let total_roses := day1_roses + day2_roses + day3_roses
  (total_tulips * tulip_price) + (total_roses * rose_price)

-- The proof statement
theorem maria_total_earnings : total_earnings = 420 := by
  sorry

end maria_total_earnings_l211_21125


namespace sheepdog_catches_sheep_l211_21187

-- Define the speeds and the time taken
def v_s : ℝ := 12 -- speed of the sheep in feet/second
def v_d : ℝ := 20 -- speed of the sheepdog in feet/second
def t : ℝ := 20 -- time in seconds

-- Define the initial distance between the sheep and the sheepdog
def initial_distance (v_s v_d t : ℝ) : ℝ :=
  v_d * t - v_s * t

theorem sheepdog_catches_sheep :
  initial_distance v_s v_d t = 160 :=
by
  -- The formal proof would go here, but for now we replace it with sorry
  sorry

end sheepdog_catches_sheep_l211_21187


namespace problem_solution_l211_21192

theorem problem_solution :
  ∃ x y z : ℕ,
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x^2 + y^2 + z^2 = 2 * (y * z + 1) ∧
    x + y + z = 4032 ∧
    x^2 * y + z = 4031 :=
by
  sorry

end problem_solution_l211_21192


namespace sum_of_three_numbers_l211_21102

theorem sum_of_three_numbers (a b c : ℝ) (h1 : (a + b + c) / 3 = a - 15) (h2 : (a + b + c) / 3 = c + 10) (h3 : b = 10) :
  a + b + c = 45 :=
  sorry

end sum_of_three_numbers_l211_21102


namespace greatest_common_divisor_84_n_l211_21179

theorem greatest_common_divisor_84_n :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∣ 84 ∧ d ∣ n → d = 1 ∨ d = 2 ∨ d = 4) ∧ (∀ (x y : ℕ), x ∣ 84 ∧ x ∣ n ∧ y ∣ 84 ∧ y ∣ n → x ≤ y → y = 4) :=
sorry

end greatest_common_divisor_84_n_l211_21179


namespace donation_to_second_home_l211_21146

-- Definitions of the conditions
def total_donation := 700.00
def first_home_donation := 245.00
def third_home_donation := 230.00

-- Define the unknown donation to the second home
noncomputable def second_home_donation := total_donation - first_home_donation - third_home_donation

-- The theorem to prove
theorem donation_to_second_home :
  second_home_donation = 225.00 :=
by sorry

end donation_to_second_home_l211_21146


namespace bryan_initial_pushups_l211_21121

def bryan_pushups (x : ℕ) : Prop :=
  let totalPushups := x + x + (x - 5)
  totalPushups = 40

theorem bryan_initial_pushups (x : ℕ) (hx : bryan_pushups x) : x = 15 :=
by {
  sorry
}

end bryan_initial_pushups_l211_21121


namespace harry_pencils_lost_l211_21160

-- Define the conditions
def anna_pencils : ℕ := 50
def harry_initial_pencils : ℕ := 2 * anna_pencils
def harry_current_pencils : ℕ := 81

-- Define the proof statement
theorem harry_pencils_lost :
  harry_initial_pencils - harry_current_pencils = 19 :=
by
  -- The proof is to be filled in
  sorry

end harry_pencils_lost_l211_21160


namespace students_total_l211_21118

theorem students_total (position_eunjung : ℕ) (following_students : ℕ) (h1 : position_eunjung = 6) (h2 : following_students = 7) : 
  position_eunjung + following_students = 13 :=
by
  sorry

end students_total_l211_21118


namespace volleyball_team_total_score_l211_21178

-- Define the conditions
def LizzieScore := 4
def NathalieScore := LizzieScore + 3
def CombinedLizzieNathalieScore := LizzieScore + NathalieScore
def AimeeScore := 2 * CombinedLizzieNathalieScore
def TeammatesScore := 17

-- Prove that the total team score is 50
theorem volleyball_team_total_score :
  LizzieScore + NathalieScore + AimeeScore + TeammatesScore = 50 :=
by
  sorry

end volleyball_team_total_score_l211_21178


namespace differential_savings_l211_21182

def annual_income_before_tax : ℝ := 42400
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.32

theorem differential_savings :
  annual_income_before_tax * initial_tax_rate - annual_income_before_tax * new_tax_rate = 4240 :=
by
  sorry

end differential_savings_l211_21182


namespace perimeter_one_face_of_cube_is_24_l211_21111

noncomputable def cube_volume : ℝ := 216
def perimeter_of_face_of_cube (V : ℝ) : ℝ := 4 * (V^(1/3) : ℝ)

theorem perimeter_one_face_of_cube_is_24 :
  perimeter_of_face_of_cube cube_volume = 24 := 
by
  -- This proof will invoke the calculation shown in the problem.
  sorry

end perimeter_one_face_of_cube_is_24_l211_21111


namespace cylinder_ratio_l211_21180

theorem cylinder_ratio (m r : ℝ) (h1 : m + 2 * r = Real.sqrt (m^2 + (r * Real.pi)^2)) :
  m / (2 * r) = (Real.pi^2 - 4) / 8 := by
  sorry

end cylinder_ratio_l211_21180


namespace perpendicular_condition_l211_21144

theorem perpendicular_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → (m * x + y + 1 = 0 → (2 * m - 1 = 0))) ↔ (m = 1/2) :=
by sorry

end perpendicular_condition_l211_21144


namespace find_d_value_l211_21162

/-- Let d be an odd prime number. If 89 - (d+3)^2 is the square of an integer, then d = 5. -/
theorem find_d_value (d : ℕ) (h₁ : Nat.Prime d) (h₂ : Odd d) (h₃ : ∃ m : ℤ, 89 - (d + 3)^2 = m^2) : d = 5 := 
by
  sorry

end find_d_value_l211_21162


namespace log_expression_value_l211_21150

theorem log_expression_value :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 :=
by
  sorry

end log_expression_value_l211_21150


namespace book_selection_l211_21165

theorem book_selection (total_books novels : ℕ) (choose_books : ℕ)
  (h_total : total_books = 15)
  (h_novels : novels = 5)
  (h_choose : choose_books = 3) :
  (Nat.choose 15 3 - Nat.choose 10 3) = 335 :=
by
  sorry

end book_selection_l211_21165


namespace walt_age_l211_21119

variable (W M P : ℕ)

-- Conditions
def condition1 := M = 3 * W
def condition2 := M + 12 = 2 * (W + 12)
def condition3 := P = 4 * W
def condition4 := P + 15 = 3 * (W + 15)

theorem walt_age (W M P : ℕ) (h1 : condition1 W M) (h2 : condition2 W M) (h3 : condition3 W P) (h4 : condition4 W P) : 
  W = 30 :=
sorry

end walt_age_l211_21119


namespace problem_l211_21122

variable (m n : ℝ)
variable (h1 : m + n = -1994)
variable (h2 : m * n = 7)

theorem problem (m n : ℝ) (h1 : m + n = -1994) (h2 : m * n = 7) : 
  (m^2 + 1993 * m + 6) * (n^2 + 1995 * n + 8) = 1986 := 
by
  sorry

end problem_l211_21122


namespace f_of_1_l211_21155

theorem f_of_1 (f : ℕ+ → ℕ+) (h_mono : ∀ {a b : ℕ+}, a < b → f a < f b)
  (h_fn_prop : ∀ n : ℕ+, f (f n) = 3 * n) : f 1 = 2 :=
sorry

end f_of_1_l211_21155


namespace solve_furniture_factory_l211_21173

variable (num_workers : ℕ) (tables_per_worker : ℕ) (legs_per_worker : ℕ) 
variable (tabletop_workers legs_workers : ℕ)

axiom worker_capacity : tables_per_worker = 3 ∧ legs_per_worker = 6
axiom total_workers : num_workers = 60
axiom table_leg_ratio : ∀ (x : ℕ), tabletop_workers = x → legs_workers = (num_workers - x)
axiom daily_production_eq : ∀ (x : ℕ), (4 * tables_per_worker * x = 6 * legs_per_worker * (num_workers - x))

theorem solve_furniture_factory : 
  ∃ (x y : ℕ), num_workers = x + y ∧ 
            4 * 3 * x = 6 * (num_workers - x) ∧ 
            x = 20 ∧ y = (num_workers - 20) := by
  sorry

end solve_furniture_factory_l211_21173


namespace find_constants_l211_21104

theorem find_constants :
  ∃ (A B C : ℝ), (∀ x : ℝ, x ≠ 3 → x ≠ 4 → 
  (6 * x / ((x - 4) * (x - 3) ^ 2)) = (A / (x - 4) + B / (x - 3) + C / (x - 3) ^ 2)) ∧
  A = 24 ∧
  B = - 162 / 7 ∧
  C = - 18 :=
by
  use 24, -162 / 7, -18
  sorry

end find_constants_l211_21104


namespace negation_of_universal_l211_21139

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by
  sorry

end negation_of_universal_l211_21139


namespace part1_part2_l211_21117

def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2 * a|

-- Define the first part of the problem
theorem part1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
sorry

-- Define the second part of the problem
theorem part2 (a x : ℝ) (h1 : a ≥ 1) : f x a ≥ 2 :=
sorry

end part1_part2_l211_21117


namespace quadrilateral_area_is_22_5_l211_21168

-- Define the vertices of the quadrilateral
def vertex1 : ℝ × ℝ := (3, -1)
def vertex2 : ℝ × ℝ := (-1, 4)
def vertex3 : ℝ × ℝ := (2, 3)
def vertex4 : ℝ × ℝ := (9, 9)

-- Define the function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  0.5 * (abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) 
        - (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)))

-- State that the area of the quadrilateral with given vertices is 22.5
theorem quadrilateral_area_is_22_5 :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 22.5 :=
by 
  -- We skip the proof here.
  sorry

end quadrilateral_area_is_22_5_l211_21168


namespace divide_plane_into_regions_l211_21151

theorem divide_plane_into_regions :
  (∀ (x y : ℝ), y = 3 * x ∨ y = x / 3) →
  ∃ (regions : ℕ), regions = 4 :=
by
  sorry

end divide_plane_into_regions_l211_21151


namespace problem_solution_l211_21141

def positive (n : ℕ) : Prop := n > 0
def pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1
def divides (m n : ℕ) : Prop := ∃ k, n = k * m

theorem problem_solution (a b c : ℕ) :
  positive a → positive b → positive c →
  pairwise_coprime a b c →
  divides (a^2) (b^3 + c^3) →
  divides (b^2) (a^3 + c^3) →
  divides (c^2) (a^3 + b^3) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end problem_solution_l211_21141


namespace tim_interest_rate_l211_21185

theorem tim_interest_rate
  (r : ℝ)
  (h1 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000))
  (h2 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000) + 23.5) : 
  r = 0.1 :=
by
  sorry

end tim_interest_rate_l211_21185


namespace gcd_123456_789012_l211_21112

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := sorry

end gcd_123456_789012_l211_21112


namespace complement_union_l211_21135

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem complement_union (x : ℝ) : (x ∈ Aᶜ ∪ B) ↔ (x ∈ Set.Iic (-1) ∪ Set.Ioi 0) := by
  sorry

end complement_union_l211_21135


namespace plywood_perimeter_difference_l211_21186

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l211_21186


namespace maximize_x4y3_l211_21101

theorem maximize_x4y3 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = 40) : 
    (x, y) = (160 / 7, 120 / 7) ↔ x ^ 4 * y ^ 3 ≤ (160 / 7) ^ 4 * (120 / 7) ^ 3 := 
sorry

end maximize_x4y3_l211_21101


namespace general_term_arithmetic_sequence_l211_21116

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n, a n = 2 * n - 1 := 
by
  intros h1 h2 n
  sorry

end general_term_arithmetic_sequence_l211_21116


namespace log_expression_identity_l211_21190

theorem log_expression_identity :
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 1 :=
by
  sorry

end log_expression_identity_l211_21190


namespace quadratic_no_real_roots_l211_21145

theorem quadratic_no_real_roots (m : ℝ) (h : ∀ x : ℝ, x^2 - m * x + 1 ≠ 0) : m = 0 :=
by
  sorry

end quadratic_no_real_roots_l211_21145


namespace milk_production_days_l211_21156

variable {x : ℕ}

def daily_cow_production (x : ℕ) : ℚ := (x + 4) / ((x + 2) * (x + 3))

def total_daily_production (x : ℕ) : ℚ := (x + 5) * daily_cow_production x

def required_days (x : ℕ) : ℚ := (x + 9) / total_daily_production x

theorem milk_production_days : 
  required_days x = (x + 9) * (x + 2) * (x + 3) / ((x + 5) * (x + 4)) := 
by 
  sorry

end milk_production_days_l211_21156


namespace unique_solution_l211_21152

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end unique_solution_l211_21152


namespace exists_line_equidistant_from_AB_CD_l211_21175

noncomputable def Line : Type := sorry  -- This would be replaced with an appropriate definition of a line in space

def Point : Type := sorry  -- Similarly, a point in space type definition

variables (A B C D : Point)

def perpendicularBisector (P Q : Point) : Type := sorry  -- Definition for perpendicular bisector plane of two points

def is_perpendicularBisector_of (e : Line) (P Q : Point) : Prop := sorry  -- e is perpendicular bisector plane of P and Q

theorem exists_line_equidistant_from_AB_CD (A B C D : Point) :
  ∃ e : Line, is_perpendicularBisector_of e A C ∧ is_perpendicularBisector_of e B D :=
by
  sorry

end exists_line_equidistant_from_AB_CD_l211_21175


namespace num_possible_values_for_n_l211_21166

open Real

noncomputable def count_possible_values_for_n : ℕ :=
  let log2 := log 2
  let log2_9 := log 9 / log2
  let log2_50 := log 50 / log2
  let range_n := ((6 : ℕ), 450)
  let count := range_n.2 - range_n.1 + 1
  count

theorem num_possible_values_for_n :
  count_possible_values_for_n = 445 :=
by
  sorry

end num_possible_values_for_n_l211_21166


namespace car_speed_l211_21148

variable (fuel_efficiency : ℝ) (fuel_decrease_gallons : ℝ) (time_hours : ℝ) 
          (gallons_to_liters : ℝ) (kilometers_to_miles : ℝ)
          (car_speed_mph : ℝ)

-- Conditions given in the problem
def fuelEfficiency : ℝ := 40 -- km per liter
def fuelDecreaseGallons : ℝ := 3.9 -- gallons
def timeHours : ℝ := 5.7 -- hours
def gallonsToLiters : ℝ := 3.8 -- liters per gallon
def kilometersToMiles : ℝ := 1.6 -- km per mile

theorem car_speed (fuel_efficiency fuelDecreaseGallons timeHours gallonsToLiters kilometersToMiles : ℝ) : 
  let fuelDecreaseLiters := fuelDecreaseGallons * gallonsToLiters
  let distanceKm := fuelDecreaseLiters * fuel_efficiency
  let distanceMiles := distanceKm / kilometersToMiles
  let averageSpeed := distanceMiles / timeHours
  averageSpeed = 65 := sorry

end car_speed_l211_21148


namespace polygon_sides_l211_21158

def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360

theorem polygon_sides (n : ℕ) (h : 1/4 * sum_interior_angles n - sum_exterior_angles = 90) : n = 12 := 
by
  -- sorry to skip the proof
  sorry

end polygon_sides_l211_21158


namespace blue_stripe_area_l211_21176

def cylinder_diameter : ℝ := 20
def cylinder_height : ℝ := 60
def stripe_width : ℝ := 4
def stripe_revolutions : ℕ := 3

theorem blue_stripe_area : 
  let circumference := Real.pi * cylinder_diameter
  let stripe_length := stripe_revolutions * circumference
  let expected_area := stripe_width * stripe_length
  expected_area = 240 * Real.pi :=
by
  sorry

end blue_stripe_area_l211_21176


namespace certain_number_l211_21174

theorem certain_number (x : ℤ) (h : 12 + x = 27) : x = 15 :=
by
  sorry

end certain_number_l211_21174


namespace find_mnp_l211_21115

noncomputable def equation_rewrite (a b x y : ℝ) (m n p : ℕ): Prop :=
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) ∧
  (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5

theorem find_mnp (a b x y : ℝ): 
  equation_rewrite a b x y 2 1 4 ∧ (2 * 1 * 4 = 8) :=
by 
  sorry

end find_mnp_l211_21115


namespace S_10_is_65_l211_21107

variable (a_1 d : ℤ)
variable (S : ℤ → ℤ)

-- Define the arithmetic sequence conditions
def a_3 : ℤ := a_1 + 2 * d
def S_n (n : ℤ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom a_3_is_4 : a_3 = 4
axiom S_9_minus_S_6_is_27 : S 9 - S 6 = 27

-- The target statement to be proven
theorem S_10_is_65 : S 10 = 65 :=
by
  sorry

end S_10_is_65_l211_21107


namespace negated_roots_quadratic_reciprocals_roots_quadratic_l211_21172

-- For (1)
theorem negated_roots_quadratic (x y : ℝ) : 
    (x^2 + 3 * x - 2 = 0) ↔ (y^2 - 3 * y - 2 = 0) :=
sorry

-- For (2)
theorem reciprocals_roots_quadratic (a b c x y : ℝ) (h : a ≠ 0) :
    (a * x^2 - b * x + c = 0) ↔ (c * y^2 - b * y + a = 0) :=
sorry

end negated_roots_quadratic_reciprocals_roots_quadratic_l211_21172


namespace compare_probabilities_l211_21159

theorem compare_probabilities 
  (M N : ℕ)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_million : m > 10^6)
  (h_n_million : n ≤ 10^6) :
  (M : ℝ) / (M + n / m * N) > (M : ℝ) / (M + N) :=
by
  sorry

end compare_probabilities_l211_21159


namespace remainder_problem_l211_21124

theorem remainder_problem :
  (1234567 % 135 = 92) ∧ ((92 * 5) % 27 = 1) := by
  sorry

end remainder_problem_l211_21124


namespace sum_of_cubes_form_l211_21191

theorem sum_of_cubes_form (a b : ℤ) (x1 y1 x2 y2 : ℤ)
  (h1 : a = x1^2 + 3 * y1^2) (h2 : b = x2^2 + 3 * y2^2) :
  ∃ x y : ℤ, a^3 + b^3 = x^2 + 3 * y^2 := sorry

end sum_of_cubes_form_l211_21191


namespace final_answer_is_d_l211_21195

-- Definitions of the propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x > 1
def q : Prop := false  -- since the distance between focus and directrix is not 1/6 but 3/2

-- The statement to be proven
theorem final_answer_is_d : p ∧ ¬ q := by sorry

end final_answer_is_d_l211_21195


namespace parabola_focus_equals_ellipse_focus_l211_21177

theorem parabola_focus_equals_ellipse_focus (p : ℝ) : 
  let parabola_focus := (p / 2, 0)
  let ellipse_focus := (2, 0)
  parabola_focus = ellipse_focus → p = 4 :=
by
  intros h
  sorry

end parabola_focus_equals_ellipse_focus_l211_21177


namespace variance_of_data_set_l211_21171

theorem variance_of_data_set (m : ℝ) (h_mean : (6 + 7 + 8 + 9 + m) / 5 = 8) :
    (1/5) * ((6-8)^2 + (7-8)^2 + (8-8)^2 + (9-8)^2 + (m-8)^2) = 2 := 
sorry

end variance_of_data_set_l211_21171


namespace fourth_month_sale_is_7200_l211_21157

-- Define the sales amounts for each month
def sale_first_month : ℕ := 6400
def sale_second_month : ℕ := 7000
def sale_third_month : ℕ := 6800
def sale_fifth_month : ℕ := 6500
def sale_sixth_month : ℕ := 5100
def average_sale : ℕ := 6500

-- Total requirements for the six months
def total_required_sales : ℕ := 6 * average_sale

-- Known sales for five months
def total_known_sales : ℕ := sale_first_month + sale_second_month + sale_third_month + sale_fifth_month + sale_sixth_month

-- Sale in the fourth month
def sale_fourth_month : ℕ := total_required_sales - total_known_sales

-- The theorem to prove
theorem fourth_month_sale_is_7200 : sale_fourth_month = 7200 :=
by
  sorry

end fourth_month_sale_is_7200_l211_21157


namespace triangle_side_count_l211_21167

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l211_21167


namespace a2_range_l211_21169

open Nat

noncomputable def a_seq (a : ℕ → ℝ) := ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)

theorem a2_range (a : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)) 
  (h2 : ∀ (m n : ℕ), m < n → a m ≤ a n) 
  (h3 : a 1 = 2) :
  (2 < a 2) ∧ (a 2 ≤ 4) :=
sorry

end a2_range_l211_21169


namespace point_on_parabola_dist_3_from_focus_l211_21193

def parabola (p : ℝ × ℝ) : Prop := (p.snd)^2 = 4 * p.fst

def focus : ℝ × ℝ := (1, 0)

theorem point_on_parabola_dist_3_from_focus :
  ∃ y: ℝ, ∃ x: ℝ, (parabola (x, y) ∧ (x = 2) ∧ (y = 2 * Real.sqrt 2 ∨ y = -2 * Real.sqrt 2) ∧ (Real.sqrt ((x - focus.fst)^2 + (y - focus.snd)^2) = 3)) :=
by
  sorry

end point_on_parabola_dist_3_from_focus_l211_21193


namespace g_of_f_roots_reciprocal_l211_21142

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c

theorem g_of_f_roots_reciprocal
  (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ∃ g : ℝ → ℝ, g 1 = (4 - a) / (4 * c) :=
sorry

end g_of_f_roots_reciprocal_l211_21142


namespace no_valid_number_l211_21196

theorem no_valid_number (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 9) : ¬ ∃ (y : ℕ), (x * 100 + 3 * 10 + y) % 11 = 0 :=
by
  sorry

end no_valid_number_l211_21196


namespace minimum_red_chips_l211_21106

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ w / 4) (h2 : b ≤ r / 6) (h3 : w + b ≥ 75) : r ≥ 90 :=
sorry

end minimum_red_chips_l211_21106


namespace probability_divisible_by_3_l211_21154

theorem probability_divisible_by_3 :
  ∀ (n : ℤ), (1 ≤ n) ∧ (n ≤ 99) → 3 ∣ (n * (n + 1)) :=
by
  intros n hn
  -- Detailed proof would follow here
  sorry

end probability_divisible_by_3_l211_21154


namespace starting_number_of_sequence_l211_21133

theorem starting_number_of_sequence :
  ∃ (start : ℤ), 
    (∀ n, 0 ≤ n ∧ n < 8 → start + n * 11 ≤ 119) ∧ 
    (∃ k, 1 ≤ k ∧ k ≤ 8 ∧ 119 = start + (k - 1) * 11) ↔ start = 33 :=
by
  sorry

end starting_number_of_sequence_l211_21133


namespace number_of_pipes_used_l211_21199

-- Definitions
def T1 : ℝ := 15
def T2 : ℝ := T1 - 5
def T3 : ℝ := T2 - 4
def condition : Prop := 1 / T1 + 1 / T2 = 1 / T3

-- Proof Statement
theorem number_of_pipes_used : condition → 3 = 3 :=
by intros h; sorry

end number_of_pipes_used_l211_21199


namespace shoe_count_l211_21189

theorem shoe_count 
  (pairs : ℕ)
  (total_shoes : ℕ)
  (prob : ℝ)
  (h_pairs : pairs = 12)
  (h_prob : prob = 0.043478260869565216)
  (h_total_shoes : total_shoes = pairs * 2) :
  total_shoes = 24 :=
by
  sorry

end shoe_count_l211_21189


namespace hula_hoop_ratio_l211_21128

variable (Nancy Casey Morgan : ℕ)
variable (hula_hoop_time_Nancy : Nancy = 10)
variable (hula_hoop_time_Casey : Casey = Nancy - 3)
variable (hula_hoop_time_Morgan : Morgan = 21)

theorem hula_hoop_ratio (hula_hoop_time_Nancy : Nancy = 10) (hula_hoop_time_Casey : Casey = Nancy - 3) (hula_hoop_time_Morgan : Morgan = 21) :
  Morgan / Casey = 3 := by
  sorry

end hula_hoop_ratio_l211_21128


namespace max_neg_expr_l211_21149

theorem max_neg_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (- (1 / (2 * a)) - (2 / b)) ≤ - (9 / 2) :=
sorry

end max_neg_expr_l211_21149


namespace problem_part_1_problem_part_2_l211_21197

variable (θ : Real)
variable (m : Real)
variable (h_θ : θ ∈ Ioc 0 (2 * Real.pi))
variable (h_eq : ∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ (x = Real.sin θ ∨ x = Real.cos θ))

theorem problem_part_1 : 
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 := 
by
  sorry

theorem problem_part_2 : 
  m = Real.sqrt 3 / 2 := 
by 
  sorry

end problem_part_1_problem_part_2_l211_21197


namespace find_K_find_t_l211_21137

-- Proof Problem for G9.2
theorem find_K (x : ℚ) (K : ℚ) (h1 : x = 1.9898989) (h2 : x - 1 = K / 99) : K = 98 :=
sorry

-- Proof Problem for G9.3
theorem find_t (p q r t : ℚ)
  (h_avg1 : (p + q + r) / 3 = 18)
  (h_avg2 : ((p + 1) + (q - 2) + (r + 3) + t) / 4 = 19) : t = 20 :=
sorry

end find_K_find_t_l211_21137


namespace nests_count_l211_21127

theorem nests_count :
  ∃ (N : ℕ), (6 = N + 3) ∧ (N = 3) :=
by
  sorry

end nests_count_l211_21127


namespace age_proof_l211_21126

theorem age_proof (M S Y : ℕ) (h1 : M = 36) (h2 : S = 12) (h3 : M = 3 * S) : 
  (M + Y = 2 * (S + Y)) ↔ (Y = 12) :=
by 
  sorry

end age_proof_l211_21126


namespace Toby_change_l211_21183

def change (orders_cost per_person total_cost given_amount : ℝ) : ℝ :=
  given_amount - per_person

def total_cost (cheeseburgers milkshake coke fries cookies tax : ℝ) : ℝ :=
  cheeseburgers + milkshake + coke + fries + cookies + tax

theorem Toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2.0
  let coke_cost := 1.0
  let fries_cost := 4.0
  let cookie_cost := 3 * 0.5 -- Total cost for three cookies
  let tax := 0.2
  let total := total_cost (2 * cheeseburger_cost) milkshake_cost coke_cost fries_cost cookie_cost tax
  let per_person := total / 2
  let toby_arrival := 15.0
  change total per_person total toby_arrival = 7 :=
by
  sorry

end Toby_change_l211_21183
