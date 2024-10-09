import Mathlib

namespace hours_per_shift_l1281_128117

def hourlyWage : ℝ := 4.0
def tipRate : ℝ := 0.15
def shiftsWorked : ℕ := 3
def averageOrdersPerHour : ℝ := 40.0
def totalEarnings : ℝ := 240.0

theorem hours_per_shift :
  (hourlyWage + averageOrdersPerHour * tipRate) * (8 * shiftsWorked) = totalEarnings := 
sorry

end hours_per_shift_l1281_128117


namespace perimeter_of_square_is_32_l1281_128133

-- Given conditions
def radius := 4
def diameter := 2 * radius
def side_length_of_square := diameter

-- Question: What is the perimeter of the square?
def perimeter_of_square := 4 * side_length_of_square

-- Proof statement
theorem perimeter_of_square_is_32 : perimeter_of_square = 32 :=
sorry

end perimeter_of_square_is_32_l1281_128133


namespace total_players_is_59_l1281_128121

-- Define the number of players from each sport.
def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def softball_players : ℕ := 13

-- Define the total number of players as the sum of the above.
def total_players : ℕ :=
  cricket_players + hockey_players + football_players + softball_players

-- Prove that the total number of players is 59.
theorem total_players_is_59 :
  total_players = 59 :=
by
  unfold total_players
  unfold cricket_players
  unfold hockey_players
  unfold football_players
  unfold softball_players
  sorry

end total_players_is_59_l1281_128121


namespace value_of_x_l1281_128168

theorem value_of_x (x : ℝ) (h : 0.75 * 600 = 0.50 * x) : x = 900 :=
by
  sorry

end value_of_x_l1281_128168


namespace range_of_a_l1281_128128

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem range_of_a (a : ℝ) :
  (∀ x y, 3 ≤ x ∧ x ≤ y → (x^2 - 2*a*x + 2) ≤ (y^2 - 2*a*y + 2)) → a ≤ 3 := 
sorry

end range_of_a_l1281_128128


namespace cookies_none_of_ingredients_l1281_128142

theorem cookies_none_of_ingredients (c : ℕ) (o : ℕ) (r : ℕ) (a : ℕ) (total_cookies : ℕ) :
  total_cookies = 48 ∧ c = total_cookies / 3 ∧ o = (3 * total_cookies + 4) / 5 ∧ r = total_cookies / 2 ∧ a = total_cookies / 8 → 
  ∃ n, n = 19 ∧ (∀ k, k = total_cookies - max c (max o (max r a)) → k ≤ n) :=
by sorry

end cookies_none_of_ingredients_l1281_128142


namespace find_p_q_r_l1281_128113

def f (x : ℝ) : ℝ := x^2 + 2*x + 2
def g (x p q r : ℝ) : ℝ := x^3 + 2*x^2 + 6*p*x + 4*q*x + r

noncomputable def roots_sum_f := -2
noncomputable def roots_product_f := 2

theorem find_p_q_r (p q r : ℝ) (h1 : ∀ x, f x = 0 → g x p q r = 0) :
  (p + q) * r = 0 :=
sorry

end find_p_q_r_l1281_128113


namespace two_digit_number_l1281_128147

theorem two_digit_number (x y : Nat) : 
  10 * x + y = 10 * x + y := 
by 
  sorry

end two_digit_number_l1281_128147


namespace functional_inequality_solution_l1281_128179

theorem functional_inequality_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)) ↔ (∀ x : ℝ, f x = x + 2) :=
by
  sorry

end functional_inequality_solution_l1281_128179


namespace arithmetic_geometric_mean_inequality_l1281_128198

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l1281_128198


namespace bed_width_is_4_feet_l1281_128165

def total_bags : ℕ := 16
def soil_per_bag : ℕ := 4
def bed_length : ℝ := 8
def bed_height : ℝ := 1
def num_beds : ℕ := 2

theorem bed_width_is_4_feet :
  (total_bags * soil_per_bag / num_beds) = (bed_length * 4 * bed_height) :=
by
  sorry

end bed_width_is_4_feet_l1281_128165


namespace train_length_is_300_l1281_128166

noncomputable def speed_kmph : Float := 90
noncomputable def speed_mps : Float := (speed_kmph * 1000) / 3600
noncomputable def time_sec : Float := 12
noncomputable def length_of_train : Float := speed_mps * time_sec

theorem train_length_is_300 : length_of_train = 300 := by
  sorry

end train_length_is_300_l1281_128166


namespace find_divisor_l1281_128134

-- Condition Definitions
def dividend : ℕ := 725
def quotient : ℕ := 20
def remainder : ℕ := 5

-- Target Proof Statement
theorem find_divisor (divisor : ℕ) (h : dividend = divisor * quotient + remainder) : divisor = 36 := by
  sorry

end find_divisor_l1281_128134


namespace number_of_people_l1281_128161

theorem number_of_people (clinks : ℕ) (h : clinks = 45) : ∃ x : ℕ, x * (x - 1) / 2 = clinks ∧ x = 10 :=
by
  sorry

end number_of_people_l1281_128161


namespace symmetric_circle_equation_l1281_128160

theorem symmetric_circle_equation :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 6 * x + 8 * y + 24 = 0) →
    (x - 3 * y - 5 = 0) →
    (∃ x₀ y₀ : ℝ, (x₀ - 1)^2 + (y₀ - 2)^2 = 1) :=
by
  sorry

end symmetric_circle_equation_l1281_128160


namespace area_triangle_BRS_l1281_128143

def point := ℝ × ℝ
def x_intercept (p : point) : ℝ := p.1
def y_intercept (p : point) : ℝ := p.2

noncomputable def distance_from_y_axis (p : point) : ℝ := abs p.1

theorem area_triangle_BRS (B R S : point)
  (hB : B = (4, 10))
  (h_perp : ∃ m₁ m₂, m₁ * m₂ = -1)
  (h_sum_zero : x_intercept R + x_intercept S = 0)
  (h_dist : distance_from_y_axis B = 10) :
  ∃ area : ℝ, area = 60 := 
sorry

end area_triangle_BRS_l1281_128143


namespace proof_no_natural_solutions_l1281_128111

noncomputable def no_natural_solutions : Prop :=
  ∀ x y : ℕ, y^2 ≠ x^2 + x + 1

theorem proof_no_natural_solutions : no_natural_solutions :=
by
  intros x y
  sorry

end proof_no_natural_solutions_l1281_128111


namespace cost_in_chinese_yuan_l1281_128137

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end cost_in_chinese_yuan_l1281_128137


namespace kaleb_candy_problem_l1281_128148

-- Define the initial problem with given conditions

theorem kaleb_candy_problem :
  ∀ (total_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ),
    total_boxes = 14 →
    given_away_boxes = 5 →
    pieces_per_box = 6 →
    (total_boxes - given_away_boxes) * pieces_per_box = 54 :=
by
  intros total_boxes given_away_boxes pieces_per_box
  intros h1 h2 h3
  -- Use assumptions
  sorry

end kaleb_candy_problem_l1281_128148


namespace function_y_neg3x_plus_1_quadrants_l1281_128120

theorem function_y_neg3x_plus_1_quadrants :
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x + 1) ∧ (
    (x < 0 ∧ y > 0) ∨ -- Second quadrant
    (x > 0 ∧ y > 0) ∨ -- First quadrant
    (x > 0 ∧ y < 0)   -- Fourth quadrant
  )
:= sorry

end function_y_neg3x_plus_1_quadrants_l1281_128120


namespace gcd_1230_990_l1281_128173

theorem gcd_1230_990 : Int.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l1281_128173


namespace savings_by_going_earlier_l1281_128122

/-- Define the cost of evening ticket -/
def evening_ticket_cost : ℝ := 10

/-- Define the cost of large popcorn & drink combo -/
def food_combo_cost : ℝ := 10

/-- Define the discount percentage on tickets from 12 noon to 3 pm -/
def ticket_discount : ℝ := 0.20

/-- Define the discount percentage on food combos from 12 noon to 3 pm -/
def food_combo_discount : ℝ := 0.50

/-- Prove that the total savings Trip could achieve by going to the earlier movie is $7 -/
theorem savings_by_going_earlier : 
  (ticket_discount * evening_ticket_cost) + (food_combo_discount * food_combo_cost) = 7 := by
  sorry

end savings_by_going_earlier_l1281_128122


namespace F_is_even_l1281_128115

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (x^3 - 2*x) * f x

theorem F_is_even (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_nonzero : f 1 ≠ 0) :
  is_even_function (F f) :=
sorry

end F_is_even_l1281_128115


namespace percentage_fraction_l1281_128151

theorem percentage_fraction (P : ℚ) (hP : P < 35) (h : (P / 100) * 180 = 42) : P = 7 / 30 * 100 :=
by
  sorry

end percentage_fraction_l1281_128151


namespace stratified_sampling_l1281_128181

theorem stratified_sampling (total_students boys girls sample_size x y : ℕ)
  (h1 : total_students = 8)
  (h2 : boys = 6)
  (h3 : girls = 2)
  (h4 : sample_size = 4)
  (h5 : x + y = sample_size)
  (h6 : (x : ℚ) / boys = 3 / 4)
  (h7 : (y : ℚ) / girls = 1 / 4) :
  x = 3 ∧ y = 1 :=
by
  sorry

end stratified_sampling_l1281_128181


namespace contractor_fine_per_absent_day_l1281_128177

noncomputable def fine_per_absent_day (total_days : ℕ) (pay_per_day : ℝ) (total_amount_received : ℝ) (days_absent : ℕ) : ℝ :=
  let days_worked := total_days - days_absent
  let earned := days_worked * pay_per_day
  let fine := (earned - total_amount_received) / days_absent
  fine

theorem contractor_fine_per_absent_day :
  fine_per_absent_day 30 25 425 10 = 7.5 := by
  sorry

end contractor_fine_per_absent_day_l1281_128177


namespace K_time_for_distance_l1281_128193

theorem K_time_for_distance (s : ℝ) (hs : s > 0) :
  (let K_time := 45 / s
   let M_speed := s - 1 / 2
   let M_time := 45 / M_speed
   K_time = M_time - 3 / 4) -> K_time = 45 / s := 
by
  sorry

end K_time_for_distance_l1281_128193


namespace point_in_second_quadrant_l1281_128192

def P : ℝ × ℝ := (-5, 4)

theorem point_in_second_quadrant (p : ℝ × ℝ) (hx : p.1 = -5) (hy : p.2 = 4) : p.1 < 0 ∧ p.2 > 0 :=
by
  sorry

example : P.1 < 0 ∧ P.2 > 0 :=
  point_in_second_quadrant P rfl rfl

end point_in_second_quadrant_l1281_128192


namespace soda_difference_l1281_128189

-- Define the number of regular soda bottles
def R : ℕ := 79

-- Define the number of diet soda bottles
def D : ℕ := 53

-- The theorem that states the number of regular soda bottles minus the number of diet soda bottles is 26
theorem soda_difference : R - D = 26 := 
by
  sorry

end soda_difference_l1281_128189


namespace total_hours_proof_l1281_128136

-- Definitions and conditions
def kate_hours : ℕ := 22
def pat_hours : ℕ := 2 * kate_hours
def mark_hours : ℕ := kate_hours + 110

-- Statement of the proof problem
theorem total_hours_proof : pat_hours + kate_hours + mark_hours = 198 := by
  sorry

end total_hours_proof_l1281_128136


namespace factor_expression_l1281_128152

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) :=
by
  sorry

end factor_expression_l1281_128152


namespace base_7_is_good_number_l1281_128102

def is_good_number (m: ℕ) : Prop :=
  ∃ (p: ℕ) (n: ℕ), Prime p ∧ n ≥ 2 ∧ m = p^n

theorem base_7_is_good_number : 
  ∀ b: ℕ, (is_good_number (b^2 - (2 * b + 3))) → b = 7 :=
by
  intro b h
  sorry

end base_7_is_good_number_l1281_128102


namespace seventh_term_arith_seq_l1281_128176

/-- 
The seventh term of an arithmetic sequence given that the sum of the first five terms 
is 15 and the sixth term is 7.
-/
theorem seventh_term_arith_seq (a d : ℚ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 7) : 
  a + 6 * d = 25 / 3 := 
sorry

end seventh_term_arith_seq_l1281_128176


namespace inequality_proof_l1281_128146

theorem inequality_proof (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  (x1^2 + x2^2 + x3^2)^3 / (x1^3 + x2^3 + x3^3)^2 ≤ 3 :=
sorry

end inequality_proof_l1281_128146


namespace max_product_of_slopes_l1281_128130

theorem max_product_of_slopes 
  (m₁ m₂ : ℝ)
  (h₁ : m₂ = 3 * m₁)
  (h₂ : abs ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.sqrt 3) :
  m₁ * m₂ ≤ 2 :=
sorry

end max_product_of_slopes_l1281_128130


namespace value_of_a_if_1_in_S_l1281_128108

variable (a : ℤ)
def S := { x : ℤ | 3 * x + a = 0 }

theorem value_of_a_if_1_in_S (h : 1 ∈ S a) : a = -3 :=
sorry

end value_of_a_if_1_in_S_l1281_128108


namespace weekly_spending_l1281_128172

-- Definitions based on the conditions outlined in the original problem
def weekly_allowance : ℝ := 50
def hours_per_week : ℕ := 30
def hourly_wage : ℝ := 9
def weeks_per_year : ℕ := 52
def first_year_allowance : ℝ := weekly_allowance * weeks_per_year
def second_year_earnings : ℝ := (hourly_wage * hours_per_week) * weeks_per_year
def total_car_cost : ℝ := 15000
def additional_needed : ℝ := 2000
def total_savings : ℝ := first_year_allowance + second_year_earnings

-- The amount Thomas needs over what he has saved
def total_needed : ℝ := total_savings + additional_needed
def amount_spent_on_self : ℝ := total_needed - total_car_cost
def total_weeks : ℕ := 2 * weeks_per_year

theorem weekly_spending :
  amount_spent_on_self / total_weeks = 35 := by
  sorry

end weekly_spending_l1281_128172


namespace highest_x_value_satisfies_equation_l1281_128199

theorem highest_x_value_satisfies_equation:
  ∃ x, x ≤ 4 ∧ (∀ x1, x1 ≤ 4 → x1 = 4 ↔ (15 * x1^2 - 40 * x1 + 18) / (4 * x1 - 3) + 7 * x1 = 9 * x1 - 2) :=
by
  sorry

end highest_x_value_satisfies_equation_l1281_128199


namespace smallest_fraction_numerator_l1281_128150

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (a * 4 > b * 3) ∧ (a = 73) := 
sorry

end smallest_fraction_numerator_l1281_128150


namespace mutually_exclusive_iff_complementary_l1281_128126

variables {Ω : Type} (A₁ A₂ : Set Ω) (S : Set Ω)

/-- Proposition A: Events A₁ and A₂ are mutually exclusive. -/
def mutually_exclusive : Prop := A₁ ∩ A₂ = ∅

/-- Proposition B: Events A₁ and A₂ are complementary. -/
def complementary : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = S

/-- Proposition A is a necessary but not sufficient condition for Proposition B. -/
theorem mutually_exclusive_iff_complementary :
  mutually_exclusive A₁ A₂ → (complementary A₁ A₂ S → mutually_exclusive A₁ A₂) ∧
  (¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂ S)) :=
by
  sorry

end mutually_exclusive_iff_complementary_l1281_128126


namespace algebraic_expression_l1281_128178

def a (x : ℕ) := 2005 * x + 2009
def b (x : ℕ) := 2005 * x + 2010
def c (x : ℕ) := 2005 * x + 2011

theorem algebraic_expression (x : ℕ) : 
  a x ^ 2 + b x ^ 2 + c x ^ 2 - a x * b x - b x * c x - c x * a x = 3 :=
by
  sorry

end algebraic_expression_l1281_128178


namespace smallest_sum_of_squares_l1281_128114

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l1281_128114


namespace no_solution_to_a_l1281_128153

theorem no_solution_to_a (x : ℝ) :
  (4 * x - 1) / 6 - (5 * x - 2 / 3) / 10 + (9 - x / 2) / 3 ≠ 101 / 20 := 
sorry

end no_solution_to_a_l1281_128153


namespace cos_30_eq_sqrt3_div_2_l1281_128159

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l1281_128159


namespace union_A_B_l1281_128197

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 > 1}

-- Prove the union of A and B is the expected result
theorem union_A_B : A ∪ B = {x | x ≤ 0 ∨ x > 1} :=
by
  sorry

end union_A_B_l1281_128197


namespace mango_rate_is_50_l1281_128129

theorem mango_rate_is_50 (quantity_grapes kg_grapes_perkg quantity_mangoes total_paid cost_grapes cost_mangoes rate_mangoes : ℕ) 
  (h1 : quantity_grapes = 8) 
  (h2 : kg_grapes_perkg = 70) 
  (h3 : quantity_mangoes = 9) 
  (h4 : total_paid = 1010)
  (h5 : cost_grapes = quantity_grapes * kg_grapes_perkg)
  (h6 : cost_mangoes = total_paid - cost_grapes)
  (h7 : rate_mangoes = cost_mangoes / quantity_mangoes) : 
  rate_mangoes = 50 :=
by sorry

end mango_rate_is_50_l1281_128129


namespace parabola_vertex_range_l1281_128105

def parabola_vertex_in_first_quadrant (m : ℝ) : Prop :=
  ∃ v : ℝ × ℝ, v = (m, m - 1) ∧ 0 < m ∧ 0 < (m - 1)

theorem parabola_vertex_range (m : ℝ) (h_vertex : parabola_vertex_in_first_quadrant m) :
  1 < m :=
by
  sorry

end parabola_vertex_range_l1281_128105


namespace shortest_remaining_side_l1281_128158

theorem shortest_remaining_side (a b c : ℝ) (h₁ : a = 5) (h₂ : c = 13) (h₃ : a^2 + b^2 = c^2) : b = 12 :=
by
  rw [h₁, h₂] at h₃
  sorry

end shortest_remaining_side_l1281_128158


namespace point_A_in_fourth_quadrant_l1281_128183

def Point := ℤ × ℤ

def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def point_A : Point := (3, -2)
def point_B : Point := (2, 5)
def point_C : Point := (-1, -2)
def point_D : Point := (-2, 2)

theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A :=
  sorry

end point_A_in_fourth_quadrant_l1281_128183


namespace time_to_finish_all_problems_l1281_128127

def mathProblems : ℝ := 17.0
def spellingProblems : ℝ := 15.0
def problemsPerHour : ℝ := 8.0
def totalProblems : ℝ := mathProblems + spellingProblems

theorem time_to_finish_all_problems : totalProblems / problemsPerHour = 4.0 :=
by
  sorry

end time_to_finish_all_problems_l1281_128127


namespace current_age_of_son_l1281_128164

variables (S F : ℕ)

-- Define the conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F - 8 = 4 * (S - 8)

-- The theorem statement
theorem current_age_of_son (h1 : condition1 S F) (h2 : condition2 S F) : S = 24 :=
sorry

end current_age_of_son_l1281_128164


namespace seed_germination_probability_l1281_128140

-- Define necessary values and variables
def n : ℕ := 3
def p : ℚ := 0.7
def k : ℕ := 2

-- Define the binomial probability formula
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- State the proof problem
theorem seed_germination_probability :
  binomial_probability n k p = 0.441 := 
sorry

end seed_germination_probability_l1281_128140


namespace inverse_proportion_l1281_128195

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k)
  (h2 : 6^2 * 2^4 = k) (hy : y = 4) : x^2 = 2.25 :=
by
  sorry

end inverse_proportion_l1281_128195


namespace find_number_l1281_128169

-- Define the conditions and the theorem
theorem find_number (number : ℝ)
  (h₁ : ∃ w : ℝ, w = (69.28 * number) / 0.03 ∧ abs (w - 9.237333333333334) ≤ 1e-10) :
  abs (number - 0.004) ≤ 1e-10 :=
by
  sorry

end find_number_l1281_128169


namespace neg_prop_p_equiv_l1281_128180

variable {x : ℝ}

def prop_p : Prop := ∃ x ≥ 0, 2^x = 3

theorem neg_prop_p_equiv : ¬prop_p ↔ ∀ x ≥ 0, 2^x ≠ 3 :=
by sorry

end neg_prop_p_equiv_l1281_128180


namespace no_term_un_eq_neg1_l1281_128187

theorem no_term_un_eq_neg1 (p : ℕ) [hp_prime: Fact (Nat.Prime p)] (hp_odd: p % 2 = 1) (hp_not_five: p ≠ 5) :
  ∀ n : ℕ, ∀ u : ℕ → ℤ, ((u 0 = 0) ∧ (u 1 = 1) ∧ (∀ k, k ≥ 2 → u (k-2) = 2 * u (k-1) - p * u k)) → 
    (u n ≠ -1) :=
  sorry

end no_term_un_eq_neg1_l1281_128187


namespace prove_monomial_l1281_128156

-- Definitions and conditions from step a)
def like_terms (x y : ℕ) := 
  x = 2 ∧ x + y = 5

-- Main statement to be proved
theorem prove_monomial (x y : ℕ) (h : like_terms x y) : 
  1 / 2 * x^3 - 1 / 6 * x * y^2 = 1 :=
by
  sorry

end prove_monomial_l1281_128156


namespace volleyball_team_selection_l1281_128135

noncomputable def volleyball_squad_count (n m k : ℕ) : ℕ :=
  n * (Nat.choose m k)

theorem volleyball_team_selection :
  volleyball_squad_count 12 11 7 = 3960 :=
by
  sorry

end volleyball_team_selection_l1281_128135


namespace solve_for_x_l1281_128162

namespace proof_problem

-- Define the operation a * b = 4 * a * b
def star (a b : ℝ) : ℝ := 4 * a * b

-- Given condition rewritten in terms of the operation star
def equation (x : ℝ) : Prop := star x x + star 2 x - star 2 4 = 0

-- The statement we intend to prove
theorem solve_for_x (x : ℝ) : equation x → (x = 2 ∨ x = -4) :=
by
  -- Proof omitted
  sorry

end proof_problem

end solve_for_x_l1281_128162


namespace sum_base6_l1281_128184

theorem sum_base6 : 
  ∀ (a b : ℕ) (h1 : a = 4532) (h2 : b = 3412),
  (a + b = 10414) :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end sum_base6_l1281_128184


namespace percentage_of_profit_if_no_discount_l1281_128191

-- Conditions
def discount : ℝ := 0.05
def profit_w_discount : ℝ := 0.216
def cost_price : ℝ := 100
def expected_profit : ℝ := 28

-- Proof statement
theorem percentage_of_profit_if_no_discount :
  ∃ (marked_price selling_price_no_discount : ℝ),
    selling_price_no_discount = marked_price ∧
    (marked_price - cost_price) / cost_price * 100 = expected_profit :=
by
  -- Definitions and logic will go here
  sorry

end percentage_of_profit_if_no_discount_l1281_128191


namespace proof_correct_word_choice_l1281_128163

def sentence_completion_correct (word : String) : Prop :=
  "Most of them are kind, but " ++ word ++ " is so good to me as Bruce" = "Most of them are kind, but none is so good to me as Bruce"

theorem proof_correct_word_choice : 
  (sentence_completion_correct "none") → 
  ("none" = "none") := 
by
  sorry

end proof_correct_word_choice_l1281_128163


namespace additional_life_vests_needed_l1281_128194

def num_students : ℕ := 40
def num_instructors : ℕ := 10
def life_vests_on_hand : ℕ := 20
def percent_students_with_vests : ℕ := 20

def total_people : ℕ := num_students + num_instructors
def students_with_vests : ℕ := (percent_students_with_vests * num_students) / 100
def total_vests_available : ℕ := life_vests_on_hand + students_with_vests

theorem additional_life_vests_needed : 
  total_people - total_vests_available = 22 :=
by 
  sorry

end additional_life_vests_needed_l1281_128194


namespace coin_toss_5_times_same_side_l1281_128154

noncomputable def probability_of_same_side (n : ℕ) : ℝ :=
  (1 / 2) ^ n

theorem coin_toss_5_times_same_side :
  probability_of_same_side 5 = 1 / 32 :=
by 
  -- The goal is to prove (1/2)^5 = 1/32
  sorry

end coin_toss_5_times_same_side_l1281_128154


namespace locus_C2_angle_measure_90_l1281_128155

variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)

-- Conditions for Question 1
def ellipse_C1 (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

variable (x0 y0 x1 y1 : ℝ)
variable (hA : ellipse_C1 a b x0 y0)
variable (hE : ellipse_C1 a b x1 y1)
variable (h_perpendicular : x1 * x0 + y1 * y0 = 0)

theorem locus_C2 :
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  x ≠ 0 → y ≠ 0 → 
  (x^2 / a^2 + y^2 / b^2 = (a^2 - b^2)^2 / (a^2 + b^2)^2) := 
sorry

-- Conditions for Question 2
def circle_C3 (x y : ℝ) : Prop := 
  x^2 + y^2 = 1

theorem angle_measure_90 :
  (a^2 + b^2)^3 = a^2 * b^2 * (a^2 - b^2)^2 → 
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  circle_C3 x y → 
  (∃ (theta : ℝ), θ = 90) := 
sorry

end locus_C2_angle_measure_90_l1281_128155


namespace minimum_red_points_for_square_l1281_128188

/-- Given a circle divided into 100 equal segments with points randomly colored red. 
Prove that the minimum number of red points needed to ensure at least four red points 
form the vertices of a square is 76. --/
theorem minimum_red_points_for_square (n : ℕ) (h : n = 100) (red_points : Finset ℕ)
  (hred : red_points.card ≥ 76) (hseg : ∀ i j : ℕ, i ≤ j → (j - i) % 25 ≠ 0 → ¬ (∃ a b c d : ℕ, 
  a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0)) : 
  ∃ a b c d : ℕ, a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0 :=
sorry

end minimum_red_points_for_square_l1281_128188


namespace compute_A_3_2_l1281_128145

namespace Ackermann

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem compute_A_3_2 : A 3 2 = 12 :=
sorry

end Ackermann

end compute_A_3_2_l1281_128145


namespace tutors_meet_again_l1281_128131

theorem tutors_meet_again (tim uma victor xavier: ℕ) (h1: tim = 5) (h2: uma = 6) (h3: victor = 9) (h4: xavier = 8) :
  Nat.lcm (Nat.lcm tim uma) (Nat.lcm victor xavier) = 360 := 
by 
  rw [h1, h2, h3, h4]
  show Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 8) = 360
  sorry

end tutors_meet_again_l1281_128131


namespace greatest_b_for_no_real_roots_l1281_128123

theorem greatest_b_for_no_real_roots :
  ∀ (b : ℤ), (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) ↔ b ≤ 6 := sorry

end greatest_b_for_no_real_roots_l1281_128123


namespace inverse_proportion_l1281_128124

theorem inverse_proportion (a : ℝ) (b : ℝ) (k : ℝ) : 
  (a = k / b^2) → 
  (40 = k / 12^2) → 
  (a = 10) → 
  b = 24 := 
by
  sorry

end inverse_proportion_l1281_128124


namespace intersection_S_T_l1281_128104

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4 * x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} :=
by
  sorry

end intersection_S_T_l1281_128104


namespace minimum_value_of_f_l1281_128175

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 13) / (6 * (1 + Real.exp (-x)))

theorem minimum_value_of_f : ∀ x : ℝ, 0 ≤ x → f x ≥ f 0 :=
by
  intro x hx
  unfold f
  admit

end minimum_value_of_f_l1281_128175


namespace number_of_cows_l1281_128106

theorem number_of_cows (x y : ℕ) 
  (h1 : 4 * x + 2 * y = 14 + 2 * (x + y)) : 
  x = 7 :=
by
  sorry

end number_of_cows_l1281_128106


namespace five_crows_two_hours_l1281_128139

-- Define the conditions and the question as hypotheses
def crows_worms (crows worms hours : ℕ) := 
  (crows = 3) ∧ (worms = 30) ∧ (hours = 1)

theorem five_crows_two_hours 
  (c: ℕ) (w: ℕ) (h: ℕ)
  (H: crows_worms c w h)
  : ∃ worms_eaten : ℕ, worms_eaten = 100 :=
by
  sorry

end five_crows_two_hours_l1281_128139


namespace solve_system_l1281_128101

theorem solve_system (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : 2 * b - 3 * a = 4) : b = 2 :=
by {
  -- Given the conditions, we need to show that b = 2
  sorry
}

end solve_system_l1281_128101


namespace angle_B_is_pi_over_3_range_of_expression_l1281_128190

variable {A B C a b c : ℝ}

-- Conditions
def sides_opposite_angles (A B C : ℝ) (a b c : ℝ): Prop :=
  (2 * c - a) * Real.cos B - b * Real.cos A = 0

-- Part 1: Prove B = π/3
theorem angle_B_is_pi_over_3 (h : sides_opposite_angles A B C a b c) : 
    B = Real.pi / 3 := 
  sorry

-- Part 2: Prove the range of sqrt(3) * (sin A + sin(C - π/6)) is (1, 2]
theorem range_of_expression (h : 0 < A ∧ A < 2 * Real.pi / 3) : 
    (1:ℝ) < Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) 
    ∧ Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) ≤ 2 := 
  sorry

end angle_B_is_pi_over_3_range_of_expression_l1281_128190


namespace Mike_watches_TV_every_day_l1281_128149

theorem Mike_watches_TV_every_day :
  (∃ T : ℝ, 
  (3 * (T / 2) + 7 * T = 34) 
  → T = 4) :=
by
  let T := 4
  sorry

end Mike_watches_TV_every_day_l1281_128149


namespace even_number_representation_l1281_128116

-- Definitions for conditions
def even_number (k : Int) : Prop := ∃ m : Int, k = 2 * m
def perfect_square (n : Int) : Prop := ∃ p : Int, n = p * p
def sum_representation (a b : Int) : Prop := ∃ k : Int, a + b = 2 * k ∧ perfect_square (a * b)
def difference_representation (d k e : Int) : Prop := d * (d - 2 * k) = e * e

-- The theorem statement
theorem even_number_representation {k : Int} (hk : even_number k) :
  (∃ a b : Int, sum_representation a b ∧ 2 * k = a + b) ∨
  (∃ d e : Int, difference_representation d k e ∧ d ≠ 0) :=
sorry

end even_number_representation_l1281_128116


namespace actual_price_of_food_l1281_128138

theorem actual_price_of_food (P : ℝ) (h : 1.32 * P = 132) : P = 100 := 
by
  sorry

end actual_price_of_food_l1281_128138


namespace john_sells_20_woodburnings_l1281_128185

variable (x : ℕ)

theorem john_sells_20_woodburnings (price_per_woodburning cost profit : ℤ) 
  (h1 : price_per_woodburning = 15) (h2 : cost = 100) (h3 : profit = 200) :
  (profit = price_per_woodburning * x - cost) → 
  x = 20 :=
by
  intros h_profit
  rw [h1, h2, h3] at h_profit
  linarith

end john_sells_20_woodburnings_l1281_128185


namespace problem_statement_l1281_128112

noncomputable def f (x k : ℝ) := x^3 / (2^x + k * 2^(-x))

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def k2_eq_1_is_nec_but_not_suff (f : ℝ → ℝ) (k : ℝ) : Prop :=
  (k^2 = 1) → (is_even_function f → k = -1 ∧ ¬(k = 1))

theorem problem_statement (k : ℝ) :
  k2_eq_1_is_nec_but_not_suff (λ x => f x k) k :=
by
  sorry

end problem_statement_l1281_128112


namespace find_a_l1281_128144

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (5, 1)
noncomputable def C (a : ℝ) : ℝ × ℝ := (-4, 2 * a)

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_a (a : ℝ) : collinear (A a) B (C a) ↔ a = 4 :=
by
  sorry

end find_a_l1281_128144


namespace twenty_five_percent_of_five_hundred_is_one_twenty_five_l1281_128167

theorem twenty_five_percent_of_five_hundred_is_one_twenty_five :
  let percent := 0.25
  let amount := 500
  percent * amount = 125 :=
by
  sorry

end twenty_five_percent_of_five_hundred_is_one_twenty_five_l1281_128167


namespace angles_supplementary_l1281_128170

theorem angles_supplementary (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : ∃ S : Finset ℕ, S.card = 17 ∧ (∀ a ∈ S, ∃ k : ℕ, k * (180 / (k + 1)) = a ∧ A = a) :=
by
  sorry

end angles_supplementary_l1281_128170


namespace shaded_area_l1281_128103

theorem shaded_area (R : ℝ) (π : ℝ) (h1 : π * (R / 2)^2 * 2 = 1) : 
  (π * R^2 - (π * (R / 2)^2 * 2)) = 1 := 
by
  sorry

end shaded_area_l1281_128103


namespace relationship_between_p_and_q_l1281_128109

theorem relationship_between_p_and_q (p q : ℝ) 
  (h : ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (2*x)^2 + p*(2*x) + q = 0) :
  2 * p^2 = 9 * q :=
sorry

end relationship_between_p_and_q_l1281_128109


namespace income_calculation_l1281_128119

theorem income_calculation (savings : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (ratio_condition : income_ratio = 5 ∧ expenditure_ratio = 4) (savings_condition : savings = 3800) :
  income_ratio * savings / (income_ratio - expenditure_ratio) = 19000 :=
by
  sorry

end income_calculation_l1281_128119


namespace system_of_equations_solution_system_of_inequalities_solution_l1281_128182

-- Problem (1): Solve the system of equations
theorem system_of_equations_solution :
  ∃ (x y : ℝ), (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) ∧ (x = 7) ∧ (y = 4) :=
by
  sorry

-- Problem (2): Solve the system of linear inequalities
theorem system_of_inequalities_solution :
  ∃ (x : ℝ), (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * (x + 1)) / 3) ∧ (-3 < x) ∧ (x ≤ 3) :=
by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l1281_128182


namespace area_of_quadrilateral_l1281_128107

theorem area_of_quadrilateral 
  (area_ΔBDF : ℝ) (area_ΔBFE : ℝ) (area_ΔEFC : ℝ) (area_ΔCDF : ℝ) (h₁ : area_ΔBDF = 5)
  (h₂ : area_ΔBFE = 10) (h₃ : area_ΔEFC = 10) (h₄ : area_ΔCDF = 15) :
  (80 - (area_ΔBDF + area_ΔBFE + area_ΔEFC + area_ΔCDF)) = 40 := 
  by sorry

end area_of_quadrilateral_l1281_128107


namespace problem1_problem2_problem3_problem4_l1281_128141

-- statement for problem 1
theorem problem1 : -5 + 8 - 2 = 1 := by
  sorry

-- statement for problem 2
theorem problem2 : (-3) * (5/6) / (-1/4) = 10 := by
  sorry

-- statement for problem 3
theorem problem3 : -3/17 + (-3.75) + (-14/17) + (15/4) = -1 := by
  sorry

-- statement for problem 4
theorem problem4 : -(1^10) - ((13/14) - (11/12)) * (4 - (-2)^2) + (1/2) / 3 = -(5/6) := by
  sorry

end problem1_problem2_problem3_problem4_l1281_128141


namespace age_ratio_3_2_l1281_128196

/-
Define variables: 
  L : ℕ -- Liam's current age
  M : ℕ -- Mia's current age
  y : ℕ -- number of years until the age ratio is 3:2
-/

theorem age_ratio_3_2 (L M : ℕ) 
  (h1 : L - 4 = 2 * (M - 4)) 
  (h2 : L - 10 = 3 * (M - 10)) 
  (h3 : ∃ y, (L + y) * 2 = (M + y) * 3) : 
  ∃ y, y = 8 :=
by
  sorry

end age_ratio_3_2_l1281_128196


namespace solve_equations_l1281_128125

theorem solve_equations (x y : ℝ) (h1 : (x + y) / x = y / (x + y)) (h2 : x = 2 * y) :
  x = 0 ∧ y = 0 :=
by
  sorry

end solve_equations_l1281_128125


namespace algebra_expression_value_l1281_128171

theorem algebra_expression_value (m : ℝ) (hm : m^2 - m - 1 = 0) : m^2 - m + 2008 = 2009 :=
by
  sorry

end algebra_expression_value_l1281_128171


namespace second_discarded_number_l1281_128157

theorem second_discarded_number (S : ℝ) (X : ℝ) (h1 : S / 50 = 62) (h2 : (S - 45 - X) / 48 = 62.5) : X = 55 := 
by
  sorry

end second_discarded_number_l1281_128157


namespace calculate_selling_price_l1281_128186

noncomputable def originalPrice : ℝ := 120
noncomputable def firstDiscountRate : ℝ := 0.30
noncomputable def secondDiscountRate : ℝ := 0.15
noncomputable def taxRate : ℝ := 0.08

def discountedPrice1 (originalPrice firstDiscountRate : ℝ) : ℝ :=
  originalPrice * (1 - firstDiscountRate)

def discountedPrice2 (discountedPrice1 secondDiscountRate : ℝ) : ℝ :=
  discountedPrice1 * (1 - secondDiscountRate)

def finalPrice (discountedPrice2 taxRate : ℝ) : ℝ :=
  discountedPrice2 * (1 + taxRate)

theorem calculate_selling_price : 
  finalPrice (discountedPrice2 (discountedPrice1 originalPrice firstDiscountRate) secondDiscountRate) taxRate = 77.112 := 
sorry

end calculate_selling_price_l1281_128186


namespace students_in_class_l1281_128118

theorem students_in_class (total_pencils : ℕ) (pencils_per_student : ℕ) (n: ℕ) 
    (h1 : total_pencils = 18) 
    (h2 : pencils_per_student = 9) 
    (h3 : total_pencils = n * pencils_per_student) : 
    n = 2 :=
by 
  sorry

end students_in_class_l1281_128118


namespace ceil_sub_self_eq_half_l1281_128174

theorem ceil_sub_self_eq_half (n : ℤ) (x : ℝ) (h : x = n + 1/2) : ⌈x⌉ - x = 1/2 :=
by
  sorry

end ceil_sub_self_eq_half_l1281_128174


namespace find_k_inverse_proportion_l1281_128100

theorem find_k_inverse_proportion :
  ∃ k : ℝ, (∀ x y : ℝ, y = (k + 1) / x → (x = 1 ∧ y = -2) → k = -3) :=
by
  sorry

end find_k_inverse_proportion_l1281_128100


namespace triangle_inequality_theorem_l1281_128110

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality_theorem :
  ¬ is_triangle 2 3 5 ∧ is_triangle 5 6 10 ∧ ¬ is_triangle 1 1 3 ∧ ¬ is_triangle 3 4 9 :=
by {
  -- Proof goes here
  sorry
}

end triangle_inequality_theorem_l1281_128110


namespace smallest_n_integer_price_l1281_128132

theorem smallest_n_integer_price (p : ℚ) (h : ∃ x : ℕ, p = x ∧ 1.06 * p = n) : n = 53 :=
sorry

end smallest_n_integer_price_l1281_128132
