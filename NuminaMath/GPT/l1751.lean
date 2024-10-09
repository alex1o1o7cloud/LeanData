import Mathlib

namespace inequality_holds_l1751_175149

theorem inequality_holds (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 :=
by {
  sorry
}

end inequality_holds_l1751_175149


namespace weather_station_accuracy_l1751_175135

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k)

theorem weather_station_accuracy :
  binomial_probability 3 2 0.9 = 0.243 :=
by
  sorry

end weather_station_accuracy_l1751_175135


namespace cubic_geometric_progression_l1751_175199

theorem cubic_geometric_progression (a b c : ℝ) (α β γ : ℝ) 
    (h_eq1 : α + β + γ = -a) 
    (h_eq2 : α * β + α * γ + β * γ = b) 
    (h_eq3 : α * β * γ = -c) 
    (h_gp : ∃ k q : ℝ, α = k / q ∧ β = k ∧ γ = k * q) : 
    a^3 * c - b^3 = 0 :=
by
  sorry

end cubic_geometric_progression_l1751_175199


namespace problem1_l1751_175100

theorem problem1 :
  (Real.sqrt (3/2)) * (Real.sqrt (21/4)) / (Real.sqrt (7/2)) = 3/2 :=
sorry

end problem1_l1751_175100


namespace product_of_two_numbers_l1751_175151

-- Define HCF (Highest Common Factor) and LCM (Least Common Multiple) conditions
def hcf_of_two_numbers (a b : ℕ) : ℕ := 11
def lcm_of_two_numbers (a b : ℕ) : ℕ := 181

-- The theorem to prove
theorem product_of_two_numbers (a b : ℕ) 
  (h1 : hcf_of_two_numbers a b = 11)
  (h2 : lcm_of_two_numbers a b = 181) : 
  a * b = 1991 :=
by 
  -- This is where we would put the proof, but we can use sorry for now
  sorry

end product_of_two_numbers_l1751_175151


namespace even_heads_probability_is_17_over_25_l1751_175183

-- Definition of the probabilities of heads and tails
def prob_tails : ℚ := 1 / 5
def prob_heads : ℚ := 4 * prob_tails

-- Definition of the probability of getting an even number of heads in two flips
def even_heads_prob (p_heads p_tails : ℚ) : ℚ :=
  p_tails * p_tails + p_heads * p_heads

-- Theorem statement
theorem even_heads_probability_is_17_over_25 :
  even_heads_prob prob_heads prob_tails = 17 / 25 := by
  sorry

end even_heads_probability_is_17_over_25_l1751_175183


namespace min_value_sequence_l1751_175102

theorem min_value_sequence (a : ℕ → ℕ) (h1 : a 2 = 102) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) - a n = 4 * n) : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (a m) / m ≥ 26) :=
sorry

end min_value_sequence_l1751_175102


namespace positive_value_m_l1751_175115

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l1751_175115


namespace ratio_of_p_q_l1751_175120

theorem ratio_of_p_q (b : ℝ) (p q : ℝ) (h1 : p = -b / 8) (h2 : q = -b / 12) : p / q = 3 / 2 := 
by
  sorry

end ratio_of_p_q_l1751_175120


namespace no_solution_for_inequalities_l1751_175133

theorem no_solution_for_inequalities (x : ℝ) : ¬(4 * x ^ 2 + 7 * x - 2 < 0 ∧ 3 * x - 1 > 0) :=
by
  sorry

end no_solution_for_inequalities_l1751_175133


namespace cost_of_adult_ticket_l1751_175132

theorem cost_of_adult_ticket (x : ℕ) (total_persons : ℕ) (total_collected : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) (amount_from_children : ℕ) :
  total_persons = 280 →
  total_collected = 14000 →
  adult_tickets = 200 →
  child_ticket_cost = 25 →
  amount_from_children = 2000 →
  200 * x + amount_from_children = total_collected →
  x = 60 :=
by
  intros h_persons h_total h_adults h_child_cost h_children_amount h_eq
  sorry

end cost_of_adult_ticket_l1751_175132


namespace max_value_expression_l1751_175189

theorem max_value_expression  
    (x y : ℝ) 
    (h : 2 * x^2 + y^2 = 6 * x) : 
    x^2 + y^2 + 2 * x ≤ 15 :=
sorry

end max_value_expression_l1751_175189


namespace cubic_roots_nature_l1751_175150

-- Define the cubic polynomial function
def cubic_poly (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x - 4

-- Define the statement about the roots of the polynomial
theorem cubic_roots_nature :
  ∃ a b c : ℝ, cubic_poly a = 0 ∧ cubic_poly b = 0 ∧ cubic_poly c = 0 
  ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end cubic_roots_nature_l1751_175150


namespace fifth_hexagon_dots_l1751_175159

-- Definitions as per conditions
def dots_in_nth_layer (n : ℕ) : ℕ := 6 * (n + 2)

-- Function to calculate the total number of dots in the nth hexagon
def total_dots_in_hexagon (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + dots_in_nth_layer k) (dots_in_nth_layer 0)

-- The proof problem statement
theorem fifth_hexagon_dots : total_dots_in_hexagon 5 = 150 := sorry

end fifth_hexagon_dots_l1751_175159


namespace y_work_days_24_l1751_175155

-- Definitions of the conditions
def x_work_days := 36
def y_work_days (d : ℕ) := d
def y_worked_days := 12
def x_remaining_work_days := 18

-- Statement of the theorem
theorem y_work_days_24 : ∃ d : ℕ, (y_worked_days / y_work_days d + x_remaining_work_days / x_work_days = 1) ∧ d = 24 :=
  sorry

end y_work_days_24_l1751_175155


namespace emma_chocolates_l1751_175126

theorem emma_chocolates 
  (x : ℕ) 
  (h1 : ∃ l : ℕ, x = l + 10) 
  (h2 : ∃ l : ℕ, l = x / 3) : 
  x = 15 := 
  sorry

end emma_chocolates_l1751_175126


namespace school_spent_total_l1751_175139

noncomputable def seminar_fee (num_teachers : ℕ) : ℝ :=
  let base_fee := 150 * num_teachers
  if num_teachers >= 20 then
    base_fee * 0.925
  else if num_teachers >= 10 then
    base_fee * 0.95
  else
    base_fee

noncomputable def seminar_fee_with_tax (num_teachers : ℕ) : ℝ :=
  let fee := seminar_fee num_teachers
  fee * 1.06

noncomputable def food_allowance (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  let num_regular := num_teachers - num_special
  num_regular * 10 + num_special * 15

noncomputable def total_cost (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  seminar_fee_with_tax num_teachers + food_allowance num_teachers num_special

theorem school_spent_total (num_teachers num_special : ℕ) (h : num_teachers = 22 ∧ num_special = 3) :
  total_cost num_teachers num_special = 3470.65 :=
by
  sorry

end school_spent_total_l1751_175139


namespace range_of_a_iff_l1751_175137

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x → (Real.log x / Real.log a) ≤ x ∧ x ≤ a ^ x

theorem range_of_a_iff (a : ℝ) : (a ≥ Real.exp (Real.exp (-1))) ↔ range_of_a a :=
by
  sorry

end range_of_a_iff_l1751_175137


namespace num_ways_to_distribute_balls_l1751_175178

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end num_ways_to_distribute_balls_l1751_175178


namespace value_of_g_g_2_l1751_175192

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end value_of_g_g_2_l1751_175192


namespace number_of_guests_l1751_175179

def cook_per_minute : ℕ := 10
def time_to_cook : ℕ := 80
def guests_ate_per_guest : ℕ := 5
def guests_to_serve : ℕ := 20 -- This is what we'll prove.

theorem number_of_guests 
    (cook_per_8min : cook_per_minute = 10)
    (total_time : time_to_cook = 80)
    (eat_rate : guests_ate_per_guest = 5) :
    (time_to_cook * cook_per_minute) / guests_ate_per_guest = guests_to_serve := 
by 
  sorry

end number_of_guests_l1751_175179


namespace sale_price_after_discounts_l1751_175171

def calculate_sale_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

theorem sale_price_after_discounts :
  calculate_sale_price 500 [0.10, 0.15, 0.20, 0.25, 0.30] = 160.65 :=
by
  sorry

end sale_price_after_discounts_l1751_175171


namespace smallest_number_to_add_l1751_175127

theorem smallest_number_to_add:
  ∃ x : ℕ, x = 119 ∧ (2714 + x) % 169 = 0 :=
by
  sorry

end smallest_number_to_add_l1751_175127


namespace marla_parent_teacher_night_time_l1751_175177

def errand_time := 110 -- total minutes on the errand
def driving_time_oneway := 20 -- minutes driving one way to school
def driving_time_return := 20 -- minutes driving one way back home

def total_driving_time := driving_time_oneway + driving_time_return

def time_at_parent_teacher_night := errand_time - total_driving_time

theorem marla_parent_teacher_night_time : time_at_parent_teacher_night = 70 :=
by
  -- Lean proof goes here
  sorry

end marla_parent_teacher_night_time_l1751_175177


namespace total_cost_of_toys_l1751_175170

-- Define the costs of the yoyo and the whistle
def cost_yoyo : Nat := 24
def cost_whistle : Nat := 14

-- Prove the total cost of the yoyo and the whistle is 38 cents
theorem total_cost_of_toys : cost_yoyo + cost_whistle = 38 := by
  sorry

end total_cost_of_toys_l1751_175170


namespace quadratic_complete_square_l1751_175182

theorem quadratic_complete_square : ∃ k : ℤ, ∀ x : ℤ, x^2 + 8*x + 22 = (x + 4)^2 + k :=
by
  use 6
  sorry

end quadratic_complete_square_l1751_175182


namespace expression_eq_neg_one_l1751_175172

theorem expression_eq_neg_one (a b y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : y ≠ a) (h4 : y ≠ -a) :
  ( ( (a + b) / (a + y) + y / (a - y) ) / ( (y + b) / (a + y) - a / (a - y) ) = -1 ) ↔ ( y = a - b ) := 
sorry

end expression_eq_neg_one_l1751_175172


namespace balls_into_boxes_l1751_175101

noncomputable def countDistributions : ℕ :=
  sorry

theorem balls_into_boxes :
  countDistributions = 8 :=
  sorry

end balls_into_boxes_l1751_175101


namespace students_dont_eat_lunch_l1751_175146

theorem students_dont_eat_lunch
  (total_students : ℕ)
  (students_in_cafeteria : ℕ)
  (students_bring_lunch : ℕ)
  (students_no_lunch : ℕ)
  (h1 : total_students = 60)
  (h2 : students_in_cafeteria = 10)
  (h3 : students_bring_lunch = 3 * students_in_cafeteria)
  (h4 : students_no_lunch = total_students - (students_in_cafeteria + students_bring_lunch)) :
  students_no_lunch = 20 :=
by
  sorry

end students_dont_eat_lunch_l1751_175146


namespace externally_tangent_circles_proof_l1751_175129

noncomputable def externally_tangent_circles (r r' : ℝ) (φ : ℝ) : Prop :=
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r')

theorem externally_tangent_circles_proof (r r' φ : ℝ) 
  (h1: r > 0) (h2: r' > 0) (h3: φ ≥ 0 ∧ φ ≤ π) : 
  externally_tangent_circles r r' φ :=
sorry

end externally_tangent_circles_proof_l1751_175129


namespace carrots_eaten_after_dinner_l1751_175144

def carrots_eaten_before_dinner : ℕ := 22
def total_carrots_eaten : ℕ := 37

theorem carrots_eaten_after_dinner : total_carrots_eaten - carrots_eaten_before_dinner = 15 := by
  sorry

end carrots_eaten_after_dinner_l1751_175144


namespace largest_percent_error_l1751_175180
noncomputable def max_percent_error (d : ℝ) (d_err : ℝ) (r_err : ℝ) : ℝ :=
  let d_min := d - d * d_err
  let d_max := d + d * d_err
  let r := d / 2
  let r_min := r - r * r_err
  let r_max := r + r * r_err
  let area_actual := Real.pi * r^2
  let area_d_min := Real.pi * (d_min / 2)^2
  let area_d_max := Real.pi * (d_max / 2)^2
  let area_r_min := Real.pi * r_min^2
  let area_r_max := Real.pi * r_max^2
  let error_d_min := (area_actual - area_d_min) / area_actual * 100
  let error_d_max := (area_d_max - area_actual) / area_actual * 100
  let error_r_min := (area_actual - area_r_min) / area_actual * 100
  let error_r_max := (area_r_max - area_actual) / area_actual * 100
  max (max error_d_min error_d_max) (max error_r_min error_r_max)

theorem largest_percent_error 
  (d : ℝ) (d_err : ℝ) (r_err : ℝ) 
  (h_d : d = 30) (h_d_err : d_err = 0.15) (h_r_err : r_err = 0.10) : 
  max_percent_error d d_err r_err = 31.57 := by
  sorry

end largest_percent_error_l1751_175180


namespace wire_weight_l1751_175157

theorem wire_weight (w : ℕ → ℕ) (h_proportional : ∀ (x y : ℕ), w (x + y) = w x + w y) : 
  (w 25 = 5) → w 75 = 15 :=
by
  intro h1
  sorry

end wire_weight_l1751_175157


namespace third_consecutive_even_integer_l1751_175194

theorem third_consecutive_even_integer (n : ℤ) (h : (n + 2) + (n + 6) = 156) : (n + 4) = 78 :=
sorry

end third_consecutive_even_integer_l1751_175194


namespace jared_annual_salary_l1751_175184

def monthly_salary_diploma_holder : ℕ := 4000
def factor_degree_to_diploma : ℕ := 3
def months_in_year : ℕ := 12

theorem jared_annual_salary :
  (factor_degree_to_diploma * monthly_salary_diploma_holder) * months_in_year = 144000 :=
by
  sorry

end jared_annual_salary_l1751_175184


namespace solution_set_of_inequality_l1751_175122

theorem solution_set_of_inequality :
  { x : ℝ | (x - 4) / (3 - 2*x) < 0 ∧ 3 - 2*x ≠ 0 } = { x : ℝ | x < 3 / 2 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l1751_175122


namespace rate_percent_simple_interest_l1751_175121

theorem rate_percent_simple_interest
  (SI P : ℚ) (T : ℕ) (R : ℚ) : SI = 160 → P = 800 → T = 4 → (P * R * T / 100 = SI) → R = 5 :=
  by
  intros hSI hP hT hFormula
  -- Assertion that R = 5 is correct based on the given conditions and formula
  sorry

end rate_percent_simple_interest_l1751_175121


namespace douglas_votes_in_county_D_l1751_175147

noncomputable def percent_votes_in_county_D (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
    (total_votes : ℝ) (percent_A percent_B percent_C percent_D total_percent : ℝ) : Prop :=
  (votes_A / (5 * x) = 0.70) ∧
  (votes_B / (3 * x) = 0.58) ∧
  (votes_C / (2 * x) = 0.50) ∧
  (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62 ∧
  (votes_D / (4 * x) = percent_D)

theorem douglas_votes_in_county_D 
  (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
  (total_votes : ℝ := 14 * x) 
  (percent_A percent_B percent_C total_percent percent_D : ℝ)
  (h1 : votes_A / (5 * x) = 0.70) 
  (h2 : votes_B / (3 * x) = 0.58) 
  (h3 : votes_C / (2 * x) = 0.50) 
  (h4 : (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62) : 
  percent_votes_in_county_D x votes_A votes_B votes_C votes_D total_votes percent_A percent_B percent_C 0.61 total_percent :=
by
  constructor
  exact h1
  constructor
  exact h2
  constructor
  exact h3
  constructor
  exact h4
  sorry

end douglas_votes_in_county_D_l1751_175147


namespace total_birds_and_storks_l1751_175136

theorem total_birds_and_storks
  (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ)
  (hb : initial_birds = 3) (hs : initial_storks = 4) (has : additional_storks = 6) :
  initial_birds + (initial_storks + additional_storks) = 13 :=
by
  sorry

end total_birds_and_storks_l1751_175136


namespace power_sum_l1751_175141

theorem power_sum : 1 ^ 2009 + (-1) ^ 2009 = 0 := 
by 
  sorry

end power_sum_l1751_175141


namespace plumber_charge_shower_l1751_175114

theorem plumber_charge_shower (S : ℝ) 
  (sink_cost : ℝ := 30) 
  (toilet_cost : ℝ := 50)
  (max_earning : ℝ := 250)
  (first_job_toilets : ℝ := 3) (first_job_sinks : ℝ := 3)
  (second_job_toilets : ℝ := 2) (second_job_sinks : ℝ := 5)
  (third_job_toilets : ℝ := 1) (third_job_showers : ℝ := 2) (third_job_sinks : ℝ := 3) :
  2 * S + 1 * toilet_cost + 3 * sink_cost ≤ max_earning → S ≤ 55 :=
by
  sorry

end plumber_charge_shower_l1751_175114


namespace cos_identity_proof_l1751_175152

noncomputable def cos_eq_half : Prop :=
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7)) = 1 / 2

theorem cos_identity_proof : cos_eq_half :=
  by sorry

end cos_identity_proof_l1751_175152


namespace natalia_crates_l1751_175130

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l1751_175130


namespace gcd_372_684_is_12_l1751_175156

theorem gcd_372_684_is_12 : gcd 372 684 = 12 := by
  sorry

end gcd_372_684_is_12_l1751_175156


namespace smallest_multiple_of_4_and_14_is_28_l1751_175197

theorem smallest_multiple_of_4_and_14_is_28 :
  ∃ (a : ℕ), a > 0 ∧ (4 ∣ a) ∧ (14 ∣ a) ∧ ∀ b : ℕ, b > 0 → (4 ∣ b) → (14 ∣ b) → a ≤ b := 
sorry

end smallest_multiple_of_4_and_14_is_28_l1751_175197


namespace triangle_eq_medians_incircle_l1751_175116

-- Define a triangle and the properties of medians and incircle
structure Triangle (α : Type) [Nonempty α] :=
(A B C : α)

def is_equilateral {α : Type} [Nonempty α] (T : Triangle α) : Prop :=
  ∃ (d : α → α → ℝ), d T.A T.B = d T.B T.C ∧ d T.B T.C = d T.C T.A

def medians_segments_equal {α : Type} [Nonempty α] (T : Triangle α) (incr_len : (α → α → ℝ)) : Prop :=
  ∀ (MA MB MC : α), incr_len MA MB = incr_len MB MC ∧ incr_len MB MC = incr_len MC MA

-- The main theorem statement
theorem triangle_eq_medians_incircle {α : Type} [Nonempty α] 
  (T : Triangle α) (incr_len : α → α → ℝ) 
  (h : medians_segments_equal T incr_len) : is_equilateral T :=
sorry

end triangle_eq_medians_incircle_l1751_175116


namespace vector_relation_condition_l1751_175164

variables {V : Type*} [AddCommGroup V] (OD OE OM DO EO MO : V)

-- Given condition
theorem vector_relation_condition (h : OD + OE = OM) :

-- Option B
(OM + DO = OE) ∧ 

-- Option C
(OM - OE = OD) ∧ 

-- Option D
(DO + EO = MO) :=
by {
  -- Sorry, to focus on statement only
  sorry
}

end vector_relation_condition_l1751_175164


namespace min_ratio_l1751_175138

theorem min_ratio (x y : ℕ) 
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (mean : (x + y) = 110) :
  x / y = 1 / 9 :=
  sorry

end min_ratio_l1751_175138


namespace ratio_of_mistakes_l1751_175134

theorem ratio_of_mistakes (riley_mistakes team_mistakes : ℕ) 
  (h_riley : riley_mistakes = 3) (h_team : team_mistakes = 17) : 
  (team_mistakes - riley_mistakes) / riley_mistakes = 14 / 3 := 
by 
  sorry

end ratio_of_mistakes_l1751_175134


namespace tan_addition_sin_cos_expression_l1751_175169

noncomputable def alpha : ℝ := sorry -- this is where alpha would be defined

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem tan_addition (alpha : ℝ) (h : Real.tan alpha = 2) : (Real.tan (alpha + Real.pi / 4) = -3) :=
by sorry

theorem sin_cos_expression (alpha : ℝ) (h : Real.tan alpha = 2) : 
  (Real.sin (2 * alpha) / (Real.sin (alpha) ^ 2 - Real.cos (2 * alpha) + 1) = 1 / 3) :=
by sorry

end tan_addition_sin_cos_expression_l1751_175169


namespace distance_between_parallel_lines_l1751_175154

theorem distance_between_parallel_lines (A B c1 c2 : Real) (hA : A = 2) (hB : B = 3) 
(hc1 : c1 = -3) (hc2 : c2 = 2) : 
    (abs (c1 - c2) / Real.sqrt (A^2 + B^2)) = (5 * Real.sqrt 13 / 13) := by
  sorry

end distance_between_parallel_lines_l1751_175154


namespace not_possible_to_list_numbers_l1751_175124

theorem not_possible_to_list_numbers :
  ¬ (∃ (f : ℕ → ℕ), (∀ n, f n ≥ 1 ∧ f n ≤ 1963) ∧
                     (∀ n, Nat.gcd (f n) (f (n+1)) = 1) ∧
                     (∀ n, Nat.gcd (f n) (f (n+2)) = 1)) :=
by
  sorry

end not_possible_to_list_numbers_l1751_175124


namespace sum_lent_out_l1751_175140

theorem sum_lent_out (P R : ℝ) (h1 : 720 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 600 := by
  sorry

end sum_lent_out_l1751_175140


namespace minimum_sugar_amount_l1751_175174

theorem minimum_sugar_amount (f s : ℕ) (h1 : f ≥ 9 + s / 2) (h2 : f ≤ 3 * s) : s ≥ 4 :=
by
  -- Provided conditions: f ≥ 9 + s / 2 and f ≤ 3 * s
  -- Goal: s ≥ 4
  sorry

end minimum_sugar_amount_l1751_175174


namespace original_number_l1751_175106

theorem original_number (x : ℝ) (hx : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (Real.sqrt 10) / 100 :=
by
  sorry

end original_number_l1751_175106


namespace mean_equivalence_l1751_175195

theorem mean_equivalence :
  (20 + 30 + 40) / 3 = (23 + 30 + 37) / 3 :=
by sorry

end mean_equivalence_l1751_175195


namespace remaining_pieces_to_fold_l1751_175176

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end remaining_pieces_to_fold_l1751_175176


namespace sector_area_l1751_175153

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 6) (h_α : α = π / 3) : (1 / 2) * (α * r) * r = 6 * π :=
by
  rw [h_r, h_α]
  sorry

end sector_area_l1751_175153


namespace cannot_achieve_55_cents_with_six_coins_l1751_175118

theorem cannot_achieve_55_cents_with_six_coins :
  ¬∃ (a b c d e : ℕ), 
    a + b + c + d + e = 6 ∧ 
    a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 55 := 
sorry

end cannot_achieve_55_cents_with_six_coins_l1751_175118


namespace driver_a_driven_more_distance_l1751_175128

-- Definitions based on conditions
def initial_distance : ℕ := 787
def speed_a : ℕ := 90
def speed_b : ℕ := 80
def start_difference : ℕ := 1

-- Statement of the problem
theorem driver_a_driven_more_distance :
  let distance_a := speed_a * (start_difference + (initial_distance - speed_a) / (speed_a + speed_b))
  let distance_b := speed_b * ((initial_distance - speed_a) / (speed_a + speed_b))
  distance_a - distance_b = 131 := by
sorry

end driver_a_driven_more_distance_l1751_175128


namespace minimum_price_to_cover_costs_l1751_175166

variable (P : ℝ)

-- Conditions
def prod_cost_A := 80
def ship_cost_A := 2
def prod_cost_B := 60
def ship_cost_B := 3
def fixed_costs := 16200
def units_A := 200
def units_B := 300

-- Cost calculations
def total_cost_A := units_A * prod_cost_A + units_A * ship_cost_A
def total_cost_B := units_B * prod_cost_B + units_B * ship_cost_B
def total_costs := total_cost_A + total_cost_B + fixed_costs

-- Revenue requirement
def revenue (P_A P_B : ℝ) := units_A * P_A + units_B * P_B

theorem minimum_price_to_cover_costs :
  (units_A + units_B) * P ≥ total_costs ↔ P ≥ 103 :=
sorry

end minimum_price_to_cover_costs_l1751_175166


namespace isosceles_triangle_large_angles_l1751_175109

theorem isosceles_triangle_large_angles (y : ℝ) (h : 2 * y + 40 = 180) : y = 70 :=
by
  sorry

end isosceles_triangle_large_angles_l1751_175109


namespace parking_lot_cars_l1751_175158

theorem parking_lot_cars :
  ∀ (initial_cars cars_left cars_entered remaining_cars final_cars : ℕ),
    initial_cars = 80 →
    cars_left = 13 →
    remaining_cars = initial_cars - cars_left →
    cars_entered = cars_left + 5 →
    final_cars = remaining_cars + cars_entered →
    final_cars = 85 := 
by
  intros initial_cars cars_left cars_entered remaining_cars final_cars h1 h2 h3 h4 h5
  sorry

end parking_lot_cars_l1751_175158


namespace each_regular_tire_distance_used_l1751_175193

-- Define the conditions of the problem
def total_distance_traveled : ℕ := 50000
def spare_tire_distance : ℕ := 2000
def regular_tires_count : ℕ := 4

-- Using these conditions, we will state the problem as a theorem
theorem each_regular_tire_distance_used : 
  (total_distance_traveled - spare_tire_distance) / regular_tires_count = 12000 :=
by
  sorry

end each_regular_tire_distance_used_l1751_175193


namespace rectangle_area_l1751_175160

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l1751_175160


namespace dodecahedron_interior_diagonals_l1751_175188

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l1751_175188


namespace first_year_with_sum_of_digits_10_after_2020_l1751_175107

theorem first_year_with_sum_of_digits_10_after_2020 :
  ∃ (y : ℕ), y > 2020 ∧ (y.digits 10).sum = 10 ∧ ∀ (z : ℕ), (z > 2020 ∧ (z.digits 10).sum = 10) → y ≤ z :=
sorry

end first_year_with_sum_of_digits_10_after_2020_l1751_175107


namespace num_solutions_l1751_175198

theorem num_solutions (h : ∀ n : ℕ, (1 ≤ n ∧ n ≤ 455) → n^3 % 455 = 1) : 
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ (1 ≤ n ∧ n ≤ 455) ∧ n^3 % 455 = 1) ∧ s.card = 9) :=
sorry

end num_solutions_l1751_175198


namespace sum_of_numbers_l1751_175173

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := sorry

end sum_of_numbers_l1751_175173


namespace find_second_sum_l1751_175145

theorem find_second_sum (S : ℝ) (x : ℝ) (h : S = 2704 ∧ 24 * x / 100 = 15 * (S - x) / 100) : (S - x) = 1664 := 
  sorry

end find_second_sum_l1751_175145


namespace tan_of_right_triangle_l1751_175148

theorem tan_of_right_triangle (A B C : ℝ) (h : A^2 + B^2 = C^2) (hA : A = 30) (hC : C = 37) : 
  (37^2 - 30^2).sqrt / 30 = (469).sqrt / 30 := by
  sorry

end tan_of_right_triangle_l1751_175148


namespace arithmetic_mean_of_pq_is_10_l1751_175103

variable (p q r : ℝ)

theorem arithmetic_mean_of_pq_is_10
  (H_mean_qr : (q + r) / 2 = 20)
  (H_r_minus_p : r - p = 20) :
  (p + q) / 2 = 10 := by
  sorry

end arithmetic_mean_of_pq_is_10_l1751_175103


namespace alex_cell_phone_cost_l1751_175105

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.1
def extra_min_cost_per_minute : ℝ := 0.15
def text_messages_sent : ℕ := 150
def hours_talked : ℝ := 32
def included_hours : ℝ := 25

theorem alex_cell_phone_cost : base_cost 
  + (text_messages_sent * text_cost_per_message)
  + ((hours_talked - included_hours) * 60 * extra_min_cost_per_minute) = 98 := by
  sorry

end alex_cell_phone_cost_l1751_175105


namespace sqrt_43_between_6_and_7_l1751_175167

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 := sorry

end sqrt_43_between_6_and_7_l1751_175167


namespace area_of_regular_inscribed_polygon_f3_properties_of_f_l1751_175143

noncomputable def f (n : ℕ) : ℝ :=
  if h : n ≥ 3 then (n / 2) * Real.sin (2 * Real.pi / n) else 0

theorem area_of_regular_inscribed_polygon_f3 :
  f 3 = (3 * Real.sqrt 3) / 4 :=
by
  sorry

theorem properties_of_f (n : ℕ) (hn : n ≥ 3) :
  (f n = (n / 2) * Real.sin (2 * Real.pi / n)) ∧
  (f n < f (n + 1)) ∧ 
  (f n < f (2 * n) ∧ f (2 * n) ≤ 2 * f n) :=
by
  sorry

end area_of_regular_inscribed_polygon_f3_properties_of_f_l1751_175143


namespace compute_fraction_sum_l1751_175161

theorem compute_fraction_sum :
  8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 :=
by
  sorry

end compute_fraction_sum_l1751_175161


namespace proof_problem_l1751_175185

variables (a b : ℝ)

noncomputable def expr := (2 * a⁻¹ + (a⁻¹ / b)) / a

theorem proof_problem (h1 : a = 1/3) (h2 : b = 3) : expr a b = 21 :=
by
  sorry

end proof_problem_l1751_175185


namespace ricardo_coin_difference_l1751_175108

theorem ricardo_coin_difference (p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ 2299) :
  (11500 - 4 * p) - (11500 - 4 * (2300 - p)) = 9192 :=
by
  sorry

end ricardo_coin_difference_l1751_175108


namespace inconsistent_coordinates_l1751_175113

theorem inconsistent_coordinates
  (m n : ℝ) 
  (h1 : m - (5/2)*n + 1 = 0) 
  (h2 : (m + 1/2) - (5/2)*(n + 1) + 1 = 0) :
  false :=
by
  sorry

end inconsistent_coordinates_l1751_175113


namespace gcd_values_count_l1751_175112

noncomputable def count_gcd_values (a b : ℕ) : ℕ :=
  if (a * b = 720 ∧ a + b = 50) then 1 else 0

theorem gcd_values_count : 
  (∃ a b : ℕ, a * b = 720 ∧ a + b = 50) → count_gcd_values a b = 1 :=
by
  sorry

end gcd_values_count_l1751_175112


namespace coeff_matrix_correct_l1751_175196

-- Define the system of linear equations as given conditions
def eq1 (x y : ℝ) : Prop := 2 * x + 3 * y = 1
def eq2 (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the coefficient matrix
def coeffMatrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 3],
  ![1, -2]
]

-- The theorem stating that the coefficient matrix of the system is as defined
theorem coeff_matrix_correct (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : 
  coeffMatrix = ![
    ![2, 3],
    ![1, -2]
  ] :=
sorry

end coeff_matrix_correct_l1751_175196


namespace point_third_quadrant_l1751_175142

theorem point_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : 3 * m - 2 < 0 ∧ -n < 0 :=
by
  sorry

end point_third_quadrant_l1751_175142


namespace sin_double_alpha_l1751_175162

theorem sin_double_alpha (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end sin_double_alpha_l1751_175162


namespace sector_radian_measure_l1751_175117

theorem sector_radian_measure {r l : ℝ} 
  (h1 : 2 * r + l = 12) 
  (h2 : (1/2) * l * r = 8) : 
  (l / r = 1) ∨ (l / r = 4) :=
sorry

end sector_radian_measure_l1751_175117


namespace quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l1751_175191

theorem quadratic_real_roots_discriminant (m : ℝ) :
  (2 * (m + 1))^2 - 4 * m * (m - 1) > 0 ↔ (m > -1/2 ∧ m ≠ 0) := 
sorry

theorem quadratic_real_roots_sum_of_squares (m x1 x2 : ℝ) 
  (h1 : m > -1/2 ∧ m ≠ 0)
  (h2 : x1 + x2 = -2 * (m + 1) / m)
  (h3 : x1 * x2 = (m - 1) / m)
  (h4 : x1^2 + x2^2 = 8) : 
  m = (6 + 2 * Real.sqrt 33) / 8 := 
sorry

end quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l1751_175191


namespace D_is_necessary_but_not_sufficient_condition_for_A_l1751_175125

variable (A B C D : Prop)

-- Conditions
axiom A_implies_B : A → B
axiom not_B_implies_A : ¬ (B → A)
axiom B_iff_C : B ↔ C
axiom C_implies_D : C → D
axiom not_D_implies_C : ¬ (D → C)

theorem D_is_necessary_but_not_sufficient_condition_for_A : (A → D) ∧ ¬ (D → A) :=
by sorry

end D_is_necessary_but_not_sufficient_condition_for_A_l1751_175125


namespace solve_for_y_l1751_175190

theorem solve_for_y : ∃ y : ℝ, y = -2 ∧ y^2 + 6 * y + 8 = -(y + 2) * (y + 6) :=
by
  use -2
  sorry

end solve_for_y_l1751_175190


namespace solve_for_b_l1751_175111

theorem solve_for_b 
  (b : ℝ)
  (h : (25 * b^2) - 84 = 0) :
  b = (2 * Real.sqrt 21) / 5 ∨ b = -(2 * Real.sqrt 21) / 5 :=
by sorry

end solve_for_b_l1751_175111


namespace dream_miles_driven_l1751_175163

theorem dream_miles_driven (x : ℕ) (h : 4 * x + 4 * (x + 200) = 4000) : x = 400 :=
by
  sorry

end dream_miles_driven_l1751_175163


namespace geometric_sequence_n_l1751_175119

theorem geometric_sequence_n (a1 an q : ℚ) (n : ℕ) (h1 : a1 = 9 / 8) (h2 : an = 1 / 3) (h3 : q = 2 / 3) : n = 4 :=
by
  sorry

end geometric_sequence_n_l1751_175119


namespace max_ab_condition_max_ab_value_l1751_175104

theorem max_ab_condition (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a + b = 1) (h2 : a = b) : ab = 1 / 4 :=
sorry

end max_ab_condition_max_ab_value_l1751_175104


namespace max_vector_sum_l1751_175186

theorem max_vector_sum
  (A B C : ℝ × ℝ)
  (P : ℝ × ℝ := (2, 0))
  (hA : A.1^2 + A.2^2 = 1)
  (hB : B.1^2 + B.2^2 = 1)
  (hC : C.1^2 + C.2^2 = 1)
  (h_perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  |(2,0) - A + (2,0) - B + (2,0) - C| = 7 := sorry

end max_vector_sum_l1751_175186


namespace inequality_solution_l1751_175123

open Real

theorem inequality_solution (a x : ℝ) :
  (a = 0 ∧ x > 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a = 1 ∧ ∀ x, ¬ (a * x^2 - (2 * a + 2) * x + 4 > 0)) ∨
  (a < 0 ∧ (x < 2/a ∨ x > 2) ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (0 < a ∧ a < 1 ∧ 2 < x ∧ x < 2/a ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a > 1 ∧ 2/a < x ∧ x < 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) := 
sorry

end inequality_solution_l1751_175123


namespace gcd_of_B_l1751_175175

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l1751_175175


namespace basketball_free_throws_l1751_175131

/-
Given the following conditions:
1. The players scored twice as many points with three-point shots as with two-point shots: \( 3b = 2a \).
2. The number of successful free throws was one more than the number of successful two-point shots: \( x = a + 1 \).
3. The team’s total score was 84 points: \( 2a + 3b + x = 84 \).

Prove that the number of free throws \( x \) equals 16.
-/
theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = a + 1) 
  (h3 : 2 * a + 3 * b + x = 84) : 
  x = 16 := 
  sorry

end basketball_free_throws_l1751_175131


namespace range_of_k_l1751_175165

theorem range_of_k (k : ℝ) (hₖ : 0 < k) :
  (∃ x : ℝ, 1 = x^2 + (k^2 / x^2)) → 0 < k ∧ k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l1751_175165


namespace evaluate_ratio_is_negative_two_l1751_175168

noncomputable def evaluate_ratio (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : ℂ :=
  (a^15 + b^15) / (a + b)^15

theorem evaluate_ratio_is_negative_two (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : 
  evaluate_ratio a b h = -2 := 
sorry

end evaluate_ratio_is_negative_two_l1751_175168


namespace sum_of_consecutive_even_integers_is_24_l1751_175181

theorem sum_of_consecutive_even_integers_is_24 (x : ℕ) (h_pos : x > 0)
    (h_eq : (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2))) :
    (x - 2) + x + (x + 2) = 24 :=
sorry

end sum_of_consecutive_even_integers_is_24_l1751_175181


namespace range_of_a_second_quadrant_l1751_175187

theorem range_of_a_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0) → x < 0 ∧ y > 0) →
  a > 2 :=
sorry

end range_of_a_second_quadrant_l1751_175187


namespace sum_of_divisors_45_l1751_175110

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun i => n % i = 0) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_45 : sum_of_divisors 45 = 78 := 
  sorry

end sum_of_divisors_45_l1751_175110
