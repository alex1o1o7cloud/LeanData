import Mathlib

namespace Kim_nail_polishes_l701_70128

-- Define the conditions
variable (K : ℕ)
def Heidi_nail_polishes (K : ℕ) : ℕ := K + 5
def Karen_nail_polishes (K : ℕ) : ℕ := K - 4

-- The main statement to prove
theorem Kim_nail_polishes (K : ℕ) (H : Heidi_nail_polishes K + Karen_nail_polishes K = 25) : K = 12 := by
  sorry

end Kim_nail_polishes_l701_70128


namespace average_age_of_town_l701_70127

-- Definitions based on conditions
def ratio_of_women_to_men (nw nm : ℕ) : Prop := nw * 8 = nm * 9

def young_men (nm : ℕ) (n_young_men : ℕ) (average_age_young : ℕ) : Prop :=
  n_young_men = 40 ∧ average_age_young = 25

def remaining_men_average_age (nm n_young_men : ℕ) (average_age_remaining : ℕ) : Prop :=
  average_age_remaining = 35

def women_average_age (average_age_women : ℕ) : Prop :=
  average_age_women = 30

-- Complete problem statement we need to prove
theorem average_age_of_town (nw nm : ℕ) (total_avg_age : ℕ) :
  ratio_of_women_to_men nw nm →
  young_men nm 40 25 →
  remaining_men_average_age nm 40 35 →
  women_average_age 30 →
  total_avg_age = 32 * 17 + 6 :=
sorry

end average_age_of_town_l701_70127


namespace find_n_from_remainders_l701_70112

theorem find_n_from_remainders (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := 
by 
  -- proof needed here
  sorry

end find_n_from_remainders_l701_70112


namespace equation1_solution_equation2_solution_l701_70181

theorem equation1_solution (x : ℝ) : (x - 1) ^ 3 = 64 ↔ x = 5 := sorry

theorem equation2_solution (x : ℝ) : 25 * x ^ 2 + 3 = 12 ↔ x = 3 / 5 ∨ x = -3 / 5 := sorry

end equation1_solution_equation2_solution_l701_70181


namespace no_solution_l701_70109

theorem no_solution (x : ℝ) (h₁ : x ≠ -1/3) (h₂ : x ≠ -4/5) :
  (2 * x - 4) / (3 * x + 1) ≠ (2 * x - 10) / (5 * x + 4) := 
sorry

end no_solution_l701_70109


namespace triangle_is_isosceles_l701_70133

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (triangle : Type)

noncomputable def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) (triangle : Type) : Prop :=
  c = 2 * a * Real.cos B → A = B ∨ B = C ∨ C = A

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) (triangle : Type) (h : c = 2 * a * Real.cos B) :
  is_isosceles_triangle A B C a b c triangle :=
sorry

end triangle_is_isosceles_l701_70133


namespace share_of_y_l701_70151

-- Define the conditions as hypotheses
variables (n : ℝ) (x y z : ℝ)

-- The main theorem we need to prove
theorem share_of_y (h1 : x = n) 
                   (h2 : y = 0.45 * n) 
                   (h3 : z = 0.50 * n) 
                   (h4 : x + y + z = 78) : 
  y = 18 :=
by 
  -- insert proof here (not required as per instructions)
  sorry

end share_of_y_l701_70151


namespace soccer_score_combinations_l701_70176

theorem soccer_score_combinations :
  ∃ (x y z : ℕ), x + y + z = 14 ∧ 3 * x + y = 19 ∧ x + y + z ≥ 0 ∧ 
    ({ (3, 10, 1), (4, 7, 3), (5, 4, 5), (6, 1, 7) } = 
      { (x, y, z) | x + y + z = 14 ∧ 3 * x + y = 19 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 }) :=
by 
  sorry

end soccer_score_combinations_l701_70176


namespace f_shift_l701_70177

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the main theorem
theorem f_shift (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) :=
by
  sorry

end f_shift_l701_70177


namespace f_2016_value_l701_70163

def f : ℝ → ℝ := sorry

axiom f_prop₁ : ∀ x : ℝ, (x + 6) + f x = 0
axiom f_symmetry : ∀ x : ℝ, f (-x) = -f x ∧ f 0 = 0

theorem f_2016_value : f 2016 = 0 :=
by
  sorry

end f_2016_value_l701_70163


namespace find_range_t_l701_70110

noncomputable def f (x t : ℝ) : ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4 * x

theorem find_range_t (f : ℝ → ℝ → ℝ)
  (h : ∀ t : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ t = x₁ - 6 ∧ f x₂ t = x₂ - 6 ∧ f x₃ t = x₃ - 6)) :
  ∀ t : ℝ, 1 < t ∧ t ≤ 2 := sorry

end find_range_t_l701_70110


namespace polar_to_cartesian_l701_70106

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 6) :
  (r * Real.cos θ, r * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  rw [h_r, h_θ]
  have h_cos : Real.cos (π / 6) = Real.sqrt 3 / 2 := sorry -- This identity can be used from trigonometric property.
  have h_sin : Real.sin (π / 6) = 1 / 2 := sorry -- This identity can be used from trigonometric property.
  rw [h_cos, h_sin]
  -- some algebraic steps to simplifiy left sides to (Real.sqrt 3, 1) should follow here. using multiplication and commmutaivity properties mainly.
  sorry

end polar_to_cartesian_l701_70106


namespace M_geq_N_l701_70160

variable (x y : ℝ)
def M : ℝ := x^2 + y^2 + 1
def N : ℝ := x + y + x * y

theorem M_geq_N (x y : ℝ) : M x y ≥ N x y :=
by
sorry

end M_geq_N_l701_70160


namespace original_price_of_trouser_l701_70187

theorem original_price_of_trouser (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 30) (h2 : discount = 0.70) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l701_70187


namespace shpuntik_can_form_triangle_l701_70123

theorem shpuntik_can_form_triangle 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (hx : x1 + x2 + x3 = 1)
  (hy : y1 + y2 + y3 = 1)
  (infeasibility_vintik : x1 ≥ x2 + x3) :
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a < b + c ∧ b < a + c ∧ c < a + b :=
sorry

end shpuntik_can_form_triangle_l701_70123


namespace arithmetic_seq_finite_negative_terms_l701_70171

theorem arithmetic_seq_finite_negative_terms (a d : ℝ) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → a + n * d ≥ 0) ↔ (a < 0 ∧ d > 0) :=
by
  sorry

end arithmetic_seq_finite_negative_terms_l701_70171


namespace recycling_points_l701_70105

theorem recycling_points (chloe_recycled : ℤ) (friends_recycled : ℤ) (points_per_pound : ℤ) :
  chloe_recycled = 28 ∧ friends_recycled = 2 ∧ points_per_pound = 6 → (chloe_recycled + friends_recycled) / points_per_pound = 5 :=
by
  sorry

end recycling_points_l701_70105


namespace solution_set_fraction_inequality_l701_70175

theorem solution_set_fraction_inequality : 
  { x : ℝ | 0 < x ∧ x < 1/3 } = { x : ℝ | 1/x > 3 } :=
by
  sorry

end solution_set_fraction_inequality_l701_70175


namespace intersection_points_lie_on_ellipse_l701_70121

theorem intersection_points_lie_on_ellipse (s : ℝ) : 
  ∃ (x y : ℝ), (2 * s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) ∧ (x^2 / 16 + y^2 / 9 = 1) :=
sorry

end intersection_points_lie_on_ellipse_l701_70121


namespace tony_total_puzzle_time_l701_70104

def warm_up_puzzle_time : ℕ := 10
def number_of_puzzles : ℕ := 2
def multiplier : ℕ := 3
def time_per_puzzle : ℕ := warm_up_puzzle_time * multiplier
def total_time : ℕ := warm_up_puzzle_time + number_of_puzzles * time_per_puzzle

theorem tony_total_puzzle_time : total_time = 70 := 
by
  sorry

end tony_total_puzzle_time_l701_70104


namespace squats_day_after_tomorrow_l701_70140

theorem squats_day_after_tomorrow (initial_squats : ℕ) (daily_increase : ℕ) (today : ℕ) (tomorrow : ℕ) (day_after_tomorrow : ℕ)
  (h1 : initial_squats = 30)
  (h2 : daily_increase = 5)
  (h3 : today = initial_squats + daily_increase)
  (h4 : tomorrow = today + daily_increase)
  (h5 : day_after_tomorrow = tomorrow + daily_increase) : 
  day_after_tomorrow = 45 := 
sorry

end squats_day_after_tomorrow_l701_70140


namespace cost_per_mile_first_plan_l701_70134

theorem cost_per_mile_first_plan 
  (initial_fee : ℝ) (cost_per_mile_first : ℝ) (cost_per_mile_second : ℝ) (miles : ℝ)
  (h_first : initial_fee = 65)
  (h_cost_second : cost_per_mile_second = 0.60)
  (h_miles : miles = 325)
  (h_equal_cost : initial_fee + miles * cost_per_mile_first = miles * cost_per_mile_second) :
  cost_per_mile_first = 0.40 :=
by
  sorry

end cost_per_mile_first_plan_l701_70134


namespace express_114_as_ones_and_threes_with_min_ten_ones_l701_70152

theorem express_114_as_ones_and_threes_with_min_ten_ones :
  ∃n: ℕ, n = 35 ∧ ∃ x y : ℕ, x + 3 * y = 114 ∧ x ≥ 10 := sorry

end express_114_as_ones_and_threes_with_min_ten_ones_l701_70152


namespace sally_out_of_pocket_cost_l701_70144

/-- Definitions of the given conditions -/
def given_money : Int := 320
def cost_per_book : Int := 15
def number_of_students : Int := 35

/-- Theorem to prove the amount Sally needs to pay out of pocket -/
theorem sally_out_of_pocket_cost : 
  let total_cost := number_of_students * cost_per_book
  let amount_given := given_money
  let out_of_pocket_cost := total_cost - amount_given
  out_of_pocket_cost = 205 := by
  sorry

end sally_out_of_pocket_cost_l701_70144


namespace inequality_abc_l701_70186

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_abc :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) :=
  sorry

end inequality_abc_l701_70186


namespace profit_per_unit_and_minimum_units_l701_70183

noncomputable def conditions (x y m : ℝ) : Prop :=
  2 * x + 7 * y = 41 ∧
  x + 3 * y = 18 ∧
  0.5 * m + 0.3 * (30 - m) ≥ 13.1

theorem profit_per_unit_and_minimum_units (x y m : ℝ) :
  conditions x y m → x = 3 ∧ y = 5 ∧ m ≥ 21 :=
by
  sorry

end profit_per_unit_and_minimum_units_l701_70183


namespace arithmetic_geometric_sequence_l701_70145

theorem arithmetic_geometric_sequence
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (b : ℕ → ℕ)
  (T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : S 3 = 12)
  (h3 : (a 1 + (a 2 - a 1))^2 = a 1 * (a 1 + 2 * (a 2 - a 1) + 2))
  (h4 : ∀ n, b n = (3 ^ n) * a n) :
  (∀ n, a n = 2 * n) ∧ 
  (∀ n, T n = (2 * n - 1) * 3^(n + 1) / 2 + 3 / 2) :=
sorry

end arithmetic_geometric_sequence_l701_70145


namespace max_value_f_zero_points_range_k_l701_70196

noncomputable def f (x k : ℝ) : ℝ := 3 * x^2 + 2 * (k - 1) * x + (k + 5)

theorem max_value_f (k : ℝ) (h : k < -7/2 ∨ k ≥ -7/2) :
  ∃ max_val : ℝ, max_val = if k < -7/2 then k + 5 else 7 * k + 26 :=
sorry

theorem zero_points_range_k :
  ∀ k : ℝ, (f 0 k) * (f 3 k) ≤ 0 ↔ (-5 ≤ k ∧ k ≤ -2) :=
sorry

end max_value_f_zero_points_range_k_l701_70196


namespace range_of_m_l701_70111

theorem range_of_m (m : ℝ) : 
  (¬ (∀ x : ℝ, x^2 + m * x + 1 = 0 → x > 0) → m ≥ -2) :=
by
  sorry

end range_of_m_l701_70111


namespace relationship_of_points_l701_70195

variable (y k b x : ℝ)
variable (y1 y2 : ℝ)

noncomputable def linear_func (x : ℝ) : ℝ := k * x - b

theorem relationship_of_points
  (h_pos_k : k > 0)
  (h_point1 : linear_func k b (-1) = y1)
  (h_point2 : linear_func k b 2 = y2):
  y1 < y2 := 
sorry

end relationship_of_points_l701_70195


namespace trapezoid_segment_length_l701_70159

theorem trapezoid_segment_length (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a^2 + b^2) / 2) :=
sorry

end trapezoid_segment_length_l701_70159


namespace find_m_for_parallel_lines_l701_70113

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0 → 
  -((2 : ℝ) / (m + 1)) = -(m / 3)) → (m = 2 ∨ m = -3) :=
by
  sorry

end find_m_for_parallel_lines_l701_70113


namespace inequality_solutions_l701_70192

theorem inequality_solutions (n : ℕ) (h : n > 0) : n^3 - n < n! ↔ (n = 1 ∨ n ≥ 6) := 
by
  sorry

end inequality_solutions_l701_70192


namespace number_of_B_students_l701_70125

/- Define the assumptions of the problem -/
variable (x : ℝ)  -- the number of students who earn a B

/- Express the number of students getting each grade in terms of x -/
def number_of_A (x : ℝ) := 0.6 * x
def number_of_C (x : ℝ) := 1.3 * x
def number_of_D (x : ℝ) := 0.8 * x
def total_students (x : ℝ) := number_of_A x + x + number_of_C x + number_of_D x

/- Prove that x = 14 for the total number of students being 50 -/
theorem number_of_B_students : total_students x = 50 → x = 14 :=
by 
  sorry

end number_of_B_students_l701_70125


namespace girls_more_than_boys_l701_70114

-- Defining the conditions
def ratio_boys_girls : Nat := 3 / 4
def total_students : Nat := 42

-- Defining the hypothesis based on conditions
theorem girls_more_than_boys : (total_students * ratio_boys_girls) / (3 + 4) * (4 - 3) = 6 := by
  sorry

end girls_more_than_boys_l701_70114


namespace fraction_blue_balls_l701_70170

theorem fraction_blue_balls (total_balls : ℕ) (red_fraction : ℚ) (other_balls : ℕ) (remaining_blue_fraction : ℚ) 
  (h1 : total_balls = 360) 
  (h2 : red_fraction = 1/4) 
  (h3 : other_balls = 216) 
  (h4 : remaining_blue_fraction = 1/5) :
  (total_balls - (total_balls / 4) - other_balls) = total_balls * (5 * red_fraction / 270) := 
by
  sorry

end fraction_blue_balls_l701_70170


namespace remaining_number_is_divisible_by_divisor_l701_70124

def initial_number : ℕ := 427398
def subtracted_number : ℕ := 8
def remaining_number : ℕ := initial_number - subtracted_number
def divisor : ℕ := 10

theorem remaining_number_is_divisible_by_divisor :
  remaining_number % divisor = 0 :=
by {
  sorry
}

end remaining_number_is_divisible_by_divisor_l701_70124


namespace part1_solution_part2_solution_l701_70161

section Part1

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

theorem part1_solution (x : ℝ) : f x > 0 ↔ x < -3 ∨ x > 2 :=
sorry

end Part1

section Part2

variables (a : ℝ) (ha : a < 0)
noncomputable def g (x : ℝ) : ℝ := a*x^2 + (3 - 2*a)*x - 6

theorem part2_solution (x : ℝ) :
  if h1 : a < -3/2 then g x < 0 ↔ x < -3/a ∨ x > 2
  else if h2 : a = -3/2 then g x < 0 ↔ x ≠ 2
  else -3/2 < a ∧ a < 0 → g x < 0 ↔ x < 2 ∨ x > -3/a :=
sorry

end Part2

end part1_solution_part2_solution_l701_70161


namespace sale_on_day_five_l701_70150

def sale1 : ℕ := 435
def sale2 : ℕ := 927
def sale3 : ℕ := 855
def sale6 : ℕ := 741
def average_sale : ℕ := 625
def total_days : ℕ := 5

theorem sale_on_day_five : 
  average_sale * total_days - (sale1 + sale2 + sale3 + sale6) = 167 :=
by
  sorry

end sale_on_day_five_l701_70150


namespace vasya_hits_ship_l701_70116

theorem vasya_hits_ship (board_size : ℕ) (ship_length : ℕ) (shots : ℕ) : 
  board_size = 10 ∧ ship_length = 4 ∧ shots = 24 → ∃ strategy : Fin board_size × Fin board_size → Prop, 
  (∀ pos, strategy pos → pos.1 * board_size + pos.2 < shots) ∧ 
  ∀ (ship_pos : Fin board_size × Fin board_size) (horizontal : Bool), 
  ∃ shot_pos, strategy shot_pos ∧ 
  (if horizontal then 
    ship_pos.1 = shot_pos.1 ∧ ship_pos.2 ≤ shot_pos.2 ∧ shot_pos.2 < ship_pos.2 + ship_length 
  else 
    ship_pos.2 = shot_pos.2 ∧ ship_pos.1 ≤ shot_pos.1 ∧ shot_pos.1 < ship_pos.1 + ship_length) :=
sorry

end vasya_hits_ship_l701_70116


namespace find_missing_term_l701_70193

theorem find_missing_term (a b : ℕ) : ∃ x, (2 * a - b) * x = 4 * a^2 - b^2 :=
by
  use (2 * a + b)
  sorry

end find_missing_term_l701_70193


namespace sheela_monthly_income_l701_70191

-- Definitions from the conditions
def deposited_amount : ℝ := 5000
def percentage_of_income : ℝ := 0.20

-- The theorem to be proven
theorem sheela_monthly_income : (deposited_amount / percentage_of_income) = 25000 := by
  sorry

end sheela_monthly_income_l701_70191


namespace range_of_a_l701_70126

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a * x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := 
by 
  sorry

end range_of_a_l701_70126


namespace mary_potatoes_l701_70130

theorem mary_potatoes (original new_except : ℕ) (h₁ : original = 25) (h₂ : new_except = 7) :
  original + new_except = 32 := by
  sorry

end mary_potatoes_l701_70130


namespace coordinates_of_P_l701_70185

theorem coordinates_of_P (a : ℝ) (h : 2 * a - 6 = 0) : (2 * a - 6, a + 1) = (0, 4) :=
by 
  have ha : a = 3 := by linarith
  rw [ha]
  sorry

end coordinates_of_P_l701_70185


namespace necklaces_sold_correct_l701_70101

-- Define the given constants and conditions
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20
def ensembles_sold : ℕ := 2
def total_revenue : ℕ := 565

-- Define the equation to calculate the total revenue
def total_revenue_calculation (N : ℕ) : ℕ :=
  (necklace_price * N) + (bracelet_price * bracelets_sold) + (earring_price * earrings_sold) + (ensemble_price * ensembles_sold)

-- Define the proof problem
theorem necklaces_sold_correct : 
  ∃ N : ℕ, total_revenue_calculation N = total_revenue ∧ N = 5 := by
  sorry

end necklaces_sold_correct_l701_70101


namespace product_of_fractions_is_3_div_80_l701_70148

def product_fractions (a b c d e f : ℚ) : ℚ := (a / b) * (c / d) * (e / f)

theorem product_of_fractions_is_3_div_80 
  (h₁ : product_fractions 3 8 2 5 1 4 = 3 / 80) : True :=
by
  sorry

end product_of_fractions_is_3_div_80_l701_70148


namespace calculate_f_at_pi_div_6_l701_70155

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem calculate_f_at_pi_div_6 (ω φ : ℝ) 
  (h : ∀ x : ℝ, f (π / 3 + x) ω φ = f (-x) ω φ) :
  f (π / 6) ω φ = 2 ∨ f (π / 6) ω φ = -2 :=
sorry

end calculate_f_at_pi_div_6_l701_70155


namespace necessary_but_not_sufficient_condition_l701_70178

variables {a b : ℤ}

theorem necessary_but_not_sufficient_condition : (¬(a = 1) ∨ ¬(b = 2)) ↔ ¬(a + b = 3) :=
by
  sorry

end necessary_but_not_sufficient_condition_l701_70178


namespace man_age_twice_son_age_l701_70103

theorem man_age_twice_son_age (S M X : ℕ) (h1 : S = 28) (h2 : M = S + 30) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end man_age_twice_son_age_l701_70103


namespace tetrahedron_mistaken_sum_l701_70179

theorem tetrahedron_mistaken_sum :
  let edges := 6
  let vertices := 4
  let faces := 4
  let joe_count := vertices + 1  -- Joe counts one vertex twice
  edges + joe_count + faces = 15 := by
  sorry

end tetrahedron_mistaken_sum_l701_70179


namespace vector_combination_l701_70120

-- Definitions of the given vectors and condition of parallelism
def vec_a : (ℝ × ℝ) := (1, -2)
def vec_b (m : ℝ) : (ℝ × ℝ) := (2, m)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0

-- Goal to prove
theorem vector_combination :
  ∀ m : ℝ, are_parallel vec_a (vec_b m) → 3 * vec_a.1 + 2 * (vec_b m).1 = 7 ∧ 3 * vec_a.2 + 2 * (vec_b m).2 = -14 :=
by
  intros m h_par
  sorry

end vector_combination_l701_70120


namespace arithmetic_sequence_ratio_l701_70138

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 d : ℝ)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) (h2 : ∀ n, S n = n * (2 * a1 + (n - 1) * d) / 2)
  (h_nonzero: ∀ n, a n ≠ 0):
  (S 5) / (a 3) = 5 :=
by
  sorry

end arithmetic_sequence_ratio_l701_70138


namespace arithmetic_sequence_n_l701_70107

theorem arithmetic_sequence_n 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3_plus_a5 : a 3 + a 5 = 14)
  (Sn_eq_100 : S n = 100) :
  n = 10 :=
sorry

end arithmetic_sequence_n_l701_70107


namespace variance_is_stability_measure_l701_70122

def stability_measure (yields : Fin 10 → ℝ) : Prop :=
  let mean := (yields 0 + yields 1 + yields 2 + yields 3 + yields 4 + yields 5 + yields 6 + yields 7 + yields 8 + yields 9) / 10
  let variance := 
    ((yields 0 - mean)^2 + (yields 1 - mean)^2 + (yields 2 - mean)^2 + (yields 3 - mean)^2 + 
     (yields 4 - mean)^2 + (yields 5 - mean)^2 + (yields 6 - mean)^2 + (yields 7 - mean)^2 + 
     (yields 8 - mean)^2 + (yields 9 - mean)^2) / 10
  true -- just a placeholder, would normally state that this is the appropriate measure

theorem variance_is_stability_measure (yields : Fin 10 → ℝ) : stability_measure yields :=
by 
  sorry

end variance_is_stability_measure_l701_70122


namespace max_k_exists_l701_70173

noncomputable def max_possible_k (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) : ℝ :=
sorry

theorem max_k_exists (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) :
  ∃ k_max : ℝ, k_max = max_possible_k x y k h_pos h_eq :=
sorry

end max_k_exists_l701_70173


namespace solve_for_x_l701_70197

theorem solve_for_x : ∃ x k l : ℕ, (3 * 22 = k) ∧ (66 + l = 90) ∧ (160 * 3 / 4 = x - l) → x = 144 :=
by
  sorry

end solve_for_x_l701_70197


namespace geom_seq_sum_3000_l701_70194

noncomputable
def sum_geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n
  else a * (1 - r ^ n) / (1 - r)

theorem geom_seq_sum_3000 (a r : ℝ) (h1: sum_geom_seq a r 1000 = 300) (h2: sum_geom_seq a r 2000 = 570) :
  sum_geom_seq a r 3000 = 813 :=
sorry

end geom_seq_sum_3000_l701_70194


namespace parallel_vectors_perpendicular_vectors_l701_70147

/-- Given vectors a and b where a = (1, 2) and b = (x, 1),
    let u = a + b and v = a - b.
    Prove that if u is parallel to v, then x = 1/2. 
    Also, prove that if u is perpendicular to v, then x = 2 or x = -2. --/

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)
noncomputable def vector_u (x : ℝ) : ℝ × ℝ := (1 + x, 3)
noncomputable def vector_v (x : ℝ) : ℝ × ℝ := (1 - x, 1)

theorem parallel_vectors (x : ℝ) :
  (vector_u x).fst / (vector_v x).fst = (vector_u x).snd / (vector_v x).snd ↔ x = 1 / 2 :=
by
  sorry

theorem perpendicular_vectors (x : ℝ) :
  (vector_u x).fst * (vector_v x).fst + (vector_u x).snd * (vector_v x).snd = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end parallel_vectors_perpendicular_vectors_l701_70147


namespace income_ratio_l701_70102

theorem income_ratio (I1 I2 E1 E2 : ℕ) (h1 : I1 = 5000) (h2 : E1 / E2 = 3 / 2) (h3 : I1 - E1 = 2000) (h4 : I2 - E2 = 2000) : I1 / I2 = 5 / 4 :=
by
  /- Proof omitted -/
  sorry

end income_ratio_l701_70102


namespace math_and_english_scores_sum_l701_70100

theorem math_and_english_scores_sum (M E : ℕ) (total_score : ℕ) :
  (∀ (H : ℕ), H = (50 + M + E) / 3 → 
   50 + M + E + H = total_score) → 
   total_score = 248 → 
   M + E = 136 :=
by
  intros h1 h2;
  sorry

end math_and_english_scores_sum_l701_70100


namespace math_problem_l701_70117

theorem math_problem (x y : ℝ) (h1 : x + Real.sin y = 2023) (h2 : x + 2023 * Real.cos y = 2022) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2022 + Real.pi / 2 :=
sorry

end math_problem_l701_70117


namespace max_chord_length_l701_70129

noncomputable def family_of_curves (θ x y : ℝ) := 
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line (x y : ℝ) := 2 * x = y

theorem max_chord_length :
  (∀ (θ : ℝ), ∀ (x y : ℝ), family_of_curves θ x y → line x y) → 
  ∃ (L : ℝ), L = 8 * Real.sqrt 5 :=
by
  sorry

end max_chord_length_l701_70129


namespace gcd_of_1237_and_1957_is_one_l701_70168

noncomputable def gcd_1237_1957 : Nat := Nat.gcd 1237 1957

theorem gcd_of_1237_and_1957_is_one : gcd_1237_1957 = 1 :=
by
  unfold gcd_1237_1957
  have : Nat.gcd 1237 1957 = 1 := sorry
  exact this

end gcd_of_1237_and_1957_is_one_l701_70168


namespace smallest_positive_integer_in_form_l701_70139

theorem smallest_positive_integer_in_form (m n : ℤ) : 
  ∃ m n : ℤ, 3001 * m + 24567 * n = 1 :=
by
  sorry

end smallest_positive_integer_in_form_l701_70139


namespace train_length_l701_70198

theorem train_length :
  ∃ L : ℝ, 
    (∀ V : ℝ, V = L / 24 ∧ V = (L + 650) / 89) → 
    L = 240 :=
by
  sorry

end train_length_l701_70198


namespace range_of_a_l701_70146

def f (x : ℝ) : ℝ := 3 * x * |x|

theorem range_of_a : {a : ℝ | f (1 - a) + f (2 * a) < 0 } = {a : ℝ | a < -1} :=
by
  sorry

end range_of_a_l701_70146


namespace Mrs_Brown_points_l701_70135

-- Conditions given
variables (points_William points_Adams points_Daniel points_mean: ℝ) (num_classes: ℕ)

-- Define the conditions
def Mrs_William_points := points_William = 50
def Mr_Adams_points := points_Adams = 57
def Mrs_Daniel_points := points_Daniel = 57
def mean_condition := points_mean = 53.3
def num_classes_condition := num_classes = 4

-- Define the problem to prove
theorem Mrs_Brown_points :
  Mrs_William_points points_William ∧ Mr_Adams_points points_Adams ∧ Mrs_Daniel_points points_Daniel ∧ mean_condition points_mean ∧ num_classes_condition num_classes →
  ∃ (points_Brown: ℝ), points_Brown = 49 :=
by
  sorry

end Mrs_Brown_points_l701_70135


namespace relationship_among_abc_l701_70188

noncomputable def a : ℝ := (1/4)^(1/2)
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := (1/3)^(1/2)

theorem relationship_among_abc : b > c ∧ c > a :=
by
  -- Proof will go here
  sorry

end relationship_among_abc_l701_70188


namespace quadratic_function_integer_values_not_imply_integer_coefficients_l701_70132

theorem quadratic_function_integer_values_not_imply_integer_coefficients :
  ∃ (a b c : ℚ), (∀ x : ℤ, ∃ y : ℤ, (a * (x : ℚ)^2 + b * (x : ℚ) + c = (y : ℚ))) ∧
    (¬ (∃ (a_int b_int c_int : ℤ), a = (a_int : ℚ) ∧ b = (b_int : ℚ) ∧ c = (c_int : ℚ))) :=
by
  sorry

end quadratic_function_integer_values_not_imply_integer_coefficients_l701_70132


namespace pentagon_angle_T_l701_70184

theorem pentagon_angle_T (P Q R S T : ℝ) 
  (hPRT: P = R ∧ R = T)
  (hQS: Q + S = 180): 
  T = 120 :=
by
  sorry

end pentagon_angle_T_l701_70184


namespace price_increase_eq_20_percent_l701_70153

theorem price_increase_eq_20_percent (a x : ℝ) (h : a * (1 + x) * (1 + x) = a * 1.44) : x = 0.2 :=
by {
  -- This part will contain the proof steps.
  sorry -- Placeholder
}

end price_increase_eq_20_percent_l701_70153


namespace children_group_size_l701_70108

theorem children_group_size (x : ℕ) (h1 : 255 % 17 = 0) (h2: ∃ n : ℕ, n * 17 = 255) 
                            (h3 : ∀ a c, a = c → a = 255 → c = 255 → x = 17) : 
                            (255 / x = 15) → x = 17 :=
by
  sorry

end children_group_size_l701_70108


namespace terminal_side_in_third_quadrant_l701_70172

open Real

theorem terminal_side_in_third_quadrant (θ : ℝ) (h1 : sin θ < 0) (h2 : cos θ < 0) : 
    θ ∈ Set.Ioo (π : ℝ) (3 * π / 2) := 
sorry

end terminal_side_in_third_quadrant_l701_70172


namespace find_number_l701_70157

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l701_70157


namespace equation_of_perpendicular_line_l701_70154

theorem equation_of_perpendicular_line :
  ∃ c : ℝ, (∀ x y : ℝ, (2 * x + y + c = 0 ↔ (x = 1 ∧ y = 1))) → (c = -3) := 
by
  sorry

end equation_of_perpendicular_line_l701_70154


namespace min_value_xyz_l701_70166

open Real

theorem min_value_xyz
  (x y z : ℝ)
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : 5 * x + 16 * y + 33 * z ≥ 136) :
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 :=
sorry

end min_value_xyz_l701_70166


namespace complex_expression_proof_l701_70142

open Complex

theorem complex_expression_proof {x y z : ℂ}
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 18 :=
by
  sorry

end complex_expression_proof_l701_70142


namespace gcd_1729_1337_l701_70169

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := 
by
  sorry

end gcd_1729_1337_l701_70169


namespace factorial_divisibility_l701_70119

theorem factorial_divisibility (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : a ≥ 2 * b + 1 :=
sorry

end factorial_divisibility_l701_70119


namespace sales_worth_l701_70156

theorem sales_worth (S: ℝ) : 
  (1300 + 0.025 * (S - 4000) = 0.05 * S + 600) → S = 24000 :=
by
  sorry

end sales_worth_l701_70156


namespace find_triples_of_positive_integers_l701_70158

theorem find_triples_of_positive_integers (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn_pos : 0 < n) 
  (equation : p * (p + 3) + q * (q + 3) = n * (n + 3)) : 
  (p = 3 ∧ q = 2 ∧ n = 4) :=
sorry

end find_triples_of_positive_integers_l701_70158


namespace inequality_problem_l701_70180

theorem inequality_problem
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  (b^2 / a + a^2 / b) ≥ (a + b) :=
sorry

end inequality_problem_l701_70180


namespace system_of_equations_inconsistent_l701_70190

theorem system_of_equations_inconsistent :
  ¬∃ (x1 x2 x3 x4 x5 : ℝ), 
    (x1 + 2 * x2 - x3 + 3 * x4 - x5 = 0) ∧ 
    (2 * x1 - x2 + 3 * x3 + x4 - x5 = -1) ∧
    (x1 - x2 + x3 + 2 * x4 = 2) ∧
    (4 * x1 + 3 * x3 + 6 * x4 - 2 * x5 = 5) := 
sorry

end system_of_equations_inconsistent_l701_70190


namespace neg_p_iff_exists_ge_zero_l701_70143

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + x + 1 < 0

theorem neg_p_iff_exists_ge_zero : ¬ p ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by 
   sorry

end neg_p_iff_exists_ge_zero_l701_70143


namespace largest_sum_ABC_l701_70165

theorem largest_sum_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 3003) : 
  A + B + C ≤ 105 :=
sorry

end largest_sum_ABC_l701_70165


namespace perfect_squares_between_2_and_20_l701_70174

-- Defining the conditions and problem statement
theorem perfect_squares_between_2_and_20 : 
  ∃ n, n = 3 ∧ ∀ m, (2 < m ∧ m < 20 ∧ ∃ k, k * k = m) ↔ m = 4 ∨ m = 9 ∨ m = 16 :=
by {
  -- Start the proof process
  sorry -- Placeholder for the proof
}

end perfect_squares_between_2_and_20_l701_70174


namespace no_integer_solution_l701_70115

theorem no_integer_solution :
  ¬(∃ x : ℤ, 7 - 3 * (x^2 - 2) > 19) :=
by
  sorry

end no_integer_solution_l701_70115


namespace find_orange_juice_amount_l701_70149

variable (s y t oj : ℝ)

theorem find_orange_juice_amount (h1 : s = 0.2) (h2 : y = 0.1) (h3 : t = 0.5) (h4 : oj = t - (s + y)) : oj = 0.2 :=
by
  sorry

end find_orange_juice_amount_l701_70149


namespace cars_each_remaining_day_l701_70182

theorem cars_each_remaining_day (total_cars : ℕ) (monday_cars : ℕ) (tuesday_cars : ℕ)
  (wednesday_cars : ℕ) (thursday_cars : ℕ) (remaining_days : ℕ)
  (h_total : total_cars = 450)
  (h_mon : monday_cars = 50)
  (h_tue : tuesday_cars = 50)
  (h_wed : wednesday_cars = 2 * monday_cars)
  (h_thu : thursday_cars = 2 * monday_cars)
  (h_remaining : remaining_days = (total_cars - (monday_cars + tuesday_cars + wednesday_cars + thursday_cars)) / 3)
  :
  remaining_days = 50 := sorry

end cars_each_remaining_day_l701_70182


namespace total_revenue_correct_l701_70164

def items : Type := ℕ × ℝ

def magazines : items := (425, 2.50)
def newspapers : items := (275, 1.50)
def books : items := (150, 5.00)
def pamphlets : items := (75, 0.50)

def revenue (item : items) : ℝ := item.1 * item.2

def total_revenue : ℝ :=
  revenue magazines +
  revenue newspapers +
  revenue books +
  revenue pamphlets

theorem total_revenue_correct : total_revenue = 2262.50 := by
  sorry

end total_revenue_correct_l701_70164


namespace number_of_boys_l701_70199

theorem number_of_boys {total_students : ℕ} (h1 : total_students = 49)
  (ratio_girls_boys : ℕ → ℕ → Prop)
  (h2 : ratio_girls_boys 4 3) :
  ∃ boys : ℕ, boys = 21 := by
  sorry

end number_of_boys_l701_70199


namespace Nikka_stamp_collection_l701_70141

theorem Nikka_stamp_collection (S : ℝ) 
  (h1 : 0.35 * S ≥ 0) 
  (h2 : 0.2 * S ≥ 0) 
  (h3 : 0 < S) 
  (h4 : 0.45 * S = 45) : S = 100 :=
sorry

end Nikka_stamp_collection_l701_70141


namespace symmetric_circle_eq_l701_70136

open Real

-- Define the original circle equation and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation with respect to the line y = -x
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Define the new circle that is symmetric to the original circle with respect to y = -x
def new_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- The theorem to be proven
theorem symmetric_circle_eq :
  ∀ x y : ℝ, original_circle (-y) (-x) ↔ new_circle x y := 
by
  sorry

end symmetric_circle_eq_l701_70136


namespace angle_QPS_l701_70167

-- Definitions of the points and angles
variables (P Q R S : Point)
variables (angle : Point → Point → Point → ℝ)

-- Conditions about the isosceles triangles and angles
variables (isosceles_PQR : PQ = QR)
variables (isosceles_PRS : PR = RS)
variables (R_inside_PQS : ¬(R ∈ convex_hull ℝ {P, Q, S}))
variables (angle_PQR : angle P Q R = 50)
variables (angle_PRS : angle P R S = 120)

-- The theorem we want to prove
theorem angle_QPS : angle Q P S = 35 :=
sorry -- Proof goes here

end angle_QPS_l701_70167


namespace distance_AB_l701_70137

-- Definitions and conditions taken from part a)
variables (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0)

-- The main theorem statement
theorem distance_AB (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0) : 
  ∃ s : ℝ, s = Real.sqrt ((a * b * c) / (a + c - b)) := 
sorry

end distance_AB_l701_70137


namespace selina_sells_5_shirts_l701_70162

theorem selina_sells_5_shirts
    (pants_price shorts_price shirts_price : ℕ)
    (pants_sold shorts_sold shirts_bought remaining_money : ℕ)
    (total_earnings : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  pants_sold = 3 →
  shorts_sold = 5 →
  shirts_bought = 2 →
  remaining_money = 30 →
  total_earnings = remaining_money + shirts_bought * 10 →
  total_earnings = 50 →
  total_earnings = pants_sold * pants_price + shorts_sold * shorts_price + 20 →
  20 / shirts_price = 5 :=
by
  sorry

end selina_sells_5_shirts_l701_70162


namespace system_of_equations_solution_l701_70131

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x - 2 * y = 1)
  (h2 : 3 * x + 4 * y = 23) :
  x = 5 ∧ y = 2 :=
sorry

end system_of_equations_solution_l701_70131


namespace cost_formula_l701_70118

def cost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem cost_formula (P : ℕ) : 
  cost P = (if P ≤ 5 then 5 * P + 10 else 5 * P + 5) :=
by 
  sorry

end cost_formula_l701_70118


namespace percent_water_evaporated_l701_70189

theorem percent_water_evaporated (W : ℝ) (E : ℝ) (T : ℝ) (hW : W = 10) (hE : E = 0.16) (hT : T = 75) : 
  ((min (E * T) W) / W) * 100 = 100 :=
by
  sorry

end percent_water_evaporated_l701_70189
