import Mathlib

namespace largest_base_5_three_digit_in_base_10_l108_108546

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l108_108546


namespace circumcircle_circumference_thm_triangle_perimeter_thm_l108_108527

-- Definition and theorem for the circumference of the circumcircle
def circumcircle_circumference (a b c R : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * R = c / (Real.sqrt (1 - cosC^2)) 
  ∧ 2 * R * Real.pi = 3 * Real.pi

theorem circumcircle_circumference_thm (a b c R : ℝ) (cosC : ℝ) :
  circumcircle_circumference a b c R cosC → 2 * R * Real.pi = 3 * Real.pi :=
by
  intro h;
  sorry

-- Definition and theorem for the perimeter of the triangle
def triangle_perimeter (a b c : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * a = 3 * b ∧ (a + b + c) = 5 + Real.sqrt 5

theorem triangle_perimeter_thm (a b c : ℝ) (cosC : ℝ) :
  triangle_perimeter a b c cosC → (a + b + c) = 5 + Real.sqrt 5 :=
by
  intro h;
  sorry

end circumcircle_circumference_thm_triangle_perimeter_thm_l108_108527


namespace fraction_of_males_l108_108296

theorem fraction_of_males (M F : ℝ) (h1 : M + F = 1) (h2 : (7/8 * M + 9/10 * (1 - M)) = 0.885) :
  M = 0.6 :=
sorry

end fraction_of_males_l108_108296


namespace length_of_plot_is_60_l108_108439

noncomputable def plot_length (b : ℝ) : ℝ :=
  b + 20

noncomputable def plot_perimeter (b : ℝ) : ℝ :=
  2 * (plot_length b + b)

noncomputable def plot_cost_eq (b : ℝ) : Prop :=
  26.50 * plot_perimeter b = 5300

theorem length_of_plot_is_60 : ∃ b : ℝ, plot_cost_eq b ∧ plot_length b = 60 :=
sorry

end length_of_plot_is_60_l108_108439


namespace find_A_l108_108769

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) 
(h_div9 : (A + 1 + 5 + B + 9 + 4) % 9 = 0) 
(h_div11 : (A + 5 + 9 - (1 + B + 4)) % 11 = 0) : A = 5 :=
by sorry

end find_A_l108_108769


namespace total_balls_l108_108116

theorem total_balls (r b g : ℕ) (ratio : r = 2 * k ∧ b = 4 * k ∧ g = 6 * k) (green_balls : g = 36) : r + b + g = 72 :=
by
  sorry

end total_balls_l108_108116


namespace probability_of_fourth_roll_l108_108695

-- Define the conditions 
structure Die :=
(fair : Bool) 
(biased_six : Bool)
(biased_one : Bool)

-- Define the probability function
def roll_prob (d : Die) (f : Bool) : ℚ :=
  if d.fair then 1/6
  else if d.biased_six then if f then 1/2 else 1/10
  else if d.biased_one then if f then 1/10 else 1/5
  else 0

def probability_of_fourth_six (p q : ℕ) (r1 r2 r3 : Bool) (d : Die) : ℚ :=
  (if r1 && r2 && r3 then roll_prob d true else 0) 

noncomputable def final_probability (d1 d2 d3 : Die) (prob_fair distorted_rolls : Bool) : ℚ :=
  let fair_prob := if distorted_rolls then roll_prob d1 true else roll_prob d1 false
  let biased_six_prob := if distorted_rolls then roll_prob d2 true else roll_prob d2 false
  let total_prob := fair_prob + biased_six_prob
  let fair := fair_prob / total_prob
  let biased_six := biased_six_prob / total_prob
  fair * roll_prob d1 true + biased_six * roll_prob d2 true

theorem probability_of_fourth_roll
  (d1 : Die) (d2 : Die) (d3 : Die)
  (h1 : d1.fair = true)
  (h2 : d2.biased_six = true)
  (h3 : d3.biased_one = true)
  (h4 : ∀ d, d1 = d ∨ d2 = d ∨ d3 = d)
  (r1 r2 r3 : Bool)
  : ∃ p q : ℕ, p + q = 11 ∧ final_probability d1 d2 d3 true = 5/6 := 
sorry

end probability_of_fourth_roll_l108_108695


namespace f_eq_n_for_all_n_l108_108026

noncomputable def f : ℕ → ℕ := sorry

axiom f_pos_int_valued (n : ℕ) (h : 0 < n) : f n = f n

axiom f_2_eq_2 : f 2 = 2

axiom f_mul_prop (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n

axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : f m > f n

theorem f_eq_n_for_all_n (n : ℕ) (hn : 0 < n) : f n = n := sorry

end f_eq_n_for_all_n_l108_108026


namespace find_k_l108_108319

noncomputable def S (n : ℕ) : ℤ := n^2 - 8 * n
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_k (k : ℕ) (h : a k = 5) : k = 7 := by
  sorry

end find_k_l108_108319


namespace abs_gt_implies_nec_not_suff_l108_108233

theorem abs_gt_implies_nec_not_suff {a b : ℝ} : 
  (|a| > b) → (∀ (a b : ℝ), a > b → |a| > b) ∧ ¬(∀ (a b : ℝ), |a| > b → a > b) :=
by
  sorry

end abs_gt_implies_nec_not_suff_l108_108233


namespace largest_whole_x_l108_108305

theorem largest_whole_x (x : ℕ) (h : 11 * x < 150) : x ≤ 13 :=
sorry

end largest_whole_x_l108_108305


namespace john_finish_work_alone_in_48_days_l108_108267

variable {J R : ℝ}

theorem john_finish_work_alone_in_48_days
  (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 2 / 3)
  (h3 : 16 * J = 1 / 3) :
  1 / J = 48 := 
by
  sorry

end john_finish_work_alone_in_48_days_l108_108267


namespace bike_ride_distance_l108_108429

-- Definitions for conditions from a)
def speed_out := 24 -- miles per hour
def speed_back := 18 -- miles per hour
def total_time := 7 -- hours

-- Problem statement for the proof problem
theorem bike_ride_distance :
  ∃ (D : ℝ), (D / speed_out) + (D / speed_back) = total_time ∧ 2 * D = 144 :=
by {
  sorry
}

end bike_ride_distance_l108_108429


namespace f_of_f_eq_f_l108_108497

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem f_of_f_eq_f (x : ℝ) : f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 :=
by
  sorry

end f_of_f_eq_f_l108_108497


namespace original_number_is_400_l108_108345

theorem original_number_is_400 (x : ℝ) (h : 1.20 * x = 480) : x = 400 :=
sorry

end original_number_is_400_l108_108345


namespace sum_of_first_7_terms_l108_108494

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

theorem sum_of_first_7_terms (h1 : a 2 = 3) (h2 : a 6 = 11)
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2) : S 7 = 49 :=
by 
  sorry

end sum_of_first_7_terms_l108_108494


namespace taxi_speed_l108_108218

theorem taxi_speed (v : ℝ) (hA : ∀ v : ℝ, 3 * v = 6 * (v - 30)) : v = 60 :=
by
  sorry

end taxi_speed_l108_108218


namespace additional_people_needed_l108_108340

def total_days := 50
def initial_people := 40
def days_passed := 25
def work_completed := 0.40

theorem additional_people_needed : 
  ∃ additional_people : ℕ, additional_people = 8 :=
by
  -- Placeholder for the actual proof skipped with 'sorry'
  sorry

end additional_people_needed_l108_108340


namespace meat_cost_per_pound_l108_108683

def total_cost_box : ℝ := 5
def cost_per_bell_pepper : ℝ := 1.5
def num_bell_peppers : ℝ := 4
def num_pounds_meat : ℝ := 2
def total_spent : ℝ := 17

theorem meat_cost_per_pound : total_spent - (total_cost_box + num_bell_peppers * cost_per_bell_pepper) = 6 -> 
                             6 / num_pounds_meat = 3 := by
  sorry

end meat_cost_per_pound_l108_108683


namespace range_of_p_l108_108289

theorem range_of_p 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = (-1 : ℝ)^n * a n + 1/(2^n) + n - 3)
  (h2 : ∀ n : ℕ, (a (n + 1) - p) * (a n - p) < 0) :
  -3/4 < p ∧ p < 11/4 :=
sorry

end range_of_p_l108_108289


namespace determine_k_l108_108281

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

-- State the problem
theorem determine_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4)
  ↔ (k = 3 / 8 ∨ k = -3) :=
by
  sorry

end determine_k_l108_108281


namespace true_propositions_l108_108307

theorem true_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + 2*x - m = 0) ∧            -- Condition 1
  ((∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧                    -- Condition 2
   (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ) ∧
  (∀ x y : ℝ, (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)) ∧              -- Condition 3
  ¬ ( (∀ p q : Prop, ¬p → ¬ (p ∧ q)) ∧ (¬ ¬p → p ∧ q) ) ∧   -- Condition 4
  (∃ x : ℝ, x^2 + x + 3 ≤ 0)                                 -- Condition 5
:= by {
  sorry
}

end true_propositions_l108_108307


namespace total_birds_times_types_l108_108232

-- Defining the number of adults and offspring for each type of bird.
def num_ducks1 : ℕ := 2
def num_ducklings1 : ℕ := 5
def num_ducks2 : ℕ := 6
def num_ducklings2 : ℕ := 3
def num_ducks3 : ℕ := 9
def num_ducklings3 : ℕ := 6

def num_geese : ℕ := 4
def num_goslings : ℕ := 7

def num_swans : ℕ := 3
def num_cygnets : ℕ := 4

-- Calculate total number of birds
def total_ducks := (num_ducks1 * num_ducklings1 + num_ducks1) + (num_ducks2 * num_ducklings2 + num_ducks2) +
                      (num_ducks3 * num_ducklings3 + num_ducks3)

def total_geese := num_geese * num_goslings + num_geese
def total_swans := num_swans * num_cygnets + num_swans

def total_birds := total_ducks + total_geese + total_swans

-- Calculate the number of different types of birds
def num_types_of_birds : ℕ := 3 -- ducks, geese, swans

-- The final Lean statement to be proven
theorem total_birds_times_types :
  total_birds * num_types_of_birds = 438 :=
  by sorry

end total_birds_times_types_l108_108232


namespace problem_statement_l108_108452

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a + 1 / a) ^ 2 + (b + 1 / b) ^ 2 ≥ 25 / 2 := 
by
  sorry

end problem_statement_l108_108452


namespace simple_annual_interest_rate_l108_108230

-- Given definitions and conditions
def monthly_interest_payment := 225
def principal_amount := 30000
def annual_interest_payment := monthly_interest_payment * 12
def annual_interest_rate := annual_interest_payment / principal_amount

-- Theorem statement
theorem simple_annual_interest_rate :
  annual_interest_rate * 100 = 9 := by
sorry

end simple_annual_interest_rate_l108_108230


namespace simplify_expression_l108_108594

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  (x^2 + y^2 + z^2 - 2 * x * y * z) = 4 :=
by
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  sorry

end simplify_expression_l108_108594


namespace first_team_speed_l108_108908

theorem first_team_speed:
  ∃ v: ℝ, 
  (∀ (t: ℝ), t = 2.5 → 
  (∀ s: ℝ, s = 125 → 
  (v + 30) * t = s) ∧ v = 20) := 
  sorry

end first_team_speed_l108_108908


namespace triangle_area_six_parts_l108_108983

theorem triangle_area_six_parts (S S₁ S₂ S₃ : ℝ) (h₁ : S₁ ≥ 0) (h₂ : S₂ ≥ 0) (h₃ : S₃ ≥ 0) :
  S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃) ^ 2 := 
sorry

end triangle_area_six_parts_l108_108983


namespace smallest_three_digit_multiple_of_13_l108_108085

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n ∧ (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l108_108085


namespace discount_percentage_in_february_l108_108535

theorem discount_percentage_in_february (C : ℝ) (h1 : C > 0) 
(markup1 : ℝ) (markup2 : ℝ) (profit : ℝ) (D : ℝ) :
  markup1 = 0.20 → markup2 = 0.25 → profit = 0.125 →
  1.50 * C * (1 - D) = 1.125 * C → D = 0.25 :=
by
  intros
  sorry

end discount_percentage_in_february_l108_108535


namespace max_cells_cut_diagonals_l108_108883

theorem max_cells_cut_diagonals (board_size : ℕ) (k : ℕ) (internal_cells : ℕ) :
  board_size = 9 →
  internal_cells = (board_size - 2) ^ 2 →
  64 = internal_cells →
  V = internal_cells + k →
  E = 4 * k →
  k ≤ 21 :=
by
  sorry

end max_cells_cut_diagonals_l108_108883


namespace hexadecagon_area_l108_108658

theorem hexadecagon_area (r : ℝ) : 
  (∃ A : ℝ, A = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)) :=
sorry

end hexadecagon_area_l108_108658


namespace mala_usha_speed_ratio_l108_108853

noncomputable def drinking_speed_ratio (M U : ℝ) (tM tU : ℝ) (fracU : ℝ) (total_bottle : ℝ) : ℝ :=
  let U_speed := fracU * total_bottle / tU
  let M_speed := (total_bottle - fracU * total_bottle) / tM
  M_speed / U_speed

theorem mala_usha_speed_ratio :
  drinking_speed_ratio (3/50) (1/50) 10 20 (4/10) 1 = 3 :=
by
  sorry

end mala_usha_speed_ratio_l108_108853


namespace min_value_of_quadratic_expression_l108_108421

theorem min_value_of_quadratic_expression : ∃ x : ℝ, ∀ y : ℝ, y = x^2 + 12*x + 9 → y ≥ -27 :=
sorry

end min_value_of_quadratic_expression_l108_108421


namespace circle_diameter_given_area_l108_108464

theorem circle_diameter_given_area : 
  (∃ (r : ℝ), 81 * Real.pi = Real.pi * r^2 ∧ 2 * r = d) → d = 18 := by
  sorry

end circle_diameter_given_area_l108_108464


namespace integer_solution_l108_108260

theorem integer_solution (n m : ℤ) (h : (n + 2)^4 - n^4 = m^3) : (n = -1 ∧ m = 0) :=
by
  sorry

end integer_solution_l108_108260


namespace probability_of_winning_is_correct_l108_108014

theorem probability_of_winning_is_correct :
  ∀ (PWin PLoss PTie : ℚ),
    PLoss = 5/12 →
    PTie = 1/6 →
    PWin + PLoss + PTie = 1 →
    PWin = 5/12 := 
by
  intros PWin PLoss PTie hLoss hTie hSum
  sorry

end probability_of_winning_is_correct_l108_108014


namespace clarence_initial_oranges_l108_108082

variable (initial_oranges : ℕ)
variable (obtained_from_joyce : ℕ := 3)
variable (total_oranges : ℕ := 8)

theorem clarence_initial_oranges (initial_oranges : ℕ) :
  initial_oranges + obtained_from_joyce = total_oranges → initial_oranges = 5 :=
by
  sorry

end clarence_initial_oranges_l108_108082


namespace remainder_when_a_plus_b_div_40_is_28_l108_108067

theorem remainder_when_a_plus_b_div_40_is_28 :
  ∃ k j : ℤ, (a = 80 * k + 74 ∧ b = 120 * j + 114) → (a + b) % 40 = 28 := by
  sorry

end remainder_when_a_plus_b_div_40_is_28_l108_108067


namespace problem_statement_l108_108840

noncomputable def f (a x : ℝ) : ℝ := a^x + a^(-x)

theorem problem_statement (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f a 1 = 3) :
  f a 0 + f a 1 + f a 2 = 12 :=
sorry

end problem_statement_l108_108840


namespace largest_n_l108_108579

noncomputable def is_multiple_of_seven (n : ℕ) : Prop :=
  (6 * (n-3)^3 - n^2 + 10 * n - 15) % 7 = 0

theorem largest_n (n : ℕ) : n < 50000 ∧ is_multiple_of_seven n → n = 49999 :=
by sorry

end largest_n_l108_108579


namespace sqrt_eq_sum_iff_l108_108884

open Real

theorem sqrt_eq_sum_iff (a b : ℝ) : sqrt (a^2 + b^2) = a + b ↔ (a * b = 0) ∧ (a + b ≥ 0) :=
by
  sorry

end sqrt_eq_sum_iff_l108_108884


namespace m_value_if_linear_l108_108278

theorem m_value_if_linear (m : ℝ) (x : ℝ) (h : (m + 2) * x^(|m| - 1) + 8 = 0) (linear : |m| - 1 = 1) : m = 2 :=
sorry

end m_value_if_linear_l108_108278


namespace find_ordered_pair_l108_108768

theorem find_ordered_pair : ∃ k a : ℤ, 
  (∀ x : ℝ, (x^3 - 4*x^2 + 9*x - 6) % (x^2 - x + k) = 2*x + a) ∧ k = 4 ∧ a = 6 :=
sorry

end find_ordered_pair_l108_108768


namespace symmetric_point_xoz_plane_l108_108517

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_xoz (M : Point3D) : Point3D :=
  ⟨M.x, -M.y, M.z⟩

theorem symmetric_point_xoz_plane :
  let M := Point3D.mk 5 1 (-2)
  symmetric_xoz M = Point3D.mk 5 (-1) (-2) :=
by
  sorry

end symmetric_point_xoz_plane_l108_108517


namespace solve_porters_transportation_l108_108629

variable (x : ℝ)

def porters_transportation_equation : Prop :=
  (5000 / x = 8000 / (x + 600))

theorem solve_porters_transportation (x : ℝ) (h₁ : 600 > 0) (h₂ : x > 0):
  porters_transportation_equation x :=
sorry

end solve_porters_transportation_l108_108629


namespace smallest_possible_obscured_number_l108_108120

theorem smallest_possible_obscured_number (a b : ℕ) (cond : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  2 * a = b - 9 →
  42 + 25 + 56 + 10 * a + b = 4 * (4 + 2 + 2 + 5 + 5 + 6 + a + b) →
  10 * a + b = 79 :=
sorry

end smallest_possible_obscured_number_l108_108120


namespace area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l108_108463

theorem area_of_inscribed_rectangle_not_square (s : ℝ) : 
  (s > 0) ∧ (s < 1 / 2) :=
sorry

theorem area_of_inscribed_rectangle_is_square (s : ℝ) : 
  (s >= 1 / 2) ∧ (s < 1) :=
sorry

end area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l108_108463


namespace min_value_2a_b_c_l108_108290

theorem min_value_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * (a + b + c) + b * c = 4) : 
  2 * a + b + c ≥ 4 :=
sorry

end min_value_2a_b_c_l108_108290


namespace find_s_and_x_l108_108056

theorem find_s_and_x (s x t : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3.75) :
  s = 0.5 ∧ x = s / 2 → x = 0.25 :=
by
  sorry

end find_s_and_x_l108_108056


namespace car_total_distance_l108_108087

noncomputable def distance_first_segment (speed1 : ℝ) (time1 : ℝ) : ℝ :=
  speed1 * time1

noncomputable def distance_second_segment (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed2 * time2

noncomputable def distance_final_segment (speed3 : ℝ) (time3 : ℝ) : ℝ :=
  speed3 * time3

noncomputable def total_distance (d1 d2 d3 : ℝ) : ℝ :=
  d1 + d2 + d3

theorem car_total_distance :
  let d1 := distance_first_segment 65 2
  let d2 := distance_second_segment 80 1.5
  let d3 := distance_final_segment 50 2
  total_distance d1 d2 d3 = 350 :=
by
  sorry

end car_total_distance_l108_108087


namespace value_of_a4_l108_108772

theorem value_of_a4 (a : ℕ → ℕ) (r : ℕ) (h1 : ∀ n, a (n+1) = r * a n) (h2 : a 4 / a 2 - a 3 = 0) (h3 : r = 2) :
  a 4 = 8 :=
sorry

end value_of_a4_l108_108772


namespace geometric_progression_common_ratio_l108_108667

theorem geometric_progression_common_ratio (r : ℝ) :
  (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) ↔
  r = ( -1 + ((19 + 3 * Real.sqrt 33)^(1/3)) + ((19 - 3 * Real.sqrt 33)^(1/3)) ) / 3 :=
by
  sorry

end geometric_progression_common_ratio_l108_108667


namespace arithmetic_seq_2a9_a10_l108_108037

theorem arithmetic_seq_2a9_a10 (a : ℕ → ℕ) (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (arith_seq : ∀ n : ℕ, ∃ d : ℕ, a n = a 1 + (n - 1) * d) : 2 * a 9 - a 10 = 15 :=
by
  sorry

end arithmetic_seq_2a9_a10_l108_108037


namespace range_f_1_range_m_l108_108097

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 2) * (Real.log x / (2 * Real.log 2) - 1/2)

theorem range_f_1 (x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : 
  -1/8 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem range_m (m : ℝ) (x : ℝ) (h1 : 4 ≤ x) (h2 : x ≤ 16) (h3 : f x ≥ m * Real.log x / Real.log 2) :
  m ≤ 0 :=
sorry

end range_f_1_range_m_l108_108097


namespace remainder_of_max_6_multiple_no_repeated_digits_l108_108131

theorem remainder_of_max_6_multiple_no_repeated_digits (M : ℕ) 
  (hM : ∃ n, M = 6 * n) 
  (h_unique_digits : ∀ (d : ℕ), d ∈ (M.digits 10) → (M.digits 10).count d = 1) 
  (h_max_M : ∀ (k : ℕ), (∃ n, k = 6 * n) ∧ (∀ (d : ℕ), d ∈ (k.digits 10) → (k.digits 10).count d = 1) → k ≤ M) :
  M % 100 = 78 := 
sorry

end remainder_of_max_6_multiple_no_repeated_digits_l108_108131


namespace verify_option_a_l108_108703

-- Define Option A's condition
def option_a_condition (a : ℝ) : Prop :=
  2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2

-- State the theorem that Option A's factorization is correct
theorem verify_option_a (a : ℝ) : option_a_condition a := by sorry

end verify_option_a_l108_108703


namespace opposite_points_l108_108613

theorem opposite_points (A B : ℝ) (h1 : A = -B) (h2 : A < B) (h3 : abs (A - B) = 6.4) : A = -3.2 ∧ B = 3.2 :=
by
  sorry

end opposite_points_l108_108613


namespace sin_150_equals_half_l108_108330

theorem sin_150_equals_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by 
  sorry

end sin_150_equals_half_l108_108330


namespace smallest_base_b_l108_108651

theorem smallest_base_b (k : ℕ) (hk : k = 7) : ∃ (b : ℕ), b = 64 ∧ b^k > 4^20 := by
  sorry

end smallest_base_b_l108_108651


namespace solve_for_x_l108_108738

theorem solve_for_x :
  (∀ y : ℝ, 10 * x * y - 15 * y + 4 * x - 6 = 0) ↔ x = 3 / 2 :=
by
  sorry

end solve_for_x_l108_108738


namespace youngest_child_age_possible_l108_108379

theorem youngest_child_age_possible 
  (total_bill : ℝ) (mother_charge : ℝ) 
  (yearly_charge_per_child : ℝ) (minimum_charge_per_child : ℝ) 
  (num_children : ℤ) (children_total_bill : ℝ)
  (total_years : ℤ)
  (youngest_possible_age : ℤ) :
  total_bill = 15.30 →
  mother_charge = 6 →
  yearly_charge_per_child = 0.60 →
  minimum_charge_per_child = 0.90 →
  num_children = 3 →
  children_total_bill = total_bill - mother_charge →
  children_total_bill - num_children * minimum_charge_per_child = total_years * yearly_charge_per_child →
  total_years = 11 →
  youngest_possible_age = 1 :=
sorry

end youngest_child_age_possible_l108_108379


namespace value_of_x_l108_108693

theorem value_of_x (x : ℤ) (h : x + 3 = 4 ∨ x + 3 = -4) : x = 1 ∨ x = -7 := sorry

end value_of_x_l108_108693


namespace most_likely_outcomes_l108_108335

noncomputable def probability_boy_or_girl : ℚ := 1 / 2

noncomputable def probability_all_boys (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def probability_all_girls (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_3_girls_2_boys : ℚ := binom 5 3 * probability_boy_or_girl^5

noncomputable def probability_3_boys_2_girls : ℚ := binom 5 2 * probability_boy_or_girl^5

theorem most_likely_outcomes :
  probability_3_girls_2_boys = 5/16 ∧
  probability_3_boys_2_girls = 5/16 ∧
  probability_all_boys 5 = 1/32 ∧
  probability_all_girls 5 = 1/32 ∧
  (5/16 > 1/32) :=
by
  sorry

end most_likely_outcomes_l108_108335


namespace sum_zero_opposites_l108_108288

theorem sum_zero_opposites {a b : ℝ} (h : a + b = 0) : a = -b :=
by sorry

end sum_zero_opposites_l108_108288


namespace correct_statement_l108_108038

-- Define the necessary variables
variables {a b c : ℝ}

-- State the theorem including the condition and the conclusion
theorem correct_statement (h : a > b) : b - c < a - c :=
by linarith


end correct_statement_l108_108038


namespace company_percentage_increase_l108_108817

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

end company_percentage_increase_l108_108817


namespace april_roses_l108_108732

theorem april_roses (price_per_rose earnings roses_left : ℤ) 
  (h1 : price_per_rose = 4)
  (h2 : earnings = 36)
  (h3 : roses_left = 4) :
  4 + (earnings / price_per_rose) = 13 :=
by
  sorry

end april_roses_l108_108732


namespace vitamin_A_supplements_per_pack_l108_108070

theorem vitamin_A_supplements_per_pack {A x y : ℕ} (h1 : A * x = 119) (h2 : 17 * y = 119) : A = 7 :=
by
  sorry

end vitamin_A_supplements_per_pack_l108_108070


namespace frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l108_108431

-- Definitions based on the problem conditions
def machineA_first_class := 150
def machineA_total := 200
def machineB_first_class := 120
def machineB_total := 200
def total_products := machineA_total + machineB_total

-- Frequencies of first-class products
def frequency_machineA : ℚ := machineA_first_class / machineA_total
def frequency_machineB : ℚ := machineB_first_class / machineB_total

-- Values for chi-squared formula
def a := machineA_first_class
def b := machineA_total - machineA_first_class
def c := machineB_first_class
def d := machineB_total - machineB_first_class

-- Given formula for K^2
def K_squared : ℚ := (total_products * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof problem statements
theorem frequency_first_class_machineA_is_3_over_4 : frequency_machineA = 3 / 4 := by
  sorry

theorem frequency_first_class_machineB_is_3_over_5 : frequency_machineB = 3 / 5 := by
  sorry

theorem significant_quality_difference : K_squared > 6.635 := by
  sorry

end frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l108_108431


namespace beavers_working_on_home_l108_108705

noncomputable def initial_beavers : ℝ := 2.0
noncomputable def additional_beavers : ℝ := 1.0

theorem beavers_working_on_home : initial_beavers + additional_beavers = 3.0 :=
by
  sorry

end beavers_working_on_home_l108_108705


namespace ryan_flyers_l108_108746

theorem ryan_flyers (total_flyers : ℕ) (alyssa_flyers : ℕ) (scott_flyers : ℕ) (belinda_percentage : ℚ) (belinda_flyers : ℕ) (ryan_flyers : ℕ)
  (htotal : total_flyers = 200)
  (halyssa : alyssa_flyers = 67)
  (hscott : scott_flyers = 51)
  (hbelinda_percentage : belinda_percentage = 0.20)
  (hbelinda : belinda_flyers = belinda_percentage * total_flyers)
  (hryan : ryan_flyers = total_flyers - (alyssa_flyers + scott_flyers + belinda_flyers)) :
  ryan_flyers = 42 := by
    sorry

end ryan_flyers_l108_108746


namespace street_length_l108_108491

theorem street_length
  (time_minutes : ℕ)
  (speed_kmph : ℕ)
  (length_meters : ℕ)
  (h1 : time_minutes = 12)
  (h2 : speed_kmph = 9)
  (h3 : length_meters = 1800) :
  length_meters = (speed_kmph * 1000 / 60) * time_minutes :=
by sorry

end street_length_l108_108491


namespace value_of_x_l108_108852

theorem value_of_x (z : ℕ) (y : ℕ) (x : ℕ) 
  (h₁ : y = z / 5)
  (h₂ : x = y / 2)
  (h₃ : z = 60) : 
  x = 6 :=
by
  sorry

end value_of_x_l108_108852


namespace circle_passing_points_l108_108448

theorem circle_passing_points :
  ∃ (D E F : ℝ), 
    (25 + 1 + 5 * D + E + F = 0) ∧ 
    (36 + 6 * D + F = 0) ∧ 
    (1 + 1 - D + E + F = 0) ∧ 
    (∀ x y : ℝ, (x, y) = (5, 1) ∨ (x, y) = (6, 0) ∨ (x, y) = (-1, 1) → x^2 + y^2 + D * x + E * y + F = 0) → 
  x^2 + y^2 - 4 * x + 6 * y - 12 = 0 :=
by
  sorry

end circle_passing_points_l108_108448


namespace distinct_sequences_six_sided_die_rolled_six_times_l108_108186

theorem distinct_sequences_six_sided_die_rolled_six_times :
  let count := 6
  (count ^ 6 = 46656) :=
by
  let count := 6
  sorry

end distinct_sequences_six_sided_die_rolled_six_times_l108_108186


namespace vanilla_syrup_cost_l108_108841

theorem vanilla_syrup_cost :
  ∀ (unit_cost_drip : ℝ) (num_drip : ℕ)
    (unit_cost_espresso : ℝ) (num_espresso : ℕ)
    (unit_cost_latte : ℝ) (num_lattes : ℕ)
    (unit_cost_cold_brew : ℝ) (num_cold_brews : ℕ)
    (unit_cost_cappuccino : ℝ) (num_cappuccino : ℕ)
    (total_cost : ℝ) (vanilla_cost : ℝ),
  unit_cost_drip = 2.25 →
  num_drip = 2 →
  unit_cost_espresso = 3.50 →
  num_espresso = 1 →
  unit_cost_latte = 4.00 →
  num_lattes = 2 →
  unit_cost_cold_brew = 2.50 →
  num_cold_brews = 2 →
  unit_cost_cappuccino = 3.50 →
  num_cappuccino = 1 →
  total_cost = 25.00 →
  vanilla_cost =
    total_cost -
    ((unit_cost_drip * num_drip) +
    (unit_cost_espresso * num_espresso) +
    (unit_cost_latte * (num_lattes - 1)) +
    (unit_cost_cold_brew * num_cold_brews) +
    (unit_cost_cappuccino * num_cappuccino)) →
  vanilla_cost = 0.50 := sorry

end vanilla_syrup_cost_l108_108841


namespace g_neither_even_nor_odd_l108_108480

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 2) + 1/3

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) :=
by
  -- insert proof here
  sorry

end g_neither_even_nor_odd_l108_108480


namespace avg_amount_lost_per_loot_box_l108_108602

-- Define the conditions
def cost_per_loot_box : ℝ := 5
def avg_value_of_items : ℝ := 3.5
def total_amount_spent : ℝ := 40

-- Define the goal
theorem avg_amount_lost_per_loot_box : 
  (total_amount_spent / cost_per_loot_box) * (cost_per_loot_box - avg_value_of_items) / (total_amount_spent / cost_per_loot_box) = 1.5 := 
by 
  sorry

end avg_amount_lost_per_loot_box_l108_108602


namespace flower_counts_l108_108101

theorem flower_counts (R G Y : ℕ) : (R + G = 62) → (R + Y = 49) → (G + Y = 77) → R = 17 ∧ G = 45 ∧ Y = 32 :=
by
  intros h1 h2 h3
  sorry

end flower_counts_l108_108101


namespace min_value_f_l108_108646

noncomputable def f (x : ℝ) : ℝ := (8^x + 5) / (2^x + 1)

theorem min_value_f : ∃ x : ℝ, f x = 3 :=
sorry

end min_value_f_l108_108646


namespace pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l108_108411

-- Definitions based on the problem's conditions
def a (n : Nat) : Nat := n * n

def pos_count (n : Nat) : Nat :=
  List.length (List.filter (λ m : Nat => a m < n) (List.range (n + 1)))

def pos_pos_count (n : Nat) : Nat :=
  pos_count (pos_count n)

-- Theorem statements
theorem pos_count_a5_eq_2 : pos_count 5 = 2 := 
by
  -- Proof would go here
  sorry

theorem pos_pos_count_an_eq_n2 (n : Nat) : pos_pos_count n = n * n :=
by
  -- Proof would go here
  sorry

end pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l108_108411


namespace impossible_300_numbers_l108_108537

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end impossible_300_numbers_l108_108537


namespace area_of_MNFK_l108_108476

theorem area_of_MNFK (ABNF CMKD MNFK : ℝ) (BN : ℝ) (KD : ℝ) (ABMK : ℝ) (CDFN : ℝ)
  (h1 : BN = 8) (h2 : KD = 9) (h3 : ABMK = 25) (h4 : CDFN = 32) :
  MNFK = 31 :=
by
  have hx : 8 * (MNFK + 25) - 25 = 9 * (MNFK + 32) - 32 := sorry
  exact sorry

end area_of_MNFK_l108_108476


namespace find_value_of_m_l108_108302

theorem find_value_of_m (x m : ℤ) (h₁ : x = 2) (h₂ : y = m) (h₃ : 3 * x + 2 * y = 10) : m = 2 := 
by
  sorry

end find_value_of_m_l108_108302


namespace solution1_solution2_solution3_l108_108291

noncomputable def problem1 : Nat :=
  (1) * (2 - 1) * (2 + 1)

theorem solution1 : problem1 = 3 := by
  sorry

noncomputable def problem2 : Nat :=
  (2) * (2 + 1) * (2^2 + 1)

theorem solution2 : problem2 = 15 := by
  sorry

noncomputable def problem3 : Nat :=
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)

theorem solution3 : problem3 = 2^64 - 1 := by
  sorry

end solution1_solution2_solution3_l108_108291


namespace exterior_angle_polygon_l108_108258

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l108_108258


namespace cube_edge_length_l108_108801

-- Definitions based on given conditions
def paper_cost_per_kg : ℝ := 60
def paper_area_coverage_per_kg : ℝ := 20
def total_expenditure : ℝ := 1800
def surface_area_of_cube (a : ℝ) : ℝ := 6 * a^2

-- The main proof problem
theorem cube_edge_length :
  ∃ a : ℝ, surface_area_of_cube a = paper_area_coverage_per_kg * (total_expenditure / paper_cost_per_kg) ∧ a = 10 :=
by
  sorry

end cube_edge_length_l108_108801


namespace fraction_of_water_l108_108572

theorem fraction_of_water (total_weight sand_ratio water_weight gravel_weight : ℝ)
  (htotal : total_weight = 49.99999999999999)
  (hsand_ratio : sand_ratio = 1/2)
  (hwater : water_weight = total_weight - total_weight * sand_ratio - gravel_weight)
  (hgravel : gravel_weight = 15)
  : (water_weight / total_weight) = 1/5 :=
by
  sorry

end fraction_of_water_l108_108572


namespace polarEquationOfCircleCenter1_1Radius1_l108_108373

noncomputable def circleEquationInPolarCoordinates (θ : ℝ) : ℝ := 2 * Real.cos (θ - 1)

theorem polarEquationOfCircleCenter1_1Radius1 (ρ θ : ℝ) 
  (h : Real.sqrt ((ρ * Real.cos θ - Real.cos 1)^2 + (ρ * Real.sin θ - Real.sin 1)^2) = 1) :
  ρ = circleEquationInPolarCoordinates θ :=
by sorry

end polarEquationOfCircleCenter1_1Radius1_l108_108373


namespace greatest_k_l108_108083

noncomputable def n : ℕ := sorry
def k : ℕ := sorry

axiom d : ℕ → ℕ

axiom h1 : d n = 72
axiom h2 : d (5 * n) = 90

theorem greatest_k : ∃ k : ℕ, (∀ m : ℕ, m > k → ¬(5^m ∣ n)) ∧ 5^k ∣ n ∧ k = 3 :=
by
  sorry

end greatest_k_l108_108083


namespace part1_part2_l108_108052

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

-- (1) Given a = -1, prove that the inequality f(x, -1) ≤ 0 implies x ≤ -1/3
theorem part1 (x : ℝ) : (f x (-1) ≤ 0) ↔ (x ≤ -1/3) :=
by
  sorry

-- (2) Given f(x) ≥ 0 for all x ≥ -1, prove that the range for a is a ≤ -3 or a ≥ 1
theorem part2 (a : ℝ) : (∀ x, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l108_108052


namespace trig_identity_l108_108445

theorem trig_identity : 
  (2 * Real.sin (80 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l108_108445


namespace smallest_a₁_l108_108024

-- We define the sequence a_n and its recurrence relation
def a (n : ℕ) (a₁ : ℝ) : ℝ :=
  match n with
  | 0     => 0  -- this case is not used, but included for function completeness
  | 1     => a₁
  | (n+2) => 11 * a (n+1) a₁ - (n+2)

theorem smallest_a₁ : ∃ a₁ : ℝ, (a₁ = 21 / 100) ∧ ∀ n > 1, a n a₁ > 0 := 
  sorry

end smallest_a₁_l108_108024


namespace jason_worked_hours_on_saturday_l108_108370

def hours_jason_works (x y : ℝ) : Prop :=
  (4 * x + 6 * y = 88) ∧ (x + y = 18)

theorem jason_worked_hours_on_saturday (x y : ℝ) : hours_jason_works x y → y = 8 := 
by 
  sorry

end jason_worked_hours_on_saturday_l108_108370


namespace floor_x_floor_x_eq_42_l108_108447

theorem floor_x_floor_x_eq_42 (x : ℝ) : (⌊x * ⌊x⌋⌋ = 42) ↔ (7 ≤ x ∧ x < 43 / 6) :=
by sorry

end floor_x_floor_x_eq_42_l108_108447


namespace compound_interest_double_l108_108766

theorem compound_interest_double (t : ℕ) (r : ℝ) (n : ℕ) (P : ℝ) :
  r = 0.15 → n = 1 → (2 : ℝ) < (1 + r)^t → t ≥ 5 :=
by
  intros hr hn h
  sorry

end compound_interest_double_l108_108766


namespace find_prob_real_roots_l108_108926

-- Define the polynomial q(x)
def q (a : ℝ) (x : ℝ) : ℝ := x^4 + 3*a*x^3 + (3*a - 5)*x^2 + (-6*a + 4)*x - 3

-- Define the conditions for a to ensure all roots of the polynomial are real
noncomputable def all_roots_real_condition (a : ℝ) : Prop :=
  a ≤ -1/3 ∨ 1 ≤ a

-- Define the probability that given a in the interval [-12, 32] all q's roots are real
noncomputable def probability_real_roots : ℝ :=
  let total_length := 32 - (-12)
  let excluded_interval_length := 1 - (-1/3)
  let valid_interval_length := total_length - excluded_interval_length
  valid_interval_length / total_length

-- State the theorem
theorem find_prob_real_roots :
  probability_real_roots = 32 / 33 :=
sorry

end find_prob_real_roots_l108_108926


namespace top_card_is_joker_probability_l108_108963

theorem top_card_is_joker_probability :
  let totalCards := 54
  let jokerCards := 2
  let probability := (jokerCards : ℚ) / (totalCards : ℚ)
  probability = 1 / 27 :=
by
  sorry

end top_card_is_joker_probability_l108_108963


namespace sum_of_money_l108_108226

theorem sum_of_money (P R : ℝ) (h : (P * 2 * (R + 3) / 100) = (P * 2 * R / 100) + 300) : P = 5000 :=
by
    -- We are given that the sum of money put at 2 years SI rate is Rs. 300 more when rate is increased by 3%.
    sorry

end sum_of_money_l108_108226


namespace red_before_green_probability_l108_108368

open Classical

noncomputable def probability_red_before_green (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  let total_arrangements := (Nat.choose (total_chips - 1) green_chips)
  let favorable_arrangements := Nat.choose (total_chips - red_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem red_before_green_probability :
  probability_red_before_green 8 4 3 = 3 / 7 :=
sorry

end red_before_green_probability_l108_108368


namespace angle_triple_of_supplement_l108_108750

theorem angle_triple_of_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_of_supplement_l108_108750


namespace find_water_bottles_l108_108237

def water_bottles (W A : ℕ) :=
  A = W + 6 ∧ W + A = 54 → W = 24

theorem find_water_bottles (W A : ℕ) (h1 : A = W + 6) (h2 : W + A = 54) : W = 24 :=
by sorry

end find_water_bottles_l108_108237


namespace unknown_cube_edge_length_l108_108314

theorem unknown_cube_edge_length (a b c x : ℕ) (h_a : a = 6) (h_b : b = 10) (h_c : c = 12) : a^3 + b^3 + x^3 = c^3 → x = 8 :=
by
  sorry

end unknown_cube_edge_length_l108_108314


namespace total_passengers_l108_108350

theorem total_passengers (P : ℕ)
  (h1 : P / 12 + P / 8 + P / 3 + P / 6 + 35 = P) : 
  P = 120 :=
by
  sorry

end total_passengers_l108_108350


namespace factorize_expression_l108_108894

variable (m n : ℝ)

theorem factorize_expression : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 :=
by
  sorry

end factorize_expression_l108_108894


namespace range_of_a_l108_108392

noncomputable def f (x a b : ℝ) : ℝ := (2 * x^2 - a * x + b) * Real.log (x - 1)

theorem range_of_a (a b : ℝ) (h1 : ∀ x > 1, f x a b ≥ 0) : a ≤ 6 :=
by 
  let x := 2
  have hb_eq : b = 2 * a - 8 :=
    by sorry
  have ha_le_6 : a ≤ 6 :=
    by sorry
  exact ha_le_6

end range_of_a_l108_108392


namespace cost_price_of_book_l108_108160

theorem cost_price_of_book
(marked_price : ℝ)
(list_price : ℝ)
(cost_price : ℝ)
(h1 : marked_price = 69.85)
(h2 : list_price = marked_price * 0.85)
(h3 : list_price = cost_price * 1.25) :
cost_price = 65.75 :=
by
  sorry

end cost_price_of_book_l108_108160


namespace total_distance_traveled_l108_108951

noncomputable def travel_distance (speed : ℝ) (time : ℝ) (headwind : ℝ) : ℝ :=
  (speed - headwind) * time

theorem total_distance_traveled :
  let headwind := 5
  let eagle_speed := 15
  let eagle_time := 2.5
  let eagle_distance := travel_distance eagle_speed eagle_time headwind

  let falcon_speed := 46
  let falcon_time := 2.5
  let falcon_distance := travel_distance falcon_speed falcon_time headwind

  let pelican_speed := 33
  let pelican_time := 2.5
  let pelican_distance := travel_distance pelican_speed pelican_time headwind

  let hummingbird_speed := 30
  let hummingbird_time := 2.5
  let hummingbird_distance := travel_distance hummingbird_speed hummingbird_time headwind

  let hawk_speed := 45
  let hawk_time := 3
  let hawk_distance := travel_distance hawk_speed hawk_time headwind

  let swallow_speed := 25
  let swallow_time := 1.5
  let swallow_distance := travel_distance swallow_speed swallow_time headwind

  eagle_distance + falcon_distance + pelican_distance + hummingbird_distance + hawk_distance + swallow_distance = 410 :=
sorry

end total_distance_traveled_l108_108951


namespace number_of_correct_propositions_is_zero_l108_108677

-- Defining the propositions as functions
def proposition1 (f : ℝ → ℝ) (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
  (increasing_neg : ∀ x < 0, f x ≤ f (x + 1)) : Prop :=
  ∀ x1 x2, x1 ≤ x2 → f x1 ≤ f x2

def proposition2 (a b : ℝ) (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0) : Prop :=
  b^2 < 8 * a ∧ (a > 0 ∨ (a = 0 ∧ b = 0))

def proposition3 : Prop :=
  ∀ x, (x ≥ 1 → (x^2 - 2 * x - 3) ≥ (x^2 - 2 * (x + 1) - 3))

-- The main theorem to prove
theorem number_of_correct_propositions_is_zero :
  ∀ (f : ℝ → ℝ)
    (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
    (increasing_neg : ∀ x < 0, f x ≤ f (x + 1))
    (a b : ℝ)
    (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0),
    (¬ proposition1 f increasing_pos increasing_neg ∧
     ¬ proposition2 a b no_intersection ∧
     ¬ proposition3) :=
by
  sorry

end number_of_correct_propositions_is_zero_l108_108677


namespace area_of_scalene_right_triangle_l108_108806

noncomputable def area_of_triangle_DEF (DE EF : ℝ) (h1 : DE > 0) (h2 : EF > 0) (h3 : DE / EF = 3) (h4 : DE^2 + EF^2 = 16) : ℝ :=
1 / 2 * DE * EF

theorem area_of_scalene_right_triangle (DE EF : ℝ) 
  (h1 : DE > 0)
  (h2 : EF > 0)
  (h3 : DE / EF = 3)
  (h4 : DE^2 + EF^2 = 16) :
  area_of_triangle_DEF DE EF h1 h2 h3 h4 = 2.4 :=
sorry

end area_of_scalene_right_triangle_l108_108806


namespace abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l108_108765

theorem abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one 
  (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1 / a + 1 / b + 1 / c) : 
  (a = 1) ∨ (b = 1) ∨ (c = 1) :=
by
  sorry

end abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l108_108765


namespace count_balanced_integers_l108_108991

def is_balanced (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3) = d1 + (d2 + d3) ∧ (100 ≤ n) ∧ (n ≤ 999)

theorem count_balanced_integers : ∃ c, c = 330 ∧ ∀ n, 100 ≤ n ∧ n ≤ 999 → is_balanced n ↔ c = 330 :=
sorry

end count_balanced_integers_l108_108991


namespace range_of_m_is_leq_3_l108_108406

noncomputable def is_range_of_m (m : ℝ) : Prop :=
  ∀ x : ℝ, 5^x + 3 > m

theorem range_of_m_is_leq_3 (m : ℝ) : is_range_of_m m ↔ m ≤ 3 :=
by
  sorry

end range_of_m_is_leq_3_l108_108406


namespace distance_between_stations_l108_108349

theorem distance_between_stations (x : ℕ) 
  (h1 : ∃ (x : ℕ), ∀ t : ℕ, (t * 16 = x ∧ t * 21 = x + 60)) :
  2 * x + 60 = 444 :=
by sorry

end distance_between_stations_l108_108349


namespace tan_315_eq_neg_one_l108_108626

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l108_108626


namespace probability_value_at_least_75_cents_l108_108860

-- Given conditions
def box_contains (pennies nickels quarters : ℕ) : Prop :=
  pennies = 4 ∧ nickels = 3 ∧ quarters = 5

def draw_without_replacement (total_coins : ℕ) (drawn_coins : ℕ) : Prop :=
  total_coins = 12 ∧ drawn_coins = 5

def equal_probability (chosen_probability : ℚ) (total_coins : ℕ) : Prop :=
  chosen_probability = 1/total_coins

-- Probability that the value of coins drawn is at least 75 cents
theorem probability_value_at_least_75_cents
  (pennies nickels quarters total_coins drawn_coins : ℕ)
  (chosen_probability : ℚ) :
  box_contains pennies nickels quarters →
  draw_without_replacement total_coins drawn_coins →
  equal_probability chosen_probability total_coins →
  chosen_probability = 1/792 :=
by
  intros
  sorry

end probability_value_at_least_75_cents_l108_108860


namespace sum_place_values_of_specified_digits_l108_108870

def numeral := 95378637153370261

def place_values_of_3s := [3 * 100000000000, 3 * 10]
def place_values_of_7s := [7 * 10000000000, 7 * 1000000, 7 * 100]
def place_values_of_5s := [5 * 10000000000000, 5 * 1000, 5 * 10000, 5 * 1]

def sum_place_values (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def sum_of_place_values := 
  sum_place_values place_values_of_3s + 
  sum_place_values place_values_of_7s + 
  sum_place_values place_values_of_5s

theorem sum_place_values_of_specified_digits :
  sum_of_place_values = 350077055735 :=
by
  sorry

end sum_place_values_of_specified_digits_l108_108870


namespace train_speed_l108_108320

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 630) (h_time : time = 36) :
  (length / 1000) / (time / 3600) = 63 :=
by
  rw [h_length, h_time]
  sorry

end train_speed_l108_108320


namespace remainder_of_expression_l108_108142

theorem remainder_of_expression :
  (8 * 7^19 + 1^19) % 9 = 3 :=
  by
    sorry

end remainder_of_expression_l108_108142


namespace remaining_black_area_after_five_changes_l108_108968

-- Define a function that represents the change process
noncomputable def remaining_black_area (iterations : ℕ) : ℚ :=
  (3 / 4) ^ iterations

-- Define the original problem statement as a theorem in Lean
theorem remaining_black_area_after_five_changes :
  remaining_black_area 5 = 243 / 1024 :=
by
  sorry

end remaining_black_area_after_five_changes_l108_108968


namespace tan_sum_pi_over_4_l108_108008

open Real

theorem tan_sum_pi_over_4 {α : ℝ} (h₁ : cos (2 * α) + sin α * (2 * sin α - 1) = 2 / 5) (h₂ : π / 4 < α) (h₃ : α < π) : 
    tan (α + π / 4) = 1 / 7 := sorry

end tan_sum_pi_over_4_l108_108008


namespace jake_reaches_ground_later_by_2_seconds_l108_108641

noncomputable def start_floor : ℕ := 12
noncomputable def steps_per_floor : ℕ := 25
noncomputable def jake_steps_per_second : ℕ := 3
noncomputable def elevator_B_time : ℕ := 90

noncomputable def total_steps_jake := (start_floor - 1) * steps_per_floor
noncomputable def time_jake := (total_steps_jake + jake_steps_per_second - 1) / jake_steps_per_second
noncomputable def time_difference := time_jake - elevator_B_time

theorem jake_reaches_ground_later_by_2_seconds :
  time_difference = 2 := by
  sorry

end jake_reaches_ground_later_by_2_seconds_l108_108641


namespace y_worked_days_l108_108045

-- Definitions based on conditions
def work_rate_x := 1 / 20 -- x's work rate (W per day)
def work_rate_y := 1 / 16 -- y's work rate (W per day)

def remaining_work_by_x := 5 * work_rate_x -- Work finished by x after y left
def total_work := 1 -- Assume the total work W is 1 unit for simplicity

def days_y_worked (d : ℝ) := d * work_rate_y + remaining_work_by_x = total_work

-- The statement we need to prove
theorem y_worked_days :
  (exists d : ℕ, days_y_worked d ∧ d = 15) :=
sorry

end y_worked_days_l108_108045


namespace shirts_made_today_l108_108153

def shirts_per_minute : ℕ := 8
def working_minutes : ℕ := 2

theorem shirts_made_today (h1 : shirts_per_minute = 8) (h2 : working_minutes = 2) : shirts_per_minute * working_minutes = 16 := by
  sorry

end shirts_made_today_l108_108153


namespace calc_sqrt_expr_l108_108202

theorem calc_sqrt_expr :
  (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end calc_sqrt_expr_l108_108202


namespace trains_cross_each_other_in_given_time_l108_108333

noncomputable def trains_crossing_time (length1 length2 speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1 := (speed1_kmph * 1000) / 3600
  let speed2 := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1 + speed2
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem trains_cross_each_other_in_given_time :
  trains_crossing_time 300 400 36 18 = 46.67 :=
by
  -- expected proof here
  sorry

end trains_cross_each_other_in_given_time_l108_108333


namespace probability_individual_selected_l108_108456

/-- Given a population of 8 individuals, the probability that each 
individual is selected in a simple random sample of size 4 is 1/2. -/
theorem probability_individual_selected :
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  probability = (1 : ℚ) / 2 :=
by
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  sorry

end probability_individual_selected_l108_108456


namespace find_area_of_triangle_ABQ_l108_108680

noncomputable def area_triangle_ABQ {A B C P Q R : Type*}
  (AP PB : ℝ) (area_ABC area_ABQ : ℝ) (h_areas_equal : area_ABQ = 15 / 2)
  (h_triangle_area : area_ABC = 15) (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) : Prop := area_ABQ = 15

theorem find_area_of_triangle_ABQ
  (A B C P Q R : Type*) (AP PB : ℝ)
  (h_triangle_area : area_ABC = 15)
  (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) (h_areas_equal : area_ABQ = 15 / 2) :
  area_ABQ = 15 := sorry

end find_area_of_triangle_ABQ_l108_108680


namespace marla_errand_time_l108_108272

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

end marla_errand_time_l108_108272


namespace max_ab_ac_bc_l108_108753

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : 
    ab + ac + bc <= 8 :=
sorry

end max_ab_ac_bc_l108_108753


namespace solution_set_inequality_l108_108663

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) : 
  (x - 1) / x > 1 → x < 0 := 
by 
  sorry

end solution_set_inequality_l108_108663


namespace total_number_of_employees_l108_108277
  
def part_time_employees : ℕ := 2041
def full_time_employees : ℕ := 63093
def total_employees : ℕ := part_time_employees + full_time_employees

theorem total_number_of_employees : total_employees = 65134 := by
  sorry

end total_number_of_employees_l108_108277


namespace sofia_total_time_l108_108151

-- Definitions for the conditions
def laps : ℕ := 5
def track_length : ℕ := 400  -- in meters
def speed_first_100 : ℕ := 4  -- meters per second
def speed_remaining_300 : ℕ := 5  -- meters per second

-- Times taken for respective distances
def time_first_100 (distance speed : ℕ) : ℕ := distance / speed
def time_remaining_300 (distance speed : ℕ) : ℕ := distance / speed

def time_one_lap : ℕ := time_first_100 100 speed_first_100 + time_remaining_300 300 speed_remaining_300
def total_time_seconds : ℕ := laps * time_one_lap
def total_time_minutes : ℕ := 7
def total_time_extra_seconds : ℕ := 5

-- Problem statement
theorem sofia_total_time :
  total_time_seconds = total_time_minutes * 60 + total_time_extra_seconds :=
by
  sorry

end sofia_total_time_l108_108151


namespace sequence_a_11_l108_108002

theorem sequence_a_11 (a : ℕ → ℚ) (arithmetic_seq : ℕ → ℚ)
  (h1 : a 3 = 2)
  (h2 : a 7 = 1)
  (h_arith : ∀ n, arithmetic_seq n = 1 / (a n + 1))
  (arith_property : ∀ n, arithmetic_seq (n + 1) - arithmetic_seq n = arithmetic_seq (n + 2) - arithmetic_seq (n + 1)) :
  a 11 = 1 / 2 :=
by
  sorry

end sequence_a_11_l108_108002


namespace cat_weight_problem_l108_108227

variable (female_cat_weight male_cat_weight : ℕ)

theorem cat_weight_problem
  (h1 : male_cat_weight = 2 * female_cat_weight)
  (h2 : female_cat_weight + male_cat_weight = 6) :
  female_cat_weight = 2 :=
by
  sorry

end cat_weight_problem_l108_108227


namespace books_withdrawn_is_15_l108_108146

-- Define the initial condition
def initial_books : ℕ := 250

-- Define the books taken out on Tuesday
def books_taken_out_tuesday : ℕ := 120

-- Define the books returned on Wednesday
def books_returned_wednesday : ℕ := 35

-- Define the books left in library on Thursday
def books_left_thursday : ℕ := 150

-- Define the problem: Determine the number of books withdrawn on Thursday
def books_withdrawn_thursday : ℕ :=
  (initial_books - books_taken_out_tuesday + books_returned_wednesday) - books_left_thursday

-- The statement we want to prove
theorem books_withdrawn_is_15 : books_withdrawn_thursday = 15 := by sorry

end books_withdrawn_is_15_l108_108146


namespace solution_of_xyz_l108_108548

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l108_108548


namespace relationship_between_abc_l108_108782

open Real

-- Define the constants for the problem
noncomputable def a : ℝ := sqrt 2023 - sqrt 2022
noncomputable def b : ℝ := sqrt 2022 - sqrt 2021
noncomputable def c : ℝ := sqrt 2021 - sqrt 2020

-- State the theorem we want to prove
theorem relationship_between_abc : c > b ∧ b > a := 
sorry

end relationship_between_abc_l108_108782


namespace sum_of_three_numbers_l108_108744

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) 
 (h_median : b = 10) 
 (h_mean_least : (a + b + c) / 3 = a + 8)
 (h_mean_greatest : (a + b + c) / 3 = c - 20) : 
 a + b + c = 66 :=
by 
  sorry

end sum_of_three_numbers_l108_108744


namespace dice_five_prob_l108_108780

-- Define a standard six-sided die probability
def prob_five : ℚ := 1 / 6

-- Define the probability of all four dice showing five
def prob_all_five : ℚ := prob_five * prob_five * prob_five * prob_five

-- State the theorem
theorem dice_five_prob : prob_all_five = 1 / 1296 := by
  sorry

end dice_five_prob_l108_108780


namespace sum_of_three_primes_eq_86_l108_108511

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_three_primes_eq_86 (a b c : ℕ) (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (h_sum : a + b + c = 86) :
  (a, b, c) = (2, 5, 79) ∨ (a, b, c) = (2, 11, 73) ∨ (a, b, c) = (2, 13, 71) ∨ (a, b, c) = (2, 17, 67) ∨
  (a, b, c) = (2, 23, 61) ∨ (a, b, c) = (2, 31, 53) ∨ (a, b, c) = (2, 37, 47) ∨ (a, b, c) = (2, 41, 43) :=
by
  sorry

end sum_of_three_primes_eq_86_l108_108511


namespace problem_statement_l108_108488

theorem problem_statement :
  let pct := 208 / 100
  let initial_value := 1265
  let step1 := pct * initial_value
  let step2 := step1 ^ 2
  let answer := step2 / 12
  answer = 576857.87 := 
by 
  sorry

end problem_statement_l108_108488


namespace min_distance_squared_l108_108937

theorem min_distance_squared (a b c d : ℝ) (e : ℝ) (h₀ : e = Real.exp 1) 
  (h₁ : (a - 2 * Real.exp a) / b = 1) (h₂ : (2 - c) / d = 1) :
  (a - c)^2 + (b - d)^2 = 8 := by
  sorry

end min_distance_squared_l108_108937


namespace necessary_but_not_sufficient_l108_108249

-- Define \(\frac{1}{x} < 2\) and \(x > \frac{1}{2}\)
def condition1 (x : ℝ) : Prop := 1 / x < 2
def condition2 (x : ℝ) : Prop := x > 1 / 2

-- Theorem stating that condition1 is necessary but not sufficient for condition2
theorem necessary_but_not_sufficient (x : ℝ) : condition1 x → condition2 x ↔ true :=
sorry

end necessary_but_not_sufficient_l108_108249


namespace xy_range_l108_108468

open Real

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 / x + 3 * y + 4 / y = 10) : 
  1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end xy_range_l108_108468


namespace trapezoid_area_l108_108714

noncomputable def area_of_trapezoid : ℝ :=
  let y1 := 12
  let y2 := 5
  let x1 := 12 / 2
  let x2 := 5 / 2
  ((x1 + x2) / 2) * (y1 - y2)

theorem trapezoid_area : area_of_trapezoid = 29.75 := by
  sorry

end trapezoid_area_l108_108714


namespace share_equally_l108_108589

variable (Emani Howard : ℕ)
axiom h1 : Emani = 150
axiom h2 : Emani = Howard + 30

theorem share_equally : (Emani + Howard) / 2 = 135 :=
by sorry

end share_equally_l108_108589


namespace one_thirds_in_nine_halves_l108_108484

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l108_108484


namespace sum_of_distinct_digits_l108_108704

theorem sum_of_distinct_digits
  (w x y z : ℕ)
  (h1 : y + w = 10)
  (h2 : x + y = 9)
  (h3 : w + z = 10)
  (h4 : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z)
  (hw : w < 10) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  w + x + y + z = 20 := sorry

end sum_of_distinct_digits_l108_108704


namespace fraction_identity_l108_108126

theorem fraction_identity :
  (1721^2 - 1714^2 : ℚ) / (1728^2 - 1707^2) = 1 / 3 :=
by
  sorry

end fraction_identity_l108_108126


namespace joan_balloons_l108_108728

def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2
def total_balloons : ℕ := 16

theorem joan_balloons : sally_balloons + jessica_balloons = 7 ∧ total_balloons = 16 → total_balloons - (sally_balloons + jessica_balloons) = 9 :=
by
  sorry

end joan_balloons_l108_108728


namespace one_is_sum_of_others_l108_108630

theorem one_is_sum_of_others {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : |a - b| ≥ c) (h2 : |b - c| ≥ a) (h3 : |c - a| ≥ b) :
    a = b + c ∨ b = a + c ∨ c = a + b :=
sorry

end one_is_sum_of_others_l108_108630


namespace volume_of_rectangular_solid_l108_108685

theorem volume_of_rectangular_solid
  (a b c : ℝ)
  (h1 : a * b = 3)
  (h2 : a * c = 5)
  (h3 : b * c = 15) :
  a * b * c = 15 :=
sorry

end volume_of_rectangular_solid_l108_108685


namespace problem_statement_l108_108935

open Real

theorem problem_statement :
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2/3) - Real.log 4 = 50.6938 :=
by
  sorry

end problem_statement_l108_108935


namespace B_is_left_of_A_l108_108050

-- Define the coordinates of points A and B
def A_coord : ℚ := 5 / 8
def B_coord : ℚ := 8 / 13

-- The statement we want to prove: B is to the left of A
theorem B_is_left_of_A : B_coord < A_coord :=
  by {
    sorry
  }

end B_is_left_of_A_l108_108050


namespace solve_equation_l108_108193

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 2) = (x - 2) → (x = 2 ∨ x = 1 / 3) :=
by
  intro x
  intro h
  sorry

end solve_equation_l108_108193


namespace distance_at_40_kmph_l108_108552

theorem distance_at_40_kmph (x y : ℕ) 
  (h1 : x + y = 250) 
  (h2 : x / 40 + y / 60 = 6) : 
  x = 220 :=
by
  sorry

end distance_at_40_kmph_l108_108552


namespace concert_tickets_full_price_revenue_l108_108310

theorem concert_tickets_full_price_revenue :
  ∃ (f p d : ℕ), f + d = 200 ∧ f * p + d * (p / 3) = 2688 ∧ f * p = 2128 :=
by
  -- We need to find the solution steps are correct to establish the existence
  sorry

end concert_tickets_full_price_revenue_l108_108310


namespace quadratic_roots_ratio_l108_108547

theorem quadratic_roots_ratio (k : ℝ) (k1 k2 : ℝ) (a b : ℝ) 
  (h_roots : ∀ x : ℝ, k * x * x + (1 - 6 * k) * x + 8 = 0 ↔ (x = a ∨ x = b))
  (h_ab : a ≠ b)
  (h_cond : a / b + b / a = 3 / 7)
  (h_ks : k^1 - 6 * (k1 + k2) + 8 = 0)
  (h_vieta : k1 + k2 = 200 / 36 ∧ k1 * k2 = 49 / 36) : 
  (k1 / k2 + k2 / k1 = 6.25) :=
by sorry

end quadratic_roots_ratio_l108_108547


namespace exceeds_alpha_beta_l108_108075

noncomputable def condition (α β p q : ℝ) : Prop :=
  q < 50 ∧ α > 0 ∧ β > 0 ∧ p > 0 ∧ q > 0

theorem exceeds_alpha_beta (α β p q : ℝ) (h : condition α β p q) :
  (1 + p / 100) * (1 - q / 100) > 1 → p > 100 * q / (100 - q) := by
  sorry

end exceeds_alpha_beta_l108_108075


namespace sin_B_sin_C_l108_108590

open Real

noncomputable def triangle_condition (A B C : ℝ) (a b c : ℝ) : Prop :=
  cos (2 * A) - 3 * cos (B + C) = 1 ∧
  (1 / 2) * b * c * sin A = 5 * sqrt 3 ∧
  b = 5

theorem sin_B_sin_C {A B C a b c : ℝ} (h : triangle_condition A B C a b c) :
  (sin B) * (sin C) = 5 / 7 := 
sorry

end sin_B_sin_C_l108_108590


namespace right_triangle_consecutive_sides_l108_108019

theorem right_triangle_consecutive_sides (n : ℕ) (n_pos : 0 < n) :
    (n+1)^2 + n^2 = (n+2)^2 ↔ (n = 3) :=
by
  sorry

end right_triangle_consecutive_sides_l108_108019


namespace solve_for_x_l108_108543

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 :=
by
  sorry

end solve_for_x_l108_108543


namespace profit_shares_difference_l108_108857

theorem profit_shares_difference (total_profit : ℝ) (share_ratio_x share_ratio_y : ℝ) 
  (hx : share_ratio_x = 1/2) (hy : share_ratio_y = 1/3) (profit : ℝ):
  total_profit = 500 → profit = (total_profit * share_ratio_x) / ((share_ratio_x + share_ratio_y)) - (total_profit * share_ratio_y) / ((share_ratio_x + share_ratio_y)) → profit = 100 :=
by
  intros
  sorry

end profit_shares_difference_l108_108857


namespace base_conversion_l108_108810

theorem base_conversion (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 7 * A = 5 * B) : 8 * A + B = 47 :=
by
  sorry

end base_conversion_l108_108810


namespace Ben_win_probability_l108_108465

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end Ben_win_probability_l108_108465


namespace variance_of_data_set_l108_108209

theorem variance_of_data_set (a : ℝ) (ha : (1 + a + 3 + 6 + 7) / 5 = 4) : 
  (1 / 5) * ((1 - 4)^2 + (a - 4)^2 + (3 - 4)^2 + (6 - 4)^2 + (7 - 4)^2) = 24 / 5 :=
by
  sorry

end variance_of_data_set_l108_108209


namespace series_sum_proof_l108_108740

noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, if n % 3 = 0 then 1 / (27 ^ (n / 3)) * (5 / 9) else 0

theorem series_sum_proof : infinite_series_sum = 15 / 26 :=
  sorry

end series_sum_proof_l108_108740


namespace expression_value_l108_108874

theorem expression_value : 
  (2 ^ 1501 + 5 ^ 1502) ^ 2 - (2 ^ 1501 - 5 ^ 1502) ^ 2 = 20 * 10 ^ 1501 := 
by
  sorry

end expression_value_l108_108874


namespace negation_of_proposition_l108_108068

theorem negation_of_proposition (m : ℤ) : 
  (¬ (∃ x : ℤ, x^2 + 2*x + m ≤ 0)) ↔ ∀ x : ℤ, x^2 + 2*x + m > 0 :=
sorry

end negation_of_proposition_l108_108068


namespace series_sum_eq_l108_108715

theorem series_sum_eq :
  (1^25 + 2^24 + 3^23 + 4^22 + 5^21 + 6^20 + 7^19 + 8^18 + 9^17 + 10^16 + 
  11^15 + 12^14 + 13^13 + 14^12 + 15^11 + 16^10 + 17^9 + 18^8 + 19^7 + 20^6 + 
  21^5 + 22^4 + 23^3 + 24^2 + 25^1) = 66071772829247409 := 
by
  sorry

end series_sum_eq_l108_108715


namespace calculate_new_average_weight_l108_108717

noncomputable def new_average_weight (original_team_weight : ℕ) (num_original_players : ℕ) 
 (new_player1_weight : ℕ) (new_player2_weight : ℕ) (num_new_players : ℕ) : ℕ :=
 (original_team_weight + new_player1_weight + new_player2_weight) / (num_original_players + num_new_players)

theorem calculate_new_average_weight : 
  new_average_weight 847 7 110 60 2 = 113 := 
by 
sorry

end calculate_new_average_weight_l108_108717


namespace total_right_handed_players_is_60_l108_108204

def total_players : ℕ := 70
def throwers : ℕ := 40
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed_players : ℕ := right_handed_throwers + right_handed_non_throwers

theorem total_right_handed_players_is_60 : total_right_handed_players = 60 := by
  sorry

end total_right_handed_players_is_60_l108_108204


namespace cans_collected_is_232_l108_108788

-- Definitions of the conditions
def total_students : ℕ := 30
def half_students : ℕ := total_students / 2
def cans_per_half_student : ℕ := 12
def remaining_students : ℕ := 13
def cans_per_remaining_student : ℕ := 4

-- Calculate total cans collected
def total_cans_collected : ℕ := (half_students * cans_per_half_student) + (remaining_students * cans_per_remaining_student)

-- The theorem to be proved
theorem cans_collected_is_232 : total_cans_collected = 232 := by
  -- Proof would go here
  sorry

end cans_collected_is_232_l108_108788


namespace interval_of_n_l108_108797

theorem interval_of_n (n : ℕ) (h_pos : 0 < n) (h_lt_2000 : n < 2000) 
                      (h_div_99999999 : 99999999 % n = 0) (h_div_999999 : 999999 % (n + 6) = 0) : 
                      801 ≤ n ∧ n ≤ 1200 :=
by {
  sorry
}

end interval_of_n_l108_108797


namespace range_of_s_l108_108684

noncomputable def s (x : ℝ) := 1 / (2 + x)^3

theorem range_of_s :
  Set.range s = {y : ℝ | y < 0} ∪ {y : ℝ | y > 0} :=
by
  sorry

end range_of_s_l108_108684


namespace income_ratio_l108_108030

variable (U B: ℕ) -- Uma's and Bala's incomes
variable (x: ℕ)  -- Common multiplier for expenditures
variable (savings_amt: ℕ := 2000)  -- Savings amount for both
variable (ratio_expenditure_uma : ℕ := 7)
variable (ratio_expenditure_bala : ℕ := 6)
variable (uma_income : ℕ := 16000)
variable (bala_expenditure: ℕ)

-- Conditions of the problem
-- Uma's Expenditure Calculation
axiom ua_exp_calc : savings_amt = uma_income - ratio_expenditure_uma * x
-- Bala's Expenditure Calculation
axiom bala_income_calc : savings_amt = B - ratio_expenditure_bala * x

theorem income_ratio (h1: U = uma_income) (h2: B = bala_expenditure):
  U * ratio_expenditure_bala = B * ratio_expenditure_uma :=
sorry

end income_ratio_l108_108030


namespace different_picture_size_is_correct_l108_108638

-- Define constants and conditions
def memory_card_picture_capacity := 3000
def single_picture_size := 8
def different_picture_capacity := 4000

-- Total memory card capacity in megabytes
def total_capacity := memory_card_picture_capacity * single_picture_size

-- The size of each different picture
def different_picture_size := total_capacity / different_picture_capacity

-- The theorem to prove
theorem different_picture_size_is_correct :
  different_picture_size = 6 := 
by
  -- We include 'sorry' here to bypass actual proof
  sorry

end different_picture_size_is_correct_l108_108638


namespace sum_inequality_l108_108022

theorem sum_inequality (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2) :
  (x + y + z) * (x⁻¹ + y⁻¹ + z⁻¹) ≥ 6 * (x / (y + z) + y / (z + x) + z / (x + y)) := sorry

end sum_inequality_l108_108022


namespace range_of_m_l108_108324

theorem range_of_m {m : ℝ} (h : ∀ x : ℝ, (3 * m - 1) ^ x = (3 * m - 1) ^ x ∧ (3 * m - 1) > 0 ∧ (3 * m - 1) < 1) :
  1 / 3 < m ∧ m < 2 / 3 :=
by
  sorry

end range_of_m_l108_108324


namespace carlson_fraction_l108_108748

-- Define variables
variables (n m k p T : ℝ)

theorem carlson_fraction (h1 : k = 0.6 * n)
                         (h2 : p = 2.5 * m)
                         (h3 : T = n * m + k * p) :
                         k * p / T = 3 / 5 := by
  -- Omitted proof
  sorry

end carlson_fraction_l108_108748


namespace roja_speed_l108_108790

theorem roja_speed (R : ℕ) (h1 : 3 + R = 7) : R = 7 - 3 :=
by sorry

end roja_speed_l108_108790


namespace range_of_m_l108_108662

variable (x m : ℝ)

def alpha (x : ℝ) : Prop := x ≤ -5
def beta (x m : ℝ) : Prop := 2 * m - 3 ≤ x ∧ x ≤ 2 * m + 1

theorem range_of_m (x : ℝ) : (∀ x, beta x m → alpha x) → m ≤ -3 := by
  sorry

end range_of_m_l108_108662


namespace circles_intersect_probability_l108_108712

noncomputable def probability_circles_intersect : ℝ :=
  sorry

theorem circles_intersect_probability :
  probability_circles_intersect = (5 * Real.sqrt 2 - 7) / 4 :=
  sorry

end circles_intersect_probability_l108_108712


namespace hotdog_eating_ratio_l108_108737

variable (rate_first rate_second rate_third total_hotdogs time_minutes : ℕ)
variable (rate_ratio : ℕ)

def rate_first_eq : rate_first = 10 := by sorry
def rate_second_eq : rate_second = 3 * rate_first := by sorry
def total_hotdogs_eq : total_hotdogs = 300 := by sorry
def time_minutes_eq : time_minutes = 5 := by sorry
def rate_third_eq : rate_third = total_hotdogs / time_minutes := by sorry

theorem hotdog_eating_ratio :
  rate_ratio = rate_third / rate_second :=
  by sorry

end hotdog_eating_ratio_l108_108737


namespace value_subtracted_from_result_l108_108902

theorem value_subtracted_from_result (N V : ℕ) (hN : N = 1152) (h: (N / 6) - V = 3) : V = 189 :=
by
  sorry

end value_subtracted_from_result_l108_108902


namespace Loris_needs_more_books_l108_108854

noncomputable def books_needed (Loris Darryl Lamont : ℕ) :=
  (Lamont - Loris)

theorem Loris_needs_more_books
  (darryl_books: ℕ)
  (lamont_books: ℕ)
  (loris_books_total: ℕ)
  (total_books: ℕ)
  (h1: lamont_books = 2 * darryl_books)
  (h2: darryl_books = 20)
  (h3: loris_books_total + darryl_books + lamont_books = total_books)
  (h4: total_books = 97) :
  books_needed loris_books_total darryl_books lamont_books = 3 :=
sorry

end Loris_needs_more_books_l108_108854


namespace remaining_painting_time_l108_108043

-- Define the given conditions as Lean definitions
def total_rooms : ℕ := 9
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 5

-- Formulate the main theorem to prove the remaining time is 32 hours
theorem remaining_painting_time : 
  (total_rooms - rooms_painted) * hours_per_room = 32 := 
by 
  sorry

end remaining_painting_time_l108_108043


namespace find_abc_l108_108636

theorem find_abc (a b c : ℝ) (ha : a + 1 / b = 5)
                             (hb : b + 1 / c = 2)
                             (hc : c + 1 / a = 3) :
    a * b * c = 10 + 3 * Real.sqrt 11 :=
sorry

end find_abc_l108_108636


namespace least_8_heavy_three_digit_l108_108238

def is_8_heavy (n : ℕ) : Prop :=
  n % 8 > 6

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem least_8_heavy_three_digit : ∃ n : ℕ, is_three_digit n ∧ is_8_heavy n ∧ ∀ m : ℕ, is_three_digit m ∧ is_8_heavy m → n ≤ m := 
sorry

end least_8_heavy_three_digit_l108_108238


namespace quadratic_function_distinct_zeros_l108_108829

theorem quadratic_function_distinct_zeros (a : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) ↔ (a ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 0) := 
by
  sorry

end quadratic_function_distinct_zeros_l108_108829


namespace goose_eggs_calculation_l108_108974

noncomputable def goose_eggs_total (E : ℕ) : Prop :=
  let hatched := (2/3) * E
  let survived_first_month := (3/4) * hatched
  let survived_first_year := (2/5) * survived_first_month
  survived_first_year = 110

theorem goose_eggs_calculation :
  goose_eggs_total 3300 :=
by
  have h1 : (2 : ℝ) / (3 : ℝ) ≠ 0 := by norm_num
  have h2 : (3 : ℝ) / (4 : ℝ) ≠ 0 := by norm_num
  have h3 : (2 : ℝ) / (5 : ℝ) ≠ 0 := by norm_num
  sorry

end goose_eggs_calculation_l108_108974


namespace chord_length_l108_108964

theorem chord_length (a b : ℝ) (M : ℝ) (h : M * M = a * b) : ∃ AB : ℝ, AB = 2 * Real.sqrt (a * b) :=
by
  sorry

end chord_length_l108_108964


namespace number_of_apples_and_erasers_l108_108793

def totalApplesAndErasers (a e : ℕ) : Prop :=
  a + e = 84

def applesPerFriend (a : ℕ) : ℕ :=
  a / 3

def erasersPerTeacher (e : ℕ) : ℕ :=
  e / 2

theorem number_of_apples_and_erasers (a e : ℕ) (h : totalApplesAndErasers a e) :
  applesPerFriend a = a / 3 ∧ erasersPerTeacher e = e / 2 :=
by
  sorry

end number_of_apples_and_erasers_l108_108793


namespace ages_of_people_l108_108104

-- Define types
variable (A M B C : ℕ)

-- Define conditions as hypotheses
def conditions : Prop :=
  A = 2 * M ∧
  A = 4 * B ∧
  M = A - 10 ∧
  C = B + 3 ∧
  C = M / 2

-- Define what we want to prove
theorem ages_of_people :
  (conditions A M B C) →
  A = 20 ∧
  M = 10 ∧
  B = 2 ∧
  C = 5 :=
by
  sorry

end ages_of_people_l108_108104


namespace mary_total_nickels_l108_108691

theorem mary_total_nickels (n1 n2 : ℕ) (h1 : n1 = 7) (h2 : n2 = 5) : n1 + n2 = 12 := by
  sorry

end mary_total_nickels_l108_108691


namespace encoded_integer_one_less_l108_108460

theorem encoded_integer_one_less (BDF BEA BFB EAB : ℕ)
  (hBDF : BDF = 1 * 7^2 + 3 * 7 + 6)
  (hBEA : BEA = 1 * 7^2 + 5 * 7 + 0)
  (hBFB : BFB = 1 * 7^2 + 5 * 7 + 1)
  (hEAB : EAB = 5 * 7^2 + 0 * 7 + 1)
  : EAB - 1 = 245 :=
by
  sorry

end encoded_integer_one_less_l108_108460


namespace car_rental_cost_l108_108966

def daily_rental_rate : ℝ := 29
def per_mile_charge : ℝ := 0.08
def rental_duration : ℕ := 1
def distance_driven : ℝ := 214.0

theorem car_rental_cost : 
  (daily_rental_rate * rental_duration + per_mile_charge * distance_driven) = 46.12 := 
by 
  sorry

end car_rental_cost_l108_108966


namespace arithmetic_sequence_evaluation_l108_108544

theorem arithmetic_sequence_evaluation :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by 
-- Proof omitted
sorry

end arithmetic_sequence_evaluation_l108_108544


namespace roof_ratio_l108_108774

theorem roof_ratio (L W : ℝ) 
  (h1 : L * W = 784) 
  (h2 : L - W = 42) : 
  L / W = 4 := by 
  sorry

end roof_ratio_l108_108774


namespace approx_num_fish_in_pond_l108_108114

noncomputable def numFishInPond (tagged_in_second: ℕ) (total_second: ℕ) (tagged: ℕ) : ℕ :=
  tagged * total_second / tagged_in_second

theorem approx_num_fish_in_pond :
  numFishInPond 2 50 50 = 1250 := by
  sorry

end approx_num_fish_in_pond_l108_108114


namespace distance_from_center_to_chord_l108_108721

theorem distance_from_center_to_chord (a b : ℝ) : 
  ∃ d : ℝ, d = (1/4) * |a - b| := 
sorry

end distance_from_center_to_chord_l108_108721


namespace train_speed_correct_l108_108129

-- Definitions for the given conditions
def train_length : ℝ := 320
def time_to_cross : ℝ := 6

-- The speed of the train
def train_speed : ℝ := 53.33

-- The proof statement
theorem train_speed_correct : train_speed = train_length / time_to_cross :=
by
  sorry

end train_speed_correct_l108_108129


namespace machine_transport_equation_l108_108567

theorem machine_transport_equation (x : ℝ) :
  (∀ (rateA rateB : ℝ), rateB = rateA + 60 → (500 / rateA = 800 / rateB) → rateA = x → rateB = x + 60) :=
by
  sorry

end machine_transport_equation_l108_108567


namespace quadratic_interlaced_roots_l108_108946

theorem quadratic_interlaced_roots
  (p1 p2 q1 q2 : ℝ)
  (h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  ∃ (r1 r2 s1 s2 : ℝ),
    (r1^2 + p1 * r1 + q1 = 0) ∧
    (r2^2 + p1 * r2 + q1 = 0) ∧
    (s1^2 + p2 * s1 + q2 = 0) ∧
    (s2^2 + p2 * s2 + q2 = 0) ∧
    (r1 < s1 ∧ s1 < r2 ∨ s1 < r1 ∧ r1 < s2) :=
sorry

end quadratic_interlaced_roots_l108_108946


namespace son_present_age_l108_108923

variable (S M : ℕ)

-- Condition 1: M = S + 20
def man_age_relation (S M : ℕ) : Prop := M = S + 20

-- Condition 2: In two years, the man's age will be twice the age of his son
def age_relation_in_two_years (S M : ℕ) : Prop := M + 2 = 2*(S + 2)

theorem son_present_age : 
  ∀ (S M : ℕ), man_age_relation S M → age_relation_in_two_years S M → S = 18 :=
by
  intros S M h1 h2
  sorry

end son_present_age_l108_108923


namespace determine_a_l108_108280

theorem determine_a (a : ℝ): (∃ b : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → a = 8 := 
by
  sorry

end determine_a_l108_108280


namespace g_at_5_l108_108071

def g (x : ℝ) : ℝ := sorry -- Placeholder for the function definition, typically provided in further context

theorem g_at_5 : g 5 = 3 / 4 :=
by
  -- Given condition as a hypothesis
  have h : ∀ x: ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1 := sorry
  sorry  -- Full proof should go here

end g_at_5_l108_108071


namespace range_of_m_l108_108787

open Real

theorem range_of_m (a b m : ℝ) (x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 9 / b = 1) :
  a + b ≥ -x^2 + 4 * x + 18 - m ↔ m ≥ 6 :=
by sorry

end range_of_m_l108_108787


namespace product_of_two_numbers_l108_108992

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 + y^2 = 200) : x * y = 28 :=
by
  sorry

end product_of_two_numbers_l108_108992


namespace first_number_is_45_l108_108081

theorem first_number_is_45 (a b : ℕ) (h1 : a / gcd a b = 3) (h2 : b / gcd a b = 4) (h3 : lcm a b = 180) : a = 45 := by
  sorry

end first_number_is_45_l108_108081


namespace mangoes_total_l108_108784

theorem mangoes_total (Dilan Ashley Alexis : ℕ) (h1 : Alexis = 4 * (Dilan + Ashley)) (h2 : Ashley = 2 * Dilan) (h3 : Alexis = 60) : Dilan + Ashley + Alexis = 75 :=
by
  sorry

end mangoes_total_l108_108784


namespace molecular_weight_proof_l108_108886

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

def molecular_weight (n_N n_H n_I : ℕ) : ℝ :=
  n_N * atomic_weight_N + n_H * atomic_weight_H + n_I * atomic_weight_I

theorem molecular_weight_proof : molecular_weight 1 4 1 = 144.95 :=
by {
  sorry
}

end molecular_weight_proof_l108_108886


namespace total_cost_verification_l108_108105

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 8.38

theorem total_cost_verification 
  (sc : sandwich_cost = 2.45)
  (sd : soda_cost = 0.87)
  (ns : num_sandwiches = 2)
  (nd : num_sodas = 4) :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost := 
sorry

end total_cost_verification_l108_108105


namespace value_of_collection_l108_108096

theorem value_of_collection (n : ℕ) (v : ℕ → ℕ) (h1 : n = 20) 
    (h2 : v 5 = 20) (h3 : ∀ k1 k2, v k1 = v k2) : v n = 80 :=
by
  sorry

end value_of_collection_l108_108096


namespace initial_rate_of_commission_is_4_l108_108168

noncomputable def initial_commission_rate (B : ℝ) (x : ℝ) : Prop :=
  B * (x / 100) = 0.8 * B * (5 / 100)

theorem initial_rate_of_commission_is_4 (B : ℝ) (hB : B > 0) :
  initial_commission_rate B 4 :=
by
  unfold initial_commission_rate
  sorry

end initial_rate_of_commission_is_4_l108_108168


namespace find_smaller_number_l108_108833

theorem find_smaller_number (x : ℕ) (h : 3 * x + 4 * x = 420) : 3 * x = 180 :=
by
  sorry

end find_smaller_number_l108_108833


namespace ratio_wrong_to_correct_l108_108362

theorem ratio_wrong_to_correct (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36) (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 :=
by {
  -- Proof will go here
  sorry
}

end ratio_wrong_to_correct_l108_108362


namespace evaluate_expression_l108_108979

theorem evaluate_expression : 60 + (105 / 15) + (25 * 16) - 250 + (324 / 9) ^ 2 = 1513 := by
  sorry

end evaluate_expression_l108_108979


namespace word_value_at_l108_108650

def letter_value (c : Char) : ℕ :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1 else 0

def word_value (s : String) : ℕ :=
  let sum_values := s.toList.map letter_value |>.sum
  sum_values * s.length

theorem word_value_at : word_value "at" = 42 := by
  sorry

end word_value_at_l108_108650


namespace equivalence_l108_108568

-- Non-computable declaration to avoid the computational complexity.
noncomputable def is_isosceles_right_triangle (x₁ x₂ : Complex) : Prop :=
  x₂ = x₁ * Complex.I ∨ x₁ = x₂ * Complex.I

-- Definition of the polynomial roots condition.
def roots_form_isosceles_right_triangle (a b : Complex) : Prop :=
  ∃ x₁ x₂ : Complex,
    x₁ + x₂ = -a ∧
    x₁ * x₂ = b ∧
    is_isosceles_right_triangle x₁ x₂

-- Main theorem statement that matches the mathematical equivalency.
theorem equivalence (a b : Complex) : a^2 = 2*b ∧ b ≠ 0 ↔ roots_form_isosceles_right_triangle a b :=
sorry

end equivalence_l108_108568


namespace insufficient_data_l108_108941

variable (M P O : ℝ)

theorem insufficient_data
  (h1 : M < P)
  (h2 : O > M) :
  ¬(P < O) ∧ ¬(O < P) ∧ ¬(P = O) := 
sorry

end insufficient_data_l108_108941


namespace salt_percentage_l108_108565

theorem salt_percentage (salt water : ℝ) (h_salt : salt = 10) (h_water : water = 40) : 
  salt / water = 0.2 :=
by
  sorry

end salt_percentage_l108_108565


namespace seven_digit_numbers_count_l108_108471

/-- Given a six-digit phone number represented by six digits A, B, C, D, E, F:
- There are 7 positions where a new digit can be inserted: before A, between each pair of consecutive digits, and after F.
- Each of these positions can be occupied by any of the 10 digits (0 through 9).
The number of seven-digit numbers that can be formed by adding one digit to the six-digit phone number is 70. -/
theorem seven_digit_numbers_count (A B C D E F : ℕ) (hA : 0 ≤ A ∧ A < 10) (hB : 0 ≤ B ∧ B < 10) 
  (hC : 0 ≤ C ∧ C < 10) (hD : 0 ≤ D ∧ D < 10) (hE : 0 ≤ E ∧ E < 10) (hF : 0 ≤ F ∧ F < 10) : 
  ∃ n : ℕ, n = 70 :=
sorry

end seven_digit_numbers_count_l108_108471


namespace arithmetic_sequence_general_formula_l108_108559

variable (a : ℤ) 

def is_arithmetic_sequence (a1 a2 a3 : ℤ) : Prop :=
  2 * a2 = a1 + a3

theorem arithmetic_sequence_general_formula :
  ∀ {a1 a2 a3 : ℤ}, is_arithmetic_sequence a1 a2 a3 → a1 = a - 1 ∧ a2 = a + 1 ∧ a3 = 2 * a + 3 → 
  ∀ n : ℕ, a_n = 2 * n - 3
:= by
  sorry

end arithmetic_sequence_general_formula_l108_108559


namespace sequence_uniquely_determined_l108_108228

theorem sequence_uniquely_determined (a : ℕ → ℝ) (p q : ℝ) (a0 a1 : ℝ)
  (h : ∀ n, a (n + 2) = p * a (n + 1) + q * a n)
  (h0 : a 0 = a0)
  (h1 : a 1 = a1) :
  ∀ n, ∃! a_n, a n = a_n :=
sorry

end sequence_uniquely_determined_l108_108228


namespace jongkook_points_l108_108932

-- Define the conditions in the problem
def num_questions_solved_each : ℕ := 18
def shinhye_points : ℕ := 100
def jongkook_correct_6_points : ℕ := 8
def jongkook_correct_5_points : ℕ := 6
def points_per_question_6 : ℕ := 6
def points_per_question_5 : ℕ := 5
def jongkook_wrong_questions : ℕ := num_questions_solved_each - jongkook_correct_6_points - jongkook_correct_5_points

-- Calculate Jongkook's points from correct answers
def jongkook_points_from_6 : ℕ := jongkook_correct_6_points * points_per_question_6
def jongkook_points_from_5 : ℕ := jongkook_correct_5_points * points_per_question_5

-- Calculate total points
def jongkook_total_points : ℕ := jongkook_points_from_6 + jongkook_points_from_5

-- Prove that Jongkook's total points is 78
theorem jongkook_points : jongkook_total_points = 78 :=
by
  sorry

end jongkook_points_l108_108932


namespace n_energetic_all_n_specific_energetic_constraints_l108_108479

-- Proof Problem 1
theorem n_energetic_all_n (a b c : ℕ) (n : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : Nat.gcd a (Nat.gcd b c) = 1) 
(h4 : ∀ n ≥ 1, (a^n + b^n + c^n) % (a + b + c) = 0) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4) := sorry

-- Proof Problem 2
theorem specific_energetic_constraints (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
(h3 : Nat.gcd a (Nat.gcd b c) = 1) 
(h4 : (a^2004 + b^2004 + c^2004) % (a + b + c) = 0)
(h5 : (a^2005 + b^2005 + c^2005) % (a + b + c) = 0) 
(h6 : (a^2007 + b^2007 + c^2007) % (a + b + c) ≠ 0) :
  false := sorry

end n_energetic_all_n_specific_energetic_constraints_l108_108479


namespace solve_system_of_equations_l108_108553

theorem solve_system_of_equations
  (a b c : ℝ) (x y z : ℝ)
  (h1 : x + y = a)
  (h2 : y + z = b)
  (h3 : z + x = c) :
  x = (a + c - b) / 2 ∧ y = (a + b - c) / 2 ∧ z = (b + c - a) / 2 :=
by
  sorry

end solve_system_of_equations_l108_108553


namespace percentage_error_in_calculated_area_l108_108036

theorem percentage_error_in_calculated_area 
  (s : ℝ) 
  (measured_side : ℝ) 
  (h : measured_side = s * 1.04) :
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 8.16 :=
by
  sorry

end percentage_error_in_calculated_area_l108_108036


namespace solve_for_t_l108_108576

theorem solve_for_t (t : ℚ) :
  (t+2) * (4*t-4) = (4*t-6) * (t+3) + 3 → t = 7/2 :=
by {
  sorry
}

end solve_for_t_l108_108576


namespace sum_of_squares_not_7_mod_8_l108_108069

theorem sum_of_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 :=
sorry

end sum_of_squares_not_7_mod_8_l108_108069


namespace smallest_x_y_z_sum_l108_108459

theorem smallest_x_y_z_sum :
  ∃ x y z : ℝ, x + 3*y + 6*z = 1 ∧ x*y + 2*x*z + 6*y*z = -8 ∧ x*y*z = 2 ∧ x + y + z = -(8/3) := 
sorry

end smallest_x_y_z_sum_l108_108459


namespace notebooks_difference_l108_108210

noncomputable def price_more_than_dime (p : ℝ) : Prop := p > 0.10
noncomputable def payment_equation (nL nN : ℕ) (p : ℝ) : Prop :=
  (nL * p = 2.10 ∧ nN * p = 2.80)

theorem notebooks_difference (nL nN : ℕ) (p : ℝ) (h1 : price_more_than_dime p) (h2 : payment_equation nL nN p) :
  nN - nL = 2 :=
by sorry

end notebooks_difference_l108_108210


namespace dagger_example_l108_108308

def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

theorem dagger_example : dagger 5 8 3 4 = 15 := by
  sorry

end dagger_example_l108_108308


namespace each_interior_angle_of_regular_octagon_l108_108154

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l108_108154


namespace quadratic_solution_identity_l108_108363

theorem quadratic_solution_identity (a b : ℤ) (h : (1 : ℤ)^2 + a * 1 + 2 * b = 0) : 2 * a + 4 * b = -2 := by
  sorry

end quadratic_solution_identity_l108_108363


namespace first_stopover_distance_l108_108074

theorem first_stopover_distance 
  (total_distance : ℕ) 
  (second_stopover_distance : ℕ) 
  (distance_after_second_stopover : ℕ) :
  total_distance = 436 → 
  second_stopover_distance = 236 → 
  distance_after_second_stopover = 68 →
  second_stopover_distance - (total_distance - second_stopover_distance - distance_after_second_stopover) = 104 :=
by
  intros
  sorry

end first_stopover_distance_l108_108074


namespace combined_list_correct_l108_108481

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25
def combined_list : ℕ := james_friends + john_friends - shared_friends

theorem combined_list_correct :
  combined_list = 275 :=
by
  sorry

end combined_list_correct_l108_108481


namespace total_earnings_of_a_b_c_l108_108442

theorem total_earnings_of_a_b_c 
  (days_a days_b days_c : ℕ)
  (ratio_a ratio_b ratio_c : ℕ)
  (wage_c : ℕ) 
  (h_ratio : ratio_a * wage_c = 3 * (3 + 4 + 5))
  (h_ratio_a_b : ratio_b = 4 * wage_c / 5 * ratio_a / 60)
  (h_ratio_b_c : ratio_b = 4 * wage_c / 5 * ratio_c / 60):
  (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c) = 1480 := 
  by
    sorry

end total_earnings_of_a_b_c_l108_108442


namespace find_m_l108_108652

theorem find_m (x y m : ℝ) (opp_sign: y = -x) 
  (h1 : 4 * x + 2 * y = 3 * m) 
  (h2 : 3 * x + y = m + 2) : 
  m = 1 :=
by 
  -- Placeholder for the steps to prove the theorem
  sorry

end find_m_l108_108652


namespace fans_received_all_items_l108_108496

def multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, m = n * k

theorem fans_received_all_items :
  (∀ n, multiple_of 100 n → multiple_of 40 n ∧ multiple_of 60 n ∧ multiple_of 24 n ∧ n ≤ 7200 → ∃ k, n = 600 * k) →
  (∃ k : ℕ, 7200 / 600 = k ∧ k = 12) :=
by
  sorry

end fans_received_all_items_l108_108496


namespace probability_all_boxes_non_empty_equals_4_over_9_l108_108584

structure PaintingPlacement :=
  (paintings : Finset ℕ)
  (boxes : Finset ℕ)
  (num_paintings : paintings.card = 4)
  (num_boxes : boxes.card = 3)

noncomputable def probability_non_empty_boxes (pp : PaintingPlacement) : ℚ :=
  let total_outcomes := 3^4
  let favorable_outcomes := Nat.choose 4 2 * Nat.factorial 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_all_boxes_non_empty_equals_4_over_9
  (pp : PaintingPlacement) : pp.paintings.card = 4 → pp.boxes.card = 3 →
  probability_non_empty_boxes pp = 4 / 9 :=
by
  intros h1 h2
  sorry

end probability_all_boxes_non_empty_equals_4_over_9_l108_108584


namespace volume_frustum_correct_l108_108365

noncomputable def volume_of_frustum : ℚ :=
  let V_original := (1 / 3 : ℚ) * (16^2) * 10
  let V_smaller := (1 / 3 : ℚ) * (8^2) * 5
  V_original - V_smaller

theorem volume_frustum_correct :
  volume_of_frustum = 2240 / 3 :=
by
  sorry

end volume_frustum_correct_l108_108365


namespace problem_l108_108747

variable (x y : ℝ)

theorem problem
  (h : (3 * x + 1) ^ 2 + |y - 3| = 0) :
  (x + 2 * y) * (x - 2 * y) + (x + 2 * y) ^ 2 - x * (2 * x + 3 * y) = -1 :=
sorry

end problem_l108_108747


namespace solve_ordered_pair_l108_108779

theorem solve_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x^2 - y = (x - 2) + (y - 2)) :
  (x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5) :=
  sorry

end solve_ordered_pair_l108_108779


namespace niko_total_profit_l108_108327

noncomputable def calculate_total_profit : ℝ :=
  let pairs := 9
  let price_per_pair := 2
  let discount_rate := 0.10
  let shipping_cost := 5
  let profit_4_pairs := 0.25
  let profit_5_pairs := 0.20
  let tax_rate := 0.05
  let cost_socks := pairs * price_per_pair
  let discount := discount_rate * cost_socks
  let cost_after_discount := cost_socks - discount
  let total_cost := cost_after_discount + shipping_cost
  let resell_price_4_pairs := (price_per_pair * (1 + profit_4_pairs)) * 4
  let resell_price_5_pairs := (price_per_pair * (1 + profit_5_pairs)) * 5
  let total_resell_price := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax := tax_rate * total_resell_price
  let total_resell_price_after_tax := total_resell_price + sales_tax
  let total_profit := total_resell_price_after_tax - total_cost
  total_profit

theorem niko_total_profit : calculate_total_profit = 0.85 :=
by
  sorry

end niko_total_profit_l108_108327


namespace tan_7pi_over_6_l108_108668

noncomputable def tan_periodic (θ : ℝ) : Prop :=
  ∀ k : ℤ, Real.tan (θ + k * Real.pi) = Real.tan θ

theorem tan_7pi_over_6 : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 :=
by
  sorry

end tan_7pi_over_6_l108_108668


namespace sufficient_but_not_necessary_l108_108671

theorem sufficient_but_not_necessary (m : ℕ) :
  m = 9 → m > 8 ∧ ∃ k : ℕ, k > 8 ∧ k ≠ 9 :=
by
  sorry

end sufficient_but_not_necessary_l108_108671


namespace poly_has_two_distinct_negative_real_roots_l108_108384

-- Definition of the polynomial equation
def poly_eq (p x : ℝ) : Prop :=
  x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1 = 0

-- Theorem statement that needs to be proved
theorem poly_has_two_distinct_negative_real_roots (p : ℝ) :
  p > 1 → ∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly_eq p x1 ∧ poly_eq p x2 :=
by
  sorry

end poly_has_two_distinct_negative_real_roots_l108_108384


namespace tiffany_lives_problem_l108_108084

/-- Tiffany's lives problem -/
theorem tiffany_lives_problem (L : ℤ) (h1 : 43 - L + 27 = 56) : L = 14 :=
by {
  sorry
}

end tiffany_lives_problem_l108_108084


namespace sufficient_but_not_necessary_condition_l108_108798

open Real

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (x = y → |x| = |y|) ∧ (|x| = |y| → x = y) = false :=
by
  sorry

end sufficient_but_not_necessary_condition_l108_108798


namespace percentage_of_boys_is_90_l108_108856

variables (B G : ℕ)

def total_children : ℕ := 100
def future_total_children : ℕ := total_children + 100
def percentage_girls : ℕ := 5
def girls_after_increase : ℕ := future_total_children * percentage_girls / 100
def boys_after_increase : ℕ := total_children - girls_after_increase

theorem percentage_of_boys_is_90 :
  B + G = total_children →
  G = girls_after_increase →
  B = total_children - G →
  (B:ℚ) / total_children * 100 = 90 :=
by
  sorry

end percentage_of_boys_is_90_l108_108856


namespace simplify_and_evaluate_expression_l108_108500

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 5 + 1) : 
  ( ( (x^2 - 1) / x ) / (1 + 1 / x) ) = Real.sqrt 5 :=
by 
  sorry

end simplify_and_evaluate_expression_l108_108500


namespace find_b6_l108_108003

def fib (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

theorem find_b6 (b : ℕ → ℕ) (b1 b2 : ℕ)
  (h1 : b 1 = b1) (h2 : b 2 = b2) (h3 : b 5 = 55)
  (hfib : fib b) : b 6 = 84 :=
  sorry

end find_b6_l108_108003


namespace xy_difference_l108_108713

theorem xy_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end xy_difference_l108_108713


namespace billy_restaurant_bill_l108_108674

def adults : ℕ := 2
def children : ℕ := 5
def meal_cost : ℕ := 3

def total_people : ℕ := adults + children
def total_bill : ℕ := total_people * meal_cost

theorem billy_restaurant_bill : total_bill = 21 := 
by
  -- This is the placeholder for the proof.
  sorry

end billy_restaurant_bill_l108_108674


namespace min_value_3x_4y_l108_108341

theorem min_value_3x_4y
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + 4 * y = 21 :=
sorry

end min_value_3x_4y_l108_108341


namespace equivalent_single_discount_l108_108631

theorem equivalent_single_discount (P : ℝ) (hP : 0 < P) : 
    let first_discount : ℝ := 0.15
    let second_discount : ℝ := 0.25
    let single_discount : ℝ := 0.3625
    (1 - first_discount) * (1 - second_discount) * P = (1 - single_discount) * P := by
    sorry

end equivalent_single_discount_l108_108631


namespace problem_statement_l108_108098

theorem problem_statement (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end problem_statement_l108_108098


namespace valid_votes_for_candidate_a_l108_108303

theorem valid_votes_for_candidate_a (total_votes : ℕ) (invalid_percentage : ℝ) (candidate_a_percentage : ℝ) (valid_votes_a : ℝ) :
  total_votes = 560000 ∧ invalid_percentage = 0.15 ∧ candidate_a_percentage = 0.80 →
  valid_votes_a = (candidate_a_percentage * (1 - invalid_percentage) * total_votes) := 
sorry

end valid_votes_for_candidate_a_l108_108303


namespace sequence_unbounded_l108_108005

theorem sequence_unbounded 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a n = |a (n + 1) - a (n + 2)|)
  (h2 : 0 < a 0)
  (h3 : 0 < a 1)
  (h4 : a 0 ≠ a 1) :
  ¬ ∃ M : ℝ, ∀ n, |a n| ≤ M := 
sorry

end sequence_unbounded_l108_108005


namespace eq_or_neg_eq_of_eq_frac_l108_108761

theorem eq_or_neg_eq_of_eq_frac (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h : a^2 + b^3 / a = b^2 + a^3 / b) :
  a = b ∨ a = -b :=
by
  sorry

end eq_or_neg_eq_of_eq_frac_l108_108761


namespace find_point_on_curve_l108_108013

theorem find_point_on_curve :
  ∃ P : ℝ × ℝ, (P.1^3 - P.1 + 3 = P.2) ∧ (3 * P.1^2 - 1 = 2) ∧ (P = (1, 3) ∨ P = (-1, 3)) :=
sorry

end find_point_on_curve_l108_108013


namespace p_q_false_of_not_or_l108_108808

variables (p q : Prop)

theorem p_q_false_of_not_or (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by {
  sorry
}

end p_q_false_of_not_or_l108_108808


namespace total_heartbeats_during_race_l108_108473

namespace Heartbeats

def avg_heart_beats_per_minute : ℕ := 160
def pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 20

theorem total_heartbeats_during_race :
  (race_distance_miles * pace_minutes_per_mile * avg_heart_beats_per_minute = 19200) :=
by
  sorry

end Heartbeats

end total_heartbeats_during_race_l108_108473


namespace quadratic_inequality_solution_l108_108522

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -1/3 < x ∧ x < 1 → -3 * x^2 + 8 * x + 1 < 0 :=
by
  intro x
  intro h
  sorry

end quadratic_inequality_solution_l108_108522


namespace lcm_of_two_numbers_l108_108512

variable (a b hcf lcm : ℕ)

theorem lcm_of_two_numbers (ha : a = 330) (hb : b = 210) (hhcf : Nat.gcd a b = 30) :
  Nat.lcm a b = 2310 := by
  sorry

end lcm_of_two_numbers_l108_108512


namespace students_in_grades_v_vi_l108_108814

theorem students_in_grades_v_vi (n a b c p q : ℕ) (h1 : n = 100*a + 10*b + c)
  (h2 : a * b * c = p) (h3 : (p / 10) * (p % 10) = q) : n = 144 :=
sorry

end students_in_grades_v_vi_l108_108814


namespace combined_age_of_staff_l108_108608

/--
In a school, the average age of a class of 50 students is 25 years. 
The average age increased by 2 years when the ages of 5 additional 
staff members, including the teacher, are also taken into account. 
Prove that the combined age of these 5 staff members is 235 years.
-/
theorem combined_age_of_staff 
    (n_students : ℕ) (avg_age_students : ℕ) (n_staff : ℕ) (avg_age_total : ℕ)
    (h1 : n_students = 50) 
    (h2 : avg_age_students = 25) 
    (h3 : n_staff = 5) 
    (h4 : avg_age_total = 27) :
  n_students * avg_age_students + (n_students + n_staff) * avg_age_total - 
  n_students * avg_age_students = 235 :=
by
  sorry

end combined_age_of_staff_l108_108608


namespace mark_initial_money_l108_108332

theorem mark_initial_money (X : ℝ) 
  (h1 : X = (1/2) * X + 14 + (1/3) * X + 16) : X = 180 := 
  by
  sorry

end mark_initial_money_l108_108332


namespace complex_subtraction_l108_108372

def z1 : ℂ := 3 + (1 : ℂ)
def z2 : ℂ := 2 - (1 : ℂ)

theorem complex_subtraction : z1 - z2 = 1 + 2 * (1 : ℂ) :=
by
  sorry

end complex_subtraction_l108_108372


namespace g_at_10_l108_108743

noncomputable def g (n : ℕ) : ℝ := sorry

axiom g_definition : g 2 = 4
axiom g_recursive : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (3 * g (2 * m) + g (2 * n)) / 4

theorem g_at_10 : g 10 = 64 := sorry

end g_at_10_l108_108743


namespace simplify_and_evaluate_problem_l108_108867

noncomputable def problem_expression (a : ℤ) : ℚ :=
  (1 - (3 : ℚ) / (a + 1)) / ((a^2 - 4 * a + 4 : ℚ) / (a + 1))

theorem simplify_and_evaluate_problem :
  ∀ (a : ℤ), -2 ≤ a ∧ a ≤ 2 → a ≠ -1 → a ≠ 2 →
  (problem_expression a = 1 / (a - 2 : ℚ)) ∧
  (a = 0 → problem_expression a = -1 / 2) ∧
  (a = 1 → problem_expression a = -1) :=
sorry

end simplify_and_evaluate_problem_l108_108867


namespace cows_eat_husk_l108_108734

theorem cows_eat_husk :
  ∀ (cows : ℕ) (days : ℕ) (husk_per_cow : ℕ),
    cows = 45 →
    days = 45 →
    husk_per_cow = 1 →
    (cows * husk_per_cow = 45) :=
by
  intros cows days husk_per_cow h_cows h_days h_husk_per_cow
  sorry

end cows_eat_husk_l108_108734


namespace salary_increase_l108_108093

theorem salary_increase (x : ℝ) (y : ℝ) :
  (1000 : ℝ) * 80 + 50 = y → y - (50 + 80 * x) = 80 :=
by
  intros h
  sorry

end salary_increase_l108_108093


namespace gap_between_rails_should_be_12_24_mm_l108_108348

noncomputable def initial_length : ℝ := 15
noncomputable def temperature_initial : ℝ := -8
noncomputable def temperature_max : ℝ := 60
noncomputable def expansion_coefficient : ℝ := 0.000012
noncomputable def change_in_temperature : ℝ := temperature_max - temperature_initial
noncomputable def final_length : ℝ := initial_length * (1 + expansion_coefficient * change_in_temperature)
noncomputable def gap : ℝ := (final_length - initial_length) * 1000  -- converted to mm

theorem gap_between_rails_should_be_12_24_mm
  : gap = 12.24 := by
  sorry

end gap_between_rails_should_be_12_24_mm_l108_108348


namespace polynomial_operations_l108_108616

-- Define the given options for M, N, and P
def A (x : ℝ) : ℝ := 2 * x - 6
def B (x : ℝ) : ℝ := 3 * x + 5
def C (x : ℝ) : ℝ := -5 * x - 21

-- Define the original expression and its simplified form
def original_expr (M N : ℝ → ℝ) (x : ℝ) : ℝ :=
  2 * M x - 3 * N x

-- Define the simplified target expression
def simplified_expr (x : ℝ) : ℝ := -5 * x - 21

theorem polynomial_operations :
  ∀ (M N P : ℝ → ℝ),
  (original_expr M N = simplified_expr) →
  (M = A ∨ N = B ∨ P = C)
:= by
  intros M N P H
  sorry

end polynomial_operations_l108_108616


namespace original_square_area_l108_108540

-- Definitions based on the given problem conditions
variable (s : ℝ) (A : ℝ)
def is_square (s : ℝ) : Prop := s > 0
def oblique_projection (s : ℝ) (A : ℝ) : Prop :=
  (A = s^2 ∨ A = 4^2) ∧ s = 4

-- The theorem statement based on the problem question and correct answer
theorem original_square_area :
  is_square s →
  oblique_projection s A →
  ∃ A, A = 16 ∨ A = 64 := 
sorry

end original_square_area_l108_108540


namespace range_of_a_l108_108138

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 7 * x1^2 - (a + 13) * x1 + a^2 - a - 2 = 0 ∧
                 7 * x2^2 - (a + 13) * x2 + a^2 - a - 2 = 0 ∧
                 0 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < 2) →
  (-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4) :=
by
  intro h
  sorry

end range_of_a_l108_108138


namespace average_jump_difference_l108_108519

-- Define the total jumps and time
def total_jumps_liu_li : ℕ := 480
def total_jumps_zhang_hua : ℕ := 420
def time_minutes : ℕ := 5

-- Define the average jumps per minute
def average_jumps_per_minute (total_jumps : ℕ) (time : ℕ) : ℕ :=
  total_jumps / time

-- State the theorem
theorem average_jump_difference :
  average_jumps_per_minute total_jumps_liu_li time_minutes - 
  average_jumps_per_minute total_jumps_zhang_hua time_minutes = 12 := 
sorry


end average_jump_difference_l108_108519


namespace complement_union_correct_l108_108382

open Set

theorem complement_union_correct :
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  (compl P) ∪ Q = {1, 2, 4} :=
by
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  have h : (compl P) ∪ Q = {1, 2, 4} := sorry
  exact h

end complement_union_correct_l108_108382


namespace cylinder_height_proof_l108_108697

noncomputable def cone_base_radius : ℝ := 15
noncomputable def cone_height : ℝ := 25
noncomputable def cylinder_base_radius : ℝ := 10
noncomputable def cylinder_water_height : ℝ := 18.75

theorem cylinder_height_proof :
  (1 / 3 * π * cone_base_radius^2 * cone_height) = π * cylinder_base_radius^2 * cylinder_water_height :=
by sorry

end cylinder_height_proof_l108_108697


namespace least_gumballs_to_ensure_five_gumballs_of_same_color_l108_108015

-- Define the number of gumballs for each color
def red_gumballs := 12
def white_gumballs := 10
def blue_gumballs := 11

-- Define the minimum number of gumballs required to ensure five of the same color
def min_gumballs_to_ensure_five_of_same_color := 13

-- Prove the question == answer given conditions
theorem least_gumballs_to_ensure_five_gumballs_of_same_color :
  (red_gumballs + white_gumballs + blue_gumballs) = 33 → min_gumballs_to_ensure_five_of_same_color = 13 :=
by {
  sorry
}

end least_gumballs_to_ensure_five_gumballs_of_same_color_l108_108015


namespace incorrect_correlation_statement_l108_108610

/--
  The correlation coefficient measures the degree of linear correlation between two variables. 
  The linear correlation coefficient is a quantity whose absolute value is less than 1. 
  Furthermore, the larger its absolute value, the greater the degree of correlation.

  Let r be the sample correlation coefficient.

  We want to prove that the statement "D: |r| ≥ 1, and the closer |r| is to 1, the greater the degree of correlation" 
  is incorrect.
-/
theorem incorrect_correlation_statement (r : ℝ) (h1 : |r| ≤ 1) : ¬ (|r| ≥ 1) :=
by
  -- Proof steps go here
  sorry

end incorrect_correlation_statement_l108_108610


namespace plaza_area_increase_l108_108088

theorem plaza_area_increase (a : ℝ) : 
  ((a + 2)^2 - a^2 = 4 * a + 4) :=
sorry

end plaza_area_increase_l108_108088


namespace prism_volume_l108_108508

theorem prism_volume 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 :=
sorry

end prism_volume_l108_108508


namespace ellipse_m_gt_5_l108_108398

theorem ellipse_m_gt_5 (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m > 5 :=
by
  intros h
  sorry

end ellipse_m_gt_5_l108_108398


namespace xiaohong_total_score_l108_108955

theorem xiaohong_total_score :
  ∀ (midterm_score final_score : ℕ) (midterm_weight final_weight : ℝ),
    midterm_score = 80 →
    final_score = 90 →
    midterm_weight = 0.4 →
    final_weight = 0.6 →
    (midterm_score * midterm_weight + final_score * final_weight) = 86 :=
by
  intros midterm_score final_score midterm_weight final_weight
  intros h1 h2 h3 h4
  sorry

end xiaohong_total_score_l108_108955


namespace trapezium_area_l108_108759

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end trapezium_area_l108_108759


namespace souvenirs_expenses_l108_108208

/--
  Given:
  1. K = T + 146.00
  2. T + K = 548.00
  Prove: 
  - K = 347.00
-/
theorem souvenirs_expenses (T K : ℝ) (h1 : K = T + 146) (h2 : T + K = 548) : K = 347 :=
  sorry

end souvenirs_expenses_l108_108208


namespace total_number_of_lives_l108_108023

theorem total_number_of_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
                              (h1 : initial_players = 7) (h2 : additional_players = 2) (h3 : lives_per_player = 7) : 
                              initial_players + additional_players * lives_per_player = 63 :=
by
  sorry

end total_number_of_lives_l108_108023


namespace base_area_functional_relationship_base_area_when_height_4_8_l108_108309

noncomputable def cylinder_base_area (h : ℝ) : ℝ := 24 / h

theorem base_area_functional_relationship (h : ℝ) (H : h ≠ 0) :
  cylinder_base_area h = 24 / h := by
  unfold cylinder_base_area
  rfl

theorem base_area_when_height_4_8 :
  cylinder_base_area 4.8 = 5 := by
  unfold cylinder_base_area
  norm_num

end base_area_functional_relationship_base_area_when_height_4_8_l108_108309


namespace shirt_to_pants_ratio_l108_108112

noncomputable def cost_uniforms
  (pants_cost shirt_ratio socks_price total_spending : ℕ) : Prop :=
  ∃ (shirt_cost tie_cost : ℕ),
    shirt_cost = shirt_ratio * pants_cost ∧
    tie_cost = shirt_cost / 5 ∧
    5 * (pants_cost + shirt_cost + tie_cost + socks_price) = total_spending

theorem shirt_to_pants_ratio 
  (pants_cost socks_price total_spending : ℕ)
  (h1 : pants_cost = 20)
  (h2 : socks_price = 3)
  (h3 : total_spending = 355)
  (shirt_ratio : ℕ)
  (h4 : cost_uniforms pants_cost shirt_ratio socks_price total_spending) :
  shirt_ratio = 2 := by
  sorry

end shirt_to_pants_ratio_l108_108112


namespace ratio_of_x_to_y_l108_108419

variable {x y : ℝ}

theorem ratio_of_x_to_y (h1 : (3 * x - 2 * y) / (2 * x + 3 * y) = 5 / 4) (h2 : x + y = 5) : x / y = 23 / 2 := 
by {
  sorry
}

end ratio_of_x_to_y_l108_108419


namespace Joseph_has_122_socks_l108_108802

def JosephSocks : Nat := 
  let red_pairs := 9 / 2
  let white_pairs := red_pairs + 2
  let green_pairs := 2 * red_pairs
  let blue_pairs := 3 * green_pairs
  let black_pairs := blue_pairs - 5
  (red_pairs + white_pairs + green_pairs + blue_pairs + black_pairs) * 2

theorem Joseph_has_122_socks : JosephSocks = 122 := 
  by
  sorry

end Joseph_has_122_socks_l108_108802


namespace min_value_ineq_l108_108545

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 4 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 4 → (1/a + 4/b) ≥ 9/4)

theorem min_value_ineq : problem_statement :=
by
  unfold problem_statement
  sorry

end min_value_ineq_l108_108545


namespace solve_trigonometric_equation_l108_108261

theorem solve_trigonometric_equation :
  ∃ (S : Finset ℝ), (∀ X ∈ S, 0 < X ∧ X < 360 ∧ 1 + 2 * Real.sin (X * Real.pi / 180) - 4 * (Real.sin (X * Real.pi / 180))^2 - 8 * (Real.sin (X * Real.pi / 180))^3 = 0) ∧ S.card = 4 :=
by
  sorry

end solve_trigonometric_equation_l108_108261


namespace fred_initial_cards_l108_108113

variables {n : ℕ}

theorem fred_initial_cards (h : n - 22 = 18) : n = 40 :=
by {
  sorry
}

end fred_initial_cards_l108_108113


namespace second_sample_correct_l108_108916

def total_samples : ℕ := 7341
def first_sample : ℕ := 4221
def second_sample : ℕ := total_samples - first_sample

theorem second_sample_correct : second_sample = 3120 :=
by
  sorry

end second_sample_correct_l108_108916


namespace john_trip_total_time_l108_108072

theorem john_trip_total_time :
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  t1 + t2 + t3 + t4 + t5 = 872 :=
by
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  have h1: t1 + t2 + t3 + t4 + t5 = 2 + (3 * 2) + (4 * (3 * 2)) + (5 * (4 * (3 * 2))) + (6 * (5 * (4 * (3 * 2)))) := by
    sorry
  have h2: 2 + 6 + 24 + 120 + 720 = 872 := by
    sorry
  exact h2

end john_trip_total_time_l108_108072


namespace four_digit_3_or_6_l108_108762

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l108_108762


namespace scale_model_height_l108_108754

theorem scale_model_height 
  (scale_ratio : ℚ) (actual_height : ℚ)
  (h_ratio : scale_ratio = 1/30)
  (h_actual_height : actual_height = 305) 
  : Int.ceil (actual_height * scale_ratio) = 10 := by
  -- Define variables and the necessary conditions
  let height_of_model: ℚ := actual_height * scale_ratio
  -- Skip the proof steps
  sorry

end scale_model_height_l108_108754


namespace area_to_paint_correct_l108_108673

-- Define the measurements used in the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 2

-- Definition of areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length

-- Definition of total area to paint
def total_area_to_paint : ℕ := wall_area - (window1_area + window2_area)

-- Theorem statement to prove the total area to paint is 131 square feet
theorem area_to_paint_correct : total_area_to_paint = 131 := by
  sorry

end area_to_paint_correct_l108_108673


namespace maximum_value_of_k_l108_108534

-- Define the variables and conditions
variables {a b c k : ℝ}
axiom h₀ : a > b
axiom h₁ : b > c
axiom h₂ : 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0

-- State the theorem
theorem maximum_value_of_k : k ≤ 9 := sorry

end maximum_value_of_k_l108_108534


namespace statement_B_is_false_l108_108820

def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

theorem statement_B_is_false (x y : ℝ) : 3 * (heartsuit x y) ≠ heartsuit (3 * x) y := by
  sorry

end statement_B_is_false_l108_108820


namespace x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l108_108401

theorem x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 3) ∧ ¬(∀ x : ℝ, x > 3 → x > 5) :=
by 
  -- Prove implications with provided conditions
  sorry

end x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l108_108401


namespace contradiction_in_triangle_l108_108505

theorem contradiction_in_triangle :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A < 60 ∧ B < 60 ∧ C < 60 → false) :=
by
  sorry

end contradiction_in_triangle_l108_108505


namespace solve_equation_in_integers_l108_108040

theorem solve_equation_in_integers (a b c : ℤ) (h : 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) :
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
sorry

end solve_equation_in_integers_l108_108040


namespace triangle_integer_solutions_l108_108414

theorem triangle_integer_solutions (x : ℕ) (h1 : 13 < x) (h2 : x < 43) : 
  ∃ (n : ℕ), n = 29 :=
by 
  sorry

end triangle_integer_solutions_l108_108414


namespace milk_left_l108_108561

theorem milk_left (initial_milk : ℝ) (given_away : ℝ) (h_initial : initial_milk = 5) (h_given : given_away = 18 / 4) :
  ∃ remaining_milk : ℝ, remaining_milk = initial_milk - given_away ∧ remaining_milk = 1 / 2 :=
by
  use 1 / 2
  sorry

end milk_left_l108_108561


namespace red_cards_count_l108_108377

theorem red_cards_count (R B : ℕ) (h1 : R + B = 20) (h2 : 3 * R + 5 * B = 84) : R = 8 :=
sorry

end red_cards_count_l108_108377


namespace find_triplets_l108_108679

noncomputable def triplets_solution (x y z : ℝ) : Prop := 
  (x^2 + y^2 = -x + 3*y + z) ∧ 
  (y^2 + z^2 = x + 3*y - z) ∧ 
  (x^2 + z^2 = 2*x + 2*y - z) ∧ 
  (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

theorem find_triplets : 
  { (x, y, z) : ℝ × ℝ × ℝ | triplets_solution x y z } = 
  { (0, 1, -2), (-3/2, 5/2, -1/2) } :=
sorry

end find_triplets_l108_108679


namespace curve_equation_l108_108862

noncomputable def satisfies_conditions (f : ℝ → ℝ) (M₀ : ℝ × ℝ) : Prop :=
  (f M₀.1 = M₀.2) ∧ 
  (∀ (x y : ℝ) (h_tangent : ∀ x y, y = (f x) → x * y - 2 * (f x) * x = 0),
    y = f x → x * y / (y / x) = 2 * x)

theorem curve_equation (f : ℝ → ℝ) :
  satisfies_conditions f (1, 4) →
  (∀ x : ℝ, f x * x = 4) :=
by
  intro h
  sorry

end curve_equation_l108_108862


namespace Ms_Hatcher_total_students_l108_108344

noncomputable def number_of_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) : ℕ :=
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem Ms_Hatcher_total_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) 
  (h1 : third_graders = 20)
  (h2 : fourth_graders = 2 * third_graders) 
  (h3 : fifth_graders = third_graders / 2) 
  (h4 : sixth_graders = 3 * (third_graders + fourth_graders) / 4) : 
  number_of_students third_graders fourth_graders fifth_graders sixth_graders = 115 :=
by
  sorry

end Ms_Hatcher_total_students_l108_108344


namespace range_of_a_l108_108690

/--
Let f be a function defined on the interval [-1, 1] that is increasing and odd.
If f(-a+1) + f(4a-5) > 0, then the range of the real number a is (4/3, 3/2].
-/
theorem range_of_a
  (f : ℝ → ℝ)
  (h_dom : ∀ x, -1 ≤ x ∧ x ≤ 1 → f x = f x)  -- domain condition
  (h_incr : ∀ x y, x < y → f x < f y)          -- increasing condition
  (h_odd : ∀ x, f (-x) = - f x)                -- odd function condition
  (a : ℝ)
  (h_ineq : f (-a + 1) + f (4 * a - 5) > 0) :
  4 / 3 < a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l108_108690


namespace village_population_equal_in_15_years_l108_108396

theorem village_population_equal_in_15_years :
  ∀ n : ℕ, (72000 - 1200 * n = 42000 + 800 * n) → n = 15 :=
by
  intros n h
  sorry

end village_population_equal_in_15_years_l108_108396


namespace min_value_PF_PA_l108_108066

noncomputable def hyperbola_eq (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = 1

noncomputable def focus_left : ℝ × ℝ := (-4, 0)
noncomputable def focus_right : ℝ × ℝ := (4, 0)
noncomputable def point_A : ℝ × ℝ := (1, 4)

theorem min_value_PF_PA (P : ℝ × ℝ)
  (hP : hyperbola_eq P.1 P.2)
  (hP_right_branch : P.1 > 0) :
  ∃ P : ℝ × ℝ, ∀ X : ℝ × ℝ, hyperbola_eq X.1 X.2 → X.1 > 0 → 
               (dist X focus_left + dist X point_A) ≥ 9 ∧
               (dist P focus_left + dist P point_A) = 9 := 
sorry

end min_value_PF_PA_l108_108066


namespace rationalize_sqrt_35_l108_108791

theorem rationalize_sqrt_35 : (35 / Real.sqrt 35) = Real.sqrt 35 :=
  sorry

end rationalize_sqrt_35_l108_108791


namespace find_A_from_equation_and_conditions_l108_108216

theorem find_A_from_equation_and_conditions 
  (A B C D : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : 10 * A + B ≠ 0)
  (h8 : 10 * 10 * 10 * A + 10 * 10 * B + 8 * 10 + 2 - (900 + C * 10 + 9) = 490 + 3 * 10 + D) :
  A = 5 :=
by
  sorry

end find_A_from_equation_and_conditions_l108_108216


namespace girls_dropped_out_l108_108200

theorem girls_dropped_out (B_initial G_initial B_dropped G_remaining S_remaining : ℕ)
  (hB_initial : B_initial = 14)
  (hG_initial : G_initial = 10)
  (hB_dropped : B_dropped = 4)
  (hS_remaining : S_remaining = 17)
  (hB_remaining : B_initial - B_dropped = B_remaining)
  (hG_remaining : G_remaining = S_remaining - B_remaining) :
  (G_initial - G_remaining) = 3 := 
by 
  sorry

end girls_dropped_out_l108_108200


namespace prove_identical_numbers_l108_108664

variable {x y : ℝ}

theorem prove_identical_numbers (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + (1 / y^2) = y + (1 / x^2))
    (h2 : y^2 + (1 / x) = x^2 + (1 / y)) : x = y :=
by 
  sorry

end prove_identical_numbers_l108_108664


namespace integer_to_the_fourth_l108_108823

theorem integer_to_the_fourth (a : ℤ) (h : a = 243) : 3^12 * 3^8 = a^4 :=
by {
  sorry
}

end integer_to_the_fourth_l108_108823


namespace number_of_uncracked_seashells_l108_108152

theorem number_of_uncracked_seashells (toms_seashells freds_seashells cracked_seashells : ℕ) 
  (h_tom : toms_seashells = 15) 
  (h_fred : freds_seashells = 43) 
  (h_cracked : cracked_seashells = 29) : 
  toms_seashells + freds_seashells - cracked_seashells = 29 :=
by
  sorry

end number_of_uncracked_seashells_l108_108152


namespace intersection_eq_N_l108_108172

def U := Set ℝ                                        -- Universal set U = ℝ
def M : Set ℝ := {x | x ≥ 0}                         -- Set M = {x | x ≥ 0}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}                 -- Set N = {x | 0 ≤ x ≤ 1}

theorem intersection_eq_N : M ∩ N = N := by
  sorry

end intersection_eq_N_l108_108172


namespace union_M_N_is_real_l108_108967

def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

theorem union_M_N_is_real : M ∪ N = Set.univ := by
  sorry

end union_M_N_is_real_l108_108967


namespace exists_group_of_four_l108_108119

-- Define the given conditions
variables (students : Finset ℕ) (h_size : students.card = 21)
variables (done_homework : Finset ℕ → Prop)
variables (hw_unique : ∀ (s : Finset ℕ), s.card = 3 → done_homework s)

-- Define the theorem with the assertion to be proved
theorem exists_group_of_four (students : Finset ℕ) (h_size : students.card = 21)
  (done_homework : Finset ℕ → Prop)
  (hw_unique : ∀ s, s.card = 3 → done_homework s) :
  ∃ (grp : Finset ℕ), grp.card = 4 ∧ 
    (∀ (s : Finset ℕ), s ⊆ grp ∧ s.card = 3 → done_homework s) :=
sorry

end exists_group_of_four_l108_108119


namespace rays_form_straight_lines_l108_108953

theorem rays_form_straight_lines
  (α β : ℝ)
  (h1 : 2 * α + 2 * β = 360) :
  α + β = 180 :=
by
  -- proof details are skipped using sorry
  sorry

end rays_form_straight_lines_l108_108953


namespace find_a3_a4_a5_l108_108672

variable (a : ℕ → ℝ)

-- Recurrence relation for the sequence (condition for n ≥ 2)
axiom rec_relation (n : ℕ) (h : n ≥ 2) : 2 * a n = a (n - 1) + a (n + 1)

-- Additional conditions
axiom cond1 : a 1 + a 3 + a 5 = 9
axiom cond2 : a 3 + a 5 + a 7 = 15

-- Statement to prove
theorem find_a3_a4_a5 : a 3 + a 4 + a 5 = 12 :=
  sorry

end find_a3_a4_a5_l108_108672


namespace ratio_increase_productivity_l108_108980

theorem ratio_increase_productivity (initial current: ℕ) 
  (h_initial: initial = 10) 
  (h_current: current = 25) : 
  (current - initial) / initial = 3 / 2 := 
by
  sorry

end ratio_increase_productivity_l108_108980


namespace range_of_m_l108_108196

theorem range_of_m (m : ℝ) :
  let p := (2 < m ∧ m < 4)
  let q := (m > 1 ∧ 4 - 4 * m < 0)
  (¬ (p ∧ q) ∧ (p ∨ q)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 4) :=
by intros p q h
   let p := 2 < m ∧ m < 4
   let q := m > 1 ∧ 4 - 4 * m < 0
   sorry

end range_of_m_l108_108196


namespace product_xyz_l108_108346

theorem product_xyz (x y z : ℝ) (h1 : x = y) (h2 : x = 2 * z) (h3 : x = 7.999999999999999) :
    x * y * z = 255.9999999999998 := by
  sorry

end product_xyz_l108_108346


namespace number_of_sets_l108_108999

theorem number_of_sets (a n : ℕ) (M : Finset ℕ) (h_consecutive : ∀ x ∈ M, ∃ k, x = a + k ∧ k < n) (h_card : M.card ≥ 2) (h_sum : M.sum id = 2002) : n = 7 :=
sorry

end number_of_sets_l108_108999


namespace Christine_picked_10_pounds_l108_108736

-- Variable declarations for the quantities involved
variable (C : ℝ) -- Pounds of strawberries Christine picked
variable (pieStrawberries : ℝ := 3) -- Pounds of strawberries per pie
variable (pies : ℝ := 10) -- Number of pies
variable (totalStrawberries : ℝ := 30) -- Total pounds of strawberries for pies

-- The condition that Rachel picked twice as many strawberries as Christine
variable (R : ℝ := 2 * C)

-- The condition for the total pounds of strawberries picked by Christine and Rachel
axiom strawberries_eq : C + R = totalStrawberries

-- The goal is to prove that Christine picked 10 pounds of strawberries
theorem Christine_picked_10_pounds : C = 10 := by
  sorry

end Christine_picked_10_pounds_l108_108736


namespace age_ratio_l108_108618

theorem age_ratio (s a : ℕ) (h1 : s - 3 = 2 * (a - 3)) (h2 : s - 7 = 3 * (a - 7)) :
  ∃ x : ℕ, (x = 23) ∧ (s + x) / (a + x) = 3 / 2 :=
by
  sorry

end age_ratio_l108_108618


namespace determine_a_l108_108031

theorem determine_a (a b c : ℤ) (h : (b + 11) * (c + 11) = 2) (hb : b + 11 = -2) (hc : c + 11 = -1) :
  a = 13 := by
  sorry

end determine_a_l108_108031


namespace inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l108_108400

-- Proof Problem 1
theorem inverse_proportional_t (t : ℝ) (h1 : 1 ≤ t ∧ t ≤ 2023) : t = 1 :=
sorry

-- Proof Problem 2
theorem no_linear_function_2k_times (k : ℝ) (h_pos : 0 < k) : ¬ ∃ a b : ℝ, (a < b) ∧ (∀ x, a ≤ x ∧ x ≤ b → (2 * k * a ≤ k * x + 2 ∧ k * x + 2 ≤ 2 * k * b)) :=
sorry

-- Proof Problem 3
theorem quadratic_function_5_times (a b : ℝ) (h_ab : a < b) (h_quad : ∀ x, a ≤ x ∧ x ≤ b → (5 * a ≤ x^2 - 4 * x - 7 ∧ x^2 - 4 * x - 7 ≤ 5 * b)) :
  (a = -2 ∧ b = 1) ∨ (a = -(11/5) ∧ b = (9 + Real.sqrt 109) / 2) :=
sorry

end inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l108_108400


namespace no_three_consecutive_geometric_l108_108824

open Nat

def a (n : ℕ) : ℤ := 3^n - 2^n

theorem no_three_consecutive_geometric :
  ∀ (k : ℕ), ¬ (∃ n m : ℕ, m = n + 1 ∧ k = m + 1 ∧ (a n) * (a k) = (a m)^2) :=
by
  sorry

end no_three_consecutive_geometric_l108_108824


namespace probability_three_common_books_l108_108868

-- Defining the total number of books
def total_books : ℕ := 12

-- Defining the number of books each of Harold and Betty chooses
def books_per_person : ℕ := 6

-- Assertion that the probability of choosing exactly 3 common books is 50/116
theorem probability_three_common_books :
  ((Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3)) /
  ((Nat.choose 12 6) * (Nat.choose 12 6)) = 50 / 116 := by
  sorry

end probability_three_common_books_l108_108868


namespace distance_geologists_probability_l108_108297

theorem distance_geologists_probability :
  let speed := 4 -- km/h
  let n_roads := 6
  let travel_time := 1 -- hour
  let distance_traveled := speed * travel_time -- km
  let distance_threshold := 6 -- km
  let n_outcomes := n_roads * n_roads
  let favorable_outcomes := 18 -- determined from the solution steps
  let probability := favorable_outcomes / n_outcomes
  probability = 0.5 := by
  sorry

end distance_geologists_probability_l108_108297


namespace test_score_based_on_preparation_l108_108924

theorem test_score_based_on_preparation :
  (grade_varies_directly_with_effective_hours : Prop) →
  (effective_hour_constant : ℝ) →
  (actual_hours_first_test : ℕ) →
  (actual_hours_second_test : ℕ) →
  (score_first_test : ℕ) →
  effective_hour_constant = 0.8 →
  actual_hours_first_test = 5 →
  score_first_test = 80 →
  actual_hours_second_test = 6 →
  grade_varies_directly_with_effective_hours →
  ∃ score_second_test : ℕ, score_second_test = 96 := by
  sorry

end test_score_based_on_preparation_l108_108924


namespace find_rate_per_kg_of_mangoes_l108_108474

theorem find_rate_per_kg_of_mangoes (r : ℝ) 
  (total_units_paid : ℝ) (grapes_kg : ℝ) (grapes_rate : ℝ)
  (mangoes_kg : ℝ) (total_grapes_cost : ℝ)
  (total_mangoes_cost : ℝ) (total_cost : ℝ) :
  grapes_kg = 8 →
  grapes_rate = 70 →
  mangoes_kg = 10 →
  total_units_paid = 1110 →
  total_grapes_cost = grapes_kg * grapes_rate →
  total_mangoes_cost = total_units_paid - total_grapes_cost →
  r = total_mangoes_cost / mangoes_kg →
  r = 55 := by
  intros
  sorry

end find_rate_per_kg_of_mangoes_l108_108474


namespace larger_number_l108_108033

theorem larger_number (t a b : ℝ) (h1 : a + b = t) (h2 : a ^ 2 - b ^ 2 = 208) (ht : t = 104) :
  a = 53 :=
by
  sorry

end larger_number_l108_108033


namespace sheila_hourly_rate_is_6_l108_108529

variable (weekly_earnings : ℕ) (hours_mwf : ℕ) (days_mwf : ℕ) (hours_tt: ℕ) (days_tt : ℕ)
variable [NeZero hours_mwf] [NeZero days_mwf] [NeZero hours_tt] [NeZero days_tt]

-- Define Sheila's working hours and weekly earnings as given conditions
def weekly_hours := (hours_mwf * days_mwf) + (hours_tt * days_tt)
def hourly_rate := weekly_earnings / weekly_hours

-- Specific values from the given problem
def sheila_weekly_earnings : ℕ := 216
def sheila_hours_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_tt : ℕ := 6
def sheila_days_tt : ℕ := 2

-- The theorem to prove
theorem sheila_hourly_rate_is_6 :
  (sheila_weekly_earnings / ((sheila_hours_mwf * sheila_days_mwf) + (sheila_hours_tt * sheila_days_tt))) = 6 := by
  sorry

end sheila_hourly_rate_is_6_l108_108529


namespace division_by_fraction_l108_108137

theorem division_by_fraction :
  (5 / (8 / 15) : ℚ) = 75 / 8 :=
by
  sorry

end division_by_fraction_l108_108137


namespace lily_spent_amount_l108_108007

def num_years (start_year end_year : ℕ) : ℕ :=
  end_year - start_year

def total_spent (cost_per_plant num_years : ℕ) : ℕ :=
  cost_per_plant * num_years

theorem lily_spent_amount :
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  num_years start_year end_year = 32 →
  total_spent cost_per_plant 32 = 640 :=
by
  intros
  sorry

end lily_spent_amount_l108_108007


namespace min_xyz_value_l108_108148

theorem min_xyz_value (x y z : ℝ) (h1 : x + y + z = 1) (h2 : z = 2 * y) (h3 : y ≤ (1 / 3)) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (∀ a b c : ℝ, (a + b + c = 1) → (c = 2 * b) → (b ≤ (1 / 3)) → 0 < a → 0 < b → 0 < c → (a * b * c) ≥ (x * y * z) → (a * b * c) = (8 / 243)) :=
by sorry

end min_xyz_value_l108_108148


namespace cocoa_powder_total_l108_108121

variable (already_has : ℕ) (still_needs : ℕ)

theorem cocoa_powder_total (h₁ : already_has = 259) (h₂ : still_needs = 47) : already_has + still_needs = 306 :=
by
  sorry

end cocoa_powder_total_l108_108121


namespace triangle_largest_angle_l108_108241

theorem triangle_largest_angle (x : ℝ) (hx : x + 2 * x + 3 * x = 180) :
  3 * x = 90 :=
by
  sorry

end triangle_largest_angle_l108_108241


namespace major_axis_range_l108_108909

theorem major_axis_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ x M N : ℝ, (x + (1 - x)) = 1 → x * (1 - x) = 0) 
  (e : ℝ) (h4 : (Real.sqrt 3 / 3) ≤ e ∧ e ≤ (Real.sqrt 2 / 2)) :
  ∃ a : ℝ, 2 * (Real.sqrt 5) ≤ 2 * a ∧ 2 * a ≤ 2 * (Real.sqrt 6) := 
sorry

end major_axis_range_l108_108909


namespace not_right_triangle_A_l108_108542

def is_right_triangle (a b c : Real) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_A : ¬ (is_right_triangle 1.5 2 3) :=
by sorry

end not_right_triangle_A_l108_108542


namespace value_of_a_plus_b_l108_108360

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l108_108360


namespace find_difference_l108_108981

noncomputable def expression (x y : ℝ) : ℝ :=
  (|x + y| / (|x| + |y|))^2

theorem find_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  let m := 0
  let M := 1
  M - m = 1 :=
by
  -- Please note that the proof is omitted and replaced with sorry
  sorry

end find_difference_l108_108981


namespace minimum_questions_needed_a_l108_108041

theorem minimum_questions_needed_a (n : ℕ) (m : ℕ) (h1 : m = n) (h2 : m < 2 ^ n) :
  ∃Q : ℕ, Q = n := sorry

end minimum_questions_needed_a_l108_108041


namespace root_polynomial_satisfies_expression_l108_108394

noncomputable def roots_of_polynomial (x : ℕ) : Prop :=
  x^3 - 15 * x^2 + 25 * x - 10 = 0

theorem root_polynomial_satisfies_expression (p q r : ℕ) 
    (h1 : roots_of_polynomial p)
    (h2 : roots_of_polynomial q)
    (h3 : roots_of_polynomial r)
    (h_sum : p + q + r = 15)
    (h_prod : p*q + q*r + r*p = 25) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end root_polynomial_satisfies_expression_l108_108394


namespace multiplier_of_product_l108_108994

variable {a b : ℝ}

theorem multiplier_of_product (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a + b = k * (a * b))
  (h4 : (1 / a) + (1 / b) = 6) : k = 6 := by
  sorry

end multiplier_of_product_l108_108994


namespace algebraic_expression_evaluation_l108_108711

theorem algebraic_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ( ( (a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) ) = 1 :=
by sorry

end algebraic_expression_evaluation_l108_108711


namespace no_integer_solutions_l108_108595

theorem no_integer_solutions (x y : ℤ) : 2 * x^2 - 5 * y^2 ≠ 7 :=
  sorry

end no_integer_solutions_l108_108595


namespace factorize_expression_l108_108020

theorem factorize_expression (a x y : ℤ) : a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l108_108020


namespace worker_time_proof_l108_108989

theorem worker_time_proof (x : ℝ) (h1 : x > 2) (h2 : (100 / (x - 2) - 100 / x) = 5 / 2) : 
  (x = 10) ∧ (x - 2 = 8) :=
by
  sorry

end worker_time_proof_l108_108989


namespace stocking_stuffers_total_l108_108554

-- Defining the number of items per category
def candy_canes := 4
def beanie_babies := 2
def books := 1
def small_toys := 3
def gift_cards := 1

-- Total number of stocking stuffers per child
def items_per_child := candy_canes + beanie_babies + books + small_toys + gift_cards

-- Number of children
def number_of_children := 3

-- Total number of stocking stuffers for all children
def total_stocking_stuffers := items_per_child * number_of_children

-- Statement to be proved
theorem stocking_stuffers_total : total_stocking_stuffers = 33 := by
  sorry

end stocking_stuffers_total_l108_108554


namespace width_of_carton_is_25_l108_108282

-- Definitions for the given problem
def carton_width := 25
def carton_length := 60
def width_or_height := min carton_width carton_length

theorem width_of_carton_is_25 : width_or_height = 25 := by
  sorry

end width_of_carton_is_25_l108_108282


namespace mr_wang_returns_to_start_elevator_electricity_consumption_l108_108819

-- Definition for the first part of the problem
def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]

theorem mr_wang_returns_to_start : List.sum floor_movements = 0 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

-- Definitions for the second part of the problem
def height_per_floor : Int := 3
def electricity_per_meter : Float := 0.2

-- Calculation of electricity consumption (distance * electricity_per_meter per floor)
def total_distance_traveled : Int := 
  (floor_movements.map Int.natAbs).sum * height_per_floor

theorem elevator_electricity_consumption : 
  (Float.ofInt total_distance_traveled) * electricity_per_meter = 33.6 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

end mr_wang_returns_to_start_elevator_electricity_consumption_l108_108819


namespace find_p_current_age_l108_108175

theorem find_p_current_age (x p q : ℕ) (h1 : p - 3 = 4 * x) (h2 : q - 3 = 3 * x) (h3 : (p + 6) / (q + 6) = 7 / 6) : p = 15 := 
sorry

end find_p_current_age_l108_108175


namespace distinct_sequences_count_l108_108191

-- Defining the set of letters in "PROBLEMS"
def letters : List Char := ['P', 'R', 'O', 'B', 'L', 'E', 'M']

-- Defining a sequence constraint: must start with 'S' and not end with 'M'
def valid_sequence (seq : List Char) : Prop :=
  seq.head? = some 'S' ∧ seq.getLast? ≠ some 'M'

-- Counting valid sequences according to the constraints
noncomputable def count_valid_sequences : Nat :=
  6 * 120

theorem distinct_sequences_count :
  count_valid_sequences = 720 := by
  sorry

end distinct_sequences_count_l108_108191


namespace reaction_rate_reduction_l108_108971

theorem reaction_rate_reduction (k : ℝ) (NH3 Br2 NH3_new : ℝ) (v1 v2 : ℝ):
  (v1 = k * NH3^8 * Br2) →
  (v2 = k * NH3_new^8 * Br2) →
  (v2 / v1 = 60) →
  NH3_new = 60 ^ (1 / 8) :=
by
  intro hv1 hv2 hratio
  sorry

end reaction_rate_reduction_l108_108971


namespace combined_distance_is_twelve_l108_108028

-- Definitions based on the conditions
def distance_second_lady : ℕ := 4
def distance_first_lady : ℕ := 2 * distance_second_lady
def total_distance : ℕ := distance_second_lady + distance_first_lady

-- Theorem statement
theorem combined_distance_is_twelve : total_distance = 12 := by
  sorry

end combined_distance_is_twelve_l108_108028


namespace calculate_expression_l108_108130

theorem calculate_expression :
  15^2 + 2 * 15 * 5 + 5^2 + 5^3 = 525 := 
sorry

end calculate_expression_l108_108130


namespace max_value_func_l108_108877

noncomputable def func (x : ℝ) : ℝ :=
  Real.sin x - Real.sqrt 3 * Real.cos x

theorem max_value_func : ∃ x : ℝ, func x = 2 :=
by
  -- proof steps will be provided here
  sorry

end max_value_func_l108_108877


namespace even_function_a_value_l108_108159

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (a * (-x)^2 + (2 * a + 1) * (-x) - 1) = (a * x^2 + (2 * a + 1) * x - 1)) →
  a = - 1 / 2 :=
by sorry

end even_function_a_value_l108_108159


namespace find_digit_l108_108299

theorem find_digit {x : ℕ} (hx : x = 7) : (10 * (x - 3) + x) = 47 :=
by
  sorry

end find_digit_l108_108299


namespace line_equation_direction_point_l108_108035

theorem line_equation_direction_point 
  (d : ℝ × ℝ) (A : ℝ × ℝ) :
  d = (2, -1) →
  A = (1, 0) →
  ∃ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = -1 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 ↔ x + 2 * y - 1 = 0 :=
by
  sorry

end line_equation_direction_point_l108_108035


namespace square_paper_side_length_l108_108223

theorem square_paper_side_length :
  ∀ (edge_length : ℝ) (num_pieces : ℕ) (side_length : ℝ),
  edge_length = 12 ∧ num_pieces = 54 ∧ 6 * (edge_length ^ 2) = num_pieces * (side_length ^ 2)
  → side_length = 4 :=
by
  intros edge_length num_pieces side_length h
  sorry

end square_paper_side_length_l108_108223


namespace helga_shoes_l108_108128

theorem helga_shoes (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := 
by
  sorry

end helga_shoes_l108_108128


namespace sum_seven_consecutive_l108_108850

theorem sum_seven_consecutive (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 :=
by
  sorry

end sum_seven_consecutive_l108_108850


namespace forty_percent_of_n_l108_108936

theorem forty_percent_of_n (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : 0.40 * N = 384 :=
by
  sorry

end forty_percent_of_n_l108_108936


namespace find_positive_number_l108_108723
-- Prove the positive number x that satisfies the condition is 8
theorem find_positive_number (x : ℝ) (hx : 0 < x) :
    x + 8 = 128 * (1 / x) → x = 8 :=
by
  intro h
  sorry

end find_positive_number_l108_108723


namespace merchant_marked_price_l108_108843

variable (L C M S : ℝ)

-- Conditions
def condition1 : Prop := C = 0.7 * L
def condition2 : Prop := C = 0.7 * S
def condition3 : Prop := S = 0.8 * M

-- The main statement
theorem merchant_marked_price (h1 : condition1 L C) (h2 : condition2 C S) (h3 : condition3 S M) : M = 1.25 * L :=
by
  sorry

end merchant_marked_price_l108_108843


namespace distance_between_intersections_l108_108562

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 25 = 1

def is_focus_of_ellipse (fx fy : ℝ) : Prop := (fx = 0 ∧ (fy = 4 ∨ fy = -4))

def parabola_eq (x y : ℝ) : Prop := y = x^2 / 8 + 2

theorem distance_between_intersections :
  let d := 12 * Real.sqrt 2 / 5
  ∃ x1 x2 y1 y2 : ℝ, 
    ellipse_eq x1 y1 ∧ 
    parabola_eq x1 y1 ∧
    ellipse_eq x2 y2 ∧
    parabola_eq x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

end distance_between_intersections_l108_108562


namespace negation_of_proposition_l108_108643

theorem negation_of_proposition (x : ℝ) (h : 2 * x + 1 ≤ 0) : ¬ (2 * x + 1 ≤ 0) ↔ 2 * x + 1 > 0 := 
by
  sorry

end negation_of_proposition_l108_108643


namespace stuart_segments_return_l108_108117

theorem stuart_segments_return (r1 r2 : ℝ) (tangent_chord : ℝ)
  (angle_ABC : ℝ) (h1 : r1 < r2) (h2 : tangent_chord = r1 * 2)
  (h3 : angle_ABC = 75) :
  ∃ (n : ℕ), n = 24 ∧ tangent_chord * n = 360 * (n / 24) :=
by {
  sorry
}

end stuart_segments_return_l108_108117


namespace no_distributive_laws_hold_l108_108150

def tripledAfterAdding (a b : ℝ) : ℝ := 3 * (a + b)

theorem no_distributive_laws_hold (x y z : ℝ) :
  ¬ (tripledAfterAdding x (y + z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) ∧
  ¬ (x + (tripledAfterAdding y z) = tripledAfterAdding (x + y) (x + z)) ∧
  ¬ (tripledAfterAdding x (tripledAfterAdding y z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) :=
by sorry

end no_distributive_laws_hold_l108_108150


namespace cars_in_parking_lot_l108_108891

theorem cars_in_parking_lot (C : ℕ) (customers_per_car : ℕ) (total_purchases : ℕ) 
  (h1 : customers_per_car = 5)
  (h2 : total_purchases = 50)
  (h3 : C * customers_per_car = total_purchases) : 
  C = 10 := 
by
  sorry

end cars_in_parking_lot_l108_108891


namespace incorrect_arrangements_hello_l108_108215

-- Given conditions: the word "hello" with letters 'h', 'e', 'l', 'l', 'o'
def letters : List Char := ['h', 'e', 'l', 'l', 'o']

-- The number of permutations of the letters in "hello" excluding the correct order
-- We need to prove that the number of incorrect arrangements is 59.
theorem incorrect_arrangements_hello : 
  (List.permutations letters).length - 1 = 59 := 
by sorry

end incorrect_arrangements_hello_l108_108215


namespace avg_weight_l108_108767

theorem avg_weight (A B C : ℝ)
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by sorry

end avg_weight_l108_108767


namespace original_number_of_motorcycles_l108_108115

theorem original_number_of_motorcycles (x y : ℕ) 
  (h1 : x + 2 * y = 42) 
  (h2 : x > y) 
  (h3 : 2 * (x - 3) + 4 * y = 3 * (x + y - 3)) : x = 16 := 
sorry

end original_number_of_motorcycles_l108_108115


namespace sandy_total_sums_attempted_l108_108965

theorem sandy_total_sums_attempted (C I : ℕ) 
  (marks_per_correct_sum : ℕ := 3) 
  (marks_lost_per_incorrect_sum : ℕ := 2) 
  (total_marks : ℕ := 45) 
  (correct_sums : ℕ := 21) 
  (H : 3 * correct_sums - 2 * I = total_marks) 
  : C + I = 30 := 
by 
  sorry

end sandy_total_sums_attempted_l108_108965


namespace complete_square_solution_l108_108949

theorem complete_square_solution (x: ℝ) : (x^2 + 8 * x - 3 = 0) -> ((x + 4)^2 = 19) := 
by
  sorry

end complete_square_solution_l108_108949


namespace max_apartment_size_is_600_l108_108585

-- Define the cost per square foot and Max's budget
def cost_per_square_foot : ℝ := 1.2
def max_budget : ℝ := 720

-- Define the largest apartment size that Max should consider
def largest_apartment_size (s : ℝ) : Prop :=
  cost_per_square_foot * s = max_budget

-- State the theorem that we need to prove
theorem max_apartment_size_is_600 : largest_apartment_size 600 :=
  sorry

end max_apartment_size_is_600_l108_108585


namespace tour_group_size_l108_108342

def adult_price : ℕ := 8
def child_price : ℕ := 3
def total_spent : ℕ := 44

theorem tour_group_size :
  ∃ (x y : ℕ), adult_price * x + child_price * y = total_spent ∧ (x + y = 8 ∨ x + y = 13) :=
by
  sorry

end tour_group_size_l108_108342


namespace parallel_lines_a_value_l108_108934

theorem parallel_lines_a_value 
    (a : ℝ) 
    (l₁ : ∀ x y : ℝ, 2 * x + y - 1 = 0) 
    (l₂ : ∀ x y : ℝ, (a - 1) * x + 3 * y - 2 = 0) 
    (h_parallel : ∀ x y : ℝ, 2 / (a - 1) = 1 / 3) : 
    a = 7 := 
    sorry

end parallel_lines_a_value_l108_108934


namespace solution_is_permutations_l108_108435

noncomputable def solve_system (x y z : ℤ) : Prop :=
  x^2 = y * z + 1 ∧ y^2 = z * x + 1 ∧ z^2 = x * y + 1

theorem solution_is_permutations (x y z : ℤ) :
  solve_system x y z ↔ (x, y, z) = (1, 0, -1) ∨ (x, y, z) = (1, -1, 0) ∨ (x, y, z) = (0, 1, -1) ∨ (x, y, z) = (0, -1, 1) ∨ (x, y, z) = (-1, 1, 0) ∨ (x, y, z) = (-1, 0, 1) :=
by sorry

end solution_is_permutations_l108_108435


namespace decrypt_encryption_l108_108866

-- Encryption function description
def encrypt_digit (d : ℕ) : ℕ := 10 - (d * 7 % 10)

def encrypt_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let encrypted_digits := digits.map encrypt_digit
  encrypted_digits.foldr (λ d acc => d + acc * 10) 0
  
noncomputable def digit_match (d: ℕ) : ℕ :=
  match d with
  | 0 => 0 | 1 => 3 | 2 => 8 | 3 => 1 | 4 => 6 | 5 => 5
  | 6 => 8 | 7 => 1 | 8 => 4 | 9 => 7 | _ => 0

theorem decrypt_encryption:
encrypt_number 891134 = 473392 :=
by
  sorry

end decrypt_encryption_l108_108866


namespace ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l108_108262

theorem ellipse_equation_x_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 4 ∧ b = 3 ∧ a = 5 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 := by
  sorry

theorem ellipse_equation_y_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 3 ∧ b = 4 ∧ a = 5 ∧ (x^2 / b^2) + (y^2 / a^2) = 1 := by
  sorry

end ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l108_108262


namespace remaining_amount_after_purchase_l108_108745

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem remaining_amount_after_purchase : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end remaining_amount_after_purchase_l108_108745


namespace solution_l108_108950

theorem solution (x : ℝ) 
  (h1 : 1/x < 3)
  (h2 : 1/x > -4) 
  (h3 : x^2 - 3*x + 2 < 0) : 
  1 < x ∧ x < 2 :=
sorry

end solution_l108_108950


namespace maximize_profit_l108_108835

noncomputable def profit (x : ℝ) : ℝ :=
  16 - 4/(x+1) - x

theorem maximize_profit (a : ℝ) (h : 0 ≤ a) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ a ∧ profit x = max 13 (16 - 4/(a+1) - a) := by
  sorry

end maximize_profit_l108_108835


namespace percentage_employees_four_years_or_more_l108_108458

theorem percentage_employees_four_years_or_more 
  (x : ℝ) 
  (less_than_one_year : ℝ := 6 * x)
  (one_to_two_years : ℝ := 4 * x)
  (two_to_three_years : ℝ := 7 * x)
  (three_to_four_years : ℝ := 3 * x)
  (four_to_five_years : ℝ := 3 * x)
  (five_to_six_years : ℝ := 1 * x)
  (six_to_seven_years : ℝ := 1 * x)
  (seven_to_eight_years : ℝ := 2 * x)
  (total_employees : ℝ := 27 * x)
  (employees_four_years_or_more : ℝ := 7 * x) : 
  (employees_four_years_or_more / total_employees) * 100 = 25.93 := 
by
  sorry

end percentage_employees_four_years_or_more_l108_108458


namespace weight_of_mixture_l108_108586

noncomputable def total_weight_of_mixture (zinc_weight: ℝ) (zinc_ratio: ℝ) (total_ratio: ℝ) : ℝ :=
  (zinc_weight / zinc_ratio) * total_ratio

theorem weight_of_mixture (zinc_ratio: ℝ) (copper_ratio: ℝ) (tin_ratio: ℝ) (zinc_weight: ℝ) :
  total_weight_of_mixture zinc_weight zinc_ratio (zinc_ratio + copper_ratio + tin_ratio) = 98.95 :=
by 
  let ratio_sum := zinc_ratio + copper_ratio + tin_ratio
  let part_weight := zinc_weight / zinc_ratio
  let mixture_weight := part_weight * ratio_sum
  have h : mixture_weight = 98.95 := sorry
  exact h

end weight_of_mixture_l108_108586


namespace circle_standard_equation_l108_108887

theorem circle_standard_equation (x y : ℝ) :
  let center_x := 2
  let center_y := -1
  let radius := 3
  (center_x = 2) ∧ (center_y = -1) ∧ (radius = 3) → (x - center_x) ^ 2 + (y - center_y) ^ 2 = radius ^ 2 :=
by
  intros
  sorry

end circle_standard_equation_l108_108887


namespace nadine_white_pebbles_l108_108203

variable (W R : ℝ)

theorem nadine_white_pebbles :
  (R = 1/2 * W) →
  (W + R = 30) →
  W = 20 :=
by
  sorry

end nadine_white_pebbles_l108_108203


namespace find_three_digit_number_l108_108408

theorem find_three_digit_number (A B C : ℕ) (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) : 100 * A + 10 * B + C = 253 :=
by {
  sorry
}

end find_three_digit_number_l108_108408


namespace value_of_x_plus_y_l108_108892

theorem value_of_x_plus_y (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l108_108892


namespace cleaning_times_l108_108284

theorem cleaning_times (A B C : ℕ) (hA : A = 40) (hB : B = A / 4) (hC : C = 2 * B) : 
  B = 10 ∧ C = 20 := by
  sorry

end cleaning_times_l108_108284


namespace solve_for_A_l108_108125

theorem solve_for_A (A : ℚ) : 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 → A = -4/3 :=
by
  sorry

end solve_for_A_l108_108125


namespace solution_set_inequality_l108_108165

theorem solution_set_inequality (a b x : ℝ) (h₀ : {x : ℝ | ax - b < 0} = {x : ℝ | 1 < x}) :
  {x : ℝ | (ax + b) * (x - 3) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_inequality_l108_108165


namespace rebecca_eggs_l108_108575

theorem rebecca_eggs (groups eggs_per_group : ℕ) (h1 : groups = 3) (h2 : eggs_per_group = 6) : 
  (groups * eggs_per_group = 18) :=
by
  sorry

end rebecca_eggs_l108_108575


namespace find_x_coordinate_l108_108828

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = 6 * x ∧ x > 0 

noncomputable def is_twice_distance (x : ℝ) : Prop :=
  let focus_x : ℝ := 3 / 2
  let d1 := x + focus_x
  let d2 := x
  d1 = 2 * d2

theorem find_x_coordinate (x y : ℝ) :
  point_on_parabola x y →
  is_twice_distance x →
  x = 3 / 2 :=
by
  intros
  sorry

end find_x_coordinate_l108_108828


namespace sqrt_expression_meaningful_l108_108977

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l108_108977


namespace repeating_decimal_sum_l108_108144

-- Definitions from conditions
def repeating_decimal_1_3 : ℚ := 1 / 3
def repeating_decimal_2_99 : ℚ := 2 / 99

-- Statement to prove
theorem repeating_decimal_sum : repeating_decimal_1_3 + repeating_decimal_2_99 = 35 / 99 :=
by sorry

end repeating_decimal_sum_l108_108144


namespace P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l108_108248

-- Define the problem conditions and questions
def P_1 (n : ℕ) : ℚ := sorry
def P_2 (n : ℕ) : ℚ := sorry

-- Part (a)
theorem P2_3_eq_2_3 : P_2 3 = 2 / 3 := sorry

-- Part (b)
theorem P1_n_eq_1_n (n : ℕ) (h : n ≥ 1): P_1 n = 1 / n := sorry

-- Part (c)
theorem P2_recurrence (n : ℕ) (h : n ≥ 2) : 
  P_2 n = (2 / n) * P_1 (n-1) + ((n-2) / n) * P_2 (n-1) := sorry

-- Part (d)
theorem P2_n_eq_2_n (n : ℕ) (h : n ≥ 1): P_2 n = 2 / n := sorry

end P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l108_108248


namespace range_of_m_l108_108807

-- Define the ellipse and conditions
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2 / m) + (y^2 / 2) = 1
def point_exists (M : ℝ × ℝ) (C : ℝ → ℝ → ℝ → Prop) : Prop := ∃ p : ℝ × ℝ, C p.1 p.2 (M.1 + M.2)

-- State the theorem
theorem range_of_m (m : ℝ) (h₁ : ellipse x y m) (h₂ : point_exists M ellipse) :
  (0 < m ∧ m <= 1/2) ∨ (8 <= m) := 
sorry

end range_of_m_l108_108807


namespace students_wearing_specific_shirt_and_accessory_count_l108_108898

theorem students_wearing_specific_shirt_and_accessory_count :
  let total_students := 1000
  let blue_shirt_percent := 0.40
  let red_shirt_percent := 0.25
  let green_shirt_percent := 0.20
  let blue_shirt_students := blue_shirt_percent * total_students
  let red_shirt_students := red_shirt_percent * total_students
  let green_shirt_students := green_shirt_percent * total_students
  let blue_shirt_stripes_percent := 0.30
  let blue_shirt_polka_dots_percent := 0.35
  let red_shirt_stripes_percent := 0.20
  let red_shirt_polka_dots_percent := 0.40
  let green_shirt_stripes_percent := 0.25
  let green_shirt_polka_dots_percent := 0.25
  let accessory_hat_percent := 0.15
  let accessory_scarf_percent := 0.10
  let red_polka_dot_students := red_shirt_polka_dots_percent * red_shirt_students
  let red_polka_dot_hat_students := accessory_hat_percent * red_polka_dot_students
  let green_no_pattern_students := green_shirt_students - (green_shirt_stripes_percent * green_shirt_students + green_shirt_polka_dots_percent * green_shirt_students)
  let green_no_pattern_scarf_students := accessory_scarf_percent * green_no_pattern_students
  red_polka_dot_hat_students + green_no_pattern_scarf_students = 25 := by
    sorry

end students_wearing_specific_shirt_and_accessory_count_l108_108898


namespace find_first_offset_l108_108847

theorem find_first_offset 
  (diagonal : ℝ) (second_offset : ℝ) (area : ℝ) (first_offset : ℝ)
  (h_diagonal : diagonal = 20)
  (h_second_offset : second_offset = 4)
  (h_area : area = 90)
  (h_area_formula : area = (diagonal * (first_offset + second_offset)) / 2) :
  first_offset = 5 :=
by 
  rw [h_diagonal, h_second_offset, h_area] at h_area_formula 
  -- This would be the place where you handle solving the formula using the given conditions
  sorry

end find_first_offset_l108_108847


namespace square_of_equal_side_of_inscribed_triangle_l108_108625

theorem square_of_equal_side_of_inscribed_triangle :
  ∀ (x y : ℝ),
  (x^2 + 9 * y^2 = 9) →
  ((x = 0) → (y = 1)) →
  ((x ≠ 0) → y = (x + 1)) →
  square_of_side = (324 / 25) :=
by
  intros x y hEllipse hVertex hSlope
  sorry

end square_of_equal_side_of_inscribed_triangle_l108_108625


namespace length_of_GH_l108_108110

theorem length_of_GH (AB CD GH : ℤ) (h_parallel : AB = 240 ∧ CD = 160 ∧ (AB + CD) = GH*2) : GH = 320 / 3 :=
by sorry

end length_of_GH_l108_108110


namespace hulk_jump_distance_exceeds_1000_l108_108211

theorem hulk_jump_distance_exceeds_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → 3^m ≤ 1000) ∧ 3^n > 1000 :=
sorry

end hulk_jump_distance_exceeds_1000_l108_108211


namespace speed_of_second_train_l108_108451

/-- Given:
1. The first train has a length of 220 meters.
2. The speed of the first train is 120 kilometers per hour.
3. The time taken to cross each other is 9 seconds.
4. The length of the second train is 280.04 meters.

Prove the speed of the second train is 80 kilometers per hour. -/
theorem speed_of_second_train
    (len_first_train : ℝ := 220)
    (speed_first_train_kmph : ℝ := 120)
    (time_to_cross : ℝ := 9)
    (len_second_train : ℝ := 280.04) 
  : (len_first_train / time_to_cross + len_second_train / time_to_cross - (speed_first_train_kmph * 1000 / 3600)) * (3600 / 1000) = 80 := 
by
  sorry

end speed_of_second_train_l108_108451


namespace correct_quadratic_equation_l108_108347

def is_quadratic_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + 1 = 0"

theorem correct_quadratic_equation :
  is_quadratic_with_one_variable "x^2 + 1 = 0" :=
by {
  sorry
}

end correct_quadratic_equation_l108_108347


namespace problem_equivalent_l108_108710

noncomputable def h (y : ℝ) : ℝ := y^5 - y^3 + 2
noncomputable def k (y : ℝ) : ℝ := y^2 - 3

theorem problem_equivalent (y₁ y₂ y₃ y₄ y₅ : ℝ) (h_roots : ∀ y, h y = 0 ↔ y = y₁ ∨ y = y₂ ∨ y = y₃ ∨ y = y₄ ∨ y = y₅) :
  (k y₁) * (k y₂) * (k y₃) * (k y₄) * (k y₅) = 104 :=
sorry

end problem_equivalent_l108_108710


namespace hypotenuse_of_right_triangle_l108_108549

theorem hypotenuse_of_right_triangle (a b : ℕ) (ha : a = 140) (hb : b = 336) :
  Nat.sqrt (a * a + b * b) = 364 := by
  sorry

end hypotenuse_of_right_triangle_l108_108549


namespace number_solution_l108_108913

variable (a : ℝ) (x : ℝ)

theorem number_solution :
  (a^(-x) + 25^(-2*x) + 5^(-4*x) = 11) ∧ (x = 0.25) → a = 625 / 7890481 :=
by 
  sorry

end number_solution_l108_108913


namespace special_day_jacket_price_l108_108557

noncomputable def original_price : ℝ := 240
noncomputable def first_discount_rate : ℝ := 0.4
noncomputable def special_day_discount_rate : ℝ := 0.25

noncomputable def first_discounted_price : ℝ :=
  original_price * (1 - first_discount_rate)
  
noncomputable def special_day_price : ℝ :=
  first_discounted_price * (1 - special_day_discount_rate)

theorem special_day_jacket_price : special_day_price = 108 := by
  -- definitions and calculations go here
  sorry

end special_day_jacket_price_l108_108557


namespace find_m_l108_108634

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
sorry

end find_m_l108_108634


namespace rectangle_area_l108_108804

theorem rectangle_area (w l : ℝ) (h_width : w = 4) (h_perimeter : 2 * l + 2 * w = 30) :
    l * w = 44 :=
by 
  sorry

end rectangle_area_l108_108804


namespace decimal_representation_of_fraction_l108_108139

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l108_108139


namespace circles_internally_tangent_l108_108571

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x - 6)^2 + y^2 = 1 → 
  (x - 3)^2 + (y - 4)^2 = 36 → 
  true := 
by 
  intros x y h1 h2
  sorry

end circles_internally_tangent_l108_108571


namespace rad_times_trivia_eq_10000_l108_108596

theorem rad_times_trivia_eq_10000 
  (h a r v d m i t : ℝ)
  (H1 : h * a * r * v * a * r * d = 100)
  (H2 : m * i * t = 100)
  (H3 : h * m * m * t = 100) :
  (r * a * d) * (t * r * i * v * i * a) = 10000 := 
  sorry

end rad_times_trivia_eq_10000_l108_108596


namespace line_not_in_first_quadrant_l108_108927

theorem line_not_in_first_quadrant (m x : ℝ) (h : mx + 3 = 4) (hx : x = 1) : 
  ∀ x y : ℝ, y = (m - 2) * x - 3 → ¬(0 < x ∧ 0 < y) :=
by
  -- The actual proof would go here
  sorry

end line_not_in_first_quadrant_l108_108927


namespace divides_x_by_5_l108_108224

theorem divides_x_by_5 (x y : ℤ) (hx1 : 1 < x) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end divides_x_by_5_l108_108224


namespace simplify_fraction_l108_108094

theorem simplify_fraction (n : ℕ) (h : 2 ^ n ≠ 0) : 
  (2 ^ (n + 5) - 3 * 2 ^ n) / (3 * 2 ^ (n + 4)) = 29 / 48 := 
by
  sorry

end simplify_fraction_l108_108094


namespace direction_vector_of_line_l108_108383

theorem direction_vector_of_line : 
  ∃ v : ℝ × ℝ, 
  (∀ x y : ℝ, 2 * y + x = 3 → v = (-2, -1)) :=
by
  sorry

end direction_vector_of_line_l108_108383


namespace min_ab_sum_l108_108911

theorem min_ab_sum (a b : ℤ) (h : a * b = 72) : a + b >= -17 :=
by
  sorry

end min_ab_sum_l108_108911


namespace land_percentage_relationship_l108_108530

variable {V : ℝ} -- Total taxable value of all land in the village
variable {x y z : ℝ} -- Percentages of Mr. William's land in types A, B, C

-- Conditions
axiom total_tax_collected : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 3840
axiom mr_william_tax : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 480

-- Prove the relationship
theorem land_percentage_relationship : (0.80 * x + 0.90 * y + 0.95 * z = 48000 / V) → (x + y + z = 100) := by
  sorry

end land_percentage_relationship_l108_108530


namespace orange_probability_l108_108560

theorem orange_probability (total_apples : ℕ) (total_oranges : ℕ) (other_fruits : ℕ)
  (h1 : total_apples = 20) (h2 : total_oranges = 10) (h3 : other_fruits = 0) :
  (total_oranges : ℚ) / (total_apples + total_oranges + other_fruits) = 1 / 3 :=
by
  sorry

end orange_probability_l108_108560


namespace gini_coefficient_separate_gini_coefficient_combined_l108_108900

-- Definitions based on provided conditions
def northern_residents : ℕ := 24
def southern_residents : ℕ := 6
def price_per_set : ℝ := 2000
def northern_PPC (x : ℝ) : ℝ := 13.5 - 9 * x
def southern_PPC (x : ℝ) : ℝ := 1.5 * x^2 - 24

-- Gini Coefficient when both regions operate separately
theorem gini_coefficient_separate : 
  ∃ G : ℝ, G = 0.2 :=
  sorry

-- Gini Coefficient change when blending productions as per Northern conditions
theorem gini_coefficient_combined :
  ∃ ΔG : ℝ, ΔG = 0.001 :=
  sorry

end gini_coefficient_separate_gini_coefficient_combined_l108_108900


namespace min_triangle_area_l108_108294

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1
noncomputable def circle_with_diameter_passing_origin (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let d := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  center.1^2 + center.2^2 = d / 4

theorem min_triangle_area (A B : ℝ × ℝ)
    (hA : hyperbola A.1 A.2)
    (hB : hyperbola B.1 B.2)
    (hc : circle_with_diameter_passing_origin A B) : 
    ∃ (S : ℝ), S = 2 :=
sorry

end min_triangle_area_l108_108294


namespace simple_interest_years_l108_108192

theorem simple_interest_years
  (CI : ℝ)
  (SI : ℝ)
  (p1 : ℝ := 4000) (r1 : ℝ := 0.10) (t1 : ℝ := 2)
  (p2 : ℝ := 1750) (r2 : ℝ := 0.08)
  (h1 : CI = p1 * (1 + r1) ^ t1 - p1)
  (h2 : SI = CI / 2)
  (h3 : SI = p2 * r2 * t2) :
  t2 = 3 :=
by
  sorry

end simple_interest_years_l108_108192


namespace martin_less_than_43_l108_108642

variable (C K M : ℕ)

-- Conditions
def campbell_correct := C = 35
def kelsey_correct := K = C + 8
def martin_fewer := M < K

-- Conclusion we want to prove
theorem martin_less_than_43 (h1 : campbell_correct C) (h2 : kelsey_correct C K) (h3 : martin_fewer K M) : M < 43 := 
by {
  sorry
}

end martin_less_than_43_l108_108642


namespace smallest_palindrome_in_base3_and_base5_l108_108764

def is_palindrome_base (b n : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_palindrome_in_base3_and_base5 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome_base 3 n ∧ is_palindrome_base 5 n ∧ n = 20 :=
by
  sorry

end smallest_palindrome_in_base3_and_base5_l108_108764


namespace remainder_of_expression_l108_108558

theorem remainder_of_expression (n : ℤ) (h : n % 100 = 99) : (n^2 + 2*n + 3 + n^3) % 100 = 1 :=
by
  sorry

end remainder_of_expression_l108_108558


namespace dave_total_earnings_l108_108731

def hourly_wage (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 7 else
  if day = 2 then 9 else
  if day = 3 then 8 else 
  0

def hours_worked (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 2 else
  if day = 2 then 3 else
  if day = 3 then 5 else 
  0

def unpaid_break (day : ℕ) : ℝ :=
  if day = 0 then 0.5 else
  if day = 1 then 0.25 else
  if day = 2 then 0 else
  if day = 3 then 0.5 else 
  0

def daily_earnings (day : ℕ) : ℝ :=
  (hours_worked day - unpaid_break day) * hourly_wage day

def net_earnings (day : ℕ) : ℝ :=
  daily_earnings day - (daily_earnings day * 0.1)

def total_net_earnings : ℝ :=
  net_earnings 0 + net_earnings 1 + net_earnings 2 + net_earnings 3

theorem dave_total_earnings : total_net_earnings = 97.43 := by
  sorry

end dave_total_earnings_l108_108731


namespace frog_vertical_boundary_prob_l108_108173

-- Define the type of points on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define the type of the rectangle
structure Rectangle where
  left_bottom : Point
  right_top : Point

-- Conditions
def start_point : Point := ⟨2, 3⟩
def boundary : Rectangle := ⟨⟨0, 0⟩, ⟨5, 5⟩⟩

-- Define the probability function
noncomputable def P (p : Point) : ℚ := sorry

-- Symmetry relations and recursive relations
axiom symmetry_P23 : P ⟨2, 3⟩ = P ⟨3, 3⟩
axiom symmetry_P22 : P ⟨2, 2⟩ = P ⟨3, 2⟩
axiom recursive_P23 : P ⟨2, 3⟩ = 1 / 4 + 1 / 4 * P ⟨2, 2⟩ + 1 / 4 * P ⟨1, 3⟩ + 1 / 4 * P ⟨3, 3⟩

-- Main Theorem
theorem frog_vertical_boundary_prob :
  P start_point = 2 / 3 := sorry

end frog_vertical_boundary_prob_l108_108173


namespace lana_total_winter_clothing_l108_108800

-- Define the number of boxes, scarves per box, and mittens per box as given in the conditions
def num_boxes : ℕ := 5
def scarves_per_box : ℕ := 7
def mittens_per_box : ℕ := 8

-- The total number of pieces of winter clothing is calculated as total scarves plus total mittens
def total_winter_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

-- State the theorem that needs to be proven
theorem lana_total_winter_clothing : total_winter_clothing = 75 := by
  sorry

end lana_total_winter_clothing_l108_108800


namespace boys_playing_both_sports_l108_108017

theorem boys_playing_both_sports : 
  ∀ (total boys basketball football neither both : ℕ), 
  total = 22 → boys = 22 → basketball = 13 → football = 15 → neither = 3 → 
  boys = basketball + football - both + neither → 
  both = 9 :=
by
  intros total boys basketball football neither both
  intros h_total h_boys h_basketball h_football h_neither h_formula
  sorry

end boys_playing_both_sports_l108_108017


namespace determine_S_l108_108430

theorem determine_S :
  (∃ k : ℝ, (∀ S R T : ℝ, R = k * (S / T)) ∧ (∃ S R T : ℝ, R = 2 ∧ S = 6 ∧ T = 3 ∧ 2 = k * (6 / 3))) →
  (∀ S R T : ℝ, R = 8 ∧ T = 2 → S = 16) :=
by
  sorry

end determine_S_l108_108430


namespace police_officer_can_catch_gangster_l108_108628

theorem police_officer_can_catch_gangster
  (a : ℝ) -- length of the side of the square
  (v_police : ℝ) -- maximum speed of the police officer
  (v_gangster : ℝ) -- maximum speed of the gangster
  (h_gangster_speed : v_gangster = 2.9 * v_police) :
  ∃ (t : ℝ), t ≥ 0 ∧ (a / (2 * v_police)) = t := sorry

end police_officer_can_catch_gangster_l108_108628


namespace min_max_area_of_CDM_l108_108606

theorem min_max_area_of_CDM (x y z : ℕ) (h1 : 2 * x + y = 4) (h2 : 2 * y + z = 8) :
  z = 4 :=
by
  sorry

end min_max_area_of_CDM_l108_108606


namespace tan_eq_tan_x2_sol_count_l108_108403

noncomputable def arctan1000 := Real.arctan 1000

theorem tan_eq_tan_x2_sol_count :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, 
    0 ≤ x ∧ x ≤ arctan1000 ∧ Real.tan x = Real.tan (x^2) →
    ∃ k : ℕ, k < n ∧ x = Real.sqrt (k * Real.pi + x) :=
sorry

end tan_eq_tan_x2_sol_count_l108_108403


namespace billiard_ball_returns_l108_108004

theorem billiard_ball_returns
  (w h : ℕ)
  (launch_angle : ℝ)
  (reflect_angle : ℝ)
  (start_A : ℝ × ℝ)
  (h_w : w = 2021)
  (h_h : h = 4300)
  (h_launch : launch_angle = 45)
  (h_reflect : reflect_angle = 45)
  (h_in_rect : ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2021 ∧ 0 ≤ y ∧ y ≤ 4300) :
  ∃ (bounces : ℕ), bounces = 294 :=
by
  sorry

end billiard_ball_returns_l108_108004


namespace total_capacity_is_1600_l108_108188

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l108_108188


namespace robotics_club_neither_l108_108422

theorem robotics_club_neither (total_students cs_students e_students both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 52)
  (h3 : e_students = 45)
  (h4 : both_students = 32) :
  total_students - (cs_students - both_students + e_students - both_students + both_students) = 15 :=
by
  sorry

end robotics_club_neither_l108_108422


namespace contrapositive_l108_108044

theorem contrapositive (a : ℝ) : (a > 0 → a > 1) → (a ≤ 1 → a ≤ 0) :=
by sorry

end contrapositive_l108_108044


namespace sin_alpha_beta_value_l108_108450

theorem sin_alpha_beta_value (α β : ℝ) (h1 : 13 * Real.sin α + 5 * Real.cos β = 9) (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 :=
by
  sorry

end sin_alpha_beta_value_l108_108450


namespace total_canoes_built_by_End_of_May_l108_108301

noncomputable def total_canoes_built (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem total_canoes_built_by_End_of_May :
  total_canoes_built 7 2 5 = 217 :=
by
  -- The proof would go here.
  sorry

end total_canoes_built_by_End_of_May_l108_108301


namespace factor_transformation_option_C_l108_108432

theorem factor_transformation_option_C (y : ℝ) : 
  4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 :=
sorry

end factor_transformation_option_C_l108_108432


namespace sum_of_shaded_cells_l108_108813

theorem sum_of_shaded_cells (a b c d e f : ℕ) 
  (h1: (a = 1 ∨ a = 2 ∨ a = 3) ∧ (b = 1 ∨ b = 2 ∨ b = 3) ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ 
       (d = 1 ∨ d = 2 ∨ d = 3) ∧ (e = 1 ∨ e = 2 ∨ e = 3) ∧ (f = 1 ∨ f = 2 ∨ f = 3))
  (h2: (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
       (d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
       (a ≠ d ∧ a ≠ f ∧ d ≠ f ∧ 
        b ≠ e ∧ b ≠ f ∧ c ≠ e ∧ c ≠ f))
  (h3: c = 3 ∧ d = 3 ∧ b = 2 ∧ e = 2)
  : b + e = 4 := 
sorry

end sum_of_shaded_cells_l108_108813


namespace time_for_A_to_complete_work_l108_108099

theorem time_for_A_to_complete_work (W : ℝ) (A B C : ℝ) (W_pos : 0 < W) (B_work : B = W / 40) (C_work : C = W / 20) : 
  (10 * (W / A) + 10 * (W / B) + 10 * (W / C) = W) → A = W / 40 :=
by 
  sorry

end time_for_A_to_complete_work_l108_108099


namespace eq_of_fraction_eq_l108_108181

variable {R : Type*} [Field R]

theorem eq_of_fraction_eq (a b : R) (h : (1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b))) : a = b :=
sorry

end eq_of_fraction_eq_l108_108181


namespace probability_of_log2_condition_l108_108221

noncomputable def probability_log_condition : ℝ :=
  let a := 0
  let b := 9
  let log_lower_bound := 1
  let log_upper_bound := 2
  let exp_lower_bound := 2^log_lower_bound
  let exp_upper_bound := 2^log_upper_bound
  (exp_upper_bound - exp_lower_bound) / (b - a)

theorem probability_of_log2_condition :
  probability_log_condition = 2 / 9 :=
by
  sorry

end probability_of_log2_condition_l108_108221


namespace Eldora_total_cost_l108_108653

-- Conditions
def paper_clip_cost : ℝ := 1.85
def index_card_cost : ℝ := 3.95 -- from Finn's purchase calculation
def total_cost (clips : ℝ) (cards : ℝ) (clip_price : ℝ) (card_price : ℝ) : ℝ :=
  (clips * clip_price) + (cards * card_price)

theorem Eldora_total_cost :
  total_cost 15 7 paper_clip_cost index_card_cost = 55.40 :=
by
  sorry

end Eldora_total_cost_l108_108653


namespace triangle_inequality_equivalence_l108_108292

theorem triangle_inequality_equivalence
    (a b c : ℝ) :
  (a < b + c ∧ b < a + c ∧ c < a + b) ↔
  (|b - c| < a ∧ a < b + c ∧ |a - c| < b ∧ b < a + c ∧ |a - b| < c ∧ c < a + b) ∧
  (max a (max b c) < b + c ∧ max a (max b c) < a + c ∧ max a (max b c) < a + b) :=
by sorry

end triangle_inequality_equivalence_l108_108292


namespace right_triangle_tangent_length_l108_108669

theorem right_triangle_tangent_length (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85)
  (h3 : ∀ (EF : ℝ), DE^2 + EF^2 = DF^2 → EF = 6): FQ = 6 :=
by
  sorry

end right_triangle_tangent_length_l108_108669


namespace distance_Owlford_Highcastle_l108_108109

open Complex

theorem distance_Owlford_Highcastle :
  let Highcastle := (0 : ℂ)
  let Owlford := (900 + 1200 * I : ℂ)
  dist Highcastle Owlford = 1500 := by
  sorry

end distance_Owlford_Highcastle_l108_108109


namespace chromium_percentage_in_second_alloy_l108_108564

theorem chromium_percentage_in_second_alloy
  (x : ℝ)
  (h1 : chromium_percentage_in_first_alloy = 15)
  (h2 : weight_first_alloy = 15)
  (h3 : weight_second_alloy = 35)
  (h4 : chromium_percentage_in_new_alloy = 10.1)
  (h5 : total_weight = weight_first_alloy + weight_second_alloy)
  (h6 : chromium_in_new_alloy = chromium_percentage_in_new_alloy / 100 * total_weight)
  (h7 : chromium_in_first_alloy = chromium_percentage_in_first_alloy / 100 * weight_first_alloy)
  (h8 : chromium_in_second_alloy = x / 100 * weight_second_alloy)
  (h9 : chromium_in_new_alloy = chromium_in_first_alloy + chromium_in_second_alloy) :
  x = 8 := by
  sorry

end chromium_percentage_in_second_alloy_l108_108564


namespace prod_ineq_min_value_l108_108656

theorem prod_ineq_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 := by
  sorry

end prod_ineq_min_value_l108_108656


namespace parameterized_line_segment_problem_l108_108986

theorem parameterized_line_segment_problem
  (p q r s : ℝ)
  (hq : q = 1)
  (hs : s = 2)
  (hpq : p + q = 6)
  (hrs : r + s = 9) :
  p^2 + q^2 + r^2 + s^2 = 79 := 
sorry

end parameterized_line_segment_problem_l108_108986


namespace solve_for_x_l108_108985

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x + 4 * x = 12 + 9 + 6 → x = 3 :=
by
  sorry

end solve_for_x_l108_108985


namespace right_angled_triangle_lines_l108_108313

theorem right_angled_triangle_lines (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 4 = 0 → x - 2 * y + 5 = 0 → m * x - 3 * y + 12 = 0 → 
    (exists x₁ y₁ : ℝ, 2 * x₁ - 1 * y₁ + 4 = 0 ∧ (x₁ - 5) ^ 2 / 4 + y₁ / (4) = (2^(1/2))^2) ∨ 
    (exists x₂ y₂ : ℝ, 1/2 * x₂ * y₂ - y₂ / 3 + 1 / 6 = 0 ∧ (x₂ + 5) ^ 2 / 9 + y₂ / 4 = small)) → 
    (m = -3 / 2 ∨ m = -6) :=
sorry

end right_angled_triangle_lines_l108_108313


namespace spell_AMCB_paths_equals_24_l108_108018

def central_A_reachable_M : Nat := 4
def M_reachable_C : Nat := 2
def C_reachable_B : Nat := 3

theorem spell_AMCB_paths_equals_24 :
  central_A_reachable_M * M_reachable_C * C_reachable_B = 24 := by
  sorry

end spell_AMCB_paths_equals_24_l108_108018


namespace directrix_of_parabola_l108_108443

theorem directrix_of_parabola (y : ℝ) : 
  (∃ y : ℝ, x = 1) ↔ (x = (1 / 4 : ℝ) * y^2) := 
sorry

end directrix_of_parabola_l108_108443


namespace jorge_total_spent_l108_108021

-- Definitions based on the problem conditions
def price_adult_ticket : ℝ := 10
def price_child_ticket : ℝ := 5
def num_adult_tickets : ℕ := 12
def num_child_tickets : ℕ := 12
def discount_adult : ℝ := 0.40
def discount_child : ℝ := 0.30
def extra_discount : ℝ := 0.10

-- The desired statement to prove
theorem jorge_total_spent :
  let total_adult_cost := num_adult_tickets * price_adult_ticket
  let total_child_cost := num_child_tickets * price_child_ticket
  let discounted_adult := total_adult_cost * (1 - discount_adult)
  let discounted_child := total_child_cost * (1 - discount_child)
  let total_cost_before_extra_discount := discounted_adult + discounted_child
  let final_cost := total_cost_before_extra_discount * (1 - extra_discount)
  final_cost = 102.60 :=
by 
  sorry

end jorge_total_spent_l108_108021


namespace solve_expression_l108_108597

theorem solve_expression : 6 / 3 - 2 - 8 + 2 * 8 = 8 := 
by 
  sorry

end solve_expression_l108_108597


namespace paige_finished_problems_at_school_l108_108907

-- Definitions based on conditions
def math_problems : ℕ := 43
def science_problems : ℕ := 12
def total_problems : ℕ := math_problems + science_problems
def problems_left : ℕ := 11

-- The main theorem we need to prove
theorem paige_finished_problems_at_school : total_problems - problems_left = 44 := by
  sorry

end paige_finished_problems_at_school_l108_108907


namespace henry_final_money_l108_108612

def initial_money : ℝ := 11.75
def received_from_relatives : ℝ := 18.50
def found_in_card : ℝ := 5.25
def spent_on_game : ℝ := 10.60
def donated_to_charity : ℝ := 3.15

theorem henry_final_money :
  initial_money + received_from_relatives + found_in_card - spent_on_game - donated_to_charity = 21.75 :=
by
  -- proof goes here
  sorry

end henry_final_money_l108_108612


namespace difference_in_overlap_l108_108541

variable (total_students : ℕ) (geometry_students : ℕ) (biology_students : ℕ)

theorem difference_in_overlap
  (h1 : total_students = 232)
  (h2 : geometry_students = 144)
  (h3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students;
  let min_overlap := geometry_students + biology_students - total_students;
  max_overlap - min_overlap = 88 :=
by 
  sorry

end difference_in_overlap_l108_108541


namespace ratio_of_ages_l108_108614

variable (T N : ℕ)
variable (sum_ages : T = T) -- This is tautological based on the given condition; we can consider it a given sum
variable (age_condition : T - N = 3 * (T - 3 * N))

theorem ratio_of_ages (T N : ℕ) (sum_ages : T = T) (age_condition : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end ratio_of_ages_l108_108614


namespace time_after_3577_minutes_l108_108243

-- Definitions
def startingTime : Nat := 6 * 60 -- 6:00 PM in minutes
def startDate : String := "2020-12-31"
def durationMinutes : Nat := 3577
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24

-- Theorem to prove that 3577 minutes after 6:00 PM on December 31, 2020 is January 3 at 5:37 AM
theorem time_after_3577_minutes : 
  (durationMinutes + startingTime) % (hoursInDay * minutesInHour) = 5 * minutesInHour + 37 :=
  by
  sorry -- proof goes here

end time_after_3577_minutes_l108_108243


namespace empty_set_implies_a_range_l108_108984

theorem empty_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(a * x^2 - 2 * a * x + 1 < 0)) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end empty_set_implies_a_range_l108_108984


namespace problem1_solution_problem2_solution_problem3_solution_l108_108063

-- Problem 1
theorem problem1_solution (x : ℝ) :
  (6 * x - 1) ^ 2 = 25 ↔ (x = 1 ∨ x = -2 / 3) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) :
  4 * x^2 - 1 = 12 * x ↔ (x = 3 / 2 + (Real.sqrt 10) / 2 ∨ x = 3 / 2 - (Real.sqrt 10) / 2) :=
sorry

-- Problem 3
theorem problem3_solution (x : ℝ) :
  x * (x - 7) = 8 * (7 - x) ↔ (x = 7 ∨ x = -8) :=
sorry

end problem1_solution_problem2_solution_problem3_solution_l108_108063


namespace sequence_an_properties_l108_108727

theorem sequence_an_properties
(S : ℕ → ℝ) (a : ℕ → ℝ)
(h_mean : ∀ n, 2 * a n = S n + 2) :
a 1 = 2 ∧ a 2 = 4 ∧ ∀ n, a n = 2 ^ n :=
by
  sorry

end sequence_an_properties_l108_108727


namespace total_area_of_folded_blankets_l108_108832

-- Define the initial conditions
def initial_area : ℕ := 8 * 8
def folds : ℕ := 4
def num_blankets : ℕ := 3

-- Define the hypothesis about folding
def folded_area (initial_area : ℕ) (folds : ℕ) : ℕ :=
  initial_area / (2 ^ folds)

-- The total area of all folded blankets
def total_folded_area (initial_area : ℕ) (folds : ℕ) (num_blankets : ℕ) : ℕ :=
  num_blankets * folded_area initial_area folds

-- The theorem we want to prove
theorem total_area_of_folded_blankets : total_folded_area initial_area folds num_blankets = 12 := by
  sorry

end total_area_of_folded_blankets_l108_108832


namespace calc_expression_l108_108415

theorem calc_expression :
  (- (2 / 5) : ℝ)^0 - (0.064 : ℝ)^(1/3) + 3^(Real.log (2 / 5) / Real.log 3) + Real.log 2 / Real.log 10 - Real.log (1 / 5) / Real.log 10 = 2 := 
by
  sorry

end calc_expression_l108_108415


namespace socorro_training_hours_l108_108777

theorem socorro_training_hours :
  let daily_multiplication_time := 10  -- in minutes
  let daily_division_time := 20        -- in minutes
  let training_days := 10              -- in days
  let minutes_per_hour := 60           -- minutes in an hour
  let daily_total_time := daily_multiplication_time + daily_division_time
  let total_training_time := daily_total_time * training_days
  total_training_time / minutes_per_hour = 5 :=
by sorry

end socorro_training_hours_l108_108777


namespace cos_30_deg_plus_2a_l108_108786

theorem cos_30_deg_plus_2a (a : ℝ) (h : Real.cos (Real.pi * (75 / 180) - a) = 1 / 3) : 
  Real.cos (Real.pi * (30 / 180) + 2 * a) = 7 / 9 := 
by 
  sorry

end cos_30_deg_plus_2a_l108_108786


namespace union_of_P_and_Q_l108_108176

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem union_of_P_and_Q : (P ∪ Q) = {x | -1 < x ∧ x < 3} := by
  -- skipping the proof
  sorry

end union_of_P_and_Q_l108_108176


namespace breadth_of_landscape_l108_108122

noncomputable def landscape_breadth (L : ℕ) (playground_area : ℕ) (total_area : ℕ) (B : ℕ) : Prop :=
  B = 6 * L ∧ playground_area = 4200 ∧ playground_area = (1 / 7) * total_area ∧ total_area = L * B

theorem breadth_of_landscape : ∃ (B : ℕ), ∀ (L : ℕ), landscape_breadth L 4200 29400 B → B = 420 :=
by
  intros
  sorry

end breadth_of_landscape_l108_108122


namespace dot_product_MN_MO_is_8_l108_108049

-- Define the circle O as a set of points (x, y) such that x^2 + y^2 = 9
def is_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the length of the chord MN in the circle
def chord_length (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  (x1 - x2)^2 + (y1 - y2)^2 = 16

-- Define the vector MN and MO
def vector_dot_product (M N O : ℝ × ℝ) : ℝ :=
  let (x1, y1) := M
  let (x2, y2) := N
  let (x0, y0) := O
  let v1 := (x2 - x1, y2 - y1)
  let v2 := (x0 - x1, y0 - y1)
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the origin point O (center of the circle)
def O : ℝ × ℝ := (0, 0)

-- The theorem to prove
theorem dot_product_MN_MO_is_8 (M N : ℝ × ℝ) (hM : is_circle M.1 M.2) (hN : is_circle N.1 N.2) (hMN : chord_length M N) :
  vector_dot_product M N O = 8 :=
sorry

end dot_product_MN_MO_is_8_l108_108049


namespace eccentricity_of_ellipse_l108_108489

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem eccentricity_of_ellipse :
  let P := (2, 3)
  let F1 := (-2, 0)
  let F2 := (2, 0)
  let d1 := distance P F1
  let d2 := distance P F2
  let a := (d1 + d2) / 2
  let c := distance F1 F2 / 2
  let e := c / a
  e = 1 / 2 := 
by 
  sorry

end eccentricity_of_ellipse_l108_108489


namespace pete_backwards_speed_l108_108905

variable (speed_pete_hands : ℕ) (speed_tracy_cartwheel : ℕ) (speed_susan_walk : ℕ) (speed_pete_backwards : ℕ)

axiom pete_hands_speed : speed_pete_hands = 2
axiom pete_hands_speed_quarter_tracy_cartwheel : speed_pete_hands = speed_tracy_cartwheel / 4
axiom tracy_cartwheel_twice_susan_walk : speed_tracy_cartwheel = 2 * speed_susan_walk
axiom pete_backwards_three_times_susan_walk : speed_pete_backwards = 3 * speed_susan_walk

theorem pete_backwards_speed : 
  speed_pete_backwards = 12 :=
by
  sorry

end pete_backwards_speed_l108_108905


namespace min_value_frac_l108_108639

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) :
  (1 / x + 1 / (3 * y)) = 4 :=
by
  sorry

end min_value_frac_l108_108639


namespace known_number_is_24_l108_108287

noncomputable def HCF (a b : ℕ) : ℕ := sorry
noncomputable def LCM (a b : ℕ) : ℕ := sorry

theorem known_number_is_24 (A B : ℕ) (h1 : B = 182)
  (h2 : HCF A B = 14)
  (h3 : LCM A B = 312) : A = 24 := by
  sorry

end known_number_is_24_l108_108287


namespace range_of_2x_plus_y_l108_108605

-- Given that positive numbers x and y satisfy this equation:
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x + y + 4 * x * y = 15 / 2

-- Define the range for 2x + y
def range_2x_plus_y (x y : ℝ) : ℝ :=
  2 * x + y

-- State the theorem.
theorem range_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : satisfies_equation x y) :
  3 ≤ range_2x_plus_y x y :=
by
  sorry

end range_of_2x_plus_y_l108_108605


namespace infinite_solutions_d_eq_5_l108_108928

theorem infinite_solutions_d_eq_5 :
  ∃ (d : ℝ), d = 5 ∧ ∀ (y : ℝ), 3 * (5 + d * y) = 15 * y + 15 :=
by
  sorry

end infinite_solutions_d_eq_5_l108_108928


namespace sparrow_swallow_equations_l108_108425

theorem sparrow_swallow_equations (x y : ℝ) : 
  (5 * x + 6 * y = 16) ∧ (4 * x + y = 5 * y + x) :=
  sorry

end sparrow_swallow_equations_l108_108425


namespace problem1_problem2_l108_108954

-- Statement for problem 1
theorem problem1 : 
  (-2020 - 2 / 3) + (2019 + 3 / 4) + (-2018 - 5 / 6) + (2017 + 1 / 2) = -2 - 1 / 4 := 
sorry

-- Statement for problem 2
theorem problem2 : 
  (-1 - 1 / 2) + (-2000 - 5 / 6) + (4000 + 3 / 4) + (-1999 - 2 / 3) = -5 / 4 := 
sorry

end problem1_problem2_l108_108954


namespace fraction_of_total_money_spent_on_dinner_l108_108337

-- Definitions based on conditions
def aaron_savings : ℝ := 40
def carson_savings : ℝ := 40
def total_savings : ℝ := aaron_savings + carson_savings

def ice_cream_cost_per_scoop : ℝ := 1.5
def scoops_each : ℕ := 6
def total_ice_cream_cost : ℝ := 2 * scoops_each * ice_cream_cost_per_scoop

def total_left : ℝ := 2

def total_spent : ℝ := total_savings - total_left
def dinner_cost : ℝ := total_spent - total_ice_cream_cost

-- Target statement
theorem fraction_of_total_money_spent_on_dinner : 
  (dinner_cost = 60) ∧ (total_savings = 80) → dinner_cost / total_savings = 3 / 4 :=
by
  intros h
  sorry

end fraction_of_total_money_spent_on_dinner_l108_108337


namespace range_of_x_l108_108973

variable (x : ℝ)

theorem range_of_x (h1 : 2 - x > 0) (h2 : x - 1 ≥ 0) : 1 ≤ x ∧ x < 2 := by
  sorry

end range_of_x_l108_108973


namespace operations_on_S_l108_108526

def is_element_of_S (x : ℤ) : Prop :=
  x = 0 ∨ ∃ n : ℤ, x = 2 * n

theorem operations_on_S (a b : ℤ) (ha : is_element_of_S a) (hb : is_element_of_S b) :
  (is_element_of_S (a + b)) ∧
  (is_element_of_S (a - b)) ∧
  (is_element_of_S (a * b)) ∧
  (¬ is_element_of_S (a / b)) ∧
  (¬ is_element_of_S ((a + b) / 2)) :=
by
  sorry

end operations_on_S_l108_108526


namespace initial_birds_count_l108_108006

variable (init_birds landed_birds total_birds : ℕ)

theorem initial_birds_count :
  (landed_birds = 8) →
  (total_birds = 20) →
  (init_birds + landed_birds = total_birds) →
  (init_birds = 12) :=
by
  intros h1 h2 h3
  sorry

end initial_birds_count_l108_108006


namespace find_b_value_l108_108888

theorem find_b_value (x y z : ℝ) (u t : ℕ) (h_pos_xyx : x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ t > 0)
  (h1 : (x + y - z) / z = 1) (h2 : (x - y + z) / y = 1) (h3 : (-x + y + z) / x = 1) 
  (ha : (x + y) * (y + z) * (z + x) / (x * y * z) = 8) (hu_t : u + t + u * t = 34) : (u + t = 10) :=
by
  sorry

end find_b_value_l108_108888


namespace inequality_b_2pow_a_a_2pow_neg_b_l108_108312

theorem inequality_b_2pow_a_a_2pow_neg_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  b * 2^a + a * 2^(-b) ≥ a + b :=
sorry

end inequality_b_2pow_a_a_2pow_neg_b_l108_108312


namespace drank_bottles_of_juice_l108_108919

theorem drank_bottles_of_juice
  (bottles_in_refrigerator : ℕ)
  (bottles_in_pantry : ℕ)
  (bottles_bought : ℕ)
  (bottles_left : ℕ)
  (initial_bottles := bottles_in_refrigerator + bottles_in_pantry)
  (total_bottles := initial_bottles + bottles_bought)
  (bottles_drank := total_bottles - bottles_left) :
  bottles_in_refrigerator = 4 ∧
  bottles_in_pantry = 4 ∧
  bottles_bought = 5 ∧
  bottles_left = 10 →
  bottles_drank = 3 :=
by sorry

end drank_bottles_of_juice_l108_108919


namespace income_percentage_increase_l108_108250

theorem income_percentage_increase (b : ℝ) (a : ℝ) (h : a = b * 0.75) :
  (b - a) / a * 100 = 33.33 :=
by
  sorry

end income_percentage_increase_l108_108250


namespace exponential_function_solution_l108_108163

theorem exponential_function_solution (a : ℝ) (h₁ : ∀ x : ℝ, a ^ x > 0) :
  (∃ y : ℝ, y = a ^ 2 ∧ y = 4) → a = 2 :=
by
  sorry

end exponential_function_solution_l108_108163


namespace parabola_tangent_parameter_l108_108453

theorem parabola_tangent_parameter (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0) :
  ∃ p : ℝ, (∀ y, y^2 + (2 * p * b / a) * y + (2 * p * c^2 / a) = 0) ↔ (p = 2 * a * c^2 / b^2) := 
by
  sorry

end parabola_tangent_parameter_l108_108453


namespace xy_condition_l108_108092

theorem xy_condition : (∀ x y : ℝ, x^2 + y^2 = 0 → xy = 0) ∧ ¬ (∀ x y : ℝ, xy = 0 → x^2 + y^2 = 0) := 
by
  sorry

end xy_condition_l108_108092


namespace chickens_count_l108_108733

theorem chickens_count (rabbits frogs : ℕ) (h_rabbits : rabbits = 49) (h_frogs : frogs = 37) :
  ∃ (C : ℕ), frogs + C = rabbits + 9 ∧ C = 21 :=
by
  sorry

end chickens_count_l108_108733


namespace pie_chart_degrees_for_cherry_pie_l108_108123

theorem pie_chart_degrees_for_cherry_pie :
  ∀ (total_students chocolate_pie apple_pie blueberry_pie : ℕ)
    (remaining_students cherry_pie_students lemon_pie_students : ℕ),
    total_students = 40 →
    chocolate_pie = 15 →
    apple_pie = 10 →
    blueberry_pie = 7 →
    remaining_students = total_students - chocolate_pie - apple_pie - blueberry_pie →
    cherry_pie_students = remaining_students / 2 →
    lemon_pie_students = remaining_students / 2 →
    (cherry_pie_students : ℝ) / (total_students : ℝ) * 360 = 36 :=
by
  sorry

end pie_chart_degrees_for_cherry_pie_l108_108123


namespace power_zero_equals_one_specific_case_l108_108996

theorem power_zero_equals_one 
    (a b : ℤ) 
    (h : a ≠ 0)
    (h2 : b ≠ 0) : 
    (a / b : ℚ) ^ 0 = 1 := 
by {
  sorry
}

-- Specific case
theorem specific_case : 
  ( ( (-123456789 : ℤ) / (9876543210 : ℤ) : ℚ ) ^ 0 = 1 ) := 
by {
  apply power_zero_equals_one;
  norm_num;
  sorry
}

end power_zero_equals_one_specific_case_l108_108996


namespace cream_cheese_cost_l108_108198

theorem cream_cheese_cost:
  ∃ (B C : ℝ), (2 * B + 3 * C = 12) ∧ (4 * B + 2 * C = 14) ∧ (C = 2.5) :=
by
  sorry

end cream_cheese_cost_l108_108198


namespace total_votes_l108_108286

-- Define the conditions
variable (V : ℝ) -- total number of votes polled
variable (w : ℝ) -- votes won by the winning candidate
variable (l : ℝ) -- votes won by the losing candidate
variable (majority : ℝ) -- majority votes

-- Define the specific values for the problem
def candidate_win_percentage (V : ℝ) : ℝ := 0.70 * V
def candidate_lose_percentage (V : ℝ) : ℝ := 0.30 * V

-- Define the majority condition
def majority_condition (V : ℝ) : Prop := (candidate_win_percentage V - candidate_lose_percentage V) = 240

-- The proof statement
theorem total_votes (V : ℝ) (h : majority_condition V) : V = 600 := by
  sorry

end total_votes_l108_108286


namespace zachary_pushups_l108_108580

theorem zachary_pushups (d z : ℕ) (h1 : d = z + 30) (h2 : d = 37) : z = 7 := by
  sorry

end zachary_pushups_l108_108580


namespace rectangle_perimeter_l108_108915

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 4 * (a + b)) : 2 * (a + b) = 36 := by
  sorry

end rectangle_perimeter_l108_108915


namespace calculate_value_l108_108089

theorem calculate_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calculate_value_l108_108089


namespace land_for_cattle_l108_108409

-- Define the conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def crop_production : ℕ := 70

-- Statement to prove
theorem land_for_cattle : total_land - (house_and_machinery + future_expansion + crop_production) = 40 :=
by
  sorry

end land_for_cattle_l108_108409


namespace problem1_problem2_l108_108107

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

-- Problem (1): Prove the inequality f(x-1) > 0 given b = 1.
theorem problem1 (a x : ℝ) : f (x - 1) a 1 > 0 := sorry

-- Problem (2): Prove the values of a and b such that the range of f(x) for x ∈ [-1, 2] is [5/4, 2].
theorem problem2 (a b : ℝ) (H₁ : f (-1) a b = 5 / 4) (H₂ : f 2 a b = 2) :
    (a = 3 ∧ b = 2) ∨ (a = -4 ∧ b = -3) := sorry

end problem1_problem2_l108_108107


namespace system_of_equations_solution_l108_108194

theorem system_of_equations_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 3) : 
  x = 4 ∧ y = 1 :=
by
  sorry

end system_of_equations_solution_l108_108194


namespace fat_rings_per_group_l108_108369

theorem fat_rings_per_group (F : ℕ)
  (h1 : ∀ F, (70 * (F + 4)) = (40 * (F + 4)) + 180)
  : F = 2 :=
sorry

end fat_rings_per_group_l108_108369


namespace martha_cards_l108_108436

theorem martha_cards :
  let initial_cards := 3
  let emily_cards := 25
  let alex_cards := 43
  let jenny_cards := 58
  let sam_cards := 14
  initial_cards + emily_cards + alex_cards + jenny_cards - sam_cards = 115 := 
by
  sorry

end martha_cards_l108_108436


namespace original_cost_price_l108_108381

theorem original_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1275) 
  (h2 : loss_percentage = 15) 
  (h3 : SP = (1 - loss_percentage / 100) * C) : 
  C = 1500 := 
by 
  sorry

end original_cost_price_l108_108381


namespace find_solutions_l108_108032

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 9 * x ^ 2 + 6

theorem find_solutions :
  ∃ x1 x2 x3 : ℝ, f x1 = Real.sqrt 2 ∧ f x2 = Real.sqrt 2 ∧ f x3 = Real.sqrt 2 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
sorry

end find_solutions_l108_108032


namespace sams_charge_per_sheet_is_1_5_l108_108449

variable (x : ℝ)
variable (a : ℝ) -- John's Photo World's charge per sheet
variable (b : ℝ) -- Sam's Picture Emporium's one-time sitting fee
variable (c : ℝ) -- John's Photo World's one-time sitting fee
variable (n : ℕ) -- Number of sheets

def johnsCost (n : ℕ) (a c : ℝ) := n * a + c
def samsCost (n : ℕ) (x b : ℝ) := n * x + b

theorem sams_charge_per_sheet_is_1_5 :
  ∀ (a b c : ℝ) (n : ℕ), a = 2.75 → b = 140 → c = 125 → n = 12 →
  johnsCost n a c = samsCost n x b → x = 1.50 := by
  intros a b c n ha hb hc hn h
  sorry

end sams_charge_per_sheet_is_1_5_l108_108449


namespace CEMC_additional_employees_l108_108472

variable (t : ℝ)

def initialEmployees (t : ℝ) := t + 40

def finalEmployeesMooseJaw (t : ℝ) := 1.25 * t

def finalEmployeesOkotoks : ℝ := 26

def finalEmployeesTotal (t : ℝ) := finalEmployeesMooseJaw t + finalEmployeesOkotoks

def netChangeInEmployees (t : ℝ) := finalEmployeesTotal t - initialEmployees t

theorem CEMC_additional_employees (t : ℝ) (h : t = 120) : 
    netChangeInEmployees t = 16 := 
by
    sorry

end CEMC_additional_employees_l108_108472


namespace sum_of_digits_base_2_315_l108_108318

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l108_108318


namespace fraction_division_l108_108338

-- Definition of fractions involved
def frac1 : ℚ := 4 / 9
def frac2 : ℚ := 5 / 8

-- Statement of the proof problem
theorem fraction_division :
  (frac1 / frac2) = 32 / 45 :=
by {
  sorry
}

end fraction_division_l108_108338


namespace find_variable_l108_108416

def expand : ℤ → ℤ := 3*2*6
    
theorem find_variable (a n some_variable : ℤ) (h : (3 - 7 + a = 3)):
  some_variable = -17 :=
sorry

end find_variable_l108_108416


namespace find_c_l108_108566

theorem find_c (c : ℝ) (h : (-(c / 3) + -(c / 5) = 30)) : c = -56.25 :=
sorry

end find_c_l108_108566


namespace g_f_3_eq_1476_l108_108283

def f (x : ℝ) : ℝ := x^3 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_f_3_eq_1476 : g (f 3) = 1476 :=
by
  sorry

end g_f_3_eq_1476_l108_108283


namespace speed_second_half_l108_108199

theorem speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) :
    total_time = 12 → first_half_speed = 35 → total_distance = 560 → 
    (280 / (12 - (280 / 35)) = 70) :=
by
  intros ht hf hd
  sorry

end speed_second_half_l108_108199


namespace large_square_area_l108_108498

theorem large_square_area (l w : ℕ) (h1 : 2 * (l + w) = 28) : (l + w) * (l + w) = 196 :=
by {
  sorry
}

end large_square_area_l108_108498


namespace sector_angle_l108_108846

theorem sector_angle (r l : ℝ) (h1 : l + 2 * r = 6) (h2 : 1/2 * l * r = 2) : 
  l / r = 1 ∨ l / r = 4 := 
sorry

end sector_angle_l108_108846


namespace fraction_is_correct_l108_108441

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem fraction_is_correct : (f (g (f 3))) / (g (f (g 3))) = 59 / 19 :=
by
  sorry

end fraction_is_correct_l108_108441


namespace min_distance_PA_l108_108009

theorem min_distance_PA :
  let A : ℝ × ℝ := (0, 1)
  ∀ (P : ℝ × ℝ), (∃ x : ℝ, x > 0 ∧ P = (x, (x + 2) / x)) →
  ∃ d : ℝ, d = 2 ∧ ∀ Q : ℝ × ℝ, (∃ x : ℝ, x > 0 ∧ Q = (x, (x + 2) / x)) → dist A Q ≥ d :=
by
  sorry

end min_distance_PA_l108_108009


namespace cyclists_speeds_product_l108_108969

theorem cyclists_speeds_product (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h₁ : 6 / u = 6 / v + 1 / 12) 
  (h₂ : v / 3 = u / 3 + 4) : 
  u * v = 864 := 
by
  sorry

end cyclists_speeds_product_l108_108969


namespace trigonometric_expression_eq_neg3_l108_108809

theorem trigonometric_expression_eq_neg3
  {α : ℝ} (h : Real.tan α = 1 / 2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) /
  ((Real.sin (-α))^2 - (Real.sin (5 * π / 2 - α))^2) = -3 :=
sorry

end trigonometric_expression_eq_neg3_l108_108809


namespace more_stickers_correct_l108_108749

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23
def second_box_stickers : ℕ := total_stickers - first_box_stickers
def more_stickers_in_second_box : ℕ := second_box_stickers - first_box_stickers

theorem more_stickers_correct : more_stickers_in_second_box = 12 := by
  sorry

end more_stickers_correct_l108_108749


namespace market_survey_l108_108516

theorem market_survey (X Y : ℕ) (h1 : X / Y = 9) (h2 : X + Y = 400) : X = 360 :=
by
  sorry

end market_survey_l108_108516


namespace length_of_rope_l108_108570

-- Define the given conditions
variable (L : ℝ)
variable (h1 : 0.6 * L = 0.69)

-- The theorem to prove
theorem length_of_rope (L : ℝ) (h1 : 0.6 * L = 0.69) : L = 1.15 :=
by
  sorry

end length_of_rope_l108_108570


namespace correct_email_sequence_l108_108998

theorem correct_email_sequence :
  let a := "Open the mailbox"
  let b := "Enter the recipient's address"
  let c := "Enter the subject"
  let d := "Enter the content of the email"
  let e := "Click 'Compose'"
  let f := "Click 'Send'"
  (a, e, b, c, d, f) = ("Open the mailbox", "Click 'Compose'", "Enter the recipient's address", "Enter the subject", "Enter the content of the email", "Click 'Send'") := 
sorry

end correct_email_sequence_l108_108998


namespace has_three_real_zeros_l108_108929

noncomputable def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

theorem has_three_real_zeros (m : ℝ) : 
    (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ m = 0 ∧ f x₂ m = 0 ∧ f x₃ m = 0) ↔ (-4 < m ∧ m < 4) :=
sorry

end has_three_real_zeros_l108_108929


namespace leak_emptying_time_l108_108185

-- Definitions based on given conditions
def tank_fill_rate_without_leak : ℚ := 1 / 3
def combined_fill_and_leak_rate : ℚ := 1 / 4

-- Leak emptying time to be proven
theorem leak_emptying_time (R : ℚ := tank_fill_rate_without_leak) (C : ℚ := combined_fill_and_leak_rate) :
  (1 : ℚ) / (R - C) = 12 := by
  sorry

end leak_emptying_time_l108_108185


namespace AM_QM_Muirhead_Inequality_l108_108080

open Real

theorem AM_QM_Muirhead_Inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  ((a + b + c) / 3 = sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) ∧
  (sqrt ((a^2 + b^2 + c^2) / 3) = ((ab / c) + (bc / a) + (ca / b)) / 3 ↔ a = b ∧ b = c) :=
by sorry

end AM_QM_Muirhead_Inequality_l108_108080


namespace parallelogram_area_l108_108103

-- Define a plane rectangular coordinate system
structure PlaneRectangularCoordinateSystem :=
(axis : ℝ)

-- Define the properties of a square
structure Square :=
(side_length : ℝ)

-- Define the properties of a parallelogram in a perspective drawing
structure Parallelogram :=
(side_length: ℝ)

-- Define the conditions of the problem
def problem_conditions (s : Square) (p : Parallelogram) :=
  s.side_length = 4 ∨ s.side_length = 8 ∧ 
  p.side_length = 4

-- Statement of the problem
theorem parallelogram_area (s : Square) (p : Parallelogram)
  (h : problem_conditions s p) :
  p.side_length * p.side_length = 16 ∨ p.side_length * p.side_length = 64 :=
by {
  sorry
}

end parallelogram_area_l108_108103


namespace treaty_signed_on_friday_l108_108265

def days_between (start_date : Nat) (end_date : Nat) : Nat := sorry

def day_of_week (start_day : Nat) (days_elapsed : Nat) : Nat :=
  (start_day + days_elapsed) % 7

def is_leap_year (year : Nat) : Bool :=
  if year % 4 = 0 then
    if year % 100 = 0 then
      if year % 400 = 0 then true else false
    else true
  else false

noncomputable def days_from_1802_to_1814 : Nat :=
  let leap_years := [1804, 1808, 1812]
  let normal_year_days := 365 * 9
  let leap_year_days := 366 * 3
  normal_year_days + leap_year_days

noncomputable def days_from_feb_5_to_apr_11_1814 : Nat :=
  24 + 31 + 11 -- days in February, March, and April 11

noncomputable def total_days_elapsed : Nat :=
  days_from_1802_to_1814 + days_from_feb_5_to_apr_11_1814

noncomputable def start_day : Nat := 5 -- Friday (0 = Sunday, ..., 5 = Friday, 6 = Saturday)

theorem treaty_signed_on_friday : day_of_week start_day total_days_elapsed = 5 := sorry

end treaty_signed_on_friday_l108_108265


namespace factorize_def_l108_108374

def factorize_polynomial (p q r : Polynomial ℝ) : Prop :=
  p = q * r

theorem factorize_def (p q r : Polynomial ℝ) :
  factorize_polynomial p q r → p = q * r :=
  sorry

end factorize_def_l108_108374


namespace quadratic_equation_solutions_l108_108061

theorem quadratic_equation_solutions : ∀ x : ℝ, x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := 
by sorry

end quadratic_equation_solutions_l108_108061


namespace area_of_moon_slice_l108_108855

-- Definitions of the conditions
def larger_circle_radius := 5
def larger_circle_center := (2, 0)
def smaller_circle_radius := 2
def smaller_circle_center := (0, 0)

-- Prove the area of the moon slice
theorem area_of_moon_slice : 
  (1/4) * (larger_circle_radius^2 * Real.pi) - (1/4) * (smaller_circle_radius^2 * Real.pi) = (21 * Real.pi) / 4 :=
by
  sorry

end area_of_moon_slice_l108_108855


namespace divide_money_equally_l108_108158

-- Length of the road built by companies A, B, and total length of the road
def length_A : ℕ := 6
def length_B : ℕ := 10
def total_length : ℕ := 16

-- Money contributed by company C
def money_C : ℕ := 16 * 10^6

-- The equal contribution each company should finance
def equal_contribution := total_length / 3

-- Deviations from the expected length for firms A and B
def deviation_A := length_A - (total_length / 3)
def deviation_B := length_B - (total_length / 3)

-- The ratio based on the deviations to divide the money
def ratio_A := deviation_A * (total_length / (deviation_A + deviation_B))
def ratio_B := deviation_B * (total_length / (deviation_A + deviation_B))

-- The amount of money firms A and B should receive, respectively
def money_A := money_C * ratio_A / total_length
def money_B := money_C * ratio_B / total_length

-- Theorem statement
theorem divide_money_equally : money_A = 2 * 10^6 ∧ money_B = 14 * 10^6 :=
by 
  sorry

end divide_money_equally_l108_108158


namespace sum_of_leading_digits_l108_108229

def leading_digit (n : ℕ) (x : ℝ) : ℕ := sorry

def M := 10^500 - 1

def g (r : ℕ) : ℕ := leading_digit r (M^(1 / r))

theorem sum_of_leading_digits :
  g 3 + g 4 + g 5 + g 7 + g 8 = 10 := sorry

end sum_of_leading_digits_l108_108229


namespace library_visitors_on_sundays_l108_108034

theorem library_visitors_on_sundays 
  (average_other_days : ℕ) 
  (average_per_day : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ) 
  (total_visitors_month : ℕ)
  (visitors_other_days : ℕ) 
  (total_visitors_sundays : ℕ) :
  average_other_days = 240 →
  average_per_day = 285 →
  total_days = 30 →
  sundays = 5 →
  other_days = total_days - sundays →
  total_visitors_month = average_per_day * total_days →
  visitors_other_days = average_other_days * other_days →
  total_visitors_sundays + visitors_other_days = total_visitors_month →
  total_visitors_sundays = sundays * (510 : ℕ) :=
by
  sorry


end library_visitors_on_sundays_l108_108034


namespace seven_people_different_rolls_l108_108405

def rolls_different (rolls : Fin 7 -> Fin 6) : Prop :=
  ∀ i : Fin 7, rolls i ≠ rolls ⟨(i + 1) % 7, sorry⟩

def probability_rolls_different : ℚ :=
  (625 : ℚ) / 2799

theorem seven_people_different_rolls (rolls : Fin 7 -> Fin 6) :
  (∃ rolls, rolls_different rolls) ->
  probability_rolls_different = 625 / 2799 :=
sorry

end seven_people_different_rolls_l108_108405


namespace difference_of_squares_l108_108510

theorem difference_of_squares (n : ℤ) : 4 - n^2 = (2 + n) * (2 - n) := 
by
  -- Proof goes here
  sorry

end difference_of_squares_l108_108510


namespace other_root_is_minus_5_l108_108100

-- conditions
def polynomial (x : ℝ) := x^4 - x^3 - 18 * x^2 + 52 * x + (-40 : ℝ)
def r1 := 2
def f_of_r1_eq_zero : polynomial r1 = 0 := by sorry -- given condition

-- the proof problem
theorem other_root_is_minus_5 : ∃ r, polynomial r = 0 ∧ r ≠ r1 ∧ r = -5 :=
by
  sorry

end other_root_is_minus_5_l108_108100


namespace sum_three_numbers_l108_108770

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_three_numbers_l108_108770


namespace martin_family_ice_cream_cost_l108_108323

theorem martin_family_ice_cream_cost (R : ℤ)
  (kiddie_scoop_cost : ℤ) (double_scoop_cost : ℤ)
  (total_cost : ℤ) :
  kiddie_scoop_cost = 3 → 
  double_scoop_cost = 6 → 
  total_cost = 32 →
  2 * R + 2 * kiddie_scoop_cost + 3 * double_scoop_cost = total_cost →
  R = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end martin_family_ice_cream_cost_l108_108323


namespace letters_posting_ways_l108_108600

theorem letters_posting_ways :
  let mailboxes := 4
  let letters := 3
  (mailboxes ^ letters) = 64 :=
by
  let mailboxes := 4
  let letters := 3
  show (mailboxes ^ letters) = 64
  sorry

end letters_posting_ways_l108_108600


namespace elephant_entry_rate_l108_108420

-- Define the variables and constants
def initial_elephants : ℕ := 30000
def exit_rate : ℕ := 2880
def exit_time : ℕ := 4
def enter_time : ℕ := 7
def final_elephants : ℕ := 28980

-- Prove the rate of new elephants entering the park
theorem elephant_entry_rate :
  (final_elephants - (initial_elephants - exit_rate * exit_time)) / enter_time = 1500 :=
by
  sorry -- placeholder for the proof

end elephant_entry_rate_l108_108420


namespace time_relationship_l108_108758

variable (T x : ℝ)
variable (h : T = x + (2/6) * x)

theorem time_relationship : T = (4/3) * x := by 
sorry

end time_relationship_l108_108758


namespace cos_double_angle_sum_l108_108090

theorem cos_double_angle_sum
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 := by
  sorry

end cos_double_angle_sum_l108_108090


namespace product_in_M_l108_108326

def M : Set ℤ := {x | ∃ (a b : ℤ), x = a^2 - b^2}

theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M :=
by
  sorry

end product_in_M_l108_108326


namespace probability_less_than_8_rings_l108_108397

def P_10_ring : ℝ := 0.20
def P_9_ring : ℝ := 0.30
def P_8_ring : ℝ := 0.10

theorem probability_less_than_8_rings : 
  (1 - (P_10_ring + P_9_ring + P_8_ring)) = 0.40 :=
by
  sorry

end probability_less_than_8_rings_l108_108397


namespace angle_value_l108_108830

theorem angle_value (y : ℝ) (h1 : 2 * y + 140 = 360) : y = 110 :=
by {
  -- Proof will be written here
  sorry
}

end angle_value_l108_108830


namespace amount_of_money_l108_108300

theorem amount_of_money (x y : ℝ) 
  (h1 : x + 1/2 * y = 50) 
  (h2 : 2/3 * x + y = 50) : 
  (x + 1/2 * y = 50) ∧ (2/3 * x + y = 50) :=
by
  exact ⟨h1, h2⟩ 

end amount_of_money_l108_108300


namespace inequality_abc_l108_108523

theorem inequality_abc (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) :=
by
  -- Proof goes here
  sorry

end inequality_abc_l108_108523


namespace shortest_distance_from_parabola_to_line_l108_108922

open Real

noncomputable def parabola_point (M : ℝ × ℝ) : Prop :=
  M.snd^2 = 6 * M.fst

noncomputable def distance_to_line (M : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * M.fst + b * M.snd + c) / sqrt (a^2 + b^2)

theorem shortest_distance_from_parabola_to_line (M : ℝ × ℝ) (h : parabola_point M) :
  distance_to_line M 3 (-4) 12 = 3 :=
by
  sorry

end shortest_distance_from_parabola_to_line_l108_108922


namespace vector_satisfy_condition_l108_108987

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  parametrize : ℝ → Point

def l : Line :=
  { parametrize := λ t => {x := 1 + 4 * t, y := 4 + 3 * t} }

def m : Line :=
  { parametrize := λ s => {x := -5 + 4 * s, y := 6 + 3 * s} }

def A (t : ℝ) : Point := l.parametrize t
def B (s : ℝ) : Point := m.parametrize s

-- The specific point for A and B are not used directly in the further proof statement.

def v : Point := { x := -6, y := 8 }

theorem vector_satisfy_condition :
  ∃ v1 v2 : ℝ, (v1 * -6) + (v2 * 8) = 2 ∧ (v1 = -6 ∧ v2 = 8) :=
sorry

end vector_satisfy_condition_l108_108987


namespace no_real_coeff_quadratic_with_roots_sum_and_product_l108_108871

theorem no_real_coeff_quadratic_with_roots_sum_and_product (a b c : ℝ) (h : a ≠ 0) :
  ¬ ∃ (α β : ℝ), (α = a + b + c) ∧ (β = a * b * c) ∧ (α + β = -b / a) ∧ (α * β = c / a) :=
by
  sorry

end no_real_coeff_quadratic_with_roots_sum_and_product_l108_108871


namespace perfect_square_trinomial_l108_108490

noncomputable def p (k : ℝ) (x : ℝ) : ℝ :=
  4 * x^2 + 2 * k * x + 9

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, p k x = (2 * x + b)^2) → (k = 6 ∨ k = -6) :=
by 
  intro h
  sorry

end perfect_square_trinomial_l108_108490


namespace mean_of_y_l108_108988

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def regression_line (x : ℝ) : ℝ :=
  2 * x + 45

theorem mean_of_y (y₁ y₂ y₃ y₄ y₅ : ℝ) :
  mean [regression_line 1, regression_line 5, regression_line 7, regression_line 13, regression_line 19] = 63 := by
  sorry

end mean_of_y_l108_108988


namespace perimeter_of_stadium_l108_108424

-- Define the length and breadth as given conditions.
def length : ℕ := 100
def breadth : ℕ := 300

-- Define the perimeter function for a rectangle.
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Prove that the perimeter of the stadium is 800 meters given the length and breadth.
theorem perimeter_of_stadium : perimeter length breadth = 800 := 
by
  -- Placeholder for the formal proof.
  sorry

end perimeter_of_stadium_l108_108424


namespace initial_percentage_of_grape_juice_l108_108469

theorem initial_percentage_of_grape_juice
  (P : ℝ)    -- P is the initial percentage in decimal
  (h₁ : 0 ≤ P ∧ P ≤ 1)    -- P is a valid probability
  (h₂ : 40 * P + 10 = 0.36 * 50):    -- Given condition from the problem
  P = 0.2 := 
sorry

end initial_percentage_of_grape_juice_l108_108469


namespace first_divisor_is_13_l108_108722

theorem first_divisor_is_13 (x : ℤ) (h : (377 / x) / 29 * (1/4 : ℚ) / 2 = (1/8 : ℚ)) : x = 13 := by
  sorry

end first_divisor_is_13_l108_108722


namespace larger_square_side_length_l108_108729

theorem larger_square_side_length (s1 s2 : ℝ) (h1 : s1 = 5) (h2 : s2 = s1 * 3) (a1 a2 : ℝ) (h3 : a1 = s1^2) (h4 : a2 = s2^2) : s2 = 15 := 
by
  sorry

end larger_square_side_length_l108_108729


namespace ball_arrangement_l108_108410

theorem ball_arrangement : ∃ (n : ℕ), n = 120 ∧
  (∀ (ball_count : ℕ), ball_count = 20 → ∃ (box1 box2 box3 : ℕ), 
    box1 ≥ 1 ∧ box2 ≥ 2 ∧ box3 ≥ 3 ∧ box1 + box2 + box3 = ball_count) :=
by
  sorry

end ball_arrangement_l108_108410


namespace dad_gave_nickels_l108_108822

-- Definitions
def original_nickels : ℕ := 9
def total_nickels_after : ℕ := 12

-- Theorem to be proven
theorem dad_gave_nickels {original_nickels total_nickels_after : ℕ} : 
    total_nickels_after - original_nickels = 3 := 
by
  /- Sorry proof omitted -/
  sorry

end dad_gave_nickels_l108_108822


namespace cos_pi_over_6_minus_2alpha_l108_108466

open Real

noncomputable def tan_plus_pi_over_6 (α : ℝ) := tan (α + π / 6) = 2

theorem cos_pi_over_6_minus_2alpha (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π) 
  (h2 : tan_plus_pi_over_6 α) : 
  cos (π / 6 - 2 * α) = 4 / 5 :=
sorry

end cos_pi_over_6_minus_2alpha_l108_108466


namespace school_spent_on_grass_seeds_bottle_capacity_insufficient_l108_108513

-- Problem 1: Cost Calculation
theorem school_spent_on_grass_seeds (kg_seeds : ℝ) (cost_per_kg : ℝ) (total_cost : ℝ) 
  (h1 : kg_seeds = 3.3) (h2 : cost_per_kg = 9.48) :
  total_cost = 31.284 :=
  by
    sorry

-- Problem 2: Bottle Capacity
theorem bottle_capacity_insufficient (total_seeds : ℝ) (max_capacity_per_bottle : ℝ) (num_bottles : ℕ)
  (h1 : total_seeds = 3.3) (h2 : max_capacity_per_bottle = 0.35) (h3 : num_bottles = 9) :
  3.3 > 0.35 * 9 :=
  by
    sorry

end school_spent_on_grass_seeds_bottle_capacity_insufficient_l108_108513


namespace bill_original_selling_price_l108_108702

variable (P : ℝ) (S : ℝ) (S_new : ℝ)

theorem bill_original_selling_price :
  (S = P + 0.10 * P) ∧ (S_new = 0.90 * P + 0.27 * P) ∧ (S_new = S + 28) →
  S = 440 :=
by
  intro h
  sorry

end bill_original_selling_price_l108_108702


namespace no_integer_solutions_l108_108478

theorem no_integer_solutions (x y : ℤ) (hx : x ≠ 1) : (x^7 - 1) / (x - 1) ≠ y^5 - 1 :=
by
  sorry

end no_integer_solutions_l108_108478


namespace range_of_x_l108_108726

noncomputable 
def proposition_p (x : ℝ) : Prop := 6 - 3 * x ≥ 0

noncomputable 
def proposition_q (x : ℝ) : Prop := 1 / (x + 1) < 0

theorem range_of_x (x : ℝ) : proposition_p x ∧ ¬proposition_q x → x ∈ Set.Icc (-1 : ℝ) (2 : ℝ) := by
  sorry

end range_of_x_l108_108726


namespace new_op_4_3_l108_108839

def new_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem new_op_4_3 : new_op 4 3 = 13 :=
by
  -- Placeholder for the proof
  sorry

end new_op_4_3_l108_108839


namespace two_false_propositions_l108_108321

theorem two_false_propositions (a : ℝ) :
  (¬((a > -3) → (a > -6))) ∧ (¬((a > -6) → (a > -3))) → (¬(¬(a > -3) → ¬(a > -6))) :=
by
  sorry

end two_false_propositions_l108_108321


namespace max_regions_two_convex_polygons_l108_108716

theorem max_regions_two_convex_polygons (M N : ℕ) (hM : M > N) :
    ∃ R, R = 2 * N + 2 := 
sorry

end max_regions_two_convex_polygons_l108_108716


namespace imaginary_part_div_z1_z2_l108_108993

noncomputable def z1 := 1 - 3 * Complex.I
noncomputable def z2 := 3 + Complex.I

theorem imaginary_part_div_z1_z2 : 
  Complex.im ((1 + 3 * Complex.I) / (3 + Complex.I)) = 4 / 5 := 
by 
  sorry

end imaginary_part_div_z1_z2_l108_108993


namespace theresa_crayons_count_l108_108440

noncomputable def crayons_teresa (initial_teresa_crayons : Nat) 
                                 (initial_janice_crayons : Nat) 
                                 (shared_with_nancy : Nat)
                                 (given_to_mark : Nat)
                                 (received_from_nancy : Nat) : Nat := 
  initial_teresa_crayons + received_from_nancy

theorem theresa_crayons_count : crayons_teresa 32 12 (12 / 2) 3 8 = 40 := by
  -- Given: Theresa initially has 32 crayons.
  -- Janice initially has 12 crayons.
  -- Janice shares half of her crayons with Nancy: 12 / 2 = 6 crayons.
  -- Janice gives 3 crayons to Mark.
  -- Theresa receives 8 crayons from Nancy.
  -- Therefore: Theresa will have 32 + 8 = 40 crayons.
  sorry

end theresa_crayons_count_l108_108440


namespace composite_expr_l108_108615

open Nat

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem composite_expr (n : ℕ) : n ≥ 2 ↔ is_composite (3^(2*n + 1) - 2^(2*n + 1) - 6^n) :=
sorry

end composite_expr_l108_108615


namespace machineA_finishing_time_l108_108925

theorem machineA_finishing_time
  (A : ℝ)
  (hA : 0 < A)
  (hB : 0 < 12)
  (hC : 0 < 6)
  (h_total_time : 0 < 2)
  (h_work_done_per_hour : (1 / A) + (1 / 12) + (1 / 6) = 1 / 2) :
  A = 4 := sorry

end machineA_finishing_time_l108_108925


namespace sqrt_simplification_l108_108455

theorem sqrt_simplification : Real.sqrt 360000 = 600 :=
by 
  sorry

end sqrt_simplification_l108_108455


namespace shem_earnings_l108_108948

theorem shem_earnings (kem_hourly: ℝ) (ratio: ℝ) (workday_hours: ℝ) (shem_hourly: ℝ) (shem_daily: ℝ) :
  kem_hourly = 4 →
  ratio = 2.5 →
  shem_hourly = kem_hourly * ratio →
  workday_hours = 8 →
  shem_daily = shem_hourly * workday_hours →
  shem_daily = 80 :=
by
  -- Proof omitted
  sorry

end shem_earnings_l108_108948


namespace polynomial_divisibility_l108_108577

theorem polynomial_divisibility (n : ℕ) : (∀ x : ℤ, (x^2 + x + 1 ∣ x^(2*n) + x^n + 1)) ↔ (3 ∣ n) := by
  sorry

end polynomial_divisibility_l108_108577


namespace no_solution_exists_l108_108514

theorem no_solution_exists :
  ¬ ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 + x2 = 1) ∧
    (x2 + x3 - x4 = 1) ∧
    (0 ≤ x1) ∧
    (0 ≤ x2) ∧
    (0 ≤ x3) ∧
    (0 ≤ x4) ∧
    ∀ (F : ℝ), F = x1 - x2 + 2 * x3 - x4 → 
    ∀ (b : ℝ), F ≤ b :=
by sorry

end no_solution_exists_l108_108514


namespace find_real_numbers_l108_108730

theorem find_real_numbers :
  ∀ (x y z : ℝ), x^2 - y*z = |y - z| + 1 ∧ y^2 - z*x = |z - x| + 1 ∧ z^2 - x*y = |x - y| + 1 ↔
  (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
  (x = 4/3 ∧ y = -5/3 ∧ z = 4/3) ∨
  (x = -5/3 ∧ y = 4/3 ∧ z = 4/3) ∨
  (x = -4/3 ∧ y = -4/3 ∧ z = 5/3) ∨
  (x = -4/3 ∧ y = 5/3 ∧ z = -4/3) ∨
  (x = 5/3 ∧ y = -4/3 ∧ z = -4/3) :=
by
  sorry

end find_real_numbers_l108_108730


namespace binom_20_10_l108_108317

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l108_108317


namespace steven_set_aside_9_grapes_l108_108091

-- Define the conditions based on the problem statement
def total_seeds_needed : ℕ := 60
def average_seeds_per_apple : ℕ := 6
def average_seeds_per_pear : ℕ := 2
def average_seeds_per_grape : ℕ := 3
def apples_set_aside : ℕ := 4
def pears_set_aside : ℕ := 3
def additional_seeds_needed : ℕ := 3

-- Calculate the number of seeds from apples and pears
def seeds_from_apples : ℕ := apples_set_aside * average_seeds_per_apple
def seeds_from_pears : ℕ := pears_set_aside * average_seeds_per_pear

-- Calculate the number of seeds that Steven already has from apples and pears
def seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculate the remaining seeds needed from grapes
def seeds_needed_from_grapes : ℕ := total_seeds_needed - seeds_from_apples_and_pears - additional_seeds_needed

-- Calculate the number of grapes set aside
def grapes_set_aside : ℕ := seeds_needed_from_grapes / average_seeds_per_grape

theorem steven_set_aside_9_grapes : grapes_set_aside = 9 :=
by 
  sorry

end steven_set_aside_9_grapes_l108_108091


namespace trapezoid_perimeter_l108_108676

theorem trapezoid_perimeter (height : ℝ) (radius : ℝ) (LM KN : ℝ) (LM_eq : LM = 16.5) (KN_eq : KN = 37.5)
  (LK MN : ℝ) (LK_eq : LK = 37.5) (MN_eq : MN = 37.5) (H : height = 36) (R : radius = 11) : 
  (LM + KN + LK + MN) = 129 :=
by
  -- The proof is omitted; only the statement is provided as specified.
  sorry

end trapezoid_perimeter_l108_108676


namespace cupcake_cost_l108_108756

def initialMoney : ℝ := 20
def moneyFromMother : ℝ := 2 * initialMoney
def totalMoney : ℝ := initialMoney + moneyFromMother
def costPerBoxOfCookies : ℝ := 3
def numberOfBoxesOfCookies : ℝ := 5
def costOfCookies : ℝ := costPerBoxOfCookies * numberOfBoxesOfCookies
def moneyAfterCookies : ℝ := totalMoney - costOfCookies
def moneyLeftAfterCupcakes : ℝ := 30
def numberOfCupcakes : ℝ := 10

noncomputable def costPerCupcake : ℝ := 
  (moneyAfterCookies - moneyLeftAfterCupcakes) / numberOfCupcakes

theorem cupcake_cost :
  costPerCupcake = 1.50 :=
by 
  sorry

end cupcake_cost_l108_108756


namespace books_in_library_l108_108155

theorem books_in_library (n_shelves : ℕ) (n_books_per_shelf : ℕ) (h_shelves : n_shelves = 1780) (h_books_per_shelf : n_books_per_shelf = 8) :
  n_shelves * n_books_per_shelf = 14240 :=
by
  -- Skipping the proof as instructed
  sorry

end books_in_library_l108_108155


namespace estimated_total_fish_l108_108635

-- Let's define the conditions first
def total_fish_marked := 100
def second_catch_total := 200
def marked_in_second_catch := 5

-- The variable representing the total number of fish in the pond
variable (x : ℕ)

-- The theorem stating that given the conditions, the total number of fish is 4000
theorem estimated_total_fish
  (h1 : total_fish_marked = 100)
  (h2 : second_catch_total = 200)
  (h3 : marked_in_second_catch = 5)
  (h4 : (marked_in_second_catch : ℝ) / second_catch_total = (total_fish_marked : ℝ) / x) :
  x = 4000 := 
sorry

end estimated_total_fish_l108_108635


namespace curve_is_line_l108_108945

theorem curve_is_line (θ : ℝ) (hθ : θ = 5 * Real.pi / 6) : 
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (r : ℝ), r = 0 ↔
  (∃ p : ℝ × ℝ, p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ ∧
                p.1 * a + p.2 * b = 0) :=
sorry

end curve_is_line_l108_108945


namespace regular_square_pyramid_side_edge_length_l108_108563

theorem regular_square_pyramid_side_edge_length 
  (base_edge_length : ℝ)
  (volume : ℝ)
  (h_base_edge_length : base_edge_length = 4 * Real.sqrt 2)
  (h_volume : volume = 32) :
  ∃ side_edge_length : ℝ, side_edge_length = 5 :=
by sorry

end regular_square_pyramid_side_edge_length_l108_108563


namespace solve_eqn_l108_108244

noncomputable def root_expr (a b k x : ℝ) : ℝ := Real.sqrt ((a + b * Real.sqrt k)^x)

theorem solve_eqn: {x : ℝ | root_expr 3 2 2 x + root_expr 3 (-2) 2 x = 6} = {2, -2} :=
by
  sorry

end solve_eqn_l108_108244


namespace jony_stop_block_correct_l108_108402

-- Jony's walk parameters
def start_time : ℕ := 7 -- In hours, but it is not used directly
def start_block : ℕ := 10
def end_block : ℕ := 90
def stop_time : ℕ := 40 -- Jony stops walking after 40 minutes starting from 07:00
def speed : ℕ := 100 -- meters per minute
def block_length : ℕ := 40 -- meters

-- Function to calculate the stop block given the parameters
def stop_block (start_block end_block stop_time speed block_length : ℕ) : ℕ :=
  let total_distance := stop_time * speed
  let outbound_distance := (end_block - start_block) * block_length
  let remaining_distance := total_distance - outbound_distance
  let blocks_walked_back := remaining_distance / block_length
  end_block - blocks_walked_back

-- The statement to prove
theorem jony_stop_block_correct :
  stop_block start_block end_block stop_time speed block_length = 70 :=
by
  sorry

end jony_stop_block_correct_l108_108402


namespace nancy_initial_bottle_caps_l108_108938

theorem nancy_initial_bottle_caps (found additional_bottle_caps: ℕ) (total_bottle_caps: ℕ) (h1: additional_bottle_caps = 88) (h2: total_bottle_caps = 179) : 
  (total_bottle_caps - additional_bottle_caps) = 91 :=
by
  sorry

end nancy_initial_bottle_caps_l108_108938


namespace find_range_of_a_l108_108111

variable {a : ℝ}
variable {x : ℝ}

theorem find_range_of_a (h₁ : x ∈ Set.Ioo (-2:ℝ) (-1:ℝ)) :
  ∃ a, a ∈ Set.Icc (1:ℝ) (2:ℝ) ∧ (x + 1)^2 < Real.log (|x|) / Real.log a :=
by
  sorry

end find_range_of_a_l108_108111


namespace surcharge_X_is_2_17_percent_l108_108904

def priceX : ℝ := 575
def priceY : ℝ := 530
def surchargeY : ℝ := 0.03
def totalSaved : ℝ := 41.60

theorem surcharge_X_is_2_17_percent :
  let surchargeX := (2.17 / 100)
  let totalCostX := priceX + (priceX * surchargeX)
  let totalCostY := priceY + (priceY * surchargeY)
  (totalCostX - totalCostY = totalSaved) →
  surchargeX * 100 = 2.17 :=
by
  sorry

end surcharge_X_is_2_17_percent_l108_108904


namespace evaluate_expression_l108_108805

theorem evaluate_expression : 500 * (500 ^ 500) * 500 = 500 ^ 502 := by
  sorry

end evaluate_expression_l108_108805


namespace initial_dozens_of_doughnuts_l108_108920

theorem initial_dozens_of_doughnuts (doughnuts_eaten doughnuts_left : ℕ)
  (h_eaten : doughnuts_eaten = 8)
  (h_left : doughnuts_left = 16) :
  (doughnuts_eaten + doughnuts_left) / 12 = 2 := by
  sorry

end initial_dozens_of_doughnuts_l108_108920


namespace Jolene_raised_total_money_l108_108803

-- Definitions for the conditions
def babysits_earning_per_family : ℤ := 30
def number_of_families : ℤ := 4
def cars_earning_per_car : ℤ := 12
def number_of_cars : ℤ := 5

-- Calculation of total earnings
def babysitting_earnings : ℤ := babysits_earning_per_family * number_of_families
def car_washing_earnings : ℤ := cars_earning_per_car * number_of_cars
def total_earnings : ℤ := babysitting_earnings + car_washing_earnings

-- The proof statement
theorem Jolene_raised_total_money : total_earnings = 180 := by
  sorry

end Jolene_raised_total_money_l108_108803


namespace number_of_boys_at_reunion_l108_108699

theorem number_of_boys_at_reunion (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
by
  sorry

end number_of_boys_at_reunion_l108_108699


namespace annette_weights_more_l108_108179

variable (A C S B : ℝ)

theorem annette_weights_more :
  A + C = 95 ∧
  C + S = 87 ∧
  A + S = 97 ∧
  C + B = 100 ∧
  A + C + B = 155 →
  A - S = 8 := by
  sorry

end annette_weights_more_l108_108179


namespace intersection_a_zero_range_of_a_l108_108551

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end intersection_a_zero_range_of_a_l108_108551


namespace stratified_sampling_young_employees_l108_108931

variable (total_employees elderly_employees middle_aged_employees young_employees sample_size : ℕ)

-- Conditions
axiom total_employees_eq : total_employees = 750
axiom elderly_employees_eq : elderly_employees = 150
axiom middle_aged_employees_eq : middle_aged_employees = 250
axiom young_employees_eq : young_employees = 350
axiom sample_size_eq : sample_size = 15

-- The proof problem
theorem stratified_sampling_young_employees :
  young_employees / total_employees * sample_size = 7 :=
by
  sorry

end stratified_sampling_young_employees_l108_108931


namespace distance_between_trees_l108_108264

-- Lean 4 statement for the proof problem
theorem distance_between_trees (n : ℕ) (yard_length : ℝ) (h_n : n = 26) (h_length : yard_length = 600) :
  yard_length / (n - 1) = 24 :=
by
  sorry

end distance_between_trees_l108_108264


namespace intersection_M_N_l108_108135

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = { z | 0 ≤ z ∧ z ≤ 1 } := by
  sorry

end intersection_M_N_l108_108135


namespace sqrt_fraction_simplified_l108_108359

theorem sqrt_fraction_simplified :
  Real.sqrt (4 / 3) = 2 * Real.sqrt 3 / 3 :=
by sorry

end sqrt_fraction_simplified_l108_108359


namespace calories_in_dressing_l108_108687

noncomputable def lettuce_calories : ℝ := 50
noncomputable def carrot_calories : ℝ := 2 * lettuce_calories
noncomputable def crust_calories : ℝ := 600
noncomputable def pepperoni_calories : ℝ := crust_calories / 3
noncomputable def cheese_calories : ℝ := 400

noncomputable def salad_calories : ℝ := lettuce_calories + carrot_calories
noncomputable def pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories

noncomputable def salad_eaten : ℝ := salad_calories / 4
noncomputable def pizza_eaten : ℝ := pizza_calories / 5

noncomputable def total_eaten : ℝ := salad_eaten + pizza_eaten

theorem calories_in_dressing : ((330 : ℝ) - total_eaten) = 52.5 := by
  sorry

end calories_in_dressing_l108_108687


namespace find_cos_sum_l108_108670

-- Defining the conditions based on the problem
variable (P A B C D : Type) (α β : ℝ)

-- Assumptions stating the given conditions
def regular_quadrilateral_pyramid (P A B C D : Type) : Prop :=
  -- Placeholder for the exact definition of a regular quadrilateral pyramid
  sorry

def dihedral_angle_lateral_base (P A B C D : Type) (α : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between lateral face and base is α
  sorry

def dihedral_angle_adjacent_lateral (P A B C D : Type) (β : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between two adjacent lateral faces is β
  sorry

-- The final theorem that we want to prove
theorem find_cos_sum (P A B C D : Type) (α β : ℝ)
  (H1 : regular_quadrilateral_pyramid P A B C D)
  (H2 : dihedral_angle_lateral_base P A B C D α)
  (H3 : dihedral_angle_adjacent_lateral P A B C D β) :
  2 * Real.cos β + Real.cos (2 * α) = -1 :=
sorry

end find_cos_sum_l108_108670


namespace percentage_less_than_y_l108_108812

variable (w x y z : ℝ)

-- Given conditions
variable (h1 : w = 0.60 * x)
variable (h2 : x = 0.60 * y)
variable (h3 : z = 1.50 * w)

theorem percentage_less_than_y : ( (y - z) / y) * 100 = 46 := by
  sorry

end percentage_less_than_y_l108_108812


namespace percentage_of_men_l108_108701

theorem percentage_of_men (M W : ℝ) (h1 : M + W = 1) (h2 : 0.60 * M + 0.2364 * W = 0.40) : M = 0.45 :=
by
  sorry

end percentage_of_men_l108_108701


namespace percentage_greater_than_88_l108_108539

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h : x = 88 + percentage * 88) (hx : x = 132) : 
  percentage = 0.5 :=
by
  sorry

end percentage_greater_than_88_l108_108539


namespace quadrilateral_diagonals_l108_108426

theorem quadrilateral_diagonals (a b c d e f : ℝ) 
  (hac : a > c) 
  (hbd : b ≥ d) 
  (hapc : a = c) 
  (hdiag1 : e^2 = (a - b)^2 + b^2) 
  (hdiag2 : f^2 = (c + b)^2 + b^2) :
  e^4 - f^4 = (a + c) / (a - c) * (d^2 * (2 * a * c + d^2) - b^2 * (2 * a * c + b^2)) :=
by
  sorry

end quadrilateral_diagonals_l108_108426


namespace car_speed_return_trip_l108_108427

noncomputable def speed_return_trip (d : ℕ) (v_ab : ℕ) (v_avg : ℕ) : ℕ := 
  (2 * d * v_avg) / (2 * v_avg - v_ab)

theorem car_speed_return_trip :
  let d := 180
  let v_ab := 90
  let v_avg := 60
  speed_return_trip d v_ab v_avg = 45 :=
by
  simp [speed_return_trip]
  sorry

end car_speed_return_trip_l108_108427


namespace rhombus_diagonal_l108_108245

theorem rhombus_diagonal (side : ℝ) (short_diag : ℝ) (long_diag : ℝ) 
  (h1 : side = 37) (h2 : short_diag = 40) :
  long_diag = 62 :=
sorry

end rhombus_diagonal_l108_108245


namespace spring_excursion_participants_l108_108997

theorem spring_excursion_participants (water fruit neither both total : ℕ) 
  (h_water : water = 80) 
  (h_fruit : fruit = 70) 
  (h_neither : neither = 6) 
  (h_both : both = total / 2) 
  (h_total_eq : total = water + fruit - both + neither) : 
  total = 104 := 
  sorry

end spring_excursion_participants_l108_108997


namespace sequence_conditions_general_formulas_sum_of_first_n_terms_l108_108504

noncomputable def arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n n = a_n 1 + d * (n - 1)

noncomputable def geometric_sequence (b_n : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, q > 0 ∧ ∀ n : ℕ, b_n (n + 1) = b_n n * q

variables {a_n b_n c_n : ℕ → ℤ}
variables (d q : ℤ) (d_pos : 0 < d) (hq : q > 0)
variables (S_n : ℕ → ℤ)

axiom initial_conditions : a_n 1 = 2 ∧ b_n 1 = 2 ∧ a_n 3 = 8 ∧ b_n 3 = 8

theorem sequence_conditions : arithmetic_sequence a_n ∧ geometric_sequence b_n := sorry

theorem general_formulas :
  (∀ n : ℕ, a_n n = 3 * n - 1) ∧
  (∀ n : ℕ, b_n n = 2^n) := sorry

theorem sum_of_first_n_terms :
  (∀ n : ℕ, S_n n = 3 * 2^(n+1) - n - 6) := sorry

end sequence_conditions_general_formulas_sum_of_first_n_terms_l108_108504


namespace measure_angle_YPZ_is_142_l108_108507

variables (X Y Z : Type) [Inhabited X] [Inhabited Y] [Inhabited Z]
variables (XM YN ZO : Type) [Inhabited XM] [Inhabited YN] [Inhabited ZO]

noncomputable def angle_XYZ : ℝ := 65
noncomputable def angle_XZY : ℝ := 38
noncomputable def angle_YXZ : ℝ := 180 - angle_XYZ - angle_XZY
noncomputable def angle_YNZ : ℝ := 90 - angle_YXZ
noncomputable def angle_ZMY : ℝ := 90 - angle_XYZ
noncomputable def angle_YPZ : ℝ := 180 - angle_YNZ - angle_ZMY

theorem measure_angle_YPZ_is_142 :
  angle_YPZ = 142 := sorry

end measure_angle_YPZ_is_142_l108_108507


namespace work_completion_days_l108_108001

theorem work_completion_days
  (A B : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : A = 1 / 20)
  : 1 / (A + B / 2) = 15 :=
by 
  sorry

end work_completion_days_l108_108001


namespace points_per_game_without_bonus_l108_108073

-- Definition of the conditions
def b : ℕ := 82
def n : ℕ := 79
def P : ℕ := 15089

-- Theorem statement
theorem points_per_game_without_bonus :
  (P - b * n) / n = 109 :=
by
  -- Proof will be filled in here
  sorry

end points_per_game_without_bonus_l108_108073


namespace lina_collects_stickers_l108_108356

theorem lina_collects_stickers :
  let a := 3
  let d := 2
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 120 :=
by
  sorry

end lina_collects_stickers_l108_108356


namespace smallest_n_interval_l108_108149

theorem smallest_n_interval :
  ∃ n : ℕ, (∃ x : ℤ, ⌊10 ^ n / x⌋ = 2006) ∧ 7 ≤ n ∧ n ≤ 12 :=
sorry

end smallest_n_interval_l108_108149


namespace custom_op_4_3_l108_108355

-- Define the custom operation a * b
def custom_op (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the theorem to be proven
theorem custom_op_4_3 : custom_op 4 3 = 19 := 
by
sorry

end custom_op_4_3_l108_108355


namespace find_x_l108_108266

-- Definitions based on the problem conditions
def angle_CDE : ℝ := 90 -- angle CDE in degrees
def angle_ECB : ℝ := 68 -- angle ECB in degrees

-- Theorem statement
theorem find_x (x : ℝ) 
  (h1 : angle_CDE = 90) 
  (h2 : angle_ECB = 68) 
  (h3 : angle_CDE + x + angle_ECB = 180) : 
  x = 22 := 
by
  sorry

end find_x_l108_108266


namespace tailor_charges_30_per_hour_l108_108322

noncomputable def tailor_hourly_rate (shirts pants : ℕ) (shirt_hours pant_hours total_cost : ℝ) :=
  total_cost / (shirts * shirt_hours + pants * pant_hours)

theorem tailor_charges_30_per_hour :
  tailor_hourly_rate 10 12 1.5 3 1530 = 30 := by
  sorry

end tailor_charges_30_per_hour_l108_108322


namespace unique_integer_sequence_l108_108293

theorem unique_integer_sequence (a : ℕ → ℤ) :
  a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) →
  ∃! (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) :=
sorry

end unique_integer_sequence_l108_108293


namespace value_of_w_div_x_l108_108707

theorem value_of_w_div_x (w x y : ℝ) 
  (h1 : w / x = a) 
  (h2 : w / y = 1 / 5) 
  (h3 : (x + y) / y = 2.2) : 
  w / x = 6 / 25 := by
  sorry

end value_of_w_div_x_l108_108707


namespace fraction_sum_l108_108688

theorem fraction_sum (y : ℝ) (a b : ℤ) (h : y = 3.834834834) (h_frac : y = (a : ℝ) / b) (h_coprime : Int.gcd a b = 1) : a + b = 4830 :=
sorry

end fraction_sum_l108_108688


namespace slope_of_AB_is_1_l108_108389

noncomputable def circle1 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 + 2 * p.2 - 11 = 0 }
noncomputable def circle2 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 14 * p.1 + 12 * p.2 + 60 = 0 }
def is_on_circle1 (p : ℝ × ℝ) := p ∈ circle1
def is_on_circle2 (p : ℝ × ℝ) := p ∈ circle2

theorem slope_of_AB_is_1 :
  ∃ A B : ℝ × ℝ,
  is_on_circle1 A ∧ is_on_circle2 A ∧
  is_on_circle1 B ∧ is_on_circle2 B ∧
  (B.2 - A.2) / (B.1 - A.1) = 1 :=
sorry

end slope_of_AB_is_1_l108_108389


namespace original_cost_of_dress_l108_108863

theorem original_cost_of_dress (x : ℝ) 
  (h1 : x / 2 - 10 < x)
  (h2 : x - (x / 2 - 10) = 80) : 
  x = 140 := 
sorry

end original_cost_of_dress_l108_108863


namespace worker_usual_time_l108_108433

theorem worker_usual_time (T : ℝ) (S : ℝ) (h₀ : S > 0) (h₁ : (4 / 5) * S * (T + 10) = S * T) : T = 40 :=
sorry

end worker_usual_time_l108_108433


namespace triangle_angle_sum_l108_108062

theorem triangle_angle_sum (angle_Q R P : ℝ)
  (h1 : R = 3 * angle_Q)
  (h2 : angle_Q = 30)
  (h3 : P + angle_Q + R = 180) :
    P = 60 :=
by
  sorry

end triangle_angle_sum_l108_108062


namespace find_a_l108_108234

theorem find_a (a : ℝ) (h1 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) (ha_pos : a > 0) (ha_neq1 : a ≠ 1) : a = 2 :=
sorry

end find_a_l108_108234


namespace discount_difference_l108_108647

theorem discount_difference (x : ℝ) (h1 : x = 8000) : 
  (x * 0.7) - ((x * 0.8) * 0.9) = 160 :=
by
  rw [h1]
  sorry

end discount_difference_l108_108647


namespace michael_cleanings_total_l108_108404

theorem michael_cleanings_total (baths_per_week : ℕ) (showers_per_week : ℕ) (weeks_in_year : ℕ) 
  (h_baths : baths_per_week = 2) (h_showers : showers_per_week = 1) (h_weeks : weeks_in_year = 52) :
  (baths_per_week + showers_per_week) * weeks_in_year = 156 :=
by 
  -- Omitting proof as instructed.
  sorry

end michael_cleanings_total_l108_108404


namespace triangle_inequality_l108_108532

open Real

variables {a b c S : ℝ}

-- Assuming a, b, c are the sides of a triangle
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
-- Assuming S is the area of the triangle
axiom Herons_area : S = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_inequality : 
  a^2 + b^2 + c^2 ≥ 4 * S * sqrt 3 ∧ (a^2 + b^2 + c^2 = 4 * S * sqrt 3 ↔ a = b ∧ b = c) := sorry

end triangle_inequality_l108_108532


namespace find_z_l108_108279

-- Given conditions as Lean definitions
def consecutive (x y z : ℕ) : Prop := x = z + 2 ∧ y = z + 1 ∧ x > y ∧ y > z
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + 3 * z = 5 * y + 11

-- The statement to be proven
theorem find_z (x y z : ℕ) (h1 : consecutive x y z) (h2 : equation x y z) : z = 3 :=
sorry

end find_z_l108_108279


namespace divide_L_shaped_plaque_into_four_equal_parts_l108_108145

-- Definition of an "L"-shaped plaque and the condition of symmetric cuts
def L_shaped_plaque (a b : ℕ) : Prop := (a > 0) ∧ (b > 0)

-- Statement of the proof problem
theorem divide_L_shaped_plaque_into_four_equal_parts (a b : ℕ) (h : L_shaped_plaque a b) :
  ∃ (p1 p2 : ℕ → ℕ → Prop),
    (∀ x y, p1 x y ↔ (x < a/2 ∧ y < b/2)) ∧
    (∀ x y, p2 x y ↔ (x < a/2 ∧ y >= b/2) ∨ (x >= a/2 ∧ y < b/2) ∨ (x >= a/2 ∧ y >= b/2)) :=
sorry

end divide_L_shaped_plaque_into_four_equal_parts_l108_108145


namespace arithmetic_geometric_product_l108_108864

theorem arithmetic_geometric_product :
  let a (n : ℕ) := 2 * n - 1
  let b (n : ℕ) := 2 ^ (n - 1)
  b (a 1) * b (a 3) * b (a 5) = 4096 :=
by 
  sorry

end arithmetic_geometric_product_l108_108864


namespace model1_best_fitting_effect_l108_108106

-- Definitions for the correlation coefficients of the models
def R1 : ℝ := 0.98
def R2 : ℝ := 0.80
def R3 : ℝ := 0.50
def R4 : ℝ := 0.25

-- Main theorem stating Model 1 has the best fitting effect
theorem model1_best_fitting_effect : |R1| > |R2| ∧ |R1| > |R3| ∧ |R1| > |R4| :=
by sorry

end model1_best_fitting_effect_l108_108106


namespace multiply_exponents_l108_108482

theorem multiply_exponents (a : ℝ) : (6 * a^2) * (1/2 * a^3) = 3 * a^5 := by
  sorry

end multiply_exponents_l108_108482


namespace reachable_target_l108_108170

-- Define the initial state of the urn
def initial_urn_state : (ℕ × ℕ) := (150, 50)

-- Define the operations as changes in counts of black and white marbles
def operation1 (state : ℕ × ℕ) := (state.1 - 2, state.2)
def operation2 (state : ℕ × ℕ) := (state.1 - 1, state.2)
def operation3 (state : ℕ × ℕ) := (state.1, state.2 - 2)
def operation4 (state : ℕ × ℕ) := (state.1 + 2, state.2 - 3)

-- Define a predicate that a state can be reached from the initial state
def reachable (target : ℕ × ℕ) : Prop :=
  ∃ n1 n2 n3 n4 : ℕ, 
    operation1^[n1] (operation2^[n2] (operation3^[n3] (operation4^[n4] initial_urn_state))) = target

-- The theorem to be proved
theorem reachable_target : reachable (1, 2) :=
sorry

end reachable_target_l108_108170


namespace minimum_value_of_fraction_l108_108140

theorem minimum_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (4 / a + 9 / b) ≥ 25 :=
by
  sorry

end minimum_value_of_fraction_l108_108140


namespace dot_product_OA_OB_l108_108077

theorem dot_product_OA_OB :
  let A := (Real.cos 110, Real.sin 110)
  let B := (Real.cos 50, Real.sin 50)
  (A.1 * B.1 + A.2 * B.2) = 1 / 2 :=
by
  sorry

end dot_product_OA_OB_l108_108077


namespace rectangle_length_is_4_l108_108875

theorem rectangle_length_is_4 (w l : ℝ) (h_length : l = w + 3) (h_area : l * w = 4) : l = 4 := 
sorry

end rectangle_length_is_4_l108_108875


namespace last_digit_of_expression_l108_108816

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression (n : ℕ) : last_digit (n ^ 9999 - n ^ 5555) = 0 :=
by
  sorry

end last_digit_of_expression_l108_108816


namespace worker_and_robot_capacity_additional_workers_needed_l108_108182

-- Definitions and conditions
def worker_capacity (x : ℕ) : Prop :=
  (1 : ℕ) * x + 420 = 420 + x

def time_equivalence (x : ℕ) : Prop :=
  900 * 10 * x = 600 * (x + 420)

-- First part of the proof problem
theorem worker_and_robot_capacity (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  x = 30 ∧ x + 420 = 450 :=
by
  sorry

-- Second part of the proof problem
theorem additional_workers_needed (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  3 * (x + 420) * 2 < 3600 →
  2 * 30 * 15 ≥ 3600 - 2 * 3 * (x + 420) :=
by
  sorry

end worker_and_robot_capacity_additional_workers_needed_l108_108182


namespace correct_calculation_l108_108792

theorem correct_calculation (a b x y : ℝ) :
  (7 * a^2 * b - 7 * b * a^2 = 0) ∧ 
  (¬ (6 * a + 4 * b = 10 * a * b)) ∧ 
  (¬ (7 * x^2 * y - 3 * x^2 * y = 4 * x^4 * y^2)) ∧ 
  (¬ (8 * x^2 + 8 * x^2 = 16 * x^4)) :=
sorry

end correct_calculation_l108_108792


namespace sum_odd_digits_from_1_to_200_l108_108354

/-- Function to compute the sum of odd digits of a number -/
def odd_digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (fun d => d % 2 = 1) |>.sum

/-- Statement of the problem to prove the sum of the odd digits of numbers from 1 to 200 is 1000 -/
theorem sum_odd_digits_from_1_to_200 : (Finset.range 200).sum odd_digit_sum = 1000 := 
  sorry

end sum_odd_digits_from_1_to_200_l108_108354


namespace sum_of_ages_l108_108834

variable (S M : ℝ)  -- Variables for Sarah's and Matt's ages

-- Conditions
def sarah_older := S = M + 8
def future_age_relationship := S + 10 = 3 * (M - 5)

-- Theorem: The sum of their current ages is 41
theorem sum_of_ages (h1 : sarah_older S M) (h2 : future_age_relationship S M) : S + M = 41 := by
  sorry

end sum_of_ages_l108_108834


namespace area_of_triangle_ABC_l108_108637

/--
Given a triangle ABC where BC is 12 cm and the height from A
perpendicular to BC is 15 cm, prove that the area of the triangle is 90 cm^2.
-/
theorem area_of_triangle_ABC (BC : ℝ) (hA : ℝ) (h_BC : BC = 12) (h_hA : hA = 15) : 
  1/2 * BC * hA = 90 := 
sorry

end area_of_triangle_ABC_l108_108637


namespace visited_both_countries_l108_108696

theorem visited_both_countries {Total Iceland Norway Neither Both : ℕ} 
  (h1 : Total = 50) 
  (h2 : Iceland = 25)
  (h3 : Norway = 23)
  (h4 : Neither = 23) 
  (h5 : Total - Neither = 27) 
  (h6 : Iceland + Norway - Both = 27) : 
  Both = 21 := 
by
  sorry

end visited_both_countries_l108_108696


namespace find_a9_l108_108583

variable {a : ℕ → ℤ}  -- Define a as a sequence of integers
variable (d : ℤ) (a3 : ℤ) (a4 : ℤ)

-- Define the specific conditions given in the problem
def arithmetic_sequence_condition (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ) : Prop :=
  a 3 + a 4 = 12 ∧ d = 2

-- Define the arithmetic sequence relation
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Statement to prove
theorem find_a9 
  (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ)
  (h1 : arithmetic_sequence_condition a d a3 a4)
  (h2 : arithmetic_sequence a d) :
  a 9 = 17 :=
sorry

end find_a9_l108_108583


namespace other_train_length_l108_108387

noncomputable def length_of_other_train
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ) : ℝ :=
  let v1 := (v1_kmph * 1000) / 3600
  let v2 := (v2_kmph * 1000) / 3600
  let relative_speed := v1 + v2
  let total_distance := relative_speed * t
  total_distance - l1

theorem other_train_length
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ)
  (hl1 : l1 = 230)
  (hv1 : v1_kmph = 120)
  (hv2 : v2_kmph = 80)
  (ht : t = 9) :
  length_of_other_train l1 v1_kmph v2_kmph t = 269.95 :=
by
  rw [hl1, hv1, hv2, ht]
  -- Proof steps skipped
  sorry

end other_train_length_l108_108387


namespace solve_k_values_l108_108257

def has_positive_integer_solution (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = k * a * b * c

def infinitely_many_solutions (k : ℕ) : Prop :=
  ∃ (a b c : ℕ → ℕ), (∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0 ∧ a n^2 + b n^2 + c n^2 = k * a n * b n * c n) ∧
  (∀ n, ∃ x y: ℤ, x^2 + y^2 = (a n * b n))

theorem solve_k_values :
  ∃ k : ℕ, (k = 1 ∨ k = 3) ∧ has_positive_integer_solution k ∧ infinitely_many_solutions k :=
sorry

end solve_k_values_l108_108257


namespace contractor_net_amount_l108_108048

-- Definitions based on conditions
def total_days : ℕ := 30
def pay_per_day : ℝ := 25
def fine_per_absence_day : ℝ := 7.5
def days_absent : ℕ := 6

-- Calculate days worked
def days_worked : ℕ := total_days - days_absent

-- Calculate total earnings
def earnings : ℝ := days_worked * pay_per_day

-- Calculate total fine
def fine : ℝ := days_absent * fine_per_absence_day

-- Calculate net amount received by the contractor
def net_amount : ℝ := earnings - fine

-- Problem statement: Prove that the net amount is Rs. 555
theorem contractor_net_amount : net_amount = 555 := by
  sorry

end contractor_net_amount_l108_108048


namespace triangle_angle_l108_108417

theorem triangle_angle (A B C : ℝ) (h1 : A - C = B) (h2 : A + B + C = 180) : A = 90 :=
by
  sorry

end triangle_angle_l108_108417


namespace simplify_expression_l108_108095

theorem simplify_expression :
  let a := 2
  let b := -3
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := sorry

end simplify_expression_l108_108095


namespace milk_per_cow_per_day_l108_108423

-- Define the conditions
def num_cows := 52
def weekly_milk_production := 364000 -- ounces

-- State the theorem
theorem milk_per_cow_per_day :
  (weekly_milk_production / 7 / num_cows) = 1000 := 
by
  -- Here we would include the proof, so we use sorry as placeholder
  sorry

end milk_per_cow_per_day_l108_108423


namespace number_of_players_in_tournament_l108_108720

theorem number_of_players_in_tournament (G : ℕ) (h1 : G = 42) (h2 : ∀ n : ℕ, G = n * (n - 1)) : ∃ n : ℕ, G = 42 ∧ n = 7 :=
by
  -- Let's suppose n is the number of players, then we need to prove
  -- ∃ n : ℕ, 42 = n * (n - 1) ∧ n = 7
  sorry

end number_of_players_in_tournament_l108_108720


namespace car_owners_without_motorcycles_l108_108649

theorem car_owners_without_motorcycles (total_adults cars motorcycles no_vehicle : ℕ) 
  (h1 : total_adults = 560) (h2 : cars = 520) (h3 : motorcycles = 80) (h4 : no_vehicle = 10) : 
  cars - (total_adults - no_vehicle - cars - motorcycles) = 470 := 
by
  sorry

end car_owners_without_motorcycles_l108_108649


namespace money_given_to_cashier_l108_108461

theorem money_given_to_cashier (regular_ticket_cost : ℕ) (discount : ℕ) 
  (age1 : ℕ) (age2 : ℕ) (change : ℕ) 
  (h1 : regular_ticket_cost = 109)
  (h2 : discount = 5)
  (h3 : age1 = 6)
  (h4 : age2 = 10)
  (h5 : change = 74)
  (h6 : age1 < 12)
  (h7 : age2 < 12) :
  regular_ticket_cost + regular_ticket_cost + (regular_ticket_cost - discount) + (regular_ticket_cost - discount) + change = 500 :=
by
  sorry

end money_given_to_cashier_l108_108461


namespace scientific_notation_example_l108_108171

theorem scientific_notation_example :
  ∃ (a : ℝ) (b : ℤ), 1300000 = a * 10 ^ b ∧ a = 1.3 ∧ b = 6 :=
sorry

end scientific_notation_example_l108_108171


namespace solution_pair_exists_l108_108533

theorem solution_pair_exists :
  ∃ (p q : ℚ), 
    ∀ (x : ℚ), 
      (p * x^4 + q * x^3 + 45 * x^2 - 25 * x + 10 = 
      (5 * x^2 - 3 * x + 2) * 
      ( (5 / 2) * x^2 - 5 * x + 5)) ∧ 
      (p = (25 / 2)) ∧ 
      (q = (-65 / 2)) :=
by
  sorry

end solution_pair_exists_l108_108533


namespace value_of_nabla_expression_l108_108180

namespace MathProblem

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem value_of_nabla_expression : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end MathProblem

end value_of_nabla_expression_l108_108180


namespace ratio_of_volumes_cone_cylinder_l108_108275

theorem ratio_of_volumes_cone_cylinder (r h_cylinder : ℝ) (h_cone : ℝ) (h_radius : r = 4) (h_height_cylinder : h_cylinder = 12) (h_height_cone : h_cone = h_cylinder / 2) :
  ((1/3) * (π * r^2 * h_cone)) / (π * r^2 * h_cylinder) = 1 / 6 :=
by
  -- Definitions and assumptions are directly included from the conditions.
  sorry

end ratio_of_volumes_cone_cylinder_l108_108275


namespace difference_of_lines_in_cm_l108_108039

def W : ℝ := 7.666666666666667
def B : ℝ := 3.3333333333333335
def inch_to_cm : ℝ := 2.54

theorem difference_of_lines_in_cm :
  (W * inch_to_cm) - (B * inch_to_cm) = 11.005555555555553 := 
sorry

end difference_of_lines_in_cm_l108_108039


namespace disk_diameter_solution_l108_108952

noncomputable def disk_diameter_condition : Prop :=
∃ x : ℝ, 
  (4 * Real.sqrt 3 + 2 * Real.pi) * x^2 - 12 * x + Real.sqrt 3 = 0 ∧
  x < Real.sqrt 3 / 6 ∧ 
  2 * x = 0.36

theorem disk_diameter_solution : exists (x : ℝ), 
  disk_diameter_condition := 
sorry

end disk_diameter_solution_l108_108952


namespace tom_seashells_l108_108477

theorem tom_seashells (fred_seashells : ℕ) (total_seashells : ℕ) (tom_seashells : ℕ)
  (h1 : fred_seashells = 43)
  (h2 : total_seashells = 58)
  (h3 : total_seashells = fred_seashells + tom_seashells) : tom_seashells = 15 :=
by
  sorry

end tom_seashells_l108_108477


namespace evaluate_expression_l108_108903

theorem evaluate_expression : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end evaluate_expression_l108_108903


namespace cost_per_load_is_25_cents_l108_108167

-- Define the given conditions
def loads_per_bottle : ℕ := 80
def usual_price_per_bottle : ℕ := 2500 -- in cents
def sale_price_per_bottle : ℕ := 2000 -- in cents
def bottles_bought : ℕ := 2

-- Defining the total cost and total loads
def total_cost : ℕ := bottles_bought * sale_price_per_bottle
def total_loads : ℕ := bottles_bought * loads_per_bottle

-- Define the cost per load in cents
def cost_per_load_in_cents : ℕ := (total_cost * 100) / total_loads

-- Formal proof statement
theorem cost_per_load_is_25_cents 
    (h1 : loads_per_bottle = 80)
    (h2 : usual_price_per_bottle = 2500)
    (h3 : sale_price_per_bottle = 2000)
    (h4 : bottles_bought = 2)
    (h5 : total_cost = bottles_bought * sale_price_per_bottle)
    (h6 : total_loads = bottles_bought * loads_per_bottle)
    (h7 : cost_per_load_in_cents = (total_cost * 100) / total_loads):
  cost_per_load_in_cents = 25 := by
  sorry

end cost_per_load_is_25_cents_l108_108167


namespace remainder_when_divided_l108_108225

theorem remainder_when_divided (P K Q R K' Q' S' T : ℕ)
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T) :
  P % (K * K') = K * S' + (T / Q') :=
by
  sorry

end remainder_when_divided_l108_108225


namespace kiran_money_l108_108010

theorem kiran_money (R G K : ℕ) (h1: R / G = 6 / 7) (h2: G / K = 6 / 15) (h3: R = 36) : K = 105 := by
  sorry

end kiran_money_l108_108010


namespace sum_of_powers_l108_108939

theorem sum_of_powers (x : ℝ) (h1 : x^10 - 3*x + 2 = 0) (h2 : x ≠ 1) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 :=
by
  sorry

end sum_of_powers_l108_108939


namespace max_U_value_l108_108364

noncomputable def maximum_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) : ℝ :=
  x + y

theorem max_U_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  maximum_value x y h ≤ Real.sqrt 13 :=
  sorry

end max_U_value_l108_108364


namespace percent_less_than_l108_108133

-- Definitions based on the given conditions.
variable (y q w z : ℝ)
variable (h1 : w = 0.60 * q)
variable (h2 : q = 0.60 * y)
variable (h3 : z = 1.50 * w)

-- The theorem that the percentage by which z is less than y is 46%.
theorem percent_less_than (y q w z : ℝ) (h1 : w = 0.60 * q) (h2 : q = 0.60 * y) (h3 : z = 1.50 * w) :
  100 - (z / y * 100) = 46 :=
sorry

end percent_less_than_l108_108133


namespace smallest_n_divides_l108_108271

theorem smallest_n_divides (m : ℕ) (h1 : m % 2 = 1) (h2 : m > 2) :
  ∃ n : ℕ, 2^(1988) = n ∧ 2^1989 ∣ m^n - 1 :=
by
  sorry

end smallest_n_divides_l108_108271


namespace arithmetic_sequence_term_number_l108_108660

theorem arithmetic_sequence_term_number
  (a : ℕ → ℤ)
  (ha1 : a 1 = 1)
  (ha2 : a 2 = 3)
  (n : ℕ)
  (hn : a n = 217) :
  n = 109 :=
sorry

end arithmetic_sequence_term_number_l108_108660


namespace max_value_q_l108_108930

namespace proof

theorem max_value_q (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end proof

end max_value_q_l108_108930


namespace total_snakes_count_l108_108901

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l108_108901


namespace g_of_3_l108_108166

theorem g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 4 * g x + 3 * g (1 / x) = 2 * x) :
  g 3 = 22 / 7 :=
sorry

end g_of_3_l108_108166


namespace p_q_r_cubic_sum_l108_108132

theorem p_q_r_cubic_sum (p q r : ℚ) (h1 : p + q + r = 4) (h2 : p * q + p * r + q * r = 6) (h3 : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
  sorry

end p_q_r_cubic_sum_l108_108132


namespace length_AM_is_correct_l108_108524

-- Definitions of the problem conditions
def length_of_square : ℝ := 9

def ratio_AP_PB : ℝ × ℝ := (7, 2)

def radius_of_quarter_circle : ℝ := 9

-- The theorem to prove
theorem length_AM_is_correct
  (AP PB PE : ℝ)
  (x : ℝ)
  (AM : ℝ) 
  (H_AP_PB  : AP = 7 ∧ PB = 2 ∧ PE = 2)
  (H_QD_QE : x = 63 / 11)
  (H_PQ : PQ = 2 + x) :
  AM = 85 / 22 :=
by
  sorry

end length_AM_is_correct_l108_108524


namespace number_of_plants_l108_108222

--- The given problem conditions and respective proof setup
axiom green_leaves_per_plant : ℕ
axiom yellow_turn_fall_off : ℕ
axiom green_leaves_total : ℕ

def one_third (n : ℕ) : ℕ := n / 3

-- Specify the given conditions
axiom leaves_per_plant_cond : green_leaves_per_plant = 18
axiom fall_off_cond : yellow_turn_fall_off = one_third green_leaves_per_plant
axiom total_leaves_cond : green_leaves_total = 36

-- Proof statement for the number of tea leaf plants
theorem number_of_plants : 
  (green_leaves_per_plant - yellow_turn_fall_off) * 3 = green_leaves_total :=
by
  sorry

end number_of_plants_l108_108222


namespace quadratic_equal_real_roots_l108_108492

theorem quadratic_equal_real_roots :
  ∃ k : ℝ, (∀ x : ℝ, x^2 - 4 * x + k = 0) ∧ k = 4 := by
  sorry

end quadratic_equal_real_roots_l108_108492


namespace factorization_a_minus_b_l108_108978

theorem factorization_a_minus_b (a b : ℤ) (h1 : 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : a - b = -7 :=
by
  sorry

end factorization_a_minus_b_l108_108978


namespace mean_inequalities_l108_108573

noncomputable def arith_mean (a : List ℝ) : ℝ := 
  (a.foldr (· + ·) 0) / a.length

noncomputable def geom_mean (a : List ℝ) : ℝ := 
  Real.exp ((a.foldr (λ x y => Real.log x + y) 0) / a.length)

noncomputable def harm_mean (a : List ℝ) : ℝ := 
  a.length / (a.foldr (λ x y => 1 / x + y) 0)

def is_positive (a : List ℝ) : Prop := 
  ∀ x ∈ a, x > 0

def bounds (a : List ℝ) (m g h : ℝ) : Prop := 
  let α := List.minimum a
  let β := List.maximum a
  α ≤ h ∧ h ≤ g ∧ g ≤ m ∧ m ≤ β

theorem mean_inequalities (a : List ℝ) (h g m : ℝ) (h_assoc: h = harm_mean a) (g_assoc: g = geom_mean a) (m_assoc: m = arith_mean a) :
  is_positive a → bounds a m g h :=
  
sorry

end mean_inequalities_l108_108573


namespace gloves_selection_l108_108869

theorem gloves_selection (total_pairs : ℕ) (total_gloves : ℕ) (num_to_select : ℕ) 
    (total_ways : ℕ) (no_pair_ways : ℕ) : 
    total_pairs = 4 → 
    total_gloves = 8 → 
    num_to_select = 4 → 
    total_ways = (Nat.choose total_gloves num_to_select) → 
    no_pair_ways = 2^total_pairs → 
    (total_ways - no_pair_ways) = 54 :=
by
  intros
  sorry

end gloves_selection_l108_108869


namespace sum_of_squares_l108_108274

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : a * b + b * c + c * a = 131) : 
  a^2 + b^2 + c^2 = 138 := 
sorry

end sum_of_squares_l108_108274


namespace john_hiking_probability_l108_108065

theorem john_hiking_probability :
  let P_rain := 0.3
  let P_sunny := 0.7
  let P_hiking_if_rain := 0.1
  let P_hiking_if_sunny := 0.9

  let P_hiking := P_rain * P_hiking_if_rain + P_sunny * P_hiking_if_sunny

  P_hiking = 0.66 := by
    sorry

end john_hiking_probability_l108_108065


namespace integer_squares_l108_108212

theorem integer_squares (x y : ℤ) 
  (hx : ∃ a : ℤ, x + y = a^2)
  (h2x3y : ∃ b : ℤ, 2 * x + 3 * y = b^2)
  (h3xy : ∃ c : ℤ, 3 * x + y = c^2) : 
  x = 0 ∧ y = 0 := 
by { sorry }

end integer_squares_l108_108212


namespace calculate_sum_l108_108518

theorem calculate_sum : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 :=
by
  sorry

end calculate_sum_l108_108518


namespace basketball_game_l108_108821

variable (H E : ℕ)

theorem basketball_game (h_eq_sum : H + E = 50) (h_margin : H = E + 6) : E = 22 := by
  sorry

end basketball_game_l108_108821


namespace perfect_square_m_value_l108_108485

theorem perfect_square_m_value (M X : ℤ) (hM : M > 1) (hX_lt_max : X < 8000) (hX_gt_min : 1000 < X) (hX_eq : X = M^3) : 
  (∃ M : ℤ, M > 1 ∧ 1000 < M^3 ∧ M^3 < 8000 ∧ (∃ k : ℤ, X = k * k) ∧ M = 16) :=
by
  use 16
  -- Here, we would normally provide the proof steps to show that 1000 < 16^3 < 8000 and 16^3 is a perfect square
  sorry

end perfect_square_m_value_l108_108485


namespace defense_attorney_mistake_l108_108187

variable (P Q : Prop)

theorem defense_attorney_mistake (h1 : P → Q) (h2 : ¬ (P → Q)) : P ∧ ¬ Q :=
by {
  sorry
}

end defense_attorney_mistake_l108_108187


namespace find_incorrect_statement_l108_108682

theorem find_incorrect_statement :
  ¬ (∀ a b c : ℝ, c ≠ 0 → (a < b → a * c^2 < b * c^2)) :=
by
  sorry

end find_incorrect_statement_l108_108682


namespace fault_line_total_movement_l108_108189

theorem fault_line_total_movement (a b : ℝ) (h1 : a = 1.25) (h2 : b = 5.25) : a + b = 6.50 := by
  -- Definitions:
  rw [h1, h2]
  -- Proof:
  sorry

end fault_line_total_movement_l108_108189


namespace ball_count_l108_108624

theorem ball_count (r b y : ℕ) 
  (h1 : b + y = 9) 
  (h2 : r + y = 5) 
  (h3 : r + b = 6) : 
  r + b + y = 10 := 
  sorry

end ball_count_l108_108624


namespace ratio_of_refurb_to_new_tshirt_l108_108861

def cost_of_new_tshirt : ℤ := 5
def cost_of_pants : ℤ := 4
def cost_of_skirt : ℤ := 6

-- Total income from selling two new T-shirts, one pair of pants, four skirts, and six refurbished T-shirts is $53.
def total_income : ℤ := 53

-- Total income from selling new items.
def income_from_new_items : ℤ :=
  2 * cost_of_new_tshirt + cost_of_pants + 4 * cost_of_skirt

-- Income from refurbished T-shirts.
def income_from_refurb_tshirts : ℤ :=
  total_income - income_from_new_items

-- Number of refurbished T-shirts sold.
def num_refurb_tshirts_sold : ℤ := 6

-- Price of one refurbished T-shirt.
def cost_of_refurb_tshirt : ℤ :=
  income_from_refurb_tshirts / num_refurb_tshirts_sold

-- Prove the ratio of the price of a refurbished T-shirt to a new T-shirt is 0.5
theorem ratio_of_refurb_to_new_tshirt :
  (cost_of_refurb_tshirt : ℚ) / cost_of_new_tshirt = 0.5 := 
sorry

end ratio_of_refurb_to_new_tshirt_l108_108861


namespace solitaire_game_removal_l108_108849

theorem solitaire_game_removal (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∃ moves : ℕ, ∀ i : ℕ, i < moves → (i + 1) % 2 = (i % 2) + 1) ↔ (m % 2 = 1 ∨ n % 2 = 1) :=
sorry

end solitaire_game_removal_l108_108849


namespace sides_of_regular_polygon_l108_108231

theorem sides_of_regular_polygon 
    (sum_interior_angles : ∀ n : ℕ, (n - 2) * 180 = 1440) :
  ∃ n : ℕ, n = 10 :=
by
  sorry

end sides_of_regular_polygon_l108_108231


namespace find_length_of_side_c_find_measure_of_angle_B_l108_108418

variable {A B C a b c : ℝ}

def triangle_problem (a b c A B C : ℝ) :=
  a * Real.cos B = 3 ∧
  b * Real.cos A = 1 ∧
  A - B = Real.pi / 6 ∧
  a^2 + c^2 - b^2 - 6 * c = 0 ∧
  b^2 + c^2 - a^2 - 2 * c = 0

theorem find_length_of_side_c (h : triangle_problem a b c A B C) :
  c = 4 :=
sorry

theorem find_measure_of_angle_B (h : triangle_problem a b c A B C) :
  B = Real.pi / 6 :=
sorry

end find_length_of_side_c_find_measure_of_angle_B_l108_108418


namespace investment_of_D_l108_108259

/--
Given C and D started a business where C invested Rs. 1000 and D invested some amount.
They made a total profit of Rs. 500, and D's share of the profit is Rs. 100.
So, how much did D invest in the business?
-/
theorem investment_of_D 
  (C_invested : ℕ) (D_share : ℕ) (total_profit : ℕ) 
  (H1 : C_invested = 1000) 
  (H2 : D_share = 100) 
  (H3 : total_profit = 500) 
  : ∃ D : ℕ, D = 250 :=
by
  sorry

end investment_of_D_l108_108259


namespace find_number_l108_108914

theorem find_number (N : ℕ) : 
  (N % 13 = 11) ∧ (N % 17 = 9) ↔ N = 141 :=
by 
  sorry

end find_number_l108_108914


namespace simple_interest_rate_l108_108587

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) 
  (hT : T = 10) (hSI : (P * R * T) / 100 = (1 / 5) * P) : R = 2 :=
by
  sorry

end simple_interest_rate_l108_108587


namespace find_r_l108_108483

noncomputable def r_value (a b : ℝ) (h : a * b = 3) : ℝ :=
  let r := (a^2 + 1 / b^2) * (b^2 + 1 / a^2)
  r

theorem find_r (a b : ℝ) (h : a * b = 3) : r_value a b h = 100 / 9 := by
  sorry

end find_r_l108_108483


namespace range_of_x_in_sqrt_x_plus_3_l108_108298

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l108_108298


namespace total_dollars_l108_108214

theorem total_dollars (mark_dollars : ℚ) (carolyn_dollars : ℚ) (mark_money : mark_dollars = 7 / 8) (carolyn_money : carolyn_dollars = 2 / 5) :
  mark_dollars + carolyn_dollars = 1.275 := sorry

end total_dollars_l108_108214


namespace find_k_l108_108371

theorem find_k {k : ℝ} (h : (∃ α β : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α / β = 3 / 1 ∧ α + β = -10 ∧ α * β = k)) : k = 18.75 :=
sorry

end find_k_l108_108371


namespace visits_per_hour_l108_108247

open Real

theorem visits_per_hour (price_per_visit : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) (total_earnings : ℝ) 
  (h_price : price_per_visit = 0.10)
  (h_hours : hours_per_day = 24)
  (h_days : days_per_month = 30)
  (h_earnings : total_earnings = 3600) :
  (total_earnings / (price_per_visit * hours_per_day * days_per_month) : ℝ) = 50 :=
by
  sorry

end visits_per_hour_l108_108247


namespace part1_part2_l108_108316

-- Part 1: Number of k-tuples of ordered subsets with empty intersection
theorem part1 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (∃ (f : Fin (n) → Fin (2^k - 1)), true) :=
sorry

-- Part 2: Number of k-tuples of subsets with chain condition
theorem part2 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (S.card = (k + 1)^n) :=
sorry

end part1_part2_l108_108316


namespace Olly_needs_24_shoes_l108_108108

def dogs := 3
def cats := 2
def ferrets := 1
def paws_per_dog := 4
def paws_per_cat := 4
def paws_per_ferret := 4

theorem Olly_needs_24_shoes : (dogs * paws_per_dog) + (cats * paws_per_cat) + (ferrets * paws_per_ferret) = 24 :=
by
  sorry

end Olly_needs_24_shoes_l108_108108


namespace problem_31_36_l108_108659

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem problem_31_36 (p k : ℕ) (hp : is_prime (4 * k + 1)) :
  (∃ x y m : ℕ, x^2 + y^2 = m * p) ∧ (∀ m > 1, ∃ x y m1 : ℕ, x^2 + y^2 = m * p ∧ 0 < m1 ∧ m1 < m) :=
by sorry

end problem_31_36_l108_108659


namespace cleaning_time_together_l108_108390

theorem cleaning_time_together (t : ℝ) (h_t : 3 = t / 3) (h_john_time : 6 = 6) : 
  (5 / (1 / 6 + 1 / 9)) = 3.6 :=
by
  sorry

end cleaning_time_together_l108_108390


namespace find_a_if_odd_l108_108246

theorem find_a_if_odd :
  ∀ (a : ℝ), (∀ x : ℝ, (a * (-x)^3 + (a - 1) * (-x)^2 + (-x) = -(a * x^3 + (a - 1) * x^2 + x))) → 
  a = 1 :=
by
  sorry

end find_a_if_odd_l108_108246


namespace roots_product_eq_348_l108_108076

theorem roots_product_eq_348 (d e : ℤ) 
  (h : ∀ (s : ℂ), s^2 - 2*s - 1 = 0 → s^5 - d*s - e = 0) : 
  d * e = 348 :=
sorry

end roots_product_eq_348_l108_108076


namespace ratio_PM_MQ_eq_1_l108_108399

theorem ratio_PM_MQ_eq_1
  (A B C D E M P Q : ℝ × ℝ)
  (square_side : ℝ)
  (h_square_side : square_side = 15)
  (hA : A = (0, square_side))
  (hB : B = (square_side, square_side))
  (hC : C = (square_side, 0))
  (hD : D = (0, 0))
  (hE : E = (8, 0))
  (hM : M = ((A.1 + E.1) / 2, (A.2 + E.2) / 2))
  (h_slope_AE : E.2 - A.2 = (E.1 - A.1) * -15 / 8)
  (h_P_on_AD : P.2 = 15)
  (h_Q_on_BC : Q.2 = 0)
  (h_PM_len : dist M P = dist M Q) :
  dist P M = dist M Q :=
by sorry

end ratio_PM_MQ_eq_1_l108_108399


namespace not_right_angled_triangle_l108_108454

theorem not_right_angled_triangle 
  (m n : ℝ) 
  (h1 : m > n) 
  (h2 : n > 0)
  : ¬ (m^2 + n^2)^2 = (mn)^2 + (m^2 - n^2)^2 :=
sorry

end not_right_angled_triangle_l108_108454


namespace sufficient_condition_for_parallel_l108_108127

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Definitions of parallelism and perpendicularity
variable {Parallel Perpendicular : Line → Plane → Prop}
variable {ParallelLines : Line → Line → Prop}

-- Definition of subset relation
variable {Subset : Line → Plane → Prop}

-- Theorems or conditions
variables (a b : Line) (α β : Plane)

-- Assertion of the theorem
theorem sufficient_condition_for_parallel (h1 : ParallelLines a b) (h2 : Parallel b α) (h3 : ¬ Subset a α) : Parallel a α :=
sorry

end sufficient_condition_for_parallel_l108_108127


namespace distinct_solutions_eq_108_l108_108495

theorem distinct_solutions_eq_108 {p q : ℝ} (h1 : (p - 6) * (3 * p + 10) = p^2 - 19 * p + 50)
  (h2 : (q - 6) * (3 * q + 10) = q^2 - 19 * q + 50)
  (h3 : p ≠ q) : (p + 2) * (q + 2) = 108 := 
by
  sorry

end distinct_solutions_eq_108_l108_108495


namespace probability_is_correct_l108_108603

def num_red : ℕ := 7
def num_green : ℕ := 9
def num_yellow : ℕ := 10
def num_blue : ℕ := 5
def num_purple : ℕ := 3

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue + num_purple

def num_blue_or_purple : ℕ := num_blue + num_purple

-- Probability of selecting a blue or purple jelly bean
def probability_blue_or_purple : ℚ := num_blue_or_purple / total_jelly_beans

theorem probability_is_correct :
  probability_blue_or_purple = 4 / 17 := sorry

end probability_is_correct_l108_108603


namespace smallest_solution_l108_108391

theorem smallest_solution (x : ℝ) (h : x^2 + 10 * x - 24 = 0) : x = -12 :=
sorry

end smallest_solution_l108_108391


namespace largest_a1_l108_108962

theorem largest_a1
  (a : ℕ+ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_eq : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h_initial : a 1 = a 10) :
  ∃ (max_a1 : ℝ), max_a1 = 16 ∧ ∀ x, x = a 1 → x ≤ 16 :=
by
  sorry

end largest_a1_l108_108962


namespace arithmetic_sequence_x_value_l108_108686

theorem arithmetic_sequence_x_value
  (x : ℝ)
  (h₁ : 2 * x - (1 / 3) = (x + 4) - 2 * x) :
  x = 13 / 3 := by
  sorry

end arithmetic_sequence_x_value_l108_108686


namespace exponent_multiplication_l108_108889

theorem exponent_multiplication :
  (10^(3/4)) * (10^(-0.25)) * (10^(1.5)) = 10^2 :=
by sorry

end exponent_multiplication_l108_108889


namespace smaller_angle_at_8_15_pm_l108_108217

noncomputable def smaller_angle_between_clock_hands (minute_hand_degrees_per_min: ℝ) (hour_hand_degrees_per_min: ℝ) (time_in_minutes: ℝ) : ℝ := sorry

theorem smaller_angle_at_8_15_pm :
  smaller_angle_between_clock_hands 6 0.5 495 = 157.5 :=
sorry

end smaller_angle_at_8_15_pm_l108_108217


namespace exists_positive_x_for_inequality_l108_108755

-- Define the problem conditions and the final proof goal.
theorem exists_positive_x_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Ico (-9/4 : ℝ) (2 : ℝ) :=
by
  sorry

end exists_positive_x_for_inequality_l108_108755


namespace find_number_of_cats_l108_108921

theorem find_number_of_cats (dogs ferrets cats total_shoes shoes_per_animal : ℕ) 
  (h_dogs : dogs = 3)
  (h_ferrets : ferrets = 1)
  (h_total_shoes : total_shoes = 24)
  (h_shoes_per_animal : shoes_per_animal = 4) :
  cats = (total_shoes - (dogs + ferrets) * shoes_per_animal) / shoes_per_animal := by
  sorry

end find_number_of_cats_l108_108921


namespace sum_of_reciprocals_l108_108947

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) : 
  1/x + 1/y = 14/45 := 
sorry

end sum_of_reciprocals_l108_108947


namespace arithmetic_sequence_k_is_10_l108_108675

noncomputable def a_n (n : ℕ) (d : ℝ) : ℝ := (n - 1) * d

theorem arithmetic_sequence_k_is_10 (d : ℝ) (h : d ≠ 0) : 
  (∃ k : ℕ, a_n k d = (a_n 1 d) + (a_n 2 d) + (a_n 3 d) + (a_n 4 d) + (a_n 5 d) + (a_n 6 d) + (a_n 7 d) ∧ k = 10) := 
by
  sorry

end arithmetic_sequence_k_is_10_l108_108675


namespace solve_rings_l108_108378

variable (B : ℝ) (S : ℝ)

def conditions := (S = (5/8) * (Real.sqrt B)) ∧ (S + B = 52)

theorem solve_rings : conditions B S → (S + B = 52) := by
  intros h
  sorry

end solve_rings_l108_108378


namespace composite_number_N_l108_108253

theorem composite_number_N (y : ℕ) (hy : y > 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (y ^ 125 - 1) / (3 ^ 22 - 1) :=
by
  -- use sorry to skip the proof
  sorry

end composite_number_N_l108_108253


namespace number_of_ordered_triples_l108_108413

theorem number_of_ordered_triples :
  let b := 2023
  let n := (b ^ 2)
  ∀ (a c : ℕ), a * c = n ∧ a ≤ b ∧ b ≤ c → (∃ (k : ℕ), k = 7) :=
by
  sorry

end number_of_ordered_triples_l108_108413


namespace cricketer_stats_l108_108467

theorem cricketer_stats :
  let total_runs := 225
  let total_balls := 120
  let boundaries := 4 * 15
  let sixes := 6 * 8
  let twos := 2 * 3
  let singles := 1 * 10
  let perc_boundaries := (boundaries / total_runs.toFloat) * 100
  let perc_sixes := (sixes / total_runs.toFloat) * 100
  let perc_twos := (twos / total_runs.toFloat) * 100
  let perc_singles := (singles / total_runs.toFloat) * 100
  let strike_rate := (total_runs.toFloat / total_balls.toFloat) * 100
  perc_boundaries = 26.67 ∧
  perc_sixes = 21.33 ∧
  perc_twos = 2.67 ∧
  perc_singles = 4.44 ∧
  strike_rate = 187.5 :=
by
  sorry

end cricketer_stats_l108_108467


namespace seq_is_arithmetic_l108_108531

-- Define the sequence sum S_n and the sequence a_n
noncomputable def S (a : ℕ) (n : ℕ) : ℕ := a * n^2 + n
noncomputable def a_n (a : ℕ) (n : ℕ) : ℕ := S a n - S a (n - 1)

-- Define the property of being an arithmetic sequence
def is_arithmetic_seq (a_n : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → (a_n (n + 1) : ℤ) - (a_n n : ℤ) = d

-- The theorem to be proven
theorem seq_is_arithmetic (a : ℕ) (h : 0 < a) : is_arithmetic_seq (a_n a) :=
by
  sorry

end seq_is_arithmetic_l108_108531


namespace find_third_number_l108_108976

theorem find_third_number (N : ℤ) :
  (1274 % 12 = 2) ∧ (1275 % 12 = 3) ∧ (1285 % 12 = 1) ∧ ((1274 * 1275 * N * 1285) % 12 = 6) →
  N % 12 = 1 :=
by
  sorry

end find_third_number_l108_108976


namespace rita_book_pages_l108_108785

theorem rita_book_pages (x : ℕ) (h1 : ∃ n₁, n₁ = (1/6 : ℚ) * x + 10) 
                                  (h2 : ∃ n₂, n₂ = (1/5 : ℚ) * ((5/6 : ℚ) * x - 10) + 20)
                                  (h3 : ∃ n₃, n₃ = (1/4 : ℚ) * ((4/5 : ℚ) * ((5/6 : ℚ) * x - 10) - 20) + 25)
                                  (h4 : ((3/4 : ℚ) * ((2/3 : ℚ) * x - 28) - 25) = 50) :
    x = 192 := 
sorry

end rita_book_pages_l108_108785


namespace compute_expression_l108_108698

theorem compute_expression :
  (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end compute_expression_l108_108698


namespace fish_offspring_base10_l108_108818

def convert_base_7_to_10 (n : ℕ) : ℕ :=
  let d2 := n / 49
  let r2 := n % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d2 * 49 + d1 * 7 + d0

theorem fish_offspring_base10 :
  convert_base_7_to_10 265 = 145 :=
by
  sorry

end fish_offspring_base10_l108_108818


namespace inequality_proof_l108_108521

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_condition : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/(a^2) + 1/(b^2) + 1/(c^2) + 1/(d^2)) ≥ 36 :=
by
  sorry

end inequality_proof_l108_108521


namespace luke_piles_of_quarters_l108_108622

theorem luke_piles_of_quarters (Q D : ℕ) 
  (h1 : Q = D) -- number of piles of quarters equals number of piles of dimes
  (h2 : 3 * Q + 3 * D = 30) -- total number of coins is 30
  : Q = 5 :=
by
  sorry

end luke_piles_of_quarters_l108_108622


namespace number_of_geese_is_correct_l108_108692

noncomputable def number_of_ducks := 37
noncomputable def total_number_of_birds := 95
noncomputable def number_of_geese := total_number_of_birds - number_of_ducks

theorem number_of_geese_is_correct : number_of_geese = 58 := by
  sorry

end number_of_geese_is_correct_l108_108692


namespace pq_solution_l108_108556

theorem pq_solution :
  ∃ (p q : ℤ), (20 * x ^ 2 - 110 * x - 120 = (5 * x + p) * (4 * x + q))
    ∧ (5 * q + 4 * p = -110) ∧ (p * q = -120)
    ∧ (p + 2 * q = -8) :=
by
  sorry

end pq_solution_l108_108556


namespace associate_professors_bring_2_pencils_l108_108842

theorem associate_professors_bring_2_pencils (A B P : ℕ) 
  (h1 : A + B = 5)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 5)
  : P = 2 :=
by {
  -- Proof goes here
  sorry
}

end associate_professors_bring_2_pencils_l108_108842


namespace lcm_24_90_l108_108632

theorem lcm_24_90 : lcm 24 90 = 360 :=
by 
-- lcm is the least common multiple of 24 and 90.
-- lcm 24 90 is defined as 360.
sorry

end lcm_24_90_l108_108632


namespace jenny_profit_l108_108831

-- Define the constants given in the problem
def cost_per_pan : ℝ := 10.00
def price_per_pan : ℝ := 25.00
def num_pans : ℝ := 20.0

-- Define the total revenue function
def total_revenue (num_pans : ℝ) (price_per_pan : ℝ) : ℝ := num_pans * price_per_pan

-- Define the total cost function
def total_cost (num_pans : ℝ) (cost_per_pan : ℝ) : ℝ := num_pans * cost_per_pan

-- Define the profit function as the total revenue minus the total cost
def total_profit (num_pans : ℝ) (price_per_pan : ℝ) (cost_per_pan : ℝ) : ℝ := 
  total_revenue num_pans price_per_pan - total_cost num_pans cost_per_pan

-- The statement to prove in Lean
theorem jenny_profit : total_profit num_pans price_per_pan cost_per_pan = 300.00 := 
by 
  sorry

end jenny_profit_l108_108831


namespace length_of_AB_l108_108311

open Real

noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (sqrt 3 / 2) * t)
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, (∃ t : ℝ, line t = A) ∧ (∃ θ : ℝ, curve θ = A) ∧
                 (∃ t : ℝ, line t = B) ∧ (∃ θ : ℝ, curve θ = B) ∧
                 dist A B = 1 :=
by
  sorry

end length_of_AB_l108_108311


namespace change_in_profit_rate_l108_108836

theorem change_in_profit_rate (A B C : Type) (P : ℝ) (r1 r2 : ℝ) (income_increase : ℝ) (capital : ℝ) :
  (A_receives : ℝ) = (2 / 3) → 
  (B_C_divide : ℝ) = (1 - (2 / 3)) / 2 → 
  income_increase = 300 → 
  capital = 15000 →
  ((2 / 3) * capital * (r2 / 100) - (2 / 3) * capital * (r1 / 100)) = income_increase →
  (r2 - r1) = 3 :=
by
  intros
  sorry

end change_in_profit_rate_l108_108836


namespace length_of_plot_correct_l108_108918

noncomputable def length_of_plot (b : ℕ) : ℕ := b + 30

theorem length_of_plot_correct (b : ℕ) (cost_per_meter total_cost : ℝ) 
    (h1 : length_of_plot b = b + 30)
    (h2 : cost_per_meter = 26.50)
    (h3 : total_cost = 5300)
    (h4 : 2 * (b + (b + 30)) * cost_per_meter = total_cost) :
    length_of_plot 35 = 65 :=
by
  sorry

end length_of_plot_correct_l108_108918


namespace range_of_a_l108_108621

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + 2 * x - 3

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → f a x ≤ f a y) ↔ (- (1/4:ℝ) ≤ a ∧ a ≤ 0) := by
  sorry

end range_of_a_l108_108621


namespace amount_distribution_l108_108741

theorem amount_distribution :
  ∃ (P Q R S T : ℝ), 
    (P + Q + R + S + T = 24000) ∧ 
    (R = (3 / 5) * (P + Q)) ∧ 
    (S = 0.45 * 24000) ∧ 
    (T = (1 / 2) * R) ∧ 
    (P + Q = 7000) ∧ 
    (R = 4200) ∧ 
    (S = 10800) ∧ 
    (T = 2100) :=
by
  sorry

end amount_distribution_l108_108741


namespace perimeter_of_larger_triangle_is_65_l108_108027

noncomputable def similar_triangle_perimeter : ℝ :=
  let a := 7
  let b := 7
  let c := 12
  let longest_side_similar := 30
  let perimeter_small := a + b + c
  let ratio := longest_side_similar / c
  ratio * perimeter_small

theorem perimeter_of_larger_triangle_is_65 :
  similar_triangle_perimeter = 65 := by
  sorry

end perimeter_of_larger_triangle_is_65_l108_108027


namespace laptop_price_difference_l108_108957

theorem laptop_price_difference :
  let list_price := 59.99
  let tech_bargains_discount := 15
  let budget_bytes_discount_percentage := 0.30
  let tech_bargains_price := list_price - tech_bargains_discount
  let budget_bytes_price := list_price * (1 - budget_bytes_discount_percentage)
  let cheaper_price := min tech_bargains_price budget_bytes_price
  let expensive_price := max tech_bargains_price budget_bytes_price
  (expensive_price - cheaper_price) * 100 = 300 :=
by
  sorry

end laptop_price_difference_l108_108957


namespace divisible_by_1995_l108_108525

theorem divisible_by_1995 (n : ℕ) : 
  1995 ∣ (256^(2*n) * 7^(2*n) - 168^(2*n) - 32^(2*n) + 3^(2*n)) := 
sorry

end divisible_by_1995_l108_108525


namespace combined_mixture_nuts_l108_108501

def sue_percentage_nuts : ℝ := 0.30
def sue_percentage_dried_fruit : ℝ := 0.70

def jane_percentage_nuts : ℝ := 0.60
def combined_percentage_dried_fruit : ℝ := 0.35

theorem combined_mixture_nuts :
  let sue_contribution := 100.0
  let jane_contribution := 100.0
  let sue_nuts := sue_contribution * sue_percentage_nuts
  let jane_nuts := jane_contribution * jane_percentage_nuts
  let combined_nuts := sue_nuts + jane_nuts
  let total_weight := sue_contribution + jane_contribution
  (combined_nuts / total_weight) * 100 = 45 :=
by
  sorry

end combined_mixture_nuts_l108_108501


namespace gcd_of_polynomial_and_multiple_l108_108942

-- Definitions based on given conditions
def multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- The main statement of the problem
theorem gcd_of_polynomial_and_multiple (y : ℕ) (h : multiple_of y 56790) :
  Nat.gcd ((3 * y + 2) * (5 * y + 3) * (11 * y + 7) * (y + 17)) y = 714 :=
sorry

end gcd_of_polynomial_and_multiple_l108_108942


namespace odd_function_properties_l108_108859

theorem odd_function_properties
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 1 ≤ x ∧ x ≤ 3 ∧ 1 ≤ y ∧ y ≤ 3 ∧ x < y → f x < f y)
  (h_min_val : ∀ x, 1 ≤ x ∧ x ≤ 3 → 7 ≤ f x) :
  (∀ x y, -3 ≤ x ∧ x ≤ -1 ∧ -3 ≤ y ∧ y ≤ -1 ∧ x < y → f x < f y) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) :=
sorry

end odd_function_properties_l108_108859


namespace cost_price_of_computer_table_l108_108285

theorem cost_price_of_computer_table (SP : ℝ) (CP : ℝ) (h : SP = CP * 1.24) (h_SP : SP = 8215) : CP = 6625 :=
by
  -- Start the proof block
  sorry -- Proof is not required as per the instructions

end cost_price_of_computer_table_l108_108285


namespace number_of_pints_of_paint_l108_108331

-- Statement of the problem
theorem number_of_pints_of_paint (A B : ℝ) (N : ℕ) 
  (large_cube_paint : ℝ) (hA : A = 4) (hB : B = 2) (hN : N = 125) 
  (large_cube_paint_condition : large_cube_paint = 1) : 
  (N * (B / A) ^ 2 * large_cube_paint = 31.25) :=
by {
  -- Given the conditions
  sorry
}

end number_of_pints_of_paint_l108_108331


namespace bobs_improvement_percentage_l108_108648

-- Define the conditions
def bobs_time_minutes := 10
def bobs_time_seconds := 40
def sisters_time_minutes := 10
def sisters_time_seconds := 8

-- Convert minutes and seconds to total seconds
def bobs_total_time_seconds := bobs_time_minutes * 60 + bobs_time_seconds
def sisters_total_time_seconds := sisters_time_minutes * 60 + sisters_time_seconds

-- Define the improvement needed and calculate the percentage improvement
def improvement_needed := bobs_total_time_seconds - sisters_total_time_seconds
def percentage_improvement := (improvement_needed / bobs_total_time_seconds) * 100

-- The lean statement to prove
theorem bobs_improvement_percentage : percentage_improvement = 5 := by
  sorry

end bobs_improvement_percentage_l108_108648


namespace pow_1986_mod_7_l108_108943

theorem pow_1986_mod_7 : (5 ^ 1986) % 7 = 1 := by
  sorry

end pow_1986_mod_7_l108_108943


namespace aluminum_atomic_weight_l108_108827

theorem aluminum_atomic_weight (Al_w : ℤ) 
  (compound_molecular_weight : ℤ) 
  (num_fluorine_atoms : ℕ) 
  (fluorine_atomic_weight : ℤ) 
  (h1 : compound_molecular_weight = 84) 
  (h2 : num_fluorine_atoms = 3) 
  (h3 : fluorine_atomic_weight = 19) :
  Al_w = 27 := 
by
  -- Proof goes here, but it is skipped.
  sorry

end aluminum_atomic_weight_l108_108827


namespace percentage_increase_in_area_l108_108724

-- Defining the lengths and widths in terms of real numbers
variables (L W : ℝ)

-- Defining the new lengths and widths
def new_length := 1.2 * L
def new_width := 1.2 * W

-- Original area of the rectangle
def original_area := L * W

-- New area of the rectangle
def new_area := new_length L * new_width W

-- Proof statement for the percentage increase
theorem percentage_increase_in_area : 
  ((new_area L W - original_area L W) / original_area L W) * 100 = 44 :=
by
  sorry

end percentage_increase_in_area_l108_108724


namespace general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l108_108742

noncomputable def a_seq (n : ℕ) : ℕ :=
  if h : n > 0 then n else 1

noncomputable def b_seq (n : ℕ) : ℚ :=
  if h : n > 0 then n * (n - 1) / 4 else 0

noncomputable def c_seq (n : ℕ) : ℚ :=
  a_seq n ^ 2 - 4 * b_seq n

theorem general_formula_for_sequences (n : ℕ) (h : n > 0) :
  a_seq n = n ∧ b_seq n = (n * (n - 1)) / 4 :=
sorry

theorem c_seq_is_arithmetic (n : ℕ) (h : n > 0) : 
  ∀ m : ℕ, (h2 : m > 0) -> c_seq (m+1) - c_seq m = 1 :=
sorry

theorem fn_integer_roots (n : ℕ) : 
  ∃ k : ℤ, n = k ^ 2 ∧ k ≠ 0 :=
sorry

end general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l108_108742


namespace isosceles_triangle_base_length_l108_108825

theorem isosceles_triangle_base_length
  (a b : ℕ)
  (ha : a = 8)
  (hp : 2 * a + b = 25)
  : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l108_108825


namespace no_solution_inequality_l108_108627

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_inequality_l108_108627


namespace runner_distance_l108_108295

theorem runner_distance (track_length race_length : ℕ) (A_speed B_speed C_speed : ℚ)
  (h1 : track_length = 400) (h2 : race_length = 800)
  (h3 : A_speed = 1) (h4 : B_speed = 8 / 7) (h5 : C_speed = 6 / 7) :
  ∃ distance_from_finish : ℚ, distance_from_finish = 200 :=
by {
  -- We are not required to provide the actual proof steps, just setting up the definitions and initial statements for the proof.
  sorry
}

end runner_distance_l108_108295


namespace sum_of_polynomial_roots_l108_108270

theorem sum_of_polynomial_roots:
  ∀ (a b : ℝ),
  (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) →
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b^3 + b * a^3 = 683 := by
  intros a b h
  sorry

end sum_of_polynomial_roots_l108_108270


namespace smallest_b_greater_than_l108_108838

theorem smallest_b_greater_than (a b : ℤ) (h₁ : 9 < a) (h₂ : a < 21) (h₃ : 10 / b ≥ 2 / 3) (h₄ : b < 31) : 14 < b :=
sorry

end smallest_b_greater_than_l108_108838


namespace binom_inequality_l108_108778

-- Defining the conditions as non-computable functions
def is_nonneg_integer := ℕ

-- Defining the binomial coefficient function
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The statement of the theorem
theorem binom_inequality (n k h : ℕ) (hn : n ≥ k + h) : binom n (k + h) ≥ binom (n - k) h :=
  sorry

end binom_inequality_l108_108778


namespace probability_draw_l108_108386

theorem probability_draw (pA_win pA_not_lose : ℝ) (h1 : pA_win = 0.3) (h2 : pA_not_lose = 0.8) :
  pA_not_lose - pA_win = 0.5 :=
by 
  sorry

end probability_draw_l108_108386


namespace third_number_pascals_triangle_61_numbers_l108_108376

theorem third_number_pascals_triangle_61_numbers : (Nat.choose 60 2) = 1770 := by
  sorry

end third_number_pascals_triangle_61_numbers_l108_108376


namespace negation_of_universal_proposition_l108_108878

noncomputable def f (n : Nat) : Set ℕ := sorry

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, f n ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n, m ≤ n) ↔
  ∃ n_0 : ℕ, f n_0 ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n_0, m ≤ n_0 :=
sorry

end negation_of_universal_proposition_l108_108878


namespace find_xy_l108_108012

theorem find_xy (x y : ℝ) :
  x^2 + y^2 = 2 ∧ (x^2 / (2 - y) + y^2 / (2 - x) = 2) → (x = 1 ∧ y = 1) :=
by
  sorry

end find_xy_l108_108012


namespace computation_of_sqrt_expr_l108_108620

theorem computation_of_sqrt_expr : 
  (Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549) := 
by
  sorry

end computation_of_sqrt_expr_l108_108620


namespace floor_sum_value_l108_108776

theorem floor_sum_value (a b c d : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
(h1 : a^2 + b^2 = 2016) (h2 : c^2 + d^2 = 2016) (h3 : a * c = 1024) (h4 : b * d = 1024) :
  ⌊a + b + c + d⌋ = 127 := sorry

end floor_sum_value_l108_108776


namespace hamburger_combinations_l108_108815

theorem hamburger_combinations : 
  let condiments := 10  -- Number of available condiments
  let patty_choices := 4 -- Number of meat patty options
  2^condiments * patty_choices = 4096 :=
by sorry

end hamburger_combinations_l108_108815


namespace tunnel_connects_land_l108_108053

noncomputable def surface_area (planet : Type) : ℝ := sorry
noncomputable def land_area (planet : Type) : ℝ := sorry
noncomputable def half_surface_area (planet : Type) : ℝ := surface_area planet / 2
noncomputable def can_dig_tunnel_through_center (planet : Type) : Prop := sorry

variable {TauCeti : Type}

-- Condition: Land occupies more than half of the entire surface area.
axiom land_more_than_half : land_area TauCeti > half_surface_area TauCeti

-- Proof problem statement: Prove that inhabitants can dig a tunnel through the center of the planet.
theorem tunnel_connects_land : can_dig_tunnel_through_center TauCeti :=
sorry

end tunnel_connects_land_l108_108053


namespace hallie_made_100_per_painting_l108_108143

-- Define conditions
def num_paintings : ℕ := 3
def total_money_made : ℕ := 300

-- Define the goal
def money_per_painting : ℕ := total_money_made / num_paintings

theorem hallie_made_100_per_painting :
  money_per_painting = 100 :=
sorry

end hallie_made_100_per_painting_l108_108143


namespace relationship_between_P_and_Q_l108_108462

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem relationship_between_P_and_Q : 
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
sorry

end relationship_between_P_and_Q_l108_108462


namespace pencil_rows_l108_108079

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) : (total_pencils / pencils_per_row) = 7 :=
by
  sorry

end pencil_rows_l108_108079


namespace sum_of_transformed_parabolas_is_non_horizontal_line_l108_108251

theorem sum_of_transformed_parabolas_is_non_horizontal_line
    (a b c : ℝ)
    (f g : ℝ → ℝ)
    (hf : ∀ x, f x = a * (x - 8)^2 + b * (x - 8) + c)
    (hg : ∀ x, g x = -a * (x + 8)^2 - b * (x + 8) - (c - 3)) :
    ∃ m q : ℝ, ∀ x : ℝ, (f x + g x) = m * x + q ∧ m ≠ 0 :=
by sorry

end sum_of_transformed_parabolas_is_non_horizontal_line_l108_108251


namespace find_3a_plus_4b_l108_108617

noncomputable def g (x : ℝ) := 3 * x - 6

noncomputable def f_inverse (x : ℝ) := (3 * x - 2) / 2

noncomputable def f (x : ℝ) (a b : ℝ) := a * x + b

theorem find_3a_plus_4b (a b : ℝ) (h1 : ∀ x, g x = 2 * f_inverse x - 4) (h2 : ∀ x, f_inverse (f x a b) = x) :
  3 * a + 4 * b = 14 / 3 :=
sorry

end find_3a_plus_4b_l108_108617


namespace count_numbers_divisible_by_12_not_20_l108_108890

theorem count_numbers_divisible_by_12_not_20 : 
  let N := 2017
  let a := Nat.floor (N / 12)
  let b := Nat.floor (N / 60)
  a - b = 135 := by
    -- Definitions used
    let N := 2017
    let a := Nat.floor (N / 12)
    let b := Nat.floor (N / 60)
    -- The desired statement
    show a - b = 135
    sorry

end count_numbers_divisible_by_12_not_20_l108_108890


namespace movie_length_l108_108763

theorem movie_length (paused_midway : ∃ t : ℝ, t = t ∧ t / 2 = 30) : 
  ∃ total_length : ℝ, total_length = 60 :=
by {
  sorry
}

end movie_length_l108_108763


namespace six_digit_squares_l108_108879

theorem six_digit_squares (x y : ℕ) 
  (h1 : y < 1000)
  (h2 : (1000 * x + y) < 1000000)
  (h3 : y * (y - 1) = 1000 * x)
  (mod8 : y * (y - 1) ≡ 0 [MOD 8])
  (mod125 : y * (y - 1) ≡ 0 [MOD 125]) :
  (1000 * x + y = 390625 ∨ 1000 * x + y = 141376) :=
sorry

end six_digit_squares_l108_108879


namespace percentage_of_teachers_without_issues_l108_108242

theorem percentage_of_teachers_without_issues (total_teachers : ℕ) 
    (high_bp_teachers : ℕ) (heart_issue_teachers : ℕ) 
    (both_issues_teachers : ℕ) (h1 : total_teachers = 150) 
    (h2 : high_bp_teachers = 90) 
    (h3 : heart_issue_teachers = 60) 
    (h4 : both_issues_teachers = 30) : 
    (total_teachers - (high_bp_teachers + heart_issue_teachers - both_issues_teachers)) / total_teachers * 100 = 20 :=
by sorry

end percentage_of_teachers_without_issues_l108_108242


namespace loraine_wax_usage_proof_l108_108334

-- Conditions
variables (large_animals small_animals : ℕ)
variable (wax : ℕ)

-- Definitions based on conditions
def large_animal_wax := 4
def small_animal_wax := 2
def total_sticks := 20
def small_animals_wax := 12
def small_to_large_ratio := 3

-- Proof statement
theorem loraine_wax_usage_proof (h1 : small_animals_wax = small_animals * small_animal_wax)
  (h2 : small_animals = large_animals * small_to_large_ratio)
  (h3 : wax = small_animals_wax + large_animals * large_animal_wax) :
  wax = total_sticks := by
  sorry

end loraine_wax_usage_proof_l108_108334


namespace randy_used_36_blocks_l108_108058

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks left
def blocks_left : ℕ := 23

-- Define the number of blocks used
def blocks_used (initial left : ℕ) : ℕ := initial - left

-- Prove that Randy used 36 blocks
theorem randy_used_36_blocks : blocks_used initial_blocks blocks_left = 36 := 
by
  -- Proof will be here
  sorry

end randy_used_36_blocks_l108_108058


namespace intersection_of_sets_l108_108336

open Set

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3, 4, 5}) (hB : B = {2, 4, 6}) :
  A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_sets_l108_108336


namespace seq_expression_l108_108486

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n^2 * a n

theorem seq_expression (a : ℕ → ℝ) (h₁ : a 1 = 2) (h₂ : ∀ n ≥ 1, S n a = n^2 * a n) :
  ∀ n ≥ 1, a n = 4 / (n * (n + 1)) :=
by
  sorry

end seq_expression_l108_108486


namespace initial_flour_amount_l108_108644

theorem initial_flour_amount (initial_flour : ℕ) (additional_flour : ℕ) (total_flour : ℕ) 
  (h1 : additional_flour = 4) (h2 : total_flour = 16) (h3 : initial_flour + additional_flour = total_flour) :
  initial_flour = 12 := 
by 
  sorry

end initial_flour_amount_l108_108644


namespace dana_jellybeans_l108_108619

noncomputable def jellybeans_in_dana_box (alex_capacity : ℝ) (mul_factor : ℝ) : ℝ :=
  let alex_volume := 1 * 1 * 1.5
  let dana_volume := mul_factor * mul_factor * (mul_factor * 1.5)
  let volume_ratio := dana_volume / alex_volume
  volume_ratio * alex_capacity

theorem dana_jellybeans
  (alex_capacity : ℝ := 150)
  (mul_factor : ℝ := 3) :
  jellybeans_in_dana_box alex_capacity mul_factor = 4050 :=
by
  rw [jellybeans_in_dana_box]
  simp
  sorry

end dana_jellybeans_l108_108619


namespace determine_event_C_l108_108694

variable (A B C : Prop)
variable (Tallest Shortest : Prop)
variable (Running LongJump ShotPut : Prop)

variables (part_A_Running part_A_LongJump part_A_ShotPut
           part_B_Running part_B_LongJump part_B_ShotPut
           part_C_Running part_C_LongJump part_C_ShotPut : Prop)

variable (not_tallest_A : ¬Tallest → A)
variable (not_tallest_ShotPut : Tallest → ¬ShotPut)
variable (shortest_LongJump : Shortest → LongJump)
variable (not_shortest_B : ¬Shortest → B)
variable (not_running_B : ¬Running → B)

theorem determine_event_C :
  (¬Tallest → A) →
  (Tallest → ¬ShotPut) →
  (Shortest → LongJump) →
  (¬Shortest → B) →
  (¬Running → B) →
  part_C_Running :=
by
  intros h1 h2 h3 h4 h5
  sorry

end determine_event_C_l108_108694


namespace max_value_fraction_l108_108681

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) <= (2 / 3) := 
sorry

end max_value_fraction_l108_108681


namespace no_integer_solutions_for_eq_l108_108960

theorem no_integer_solutions_for_eq {x y : ℤ} : ¬ (∃ x y : ℤ, (x + 7) * (x + 6) = 8 * y + 3) := by
  sorry

end no_integer_solutions_for_eq_l108_108960


namespace prove_expression_l108_108499

def given_expression : ℤ := -4 + 6 / (-2)

theorem prove_expression : given_expression = -7 := 
by 
  -- insert proof here
  sorry

end prove_expression_l108_108499


namespace max_valid_committees_l108_108706

-- Define the conditions
def community_size : ℕ := 20
def english_speakers : ℕ := 10
def german_speakers : ℕ := 10
def french_speakers : ℕ := 10
def total_subsets : ℕ := Nat.choose community_size 3
def invalid_subsets_per_language : ℕ := Nat.choose 10 3

-- Lean statement to verify the number of valid committees
theorem max_valid_committees :
  total_subsets - 3 * invalid_subsets_per_language = 1020 :=
by
  simp [community_size, total_subsets, invalid_subsets_per_language]
  sorry

end max_valid_committees_l108_108706


namespace floral_shop_bouquets_l108_108655

theorem floral_shop_bouquets (T : ℕ) 
  (h1 : 12 + T + T / 3 = 60) 
  (hT : T = 36) : T / 12 = 3 :=
by
  -- Proof steps go here
  sorry

end floral_shop_bouquets_l108_108655


namespace linear_equation_l108_108213

noncomputable def is_linear (k : ℝ) : Prop :=
  2 * (|k|) = 1 ∧ k ≠ 1

theorem linear_equation (k : ℝ) : is_linear k ↔ k = -1 :=
by
  sorry

end linear_equation_l108_108213


namespace agreed_period_of_service_l108_108550

theorem agreed_period_of_service (x : ℕ) (rs800 : ℕ) (rs400 : ℕ) (servant_period : ℕ) (received_amount : ℕ) (uniform : ℕ) (half_period : ℕ) :
  rs800 = 800 ∧ rs400 = 400 ∧ servant_period = 9 ∧ received_amount = 400 ∧ half_period = x / 2 ∧ servant_period = half_period → x = 18 :=
by sorry

end agreed_period_of_service_l108_108550


namespace no_real_solutions_l108_108054

theorem no_real_solutions : ∀ (x y : ℝ), ¬ (3 * x^2 + y^2 - 9 * x - 6 * y + 23 = 0) :=
by sorry

end no_real_solutions_l108_108054


namespace amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l108_108366

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 2) + 1

theorem amplitude_of_f : (∀ x y : ℝ, |f x - f y| ≤ 2 * |x - y|) := sorry

theorem phase_shift_of_f : (∃ φ : ℝ, φ = -Real.pi / 8) := sorry

theorem vertical_shift_of_f : (∃ v : ℝ, v = 1) := sorry

end amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l108_108366


namespace speed_of_mans_train_is_80_kmph_l108_108141

-- Define the given constants
def length_goods_train : ℤ := 280 -- length in meters
def time_to_pass : ℤ := 9 -- time in seconds
def speed_goods_train : ℤ := 32 -- speed in km/h

-- Define the conversion factor from km/h to m/s
def kmh_to_ms (v : ℤ) : ℤ := v * 1000 / 3600

-- Define the speed of the goods train in m/s
def speed_goods_train_ms := kmh_to_ms speed_goods_train

-- Define the speed of the man's train in km/h
def speed_mans_train : ℤ := 80

-- Prove that the speed of the man's train is 80 km/h given the conditions
theorem speed_of_mans_train_is_80_kmph :
  ∃ V : ℤ,
    (V + speed_goods_train) * 1000 / 3600 = length_goods_train / time_to_pass → 
    V = speed_mans_train :=
by
  sorry

end speed_of_mans_train_is_80_kmph_l108_108141


namespace smallest_x_solution_l108_108304

def smallest_x_condition (x : ℝ) : Prop :=
  (x^2 - 5 * x - 84 = (x - 12) * (x + 7)) ∧
  (x ≠ 9) ∧
  (x ≠ -7) ∧
  ((x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 7))

theorem smallest_x_solution :
  ∃ x : ℝ, smallest_x_condition x ∧ ∀ y : ℝ, smallest_x_condition y → x ≤ y :=
sorry

end smallest_x_solution_l108_108304


namespace range_of_a_l108_108899

noncomputable def f (a x : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) → a ≤ 4 :=
by
  sorry

end range_of_a_l108_108899


namespace count_sums_of_three_cubes_l108_108718

theorem count_sums_of_three_cubes :
  let possible_sums := {n | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ n = a^3 + b^3 + c^3}
  ∃ unique_sums : Finset ℕ, (∀ x ∈ possible_sums, x < 1000) ∧ unique_sums.card = 153 :=
by sorry

end count_sums_of_three_cubes_l108_108718


namespace train_speed_180_kmph_l108_108601

def train_speed_in_kmph (length_meters : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_m_per_s := length_meters / time_seconds
  let speed_km_per_h := speed_m_per_s * 36 / 10
  speed_km_per_h

theorem train_speed_180_kmph:
  train_speed_in_kmph 400 8 = 180 := by
  sorry

end train_speed_180_kmph_l108_108601


namespace present_cost_after_discount_l108_108845

theorem present_cost_after_discount 
  (X : ℝ) (P : ℝ) 
  (h1 : X - 4 = (0.80 * P) / 3) 
  (h2 : P = 3 * X)
  :
  0.80 * P = 48 :=
by
  sorry

end present_cost_after_discount_l108_108845


namespace B_should_be_paid_2307_69_l108_108339

noncomputable def A_work_per_day : ℚ := 1 / 15
noncomputable def B_work_per_day : ℚ := 1 / 10
noncomputable def C_work_per_day : ℚ := 1 / 20
noncomputable def combined_work_per_day : ℚ := A_work_per_day + B_work_per_day + C_work_per_day
noncomputable def total_work : ℚ := 1
noncomputable def total_wages : ℚ := 5000
noncomputable def time_taken : ℚ := total_work / combined_work_per_day
noncomputable def B_share_of_work : ℚ := B_work_per_day / combined_work_per_day
noncomputable def B_share_of_wages : ℚ := B_share_of_work * total_wages

theorem B_should_be_paid_2307_69 : B_share_of_wages = 2307.69 := by
  sorry

end B_should_be_paid_2307_69_l108_108339


namespace ratio_ba_in_range_l108_108958

theorem ratio_ba_in_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (h1 : a + 2 * b = 7) (h2 : a^2 + b^2 ≤ 25) : 
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ 4 / 3 :=
by {
  sorry
}

end ratio_ba_in_range_l108_108958


namespace second_number_in_set_l108_108388

theorem second_number_in_set (avg1 avg2 n1 n2 n3 : ℕ) (h1 : avg1 = (10 + 70 + 19) / 3) (h2 : avg2 = avg1 + 7) (h3 : n1 = 20) (h4 : n3 = 60) :
  n2 = n3 := 
  sorry

end second_number_in_set_l108_108388


namespace maximum_value_of_sum_l108_108739

variables (x y : ℝ)

def s : ℝ := x + y

theorem maximum_value_of_sum (h : s ≤ 9) : s = 9 :=
sorry

end maximum_value_of_sum_l108_108739


namespace no_such_integers_l108_108328

theorem no_such_integers :
  ¬ (∃ a b c d : ℤ, a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_l108_108328


namespace circle_through_point_and_same_center_l108_108591

theorem circle_through_point_and_same_center :
  ∃ (x_0 y_0 r : ℝ),
    (∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      x^2 + y^2 - 4 * x + 6 * y - 3 = 0)
    ∧
    ∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      (x - 2)^2 + (y + 3)^2 = 25 := sorry

end circle_through_point_and_same_center_l108_108591


namespace max_S_n_l108_108633

/-- Arithmetic sequence proof problem -/
theorem max_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 + a 3 + a 5 = 15)
  (h2 : a 2 + a 4 + a 6 = 0)
  (d : ℝ) (h3 : ∀ n, a (n + 1) = a n + d) :
  (∃ n, S n = 30) :=
sorry

end max_S_n_l108_108633


namespace greatest_integer_difference_l108_108164

theorem greatest_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  ∃ d : ℤ, d = y - x ∧ ∀ z, 4 < z ∧ z < 8 ∧ 8 < y ∧ y < 12 → (y - z ≤ d) :=
sorry

end greatest_integer_difference_l108_108164


namespace average_stamps_per_day_l108_108016

theorem average_stamps_per_day :
  let a1 := 8
  let d := 8
  let n := 6
  let stamps_collected : Fin n → ℕ := λ i => a1 + i * d
  -- sum the stamps collected over six days
  let S := List.sum (List.ofFn stamps_collected)
  -- calculate average
  let average := S / n
  average = 28 :=
by sorry

end average_stamps_per_day_l108_108016


namespace phone_charges_equal_l108_108555

theorem phone_charges_equal (x : ℝ) : 
  (0.60 + 14 * x = 0.08 * 18) → (x = 0.06) :=
by
  intro h
  have : 14 * x = 1.44 - 0.60 := sorry
  have : 14 * x = 0.84 := sorry
  have : x = 0.06 := sorry
  exact this

end phone_charges_equal_l108_108555


namespace smallest_w_l108_108157

theorem smallest_w (w : ℕ) (w_pos : w > 0) (h1 : ∀ n : ℕ, 2^4 ∣ 1452 * w)
                              (h2 : ∀ n : ℕ, 3^3 ∣ 1452 * w)
                              (h3 : ∀ n : ℕ, 13^3 ∣ 1452 * w) :
  w = 676 := sorry

end smallest_w_l108_108157


namespace empty_solution_set_l108_108990

theorem empty_solution_set 
  (x : ℝ) 
  (h : -2 + 3 * x - 2 * x^2 > 0) : 
  false :=
by
  -- Discriminant calculation to prove empty solution set
  let delta : ℝ := 9 - 4 * 2 * 2
  have h_delta : delta < 0 := by norm_num
  sorry

end empty_solution_set_l108_108990


namespace time_to_fill_remaining_l108_108506

-- Define the rates at which pipes P and Q fill the cistern
def rate_P := 1 / 12
def rate_Q := 1 / 15

-- Define the time both pipes are open together
def time_both_open := 4

-- Calculate the combined rate when both pipes are open
def combined_rate := rate_P + rate_Q

-- Calculate the amount of the cistern filled in the time both pipes are open
def filled_amount_both_open := time_both_open * combined_rate

-- Calculate the remaining amount to fill after Pipe P is turned off
def remaining_amount := 1 - filled_amount_both_open

-- Calculate the time it will take for Pipe Q alone to fill the remaining amount
def time_Q_to_fill_remaining := remaining_amount / rate_Q

-- The final theorem
theorem time_to_fill_remaining : time_Q_to_fill_remaining = 6 := by
  sorry

end time_to_fill_remaining_l108_108506


namespace total_yards_run_l108_108206

-- Define the yardages and games for each athlete
def Malik_yards_per_game : ℕ := 18
def Malik_games : ℕ := 5

def Josiah_yards_per_game : ℕ := 22
def Josiah_games : ℕ := 7

def Darnell_yards_per_game : ℕ := 11
def Darnell_games : ℕ := 4

def Kade_yards_per_game : ℕ := 15
def Kade_games : ℕ := 6

-- Prove that the total yards run by the four athletes is 378
theorem total_yards_run :
  (Malik_yards_per_game * Malik_games) +
  (Josiah_yards_per_game * Josiah_games) +
  (Darnell_yards_per_game * Darnell_games) +
  (Kade_yards_per_game * Kade_games) = 378 :=
by
  sorry

end total_yards_run_l108_108206


namespace pyramid_volume_l108_108000

-- Define the conditions
def height_vertex_to_center_base := 12 -- cm
def side_of_square_base := 10 -- cm
def base_area := side_of_square_base * side_of_square_base -- cm²
def volume := (1 / 3) * base_area * height_vertex_to_center_base -- cm³

-- State the theorem
theorem pyramid_volume : volume = 400 := 
by
  -- Placeholder for the proof
  sorry

end pyramid_volume_l108_108000


namespace sam_gave_2_puppies_l108_108201

theorem sam_gave_2_puppies (original_puppies given_puppies remaining_puppies : ℕ) 
  (h1 : original_puppies = 6) (h2 : remaining_puppies = 4) :
  given_puppies = original_puppies - remaining_puppies := by 
  sorry

end sam_gave_2_puppies_l108_108201


namespace B_work_days_l108_108956

theorem B_work_days (a b : ℝ) (h1 : a + b = 1/4) (h2 : a = 1/14) : 1 / b = 5.6 :=
by
  sorry

end B_work_days_l108_108956


namespace striped_octopus_has_eight_legs_l108_108207

variable (has_even_legs : ℕ → Prop)
variable (lie_told : ℕ → Prop)

variable (green_leg_count : ℕ)
variable (blue_leg_count : ℕ)
variable (violet_leg_count : ℕ)
variable (striped_leg_count : ℕ)

-- Conditions
axiom even_truth_lie_relation : ∀ n, has_even_legs n ↔ ¬lie_told n
axiom green_statement : lie_told green_leg_count ↔ (has_even_legs green_leg_count ∧ lie_told blue_leg_count)
axiom blue_statement : lie_told blue_leg_count ↔ (has_even_legs blue_leg_count ∧ lie_told green_leg_count)
axiom violet_statement : lie_told violet_leg_count ↔ (has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count)
axiom striped_statement : ¬has_even_legs green_leg_count ∧ ¬has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count ∧ has_even_legs striped_leg_count

-- The Proof Goal
theorem striped_octopus_has_eight_legs : has_even_legs striped_leg_count ∧ striped_leg_count = 8 :=
by
  sorry -- Proof to be filled in

end striped_octopus_has_eight_legs_l108_108207


namespace selling_prices_l108_108434

theorem selling_prices {x y : ℝ} (h1 : y - x = 10) (h2 : (y - 5) - 1.10 * x = 1) :
  x = 40 ∧ y = 50 := by
  sorry

end selling_prices_l108_108434


namespace distance_traveled_by_both_cars_l108_108837

def car_R_speed := 34.05124837953327
def car_P_speed := 44.05124837953327
def car_R_time := 8.810249675906654
def car_P_time := car_R_time - 2

def distance_car_R := car_R_speed * car_R_time
def distance_car_P := car_P_speed * car_P_time

theorem distance_traveled_by_both_cars :
  distance_car_R = 300 :=
by
  sorry

end distance_traveled_by_both_cars_l108_108837


namespace merchant_marked_price_l108_108709

theorem merchant_marked_price (L P x S : ℝ)
  (h1 : L = 100)
  (h2 : P = 70)
  (h3 : S = 0.8 * x)
  (h4 : 0.8 * x - 70 = 0.3 * (0.8 * x)) :
  x = 125 :=
by
  sorry

end merchant_marked_price_l108_108709


namespace vacant_seats_l108_108205

open Nat

-- Define the conditions as Lean definitions
def num_tables : Nat := 5
def seats_per_table : Nat := 8
def occupied_tables : Nat := 2
def people_per_occupied_table : Nat := 3
def unusable_tables : Nat := 1

-- Calculate usable tables
def usable_tables : Nat := num_tables - unusable_tables

-- Calculate total occupied people
def total_occupied_people : Nat := occupied_tables * people_per_occupied_table

-- Calculate total seats for occupied tables
def total_seats_occupied_tables : Nat := occupied_tables * seats_per_table

-- Calculate vacant seats in occupied tables
def vacant_seats_occupied_tables : Nat := total_seats_occupied_tables - total_occupied_people

-- Calculate completely unoccupied tables
def unoccupied_tables : Nat := usable_tables - occupied_tables

-- Calculate total seats for unoccupied tables
def total_seats_unoccupied_tables : Nat := unoccupied_tables * seats_per_table

-- Calculate total vacant seats
def total_vacant_seats : Nat := vacant_seats_occupied_tables + total_seats_unoccupied_tables

-- Theorem statement to prove
theorem vacant_seats : total_vacant_seats = 26 := by
  sorry

end vacant_seats_l108_108205


namespace largest_neg_int_solution_l108_108269

theorem largest_neg_int_solution :
  ∃ x : ℤ, 26 * x + 8 ≡ 4 [ZMOD 18] ∧ ∀ y : ℤ, 26 * y + 8 ≡ 4 [ZMOD 18] → y < -14 → false :=
by
  sorry

end largest_neg_int_solution_l108_108269


namespace problem_statement_l108_108329

noncomputable def theta (h1 : 2 * Real.cos θ + Real.sin θ = 0) (h2 : 0 < θ ∧ θ < Real.pi) : Real :=
θ

noncomputable def varphi (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) : Real :=
φ

theorem problem_statement
  (θ : Real) (φ : Real)
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧
  Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
  Real.cos θ = -Real.sqrt 5 / 5 ∧
  Real.cos φ = -Real.sqrt 2 / 10 :=
by
  sorry

end problem_statement_l108_108329


namespace expression_evaluation_l108_108666

theorem expression_evaluation (a b c d : ℤ) : 
  a / b - c * d^2 = a / (b - c * d^2) :=
sorry

end expression_evaluation_l108_108666


namespace max_sum_l108_108457

open Real

theorem max_sum (a b c : ℝ) (h : a^2 + (b^2) / 4 + (c^2) / 9 = 1) : a + b + c ≤ sqrt 14 :=
sorry

end max_sum_l108_108457


namespace percentage_exceeds_self_l108_108059

theorem percentage_exceeds_self (N : ℝ) (P : ℝ) (hN : N = 75) (h_condition : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end percentage_exceeds_self_l108_108059


namespace steve_and_laura_meet_time_l108_108190

structure PathsOnParallelLines where
  steve_speed : ℝ
  laura_speed : ℝ
  path_separation : ℝ
  art_diameter : ℝ
  initial_distance_hidden : ℝ

def meet_time (p : PathsOnParallelLines) : ℝ :=
  sorry -- To be proven

-- Define the specific case for Steve and Laura
def steve_and_laura_paths : PathsOnParallelLines :=
  { steve_speed := 3,
    laura_speed := 1,
    path_separation := 240,
    art_diameter := 80,
    initial_distance_hidden := 230 }

theorem steve_and_laura_meet_time :
  meet_time steve_and_laura_paths = 45 :=
  sorry

end steve_and_laura_meet_time_l108_108190


namespace g_g_g_3_equals_107_l108_108255

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end g_g_g_3_equals_107_l108_108255


namespace find_integer_for_perfect_square_l108_108136

theorem find_integer_for_perfect_square :
  ∃ (n : ℤ), ∃ (m : ℤ), n^2 + 20 * n + 11 = m^2 ∧ n = 35 := by
  sorry

end find_integer_for_perfect_square_l108_108136


namespace range_b_intersects_ellipse_l108_108493

open Real

noncomputable def line_intersects_ellipse (b : ℝ) : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < π → ∃ x y : ℝ, x = 2 * cos θ ∧ y = 4 * sin θ ∧ y = x + b

theorem range_b_intersects_ellipse :
  ∀ b : ℝ, line_intersects_ellipse b ↔ b ∈ Set.Icc (-2 : ℝ) (2 * sqrt 5) :=
by
  sorry

end range_b_intersects_ellipse_l108_108493


namespace ends_with_two_zeros_l108_108678

theorem ends_with_two_zeros (x y : ℕ) (h : (x^2 + x * y + y^2) % 10 = 0) : (x^2 + x * y + y^2) % 100 = 0 :=
sorry

end ends_with_two_zeros_l108_108678


namespace no_discrepancy_l108_108385

-- Definitions based on the conditions
def t1_hours : ℝ := 1.5 -- time taken clockwise in hours
def t2_minutes : ℝ := 90 -- time taken counterclockwise in minutes

-- Lean statement to prove the equivalence
theorem no_discrepancy : t1_hours * 60 = t2_minutes :=
by sorry

end no_discrepancy_l108_108385


namespace find_positive_integer_l108_108055

def product_of_digits (n : Nat) : Nat :=
  -- Function to compute product of digits, assume it is defined correctly
  sorry

theorem find_positive_integer (x : Nat) (h : x > 0) :
  product_of_digits x = x * x - 10 * x - 22 ↔ x = 12 :=
by
  sorry

end find_positive_integer_l108_108055


namespace proof_problem_l108_108183

structure Plane := (name : String)
structure Line := (name : String)

def parallel_planes (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def parallel_lines (m n : Line) : Prop := sorry

theorem proof_problem (m : Line) (α β : Plane) :
  parallel_planes α β → in_plane m α → parallel_lines m (Line.mk β.name) :=
sorry

end proof_problem_l108_108183


namespace cost_of_book_l108_108306

-- Definitions based on the conditions
def cost_pen : ℕ := 4
def cost_ruler : ℕ := 1
def fifty_dollar_bill : ℕ := 50
def change_received : ℕ := 20
def total_spent : ℕ := fifty_dollar_bill - change_received

-- Problem Statement: Prove the cost of the book
theorem cost_of_book : ∀ (cost_pen cost_ruler total_spent : ℕ), 
  total_spent = 50 - 20 → cost_pen = 4 → cost_ruler = 1 →
  (total_spent - (cost_pen + cost_ruler) = 25) :=
by
  intros cost_pen cost_ruler total_spent h1 h2 h3
  sorry

end cost_of_book_l108_108306


namespace max_distinct_values_is_two_l108_108276

-- Definitions of non-negative numbers and conditions
variable (a b c d : ℝ)
variable (ha : 0 ≤ a)
variable (hb : 0 ≤ b)
variable (hc : 0 ≤ c)
variable (hd : 0 ≤ d)
variable (h1 : Real.sqrt (a + b) + Real.sqrt (c + d) = Real.sqrt (a + c) + Real.sqrt (b + d))
variable (h2 : Real.sqrt (a + c) + Real.sqrt (b + d) = Real.sqrt (a + d) + Real.sqrt (b + c))

-- Theorem stating that the maximum number of distinct values among a, b, c, d is 2.
theorem max_distinct_values_is_two : 
  ∃ (u v : ℝ), 0 ≤ u ∧ 0 ≤ v ∧ (u = a ∨ u = b ∨ u = c ∨ u = d) ∧ (v = a ∨ v = b ∨ v = c ∨ v = d) ∧ 
  ∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x = y ∨ x = u ∨ x = v :=
sorry

end max_distinct_values_is_two_l108_108276


namespace green_pill_cost_l108_108872

variable (x : ℝ) -- cost of a green pill in dollars
variable (y : ℝ) -- cost of a pink pill in dollars
variable (total_cost : ℝ) -- total cost for 21 days

theorem green_pill_cost
  (h1 : x = y + 2) -- a green pill costs $2 more than a pink pill
  (h2 : total_cost = 819) -- total cost for 21 days is $819
  (h3 : ∀ n, n = 21 ∧ total_cost / n = (x + y)) :
  x = 20.5 :=
by
  sorry

end green_pill_cost_l108_108872


namespace loaves_at_start_l108_108995

variable (X : ℕ) -- X represents the number of loaves at the start of the day.

-- Conditions given in the problem:
def final_loaves (X : ℕ) : Prop := X - 629 + 489 = 2215

-- The theorem to be proved:
theorem loaves_at_start (h : final_loaves X) : X = 2355 :=
by sorry

end loaves_at_start_l108_108995


namespace find_a2_l108_108775

def arithmetic_sequence (a : ℕ → ℚ) := 
  (a 1 = 1) ∧ ∀ n, a (n + 2) - a n = 3

theorem find_a2 (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 2 = 5 / 2 := 
by
  -- Conditions
  have a1 : a 1 = 1 := h.1
  have h_diff : ∀ n, a (n + 2) - a n = 3 := h.2
  -- Proof steps can be written here
  sorry

end find_a2_l108_108775


namespace min_ratio_cyl_inscribed_in_sphere_l108_108380

noncomputable def min_surface_area_to_volume_ratio (R r : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (R^2 - r^2)
  let A := 2 * Real.pi * r * (h + r)
  let V := Real.pi * r^2 * h
  A / V

theorem min_ratio_cyl_inscribed_in_sphere (R : ℝ) :
  ∃ r h, h = 2 * Real.sqrt (R^2 - r^2) ∧
         min_surface_area_to_volume_ratio R r = (Real.sqrt (Real.sqrt 4 + 1))^3 / R := 
by {
  sorry
}

end min_ratio_cyl_inscribed_in_sphere_l108_108380


namespace increased_amount_is_30_l108_108588

noncomputable def F : ℝ := (3 / 2) * 179.99999999999991
noncomputable def F' : ℝ := (5 / 3) * 179.99999999999991
noncomputable def J : ℝ := 179.99999999999991
noncomputable def increased_amount : ℝ := F' - F

theorem increased_amount_is_30 : increased_amount = 30 :=
by
  -- Placeholder for proof. Actual proof goes here.
  sorry

end increased_amount_is_30_l108_108588


namespace mean_median_difference_is_minus_4_l108_108972

-- Defining the percentages of students scoring specific points
def perc_60 : ℝ := 0.20
def perc_75 : ℝ := 0.55
def perc_95 : ℝ := 0.10
def perc_110 : ℝ := 1 - (perc_60 + perc_75 + perc_95) -- 0.15

-- Defining the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_95 : ℝ := 95
def score_110 : ℝ := 110

-- Calculating the mean score
def mean_score : ℝ := (perc_60 * score_60) + (perc_75 * score_75) + (perc_95 * score_95) + (perc_110 * score_110)

-- Given the median score
def median_score : ℝ := score_75

-- Defining the expected difference
def expected_difference : ℝ := mean_score - median_score

theorem mean_median_difference_is_minus_4 :
  expected_difference = -4 := by sorry

end mean_median_difference_is_minus_4_l108_108972


namespace smallest_n_l108_108912

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3 * n = k ^ 2) (h2 : ∃ m : ℕ, 5 * n = m ^ 5) : n = 151875 := sorry

end smallest_n_l108_108912


namespace sin_35pi_over_6_l108_108047

theorem sin_35pi_over_6 : Real.sin (35 * Real.pi / 6) = -1 / 2 := by
  sorry

end sin_35pi_over_6_l108_108047


namespace union_complement_eq_l108_108689

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {-1, 0, 3}

theorem union_complement_eq :
  A ∪ (U \ B) = {-2, -1, 0, 1, 2} := by
  sorry

end union_complement_eq_l108_108689


namespace ratio_WX_XY_l108_108118

theorem ratio_WX_XY (p q : ℝ) (h : 3 * p = 4 * q) : (4 * q) / (3 * p) = 12 / 7 := by
  sorry

end ratio_WX_XY_l108_108118


namespace bottle_caps_total_l108_108906

-- Define the conditions
def groups : ℕ := 7
def caps_per_group : ℕ := 5

-- State the theorem
theorem bottle_caps_total : groups * caps_per_group = 35 :=
by
  sorry

end bottle_caps_total_l108_108906


namespace number_of_boys_girls_l108_108826

-- Define the initial conditions.
def group_size : ℕ := 8
def total_ways : ℕ := 90

-- Define the actual proof problem.
theorem number_of_boys_girls 
  (n m : ℕ) 
  (h1 : n + m = group_size) 
  (h2 : Nat.choose n 2 * Nat.choose m 1 * Nat.factorial 3 = total_ways) 
  : n = 3 ∧ m = 5 :=
sorry

end number_of_boys_girls_l108_108826


namespace pharmacy_incurs_loss_l108_108700

variable (a b : ℝ)
variable (h : a < b)

theorem pharmacy_incurs_loss 
  (H : (41 * a + 59 * b) > 100 * (a + b) / 2) : true :=
by
  sorry

end pharmacy_incurs_loss_l108_108700


namespace hexagon_division_ratio_l108_108025

theorem hexagon_division_ratio
  (hex_area : ℝ)
  (hexagon : ∀ (A B C D E F : ℝ), hex_area = 8)
  (line_PQ_splits : ∀ (above_area below_area : ℝ), above_area = 4 ∧ below_area = 4)
  (below_PQ : ℝ)
  (unit_square_area : ∀ (unit_square : ℝ), unit_square = 1)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (triangle_area : ∀ (base height : ℝ), triangle_base = 4 ∧ (base * height) / 2 = 3)
  (XQ QY : ℝ)
  (bases_sum : ∀ (XQ QY : ℝ), XQ + QY = 4) :
  XQ / QY = 2 / 3 :=
sorry

end hexagon_division_ratio_l108_108025


namespace sandy_correct_sums_l108_108771

variables (x y : ℕ)

theorem sandy_correct_sums :
  (x + y = 30) →
  (3 * x - 2 * y = 50) →
  x = 22 :=
by
  intro h1 h2
  -- Proof will be filled in here
  sorry

end sandy_correct_sums_l108_108771


namespace bus_stop_time_per_hour_l108_108102

theorem bus_stop_time_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h1 : speed_without_stoppages = 64)
  (h2 : speed_with_stoppages = 48) : 
  ∃ t : ℝ, t = 15 := 
by
  sorry

end bus_stop_time_per_hour_l108_108102


namespace cube_sum_inequality_l108_108794

theorem cube_sum_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) : 
  a^3 + b^3 ≤ a * b^2 + a^2 * b :=
sorry

end cube_sum_inequality_l108_108794


namespace problem_1_problem_2_l108_108352

theorem problem_1 (n : ℕ) (h : n > 0) (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, (n > 0) → 
    (∃ α β, α + β = β * α + 1 ∧ 
            α * β = 1 / a n ∧ 
            a n * α^2 - a (n+1) * α + 1 = 0 ∧ 
            a n * β^2 - a (n+1) * β + 1 = 0)) :
  a (n + 1) = a n + 1 := sorry

theorem problem_2 (n : ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, (n > 0) → a (n+1) = a n + 1) :
  a n = n := sorry

end problem_1_problem_2_l108_108352


namespace find_total_results_l108_108256

noncomputable def total_results (S : ℕ) (n : ℕ) (sum_first6 sum_last6 sixth_result : ℕ) :=
  (S = 52 * n) ∧ (sum_first6 = 6 * 49) ∧ (sum_last6 = 6 * 52) ∧ (sixth_result = 34)

theorem find_total_results {S n sum_first6 sum_last6 sixth_result : ℕ} :
  total_results S n sum_first6 sum_last6 sixth_result → n = 11 :=
by
  intros h
  sorry

end find_total_results_l108_108256


namespace vertex_of_parabola_l108_108893

theorem vertex_of_parabola (c d : ℝ) (h₁ : ∀ x, -x^2 + c*x + d ≤ 0 ↔ (x ≤ -1 ∨ x ≥ 7)) : 
  ∃ v : ℝ × ℝ, v = (3, 16) :=
by
  sorry

end vertex_of_parabola_l108_108893


namespace group_total_payment_l108_108520

-- Declare the costs of the tickets as constants
def cost_adult : ℝ := 9.50
def cost_child : ℝ := 6.50

-- Conditions for the group
def total_moviegoers : ℕ := 7
def number_adults : ℕ := 3

-- Calculate the number of children
def number_children : ℕ := total_moviegoers - number_adults

-- Define the total cost paid by the group
def total_cost_paid : ℝ :=
  (number_adults * cost_adult) + (number_children * cost_child)

-- The proof problem: Prove that the total amount paid by the group is $54.50
theorem group_total_payment : total_cost_paid = 54.50 := by
  sorry

end group_total_payment_l108_108520


namespace average_visitors_per_day_l108_108795

theorem average_visitors_per_day (avg_visitors_Sunday : ℕ) (avg_visitors_other_days : ℕ) (total_days : ℕ) (starts_on_Sunday : Bool) :
  avg_visitors_Sunday = 500 → 
  avg_visitors_other_days = 140 → 
  total_days = 30 → 
  starts_on_Sunday = true → 
  (4 * avg_visitors_Sunday + 26 * avg_visitors_other_days) / total_days = 188 :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l108_108795


namespace rice_wheat_ratio_l108_108438

theorem rice_wheat_ratio (total_shi : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) (total_sample : ℕ) : 
  total_shi = 1512 ∧ sample_size = 216 ∧ wheat_in_sample = 27 ∧ total_sample = 1512 * (wheat_in_sample / sample_size) →
  total_sample = 189 :=
by
  intros h
  sorry

end rice_wheat_ratio_l108_108438


namespace new_ticket_price_l108_108268

theorem new_ticket_price (a : ℕ) (x : ℝ) (initial_price : ℝ) (revenue_increase : ℝ) (spectator_increase : ℝ)
  (h₀ : initial_price = 25)
  (h₁ : spectator_increase = 1.5)
  (h₂ : revenue_increase = 1.14)
  (h₃ : x = 0.76):
  initial_price * x = 19 :=
by
  sorry

end new_ticket_price_l108_108268


namespace find_height_of_pyramid_l108_108961

noncomputable def volume (B h : ℝ) : ℝ := (1/3) * B * h
noncomputable def area_of_isosceles_right_triangle (leg : ℝ) : ℝ := (1/2) * leg * leg

theorem find_height_of_pyramid (leg : ℝ) (h : ℝ) (V : ℝ) (B : ℝ)
  (Hleg : leg = 3)
  (Hvol : V = 6)
  (Hbase : B = area_of_isosceles_right_triangle leg)
  (Hvol_eq : V = volume B h) :
  h = 4 :=
by
  sorry

end find_height_of_pyramid_l108_108961


namespace no_such_integers_exists_l108_108607

theorem no_such_integers_exists 
  (a b c d : ℤ) 
  (h1 : a * 19^3 + b * 19^2 + c * 19 + d = 1) 
  (h2 : a * 62^3 + b * 62^2 + c * 62 + d = 2) : 
  false :=
by
  sorry

end no_such_integers_exists_l108_108607


namespace find_remainder_l108_108708

-- Definitions
variable (x y : ℕ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : (x : ℝ) / y = 96.15)
variable (h4 : approximately_equal (y : ℝ) 60)

-- Target statement
theorem find_remainder : x % y = 9 :=
sorry

end find_remainder_l108_108708


namespace archipelago_max_value_l108_108046

noncomputable def archipelago_max_islands (N : ℕ) : Prop :=
  N ≥ 7 ∧ 
  (∀ (a b : ℕ), a ≠ b → a ≤ N → b ≤ N → ∃ c : ℕ, c ≤ N ∧ (∃ d, d ≠ c ∧ d ≤ N → d ≠ a ∧ d ≠ b)) ∧ 
  (∀ (a : ℕ), a ≤ N → ∃ b, b ≠ a ∧ b ≤ N ∧ (∃ c, c ≤ N ∧ c ≠ b ∧ c ≠ a))

theorem archipelago_max_value : archipelago_max_islands 36 := sorry

end archipelago_max_value_l108_108046


namespace unbroken_seashells_l108_108895

theorem unbroken_seashells (total broken : ℕ) (h1 : total = 7) (h2 : broken = 4) : total - broken = 3 :=
by
  -- Proof goes here…
  sorry

end unbroken_seashells_l108_108895


namespace exist_rel_prime_k_l_divisible_l108_108538

theorem exist_rel_prime_k_l_divisible (a b p : ℤ) : 
  ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := 
sorry

end exist_rel_prime_k_l_divisible_l108_108538


namespace cones_sold_l108_108623

-- Define the conditions
variable (milkshakes : Nat)
variable (cones : Nat)

-- Assume the given conditions
axiom h1 : milkshakes = 82
axiom h2 : milkshakes = cones + 15

-- State the theorem to prove
theorem cones_sold : cones = 67 :=
by
  -- Proof goes here
  sorry

end cones_sold_l108_108623


namespace probability_at_least_75_cents_l108_108598

def total_coins : ℕ := 3 + 5 + 4 + 3 -- total number of coins

def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 4
def quarters : ℕ := 3

def successful_outcomes_case1 : ℕ := (Nat.choose 3 3) * (Nat.choose 12 3)
def successful_outcomes_case2 : ℕ := (Nat.choose 3 2) * (Nat.choose 4 2) * (Nat.choose 5 2)

def total_outcomes : ℕ := Nat.choose 15 6
def successful_outcomes : ℕ := successful_outcomes_case1 + successful_outcomes_case2

def probability : ℚ := successful_outcomes / total_outcomes

theorem probability_at_least_75_cents :
  probability = 400 / 5005 := by
  sorry

end probability_at_least_75_cents_l108_108598


namespace smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l108_108944

theorem smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6 : 
  ∃ n : ℕ, (∃ k : ℕ, n = 60 * k + 1) ∧ n % 9 = 0 ∧ ∀ m : ℕ, (∃ k' : ℕ, m = 60 * k' + 1) ∧ m % 9 = 0 → n ≤ m :=
by
  sorry

end smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l108_108944


namespace rectangular_field_length_l108_108896

theorem rectangular_field_length (w : ℝ) (h₁ : w * (w + 10) = 171) : w + 10 = 19 := 
by
  sorry

end rectangular_field_length_l108_108896


namespace factorization_correct_l108_108811

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

end factorization_correct_l108_108811


namespace percent_answered_second_correctly_l108_108897

theorem percent_answered_second_correctly
  (nA : ℝ) (nAB : ℝ) (n_neither : ℝ) :
  nA = 0.80 → nAB = 0.60 → n_neither = 0.05 → 
  (nA + nB - nAB + n_neither = 1) → 
  ((1 - n_neither) = nA + nB - nAB) → 
  nB = 0.75 :=
by
  intros h1 h2 h3 hUnion hInclusion
  sorry

end percent_answered_second_correctly_l108_108897


namespace expression_values_l108_108156

theorem expression_values (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ b)
  (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 := 
sorry

end expression_values_l108_108156


namespace simplify_one_simplify_two_simplify_three_simplify_four_l108_108169

-- (1) Prove that (1 / 2) * sqrt(4 / 7) = sqrt(7) / 7
theorem simplify_one : (1 / 2) * Real.sqrt (4 / 7) = Real.sqrt 7 / 7 := sorry

-- (2) Prove that sqrt(20 ^ 2 - 15 ^ 2) = 5 * sqrt(7)
theorem simplify_two : Real.sqrt (20 ^ 2 - 15 ^ 2) = 5 * Real.sqrt 7 := sorry

-- (3) Prove that sqrt((32 * 9) / 25) = (12 * sqrt(2)) / 5
theorem simplify_three : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := sorry

-- (4) Prove that sqrt(22.5) = (3 * sqrt(10)) / 2
theorem simplify_four : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := sorry

end simplify_one_simplify_two_simplify_three_simplify_four_l108_108169


namespace incircle_angle_b_l108_108078

open Real

theorem incircle_angle_b
    (α β γ : ℝ)
    (h1 : α + β + γ = 180)
    (angle_AOC_eq_4_MKN : ∀ (MKN : ℝ), 4 * MKN = 180 - (180 - γ) / 2 - (180 - α) / 2) :
    β = 108 :=
by
  -- Proof will be handled here.
  sorry

end incircle_angle_b_l108_108078


namespace arithmetic_sequence_properties_l108_108240

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (T : ℕ → ℤ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) (h2 : a 4 = a 2 + 4) (h3 : a 3 = 6) :
  (∀ n, a n = 2 * n) ∧ (∀ n, T n = (4 / 3 * (4^n - 1))) :=
by
  sorry

end arithmetic_sequence_properties_l108_108240


namespace correct_sum_is_1826_l108_108444

-- Define the four-digit number representation
def four_digit (A B C D : ℕ) := 1000 * A + 100 * B + 10 * C + D

-- Condition: Yoongi confused the units digit (9 as 6)
-- The incorrect number Yoongi used
def incorrect_number (A B C : ℕ) := four_digit A B C 6

-- The correct number
def correct_number (A B C : ℕ) := four_digit A B C 9

-- The sum obtained by Yoongi
def yoongi_sum (A B C : ℕ) := incorrect_number A B C + 57

-- The correct sum 
def correct_sum (A B C : ℕ) := correct_number A B C + 57

-- Condition: Yoongi's sum is 1823
axiom yoongi_sum_is_1823 (A B C: ℕ) : yoongi_sum A B C = 1823

-- Proof Problem: Prove that the correct sum is 1826
theorem correct_sum_is_1826 (A B C : ℕ) : correct_sum A B C = 1826 := by
  -- The proof goes here
  sorry

end correct_sum_is_1826_l108_108444


namespace quadratic_inequality_hold_l108_108848

theorem quadratic_inequality_hold (α : ℝ) (h : 0 ≤ α ∧ α ≤ π) :
    (∀ x : ℝ, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) ↔ 
    (α ∈ Set.Icc 0 (π / 6) ∨ α ∈ Set.Icc (5 * π / 6) π) :=
sorry

end quadratic_inequality_hold_l108_108848


namespace range_of_m_l108_108351

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h1 : y1 = (1 + 2 * m) / x1) (h2 : y2 = (1 + 2 * m) / x2)
    (hx : x1 < 0 ∧ 0 < x2) (hy : y1 < y2) : m > -1 / 2 :=
sorry

end range_of_m_l108_108351


namespace largest_volume_sold_in_august_is_21_l108_108475

def volumes : List ℕ := [13, 15, 16, 17, 19, 21]

theorem largest_volume_sold_in_august_is_21
  (sold_volumes_august : List ℕ)
  (sold_volumes_september : List ℕ) :
  sold_volumes_august.length = 3 ∧
  sold_volumes_september.length = 2 ∧
  2 * (sold_volumes_september.sum) = sold_volumes_august.sum ∧
  (sold_volumes_august ++ sold_volumes_september).sum = volumes.sum →
  21 ∈ sold_volumes_august :=
sorry

end largest_volume_sold_in_august_is_21_l108_108475


namespace monthly_salary_l108_108933

variable {S : ℝ}

-- Conditions based on the problem description
def spends_on_food (S : ℝ) : ℝ := 0.40 * S
def spends_on_house_rent (S : ℝ) : ℝ := 0.20 * S
def spends_on_entertainment (S : ℝ) : ℝ := 0.10 * S
def spends_on_conveyance (S : ℝ) : ℝ := 0.10 * S
def savings (S : ℝ) : ℝ := 0.20 * S

-- Given savings
def savings_amount : ℝ := 2500

-- The proof statement for the monthly salary
theorem monthly_salary (h : savings S = savings_amount) : S = 12500 := by
  sorry

end monthly_salary_l108_108933


namespace min_value_frac_l108_108395

variable (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1)

theorem min_value_frac : (1 / a + 4 / b) = 9 :=
by sorry

end min_value_frac_l108_108395


namespace only_n_divides_2_n_minus_1_l108_108654

theorem only_n_divides_2_n_minus_1 :
  ∀ n : ℕ, n ≥ 1 → (n ∣ (2^n - 1)) → n = 1 :=
by
  sorry

end only_n_divides_2_n_minus_1_l108_108654


namespace beth_students_proof_l108_108882

-- Let initial := 150
-- Let joined := 30
-- Let left := 15
-- final := initial + joined - left
-- Prove final = 165

def beth_final_year_students (initial joined left final : ℕ) : Prop :=
  initial = 150 ∧ joined = 30 ∧ left = 15 ∧ final = initial + joined - left

theorem beth_students_proof : ∃ final, beth_final_year_students 150 30 15 final ∧ final = 165 :=
by
  sorry

end beth_students_proof_l108_108882


namespace division_by_ab_plus_one_is_perfect_square_l108_108393

theorem division_by_ab_plus_one_is_perfect_square
    (a b : ℕ) (h : 0 < a ∧ 0 < b)
    (hab : (ab + 1) ∣ (a^2 + b^2)) :
    ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) := 
sorry

end division_by_ab_plus_one_is_perfect_square_l108_108393


namespace probability_age_21_to_30_l108_108407

theorem probability_age_21_to_30 : 
  let total_people := 160 
  let people_10_to_20 := 40
  let people_21_to_30 := 70
  let people_31_to_40 := 30
  let people_41_to_50 := 20
  (people_21_to_30 / total_people : ℚ) = 7 / 16 := by
  sorry

end probability_age_21_to_30_l108_108407


namespace S6_value_l108_108239

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ := x^m + (1/x)^m

theorem S6_value (x : ℝ) (h : x + 1/x = 4) : S_m x 6 = 2700 :=
by
  -- Skipping proof
  sorry

end S6_value_l108_108239


namespace equalize_expenses_l108_108064

variable {x y : ℝ} 

theorem equalize_expenses (h : x > y) : (x + y) / 2 - y = (x - y) / 2 :=
by sorry

end equalize_expenses_l108_108064


namespace find_9b_l108_108375

variable (a b : ℚ)

theorem find_9b (h1 : 7 * a + 3 * b = 0) (h2 : a = b - 4) : 9 * b = 126 / 5 := 
by
  sorry

end find_9b_l108_108375


namespace empty_set_condition_l108_108057

def isEmptySet (s : Set ℝ) : Prop := s = ∅

def A : Set ℕ := {n : ℕ | n^2 ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def C : Set ℝ := {x : ℝ | x^2 + x + 1 = 0}
def D : Set ℝ := {0}

theorem empty_set_condition : isEmptySet C := by
  sorry

end empty_set_condition_l108_108057


namespace find_d_l108_108796

theorem find_d (a b c d : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (hd : 1 < d) 
  (h_eq : ∀ M : ℝ, M ≠ 1 → (M^(1/a)) * (M^(1/(a * b))) * (M^(1/(a * b * c))) * (M^(1/(a * b * c * d))) = M^(17/24)) : d = 8 :=
sorry

end find_d_l108_108796


namespace original_number_is_76_l108_108029

-- Define the original number x and the condition given
def original_number_condition (x : ℝ) : Prop :=
  (3 / 4) * x = x - 19

-- State the theorem that the original number x must be 76 if it satisfies the condition
theorem original_number_is_76 (x : ℝ) (h : original_number_condition x) : x = 76 :=
sorry

end original_number_is_76_l108_108029


namespace max_value_theorem_l108_108503

open Real

noncomputable def max_value (x y : ℝ) : ℝ :=
  x * y * (75 - 5 * x - 3 * y)

theorem max_value_theorem :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y < 75 ∧ max_value x y = 3125 / 3 := by
  sorry

end max_value_theorem_l108_108503


namespace janeth_balloons_count_l108_108881

-- Define the conditions
def bags_round_balloons : Nat := 5
def balloons_per_bag_round : Nat := 20
def bags_long_balloons : Nat := 4
def balloons_per_bag_long : Nat := 30
def burst_round_balloons : Nat := 5

-- Proof statement
theorem janeth_balloons_count:
  let total_round_balloons := bags_round_balloons * balloons_per_bag_round
  let total_long_balloons := bags_long_balloons * balloons_per_bag_long
  let total_balloons := total_round_balloons + total_long_balloons
  total_balloons - burst_round_balloons = 215 :=
by {
  sorry
}

end janeth_balloons_count_l108_108881


namespace object_travel_distance_in_one_hour_l108_108353

/-- If an object travels at 3 feet per second, then it travels 10800 feet in one hour. -/
theorem object_travel_distance_in_one_hour
  (speed : ℕ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ)
  (h_speed : speed = 3)
  (h_seconds_in_minute : seconds_in_minute = 60)
  (h_minutes_in_hour : minutes_in_hour = 60) :
  (speed * (seconds_in_minute * minutes_in_hour) = 10800) :=
by
  sorry

end object_travel_distance_in_one_hour_l108_108353


namespace minimum_questions_to_identify_white_ball_l108_108086

theorem minimum_questions_to_identify_white_ball (n : ℕ) (even_white : ℕ) 
  (h₁ : n = 2004) 
  (h₂ : even_white % 2 = 0) 
  (h₃ : 1 ≤ even_white ∧ even_white ≤ n) :
  ∃ m : ℕ, m = 2003 := 
sorry

end minimum_questions_to_identify_white_ball_l108_108086


namespace sequence_periodicity_l108_108042

theorem sequence_periodicity (a : ℕ → ℕ) (n : ℕ) (h : ∀ k, a k = 6^k) :
  a (n + 5) % 100 = a n % 100 :=
by sorry

end sequence_periodicity_l108_108042


namespace total_accepted_cartons_l108_108502

-- Definitions for the number of cartons delivered and damaged for each customer
def cartons_delivered_first_two : Nat := 300
def cartons_delivered_last_three : Nat := 200

def cartons_damaged_first : Nat := 70
def cartons_damaged_second : Nat := 50
def cartons_damaged_third : Nat := 40
def cartons_damaged_fourth : Nat := 30
def cartons_damaged_fifth : Nat := 20

-- Statement to prove
theorem total_accepted_cartons :
  let accepted_first := cartons_delivered_first_two - cartons_damaged_first
  let accepted_second := cartons_delivered_first_two - cartons_damaged_second
  let accepted_third := cartons_delivered_last_three - cartons_damaged_third
  let accepted_fourth := cartons_delivered_last_three - cartons_damaged_fourth
  let accepted_fifth := cartons_delivered_last_three - cartons_damaged_fifth
  accepted_first + accepted_second + accepted_third + accepted_fourth + accepted_fifth = 990 :=
by
  sorry

end total_accepted_cartons_l108_108502


namespace wall_length_is_7_5_meters_l108_108124

noncomputable def brick_volume : ℚ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℚ := 6000 * brick_volume

noncomputable def wall_cross_section : ℚ := 600 * 22.5

noncomputable def wall_length (total_volume : ℚ) (cross_section : ℚ) : ℚ := total_volume / cross_section

theorem wall_length_is_7_5_meters :
  wall_length total_brick_volume wall_cross_section = 7.5 := by
sorry

end wall_length_is_7_5_meters_l108_108124


namespace evaluate_expression_l108_108060

theorem evaluate_expression : (-1:ℤ)^2022 + |(-2:ℤ)| - (1/2 : ℚ)^0 - 2 * Real.tan (Real.pi / 4) = 0 := 
by
  sorry

end evaluate_expression_l108_108060


namespace average_weight_increase_l108_108865

theorem average_weight_increase 
  (w_old : ℝ) (w_new : ℝ) (n : ℕ) 
  (h1 : w_old = 65) 
  (h2 : w_new = 93) 
  (h3 : n = 8) : 
  (w_new - w_old) / n = 3.5 := 
by 
  sorry

end average_weight_increase_l108_108865


namespace least_positive_t_geometric_progression_l108_108799

open Real

theorem least_positive_t_geometric_progression (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) : 
  ∃ t : ℕ, ∀ t' : ℕ, (t' > 0) → 
  (|arcsin (sin (t' * α)) - 8 * α| = 0) → t = 8 :=
by
  sorry

end least_positive_t_geometric_progression_l108_108799


namespace floor_multiple_of_floor_l108_108844

noncomputable def r : ℝ := sorry

theorem floor_multiple_of_floor (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : ∃ k, n = k * m) (hr : r ≥ 1) 
  (floor_multiple : ∀ (m n : ℕ), (∃ k : ℕ, n = k * m) → ∃ l, ⌊n * r⌋ = l * ⌊m * r⌋) :
  ∃ k : ℤ, r = k := 
sorry

end floor_multiple_of_floor_l108_108844


namespace candy_bar_calories_l108_108719

theorem candy_bar_calories:
  ∀ (calories_per_candy_bar : ℕ) (num_candy_bars : ℕ), 
  calories_per_candy_bar = 3 → 
  num_candy_bars = 5 → 
  calories_per_candy_bar * num_candy_bars = 15 :=
by
  sorry

end candy_bar_calories_l108_108719


namespace ratio_of_packets_to_tent_stakes_l108_108757

-- Definitions based on the conditions provided
def total_items (D T W : ℕ) : Prop := D + T + W = 22
def tent_stakes (T : ℕ) : Prop := T = 4
def bottles_of_water (W T : ℕ) : Prop := W = T + 2

-- The goal is to prove the ratio of packets of drink mix to tent stakes
theorem ratio_of_packets_to_tent_stakes (D T W : ℕ) :
  total_items D T W →
  tent_stakes T →
  bottles_of_water W T →
  D = 3 * T :=
by
  sorry

end ratio_of_packets_to_tent_stakes_l108_108757


namespace derivative_at_one_l108_108509

section

variable {f : ℝ → ℝ}

-- Define the condition
def limit_condition (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (1 + Δx) - f (1 - Δx)) / Δx + 6) < ε

-- State the main theorem
theorem derivative_at_one (h : limit_condition f) : deriv f 1 = -3 :=
by
  sorry

end

end derivative_at_one_l108_108509


namespace taehyung_collected_most_points_l108_108735

def largest_collector : Prop :=
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  taehyung_points > yoongi_points ∧ 
  taehyung_points > jungkook_points ∧ 
  taehyung_points > yuna_points ∧ 
  taehyung_points > yoojung_points

theorem taehyung_collected_most_points : largest_collector :=
by
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  sorry

end taehyung_collected_most_points_l108_108735


namespace geometric_representation_l108_108437

variables (a : ℝ)

-- Definition of the area of the figure
def total_area := a^2 + 1.5 * a

-- Definition of the perimeter of the figure
def total_perimeter := 4 * a + 3

theorem geometric_representation :
  total_area a = a^2 + 1.5 * a ∧ total_perimeter a = 4 * a + 3 :=
by
  exact ⟨rfl, rfl⟩

end geometric_representation_l108_108437


namespace simplify_expression_l108_108851

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l108_108851


namespace four_digit_number_with_divisors_l108_108578

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_minimal_divisor (n p : Nat) : Prop :=
  p > 1 ∧ n % p = 0
  
def is_maximal_divisor (n q : Nat) : Prop :=
  q < n ∧ n % q = 0
  
theorem four_digit_number_with_divisors :
  ∃ (n p : Nat), is_four_digit n ∧ is_minimal_divisor n p ∧ n = 49 * p * p :=
by
  sorry

end four_digit_number_with_divisors_l108_108578


namespace num_natural_a_l108_108325

theorem num_natural_a (a b : ℕ) : 
  (a^2 + a + 100 = b^2) → ∃ n : ℕ, n = 4 := sorry

end num_natural_a_l108_108325


namespace net_change_over_week_l108_108645

-- Definitions of initial quantities on Day 1
def baking_powder_day1 : ℝ := 4
def flour_day1 : ℝ := 12
def sugar_day1 : ℝ := 10
def chocolate_chips_day1 : ℝ := 6

-- Definitions of final quantities on Day 7
def baking_powder_day7 : ℝ := 2.5
def flour_day7 : ℝ := 7
def sugar_day7 : ℝ := 6.5
def chocolate_chips_day7 : ℝ := 3.7

-- Definitions of changes in quantities
def change_baking_powder : ℝ := baking_powder_day1 - baking_powder_day7
def change_flour : ℝ := flour_day1 - flour_day7
def change_sugar : ℝ := sugar_day1 - sugar_day7
def change_chocolate_chips : ℝ := chocolate_chips_day1 - chocolate_chips_day7

-- Statement to prove
theorem net_change_over_week : change_baking_powder + change_flour + change_sugar + change_chocolate_chips = 12.3 :=
by
  -- (Proof omitted)
  sorry

end net_change_over_week_l108_108645


namespace last_digit_of_large_prime_l108_108428

theorem last_digit_of_large_prime : 
  (859433 = 214858 * 4 + 1) → 
  (∃ d, (2 ^ 859433 - 1) % 10 = d ∧ d = 1) :=
by
  intro h
  sorry

end last_digit_of_large_prime_l108_108428


namespace both_true_sufficient_but_not_necessary_for_either_l108_108609

variable (p q : Prop)

theorem both_true_sufficient_but_not_necessary_for_either:
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end both_true_sufficient_but_not_necessary_for_either_l108_108609


namespace perfect_square_divisible_by_12_l108_108582

theorem perfect_square_divisible_by_12 (k : ℤ) : 12 ∣ (k^2 * (k^2 - 1)) :=
by sorry

end perfect_square_divisible_by_12_l108_108582


namespace smallest_a1_value_l108_108873

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 0 then 29 / 98 else if n > 0 then 15 * a_seq (n - 1) - 2 * n else 0

theorem smallest_a1_value :
  (∃ f : ℕ → ℝ, (∀ n > 0, f n = 15 * f (n - 1) - 2 * n) ∧ (∀ n, f n > 0) ∧ (f 1 = 29 / 98)) :=
sorry

end smallest_a1_value_l108_108873


namespace polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l108_108917

theorem polynomial_pattern_1 (a b : ℝ) : (a + b) * (a ^ 2 - a * b + b ^ 2) = a ^ 3 + b ^ 3 :=
sorry

theorem polynomial_pattern_2 (a b : ℝ) : (a - b) * (a ^ 2 + a * b + b ^ 2) = a ^ 3 - b ^ 3 :=
sorry

theorem polynomial_calculation (a b : ℝ) : (a + 2 * b) * (a ^ 2 - 2 * a * b + 4 * b ^ 2) = a ^ 3 + 8 * b ^ 3 :=
sorry

theorem polynomial_factorization (a : ℝ) : a ^ 3 - 8 = (a - 2) * (a ^ 2 + 2 * a + 4) :=
sorry

end polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l108_108917


namespace smallest_k_for_square_l108_108657

theorem smallest_k_for_square : ∃ k : ℕ, (2016 * 2017 * 2018 * 2019 + k) = n^2 ∧ k = 1 :=
by
  use 1
  sorry

end smallest_k_for_square_l108_108657


namespace treasure_coins_l108_108665

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l108_108665


namespace jamshid_takes_less_time_l108_108781

open Real

theorem jamshid_takes_less_time (J : ℝ) (hJ : J < 15) (h_work_rate : (1 / J) + (1 / 15) = 1 / 5) :
  (15 - J) / 15 * 100 = 50 :=
by
  sorry

end jamshid_takes_less_time_l108_108781


namespace find_a_plus_d_l108_108147

theorem find_a_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : c + d = 3) : a + d = -1 := 
by 
  -- omit proof
  sorry

end find_a_plus_d_l108_108147


namespace real_number_a_pure_imaginary_l108_108593

-- Definition of an imaginary number
def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- Given conditions and the proof problem statement
theorem real_number_a_pure_imaginary (a : ℝ) :
  pure_imaginary (⟨(a + 1) / 2, (1 - a) / 2⟩) → a = -1 :=
by
  sorry

end real_number_a_pure_imaginary_l108_108593


namespace sequence_difference_constant_l108_108011

theorem sequence_difference_constant :
  ∀ (x y : ℕ → ℕ), x 1 = 2 → y 1 = 1 →
  (∀ k, k > 1 → x k = 2 * x (k - 1) + 3 * y (k - 1)) →
  (∀ k, k > 1 → y k = x (k - 1) + 2 * y (k - 1)) →
  ∀ k, x k ^ 2 - 3 * y k ^ 2 = 1 :=
by
  -- Insert the proof steps here
  sorry

end sequence_difference_constant_l108_108011


namespace equal_roots_h_l108_108315

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0) ↔ h = 4 := by
  -- proof goes here
  sorry

end equal_roots_h_l108_108315


namespace find_a_perpendicular_lines_l108_108177

theorem find_a_perpendicular_lines (a : ℝ) :
  (∀ (x y : ℝ),
    a * x + 2 * y + 6 = 0 → 
    x + (a - 1) * y + a^2 - 1 = 0 → (a * 1 + 2 * (a - 1) = 0)) → 
  a = 2/3 :=
by
  intros h
  sorry

end find_a_perpendicular_lines_l108_108177


namespace find_a2023_l108_108725

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∃ a1 : ℤ, ∀ n : ℕ, a n = a1 + n * d

theorem find_a2023 (a : ℕ → ℤ) (h_arith : arithmetic_sequence a)
  (h_cond1 : a 2 + a 7 = a 8 + 1)
  (h_cond2 : (a 4)^2 = a 2 * a 8) :
  a 2023 = 2023 := 
sorry

end find_a2023_l108_108725


namespace old_selling_price_l108_108361

theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) :
  C + 0.10 * C = 88 :=
by
  sorry

end old_selling_price_l108_108361


namespace number_of_friends_l108_108751

-- Conditions/Definitions
def total_cost : ℤ := 13500
def cost_per_person : ℤ := 900

-- Prove that Dawson is going with 14 friends.
theorem number_of_friends (h1 : total_cost = 13500) (h2 : cost_per_person = 900) :
  (total_cost / cost_per_person) - 1 = 14 :=
by
  sorry

end number_of_friends_l108_108751


namespace discount_per_bear_l108_108880

/-- Suppose the price of the first bear is $4.00 and Wally pays $354.00 for 101 bears.
 Prove that the discount per bear after the first bear is $0.50. -/
theorem discount_per_bear 
  (price_first : ℝ) (total_bears : ℕ) (total_paid : ℝ) (price_rest_bears : ℝ )
  (h1 : price_first = 4.0) (h2 : total_bears = 101) (h3 : total_paid = 354.0) : 
  (price_first + (total_bears - 1) * price_rest_bears - total_paid) / (total_bears - 1) = 0.50 :=
sorry

end discount_per_bear_l108_108880


namespace max_composite_numbers_l108_108263

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l108_108263


namespace evaluate_at_minus_two_l108_108528

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem evaluate_at_minus_two : f (-2) = -1 := 
by 
  unfold f 
  sorry

end evaluate_at_minus_two_l108_108528


namespace hall_100_guests_67_friends_find_clique_l108_108982

theorem hall_100_guests_67_friends_find_clique :
  ∀ (P : Fin 100 → Fin 100 → Prop) (n : Fin 100),
    (∀ i : Fin 100, ∃ S : Finset (Fin 100), (S.card ≥ 67) ∧ (∀ j ∈ S, P i j)) →
    (∃ (A B C D : Fin 100), P A B ∧ P A C ∧ P A D ∧ P B C ∧ P B D ∧ P C D) :=
by
  sorry

end hall_100_guests_67_friends_find_clique_l108_108982


namespace rate_of_A_is_8_l108_108134

noncomputable def rate_of_A (a b : ℕ) : ℕ :=
  if b = a + 4 ∧ 48 * b = 72 * a then a else 0

theorem rate_of_A_is_8 {a b : ℕ} 
  (h1 : b = a + 4)
  (h2 : 48 * b = 72 * a) : 
  rate_of_A a b = 8 :=
by
  -- proof steps can be added here
  sorry

end rate_of_A_is_8_l108_108134


namespace number_of_true_propositions_eq_2_l108_108357

theorem number_of_true_propositions_eq_2 :
  (¬(∀ (a b : ℝ), a < 0 → b > 0 → a + b < 0)) ∧
  (∀ (α β : ℝ), α = 90 → β = 90 → α = β) ∧
  (∀ (α β : ℝ), α + β = 90 → (∀ (γ : ℝ), γ + α = 90 → β = γ)) ∧
  (¬(∀ (ℓ m n : ℕ), (ℓ ≠ m ∧ ℓ ≠ n ∧ m ≠ n) → (∀ (α β : ℝ), α = β))) →
  2 = 2 :=
by
  sorry

end number_of_true_propositions_eq_2_l108_108357


namespace minimum_balls_to_draw_l108_108470

theorem minimum_balls_to_draw
  (red green yellow blue white : ℕ)
  (h_red : red = 30)
  (h_green : green = 25)
  (h_yellow : yellow = 20)
  (h_blue : blue = 15)
  (h_white : white = 10) :
  ∃ (n : ℕ), n = 81 ∧
    (∀ (r g y b w : ℕ), 
       (r + g + y + b + w >= n) →
       ((r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20) ∧ 
        (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10))
    ) := sorry

end minimum_balls_to_draw_l108_108470


namespace find_f_property_l108_108661

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_property :
  (f 0 = 3) ∧ (∀ x y : ℝ, f (xy) = f ((x^2 + y^2) / 2) + (x - y)^2) →
  (∀ x : ℝ, 0 ≤ x → f x = 3 - 2 * x) :=
by
  intros hypothesis
  -- Proof would be placed here
  sorry

end find_f_property_l108_108661


namespace zero_pow_2014_l108_108592

-- Define the condition that zero raised to any positive power is zero
def zero_pow_pos {n : ℕ} (h : 0 < n) : (0 : ℝ)^n = 0 := by
  sorry

-- Use this definition to prove the specific case of 0 ^ 2014 = 0
theorem zero_pow_2014 : (0 : ℝ)^(2014) = 0 := by
  have h : 0 < 2014 := by decide
  exact zero_pow_pos h

end zero_pow_2014_l108_108592


namespace same_speed_4_l108_108358

theorem same_speed_4 {x : ℝ} (hx : x ≠ -7)
  (H1 : ∀ (x : ℝ), (x^2 - 7*x - 60)/(x + 7) = x - 12) 
  (H2 : ∀ (x : ℝ), x^3 - 5*x^2 - 14*x + 104 = x - 12) :
  ∃ (speed : ℝ), speed = 4 :=
by
  sorry

end same_speed_4_l108_108358


namespace tim_paid_correct_amount_l108_108412

-- Define the conditions given in the problem
def mri_cost : ℝ := 1200
def doctor_hourly_rate : ℝ := 300
def doctor_time_hours : ℝ := 0.5 -- 30 minutes is half an hour
def fee_for_being_seen : ℝ := 150
def insurance_coverage_rate : ℝ := 0.80

-- Total amount Tim paid calculation
def total_cost_before_insurance : ℝ :=
  mri_cost + (doctor_hourly_rate * doctor_time_hours) + fee_for_being_seen

def insurance_coverage : ℝ :=
  total_cost_before_insurance * insurance_coverage_rate

def amount_tim_paid : ℝ :=
  total_cost_before_insurance - insurance_coverage

-- Prove that Tim paid $300
theorem tim_paid_correct_amount : amount_tim_paid = 300 :=
by
  sorry

end tim_paid_correct_amount_l108_108412


namespace solve_inequality_l108_108236

theorem solve_inequality (x : ℝ) : 
  (x / (x^2 + x - 6) ≥ 0) ↔ (x < -3) ∨ (x = 0) ∨ (0 < x ∧ x < 2) :=
by 
  sorry 

end solve_inequality_l108_108236


namespace reverse_addition_unique_l108_108910

theorem reverse_addition_unique (k : ℤ) (h t u : ℕ) (n : ℤ)
  (hk : 100 * h + 10 * t + u = k) 
  (h_k_range : 100 < k ∧ k < 1000)
  (h_reverse_addition : 100 * u + 10 * t + h = k + n)
  (digits_range : 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9) :
  n = 99 :=
sorry

end reverse_addition_unique_l108_108910


namespace school_distance_l108_108858

theorem school_distance (T D : ℝ) (h1 : 5 * (T + 6) = 630) (h2 : 7 * (T - 30) = 630) :
  D = 630 :=
sorry

end school_distance_l108_108858


namespace trig_identity_l108_108446

open Real

theorem trig_identity (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 6) (h : sin α ^ 6 + cos α ^ 6 = 7 / 12) : 1998 * cos α = 333 * Real.sqrt 30 :=
sorry

end trig_identity_l108_108446


namespace mass_percentage_Cl_correct_l108_108975

-- Define the given condition
def mass_percentage_of_Cl := 66.04

-- Statement to prove
theorem mass_percentage_Cl_correct : mass_percentage_of_Cl = 66.04 :=
by
  -- This is where the proof would go, but we use sorry as placeholder.
  sorry

end mass_percentage_Cl_correct_l108_108975


namespace length_of_first_train_is_140_l108_108876

theorem length_of_first_train_is_140 
  (speed1 : ℝ) (speed2 : ℝ) (time_to_cross : ℝ) (length2 : ℝ) 
  (h1 : speed1 = 60) 
  (h2 : speed2 = 40) 
  (h3 : time_to_cross = 12.239020878329734) 
  (h4 : length2 = 200) : 
  ∃ (length1 : ℝ), length1 = 140 := 
by
  sorry

end length_of_first_train_is_140_l108_108876


namespace smallest_N_exists_l108_108515

theorem smallest_N_exists (c1 c2 c3 c4 c5 c6 : ℕ) (N : ℕ) :
  (c1 = 6 * c3 - 2) →
  (N + c2 = 6 * c1 - 5) →
  (2 * N + c3 = 6 * c5 - 2) →
  (3 * N + c4 = 6 * c6 - 2) →
  (4 * N + c5 = 6 * c4 - 1) →
  (5 * N + c6 = 6 * c2 - 5) →
  N = 75 :=
by sorry

end smallest_N_exists_l108_108515


namespace square_projection_exists_l108_108273

structure Point :=
(x y : Real)

structure Line :=
(a b c : Real) -- Line equation ax + by + c = 0

def is_on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

theorem square_projection_exists (P : Point) (l : Line) :
  ∃ (A B C D : Point), 
  is_on_line A l ∧ 
  is_on_line B l ∧
  (A.x + B.x) / 2 = P.x ∧ 
  (A.y + B.y) / 2 = P.y ∧ 
  (A.x = B.x ∨ A.y = B.y) ∧ -- assuming one of the sides lies along the line
  (C.x + D.x) / 2 = P.x ∧ 
  (C.y + D.y) / 2 = P.y ∧ 
  C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B :=
sorry

end square_projection_exists_l108_108273


namespace miles_collection_height_l108_108611

-- Definitions based on conditions
def pages_per_inch_miles : ℕ := 5
def pages_per_inch_daphne : ℕ := 50
def daphne_height_inches : ℕ := 25
def longest_collection_pages : ℕ := 1250

-- Theorem to prove the height of Miles's book collection.
theorem miles_collection_height :
  (longest_collection_pages / pages_per_inch_miles) = 250 := by sorry

end miles_collection_height_l108_108611


namespace total_students_l108_108940

theorem total_students (T : ℝ) 
  (h1 : 0.28 * T = 280) : 
  T = 1000 :=
by {
  sorry
}

end total_students_l108_108940


namespace problem_solution_l108_108487

def p1 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def p2 : Prop := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_solution : (¬ p1) ∨ (¬ p2) :=
by
  sorry

end problem_solution_l108_108487


namespace evaluate_expression_at_x_eq_2_l108_108773

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l108_108773


namespace math_problem_A_B_M_l108_108970

theorem math_problem_A_B_M :
  ∃ M : Set ℝ,
    M = {m | ∃ A B : Set ℝ,
      A = {x | x^2 - 5 * x + 6 = 0} ∧
      B = {x | m * x - 1 = 0} ∧
      A ∩ B = B ∧
      M = {0, (1:ℝ)/2, (1:ℝ)/3}} ∧
    ∃ subsets : Set (Set ℝ),
      subsets = {∅, {0}, {(1:ℝ)/2}, {(1:ℝ)/3}, {0, (1:ℝ)/2}, {(1:ℝ)/2, (1:ℝ)/3}, {0, (1:ℝ)/3}, {0, (1:ℝ)/2, (1:ℝ)/3}} :=
by
  sorry

end math_problem_A_B_M_l108_108970


namespace bake_sale_donation_l108_108235

theorem bake_sale_donation :
  let total_earning := 400
  let cost_of_ingredients := 100
  let donation_homeless_piggy := 10
  let total_donation_homeless := 160
  let donation_homeless := total_donation_homeless - donation_homeless_piggy
  let available_for_donation := total_earning - cost_of_ingredients
  let donation_food_bank := available_for_donation - donation_homeless
  (donation_homeless / donation_food_bank) = 1 := 
by
  sorry

end bake_sale_donation_l108_108235


namespace no_solution_for_equation_l108_108252

theorem no_solution_for_equation (x y z : ℤ) : x^3 + y^3 ≠ 9 * z + 5 := 
by
  sorry

end no_solution_for_equation_l108_108252


namespace value_of_expression_l108_108574

theorem value_of_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : x * y - x = 9 := 
by
  sorry

end value_of_expression_l108_108574


namespace john_bought_metres_l108_108184

-- Define the conditions
def total_cost := 425.50
def cost_per_metre := 46.00

-- State the theorem
theorem john_bought_metres : total_cost / cost_per_metre = 9.25 :=
by
  sorry

end john_bought_metres_l108_108184


namespace damaged_books_l108_108197

theorem damaged_books (O D : ℕ) (h1 : O = 6 * D - 8) (h2 : D + O = 69) : D = 11 :=
by
  sorry

end damaged_books_l108_108197


namespace combined_fractions_value_l108_108569

theorem combined_fractions_value (N : ℝ) (h1 : 0.40 * N = 168) : 
  (1/4) * (1/3) * (2/5) * N = 14 :=
by
  sorry

end combined_fractions_value_l108_108569


namespace problem_statement_l108_108174

theorem problem_statement (p q m n : ℕ) (x : ℚ)
  (h1 : p / q = 4 / 5) (h2 : m / n = 4 / 5) (h3 : x = 1 / 7) :
  x + (2 * q - p + 3 * m - 2 * n) / (2 * q + p - m + n) = 71 / 105 :=
by
  sorry

end problem_statement_l108_108174


namespace distance_focus_asymptote_l108_108220

noncomputable def focus := (Real.sqrt 6 / 2, 0)
def asymptote (x y : ℝ) := x - Real.sqrt 2 * y = 0
def hyperbola (x y : ℝ) := x^2 - 2 * y^2 = 1

theorem distance_focus_asymptote :
  let d := (Real.sqrt 6 / 2, 0)
  let A := 1
  let B := -Real.sqrt 2
  let C := 0
  let numerator := abs (A * d.1 + B * d.2 + C)
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator = Real.sqrt 2 / 2 :=
sorry

end distance_focus_asymptote_l108_108220


namespace no_five_coins_sum_to_43_l108_108536

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem no_five_coins_sum_to_43 :
  ¬ ∃ (a b c d e : ℕ), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧ (a + b + c + d + e = 43) :=
sorry

end no_five_coins_sum_to_43_l108_108536


namespace hyperbola_condition_l108_108783

theorem hyperbola_condition (k : ℝ) : 
  (0 < k ∧ k < 1) → ¬((k > 1 ∨ k < -2) ↔ (0 < k ∧ k < 1)) :=
by
  intro hk
  sorry

end hyperbola_condition_l108_108783


namespace temperature_celsius_range_l108_108760

theorem temperature_celsius_range (C : ℝ) :
  (∀ C : ℝ, let F_approx := 2 * C + 30;
             let F_exact := (9 / 5) * C + 32;
             abs ((2 * C + 30 - ((9 / 5) * C + 32)) / ((9 / 5) * C + 32)) ≤ 0.05) →
  (40 / 29) ≤ C ∧ C ≤ (360 / 11) :=
by
  intros h
  sorry

end temperature_celsius_range_l108_108760


namespace john_total_distance_l108_108161

theorem john_total_distance (speed1 time1 speed2 time2 : ℕ) (distance1 distance2 : ℕ) :
  speed1 = 35 →
  time1 = 2 →
  speed2 = 55 →
  time2 = 3 →
  distance1 = speed1 * time1 →
  distance2 = speed2 * time2 →
  distance1 + distance2 = 235 := by
  intros
  sorry

end john_total_distance_l108_108161


namespace house_spirits_elevator_l108_108789

-- Define the given conditions
def first_floor_domovoi := 1
def middle_floor_domovoi := 2
def last_floor_domovoi := 1
def total_floors := 7
def spirits_per_cycle := first_floor_domovoi + 5 * middle_floor_domovoi + last_floor_domovoi

-- Prove the statement
theorem house_spirits_elevator (n : ℕ) (floor : ℕ) (h1 : total_floors = 7) (h2 : spirits_per_cycle = 12) (h3 : n = 1000) :
  floor = 4 :=
by
  sorry

end house_spirits_elevator_l108_108789


namespace total_bees_is_25_l108_108367

def initial_bees : ℕ := 16
def additional_bees : ℕ := 9

theorem total_bees_is_25 : initial_bees + additional_bees = 25 := by
  sorry

end total_bees_is_25_l108_108367


namespace inequality_proof_equality_condition_l108_108162

theorem inequality_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b)) → a = b :=
sorry

end inequality_proof_equality_condition_l108_108162


namespace divide_two_equal_parts_divide_four_equal_parts_l108_108195

-- the figure is bounded by three semicircles
def figure_bounded_by_semicircles 
-- two have the same radius r1 
(r1 r2 r3 : ℝ) 
-- the third has twice the radius r3 = 2 * r1
(h_eq : r3 = 2 * r1) 
-- Let's denote the figure as F
(F : Type) :=
-- conditions for r1 and r2
r1 > 0 ∧ r2 = r1 ∧ r3 = 2 * r1

-- Prove the figure can be divided into two equal parts.
theorem divide_two_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 : F), H1 ≠ H2 ∧ H1 = H2 :=
sorry

-- Prove the figure can be divided into four equal parts.
theorem divide_four_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 H3 H4 : F), H1 ≠ H2 ∧ H2 ≠ H3 ∧ H3 ≠ H4 ∧ H1 = H2 ∧ H2 = H3 ∧ H3 = H4 :=
sorry

end divide_two_equal_parts_divide_four_equal_parts_l108_108195


namespace arithmetic_sequence_geometric_term_ratio_l108_108219

theorem arithmetic_sequence_geometric_term_ratio (a : ℕ → ℤ) (d : ℤ) (h₀ : d ≠ 0)
  (h₁ : a 1 = a 1)
  (h₂ : a 3 = a 1 + 2 * d)
  (h₃ : a 4 = a 1 + 3 * d)
  (h_geom : (a 1 + 2 * d)^2 = a 1 * (a 1 + 3 * d)) :
  (a 1 + a 5 + a 17) / (a 2 + a 6 + a 18) = 8 / 11 :=
by
  sorry

end arithmetic_sequence_geometric_term_ratio_l108_108219


namespace quadratic_root_exists_l108_108604

theorem quadratic_root_exists (a b c : ℝ) (ha : a ≠ 0)
  (h1 : a * (0.6 : ℝ)^2 + b * 0.6 + c = -0.04)
  (h2 : a * (0.7 : ℝ)^2 + b * 0.7 + c = 0.19) :
  ∃ x : ℝ, 0.6 < x ∧ x < 0.7 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_exists_l108_108604


namespace pq_plus_qr_plus_rp_cubic_1_l108_108051

theorem pq_plus_qr_plus_rp_cubic_1 (p q r : ℝ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + p * r + q * r = -2)
  (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -6 :=
by
  sorry

end pq_plus_qr_plus_rp_cubic_1_l108_108051


namespace price_of_baseball_cards_l108_108752

theorem price_of_baseball_cards 
    (packs_Digimon : ℕ)
    (price_per_pack : ℝ)
    (total_spent : ℝ)
    (total_cost_Digimon : ℝ) 
    (price_baseball_deck : ℝ) 
    (h1 : packs_Digimon = 4) 
    (h2 : price_per_pack = 4.45) 
    (h3 : total_spent = 23.86) 
    (h4 : total_cost_Digimon = packs_Digimon * price_per_pack) 
    (h5 : price_baseball_deck = total_spent - total_cost_Digimon) : 
    price_baseball_deck = 6.06 :=
sorry

end price_of_baseball_cards_l108_108752


namespace total_distance_karl_drove_l108_108640

theorem total_distance_karl_drove :
  ∀ (consumption_rate miles_per_gallon : ℕ) 
    (tank_capacity : ℕ) 
    (initial_gas : ℕ) 
    (distance_leg1 : ℕ) 
    (purchased_gas : ℕ) 
    (remaining_gas : ℕ)
    (final_gas : ℕ),
  consumption_rate = 25 → 
  tank_capacity = 18 →
  initial_gas = 12 →
  distance_leg1 = 250 →
  purchased_gas = 10 →
  remaining_gas = initial_gas - distance_leg1 / consumption_rate + purchased_gas →
  final_gas = remaining_gas - distance_leg2 / consumption_rate →
  remaining_gas - distance_leg2 / consumption_rate = final_gas →
  distance_leg2 = (initial_gas - remaining_gas + purchased_gas - final_gas) * miles_per_gallon →
  miles_per_gallon = 25 →
  distance_leg2 + distance_leg1 = 475 :=
sorry

end total_distance_karl_drove_l108_108640


namespace train_problem_l108_108959

theorem train_problem (Sat M S C : ℕ) 
  (h_boarding_day : true)
  (h_arrival_day : true)
  (h_date_matches_car_on_monday : M = C)
  (h_seat_less_than_car : S < C)
  (h_sat_date_greater_than_car : Sat > C) :
  C = 2 ∧ S = 1 :=
by sorry

end train_problem_l108_108959


namespace office_light_ratio_l108_108254

theorem office_light_ratio (bedroom_light: ℕ) (living_room_factor: ℕ) (total_energy: ℕ) 
  (time: ℕ) (ratio: ℕ) (office_light: ℕ) :
  bedroom_light = 6 →
  living_room_factor = 4 →
  total_energy = 96 →
  time = 2 →
  ratio = 3 →
  total_energy = (bedroom_light * time) + (office_light * time) + ((bedroom_light * living_room_factor) * time) →
  (office_light / bedroom_light) = ratio :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  -- The actual solution steps would go here
  sorry

end office_light_ratio_l108_108254


namespace solve_inequality_l108_108343

noncomputable def rational_inequality_solution (x : ℝ) : Prop :=
  3 - (x^2 - 4 * x - 5) / (3 * x + 2) > 1

theorem solve_inequality (x : ℝ) :
  rational_inequality_solution x ↔ (x > -2 / 3 ∧ x < 9) :=
by
  sorry

end solve_inequality_l108_108343


namespace coupons_per_coloring_book_l108_108178

theorem coupons_per_coloring_book 
  (initial_books : ℝ) (books_sold : ℝ) (coupons_used : ℝ)
  (h1 : initial_books = 40) (h2 : books_sold = 20) (h3 : coupons_used = 80) : 
  (coupons_used / (initial_books - books_sold) = 4) :=
by 
  simp [*, sub_eq_add_neg]
  sorry

end coupons_per_coloring_book_l108_108178


namespace race_distance_between_Sasha_and_Kolya_l108_108581

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l108_108581


namespace cost_of_ticket_when_Matty_was_born_l108_108885

theorem cost_of_ticket_when_Matty_was_born 
    (cost : ℕ → ℕ) 
    (h_halved : ∀ t : ℕ, cost (t + 10) = cost t / 2) 
    (h_age_30 : cost 30 = 125000) : 
    cost 0 = 1000000 := 
by 
  sorry

end cost_of_ticket_when_Matty_was_born_l108_108885


namespace oranges_picked_l108_108599

theorem oranges_picked (total_oranges second_tree third_tree : ℕ) 
    (h1 : total_oranges = 260) 
    (h2 : second_tree = 60) 
    (h3 : third_tree = 120) : 
    total_oranges - (second_tree + third_tree) = 80 := by 
  sorry

end oranges_picked_l108_108599
