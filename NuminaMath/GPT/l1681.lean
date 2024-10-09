import Mathlib

namespace problem_solution_l1681_168121

theorem problem_solution
  (a b c : ℕ)
  (h_pos_a : 0 < a ∧ a ≤ 10)
  (h_pos_b : 0 < b ∧ b ≤ 10)
  (h_pos_c : 0 < c ∧ c ≤ 10)
  (h1 : abc % 11 = 2)
  (h2 : 7 * c % 11 = 3)
  (h3 : 8 * b % 11 = 4 + b % 11) : 
  (a + b + c) % 11 = 0 := 
by
  sorry

end problem_solution_l1681_168121


namespace total_earnings_correct_l1681_168198

section
  -- Define the conditions
  def wage : ℕ := 8
  def hours_Monday : ℕ := 8
  def hours_Tuesday : ℕ := 2

  -- Define the calculation for the total earnings
  def earnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

  -- State the total earnings
  def total_earnings : ℕ := earnings hours_Monday wage + earnings hours_Tuesday wage

  -- Theorem: Prove that Will's total earnings in those two days is $80
  theorem total_earnings_correct : total_earnings = 80 := by
    sorry
end

end total_earnings_correct_l1681_168198


namespace line_plane_intersection_l1681_168161

theorem line_plane_intersection :
  ∃ (x y z : ℝ), (∃ t : ℝ, x = -3 + 2 * t ∧ y = 1 + 3 * t ∧ z = 1 + 5 * t) ∧ (2 * x + 3 * y + 7 * z - 52 = 0) ∧ (x = -1) ∧ (y = 4) ∧ (z = 6) :=
sorry

end line_plane_intersection_l1681_168161


namespace distance_range_l1681_168117

variable (x : ℝ)
variable (starting_fare : ℝ := 6) -- fare in yuan for up to 2 kilometers
variable (surcharge : ℝ := 1) -- yuan surcharge per ride
variable (additional_fare : ℝ := 1) -- fare for every additional 0.5 kilometers
variable (additional_distance : ℝ := 0.5) -- distance in kilometers for every additional fare

theorem distance_range (h_total_fare : 9 = starting_fare + (x - 2) / additional_distance * additional_fare + surcharge) :
  2.5 < x ∧ x ≤ 3 :=
by
  -- Proof goes here
  sorry

end distance_range_l1681_168117


namespace number_of_shelves_l1681_168156

def initial_bears : ℕ := 17
def shipment_bears : ℕ := 10
def bears_per_shelf : ℕ := 9

theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 3 :=
by
  sorry

end number_of_shelves_l1681_168156


namespace circle_center_l1681_168148

theorem circle_center (x y : ℝ) : 
    (∃ x y : ℝ, x^2 - 8*x + y^2 - 4*y = 16) → (x, y) = (4, 2) := by
  sorry

end circle_center_l1681_168148


namespace interval_of_monotonic_decrease_minimum_value_in_interval_l1681_168176

noncomputable def f (x a : ℝ) : ℝ := 1 / x + a * Real.log x

-- Define the derivative of f
noncomputable def f_prime (x a : ℝ) : ℝ := (a * x - 1) / x^2

-- Prove that the interval of monotonic decrease is as specified
theorem interval_of_monotonic_decrease (a : ℝ) :
  if a ≤ 0 then ∀ x ∈ Set.Ioi (0 : ℝ), f_prime x a < 0
  else ∀ x ∈ Set.Ioo 0 (1/a), f_prime x a < 0 := sorry

-- Prove that, given x in [1/2, 1], the minimum value of f(x) is 0 when a = 2 / log 2
theorem minimum_value_in_interval :
  ∃ a : ℝ, (a = 2 / Real.log 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f x a ≥ 0 ∧ (∃ y ∈ Set.Icc (1/2 : ℝ) 1, f y a = 0) := sorry

end interval_of_monotonic_decrease_minimum_value_in_interval_l1681_168176


namespace tan_neg_3900_eq_sqrt3_l1681_168134

theorem tan_neg_3900_eq_sqrt3 : Real.tan (-3900 * Real.pi / 180) = Real.sqrt 3 := by
  -- Definitions of trigonometric values at 60 degrees
  have h_cos : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h_sin : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Using periodicity of the tangent function
  sorry

end tan_neg_3900_eq_sqrt3_l1681_168134


namespace nonoverlapping_unit_squares_in_figure_50_l1681_168127

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem nonoverlapping_unit_squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_50_l1681_168127


namespace parabola_directrix_p_l1681_168160

/-- Given a parabola with equation y^2 = 2px and directrix x = -2, prove that p = 4 -/
theorem parabola_directrix_p (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x) ∧ (∀ x : ℝ, x = -2 → True) → p = 4 :=
by
  sorry

end parabola_directrix_p_l1681_168160


namespace carol_first_six_probability_l1681_168181

theorem carol_first_six_probability :
  let p := 1 / 6
  let q := 5 / 6
  let prob_cycle := q^4
  (p * q^3) / (1 - prob_cycle) = 125 / 671 :=
by
  sorry

end carol_first_six_probability_l1681_168181


namespace radius_of_circle_l1681_168193

theorem radius_of_circle : 
  ∀ (r : ℝ), 3 * (2 * Real.pi * r) = 2 * Real.pi * r ^ 2 → r = 3 :=
by
  intro r
  intro h
  sorry

end radius_of_circle_l1681_168193


namespace ones_digit_of_p_l1681_168138

theorem ones_digit_of_p (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hseq : q = p + 4 ∧ r = p + 8 ∧ s = p + 12) (hpg : p > 5) : (p % 10) = 9 :=
by
  sorry

end ones_digit_of_p_l1681_168138


namespace range_of_a_for_three_zeros_l1681_168126

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l1681_168126


namespace stones_max_value_50_l1681_168139

-- Define the problem conditions in Lean
def value_of_stones (x y z : ℕ) : ℕ := 14 * x + 11 * y + 2 * z

def weight_of_stones (x y z : ℕ) : ℕ := 5 * x + 4 * y + z

def max_value_stones {x y z : ℕ} (h_w : weight_of_stones x y z ≤ 18) (h_x : x ≥ 0) (h_y : y ≥ 0) (h_z : z ≥ 0) : Prop :=
  value_of_stones x y z ≤ 50

theorem stones_max_value_50 : ∃ (x y z : ℕ), weight_of_stones x y z ≤ 18 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ value_of_stones x y z = 50 :=
by
  sorry

end stones_max_value_50_l1681_168139


namespace fraction_problem_l1681_168153

-- Definitions of x and y based on the given conditions
def x : ℚ := 3 / 5
def y : ℚ := 7 / 9

-- The theorem stating the mathematical equivalence to be proven
theorem fraction_problem : (5 * x + 9 * y) / (45 * x * y) = 10 / 21 :=
by
  sorry

end fraction_problem_l1681_168153


namespace minimum_value_l1681_168152

def f (x : ℝ) : ℝ := |x - 4| + |x + 7| + |x - 5|

theorem minimum_value : ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = 4 :=
by
  -- Sorry is used here to skip the proof
  sorry

end minimum_value_l1681_168152


namespace square_of_999_l1681_168158

theorem square_of_999 : 999 * 999 = 998001 := by
  sorry

end square_of_999_l1681_168158


namespace solve_abs_equation_l1681_168141

theorem solve_abs_equation (y : ℝ) (h8 : y < 8) (h_eq : |y - 8| + 2 * y = 12) : y = 4 :=
sorry

end solve_abs_equation_l1681_168141


namespace problem1_problem2_problem3_l1681_168166

-- Given conditions for the sequence
axiom pos_seq {a : ℕ → ℝ} : (∀ n : ℕ, 0 < a n)
axiom relation1 {a : ℕ → ℝ} (t : ℝ) : (∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
axiom relation2 {a : ℕ → ℝ} : 2 * (a 3) = (a 2) + (a 4)

-- Proof Requirements

-- (1) Find the value of (a1 + a3) / a2
theorem problem1 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  (a 1 + a 3) / a 2 = 2 :=
sorry

-- (2) Prove that the sequence is an arithmetic sequence
theorem problem2 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  ∀ n : ℕ, a (n+2) - a (n+1) = a (n+1) - a n :=
sorry

-- (3) Show p and r such that (1/a_k), (1/a_p), (1/a_r) form an arithmetic sequence
theorem problem3 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) (k : ℕ) (hk : k ≠ 0) :
  (k = 1 → ∀ p r : ℕ, ¬((k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p))) ∧ 
  (k ≥ 2 → ∃ p r : ℕ, (k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p) ∧ p = 2 * k - 1 ∧ r = k * (2 * k - 1)) :=
sorry

end problem1_problem2_problem3_l1681_168166


namespace number_of_males_choosing_malt_l1681_168123

-- Definitions of conditions as provided in the problem
def total_males : Nat := 10
def total_females : Nat := 16

def total_cheerleaders : Nat := total_males + total_females

def females_choosing_malt : Nat := 8
def females_choosing_coke : Nat := total_females - females_choosing_malt

noncomputable def cheerleaders_choosing_malt (M_males : Nat) : Nat :=
  females_choosing_malt + M_males

noncomputable def cheerleaders_choosing_coke (M_males : Nat) : Nat :=
  females_choosing_coke + (total_males - M_males)

theorem number_of_males_choosing_malt : ∃ (M_males : Nat), 
  cheerleaders_choosing_malt M_males = 2 * cheerleaders_choosing_coke M_males ∧
  cheerleaders_choosing_malt M_males + cheerleaders_choosing_coke M_males = total_cheerleaders ∧
  M_males = 9 := 
by
  sorry

end number_of_males_choosing_malt_l1681_168123


namespace TV_cost_is_1700_l1681_168146

def hourlyRate : ℝ := 10
def workHoursPerWeek : ℝ := 30
def weeksPerMonth : ℝ := 4
def additionalHours : ℝ := 50

def weeklyEarnings : ℝ := hourlyRate * workHoursPerWeek
def monthlyEarnings : ℝ := weeklyEarnings * weeksPerMonth
def additionalEarnings : ℝ := hourlyRate * additionalHours

def TVCost : ℝ := monthlyEarnings + additionalEarnings

theorem TV_cost_is_1700 : TVCost = 1700 := sorry

end TV_cost_is_1700_l1681_168146


namespace probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l1681_168157

theorem probability_exactly_2_boys_1_girl 
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (Nat.choose boys 2 * Nat.choose girls 1 / (Nat.choose total_group select) : ℚ) = 3 / 5 :=
by sorry

theorem probability_at_least_1_girl
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (1 - (Nat.choose boys select / Nat.choose total_group select : ℚ)) = 4 / 5 :=
by sorry

end probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l1681_168157


namespace find_smallest_x_satisfying_condition_l1681_168111

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l1681_168111


namespace urea_formation_l1681_168188

theorem urea_formation
  (CO2 NH3 Urea : ℕ) 
  (h_CO2 : CO2 = 1)
  (h_NH3 : NH3 = 2) :
  Urea = 1 := by
  sorry

end urea_formation_l1681_168188


namespace mom_chicken_cost_l1681_168143

def cost_bananas : ℝ := 2 * 4 -- bananas cost
def cost_pears : ℝ := 2 -- pears cost
def cost_asparagus : ℝ := 6 -- asparagus cost
def total_expenses_other_than_chicken : ℝ := cost_bananas + cost_pears + cost_asparagus -- total cost of other items
def initial_money : ℝ := 55 -- initial amount of money
def remaining_money_after_other_purchases : ℝ := initial_money - total_expenses_other_than_chicken -- money left after covering other items

theorem mom_chicken_cost : 
  (remaining_money_after_other_purchases - 28 = 11) := 
by
  sorry

end mom_chicken_cost_l1681_168143


namespace average_speed_train_l1681_168124

theorem average_speed_train (x : ℝ) (h1 : x ≠ 0) :
  let t1 := x / 40
  let t2 := 2 * x / 20
  let t3 := 3 * x / 60
  let total_time := t1 + t2 + t3
  let total_distance := 6 * x
  let average_speed := total_distance / total_time
  average_speed = 240 / 7 := by
  sorry

end average_speed_train_l1681_168124


namespace cost_per_book_l1681_168164

theorem cost_per_book (initial_amount : ℤ) (remaining_amount : ℤ) (num_books : ℤ) (cost_per_book : ℤ) :
  initial_amount = 79 →
  remaining_amount = 16 →
  num_books = 9 →
  cost_per_book = (initial_amount - remaining_amount) / num_books →
  cost_per_book = 7 := 
by
  sorry

end cost_per_book_l1681_168164


namespace inequality_proof_l1681_168168

theorem inequality_proof
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 
  64 * a * b * c * d * abs ((a - b) * (b - c) * (c - d) * (d - a)) := 
by
  sorry

end inequality_proof_l1681_168168


namespace prove_intersection_area_is_correct_l1681_168183

noncomputable def octahedron_intersection_area 
  (side_length : ℝ) (cut_height_factor : ℝ) : ℝ :=
  have height_triangular_face := Real.sqrt (side_length^2 - (side_length / 2)^2)
  have plane_height := cut_height_factor * height_triangular_face
  have proportional_height := plane_height / height_triangular_face
  let new_side_length := proportional_height * side_length
  have hexagon_area := (3 * Real.sqrt 3 / 2) * (new_side_length^2) / 2 
  (3 * Real.sqrt 3 / 2) * (new_side_length^2)

theorem prove_intersection_area_is_correct 
  : 
  octahedron_intersection_area 2 (3 / 4) = 9 * Real.sqrt 3 / 8 :=
  sorry 

example : 9 + 3 + 8 = 20 := 
  by rfl

end prove_intersection_area_is_correct_l1681_168183


namespace total_number_of_workers_is_49_l1681_168142

-- Definitions based on the conditions
def avg_salary_all_workers := 8000
def num_technicians := 7
def avg_salary_technicians := 20000
def avg_salary_non_technicians := 6000

-- Prove that the total number of workers in the workshop is 49
theorem total_number_of_workers_is_49 :
  ∃ W, (avg_salary_all_workers * W = avg_salary_technicians * num_technicians + avg_salary_non_technicians * (W - num_technicians)) ∧ W = 49 := 
sorry

end total_number_of_workers_is_49_l1681_168142


namespace slower_pipe_filling_time_l1681_168180

theorem slower_pipe_filling_time
  (t : ℝ)
  (H1 : ∀ (time_slow : ℝ), time_slow = t)
  (H2 : ∀ (time_fast : ℝ), time_fast = t / 3)
  (H3 : 1 / t + 1 / (t / 3) = 1 / 40) :
  t = 160 :=
sorry

end slower_pipe_filling_time_l1681_168180


namespace opposite_of_neg_3_l1681_168147

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l1681_168147


namespace additional_donation_l1681_168178

theorem additional_donation
  (t : ℕ) (c d₁ d₂ T a : ℝ)
  (h1 : t = 25)
  (h2 : c = 2.00)
  (h3 : d₁ = 15.00) 
  (h4 : d₂ = 15.00)
  (h5 : T = 100.00)
  (h6 : t * c + d₁ + d₂ + a = T) :
  a = 20.00 :=
by
  sorry

end additional_donation_l1681_168178


namespace number_of_zeros_l1681_168107

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 - 2 * a * x + a + 1
noncomputable def g (b : ℝ) (x : ℝ) := b * x^3 - 2 * b * x^2 + b * x - 4 / 27

theorem number_of_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (x : ℝ), g b (f a x) = 0 := sorry

end number_of_zeros_l1681_168107


namespace c_work_rate_l1681_168133

theorem c_work_rate {A B C : ℚ} (h1 : A + B = 1/6) (h2 : B + C = 1/8) (h3 : C + A = 1/12) : C = 1/48 :=
by
  sorry

end c_work_rate_l1681_168133


namespace machines_work_together_time_l1681_168175

theorem machines_work_together_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 20) (h2 : rate2 = 1 / 30) :
  (1 / (rate1 + rate2)) = 12 :=
by
  sorry

end machines_work_together_time_l1681_168175


namespace alcohol_quantity_in_mixture_l1681_168184

theorem alcohol_quantity_in_mixture : 
  ∃ (A W : ℕ), (A = 8) ∧ (A * 3 = 4 * W) ∧ (A * 5 = 4 * (W + 4)) :=
by
  sorry -- This is a placeholder; the proof itself is not required.

end alcohol_quantity_in_mixture_l1681_168184


namespace correct_exponentiation_l1681_168194

theorem correct_exponentiation (a : ℝ) : a^5 / a = a^4 := 
  sorry

end correct_exponentiation_l1681_168194


namespace equation_infinitely_many_solutions_l1681_168106

theorem equation_infinitely_many_solutions (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27 / 4 :=
sorry

end equation_infinitely_many_solutions_l1681_168106


namespace green_pen_count_l1681_168135

theorem green_pen_count 
  (blue_pens green_pens : ℕ)
  (h_ratio : blue_pens = 5 * green_pens / 3)
  (h_blue_pens : blue_pens = 20)
  : green_pens = 12 :=
by
  sorry

end green_pen_count_l1681_168135


namespace patrick_age_l1681_168105

theorem patrick_age (r_age_future : ℕ) (years_future : ℕ) (half_age : ℕ → ℕ) 
  (h1 : r_age_future = 30) (h2 : years_future = 2) 
  (h3 : ∀ n, half_age n = n / 2) :
  half_age (r_age_future - years_future) = 14 :=
by
  sorry

end patrick_age_l1681_168105


namespace probability_four_heads_l1681_168100

-- Definitions for use in the conditions
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def biased_coin (h : ℚ) (n k : ℕ) : ℚ :=
  binomial_coefficient n k * (h ^ k) * ((1 - h) ^ (n - k))

-- Condition: probability of getting heads exactly twice is equal to getting heads exactly three times.
def condition (h : ℚ) : Prop :=
  biased_coin h 5 2 = biased_coin h 5 3

-- Theorem to be proven: probability of getting heads exactly four times out of five is 5/32.
theorem probability_four_heads (h : ℚ) (cond : condition h) : biased_coin h 5 4 = 5 / 32 :=
by
  sorry

end probability_four_heads_l1681_168100


namespace kolya_or_leva_wins_l1681_168116

-- Definitions for segment lengths
variables (k l : ℝ)

-- Definition of the condition when Kolya wins
def kolya_wins (k l : ℝ) : Prop :=
  k > l

-- Definition of the condition when Leva wins
def leva_wins (k l : ℝ) : Prop :=
  k ≤ l

-- Theorem statement for the proof problem
theorem kolya_or_leva_wins (k l : ℝ) : kolya_wins k l ∨ leva_wins k l :=
sorry

end kolya_or_leva_wins_l1681_168116


namespace total_glasses_l1681_168132

theorem total_glasses
  (x y : ℕ)
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15) :
  12 * x + 16 * y = 480 :=
by
  sorry

end total_glasses_l1681_168132


namespace jane_total_drying_time_l1681_168162

theorem jane_total_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let nail_art_1 := 8
  let nail_art_2 := 10
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + nail_art_1 + nail_art_2 + top_coat = 49 :=
by 
  sorry

end jane_total_drying_time_l1681_168162


namespace valid_sequences_count_l1681_168186

noncomputable def number_of_valid_sequences
(strings : List (List Nat))
(ball_A_shot : Nat)
(ball_B_shot : Nat) : Nat := 144

theorem valid_sequences_count :
  let strings := [[1, 2], [3, 4, 5], [6, 7, 8, 9]];
  let ball_A := 1;  -- Assuming A is the first ball in the first string
  let ball_B := 3;  -- Assuming B is the first ball in the second string
  ball_A = 1 →
  ball_B = 3 →
  ball_A_shot = 5 →
  ball_B_shot = 6 →
  number_of_valid_sequences strings ball_A_shot ball_B_shot = 144 :=
by
  intros strings ball_A ball_B hA hB hAShot hBShot
  sorry

end valid_sequences_count_l1681_168186


namespace y_coordinates_difference_l1681_168140

theorem y_coordinates_difference {m n k : ℤ}
  (h1 : m = 2 * n + 5)
  (h2 : m + 4 = 2 * (n + k) + 5) :
  k = 2 :=
by
  sorry

end y_coordinates_difference_l1681_168140


namespace percentage_difference_l1681_168177

theorem percentage_difference (water_yesterday : ℕ) (water_two_days_ago : ℕ) (h1 : water_yesterday = 48) (h2 : water_two_days_ago = 50) : 
  (water_two_days_ago - water_yesterday) / water_two_days_ago * 100 = 4 :=
by
  sorry

end percentage_difference_l1681_168177


namespace conic_section_hyperbola_l1681_168154

theorem conic_section_hyperbola (x y : ℝ) : 
  (2 * x - 7)^2 - 4 * (y + 3)^2 = 169 → 
  -- Explain that this equation is of a hyperbola
  true := 
sorry

end conic_section_hyperbola_l1681_168154


namespace smallest_positive_multiple_of_18_with_digits_9_or_0_l1681_168159

noncomputable def m : ℕ := 90
theorem smallest_positive_multiple_of_18_with_digits_9_or_0 : m = 90 ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 9) ∧ m % 18 = 0 → m / 18 = 5 :=
by
  intro h
  sorry

end smallest_positive_multiple_of_18_with_digits_9_or_0_l1681_168159


namespace sqrt_expression_evaluation_l1681_168125

theorem sqrt_expression_evaluation : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end sqrt_expression_evaluation_l1681_168125


namespace simple_interest_rate_l1681_168109

theorem simple_interest_rate :
  ∀ (P R : ℝ), 
  (R * 25 / 100 = 1) → 
  R = 4 := 
by
  intros P R h
  sorry

end simple_interest_rate_l1681_168109


namespace cubic_equation_roots_l1681_168190

theorem cubic_equation_roots (a b c d : ℝ) (h_a : a ≠ 0) 
(h_root1 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
(h_root2 : a * (-3)^3 + b * (-3)^2 - 3 * c + d = 0) :
 (b + c) / a = -13 :=
by sorry

end cubic_equation_roots_l1681_168190


namespace mary_total_nickels_l1681_168137

def mary_initial_nickels : ℕ := 7
def mary_dad_nickels : ℕ := 5

theorem mary_total_nickels : mary_initial_nickels + mary_dad_nickels = 12 := by
  sorry

end mary_total_nickels_l1681_168137


namespace sniper_B_has_greater_chance_of_winning_l1681_168169

def pA (n : ℕ) : ℝ :=
  if n = 1 then 0.4 else if n = 2 then 0.1 else if n = 3 then 0.5 else 0

def pB (n : ℕ) : ℝ :=
  if n = 1 then 0.1 else if n = 2 then 0.6 else if n = 3 then 0.3 else 0

noncomputable def expected_score (p : ℕ → ℝ) : ℝ :=
  (1 * p 1) + (2 * p 2) + (3 * p 3)

theorem sniper_B_has_greater_chance_of_winning :
  expected_score pB > expected_score pA :=
by
  sorry

end sniper_B_has_greater_chance_of_winning_l1681_168169


namespace cindy_hit_section_8_l1681_168182

inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen
deriving DecidableEq

structure DartContest :=
(player : Player)
(score : ℕ)

def ContestConditions (dc : DartContest) : Prop :=
  match dc with
  | ⟨Player.Alice, 10⟩ => True
  | ⟨Player.Ben, 6⟩ => True
  | ⟨Player.Cindy, 9⟩ => True
  | ⟨Player.Dave, 15⟩ => True
  | ⟨Player.Ellen, 19⟩ => True
  | _ => False

def isScoreSection8 (dc : DartContest) : Prop :=
  dc.player = Player.Cindy ∧ dc.score = 8

theorem cindy_hit_section_8 
  (cond : ∀ (dc : DartContest), ContestConditions dc) : 
  ∃ (dc : DartContest), isScoreSection8 dc := by
  sorry

end cindy_hit_section_8_l1681_168182


namespace dalton_needs_more_money_l1681_168102

theorem dalton_needs_more_money :
  let jump_rope_cost := 9
  let board_game_cost := 15
  let playground_ball_cost := 5
  let puzzle_cost := 8
  let saved_allowance := 7
  let uncle_gift := 14
  let total_cost := jump_rope_cost + board_game_cost + playground_ball_cost + puzzle_cost
  let total_money := saved_allowance + uncle_gift
  (total_cost - total_money) = 16 :=
by
  sorry

end dalton_needs_more_money_l1681_168102


namespace unique_solution_l1681_168173

theorem unique_solution (k n : ℕ) (hk : k > 0) (hn : n > 0) (h : (7^k - 3^n) ∣ (k^4 + n^2)) : (k = 2 ∧ n = 4) :=
by
  sorry

end unique_solution_l1681_168173


namespace village_food_sales_l1681_168167

theorem village_food_sales :
  ∀ (customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
      price_per_head_of_lettuce price_per_tomato : ℕ) 
    (H1 : customers_per_month = 500)
    (H2 : heads_of_lettuce_per_person = 2)
    (H3 : tomatoes_per_person = 4)
    (H4 : price_per_head_of_lettuce = 1)
    (H5 : price_per_tomato = 1 / 2), 
  customers_per_month * ((heads_of_lettuce_per_person * price_per_head_of_lettuce) 
    + (tomatoes_per_person * (price_per_tomato : ℝ))) = 2000 := 
by 
  intros customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
         price_per_head_of_lettuce price_per_tomato 
         H1 H2 H3 H4 H5
  sorry

end village_food_sales_l1681_168167


namespace buoy_radius_proof_l1681_168196

/-
We will define the conditions:
- width: 30 cm
- radius_ice_hole: 15 cm (half of width)
- depth: 12 cm
Then prove the radius of the buoy (r) equals 15.375 cm.
-/
noncomputable def radius_of_buoy : ℝ :=
  let width : ℝ := 30
  let depth : ℝ := 12
  let radius_ice_hole : ℝ := width / 2
  let r : ℝ := (369 / 24)
  r    -- the radius of the buoy

theorem buoy_radius_proof : radius_of_buoy = 15.375 :=
by 
  -- We assert that the above definition correctly computes the radius.
  sorry   -- Actual proof omitted

end buoy_radius_proof_l1681_168196


namespace gcd_of_powers_of_two_minus_one_l1681_168179

theorem gcd_of_powers_of_two_minus_one : 
  gcd (2^1015 - 1) (2^1020 - 1) = 1 :=
sorry

end gcd_of_powers_of_two_minus_one_l1681_168179


namespace smallest_whole_number_larger_than_perimeter_l1681_168131

theorem smallest_whole_number_larger_than_perimeter {s : ℝ} (h1 : 16 < s) (h2 : s < 30) :
  61 > 7 + 23 + s :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l1681_168131


namespace digits_of_number_l1681_168145

theorem digits_of_number (d : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9) (h2 : (10 * (50 + d) + 2) % 6 = 0) : (5 * 10 + d) * 10 + 2 = 522 :=
by sorry

end digits_of_number_l1681_168145


namespace roots_cubic_identity_l1681_168130

theorem roots_cubic_identity (p q r s : ℝ) (h1 : r + s = p) (h2 : r * s = -q) (h3 : ∀ x : ℝ, x^2 - p*x - q = 0 → (x = r ∨ x = s)) :
  r^3 + s^3 = p^3 + 3*p*q := by
  sorry

end roots_cubic_identity_l1681_168130


namespace flour_needed_l1681_168189

theorem flour_needed (flour_per_40_cookies : ℝ) (cookies : ℕ) (desired_cookies : ℕ) (flour_needed : ℝ) 
  (h1 : flour_per_40_cookies = 3) (h2 : cookies = 40) (h3 : desired_cookies = 100) :
  flour_needed = 7.5 :=
by
  sorry

end flour_needed_l1681_168189


namespace range_of_c_l1681_168101

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a + b = a * b) (habc : a + b + c = a * b * c) : 1 < c ∧ c ≤ 4 / 3 :=
by
  sorry

end range_of_c_l1681_168101


namespace rate_percent_per_annum_l1681_168113

theorem rate_percent_per_annum (P : ℝ) (SI_increase : ℝ) (T_increase : ℝ) (R : ℝ) 
  (hP : P = 2000) (hSI_increase : SI_increase = 40) (hT_increase : T_increase = 4) 
  (h : SI_increase = P * R * T_increase / 100) : R = 0.5 :=
by  
  sorry

end rate_percent_per_annum_l1681_168113


namespace solve_for_x_l1681_168163

theorem solve_for_x : (∃ x : ℝ, 5 * x + 4 = -6) → x = -2 := 
by
  sorry

end solve_for_x_l1681_168163


namespace lea_notebooks_count_l1681_168192

theorem lea_notebooks_count
  (cost_book : ℕ)
  (cost_binder : ℕ)
  (num_binders : ℕ)
  (cost_notebook : ℕ)
  (total_cost : ℕ)
  (h_book : cost_book = 16)
  (h_binder : cost_binder = 2)
  (h_num_binders : num_binders = 3)
  (h_notebook : cost_notebook = 1)
  (h_total : total_cost = 28) :
  ∃ num_notebooks : ℕ, num_notebooks = 6 ∧
    total_cost = cost_book + num_binders * cost_binder + num_notebooks * cost_notebook := 
by
  sorry

end lea_notebooks_count_l1681_168192


namespace yellow_shirts_count_l1681_168191

theorem yellow_shirts_count (total_shirts blue_shirts green_shirts red_shirts yellow_shirts : ℕ) 
  (h1 : total_shirts = 36) 
  (h2 : blue_shirts = 8) 
  (h3 : green_shirts = 11) 
  (h4 : red_shirts = 6) 
  (h5 : yellow_shirts = total_shirts - (blue_shirts + green_shirts + red_shirts)) :
  yellow_shirts = 11 :=
by
  sorry

end yellow_shirts_count_l1681_168191


namespace smallest_four_digit_2_mod_11_l1681_168174

theorem smallest_four_digit_2_mod_11 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 11 = 2 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 11 = 2 → n ≤ m) := 
by 
  use 1003
  sorry

end smallest_four_digit_2_mod_11_l1681_168174


namespace andrew_spent_total_amount_l1681_168195

/-- Conditions:
1. Andrew played a total of 7 games.
2. Cost distribution for games:
   - 3 games cost $9.00 each
   - 2 games cost $12.50 each
   - 2 games cost $15.00 each
3. Additional expenses:
   - $25.00 on snacks
   - $20.00 on drinks
-/
def total_cost_games : ℝ :=
  (3 * 9) + (2 * 12.5) + (2 * 15)

def cost_snacks : ℝ := 25
def cost_drinks : ℝ := 20

def total_spent (cost_games cost_snacks cost_drinks : ℝ) : ℝ :=
  cost_games + cost_snacks + cost_drinks

theorem andrew_spent_total_amount :
  total_spent total_cost_games 25 20 = 127 := by
  -- The proof is omitted
  sorry

end andrew_spent_total_amount_l1681_168195


namespace maximum_people_shaked_hands_l1681_168151

-- Given conditions
variables (N : ℕ) (hN : N > 4)
def has_not_shaken_hands_with (a b : ℕ) : Prop := sorry -- This should define the shaking hand condition

-- Main statement
theorem maximum_people_shaked_hands (h : ∃ i, has_not_shaken_hands_with i 2) :
  ∃ k, k = N - 3 := 
sorry

end maximum_people_shaked_hands_l1681_168151


namespace monotonic_increase_interval_l1681_168108

noncomputable def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

theorem monotonic_increase_interval 
    (ω : ℝ)
    (hω : 0 < ω)
    (hperiod : Real.pi = 2 * Real.pi / ω) :
    ∀ k : ℤ, ∃ I : Set ℝ, I = interval_of_monotonic_increase k := 
by
  sorry

end monotonic_increase_interval_l1681_168108


namespace division_remainder_l1681_168136

theorem division_remainder : 
  ∃ q r, 1234567 = 123 * q + r ∧ r < 123 ∧ r = 41 := 
by
  sorry

end division_remainder_l1681_168136


namespace line_y_axis_intersection_l1681_168185

-- Conditions: Line contains points (3, 20) and (-9, -6)
def line_contains_points : Prop :=
  ∃ m b : ℚ, ∀ (x y : ℚ), ((x = 3 ∧ y = 20) ∨ (x = -9 ∧ y = -6)) → (y = m * x + b)

-- Question: Prove that the line intersects the y-axis at (0, 27/2)
theorem line_y_axis_intersection :
  line_contains_points → (∃ (y : ℚ), y = 27/2) :=
by
  sorry

end line_y_axis_intersection_l1681_168185


namespace count_special_positive_integers_l1681_168150

theorem count_special_positive_integers : 
  ∃! n : ℕ, n < 10^6 ∧ 
  ∃ a b : ℕ, n = 2 * a^2 ∧ n = 3 * b^3 ∧ 
  ((n = 2592) ∨ (n = 165888)) :=
by
  sorry

end count_special_positive_integers_l1681_168150


namespace find_XY_base10_l1681_168171

theorem find_XY_base10 (X Y : ℕ) (h₁ : Y + 2 = X) (h₂ : X + 5 = 11) : X + Y = 10 := 
by 
  sorry

end find_XY_base10_l1681_168171


namespace algorithm_output_is_127_l1681_168104
-- Import the entire Mathlib library

-- Define the possible values the algorithm can output
def possible_values : List ℕ := [15, 31, 63, 127]

-- Define the property where the value is of the form 2^n - 1
def is_exp2_minus_1 (x : ℕ) := ∃ n : ℕ, x = 2^n - 1

-- Define the main theorem to prove the algorithm's output is 127
theorem algorithm_output_is_127 : (∀ x ∈ possible_values, is_exp2_minus_1 x) →
                                      ∃ n : ℕ, 127 = 2^n - 1 :=
by
  -- Define the conditions and the proof steps are left out
  sorry

end algorithm_output_is_127_l1681_168104


namespace area_of_square_with_perimeter_l1681_168155

def perimeter_of_square (s : ℝ) : ℝ := 4 * s

def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_square_with_perimeter (p : ℝ) (h : perimeter_of_square (3 * p) = 12 * p) : area_of_square (3 * p) = 9 * p^2 := by
  sorry

end area_of_square_with_perimeter_l1681_168155


namespace each_player_plays_36_minutes_l1681_168103

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l1681_168103


namespace jamestown_theme_parks_l1681_168128

theorem jamestown_theme_parks (J : ℕ) (Venice := J + 25) (MarinaDelRay := J + 50) (total := J + Venice + MarinaDelRay) (h : total = 135) : J = 20 :=
by
  -- proof step to be done here
  sorry

end jamestown_theme_parks_l1681_168128


namespace union_complement_correct_l1681_168119

open Set

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem union_complement_correct : A ∪ (compl B) = Ioo (-1 : ℝ) 3 ∪ {3} := by
  sorry

end union_complement_correct_l1681_168119


namespace sin_minus_cos_sqrt_l1681_168199

theorem sin_minus_cos_sqrt (θ : ℝ) (b : ℝ) (h₁ : 0 < θ ∧ θ < π / 2) (h₂ : Real.cos (2 * θ) = b) :
  Real.sin θ - Real.cos θ = Real.sqrt (1 - b) :=
sorry

end sin_minus_cos_sqrt_l1681_168199


namespace shaded_area_of_modified_design_l1681_168187

noncomputable def radius_of_circles (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  (side_length / grid_size) / 2

noncomputable def area_of_circle (radius : ℝ) : ℝ :=
  Real.pi * radius^2

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length^2

noncomputable def shaded_area (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  let r := radius_of_circles side_length grid_size
  let total_circle_area := 9 * area_of_circle r
  area_of_square side_length - total_circle_area

theorem shaded_area_of_modified_design :
  shaded_area 24 3 = (576 - 144 * Real.pi) :=
by
  sorry

end shaded_area_of_modified_design_l1681_168187


namespace rhombus_other_diagonal_l1681_168129

theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
  (h1 : d1 = 50) 
  (h2 : area = 625) 
  (h3 : area = (d1 * d2) / 2) : 
  d2 = 25 :=
by
  sorry

end rhombus_other_diagonal_l1681_168129


namespace mean_marks_second_section_l1681_168122

-- Definitions for the problem conditions
def num_students (section1 section2 section3 section4 : ℕ) : ℕ :=
  section1 + section2 + section3 + section4

def total_marks (section1 section2 section3 section4 : ℕ) (mean1 mean2 mean3 mean4 : ℝ) : ℝ :=
  section1 * mean1 + section2 * mean2 + section3 * mean3 + section4 * mean4

-- The final problem translated into a lean statement
theorem mean_marks_second_section :
  let section1 := 65
  let section2 := 35
  let section3 := 45
  let section4 := 42
  let mean1 := 50
  let mean3 := 55
  let mean4 := 45
  let overall_average := 51.95
  num_students section1 section2 section3 section4 = 187 →
  ((section1 : ℝ) * mean1 + (section2 : ℝ) * M + (section3 : ℝ) * mean3 + (section4 : ℝ) * mean4)
    = 187 * overall_average →
  M = 59.99 :=
by
  intros section1 section2 section3 section4 mean1 mean3 mean4 overall_average Hnum Htotal
  sorry

end mean_marks_second_section_l1681_168122


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l1681_168144

-- Part 1: Expression of the quadratic function
theorem quadratic_function_expression (a : ℝ) (h : a = 0) : 
  ∀ x, (x^2 + (a-2)*x + 3) = x^2 - 2*x + 3 :=
by sorry

-- Part 2: Range of y for 0 < x < 3
theorem quadratic_function_range (x y : ℝ) (h : ∀ x, y = x^2 - 2*x + 3) (hx : 0 < x ∧ x < 3) :
  2 ≤ y ∧ y < 6 :=
by sorry

-- Part 3: Range of m for y1 > y2
theorem quadratic_function_m_range (m y1 y2 : ℝ) (P Q : ℝ × ℝ)
  (h1 : P = (m - 1, y1)) (h2 : Q = (m, y2)) (h3 : y1 > y2) :
  m < 3 / 2 :=
by sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l1681_168144


namespace total_cases_of_candy_correct_l1681_168149

-- Define the number of cases of chocolate bars and lollipops
def cases_of_chocolate_bars : ℕ := 25
def cases_of_lollipops : ℕ := 55

-- Define the total number of cases of candy
def total_cases_of_candy : ℕ := cases_of_chocolate_bars + cases_of_lollipops

-- Prove that the total number of cases of candy is 80
theorem total_cases_of_candy_correct : total_cases_of_candy = 80 := by
  sorry

end total_cases_of_candy_correct_l1681_168149


namespace probability_of_C_l1681_168114

theorem probability_of_C (P_A P_B P_C P_D P_E : ℚ)
  (hA : P_A = 2/5)
  (hB : P_B = 1/5)
  (hCD : P_C = P_D)
  (hE : P_E = 2 * P_C)
  (h_total : P_A + P_B + P_C + P_D + P_E = 1) : P_C = 1/10 :=
by
  -- To prove this theorem, you will use the conditions provided in the hypotheses.
  -- Here's how you start the proof:
  sorry

end probability_of_C_l1681_168114


namespace amy_total_tickets_l1681_168165

def amy_initial_tickets : ℕ := 33
def amy_additional_tickets : ℕ := 21

theorem amy_total_tickets : amy_initial_tickets + amy_additional_tickets = 54 := by
  sorry

end amy_total_tickets_l1681_168165


namespace n_power_of_3_l1681_168197

theorem n_power_of_3 (n : ℕ) (h_prime : Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end n_power_of_3_l1681_168197


namespace fractional_part_wall_in_12_minutes_l1681_168170

-- Definitions based on given conditions
def time_to_paint_wall : ℕ := 60
def time_spent_painting : ℕ := 12

-- The goal is to prove that the fraction of the wall Mark can paint in 12 minutes is 1/5
theorem fractional_part_wall_in_12_minutes (t_pw: ℕ) (t_sp: ℕ) (h1: t_pw = 60) (h2: t_sp = 12) : 
  (t_sp : ℚ) / (t_pw : ℚ) = 1 / 5 :=
by 
  sorry

end fractional_part_wall_in_12_minutes_l1681_168170


namespace perfect_square_trinomial_m_l1681_168110

theorem perfect_square_trinomial_m (m : ℤ) : (∀ x : ℤ, ∃ k : ℤ, x^2 + 2*m*x + 9 = (x + k)^2) ↔ m = 3 ∨ m = -3 :=
by
  sorry

end perfect_square_trinomial_m_l1681_168110


namespace unique_positive_real_solution_l1681_168115

def f (x : ℝ) := x^11 + 5 * x^10 + 20 * x^9 + 1000 * x^8 - 800 * x^7

theorem unique_positive_real_solution :
  ∃! (x : ℝ), 0 < x ∧ f x = 0 :=
sorry

end unique_positive_real_solution_l1681_168115


namespace inequality_holds_iff_m_lt_2_l1681_168118

theorem inequality_holds_iff_m_lt_2 :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → x^2 - m * x + m > 0) ↔ m < 2 :=
by
  sorry

end inequality_holds_iff_m_lt_2_l1681_168118


namespace brad_running_speed_l1681_168120

-- Definitions based on the given conditions
def distance_between_homes : ℝ := 24
def maxwell_walking_speed : ℝ := 4
def maxwell_time_to_meet : ℝ := 3

/-- Brad's running speed is 6 km/h given the conditions of the problem. -/
theorem brad_running_speed : (distance_between_homes - (maxwell_walking_speed * maxwell_time_to_meet)) / (maxwell_time_to_meet - 1) = 6 := by
  sorry

end brad_running_speed_l1681_168120


namespace fraction_one_bedroom_apartments_l1681_168112

theorem fraction_one_bedroom_apartments :
  ∃ x : ℝ, (x + 0.33 = 0.5) ∧ x = 0.17 :=
by
  sorry

end fraction_one_bedroom_apartments_l1681_168112


namespace water_dispenser_capacity_l1681_168172

theorem water_dispenser_capacity :
  ∀ (x : ℝ), (0.25 * x = 60) → x = 240 :=
by
  intros x h
  sorry

end water_dispenser_capacity_l1681_168172
