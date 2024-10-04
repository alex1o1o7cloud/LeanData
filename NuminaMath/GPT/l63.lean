import Mathlib

namespace equation_solution_l63_63513

theorem equation_solution (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2)) : x = 9 :=
by
  sorry

end equation_solution_l63_63513


namespace triangle_angle_ratio_l63_63243

theorem triangle_angle_ratio (a b c : ℝ) (h₁ : a + b + c = 180)
  (h₂ : b = 2 * a) (h₃ : c = 3 * a) : a = 30 ∧ b = 60 ∧ c = 90 :=
by
  sorry

end triangle_angle_ratio_l63_63243


namespace solve_equation_l63_63788

theorem solve_equation 
    (x : ℝ) 
    (hx_floor : ⌊x⌋ = floor x)
    (hx_fractional : {x} = x - floor x)
    (hx_eq : ⌊x⌋ * {x} = 1991 * x) :
    x = 0 ∨ x = -1 / 1992 :=
by
  sorry

end solve_equation_l63_63788


namespace hundredth_odd_integer_l63_63862

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l63_63862


namespace multiply_or_divide_inequality_by_negative_number_l63_63007

theorem multiply_or_divide_inequality_by_negative_number {a b c : ℝ} (h : a < b) (hc : c < 0) :
  c * a > c * b ∧ a / c > b / c :=
sorry

end multiply_or_divide_inequality_by_negative_number_l63_63007


namespace order_DABC_l63_63934

-- Definitions of the variables given in the problem
def A : ℕ := 77^7
def B : ℕ := 7^77
def C : ℕ := 7^7^7
def D : ℕ := Nat.factorial 7

-- The theorem stating the required ascending order
theorem order_DABC : D < A ∧ A < B ∧ B < C :=
by sorry

end order_DABC_l63_63934


namespace count_two_digit_numbers_l63_63952

theorem count_two_digit_numbers (n : ℕ) (h_pos : 1 < n) : 
  let count := ((n - 1) * n) / 2 in
  ∀ (x y : ℕ), 1 ≤ x ∧ x < n ∧ 0 ≤ y ∧ y < n ∧ x + y >= n → count = ((n - 1) * n) / 2 :=
by sorry

end count_two_digit_numbers_l63_63952


namespace extremum_at_one_lambda_range_l63_63207

-- Problem 1: Finding the value of 'a'
theorem extremum_at_one (a : ℝ) : deriv (fun x => a * x^3 + x) 1 = 0 → a = -1/3 :=
by {
  sorry
}

-- Problem 2: Finding the range of 'λ'
theorem lambda_range (λ : ℝ) (p : ℝ := 4) (q : ℝ := 3) :
  (∀ x : ℝ, x ≥ 1 → (x^2 + p * x + q) ≥ (6 + λ) * x - λ * log x + 3) → λ ≤ -1 :=
by {
  sorry
}

end extremum_at_one_lambda_range_l63_63207


namespace probability_one_pair_one_triplet_correct_l63_63108

noncomputable def probability_one_pair_one_triplet : ℚ :=
  let total_outcomes := 6^6 in
  let successful_outcomes := (Nat.choose 6 2) * 2 * (Nat.choose 6 2) * (Nat.choose 4 3) * 4 in
  successful_outcomes / total_outcomes

theorem probability_one_pair_one_triplet_correct :
  probability_one_pair_one_triplet = 25/162 := 
sorry

end probability_one_pair_one_triplet_correct_l63_63108


namespace hiker_total_distance_l63_63884

def hiker_distance (day1_hours day1_speed day2_speed : ℕ) : ℕ :=
  let day2_hours := day1_hours - 1
  let day3_hours := day1_hours
  (day1_hours * day1_speed) + (day2_hours * day2_speed) + (day3_hours * day2_speed)

theorem hiker_total_distance :
  hiker_distance 6 3 4 = 62 := 
by 
  sorry

end hiker_total_distance_l63_63884


namespace find_mans_speed_l63_63451

theorem find_mans_speed (x : ℝ) :
  (∀ w_speed wife_time mans_time : ℝ, w_speed = 50 ∧ wife_time = 2 ∧ mans_time = 2.5 → 2.5 * x = 100) → x = 40 :=
by {
  -- Hypotheses
  intros h,
  -- From the given speeds and times
  apply h,
  -- Given conditions
  exact 50,
  exact 2,
  exact 2.5,
  -- Simplifications
  { simp [div_eq_iff] }
}

end find_mans_speed_l63_63451


namespace max_value_expression_l63_63820

theorem max_value_expression (x y z w : ℕ) (h1 : x ∈ {2, 3, 4, 5}) (h2 : y ∈ {2, 3, 4, 5}) (h3 : z ∈ {2, 3, 4, 5}) (h4 : w ∈ {2, 3, 4, 5}) (h5 : x ≠ y) (h6 : x ≠ z) (h7 : x ≠ w) (h8 : y ≠ z) (h9 : y ≠ w) (h10 : z ≠ w) (h11 : x + y + z + w = 14) :
  xy + yz + zw + wx + 10 ≤ 59 :=
sorry

end max_value_expression_l63_63820


namespace fibonacci_mod_5_50_l63_63105

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

-- The theorem to prove
theorem fibonacci_mod_5_50 : (fibonacci 50) % 5 = 0 :=
sorry

end fibonacci_mod_5_50_l63_63105


namespace probability_first_three_cards_spades_l63_63457

theorem probability_first_three_cards_spades :
  let num_spades : ℕ := 13
  let total_cards : ℕ := 52
  let prob_first_spade : ℚ := num_spades / total_cards
  let prob_second_spade_given_first : ℚ := (num_spades - 1) / (total_cards - 1)
  let prob_third_spade_given_first_two : ℚ := (num_spades - 2) / (total_cards - 2)
  let prob_all_three_spades : ℚ := prob_first_spade * prob_second_spade_given_first * prob_third_spade_given_first_two
  prob_all_three_spades = 33 / 2550 :=
by
  sorry

end probability_first_three_cards_spades_l63_63457


namespace distance_CD_l63_63094

theorem distance_CD (x y : ℝ) : 
  16 * (x - 2)^2 + 4 * y^2 = 64 → 
  (let a := 4 in
   let b := 2 in
   let C := (2, a) in
   let D := (2 + b, 0) in
   dist C D = 2 * Real.sqrt 5) :=
by
  intros h
  let a := 4
  let b := 2
  let C := (2, a)
  let D := (2 + b, 0)
  calc dist C D = 2 * Real.sqrt 5 : sorry

end distance_CD_l63_63094


namespace distinct_students_count_l63_63483

theorem distinct_students_count
  (algebra_students : ℕ)
  (calculus_students : ℕ)
  (statistics_students : ℕ)
  (algebra_statistics_overlap : ℕ)
  (no_other_overlaps : algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32) :
  algebra_students = 13 → calculus_students = 10 → statistics_students = 12 → algebra_statistics_overlap = 3 → 
  algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end distinct_students_count_l63_63483


namespace inverse_value_l63_63196

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value (h : f 2 = 37) : f⁻¹(37) = 2 :=
by
  sorry

end inverse_value_l63_63196


namespace pizza_siblings_order_l63_63555

theorem pizza_siblings_order :
  let slices := 60 in
  let Alex := slices * (1/6) in
  let Beth := slices * (1/4) in
  let Cyril := slices * (1/5) in
  let Dan := slices * (1/3) in
  let Emma := slices - (Alex + Beth + Cyril + Dan) in
  [Dan, Beth, Cyril, Alex, Emma] = [60 * (1/3), 60 * (1/4), 60 * (1/5), 60 * (1/6), 60 - (60 * (1/3) + 60 * (1/4) + 60 * (1/5) + 60 * (1/6))]
  →
  [Dan, Beth, Cyril, Alex, Emma] = [20, 15, 12, 10, 3] :=
by
  sorry

end pizza_siblings_order_l63_63555


namespace pentagon_area_l63_63322

theorem pentagon_area (P Q R S T : Point)
  (h1 : is_square P Q R S)
  (h2 : is_perpendicular P T R)
  (h3 : distance P T = 5)
  (h4 : distance T R = 12) :
  area_of_pentagon P T R S Q = 139 := by
  sorry

end pentagon_area_l63_63322


namespace functional_equation_sol_l63_63421

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_sol (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(y - f(x)) = f(x) - 2 * x + f(f(y))) →
  (∀ x : ℝ, f(x) = x) :=
by
  sorry

end functional_equation_sol_l63_63421


namespace number_of_correct_propositions_l63_63222

theorem number_of_correct_propositions (m n : Line) (α β : Plane) :
  (∀ (m_perp_n : m ⟂ n) (m_perp_alpha : m ⟂ α), (n ∥ α) = false) ∧
  (∀ (m_parallel_alpha : m ∥ α) (alpha_perp_beta : α ⟂ β), (m ⟂ β) = false) ∧
  (∀ (m_perp_n : m ⟂ n) (m_perp_alpha : m ⟂ α) (n_perp_beta : n ⟂ β), (α ⟂ β)) ∧
  (∀ (m_perp_beta : m ⟂ β) (alpha_perp_beta : α ⟂ β), (m ∥ α ∨ m ⊆ α)) →
  correct_propositions = 2 :=
by 
  -- Proof is left as an exercise to the reader.
  sorry

end number_of_correct_propositions_l63_63222


namespace football_team_throwers_l63_63305

theorem football_team_throwers
    (total_players : ℕ)
    (right_handed_players : ℕ)
    (one_third : ℚ)
    (number_throwers : ℕ)
    (number_non_throwers : ℕ)
    (right_handed_non_throwers : ℕ)
    (left_handed_non_throwers : ℕ)
    (h1 : total_players = 70)
    (h2 : right_handed_players = 63)
    (h3 : one_third = 1 / 3)
    (h4 : number_non_throwers = total_players - number_throwers)
    (h5 : right_handed_non_throwers = right_handed_players - number_throwers)
    (h6 : left_handed_non_throwers = one_third * number_non_throwers)
    (h7 : 2 * left_handed_non_throwers = right_handed_non_throwers)
    : number_throwers = 49 := 
by
  sorry

end football_team_throwers_l63_63305


namespace ratio_new_circumference_diameter_l63_63672

variable (r : ℝ)

def new_radius : ℝ := r + 2
def new_diameter : ℝ := 2 * new_radius
def new_circumference : ℝ := 2 * Real.pi * new_radius

theorem ratio_new_circumference_diameter : new_circumference r / new_diameter r = Real.pi := 
by
  rw [new_circumference, new_diameter]
  simp [new_radius]
  sorry

end ratio_new_circumference_diameter_l63_63672


namespace hundredth_odd_integer_l63_63859

theorem hundredth_odd_integer : ∃ (x : ℕ), 2 * x - 1 = 199 ∧ x = 100 :=
by
  use 100
  split
  . exact calc
      2 * 100 - 1 = 200 - 1 : by ring
      _ = 199 : by norm_num
  . refl

end hundredth_odd_integer_l63_63859


namespace bus_price_during_train_service_total_passengers_when_train_stops_l63_63020

-- Define the demand function
def demand (p : ℝ) : ℝ := 3000 - 20 * p

-- Define total cost function for bus transportation
def TC (y : ℝ) : ℝ := y + 5

-- Conditions as constants
def train_fare : ℝ := 10
def train_capacity : ℝ := 1000

-- Problem (a) - What price will the bus company set?
theorem bus_price_during_train_service : 
  ∃ p : ℝ, p = 50.5 ∧ (∀ y, TC(y) = y + 5) ∧ (∀ p, demand(p) = 3000 - 20 * p) ∧ (train_fare = 10) ∧ (train_capacity = 1000) := 
sorry

-- Problem (b) - How will the total number of passengers transported change if the railway station closes?
theorem total_passengers_when_train_stops : 
  ∃ q : ℝ, q = 1490 ∧ (∀ y, TC(y) = y + 5) ∧ (∀ p, demand(p) = 3000 - 20 * p) ∧ (train_fare = 10) ∧ (train_capacity = 1000) :=
sorry

end bus_price_during_train_service_total_passengers_when_train_stops_l63_63020


namespace num_valid_points_l63_63821

-- Define the parabola Q having its focus at (1, 1) and passing through (3, 4) and (-3, -4)
def parabolaQ (p : ℝ × ℝ) : Prop :=
  let focus := (1 : ℝ, 1 : ℝ)
  let point1 := (3 : ℝ, 4 : ℝ)
  let point2 := (-3 : ℝ, -4 : ℝ)
  (p.1 - 3)^2 + (p.2 - 4)^2 = (p.1 + 3)^2 + (p.2 + 4)^2 -- Using the given points and symmetry
  
-- Define the condition |5x + 4y| ≤ 800 for points (x, y) belonging to the parabola
noncomputable def valid_points (p : ℕ × ℕ) : Prop :=
  abs (5 * p.1 + 4 * p.2) ≤ 800

-- Define a function to count the integer coordinate points satisfying the conditions on parabola Q
noncomputable def count_valid_points : ℕ :=
  (finset.range 320*2).filter (λ i, (finset.range 320*2).filter (λ j, parabolaQ (i, j) ∧ valid_points (i, j))).card

-- Prove that the count of valid integer points is 65
theorem num_valid_points : count_valid_points = 65 := 
by {
  sorry
}

end num_valid_points_l63_63821


namespace maya_total_pages_l63_63299

def books_first_week : ℕ := 5
def pages_per_book_first_week : ℕ := 300
def books_second_week := books_first_week * 2
def pages_per_book_second_week : ℕ := 350
def books_third_week := books_first_week * 3
def pages_per_book_third_week : ℕ := 400

def total_pages_first_week : ℕ := books_first_week * pages_per_book_first_week
def total_pages_second_week : ℕ := books_second_week * pages_per_book_second_week
def total_pages_third_week : ℕ := books_third_week * pages_per_book_third_week

def total_pages_maya_read : ℕ := total_pages_first_week + total_pages_second_week + total_pages_third_week

theorem maya_total_pages : total_pages_maya_read = 11000 := by
  sorry

end maya_total_pages_l63_63299


namespace sum_of_coeffs_eq_negative_21_l63_63520

noncomputable def expand_and_sum_coeff (d : ℤ) : ℤ :=
  let expression := -(4 - d) * (d + 2 * (4 - d))
  let expanded_form := -d^2 + 12*d - 32
  let sum_of_coeffs := -1 + 12 - 32
  sum_of_coeffs

theorem sum_of_coeffs_eq_negative_21 (d : ℤ) : expand_and_sum_coeff d = -21 := by
  sorry

end sum_of_coeffs_eq_negative_21_l63_63520


namespace prob_three_blue_is_correct_l63_63900

-- Definitions corresponding to the problem conditions
def total_jellybeans : ℕ := 20
def blue_jellybeans_start : ℕ := 10
def red_jellybeans : ℕ := 10

-- Probabilities calculation steps as definitions
def prob_first_blue : ℚ := blue_jellybeans_start / total_jellybeans
def prob_second_blue_given_first_blue : ℚ := (blue_jellybeans_start - 1) / (total_jellybeans - 1)
def prob_third_blue_given_first_two_blue : ℚ := (blue_jellybeans_start - 2) / (total_jellybeans - 2)

-- Total probability of drawing three blue jellybeans
def prob_three_blue : ℚ := 
  prob_first_blue *
  prob_second_blue_given_first_blue *
  prob_third_blue_given_first_two_blue

-- Formal statement of the proof problem
theorem prob_three_blue_is_correct : prob_three_blue = 2 / 19 :=
by
  -- Fill the proof here
  sorry

end prob_three_blue_is_correct_l63_63900


namespace cos_acute_angle_l63_63188

variable (α : ℝ)
hypothesis h1 : 0 < α ∧ α < π / 2
hypothesis h2 : sin (α - π / 6) = 1 / 3

theorem cos_acute_angle (h1 : 0 < α ∧ α < π / 2) (h2 : sin (α - π / 6) = 1 / 3) :
  cos α = (2 * sqrt 6 - 1) / 6 :=
sorry

end cos_acute_angle_l63_63188


namespace one_cow_one_bag_in_46_days_l63_63252

-- Defining the conditions
def cows_eat_husk (n_cows n_bags n_days : ℕ) := n_cows = n_bags ∧ n_cows = n_days ∧ n_bags = n_days

-- The main theorem to be proved
theorem one_cow_one_bag_in_46_days (h : cows_eat_husk 46 46 46) : 46 = 46 := by
  sorry

end one_cow_one_bag_in_46_days_l63_63252


namespace parabola_focus_standard_equation_l63_63593

theorem parabola_focus_standard_equation :
  ∃ (a b : ℝ), (a = 16 ∧ b = 0) ∨ (a = 0 ∧ b = -8) →
  (∃ (F : ℝ × ℝ), F = (4, 0) ∨ F = (0, -2) ∧ F ∈ {p : ℝ × ℝ | (p.1 - 2 * p.2 - 4 = 0)} →
  (∃ (x y : ℝ), (y^2 = a * x) ∨ (x^2 = b * y))) := sorry

end parabola_focus_standard_equation_l63_63593


namespace dot_product_a_b_l63_63218

-- Define the given vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

-- Define the dot product function
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- State the theorem with the correct answer
theorem dot_product_a_b : dot_product a b = 1 :=
by
  sorry

end dot_product_a_b_l63_63218


namespace area_after_trimming_l63_63461

-- Define the conditions
def original_side_length : ℝ := 22
def trim_x : ℝ := 6
def trim_y : ℝ := 5

-- Calculate dimensions after trimming
def new_length : ℝ := original_side_length - trim_x
def new_width : ℝ := original_side_length - trim_y

-- Define the goal
theorem area_after_trimming : new_length * new_width = 272 := by
  sorry

end area_after_trimming_l63_63461


namespace polynomial_no_real_roots_l63_63766

noncomputable def polynomial_has_no_real_roots (λ μ ν : ℝ) (P : ℝ → ℝ) :=
  (∀ x : ℝ, P x ≠ 0)

theorem polynomial_no_real_roots 
  (λ μ ν : ℝ)
  (h : |λ| + |μ| + |ν| ≤ real.sqrt 2)
  : polynomial_has_no_real_roots λ μ ν (λ x, x^4 + λ * x^3 + μ * x^2 + ν * x + 1) :=
by sorry

end polynomial_no_real_roots_l63_63766


namespace parallelogram_property_not_complementary_l63_63930

theorem parallelogram_property_not_complementary
  (P : Type) [parallelogram P] :
  ¬ (∀ (p : P), opposite_angles_are_complementary p) :=
sorry

end parallelogram_property_not_complementary_l63_63930


namespace JameMade112kProfit_l63_63711

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l63_63711


namespace minimum_distance_exp_ln_l63_63590

-- Definitions of the curves
def y_eq_exp_x (x : ℝ) : ℝ := Real.exp x
def y_eq_ln_x (x : ℝ) : ℝ := Real.log x

-- Definition of the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Statement of the problem as a theorem in Lean 4
theorem minimum_distance_exp_ln : 
  (∃ (P Q : ℝ × ℝ), P.2 = y_eq_exp_x P.1 ∧ Q.2 = y_eq_ln_x Q.1) →
  ∃ PQ_min : ℝ,
    PQ_min = sqrt 2 ∧ 
    ∀ P Q : ℝ × ℝ, (P.2 = y_eq_exp_x P.1 ∧ Q.2 = y_eq_ln_x Q.1) → distance P Q ≥ PQ_min :=
begin
  sorry
end

end minimum_distance_exp_ln_l63_63590


namespace problem_1_problem_2_problem_3_l63_63368

open Real

-- Define the function f
def f (x a : ℝ) : ℝ := cos x ^ 2 + a * sin x + a + 1

-- (I) Minimum value function g(a)
def g (a : ℝ) : ℝ :=
  if a ≥ 0 then 1 else 2 * a + 1

-- (II) Condition for f(x) ≥ 0 ∀x ∈ ℝ
def condition_f_non_negative (a : ℝ) : Prop :=
  ∀ x : ℝ, f x a ≥ 0

-- (III) Given a ∈ [-2, 0], find the range of x
def condition_x_range (x : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * π - π ≤ x ∧ x ≤ 2 * k * π

-- Equivalent mathematical proof problems
theorem problem_1 (a : ℝ) :
  g a = if a ≥ 0 then 1 else 2 * a + 1 :=
sorry

theorem problem_2 (a : ℝ) :
  condition_f_non_negative a → a ≥ -1 / 2 :=
sorry

theorem problem_3 (a : ℝ) (h : -2 ≤ a ∧ a ≤ 0) (x : ℝ) :
  condition_f_non_negative a → condition_x_range x :=
sorry

end problem_1_problem_2_problem_3_l63_63368


namespace range_of_a_l63_63646

theorem range_of_a (a : ℝ) 
  (hf : ∀ x ∈ set.Icc 1 2, deriv (λ x, -x^2 + 2*a*x) ≤ 0)
  (hg : ∀ x ∈ set.Icc 1 2, deriv (λ x, a / x) ≤ 0) :
  0 < a ∧ a ≤ 1 := 
sorry

end range_of_a_l63_63646


namespace intersecting_line_and_hyperbola_l63_63608

theorem intersecting_line_and_hyperbola (a : ℝ)
  (h1 : ∃ A B : ℝ × ℝ, (line_eq : ∀ x y, y = a * x + 1) ∧ (hyperbola_eq : ∀ x y, 3 * x^2 - y^2 = 1) ∧ ∃ x1 x2 : ℝ, x1 + x2 = 2 * a / (3 - a^2) ∧ x1 * x2 = -2 / (3 - a^2) ∧ (a * x1 + 1) * (a * x2 + 1) = 1)
  (h2 : (∃ x1 x2 y1 y2 : ℝ, (line_eq x1 y1) ∧ (line_eq x2 y2) ∧ (hyperbola_eq x1 y1) ∧ (hyperbola_eq x2 y2) ∧ (x1 * x2 + y1 * y2 = 0))) :
  a = 1 ∨ a = -1 := 
sorry

end intersecting_line_and_hyperbola_l63_63608


namespace solution_set_of_inequality_l63_63205

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h₁ : f 2 = 3)
  (h₂ : ∀ x, f' x < 1)
  (h_deriv : ∀ x, derivative f = f')
  : { x : ℝ | f (x^2) < x^2 + 1 } = { x : ℝ | x < -sqrt 2 } ∪ { x : ℝ | x > sqrt 2 } := 
sorry

end solution_set_of_inequality_l63_63205


namespace committee_casey_l63_63440
open Nat

noncomputable def choose (n k : ℕ) : ℕ :=
  n.choose k

theorem committee_casey :
  choose 11 5 = 462 := 
by
  sorry

end committee_casey_l63_63440


namespace num_distinct_exponentiation_values_l63_63468

theorem num_distinct_exponentiation_values : 
  let a := 2
  let b1 := 2
  let b2 := 2
  let b3 := 2
  let standard_value := (a ^ (b1 ^ (b2 ^ b3)))
  let val_1 := (a ^ (a ^ a)) ^ a
  let val_2 := a ^ ((a ^ a) ^ a)
  let val_3 := ((a ^ a) ^ a) ^ a
  let val_4 := (a ^ (a ^ a)) ^ a
  let val_5 := (a ^ a) ^ (a ^ a)
  in 
  (∃ values : Finset ℕ, values.card = 2 ∧
  standard_value ∈ values ∧ 
  (Finset.erase values standard_value).card = 1 ∧ 
  val_1 ∈ values ∧ val_2 ∈ values ∧ val_3 ∈ values ∧ 
  val_4 ∈ values ∧ val_5 ∈ values) :=
by
  let a := 2
  let b1 := 2
  let b2 := 2
  let b3 := 2
  let standard_value := (a ^ (b1 ^ (b2 ^ b3)))
  let val_1 := (a ^ (a ^ a)) ^ a
  let val_2 := a ^ ((a ^ a) ^ a)
  let val_3 := ((a ^ a) ^ a) ^ a
  let val_4 := (a ^ (a ^ a)) ^ a
  let val_5 := (a ^ a) ^ (a ^ a)
  have h : ∃ values : Finset ℕ, values.card = 2 ∧
    standard_value ∈ values ∧ 
    (Finset.erase values standard_value).card = 1 ∧ 
    val_1 ∈ values ∧ val_2 ∈ values ∧ val_3 ∈ values ∧ 
    val_4 ∈ values ∧ val_5 ∈ values := sorry
  exact h

end num_distinct_exponentiation_values_l63_63468


namespace trigonometric_eq_solution_count_l63_63345

theorem trigonometric_eq_solution_count :
  ∃ B : Finset ℤ, B.card = 250 ∧ ∀ x ∈ B, 2000 ≤ x ∧ x ≤ 3000 ∧ 
  2 * Real.sqrt 2 * Real.sin (Real.pi * x / 4)^3 = Real.sin (Real.pi / 4 * (1 + x)) :=
sorry

end trigonometric_eq_solution_count_l63_63345


namespace integral_f_exists_l63_63289

noncomputable def f (x : ℝ) := x / (1 + x^6 * (Real.sin x) ^ 2)

theorem integral_f_exists : ∃ l, ∫ x in 0..∞, f x = l :=
by
  sorry

end integral_f_exists_l63_63289


namespace find_m_value_l63_63731

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)
variables (m : ℝ)
variables (A B C D : V)

-- Assuming vectors a and b are non-collinear
axiom non_collinear (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (∃ (k : ℝ), a = k • b)

-- Given vectors
axiom hAB : B - A = 9 • a + m • b
axiom hBC : C - B = -2 • a - 1 • b
axiom hDC : C - D = a - 2 • b

-- Collinearity condition for A, B, and D
axiom collinear (k : ℝ) : B - A = k • (B - D)

theorem find_m_value : m = -3 :=
by sorry

end find_m_value_l63_63731


namespace centroid_of_triangle_l63_63765

variables {R : Type*} [linear_ordered_field R]

/-- Given points (x1, y1), (x2, y2), and (x3, y3) in a plane, prove that the centroid (x0, y0)
is such that x0 = (x1 + x2 + x3) / 3 and y0 = (y1 + y2 + y3) / 3. -/
theorem centroid_of_triangle (x1 y1 x2 y2 x3 y3 x0 y0 : R) (h : (x0, y0) = ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)) :
  true :=
by {
  sorry
}

end centroid_of_triangle_l63_63765


namespace count_k_with_function_number_of_ks_l63_63720

theorem count_k_with_function (k_bounded: ∀ k : ℕ, k ≤ 2018) :
    (∃ f : ℕ → ℕ, (∀ n : ℕ, f (f n) = 2 * n) ∧ f k = 2018) → sorry := sorry

theorem number_of_ks : {k : ℕ // k ≤ 2018 ∧ (∃ f : ℕ → ℕ, (∀ n : ℕ, f (f n) = 2 * n) ∧ f k = 2018)}.card = 1512 := sorry

end count_k_with_function_number_of_ks_l63_63720


namespace cos_of_right_angle_is_zero_l63_63691

-- Definitions based on the conditions
structure RightTriangle :=
  (D E F : Point)
  (angle_D : ∠D = 90)
  (DE EF : ℝ)
  (DE_length : DE = 9)
  (EF_length : EF = 40)
  (right_angle : angle D = 90)

-- Statement of the proof problem
theorem cos_of_right_angle_is_zero 
  (T : RightTriangle) 
  (h₁ : T.angle_D = 90) 
  (h₂ : T.DE_length = 9) 
  (h₃ : T.EF_length = 40) : 
  cos T.angle_D = 0 := 
  sorry

end cos_of_right_angle_is_zero_l63_63691


namespace calculation_result_l63_63942

theorem calculation_result :
  ([((18^18 / 18^17)^3 * 9^3) / 3^6] = 5832) :=
by
  sorry

end calculation_result_l63_63942


namespace partial_derivatives_l63_63550

noncomputable def F (x y z : ℝ) : ℝ := Real.exp(z^2) - x^2 * y^2 * z^2

theorem partial_derivatives (x y z : ℝ) (h : F x y z = 0) :
  (∂ F ∂ x / ∂ F ∂ z) = z / (x * (z^2 - 1)) ∧
  (∂ F ∂ y / ∂ F ∂ z) = z / (y * (z^2 - 1)) :=
sorry

end partial_derivatives_l63_63550


namespace rectangle_area_l63_63144

theorem rectangle_area (l w : ℚ) (hl : l = 1 / 3) (hw : w = 1 / 5) : 
  l * w = 1 / 15 :=
by {
  rw [hl, hw],
  norm_num,
  sorry
}

end rectangle_area_l63_63144


namespace median_is_twelve_l63_63704

variable (x : ℕ)

def groups : List ℕ := [10, 10, x, 8]

-- Define the median (since we know our provided 'correct' answer should be 12)
def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  (sorted.get! 1 + sorted.get! 2) / 2

theorem median_is_twelve : median (groups x) = 12 := sorry

end median_is_twelve_l63_63704


namespace derivative_f_l63_63809

variable {x : ℝ}

def f (x : ℝ) : ℝ := x * log x

theorem derivative_f (h : x > 0) : deriv f x = log x + 1 := by
  sorry

end derivative_f_l63_63809


namespace percentage_of_girls_in_class_l63_63248

theorem percentage_of_girls_in_class (boys girls total : ℕ) (h_ratio : boys = 3) (h_ratio_g : girls = 4) (h_total : boys + girls = 7) (h_class_size : total = 35) :
  ((girls : ℝ) / (boys + girls : ℝ)) * 100 = 57.14 :=
by 
  have h_fraction := (4 : ℝ) / 7
  have h_percentage := h_fraction * 100
  have : h_percentage = 57.14 := by norm_num
  exact this

end percentage_of_girls_in_class_l63_63248


namespace a3_equals_neg7_l63_63296

-- Definitions based on given conditions
noncomputable def a₁ := -11
noncomputable def d : ℤ := sorry -- this is derived but unknown presently
noncomputable def a(n : ℕ) : ℤ := a₁ + (n - 1) * d

axiom condition : a 4 + a 6 = -6

-- The proof problem statement
theorem a3_equals_neg7 : a 3 = -7 :=
by
  have h₁ : a₁ = -11 := rfl
  have h₂ : a 4 + a 6 = -6 := condition
  sorry

end a3_equals_neg7_l63_63296


namespace function_with_properties_l63_63931

noncomputable def y_sin_abs : ℝ → ℝ := λ x, Real.sin (Real.abs x)
noncomputable def y_cos_abs : ℝ → ℝ := λ x, Real.cos (Real.abs x)
noncomputable def y_abs_cot : ℝ → ℝ := λ x, Real.abs (1 / Real.tan x)
noncomputable def y_log_abs_sin : ℝ → ℝ := λ x, Real.log (Real.abs (Real.sin x))

theorem function_with_properties :
  (∃ (f : ℝ → ℝ), f = y_log_abs_sin ∧
    (∀ x, f (x + Real.pi) = f x) ∧
    (∀ x, f x = f (-x)) ∧
    (∀ x, 0 < x ∧ x < Real.pi / 2 → f x < f (x + Real.pi / 2))) :=
begin
  existsi y_log_abs_sin,
  split,
  { refl },
  split,
  { intros x,
    rw [← Real.log_mul, Real.abs_mul, Real.sin_add_pi, Real.sin_neg, ← Real.abs_neg],
    exact Real.log_one },
  split,
  { intros x,
    rw [Real.log, Real.abs_sin_eq_abs_sin_iff],
    exact congr_arg Real.log (Real.abs_sin_eq_abs_sin_iff.mp rfl) },
  { intros x hx,
    have H := Real.sin_pos_of_pos_of_lt_pi_div_two hx.left hx.right,
    rw Real.log_lt_log_iff (Real.sin_pos_of_pos_of_lt_pi_div_two hx.left hx.right) (Real.sin_pos_of_pos_of_lt_pi_div_two hx.left (by linarith)),
    exact Real.sin_add_pi_div_two_pos hx.left hx.right },
end

end function_with_properties_l63_63931


namespace turtles_on_Happy_Island_l63_63835

theorem turtles_on_Happy_Island (L H : ℕ) (hL : L = 25) (hH : H = 2 * L + 10) : H = 60 :=
by
  sorry

end turtles_on_Happy_Island_l63_63835


namespace expected_value_red_balls_l63_63560

open ProbabilityTheory

-- Definitions of the conditions
def bag : set (set ℕ) := {s | s ⊆ {1, 2, 3, 4} ∧ s.card = 2}
def redBalls : set ℕ := {2, 3, 4}

-- Random variable X: number of red balls drawn
def X (s : set ℕ) : ℕ := (s ∩ redBalls).card

-- Probability measure for uniformly drawing 2 balls from 4
noncomputable def uniformMeasure : measure (set ℕ) := measure.count bag

-- Expected value of X under uniformMeasure
noncomputable def expectedValueX : ℝ := ∫ s in uniformMeasure, X s

-- Lean statement to prove the expected value equals 3/2
theorem expected_value_red_balls :
  expectedValueX = 3 / 2 :=
sorry

end expected_value_red_balls_l63_63560


namespace perpendicular_planes_parallel_l63_63286

variables (m : Line) (α β : Plane)

theorem perpendicular_planes_parallel (h₁ : m ⊥ α) (h₂ : m ⊥ β) (h₃ : α ≠ β) : α ∥ β :=
sorry

end perpendicular_planes_parallel_l63_63286


namespace grasshoppers_no_return_l63_63391

/-
There are 12 grasshoppers sitting at different points on a circle. These points divide the circle into 12 arcs. 
Mark the midpoints of these arcs. Upon signal, the grasshoppers simultaneously jump, each to the nearest marked 
point in the clockwise direction. The jumps to the midpoints of the arcs are repeated. Prove that it is impossible for 
at least one grasshopper to return to its original position after 12 or 13 jumps.
-/

-- Define the conditions
def grasshoppers (positions : Fin 12 → ℝ ) : Prop :=
  ∀ i : Fin 12, 0 ≤ positions i ∧ positions i < 1 ∧ (i < j → positions i < positions j)

def jump (positions : Fin 12 → ℝ ) : Fin 12 → ℝ :=
  λ i, (positions i + 1/24) % 1

def all_jumps (positions : Fin 12 → ℝ) (n : ℕ) : Fin 12 → ℝ :=
  (λ (pos: Fin 12 → ℝ) (i : ℕ), (positions i + i * 1/24) % 1)

-- Define the theorem
theorem grasshoppers_no_return 
  (positions : Fin 12 → ℝ) 
  (h_grasshoppers : grasshoppers positions) 
  (jumps_12 : all_jumps positions 12)
  (jumps_13 : all_jumps positions 13) : 
  ¬(∃ i : Fin 12, jumps_12 i = positions i) ∧ ¬(∃ i : Fin 12, jumps_13 i = positions i) :=
sorry

end grasshoppers_no_return_l63_63391


namespace diagonal_length_of_octagon_l63_63043

theorem diagonal_length_of_octagon 
  (r : ℝ) (s : ℝ) (has_symmetry_axes : ℕ) 
  (inscribed : r = 6) (side_length : s = 5) 
  (symmetry_condition : has_symmetry_axes = 4) : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 40 := 
by 
  sorry

end diagonal_length_of_octagon_l63_63043


namespace smallest_lcm_four_digit_integers_with_gcd_five_l63_63653

open Nat

theorem smallest_lcm_four_digit_integers_with_gcd_five : ∃ k ℓ : ℕ, 1000 ≤ k ∧ k < 10000 ∧ 1000 ≤ ℓ ∧ ℓ < 10000 ∧ gcd k ℓ = 5 ∧ lcm k ℓ = 203010 :=
by
  use 1005
  use 1010
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_lcm_four_digit_integers_with_gcd_five_l63_63653


namespace work_done_l63_63195
  
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else x + 1

theorem work_done :
  ∫ x in 0..1, F x = x^2 + ∫ x in 1..2, F x = x + 1 = (17 : ℚ)/6 :=
by
  let I1 := ∫ x in 0..1, x^2
  let I2 := ∫ x in 1..2, x + 1
  have h1 : I1 = 1/3 :=
    sorry
  have h2 : I2 = 8/3 :=
    sorry
  exact calc
    I1 + I2 = 1/3 + 8/3 := by sorry
    ... = 9/3 := by norm_num
    ... = 3 := by norm_num

#align work_done work_done

end work_done_l63_63195


namespace part1_solution_set_part2_range_a_l63_63203

-- Part (1) Problem
theorem part1_solution_set (x : ℝ) : f x = |2 * x + 1| + |2 * x - 1| → f x ≤ 2 → -1 ≤ -1 ∧ ( -1 ≤ x ∧ x ≤ 1) :=
by
  sorry

-- Part (2) Problem
theorem part2_range_a ( a x : ℝ) (hx : f x ≤ |2*x + 1|) : (∀ x ∈ (set.Icc (1 / 2) 1), f x ≤ |2*x + 1| ) → (x ∈ (set.Icc (1 / 2) 1)) → 0 ≤ a ∧ a ≤ 3 :=
by
  sorry

end part1_solution_set_part2_range_a_l63_63203


namespace cubic_polynomial_solution_l63_63048

theorem cubic_polynomial_solution 
  (p : ℚ → ℚ) 
  (h1 : p 1 = 1)
  (h2 : p 2 = 1 / 4)
  (h3 : p 3 = 1 / 9)
  (h4 : p 4 = 1 / 16)
  (h6 : p 6 = 1 / 36)
  (h0 : p 0 = -1 / 25) : 
  p 5 = 20668 / 216000 :=
sorry

end cubic_polynomial_solution_l63_63048


namespace coeff_x2_in_expansion_l63_63145

-- Conditions about the polynomials involved in the problem
def poly1 : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 4 * X + 5
def poly2 : Polynomial ℤ := 6 * X^3 + 7 * X^2 + 8 * X + 9

-- Statement to prove the coefficient of x^2 in the expansion of (poly1 * poly2) equals 85
theorem coeff_x2_in_expansion :
  (poly1 * poly2).coeff 2 = 85 :=
sorry

end coeff_x2_in_expansion_l63_63145


namespace count_integers_in_range_l63_63635

theorem count_integers_in_range : 
  { n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi }.finite.toFinset.card = 56 := 
by 
  sorry

end count_integers_in_range_l63_63635


namespace max_value_of_M_l63_63758

def is_valid_grid (grid : Fin 5 → Fin 5 → ℕ) : Prop :=
  (∀ i j, grid i j ∈ {1, 2, 3, 4, 5}) ∧
  (∀ j, ∀ i₁ i₂, abs (grid i₁ j - grid i₂ j) ≤ 2)

def min_column_sum (grid : Fin 5 → Fin 5 → ℕ) : ℕ :=
  Finset.univ.min' (λ j, ∑ i, grid i j) (by sorry) -- proof of non-empty set skipped

theorem max_value_of_M : ∃ grid : Fin 5 → Fin 5 → ℕ, 
  is_valid_grid grid ∧ min_column_sum grid = 10 :=
by
  sorry

end max_value_of_M_l63_63758


namespace frog_jumps_within_distance_l63_63912

noncomputable section

open ProbTheory

def frog_jumps_probability (n : ℕ) (radius : ℝ) (jump_length : ℝ) : ℝ :=
  -- Here, we'd define the probability based on random walk theory
  sorry -- actual implementation would go here

theorem frog_jumps_within_distance :
  frog_jumps_probability 4 1.5 1 = 1 / 3 :=
sorry -- proof would go here

end frog_jumps_within_distance_l63_63912


namespace division_sum_l63_63886

theorem division_sum (quotient divisor remainder : ℕ) (hquot : quotient = 65) (hdiv : divisor = 24) (hrem : remainder = 5) : 
  (divisor * quotient + remainder) = 1565 := by 
  sorry

end division_sum_l63_63886


namespace calculate_profit_l63_63710

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l63_63710


namespace max_value_expression_l63_63543

theorem max_value_expression
  (a b c x y z : ℝ)
  (h₁ : 2 ≤ a ∧ a ≤ 3)
  (h₂ : 2 ≤ b ∧ b ≤ 3)
  (h₃ : 2 ≤ c ∧ c ≤ 3)
  (h4 : {x, y, z} = {a, b, c}) :
  (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_value_expression_l63_63543


namespace stratified_sampling_selection_l63_63680

theorem stratified_sampling_selection :
  let num_supermarket := 200
  let num_measurement := 150
  let num_learning_methods := 300
  let num_others := 50
  let total_students := num_supermarket + num_measurement + num_learning_methods + num_others
  let num_to_select := 14

  total_students = 700 → -- Ensuring the total student count matches the conditions
  (num_to_select / total_students : ℚ) = 1 / 50 → -- The probability of selection
  (num_learning_methods * (num_to_select / total_students) : ℝ) = 6 :=
by
  intros
  sorry

end stratified_sampling_selection_l63_63680


namespace product_nonreal_roots_l63_63153

theorem product_nonreal_roots (x : ℂ) :
  (x^4 - 4*x^3 + 6*x^2 - 4*x = 1007) → 
  ((1 + complex.I * complex.sqrt (number.comm_root4 1008)) * (1 - complex.I * complex.sqrt (number.comm_root4 1008)) = 1 + complex.sqrt 1008) := 
by
  intros h_eq
  sorry

end product_nonreal_roots_l63_63153


namespace water_level_after_removal_l63_63295

theorem water_level_after_removal (r : ℝ) :
  let h := (15:ℝ)^(1/3) * r
  -- Conditions
  (container_axial_section_equilateral_triangle : True) 
  (water_poured_in_container : True) 
  (sphere_placed_inside_container : True) 
  (sphere_tangent_to_walls_and_surface : True)
  -- Question == Answer
  h = (15:ℝ)^(1/3) * r :=
by
  -- Add the assumptions directly as part of the theorem
  trivial

end water_level_after_removal_l63_63295


namespace min_x_plus_y_of_positive_l63_63586

open Real

theorem min_x_plus_y_of_positive (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_of_positive_l63_63586


namespace hexagon_area_ratio_l63_63254

def regular_hexagon_area_ratio_zero (ABCDEF : Type) 
  (points_on_sides : (W X Y Z : ABCDEF)) 
  (parallel_lines : (parallel : Prop))
  (first_spacing : ℝ)
  (second_spacing : ℝ)
  (height : ℝ) : Prop :=
  let h := height in
  parallel_lines ∧
  first_spacing = h / 4 ∧
  second_spacing = h / 2 →
  let total_spacing := first_spacing + second_spacing + first_spacing in
  total_spacing = h →
  let x := h - (2 * first_spacing + second_spacing) in
  x = 0 → 
  0 / (height * (3 / 2) * (ABCDEF)) = 0

theorem hexagon_area_ratio : ∀ (ABCDEF : Type) 
  (W X Y Z : ABCDEF) 
  (parallel : Prop)
  (first_spacing second_spacing h : ℝ), 
  regular_hexagon_area_ratio_zero ABCDEF 
    (W, X, Y, Z) 
    parallel 
    first_spacing 
    second_spacing 
    h :=
by 
  intros
  sorry

end hexagon_area_ratio_l63_63254


namespace store_revenue_is_1210_l63_63908

noncomputable def shirt_price : ℕ := 10
noncomputable def jeans_price : ℕ := 2 * shirt_price
noncomputable def jacket_price : ℕ := 3 * jeans_price
noncomputable def discounted_jacket_price : ℕ := jacket_price - (jacket_price / 10)

noncomputable def total_revenue : ℕ :=
  20 * shirt_price + 10 * jeans_price + 15 * discounted_jacket_price

theorem store_revenue_is_1210 :
  total_revenue = 1210 :=
by
  sorry

end store_revenue_is_1210_l63_63908


namespace proper_subsets_count_l63_63819

theorem proper_subsets_count : 
  let s := {x : ℕ | 10 ≤ x ∧ x < 100} in
  (∃ n, s.card = n ∧ 2^n - 1 = 2^90 - 1) :=
by
  let s := {x : ℕ | 10 ≤ x ∧ x < 100}
  have h_card : s.card = 90 := sorry
  use 90
  split
  · exact h_card
  · simp

end proper_subsets_count_l63_63819


namespace students_use_red_color_l63_63035

theorem students_use_red_color
  (total_students : ℕ)
  (students_use_green : ℕ)
  (students_use_both : ℕ)
  (total_students_eq : total_students = 70)
  (students_use_green_eq : students_use_green = 52)
  (students_use_both_eq : students_use_both = 38) :
  ∃ (students_use_red : ℕ), students_use_red = 56 :=
by
  -- We will skip the proof part as specified
  sorry

end students_use_red_color_l63_63035


namespace length_of_chord_intercepted_by_x_axis_l63_63176

open Classical

-- Definitions based on conditions:
def parabolaDirectrix := -1
def centerA := (4 : ℝ, 4 : ℝ)
def radius := (4 - parabolaDirectrix : ℝ)
def circleEquation (x y : ℝ) := (x - 4) ^ 2 + (y - 4) ^ 2 = radius ^ 2

-- Question rephrased as a Lean theorem:
theorem length_of_chord_intercepted_by_x_axis :
  (circleEquation 1 0) ∧ (circleEquation 7 0) → (7 - 1) = 6 := by
  sorry

end length_of_chord_intercepted_by_x_axis_l63_63176


namespace solve_transformed_system_l63_63617

variables {a1 b1 c1 a2 b2 c2 : ℝ}
variables (x y : ℝ)

-- Conditions from the problem
def original_system := a1 * 3 - b1 * 5 = c1 ∧ a2 * 3 + b2 * 5 = c2

-- Transformed system we need to prove the solution for
def transformed_system := a1 * (x - 2) - b1 * y = 3 * c1 ∧ a2 * (x - 2) + b2 * y = 3 * c2

theorem solve_transformed_system : 
  original_system -> transformed_system 11 15 :=
by
  sorry

end solve_transformed_system_l63_63617


namespace football_goals_l63_63072

theorem football_goals :
  (exists A B C : ℕ,
    (A = 3 ∧ B ≠ 1 ∧ (C = 5 ∧ V = 6 ∧ A ≠ 2 ∧ V = 5)) ∨
    (A ≠ 3 ∧ B = 1 ∧ (C ≠ 5 ∧ V = 6 ∧ A = 2 ∧ V ≠ 5))) →
  A + B + C ≠ 10 :=
by {
  sorry
}

end football_goals_l63_63072


namespace ratio_bud_to_uncle_l63_63487

theorem ratio_bud_to_uncle (bud_age uncle_age : ℕ) (hab : bud_age = 8) (hau : uncle_age = 24) :
  (bud_age : ℚ) / uncle_age = 1 / 3 :=
by
  rw [hab, hau]
  norm_num
  sorry

end ratio_bud_to_uncle_l63_63487


namespace determine_n_l63_63508

-- Define the necessary predicates
def no_prime_cube_divides (x : ℕ) : Prop := 
  ∀ p : ℕ, Prime p → ¬ p^3 ∣ x

-- Define the main proposition
def satisfies_condition (n a b : ℕ) : Prop :=
  (no_prime_cube_divides (a^2 + b + 3)) ∧ 
  (a * b + 3 * b + 8 = n * (a^2 + b + 3))

-- The main theorem to prove
theorem determine_n : ∀ n : ℕ, n ≥ 1 → 
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ satisfies_condition n a b) → 
  n = 2 := 
begin
  sorry
end

end determine_n_l63_63508


namespace roots_quadratic_identity_l63_63661

theorem roots_quadratic_identity (p q : ℝ) (r s : ℝ) (h1 : r + s = 3 * p) (h2 : r * s = 2 * q) :
  r^2 + s^2 = 9 * p^2 - 4 * q := 
by 
  sorry

end roots_quadratic_identity_l63_63661


namespace mother_to_grandfather_age_ratio_l63_63768

theorem mother_to_grandfather_age_ratio
  (rachel_age : ℕ)
  (grandfather_ratio : ℕ)
  (father_mother_gap : ℕ) 
  (future_rachel_age: ℕ) 
  (future_father_age : ℕ)
  (current_father_age current_mother_age current_grandfather_age : ℕ) 
  (h1 : rachel_age = 12)
  (h2 : grandfather_ratio = 7)
  (h3 : father_mother_gap = 5)
  (h4 : future_rachel_age = 25)
  (h5 : future_father_age = 60)
  (h6 : current_father_age = future_father_age - (future_rachel_age - rachel_age))
  (h7 : current_mother_age = current_father_age - father_mother_gap)
  (h8 : current_grandfather_age = grandfather_ratio * rachel_age) :
  current_mother_age = current_grandfather_age / 2 :=
by
  sorry

end mother_to_grandfather_age_ratio_l63_63768


namespace compound_interest_correct_l63_63244

theorem compound_interest_correct:
  ∃ (Principal : ℝ), 
    (56 = (Principal * 5 * 2) / 100) → 
    (560 × (1 + 5 / 100)^2 - 560 = 57.4) :=
by
  sorry

end compound_interest_correct_l63_63244


namespace d_not_unique_minimum_l63_63175

noncomputable def d (n : ℕ) (x : Fin n → ℝ) (t : ℝ) : ℝ :=
  (Finset.min' (Finset.univ.image (λ i => abs (x i - t))) sorry + 
  Finset.max' (Finset.univ.image (λ i => abs (x i - t))) sorry) / 2

theorem d_not_unique_minimum (n : ℕ) (x : Fin n → ℝ) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ d n x t1 = d n x t2 := sorry

end d_not_unique_minimum_l63_63175


namespace symmetric_slope_angle_l63_63216

-- Define the problem conditions in Lean
def slope_angle (θ : Real) : Prop :=
  0 ≤ θ ∧ θ < Real.pi

-- Statement of the theorem in Lean
theorem symmetric_slope_angle (θ : Real) (h : slope_angle θ) :
  θ = 0 ∨ θ = Real.pi - θ :=
sorry

end symmetric_slope_angle_l63_63216


namespace rose_needs_more_money_l63_63328

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end rose_needs_more_money_l63_63328


namespace triangle_obtuse_l63_63380

theorem triangle_obtuse (a b c : ℝ) (ha : a = 12 * S) (hb : b = 20 * S) (hc : c = 28 * S) : 
  (cos ((a^2 + b^2 - c^2) / (2 * a * b)) < 0) :=
begin
 sorry
end

end triangle_obtuse_l63_63380


namespace planes_divide_number_of_parts_l63_63703

-- Define the conditions and the final recursive formula for K(n)
def planes_divide_space (n : ℕ) : ℕ :=
  (n^3 + 5*n + 6) / 6

-- Lean theorem stating the problem: proving the number of parts n planes divide the space into.
theorem planes_divide (n : ℕ) : ℕ :=
  planes_divide_space n = (n^3 + 5*n + 6) / 6

-- Proof not provided, proof placeholder
theorem number_of_parts (n : ℕ) : planes_divide_space n = (n^3 + 5*n + 6) / 6 := 
  sorry

end planes_divide_number_of_parts_l63_63703


namespace nomogram_relation_l63_63485

noncomputable def root_of_eq (x p q : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem nomogram_relation (x p q : ℝ) (hx : root_of_eq x p q) : 
  q = -x * p - x^2 :=
by 
  sorry

end nomogram_relation_l63_63485


namespace range_of_x0_l63_63462

def isTangent (M N : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) : Prop :=
  ((M.1 - O.1) * (N.1 - O.1) + (M.2 - O.2) * (N.2 - O.2) = 0)
  ∧ (N.1 - O.1)^2 + (N.2 - O.2)^2 = r^2

theorem range_of_x0 
  (x0 : ℝ) 
  (M : ℝ × ℝ := (x0, real.sqrt 3))
  (O : ℝ × ℝ := (0, 0)) 
  (r : ℝ := 1) 
  (N : ℝ × ℝ) 
  (h_tangent : isTangent M N O r)
  (h_angle : ∠ O M N ≥ real.pi / 6) :
  -1 ≤ x0 ∧ x0 ≤ 1 :=
by
  sorry

end range_of_x0_l63_63462


namespace max_elements_in_set_l63_63919

theorem max_elements_in_set (S : Finset ℕ) (h1 : 1 ∈ S) (h2 : 2500 ∈ S) (h_distinct : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → x ≠ y) :
  ∀ x ∈ S, ((Finset.sum S) - x) % (Finset.card S - 1) = 0 → Finset.card S ≤ 4 :=
by
  sorry

end max_elements_in_set_l63_63919


namespace trigonometric_eq_solution_count_l63_63344

theorem trigonometric_eq_solution_count :
  ∃ B : Finset ℤ, B.card = 250 ∧ ∀ x ∈ B, 2000 ≤ x ∧ x ≤ 3000 ∧ 
  2 * Real.sqrt 2 * Real.sin (Real.pi * x / 4)^3 = Real.sin (Real.pi / 4 * (1 + x)) :=
sorry

end trigonometric_eq_solution_count_l63_63344


namespace replaced_weight_calc_l63_63357

theorem replaced_weight_calc 
  (eight_persons_avg_increase_by : ℝ)
  (new_person_weight : ℝ)
  (increase_per_person : ℝ)
  (total_increase : ℝ) : 
  eight_persons_avg_increase_by = 3.5 → 
  new_person_weight = 93 → 
  increase_per_person = 8 * 3.5 → 
  total_increase = increase_per_person → 
  new_person_weight - total_increase = 65 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end replaced_weight_calc_l63_63357


namespace line_does_not_pass_first_quadrant_l63_63595

open Real

theorem line_does_not_pass_first_quadrant (a b : ℝ) (h₁ : a > 0) (h₂ : b < 0) : 
  ¬∃ x y : ℝ, (x > 0) ∧ (y > 0) ∧ (ax + y - b = 0) :=
sorry

end line_does_not_pass_first_quadrant_l63_63595


namespace min_like_both_l63_63686

-- Declare the total number of people surveyed
def total_people : ℕ := 150

-- Declare the number of people who like tea
def like_tea : ℕ := 120

-- Declare the number of people who like coffee
def like_coffee : ℕ := 100

-- The statement proving the minimum number who like both, given the conditions
theorem min_like_both (total_people like_tea like_coffee : ℕ) : 
  total_people = 150 → like_tea = 120 → like_coffee = 100 → 
  ∃ (both_like : ℕ), both_like = 70 ∧
  (like_tea - (total_people - like_coffee) = both_like) := 
by
  intros h_total h_tea h_coffee
  use 70
  rw [h_total, h_tea, h_coffee]
  simp
  sorry

end min_like_both_l63_63686


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l63_63878

theorem option_A_incorrect (a : ℝ) : (a^2) * (a^3) ≠ a^6 :=
by sorry

theorem option_B_incorrect (a : ℝ) : (a^2)^3 ≠ a^5 :=
by sorry

theorem option_C_incorrect (a : ℝ) : (a^6) / (a^2) ≠ a^3 :=
by sorry

theorem option_D_correct (a b : ℝ) : (a + 2 * b) * (a - 2 * b) = a^2 - 4 * b^2 :=
by sorry

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l63_63878


namespace red_marbles_count_l63_63039

theorem red_marbles_count (R : ℕ) (h1 : 48 - R > 0) (h2 : ((48 - R) / 48 : ℚ) * ((48 - R) / 48) = 9 / 16) : R = 12 :=
sorry

end red_marbles_count_l63_63039


namespace altitude_leg_ratio_in_isosceles_triangle_l63_63689

theorem altitude_leg_ratio_in_isosceles_triangle (a b : ℝ) (h : 0 < a) 
  (h_isosceles : ∀ (x y : ℝ), x = a → y = a → (x + y) / (b / 2) = 150) : 
  (altitude (triangle a a b) / a = 1 / 2) := sorry

end altitude_leg_ratio_in_isosceles_triangle_l63_63689


namespace base8_to_base10_correct_l63_63096

def base8_to_base10_conversion : Prop :=
  (2 * 8^2 + 4 * 8^1 + 6 * 8^0 = 166)

theorem base8_to_base10_correct : base8_to_base10_conversion :=
by
  sorry

end base8_to_base10_correct_l63_63096


namespace area_of_triangle_l63_63336

theorem area_of_triangle 
  (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ) 
  (h1 : a1 * b2 - a2 * b1 ≠ 0)
  (h2 : a2 * b3 - a3 * b2 ≠ 0)
  (h3 : a3 * b1 - a1 * b3 ≠ 0) :
  let Δ := (a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2))
  in 1/2 * |Δ^2| = 1 / 2 * (Δ ^ 2) /  |(a2 * b3 - a3 * b2) * (a3 * b1 - a1 * b3) * (a1 * b2 - a2 * b1)| :=
sorry

end area_of_triangle_l63_63336


namespace num_possible_medians_of_S_l63_63282

def S : Set ℤ := {a | a ∈ {5, 7, 8, 13, 16, 20}}

def is_distinct (s : Set ℤ) : Prop := s.card = 11 ∧ ∀ (x y : ℤ), x ∈ s → y ∈ s → x ≠ y → x ≠ y

def sum_greater_than_24 (s : Set ℤ) : Prop := 
  ∃ (a b c : ℤ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a < b ∧ b < c ∧ a + b + c > 24)

def median_values (s : Set ℤ) : Set ℤ := 
  (let sorted_s := (list.of_list s).sort (≤)
  in {x | ∃ m n, sorted_s = m ++ [x] ++ n})

theorem num_possible_medians_of_S : 
  ∀ (s : Set ℤ), is_distinct s ∧ sum_greater_than_24 s → (median_values s).card = 6 := 
sorry

end num_possible_medians_of_S_l63_63282


namespace cot_neg_45_l63_63134

-- Define the conditions
lemma cot_def (x : ℝ) : Real.cot x = 1 / Real.tan x := sorry
lemma tan_neg (x : ℝ) : Real.tan (-x) = -Real.tan x := sorry
lemma tan_45 : Real.tan (Real.pi / 4) = 1 := sorry

-- State the theorem to prove
theorem cot_neg_45 : Real.cot (-Real.pi / 4) = -1 :=
by
  apply cot_def
  apply tan_neg
  apply tan_45
  sorry

end cot_neg_45_l63_63134


namespace find_a_maximize_profit_l63_63435

noncomputable def sales_volume (a x : ℝ) : ℝ :=
  a / (x - 5) + 10 * (x - 8) ^ 2

def cost_per_kg : ℝ := 5

noncomputable def profit (a x : ℝ) : ℝ :=
  (x - cost_per_kg) * sales_volume a x

theorem find_a (a : ℝ) : sales_volume a 7 = 11 → a = 2 := 
by {
  intros h,
  sorry
}

theorem maximize_profit (a : ℝ) (h : a = 2) : 
  ∀ x : ℝ, 5 < x ∧ x < 8 → 
  (∀ y : ℝ, 5 < y ∧ y < 8 → profit a y ≤ profit a 6) :=
by {
  intros x h1 y h2,
  sorry
}

end find_a_maximize_profit_l63_63435


namespace hundredth_odd_positive_integer_l63_63867

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l63_63867


namespace AD_length_l63_63799

variables {A B C D : Type} [metric_space ↥Real]  -- Triangle vertices and point D on BC
variables [AffineSpace.Real A] [AffineSpace.Real B] [AffineSpace.Real C] [AffineSpace.Real D]
variable (AD : AffineLine A)  -- AD is the angle bisector of ∠BAC
variable (l : AffineLine ↥Real)  -- line ℓ through A perpendicular to AD
variable (B_dist_to_l : ℝ)  -- Distance from B to ℓ
variable (C_dist_to_l : ℝ)  -- Distance from C to ℓ

-- Conditions as defined
axiom D_on_BC : B ∈ AffineLine ↥Real ∧ C ∈ AffineLine ↥Real  -- D lies on BC
axiom AD_bisects_BAC : AD.angle_at A == B.angle_at D + D.angle_at C  -- AD is the angle bisector
axiom l_perpendicular_to_AD : ∀ A ∈ l, (AD.origin == A) → AffineLine ℝ  -- Line ℓ is perpendicular to AD
axiom distances : (B_dist_to_l = 5) ∧ (C_dist_to_l = 6)  -- Distances are 5 and 6 respectively

theorem AD_length : AD.length = 60/11 :=
by
  ... sorry  -- Proof not required

end AD_length_l63_63799


namespace clock_in_probability_l63_63924

-- Definitions
def start_time := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_start := 495 -- 8:15 in minutes from 00:00 (495 minutes)
def arrival_start := 470 -- 7:50 in minutes from 00:00 (470 minutes)
def arrival_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)

-- Conditions
def arrival_window := arrival_end - arrival_start -- 40 minutes window
def valid_clock_in_window := valid_clock_in_end - valid_clock_in_start -- 15 minutes window

-- Required proof statement
theorem clock_in_probability :
  (valid_clock_in_window : ℚ) / (arrival_window : ℚ) = 3 / 8 :=
by
  sorry

end clock_in_probability_l63_63924


namespace isosceles_triangle_l63_63247

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.cos B = b * Real.cos A) :
  (a = b) ∨ (a = c) ∨ (b = c) :=
begin
  sorry
end

end isosceles_triangle_l63_63247


namespace increase_breadth_percentage_l63_63379

theorem increase_breadth_percentage :
  ∀ (L B : ℝ),
    (let L' := 1.03 * L in
     let p := 100 * (1.06 - 1 : ℝ) in
     let B' := B * (1 + p / 100) in
     let A := L * B in
     let A' := 1.0918 * A in
     L' * B' = A'
    ) → p = 6 := 
by
  assume (L B : ℝ)
  have L' := 1.03 * L
  have p := 100 * (1.06 - 1 : ℝ)
  have B' := B * (1 + p / 100)
  have A := L * B
  have A' := 1.0918 * A
  have h1 : L' * B' = A' := by sorry
  show p = 6 from sorry

end increase_breadth_percentage_l63_63379


namespace cinema_max_value_k_l63_63114

theorem cinema_max_value_k :
  ∃ k, 
    k = 776 ∧ ∀ (ages : Fin 50 → ℕ), 
      (∀ i j, i ≠ j → ages i ≠ ages j) → 
      (∑ i, ages i = 1555) → 
      ∃ selected, 
        Multiset.card selected = 16 ∧ 
        Multiset.sum selected ≥ k :=
by
  sorry

end cinema_max_value_k_l63_63114


namespace max_determinant_value_l63_63954

noncomputable def max_determinant (θ φ : ℝ) : ℝ :=
  det (matrix.of ![
    ![1, 1, 1],
    ![1, 1 + sin θ, 1 + cos φ],
    ![1 + cos θ, 1 + sin φ, 1]])

theorem max_determinant_value : ∀ θ φ : ℝ, max_determinant θ φ ≤ 1 :=
sorry

end max_determinant_value_l63_63954


namespace three_students_two_groups_l63_63836

theorem three_students_two_groups : 
  (2 : ℕ) ^ 3 = 8 := 
by
  sorry

end three_students_two_groups_l63_63836


namespace logarithm_inequalities_l63_63656

theorem logarithm_inequalities (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) :
  let a := Real.log x,
      b := Real.log (x^2),
      c := (Real.log x)^2 in
  c < a ∧ a < b := 
by
  sorry

end logarithm_inequalities_l63_63656


namespace max_value_expression_l63_63535

theorem max_value_expression (a b c x y z : ℝ) (h1 : 2 ≤ a ∧ a ≤ 3) (h2 : 2 ≤ b ∧ b ≤ 3) (h3 : 2 ≤ c ∧ c ≤ 3)
    (perm : ∃ p : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ), p (a, b, c) = (x, y, z)) :
    (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z) ≤ 15 / 4) :=
begin
  sorry
end

end max_value_expression_l63_63535


namespace sum_of_even_integers_l63_63410

theorem sum_of_even_integers (first last common_d: ℕ) (h1: first = 100) (h2: last = 300) (h3: common_d = 2) : 
  let n := (last - first) / common_d + 1 in
  (n / 2 * (first + last) = 20200) := 
by
  sorry

end sum_of_even_integers_l63_63410


namespace least_integer_a_divisible_by_240_l63_63665

theorem least_integer_a_divisible_by_240 (a : ℤ) (h1 : 240 ∣ a^3) : a ≥ 60 := by
  sorry

end least_integer_a_divisible_by_240_l63_63665


namespace original_price_l63_63425

theorem original_price (x : ℝ)
  (installment_price : x * 1.04)
  (cash_price : x * 0.9)
  (price_difference : installment_price - cash_price = 700) :
  x = 5000 :=
by
  sorry

end original_price_l63_63425


namespace smallest_lcm_four_digit_integers_with_gcd_five_l63_63654

open Nat

theorem smallest_lcm_four_digit_integers_with_gcd_five : ∃ k ℓ : ℕ, 1000 ≤ k ∧ k < 10000 ∧ 1000 ≤ ℓ ∧ ℓ < 10000 ∧ gcd k ℓ = 5 ∧ lcm k ℓ = 203010 :=
by
  use 1005
  use 1010
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_lcm_four_digit_integers_with_gcd_five_l63_63654


namespace max_value_expression_l63_63537

theorem max_value_expression (a b c x y z : ℝ) (h1 : 2 ≤ a ∧ a ≤ 3) (h2 : 2 ≤ b ∧ b ≤ 3) (h3 : 2 ≤ c ∧ c ≤ 3)
    (perm : ∃ p : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ), p (a, b, c) = (x, y, z)) :
    (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z) ≤ 15 / 4) :=
begin
  sorry
end

end max_value_expression_l63_63537


namespace cinema_max_value_k_l63_63115

theorem cinema_max_value_k :
  ∃ k, 
    k = 776 ∧ ∀ (ages : Fin 50 → ℕ), 
      (∀ i j, i ≠ j → ages i ≠ ages j) → 
      (∑ i, ages i = 1555) → 
      ∃ selected, 
        Multiset.card selected = 16 ∧ 
        Multiset.sum selected ≥ k :=
by
  sorry

end cinema_max_value_k_l63_63115


namespace expected_value_one_roll_correct_expected_value_two_rolls_correct_expected_value_three_rolls_correct_l63_63798

noncomputable def expected_value_one_roll : ℝ := (1 + 2 + 3 + 4 + 5 + 6) / 6

noncomputable def expected_value_two_rolls : ℝ :=
  let E1 := expected_value_one_roll in
  (1 / 2) * ( (4 + 5 + 6) / 3 ) + (1 / 2) * E1

noncomputable def expected_value_three_rolls : ℝ :=
  let E2 := expected_value_two_rolls in
  (1 / 3) * ((5 + 6) / 2) + (2 / 3) * E2

theorem expected_value_one_roll_correct : expected_value_one_roll = 3.5 := by
  sorry

theorem expected_value_two_rolls_correct : expected_value_two_rolls = 4.25 := by
  sorry

theorem expected_value_three_rolls_correct : expected_value_three_rolls = 14 / 3 := by
  sorry

end expected_value_one_roll_correct_expected_value_two_rolls_correct_expected_value_three_rolls_correct_l63_63798


namespace area_triangle_QXY_l63_63816

-- Definition of the problem
def length_rectangle (PQ PS : ℝ) : Prop :=
  PQ = 8 ∧ PS = 6

def diagonal_division (PR : ℝ) (X Y : ℝ) : Prop :=
  PR = 10 ∧ X = 2.5 ∧ Y = 2.5

-- The statement we need to prove
theorem area_triangle_QXY
  (PQ PS PR X Y : ℝ)
  (h1 : length_rectangle PQ PS)
  (h2 : diagonal_division PR X Y)
  : ∃ (A : ℝ), A = 6 := by
  sorry

end area_triangle_QXY_l63_63816


namespace valid_coloring_count_l63_63690

-- Define the colors.
inductive Color
| red
| blue
| green

-- Define a type for the board, with constraints.
structure Board (m n : Nat) :=
  (color : Fin m → Fin n → Color)
  (valid : ∀ (i : Fin m) (j : Fin n), 
              (j < n-1 → color i j ≠ color i (j + 1)) ∧                 -- adjacent horizontally
              (i < m-1 → color i j ≠ color (i + 1) j))                   -- adjacent vertically

-- Specifically for a 2 x 9 board with no two adjacent cells having the same color
def valid_coloring : Nat :=
  2 * 9

-- Function to count the number of valid colorings
noncomputable def count_valid_colorings (b : Board 2 9) : Nat :=
  6 * 3 ^ 8

-- The theorem that states the number of valid colorings
theorem valid_coloring_count : count_valid_colorings _ = 39366 := by
  sorry

end valid_coloring_count_l63_63690


namespace time_to_meet_15th_l63_63090

-- Given constants and initial meeting time t
variables (v1 v2 : ℝ) (t : ℝ)

-- Conditions based on problem description
axiom car_a_time_to_B_after_first_meet : t + 4 * v1 = t * v2
axiom car_b_time_to_A_after_first_meet : (t + 4) * v1 = (t + 1) * v2

-- Define meeting interval times for Car A and Car B
def car_a_trip_time : ℝ := t + 4
def car_b_trip_time : ℝ := t + 1

-- Time taken by cars to meet 15th time (excluding at start/end)
def time_of_15th_meet : ℝ := 15 * t - 2

-- The theorem to prove
theorem time_to_meet_15th : time_of_15th_meet v1 v2 t = 86 := by
  sorry

end time_to_meet_15th_l63_63090


namespace rose_needs_more_money_l63_63329

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end rose_needs_more_money_l63_63329


namespace second_option_cost_per_day_l63_63274

theorem second_option_cost_per_day :
  let distance_one_way := 150
  let rental_first_option := 50
  let kilometers_per_liter := 15
  let cost_per_liter := 0.9
  let savings := 22
  let total_distance := distance_one_way * 2
  let total_liters := total_distance / kilometers_per_liter
  let gasoline_cost := total_liters * cost_per_liter
  let total_cost_first_option := rental_first_option + gasoline_cost
  let second_option_cost := total_cost_first_option + savings
  second_option_cost = 90 :=
by
  sorry

end second_option_cost_per_day_l63_63274


namespace find_a_of_pure_imaginary_l63_63186

theorem find_a_of_pure_imaginary (a : ℝ) (h : (a + 2 * complex.I) / (1 + complex.I) = complex.I * (x : ℝ)) : a = -2 :=
sorry

end find_a_of_pure_imaginary_l63_63186


namespace browser_usage_information_is_false_l63_63056

def num_people_using_A : ℕ := 316
def num_people_using_B : ℕ := 478
def num_people_using_both_A_and_B : ℕ := 104
def num_people_only_using_one_browser : ℕ := 567

theorem browser_usage_information_is_false :
  num_people_only_using_one_browser ≠ (num_people_using_A - num_people_using_both_A_and_B) + (num_people_using_B - num_people_using_both_A_and_B) :=
by
  sorry

end browser_usage_information_is_false_l63_63056


namespace winnie_retains_lollipops_l63_63881

theorem winnie_retains_lollipops :
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  lollipops_total % friends = 10 :=
by
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  show lollipops_total % friends = 10
  sorry

end winnie_retains_lollipops_l63_63881


namespace compare_real_numbers_l63_63476

theorem compare_real_numbers (a b c d : ℝ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = 2) :
  d > a ∧ d > b ∧ d > c :=
by
  sorry

end compare_real_numbers_l63_63476


namespace integer_pairs_solution_l63_63527

theorem integer_pairs_solution (k : ℕ) (h : k ≠ 1) : 
  ∃ (m n : ℤ), 
    ((m - n) ^ 2 = 4 * m * n / (m + n - 1)) ∧ 
    (m = k^2 + k / 2 ∧ n = k^2 - k / 2) ∨ 
    (m = k^2 - k / 2 ∧ n = k^2 + k / 2) :=
sorry

end integer_pairs_solution_l63_63527


namespace linear_function_through_point_parallel_line_l63_63373

noncomputable def function_expr (x : ℝ) : ℝ := 2 * x + 3

def point_A : ℝ × ℝ := (-2, -1)

def parallel_line (x : ℝ) : ℝ := 2 * x - 3

theorem linear_function_through_point_parallel_line :
  ∃ b : ℝ, (∀ x : ℝ, function_expr x = 2 * x + b) ∧ (function_expr (fst point_A) = snd point_A) :=
by
  use 3
  split
  . intro x
    refl
  . simp [function_expr, point_A]
    sorry

end linear_function_through_point_parallel_line_l63_63373


namespace cot_neg_45_l63_63131

-- Define the conditions
lemma cot_def (x : ℝ) : Real.cot x = 1 / Real.tan x := sorry
lemma tan_neg (x : ℝ) : Real.tan (-x) = -Real.tan x := sorry
lemma tan_45 : Real.tan (Real.pi / 4) = 1 := sorry

-- State the theorem to prove
theorem cot_neg_45 : Real.cot (-Real.pi / 4) = -1 :=
by
  apply cot_def
  apply tan_neg
  apply tan_45
  sorry

end cot_neg_45_l63_63131


namespace limit_of_sequence_limit_frac_seq_l63_63023

def N (ε : ℝ) : ℕ := ⌈((5 / ε) - 1) / 2⌉.toNat

theorem limit_of_sequence (ε : ℝ) (n : ℕ) (hn : n ≥ N ε) 
  (hε_pos : ε > 0) : 
  abs ((4 * n - 3) / (2 * n + 1) - 2) < ε :=
sorry

theorem limit_frac_seq : 
  tendsto (λ n, (4 * n - 3) / (2 * n + 1)) at_top (𝓝 2) :=
begin
  intros ε hε,
  use N ε,
  intros n hn,
  exact limit_of_sequence ε n hn hε,
end

end limit_of_sequence_limit_frac_seq_l63_63023


namespace unique_solution_for_2_3_6_eq_7_l63_63969

theorem unique_solution_for_2_3_6_eq_7 (x : ℝ) : 2^x + 3^x + 6^x = 7^x → x = 2 :=
by
  intro h
  -- Add the relevant proof tactic steps here
  sorry

end unique_solution_for_2_3_6_eq_7_l63_63969


namespace convert_512_to_base_7_l63_63095

/-- Convert 512 in base 10 to base 7 and verify the result is 1331 in base 7 -/
theorem convert_512_to_base_7 : Nat.toDigits 7 512 = [1, 3, 3, 1] := sorry

end convert_512_to_base_7_l63_63095


namespace maximum_value_A_l63_63984

theorem maximum_value_A (x y z : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  ∃ A ≤ 2, A = (√(8 * x^4 + y) + √(8 * y^4 + z) + √(8 * z^4 + x) - 3) / (x + y + z) := 
sorry

end maximum_value_A_l63_63984


namespace simplify_factorial_expression_l63_63783

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem simplify_factorial_expression :
  (factorial 13) / (factorial 11 + 3 * factorial 10) = 122 := by
  sorry

end simplify_factorial_expression_l63_63783


namespace pushups_total_l63_63297

theorem pushups_total (x melanie david karen john : ℕ) 
  (hx : x = 51)
  (h_melanie : melanie = 2 * x - 7)
  (h_david : david = x + 22)
  (h_avg : (x + melanie + david) / 3 = (x + (2 * x - 7) + (x + 22)) / 3)
  (h_karen : karen = (x + (2 * x - 7) + (x + 22)) / 3 - 5)
  (h_john : john = (x + 22) - 4) :
  john + melanie + karen = 232 := by
  sorry

end pushups_total_l63_63297


namespace option_A_is_quadratic_l63_63011

def is_quadratic (p : Polynomial ℤ) : Prop :=
  p.degree = 2

def poly_A := (Polynomial.C 1) - (Polynomial.X) - (Polynomial.X ^ 2)
def poly_C := (Polynomial.C 1) + (Polynomial.C 2) * (Polynomial.X)

theorem option_A_is_quadratic :
  is_quadratic poly_A ∧ ¬(is_quadratic (Polynomial.C 1)) ∧ ¬(is_quadratic poly_C) ∧ ¬(is_quadratic ((Polynomial.X ^ (-1)) + (Polynomial.X))) :=
by {
  sorry
}

end option_A_is_quadratic_l63_63011


namespace solve_trigonometric_equation_count_solutions_l63_63347

theorem solve_trigonometric_equation :
  ∀ x : ℝ, 2000 ≤ x ∧ x ≤ 3000 →
  2 * real.sqrt 2 * real.sin (real.pi * x / 4) ^ 3 = real.sin (real.pi / 4 * (1 + x)) →
  ∃! (n : ℤ), 500 ≤ n ∧ n ≤ 749 ∧ x = 1 + 4 * n :=
sorry

-- Count the unique solutions within the given range
theorem count_solutions :
  let num_solutions := (749 - 500 + 1 : ℤ) in
  num_solutions = 250 :=
by
  simp [Int.ofNat_sub, Int.add_one, Int.ofNat_one, Int.ofNat_add]
  linarith

end solve_trigonometric_equation_count_solutions_l63_63347


namespace steve_speed_l63_63717

theorem steve_speed (v : ℝ) : 
  (John_initial_distance_behind_Steve = 15) ∧ 
  (John_final_distance_ahead_of_Steve = 2) ∧ 
  (John_speed = 4.2) ∧ 
  (final_push_duration = 34) → 
  v * final_push_duration = (John_speed * final_push_duration) - (John_initial_distance_behind_Steve + John_final_distance_ahead_of_Steve) →
  v = 3.7 := 
by
  intros hconds heq
  exact sorry

end steve_speed_l63_63717


namespace unique_solution_l63_63965

noncomputable def equation_satisfied (x : ℝ) : Prop :=
  2^x + 3^x + 6^x = 7^x

theorem unique_solution : ∀ x : ℝ, equation_satisfied x ↔ x = 2 := by
  sorry

end unique_solution_l63_63965


namespace number_of_77s_l63_63084

theorem number_of_77s (a b : ℕ) :
  (∃ a : ℕ, 1015 = a + 3 * 77 ∧ a + 21 = 10)
  ∧ (∃ b : ℕ, 2023 = b + 6 * 77 + 2 * 777 ∧ b = 7)
  → 6 = 6 := 
by
    sorry

end number_of_77s_l63_63084


namespace rose_needs_more_money_l63_63327

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end rose_needs_more_money_l63_63327


namespace eccentricity_of_ellipse_minimum_value_of_dot_product_l63_63576

-- Define the ellipse equation and conditions
def ellipse (x y a : ℝ) : Prop := x^2 / a^2 + y^2 / 2 = 1

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Part 1: Prove the eccentricity of the ellipse
theorem eccentricity_of_ellipse (a c e : ℝ) (h1 : a = 4) (h2 : c = sqrt(8)) : 
  e = c / a := by
  sorry

-- Conditions for Part 2
def line (x y b : ℝ) : Prop := y = - (sqrt 2 / 2) * x + b

-- Conditions for points M and N, and point Q
def intersection_condition (x y b : ℝ) : Prop := 
  ∀ {M N : ℝ × ℝ}, ellipse M.1 M.2 4 → ellipse N.1 N.2 4 → 
  line M.1 M.2 b ∧ line N.1 N.2 b → 
  - sqrt 10 < b ∧ b < sqrt 10

def vector_dot_product (QM QN : ℝ × ℝ) : ℝ := 
  QM.1 * QN.1 + QM.2 * QN.2

-- Point Q
def Q : ℝ × ℝ := (- sqrt 2, 0)

-- Part 2: Prove the minimum value of the dot product
theorem minimum_value_of_dot_product (b : ℝ) :
  intersection_condition Q.1 0 b → 
  ∃ M N : ℝ × ℝ, ellipse M.1 M.2 4 ∧ ellipse N.1 N.2 4 ∧ 
  line M.1 M.2 b ∧ line N.1 N.2 b ∧
  ∀ (val : ℝ), vector_dot_product (M.1 + Q.1, M.2 + Q.2) (N.1 + Q.1, N.2 + Q.2) = val → val = -38 / 9 := by
  sorry

end eccentricity_of_ellipse_minimum_value_of_dot_product_l63_63576


namespace magnitude_of_product_of_sums_l63_63946

noncomputable def complex_numbers_equilateral_triangle_with_properties (x y z : ℂ) (side_length : ℝ) :=
  ∀ x y z, abs (x - y) = side_length ∧ abs (y - z) = side_length ∧ abs (z - x) = side_length

theorem magnitude_of_product_of_sums (x y z : ℂ) (h_triangle: complex_numbers_equilateral_triangle_with_properties x y z 20) (h_sum: abs (x + y + z) = 40) :
  abs (x * y + x * z + y * z) = 1600 / 3 :=
by
  sorry

end magnitude_of_product_of_sums_l63_63946


namespace pentagon_area_l63_63321

theorem pentagon_area (P Q R S T : Point)
  (h1 : is_square P Q R S)
  (h2 : is_perpendicular P T R)
  (h3 : distance P T = 5)
  (h4 : distance T R = 12) :
  area_of_pentagon P T R S Q = 139 := by
  sorry

end pentagon_area_l63_63321


namespace unique_solution_for_2_3_6_eq_7_l63_63971

theorem unique_solution_for_2_3_6_eq_7 (x : ℝ) : 2^x + 3^x + 6^x = 7^x → x = 2 :=
by
  intro h
  -- Add the relevant proof tactic steps here
  sorry

end unique_solution_for_2_3_6_eq_7_l63_63971


namespace sum_ratios_geq_n_l63_63293

theorem sum_ratios_geq_n (n : ℕ) (x : Finₓ n → ℝ) (h_pos : ∀ i, 0 < x i) : 
  (∑ i, (x i) / (x ((i + 1) % n))) ≥ n := 
sorry

end sum_ratios_geq_n_l63_63293


namespace bob_pays_more_than_samantha_l63_63333

theorem bob_pays_more_than_samantha
  (total_slices : ℕ := 12)
  (cost_plain_pizza : ℝ := 12)
  (cost_olives : ℝ := 3)
  (slices_one_third_pizza : ℕ := total_slices / 3)
  (total_cost : ℝ := cost_plain_pizza + cost_olives)
  (cost_per_slice : ℝ := total_cost / total_slices)
  (bob_slices_total : ℕ := slices_one_third_pizza + 3)
  (samantha_slices_total : ℕ := total_slices - bob_slices_total)
  (bob_total_cost : ℝ := bob_slices_total * cost_per_slice)
  (samantha_total_cost : ℝ := samantha_slices_total * cost_per_slice) :
  bob_total_cost - samantha_total_cost = 2.5 :=
by
  sorry

end bob_pays_more_than_samantha_l63_63333


namespace count_integers_in_interval_l63_63640

theorem count_integers_in_interval :
  {n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi}.toFinset.card = 57 :=
by
  sorry

end count_integers_in_interval_l63_63640


namespace range_of_b_l63_63211

noncomputable def divide_triangle (a b : ℝ) (ha : a > 0) : Prop :=
  ∃ (x y : ℝ), 
    triangle_area (point (-1) 0) (point 1 0) (point 0 1) / 2 = 
    triangle_area (point (-1) 0) (point x y) (point 0 1) + 
    triangle_area (point x y) (point 1 0) (point 0 1) ∧
    y = a * x + b

theorem range_of_b :
  ∀ (a : ℝ), a > 0 →
  ∃ (b : ℝ), b ∈ (1 - real.sqrt 2 / 2, 1 / 2) ∧ 
  divide_triangle a b (by assumption) :=
sorry

end range_of_b_l63_63211


namespace sum_of_all_possible_values_of_g_l63_63734

theorem sum_of_all_possible_values_of_g :
  let f (x : ℝ) := x^3 - 9 * x^2 + 27 * x - 25,
      g (y : ℝ) := 3 * y + 4 in
  (∃ x : ℝ, f x = 7 ∧ g ($(f 7)) = 39) :=
by
  sorry

end sum_of_all_possible_values_of_g_l63_63734


namespace range_of_x_div_y_l63_63587

theorem range_of_x_div_y {x y : ℝ} (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) : 
  (1/8 < x / y) ∧ (x / y < 3) :=
sorry

end range_of_x_div_y_l63_63587


namespace max_age_for_16_viewers_l63_63110

def maximum_group_age (n : ℕ) (total_age : ℕ) (unique_ages : ℕ → Prop) : ℕ :=
if (n = 50 ∧ total_age = 1555 ∧ ∀ i j, i ≠ j → unique_ages i ≠ unique_ages j) 
then 776 
else 0

theorem max_age_for_16_viewers 
  (ages : Fin 50 → ℕ) 
  (h_unique : Function.Injective ages) 
  (h_sum : ∑ i, ages i = 1555) : 
  ∃ k, k = 776 ∧ ∃ s : Finset (Fin 50), s.card = 16 ∧ ∑ i in s, ages i ≥ k :=
begin
  use 776,
  intro s,
  have : s.card = 16 := sorry,
  use s,
  split,
  { exact this },
  { sorry }
end

end max_age_for_16_viewers_l63_63110


namespace absolute_value_expression_correct_l63_63496

theorem absolute_value_expression_correct (π : ℝ) :
  | 3 * π - | π + 7 | | = 7 - 2 * π := by
  sorry

end absolute_value_expression_correct_l63_63496


namespace total_number_of_boys_in_all_class_sections_is_380_l63_63253

theorem total_number_of_boys_in_all_class_sections_is_380 :
  let students_section1 := 160
  let students_section2 := 200
  let students_section3 := 240
  let girls_section1 := students_section1 / 4
  let boys_section1 := students_section1 - girls_section1
  let boys_section2 := (3 / 5) * students_section2
  let total_parts := 7 + 5
  let boys_section3 := (7 / total_parts) * students_section3
  boys_section1 + boys_section2 + boys_section3 = 380 :=
sorry

end total_number_of_boys_in_all_class_sections_is_380_l63_63253


namespace solution_l63_63988

noncomputable def g (a : ℝ) : Polynomials ℝ := X^3 + a * X^2 + 2 * X + 15
noncomputable def f (a b c : ℝ) : Polynomials ℝ := X^4 + 2 * X^3 + b * X^2 + 150 * X + c

theorem solution (a b c : ℝ) (h1 : (g a).roots.nodup) (h2 : ∀ r ∈ (g a).roots, f a b c).is_root r) :
  (f a b c).eval 1 = -15640 :=
by 
  sorry

end solution_l63_63988


namespace functional_equation_l63_63529

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1) :
  f = λ x, 1 - (x^2 / 2) :=
by sorry

end functional_equation_l63_63529


namespace square_problem_l63_63261

theorem square_problem
  (PQRS : square)
  (P Q R S T M N : point)
  (PQ : segment PQRS)
  (center_T : center PQRS T)
  (side_PQ_eq_800 : length PQ = 800)
  (M_between_PN : between M P N)
  (PM_lt_NQ : length (segment P M) < length (segment N Q))
  (angle_MTN_eq_45 : angle (∠ M T N) = 45)
  (MN_eq_300 : length (segment M N) = 300)
  (NQ_eq_a_b_sqrt_c : ∃ a b c : ℕ, NQ = a + b * sqrt c ∧ c ∉ { p * p | p ∣ c}) :
  ∀ a b c : ℕ, MN = a + b * sqrt c → a + b + c = 310 :=
sorry

end square_problem_l63_63261


namespace distances_from_inscribed_circle_satisfy_relation_l63_63685

theorem distances_from_inscribed_circle_satisfy_relation
  (x y : ℝ) (a b c d : ℝ)
  (h1 : x^2 + y^2 = 1)
  (ha : a = real.sqrt ((x - 1)^2 + (y - 1)^2))
  (hb : b = real.sqrt ((x + 1)^2 + (y - 1)^2))
  (hc : c = real.sqrt ((x + 1)^2 + (y + 1)^2))
  (hd : d = real.sqrt ((x - 1)^2 + (y + 1)^2)) :
  a^2 * c^2 + b^2 * d^2 = 10 :=
sorry

end distances_from_inscribed_circle_satisfy_relation_l63_63685


namespace area_of_pentagon_PTRSQ_l63_63317

theorem area_of_pentagon_PTRSQ (PQRS : Type) [geometry PQRS]
  {P Q R S T : PQRS} 
  (h1 : square P Q R S) 
  (h2 : perp PT TR) 
  (h3 : distance P T = 5) 
  (h4 : distance T R = 12) : 
  area_pentagon PTRSQ = 139 :=
sorry

end area_of_pentagon_PTRSQ_l63_63317


namespace rose_needs_more_money_l63_63330

def cost_paintbrush : ℝ := 2.40
def cost_paints : ℝ := 9.20
def cost_easel : ℝ := 6.50
def money_rose_has : ℝ := 7.10

theorem rose_needs_more_money : 
  cost_paintbrush + cost_paints + cost_easel - money_rose_has = 11.00 :=
begin
  sorry
end

end rose_needs_more_money_l63_63330


namespace power_equal_20mn_l63_63281

theorem power_equal_20mn (m n : ℕ) (P Q : ℕ) (hP : P = 2^m) (hQ : Q = 5^n) : 
  P^(2 * n) * Q^m = (20^(m * n)) :=
by
  sorry

end power_equal_20mn_l63_63281


namespace brother_current_age_l63_63850

-- Definition of Viggo's younger brother's age currently
def B := Nat

-- Assumptions
variable (B : ℕ) (current_age_sum : B + (B + 12) = 32)

-- Proof statement (the question)
theorem brother_current_age : B = 10 :=
by
  -- introducing the assumptions
  have h1 : (B + 12) + B = 32 := current_age_sum

  -- simplifying the equation
  sorry

end brother_current_age_l63_63850


namespace max_age_for_16_viewers_l63_63112

def maximum_group_age (n : ℕ) (total_age : ℕ) (unique_ages : ℕ → Prop) : ℕ :=
if (n = 50 ∧ total_age = 1555 ∧ ∀ i j, i ≠ j → unique_ages i ≠ unique_ages j) 
then 776 
else 0

theorem max_age_for_16_viewers 
  (ages : Fin 50 → ℕ) 
  (h_unique : Function.Injective ages) 
  (h_sum : ∑ i, ages i = 1555) : 
  ∃ k, k = 776 ∧ ∃ s : Finset (Fin 50), s.card = 16 ∧ ∑ i in s, ages i ≥ k :=
begin
  use 776,
  intro s,
  have : s.card = 16 := sorry,
  use s,
  split,
  { exact this },
  { sorry }
end

end max_age_for_16_viewers_l63_63112


namespace find_a_plus_b_l63_63948

theorem find_a_plus_b (a b : ℝ)
  (h1 : ab^2 = 0)
  (h2 : 2 * a^2 * b = 0)
  (h3 : a^3 + b^2 = 0)
  (h4 : ab = 1) : a + b = -2 :=
sorry

end find_a_plus_b_l63_63948


namespace cat_mice_hunting_days_l63_63015

theorem cat_mice_hunting_days
  (cat_rate : ℝ)
  (total_work : ℝ := 1)
  (initial_cats : ℕ := 2)
  (additional_cats : ℕ := 3)
  (initial_days : ℕ := 5)
  (initial_work_fraction : ℝ := 0.5)
  (total_cats : ℕ := initial_cats + additional_cats) :
  ∀ (remaining_work : ℝ := total_work - initial_work_fraction),
  let total_days := initial_days + remaining_work / (cat_rate * total_cats) in
  (initial_cats * cat_rate * initial_days = initial_work_fraction * total_work) →
  total_days = 7 :=
by
  intros remaining_work total_days h
  sorry

end cat_mice_hunting_days_l63_63015


namespace fraction_of_boys_at_sports_event_equals_117_230_l63_63833

def total_students_lincoln : ℕ := 300
def boys_to_girls_ratio_lincoln : ℕ × ℕ := (3, 2)

def total_students_jackson : ℕ := 240
def boys_to_girls_ratio_jackson : ℕ × ℕ := (2, 3)

def total_students_franklin : ℕ := 150
def boys_to_girls_ratio_franklin : ℕ × ℕ := (1, 1)

theorem fraction_of_boys_at_sports_event_equals_117_230 :
  let boys_lincoln := 3 * (total_students_lincoln / (boys_to_girls_ratio_lincoln.1 + boys_to_girls_ratio_lincoln.2)),
      boys_jackson := 2 * (total_students_jackson / (boys_to_girls_ratio_jackson.1 + boys_to_girls_ratio_jackson.2)),
      boys_franklin := 1 * (total_students_franklin / (boys_to_girls_ratio_franklin.1 + boys_to_girls_ratio_franklin.2)),
      total_boys := boys_lincoln + boys_jackson + boys_franklin,
      total_students := total_students_lincoln + total_students_jackson + total_students_franklin
  in total_boys.to_rat / total_students.to_rat = (117 : ℚ) / 230 :=
by
  sorry

end fraction_of_boys_at_sports_event_equals_117_230_l63_63833


namespace max_true_statements_l63_63738

theorem max_true_statements (x : ℝ) :
  (0 < x^3 ∧ x^3 < 1 ∨ x^3 > 1 ∨ -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 0 < x^2 - x^3 ∧ x^2 - x^3 < 1) →
  ((0 < x^3 ∧ x^3 < 1 ∨ x^3 > 1) ∧ (-1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1) ∧ (0 < x^2 - x^3 ∧ x^2 - x^3 < 1) → false) ∧ 
  (0 < x^3 ∧ x^3 < 1 ∧ 0 < x ∧ x < 1 ∧ 0 < x^2 - x^3 ∧ x^2 - x^3 < 1) → true :=
by
sorrry

end max_true_statements_l63_63738


namespace machine_x_sprockets_per_hour_l63_63016

theorem machine_x_sprockets_per_hour:
  ∀ (S:ℝ) (T:ℝ),
  (660 = S * (T + 10)) ∧ (660 = 1.1 * S * T) → S = 6 :=
by
  intros S T h,
  cases h with h1 h2,
  sorry

end machine_x_sprockets_per_hour_l63_63016


namespace intersection_dist_general_l63_63174

theorem intersection_dist_general {a b : ℝ} 
  (h1 : (a^2 + 1) * (a^2 + 4 * (b + 1)) = 34)
  (h2 : (a^2 + 1) * (a^2 + 4 * (b + 2)) = 42) : 
  ∀ x1 x2 : ℝ, 
  x1 ≠ x2 → 
  (x1 * x1 = a * x1 + b - 1 ∧ x2 * x2 = a * x2 + b - 1) → 
  |x2 - x1| = 3 * Real.sqrt 2 :=
by
  sorry

end intersection_dist_general_l63_63174


namespace hyperbola_eccentricity_l63_63591

theorem hyperbola_eccentricity (a b c : ℝ) (h_asymptotes : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  (c / a = 5 / 4) ∨ (c / a = 5 / 3) :=
by
  -- Proof omitted
  sorry

end hyperbola_eccentricity_l63_63591


namespace original_number_of_people_l63_63267

variable (x : ℕ)
-- Conditions
axiom one_third_left : x / 3 > 0
axiom half_dancing : 18 = x / 3

-- Theorem Statement
theorem original_number_of_people (x : ℕ) (one_third_left : x / 3 > 0) (half_dancing : 18 = x / 3) : x = 54 := sorry

end original_number_of_people_l63_63267


namespace find_y_l63_63553

theorem find_y (y : ℝ) (h : sqrt (7 * y) / sqrt (4 * (y - 2)) = 3) : y = 72 / 29 :=
by
  sorry

end find_y_l63_63553


namespace sum_first_n_terms_eq_sum_b_n_eq_T_n_l63_63575

variable (n : ℕ)

def a_n (n : ℕ) : ℕ := 2 * n + 1

def S_n (n : ℕ) : ℕ := n^2 + 2 * n

theorem sum_first_n_terms_eq (n : ℕ) : 
  (∑ k in Finset.range (n + 1), a_n k) = S_n n := sorry

def b_n (n : ℕ) : ℚ := 1 / ((a_n n)^2 - 1)

def T_n (n : ℕ) : ℚ := (n : ℚ) / (4 * ((n + 1) : ℚ))

theorem sum_b_n_eq_T_n (n : ℕ) : 
  (∑ k in Finset.range (n + 1), b_n k) = T_n n := sorry

end sum_first_n_terms_eq_sum_b_n_eq_T_n_l63_63575


namespace sinx_sufficient_not_necessary_cosx_l63_63889

def sin_cos_relationship : Prop :=
  ∀ x : ℝ, sin x = 1 → cos x = 0 ∧ ¬(cos x = 0 → sin x = 1)

theorem sinx_sufficient_not_necessary_cosx : sin_cos_relationship :=
by
  sorry

end sinx_sufficient_not_necessary_cosx_l63_63889


namespace medal_winners_determined_in_39_matches_l63_63028

noncomputable def tournament_matches (competitors : ℕ) : ℕ :=
  if competitors = 32 then
    39
  else
    32 -- other case (not necessary here, just for completeness)

theorem medal_winners_determined_in_39_matches : 
  ∀ (competitors : ℕ), competitors = 32 →
  (∀ x y : ℕ, x ≠ y) →  -- No two competitors are equal
  (∀ x y : ℕ, x ≤ y ∨ y ≤ x) → -- Strict ordering of competitors
  tournament_matches competitors = 39 := begin
  sorry
end

end medal_winners_determined_in_39_matches_l63_63028


namespace total_amount_spent_l63_63443

theorem total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) 
  (h_food_price : food_price = 100) (h_sales_tax_rate : sales_tax_rate = 0.10) 
  (h_tip_rate : tip_rate = 0.20) : 
  let sales_tax := food_price * sales_tax_rate in
  let total_before_tip := food_price + sales_tax in
  let tip := total_before_tip * tip_rate in
  let total_amount_spent := total_before_tip + tip in
  total_amount_spent = 132 :=
by 
  sorry

end total_amount_spent_l63_63443


namespace cot_neg_45_eq_neg_1_l63_63136

-- Hypotheses
variable (θ : ℝ)
variable (h1 : 𝔸.cot θ = 1 / 𝔸.tan θ)
variable (h2 : 𝔸.tan (-45) = -𝔸.tan 45)
variable (h3 : 𝔸.tan 45 = 1)

-- Theorem
theorem cot_neg_45_eq_neg_1 :
  𝔸.cot (-45) = -1 := by
  sorry

end cot_neg_45_eq_neg_1_l63_63136


namespace hyperbola_asymptote_equation_l63_63189

noncomputable def parabola_focus (a : ℝ) : ℝ × ℝ :=
  (a, 0)

noncomputable def parabola_directrix (a : ℝ) : ℝ :=
  -a

noncomputable def hyperbola_asymptote (a b : ℝ) : ℝ → ℝ :=
  λ x, (b / a) * x

theorem hyperbola_asymptote_equation :
  let a := 2
  let directrix := parabola_directrix a
  let focus := (-2, 0)  -- directrix passes through one focus of C2 which is at (-2,0)
  let intercepted_chord_len := 6 in
  ∃ (b : ℝ),
    (directrix = -2) ∧
    (intercepted_chord_len = 6) ∧
    (hyperbola_asymptote 1 (sqrt 3) = λ x, sqrt 3 * x) ∧
    (hyperbola_asymptote 1 (sqrt 3) = λ x, sqrt 3 * x) :=
by
  sorry

end hyperbola_asymptote_equation_l63_63189


namespace hyperbola_asymptotes_l63_63208

theorem hyperbola_asymptotes 
  (a b : ℝ)
  (hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (F₁ F₂ P : ℝ × ℝ)
  (line_perpendicular : ∃ c : ℝ, (F₂.1 = c))
  (angle_condition : ∃ (θ : ℝ), θ = 𝜋 / 6 ∧ angle P F₁ F₂ = θ) :
  asymptotes ? := sorry -- to be defined

end hyperbola_asymptotes_l63_63208


namespace triangle_angle_ratio_l63_63242

theorem triangle_angle_ratio (a b c : ℝ) (h₁ : a + b + c = 180)
  (h₂ : b = 2 * a) (h₃ : c = 3 * a) : a = 30 ∧ b = 60 ∧ c = 90 :=
by
  sorry

end triangle_angle_ratio_l63_63242


namespace largest_in_column_smallest_in_row_l63_63009

def array := 
  [[12, 7, 9, 6, 3], 
   [14, 9, 16, 13, 11], 
   [10, 5, 6, 8, 12], 
   [15, 6, 18, 14, 4], 
   [9, 4, 7, 12, 5]]

def largest_in_column (arr : List (List Nat)) (col : Nat) : Nat :=
  arr.foldr (λ row acc => max (row.getD col 0) acc) 0

def smallest_in_row (arr : List (List Nat)) (row : Nat) : Nat :=
  arr.getD row [].foldr min ⊤

theorem largest_in_column_smallest_in_row :
  ∃ row col, array.getD row [] = [14, 9, 16, 13, 11] ∧ 
            array.getD 1 [] = [14, 9, 16, 13, 11] ∧
            largest_in_column array col = 9 ∧
            smallest_in_row array row = 9 :=
sorry

end largest_in_column_smallest_in_row_l63_63009


namespace minimize_sum_of_distances_l63_63180

-- Definitions for the problem
variables {A B C D O : Point}

-- Theorem statement
theorem minimize_sum_of_distances (A B C D : Point) :
  ∃ O : Point, (
    O = intersection (diagonal A C) (diagonal B D) ∧ 
    ∀ Q : Point, sum_of_distances Q [A, B, C, D] ≥ sum_of_distances O [A, B, C, D]
  ) :=
sorry

end minimize_sum_of_distances_l63_63180


namespace incorrect_option_C_l63_63565

def A : Set ℕ := {1, a} -- Assuming a is of type ℕ

variable (a : ℕ)

theorem incorrect_option_C (h1 : 1 ∈ A) (h2 : a ∈ A) (h3 : 1 ≠ a) : ¬ (1, a) ∈ A :=
begin
  sorry
end

end incorrect_option_C_l63_63565


namespace math_proof_l63_63885

noncomputable def problem (a b : ℝ) : Prop :=
  a - b = 2 ∧ a^2 + b^2 = 25 → a * b = 10.5

-- We state the problem as a theorem:
theorem math_proof (a b : ℝ) (h1: a - b = 2) (h2: a^2 + b^2 = 25) : a * b = 10.5 :=
by {
  sorry -- Proof goes here
}

end math_proof_l63_63885


namespace valid_x_y_sum_l63_63657

-- Setup the initial conditions as variables.
variables (x y : ℕ)

-- Declare the conditions as hypotheses.
theorem valid_x_y_sum (h1 : 0 < x) (h2 : x < 25)
  (h3 : 0 < y) (h4 : y < 25) (h5 : x + y + x * y = 119) :
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end valid_x_y_sum_l63_63657


namespace extra_days_added_l63_63837

def runs_weekly_hours (days_per_week : ℕ) (hours_per_day : ℕ) : ℕ :=
  days_per_week * hours_per_day

def original_days_per_week := 3
def current_hours_per_week := 10
def hours_per_day := 2  -- 1 hour in the morning + 1 hour in the evening

theorem extra_days_added : 
  let current_days_per_week := current_hours_per_week / hours_per_day in
  current_days_per_week - original_days_per_week = 2 :=
by
  -- The proof itself would go here if needed
  sorry

end extra_days_added_l63_63837


namespace probability_of_Ace_then_King_l63_63845

def numAces : ℕ := 4
def numKings : ℕ := 4
def totalCards : ℕ := 52

theorem probability_of_Ace_then_King : 
  (numAces / totalCards) * (numKings / (totalCards - 1)) = 4 / 663 :=
by
  sorry

end probability_of_Ace_then_King_l63_63845


namespace accurate_place_24000_scientific_notation_46400000_l63_63353

namespace MathProof

def accurate_place (n : ℕ) : String :=
  if n = 24000 then "hundred's place" else "unknown"

def scientific_notation (n : ℕ) : String :=
  if n = 46400000 then "4.64 × 10^7" else "unknown"

theorem accurate_place_24000 : accurate_place 24000 = "hundred's place" :=
by
  sorry

theorem scientific_notation_46400000 : scientific_notation 46400000 = "4.64 × 10^7" :=
by
  sorry

end MathProof

end accurate_place_24000_scientific_notation_46400000_l63_63353


namespace problem_l63_63437

variables (A B C D K L M N P: Type*) [Point A] [Point B] [Point C] [Point D]
  [Point K] [Point L] [Point M] [Point N] [Point P] (circ : K → L → M → convex_quad ABCD) 
  (line_l : L → parallel AD)

theorem problem (K_L_M : circle_tangent (AD DC CB) K L M) 
              (line_l_properties : line_through L parallel_to AD) 
              (line_intersects : ∀ (line_l : Line), intersects line_l (KM KC) = (N, P)) :
  PL = PN :=
sorry

end problem_l63_63437


namespace solution_set_of_inequality_l63_63159

variable {R : Type*} [OrderedField R]

noncomputable def is_odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

noncomputable def monotonic_increasing (f : R → R) : Prop :=
  ∀ a b : R, a ≠ b → (f a - f b) / (a - b) > 0

theorem solution_set_of_inequality (f : R → R) (h_odd : is_odd_function f) (h_mono_inc : monotonic_increasing f) :
  { m : R | f (m + 2) + f (m - 6) > 0 } = { m : R | m > 2 } :=
by
  sorry

end solution_set_of_inequality_l63_63159


namespace log_expression_simplified_l63_63877

-- Definitions for conditions
def logBase4_16 := logBase 4 16
def logBase4_1_div_16 := logBase 4 (1 / 16)
def logBase4_32 := logBase 4 32

-- Theorem to be proved
theorem log_expression_simplified : 
  logBase4_16 / logBase4_1_div_16 + logBase4_32 = 1.5 := 
by 
  sorry

end log_expression_simplified_l63_63877


namespace pentagon_PTRSQ_area_proof_l63_63315

-- Define the geometric setup and properties
def quadrilateral_PQRS_is_square (P Q R S T : Type) : Prop :=
  -- Here, we will skip the precise geometric construction and assume the properties directly.
  sorry

def segment_PT_perpendicular_to_TR (P T R : Type) : Prop :=
  sorry

def PT_eq_5 (PT : ℝ) : Prop :=
  PT = 5

def TR_eq_12 (TR : ℝ) : Prop :=
  TR = 12

def area_PTRSQ (area : ℝ) : Prop :=
  area = 139

theorem pentagon_PTRSQ_area_proof
  (P Q R S T : Type)
  (PQRS_is_square : quadrilateral_PQRS_is_square P Q R S T)
  (PT_perpendicular_TR : segment_PT_perpendicular_to_TR P T R)
  (PT_length : PT_eq_5 5)
  (TR_length : TR_eq_12 12)
  : area_PTRSQ 139 :=
  sorry

end pentagon_PTRSQ_area_proof_l63_63315


namespace find_6_games_with_12_players_l63_63687

theorem find_6_games_with_12_players 
  (players : Finset ℕ) (games : Finset (ℕ × ℕ)) 
  (h_players_count : players.card = 20)
  (h_games_count : games.card = 14)
  (h_each_player_in_a_game : ∀ p ∈ players, ∃ g ∈ games, p ∈ g.1 ∨ p ∈ g.2) :
  ∃ subset_games : Finset (ℕ × ℕ), subset_games.card = 6 ∧ 
  (∀ g ∈ subset_games, ∃ (p1 p2 : ℕ), g = (p1, p2) ∧ p1 ≠ p2 ∧ p1 ∈ players ∧ p2 ∈ players) ∧
  ((subset_games.map (λ g : ℕ × ℕ, g.1)).to_finset ∪ (subset_games.map (λ g : ℕ × ℕ, g.2)).to_finset).card = 12 :=
begin
  sorry
end

end find_6_games_with_12_players_l63_63687


namespace solve_for_x_l63_63097

def operation (a b : ℝ) : ℝ := a^2 - 3*a + b

theorem solve_for_x (x : ℝ) : operation x 2 = 6 → (x = -1 ∨ x = 4) :=
by
  sorry

end solve_for_x_l63_63097


namespace cube_volume_proof_l63_63683

-- Define the conditions
def len_inch : ℕ := 48
def width_inch : ℕ := 72
def total_surface_area_inch : ℕ := len_inch * width_inch
def num_faces : ℕ := 6
def area_one_face_inch : ℕ := total_surface_area_inch / num_faces
def inches_to_feet (length_in_inches : ℕ) : ℕ := length_in_inches / 12

-- Define the key elements of the proof problem
def side_length_inch : ℕ := Int.natAbs (Nat.sqrt area_one_face_inch)
def side_length_ft : ℕ := inches_to_feet side_length_inch
def volume_ft3 : ℕ := side_length_ft ^ 3

-- State the proof problem
theorem cube_volume_proof : volume_ft3 = 8 := by
  -- The proof would be implemented here
  sorry

end cube_volume_proof_l63_63683


namespace smallest_positive_period_of_f_maximum_value_of_f_on_interval_minimum_value_of_f_on_interval_l63_63200

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6) + 2 * (cos x)^2

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem maximum_value_of_f_on_interval : ∃ x ∈ set.Icc 0 (π / 2), f x = 2 := sorry

theorem minimum_value_of_f_on_interval : ∃ x ∈ set.Icc 0 (π / 2), f x = 1 / 2 := sorry

end smallest_positive_period_of_f_maximum_value_of_f_on_interval_minimum_value_of_f_on_interval_l63_63200


namespace sum_a_n_2001_l63_63985

def a_n (n : ℕ) : ℕ :=
  if n % 182 = 0 then 11
  else if n % 154 = 0 then 13
  else if n % 143 = 0 then 14
  else 0

theorem sum_a_n_2001 : ∑ n in Finset.range 2001.succ, a_n n = 448 :=
by
  sorry

end sum_a_n_2001_l63_63985


namespace expected_digits_icosahedral_die_l63_63492

theorem expected_digits_icosahedral_die :
  let prob_1_digit := 9 / 20
  let prob_2_digit := 10 / 20
  let prob_3_digit := 1 / 20
  let expected_value := prob_1_digit * 1 + prob_2_digit * 2 + prob_3_digit * 3
  in expected_value = 1.6 := 
by
  have prob_1_digit : ℝ := 9 / 20
  have prob_2_digit : ℝ := 10 / 20
  have prob_3_digit : ℝ := 1 / 20
  have expected_value : ℝ := prob_1_digit * 1 + prob_2_digit * 2 + prob_3_digit * 3
  show expected_value = 1.6
  exact sorry

end expected_digits_icosahedral_die_l63_63492


namespace program_output_l63_63239

/-
If the input value of x is 351, then the output of the following program is 153.

Conditions:
1. 100 < x
2. x < 1000
3. x = 351

Steps:
a = x / 100
b = (x - a * 100) / 10
c = x % 10
x' = 100 * c + 10 * b + a

We need to prove that x' = 153 given the initial value x = 351 and the conditions 100 < x and x < 1000.
-/

theorem program_output {x: ℕ} (h1 : 100 < x) (h2 : x < 1000) (h3 : x = 351) : 
  let a := x / 100,
      b := (x - a * 100) / 10,
      c := x % 10,
      x' := 100 * c + 10 * b + a
  in x' = 153 :=
by
  sorry

end program_output_l63_63239


namespace distinct_sequences_example_l63_63625

theorem distinct_sequences_example :
  let letters := "EXAMPLE".toList
  let is_valid_sequence (s : List Char) : Bool :=
    s.head? = some 'X' ∧ (s.reverse.head? ≠ some 'E') ∧ s.length = 4 ∧ s.nodup
  (letters.permutations.lift (List.filter is_valid_sequence)).length = 80 := 
by sorry

end distinct_sequences_example_l63_63625


namespace max_age_for_16_viewers_l63_63111

def maximum_group_age (n : ℕ) (total_age : ℕ) (unique_ages : ℕ → Prop) : ℕ :=
if (n = 50 ∧ total_age = 1555 ∧ ∀ i j, i ≠ j → unique_ages i ≠ unique_ages j) 
then 776 
else 0

theorem max_age_for_16_viewers 
  (ages : Fin 50 → ℕ) 
  (h_unique : Function.Injective ages) 
  (h_sum : ∑ i, ages i = 1555) : 
  ∃ k, k = 776 ∧ ∃ s : Finset (Fin 50), s.card = 16 ∧ ∑ i in s, ages i ≥ k :=
begin
  use 776,
  intro s,
  have : s.card = 16 := sorry,
  use s,
  split,
  { exact this },
  { sorry }
end

end max_age_for_16_viewers_l63_63111


namespace ratio_of_segments_in_parallelogram_l63_63276

theorem ratio_of_segments_in_parallelogram
  (A B C D E F P Q R : Point)
  (h_parallelogram : parallelogram A B C D)
  (h_circumcircle : is_circumcircle (triangle A B D) Γ)
  (h_E : E ∈ line_through B C ∧ E ∈ Γ)
  (h_F : F ∈ line_through D C ∧ F ∈ Γ)
  (h_P : P = intersection_of_lines (line_through E D) (line_through B A))
  (h_Q : Q = intersection_of_lines (line_through F B) (line_through D A))
  (h_R : R = intersection_of_lines (line_through P Q) (line_through C A)) :
  (segment_ratio P R Q R) = (segment_ratio B C D C) ^ 2 :=
sorry

end ratio_of_segments_in_parallelogram_l63_63276


namespace smallest_lcm_value_theorem_l63_63652

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end smallest_lcm_value_theorem_l63_63652


namespace siding_cost_l63_63334

noncomputable def wall_area (w h : ℕ) : ℕ := w * h
noncomputable def roof_area (l w : ℕ) : ℕ := l * w
noncomputable def total_roof_area (l w : ℕ) (n : ℕ) : ℕ := (roof_area l w) * n
noncomputable def total_area (wall_area roof_area : ℕ) : ℕ := wall_area + roof_area
noncomputable def siding_area (w h : ℕ) : ℕ := w * h
noncomputable def number_of_sections (total_area siding_area : ℕ) : ℕ :=
  if total_area % siding_area = 0
  then total_area / siding_area
  else (total_area / siding_area) + 1
noncomputable def total_cost (sections : ℕ) (cost : ℕ) : ℕ := sections * cost

theorem siding_cost :
  let wall_width := 10
  let wall_height := 8
  let roof_length := 10
  let roof_width := 6
  let number_of_roof_sections := 2
  let siding_width := 8
  let siding_height := 12
  let section_cost := 2730
  let wallA := wall_area wall_width wall_height
  let roofA := total_roof_area roof_length roof_width number_of_roof_sections
  let totalA := total_area wallA roofA
  let sidingA := siding_area siding_width siding_height
  let sections_needed := number_of_sections totalA sidingA
  let totalC := total_cost sections_needed section_cost
  totalC = 8190 :=
by
  -- Simplification of given expressions
  let wallA := wall_area wall_width wall_height
  let roofA := total_roof_area roof_length roof_width number_of_roof_sections
  let totalA := total_area wallA roofA
  let sidingA := siding_area siding_width siding_height
  let sections_needed := number_of_sections totalA sidingA
  have sections_needed_is_3 : sections_needed = 3 := sorry
  have totalC := total_cost 3 section_cost
  show totalC = 8190, from sorry

end siding_cost_l63_63334


namespace correct_propositions_l63_63475

-- Proving that if every line in a plane is parallel to another plane and if two intersecting lines in a plane
-- are parallel to another plane, then the two planes are parallel.

theorem correct_propositions (P Q : Type) [plane P] [plane Q]
  (cond1 : ∀ (l : line_in P), parallel_line_plane l Q)
  (cond2 : (∃ (l m : line_in P), intersect_lines l m ∧ parallel_line_plane l Q ∧ parallel_line_plane m Q))
  : parallel_planes P Q :=
by
  sorry -- Proof is omitted.

end correct_propositions_l63_63475


namespace find_n_for_quadratic_l63_63155

theorem find_n_for_quadratic (a b c m n p : ℕ) (h1 : a = 3) (h2 : b = -7) (h3 : c = 1)
  (h_eq : 3 * m + 7 * n + c = 0)
  (h_gcd : Int.gcd (Int.natAbs m) (Int.gcd (Int.natAbs n) (Int.natAbs p)) = 1) :
  n = 37 :=
by
  -- We assume the values for the purpose of building a valid theorem statement
  have ha : a = 3 := h1
  have hb : b = -7 := h2
  have hc : c = 1 := h3
    
  -- The standard form of the quadratic equation roots
  have h_roots : ∀ x, 3 * x * x - 7 * x + 1 = 0 ↔ x = (7 + ⟩ (37 : ℕ))
    := sorry
    
  -- Conclusion based on the shape of the roots and gcd condition
  have h_gcd' : Int.gcd 7 (Int.gcd 37 6) = 1 := sorry

  -- Finally, deduce the value of n
  exact eq.trans h_eq h_roots

end find_n_for_quadratic_l63_63155


namespace find_y_minus_x_l63_63292

theorem find_y_minus_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : Real.sqrt x + Real.sqrt y = 1) 
  (h5 : Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3) : 
  y - x = 1 / 2 :=
sorry

end find_y_minus_x_l63_63292


namespace translate_function_l63_63403

theorem translate_function :
  ∀ (x : ℝ), ((λ x, 2 * x^2) x + 3) = (λ x, 2 * (x + 1)^2 + 3) x := 
by
  sorry

end translate_function_l63_63403


namespace average_speed_of_car_l63_63040

noncomputable def average_speed_round_trip (d: ℝ) (v₁ v₂: ℝ) : ℝ :=
  2 * d / ((d / v₁) + (d / v₂))

theorem average_speed_of_car (d : ℝ) (h₀ : d > 0) (speed_AB : 60) (speed_BA : 40) :
  average_speed_round_trip d speed_AB speed_BA = 48 :=
by
  sorry

end average_speed_of_car_l63_63040


namespace radii_equal_and_proportional_l63_63741

-- Define the circles and the properties
variable {O O1 O2 O3 O4 : Type}
variable [circle O] [circle O1] [circle O2] [circle O3] [circle O4]

-- Define the geometric properties
variable (A B M P Q : Point)
variable (diamO : AB = diameter O)
variable (M_on_AB : M ∈ diameter O)
variable (perpendicularPQ : PQ ⊥ AB)
variable (diamO1 : AM = diameter O1)
variable (diamO2 : BM = diameter O2)
variable (tangent_O3_O : tangent O3 O)
variable (tangent_O4_O : tangent O4 O)
variable (tangent_O3_PQ : tangent O3 PQ)
variable (tangent_O4_PQ : tangent O4 PQ)
variable (tangent_O3_O1 : tangent O3 O1)
variable (tangent_O4_O2 : tangent O4 O2)

-- Declare the theorem
theorem radii_equal_and_proportional :
  (radii O3 = radii O4) ∧
  (∃ k, radii O1 = k * radii O ∧ radii O2 = k * radii O1 ∧ radii O3 = k * radii O2) :=
by
  sorry

end radii_equal_and_proportional_l63_63741


namespace limit_expression_l63_63187

variable (f : ℝ → ℝ) (x₀ : ℝ)

def f_derivative_at_x₀_eq_two : Prop :=
  deriv f x₀ = 2

theorem limit_expression (h : f_derivative_at_x₀_eq_two f x₀) :
  (tendsto (λ k : ℝ, (f (x₀ - k) - f x₀) / (2 * k)) (nhds 0) (nhds (-1))) := sorry

end limit_expression_l63_63187


namespace supremum_of_expression_l63_63474

noncomputable def is_supremum (x : ℝ) (f : ℝ → ℝ) :=
  ∀ y : ℝ, (∀ z : ℝ, f z ≤ y) → x ≤ y

theorem supremum_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  is_supremum (-9/2) (λ x, - (1 / (2 * a) + 2 / b)) :=
sorry

end supremum_of_expression_l63_63474


namespace sum_of_digits_mod_8_l63_63937

def S (n : ℕ) : ℕ :=
  -- definition of the sum of the digits
  sorry

theorem sum_of_digits_mod_8 (n : ℕ) (h1 : S(n) = 365) (h2 : n % 8 = 5) :
    (S(n + 5) = 370) :=
  -- proof is to be filled here
  sorry

end sum_of_digits_mod_8_l63_63937


namespace wheel_speed_is_12_mph_l63_63735

theorem wheel_speed_is_12_mph
  (r : ℝ) -- speed in miles per hour
  (C : ℝ := 15 / 5280) -- circumference in miles
  (H1 : ∃ t, r * t = C * 3600) -- initial condition that speed times time for one rotation equals 15/5280 miles in seconds
  (H2 : ∃ t, (r + 7) * (t - 1/21600) = C * 3600) -- condition that speed increases by 7 mph when time shortens by 1/6 second
  : r = 12 :=
sorry

end wheel_speed_is_12_mph_l63_63735


namespace gcd_72_120_168_l63_63376

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
by
  sorry

end gcd_72_120_168_l63_63376


namespace glucose_solution_volume_l63_63448

theorem glucose_solution_volume (V : ℕ) (h : 500 / 10 = V / 20) : V = 1000 :=
sorry

end glucose_solution_volume_l63_63448


namespace card_pairs_sum_divisible_by_100_l63_63572

theorem card_pairs_sum_divisible_by_100 :
  let n := 2117 in
  let pairs := (Finset.range (n + 1)).powersetLen 2 in
  let filter_pairs := pairs.filter (λ p, (p.sum % 100 = 0)) in
  filter_pairs.card = 23058 := 
by
  -- Definitions for conditions
  let n := 2117
  let pairs := (Finset.range (n + 1)).powersetLen 2
  let filter_pairs := pairs.filter (λ p, (p.sum % 100 = 0))
  
  -- Final assertion
  exact filter_pairs.card = 23058

end card_pairs_sum_divisible_by_100_l63_63572


namespace binary_10011_is_19_l63_63951

def bin_to_dec (bin : List ℕ) : ℕ :=
  bin.reverse.foldl (λ acc x, acc * 2 + x) 0

theorem binary_10011_is_19 : bin_to_dec [1, 0, 0, 1, 1] = 19 :=
by
  sorry

end binary_10011_is_19_l63_63951


namespace three_pow_two_digits_count_l63_63229

theorem three_pow_two_digits_count : 
  ∃ n_set : Finset ℕ, (∀ n ∈ n_set, 10 ≤ 3^n ∧ 3^n < 100) ∧ n_set.card = 2 := 
sorry

end three_pow_two_digits_count_l63_63229


namespace problem1_problem2_l63_63600

-- Definition of the function
def f (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - Real.log x - 2

-- (1) Tangent line at the point (1, f(1)) when a = 1
def tangent_line_at_one : Prop :=
  let a := 1
  let f' (x : ℝ) : ℝ := x - 1 / x
  f' 1 = 0 ∧
  f a 1 = -3/2 ∧
  (∀ (x : ℝ), (f a 1 + (x - 1) * f' 1 = -3/2))

-- (2) Monotonicity
def monotonicity (a : ℝ) : Prop :=
  let f' (x : ℝ) : ℝ := (a * x^2 - 1) / x
  if a ≤ 0 then
    ∀ (x : ℝ), x > 0 → f' x < 0
  else
    (∀ (x : ℝ), 0 < x ∧ x < Real.sqrt a / a → f' x < 0) ∧
    (∀ (x : ℝ), x > Real.sqrt a / a → f' x > 0)

-- Statements
theorem problem1 : tangent_line_at_one := by
  sorry

theorem problem2 (a : ℝ) : monotonicity a := by
  sorry

end problem1_problem2_l63_63600


namespace probability_sum_is_prime_l63_63398

theorem probability_sum_is_prime :
  (∃ (d1 d2 d3 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧
  (d1 + d2 + d3 = 3 ∨ d1 + d2 + d3 = 5 ∨ d1 + d2 + d3 = 7 ∨ d1 + d2 + d3 = 11 ∨ d1 + d2 + d3 = 13 ∨ d1 + d2 + d3 = 17)) →
  (∃ p, p = (64/216 : ℚ) ∧ p = (8/27 : ℚ)) :=
begin
  sorry
end

end probability_sum_is_prime_l63_63398


namespace area_centroid_quadrilateral_l63_63350

-- Definitions based on the conditions
noncomputable def side_length : ℝ := 40
noncomputable def AQ : ℝ := 16
noncomputable def BQ : ℝ := 34

-- Statement to prove
theorem area_centroid_quadrilateral :
  ∃ G1 G2 G3 G4 : ℝ × ℝ, -- centroids of triangles ABQ, BCQ, CDQ, DAQ
  (G1, G2, G3, G4) ∈ set_of_centroids (triangle ABQ) (triangle BCQ) (triangle CDQ) (triangle DAQ) ∧
  area_of_quadrilateral G1 G2 G3 G4 = 355.56 :=
sorry

end area_centroid_quadrilateral_l63_63350


namespace solve_fractional_equation_l63_63789

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 6 * x - 7) = (3 - x) / (x - 1) ↔ x = -5 ∨ x = 3 :=
sorry

end solve_fractional_equation_l63_63789


namespace three_n_equals_27_implies_n_equals_9_l63_63660

theorem three_n_equals_27_implies_n_equals_9 (n : ℕ) (h : 3 * n = 27) : n = 9 :=
begin
  sorry
end

end three_n_equals_27_implies_n_equals_9_l63_63660


namespace product_of_number_and_sum_of_digits_l63_63259

-- Definitions according to the conditions
def units_digit (a b : ℕ) : Prop := b = a + 2
def number_equals_24 (a b : ℕ) : Prop := 10 * a + b = 24

-- The main statement to prove the product of the number and the sum of its digits
theorem product_of_number_and_sum_of_digits :
  ∃ (a b : ℕ), units_digit a b ∧ number_equals_24 a b ∧ (24 * (a + b) = 144) :=
sorry

end product_of_number_and_sum_of_digits_l63_63259


namespace cot_neg_45_l63_63130

-- Define the conditions
lemma cot_def (x : ℝ) : Real.cot x = 1 / Real.tan x := sorry
lemma tan_neg (x : ℝ) : Real.tan (-x) = -Real.tan x := sorry
lemma tan_45 : Real.tan (Real.pi / 4) = 1 := sorry

-- State the theorem to prove
theorem cot_neg_45 : Real.cot (-Real.pi / 4) = -1 :=
by
  apply cot_def
  apply tan_neg
  apply tan_45
  sorry

end cot_neg_45_l63_63130


namespace boat_speed_still_water_l63_63903

theorem boat_speed_still_water (V_b : ℝ) 
  (downstream_time : ℝ := 3) 
  (upstream_time : ℝ := 4.5)
  (current_start_speed : ℝ := 2)
  (current_end_speed : ℝ := 4)
  (wind_slowdown : ℝ := 1) :
  let average_current_speed := (current_start_speed + current_end_speed) / 2,
      downstream_effective_speed := V_b + average_current_speed - wind_slowdown,
      upstream_effective_speed := V_b - average_current_speed - wind_slowdown,
      D_downstream := downstream_effective_speed * downstream_time,
      D_upstream := upstream_effective_speed * upstream_time
  in D_downstream = D_upstream → V_b = 16 :=
begin
  intros h,
  sorry
end

end boat_speed_still_water_l63_63903


namespace range_of_a_l63_63667

def is_in_third_quadrant (A : ℝ × ℝ) : Prop :=
  A.1 < 0 ∧ A.2 < 0

theorem range_of_a (a : ℝ) (h : is_in_third_quadrant (a, a - 1)) : a < 0 :=
by
  sorry

end range_of_a_l63_63667


namespace max_children_l63_63053

/-- Total quantities -/
def total_apples : ℕ := 55
def total_cookies : ℕ := 114
def total_chocolates : ℕ := 83

/-- Leftover quantities after distribution -/
def leftover_apples : ℕ := 3
def leftover_cookies : ℕ := 10
def leftover_chocolates : ℕ := 5

/-- Distributed quantities -/
def distributed_apples : ℕ := total_apples - leftover_apples
def distributed_cookies : ℕ := total_cookies - leftover_cookies
def distributed_chocolates : ℕ := total_chocolates - leftover_chocolates

/-- The theorem states the maximum number of children -/
theorem max_children : Nat.gcd (Nat.gcd distributed_apples distributed_cookies) distributed_chocolates = 26 :=
by
  sorry

end max_children_l63_63053


namespace correct_inequality_l63_63287

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f(-x) = f(x))  -- Even function property
variable (h2 : ∀ x y, x < y → x ≤ -1 → y ≤ -1 → f(x) < f(y))  -- Increasing property on (-∞, -1]

theorem correct_inequality : f (-real.sqrt 2) > f (real.sqrt 3) :=
by
  sorry

end correct_inequality_l63_63287


namespace probability_of_two_pairs_of_same_value_is_correct_l63_63343

def total_possible_outcomes := 6^6
def number_of_ways_to_form_pairs := 15
def choose_first_pair := 6
def choose_second_pair := 15
def choose_third_pair := 6
def choose_fourth_die := 4
def choose_fifth_die := 3

def successful_outcomes := number_of_ways_to_form_pairs *
                           choose_first_pair *
                           choose_second_pair *
                           choose_third_pair *
                           choose_fourth_die *
                           choose_fifth_die

def probability_of_two_pairs_of_same_value := (successful_outcomes : ℚ) / total_possible_outcomes

theorem probability_of_two_pairs_of_same_value_is_correct :
  probability_of_two_pairs_of_same_value = 25 / 72 :=
by
  -- proof omitted
  sorry

end probability_of_two_pairs_of_same_value_is_correct_l63_63343


namespace slope_and_angle_of_inclination_l63_63388

noncomputable def line_slope_and_inclination : Prop :=
  ∀ (x y : ℝ), (x - y - 3 = 0) → (∃ m : ℝ, m = 1) ∧ (∃ θ : ℝ, θ = 45)

theorem slope_and_angle_of_inclination (x y : ℝ) (h : x - y - 3 = 0) : line_slope_and_inclination :=
by
  sorry

end slope_and_angle_of_inclination_l63_63388


namespace fibonacci_a7_is_13_sum_first_2016_terms_l63_63800

def Fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => Fibonacci (n + 1) + Fibonacci n

theorem fibonacci_a7_is_13 : Fibonacci 7 = 13 := by
  sorry

theorem sum_first_2016_terms (k : ℕ) (h : Fibonacci 2018 = k) :
  (∑ i in Finset.range 2016, Fibonacci (i + 1)) = k - 1 := by
  sorry

end fibonacci_a7_is_13_sum_first_2016_terms_l63_63800


namespace find_num_teachers_l63_63917

noncomputable def num_teachers (total_population : ℕ) (sample_size : ℕ) (sampled_students : ℕ) : ℕ :=
  let x := ((sample_size * total_population) - (sampled_students * total_population))
           / (sample_size - sampled_students) in x

theorem find_num_teachers (total_population sample_size sampled_students : ℕ) 
    (H_total : total_population = 2400)
    (H_sample_size : sample_size = 150)
    (H_sampled_students : sampled_students = 135) :
  num_teachers total_population sample_size sampled_students = 240 :=
by
  sorry

end find_num_teachers_l63_63917


namespace max_expression_value_l63_63541

theorem max_expression_value
  (a b c x y z : ℝ)
  (h1 : 2 ≤ a ∧ a ≤ 3)
  (h2 : 2 ≤ b ∧ b ≤ 3)
  (h3 : 2 ≤ c ∧ c ≤ 3)
  (hx : x ∈ {a, b, c})
  (hy : y ∈ {a, b, c})
  (hz : z ∈ {a, b, c})
  (h_perm : \[x, y, z\] = [a, b, c])
  : (a / x) + ((a + b) / (x + y)) + ((a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_expression_value_l63_63541


namespace unique_solution_l63_63963

noncomputable def equation_satisfied (x : ℝ) : Prop :=
  2^x + 3^x + 6^x = 7^x

theorem unique_solution : ∀ x : ℝ, equation_satisfied x ↔ x = 2 := by
  sorry

end unique_solution_l63_63963


namespace isosceles_triangle_vertex_angle_l63_63006

theorem isosceles_triangle_vertex_angle (a b : ℝ) (α : ℝ) :
  (2 * Real.arcsin (Real.sqrt 2 - 1) < α ∧ α < Real.pi) ↔
  (isosceles_triangle a b α ∧ exactly_three_bisecting_lines a b α) := 
sorry

end isosceles_triangle_vertex_angle_l63_63006


namespace pulley_distance_l63_63933

-- Define the radii of the pulleys
def r1 : ℝ := 10
def r2 : ℝ := 6

-- Define the distance between the belt contact points
def d : ℝ := 30

-- Define the difference in radii
def delta_r : ℝ := r1 - r2

-- State the theorem
theorem pulley_distance : sqrt (d^2 + delta_r^2) = sqrt 916 := by
  sorry

end pulley_distance_l63_63933


namespace camp_cedar_counselors_l63_63944

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) 
(counselors_for_boys : ℕ) (counselors_for_girls : ℕ) 
(total_counselors : ℕ) 
(h1 : boys = 80)
(h2 : girls = 6 * boys - 40)
(h3 : counselors_for_boys = boys / 5)
(h4 : counselors_for_girls = (girls + 11) / 12)  -- +11 to account for rounding up
(h5 : total_counselors = counselors_for_boys + counselors_for_girls) : 
total_counselors = 53 :=
by
  sorry

end camp_cedar_counselors_l63_63944


namespace pentagon_area_l63_63320

theorem pentagon_area (P Q R S T : Point)
  (h1 : is_square P Q R S)
  (h2 : is_perpendicular P T R)
  (h3 : distance P T = 5)
  (h4 : distance T R = 12) :
  area_of_pentagon P T R S Q = 139 := by
  sorry

end pentagon_area_l63_63320


namespace perpendicular_t_l63_63621

def vector_perpendicular := ∀ (a b : ℝ × ℝ), (a.1 * b.1 + a.2 * b.2 = 0)

variable (t : ℝ)
variable (a b : ℝ × ℝ)
variable ha : a = (1, -1)
variable hb : b = (6, -4)

theorem perpendicular_t (ht : vector_perpendicular a (t • a + b)) : t = -5 := by
  sorry

end perpendicular_t_l63_63621


namespace minimum_broken_lines_cover_all_vertices_l63_63949

-- Define the grid dimensions
def gridWidth : ℕ := 100
def gridHeight : ℕ := 100

-- Define the properties of a shortest path from one corner to the other in the grid
def shortestPath(c1 c2 : ℕ × ℕ) : Prop :=
  c1.2 = 0 ∧ c2.1 = gridWidth ∧ ∀ i, (i < gridWidth → (i+1, i) = (i, i+1)) ∧

-- Define a function that checks if a vertex is covered by a path
def isVertexCovered (p : array (ℕ × ℕ)) (v : ℕ × ℕ) : Prop :=
  ∃ i, p.get? i = some v

-- Define the statement to be proven
theorem minimum_broken_lines_cover_all_vertices :
  ∃ n : ℕ, n = 101 ∧ ∀ v : ℕ × ℕ, v.1 ≤ gridWidth ∧ v.2 ≤ gridHeight → 
    (∃ paths : array (array (ℕ × ℕ)), paths.size ≤ n ∧ 
    ∀ path, path ∈ paths → shortestPath (0,0) (gridWidth, gridHeight) ∧
            ∀ pathVertex v, isVertexCovered path pathVertex →
            isVertexCovered pathVertex v) := by
  sorry

end minimum_broken_lines_cover_all_vertices_l63_63949


namespace find_y_l63_63554

theorem find_y (y : ℝ) (h : sqrt (7 * y) / sqrt (4 * (y - 2)) = 3) : y = 72 / 29 :=
by
  sorry

end find_y_l63_63554


namespace color_n_gon_l63_63962

theorem color_n_gon (n : ℕ) (h₁ : n ≥ 5) : 
  (∃ (color : fin n → fin 6), ∀ (i : fin n), (∀ j : fin 5, color (i + j) ≠ color (i + j + 1))) ↔ 
  n ∉ {7, 8, 9, 13, 14, 19} :=
sorry

end color_n_gon_l63_63962


namespace smallest_number_of_cubes_l63_63883

def length : ℕ := 24
def width : ℕ := 40
def depth : ℕ := 16

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def cube_side_length : ℕ := gcd length (gcd width depth)

def cubes_in_length : ℕ := length / cube_side_length
def cubes_in_width : ℕ := width / cube_side_length
def cubes_in_depth : ℕ := depth / cube_side_length

def total_cubes : ℕ := cubes_in_length * cubes_in_width * cubes_in_depth

theorem smallest_number_of_cubes : total_cubes = 30 := by
  -- Proof to be filled in here
  sorry

end smallest_number_of_cubes_l63_63883


namespace shekar_english_marks_l63_63781

theorem shekar_english_marks (m s so b a e : ℕ) (h1 : m = 76) (h2 : s = 65) (h3 : so = 82) (h4 : b = 85) (h5 : a = 71) :
  e = 47 :=
by
  let t_o := m + s + so + b
  have h6 : t_o = 308 := by sorry
  let t_e := 5 * a
  have h7 : t_e = 355 := by sorry
  have h8 : e = t_e - t_o := by sorry
  show e = 47 from by
    calc
      e = t_e - t_o : h8
      ... = 355 - 308 : by congr; exact h7; exact h6
      ... = 47 : by norm_num

end shekar_english_marks_l63_63781


namespace sum_of_even_factors_900_l63_63950

theorem sum_of_even_factors_900 : 
  ∃ (S : ℕ), 
  (∀ a b c : ℕ, 900 = 2^a * 3^b * 5^c → 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → 0 ≤ c ∧ c ≤ 2) → 
  (∀ a : ℕ, 1 ≤ a ∧ a ≤ 2 → ∃ b c : ℕ, 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ (2^a * 3^b * 5^c = 900 ∧ a ≠ 0)) → 
  S = 2418 := 
sorry

end sum_of_even_factors_900_l63_63950


namespace non_intersecting_segments_l63_63892

theorem non_intersecting_segments (n : ℕ) (red_points blue_points : Fin n → ℝ × ℝ) :
  (∀ (i j k : Fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k → 
    collinear ({red_points i, red_points j, red_points k} : Set (ℝ × ℝ)) = false) →
  (∀ (i j k : Fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k → 
    collinear ({blue_points i, blue_points j, blue_points k} : Set (ℝ × ℝ)) = false) →
  ∃ f : Fin n → Fin n,
    (∀ i j, i ≠ j → disjoint({(red_points i, blue_points (f i))}, {(red_points j, blue_points (f j))})) := 
  sorry

end non_intersecting_segments_l63_63892


namespace cross_section_area_l63_63074

noncomputable def regular_triang_prism_cross_section_area (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 19) / 16

theorem cross_section_area (a : ℝ) :
  (∀ (a : ℝ), a > 0) →
  regular_triang_prism_cross_section_area a = (3 * a^2 * Real.sqrt 19) / 16 :=
by {
  intro ha,
  unfold regular_triang_prism_cross_section_area,
  sorry -- proof will be inserted here
}

end cross_section_area_l63_63074


namespace sum_of_coeffs_expansion_l63_63524

theorem sum_of_coeffs_expansion (d : ℝ) : 
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
in coeff_sum = -21 := 
by
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
    exact rfl

end sum_of_coeffs_expansion_l63_63524


namespace trapezoid_area_correct_l63_63008

-- Definitions for vertices of the trapezoid
def A : ℝ × ℝ := (1, -3)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (7, 9)
def D : ℝ × ℝ := (7, 0)

-- Function to calculate the distance between two points with the same x-coordinate
def vertical_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  |p1.2 - p2.2|

-- Function to calculate the distance between two points with the same y-coordinate
def horizontal_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  |p1.1 - p2.1|

-- Function to calculate the area of a trapezoid given two base lengths and height
def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  (b1 + b2) * h / 2

-- Calculate the lengths of the bases
def base_length_AB : ℝ := vertical_distance A B
def base_length_CD : ℝ := vertical_distance C D

-- Calculate the height of the trapezoid
def height_AD : ℝ := horizontal_distance A D

-- Prove that the area of the trapezoid is 42 square units
theorem trapezoid_area_correct :
  trapezoid_area base_length_AB base_length_CD height_AD = 42 :=
  sorry

end trapezoid_area_correct_l63_63008


namespace monotonic_increase_interval_l63_63377

noncomputable def f (x : ℝ) : ℝ := -(1/3) * x^3 + x^2 + 3 * x

theorem monotonic_increase_interval : ∀ x : ℝ, x ∈ Set.Ioc (-1 : ℝ) (3 : ℝ) → f' x > 0 := by
  sorry

end monotonic_increase_interval_l63_63377


namespace min_translation_value_l63_63375

theorem min_translation_value (φ : ℝ) (h1 : φ > 0)
    (h2 : ∀ x, sin (2 * (x + φ)) = sin (2 * x + 2 * φ))
    (h3 : sin (2 * (π / 6 + φ)) = sqrt 3 / 2) : φ = π / 6 :=
  sorry

end min_translation_value_l63_63375


namespace rope_remaining_length_after_six_cuts_l63_63063

theorem rope_remaining_length_after_six_cuts :
  ∀ n, n = 6 → let initial_length := 1 in let cut_fraction := (2 / 3)^n in initial_length * cut_fraction = (2 / 3)^6 :=
by
  intro n hn
  rw [hn]
  let initial_length := 1
  let cut_fraction := (2 / 3)^6
  show initial_length * cut_fraction = (2 / 3)^6
  sorry

end rope_remaining_length_after_six_cuts_l63_63063


namespace standard_deviation_is_2_l63_63829

open Real

def dataset : List ℝ := [5, 7, 7, 8, 10, 11]

def mean (l : List ℝ) : ℝ := (l.sum / l.length)

def variance (l : List ℝ) : ℝ := 
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

def stddev (l : List ℝ) : ℝ := sqrt (variance l)

theorem standard_deviation_is_2 : stddev dataset = 2 :=
sorry

end standard_deviation_is_2_l63_63829


namespace greatest_possible_average_speed_l63_63085

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem greatest_possible_average_speed :
  ∀ (o₁ o₂ : ℕ) (v_max t : ℝ), 
  is_palindrome o₁ → 
  is_palindrome o₂ → 
  o₁ = 12321 → 
  t = 2 ∧ v_max = 65 → 
  (∃ d, d = o₂ - o₁ ∧ d / t <= v_max) → 
  d / t = v_max :=
sorry

end greatest_possible_average_speed_l63_63085


namespace prob_at_least_one_head_is_7_over_8_l63_63911

-- Define the event and probability calculation
def probability_of_tails_all_three_tosses : ℚ :=
  (1 / 2) ^ 3

def probability_of_at_least_one_head : ℚ :=
  1 - probability_of_tails_all_three_tosses

-- Prove the probability of at least one head is 7/8
theorem prob_at_least_one_head_is_7_over_8 : probability_of_at_least_one_head = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_is_7_over_8_l63_63911


namespace product_of_integer_with_100_l63_63444

theorem product_of_integer_with_100 (x : ℝ) (h : 10 * x = x + 37.89) : 100 * x = 421 :=
by
  -- insert the necessary steps to solve the problem
  sorry

end product_of_integer_with_100_l63_63444


namespace james_profit_l63_63706

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l63_63706


namespace ratio_of_perimeters_l63_63871

noncomputable def sqrt2 : ℝ := Real.sqrt 2

theorem ratio_of_perimeters (d1 : ℝ) :
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2 
  (P2 / P1 = 1 + sqrt2) :=
by
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2
  sorry

end ratio_of_perimeters_l63_63871


namespace yellow_tickets_needed_l63_63839

def yellow_from_red (r : ℕ) : ℕ := r / 10
def red_from_blue (b : ℕ) : ℕ := b / 10
def blue_needed (current_blue : ℕ) (additional_blue : ℕ) : ℕ := current_blue + additional_blue
def total_blue_from_tickets (y : ℕ) (r : ℕ) (b : ℕ) : ℕ := (y * 10 * 10) + (r * 10) + b

theorem yellow_tickets_needed (y r b additional_blue : ℕ) (h : total_blue_from_tickets y r b + additional_blue = 1000) :
  yellow_from_red (red_from_blue (total_blue_from_tickets y r b + additional_blue)) = 10 := 
by
  sorry

end yellow_tickets_needed_l63_63839


namespace remainder_when_27_pow_27_plus_27_div_28_l63_63422

theorem remainder_when_27_pow_27_plus_27_div_28:
  (27^27 + 27) % 28 = 26 :=
by
  have h1 : 27 % 28 = -1 % 28 := by sorry
  have h2 : (-1)^27 = -1 := by sorry
  calc
    (27^27 + 27) % 28 = ((-1)^27 + 27) % 28 := by sorry
    ... = (-1 + 27) % 28 := by sorry
    ... = 26 % 28 := by sorry
    ... = 26 := by sorry

end remainder_when_27_pow_27_plus_27_div_28_l63_63422


namespace determine_complex_number_l63_63101

noncomputable def z : ℂ := -3 + ((24 : ℝ) / 7 : ℂ) * complex.I

theorem determine_complex_number (z : ℂ) (h : 3 * z - 4 * complex.conj z = 3 + 24 * complex.I) :
  z = -3 + ((24 : ℝ) / 7 : ℂ) * complex.I :=
sorry

end determine_complex_number_l63_63101


namespace pages_needed_l63_63415

theorem pages_needed 
  (new_cards : ℕ) 
  (old_cards : ℕ) 
  (cards_per_page : ℕ) 
  (cards_total : ℕ := new_cards + old_cards) 
  (pages : ℕ := cards_total / cards_per_page) : 
  new_cards = 8 → old_cards = 10 → cards_per_page = 3 → pages = 6 :=
by
  intros hnew hold hcpage
  rw [hnew, hold, hcpage]
  define cards_total := 8 + 10
  define pages := cards_total / 3
  norm_num
  done
  sorry

end pages_needed_l63_63415


namespace fraction_of_monkeys_is_simplified_l63_63945

def initial_counts := {monkeys := 6, birds := 9, squirrels := 3, cats := 5}

def resultant_counts (initial_counts) :=
  let monkeys := initial_counts.monkeys
  let birds := initial_counts.birds - 2 - 2 + 4
  let squirrels := nat.div (initial_counts.squirrels) 2 - 1
  let cats := initial_counts.cats + 2
  let total := monkeys + birds + squirrels + cats
  { monkeys := monkeys, total := total }

theorem fraction_of_monkeys_is_simplified :
  let counts := resultant_counts initial_counts in
  counts.monkeys / counts.total = 3 / 11 :=
by sorry

end fraction_of_monkeys_is_simplified_l63_63945


namespace annual_interest_rate_is_8_percent_l63_63795

noncomputable def principal (A I : ℝ) : ℝ := A - I

noncomputable def annual_interest_rate (A P n t : ℝ) : ℝ :=
(A / P)^(1 / (n * t)) - 1

theorem annual_interest_rate_is_8_percent :
  let A := 19828.80
  let I := 2828.80
  let P := principal A I
  let n := 1
  let t := 2
  let r := annual_interest_rate A P n t
  r * 100 = 8 :=
by
  let A := 19828.80
  let I := 2828.80
  let P := principal A I
  let n := 1
  let t := 2
  let r := annual_interest_rate A P n t
  sorry

end annual_interest_rate_is_8_percent_l63_63795


namespace general_term_of_c_l63_63612

theorem general_term_of_c (a b : ℕ → ℕ) (c : ℕ → ℕ) : 
  (∀ n, a n = 2 ^ n) →
  (∀ n, b n = 3 * n + 2) →
  (∀ n, ∃ m k, a n = b m ∧ n = 2 * k + 1 → c k = a n) →
  ∀ n, c n = 2 ^ (2 * n + 1) :=
by
  intros ha hb hc n
  have h' := hc n
  sorry

end general_term_of_c_l63_63612


namespace probability_real_number_l63_63323

noncomputable def rational_numbers_in_interval (n d : ℕ) : Prop := 
  1 ≤ d ∧ d ≤ 5 ∧ 0 ≤ (n : ℚ) / d ∧ (n : ℚ) / d < 2

theorem probability_real_number (a b : ℚ) (ha : ∃ n d, rational_numbers_in_interval n d ∧ a = n / d)
(bh : ∃ n d, rational_numbers_in_interval n d ∧ b = n / d) : 
  (∃ p q : ℕ, rational_numbers_in_interval p q ∧ a = p / q) ∧ (∃ p q : ℕ, rational_numbers_in_interval p q ∧ b = p / q) → 
  (↑(finset.card {ab | let ⟨a, b⟩ := ab in (cos (a * real.pi) + real.sin (b * real.pi) * complex.I) ^ 6}.filter (λ ab, (cos (prod.fst ab * real.pi) + real.sin (prod.snd ab * real.pi) * complex.I) ^ 6 ∈ real)) 
  / 400 = 17 / 100) :=
sorry

end probability_real_number_l63_63323


namespace max_value_of_expr_l63_63549

noncomputable def max_value_expr (a b c x y z : ℝ) : ℝ :=
  (a / x) + (a + b) / (x + y) + (a + b + c) / (x + y + z)

theorem max_value_of_expr {a b c x y z : ℝ} 
  (ha : 2 ≤ a ∧ a ≤ 3) (hb : 2 ≤ b ∧ b ≤ 3) (hc : 2 ≤ c ∧ c ≤ 3)
  (hperm : {x, y, z} = {a, b, c}) : 
  max_value_expr a b c x y z ≤ 15 / 4 :=
by
  sorry

end max_value_of_expr_l63_63549


namespace complex_fraction_simplification_l63_63338

noncomputable def complex_exponentiation : ℂ := (2 + complex.i) / (2 - complex.i)

theorem complex_fraction_simplification :
  (complex_exponentiation) ^ 150 = 1 :=
by
  sorry

end complex_fraction_simplification_l63_63338


namespace hyperbola_asymptotes_to_tangent_l63_63592

open Real

theorem hyperbola_asymptotes_to_tangent (a : ℝ) :
  (∀ x : ℝ, let tangent_line_pos := (2 / 3) * x,
            let tangent_line_neg := -(2 / 3) * x,
            let curve := a * x ^ 2  + (1 / 3),
            (curve = tangent_line_pos ∨ curve = tangent_line_neg)) →
  a = (1 / 3) := 
by
  sorry

end hyperbola_asymptotes_to_tangent_l63_63592


namespace find_m_values_l63_63214

def has_unique_solution (m : ℝ) (A : Set ℝ) : Prop :=
  ∀ x1 x2, x1 ∈ A → x2 ∈ A → x1 = x2

theorem find_m_values :
  {m : ℝ | ∃ A : Set ℝ, has_unique_solution m A ∧ (A = {x | m * x^2 + 2 * x + 3 = 0})} = {0, 1/3} :=
by
  sorry

end find_m_values_l63_63214


namespace findAnalyticalExpression_l63_63371

-- Defining the point A as a structure with x and y coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Defining a line as having a slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Condition: Line 1 is parallel to y = 2x - 3
def line1 : Line := {slope := 2, intercept := -3}

-- Condition: Line 2 passes through point A
def point_A : Point := {x := -2, y := -1}

-- The theorem statement:
theorem findAnalyticalExpression : 
  ∃ b : ℝ, (∀ x : ℝ, (point_A.y = line1.slope * point_A.x + b) → b = 3) ∧ 
            ∀ x : ℝ, (line1.slope * x + b = 2 * x + 3) :=
sorry

end findAnalyticalExpression_l63_63371


namespace colored_copies_count_l63_63493

theorem colored_copies_count :
  ∃ C W : ℕ, (C + W = 400) ∧ (10 * C + 5 * W = 2250) ∧ (C = 50) :=
by
  sorry

end colored_copies_count_l63_63493


namespace red_hair_is_10_l63_63805

variable (total_campers campers_brown campers_green campers_black campers_red : ℕ)

-- Definitions derived from the problem's conditions
def percentage_brown := 0.5 * total_campers
def red_hair := total_campers - (campers_brown + campers_green + campers_black)

-- Conditions
axiom h1 : campers_brown = 25
axiom h2 : campers_green = 10
axiom h3 : campers_black = 5
axiom h4 : 0.5 * total_campers = campers_brown

theorem red_hair_is_10 : red_hair = 10 :=
by
  have total_campers_val : total_campers = 50 := sorry,
  rw [total_campers_val],
  rw [h1, h2, h3],
  exact Eq.symm (show 10 = 10 from by rfl)

end red_hair_is_10_l63_63805


namespace cinema_max_value_k_l63_63113

theorem cinema_max_value_k :
  ∃ k, 
    k = 776 ∧ ∀ (ages : Fin 50 → ℕ), 
      (∀ i j, i ≠ j → ages i ≠ ages j) → 
      (∑ i, ages i = 1555) → 
      ∃ selected, 
        Multiset.card selected = 16 ∧ 
        Multiset.sum selected ≥ k :=
by
  sorry

end cinema_max_value_k_l63_63113


namespace modulus_of_complex_pure_imag_l63_63999

-- Define the necessary conditions and result
theorem modulus_of_complex_pure_imag (a : ℝ) (ha : let z := (a - 2 * complex.I) / (1 + complex.I) in ∀ r : ℝ, z = 0 + r * complex.I) :
  complex.abs (1 + a * complex.I) = real.sqrt 5 :=
begin
  sorry
end

end modulus_of_complex_pure_imag_l63_63999


namespace smallest_four_digit_divisible_by_5_and_9_with_even_digits_l63_63000

-- Definition of the conditions.
def even_digit (n : ℕ) : Prop := n % 2 = 0

def four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def divisible_by_5_and_9 (n : ℕ) : Prop := n % 5 = 0 ∧ n % 9 = 0

def has_only_even_digits (n : ℕ) : Prop := (∀ d : ℕ, d ∈ n.digits 10 → even_digit d)

-- The statement to prove
theorem smallest_four_digit_divisible_by_5_and_9_with_even_digits :
  ∃ n, four_digit n ∧ divisible_by_5_and_9 n ∧ has_only_even_digits n ∧
       (∀ m, four_digit m ∧ divisible_by_5_and_9 m ∧ has_only_even_digits m → n ≤ m) :=
begin
  use 2880,
  split,
  { sorry },  -- Proof that 2880 is a four-digit number
  split,
  { sorry },  -- Proof that 2880 is divisible by 5 and 9
  split,
  { sorry },  -- Proof that 2880 has only even digits
  { sorry }   -- Proof that 2880 is the smallest such number
end

end smallest_four_digit_divisible_by_5_and_9_with_even_digits_l63_63000


namespace minimize_expression_l63_63832

theorem minimize_expression : 
  ∃ x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ : ℕ, 
  (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₁ ≠ y₁) ∧ (x₁ ≠ y₂) ∧
  (x₁ ≠ y₃) ∧ (x₁ ≠ z₁) ∧ (x₁ ≠ z₂) ∧ (x₁ ≠ z₃) ∧
  (x₂ ≠ x₃) ∧ (x₂ ≠ y₁) ∧ (x₂ ≠ y₂) ∧ (x₂ ≠ y₃) ∧
  (x₂ ≠ z₁) ∧ (x₂ ≠ z₂) ∧ (x₂ ≠ z₃) ∧ (x₃ ≠ y₁) ∧
  (x₃ ≠ y₂) ∧ (x₃ ≠ y₃) ∧ (x₃ ≠ z₁) ∧ (x₃ ≠ z₂) ∧
  (x₃ ≠ z₃) ∧ (y₁ ≠ y₂) ∧ (y₁ ≠ y₃) ∧ (y₁ ≠ z₁) ∧
  (y₁ ≠ z₂) ∧ (y₁ ≠ z₃) ∧ (y₂ ≠ y₃) ∧ (y₂ ≠ z₁) ∧
  (y₂ ≠ z₂) ∧ (y₂ ≠ z₃) ∧ (y₃ ≠ z₁) ∧ (y₃ ≠ z₂) ∧
  (y₃ ≠ z₃) ∧ (z₁ ≠ z₂) ∧ (z₁ ≠ z₃) ∧ (z₂ ≠ z₃) ∧
  (List.perm [x₁, x₂, x₃, y₁, y₂, y₃, z₁, z₂, z₃] [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃ + x₁ * y₁ * z₁ = 884) := sorry

end minimize_expression_l63_63832


namespace num_integers_in_range_l63_63633

theorem num_integers_in_range : 
  let π_approx := 3.14 in
  let lower_bound := -6 * π_approx in
  let upper_bound := 12 * π_approx in
  let n_start := Int.ceil lower_bound in
  let n_end := Int.floor upper_bound in
  (n_end - n_start + 1) = 56 := 
by
  sorry

end num_integers_in_range_l63_63633


namespace weight_of_balls_l63_63427

theorem weight_of_balls (x y : ℕ) (h1 : 5 * x + 3 * y = 42) (h2 : 5 * y + 3 * x = 38) :
  x = 6 ∧ y = 4 :=
by
  sorry

end weight_of_balls_l63_63427


namespace fraction_simplification_l63_63341

theorem fraction_simplification : (98 / 210 : ℚ) = 7 / 15 := 
by 
  sorry

end fraction_simplification_l63_63341


namespace smallest_number_cathy_l63_63080

theorem smallest_number_cathy (Anne_number : ℕ) (Cathy_number : ℕ) 
  (h₀ : Anne_number = 36)
  (h₁ : ∀ p : ℕ, prime p → p ∣ Cathy_number → p ∣ Anne_number) 
  (h₂ : ∀ q : ℕ, prime q → q ∣ Anne_number → q ∣ Cathy_number → q = 2 ∨ q = 3) :
  Cathy_number = 6 := by
  sorry

end smallest_number_cathy_l63_63080


namespace integer_count_in_interval_l63_63629

theorem integer_count_in_interval : 
  let pi := Real.pi in
  let lower_bound := -6 * pi in
  let upper_bound := 12 * pi in
  ∃ (count : ℕ), count = 56 ∧ ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound ↔ (-18 ≤ n ∧ n ≤ 37) :=
by
  let pi := Real.pi
  let lower_bound := -6 * pi
  let upper_bound := 12 * pi
  use 56
  split
  · exact rfl
  · intro n
    split
    · intro h
      split
      · linarith
      · linarith
    · intro h
      split
      · linarith
      · linarith
  sorry

end integer_count_in_interval_l63_63629


namespace sum_of_isosceles_angles_l63_63872

noncomputable def isosceles_triangle (x : ℝ) : Prop :=
  let A := (Real.cos (π / 6), Real.sin (π / 6))
  let B := (0, 1)
  let C := (Real.cos x, Real.sin x)
  let distance (p q : ℝ × ℝ) := ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  distance A B = distance A C ∨ distance B A = distance B C ∨ distance C A = distance C B

theorem sum_of_isosceles_angles :
  ∑ x in {x | 0 ≤ x ∧ x ≤ 2 * π ∧ isosceles_triangle x}.to_finset, x = 7 * π / 3 :=
sorry

end sum_of_isosceles_angles_l63_63872


namespace count_integers_in_range_l63_63636

theorem count_integers_in_range : 
  { n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi }.finite.toFinset.card = 56 := 
by 
  sorry

end count_integers_in_range_l63_63636


namespace max_expression_value_l63_63538

theorem max_expression_value
  (a b c x y z : ℝ)
  (h1 : 2 ≤ a ∧ a ≤ 3)
  (h2 : 2 ≤ b ∧ b ≤ 3)
  (h3 : 2 ≤ c ∧ c ≤ 3)
  (hx : x ∈ {a, b, c})
  (hy : y ∈ {a, b, c})
  (hz : z ∈ {a, b, c})
  (h_perm : \[x, y, z\] = [a, b, c])
  : (a / x) + ((a + b) / (x + y)) + ((a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_expression_value_l63_63538


namespace line_equation_through_M_P_Q_l63_63173

-- Given that M is the midpoint between P and Q, we should have:
-- M = (1, -2)
-- P = (2, 0)
-- Q = (0, -4)
-- We need to prove that the line passing through these points has the equation 2x - y - 4 = 0

theorem line_equation_through_M_P_Q :
  ∀ (x y : ℝ), (1 - 2 = (2 * (x - 1)) ∧ 0 - 2 = (2 * (0 - (-2)))) ->
  (x - y - 4 = 0) := 
by
  sorry

end line_equation_through_M_P_Q_l63_63173


namespace extreme_values_and_sum_of_reciprocals_l63_63602

open Real -- Open the Real namespace for mathematical operations

theorem extreme_values_and_sum_of_reciprocals (n : ℕ) (h : n ≥ 2) :
  let f := λ x : ℝ, x * sin x + cos x in
  (let min_extreme := -3 * real.pi / 2 in 
   let max_extreme := real.pi / 2 in 
   (∀ x ∈ Icc (0 : ℝ) (2 * real.pi), 
     (f' x = 0) → 
     (x = real.pi / 2 → f x = max_extreme) 
     ∧ (x = 3 * real.pi / 2 → f x = min_extreme))) 
  ∧ 
  (let xi := λ i, (2 * i - 1) * real.pi / 2 in 
   ∑ i in range' 2 n, 1 / (xi i)^2 < 2 / 9) :=
sorry

end extreme_values_and_sum_of_reciprocals_l63_63602


namespace exists_permutation_with_large_neighbor_difference_l63_63423

theorem exists_permutation_with_large_neighbor_difference :
  ∃ (σ : Fin 100 → Fin 100), 
    (∀ (i : Fin 99), (|σ i.succ - σ i| ≥ 50)) :=
sorry

end exists_permutation_with_large_neighbor_difference_l63_63423


namespace cow_goat_goose_grass_eaten_in_40_days_l63_63049

-- Define the rates of Cow, Goat, and Goose
variable (C Gt Ge : ℝ)

-- Conversion conditions provided into Lean definitions
def condition1 : Prop := C + Gt = 1 / 45
def condition2 : Prop := C + Ge = 1 / 60
def condition3 : Prop := Gt + Ge = 1 / 90

-- The final theorem stating the answer
theorem cow_goat_goose_grass_eaten_in_40_days (h1 : condition1 C Gt Ge) 
                                               (h2 : condition2 C Gt Ge) 
                                               (h3 : condition3 C Gt Ge) : 
  (C + Gt + Ge) = 1 / 40 :=
  sorry  -- Proof is omitted

end cow_goat_goose_grass_eaten_in_40_days_l63_63049


namespace area_of_circle_above_line_l63_63408

noncomputable def circle_area_above_line (x y : ℝ) : ℝ := 
  if (x^2 - 4 * x + y^2 - 8 * y + 12 = 0) ∧ (y > 4) then
    4 * real.pi
  else
    0

theorem area_of_circle_above_line :
  (∃ x y : ℝ, x^2 - 4 * x + y^2 - 8 * y + 12 = 0 ∧ y > 4) →
  circle_area_above_line 0 5 = 4 * real.pi :=
by sorry

end area_of_circle_above_line_l63_63408


namespace arctan_tan_75_minus_3tan_30_minus_tan_15_eq_60_l63_63495

noncomputable def tan_30_deg : ℝ := 1 / real.sqrt 3
noncomputable def tan_15_deg : ℝ := 2 - real.sqrt 3
noncomputable def tan_75_deg : ℝ := 2 + real.sqrt 3

theorem arctan_tan_75_minus_3tan_30_minus_tan_15_eq_60 :
  real.arctan (tan_75_deg - 3 * tan_30_deg - tan_15_deg) = real.pi / 3 :=
begin
  sorry
end

end arctan_tan_75_minus_3tan_30_minus_tan_15_eq_60_l63_63495


namespace sum_of_permutations_eq_133320_l63_63001

theorem sum_of_permutations_eq_133320 :
  let digits := [2, 4, 6, 8]
  all_permutations_sum digits = 133320 :=
by
  sorry

end sum_of_permutations_eq_133320_l63_63001


namespace no_closed_polygonal_chain_l63_63834

open Set

-- Define the condition: all pairwise distances are different
def different_pairwise_distances (points : Set Point) : Prop :=
  ∀ p₁ p₂ p₃ : Point, p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ →
    dist p₁ p₂ ≠ dist p₁ p₃ ∧ dist p₁ p₂ ≠ dist p₂ p₃

-- Define the condition: each point is connected to the nearest one
def connected_to_nearest (points : Set Point) (nearest : Point → Point) : Prop :=
  ∀ p : Point, nearest p ∈ points ∧
    (∀ q : Point, q ∈ points → q ≠ p → dist p (nearest p) ≤ dist p q)

-- Main theorem
theorem no_closed_polygonal_chain
  (points : Set Point)
  (h1 : 2 ≤ points.card)
  (h2 : different_pairwise_distances points)
  (nearest : Point → Point)
  (h3 : connected_to_nearest points nearest) :
  ¬ ∃ (closed_chain : List Point),
    (∀ p : Point, p ∈ closed_chain → nearest p ∈ closed_chain) ∧
    List.chain' (λ p₁ p₂, nearest p₁ = p₂) closed_chain :=
sorry

end no_closed_polygonal_chain_l63_63834


namespace cost_price_of_watch_l63_63420

theorem cost_price_of_watch 
  (CP : ℝ)
  (h1 : 0.88 * CP = SP_loss)
  (h2 : 1.04 * CP = SP_gain)
  (h3 : SP_gain - SP_loss = 140) :
  CP = 875 := 
sorry

end cost_price_of_watch_l63_63420


namespace distance_P_to_origin_eq_five_l63_63262

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0) ^ 2 + (y - 0) ^ 2)

theorem distance_P_to_origin_eq_five : distance_from_origin (-4) 3 = 5 :=
by 
  sorry

end distance_P_to_origin_eq_five_l63_63262


namespace find_x_l63_63916

noncomputable def x_value (x : ℝ) : Prop :=
  let seq := [10.0, 12.0, 13.0, x, 17.0, 19.0, 21.0, 24.0]
  (seq[(seq.length / 2) - 1] + seq[seq.length / 2]) / 2 = 16

theorem find_x : ∃ (x : ℝ), x_value x ∧ x = 15 :=
by
  use 15
  unfold x_value
  sorry

end find_x_l63_63916


namespace correct_statements_l63_63198

open Real

/-- Given function definition -/
def f (x p q : ℝ) : ℝ := x * abs x + p * x + q

/-- Statement 1: f(x) is odd if and only if q = 0 -/
lemma f_odd_iff_q_zero (p : ℝ) : (∀ x : ℝ, f x p 0 = -f (-x) p 0) ↔ (0 = 0) := 
by sorry

/-- Statement 2: The graph of f(x) is symmetric about the point (0, q). -/
lemma f_symmetric_about_0_q (p q : ℝ) : ∀ x : ℝ, f x p q = f (-x) p q ↔ (0 = q) :=
by sorry

/-- Statement 3: When p = 0, the solution set of the equation f(x) = 0 is always non-empty. -/
lemma f_has_solution_when_p_zero (q : ℝ) : (∃ x : ℝ, f x 0 q = 0) :=
by sorry

/-- Statement 4: The number of solutions to the equation f(x) = 0 is always no more than two is false. -/
lemma not_f_solution_two (p q : ℝ) : ¬ (∀ x1 x2 : ℝ, f x1 p q = 0 → f x2 p q = 0 → |x1 - x2| ≤ 2) :=
by sorry

-- Solution to the problem
theorem correct_statements : true :=
begin
 split,
 exact f_odd_iff_q_zero p,
 exact f_symmetric_about_0_q p q,
 exact f_has_solution_when_p_zero q,
 exact not_f_solution_two p q,
end

end correct_statements_l63_63198


namespace new_fraction_blue_marbles_l63_63250

variables (y : ℕ)
def fraction_blue_marbles (initial_total : ℕ) : ℚ := 2 / 3
def fraction_red_marbles (initial_total : ℕ) : ℚ := 1 / 3

theorem new_fraction_blue_marbles (initial_total : ℕ) :
  ((2 * initial_total) : ℚ / (7 / 3 * initial_total) = 6 / 7) :=
sorry

end new_fraction_blue_marbles_l63_63250


namespace averageExpenditureFebToJul_l63_63473

/-- Amithab's expenditure in January is 1200 Rs -/
def JanuaryExpenditure := 1200

/-- Average expenditure from January to June is 4200 Rs -/
def AverageExpenditureJanToJun := 4200

/-- Expenditure in July is 1500 Rs -/
def JulyExpenditure := 1500

/-- 
Given the above expenditures, the average expenditure from February to July can be computed. 
-/
theorem averageExpenditureFebToJul : 
  let TotalExpenditureJanToJun := AverageExpenditureJanToJun * 6 in
  let TotalExpenditureFebToJun := TotalExpenditureJanToJun - JanuaryExpenditure in
  let TotalExpenditureFebToJul := TotalExpenditureFebToJun + JulyExpenditure in
  let AverageExpenditureFebToJul := TotalExpenditureFebToJul / 6 in
  AverageExpenditureFebToJul = 4250 := by
  sorry

end averageExpenditureFebToJul_l63_63473


namespace inequality_proof_l63_63764

variable (a b c d : ℝ)

theorem inequality_proof (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (1 / (1 / a + 1 / b)) + (1 / (1 / c + 1 / d)) ≤ (1 / (1 / (a + c) + 1 / (b + d))) :=
by
  sorry

end inequality_proof_l63_63764


namespace each_person_paid_l63_63505

theorem each_person_paid (total_paid : ℕ) (number_of_people : ℕ) (payment_per_person : ℕ) :
  total_paid = 490 → number_of_people = 7 → payment_per_person = total_paid / number_of_people → 
  payment_per_person = 70 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  exact h3
  sorry

end each_person_paid_l63_63505


namespace speed_of_first_train_l63_63429

-- Definitions for the given conditions
def train_length_1 : ℝ := 230 -- in meters
def train_speed_2 : ℝ := 80 -- in km/hr
def crossing_time : ℝ := 9 -- in seconds
def train_length_2 : ℝ := 270.04 -- in meters

-- Convert given lengths and time to consistent units (km and hours)
def total_length_km : ℝ := (train_length_1 + train_length_2) / 1000
def crossing_time_hr : ℝ := crossing_time / 3600

-- Define the relative speed
def relative_speed : ℝ := total_length_km / crossing_time_hr

-- Define the speed of the first train
def train_speed_1 : ℝ := relative_speed - train_speed_2

-- The theorem to prove
theorem speed_of_first_train :
  train_speed_1 = 120.016 :=
by
  -- skip the proof
  sorry

end speed_of_first_train_l63_63429


namespace tetrahedron_sphere_surface_area_l63_63066

noncomputable def surface_area_of_sphere : ℝ := 4 * Real.pi * (Real.sqrt 3 / 2) ^ 2

theorem tetrahedron_sphere_surface_area :
  let edge_length := Real.sqrt 2 in
  let radius := Real.sqrt 3 / 2 in
  let area := 4 * Real.pi * radius ^ 2 in
  area = 3 * Real.pi := 
by
  sorry

end tetrahedron_sphere_surface_area_l63_63066


namespace cot_neg_45_is_neg_1_l63_63122

theorem cot_neg_45_is_neg_1 : Real.cot (Real.pi * -45 / 180) = -1 :=
by
  sorry

end cot_neg_45_is_neg_1_l63_63122


namespace exact_two_class_count_l63_63939

theorem exact_two_class_count :
  let S := 20 -- number of campers in swimming class
    A := 20 -- number of campers in archery class
    R := 20 -- number of campers in rock climbing class
    three_classes := 4  -- campers in all three classes
    exact_one_class := 24 -- campers in exactly one class
  in 
  let n := 20 + 20 + 20 - (|S ∩ A| + |A ∩ R| + |R ∩ S|) + 4 in
  ∃ y, (n = 28 + y) ∧ (n = 52 - y) ∧ (y = 12) :=
begin
  sorry
end

end exact_two_class_count_l63_63939


namespace power_function_fixed_point_l63_63813

theorem power_function_fixed_point (a : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x, f(x) = x^(-1/2))
  (h₂ : (2 : ℝ, (log a (2 * (2 : ℝ) - 3) + (sqrt 2 / 2)) = (2 : ℝ, sqrt 2 / 2))
  (h_pt : (2 : ℝ, sqrt 2 / 2) ∈ set_of f):
  f(9) = 1/3 :=
by
  simp [h₁] ⊢
  sorry

end power_function_fixed_point_l63_63813


namespace remainder_b_mod_2015_l63_63288

def b := (∑ k in Finset.range 1007, (2 * k + 1)^2 - (2 * k + 2)^2)

theorem remainder_b_mod_2015 : b % 2015 = 1 := 
by
suffices h : b = 1007 * 2013
suffices h_mod : 1007 * 2013 % 2015 = 1
from h_mod
sorry

end remainder_b_mod_2015_l63_63288


namespace required_number_l63_63157

noncomputable def find_number (x : ℝ) : Prop :=
( (x^2 * 81) / 356 = 51.193820224719104 ) → 
(x ≈ 15.000205576)

-- sorry is added to skip the proof.
theorem required_number : ∃ x : ℝ, find_number x :=
sorry

end required_number_l63_63157


namespace frog_count_total_frogs_l63_63432

noncomputable def frog_problem (T : ℝ) (N : ℕ) : Prop :=
  50 * (0.3 * T / 50) = 0.3 * T ∧
  44 * (0.27 * T / 44) = 0.27 * T ∧
  (N - 94) * (0.43 * T / (N - 94)) = 0.43 * T ∧
  164 ≤ N ∧ N ≤ 165.67

theorem frog_count_total_frogs (T : ℝ) :
  ∃ N : ℕ, frog_problem T N ∧ N = 165 :=
begin
  use 165,
  split,
  { unfold frog_problem,
    repeat { split; norm_num },
    rw [mul_div_cancel, mul_div_cancel, mul_div_cancel],
    { norm_num, },
    { exact ne_of_gt (by norm_num) },
    { exact ne_of_gt (by norm_num) },
    { exact ne_of_gt (by norm_num) } },
  { refl },
end

end frog_count_total_frogs_l63_63432


namespace cot_neg_45_l63_63126

-- Define the given conditions
def tan_neg_angle (x : ℝ) : Prop := ∀ θ : ℝ, tan (-θ) = -tan(θ)
def tan_45 : Prop := tan (45 * (π / 180)) = 1
def cot_def (x : ℝ) : Prop := ∀ θ : ℝ, cot(θ) = 1 / tan(θ)

-- Prove that cot(-45°) = -1 given the conditions
theorem cot_neg_45 : cot (-45 * (π / 180)) = -1 :=
by 
  have h1 := tan_neg_angle (-45 * (π / 180)),
  have h2 := tan_45,
  have h3 := cot_def (-45 * (π / 180)),
  sorry -- Proof steps skipped

end cot_neg_45_l63_63126


namespace maximum_value_of_a_l63_63238

theorem maximum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) ↔ a ≤ 5 :=
by
  sorry

end maximum_value_of_a_l63_63238


namespace percentage_of_same_grade_is_48_l63_63047

def students_with_same_grade (grades : ℕ × ℕ → ℕ) : ℕ :=
  grades (0, 0) + grades (1, 1) + grades (2, 2) + grades (3, 3) + grades (4, 4)

theorem percentage_of_same_grade_is_48
  (grades : ℕ × ℕ → ℕ)
  (h : grades (0, 0) = 3 ∧ grades (1, 1) = 6 ∧ grades (2, 2) = 8 ∧ grades (3, 3) = 4 ∧ grades (4, 4) = 3)
  (total_students : ℕ) (h_students : total_students = 50) :
  (students_with_same_grade grades / 50 : ℚ) * 100 = 48 :=
by
  sorry

end percentage_of_same_grade_is_48_l63_63047


namespace hundredth_odd_integer_l63_63858

theorem hundredth_odd_integer : ∃ (x : ℕ), 2 * x - 1 = 199 ∧ x = 100 :=
by
  use 100
  split
  . exact calc
      2 * 100 - 1 = 200 - 1 : by ring
      _ = 199 : by norm_num
  . refl

end hundredth_odd_integer_l63_63858


namespace existence_of_another_circle_condition_l63_63044

variable {A B C D E F G H : Point}
variable [convex_quadrilateral : ConvexQuadrilateral A B C D]
variable [circle_tangential_AB_AD : CircleTangentialToSides A B D G H]
variable [circle_intersects_diagonal_AC_EF : CircleIntersectsDiagonalAC A C E F]

theorem existence_of_another_circle_condition : 
  (∃ Q : Circle, Q.pass_through E F ∧ Q.tangential_to_extended_sides D A D C) ↔ 
  (AB + CD = BC + DA) :=
sorry

end existence_of_another_circle_condition_l63_63044


namespace initial_kittens_count_l63_63941

-- Define initial conditions
variable (K : ℕ) -- initial number of kittens
variable (P_initial : ℕ := 7) -- initial number of puppies
variable (P_sold : ℕ := 2) -- number of puppies sold
variable (K_sold : ℕ := 3) -- number of kittens sold
variable (total_remaining : ℕ := 8) -- total number of pets remaining

-- Main theorem statement
theorem initial_kittens_count :
  let P_remaining := P_initial - P_sold in
  let K_remaining := K - K_sold in
  P_remaining + K_remaining = total_remaining → K = 6 :=
by
  sorry

end initial_kittens_count_l63_63941


namespace blue_ball_higher_probability_l63_63902

noncomputable def probability_blue_ball_higher : ℝ :=
  let p (k : ℕ) : ℝ := 1 / (2^k : ℝ)
  let same_bin_prob := ∑' k : ℕ, (p (k + 1))^2
  let higher_prob := (1 - same_bin_prob) / 2
  higher_prob

theorem blue_ball_higher_probability :
  probability_blue_ball_higher = 1 / 3 :=
by
  sorry

end blue_ball_higher_probability_l63_63902


namespace part_I_part_II_l63_63201

noncomputable def f (x : ℝ) (a : ℝ) := a * x^(3/2) - log x - (2/3 : ℝ)
noncomputable def f' (x : ℝ) (a : ℝ) := (3/2) * a * sqrt x - 1 / x
noncomputable def g (x : ℝ) (a : ℝ) := abs (f x a + f' x a)

theorem part_I (a : ℝ) : (∃ x0 : ℝ, f x0 a = 0 ∧ f' x0 a = 0) → a = 2/3 := sorry

theorem part_II (a : ℝ) (x1 x2 : ℝ) (h : x1 ≠ x2) : g x1 a = g x2 a → x1 * x2 < 1 := sorry

end part_I_part_II_l63_63201


namespace hyperbola_solution_l63_63606

noncomputable def hyperbola_eccentricity (a : ℝ) (c : ℝ) : ℝ := c / a

theorem hyperbola_solution (a : ℝ) (h1 : a > 0) (h2 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / (1 - a^2)) = 1)
  (h3 : hyperbola_eccentricity a 1 = √2) : a = √2 / 2 := 
begin
  sorry
end

end hyperbola_solution_l63_63606


namespace questionnaires_drawn_from_D_l63_63453

theorem questionnaires_drawn_from_D (a1 a2 a3 a4 total sample_b sample_total sample_d : ℕ)
  (h1 : a2 - a1 = a3 - a2)
  (h2 : a3 - a2 = a4 - a3)
  (h3 : a1 + a2 + a3 + a4 = total)
  (h4 : total = 1000)
  (h5 : sample_b = 30)
  (h6 : a2 = 200)
  (h7 : sample_total = 150)
  (h8 : sample_d * total = sample_total * a4) :
  sample_d = 60 :=
by sorry

end questionnaires_drawn_from_D_l63_63453


namespace sum_c_d_l63_63087

-- Given integers c and d such that d > 1 and c^d is the greatest value less than 630,
-- prove that the sum of c and d equals 27.
theorem sum_c_d (c d : ℤ) (h1 : d > 1) (h2 : c^d < 630)
  (h3 : ∀ c' d', d' > 1 → c'^d' < 630 → c^d ≥ c'^d') : c + d = 27 := 
sorry

end sum_c_d_l63_63087


namespace deductive_reasoning_is_C_l63_63477

-- Definitions for the reasoning options
def optionA : Prop :=
  ∀ metals, (metals = gold ∨ metals = silver ∨ metals = copper ∨ metals = iron) → conducts_electricity metals

def optionB : Prop :=
  ∃ n ∈ ℕ, ∀ n, a_n = 1 / (n * (n + 1))

def optionC : Prop :=
  ∀ r, (S_circle r = π * r^2) → (r = 1) → S_circle 1 = π

def optionD : Prop :=
  ∀ (x y z a b c r : ℝ), (x - a)^2 + (y - b)^2 = r^2 → (x - a)^2 + (y - b)^2 + (z - c)^2 = r^2

-- Prove that option C represents deductive reasoning.
theorem deductive_reasoning_is_C : optionC :=
by sorry

end deductive_reasoning_is_C_l63_63477


namespace solve_inequality_l63_63348

theorem solve_inequality (x : ℝ) : x ∈ set.Ioo (NegInf) 2 ∪ set.Ioo 2 5 ↔ (x - 5) / ((x - 2)^2) < 0 :=
sorry

end solve_inequality_l63_63348


namespace find_a6_geometric_sequence_l63_63263

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem find_a6_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h1 : geom_seq a q) (h2 : a 4 = 7) (h3 : a 8 = 63) : 
  a 6 = 21 :=
sorry

end find_a6_geometric_sequence_l63_63263


namespace surface_area_of_sphere_l63_63582

-- Given: OA is the radius of sphere O
-- M is the midpoint of OA
-- A plane passing through M and forming a 45 degree angle with OA intersects the surface of sphere O to form circle C 
-- The area of circle C is 7π/4

variables {R r : ℝ}

noncomputable def radius_relation : Prop :=
  r^2 = 7/4

noncomputable def sphere_radius_relation : Prop :=
  R^2 = (sqrt(2)/4 * R)^2 + r^2

theorem surface_area_of_sphere 
  (h₁ : radius_relation)
  (h₂ : sphere_radius_relation)
  : 4 * π * R^2 = 8 * π :=
by
  unfold radius_relation at h₁
  unfold sphere_radius_relation at h₂
  sorry

end surface_area_of_sphere_l63_63582


namespace nm_perpendicular_to_bc_l63_63726

-- Definitions and Conditions
variables {A B C N M : Type}
variables (A B C N M : Point)
variables {α β : Angle}
variables {Γ : Circle}

-- Define the right triangle
def is_right_triangle (A B C : Point) : Prop :=
  ∃ (α : Angle), α = 90 ∧ ∠ACB = 30

-- Define Γ passing through A and tangent to midpoint P of BC
def circle_tangent_to_midpoint (Γ : Circle) (A P : Point) : Prop :=
  Γ.PassesThrough A ∧ Γ.TangentAtMidpointOf BC P

-- Define intersections
def intersections (Γ : Circle) (A C N M : Point) (circABC : Circle) : Prop :=
  Γ.Intersects AC N ∧ Γ.Intersects circABC M

-- Goal (proof problem)
theorem nm_perpendicular_to_bc (A B C N M : Point) (Γ circABC : Circle)
  (h1 : is_right_triangle A B C)
  (h2 : circle_tangent_to_midpoint Γ A (midpoint B C))
  (h3 : intersections Γ A C N M circABC) :
  Perpendicular N M B C :=
sorry

end nm_perpendicular_to_bc_l63_63726


namespace coprime_n_minus_2_n_squared_minus_n_minus_1_l63_63893

theorem coprime_n_minus_2_n_squared_minus_n_minus_1 (n : ℕ) : n - 2 ∣ n^2 - n - 1 → False :=
by
-- proof omitted as per instructions
sorry

end coprime_n_minus_2_n_squared_minus_n_minus_1_l63_63893


namespace correct_sampling_methods_l63_63494

-- Define the surveys with their corresponding conditions
structure Survey1 where
  high_income : Nat
  middle_income : Nat
  low_income : Nat
  total_households : Nat

structure Survey2 where
  total_students : Nat
  sample_students : Nat
  differences_small : Bool
  sizes_small : Bool

-- Define the conditions
def survey1_conditions (s : Survey1) : Prop :=
  s.high_income = 125 ∧ s.middle_income = 280 ∧ s.low_income = 95 ∧ s.total_households = 100

def survey2_conditions (s : Survey2) : Prop :=
  s.total_students = 15 ∧ s.sample_students = 3 ∧ s.differences_small = true ∧ s.sizes_small = true

-- Define the answer predicate
def correct_answer (method1 method2 : String) : Prop :=
  method1 = "stratified sampling" ∧ method2 = "simple random sampling"

-- The theorem statement
theorem correct_sampling_methods (s1 : Survey1) (s2 : Survey2) :
  survey1_conditions s1 → survey2_conditions s2 → correct_answer "stratified sampling" "simple random sampling" :=
by
  -- Proof skipped for problem statement purpose
  sorry

end correct_sampling_methods_l63_63494


namespace find_n_satisfying_equation_l63_63099

def P (n : ℕ) : ℕ := -- Define the greatest prime divisor of n, assume this is already defined

noncomputable def floor (x : ℝ) : ℤ := -- Define the floor function, assume it is already defined

theorem find_n_satisfying_equation :
  ∀ (n : ℕ), n ≥ 2 ∧ (P n + floor (Real.sqrt n) = P (n + 1) + floor (Real.sqrt (n + 1))) → n = 3 :=
by 
  intros n hn heq
  sorry

end find_n_satisfying_equation_l63_63099


namespace sets_difference_M_star_N_l63_63280

def M (y : ℝ) : Prop := y ≤ 2

def N (y : ℝ) : Prop := 0 ≤ y ∧ y ≤ 3

def M_star_N (y : ℝ) : Prop := y < 0

theorem sets_difference_M_star_N : {y : ℝ | M y ∧ ¬ N y} = {y : ℝ | M_star_N y} :=
by {
  sorry
}

end sets_difference_M_star_N_l63_63280


namespace product_of_area_and_perimeter_of_EFGH_l63_63306

-- Define the coordinates of each point in the 6 by 6 grid
def E : (ℝ × ℝ) := (1, 4)
def F : (ℝ × ℝ) := (4, 5)
def G : (ℝ × ℝ) := (5, 2)
def H : (ℝ × ℝ) := (2, 1)

-- Compute the distance formula between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the sides of the rectangle
def EF : ℝ := distance E F
def FG : ℝ := distance F G
def GH : ℝ := distance G H
def HE : ℝ := distance H E

-- Define the area and perimeter of the rectangle
def area : ℝ := EF * FG
def perimeter : ℝ := EF + FG + GH + HE

-- Define the product of the area and perimeter
def product : ℝ := area * perimeter

-- The proof statement to be proved
theorem product_of_area_and_perimeter_of_EFGH : product = 40 * real.sqrt 10 :=
by
  sorry

end product_of_area_and_perimeter_of_EFGH_l63_63306


namespace det_A_mul_B_l63_63729

-- Define variables A and B such that det(A) = -3 and B = A⁻¹
variables {A B : Matrix ℝ ℝ}

-- Given conditions
def cond_det_A : det A = -3 := sorry
def cond_B_inv_A : B = A⁻¹ := sorry

-- Prove that det(AB) = 1
theorem det_A_mul_B : det (A * B) = 1 :=
by
  sorry

end det_A_mul_B_l63_63729


namespace parallelogram_base_length_l63_63177

theorem parallelogram_base_length (b h : ℝ) (area : ℝ) (angle : ℝ) (h_area : area = 200) 
(h_altitude : h = 2 * b) (h_angle : angle = 60) : b = 10 :=
by
  -- Placeholder for proof
  sorry

end parallelogram_base_length_l63_63177


namespace a_6_eq_10_a_n_formula_l63_63614

noncomputable def a_n (n : ℕ) [fact (2 < n)] : ℕ :=
  if n % 2 = 1 then
    2^(n - 2)
  else
    2^(n - 2) - nat.choose (n - 2) ((n - 2) / 2)

theorem a_6_eq_10 : a_n 6 = 10 := sorry

theorem a_n_formula (n : ℕ) [fact (2 < n)] :
  a_n n = if n % 2 = 1 then 2^(n - 2) else 2^(n - 2) - nat.choose (n - 2) ((n - 2) / 2) := sorry

end a_6_eq_10_a_n_formula_l63_63614


namespace difference_in_pay_l63_63751

-- Definitions for the problem conditions
def oula_pay_per_delivery : ℕ := 100
def oula_deliveries : ℕ := 96
def tona_multiplier : ℚ := 3 / 4

-- Statements derived from the conditions
def oula_earnings : ℕ := oula_deliveries * oula_pay_per_delivery

def tona_deliveries : ℚ := tona_multiplier * oula_deliveries
def tona_earnings : ℕ := ((tona_deliveries : ℕ) * oula_pay_per_delivery)

-- Theorem stating the difference in pay
theorem difference_in_pay : oula_earnings - tona_earnings = 2400 :=
by sorry

end difference_in_pay_l63_63751


namespace correct_statements_count_l63_63194

def z : ℂ := 1 + complex.I

def statement1 : Prop := complex.abs z = real.sqrt 2
def statement2 : Prop := complex.conj z = 1 - complex.I
def statement3 : Prop := complex.im z = complex.I

def num_correct_statements : ℕ := if statement1 ∧ statement2 ∧ ¬statement3 then 2 else 0

-- Main theorem statement
theorem correct_statements_count : num_correct_statements = 2 :=
by
  -- Proof would go here, but for now we assume it's true
  sorry

end correct_statements_count_l63_63194


namespace maximize_profit_successful_investment_l63_63041

noncomputable def investment_allocation : ℝ → ℝ :=
  λ x, -0.2 * x + sqrt x + 30

theorem maximize_profit : 
  ∃ x ∈ Icc 0 100, investment_allocation x = 31.25 :=
begin
  use 6.25,
  split,
  { linarith, },
  { sorry, }
end

def total_resource_fee : ℝ := 2 * (1 + 1.1^1 + 1.1^2)

noncomputable def median_expected_profit_after_fees : ℝ :=
  (31.25 + 20) / 2 - total_resource_fee

theorem successful_investment :
  median_expected_profit_after_fees / 100 > 0.18 :=
begin
  sorry
end

end maximize_profit_successful_investment_l63_63041


namespace unique_int_function_l63_63150

noncomputable def unique_int_function_eq : Prop :=
  ∃! (f : ℤ → ℤ), ∀ (a b : ℤ), f(a + b) - f(ab) = f(a) * f(b) - 1

-- insert sorry here to indicate the proof is omitted
theorem unique_int_function : unique_int_function_eq :=
sorry

end unique_int_function_l63_63150


namespace num_distinct_exponentiation_values_l63_63467

theorem num_distinct_exponentiation_values : 
  let a := 2
  let b1 := 2
  let b2 := 2
  let b3 := 2
  let standard_value := (a ^ (b1 ^ (b2 ^ b3)))
  let val_1 := (a ^ (a ^ a)) ^ a
  let val_2 := a ^ ((a ^ a) ^ a)
  let val_3 := ((a ^ a) ^ a) ^ a
  let val_4 := (a ^ (a ^ a)) ^ a
  let val_5 := (a ^ a) ^ (a ^ a)
  in 
  (∃ values : Finset ℕ, values.card = 2 ∧
  standard_value ∈ values ∧ 
  (Finset.erase values standard_value).card = 1 ∧ 
  val_1 ∈ values ∧ val_2 ∈ values ∧ val_3 ∈ values ∧ 
  val_4 ∈ values ∧ val_5 ∈ values) :=
by
  let a := 2
  let b1 := 2
  let b2 := 2
  let b3 := 2
  let standard_value := (a ^ (b1 ^ (b2 ^ b3)))
  let val_1 := (a ^ (a ^ a)) ^ a
  let val_2 := a ^ ((a ^ a) ^ a)
  let val_3 := ((a ^ a) ^ a) ^ a
  let val_4 := (a ^ (a ^ a)) ^ a
  let val_5 := (a ^ a) ^ (a ^ a)
  have h : ∃ values : Finset ℕ, values.card = 2 ∧
    standard_value ∈ values ∧ 
    (Finset.erase values standard_value).card = 1 ∧ 
    val_1 ∈ values ∧ val_2 ∈ values ∧ val_3 ∈ values ∧ 
    val_4 ∈ values ∧ val_5 ∈ values := sorry
  exact h

end num_distinct_exponentiation_values_l63_63467


namespace intervals_of_monotonicity_l63_63510

noncomputable def y (x : ℝ) : ℝ := 2 ^ (x^2 - 2*x + 4)

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x > 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ < y x₂)) ∧
  (∀ x : ℝ, x < 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ > y x₂)) :=
by
  sorry

end intervals_of_monotonicity_l63_63510


namespace number_of_divisors_600_l63_63852

def prime_factors_600 : List (ℕ × ℕ) := [(2, 3), (3, 1), (5, 2)]

theorem number_of_divisors_600 : 
  (prime_factors_600.map (fun x => x.snd + 1)).prod = 24 := 
by 
  unfold prime_factors_600
  norm_num
  sorry

end number_of_divisors_600_l63_63852


namespace cot_neg_45_l63_63125

-- Define the given conditions
def tan_neg_angle (x : ℝ) : Prop := ∀ θ : ℝ, tan (-θ) = -tan(θ)
def tan_45 : Prop := tan (45 * (π / 180)) = 1
def cot_def (x : ℝ) : Prop := ∀ θ : ℝ, cot(θ) = 1 / tan(θ)

-- Prove that cot(-45°) = -1 given the conditions
theorem cot_neg_45 : cot (-45 * (π / 180)) = -1 :=
by 
  have h1 := tan_neg_angle (-45 * (π / 180)),
  have h2 := tan_45,
  have h3 := cot_def (-45 * (π / 180)),
  sorry -- Proof steps skipped

end cot_neg_45_l63_63125


namespace rose_needs_more_money_l63_63324

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end rose_needs_more_money_l63_63324


namespace polynomial_product_l63_63823

theorem polynomial_product :
  ∀ (p1 p2 : Polynomial ℂ),
    (degree p1 = 2) →
    (degree p2 = 3) →
    (p1 * p2).roots = [-1, 2, -3] →
    ∃ (a b : ℂ), 
      p1 = Polynomial.Coeff 2 1 + Polynomial.Coeff 1 a + Polynomial.Coeff 0 b ∧
      p2 = Polynomial.mul (Polynomial.Coeff 1 (-2)) p1 ∧
      p1 * p2 = Polynomial.mul (Polynomial.Coeff 1 (-2)) (Polynomial.mul p1 p1) :=
by
  intros p1 p2 h1 h2 h3
  use [1, -2]
  split
  . sorry
  . split
    . sorry
    . sorry

end polynomial_product_l63_63823


namespace solve_missing_digit_A_l63_63526

def is_digit (d : ℕ) : Prop := d ∈ finset.range 10

theorem solve_missing_digit_A (A : ℕ) (h1 : is_digit A) (h2 : ∃ k : ℤ, 203 + 10 * A = 15 * k) : A = 5 :=
by
  sorry

end solve_missing_digit_A_l63_63526


namespace emily_furniture_assembly_time_l63_63029

-- Definitions based on conditions
def chairs := 4
def tables := 2
def time_per_piece := 8

-- Proof statement
theorem emily_furniture_assembly_time : (chairs + tables) * time_per_piece = 48 :=
by
  sorry

end emily_furniture_assembly_time_l63_63029


namespace circle_points_coordinates_l63_63472

theorem circle_points_coordinates :
    let r1 := 2
    let r2 := 4
    let r3 := 6
    let r4 := 8
    let r5 := 10
    let C1 := 2 * Real.pi * r1
    let C2 := 2 * Real.pi * r2
    let C3 := 2 * Real.pi * r3
    let C4 := 2 * Real.pi * r4
    let C5 := 2 * Real.pi * r5
    let A1 := Real.pi * r1^2
    let A2 := Real.pi * r2^2
    let A3 := Real.pi * r3^2
    let A4 := Real.pi * r4^2
    let A5 := Real.pi * r5^2
    in [(C1, A1), (C2, A2), (C3, A3), (C4, A4), (C5, A5)] = 
       [(4 * Real.pi, 4 * Real.pi), (8 * Real.pi, 16 * Real.pi), 
        (12 * Real.pi, 36 * Real.pi), (16 * Real.pi, 64 * Real.pi), 
        (20 * Real.pi, 100 * Real.pi)] :=
by {
  sorry
}

end circle_points_coordinates_l63_63472


namespace proportion_overcrowded_cars_min_proportion_passengers_overcrowded_passengers_proportion_not_less_than_cars_l63_63026

-- Definitions for histogram data
def carriages_percentages : List (ℕ × ℕ) :=
  [
    (4, 19), (6, 29), (12, 39), (18, 49), 
    (20, 59), (20, 69), (14, 79), (6, 89)
  ]

def overcrowded_threshold : ℕ := 60

-- Part (a): Proof that the proportion of overcrowded train cars is 40%
theorem proportion_overcrowded_cars :
  let overcrowded_percentages : List ℕ := [20, 14, 6]
  let total_percent : ℕ := overcrowded_percentages.sum
  total_percent = 40 := by
  sorry

-- Part (b): Proof that the minimum possible proportion of passengers in overcrowded cars is 49%
theorem min_proportion_passengers_overcrowded :
  let num_passengers_non_overcrowded := 0.76 * N + 1.74 * N + 4.68 * N + 8.82 * N + 11.8 * N
  let num_passengers_overcrowded := 12 * N + 9.8 * N + 4.8 * N
  let total_passengers := num_passengers_non_overcrowded + num_passengers_overcrowded
  let proportion := num_passengers_overcrowded / total_passengers
  proportion ≈ 0.49 := by
  sorry

-- Part (c): Proof that the proportion of passengers in overcrowded cars cannot be less than that of overcrowded cars
theorem passengers_proportion_not_less_than_cars :
  let total_cars := N
  let total_passengers := n
  let overcrowded_cars := M
  let passengers_overcrowded := m
  passengers_overcrowded / total_passengers ≥ overcrowded_cars / total_cars := by
  sorry

end proportion_overcrowded_cars_min_proportion_passengers_overcrowded_passengers_proportion_not_less_than_cars_l63_63026


namespace triangle_shape_l63_63246

theorem triangle_shape (A B C a b c : ℝ)
  (hA : A = π / 3)
  (ha : a = sqrt 3)
  (hb : b = 1)
  (h_angles : A + B + C = π)
  (h_law_of_sines : a / sin A = b / sin B)
  (ha_c : a^2 = b^2 + c^2) :
  C = π / 2 := 
by 
  sorry

end triangle_shape_l63_63246


namespace intersection_A_B_l63_63185

noncomputable def A : set ℝ := {x | log 2 x < 4}
noncomputable def B : set ℝ := {x | abs x ≤ 2}

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_B_l63_63185


namespace point_in_first_quadrant_l63_63264

theorem point_in_first_quadrant (k : ℝ) (h : 0 < k) : (sqrt 3, k).1 > 0 ∧ (sqrt 3, k).2 > 0 :=
by sorry

end point_in_first_quadrant_l63_63264


namespace no_function_f_l63_63722

noncomputable def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_function_f (a b c : ℝ) (h : ∀ x, g a b c (g a b c x) = x) :
  ¬ ∃ f : ℝ → ℝ, ∀ x, f (f x) = g a b c x := 
sorry

end no_function_f_l63_63722


namespace john_hourly_wage_l63_63272

theorem john_hourly_wage (days_off: ℕ) (hours_per_day: ℕ) (weekly_wage: ℕ) 
  (days_off_eq: days_off = 3) (hours_per_day_eq: hours_per_day = 4) (weekly_wage_eq: weekly_wage = 160):
  (weekly_wage / ((7 - days_off) * hours_per_day) = 10) :=
by
  /-
  Given:
  days_off = 3
  hours_per_day = 4
  weekly_wage = 160

  To prove:
  weekly_wage / ((7 - days_off) * hours_per_day) = 10
  -/
  sorry

end john_hourly_wage_l63_63272


namespace tarantulas_per_egg_sac_l63_63236

-- Condition: Each tarantula has 8 legs
def legs_per_tarantula : ℕ := 8

-- Condition: There are 32000 baby tarantula legs
def total_legs : ℕ := 32000

-- Condition: Number of egg sacs is one less than 5
def number_of_egg_sacs : ℕ := 5 - 1

-- Calculated: Number of tarantulas in total
def total_tarantulas : ℕ := total_legs / legs_per_tarantula

-- Proof Statement: Number of tarantulas per egg sac
theorem tarantulas_per_egg_sac : total_tarantulas / number_of_egg_sacs = 1000 := by
  sorry

end tarantulas_per_egg_sac_l63_63236


namespace interval_of_expression_l63_63100

theorem interval_of_expression (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧ 
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
by sorry

end interval_of_expression_l63_63100


namespace hundredth_odd_positive_integer_l63_63866

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l63_63866


namespace trip_time_difference_l63_63923

-- Definitions of the given conditions
def speed_AB := 160 -- speed from A to B in km/h
def speed_BA := 120 -- speed from B to A in km/h
def distance_AB := 480 -- distance between A and B in km

-- Calculation of the time for each trip
def time_AB := distance_AB / speed_AB
def time_BA := distance_AB / speed_BA

-- The statement we need to prove
theorem trip_time_difference :
  (time_BA - time_AB) = 1 :=
by
  sorry

end trip_time_difference_l63_63923


namespace max_numbers_such_that_product_times_240_is_perfect_square_l63_63562

def is_natural_number (n : ℕ) : Prop := n > 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem max_numbers_such_that_product_times_240_is_perfect_square :
  (∃ S : set ℕ, (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2015) ∧ 
                 (∀ n ∈ S, is_natural_number n) ∧
                 (∏ n in S, n) * 240 = k ∧ is_perfect_square k ∧ 
                 (∀ x : ℕ, (∀ n ∈ S, n ≠ x ∧ is_perfect_square ((∏ n in S, n) * 240)) → ∃ S' : set ℕ, S ⊆ S' ∧ S'.card < S.card))
  ∧ ∑ n in S, n = 134 :=
by sorry


end max_numbers_such_that_product_times_240_is_perfect_square_l63_63562


namespace smallest_visible_sum_l63_63037

def opposite_faces_sum_seven (die : ℕ → ℕ → Prop) : Prop :=
  ∀ (a b : ℕ), die a b → a + b = 7

def adjacent_dice_sum_less_than_twelve (cube : ℕ → ℕ → ℕ → ℕ → ℕ → Prop) : Prop :=
  ∀ (x₁ y₁ z₁ x₂ y₂ z₂ a₁ a₂ : ℕ), cube x₁ y₁ z₁ a₁ → cube x₂ y₂ z₂ a₂ →
  (∃ (dx dy dz : ℕ), (dx = |x₂ - x₁| ∧ dy = |y₂ - y₁| ∧ dz = |z₂ - z₁| ∧ dx + dy + dz = 1)) →
  a₁ + a₂ < 12

def visible_face_sum (cube : ℕ → ℕ → ℕ → ℕ → Prop) : ℕ :=
  ∑ x in finset.range 4, ∑ y in finset.range 4, ∑ z in finset.range 4, 
  if x = 0 ∨ x = 3 ∨ y = 0 ∨ y = 3 ∨ z = 0 ∨ z = 3 then
    finset.range 6 (λ a, if cube x y z a then a else 0)
  else 0

theorem smallest_visible_sum :
  (∃ (die : ℕ → ℕ → Prop) (cube : ℕ → ℕ → ℕ → ℕ → Prop),
  opposite_faces_sum_seven die ∧ adjacent_dice_sum_less_than_twelve cube) →
  visible_face_sum cube = 168 :=
by {
  assume h,
  sorry
}

end smallest_visible_sum_l63_63037


namespace nth_odd_positive_integer_is_199_l63_63854

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l63_63854


namespace equilateral_triangle_rotation_exists_l63_63491

theorem equilateral_triangle_rotation_exists :
  ∀ (A B C : Point), ∃ (X Y : Point),
  dist A X = dist B Y ∧ dist A X = dist A B ∧
  dist B X = dist C Y ∧ dist B X = dist B C ∧
  dist C X = dist A Y ∧ dist C X = dist C A :=
by
  sorry

end equilateral_triangle_rotation_exists_l63_63491


namespace largest_divisor_of_product_l63_63077

theorem largest_divisor_of_product :
  ∀ (die_faces : Finset ℕ),
    die_faces = {1, 2, 3, 4, 5, 6, 7, 8} →
    ∀ (unseen_faces : Finset ℕ),
      unseen_faces.card = 2 →
      unseen_faces ⊆ die_faces →
      ∃ (P : ℕ),
        P = (die_faces \ unseen_faces).prod id →
        96 ∣ P :=
by
  intros die_faces h_die_faces unseen_faces h_unseen_card h_subset
  obtain ⟨P, hP⟩ := exists_eq_prod (die_faces \ unseen_faces) id
  exact ⟨P, hP.symm ▸ sorry⟩

end largest_divisor_of_product_l63_63077


namespace coprime_congruence_l63_63290

theorem coprime_congruence {a b k m : ℕ} (h : k * a ≡ k * b [MOD m]) (co_prime : Nat.coprime k m) : a ≡ b [MOD m] :=
by
  sorry

end coprime_congruence_l63_63290


namespace price_per_gallon_in_NC_l63_63249

variable (P : ℝ)
variable (price_nc := P) -- price per gallon in North Carolina
variable (price_va := P + 1) -- price per gallon in Virginia
variable (gallons_nc := 10) -- gallons bought in North Carolina
variable (gallons_va := 10) -- gallons bought in Virginia
variable (total_cost := 50) -- total amount spent on gas

theorem price_per_gallon_in_NC :
  (gallons_nc * price_nc) + (gallons_va * price_va) = total_cost → price_nc = 2 :=
by
  sorry

end price_per_gallon_in_NC_l63_63249


namespace frog_ends_within_2_meters_l63_63446

noncomputable def frog_jump_probability : ℚ := 1 / 3

theorem frog_ends_within_2_meters :
  ∀ (jumps: Fin 5 → ℝ × ℝ), 
    (∀ i, ∥jumps i∥ = 1) →
    (Prob (fun x => ∥(0,0) + jumps 0 + jumps 1 + jumps 2 + jumps 3 + jumps 4∥ ≤ 2) = frog_jump_probability) := 
sorry

end frog_ends_within_2_meters_l63_63446


namespace f_is_ab_type_function_g_m_range_l63_63172

def is_type_ab_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x : ℝ, f(a + x) * f(a - x) = b

def f : ℝ → ℝ := λ x, 4^x

theorem f_is_ab_type_function : ∃ a b : ℝ, is_type_ab_function f a b :=
sorry

def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 1

theorem g_m_range (m : ℝ) (h : 0 < m) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 1 ≤ g m x ∧ g m x ≤ 3) →
  2 - (2 * Real.sqrt 6) / 3 ≤ m ∧ m ≤ 2 :=
sorry

end f_is_ab_type_function_g_m_range_l63_63172


namespace count_integers_in_interval_l63_63641

theorem count_integers_in_interval :
  {n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi}.toFinset.card = 57 :=
by
  sorry

end count_integers_in_interval_l63_63641


namespace regular_polygons_cover_plane_l63_63414

theorem regular_polygons_cover_plane (n : ℕ) (h_n_ge_3 : 3 ≤ n)
    (h_angle_eq : ∀ n, (180 * (1 - (2 / n)) : ℝ) = (internal_angle : ℝ))
    (h_summation_eq : ∃ k : ℕ, k * internal_angle = 360) :
    n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_polygons_cover_plane_l63_63414


namespace Irene_hours_worked_l63_63268

open Nat

theorem Irene_hours_worked (x totalHours : ℕ) : 
  (500 + 20 * x = 700) → 
  (totalHours = 40 + x) → 
  totalHours = 50 :=
by
  sorry

end Irene_hours_worked_l63_63268


namespace total_frisbees_l63_63064

-- Let x be the number of $3 frisbees and y be the number of $4 frisbees.
variables (x y : ℕ)

-- Condition 1: Total sales amount is 200 dollars.
def condition1 : Prop := 3 * x + 4 * y = 200

-- Condition 2: At least 8 $4 frisbees were sold.
def condition2 : Prop := y >= 8

-- Prove that the total number of frisbees sold is 64.
theorem total_frisbees (h1 : condition1 x y) (h2 : condition2 y) : x + y = 64 :=
by
  sorry

end total_frisbees_l63_63064


namespace isosceles_triangle_vertex_angle_range_l63_63003

theorem isosceles_triangle_vertex_angle_range
  (a b : ℝ)
  (α : ℝ)
  (triangle : isosceles_triangle a b α)
  (three_lines_exist : ∃ lines : list (line ℝ), 
    (∀ l ∈ lines, bisects_area_triangle l triangle ∧ bisects_perimeter_triangle l triangle) 
    ∧ lines.length = 3) : 
  2 * real.arcsin (real.sqrt 2 - 1) < α ∧ α < real.pi :=
sorry

end isosceles_triangle_vertex_angle_range_l63_63003


namespace single_point_graph_value_of_d_l63_63352

theorem single_point_graph_value_of_d (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + d = 0 → x = -2 ∧ y = 3) ↔ d = 21 := 
by 
  sorry

end single_point_graph_value_of_d_l63_63352


namespace find_n_cosine_l63_63147

theorem find_n_cosine (n : ℤ) (h1 : 100 ≤ n ∧ n ≤ 300) (h2 : Real.cos (n : ℝ) = Real.cos 140) : n = 220 :=
by
  sorry

end find_n_cosine_l63_63147


namespace tan_identity_l63_63566

theorem tan_identity (α : ℝ) (h1 : sin α + cos α = - (sqrt 10) / 5) (h2 : 0 < α ∧ α < π) : 
  tan α = -1/3 :=
sorry

end tan_identity_l63_63566


namespace william_marbles_l63_63014

theorem william_marbles :
  let initial_marbles := 10
  let shared_marbles := 3
  (initial_marbles - shared_marbles) = 7 := 
by
  sorry

end william_marbles_l63_63014


namespace annular_region_area_l63_63404

noncomputable def area_annulus (r1 r2 : ℝ) : ℝ :=
  (Real.pi * r2 ^ 2) - (Real.pi * r1 ^ 2)

theorem annular_region_area :
  area_annulus 4 7 = 33 * Real.pi :=
by 
  sorry

end annular_region_area_l63_63404


namespace six_digit_palindromic_divisible_by_forty_five_l63_63528

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def divisible_by (m n : ℕ) : Prop := m % n = 0

theorem six_digit_palindromic_divisible_by_forty_five
  (P : ℕ)
  (h_palindromic : is_palindromic P)
  (h_six_digit : 100000 ≤ P ∧ P < 1000000)
  (h_divisible_45 : divisible_by P 45) :
  P ∈ {504405, 513315, 522225, 531135, 540045, 549945, 558855, 567765, 576675, 585585, 594495} := sorry

end six_digit_palindromic_divisible_by_forty_five_l63_63528


namespace num_perpendicular_line_plane_pairs_in_cube_l63_63666

-- Definitions based on the problem conditions

def is_perpendicular_line_plane_pair (l : line) (p : plane) : Prop :=
  -- Assume an implementation that defines when a line is perpendicular to a plane
  sorry

-- Define a cube structure with its vertices, edges, and faces
structure Cube :=
  (vertices : Finset Point)
  (edges : Finset (Point × Point))
  (faces : Finset (Finset Point))

-- Make assumptions about cube properties
variable (cube : Cube)

-- Define the property of counting perpendicular line-plane pairs
def count_perpendicular_line_plane_pairs (c : Cube) : Nat :=
  -- Assume an implementation that counts the number of such pairs in the cube
  sorry

-- The theorem to prove
theorem num_perpendicular_line_plane_pairs_in_cube (c : Cube) :
  count_perpendicular_line_plane_pairs c = 36 :=
  sorry

end num_perpendicular_line_plane_pairs_in_cube_l63_63666


namespace trig_identity_problem_l63_63659

theorem trig_identity_problem
  (x : ℝ) (a b c : ℕ)
  (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.sin x - Real.cos x = Real.pi / 4)
  (h3 : Real.tan x + 1 / Real.tan x = (a : ℝ) / (b - Real.pi^c)) :
  a + b + c = 50 :=
sorry

end trig_identity_problem_l63_63659


namespace number_of_weavers_l63_63349

theorem number_of_weavers (W : ℕ) 
  (h1 : ∀ t : ℕ, t = 4 → 4 = W * (1 * t)) 
  (h2 : ∀ t : ℕ, t = 16 → 64 = 16 * (1 / (W:ℝ) * t)) : 
  W = 4 := 
by {
  sorry
}

end number_of_weavers_l63_63349


namespace average_marks_second_class_l63_63531

variable (average_marks_first_class : ℝ) (students_first_class : ℕ)
variable (students_second_class : ℕ) (combined_average_marks : ℝ)

theorem average_marks_second_class (H1 : average_marks_first_class = 60)
  (H2 : students_first_class = 55) (H3 : students_second_class = 48)
  (H4 : combined_average_marks = 59.067961165048544) :
  48 * 57.92 = 103 * 59.067961165048544 - 3300 := by
  sorry

end average_marks_second_class_l63_63531


namespace combined_balance_l63_63563

theorem combined_balance (b : ℤ) (g1 g2 : ℤ) (h1 : b = 3456) (h2 : g1 = b / 4) (h3 : g2 = b / 4) : g1 + g2 = 1728 :=
by {
  sorry
}

end combined_balance_l63_63563


namespace grace_hours_pulling_weeds_l63_63623

variable (Charge_mowing : ℕ) (Charge_weeding : ℕ) (Charge_mulching : ℕ)
variable (H_m : ℕ) (H_u : ℕ) (E_s : ℕ)

theorem grace_hours_pulling_weeds 
  (Charge_mowing_eq : Charge_mowing = 6)
  (Charge_weeding_eq : Charge_weeding = 11)
  (Charge_mulching_eq : Charge_mulching = 9)
  (H_m_eq : H_m = 63)
  (H_u_eq : H_u = 10)
  (E_s_eq : E_s = 567) :
  ∃ W : ℕ, 6 * 63 + 11 * W + 9 * 10 = 567 ∧ W = 9 := by
  sorry

end grace_hours_pulling_weeds_l63_63623


namespace limit_of_an_l63_63024

theorem limit_of_an (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n, a_n n = (4 * n - 3) / (2 * n + 1)) → 
  a = 2 → 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros ha hA ε hε
  sorry

end limit_of_an_l63_63024


namespace max_value_x_y_squared_l63_63737

theorem max_value_x_y_squared (x y : ℝ) (h : 3 * (x^3 + y^3) = x + y^2) : x + y^2 ≤ 1/3 :=
sorry

end max_value_x_y_squared_l63_63737


namespace false_proposition_discrete_random_variable_l63_63231

noncomputable theory

open ProbabilityTheory

variables {Ω : Type*} (P : MeasureTheory.ProbabilityMeasure Ω)

theorem false_proposition_discrete_random_variable (X : DiscreteRandomVariable ℝ) :
  (∀ x_i, 0 ≤ P {ω | X ω = x_i}) ∧
  (∑' i, P {ω | X ω = X i} = 1) ∧
  (∀ A : Set ℝ, P {ω | X ω ∈ A} = ∑ i in A, P {ω | X ω = i}) →
  ¬ (∃ R : Set ℝ, P {ω | X ω ∈ R} > ∑ i in R, P {ω | X ω = i}) :=
by {
  intros h,
  sorry
}

end false_proposition_discrete_random_variable_l63_63231


namespace hyperbola_equation_l63_63573

theorem hyperbola_equation
  (a : ℝ)
  (c : ℝ)
  (center : Prop)
  (foci : Prop)
  (axes_equal : Prop)
  (asymptote_distance : Prop)
  (dist_eq_sqrt2 : c = Real.sqrt 2 * a)
  : x^2 - y^2 = 2 := 
begin
  sorry,
end

end hyperbola_equation_l63_63573


namespace average_age_of_coaches_l63_63355

variables 
  (total_members : ℕ) (avg_age_total : ℕ) 
  (num_girls : ℕ) (num_boys : ℕ) (num_coaches : ℕ) 
  (avg_age_girls : ℕ) (avg_age_boys : ℕ)

theorem average_age_of_coaches 
  (h1 : total_members = 50) 
  (h2 : avg_age_total = 18)
  (h3 : num_girls = 25) 
  (h4 : num_boys = 20) 
  (h5 : num_coaches = 5)
  (h6 : avg_age_girls = 16)
  (h7 : avg_age_boys = 17) : 
  (900 - (num_girls * avg_age_girls + num_boys * avg_age_boys)) / num_coaches = 32 :=
by
  sorry

end average_age_of_coaches_l63_63355


namespace speed_of_truck_l63_63906

theorem speed_of_truck
  (v : ℝ)                         -- Let \( v \) be the speed of the truck.
  (car_speed : ℝ := 55)           -- Car speed is 55 mph.
  (start_delay : ℝ := 1)          -- Truck starts 1 hour later.
  (catchup_time : ℝ := 6.5)       -- Truck takes 6.5 hours to pass the car.
  (additional_distance_car : ℝ := car_speed * catchup_time)  -- Additional distance covered by the car in 6.5 hours.
  (total_distance_truck : ℝ := car_speed * start_delay + additional_distance_car)  -- Total distance truck must cover to pass the car.
  (truck_distance_eq : v * catchup_time = total_distance_truck)  -- Distance equation for the truck.
  : v = 63.46 :=                -- Prove the truck's speed is 63.46 mph.
by
  -- Original problem solution confirms truck's speed as 63.46 mph. 
  sorry

end speed_of_truck_l63_63906


namespace panda_babies_l63_63925

theorem panda_babies (total_pandas : ℕ) (pregnancy_rate : ℚ) (pandas_coupled : total_pandas % 2 = 0) (couple_has_one_baby : ℕ → ℕ) :
  total_pandas = 16 → pregnancy_rate = 0.25 → couple_has_one_baby ((total_pandas / 2) * pregnancy_rate).natAbs = 2 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end panda_babies_l63_63925


namespace num_subsets_div_by_three_l63_63956

open Finset

def set : Finset ℕ := {101, 106, 111, 146, 154, 159}

def sums_to_three (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ (s.sum % 3 = 0)

theorem num_subsets_div_by_three :
  (set.powerset.filter sums_to_three).card = 5 := 
sorry

end num_subsets_div_by_three_l63_63956


namespace max_value_expression_l63_63536

theorem max_value_expression (a b c x y z : ℝ) (h1 : 2 ≤ a ∧ a ≤ 3) (h2 : 2 ≤ b ∧ b ≤ 3) (h3 : 2 ≤ c ∧ c ≤ 3)
    (perm : ∃ p : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ), p (a, b, c) = (x, y, z)) :
    (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z) ≤ 15 / 4) :=
begin
  sorry
end

end max_value_expression_l63_63536


namespace nth_odd_positive_integer_is_199_l63_63853

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l63_63853


namespace simplify_expression_l63_63234

theorem simplify_expression (x y : ℝ) (h : y = x / (1 - 2 * x)) :
    (2 * x - 3 * x * y - 2 * y) / (y + x * y - x) = -7 / 3 := 
by {
  sorry
}

end simplify_expression_l63_63234


namespace exit_condition_l63_63673

-- Define the loop structure in a way that is consistent with how the problem is described
noncomputable def program_loop (k : ℕ) : ℕ :=
  if k < 7 then 35 else sorry -- simulate the steps of the program

-- The proof goal is to show that the condition which stops the loop when s = 35 is k ≥ 7
theorem exit_condition (k : ℕ) (s : ℕ) : 
  (program_loop k = 35) → (k ≥ 7) :=
by {
  sorry
}

end exit_condition_l63_63673


namespace max_value_expression_l63_63545

theorem max_value_expression
  (a b c x y z : ℝ)
  (h₁ : 2 ≤ a ∧ a ≤ 3)
  (h₂ : 2 ≤ b ∧ b ≤ 3)
  (h₃ : 2 ≤ c ∧ c ≤ 3)
  (h4 : {x, y, z} = {a, b, c}) :
  (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_value_expression_l63_63545


namespace coeff_sum_eq_neg_242_and_abs_coeff_sum_eq_2882_l63_63596

theorem coeff_sum_eq_neg_242_and_abs_coeff_sum_eq_2882 {
  (a a1 a2 a3 a4 a5 : ℤ)
  (eqn : (3 - 2 * x)^5 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) :
  (a1 + a2 + a3 + a4 + a5 = -242) ∧ 
  (|a1| + |a2| + |a3| + |a4| + |a5| = 2882) :=
sorry

end coeff_sum_eq_neg_242_and_abs_coeff_sum_eq_2882_l63_63596


namespace trigonometric_relationship_l63_63167

theorem trigonometric_relationship :
  let a := sin (cos (2016 * real.pi / 180))
  let b := sin (sin (2016 * real.pi / 180))
  let c := cos (sin (2016 * real.pi / 180))
  let d := cos (cos (2016 * real.pi / 180))
  in c > d ∧ d > b ∧ b > a :=
by {
  sorry
}

end trigonometric_relationship_l63_63167


namespace cyl_tube_vol_diff_l63_63469

theorem cyl_tube_vol_diff
  (w l : ℝ)
  (hw : w = 7)
  (hl : l = 10) :
  (Real.pi * (|((Real.pi * ((Real.sqrt (w^2 + l^2)) / (2 * Real.pi))^2 * l) -
               (Real.pi * ((Real.sqrt (w^2 + l^2)) / (2 * Real.pi))^2 * w)|)))
  = 111.75 := by
  sorry

end cyl_tube_vol_diff_l63_63469


namespace difference_seven_three_times_l63_63780

theorem difference_seven_three_times (n : ℝ) (h1 : n = 3) 
  (h2 : 7 * n = 3 * n + (21.0 - 9.0)) :
  7 * n - 3 * n = 12.0 := by
  sorry

end difference_seven_three_times_l63_63780


namespace graph_shift_sin_l63_63838

theorem graph_shift_sin (x : ℝ) :
  ∃ (c : ℝ), (∀ x, sin (3 * x - (π / 3)) = sin(3 * (x - c))) ∧ c = π / 9 :=
by
  use π / 9
  split
  { intro x,
    sorry }
  { refl }

end graph_shift_sin_l63_63838


namespace fuel_consumed_at_40_minimum_fuel_consumption_l63_63065

-- Definitions from the problem statement
def fuel_consumption_per_hour (x : ℝ) : ℝ :=
  (1/128000) * x^3 - (3/80) * x + 8

def distance := 100 -- kilometers

def travel_time (x : ℝ) : ℝ :=
  distance / x

def total_fuel_consumption (x : ℝ) : ℝ :=
  fuel_consumption_per_hour(x) * travel_time(x)

-- Part (1)
theorem fuel_consumed_at_40 : total_fuel_consumption(40) = 17.5 :=
by sorry

-- Part (2)
def h (x : ℝ) : ℝ :=
  (1/1280) * x^2 + (800 / x) - (15 / 4)

theorem minimum_fuel_consumption :
  (∀ x, 0 < x ∧ x ≤ 120 → h(x) ≥ 11.25) ∧ (h 80 = 11.25) :=
by sorry

end fuel_consumed_at_40_minimum_fuel_consumption_l63_63065


namespace cot_neg_45_l63_63129

-- Define the given conditions
def tan_neg_angle (x : ℝ) : Prop := ∀ θ : ℝ, tan (-θ) = -tan(θ)
def tan_45 : Prop := tan (45 * (π / 180)) = 1
def cot_def (x : ℝ) : Prop := ∀ θ : ℝ, cot(θ) = 1 / tan(θ)

-- Prove that cot(-45°) = -1 given the conditions
theorem cot_neg_45 : cot (-45 * (π / 180)) = -1 :=
by 
  have h1 := tan_neg_angle (-45 * (π / 180)),
  have h2 := tan_45,
  have h3 := cot_def (-45 * (π / 180)),
  sorry -- Proof steps skipped

end cot_neg_45_l63_63129


namespace orthocenters_collinear_l63_63277

-- Define the points A, B, C, K, M, L, N
variables {A B C K M L N : Type*}

-- Define the conditions
variables {triangle_ABC : Triangle A B C}
variables {K_on_AB : OnSegment K A B}
variables {M_on_AB : OnSegment M A B}
variables {L_on_AC : OnSegment L A C}
variables {N_on_AC : OnSegment N A C}
variables {K_between_M_B : Between K M B}
variables {L_between_N_C : Between L N C}
variables {ratio_condition : BK / KM = CL / LN}

-- The statement of the problem
theorem orthocenters_collinear :
  Orthocenter (Triangle A B C) ∈ L ∧
  Orthocenter (Triangle A K L) ∈ L ∧
  Orthocenter (Triangle A M N) ∈ L :=
  sorry

end orthocenters_collinear_l63_63277


namespace first_digit_units_modified_fib_is_0_l63_63500

theorem first_digit_units_modified_fib_is_0 (G : ℕ → ℕ) 
  (h1 : G 1 = 2) 
  (h2 : G 2 = 1) 
  (h3 : ∀ n ≥ 3, G n = G (n - 1) + G (n - 2)) :
  ∃ n, G n % 10 = 0 :=
begin
  sorry
end

end first_digit_units_modified_fib_is_0_l63_63500


namespace number_of_filter_families_l63_63057

def filter_family_count (n : ℕ) : ℕ :=
  ∑ k in Finset.range n, (Nat.choose n k) * (2^k - 1)

theorem number_of_filter_families (n : ℕ) :
  ∃ f : Finset (Finset (Fin n)) → Prop,
    (∀ A B ∈ f, ∃ C ∈ f, C ⊆ (A ∩ B)) ∧
    filter_family_count n = ∑ k in Finset.range n, (Nat.choose n k) * (2^k - 1) := 
sorry

end number_of_filter_families_l63_63057


namespace solve_for_c_and_d_l63_63212

-- Define the vectors 
structure Vector3 :=
  (x : ℝ) (y : ℝ) (z : ℝ)

-- Define the cross product of two vectors
def cross_product (v1 v2 : Vector3) : Vector3 :=
  ⟨v1.y * v2.z - v1.z * v2.y, 
   v1.z * v2.x - v1.x * v2.z, 
   v1.x * v2.y - v1.y * v2.x⟩

-- Define the zero vector
def zero_vector : Vector3 := ⟨0, 0, 0⟩

-- The problem statement is to prove c = 5/2 and d = -18 given the cross product condition
theorem solve_for_c_and_d (c d : ℝ) 
  (h : cross_product ⟨3, c, -9⟩ ⟨6, 5, d⟩ = zero_vector) : 
  c = 5/2 ∧ d = -18 :=
sorry

end solve_for_c_and_d_l63_63212


namespace intersection_eq_l63_63163

namespace lean_proof

def setA : Set ℝ := { x | abs x < 3 }
def setB : Set ℝ := { x | 2 ^ x > 1 }
def setIntersection : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem intersection_eq :
  setA ∩ setB = setIntersection :=
sorry

end lean_proof

end intersection_eq_l63_63163


namespace cistern_water_depth_l63_63046

theorem cistern_water_depth
  (length width : ℝ) 
  (wet_surface_area : ℝ)
  (h : ℝ) 
  (hl : length = 7)
  (hw : width = 4)
  (ha : wet_surface_area = 55.5)
  (h_eq : 28 + 22 * h = wet_surface_area) 
  : h = 1.25 := 
  by 
  sorry

end cistern_water_depth_l63_63046


namespace proof_that_question_is_correct_l63_63245

noncomputable def problem_condition (x : ℝ) : Prop :=
  (sqrt x - 8) / 13 = 6

noncomputable def problem_question (x : ℝ) : ℝ :=
  (x^2 - 45) / 23

theorem proof_that_question_is_correct (x : ℝ) (h : problem_condition x) :
  problem_question x = 2380011 :=
sorry

end proof_that_question_is_correct_l63_63245


namespace arithmetic_sequence_common_difference_sum_l63_63478

theorem arithmetic_sequence_common_difference_sum :
  let d := (gcd 465 1550) in
  d ∣ 465 ∧ d ∣ 1550 ∧ d > 1 →
  ∃ (s : Int), s = (5 + 31 + 155) :=
by {
  sorry
}

end arithmetic_sequence_common_difference_sum_l63_63478


namespace num_zeros_f_on_interval_l63_63171
noncomputable theory

def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 4 then x^2 - 2^x else sorry

theorem num_zeros_f_on_interval :
  (∀ x : ℝ, f x + f (x + 4) = 23) →
  (∀ x : ℝ, 0 < x ∧ x ≤ 4 → f x = x^2 - 2^x) →
  (set.countable {x : ℝ | f x = 0 } ∩ Icc (-4 : ℝ) 2023).card = 506 :=
sorry

end num_zeros_f_on_interval_l63_63171


namespace episodes_in_show_l63_63298

theorem episodes_in_show (episodes_per_monday : ℕ) (episodes_per_wednesday : ℕ) (weeks : ℕ) :
  episodes_per_monday = 1 → episodes_per_wednesday = 2 → weeks = 67 →
  (episodes_per_monday + episodes_per_wednesday) * weeks = 201 :=
by
  intros h_mon h_wed h_weeks
  rw [h_mon, h_wed, h_weeks]
  sorry

end episodes_in_show_l63_63298


namespace number_of_other_values_l63_63465

def orig_value : ℕ := 2 ^ (2 ^ (2 ^ 2))

def other_values : Finset ℕ :=
  {2 ^ (2 ^ (2 ^ 2)), 2 ^ ((2 ^ 2) ^ 2), ((2 ^ 2) ^ 2) ^ 2, (2 ^ (2 ^ 2)) ^ 2, (2 ^ 2) ^ (2 ^ 2)}

theorem number_of_other_values :
  other_values.erase orig_value = {256} :=
by
  sorry

end number_of_other_values_l63_63465


namespace cans_of_soda_l63_63796

variable (T R E : ℝ)

theorem cans_of_soda (hT: T > 0) (hR: R > 0) (hE: E > 0) : 5 * E * T / R = (5 * E) / R * T :=
by
  sorry

end cans_of_soda_l63_63796


namespace inequality1_inequality2_l63_63605

-- Problem 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem inequality1 (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := sorry

-- Problem 2
def g (x : ℝ) : ℝ := f x + f (-x)

theorem inequality2 (k : ℝ) (h : ∀ x : ℝ, |k - 1| < g x) : -3 < k ∧ k < 5 := sorry

end inequality1_inequality2_l63_63605


namespace correct_relation_l63_63613

theorem correct_relation (M : Set Int) (hM : M = {-1, 0, 1}) : (0 ∈ M) :=
by {
  rw hM,
  simp,
}

end correct_relation_l63_63613


namespace total_cost_is_18_l63_63773

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_l63_63773


namespace sector_area_150_degrees_l63_63806

def sector_area (radius : ℝ) (central_angle : ℝ) : ℝ :=
  0.5 * radius^2 * central_angle

theorem sector_area_150_degrees (r : ℝ) (angle_rad : ℝ) (h1 : r = Real.sqrt 3) (h2 : angle_rad = (5 * Real.pi) / 6) : 
  sector_area r angle_rad = (5 * Real.pi) / 4 :=
by
  simp [sector_area, h1, h2]
  sorry

end sector_area_150_degrees_l63_63806


namespace tigers_count_l63_63484

theorem tigers_count (T C : ℝ) 
  (h1 : 12 + T + C = 39) 
  (h2 : C = 0.5 * (12 + T)) : 
  T = 14 := by
  sorry

end tigers_count_l63_63484


namespace monic_quartic_polynomial_l63_63140

noncomputable def quartic_poly : Polynomial ℚ :=
  Polynomial.ofIntPolynomial (Polynomial.Coeff.mapRat 
    ([1, -14, 57, -132, 36] : Polynomial ℤ))

theorem monic_quartic_polynomial :
  ∃ P : Polynomial ℚ, Polynomial.monic P ∧ 
  (P = quartic_poly) ∧ 
  (P.eval (3 + Real.sqrt 5) = 0) ∧ 
  (P.eval (3 - Real.sqrt 5) = 0) ∧ 
  (P.eval (4 - Real.sqrt 7) = 0) ∧ 
  (P.eval (4 + Real.sqrt 7) = 0) :=
by
  use quartic_poly
  split
  · -- Prove that the polynomial is monic
    sorry
  split
  · -- Prove that the polynomial is x^4 - 14x^3 + 57x^2 - 132x + 36
    sorry
  split
  · -- Prove that 3 + sqrt(5) is a root
    sorry
  split
  · -- Prove that 3 - sqrt(5) is a root
    sorry
  split
  · -- Prove that 4 - sqrt(7) is a root
    sorry
  · -- Prove that 4 + sqrt(7) is a root
    sorry

end monic_quartic_polynomial_l63_63140


namespace three_out_of_five_correct_prob_l63_63393

theorem three_out_of_five_correct_prob :
  let n := 5 in
  let favorable := (numberOfCombinations n 3) * derangements 2 in
  let total := factorial n in
  favorable / total = 1 / 12 := by
  sorry

end three_out_of_five_correct_prob_l63_63393


namespace clique_of_4_exists_l63_63255

-- Define the basic structure for the problem
variables (People : Type) [fintype People] [decidable_eq People]
variable [has_mem People People]
variable [group_3 : ∀ (p q r : People), p ∈ group_3 → q ∈ group_3 → r ∈ group_3 → p.know_each_other q ∨ p.know_each_other r ∨ q.know_each_other r]

-- Define the concept of knowing each other
def know_each_other (p q : People) : Prop := sorry

-- Define the main theorem
theorem clique_of_4_exists (h : fintype.card People = 9) :
  ∃ (a b c d : People), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  know_each_other a b ∧ know_each_other a c ∧ know_each_other a d ∧ 
  know_each_other b c ∧ know_each_other b d ∧ know_each_other c d :=
sorry

end clique_of_4_exists_l63_63255


namespace compare_areas_pentagon_octagon_l63_63078

noncomputable def apothem (s : ℝ) (θ : ℝ) : ℝ := s * Real.cot θ
noncomputable def circumradius (s : ℝ) (θ : ℝ) : ℝ := s * Real.csc θ

theorem compare_areas_pentagon_octagon :
  let s1 := 3;
      θ1 := 36;
      s2 := 2;
      θ2 := 22.5;
      A1 := apothem s1 θ1;
      R1 := circumradius s1 θ1;
      A2 := apothem s2 θ2;
      R2 := circumradius s2 θ2;
  in π * (R1 ^ 2 - A1 ^ 2) > π * (R2 ^ 2 - A2 ^ 2) := 
by
  sorry

end compare_areas_pentagon_octagon_l63_63078


namespace polynomial_root_relation_l63_63509

theorem polynomial_root_relation :
  (∀ r, r^2 - 2*r - 1 = 0 →
    r^5 - b*r - c = 0) →
  b = 29 → 
  c = 12 → 
  b * c = 348 := by
  intros h hb hc
  rw [hb, hc]
  exact eq.refl 348

end polynomial_root_relation_l63_63509


namespace integer_count_in_interval_l63_63626

theorem integer_count_in_interval : 
  let pi := Real.pi in
  let lower_bound := -6 * pi in
  let upper_bound := 12 * pi in
  ∃ (count : ℕ), count = 56 ∧ ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound ↔ (-18 ≤ n ∧ n ≤ 37) :=
by
  let pi := Real.pi
  let lower_bound := -6 * pi
  let upper_bound := 12 * pi
  use 56
  split
  · exact rfl
  · intro n
    split
    · intro h
      split
      · linarith
      · linarith
    · intro h
      split
      · linarith
      · linarith
  sorry

end integer_count_in_interval_l63_63626


namespace max_value_of_expr_l63_63546

noncomputable def max_value_expr (a b c x y z : ℝ) : ℝ :=
  (a / x) + (a + b) / (x + y) + (a + b + c) / (x + y + z)

theorem max_value_of_expr {a b c x y z : ℝ} 
  (ha : 2 ≤ a ∧ a ≤ 3) (hb : 2 ≤ b ∧ b ≤ 3) (hc : 2 ≤ c ∧ c ≤ 3)
  (hperm : {x, y, z} = {a, b, c}) : 
  max_value_expr a b c x y z ≤ 15 / 4 :=
by
  sorry

end max_value_of_expr_l63_63546


namespace smallest_positive_shift_l63_63351

noncomputable def g : ℝ → ℝ := sorry

theorem smallest_positive_shift
  (H1 : ∀ x, g (x - 20) = g x) : 
  ∃ a > 0, (∀ x, g ((x - a) / 10) = g (x / 10)) ∧ a = 200 :=
sorry

end smallest_positive_shift_l63_63351


namespace marathon_checkpoint_distance_l63_63036

theorem marathon_checkpoint_distance :
  ∀ (total_distance num_checkpoints segment_before start_finish_distance : ℕ),
    total_distance = 26 →
    num_checkpoints = 4 →
    segment_before = 1 →
    start_finish_distance = total_distance - 2 * segment_before →
    (start_finish_distance / (num_checkpoints - 1)) = 8 :=
by
  intros total_distance num_checkpoints segment_before start_finish_distance
  assume h1 : total_distance = 26
  assume h2 : num_checkpoints = 4
  assume h3 : segment_before = 1
  assume h4 : start_finish_distance = total_distance - 2 * segment_before
  sorry

end marathon_checkpoint_distance_l63_63036


namespace arrangement_ways_l63_63803

def Animal := { chickens := 4, dogs := 3, fishes := 5 }

def ways_to_place (a : Animal) : Nat :=
  (3.factorial * 4.factorial * 3.factorial * 5.factorial)

theorem arrangement_ways : ways_to_place { chickens := 4, dogs := 3, fishes := 5 } = 103680 :=
  by
    unfold ways_to_place
    norm_num
    sorry

end arrangement_ways_l63_63803


namespace smallest_integer_inequality_l63_63551

theorem smallest_integer_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ 
           (∀ m : ℤ, m < n → ¬∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) :=
by
  sorry

end smallest_integer_inequality_l63_63551


namespace quad_inequality_solution_set_is_reals_l63_63359

theorem quad_inequality_solution_set_is_reals (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) := 
sorry

end quad_inequality_solution_set_is_reals_l63_63359


namespace alternating_binomial_sum_l63_63960

theorem alternating_binomial_sum :
  ∑ k in finset.range 51, (-1 : ℤ)^k * (k + 1) * nat.choose 50 k = 0 :=
sorry

end alternating_binomial_sum_l63_63960


namespace eval_f_7_plus_f_0_l63_63599

def f (x : ℝ) : ℝ :=
if x < 1 then 3 ^ x + 2 else log (x + 2) / log 3

theorem eval_f_7_plus_f_0 : f 7 + f 0 = 5 :=
by
  have h1 : f 7 = 2 := by sorry -- Evaluate f(7)
  have h2 : f 0 = 3 := by sorry -- Evaluate f(0)
  rw [h1, h2]
  exact add_comm 2 3

end eval_f_7_plus_f_0_l63_63599


namespace cot_neg_45_eq_neg_1_l63_63138

-- Hypotheses
variable (θ : ℝ)
variable (h1 : 𝔸.cot θ = 1 / 𝔸.tan θ)
variable (h2 : 𝔸.tan (-45) = -𝔸.tan 45)
variable (h3 : 𝔸.tan 45 = 1)

-- Theorem
theorem cot_neg_45_eq_neg_1 :
  𝔸.cot (-45) = -1 := by
  sorry

end cot_neg_45_eq_neg_1_l63_63138


namespace expand_expression_l63_63525

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := 
by
  sorry

end expand_expression_l63_63525


namespace unique_solution_for_2_3_6_eq_7_l63_63970

theorem unique_solution_for_2_3_6_eq_7 (x : ℝ) : 2^x + 3^x + 6^x = 7^x → x = 2 :=
by
  intro h
  -- Add the relevant proof tactic steps here
  sorry

end unique_solution_for_2_3_6_eq_7_l63_63970


namespace probability_of_detecting_unqualified_products_l63_63433

theorem probability_of_detecting_unqualified_products :
  let cans := {1, 2, 3, 4, 'a', 'b'}
  let qualified := {1, 2, 3, 4}
  let unqualified := {'a', 'b'}
  let outcomes := ({1, 2}, {1, 3}, {1, 4}, {1, 'a'}, {1, 'b'},
                   {2, 3}, {2, 4}, {2, 'a'}, {2, 'b'},
                   {3, 4}, {3, 'a'}, {3, 'b'},
                   {4, 'a'}, {4, 'b'}, {'a', 'b'} : Set (Set _))
  let unqualified_outcomes := ({1, 'a'}, {1, 'b'}, {2, 'a'}, {2, 'b'},
                               {3, 'a'}, {3, 'b'}, {4, 'a'}, {4, 'b'}, {'a', 'b'} : Set (Set _))
  (unqualified_outcomes.toFinset.card / outcomes.toFinset.card : ℚ) = 3 / 5 := by
sorry

end probability_of_detecting_unqualified_products_l63_63433


namespace num_panda_babies_l63_63928

-- Definitions based on conditions
def pandas := 16
def couples := pandas / 2
def pregnancy_rate := 0.25
def pregnant_couples := pregnancy_rate * couples
def babies_per_couple := 1

-- Theorem stating the problem
theorem num_panda_babies : (pregnant_couples * babies_per_couple) = 2 := by
  sorry

end num_panda_babies_l63_63928


namespace simplify_factorial_division_l63_63785

theorem simplify_factorial_division : (13.factorial / (11.factorial + 3 * 10.factorial)) = 1716 := by
  sorry

end simplify_factorial_division_l63_63785


namespace calculate_expression_l63_63943

theorem calculate_expression :
  6 * tan (real.pi / 4) - 2 * cos (real.pi / 3) = 5 := by
  -- Conditions from the problem
  have h1 : tan (real.pi / 4) = 1 := by
    -- prove tan (pi/4) = 1
    sorry
  have h2 : cos (real.pi / 3) = 1 / 2 := by
    -- prove cos (pi/3) = 1/2
    sorry
  -- Main calculation
  calc
    6 * tan (real.pi / 4) - 2 * cos (real.pi / 3)
        = 6 * 1 - 2 * (1 / 2) : by rw [h1, h2]
    ... = 6 - 1 : by norm_num
    ... = 5 : by norm_num

end calculate_expression_l63_63943


namespace basketball_free_throws_l63_63677

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 4 * a) 
  (h2 : x = 2 * a) 
  (h3 : 2 * a + 3 * b + x = 72) : 
  x = 18 := 
sorry

end basketball_free_throws_l63_63677


namespace pebbles_in_smaller_piles_l63_63059

theorem pebbles_in_smaller_piles 
  (a_1 a_2 a_3 a_4 b_1 b_2 b_3 b_4 b_5 N : ℕ) 
  (h1 : a_1 + a_2 + a_3 + a_4 = N)
  (h2 : b_1 + b_2 + b_3 + b_4 + b_5 = N)
  (h3 : 1 / a_1 + 1 / a_2 + 1 / a_3 + 1 / a_4 = 4)
  (h4 : 1 / b_1 + 1 / b_2 + 1 / b_3 + 1 / b_4 + 1 / b_5 = 5)
  : ∃ j1 j2, j1 ≠ j2 ∧ j1 ∈ {1, 2, 3, 4} ∧ j2 ∈ {1, 2, 3, 4, 5} ∧ a_j1 > b_j2 := by 
  sorry

end pebbles_in_smaller_piles_l63_63059


namespace continuous_f_iff_continuous_V_l63_63743

open Set

variables {a b x x₀ : ℝ} (f : ℝ → ℝ)

noncomputable def variation (f : ℝ → ℝ) (a x : ℝ) : ℝ :=
sorry  -- Assume some definition or import from an existing library

def is_bounded_variation (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ (V : ℝ → ℝ), (∀ x ∈ Icc a b, V x = variation f a x) ∧ V a = 0

def continuous_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
∀ ε > 0, ∃ δ > 0, ∀ x', abs (x' - x) < δ → abs (f x' - f x) < ε 

theorem continuous_f_iff_continuous_V
  (h_bounded_variation : is_bounded_variation f a b)
  (h_x₀_in_interval : x₀ ∈ Icc a b) :
  continuous_at f x₀ ↔ continuous_at (λ x, variation f a x) x₀ :=
sorry

end continuous_f_iff_continuous_V_l63_63743


namespace Mark_walk_distance_in_15_min_l63_63271

/-- It takes 18 minutes for Mark to walk one mile. -/
def Mark_walking_rate : ℝ := 1 / 18

/-- Given the rate, calculate how far Mark will walk in 15 minutes -/
theorem Mark_walk_distance_in_15_min : 
  let time_in_minutes := 15 in
  (Mark_walking_rate * time_in_minutes) ≈ 0.8 :=
by
  sorry

end Mark_walk_distance_in_15_min_l63_63271


namespace find_number_divided_l63_63748

theorem find_number_divided (n : ℕ) (h : n = 21 * 9 + 1) : n = 190 :=
by
  sorry

end find_number_divided_l63_63748


namespace total_flooring_cost_l63_63089

theorem total_flooring_cost :
  let 
    area1 := 5.5 * 3.75,
    area2 := 6 * 4.2,
    area3 := 4.8 * 3.25,
    cost1 := area1 * 1200,
    cost2 := area2 * 1350,
    cost3 := area3 * 1450,
    total_cost := cost1 + cost2 + cost3
  in total_cost = 81390 :=
by
  sorry

end total_flooring_cost_l63_63089


namespace theta_range_l63_63571

theorem theta_range (θ : ℝ) (hx : ∀ x : ℝ, x ∈ set.Icc (-1) 0 → x^2 * real.cos θ + (x+1)^2 * real.sin θ + x^2 + x > 0) (hθ : 0 ≤ θ ∧ θ < real.pi) :
  (real.pi / 12) < θ ∧ θ < (5 * real.pi / 12) :=
sorry

end theta_range_l63_63571


namespace mustard_bottles_total_l63_63086

theorem mustard_bottles_total (b1 b2 b3 : ℝ) (h1 : b1 = 0.25) (h2 : b2 = 0.25) (h3 : b3 = 0.38) :
  b1 + b2 + b3 = 0.88 :=
by
  sorry

end mustard_bottles_total_l63_63086


namespace convex_n_gon_min_k_convex_n_gon_max_k_l63_63170

noncomputable def min_k (n : ℕ) : ℕ :=
  3

noncomputable def max_k (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 1 else n

theorem convex_n_gon_min_k (n : ℕ) (h₁ : n > 2) (points : Fin n → ℝ × ℝ) (valid : ∀ i, (i < n) →
  ∃ (k : Fin n), some_line_condition holds) : min_k n = 3 := sorry

theorem convex_n_gon_max_k (n : ℕ) (h₁ : n > 2) (points : Fin n → ℝ × ℝ) (valid : ∀ i, (i < n) →
  ∃ (k : Fin n), some_line_condition holds) : max_k n = 
if n % 2 = 0 then n - 1 else n := sorry

end convex_n_gon_min_k_convex_n_gon_max_k_l63_63170


namespace first_problem_second_problem_l63_63745

-- Define the l-increasing property of a function f.
def l_increasing {α : Type*} [OrderedRing α] (f : α → α) (l : α) (D : Set α) :=
  ∀ x ∈ D, x + l ∈ D ∧ f(x + l) ≥ f(x)

-- Conditions and proof for the first problem
theorem first_problem :
  let f : ℝ → ℝ := λ x, x^2
  let D : Set ℝ := {x | -1 ≤ x}
  ∃ m > 0, l_increasing f m D → m ≥ 2 := 
  sorry

-- Conditions and proof for the second problem
theorem second_problem :
  ∃ a : ℝ, 
  let f : ℝ → ℝ := λ x, if x ≥ 0 then |x - a^2| - a^2 else -(|-x - a^2| - a^2)
  (∀ x : ℝ, f (-x) = -f x) → -- f is odd
  l_increasing f 8 Set.univ → -- f is 8-increasing on ℝ
  -2 ≤ a ∧ a ≤ 2 := 
  sorry

end first_problem_second_problem_l63_63745


namespace fliers_remaining_next_day_l63_63417

def remaining_fliers (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ) : ℕ :=
  let morning_sent := morning_fraction * total
  let remaining_after_morning := total - morning_sent
  let afternoon_sent := afternoon_fraction * remaining_after_morning
  remaining_after_morning - afternoon_sent

theorem fliers_remaining_next_day :
  remaining_fliers 3000 (1/5 : ℚ) (1/4 : ℚ) = 1800 := by
  sorry

end fliers_remaining_next_day_l63_63417


namespace sum_of_coeffs_expansion_l63_63523

theorem sum_of_coeffs_expansion (d : ℝ) : 
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
in coeff_sum = -21 := 
by
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
    exact rfl

end sum_of_coeffs_expansion_l63_63523


namespace tan_inequality_l63_63192

open Real

theorem tan_inequality (α β : ℝ) 
  (h1 : α ∈ Ioo (π / 2) π)
  (h2 : β ∈ Ioo (π / 2) π)
  (h3 : tan α < tan (π / 2 - β)) :
  α + β < 3 * π / 2 := 
sorry

end tan_inequality_l63_63192


namespace percent_sugar_in_resulting_solution_l63_63749

theorem percent_sugar_in_resulting_solution (W : ℝ) (hW : W > 0) :
  let original_sugar_percent := 22 / 100
  let second_solution_sugar_percent := 74 / 100
  let remaining_original_weight := (3 / 4) * W
  let removed_weight := (1 / 4) * W
  let sugar_from_remaining_original := (original_sugar_percent * remaining_original_weight)
  let sugar_from_added_second_solution := (second_solution_sugar_percent * removed_weight)
  let total_sugar := sugar_from_remaining_original + sugar_from_added_second_solution
  let resulting_sugar_percent := total_sugar / W
  resulting_sugar_percent = 35 / 100 :=
by
  sorry

end percent_sugar_in_resulting_solution_l63_63749


namespace cyclic_shift_diagonal_sum_nonnegative_l63_63260

theorem cyclic_shift_diagonal_sum_nonnegative (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
  (h_nonneg : 0 ≤ ∑ i j, A i j) :
  ∃ k : ℕ, (∑ i, A (Fin.ofNat (n - 1 - i)) (Fin.ofNat ((i + k) % n))) ≥ 0 := sorry

end cyclic_shift_diagonal_sum_nonnegative_l63_63260


namespace max_value_expression_l63_63542

theorem max_value_expression
  (a b c x y z : ℝ)
  (h₁ : 2 ≤ a ∧ a ≤ 3)
  (h₂ : 2 ≤ b ∧ b ≤ 3)
  (h₃ : 2 ≤ c ∧ c ≤ 3)
  (h4 : {x, y, z} = {a, b, c}) :
  (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_value_expression_l63_63542


namespace average_score_last_4_matches_l63_63356

theorem average_score_last_4_matches (avg_10_matches : ℝ) (avg_6_matches : ℝ)
  (total_matches : ℕ) (first_matches : ℕ) (last_matches : ℕ) :
  avg_10_matches = 38.9 → avg_6_matches = 41 → total_matches = 10 → first_matches = 6
  → last_matches = 4 → 
  (avg_10_matches * total_matches - avg_6_matches * first_matches) / last_matches = 35.75 := 
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end average_score_last_4_matches_l63_63356


namespace smallest_lcm_l63_63647

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l63_63647


namespace find_distinct_pairs_l63_63739

def A := { x : ℕ | ∃ (a a1 a2 : ℕ), a ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                                  a1 ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                                  a2 ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                                  x = a + a1 * 10 + a2 * 100 }

def conditions (x y : ℕ) := x ∈ A ∧ y ∈ A ∧ x + y = 636

theorem find_distinct_pairs :
  { (x, y) : ℕ × ℕ | conditions x y }.card = 90 := 
sorry

end find_distinct_pairs_l63_63739


namespace problem1_problem2_l63_63197

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2^x) / (4^x + 1)

-- Problem 1: Prove that f(log_(sqrt(2))(3)) = 9/82
theorem problem1 : f (Real.log 3 / Real.log (Real.sqrt 2)) = 9/82 :=
by
  sorry

-- Problem 2: Prove that f(x) is decreasing on the interval (0, +∞)
theorem problem2 (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) : f x₁ > f x₂ :=
by
  sorry

end problem1_problem2_l63_63197


namespace limit_of_an_l63_63025

theorem limit_of_an (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n, a_n n = (4 * n - 3) / (2 * n + 1)) → 
  a = 2 → 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros ha hA ε hε
  sorry

end limit_of_an_l63_63025


namespace line_equation_proof_l63_63362

-- Define the point (1, 0)
def point := (1, 0)

-- Define the original line equation
def original_line_eq (x y : ℝ) := x - y = 0

-- Define the condition of the line passing through (1, 0)
def passes_through_point (x y : ℝ) := (x, y) = point

-- Define the condition of the line being perpendicular to another line with a given slope
def perpendicular_to (m1 m2 : ℝ) := m1 * m2 = -1

-- Provide the slope of the original line
def slope_original := 1

-- Define the slope of the perpendicular line
def slope_perpendicular := -1

-- Define the equation of the line we are trying to prove
def desired_line_eq (x y : ℝ) := x + y - 1 = 0

-- Main theorem to prove
theorem line_equation_proof : 
  ∀ x y : ℝ, passes_through_point x y → original_line_eq x y → perpendicular_to slope_original slope_perpendicular → desired_line_eq x y :=
by
  sorry

end line_equation_proof_l63_63362


namespace alex_additional_coins_l63_63073

theorem alex_additional_coins
  (friends : ℕ)
  (initial_coins : ℕ)
  (distinct_distribution : ∀ i j, i ≠ j → coins i ≠ coins j)
  (coins : ℕ → ℕ) :
  friends = 15 ∧ initial_coins = 60 ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ friends → coins(i) ≥ 1) → 
  (∑ i in finset.range friends, coins(i)) = 120 → 
  ∑ coins(15) = 120 - 60 :=
sorry

end alex_additional_coins_l63_63073


namespace area_trapezoid_ABCE_is_correct_l63_63936

-- Define the rectangle ABCD with given lengths
def is_rectangle (A B C D : Type) (AB BC CD DA : ℝ) := 
  AB = 6 ∧ BC = 8 ∧ AB = CD ∧ BC = DA ∧ (AB + BC) = (CD + DA)

-- Define the folding condition along CE
def folding_condition (A B C D E F : Type) (AC : ℝ): 
  AC = 10 → CE = 8 → True

-- The proof goal: Area of trapezoid ABCE
theorem area_trapezoid_ABCE_is_correct (A B C D E F : Type) (AB BC CD DA AC AE : ℝ) :
  is_rectangle A B C D AB BC CD DA → 
  CD = 6 →
  AE = 5 → 
  (AE + BC) * AB / 2 = 39 := 
by 
  intros h1 h2 h3
  exact sorry

end area_trapezoid_ABCE_is_correct_l63_63936


namespace total_time_to_row_around_lake_l63_63718

-- Definitions of conditions
def lake_side_length : ℕ := 25 -- in miles
def swim_time_per_mile : ℕ := 20 -- in minutes
def swimming_speed : ℚ := 1 / (swim_time_per_mile * (1 / 60 : ℚ)) -- in miles per hour
def rowing_speed_without_current : ℚ := swimming_speed * 2 -- Jake rows at twice his swimming speed

-- Current effects
def against_current_reduction : ℚ := 0.15
def with_current_increase : ℚ := 0.10
def rowing_speed_against_current : ℚ := rowing_speed_without_current * (1 - against_current_reduction)
def rowing_speed_with_current : ℚ := rowing_speed_without_current * (1 + with_current_increase)

-- Time to row each side
def time_to_row_side_against_current : ℚ := lake_side_length / rowing_speed_against_current
def time_to_row_side_with_current : ℚ := lake_side_length / rowing_speed_with_current

-- Proof of total time
theorem total_time_to_row_around_lake : 
  2 * time_to_row_side_against_current + 2 * time_to_row_side_with_current = 17.38 :=
by {
  -- calculations
  have swim_speed_calc : swimming_speed = 3 := by sorry,
  have rowing_speed_calc : rowing_speed_without_current = 6 := by sorry,
  have against_speed_calc : rowing_speed_against_current = 5.1 := by sorry,
  have with_speed_calc : rowing_speed_with_current = 6.6 := by sorry,
  have time_against_side_calc : time_to_row_side_against_current = 25 / 5.1 := by sorry,
  have time_with_side_calc : time_to_row_side_with_current = 25 / 6.6 := by sorry,
  calc
    2 * (25 / 5.1) + 2 * (25 / 6.6) = 9.804 + 7.576 : by sorry
    ... = 17.38 : by sorry
}

end total_time_to_row_around_lake_l63_63718


namespace num_routes_M_to_N_l63_63482

-- Define the relevant points and connections as predicates
def can_reach_directly (x y : String) : Prop :=
  if (x = "C" ∧ y = "N") ∨ (x = "D" ∧ y = "N") ∨ (x = "B" ∧ y = "N") then true else false

def can_reach_via (x y z : String) : Prop :=
  if (x = "A" ∧ y = "C" ∧ z = "N") ∨ (x = "A" ∧ y = "D" ∧ z = "N") ∨ (x = "B" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "B" ∧ y = "C" ∧ z = "N") ∨ (x = "E" ∧ y = "B" ∧ z = "N") ∨ (x = "F" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "F" ∧ y = "B" ∧ z = "N") then true else false

-- Define a function to compute the number of ways from a starting point to "N"
noncomputable def num_routes_to_N : String → ℕ
| "N" => 1
| "C" => 1
| "D" => 1
| "A" => 2 -- from C to N and D to N
| "B" => 4 -- from B to N directly, from B to N via A (2 ways), from B to N via C
| "E" => 4 -- from E to N via B
| "F" => 6 -- from F to N via A (2 ways), from F to N via B (4 ways)
| "M" => 16 -- from M to N via A, B, E, F
| _ => 0

-- The theorem statement
theorem num_routes_M_to_N : num_routes_to_N "M" = 16 :=
by
  sorry

end num_routes_M_to_N_l63_63482


namespace angle_between_vectors_is_correct_l63_63530

noncomputable def vector_angle : ℝ :=
  let v₁ := (4 : ℝ, -2 : ℝ)
  let v₂ := (5 : ℝ, 3 : ℝ)
  let dot_product := v₁.1 * v₂.1 + v₁.2 * v₂.2
  let magnitude_v₁ := Real.sqrt (v₁.1 ^ 2 + v₁.2 ^ 2)
  let magnitude_v₂ := Real.sqrt (v₂.1 ^ 2 + v₂.2 ^ 2)
  Real.arccos (dot_product / (magnitude_v₁ * magnitude_v₂))

theorem angle_between_vectors_is_correct :
  vector_angle = Real.arccos ((14 * Real.sqrt 340) / 340) := by
  sorry

end angle_between_vectors_is_correct_l63_63530


namespace at_least_one_non_negative_l63_63168

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 := 
sorry

end at_least_one_non_negative_l63_63168


namespace cot_neg_45_eq_neg_1_l63_63135

-- Hypotheses
variable (θ : ℝ)
variable (h1 : 𝔸.cot θ = 1 / 𝔸.tan θ)
variable (h2 : 𝔸.tan (-45) = -𝔸.tan 45)
variable (h3 : 𝔸.tan 45 = 1)

-- Theorem
theorem cot_neg_45_eq_neg_1 :
  𝔸.cot (-45) = -1 := by
  sorry

end cot_neg_45_eq_neg_1_l63_63135


namespace lansing_elementary_schools_l63_63719

theorem lansing_elementary_schools :
  ∀ (students_per_school total_students : ℕ), students_per_school = 247 ∧ total_students = 6175 → total_students / students_per_school = 25 :=
by
  intros students_per_school total_students h
  rcases h with ⟨h1, h2⟩
  rw [h1, h2]
  norm_num
  sorry

end lansing_elementary_schools_l63_63719


namespace police_coverage_l63_63363

theorem police_coverage (A B C D E F G H I J K : Prop)
(street1 : A ∧ B ∧ C ∧ D)
(street2 : E ∧ F ∧ G)
(street3 : H ∧ I ∧ J ∧ K)
(street4 : A ∧ E ∧ H)
(street5 : B ∧ F ∧ I)
(street6 : D ∧ G ∧ J)
(street7 : H ∧ F ∧ C)
(street8 : C ∧ G ∧ K)
(policeman_at_B : B)
(policeman_at_G : G)
(policeman_at_H : H) :
(street1 ∨ street2 ∨ street3 ∨ street4 ∨ street5 ∨ street6 ∨ street7 ∨ street8) :=
sorry

end police_coverage_l63_63363


namespace triangle_perimeter_l63_63070

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 12) (h2 : b = 15) (h3 : c = 9)
    (ineq1: a + b > c) (ineq2: a + c > b) (ineq3: b + c > a) :
    a + b + c = 36 := 
by
  -- Given the conditions, the remaining task is proving the statement
  rw [h1, h2, h3]
  norm_num
  sorry

end triangle_perimeter_l63_63070


namespace min_value_inequality_l63_63732

open Real

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 3 * a + 2 * b + c ≥ 18 := 
sorry

end min_value_inequality_l63_63732


namespace domain_f_l63_63102

def domain_of_f (x : ℝ) : Prop :=
  (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x < 4)

theorem domain_f :
  ∀ x, domain_of_f x ↔ (x ≥ 2 ∧ x < 4) ∧ x ≠ 3 :=
by
  sorry

end domain_f_l63_63102


namespace total_number_of_games_l63_63801

def number_of_teams : Nat := 10
def games_per_team_outside_conference : Nat := 6

theorem total_number_of_games : 90 + 60 = 150 :=
by
  -- conference games calculation
  have conference_games : Nat := ((number_of_teams * (number_of_teams - 1)) / 2) * 2
  -- non-conference games calculation
  have non_conference_games : Nat := number_of_teams * games_per_team_outside_conference
  -- total games calculation
  have total_games : Nat := conference_games + non_conference_games
  -- assertion
  show total_games = 150
  from rfl

end total_number_of_games_l63_63801


namespace multiple_of_pumpkins_l63_63938

theorem multiple_of_pumpkins (M S : ℕ) (hM : M = 14) (hS : S = 54) (h : S = x * M + 12) : x = 3 := sorry

end multiple_of_pumpkins_l63_63938


namespace q_sufficient_not_necessary_p_l63_63570

theorem q_sufficient_not_necessary_p (x : ℝ) (p : Prop) (q : Prop) :
  (p ↔ |x| < 2) →
  (q ↔ x^2 - x - 2 < 0) →
  (q → p) ∧ (p ∧ ¬q) :=
by
  sorry

end q_sufficient_not_necessary_p_l63_63570


namespace total_cost_sean_bought_l63_63772

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end total_cost_sean_bought_l63_63772


namespace max_value_of_f_l63_63032

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ∈ Icc (-1 : ℝ) 0 then -a else a * 2 * x - 4 * x

theorem max_value_of_f (a : ℝ) :
  ∃ M : ℝ, (∀ x ∈ Icc (0 : ℝ) 1, f a x ≤ M) ∧
  ((a ≤ 2 ∧ M = a - 1) ∨
   (2 < a ∧ a < 4 ∧ M = a ^ 2 / 4) ∨
   (4 ≤ a ∧ M = 2 * a - 4)) :=
sorry

end max_value_of_f_l63_63032


namespace solve_eq_sqrt_2_sub_3z_9_l63_63141

theorem solve_eq_sqrt_2_sub_3z_9 :
  (z : ℝ) → sqrt (2 - 3 * z) = 9 → z = -79 / 3 :=
by
  intro z h
  sorry

end solve_eq_sqrt_2_sub_3z_9_l63_63141


namespace correct_conclusions_l63_63568

noncomputable def f (x : ℝ) : ℝ := f(0) * Real.exp x + (f'(0) / 2) * x

axiom f_prime_at_1 : f' 1 = Real.exp 1 + 1

theorem correct_conclusions :
  (∃ x0 : ℝ, f (x0 - 1) < 2 * x0 - 1) = false ∧
  (∃ x0 : ℝ, -1 < x0 ∧ x0 < -1/2 ∧ f x0 = 0) = true ∧
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 + f x2) / 2 < f ((x1 + x2) / 2)) = false ∧
  (∀ a : ℝ, (∃ x1 x2, (x1 ≠ x2) ∧ y - (f x1) = (Real.exp x1 + 1) * (1 - x1) 
                          ∧ y - (f x2) = (Real.exp x2 + 1) * (1 - x2)) → 1 < a ∧ a < Real.exp 1 + 1) = true := 
sorry

end correct_conclusions_l63_63568


namespace sum_of_numbers_with_lcm_36_and_ratio_2_3_l63_63017

theorem sum_of_numbers_with_lcm_36_and_ratio_2_3 :
  ∃ a b : ℕ, nat.lcm a b = 36 ∧ a = 2 * b / 3 ∧ (a + b) = 60 :=
by
  sorry

end sum_of_numbers_with_lcm_36_and_ratio_2_3_l63_63017


namespace calculate_eccentricity_max_area_triangle_l63_63581

noncomputable def ellipse_eccentricity (a b c : ℝ) (h1 : 0 < b) (h2 : b < a) : ℝ :=
  c / a

theorem calculate_eccentricity (a b c : ℝ) (P : ℝ × ℝ) 
    (h1 : 0 < b) (h2 : b < a) 
    (h3 : P.1 = c) (h4 : c^2 = a^2 - b^2) 
    (dot_product : ((-2*c, -b^2/a) : ℝ × ℝ) • (0, -b^2/a) = 1/16 * a^2) : 
  ellipse_eccentricity a b c h1 h2 = sqrt 3 / 2 :=
by
  sorry

theorem max_area_triangle (a b c : ℝ) (P : ℝ × ℝ) 
    (h1 : 0 < b) (h2 : b < a) 
    (h3 : P.1 = c) (h4 : P.2 = b^2/a)
    (h5 : |(-c, 0) - P| + |(c, 0) - P| = 2a)
    (perimeter : 2a + 2c = 2 + √3) : 
  ∃ (A B : ℝ × ℝ), 
    triangle_area (A::B::(c, 0)::[]) = 1/2 :=
by
  sorry

end calculate_eccentricity_max_area_triangle_l63_63581


namespace hundredth_odd_integer_l63_63860

theorem hundredth_odd_integer : ∃ (x : ℕ), 2 * x - 1 = 199 ∧ x = 100 :=
by
  use 100
  split
  . exact calc
      2 * 100 - 1 = 200 - 1 : by ring
      _ = 199 : by norm_num
  . refl

end hundredth_odd_integer_l63_63860


namespace smallest_lcm_l63_63649

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l63_63649


namespace smallest_lcm_value_theorem_l63_63650

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end smallest_lcm_value_theorem_l63_63650


namespace distinct_permutations_mod_factorial_l63_63744

open BigOperators

theorem distinct_permutations_mod_factorial 
  (n : ℕ) (hn : n > 1 ∧ Odd n) 
  (k : Fin n → ℤ) : 
  ∃ (b c : Fin n → Fin n), b ≠ c ∧ ∑ i in Finset.univ, k i * b i - ∑ i in Finset.univ, k i * c i ≡ 0 [MOD nat.factorial n] := 
sorry

end distinct_permutations_mod_factorial_l63_63744


namespace hexagon_area_proof_l63_63455

-- Define the coordinates of points A and C
def A : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (8, 5)

-- Define a regular hexagon using points A and C
noncomputable def hexagon_area : ℝ :=
  let AC := real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
  2 * (real.sqrt(3) / 4) * (AC) ^ 2

-- The theorem statement that the area of the hexagon is 29√3
theorem hexagon_area_proof : hexagon_area = 29 * real.sqrt 3 :=
by
  -- skip the proof
  sorry

end hexagon_area_proof_l63_63455


namespace possible_distances_l63_63021

structure IslanderRow where
  knights : Finset ℕ
  liars : Finset ℕ
  position : ℕ → ℕ -- Returns the position (0, 1, 2, 3) in the row

noncomputable def islander_statement_distance : IslanderRow → ℕ → ℕ
  | row, i => if i ∈ row.knights then
                if i = 0 then 3
                else if i = 3 then 2
                else 0
              else 0 -- Liars statements can be any values, left as 0 for default

theorem possible_distances (row : IslanderRow) (h1 : row.knights.card = 2)
  (h2 : row.liar.card = 2) : 
  (row.position 1 ∈ row.knights → islander_statement_distance row 1 = 2) ∧ 
  (row.position 2 ∉ row.knights → 
    islander_statement_distance row 2 ∈ {1, 3, 4}) :=
sorry

end possible_distances_l63_63021


namespace y_intercept_of_line_l63_63143

theorem y_intercept_of_line : ∃ y, (0, y) is the y_intercept of the line if and only if 7 * y = 28 :=
by {
  existsi (4 : ℝ),
  split,
  { intro h₀,
    rw [←h₀],
    field_simp,
    norm_num, },
  { intro h1,
    exact h1, },
  sorry,
}

end y_intercept_of_line_l63_63143


namespace problem1_range_of_x_problem2_value_of_a_l63_63603

open Set

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |x + 3| + |x - a|

-- Problem 1
theorem problem1_range_of_x (a : ℝ) (h : a = 4) (h_eq : ∀ x : ℝ, f x a = 7 ↔ x ∈ Icc (-3 : ℝ) 4) :
  ∀ x : ℝ, f x 4 = 7 ↔ x ∈ Icc (-3 : ℝ) 4 := by
  sorry

-- Problem 2
theorem problem2_value_of_a (h₁ : ∀ x : ℝ, x ∈ {x : ℝ | f x 4 ≥ 6} ↔ x ≤ -4 ∨ x ≥ 2) :
  f x a ≥ 6 ↔  x ≤ -4 ∨ x ≥ 2 :=
  by
  sorry

end problem1_range_of_x_problem2_value_of_a_l63_63603


namespace bottles_per_case_l63_63624

theorem bottles_per_case (days: ℕ) (daily_intake: ℚ) (total_spent: ℚ) (case_cost: ℚ) (total_cases: ℕ) (total_bottles: ℕ) (B: ℕ) 
    (H1 : days = 240)
    (H2 : daily_intake = 1/2)
    (H3 : total_spent = 60)
    (H4 : case_cost = 12)
    (H5 : total_cases = total_spent / case_cost)
    (H6 : total_bottles = days * daily_intake)
    (H7 : B = total_bottles / total_cases) :
    B = 24 :=
by
    sorry

end bottles_per_case_l63_63624


namespace calculate_profit_l63_63709

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l63_63709


namespace cot_neg_45_is_neg_1_l63_63123

theorem cot_neg_45_is_neg_1 : Real.cot (Real.pi * -45 / 180) = -1 :=
by
  sorry

end cot_neg_45_is_neg_1_l63_63123


namespace five_fourths_sum_eq_five_l63_63142

-- Definitions for the conditions in math problem
def fraction1 := 6 / 3
def fraction2 := 8 / 4
def sum_fractions := fraction1 + fraction2
def five_fourths (x : ℝ) := (5 / 4) * x

-- Statement to be proven
theorem five_fourths_sum_eq_five : five_fourths sum_fractions = 5 :=
by
  -- Definitions
  have h_fraction1 := fraction1 = 2
  have h_fraction2 := fraction2 = 2
  have h_sum_fractions := sum_fractions = 4
  -- Proof
  sorry

end five_fourths_sum_eq_five_l63_63142


namespace max_sum_of_16_ages_at_least_k_l63_63118

theorem max_sum_of_16_ages_at_least_k (ages : Fin 50 → ℕ) (h_distinct : Function.Injective ages) (h_sum : (∑ i, ages i) = 1555) :
  ∃ (k : ℕ), k = 776 ∧ ∃ (S : Finset (Fin 50)), S.card = 16 ∧ (∑ i in S, ages i) ≥ k :=
by {
  -- This is the definition of the theorem, proof is omitted
  sorry
}

end max_sum_of_16_ages_at_least_k_l63_63118


namespace simplify_f_solve_for_tan_l63_63569

noncomputable def f (α : ℝ) : ℝ := 
  (tan (-α - π) * sin (-α - π)^2) / 
  (sin (α - π/2) * cos (π/2 + α) * tan (π - α))

theorem simplify_f (α : ℝ) : f(α) = tan(α) := sorry

theorem solve_for_tan 
  (h : f(α) = 2) : (3 * sin(α) + cos(α)) / (2 * sin(α) - cos(α)) = 7 / 3 := sorry

end simplify_f_solve_for_tan_l63_63569


namespace three_num_ordering_l63_63104

open Real

noncomputable def log := Real.log10

theorem three_num_ordering :
  0.76 < 1 →
  (∀ x y : ℝ, x < y → log x < log y) →
  log 1 = 0 →
  log 0.76 < 0.76 ∧ 0.76 < 60.7 :=
by
  intros h1 h2 h3
  have h4 : log 0.76 < log 1 := h2 0.76 1 h1
  have h5 : log 0.76 < 0 := h4.trans_eq h3
  split
  . exact h5.trans_le le_of_lt
  . linarith

end three_num_ordering_l63_63104


namespace quartic_polynomial_unique_l63_63978

noncomputable def q (x : ℝ) : ℝ := - (1 / 6) * x ^ 4 + (4 / 3) * x ^ 3 - (4 / 3) * x ^ 2 - (8 / 3) * x

theorem quartic_polynomial_unique :
  (q 1 = -3) ∧ (q 2 = -5) ∧ (q 3 = -9) ∧ (q 4 = -17) ∧ (q 5 = -35) :=
by
  -- Assuming the correctness of q
  have hq : q(1) = (-1/6) * 1^4 + (4/3) * 1^3 - (4/3) * 1^2 - (8/3) * 1 := by sorry
  have h1 : q(1) = -3 := by rw [hq]; sorry
  have h2 : q(2) = -5 := by sorry
  have h3 : q(3) = -9 := by sorry
  have h4 : q(4) = -17 := by sorry
  have h5 : q(5) = -35 := by sorry
  exact ⟨h1, ⟨h2, ⟨h3, ⟨h4, h5⟩⟩⟩⟩.

end quartic_polynomial_unique_l63_63978


namespace sequence_satisfies_condition_l63_63258

noncomputable def a_n (n : ℕ) (h : 0 < n) : ℝ := 
  real.sqrt n - real.sqrt (n - 1)

def S_n (n : ℕ) (h : 0 < n) : ℝ :=
  (∑ i in finset.range n, a_n (i + 1) (nat.succ_pos i))

theorem sequence_satisfies_condition (n : ℕ) (h : 0 < n) : 
  S_n n h = real.sqrt n :=
by
  sorry

end sequence_satisfies_condition_l63_63258


namespace hundredth_odd_integer_l63_63857

theorem hundredth_odd_integer : ∃ (x : ℕ), 2 * x - 1 = 199 ∧ x = 100 :=
by
  use 100
  split
  . exact calc
      2 * 100 - 1 = 200 - 1 : by ring
      _ = 199 : by norm_num
  . refl

end hundredth_odd_integer_l63_63857


namespace complex_pure_imaginary_l63_63668

theorem complex_pure_imaginary (a : ℝ) : 
  ((a^2 - 3*a + 2) = 0) → (a = 2) := 
  by 
  sorry

end complex_pure_imaginary_l63_63668


namespace average_speed_of_train_is_correct_l63_63067

variable (x : ℝ)

-- Definitions based on conditions.
def time_for_x_km_at_30_kmph : ℝ := x / 30
def time_for_2x_km_at_20_kmph : ℝ := 2 * x / 20
def total_time : ℝ := time_for_x_km_at_30_kmph x + time_for_2x_km_at_20_kmph x
def total_distance : ℝ := 3 * x

-- The average speed calculation.
def average_speed : ℝ := total_distance x / total_time x

-- Statement to be proven.
theorem average_speed_of_train_is_correct (x : ℝ) : average_speed x = 22.5 := by
  sorry

end average_speed_of_train_is_correct_l63_63067


namespace total_cost_of_items_l63_63777

theorem total_cost_of_items (cost_of_soda : ℕ) (cost_of_soup : ℕ) (cost_of_sandwich : ℕ) (total_cost : ℕ) 
  (h1 : cost_of_soda = 1)
  (h2 : cost_of_soup = 3 * cost_of_soda)
  (h3 : cost_of_sandwich = 3 * cost_of_soup) :
  total_cost = 3 * cost_of_soda + 2 * cost_of_soup + cost_of_sandwich :=
by
  unfold total_cost
  show 3 * 1 + 2 * (3 * 1) + (3 * (3 * 1)) = 18
  rfl

end total_cost_of_items_l63_63777


namespace findAnalyticalExpression_l63_63370

-- Defining the point A as a structure with x and y coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Defining a line as having a slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Condition: Line 1 is parallel to y = 2x - 3
def line1 : Line := {slope := 2, intercept := -3}

-- Condition: Line 2 passes through point A
def point_A : Point := {x := -2, y := -1}

-- The theorem statement:
theorem findAnalyticalExpression : 
  ∃ b : ℝ, (∀ x : ℝ, (point_A.y = line1.slope * point_A.x + b) → b = 3) ∧ 
            ∀ x : ℝ, (line1.slope * x + b = 2 * x + 3) :=
sorry

end findAnalyticalExpression_l63_63370


namespace max_value_of_expr_l63_63547

noncomputable def max_value_expr (a b c x y z : ℝ) : ℝ :=
  (a / x) + (a + b) / (x + y) + (a + b + c) / (x + y + z)

theorem max_value_of_expr {a b c x y z : ℝ} 
  (ha : 2 ≤ a ∧ a ≤ 3) (hb : 2 ≤ b ∧ b ≤ 3) (hc : 2 ≤ c ∧ c ≤ 3)
  (hperm : {x, y, z} = {a, b, c}) : 
  max_value_expr a b c x y z ≤ 15 / 4 :=
by
  sorry

end max_value_of_expr_l63_63547


namespace count_integers_in_interval_l63_63638

theorem count_integers_in_interval :
  {n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi}.toFinset.card = 57 :=
by
  sorry

end count_integers_in_interval_l63_63638


namespace area_of_pentagon_PTRSQ_l63_63318

theorem area_of_pentagon_PTRSQ (PQRS : Type) [geometry PQRS]
  {P Q R S T : PQRS} 
  (h1 : square P Q R S) 
  (h2 : perp PT TR) 
  (h3 : distance P T = 5) 
  (h4 : distance T R = 12) : 
  area_pentagon PTRSQ = 139 :=
sorry

end area_of_pentagon_PTRSQ_l63_63318


namespace find_alpha_plus_beta_l63_63162

theorem find_alpha_plus_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = (Real.sqrt 5) / 5) (h4 : Real.sin β = (3 * Real.sqrt 10) / 10) : 
  α + β = 3 * π / 4 :=
sorry

end find_alpha_plus_beta_l63_63162


namespace more_apples_than_pears_l63_63392

-- Define the variables
def apples := 17
def pears := 9

-- Theorem: The number of apples minus the number of pears equals 8
theorem more_apples_than_pears : apples - pears = 8 :=
by
  sorry

end more_apples_than_pears_l63_63392


namespace problem1_problem2_l63_63366

-- Define the function f based on the given conditions
variables (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, f(x + y) - f(y) = (x + 2*y + 2)*x)
variable (h2 : f 2 = 12)

-- The first task is to prove that f(0) = 4
theorem problem1 : f 0 = 4 :=
sorry

-- The second task is to determine the range of a for which there exists an x₀ in (1, 4)
-- such that f(x₀) - 8 = a*x₀
theorem problem2 : ∃ a : ℝ, (∀ x₀ : ℝ, (1 < x₀ ∧ x₀ < 4) → f(x₀) - 8 = a*x₀) → a ∈ set.Ioo (-1 : ℝ) 5 :=
sorry

end problem1_problem2_l63_63366


namespace find_k_l63_63979

theorem find_k (k : ℤ) : (∃ x : ℝ, log x = 4 - x ∧ x ∈ (k : ℝ,↑k + 1)) ↔ k = 3 :=
by
  sorry

end find_k_l63_63979


namespace magnitude_of_sum_l63_63221

variables (p q : ℝ × ℝ) (x : ℝ)
def vector_p := (2 : ℝ, -3 : ℝ)
def vector_q (x : ℝ) := (x : ℝ, 6 : ℝ)
def is_parallel (p q : ℝ × ℝ) : Prop := ∃ k : ℝ, q = (k * p.1, k * p.2)

theorem magnitude_of_sum (h : is_parallel vector_p (vector_q x)) 
  (collinearity : 2 * 6 - (-3) * x = 0) :
  (real.sqrt ((2 + x)^2 + (6 - 3)^2) = real.sqrt 13) :=
sorry

end magnitude_of_sum_l63_63221


namespace integer_part_sum_l63_63827

-- Definition of the sequence {x_n}
noncomputable def x : ℕ → ℚ
| 0     := 1/2
| (n+1) := x n ^ 2 + x n

-- Sum definition
def sum_seq (n : ℕ) : ℚ :=
∑ k in Finset.range n, 1 / (1 + x (k + 1))

-- Proof statement
theorem integer_part_sum (n : ℕ) (h : n = 2000) : 
  ⌊sum_seq n⌋ = 1 :=
by
  rw h
  sorry

end integer_part_sum_l63_63827


namespace Smith_gave_Randy_l63_63769

theorem Smith_gave_Randy {original_money Randy_keeps gives_Sally Smith_gives : ℕ}
  (h1: original_money = 3000)
  (h2: Randy_keeps = 2000)
  (h3: gives_Sally = 1200)
  (h4: Randy_keeps + gives_Sally = original_money + Smith_gives) :
  Smith_gives = 200 :=
by
  sorry

end Smith_gave_Randy_l63_63769


namespace pentagon_roll_final_position_l63_63061

-- Define the angles involved
def interior_angle_octagon : ℝ := (8 - 2) * 180 / 8
def interior_angle_pentagon : ℝ := (5 - 2) * 180 / 5

-- Define the rotation per movement
def rotation_per_movement : ℝ := 360 - (interior_angle_octagon + interior_angle_pentagon)

-- Define the total rotation after one full roll
def total_rotation : ℝ := 8 * rotation_per_movement

-- Define the effective rotation (modulo 360 degrees)
def effective_rotation : ℝ := total_rotation % 360

-- Prove the final position of the triangle
theorem pentagon_roll_final_position :
  effective_rotation = 216 → "the solid triangle is two vertices to the left of its original bottom position" :=
by
  sorry

end pentagon_roll_final_position_l63_63061


namespace pentagon_PTRSQ_area_proof_l63_63314

-- Define the geometric setup and properties
def quadrilateral_PQRS_is_square (P Q R S T : Type) : Prop :=
  -- Here, we will skip the precise geometric construction and assume the properties directly.
  sorry

def segment_PT_perpendicular_to_TR (P T R : Type) : Prop :=
  sorry

def PT_eq_5 (PT : ℝ) : Prop :=
  PT = 5

def TR_eq_12 (TR : ℝ) : Prop :=
  TR = 12

def area_PTRSQ (area : ℝ) : Prop :=
  area = 139

theorem pentagon_PTRSQ_area_proof
  (P Q R S T : Type)
  (PQRS_is_square : quadrilateral_PQRS_is_square P Q R S T)
  (PT_perpendicular_TR : segment_PT_perpendicular_to_TR P T R)
  (PT_length : PT_eq_5 5)
  (TR_length : TR_eq_12 12)
  : area_PTRSQ 139 :=
  sorry

end pentagon_PTRSQ_area_proof_l63_63314


namespace num_even_five_digit_numbers_num_less_than_32000_five_digit_numbers_l63_63992

-- Definitions about the problem
def digits := {0, 1, 2, 3, 4}

-- Problem 1: Prove the number of five-digit even numbers using the given digits is 60
theorem num_even_five_digit_numbers : 
  ∃! l : List ℕ, l.length = 5 ∧ l.to_set ⊆ digits ∧ l.nodup ∧ (l.last! % 2 = 0) ∧ l.permutations.length = 60 := sorry

-- Problem 2: Prove the number of five-digit numbers less than 32000 using the given digits is 54
theorem num_less_than_32000_five_digit_numbers : 
  ∃! l : List ℕ, l.length = 5 ∧ l.to_set ⊆ digits ∧ l.nodup ∧ (l.head! < 3 ∨ (l.head! = 3 ∧ l.get? 1 = some 1)) ∧ l.permutations.length = 54 := sorry

end num_even_five_digit_numbers_num_less_than_32000_five_digit_numbers_l63_63992


namespace proof_problem_1_proof_problem_2_l63_63622

variables (a b : ℝ^3)

noncomputable def magnitude (v : ℝ^3) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def angle_between (u v : ℝ^3) : ℝ :=
  Real.arccos ((u.1 * v.1 + u.2 * v.2 + u.3 * v.3) / (magnitude u * magnitude v))

noncomputable def problem_conditions : Prop :=
  magnitude a = 1 ∧ magnitude b = 6 ∧ (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2) + a.3 * (b.3 - a.3)) = 2

noncomputable def q1_answer : ℝ := Real.pi / 3

noncomputable def q2_answer (a b : ℝ^3) : ℝ :=
  magnitude (⟨2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3⟩)

theorem proof_problem_1 (h : problem_conditions a b) : angle_between a b = q1_answer := sorry

theorem proof_problem_2 (h : problem_conditions a b) : q2_answer a b = 2 * Real.sqrt 7 := sorry

end proof_problem_1_proof_problem_2_l63_63622


namespace cot_neg_45_is_neg_1_l63_63120

theorem cot_neg_45_is_neg_1 : Real.cot (Real.pi * -45 / 180) = -1 :=
by
  sorry

end cot_neg_45_is_neg_1_l63_63120


namespace sum_integer_solutions_l63_63154

open Polynomial

theorem sum_integer_solutions : 
    (finset.univ.filter (λ x : ℤ, (x^2 - 21*x + 100 = 0)).sum (λ x, x)) = 0 :=
sorry

end sum_integer_solutions_l63_63154


namespace floor_ratio_of_coefficients_l63_63742

-- Definitions based on given conditions.
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def a_k (k : ℕ) : ℕ := (Finset.range 99).sum (λ n, binomial_coefficient (n + 1) k)

-- Statement of our proof problem.
theorem floor_ratio_of_coefficients : 
  (Int.floor (a_k 4 / a_k 3 : ℚ) = 19) :=
sorry -- Proof to be filled in.

end floor_ratio_of_coefficients_l63_63742


namespace length_of_intervals_l63_63337

noncomputable def sum_fractions (x : ℝ) : ℝ :=
  ∑ k in Finset.range 1 64, (k : ℝ) / (x - k)

theorem length_of_intervals :
  { x : ℝ | sum_fractions x ≥ 1 }.to_Union_Icc.Inter.le (64 - 1 : ℝ) = (2016 : ℝ) :=
  sorry

end length_of_intervals_l63_63337


namespace cube_root_equation_l63_63669

theorem cube_root_equation (x : ℝ) (h : (2 * x - 14)^(1/3) = -2) : 2 * x + 3 = 9 := by
  sorry

end cube_root_equation_l63_63669


namespace series_items_increase_l63_63849

theorem series_items_increase (n : ℕ) (hn : n ≥ 2) :
  (2^n + 1) - 2^(n-1) - 1 = 2^(n-1) :=
by
  sorry

end series_items_increase_l63_63849


namespace compute_star_expression_l63_63098

def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

theorem compute_star_expression : star 2 (star 3 (star 4 5)) = (1 / 4) := by
  sorry

end compute_star_expression_l63_63098


namespace sin_3pi_over_4_minus_alpha_l63_63165

theorem sin_3pi_over_4_minus_alpha (α : ℝ) (h : sin (π / 4 + α) = 3 / 5) :
  sin (3 * π / 4 - α) = 3 / 5 :=
by
  -- since we rely on the result, we don't need to show the proof steps
  sorry

end sin_3pi_over_4_minus_alpha_l63_63165


namespace cot_neg_45_is_neg_1_l63_63124

theorem cot_neg_45_is_neg_1 : Real.cot (Real.pi * -45 / 180) = -1 :=
by
  sorry

end cot_neg_45_is_neg_1_l63_63124


namespace math_problem_l63_63874

theorem math_problem :
  (10^2 + 6^2) / 2 = 68 :=
by
  sorry

end math_problem_l63_63874


namespace quadratic_real_roots_l63_63658

theorem quadratic_real_roots (b : ℝ) :
  ∃ x : ℝ, (x^2 + b * x + 25 = 0) ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 :=
by sorry

end quadratic_real_roots_l63_63658


namespace lines_intersect_at_point_l63_63450

/-
Given two lines parameterized as:
Line 1: (x, y) = (2, 0) + s * (3, -4)
Line 2: (x, y) = (6, -10) + v * (5, 3)
Prove that these lines intersect at (242/29, -248/29).
-/

def parametric_line_1 (s : ℚ) : ℚ × ℚ :=
  (2 + 3 * s, -4 * s)

def parametric_line_2 (v : ℚ) : ℚ × ℚ :=
  (6 + 5 * v, -10 + 3 * v)

theorem lines_intersect_at_point :
  ∃ (s v : ℚ), parametric_line_1 s = parametric_line_2 v ∧ parametric_line_1 s = (242 / 29, -248 / 29) :=
sorry

end lines_intersect_at_point_l63_63450


namespace prob_three_blue_is_correct_l63_63899

-- Definitions corresponding to the problem conditions
def total_jellybeans : ℕ := 20
def blue_jellybeans_start : ℕ := 10
def red_jellybeans : ℕ := 10

-- Probabilities calculation steps as definitions
def prob_first_blue : ℚ := blue_jellybeans_start / total_jellybeans
def prob_second_blue_given_first_blue : ℚ := (blue_jellybeans_start - 1) / (total_jellybeans - 1)
def prob_third_blue_given_first_two_blue : ℚ := (blue_jellybeans_start - 2) / (total_jellybeans - 2)

-- Total probability of drawing three blue jellybeans
def prob_three_blue : ℚ := 
  prob_first_blue *
  prob_second_blue_given_first_blue *
  prob_third_blue_given_first_two_blue

-- Formal statement of the proof problem
theorem prob_three_blue_is_correct : prob_three_blue = 2 / 19 :=
by
  -- Fill the proof here
  sorry

end prob_three_blue_is_correct_l63_63899


namespace maximum_value_of_A_l63_63981

theorem maximum_value_of_A {x y z : ℝ} 
  (h1 : 0 < x) (h2 : x ≤ 1) 
  (h3 : 0 < y) (h4 : y ≤ 1) 
  (h5 : 0 < z) (h6 : z ≤ 1) : 
  let A := (sqrt (8 * x^4 + y) + sqrt (8 * y^4 + z) + sqrt (8 * z^4 + x) - 3) / (x + y + z) in
  A ≤ 2 :=
by sorry

end maximum_value_of_A_l63_63981


namespace number_of_other_values_l63_63466

def orig_value : ℕ := 2 ^ (2 ^ (2 ^ 2))

def other_values : Finset ℕ :=
  {2 ^ (2 ^ (2 ^ 2)), 2 ^ ((2 ^ 2) ^ 2), ((2 ^ 2) ^ 2) ^ 2, (2 ^ (2 ^ 2)) ^ 2, (2 ^ 2) ^ (2 ^ 2)}

theorem number_of_other_values :
  other_values.erase orig_value = {256} :=
by
  sorry

end number_of_other_values_l63_63466


namespace calculation_l63_63556

-- Define s* as greatest positive even integer less than or equal to s
def greatest_even_le (s : ℝ) : ℝ := (if s < 2 then 0 else 2 * ⌊s / 2⌋)

-- The main statement we need to prove
theorem calculation : 5.2 - greatest_even_le 5.2 = 1.2 :=
by
  sorry -- Proof omitted

end calculation_l63_63556


namespace sum_of_all_paintable_integers_l63_63091

def harold_paints (h : ℕ) (picket : ℕ) : Prop :=
  picket % h = 1

def tanya_paints (t : ℕ) (picket : ℕ) : Prop :=
  picket % t = 2

def ulysses_paints (u : ℕ) (picket : ℕ) : Prop :=
  picket % u = 0

def paintable (h t u : ℕ) : Prop :=
  (∀ picket, (harold_paints h picket ∨ tanya_paints t picket ∨ ulysses_paints u picket) ↔ 
             (harold_paints h picket → ¬ tanya_paints t picket) ∧
             (harold_paints h picket → ¬ ulysses_paints u picket) ∧
             (tanya_paints t picket → ¬ ulysses_paints u picket) ∧
             (∑ p in (finset.range 100), (if harold_paints h p then 1 else 0) +
             (if tanya_paints t p then 1 else 0) + (if ulysses_paints u p then 1 else 0)) = 2 * (finset.range 100).card)

theorem sum_of_all_paintable_integers :
  ∃ h t u, paintable h t u ∧ 100 * h + 10 * t + u = 442 :=
sorry


end sum_of_all_paintable_integers_l63_63091


namespace fire_alarms_and_passengers_discrete_l63_63891

-- Definitions of the random variables
def xi₁ : ℕ := sorry  -- number of fire alarms in a city within one day
def xi₂ : ℝ := sorry  -- temperature in a city within one day
def xi₃ : ℕ := sorry  -- number of passengers at a train station in a city within a month

-- Defining the concept of discrete random variable
def is_discrete (X : Type) : Prop := 
  ∃ f : X → ℕ, ∀ x : X, ∃ n : ℕ, f x = n

-- Statement of the proof problem
theorem fire_alarms_and_passengers_discrete :
  is_discrete ℕ ∧ is_discrete ℕ ∧ ¬ is_discrete ℝ :=
by
  have xi₁_discrete : is_discrete ℕ := sorry
  have xi₃_discrete : is_discrete ℕ := sorry
  have xi₂_not_discrete : ¬ is_discrete ℝ := sorry
  exact ⟨xi₁_discrete, xi₃_discrete, xi₂_not_discrete⟩

end fire_alarms_and_passengers_discrete_l63_63891


namespace probability_of_prime_sum_l63_63399

def is_sum_of_numbers_prime_probability : ℚ :=
  let total_outcomes := 216
  let favorable_outcomes := 73
  favorable_outcomes / total_outcomes

theorem probability_of_prime_sum :
  is_sum_of_numbers_prime_probability = 73 / 216 :=
by
  sorry

end probability_of_prime_sum_l63_63399


namespace g_is_odd_l63_63269

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + 1 / 3

theorem g_is_odd (x : ℝ) : 
  g (-x) = -g x := 
by sorry

end g_is_odd_l63_63269


namespace simplify_expression_l63_63339

def i_squared_neg_one : Prop := complex.i * complex.i = -1

theorem simplify_expression :
  ( -5 + 3 * complex.i - (2 - 7 * complex.i) + (1 + 2 * complex.i) * (4 - 3 * complex.i) ) = 
    (3 + 15 * complex.i) :=
by 
  have h : complex.i * complex.i = -1 := i_squared_neg_one
  sorry

end simplify_expression_l63_63339


namespace minesweeper_sum_invariance_l63_63428

theorem minesweeper_sum_invariance (grid : Matrix (Fin 10) (Fin 10) (Option Nat)) :
    let original_sum := sum (fun i j => match grid i j with | some n => n | none => 0)
        new_grid := Matrix (Fin 10) (Fin 10) (Option Nat)
        new_sum := sum (fun i j => match new_grid i j with | some n => n | none => 0)
    in original_sum = new_sum :=
  sorry

end minesweeper_sum_invariance_l63_63428


namespace colored_rectangle_exists_l63_63517

theorem colored_rectangle_exists (p : ℕ) (c : ℤ × ℤ → Fin p) : 
  ∃ (a b c d : ℤ × ℤ), a ≠ b ∧ a = c ∧ b = d ∧ c ≠ d ∧ (c a = c b) ∧ (c a = c c) ∧ (c a = c d) ∧ (c b = c c) ∧ (c b = c d) ∧ (c c = c d) := 
sorry

end colored_rectangle_exists_l63_63517


namespace a1_a2_sum_l63_63725

theorem a1_a2_sum (a0 a1 a2 a3 : ℝ) (x : ℝ) (h : x ≠ 0) :
  (1 - (2 / x)) ^ 3 = a0 + a1 * (1 / x) + a2 * (1 / x) ^ 2 + a3 * (1 / x) ^ 3 →
  a1 + a2 = 6 :=
by
  intro h_eq
  -- Expand the left-hand side using the binomial theorem
  have binom_exp : (1 - (2 / x)) ^ 3 = 1 - 6 * (1 / x) + 12 * (1 / x) ^ 2 - 8 * (1 / x) ^ 3,
  { sorry },
  -- Comparing coefficients from the binomial expansion and h_eq
  have h_coeffs : (a0, a1, a2, a3) = (1, -6, 12, -8),
  { sorry },
  -- From h_coeffs we get a1 = -6 and a2 = 12
  rw [h_coeffs],
  -- Finally, summing a1 and a2 gives:
  show -6 + 12 = 6

end a1_a2_sum_l63_63725


namespace find_number_l63_63895

theorem find_number (x : ℝ) (h : 1345 - x / 20.04 = 1295) : x = 1002 :=
sorry

end find_number_l63_63895


namespace hundredth_odd_integer_l63_63861

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l63_63861


namespace selling_price_approx_l63_63918

noncomputable def cost_price : Real := 17
noncomputable def selling_price : Real := (5 / 6) * cost_price

theorem selling_price_approx :
  selling_price ≈ 14.17 := by
  -- Proof goes here
  sorry

end selling_price_approx_l63_63918


namespace intersection_is_empty_l63_63103

-- Define the domain and range sets
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | 0 < x}

-- The Lean theorem to prove that the intersection of A and B is the empty set
theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l63_63103


namespace tangent_line_at_one_range_of_m_l63_63597

noncomputable def f (x : ℝ) : ℝ := 4 * Real.log x - 2 * x^2 + 3 * x

-- Condition 1: a = 1
theorem tangent_line_at_one : 
  TangentLineEquation (f 1) 1 (f' 1) :=
by sorry

-- Define function g
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 4 * Real.log x - 2 * x^2 + m

-- Condition 2: g(x) has two zeros in the interval [1/e, e]
theorem range_of_m (m : ℝ) : 
  (2 : ℝ) < m ∧ m ≤ (4 + 2 / Real.exp 2) :=
by sorry

end tangent_line_at_one_range_of_m_l63_63597


namespace min_project_time_l63_63480

theorem min_project_time (A B C : ℝ) (D : ℝ := 12) :
  (1 / B + 1 / C) = 1 / 2 →
  (1 / A + 1 / C) = 1 / 3 →
  (1 / A + 1 / B) = 1 / 4 →
  (1 / D) = 1 / 12 →
  ∃ x : ℝ, x = 8 / 5 ∧ 1 / x = 1 / A + 1 / B + 1 / C + 1 / (12:ℝ) :=
by
  intros h1 h2 h3 h4
  -- Combination of given hypotheses to prove the goal
  sorry

end min_project_time_l63_63480


namespace question_I_question_II_l63_63585

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)

def F (x a : ℝ) : ℝ := f x a + f (2 * x) a

def g (x a : ℝ) : ℝ := f x a - f (-x) a

theorem question_I (a : ℝ) (h₁ : a > 0) (h₂ : ∀ x, F x a ≥ 3) : a = 6 :=
sorry

theorem question_II (m n : ℝ) (hm : m > 0) (hn : n > 0) (h₁ : g 0 2 = 4) (h₂ : 2 * m + 3 * n = 4) :
  (1 / m + 2 / (3 * n)) ≥ 2 :=
sorry

end question_I_question_II_l63_63585


namespace smallest_lcm_l63_63648

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l63_63648


namespace rectangle_area_l63_63060

def radius : ℝ := 10
def width : ℝ := 2 * radius
def length : ℝ := 3 * width
def area_of_rectangle : ℝ := length * width

theorem rectangle_area : area_of_rectangle = 1200 :=
  by sorry

end rectangle_area_l63_63060


namespace max_profit_at_9_l63_63907

def variable_cost (x : ℝ) : ℝ :=
  if x ≤ 14 then (2/3) * x^2 + 4 * x else 17 * x + 400 / x - 80

def profit (x : ℝ) : ℝ :=
  if x ≤ 14 then -(2/3) * x^2 + 12 * x - 30 else 50 - x - 400 / x

theorem max_profit_at_9 :
  ∀ (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 35), profit x ≤ profit 9 :=
by
  intros x h1 h2
  -- Sorry, proof will be provided here
  sorry

end max_profit_at_9_l63_63907


namespace find_integer_mod_l63_63975

theorem find_integer_mod (n : ℤ) (h₀ : 10 ≤ n) (h₁ : n ≤ 15) (h₂ : n ≡ 12345 [MOD 7]) : n = 11 := 
by
  sorry

end find_integer_mod_l63_63975


namespace change_count_of_quantities_l63_63308

theorem change_count_of_quantities (A B C P: Type) 
  [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder P] 
  (AB AC BC: Type)
  (ca cb mn perimeter_triangle area_triangle area_trapezoid: Type)
  (midpoint: AC → (P → Type))
  (sum_eq: AC → BC → AB)
  (length_eq_ca_sum: mn = ca) (length_eq_cb_sum: mn = cb)
  (no_change_ca_cb: (ca + cb) = AB)
  (triangle_area: AB → (P → Type)) 
  (trapezoid_area: (AB → mn) → (P → Type)) :
  2 = finset.card (finset.filter (λ q, q ≠ mn ∧ q ≠ perimeter_triangle) {area_triangle, area_trapezoid} ) := 
sorry

end change_count_of_quantities_l63_63308


namespace inverse_function_correct_l63_63533

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) ^ 2 + 1

noncomputable def f_inv (y : ℝ) : ℝ :=
  1 - Real.sqrt (y - 1)

theorem inverse_function_correct (x : ℝ) (hx : x ≥ 2) :
  f_inv x = 1 - Real.sqrt (x - 1) ∧ ∀ y : ℝ, (y ≤ 0) → f y = x → y = f_inv x :=
by {
  sorry
}

end inverse_function_correct_l63_63533


namespace count_integers_in_range_l63_63634

theorem count_integers_in_range : 
  { n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi }.finite.toFinset.card = 56 := 
by 
  sorry

end count_integers_in_range_l63_63634


namespace max_value_expression_l63_63534

theorem max_value_expression (a b c x y z : ℝ) (h1 : 2 ≤ a ∧ a ≤ 3) (h2 : 2 ≤ b ∧ b ≤ 3) (h3 : 2 ≤ c ∧ c ≤ 3)
    (perm : ∃ p : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ), p (a, b, c) = (x, y, z)) :
    (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z) ≤ 15 / 4) :=
begin
  sorry
end

end max_value_expression_l63_63534


namespace length_of_train_l63_63419

-- We define the conditions
def crosses_platform_1 (L : ℝ) : Prop := 
  let v := (L + 100) / 15
  v = (L + 100) / 15

def crosses_platform_2 (L : ℝ) : Prop := 
  let v := (L + 250) / 20
  v = (L + 250) / 20

-- We state the main theorem we need to prove
theorem length_of_train :
  ∃ L : ℝ, crosses_platform_1 L ∧ crosses_platform_2 L ∧ (L = 350) :=
sorry

end length_of_train_l63_63419


namespace hexagon_area_ratio_l63_63042

-- Definitions
def r : ℝ := sorry -- Radius of the circle
def a_large_hexagon : ℝ := (3 * Real.sqrt 3 / 2) * (2 * r) ^ 2
def a_small_hexagon : ℝ := (3 * Real.sqrt 3 / 2) * r ^ 2

-- Theorem statement
theorem hexagon_area_ratio (h₁ : a_large_hexagon = (3 * Real.sqrt 3 / 2) * (2 * r) ^ 2)
                           (h₂ : a_small_hexagon = (3 * Real.sqrt 3 / 2) * r ^ 2) :
                           (a_small_hexagon / a_large_hexagon) = 0.25 :=
sorry

end hexagon_area_ratio_l63_63042


namespace vectorBC_computation_l63_63997

open Vector

def vectorAB : ℝ × ℝ := (2, 4)

def vectorAC : ℝ × ℝ := (1, 3)

theorem vectorBC_computation :
  (vectorAC.1 - vectorAB.1, vectorAC.2 - vectorAB.2) = (-1, -1) :=
sorry

end vectorBC_computation_l63_63997


namespace count_distinct_four_digit_numbers_ending_in_25_l63_63224

-- Define what it means for a number to be a four-digit number according to the conditions in (a).
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define what it means for a number to be divisible by 5 and end in 25 according to the conditions in (a).
def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the problem as a theorem in Lean
theorem count_distinct_four_digit_numbers_ending_in_25 : 
  ∃ (count : ℕ), count = 90 ∧ 
    (∀ n, is_four_digit_number n ∧ ends_in_25 n → n % 5 = 0) :=
by
  sorry

end count_distinct_four_digit_numbers_ending_in_25_l63_63224


namespace trig_identity_proof_l63_63233

theorem trig_identity_proof (α : ℝ) (h1 : Real.tan α = 3) (h2 : α ∈ Ioo (π/4) (π/2)) : 
  Real.sin (2 * α + π / 4) + 2 * Real.cos (π / 4) * (Real.cos α) ^ 2 = 0 :=
by sorry

end trig_identity_proof_l63_63233


namespace parabola_directrix_l63_63670

noncomputable def ellipse_foci (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)

theorem parabola_directrix :
  ∀ (p : ℝ), p = 4 →
  (∀ (a b : ℝ), a = 3 ∧ b = real.sqrt 5 ∧ ellipse_foci 3 (real.sqrt 5) = 2) →
  (∀ (x y : ℝ), (y^2 = 2 * p * x) → (x = 2) →
  (directrix x y = -2)) :=
begin
  intros p hp h_ellipse hx y parabola_eq focus_eq,
  have h_c : c = ellipse_foci 3 (real.sqrt 5), from real.sqrt (3^2 - (real.sqrt 5)^2),
  rw h_ellipse,
  rw hp,
  rw parabola_eq,
  sorry,
end

end parabola_directrix_l63_63670


namespace smallest_lcm_four_digit_integers_with_gcd_five_l63_63655

open Nat

theorem smallest_lcm_four_digit_integers_with_gcd_five : ∃ k ℓ : ℕ, 1000 ≤ k ∧ k < 10000 ∧ 1000 ≤ ℓ ∧ ℓ < 10000 ∧ gcd k ℓ = 5 ∧ lcm k ℓ = 203010 :=
by
  use 1005
  use 1010
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_lcm_four_digit_integers_with_gcd_five_l63_63655


namespace arrangement_count_l63_63896

def people : Type := {A, B, C, D, E}

def is_adjacent (x y : people) (arrangement : list people) : Prop :=
  ∃ (i : ℕ), i < arrangement.length - 1 ∧ arrangement.nth_le i sorry = x ∧ arrangement.nth_le (i+1) sorry = y

def is_not_adjacent (x y : people) (arrangement : list people) : Prop :=
  ¬is_adjacent x y arrangement ∧ ¬is_adjacent y x arrangement

def valid_arrangement (arrangement : list people) : Prop :=
  arrangement.length = 5 ∧
  is_adjacent A B arrangement ∧
  is_not_adjacent A C arrangement ∧
  is_not_adjacent B C arrangement

theorem arrangement_count : ∃ (arrangements : finset (list people)), 
  (∀ arrangement ∈ arrangements, valid_arrangement arrangement) ∧
  arrangements.card = 24 := 
sorry

end arrangement_count_l63_63896


namespace construct_triangle_l63_63502

theorem construct_triangle (a r r_b : ℝ) 
  (h1 : a > 0)
  (h2 : r > 0)
  (h3 : r_b > 0) :
  ∃ (A B C : Type) (triangle : A × B × C),
  -- Conditions ensuring that we can construct the triangle with given side and circle radii.
  -- The vertices and the geometric constructions should exist.
  ∃ (AC BC AB : ℝ) 
    (inradius exradius : ℝ),
  AC = a ∧
  inradius = r ∧
  exradius = r_b ∧
  is_triangle A B C AC BC AB ∧
  circle A B C AC BC AB inradius exradius := 
sorry

end construct_triangle_l63_63502


namespace least_number_to_add_l63_63411

theorem least_number_to_add (n : ℕ) : (3457 + n) % 103 = 0 ↔ n = 45 :=
by sorry

end least_number_to_add_l63_63411


namespace brick_length_given_wall_dimensions_and_bricks_count_l63_63401

theorem brick_length_given_wall_dimensions_and_bricks_count (
    (wall_length : ℝ) (wall_height : ℝ) (wall_width : ℝ)
    (bricks_count : ℕ) (brick_height : ℝ) (brick_width : ℝ) (brick_volume : ℝ)
    (total_wall_volume : ℝ)
    (wall_length = 800) (wall_height = 600) (wall_width = 22.5)
    (bricks_count = 1600) (brick_height = 11.25) (brick_width = 6)
    (total_wall_volume = wall_length * wall_height * wall_width) :
    ∃ (brick_length : ℝ), total_wall_volume = bricks_count * (brick_length * brick_height * brick_width) ∧ 
    brick_length = 10800000 / (1600 * 67.5) := sorry

end brick_length_given_wall_dimensions_and_bricks_count_l63_63401


namespace anna_has_winning_strategy_l63_63079

noncomputable def isLosingPosition (piles : List ℕ) : Bool := sorry

theorem anna_has_winning_strategy : 
  ∃ (winning_move : List ℕ → List ℕ), 
    winning_move [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = [piles] ∧ ¬ isLosingPosition piles := 
sorry

end anna_has_winning_strategy_l63_63079


namespace integer_count_in_interval_l63_63628

theorem integer_count_in_interval : 
  let pi := Real.pi in
  let lower_bound := -6 * pi in
  let upper_bound := 12 * pi in
  ∃ (count : ℕ), count = 56 ∧ ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound ↔ (-18 ≤ n ∧ n ≤ 37) :=
by
  let pi := Real.pi
  let lower_bound := -6 * pi
  let upper_bound := 12 * pi
  use 56
  split
  · exact rfl
  · intro n
    split
    · intro h
      split
      · linarith
      · linarith
    · intro h
      split
      · linarith
      · linarith
  sorry

end integer_count_in_interval_l63_63628


namespace find_tax_rate_l63_63504

variable (total_spent : ℝ) (sales_tax : ℝ) (tax_free_cost : ℝ) (taxable_items_cost : ℝ) 
variable (T : ℝ)

theorem find_tax_rate (h1 : total_spent = 25) 
                      (h2 : sales_tax = 0.30)
                      (h3 : tax_free_cost = 21.7)
                      (h4 : taxable_items_cost = total_spent - tax_free_cost - sales_tax)
                      (h5 : sales_tax = (T / 100) * taxable_items_cost) :
  T = 10 := 
sorry

end find_tax_rate_l63_63504


namespace seq_problems_sum_of_first_n_terms_l63_63594

open Nat

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, b 3 ^ 2 = b 1 * b 6

theorem seq_problems  (a b : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a →
  (∀ n : ℕ, b n = a n + n + 4) →
  (is_geometric_sequence b) →
  (b 2 = a 8) →
  (∀ n : ℕ, a n = n + 2 ∧ b n = 2 * n + 6) :=
by
  intros h_arith h_bn h_geom h_b2
  sorry

theorem sum_of_first_n_terms {a b : ℕ → ℕ} :
  (∀ n : ℕ, a n = n + 2 ∧ b n = 2 * n + 6) →
  (∀ n : ℕ, 
    let s := (∑ i in range n, 1 / ((a i) * (b i)))
    S_n = 1 / 2 * (1 / 3 - 1 / (n + 3))) :=
by
  intros h_seq
  sorry

end seq_problems_sum_of_first_n_terms_l63_63594


namespace perpendicular_to_third_side_l63_63054

theorem perpendicular_to_third_side {α : Type*} {P Q R L : α}
    [Nonempty α] [MetricSpace α] [InnerProductSpace ℝ α] 
    (hPQ : L ⟂ line[P, Q]) 
    (hPR : L ⟂ line[P, R]) :
    L ⟂ line[Q, R] := 
  sorry

end perpendicular_to_third_side_l63_63054


namespace limit_of_sequence_limit_frac_seq_l63_63022

def N (ε : ℝ) : ℕ := ⌈((5 / ε) - 1) / 2⌉.toNat

theorem limit_of_sequence (ε : ℝ) (n : ℕ) (hn : n ≥ N ε) 
  (hε_pos : ε > 0) : 
  abs ((4 * n - 3) / (2 * n + 1) - 2) < ε :=
sorry

theorem limit_frac_seq : 
  tendsto (λ n, (4 * n - 3) / (2 * n + 1)) at_top (𝓝 2) :=
begin
  intros ε hε,
  use N ε,
  intros n hn,
  exact limit_of_sequence ε n hn hε,
end

end limit_of_sequence_limit_frac_seq_l63_63022


namespace f_odd_f_increasing_on_2_infty_solve_inequality_f_l63_63202

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) = -f x := by
  sorry

theorem f_increasing_on_2_infty (x₁ x₂ : ℝ) (hx₁ : 2 < x₁) (hx₂ : 2 < x₂) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

theorem solve_inequality_f (x : ℝ) (hx : -5 < x ∧ x < -1) : f (2*x^2 + 5*x + 8) + f (x - 3 - x^2) < 0 := by
  sorry

end f_odd_f_increasing_on_2_infty_solve_inequality_f_l63_63202


namespace tangent_line_at_zero_l63_63265

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x + 1

def tangent_line {m b : ℝ} (line : ℝ → ℝ) (x0 : ℝ) := line x0 = f x0 ∧ derivative line x0 = derivative f x0

theorem tangent_line_at_zero :
  ∃ line : ℝ → ℝ, 
    tangent_line line 0 ∧ (∀ x, line x = 3 * x + 2) :=
begin
  use (λ x, 3 * x + 2),
  split,
  { -- Prove that the line passes through the point of tangency
    split,
    { -- The line passes through (0, f(0))
      simp [f],
      norm_num,
      rw [Real.exp_zero],
      norm_num, },
    { -- The line has the same slope as the derivative of f at 0
      simp [derivative],
      norm_num,
      apply Real.deriv_add,
      { apply Real.deriv_exp, },
      { apply Real.deriv_const_mul,
        norm_num, },
      { apply Real.deriv_const, },
      norm_num,
      rw [Real.exp_zero],
      norm_num, } },
  { -- Prove that the line equation is y = 3x + 2
    intros x,
    refl, },
end

end tangent_line_at_zero_l63_63265


namespace sum_of_squares_of_products_eq_factorial_minus_one_l63_63092

noncomputable def sum_of_squares_of_products : ℕ → ℕ
| 0       := 0
| (n + 1) := (Finset.powerset (Finset.range (n + 1))).filter (λ s, ∀ x ∈ s, ∀ y ∈ s, y ≤ x + 1 → y = x → x = y).sum (λ s, s.prod (λ x, (x + 1)^2))

theorem sum_of_squares_of_products_eq_factorial_minus_one (n : ℕ) :
  sum_of_squares_of_products n = (n + 1)! - 1 := 
sorry

end sum_of_squares_of_products_eq_factorial_minus_one_l63_63092


namespace annual_interest_rate_is_correct_l63_63793

noncomputable def find_annual_interest_rate
  (interest : ℝ) (total_amount : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  let principal := total_amount - interest in
  ((total_amount / principal) ^ (1 / (n * t)) - 1) * 100

theorem annual_interest_rate_is_correct :
  find_annual_interest_rate 2828.80 19828.80 1 2 = 8 :=
  by simp [find_annual_interest_rate]; sorry

end annual_interest_rate_is_correct_l63_63793


namespace order_of_abc_l63_63166

noncomputable def a := (1/2)^Real.pi
noncomputable def b := Real.pi^(1/2)
noncomputable def c := Real.log_base (1/2) Real.pi

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end order_of_abc_l63_63166


namespace cot_neg_45_l63_63132

-- Define the conditions
lemma cot_def (x : ℝ) : Real.cot x = 1 / Real.tan x := sorry
lemma tan_neg (x : ℝ) : Real.tan (-x) = -Real.tan x := sorry
lemma tan_45 : Real.tan (Real.pi / 4) = 1 := sorry

-- State the theorem to prove
theorem cot_neg_45 : Real.cot (-Real.pi / 4) = -1 :=
by
  apply cot_def
  apply tan_neg
  apply tan_45
  sorry

end cot_neg_45_l63_63132


namespace FH_tangent_to_Ω2_l63_63309

open Set

-- Defines the points A, B and P on the circumference of a circle Ω1
variables (Ω1 : Circle) (A B P : Point)
-- Condition: ∠APB is an obtuse angle
variable (is_obtuse_angle_APB : ∠(A, P, B) > pi / 2)
-- Defines Q as the foot of the perpendicular from P to AB
variable (Q : Point) (hQ : FootOfPerpendicular Q P (LineThrough A B))
-- Defines the second circle Ω2 with center P and radius PQ
variables (Ω2 : Circle) (center_Ω2 : Center(Ω2) = P) (radius_Ω2 : Radius(Ω2) = dist P Q)
-- Points where tangents from A and B to Ω2 intersect Ω1 again 
variables (F H : Point) 
  (F_on_Ω1 : OnCircle(F, Ω1)) 
  (H_on_Ω1 : OnCircle(H, Ω1))
  (tan_A_Ω2_F : TangentThrough A F Ω2)
  (tan_B_Ω2_H : TangentThrough B H Ω2)

-- Goal: Prove FH is tangent to Ω2
theorem FH_tangent_to_Ω2 : TangentThrough (LineThrough F H) Ω2 := by
  sorry

end FH_tangent_to_Ω2_l63_63309


namespace possible_values_k_l63_63730

variables (a b c : ℝ^3)
variables (k : ℝ)

-- Assuming the vectors are unit vectors
axiom ha : ∥a∥ = 1
axiom hb : ∥b∥ = 1
axiom hc : ∥c∥ = 1

-- Assuming orthogonal conditions
axiom hab : a • b = 0
axiom hac : a • c = 0

-- Assuming the angle between b and c is π/3
axiom angle_bc : angle b c = real.pi / 3

-- The goal to prove
theorem possible_values_k : a = k • (b × c) ↔ k = (2 * real.sqrt 3) / 3 ∨ k = -(2 * real.sqrt 3) / 3 :=
sorry

end possible_values_k_l63_63730


namespace part_I_part_II_l63_63093

structure EllipsoidData where
  a b c : ℝ
  majorAxis : ℝ := 2*a
  minorAxis : ℝ := 2*b
  focalDistance : ℝ := 2*c
  ecc : ℝ := c/a
  ellipseEqn : (ℝ × ℝ) → Prop := λ (x, y) => x^2 / a^2 + y^2 / b^2 = 1
  abcArea : ℝ := 50*c/9

theorem part_I 
  (ellipse : EllipsoidData)
  (h1 : ellipse.a > ellipse.b ∧ ellipse.b > 0)
  (h2 : 2*ellipse.b = ellipse.a + ellipse.c) :
  ellipse.ecc = 3/5 :=
sorry

theorem part_II 
  (ellipse : EllipsoidData)
  (h3 : ellipse.b = 2)
  (h4 : ellipse.ellipseEqn (0, -2))
  (h5 : ellipse.abcArea = 50*ellipse.c/9) :
  ellipse.a^2 = 5 ∧ ellipse.b^2 = 4 :=
sorry

def EllipseEqn (a b : ℝ) : Prop :=
  ∀ x y, EllipsoidData.ellipseEqn ⟨a, b, sqrt (a^2 - b^2)⟩ (x, y)

example : EllipseEqn 5 4 :=
sorry

end part_I_part_II_l63_63093


namespace problem_integer_solutions_l63_63151

theorem problem_integer_solutions :
  let inequality (x : ℝ) := sqrt (3 * cos (π * x / 2) - cos (π * x / 4) + 1) - sqrt 6 * cos (π * x / 4)
  ∃ (S : Finset ℤ), (∀ x ∈ S, (1991 ≤ x ∧ x ≤ 2013) ∧ inequality x ≥ 0) ∧ S.card = 9 := 
sorry

end problem_integer_solutions_l63_63151


namespace expression_is_integer_if_k_eq_2_l63_63990

def binom (n k : ℕ) := n.factorial / (k.factorial * (n-k).factorial)

theorem expression_is_integer_if_k_eq_2 
  (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : k = 2) : 
  ∃ (m : ℕ), m = (n - 3 * k + 2) * binom n k / (k + 2) := sorry

end expression_is_integer_if_k_eq_2_l63_63990


namespace max_value_fraction_sum_l63_63283

theorem max_value_fraction_sum (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 3) :
  (ab / (a + b + 1) + ac / (a + c + 1) + bc / (b + c + 1) ≤ 3 / 2) :=
sorry

end max_value_fraction_sum_l63_63283


namespace jordan_meets_emily_after_total_time_l63_63958

noncomputable def meet_time
  (initial_distance : ℝ)
  (speed_ratio : ℝ)
  (decrease_rate : ℝ)
  (time_until_break : ℝ)
  (break_duration : ℝ)
  (total_meet_time : ℝ) : Prop :=
  initial_distance = 30 ∧
  speed_ratio = 2 ∧
  decrease_rate = 2 ∧
  time_until_break = 10 ∧
  break_duration = 5 ∧
  total_meet_time = 17

theorem jordan_meets_emily_after_total_time :
  meet_time 30 2 2 10 5 17 := 
by {
  -- The conditions directly state the requirements needed for the proof.
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ -- This line confirms that all inputs match the given conditions.
}

end jordan_meets_emily_after_total_time_l63_63958


namespace divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l63_63291

open Nat

noncomputable def num_divisors (n : ℕ) : ℕ :=
  (factors n).eraseDups.length

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

variables {p q m n : ℕ}
variables (hp : is_prime p) (hq : is_prime q) (hdist : p ≠ q) (hm : 0 ≤ m) (hn : 0 ≤ n)

-- a) Prove the number of divisors of pq is 4
theorem divisors_pq : num_divisors (p * q) = 4 :=
sorry

-- b) Prove the number of divisors of p^2 q is 6
theorem divisors_p2q : num_divisors (p^2 * q) = 6 :=
sorry

-- c) Prove the number of divisors of p^2 q^2 is 9
theorem divisors_p2q2 : num_divisors (p^2 * q^2) = 9 :=
sorry

-- d) Prove the number of divisors of p^m q^n is (m + 1)(n + 1)
theorem divisors_pmqn : num_divisors (p^m * q^n) = (m + 1) * (n + 1) :=
sorry

end divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l63_63291


namespace jimmy_points_l63_63715

theorem jimmy_points (eng_pts init_eng_pts : ℕ) (math_pts init_math_pts : ℕ) 
  (sci_pts init_sci_pts : ℕ) (hist_pts init_hist_pts : ℕ) 
  (phy_pts init_phy_pts : ℕ) (eng_penalty math_penalty sci_penalty hist_penalty phy_penalty : ℕ)
  (passing_points : ℕ) (total_points_required : ℕ):
  init_eng_pts = 60 →
  init_math_pts = 55 →
  init_sci_pts = 40 →
  init_hist_pts = 70 →
  init_phy_pts = 50 →
  eng_penalty = 5 →
  math_penalty = 3 →
  sci_penalty = 8 →
  hist_penalty = 2 →
  phy_penalty = 6 →
  passing_points = 250 →
  total_points_required = (init_eng_pts - eng_penalty) + (init_math_pts - math_penalty) + 
                         (init_sci_pts - sci_penalty) + (init_hist_pts - hist_penalty) + 
                         (init_phy_pts - phy_penalty) →
  ∀ extra_loss, (total_points_required - extra_loss ≥ passing_points) → extra_loss ≤ 1 :=
by {
  sorry
}

end jimmy_points_l63_63715


namespace distance_between_foci_l63_63557

theorem distance_between_foci :
  let x := ℝ
  let y := ℝ
  ∀ (x y : ℝ), 9*x^2 + 36*x + 4*y^2 - 8*y + 1 = 0 →
  ∃ (d : ℝ), d = (Real.sqrt 351) / 3 :=
sorry

end distance_between_foci_l63_63557


namespace a_cannot_ensure_king_path_l63_63431

def king_walk_problem : Prop :=
  ∀ (A B : ℕ), (A % 2 = 0 ∧ B % 2 = 1) → 
  ∃ (f : ℕ × ℕ → bool), 
  (∀ (i j : ℕ), 0 ≤ i ∧ i < 25 ∧ 0 ≤ j ∧ j < 25 → f (i, j) = ff ∨ f (i, j) = tt) ∧
  (f (0, 0) = tt) ∧
  (∀ (i j : ℕ), 0 ≤ i ∧ i < 25 ∧ 0 ≤ j ∧ j < 25 →
    ∃ (n : ℕ), (∀ (k l : ℕ), 0 ≤ k ∧ k < 25 ∧ 0 ≤ l ∧ l < 25 → f (k, l) = tt → 
      (k = i ∧ l = j) ∨ ((1 ≤ i + k ∧ i + k < 3) ∧ (1 ≤ j + l ∧ j + l < 3) → f (k, l) = tt)) ∧
    (n = i) ∧ (∃ B_moves : ℕ → ℕ × ℕ, (∀ (m : ℕ), 0 ≤ m ∧ m < B → f (B_moves m) = ff 
      ∧ ((B_moves 0).fst ≠ 24 ∧ (B_moves 0).snd ≠ 24) 
      ∧ (B_moves 0).fst + (B_moves 0).snd ≤ 1))) 

theorem a_cannot_ensure_king_path : ¬ king_walk_problem :=
sorry

end a_cannot_ensure_king_path_l63_63431


namespace color_roads_l63_63700

structure City :=
  (name : String)
  (republic : String)

structure Road :=
  (city1 city2 : City)
  (republic_relation: city1.republic ≠ city2.republic)

axiom max_roads_originating : ∀ (c : City), ∃ (r : Fin 11), True

theorem color_roads (cities : List City) (roads : List Road) :
  (∀ c : City, c ∈ cities → ∀ r1 r2 : Road, r1 ∈ roads ∧ r2 ∈ roads ∧ r1.city1 = c ∧ r2.city1 = c → r1 != r2) →
  ∃ (coloring : Road → Fin 10),
    ∀ c : City, c ∈ cities → ∀ r1 r2 : Road, r1 ∈ roads ∧ r2 ∈ roads ∧ r1.city1 = c ∧ r2.city1 = c → 
                 coloring r1 ≠ coloring r2 :=
by
  sorry

end color_roads_l63_63700


namespace probability_of_prime_sum_l63_63400

def is_sum_of_numbers_prime_probability : ℚ :=
  let total_outcomes := 216
  let favorable_outcomes := 73
  favorable_outcomes / total_outcomes

theorem probability_of_prime_sum :
  is_sum_of_numbers_prime_probability = 73 / 216 :=
by
  sorry

end probability_of_prime_sum_l63_63400


namespace student_total_marks_l63_63459

variables {M P C : ℕ}

theorem student_total_marks
  (h1 : C = P + 20)
  (h2 : (M + C) / 2 = 35) :
  M + P = 50 :=
sorry

end student_total_marks_l63_63459


namespace number_of_ordered_pairs_l63_63987

-- Formal statement of the problem in Lean 4
theorem number_of_ordered_pairs : 
  ∃ (n : ℕ), n = 128 ∧ 
  ∀ (a b : ℝ), (∃ (x y : ℤ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 65)) ↔ n = 128 :=
sorry

end number_of_ordered_pairs_l63_63987


namespace travel_time_l63_63904

theorem travel_time (distance speed : ℕ) (h_distance : distance = 810) (h_speed : speed = 162) :
  distance / speed = 5 :=
by
  sorry

end travel_time_l63_63904


namespace train_crossing_signal_pole_l63_63430

theorem train_crossing_signal_pole
  (length_train : ℕ)
  (same_length_platform : ℕ)
  (time_crossing_platform : ℕ)
  (h_train_platform : length_train = 420)
  (h_platform : same_length_platform = 420)
  (h_time_platform : time_crossing_platform = 60) : 
  (length_train / (length_train + same_length_platform / time_crossing_platform)) = 30 := 
by 
  sorry

end train_crossing_signal_pole_l63_63430


namespace cot_neg_45_eq_neg_1_l63_63139

-- Hypotheses
variable (θ : ℝ)
variable (h1 : 𝔸.cot θ = 1 / 𝔸.tan θ)
variable (h2 : 𝔸.tan (-45) = -𝔸.tan 45)
variable (h3 : 𝔸.tan 45 = 1)

-- Theorem
theorem cot_neg_45_eq_neg_1 :
  𝔸.cot (-45) = -1 := by
  sorry

end cot_neg_45_eq_neg_1_l63_63139


namespace T_is_non_degenerate_convex_quadrilateral_l63_63914

open Set

def is_lattice_point (p : ℝ × ℝ) : Prop := ∃ (x y : ℤ), p = (x, y)

def E : Set (ℝ × ℝ) := { p | is_lattice_point p }

variables {P Q : Set (ℝ × ℝ)}

def T : Set (ℝ × ℝ) := P ∩ Q

-- Given conditions P and Q are convex and their vertices are on lattice points, and T is not empty and has no lattice points.
variables (hP : IsConvex P) (hQ : IsConvex Q) (hPV : ∀ p ∈ P, is_lattice_point p)
          (hQV : ∀ q ∈ Q, is_lattice_point q) (hTnonempty : T ≠ ∅) (hTnoE : T ∩ E = ∅)

theorem T_is_non_degenerate_convex_quadrilateral : 
  (∃ a b c d : ℝ × ℝ, a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ IsConvex ({a, b, c, d}) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ d) ∧ (d ≠ a)) := 
sorry

end T_is_non_degenerate_convex_quadrilateral_l63_63914


namespace trapezoid_sides_l63_63438

-- Given definitions and assumptions
variables (r : ℝ)
def inscribed_circle_radius (trapezoid : ℝ × ℝ × ℝ) := r

def shorter_base (trapezoid : ℝ × ℝ × ℝ) := trapezoid.1 = 4 * r / 3

-- Statement of the theorem
theorem trapezoid_sides {trapezoid : ℝ × ℝ × ℝ} (h_inscribed : inscribed_circle_radius trapezoid = r)
    (h_base : shorter_base trapezoid) :
  trapezoid.1 = 4 * r ∧ trapezoid.2 = 10 * r / 3 ∧ trapezoid.3 = 2 * r :=
sorry

end trapezoid_sides_l63_63438


namespace stay_nights_l63_63390

theorem stay_nights (cost_per_night : ℕ) (num_people : ℕ) (total_cost : ℕ) (n : ℕ) 
    (h1 : cost_per_night = 40) (h2 : num_people = 3) (h3 : total_cost = 360) (h4 : cost_per_night * num_people * n = total_cost) :
    n = 3 :=
sorry

end stay_nights_l63_63390


namespace euler_totient_sum_geometric_series_l63_63230

theorem euler_totient_sum_geometric_series (p n : ℕ) (hp : Nat.Prime p) (h1 : 1 < n) (hn : n ≤ p):
  ∃ φ : ℕ → ℕ, φ (Nat.sum (λ k, n ^ k) (Finset.range p)) % p = 0 :=
by
  sorry

end euler_totient_sum_geometric_series_l63_63230


namespace train_station_to_tourist_base_l63_63561

theorem train_station_to_tourist_base
  (x v : ℕ)
  (h1 : x - 5 = (x - 5 : ℕ)) -- Ensure compatibility with Lean's type system
  (h2 : v > 3)
  (h3 : (x : ℝ) / (v : ℝ) - (x : ℝ - 5) / 3 = 1) :
  x = 8 ∧ v = 4 :=
sorry

end train_station_to_tourist_base_l63_63561


namespace JameMade112kProfit_l63_63713

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l63_63713


namespace ratio_AN_MB_l63_63069

theorem ratio_AN_MB (A B C M N : Point) (O : Point) (h_angleA : ∠A B C = 60)
  (h_M_on_AB : lies_on M A B) (h_N_on_AC : lies_on N A C)
  (h_O_bisects_MN : midpoint O M N) (h_oc : circumcenter O A B C) : 
  AN / MB = 1 := 
by
  sorry

end ratio_AN_MB_l63_63069


namespace fill_tank_with_leak_l63_63757

theorem fill_tank_with_leak (A L : ℝ) (h1 : A = 1 / 6) (h2 : L = 1 / 18) : (1 / (A - L)) = 9 :=
by
  sorry

end fill_tank_with_leak_l63_63757


namespace problem_l63_63601

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem problem (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) : 
  ((f b - f a) / (b - a) < 1 / (a * (a + 1))) :=
by
  sorry -- Proof steps go here

end problem_l63_63601


namespace cost_unit_pen_max_profit_and_quantity_l63_63458

noncomputable def cost_pen_A : ℝ := 5
noncomputable def cost_pen_B : ℝ := 10
noncomputable def profit_pen_A : ℝ := 2
noncomputable def profit_pen_B : ℝ := 3
noncomputable def spent_on_A : ℝ := 400
noncomputable def spent_on_B : ℝ := 800
noncomputable def total_pens : ℝ := 300

theorem cost_unit_pen : (spent_on_A / cost_pen_A) = (spent_on_B / (cost_pen_A + 5)) := by
  sorry

theorem max_profit_and_quantity
    (xa xb : ℝ)
    (h1 : xa ≥ 4 * xb)
    (h2 : xa + xb = total_pens)
    : ∃ (wa : ℝ), wa = 2 * xa + 3 * xb ∧ xa = 240 ∧ xb = 60 ∧ wa = 660 := by
  sorry

end cost_unit_pen_max_profit_and_quantity_l63_63458


namespace length_AB_slope_one_OA_dot_OB_const_l63_63610

open Real

def parabola (x y : ℝ) : Prop := y * y = 4 * x
def line_through_focus (x y : ℝ) (k : ℝ) : Prop := x = k * y + 1
def line_slope_one (x y : ℝ) : Prop := y = x - 1

theorem length_AB_slope_one {x1 x2 y1 y2 : ℝ} (hA : parabola x1 y1) (hB : parabola x2 y2) 
  (hL : line_slope_one x1 y1) (hL' : line_slope_one x2 y2) : abs (x1 - x2) + abs (y1 - y2) = 8 := 
by
  sorry

theorem OA_dot_OB_const {x1 x2 y1 y2 : ℝ} {k : ℝ} (hA : parabola x1 y1)
  (hB : parabola x2 y2) (hL : line_through_focus x1 y1 k) (hL' : line_through_focus x2 y2 k) :
  x1 * x2 + y1 * y2 = -3 :=
by
  sorry

end length_AB_slope_one_OA_dot_OB_const_l63_63610


namespace kittens_given_away_l63_63929

-- Conditions
def initial_kittens : ℕ := 8
def remaining_kittens : ℕ := 4

-- Statement to prove
theorem kittens_given_away : initial_kittens - remaining_kittens = 4 :=
by
  sorry

end kittens_given_away_l63_63929


namespace remaining_pieces_l63_63679

theorem remaining_pieces (initial_pieces : ℕ) (arianna_lost : ℕ) (samantha_lost : ℕ) (diego_lost : ℕ) (lucas_lost : ℕ) :
  initial_pieces = 128 → arianna_lost = 3 → samantha_lost = 9 → diego_lost = 5 → lucas_lost = 7 →
  initial_pieces - (arianna_lost + samantha_lost + diego_lost + lucas_lost) = 104 := by
  sorry

end remaining_pieces_l63_63679


namespace ratio_of_shapes_l63_63674

theorem ratio_of_shapes (a b r : ℝ) (h : a = a) : 
  (a^3 : b^3 : (π * r^2 * a) = 81 : 25 : 40) →
  (a : b = 3 : 5) ∧ (a : r = 9 * √π : √40) :=
  sorry

end ratio_of_shapes_l63_63674


namespace nth_odd_positive_integer_is_199_l63_63856

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l63_63856


namespace pentagon_PTRSQ_area_proof_l63_63316

-- Define the geometric setup and properties
def quadrilateral_PQRS_is_square (P Q R S T : Type) : Prop :=
  -- Here, we will skip the precise geometric construction and assume the properties directly.
  sorry

def segment_PT_perpendicular_to_TR (P T R : Type) : Prop :=
  sorry

def PT_eq_5 (PT : ℝ) : Prop :=
  PT = 5

def TR_eq_12 (TR : ℝ) : Prop :=
  TR = 12

def area_PTRSQ (area : ℝ) : Prop :=
  area = 139

theorem pentagon_PTRSQ_area_proof
  (P Q R S T : Type)
  (PQRS_is_square : quadrilateral_PQRS_is_square P Q R S T)
  (PT_perpendicular_TR : segment_PT_perpendicular_to_TR P T R)
  (PT_length : PT_eq_5 5)
  (TR_length : TR_eq_12 12)
  : area_PTRSQ 139 :=
  sorry

end pentagon_PTRSQ_area_proof_l63_63316


namespace minimumBlackEdgesIs3_l63_63516

-- | Define a cube with colored edges
structure Cube :=
  (edges : Fin 12 → Bool) -- 12 edges, Bool represents black (true) or red (false)

-- | Define the condition that every face of the cube has at least one black edge
def faceHasBlackEdge (c : Cube) (f : Fin 6) : Prop :=
  ∃ e, (faceEdges f).contains e ∧ c.edges e = true

-- | Define the verify function
def validCube (c : Cube) : Prop :=
  ∀ f, faceHasBlackEdge c f

-- | Define the minimal requirement number of black edges
def minimalBlackEdges (n : Nat) (c : Cube) : Prop :=
  validCube c ∧ (c.edges.to_lex_strings.countp (λ i, i) ≥ n)

-- | Prove the smallest number possible of black edges is 3
theorem minimumBlackEdgesIs3 : ∃ c : Cube, minimalBlackEdges 3 c :=
by
  sorry

end minimumBlackEdgesIs3_l63_63516


namespace quadratic_equation_real_roots_l63_63609

noncomputable def quadratic_discriminant_roots (k : ℝ) : Prop :=
  let a := 1
  let b := -(2 * k + 2)
  let c := 2 * k + 1
  let Δ := b^2 - 4 * a * c
  Δ = 4 * k^2 ∧ Δ ≥ 0 ∧ ∀ x1 x2 : ℝ,
    (x1, x2) = ((2 * k + 2 - sqrt (4 * k^2)) / 2, (2 * k + 2 + sqrt (4 * k^2)) / 2) →
    (x1 = 1 ∨ x2 = 1 ∧ x2 > 3 → k > 1)

theorem quadratic_equation_real_roots (k : ℝ) : quadratic_discriminant_roots k :=
by
  sorry

end quadratic_equation_real_roots_l63_63609


namespace batsman_average_increase_l63_63901

theorem batsman_average_increase
  (prev_avg : ℝ) -- average before the 17th innings
  (total_runs_16 : ℝ := 16 * prev_avg) -- total runs scored in the first 16 innings
  (score_17th : ℝ := 85) -- score in the 17th innings
  (new_avg : ℝ := 37) -- new average after 17 innings
  (total_runs_17 : ℝ := total_runs_16 + score_17th) -- total runs after 17 innings
  (calc_total_runs_17 : ℝ := 17 * new_avg) -- new total runs calculated by the new average
  (h : total_runs_17 = calc_total_runs_17) -- given condition: total_runs_17 = calc_total_runs_17
  : (new_avg - prev_avg) = 3 := 
by
  sorry

end batsman_average_increase_l63_63901


namespace triangle_area_midpoints_l63_63310

-- Define the conditions
theorem triangle_area_midpoints (s : ℝ) (area_square : ℝ) (A B C : ℝ) 
  (h1 : s^2 = 128) 
  (h2 : A = s / 2) 
  (h3 : B = s / 2) 
  (h4 : C = s / 2) :
  let area_triangle := 1/2 * A * B in
  area_triangle = 16 := 
by
  sorry

end triangle_area_midpoints_l63_63310


namespace four_points_collinear_or_cyclic_l63_63559

theorem four_points_collinear_or_cyclic
  (A1 A2 B1 B2 : Type*) [metric_space A1] [metric_space A2] [metric_space B1] [metric_space B2]
  (h_distinct : ∀ (X Y : Type*) [metric_space X] [metric_space Y], X ≠ Y)
  (h_circles_intersect : ∀ (C1 : set (Type*)) (C2 : set (Type*)), 
    (∀ (P1 P2 : A1), P1 ∈ C1 ∧ P2 ∈ C1 → ∃ P3 ∈ C2, P1 = P3 ∨ P2 = P3) ↔ 
    (∀ (Q1 Q2 : B1), Q1 ∈ C2 ∧ Q2 ∈ C2 → ∃ Q3 ∈ C1, Q1 = Q3 ∨ Q2 = Q3)) :
  (collinear A1 A2 B1 B2 ∨ (circle (A1 : B1) ∧ circle (A2 : B2) :=
sorry

end four_points_collinear_or_cyclic_l63_63559


namespace product_of_y_coordinates_l63_63762

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.fst - p1.fst)^2 + (p2.snd - p1.snd)^2)

theorem product_of_y_coordinates :
  ∀ y : ℝ, distance (-5, y) (4, 5) = 12 → (y = 5 + real.sqrt 63 ∨ y = 5 - real.sqrt 63) →
    (5 + real.sqrt 63) * (5 - real.sqrt 63) = -38 :=
by
  intros y hy h_solution
  sorry

end product_of_y_coordinates_l63_63762


namespace polynomial_coefficients_sum_l63_63996

theorem polynomial_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 160 :=
by
  sorry

end polynomial_coefficients_sum_l63_63996


namespace hundredth_odd_positive_integer_l63_63865

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l63_63865


namespace quadratic_one_positive_root_l63_63558

theorem quadratic_one_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y ∈ {t | t^2 - a * t + a - 2 = 0} → y = x)) → a ≤ 2 :=
by
  sorry

end quadratic_one_positive_root_l63_63558


namespace unique_solution_l63_63968

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end unique_solution_l63_63968


namespace k_value_correct_l63_63675

noncomputable def polynomial_is_divisible (k : ℝ) : Prop :=
  (Polynomial.eval 1 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C k * Polynomial.X + Polynomial.C (-3)) = 0)

theorem k_value_correct (k : ℝ) (h : polynomial_is_divisible k) : k = 2 :=
begin
  sorry
end

end k_value_correct_l63_63675


namespace cody_tickets_l63_63940

theorem cody_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (won_tickets : ℕ) : 
  initial_tickets = 49 ∧ spent_tickets = 25 ∧ won_tickets = 6 → 
  initial_tickets - spent_tickets + won_tickets = 30 :=
by sorry

end cody_tickets_l63_63940


namespace triangle_with_angle_ratio_is_right_l63_63240

theorem triangle_with_angle_ratio_is_right (
  (k : ℝ) 
  (h_ratio : (k + 2 * k + 3 * k = 180)) 
) : (30 ≤ k ∧ k ≤ 30) ∧ (2 * k = 2 * 30) ∧ (3 * k = 3 * 30) 
  ∧ (k = 30) ∧ (90 = 3 * k) :=
by {
  sorry
}

end triangle_with_angle_ratio_is_right_l63_63240


namespace count_distinct_four_digit_numbers_ending_in_25_l63_63223

-- Define what it means for a number to be a four-digit number according to the conditions in (a).
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define what it means for a number to be divisible by 5 and end in 25 according to the conditions in (a).
def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the problem as a theorem in Lean
theorem count_distinct_four_digit_numbers_ending_in_25 : 
  ∃ (count : ℕ), count = 90 ∧ 
    (∀ n, is_four_digit_number n ∧ ends_in_25 n → n % 5 = 0) :=
by
  sorry

end count_distinct_four_digit_numbers_ending_in_25_l63_63223


namespace panda_babies_l63_63926

theorem panda_babies (total_pandas : ℕ) (pregnancy_rate : ℚ) (pandas_coupled : total_pandas % 2 = 0) (couple_has_one_baby : ℕ → ℕ) :
  total_pandas = 16 → pregnancy_rate = 0.25 → couple_has_one_baby ((total_pandas / 2) * pregnancy_rate).natAbs = 2 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end panda_babies_l63_63926


namespace count_integers_in_interval_l63_63639

theorem count_integers_in_interval :
  {n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi}.toFinset.card = 57 :=
by
  sorry

end count_integers_in_interval_l63_63639


namespace larger_triangle_arithmetic_l63_63406

def right_angle_triangle (α β : ℝ) : Prop :=
  α + β = 90

def arithmetic_mean (x y z : ℝ) : Prop :=
  ∃ (a : ℝ), a = (x + y) / 2 ∧ a = z

/-- Given two identical right-angled triangles, placed together with identical legs adjacent, 
     and forming a larger triangle where one angle is the arithmetic mean of the other two angles:
     - (1) Can the length of one side of the larger triangle be the arithmetic mean of the lengths of the other two sides.
     - (2) Can the length of any side of the larger triangle NOT be equal to the arithmetic mean of the lengths of the other two sides.
-/
theorem larger_triangle_arithmetic (α β x y z : ℝ) 
  (h1 : right_angle_triangle α β)
  (h2 : arithmetic_mean α β x)
  (h3 : arithmetic_mean β x y)
  (h4 : arithmetic_mean x y z) :
  ∃ (a b c : ℝ), 
  (arithmetic_mean a b c) = true ∧ 
  (∀ d e, (arithmetic_mean d e (d + e) / 2) = false) := 
sorry

end larger_triangle_arithmetic_l63_63406


namespace domain_of_f_l63_63360

noncomputable def f (x : ℝ) : ℝ := (sqrt (2 - x)) / (Real.log x)

theorem domain_of_f :
  (∀ x, 0 < x → x ≤ 2 → x ≠ 1 → (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)) :=
by
  intro x hx1 hx2 hx3
  have hx2' : x ∈ set.Icc 0 2 := sorry
  have hx3' : x ∉ set.Ici 1 := sorry
  exact sorry

end domain_of_f_l63_63360


namespace BD_tangent_circumcircle_TSH_l63_63692

noncomputable def quadrilateral (A B C D : Point) : Prop :=
  convex_quadrilateral A B C D ∧
  angle A B C = pi / 2 ∧
  angle C D A = pi / 2

theorem BD_tangent_circumcircle_TSH
    {A B C D H S T: Point}
    (h_quadrilateral : quadrilateral A B C D)
    (h_H_foot : is_foot H A B D)
    (h_H_in_triangle : inside_triangle H S C T)
    (h_angle_CHS_CSB : angle C H S - angle C S B = pi / 2)
    (h_angle_THC_DTC : angle T H C - angle D T C = pi / 2) :
    is_tangent (line_through B D) (circumcircle T S H) :=
sorry

end BD_tangent_circumcircle_TSH_l63_63692


namespace simplify_factorial_expression_l63_63784

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem simplify_factorial_expression :
  (factorial 13) / (factorial 11 + 3 * factorial 10) = 122 := by
  sorry

end simplify_factorial_expression_l63_63784


namespace fraction_equality_l63_63663

variable (a b : ℚ)

theorem fraction_equality (h : (4 * a + 3 * b) / (4 * a - 3 * b) = 4) : a / b = 5 / 4 := by
  sorry

end fraction_equality_l63_63663


namespace projection_of_a_onto_b_is_correct_l63_63220

open Real

noncomputable def a : ℝ × ℝ := (2, 3)
noncomputable def b : ℝ × ℝ := (-4, 7)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

def projection (v w : ℝ × ℝ) : ℝ :=
  (dot_product v w) / (magnitude w)

theorem projection_of_a_onto_b_is_correct :
  projection a b = (sqrt 65) / 5 :=
  sorry

end projection_of_a_onto_b_is_correct_l63_63220


namespace intersection_sets_l63_63184

def setA : Set ℝ := {x | -2 < x ∧ x < 2}
def setB : Set ℝ := {x | x > sqrt 3}
def setC : Set ℝ := {x | sqrt(3) < x ∧ x < 2}

theorem intersection_sets : setA ∩ setB = setC := by
  sorry

end intersection_sets_l63_63184


namespace find_d_l63_63728

-- Define the arithmetic sequence {C_n} with the first term 2 and common difference d
def C (n : ℕ) (d : ℝ) : ℝ := 2 + (n - 1) * d

-- Define the function T_n to be the sum of the first n terms of the sequence C_n
def T (n : ℕ) (d : ℝ) : ℝ := n * (2 + (n - 1) * d / 2)

-- Condition: {C_n} is a "sum-geometric sequence"
def is_sum_geometric_sequence (d : ℝ) := 
  ∀ (n : ℕ) (k : ℝ), (n > 0) → T (2 * n) d / T n d = k

-- Theorem: d = 4
theorem find_d (d : ℝ) (hd : d ≠ 0) : is_sum_geometric_sequence d → d = 4 :=
by
  intros h
  sorry

end find_d_l63_63728


namespace find_integer_mod_l63_63974

theorem find_integer_mod (n : ℤ) (h₀ : 10 ≤ n) (h₁ : n ≤ 15) (h₂ : n ≡ 12345 [MOD 7]) : n = 11 := 
by
  sorry

end find_integer_mod_l63_63974


namespace price_returns_to_initial_l63_63385

theorem price_returns_to_initial (x : ℝ) (h : 0.918 * (100 + x) = 100) : x = 9 := 
by
  sorry

end price_returns_to_initial_l63_63385


namespace b_plus_d_l63_63279

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem b_plus_d 
  (a b c d : ℝ) 
  (h1 : f a b c d 1 = 20) 
  (h2 : f a b c d (-1) = 16) 
: b + d = 18 :=
sorry

end b_plus_d_l63_63279


namespace pen_rubber_length_difference_l63_63454

theorem pen_rubber_length_difference (P R : ℕ) 
    (h1 : P = R + 3)
    (h2 : P = 12 - 2) 
    (h3 : R + P + 12 = 29) : 
    P - R = 3 :=
  sorry

end pen_rubber_length_difference_l63_63454


namespace cot_neg_45_l63_63127

-- Define the given conditions
def tan_neg_angle (x : ℝ) : Prop := ∀ θ : ℝ, tan (-θ) = -tan(θ)
def tan_45 : Prop := tan (45 * (π / 180)) = 1
def cot_def (x : ℝ) : Prop := ∀ θ : ℝ, cot(θ) = 1 / tan(θ)

-- Prove that cot(-45°) = -1 given the conditions
theorem cot_neg_45 : cot (-45 * (π / 180)) = -1 :=
by 
  have h1 := tan_neg_angle (-45 * (π / 180)),
  have h2 := tan_45,
  have h3 := cot_def (-45 * (π / 180)),
  sorry -- Proof steps skipped

end cot_neg_45_l63_63127


namespace sum_s_r_l63_63736

-- Define r(x) has a domain {-1, 0, 1, 2, 3} and range {0, 1, 2, 3, 4}
def r_domain := {-1, 0, 1, 2, 3}
def r_range := {0, 1, 2, 3, 4}
noncomputable def r : r_domain → r_range := sorry

-- Define s(x) over the domain {1, 2, 3, 4, 5}
def s_domain := {1, 2, 3, 4, 5}
def s (x : s_domain) : ℝ := x + 2

-- The proof statement
theorem sum_s_r : ∑ y in {1, 2, 3, 4}, s y = 18 :=
by
  -- all possible values of s(r(x)) are {s(1), s(2), s(3), s(4)}
  have values := {s(1), s(2), s(3), s(4)}
  -- calculate s(1), s(2), s(3), s(4)
  have h1 : s(1) = 3 := rfl
  have h2 : s(2) = 4 := rfl
  have h3 : s(3) = 5 := rfl
  have h4 : s(4) = 6 := rfl
  -- sum them up
  show ∑ y in {1, 2, 3, 4}, s y = 18, from
    calc
      ∑ y in {1, 2, 3, 4}, s y = 3 + 4 + 5 + 6 : by rw [h1, h2, h3, h4]
                            ... = 18 : by norm_num

end sum_s_r_l63_63736


namespace brigade_harvest_time_l63_63808

theorem brigade_harvest_time (t : ℕ) :
  (t - 5 = (3 * t / 5) + ((t * (t - 8)) / (5 * (t - 4)))) → t = 20 := sorry

end brigade_harvest_time_l63_63808


namespace glycerin_percentage_proof_l63_63920

-- Conditions given in problem
def original_percentage : ℝ := 0.90
def original_volume : ℝ := 4
def added_volume : ℝ := 0.8

-- Total glycerin in original solution
def glycerin_amount : ℝ := original_percentage * original_volume

-- Total volume after adding water
def new_volume : ℝ := original_volume + added_volume

-- Desired percentage proof statement
theorem glycerin_percentage_proof : 
  (glycerin_amount / new_volume) * 100 = 75 := 
by
  sorry

end glycerin_percentage_proof_l63_63920


namespace part1_part2_l63_63578

variable {x m : ℝ}

section
  def A : Set ℝ := {x | x^2 - 2x - 3 < 0}
  def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Part (Ⅰ)
theorem part1 (hm : m = 0) : ∀ x, x ∈ (A ∩ B m) ↔ x ∈ Set.Ico 1 3 :=
by
  intro x
  rw [hm]
  sorry

-- Part (Ⅱ)
theorem part2 : (∀ x, (x^2 - 2x - 3 < 0) → ((x - m + 1) * (x - m - 1) ≥ 0)) ↔ (m ≥ 4 ∨ m ≤ -2) :=
by
  sorry
end

end part1_part2_l63_63578


namespace problem_solution_l63_63270

-- Define the sequence {a_n}
def seq : ℕ → ℤ
| 0     := 1
| 1     := 3
| (n+2) := seq (n+1) - seq n

-- Define the sum S_n
def S (n : ℕ) : ℤ :=
  (Finset.range (n+1)).sum seq

-- State the theorem for the assertions 'a_100 = -1' and 'S_100 = 5'
theorem problem_solution :
  seq 99 = -1 ∧ S 99 = 5 :=
by
  sorry

end problem_solution_l63_63270


namespace difference_in_pay_l63_63753

theorem difference_in_pay (pay_per_delivery : ℕ) (oula_deliveries : ℕ) (fraction_tona_oula : ℚ)
  (oula_pay : ℕ) (tona_pay : ℕ) (difference_pay : ℕ) :
  pay_per_delivery = 100 →
  oula_deliveries = 96 →
  fraction_tona_oula = 3/4 →
  oula_pay = oula_deliveries * pay_per_delivery →
  tona_pay = (oula_deliveries * fraction_tona_oula).natAbs * pay_per_delivery →
  difference_pay = oula_pay - tona_pay →
  difference_pay = 2400 :=
by
  intros
  sorry

end difference_in_pay_l63_63753


namespace remaining_notes_denomination_l63_63055

theorem remaining_notes_denomination
  (total_money : ℕ)
  (total_notes : ℕ)
  (fifty_notes : ℕ)
  (fifty_notes_value : ℕ)
  (denomination_value : ℕ) :
  total_money = 10350 → total_notes = 72 → fifty_notes = 57 → fifty_notes_value = 50 →
  let remaining_notes := total_notes - fifty_notes in
  let remaining_money := total_money - (fifty_notes * fifty_notes_value) in
  denomination_value = remaining_money / remaining_notes →
  denomination_value = 500 :=
by
  intros h1 h2 h3 h4 h5
  simp at h5
  sorry

end remaining_notes_denomination_l63_63055


namespace max_intersections_l63_63405

def is_convex (P : Set ℝ) : Prop := 
  ∀ {x y ∈ P} {t: ℝ}, 0 ≤ t → t ≤ 1 → t • x + (1 - t) • y ∈ P

noncomputable def Q1 : Set ℝ := sorry -- Define the set representing polygon Q1
noncomputable def Q2 : Set ℝ := sorry -- Define the set representing polygon Q2

def m1 : ℕ := 5
def m2 : ℕ := 7

axiom Q1_convex : is_convex Q1
axiom Q2_convex : is_convex Q2
axiom Q1_inside_Q2 : Q1 ⊆ Q2

theorem max_intersections : 
  let intersections := ∑ i in finset.range m1, ∑ j in finset.range m2, 1 in
  intersections = 35 :=
by simp [finset.sum_const, m1, m2]; norm_num

end max_intersections_l63_63405


namespace isosceles_triangle_vertex_angle_l63_63005

theorem isosceles_triangle_vertex_angle (a b : ℝ) (α : ℝ) :
  (2 * Real.arcsin (Real.sqrt 2 - 1) < α ∧ α < Real.pi) ↔
  (isosceles_triangle a b α ∧ exactly_three_bisecting_lines a b α) := 
sorry

end isosceles_triangle_vertex_angle_l63_63005


namespace problem1_simplified_problem2_simplified_l63_63030

-- Definition and statement for the first problem
def problem1_expression (x y : ℝ) : ℝ := 
  -3 * x * y - 3 * x^2 + 4 * x * y + 2 * x^2

theorem problem1_simplified (x y : ℝ) : 
  problem1_expression x y = x * y - x^2 := 
by
  sorry

-- Definition and statement for the second problem
def problem2_expression (a b : ℝ) : ℝ := 
  3 * (a^2 - 2 * a * b) - 5 * (a^2 + 4 * a * b)

theorem problem2_simplified (a b : ℝ) : 
  problem2_expression a b = -2 * a^2 - 26 * a * b :=
by
  sorry

end problem1_simplified_problem2_simplified_l63_63030


namespace new_volume_is_108_l63_63825

-- Translate given conditions into Lean definitions
variables {r h : ℝ} -- original radius and height
def original_volume : ℝ := 24
def original_volume_formula (r h : ℝ) : ℝ := Real.pi * r^2 * h
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2
def new_volume_formula (r h : ℝ) : ℝ := 9 / 2 * Real.pi * r^2 * h

-- The theorem to be proved
theorem new_volume_is_108 :
  original_volume_formula r h = original_volume →
  new_volume_formula r h = 108 :=
by
  sorry

end new_volume_is_108_l63_63825


namespace hundredth_odd_integer_l63_63864

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l63_63864


namespace max_xy_on_line_AB_l63_63577

noncomputable def pointA : ℝ × ℝ := (3, 0)
noncomputable def pointB : ℝ × ℝ := (0, 4)

-- Define the line passing through points A and B
def on_line_AB (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P.1 = 3 - 3 * t ∧ P.2 = 4 * t

theorem max_xy_on_line_AB : ∃ (P : ℝ × ℝ), on_line_AB P ∧ P.1 * P.2 = 3 := 
sorry

end max_xy_on_line_AB_l63_63577


namespace min_value_l63_63182

theorem min_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 1) :
  (a + 1)^2 + 4 * b^2 + 9 * c^2 ≥ 144 / 49 :=
sorry

end min_value_l63_63182


namespace angle_sum_zero_l63_63724

theorem angle_sum_zero {n : ℕ} (h : n > 1) (points : Fin 2n → Fin 2n → Fin 2n → ℝ) :
  (∀ A B C D : Fin 2n, A ≠ B ∧ B ≠ C ∧ C ≠ D → A ≠ C ∧ B ≠ D) →
  (∀ A B C : Fin 2n, 0 < points A B C ∧ points A B C < 180) →
  ∃ (perm : Fin 2n → Fin 2n), 
    ∑ i in Finset.range (2n), points (perm i) (perm (i + 1)) (perm (i + 2)) = 0 :=
by
  sorry

end angle_sum_zero_l63_63724


namespace animals_meet_in_nine_days_l63_63915

def horse_distance (n : ℕ) : ℕ :=
  103 * n + (n * (n - 1) * 13) / 2

def mule_distance (n : ℕ) : ℕ :=
  97 * n + (n * (n - 1) * (-0.5)) / 2

theorem animals_meet_in_nine_days : ∃ (m : ℕ), 
  103 * m + (m * (m - 1) * 13) / 2 + 97 * m + (m * (m - 1) * (-0.5)) / 2 = 2250 ∧ 
  m = 9 :=
sorry

end animals_meet_in_nine_days_l63_63915


namespace total_cost_sean_bought_l63_63771

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end total_cost_sean_bought_l63_63771


namespace shane_gum_problem_l63_63518

-- Definitions according to the conditions provided
def initially_gum := 100
def rick_gum := initially_gum / 2
def shane_gum_from_rick := rick_gum / 3
def additional_gum_from_cousin := 10
def shane_initial_gum := (shane_gum_from_rick : ℤ) + additional_gum_from_cousin
def chewed_gum := 11
def shane_gum_after_chewing := shane_initial_gum - chewed_gum
def shared_with_sister := shane_gum_after_chewing / 2
def shane_final_gum := (shane_gum_after_chewing - shared_with_sister) : ℤ

-- Proof problem statement in Lean 4
theorem shane_gum_problem : shane_final_gum = 8 := by
  unfold initially_gum rick_gum shane_gum_from_rick additional_gum_from_cousin shane_initial_gum chewed_gum shane_gum_after_chewing shared_with_sister shane_final_gum
  norm_num
  sorry

end shane_gum_problem_l63_63518


namespace count_four_digit_numbers_ending_25_l63_63225

theorem count_four_digit_numbers_ending_25 : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 100]) → ∃ n : ℕ, n = 100 :=
by
  sorry

end count_four_digit_numbers_ending_25_l63_63225


namespace domain_range_g_l63_63052

variable (f : ℝ → ℝ)
variable domain_f : Set.Icc (1 : ℝ) 3
variable range_f : Set.Icc (-1 : ℝ) 2

def g (x : ℝ) := 2 - f (x + 2)

theorem domain_range_g :
  (∀ x, Set.Icc (-1 : ℝ) 1 (x) ↔ x ∈ domain_f) →
  (∀ y, Set.Icc (0 : ℝ) 3 (y) ↔ y ∈ range_f) →
  (Set.Icc (-1 : ℝ) 1, Set.Icc (0 : ℝ) 3) = (-1, 1, 0, 3) :=
by
  sorry

end domain_range_g_l63_63052


namespace measure_of_angle_EGJ_l63_63335

noncomputable def degree_measure_EGJ (EF G H : Point) (semi_circle : Circle EF) (full_circle : Circle (distance G H)) (GJ : LineSegment G Point) := sorry

theorem measure_of_angle_EGJ (EF : Segment) (G : Point) (H: Point) (semi_circle : Circle EF) (full_circle : Circle (distance G H)) (GJ : LineSegment G Point)
  (h1 : midpoint EF G)
  (h2 : midpoint FG H)
  (h3 : circle EF = semi_circle)
  (h4 : circle (distance G H) = full_circle)
  (h5 : splits_equally_regions GJ semi_circle full_circle) :
  measure_of_angle EGJ = 180 :=
sorry

end measure_of_angle_EGJ_l63_63335


namespace fractional_shaded_area_l63_63922

noncomputable def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)

theorem fractional_shaded_area :
  let a := (7 : ℚ) / 16
  let r := (1 : ℚ) / 16
  geometric_series_sum a r = 7 / 15 :=
by
  sorry

end fractional_shaded_area_l63_63922


namespace num_panda_babies_l63_63927

-- Definitions based on conditions
def pandas := 16
def couples := pandas / 2
def pregnancy_rate := 0.25
def pregnant_couples := pregnancy_rate * couples
def babies_per_couple := 1

-- Theorem stating the problem
theorem num_panda_babies : (pregnant_couples * babies_per_couple) = 2 := by
  sorry

end num_panda_babies_l63_63927


namespace coconuts_total_l63_63083

theorem coconuts_total (B_trips : Nat) (Ba_coconuts_per_trip : Nat) (Br_coconuts_per_trip : Nat) (combined_trips : Nat) (B_totals : B_trips = 12) (Ba_coconuts : Ba_coconuts_per_trip = 4) (Br_coconuts : Br_coconuts_per_trip = 8) : combined_trips * (Ba_coconuts_per_trip + Br_coconuts_per_trip) = 144 := 
by
  simp [B_totals, Ba_coconuts, Br_coconuts]
  sorry

end coconuts_total_l63_63083


namespace trig_expression_eval_l63_63998

theorem trig_expression_eval (a α m : ℝ) (h : a * (5 * Real.pi + α) = m) : 
  (sin (α - 3 * Real.pi) + cos 0) / (sin α - cos (Real.pi + α)) = (m + 1) / (m - 1) :=
  sorry

end trig_expression_eval_l63_63998


namespace min_policemen_required_l63_63439

theorem min_policemen_required (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  ∃ p, p = (m - 1) * (n - 1) :=
by
  let p := (m - 1) * (n - 1)
  use p
  sorry

end min_policemen_required_l63_63439


namespace time_for_first_and_last_collision_l63_63980

theorem time_for_first_and_last_collision (v1 v2 v3 v4 v5 l : ℝ)
 (hv1 : v1 = 0.5) (hv2 : v2 = 0.5) (hv3 : v3 = 0.1) (hv4 : v4 = 0.1) (hv5 : v5 = 0.1) (hl : l = 2) 
 (elastic : ∀ u v : ℝ, u + v = v + u) :
  3 * l / (v1 + v5) = 10 :=
by
  -- Assuming elastic collisions and initial conditions
  have hrel_vel : v1 + v5 = 0.6, from calc
    v1 + v5 = 0.5 + 0.1 : by rw [hv1, hv5]
         ... = 0.6       : by norm_num,
  have hdist : 3 * l = 6, from calc
    3 * l = 3 * 2 : by rw [hl]
        ... = 6   : by norm_num,
  show 3 * l / (v1 + v5) = 10, from calc
    3 * l / (v1 + v5) = 6 / 0.6 : by rw [hdist, hrel_vel]
                     ... = 10   : by norm_num

#check time_for_first_and_last_collision

end time_for_first_and_last_collision_l63_63980


namespace rose_needs_more_money_l63_63326

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end rose_needs_more_money_l63_63326


namespace conjugate_of_z_l63_63193

-- Define the complex number z.
def z : ℂ := (2 + complex.I) / (1 - complex.I)

-- State the theorem to prove the conjugate of z.
theorem conjugate_of_z : complex.conj z = (1/2 : ℂ) - (3/2 : ℂ) * complex.I :=
by
  sorry

end conjugate_of_z_l63_63193


namespace count_perfect_cubes_l63_63489

theorem count_perfect_cubes : ∀ (x y : ℕ), 
  x = 2^7 + 1 →
  y = 2^12 + 1 →
  (∃ (count : ℕ), count = 11 ∧ (∀ (n : ℕ), n^3 ≥ x → n^3 ≤ y → n ≥ 6 ∧ n ≤ 16)) :=
by {
  intros x y hx hy,
  rw [hx, hy],
  use 11,
  split,
  {
    exact rfl,
  },
  {
    intros n hnx hny,
    split,
    {
      linarith,
    },
    {
      linarith,
    }
  },
  sorry
}

end count_perfect_cubes_l63_63489


namespace cot_neg_45_eq_neg_1_l63_63137

-- Hypotheses
variable (θ : ℝ)
variable (h1 : 𝔸.cot θ = 1 / 𝔸.tan θ)
variable (h2 : 𝔸.tan (-45) = -𝔸.tan 45)
variable (h3 : 𝔸.tan 45 = 1)

-- Theorem
theorem cot_neg_45_eq_neg_1 :
  𝔸.cot (-45) = -1 := by
  sorry

end cot_neg_45_eq_neg_1_l63_63137


namespace line_equation_triangle_area_l63_63181

-- Given points M, N, and P
def M := (0 : ℝ, 3 : ℝ)
def N := (-4 : ℝ, 0 : ℝ)
def P := (-2 : ℝ, 4 : ℝ)

-- Equation of the line passing through point P and parallel to MN
theorem line_equation (M N P: ℝ × ℝ) : 3 * P.1 - 4 * P.2 + 22 = 0 :=
by sorry

-- Area of triangle MNP
theorem triangle_area (M N P: ℝ × ℝ) : (1/2) * (5 : ℝ) * 2 = 5 :=
by sorry

end line_equation_triangle_area_l63_63181


namespace inequality_l63_63424

-- Definitions
variables (A B C T H B1 : Type)
variables (d_AT d_BT d_CT d_AB d_BC d_CA d_TB1 d_TH : ℝ)

-- Conditions
axiom midpoint_B1_BT : d_TB1 = d_BT / 2 -- Midpoint condition
axiom TH_eq_TB1 : d_TH = d_TB1 -- Equality of lengths TH and TB1
axiom angle_THB1 : ∠ T H B1 = 60 -- Given angle measure
axiom angle_TB1H : ∠ T B1 H = 60 -- Given angle measure
axiom angle_ATB1 : ∠ A T B1 = 120 -- Given angle measure
axiom d_HB1_EQ_d_TB1 : d_H B1 = d_TB1 -- Equality of lengths HB1 and TB1
axiom d_TB1_EQ_d_B1B : d_TB1 = d_B1 B -- Equality of lengths TB1 and B1B
axiom angle_BHB1 : ∠ B H B1 = 30 -- Given angle measure
axiom angle_B1BH : ∠ B1 B H = 30 -- Given angle measure
axiom angle_BHA_EQ_90 : ∠ B H A = 90 -- Given angle measure
axiom AB_GT_AH : d_AB > d_AT + d_TH -- Inequality on segment lengths
axiom AC_GT_AT_CT2 : d_CA > d_AT + d_CT / 2 -- Inequality on segment lengths
axiom BC_GT_BT_CT2 : d_BC > d_BT + d_CT / 2 -- Inequality on segment lengths

-- Theorem to prove
theorem inequality
  (A B C T H B1 : Type)
  (d_AT d_BT d_CT d_AB d_BC d_CA d_TB1 d_TH : ℝ)
  (midpoint_B1_BT : d_TB1 = d_BT / 2)
  (TH_eq_TB1 : d_TH = d_TB1)
  (angle_THB1 : ∠ T H B1 = 60)
  (angle_TB1H : ∠ T B1 H = 60)
  (angle_ATB1 : ∠ A T B1 = 120)
  (d_HB1_EQ_d_TB1 : d_H B1 = d_TB1)
  (d_TB1_EQ_d_B1B : d_TB1 = d_B1 B)
  (angle_BHB1 : ∠ B H B1 = 30)
  (angle_B1BH : ∠ B1 B H = 30)
  (angle_BHA_EQ_90 : ∠ B H A = 90)
  (AB_GT_AH : d_AB > d_AT + d_TH)
  (AC_GT_AT_CT2 : d_CA > d_AT + d_CT / 2)
  (BC_GT_BT_CT2 : d_BC > d_BT + d_CT / 2) :
  2 * d_AB + 2 * d_BC + 2 * d_CA > 4 * d_AT + 3 * d_BT + 2 * d_CT :=
sorry

end inequality_l63_63424


namespace num_integers_in_range_l63_63630

theorem num_integers_in_range : 
  let π_approx := 3.14 in
  let lower_bound := -6 * π_approx in
  let upper_bound := 12 * π_approx in
  let n_start := Int.ceil lower_bound in
  let n_end := Int.floor upper_bound in
  (n_end - n_start + 1) = 56 := 
by
  sorry

end num_integers_in_range_l63_63630


namespace negative_electrode_correct_l63_63033

-- Given conditions summarized as constants
constant methanol : Type
constant oxygen : Type
constant water : Type
constant carbon_dioxide : Type
constant proton : Type
constant electron : Type
constant negative_electrode_reaction : methanol → water → electron → carbon_dioxide → proton → Prop

-- Defining the options as constants
constant option_A : methanol → oxygen → electron → water → carbon_dioxide → proton → Prop
constant option_B : oxygen → proton → electron → water → Prop
constant option_C : methanol → water → electron → carbon_dioxide → proton → Prop
constant option_D : oxygen → water → electron → proton → Prop

-- The proof statement
theorem negative_electrode_correct :
  negative_electrode_reaction = option_C :=
sorry

end negative_electrode_correct_l63_63033


namespace complement_intersection_l63_63619

namespace SetProof

variable (U M N : Set ℕ)
variable [U_eq : U = {0, 1, 2, 3, 4}]
variable [M_eq : M = {0, 1, 2}]
variable [N_eq : N = {2, 3}]

theorem complement_intersection :
  (U \ M) ∩ N = {3} :=
by
  sorry
  
end SetProof

end complement_intersection_l63_63619


namespace positive_three_digit_integers_divisible_by_12_and_7_l63_63228

theorem positive_three_digit_integers_divisible_by_12_and_7 : 
  ∃ n : ℕ, n = 11 ∧ ∀ k : ℕ, (k ∣ 12) ∧ (k ∣ 7) ∧ (100 ≤ k) ∧ (k < 1000) :=
by
  sorry

end positive_three_digit_integers_divisible_by_12_and_7_l63_63228


namespace primary_contradiction_decisive_l63_63396

def analogy_interpretation (career_aspects_zero health_one : Prop) :=
  (∀ career healthy_objs, (career_aspects_zero → health_one) → healthy_objs → career ↔ endless_possibilities career healthy_objs)

theorem primary_contradiction_decisive 
  (career_aspects_zero : ∀ career, Prop)
  (health_one : ∀ health_objs, Prop)
  (primary_contradiction := analogy_interpretation career_aspects_zero health_one) :
  primary_contradiction ∧ (∀ (p q : Prop), q = false → primary_contradiction) :=
begin
  sorry
end

end primary_contradiction_decisive_l63_63396


namespace stops_to_mall_l63_63802

-- Assume the average speed of the bus is 60 km/h and the stops frequency is every 5 minutes.
def average_speed : ℝ := 60      -- in km/h
def stop_interval : ℝ := 5 / 60  -- in hours which is every 5 minutes
def distance_to_mall : ℝ := 25   -- distance from Yahya’s house to the Pinedale mall in kilometers

-- Claim: The number of stops from Yahya’s house to the Pinedale mall is 5.
theorem stops_to_mall : 
  ∀ average_speed stop_interval distance_to_mall, 
  average_speed = 60 → 
  stop_interval = 5 / 60 → 
  distance_to_mall = 25 → 
  (distance_to_mall / (average_speed * stop_interval)) = 5 := 
by 
  intros; 
  sorry

end stops_to_mall_l63_63802


namespace coefficient_of_third_term_l63_63697

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_third_term :
  let a := (x : ℚ)
  let b := (y : ℚ)
  let n := 8
  let r := 2
  a = x → b = 1/(2*√x) → ∑ k in range(n+1), binom n k * a^(n-k) * b^k = (7 : ℚ) :=
by
  sorry


end coefficient_of_third_term_l63_63697


namespace unique_solution_l63_63964

noncomputable def equation_satisfied (x : ℝ) : Prop :=
  2^x + 3^x + 6^x = 7^x

theorem unique_solution : ∀ x : ℝ, equation_satisfied x ↔ x = 2 := by
  sorry

end unique_solution_l63_63964


namespace min_value_fraction_l63_63991

theorem min_value_fraction (x : ℝ) (h : x > 0) : ∃ y, y = 4 ∧ (∀ z, z = (x + 5) / Real.sqrt (x + 1) → y ≤ z) := sorry

end min_value_fraction_l63_63991


namespace taxi_speed_l63_63418

theorem taxi_speed (v : ℝ) (h1 : ∀ t : ℝ, t ≥ 0 → t ≤ 2 → 
  v * t = (v - 30) * (t + 4) + (v - 30) * t)
(h2 : ∀ t : ℝ, t = 2) : v = 45 :=
by
  have bus_distance := (4 * (v - 30) + 2 * (v - 30))
  hyperref.transport.open_embedding.L = dia.B "2 v" begin sorry, sorry

end taxi_speed_l63_63418


namespace number_of_rabbits_is_38_l63_63109

-- Conditions: 
def ducks : ℕ := 52
def chickens : ℕ := 78
def condition (ducks rabbits chickens : ℕ) : Prop := 
  chickens = ducks + rabbits - 12

-- Statement: Prove that the number of rabbits is 38
theorem number_of_rabbits_is_38 : ∃ R : ℕ, condition ducks R chickens ∧ R = 38 := by
  sorry

end number_of_rabbits_is_38_l63_63109


namespace compound_has_two_hydrogen_atoms_l63_63909

noncomputable def number_of_hydrogen_atoms 
  (weight_compound : ℝ) (weight_Ca : ℝ) (weight_O : ℝ) (weight_H : ℝ) 
  (num_Ca : ℕ) (num_O : ℕ) 
  (total_weight : ℝ) 
  (atomic_weight_Ca : ℝ) (atomic_weight_O : ℝ) (atomic_weight_H : ℝ) :=
  (total_weight - (num_Ca * atomic_weight_Ca + num_O * atomic_weight_O)) / atomic_weight_H

theorem compound_has_two_hydrogen_atoms :
  number_of_hydrogen_atoms 74 40.08 16.00 1.008 1 2 74 40.08 16.00 1.008 ≈ 2 :=
by
  sorry

end compound_has_two_hydrogen_atoms_l63_63909


namespace alphonse_winning_strategy_l63_63471

theorem alphonse_winning_strategy :
  ∃ strategy : (ℕ → (list ℕ) → (list ℕ)), ∀ (turn : ℕ) (board : list ℕ),
  (turn % 2 = 1 → 
    (strategy turn board = (split_strategy (board.head) ++ board.tail))
    ∧ (∀ x ∈ board, (x > 1 → (split x).length = 2))) →
  (turn % 2 = 0 → 
    (strategy turn board = (erase_strategy board)) 
    ∧ (∃ x ∈ board, (board.count x) > 1)) →
  (∃ turn_end game_end, (game_end [10^2011] = true) → turn_end = alphonse)
:= 
begin 
  sorry
end

end alphonse_winning_strategy_l63_63471


namespace sum_of_coeffs_eq_negative_21_l63_63519

noncomputable def expand_and_sum_coeff (d : ℤ) : ℤ :=
  let expression := -(4 - d) * (d + 2 * (4 - d))
  let expanded_form := -d^2 + 12*d - 32
  let sum_of_coeffs := -1 + 12 - 32
  sum_of_coeffs

theorem sum_of_coeffs_eq_negative_21 (d : ℤ) : expand_and_sum_coeff d = -21 := by
  sorry

end sum_of_coeffs_eq_negative_21_l63_63519


namespace fill_parentheses_l63_63693

variable (a b : ℝ)

theorem fill_parentheses :
  1 - a^2 + 2 * a * b - b^2 = 1 - (a^2 - 2 * a * b + b^2) :=
by
  sorry

end fill_parentheses_l63_63693


namespace triangle_max_sum_bc_correct_l63_63701

noncomputable def triangle_max_sum_bc (a b c : ℝ) (A B C : ℝ) : ℝ :=
if h : a = 3 ∧ (1 + tan A / tan B) = (2 * c / b)
then 3 * Real.sqrt 2
else 0

theorem triangle_max_sum_bc_correct {a b c A B C : ℝ} (h1 : a = 3) (h2 : 1 + tan A / tan B = 2 * c / b) :
  triangle_max_sum_bc a b c A B C = 3 * Real.sqrt 2 :=
begin
  sorry
end

end triangle_max_sum_bc_correct_l63_63701


namespace unique_solution_l63_63642

theorem unique_solution : ∃! (n : ℕ), (n > 0) ∧ (n + 1100) / 80 = int.floor (real.sqrt n) :=
by
  sorry

end unique_solution_l63_63642


namespace reconstruct_triangle_ABC_unique_l63_63407

noncomputable def reconstruct_triangle (M I Q_a : Point) (hM : is_centroid M) (hI : is_incenter I) (hQa : is_incircle_touch_point Q_a BC) : Triangle :=
  sorry

theorem reconstruct_triangle_ABC_unique (M I Q_a : Point) (hM : is_centroid M) (hI : is_incenter I) (hQa : is_incircle_touch_point Q_a BC) : 
  ∃! (ABC : Triangle), reconstruct_triangle M I Q_a hM hI hQa = ABC :=
  sorry

end reconstruct_triangle_ABC_unique_l63_63407


namespace max_value_of_expr_l63_63548

noncomputable def max_value_expr (a b c x y z : ℝ) : ℝ :=
  (a / x) + (a + b) / (x + y) + (a + b + c) / (x + y + z)

theorem max_value_of_expr {a b c x y z : ℝ} 
  (ha : 2 ≤ a ∧ a ≤ 3) (hb : 2 ≤ b ∧ b ≤ 3) (hc : 2 ≤ c ∧ c ≤ 3)
  (hperm : {x, y, z} = {a, b, c}) : 
  max_value_expr a b c x y z ≤ 15 / 4 :=
by
  sorry

end max_value_of_expr_l63_63548


namespace heroes_on_the_back_l63_63013

theorem heroes_on_the_back (total_heroes front_heroes : ℕ) (h1 : total_heroes = 9) (h2 : front_heroes = 2) :
  total_heroes - front_heroes = 7 := by
  sorry

end heroes_on_the_back_l63_63013


namespace trigonometric_relationship_l63_63164

theorem trigonometric_relationship (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = (1 - Real.sin β) / Real.cos β) : 
  2 * α + β = π / 2 := 
sorry

end trigonometric_relationship_l63_63164


namespace count_n_for_perfect_square_l63_63161

theorem count_n_for_perfect_square (n : ℤ) : 
  (∃ k : ℤ, (n / (15 - n) = k * k)) → 
  0 ≤ n ∧ n < 15 → 
  (finset.count (λ (n : ℤ), ∃ k : ℤ, (n / (15 - n) = k * k)) (finset.Ico 0 15)) = 2 := 
sorry

end count_n_for_perfect_square_l63_63161


namespace calculate_x_l63_63088

theorem calculate_x :
  529 - 2 * 23 * 8 + 64 = 225 := 
begin
  sorry
end

end calculate_x_l63_63088


namespace rose_needs_more_money_l63_63332

def cost_paintbrush : ℝ := 2.40
def cost_paints : ℝ := 9.20
def cost_easel : ℝ := 6.50
def money_rose_has : ℝ := 7.10

theorem rose_needs_more_money : 
  cost_paintbrush + cost_paints + cost_easel - money_rose_has = 11.00 :=
begin
  sorry
end

end rose_needs_more_money_l63_63332


namespace simplify_factorial_division_l63_63786

theorem simplify_factorial_division : (13.factorial / (11.factorial + 3 * 10.factorial)) = 1716 := by
  sorry

end simplify_factorial_division_l63_63786


namespace linear_function_through_point_parallel_line_l63_63372

noncomputable def function_expr (x : ℝ) : ℝ := 2 * x + 3

def point_A : ℝ × ℝ := (-2, -1)

def parallel_line (x : ℝ) : ℝ := 2 * x - 3

theorem linear_function_through_point_parallel_line :
  ∃ b : ℝ, (∀ x : ℝ, function_expr x = 2 * x + b) ∧ (function_expr (fst point_A) = snd point_A) :=
by
  use 3
  split
  . intro x
    refl
  . simp [function_expr, point_A]
    sorry

end linear_function_through_point_parallel_line_l63_63372


namespace converse_even_sum_l63_63012

theorem converse_even_sum (a b : ℕ) (ha : Even a) (hb : Even b) : Even (a + b) :=
sorry

end converse_even_sum_l63_63012


namespace quadrilateral_area_ratio_l63_63767

theorem quadrilateral_area_ratio (AB BC CD DA : ℝ) (φ : ℝ) 
  (hABCD_area : (AB * BC) / 2 * (sin φ) = 2 * (((AB / 2) * (BC / 2)) / 2) * (sin φ)) :
  (AB * BC) / (2 * (((AB / 2) * (BC / 2)) / 4)) = 2 :=
by
  sorry

end quadrilateral_area_ratio_l63_63767


namespace probability_all_blue_jellybeans_removed_l63_63898

def num_red_jellybeans : ℕ := 10
def num_blue_jellybeans : ℕ := 10
def total_jellybeans : ℕ := num_red_jellybeans + num_blue_jellybeans

def prob_first_blue : ℚ := num_blue_jellybeans / total_jellybeans
def prob_second_blue : ℚ := (num_blue_jellybeans - 1) / (total_jellybeans - 1)
def prob_third_blue : ℚ := (num_blue_jellybeans - 2) / (total_jellybeans - 2)

def prob_all_blue : ℚ := prob_first_blue * prob_second_blue * prob_third_blue

theorem probability_all_blue_jellybeans_removed :
  prob_all_blue = 1 / 9.5 := sorry

end probability_all_blue_jellybeans_removed_l63_63898


namespace principal_amount_l63_63460

theorem principal_amount (A2 A3 : ℝ) (interest : ℝ) (principal : ℝ) (h1 : A2 = 3450) 
  (h2 : A3 = 3655) (h_interest : interest = A3 - A2) (h_principal : principal = A2 - interest) : 
  principal = 3245 :=
by
  sorry

end principal_amount_l63_63460


namespace not_possible_to_make_all_equal_1984_l63_63045

theorem not_possible_to_make_all_equal_1984 :
  ∀ (n : ℕ), ∑ i in finset.range 12, (i + 1) = 78 ∧ n * 78 ≠ 12 * 1984 :=
by
  intro n
  have sum_1_to_12 : ∑ i in finset.range 12, (i + 1) = 78 :=
    calc
      ∑ i in finset.range 12, (i + 1) = (12 * (12 + 1)) / 2 : by sorry 
      ... = 78 : by sorry
  have target_sum : 12 * 1984 = 23808 := by sorry
  have n_times_78 : n * 78 ≠ 23808 := by sorry
  exact ⟨sum_1_to_12, n_times_78⟩

end not_possible_to_make_all_equal_1984_l63_63045


namespace solve_for_k_l63_63512

theorem solve_for_k : 
  (∫ x in 0..1, 2 * x + k) = 2 → k = 1 := by
sorry

end solve_for_k_l63_63512


namespace difference_in_pay_l63_63754

theorem difference_in_pay (pay_per_delivery : ℕ) (oula_deliveries : ℕ) (fraction_tona_oula : ℚ)
  (oula_pay : ℕ) (tona_pay : ℕ) (difference_pay : ℕ) :
  pay_per_delivery = 100 →
  oula_deliveries = 96 →
  fraction_tona_oula = 3/4 →
  oula_pay = oula_deliveries * pay_per_delivery →
  tona_pay = (oula_deliveries * fraction_tona_oula).natAbs * pay_per_delivery →
  difference_pay = oula_pay - tona_pay →
  difference_pay = 2400 :=
by
  intros
  sorry

end difference_in_pay_l63_63754


namespace find_unknown_polynomial_l63_63235

theorem find_unknown_polynomial (m : ℤ) : 
  ∃ q : ℤ, (q + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1) → q = 2 * m^2 + 3 * m - 4 :=
by {
  sorry
}

end find_unknown_polynomial_l63_63235


namespace find_point_P_l63_63564

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

def magnitude_ratio (P A B : Point) (r : ℝ) : Prop :=
  let AP := vector A P
  let PB := vector P B
  (AP.x, AP.y) = (r * PB.x, r * PB.y)

theorem find_point_P (P : Point) : 
  magnitude_ratio P A B (4/3) → (P.x = 10 ∧ P.y = -21) :=
sorry

end find_point_P_l63_63564


namespace problem_statement_l63_63818

noncomputable def f (x t: ℝ) : ℝ := abs (x^2 + x - t)

def maximum_value_condition (t : ℝ) : Prop := 
  ∀ x ∈ set.Icc (-1 : ℝ) (2 : ℝ), f x t ≤ 4

theorem problem_statement (t : ℝ) : maximum_value_condition t ↔ (t = 2 ∨ t = 15 / 4) :=
by
  sorry

end problem_statement_l63_63818


namespace tangent_line_sin_at_pi_l63_63812

open Real

theorem tangent_line_sin_at_pi :
  ∀ (x y : ℝ), y = sin x → ((x = π ∧ y = 0) → (x + y - π = 0)) :=
by {
  intros,
  sorry
}

end tangent_line_sin_at_pi_l63_63812


namespace arrange_descending_order_l63_63081

theorem arrange_descending_order :
  let a := (2 : ℝ)^(-333)
  let b := (3 : ℝ)^(-222)
  let c := (5 : ℝ)^(-111)
  c > a ∧ a > b :=
by sorry

end arrange_descending_order_l63_63081


namespace exists_divisible_by_2_n_l63_63723

theorem exists_divisible_by_2_n (n : ℕ) (hn : n > 0) : 
  ∃ N : ℕ, (∃ (m : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → m_i ∈ {1, 2}) ∧ N = ∑ i in finset.range n, m_i * 10^(n-i)) ∧ N % 2^n = 0 :=
sorry

end exists_divisible_by_2_n_l63_63723


namespace proposition_d_l63_63645

open_locale classical

variables {Point Line Plane : Type} [linear_space Point Line Plane]

/-- Defining relations and properties --/
def is_parallel (a b : Line) : Prop := sorry -- Define parallel condition
def is_skew (a b : Line) : Prop := sorry -- Define skew condition
def is_contained_in (b : Line) (α : Plane) : Prop := sorry -- b is contained in α

/-- Main theorem statement --/
theorem proposition_d (a b : Line) (α : Plane)
  (h1 : is_parallel a α) (h2 : is_contained_in b α) :
  is_parallel a b ∨ is_skew a b :=
sorry

end proposition_d_l63_63645


namespace quadratic_inequality_solution_l63_63826

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 3 * x + 2 < 0) : 
  -0.25 < x^2 - 3 * x + 2 ∧ x^2 - 3 * x + 2 < 0 :=
by
  have h_interval : 1 < x ∧ x < 2 :=
    sorry  -- This part of the proof would involve proving that the inequality defines the interval (1, 2)
  refine ⟨_, h⟩
  have eval_at_mid : (1.5 : ℝ)^2 - 3 * 1.5 + 2 = -0.25 :=
    by norm_num
  sorry  -- This would be where we show that -0.25 is the lower bound for x^2 - 3 * x + 2 in the interval (1, 2)

end quadratic_inequality_solution_l63_63826


namespace student_community_arrangements_l63_63993

theorem student_community_arrangements :
  (3 ^ 4) = 81 :=
by
  sorry

end student_community_arrangements_l63_63993


namespace selling_price_same_as_loss_l63_63382

open FloatingPoint
open Real

noncomputable def cost_price : ℝ := 1750 / 1.25
noncomputable def percentage_loss : ℝ := (1280 - cost_price) / cost_price * 100 / 100 -- Simplified to percent form
noncomputable def required_selling_price : ℝ := cost_price + percentage_loss * cost_price

theorem selling_price_same_as_loss (h1: percentage_loss = 8.57) : required_selling_price = 1519.98 :=
by sorry

end selling_price_same_as_loss_l63_63382


namespace general_term_of_sequence_l63_63389

theorem general_term_of_sequence
  (S : ℕ → ℕ)
  (hS : ∀ n : ℕ, S (n+1) = 2 ^ (n + 1) - 1)
: ∀ n : ℕ, n > 0 → (S n - S (n - 1) = 2 ^ (n-1)) :=
begin
  intros n hn,
  cases n,
  { exact (by linarith : absurd $ by {simp} : 0 > 0).elim },
  { have h₀ := hS n,
    have h₁ := if hn : n = 0 then 1 else 2 ^ n - 1,
    rw hS n,
    rw hS (n-1),
    simp,
    linarith }
end

end general_term_of_sequence_l63_63389


namespace num_palindrome_divisible_by_9_l63_63913

theorem num_palindrome_divisible_by_9 : 
  ∃ n : ℕ, n = 10 ∧
  (∀ x : ℕ, 1000 ≤ x ∧ x ≤ 9999 ∧
    (let digit1 := x / 1000,
         digit2 := (x % 1000) / 100,
         digit3 := (x % 100) / 10,
         digit4 := x % 10 in
     digit1 = digit4 ∧ digit2 = digit3 ∧
     (digit1 + digit2 + digit3 + digit4) % 9 = 0) →
    (∃ k : ℕ, x = 1001 * k ∧ k < 10)) :=
by
  sorry

end num_palindrome_divisible_by_9_l63_63913


namespace nth_equation_l63_63304

theorem nth_equation (n : ℕ) : 
  (finset.sum (finset.range (2 * n - 1)) (λ i, n + i)) = (2 * n - 1) ^ 2 :=
by
  sorry

end nth_equation_l63_63304


namespace degree_measure_of_angle_is_correct_l63_63146

noncomputable def degree_measure_of_angle : ℝ :=
  let x := 60 in
  x

theorem degree_measure_of_angle_is_correct (x : ℝ) :
  (90 - x = (1 / 4) * (180 - x)) → x = 60 :=
by
  intro h
  sorry

end degree_measure_of_angle_is_correct_l63_63146


namespace james_profit_l63_63705

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l63_63705


namespace range_of_a_l63_63671

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1
def intersects_at_single_point (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
∃! x, f x a = 3

theorem range_of_a (a : ℝ) :
  intersects_at_single_point f a ↔ -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l63_63671


namespace cot_neg_45_is_neg_1_l63_63121

theorem cot_neg_45_is_neg_1 : Real.cot (Real.pi * -45 / 180) = -1 :=
by
  sorry

end cot_neg_45_is_neg_1_l63_63121


namespace distinct_sequences_triangle_l63_63227

open Finset

theorem distinct_sequences_triangle : 
  let available_letters := {'R', 'I', 'A', 'N', 'G', 'L'} : Finset Char,
      choose_4_letters := available_letters.choose 4,
      permute_4_letters := \u (x : Finset (Finset Char)) (x ∈ choose_4_letters), x.card.perm 4
  in ∑ val in choose_4_letters, val.card.perm (4 : ℕ) = 360 := sorry

end distinct_sequences_triangle_l63_63227


namespace mrs_hilt_candy_l63_63302

theorem mrs_hilt_candy : 2 * 9 + 3 * 9 + 1 * 9 = 54 :=
by
  sorry

end mrs_hilt_candy_l63_63302


namespace real_y_iff_x_ranges_l63_63662

-- Definitions for conditions
variable (x y : ℝ)

-- Condition for the equation
def equation := 9 * y^2 - 6 * x * y + 2 * x + 7 = 0

-- Theorem statement
theorem real_y_iff_x_ranges :
  (∃ y : ℝ, equation x y) ↔ (x ≤ -2 ∨ x ≥ 7) :=
sorry

end real_y_iff_x_ranges_l63_63662


namespace number_of_small_triangles_required_l63_63932

-- Define the main statement
theorem number_of_small_triangles_required :
  let large_triangle_side := 10
  let small_triangle_side := 1
  equilateral_triangle_area(large_triangle_side) / equilateral_triangle_area(small_triangle_side) = 100 := by
  sorry

-- Helper function to calculate area of equilateral triangle given its side length
noncomputable def equilateral_triangle_area (s : ℕ) : ℝ :=
  (math.sqrt 3 / 4) * (s ^ 2)

#eval number_of_small_triangles_required

end number_of_small_triangles_required_l63_63932


namespace isosceles_triangle_vertex_angle_range_l63_63004

theorem isosceles_triangle_vertex_angle_range
  (a b : ℝ)
  (α : ℝ)
  (triangle : isosceles_triangle a b α)
  (three_lines_exist : ∃ lines : list (line ℝ), 
    (∀ l ∈ lines, bisects_area_triangle l triangle ∧ bisects_perimeter_triangle l triangle) 
    ∧ lines.length = 3) : 
  2 * real.arcsin (real.sqrt 2 - 1) < α ∧ α < real.pi :=
sorry

end isosceles_triangle_vertex_angle_range_l63_63004


namespace solution_set_max_value_l63_63206

-- Given function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |x - 1|

-- (I) Prove the solution set of f(x) ≤ 4 is {x | -2/3 ≤ x ≤ 2}
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 2} :=
sorry

-- (II) Given m is the minimum value of f(x)
def m := 1 / 2

-- Given a, b, c ∈ ℝ^+ and a + b + c = m
variables (a b c : ℝ)
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a + b + c = m)

-- Prove the maximum value of √(2a + 1) + √(2b + 1) + √(2c + 1) is 2√3
theorem max_value : (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1)) ≤ 2 * Real.sqrt 3 :=
sorry

end solution_set_max_value_l63_63206


namespace ladybugs_count_l63_63273

theorem ladybugs_count (n : ℕ) (h6 : ∃ lb6 : ℕ, lb6 >= 0) (h4 : ∃ lb4 : ℕ, lb4 >= 0) :
  (∃ lb1 lb2 lb3 : ℕ, lb1 + lb2 + lb3 = n) ∧
  ((∀ i, lb1 = lb2 ∧ lb2 = lb3) ∨ (lb1 ≠ lb2 ∨ lb2 ≠ lb3 ∨ lb1 ≠ lb3)) →
  (lb2 * 4 + lb3 * 6 + 4 * (n - (lb2 + lb3)) = 30 ∨ lb2 * 4 + lb3 * 6 + 4 * (n - (lb2 + lb3)) = 26) →
  (lb1 = 6 ↔ lb2 > 0) →
  (lb1 = 4 ↔ lb3 > 0) →
  n = 5 :=
by
  sorry

end ladybugs_count_l63_63273


namespace math_proof_problem_l63_63199

noncomputable def f : ℝ → ℝ :=
sorry -- Definition is implicit from conditions

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_one : f 1 = 1

theorem math_proof_problem :
  (∑ i in finset.range 1009, (f (2*i + 1) + 1) / f (2*i)) = 2018 :=
by
  sorry

end math_proof_problem_l63_63199


namespace meet_distance_l63_63756

-- Definitions based on conditions
variables {v_A v_B : ℝ} (t : ℝ)
variable {A B C D : ℝ} -- Represent points and distances between them

-- Given conditions
axiom condition1 : C = (A + B) / 2 -- Midpoint
axiom condition2 : (C - B).abs = 240 -- B is 240 meters away from C when A reaches C
axiom condition3 : (A + 360 - C).abs = 360 -- A is 360 meters past C when B reaches C

-- Speed ratio from the conditions:
axiom speed_ratio : v_A / v_B = 3 / 2

-- Time taken for Yi to travel 240 meters
axiom time_B_reaches_C : v_B * t = 240
axiom time_A_travels_360 : v_A * t = 360

-- Define the distance between C and D to be proved
def distance_CD : ℝ := 240 * (3 / (2 + 3))

theorem meet_distance : distance_CD = 144 :=
by
  sorry

end meet_distance_l63_63756


namespace collinear_and_ratio_l63_63313

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C : V)

-- Define O as the circumcenter
def circumcenter (A B C : V) : V := sorry

-- Define G as the centroid
def centroid (A B C : V) : V := (A + B + C) / 3

-- Define H as the orthocenter
def orthocenter (A B C : V) : V := A + B + C

-- The theorem we need to prove
theorem collinear_and_ratio (O G H : V) (hO: O = circumcenter A B C) 
  (hG: G = centroid A B C) (hH: H = orthocenter A B C) : 
  collinear ({O, G, H} : set V) ∧ dist O G = dist G H * 2 := 
sorry

end collinear_and_ratio_l63_63313


namespace liars_and_truth_tellers_l63_63027

theorem liars_and_truth_tellers :
  ∃ (liars truth_tellers : Fin 2000 → Prop), 
  (∀ n, (liars n ∨ truth_tellers n)) ∧
  (∀ n, ¬(liars n ∧ truth_tellers n)) ∧
  (∀ n, if liars n then 
            (∃ k, k < n ∧ liars k) ∧ 
            ∃ m, (m > n → truth_tellers m)) ∧
            (¬(exists t, (t > n ∧ ¬truth_tellers t) ∧ (∃ k, (k < n ∧ liars k)))
   else 
            (∃ k, (k < n ∧ liars k) ∧ (∀ m, (m > n → truth_tellers m))))) ∧ 
  (∃ n, liars n) ∧ 
  (∃ n, truth_tellers n) ∧
  (∃ count_liars count_truth_tellers, 
  (count_liars = 1000) ∧ 
  (count_truth_tellers = 1000)) := sorry

end liars_and_truth_tellers_l63_63027


namespace max_sum_visible_faces_of_stacked_dice_l63_63342

def opp_faces : (ℕ × ℕ) := [(1, 6), (2, 5), (3, 4)] 

theorem max_sum_visible_faces_of_stacked_dice
  (dice : list (ℕ × ℕ))
  (H : dice = opp_faces) :
  ∃ max_sum : ℕ, max_sum = 89 :=
by
  sorry

end max_sum_visible_faces_of_stacked_dice_l63_63342


namespace ace_then_king_probability_l63_63846

theorem ace_then_king_probability :
  let total_cards := 52
  let aces := 4
  let kings := 4
  let first_ace_prob := (aces : ℚ) / (total_cards : ℚ)
  let second_king_given_ace_prob := (kings : ℚ) / (total_cards - 1 : ℚ)
  (first_ace_prob * second_king_given_ace_prob = (4 : ℚ) / 663) :=
by
  let total_cards := 52
  let aces := 4
  let kings := 4
  let first_ace_prob := (aces : ℚ) / (total_cards : ℚ)
  let second_king_given_ace_prob := (kings : ℚ) / (total_cards - 1 : ℚ)
  exact (first_ace_prob * second_king_given_ace_prob = (4 : ℚ) / 663)
  sorry

end ace_then_king_probability_l63_63846


namespace hundredth_odd_integer_l63_63863

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l63_63863


namespace no_integer_solutions_l63_63532

theorem no_integer_solutions (x y : ℤ) : 19 * x^3 - 84 * y^2 ≠ 1984 :=
by
  sorry

end no_integer_solutions_l63_63532


namespace least_positive_integer_solution_l63_63148

theorem least_positive_integer_solution :
  ∃ x : ℕ, (x + 7391) % 12 = 167 % 12 ∧ x = 8 :=
by 
  sorry

end least_positive_integer_solution_l63_63148


namespace proposition_C_proposition_D_l63_63413

-- Define the vectors and conditions for proposition C
def n1 : ℝ × ℝ × ℝ := (2, -1, 0)
def n2 : ℝ × ℝ × ℝ := (-4, 2, 0)

-- Define points and the normal vector condition for proposition D
def A : ℝ × ℝ × ℝ := (1, 0, -1)
def B : ℝ × ℝ × ℝ := (0, 1, 0)
def C : ℝ × ℝ × ℝ := (-1, 2, 0)
def n (u t : ℝ) : ℝ × ℝ × ℝ := (1, u, t)

-- The propositions to prove
theorem proposition_C : ∃ k : ℝ, n2 = (k * n1.1, k * n1.2, k * n1.3) := by
  sorry

theorem proposition_D (u t : ℝ) : 
  (1, u, t) = (1, u, t) →
  (u * (B.1 - A.1) + u * (B.2 - A.2) + t * (C.2 - B.2) = 0) →
  u + t = 1 := by 
  sorry

end proposition_C_proposition_D_l63_63413


namespace area_of_pentagon_PTRSQ_l63_63319

theorem area_of_pentagon_PTRSQ (PQRS : Type) [geometry PQRS]
  {P Q R S T : PQRS} 
  (h1 : square P Q R S) 
  (h2 : perp PT TR) 
  (h3 : distance P T = 5) 
  (h4 : distance T R = 12) : 
  area_pentagon PTRSQ = 139 :=
sorry

end area_of_pentagon_PTRSQ_l63_63319


namespace max_correct_answers_l63_63676

variables {a b c : ℕ} -- Define a, b, and c as natural numbers

theorem max_correct_answers : 
  ∀ a b c : ℕ, (a + b + c = 50) → (5 * a - 2 * c = 150) → a ≤ 35 :=
by
  -- Proof steps can be skipped by adding sorry
  sorry

end max_correct_answers_l63_63676


namespace max_value_expression_l63_63544

theorem max_value_expression
  (a b c x y z : ℝ)
  (h₁ : 2 ≤ a ∧ a ≤ 3)
  (h₂ : 2 ≤ b ∧ b ≤ 3)
  (h₃ : 2 ≤ c ∧ c ≤ 3)
  (h4 : {x, y, z} = {a, b, c}) :
  (a / x + (a + b) / (x + y) + (a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_value_expression_l63_63544


namespace sum_of_intercepts_l63_63552

theorem sum_of_intercepts (x y : ℝ) (h : 3 * x - 4 * y - 12 = 0) :
    (y = -3 ∧ x = 4) → x + y = 1 :=
by
  intro h'
  obtain ⟨hy, hx⟩ := h'
  rw [hy, hx]
  norm_num
  done

end sum_of_intercepts_l63_63552


namespace number_of_valid_arrangements_l63_63501

-- Definitions of the marbles
inductive Marble
| Aggie
| Bumblebee
| Steelie
| Tiger
| CatsEye

open Marble

-- Definition of arrangements of marbles
def is_not_adjacent (xs : List Marble) (x y : Marble) : Prop :=
  ¬(List.pairwise_adjacent xs (fun a b => a = x ∧ b = y ∨ a = y ∧ b = x))

def valid_arrangement (arrangement : List Marble) : Prop :=
  arrangement.length = 5 ∧
  is_not_adjacent arrangement Steelie Tiger ∧
  is_not_adjacent arrangement Bumblebee CatsEye

-- The proof statement
theorem number_of_valid_arrangements : 
  (List.permutations [Aggie, Bumblebee, Steelie, Tiger, CatsEye]).countp valid_arrangement = 120 :=
sorry

end number_of_valid_arrangements_l63_63501


namespace total_cost_of_items_l63_63776

theorem total_cost_of_items (cost_of_soda : ℕ) (cost_of_soup : ℕ) (cost_of_sandwich : ℕ) (total_cost : ℕ) 
  (h1 : cost_of_soda = 1)
  (h2 : cost_of_soup = 3 * cost_of_soda)
  (h3 : cost_of_sandwich = 3 * cost_of_soup) :
  total_cost = 3 * cost_of_soda + 2 * cost_of_soup + cost_of_sandwich :=
by
  unfold total_cost
  show 3 * 1 + 2 * (3 * 1) + (3 * (3 * 1)) = 18
  rfl

end total_cost_of_items_l63_63776


namespace paige_team_total_players_l63_63755

theorem paige_team_total_players 
    (total_points : ℕ)
    (paige_points : ℕ)
    (other_points_per_player : ℕ)
    (other_players : ℕ) :
    total_points = paige_points + other_points_per_player * other_players →
    (other_players + 1) = 6 :=
by
  intros h
  sorry

end paige_team_total_players_l63_63755


namespace area_triangle_PQR_l63_63822

-- Define points P, Q, R.
def P : (ℝ × ℝ) := (5, 3)
def Q : (ℝ × ℝ) := (-5, 3)
def R : (ℝ × ℝ) := (-3, 5)

-- Function to compute the distance between two points.
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- Function to compute the area of a triangle given its vertices.
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Theorem statement: Prove the area of triangle PQR is equal to 10.
theorem area_triangle_PQR : triangle_area P Q R = 10 := sorry

end area_triangle_PQR_l63_63822


namespace probability_of_Ace_then_King_l63_63844

def numAces : ℕ := 4
def numKings : ℕ := 4
def totalCards : ℕ := 52

theorem probability_of_Ace_then_King : 
  (numAces / totalCards) * (numKings / (totalCards - 1)) = 4 / 663 :=
by
  sorry

end probability_of_Ace_then_King_l63_63844


namespace average_multiplied_by_five_l63_63804

theorem average_multiplied_by_five (s : Fin 7 → ℝ) (h : (∑ i, s i) / 7 = 25) :
  ((∑ i, 5 * s i) / 7) = 125 :=
by
  sorry

end average_multiplied_by_five_l63_63804


namespace problem_inequality_l63_63721

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then 2 else (n^2 + 1) / real.sqrt (n^3 - 2*n^2 + n)

noncomputable def sequence_s (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, sequence_a (k + 1))

theorem problem_inequality (n : ℕ) (hn : n > 0) :
  (finset.range n).sum (λ k, 1 / (sequence_s k.succ * sequence_s (k.succ + 1))) < 1 / 5 :=
begin
  sorry
end

end problem_inequality_l63_63721


namespace calculate_profit_l63_63708

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l63_63708


namespace polar_to_rectangular_l63_63503

theorem polar_to_rectangular (ρ θ x y : ℝ) (h1 : ρ * cos θ = x) (h2 : ρ * sin θ = y) (h3 : ρ^2 = x^2 + y^2) : 
  (x^2 + y^2 = 0 ∨ x = 1) ↔ (ρ^2 * cos θ - ρ = 0) := by
  sorry

end polar_to_rectangular_l63_63503


namespace sum_of_coeffs_expansion_l63_63522

theorem sum_of_coeffs_expansion (d : ℝ) : 
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
in coeff_sum = -21 := 
by
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
    exact rfl

end sum_of_coeffs_expansion_l63_63522


namespace point_on_graph_of_even_function_l63_63589

variable {α β : Type} [OrderedRing α] [LinearOrderedField β]

-- Conditions
def is_even_function (f : α → β) : Prop := ∀ x : α, f (-x) = f (x)

variable {f : α → β} (a : α)

-- Problem Statement
theorem point_on_graph_of_even_function (h : is_even_function f) :
  (∀ a : α, (a, f a) ∈ SetOf (λ (p : α × β), p.snd = f p.fst)) →
  (-a, f a) ∈ SetOf (λ (p : α × β), p.snd = f p.fst) :=
by
  intro ha_condition
  sorry

end point_on_graph_of_even_function_l63_63589


namespace sample_size_l63_63910

theorem sample_size (n : ℕ) : 
  let ratio_A := 1
  let ratio_B := 3
  let ratio_C := 5 
  let total_ratio := 1 + 3 + 5
  let sampling_ratio_B := ratio_B / total_ratio
  let num_items_B := 12
  n = num_items_B / sampling_ratio_B := 36 :=
sorry

end sample_size_l63_63910


namespace ed_initial_money_l63_63957

-- Define initial conditions
def cost_per_hour_night : ℝ := 1.50
def hours_at_night : ℕ := 6
def cost_per_hour_morning : ℝ := 2
def hours_in_morning : ℕ := 4
def money_left : ℝ := 63

-- Total cost calculation
def total_cost : ℝ :=
  (cost_per_hour_night * hours_at_night) + (cost_per_hour_morning * hours_in_morning)

-- Problem statement to prove
theorem ed_initial_money : money_left + total_cost = 80 :=
by sorry

end ed_initial_money_l63_63957


namespace proof_problem_l63_63740

variables {A B C D E : Type} -- Define points A, B, C, D, E
variables [EuclideanGeometry A B C D E] -- Assume Euclidean geometry for these points
variables (concyclic : Concyclic A B C D) (intersect : Intersect (LineSegment A B) (LineSegment C D) E)

theorem proof_problem 
  (concyclic_points : Concyclic A B C D)
  (intersecting_lines : Intersect (LineSegment A B) (LineSegment C D) E) :
  ( (Distance A C) / (Distance B C) ) * ( (Distance A D) / (Distance B D) ) = ( (Distance A E) / (Distance B E) ) :=
  sorry

end proof_problem_l63_63740


namespace ab_div_c_eq_one_l63_63019

theorem ab_div_c_eq_one (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hne1 : A ≠ B) (hne2 : A ≠ C) (hne3 : B ≠ C) :
  (1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / 1))) → (A + B) / C = 1 :=
by sorry

end ab_div_c_eq_one_l63_63019


namespace monotonic_intervals_max_and_min_values_in_interval_l63_63604

def f (x : ℝ) : ℝ := x^2 * (x - 1)

theorem monotonic_intervals :
  (∀ x, x < 0 → (deriv f x) > 0) ∧ (∀ x, x > (2/3 : ℝ) → (deriv f x) > 0) ∧ 
  (∀ x, 0 < x ∧ x < (2/3 : ℝ) → (deriv f x) < 0) :=
sorry

theorem max_and_min_values_in_interval :
  ∃ max min,
  max = 4 ∧ min = -2 ∧ is_max_on f (set.Icc (-1 : ℝ) 2) 4 ∧ is_min_on f (set.Icc (-1 : ℝ) 2) (-2) :=
sorry

end monotonic_intervals_max_and_min_values_in_interval_l63_63604


namespace series_sum_eq_l63_63497

theorem series_sum_eq : 
  (∑' n, (4 * n + 3) / ((4 * n - 2) ^ 2 * (4 * n + 2) ^ 2)) = 1 / 128 := by
sorry

end series_sum_eq_l63_63497


namespace sodium_bicarbonate_moles_l63_63977

theorem sodium_bicarbonate_moles (HCl NaHCO3 CO2 : ℕ) (h1 : HCl = 1) (h2 : CO2 = 1) :
  NaHCO3 = 1 :=
by sorry

end sodium_bicarbonate_moles_l63_63977


namespace mileage_per_gallon_l63_63905

noncomputable def car_mileage (distance: ℝ) (gasoline: ℝ) : ℝ :=
  distance / gasoline

theorem mileage_per_gallon :
  car_mileage 190 4.75 = 40 :=
by
  -- proof omitted
  sorry

end mileage_per_gallon_l63_63905


namespace BoatCrafters_boats_total_l63_63486

theorem BoatCrafters_boats_total
  (n_february: ℕ)
  (h_february: n_february = 5)
  (h_march: 3 * n_february = 15)
  (h_april: 3 * 15 = 45) :
  n_february + 15 + 45 = 65 := 
sorry

end BoatCrafters_boats_total_l63_63486


namespace s_1_eq_1_l63_63887

def perfect_square (n : ℕ) : ℕ := n * n

def s (n : ℕ) : ℕ := 
  let seq := List.range n -- generates sequence [0, 1, 2, ..., n-1]
  let squares := seq.map perfect_square -- maps sequence to their squares
  let concatenated_string := squares.foldl (λ acc sq, acc ++ sq.repr) "" -- convert each square to string and concatenate
  concatenated_string.to_nat -- convert final concatenated string to natural number

theorem s_1_eq_1 : s 1 = 1 :=
by
  sorry

end s_1_eq_1_l63_63887


namespace minimum_value_fraction_l63_63183

theorem minimum_value_fraction (m n : ℝ) (h0 : 0 ≤ m) (h1 : 0 ≤ n) (h2 : m + n = 1) :
  ∃ min_val, min_val = (1 / 4) ∧ (∀ m n, 0 ≤ m → 0 ≤ n → m + n = 1 → (m^2) / (m + 2) + (n^2) / (n + 1) ≥ min_val) :=
sorry

end minimum_value_fraction_l63_63183


namespace correct_answer_is_A_l63_63880

-- Definitions derived from problem conditions
def algorithm := Type
def has_sequential_structure (alg : algorithm) : Prop := sorry -- Actual definition should define what a sequential structure is for an algorithm

-- Given: An algorithm must contain a sequential structure.
theorem correct_answer_is_A (alg : algorithm) : has_sequential_structure alg :=
sorry

end correct_answer_is_A_l63_63880


namespace count_integers_in_range_l63_63637

theorem count_integers_in_range : 
  { n : ℤ | -6 * Real.pi ≤ n ∧ n ≤ 12 * Real.pi }.finite.toFinset.card = 56 := 
by 
  sorry

end count_integers_in_range_l63_63637


namespace probability_sum_is_prime_l63_63397

theorem probability_sum_is_prime :
  (∃ (d1 d2 d3 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧
  (d1 + d2 + d3 = 3 ∨ d1 + d2 + d3 = 5 ∨ d1 + d2 + d3 = 7 ∨ d1 + d2 + d3 = 11 ∨ d1 + d2 + d3 = 13 ∨ d1 + d2 + d3 = 17)) →
  (∃ p, p = (64/216 : ℚ) ∧ p = (8/27 : ℚ)) :=
begin
  sorry
end

end probability_sum_is_prime_l63_63397


namespace cos_polynomials_l63_63507

noncomputable def cos_formula (n α : ℝ) : Prop :=
  cos ((n + 1) * α) = 2 * cos α * cos (n * α) - cos ((n - 1) * α)

theorem cos_polynomials (α : ℝ) :
  cos (3 * α) = 4 * (cos α) ^ 3 - 3 * cos α ∧
  cos (4 * α) = 8 * (cos α) ^ 4 - 8 * (cos α) ^ 2 + 1 := by
  sorry

end cos_polynomials_l63_63507


namespace limit_ratio_l63_63611

noncomputable def sequence_an : ℕ → ℤ
| 0       := 1
| 1       := 3
| (n + 2) := if sequence_an(n + 1) >= sequence_an(n) then sequence_an(n + 1) - 2^n else sequence_an(n + 1) + 2^n

def increasing_subsequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (2 * (n + 1) - 1) < a (2 * (n + 2) - 1)

def decreasing_subsequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (2 * (n + 2)) < a (2 * (n + 1))

def satisfies_conditions (a : ℕ → ℤ) : Prop :=
a 0 = 1 ∧ a 1 = 3 ∧ (∀ n : ℕ, |a (n + 2) - a (n + 1)| = 2^n) ∧
increasing_subsequence a ∧
decreasing_subsequence a

theorem limit_ratio (a : ℕ → ℤ) (h : satisfies_conditions a) :
    tendsto (λ n, (a (2 * n - 1) : ℚ) / (a (2 * n) : ℚ)) at_top (nhds (-1/2)) :=
sorry

end limit_ratio_l63_63611


namespace farmer_plough_rate_l63_63050

-- Define the problem statement and the required proof 

theorem farmer_plough_rate :
  ∀ (x y : ℕ),
  90 * x = 3780 ∧ y * (x + 2) = 3740 → y = 85 :=
by
  sorry

end farmer_plough_rate_l63_63050


namespace area_triangle_CDM_l63_63843

section
variables (A B C D M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M]

/-- Define the properties for the specific points in the Euclidean space. --/
def Point(C1 : C, A1 B1: A, M1 : M, D1 : D) : Prop :=
  ∃ (C1 A1 B1 M1 : A) (D1 : D),
    -- Condition 1: Right angle at vertex C in triangle ABC
    ∠ACB = π / 2 ∧
    -- Condition 2: Side lengths given
    dist A1 C1 = 8 ∧ dist B1 C1 = 15 ∧
    -- Condition 3: Distance from A and B to D and mid of AB
    dist A1 D1 = 17 ∧ dist B1 D1 = 17 ∧
    dist M1 A1 = dist M1 B1
    
-- Main theorem statement
theorem area_triangle_CDM :
  ∀ (A B C D M : Type),
  Point (A B C D M) →
  ∃ (m n p : ℕ),
    m = 1725 ∧
    n = 1 ∧
    p = 136 ∧
    m + n + p = 1862 :=
sorry
end

end area_triangle_CDM_l63_63843


namespace rectangle_area_same_width_l63_63921

theorem rectangle_area_same_width
  (square_area : ℝ) (area_eq : square_area = 36)
  (rect_width_eq_side : ℝ → ℝ → Prop) (width_eq : ∀ s, rect_width_eq_side s s)
  (rect_length_eq_3_times_width : ℝ → ℝ → Prop) (length_eq : ∀ w, rect_length_eq_3_times_width w (3 * w)) :
  (∃ s l w, s = 6 ∧ w = s ∧ l = 3 * w ∧ square_area = s * s ∧ rect_width_eq_side w s ∧ rect_length_eq_3_times_width w l ∧ w * l = 108) :=
by {
  sorry
}

end rectangle_area_same_width_l63_63921


namespace simplify_fraction_l63_63340

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : (2 + 4 * i) / (1 - 5 * i) = (-9 / 13) + (7 / 13) * i :=
by sorry

end simplify_fraction_l63_63340


namespace total_cost_of_items_l63_63778

theorem total_cost_of_items (cost_of_soda : ℕ) (cost_of_soup : ℕ) (cost_of_sandwich : ℕ) (total_cost : ℕ) 
  (h1 : cost_of_soda = 1)
  (h2 : cost_of_soup = 3 * cost_of_soda)
  (h3 : cost_of_sandwich = 3 * cost_of_soup) :
  total_cost = 3 * cost_of_soda + 2 * cost_of_soup + cost_of_sandwich :=
by
  unfold total_cost
  show 3 * 1 + 2 * (3 * 1) + (3 * (3 * 1)) = 18
  rfl

end total_cost_of_items_l63_63778


namespace sequence_difference_l63_63574

theorem sequence_difference : 
  (∃ (a : ℕ → ℤ) (S : ℕ → ℤ), 
    (∀ n : ℕ, S n = n^2 + 2 * n) ∧ 
    (∀ n : ℕ, n > 0 → a n = S n - S (n - 1) ) ∧ 
    (a 4 - a 2 = 4)) :=
by
  sorry

end sequence_difference_l63_63574


namespace nth_odd_positive_integer_is_199_l63_63855

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l63_63855


namespace middle_trapezoid_inscribed_circle_radius_l63_63068

noncomputable def radius_of_middle_trapezoid (R r : ℝ) : ℝ :=
  Real.sqrt (r * R)

theorem middle_trapezoid_inscribed_circle_radius (R r x : ℝ) 
  (h₀ : R > 0) (h₁ : r > 0) 
  (h₂ : ∃ (a b c : ℝ), a > b ∧ b > c ∧ trapezoid_exists_with_inscribed_circles a b c R r x)
  (h₃ : ∀ (a b c : ℝ), trapezoid_exists_with_inscribed_circles a b c R r x → R * r = x^2) : 
  x = radius_of_middle_trapezoid R r := 
sorry

end middle_trapezoid_inscribed_circle_radius_l63_63068


namespace cot_neg_45_l63_63133

-- Define the conditions
lemma cot_def (x : ℝ) : Real.cot x = 1 / Real.tan x := sorry
lemma tan_neg (x : ℝ) : Real.tan (-x) = -Real.tan x := sorry
lemma tan_45 : Real.tan (Real.pi / 4) = 1 := sorry

-- State the theorem to prove
theorem cot_neg_45 : Real.cot (-Real.pi / 4) = -1 :=
by
  apply cot_def
  apply tan_neg
  apply tan_45
  sorry

end cot_neg_45_l63_63133


namespace calculate_f_ff_f60_l63_63506

def f (N : ℝ) : ℝ := 0.3 * N + 2

theorem calculate_f_ff_f60 : f (f (f 60)) = 4.4 := by
  sorry

end calculate_f_ff_f60_l63_63506


namespace curve_to_cartesian_intersection_to_polar_l63_63210

theorem curve_to_cartesian (θ ρ : ℝ) : 
  (sin θ = √3 * ρ * cos θ ^ 2) → ∃ (x y : ℝ), y = √3 * x ^ 2 :=
by
  sorry

theorem intersection_to_polar (t : ℝ) : 
  (∀ t, (x = 1 + 1/2 * t ∧ y = √3 + √3 * t)) →
  (x y : ℝ) (y = √3 * x^2) → 
  ∃ ρ θ, (ρ, θ) = (2, π / 3) :=
by
  sorry

end curve_to_cartesian_intersection_to_polar_l63_63210


namespace team_a_first_half_points_eq_eight_l63_63107

-- Define the conditions
variables (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)

-- The points scored by Team A in the first half
def team_a_first_half := A

-- The points scored by Team B in the first half
def team_b_first_half := A / 2

-- The points scored by Team B in the second half
def team_b_second_half := A

-- The points scored by Team A in the second half
def team_a_second_half := A - 2

-- Define the total points scored by both teams
def total_points := team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half

-- Proof statement
theorem team_a_first_half_points_eq_eight : team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half = 26 → A = 8 :=
by
  intros h
  rw [team_a_first_half, team_b_first_half, team_a_second_half, team_b_second_half] at h
  sorry

end team_a_first_half_points_eq_eight_l63_63107


namespace mean_median_difference_is_3_point_5_l63_63681

-- Define the properties of the test scores
def percent_scored_65 := 0.15
def percent_scored_75 := 0.25
def percent_scored_85 := 0.40
def percent_scored_95 := 1 - (percent_scored_65 + percent_scored_75 + percent_scored_85) -- 0.20

-- Define the scores
def score_65 := 65
def score_75 := 75
def score_85 := 85
def score_95 := 95

-- Calculate the mean score
def mean_score :=
  percent_scored_65 * score_65 +
  percent_scored_75 * score_75 +
  percent_scored_85 * score_85 +
  percent_scored_95 * score_95

-- Assume the median score
def median_score := score_85

-- Define the difference between the median and mean scores
def difference := median_score - mean_score

-- The theorem stating the difference between the mean and median scores
theorem mean_median_difference_is_3_point_5 :
  difference = 3.5 :=
by
  -- Calculation here is omitted, proof to be provided manually.
  sorry

end mean_median_difference_is_3_point_5_l63_63681


namespace geometric_product_eq_64_l63_63682

noncomputable def geo_seq (n : ℕ) : ℕ → ℝ := sorry -- The general form of a geometric sequence

axiom pos_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h1 : a 0 = a1) (h2 : ∀ n, a (n + 1) = a n * r)
  (h_pos: ∀ n, a n > 0) : ∀ n, a n = a 0 * r ^ n

axiom root_condition {α : Type*} [field α] (a1 a19 : α) :
  a1 ≠ 0 ∧ a19 ≠ 0 ∧ a1 + a19 = 10 ∧ a1 * a19 = 16

theorem geometric_product_eq_64 (a : ℕ → ℝ) (r : ℝ) (a1 a19 : ℝ)
  (hg_seq : ∀ n, a n = a 0 * r ^ n)
  (h_prod : a1 * a19 = 16)
  (h1 : a 1 = a1)
  (h19 : a 19 = a19) :
  a 8 * a 10 * a 12 = 64 := by
  sorry

end geometric_product_eq_64_l63_63682


namespace equilateral_triangle_BDE_l63_63620

-- Definitions in Lean 4 corresponding to the conditions

variables {P : Type*} [euclidean_geometry P]

-- Given: Two perpendicular lines on a plane (which implies the existence of an orthogonal basis under Euclidean geometry)
variables (a b : line P) (O A B C D E : P)
-- Assumptions for given conditions
variable (x : ℝ)
variable (compass : ℝ → ℝ → ℝ)
variables [a_perp_b : a ⊥ b]
variables (O_on_a : O ∈ a) (O_on_b : O ∈ b)
variables (OA_eq_x : dist O A = x) (OB_eq_x : dist O B = x)
variables (AB_eq_xsqrt2 : dist A B = x * real.sqrt 2)
variables (OC_eq_AB : dist O C = dist A B)
variables (OC_on_a : O ∈ a → C ∈ a)
variables (BC_eq_xsqrt3 : dist B C = x * real.sqrt 3)
variables (OD_eq_BC : dist O D = dist B C)
variables (OD_on_a : O ∈ a → D ∈ a)
variables (OE_eq_OB : dist O E = dist O B)
variables (OE_on_b : O ∈ b → E ∈ b)
variables (BE_eq_2x : dist B E = 2 * x)
variables (BD_eq_2x : dist B D = 2 * x)
variables (DE_eq_2x : dist D E = 2 * x)

-- Objective: Prove that the triangle BDE is an equilateral triangle
theorem equilateral_triangle_BDE :
  dist B D = dist D E ∧ dist D E = dist B E ∧ dist B E = dist B D :=
sorry

end equilateral_triangle_BDE_l63_63620


namespace original_daily_production_quantity_l63_63402

theorem original_daily_production_quantity (x : ℕ) (h1 : 1.2 * x = 1.2 * 200) 
(h2 : (2200 / x) - (2400 / (1.2 * x)) = 1) : x = 200 :=
begin
  sorry
end

end original_daily_production_quantity_l63_63402


namespace ball_never_returns_l63_63947

-- Define the billiard table as a polygon with specified conditions
def billiard_table (P : Type) [polygon P] := 
  ∀ A ∈ vertices P, angle A = 1 ∧ all_angles_whole_numbers P

-- Define the conditions for the polygon and angles
class polygon (P : Type) :=
  (vertices : list P)
  (all_angles_whole_numbers : (∀ A ∈ vertices, ∃ n : ℕ, angle A = n))

-- Define the law of reflection
def law_of_reflection (A B : geom.Point) (α : ℝ) : Prop :=
  ∀ (θ : ℝ), θ = α -- Simplified form, as we can define reflections more detailed if needed.

-- Statement for the proof problem
theorem ball_never_returns (P : Type) [polygon P] (A : geom.Point) (α : ℝ) 
  (hα : α = 1) (h_angles : ∀ A ∈ vertices, ∃ n : ℕ, angle A = n) :
  ¬ ∃ t, (returns_to_vertex A t := 
  sorry -- Proof omitted

end ball_never_returns_l63_63947


namespace scaling_transformation_l63_63841

theorem scaling_transformation (P P' : ℝ × ℝ)
  (hP : P = (-2, 2))
  (hP' : P' = (-6, 1)) :
  let λ := 3
  let μ := (1 : ℝ) / 2
  in P' = (λ * (P.1), μ * (P.2)) :=
by
  sorry

end scaling_transformation_l63_63841


namespace square_binomial_formula_l63_63879

variable {x y : ℝ}

theorem square_binomial_formula :
  (2 * x + y) * (y - 2 * x) = y^2 - 4 * x^2 := 
  sorry

end square_binomial_formula_l63_63879


namespace find_number_l63_63394

theorem find_number (N : ℕ) (k : ℕ) (Q : ℕ)
  (h1 : N = 9 * k)
  (h2 : Q = 25 * 9 + 7)
  (h3 : N / 9 = Q) :
  N = 2088 :=
by
  sorry

end find_number_l63_63394


namespace unique_solution_l63_63966

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end unique_solution_l63_63966


namespace scaled_badge_height_proportional_l63_63301

theorem scaled_badge_height_proportional 
  (original_width original_height desired_width : ℕ) 
  (h₁ : original_width = 4) (h₂ : original_height = 3) (h₃ : desired_width = 12) :
  let scale_factor := desired_width / original_width in
  let new_height := original_height * scale_factor in
  new_height = 9 := by
sorry

end scaled_badge_height_proportional_l63_63301


namespace annual_interest_rate_is_8_percent_l63_63794

noncomputable def principal (A I : ℝ) : ℝ := A - I

noncomputable def annual_interest_rate (A P n t : ℝ) : ℝ :=
(A / P)^(1 / (n * t)) - 1

theorem annual_interest_rate_is_8_percent :
  let A := 19828.80
  let I := 2828.80
  let P := principal A I
  let n := 1
  let t := 2
  let r := annual_interest_rate A P n t
  r * 100 = 8 :=
by
  let A := 19828.80
  let I := 2828.80
  let P := principal A I
  let n := 1
  let t := 2
  let r := annual_interest_rate A P n t
  sorry

end annual_interest_rate_is_8_percent_l63_63794


namespace residue_neg_1234_mod_31_l63_63106

theorem residue_neg_1234_mod_31 : -1234 % 31 = 6 := 
by sorry

end residue_neg_1234_mod_31_l63_63106


namespace find_q_l63_63383

noncomputable def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h₁ : -p / 3 = q) (h₂ : q = 1 + p + q + 5) (h₃ : d = 5) : q = 2 :=
by
  sorry

end find_q_l63_63383


namespace intersection_area_of_rectangles_l63_63499

-- Conditions
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  bottom_left : Point
  top_right : Point

-- Given Rectangles
def rect1 : Rectangle := {
  bottom_left := {x := 0, y := 0},
  top_right := {x := 3, y := 2}
}

def rect2 : Rectangle := {
  bottom_left := {x := 1, y := 1},
  top_right := {x := 4, y := 3}
}

-- Function to calculate the intersection area
def intersection_area (r1 r2 : Rectangle) : ℝ :=
  let x_overlap := max 0 ((min r1.top_right.x r2.top_right.x) - (max r1.bottom_left.x r2.bottom_left.x))
  let y_overlap := max 0 ((min r1.top_right.y r2.top_right.y) - (max r1.bottom_left.y r2.bottom_left.y))
  x_overlap * y_overlap

-- Theorem to prove the intersection area
theorem intersection_area_of_rectangles : intersection_area rect1 rect2 = 2 := by
  sorry

end intersection_area_of_rectangles_l63_63499


namespace pipe_b_shut_off_time_l63_63307

-- Define the rates at which Pipe A and Pipe B fill the tank
def rate_pipe_a := (1 / 2 : ℝ)  -- Pipe A fills 0.5 tanks per hour
def rate_pipe_b := (1 : ℝ)      -- Pipe B fills 1 tank per hour

-- Define the combined rate when both pipes are open
def combined_rate := rate_pipe_a + rate_pipe_b  -- Combined rate is 1.5 tanks per hour

-- The time until overflow in hours
def time_until_overflow := (1 / 2 : ℝ)  -- 30 minutes is 0.5 hours

-- The fraction of the tank filled in the given time
def filled_with_both_pipes_open := combined_rate * time_until_overflow  -- Should be 0.75 tanks

-- The remaining part of the tank to be filled by Pipe A after Pipe B is shut off
def remaining_tank := 1 - filled_with_both_pipes_open  -- Should be 0.25 tanks

-- The time taken by Pipe A to fill the remaining tank
def time_for_pipe_a_to_fill_remaining := remaining_tank / rate_pipe_a  -- Should be 0.5 hours

theorem pipe_b_shut_off_time :
  time_for_pipe_a_to_fill_remaining = time_until_overflow :=
by
  sorry

end pipe_b_shut_off_time_l63_63307


namespace incorrect_option_D_l63_63364

def f (x : ℝ) : ℝ :=
  if x.is_rat then 1 else 0

theorem incorrect_option_D :
  ¬ (∀ x, f(f(x)) = f(x) → x = 1) :=
sorry

end incorrect_option_D_l63_63364


namespace geometric_sequence_general_term_T_n_formula_l63_63828

noncomputable def a (n : ℕ) : ℤ :=
  if n = 0 then 4 else (-2)^(n+1)

def b (n : ℕ) : ℤ :=
  nat.log 2 (abs (a n))

def T (n : ℕ) : ℚ :=
  (range n).sum (λ k, 1 / ((b k) * (b (k + 1))))

theorem geometric_sequence_general_term :
  ∀ n : ℕ, a n = (-2)^(n+1) := 
by
  sorry

theorem T_n_formula :
  ∀ n : ℕ, T n = n / (2 * (n + 2)) := 
by
  sorry

end geometric_sequence_general_term_T_n_formula_l63_63828


namespace max_inverse_sum_eccentricities_l63_63190

theorem max_inverse_sum_eccentricities 
  (e1 e2 b1 b2 : ℝ) 
  (a1 a2 c : ℝ) 
  (h1 : b1 = 3 * b2) 
  (h2 : a1^2 + 9 * a2^2 = 10 * c^2) 
  : ∃ θ φ : ℝ, ∀ (h_interior : 0 ≤ θ ∧ θ ≤ π ∧ 0 ≤ φ ∧ φ ≤ π), 
      let a1 := sqrt 10 * c * sin θ in 
      let a2 := (sqrt 10 / 3) * c * cos θ in
      ∃H: ¬ False, (1 / e1 + 1 / e2) ≤ (10 / 3). 
{
  sorry
}

end max_inverse_sum_eccentricities_l63_63190


namespace zoe_did_not_sell_bars_l63_63515

theorem zoe_did_not_sell_bars : 
  ∀ (cost_per_bar total_bars total_earnings bars_sold bars_not_sold : ℕ), 
    cost_per_bar = 6 →
    total_bars = 13 →
    total_earnings = 42 →
    bars_sold = total_earnings / cost_per_bar →
    bars_not_sold = total_bars - bars_sold →
    bars_not_sold = 6 :=
by 
  intros cost_per_bar total_bars total_earnings bars_sold bars_not_sold
  assume h1 : cost_per_bar = 6
  assume h2 : total_bars = 13
  assume h3 : total_earnings = 42
  assume h4 : bars_sold = total_earnings / cost_per_bar
  assume h5 : bars_not_sold = total_bars - bars_sold
  rw [h1, h2, h3, h4, h5]
  norm_num


end zoe_did_not_sell_bars_l63_63515


namespace factorial_simplification_l63_63002

theorem factorial_simplification :
  (13! - 12!) / 10! = 1584 := 
sorry

end factorial_simplification_l63_63002


namespace slices_served_during_dinner_l63_63456

theorem slices_served_during_dinner (slices_lunch slices_total slices_dinner : ℕ)
  (h1 : slices_lunch = 7)
  (h2 : slices_total = 12)
  (h3 : slices_dinner = slices_total - slices_lunch) :
  slices_dinner = 5 := 
by 
  sorry

end slices_served_during_dinner_l63_63456


namespace fraction_calculation_l63_63498

theorem fraction_calculation :
  ( (12^4 + 324) * (26^4 + 324) * (38^4 + 324) * (50^4 + 324) * (62^4 + 324)) /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)) =
  73.481 :=
by
  sorry

end fraction_calculation_l63_63498


namespace exists_four_numbers_satisfy_inequality_l63_63311

theorem exists_four_numbers_satisfy_inequality:
  ∀ (s : Fin 9 → ℝ), (Function.Injective s) →
  ∃ a b c d : Fin 9, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (s a * s c + s b * s d)^2 ≥ (9 / 10) * ((s a)^2 + (s b)^2) * ((s c)^2 + (s d)^2) :=
begin
  sorry,
end

end exists_four_numbers_satisfy_inequality_l63_63311


namespace domain_of_f_l63_63953

-- Function definition
def f (x : ℝ) : ℝ := (Real.logBase 2 (x - 1)) / (sqrt (2 - x))

-- Domain condition
def domain_condition (x : ℝ) : Prop := (1 < x) ∧ (x < 2)

-- Proof statement
theorem domain_of_f : ∀ x : ℝ, domain_condition x ↔ (f x = f x) :=
by
  intros x
  unfold domain_condition f
  have h1 : x - 1 > 0 ↔ 1 < x, by linarith
  have h2 : 2 - x > 0 ↔ x < 2, by linarith
  simp [h1, h2]
  sorry
  

end domain_of_f_l63_63953


namespace A_share_l63_63071

-- Definitions of each partner's investment conditions
def A_investment (P : ℝ) : ℝ := 0.05 * P * 3
def B_investment (P : ℝ) : ℝ := 0.07 * 1.5 * P * (32 / 12)
def C_investment (P : ℝ) : ℝ := 0.10 * 3 * P * (29 / 12)
def D_investment (P : ℝ) : ℝ := 0.12 * 2.5 * P * (27 / 12)
def total_gain (P : ℝ) : ℝ := A_investment P + B_investment P + C_investment P + D_investment P

-- Theorem to prove A's share in the gain
theorem A_share (P : ℝ) : total_gain P = 75000 → A_investment P = 5921.05 :=
by
  sorry

end A_share_l63_63071


namespace triangle_area_example_l63_63869

noncomputable def triangleArea (p1 p2 p3 : (ℝ × ℝ)) : ℝ :=
  Real.abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

theorem triangle_area_example :
  triangleArea (3, 2) (11, 6) (3, 8) = 24 :=
by
  sorry

end triangle_area_example_l63_63869


namespace product_of_fractions_l63_63490

theorem product_of_fractions :
  (∏ n in finset.range 503, (↑n + 2)/(↑n + 3)) = (2 : ℚ) / 505 := 
sorry

end product_of_fractions_l63_63490


namespace question_A_question_B_question_C_question_D_l63_63010

open Real
open Probability

theorem question_A : 
  let data := [1, 2, 4, 5, 6, 8, 9]
  let n := 4.2
  data.nth (5 - 1) = 6 := sorry

theorem question_B (X : ℕ → ℝ) (n : ℕ) (hX : X ∼ binomial n (1/3)) (hE : E (3*X + 1) = 6) : 
  n = 5 := sorry

theorem question_C (x y : ℝ) (b : ℝ) :
  let regression_eq := λ x : ℝ, b * x + 1.8
  average x = 2 →
  average y = 20 →
  b = 9.1 := sorry

theorem question_D (x y : ℝ) (chisq : ℝ) : 
  (chisq < some_threshold) → not (x and y are more related) := sorry

end question_A_question_B_question_C_question_D_l63_63010


namespace solve_system_of_inequalities_l63_63790

theorem solve_system_of_inequalities (x : ℝ) : 
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2) → -2 < x ∧ x < -1 :=
by {
  sorry
}

end solve_system_of_inequalities_l63_63790


namespace solve_inequality_l63_63285

variables {f : ℝ → ℝ} {f' : ℝ → ℝ}

-- conditions
axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom derivative_f : ∀ x : ℝ, f' x = D[λ y, f y] x
axiom condition_x_lt_0 : ∀ x : ℝ, x < 0 → x * f' x - f x > 0
axiom f_at_1 : f 1 = 0

-- statement of the proof problem
theorem solve_inequality :
  {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 1} = {x : ℝ | f x / x < 0} :=
sorry

end solve_inequality_l63_63285


namespace tangent_lines_through_point_tangent_to_circle_l63_63810

def point (x y : ℝ) := (x, y)
def circle (cx cy r : ℝ) (x y : ℝ) := (x - cx)^2 + (y - cy)^2 = r^2
def line (a b c : ℝ) (x y : ℝ) := a * x + b * y + c = 0
def is_tangent_to_circle (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) :=
  ∃ P : ℝ × ℝ, 
    circle P.1 P.2 ∧ 
    line P.1 P.2 ∧
    ∀ Q : ℝ × ℝ, 
      circle Q.1 Q.2 → 
      line Q.1 Q.2 → 
      P = Q

theorem tangent_lines_through_point_tangent_to_circle :
  ∀ (P : ℝ × ℝ), P = (-1, 6) →
  ∀ (circle : ℝ → ℝ → Prop), circle = circle (-3) 2 2 →
  ∃ l1 l2 : ℝ → ℝ → Prop,
    (l1 = line 3 (-4) 27 ∨ l1 = line 1 0 1) ∧ 
    (l2 = line 3 (-4) 27 ∨ l2 = line 1 0 1) ∧
    l1 ≠ l2 ∧
    is_tangent_to_circle l1 circle ∧ 
    is_tangent_to_circle l2 circle :=
sorry

end tangent_lines_through_point_tangent_to_circle_l63_63810


namespace fans_received_all_three_items_l63_63514

-- Define the conditions
def every_sixtieth (n : ℕ) : Prop := n % 60 = 0
def every_fortieth (n : ℕ) : Prop := n % 40 = 0
def every_ninetieth (n : ℕ) : Prop := n % 90 = 0
def stadium_capacity : ℕ := 3600

-- Prove the number of fans who received all three items is 10
theorem fans_received_all_three_items : {n : ℕ | n ≤ stadium_capacity ∧ every_sixtieth n ∧ every_fortieth n ∧ every_ninetieth n}.card = 10 := 
by
  sorry

end fans_received_all_three_items_l63_63514


namespace compare_abc_l63_63284

noncomputable def a := Real.exp (Real.sqrt 2)
noncomputable def b := 2 + Real.sqrt 2
noncomputable def c := Real.log (12 + 6 * Real.sqrt 2)

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end compare_abc_l63_63284


namespace find_m_equal_l63_63031

variables (A B C D X Y P Q R S : Point)
variables (m : ℝ)

def is_square (ABCD : Quadruple Point) (side_length : ℝ) : Prop := sorry
def points_on_sides (X BC Y CD : Point) (m : ℝ) : Prop := sorry
def extended_meets (A B D X P: Point) (A D B Y Q: Point) (A X D C R : Point) (A Y B C S : Point) : Prop := sorry
def collinear (P Q R S: Point) : Prop := sorry
def m_value (m : ℝ) : Prop := m = (3 - (Real.sqrt 5)) / 2

theorem find_m_equal {
  ∀ (ABCD : Quadruple Point), 
  is_square ABCD 1 → 
  points_on_sides X (side BC) Y (side CD) m → 
  extended_meets A B D X P A D B Y Q A X D C R A Y B C S →
  (collinear P Q R S ↔ m_value m) 
} := 
sorry

end find_m_equal_l63_63031


namespace complex_quadrant_l63_63237

theorem complex_quadrant (z : ℂ) (h : (2 - I) * z = 1 + I) : 
  0 < z.re ∧ 0 < z.im := 
by 
  -- Proof will be provided here 
  sorry

end complex_quadrant_l63_63237


namespace calculate_cyl_height_l63_63441

-- Define the parameters of the cone
def cone_radius := 14 -- in cm
def cone_height := 20 -- in cm

-- Define the parameters of the cylindrical container
def cyl_radius := 28 -- in cm

-- Define the 10% spill condition
def spill_percentage := 0.1

-- Define the volume formula for cone and the resulting remaining volume after spill
def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h
def remaining_volume (V : ℝ) : ℝ := (1 - spill_percentage) * V

-- Calculate the height of water in the cylindrical container after spill
def cyl_height_after_spill (r V : ℝ) : ℝ := V / (Real.pi * r^2)

-- Formalize the proof statement
theorem calculate_cyl_height : cyl_height_after_spill cyl_radius (remaining_volume (cone_volume cone_radius cone_height)) = 1.5 :=
by
  sorry

end calculate_cyl_height_l63_63441


namespace ellipse_foci_distance_l63_63374

-- Define the points F1 and F2
def F1 : (ℝ × ℝ) := (4, -5)
def F2 : (ℝ × ℝ) := (-6, 9)

-- Define the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the ellipse condition
def ellipse (x y : ℝ) : Prop :=
  real.sqrt((x-4)^2 + (y+5)^2) + real.sqrt((x+6)^2 + (y-9)^2) = 24

-- The theorem to prove
theorem ellipse_foci_distance :
  distance F1 F2 = 2 * real.sqrt 74 :=
by
  sorry

end ellipse_foci_distance_l63_63374


namespace ratio_y_x_correct_l63_63479

theorem ratio_y_x_correct (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) : 
  y / x = 25 / 16 :=
by
  have h1 : y = 1.25 * c := hy
  have h2 : x = 0.8 * c := hx
  have h3 : y / x = (1.25 * c) / (0.8 * c) := by rw [h1, h2]
  have h4 : y / x = 1.25 / 0.8 := by rw [mul_div_mul_right _ c (ne_of_gt zero_lt_one)]
  have h5 : 1.25 / 0.8 = 25 / 16 := by norm_num
  rw [h5] at h4
  exact h4

end ratio_y_x_correct_l63_63479


namespace integer_count_in_interval_l63_63627

theorem integer_count_in_interval : 
  let pi := Real.pi in
  let lower_bound := -6 * pi in
  let upper_bound := 12 * pi in
  ∃ (count : ℕ), count = 56 ∧ ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound ↔ (-18 ≤ n ∧ n ≤ 37) :=
by
  let pi := Real.pi
  let lower_bound := -6 * pi
  let upper_bound := 12 * pi
  use 56
  split
  · exact rfl
  · intro n
    split
    · intro h
      split
      · linarith
      · linarith
    · intro h
      split
      · linarith
      · linarith
  sorry

end integer_count_in_interval_l63_63627


namespace inequality_proof_l63_63169

variable (x y z : ℝ)

theorem inequality_proof (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  x * (1 - 2 * x) * (1 - 3 * x) + y * (1 - 2 * y) * (1 - 3 * y) + z * (1 - 2 * z) * (1 - 3 * z) ≥ 0 := 
sorry

end inequality_proof_l63_63169


namespace polar_equation_curve_distance_OP_range_l63_63699

section Problem

variables {θ α ρ λ : ℝ}

-- Condition: Parametric equations
def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (-1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

-- Polar coordinate equation
theorem polar_equation_curve (ρ β : ℝ) :
  (∃ θ : ℝ, parametric_curve θ = (ρ * Real.cos β, ρ * Real.sin β)) ↔
  ρ^2 + 2 * ρ * Real.cos β - 2 * ρ * Real.sin β - 2 = 0 :=
sorry

-- The polar coordinate equation of the line l
def polar_coordinate_line (α : ℝ) (β : ℝ) : Prop :=
  β = α ∧ α ∈ Set.Ico 0 π

-- The distance |OP| and range for λ
theorem distance_OP_range (α : ℝ) :
  (∀ ρ1 ρ2 : ℝ,
    (ρ1^2 + 2 * ρ1 * (Real.cos α - Real.sin α) - 2 = 0) ∧
    (ρ2^2 + 2 * ρ2 * (Real.cos α - Real.sin α) - 2 = 0) →
    |((ρ1 + ρ2) / 2 : ℝ)| ≤ λ) ↔
  λ ∈ Set.Ici (Real.sqrt 2) :=
sorry

end Problem

end polar_equation_curve_distance_OP_range_l63_63699


namespace mila_viewable_area_l63_63300
-- import the necessary libraries

-- Define the side length of the square and the visible radius
def side_length : ℝ := 4
def visibility_radius : ℝ := 1

-- Define the areas as outlined in the solution steps
def interior_area_unseen : ℝ := side_length^2 - (side_length - 2 * visibility_radius)^2
def exterior_rectangles_area : ℝ := 4 * (side_length * visibility_radius)
def corner_circles_area : ℝ := 4 * (π * visibility_radius^2 / 4)

-- Total viewable area
def total_viewable_area : ℝ := interior_area_unseen + exterior_rectangles_area + corner_circles_area

-- Define the target approximation
def rounded_viewable_area : ℝ := Real.floor (total_viewable_area + 0.5)

-- Propositional statement
theorem mila_viewable_area : rounded_viewable_area = 31 :=
by
  sorry

end mila_viewable_area_l63_63300


namespace sin_pi_over_3_l63_63890

theorem sin_pi_over_3 : Real.sin (π / 3) = sqrt 3 / 2 :=
by {
  -- This is where the proof would go
  sorry
}

end sin_pi_over_3_l63_63890


namespace not_homeostasis_C_l63_63412

-- Define what constitutes homeostasis
def homeostasis (condition : Prop) : Prop := condition

-- Define the conditions mentioned in the problem
def condition_A : Prop := ∀ (HCO₃⁻ HPO₄²⁻ : Prop), 7.35 ≤ blood_pH HCO₃⁻ HPO₄²⁻ ∧ blood_pH HCO₃⁻ HPO₄²⁻ ≤ 7.45
def condition_B : Prop := ∀ (phagocytes aging_cells : Prop), phagocytes eliminate aging_cells
def condition_C : Prop := ∀ (exercise intensity : Prop), myoglobin_content intensity = constant
def condition_D : Prop := ∀ (exercise sweat : Prop), osmotic_pressure sweat > base_level

-- Prove that option C is NOT an example of homeostasis given the other conditions are examples of homeostasis
theorem not_homeostasis_C : 
  homeostasis condition_A ∧ 
  homeostasis condition_B ∧ 
  homeostasis condition_D → 
  ¬ homeostasis condition_C := 
begin
  sorry
end

end not_homeostasis_C_l63_63412


namespace distance_traveled_eq_2400_l63_63807

-- Definitions of the conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 32
def revolutions_difference : ℕ := 5

-- Define the number of revolutions made by the back wheel
def revs_back (R : ℕ) := R

-- Define the number of revolutions made by the front wheel
def revs_front (R : ℕ) := R + revolutions_difference

-- Define the distance traveled by the back and front wheels
def distance_back (R : ℕ) : ℕ := revs_back R * circumference_back
def distance_front (R : ℕ) : ℕ := revs_front R * circumference_front

-- State the theorem without a proof (using sorry)
theorem distance_traveled_eq_2400 :
  ∃ R : ℕ, distance_back R = 2400 ∧ distance_back R = distance_front R :=
by {
  sorry
}

end distance_traveled_eq_2400_l63_63807


namespace smallest_positive_integer_l63_63409

theorem smallest_positive_integer :
  ∃ x : ℕ,
    x % 5 = 4 ∧
    x % 7 = 5 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    (∀ y : ℕ, (y % 5 = 4 ∧ y % 7 = 5 ∧ y % 11 = 9 ∧ y % 13 = 11) → y ≥ x) ∧ x = 999 :=
by
  sorry

end smallest_positive_integer_l63_63409


namespace odd_integers_count_l63_63152

theorem odd_integers_count : 
  ∃ n : ℕ, n = 1010 ∧ (∀ k, (1 ≤ k ∧ k ≤ 2019) → (k % 2 = 1) ↔ (∃ m, m < n ∧ k = 2 * m + 1)) :=
begin
  -- Proof goes here
  sorry
end

end odd_integers_count_l63_63152


namespace game_ends_in_draw_for_all_n_l63_63791

noncomputable def andrey_representation_count (n : ℕ) : ℕ := 
  -- The function to count Andrey's representation should be defined here
  sorry

noncomputable def petya_representation_count (n : ℕ) : ℕ := 
  -- The function to count Petya's representation should be defined here
  sorry

theorem game_ends_in_draw_for_all_n (n : ℕ) (h : 0 < n) : 
  andrey_representation_count n = petya_representation_count n :=
  sorry

end game_ends_in_draw_for_all_n_l63_63791


namespace rose_needs_more_money_l63_63331

def cost_paintbrush : ℝ := 2.40
def cost_paints : ℝ := 9.20
def cost_easel : ℝ := 6.50
def money_rose_has : ℝ := 7.10

theorem rose_needs_more_money : 
  cost_paintbrush + cost_paints + cost_easel - money_rose_has = 11.00 :=
begin
  sorry
end

end rose_needs_more_money_l63_63331


namespace hexagon_opposite_sides_diagonals_intersect_l63_63381

theorem hexagon_opposite_sides_diagonals_intersect 
  {A B C D E F : Point}
  (h1 : AB = DE) (h2 : AB ∥ DE)
  (h3 : BC = EF) (h4 : BC ∥ EF)
  (h5 : CD = FA) (h6 : CD ∥ FA) :
  ∃ O : Point, (line_segment A D).midpoint = O 
               ∧ (line_segment B E).midpoint = O 
               ∧ (line_segment C F).midpoint = O :=
sorry

end hexagon_opposite_sides_diagonals_intersect_l63_63381


namespace remaining_amount_is_1520_l63_63664

noncomputable def totalAmountToBePaid (deposit : ℝ) (depositRate : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  let fullPrice := deposit / depositRate
  let salesTax := taxRate * fullPrice
  let totalAdditionalExpenses := salesTax + processingFee
  (fullPrice - deposit) + totalAdditionalExpenses

theorem remaining_amount_is_1520 :
  totalAmountToBePaid 140 0.10 0.15 50 = 1520 := by
  sorry

end remaining_amount_is_1520_l63_63664


namespace train_cross_time_l63_63702

-- Given conditions
def train_length : ℝ := 350
def train_speed_kmph : ℝ := 144
def conversion_factor : ℝ := 1000 / 3600
def train_speed_mps : ℝ := train_speed_kmph * conversion_factor

-- Question and answer to be proven
def time_to_cross_pole : ℝ := train_length / train_speed_mps

theorem train_cross_time :
  time_to_cross_pole = 8.75 :=
by
  -- Definitions and known values
  have h1 : train_length = 350 := rfl
  have h2 : train_speed_kmph = 144 := rfl
  have h3 : conversion_factor = (1000 / 3600) := rfl
  have h4 : train_speed_mps = train_speed_kmph * conversion_factor := rfl

  -- Conversion of speed
  have h5 : train_speed_mps = 144 * (1000 / 3600) := by rw [h2, h3]
  have h6 : train_speed_mps = 40 := by norm_num [h5]

  -- Calculate time
  have h7 : time_to_cross_pole = train_length / train_speed_mps := rfl
  have h8 : time_to_cross_pole = 350 / 40 := by rw [h7, h6]
  have h9 : time_to_cross_pole = 8.75 := by norm_num [h8]

  -- Final proof
  exact h9

end train_cross_time_l63_63702


namespace four_triangles_cover_l63_63179

theorem four_triangles_cover 
  (s : ℝ) 
  (h_s : 0 < s ∧ s < 1)
  (cover_by_five : ∃ t1 t2 t3 t4 t5 : set (ℝ × ℝ), 
    (∀ t, t ∈ {t1, t2, t3, t4, t5} → ∃ center : ℝ × ℝ, is_equilateral_triangle t s ∧ t = centered_equilateral_triangle center s) ∧
    covers_large_triangle {t1, t2, t3, t4, t5} (equilateral_triangle 1)) : 
  ∃ t1 t2 t3 t4 : set (ℝ × ℝ), 
    (∀ t, t ∈ {t1, t2, t3, t4} → ∃ center : ℝ × ℝ, is_equilateral_triangle t s ∧ t = centered_equilateral_triangle center s) ∧
    covers_large_triangle {t1, t2, t3, t4} (equilateral_triangle 1) :=
sorry

end four_triangles_cover_l63_63179


namespace count_not_divisible_by_5_or_7_l63_63995

def isDivisible (n k : Nat) : Bool :=
  n % k = 0

def countNotDivisibleBy5or7 (n : Nat) : Nat :=
  (Nat.range n).filter (λ x => ¬ (isDivisible x 5 ∨ isDivisible x 7)).length

theorem count_not_divisible_by_5_or_7 (n : Nat) (h : n = 121) : countNotDivisibleBy5or7 n = 83 := 
by
  unfold countNotDivisibleBy5or7
  sorry

end count_not_divisible_by_5_or_7_l63_63995


namespace sum_of_coeffs_eq_negative_21_l63_63521

noncomputable def expand_and_sum_coeff (d : ℤ) : ℤ :=
  let expression := -(4 - d) * (d + 2 * (4 - d))
  let expanded_form := -d^2 + 12*d - 32
  let sum_of_coeffs := -1 + 12 - 32
  sum_of_coeffs

theorem sum_of_coeffs_eq_negative_21 (d : ℤ) : expand_and_sum_coeff d = -21 := by
  sorry

end sum_of_coeffs_eq_negative_21_l63_63521


namespace value_of_y_plus_z_l63_63219

theorem value_of_y_plus_z (z y : ℝ)
  (h : (3, -1, z).1 * (-2, -y, 1).1 + (3, -1, z).2 * (-2, -y, 1).2 + (3, -1, z).3 * (-2, -y, 1).3 = 0) :
  y + z = 6 :=
by
  -- here goes the proof
  sorry

end value_of_y_plus_z_l63_63219


namespace triangle_with_angle_ratio_is_right_l63_63241

theorem triangle_with_angle_ratio_is_right (
  (k : ℝ) 
  (h_ratio : (k + 2 * k + 3 * k = 180)) 
) : (30 ≤ k ∧ k ≤ 30) ∧ (2 * k = 2 * 30) ∧ (3 * k = 3 * 30) 
  ∧ (k = 30) ∧ (90 = 3 * k) :=
by {
  sorry
}

end triangle_with_angle_ratio_is_right_l63_63241


namespace max_expression_value_l63_63539

theorem max_expression_value
  (a b c x y z : ℝ)
  (h1 : 2 ≤ a ∧ a ≤ 3)
  (h2 : 2 ≤ b ∧ b ≤ 3)
  (h3 : 2 ≤ c ∧ c ≤ 3)
  (hx : x ∈ {a, b, c})
  (hy : y ∈ {a, b, c})
  (hz : z ∈ {a, b, c})
  (h_perm : \[x, y, z\] = [a, b, c])
  : (a / x) + ((a + b) / (x + y)) + ((a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_expression_value_l63_63539


namespace inclination_angle_l63_63815

open Real

theorem inclination_angle (a : ℝ) : 
  let k := sqrt 3 in
  atan k = π / 3 :=
by
  sorry

end inclination_angle_l63_63815


namespace max_expression_value_l63_63540

theorem max_expression_value
  (a b c x y z : ℝ)
  (h1 : 2 ≤ a ∧ a ≤ 3)
  (h2 : 2 ≤ b ∧ b ≤ 3)
  (h3 : 2 ≤ c ∧ c ≤ 3)
  (hx : x ∈ {a, b, c})
  (hy : y ∈ {a, b, c})
  (hz : z ∈ {a, b, c})
  (h_perm : \[x, y, z\] = [a, b, c])
  : (a / x) + ((a + b) / (x + y)) + ((a + b + c) / (x + y + z)) ≤ 15 / 4 :=
sorry

end max_expression_value_l63_63540


namespace y_48_y_divisible_by_24_l63_63156

theorem y_48_y_divisible_by_24 (y : ℕ) (hy : y < 10) : 
  (2 * y + 12) % 3 = 0 ∧ (48 + y) % 8 = 0 ↔ y = 6 :=
by {
  sorry,
}

end y_48_y_divisible_by_24_l63_63156


namespace leftmostThreeNonzeroDigits_l63_63579

-- Definitions based on the conditions
def numberOfRings : ℕ := 10
def numberOfFingers : ℕ := 5
def ringsToUse : ℕ := 7
def dividers : ℕ := 4

-- The mathematical operations needed
def binomial (n k : ℕ) : ℕ := Nat.choose n k
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the total number of arrangements
noncomputable def totalArrangements :=
  binomial numberOfRings ringsToUse * factorial ringsToUse * binomial (ringsToUse + dividers) dividers

-- Prove the leftmost three nonzero digits of the total number of arrangements is 200.
theorem leftmostThreeNonzeroDigits :
  (totalArrangements / 10^((totalArrangements.digits.length - 3))) = 200 := sorry

end leftmostThreeNonzeroDigits_l63_63579


namespace max_flight_height_l63_63449

def flight_height (x : ℝ) : ℝ :=
  - (1 / 50) * (x - 25) ^ 2 + 12

theorem max_flight_height : ∃ y_max : ℝ, y_max = 12 ∧ (∀ x : ℝ, flight_height x ≤ y_max) :=
by {
  sorry
}

end max_flight_height_l63_63449


namespace insert_plus_signs_to_make_correct_equation_l63_63416

theorem insert_plus_signs_to_make_correct_equation :
  ∃ (a b c : ℕ), a = 87 ∧ b = 899 ∧ c = 24 ∧ a + b + c = 1010 :=
by
  use 87, 899, 24
  split
  { refl }
  split
  { refl }
  split
  { refl }
  { sorry }

end insert_plus_signs_to_make_correct_equation_l63_63416


namespace partition_function_bounds_l63_63824

def partition_function (n : ℕ) : ℕ := 
  if n = 0 then 1 else (finset.powerset (finset.range n)).card

theorem partition_function_bounds (n : ℕ) (hn : 2 ≤ n) :
  2 ^ (⌊real.sqrt n⌋₊) < partition_function n ∧ partition_function n < n ^ (3 * ⌊real.sqrt n⌋₊) :=
sorry

end partition_function_bounds_l63_63824


namespace all_students_sleeping_l63_63434

theorem all_students_sleeping (α : Type) (students : set α) (S : α → set ℝ) (finite_students : students.finite)
  (at_least_two : 2 ≤ set.to_finset students.card)
  (each_sleep_once : ∀ s ∈ students, ∃ a b : ℝ, a < b ∧ S s = set.Icc a b)
  (overlap_condition : ∀ (s t ∈ students), ∃ t0 : ℝ, t0 ∈ S s ∧ t0 ∈ S t) :
  ∃ t : ℝ, ∀ s ∈ students, t ∈ S s :=
sorry

end all_students_sleeping_l63_63434


namespace tom_age_ratio_l63_63840

-- Define the conditions
variable (T N : ℕ) (ages_of_children_sum : ℕ)

-- Given conditions as definitions
def condition1 : Prop := T = ages_of_children_sum
def condition2 : Prop := (T - N) = 3 * (T - 4 * N)

-- The theorem statement to be proven
theorem tom_age_ratio : condition1 T ages_of_children_sum ∧ condition2 T N → T / N = 11 / 2 :=
by sorry

end tom_age_ratio_l63_63840


namespace number_of_triangles_l63_63716

-- Defining the side length of the square in centimeters
def side_length_square : ℝ := 10

-- Defining the width and height of the right triangle in centimeters
def width_triangle : ℝ := 1
def height_triangle : ℝ := 3

-- Calculating the area of the square
def area_square : ℝ := side_length_square ^ 2

-- Calculating the area of one right triangle
def area_triangle : ℝ := (width_triangle * height_triangle) / 2

-- Proving the number of right triangle-shaped pieces Jina can have
theorem number_of_triangles : ⌊ area_square / area_triangle ⌋ = 66 :=
by
  sorry

end number_of_triangles_l63_63716


namespace twelfth_term_geometric_sequence_l63_63873

-- Define the first term and common ratio
def a1 : Int := 5
def r : Int := -3

-- Define the formula for the nth term of the geometric sequence
def nth_term (n : Nat) : Int := a1 * r^(n-1)

-- The statement to be proved: that the twelfth term is -885735
theorem twelfth_term_geometric_sequence : nth_term 12 = -885735 := by
  sorry

end twelfth_term_geometric_sequence_l63_63873


namespace smallest_number_condition_l63_63888

theorem smallest_number_condition 
  (x : ℕ) 
  (h1 : ∃ k : ℕ, x - 6 = k * 12)
  (h2 : ∃ k : ℕ, x - 6 = k * 16)
  (h3 : ∃ k : ℕ, x - 6 = k * 18)
  (h4 : ∃ k : ℕ, x - 6 = k * 21)
  (h5 : ∃ k : ℕ, x - 6 = k * 28)
  (h6 : ∃ k : ℕ, x - 6 = k * 35)
  (h7 : ∃ k : ℕ, x - 6 = k * 39) 
  : x = 65526 :=
sorry

end smallest_number_condition_l63_63888


namespace count_irrational_numbers_l63_63378

-- Definitions of the numbers provided in the problem
noncomputable def neg_pi_div_2 : ℝ := -π / 2
noncomputable def sqrt_8 : ℝ := real.sqrt 8
def one_third : ℝ := 1 / 3
def abs_neg_3 : ℝ := abs (-3)
def sqrt_4 : ℝ := real.sqrt 4
def cbrt_neg_8 : ℝ := real.cbrt (-8)
noncomputable def sqrt_7 : ℝ := real.sqrt 7
noncomputable def weird_decimal : ℝ := 0.3131131113 -- conceptual placeholder for actual sequence definition

-- The target theorem statement
theorem count_irrational_numbers : 
  (set_of (λ x, irrational x) 
    ∈ {neg_pi_div_2, sqrt_8, one_third, abs_neg_3, sqrt_4, cbrt_neg_8, sqrt_7, weird_decimal}).card = 4 := 
sorry

end count_irrational_numbers_l63_63378


namespace smallest_sum_ABC_8_l63_63644

noncomputable def smallest_sum_ABC : Nat :=
  let cond1 := {A B : ℕ // A ≠ B ∧ A < 4 ∧ B < 4}
  ∃ A B b : Nat , (17 * A + 4 * B = 3 * b + 3) ∧ (b > 5) ∧ (A ≠ B) ∧ (A < 4) ∧ (B < 4) ∧ (A + B + b = 8)

theorem smallest_sum_ABC_8 : smallest_sum_ABC = 8 :=
by
  sorry

end smallest_sum_ABC_8_l63_63644


namespace correct_options_l63_63209

-- Definitions for lines l and n
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def line_n (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- The condition for lines to be parallel, equating the slopes
def parallel_lines (a : ℝ) : Prop := -(a + 2) / a = -(a - 2) / 3

-- The condition that line l passes through the point (1, -1)
def passes_through_point (a : ℝ) : Prop := line_l a 1 (-1)

-- The theorem statement
theorem correct_options (a : ℝ) :
  (parallel_lines a → a = 6 ∨ a = -1) ∧ (passes_through_point a) :=
by
  sorry

end correct_options_l63_63209


namespace solve_problem_l63_63580

theorem solve_problem (Δ q : ℝ) (h1 : 2 * Δ + q = 134) (h2 : 2 * (Δ + q) + q = 230) : Δ = 43 := by
  sorry

end solve_problem_l63_63580


namespace part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l63_63760

-- Definitions for part (1)
def P_X_1 : ℚ := 1 / 6
def P_X_2 : ℚ := 5 / 36
def P_X_3 : ℚ := 25 / 216
def P_X_4 : ℚ := 125 / 216
def E_X : ℚ := 671 / 216

theorem part1_prob_dist (X : ℚ) :
  (X = 1 → P_X_1 = 1 / 6) ∧
  (X = 2 → P_X_2 = 5 / 36) ∧
  (X = 3 → P_X_3 = 25 / 216) ∧
  (X = 4 → P_X_4 = 125 / 216) := 
by sorry

theorem part1_expectation :
  E_X = 671 / 216 :=
by sorry

-- Definition for part (2)
def P_A_wins_n_throws (n : ℕ) : ℚ := 1 / 6 * (5 / 6) ^ (2 * n - 2)

theorem part2_prob_A_wins_n_throws (n : ℕ) (hn : n ≥ 1) :
  P_A_wins_n_throws n = 1 / 6 * (5 / 6) ^ (2 * n - 2) :=
by sorry

end part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l63_63760


namespace symmetric_points_coords_l63_63464

section

variable (A B C : ℝ × ℝ)

/-- The symmetric point about the y-axis -/
def symmetric_about_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem symmetric_points_coords (A B C : ℝ × ℝ)
  (A' B' C' : ℝ × ℝ)
  (hA : symmetric_about_y A = (3, 2))
  (hB : symmetric_about_y B = (4, -3))
  (hC : symmetric_about_y C = (1, -1)) :
  (A', B', C') = (symmetric_about_y A, symmetric_about_y B, symmetric_about_y C) :=
begin
  sorry
end

end

end symmetric_points_coords_l63_63464


namespace find_integer_n_l63_63973

theorem find_integer_n :
  ∃ (n : ℤ), 10 ≤ n ∧ n ≤ 15 ∧ n % 7 = 6 :=
by
  use 13
  split
  · norm_num
  split
  · norm_num
  · norm_num
  · sorry

end find_integer_n_l63_63973


namespace arithmetic_sequence_ratio_l63_63986

-- Definitions of arithmetic sequences and their sums
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) :=
  (n * (a 1 + a n) / 2)

-- The problem statement
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ)
  (S T : ℕ → ℝ) (h_a : is_arithmetic_sequence a)
  (h_b : is_arithmetic_sequence b)
  (h_S : ∀ n : ℕ, S n = sum_first_n_terms a n)
  (h_T : ∀ n : ℕ, T n = sum_first_n_terms b n)
  (h_ratio : ∀ n : ℕ, S n / T n = 2 * n / (3 * n + 1)) :
  a 5 / b 5 = 9 / 14 :=
  sorry

end arithmetic_sequence_ratio_l63_63986


namespace hundredth_odd_positive_integer_l63_63868

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l63_63868


namespace rose_needs_more_money_l63_63325

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end rose_needs_more_money_l63_63325


namespace prime_factors_of_n_l63_63470

def n : ℕ := 400000001

def is_prime (p: ℕ) : Prop := Nat.Prime p

theorem prime_factors_of_n (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : n = p * q) : 
  (p = 19801 ∧ q = 20201) ∨ (p = 20201 ∧ q = 19801) :=
by
  sorry

end prime_factors_of_n_l63_63470


namespace find_x_l63_63119

theorem find_x (x : ℝ) (h₀ : ⌊x⌋ * x = 162) : x = 13.5 :=
sorry

end find_x_l63_63119


namespace find_q_l63_63643

theorem find_q (q : Nat) (h : 81 ^ 6 = 3 ^ q) : q = 24 :=
by
  sorry

end find_q_l63_63643


namespace find_integer_n_l63_63972

theorem find_integer_n :
  ∃ (n : ℤ), 10 ≤ n ∧ n ≤ 15 ∧ n % 7 = 6 :=
by
  use 13
  split
  · norm_num
  split
  · norm_num
  · norm_num
  · sorry

end find_integer_n_l63_63972


namespace problem_1_even_problem_1_odd_problem_2_even_problem_2_odd_l63_63062

def number_of_triangles_even (n : ℕ) : ℕ := (1 / 8) * n * (n + 2) * (2 * n + 1)
def number_of_triangles_odd (n : ℕ) : ℕ := (1 / 8) * (n + 1) * (2 * n ^ 2 + 3 * n - 1)
def number_of_rhombuses_even (n : ℕ) : ℕ := (1 / 8) * n * (n + 2) * (2 * n - 1)
def number_of_rhombuses_odd (n : ℕ) : ℕ := (1 / 8) * (n - 1) * (n + 1) * (2 * n + 3)

theorem problem_1_even (n : ℕ) (h : n % 2 = 0) : 
  -- Prove that the number of smaller regular triangles when n is even is correct
  number_of_triangles h = (1 / 8) * n * (n + 2) * (2 * n + 1) := sorry

theorem problem_1_odd (n : ℕ) (h : n % 2 = 1) : 
  -- Prove that the number of smaller regular triangles when n is odd is correct
  number_of_triangles h = (1 / 8) * (n + 1) * (2 * n ^ 2 + 3 * n - 1) := sorry 

theorem problem_2_even (n : ℕ) (h : n % 2 = 0) : 
  -- Prove that the number of rhombuses when n is even is correct
  number_of_rhombuses h = (1 / 8) * n * (n + 2) * (2 * n - 1) := sorry

theorem problem_2_odd (n : ℕ) (h : n % 2 = 1) : 
  -- Prove that the number of rhombuses when n is odd is correct
  number_of_rhombuses h = (1 / 8) * (n - 1) * (n + 1) * (2 * n + 3) := sorry

end problem_1_even_problem_1_odd_problem_2_even_problem_2_odd_l63_63062


namespace count_four_digit_numbers_ending_25_l63_63226

theorem count_four_digit_numbers_ending_25 : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 100]) → ∃ n : ℕ, n = 100 :=
by
  sorry

end count_four_digit_numbers_ending_25_l63_63226


namespace ranking_exists_l63_63082

variable (n : ℕ)
variable (judges : list (list ℕ)) -- List of lists, each list is a permutation of participants

-- Condition on the rankings by judges
def valid_ranking (judges : list (list ℕ)) :=
  ∀ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C →
  ¬ (∃ j1 j2 j3 : ℕ, j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3 ∧
   (judges.nth j1).bind (λ x, x.indexOf? A) < 
   (judges.nth j1).bind (λ x, x.indexOf? B) ∧
   (judges.nth j2).bind (λ x, x.indexOf? B) < 
   (judges.nth j2).bind (λ x, x.indexOf? C) ∧
   (judges.nth j3).bind (λ x, x.indexOf? C) < 
   (judges.nth j3).bind (λ x, x.indexOf? A))

-- Statement of the proof problem
theorem ranking_exists (n : ℕ) (judges : list (list ℕ)) (h : valid_ranking judges) : 
  ∃ ranking : list ℕ, ∀ A B : ℕ, A ≠ B → ranking.indexOf A < ranking.indexOf B →
  (list.count (λ j, (judges.nth j).bind (λ x, x.indexOf? A) < (judges.nth j).bind (λ x, x.indexOf? B)) judges) ≥ 51 :=
sorry

end ranking_exists_l63_63082


namespace find_b_minus_a_l63_63191

theorem find_b_minus_a (a b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x ^ 2 + b * x + Real.log x) 
  (h_tangent : ∀ x, f 1 = 2 ∧ Deriv.deriv f 1 = 4) : b - a = 0 :=
by
  sorry

end find_b_minus_a_l63_63191


namespace c_share_profit_l63_63882

-- Declare the investments and conditions
variables (x : ℝ) (profit : ℝ)

-- Given conditions
def investment_B := x
def investment_A := 3 * x
def investment_C := (3 * x) * (3/2)

-- Define the ratios
def total_parts := (3 * x) + x + (9/2) * x

-- Given total profit
def total_profit := 66000

-- C's share of the profit
def c_share := (9 / 17) * total_profit

-- Main theorem to prove C's share of the profit
theorem c_share_profit : c_share ≈ 34941.18 :=
by sorry

end c_share_profit_l63_63882


namespace total_cost_sean_bought_l63_63770

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end total_cost_sean_bought_l63_63770


namespace pyramid_volume_l63_63358

noncomputable section

-- Defining the volume of the pyramid under given conditions
def volume_of_pyramid (c : ℝ) : ℝ :=
  (sqrt 3 / 48) * c^3

-- Lean statement to prove the theorem
theorem pyramid_volume (c : ℝ) (h0 : 0 < c) :
  ∃ V, V = volume_of_pyramid c :=
by
  use (sqrt 3 / 48) * c^3
  sorry

end pyramid_volume_l63_63358


namespace giant_spider_weight_ratio_l63_63447

theorem giant_spider_weight_ratio 
    (W_previous : ℝ)
    (A_leg : ℝ)
    (P : ℝ)
    (n : ℕ)
    (W_previous_eq : W_previous = 6.4)
    (A_leg_eq : A_leg = 0.5)
    (P_eq : P = 4)
    (n_eq : n = 8):
    (P * A_leg * n) / W_previous = 2.5 := by
  sorry

end giant_spider_weight_ratio_l63_63447


namespace max_reflections_l63_63848

theorem max_reflections (α : ℝ) (h1 : 0 < α) (h2 : α < 180) : 
  ∃ m : ℤ, m = - int.floor (- 180 / α) :=
by
  sorry

end max_reflections_l63_63848


namespace value_of_b_l63_63875

theorem value_of_b (b : ℚ) : 
  (∀ x : ℚ, 5 * (3 * x - b) = 3 * (5 * x - 9)) ↔ b = 27 / 5 :=
begin
  sorry
end

end value_of_b_l63_63875


namespace max_sum_of_16_ages_at_least_k_l63_63117

theorem max_sum_of_16_ages_at_least_k (ages : Fin 50 → ℕ) (h_distinct : Function.Injective ages) (h_sum : (∑ i, ages i) = 1555) :
  ∃ (k : ℕ), k = 776 ∧ ∃ (S : Finset (Fin 50)), S.card = 16 ∧ (∑ i in S, ages i) ≥ k :=
by {
  -- This is the definition of the theorem, proof is omitted
  sorry
}

end max_sum_of_16_ages_at_least_k_l63_63117


namespace swimming_contest_outcomes_l63_63935

theorem swimming_contest_outcomes : 
  let participants := ["Arthur", "Ben", "Charles", "Devin", "Eli", "Frank"] in
  (list.permutations participants).count (λ p, p.take 3.length == participants.length) = 120 :=
by
  sorry

end swimming_contest_outcomes_l63_63935


namespace num_integers_in_range_l63_63631

theorem num_integers_in_range : 
  let π_approx := 3.14 in
  let lower_bound := -6 * π_approx in
  let upper_bound := 12 * π_approx in
  let n_start := Int.ceil lower_bound in
  let n_end := Int.floor upper_bound in
  (n_end - n_start + 1) = 56 := 
by
  sorry

end num_integers_in_range_l63_63631


namespace simplify_and_evaluate_expression_l63_63787

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1))

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2 + Real.sqrt 3) :
  given_expression a = (2 * Real.sqrt 3 + 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l63_63787


namespace value_set_of_x_l63_63746

def f (x : ℝ) : ℝ := |2 * x - 1|

def g (a : ℝ) : ℝ := |a + 1| - |2 * a - 1| / |a|

theorem value_set_of_x :
  (∀ a : ℝ, a ≠ 0 → f x ≥ g a) → (x ∈ set.Iic (-1) ∨ x ∈ set.Ici 2) :=
by
  sorry

end value_set_of_x_l63_63746


namespace inclination_angle_is_sixty_degrees_l63_63976

-- Define the given line equation and its slope
def line_equation := ∀ x: ℝ, y: ℝ, y = sqrt(3) * x - 2

-- Define the angle of inclination
def inclination_angle (α: ℝ) := tan α = sqrt(3)

-- State the theorem to be proven: The angle of inclination for the line y = sqrt(3)x - 2 is 60 degrees
theorem inclination_angle_is_sixty_degrees : inclination_angle (real.atan (sqrt 3)) = (π / 3) :=
by 
  sorry

end inclination_angle_is_sixty_degrees_l63_63976


namespace maximum_value_of_A_l63_63982

theorem maximum_value_of_A {x y z : ℝ} 
  (h1 : 0 < x) (h2 : x ≤ 1) 
  (h3 : 0 < y) (h4 : y ≤ 1) 
  (h5 : 0 < z) (h6 : z ≤ 1) : 
  let A := (sqrt (8 * x^4 + y) + sqrt (8 * y^4 + z) + sqrt (8 * z^4 + x) - 3) / (x + y + z) in
  A ≤ 2 :=
by sorry

end maximum_value_of_A_l63_63982


namespace solve_trigonometric_equation_count_solutions_l63_63346

theorem solve_trigonometric_equation :
  ∀ x : ℝ, 2000 ≤ x ∧ x ≤ 3000 →
  2 * real.sqrt 2 * real.sin (real.pi * x / 4) ^ 3 = real.sin (real.pi / 4 * (1 + x)) →
  ∃! (n : ℤ), 500 ≤ n ∧ n ≤ 749 ∧ x = 1 + 4 * n :=
sorry

-- Count the unique solutions within the given range
theorem count_solutions :
  let num_solutions := (749 - 500 + 1 : ℤ) in
  num_solutions = 250 :=
by
  simp [Int.ofNat_sub, Int.add_one, Int.ofNat_one, Int.ofNat_add]
  linarith

end solve_trigonometric_equation_count_solutions_l63_63346


namespace trapezoid_perimeter_is_correct_l63_63688

-- Definitions of isosceles trapezoid and its properties
structure IsoscelesTrapezoid (A B C D O : Type) :=
(base1: ℝ) -- DC
(base2: ℝ) -- AB
(BO: ℝ)
(OD: ℝ)
(angle_ABD: ℝ)
(intersect: bool)

def isIsoscelesTrapezoid (t : IsoscelesTrapezoid) : Prop :=
  t.angle_ABD = 90 ∧ t.intersect = tt

def calcPerimeter (t : IsoscelesTrapezoid): ℝ :=
  2 * t.base2 + t.base1

noncomputable def AB := 3
noncomputable def BC {x:ℝ} : ℝ := 7 * x
noncomputable def AD {x:ℝ} : ℝ := 25 * x

theorem trapezoid_perimeter_is_correct :
  ∀(x:ℝ), isIsoscelesTrapezoid ⟨BC x, AD x, 7/8, 25/8, 90, tt⟩ →
  calcPerimeter ⟨BC x, AB, 7/8, 25/8, 90, tt⟩ = 62/5 :=
by
  intro x h
  sorry

end trapezoid_perimeter_is_correct_l63_63688


namespace mass_percentage_Cr_H2CrO4_correct_l63_63149

-- Define constants for atomic masses
def mass_H : ℝ := 1.01
def mass_Cr : ℝ := 52.00
def mass_O : ℝ := 16.00

-- Define the formula for calculating molar mass of H2CrO4
def molar_mass_H2CrO4 : ℝ :=
  (2 * mass_H) + (1 * mass_Cr) + (4 * mass_O)

-- Define the formula for calculating mass percentage of Cr in H2CrO4
def mass_percentage_Cr_in_H2CrO4 : ℝ :=
  (mass_Cr / molar_mass_H2CrO4) * 100

-- Theorem statement: mass percentage of Cr in H2CrO4 is 44.06%
theorem mass_percentage_Cr_H2CrO4_correct :
  mass_percentage_Cr_in_H2CrO4 = 44.06 := sorry

end mass_percentage_Cr_H2CrO4_correct_l63_63149


namespace ball_arrangement_l63_63759

open Classical

noncomputable def num_arrangements (A B C D : ℕ) (boxes : Finset ℕ) : ℕ :=
  if (A ≠ B ∧ A ∈ boxes ∧ B ∈ boxes ∧ C ∈ boxes ∧ D ∈ boxes) then 30 else 0

theorem ball_arrangement :
  ∃ boxes : Finset ℕ, boxes = {1, 2, 3} ∧
  ∃ (A B C D : ℕ), 
    A ≠ B ∧
    ∀ b ∈ boxes, b ∈ {A, B, C, D} ∧
    num_arrangements A B C D boxes = 30 :=
by
  sorry

end ball_arrangement_l63_63759


namespace coefficient_third_term_l63_63694

theorem coefficient_third_term (x : ℝ) : 
  ∀ (x ≠ 0), 
  let binom_exp := (x + (1 / (2 * Real.sqrt x))) ^ 8 in
  -- The coefficient of the third term is 7
  (∃ c : ℝ, c = 7 ∧ binom_exp = c * x ^ 5 + ...) :=
by {
  sorry
}

end coefficient_third_term_l63_63694


namespace number_of_sets_given_to_sister_l63_63714

-- Defining the total number of cards, sets given to his brother and friend, total cards given away,
-- number of cards per set, and expected answer for sets given to his sister.
def total_cards := 365
def sets_given_to_brother := 8
def sets_given_to_friend := 2
def total_cards_given_away := 195
def cards_per_set := 13
def sets_given_to_sister := 5

theorem number_of_sets_given_to_sister :
  sets_given_to_brother * cards_per_set + 
  sets_given_to_friend * cards_per_set + 
  sets_given_to_sister * cards_per_set = total_cards_given_away :=
by
  -- It skips the proof but ensures the statement is set up correctly.
  sorry

end number_of_sets_given_to_sister_l63_63714


namespace football_field_area_l63_63445

-- Define the conditions
def fertilizer_spread : ℕ := 1200
def area_partial : ℕ := 3600
def fertilizer_partial : ℕ := 400

-- Define the expected result
def area_total : ℕ := 10800

-- Theorem to prove
theorem football_field_area :
  (fertilizer_spread / (fertilizer_partial / area_partial)) = area_total :=
by sorry

end football_field_area_l63_63445


namespace fourth_individual_is_09_l63_63384

def population := (List.range 50).map (· + 1) -- Population from 1 to 50

def random_table : List ℕ :=
  [6667, 4067, 1464, 0571, 9586, 1105, 6509, 6876, 8320, 3790,
   5716, 0011, 6614, 9084, 4511, 7573, 8805, 9052, 2741, 1486]

def selected_individuals (table : List ℕ) (start : ℕ) (pop : List ℕ) : List ℕ :=
  let indices := List.drop start table
  let valid_indices := indices.filter (λ x => x <= 50 ∧ x ∈ pop)
  valid_indices.nodup.erase_duplicates

def solution : Nat :=
  (selected_individuals random_table 8 population).getD 3 0

theorem fourth_individual_is_09 : solution = 9 :=
  by
    sorry

end fourth_individual_is_09_l63_63384


namespace triple_nested_application_l63_63294

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2 * n + 3

theorem triple_nested_application : g (g (g 3)) = 49 := by
  sorry

end triple_nested_application_l63_63294


namespace p_as_percentage_of_x_l63_63797

-- Given conditions
variables (x y z w t u p : ℝ)
variables (h1 : 0.37 * z = 0.84 * y)
variables (h2 : y = 0.62 * x)
variables (h3 : 0.47 * w = 0.73 * z)
variables (h4 : w = t - u)
variables (h5 : u = 0.25 * t)
variables (h6 : p = z + t + u)

-- Prove that p is 505.675% of x
theorem p_as_percentage_of_x : p = 5.05675 * x := by
  sorry

end p_as_percentage_of_x_l63_63797


namespace function_range_l63_63387

def function_defined (x : ℝ) : Prop := x ≠ 5

theorem function_range (x : ℝ) : x ≠ 5 → function_defined x :=
by
  intro h
  exact h

end function_range_l63_63387


namespace solve_expr_l63_63583

noncomputable def given_expr (θ : ℝ) : ℝ :=
  (cos (3 * real.pi + θ) / (cos θ * (cos (real.pi + θ) - 1))) + 
  (cos (θ - 4 * real.pi) / (cos (θ + 2 * real.pi) * cos (3 * real.pi + θ) + cos (-θ)))

theorem solve_expr (θ : ℝ) (h : sin (3 * real.pi + θ) = 1 / 2) : given_expr θ = 8 :=
sorry

end solve_expr_l63_63583


namespace find_a_l63_63616

open Set

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a (a : ℝ) :
  ∅ ⊂ (A a ∩ B) ∧ A a ∩ C = ∅ → a = -2 :=
by
  sorry

end find_a_l63_63616


namespace cosine_identity_l63_63511

theorem cosine_identity : cos (-20 * π / 3) = -1 / 2 :=
by
  sorry

end cosine_identity_l63_63511


namespace find_min_avg_comprehensive_cost_l63_63463

def avg_comprehensive_cost (x : ℕ) : ℝ := 
  520 + 50 * (x : ℝ) + 12800 / (x : ℝ)

def min_cost_floors : ℕ := 16

def min_avg_comprehensive_cost : ℝ := 2120

theorem find_min_avg_comprehensive_cost :
  (∀ x : ℕ, x ≥ 12 -> avg_comprehensive_cost min_cost_floors ≤ avg_comprehensive_cost x) ∧
  avg_comprehensive_cost min_cost_floors = min_avg_comprehensive_cost :=
by sorry

end find_min_avg_comprehensive_cost_l63_63463


namespace number_13_on_top_after_folds_l63_63038

/-
A 5x5 grid of numbers from 1 to 25 with the following sequence of folds:
1. Fold along the diagonal from bottom-left to top-right
2. Fold the left half over the right half
3. Fold the top half over the bottom half
4. Fold the bottom half over the top half
Prove that the number 13 ends up on top after all folds.
-/

def grid := (⟨ 5, 5 ⟩ : Nat × Nat)

def initial_grid : ℕ → ℕ := λ n => if 1 ≤ n ∧ n ≤ 25 then n else 0

def fold_diagonal (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 1 fold

def fold_left_over_right (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 2 fold

def fold_top_over_bottom (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 3 fold

def fold_bottom_over_top (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 4 fold

theorem number_13_on_top_after_folds : (fold_bottom_over_top (fold_top_over_bottom (fold_left_over_right (fold_diagonal initial_grid)))) 13 = 13 :=
by {
  sorry
}

end number_13_on_top_after_folds_l63_63038


namespace part1_part2_l63_63215

-- Define set A for a given a
def setA (a : ℝ) : set ℝ := {x | x^2 - a*x + a - 1 ≤ 0}

-- Part (1): Prove that for a = 5, A = { x | 1 ≤ x ≤ 4 }
theorem part1 : setA 5 = {x | 1 ≤ x ∧ x ≤ 4} :=
sorry

-- Define set B
def setB : set ℝ := {x | x = 2 ∨ x = 3}

-- Part (2): Prove that if B ⊆ A, then a ≥ 4
theorem part2 (a : ℝ) (h : setB ⊆ setA a) : 4 ≤ a :=
sorry

end part1_part2_l63_63215


namespace find_f_at_1_l63_63365

theorem find_f_at_1 (f : ℝ → ℝ)
  (h1 : ∀ x, 0 < x → ∀ x, differentiable ℝ f)
  (h2 : ∀ x, 0 < x ∧ x ≠ 1 → (2 * f x + x * (deriv f x)) / (x - 1) > 0)
  (h3 : deriv f 1 = -3 / 4) :
  f 1 = 3 / 8 :=
sorry

end find_f_at_1_l63_63365


namespace slope_range_l63_63817

theorem slope_range (k : ℝ) : 
  (∃ (x : ℝ), ∀ (y : ℝ), y = k * (x - 1) + 1) ∧ (0 < 1 - k ∧ 1 - k < 2) → (-1 < k ∧ k < 1) :=
by
  sorry

end slope_range_l63_63817


namespace gcd_of_consecutive_digit_sums_l63_63831

theorem gcd_of_consecutive_digit_sums :
  ∀ x y z : ℕ, x + 1 = y → y + 1 = z → gcd (101 * (x + z) + 10 * y) 212 = 212 :=
by
  sorry

end gcd_of_consecutive_digit_sums_l63_63831


namespace solve_for_x_l63_63232

theorem solve_for_x (x : ℝ) (h : (10 + sqrt x) ^ (1 / 2) = 4) : x = 36 :=
by
  sorry

end solve_for_x_l63_63232


namespace nancy_weight_l63_63158

variable (w : ℝ) -- Nancy's weight in pounds
variable (water_intake : ℝ) -- Nancy's daily water intake in pounds

-- Definition for the conditions
def daily_water_intake (w : ℝ) : ℝ := 0.60 * w
-- Nancy's condition of daily water intake
def intake_condition : Prop := daily_water_intake w = 54
-- Nancy's actual weight is expected to be 90 pounds
def expected_weight : Prop := w = 90

-- The theorem to prove the equivalence
theorem nancy_weight (h : intake_condition) : expected_weight := 
by
  sorry

end nancy_weight_l63_63158


namespace coefficient_third_term_l63_63695

theorem coefficient_third_term (x : ℝ) : 
  ∀ (x ≠ 0), 
  let binom_exp := (x + (1 / (2 * Real.sqrt x))) ^ 8 in
  -- The coefficient of the third term is 7
  (∃ c : ℝ, c = 7 ∧ binom_exp = c * x ^ 5 + ...) :=
by {
  sorry
}

end coefficient_third_term_l63_63695


namespace ace_then_king_probability_l63_63847

theorem ace_then_king_probability :
  let total_cards := 52
  let aces := 4
  let kings := 4
  let first_ace_prob := (aces : ℚ) / (total_cards : ℚ)
  let second_king_given_ace_prob := (kings : ℚ) / (total_cards - 1 : ℚ)
  (first_ace_prob * second_king_given_ace_prob = (4 : ℚ) / 663) :=
by
  let total_cards := 52
  let aces := 4
  let kings := 4
  let first_ace_prob := (aces : ℚ) / (total_cards : ℚ)
  let second_king_given_ace_prob := (kings : ℚ) / (total_cards - 1 : ℚ)
  exact (first_ace_prob * second_king_given_ace_prob = (4 : ℚ) / 663)
  sorry

end ace_then_king_probability_l63_63847


namespace quinary_country_50_cities_has_125_air_lines_no_quinary_country_with_46_air_lines_l63_63442

/-- A country is called a "quinary country" if each city in it is connected by air lines with exactly five other cities (there are no international flights). -/
def is_quinary_country (n : ℕ) (connections : ℕ) : Prop :=
  connections = (n * 5) / 2

theorem quinary_country_50_cities_has_125_air_lines :
  is_quinary_country 50 125 :=
by
  unfold is_quinary_country
  norm_num
  sorry

theorem no_quinary_country_with_46_air_lines :
  ¬ ∃ n, is_quinary_country n 46 :=
by
  intro h
  cases h with n hn
  unfold is_quinary_country at hn
  have : (5 * n) / 2 = 46, from hn,
  linarith only [this]
  sorry

end quinary_country_50_cities_has_125_air_lines_no_quinary_country_with_46_air_lines_l63_63442


namespace isosceles_right_triangle_l63_63584

theorem isosceles_right_triangle (a b c : ℝ) (h : sqrt(c^2 - a^2 - b^2) + abs(a - b) = 0) :
  (c^2 = a^2 + b^2) ∧ (a = b) :=
sorry

end isosceles_right_triangle_l63_63584


namespace find_a_find_min_value_l63_63204

def f (a x : ℝ) : ℝ := x^3 - 3 * a * x - 1

theorem find_a :
  (∃ a : ℝ, ∃ x : ℝ, x = -1 ∧ f a x = 0) ↔ a = 1 :=
begin
  sorry
end

theorem find_min_value :
  (∀ x, x ∈ set.Icc (-2 : ℝ) 1 → f 1 x ≥ -3 ) ∧ 
  (∃ x ∈ set.Icc (-2 : ℝ) 1, f 1 x = -3) :=
begin
  sorry
end

end find_a_find_min_value_l63_63204


namespace calculate_N_O_N_O_N_O_2_l63_63727

def N (x : ℝ) := 3 * Real.sqrt x
def O (x : ℝ) := x^2

theorem calculate_N_O_N_O_N_O_2 :
  N (O (N (O (N (O 2))))) = 54 := by
  sorry

end calculate_N_O_N_O_N_O_2_l63_63727


namespace divisible_by_900_l63_63312

theorem divisible_by_900 (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) := 
by 
  sorry

end divisible_by_900_l63_63312


namespace find_ff_neg3_l63_63367

def f (x : ℝ) : ℝ :=
  if x > 0 then 1 / Real.sqrt x else x^2

theorem find_ff_neg3 : f (f (-3)) = 1 / 3 :=
by
  sorry

end find_ff_neg3_l63_63367


namespace rotate_point_to_plane_l63_63395

-- Consider the mathematically equivalent problem statement
variables {P : Type*} {π : Type*} {t : Type*}

-- Defining a plane π
def plane (π : Type*) := π

-- Defining a point P that is not on the plane π
def point_off_plane (P : Type*) (π : Type*) := P ↔ ¬ π

-- Defining an axis t perpendicular to the plane π
def axis_perpendicular (t : Type*) (π : Type*) := t ⟂ π

-- The theorem stating that we can rotate point P to lie in plane π using axis t
theorem rotate_point_to_plane (P : Type*) (π : Type*) (t : Type*)
    (h1 : plane π)
    (h2 : point_off_plane P π)
    (h3 : axis_perpendicular t π) :
    ∃ P', P' ∈ π ∧ (P' = rotate P t where rotate is a preset rotation function) := 
sorry

end rotate_point_to_plane_l63_63395


namespace possible_remainder_degrees_l63_63876

open Polynomial

noncomputable def divisor := (C 2) * (X ^ 6) - (C 1) * (X ^ 4) + (C 3) * (X ^ 2) - (C 5)

theorem possible_remainder_degrees (p r : Polynomial ℤ) (h : p = q * divisor + r) :
  degree r < degree divisor :=
sorry

end possible_remainder_degrees_l63_63876


namespace range_of_f_l63_63733

def f (x : ℝ) : ℝ := (Real.sin x)^4 + (Real.cos x)^2 + (Real.tan x)^2

theorem range_of_f : ∀ x : ℝ, 1 ≤ f x := by
  sorry

end range_of_f_l63_63733


namespace no_distinct_pairs_l63_63955

theorem no_distinct_pairs (x y : ℕ) (h1 : 0 < x) (h2 : x < y) (h3 : real.sqrt 1452 = real.sqrt x + real.sqrt y) : false :=
sorry

end no_distinct_pairs_l63_63955


namespace sum_of_multiples_l63_63386

-- Define the three consecutive multiples of 5
def mult1 (x : ℝ) : ℝ := 5 * x - 5
def mult2 (x : ℝ) : ℝ := 5 * x
def mult3 (x : ℝ) : ℝ := 5 * x + 5

-- Define the product of the three multiples
def product (x : ℝ) : ℝ := (mult1 x) * (mult2 x) * (mult3 x)

-- Define the sum of the three multiples
def sum (x : ℝ) : ℝ := (mult1 x) + (mult2 x) + (mult3 x)

-- Define the condition based on the problem statement
noncomputable def problem_condition (x : ℝ) : Prop := product x = 30 * sum x

theorem sum_of_multiples : ∃ x, problem_condition x ∧ sum x = 30 + 15 * Real.sqrt 27 :=
by
  sorry

end sum_of_multiples_l63_63386


namespace trajectory_of_point_P_l63_63452

theorem trajectory_of_point_P
  (P A B : ℝ²)
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 = 1)
  (h_tangents : P ≠ A ∧ P ≠ B)
  (h_angle : ∠(A, P, B) = 60) : 
  P.1^2 + P.2^2 = 4 :=
sorry

end trajectory_of_point_P_l63_63452


namespace matrix_implication_l63_63275

variables {n : Type*} [Fintype n] [DecidableEq n]
variables {A B C : Matrix n n ℝ} (hA : A.det ≠ 0)

theorem matrix_implication (h : (A - B) ⬝ C = B ⬝ A⁻¹) : 
  C ⬝ (A - B) = A⁻¹ ⬝ B :=
by 
  sorry

end matrix_implication_l63_63275


namespace smallest_lcm_value_theorem_l63_63651

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end smallest_lcm_value_theorem_l63_63651


namespace imaginary_part_of_i_minus_2_squared_l63_63814

theorem imaginary_part_of_i_minus_2_squared : 
  complex.im ((complex.i - 2) ^ 2) = -4 := by
sorry

end imaginary_part_of_i_minus_2_squared_l63_63814


namespace trig_proof_l63_63588

theorem trig_proof {x : ℝ} (h1 : π / 2 < x ∧ x < π) (h2 : tan x ^ 2 + 3 * tan x - 4 = 0) :
  (sin x + cos x) / (2 * sin x - cos x) = 1 / 3 := 
  sorry

end trig_proof_l63_63588


namespace triangle_formation_condition_l63_63763

variable {P Q R S : Type}
variable {a b c : ℝ} [LinearOrder ℝ]

-- Assume P, Q, R, S are distinct points on a straight line
variable (distinct : P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ P ≠ S)

-- Assume lengths of segments PQ, PR, PS
variable (hPQ : dist P Q = a)
variable (hPR : dist P R = b)
variable (hPS : dist P S = c)

-- Given condition b = a + c
variable (h_eq : b = a + c)

-- Theorem to prove
theorem triangle_formation_condition : c > 2 * a := by
  sorry

end triangle_formation_condition_l63_63763


namespace number_of_possible_orders_l63_63251

-- Definitions of the conditions
def number_of_questions : ℕ := 10

-- Main statement to prove
theorem number_of_possible_orders : 
  (∃ n : ℕ, n = 2 ^ (number_of_questions - 1) ∧ n = 512) :=
begin
  use 512,
  split,
  { refl, },
  { norm_num, },
end

end number_of_possible_orders_l63_63251


namespace annual_interest_rate_is_correct_l63_63792

noncomputable def find_annual_interest_rate
  (interest : ℝ) (total_amount : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  let principal := total_amount - interest in
  ((total_amount / principal) ^ (1 / (n * t)) - 1) * 100

theorem annual_interest_rate_is_correct :
  find_annual_interest_rate 2828.80 19828.80 1 2 = 8 :=
  by simp [find_annual_interest_rate]; sorry

end annual_interest_rate_is_correct_l63_63792


namespace find_20_paise_coins_l63_63018

theorem find_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7100) : x = 200 :=
by
  -- Given the conditions, we need to prove x = 200.
  -- Steps and proofs are omitted here.
  sorry

end find_20_paise_coins_l63_63018


namespace unique_solution_l63_63967

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end unique_solution_l63_63967


namespace solve_for_x_l63_63217

theorem solve_for_x (x y z : ℝ) 
  (h1 : x * y + 3 * x + 2 * y = 12) 
  (h2 : y * z + 5 * y + 3 * z = 15) 
  (h3 : x * z + 5 * x + 4 * z = 40) :
  x = 4 :=
by
  sorry

end solve_for_x_l63_63217


namespace smallest_k_bound_l63_63779

noncomputable def u (k : ℕ) : ℚ :=
nat.rec_on k (1 / 4) (λ n u_n, 3 * u_n - 3 * u_n^2)

def L : ℚ := 1 / 2

def epsilon : ℚ := 1 / 2 ^ 2000

theorem smallest_k_bound (k : ℕ) :
  (∀ m < k, |u m - L| > epsilon) → |u k - L| ≤ epsilon :=
sorry

end smallest_k_bound_l63_63779


namespace maximum_value_A_l63_63983

theorem maximum_value_A (x y z : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  ∃ A ≤ 2, A = (√(8 * x^4 + y) + √(8 * y^4 + z) + √(8 * z^4 + x) - 3) / (x + y + z) := 
sorry

end maximum_value_A_l63_63983


namespace fixed_point_exists_l63_63369

-- Defining the function f
def f (a x : ℝ) : ℝ := a * x - 3 + 3

-- Stating that there exists a fixed point (3, 3a)
theorem fixed_point_exists (a : ℝ) : ∃ y : ℝ, f a 3 = y :=
by
  use (3 * a)
  simp [f]
  sorry

end fixed_point_exists_l63_63369


namespace point_on_line_and_in_first_quadrant_l63_63761

theorem point_on_line_and_in_first_quadrant (x y : ℝ) (hline : y = -2 * x + 3) (hfirst_quadrant : x > 0 ∧ y > 0) :
    (x, y) = (1, 1) :=
by
  sorry

end point_on_line_and_in_first_quadrant_l63_63761


namespace proper_subsets_of_A_eq_7_l63_63615

noncomputable def U : Set ℝ := {-1, 0, 1, 2}
noncomputable def A : Set ℝ := {y | ∃ x ∈ U, y = Real.sqrt (x^2 + 1)}

theorem proper_subsets_of_A_eq_7 : (Finset.powerset A).card = 8 := by
  sorry

end proper_subsets_of_A_eq_7_l63_63615


namespace constant_expression_l63_63782

theorem constant_expression 
  (x y : ℝ) 
  (h₁ : x + y = 1) 
  (h₂ : x ≠ 1) 
  (h₃ : y ≠ 1) : 
  (x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3)) = 0 :=
by 
  sorry

end constant_expression_l63_63782


namespace cross_ratios_equal_l63_63618

-- Definitions of points and lines in general plane geometry.
noncomputable def Point := ℝ^2
noncomputable def Line := set Point

-- Definitions following the conditions mentioned in the problem description.
variables {S : Point → Line} {a b c d : Point}
variables {A B C D A' B' C' D' : Point}

-- Define the cross ratio (A B C D) as per the given formula.
def cross_ratio (A B C D : Point) : ℝ :=
  (dist A C / dist B C) / (dist A D / dist B D)

-- Assume that A, B, C, D, and A', B', C', D' all lie on the lines intersecting rays from point C.
axiom intersecting_rays (r : S a ∩ S b ∩ S c ∩ S d) :
  ∃ (l1 l2 : Line) (C : Point),
    {A, B, C, D}.subset l1 ∧ {A', B', C', D'}.subset l2

-- Prove that the cross ratios are equal.
theorem cross_ratios_equal :
  cross_ratio A B C D = cross_ratio A' B' C' D' :=
by sorry

end cross_ratios_equal_l63_63618


namespace even_number_of_rooks_on_black_squares_l63_63747

theorem even_number_of_rooks_on_black_squares
    (rooks : Finset (Fin 8 × Fin 8))  -- set of rook coordinates
    (non_attacking : ∀ (r1 r2 : Fin 8 × Fin 8), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 → (r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2))  -- non-attacking condition
    (h_rooks_count : rooks.card = 8) -- 8 rooks condition
    : (Finset.filter (λ rook, (rook.1 + rook.2) % 2 = 1) rooks).card % 2 = 0 := 
by
  sorry

end even_number_of_rooks_on_black_squares_l63_63747


namespace disembark_ways_l63_63894

theorem disembark_ways (passengers stops : ℕ) (h_passengers : passengers = 100) (h_stops : stops = 16) :
  (stops ^ passengers) = 16 ^ 100 :=
by
  rw [h_passengers, h_stops]
  sorry

end disembark_ways_l63_63894


namespace geometric_sequence_a3_l63_63698

theorem geometric_sequence_a3 :
  ∃ (q : ℝ), (a1 a5 : ℝ)
  (a1 = 1) (a5 = 81)
  (a5 = a1 * q^4) (q^2 = 9)
  (a3 = a1 * q^2)
  (a3 = 9) := by
{
  sorry
}

end geometric_sequence_a3_l63_63698


namespace total_cost_is_18_l63_63775

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_l63_63775


namespace retailer_cost_pants_l63_63058

theorem retailer_cost_pants :
  ∃ (C : ℝ), (C > 0) ∧
             ((1.04 * C) = 130) :=
begin
  use 125,
  split,
  { exact zero_lt_one.trans_le (show 0 < 125, by norm_num) },
  { norm_num }
end

end retailer_cost_pants_l63_63058


namespace factorial_sum_mod_p_l63_63989

theorem factorial_sum_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (odd_p : p % 2 = 1) : 
  (∑ i in (Finset.range p).map (Fin.val), i!) - ⌊(p - 1)! / Real.exp 1⌋ ≡ 0 [MOD p] :=
by
  sorry

end factorial_sum_mod_p_l63_63989


namespace boy_present_age_l63_63994

theorem boy_present_age : ∃ x : ℕ, (x + 4 = 2 * (x - 6)) ∧ x = 16 := by
  sorry

end boy_present_age_l63_63994


namespace number_of_oranges_l63_63678

def bananas : ℕ := 7
def apples : ℕ := 2 * bananas
def pears : ℕ := 4
def grapes : ℕ := apples / 2
def total_fruits : ℕ := 40

theorem number_of_oranges : total_fruits - (bananas + apples + pears + grapes) = 8 :=
by sorry

end number_of_oranges_l63_63678


namespace temperature_conversion_l63_63851

theorem temperature_conversion (F : ℝ) (C : ℝ) (hF122 : F = 122) (hC : C = (F - 32) * 5 / 9) : C = 50 :=
by
  rw [hF122] at hC
  simp at hC
  assumption

#print temperature_conversion

end temperature_conversion_l63_63851


namespace wall_print_costs_are_15_l63_63959

-- Define the cost of curtains, installation, total cost, and number of wall prints.
variable (cost_curtain : ℕ := 30)
variable (num_curtains : ℕ := 2)
variable (cost_installation : ℕ := 50)
variable (num_wall_prints : ℕ := 9)
variable (total_cost : ℕ := 245)

-- Define the total cost of curtains
def total_cost_curtains : ℕ := num_curtains * cost_curtain

-- Define the total fixed costs
def total_fixed_costs : ℕ := total_cost_curtains + cost_installation

-- Define the total cost of wall prints
def total_cost_wall_prints : ℕ := total_cost - total_fixed_costs

-- Define the cost per wall print
def cost_per_wall_print : ℕ := total_cost_wall_prints / num_wall_prints

-- Prove the cost per wall print is $15.00
theorem wall_print_costs_are_15 : cost_per_wall_print = 15 := by
  -- This is a placeholder for the proof
  sorry

end wall_print_costs_are_15_l63_63959


namespace campers_evening_l63_63034

-- Defining the conditions from the problem
def campers_morning : Nat := 33
def campers_afternoon : Nat := 34
def difference_afternoon_evening : Nat := 24

-- The main theorem statement
theorem campers_evening : (campers_afternoon = (campers_evening + difference_afternoon_evening)) → campers_evening = 10 :=
by
  sorry

end campers_evening_l63_63034


namespace unique_root_eqn_unique_root_comparison_l63_63361

theorem unique_root_eqn (x : ℝ) : 
(4^x + 10^x = 25^x) ↔ x = Real.logb (2 / 5) ((Real.sqrt 5 - 1) / 2) :=
by
  sorry

theorem unique_root_comparison (x : ℝ) :
(4^x + 10^x = 25^x) → 
0 < x ∧ x < 1 :=
by
  sorry

end unique_root_eqn_unique_root_comparison_l63_63361


namespace median_of_sequence_l63_63684

theorem median_of_sequence :
  let seq := list.finRange 301 >>= (λ n, list.repeat n n)
  let N := (300 * 301) / 2
  N = 45150 →
  let median_pos := N / 2
  (22575 ≤ median_pos ∧ median_pos + 1 ≤ 22578) →
  (seq[mid int.div_le 45150 2] = 212 ∧ seq[mid (45150 / 2 - 1)] = 212) :=
begin
  intro seq,
  intro N,
  intro hN,
  intro median_pos,
  intro h_pos,
  split,
  { sorry }, -- proof that the 22575th element is 212
  { sorry }  -- proof that the 22576th element is 212
end

end median_of_sequence_l63_63684


namespace difference_in_pay_l63_63752

-- Definitions for the problem conditions
def oula_pay_per_delivery : ℕ := 100
def oula_deliveries : ℕ := 96
def tona_multiplier : ℚ := 3 / 4

-- Statements derived from the conditions
def oula_earnings : ℕ := oula_deliveries * oula_pay_per_delivery

def tona_deliveries : ℚ := tona_multiplier * oula_deliveries
def tona_earnings : ℕ := ((tona_deliveries : ℕ) * oula_pay_per_delivery)

-- Theorem stating the difference in pay
theorem difference_in_pay : oula_earnings - tona_earnings = 2400 :=
by sorry

end difference_in_pay_l63_63752


namespace problem_l63_63567

noncomputable def f (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), 1 / (i + 1)

theorem problem (n : ℕ) (h_pos : 0 < n) :
  f (2 ^ n) > (n + 2 : ℚ) / 2 := by
  sorry

end problem_l63_63567


namespace even_function_symmetric_y_axis_l63_63076

theorem even_function_symmetric_y_axis (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = f (-x) := by
  sorry

end even_function_symmetric_y_axis_l63_63076


namespace trig_identity_sin_230_tan_170_l63_63961

theorem trig_identity_sin_230_tan_170 :
  sin (230 * real.pi / 180) * (1 - real.sqrt 3 * tan (170 * real.pi / 180)) = -1 :=
by sorry

end trig_identity_sin_230_tan_170_l63_63961


namespace explicit_f_and_intervals_no_zeros_condition_l63_63598

noncomputable def f (m : ℝ) (x : ℝ) := (m * x) / (Real.log x)

noncomputable def g (k : ℝ) (x : ℝ) := f (-1) x - (k * x^2) / (x - 1)

theorem explicit_f_and_intervals :
  (∃ m, ∀ x > 0, differentiable_at ℝ (f m) x ∧ m = -1 ∧ 
  (∀ x ∈ Icc 0 1, deriv (f m) x ≤ 0) ∧ 
  (∀ x ∈ Icc e ∞, deriv (f m) x ≤ 0) ↔ 
  ∃ m = -1) :=
sorry

theorem no_zeros_condition : 
  ∀ k : ℝ, 
  ¬ ∃ x, g k x = 0 ↔ k ≤ 0 ∨ k = 2 :=
sorry

end explicit_f_and_intervals_no_zeros_condition_l63_63598


namespace arithmetic_sequence_difference_l63_63870

def arithmetic_sequence (a d n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_difference :
  let a := 3
  let d := 7
  let a₁₀₀₀ := arithmetic_sequence a d 1000
  let a₁₀₀₃ := arithmetic_sequence a d 1003
  abs (a₁₀₀₃ - a₁₀₀₀) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l63_63870


namespace weekly_allowance_l63_63303

theorem weekly_allowance
  (video_game_cost : ℝ)
  (sales_tax_percentage : ℝ)
  (weeks_to_save : ℕ)
  (total_with_tax : ℝ)
  (total_savings : ℝ) :
  video_game_cost = 50 →
  sales_tax_percentage = 0.10 →
  weeks_to_save = 11 →
  total_with_tax = video_game_cost * (1 + sales_tax_percentage) →
  total_savings = weeks_to_save * (0.5 * total_savings) →
  total_savings = total_with_tax →
  total_savings = 55 :=
by
  intros
  sorry

end weekly_allowance_l63_63303


namespace exactly_three_odd_divisors_with_units_digit_three_l63_63160

-- Define the set of positive divisors of n
def divisors (n : ℕ) : set ℕ := { d | d > 0 ∧ n % d = 0 }

-- Define a predicate for having a units digit of 3
def units_digit_is_three (x : ℕ) : Prop := x % 10 = 3

-- Define a predicate for being odd
def is_odd (x : ℕ) : Prop := x % 2 = 1

-- Main theorem statement
theorem exactly_three_odd_divisors_with_units_digit_three (n : ℕ) (h : 0 < n) : 
  (divisors n).count (λ d, units_digit_is_three d ∧ is_odd d) = 3 := by
  sorry

end exactly_three_odd_divisors_with_units_digit_three_l63_63160


namespace points_A_B_D_G_concyclic_points_O1_O2_M_N_concyclic_l63_63178

variables {A B C H E F D G M O1 O2 N : Type}
variables [Preorder A] [Preorder B] [Preorder C] [Preorder H] [Preorder E]
          [Preorder F] [Preorder D] [Preorder G] [Preorder M] [Preorder O1]
          [Preorder O2] [Preorder N]
variables (triangle_ABC : abc_triangle A B C H)
variables (point_E_on_CH : E ∈ segment CH)
variables (F_on_extended_CH : extend_CH CH E F)
variables (CE_eq_HF : dist CE HF = dist HF CE)
variables (FD_perpendicular_BC : perp_to_BC FD D)
variables (EG_perpendicular_BH : perp_to_BH EG G)
variables (M_midpoint_CF : midpoint_CF M CF)
variables (O1_circumcenter_ABG : circumcenter O1 ABG)
variables (O2_circumcenter_BCH : circumcenter O2 BCH)
variables (N_other_intersection : other_intersection_circles O1 O2 N)

theorem points_A_B_D_G_concyclic :
  concyclic A B D G :=
sorry

theorem points_O1_O2_M_N_concyclic :
  concyclic O1 O2 M N :=
sorry

end points_A_B_D_G_concyclic_points_O1_O2_M_N_concyclic_l63_63178


namespace possible_values_for_x_l63_63051

theorem possible_values_for_x (x y z : Nat) (h_departure : x < 24) (h_arrival : y < 24) (h_travel : z < 24) :
  (let total_departure_time := x * 60 + y,
       total_arrival_time := y * 60 + z,
       total_travel_time := z * 60 + x in
   (total_arrival_time - total_departure_time = total_travel_time) → (x = 0 ∨ x = 12)) :=
sorry

end possible_values_for_x_l63_63051


namespace seating_capacity_l63_63481

theorem seating_capacity (n : ℕ) (m : ℕ) (h1 : n = 5) (h2 : m = 2) 
  (cond1 : ∀n, n = 1 → seats 1 = 6)
  (cond2 : ∀n, n = 2 → seats 2 = 10)
  (cond3 : ∀n, n = 3 → seats 3 = 14) :
  seats (n * m) = 80 :=
by
  sorry

end seating_capacity_l63_63481


namespace average_is_20_l63_63354

-- Define the numbers and the variable n
def a := 3
def b := 16
def c := 33
def n := 27
def d := n + 1

-- Define the sum of the numbers
def sum := a + b + c + d

-- Define the average as sum divided by 4
def average := sum / 4

-- Prove that the average is 20
theorem average_is_20 : average = 20 := by
  sorry

end average_is_20_l63_63354


namespace sum_XA_XB_XC_l63_63842

-- Definitions for the problem
def length_AB : ℝ := 12
def length_BC : ℝ := 15
def length_AC : ℝ := 13

def midpoint (A B : Point) : Point := Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

variable {A B C D E F X : Point}

-- Specific midpoints
def mid_D : Point := midpoint A B
def mid_E : Point := midpoint B C
def mid_F : Point := midpoint A C

-- Conditions from the problem
axiom D_midpoint : D = mid_D
axiom E_midpoint : E = mid_E
axiom F_midpoint : F = mid_F
axiom X_intersection : ∃! X ≠ E, is_intersection_of_circumcircles B D E C E F X 

-- Objective: Prove the sum of distances
theorem sum_XA_XB_XC : distance X A + distance X B + distance X C = 38.88 := 
sorry

end sum_XA_XB_XC_l63_63842


namespace cot_neg_45_l63_63128

-- Define the given conditions
def tan_neg_angle (x : ℝ) : Prop := ∀ θ : ℝ, tan (-θ) = -tan(θ)
def tan_45 : Prop := tan (45 * (π / 180)) = 1
def cot_def (x : ℝ) : Prop := ∀ θ : ℝ, cot(θ) = 1 / tan(θ)

-- Prove that cot(-45°) = -1 given the conditions
theorem cot_neg_45 : cot (-45 * (π / 180)) = -1 :=
by 
  have h1 := tan_neg_angle (-45 * (π / 180)),
  have h2 := tan_45,
  have h3 := cot_def (-45 * (π / 180)),
  sorry -- Proof steps skipped

end cot_neg_45_l63_63128


namespace b_seq_arithmetic_sum_reciprocal_S_Tn_less_than_three_quarters_l63_63213

-- Definitions 
def a_seq (n : ℕ) : ℝ
| 1     := 2
| (n+1) := 2 - (1 / a_seq n)

def b_seq (n : ℕ) : ℝ :=
1 / (a_seq n - 1)

def S (n : ℕ) : ℝ :=
(1 / 3) * (list.sum (list.range n).map b_seq)

def T (n : ℕ) : ℝ :=
list.sum (list.range n).map (λ i, (1 / 3 ^ (i + 1)) * b_seq (i + 1))

-- Statements to prove

-- 1. Prove that {b_n} is an arithmetic sequence
theorem b_seq_arithmetic (n : ℕ) : b_seq (n + 1) - b_seq n = 1 := sorry

-- 2. Find sum {1/S_i}, with S_i := (1 / 3) * sum of b_i
theorem sum_reciprocal_S (n : ℕ) : (list.sum (list.range n).map (λ i, 1 / S (i + 1))) = 6 * n / (n + 1) := sorry

-- 3. Prove that {T_n} < 3/4
theorem Tn_less_than_three_quarters (n : ℕ) : T n < 3 / 4 := sorry

end b_seq_arithmetic_sum_reciprocal_S_Tn_less_than_three_quarters_l63_63213


namespace rational_count_l63_63075

open Set

noncomputable def sqrt_2 : ℝ := real.sqrt 2
noncomputable def cbrt_3 : ℝ := real.cbrt 3
noncomputable def pi_value : ℝ := real.pi

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ (a : ℝ) / b = x

theorem rational_count : card ({x : ℝ | is_rational x ∧ (x = sqrt_2 ∨ x = cbrt_3 ∨ x = 0 ∨ x = pi_value)}) = 1 :=
by
  sorry

end rational_count_l63_63075


namespace symmetry_center_example_l63_63750

-- Define the function tan(2x - π/4)
noncomputable def func (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

-- Define what it means to be a symmetry center for the function
def is_symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * (p.1) - x) = 2 * p.2 - f x

-- Statement of the proof problem
theorem symmetry_center_example : is_symmetry_center func (-Real.pi / 8, 0) :=
sorry

end symmetry_center_example_l63_63750


namespace binomial_sum_alternate_l63_63488

theorem binomial_sum_alternate :
  ∑ k in Finset.range 51, if k % 2 = 0 then binom 50 k else -2 * binom 50 k = 0 :=
by
  sorry

end binomial_sum_alternate_l63_63488


namespace three_digit_sum_mod_8_9_eq_6492_l63_63830

theorem three_digit_sum_mod_8_9_eq_6492 :
  let lcm_8_9 := Nat.lcm 8 9 in
  let seq := List.range' 145 937 72 in
  seq.sum = 6492 :=
by
  have lcm_72 : Nat.lcm 8 9 = 72 := by norm_num
  sorry

end three_digit_sum_mod_8_9_eq_6492_l63_63830


namespace coefficient_of_third_term_l63_63696

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_third_term :
  let a := (x : ℚ)
  let b := (y : ℚ)
  let n := 8
  let r := 2
  a = x → b = 1/(2*√x) → ∑ k in range(n+1), binom n k * a^(n-k) * b^k = (7 : ℚ) :=
by
  sorry


end coefficient_of_third_term_l63_63696


namespace inequality_81_over_4_l63_63278

theorem inequality_81_over_4 (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hprod : a * b * c * d = 1 / 4) :
  (16 * a * c + a / (c^2 * b) + 16 * c / (a^2 * d) + 4 / (a * c)) *
  (b * d + b / (256 * d^2 * c) + d / (b^2 * a) + 1 / (64 * b * d)) ≥ 81 / 4 :=
begin
  sorry
end

end inequality_81_over_4_l63_63278


namespace cost_price_correct_purchase_and_max_profit_l63_63436

-- Definitions
variable (x : ℝ) (costA : ℝ) (costB : ℝ)
variable (a b : ℝ) -- number of type A and type B lamps
variable (sellA sellB : ℝ) -- selling prices of type A and B
variable (budget : ℝ)
variable (totalLamps : ℝ)
variable (maximize_profit: ℝ)

-- Type constraints
axiom cost_diff : costA = costB + 40
axiom purchase_equiv : 2000 / costA = 1600 / costB
axiom budget_constraint : costA * a + costB * b ≤ 14550
axiom total_lamps : a + b = 80
axiom sell_prices : sellA = 300 ∧ sellB = 200
axiom maximize_profit_def : 
    maximize_profit = (sellA - costA) * a + (sellB - costB) * b

-- Correct answers
axiom correct_costA : costA = 200
axiom correct_costB : costB = 160
axiom correct_a : a = 43
axiom correct_b : b = 37
axiom correct_max_profit : maximize_profit = 5780

-- Theorems to prove
theorem cost_price_correct : costA = 200 ∧ costB = 160 :=
begin
  rw [←cost_diff, ←purchase_equiv],
  suffices : costA = 200,
  { split; norm_num },
  sorry
end

theorem purchase_and_max_profit : 
  a = 43 ∧ b = 37 ∧ maximize_profit = 5780 :=
begin
  have h1 : 40 * a ≤ 1750, sorry,
  have h2 : (costA * a + costB * b ≤ budget), sorry,
  have max_profit: (maximize_profit = (300 - 200) * a + (200 - 160) * b), sorry,
  split,
  { exact correct_a },
  split,
  { exact correct_b },
  { exact correct_max_profit }
end

end cost_price_correct_purchase_and_max_profit_l63_63436


namespace probability_all_blue_jellybeans_removed_l63_63897

def num_red_jellybeans : ℕ := 10
def num_blue_jellybeans : ℕ := 10
def total_jellybeans : ℕ := num_red_jellybeans + num_blue_jellybeans

def prob_first_blue : ℚ := num_blue_jellybeans / total_jellybeans
def prob_second_blue : ℚ := (num_blue_jellybeans - 1) / (total_jellybeans - 1)
def prob_third_blue : ℚ := (num_blue_jellybeans - 2) / (total_jellybeans - 2)

def prob_all_blue : ℚ := prob_first_blue * prob_second_blue * prob_third_blue

theorem probability_all_blue_jellybeans_removed :
  prob_all_blue = 1 / 9.5 := sorry

end probability_all_blue_jellybeans_removed_l63_63897


namespace contrapositive_equivalence_l63_63426

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1)) ↔ (∀ x : ℝ, (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0)) :=
by {
  sorry
}

end contrapositive_equivalence_l63_63426


namespace sarah_can_gather_info_l63_63257

noncomputable def probability_gather_info_both_classes : ℚ :=
  let total_students := 30
  let german_students := 22
  let italian_students := 26
  let both_classes_students := german_students + italian_students - total_students
  let only_german := german_students - both_classes_students
  let only_italian := italian_students - both_classes_students
  let total_pairs := (total_students * (total_students - 1)) / 2
  let german_pairs := (only_german * (only_german - 1)) / 2
  let italian_pairs := (only_italian * (only_italian - 1)) / 2
  let unfavorable_pairs := german_pairs + italian_pairs
  let favorable_pairs := total_pairs - unfavorable_pairs
  favorable_pairs /. total_pairs

theorem sarah_can_gather_info :
  probability_gather_info_both_classes = 401 / 435 :=
sorry

end sarah_can_gather_info_l63_63257


namespace find_last_two_digits_l63_63256

noncomputable def tenth_digit (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ) : ℕ :=
d7 + d8

noncomputable def ninth_digit (d1 d2 d3 d4 d5 d6 d7 : ℕ) : ℕ :=
d6 + d7

theorem find_last_two_digits :
  ∃ d9 d10 : ℕ, d9 = ninth_digit 1 1 2 3 5 8 13 ∧ d10 = tenth_digit 1 1 2 3 5 8 13 21 :=
by
  sorry

end find_last_two_digits_l63_63256


namespace total_cost_is_18_l63_63774

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_l63_63774


namespace james_profit_l63_63707

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l63_63707


namespace JameMade112kProfit_l63_63712

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l63_63712


namespace max_triangle_area_l63_63266

noncomputable def calculate_area (a b c r : ℝ) : ℝ :=
  let s := (a + b + c) / 2 -- semi-perimeter
  in Math.sqrt (s * (s - a) * (s - b) * (s - c)) -- Heron's formula

theorem max_triangle_area :
  let AB := 12
  let BC := 18
  let CA := 22
  -- circumcircles intersect at distinct points P and D
  let I_B := sorry -- incenters of triangle ABD
  let I_C := sorry -- incenters of triangle ACD
  -- P is the intersection of the circumcircles
  Π (D P : Point) (in_BC : (BC.contains D)) (circ_BI_BD circ_CI_CD : Circle),
  circ_BI_BD.contains P ∧ circ_CI_CD.contains P ∧ P ≠ D ∧
  let angle_CPB := 120 * (Real.pi / 180) -- 120 degrees in radians
  let max_distance := BC * (Real.sin (angle_CPB / 2)) -- max height/radius
  calculate_area AB max_distance BC = 81 :=
begin
  sorry -- Proof will go here
end

end max_triangle_area_l63_63266


namespace max_sum_of_16_ages_at_least_k_l63_63116

theorem max_sum_of_16_ages_at_least_k (ages : Fin 50 → ℕ) (h_distinct : Function.Injective ages) (h_sum : (∑ i, ages i) = 1555) :
  ∃ (k : ℕ), k = 776 ∧ ∃ (S : Finset (Fin 50)), S.card = 16 ∧ (∑ i in S, ages i) ≥ k :=
by {
  -- This is the definition of the theorem, proof is omitted
  sorry
}

end max_sum_of_16_ages_at_least_k_l63_63116


namespace line_through_point_and_perpendicular_l63_63811

theorem line_through_point_and_perpendicular (x y : ℝ) (c : ℝ) :
  (0, 3) ∈ {p : ℝ × ℝ | p.1 - 2 * p.2 + c = 0} ∧
  (∀ x y, 2 * x + y - 5 = 0 → ∃ k, p.1 - 2 * p.2 - k = 0) →
  c = 6 →
  {p : ℝ × ℝ | p.1 - 2 * p.2 + c = 0} = {p : ℝ × ℝ | p.1 - 2 * p.2 + 6 = 0} :=
by
  sorry

end line_through_point_and_perpendicular_l63_63811


namespace num_integers_in_range_l63_63632

theorem num_integers_in_range : 
  let π_approx := 3.14 in
  let lower_bound := -6 * π_approx in
  let upper_bound := 12 * π_approx in
  let n_start := Int.ceil lower_bound in
  let n_end := Int.floor upper_bound in
  (n_end - n_start + 1) = 56 := 
by
  sorry

end num_integers_in_range_l63_63632


namespace inequality_solution_l63_63607

theorem inequality_solution (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by
  sorry

end inequality_solution_l63_63607
