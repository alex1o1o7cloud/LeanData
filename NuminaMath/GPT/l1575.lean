import Mathlib

namespace inequality_solution_l1575_157576

theorem inequality_solution (x : ℝ) : (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) ↔ (-1/2 ≤ x ∧ x < 1) :=
by
  sorry

end inequality_solution_l1575_157576


namespace repaired_shoes_lifespan_l1575_157532

-- Definitions of given conditions
def cost_repair : Float := 11.50
def cost_new : Float := 28.00
def lifespan_new : Float := 2.0
def percentage_increase : Float := 21.73913043478261 / 100

-- Cost per year of new shoes
def cost_per_year_new : Float := cost_new / lifespan_new

-- Cost per year of repaired shoes
def cost_per_year_repair (T : Float) : Float := cost_repair / T

-- Theorem statement (goal)
theorem repaired_shoes_lifespan (T : Float) (h : cost_per_year_new = cost_per_year_repair T * (1 + percentage_increase)) : T = 0.6745 :=
by
  sorry

end repaired_shoes_lifespan_l1575_157532


namespace smallest_k_for_a_n_digital_l1575_157501

theorem smallest_k_for_a_n_digital (a n : ℕ) (h : 10^2013 ≤ a^n ∧ a^n < 10^2014) : 
  ∀ k : ℕ, (∀ b : ℕ, 10^(k-1) ≤ b → b < 10^k → (¬(10^2013 ≤ b^n ∧ b^n < 10^2014))) ↔ k = 2014 :=
by 
  sorry

end smallest_k_for_a_n_digital_l1575_157501


namespace roden_gold_fish_count_l1575_157546

theorem roden_gold_fish_count
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (gold_fish : ℕ)
  (h1 : total_fish = 22)
  (h2 : blue_fish = 7)
  (h3 : total_fish = blue_fish + gold_fish) : gold_fish = 15 :=
by
  sorry

end roden_gold_fish_count_l1575_157546


namespace pete_nickels_spent_l1575_157589

-- Definitions based on conditions
def initial_amount_per_person : ℕ := 250 -- 250 cents for $2.50
def total_initial_amount : ℕ := 2 * initial_amount_per_person
def total_expense : ℕ := 200 -- they spent 200 cents in total
def raymond_dimes_left : ℕ := 7
def value_of_dime : ℕ := 10
def raymond_remaining_amount : ℕ := raymond_dimes_left * value_of_dime
def raymond_spent_amount : ℕ := total_expense - raymond_remaining_amount
def value_of_nickel : ℕ := 5

-- Theorem to prove Pete spent 14 nickels
theorem pete_nickels_spent : 
  (total_expense - raymond_spent_amount) / value_of_nickel = 14 :=
by
  sorry

end pete_nickels_spent_l1575_157589


namespace segment_combination_l1575_157579

theorem segment_combination (x y : ℕ) :
  7 * x + 12 * y = 100 ↔ (x, y) = (4, 6) :=
by
  sorry

end segment_combination_l1575_157579


namespace min_students_orchestra_l1575_157570

theorem min_students_orchestra (n : ℕ) 
  (h1 : n % 9 = 0)
  (h2 : n % 10 = 0)
  (h3 : n % 11 = 0) : 
  n ≥ 990 ∧ ∃ k, n = 990 * k :=
by
  sorry

end min_students_orchestra_l1575_157570


namespace theodoreEarningsCorrect_l1575_157545

noncomputable def theodoreEarnings : ℝ := 
  let s := 10
  let ps := 20
  let w := 20
  let pw := 5
  let b := 15
  let pb := 15
  let m := 150
  let l := 200
  let t := 0.10
  let totalEarnings := (s * ps) + (w * pw) + (b * pb)
  let expenses := m + l
  let earningsBeforeTaxes := totalEarnings - expenses
  let taxes := t * earningsBeforeTaxes
  earningsBeforeTaxes - taxes

theorem theodoreEarningsCorrect :
  theodoreEarnings = 157.50 :=
by sorry

end theodoreEarningsCorrect_l1575_157545


namespace flower_team_participation_l1575_157523

-- Definitions based on the conditions in the problem
def num_rows : ℕ := 60
def first_row_people : ℕ := 40
def people_increment : ℕ := 1

-- Statement to be proved in Lean
theorem flower_team_participation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ num_rows) : 
  ∃ y : ℕ, y = first_row_people - people_increment + x :=
by
  -- Placeholder for the proof
  sorry

end flower_team_participation_l1575_157523


namespace solve_inequality_l1575_157533

-- Define the domain and inequality conditions
def inequality_condition (x : ℝ) : Prop := (1 / (x - 1)) > 1
def domain_condition (x : ℝ) : Prop := x ≠ 1

-- State the theorem to be proved.
theorem solve_inequality (x : ℝ) : domain_condition x → inequality_condition x → 1 < x ∧ x < 2 :=
by
  intros h_domain h_ineq
  sorry

end solve_inequality_l1575_157533


namespace radius_of_circle_l1575_157538

theorem radius_of_circle (r : ℝ) : 
  (∀ x : ℝ, (4 * x^2 + r = x) → (1 - 16 * r = 0)) → r = 1 / 16 :=
by
  intro H
  have h := H 0
  simp at h
  sorry

end radius_of_circle_l1575_157538


namespace maximum_small_circles_l1575_157581

-- Definitions for small circle radius, large circle radius, and the maximum number n.
def smallCircleRadius : ℝ := 1
def largeCircleRadius : ℝ := 11

-- Function to check if small circles can be placed without overlapping
def canPlaceCircles (n : ℕ) : Prop := n * 2 < 2 * Real.pi * (largeCircleRadius - smallCircleRadius)

theorem maximum_small_circles : ∀ n : ℕ, canPlaceCircles n → n ≤ 31 := by
  sorry

end maximum_small_circles_l1575_157581


namespace circle_equation_solution_l1575_157512

theorem circle_equation_solution (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 2 * m * y + 2 * m^2 + m - 1 = 0) ↔ m < 1 :=
sorry

end circle_equation_solution_l1575_157512


namespace rate_percent_calculation_l1575_157520

theorem rate_percent_calculation 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ) 
  (h1 : SI = 3125) 
  (h2 : P = 12500) 
  (h3 : T = 7) 
  (h4 : SI = P * R * T / 100) :
  R = 3.57 :=
by
  sorry

end rate_percent_calculation_l1575_157520


namespace function_quadrants_l1575_157530

theorem function_quadrants (a b : ℝ) (h_a : a > 1) (h_b : b < -1) :
  (∀ x : ℝ, a^x + b > 0 → ∃ x1 : ℝ, a^x1 + b < 0 → ∃ x2 : ℝ, a^x2 + b < 0) :=
sorry

end function_quadrants_l1575_157530


namespace area_of_rectangle_l1575_157502

-- Define the lengths in meters
def length : ℝ := 1.2
def width : ℝ := 0.5

-- Define the function to calculate the area of a rectangle
def area (l w : ℝ) : ℝ := l * w

-- Prove that the area of the rectangle with given length and width is 0.6 square meters
theorem area_of_rectangle :
  area length width = 0.6 := by
  -- This is just the statement. We omit the proof with sorry.
  sorry

end area_of_rectangle_l1575_157502


namespace amelia_distance_l1575_157513

theorem amelia_distance (total_distance amelia_monday_distance amelia_tuesday_distance : ℕ) 
  (h1 : total_distance = 8205) 
  (h2 : amelia_monday_distance = 907) 
  (h3 : amelia_tuesday_distance = 582) : 
  total_distance - (amelia_monday_distance + amelia_tuesday_distance) = 6716 := 
by 
  sorry

end amelia_distance_l1575_157513


namespace carol_savings_l1575_157511

theorem carol_savings (S : ℝ) (h1 : ∀ t : ℝ, t = S - (2/3) * S) (h2 : S + (S - (2/3) * S) = 1/4) : S = 3/16 :=
by {
  sorry
}

end carol_savings_l1575_157511


namespace least_positive_integer_condition_l1575_157524

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_l1575_157524


namespace rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l1575_157534

theorem rhombus_diagonal_BD_equation (A C : ℝ × ℝ) (AB_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 1 ∧ b = 6 ∧ ∀ x y : ℝ, x + y - 6 = 0 := by
  sorry

theorem rhombus_diagonal_AD_equation (A C : ℝ × ℝ) (AB_eq BD_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0 ∧ x + y - 6 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 3 ∧ b = 14 ∧ ∀ x y : ℝ, x - 3 * y + 14 = 0 := by
  sorry

end rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l1575_157534


namespace geometric_ratio_l1575_157548

theorem geometric_ratio (a₁ q : ℝ) (h₀ : a₁ ≠ 0) (h₁ : a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) : q = -2 ∨ q = 1 :=
by
  sorry

end geometric_ratio_l1575_157548


namespace pentagon_area_l1575_157583

-- Definitions of the side lengths of the pentagon
def side1 : ℕ := 12
def side2 : ℕ := 17
def side3 : ℕ := 25
def side4 : ℕ := 18
def side5 : ℕ := 17

-- Definitions for the rectangle and triangle dimensions
def rectangle_width : ℕ := side4
def rectangle_height : ℕ := side1
def triangle_base : ℕ := side4
def triangle_height : ℕ := side3 - side1

-- The area of the pentagon proof statement
theorem pentagon_area : rectangle_width * rectangle_height +
    (triangle_base * triangle_height) / 2 = 333 := by
  sorry

end pentagon_area_l1575_157583


namespace coefficient_of_q_is_correct_l1575_157578

theorem coefficient_of_q_is_correct (q' : ℕ → ℕ) : 
  (∀ q : ℕ, q' q = 3 * q - 3) ∧  q' (q' 7) = 306 → ∃ a : ℕ, (∀ q : ℕ, q' q = a * q - 3) ∧ a = 17 :=
by
  sorry

end coefficient_of_q_is_correct_l1575_157578


namespace balance_balls_l1575_157518

-- Define the weights of the balls as variables
variables (B R O S : ℝ)

-- Given conditions
axiom h1 : R = 2 * B
axiom h2 : O = (7 / 3) * B
axiom h3 : S = (5 / 3) * B

-- Statement to prove
theorem balance_balls :
  (5 * R + 3 * O + 4 * S) = (71 / 3) * B :=
by {
  -- The proof is omitted
  sorry
}

end balance_balls_l1575_157518


namespace value_of_A_l1575_157508

theorem value_of_A (A B C D : ℕ) (h1 : A * B = 60) (h2 : C * D = 60) (h3 : A - B = C + D) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : A ≠ D) (h7 : B ≠ C) (h8 : B ≠ D) (h9 : C ≠ D) : A = 20 :=
by sorry

end value_of_A_l1575_157508


namespace right_triangle_area_l1575_157505

/-- Given a right triangle with one leg of length 3 and the hypotenuse of length 5,
    the area of the triangle is 6. -/
theorem right_triangle_area (a b c : ℝ) (h₁ : a = 3) (h₂ : c = 5) (h₃ : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 6 := 
sorry

end right_triangle_area_l1575_157505


namespace no_nat_n_exists_l1575_157592

theorem no_nat_n_exists (n : ℕ) : ¬ ∃ n, ∃ k, n ^ 2012 - 1 = 2 ^ k := by
  sorry

end no_nat_n_exists_l1575_157592


namespace application_methods_count_l1575_157560

theorem application_methods_count (total_universities: ℕ) (universities_with_coinciding_exams: ℕ) (chosen_universities: ℕ) 
  (remaining_universities: ℕ) (remaining_combinations: ℕ) : 
  total_universities = 6 → universities_with_coinciding_exams = 2 → chosen_universities = 3 → 
  remaining_universities = 4 → remaining_combinations = 16 := 
by
  intros
  sorry

end application_methods_count_l1575_157560


namespace four_pow_four_mul_five_pow_four_l1575_157529

theorem four_pow_four_mul_five_pow_four : (4 ^ 4) * (5 ^ 4) = 160000 := by
  sorry

end four_pow_four_mul_five_pow_four_l1575_157529


namespace values_of_k_for_exactly_one_real_solution_l1575_157536

variable {k : ℝ}

def quadratic_eq (k : ℝ) : Prop := 3 * k^2 + 42 * k - 573 = 0

theorem values_of_k_for_exactly_one_real_solution :
  quadratic_eq k ↔ k = 8 ∨ k = -22 := by
  sorry

end values_of_k_for_exactly_one_real_solution_l1575_157536


namespace beavers_fraction_l1575_157558

theorem beavers_fraction (total_beavers : ℕ) (swim_percentage : ℕ) (work_percentage : ℕ) (fraction_working : ℕ) : 
total_beavers = 4 → 
swim_percentage = 75 → 
work_percentage = 100 - swim_percentage → 
fraction_working = 1 →
(work_percentage * total_beavers) / 100 = fraction_working → 
fraction_working / total_beavers = 1 / 4 :=
by 
  intros h1 h2 h3 h4 h5 
  sorry

end beavers_fraction_l1575_157558


namespace wire_length_l1575_157598

variables (L M S W : ℕ)

def ratio_condition (L M S : ℕ) : Prop :=
  L * 2 = 7 * S ∧ M * 2 = 3 * S

def total_length (L M S : ℕ) : ℕ :=
  L + M + S

theorem wire_length (h : ratio_condition L M 16) : total_length L M 16 = 96 :=
by sorry

end wire_length_l1575_157598


namespace smallest_k_for_abk_l1575_157588

theorem smallest_k_for_abk : ∃ (k : ℝ), (∀ (a b : ℝ), a + b = k ∧ ab = k → k = 4) :=
sorry

end smallest_k_for_abk_l1575_157588


namespace horizontal_force_magnitude_l1575_157554

-- We state our assumptions and goal
theorem horizontal_force_magnitude (W : ℝ) : 
  (∀ μ : ℝ, μ = (Real.sin (Real.pi / 6)) / (Real.cos (Real.pi / 6)) ∧ 
    (∀ P : ℝ, 
      (P * (Real.sin (Real.pi / 3))) = 
      ((μ * (W * (Real.cos (Real.pi / 6)) + P * (Real.cos (Real.pi / 3)))) + W * (Real.sin (Real.pi / 6))) →
      P = W * Real.sqrt 3)) :=
sorry

end horizontal_force_magnitude_l1575_157554


namespace estimate_pi_l1575_157594

theorem estimate_pi (m : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (h1 : m = 56) (h2 : n = 200) (h3 : a = 1/2) (h4 : b = 1/4) :
  (m / n) = (π / 4 - 1 / 2) ↔ π = 78 / 25 :=
by
  sorry

end estimate_pi_l1575_157594


namespace union_of_A_B_complement_intersection_l1575_157597

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -x^2 + 2*x + 15 ≤ 0 }

def B : Set ℝ := { x | |x - 5| < 1 }

theorem union_of_A_B :
  A ∪ B = { x | x ≤ -3 ∨ x > 4 } :=
by
  sorry

theorem complement_intersection :
  (U \ A) ∩ B = { x | 4 < x ∧ x < 5 } :=
by
  sorry

end union_of_A_B_complement_intersection_l1575_157597


namespace find_pq_l1575_157543

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_pq (p q : ℕ) 
(hp : is_prime p) 
(hq : is_prime q) 
(h : is_prime (q^2 - p^2)) : 
  p * q = 6 :=
by sorry

end find_pq_l1575_157543


namespace arithmetic_sequence_a3_l1575_157547

theorem arithmetic_sequence_a3 (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : a2 = a1 + (a1 + a5 - a1) / 4)
  (h2 : a3 = a1 + 2 * (a1 + a5 - a1) / 4) 
  (h3 : a4 = a1 + 3 * (a1 + a5 - a1) / 4) 
  (h4 : a5 = a1 + 4 * (a1 + a5 - a1) / 4)
  (h_sum : 5 * a3 = 15) : 
  a3 = 3 :=
sorry

end arithmetic_sequence_a3_l1575_157547


namespace maximize_profit_l1575_157517

noncomputable def profit (m : ℝ) : ℝ := 
  29 - (16 / (m + 1) + (m + 1))

theorem maximize_profit : 
  ∃ m : ℝ, m = 3 ∧ m ≥ 0 ∧ profit m = 21 :=
by
  use 3
  repeat { sorry }

end maximize_profit_l1575_157517


namespace a2009_equals_7_l1575_157542

def sequence_element (n k : ℕ) : ℚ :=
  if k = 0 then 0 else (n - k + 1) / k

def cumulative_count (n : ℕ) : ℕ := n * (n + 1) / 2

theorem a2009_equals_7 : 
  let n := 63
  let m := 2009
  let subset_cumulative_count := cumulative_count n
  (2 * m = n * (n + 1) - 14 ∧
   m = subset_cumulative_count - 7 ∧ 
   sequence_element n 8 = 7) →
  sequence_element n (subset_cumulative_count - m + 1) = 7 :=
by
  -- proof steps to be filled here
  sorry

end a2009_equals_7_l1575_157542


namespace opposite_of_negative_six_is_six_l1575_157552

-- Define what it means for one number to be the opposite of another.
def is_opposite (a b : Int) : Prop :=
  a = -b

-- The statement to be proved: the opposite number of -6 is 6.
theorem opposite_of_negative_six_is_six : is_opposite (-6) 6 :=
  by sorry

end opposite_of_negative_six_is_six_l1575_157552


namespace total_protest_days_l1575_157528

-- Definitions for the problem conditions
def first_protest_days : ℕ := 4
def second_protest_days : ℕ := first_protest_days + (first_protest_days / 4)

-- The proof statement
theorem total_protest_days : first_protest_days + second_protest_days = 9 := sorry

end total_protest_days_l1575_157528


namespace polynomial_divisibility_l1575_157519

theorem polynomial_divisibility 
  (a b c : ℤ)
  (P : ℤ → ℤ)
  (root_condition : ∃ u v : ℤ, u * v * (u + v) = -c ∧ u * v = b) 
  (P_def : ∀ x, P x = x^3 + a * x^2 + b * x + c) :
  2 * P (-1) ∣ (P 1 + P (-1) - 2 * (1 + P 0)) :=
by
  sorry

end polynomial_divisibility_l1575_157519


namespace max_plus_min_value_of_f_l1575_157551

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_plus_min_value_of_f :
  let M := ⨆ x, f x
  let m := ⨅ x, f x
  M + m = 4 :=
by 
  sorry

end max_plus_min_value_of_f_l1575_157551


namespace unique_solution_l1575_157565

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1))

theorem unique_solution : ∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) → system_of_equations x y z → (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  intro x y z hx hy hz h
  sorry

end unique_solution_l1575_157565


namespace find_exponent_l1575_157514

theorem find_exponent (m x y a : ℝ) (h : y = m * x ^ a) (hx : x = 1 / 4) (hy : y = 1 / 2) : a = 1 / 2 :=
by
  sorry

end find_exponent_l1575_157514


namespace unsolved_problems_exist_l1575_157516

noncomputable def main_theorem: Prop :=
  ∃ (P : Prop), ¬(P = true) ∧ ¬(P = false)

theorem unsolved_problems_exist : main_theorem :=
sorry

end unsolved_problems_exist_l1575_157516


namespace min_pieces_for_net_l1575_157572

theorem min_pieces_for_net (n : ℕ) : ∃ (m : ℕ), m = n * (n + 1) := by
  sorry

end min_pieces_for_net_l1575_157572


namespace solve_quadratic_1_solve_quadratic_2_l1575_157585

theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 8 * x + 4 = 0 ↔ x = 2/3 ∨ x = 2 := by
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1)^2 = (x - 3)^2 ↔ x = 4/3 ∨ x = -2 := by
  sorry

end solve_quadratic_1_solve_quadratic_2_l1575_157585


namespace expr_min_value_expr_min_at_15_l1575_157522

theorem expr_min_value (a x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  (|x - a| + |x - 15| + |x - (a + 15)|) = 30 - x := 
sorry

theorem expr_min_at_15 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 15) : 
  (|15 - a| + |15 - 15| + |15 - (a + 15)|) = 15 := 
sorry

end expr_min_value_expr_min_at_15_l1575_157522


namespace games_did_not_work_l1575_157593

theorem games_did_not_work 
  (games_from_friend : ℕ) 
  (games_from_garage_sale : ℕ) 
  (good_games : ℕ) 
  (total_games : ℕ := games_from_friend + games_from_garage_sale) 
  (did_not_work : ℕ := total_games - good_games) :
  games_from_friend = 41 ∧ 
  games_from_garage_sale = 14 ∧ 
  good_games = 24 → 
  did_not_work = 31 := 
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end games_did_not_work_l1575_157593


namespace percent_only_cats_l1575_157510

def total_students := 500
def total_cats := 120
def total_dogs := 200
def both_cats_and_dogs := 40
def only_cats := total_cats - both_cats_and_dogs

theorem percent_only_cats:
  (only_cats : ℕ) / (total_students : ℕ) * 100 = 16 := 
by 
  sorry

end percent_only_cats_l1575_157510


namespace smallest_fraction_divides_exactly_l1575_157537

theorem smallest_fraction_divides_exactly (a b c p q r m n : ℕ)
    (h1: a = 6) (h2: b = 5) (h3: c = 10) (h4: p = 7) (h5: q = 14) (h6: r = 21)
    (h1_frac: 6/7 = a/p) (h2_frac: 5/14 = b/q) (h3_frac: 10/21 = c/r)
    (h_lcm: m = Nat.lcm p (Nat.lcm q r)) (h_gcd: n = Nat.gcd a (Nat.gcd b c)) :
  (n/m) = 1/42 :=
by 
  sorry

end smallest_fraction_divides_exactly_l1575_157537


namespace biology_marks_l1575_157506

theorem biology_marks 
  (e : ℕ) (m : ℕ) (p : ℕ) (c : ℕ) (a : ℕ) (n : ℕ) (b : ℕ) 
  (h_e : e = 96) (h_m : m = 95) (h_p : p = 82) (h_c : c = 97) (h_a : a = 93) (h_n : n = 5)
  (h_total : e + m + p + c + b = a * n) :
  b = 95 :=
by 
  sorry

end biology_marks_l1575_157506


namespace real_root_uncertainty_l1575_157553

noncomputable def f (x m : ℝ) : ℝ := m * x^2 - 2 * (m + 2) * x + m + 5
noncomputable def g (x m : ℝ) : ℝ := (m - 5) * x^2 - 2 * (m + 2) * x + m

theorem real_root_uncertainty (m : ℝ) :
  (∀ x : ℝ, f x m ≠ 0) → 
  (m ≤ 5 → ∃ x : ℝ, g x m = 0 ∧ ∀ y : ℝ, y ≠ x → g y m = 0) ∧
  (m > 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0) :=
sorry

end real_root_uncertainty_l1575_157553


namespace smallest_six_digit_number_exists_l1575_157561

def three_digit_number (n : ℕ) := n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ 100 ≤ n ∧ n < 1000

def valid_six_digit_number (m n : ℕ) := 
  (m * 1000 + n) % 4 = 0 ∧ (m * 1000 + n) % 5 = 0 ∧ (m * 1000 + n) % 6 = 0 ∧ 
  three_digit_number n ∧ 0 ≤ m ∧ m < 1000

theorem smallest_six_digit_number_exists : 
  ∃ m n, valid_six_digit_number m n ∧ (∀ m' n', valid_six_digit_number m' n' → m * 1000 + n ≤ m' * 1000 + n') :=
sorry

end smallest_six_digit_number_exists_l1575_157561


namespace calculate_s_at_2_l1575_157515

-- Given definitions
def t (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 1
def s (p : ℝ) : ℝ := p^3 - 4 * p^2 + p + 6

-- The target statement
theorem calculate_s_at_2 : s 2 = ((5 + Real.sqrt 33) / 4)^3 - 4 * ((5 + Real.sqrt 33) / 4)^2 + ((5 + Real.sqrt 33) / 4) + 6 := 
by 
  sorry

end calculate_s_at_2_l1575_157515


namespace right_triangle_hypotenuse_length_l1575_157539

theorem right_triangle_hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = 10) (h₂ : b = 24) (h₃ : c^2 = a^2 + b^2) : c = 26 :=
by
  -- sorry is used to skip the actual proof
  sorry

end right_triangle_hypotenuse_length_l1575_157539


namespace sweater_markup_l1575_157527

-- Conditions
variables (W R : ℝ)
axiom h1 : 0.40 * R = 1.20 * W

-- Theorem statement
theorem sweater_markup (W R : ℝ) (h1 : 0.40 * R = 1.20 * W) : (R - W) / W * 100 = 200 :=
sorry

end sweater_markup_l1575_157527


namespace contrapositive_roots_l1575_157540

theorem contrapositive_roots {a b c : ℝ} (h : a ≠ 0) (hac : a * c ≤ 0) :
  ¬ (∀ x : ℝ, (a * x^2 - b * x + c = 0) → x > 0) :=
sorry

end contrapositive_roots_l1575_157540


namespace initial_marbles_l1575_157575

theorem initial_marbles (total_marbles now found: ℕ) (h_found: found = 7) (h_now: now = 28) : 
  total_marbles = now - found → total_marbles = 21 := by
  -- Proof goes here.
  sorry

end initial_marbles_l1575_157575


namespace inequality_solution_l1575_157582

theorem inequality_solution (x : ℝ) :
  (abs ((x^2 - 5 * x + 4) / 3) < 1) ↔ 
  ((5 - Real.sqrt 21) / 2 < x) ∧ (x < (5 + Real.sqrt 21) / 2) := 
sorry

end inequality_solution_l1575_157582


namespace neg_two_squared_result_l1575_157562

theorem neg_two_squared_result : -2^2 = -4 :=
by
  sorry

end neg_two_squared_result_l1575_157562


namespace lemon_pie_degrees_l1575_157566

def total_students : ℕ := 45
def chocolate_pie_students : ℕ := 15
def apple_pie_students : ℕ := 10
def blueberry_pie_students : ℕ := 7
def cherry_and_lemon_students := total_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
def lemon_pie_students := cherry_and_lemon_students / 2

theorem lemon_pie_degrees (students_nonnegative : lemon_pie_students ≥ 0) (students_rounding : lemon_pie_students = 7) :
  (lemon_pie_students * 360 / total_students) = 56 := 
by
  -- Proof to be provided
  sorry

end lemon_pie_degrees_l1575_157566


namespace parallel_vectors_l1575_157596

open Real

theorem parallel_vectors (k : ℝ) 
  (a : ℝ × ℝ := (k-1, 1)) 
  (b : ℝ × ℝ := (k+3, k)) 
  (h : a.1 * b.2 = a.2 * b.1) : 
  k = 3 ∨ k = -1 :=
by
  sorry

end parallel_vectors_l1575_157596


namespace man_l1575_157504

theorem man's_speed_downstream (v : ℝ) (speed_of_stream : ℝ) (speed_upstream : ℝ) : 
  speed_upstream = v - speed_of_stream ∧ speed_of_stream = 1.5 ∧ speed_upstream = 8 → v + speed_of_stream = 11 :=
by
  sorry

end man_l1575_157504


namespace cost_of_items_l1575_157573

variable (p q r : ℝ)

theorem cost_of_items :
  8 * p + 2 * q + r = 4.60 → 
  2 * p + 5 * q + r = 3.90 → 
  p + q + 3 * r = 2.75 → 
  4 * p + 3 * q + 2 * r = 7.4135 :=
by
  intros h1 h2 h3
  sorry

end cost_of_items_l1575_157573


namespace johns_shell_arrangements_l1575_157599

-- Define the total number of arrangements without considering symmetries
def totalArrangements := Nat.factorial 12

-- Define the number of equivalent arrangements due to symmetries
def symmetries := 6 * 2

-- Define the number of distinct arrangements
def distinctArrangements : Nat := totalArrangements / symmetries

-- State the theorem
theorem johns_shell_arrangements : distinctArrangements = 479001600 :=
by
  sorry

end johns_shell_arrangements_l1575_157599


namespace proof_problem_l1575_157567

-- Variables representing the numbers a, b, and c
variables {a b c : ℝ}

-- Given condition
def given_condition (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (b^2 + c^2) = a / c

-- Required to prove
def to_prove (a b c : ℝ) : Prop :=
  (a / b = b / c) → False

-- Theorem stating that the given condition does not imply the required assertion
theorem proof_problem (a b c : ℝ) (h : given_condition a b c) : to_prove a b c :=
sorry

end proof_problem_l1575_157567


namespace only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l1575_157590

theorem only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c
  (n a b c : ℕ) (hn : n > 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hca : c > a) (hcb : c > b) (hab : a ≤ b) :
  n * a + n * b = n * c ↔ (n = 2 ∧ b = a ∧ c = a + 1) := by
  sorry

end only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l1575_157590


namespace side_increase_percentage_l1575_157556

theorem side_increase_percentage (s : ℝ) (p : ℝ) 
  (h : (s^2) * (1.5625) = (s * (1 + p / 100))^2) : p = 25 := 
sorry

end side_increase_percentage_l1575_157556


namespace range_of_PF1_minus_PF2_l1575_157509

noncomputable def ellipse_property (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : Prop :=
  ∃ f : ℝ, f = (2 * Real.sqrt 5 / 5) * x0 ∧ f > 0 ∧ f < 2

theorem range_of_PF1_minus_PF2 (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : 
  ellipse_property x0 h1 h2 := by
  sorry

end range_of_PF1_minus_PF2_l1575_157509


namespace intersection_points_of_graphs_l1575_157549

open Real

theorem intersection_points_of_graphs (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃! x : ℝ, (f (x^3) = f (x^6)) ∧ (x = -1 ∨ x = 0 ∨ x = 1) :=
by
  -- Provide the structure of the proof
  sorry

end intersection_points_of_graphs_l1575_157549


namespace sequence_count_l1575_157526

theorem sequence_count :
  ∃ (a : ℕ → ℕ), 
    a 10 = 3 * a 1 ∧ 
    a 2 + a 8 = 2 * a 5 ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 9 → a (i + 1) = 1 + a i ∨ a (i + 1) = 2 + a i) ∧ 
    (∃ n, n = 80) :=
sorry

end sequence_count_l1575_157526


namespace interest_rate_l1575_157568

/-- 
Given a principal amount that doubles itself in 10 years at simple interest,
prove that the rate of interest per annum is 10%.
-/
theorem interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h1 : SI = P) (h2 : T = 10) (h3 : SI = P * R * T / 100) : 
  R = 10 := by
  sorry

end interest_rate_l1575_157568


namespace odd_function_negative_value_l1575_157525

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_value {f : ℝ → ℝ} (h_odd : is_odd_function f) :
  (∀ x, 0 < x → f x = x^2 - x - 1) → (∀ x, x < 0 → f x = -x^2 - x + 1) :=
by
  sorry

end odd_function_negative_value_l1575_157525


namespace triangle_side_c_l1575_157591

theorem triangle_side_c
  (a b c : ℝ)
  (A B C : ℝ)
  (h_bc : b = 3)
  (h_sinC : Real.sin C = 56 / 65)
  (h_sinB : Real.sin B = 12 / 13)
  (h_Angles : A + B + C = π)
  (h_valid_triangle : ∀ {x y z : ℝ}, x + y > z ∧ x + z > y ∧ y + z > x):
  c = 14 / 5 :=
sorry

end triangle_side_c_l1575_157591


namespace eval_ceil_floor_sum_l1575_157535

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l1575_157535


namespace functional_equation_solution_l1575_157531

noncomputable def function_nat_nat (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, f (x + y) = f x + f y

theorem functional_equation_solution :
  ∀ f : ℕ → ℕ, function_nat_nat f → ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by
  sorry

end functional_equation_solution_l1575_157531


namespace ronald_profit_fraction_l1575_157571

theorem ronald_profit_fraction:
  let initial_units : ℕ := 200
  let total_investment : ℕ := 3000
  let selling_price_per_unit : ℕ := 20
  let total_selling_price := initial_units * selling_price_per_unit
  let total_profit := total_selling_price - total_investment
  (total_profit : ℚ) / total_investment = (1 : ℚ) / 3 :=
by
  -- here we will put the steps needed to prove the theorem.
  sorry

end ronald_profit_fraction_l1575_157571


namespace unique_integer_in_ranges_l1575_157580

theorem unique_integer_in_ranges {x : ℤ} :
  1 < x ∧ x < 9 → 
  2 < x ∧ x < 15 → 
  -1 < x ∧ x < 7 → 
  0 < x ∧ x < 4 → 
  x + 1 < 5 → 
  x = 3 := by
  intros _ _ _ _ _
  sorry

end unique_integer_in_ranges_l1575_157580


namespace conic_section_is_hyperbola_l1575_157559

noncomputable def is_hyperbola (x y : ℝ) : Prop :=
  (x - 4) ^ 2 = 9 * (y + 3) ^ 2 + 27

theorem conic_section_is_hyperbola : ∀ x y : ℝ, is_hyperbola x y → "H" = "H" := sorry

end conic_section_is_hyperbola_l1575_157559


namespace tangent_parallel_line_coordinates_l1575_157586

theorem tangent_parallel_line_coordinates :
  ∃ (m n : ℝ), 
    (∀ x : ℝ, (deriv (λ x => x^4 + x) x = 4 * x^3 + 1)) ∧ 
    (deriv (λ x => x^4 + x) m = -3) ∧ 
    (n = m^4 + m) ∧ 
    (m, n) = (-1, 0) :=
by
  sorry

end tangent_parallel_line_coordinates_l1575_157586


namespace park_bench_problem_l1575_157569

/-- A single bench section at a park can hold either 8 adults or 12 children.
When N bench sections are connected end to end, an equal number of adults and 
children seated together will occupy all the bench space.
This theorem states that the smallest positive integer N such that this condition 
is satisfied is 3. -/
theorem park_bench_problem : ∃ N : ℕ, N > 0 ∧ (8 * N = 12 * N) ∧ N = 3 :=
by
  sorry

end park_bench_problem_l1575_157569


namespace find_a_l1575_157587

theorem find_a (a x : ℝ) (h1 : 3 * a - x = x / 2 + 3) (h2 : x = 2) : a = 2 := 
by
  sorry

end find_a_l1575_157587


namespace mean_of_four_numbers_l1575_157557

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 1/2) : (a + b + c + d) / 4 = 1 / 8 :=
by
  -- proof skipped
  sorry

end mean_of_four_numbers_l1575_157557


namespace B_and_C_have_together_l1575_157574

theorem B_and_C_have_together
  (A B C : ℕ)
  (h1 : A + B + C = 700)
  (h2 : A + C = 300)
  (h3 : C = 200) :
  B + C = 600 := by
  sorry

end B_and_C_have_together_l1575_157574


namespace proof_expression_value_l1575_157541

noncomputable def a : ℝ := 0.15
noncomputable def b : ℝ := 0.06
noncomputable def x : ℝ := a^3
noncomputable def y : ℝ := b^3
noncomputable def z : ℝ := a^2
noncomputable def w : ℝ := b^2

theorem proof_expression_value :
  ( (x - y) / (z + w) ) + 0.009 + w^4 = 0.1300341679616 := sorry

end proof_expression_value_l1575_157541


namespace birds_not_hawks_warbler_kingfisher_l1575_157521

variables (B : ℝ)
variables (hawks paddyfield_warblers kingfishers : ℝ)

-- Conditions
def condition1 := hawks = 0.30 * B
def condition2 := paddyfield_warblers = 0.40 * (B - hawks)
def condition3 := kingfishers = 0.25 * paddyfield_warblers

-- Question: Prove the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers is 35%
theorem birds_not_hawks_warbler_kingfisher (B hawks paddyfield_warblers kingfishers : ℝ) 
 (h1 : hawks = 0.30 * B) 
 (h2 : paddyfield_warblers = 0.40 * (B - hawks)) 
 (h3 : kingfishers = 0.25 * paddyfield_warblers) : 
 (1 - (hawks + paddyfield_warblers + kingfishers) / B) * 100 = 35 :=
by
  sorry

end birds_not_hawks_warbler_kingfisher_l1575_157521


namespace P_iff_nonQ_l1575_157577

-- Given conditions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x ≠ 0 ∨ y ≠ 0
def nonQ (x y : ℝ) : Prop := x = 0 ∧ y = 0

-- Main statement
theorem P_iff_nonQ (x y : ℝ) : P x y ↔ nonQ x y :=
sorry

end P_iff_nonQ_l1575_157577


namespace radius_decrease_l1575_157564

theorem radius_decrease (r r' : ℝ) (A A' : ℝ) (h_original_area : A = π * r^2)
  (h_area_decrease : A' = 0.25 * A) (h_new_area : A' = π * r'^2) : r' = 0.5 * r :=
by
  sorry

end radius_decrease_l1575_157564


namespace system_solve_l1575_157555

theorem system_solve (x y : ℚ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 2 * y = 12) : x + y = 3 / 7 :=
by
  -- The proof will go here, but we skip it for now.
  sorry

end system_solve_l1575_157555


namespace max_a_value_l1575_157503

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a - 1) * x^2 - (a - 1) * x + 2022 ∧ 
                                (a - 1) * x^2 - (a - 1) * x + 2022 ≤ 2022) →
  a = 16177 :=
sorry

end max_a_value_l1575_157503


namespace fraction_numerator_l1575_157563

theorem fraction_numerator (x : ℕ) (h1 : 4 * x - 4 > 0) (h2 : (x : ℚ) / (4 * x - 4) = 3 / 8) : x = 3 :=
by {
  sorry
}

end fraction_numerator_l1575_157563


namespace position_of_point_l1575_157507

theorem position_of_point (a b : ℝ) (h_tangent: (a ≠ 0 ∨ b ≠ 0) ∧ (a^2 + b^2 = 1)) : a^2 + b^2 = 1 :=
by
  sorry

end position_of_point_l1575_157507


namespace R_and_D_expense_corresponding_to_productivity_increase_l1575_157595

/-- Given values for R&D expenses and increase in average labor productivity -/
def R_and_D_t : ℝ := 2640.92
def Delta_APL_t_plus_2 : ℝ := 0.81

/-- Statement to be proved: the R&D expense in million rubles corresponding 
    to an increase in average labor productivity by 1 million rubles per person -/
theorem R_and_D_expense_corresponding_to_productivity_increase : 
  R_and_D_t / Delta_APL_t_plus_2 = 3260 := 
by
  sorry

end R_and_D_expense_corresponding_to_productivity_increase_l1575_157595


namespace calculate_allocations_l1575_157550

variable (new_revenue : ℝ)
variable (ratio_employee_salaries ratio_stock_purchases ratio_rent ratio_marketing_costs : ℕ)

theorem calculate_allocations :
  let total_ratio := ratio_employee_salaries + ratio_stock_purchases + ratio_rent + ratio_marketing_costs
  let part_value := new_revenue / total_ratio
  let employee_salary_alloc := ratio_employee_salaries * part_value
  let rent_alloc := ratio_rent * part_value
  let marketing_costs_alloc := ratio_marketing_costs * part_value
  employee_salary_alloc + rent_alloc + marketing_costs_alloc = 7800 :=
by
  sorry

end calculate_allocations_l1575_157550


namespace slope_of_tangent_at_0_l1575_157584

theorem slope_of_tangent_at_0 (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (2 * x)) : 
  (deriv f 0) = 2 :=
sorry

end slope_of_tangent_at_0_l1575_157584


namespace distance_AB_l1575_157500

def C1_polar (ρ θ : Real) : Prop :=
  ρ = 2 * Real.cos θ

def C2_polar (ρ θ : Real) : Prop :=
  ρ^2 * (1 + (Real.sin θ)^2) = 2

def ray_polar (θ : Real) : Prop :=
  θ = Real.pi / 6

theorem distance_AB :
  let ρ1 := 2 * Real.cos (Real.pi / 6)
  let ρ2 := Real.sqrt 10 * 2 / 5
  |ρ1 - ρ2| = Real.sqrt 3 - (2 * Real.sqrt 10) / 5 :=
by
  sorry

end distance_AB_l1575_157500


namespace g_g_g_g_of_2_eq_242_l1575_157544

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 3 * x + 2

theorem g_g_g_g_of_2_eq_242 : g (g (g (g 2))) = 242 :=
by
  sorry

end g_g_g_g_of_2_eq_242_l1575_157544
