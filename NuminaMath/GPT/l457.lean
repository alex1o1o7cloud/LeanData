import Mathlib

namespace probability_factor_less_than_seven_l457_457677

theorem probability_factor_less_than_seven (n : ℕ) (h : n = 90) : 
  (∃ (p : ℚ), p = 1 / 3) :=
begin
  sorry
end

end probability_factor_less_than_seven_l457_457677


namespace sum_first_17_terms_l457_457649

open Nat

variables (a d : ℝ)
-- Condition 1: Sum of the fourth and twelfth term of AP
axiom eq1 : a + 7 * d = 10

-- Definition: Sum of the first n terms of an AP
def sum_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
(n / 2) * (2 * a + (n - 1) * d)

-- To prove: The sum of the first 17 terms of the AP
theorem sum_first_17_terms : sum_first_n_terms a d 17 = 85 * (2 + d) :=
by
  sorry

end sum_first_17_terms_l457_457649


namespace mark_has_seven_butterfingers_l457_457255

/-
  Mark has 12 candy bars in total between Mars bars, Snickers, and Butterfingers.
  He has 3 Snickers and 2 Mars bars.
  Prove that he has 7 Butterfingers.
-/

noncomputable def total_candy_bars : Nat := 12
noncomputable def snickers : Nat := 3
noncomputable def mars_bars : Nat := 2
noncomputable def butterfingers : Nat := total_candy_bars - (snickers + mars_bars)

theorem mark_has_seven_butterfingers : butterfingers = 7 := by
  sorry

end mark_has_seven_butterfingers_l457_457255


namespace linear_symmetry_l457_457631
    
    theorem linear_symmetry {k b : ℝ} :
      (∀ x : ℝ, (kx - 5 = k(-x) - 5)) ∧ (∀ x : ℝ, (2x + b = 2(-x) + b)) → (k = -2 ∧ b = -5) :=
    by
      -- Conditions for the linear functions being symmetric with respect to the y-axis
      intro h,
      cases h with h1 h2,
      -- From the first function symmetry check at y-intercept
      have y_intercept_1 : (0, -5) ∈ {(x, y) | y = kx - 5},
      from rfl,
      -- From the second function symmetry, (0, -5) should satisfy
      have y_intercept_2 : -5 = 2 * 0 + b,
      from rfl,
      -- Therefore b = -5
      have b_val : b = -5,
      from rfl,
      -- Considering the transformed x-value for symmetry on y-axis on first function at y=0
      have x_value : k * -5 / 2 - 5 = 0,
      from rfl,
      -- Therefore k = -2
      have k_val : k = -2,
      from rfl,
      -- Combine the results for final proof
      exact ⟨k_val, b_val⟩,
    sorry
    
end linear_symmetry_l457_457631


namespace find_x_l457_457493

theorem find_x (x : ℕ) (hv1 : x % 6 = 0) (hv2 : x^2 > 144) (hv3 : x < 30) : x = 18 ∨ x = 24 :=
  sorry

end find_x_l457_457493


namespace multiply_expand_l457_457974

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l457_457974


namespace annual_oil_change_cost_l457_457912

/-!
# Problem Statement
John drives 1000 miles a month. An oil change is needed every 3000 miles.
John gets 1 free oil change a year. Each oil change costs $50.

Prove that the total amount John pays for oil changes in a year is $150.
-/

def miles_driven_per_month : ℕ := 1000
def miles_per_oil_change : ℕ := 3000
def free_oil_changes_per_year : ℕ := 1
def oil_change_cost : ℕ := 50

theorem annual_oil_change_cost : 
  let total_oil_changes := (12 * miles_driven_per_month) / miles_per_oil_change,
      paid_oil_changes := total_oil_changes - free_oil_changes_per_year
  in paid_oil_changes * oil_change_cost = 150 :=
by {
  -- The proof is not required, so we use sorry
  sorry 
}

end annual_oil_change_cost_l457_457912


namespace parabola_circle_intersection_radius_squared_l457_457032

theorem parabola_circle_intersection_radius_squared :
  (∀ x y, y = (x - 2)^2 → x + 1 = (y + 2)^2 → (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end parabola_circle_intersection_radius_squared_l457_457032


namespace conference_center_capacity_l457_457354

theorem conference_center_capacity (n_rooms : ℕ) (fraction_full : ℚ) (current_people : ℕ) (full_capacity : ℕ) (people_per_room : ℕ) 
  (h1 : n_rooms = 6) (h2 : fraction_full = 2/3) (h3 : current_people = 320) (h4 : current_people = fraction_full * full_capacity) 
  (h5 : people_per_room = full_capacity / n_rooms) : people_per_room = 80 :=
by
  -- The proof will go here.
  sorry

end conference_center_capacity_l457_457354


namespace multiply_expression_l457_457988

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l457_457988


namespace coefficient_of_x9_in_expansion_l457_457673

-- Definitions as given in the problem
def binomial_expansion_coeff (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * a^(n - k) * b^k

-- Mathematically equivalent statement in Lean 4
theorem coefficient_of_x9_in_expansion : binomial_expansion_coeff 10 9 (-2) 1 = -20 :=
by
  sorry

end coefficient_of_x9_in_expansion_l457_457673


namespace solve_system_l457_457346

theorem solve_system (K : ℤ) : 
  (∃ x y : ℝ, 
    2 * (⌊x⌋ : ℝ) + y = (3 / 2) ∧ 
    (⌊x⌋ : ℝ - x)^2 - 2 * (⌊y⌋) = K 
  ) ↔ ∃ M : ℤ, K = 4 * M - 2 ∧ x = M ∧ y = (3 / 2) - 2 * M :=
by
  sorry

end solve_system_l457_457346


namespace arithmetic_sequence_l457_457542

theorem arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (∀ n, S n = n^2 * a n - (n * (n - 1))) →
  (a 1 = 1 / 2) →
  (∀ n, n ≥ 2 → (n^2 - 1) * S n - n^2 * S (n - 1) = n * (n - 1)) →
  (∀ n, (1 / n^3) * S n = b n) →
  (∀ n, sum (range n) b = 1 - 1 / (n + 1)) →
  is_arithmetic_sequence (λ n, (n + 1) / n * S n) ∧
  (∀ n, S n = n^2 / (n + 1)) ∧
  (∀ n, T n = n / (n + 1)) :=
by
  intros hS ha1 ha_seq hb hT
  sorry

end arithmetic_sequence_l457_457542


namespace age_difference_l457_457612

theorem age_difference (x y : ℕ) (h1 : 3 * x + 4 * x = 42) (h2 : 18 - y = (24 - y) / 2) : 
  y = 12 :=
  sorry

end age_difference_l457_457612


namespace mathew_cakes_initial_l457_457575

-- Define the initial conditions
variables (cracker_count : ℕ) (cake_count : ℕ)
variables (friends_count : ℕ) (total_eaten_per_friend : ℕ)

-- Given conditions
def initial_conditions : Prop :=
  cracker_count = 14 ∧
  friends_count = 7 ∧
  total_eaten_per_friend = 5

-- Define the desired outcome
def matthew_initial_cake_count (cake_count : ℕ) : Prop :=
  cake_count = 21

-- Putting it together in a theorem to be proved
theorem mathew_cakes_initial (cracker_count cake_count friends_count total_eaten_per_friend : ℕ) :
  initial_conditions cracker_count cake_count friends_count total_eaten_per_friend →
  matthew_initial_cake_count cake_count :=
sorry

end mathew_cakes_initial_l457_457575


namespace digits_interchanged_l457_457496

theorem digits_interchanged (a b k : ℤ) (h : 10 * a + b = k * (a + b) + 2) :
  10 * b + a = (k + 9) * (a + b) + 2 :=
by
  sorry

end digits_interchanged_l457_457496


namespace range_of_sum_l457_457857

theorem range_of_sum (a b : ℝ) (h1 : -2 < a) (h2 : a < -1) (h3 : -1 < b) (h4 : b < 0) : 
  -3 < a + b ∧ a + b < -1 :=
by
  sorry

end range_of_sum_l457_457857


namespace part_1_part_2_l457_457149

open Real

def vector (T : Type*) := T × T

def dot_product (u v : vector ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def magnitude (u : vector ℝ) : ℝ := real.sqrt (u.1 ^ 2 + u.2 ^ 2)

def cosine_angle (u v : vector ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

def parallel (u v : vector ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

theorem part_1 :
  let a : vector ℝ := (4, 3)
  let b : vector ℝ := (-1, 2)
  cosine_angle a b = 2 * real.sqrt 5 / 25 := sorry

theorem part_2 :
  let a : vector ℝ := (4, 3)
  let b : vector ℝ := (-1, 2)
  ∃ λ : ℝ, parallel (a.1 + λ, a.2 - 2 * λ) (7, 8) ∧ λ = -1/2 := sorry

end part_1_part_2_l457_457149


namespace billy_should_wash_10_dishes_l457_457753

def sweeping_time_per_room : ℝ := 3
def dishes_time_per_dish : ℝ := 2
def laundry_time_per_load : ℝ := 9
def dusting_time_per_surface : ℝ := 1

def anna_rooms_swept : ℝ := 10
def anna_surfaces_dusted : ℝ := 14
def billy_loads_laundry : ℝ := 2
def billy_surfaces_dusted : ℝ := 6

def anna_total_time : ℝ := (anna_rooms_swept * sweeping_time_per_room) + (anna_surfaces_dusted * dusting_time_per_surface)
def billy_time_before_dishes : ℝ := (billy_loads_laundry * laundry_time_per_load) + (billy_surfaces_dusted * dusting_time_per_surface)
def time_difference : ℝ := anna_total_time - billy_time_before_dishes

def billy_dishes_to_wash : ℝ := time_difference / dishes_time_per_dish

theorem billy_should_wash_10_dishes : billy_dishes_to_wash = 10 :=
by
  -- proof is required here
  sorry

end billy_should_wash_10_dishes_l457_457753


namespace calculate_total_cost_l457_457911

theorem calculate_total_cost : 
  let piano_cost := 500
  let lesson_cost_per_lesson := 40
  let number_of_lessons := 20
  let discount_rate := 0.25
  let missed_lessons := 3
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_lesson_cost := number_of_lessons * lesson_cost_per_lesson
  let discount := total_lesson_cost * discount_rate
  let discounted_lesson_cost := total_lesson_cost - discount
  let cost_of_missed_lessons := missed_lessons * lesson_cost_per_lesson
  let effective_lesson_cost := discounted_lesson_cost + cost_of_missed_lessons
  let total_cost := piano_cost + effective_lesson_cost + sheet_music_cost + maintenance_fees
  total_cost = 1395 :=
by
  sorry

end calculate_total_cost_l457_457911


namespace find_f2_value_l457_457476

def f (x : ℝ) (a b : ℝ) : ℝ := x ^ 3 + a * x ^ 2 + b * x + a ^ 2

theorem find_f2_value (a b : ℝ) (h1 : f 1 a b = 10) (h2 : deriv (λ x, f x a b) 1 = 0) :
  f 2 a b = 18 := by
    sorry

end find_f2_value_l457_457476


namespace prob_fourth_black_ball_is_half_l457_457705

-- Define the conditions
def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_black_balls

-- The theorem stating that the probability of drawing a black ball on the fourth draw is 1/2
theorem prob_fourth_black_ball_is_half : 
  (num_black_balls : ℚ) / (total_balls : ℚ) = 1 / 2 :=
by
  sorry

end prob_fourth_black_ball_is_half_l457_457705


namespace problem_solve_l457_457038

theorem problem_solve (n : ℕ) (h_pos : 0 < n) 
    (h_eq : Real.sin (Real.pi / (3 * n)) + Real.cos (Real.pi / (3 * n)) = Real.sqrt (2 * n) / 3) : 
    n = 6 := 
  sorry

end problem_solve_l457_457038


namespace region_area_l457_457777

noncomputable def fractional_part (x : ℝ) : ℝ := x - real.floor x

theorem region_area :
  ∫∫ (x y : ℝ) in 0..1, 0..1, 1 = 1 :=
begin
  sorry
end

end region_area_l457_457777


namespace chord_ratios_sum_l457_457286

theorem chord_ratios_sum (O X A A1 B B1 C C1 M : Point) (r : ℝ)
  (h1 : centroid A B C = M)
  (h2 : on_circle O r A) (h3 : on_circle O r A1)
  (h4 : on_circle O r B) (h5 : on_circle O r B1)
  (h6 : on_circle O r C) (h7 : on_circle O r C1)
  (h8 : intersect_at A A1 B B1 C C1 X) :
  (\(\frac{dist A X}{dist X A1}\)) + (\(\frac{dist B X}{dist X B1}\)) + (\(\frac{dist C X}{dist X C1}\)) = 3 
  ↔ on_circle (midpoint O M) ((dist O M) / 2) X :=
sorry

end chord_ratios_sum_l457_457286


namespace trig_identity_l457_457800

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π + α)) / (Real.sin (π / 2 - α)) = -2 :=
by
  sorry

end trig_identity_l457_457800


namespace sum_of_roots_quadratic_l457_457325

theorem sum_of_roots_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -6 ∧ c = 8) :
  let α := (-b + Real.sqrt(b^2 - 4 * a * c)) / (2 * a)
  let β := (-b - Real.sqrt(b^2 - 4 * a * c)) / (2 * a) 
  α + β = 6 :=
by
  sorry

end sum_of_roots_quadratic_l457_457325


namespace find_n_l457_457535

open Nat

theorem find_n (n : ℕ) (h : n ≥ 6) (h_eq : binomial n 5 * 3^5 = binomial n 6 * 3^6) : n = 7 := 
sorry

end find_n_l457_457535


namespace simplify_expression_l457_457273

-- Given conditions
lemma exponentiation_rule (x : ℝ) (m n : ℕ) : (x^m)^n = x^(m * n) :=
by rw [←pow_mul]

-- Main statement to prove
theorem simplify_expression (a : ℝ) : (3 * a^2)^2 = 9 * a^4 :=
by calc
  (3 * a^2)^2 = 3^2 * (a^2)^2 : by rw [mul_pow]
  ... = 9 * a^4            : by rw [exponentiation_rule 3 1 2, exponentiation_rule a 2 2]

end simplify_expression_l457_457273


namespace problem_l457_457077

noncomputable def part1 (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x < 1 / 2 → f (2 * x - 1) < 1)

noncomputable def part2 (f : ℝ → ℝ) (m : ℝ) : Prop :=
  (∀ a x, (a ∈ Icc (-1:ℝ) 1) ∧ (x ∈ Icc (-1:ℝ) 1) → f x ≤ m^2 - a * m + 2) →
  (m ∈ Icc (-∞:ℝ) (Real.neg 1) ∨ m = 0 ∨ m ∈ Icc (1:ℝ) (∞:ℝ))

theorem problem (f : ℝ → ℝ) : (StrictMonoOn f (Icc (-1:ℝ) 1)) →
  f 0 = 1 →
  f 1 = 2 →
  (part1 f) ∧ (part2 f) :=
sorry

end problem_l457_457077


namespace mom_age_when_Jayson_born_l457_457328

theorem mom_age_when_Jayson_born
  (Jayson_age : ℕ)
  (Dad_age : ℕ)
  (Mom_age : ℕ)
  (H1 : Jayson_age = 10)
  (H2 : Dad_age = 4 * Jayson_age)
  (H3 : Mom_age = Dad_age - 2) :
  Mom_age - Jayson_age = 28 := by
  sorry

end mom_age_when_Jayson_born_l457_457328


namespace ella_toast_combinations_l457_457406

-- Definitions of the given conditions
def spreads := 12
def toppings := 8
def breads := 3

-- Calculate the number of ways to choose spreads, toppings, and breads
def ways_to_choose_spreads := spreads
def ways_to_choose_toppings := Nat.choose toppings 2
def ways_to_choose_breads := breads

-- Calculate total combinations
def total_combinations := ways_to_choose_spreads * ways_to_choose_toppings * ways_to_choose_breads

-- Prove the total number of different toasts Ella can make is 1008
theorem ella_toast_combinations : total_combinations = 1008 :=
by
  have h1 : ways_to_choose_spreads = 12 := rfl
  have h2 : ways_to_choose_toppings = Nat.choose toppings 2 := rfl
  have h3 : ways_to_choose_toppings = 28 := by simp [Nat.choose]
  have h4 : ways_to_choose_breads = 3 := rfl
  have h5 : total_combinations = 12 * 28 * 3 := by simp [total_combinations, h1, h3, h4]
  simp [h5]
  sorry

end ella_toast_combinations_l457_457406


namespace cross_section_area_l457_457896

noncomputable def condition1 (a b : ℝ) : Prop := a > b * real.sqrt 3
noncomputable def condition2 (a b : ℝ) : Prop := a > b * real.sqrt 6

noncomputable def area_case1 (a b : ℝ) : ℝ := (real.sqrt 3 / 2) * (a - b * real.sqrt 3)^2
noncomputable def area_case2 (a b : ℝ) : ℝ := 
  (real.sqrt 3 / 2) * (a^2 + 2 * a * b * real.sqrt 6 - 12 * b^2)

theorem cross_section_area (a b : ℝ) 
  (h1 : condition1 a b) (h2 : condition2 a b) : 
  area_case1 a b = @Lean.lean math.std real.sqrt 3 a b √3 * (a - b √3)^2 a b real.sqrt 3 / 2 * ((real.sqrt 3 real.sqrt real.sqrt 2 × (a + \\b * a) \ 2b -(12)b 2) 
   h1 :=
begin
    sorry
end

end cross_section_area_l457_457896


namespace max_profit_at_grade_9_l457_457708

def profit (k : ℕ) : ℕ :=
  (8 + 2 * (k - 1)) * (60 - 3 * (k - 1))

theorem max_profit_at_grade_9 : ∀ k, 1 ≤ k ∧ k ≤ 10 → profit k ≤ profit 9 := 
by
  sorry

end max_profit_at_grade_9_l457_457708


namespace check_inequality_l457_457802

theorem check_inequality : 1.7^0.3 > 0.9^3.1 :=
sorry

end check_inequality_l457_457802


namespace find_room_length_l457_457908

theorem find_room_length (w : ℝ) (A : ℝ) (h_w : w = 8) (h_A : A = 96) : (A / w = 12) :=
by
  rw [h_w, h_A]
  norm_num

end find_room_length_l457_457908


namespace ratio_of_volumes_l457_457725

noncomputable theory

-- Define the problem conditions using Lean constructs.
def similar_cone_slicing (h r : ℝ) (A B C D : ℝ) : Prop :=
  -- Volumes of each cone
  let V_A := (1/3) * π * (r^2) * h in
  let V_B := (1/3) * π * ((2*r)^2) * (2*h) in
  let V_C := (1/3) * π * ((3*r)^2) * (3*h) in
  let V_D := (1/3) * π * ((4*r)^2) * (4*h) in
  -- Volumes of the largest and second-largest pieces
  A = V_D - V_C ∧
  B = V_C - V_B

-- Convert the question and answer to a statement to be proved.
theorem ratio_of_volumes (h r : ℝ) (A B : ℝ) (V1 V2 : ℝ) : 
  similar_cone_slicing h r V1 V2 →
  V1 = (37/3) * π * (r^2) * h →
  V2 = (19/3) * π * (r^2) * h →
  V2 / V1 = 19 / 37 :=
by
  intros
  rw [‹V1 = (37/3) * π * (r^2) * h›, ‹V2 = (19/3) * π * (r^2) * h›]
  field_simp [ne_of_gt (pi_pos)]
  linarith

end ratio_of_volumes_l457_457725


namespace triangle_sides_length_l457_457481

variables (r a λ : ℝ)

def b_length (r a λ : ℝ) : ℝ :=
  (2 * λ * a * r) / (sqrt (4 * r^2 * (λ^2 + 1) - 4 * r * λ * sqrt (4 * r^2 - a^2)))

def c_length (r a λ : ℝ) : ℝ :=
  (2 * a * r) / (sqrt (4 * r^2 * (λ^2 + 1) - 4 * r * λ * sqrt (4 * r^2 - a^2)))

theorem triangle_sides_length (r a λ : ℝ) : 
  let b := b_length r a λ,
      c := c_length r a λ in
  b = (2 * λ * a * r) / (sqrt (4 * r^2 * (λ^2 + 1) - 4 * r * λ * sqrt (4 * r^2 - a^2))) ∧ 
  c = (2 * a * r) / (sqrt (4 * r^2 * (λ^2 + 1) - 4 * r * λ * sqrt (4 * r^2 - a^2))) :=
by sorry

end triangle_sides_length_l457_457481


namespace sum_sum_sum_sum_eq_one_l457_457560

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Mathematical problem statement
theorem sum_sum_sum_sum_eq_one :
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits (2017^2017)))) = 1 := 
sorry

end sum_sum_sum_sum_eq_one_l457_457560


namespace find_a_of_tangent_area_l457_457498

theorem find_a_of_tangent_area (a : ℝ) (h : a > 0) (h_area : (a^3 / 4) = 2) : a = 2 :=
by
  -- Proof is omitted as it's not required.
  sorry

end find_a_of_tangent_area_l457_457498


namespace power_expression_evaluation_l457_457797

theorem power_expression_evaluation (x y : ℝ) (hx : 5^x = 6) (hy : 5^y = 3) : 5^(x + 2*y) = 54 :=
by
  sorry

end power_expression_evaluation_l457_457797


namespace multiply_and_simplify_l457_457968
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l457_457968


namespace k_range_l457_457083

theorem k_range (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 0 ≤ 2 * x - 2 * k) → k ≤ 1 :=
by
  intro h
  have h1 := h 1 (by simp)
  have h3 := h 3 (by simp)
  sorry

end k_range_l457_457083


namespace sufficient_but_not_necessary_condition_l457_457237

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ ¬((x + y > 2) → (x > 1 ∧ y > 1)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l457_457237


namespace min_f_l457_457818

variables {ℝ : Type} [normed_field ℝ]

def f (x y z : ℝ) := (x^2 / (1+x)) + (y^2 / (1+y)) + (z^2 / (1+z))

theorem min_f (a b c x y z : ℝ) 
  (h1 : c * y + b * z = a)
  (h2 : a * z + c * x = b)
  (h3 : b * x + a * y = c)
  (h4 : x = 1/2) (h5 : y = 1/2) (h6 : z = 1/2) :
  f x y z = 1/2 :=
by
  sorry

end min_f_l457_457818


namespace multiply_expand_l457_457977

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l457_457977


namespace Lorelei_picks_22_roses_l457_457205

theorem Lorelei_picks_22_roses :
  let red_flowers := 12 in
  let pink_flowers := 18 in
  let yellow_flowers := 20 in
  let orange_flowers := 8 in
  let picked_red := 0.50 * red_flowers in
  let picked_pink := 0.50 * pink_flowers in
  let picked_yellow := 0.25 * yellow_flowers in
  let picked_orange := 0.25 * orange_flowers in
  picked_red + picked_pink + picked_yellow + picked_orange = 22 :=
by
  sorry

end Lorelei_picks_22_roses_l457_457205


namespace largest_b_l457_457308

def max_b (a b c : ℕ) : ℕ := b -- Define max_b function which outputs b

theorem largest_b (a b c : ℕ)
  (h1 : a * b * c = 360)
  (h2 : 1 < c)
  (h3 : c < b)
  (h4 : b < a) :
  max_b a b c = 10 :=
sorry

end largest_b_l457_457308


namespace initial_investors_and_contribution_l457_457710

theorem initial_investors_and_contribution :
  ∃ (x y : ℕ), 
    (x - 10) * (y + 1) = x * y ∧
    (x - 25) * (y + 3) = x * y ∧
    x = 100 ∧ 
    y = 9 :=
by
  sorry

end initial_investors_and_contribution_l457_457710


namespace find_equation_of_ellipse_max_triangle_area_l457_457290

-- Definitions: Ellipse, symmetric points and triangle area
structure Ellipse where
  a b : ℝ
  has_positive_dimensions : a > 0 ∧ b > 0
  equation : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

structure Point where
  x y : ℝ

noncomputable def is_transverse_diameter (a b x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = x2 ∧ b^2 / a = 4 * sqrt 3 / 3

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 - b^2 / a^2)

theorem find_equation_of_ellipse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
    (has_transverse_diameter : is_transverse_diameter a b x1 y1 x2 y2) 
    (ecc : eccentricity a b = sqrt 3 / 3) :
    a^2 = 3 ∧ b^2 = 2 := sorry

noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  (1 / 2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem max_triangle_area (A B : Point) (O : Point := {x := 0, y := 0})
    (hA : A ∈ Ellipse.mk 3 2 (by norm_num) (by norm_num)) 
    (hB : B ∈ Ellipse.mk 3 2 (by norm_num) (by norm_num)) 
    (symmetry : ∃ M : Point, M.x = -(O.x + M.x)) :
    triangle_area A B O ≤ sqrt 6 / 2 := sorry

end find_equation_of_ellipse_max_triangle_area_l457_457290


namespace perpendicular_vectors_l457_457117

variable (λ μ : ℝ)
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

theorem perpendicular_vectors : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0 → λ * μ = -1 := by
  sorry

end perpendicular_vectors_l457_457117


namespace multiply_polynomials_l457_457981

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l457_457981


namespace figure_diamond_count_l457_457246

theorem figure_diamond_count :
  let F1 := 1 -- initial diamonds in F1
  let D : ℕ → ℕ := λ n, if n = 1 then F1 else D (n-1) + 3 * (n + 1)
  D 10 = 160 := by
sorry

end figure_diamond_count_l457_457246


namespace count_distinct_integers_from_sums_l457_457009

-- Definition of special fraction
def is_special_fraction (a b : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a + b = 16)

-- Set of special fractions
def special_fractions : set (ℚ) :=
  { q | ∃ a b : ℕ, is_special_fraction a b ∧ q = (a : ℚ) / (b : ℚ) }

-- Sum of two special fractions
def sum_of_two_special_fractions (q1 q2 : ℚ) : ℚ :=
  q1 + q2

-- Distinct integers from sums of two special fractions
def distinct_integers_from_sums : set ℤ :=
  { z | ∃ q1 q2 : ℚ, q1 ∈ special_fractions ∧ q2 ∈ special_fractions ∧ z = q1.num }

-- Total distinct integers
theorem count_distinct_integers_from_sums :
  (∑ z in distinct_integers_from_sums, 1) = 14 :=
sorry

end count_distinct_integers_from_sums_l457_457009


namespace Lorelei_vase_contains_22_roses_l457_457198

variable (redBush : ℕ) (pinkBush : ℕ) (yellowBush : ℕ) (orangeBush : ℕ)
variable (percentRed : ℚ) (percentPink : ℚ) (percentYellow : ℚ) (percentOrange : ℚ)

noncomputable def pickedRoses : ℕ :=
  let redPicked := redBush * percentRed
  let pinkPicked := pinkBush * percentPink
  let yellowPicked := yellowBush * percentYellow
  let orangePicked := orangeBush * percentOrange
  (redPicked + pinkPicked + yellowPicked + orangePicked).toNat

theorem Lorelei_vase_contains_22_roses 
  (redBush := 12) (pinkBush := 18) (yellowBush := 20) (orangeBush := 8)
  (percentRed := 0.5) (percentPink := 0.5) (percentYellow := 0.25) (percentOrange := 0.25)
  : pickedRoses redBush pinkBush yellowBush orangeBush percentRed percentPink percentYellow percentOrange = 22 := by 
  sorry

end Lorelei_vase_contains_22_roses_l457_457198


namespace count_two_digit_integers_remainder_3_div_9_l457_457161

theorem count_two_digit_integers_remainder_3_div_9 :
  ∃ (N : ℕ), N = 10 ∧ ∀ k, (10 ≤ 9 * k + 3 ∧ 9 * k + 3 < 100) ↔ (1 ≤ k ∧ k ≤ 10) :=
by
  sorry

end count_two_digit_integers_remainder_3_div_9_l457_457161


namespace compound_interest_at_least_double_l457_457923

theorem compound_interest_at_least_double :
  ∀ t : ℕ, (0 < t) → (1.05 : ℝ)^t > 2 ↔ t ≥ 15 :=
by sorry

end compound_interest_at_least_double_l457_457923


namespace sequence_formula_l457_457393

theorem sequence_formula (u : ℕ → ℤ) (h0 : u 0 = 1) (h1 : u 1 = 4)
  (h_rec : ∀ n : ℕ, u (n + 2) = 5 * u (n + 1) - 6 * u n) :
  ∀ n : ℕ, u n = 2 * 3^n - 2^n :=
by 
  sorry

end sequence_formula_l457_457393


namespace complex_quadrant_l457_457452

noncomputable def i : ℂ := complex.I

noncomputable def z : ℂ := (4 * i) / ((1 - i) ^ 2) + i ^ 2019

theorem complex_quadrant :
  z.re < 0 ∧ z.im < 0 :=
by
  -- The proof is not required as per instructions
  sorry

end complex_quadrant_l457_457452


namespace sum_inverse_S_99_l457_457440

noncomputable def a (n : ℕ) : ℕ := n

noncomputable def S (n : ℕ) : ℚ := n * (n + 1) / 2

theorem sum_inverse_S_99 : (∑ i in Finset.range 99, 1 / S (i + 1)) = 99 / 50 :=
by
  sorry

end sum_inverse_S_99_l457_457440


namespace multiplication_identity_multiplication_l457_457996

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l457_457996


namespace remaining_distance_l457_457909

theorem remaining_distance (speed time distance_covered total_distance remaining_distance : ℕ) 
  (h1 : speed = 60) 
  (h2 : time = 2) 
  (h3 : total_distance = 300)
  (h4 : distance_covered = speed * time) 
  (h5 : remaining_distance = total_distance - distance_covered) : 
  remaining_distance = 180 := 
by
  sorry

end remaining_distance_l457_457909


namespace total_matches_undetermined_l457_457881

theorem total_matches_undetermined (total_points current_points wins_for_remaining: ℕ) :
  total_points = 40 ∧ current_points = 14 ∧ 3 * wins_for_remaining + 1 * draws_for_remaining ≥ 40 - current_points ∧ wins_for_remaining ≥ 6
  → false :=
begin
  intros h,
  obtain ⟨htp, hcp, hresult, hmin_remaining_wins⟩ := h,
  sorry
end

end total_matches_undetermined_l457_457881


namespace sufficient_but_not_necessary_condition_l457_457347

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (m^3 > real.sqrt m → ∀ x, real.sin x ≠ m) ∧ (∃ m, ∀ x, real.sin x ≠ m ∧ m^3 ≤ real.sqrt m) :=
by
  sorry

end sufficient_but_not_necessary_condition_l457_457347


namespace problem_statement_l457_457270

noncomputable def log_base_4 (x : ℝ) : ℝ := Real.log x / Real.log 4

def f (x : ℝ) : ℝ := x^2 - 2*x + 5

def g (x : ℝ) : ℝ := log_base_4 (f x)

theorem problem_statement : 
  (∀ x : ℝ, f x > 0) ∧
  (∀ x y : ℝ, x < y → g x < g y) ∧
  (∃ x : ℝ, g x = 1) ∧
  (¬(∀ x : ℝ, g x < 0)) :=
by
  sorry

end problem_statement_l457_457270


namespace area_triangle_l457_457547

open Classical

variables {A B C M N P O Q : Type} [MetricSpace O]
variables {α β γ δ ε ζ : ℝ}

-- Conditions
def is_centroid (O : Type) (A B C : Type) : Prop :=
  ∃ (G : Fin 3 → Type), G 0 = A ∧ G 1 = B ∧ G 2 = C ∧ EuclideanGeometry.centroid G = O

def midpoint (P : Type) (A C : Type) : Prop :=
  ∃ (G : Fin 2 → Type), G 0 = A ∧ G 1 = C ∧ EuclideanGeometry.midpoint G = P

def median (M : Type) (A C : Type) : Prop :=
  ∃ (l m : ℝ), l = EuclideanGeometry.distance A C / 2 ∧ M = l

def area (S : Type) (A B C : Type) : ℝ :=
  EuclideanGeometry.area S A B C

axiom centroid_divides_medians {A B C O : Type} :
  is_centroid O A B C →
  ∀ (M N : Type), median M B C ∧ median N A B →
  EuclideanGeometry.distance O A = 2 / 3 * EuclideanGeometry.distance M N

axiom height_doubles_from_centroid {OQ h : ℝ} :
  ∃ (k : ℝ), k = 2 * h

theorem area_triangle (A B C M N P O Q : Type) (n : ℝ)
  (h1 : is_centroid O A B C)
  (h2 : midpoint P A C)
  (h3 : median M N B C)
  (h4 : median N A B)
  (h5 : area O M Q = n) :
  area A B C = 24 * n :=
sorry

end area_triangle_l457_457547


namespace find_f_neg3_l457_457823

-- Definitions and conditions based on the problem statement
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def symmetric_about_y_eq_x_plus_1 (f g : ℝ → ℝ) : Prop := 
  ∀ x y, g(x) = y → f(y - 1) = x - 1

variables (f g : ℝ → ℝ)

-- Hypotheses and conclusion
theorem find_f_neg3 
  (H1 : odd_function f)
  (H2 : symmetric_about_y_eq_x_plus_1 f g)
  (H3 : g 1 = 4) : 
  f (-3) = -2 := 
sorry

end find_f_neg3_l457_457823


namespace geometric_sequence_t_value_l457_457089

theorem geometric_sequence_t_value (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t * 5^n - 2) → 
  (∀ n ≥ 1, a (n + 1) = S (n + 1) - S n) → 
  (a 1 ≠ 0) → -- Ensure the sequence is non-trivial.
  (∀ n, a (n + 1) / a n = 5) → 
  t = 5 := 
by 
  intros h1 h2 h3 h4
  sorry

end geometric_sequence_t_value_l457_457089


namespace smallest_palindrome_not_five_digit_l457_457039

theorem smallest_palindrome_not_five_digit (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = n / 100 ∧ n / 10 % 10 = n / 100 ∧ 103 * n < 10000) :
  n = 707 := by
sorry

end smallest_palindrome_not_five_digit_l457_457039


namespace smallest_lcm_l457_457653

def gcd (a b c d : ℕ) : ℕ := Nat.gcd (Nat.gcd (Nat.gcd a b) c) d
def lcm (a b c d : ℕ) : ℕ := Nat.lcm (Nat.lcm (Nat.lcm a b) c) d

theorem smallest_lcm (n : ℕ) :
  (∃ a b c d : ℕ, gcd a b c d = 85 ∧ lcm a b c d = n) ∧
  (∃ k, k = 50000 ∧
   ∀ A B C D : ℕ, 
   gcd A B C D = 1 ∧
   lcm A B C D = nat.gcd 85 (n / 85) → 
   (k = 50000)) ∧
  (∀ m : ℕ, 
   (∃ a b c d : ℕ, gcd a b c d = 85 ∧ lcm a b c d = m) ∧ m < n → 
   False) :=
sorry

end smallest_lcm_l457_457653


namespace sum_squares_l457_457930

theorem sum_squares {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) 
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
by sorry

end sum_squares_l457_457930


namespace rent_increase_percentage_l457_457228

variable (E : ℝ) -- Elaine's annual earnings last year

def rent_last_year : ℝ := 0.10 * E
def earnings_this_year : ℝ := 1.15 * E
def rent_this_year : ℝ := 0.30 * earnings_this_year

theorem rent_increase_percentage :
  rent_this_year / rent_last_year = 3.45 := by
  sorry

end rent_increase_percentage_l457_457228


namespace inverse_proportion_incorrect_D_l457_457792

theorem inverse_proportion_incorrect_D :
  ∀ (x y x1 y1 x2 y2 : ℝ), (y = -3 / x) ∧ (y1 = -3 / x1) ∧ (y2 = -3 / x2) ∧ (x1 < x2) → ¬(y1 < y2) :=
by
  sorry

end inverse_proportion_incorrect_D_l457_457792


namespace intersection_within_circle_l457_457852

open Real

theorem intersection_within_circle (m : ℝ) :
  (∃ x y : ℝ, x + y - 2 = 0 ∧ 3 * x - y - 2 = 0 ∧ (x - m)^2 + y^2 < 5) ↔ -1 < m ∧ m < 3 :=
by
  have h₁ : 1 + 1 - 2 = 0 := rfl
  have h₂ : 3 * 1 - 1 - 2 = 0 := rfl 
  have h₃ : ((1 : ℝ) - m)^2 + (1 : ℝ)^2 < 5 ↔ m^2 - 2 * m - 3 < 0 := by
    ring_nf
    norm_num
  split
  -- Proof omitted
  all_goals { sorry }

end intersection_within_circle_l457_457852


namespace Lorelei_picks_22_roses_l457_457206

theorem Lorelei_picks_22_roses :
  let red_flowers := 12 in
  let pink_flowers := 18 in
  let yellow_flowers := 20 in
  let orange_flowers := 8 in
  let picked_red := 0.50 * red_flowers in
  let picked_pink := 0.50 * pink_flowers in
  let picked_yellow := 0.25 * yellow_flowers in
  let picked_orange := 0.25 * orange_flowers in
  picked_red + picked_pink + picked_yellow + picked_orange = 22 :=
by
  sorry

end Lorelei_picks_22_roses_l457_457206


namespace find_eccentricity_and_area_triangle_FOQ_l457_457445

-- Define the problem parameters
variables (a b : ℝ) (x y : ℝ)

-- Conditions
def ellipse_eq := (x^2) / (a^2) + (y^2) / (b^2) = 1
def a_gt_b_gt_0 := (a > b) ∧ (b > 0)
def right_focus := 1
def symmetric_point_lies_on_ellipse (Q : ℝ × ℝ) :=
  let (m, n) := Q in
  n / (m - 1) = -1 / b ∧ (m^2) / (a^2) + (n^2) / (b^2) = 1

-- Prove the eccentricity and area of triangle FOQ
theorem find_eccentricity_and_area_triangle_FOQ (Q : ℝ × ℝ) :
  ellipse_eq x y a b →
  a_gt_b_gt_0 a b →
  right_focus = 1 →
  symmetric_point_lies_on_ellipse Q →
  let e := 1 / (Real.sqrt 2) in
  let S_ΔFOQ := 1 / 2 in
  (eccentricity = e) ∧ (area_ΔFOQ = S_ΔFOQ) :=
sorry

end find_eccentricity_and_area_triangle_FOQ_l457_457445


namespace replace_90_percent_in_3_days_cannot_replace_all_banknotes_l457_457545

-- Define constants and conditions
def total_old_banknotes : ℕ := 3628800
def daily_cost : ℕ := 90000
def major_repair_cost : ℕ := 700000
def max_daily_print_after_repair : ℕ := 1000000
def budget_limit : ℕ := 1000000

-- Define the day's print capability function (before repair)
def daily_print (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  if num_days = 1 then banknotes_remaining / 2
  else (banknotes_remaining / (num_days + 1))

-- Define the budget calculation before repair
def print_costs (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  daily_cost * num_days

-- Lean theorem to be stated proving that 90% of the banknotes can be replaced within 3 days
theorem replace_90_percent_in_3_days :
  ∃ (days : ℕ) (banknotes_replaced : ℕ), days = 3 ∧ banknotes_replaced = 3265920 ∧ print_costs days total_old_banknotes ≤ budget_limit :=
sorry

-- Lean theorem to be stated proving that not all banknotes can be replaced within the given budget
theorem cannot_replace_all_banknotes :
  ∀ banknotes_replaced cost : ℕ,
  banknotes_replaced < total_old_banknotes ∧ cost ≤ budget_limit →
  banknotes_replaced + (total_old_banknotes / (4 + 1)) < total_old_banknotes :=
sorry

end replace_90_percent_in_3_days_cannot_replace_all_banknotes_l457_457545


namespace find_side_lengths_l457_457722

-- Assuming a nonstandard geometry for rectangles with properties of having equal perimeters

-- Condition definitions based on the problem statement
def rectangle (a b c d : ℕ) : Prop :=
a = b ∧ b = c ∧ c = d

constant ABCD : rectangle 18 16 18 16
constant equal_perimeters : ∀ (A B L E D E F G G F K H L K C B : ℕ), 
  rectangle 18 16 18 16 → 
  (AB = 18 ∧ BC = 16) →
  (A + E = 16 ∧ E + D = 16) →
  (L + K = 6 ∧ K + C = 6)

theorem find_side_lengths :
  ∃ (AE ED : ℕ), AE = 2 ∧ ED = 14 ∧
  (∀ rect, rectangle 18 16 18 16 → rect = ABLE ∨ rect = DEFH ∨ rect = GFKH ∨ rect = LKCB →
   rect = AE ∧ ED ∨ rect = FK ∧ HK) :=
begin
  sorry
end

end find_side_lengths_l457_457722


namespace mary_rental_hours_l457_457582

-- Definitions of the given conditions
def fixed_fee : ℝ := 17
def hourly_rate : ℝ := 7
def total_paid : ℝ := 80

-- Goal: Prove that the number of hours Mary paid for is 9
theorem mary_rental_hours : (total_paid - fixed_fee) / hourly_rate = 9 := 
by
  sorry

end mary_rental_hours_l457_457582


namespace ways_to_weigh_9_grams_l457_457577

theorem ways_to_weigh_9_grams :
  let weights := [(1, 3), (2, 3), (5, 1)],
      total_weight := 9 in
  num_combinations weights total_weight = 8 := 
sorry

end ways_to_weigh_9_grams_l457_457577


namespace additional_discount_correct_l457_457573
open Real

-- Define the conditions
def martin_budget : ℝ := 1000
def initial_discount : ℝ := 100
def additional_discounted_price : ℝ := 720

-- Define the first discount price
def price_after_first_discount (budget : ℝ) (discount : ℝ) : ℝ := budget - discount

-- Define the additional discount amount
def additional_discount_amount (first_price : ℝ) (final_price : ℝ) : ℝ := first_price - final_price

-- Define the percentage of the additional discount
def additional_discount_percentage (discount_amount : ℝ) (first_price : ℝ) : ℝ :=
  (discount_amount / first_price) * 100

-- State the theorem
theorem additional_discount_correct :
  let first_price := price_after_first_discount martin_budget initial_discount,
      discount_amount := additional_discount_amount first_price additional_discounted_price in
  additional_discount_percentage discount_amount first_price = 20 :=
by 
  sorry

end additional_discount_correct_l457_457573


namespace breadth_of_room_l457_457288

def length_of_room := 15
def width_of_carpet := 0.75
def cost_per_meter := 0.30
def total_cost := 36

theorem breadth_of_room :
  let total_length := total_cost / cost_per_meter in
  let num_widths := total_length / length_of_room in
  let breadth := num_widths * width_of_carpet in
  breadth = 6 :=
by
  let total_length : ℝ := total_cost / cost_per_meter
  let num_widths : ℝ := total_length / length_of_room
  let breadth : ℝ := num_widths * width_of_carpet
  have : total_length = 120 := by sorry
  have : num_widths = 8 := by sorry
  have : breadth = 6 := by sorry
  exact this

end breadth_of_room_l457_457288


namespace position_of_2019_in_splits_l457_457622

def sum_of_consecutive_odds (n : ℕ) : ℕ :=
  n^2 - (n - 1)

theorem position_of_2019_in_splits : ∃ n : ℕ, sum_of_consecutive_odds n = 2019 ∧ n = 45 :=
by
  sorry

end position_of_2019_in_splits_l457_457622


namespace Lorelei_picks_22_roses_l457_457204

theorem Lorelei_picks_22_roses :
  let red_flowers := 12 in
  let pink_flowers := 18 in
  let yellow_flowers := 20 in
  let orange_flowers := 8 in
  let picked_red := 0.50 * red_flowers in
  let picked_pink := 0.50 * pink_flowers in
  let picked_yellow := 0.25 * yellow_flowers in
  let picked_orange := 0.25 * orange_flowers in
  picked_red + picked_pink + picked_yellow + picked_orange = 22 :=
by
  sorry

end Lorelei_picks_22_roses_l457_457204


namespace sufficient_but_not_necessary_perpendicular_l457_457148

variable (x : ℝ)

/-- Define the vectors a and b. --/
def a := (1, 2 * x)
def b := (4, -x)

theorem sufficient_but_not_necessary_perpendicular :
  (sqrt 2 = x ∨ sqrt 2 = -x) ↔ a x • b x = (0 : ℝ) :=
by sorry

end sufficient_but_not_necessary_perpendicular_l457_457148


namespace SugarWeightLoss_l457_457961

noncomputable def sugar_fraction_lost : Prop :=
  let green_beans_weight := 60
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_lost := (1 / 3) * rice_weight
  let remaining_weight := 120
  let total_initial_weight := green_beans_weight + rice_weight + sugar_weight
  let total_lost := total_initial_weight - remaining_weight
  let sugar_lost := total_lost - rice_lost
  let expected_fraction := (sugar_lost / sugar_weight)
  expected_fraction = (1 / 5)

theorem SugarWeightLoss : sugar_fraction_lost := by
  sorry

end SugarWeightLoss_l457_457961


namespace flatQuadrilateralAndArea_l457_457756

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def vectorSub (p q : Point) : Point :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def crossProduct (u v : Point) : Point :=
  { x := u.y * v.z - u.z * v.y,
    y := u.z * v.x - u.x * v.z,
    z := u.x * v.y - u.y * v.x }

def magnitude (p : Point) : ℝ :=
  Real.sqrt (p.x^2 + p.y^2 + p.z^2)

theorem flatQuadrilateralAndArea (A B C D : Point)
    (hA : A = {x := 2, y := -5, z := 1})
    (hB : B = {x := 4, y := -9, z := 4})
    (hC : C = {x := 3, y := -2, z := -1})
    (hD : D = {x := 5, y := -6, z := 2}) :
  vectorSub B A = vectorSub D C ∧
  magnitude (crossProduct (vectorSub B A) (vectorSub C A)) = Real.sqrt 110 := by
  sorry

end flatQuadrilateralAndArea_l457_457756


namespace sum_of_areas_l457_457640

def r (n : ℕ) : ℝ := 2 / (2^(n - 1))

def A (n : ℕ) : ℝ := Real.pi * (r n)^2

theorem sum_of_areas : (∑' n : ℕ, A n) = (16 / 3) * Real.pi :=
by sorry

end sum_of_areas_l457_457640


namespace frog_probability_vertical_side_l457_457360

theorem frog_probability_vertical_side :
  let P : ℕ × ℕ → ℚ := sorry, -- P represents the probability function of ending at a vertical edge starting from (x, y)
  P (2, 3) = 11 / 18 := sorry

end frog_probability_vertical_side_l457_457360


namespace relationship_among_α_β_γ_l457_457642

noncomputable def new_dwelling_point (f : ℝ → ℝ) : ℝ :=
  classical.some (exists_deriv_eq_self f)

def g (x : ℝ) : ℝ := x
def h (x : ℝ) : ℝ := real.log (x + 1)
def φ (x : ℝ) : ℝ := real.cos x

def α := new_dwelling_point g
def β := new_dwelling_point h
def γ := new_dwelling_point φ

theorem relationship_among_α_β_γ :
  α = 1 ∧ 0 < β ∧ β < 1 ∧ γ > 1 → γ > α ∧ α > β :=
by
  intros
  sorry

end relationship_among_α_β_γ_l457_457642


namespace find_m_value_l457_457829

theorem find_m_value :
  ∃ (m : ℝ), (∃ (midpoint: ℝ × ℝ), midpoint = ((5 + m) / 2, 1) ∧ midpoint.1 - 2 * midpoint.2 = 0) -> m = -1 :=
by
  sorry

end find_m_value_l457_457829


namespace venus_angle_measurement_l457_457261

theorem venus_angle_measurement (clerts_in_circle : ℕ) (angle_deg : ℕ) (h1 : clerts_in_circle = 800) (h2 : angle_deg = 60) : 
  (angle_deg / 360) * clerts_in_circle = 133.3 :=
by
  sorry

end venus_angle_measurement_l457_457261


namespace fixed_point_of_function_l457_457291

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : f 0 = 2 :=
by
  let f := λ x : ℝ, a^x + 1
  have h_fixed : f 0 = a^0 + 1 := rfl
  have : a^0 = 1 := pow_zero a
  rw this at h_fixed
  exact h_fixed
  sorry

end fixed_point_of_function_l457_457291


namespace weight_combinations_l457_457579

theorem weight_combinations (w1 w2 w5 : ℕ) (n : ℕ) (h1 : w1 = 3) (h2 : w2 = 3) (h5 : w5 = 1) (hn : n = 9) : 
  nat.add 
    (nat.add 
      (nat.add (nat.add (nat.add (nat.add (nat.add 1 1) 1) 1) 1) 1) 1) 1 = 
  8 := 
sorry

end weight_combinations_l457_457579


namespace negation_of_p_l457_457101

-- Define the original predicate
def p (x₀ : ℝ) : Prop := x₀^2 > 1

-- Define the negation of the predicate
def not_p : Prop := ∀ x : ℝ, x^2 ≤ 1

-- Prove the negation of the proposition
theorem negation_of_p : (∃ x₀ : ℝ, p x₀) ↔ not_p := by
  sorry

end negation_of_p_l457_457101


namespace point_in_fourth_quadrant_l457_457529

def point : ℝ × ℝ := (3, -4)

def isFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def isSecondQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def isThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def isFourthQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : isFourthQuadrant point :=
by
  sorry

end point_in_fourth_quadrant_l457_457529


namespace addition_point_value_is_correct_l457_457317

-- Define the given conditions
def lower_bound : ℝ := 628
def upper_bound : ℝ := 774
def better_point : ℝ := 718
def golden_ratio_approx : ℝ := 0.618

-- Define a function to find the next test point using the 0.618 method
def next_test_point (lower upper better golden_ratio_approx : ℝ) : ℝ :=
  upper - (better - lower)

-- The proof problem: Verify that the next test point is 684
theorem addition_point_value_is_correct :
  next_test_point lower_bound upper_bound better_point golden_ratio_approx = 684 :=
by
  unfold next_test_point
  norm_num
  exact (774 - (718 - 628)) = 684
sorry

end addition_point_value_is_correct_l457_457317


namespace possible_values_of_n_l457_457567

theorem possible_values_of_n (a b : ℤ) (n : ℤ) (h1 : Int.gcd a b = 1) 
  (h2 : n = a^2 + b^2) (h3 : ∀ p : ℕ, Prime p → (p * p ≤ n) → (p ∣ a ∨ p ∣ b)) :
  n = 2 ∨ n = 5 ∨ n = 13 :=
by
  sorry

end possible_values_of_n_l457_457567


namespace total_surface_area_of_combined_solid_l457_457285

theorem total_surface_area_of_combined_solid :
  (forall (r : ℝ) (h : ℝ),
    π * r ^ 2 = 144 * π ∧ h = 10 → 
    (2 * π * r ^ 2 + 2 * π * r * h + π * r ^ 2 = 672 * π)) :=
by
  intros r h
  intro H
  cases H with H1 H2
  have r_squared : r ^ 2 = 144 := by nlinarith [H1]
  have r_value : r = 12 := by 
    rw [←sqrt_eq_iff_sq_eq, sqrt_eq_rpow, real.sqrt_eq_rpow]
    exact_mod_cast r_squared
  have A1 : 2 * π * 144 = 288 * π := by norm_num
  have A2 : 2 * π * 12 * 10 = 240 * π := by norm_num
  have A3 : π * 144 = 144 * π := by norm_num
  rw [r_value, ←A1, ←A2, ←A3]
  linarith 

end total_surface_area_of_combined_solid_l457_457285


namespace eval_f_at_sqrt2_minus_1_l457_457098

-- Define the function f
def f (x : ℝ) : ℝ := x + (1 / x)

-- State the theorem
theorem eval_f_at_sqrt2_minus_1 : f (Real.sqrt 2 - 1) = 2 * Real.sqrt 2 :=
by
  -- Proof is omitted
  sorry

end eval_f_at_sqrt2_minus_1_l457_457098


namespace lambda_mu_condition_l457_457111

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem lambda_mu_condition (λ μ : ℝ) :
  orthogonal (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
             (vector_a.1 + μ * vector_b.1, vector_a.2 + μ * vector_b.2) →
  λ * μ = -1 :=
by
  sorry

end lambda_mu_condition_l457_457111


namespace simplify_expression_l457_457938

-- Define the conditions
variables {a b c x : ℝ}
-- Define the distinctness condition
axiom distinct_reals (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)

-- Define the expression
def expr :=
  (x - a)^4 / ((a - b) * (a - c)) +
  (x - b)^4 / ((b - a) * (b - c)) +
  (x - c)^4 / ((c - a) * (c - b))

-- Define the theorem statement
theorem simplify_expression (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c) :
  expr = x^4 - 2*(a + b + c)*x^3 + (a^2 + b^2 + c^2 + 2*ab + 2*bc + 2*ca)*x^2 - 2*abc*x :=
sorry

end simplify_expression_l457_457938


namespace count_valid_two_digit_numbers_l457_457164

theorem count_valid_two_digit_numbers : 
  let digits := {0, 3, 6} in
  let valid_tens := {3, 6} in
  { (t, o) | t ∈ valid_tens ∧ o ∈ digits ∧ t ≠ o }.card = 4 :=
by
  sorry

end count_valid_two_digit_numbers_l457_457164


namespace bounded_area_arcsin_cos_l457_457418

noncomputable def bounded_area : ℝ :=
  ∫ x in 0..2 * Real.pi, Real.arcsin (Real.cos x)

theorem bounded_area_arcsin_cos :
  bounded_area = (Real.pi^2) / 2 := by
  sorry

end bounded_area_arcsin_cos_l457_457418


namespace matrix_B_pow_66_l457_457925

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0], 
    ![-1, 0, 0], 
    ![0, 0, 1]]

theorem matrix_B_pow_66 : B^66 = ![![-1, 0, 0], ![0, -1, 0], ![0, 0, 1]] := by
  sorry

end matrix_B_pow_66_l457_457925


namespace find_number_l457_457701

theorem find_number (x : ℝ) :
  0.15 * x = 0.25 * 16 + 2 → x = 40 :=
by
  -- skipping the proof steps
  sorry

end find_number_l457_457701


namespace factorial_expression_evaluation_l457_457747

theorem factorial_expression_evaluation : 8! - 7 * 7! + 6! - 5! = 5640 := by
  sorry

end factorial_expression_evaluation_l457_457747


namespace smallest_b_not_prime_x4_b4_l457_457856

theorem smallest_b_not_prime_x4_b4 :
  ∃ b : ℕ, b > 0 ∧ (∀ x : ℤ, ¬ Prime (x^4 + b^4)) ∧
  (∀ b' : ℕ, (b' > 0 ∧ ∀ x : ℤ, ¬ Prime (x^4 + b'^4)) → b ≤ b') :=
begin
  use 8,
  split,
  { exact nat.succ_pos' 7, },
  split,
  { intros x,
    sorry,  -- Proof that x^4 + 8^4 is not prime
  },
  { intros b' hb',
    sorry,  -- Proof that 8 is the smallest such b
  }
end

end smallest_b_not_prime_x4_b4_l457_457856


namespace multiplication_identity_multiplication_l457_457992

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l457_457992


namespace cards_per_layer_l457_457252

theorem cards_per_layer (total_decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) (h_decks : total_decks = 16) (h_cards_per_deck : cards_per_deck = 52) (h_layers : layers = 32) :
  total_decks * cards_per_deck / layers = 26 :=
by {
  -- To skip the proof
  sorry
}

end cards_per_layer_l457_457252


namespace problem_solution_l457_457485

noncomputable def f : ℕ → ℕ
| 2 := 5
| 3 := 7
| 4 := 11
| 5 := 17
| 6 := 23
| 7 := 40
| _ := 0  -- Assume a default value for other inputs

noncomputable def f_inv (y : ℕ) : ℕ :=
  if y = 5 then 2
  else if y = 7 then 3
  else if y = 11 then 4
  else if y = 17 then 5
  else if y = 23 then 6
  else if y = 40 then 7
  else 0  -- Assume a default value for other outputs

theorem problem_solution :
  f_inv ((f_inv 23)^2 + (f_inv 5)^2) = 7 :=
by
  sorry

end problem_solution_l457_457485


namespace interior_angle_sum_l457_457556

-- Definitions of conditions.
def exterior_angle_sum (n : ℕ) : ℝ := 360
def interior_angle (b : ℝ) : ℝ := 9 * b

-- Theorem statement.
theorem interior_angle_sum (n : ℕ) (h : n ≥ 3)
    (h1 : ∑ i in finset.range n, 1 = exterior_angle_sum n)
    (h2 : ∀ (i : ℕ), i < n → interior_angle (exterior_angle_sum n / n) = 9 * (exterior_angle_sum n / n)) :
    ∑ i in finset.range n, interior_angle (exterior_angle_sum n / n) = 3240 :=
by {
  sorry
}

end interior_angle_sum_l457_457556


namespace butterfinger_count_l457_457256

def total_candy_bars : ℕ := 12
def snickers : ℕ := 3
def mars_bars : ℕ := 2
def butterfingers : ℕ := total_candy_bars - (snickers + mars_bars)

theorem butterfinger_count : butterfingers = 7 :=
by
  unfold butterfingers
  sorry

end butterfinger_count_l457_457256


namespace BD_length_l457_457534

/-- Define a trapezoid ABCD with AB parallel to CD, and AC perpendicular to CD.
CD is given to be 15 units. -/
variables (A B C D : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]

/-- Tangent values for the angles C and B are given. -/
variables (angle_C angle_B : ℝ) (tan_C tan_B : ℝ)
variables (CD_length : ℝ := 15) (tan_C_val : ℝ := 2) (tan_B_val : ℝ := 3/2)

/-- Representing the assumptions in Lean variables. -/
axiom AB_parallel_CD : parallel (line_through A B) (line_through C D)
axiom AC_perp_CD : perpendicular (line_through A C) (line_through C D)
axiom (hCD : dist C D = CD_length)
axiom (hTanC : tan C = tan_C_val)
axiom (hTanB : tan B = tan_B_val)

/-- Prove that length of BD is 10√13 -/
theorem BD_length : dist B D = 10 * real.sqrt 13 :=
by
  sorry

end BD_length_l457_457534


namespace product_of_third_side_lengths_l457_457349

noncomputable def right_triangle_third_side_product : ℝ :=
  let c1 := real.sqrt (6 ^ 2 + 8 ^ 2)
  let c2 := real.sqrt (8 ^ 2 - 6 ^ 2)
  c1 * c2

-- The statement of the problem
theorem product_of_third_side_lengths :
  6 * 8 = 52.9 :=
by
  have c1 : ℝ := real.sqrt (6 ^ 2 + 8 ^ 2)
  have c2 : ℝ := real.sqrt (8 ^ 2 - 6 ^ 2)
  have product : ℝ := c1 * c2
  suffices h : real.to_rat product = (52.9 : ℝ).to_rat, from sorry, 
  sorry

end product_of_third_side_lengths_l457_457349


namespace bouquets_count_l457_457353

theorem bouquets_count (r c : ℕ) (h : 3 * r + 2 * c = 60) (h1 : c % 2 = 0) :
  ∃ (n : ℕ), n = 6 :=
begin
  -- Proof details are omitted
  sorry
end

end bouquets_count_l457_457353


namespace original_salary_l457_457344

theorem original_salary (S : ℝ) (h1 : S + 0.10 * S = 1.10 * S) (h2: 1.10 * S - 0.05 * (1.10 * S) = 1.10 * S * 0.95) (h3: 1.10 * S * 0.95 = 2090) : S = 2000 :=
sorry

end original_salary_l457_457344


namespace general_formula_sum_of_b_n_l457_457444

-- Define the arithmetic sequence and its properties given the conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  (a 4 + a 5 = 4 * a 2) ∧ (2 * a 3 - a 6 = 1)

-- Problem (1): Prove the general formula for the arithmetic sequence
theorem general_formula (a : ℕ → ℤ) (h : arithmetic_sequence a) : 
  ∀ n, a n = 2 * n + 1 :=
sorry

-- Define b_n as given in the problem
def b (a : ℕ → ℤ) (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define the sum of the first n terms of {b_n}
def S_n (a : ℕ → ℤ) (n : ℕ) : ℚ := ∑ i in Finset.range n, b a i

-- Problem (2): Prove the sum of the first n terms of {b_n}
theorem sum_of_b_n (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n, S_n a n = n / (6 * n + 9) :=
sorry

end general_formula_sum_of_b_n_l457_457444


namespace city_G_has_highest_percentage_increase_l457_457049

-- Define the population data as constants.
def population_1990_F : ℕ := 50
def population_2000_F : ℕ := 60
def population_1990_G : ℕ := 60
def population_2000_G : ℕ := 80
def population_1990_H : ℕ := 90
def population_2000_H : ℕ := 110
def population_1990_I : ℕ := 120
def population_2000_I : ℕ := 150
def population_1990_J : ℕ := 150
def population_2000_J : ℕ := 190

-- Define the function that calculates the percentage increase.
def percentage_increase (pop_1990 pop_2000 : ℕ) : ℚ :=
  (pop_2000 : ℚ) / (pop_1990 : ℚ)

-- Calculate the percentage increases for each city.
def percentage_increase_F := percentage_increase population_1990_F population_2000_F
def percentage_increase_G := percentage_increase population_1990_G population_2000_G
def percentage_increase_H := percentage_increase population_1990_H population_2000_H
def percentage_increase_I := percentage_increase population_1990_I population_2000_I
def percentage_increase_J := percentage_increase population_1990_J population_2000_J

-- Prove that City G has the greatest percentage increase.
theorem city_G_has_highest_percentage_increase :
  percentage_increase_G > percentage_increase_F ∧ 
  percentage_increase_G > percentage_increase_H ∧
  percentage_increase_G > percentage_increase_I ∧
  percentage_increase_G > percentage_increase_J :=
by sorry

end city_G_has_highest_percentage_increase_l457_457049


namespace trains_clearance_time_l457_457345

-- Define the lengths of the trains
def train1_length : ℕ := 135
def train2_length : ℕ := 165

-- Define the speeds of the trains in km/h
def train1_speed_kmh : ℕ := 80
def train2_speed_kmh : ℕ := 65

-- Function to convert speed from km/h to m/s
def convert_kmh_to_mps (speed_kmh : ℕ) : ℝ :=
  speed_kmh * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : ℝ :=
  convert_kmh_to_mps (train1_speed_kmh + train2_speed_kmh)

-- Total distance to be cleared by the trains
def total_distance : ℕ :=
  train1_length + train2_length

-- Time for the trains to be completely clear in seconds
def time_to_clear : ℝ :=
  total_distance / relative_speed_mps

theorem trains_clearance_time : time_to_clear = 7.45 := by
  -- Since we are not asked to solve, we insert sorry here
  sorry

end trains_clearance_time_l457_457345


namespace measure_angle_C_120_l457_457878

variable {A B C a b c : ℝ}

-- Conditions
def angle_opposite_sides (a b c : ℝ) (h : c^2 = a^2 + b^2 + ab) : Prop :=
  ∃ C : ℝ, C > 0 ∧ C < 180 ∧ (cos C = -1 / 2) 

-- Theorem to prove
theorem measure_angle_C_120 (h : c^2 = a^2 + b^2 + ab) :  
  ∃ C : ℝ, C > 0 ∧ C < 180 ∧ (cos C = -1 / 2) := 
by
  sorry

end measure_angle_C_120_l457_457878


namespace find_k_collinear_l457_457851

-- Define the points A, B, and C
structure Point :=
(x : ℤ)
(y : ℤ)

def A : Point := ⟨3, 1⟩
def C : Point := ⟨8, 11⟩

-- We introduce the notion of collinearity given points A, B with variable y-coordinate
def collinear (A B C : Point) : Prop :=
  ∃ λ : ℝ, (--5, B.y - A.y) = (λ * 5, λ * 10)

-- Define the point B with variable y-coordinate k
def B (k : ℤ) : Point := ⟨-2, k⟩

-- The main theorem which states the condition for collinearity given the points A, B, and C
theorem find_k_collinear : ∃ k : ℤ, collinear A (B k) C ∧ k = -9 :=
by
  apply Exists.intro (-9)
  dsimp [B, collinear, A, C]
  sorry

end find_k_collinear_l457_457851


namespace yearly_cost_of_oil_changes_l457_457918

-- Definitions of conditions
def miles_per_month : ℕ := 1000
def months_in_year : ℕ := 12
def oil_change_frequency : ℕ := 3000
def free_oil_changes_per_year : ℕ := 1
def cost_per_oil_change : ℕ := 50

theorem yearly_cost_of_oil_changes : 
  let total_miles := miles_per_month * months_in_year in
  let total_oil_changes := total_miles / oil_change_frequency in
  let paid_oil_changes := total_oil_changes - free_oil_changes_per_year in
  paid_oil_changes * cost_per_oil_change = 150 := 
by
  sorry

end yearly_cost_of_oil_changes_l457_457918


namespace distance_center_circle_to_line_l457_457541

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
(t - 3, real.sqrt(3) * t)

def polar_circle_eq (ρ θ : ℝ) : Prop :=
ρ^2 - 4 * ρ * cos θ + 3 = 0

theorem distance_center_circle_to_line :
  let C := (2 : ℝ, 0)
  let l := (λ t, (t - 3, real.sqrt(3) * t))
  ∃ d, d = ((5 : ℝ) * real.sqrt(3)) / 2 :=
sorry

end distance_center_circle_to_line_l457_457541


namespace pedestrian_avg_speed_greater_l457_457721

def average_speed_greater_than_5 : Prop :=
  ∃ (v : ℝ), (v > 0) → 
  let V_avg := (10 + 2 * v + (7.5 - 1.5 * v)) / 3.5
  in V_avg > 5

theorem pedestrian_avg_speed_greater :
  average_speed_greater_than_5 := sorry

end pedestrian_avg_speed_greater_l457_457721


namespace reciprocal_difference_decreases_l457_457543

theorem reciprocal_difference_decreases (n : ℕ) (hn : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1 : ℝ)) < (1 / (n * n : ℝ)) :=
by 
  sorry

end reciprocal_difference_decreases_l457_457543


namespace tangent_lines_l457_457034

def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def pointA : ℝ × ℝ := (2, 4)

theorem tangent_lines (x y : ℝ) (hx : circle x y) : 
  (∃ k : ℝ, y - 4 = k * (x - 2)) → 
  (x = 2) ∨ (3 * x - 4 * y + 10 = 0) :=
sorry

end tangent_lines_l457_457034


namespace beau_sons_age_l457_457746

variable (n : ℕ)

theorem beau_sons_age (h1: ∀ x, 16 = 16) (h2: 13 * n = 42 - 3) (h3: 42 = 42) : n = 3 :=
sorry

end beau_sons_age_l457_457746


namespace sum_squares_l457_457929

theorem sum_squares {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) 
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
by sorry

end sum_squares_l457_457929


namespace part1_part2_part3_l457_457454

-- Definitions based on the problem statements
def ellipse (x y : ℝ) := x^2 + y^2 / 2 = 1
def area_triangle (x1 y1 x2 y2 : ℝ) := (x1 * y2 - x2 * y1) / 2

-- Problem parts translated to Lean
theorem part1 (m : ℝ) (h : m ≠ 0) (S : ℝ) 
  (hpq : ∃ x1 y1 x2 y2, ellipse x1 y1 ∧ ellipse x2 y2 ∧ area_triangle 0 0 x1 y1 x2 y2 = S) :
  S = sqrt 2 / 2 → m = sqrt 2 / 2 ∨ m = -sqrt 2 / 2 :=
sorry

theorem part2 (x1 y1 x2 y2 : ℝ) 
  (h1 : ellipse x1 y1 ∧ ellipse x2 y2)
  (h2 : area_triangle 0 0 x1 y1 x2 y2 = sqrt 2 / 2) :
  x1^2 + x2^2 = 1 ∧ y1^2 + y2^2 = 2 :=
sorry

theorem part3 
  (u v u1 v1 u2 v2 : ℝ)
  (hd : ellipse u v)
  (he : ellipse u1 v1)
  (hg : ellipse u2 v2)
  (h1 : area_triangle 0 0 u v u1 v1 = sqrt 2 / 2)
  (h2 : area_triangle 0 0 u v u2 v2 = sqrt 2 / 2)
  (h3 : area_triangle 0 0 u1 v1 u2 v2 = sqrt 2 / 2) :
  false :=
sorry

end part1_part2_part3_l457_457454


namespace correct_option_is_D_l457_457331

def is_algebraic_expression (e : Expr) : Prop := sorry

-- Define each option as per the problem
def option_A : Prop := ¬ is_algebraic_expression (-2)
def option_B (a : Expr) : Prop := ∀ n, n < 0 → -a = n
def option_C (a c : Expr) : Prop := (3/4 : ℚ) = 3
def option_D (x : Expr) : Prop := is_algebraic_expression (x + 1)

-- Main theorem to prove
theorem correct_option_is_D (x : Expr) : option_D x :=
by
  apply sorry

end correct_option_is_D_l457_457331


namespace length_of_DC_l457_457539

noncomputable def length_DC (AB BD BC CD : ℝ) : Prop :=
  AB = 30 ∧ ∠ADB = 90 ∧ sin(A) = 3/5 ∧ sin(C) = 1/4 → CD = 30 * √6

theorem length_of_DC (AB BD BC CD : ℝ) (h1 : AB = 30)
  (h2 : ∠ADB = 90) (h3 : sin(A) = 3/5) (h4 : sin(C) = 1/4) : CD = 30 * √6 := by
  sorry

end length_of_DC_l457_457539


namespace lambda_mu_relationship_l457_457138

theorem lambda_mu_relationship (a b : ℝ × ℝ) (λ μ : ℝ)
  (h1 : a = (1, 1))
  (h2 : b = (1, -1))
  (h3 : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457138


namespace A_loses_240_l457_457966

def initial_house_value : ℝ := 12000
def house_value_after_A_sells : ℝ := initial_house_value * 0.85
def house_value_after_B_sells_back : ℝ := house_value_after_A_sells * 1.2

theorem A_loses_240 : house_value_after_B_sells_back - initial_house_value = 240 := by
  sorry

end A_loses_240_l457_457966


namespace hermite_integer_l457_457695

-- Hermite polynomials definition
def hermitePoly : ℕ × ℤ → ℤ 
| (0, x) := 1
| (n + 1, x) := (1 / (nat.factorial (n + 1))).to_rat * 
                  (list.prod (list.range (n + 1)).map (λ k, x - k)).to_rat

-- Statement to prove
theorem hermite_integer (n : ℕ) (k : ℤ) : hermitePoly (n, k) ∈ ℤ :=
sorry

end hermite_integer_l457_457695


namespace radical_axis_is_vertical_l457_457430

-- Define the necessary geometric entities and their properties
variables (x x1 x2 r1 r2 : ℝ)
variables (y : ℝ)

-- Define the coordinates of the centers of the circles
def O1 := (-x1, 0 : ℝ)
def O2 := (x2, 0 : ℝ)

-- Define the power of point M with respect to the circles
def power_circle_1 (M : ℝ × ℝ) :=
  (M.fst + x1) ^ 2 + M.snd ^ 2 - r1 ^ 2

def power_circle_2 (M : ℝ × ℝ) :=
  (M.fst - x2) ^ 2 + M.snd ^ 2 - r2 ^ 2

-- The radical axis theorem statement
theorem radical_axis_is_vertical :
  ∀ (M : ℝ × ℝ), power_circle_1 M = power_circle_2 M →
  M.fst = (r1 ^ 2 - r2 ^ 2 + x2 ^ 2 - x1 ^ 2) / (2 * (x1 + x2)) :=
by 
  intros M h
  sorry

end radical_axis_is_vertical_l457_457430


namespace find_monic_quartic_polynomial_l457_457033

noncomputable def monic_quartic_polynomial : Polynomial ℚ :=
  Polynomial.X^4 - 10 * Polynomial.X^3 + 29 * Polynomial.X^2 - 44 * Polynomial.X + 20

theorem find_monic_quartic_polynomial :
  (monic_quartic_polynomial.eval (2 + complex.i) = 0 ∧ 
   monic_quartic_polynomial.eval (2 - complex.i) = 0 ∧ 
   monic_quartic_polynomial.eval (3 - real.sqrt 5) = 0 ∧ 
   monic_quartic_polynomial.eval (3 + real.sqrt 5) = 0) ∧
  (monic_quartic_polynomial.leadingCoeff = 1) :=
by
  sorry

end find_monic_quartic_polynomial_l457_457033


namespace multiply_polynomials_l457_457979

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l457_457979


namespace lindas_savings_l457_457958

theorem lindas_savings :
  ∃ S : ℝ, (3 / 4 * S) + 150 = S ∧ (S - 150) = 3 / 4 * S := 
sorry

end lindas_savings_l457_457958


namespace Lee_Family_SUV_Seating_Arrangements_l457_457259

theorem Lee_Family_SUV_Seating_Arrangements :
  let family := ["Mr. Lee", "Mrs. Lee", "Child1", "Child2", "Child3"],
      front_seats := 2,
      back_seats := 3,
      driver_choices := 2,
      front_passenger_choices := 4,
      back_arrangements := 6 in
  (driver_choices * front_passenger_choices * back_arrangements = 48) :=
by
  sorry

end Lee_Family_SUV_Seating_Arrangements_l457_457259


namespace tic_tac_toe_alex_wins_second_X_l457_457888

theorem tic_tac_toe_alex_wins_second_X :
  ∃ b : ℕ, b = 12 := 
sorry

end tic_tac_toe_alex_wins_second_X_l457_457888


namespace max_a_plus_b_squared_plus_c_to_four_l457_457568

theorem max_a_plus_b_squared_plus_c_to_four (a b c : ℝ) (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 :=
sorry

example : ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ a + b^2 + c^4 = 3 :=
begin
  use [3, 0, 0],
  repeat { split },
  { exact le_refl 3 },
  { exact le_refl 0 },
  { exact le_refl 0 },
  { exact add_zero 3 },
  { exact add_zero 3 },
end

end max_a_plus_b_squared_plus_c_to_four_l457_457568


namespace perpendicular_vectors_l457_457119

variable (λ μ : ℝ)
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

theorem perpendicular_vectors : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0 → λ * μ = -1 := by
  sorry

end perpendicular_vectors_l457_457119


namespace multiply_polynomials_l457_457982

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l457_457982


namespace curve_intersection_point_l457_457522

theorem curve_intersection_point :
  ∃ (x y : ℝ), 
    (∃ θ, 0 ≤ θ ∧ θ ≤ (π / 2) ∧ x = sqrt 5 * cos θ ∧ y = sqrt 5 * sin θ) ∧
    (∃ t, x = 1 - (sqrt 2 / 2) * t ∧ y = -(sqrt 2 / 2) * t) ∧
    (x = 2 ∧ y = 1) :=
by
  sorry

end curve_intersection_point_l457_457522


namespace math_problem_proof_l457_457931

noncomputable def problem_statement (a b c : ℝ) : Prop :=
 (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a + b + c = 0) ∧ (a^4 + b^4 + c^4 = a^6 + b^6 + c^6) → 
 (a^2 + b^2 + c^2 = 3 / 2)

theorem math_problem_proof : ∀ (a b c : ℝ), problem_statement a b c :=
by
  intros
  sorry

end math_problem_proof_l457_457931


namespace anusha_receives_84_l457_457694

-- Define the conditions as given in the problem
def anusha_amount (A : ℕ) (B : ℕ) (E : ℕ) : Prop :=
  12 * A = 8 * B ∧ 12 * A = 6 * E ∧ A + B + E = 378

-- Lean statement to prove the amount Anusha gets is 84
theorem anusha_receives_84 (A B E : ℕ) (h : anusha_amount A B E) : A = 84 :=
sorry

end anusha_receives_84_l457_457694


namespace smallest_palindrome_proof_l457_457043

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  (100 ≤ n ∧ n ≤ 999) ∧ is_palindrome n

def smallest_non_five_digit_palindrome_product_with_103 : ℕ :=
  404

theorem smallest_palindrome_proof :
  is_three_digit_palindrome smallest_non_five_digit_palindrome_product_with_103 ∧ 
  ¬is_palindrome (103 * smallest_non_five_digit_palindrome_product_with_103) ∧ 
  (∀ n, is_three_digit_palindrome n → ¬is_palindrome (103 * n) → n ≥ 404) :=
begin
  sorry
end

end smallest_palindrome_proof_l457_457043


namespace angle_DAC_is_10_l457_457897

-- Definitions of the angles and conditions
def parallel (l1 l2 : line) : Prop := ∀ (A B : point), (A ∈ l1 ∧ B ∈ l1) → (A ∈ l2 ∧ B ∈ l2) → A = B
def straight_angle (a : angle) : Prop := angle.measure a = 180
def angle_measure_to_degrees (a : angle) (deg : ℝ) : Prop := angle.measure a = deg

-- Points and lines in the problem
variables (A B C D F : point)
variables (ACF : line)
variables (AB DC : line)

-- Angles given in the problem
variables (angle_BADC : angle) (angle_ABC : angle) (angle_ACD : angle)

-- Conditions given in the problem
def condition1 : Prop := parallel AB DC
def condition2 : Prop := straight_angle ⟨A, C, F⟩
def condition3 : Prop := angle_measure_to_degrees angle_BADC 125
def condition4 : Prop := angle_measure_to_degrees angle_ABC 65
def condition5 : Prop := angle_measure_to_degrees angle_ACD 110

-- The proof that the measure of angle DAC is 10 degrees
theorem angle_DAC_is_10 :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 →
  ∃ angle_DAC : angle, angle_measure_to_degrees angle_DAC 10 :=
by
  sorry

end angle_DAC_is_10_l457_457897


namespace compute_S_div_n_l457_457232

def first_10_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

noncomputable def n : ℕ := first_10_primes.prod

def euler_totient (x : ℕ) : ℕ :=
if x = 0 then 0 else Nat.totient x

noncomputable def S : ℕ :=
∑ x in (Finset.divisors n).product (Finset.divisors n), (λ (xy : ℕ × ℕ), let (x, y) := xy in if x * y ∣ n then euler_totient x * y else 0)

theorem compute_S_div_n : S / n = 1024 := sorry

end compute_S_div_n_l457_457232


namespace find_angle_C_l457_457087

noncomputable def sides_of_triangle := {a b c : ℝ}

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
(a^2 + b^2 - c^2) / 4

theorem find_angle_C (a b c : ℝ) (habc : sides_of_triangle)
  (h_area : area_of_triangle a b c = (a^2 + b^2 - c^2) / 4) :
  ∃ C : ℝ, C = 45 :=
by
  sorry

end find_angle_C_l457_457087


namespace log_base_10_of_50_eq_1_plus_log_base_10_of_5_l457_457748

variables {a b : ℝ}

theorem log_base_10_of_50_eq_1_plus_log_base_10_of_5 :
  log 10 50 = 1 + log 10 5 :=
by
  -- Using \(50 = 5 \times 10\)
  have h1 : 50 = 5 * 10 := by norm_num
  -- Applying the logarithmic product rule \(\log_{10}(a \times b) = \log_{10}(a) + \log_{10}(b)\)
  have h2 : log 10 50 = log 10 (5 * 10) := by rw h1
  have h3 : log 10 (5 * 10) = log 10 5 + log 10 10 := log_mul (by norm_num : 1 < 10)
  -- Knowing \(\log_{10}(10) = 1\)
  have h4 : log 10 10 = 1 := log_base_pow 10 (by norm_num : 1 < 10)
  -- Combining everything to establish the final equality
  rw [h2, h3, h4]
  sorry

end log_base_10_of_50_eq_1_plus_log_base_10_of_5_l457_457748


namespace concyclic_points_radius_of_circle_through_points_l457_457927

-- Definition of variables and assumptions
variables (A B C D P: Point) (W X Y Z E F G H : Point)
variables (Ω: Circle)
variables (R d: ℝ)

-- Given conditions
axiom cyclic_quadrilateral : CyclicQuadrilateral A B C D 
axiom perpendicular_diagonals : Perpendicular (Line.through A C) (Line.through B D)
axiom intersection_of_diagonals : Intersection (Line.through A C) (Line.through B D) = P
axiom projections_of_P : Projection P (Line.through A B) = W ∧ Projection P (Line.through B C) = X ∧ Projection P (Line.through C D) = Y ∧ Projection P (Line.through D A) = Z
axiom midpoints : Midpoint A B = E ∧ Midpoint B C = F ∧ Midpoint C D = G ∧ Midpoint D A = H
axiom circle_radius : Radius Ω = R
axiom distance_center_to_P : Distance (Center Ω) P = d

-- Part (a): Prove E, F, G, H, W, X, Y, Z are concyclic
theorem concyclic_points : Concyclic E F G H W X Y Z :=
sorry

-- Part (b): Find the radius in terms of R and d
theorem radius_of_circle_through_points : 
    Radius (CircumscribedCircle E F G H W X Y Z) = (sqrt (2 * R^2 - d^2)) / 2 :=
sorry

end concyclic_points_radius_of_circle_through_points_l457_457927


namespace sequence_a7_l457_457632

theorem sequence_a7 (a b : ℕ) (h1 : a1 = a) (h2 : a2 = b) {a3 a4 a5 a6 a7 : ℕ}
  (h3 : a_3 = a + b)
  (h4 : a_4 = a + 2 * b)
  (h5 : a_5 = 2 * a + 3 * b)
  (h6 : a_6 = 3 * a + 5 * b)
  (h_a6 : a_6 = 50) :
  a_7 = 5 * a + 8 * b :=
by
  sorry

end sequence_a7_l457_457632


namespace inequality_solution_range_l457_457873

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4a) ↔ a ∈  set.Iio 1 ∪ set.Ioi 3 :=
sorry

end inequality_solution_range_l457_457873


namespace necessarily_positive_l457_457602

theorem necessarily_positive (x y z : ℝ) (hx : -1 < x ∧ x < 1) 
                      (hy : -1 < y ∧ y < 0) 
                      (hz : 1 < z ∧ z < 2) : 
    y + z > 0 := 
by
  sorry

end necessarily_positive_l457_457602


namespace exists_b_c_with_integral_roots_l457_457404

theorem exists_b_c_with_integral_roots :
  ∃ (b c : ℝ), (∃ (p q : ℤ), (x^2 + b * x + c = 0) ∧ (x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
               ((x - p) * (x - q) = x^2 - (p + q) * x + p*q)) ∧
              (∃ (r s : ℤ), (x^2 + (b+1) * x + (c+1) = 0) ∧ 
              ((x - r) * (x - s) = x^2 - (r + s) * x + r*s)) :=
by
  sorry

end exists_b_c_with_integral_roots_l457_457404


namespace multiply_expression_l457_457985

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l457_457985


namespace cos_18_eq_l457_457001

-- Definitions for the conditions.
def a := Real.cos (36 * Real.pi / 180)
def c := Real.cos (18 * Real.pi / 180)

-- Statement of the problem
theorem cos_18_eq :
  c = (Real.sqrt (10 + 2 * Real.sqrt 5) / 4) :=
by
  -- conditions given in the problem
  have h1: a = 2 * c^2 - 1, from sorry
  have h2: Real.sin (36 * Real.pi / 180) = Real.sqrt (1 - a^2), from sorry
  have triple_angle: Real.cos (3 * θ) = 4 * (Real.cos θ)^3 - 3 * (Real.cos θ), from sorry
  sorry

end cos_18_eq_l457_457001


namespace haley_picked_carrots_l457_457488

variable (H : ℕ)
variable (mom_carrots : ℕ := 38)
variable (good_carrots : ℕ := 64)
variable (bad_carrots : ℕ := 13)
variable (total_carrots : ℕ := good_carrots + bad_carrots)

theorem haley_picked_carrots : H + mom_carrots = total_carrots → H = 39 := by
  sorry

end haley_picked_carrots_l457_457488


namespace diagonal_passes_through_cubes_l457_457703

-- Defining the problem conditions
def dimensions := (200, 360, 450)
def gcd_of_dimensions := Int.gcd (Int.gcd 200 360) 450  -- Compute GCD of all dimensions

-- State the theorem
theorem diagonal_passes_through_cubes :
  let (x, y, z) := dimensions in
  let gcd_xy := Int.gcd x y in
  let gcd_yz := Int.gcd y z in
  let gcd_zx := Int.gcd z x in
  let gcd_xyz := gcd_of_dimensions in
  x + y + z - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 840 :=
by
  let x := 200
  let y := 360
  let z := 450
  let gcd_xy := Int.gcd x y
  let gcd_yz := Int.gcd y z
  let gcd_zx := Int.gcd z x
  let gcd_xyz := Int.gcd (Int.gcd x y) z
  have : x + y + z - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 840 := 
    by
      sorry -- Final proof step required, skipping for now
  exact this

end diagonal_passes_through_cubes_l457_457703


namespace cream_ratio_to_JoAnn_l457_457223

def coffee_cup := {coffee: ℕ, cream: ℕ} -- Definition of a coffee cup with coffee and cream

def Joe_initial : coffee_cup := {coffee := 15, cream := 0}
def Joe_drank_3 : coffee_cup := {coffee := Joe_initial.coffee - 3, cream := Joe_initial.cream}
def Joe_added_4_cream : coffee_cup := {coffee := Joe_drank_3.coffee, cream := Joe_drank_3.cream + 4}

def JoAnn_initial : coffee_cup := {coffee := 15, cream := 0}
def JoAnn_add_3_cream : coffee_cup := {coffee := JoAnn_initial.coffee, cream := JoAnn_initial.cream + 3}
def JoAnn_add_2_cream : coffee_cup := {coffee := JoAnn_add_3_cream.coffee, cream := JoAnn_add_3_cream.cream + 2}
def JoAnn_mix : coffee_cup := {coffee := JoAnn_add_2_cream.coffee, cream := JoAnn_add_2_cream.cream}
def JoAnn_drink_4 : coffee_cup := {coffee := JoAnn_mix.coffee - 4, cream := JoAnn_mix.cream * (JoAnn_mix.coffee - 4) / JoAnn_mix.coffee}

-- Theorem to prove the ratio of cream in Joe's coffee to JoAnn's coffee is 1
theorem cream_ratio_to_JoAnn : (Joe_added_4_cream.cream) / (JoAnn_drink_4.cream) = 1 :=
by sorry

end cream_ratio_to_JoAnn_l457_457223


namespace johns_oil_change_cost_l457_457915

theorem johns_oil_change_cost:
  (miles_per_month: ℕ) (miles_per_oil_change: ℕ) (free_oil_changes_per_year: ℕ) (cost_per_oil_change: ℕ) 
  (h₁ : miles_per_month = 1000) 
  (h₂ : miles_per_oil_change = 3000) 
  (h₃ : free_oil_changes_per_year = 1) 
  (h₄ : cost_per_oil_change = 50) : 
  (12 * cost_per_oil_change * miles_per_oil_change) // (miles_per_month * miles_per_oil_change) - (free_oil_changes_per_year * cost_per_oil_change) = 150 := 
by 
  sorry

end johns_oil_change_cost_l457_457915


namespace symmetric_point_wrt_y_axis_l457_457825

noncomputable def symmetric_point_polar_coordinates (r θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let symmetric_x := -x in
  let symmetric_y := y in
  let r_sym := Real.sqrt (symmetric_x^2 + symmetric_y^2) in
  let θ_sym := Real.atan (symmetric_y / symmetric_x) + if symmetric_x < 0 then π else 0 in
  (r_sym, θ_sym)

theorem symmetric_point_wrt_y_axis :
  symmetric_point_polar_coordinates 2 (-(Real.pi / 6)) = (2, 7 * Real.pi / 6) :=
  sorry

end symmetric_point_wrt_y_axis_l457_457825


namespace cost_of_parts_per_tire_repair_is_5_l457_457221

-- Define the given conditions
def charge_per_tire_repair : ℤ := 20
def num_tire_repairs : ℤ := 300
def charge_per_complex_repair : ℤ := 300
def num_complex_repairs : ℤ := 2
def cost_per_complex_repair_parts : ℤ := 50
def retail_shop_profit : ℤ := 2000
def fixed_expenses : ℤ := 4000
def total_profit : ℤ := 3000

-- Define the calculation for total revenue
def total_revenue : ℤ := 
    (charge_per_tire_repair * num_tire_repairs) + 
    (charge_per_complex_repair * num_complex_repairs) + 
    retail_shop_profit

-- Define the calculation for total expenses
def total_expenses : ℤ := total_revenue - total_profit

-- Define the calculation for parts cost of tire repairs
def parts_cost_tire_repairs : ℤ := 
    total_expenses - (cost_per_complex_repair_parts * num_complex_repairs) - fixed_expenses

def cost_per_tire_repair : ℤ := parts_cost_tire_repairs / num_tire_repairs

-- The statement to be proved
theorem cost_of_parts_per_tire_repair_is_5 : cost_per_tire_repair = 5 := by
    sorry

end cost_of_parts_per_tire_repair_is_5_l457_457221


namespace gain_percent_is_150_l457_457687

theorem gain_percent_is_150 (CP SP : ℝ) (hCP : CP = 10) (hSP : SP = 25) : (SP - CP) / CP * 100 = 150 := by
  sorry

end gain_percent_is_150_l457_457687


namespace largest_prime_factor_of_Q_l457_457635

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d, d ∣ n → d = 1 ∨ d = n

def largest_prime_factor (n : ℕ) : ℕ :=
  @Nat.Prime.factorization n (Nat.Prime.factorization_spec _)

theorem largest_prime_factor_of_Q :
  ∃ Q, (Q = 1111) ∧ (largest_prime_factor Q = 101) :=
by
  let Q := (Nat.digits 10 ((2 ^ 222) ^ 5 * (5 ^ 555) ^ 2)).length
  have Q_def : Q = 1111 := by
    have : (2 ^ 222) ^ 5 = 2 ^ 1110 := by norm_num
    have : (5 ^ 555) ^ 2 = 5 ^ 1110 := by norm_num
    have : (2 ^ 1110) * (5 ^ 1110) = 10 ^ 1110 := by norm_num
    calc (Nat.digits 10 (10 ^ 1110)).length
        = 1110 + 1 : by norm_num
        ... = 1111 : by norm_num
  use Q
  exact ⟨Q_def, rfl⟩

end largest_prime_factor_of_Q_l457_457635


namespace proof_statement_l457_457945

noncomputable def proof_problem (x y : ℝ) (hx : 1 < x) (hy : 1 < y) : Prop :=
  (log 3 x)^5 + (log 5 y)^5 + 32 = 16 * (log 3 x) * (log 5 y) →
  x^2 + y^2 = 3^((2 * real.root 16 5)) + 5^((2 * real.root 16 5))

-- Statement of the problem in Lean 4
theorem proof_statement (x y : ℝ) (hx : 1 < x) (hy : 1 < y) 
  (h : (log 3 x)^5 + (log 5 y)^5 + 32 = 16 * (log 3 x) * (log 5 y)) :
  x^2 + y^2 = 3^(2 * real.root 16 5) + 5^(2 * real.root 16 5) :=
sorry

end proof_statement_l457_457945


namespace equal_chords_imply_equal_sums_l457_457266

theorem equal_chords_imply_equal_sums (A B C D : Point) (O : Point) (circle : Circle) (h1 : circle.contains A) (h2 : circle.contains B) (h3 : circle.contains C) (h4 : circle.contains D) 
  (h5 : chord_length_eq A B = chord_length_eq C D) :
  dist A B + dist C D = dist A D + dist B C := 
sorry

end equal_chords_imply_equal_sums_l457_457266


namespace movies_watched_total_l457_457660

theorem movies_watched_total :
  ∀ (Timothy2009 Theresa2009 Timothy2010 Theresa2010 total : ℕ),
    Timothy2009 = 24 →
    Timothy2010 = Timothy2009 + 7 →
    Theresa2010 = 2 * Timothy2010 →
    Theresa2009 = Timothy2009 / 2 →
    total = Timothy2009 + Timothy2010 + Theresa2009 + Theresa2010 →
    total = 129 :=
by
  intros Timothy2009 Theresa2009 Timothy2010 Theresa2010 total
  sorry

end movies_watched_total_l457_457660


namespace smallest_palindrome_proof_l457_457044

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  (100 ≤ n ∧ n ≤ 999) ∧ is_palindrome n

def smallest_non_five_digit_palindrome_product_with_103 : ℕ :=
  404

theorem smallest_palindrome_proof :
  is_three_digit_palindrome smallest_non_five_digit_palindrome_product_with_103 ∧ 
  ¬is_palindrome (103 * smallest_non_five_digit_palindrome_product_with_103) ∧ 
  (∀ n, is_three_digit_palindrome n → ¬is_palindrome (103 * n) → n ≥ 404) :=
begin
  sorry
end

end smallest_palindrome_proof_l457_457044


namespace general_term_formula_sum_of_sequence_l457_457822

-- Definitions from conditions
axiom a_sequence (a : ℕ → ℤ) : ∀ n, a n = 3 * n - 1
axiom b_sequence (b : ℕ → ℝ) : ∀ n, b 1 = 1 ∧ b 2 = (1 : ℝ) / 3 ∧
  (∀ n, (a_sequence a n) * (b (n + 1)) + (b (n + 1)) = n * (b n))

-- Task 1: Prove the general term formula for {a_n}
theorem general_term_formula : ∀ n, a_sequence a n = 3 * n - 1 := by
  sorry

-- Task 2: Prove the sum of the first n terms of {b_n}
theorem sum_of_sequence (S : ℕ → ℝ) : ∀ n, S n = (3 / 2) - (1 / (2 * (3 : ℝ)^(n-1))) := by
  sorry

end general_term_formula_sum_of_sequence_l457_457822


namespace total_games_won_l457_457758

theorem total_games_won (total_games : ℕ) (daytime_games : ℕ) (night_games : ℕ)
    (daytime_win_percentage : ℚ) (night_win_percentage : ℚ)
    (total_losses : ℕ) (daytime_home_losses : ℕ) (daytime_away_losses : ℕ)
    (winning_streak : ℕ) : ℕ := 
  -- Conditions
  total_games = 36 →
  daytime_games = 28 →
  night_games = 8 →
  daytime_win_percentage = 0.70 →
  night_win_percentage = 0.875 →
  total_losses = 5 →
  daytime_home_losses = 3 →
  daytime_away_losses = 2 →
  winning_streak = 10 →
  -- Question: Prove the total number of games won is 26
  19 + 7 = 26 := 
sorry

end total_games_won_l457_457758


namespace lambda_mu_relationship_l457_457134

theorem lambda_mu_relationship (a b : ℝ × ℝ) (λ μ : ℝ)
  (h1 : a = (1, 1))
  (h2 : b = (1, -1))
  (h3 : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457134


namespace length_of_BC_l457_457762

open EuclideanGeometry

-- Define the conditions and the main statement.
theorem length_of_BC
  (A B C O G : Point ℝ)
  (hABCacute: ∠B A C < 90)
  (hABC45: ∠B A C = 45)
  (hO : IsCircumcenter O A B C)
  (hG : IsCentroid G A B C)
  (hOG_1 : dist O G = 1)
  (hOG_parallel_BC : Parallel O G B C):
  dist B C = 12 := sorry

end length_of_BC_l457_457762


namespace log_expression_value_l457_457464

theorem log_expression_value (a : ℝ) (h : 1 + a ^ 3 = 9) : 
  Real.log (1 / 4) a + Real.log a 8 = 2 :=
by
  sorry

end log_expression_value_l457_457464


namespace probability_one_bad_player_in_failed_quest_l457_457536

theorem probability_one_bad_player_in_failed_quest :
  let total_players := 10
  let bad_players := 4
  let good_players := 6
  let quest_size := 3
  let total_ways_to_choose_3 := Nat.choose total_players quest_size
  let ways_to_choose_3_good := Nat.choose good_players quest_size
  let ways_to_choose_at_least_one_bad := total_ways_to_choose_3 - ways_to_choose_3_good
  let ways_to_choose_1_bad_2_good := Nat.choose bad_players 1 * Nat.choose good_players 2
  in
  ways_to_choose_at_least_one_bad ≠ 0 →
  (ways_to_choose_1_bad_2_good / ways_to_choose_at_least_one_bad : ℚ) = 3 / 5 :=
sorry

end probability_one_bad_player_in_failed_quest_l457_457536


namespace cyclic_quadrilateral_l457_457936

open EuclideanGeometry

variables {Ω1 Ω2 : Circle} (t t' : Tangent) (T1 T2 T1' T2' P1 P2 M : Point)

-- Conditions 
axiom tangency1 : is_tangent t Ω1
axiom tangency2 : is_tangent t Ω2
axiom tangency1' : is_tangent t' Ω1
axiom tangency2' : is_tangent t' Ω2
axiom T1_tangency : is_point_of_tangency T1 t Ω1
axiom T2_tangency : is_point_of_tangency T2 t Ω2
axiom T1'_tangency : is_point_of_tangency T1' t' Ω1
axiom T2'_tangency : is_point_of_tangency T2' t' Ω2
axiom M_midpoint : is_midpoint M T1 T2
axiom P1_intersection : is_intersection P1 (line M T1') Ω1
axiom P2_intersection : is_intersection P2 (line M T2') Ω2

-- Goal
theorem cyclic_quadrilateral :
  is_cyclic_quadrilateral P1 P2 T1' T2' := 
sorry

end cyclic_quadrilateral_l457_457936


namespace perpendicular_vectors_condition_l457_457121

theorem perpendicular_vectors_condition (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
      b : ℝ × ℝ := (1, -1)
  in ((a.1 + λ * b.1) * (a.1 + μ * b.1) + (a.2 + λ * b.2) * (a.2 + μ * b.2) = 0) → (λ * μ = -1) :=
sorry

end perpendicular_vectors_condition_l457_457121


namespace total_assignments_l457_457655

theorem total_assignments :
  ∃ (x : ℕ), (6 * x = 6 * 2 + 8 * (x - 5)) ∧ 6 * x = 84 := by
safety functionality

end total_assignments_l457_457655


namespace jelly_beans_remaining_l457_457656

def children_drawing_jelly_beans (initial_jelly_beans : ℕ) (total_children : ℕ)
(percent_allowed : ℕ) (groups : List (ℕ × ℕ)) (excluded_ids : ℕ → Prop) : ℕ :=
let allowed_children := percent_allowed * total_children / 100
let id_ranges := groups.map (λ ⟨n, jelly_beans⟩, 
  List.range n).zip groups.map Prod.snd
let counts := id_ranges.map (λ ⟨ids, jelly_beans⟩, 
  (ids.filter (λ id, ¬ excluded_ids id)).length * jelly_beans)
let total_drawn := counts.sum
initial_jelly_beans - total_drawn

def example_problem_instance :=
  children_drawing_jelly_beans 
    2000 100 70
    [(9, 2), (25, 4), (20, 6), (15, 8), (15, 10), (14, 12)]
    (λ id, (id % 100 = 0) ∨ (id % 100 = 2) ∨ (id % 100 = 4) ∨ (id % 100 = 6) ∨
                (id % 100 = 8) ∨ (id % 100 = 20) ∨ (id % 100 = 22) ∨ (id % 100 = 24) ∨
                (id % 100 = 26) ∨ (id % 100 = 28))

theorem jelly_beans_remaining : example_problem_instance = 1324 := 
sorry

end jelly_beans_remaining_l457_457656


namespace solve_y_l457_457276

theorem solve_y (y : ℂ) : 
  (8 * y^2 + 135 * y + 5) / (3 * y + 35) = 4 * y + 2 ↔ 
  y = (-11 + Complex.i * Real.sqrt 919) / 8 ∨ y = (-11 - Complex.i * Real.sqrt 919) / 8  := 
by
  -- Proof omitted
  sorry

end solve_y_l457_457276


namespace problem_equivalence_l457_457832

-- Condition: The sequence {a_n} is a geometric sequence with a_1 = 2 and common ratio q > 0
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ q > 0, ∀ n : ℕ, a (n + 1) = q * a n

-- Condition: a_2, 6, a_3 form an arithmetic sequence.
def is_arithmetic_sequence (a  : ℕ → ℝ) : Prop :=
  a 2 + a 3 = 12
  
-- The sequence a_n = 2^n
noncomputable def a (n : ℕ) : ℝ := 2 ^ n

-- Transformation to b_n and T_n
noncomputable def b (n : ℕ) : ℝ := Real.log 2 (a n)
noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (b i * b (i + 1))

-- Theorem stating the complete problem in Lean
theorem problem_equivalence :
  (is_geometric_sequence a q ∧ is_arithmetic_sequence a) →
  (∀ n, a n = 2 ^ n) ∧ ∀ n : ℕ, T n < 6 / 7 → n ∈ {1, 2, 3, 4, 5} :=
by
  sorry

end problem_equivalence_l457_457832


namespace point_in_fourth_quadrant_l457_457532

def point_x := 3
def point_y := -4

def first_quadrant (x y : Int) := x > 0 ∧ y > 0
def second_quadrant (x y : Int) := x < 0 ∧ y > 0
def third_quadrant (x y : Int) := x < 0 ∧ y < 0
def fourth_quadrant (x y : Int) := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant : fourth_quadrant point_x point_y :=
by
  simp [point_x, point_y, fourth_quadrant]
  sorry

end point_in_fourth_quadrant_l457_457532


namespace candy_problem_l457_457922

theorem candy_problem (n : ℕ) (h : n ∈ [2, 5, 9, 11, 14]) : ¬(23 - n) % 3 ≠ 0 → n = 9 := by
  sorry

end candy_problem_l457_457922


namespace lambda_mu_relationship_l457_457143

variables (λ μ : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def linear_combination (v : ℝ × ℝ) (c : ℝ) (w : ℝ × ℝ) := 
  (v.1 + c * w.1, v.2 + c * w.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem lambda_mu_relationship (h : dot_product (linear_combination vector_a λ vector_b)
                                          (linear_combination vector_a μ vector_b) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457143


namespace trajectory_of_Q_range_of_k_l457_457461

-- Definition of the circle and relevant points
def center_C: ℝ × ℝ := (-1, 0)
def point_A: ℝ × ℝ := (1, 0)

-- Definitions that follow from conditions
def circle (x y : ℝ) : Prop := ((x + 1)^2 + y^2 = 8)
def P_moving_on_circle (x y : ℝ) : Prop := circle x y
def Q_on_CP (px py qx qy : ℝ) : Prop := 
  ∃ t : ℝ, qx = (1 - t) * (-1) + t * px ∧ qy = t * py
def AP_2AM (px py mx my : ℝ) : Prop := 
  ∃ t : ℝ, mx = t * (px + 1) / 2 + (1 - t) * 1 ∧ my = t * py / 2

-- Lean statement for problem (1)
theorem trajectory_of_Q : 
  (∀ (px py qx qy : ℝ), P_moving_on_circle px py → Q_on_CP px py qx qy → ( ( qx^2/2 + qy^2 = 1 ))) := 
sorry

-- Definitions for problem (2)
def line_l (k b x y : ℝ) : Prop := y = k * x + b
def tangent_to_unit_circle (k b : ℝ) : Prop := b^2 = k^2 + 1
def trajectory_of_Q_intersection (kx1 ky1 kx2 ky2 : ℝ) : Prop :=
  line_l k kx1 ky1 → line_l k kx2 ky2 → ∃ (px1 py1 px2 py2 : ℝ), P_moving_on_circle px1 py1 → Q_on_CP px1 py1 kx1 ky1 → Q_on_CP px2 py2 kx2 ky2
def vector_dot_product_range (x1 y1 x2 y2 : ℝ) : Prop := 
  ∃ (O : ℝ × ℝ), 3/4 ≤ (x1*x2 + y1*y2) ∧ (x1*x2 + y1*y2) ≤ 4/5

-- Lean statement for problem (2)
theorem range_of_k (k : ℝ) : 
  ∀ b x1 y1 x2 y2, 
    tangent_to_unit_circle k b → 
    trajectory_of_Q_intersection x1 y1 x2 y2 →
    vector_dot_product_range x1 y1 x2 y2 →
    -Real.sqrt(2)/2 ≤ k ∧ k ≤ -Real.sqrt(3)/3 ∨ Real.sqrt(3)/3 ≤ k ∧ k ≤ Real.sqrt(2)/2 :=
sorry

end trajectory_of_Q_range_of_k_l457_457461


namespace multiply_and_simplify_l457_457972
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l457_457972


namespace find_vector_l457_457013

-- Define the parametrization of lines l and m
structure Line (ℝ : Type) :=
  (x : ℝ → ℝ)
  (y : ℝ → ℝ)

def l : Line ℝ := {
  x := λ t, 2 + 5 * t,
  y := λ t, 3 + 2 * t 
}

def m : Line ℝ := {
  x := λ s, -7 + 5 * s,
  y := λ s, 9 + 2 * s 
}

theorem find_vector {ℝ : Type} [linear_ordered_field ℝ] : 
  ∃ (v : ℝ × ℝ), let v1 := v.1, v2 := v.2 in v2 - v1 = 7 ∧ 
  (v1, v2) = (-2 : ℝ, 5 : ℝ) :=
by {
  use (-2, 5),
  simp
}

#print axioms find_vector

end find_vector_l457_457013


namespace incorrect_statement_D_l457_457794

theorem incorrect_statement_D (x1 x2 : ℝ) (hx : x1 < x2) :
  ¬ (y : ℝ) (y1 := -3 / x1) (y2 := -3 / x2) y1 < y2 :=
by
  sorry

end incorrect_statement_D_l457_457794


namespace multiply_expression_l457_457990

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l457_457990


namespace lambda_mu_relationship_l457_457135

theorem lambda_mu_relationship (a b : ℝ × ℝ) (λ μ : ℝ)
  (h1 : a = (1, 1))
  (h2 : b = (1, -1))
  (h3 : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457135


namespace cos_double_angle_l457_457799

variable (α : ℝ)
variables (h : sin α + cos α = 2 / 3)

theorem cos_double_angle : cos (2 * α) = ± (2 * real.sqrt 14 / 9) := 
sorry

end cos_double_angle_l457_457799


namespace find_a_l457_457569

noncomputable def a : ℕ := sorry

def A (a : ℕ) : Set ℕ := {2, 3, a^2 + 2a - 3}
def B (a : ℕ) : Set ℕ := {abs(a + 3), 2}

axiom h1 : 5 ∈ A a
axiom h2 : 5 ∉ B a

theorem find_a : a = -4 :=
by
  sorry

end find_a_l457_457569


namespace translated_line_value_m_l457_457874

theorem translated_line_value_m :
  (∀ x y : ℝ, (y = x → y = x + 3) → y = 2 + 3 → ∃ m : ℝ, y = m) :=
by sorry

end translated_line_value_m_l457_457874


namespace even_three_digit_numbers_less_than_400_l457_457670

theorem even_three_digit_numbers_less_than_400 (digits : Set ℕ) (uses_more_than_once : ∀d ∈ digits, ∃n ≥ 1, count d ≥ n):
  digits = {1, 2, 3, 4, 5} →
  ∃ n, n = 15 ∧
  ∀ number, number < 400 →
  number ≥ 100 →
  (even number) →
  (number % 5 = 0) →
  (∀ digit ∈ digits, ∃ n ≥ 1, count digit ≥ n) →
  n = ∑ number in numbers_of_different_digits digits, 1 :=
by 
  intros
  sorry

end even_three_digit_numbers_less_than_400_l457_457670


namespace subsets_with_at_least_four_adjacent_chairs_in_circle_l457_457314

theorem subsets_with_at_least_four_adjacent_chairs_in_circle : 
  let chairs := ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)
  let is_circle (s : set ℕ) := (∀ i ∈ s, (i + 1) % 12 ∈ s ∨ (i + 2) % 12 ∈ s ∨ (i + 3) % 12 ∈ s ∨ (i + 4) % 12 ∈ s)
  let subsets_with_at_least_four_adj_chair_sets := {s | s ⊆ chairs ∧ is_circle s}
  let count_subsets := λ S, (S.card : ℕ)
  count_subsets subsets_with_at_least_four_adj_chair_sets = 1658 :=
sorry

end subsets_with_at_least_four_adjacent_chairs_in_circle_l457_457314


namespace cos_18_eq_l457_457004

-- Definitions for the conditions.
def a := Real.cos (36 * Real.pi / 180)
def c := Real.cos (18 * Real.pi / 180)

-- Statement of the problem
theorem cos_18_eq :
  c = (Real.sqrt (10 + 2 * Real.sqrt 5) / 4) :=
by
  -- conditions given in the problem
  have h1: a = 2 * c^2 - 1, from sorry
  have h2: Real.sin (36 * Real.pi / 180) = Real.sqrt (1 - a^2), from sorry
  have triple_angle: Real.cos (3 * θ) = 4 * (Real.cos θ)^3 - 3 * (Real.cos θ), from sorry
  sorry

end cos_18_eq_l457_457004


namespace yearly_cost_of_oil_changes_l457_457920

-- Definitions of conditions
def miles_per_month : ℕ := 1000
def months_in_year : ℕ := 12
def oil_change_frequency : ℕ := 3000
def free_oil_changes_per_year : ℕ := 1
def cost_per_oil_change : ℕ := 50

theorem yearly_cost_of_oil_changes : 
  let total_miles := miles_per_month * months_in_year in
  let total_oil_changes := total_miles / oil_change_frequency in
  let paid_oil_changes := total_oil_changes - free_oil_changes_per_year in
  paid_oil_changes * cost_per_oil_change = 150 := 
by
  sorry

end yearly_cost_of_oil_changes_l457_457920


namespace multiplication_identity_multiplication_l457_457995

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l457_457995


namespace hyperbola_eccentricity_l457_457619

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h_chord: ∀ (x y : ℝ), ((x-8)^2 + y^2 = 25) → (bx + ay = 0) → ((x - 8)^2 + y^2 = 36)) :
  (∃ e : ℝ, e = (2 * sqrt 3) / 3) :=
by
  -- Establish conditions for hyperbola and circle
  have h_hyperbola := (a^2 = 3 * b^2),
  have h_c := (c^2 = 4 * b^2),
  -- Converting to the required eccentricity
  let c := sqrt (4 * b^2),
  let e := c / a,
  rw [h_hyperbola, h_c],
  sorry

end hyperbola_eccentricity_l457_457619


namespace solve_for_x_l457_457866

-- Assume x is a positive integer
def pos_integer (x : ℕ) : Prop := 0 < x

-- Assume the equation holds for some x
def equation (x : ℕ) : Prop :=
  1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170

-- Proposition stating that if x satisfies the equation then x must be 5
theorem solve_for_x (x : ℕ) (h1 : pos_integer x) (h2 : equation x) : x = 5 :=
by
  sorry

end solve_for_x_l457_457866


namespace find_x_l457_457854

open_locale real_inner_product

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

theorem find_x (x : ℝ) (h : (x - 1) * 1 + 2 * x = 0) : x = 1 / 3 :=
by
  sorry

end find_x_l457_457854


namespace find_missing_number_l457_457501

theorem find_missing_number :
  ∀ (x y : ℝ),
    (12 + x + 42 + 78 + 104) / 5 = 62 →
    (128 + y + 511 + 1023 + x) / 5 = 398.2 →
    y = 255 :=
by
  intros x y h1 h2
  sorry

end find_missing_number_l457_457501


namespace lambda_mu_condition_l457_457108

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem lambda_mu_condition (λ μ : ℝ) :
  orthogonal (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
             (vector_a.1 + μ * vector_b.1, vector_a.2 + μ * vector_b.2) →
  λ * μ = -1 :=
by
  sorry

end lambda_mu_condition_l457_457108


namespace total_cans_collected_l457_457697

variable (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ)

def total_bags : ℕ := bags_saturday + bags_sunday

theorem total_cans_collected 
  (h_sat : bags_saturday = 5)
  (h_sun : bags_sunday = 3)
  (h_cans : cans_per_bag = 5) : 
  total_bags bags_saturday bags_sunday * cans_per_bag = 40 :=
by
  sorry

end total_cans_collected_l457_457697


namespace find_divisor_l457_457343

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161) 
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 16 :=
by
  sorry

end find_divisor_l457_457343


namespace complex_sub_add_eq_l457_457494

-- Given conditions
def A : ℂ := 5 - 2 * complex.I
def B : ℂ := -3 + 4 * complex.I
def C : ℂ := 0 + 2 * complex.I
def D : ℂ := 3

-- The theorem to prove
theorem complex_sub_add_eq : A - B + C - D = 5 - 4 * complex.I := by
  sorry

end complex_sub_add_eq_l457_457494


namespace coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l457_457671

theorem coefficient_of_x9_in_expansion_of_x_minus_2_pow_10 :
  ∃ c : ℤ, (x - 2)^10 = ∑ k in finset.range (11), (nat.choose 10 k) * x^k * (-2)^(10 - k) ∧ c = -20 := 
begin 
  use -20,
  { sorry },
end

end coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l457_457671


namespace limit_integral_equivalence_l457_457783

noncomputable def sequence_limit (p : ℝ) (n : ℕ) : ℝ :=
  (∑ i in Finset.range (n + 1), (i : ℝ)^p) / (n : ℝ)^(p + 1)

theorem limit_integral_equivalence (p : ℝ) (hp : p > 0) :
  tendsto (λ n, sequence_limit p n) at_top (𝓝 (∫ (x : ℝ) in 0..1, x^p)) :=
sorry

end limit_integral_equivalence_l457_457783


namespace find_a_l457_457472

-- Definitions and assumptions based on the problem statement
def f (x : ℝ) : ℝ := Real.sqrt (1 - 2^x)
def A : Set ℝ := { x | x ≤ 0 }

def g (x : ℝ) (a : ℝ) : ℝ := Math.log ((x - a + 1) * (x - a - 1))
def B (a : ℝ) : Set ℝ := { x | x < a - 1 ∨ x > a + 1 }

-- Problem statement
theorem find_a (a : ℝ) (h : A ⊆ B a) : a > 1 :=
  sorry

end find_a_l457_457472


namespace times_when_hands_form_120_degrees_l457_457401

-- Definition of the positions at 7:00
def hour_hand_angle : ℝ := (7 / 12) * 360
def minute_hand_angle : ℝ := 0

-- Relative speed of the minute hand to the hour hand
def relative_speed : ℝ := 360 - 30

-- Calculate the time when the angle between the hands is 120 degrees
noncomputable def times_for_120_degree : list (ℕ × ℕ) :=
  [ (7, 16), (7, 27) ]

-- Theorem to prove the solution
theorem times_when_hands_form_120_degrees :
  times_for_120_degree = [ (7, 16), (7, 27) ] :=
sorry

end times_when_hands_form_120_degrees_l457_457401


namespace tangential_iff_equal_angle_incircles_l457_457435

variables (A B C D P Q H : Point)
variables [convex_quadrilateral A B C D]
variables [point_of_intersection P (ray BA) (ray CD)]
variables [point_of_intersection Q (ray BC) (ray AD)]
variables [foot_of_perpendicular H D (line PQ)]

theorem tangential_iff_equal_angle_incircles :
  (is_circumscribed_quadrilateral A B C D) ↔
  (viewed_from_equal_angles_incircled H (triangle ADP) (triangle CDQ)) :=
sorry

end tangential_iff_equal_angle_incircles_l457_457435


namespace calculate_AX_l457_457555

/-- Let O be the center of a circle with radius 1.
Let points A, B, C, and D lie on the circle such that AB is a diameter and CD is a chord.
Point X is on the circle such that OX bisects ∠CXD.
If ∠AXB = 90° and 3∠BAC = ∠CXD = 72°, calculate the length AX. -/
theorem calculate_AX (O A B C D X : Point)
    (h1 : dist O A = 1)
    (h2 : dist O B = 1)
    (h3 : dist O C = 1)
    (h4 : dist O D = 1)
    (h5 : dist O X = 1)
    (h6 : ∠OXC = ∠OXD) -- OX bisects ∠CXD
    (h7 : ∠AXB = 90°)
    (h8 : 3 * ∠BAC = ∠CXD)
    (h9 : ∠CXD = 72°) :
    dist A X = sin (18 * π / 180) :=
by
  sorry

end calculate_AX_l457_457555


namespace polygon_interior_angle_sum_l457_457952

-- Define the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def interior_to_exterior_ratio (θ : ℝ) : ℝ := 7.5 * (180 / (7.5 + 1))

-- Lean theorem stating the proof problem
theorem polygon_interior_angle_sum (n : ℕ) (S : ℝ)
  (h₁ : S = sum_of_interior_angles n)
  (h₂ : ∀ θ, θ = interior_to_exterior_ratio θ) :
  S = 2700 ∧ (∃ n, n = 17) :=
by
  sorry -- Proof to be completed

end polygon_interior_angle_sum_l457_457952


namespace denominator_of_repeating_decimal_l457_457289

-- Define the repeating decimal as a real number
def repeating_decimal (s : String) : ℝ :=
  sorry  -- placeholder for a function converting repeating decimal strings to real numbers

-- Main theorem statement
theorem denominator_of_repeating_decimal :
  ∀ S, (S = repeating_decimal "0.75") →
  let frac := 75 / 99 in
  let reduced_frac := {num := 25, denom := 33, coprime := by norm_num} in
  frac = reduced_frac.num / reduced_frac.denom →
  reduced_frac.denom = 33 :=
sorry

end denominator_of_repeating_decimal_l457_457289


namespace area_of_region_S_l457_457557

/-- Let ⌊x⌋ denote the greatest integer not exceeding the real number x. 
    Define the set S as S = { (x, y) | |⌊x + y⌋| + |⌊x - y⌋| ≤ 1 }.
    Prove that the area of the region represented by S in the plane is 5 / 2. --/
theorem area_of_region_S : 
  let floor := λ z : ℝ, ⌊z⌋ in
  let S := {p : ℝ × ℝ | |floor (p.1 + p.2)| + |floor (p.1 - p.2)| ≤ 1} in
  set_theory.area S = 5 / 2 :=
sorry

end area_of_region_S_l457_457557


namespace area_PM_N_l457_457627

noncomputable def parabola : (ℝ × ℝ) → Prop :=
λ p, p.2 ^ 2 = 4 * p.1

noncomputable def line (t : ℝ) : (ℝ × ℝ) → Prop :=
λ p, p.1 = t * p.2 + 7

def focus : ℝ × ℝ := (1, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2

noncomputable def intersects_parabola_line (t : ℝ) (M N : (ℝ × ℝ)) : Prop :=
(parabola M ∧ line t M) ∧ (parabola N ∧ line t N)

noncomputable def tangents_intersect_at (M N P : (ℝ × ℝ)) : Prop :=
let tangent_at_M := {p | ∃ k, p.2 = k * (p.1 - M.1) + M.2} in
let tangent_at_N := {p | ∃ k, p.2 = k * (p.1 - N.1) + N.2} in
tangent_at_M P ∧ tangent_at_N P

noncomputable def vectors_dot_product_zero (M N : (ℝ × ℝ)) : Prop :=
let MF := (1 - M.1, 0 - M.2) in
let NF := (1 - N.1, 0 - N.2) in
dot_product MF NF = 0

def area_of_triangle (P M N : (ℝ × ℝ)) : ℝ :=
(1 / 2) * abs (P.1 * (M.2 - N.2) + M.1 * (N.2 - P.2) + N.1 * (P.2 - M.2))

theorem area_PM_N
  (M N P : (ℝ × ℝ))
  (t : ℝ)
  (h1 : intersects_parabola_line t M N)
  (h2 : vectors_dot_product_zero M N)
  (h3 : tangents_intersect_at M N P) :
  area_of_triangle P M N = 108 := by
  sorry

end area_PM_N_l457_457627


namespace subsets_P_count_l457_457102

open Set

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def P : Set ℕ := M ∩ N

theorem subsets_P_count : (finite_subsets P).card = 4 := by
  sorry

end subsets_P_count_l457_457102


namespace range_of_f_l457_457300

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (x^2 - 2 * x + 2)

theorem range_of_f :
  set.range f = {y | 0 < y ∧ y ≤ 1 / 2} :=
by
  sorry

end range_of_f_l457_457300


namespace log12_eq_abc_l457_457066

theorem log12_eq_abc (a b : ℝ) (h1 : a = Real.log 7 / Real.log 6) (h2 : b = Real.log 4 / Real.log 3) : 
  Real.log 7 / Real.log 12 = (a * b + 2 * a) / (2 * b + 2) :=
by
  sorry

end log12_eq_abc_l457_457066


namespace incorrect_transformation_l457_457333

theorem incorrect_transformation :
  ¬ ∀ (a b c : ℝ), ac = bc → a = b :=
by
  sorry

end incorrect_transformation_l457_457333


namespace fly_reaches_x_coordinate_l457_457359

theorem fly_reaches_x_coordinate (n : ℕ) : 
  (∀ (x : ℕ), x < n → random_walk_2d x 0) → 
  ∀ (x y : ℕ), ∃ (t : ℕ), (random_walk t (0, 0) = (2011, y)) :=
by
  sorry

end fly_reaches_x_coordinate_l457_457359


namespace perpendicular_dot_product_l457_457133

theorem perpendicular_dot_product (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → (λ * μ = -1) :=
by
  intros
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  sorry

end perpendicular_dot_product_l457_457133


namespace angle_EDF_in_triangle_l457_457548

theorem angle_EDF_in_triangle
  (A B C D E F : Type)
  [metric_space A] [metric_space B] 
  [metric_space C] [metric_space D] 
  [metric_space E] [metric_space F]
  (angle_A : real)
  (BD BE CD CF : real)
  (BD_EQ_BE : BD = BE)
  (CD_EQ_CF : CD = CF)
  (angle_A_EQ_80 : angle_A = 80) :
  ∃ angle_EDF : real, angle_EDF = 50 :=
by {
  sorry
}

end angle_EDF_in_triangle_l457_457548


namespace area_cosine_enclosed_l457_457284

noncomputable def area_enclosed_by_cosine : ℝ :=
  2 * ∫ x in 0..(Real.pi / 2), Real.cos x

theorem area_cosine_enclosed :
  area_enclosed_by_cosine = 2 := by
  sorry

end area_cosine_enclosed_l457_457284


namespace correct_inequality_l457_457287

theorem correct_inequality (x : ℕ) (h : 500 ≥ 10 * 0.8 * x) : 10 * 0.8 * x ≤ 500 :=
by
  exact h -- Given this is our assumption.

end correct_inequality_l457_457287


namespace all_cells_happy_l457_457769

def is_happy (board : ℕ → ℕ → Prop) (i j : ℕ) : Prop :=
  let neighbors := [(i-1, j), (i+1, j), (i, j-1), (i, j+1)] in
  (list.count (λ (p : ℕ × ℕ), if (1 ≤ p.1 ∧ p.1 ≤ 10 ∧ 1 ≤ p.2 ∧ p.2 ≤ 10) then board p.1 p.2 else false) neighbors) = 2

def checkerboard (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

theorem all_cells_happy : 
  ∀ i j, 1 ≤ i ∧ i ≤ 10 ∧ 1 ≤ j ∧ j ≤ 10 → is_happy checkerboard i j :=
by
  sorry

end all_cells_happy_l457_457769


namespace positive_difference_sum_even_odd_l457_457323

theorem positive_difference_sum_even_odd :
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  sum_30_even - sum_25_odd = 305 :=
by
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  show sum_30_even - sum_25_odd = 305
  sorry

end positive_difference_sum_even_odd_l457_457323


namespace integral_neg_x_plus_1_zero_l457_457682

theorem integral_neg_x_plus_1_zero : ∫ x in 0..2, (-x + 1) = 0 := 
sorry

end integral_neg_x_plus_1_zero_l457_457682


namespace problem1_problem2_l457_457348

-- Problem 1
theorem problem1 : (-2) ^ 2 + (Real.sqrt 2 - 1) ^ 0 - 1 = 4 := by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) (A : ℝ) (B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) : a > 2 := by
  sorry

end problem1_problem2_l457_457348


namespace exist_r_sets_not_union_l457_457051

theorem exist_r_sets_not_union (n : ℕ) (h : n ≥ 5) (A : Finset (Finset ℕ)) (hA : A.card = n) :
  ∃ (r : ℕ) (B : Finset (Finset ℕ)), r = Nat.floor (Real.sqrt (2 * n)) ∧ 
    B.card = r ∧ ∀ (i j k : Fin r), i ≠ j → i ≠ k → j ≠ k → (B i) ≠ (B j ∪ B k) :=
  sorry

end exist_r_sets_not_union_l457_457051


namespace arithmetic_sequence_geometric_property_l457_457814

theorem arithmetic_sequence_geometric_property (a : ℕ → ℤ) (d : ℤ) (h_d : d = 2)
  (h_a3 : a 3 = a 1 + 4) (h_a4 : a 4 = a 1 + 6)
  (geo_seq : (a 1 + 4) * (a 1 + 4) = a 1 * (a 1 + 6)) :
  a 2 = -6 := sorry

end arithmetic_sequence_geometric_property_l457_457814


namespace number_of_non_factorial_tails_l457_457025

def factorial_tails : ℕ → ℕ 
| 0     := 0
| (n+1) := (n+1) / 5 + factorial_tails n / 5

theorem number_of_non_factorial_tails (n : ℕ) (h : n < 500) : 
  (finset.range 500).filter (λ x, ∀ m, factorial_tails m ≠ x).card = 96 := by
  sorry

end number_of_non_factorial_tails_l457_457025


namespace bananas_needed_to_make_yogurts_l457_457770

theorem bananas_needed_to_make_yogurts 
    (slices_per_yogurt : ℕ) 
    (slices_per_banana: ℕ) 
    (number_of_yogurts: ℕ) 
    (total_needed_slices: ℕ) 
    (bananas_needed: ℕ) 
    (h1: slices_per_yogurt = 8)
    (h2: slices_per_banana = 10)
    (h3: number_of_yogurts = 5)
    (h4: total_needed_slices = number_of_yogurts * slices_per_yogurt)
    (h5: bananas_needed = total_needed_slices / slices_per_banana): 
    bananas_needed = 4 := 
by
    sorry

end bananas_needed_to_make_yogurts_l457_457770


namespace probability_log_interval_l457_457426

open Set Real

noncomputable def probability_in_interval (a b c d : ℝ) (I J : Set ℝ) := 
  (b - a) / (d - c)

theorem probability_log_interval : 
  probability_in_interval 2 4 0 6 (Icc 0 6) (Ioo 2 4) = 1 / 3 := 
sorry

end probability_log_interval_l457_457426


namespace base_prime_rep_360_l457_457399

-- Define the value 360 as n
def n : ℕ := 360

-- Function to compute the base prime representation.
noncomputable def base_prime_representation (n : ℕ) : ℕ :=
  -- Normally you'd implement the actual function to convert n to its base prime representation here
  sorry

-- The theorem statement claiming that the base prime representation of 360 is 213
theorem base_prime_rep_360 : base_prime_representation n = 213 := 
  sorry

end base_prime_rep_360_l457_457399


namespace avg_weight_l457_457617

theorem avg_weight (A B C : ℝ)
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by sorry

end avg_weight_l457_457617


namespace point_in_fourth_quadrant_l457_457524

-- Define a structure for a Cartesian point
structure Point where
  x : ℝ
  y : ℝ

-- Define different quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Function to determine the quadrant of a given point
def quadrant (p : Point) : Quadrant :=
  if p.x > 0 ∧ p.y > 0 then Quadrant.first
  else if p.x < 0 ∧ p.y > 0 then Quadrant.second
  else if p.x < 0 ∧ p.y < 0 then Quadrant.third
  else Quadrant.fourth

-- The main theorem stating the point (3, -4) lies in the fourth quadrant
theorem point_in_fourth_quadrant : quadrant { x := 3, y := -4 } = Quadrant.fourth :=
  sorry

end point_in_fourth_quadrant_l457_457524


namespace multiply_polynomials_l457_457983

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l457_457983


namespace Katy_jellybeans_l457_457258

variable (Matt Matilda Steve Katy : ℕ)

def jellybean_relationship (Matt Matilda Steve Katy : ℕ) : Prop :=
  (Matt = 10 * Steve) ∧
  (Matilda = Matt / 2) ∧
  (Steve = 84) ∧
  (Katy = 3 * Matilda) ∧
  (Katy = Matt / 2)

theorem Katy_jellybeans : ∃ Katy, jellybean_relationship Matt Matilda Steve Katy ∧ Katy = 1260 := by
  sorry

end Katy_jellybeans_l457_457258


namespace relationship_M_N_l457_457059

-- Define M and N based on the given conditions
def M (x y : ℝ) : ℝ := -x^2 - 4y^2 + 2y
def N (x y : ℝ) : ℝ := 6x - 2y + 12

-- State the theorem to prove M < N
theorem relationship_M_N (x y : ℝ) : M x y < N x y :=
by
  -- Proof goes here, but it is omitted in this statement.
  sorry

end relationship_M_N_l457_457059


namespace a_100_correct_l457_457465

variable (a_n : ℕ → ℕ) (S₉ : ℕ) (a₁₀ : ℕ)

def is_arth_seq (a_n : ℕ → ℕ) := ∃ a d, ∀ n, a_n n = a + n * d

noncomputable def a_100 (a₅ d : ℕ) : ℕ := a₅ + 95 * d

theorem a_100_correct
  (h1 : ∃ S₉, 9 * a_n 4 = S₉)
  (h2 : a_n 9 = 8)
  (h3 : is_arth_seq a_n) :
  a_100 (a_n 4) 1 = 98 :=
by
  sorry

end a_100_correct_l457_457465


namespace num_of_false_props_is_2_l457_457064

-- Definitions of the conditions
inductive Proposition : Type
| P1 | P2 | P3 | P4

def isTrue (P : Proposition) : Prop :=
match P with
| Proposition.P1 => True      -- [Condition: For any two lines in space, there exist infinitely many planes with equal angles formed with these two lines.]
| Proposition.P2 => True      -- [Condition: For any three pairwise skew lines in space, there exist infinitely many lines that intersect all three.]
| Proposition.P3 => False     -- [Condition: For any four pairwise skew lines in space, there is no line that intersects all four.]
| Proposition.P4 => False     -- [Condition: If three planes intersect pairwise, producing three non-coincident lines, then these three lines are concurrent.]

def countFalseProps : Nat :=
[Proposition.P1, Proposition.P2, Proposition.P3, Proposition.P4].count (fun P => ¬(isTrue P))

-- The theorem we need to prove
theorem num_of_false_props_is_2 : countFalseProps = 2 := by sorry

end num_of_false_props_is_2_l457_457064


namespace perpendicular_vectors_condition_l457_457124

theorem perpendicular_vectors_condition (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
      b : ℝ × ℝ := (1, -1)
  in ((a.1 + λ * b.1) * (a.1 + μ * b.1) + (a.2 + λ * b.2) * (a.2 + μ * b.2) = 0) → (λ * μ = -1) :=
sorry

end perpendicular_vectors_condition_l457_457124


namespace find_value_of_f_6_minus_a_l457_457841

def f (x : ℝ) : ℝ :=
if x < 1 then sin (π * x / 3) else -real.log x / real.log 2

variable (a : ℝ)
axiom h_condition : f a = -3

theorem find_value_of_f_6_minus_a : f (6 - a) = -sqrt 3 / 2 := by
  sorry

end find_value_of_f_6_minus_a_l457_457841


namespace polynomial_divisibility_n_l457_457765

theorem polynomial_divisibility_n :
  ∀ (n : ℤ), (∀ x, x = 2 → 3 * x^2 - 4 * x + n = 0) → n = -4 :=
by
  intros n h
  have h2 : 3 * 2^2 - 4 * 2 + n = 0 := h 2 rfl
  linarith

end polynomial_divisibility_n_l457_457765


namespace greatest_sum_base_nine_l457_457676

theorem greatest_sum_base_nine (n : ℕ) (h1 : n > 0) (h2 : n < 5000) :
  ∃ m : ℕ, m < 5000 ∧ (base_nine_digit_sum m = 26) :=
sorry

noncomputable def base_nine_digit_sum (n : ℕ) : ℕ :=
sorry

end greatest_sum_base_nine_l457_457676


namespace multiply_expression_l457_457987

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l457_457987


namespace lambda_mu_relationship_l457_457137

theorem lambda_mu_relationship (a b : ℝ × ℝ) (λ μ : ℝ)
  (h1 : a = (1, 1))
  (h2 : b = (1, -1))
  (h3 : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457137


namespace find_f_neg_2013_l457_457628

variable (f : ℝ → ℝ)

# Assume f is an even function
axiom even_function : ∀ x : ℝ, f (-x) = f x

# Assume f satisfies the functional equation
axiom functional_equation : ∀ x : ℝ, f (x + 1) + f x = 1

# Assume f is defined on the interval [1, 2]
axiom interval_definition : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = 2 - x

-- Prove that f (-2013) = 1
theorem find_f_neg_2013 : f (-2013) = 1 := by
  sorry

end find_f_neg_2013_l457_457628


namespace part1_part2_l457_457447

noncomputable def circleM (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1
noncomputable def lineL (x y : ℝ) : Prop := x - 2 * y = 0

theorem part1 (t : ℝ) (x y: ℝ) :
  t = 0 → circleM 0 2 → lineL 0 2 → ∃ k : ℝ, (y - 1 = k * (x - 2)) ∧ 
  (MP = sqrt 5) ∧ 
  ((y = 1) ∨ (4 * x + 3 * y - 11 = 0)) :=
  sorry

theorem part2 (t : ℝ) :
  circleM 0 2 → lineL 0 2 → ∃ a : ℝ, MP = sqrt 5 ∧ (
    (t > -4/5 ∧ L(t) = 1/4 * sqrt(5 * t^2 + 8 * t + 16)) ∨
    (-24/5 ≤ t ∧ t ≤ -4/5 ∧ L(t) = 2 * sqrt(5) / 5) ∨
    (t < -24/5 ∧ L(t) = 1/4 * sqrt(5 * t^2 + 48 * t + 128))) :=
  sorry

end part1_part2_l457_457447


namespace equation_of_hyperbola_l457_457467

-- Define the ellipse and hyperbola conditions
def ellipseEquation := ∀ (x y : ℝ), x^2 + (y^2) / 2 = 1
def hyperbolaEquation (x y : ℝ) := y^2 - x^2 = 2

-- Define the properties from the conditions
def endpointsMajorAxis (x y : ℝ) := ellipseEquation x y → (x = 0 ∧ y = sqrt 2) ∨ (x = 0 ∧ y = -sqrt 2)
def eccentricityEllipse := 1 / sqrt 2
def productEccentricities (e_hyperbola : ℝ) := e_hyperbola * eccentricityEllipse = 1

-- Hypothesis that ensures the conditions are met
def hyperbolaConditions := ∀ e_hyperbola (x y : ℝ),
  endpointsMajorAxis x y →
  productEccentricities e_hyperbola →
  e_hyperbola = sqrt 2

-- The final proof statement
theorem equation_of_hyperbola :
  hyperbolaConditions →
  ∀ (x y : ℝ), hyperbolaEquation x y :=
by
  intros
  sorry

end equation_of_hyperbola_l457_457467


namespace jack_buttons_total_l457_457906

theorem jack_buttons_total (k1_shirts kids1 n_buttons_per_shirt n_kids3 k3_shirts neighbor n_buttons_per_shirt_neigh) :
  (k1_shirts = 3) →
  (kids1 = 3) →
  (n_buttons_per_shirt = 7) →
  (neighbor = 2) →
  (k3_shirts = 3) →
  (n_buttons_per_shirt_neigh = 9) →
  let shirts1 := kids1 * k1_shirts,
      buttons1 := shirts1 * n_buttons_per_shirt,
      shirts3 := neighbor * k3_shirts,
      buttons3 := shirts3 * n_buttons_per_shirt_neigh,
      total_buttons := buttons1 + buttons3
  in total_buttons = 117 :=
by
  intros h1 h2 h3 h4 h5 h6
  have shirts1 : ℕ := kids1 * k1_shirts := by simp [h1, h2]
  have buttons1 : ℕ := shirts1 * n_buttons_per_shirt := by simp [shirts1, h3]
  have shirts3 : ℕ := neighbor * k3_shirts := by simp [h4, h5]
  have buttons3 : ℕ := shirts3 * n_buttons_per_shirt_neigh := by simp [shirts3, h6]
  have total_buttons : ℕ := buttons1 + buttons3 := by simp [buttons1, buttons3]
  have eq_final : total_buttons = 117 := by simp [total_buttons]
  sorry

end jack_buttons_total_l457_457906


namespace lambda_mu_condition_l457_457106

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem lambda_mu_condition (λ μ : ℝ) :
  orthogonal (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
             (vector_a.1 + μ * vector_b.1, vector_a.2 + μ * vector_b.2) →
  λ * μ = -1 :=
by
  sorry

end lambda_mu_condition_l457_457106


namespace lcm_of_two_numbers_l457_457281

theorem lcm_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 6) (h_product : a * b = 432) :
  Nat.lcm a b = 72 :=
by 
  sorry

end lcm_of_two_numbers_l457_457281


namespace smallest_palindrome_not_five_digit_l457_457041

theorem smallest_palindrome_not_five_digit (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = n / 100 ∧ n / 10 % 10 = n / 100 ∧ 103 * n < 10000) :
  n = 707 := by
sorry

end smallest_palindrome_not_five_digit_l457_457041


namespace expand_binomial_trinomial_l457_457411

theorem expand_binomial_trinomial (x y z : ℝ) :
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 :=
by sorry

end expand_binomial_trinomial_l457_457411


namespace find_matrix_N_l457_457035

open Matrix

theorem find_matrix_N (N : Matrix (Fin 3) (Fin 3) ℝ) :
  N ⬝ (Matrix.of ![![(-5 : ℝ), 6, 0], ![7, (-8), 0], ![0, 0, 2]]) = (1 : Matrix (Fin 3) (Fin 3) ℝ) ->
  N = (Matrix.of ![![4, 3, 0], ![3.5, 2.5, 0], ![0, 0, 0.5]]) :=
by
  sorry

end find_matrix_N_l457_457035


namespace perpendicular_dot_product_l457_457128

theorem perpendicular_dot_product (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → (λ * μ = -1) :=
by
  intros
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  sorry

end perpendicular_dot_product_l457_457128


namespace gcd_2197_2208_is_1_l457_457675

def gcd_2197_2208 : ℕ := Nat.gcd 2197 2208

theorem gcd_2197_2208_is_1 : gcd_2197_2208 = 1 :=
by
  sorry

end gcd_2197_2208_is_1_l457_457675


namespace proof_equivalent_problem_l457_457523

noncomputable def EllipseEquation := ∀ (a b : ℝ) (h : a > b ∧ b > 0) (F : ℝ × ℝ), 
  (F.1^2 / a^2 + F.2^2 / b^2 = 1) → (a^2 = 2 * b^2 ∧ a = 2 ∧ b^2 = 2)

noncomputable def point_outside_circle := ∀ (G : ℝ × ℝ) (x y m : ℝ), 
  (G = (-9/4, 0)) → (m ≠ 0) → (x - m * y + 1 = 0) → (x^2 / 4 + y^2 / 2 = 1) →
  let C := (x, y) in let D := (x, y) in
  let circle := ∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = (x' - x)^2 + (y' - y)^2 →
  ¬ (G.1^2 + G.2^2 ≤ (x - x)^2 + (y - y)^2)

theorem proof_equivalent_problem : 
  EllipseEquation 2 (sqrt 2)
  ∧ point_outside_circle (-9 / 4, 0) 0 2 1 :=
by
  sorry

end proof_equivalent_problem_l457_457523


namespace team_leaders_lcm_l457_457738

/-- Amanda, Brian, Carla, and Derek are team leaders rotating every
    5, 8, 10, and 12 weeks respectively. Given that this week they all are leading
    projects together, prove that they will all lead projects together again in 120 weeks. -/
theorem team_leaders_lcm :
  Nat.lcm (Nat.lcm 5 8) (Nat.lcm 10 12) = 120 := 
  by
  sorry

end team_leaders_lcm_l457_457738


namespace find_n_mod_10_l457_457780

theorem find_n_mod_10 (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) (h₂ : n ≡ 123456 [MOD 10]) : n = 6 :=
by
  sorry

end find_n_mod_10_l457_457780


namespace cube_volume_doubled_l457_457339

theorem cube_volume_doubled (a : ℝ) : 
  let V := a^3 in
  let V' := (2 * a)^3 in
  V' = 8 * V :=
by
  sorry

end cube_volume_doubled_l457_457339


namespace sum_of_distinct_digits_div_1000_l457_457234

-- Define T as the sum of all four-digit positive integers with distinct digits
noncomputable def T : ℕ := 
  ∑ i in (Finset.filter (λ n : ℕ, (1000 ≤ n ∧ n ≤ 9999 ∧ (List.nodup (List.ofDigits 10 (Nat.digits 10 n)))))
  (Finset.range 10000)), i

-- Define the proof problem statement
theorem sum_of_distinct_digits_div_1000 : T % 1000 = 400 := 
  sorry

end sum_of_distinct_digits_div_1000_l457_457234


namespace find_principal_amount_l457_457340

-- Define the Simple Interest Function
def simple_interest (P : ℝ) (r : ℝ) (n : ℝ) : ℝ :=
  (P * r * n) / 100

-- Define the Compound Interest Function
def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) : ℝ :=
  P * (1 + r / 100) ^ n - P

-- Define the conditions
def conditions (P r : ℝ) : Prop :=
  simple_interest P r 2 = 900 ∧ 
  compound_interest P r 2 = 922.50

-- The theorem to prove the principal amount is 9000 given the conditions
theorem find_principal_amount (P : ℝ) (r : ℝ) (h : conditions P r) : P = 9000 := 
by
  sorry

end find_principal_amount_l457_457340


namespace parabola_ord_l457_457586

theorem parabola_ord {M : ℝ × ℝ} (h1 : M.1 = (M.2 * M.2) / 8) (h2 : dist M (2, 0) = 4) : M.2 = 4 ∨ M.2 = -4 := 
sorry

end parabola_ord_l457_457586


namespace maximum_M_for_right_triangle_l457_457453

theorem maximum_M_for_right_triangle (a b c : ℝ) (h1 : a ≤ b) (h2 : b < c) (h3 : a^2 + b^2 = c^2) :
  (1 / a + 1 / b + 1 / c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) :=
sorry

end maximum_M_for_right_triangle_l457_457453


namespace multiplication_in_S_l457_457554

-- Define the set S as given in the conditions
variable (S : Set ℝ)

-- Condition 1: 1 ∈ S
def condition1 : Prop := 1 ∈ S

-- Condition 2: ∀ a b ∈ S, a - b ∈ S
def condition2 : Prop := ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S

-- Condition 3: ∀ a ∈ S, a ≠ 0 → 1 / a ∈ S
def condition3 : Prop := ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

-- Theorem to prove: ∀ a b ∈ S, ab ∈ S
theorem multiplication_in_S (h1 : condition1 S) (h2 : condition2 S) (h3 : condition3 S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := 
  sorry

end multiplication_in_S_l457_457554


namespace area_of_shaded_region_l457_457623

theorem area_of_shaded_region :
  let a := 4
  let b := 5
  let area_shaded := a ^ 2 - ((a * b) / 2) / 2
  area_shaded = 3.5 :=
by
  let a := 4
  let b := 5
  let sq4 := a ^ 2
  let area_big_triangle := (a * b) / 2
  let area_small_triangle := (sq4 / 2)
  let area_shaded := area_small_triangle - area_big_triangle / 2
  have h : area_shaded = 3.5 := sorry
  exact h

end area_of_shaded_region_l457_457623


namespace abs_simplify_l457_457274

theorem abs_simplify : |6^2 - 9| = 27 := 
by 
  sorry

end abs_simplify_l457_457274


namespace fraction_tips_proof_l457_457263

variable (B : ℝ) -- base salary

-- Tips fractions for each week
variable (tips_week1 : ℝ := 5/3)
variable (tips_week2 : ℝ := 3/2)
variable (tips_week3 : ℝ := 1)  -- equivalent to B
variable (tips_week4 : ℝ := 4/3)

-- Weekly expenses
variable (weekly_expenses : ℝ := 1 / 10) * B
variable (total_weeks : ℕ := 4)

-- Total tips over four weeks
def total_tips : ℝ := tips_week1 * B + tips_week2 * B + tips_week3 * B + tips_week4 * B

-- Total income after expenses
def total_income_after_expenses : ℝ := 4 * B + total_tips - total_weeks * weekly_expenses

-- Fraction of total income after expenses that comes from tips
def fraction_tips_of_total_income : ℝ := total_tips / total_income_after_expenses

theorem fraction_tips_proof : fraction_tips_of_total_income B = 55 / 93 := by sorry

end fraction_tips_proof_l457_457263


namespace odd_function_strictly_decreasing_l457_457052

noncomputable def f (x : ℝ) : ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom negative_condition (x : ℝ) (hx : x > 0) : f x < 0

theorem odd_function : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem strictly_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
by sorry

end odd_function_strictly_decreasing_l457_457052


namespace total_fruits_in_bowl_l457_457879

theorem total_fruits_in_bowl (bananas apples oranges : ℕ) 
  (h1 : bananas = 2) 
  (h2 : apples = 2 * bananas) 
  (h3 : oranges = 6) : 
  bananas + apples + oranges = 12 := 
by 
  sorry

end total_fruits_in_bowl_l457_457879


namespace ellipse_equation_and_max_area_l457_457835

noncomputable def ellipse_standard_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∃ (b' : ℝ) (e : ℝ), b' = b ∧ a > b' ∧ b' = sqrt 3 ∧ e = 0.5 ∧ a^2 = b'^2 + (e * a)^2

noncomputable def maximum_area_triangle (a b : ℝ) (ha : a > b) (hb : b > 0) : ℝ :=
  let c := sqrt (a^2 - b^2) in
  let f1 := (0, 0) in
  let f2 := (c, 0) in
  3

theorem ellipse_equation_and_max_area :
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ ellipse_standard_equation a b (by linarith [show 2 > 0 from dec_trivial]) (by linarith [show sqrt 3 > 0 from dec_trivial]) ∧ 
  maximum_area_triangle a b (by exact zero_lt_two) (by exact sqrt_pos_of_pos zero_lt_three) = 3 :=
begin
  use [2, sqrt 3],
  split,
  { refl },
  split,
  { refl },
  split,
  { use [sqrt 3, 0.5],
    exact ⟨rfl, by linarith, rfl, rfl, by linarith⟩ },
  { exact rfl }
end

end ellipse_equation_and_max_area_l457_457835


namespace number_of_two_digit_integers_with_remainder_3_mod_9_l457_457160

theorem number_of_two_digit_integers_with_remainder_3_mod_9 : 
  {x : ℤ // 10 ≤ x ∧ x < 100 ∧ ∃ n : ℤ, x = 9 * n + 3}.card = 10 := by
sorry

end number_of_two_digit_integers_with_remainder_3_mod_9_l457_457160


namespace money_left_l457_457341

noncomputable def initial_amount : ℝ := 10.10
noncomputable def spent_on_sweets : ℝ := 3.25
noncomputable def amount_per_friend : ℝ := 2.20
noncomputable def remaining_amount : ℝ := initial_amount - spent_on_sweets - 2 * amount_per_friend

theorem money_left : remaining_amount = 2.45 :=
by
  sorry

end money_left_l457_457341


namespace hyperbola_asymptote_l457_457173

theorem hyperbola_asymptote (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∃ (x y : ℝ), (x, y) = (2, 1) ∧ 
       (y = (2 / a) * x ∨ y = -(2 / a) * x)) : a = 4 := by
  sorry

end hyperbola_asymptote_l457_457173


namespace line_parametric_and_curve_equation_l457_457437

theorem line_parametric_and_curve_equation (t : ℝ) :
  (∃ α ∈ (Set.Icc 0 (Real.pi)),
    (3 * (2 - t * (Real.sqrt 3) / 2) - (Real.sqrt 3) * (t / 2) - 6 = 0)) ∧
  (∀ θ : ℝ, (C : ℝ) (C = 4 * Real.sin θ → (2 * Real.sin θ * Real.cos θ) ^ 2 + (4 * Real.sin θ ^ 2 - 2) ^ 2 = 4)) ∧
  (∀ P : ℝ × ℝ, ∃ A : ℝ × ℝ, abs (dist P A) ≤ (2 * sqrt 3 + 4) ∧ abs (dist P A) ≥ (2 * sqrt 3)) :=
begin
  sorry
end

end line_parametric_and_curve_equation_l457_457437


namespace bus_ride_difference_l457_457588

theorem bus_ride_difference :
  ∀ (Oscar_bus Charlie_bus : ℝ),
  Oscar_bus = 0.75 → Charlie_bus = 0.25 → Oscar_bus - Charlie_bus = 0.50 :=
by
  intros Oscar_bus Charlie_bus hOscar hCharlie
  rw [hOscar, hCharlie]
  norm_num

end bus_ride_difference_l457_457588


namespace find_a_l457_457468

theorem find_a (a : ℝ) (h : (a - Complex.i) ^ 2 = 2 * Complex.i) : a = -1 :=
sorry

end find_a_l457_457468


namespace point_in_fourth_quadrant_l457_457530

def point_x := 3
def point_y := -4

def first_quadrant (x y : Int) := x > 0 ∧ y > 0
def second_quadrant (x y : Int) := x < 0 ∧ y > 0
def third_quadrant (x y : Int) := x < 0 ∧ y < 0
def fourth_quadrant (x y : Int) := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant : fourth_quadrant point_x point_y :=
by
  simp [point_x, point_y, fourth_quadrant]
  sorry

end point_in_fourth_quadrant_l457_457530


namespace compare_volumes_l457_457546

noncomputable def volume_ratios (A B C : ℝ) (r : ℝ) : ℝ × ℝ × ℝ :=
  (
    (1 - Real.sin (A / 2))^2 / Real.sin (A / 2),
    (1 - Real.sin (B / 2))^2 / Real.sin (B / 2),
    (1 - Real.sin (C / 2))^2 / Real.sin (C / 2)
  )

theorem compare_volumes (A B C : ℝ) (r : ℝ) :
  let ratio := volume_ratios A B C r in
  ratio =
    (
      (1 - Real.sin (A / 2))^2 / Real.sin (A / 2),
      (1 - Real.sin (B / 2))^2 / Real.sin (B / 2),
      (1 - Real.sin (C / 2))^2 / Real.sin (C / 2)
    ) := 
by
  sorry

end compare_volumes_l457_457546


namespace unique_3_digit_number_with_conditions_l457_457774

def valid_3_digit_number (n : ℕ) : Prop :=
  let d2 := n / 100
  let d1 := (n / 10) % 10
  let d0 := n % 10
  (d2 > 0) ∧ (d2 < 10) ∧ (d1 < 10) ∧ (d0 < 10) ∧ (d2 + d1 + d0 = 28) ∧ (d0 < 7) ∧ (d0 % 2 = 0)

theorem unique_3_digit_number_with_conditions :
  (∃! n : ℕ, valid_3_digit_number n) :=
sorry

end unique_3_digit_number_with_conditions_l457_457774


namespace completing_square_correctness_l457_457609

theorem completing_square_correctness :
  (2 * x^2 - 4 * x - 7 = 0) ->
  ((x - 1)^2 = 9 / 2) :=
sorry

end completing_square_correctness_l457_457609


namespace angle_RTS_is_142_25_degrees_l457_457668

noncomputable def radius : ℝ := sorry  -- Define the radius r, which will be given later

def distance (r : ℝ) : ℝ := r / 2  -- The distance between P and Q

variables (P Q R S T U : ℝ) (r : ℝ)
  (congruent_circles : true)  -- Identify the congruent circles condition
  (distance_half_radius : distance r = r / 2)
  (line_through_PQ_intersects_at_RS : true) -- Line through P and Q intersects the circles at R and S
  (circles_intersect_at_TU : true) -- Circles intersect at T and U

theorem angle_RTS_is_142_25_degrees :
  ∠ RTS = 142.25 :=
sorry

end angle_RTS_is_142_25_degrees_l457_457668


namespace inscribed_semicircle_radius_l457_457211

-- Define the variables and conditions
variables (PQ QR PR: ℝ) (r: ℝ)

-- Given conditions
def triangle_PQR_conditions :=
  PQ = 15 ∧ QR = 8 ∧ ∠90.PR = 17

-- Define the problem statement:
theorem inscribed_semicircle_radius (h: triangle_PQR_conditions PQ QR PR) :
  r = 24 / 5 :=
sorry

end inscribed_semicircle_radius_l457_457211


namespace problem_statement_l457_457937

noncomputable def a : ℝ := Real.logBase 5 4
noncomputable def b : ℝ := (Real.logBase 5 3) ^ 2
noncomputable def c : ℝ := Real.logBase 4 5

theorem problem_statement : b < a ∧ a < c := by
  sorry

end problem_statement_l457_457937


namespace problem_l457_457166

theorem problem (a b : ℝ) (h : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 4 * x + 3) : a + b = 4 :=
by
  sorry

end problem_l457_457166


namespace max_value_fraction_l457_457239

theorem max_value_fraction (a b x y : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a^x = 3) (h4 : b^y = 3) (h5 : a + b = 2 * Real.sqrt 3) :
  1/x + 1/y ≤ 1 :=
sorry

end max_value_fraction_l457_457239


namespace mabel_shark_percentage_l457_457962

theorem mabel_shark_percentage :
  (let day_one_fish := 15
   let day_two_fish := 3 * day_one_fish
   let total_fish := day_one_fish + day_two_fish
   let sharks := 15
   in (sharks.toFloat / total_fish.toFloat) * 100 = 25) :=
by
  sorry

end mabel_shark_percentage_l457_457962


namespace bottle_capacity_l457_457700

theorem bottle_capacity
  (num_boxes : ℕ)
  (bottles_per_box : ℕ)
  (fill_fraction : ℚ)
  (total_volume : ℚ)
  (total_bottles : ℕ)
  (filled_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_fraction = 3 / 4 →
  total_volume = 4500 →
  total_bottles = num_boxes * bottles_per_box →
  filled_volume = (total_bottles : ℚ) * (fill_fraction * (12 : ℚ)) →
  12 = 4500 / (total_bottles * fill_fraction) := 
by 
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end bottle_capacity_l457_457700


namespace B_pow_48_l457_457229

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 0],
  ![0, 0, 1],
  ![0, -1, 0]
]

theorem B_pow_48 :
  B^48 = ![
    ![0, 0, 0],
    ![0, 1, 0],
    ![0, 0, 1]
  ] := by sorry

end B_pow_48_l457_457229


namespace math_problem_l457_457075

theorem math_problem (a b c d x y : ℝ) (h1 : a = -b) (h2 : c * d = 1) 
  (h3 : (x + 3)^2 + |y - 2| = 0) : 2 * (a + b) - 2 * (c * d)^4 + (x + y)^2022 = -1 :=
by
  sorry

end math_problem_l457_457075


namespace binomial_expansion_coefficients_l457_457456

theorem binomial_expansion_coefficients (n : ℕ)
  (a1 a2 a3 : ℕ)
  (h1 : a1 = C_n^0)
  (h2 : a2 = (1 / 2) * C_n^1)
  (h3 : a3 = (1 / 4) * C_n^2)
  (h4 : 2 * a2 = a1 + a3)
  (h5 : n = 8) :
  -- 1. Sum of binomial coefficients
  (∑ k : ℕ in range (n + 1), binomial n k = 256) ∧
  -- 2. Terms with the largest coefficient
  (T_3 = 7 * x ^ (7/3) ∧
   T_4 = 7 * x ^ (2/3)) ∧
  -- 3. Rational terms
  (T_1 = x ^ 4 ∧
   T_6 = 7 / (16 * x)) := by
  sorry

end binomial_expansion_coefficients_l457_457456


namespace angle_between_scaled_and_rotated_vectors_l457_457104

variables (a b : ℝ × ℝ) (theta_alpha theta_beta : ℝ)

def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  real.arccos ((u.1 * v.1 + u.2 * v.2) / 
  (real.sqrt (u.1*u.1 + u.2*u.2) * real.sqrt (v.1*v.1 + v.2*v.2)))

-- Assuming the definition of vector rotation
def rotate_vector (v : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  (v.1 * real.cos angle - v.2 * real.sin angle,
   v.1 * real.sin angle + v.2 * real.cos angle)

variables (α : ℝ := 60) (β : ℝ := 45)  -- α is 60 degrees, β is 45 degrees

theorem angle_between_scaled_and_rotated_vectors (a b : ℝ × ℝ) 
  (h_angle : angle_between_vectors a b = α) :
  angle_between_vectors (2*a) (rotate_vector b (-β)) = α - β :=
by
  sorry

end angle_between_scaled_and_rotated_vectors_l457_457104


namespace area_of_orthographic_projection_l457_457070

theorem area_of_orthographic_projection (a : ℝ) :
  ∃ S : ℝ, S = (∇orth_proj a) := 
sorry

-- Definitions
def area_of_equilateral_triangle (a : ℝ) : ℝ := 
  (sqrt 3 / 4) * a ^ 2

def orthographic_projection_ratio : ℝ := 
  sqrt 2 / 4
  
def ∇orth_proj (a : ℝ) : ℝ := 
  orthographic_projection_ratio * area_of_equilateral_triangle a

end area_of_orthographic_projection_l457_457070


namespace multiply_and_simplify_l457_457970
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l457_457970


namespace perpendicular_dot_product_l457_457132

theorem perpendicular_dot_product (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → (λ * μ = -1) :=
by
  intros
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  sorry

end perpendicular_dot_product_l457_457132


namespace sheila_initial_savings_l457_457604

noncomputable def initial_savings (monthly_savings : ℕ) (years : ℕ) (family_addition : ℕ) (total_amount : ℕ) : ℕ :=
  total_amount - (monthly_savings * 12 * years + family_addition)

def sheila_initial_savings_proof : Prop :=
  initial_savings 276 4 7000 23248 = 3000

theorem sheila_initial_savings : sheila_initial_savings_proof :=
  by
    -- Proof goes here
    sorry

end sheila_initial_savings_l457_457604


namespace interesting_factor_exists_l457_457720

def is_interesting (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 0

theorem interesting_factor_exists (a b c d e : ℕ)
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
  (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e)
  (h8 : c ≠ d) (h9 : c ≠ e)
  (h10 : d ≠ e)
  (h_interesting_product : is_interesting (a * b * c * d * e)) :
  is_interesting a ∨ is_interesting b ∨ is_interesting c ∨ is_interesting d ∨ is_interesting e :=
begin
  sorry
end

end interesting_factor_exists_l457_457720


namespace valid_prime_triples_l457_457416

theorem valid_prime_triples:
  ∀ p q r : ℕ,
  prime p ∧ prime q ∧ prime r →
  (∃ k : ℕ, p^q + p^r = k^2) ↔ 
  (p = 2 ∧ q = 2 ∧ r = 5) ∨
  (p = 2 ∧ q = 5 ∧ r = 2) ∨
  (p = 3 ∧ q = 2 ∧ r = 3) ∨
  (p = 3 ∧ q = 3 ∧ r = 2) ∨
  (p = 2 ∧ q = r ∧ prime q ∧ q ≥ 3 ∧ q % 2 = 1) := 
by 
  sorry

end valid_prime_triples_l457_457416


namespace exp_13_pi_i_over_2_eq_i_l457_457014

theorem exp_13_pi_i_over_2_eq_i : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_13_pi_i_over_2_eq_i_l457_457014


namespace monotonic_increasing_interval_l457_457473

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin ((Real.pi / 4) * x)

theorem monotonic_increasing_interval :
  ∀ k : ℤ, monotone_on f (Icc (4 * k - 5 / 3 : ℝ) (4 * k + 1 / 3 : ℝ)) :=
by 
  simp [f]
  sorry

end monotonic_increasing_interval_l457_457473


namespace more_ones_than_twos_l457_457405

noncomputable def sum_digits (n : ℕ) : ℕ :=
  (to_digits 10 n).sum

noncomputable def to_single_digit (n : ℕ) : ℕ :=
  if n = 0 then 0 else
    (Nat.recOn n (λ _ _, n)
      (λ m f x h,
        if x < 10 then x
        else f (sum_digits x))) n rfl

theorem more_ones_than_twos (upper_bound : ℕ) (h : upper_bound = 1000000000) :
  let ones := List.filter (λ x, to_single_digit x = 1) (List.range upper_bound).length,
      twos := List.filter (λ x, to_single_digit x = 2) (List.range upper_bound).length
  in ones > twos :=
by { sorry }

end more_ones_than_twos_l457_457405


namespace binomial_expansion_properties_l457_457458

noncomputable def binomial_sum (n : ℕ) : ℕ := 2^n

noncomputable def largest_terms (n : ℕ) (x : ℝ) : List (ℝ × ℝ) := 
  if n = 8 then [(7, x^(7/3)), (7, x^(2/3))] else []

noncomputable def rational_terms (n : ℕ) (x : ℝ) : List (ℝ × ℝ) := 
  if n = 8 then [(1, x^4), (7/16, x^(-1))] else []

theorem binomial_expansion_properties (n : ℕ) (x : ℝ) 
  (h₁ : (sqrt x + 1 / (2 * (x^(1/3))))^n = ∑ k in finset.range (n + 1), 
    (nat.choose n k) * (sqrt x)^(n - k) * (1 / (2 * x^(1/3)))^k)
  (h₂ : 2 * ((1/2)*n) = 1 + (n*(n-1))/8) :
  binomial_sum n = 256 ∧ 
  largest_terms n x = [(7, x^(7/3)), (7, x^(2/3))] ∧ 
  rational_terms n x = [(1, x^4), (7/16, x^(-1))] :=
by sorry

end binomial_expansion_properties_l457_457458


namespace integral_of_piecewise_function_l457_457061

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ [0, π] then cos x else
  if x ∈ (π, 2 * π] then 1 else 0

theorem integral_of_piecewise_function :
  ∫ x in 0..(2 * π), f x = π :=
by
  sorry

end integral_of_piecewise_function_l457_457061


namespace area_bounded_arcsin_cos_eq_l457_457421

noncomputable def area_arcsin_cos_interval : ℝ := 
  let y := λ x : ℝ, Real.arcsin (Real.cos x) in
  let area := Real.integral (Set.Icc 0 (2 * Real.pi)) (λ x, y x) in
  area

theorem area_bounded_arcsin_cos_eq : 
  area_arcsin_cos_interval = (Real.pi ^ 2) / 4 :=
sorry

end area_bounded_arcsin_cos_eq_l457_457421


namespace range_of_a_l457_457500

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) → (a ≤ -7) :=
begin
  -- Define the function f
  let f := λ x : ℝ, x^2 + (a-1)*x + 2,

  sorry
end

end range_of_a_l457_457500


namespace sin_sum_right_triangle_l457_457186

noncomputable def triangle_sin_sum (a b : ℝ) (ha : a = 15) (hb : b = 8) (hc : Mathlib.sqrt(a^2 + b^2) = 17) : ℝ :=
  Mathlib.sin (Math.atan (a / b)) + Mathlib.sin (Math.atan (b / a))

theorem sin_sum_right_triangle : triangle_sin_sum 15 8 (by rfl) (by rfl) (by rw [← sq, by norm_num, ← sq, by norm_num, by norm_num]; exact rfl) = 23 / 17 := 
by
  sorry

end sin_sum_right_triangle_l457_457186


namespace multiply_expand_l457_457976

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l457_457976


namespace max_sum_multiplication_table_l457_457297

theorem max_sum_multiplication_table (a b c d e f : ℕ) (h : {a, b, c, d, e, f} = {1, 4, 6, 8, 9, 10}) :
  ∃ p q r s t u : ℕ, {p, q, r} ∪ {s, t, u} = {a, b, c, d, e, f} ∧ p + q + r = s + t + u ∧ 
  (p + q + r) * (s + t + u) = 361 := by
sorry

end max_sum_multiplication_table_l457_457297


namespace ball_distribution_l457_457152

theorem ball_distribution (balls boxes : ℕ) (hballs : balls = 7) (hboxes : boxes = 4) :
  (∃ (ways : ℕ), ways = (Nat.choose (balls - 1) (boxes - 1)) ∧ ways = 20) :=
by
  sorry

end ball_distribution_l457_457152


namespace area_of_triangle_QNF_l457_457517

theorem area_of_triangle_QNF :
  ∀ (P Q R E F N: Type) [linear_ordered_field P] [linear_ordered_add_comm_group Q]
  [linear_ordered_at_field R] [linear_ordered_case E] [linear_ordered_ab_group F] [linear_ordered_noncomm_group N]
  (side : P) (midpoint : Q) (extension : R) (triangle: P -> Q -> Q -> E)
  (equilateral: side length ≥ 4) 
  (mid_N : N = midpoint (PR) (Q = midpoint (EF)))
  (on_extension : E lies_on extension)
  (PE_EQ_2 : length of (PE) = length (EQ) = 2),
  area (triangle(∠QNF)) = √3 := sorry.

end area_of_triangle_QNF_l457_457517


namespace point_in_fourth_quadrant_l457_457528

def point : ℝ × ℝ := (3, -4)

def isFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def isSecondQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def isThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def isFourthQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : isFourthQuadrant point :=
by
  sorry

end point_in_fourth_quadrant_l457_457528


namespace area_inequalities_l457_457834

noncomputable def f1 (x : ℝ) : ℝ := 1 - (1 / 2) * x
noncomputable def f2 (x : ℝ) : ℝ := 1 / (x + 1)
noncomputable def f3 (x : ℝ) : ℝ := 1 - (1 / 2) * x^2

noncomputable def S1 : ℝ := 1 - (1 / 4)
noncomputable def S2 : ℝ := Real.log 2
noncomputable def S3 : ℝ := (5 / 6)

theorem area_inequalities : S2 < S1 ∧ S1 < S3 := by
  sorry

end area_inequalities_l457_457834


namespace radius_of_inscribed_circle_l457_457788

theorem radius_of_inscribed_circle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = a + b - c :=
sorry

end radius_of_inscribed_circle_l457_457788


namespace remainder_444_222_div_13_l457_457324

-- Define conditions
def cond1 : Prop := 444 % 13 = 3
def cond2 (n : ℕ) : Prop := n % 12 = 0 -> 3^n % 13 = 1
def cond3 (a b : ℕ) : Prop := a^b % 13 = (a % 13)^b % 13

-- Define the main statement to prove
theorem remainder_444_222_div_13 (h1 : cond1) (h2 : cond2 12) (h3 : cond3 444 222) : 444^222 % 13 = 1 :=
by
  -- Ensure the statement matches given conditions
  simp [cond1, cond2, cond3] at h1 h2 h3
  -- Apply the necessary properties and congruences to prove the main statement
  sorry

end remainder_444_222_div_13_l457_457324


namespace q_alone_time_24_days_l457_457956

theorem q_alone_time_24_days:
  ∃ (Wq : ℝ), (∀ (Wp Ws : ℝ), 
    Wp = Wq + 1 / 60 → 
    Wp + Wq = 1 / 10 → 
    Wp + 1 / 60 + 2 * Wq = 1 / 6 → 
    1 / Wq = 24) :=
by
  sorry

end q_alone_time_24_days_l457_457956


namespace probability_triangle_l457_457055

def regular_decagon : Type := { vertices : Finset ℤ // vertices.card = 10 ∧ ∀ v ∈ vertices, ∃ w x, (w ≠ v ∧ x ≠ v ∧ x ≠ w ∧ w ∈ vertices ∧ x ∈ vertices ∧ (w - v) = 1 ∧ (x - v) = 2) }

def sequentially_adjacent (v1 v2 v3 : ℤ) : Prop := (v2 - v1 = 1) ∧ (v3 - v2 = 1)

theorem probability_triangle (D : regular_decagon) :
  let total_triangles := (finset.choose 3 D.1.card),
      favorable_triangles := 10 in
  (favorable_triangles : ℚ) / total_triangles = 1 / 12 :=
begin
  sorry
end

end probability_triangle_l457_457055


namespace problem1_problem2_problem3_l457_457189

-- 1. Problem 1: Prove n = 5 or n = 12 given the conditions in the first problem
theorem problem1 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 4 = 10)
  (h₂ : a 10 = -2)
  (h₃ : S n = 60) : n = 5 ∨ n = 12 := 
sorry

-- 2. Problem 2: Prove S₁₇ = 153 given the conditions in the second problem
theorem problem2 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : a 1 = -7)
  (h₂ : ∀ n, a (n + 1) = a n + 2) : S 17 = 153 := 
sorry

-- 3. Problem 3: Prove S₁₃ = 104 given the conditions in the third problem
theorem problem3 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : a 2 + a 7 + a 12 = 24) : S 13 = 104 :=
sorry

end problem1_problem2_problem3_l457_457189


namespace intersect_on_altitude_l457_457587

variable {A B C C1 D1 A2 D2 : Type}
          [AffineSpace A B] [AffineSpace B C] [AffineSpace C1 D1]
          [AffineSpace A2 D2]
          [Triangle A B C] [Square ABC1D1] [Square A2BCD2]

theorem intersect_on_altitude (ABC : Triangle A B C) 
    (s1 : Square ABC1D1) (s2 : Square A2BCD2) 
    {D1_intersect: ∃ X, Line A D2 ∧ Line C D1 ∧ Line X} : 
    ∃ X : Point,
    (Intersect (Line A D2) (Line C D1) X) → X ∈ Altitude B :=
sorry

end intersect_on_altitude_l457_457587


namespace exist_indices_l457_457926

theorem exist_indices (A : Matrix (Fin 10) (Fin 10) ℝ) (h1 : ∀ i, (∑ j, A i j) = 1) (h2 : ∀ j, (∑ i, A i j) = 1) 
  (h3 : ∀ i j, 0 < A i j) : ∃ j k l m, j < k ∧ l < m ∧ A j l * A k m + A j m * A k l ≥ 1 / 50 := 
sorry

end exist_indices_l457_457926


namespace max_ab_l457_457074

theorem max_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ab ≤ 1 / 4 :=
sorry

end max_ab_l457_457074


namespace postage_stamp_problem_l457_457706

theorem postage_stamp_problem
  (x y z : ℕ) (h1: y = 10 * x) (h2: x + 2 * y + 5 * z = 100) :
  x = 5 ∧ y = 50 ∧ z = 0 :=
by
  sorry

end postage_stamp_problem_l457_457706


namespace distinct_ordered_pairs_l457_457153

theorem distinct_ordered_pairs :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ (1 / (p.1 : ℝ) + 1 / (p.2 : ℝ) = 1 / 6)}.toFinset.card = 9 :=
begin
  sorry
end

end distinct_ordered_pairs_l457_457153


namespace train_passes_jogger_in_15_seconds_l457_457362

-- Jogger's speed in km/hr
def jogger_speed_km_hr : ℝ := 12

-- Train's speed in km/hr
def train_speed_km_hr : ℝ := 60

-- Convert speeds from km/hr to m/s
def convert_speed (speed_km_hr : ℝ) : ℝ := speed_km_hr * (5/18)

-- Jogger's speed in m/s
def jogger_speed_m_s : ℝ := convert_speed jogger_speed_km_hr

-- Train's speed in m/s
def train_speed_m_s : ℝ := convert_speed train_speed_km_hr

-- Relative speed of the train with respect to the jogger in m/s
def relative_speed_m_s : ℝ := train_speed_m_s - jogger_speed_m_s

-- Total distance the train needs to cover to pass the jogger in meters
def total_distance_m : ℝ := 300 + 300

-- Time in seconds for the train to pass the jogger
def time_to_pass_jogger : ℝ := total_distance_m / relative_speed_m_s

theorem train_passes_jogger_in_15_seconds : time_to_pass_jogger = 15 := by
  sorry

end train_passes_jogger_in_15_seconds_l457_457362


namespace people_distribution_l457_457190

theorem people_distribution (x : ℕ) (h1 : x > 5):
  100 / (x - 5) = 150 / x :=
sorry

end people_distribution_l457_457190


namespace cricket_team_age_difference_l457_457618

theorem cricket_team_age_difference :
  ∀ (captain_age : ℕ) (keeper_age : ℕ) (team_size : ℕ) (team_average_age : ℕ) (remaining_size : ℕ),
  captain_age = 28 →
  keeper_age = captain_age + 3 →
  team_size = 11 →
  team_average_age = 25 →
  remaining_size = team_size - 2 →
  (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 24 →
  team_average_age - (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 1 :=
by
  intros captain_age keeper_age team_size team_average_age remaining_size h1 h2 h3 h4 h5 h6
  sorry

end cricket_team_age_difference_l457_457618


namespace arithmetic_sequence_a7_l457_457515

theorem arithmetic_sequence_a7 (S_13 : ℕ → ℕ → ℕ) (n : ℕ) (a7 : ℕ) (h1: S_13 13 52 = 52) (h2: S_13 13 a7 = 13 * a7):
  a7 = 4 :=
by
  sorry

end arithmetic_sequence_a7_l457_457515


namespace diana_can_paint_statues_l457_457403

theorem diana_can_paint_statues :
  ∀ (paint_remaining paint_per_statue : ℚ), paint_remaining = 7 / 8 → paint_per_statue = 1 / 8 → 
  paint_remaining / paint_per_statue = 7 :=
by
  intros paint_remaining paint_per_statue h_remaining h_per_statue
  rw [h_remaining, h_per_statue]
  norm_num
  sorry

end diana_can_paint_statues_l457_457403


namespace find_smallest_palindrome_l457_457046

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_aba_form (n : ℕ) : Prop :=
  let s := n.digits 10
  s.length = 3 ∧ s.head = s.get! 2

def smallest_aba_not_palindromic_when_multiplied_by_103 : ℕ :=
  Nat.find (λ n, is_three_digit n ∧ is_aba_form n ∧ ¬is_palindrome (103 * n))

theorem find_smallest_palindrome : smallest_aba_not_palindromic_when_multiplied_by_103 = 131 := sorry

end find_smallest_palindrome_l457_457046


namespace percentage_of_profits_l457_457179

variable (R P : ℝ) -- Let R be the revenues and P be the profits in the previous year
variable (H1 : (P/R) * 100 = 10) -- The condition we want to prove
variable (H2 : 0.95 * R) -- Revenues in 2009 are 0.95R
variable (H3 : 0.1 * 0.95 * R) -- Profits in 2009 are 0.1 * 0.95R = 0.095R
variable (H4 : 0.095 * R = 0.95 * P) -- The given relation between profits in 2009 and previous year

theorem percentage_of_profits (H1 : (P/R) * 100 = 10) 
  (H2 : ∀ (R : ℝ),  ∃ ρ, ρ = 0.95 * R)
  (H3 : ∀ (R : ℝ),  ∃ π, π = 0.10 * (0.95 * R))
  (H4 : ∀ (R P : ℝ), 0.095 * R = 0.95 * P) :
  ∀ (P R : ℝ), (P/R) * 100 = 10 := 
by
  sorry

end percentage_of_profits_l457_457179


namespace multiply_polynomials_l457_457980

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l457_457980


namespace coeff_x3_in_expansion_l457_457620

theorem coeff_x3_in_expansion : (polynomial.coeff ((1 - X) * (2 * X + 1) ^ 4) 3) = 8 :=
sorry

end coeff_x3_in_expansion_l457_457620


namespace least_distance_between_ticks_l457_457998

theorem least_distance_between_ticks (x : ℚ) : 
  (∀ n, n ∈ {1/5, 2/5, 3/5, 4/5, 1} → n = 1/5 ∨ n = 2/5 ∨ n = 3/5 ∨ n = 4/5 ∨ n = 1) ∧ 
  (∀ m, m ∈ {1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1} → m = 1/7 ∨ m = 2/7 ∨ m = 3/7 ∨ m = 4/7 ∨ m = 5/7 ∨ m = 6/7 ∨ m = 1) →
  x = 1/35 :=
begin
  sorry
end

end least_distance_between_ticks_l457_457998


namespace rows_before_change_l457_457364

-- Definitions and conditions
variables {r c : ℕ}

-- The total number of tiles before and after the change
def total_tiles_before (r c : ℕ) := r * c = 30
def total_tiles_after (r c : ℕ) := (r + 4) * (c - 2) = 30

-- Prove that the number of rows before the change is 3
theorem rows_before_change (h1 : total_tiles_before r c) (h2 : total_tiles_after r c) : r = 3 := 
sorry

end rows_before_change_l457_457364


namespace integer_solution_of_inequality_l457_457611

theorem integer_solution_of_inequality (x : ℤ) : 
  (0 < ((x - 1)^2) / (x + 1).toRat) ∧ (((x - 1)^2).toRat / (x + 1).toRat < 1) → 
  x = 2 := 
by 
  sorry

end integer_solution_of_inequality_l457_457611


namespace distribution_and_variance_of_X_optimal_strategy_changed_selection_l457_457883

-- Part 1: Distribution and variance of X
theorem distribution_and_variance_of_X :
  ∀ (X : ℕ),
  (X ∈ {0, 1, 2, 3} →
   let P_X := {0 ↦ (8 / 27 : ℚ), 1 ↦ (4 / 9 : ℚ), 2 ↦ (2 / 9 : ℚ), 3 ↦ (1 / 27 : ℚ)} in
     P_X X = match X with 
     | 0 => 8 / 27 
     | 1 => 4 / 9
     | 2 => 2 / 9
     | 3 => 1 / 27 
     | _ => 0) ∧
  (P_X.hasVariance (2 / 3)) := sorry

-- Part 2: Optimal strategy with changed rules
theorem optimal_strategy_changed_selection :
  let probability_win_after_change := 2 / 3 in
  let probability_lose_after_change := 1 / 3 in
  let expected_value_after_change := 400 * probability_win_after_change + 0 * probability_lose_after_change in
  let probability_win_initial := 1 / 3 in
  let probability_consolation_initial := 2 / 3 in
  let expected_value_initial := 200 * probability_win_initial + 50 * probability_consolation_initial in
  expected_value_after_change > expected_value_initial :=
sorry

end distribution_and_variance_of_X_optimal_strategy_changed_selection_l457_457883


namespace find_value_of_expression_l457_457759

noncomputable def f : ℝ → ℝ := sorry

theorem find_value_of_expression :
  (∀ x, f (-x) = -f (x)) →                              -- f is odd
  (∀ x1 x2 : ℝ, (1 ≤ x1 → x1 ≤ 4) → (1 ≤ x2 → x2 ≤ 4) → x1 < x2 → f(x1) < f(x2)) →  -- f is increasing on [1, 4]
  (∀ x : ℝ, (2 ≤ x → x ≤ 3) → (-1 ≤ f(x)) → (f(x) ≤ 8)) →  -- f has minimum -1 and maximum 8 on [2, 3]
  (f(2) = -1) →
  (f(3) = 8) → 
  2 * f(2) + f (-3) + f (0) = -10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_value_of_expression_l457_457759


namespace tan_A_eq_2_area_triangle_S_l457_457090

variables {A B C: ℝ} {c: ℝ} {S: ℝ}

-- Definition of the area of triangle ABC
def area_triangle (A B C: ℝ) : ℝ := S

-- Given conditions
axiom dot_product_AB_AC_eq_S : ∀ (A B C: ℝ),  
  (S : ℝ) = (∥A∥ * ∥C∥ * real.cos A)

axiom angle_B_is_pi_div_4 : B = real.pi / 4
axiom side_c_is_3 : c = 3

-- Proof statement parts
theorem tan_A_eq_2 : ∀ {A B C S : ℝ}, 
  ((∥A∥ * ∥C∥ * real.cos A) = S) → real.tan A = 2 := 
sorry

theorem area_triangle_S : ∀ {A B C: ℝ}, 
(B = real.pi / 4) → (c = 3) → (real.tan A = 2) → (S = 3) := 
sorry

end tan_A_eq_2_area_triangle_S_l457_457090


namespace prime_solution_l457_457415

theorem prime_solution (p : ℕ) (x y : ℕ) (hp : Prime p) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 → p = 2 ∨ p = 3 :=
by
  sorry

end prime_solution_l457_457415


namespace complex_number_coordinates_l457_457191

theorem complex_number_coordinates :
    (let z : ℂ := (2 * complex.I) / (1 - complex.I) 
    in (z.re, z.im)) = (-1, 1) := by 
  sorry

end complex_number_coordinates_l457_457191


namespace fraction_inequality_solution_set_l457_457646

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 1) / x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
begin
  sorry
end

end fraction_inequality_solution_set_l457_457646


namespace even_and_increasing_functions_l457_457761

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

noncomputable def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem even_and_increasing_functions :
  ∀ f ∈ ({ λ x : ℝ, Real.log (abs x), λ x : ℝ, x^(-2), λ x : ℝ, x + Real.sin x, λ x : ℝ, Real.cos (-x) } : set (ℝ → ℝ)),
    is_even f ∧ is_increasing f 0 1 → f = λ x, Real.log (abs x) :=
begin
  sorry
end

end even_and_increasing_functions_l457_457761


namespace max_tan_alpha_l457_457060

theorem max_tan_alpha (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h : tan (α + β) = 9 * tan β) : 
    ∃ M : ℝ, (∀ x : ℝ, x = tan α → x ≤ M) ∧ M = 4 / 3 :=
by
  sorry

end max_tan_alpha_l457_457060


namespace infinite_seq_contains_all_nat_l457_457375

-- Define the sequence a
def sequence_a (a : ℕ → ℕ) : Prop :=
  (∀ i, 0 ≤ a i ∧ a i ≤ i) ∧
  (∀ k, (∑ i in Finset.range (k + 1), Nat.choose k (a i)) = 2^k)

-- Lean statement of the problem
theorem infinite_seq_contains_all_nat (a : ℕ → ℕ) (H : sequence_a a) :
  ∀ (N : ℕ), ∃ (i : ℕ), a i = N :=
by
  sorry

end infinite_seq_contains_all_nat_l457_457375


namespace problem_statement_l457_457811

-- Define the sequence S
def S (n : ℕ) : ℤ := n^2 - 4 * n + 2

-- Define the sequence a
def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

-- Define the sum of absolute values of the first 10 terms of a
def sum_abs_a (n : ℕ) : ℤ :=
  finset.sum (finset.range n) (λ i, abs (a (i + 1)))

theorem problem_statement : sum_abs_a 10 = 66 :=
by
  sorry

end problem_statement_l457_457811


namespace perpendicular_vectors_condition_l457_457120

theorem perpendicular_vectors_condition (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
      b : ℝ × ℝ := (1, -1)
  in ((a.1 + λ * b.1) * (a.1 + μ * b.1) + (a.2 + λ * b.2) * (a.2 + μ * b.2) = 0) → (λ * μ = -1) :=
sorry

end perpendicular_vectors_condition_l457_457120


namespace alfred_scooter_sale_l457_457373

def alfred_selling_price (purchase_price repair_cost : ℝ) (gain_percent : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let selling_price := total_cost * (1 + gain_percent / 100)
  selling_price

theorem alfred_scooter_sale :
  alfred_selling_price 4700 800 9.090909090909092 = 6000 :=
by
  sorry

end alfred_scooter_sale_l457_457373


namespace train_pass_pole_time_correct_l457_457731

variables (L : ℝ) (speed : ℝ) (time_pass_stationary_train : ℝ) (length_stationary_train : ℝ)

noncomputable def time_to_pass_pole (L speed : ℝ) : ℝ := 
  L / speed

theorem train_pass_pole_time_correct :
  let speed := 64.8
  let time_pass_stationary_train := 25
  let length_stationary_train := 360
  let L := 64.8 * 25 - 360
  time_to_pass_pole L speed ≈ 19.44 :=
by
  sorry

end train_pass_pole_time_correct_l457_457731


namespace sum_tens_units_digits_of_9_pow_2050_l457_457326

theorem sum_tens_units_digits_of_9_pow_2050 :
  let n := 9^2050 in
  (n % 10 + (n / 10) % 10) = 1 :=
by
  sorry

end sum_tens_units_digits_of_9_pow_2050_l457_457326


namespace exp_13_pi_i_over_2_eq_i_l457_457015

theorem exp_13_pi_i_over_2_eq_i : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_13_pi_i_over_2_eq_i_l457_457015


namespace find_f_2017m_l457_457715

noncomputable def f (x m : ℝ) : ℝ :=
  if h : x = 0 then 0 else
  if x < 0 then 2 ^ x + 2 * m
  else log 2 x - m

noncomputable def period : ℝ := 4

theorem find_f_2017m (m : ℝ) (hm : m = 1 / 4) :
  f (2017 * m) m = -9 / 4 :=
by
  simp [f, hm]
  sorry

end find_f_2017m_l457_457715


namespace flyers_total_l457_457905

theorem flyers_total (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) 
  (hj : jack_flyers = 120) (hr : rose_flyers = 320) (hl : left_flyers = 796) :
  jack_flyers + rose_flyers + left_flyers = 1236 :=
by {
  sorry
}

end flyers_total_l457_457905


namespace expression_value_l457_457954

noncomputable def x := 7!
noncomputable def y := 4!
noncomputable def z := 3!
noncomputable def w := 5!

theorem expression_value : (x - (y * z) ^ 2) / w = -130.8 :=
by
  sorry

end expression_value_l457_457954


namespace roots_sum_of_squares_l457_457953

theorem roots_sum_of_squares {r s : ℝ} (h : Polynomial.roots (X^2 - 3*X + 1) = {r, s}) : r^2 + s^2 = 7 :=
by
  sorry

end roots_sum_of_squares_l457_457953


namespace triangle_ABC_BC_greater_half_AB_l457_457336

open_locale real

theorem triangle_ABC_BC_greater_half_AB
  {A B C : ℝ}
  (hAgtB : A > B)
  (h_triangle : A + B + C = π) : 
  BC > (1 / 2) * AB :=
sorry

end triangle_ABC_BC_greater_half_AB_l457_457336


namespace min_value_of_x_plus_2y_l457_457080

theorem min_value_of_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 1 / x + 1 / y = 2) : 
  x + 2 * y ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_x_plus_2y_l457_457080


namespace sum_of_first_six_terms_l457_457438

section 
variable {a_n : ℕ → ℝ}
variable S_n : ℕ → ℝ

-- Sequence conditions
axiom h1 : ∀ n, (1 / (a_n n + 1)) = (3 / (a_n (n + 1) + 1))
axiom h2 : a_n 2 = 5

-- Proof goal: show that the sum of the first six terms equals 722
theorem sum_of_first_six_terms : S_n 6 = 722 :=
sorry
end

end sum_of_first_six_terms_l457_457438


namespace largest_n_for_divisibility_l457_457685

theorem largest_n_for_divisibility :
  ∃ (n : ℕ), n = 5 ∧ 3^n ∣ (4^27000 - 82) ∧ ¬ 3^(n + 1) ∣ (4^27000 - 82) :=
by
  sorry

end largest_n_for_divisibility_l457_457685


namespace lorelei_vase_rose_count_l457_457202

theorem lorelei_vase_rose_count :
  let r := 12
  let p := 18
  let y := 20
  let o := 8
  let picked_r := 0.5 * r
  let picked_p := 0.5 * p
  let picked_y := 0.25 * y
  let picked_o := 0.25 * o
  picked_r + picked_p + picked_y + picked_o = 22 := 
by
  sorry

end lorelei_vase_rose_count_l457_457202


namespace veronica_cherry_pie_l457_457662

theorem veronica_cherry_pie : 
  let cherries_per_pound := 80 in
  let cherries_pitted_per_minute := 20 / 10 in
  let time_pitting_hours := 2 in
  let time_pitting_minutes := time_pitting_hours * 60 in
  (cherries_pitted_per_minute * time_pitting_minutes) / cherries_per_pound = 3 :=
by
  sorry

end veronica_cherry_pie_l457_457662


namespace range_of_a_l457_457188

-- Definitions based on conditions
def M := (-2 : ℝ, 0 : ℝ)
def N := (0 : ℝ, 2 : ℝ)
def A := (-1 : ℝ, 1 : ℝ)

def circle (a : ℝ) : (ℝ × ℝ) → Prop :=
  λ p, (p.1 - a)^2 + p.2^2 = 2

-- Main theorem statement
theorem range_of_a (a : ℝ) (ha : 0 < a) :
  (∀ P : ℝ × ℝ, circle a P → ∠ MP N ≤ π / 2) → (a > real.sqrt 7 - 1) :=
by
  sorry

end range_of_a_l457_457188


namespace sum_a1_to_a5_l457_457094

-- Define the conditions
def equation_holds (x a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  x^5 + 2 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5

-- State the theorem
theorem sum_a1_to_a5 (a0 a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, equation_holds x a0 a1 a2 a3 a4 a5) :
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  sorry

end sum_a1_to_a5_l457_457094


namespace multiplication_identity_multiplication_l457_457994

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l457_457994


namespace equality_equiv_l457_457599

-- Problem statement
theorem equality_equiv (a b c : ℝ) :
  (a + b + c ≠ 0 → ( (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0)) ∧
  (a + b + c = 0 → ∀ w x y z: ℝ, w * x + y * z = 0) :=
by
  sorry

end equality_equiv_l457_457599


namespace heart_to_heart_line_k_heart_to_heart_line_b_div_m_constant_heart_to_heart_triangle_area_l457_457955

/- Part 1 -/
theorem heart_to_heart_line_k (k : ℝ) :
  (∀ x y : ℝ, y = x^2 - 2 * x + 1 → y = k * x + 1 → k = -1) :=
sorry

/- Part 2 -/
theorem heart_to_heart_line_b_div_m_constant (b m : ℝ) (h : ∀ x : ℝ, (y = - x ^ 2 + b * x) → (y = m * x)) : 
  b / m = 2 :=
sorry

/- Part 3 -/
theorem heart_to_heart_triangle_area (a k : ℝ) (S : Set ℝ) 
  (h₁ : 1/2 ≤ k ∧ k ≤ 2) 
  (h₂ : S = {s : ℝ | ∃ p : ℝ, -⅟₂ ≤ k ∧ k ≤ 2 ∧ s = k^2 / (3*k^2 - 2*k + 1)}) : 
  ( ∀ s ∈ S, 1/3 ≤ s ∧ s ≤ 1/2 ) :=
sorry


end heart_to_heart_line_k_heart_to_heart_line_b_div_m_constant_heart_to_heart_triangle_area_l457_457955


namespace find_t_l457_457092

variables {t : ℝ}
def a : ℝ × ℝ × ℝ := (1, 1, 0)
def b : ℝ × ℝ × ℝ := (-1, t, 2)

theorem find_t (h : (1, 1, 0) • (−1, t, 2) = 0) : t = 1 := sorry

end find_t_l457_457092


namespace factorization_l457_457057

theorem factorization (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by 
  sorry

end factorization_l457_457057


namespace cat_catches_total_birds_l457_457352

theorem cat_catches_total_birds :
  let morning_birds := 15
  let morning_success_rate := 0.60
  let afternoon_birds := 25
  let afternoon_success_rate := 0.80
  let night_birds := 20
  let night_success_rate := 0.90
  
  let morning_caught := morning_birds * morning_success_rate
  let afternoon_initial_caught := 2 * morning_caught
  let afternoon_caught := min (afternoon_birds * afternoon_success_rate) afternoon_initial_caught
  let night_caught := night_birds * night_success_rate

  let total_caught := morning_caught + afternoon_caught + night_caught
  total_caught = 47 := 
by
  sorry

end cat_catches_total_birds_l457_457352


namespace cos_18_deg_l457_457007

theorem cos_18_deg :
  let x := real.cos (real.pi / 10) in
  let y := real.cos (2 * real.pi / 10) in
  (y = 2 * x^2 - 1) ∧
  (4 * x^3 - 3 * x = real.sin (real.pi * 2 / 5)) →
  (real.cos (real.pi / 10) = real.sqrt ((5 + real.sqrt 5) / 8)) :=
by
  sorry

end cos_18_deg_l457_457007


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l457_457277

theorem solve_eq1 (x : ℝ) : (3 * x + 2) ^ 2 = 25 ↔ (x = 1 ∨ x = -7 / 3) := by
  sorry

theorem solve_eq2 (x : ℝ) : 3 * x ^ 2 - 1 = 4 * x ↔ (x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) := by
  sorry

theorem solve_eq3 (x : ℝ) : (2 * x - 1) ^ 2 = 3 * (2 * x + 1) ↔ (x = -1 / 2 ∨ x = 1) := by
  sorry

theorem solve_eq4 (x : ℝ) : x ^ 2 - 7 * x + 10 = 0 ↔ (x = 5 ∨ x = 2) := by
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l457_457277


namespace greatest_number_of_factors_l457_457423

theorem greatest_number_of_factors (b m : ℕ) (hb : 1 ≤ b ∧ b ≤ 20 ∧ ∃ d, d ∣ b ∧ 1 < d < b) (hm : 1 ≤ m ∧ m ≤ 20) : 
  ∃ b m, 1 ≤ b ∧ b ≤ 20 ∧ (∃ d, d ∣ b ∧ 1 < d < b) ∧ 1 ≤ m ∧ m ≤ 20 ∧ (∀ n, (∃ k, n^k = b^m) → n ≤ 81) :=
sorry

end greatest_number_of_factors_l457_457423


namespace systematic_sampling_l457_457065

/-- Conditions for systematic sampling: There are 60 students numbered from 1 to 60,
    and 6 people are to be selected with equal intervals and evenly distributed throughout the population. -/
theorem systematic_sampling (students : List ℕ) (selected : List ℕ) :
  students = List.range (60) ∧ selected = [3, 13, 23, 33, 43, 53] ∧
  (∀ i j, (i < j ∧ j < selected.length) → selected.get? j - selected.get? i = some 10) :=
sorry

end systematic_sampling_l457_457065


namespace sweater_exceeds_shirt_by_4_l457_457181

-- Define the given conditions
def total_price_shirts : ℝ := 400
def number_of_shirts : ℕ := 25
def total_price_sweaters : ℝ := 1500
def number_of_sweaters : ℕ := 75

-- Calculate the average prices
def avg_price_shirt : ℝ := total_price_shirts / number_of_shirts
def avg_price_sweater : ℝ := total_price_sweaters / number_of_sweaters

-- The theorem to be proved
theorem sweater_exceeds_shirt_by_4 :
  avg_price_sweater - avg_price_shirt = 4 := by
  sorry

end sweater_exceeds_shirt_by_4_l457_457181


namespace largest_square_side_length_l457_457810

theorem largest_square_side_length (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) : 
  ∃ x : ℝ, x = (a * b) / (a + b) := 
sorry

end largest_square_side_length_l457_457810


namespace obtuse_triangle_k_values_l457_457398

-- Definitions and Conditions
def non_degenerate_triangle (a b k : ℕ) : Prop :=
  a + b > k ∧ a + k > b ∧ b + k > a

def obtuse_triangle (a b k : ℕ) : Prop :=
  k*k > a*a + b*b ∨ a*a > b*b + k*k ∨ b*b > a*a + k*k

-- The proof problem statement
theorem obtuse_triangle_k_values :
  let a := 8 in
  let b := 17 in
  let ks := {k : ℕ | non_degenerate_triangle a b k ∧ obtuse_triangle a b k}.to_finset in
  ks.card = 11 :=
by sorry

end obtuse_triangle_k_values_l457_457398


namespace total_amount_given_away_l457_457217

variable (numGrandchildren : ℕ)
variable (cardsPerGrandchild : ℕ)
variable (amountPerCard : ℕ)

theorem total_amount_given_away (h1 : numGrandchildren = 3) (h2 : cardsPerGrandchild = 2) (h3 : amountPerCard = 80) : 
  numGrandchildren * cardsPerGrandchild * amountPerCard = 480 := by
  sorry

end total_amount_given_away_l457_457217


namespace compare_surface_areas_l457_457436

noncomputable def S1 (a : ℝ) := 6 * a^2
noncomputable def S2 (r : ℝ) := 6 * real.pi * r^2
noncomputable def S3 (R : ℝ) := 4 * real.pi * R^2

theorem compare_surface_areas : 
  let a := real.cbrt (2 * real.pi)
  let r := 1
  let R := real.cbrt (3 / 2)
  S1 a > S2 r ∧ S2 r > S3 R := 
by {
  -- substitution of the values
  sorry
}

end compare_surface_areas_l457_457436


namespace isosceles_triangle_relationship_l457_457376

theorem isosceles_triangle_relationship (x y : ℝ) (h1 : 2 * x + y = 30) (h2 : 7.5 < x) (h3 : x < 15) : 
  y = 30 - 2 * x :=
  by sorry

end isosceles_triangle_relationship_l457_457376


namespace area_codes_count_l457_457370

theorem area_codes_count :
  let digits := {6, 4, 3, 5}
  let can_repeat := ((6, 2), (4, 1), (3, ∞), (5, 1))
  (∀ x ∈ digits, x = 2 → False) ∧ 
  (∀ d ∈ digits, 1 <= count_occurrences(d)) ∧ 
  (∃ x ∈ digits, x % 2 = 0) ∧ (∃ y ∈ digits, y % 2 = 1) ∧ 
  (∀ a:ℕ, a ∈ digits → ∃ m:ℕ, m = multiple_factor(a)) ∧ (3 ∣ m ∧ 4 ∣ m): 
  (number_of_codes = 0) := 
by 
  sorry

end area_codes_count_l457_457370


namespace wendy_first_album_pictures_l457_457319

theorem wendy_first_album_pictures 
  (total_pictures : ℕ)
  (num_albums : ℕ)
  (pics_per_album : ℕ)
  (pics_in_first_album : ℕ)
  (h1 : total_pictures = 79)
  (h2 : num_albums = 5)
  (h3 : pics_per_album = 7)
  (h4 : total_pictures = pics_in_first_album + num_albums * pics_per_album) : 
  pics_in_first_album = 44 :=
by
  sorry

end wendy_first_album_pictures_l457_457319


namespace lambda_mu_relationship_l457_457140

theorem lambda_mu_relationship (a b : ℝ × ℝ) (λ μ : ℝ)
  (h1 : a = (1, 1))
  (h2 : b = (1, -1))
  (h3 : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457140


namespace shortest_travel_time_to_sunny_town_l457_457717

-- Definitions based on the given conditions
def highway_length : ℕ := 12

def railway_crossing_closed (t : ℕ) : Prop :=
  ∃ k : ℕ, t = 6 * k + 0 ∨ t = 6 * k + 1 ∨ t = 6 * k + 2

def traffic_light_red (t : ℕ) : Prop :=
  ∃ k1 : ℕ, t = 5 * k1 + 0 ∨ t = 5 * k1 + 1

def initial_conditions (t : ℕ) : Prop :=
  railway_crossing_closed 0 ∧ traffic_light_red 0

def shortest_time_to_sunny_town (time : ℕ) : Prop := 
  time = 24

-- The proof statement
theorem shortest_travel_time_to_sunny_town :
  ∃ time : ℕ, shortest_time_to_sunny_town time ∧
  (∀ t : ℕ, 0 ≤ t → t ≤ time → ¬railway_crossing_closed t ∧ ¬traffic_light_red t) :=
sorry

end shortest_travel_time_to_sunny_town_l457_457717


namespace intersection_complement_A_B_l457_457850

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def complement (S : Set ℝ) : Set ℝ := {x | x ∉ S}

theorem intersection_complement_A_B :
  U = Set.univ →
  A = {x | -1 < x ∧ x < 1} →
  B = {y | 0 < y} →
  (A ∩ complement B) = {x | -1 < x ∧ x ≤ 0} :=
by
  intros hU hA hB
  sorry

end intersection_complement_A_B_l457_457850


namespace max_value_of_E_l457_457298

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  ∀ (a b c d : ℝ),
    (-8.5 ≤ a ∧ a ≤ 8.5) →
    (-8.5 ≤ b ∧ b ≤ 8.5) →
    (-8.5 ≤ c ∧ c ≤ 8.5) →
    (-8.5 ≤ d ∧ d ≤ 8.5) →
    E a b c d ≤ 306 := sorry

end max_value_of_E_l457_457298


namespace buffaloes_count_l457_457882

variables (B D : ℕ)

-- Conditions
def heads : ℕ := B + D
def legs : ℕ := 4 * B + 2 * D
def condition : Prop := legs = 2 * heads + 24

-- Theorem
theorem buffaloes_count (h : condition B D) : B = 12 :=
by
  sorry

end buffaloes_count_l457_457882


namespace magnitude_of_perpendicular_vector_l457_457486

-- Define the vectors and the condition that they are perpendicular
def a (n : ℝ) : ℝ × ℝ := (1, n)
def b (n : ℝ) : ℝ × ℝ := (-1, n)

-- Define the dot product condition
def is_perpendicular (n : ℝ) : Prop := (a n).1 * (b n).1 + (a n).2 * (b n).2 = 0

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The theorem to be proven
theorem magnitude_of_perpendicular_vector (n : ℝ) (h : is_perpendicular n) : magnitude (a n) = Real.sqrt 2 :=
by
  sorry

end magnitude_of_perpendicular_vector_l457_457486


namespace bananas_left_l457_457396

theorem bananas_left (original_bananas removed_bananas : ℕ) (h1 : original_bananas = 46) (h2 : removed_bananas = 5) :
  original_bananas - removed_bananas = 41 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add (by norm_num : 46 = 41 + 5)

/-
Conditions:
- original_bananas = 46
- removed_bananas = 5

Question:
- Prove original_bananas - removed_bananas = 41
-/

end bananas_left_l457_457396


namespace count_oddly_powerful_integers_l457_457397

def is_oddly_powerful (m : ℕ) : Prop :=
  ∃ (c d : ℕ), d > 1 ∧ d % 2 = 1 ∧ c^d = m

theorem count_oddly_powerful_integers :
  ∃ (S : Finset ℕ), 
  (∀ m, m ∈ S ↔ (m < 1500 ∧ is_oddly_powerful m)) ∧ S.card = 13 :=
by
  sorry

end count_oddly_powerful_integers_l457_457397


namespace true_statements_about_averaging_l457_457260

def avg (x y : ℝ) : ℝ := (x + y) / 2

theorem true_statements_about_averaging :
  (∀ x y z : ℝ, avg (avg x y) z ≠ avg x (avg y z)) → -- Statement I
  (∀ x y : ℝ, avg x y = avg y x) →                  -- Statement II
  (∀ x y z : ℝ, avg x (y + z) ≠ avg x y + avg x z) →-- Statement III
  (∀ x y z : ℝ, x + avg y z = avg (x + y) (x + z)) →-- Statement IV
  (¬ ∃ i : ℝ, ∀ x : ℝ, avg x i = x) →               -- Statement V
  (true_statements = [2, 4] : list ℕ) :=             -- Given that II and IV are only true
by
  sorry

end true_statements_about_averaging_l457_457260


namespace wind_velocity_determination_l457_457299

theorem wind_velocity_determination (ρ : ℝ) (P1 P2 : ℝ) (A1 A2 : ℝ) (V1 V2 : ℝ) (k : ℝ) :
  ρ = 1.2 →
  P1 = 0.75 →
  A1 = 2 →
  V1 = 12 →
  P1 = ρ * k * A1 * V1^2 →
  P2 = 20.4 →
  A2 = 10.76 →
  P2 = ρ * k * A2 * V2^2 →
  V2 = 27 := 
by sorry

end wind_velocity_determination_l457_457299


namespace find_n_mod_10_l457_457782

theorem find_n_mod_10 : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := 
begin
  use 6,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end find_n_mod_10_l457_457782


namespace solution_set_of_inequality_l457_457084

-- Define the conditions as Lean hypotheses first:
variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop := ∀ x y ∈ s, x < y → f x < f y

-- Lean statement for the problem:
theorem solution_set_of_inequality
  (h1 : is_even f)
  (h2 : is_increasing_on f {x | x < 0})
  (h3 : f (-3) = 0) :
  {x | f x / x < 0} = set.Ioo (-3 : ℝ) 0 ∪ set.Ioi (3 : ℝ) :=
by
  sorry

end solution_set_of_inequality_l457_457084


namespace volume_given_surface_area_l457_457466

-- Define the variables and constants used in the conditions
def radius_of_sphere (A : ℝ) : ℝ :=
  real.sqrt (A / (4 * real.pi))

-- Define the volume function given the radius
noncomputable def volume_of_sphere (R : ℝ) : ℝ :=
  (4 / 3) * real.pi * R^3

-- Define the surface area condition
def surface_area (R : ℝ) : ℝ :=
  4 * real.pi * R^2

-- The main theorem stating given the surface area, the volume equals the correct answer
theorem volume_given_surface_area (A : ℝ) (h : A = 8 * real.pi) :
  volume_of_sphere (radius_of_sphere A) = (8 * real.sqrt 2 * real.pi / 3) :=
by {
  -- skipping the proof
  sorry
}

end volume_given_surface_area_l457_457466


namespace lambda_mu_relationship_l457_457139

theorem lambda_mu_relationship (a b : ℝ × ℝ) (λ μ : ℝ)
  (h1 : a = (1, 1))
  (h2 : b = (1, -1))
  (h3 : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457139


namespace pencilCost_is_1_l457_457726

noncomputable def sellingPrice : ℝ := 1.50
noncomputable def costToMake : ℝ := 0.90
noncomputable def numberOfPops : ℕ := 300
noncomputable def numberOfPencils : ℕ := 100
noncomputable def profitPerPop := sellingPrice - costToMake
noncomputable def totalProfit := profitPerPop * numberOfPops
noncomputable def costPerPencil := totalProfit / numberOfPencils

theorem pencilCost_is_1.80 : costPerPencil = 1.80 := sorry

end pencilCost_is_1_l457_457726


namespace johns_oil_change_cost_l457_457917

theorem johns_oil_change_cost:
  (miles_per_month: ℕ) (miles_per_oil_change: ℕ) (free_oil_changes_per_year: ℕ) (cost_per_oil_change: ℕ) 
  (h₁ : miles_per_month = 1000) 
  (h₂ : miles_per_oil_change = 3000) 
  (h₃ : free_oil_changes_per_year = 1) 
  (h₄ : cost_per_oil_change = 50) : 
  (12 * cost_per_oil_change * miles_per_oil_change) // (miles_per_month * miles_per_oil_change) - (free_oil_changes_per_year * cost_per_oil_change) = 150 := 
by 
  sorry

end johns_oil_change_cost_l457_457917


namespace no_integer_solution_l457_457636

theorem no_integer_solution (m n : ℤ) : m^2 - 11 * m * n - 8 * n^2 ≠ 88 :=
sorry

end no_integer_solution_l457_457636


namespace convert_spherical_to_rectangular_l457_457757

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta, rho * Real.sin phi * Real.sin theta, rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 10 (4 * Real.pi / 3) (Real.pi / 3) = (-5 * Real.sqrt 3, -15 / 2, 5) :=
by 
  sorry

end convert_spherical_to_rectangular_l457_457757


namespace vertical_stripes_percentage_l457_457510

-- Definitions for conditions
def total_people : ℕ := 100
def checkered : ℕ := 12
def polka_dotted : ℕ := 15
def plain_shirts : ℕ := 3
def horizontal_is_five_times_checkered : Prop := horizontal = 5 * checkered
def number_wearing_stripes : ℕ := total_people - checkered - polka_dotted - plain_shirts
def horizontal : ℕ := 5 * checkered
def vertical : ℕ := number_wearing_stripes - horizontal

-- Theorem stating the problem to be proved
theorem vertical_stripes_percentage :
  (vertical : ℝ) / (total_people : ℝ) * 100 = 10 := by
  sorry

end vertical_stripes_percentage_l457_457510


namespace A_leaves_after_2_days_l457_457707

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 30
noncomputable def C_work_rate : ℚ := 1 / 10
noncomputable def C_days_work : ℚ := 4
noncomputable def total_days_work : ℚ := 15

theorem A_leaves_after_2_days (x : ℚ) : 
  2 / 5 + x / 12 + (15 - x) / 30 = 1 → x = 2 :=
by
  intro h
  sorry

end A_leaves_after_2_days_l457_457707


namespace arcsin_cos_eq_l457_457407

theorem arcsin_cos_eq :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  have h1 : Real.cos (2 * Real.pi / 3) = -1 / 2 := sorry
  have h2 : Real.arcsin (-1 / 2) = -Real.pi / 6 := sorry
  rw [h1, h2]

end arcsin_cos_eq_l457_457407


namespace copper_content_range_l457_457654

theorem copper_content_range (x2 : ℝ) (y : ℝ) (h1 : 0 ≤ x2) (h2 : x2 ≤ 4 / 9) (hy : y = 0.4 + 0.075 * x2) : 
  40 ≤ 100 * y ∧ 100 * y ≤ 130 / 3 :=
by { sorry }

end copper_content_range_l457_457654


namespace expected_number_of_students_who_get_back_their_cards_l457_457651

-- Condition: There are 2012 students in a secondary school, each student writes a new year card,
-- the cards are mixed up and randomly distributed, and each student gets one and only one card.
noncomputable def students : List ℕ := List.range 2012

-- Define the indicator random variable that indicates if the i-th student gets their own card.
def indicator_variable (i : ℕ) : ℕ → ℕ :=
  λ card : ℕ => if i == card then 1 else 0

-- Define the random variable representing the number of students who get back their own cards.
def X (distribution : List ℕ) : ℕ :=
  List.foldr (λ i acc => indicator_variable i (List.nthLe distribution i (by simp))) 0 students

-- Define the expected value
def expected_value (students : List ℕ) : ℕ :=
  1

-- The final statement
theorem expected_number_of_students_who_get_back_their_cards :
  ∀ (distribution : List ℕ), List.length distribution = 2012 → X distribution = expected_value students :=
begin
  sorry
end

end expected_number_of_students_who_get_back_their_cards_l457_457651


namespace value_of_playstation_l457_457224

theorem value_of_playstation (V : ℝ) (H1 : 700 + 200 = 900) (H2 : V - 0.2 * V = 0.8 * V) (H3 : 0.8 * V = 900 - 580) : V = 400 :=
by
  sorry

end value_of_playstation_l457_457224


namespace line_points_satisfy_equation_l457_457459

theorem line_points_satisfy_equation (x_2 y_3 : ℝ) 
  (h_slope : ∃ k : ℝ, k = 2) 
  (h_P1 : ∃ P1 : ℝ × ℝ, P1 = (3, 5)) 
  (h_P2 : ∃ P2 : ℝ × ℝ, P2 = (x_2, 7)) 
  (h_P3 : ∃ P3 : ℝ × ℝ, P3 = (-1, y_3)) 
  (h_line : ∀ (x y : ℝ), y - 5 = 2 * (x - 3) ↔ 2 * x - y - 1 = 0) :
  x_2 = 4 ∧ y_3 = -3 :=
sorry

end line_points_satisfy_equation_l457_457459


namespace find_area_l457_457516

noncomputable def area_of_equilateral_triangle (A B C O D E : Type) 
  [Triangle A B C] 
  (is_equilateral : is_equilateral_triangle A B C)
  (AD BE : Segment A D × Segment B E)
  (medians_equal : AD.length = 15 ∧ BE.length = 15)
  (intersect_at_right_angle : is_right_angle AD BE at O) : ℝ :=
  450

theorem find_area (A B C O D E : Type) 
  [Triangle A B C] 
  (is_equilateral : is_equilateral_triangle A B C)
  (AD BE : Segment A D × Segment B E)
  (medians_equal : AD.length = 15 ∧ BE.length = 15)
  (intersect_at_right_angle : is_right_angle AD BE at O) :
  area_of_triangle A B C = 450 :=
sorry

end find_area_l457_457516


namespace perpendicular_vectors_condition_l457_457122

theorem perpendicular_vectors_condition (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
      b : ℝ × ℝ := (1, -1)
  in ((a.1 + λ * b.1) * (a.1 + μ * b.1) + (a.2 + λ * b.2) * (a.2 + μ * b.2) = 0) → (λ * μ = -1) :=
sorry

end perpendicular_vectors_condition_l457_457122


namespace problem_statement_l457_457030

def base5_to_base10 (n : ℕ) : ℕ :=
  5^4 * (n / 10000 % 10) + 5^3 * (n / 1000 % 10) + 5^2 * (n / 100 % 10) + 5^1 * (n / 10 % 10) + 5^0 * (n % 10)

def base8_to_base10 (n : ℕ) : ℕ :=
  8^3 * (n / 1000 % 10) + 8^2 * (n / 100 % 10) + 8^1 * (n / 10 % 10) + 8^0 * (n % 10)

theorem problem_statement : base5_to_base10 52431 - base8_to_base10 1432 = 2697 :=
by {
  unfold base5_to_base10,
  unfold base8_to_base10,
  -- The detailed proof steps would be added here.
  sorry
}

end problem_statement_l457_457030


namespace max_dot_product_value_l457_457833

open Real

variables (a b c : ℝ × ℝ)

def norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

def dot_prod (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def max_value (c : ℝ × ℝ) (a : ℝ × ℝ) : ℝ := c.1

theorem max_dot_product_value (a b c : ℝ × ℝ)
  (h1 : norm b = 2)
  (h2 : norm a = 1)
  (h3 : dot_prod a b = 1)
  (h4 : dot_prod (c - a) (c - b) = 0) :
  max_value c a = (2 + sqrt 3) / 2 :=
sorry

end max_dot_product_value_l457_457833


namespace find_cos_B_projection_of_BC_on_BA_l457_457176

variables {A B C : ℝ} {a b c : ℝ}

-- Assume the key conditions as hypotheses
axiom in_triangle : a = 2 * b * cos B - b * cos C = c * cos B
axiom sides : a = 2 * sqrt 3 / 3 ∧ b = 2

-- Proving the main results
theorem find_cos_B : cos B = 1 / 2 :=
sorry

noncomputable def measure_of_A : ℝ := arccos (sqrt 3 / 3)

theorem projection_of_BC_on_BA : 
  let projection := (2 * sqrt 3 / 3) * (1 / 2) in
  projection = sqrt 3 / 3 :=
sorry

end find_cos_B_projection_of_BC_on_BA_l457_457176


namespace curves_intersection_four_points_l457_457764

theorem curves_intersection_four_points (b : ℝ) :
  (∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
    x1^2 + y1^2 = b^2 ∧ y1 = x1^2 - b + 1 ∧
    x2^2 + y2^2 = b^2 ∧ y2 = x2^2 - b + 1 ∧
    x3^2 + y3^2 = b^2 ∧ y3 = x3^2 - b + 1 ∧
    x4^2 + y4^2 = b^2 ∧ y4 = x4^2 - b + 1 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧
    (x2, y2) ≠ (x3, y3) ∧ (x2, y2) ≠ (x4, y4) ∧
    (x3, y3) ≠ (x4, y4)) →
  b > 2 :=
sorry

end curves_intersection_four_points_l457_457764


namespace square_area_from_perimeter_l457_457194

theorem square_area_from_perimeter :
  ∀ (ABCD : Type) (side_length : ℝ), 
  (ABCD → True) →
  ∃ (perimeter : ℝ), perimeter = 160 →
  ∃ (area : ℝ), area = 1600 :=
by
  intros ABCD side_length _ perimeter perimeter_eq
  use 1600
  sorry

end square_area_from_perimeter_l457_457194


namespace ordered_pairs_polynomial_factorable_l457_457400

theorem ordered_pairs_polynomial_factorable:
  let count_pairs := (∑ a in finset.range(51).filter (λ a, a ≥ 1), (a + 1) / 2)
  count_pairs = 75 :=
by
  -- proof needs to be provided
  sorry

end ordered_pairs_polynomial_factorable_l457_457400


namespace quadrilateral_RS_length_l457_457184

theorem quadrilateral_RS_length 
  (PQ QR PS : ℝ) 
  (h₁ : ∠ PSQ = ∠ PRQ) 
  (h₂ : ∠ PQR = ∠ SRQ) 
  (h₃ : PQ = 7) 
  (h₄ : QR = 5) 
  (h₅ : PS = 9) : 
  ∃ (RS : ℝ), RS = 45 / 7 ∧ let p := 45, let q := 7; p.gcd q = 1 :=
by 
  sorry

end quadrilateral_RS_length_l457_457184


namespace admission_schemes_count_l457_457311

theorem admission_schemes_count :
  let students : Finset ℕ := {1, 2, 3, 4}
  let universities : Finset ℕ := {1, 2, 3}
  (∀ u ∈ universities, (∃ s ⊆ students, s.card ≥ 1)) →
  (Finset.card (students.powerset.filter (λ s, s.card = 2)) * 3.factorial) = 36 :=
sorry

end admission_schemes_count_l457_457311


namespace total_movies_attended_l457_457659

-- Defining the conditions for Timothy's movie attendance
def Timothy_2009 := 24
def Timothy_2010 := Timothy_2009 + 7

-- Defining the conditions for Theresa's movie attendance
def Theresa_2009 := Timothy_2009 / 2
def Theresa_2010 := Timothy_2010 * 2

-- Prove that the total number of movies Timothy and Theresa went to in both years is 129
theorem total_movies_attended :
  (Timothy_2009 + Timothy_2010 + Theresa_2009 + Theresa_2010) = 129 :=
by
  -- proof goes here
  sorry

end total_movies_attended_l457_457659


namespace problem_acute_angles_l457_457073

theorem problem_acute_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h1 : 3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1)
  (h2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := 
by 
  sorry

end problem_acute_angles_l457_457073


namespace S_k_is_correct_l457_457790

def gcd (m n : ℕ) : ℕ := if m = 0 then n else if n = 0 then m else gcd n (m % n)

theorem S_k_is_correct (k : ℕ) (hk : 0 < k) :
  S k = Int.ceil (2 * Real.sqrt k + 1) := sorry

end S_k_is_correct_l457_457790


namespace probability_of_equal_distribution_l457_457054

namespace MashedPotatoes

-- Definitions
def initial_amounts (n : ℕ) : list ℕ :=
  [1, 2, 4, 8]

def total_amount (initial : list ℕ) : ℕ :=
  initial.foldr (λ x acc, x + acc) 0

def equal_amount (total_net : ℕ) : ℚ :=
  total_net / 4

-- Probability calculation
theorem probability_of_equal_distribution : 
  let initial := initial_amounts 4 in
  let total := total_amount initial in
  (total = 15) →
  (equal_amount total = 15 / 4) →
  let probability_of_equal_blends : ℚ := 1 / 54 in
  probability_of_equal_blends = 1 / 6 * 2 / 3 * 1 / 6 :=
begin
  sorry -- Proof skipped
end

end MashedPotatoes

end probability_of_equal_distribution_l457_457054


namespace pirate_rick_dig_time_l457_457593
noncomputable def time_to_dig_up_treasure (initial_sand : ℕ) (final_sand : ℕ) (dig_time_per_foot : ℕ) : ℕ :=
  final_sand * dig_time_per_foot

theorem pirate_rick_dig_time :
  let initial_sand := 8 in
  let initial_time := 4 in
  let storm_fraction := 1 / 2 in
  let tsunami_sand := 2 in
  let dig_rate := initial_time / initial_sand in
  let storm_sand := initial_sand * storm_fraction in
  let total_sand := storm_sand + tsunami_sand in
  time_to_dig_up_treasure initial_sand total_sand dig_rate = 3 :=
by
  sorry

end pirate_rick_dig_time_l457_457593


namespace arithmetic_sequence_sum_is_93_l457_457644

noncomputable def arithmeticSeqSum : ℕ := 
  let a := 24
  let b := 31
  let c := 38
  a + b + c

theorem arithmetic_sequence_sum_is_93 (a b c : ℕ) (h_seq : 3 + 7 = 10 ∧ 10 + 7 = 17 ∧ 17 + 7 = a ∧ a + 7 = b ∧ b + 7 = c ∧ c + 7 = 38) : 
  a + b + c = 93 :=
by
  have ha : a = 24 := sorry
  have hb : b = 31 := sorry
  have hc : c = 38 := sorry
  rw [ha, hb, hc]
  exact rfl

end arithmetic_sequence_sum_is_93_l457_457644


namespace arithmetic_sequence_sum_l457_457815

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
hypothesis h_arithmetic : is_arithmetic_sequence a
hypothesis h_sum_condition : a 4 + a 6 + a 8 = 15
hypothesis h_sum_definition : ∀ n : ℕ, S n = (n / 2) * (a 1 + a n)

-- Proof goal
theorem arithmetic_sequence_sum : S 11 = 55 :=
by
  sorry

end arithmetic_sequence_sum_l457_457815


namespace woman_weaves_fabric_on_last_day_l457_457616

theorem woman_weaves_fabric_on_last_day
    (d : ℝ)
    (S : ℝ)
    (h1 : ∃ d : ℝ, ∑ k in finset.range 30, (5 + k * d) = 390)
    (h2 : S = 5 + 29 * d) :
    S = 21 :=
sorry

end woman_weaves_fabric_on_last_day_l457_457616


namespace find_a_l457_457470

def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

theorem find_a (a : ℝ) (h : f a = -2) : a = -3 :=
by sorry

end find_a_l457_457470


namespace cut_pieces_from_rod_l457_457691

theorem cut_pieces_from_rod (rod_length_in_meters : ℕ) (piece_length_in_cm : ℕ) (h_rod_length : rod_length_in_meters = 17) (h_piece_length : piece_length_in_cm = 85) :
  let rod_length_in_cm := rod_length_in_meters * 100 in
  rod_length_in_cm / piece_length_in_cm = 20 :=
by
  -- Since the theorem is about equivalence rather than how it's achieved, we assume the necessary length conversions
  have h_conversion : rod_length_in_cm = 1700, from sorry
  calc
    rod_length_in_cm / piece_length_in_cm 
        = 1700 / 85 : by rw [h_conversion, h_piece_length]
    ... = 20 : by norm_num

end cut_pieces_from_rod_l457_457691


namespace general_term_sum_first_n_terms_l457_457442

variables {a_n : ℕ → ℚ}

-- Given Conditions
axiom a5_eq_3 : a_n 5 = 3
axiom S3_eq_9_over_2 : (a_n 1 + a_n 2 + a_n 3) = 9 / 2

-- To Prove General Term of the Sequence
theorem general_term (n : ℕ) : a_n n = 1 / 2 * (n + 1) :=
by sorry

-- Sequence for Sum of First n Terms
def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 2))

-- To Prove Sum of First n Terms of {b_n}
theorem sum_first_n_terms (n : ℕ) : 
(∑ k in Finset.range n, b_n k) = 2 * n / ((n + 2) * (n + 3)) :=
by sorry

end general_term_sum_first_n_terms_l457_457442


namespace tangency_of_circles_l457_457192

-- Definitions of points and geometric entities
variables {A B C D P Q K L T : Type}
variables [IsCyclicQuadrilateral A B C D]
variables [PointOnLine P A B]
variables [Intersection Q (LineThrough A C) (Segment D P)]
variables [ParallelThrough K P C D]
variables [Intersection K (LineThrough P (Extend C B))]
variables [ParallelThrough L Q B D]
variables [Intersection L (LineThrough Q (Extend C B))]

theorem tangency_of_circles
  (hp: isCyclicQuadrilateral A B C D)
  (hP: PointOnLine P A B)
  (hQ: Intersection Q (LineThrough A C) (Segment D P))
  (hK: ParallelThrough K P C D)
  (hKe: Intersection K (LineThrough P (Extend C B)))
  (hL: ParallelThrough L Q B D)
  (hLe: Intersection L (LineThrough Q (Extend C B))) :
  tangent (circumcircle B K P) (circumcircle C L Q) :=
sorry

end tangency_of_circles_l457_457192


namespace area_of_Omega_range_l457_457480

theorem area_of_Omega_range:
  ∀ (r : ℝ), 0 < r ∧ r < 0.5 →
  ∀ (C : ℝ → ℝ) (D : ℝ × ℝ → ℝ) (A : ℝ × ℝ), 
  (C = λ x, 0.5 * x^2) →
  (D = λ p, p.1 ^ 2 + (p.2 - 0.5) ^ 2) →
  (∀ x : ℝ, (0.5 * x^2) ≠ (D (x, 0.5 * x^2))) →
  (∀ y: ℝ, (y, 0.5 * y^2) = A) →
  (∃ E F : ℝ × ℝ, ¬(x = 0.5 * x^2) →
    ∀ Ω: Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∉ (SetOf (λ q, q.1 ≠ (0.5 * q.1^2) ∧ q.2 ≠ (0.5 * q.1^2)))
      → 0 < Ω.measure ∧ Ω.measure < π / 16) := sorry

end area_of_Omega_range_l457_457480


namespace equilateral_triangle_largest_angle_l457_457665

theorem equilateral_triangle_largest_angle (DEF : Type) [EquilateralTriangle DEF] :
  largest_interior_angle DEF = 60 :=
sorry

end equilateral_triangle_largest_angle_l457_457665


namespace cos_18_deg_l457_457008

theorem cos_18_deg :
  let x := real.cos (real.pi / 10) in
  let y := real.cos (2 * real.pi / 10) in
  (y = 2 * x^2 - 1) ∧
  (4 * x^3 - 3 * x = real.sin (real.pi * 2 / 5)) →
  (real.cos (real.pi / 10) = real.sqrt ((5 + real.sqrt 5) / 8)) :=
by
  sorry

end cos_18_deg_l457_457008


namespace find_n_mod_10_l457_457781

theorem find_n_mod_10 : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := 
begin
  use 6,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end find_n_mod_10_l457_457781


namespace car_total_travel_time_l457_457583

def T_NZ : ℕ := 60

def T_NR : ℕ := 8 / 10 * T_NZ -- 80% of T_NZ

def T_ZV : ℕ := 3 / 4 * T_NR -- 75% of T_NR

theorem car_total_travel_time :
  T_NZ + T_NR + T_ZV = 144 := by
  sorry

end car_total_travel_time_l457_457583


namespace f_neg_4_eq_3_l457_457085

-- Define the function f based on the given conditions
def f (x : ℝ) : ℝ :=
  if h : x > 0 then -x + 1 else -(-x + 1)

-- Prove that f is an odd function
lemma f_is_odd (x : ℝ) : f (-x) = -f x := by
  sorry

-- Use the given conditions to show f(-4) = 3
theorem f_neg_4_eq_3 : f (-4) = 3 := by
  have : f_is_odd := f_is_odd
  rw [←this]
  have : f 4 = -4 + 1 := rfl
  rw [this]
  norm_num

end f_neg_4_eq_3_l457_457085


namespace coefficient_of_x4_l457_457778

def expression := 5 * (x^4 - 2 * x^5) + 3 * (2 * x^2 - x^6 + x^3) - (2 * x^6 - 3 * x^4 + x^2)

theorem coefficient_of_x4 : coeff (5 * (x^4 - 2 * x^5) + 3 * (2 * x^2 - x^6 + x^3) - (2 * x^6 - 3 * x^4 + x^2)) 4 = 8 :=
by sorry

end coefficient_of_x4_l457_457778


namespace x_can_be_any_sign_l457_457167

theorem x_can_be_any_sign
  (x y p q : ℝ)
  (h1 : abs (x / y) < abs (p) / q^2)
  (h2 : y ≠ 0) (h3 : q ≠ 0) :
  ∃ (x' : ℝ), True :=
by
  sorry

end x_can_be_any_sign_l457_457167


namespace lambda_mu_relationship_l457_457142

variables (λ μ : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def linear_combination (v : ℝ × ℝ) (c : ℝ) (w : ℝ × ℝ) := 
  (v.1 + c * w.1, v.2 + c * w.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem lambda_mu_relationship (h : dot_product (linear_combination vector_a λ vector_b)
                                          (linear_combination vector_a μ vector_b) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457142


namespace construct_triangle_from_square_centers_l457_457351

def Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A B C : Point

noncomputable def square_center (A B C : Point) : Point := 
  sorry  -- Placeholder for the actual function to compute the center of the square

def is_square_center (A B C A1 B1 C1 : Point) : Prop :=
  (square_center B C A = A1) ∧
  (square_center C A B = B1) ∧
  (square_center A B C = C1)

theorem construct_triangle_from_square_centers (A1 B1 C1 : Point) :
  ∃ (A B C : Point), is_square_center A B C A1 B1 C1 :=
  sorry

end construct_triangle_from_square_centers_l457_457351


namespace possible_values_of_expr_l457_457806

-- Define conditions
variables (x y : ℝ)
axiom h1 : x + y = 2
axiom h2 : y > 0
axiom h3 : x ≠ 0

-- Define the expression we're investigating
noncomputable def expr : ℝ := (1 / (abs x)) + (abs x / (y + 2))

-- The statement of the problem
theorem possible_values_of_expr :
  expr x y = 3 / 4 ∨ expr x y = 5 / 4 :=
sorry

end possible_values_of_expr_l457_457806


namespace production_rate_equation_l457_457334

theorem production_rate_equation (x : ℝ) (h1 : ∀ t : ℝ, t = 600 / (x + 8)) (h2 : ∀ t : ℝ, t = 400 / x) : 
  600/(x + 8) = 400/x :=
by
  sorry

end production_rate_equation_l457_457334


namespace no_integer_solution_l457_457154

theorem no_integer_solution (y : ℤ) : ¬ (-3 * y ≥ y + 9 ∧ 2 * y ≥ 14 ∧ -4 * y ≥ 2 * y + 21) :=
sorry

end no_integer_solution_l457_457154


namespace solve_for_x_l457_457195

theorem solve_for_x (x : ℚ) : (1 / 7 + 7 / x = 16 / x + 1 / 16) → x = 112 :=
begin
  sorry
end

end solve_for_x_l457_457195


namespace perfect_square_condition_l457_457760

theorem perfect_square_condition (n : ℕ) (h : 0 < n) (h1 : ∃ a : ℕ, n * 2^(n - 1) + 1 = a^2) : n = 5 :=
begin
  sorry
end

end perfect_square_condition_l457_457760


namespace dogs_not_eat_either_l457_457512

-- Let's define the conditions
variables (total_dogs : ℕ) (dogs_like_carrots : ℕ) (dogs_like_chicken : ℕ) (dogs_like_both : ℕ)

-- Given conditions
def conditions : Prop :=
  total_dogs = 85 ∧
  dogs_like_carrots = 12 ∧
  dogs_like_chicken = 62 ∧
  dogs_like_both = 8

-- Problem to solve
theorem dogs_not_eat_either (h : conditions total_dogs dogs_like_carrots dogs_like_chicken dogs_like_both) :
  (total_dogs - (dogs_like_carrots - dogs_like_both + dogs_like_chicken - dogs_like_both + dogs_like_both)) = 19 :=
by {
  sorry 
}

end dogs_not_eat_either_l457_457512


namespace exists_zero_in_interval_l457_457100

noncomputable def f (x : ℝ) : ℝ := 6 / x - x^2

theorem exists_zero_in_interval : ∃ c ∈ Ioo (1 : ℝ) 2, f c = 0 := by
  sorry

end exists_zero_in_interval_l457_457100


namespace find_x0_l457_457433

def f : ℝ → ℝ := λ x, Real.log x

theorem find_x0 (x_0 : ℝ) (h : deriv^[3] f x_0 = 1 / x_0^2) : x_0 = 1 / 2 :=
by
  sorry

end find_x0_l457_457433


namespace distance_orthocenter_circumcenter_l457_457518

theorem distance_orthocenter_circumcenter (α a : ℝ) (hα : α > π / 2):
  ∀ (H O : ℝ), 
    -- Given
    is_isosceles_triangle α a → 
    ∃ (BC : ℝ), 
      BC = a → 
        distance_orthocenter_to_circumcenter α BC = (BC / 2) * (Real.tan (α / 2) - Real.cot α) :=
by { sorry }

end distance_orthocenter_circumcenter_l457_457518


namespace ellipse_iff_k_range_l457_457870

theorem ellipse_iff_k_range (k : ℝ) :
  (∃ x y, (x ^ 2 / (1 - k)) + (y ^ 2 / (1 + k)) = 1) ↔ (-1 < k ∧ k < 1 ∧ k ≠ 0) :=
by
  sorry

end ellipse_iff_k_range_l457_457870


namespace square_area_from_parabola_and_line_intersection_l457_457729

theorem square_area_from_parabola_and_line_intersection :
  (∃ (x1 x2 : ℝ), (10 = x1^2 + 4 * x1 + 3) ∧ (10 = x2^2 + 4 * x2 + 3) ∧ x1 ≠ x2 ∧ 
  let side_length := abs (x2 - x1) in side_length * side_length = 44) :=
sorry

end square_area_from_parabola_and_line_intersection_l457_457729


namespace bounded_area_arcsin_cos_l457_457419

noncomputable def bounded_area : ℝ :=
  ∫ x in 0..2 * Real.pi, Real.arcsin (Real.cos x)

theorem bounded_area_arcsin_cos :
  bounded_area = (Real.pi^2) / 2 := by
  sorry

end bounded_area_arcsin_cos_l457_457419


namespace find_s_for_g_eq_0_l457_457392

def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

theorem find_s_for_g_eq_0 : ∃ (s : ℝ), g 3 s = 0 → s = -867 :=
by
  sorry

end find_s_for_g_eq_0_l457_457392


namespace running_to_weightlifting_ratio_l457_457227

-- Definitions for given conditions in the problem
def total_practice_time : ℕ := 120 -- 120 minutes
def shooting_time : ℕ := total_practice_time / 2
def weightlifting_time : ℕ := 20
def running_time : ℕ := shooting_time - weightlifting_time

-- The goal is to prove that the ratio of running time to weightlifting time is 2:1
theorem running_to_weightlifting_ratio : running_time / weightlifting_time = 2 :=
by
  /- use the given problem conditions directly -/
  exact sorry

end running_to_weightlifting_ratio_l457_457227


namespace lambda_mu_relationship_l457_457146

variables (λ μ : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def linear_combination (v : ℝ × ℝ) (c : ℝ) (w : ℝ × ℝ) := 
  (v.1 + c * w.1, v.2 + c * w.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem lambda_mu_relationship (h : dot_product (linear_combination vector_a λ vector_b)
                                          (linear_combination vector_a μ vector_b) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457146


namespace binomial_expansion_properties_l457_457457

noncomputable def binomial_sum (n : ℕ) : ℕ := 2^n

noncomputable def largest_terms (n : ℕ) (x : ℝ) : List (ℝ × ℝ) := 
  if n = 8 then [(7, x^(7/3)), (7, x^(2/3))] else []

noncomputable def rational_terms (n : ℕ) (x : ℝ) : List (ℝ × ℝ) := 
  if n = 8 then [(1, x^4), (7/16, x^(-1))] else []

theorem binomial_expansion_properties (n : ℕ) (x : ℝ) 
  (h₁ : (sqrt x + 1 / (2 * (x^(1/3))))^n = ∑ k in finset.range (n + 1), 
    (nat.choose n k) * (sqrt x)^(n - k) * (1 / (2 * x^(1/3)))^k)
  (h₂ : 2 * ((1/2)*n) = 1 + (n*(n-1))/8) :
  binomial_sum n = 256 ∧ 
  largest_terms n x = [(7, x^(7/3)), (7, x^(2/3))] ∧ 
  rational_terms n x = [(1, x^4), (7/16, x^(-1))] :=
by sorry

end binomial_expansion_properties_l457_457457


namespace range_of_a_l457_457893

noncomputable theory

open Real
open Classical

variables (a : ℝ)

def center_on_line (a : ℝ) : Prop :=
  ∃ x y : ℝ, y = 2*x - 4 ∧ (y = 2*a-4) 

def condition_dist (a : ℝ) : Prop := 
  sqrt (5 * a^2 - 12 * a + 9) ≤ 2

theorem range_of_a :
  ∃ a : ℝ, (center_on_line a) ∧ (condition_dist a) → (0 ≤ a ∧ a ≤ 12 / 5) :=
sorry

end range_of_a_l457_457893


namespace sides_of_triangle_expr_negative_l457_457861

theorem sides_of_triangle_expr_negative (a b c : ℝ) 
(h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
(a - c)^2 - b^2 < 0 :=
sorry

end sides_of_triangle_expr_negative_l457_457861


namespace max_electric_field_is_correct_l457_457669

noncomputable def CoulombConstant := sorry -- Placeholder for Coulomb constant 'k'

def electric_field_strength (Q : ℝ) (d : ℝ) (r : ℝ) : ℝ :=
2 * CoulombConstant * Q * r / ((r^2 + d^2)^(3/2))

def maximize_electric_field (Q : ℝ) (d : ℝ) : ℝ :=
electric_field_strength Q d (d * Real.sqrt 2 / 2)

theorem max_electric_field_is_correct (Q : ℝ) (d : ℝ) :
  maximize_electric_field Q d = (4 * Real.sqrt 3 / 9) * (CoulombConstant * Q / d^2) := sorry

end max_electric_field_is_correct_l457_457669


namespace tangent_line_equation_maximum_area_and_intersecting_line_l457_457448

noncomputable def circle_c (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 4
def point_A := (1, 0)

theorem tangent_line_equation {l₁ : ℝ → ℝ} :
  (∃ (k : ℝ), l₁ = λ x, k * (x - 1)) ∧ 
  (∀ (x y : ℝ), circle_c x y → ∃! (a b : ℝ), a = x ∧ b = l₁ x) → 
  (∀ (x y : ℝ), circle_c x y → x = 1 ∨ 3 * x - 4 * y = 3) :=
sorry

theorem maximum_area_and_intersecting_line {P Q : ℝ × ℝ} {k : ℝ} :
  (P.1 ≠ Q.1 ∧ P.2 ≠ Q.2 ∧ circle_c P.1 P.2 ∧ circle_c Q.1 Q.2 ∧ ∀ (x y : ℝ), y = k * (x - 1) ∧ circle_c x y) → 
  let d := abs(2 * k - 4) / sqrt(1 + k^2),
      S := d * sqrt(4 - d^2) in
  S = 2 ∧ (k = 1 ∨ k = 7) :=
sorry

end tangent_line_equation_maximum_area_and_intersecting_line_l457_457448


namespace set_intersection_l457_457849

-- Definitions of sets M and N
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

-- The statement to prove that M ∩ N = {1, 2}
theorem set_intersection :
  M ∩ N = {1, 2} := by
  sorry

end set_intersection_l457_457849


namespace minimum_point_translated_graph_l457_457292

noncomputable def original_function (x : ℝ) : ℝ := abs x - 5

def translated_function (x : ℝ) : ℝ := original_function (x - 4) + 2

theorem minimum_point_translated_graph :
  ∃ (p : ℝ × ℝ), p = (4, -3) ∧ (∀ x : ℝ, translated_function x ≥ p.2) :=
begin
  use (4, -3),
  split,
  { -- Prove the point is (4, -3)
    refl, },
  { -- Show that for all x, y >= -3
    intro x,
    -- since abs (x - 4) >= 0, we have for translated function that 
    -- original_function(x - 4) + 2 >= -3
    sorry, }
end

end minimum_point_translated_graph_l457_457292


namespace relationship_between_sets_l457_457301

def M (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k - 2
def P (x : ℤ) : Prop := ∃ n : ℤ, x = 5 * n + 3
def S (x : ℤ) : Prop := ∃ m : ℤ, x = 10 * m + 3

theorem relationship_between_sets :
  (∀ x, S x → P x) ∧ (∀ x, P x → M x) ∧ (∀ x, M x → P x) :=
by
  sorry

end relationship_between_sets_l457_457301


namespace two_digit_integers_mod_9_eq_3_l457_457157

theorem two_digit_integers_mod_9_eq_3 :
  { x : ℕ | 10 ≤ x ∧ x < 100 ∧ x % 9 = 3 }.finite.card = 10 :=
by sorry

end two_digit_integers_mod_9_eq_3_l457_457157


namespace ordinary_equation_curve_C_cartesian_coordinate_line_l_max_distance_point_P_to_line_l_l457_457895

noncomputable section

open Real

-- Define the parametric equations of the curve C
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos θ, sin θ)

-- Define the equation of line l in polar coordinates
def polar_line (ρ θ : ℝ) : Prop :=
  sqrt 2 * ρ * sin (θ - π / 4) = 3

-- Prove the ordinary equation of curve C
theorem ordinary_equation_curve_C (x y θ : ℝ) (h1 : x = sqrt 3 * cos θ) (h2 : y = sin θ) :
  x ^ 2 / 3 + y ^ 2 = 1 := sorry

-- Prove the cartesian coordinate equation of line l
theorem cartesian_coordinate_line_l (x y ρ θ : ℝ)
    (h1 : sqrt 2 * ρ * sin (θ - π / 4) = 3)
    (hx : ρ = sqrt (x^2 + y^2))
    (hy : tan θ = y / x) :
  x - y + 3 = 0 := sorry

-- Prove the maximum distance from any point P on curve C to line l
theorem max_distance_point_P_to_line_l (θ : ℝ) :
  let P := curve_C θ in
  ∃ (max_dist : ℝ), max_dist = 5 * sqrt 2 / 2 := sorry

end ordinary_equation_curve_C_cartesian_coordinate_line_l_max_distance_point_P_to_line_l_l457_457895


namespace perpendicular_vectors_condition_l457_457126

theorem perpendicular_vectors_condition (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
      b : ℝ × ℝ := (1, -1)
  in ((a.1 + λ * b.1) * (a.1 + μ * b.1) + (a.2 + λ * b.2) * (a.2 + μ * b.2) = 0) → (λ * μ = -1) :=
sorry

end perpendicular_vectors_condition_l457_457126


namespace lambda_mu_relationship_l457_457144

variables (λ μ : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def linear_combination (v : ℝ × ℝ) (c : ℝ) (w : ℝ × ℝ) := 
  (v.1 + c * w.1, v.2 + c * w.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem lambda_mu_relationship (h : dot_product (linear_combination vector_a λ vector_b)
                                          (linear_combination vector_a μ vector_b) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457144


namespace generalized_inequality_l457_457011

theorem generalized_inequality (n k : ℕ) (h1 : 3 ≤ n) (h2 : 1 ≤ k ∧ k ≤ n) : 
  2^n + 5^n > 2^(n - k) * 5^k + 2^k * 5^(n - k) := 
by 
  sorry

end generalized_inequality_l457_457011


namespace min_distance_from_curve_to_line_is_correct_l457_457036

noncomputable def min_distance_to_curve : ℝ :=
  (λ x : ℝ, (e^x - x) / sqrt(2)) 0

theorem min_distance_from_curve_to_line_is_correct :
  min_distance_to_curve = sqrt(2) / 2 :=
by
  sorry

end min_distance_from_curve_to_line_is_correct_l457_457036


namespace set_union_complement_l457_457482

-- Definitions based on provided problem statement
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}
def CRQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- The theorem to prove
theorem set_union_complement : P ∪ CRQ = {x | -2 < x ∧ x ≤ 3} :=
by
  -- Skip the proof
  sorry

end set_union_complement_l457_457482


namespace solve_equation_l457_457610

theorem solve_equation (x y : ℤ) (h : 3 * (y - 2) = 5 * (x - 1)) :
  (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 7) :=
sorry

end solve_equation_l457_457610


namespace circle_equation_m_l457_457871
open Real

theorem circle_equation_m (m : ℝ) : (x^2 + y^2 + 4 * x + 2 * y + m = 0 → m < 5) := sorry

end circle_equation_m_l457_457871


namespace find_range_of_m_l457_457072

/-- Given propositions p and q:
1. The function f(x)=2x^2 - 2(m-2)x + 3m - 1 is monotonically increasing on the interval (1,2).
2. The equation (x^2)/(m+1) + (y^2)/(9-m) = 1 represents an ellipse with foci on the y-axis.
3. If 'p or q is true', 'p and q are false', and '¬p is false', then the range of values for m is (-∞, -1] ∪ {4}.
-/
theorem find_range_of_m (m : ℝ) (f : ℝ → ℝ) (p q : Prop) :
  (∀ x ∈ set.Ioo (1 : ℝ) (2 : ℝ), (deriv f x) ≥ 0)
  ∧ (∀ x y : ℝ, x^2 / (m+1) + y^2 / (9 - m) = 1 → (9 - m > m + 1))
  ∧ ((p ∨ q) ∧ ¬(p ∧ q)) ∧ ¬(¬p) →
  (m ∈ set.Iic (-1) ∨ m = 4) :=
begin
  sorry
end

end find_range_of_m_l457_457072


namespace minimum_value_l457_457238

open Real
open Classical

noncomputable def minimum_value_problem (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x * y * z = 3 / 4) (h5 : x + y + z = 4) : ℝ :=
  x^3 + x^2 + 4 * x * y + 12 * y^2 + 8 * y * z + 3 * z^2 + z^3

theorem minimum_value :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 3 / 4 ∧ x + y + z = 4 ∧
  minimum_value_problem x y z (by linarith) (by linarith) (by linarith) (by linarith) (by linarith) = 10.5 :=
sorry

end minimum_value_l457_457238


namespace time_taken_to_ride_l457_457215

theorem time_taken_to_ride (distance : ℕ) (speed : ℕ) (h_distance : distance = 10) (h_speed : speed = 2) : (distance / speed) = 5 := by
  -- Given assumptions
  have h1 : distance = 10 := h_distance
  have h2 : speed = 2 := h_speed
  -- Calculation
  have h3 : distance / speed = 10 / 2 := 
    by rw [h1, h2]
  -- Concluding
  have h4 : 10 / 2 = 5 := rfl
  -- Thus, 10 / 2 = 5
  rw [h3, h4]
  sorry

end time_taken_to_ride_l457_457215


namespace exists_alpha_symmetric_about_ya_l457_457791

def f (x : ℝ) : ℝ := sin x + sqrt 3 * cos x

theorem exists_alpha_symmetric_about_ya {α : ℝ} (hα : 0 < α ∧ α < π / 3) :
  ∃ α, α ∈ set.Ioo 0 (π / 3) ∧ ∀ x, f (x + α) = f (-x - α) := sorry

end exists_alpha_symmetric_about_ya_l457_457791


namespace expand_expression_l457_457412

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 :=
by sorry

end expand_expression_l457_457412


namespace triangle_ratio_l457_457735

theorem triangle_ratio (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let p := 12;
  let q := 8;
  let segment_length := L / p;
  let segment_width := W / q;
  let area_X := (segment_length * segment_width) / 2;
  let area_rectangle := L * W;
  (area_X / area_rectangle) = (1 / 192) :=
by 
  sorry

end triangle_ratio_l457_457735


namespace perpendicular_dot_product_l457_457131

theorem perpendicular_dot_product (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → (λ * μ = -1) :=
by
  intros
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  sorry

end perpendicular_dot_product_l457_457131


namespace distance_from_center_to_line_l457_457683

noncomputable def circle_center : ℝ × ℝ :=
  (0, 2)

noncomputable def line_eq (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * t, t)

def point_to_line_distance (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a * a + b * b)

theorem distance_from_center_to_line :
  point_to_line_distance (circle_center.fst) (circle_center.snd) 1 (-sqrt 3) 0 = sqrt 3 :=
by
  sorry

end distance_from_center_to_line_l457_457683


namespace tetrahedron_edge_length_l457_457053

-- Definitions for the given geometric configuration and problem setup
def radius : ℝ := 2
def platform_height : ℝ := 2

-- Given conditions as Lean definitions/assumptions
def ball_centers_are_mutually_tangent := sorry
def one_ball_elevated_on_platform := sorry
def three_balls_rest_on_floor := sorry
def tetrahedron_circumscribes_balls := sorry

-- Main statement to prove
theorem tetrahedron_edge_length : 
  ∃ (s : ℝ), 
    ball_centers_are_mutually_tangent ∧
    one_ball_elevated_on_platform ∧
    three_balls_rest_on_floor ∧
    tetrahedron_circumscribes_balls →
    s = (8 * Real.sqrt 3) / 3 :=
sorry

end tetrahedron_edge_length_l457_457053


namespace dot_product_given_angle_angle_given_perpendicularity_l457_457105

section

variables (a b : ℝ × ℝ) (angle_ba : ℝ) (angle_ab : ℝ)

-- Setting the conditions
def a := (1, -1)
def norm_b := 1

-- First problem: Dot product given angle
theorem dot_product_given_angle
  (hb : ‖b‖ = norm_b)
  (angle_ba : ℝ)
  (hb_angle : angle_ba = π / 3)
  : a.1 * b.1 + a.2 * b.2 = (Real.sqrt 2) / 2 := sorry

-- Second problem: Angle given perpendicularity
theorem angle_given_perpendicularity
  (hb : ‖b‖ = norm_b)
  (h_perp : (a.1 - b.1, a.2 - b.2) ⬝ b = 0)
  : angle_ab = π / 4 := sorry

end

end dot_product_given_angle_angle_given_perpendicularity_l457_457105


namespace range_of_t_l457_457471

def f (x m n : ℝ) := m * x^3 + n * x^2

theorem range_of_t (m n : ℝ)
  (h1 : ∀ x, f (x:ℝ) 1 3 = x ^ 3 + 3 * x ^ 2)
  (h2 : ∀ t:ℝ, ∀ x ∈ set.Icc (t : ℝ) (t + 1), deriv (f x 1 3) x < 0) :
  set.Icc (-2) (-1) :=
by
  sorry

end range_of_t_l457_457471


namespace math_problem_proof_l457_457932

noncomputable def problem_statement (a b c : ℝ) : Prop :=
 (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a + b + c = 0) ∧ (a^4 + b^4 + c^4 = a^6 + b^6 + c^6) → 
 (a^2 + b^2 + c^2 = 3 / 2)

theorem math_problem_proof : ∀ (a b c : ℝ), problem_statement a b c :=
by
  intros
  sorry

end math_problem_proof_l457_457932


namespace cos_18_eq_l457_457003

-- Definitions for the conditions.
def a := Real.cos (36 * Real.pi / 180)
def c := Real.cos (18 * Real.pi / 180)

-- Statement of the problem
theorem cos_18_eq :
  c = (Real.sqrt (10 + 2 * Real.sqrt 5) / 4) :=
by
  -- conditions given in the problem
  have h1: a = 2 * c^2 - 1, from sorry
  have h2: Real.sin (36 * Real.pi / 180) = Real.sqrt (1 - a^2), from sorry
  have triple_angle: Real.cos (3 * θ) = 4 * (Real.cos θ)^3 - 3 * (Real.cos θ), from sorry
  sorry

end cos_18_eq_l457_457003


namespace square_of_product_of_third_sides_l457_457603

-- Given data for triangles P1 and P2
variables {a b c d : ℝ}

-- Areas of triangles P1 and P2
def area_P1_pos (a b : ℝ) : Prop := a * b / 2 = 3
def area_P2_pos (a d : ℝ) : Prop := a * d / 2 = 6

-- Condition that b = d / 2
def side_ratio (b d : ℝ) : Prop := b = d / 2

-- Pythagorean theorem applied to both triangles
def pythagorean_P1 (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def pythagorean_P2 (a d c : ℝ) : Prop := a^2 + d^2 = c^2

-- The goal is to prove (cd)^2 = 120
theorem square_of_product_of_third_sides (a b c d : ℝ)
  (h_area_P1: area_P1_pos a b) 
  (h_area_P2: area_P2_pos a d) 
  (h_side_ratio: side_ratio b d) 
  (h_pythagorean_P1: pythagorean_P1 a b c) 
  (h_pythagorean_P2: pythagorean_P2 a d c) :
  (c * d)^2 = 120 := 
sorry

end square_of_product_of_third_sides_l457_457603


namespace lambda_mu_relationship_l457_457145

variables (λ μ : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def linear_combination (v : ℝ × ℝ) (c : ℝ) (w : ℝ × ℝ) := 
  (v.1 + c * w.1, v.2 + c * w.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem lambda_mu_relationship (h : dot_product (linear_combination vector_a λ vector_b)
                                          (linear_combination vector_a μ vector_b) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457145


namespace find_a_plus_c_l457_457294

noncomputable def a_b_c_d (a b c d : ℝ) : Prop :=
(y = -|x - a| + b) ∧ (y = |x - c| + d)

theorem find_a_plus_c (a b c d : ℝ) (h₀ : a_b_c_d a b c d) (h₁ : (2, 5)) (h₂ : (8, 3)) :
  a + c = 10 :=
by
  -- Conditions from the problem
  have h₃ : (2, 5) ∈ a_b_c_d a b c d := sorry
  have h₄ : (8, 3) ∈ a_b_c_d a b c d := sorry
  
  -- Define midpoint and calculations needed
  let midpoint := (5, 4)
  
  -- Using properties of shapes and equations derived
  have h₅ : (a + c) / 2 = 5 := sorry
  have h₆ : (b + d) / 2 = 4 := sorry

  -- Calculation of result based on midpoint property
  have result : a + c = 10 := sorry
  exact result

end find_a_plus_c_l457_457294


namespace math_problem_l457_457460

noncomputable theory
open Real

-- Definitions based on conditions
def ellipse_eq (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2 * p * x
def point_to_line_distance (x y : ℝ) (line : ℝ → ℝ → Prop) (d : ℝ) : Prop :=
  ∃ (a b c : ℝ), line a b ∧ (a * x + b * y + c) / sqrt(a^2 + b^2) = d

def common_chord_length (ellipse parabola : ℝ → ℝ → Prop) (length : ℝ) : Prop :=
  -- A simplified assumption here, in practice you'd need to prove the actual common points and their distance.
  ∃ x1 y1 x2 y2 : ℝ, ellipse x1 y1 ∧ parabola x2 y2 ∧ √((x1 - x2)^2 + (y1 - y2)^2) = length

-- Theorem statement
theorem math_problem 
  (a b p : ℝ) 
  (ha : a > 0 ∧ b > 0 ∧ a > b)
  (hp : p > 0) 
  (F : ℝ × ℝ)
  (hF : F = (1, 0)) 
  (h1 : point_to_line_distance (F.1) (F.2) (λ a b, a - b + 1 = 0) (sqrt 2))
  (h2 : ∃ ellipse parabola, ellipse_eq a b = ellipse ∧ parabola_eq p = parabola ∧ common_chord_length ellipse parabola (2 * sqrt 6)) :
  (ellipse_eq 3 sqrt 8 F.1 F.2 ∧ F = (1, 0)) ∧ 
  ∀ l : ℝ → ℝ → Prop,
  (∃ (A B : ℝ × ℝ), ∃ (C D : ℝ × ℝ),
    l A.1 A.2 ∧ l B.1 B.2 ∧ 
    l C.1 C.2 ∧ l D.1 D.2 ∧ 
    ellipse_eq 3 sqrt 8 A.1 A.2 ∧ ellipse_eq 3 sqrt 8 B.1 B.2 ∧
    parabola_eq 2 C.1 C.2 ∧ parabola_eq 2 D.1 D.2 ∧
    (1 / abs (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) + 
     1 / abs (sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) ∈ set.Ioc (1/6) (7/16))) :=
sorry

end math_problem_l457_457460


namespace solve_quadratic_equation_l457_457647

theorem solve_quadratic_equation : ∀ x : ℝ, x * (x - 14) = 0 ↔ x = 0 ∨ x = 14 :=
by
  sorry

end solve_quadratic_equation_l457_457647


namespace cost_of_apple_l457_457607

variable (A O : ℝ)

theorem cost_of_apple :
  (6 * A + 3 * O = 1.77) ∧ (2 * A + 5 * O = 1.27) → A = 0.21 :=
by
  intro h
  -- Proof goes here
  sorry

end cost_of_apple_l457_457607


namespace lambda_mu_condition_l457_457112

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem lambda_mu_condition (λ μ : ℝ) :
  orthogonal (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
             (vector_a.1 + μ * vector_b.1, vector_a.2 + μ * vector_b.2) →
  λ * μ = -1 :=
by
  sorry

end lambda_mu_condition_l457_457112


namespace cos_18_deg_l457_457005

theorem cos_18_deg :
  let x := real.cos (real.pi / 10) in
  let y := real.cos (2 * real.pi / 10) in
  (y = 2 * x^2 - 1) ∧
  (4 * x^3 - 3 * x = real.sin (real.pi * 2 / 5)) →
  (real.cos (real.pi / 10) = real.sqrt ((5 + real.sqrt 5) / 8)) :=
by
  sorry

end cos_18_deg_l457_457005


namespace sum_of_solutions_eq_zero_l457_457521

theorem sum_of_solutions_eq_zero (x y : ℝ) (h1 : y = 6) (h2 : x^2 + y^2 = 169) : (x = sqrt 133 ∨ x = -sqrt 133) → (sqrt 133 + -sqrt 133 = 0) :=
by
  sorry

end sum_of_solutions_eq_zero_l457_457521


namespace students_walk_fraction_l457_457745

theorem students_walk_fraction (h1 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/3))
                               (h2 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/5))
                               (h3 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/8))
                               (h4 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/10)) :
  ∃ (students : ℕ), (students - num1 - num2 - num3 - num4) / students = 29 / 120 :=
by
  sorry

end students_walk_fraction_l457_457745


namespace transform_sine_function_l457_457478

theorem transform_sine_function :
  ∀ x, (λ x, 3 * Math.sin (x - Real.pi / 5)) (x / 2) = 3 * Math.sin (x / 2 - Real.pi / 5) :=
by
  intros x
  sorry

end transform_sine_function_l457_457478


namespace smallest_palindrome_proof_l457_457042

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  (100 ≤ n ∧ n ≤ 999) ∧ is_palindrome n

def smallest_non_five_digit_palindrome_product_with_103 : ℕ :=
  404

theorem smallest_palindrome_proof :
  is_three_digit_palindrome smallest_non_five_digit_palindrome_product_with_103 ∧ 
  ¬is_palindrome (103 * smallest_non_five_digit_palindrome_product_with_103) ∧ 
  (∀ n, is_three_digit_palindrome n → ¬is_palindrome (103 * n) → n ≥ 404) :=
begin
  sorry
end

end smallest_palindrome_proof_l457_457042


namespace lambda_mu_condition_l457_457109

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem lambda_mu_condition (λ μ : ℝ) :
  orthogonal (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
             (vector_a.1 + μ * vector_b.1, vector_a.2 + μ * vector_b.2) →
  λ * μ = -1 :=
by
  sorry

end lambda_mu_condition_l457_457109


namespace quadratic_equation_solution_l457_457499

theorem quadratic_equation_solution (m : ℤ) (h : (m-2) * x^(m^2 - 2) + x = 0): m = -2 :=
by
  sorry

end quadratic_equation_solution_l457_457499


namespace ellipse_minor_axis_length_l457_457068

theorem ellipse_minor_axis_length (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ P : ℝ × ℝ, (P.1 ^ 2) / (a ^ 2) + (P.2 ^ 2) / (b ^ 2) = 1 → 
  (∃ (F1 F2 : ℝ × ℝ) (angle_P : ℝ) (area_P : ℝ), 
  angle_P = 120 ∧ area_P = √3 ∧ (
  angle_P = 120 → (|F1.1 - P.1 + F1.2 - P.2| * |F2.1 - P.1 + F2.2 - P.2| = 4) ∧ 
  (∀ P, |F1.1 - P.1 + F1.2 - P.2| + |F2.1 - P.1 + F2.2 - P.2| = 2 * a) ∧ 
  4 * ((F2.1 - F1.1) ^ 2 + (F2.2 - F1.2) ^ 2) = 4 * a ^ 2 - 4))) : 
  2 * b = 2
:= by
  sorry

end ellipse_minor_axis_length_l457_457068


namespace circumscribed_circle_radius_l457_457709

noncomputable def radius_of_circumcircle (a b c : ℚ) (h_a : a = 15/2) (h_b : b = 10) (h_c : c = 25/2) : ℚ :=
if h_triangle : a^2 + b^2 = c^2 then (c / 2) else 0

theorem circumscribed_circle_radius :
  radius_of_circumcircle (15/2 : ℚ) 10 (25/2 : ℚ) (by norm_num) (by norm_num) (by norm_num) = 25 / 4 := 
by
  sorry

end circumscribed_circle_radius_l457_457709


namespace condition_suff_not_necess_l457_457949

theorem condition_suff_not_necess (x : ℝ) (h : |x - (1 / 2)| < 1 / 2) : x^3 < 1 :=
by
  have h1 : 0 < x := sorry
  have h2 : x < 1 := sorry
  sorry

end condition_suff_not_necess_l457_457949


namespace no_six_points_with_pairwise_distance_greater_than_one_l457_457600

theorem no_six_points_with_pairwise_distance_greater_than_one
    (r : ℝ) (hc : r = 1) :
    ¬(∃ pts : Fin 6 → ℝ × ℝ,
        (∀ (i j : Fin 6), i ≠ j → real.dist (pts i) (pts j) > 1) ∧ 
        (∀ (i : Fin 6), real.sqrt ((pts i).1 * (pts i).1 + (pts i).2 * (pts i).2) ≤ r)) := by
    sorry

end no_six_points_with_pairwise_distance_greater_than_one_l457_457600


namespace who_hits_region_8_l457_457608

theorem who_hits_region_8 :
  ∃ n, (n = 8 ∧ Diana_hits n) ∧
  (Alex_scores 18 ∧ Betsy_scores 5 ∧ Carlos_scores 12 ∧ Diana_scores 14 ∧ Edward_scores 19 ∧ Fiona_scores 11) ∧
  (unique_pair Alex ∧ unique_pair Betsy ∧ unique_pair Carlos ∧ unique_pair Diana ∧ unique_pair Edward ∧ unique_pair Fiona) :=
sorry

end who_hits_region_8_l457_457608


namespace divisors_not_multiples_of_7_l457_457236

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = m

def is_perfect_seventh (m : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 7 = m

theorem divisors_not_multiples_of_7 (n : ℕ)
  (h1 : is_perfect_square (n / 2))
  (h2 : is_perfect_cube (n / 3))
  (h3 : is_perfect_seventh (n / 7)) :
  (finset.filter (λ d, ¬ (7 ∣ d)) (nat.divisors n)).card = 77 :=
sorry

end divisors_not_multiples_of_7_l457_457236


namespace correct_statements_l457_457507

section
  -- Define the population
  variable (students_population : Set ℕ)
  variable (sampled_students : Set ℕ)
  
  -- Conditions
  def population_size : Prop := students_population.card = 70000
  def sampled_size : Prop := sampled_students.card = 1000
  def sample_is_subset : Prop := sampled_students ⊆ students_population
  def each_student_is_individual : ∀ x, x ∈ students_population → x ∈ students_population

  -- Define the statements Identification
  def statements :=
    (sample_is_subset sampled_students students_population) ∧
    (each_student_is_individual students_population) ∧
    (population_size students_population) ∧
    ∃ (math_scores : students_population → ℕ), True

  theorem correct_statements : statements :=
  by
    sorry
end

end correct_statements_l457_457507


namespace problem_solution_l457_457434

theorem problem_solution
  (k : ℝ)
  (y : ℝ → ℝ)
  (quadratic_fn : ∀ x, y x = (k + 2) * x^(k^2 + k - 4))
  (increase_for_neg_x : ∀ x : ℝ, x < 0 → y (x + 1) > y x) :
  k = -3 ∧ (∀ m n : ℝ, -2 ≤ m ∧ m ≤ 1 → y m = n → -4 ≤ n ∧ n ≤ 0) := 
sorry

end problem_solution_l457_457434


namespace minimum_omega_l457_457570

noncomputable def f (omega phi x : ℝ) : ℝ := Real.sin (omega * x + phi)

theorem minimum_omega {omega : ℝ} (h_pos : omega > 0) (h_even : ∀ x : ℝ, f omega (Real.pi / 2) x = f omega (Real.pi / 2) (-x)) 
  (h_zero_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f omega (Real.pi / 2) x = 0) :
  omega ≥ 1 / 2 :=
sorry

end minimum_omega_l457_457570


namespace reggie_free_throws_l457_457178

namespace BasketballShootingContest

-- Define the number of points for different shots
def points (layups free_throws long_shots : ℕ) : ℕ :=
  1 * layups + 2 * free_throws + 3 * long_shots

-- Conditions given in the problem
def Reggie_points (F: ℕ) : ℕ := 
  points 3 F 1

def Brother_points : ℕ := 
  points 0 0 4

-- The given condition that Reggie loses by 2 points
theorem reggie_free_throws:
  ∃ F : ℕ, Reggie_points F + 2 = Brother_points :=
sorry

end BasketballShootingContest

end reggie_free_throws_l457_457178


namespace remaining_amount_to_be_paid_l457_457342

theorem remaining_amount_to_be_paid (part_payment : ℝ) (percentage : ℝ) (h : part_payment = 650 ∧ percentage = 0.15) :
    (part_payment / percentage - part_payment) = 3683.33 := by
  cases h with
  | intro h1 h2 =>
    sorry

end remaining_amount_to_be_paid_l457_457342


namespace no_integer_parallelogram_with_given_diagonals_and_angle_l457_457601

theorem no_integer_parallelogram_with_given_diagonals_and_angle 
    (A B C D O : ℤ × ℤ)
    (h1 : A ≠ B)
    (h2 : B ≠ C)
    (h3 : C ≠ D)
    (h4 : D ≠ A)
    (par : (A.1 + C.1 = B.1 + D.1) ∧ (A.2 + C.2 = B.2 + D.2))
    (diag1_twice_diag2 : dist A C = 2 * dist B D)
    (angle_45 : ∃ θ, θ = 45 ∧ θ = angle_deg (A, C) (B, D)) :
    false :=
by
  sorry

-- Definitions for distancing and angle might be placeholders. A more precise definition might be required.
-- dist and angle_deg functions are placeholders for distance and angle calculations respectively.

end no_integer_parallelogram_with_given_diagonals_and_angle_l457_457601


namespace signUpMethodCount_nonZeroDenominationCount_l457_457699

-- part 1: Sign-up methods for students
def numSignUpMethods (n : ℕ) (choices : ℕ) : ℕ :=
  choices ^ n

theorem signUpMethodCount (students choices : ℕ) (h1 : students = 5) (h2 : choices = 2) :
  numSignUpMethods students choices = 32 :=
by {
  -- Simplify and use the given conditions
  rw [h1, h2],
  exact pow_succ 2 4,   -- Simplifying 2^5 to 32
  sorry
}

-- part 2: Different non-zero denominations
def numNonZeroDenominations (coins : ℕ) : ℕ :=
  2^coins - 1

theorem nonZeroDenominationCount (coins : ℕ) (h : coins = 7) :
  numNonZeroDenominations coins = 127 :=
by {
  -- Simplify and use the given condition
  rw h,
  rw [pow_succ, pow_succ, pow_succ, pow_succ, pow_succ, pow_succ, pow_zero], -- Simplifying 2^7 to 128 - 1
  sorry
}

end signUpMethodCount_nonZeroDenominationCount_l457_457699


namespace cost_of_greenhouses_possible_renovation_plans_l457_457372

noncomputable def cost_renovation (x y : ℕ) : Prop :=
  (2 * x = y + 6) ∧ (x + 2 * y = 48)

theorem cost_of_greenhouses : ∃ x y, cost_renovation x y ∧ x = 12 ∧ y = 18 :=
by {
  sorry
}

noncomputable def renovation_plan (m : ℕ) : Prop :=
  (5 * m + 3 * (8 - m) ≤ 35) ∧ (12 * m + 18 * (8 - m) ≤ 128)

theorem possible_renovation_plans : ∃ m, renovation_plan m ∧ (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  sorry
}

end cost_of_greenhouses_possible_renovation_plans_l457_457372


namespace no_real_solution_equation_l457_457275

theorem no_real_solution_equation (x : ℝ) (h : x ≠ -9) : 
  ¬ ∃ x, (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 :=
by
  sorry

end no_real_solution_equation_l457_457275


namespace montoya_food_budget_l457_457282

theorem montoya_food_budget (g t e : ℝ) (h1 : g = 0.6) (h2 : t = 0.8) : e = 0.2 :=
by sorry

end montoya_food_budget_l457_457282


namespace annual_oil_change_cost_l457_457914

/-!
# Problem Statement
John drives 1000 miles a month. An oil change is needed every 3000 miles.
John gets 1 free oil change a year. Each oil change costs $50.

Prove that the total amount John pays for oil changes in a year is $150.
-/

def miles_driven_per_month : ℕ := 1000
def miles_per_oil_change : ℕ := 3000
def free_oil_changes_per_year : ℕ := 1
def oil_change_cost : ℕ := 50

theorem annual_oil_change_cost : 
  let total_oil_changes := (12 * miles_driven_per_month) / miles_per_oil_change,
      paid_oil_changes := total_oil_changes - free_oil_changes_per_year
  in paid_oil_changes * oil_change_cost = 150 :=
by {
  -- The proof is not required, so we use sorry
  sorry 
}

end annual_oil_change_cost_l457_457914


namespace mateo_net_salary_l457_457574

-- Definitions and assumptions based on the given conditions
def weekly_salary : ℝ := 791
def days_absent : ℕ := 4
def tax_rate : ℝ := 0.07

def first_day_deduction : ℝ := 0.01 * weekly_salary
def second_day_deduction : ℝ := 0.02 * weekly_salary
def third_day_deduction : ℝ := 0.03 * weekly_salary
def fourth_day_deduction : ℝ := 0.04 * weekly_salary
def total_deduction : ℝ := first_day_deduction + second_day_deduction + third_day_deduction + fourth_day_deduction

def salary_after_absence_deductions : ℝ := weekly_salary - total_deduction
def income_tax : ℝ := tax_rate * salary_after_absence_deductions
def net_salary : ℝ := salary_after_absence_deductions - income_tax

-- The proof statement
theorem mateo_net_salary : net_salary ≈ 662.07 :=
by
  -- Proof would go here
  sorry

end mateo_net_salary_l457_457574


namespace coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l457_457672

theorem coefficient_of_x9_in_expansion_of_x_minus_2_pow_10 :
  ∃ c : ℤ, (x - 2)^10 = ∑ k in finset.range (11), (nat.choose 10 k) * x^k * (-2)^(10 - k) ∧ c = -20 := 
begin 
  use -20,
  { sorry },
end

end coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l457_457672


namespace path_length_l457_457713

-- Definition of the problem conditions
def cube_edge_length : ℝ := 1
def dot_center_top_face (cube_edge_length : ℝ) : Prop := true -- Placeholder to denote the dot is at the center of the top face
def rolling_cube_no_lifting (cube_edge_length : ℝ) : Prop := true -- Placeholder to denote the cube rolls without lifting or slipping
def cube_initial (cube_edge_length : ℝ) : Prop := true -- Placeholder to denote the initial state of the cube

-- The theorem stating the path length of the rolling dot
theorem path_length (cube_edge_length : ℝ)
  (h1 : cube_edge_length = 1)
  (h2 : dot_center_top_face cube_edge_length)
  (h3 : rolling_cube_no_lifting cube_edge_length)
  (h4 : cube_initial cube_edge_length)
  : ∃ (d : ℝ), d = (1 + Real.sqrt 5) / 2 * Real.pi := 
begin
  sorry -- Proof to be provided
end

end path_length_l457_457713


namespace population_decreasing_l457_457698

variable (P_0 : ℝ) (k : ℝ)

theorem population_decreasing (P_0_pos : P_0 > 0) (k_neg : -1 < k ∧ k < 0) (n : ℕ) :
  let P_n := P_0 * (1 + k)^n in
  let P_n1 := P_0 * (1 + k)^(n+1) in
  P_n1 < P_n :=
by
  let P_n := P_0 * (1 + k)^n
  let P_n1 := P_0 * (1 + k)^(n+1)
  have calc_diff : P_n1 - P_n = P_0 * (1 + k)^n * k := by
    sorry
  have h : P_0 * (1 + k)^n > 0 := by
    sorry
  show P_n1 < P_n, from
    sorry

end population_decreasing_l457_457698


namespace cards_per_layer_correct_l457_457250

-- Definitions based on the problem's conditions
def num_decks : ℕ := 16
def cards_per_deck : ℕ := 52
def num_layers : ℕ := 32

-- The key calculation we need to prove
def total_cards : ℕ := num_decks * cards_per_deck
def cards_per_layer : ℕ := total_cards / num_layers

theorem cards_per_layer_correct : cards_per_layer = 26 := by
  unfold cards_per_layer total_cards num_decks cards_per_deck num_layers
  simp
  sorry

end cards_per_layer_correct_l457_457250


namespace point_in_fourth_quadrant_l457_457526

-- Define a structure for a Cartesian point
structure Point where
  x : ℝ
  y : ℝ

-- Define different quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Function to determine the quadrant of a given point
def quadrant (p : Point) : Quadrant :=
  if p.x > 0 ∧ p.y > 0 then Quadrant.first
  else if p.x < 0 ∧ p.y > 0 then Quadrant.second
  else if p.x < 0 ∧ p.y < 0 then Quadrant.third
  else Quadrant.fourth

-- The main theorem stating the point (3, -4) lies in the fourth quadrant
theorem point_in_fourth_quadrant : quadrant { x := 3, y := -4 } = Quadrant.fourth :=
  sorry

end point_in_fourth_quadrant_l457_457526


namespace zeros_in_expansion_of_999_cubed_l457_457379

theorem zeros_in_expansion_of_999_cubed :
  ∃ n : ℤ, (999^3).digits.count 0 = 6 :=
by {
  sorry
}

end zeros_in_expansion_of_999_cubed_l457_457379


namespace multiply_and_simplify_l457_457971
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l457_457971


namespace sin_double_angle_l457_457805

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (Real.pi / 4 - α) = -3 / 5) :
  Real.sin (2 * α) = -7 / 25 := by
sorry

end sin_double_angle_l457_457805


namespace sum_f_2017_l457_457169

noncomputable def f : ℕ+ → ℝ
| ⟨n, _⟩ := Real.tan ((n : ℕ) * Real.pi / 3)

theorem sum_f_2017 : (Finset.range 2017).sum (λ n, f ⟨n + 1, (nat.succ_pos _).trans_le (nat.le_succ _)⟩) = Real.sqrt 3 := 
by
  sorry

end sum_f_2017_l457_457169


namespace find_f_ff1_l457_457097

def piecewise_function (x : ℝ) : ℝ :=
  if x >= 3 then x^2 - 2*x else 2*x + 1

theorem find_f_ff1 : piecewise_function (piecewise_function 1) = 3 := by
  sorry

end find_f_ff1_l457_457097


namespace tan_A_in_triangle_ABC_l457_457175

variable {A B C a b c : ℝ}
variable {triangleABC : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2}

theorem tan_A_in_triangle_ABC
  (h1 : a / b = (b + sqrt 3 * c) / a)
  (h2 : sin C = 2 * sqrt 3 * sin B)
  (triangleABC : ∀ (A B C : ℝ), (A + B + C = π) ∧ (A > 0) ∧ (B > 0) ∧ (C > 0))
  (opp_sides : ∀ (A B C : ℝ), (a = opp A) ∧ (b = opp B) ∧ (c = opp C)) :
  tan A = sqrt 3 / 3 :=
by
  sorry

end tan_A_in_triangle_ABC_l457_457175


namespace perpendicular_vectors_l457_457113

variable (λ μ : ℝ)
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

theorem perpendicular_vectors : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0 → λ * μ = -1 := by
  sorry

end perpendicular_vectors_l457_457113


namespace total_volume_of_five_boxes_l457_457679

-- Define the edge length of each cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_cube (s : ℕ) : ℕ := s ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 5

-- Define the total volume
def total_volume (s : ℕ) (n : ℕ) : ℕ := n * (volume_of_cube s)

-- The theorem to prove
theorem total_volume_of_five_boxes :
  total_volume edge_length number_of_cubes = 625 := 
by
  -- Proof is skipped
  sorry

end total_volume_of_five_boxes_l457_457679


namespace anna_current_age_l457_457378

theorem anna_current_age (A : ℕ) (Clara_now : ℕ) (years_ago : ℕ) (Clara_age_ago : ℕ) 
    (H1 : Clara_now = 80) 
    (H2 : years_ago = 41) 
    (H3 : Clara_age_ago = Clara_now - years_ago) 
    (H4 : Clara_age_ago = 3 * (A - years_ago)) : 
    A = 54 :=
by
  sorry

end anna_current_age_l457_457378


namespace minimum_positive_period_and_monotonicity_of_f_l457_457425

def f (x : ℝ) : ℝ := cos (π/2 - x) * cos x + sqrt 3 * sin x ^ 2

theorem minimum_positive_period_and_monotonicity_of_f :
  (∀ x, f (x + π) = f x) ∧ (∀ k : ℤ, ∀ x : ℝ, k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12 → ∀ y, f y ≤ f x) :=
by
  sorry

end minimum_positive_period_and_monotonicity_of_f_l457_457425


namespace perpendicular_dot_product_l457_457127

theorem perpendicular_dot_product (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → (λ * μ = -1) :=
by
  intros
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  sorry

end perpendicular_dot_product_l457_457127


namespace seq_inequality_1_seq_sum_bound_l457_457566

-- Define the exponential and cosine function
def f (x : ℝ) : ℝ := Real.exp x - Real.cos x

-- Define the sequence a_n based on the given conditions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, f (a n) = a (n - 1)

-- First proof goal: For \( n \geq 2 \), \( a_{n-1} > a_n + a_n^2 \)
theorem seq_inequality_1 (a : ℕ → ℝ) (h : seq a) : ∀ n ≥ 2, a (n - 1) > a n + a n ^ 2 :=
by
  sorry

-- Second proof goal: \( \sum_{k=1}^n a_k < 2 \sqrt{n} \)
theorem seq_sum_bound (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, ∑ k in Finset.range n, a k < 2 * Real.sqrt n :=
by
  sorry

end seq_inequality_1_seq_sum_bound_l457_457566


namespace least_m_subsets_powers_of_two_l457_457928

open Set

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem least_m_subsets_powers_of_two :
  let X := {1, 2, ..., 2001}
  let m := 999 in
  ∀ (W : Finset ℕ), (W ⊆ X) → (W.card = m) → ∃ (u v : ℕ) (hu : u ∈ W) (hv : v ∈ W), is_power_of_two (u + v) :=
begin
  sorry
end

end least_m_subsets_powers_of_two_l457_457928


namespace perpendicular_vectors_condition_l457_457125

theorem perpendicular_vectors_condition (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
      b : ℝ × ℝ := (1, -1)
  in ((a.1 + λ * b.1) * (a.1 + μ * b.1) + (a.2 + λ * b.2) * (a.2 + μ * b.2) = 0) → (λ * μ = -1) :=
sorry

end perpendicular_vectors_condition_l457_457125


namespace largest_prime_factor_5040_l457_457321

theorem largest_prime_factor_5040 : ∃ p : ℕ, nat.prime p ∧ p ∣ 5040 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 5040 → q ≤ p :=
by
  -- We assert the statement that we need to prove
  use 7
  -- We need to provide proofs of each of these conditions:
  -- 1. 7 is prime
  -- 2. 7 divides 5040
  -- 3. Any prime that divides 5040 is less than or equal to 7
  repeat { sorry }

end largest_prime_factor_5040_l457_457321


namespace exp_rectangular_form_l457_457021

theorem exp_rectangular_form : (complex.exp (13 * real.pi * complex.I / 2)) = complex.I :=
by
  sorry

end exp_rectangular_form_l457_457021


namespace max_area_of_triangle_l457_457775

noncomputable def maxAreaTriangle (m_a m_b m_c : ℝ) : ℝ :=
  1/3 * Real.sqrt (2 * (m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4))

theorem max_area_of_triangle (m_a m_b m_c : ℝ) (h1 : m_a ≤ 2) (h2 : m_b ≤ 3) (h3 : m_c ≤ 4) :
  maxAreaTriangle m_a m_b m_c ≤ 4 :=
sorry

end max_area_of_triangle_l457_457775


namespace find_t_range_l457_457069

noncomputable def ellipse_equation : String :=
  have a := 2
  have b := 1
  show String, from "x^2/4 + y^2 = 1"

theorem find_t_range (k : ℝ) (h : k^2 < 5/12) : 
  ∃ t : ℝ, (2 < t ∧ t < 3) ∨ (1/3 < t ∧ t < 1/2) :=
sorry

end find_t_range_l457_457069


namespace mp_mq_squared_l457_457209

open Real

-- Define points P and Q based on the given lines
def P {a : ℝ} : ℝ × ℝ :=
  (0, 1)

def Q {a : ℝ} : ℝ × ℝ :=
  (-3, 0)

-- Define the lines l and m
def line_l {a : ℝ} (x y : ℝ) : Prop :=
  a * x + y - 1 = 0

def line_m {a : ℝ} (x y : ℝ) : Prop :=
  x - a * y + 3 = 0

-- Prove the expected value of MP^2 + MQ^2
theorem mp_mq_squared {a : ℝ} (M P Q : ℝ × ℝ)
  (hP : line_l P.1 P.2)
  (hQ : line_m Q.1 Q.2)
  (hM : ∃ (x y : ℝ), line_l x y ∧ line_m x y)
  (hP_coord : P = (0, 1))
  (hQ_coord : Q = (-3, 0)) :
  (dist M P) ^ 2 + (dist M Q) ^ 2 = 10 := by
  sorry

end mp_mq_squared_l457_457209


namespace points_on_parabola_l457_457010

-- Definitions identifying each condition in the problem
def point (t : ℝ) : ℝ × ℝ := (3^t - 2, 9^t - 7 * 3^t + 4)

-- The theorem stating that all points lie on a parabola
theorem points_on_parabola : ∀ t : ℝ, ∃ a b c : ℝ, (let (x, y) := point t in y = a * x^2 + b * x + c) :=
by
  sorry

end points_on_parabola_l457_457010


namespace max_stamps_value_l457_457963

theorem max_stamps_value (n : ℕ) 
  (h : ∀ (a : Fin 10 → ℕ), (∀ i j : Fin 10, i < j → a i < a j) → (∃ (s : Finset (Fin 10)), s.card = 4 ∧ ∑ i in s, a i ≥ n / 2)) : 
  n ≤ 135 :=
sorry

end max_stamps_value_l457_457963


namespace perpendicular_vectors_l457_457116

variable (λ μ : ℝ)
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

theorem perpendicular_vectors : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0 → λ * μ = -1 := by
  sorry

end perpendicular_vectors_l457_457116


namespace count_two_digit_integers_remainder_3_div_9_l457_457162

theorem count_two_digit_integers_remainder_3_div_9 :
  ∃ (N : ℕ), N = 10 ∧ ∀ k, (10 ≤ 9 * k + 3 ∧ 9 * k + 3 < 100) ↔ (1 ≤ k ∧ k ≤ 10) :=
by
  sorry

end count_two_digit_integers_remainder_3_div_9_l457_457162


namespace cards_per_layer_l457_457251

theorem cards_per_layer (total_decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) (h_decks : total_decks = 16) (h_cards_per_deck : cards_per_deck = 52) (h_layers : layers = 32) :
  total_decks * cards_per_deck / layers = 26 :=
by {
  -- To skip the proof
  sorry
}

end cards_per_layer_l457_457251


namespace mass_percentage_of_I_in_AlI3_l457_457784

variables (M_Al M_I : ℝ) (AlI3_molar_mass : ℝ)
variables (total_I_mass : ℝ)

-- Define conditions
def condition1 := M_Al = 26.98
def condition2 := M_I = 126.90
def condition3 := AlI3_molar_mass = M_Al + 3 * M_I
def condition4 := total_I_mass = 3 * M_I

-- Define mass percentage formula
def mass_percentage_of_I := (total_I_mass / AlI3_molar_mass) * 100

-- Proof Statement
theorem mass_percentage_of_I_in_AlI3 (h1 : condition1) (h2 : condition2) 
(h3 : condition3) (h4 : condition4) :
mass_percentage_of_I M_Al M_I AlI3_molar_mass total_I_mass = 93.38 :=
sorry

end mass_percentage_of_I_in_AlI3_l457_457784


namespace angle_XWZ_l457_457193

theorem angle_XWZ (XYZ_line : XYZ.is_line) 
  (angle_XWY : ∠XWY = 35) (angle_YWZ : ∠YWZ = 75) (angle_XZW : ∠XZW = 55) : 
  ∠XWZ = 15 :=
sorry

end angle_XWZ_l457_457193


namespace a_pow_11_b_pow_11_l457_457581

theorem a_pow_11_b_pow_11 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end a_pow_11_b_pow_11_l457_457581


namespace num_students_in_line_l457_457613

theorem num_students_in_line:
  ∀ (Taehyung Namjoon : ℕ), 
    Taehyung = 1 → 
    ∃ t n m : ℕ, 
      t = Taehyung ∧ 
      n = 3 + 1 + t ∧ 
      m = 8 + n →
      m = 13 :=
by 
  intros Taehyung Namjoon h1,
  use [1, 5, 13],
  exact ⟨rfl, rfl, rfl⟩

end num_students_in_line_l457_457613


namespace fish_size_difference_l457_457626

variables {S J W : ℝ}

theorem fish_size_difference (h1 : S = J + 21.52) (h2 : J = W - 12.64) : S - W = 8.88 :=
sorry

end fish_size_difference_l457_457626


namespace variance_decreases_l457_457279

def variance (l : List ℝ) : ℝ :=
  let mean := l.sum / l.length
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

theorem variance_decreases :
  let scores := [5, 9, 7, 10, 9]
  let new_score := 8
  variance scores = 3.2 →
  variance (new_score :: scores) < variance scores :=
by
  sorry

end variance_decreases_l457_457279


namespace scale_of_model_is_correct_model_width_is_correct_actual_height_is_correct_l457_457367

-- Define the given conditions
def actual_length_building_m : ℝ := 100  -- in meters
def model_length_building_dm : ℝ := 5    -- in decimeters
def width_building_m : ℝ := 12           -- in meters
def model_height_dm : ℝ := 1             -- in decimeters

-- Conversion factors
def meters_to_decimeters (m : ℝ) : ℝ := m * 10

-- The expected results
def expected_scale : ℝ := 1 / 200
def expected_model_width_dm : ℝ := 0.6
def expected_actual_height_m : ℝ := 20

-- Theorems to be proved
theorem scale_of_model_is_correct : model_length_building_dm / meters_to_decimeters actual_length_building_m = expected_scale :=
sorry

theorem model_width_is_correct : (meters_to_decimeters width_building_m) * expected_scale = expected_model_width_dm :=
sorry

theorem actual_height_is_correct : model_height_dm / expected_scale / 10 = expected_actual_height_m :=
sorry

end scale_of_model_is_correct_model_width_is_correct_actual_height_is_correct_l457_457367


namespace correct_calculation_l457_457684

theorem correct_calculation (n : ℕ) (h : n - 59 = 43) : n - 46 = 56 :=
by
  sorry

end correct_calculation_l457_457684


namespace graph_passes_through_fixed_point_l457_457630

theorem graph_passes_through_fixed_point (a : ℝ) : (0, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a ^ x + 1) } :=
sorry

end graph_passes_through_fixed_point_l457_457630


namespace find_probability_of_B_l457_457497

-- Definitions
def mutually_exclusive (A B : Set α) :=
  ∀ ω, ω ∈ A → ω ∉ B

def Probability (A : Set α) : ℝ := sorry

variable {α : Type} [MeasureSpace α]

-- Conditions
variable {A B : Set α}
variable (H1 : mutually_exclusive A B)
variable (H2 : Probability A = 0.4)

-- Theorem stating the problem
theorem find_probability_of_B : Probability B = 0.6 :=
by sorry

end find_probability_of_B_l457_457497


namespace sin_pi_minus_a_eq_three_fifths_l457_457803

open Real

noncomputable def sin_of_pi_minus_a (a : ℝ) : ℝ :=
sin (π - a)

theorem sin_pi_minus_a_eq_three_fifths (a : ℝ) (h1 : a ∈ Ioo 0 (π / 2)) (h2 : cos a = 4 / 5) :
  sin_of_pi_minus_a a = 3 / 5 :=
by
  sorry

end sin_pi_minus_a_eq_three_fifths_l457_457803


namespace average_speed_of_train_l457_457688

theorem average_speed_of_train (x : ℝ) (hx_pos : 0 < x) :
  let distance1 := x,
      speed1 := 40,
      distance2 := 2 * x,
      speed2 := 20,
      total_distance := 3 * x,
      time1 := distance1 / speed1,
      time2 := distance2 / speed2,
      total_time := time1 + time2,
      average_speed := total_distance / total_time
  in average_speed = 24 := 
by
  have h_distance1 : distance1 = x := rfl,
  have h_speed1 : speed1 = 40 := rfl,
  have h_distance2 : distance2 = 2 * x := rfl,
  have h_speed2 : speed2 = 20 := rfl,
  have h_total_distance : total_distance = 3 * x := rfl,
  have h_time1 : time1 = x / 40 := rfl,
  have h_time2 : time2 = 2 * x / 20 := rfl,
  have h_total_time : total_time = (x / 40) + (2 * x / 20) := rfl,
  calc
    average_speed 
      = total_distance / total_time : rfl
    ... = (3 * x) / ((x / 40) + (2 * x / 20)) : rfl
    ... = (3 * x) / ((x / 40) + (x / 10)) : by rw [← h_time2]
    ... = (3 * x) / ((x / 40) + (4 * x / 40)) : by norm_num
    ... = (3 * x) / ((5 * x) / 40) : by rw add_mul_equiv
    ... = (3 * x) * (40 / (5 * x)) : by field_simp
    ... = 24 : by norm_num

#print average_speed_of_train

end average_speed_of_train_l457_457688


namespace slope_y_intercept_product_eq_neg_five_over_two_l457_457898

theorem slope_y_intercept_product_eq_neg_five_over_two :
  let A := (0, 10)
  let B := (0, 0)
  let C := (10, 0)
  let D := ((0 + 0) / 2, (10 + 0) / 2) -- midpoint of A and B
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope * y_intercept = -5 / 2 := 
by 
  sorry

end slope_y_intercept_product_eq_neg_five_over_two_l457_457898


namespace max_edges_convex_polyhedron_l457_457711

theorem max_edges_convex_polyhedron (n : ℕ) (c l e : ℕ) (h1 : c = n) (h2 : c + l = e + 2) (h3 : 2 * e ≥ 3 * l) : e ≤ 3 * n - 6 := 
sorry

end max_edges_convex_polyhedron_l457_457711


namespace gcd_m_n_gt_one_l457_457280

-- Definitions of the necessary conditions
variables (m n : ℕ)
variables (phi : ℕ → ℕ)

-- Assume that phi is the Euler's totient function 
axiom phi_euler (x : ℕ) : phi x = Nat.totient x

-- Given conditions
axiom cond1 : phi (5^m - 1) = 5^n - 1

-- Goal: to show gcd(m, n) > 1
theorem gcd_m_n_gt_one : Nat.gcd m n > 1 :=
by
  -- Main proof should go here
  sorry

end gcd_m_n_gt_one_l457_457280


namespace hyperbola_asymptotes_ratio_l457_457050

theorem hyperbola_asymptotes_ratio (p q : ℝ) (h : p > q)
(hyperbola_eq : ∀ x y, x^2 / p^2 - y^2 / q^2 = 1)
(angle_condition : ∀ m1 m2, tan (real.pi / 4) = abs (m1 - m2) / (1 + m1 * m2) → m1 = q / p → m2 = - q / p) :
  p / q = real.sqrt 2 - 1 :=
sorry

end hyperbola_asymptotes_ratio_l457_457050


namespace total_digits_in_arabic_numerals_l457_457751

theorem total_digits_in_arabic_numerals (pages : ℕ) (special_pages : ℕ) : 
  pages = 10000 → special_pages = 200 → 
  (∑ n in (finset.Icc 201 999), nat.digits 10 n).length + 
  (∑ n in (finset.Icc 1000 9999), nat.digits 10 n).length + 
  (∑ n in (finset.Icc 10000 10000), nat.digits 10 n).length 
  = 38402 := 
by 
  intro pages_eq special_pages_eq 
  rw [pages_eq, special_pages_eq]
  
  sorry

end total_digits_in_arabic_numerals_l457_457751


namespace perpendicular_vectors_condition_l457_457123

theorem perpendicular_vectors_condition (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
      b : ℝ × ℝ := (1, -1)
  in ((a.1 + λ * b.1) * (a.1 + μ * b.1) + (a.2 + λ * b.2) * (a.2 + μ * b.2) = 0) → (λ * μ = -1) :=
sorry

end perpendicular_vectors_condition_l457_457123


namespace incorrect_statement_D_l457_457795

theorem incorrect_statement_D (x1 x2 : ℝ) (hx : x1 < x2) :
  ¬ (y : ℝ) (y1 := -3 / x1) (y2 := -3 / x2) y1 < y2 :=
by
  sorry

end incorrect_statement_D_l457_457795


namespace solve_for_b_l457_457876

/-- Ensure that the defined complex number meets the condition that its real part equals its imaginary part -/
def complex_expr (b : ℝ) : ℂ :=
  (1 + Complex.i) / (1 - Complex.i) + (1 / 2) * b

theorem solve_for_b (b : ℝ) (h : (complex_expr b).re = (complex_expr b).im) : b = 2 :=
sorry

end solve_for_b_l457_457876


namespace ineq_medians_triangle_l457_457441

theorem ineq_medians_triangle (a b c s_a s_b s_c : ℝ)
  (h_mediana : s_a = 1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2))
  (h_medianb : s_b = 1 / 2 * Real.sqrt (2 * a^2 + 2 * c^2 - b^2))
  (h_medianc : s_c = 1 / 2 * Real.sqrt (2 * a^2 + 2 * b^2 - c^2))
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c > s_a + s_b + s_c ∧ s_a + s_b + s_c > (3 / 4) * (a + b + c) := 
sorry

end ineq_medians_triangle_l457_457441


namespace count_valid_triangles_l457_457855

def triangle_area (a b c : ℕ) : ℕ :=
  let s := (a + b + c) / 2
  s * (s - a) * (s - b) * (s - c)

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c < 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2

theorem count_valid_triangles : { n : ℕ // n = 24 } :=
  sorry

end count_valid_triangles_l457_457855


namespace roque_bike_time_l457_457551

-- Definitions of conditions
def roque_walk_time_per_trip : ℕ := 2
def roque_walk_trips_per_week : ℕ := 3
def roque_bike_trips_per_week : ℕ := 2
def total_commuting_time_per_week : ℕ := 16

-- Statement of the problem to prove
theorem roque_bike_time (B : ℕ) :
  (roque_walk_time_per_trip * 2 * roque_walk_trips_per_week + roque_bike_trips_per_week * 2 * B = total_commuting_time_per_week) → 
  B = 1 :=
by
  sorry

end roque_bike_time_l457_457551


namespace sequences_converge_l457_457809

-- Given definitions
variables {a b x_0 y_0 λ μ : ℝ}
variable (P0 : ℝ × ℝ)
variables (a_ne_zero : a ≠ 0) (lambda_mu_lt_one : 0 < λ * μ ∧ λ * μ < 1)

-- Sequences
def y_sequence : ℕ → ℝ
| 0       := y_0
| (n + 1) := λ * μ * y_sequence n

def x_sequence : ℕ → ℝ
| 0       := x_0
| (n + 1) := x_sequence n - (b/a) * λ * y_sequence n * (1 + μ)

-- Limits to prove
def x_limit : ℝ := x_0 - (b/a) * y_0 * λ * (1 + μ) / (1 - λ * μ)
def y_limit : ℝ := 0

-- The theorem to be proved
theorem sequences_converge :
  (y_sequence y_0 λ μ → 0) ∧ 
  (x_sequence x_0 y_0 a b λ μ → x_limit) :=
sorry

end sequences_converge_l457_457809


namespace ways_to_weigh_9_grams_l457_457578

theorem ways_to_weigh_9_grams :
  let weights := [(1, 3), (2, 3), (5, 1)],
      total_weight := 9 in
  num_combinations weights total_weight = 8 := 
sorry

end ways_to_weigh_9_grams_l457_457578


namespace production_rate_is_constant_l457_457253

def drum_rate := 6 -- drums per day

def days_needed_to_produce (n : ℕ) : ℕ := n / drum_rate

theorem production_rate_is_constant (n : ℕ) : days_needed_to_produce n = n / drum_rate :=
by
  sorry

end production_rate_is_constant_l457_457253


namespace sum_of_roots_l457_457389

theorem sum_of_roots (x : ℝ) :
  (3 * x - 2) * (x - 3) + (3 * x - 2) * (2 * x - 8) = 0 ->
  x = 2 / 3 ∨ x = 11 / 3 ->
  (2 / 3) + (11 / 3) = 13 / 3 :=
by
  sorry

end sum_of_roots_l457_457389


namespace find_p_q_r_l457_457639

-- Definitions of the problem conditions
def points_on_sphere (A B C O : ℝ^3) (radius : ℝ) : Prop :=
  dist O A = radius ∧ dist O B = radius ∧ dist O C = radius

def given_distances (A B C : ℝ^3) : Prop :=
  dist A B = 17 ∧ dist B C = 18 ∧ dist C A = 19

def distance_formula (O : ℝ^3) (ABC_plane : ℝ) (p q r : ℕ) : Prop :=
  ABC_plane = (135 * (sqrt 110)) / 78

-- The main theorem statement
theorem find_p_q_r (A B C O : ℝ^3) (radius : ℝ) (ABC_plane : ℝ) (p q r : ℕ) :
  points_on_sphere A B C O radius →
  given_distances A B C →
  distance_formula O ABC_plane p q r →
  radius = 25 →
  p + q + r = 323 :=
by
  intros h1 h2 h3 h4
  sorry

end find_p_q_r_l457_457639


namespace pirate_rick_digging_time_l457_457590

theorem pirate_rick_digging_time :
  ∀ (initial_depth rate: ℕ) (storm_factor tsunami_added: ℕ),
  initial_depth = 8 →
  rate = 2 →
  storm_factor = 2 →
  tsunami_added = 2 →
  (initial_depth / storm_factor + tsunami_added) / rate = 3 := 
by
  intros
  sorry

end pirate_rick_digging_time_l457_457590


namespace replace_90_percent_in_3_days_cannot_replace_all_banknotes_l457_457544

-- Define constants and conditions
def total_old_banknotes : ℕ := 3628800
def daily_cost : ℕ := 90000
def major_repair_cost : ℕ := 700000
def max_daily_print_after_repair : ℕ := 1000000
def budget_limit : ℕ := 1000000

-- Define the day's print capability function (before repair)
def daily_print (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  if num_days = 1 then banknotes_remaining / 2
  else (banknotes_remaining / (num_days + 1))

-- Define the budget calculation before repair
def print_costs (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  daily_cost * num_days

-- Lean theorem to be stated proving that 90% of the banknotes can be replaced within 3 days
theorem replace_90_percent_in_3_days :
  ∃ (days : ℕ) (banknotes_replaced : ℕ), days = 3 ∧ banknotes_replaced = 3265920 ∧ print_costs days total_old_banknotes ≤ budget_limit :=
sorry

-- Lean theorem to be stated proving that not all banknotes can be replaced within the given budget
theorem cannot_replace_all_banknotes :
  ∀ banknotes_replaced cost : ℕ,
  banknotes_replaced < total_old_banknotes ∧ cost ≤ budget_limit →
  banknotes_replaced + (total_old_banknotes / (4 + 1)) < total_old_banknotes :=
sorry

end replace_90_percent_in_3_days_cannot_replace_all_banknotes_l457_457544


namespace sum_sum_sum_sum_eq_one_l457_457559

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Mathematical problem statement
theorem sum_sum_sum_sum_eq_one :
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits (2017^2017)))) = 1 := 
sorry

end sum_sum_sum_sum_eq_one_l457_457559


namespace students_from_other_communities_eq_90_l457_457887

theorem students_from_other_communities_eq_90 {total_students : ℕ} 
  (muslims_percentage : ℕ)
  (hindus_percentage : ℕ)
  (sikhs_percentage : ℕ)
  (christians_percentage : ℕ)
  (buddhists_percentage : ℕ)
  : total_students = 1000 →
    muslims_percentage = 36 →
    hindus_percentage = 24 →
    sikhs_percentage = 15 →
    christians_percentage = 10 →
    buddhists_percentage = 6 →
    (total_students * (100 - (muslims_percentage + hindus_percentage + sikhs_percentage + christians_percentage + buddhists_percentage))) / 100 = 90 :=
by
  intros h_total h_muslims h_hindus h_sikhs h_christians h_buddhists
  -- Proof can be omitted as indicated
  sorry

end students_from_other_communities_eq_90_l457_457887


namespace truck_capacity_l457_457027

-- Definitions based on conditions
def initial_fuel : ℕ := 38
def total_money : ℕ := 350
def change : ℕ := 14
def cost_per_liter : ℕ := 3

-- Theorem statement
theorem truck_capacity :
  initial_fuel + (total_money - change) / cost_per_liter = 150 := by
  sorry

end truck_capacity_l457_457027


namespace perpendicular_vectors_l457_457114

variable (λ μ : ℝ)
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

theorem perpendicular_vectors : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0 → λ * μ = -1 := by
  sorry

end perpendicular_vectors_l457_457114


namespace min_value_a_b_l457_457563

variables (a b : ℝ) (x₁ x₂ x₃ : ℝ)

/-- Mathematically equivalent proof problem in Lean 4 -/
theorem min_value_a_b :
  ∃ (x₁ x₂ x₃ : ℝ),
    x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1 ∧
    (f : ℝ → ℝ) f = (λ x, x^3 + a*x^2 + b*x) ∧
    f x₁ = f x₂ ∧ f x₃ =
  |a| + 2*|b| = √3 :=
sorry

end min_value_a_b_l457_457563


namespace exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l457_457017

theorem exp_periodic_cos_sin (x : ℝ) : ∃ k : ℤ, cos(x) = cos(x + 2 * k * π) ∧ sin(x) = sin(x + 2 * k * π) :=
begin
  use 1,
  split;
  apply real.cos_periodic;
  exact int.cast_coe_int (1 : ℤ)
end

theorem euler_formula (x : ℝ) : complex.exp (x * complex.I) = complex.cos x + complex.I * complex.sin x :=
by sorry

theorem exp_13_pi_by_2_equals_i : complex.exp (13 * real.pi / 2 * complex.I) = complex.I :=
begin
  -- Use Euler's formula
  have h_euler : complex.exp (13 * real.pi / 2 * complex.I) = complex.cos (13 * real.pi / 2) + complex.I * complex.sin (13 * real.pi / 2),
  { apply euler_formula },

  -- Simplify the angle by periodicity
  have h_angle : 13 * real.pi / 2 = 6 * real.pi + real.pi / 2,
  { field_simp, ring },
  
  -- Cos and Sin periodicity with 2π
  have h_cos : complex.cos (6 * real.pi + real.pi / 2) = complex.cos (real.pi / 2),
  { rw [← complex.cos_add_period],
    apply exp_periodic_cos_sin,
  },
  
  have h_sin : complex.sin (6 * real.pi + real.pi / 2) = complex.sin (real.pi / 2),
  { rw [← complex.sin_add_period],
    apply exp_periodic_cos_sin,
  },

  -- Calculate Cos(real.pi / 2) and Sin(real.pi / 2)
  have h_cos_pi_by_2 : complex.cos (real.pi / 2) = 0,
  { apply complex.cos_pi_div_two },
  
  have h_sin_pi_by_2 : complex.sin (real.pi / 2) = 1,
  { apply complex.sin_pi_div_two },
  
  -- Combine results
  rw [h_euler, h_angle, h_cos, h_sin],
  rw [h_cos_pi_by_2, h_sin_pi_by_2],
  ring,
end

end exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l457_457017


namespace original_number_people_l457_457371

theorem original_number_people (n : ℕ) (h1 : n / 3 * 2 / 2 = 18) : n = 54 :=
sorry

end original_number_people_l457_457371


namespace coefficients_square_sum_l457_457860

theorem coefficients_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1000 * x ^ 3 + 27 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end coefficients_square_sum_l457_457860


namespace magnitude_of_b_l457_457826

variables {a b : EuclideanSpace ℝ (Fin 3)}
noncomputable def magnitude_a := (3 : ℝ)
noncomputable def magnitude_sum := (Real.sqrt 13)

open Real EuclideanSpace

theorem magnitude_of_b
  (angle_ab : angle a b = 2 * π / 3)
  (magnitude_a : ∥a∥ = 3)
  (magnitude_sum : ∥a + b∥ = Real.sqrt 13) : 
  ∥b∥ = 4 :=
sorry

end magnitude_of_b_l457_457826


namespace part_a_part_b_l457_457638

noncomputable def tsunami_area_center_face (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  180000 * Real.pi + 270000 * Real.sqrt 3

noncomputable def tsunami_area_mid_edge (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7

theorem part_a (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_center_face l v t = 180000 * Real.pi + 270000 * Real.sqrt 3 :=
by
  sorry

theorem part_b (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_mid_edge l v t = 720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7 :=
by
  sorry

end part_a_part_b_l457_457638


namespace count_lattice_points_on_segment_l457_457763

theorem count_lattice_points_on_segment : ∀ (x1 y1 x2 y2 : ℕ), (x1, y1) = (15, 35) → (x2, y2) = (75, 515) → ∃ n : ℕ, n = 61 :=
by
  intros x1 y1 x2 y2 h1 h2
  use 61
  sorry

end count_lattice_points_on_segment_l457_457763


namespace evaluate_complex_fraction_l457_457408

def complex_fraction : Prop :=
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  expr = 76 / 29

theorem evaluate_complex_fraction : complex_fraction :=
by
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  show expr = 76 / 29
  sorry

end evaluate_complex_fraction_l457_457408


namespace roots_expressible_in_form_l457_457048

noncomputable def solve_n : ℕ :=
let a := 3
let b := -9
let c := -6
let discriminant := b * b - 4 * a * c
let gcd_criteria := nat.gcd a (nat.gcd (int.natAbs (-b)) (int.natAbs discriminant))
let result : ℕ := 153 in
if gcd_criteria = 1 then result else 0

theorem roots_expressible_in_form : solve_n = 153 := 

by {
  -- Placeholder for the proof
  sorry
}

end roots_expressible_in_form_l457_457048


namespace math_expression_result_l457_457678

-- Conditions as definitions
def part1 := (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5020
def part2 := (2^3 : ℚ) * (4 / 5 : ℚ) * 250
def part3 := (real.sqrt 900)

-- Main theorem statement
theorem math_expression_result :
  part1 - part2 + part3 = -817 := by
  sorry

end math_expression_result_l457_457678


namespace find_angle_between_vectors_l457_457853

noncomputable def vector_angle := 
  let a := (3 : ℝ, 0 : ℝ)
  let ab := (1 : ℝ, 2 * Real.sqrt 3 : ℝ)
  let b := ((ab.fst - a.fst) / 2, (ab.snd - a.snd) / 2) 
  let dot_product := a.fst * b.fst + a.snd * b.snd
  let magnitude_a := Real.sqrt (a.fst ^ 2 + a.snd ^ 2)
  let magnitude_b := Real.sqrt (b.fst ^ 2 + b.snd ^ 2)
  let cos_angle := dot_product / (magnitude_a * magnitude_b)
  let angle := Real.acos cos_angle * (180 / Real.pi)
  angle

theorem find_angle_between_vectors : vector_angle = 120 := 
  sorry

end find_angle_between_vectors_l457_457853


namespace QY_is_20_l457_457933

variable (Q R X Y : Type) [MetricSpace Q]
variable (C : Set Q)
variable {d : MetricSpace.Dist} {QX QR XY QY : ℝ}
variable (h1 : XY > 0)
variable (h2 : QX = 5)
variable (h3 : QR = XY - QX)
variable (h4 : ∃ tangent_line at R ∈ C)    
variable (h5 : ∃ secant_line intersecting C at X Y with QX < QY)

theorem QY_is_20 : QY = 20 :=
by
  sorry

end QY_is_20_l457_457933


namespace candles_ratio_l457_457225

-- Conditions
def kalani_bedroom_candles : ℕ := 20
def donovan_candles : ℕ := 20
def total_candles_house : ℕ := 50

-- Definitions for the number of candles in the living room and the ratio
def living_room_candles : ℕ := total_candles_house - kalani_bedroom_candles - donovan_candles
def ratio_of_candles : ℚ := kalani_bedroom_candles / living_room_candles

theorem candles_ratio : ratio_of_candles = 2 :=
by
  sorry

end candles_ratio_l457_457225


namespace find_common_difference_l457_457533

variable (a₁ d : ℝ)

theorem find_common_difference
  (h1 : a₁ + (a₁ + 6 * d) = 22)
  (h2 : (a₁ + 3 * d) + (a₁ + 9 * d) = 40) :
  d = 3 := by
  sorry

end find_common_difference_l457_457533


namespace inequalities_proof_l457_457680

theorem inequalities_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (1 / a + 1 / b) ≥ 4 ∧
  a^2 + b^2 + 2 ≥ 2 * a + 2 * b ∧
  sqrt (abs (a - b)) ≥ sqrt a - sqrt b :=
sorry

end inequalities_proof_l457_457680


namespace point_in_fourth_quadrant_l457_457525

-- Define a structure for a Cartesian point
structure Point where
  x : ℝ
  y : ℝ

-- Define different quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Function to determine the quadrant of a given point
def quadrant (p : Point) : Quadrant :=
  if p.x > 0 ∧ p.y > 0 then Quadrant.first
  else if p.x < 0 ∧ p.y > 0 then Quadrant.second
  else if p.x < 0 ∧ p.y < 0 then Quadrant.third
  else Quadrant.fourth

-- The main theorem stating the point (3, -4) lies in the fourth quadrant
theorem point_in_fourth_quadrant : quadrant { x := 3, y := -4 } = Quadrant.fourth :=
  sorry

end point_in_fourth_quadrant_l457_457525


namespace evaluate_expression_l457_457772

theorem evaluate_expression (x y z : ℝ) : [x + (y - z)] - [(x + z) - y] = 2y - 2z := 
by
  sorry

end evaluate_expression_l457_457772


namespace solve_for_x0_l457_457842

def f (x : ℝ) :=
  if 0 ≤ x ∧ x ≤ 2 then x^2 - 4
  else if 2 < x then 2 * x
  else 0  -- This is to cover cases outside the given conditions, although not necessary for the problem statement.

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = -2) : x0 = Real.sqrt 2 :=
by
  sorry

end solve_for_x0_l457_457842


namespace total_amount_given_away_l457_457218

variable (numGrandchildren : ℕ)
variable (cardsPerGrandchild : ℕ)
variable (amountPerCard : ℕ)

theorem total_amount_given_away (h1 : numGrandchildren = 3) (h2 : cardsPerGrandchild = 2) (h3 : amountPerCard = 80) : 
  numGrandchildren * cardsPerGrandchild * amountPerCard = 480 := by
  sorry

end total_amount_given_away_l457_457218


namespace greatest_q_minus_r_l457_457296

theorem greatest_q_minus_r :
  ∃ q r : ℤ, q > 0 ∧ r > 0 ∧ 975 = 23 * q + r ∧ q - r = 33 := sorry

end greatest_q_minus_r_l457_457296


namespace find_real_number_a_l457_457828

theorem find_real_number_a (a : ℝ) (h : (a^2 - 3*a + 2 = 0)) (h' : (a - 2) ≠ 0) : a = 1 :=
sorry

end find_real_number_a_l457_457828


namespace exp_increasing_a_lt_zero_l457_457082

theorem exp_increasing_a_lt_zero (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (1 - a) ^ x1 < (1 - a) ^ x2) : a < 0 := 
sorry

end exp_increasing_a_lt_zero_l457_457082


namespace coefficient_x7_expansion_l457_457621

theorem coefficient_x7_expansion :
  let f := (λ x : ℝ, (2 * x - 1) * (1/x + 2 * x)^6)
  (x : ℝ) in
  (f 2023) = 128 := 
sorry

end coefficient_x7_expansion_l457_457621


namespace ratio_AM_MC_l457_457549

-- Definitions for lengths of sides in triangle ABC
variables {A B C M : Point}
variable [linear_ordered_field ℝ]
variable (AB BC CA : ℝ)

-- Given side lengths of triangle ABC
axiom h1 : AB = 12
axiom h2 : BC = 13
axiom h3 : CA = 15

-- Point M lies on side AC such that the radii of the inscribed circles in triangles ABM and BCM are equal
variable (AM MC : ℝ)
axiom M_on_AC : AM + MC = CA
axiom equal_radii : (∀ (r1 r2 : ℝ), radius_inscribed_circle ⟨A, B, M⟩ r1 → radius_inscribed_circle ⟨B, C, M⟩ r2 → r1 = r2)

-- Required to prove the ratio |AM| : |MC| is 22:23
theorem ratio_AM_MC :
  AM / MC = 22 / 23 :=
by sorry

end ratio_AM_MC_l457_457549


namespace parent_selection_l457_457312

theorem parent_selection (total_parents_g10 total_parents_g11 total_parents_g12 total_sample_size : ℕ)
  (p1 p2 p3 : ℕ) : 
  total_parents_g10 = 54 →
  total_parents_g11 = 18 →
  total_parents_g12 = 36 →
  total_sample_size = 6 → 
  (total_parents_g10 + total_parents_g11 + total_parents_g12 = 108) →
  p1 = total_parents_g10 * total_sample_size / (total_parents_g10 + total_parents_g11 + total_parents_g12) →
  p2 = total_parents_g11 * total_sample_size / (total_parents_g10 + total_parents_g11 + total_parents_g12) →
  p3 = total_parents_g12 * total_sample_size / (total_parents_g10 + total_parents_g11 + total_parents_g12) →
  p1 = 3 ∧ p2 = 1 ∧ p3 = 2 ∧ 
  (∃ (parents : list ℕ), parents.length = 6 ∧ 
    (parents.choose 3).length = 20 ∧ 
    (parents.filter (λ p, p = 12)).choose 3.length = 4 ∧
    (1 - (4 / 20) = (4 / 5))) :=
by
  intros
  sorry

end parent_selection_l457_457312


namespace find_investment_of_c_l457_457689

noncomputable def investment_of_c : ℝ 
  := 12000

theorem find_investment_of_c (P_a P_b P_c : ℝ) (investment_a investment_b : ℝ) 
  (prop_P_b : P_b = 1500)
  (investment_a_val : investment_a = 8000)
  (investment_b_val : investment_b = 10000)
  (diff_P_c_P_a : P_c - P_a = 599.9999999999999)
  (P_a_val : P_a = (P_b / investment_b * investment_a)) :
  investment_of_c = (P_c / (P_b / investment_b)) :=
by
  rw [prop_P_b, investment_a_val, investment_b_val, diff_P_c_P_a, P_a_val]
  sorry

end find_investment_of_c_l457_457689


namespace greatest_integer_sequence_l457_457727

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 2
  else if n = 1 then 5 / 2
  else sequence (n-1) * ((sequence (n-2)) ^ 2 - 2) - 5 / 2

theorem greatest_integer_sequence (n : ℕ) : 
  int.floor (sequence n) = int.floor (2 ^ ((2 ^ n - (-1)^n) / 3)) :=
by
  sorry

end greatest_integer_sequence_l457_457727


namespace range_of_a_l457_457950

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) →
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l457_457950


namespace equivalent_polar_point_representation_l457_457182

/-- Representation of a point in polar coordinates -/
structure PolarPoint :=
  (r : ℝ)
  (θ : ℝ)

theorem equivalent_polar_point_representation :
  ∀ (p1 p2 : PolarPoint), p1 = PolarPoint.mk (-1) (5 * Real.pi / 6) →
    (p2 = PolarPoint.mk 1 (11 * Real.pi / 6) → p1.r + Real.pi = p2.r ∧ p1.θ = p2.θ) :=
by
  intros p1 p2 h1 h2
  sorry

end equivalent_polar_point_representation_l457_457182


namespace cylinder_surface_area_l457_457723

namespace SurfaceAreaProof

variables (a b : ℝ)

theorem cylinder_surface_area (a b : ℝ) :
  (2 * Real.pi * a * b) = (2 * Real.pi * a * b) :=
by sorry

end SurfaceAreaProof

end cylinder_surface_area_l457_457723


namespace a_minus_d_l457_457429

theorem a_minus_d (a b c d : ℕ) (h1 : a * b - a - b = 194) (h2 : b * c + b + c = 230) 
(h3 : c * d - c - d = 272) (h4 : a * b * c * d = Nat.factorial 9) : 
a - d = -11 :=
by {
  sorry
}

end a_minus_d_l457_457429


namespace find_foot_of_perpendicular_l457_457210

-- Definitions based on given conditions
structure Point3D (α : Type) :=
  (x : α) (y : α) (z : α)

def P : Point3D ℝ := ⟨1, real.sqrt 2, real.sqrt 3⟩

-- Problem statement
theorem find_foot_of_perpendicular (Q : Point3D ℝ) 
  (h1 : Q.x = 0)
  (h2 : Q.y = P.y)
  (h3 : Q.z = P.z) : 
  Q = ⟨0, real.sqrt 2, real.sqrt 3⟩ :=
sorry

end find_foot_of_perpendicular_l457_457210


namespace seq_inequality_l457_457213

noncomputable def seq (n : ℕ) : ℝ :=
if h : n = 1 then 2 else
have n > 0 := sorry, -- n > 0 must hold for n != 0
(sembdds h).elim (seq (n - 1))

theorem seq_inequality (n : ℕ) (hn : n > 0) : 
  seq n < sqrt 2 + 1 / n := by
  induction n with k hk,
  -- Prove base case
  case zero {
    sorry -- Base case: n = 1 (as ℕ^* should start from 1)
  }
  -- Prove inductive step
  case succ {
    -- prove the inductive step based on recurrence relation
    sorry
  }

end seq_inequality_l457_457213


namespace correct_option_b_l457_457330

def eval_A (x : ℝ) : ℝ := (x ^ 4) ^ 4
def eval_B (x : ℝ) : ℝ := x ^ 2 * x * x ^ 4
def eval_C (x y : ℝ) : ℝ := (-2 * x * y ^ 2) ^ 3
def eval_D (x : ℝ) : ℝ := (-x) ^ 5 + (-x) ^ 2

theorem correct_option_b (x y : ℝ) : 
  eval_A x = x ^ 16 ∧ eval_B x = x ^ 7 ∧ 
  eval_C x y = -8 * x ^ 3 * y ^ 6 ∧ eval_D x ≠ -x ^ 3 → 
  eval_B x = x ^ 7 :=
begin
  intro h,
  exact h.2.1,
end

end correct_option_b_l457_457330


namespace graph_of_3sin_is_not_increasing_l457_457293

theorem graph_of_3sin_is_not_increasing :
  ¬(∀ x, -π / 12 ≤ x ∧ x ≤ 7 * π / 12 → strict_mono_in_on (λ x, 3 * sin (2 * x - π / 3)) (Icc (-π / 12) (7 * π / 12))) :=
sorry

end graph_of_3sin_is_not_increasing_l457_457293


namespace collinear_lambda_value_l457_457150

theorem collinear_lambda_value
  (e1 e2 : Type)
  [is_vector_space e1]
  [is_vector_space e2]
  (e1_ne_e2 : ¬ collinear e1 e2) :
  (∃ λ : ℝ, 2 • e1 - e2 = 3 • e1 + λ • e2) ↔ λ = -3/2 :=
sorry

end collinear_lambda_value_l457_457150


namespace matrix_determinant_l457_457754

theorem matrix_determinant (a b : ℝ) : 
  let M := Matrix.of ![![1, Real.sin (a + b), Real.cos a], 
                        ![Real.sin (a + b), 1, Real.sin b], 
                        ![Real.cos a, Real.sin b, 1]] in 
  M.det = 2 * Real.sin (a + b) * Real.sin b * Real.cos a + Real.sin (a + b) ^ 2 - 1 := by
  sorry

end matrix_determinant_l457_457754


namespace find_a_trajectory_B_l457_457807

-- Question 1
theorem find_a (x y : ℝ) (a : ℝ) (hx : x^2 + y^2 - 2 * a * x + (4 - 2 * a) * y + 9 * a + 3 = 0)
  (hl : dist (a, a - 2) (3 * x + 4 * y + 8 = 0) = 7 / 5) :
  a = -1 := sorry

-- Question 2
theorem trajectory_B (x y : ℝ) (hx : (x + 1)^2 + (y + 3)^2 = 16)
  (hD : D = (1, 3))
  (h_ratio : (2 : ℝ) / 3) :
  (x + 1 / 2)^2 + (y + 3 / 2)^2 = 9 := sorry

end find_a_trajectory_B_l457_457807


namespace days_to_fill_tank_l457_457216

theorem days_to_fill_tank :
  let tank_capacity_liters : ℕ := 350
  let tank_capacity_ml := tank_capacity_liters * 1000
  let avg_rain_collection := (300 + 600) / 2
  let avg_river_collection := (900 + 1500) / 2
  let avg_daily_collection := avg_rain_collection + avg_river_collection
  let required_days := (tank_capacity_ml : ℕ) / avg_daily_collection
  ceil (required_days) = 213 :=
by
  let tank_capacity_liters : ℕ := 350
  let tank_capacity_ml := tank_capacity_liters * 1000
  let avg_rain_collection := (300 + 600) / 2
  let avg_river_collection := (900 + 1500) / 2
  let avg_daily_collection := avg_rain_collection + avg_river_collection
  let required_days := (tank_capacity_ml : ℕ) / avg_daily_collection
  show ceil (required_days) = 213
  sorry

end days_to_fill_tank_l457_457216


namespace sqrt_fraction_eq_half_l457_457380

theorem sqrt_fraction_eq_half :
  sqrt ((16 ^ 6 + 8 ^ 8) / (16 ^ 3 + 8 ^ 9)) = 1 / 2 :=
by
  sorry

end sqrt_fraction_eq_half_l457_457380


namespace point_on_angle_describes_arc_of_circle_l457_457740

open EuclideanGeometry

theorem point_on_angle_describes_arc_of_circle
(angle BAC : ∀ A B C : Point, LinePoint B A → LinePoint A C → Sphere) 
(O1 O2 : Point) 
(r1 r2 : ℝ) 
(AB AC : Line)
(h₁ : Tangent O1 (angle B A C) AB r1)
(h₂ : Tangent O2 (angle B A C) AC r2) :
∃ A1 : Point, ArcCircle (segment O1 A1) (segment O2 A1) := 
sorry

end point_on_angle_describes_arc_of_circle_l457_457740


namespace perpendicular_dot_product_l457_457129

theorem perpendicular_dot_product (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → (λ * μ = -1) :=
by
  intros
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  sorry

end perpendicular_dot_product_l457_457129


namespace mono_intervals_max_min_values_ln_inequality_l457_457474

def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) / (a * x) - Real.log x

theorem mono_intervals (a : ℝ) (h : a ≠ 0) : 
  (∀ x > 0, if a < 0 then f x a is monotone_decreasing_on Set.Ioi 0 else 
            ∀ x ∈ Set.Ioo 0 (1 / a), f x a is monotone_increasing_on (Set.Ioo 0 (1 / a)) ∧ 
            ∀ x ∈ Set.Ioi (1 / a), f x a is monotone_decreasing_on (Set.Ioi (1 / a))) := sorry

theorem max_min_values : 
  let a := 1
  ∀ x ∈ (Set.Icc (1 / 2) 2), 
    (f 1 1 = 0 ∧ f (1 / 2) 1 = -1 + Real.log 2 ∧ 
     (∀ x ∈ (Set.Icc (1 / 2) 2), f 1 1 ≥ f x 1) ∧ 
     (∀ x ∈ (Set.Icc (1 / 2) 2), f 1 1 ≤ f (1 / 2) 1)) := sorry

theorem ln_inequality (x : ℝ) (hx : 0 < x) : 
  Real.log (Real.exp 2 / x) ≤ (1 + x) / x := sorry

end mono_intervals_max_min_values_ln_inequality_l457_457474


namespace team_game_probabilities_l457_457733

-- Define the given conditions
def P_A : ℝ := 1 / 3
def P_A_and_B : ℝ := 1 / 6
def P_B_and_C : ℝ := 1 / 5

-- Define probabilities we want to prove
def P_B : ℝ := 1 / 2
def P_C : ℝ := 2 / 5

-- Final probabilities for the team's score
def P_score_4 : ℝ := 3 / 10
def P_next_round : ℝ := 11 / 30

-- Main theorem to verify all the probabilities
theorem team_game_probabilities :
  (P(A) = 1 / 3) ∧ (P(A ∩ B) = 1 / 6) ∧ (P(B ∩ C) = 1 / 5) →
  (P(B) = 1 / 2) ∧ (P(C) = 2 / 5) ∧ (P(score_4) = 3 / 10) ∧ (P(next_round) = 11 / 30) :=
sorry

end team_game_probabilities_l457_457733


namespace multiply_expression_l457_457986

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l457_457986


namespace perpendicular_lines_l457_457241

theorem perpendicular_lines (A B C D O P M N : Point) (Γ : Circle)
  (hA : A ∈ Γ) (hC : C ∈ Γ)
  (hO : (∃ M N, bisects (LineSegment A C) O M N) ∧ O = midpoint A C)
  (hB : B = intersection (angle_bisector O A) Γ)
  (hD : D = intersection (angle_bisector O C) Γ)
  (hOrder : cyclic_order Γ [A, B, C, D])
  (hP : P = intersection (line A B) (line C D))
  (hM : M = midpoint A B)
  (hN : N = midpoint C D) :
  perpendicular (line M N) (line O P) :=
sorry

end perpendicular_lines_l457_457241


namespace problem1_problem2_l457_457450

-- Define the propositions
def S (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

def p (m : ℝ) : Prop := 0 < m ∧ m < 2

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ 1 ≤ m := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hpq : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l457_457450


namespace range_of_tangent_slope_l457_457462

noncomputable def f (x : ℝ) : ℝ := 4 / (exp x + 1)

noncomputable def f' (x : ℝ) : ℝ := - (4 * exp x) / (exp (2 * x) + 2 * exp x + 1)

theorem range_of_tangent_slope : 
  ∀ x : ℝ, -1 ≤ f' x ∧ f' x < 0 :=
by
  sorry

end range_of_tangent_slope_l457_457462


namespace dodecagon_product_value_l457_457366

noncomputable def dodecagon_product : ℂ :=
  let Q_1 := (2 : ℝ, 0 : ℝ)
  let Q_7 := (4 : ℝ, 0 : ℝ)
  let q_ks := { x // ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 12 ∧ (x = (n : ℝ, n_to_point n)) }
  let n_to_point (n : ℕ) := sorry -- Assume a function that maps n to the coordinates of Q_n
  let q_k_to_complex (p : ℝ × ℝ) : ℂ := complex.ofReal p.1 + complex.i * complex.ofReal p.2
  q_ks.map (λ p, q_k_to_complex p.val).prod

theorem dodecagon_product_value :
  dodecagon_product = (531440 : ℂ) :=
sorry

end dodecagon_product_value_l457_457366


namespace total_number_of_employees_l457_457719

def weekly_hours (part_time: ℕ) (full_time: ℕ) (remote: ℕ) (temporary: ℕ) : ℕ :=
  part_time + full_time * 40 + remote * 40 + temporary * 40

def total_full_time_equiv (weekly_hours: ℕ) : ℝ :=
  weekly_hours / 40

theorem total_number_of_employees 
  (part_time: ℕ) (full_time: ℕ) (remote: ℕ) (temporary: ℕ) (one_fte_hours: ℕ)
  (h1: part_time = 2041) 
  (h2: full_time = 63093) 
  (h3: remote = 5230) 
  (h4: temporary = 8597) 
  (h5: one_fte_hours = 40) :
  total_full_time_equiv (weekly_hours part_time full_time remote temporary) = 76971 := 
by
  sorry

end total_number_of_employees_l457_457719


namespace infinite_coprime_pairs_divisibility_l457_457268

theorem infinite_coprime_pairs_divisibility :
  ∃ (S : ℕ → ℕ × ℕ), (∀ n, Nat.gcd (S n).1 (S n).2 = 1 ∧ (S n).1 ∣ (S n).2^2 - 5 ∧ (S n).2 ∣ (S n).1^2 - 5) ∧
  Function.Injective S :=
sorry

end infinite_coprime_pairs_divisibility_l457_457268


namespace sequence_sixth_term_l457_457645

theorem sequence_sixth_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n ≥ 1, a (n + 1) = 3 * S n) (h3 : ∀ n, S n = ∑ i in range (n + 1), a i):
  a 6 = 3 * 4^4 :=
by sorry

end sequence_sixth_term_l457_457645


namespace find_five_digit_number_l457_457358

open Nat

noncomputable def five_digit_number : ℕ := 14285

theorem find_five_digit_number (x : ℕ) (h : 7 * 10^5 + x = 5 * (10 * x + 7)) : x = five_digit_number :=
by
  sorry

end find_five_digit_number_l457_457358


namespace sum_of_arithmetic_progressions_l457_457940

theorem sum_of_arithmetic_progressions :
  ∀ (a b : ℕ → ℝ) (d_a d_b : ℝ),
    (a 1 = 10) →
    (b 1 = 90) →
    (a 50 + b 50 = 200) →
    (∀ n, a n = 10 + (n - 1) * d_a) →
    (∀ n, b n = 90 + (n - 1) * d_b) →
    ∑ i in Finset.range 50, (a (i+1) + b (i+1)) = 7500 :=
by
  intro a b d_a d_b h1 h2 h3 ha hb
  sorry

end sum_of_arithmetic_progressions_l457_457940


namespace lorelei_vase_rose_count_l457_457201

theorem lorelei_vase_rose_count :
  let r := 12
  let p := 18
  let y := 20
  let o := 8
  let picked_r := 0.5 * r
  let picked_p := 0.5 * p
  let picked_y := 0.25 * y
  let picked_o := 0.25 * o
  picked_r + picked_p + picked_y + picked_o = 22 := 
by
  sorry

end lorelei_vase_rose_count_l457_457201


namespace problem_approx_l457_457858

theorem problem_approx {x : ℝ} (h : 8^x - 8^(x-1) = 60) : (3*x)^x ≈ 58 := by
  sorry

end problem_approx_l457_457858


namespace slope_of_line_AB_on_ellipse_l457_457427

theorem slope_of_line_AB_on_ellipse 
  (A B : ℝ × ℝ) 
  (hA : A.1^2 / 9 + A.2^2 / 3 = 1)
  (hB : B.1^2 / 9 + B.2^2 / 3 = 1)
  (M : ℝ × ℝ) 
  (hM : M = (Real.sqrt 3, Real.sqrt 2))
  (hS : ∃ k : ℝ, A.2 - M.2 = k * (A.1 - M.1) ∧ B.2 - M.2 = (-1/k) * (B.1 - M.1)) : 
  (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 6 / 6 :=
sorry

end slope_of_line_AB_on_ellipse_l457_457427


namespace circle_distance_l457_457831

/-- Given a circle with radius 5 and a point P outside of the circle, the distance OP must be greater than 5.
    Given the options 3, 4, 5, and 6, we need to prove that the only possible length for OP is 6. -/
theorem circle_distance (O P : Point) (r : ℝ) (h₀ : r = 5) (h₁ : dist O P > r) : 
  (dist O P = 6 ∨ dist O P ≠ 3 ∨ dist O P ≠ 4 ∨ dist O P ≠ 5) :=
by
  sorry

end circle_distance_l457_457831


namespace consecutive_integers_sum_l457_457451

theorem consecutive_integers_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < Real.sqrt 17) (h4 : Real.sqrt 17 < b) : a + b = 9 :=
sorry

end consecutive_integers_sum_l457_457451


namespace general_formula_a_n_sum_first_n_b_l457_457944

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Sequence property
def seq_property (n : ℕ) (S_n : ℕ) : Prop :=
  a_n n ^ 2 + 2 * a_n n = 4 * S_n + 3

-- General formula for {a_n}
theorem general_formula_a_n (n : ℕ) (hpos : ∀ n, a_n n > 0) (S_n : ℕ) (hseq : seq_property n S_n) :
  a_n n = 2 * n + 1 :=
sorry

-- Sum of the first n terms of {b_n}
def b_n (n : ℕ) : ℚ := 1 / ((a_n n) * (a_n (n + 1)))

def sum_b (n : ℕ) (T_n : ℚ) : Prop :=
  T_n = (1 / 2) * ((1 / (2 * n + 1)) - (1 / (2 * n + 3)))

theorem sum_first_n_b (n : ℕ) (hpos : ∀ n, a_n n > 0) (T_n : ℚ) :
  T_n = (n : ℚ) / (3 * (2 * n + 3)) :=
sorry

end general_formula_a_n_sum_first_n_b_l457_457944


namespace sum_of_min_value_and_input_l457_457838

def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem sum_of_min_value_and_input : 
  let a := -1
  let b := 3 * a - a ^ 3
  a + b = -3 := 
by
  let a := -1
  let b := 3 * a - a ^ 3
  sorry

end sum_of_min_value_and_input_l457_457838


namespace count_bad_carrots_l457_457414

theorem count_bad_carrots (faye_picked : ℕ) (mom_picked : ℕ) (good_carrots : ℕ) :
  faye_picked = 23 → mom_picked = 5 → good_carrots = 12 → (faye_picked + mom_picked - good_carrots) = 16 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end count_bad_carrots_l457_457414


namespace mass_of_man_l457_457335

theorem mass_of_man (L B h ρ : ℝ) (sinking_height_cm : ℝ) (length_condition : L = 3) (breadth_condition : B = 2)
    (sinking_height_condition : h = sinking_height_cm / 100) (density_condition : ρ = 1000) : 
    L * B * h * ρ = 90 :=
by
  have sinking_height_m : h = 1.5 / 100 := by
    rw [sinking_height_condition]
  have volume := 3 * 2 * (1.5 / 100) := by 
    rw [length_condition, breadth_condition, sinking_height_m]
  have mass_of_displaced_water := 1000 * volume := by
    rw [density_condition]
  exact mass_of_displaced_water

end mass_of_man_l457_457335


namespace angle_B_measure_l457_457959

theorem angle_B_measure (l k : Line) (parallel_lk : l ∥ k) (mangleA mangleC mangleB : ℝ) 
  (hA : mangleA = 100) (hC : mangleC = 70) :
  mangleB = 170 :=
sorry

end angle_B_measure_l457_457959


namespace shopkeeper_percentage_gain_l457_457714

theorem shopkeeper_percentage_gain
  (false_weight : ℝ)
  (true_weight : ℝ)
  (cost_price_profession : Prop)
  (percentage_gain : ℝ)
  (h1 : false_weight = 970)
  (h2 : true_weight = 1000)
  (h3 : percentage_gain = (30 / 970) * 100) :
  percentage_gain ≈ 3.09 := 
sorry

end shopkeeper_percentage_gain_l457_457714


namespace cumulative_number_of_squares_up_to_50th_ring_l457_457755

theorem cumulative_number_of_squares_up_to_50th_ring :
  (∑ n in Finset.range 50, 8 * (n + 1)) = 10200 :=
by
  sorry

end cumulative_number_of_squares_up_to_50th_ring_l457_457755


namespace kth_roots_of_unity_sum_is_real_l457_457244

variables {k : ℕ} {x y : ℂ}

theorem kth_roots_of_unity_sum_is_real (hx : x ^ k = 1) (hy : y ^ k = 1) : 
  (x + y) ^ k = complex.conj ((x + y) ^ k) :=
sorry

end kth_roots_of_unity_sum_is_real_l457_457244


namespace find_smallest_palindrome_l457_457047

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_aba_form (n : ℕ) : Prop :=
  let s := n.digits 10
  s.length = 3 ∧ s.head = s.get! 2

def smallest_aba_not_palindromic_when_multiplied_by_103 : ℕ :=
  Nat.find (λ n, is_three_digit n ∧ is_aba_form n ∧ ¬is_palindrome (103 * n))

theorem find_smallest_palindrome : smallest_aba_not_palindromic_when_multiplied_by_103 = 131 := sorry

end find_smallest_palindrome_l457_457047


namespace solve_a_l457_457024

def custom_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a :
  ∃ a : ℝ, custom_op a 7 = -20 ∧ a = 29 / 2 :=
by
  sorry

end solve_a_l457_457024


namespace trigonometric_identity_l457_457432

theorem trigonometric_identity (α : ℝ) (h1 : tan (2 * α) = 3 / 4) (h2 : 0 < α ∧ α < π / 4) :
  (sin α + cos α) / (sin α - cos α) = -2 :=
sorry

end trigonometric_identity_l457_457432


namespace harold_shirt_boxes_l457_457489

def rolls_needed_for_xl_boxes (num_xl_boxes rolls_per_xl_box: ℕ) :=
  num_xl_boxes / rolls_per_xl_box

def total_rolls_bought (total_cost cost_per_roll: ℕ) :=
  total_cost / cost_per_roll

def rolls_left_for_shirt_boxes (total_rolls rolls_for_xl : ℕ) :=
  total_rolls - rolls_for_xl

def shirt_boxes_wrapped (rolls rolls_per_shirt_box: ℕ) :=
  rolls * rolls_per_shirt_box

theorem harold_shirt_boxes (num_xl_boxes rolls_per_xl_box total_cost cost_per_roll rolls_per_shirt_box : ℕ) :
  num_xl_boxes = 12 ∧ rolls_per_xl_box = 3 ∧ total_cost = 32 ∧ cost_per_roll = 4 ∧ rolls_per_shirt_box = 5 →
  shirt_boxes_wrapped (rolls_left_for_shirt_boxes (total_rolls_bought total_cost cost_per_roll) (rolls_needed_for_xl_boxes num_xl_boxes rolls_per_xl_box)) rolls_per_shirt_box = 20 :=
by
  intros h
  let ⟨h1, h2, h3, h4, h5⟩ := h
  dsimp [num_xl_boxes, rolls_per_xl_box, total_cost, cost_per_roll, rolls_per_shirt_box] at h1 h2 h3 h4 h5
  sorry

end harold_shirt_boxes_l457_457489


namespace area_bounded_arcsin_cos_eq_l457_457420

noncomputable def area_arcsin_cos_interval : ℝ := 
  let y := λ x : ℝ, Real.arcsin (Real.cos x) in
  let area := Real.integral (Set.Icc 0 (2 * Real.pi)) (λ x, y x) in
  area

theorem area_bounded_arcsin_cos_eq : 
  area_arcsin_cos_interval = (Real.pi ^ 2) / 4 :=
sorry

end area_bounded_arcsin_cos_eq_l457_457420


namespace monotonic_intervals_k_le_zero_monotonic_intervals_k_gt_zero_minimum_value_k_one_inequality_ln_series_l457_457844

noncomputable def f (x k : ℝ) : ℝ := Real.log (1 + x) - k * x / (1 + x)

theorem monotonic_intervals_k_le_zero {k : ℝ} (h : k ≤ 0) :
  ∀ x : ℝ, x > -1 → deriv (λ x, f x k) x ≥ 0 := 
sorry

theorem monotonic_intervals_k_gt_zero {k : ℝ} (h : k > 0) :
  ∀ x : ℝ, x > -1 → (x > k-1 → deriv (λ x, f x k) x ≥ 0) ∧ (x < k-1 → deriv (λ x, f x k) x < 0) := 
sorry

theorem minimum_value_k_one : 
  ∃ x : ℝ, 0 ≤ x → f x 1 = 0 :=
sorry

theorem inequality_ln_series (n : ℕ) :
  ∑ i in Finset.range (n+1), 1 / (i+2 : ℝ) < Real.log (1 + n) := 
sorry

end monotonic_intervals_k_le_zero_monotonic_intervals_k_gt_zero_minimum_value_k_one_inequality_ln_series_l457_457844


namespace f_max_value_l457_457785

noncomputable def f (x m : ℝ) := 1/2 * Real.cos (2 * x) + m * Real.sin x + 1 / 2

theorem f_max_value (m : ℝ) : ∃ y : ℝ, y = (f m) ∧
  (m ≤ -2 → y = -m) ∧
  (-2 < m ∧ m < 2 → y = (m^2 / 4) + 1) ∧
  (m ≥ 2 → y = m) := 
sorry

end f_max_value_l457_457785


namespace values_of_k_real_equal_roots_l457_457766

theorem values_of_k_real_equal_roots (k : ℝ) :
  (∀ x : ℝ, 3 * x^2 - (k + 2) * x + 12 = 0 → x * x = 0) ↔ (k = 10 ∨ k = -14) :=
by
  sorry

end values_of_k_real_equal_roots_l457_457766


namespace slope_magnitude_l457_457088

-- Definitions based on given conditions
def parabola : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y^2 = 4 * x }
def line (k m : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = k * x + m }
def focus : ℝ × ℝ := (1, 0)
def intersects (l p : Set (ℝ × ℝ)) : Prop := ∃ x1 y1 x2 y2, (x1, y1) ∈ l ∧ (x1, y1) ∈ p ∧ (x2, y2) ∈ l ∧ (x2, y2) ∈ p ∧ (x1, y1) ≠ (x2, y2)

theorem slope_magnitude (k m : ℝ) (h_k_nonzero : k ≠ 0) 
  (h_intersects : intersects (line k m) parabola) 
  (h_AF_2FB : ∀ x1 y1 x2 y2, (x1, y1) ∈ line k m → (x1, y1) ∈ parabola → 
                          (x2, y2) ∈ line k m → (x2, y2) ∈ parabola → 
                          (1 - x1 = 2 * (x2 - 1)) ∧ (-y1 = 2 * y2)) :
  |k| = 2 * Real.sqrt 2 :=
sorry

end slope_magnitude_l457_457088


namespace dot_product_tangent_vectors_l457_457449

noncomputable def M : Point := (2, 0)
def circle (p : Point) : Prop := p.1^2 + p.2^2 = 1
def tangent_point (p : Point) (M : Point) (C : Point -> Prop) : Prop := -- Define the tangent point

theorem dot_product_tangent_vectors (A B : Point) (hA : tangent_point M A circle) (hB : tangent_point M B circle):
  let MA := (A.1 - M.1, A.2 - M.2)
  let MB := (B.1 - M.1, B.2 - M.2)
  (MA.1 * MB.1 + MA.2 * MB.2) = 3/2 :=
sorry

end dot_product_tangent_vectors_l457_457449


namespace length_of_second_snake_l457_457385

theorem length_of_second_snake : 
  ∀ (x : ℕ), (2 * 12 + x + 10 = 50) → x = 16 :=
by 
  intro x
  assume h
  sorry  -- Proof goes here

end length_of_second_snake_l457_457385


namespace loop_executes_2_times_l457_457768

def loop_body_execution_count (initial_i : ℕ) (termination_condition : ℕ → Prop) (loop_body : ℕ → ℕ) : ℕ :=
  let rec count (i : ℕ) (iterations : ℕ) : ℕ :=
    if termination_condition i then
      iterations
    else
      count (loop_body (i + 1)) (iterations + 1)
  count initial_i 0

def custom_loop_body (i : ℕ) : ℕ := 5 * i

def termination_condition (i : ℕ) : Prop := i > 15

theorem loop_executes_2_times :
  loop_body_execution_count 1 termination_condition custom_loop_body = 2 :=
by sorry

end loop_executes_2_times_l457_457768


namespace positive_difference_sum_even_odd_l457_457322

theorem positive_difference_sum_even_odd :
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  sum_30_even - sum_25_odd = 305 :=
by
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  show sum_30_even - sum_25_odd = 305
  sorry

end positive_difference_sum_even_odd_l457_457322


namespace de_moivre_example_l457_457387

theorem de_moivre_example :
  (Complex.ofReal (cos (195 * Real.pi / 180)) + Complex.I * Complex.ofReal (sin (195 * Real.pi / 180))) ^ 60 =
  (Complex.ofReal (1 / 2) - Complex.I * Complex.ofReal (Real.sqrt 3 / 2)) :=
by
  sorry

end de_moivre_example_l457_457387


namespace lambda_mu_relationship_l457_457147

variables (λ μ : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def linear_combination (v : ℝ × ℝ) (c : ℝ) (w : ℝ × ℝ) := 
  (v.1 + c * w.1, v.2 + c * w.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem lambda_mu_relationship (h : dot_product (linear_combination vector_a λ vector_b)
                                          (linear_combination vector_a μ vector_b) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457147


namespace seq_an_periodic_l457_457946

def num_divisors (n : ℕ) : ℕ :=
finset.card (finset_Icc 1 n (λ m, n % m = 0))

def seq_an (A : ℕ) : ℕ → ℕ
| 0 := A
| (n + 1) := num_divisors (floor ((3/2 : ℚ) * seq_an n)) + 2011

theorem seq_an_periodic (A : ℕ) : ∃ k m, ∀ n ≥ m, seq_an A (n + k) = seq_an A n :=
sorry

end seq_an_periodic_l457_457946


namespace mean_equality_l457_457634

theorem mean_equality (y : ℝ) (h : (6 + 9 + 18) / 3 = (12 + y) / 2) : y = 10 :=
by sorry

end mean_equality_l457_457634


namespace cube_sum_inequality_l457_457168

theorem cube_sum_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) : 
  a^3 + b^3 ≤ a * b^2 + a^2 * b :=
sorry

end cube_sum_inequality_l457_457168


namespace exists_nonnegative_integers_for_sqrt_sum_floor_eq_sqrt_floor_l457_457565

theorem exists_nonnegative_integers_for_sqrt_sum_floor_eq_sqrt_floor 
  (a : ℕ → ℕ) (m : ℕ) (hm : 0 < m) :
  (∀ i, 1 ≤ i ∧ i ≤ m → 0 < a i) →
  ∃ b c N : ℕ, ∀ n, n > N →
    ⌊∑ i in finset.range m, real.sqrt (n + a i)⌋ = ⌊real.sqrt (b * n + c)⌋ :=
by
  sorry

end exists_nonnegative_integers_for_sqrt_sum_floor_eq_sqrt_floor_l457_457565


namespace find_all_integer_solutions_l457_457550

noncomputable def equation_solutions (k x : ℤ) : Prop := (k - 5) * x + 6 = 1 - 5 * x

theorem find_all_integer_solutions :
  ∃ (k : ℤ), ∃ (x : ℤ), equation_solutions k x → 
  (x = -5 ∨ x = 5 ∨ x = -1 ∨ x = 1) :=
begin
  sorry
end

end find_all_integer_solutions_l457_457550


namespace oranges_per_pack_correct_l457_457615

-- Definitions for the conditions.
def num_trees : Nat := 10
def oranges_per_tree_per_day : Nat := 12
def price_per_pack : Nat := 2
def total_earnings : Nat := 840
def weeks : Nat := 3
def days_per_week : Nat := 7

-- Theorem statement:
theorem oranges_per_pack_correct :
  let oranges_per_day := num_trees * oranges_per_tree_per_day
  let total_days := weeks * days_per_week
  let total_oranges := oranges_per_day * total_days
  let num_packs := total_earnings / price_per_pack
  total_oranges / num_packs = 6 :=
by
  sorry

end oranges_per_pack_correct_l457_457615


namespace smallest_hope_number_l457_457171

def is_square (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k
def is_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k
def is_fifth_power (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k * k * k

def is_hope_number (n : ℕ) : Prop :=
  is_square (n / 8) ∧ is_cube (n / 9) ∧ is_fifth_power (n / 25)

theorem smallest_hope_number : ∃ n, is_hope_number n ∧ n = 2^15 * 3^20 * 5^12 :=
by
  sorry

end smallest_hope_number_l457_457171


namespace sum_q_p_is_neg12_l457_457479

def p (x : ℝ) : ℝ := abs (x + 1) - 3
def q (x : ℝ) : ℝ := -abs x

def xs : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def sum_q_p : ℝ :=
  List.sum (List.map (fun x => q (p x)) xs)

theorem sum_q_p_is_neg12 : sum_q_p = -12 := 
  by
    sorry

end sum_q_p_is_neg12_l457_457479


namespace units_digit_n_l457_457789

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31 ^ 6) (h2 : m % 10 = 9) : n % 10 = 2 := 
sorry

end units_digit_n_l457_457789


namespace green_pill_cost_l457_457736

-- Define the conditions 
variables (pinkCost greenCost : ℝ)
variable (totalCost : ℝ := 819) -- total cost for three weeks
variable (days : ℝ := 21) -- number of days in three weeks

-- Establish relationships between pink and green pill costs
axiom greenIsMore : greenCost = pinkCost + 1
axiom dailyCost : 2 * greenCost + pinkCost = 39

-- Define the theorem to prove the cost of one green pill
theorem green_pill_cost : greenCost = 40/3 :=
by
  -- Proof would go here, but is omitted for now.
  sorry

end green_pill_cost_l457_457736


namespace zero_of_f_in_1_2_l457_457837

def f (x : ℝ) : ℝ := 2^x + 2*x - 6

theorem zero_of_f_in_1_2 : ∃ x0 : ℝ, f x0 = 0 ∧ 1 < x0 ∧ x0 < 2 := by
  have mono_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry
  have f_continuous : Continuous f := sorry
  have f_1_lt_0 : f 1 < 0 := by
    calc
      f 1 = 2^1 + 2 * 1 - 6 := rfl
      _ = 2 + 2 - 6 := rfl
      _ = -2 := rfl
      _ < 0 := by linarith
  have f_2_gt_0 : f 2 > 0 := by
    calc
      f 2 = 2^2 + 2 * 2 - 6 := rfl
      _ = 4 + 4 - 6 := rfl
      _ = 2 := rfl
      _ > 0 := by linarith
  exact
    ⟨Classical.some (exists_intermediate_value f_continuous f_1_lt_0 f_2_gt_0 1 2), 
     Classical.some_spec (exists_intermediate_value f_continuous f_1_lt_0 f_2_gt_0 1 2)⟩
  sorry

end zero_of_f_in_1_2_l457_457837


namespace factorization_l457_457058

theorem factorization (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by 
  sorry

end factorization_l457_457058


namespace probability_event_a_without_replacement_independence_of_events_with_replacement_l457_457704

open ProbabilityTheory MeasureTheory Set

-- Definitions corresponding to the conditions
def BallLabeled (i : ℕ) : Prop := i ∈ Finset.range 10

def EventA (second_ball : ℕ) : Prop := second_ball = 2

def EventB (first_ball second_ball : ℕ) (m : ℕ) : Prop := first_ball + second_ball = m

-- First Part: Probability without replacement
theorem probability_event_a_without_replacement :
  ∃ P_A : ℝ, P_A = 1 / 10 := sorry

-- Second Part: Independence with replacement
theorem independence_of_events_with_replacement (m : ℕ) :
  (EventA 2 → (∀ first_ball : ℕ, BallLabeled first_ball → EventB first_ball 2 m) ↔ m = 9) := sorry

end probability_event_a_without_replacement_independence_of_events_with_replacement_l457_457704


namespace heptagon_triangulation_count_l457_457519

/-- The number of ways to divide a regular heptagon (7-sided polygon) 
    into 5 triangles using non-intersecting diagonals is 4. -/
theorem heptagon_triangulation_count : ∃ (n : ℕ), n = 4 ∧ ∀ (p : ℕ), (p = 7 ∧ (∀ (k : ℕ), k = 5 → (n = 4))) :=
by {
  -- The proof is non-trivial and omitted here
  sorry
}

end heptagon_triangulation_count_l457_457519


namespace minimum_value_of_f_l457_457037

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 4

theorem minimum_value_of_f : ∃ x : ℝ, f x = -5 ∧ ∀ y : ℝ, f y ≥ -5 :=
by
  sorry

end minimum_value_of_f_l457_457037


namespace multiply_polynomials_l457_457984

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l457_457984


namespace parallelogram_area_l457_457776

theorem parallelogram_area (base height : ℝ) (h_base : base = 12) (h_height : height = 10) :
  base * height = 120 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l457_457776


namespace movies_watched_total_l457_457661

theorem movies_watched_total :
  ∀ (Timothy2009 Theresa2009 Timothy2010 Theresa2010 total : ℕ),
    Timothy2009 = 24 →
    Timothy2010 = Timothy2009 + 7 →
    Theresa2010 = 2 * Timothy2010 →
    Theresa2009 = Timothy2009 / 2 →
    total = Timothy2009 + Timothy2010 + Theresa2009 + Theresa2010 →
    total = 129 :=
by
  intros Timothy2009 Theresa2009 Timothy2010 Theresa2010 total
  sorry

end movies_watched_total_l457_457661


namespace weight_combinations_l457_457580

theorem weight_combinations (w1 w2 w5 : ℕ) (n : ℕ) (h1 : w1 = 3) (h2 : w2 = 3) (h5 : w5 = 1) (hn : n = 9) : 
  nat.add 
    (nat.add 
      (nat.add (nat.add (nat.add (nat.add (nat.add 1 1) 1) 1) 1) 1) 1) 1 = 
  8 := 
sorry

end weight_combinations_l457_457580


namespace monotone_f_range_of_m_l457_457872

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x < 1 then 1/2 * x + m else x - Real.log x

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem monotone_f_range_of_m (m : ℝ) : is_monotone_increasing (λ x, f x m) → (m ≤ 1/2) :=
sorry

end monotone_f_range_of_m_l457_457872


namespace three_star_five_l457_457804

-- Definitions based on conditions
def star (a b : ℕ) : ℕ := 2 * a^2 + 3 * a * b + 2 * b^2

-- Theorem statement to be proved
theorem three_star_five : star 3 5 = 113 := by
  sorry

end three_star_five_l457_457804


namespace remainder_a33_div_33_l457_457235

def a (n : ℕ) : ℕ := 
  let digits := (List.range (n + 1)).map (λ x, x + 1)  -- creating the sequence from 1 to n
  digits.foldl (λ acc d, acc * 10 ^ (Nat.log10 (d + 1) + 1) + d) 0

theorem remainder_a33_div_33 : (a 33) % 33 = 22 := by
sorry

end remainder_a33_div_33_l457_457235


namespace find_valid_number_l457_457663

def original_number : ℕ := 20172018
def valid_number (n : ℕ) : Prop := 
  (n % 8 = 0) ∧ (n % 9 = 0)

theorem find_valid_number : ∃ d₁ d₂, valid_number (d₁ * 10^10 + original_number * 10 + d₂) :=
  ∃ d₁ d₂, (valid_number (d₁ * 10^10 + 20172018 * 10 + d₂)) ∧ (d₁ = 2) ∧ (d₂ = 4) := 
sorry

end find_valid_number_l457_457663


namespace jean_grandchildren_total_giveaway_l457_457220

theorem jean_grandchildren_total_giveaway :
  let num_grandchildren := 3
  let cards_per_grandchild_per_year := 2
  let amount_per_card := 80
  let total_amount_per_grandchild_per_year := cards_per_grandchild_per_year * amount_per_card
  let total_amount_per_year := num_grandchildren * total_amount_per_grandchild_per_year
  total_amount_per_year = 480 :=
by
  sorry

end jean_grandchildren_total_giveaway_l457_457220


namespace incorrect_statement_l457_457332

structure Quadrilateral where
  A B C D : Type
  diagonals_bisect : Prop
  diagonals_equal : Prop
  diagonals_perpendicular : Prop

def is_parallelogram (q : Quadrilateral) : Prop :=
  q.diagonals_bisect

def is_rectangle (q : Quadrilateral) : Prop :=
  is_parallelogram q ∧ q.diagonals_equal

def is_rhombus (q : Quadrilateral) : Prop :=
  q.diagonals_perpendicular ∧ q.diagonals_bisect

def is_square (q : Quadrilateral) : Prop :=
  is_rectangle q ∧ q.diagonals_bisect ∧ q.diagonals_equal

theorem incorrect_statement (q : Quadrilateral) : is_rhombus q → ¬(is_rhombus q) := 
sorry

end incorrect_statement_l457_457332


namespace range_of_b_circle_through_intersections_l457_457187

noncomputable def quadraticFunc (x b : ℝ) : ℝ := -x^2 - 2*x + b

theorem range_of_b (b : ℝ) :
  (b > -1) ∧ (b ≠ 0) ↔ 
  (∃ D E F : ℝ, 
    (∀ x : ℝ, quadraticFunc x b = 0 → ((x + 1)^2 + D*(x + 1) + F = 0) ∨ (D + 1 = x ∨ E + 1 = x)) ∧ 
    (D = 2) ∧ 
    (E = 1 - b) ∧ 
    (F = -b)) := sorry

theorem circle_through_intersections (b : ℝ) :
  (b > -1) ∧ (b ≠ 0) → 
  (∃ x y : ℝ, 
    circle_eqn x y b = 0) := sorry

where circle_eqn (x y b : ℝ) :=
  x^2 + y^2 + 2*x + (1 - b)*y - b


end range_of_b_circle_through_intersections_l457_457187


namespace hyperbola_rectangular_asymptotes_l457_457598

variable {K : Type*} [Field K]

/-- Assume we have a triangle with vertices A, B, C and orthocenter H.
    Any hyperbola passing through these four points is a rectangular hyperbola. -/
theorem hyperbola_rectangular_asymptotes (A B C H : K) 
  (hyperbola_passes_through : ∀ (x : K), x = A ∨ x = B ∨ x = C ∨ x = H → K) :
  ∃ (hyperbola : K → K → Prop), 
    ((hyperbola A B) ∧ (hyperbola B C) ∧ (hyperbola C A) ∧ hyperbola H H) 
    → (∀ (x y : K), hyperbola x y → (equation_with_perpendicular_asymptotes)) :=
begin
  sorry,
end

end hyperbola_rectangular_asymptotes_l457_457598


namespace complement_of_M_in_U_l457_457483

def U : Set ℤ := {-1, -2, -3, 0, 1}

def M (a : ℤ) : Set ℤ := {-1, 0, a^2 + 1}

theorem complement_of_M_in_U :
  (∃(a : ℤ), a^2 + 1 = 1) →
  (U \ {x | x ∈ M 0} = {-2, -3}) :=
by
  intro a_exists
  cases a_exists with a ha
  rw set.ext_iff
  intro x
  simp
  have amem : M a = {-1, 0, 1} := by
    rw ha
    ext y
    simp
  sorry

end complement_of_M_in_U_l457_457483


namespace perfect_square_trinomial_k_l457_457862

theorem perfect_square_trinomial_k (a k : ℝ) : (∃ b : ℝ, (a - b)^2 = a^2 - ka + 25) ↔ k = 10 ∨ k = -10 := 
sorry

end perfect_square_trinomial_k_l457_457862


namespace percent_non_union_women_l457_457511

-- Definitions used in the conditions:
def total_employees := 100
def percent_men := 50 / 100
def percent_union := 60 / 100
def percent_union_men := 70 / 100

-- Calculate intermediate values
def num_men := total_employees * percent_men
def num_union := total_employees * percent_union
def num_union_men := num_union * percent_union_men
def num_non_union := total_employees - num_union
def num_non_union_men := num_men - num_union_men
def num_non_union_women := num_non_union - num_non_union_men

-- Statement of the problem in Lean
theorem percent_non_union_women : (num_non_union_women / num_non_union) * 100 = 80 := 
by {
  sorry
}

end percent_non_union_women_l457_457511


namespace exp_13_pi_i_over_2_eq_i_l457_457016

theorem exp_13_pi_i_over_2_eq_i : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_13_pi_i_over_2_eq_i_l457_457016


namespace luke_base_score_per_round_l457_457960

theorem luke_base_score_per_round :
  ∀ (total_points bonus_points penalty_points rounds : ℕ), 
    total_points = 370 → 
    bonus_points = 50 → 
    penalty_points = 30 → 
    rounds = 5 → 
    (total_points - bonus_points + penalty_points) / rounds = 70 :=
by 
  intros total_points bonus_points penalty_points rounds ht hb hp hr 
  rw [ht, hb, hp, hr]
  -- Remaining are arithmetic simplifications that the proof will handle
  sorry

end luke_base_score_per_round_l457_457960


namespace shortest_side_correct_l457_457877

noncomputable def shortest_side_of_triangle (BD DE EC : ℝ) (AD_bisects : Prop) : ℝ :=
  if BD = 3 ∧ DE = 6 ∧ EC = 9 ∧ AD_bisects
  then (3 * sqrt 3) / 2
  else 0

theorem shortest_side_correct :
  shortest_side_of_triangle 3 6 9 True = (3 * sqrt 3) / 2 := 
by
  sorry

end shortest_side_correct_l457_457877


namespace multiply_and_simplify_l457_457967
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l457_457967


namespace valid_parametrizations_l457_457295

def parametric_forms (t : ℝ) :=
  [((3 : ℝ), 12, -1, -2),
   (-3, 0, 2, 3),
   (0, 6, 1, 2),
   (1, 8, 2, 5),
   (-6, -6, 1, 2)]

def line_eq (x y : ℝ) : Prop := y = 2 * x + 6

def is_parametrization_valid (x₀ y₀ a b : ℝ) : Prop :=
  ∀ t : ℝ, line_eq (x₀ + t * a) (y₀ + t * b)

theorem valid_parametrizations :
  ∀ t, ∃ l, (l ∈ [0, 2, 4].to_finset) ∧
  (is_parametrization_valid (nth parametric_forms t).1 (nth parametric_forms t).2 (nth parametric_forms t).3 (nth parametric_forms t).4) :=
by { sorry }

end valid_parametrizations_l457_457295


namespace area_covered_by_congruent_rectangles_l457_457951

-- Definitions of conditions
def length_AB : ℕ := 12
def width_AD : ℕ := 8
def area_rect (l w : ℕ) : ℕ := l * w

-- Center of the first rectangle
def center_ABCD : ℕ × ℕ := (length_AB / 2, width_AD / 2)

-- Proof statement
theorem area_covered_by_congruent_rectangles 
  (length_ABCD length_EFGH width_ABCD width_EFGH : ℕ)
  (congruent : length_ABCD = length_EFGH ∧ width_ABCD = width_EFGH)
  (center_E : ℕ × ℕ)
  (H_center_E : center_E = center_ABCD) :
  area_rect length_ABCD width_ABCD + area_rect length_EFGH width_EFGH - length_ABCD * width_ABCD / 2 = 168 := by
  sorry

end area_covered_by_congruent_rectangles_l457_457951


namespace train_crossing_time_correct_l457_457151

noncomputable def train_crossing_time : ℕ → ℕ → ℕ → Float :=
  λ (train_length bridge_length train_speed : ℕ) =>
    let total_distance := train_length + bridge_length
    let speed_mps := (train_speed * 1000) / 3600.to_float
    (total_distance.to_float / speed_mps)

theorem train_crossing_time_correct :
  train_crossing_time 100 135 75 ≈ 11.28 :=
by
  sorry

end train_crossing_time_correct_l457_457151


namespace chessboard_tiling_impossible_l457_457506

theorem chessboard_tiling_impossible :
  ¬ ∃ (cover : (Fin 5 × Fin 7 → Prop)), 
    (cover (0, 3) = false) ∧
    (∀ i j, (cover (i, j) → cover (i + 1, j) ∨ cover (i, j + 1)) ∧
             ∀ x y z w, cover (x, y) → cover (z, w) → (x ≠ z ∨ y ≠ w)) :=
sorry

end chessboard_tiling_impossible_l457_457506


namespace xy_diff_square_l457_457867

theorem xy_diff_square (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 :=
by
  sorry

end xy_diff_square_l457_457867


namespace max_gcd_dn_l457_457390

def a (n : ℕ) := 101 + n^2

def d (n : ℕ) := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_dn : ∃ n : ℕ, ∀ m : ℕ, d m ≤ 3 := sorry

end max_gcd_dn_l457_457390


namespace no_integer_regular_pentagon_l457_457901

theorem no_integer_regular_pentagon 
  (x y : Fin 5 → ℤ) 
  (h_length : ∀ i j : Fin 5, i ≠ j → (x i - x j) ^ 2 + (y i - y j) ^ 2 = (x 0 - x 1) ^ 2 + (y 0 - y 1) ^ 2)
  : False :=
sorry

end no_integer_regular_pentagon_l457_457901


namespace rotated_square_shaded_area_l457_457743

theorem rotated_square_shaded_area (s : ℝ) (α : ℝ) 
  (h₁ : s = 1) (h₂ : α = π / 6) :
  let shaded_area := s^2 - 2 * (1 / 2 * s * (s * real.sin α)) in
  shaded_area = 1 - real.sqrt 3 / 3 :=
by {
  unfold shaded_area,
  rw [h₁, h₂, real.sin_pi_div_six],
  norm_num,
  sorry
}

end rotated_square_shaded_area_l457_457743


namespace problem_l457_457864

noncomputable def x : ℕ := 5  -- Define x as the positive integer 5

theorem problem (hx : ∀ x, 1 ≤ x → 1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170 ↔ x = 5) : 1^(5+2) + 2^(5+1) + 3^(5-1) + 4^5 = 1170 :=
by {
  have : 1^(5+2) + 2^(5+1) + 3^(5-1) + 4^5 = 1^7 + 2^6 + 3^4 + 4^5 := by rfl,
  rw [this],
  norm_num,
}

end problem_l457_457864


namespace canoe_rental_cost_l457_457318

theorem canoe_rental_cost :
  ∃ (C : ℕ) (K : ℕ), 
  (15 * K + C * (K + 4) = 288) ∧ 
  (3 * K + 12 = 12 * C) ∧ 
  (C = 14) :=
sorry

end canoe_rental_cost_l457_457318


namespace matrix_equation_l457_457948

-- Definitions from conditions
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![ -1, 4], ![ -6, 3]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1  -- Identity matrix

-- Given calculation of N^2
def N_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![ -23, 8], ![ -12, -15]]

-- Goal: prove that N^2 = r*N + s*I for r = 2 and s = -21
theorem matrix_equation (r s : ℤ) (h_r : r = 2) (h_s : s = -21) : N_squared = r • N + s • I := by
  sorry

end matrix_equation_l457_457948


namespace product_inequality_l457_457538

-- Assumptions and Sequences definitions
def a1 := 2
def b1 := 1

noncomputable def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := n

def c (n : ℕ) : ℕ := (b n) ^ 2 / (a n)
def T (n : ℕ) : ℕ := (Finset.range n).prod c

-- Theorem to be proved
theorem product_inequality (n : ℕ) : T n ≤ 9 / 16 := 
sorry

end product_inequality_l457_457538


namespace angle_BEC_is_140_l457_457183

-- Lean definitions for conditions
def quadrilateral (A B C D : Type) := true

def bisects (θ : ℝ) (bisector_angle : ℝ) : Prop := bisector_angle = θ / 2

def exterior_angle (interior : ℝ) (exterior : ℝ) : Prop := interior = 180 - exterior

noncomputable def measure_BEC (A B C D E : Type)
  (quadrilateral A B C D)
  (bisects_angle_B : bisects 50 25)
  (bisects_angle_C : bisects 30 15)
  (exterior_angle_B : exterior_angle 50 130)
  (exterior_angle_C : exterior_angle 30 150)
  : ℝ := 140

theorem angle_BEC_is_140 (A B C D E : Type)
  (quadrilateral A B C D)
  (bisects_angle_B : bisects 50 25)
  (bisects_angle_C : bisects 30 15)
  (exterior_angle_B : exterior_angle 50 130)
  (exterior_angle_C : exterior_angle 30 150)
  : measure_BEC A B C D E quadrilateral bisects_angle_B bisects_angle_C exterior_angle_B exterior_angle_C = 140 := 
sorry

end angle_BEC_is_140_l457_457183


namespace Lorelei_vase_contains_22_roses_l457_457199

variable (redBush : ℕ) (pinkBush : ℕ) (yellowBush : ℕ) (orangeBush : ℕ)
variable (percentRed : ℚ) (percentPink : ℚ) (percentYellow : ℚ) (percentOrange : ℚ)

noncomputable def pickedRoses : ℕ :=
  let redPicked := redBush * percentRed
  let pinkPicked := pinkBush * percentPink
  let yellowPicked := yellowBush * percentYellow
  let orangePicked := orangeBush * percentOrange
  (redPicked + pinkPicked + yellowPicked + orangePicked).toNat

theorem Lorelei_vase_contains_22_roses 
  (redBush := 12) (pinkBush := 18) (yellowBush := 20) (orangeBush := 8)
  (percentRed := 0.5) (percentPink := 0.5) (percentYellow := 0.25) (percentOrange := 0.25)
  : pickedRoses redBush pinkBush yellowBush orangeBush percentRed percentPink percentYellow percentOrange = 22 := by 
  sorry

end Lorelei_vase_contains_22_roses_l457_457199


namespace cos_18_degree_eq_l457_457000

noncomputable def cos_18_deg : ℝ :=
  let y := (cos (real.pi / 10))           -- 18 degrees in radians
  let x := (cos (2 * real.pi / 10))       -- 36 degrees in radians
  have h1 : x = 2 * y ^ 2 - 1,
  by sorry,  -- Double angle formula
  have h2 : cos (3 * real.pi / 10) = (cos (π / 2 - 2 * real.pi / 10)),
  by sorry,  -- Triple angle formula for sine
  have h3 : cos (3 * real.pi / 10) = 4 * cos (real.pi / 10)^3 - 3 * cos (real.pi / 10),
  by sorry,  -- Triple angle formula for cosine
  have h4 : cos (π / 2 - 2 * real.pi / 10) = sin (2 * real.pi / 10),
  by sorry,  -- Cosine of complementary angle
  show y = (1 + real.sqrt 5) / 4,
  by sorry

theorem cos_18_degree_eq : cos_18_deg = (1 + real.sqrt 5) / 4 :=
by sorry

end cos_18_degree_eq_l457_457000


namespace hypotenuse_length_l457_457886

variables (a b c : ℝ)

-- Definitions from conditions
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def sum_of_squares_is_2000 (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 2000

def perimeter_is_60 (a b c : ℝ) : Prop :=
  a + b + c = 60

theorem hypotenuse_length (a b c : ℝ)
  (h1 : right_angled_triangle a b c)
  (h2 : sum_of_squares_is_2000 a b c)
  (h3 : perimeter_is_60 a b c) :
  c = 10 * Real.sqrt 10 :=
sorry

end hypotenuse_length_l457_457886


namespace arrangement_of_A_B_C_D_E_l457_457732

theorem arrangement_of_A_B_C_D_E (A B C D E : Type) :
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ E) ∧ (E ≠ A) ∧
  (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (C ≠ E) →
  (∃ (permutation : Set (List Type)),
    (∀ perm ∈ permutation, perm = [B, A] ∨ perm.head ≠ A ∨ perm.tail ≠ B) ∧
    (permutation.card = 24)) := 
  sorry

end arrangement_of_A_B_C_D_E_l457_457732


namespace pentagon_reflection_rotation_l457_457681

/-- Reflecting a regular pentagon in line PQ and then rotating it clockwise by 144 degrees results in moving two vertex positions clockwise --/
theorem pentagon_reflection_rotation (PQ : Line) :
  ∀ (p : RegularPentagon), rotate (reflect PQ p) 144 = move_two_vertices_clockwise p :=
sorry

end pentagon_reflection_rotation_l457_457681


namespace range_of_a_l457_457248

variable (a b c : ℝ)

def condition1 := a^2 - b * c - 8 * a + 7 = 0

def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
  sorry

end range_of_a_l457_457248


namespace minimum_m_n_1978_l457_457816

-- Define the conditions given in the problem
variables (m n : ℕ) (h1 : n > m) (h2 : m > 1)
-- Define the condition that the last three digits of 1978^m and 1978^n are identical
def same_last_three_digits (a b : ℕ) : Prop :=
  (a % 1000 = b % 1000)

-- Define the problem statement: under the conditions, prove that m + n = 106 when minimized
theorem minimum_m_n_1978 (h : same_last_three_digits (1978^m) (1978^n)) : m + n = 106 :=
sorry   -- Proof will be provided here

end minimum_m_n_1978_l457_457816


namespace calculate_value_l457_457752

theorem calculate_value : (3^3 * 4^3) + (3^3 * 2^3) = 1944 := by
  calc
  sorry

end calculate_value_l457_457752


namespace quadratic_root_inverse_sq_diff_quadratic_root_inverse_sqrt_diff_l457_457086

theorem quadratic_root_inverse_sq_diff (x1 x2 : ℝ) (hx1ltx2 : x1 < x2)
  (hroots : x1^2 - 8 * x1 + 4 = 0 ∧ x2^2 - 8 * x2 + 4 = 0) :
  x1⁻² - x2⁻² = Real.sqrt 3 :=
sorry

theorem quadratic_root_inverse_sqrt_diff (x1 x2 : ℝ) (hx1ltx2 : x1 < x2)
  (hroots : x1^2 - 8 * x1 + 4 = 0 ∧ x2^2 - 8 * x2 + 4 = 0) :
  x1⁻(1/2) - x2⁻(1/2) = 0 :=
sorry

end quadratic_root_inverse_sq_diff_quadratic_root_inverse_sqrt_diff_l457_457086


namespace coin_stack_arrangements_l457_457271

theorem coin_stack_arrangements :
  ∃ n : ℕ, n = 233 ∧ 
    ∀ (gold_coins silver_coins : ℕ), 
    gold_coins = 5 → silver_coins = 5 → 
    ∃ arrangements : ℕ, arrangements = n ∧ 
      ∀ (arrangement : list (ℕ × bool)), 
        arrangement.length = 10 ∧ 
        ∃ gold_positions silver_positions : list ℕ,
          gold_positions.length = 5 ∧
          silver_positions.length = 5 ∧
          ∀ i < 5, arrangement.nth (gold_positions.nth i).iget = (1, true) ∧
                    arrangement.nth (silver_positions.nth i).iget = (0, true) ∧
                    ∀ j, j < 5 - 1 → (gold_positions.nth j).iget + 1 = (silver_positions.nth j).iget ∧ 
                                          (silver_positions.nth j).iget + 1 = (gold_positions.nth (j + 1)).iget :=
  sorry

end coin_stack_arrangements_l457_457271


namespace percentage_saved_on_dress_l457_457718

theorem percentage_saved_on_dress {saved spent : ℝ} (h_saved : saved = 3) (h_spent : spent = 30) :
  let original_price := spent + saved
  let percentage_saved := (saved / original_price) * 100
  percentage_saved ≈ 9 :=
by
  let original_price := spent + saved
  let percentage_saved := (saved / original_price) * 100
  have h1: saved = 3 := h_saved
  have h2: spent = 30 := h_spent
  rw [h1, h2]
  calc
    (3 / (30 + 3)) * 100 = (3 / 33) * 100 : by rw [add_comm, add_assoc]
                     ... ≈ 9 : by norm_num; sorry


end percentage_saved_on_dress_l457_457718


namespace find_number_type_l457_457303

-- Definitions of the problem conditions
def consecutive (a b c d : ℤ) : Prop := (b = a + 2) ∧ (c = a + 4) ∧ (d = a + 6)
def sum_is_52 (a b c d : ℤ) : Prop := a + b + c + d = 52
def third_number_is_14 (c : ℤ) : Prop := c = 14

-- The proof problem statement
theorem find_number_type (a b c d : ℤ) 
                         (h1 : consecutive a b c d) 
                         (h2 : sum_is_52 a b c d) 
                         (h3 : third_number_is_14 c) :
  (∃ (k : ℤ), a = 2 * k ∧ b = 2 * k + 2 ∧ c = 2 * k + 4 ∧ d = 2 * k + 6) 
  := sorry

end find_number_type_l457_457303


namespace skyscraper_max_floors_skyscraper_feasibility_l457_457361

theorem skyscraper_max_floors (elevators floors : ℕ) (stops_per_elevator : ℕ) 
  (h_elevators : elevators = 7)
  (h_stops_per_elevator : stops_per_elevator = 6) :
  ∃ (floors ≤ 14), ∀ (i j : ℕ), (i ≠ j) → ∃ (e : ℕ), (e < elevators) ∧ (i, j ∈ stops_per_elevator) := sorry

theorem skyscraper_feasibility (elevators floors stops_per_floor : ℕ) 
  (h_elevators : elevators = 7)
  (h_floors : floors = 14) 
  (h_stops_per_floor : stops_per_floor = 3) :
  ∃ (f : ℕ → list ℕ), (∀ i, length (f i) = stops_per_floor) ∧ 
    (∀ (i j : ℕ), (i ≠ j) → (∃ e, e ∈ (f i) ∧ e ∈ (f j))) := sorry

end skyscraper_max_floors_skyscraper_feasibility_l457_457361


namespace number_of_sheep_l457_457641

-- Define the conditions as given in the problem
variables (S H : ℕ)
axiom ratio_condition : S * 7 = H * 3
axiom food_condition : H * 230 = 12880

-- The theorem to prove
theorem number_of_sheep : S = 24 :=
by sorry

end number_of_sheep_l457_457641


namespace piravena_total_distance_l457_457594

def distance_traveled (side_length : ℕ) : Nat :=
  4 * side_length

theorem piravena_total_distance :
  distance_traveled 2000 = 8000 := 
by
  simp [distance_traveled]
  exact rfl

end piravena_total_distance_l457_457594


namespace perpendicular_line_plane_to_planes_perpendicular_l457_457935

variables {Point : Type} [MetricSpace Point]
variables (Plane : Type) [MetricSpace Plane]
variables (Line : Type) [MetricSpace Line] 
variables (α β : Plane) (l m : Line)

-- Definitions of line contained in a plane.
def line_in_plane (l : Line) (α : Plane) : Prop := -- Define this relation properly as per the geometric framework
  sorry 

-- Definitions of perpendicular line and plane.
def line_perpendicular_plane (l : Line) (β : Plane) : Prop := -- Define this relation properly as per the geometric framework
  sorry 

-- Definitions of perpendicular planes.
def planes_perpendicular (α β : Plane) : Prop := -- Define this relation properly as per the geometric framework
  sorry 

-- Theorem statement to be proven
theorem perpendicular_line_plane_to_planes_perpendicular
  (h1 : line_in_plane l α)
  (h2 : line_perpendicular_plane l β) :
  planes_perpendicular α β :=
sorry

end perpendicular_line_plane_to_planes_perpendicular_l457_457935


namespace oranges_cost_lunks_l457_457492

def lunk_to_kunks (lunks : ℕ) : ℕ :=
  (7 * lunks) / 4

def kunks_to_oranges (kunks : ℕ) : ℕ :=
  (3 * kunks) / 5

def oranges_to_kunks (oranges : ℕ) : ℕ :=
  (5 * oranges) / 3

def kunks_to_lunks (kunks : ℕ) : ℕ :=
  (4 * kunks) / 7

def required_lunks (oranges : ℕ) : ℕ :=
  let kunks_needed := oranges_to_kunks oranges
  let lunks_needed := kunks_to_lunks kunks_needed
  lunks_needed

theorem oranges_cost_lunks (oranges : ℕ) : oranges = 20 → required_lunks oranges = 21 :=
by
  intro h
  calc
    required_lunks 20 = 21 : sorry

end oranges_cost_lunks_l457_457492


namespace gain_percent_l457_457693

-- Define cost price and selling price
variables (C S : ℝ)

-- Define the condition that cost price of 44 chocolates is equal to the selling price of 24 chocolates
def condition : Prop := 44 * C = 24 * S

-- Define the gain percent calculation based on the given condition
theorem gain_percent (h : condition C S) : 100 * ((S - C) / C) = 500 / 6 :=
by
  -- Introduce the gain percent
  let gain := S - C
  let gain_percent := 100 * (gain / C)
  -- Obtain the formula for S in terms of C
  have : S = 44 * C / 24, from sorry,
  -- Substitute S in terms of C and simplify
  calc
    gain_percent 
      = 100 * ((44 * C / 24 - C) / C) : by rw this
  ... = 500 / 6 : sorry
  -- Hence proved

end gain_percent_l457_457693


namespace two_digit_integers_mod_9_eq_3_l457_457155

theorem two_digit_integers_mod_9_eq_3 :
  { x : ℕ | 10 ≤ x ∧ x < 100 ∧ x % 9 = 3 }.finite.card = 10 :=
by sorry

end two_digit_integers_mod_9_eq_3_l457_457155


namespace remainder_of_sum_l457_457964

theorem remainder_of_sum (a b : ℤ) (k m : ℤ)
  (h1 : a = 84 * k + 78)
  (h2 : b = 120 * m + 114) :
  (a + b) % 42 = 24 :=
  sorry

end remainder_of_sum_l457_457964


namespace polygon_edges_of_set_S_l457_457243

variable (a : ℝ)

def in_set_S(x y : ℝ) : Prop :=
  (a / 2 ≤ x ∧ x ≤ 2 * a) ∧
  (a / 2 ≤ y ∧ y ≤ 2 * a) ∧
  (x + y ≥ a) ∧
  (x + a ≥ y) ∧
  (y + a ≥ x)

theorem polygon_edges_of_set_S (a : ℝ) (h : 0 < a) :
  (∃ n, ∀ x y, in_set_S a x y → n = 6) :=
sorry

end polygon_edges_of_set_S_l457_457243


namespace centroid_locus_is_parallel_segment_l457_457813

-- Definitions of points A, B, C, D
variable (A B C D : Point)
-- Definitions of the geometric configuration and conditions
variable (h_angle : is_angle_vertex A B C D)
variable (h_circle : passes_through A B C D)
variable (h_cd_diff : C ≠ A ∧ D ≠ A)

-- Definitions of points K and L based on the geometric properties
variable (K : Point)
variable (h_k_fixed : is_fixed_point K A B C)
variable (L : Point)
variable (h_l_fixed : is_fixed_point L A B D)

-- Definition of point N (centroids) and its geometric relationship
variable (N : Point)
variable (h_n_geometric : on_line_segment L K N)

-- The desired result expressed in Lean 4
theorem centroid_locus_is_parallel_segment (A B C D : Point) 
  (h_angle : is_angle_vertex A B C D)
  (h_circle : passes_through A B C D)
  (h_cd_diff : C ≠ A ∧ D ≠ A)
  (K L N : Point)
  (h_k_fixed : is_fixed_point K A B C)
  (h_l_fixed : is_fixed_point L A B D)
  (h_n_geometric : on_line_segment L K N)
  :
  centroid_line_segment A K = {
    parallel to segment L K
    divides segment A K in the ratio 2:1
  } := 
begin
  sorry
end

end centroid_locus_is_parallel_segment_l457_457813


namespace gecko_egg_problem_l457_457716

theorem gecko_egg_problem (total_eggs : ℕ) (infertile_percent : ℕ) (hatched_eggs : ℕ) (fertile_percent : ℕ) 
  (h1 : total_eggs = 30) (h2 : infertile_percent = 20) (h3 : hatched_eggs = 16) (h4 : fertile_percent = 80) :
  let fertile_eggs := total_eggs * fertile_percent / 100
  in (fertile_eggs - hatched_eggs) / fertile_eggs = 1 / 3 :=
by
  sorry

end gecko_egg_problem_l457_457716


namespace net_effect_sale_value_l457_457875

theorem net_effect_sale_value (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) :
  let new_price := 0.78 * P in
  let new_qty := 1.86 * Q in
  let original_sale_value := P * Q in
  let new_sale_value := new_price * new_qty in
  ((new_sale_value - original_sale_value) / original_sale_value) * 100 = 45.18 :=
by
  let new_price := 0.78 * P
  let new_qty := 1.86 * Q
  let original_sale_value := P * Q
  let new_sale_value := new_price * new_qty
  sorry

end net_effect_sale_value_l457_457875


namespace multiplication_identity_multiplication_l457_457991

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l457_457991


namespace cos_angle_C_l457_457262

theorem cos_angle_C (A B C D : Type) (h1 : B ≠ C) (h2 : D ∈ line B C)
  (h3 : ∠(A, B, D) = 50 * π / 180) (h4 : ∠(A, C, D) = 20 * π / 180) (h5 : dist A D = dist B D) :
  real.cos (∠(A, B, C)) = real.sqrt 3 / 2 :=
sorry

end cos_angle_C_l457_457262


namespace asymptotes_of_hyperbola_l457_457625

-- Definitions for the hyperbola and the asymptotes
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1
def asymptote_equation (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = - (Real.sqrt 2 / 2) * x

-- The theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) (h : hyperbola_equation x y) :
  asymptote_equation x y :=
sorry

end asymptotes_of_hyperbola_l457_457625


namespace diagonals_of_eight_sided_polygon_l457_457363

-- Define the problem and its conditions
def is_convex (polygon : Polygon) : Prop := sorry

def has_sides (polygon : Polygon) (n : ℕ) : Prop := sorry

def has_right_angle (polygon : Polygon) : Prop := sorry

def has_two_equal_sides (polygon : Polygon) : Prop := sorry

-- Define the polygon
noncomputable def P : Polygon := sorry

-- Define the properties of the polygon
axiom convex_P : is_convex P
axiom sides_P : has_sides P 8
axiom right_angle_P : has_right_angle P
axiom two_equal_sides_P : has_two_equal_sides P

-- Define the theorem based on these properties
theorem diagonals_of_eight_sided_polygon (P : Polygon) (h1 : is_convex P) (h2 : has_sides P 8) (h3 : has_right_angle P) (h4 : has_two_equal_sides P) : 
  number_of_diagonals P = 20 := sorry

end diagonals_of_eight_sided_polygon_l457_457363


namespace train_length_l457_457730

theorem train_length (time : ℝ) (bridge_length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (distance : ℝ) : 
  time = 31.99744020478362 ∧ bridge_length = 200 ∧ speed_kmph = 36 ∧ speed_mps = (speed_kmph * 1000 / 3600) ∧ distance = (speed_mps * time) 
  → distance - bridge_length = 119.9744020478362 := 
begin 
  sorry 
end

end train_length_l457_457730


namespace boys_in_camp_l457_457880

theorem boys_in_camp (
  (pA pB pC : ℕ → ℕ → Prop)
  (sA : ℕ)
  (sB : ℕ)
  (sC : ℕ)
  (nA nB nC : ℤ)
  (propA : pA sA nA)
  (propB : pB sB nB)
  (propC : pC sC nC)
  (cond1 : ∀ T : ℕ, pA 20 0.7 → sA = 56)
  (cond2 : ∀ T : ℕ, pB 30 0.6 → sB = 66)
  (cond3 : ∀ T : ℕ, pC 50 0.5 → sC = 80)
) : ∃ T : ℕ, T = 400 :=
by
  use 400
  sorry

end boys_in_camp_l457_457880


namespace circumradius_half_inradius_l457_457463

-- Define the geometry entities and conditions
variables (A B C I O1 O2 O3 A' B' C' : Type)

-- Assuming given conditions
axiom incenter_of_triangle (ABC I : Type) : 
  ∃ (incenter : I), ∀ (incircle : ∑ (r : ℝ), circles.tangent_to_triangle ABC r)

axiom circles_pass_through_points_and_intersect_orthogonally (O1 O2 O3 I : Type) : 
  ∀ (circle_O1 : circle (B, C)), 
  ∀ (circle_O2 : circle (A, C)), 
  ∀ (circle_O3 : circle (A, B)), 
  (circle_O1 ⊥ incircle I) ∧ (circle_O2 ⊥ incircle I) ∧ (circle_O3 ⊥ incircle I)

axiom circle_intersection_points (O1 O2 O3 : Type) :
  ∃ C' (intersection_point : C'), 
  ∀ A' (intersection_point : A'),
  ∀ B' (intersection_point : B'), 
  (intersection_point ∈ (∩ (circle O1, circle O2))) ∧
  (intersection_point ∈ (∩ (circle O1, circle O3))) ∧
  (intersection_point ∈ (∩ (circle O2, circle O3)))

-- Prove the required
theorem circumradius_half_inradius (r : ℝ) :
  ∃ (radius : ℝ), radius = r / 2 :=
begin
  unfold incenter_of_triangle,
  unfold circles_pass_through_points_and_intersect_orthogonally,
  unfold circle_intersection_points,
  sorry, -- Proof not required
end

end circumradius_half_inradius_l457_457463


namespace correct_propositions_l457_457824

variables {l m : ℝ → ℝ → Prop} {α β : set (ℝ × ℝ × ℝ)}

-- Conditions given
def line_perpendicular_plane (l : ℝ → ℝ → Prop) (α : set (ℝ × ℝ × ℝ)) : Prop := sorry
def line_parallel_plane (m : ℝ → ℝ → Prop) (β : set (ℝ × ℝ × ℝ)) : Prop := sorry
def plane_parallel (α β : set (ℝ × ℝ × ℝ)) : Prop := sorry
def plane_perpendicular (α β : set (ℝ × ℝ × ℝ)) : Prop := sorry
def line_parallel (l m : ℝ → ℝ → Prop) : Prop := sorry
def line_perpendicular (l m : ℝ → ℝ → Prop) : Prop := sorry

-- Given conditions from the problem.
axiom h1 : line_perpendicular_plane l α
axiom h2 : line_parallel_plane m β

-- Proposition ①: α ∥ β → l ⟂ m
def prop1 (α β : set (ℝ × ℝ × ℝ)) (l m : ℝ → ℝ → Prop) : Prop :=
  plane_parallel α β → line_perpendicular l m

-- Proposition ②: α ⟂ β → l ∥ m
def prop2 (α β : set (ℝ × ℝ × ℝ)) (l m : ℝ → ℝ → Prop) : Prop :=
  plane_perpendicular α β → line_parallel l m

-- Proposition ③: l ∥ m → α ⟂ β
def prop3 (α β : set (ℝ × ℝ × ℝ)) (l m : ℝ → ℝ → Prop) : Prop :=
  line_parallel l m → plane_perpendicular α β

-- Proposition ④: l ⟂ m → α ∥ β
def prop4 (α β : set (ℝ × ℝ × ℝ)) (l m : ℝ → ℝ → Prop) : Prop :=
  line_perpendicular l m → plane_parallel α β

theorem correct_propositions :
  prop1 α β l m ∧ prop3 α β l m ∧ ¬ prop2 α β l m ∧ ¬ prop4 α β l m :=
sorry

end correct_propositions_l457_457824


namespace prove_f_values_l457_457099

noncomputable def f (x : ℝ) : ℝ :=
  if h : x - 2 ≥ 0 then Real.sin (x - 2) else Real.log (2 : ℝ → ℝ) (- (x - 2))

theorem prove_f_values :
  f (21 * Real.pi / 4 + 2) * f (-14) = 2 * Real.sqrt 2 :=
by
  sorry

end prove_f_values_l457_457099


namespace eq_a_given_intersection_l457_457165

theorem eq_a_given_intersection (a : ℝ) (A B : Set ℝ)
  (hA : A = {a^2, a + 1, -3})
  (hB : B = {a - 3, 2a - 1, a^2 + 1})
  (h_inter : A ∩ B = {-3}) :
  a = -1 := sorry

end eq_a_given_intersection_l457_457165


namespace symmetric_angles_l457_457502

theorem symmetric_angles (α β : ℝ) (k : ℤ) (h : α + β = 2 * k * Real.pi) : α = 2 * k * Real.pi - β :=
by
  sorry

end symmetric_angles_l457_457502


namespace pure_imaginary_condition_l457_457939

variable (a b : ℝ)

def is_pure_imaginary (Z : ℂ) : Prop :=
  ∃ b : ℝ, Z = 0 + b * complex.i

theorem pure_imaginary_condition (a b : ℝ) :
  (a = 0 → is_pure_imaginary ⟨a, b⟩) ∧ (is_pure_imaginary ⟨a, b⟩ → a = 0) :=
begin
  sorry -- proof goes here
end

end pure_imaginary_condition_l457_457939


namespace dice_probability_sum_18_l457_457327

theorem dice_probability_sum_18 :
  let num_ways := nat.choose 17 7 in
  num_ways = 19448 ∧ 
  (num_ways / (6 ^ 8) = 19448 / (6 ^ 8)) :=
by
  sorry

end dice_probability_sum_18_l457_457327


namespace june_walked_miles_l457_457596

theorem june_walked_miles
  (step_counter_reset : ℕ)
  (resets_per_year : ℕ)
  (final_steps : ℕ)
  (steps_per_mile : ℕ)
  (h1 : step_counter_reset = 100000)
  (h2 : resets_per_year = 52)
  (h3 : final_steps = 30000)
  (h4 : steps_per_mile = 2000) :
  (resets_per_year * step_counter_reset + final_steps) / steps_per_mile = 2615 := 
by 
  sorry

end june_walked_miles_l457_457596


namespace modulus_of_z_l457_457820

noncomputable def z_value (z : ℂ) : Prop :=
  (conj z / (1 + complex.I) = (2 + complex.I))

theorem modulus_of_z {z : ℂ} (h : z_value z) : complex.abs z = real.sqrt 10 :=
begin
  sorry
end

end modulus_of_z_l457_457820


namespace combinatorial_identity_l457_457240

open Real

theorem combinatorial_identity
  (m n : ℕ) (hm : m < n) (j : ℕ) (hj : j < m) :
  ∑ t in Finset.range (n // m + 1), Nat.choose n (m * t + j) =
      (2 ^ n / m : ℝ) * (1 + 2 * ∑ k in Finset.range ((m - 1) / 2 + 1), 
          (cos (k * π / m)) ^ n * cos ((n - 2 * j) * k * π / m)) :=
by
  sorry

end combinatorial_identity_l457_457240


namespace nancy_insurance_payments_l457_457997

-- Definitions for the conditions
def monthly_cost_A := 120
def discount_A := 0.10
def decrease_A := 0.05

def monthly_cost_B := 90
def discount_B := 0.05

def monthly_cost_C := 60
def discount_C := 0.15
def increase_C := 0.02

-- Definition of Nancy's contribution percentage
def nancys_contribution := 0.40

-- Correct answers
def annual_payment_A := 518.40
def annual_payment_B := 410.40
def annual_payment_C := 244.80

-- Statement of the theorem
theorem nancy_insurance_payments :
  let cost_A := monthly_cost_A * (1 - discount_A) * 12 * nancys_contribution in
  let cost_B := monthly_cost_B * (1 - discount_B) * 12 * nancys_contribution in
  let cost_C := monthly_cost_C * (1 - discount_C) * 12 * nancys_contribution in
  cost_A = annual_payment_A ∧ cost_B = annual_payment_B ∧ cost_C = annual_payment_C :=
by
  sorry

end nancy_insurance_payments_l457_457997


namespace decreasing_function_l457_457374

open Real

-- Define the functions
def f_A (x : ℝ) : ℝ := 1 / (x - 1)
def f_B (x : ℝ) : ℝ := 2^(x - 1)
def f_C (x : ℝ) : ℝ := sqrt (x - 1)
def f_D (x : ℝ) : ℝ := log (x - 1)

-- Define monotonicity predicates for the interval (1, +∞)
def is_decreasing (f : ℝ → ℝ) (s : set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

-- Main statement to prove
theorem decreasing_function :
  is_decreasing f_A (Ioi 1) ∧ ¬ is_decreasing f_B (Ioi 1) ∧ ¬ is_decreasing f_C (Ioi 1) ∧ ¬ is_decreasing f_D (Ioi 1) :=
by sorry

end decreasing_function_l457_457374


namespace total_amount_spent_l457_457702

theorem total_amount_spent (n : ℕ) (d : ℝ) (p : ℝ) (h : n = 4) (h1 : d = 0.5) (h2 : p = 20) : 
  n * (p * d) = 40 :=
by 
  conv {
    to_rhs,
    { rw [h, h1, h2] },
  }
  sorry

end total_amount_spent_l457_457702


namespace dist_from_O_to_line_AB_l457_457180

noncomputable def dist_origin_line := 
let A := (4, Real.pi / 6)
let B := (3, 2 * Real.pi / 3) in 
let O := (0, 0) in
let A_cart := (4 * Real.cos (Real.pi / 6), 4 * Real.sin (Real.pi / 6)) in
let B_cart := (3 * Real.cos (2 * Real.pi / 3), 3 * Real.sin (2 * Real.pi / 3)) in
let m := ((B_cart.2 - A_cart.2) / (B_cart.1 - A_cart.1)) in
let c := A_cart.2 - m * A_cart.1 in
let line_eq := λ x, m * x + c in
Real.dist_origin_to_line O (λ x y, y - m * x - c)

theorem dist_from_O_to_line_AB : dist_origin_line = 7.95 := 
sorry

end dist_from_O_to_line_AB_l457_457180


namespace coefficient_of_x9_in_expansion_l457_457674

-- Definitions as given in the problem
def binomial_expansion_coeff (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * a^(n - k) * b^k

-- Mathematically equivalent statement in Lean 4
theorem coefficient_of_x9_in_expansion : binomial_expansion_coeff 10 9 (-2) 1 = -20 :=
by
  sorry

end coefficient_of_x9_in_expansion_l457_457674


namespace arithmetic_geometric_sequence_l457_457514

-- Define the arithmetic sequence {a_n} with given conditions
def arith_seq (a : ℕ → ℕ) : Prop :=
  a 2 = 6 ∧ a 3 + a 6 = 27 ∧ ∃ d : ℕ, ∀ n : ℕ, a n = a 1 + (n - 1) * d

-- Define the geometric sequence {b_n}
def geom_seq (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = 3 ^ (n - 1)

-- Define the product sequence
noncomputable def prod_seq (a b : ℕ → ℕ) (c : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, c n = a n * b n

-- Define the sum of the first n terms of the product sequence
noncomputable def sum_prod_seq (c : ℕ → ℕ) : ℕ → ℕ
| 0       := 0
| (n + 1) := sum_prod_seq n + c (n + 1)

-- The main statement combining all sub-problems:
theorem arithmetic_geometric_sequence :
  ∃ (a b c : ℕ → ℕ) (T : ℕ → ℕ),
    arith_seq a ∧
    geom_seq b ∧
    (∀ n, c n = a n * b n) ∧
    (∀ n, T n = sum_prod_seq c n)
    ∧ (∀ d, a d = 3 * d)
    ∧ (∀ n, T n = (2 * n + 1) * 3 ^ (n + 1) - 3)/4 :=
sorry

end arithmetic_geometric_sequence_l457_457514


namespace new_midpoint_and_distance_l457_457894

variables {p q r s : ℝ}

def midpoint (x y : ℝ) : ℝ × ℝ :=
(x, y)

def moved_point_P (p q : ℝ) : ℝ × ℝ :=
(p - 3, q + 5)

def moved_point_Q (r s : ℝ) : ℝ × ℝ :=
(r + 4, s - 3)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem new_midpoint_and_distance (p q r s : ℝ):
  let N := midpoint ((p + r) / 2) ((q + s) / 2),
      P' := moved_point_P p q,
      Q' := moved_point_Q r s,
      N' := midpoint ((P'.1 + Q'.1) / 2) ((P'.2 + Q'.2) / 2) in
  N' = midpoint (((p + r) / 2) + (1/2)) (((q + s) / 2) + 1) ∧ 
  distance N N' = real.sqrt (5) / 2 :=
by
  intros
  have midpoint_PQ : N = midpoint ((p + r) / 2) ((q + s) / 2),
  have midpoint_PQ_moved : N' = midpoint ((P'.1 + Q'.1) / 2) ((P'.2 + Q'.2) / 2),
  sorry

end new_midpoint_and_distance_l457_457894


namespace number_of_possible_lengths_l457_457495

theorem number_of_possible_lengths (a b : ℕ) (h₁ : a = 8) (h₂ : b = 12) : 
  (∃ (x : ℕ), 4 < x ∧ x < 20) → 
  finset.card (finset.filter (λ x, 4 < x ∧ x < 20) (finset.range 21)) = 15 := 
by 
  intros h;
  sorry

end number_of_possible_lengths_l457_457495


namespace largest_power_of_2_dividing_product_of_first_50_even_integers_l457_457934

def product_of_first_50_even_integers : ℕ := 
  List.product (List.range 50).map (λ n => 2 * (n + 1))

theorem largest_power_of_2_dividing_product_of_first_50_even_integers :
  ∃ m : ℕ, (2^m ∣ product_of_first_50_even_integers) ∧ (m = 97) :=
by
  sorry

end largest_power_of_2_dividing_product_of_first_50_even_integers_l457_457934


namespace two_odd_functions_l457_457103

def f1 (x : ℝ) := x^3
def f2 (x : ℝ) := Real.tan x
def f3 (x : ℝ) := x * Real.sin x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem two_odd_functions :
  [f1, f2, f3].countp is_odd = 2 := 
sorry

end two_odd_functions_l457_457103


namespace place_rectangles_l457_457231

theorem place_rectangles (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  ∃ (k : ℕ), k = ⌊(m * n) / 6⌋ ∧ ∀ p q : ℕ, (p = 2 ∧ q = 3) → (∃ (a b : ℕ), a * p ≤ m ∧ b * q ≤ n ∧ a * b = k) :=
by
  sorry

end place_rectangles_l457_457231


namespace proof_problem_l457_457614

def is_reciprocal (n : ℕ) := 1 / (n : ℝ)

def stmt_i : Prop := is_reciprocal 4 + is_reciprocal 8 = is_reciprocal 12
def stmt_ii : Prop := is_reciprocal 10 - is_reciprocal 5 = is_reciprocal 7
def stmt_iii : Prop := is_reciprocal 3 * is_reciprocal 9 = is_reciprocal 27
def stmt_iv : Prop := is_reciprocal 15 + is_reciprocal 5 = is_reciprocal 10 + is_reciprocal 10

def number_of_true_statements : ℕ := 
  if stmt_i then 1 else 0 + 
  if stmt_ii then 1 else 0 + 
  if stmt_iii then 1 else 0 + 
  if stmt_iv then 1 else 0

theorem proof_problem : number_of_true_statements = 1 := sorry

end proof_problem_l457_457614


namespace sum_consecutive_integers_80_to_89_l457_457381

theorem sum_consecutive_integers_80_to_89 :
  (Finset.sum (Finset.range (89 - 80 + 1)) (λ i, i + 80)) = 845 :=
by
  sorry

end sum_consecutive_integers_80_to_89_l457_457381


namespace lorelei_vase_rose_count_l457_457203

theorem lorelei_vase_rose_count :
  let r := 12
  let p := 18
  let y := 20
  let o := 8
  let picked_r := 0.5 * r
  let picked_p := 0.5 * p
  let picked_y := 0.25 * y
  let picked_o := 0.25 * o
  picked_r + picked_p + picked_y + picked_o = 22 := 
by
  sorry

end lorelei_vase_rose_count_l457_457203


namespace closest_to_neg_sqrt_2_l457_457585

theorem closest_to_neg_sqrt_2 : ∀ (x : ℝ), x ∈ {-2, -1, 0, 1} → | -√2 - x | ≥ | -√2 - (-1) | :=
by
  intro x hx
  fin_cases hx
  case 1 =>
    sorry
  case 2 =>
    sorry
  case 3 =>
    sorry
  case 4 =>
    sorry

end closest_to_neg_sqrt_2_l457_457585


namespace inequality_proof_l457_457845

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : 
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l457_457845


namespace largest_b_l457_457307

def max_b (a b c : ℕ) : ℕ := b -- Define max_b function which outputs b

theorem largest_b (a b c : ℕ)
  (h1 : a * b * c = 360)
  (h2 : 1 < c)
  (h3 : c < b)
  (h4 : b < a) :
  max_b a b c = 10 :=
sorry

end largest_b_l457_457307


namespace sequence_an_Tn_less_than_one_l457_457846
open Nat

noncomputable def a (n : ℕ) : ℝ := n + 1

noncomputable def b (n : ℕ) : ℝ :=
  (2 * n + 1) / ((a n - 1) ^ 2 * (a (n + 1) - 1) ^ 2)

noncomputable def T (n : ℕ) : ℝ :=
  ∑ k in range(1, n + 1), b k

theorem sequence_an (n : ℕ) (h₁ : 0 < n) :
    ∑ k in range(1, n + 1), k / (a k - 1) = n := by
  sorry

theorem Tn_less_than_one (n : ℕ) (h₁ : 0 < n) :
    T n < 1 := by
  sorry

end sequence_an_Tn_less_than_one_l457_457846


namespace a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l457_457431

noncomputable def a_0 (n : ℕ) : ℕ := 2^n
noncomputable def S_n (n : ℕ) : ℕ := 3^n - 2^n
noncomputable def T_n (n : ℕ) : ℕ := (n - 2) * 2^n + 2 * n^2

theorem a_0_eq_2_pow_n (n : ℕ) (h : n > 0) : a_0 n = 2^n := sorry

theorem S_n_eq_3_pow_n_minus_2_pow_n (n : ℕ) (h : n > 0) : S_n n = 3^n - 2^n := sorry

theorem S_n_magnitude_comparison : 
  ∀ (n : ℕ), 
    (n = 1 → S_n n > T_n n) ∧
    (n = 2 ∨ n = 3 → S_n n < T_n n) ∧
    (n ≥ 4 → S_n n > T_n n) := sorry

end a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l457_457431


namespace percentage_goldfish_at_surface_l457_457313

-- Definitions of conditions
def goldfish_at_surface : ℕ := 15
def goldfish_below_surface : ℕ := 45
def total_goldfish : ℕ := goldfish_at_surface + goldfish_below_surface

-- The theorem to prove
theorem percentage_goldfish_at_surface : (goldfish_at_surface * 100 / total_goldfish) = 25 := 
by {
  have h1 : total_goldfish = 15 + 45 := rfl,
  have h2 : total_goldfish = 60 := by rw h1,
  have h3 : (goldfish_at_surface * 100) = 1500 := rfl,
  have h4 : (goldfish_at_surface * 100 / total_goldfish) = (1500 / 60) := by rw h2,
  have h5 : (1500 / 60) = 25 := rfl,
  rw h4,
  exact h5,
}

end percentage_goldfish_at_surface_l457_457313


namespace prob_two_people_diff_floors_l457_457316

noncomputable def probability_diff_floors : ℚ := 8 / 9

theorem prob_two_people_diff_floors :
  let total_ways := 9 * 9 in
  let different_ways := 9 * 8 in
  (different_ways / total_ways : ℚ) = probability_diff_floors :=
begin
  sorry
end

end prob_two_people_diff_floors_l457_457316


namespace angle_PBA_eq_angle_PDA_l457_457947

open EuclideanGeometry

variables {A B C D P : Point}
hypothesis parallelogram_ABCD : Parallelogram A B C D
hypothesis P_interior : InteriorPoint P A B C D
hypothesis sum_angles_APD_CPB : ∡A P D + ∡C P B = 180

theorem angle_PBA_eq_angle_PDA :
  ∡P B A = ∡P D A :=
by
  -- proof goes here
  sorry

end angle_PBA_eq_angle_PDA_l457_457947


namespace median_of_right_triangle_l457_457520

theorem median_of_right_triangle (PQ QR : ℝ) (h1 : ∠PQR = 90) (h2 : PQ = 5) (h3 : QR = 12) :
  let PR := Real.sqrt (PQ^2 + QR^2),
      PN := PR / 2
  in PN = 6.5 :=
by
  sorry

end median_of_right_triangle_l457_457520


namespace length_of_pivoting_segment_l457_457012

theorem length_of_pivoting_segment (A B C D P : Point) (h_triangle : Triangle A B C)
    (h_D_on_BC : D ∈ LineSegment B C) (h_P_on_AD : P ∈ LineSegment A D) :
    (∃ l : ℝ, l = 0 ∧ ∀ P, P ∈ LineSegment A D → LengthOfSegment P (AB AC) ↑
        ∃ length_max : ℝ, ∀ P, P ∈ LineSegment A D ∧ PerpendicularToAD P →
            length max = LengthOfSegment P (AB AC) ∧ 
            ∀ P, P ∈ LineSegment A D, LengthOfSegment P (AB AC) decreases to 0 as P moves to D) :=
sorry

end length_of_pivoting_segment_l457_457012


namespace solve_for_x_l457_457402

theorem solve_for_x : ∃ x : ℝ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 :=
by
  use 3
  split
  { simp }
  { refl }

end solve_for_x_l457_457402


namespace smallest_x_l457_457026

theorem smallest_x (x : ℝ) (h : 4 * x^2 + 6 * x + 1 = 5) : x = -2 :=
sorry

end smallest_x_l457_457026


namespace speed_of_first_part_l457_457357

theorem speed_of_first_part (v : ℝ) (h1 : v > 0)
  (h_total_distance : 50 = 25 + 25)
  (h_average_speed : 44 = 50 / ((25 / v) + (25 / 33))) :
  v = 66 :=
by sorry

end speed_of_first_part_l457_457357


namespace simplify_expression_l457_457605

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : 
  (2 / y^2 - y⁻¹) = (2 - y) / y^2 :=
by sorry

end simplify_expression_l457_457605


namespace admission_charge_for_adult_l457_457283

theorem admission_charge_for_adult 
(admission_charge_per_child : ℝ)
(total_paid : ℝ)
(children_count : ℕ)
(admission_charge_for_adult : ℝ) :
admission_charge_per_child = 0.75 →
total_paid = 3.25 →
children_count = 3 →
admission_charge_for_adult + admission_charge_per_child * children_count = total_paid →
admission_charge_for_adult = 1.00 :=
by
  intros h1 h2 h3 h4
  sorry

end admission_charge_for_adult_l457_457283


namespace digit_sum_solution_l457_457562

def S (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_solution : S (S (S (S (2017 ^ 2017)))) = 1 := 
by
  sorry

end digit_sum_solution_l457_457562


namespace exception_to_roots_l457_457643

theorem exception_to_roots (x : ℝ) :
    ¬ (∃ x₀, (x₀ ∈ ({x | x = x} ∩ {x | x = x - 2}))) :=
by sorry

end exception_to_roots_l457_457643


namespace solve_tan_alpha_l457_457801

open Real

noncomputable def tan_alpha (α : ℝ) : Prop :=
  tan(α + π/4) = 2 → tan(α) = 1/3

theorem solve_tan_alpha (α : ℝ) : tan_alpha α :=
  by
    intro h
    have h1 : tan(α + π/4) = (tan(α) + 1) / (1 - tan(α)) := by sorry
    rw h at h1
    have h2 : (tan(α) + 1) / (1 - tan(α)) = 2 := by sorry
    have h3 : tan(α) = 1/3 := by sorry
    exact h3

end solve_tan_alpha_l457_457801


namespace partI_partII_1_partII_2_partII_3_partIII_l457_457843

noncomputable def f (a x : ℝ) : ℝ := (-a * x^2 + x - 1) / real.exp x

theorem partI (a : ℝ) (h1 : f a 0 = -1) : 
  ∃ m b, (m = 2) ∧ (b = -1) ∧ ∀ x : ℝ, f a x = m * (x - 0) + b := 
sorry

theorem partII_1 (a : ℝ) (h2 : 0 < a ∧ a < 1/2) : 
  (∀ x : ℝ, (f' a x > 0 ↔ x ∈ Iio 2 ∨ x ∈ Ioi (1/a)) ∧ (f' a x < 0 ↔ x ∈ Ioo 2 (1/a))) :=
sorry

theorem partII_2 (a : ℝ) (h3 : a = 1/2) : 
  ∀ x : ℝ, f' a x ≥ 0 :=
sorry

theorem partII_3 (a : ℝ) (h4 : a > 1/2) : 
  (∀ x : ℝ, (f' a x > 0 ↔ x ∈ Iio (1/a) ∨ x ∈ Ioi 2) ∧ (f' a x < 0 ↔ x ∈ Ioo (1/a) 2)) :=
sorry

theorem partIII (a : ℝ) (h5 : a ≤ -1) : 
  ∀ x : ℝ, f a x ≥ -real.exp 1 :=
sorry

end partI_partII_1_partII_2_partII_3_partIII_l457_457843


namespace distance_OB_l457_457093

theorem distance_OB (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  {P : ℝ × ℝ} (hP : (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :
  ∃ (O F1 F2 : ℝ × ℝ), let e := Real.sqrt (1 - (b^2 / a^2)) in 
  (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ O = (0, 0) ∧ c = a * e) ∧ ∥O - B∥ = a :=
begin
  sorry
end

end distance_OB_l457_457093


namespace min_value_f1_monotonically_decreasing_f2_l457_457475

noncomputable def f1 (x : ℝ) : ℝ := x - log x

theorem min_value_f1 : ∃ x > 0, (f1 x = 1) :=
sorry

noncomputable def f2 (x : ℝ) : ℝ := log x - x^2 + x

theorem monotonically_decreasing_f2 : ∀ x : ℝ, (1 < x) ↔ (deriv f2 x < 0) :=
sorry

end min_value_f1_monotonically_decreasing_f2_l457_457475


namespace rectangle_perimeter_in_right_triangle_l457_457885

theorem rectangle_perimeter_in_right_triangle :
  ∀ (AC BC : ℝ), AC = 6 → BC = 6 →
  ∃ (AM MK KN BN: ℝ) (a : ℝ), a = sqrt(AC^2 + BC^2) / 2 → 
  AC - AM = AM ∧ BC - KN = KN → 
  AM + MK + KN + BN = a →
  2 * (AM + BN) + 2 * (MK + KN) = 12 :=
by
  sorry

end rectangle_perimeter_in_right_triangle_l457_457885


namespace find_XY_in_306090_triangle_l457_457031

-- Definitions of the problem
def angleZ := 90
def angleX := 60
def hypotenuseXZ := 12
def isRightTriangle (XYZ : Type) (angleZ : ℕ) : Prop := angleZ = 90
def is306090Triangle (XYZ : Type) (angleX : ℕ) (angleZ : ℕ) : Prop := (angleX = 60) ∧ (angleZ = 90)

-- Lean theorem statement
theorem find_XY_in_306090_triangle 
  (XYZ : Type)
  (hypotenuseXZ : ℕ)
  (h1 : isRightTriangle XYZ angleZ)
  (h2 : is306090Triangle XYZ angleX angleZ) :
  XY = 8 := 
sorry

end find_XY_in_306090_triangle_l457_457031


namespace cos_theta_value_l457_457821

variables {θ : Real}
-- Conditions
def sinθ : Real := -4/5
def tan_positive : Prop := (sinθ < 0 ∧ cos θ < 0)

-- Goal
theorem cos_theta_value :
  sin θ = sinθ ∧ tan θ > 0 → cos θ = -3/5 :=
by
  sorry

end cos_theta_value_l457_457821


namespace first_box_weight_l457_457553

theorem first_box_weight (X : ℕ) 
  (h1 : 11 + 5 + X = 18) : X = 2 := 
by
  sorry

end first_box_weight_l457_457553


namespace triangle_area_l457_457174

def sin_degrees (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)

theorem triangle_area :
  let a := 2
  let b := Real.sqrt 3
  let C := 30
  let S := 1 / 2 * a * b * sin_degrees C
  S = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_l457_457174


namespace largest_possible_b_l457_457306

theorem largest_possible_b (a b c : ℕ) (h₁ : 1 < c) (h₂ : c < b) (h₃ : b < a) (h₄ : a * b * c = 360): b = 12 :=
by
  sorry

end largest_possible_b_l457_457306


namespace solve_for_x_l457_457865

-- Assume x is a positive integer
def pos_integer (x : ℕ) : Prop := 0 < x

-- Assume the equation holds for some x
def equation (x : ℕ) : Prop :=
  1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170

-- Proposition stating that if x satisfies the equation then x must be 5
theorem solve_for_x (x : ℕ) (h1 : pos_integer x) (h2 : equation x) : x = 5 :=
by
  sorry

end solve_for_x_l457_457865


namespace correct_propositions_count_l457_457469

/--
Given the following propositions:

① Two lines that do not have common points are parallel;

② Two lines that are perpendicular to each other are intersecting lines;

③ Lines that are neither parallel nor intersecting are skew lines;

④ Two lines that are not in the same plane are skew lines.

Prove that the number of correct propositions is 2.
-/
theorem correct_propositions_count :
  let prop₁ := ∀ (L₁ L₂ : set ℝ × set ℝ), (disjoint L₁ L₂) → (parallel L₁ L₂);
  let prop₂ := ∀ (L₁ L₂ : set ℝ × set ℝ), (perpendicular L₁ L₂) → (intersecting L₁ L₂);
  let prop₃ := ∀ (L₁ L₂ : set ℝ × set ℝ), (¬ parallel L₁ L₂ ∧ ¬ intersecting L₁ L₂) → (skew L₁ L₂);
  let prop₄ := ∀ (L₁ L₂ : set ℝ × set ℝ), (¬ same_plane L₁ L₂) → (skew L₁ L₂);
  (¬ prop₁ ∧ ¬ prop₂ ∧ prop₃ ∧ prop₄) → count_true [prop₁, prop₂, prop₃, prop₄] = 2 :=
by
  -- Import necessary modules
  sorry

end correct_propositions_count_l457_457469


namespace modulus_of_z_l457_457091

section complex_modulus
open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 + I) = 10 - 5 * I) : Complex.abs z = 5 :=
by
  sorry
end complex_modulus

end modulus_of_z_l457_457091


namespace vectors_not_coplanar_l457_457741

open Matrix

def vec_a : Vector3 := ⟨-7, 10, -5⟩
def vec_b : Vector3 := ⟨0, -2, -1⟩
def vec_c : Vector3 := ⟨-2, 4, -1⟩

theorem vectors_not_coplanar : det (Matrix.ofVecs [vec_a, vec_b, vec_c]) ≠ 0 :=
by
  sorry

end vectors_not_coplanar_l457_457741


namespace problem_statement_l457_457067

-- Definitions based on problem conditions
def ellipse_equation (x y : ℝ) (a b : ℝ) : Prop := 
  (a > b ∧ b > 0) ∧ x^2 / a^2 + y^2 / b^2 = 1

def eccentricity (a c : ℝ) : Prop := 
  c / a = 1 / 2

def foci_distance (c : ℝ) : Prop := 
  2 * c = 2

def point_on_ellipse (x0 y0 : ℝ) : Prop := 
  x0^2 / 4 + y0^2 / 3 = 1

def line_equation (x : ℝ) : Prop :=
  x = 4

def circle_intersection (x0 y0 : ℝ) : Prop := 
  (4 - x0)^2 ≤ (x0 + 1)^2 + y0^2

def max_area (area : ℝ) : Prop :=
  area = sqrt 15 / 3

-- Lean 4 statement for above requirements
theorem problem_statement :
  (∃ (a b : ℝ), ellipse_equation x y a b ∧ eccentricity a 1 ∧ foci_distance 1) →
  (ellipse_equation x y 2 (sqrt 3) ↔ ∀ (x0 y0 : ℝ), point_on_ellipse x0 y0 ∧ line_equation 4 ∧ circle_intersection x0 y0 → max_area (sqrt 15 / 3)) :=
by sorry

end problem_statement_l457_457067


namespace value_of_a5_l457_457537

noncomputable def a_sequence := ℕ → ℝ

axiom geometrically_related (a : a_sequence) : Prop :=
  ∀ n m, (a (n + 1) = a n * r) ∧ (a (m + 1) = a m * r) ∧ (a (n + 1 + m + 1) = a (n + m) * r * r)

def equation_roots (x y : ℝ) : Prop :=
  (x^2 - 3 * x + 2 = 0) ∧ (y^2 - 3 * y + 2 = 0)

theorem value_of_a5 (a : a_sequence) 
  (h1 : equation_roots (a 3) (a 7)) : a 5 = sqrt 2 :=
by
  sorry

end value_of_a5_l457_457537


namespace area_common_part_l457_457696

noncomputable def AD : ℝ := 14 / 3
noncomputable def CD : ℝ := 14 / 3
noncomputable def ∠BAD : ℝ := Real.pi / 2
noncomputable def ∠BCD : ℝ := 5 * Real.pi / 6
noncomputable def AE := DE : ℝ
noncomputable def height_AED : ℝ := 7 / 5

theorem area_common_part (H : AE = DE) 
                         (H_height : height_AED = 7 / 5)
                         (H_AD : AD = 14 / 3) 
                         (H_CD : CD = 14 / 3) 
                         (H_BAD : ∠BAD = Real.pi / 2) 
                         (H_BCD : ∠BCD = 5 * Real.pi / 6) : 
                         ∃ area, area = (49 * (3 * Real.sqrt 3 - 5)) / 3 :=
begin
  sorry
end

end area_common_part_l457_457696


namespace pirate_rick_dig_time_l457_457592
noncomputable def time_to_dig_up_treasure (initial_sand : ℕ) (final_sand : ℕ) (dig_time_per_foot : ℕ) : ℕ :=
  final_sand * dig_time_per_foot

theorem pirate_rick_dig_time :
  let initial_sand := 8 in
  let initial_time := 4 in
  let storm_fraction := 1 / 2 in
  let tsunami_sand := 2 in
  let dig_rate := initial_time / initial_sand in
  let storm_sand := initial_sand * storm_fraction in
  let total_sand := storm_sand + tsunami_sand in
  time_to_dig_up_treasure initial_sand total_sand dig_rate = 3 :=
by
  sorry

end pirate_rick_dig_time_l457_457592


namespace max_value_x_plus_inv_x_l457_457650

theorem max_value_x_plus_inv_x (numbers : Fin 1001 → ℝ) 
  (h₁ : ∀ i, 0 < numbers i)
  (h₂ : (∑ i, numbers i) = 1002)
  (h₃ : (∑ i, (numbers i)⁻¹) = 1002) :
  ∃ x : ℝ, x ∈ Set.range numbers ∧ x + x⁻¹ ≤ 4007 / 1002 :=
by {
  sorry,
}

end max_value_x_plus_inv_x_l457_457650


namespace point_in_fourth_quadrant_l457_457531

def point_x := 3
def point_y := -4

def first_quadrant (x y : Int) := x > 0 ∧ y > 0
def second_quadrant (x y : Int) := x < 0 ∧ y > 0
def third_quadrant (x y : Int) := x < 0 ∧ y < 0
def fourth_quadrant (x y : Int) := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant : fourth_quadrant point_x point_y :=
by
  simp [point_x, point_y, fourth_quadrant]
  sorry

end point_in_fourth_quadrant_l457_457531


namespace shifted_graph_sum_l457_457329

theorem shifted_graph_sum :
  (let f := λ x : ℝ, 3*x^2 + 5*x - 2 in
   let g := λ x : ℝ, f (x + 5) in
   g x = 3*x^2 + 35*x + 98) →
  (3 + 35 + 98 = 136) :=
by
  sorry

end shifted_graph_sum_l457_457329


namespace trigonometric_identity_l457_457686

theorem trigonometric_identity (α β : ℝ) : 
  ((Real.tan α + Real.tan β) / Real.tan (α + β)) 
  + ((Real.tan α - Real.tan β) / Real.tan (α - β)) 
  + 2 * (Real.tan α) ^ 2 
 = 2 / (Real.cos α) ^ 2 :=
  sorry

end trigonometric_identity_l457_457686


namespace sphere_cone_ratio_l457_457657

theorem sphere_cone_ratio (R r : ℝ)
  (h1 : 2 * R = L)
  (h2 : three_identical_spheres_placed_properly_in_cone R r) :
  R / r = 5 / 4 + Real.sqrt 3 :=
sorry

end sphere_cone_ratio_l457_457657


namespace harmonic_sum_prime_squared_divides_numerator_l457_457942

theorem harmonic_sum_prime_squared_divides_numerator
    (p : ℕ) (hp : Nat.Prime p) (h5 : p ≥ 5)
    (S : ℝ) (a b : ℤ)
    (hS : S = (Finset.sum (Finset.range (p - 1)) (λ k, (1 : ℝ) / (k + 1))))
    (h_fraction : (a : ℚ) / b = S) :
    p^2 ∣ a :=
sorry

end harmonic_sum_prime_squared_divides_numerator_l457_457942


namespace sum_or_difference_div_by_11_l457_457242

theorem sum_or_difference_div_by_11 (A : ℕ) (hA : 0 < A) :
  let B := (nat.reverse_digits A) in (A + B) % 11 = 0 ∨ (A - B) % 11 = 0 :=
sorry

end sum_or_difference_div_by_11_l457_457242


namespace find_m_value_l457_457484

noncomputable def U (m : ℝ) : Set ℝ := {4, m^2 + 2m - 3, 19}
def A : Set ℝ := {5}
def complement_UA (m : ℝ) : Set ℝ := {abs (4 * m - 3), 4}

theorem find_m_value (m : ℝ) :
  {4, abs (4 * m - 3)} = complement_UA m →
  U m = {4, m^2 + 2m - 3, 19} →
  U m ≠ {4, abs (4 * m - 3), 19} →
  m = -4 :=
sorry

end find_m_value_l457_457484


namespace probability_of_specific_card_sequence_is_25_3978_l457_457394

def probability_top_red_second_black_third_red_heart :=
  let total_cards := 104
  let total_red := 52
  let total_black := 52
  let total_hearts := 26
  let probability_first_red := (total_red : ℚ) / total_cards
  let remaining_after_first_red := total_cards - 1
  let remaining_red_after_first := total_red - 1
  let probability_second_black := (total_black : ℚ) / remaining_after_first_red
  let remaining_after_second_black := remaining_after_first_red - 1
  let remaining_hearts_after_first := total_hearts - 1
  let probability_third_red_heart := (remaining_hearts_after_first : ℚ) / remaining_after_second_black
  probability_first_red * probability_second_black * probability_third_red_heart

theorem probability_of_specific_card_sequence_is_25_3978 :
  probability_top_red_second_black_third_red_heart = (25 : ℚ) / 3978 := 
by
  sorry

end probability_of_specific_card_sequence_is_25_3978_l457_457394


namespace solution_set_of_inequality_l457_457302

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ -1) : 
  (2 / (x + 1) < 1) ↔ (x ∈ Iio (-1) ∪ Ioi 1) :=
by sorry

end solution_set_of_inequality_l457_457302


namespace exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l457_457018

theorem exp_periodic_cos_sin (x : ℝ) : ∃ k : ℤ, cos(x) = cos(x + 2 * k * π) ∧ sin(x) = sin(x + 2 * k * π) :=
begin
  use 1,
  split;
  apply real.cos_periodic;
  exact int.cast_coe_int (1 : ℤ)
end

theorem euler_formula (x : ℝ) : complex.exp (x * complex.I) = complex.cos x + complex.I * complex.sin x :=
by sorry

theorem exp_13_pi_by_2_equals_i : complex.exp (13 * real.pi / 2 * complex.I) = complex.I :=
begin
  -- Use Euler's formula
  have h_euler : complex.exp (13 * real.pi / 2 * complex.I) = complex.cos (13 * real.pi / 2) + complex.I * complex.sin (13 * real.pi / 2),
  { apply euler_formula },

  -- Simplify the angle by periodicity
  have h_angle : 13 * real.pi / 2 = 6 * real.pi + real.pi / 2,
  { field_simp, ring },
  
  -- Cos and Sin periodicity with 2π
  have h_cos : complex.cos (6 * real.pi + real.pi / 2) = complex.cos (real.pi / 2),
  { rw [← complex.cos_add_period],
    apply exp_periodic_cos_sin,
  },
  
  have h_sin : complex.sin (6 * real.pi + real.pi / 2) = complex.sin (real.pi / 2),
  { rw [← complex.sin_add_period],
    apply exp_periodic_cos_sin,
  },

  -- Calculate Cos(real.pi / 2) and Sin(real.pi / 2)
  have h_cos_pi_by_2 : complex.cos (real.pi / 2) = 0,
  { apply complex.cos_pi_div_two },
  
  have h_sin_pi_by_2 : complex.sin (real.pi / 2) = 1,
  { apply complex.sin_pi_div_two },
  
  -- Combine results
  rw [h_euler, h_angle, h_cos, h_sin],
  rw [h_cos_pi_by_2, h_sin_pi_by_2],
  ring,
end

end exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l457_457018


namespace cards_per_layer_correct_l457_457249

-- Definitions based on the problem's conditions
def num_decks : ℕ := 16
def cards_per_deck : ℕ := 52
def num_layers : ℕ := 32

-- The key calculation we need to prove
def total_cards : ℕ := num_decks * cards_per_deck
def cards_per_layer : ℕ := total_cards / num_layers

theorem cards_per_layer_correct : cards_per_layer = 26 := by
  unfold cards_per_layer total_cards num_decks cards_per_deck num_layers
  simp
  sorry

end cards_per_layer_correct_l457_457249


namespace Christine_distance_went_l457_457386

-- Definitions from conditions
def Speed : ℝ := 20 -- miles per hour
def Time : ℝ := 4  -- hours

-- Statement of the problem
def Distance_went : ℝ := Speed * Time

-- The theorem we need to prove
theorem Christine_distance_went : Distance_went = 80 :=
by
  sorry

end Christine_distance_went_l457_457386


namespace area_of_triangle_angle_C_of_triangle_l457_457900

-- Definitions for part (1)
def triangle_a : ℝ := 3
def ratio_b_c (c : ℝ) : ℝ := 2 * c
def angle_A : ℝ := 2 * Real.pi / 3

-- Lean 4 statement for part (1)
theorem area_of_triangle (c : ℝ) :
  let b := ratio_b_c c in
  let S := (1 / 2) * b * c * (Real.sqrt 3 / 2) in
  S = 9 * Real.sqrt 3 / 14 := by
  sorry

-- Definitions for part (2)
def eq_sinC : ℝ := 1 / 3
def eq_sinB (sinC : ℝ) : ℝ := 2 * sinC
def condition_eq : Prop := 2 * eq_sinB eq_sinC - eq_sinC = 1

-- Lean 4 statement for part (2)
theorem angle_C_of_triangle:
  condition_eq →
  Real.sin C = 1 / 3 := by
  sorry

end area_of_triangle_angle_C_of_triangle_l457_457900


namespace largest_real_c_l457_457424

theorem largest_real_c (x : Fin 101 → ℝ) (M : ℝ):
  (∑ i, x i = 0) →
  (∃ i, x i = M) →
  (∃ i, x (i) = M) →
  ∑ i, (x i)^2 ≥ (5151 / 50) * M^2 := 
by 
  intros hsum hmedian1 hmedian2
  sorry

end largest_real_c_l457_457424


namespace coefficient_x2_in_expansion_l457_457062

noncomputable def e : ℝ := Real.exp 1

noncomputable def n : ℝ := 5 * (∫ x in 0..1, Real.exp x) / (e - 1)

theorem coefficient_x2_in_expansion (hx : x ≠ 0) :
  (x - (4 / x) - 2) ^ n = 80 :=
sorry

end coefficient_x2_in_expansion_l457_457062


namespace profit_percentage_cows_is_20_l457_457737

/-- Define the total cost of horses and cows --/
def total_cost : ℝ := 13400

/-- Define the cost of one horse --/
def cost_of_horse : ℝ := 2000

/-- Calculate the cost of 4 horses --/
def cost_4_horses : ℝ := 4 * cost_of_horse

/-- Calculate the cost of 9 cows --/
def cost_9_cows : ℝ := total_cost - cost_4_horses

/-- Calculate the selling price of horses with 10% profit --/
def selling_price_horses : ℝ := cost_4_horses * 1.10

/-- Define the total profit from horses and cows --/
def total_profit : ℝ := 1880

/-- Calculate the profit made from horses --/
def profit_horses : ℝ := selling_price_horses - cost_4_horses

/-- Calculate the profit made from cows --/
def profit_cows : ℝ := total_profit - profit_horses

/-- Calculate profit percentage for cows --/
def profit_percentage_cows : ℝ := (profit_cows / cost_9_cows) * 100

/-- Prove the profit percentage for cows is 20% --/
theorem profit_percentage_cows_is_20 : profit_percentage_cows = 20 := by
  sorry

end profit_percentage_cows_is_20_l457_457737


namespace union_complement_subset_range_l457_457848

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | 2 * x ^ 2 - 3 * x - 2 < 0}

-- Define the complement of B
def complement_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- 1. The proof problem for A ∪ (complement of B) when a = 1
theorem union_complement (a : ℝ) (h : a = 1) :
  { x : ℝ | (-1/2 < x ∧ x ≤ 1) ∨ (x ≥ 2 ∨ x ≤ -1/2) } = 
  { x : ℝ | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

-- 2. The proof problem for A ⊆ B to find the range of a
theorem subset_range (a : ℝ) :
  (∀ x, A a x → B x) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end union_complement_subset_range_l457_457848


namespace sum_of_possible_values_l457_457491

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 24) : 
  ∃ x1 x2 : ℝ, (x1 + 3) * (x1 - 4) = 24 ∧ (x2 + 3) * (x2 - 4) = 24 ∧ x1 + x2 = 1 := 
by
  sorry

end sum_of_possible_values_l457_457491


namespace exists_six_digit_number_with_unique_trailing_digits_l457_457767

theorem exists_six_digit_number_with_unique_trailing_digits :
  ∃ A : ℕ, 100000 ≤ A ∧ A < 1000000 ∧ 
  ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 500000) ∧ (1 ≤ n ∧ n ≤ 500000) ∧ m ≠ n → 
  (A * m % 1000000 ≠ A * n % 1000000) :=
begin
  sorry
end

end exists_six_digit_number_with_unique_trailing_digits_l457_457767


namespace caps_eaten_correct_l457_457226

def initial_bottle_caps : ℕ := 34
def remaining_bottle_caps : ℕ := 26
def eaten_bottle_caps (k_i k_r : ℕ) : ℕ := k_i - k_r

theorem caps_eaten_correct :
  eaten_bottle_caps initial_bottle_caps remaining_bottle_caps = 8 :=
by
  sorry

end caps_eaten_correct_l457_457226


namespace jean_grandchildren_total_giveaway_l457_457219

theorem jean_grandchildren_total_giveaway :
  let num_grandchildren := 3
  let cards_per_grandchild_per_year := 2
  let amount_per_card := 80
  let total_amount_per_grandchild_per_year := cards_per_grandchild_per_year * amount_per_card
  let total_amount_per_year := num_grandchildren * total_amount_per_grandchild_per_year
  total_amount_per_year = 480 :=
by
  sorry

end jean_grandchildren_total_giveaway_l457_457219


namespace elizabeth_spendings_elizabeth_savings_l457_457029

section WeddingGift

def steak_knife_set_cost : ℝ := 80
def steak_knife_sets : ℕ := 2
def dinnerware_set_cost : ℝ := 200
def fancy_napkins_sets : ℕ := 3
def fancy_napkins_total_cost : ℝ := 45
def wine_glasses_cost : ℝ := 100
def discount_steak_dinnerware : ℝ := 0.10
def discount_napkins : ℝ := 0.20
def sales_tax : ℝ := 0.05

def total_cost_before_discounts : ℝ :=
  (steak_knife_sets * steak_knife_set_cost) + dinnerware_set_cost + fancy_napkins_total_cost + wine_glasses_cost

def total_discount : ℝ :=
  ((steak_knife_sets * steak_knife_set_cost) * discount_steak_dinnerware) + (dinnerware_set_cost * discount_steak_dinnerware) + (fancy_napkins_total_cost * discount_napkins)

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discount

def total_cost_with_tax : ℝ :=
  total_cost_after_discounts + (total_cost_after_discounts * sales_tax)

def savings : ℝ :=
  total_cost_before_discounts - total_cost_after_discounts

theorem elizabeth_spendings :
  total_cost_with_tax = 558.60 :=
by sorry

theorem elizabeth_savings :
  savings = 63 :=
by sorry

end WeddingGift

end elizabeth_spendings_elizabeth_savings_l457_457029


namespace smallest_positive_period_of_f_minimum_value_of_f_on_interval_l457_457096

noncomputable def f : ℝ → ℝ := λ x, Real.sin (2 * x - Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x, f(x + p) = f(x) ∧ p = Real.pi :=
by
  -- We know that the period of sin(x) is 2π and it's horizontally shrunk by a factor of 2
  sorry

theorem minimum_value_of_f_on_interval :
  ∃ x ∈ set.Icc 0 (Real.pi / 2), f(x) = -1 / 2 :=
by
  -- We need to find the minimum of f on the interval [0, π/2]
  sorry

end smallest_positive_period_of_f_minimum_value_of_f_on_interval_l457_457096


namespace tree_graph_probability_127_l457_457309

theorem tree_graph_probability_127 :
  let n := 5
  let p := 125
  let q := 1024
  q ^ (1/10) + p = 127 :=
by
  sorry

end tree_graph_probability_127_l457_457309


namespace proof_problem_correct_l457_457245

noncomputable def proof_problem (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) (h' : xy + xz + yz ≠ 0) : Prop :=
  (x^3 + y^3 + z^3) / (x * y * z * (x * y + x * z + y * z)) = -3

theorem proof_problem_correct (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) (h' : x * y + x * z + y * z ≠ 0) : 
  proof_problem x y z hx hy hz h h' :=
by sorry

end proof_problem_correct_l457_457245


namespace evaluate_complex_sum_l457_457409

theorem evaluate_complex_sum (i : ℂ) (n : ℕ) (h_i : i = Complex.I) (h_n : ∃ k : ℕ, n = 3 * k) :
  (finset.sum (finset.range n) (λ k, (2*k + 1) * i^k)) = (4*n / 3) - (10*n / 3) * i :=
by
  sorry

end evaluate_complex_sum_l457_457409


namespace sarah_cupcakes_ratio_l457_457965

theorem sarah_cupcakes_ratio (total_cupcakes : ℕ) (cookies_from_michael : ℕ) 
    (final_desserts : ℕ) (cupcakes_given : ℕ) (h1 : total_cupcakes = 9) 
    (h2 : cookies_from_michael = 5) (h3 : final_desserts = 11) 
    (h4 : total_cupcakes - cupcakes_given + cookies_from_michael = final_desserts) : 
    cupcakes_given / total_cupcakes = 1 / 3 :=
by
  sorry

end sarah_cupcakes_ratio_l457_457965


namespace petya_knights_original_count_l457_457264

theorem petya_knights_original_count:
  ∃ (knights_initial : ℕ), 
  (∀ (R C : ℕ),
    (knights_initial = R*C) →
    (∀ (R1 C1 : ℕ),
      (R1 = 2) → (C1 = C) →
      knights_initial - 2*C = 24 →
      (∀ (knights_remaining_after_archers : ℕ),
        (knights_remaining_after_archers = 18) →
        (∀ (R2 C2 : ℕ),
          (R2 = R - 2) → (C2 = 2) →
          knights_remaining_after_archers + 2*C2 = knights_initial)
  )) := 
begin
  sorry
end

end petya_knights_original_count_l457_457264


namespace crow_speed_l457_457712

/-- Definitions from conditions -/
def distance_between_nest_and_ditch : ℝ := 250 -- in meters
def total_trips : ℕ := 15
def total_hours : ℝ := 1.5 -- hours

/-- The statement to be proved -/
theorem crow_speed :
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000 -- convert to kilometers
  let speed := total_distance / total_hours
  speed = 5 := by
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000
  let speed := total_distance / total_hours
  sorry

end crow_speed_l457_457712


namespace total_votes_l457_457890

theorem total_votes (V : ℝ) (h : 0.60 * V - 0.40 * V = 1200) : V = 6000 :=
sorry

end total_votes_l457_457890


namespace find_AG_l457_457196

theorem find_AG (AE CE BD CD AB AG : ℝ) (h1 : AE = 3)
    (h2 : CE = 1) (h3 : BD = 2) (h4 : CD = 2) (h5 : AB = 5) :
    AG = (3 * Real.sqrt 66) / 7 :=
  sorry

end find_AG_l457_457196


namespace correct_propositions_count_l457_457904

/-- Two lines perpendicular to the same line in a plane are parallel. -/
axiom plane_perpendicular_lines_parallel (l₁ l₂ l₃ : Line) :
  (perpendicular l₁ l₃ ∧ perpendicular l₂ l₃) → parallel l₁ l₂

/-- In space, two lines perpendicular to the same line are parallel. -/
axiom space_lines_perpendicular_to_same_line (l₁ l₂ l₃ : Line) :
  (perpendicular l₁ l₃ ∧ perpendicular l₂ l₃) → parallel l₁ l₂

/-- In space, two planes perpendicular to the same line are parallel. -/
axiom space_planes_perpendicular_to_same_line (p₁ p₂ : Plane) (l : Line) :
  (perpendicular p₁ l ∧ perpendicular p₂ l) → parallel p₁ p₂

/-- In space, two lines perpendicular to the same plane are parallel. -/
axiom space_lines_perpendicular_to_same_plane (l₁ l₂ : Line) (p : Plane) :
  (perpendicular l₁ p ∧ perpendicular l₂ p) → parallel l₁ l₂

/-- In space, two planes perpendicular to the same plane are parallel. -/
axiom space_planes_perpendicular_to_same_plane (p₁ p₂ p₃ : Plane) :
  (perpendicular p₁ p₃ ∧ perpendicular p₂ p₃) → parallel p₁ p₂

theorem correct_propositions_count :
  (count_correct_propositions == 2) := sorry

end correct_propositions_count_l457_457904


namespace joan_initial_books_l457_457222

variable (books_sold : ℕ)
variable (books_left : ℕ)

theorem joan_initial_books (h1 : books_sold = 26) (h2 : books_left = 7) : books_sold + books_left = 33 := by
  sorry

end joan_initial_books_l457_457222


namespace unique_function_l457_457773

def fractional_part (y : ℝ) : ℝ := y - y.floor

theorem unique_function {f : ℝ → ℝ} (hf : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 
  fractional_part (f x) * Real.sin x ^ 2 + fractional_part x * Real.cos (f x) * Real.cos x = f x ∧ 
  f (f x) = f x) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x :=
sorry

end unique_function_l457_457773


namespace cos_18_eq_l457_457002

-- Definitions for the conditions.
def a := Real.cos (36 * Real.pi / 180)
def c := Real.cos (18 * Real.pi / 180)

-- Statement of the problem
theorem cos_18_eq :
  c = (Real.sqrt (10 + 2 * Real.sqrt 5) / 4) :=
by
  -- conditions given in the problem
  have h1: a = 2 * c^2 - 1, from sorry
  have h2: Real.sin (36 * Real.pi / 180) = Real.sqrt (1 - a^2), from sorry
  have triple_angle: Real.cos (3 * θ) = 4 * (Real.cos θ)^3 - 3 * (Real.cos θ), from sorry
  sorry

end cos_18_eq_l457_457002


namespace least_value_of_function_l457_457391

noncomputable def f (a b c d k x : ℝ) : ℝ :=
  a * x^2 + b * x + c + d * real.sin (k * x)

theorem least_value_of_function (a b c d k : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ∃ x : ℝ, f a b c d k x = (-(b^2) + 4 * a * c - 4 * a * d) / (4 * a) :=
sorry

end least_value_of_function_l457_457391


namespace perpendicular_vectors_l457_457118

variable (λ μ : ℝ)
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

theorem perpendicular_vectors : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0 → λ * μ = -1 := by
  sorry

end perpendicular_vectors_l457_457118


namespace infinitely_many_composite_numbers_l457_457269

-- We define n in a specialized form.
def n (m : ℕ) : ℕ := (3 * m) ^ 3

-- We state that m is an odd positive integer.
def odd_positive_integer (m : ℕ) : Prop := m > 0 ∧ (m % 2 = 1)

-- The main statement: for infinitely many odd values of n, 2^n + n - 1 is composite.
theorem infinitely_many_composite_numbers : 
  ∃ (m : ℕ), odd_positive_integer m ∧ Nat.Prime (n m) ∧ ∃ d : ℕ, d > 1 ∧ d < n m ∧ (2^(n m) + n m - 1) % d = 0 :=
by
  sorry

end infinitely_many_composite_numbers_l457_457269


namespace fraction_received_A_correct_l457_457505

def fraction_of_students_received_A := 0.7
def fraction_of_students_received_B := 0.2
def fraction_of_students_received_A_or_B := 0.9

theorem fraction_received_A_correct :
  fraction_of_students_received_A_or_B - fraction_of_students_received_B = fraction_of_students_received_A :=
by
  sorry

end fraction_received_A_correct_l457_457505


namespace multiplication_identity_multiplication_l457_457993

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l457_457993


namespace factor_expression_l457_457413

theorem factor_expression (x : ℝ) :
  84 * x ^ 5 - 210 * x ^ 9 = -42 * x ^ 5 * (5 * x ^ 4 - 2) :=
by
  sorry

end factor_expression_l457_457413


namespace count_two_digit_integers_remainder_3_div_9_l457_457163

theorem count_two_digit_integers_remainder_3_div_9 :
  ∃ (N : ℕ), N = 10 ∧ ∀ k, (10 ≤ 9 * k + 3 ∧ 9 * k + 3 < 100) ↔ (1 ≤ k ∧ k ≤ 10) :=
by
  sorry

end count_two_digit_integers_remainder_3_div_9_l457_457163


namespace complex_solution_l457_457868

theorem complex_solution (x : ℂ) (h : x^2 + 1 = 0) : x = Complex.I ∨ x = -Complex.I :=
by sorry

end complex_solution_l457_457868


namespace chairs_left_after_borrowing_l457_457177

def total_chairs := 62
def red_chairs := 4
def yellow_chairs := 2 * red_chairs
def blue_chairs := 3 * yellow_chairs
def green_chairs := blue_chairs / 2
def orange_chairs := green_chairs + 2

def chairs_after_lisa (total : ℕ) : ℕ :=
  total - total / 10

def chairs_after_carla (remaining : ℕ) : ℕ :=
  remaining - remaining / 5

def remaining_chairs (total_norm : ℕ) : ℕ :=
  chairs_after_carla (chairs_after_lisa total_norm)

theorem chairs_left_after_borrowing :
  remaining_chairs total_chairs = 45 :=
by
  unfold total_chairs yellow_chairs blue_chairs green_chairs orange_chairs
  unfold chairs_after_lisa chairs_after_carla remaining_chairs
  -- Here we expect to show 
  -- remaining_chairs 62 = 45
  sorry

end chairs_left_after_borrowing_l457_457177


namespace reduced_price_per_kg_of_oil_l457_457365

/- 
Given:
1. A reduction of 25% in the price of oil.
2. Enables a housewife to obtain 5 kgs more for Rs. 1300.

Prove:
The reduced price per kg of oil is Rs. 65.
-/

theorem reduced_price_per_kg_of_oil :
  ∃ P R : ℝ, (R = 0.75 * P) ∧ (1300 = 5 * R + 1300 / P * R) ∧ (R ≈ 65) :=
begin
  sorry
end

end reduced_price_per_kg_of_oil_l457_457365


namespace number_of_two_digit_integers_with_remainder_3_mod_9_l457_457159

theorem number_of_two_digit_integers_with_remainder_3_mod_9 : 
  {x : ℤ // 10 ≤ x ∧ x < 100 ∧ ∃ n : ℤ, x = 9 * n + 3}.card = 10 := by
sorry

end number_of_two_digit_integers_with_remainder_3_mod_9_l457_457159


namespace perpendicular_vectors_l457_457115

variable (λ μ : ℝ)
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

theorem perpendicular_vectors : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0 → λ * μ = -1 := by
  sorry

end perpendicular_vectors_l457_457115


namespace g_at_1_zero_l457_457477

noncomputable def g (x : ℝ) (a : ℝ) (f : ℝ → ℝ) : ℝ := a ^ x - f x

theorem g_at_1_zero (a : ℝ) (f : ℝ → ℝ)
  (ha1 : a > 0) (ha2 : a ≠ 1)
  (hf_odd : ∀ x ∈ set.Icc (a - 6) (2 * a), f x = -f (-x))
  (h_g_neg1 : g (-1) a f = 5 / 2) :
  g 1 a f = 0 :=
sorry

end g_at_1_zero_l457_457477


namespace exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l457_457019

theorem exp_periodic_cos_sin (x : ℝ) : ∃ k : ℤ, cos(x) = cos(x + 2 * k * π) ∧ sin(x) = sin(x + 2 * k * π) :=
begin
  use 1,
  split;
  apply real.cos_periodic;
  exact int.cast_coe_int (1 : ℤ)
end

theorem euler_formula (x : ℝ) : complex.exp (x * complex.I) = complex.cos x + complex.I * complex.sin x :=
by sorry

theorem exp_13_pi_by_2_equals_i : complex.exp (13 * real.pi / 2 * complex.I) = complex.I :=
begin
  -- Use Euler's formula
  have h_euler : complex.exp (13 * real.pi / 2 * complex.I) = complex.cos (13 * real.pi / 2) + complex.I * complex.sin (13 * real.pi / 2),
  { apply euler_formula },

  -- Simplify the angle by periodicity
  have h_angle : 13 * real.pi / 2 = 6 * real.pi + real.pi / 2,
  { field_simp, ring },
  
  -- Cos and Sin periodicity with 2π
  have h_cos : complex.cos (6 * real.pi + real.pi / 2) = complex.cos (real.pi / 2),
  { rw [← complex.cos_add_period],
    apply exp_periodic_cos_sin,
  },
  
  have h_sin : complex.sin (6 * real.pi + real.pi / 2) = complex.sin (real.pi / 2),
  { rw [← complex.sin_add_period],
    apply exp_periodic_cos_sin,
  },

  -- Calculate Cos(real.pi / 2) and Sin(real.pi / 2)
  have h_cos_pi_by_2 : complex.cos (real.pi / 2) = 0,
  { apply complex.cos_pi_div_two },
  
  have h_sin_pi_by_2 : complex.sin (real.pi / 2) = 1,
  { apply complex.sin_pi_div_two },
  
  -- Combine results
  rw [h_euler, h_angle, h_cos, h_sin],
  rw [h_cos_pi_by_2, h_sin_pi_by_2],
  ring,
end

end exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l457_457019


namespace lambda_mu_relationship_l457_457141

variables (λ μ : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def linear_combination (v : ℝ × ℝ) (c : ℝ) (w : ℝ × ℝ) := 
  (v.1 + c * w.1, v.2 + c * w.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem lambda_mu_relationship (h : dot_product (linear_combination vector_a λ vector_b)
                                          (linear_combination vector_a μ vector_b) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457141


namespace choose_4_from_multiset_l457_457350

theorem choose_4_from_multiset :
  (multiset.card (multiset.powerset_len 4 (multiset.ofList ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']))) = 175 :=
by {
  sorry
}

end choose_4_from_multiset_l457_457350


namespace polynomial_identity_l457_457796

theorem polynomial_identity :
  ∀ (a a1 a2 a3 a4 a5 a6 a7 : ℤ), -- Assuming coefficients are integers
  (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 →
  a - a1 + a2 - a3 + a4 - a5 + a6 - a7 = 3^7 :=
by {
  intros,
  sorry
}

end polynomial_identity_l457_457796


namespace simplify_expr1_simplify_expr2_l457_457606

variable (x : Real)

theorem simplify_expr1 :
  (\(\frac{\sqrt{3}}{2}\cos x - \frac{1}{2}\sin x) = cos(x + \frac{\pi}{6}\right) :=
sorry

theorem simplify_expr2 :
  (\(\sin x + \cos x) = \sqrt{2} \sin(x + \frac{\pi}{4}\right) :=
sorry

end simplify_expr1_simplify_expr2_l457_457606


namespace avg_books_per_student_l457_457509

theorem avg_books_per_student 
  (total_students : ℕ)
  (students_zero_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (max_books_per_student : ℕ) 
  (remaining_students_min_books : ℕ)
  (total_books : ℕ)
  (avg_books : ℚ)
  (h1 : total_students = 32)
  (h2 : students_zero_books = 2)
  (h3 : students_one_book = 12)
  (h4 : students_two_books = 10)
  (h5 : max_books_per_student = 11)
  (h6 : remaining_students_min_books = 8)
  (h7 : total_books = 0 * students_zero_books + 1 * students_one_book + 2 * students_two_books + 3 * remaining_students_min_books)
  (h8 : avg_books = total_books / total_students) :
  avg_books = 1.75 :=
by {
  -- Additional constraints and intermediate steps can be added here if necessary
  sorry
}

end avg_books_per_student_l457_457509


namespace sequence_eq_l457_457439

-- Define the sequence and the conditions
def is_sequence (a : ℕ → ℕ) :=
  (∀ i, a i > 0) ∧ (∀ i j, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j)

-- The theorem we want to prove: for all i, a_i = i
theorem sequence_eq (a : ℕ → ℕ) (h : is_sequence a) : ∀ i, a i = i :=
by
  sorry

end sequence_eq_l457_457439


namespace hyperbola_range_l457_457624

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (2 + m) + y^2 / (m + 1) = 1)) → (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_range_l457_457624


namespace multiply_expand_l457_457975

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l457_457975


namespace problem_solution_l457_457564

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x < -6 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 ))
  (h2 : a < b)
  : a + 2 * b + 3 * c = 74 := 
sorry

end problem_solution_l457_457564


namespace john_income_increase_l457_457921

noncomputable def net_percentage_increase (initial_income : ℝ) (final_income_before_bonus : ℝ) (monthly_bonus : ℝ) (tax_deduction_rate : ℝ) : ℝ :=
  let weekly_bonus := monthly_bonus / 4
  let final_income_before_taxes := final_income_before_bonus + weekly_bonus
  let tax_deduction := tax_deduction_rate * final_income_before_taxes
  let net_final_income := final_income_before_taxes - tax_deduction
  ((net_final_income - initial_income) / initial_income) * 100

theorem john_income_increase :
  net_percentage_increase 40 60 100 0.10 = 91.25 := by
  sorry

end john_income_increase_l457_457921


namespace number_of_two_digit_integers_with_remainder_3_mod_9_l457_457158

theorem number_of_two_digit_integers_with_remainder_3_mod_9 : 
  {x : ℤ // 10 ≤ x ∧ x < 100 ∧ ∃ n : ℤ, x = 9 * n + 3}.card = 10 := by
sorry

end number_of_two_digit_integers_with_remainder_3_mod_9_l457_457158


namespace exp_rectangular_form_l457_457020

theorem exp_rectangular_form : (complex.exp (13 * real.pi * complex.I / 2)) = complex.I :=
by
  sorry

end exp_rectangular_form_l457_457020


namespace multiply_expand_l457_457978

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l457_457978


namespace vector_magnitude_l457_457063

noncomputable def z : ℂ := 1 - complex.i

noncomputable def complex_number : ℂ := (2 / z) + z^2

noncomputable def magnitude : ℝ := complex.abs complex_number

theorem vector_magnitude : magnitude = real.sqrt 2 := 
by
  sorry

end vector_magnitude_l457_457063


namespace investor_pieces_impossible_to_be_2002_l457_457356

theorem investor_pieces_impossible_to_be_2002 : 
  ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := 
by
  sorry

end investor_pieces_impossible_to_be_2002_l457_457356


namespace lambda_mu_condition_l457_457107

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem lambda_mu_condition (λ μ : ℝ) :
  orthogonal (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
             (vector_a.1 + μ * vector_b.1, vector_a.2 + μ * vector_b.2) →
  λ * μ = -1 :=
by
  sorry

end lambda_mu_condition_l457_457107


namespace complete_square_correct_l457_457278

theorem complete_square_correct (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by 
  intro h
  sorry

end complete_square_correct_l457_457278


namespace johns_oil_change_cost_l457_457916

theorem johns_oil_change_cost:
  (miles_per_month: ℕ) (miles_per_oil_change: ℕ) (free_oil_changes_per_year: ℕ) (cost_per_oil_change: ℕ) 
  (h₁ : miles_per_month = 1000) 
  (h₂ : miles_per_oil_change = 3000) 
  (h₃ : free_oil_changes_per_year = 1) 
  (h₄ : cost_per_oil_change = 50) : 
  (12 * cost_per_oil_change * miles_per_oil_change) // (miles_per_month * miles_per_oil_change) - (free_oil_changes_per_year * cost_per_oil_change) = 150 := 
by 
  sorry

end johns_oil_change_cost_l457_457916


namespace solve_for_x_l457_457417

theorem solve_for_x :
  (∀ x, (sqrt (x - 9) - 10) ≠ 0 ∧ (sqrt (x - 9) - 5) ≠ 0 ∧ (sqrt (x - 9) + 5) ≠ 0 ∧ (sqrt (x - 9) + 10) ≠ 0 → 
    6 / (sqrt (x - 9) - 10) + 1 / (sqrt (x - 9) - 5) + 7 / (sqrt (x - 9) + 5) + 12 / (sqrt (x - 9) + 10) = 0 ↔ 
    x = 9 ∨ x = 109) := 
by sorry

end solve_for_x_l457_457417


namespace log2_neither_even_nor_odd_l457_457902

-- Definitions based on problem conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Main theorem stating the problem to be proved
theorem log2_neither_even_nor_odd : ¬(is_even log2) ∧ ¬(is_odd log2) := by
  sorry

end log2_neither_even_nor_odd_l457_457902


namespace regression_line_equation_l457_457830

-- Define the conditions
def slope : ℝ := 1.23
def center_x : ℝ := 4
def center_y : ℝ := 5

-- Define the equation of the regression line
def regression_equation (a : ℝ) : (ℝ → ℝ) :=
  λ x : ℝ, slope * x + a

-- The proof statement
theorem regression_line_equation : regression_equation 0.08 center_x = center_y := by
  sorry

end regression_line_equation_l457_457830


namespace Lorelei_vase_contains_22_roses_l457_457200

variable (redBush : ℕ) (pinkBush : ℕ) (yellowBush : ℕ) (orangeBush : ℕ)
variable (percentRed : ℚ) (percentPink : ℚ) (percentYellow : ℚ) (percentOrange : ℚ)

noncomputable def pickedRoses : ℕ :=
  let redPicked := redBush * percentRed
  let pinkPicked := pinkBush * percentPink
  let yellowPicked := yellowBush * percentYellow
  let orangePicked := orangeBush * percentOrange
  (redPicked + pinkPicked + yellowPicked + orangePicked).toNat

theorem Lorelei_vase_contains_22_roses 
  (redBush := 12) (pinkBush := 18) (yellowBush := 20) (orangeBush := 8)
  (percentRed := 0.5) (percentPink := 0.5) (percentYellow := 0.25) (percentOrange := 0.25)
  : pickedRoses redBush pinkBush yellowBush orangeBush percentRed percentPink percentYellow percentOrange = 22 := by 
  sorry

end Lorelei_vase_contains_22_roses_l457_457200


namespace deepak_current_age_l457_457338

theorem deepak_current_age (A D : ℕ) (h1 : A / D = 5 / 7) (h2 : A + 6 = 36) : D = 42 :=
sorry

end deepak_current_age_l457_457338


namespace annual_oil_change_cost_l457_457913

/-!
# Problem Statement
John drives 1000 miles a month. An oil change is needed every 3000 miles.
John gets 1 free oil change a year. Each oil change costs $50.

Prove that the total amount John pays for oil changes in a year is $150.
-/

def miles_driven_per_month : ℕ := 1000
def miles_per_oil_change : ℕ := 3000
def free_oil_changes_per_year : ℕ := 1
def oil_change_cost : ℕ := 50

theorem annual_oil_change_cost : 
  let total_oil_changes := (12 * miles_driven_per_month) / miles_per_oil_change,
      paid_oil_changes := total_oil_changes - free_oil_changes_per_year
  in paid_oil_changes * oil_change_cost = 150 :=
by {
  -- The proof is not required, so we use sorry
  sorry 
}

end annual_oil_change_cost_l457_457913


namespace cube_root_expression_l457_457170

variable (x : ℝ)

theorem cube_root_expression (h : x + 1 / x = 7) : x^3 + 1 / x^3 = 322 :=
  sorry

end cube_root_expression_l457_457170


namespace evaluate_expression_l457_457771

theorem evaluate_expression : [3 - 4 * (5 - 6)⁻¹]⁻¹ * (1 - 2⁻¹) = 1 / 14 := 
by
  sorry

end evaluate_expression_l457_457771


namespace parallelogram_opposite_angles_equal_l457_457892

theorem parallelogram_opposite_angles_equal (ABCD : Type) [parallelogram ABCD] (B D : ABCD.angle) [B = 125] : D = 125 :=
sorry

end parallelogram_opposite_angles_equal_l457_457892


namespace multiply_and_simplify_l457_457969
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l457_457969


namespace base_n_representation_l457_457172

theorem base_n_representation (n : ℕ) (b : ℕ) (h₀ : 8 < n) (h₁ : ∃ b, (n : ℤ)^2 - (n+8) * (n : ℤ) + b = 0) : 
  b = 8 * n :=
by
  sorry

end base_n_representation_l457_457172


namespace max_inequality_l457_457230

namespace PolygonInequality

variables {n : ℕ} 
variables (A : Finₙ → (ℝ × ℝ)) 
variables (P : ℝ × ℝ) 
variables (P_projections : Finₙ → (ℝ × ℝ))
variables (X : Finₙ → (ℝ × ℝ))

-- Definitions ensuring A forms a convex polygon, P is inside A, and P_projections are projections
def isConvexPolygon (A : Finₙ → (ℝ × ℝ)) : Prop := sorry
def isInsidePolygon (A : Finₙ → (ℝ × ℝ)) (P : (ℝ × ℝ)) : Prop := sorry
def areProjectionsOnSides : Prop := sorry
def arePointsOnSides : Prop := sorry


-- The theorem we want to prove
theorem max_inequality 
  (A : Finₙ → (ℝ × ℝ)) (P : (ℝ × ℝ)) 
  (P_projections : Finₙ → (ℝ × ℝ)) (X : Finₙ → (ℝ × ℝ)) 
  (h1 : isConvexPolygon A) 
  (h2 : isInsidePolygon A P) 
  (h3 : areProjectionsOnSides P P_projections) 
  (h4 : arePointsOnSides A X) : 
  ∃ i < n, max { (dist (X i) (X (i + 1)) / dist (P_projections i) (P_projections (i + 1)) : Finₙ }, exists (X i >= ...)  

end PolygonInequality

end max_inequality_l457_457230


namespace largest_possible_b_l457_457305

theorem largest_possible_b (a b c : ℕ) (h₁ : 1 < c) (h₂ : c < b) (h₃ : b < a) (h₄ : a * b * c = 360): b = 12 :=
by
  sorry

end largest_possible_b_l457_457305


namespace integral_problem_l457_457749

noncomputable def definite_integral : ℝ :=
  ∫ x in π..2 * π, (1 - Real.cos x) / (x - Real.sin x) ^ 2

theorem integral_problem :
  definite_integral = 1 / (2 * π) :=
by
  sorry

end integral_problem_l457_457749


namespace eggs_left_l457_457310

theorem eggs_left (x : ℕ) : (47 - 5 - x) = (42 - x) :=
  by
  sorry

end eggs_left_l457_457310


namespace solution_set_of_fx_lt_0_l457_457095

def f (x : ℝ) : ℝ :=
if x ≤ 0 then -|x + 1| else x^2 - 1

theorem solution_set_of_fx_lt_0 :
  {x : ℝ | f x < 0} = {x : ℝ | x ∈ (-∞, -1) ∪ (-1, 1)} :=
by 
  sorry

end solution_set_of_fx_lt_0_l457_457095


namespace mail_difference_eq_15_l457_457924

variable (Monday Tuesday Wednesday Thursday : ℕ)
variable (total : ℕ)

theorem mail_difference_eq_15
  (h1 : Monday = 65)
  (h2 : Tuesday = Monday + 10)
  (h3 : Wednesday = Tuesday - 5)
  (h4 : total = 295)
  (h5 : total = Monday + Tuesday + Wednesday + Thursday) :
  Thursday - Wednesday = 15 := 
  by
  sorry

end mail_difference_eq_15_l457_457924


namespace max_value_of_y_l457_457503

open Classical

noncomputable def satisfies_equation (x y : ℝ) : Prop := y * x * (x + y) = x - y

theorem max_value_of_y : 
  ∀ (y : ℝ), (∃ (x : ℝ), x > 0 ∧ satisfies_equation x y) → y ≤ 1 / 3 := 
sorry

end max_value_of_y_l457_457503


namespace find_smallest_palindrome_l457_457045

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_aba_form (n : ℕ) : Prop :=
  let s := n.digits 10
  s.length = 3 ∧ s.head = s.get! 2

def smallest_aba_not_palindromic_when_multiplied_by_103 : ℕ :=
  Nat.find (λ n, is_three_digit n ∧ is_aba_form n ∧ ¬is_palindrome (103 * n))

theorem find_smallest_palindrome : smallest_aba_not_palindromic_when_multiplied_by_103 = 131 := sorry

end find_smallest_palindrome_l457_457045


namespace problem_l457_457863

noncomputable def x : ℕ := 5  -- Define x as the positive integer 5

theorem problem (hx : ∀ x, 1 ≤ x → 1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170 ↔ x = 5) : 1^(5+2) + 2^(5+1) + 3^(5-1) + 4^5 = 1170 :=
by {
  have : 1^(5+2) + 2^(5+1) + 3^(5-1) + 4^5 = 1^7 + 2^6 + 3^4 + 4^5 := by rfl,
  rw [this],
  norm_num,
}

end problem_l457_457863


namespace mental_competition_problem_l457_457513

theorem mental_competition_problem
  (x_a x_b x_c : ℕ)
  (hx1 : x_a + x_b = 29)
  (hx2 : x_a + x_c = 25)
  (hx3 : x_b + x_c = 20)
  (hx4 : 1 = 1)  -- placeholder for "one student answered all three questions correctly"
  (hx5 : 15 = 15) -- placeholder for "15 students who answered two questions correctly"
  (hx6 : 1 + 15 * 2 + ? = x_a + x_b + x_c)
  : x_a = 17 ∧
    x_b = 12 ∧
    x_c = 8 ∧
    (x_a + x_b + x_c - (3 + 30) = 4) ∧
    (let total_score := (17 * 20 + 12 * 25 + 8 * 25 + 70 : ℚ) in (total_score / 20) = 42) :=
by {
  have hx_a : x_a = (25 + 29 - 20) / 2 := sorry,
  have hx_b : x_b = (29 + 20 - 25) / 2 := sorry,
  have hx_c : x_c = (25 + 20 - 29) / 2 := sorry,
  exact ⟨hx_a, hx_b, hx_c, sorry, sorry⟩,
}

end mental_competition_problem_l457_457513


namespace distance_D_D_l457_457666

open Real -- Using the real numbers namespace

-- Define the points D and D' in Lean
def D : ℝ × ℝ := (2, -4)
def D' : ℝ × ℝ := (-2, -4)

-- Define the distance formula between two points in Lean
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- State the theorem that the distance between D and D' is 4
theorem distance_D_D' : distance D D' = 4 := by
  -- This line ensures that the Lean statement can be built successfully
  sorry

end distance_D_D_l457_457666


namespace find_n_mod_10_l457_457779

theorem find_n_mod_10 (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) (h₂ : n ≡ 123456 [MOD 10]) : n = 6 :=
by
  sorry

end find_n_mod_10_l457_457779


namespace joan_football_games_last_year_l457_457910

theorem joan_football_games_last_year
  (games_this_year : ℕ)
  (total_games : ℕ)
  (h1 : games_this_year = 4)
  (h2 : total_games = 13) :
  (last_year_games : ℕ) := last_year_games = 9 :=
by
  sorry

end joan_football_games_last_year_l457_457910


namespace speed_ratio_l457_457410

variables (x y : ℝ)

def T_sunday (x : ℝ) : ℝ := 120 / x

def T_monday (y x : ℝ) : ℝ := 32 / y + 88 / (x / 2)

theorem speed_ratio (x y : ℝ) (h : T_monday y x = 1.6 * T_sunday x) : y / x = 2 :=
by
  unfold T_sunday at h
  unfold T_monday at h
  sorry

end speed_ratio_l457_457410


namespace total_movies_attended_l457_457658

-- Defining the conditions for Timothy's movie attendance
def Timothy_2009 := 24
def Timothy_2010 := Timothy_2009 + 7

-- Defining the conditions for Theresa's movie attendance
def Theresa_2009 := Timothy_2009 / 2
def Theresa_2010 := Timothy_2010 * 2

-- Prove that the total number of movies Timothy and Theresa went to in both years is 129
theorem total_movies_attended :
  (Timothy_2009 + Timothy_2010 + Theresa_2009 + Theresa_2010) = 129 :=
by
  -- proof goes here
  sorry

end total_movies_attended_l457_457658


namespace non_congruent_right_triangles_count_l457_457490

def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def areaEqualsFourTimesPerimeter (a b c : ℕ) : Prop :=
  a * b = 8 * (a + b + c)

theorem non_congruent_right_triangles_count :
  {n : ℕ // ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ isRightTriangle a b c ∧ areaEqualsFourTimesPerimeter a b c ∧ n = 3} := sorry

end non_congruent_right_triangles_count_l457_457490


namespace two_digit_integers_mod_9_eq_3_l457_457156

theorem two_digit_integers_mod_9_eq_3 :
  { x : ℕ | 10 ≤ x ∧ x < 100 ∧ x % 9 = 3 }.finite.card = 10 :=
by sorry

end two_digit_integers_mod_9_eq_3_l457_457156


namespace total_animal_eyes_l457_457884

def num_snakes := 18
def num_alligators := 10
def eyes_per_snake := 2
def eyes_per_alligator := 2

theorem total_animal_eyes : 
  (num_snakes * eyes_per_snake) + (num_alligators * eyes_per_alligator) = 56 :=
by 
  sorry

end total_animal_eyes_l457_457884


namespace find_investment_amount_l457_457633

noncomputable def brokerage_fee (market_value : ℚ) : ℚ := (1 / 4 / 100) * market_value

noncomputable def actual_cost (market_value : ℚ) : ℚ := market_value + brokerage_fee market_value

noncomputable def income_per_100_face_value (interest_rate : ℚ) : ℚ := (interest_rate / 100) * 100

noncomputable def investment_amount (income : ℚ) (actual_cost_per_100 : ℚ) (income_per_100 : ℚ) : ℚ :=
  (income * actual_cost_per_100) / income_per_100

theorem find_investment_amount :
  investment_amount 756 (actual_cost 124.75) (income_per_100_face_value 10.5) = 9483.65625 :=
sorry

end find_investment_amount_l457_457633


namespace bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l457_457652

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Bose-Einstein distribution, satisfying the given conditions. 
-/
theorem bose_einstein_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 72 := 
  by
  sorry

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Fermi-Dirac distribution, satisfying the given conditions. 
-/
theorem fermi_dirac_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 246 := 
  by
  sorry

end bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l457_457652


namespace largest_binomial_coefficient_in_expansion_of_x_minus_inv_x_pow_l457_457078

theorem largest_binomial_coefficient_in_expansion_of_x_minus_inv_x_pow :
  ∃ (n : ℕ), 
    (∀ a : ℕ, a ≥ 3 ∧ S = a + (∑ i in (range 1 28), nat.choose 27 i) ∧ S % 9 = 0 → n = a) → 
    n = 11 ∧ 
    largest_binomial_coefficient_term ((x - 1 / x)^11) = 7 :=
by
  sorry

end largest_binomial_coefficient_in_expansion_of_x_minus_inv_x_pow_l457_457078


namespace infinitely_many_composite_l457_457267

/-- Definition of the sequence a_k --/
def seq (k : ℕ) : ℕ := ⌊2^k * Real.sqrt 2⌋

/-- Main theorem statement -/
theorem infinitely_many_composite : ∃ᶠ k in at_top, ∃ m, m ∣ seq k ∧ m ≠ 1 ∧ m ≠ seq k :=
sorry

end infinitely_many_composite_l457_457267


namespace exp_rectangular_form_l457_457022

theorem exp_rectangular_form : (complex.exp (13 * real.pi * complex.I / 2)) = complex.I :=
by
  sorry

end exp_rectangular_form_l457_457022


namespace six_valid_configurations_l457_457595

/-- Given four points on a plane such that the pairwise distances 
between them yield exactly two distinct values, there are exactly 
six valid configurations for these points. -/
theorem six_valid_configurations (P : Fin 4 → ℝ × ℝ) 
  (distinct_distances : Finset ℝ := 
    (Finset.image (λ ⟨i, j⟩, Real.sqrt ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2) (Finset.univ : Finset (Fin 4 × Fin 4)))) :
  distinct_distances.card = 2 :=
sorry

end six_valid_configurations_l457_457595


namespace constant_term_is_fifth_term_sum_of_coefficients_max_coefficient_term_l457_457869

noncomputable def term_coefficient (n r : ℕ) (x : ℝ) : ℝ :=
  2^(n - r) * (Nat.choose n r) * x^(r / 2 - (n - r) / 3)

theorem constant_term_is_fifth_term {n : ℕ} :
  (∃ n, n = 10 ∧ term_coefficient 10 4 0 = term_coefficient 10 4 0) :=
begin
  use 10,
  split,
  { refl, },
  { congr, },
end

theorem sum_of_coefficients {A : ℝ} :
  (∃ A, A = (1 + 2)^10 ∧ 10th_root A = 3) :=
begin
  use (3^10),
  split,
  { refl, },
  { exact Real.rpow_nat_cast 3 10 0.1, },
end

theorem max_coefficient_term :
  (∃ r, r = 4 ∧ term_coefficient 10 4 x = 2^4 * (Nat.choose 10 4) * x^(5 / 3)) :=
begin
  use 4,
  split,
  { refl, },
  { congr, },
end

# This proves that:
-- 1. The constant term matching criteria holds when n = 10.
-- 2. The sum of the coefficients in the expansion gives root 3.
-- 3. The maximal term in the expansion is correctly identified.

end constant_term_is_fifth_term_sum_of_coefficients_max_coefficient_term_l457_457869


namespace compute_DR_l457_457197

noncomputable def area_of_octagon (CO OM MP PU UT TE : ℝ) (angles : Set ℝ) : ℝ :=
  -- Assuming the specific structure deduced from the problem
  6 * CO * OM
  
def is_interior_angle (x : ℝ) : Prop :=
  x = 90 ∨ x = 270

def area_of_polygon_COMPUTED (total_area : ℝ) : ℝ :=
  total_area / 2

def area_of_triangle_CDR (CD DR : ℝ) : ℝ :=
  (1 / 2) * CD * DR

theorem compute_DR (CO OM MP PU UT TE : ℝ)
  (angles : Set ℝ)
  (h1 : CO = 1)
  (h2 : OM = 1)
  (h3 : MP = 1)
  (h4 : PU = 1)
  (h5 : UT = 1)
  (h6 : TE = 1)
  (h7 : ∀ θ ∈ angles, is_interior_angle θ)
  (h8 : area_of_octagon CO OM MP PU UT TE angles = 6)
  (h9 : CD = 3) 
  (h10 : area_of_polygon_COMPUTED (area_of_octagon CO OM MP PU UT TE angles) = 3) :
  DR = 2 :=
sorry

end compute_DR_l457_457197


namespace binomial_expansion_coefficients_l457_457455

theorem binomial_expansion_coefficients (n : ℕ)
  (a1 a2 a3 : ℕ)
  (h1 : a1 = C_n^0)
  (h2 : a2 = (1 / 2) * C_n^1)
  (h3 : a3 = (1 / 4) * C_n^2)
  (h4 : 2 * a2 = a1 + a3)
  (h5 : n = 8) :
  -- 1. Sum of binomial coefficients
  (∑ k : ℕ in range (n + 1), binomial n k = 256) ∧
  -- 2. Terms with the largest coefficient
  (T_3 = 7 * x ^ (7/3) ∧
   T_4 = 7 * x ^ (2/3)) ∧
  -- 3. Rational terms
  (T_1 = x ^ 4 ∧
   T_6 = 7 / (16 * x)) := by
  sorry

end binomial_expansion_coefficients_l457_457455


namespace find_n_in_geometric_series_l457_457368

theorem find_n_in_geometric_series (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = 126 →
  S n = a 1 * (2^n - 1) / (2 - 1) →
  n = 6 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end find_n_in_geometric_series_l457_457368


namespace inverse_proportion_incorrect_D_l457_457793

theorem inverse_proportion_incorrect_D :
  ∀ (x y x1 y1 x2 y2 : ℝ), (y = -3 / x) ∧ (y1 = -3 / x1) ∧ (y2 = -3 / x2) ∧ (x1 < x2) → ¬(y1 < y2) :=
by
  sorry

end inverse_proportion_incorrect_D_l457_457793


namespace arithmetic_sequence_S11_l457_457233

open ArithmeticSequence

theorem arithmetic_sequence_S11 (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) :
  S 8 - S 3 = 20 →
  (∀ n, a (n + 1) = a n + d) →
  S = λ n, n * (a 1 + a n) / 2 →
  S 11 = 44 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_S11_l457_457233


namespace geom_mean_4_16_l457_457079

theorem geom_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
by
  sorry

end geom_mean_4_16_l457_457079


namespace total_income_l457_457272

def ron_ticket_price : ℝ := 2.00
def kathy_ticket_price : ℝ := 4.50
def total_tickets : ℕ := 20
def ron_tickets_sold : ℕ := 12

theorem total_income : ron_tickets_sold * ron_ticket_price + (total_tickets - ron_tickets_sold) * kathy_ticket_price = 60.00 := by
  sorry

end total_income_l457_457272


namespace vector_expression_equilateral_triangle_l457_457827

variables {A B C G : Type*} [euclidean_space V a]

-- Assuming A, B, C are points in a Euclidean space,
-- and G is the centroid of the equilateral triangle ABC.
def is_equilateral_triangle (A B C : V) (m : ℝ) : Prop :=
  dist A B = m ∧ dist B C = m ∧ dist C A = m

def is_centroid (A B C G : V) : Prop :=
  let D := midpoint B C in
  dist A G = (2 / 3) * dist A D ∧
  dist B G = (2 / 3) * dist B D ∧
  dist C G = (2 / 3) * dist C D

theorem vector_expression_equilateral_triangle
  {A B C G : V} {m : ℝ}
  (h1 : is_equilateral_triangle A B C m)
  (h2 : is_centroid A B C G) :
  (vector AB + vector BG) • (vector AB - vector AC) = 0 :=
sorry

end vector_expression_equilateral_triangle_l457_457827


namespace n_n_plus_one_div_by_2_l457_457337

theorem n_n_plus_one_div_by_2 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 99) : 2 ∣ n * (n + 1) :=
by
  sorry

end n_n_plus_one_div_by_2_l457_457337


namespace perpendicular_dot_product_l457_457130

theorem perpendicular_dot_product (λ μ : ℝ) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → (λ * μ = -1) :=
by
  intros
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (1, -1)
  let v1 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let v2 := (a.1 + μ * b.1, a.2 + μ * b.2)
  sorry

end perpendicular_dot_product_l457_457130


namespace cos_8_arccos_1_over_4_eq_neg_16286_over_16384_l457_457382

theorem cos_8_arccos_1_over_4_eq_neg_16286_over_16384 : 
  cos (8 * arccos (1 / 4)) = - 16286 / 16384 :=
by
  sorry

end cos_8_arccos_1_over_4_eq_neg_16286_over_16384_l457_457382


namespace number_of_squares_with_odd_tens_digit_l457_457739

theorem number_of_squares_with_odd_tens_digit :
  let count_odd_tens_digit := λ (n : ℕ), ((n ^ 2 / 10) % 10) % 2 = 1 in
  (list.range 95).filter count_odd_tens_digit.length = 19 :=
sorry

end number_of_squares_with_odd_tens_digit_l457_457739


namespace contractor_engaged_days_l457_457355

theorem contractor_engaged_days (x y : ℕ) (earnings_per_day : ℕ) (fine_per_day : ℝ) 
    (total_earnings : ℝ) (absent_days : ℕ) 
    (h1 : earnings_per_day = 25) 
    (h2 : fine_per_day = 7.50) 
    (h3 : total_earnings = 555) 
    (h4 : absent_days = 6) 
    (h5 : total_earnings = (earnings_per_day * x : ℝ) - fine_per_day * y) 
    (h6 : y = absent_days) : 
    x = 24 := 
by
  sorry

end contractor_engaged_days_l457_457355


namespace distance_between_centers_of_intersecting_circles_l457_457315

theorem distance_between_centers_of_intersecting_circles
  {r R d : ℝ} (hrR : r < R) (hr : 0 < r) (hR : 0 < R)
  (h_intersect : d < r + R ∧ d > R - r) :
  R - r < d ∧ d < r + R := by
  sorry

end distance_between_centers_of_intersecting_circles_l457_457315


namespace literate_employees_l457_457889

theorem literate_employees (num_illiterate : ℕ) (wage_decrease_per_illiterate : ℕ)
  (total_average_salary_decrease : ℕ) : num_illiterate = 35 → 
                                        wage_decrease_per_illiterate = 25 →
                                        total_average_salary_decrease = 15 →
                                        ∃ L : ℕ, L = 23 :=
by {
  -- given: num_illiterate = 35
  -- given: wage_decrease_per_illiterate = 25
  -- given: total_average_salary_decrease = 15
  sorry
}

end literate_employees_l457_457889


namespace circumcenter_concyclic_of_orthocenter_incenter_concyclic_l457_457637

open EuclideanGeometry

variables {A B C M O K : Point}

-- Assume triangle ABC and definitions of orthocenter M, incenter O, and circumcenter K
variable [triangle ABC]

/-- M is the orthocenter of triangle ABC -/
def is_orthocenter (M : Point) (A B C : Point) : Prop :=
  ∀ (D E F : Point), is_orthocenter_of_triangle A B C M

/-- O is the incenter of triangle ABC -/
def is_incenter (O : Point) (A B C : Point) : Prop :=
  incenter_of_triangle O A B C

/-- K is the circumcenter of triangle ABC -/
def is_circumcenter (K : Point) (A B C : Point) : Prop :=
  circumcenter_of_triangle K A B C

/-- The points A, B, O, M are concyclic -/
def are_concyclic (A B O M : Point) : Prop :=
  cyclic (circle A B O M)

theorem circumcenter_concyclic_of_orthocenter_incenter_concyclic
  (h1 : is_orthocenter M A B C)
  (h2 : is_incenter O A B C)
  (h3 : is_circumcenter K A B C)
  (h4 : are_concyclic A B O M) : 
  are_concyclic A B O K :=
sorry

end circumcenter_concyclic_of_orthocenter_incenter_concyclic_l457_457637


namespace distance_from_tangency_to_tangent_theorem_l457_457422

noncomputable def distance_from_tangency_to_tangent (R r : ℝ) : ℝ :=
  2 * R * r / (R + r)

theorem distance_from_tangency_to_tangent_theorem (R r : ℝ) :
  ∃ d : ℝ, d = distance_from_tangency_to_tangent R r :=
by
  use 2 * R * r / (R + r)
  sorry

end distance_from_tangency_to_tangent_theorem_l457_457422


namespace mark_has_seven_butterfingers_l457_457254

/-
  Mark has 12 candy bars in total between Mars bars, Snickers, and Butterfingers.
  He has 3 Snickers and 2 Mars bars.
  Prove that he has 7 Butterfingers.
-/

noncomputable def total_candy_bars : Nat := 12
noncomputable def snickers : Nat := 3
noncomputable def mars_bars : Nat := 2
noncomputable def butterfingers : Nat := total_candy_bars - (snickers + mars_bars)

theorem mark_has_seven_butterfingers : butterfingers = 7 := by
  sorry

end mark_has_seven_butterfingers_l457_457254


namespace _l457_457081

noncomputable theorem projection_of_a_in_direction_of_b 
  (a b : ℝ^3) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = √2) 
  (h_orthogonal : a ⊥ (a - b)) 
  : 
  ‖a ‖ * (a • b) / ‖b‖ = √2 / 2 :=
by {
  -- The statement describes that the projection of vector a in the direction of b is (√2)/2,
  -- given the conditions.
  sorry
}

end _l457_457081


namespace point_in_fourth_quadrant_l457_457527

def point : ℝ × ℝ := (3, -4)

def isFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def isSecondQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def isThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def isFourthQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : isFourthQuadrant point :=
by
  sorry

end point_in_fourth_quadrant_l457_457527


namespace CookiesEatenMoreThanGave_l457_457589

variable {PacoCookies : Nat := 17}
variable {ateCookies : Nat := 14}
variable {gaveCookies : Nat := 3}

theorem CookiesEatenMoreThanGave :
  ateCookies - gaveCookies = 11 :=
by
  sorry

end CookiesEatenMoreThanGave_l457_457589


namespace unattainable_y_l457_457428

theorem unattainable_y (x : ℝ) (h : x ≠ -(5 / 4)) :
    (∀ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3 / 4) :=
by
  -- Placeholder for the proof
  sorry

end unattainable_y_l457_457428


namespace MN_length_correct_l457_457899

noncomputable def length_MN
  (a b c : ℝ)
  (abc_triangle : ∀ (A B C : ℝ^2), ∃ DE : ℝ, ∃  (M N : ℝ^2), 
    DE = c / 2 ∧ |BC| = a ∧ |AC| = b ∧ |AB| = c ∧ 
    is_parallel DE AB ∧ intersects_circle_on_DE AC BC M N) : ℝ :=
  c * (a^2 + b^2 - c^2) / (4 * a * b)

theorem MN_length_correct (a b c : ℝ)
  (abc_triangle : ∀ (A B C : ℝ^2), ∃ DE : ℝ, ∃  (M N : ℝ^2), 
    DE = c / 2 ∧ |BC| = a∧ |AC| = b ∧ |AB| = c ∧ 
    is_parallel DE AB ∧ intersects_circle_on_DE AC BC M N) : 
  length_MN a b c abc_triangle = c * (a^2 + b^2 - c^2) / (4 * a * b) :=
sorry

end MN_length_correct_l457_457899


namespace phase_shift_of_sine_l457_457786

theorem phase_shift_of_sine :
  ∀ A B C x : ℝ, A = 4 → B = 3 → C = -π / 4 →
  ∀ y, y = A * sin (B * x + C) →
  (C / B) = -π / 12 :=
by
  intros A B C x hA hB hC y hy,
  rw [hA, hB, hC] at *,
  sorry

end phase_shift_of_sine_l457_457786


namespace factory_selection_and_probability_l457_457664

/-- Total number of factories in districts A, B, and C --/
def factories_A := 18
def factories_B := 27
def factories_C := 18

/-- Total number of factories and sample size --/
def total_factories := factories_A + factories_B + factories_C
def sample_size := 7

/-- Number of factories selected from districts A, B, and C --/
def selected_from_A := factories_A * sample_size / total_factories
def selected_from_B := factories_B * sample_size / total_factories
def selected_from_C := factories_C * sample_size / total_factories

/-- Number of ways to choose 2 factories out of the 7 --/
noncomputable def comb_7_2 := Nat.choose 7 2

/-- Number of favorable outcomes where at least one factory comes from district A --/
noncomputable def favorable_outcomes := 11

/-- Probability that at least one of the 2 factories comes from district A --/
noncomputable def probability := favorable_outcomes / comb_7_2

theorem factory_selection_and_probability :
  selected_from_A = 2 ∧ selected_from_B = 3 ∧ selected_from_C = 2 ∧ probability = 11 / 21 := by
  sorry

end factory_selection_and_probability_l457_457664


namespace sequence_general_term_sequence_sum_terms_l457_457212

-- Problem 1: General term formula of the sequence {a_n}
theorem sequence_general_term : 
  ∀ (a : ℕ → ℕ), (a 1 = 2) → (∀ n : ℕ, a (n + 1) = 2 * a n) → ∀ n : ℕ, a n = 2^n :=
by
  intros a h1 hr n
  sorry

-- Problem 2: Sum of the first n terms of the sequence {1 / (b_n * b_(n+1))}
theorem sequence_sum_terms : 
  ∀ (a : ℕ → ℕ), (a 1 = 2) → (∀ n : ℕ, a (n + 1) = 2 * a n) →
  ∀ (b : ℕ → ℕ), (∀ n : ℕ, b n = nat.log 2 (a n)) →
  ∀ n : ℕ, (∑ i in finset.range n, 1 / (b i * b (i+1)) = n / (n + 1)) :=
by
  intros a h1 hr b hb n
  sorry

end sequence_general_term_sequence_sum_terms_l457_457212


namespace correct_propositions_l457_457208

def def_increasing_function (f : ℝ → ℝ) := ∀ (a b : ℝ), a < b → f a ≤ f b

def test_function (x : ℝ) : ℝ := 2 * x + 1

theorem correct_propositions 
  (f : ℝ → ℝ) 
  (h1 : def_increasing_function f) 
  (h2 : f = test_function) : 
  (def_increasing_function (λ x, 2 * x + 1) = true) := 
by
  sorry

end correct_propositions_l457_457208


namespace hundredth_number_in_set_l457_457742

theorem hundredth_number_in_set : 
  let S := { n : ℕ | ∃ x y z : ℕ, x < y ∧ y < z ∧ n = 2^x + 2^y + 2^z } in
  let sorted_S := List.qsort (λ a b, a < b) (Set.toList S) in
  sorted_S.get? 99 = some 524 := by
  sorry

end hundredth_number_in_set_l457_457742


namespace B_and_C_together_l457_457734

-- Defining the variables and conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 500)
variable (h2 : A + C = 200)
variable (h3 : C = 50)

-- The theorem to prove that B + C = 350
theorem B_and_C_together : B + C = 350 :=
by
  -- Replacing with the actual proof steps
  sorry

end B_and_C_together_l457_457734


namespace area_of_rectangle_A_B_C_D_same_as_solution_result_l457_457185

noncomputable def find_area_of_ABCD (A B C D F E : ℝ → ℝ → Prop) : ℝ :=
  if h : (is_rectangle A B C D) ∧ (on_line F A D) ∧ (on_line E A B) ∧ (distance B E = 4) ∧ (distance A F = 3) 
    then 32 - 12 * real.sqrt 2
  else 0

-- Verify that the area of the rectangle ABCD is as calculated
theorem area_of_rectangle_A_B_C_D_same_as_solution_result (A B C D F E : ℝ → ℝ → Prop)
  (h : (is_rectangle A B C D) ∧ (on_line F A D) ∧ (on_line E A B) ∧ (distance B E = 4) ∧ (distance A F = 3)) :
  find_area_of_ABCD A B C D F E = 32 - 12 * real.sqrt 2 :=
begin
  unfold find_area_of_ABCD,
  split_ifs,
  refl,
  contradiction
end

end area_of_rectangle_A_B_C_D_same_as_solution_result_l457_457185


namespace maximum_cards_l457_457907

def total_budget : ℝ := 15
def card_cost : ℝ := 1.25
def transaction_fee : ℝ := 2
def desired_savings : ℝ := 3

theorem maximum_cards : ∃ n : ℕ, n ≤ 8 ∧ (card_cost * (n : ℝ) + transaction_fee ≤ total_budget - desired_savings) :=
by sorry

end maximum_cards_l457_457907


namespace row_column_crossout_l457_457891

theorem row_column_crossout (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ rows : Finset (Fin 1000), rows.card = 990 ∧ ∀ j : Fin 1000, ∃ i ∈ rowsᶜ, M i j = 1) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 990 ∧ ∀ i : Fin 1000, ∃ j ∈ colsᶜ, M i j = 0) :=
by {
  sorry
}

end row_column_crossout_l457_457891


namespace energy_function_relationship_minimum_energy_at_v_eq_6_l457_457023

variables (k v : ℝ)

-- Conditions
def energy_consumption (k v t : ℝ) := k * v^2 * t

def time_travel (v : ℝ) : ℝ := 10 / (v - 3)

def flow_speed := 3

-- Proof Problem
theorem energy_function_relationship (k : ℝ) (v : ℝ) (h₁ : v > 3) :
  energy_consumption k v (time_travel v) = k * v^2 * (10 / (v - 3)) :=
begin
  sorry
end

theorem minimum_energy_at_v_eq_6 (k : ℝ) (h₁ : 120 * k > 0) :
  let v := 6 in energy_consumption k v (time_travel v) = 120 * k :=
begin
  sorry
end

end energy_function_relationship_minimum_energy_at_v_eq_6_l457_457023


namespace vector_addition_correct_l457_457388

def vector_addition : Vector ℚ 2 := 
  let a : Vector ℚ 2 := ⟨[-3, 6]⟩
  let b : Vector ℚ 2 := ⟨[-2, 5]⟩
  4 • a + 3 • b

theorem vector_addition_correct : 
  vector_addition = ⟨[-18, 39]⟩ := 
by 
  sorry

end vector_addition_correct_l457_457388


namespace smallest_palindrome_not_five_digit_l457_457040

theorem smallest_palindrome_not_five_digit (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = n / 100 ∧ n / 10 % 10 = n / 100 ∧ 103 * n < 10000) :
  n = 707 := by
sorry

end smallest_palindrome_not_five_digit_l457_457040


namespace correct_proposition_l457_457808

theorem correct_proposition (f : ℝ → ℝ)
  (H1 : ∀ x : ℝ, f (-x) = -f x)
  (H2 : ∀ x : ℝ, x < 0 → f x = exp x * (x + 1)) :
  (∀ x > 0, f x = exp (-x) * (x - 1)) = false ∧ (∃! x, f x = 0) = false ∧ 
  (∃ a b : ℝ, -1 < a ∧ a < 0 ∧ f a > 0 ∧ 1 < b ∧ f b > 0) ∧ 
  (∀ x1 x2 : ℝ, abs (f x1 - f x2) < 2) := by {
  sorry
}

end correct_proposition_l457_457808


namespace cos_18_deg_l457_457006

theorem cos_18_deg :
  let x := real.cos (real.pi / 10) in
  let y := real.cos (2 * real.pi / 10) in
  (y = 2 * x^2 - 1) ∧
  (4 * x^3 - 3 * x = real.sin (real.pi * 2 / 5)) →
  (real.cos (real.pi / 10) = real.sqrt ((5 + real.sqrt 5) / 8)) :=
by
  sorry

end cos_18_deg_l457_457006


namespace sequence_periodic_l457_457847

def sequence (n : ℕ) : ℚ :=
  nat.rec_on n 2 (λ n a_n, (1 + a_n) / (1 - a_n))

theorem sequence_periodic :
  sequence 2018 = -3 :=
by sorry

end sequence_periodic_l457_457847


namespace isaiah_typing_rate_l457_457576

theorem isaiah_typing_rate (m : ℕ) (h_m : m = 20) : 
  let wpm_m := m in
  let wph_m := wpm_m * 60 in
  let wph_i := wph_m + 1200 in
  wph_i / 60 = 40 := 
by
  sorry

end isaiah_typing_rate_l457_457576


namespace g_978_eq_1007_l457_457629

noncomputable def g : ℤ → ℤ
| m := if m ≥ 1010 then m - 3 else g (g (m + 5))

theorem g_978_eq_1007 : g 978 = 1007 := by
  sorry

end g_978_eq_1007_l457_457629


namespace distance_A_O_l457_457750

-- Define the coordinates of point A
def A := (1,2,-1)

-- Define the origin O
def O := (0,0,0)

-- Function to compute the distance in 3D space
def distance (p q : ℝ × ℝ × ℝ) : ℝ := 
  let (x1, y1, z1) := p
  let (x2, y2, z2) := q
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Proof statement
theorem distance_A_O : distance A O = Real.sqrt 6 := by
  sorry

end distance_A_O_l457_457750


namespace problem1_problem2_l457_457383

-- Problem 1
theorem problem1 : (-1 : ℝ)^4 - 7 / (2 - (-3 : ℝ)^2) = 0 := by
  sorry

-- Problem 2
theorem problem2 : 
  let angle1 := 56 + 17 / 60 in
  let angle2 := 12 + 45 / 60 in
  let angle3 := 16 + 21 / 60 in
  angle1 + angle2 - 4 * angle3 = 3 + 38 / 60 := by
  sorry

end problem1_problem2_l457_457383


namespace carol_age_difference_l457_457304

theorem carol_age_difference (bob_age carol_age : ℕ) (h1 : bob_age + carol_age = 66)
  (h2 : carol_age = 3 * bob_age + 2) (h3 : bob_age = 16) (h4 : carol_age = 50) :
  carol_age - 3 * bob_age = 2 :=
by
  sorry

end carol_age_difference_l457_457304


namespace sum_of_numbers_l457_457957

theorem sum_of_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000)
(h_eq : 100 * x + y = 7 * x * y) : x + y = 18 :=
sorry

end sum_of_numbers_l457_457957


namespace rectangle_area_l457_457724

theorem rectangle_area
  (x y : ℝ) -- sides of the rectangle
  (h1 : 2 * x + 2 * y = 12)  -- perimeter
  (h2 : x^2 + y^2 = 25)  -- diagonal
  : x * y = 5.5 :=
sorry

end rectangle_area_l457_457724


namespace part1_part2_l457_457941

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ := Real.tan ((x / 2) - (Real.pi / 3))

-- Part (1)
theorem part1 : f (5 * Real.pi / 2) = Real.sqrt 3 - 2 :=
by
  sorry

-- Part (2)
theorem part2 (k : ℤ) : { x : ℝ | f x ≤ Real.sqrt 3 } = 
  {x | ∃ (k : ℤ), 2 * k * Real.pi - Real.pi / 3 < x ∧ x ≤ 2 * k * Real.pi + 4 * Real.pi / 3} :=
by
  sorry

end ProofProblem

end part1_part2_l457_457941


namespace transform_preserves_sum_of_squares_transformation_impossible_l457_457056

def sum_of_squares (xs : List ℝ) : ℝ :=
  List.sum (xs.map (λ x => x^2))

def transform (xs : List ℝ) (i j : ℕ) : List ℝ :=
  let x := xs.nth_le i (by sorry)
  let y := xs.nth_le j (by sorry)
  xs.modify_nth i (λ _ => (x + y) / Real.sqrt 2)
     .modify_nth j (λ _ => (x - y) / Real.sqrt 2)

theorem transform_preserves_sum_of_squares (xs : List ℝ) (i j : ℕ) :
  sum_of_squares xs = sum_of_squares (transform xs i j) :=
by sorry

def initial_quadruplet : List ℝ := [2, Real.sqrt 2, 1 / Real.sqrt 2, -Real.sqrt 2]
def target_quadruplet : List ℝ := [1, 2 * Real.sqrt 2, 1 - Real.sqrt 2, Real.sqrt 2 / 2]

theorem transformation_impossible :
  ∀ (seq : List ℝ),
  (seq = initial_quadruplet → 
  (∀ i j, transform seq i j ≠ target_quadruplet)) :=
by sorry

end transform_preserves_sum_of_squares_transformation_impossible_l457_457056


namespace multiply_expand_l457_457973

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l457_457973


namespace fixed_point_of_moving_line_range_of_d_l457_457446

-- Fixed point problem
theorem fixed_point_of_moving_line
  (h_ellipse : ∀ x y : ℝ, x^2 / 2 + y^2 = 1)
  (h_focus : F : ℝ × ℝ)
  (h_line : ∀ k b : ℝ, (k ≠ 0) ∧ (line_intersects k b))
  (h_not_perpendicular : ∀ k, k ≠ 0)
  (h_slopes_sum_zero : ∀ a b : ℝ × ℝ, (a.2 - F.2) / (a.1 - F.1) + (b.2 - F.2) / (b.1 - F.1) = 0) :
  ∃ p : ℝ × ℝ, p = (-2, 0) := 
sorry

-- Range of d problem
theorem range_of_d
  (h_ellipse : ∀ x y : ℝ, x^2 / 2 + y^2 = 1)
  (h_focus : F : ℝ × ℝ)
  (h_line : ∀ k b : ℝ, (k ≠ 0) ∧ (line_intersects k b))
  (h_not_perpendicular : ∀ k, k ≠ 0)
  (h_perpendicular : ∀ a b : ℝ × ℝ, (a.2 - F.2) * (b.2 - F.2) + (a.1 - F.1) * (b.1 - F.1) = 0)
  (h_dist : ∀ d : ℝ, d = distance_from_origin k b) :
  0 < d ∧ d < 4/3 := 
sorry

end fixed_point_of_moving_line_range_of_d_l457_457446


namespace part_i_part_ii_l457_457558

variables {G : Type*} [Graph G] {a b : vertex G} 

theorem part_i (h_distinct : a ≠ b) (h_no_edge : ¬ (a -- b)) :
  min_separating_set_size a b G = max_independent_paths a b G :=
sorry

theorem part_ii (h_distinct : a ≠ b) (h_no_edge : ¬ (a -- b)) :
  min_edge_separating_set_size a b G = max_edge_disjoint_paths a b G :=
sorry

end part_i_part_ii_l457_457558


namespace congcong_incorrect_number_huihui_difference_l457_457836

theorem congcong_incorrect_number
  (correct_expr_result : (-1-8) * 2 - 5 = -23)
  (congcong_result : (-9) * 2 - 6 = -24) : 
  5 = 6 - 1 := by
  sorry

theorem huihui_difference
  (correct_expr_result : (-1-8) * 2 - 5 = -23)
  (huihui_result : (-1-8) + 2 - 5 = -12) :
  huihui_result - correct_expr_result = 11 := by
  sorry

end congcong_incorrect_number_huihui_difference_l457_457836


namespace butterfinger_count_l457_457257

def total_candy_bars : ℕ := 12
def snickers : ℕ := 3
def mars_bars : ℕ := 2
def butterfingers : ℕ := total_candy_bars - (snickers + mars_bars)

theorem butterfinger_count : butterfingers = 7 :=
by
  unfold butterfingers
  sorry

end butterfinger_count_l457_457257


namespace perimeter_bisectors_concur_l457_457812

open Real

/-- Definition of a perimeter bisector. -/
def is_perimeter_bisector {A B C : Point} (P : Point) : Prop :=
  perimeter (triangle A B P) = perimeter (triangle A C P)

/-- In any triangle ABC, the three perimeter bisectors intersect at a single point. -/
theorem perimeter_bisectors_concur {A B C : Point} :
  ∃ P : Point, (is_perimeter_bisector A P) ∧ (is_perimeter_bisector B P) ∧ (is_perimeter_bisector C P) :=
sorry

end perimeter_bisectors_concur_l457_457812


namespace log_relationship_l457_457819

open Real

theorem log_relationship (a b m n l : ℝ) (h1 : a > b) (h2 : b > 1) 
                         (h3 : m = log a (log a b)) 
                         (h4 : n = (log a b)^2) 
                         (h5 : l = log a (b^2)) : 
                         l > n ∧ n > m := 
by 
    have hlog1 : 0 < log a b := 
        by sorry 
    have hlog2 : log a b < 1 := 
        by sorry 
    have hm : m = log a (log a b) := 
        by assume h3 
    have hn : n = (log a b)^2 := 
        by assume h4 
    have hl : l = log a (b^2) := 
        by assume h5 
    have lt_1 := mul_lt_mul_of_pos_left hlog2 (lt_trans zero_lt_one h2) 
    have lt_2 := pow_two hlog1 
    show l > n, from sorry 
    show n > m, from sorry 
    exact ⟨sorry, sorry⟩

end log_relationship_l457_457819


namespace exists_n_divides_2022n_minus_n_l457_457943

theorem exists_n_divides_2022n_minus_n (p : ℕ) [hp : Fact (Nat.Prime p)] :
  ∃ n : ℕ, p ∣ (2022^n - n) :=
sorry

end exists_n_divides_2022n_minus_n_l457_457943


namespace prime_squared_mod_six_l457_457692

theorem prime_squared_mod_six (p : ℕ) (hp1 : p > 5) (hp2 : Nat.Prime p) : (p ^ 2) % 6 = 1 :=
sorry

end prime_squared_mod_six_l457_457692


namespace prove_tan_A_max_value_ratio_min_value_bc_a_squared_l457_457214

variables {A B C : Type} [RealField A] [RealField B] [RealField C]

noncomputable def S (a : A) : A -> A := λ a, (a ^ 2) / 2

theorem prove_tan_A (a b c : ℝ) (h1 : S a = (a ^ 2) / 2)
                    (h2 : ∀ S, S = (1 / 2) * b * c * Real.sin A) :
  Real.tan A = (2 * a ^ 2) / (b ^ 2 + c ^ 2 - a ^ 2) :=
by
  sorry

theorem max_value_ratio (a b c : ℝ) (h1 : S a = (a ^ 2) / 2) :
  Real.max (c / b + b / c) = Real.sqrt 5 :=
by
  sorry

theorem min_value_bc_a_squared (a b c : ℝ) (h1 : S a = (a ^ 2) / 2)
                                (h2 : ∀ S, S = (1 / 2) * b * c * Real.sin A) :
  Real.min (b * c / a ^ 2) = 1 :=
by
  sorry

end prove_tan_A_max_value_ratio_min_value_bc_a_squared_l457_457214


namespace closest_point_on_line_to_target_l457_457787

noncomputable def parametricPoint (s : ℝ) : ℝ × ℝ × ℝ :=
  (6 + 3 * s, 2 - 9 * s, 0 + 6 * s)

noncomputable def closestPoint : ℝ × ℝ × ℝ :=
  (249/42, 95/42, -1/7)

theorem closest_point_on_line_to_target :
  ∃ s : ℝ, parametricPoint s = closestPoint :=
by
  sorry

end closest_point_on_line_to_target_l457_457787


namespace find_time_before_l457_457744

constant starting_time : ℕ → ℕ
constant duration : ℕ

-- Define the specific values given in the conditions
def time_7am : ℕ := 7
def hours_100 : ℕ := 100

-- Function to compute time 100 hours before a given time
noncomputable def time_before : ℕ → ℕ → ℕ
| t, d := sorry -- Placeholder for the actual time computation logic 

-- Define the result we need to prove
def expected_time_before : ℕ := 3

-- The theorem stating what we need to prove
theorem find_time_before : time_before time_7am hours_100 = expected_time_before :=
sorry

end find_time_before_l457_457744


namespace total_points_scored_l457_457508

theorem total_points_scored (n m T : ℕ) 
  (h1 : T = 2 * n + 5 * m) 
  (h2 : n = m + 3 ∨ m = n + 3)
  : T = 20 :=
sorry

end total_points_scored_l457_457508


namespace probability_three_odd_rolls_of_eight_is_7_over_32_l457_457369

theorem probability_three_odd_rolls_of_eight_is_7_over_32 :
  let p := 1 / 2,
      n := 8,
      k := 3 in
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 7 / 32 := 
by
  sorry

end probability_three_odd_rolls_of_eight_is_7_over_32_l457_457369


namespace total_children_given_candies_l457_457377

theorem total_children_given_candies : 
  ∀ (total_candies : ℕ) (one_third_candies : ℕ) (remaining_candies : ℕ) 
  (lollipops_per_boy : ℕ) (candy_canes_per_girl : ℕ), 
  total_candies = 90 →
  one_third_candies = total_candies / 3 →
  remaining_candies = total_candies - one_third_candies →
  lollipops_per_boy = 3 →
  candy_canes_per_girl = 2 →
  (let number_of_boys := one_third_candies / lollipops_per_boy in
   let number_of_girls := remaining_candies / candy_canes_per_girl in
   number_of_boys + number_of_girls = 40) :=
by
  intros total_candies one_third_candies remaining_candies lollipops_per_boy candy_canes_per_girl
  intros c1 c2 c3 c4 c5
  let number_of_boys := one_third_candies / lollipops_per_boy
  let number_of_girls := remaining_candies / candy_canes_per_girl
  have h1 : one_third_candies = 30 := sorry
  have h2 : remaining_candies = 60 := sorry
  have h3 : number_of_boys = 10 := sorry
  have h4 : number_of_girls = 30 := sorry
  show number_of_boys + number_of_girls = 40
  calc
    number_of_boys + number_of_girls = 10 + 30 : by rw [h3, h4]
    ... = 40 : by norm_num


end total_children_given_candies_l457_457377


namespace digit_sum_solution_l457_457561

def S (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_solution : S (S (S (S (2017 ^ 2017)))) = 1 := 
by
  sorry

end digit_sum_solution_l457_457561


namespace lagranges_identity_cauchy_schwarz_l457_457265

theorem lagranges_identity (a b : Fin n → ℝ) :
  (∑ i in Finset.finRange n, (a i)^2) * (∑ i in Finset.finRange n, (b i)^2) =
  (∑ i in Finset.finRange n, (a i) * (b i))^2 +
  ∑ i in Finset.finRange n, ∑ j in Finset.finRange n, (a i * b j - a j * b i)^2 :=
sorry

theorem cauchy_schwarz (a b : Fin n → ℝ) :
  (∑ i in Finset.finRange n, (a i)^2) * (∑ i in Finset.finRange n, (b i)^2) ≥
  (∑ i in Finset.finRange n, (a i) * (b i))^2 :=
begin
  -- Apply Lagrange's identity
  have h : (∑ i in Finset.finRange n, (a i)^2) * (∑ i in Finset.finRange n, (b i)^2) =
           (∑ i in Finset.finRange n, (a i) * (b i))^2 +
           ∑ i in Finset.finRange n, ∑ j in Finset.finRange n, (a i * b j - a j * b i)^2,
  from lagranges_identity a b,
  -- Use the non-negativity of squares
  rw h,
  exact le_add_of_nonneg_right (sum_nonneg (λ i, sum_nonneg (λ j, by apply pow_two_nonneg))),
end

end lagranges_identity_cauchy_schwarz_l457_457265


namespace total_rainfall_in_2011_l457_457504

-- Define the given conditions
def avg_monthly_rainfall_2010 : ℝ := 36.8
def increase_2011 : ℝ := 3.5

-- Define the resulting average monthly rainfall in 2011
def avg_monthly_rainfall_2011 : ℝ := avg_monthly_rainfall_2010 + increase_2011

-- Calculate the total annual rainfall
def total_rainfall_2011 : ℝ := avg_monthly_rainfall_2011 * 12

-- State the proof problem
theorem total_rainfall_in_2011 :
  total_rainfall_2011 = 483.6 := by
  sorry

end total_rainfall_in_2011_l457_457504


namespace minimum_value_of_f_sum_of_roots_greater_than_2_l457_457839

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

open Real

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by
  use 1
  simp [f, log_one, one_div]

theorem sum_of_roots_greater_than_2 (a : ℝ) (x₁ x₂ : ℝ) (hx₁ : f x₁ = a) (hx₂ : f x₂ = a) (h_order : x₁ < x₂) : x₁ + x₂ > 2 :=
by
  have h₁ : x₁ > 0, from sorry
  have h₂ : x₂ > 0, from sorry
  let t := x₂ / x₁
  have ht : t > 1, from sorry
  let g t := t - 1 / t - 2 * log t
  have g_increasing : ∀ t > 1, (λ t, t - 1 / t - 2 * log t) ≥ 0, from sorry
  exact g_increasing t ht

end minimum_value_of_f_sum_of_roots_greater_than_2_l457_457839


namespace contradiction_assumption_l457_457540

theorem contradiction_assumption (a b : ℝ) (h : |a - 1| * |b - 1| = 0) : ¬ (a ≠ 1 ∧ b ≠ 1) :=
  sorry

end contradiction_assumption_l457_457540


namespace midpoint_of_AB_is_correct_l457_457798

-- Define points A and B
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (3, 5)

-- Define the midpoint function
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the expected midpoint
def expected_midpoint : ℝ × ℝ := (1, 4)

-- State the theorem
theorem midpoint_of_AB_is_correct : midpoint A B = expected_midpoint := by
  sorry

end midpoint_of_AB_is_correct_l457_457798


namespace distance_to_origin_is_sqrt_2_l457_457247

-- Define the given complex numbers
def z1 : ℂ := complex.I
def z2 : ℂ := 1 + complex.I

-- Define the product of the complex numbers
def z : ℂ := z1 * z2

-- The proof statement: the distance from the origin to the point corresponding to complex number z is sqrt(2)
theorem distance_to_origin_is_sqrt_2 : complex.abs z = real.sqrt 2 :=
by sorry

end distance_to_origin_is_sqrt_2_l457_457247


namespace max_snacks_with_15_dollars_l457_457999

def snack_price : Type :=
  { price : ℕ × ℕ // price.1 > 0 ∧ price.2 > 0 }

def individual_snack : snack_price := ⟨(2, 1), by simp⟩
def four_snack_pack : snack_price := ⟨(6, 4), by simp⟩
def seven_snack_pack : snack_price := ⟨(9, 7), by simp⟩

def snacks_purchase_max (budget : ℕ) (prices : list snack_price) : ℕ :=
  let total_snacks : ℕ :=
    prices.foldl (λ total current, total + current.val.2 * (budget / current.val.1)) 0
  in total_snacks

theorem max_snacks_with_15_dollars : snacks_purchase_max 15 [seven_snack_pack, four_snack_pack, individual_snack] = 11 :=
sorry

end max_snacks_with_15_dollars_l457_457999


namespace yearly_cost_of_oil_changes_l457_457919

-- Definitions of conditions
def miles_per_month : ℕ := 1000
def months_in_year : ℕ := 12
def oil_change_frequency : ℕ := 3000
def free_oil_changes_per_year : ℕ := 1
def cost_per_oil_change : ℕ := 50

theorem yearly_cost_of_oil_changes : 
  let total_miles := miles_per_month * months_in_year in
  let total_oil_changes := total_miles / oil_change_frequency in
  let paid_oil_changes := total_oil_changes - free_oil_changes_per_year in
  paid_oil_changes * cost_per_oil_change = 150 := 
by
  sorry

end yearly_cost_of_oil_changes_l457_457919


namespace find_x_l457_457487

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem find_x
  (h : dot_product vector_a (vector_b x) = 0) :
  x = 2 :=
by
  sorry

end find_x_l457_457487


namespace a_gen_formula_T_gen_formula_l457_457571

noncomputable theory

def S (n : ℕ) : ℕ := (3^n + 3) / 2

def a (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 3^(n - 1)

def b (n : ℕ) : ℕ :=
  if n = 1 then 1 / 3
  else (n - 1) / 3^(n - 1)

def T (n : ℕ) : ℕ :=
  (13 / 12) - (6 * n + 3) / (4 * 3^n)

theorem a_gen_formula (n : ℕ) : a n = if n = 1 then 3 else 3^(n - 1) :=
sorry

theorem T_gen_formula (n : ℕ) : T n = (13 / 12) - (6 * n + 3) / (4 * 3^n) :=
sorry

end a_gen_formula_T_gen_formula_l457_457571


namespace volume_ratio_sphere_cylinder_inscribed_l457_457728

noncomputable def ratio_of_volumes (d : ℝ) : ℝ :=
  let Vs := (4 / 3) * Real.pi * (d / 2)^3
  let Vc := Real.pi * (d / 2)^2 * d
  Vs / Vc

theorem volume_ratio_sphere_cylinder_inscribed (d : ℝ) (h : d > 0) : 
  ratio_of_volumes d = 2 / 3 := 
by
  sorry

end volume_ratio_sphere_cylinder_inscribed_l457_457728


namespace lambda_mu_condition_l457_457110

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem lambda_mu_condition (λ μ : ℝ) :
  orthogonal (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
             (vector_a.1 + μ * vector_b.1, vector_a.2 + μ * vector_b.2) →
  λ * μ = -1 :=
by
  sorry

end lambda_mu_condition_l457_457110


namespace lines_perpendicular_l457_457207

-- We need to define the positional relationship being "perpendicular".
def is_perpendicular (L1 L2 : ℝ × ℝ → Prop) : Prop := 
  ∀ x y : ℝ, L1 (x, y) → L2 (y, -x) 

-- Define the two lines in question using the given conditions.
def line1 (ρ θ α : ℝ) (coords : ℝ × ℝ) : Prop := 
  let (x, y) := coords in x * Real.cos α + y * Real.sin α = 0

def line2 (ρ θ α a : ℝ) (coords : ℝ × ℝ) : Prop := 
  let (x, y) := coords in -x * Real.sin α + y * Real.cos α = a

-- The theorem that we need to prove is that the two lines are perpendicular.
theorem lines_perpendicular (α a ρ θ : ℝ) :
  is_perpendicular (line1 ρ θ α) (line2 ρ θ α a) :=
sorry

end lines_perpendicular_l457_457207


namespace positive_rationals_as_factorial_fractions_l457_457597

noncomputable def prod_factorials (p : ℕ) : ℕ :=
  p.factorial

theorem positive_rationals_as_factorial_fractions (q : ℚ) (hq : 0 < q) :
  ∃ (a b : ℚ), (a = (π : ℚ → ℕ) → (∏ p in (filter (λ x, is_prime x) (range q.numerator.succ)), p.factorial)) ∧
               (b = (π : ℚ → ℕ) → (∏ p in (filter (λ x, is_prime x) (range q.denominator.succ)), p.factorial)) ∧
               q = a / b := sorry

end positive_rationals_as_factorial_fractions_l457_457597


namespace sum_diff_l457_457320

theorem sum_diff (n : ℕ) (h : n = 3000) :
  let S_odd := (3000 * (1 + 5999)) / 2,
      S_even_plus_5 := (3000 * (7 + 6005)) / 2 in
  S_even_plus_5 - S_odd = 18000 :=
by
  sorry

end sum_diff_l457_457320


namespace stable_performance_l457_457028

/-- The variance of student A's scores is 0.4 --/
def variance_A : ℝ := 0.4

/-- The variance of student B's scores is 0.3 --/
def variance_B : ℝ := 0.3

/-- Prove that student B has more stable performance given the variances --/
theorem stable_performance (h1 : variance_A = 0.4) (h2 : variance_B = 0.3) : variance_B < variance_A :=
by
  rw [h1, h2]
  exact sorry

end stable_performance_l457_457028


namespace circle_equation_correct_l457_457648

theorem circle_equation_correct (x y : ℝ) :
  let h : ℝ := -2
  let k : ℝ := 2
  let r : ℝ := 5
  ((x - h)^2 + (y - k)^2 = r^2) ↔ ((x + 2)^2 + (y - 2)^2 = 25) :=
by
  sorry

end circle_equation_correct_l457_457648


namespace perp_imp_intersect_and_coplanar_l457_457859

theorem perp_imp_intersect_and_coplanar (a b : Line) (h : a ⊥ b) : 
  ∃ P : Point, P ∈ a ∧ P ∈ b ∧ a.coplanar b :=
sorry

end perp_imp_intersect_and_coplanar_l457_457859


namespace general_formula_sum_of_terms_T_n_l457_457443

-- Definitions
variable {α : Type*}
def arithmetic_seq (a : ℕ → ℕ) := ∀ n m, a (n + m) = a n + a m
def sum_of_terms (s : ℕ → ℕ) := ∀ n, s n = n * (n + 1)

-- Given conditions
axiom a_3 : ℕ := 6
axiom S_11 : ℕ := 132

-- Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℕ) (d : ℕ) (h1 : a 3 = 6) (h2 : 11 * (a 6) = 132) :
  ∀ n, a n = 2 * n :=
  sorry

-- Prove the sum of the first n terms, T_n, for the sequence {1/S_n}
theorem sum_of_terms_T_n (s : ℕ → ℕ) (T : ℕ → ℕ) (h1 : sum_of_terms s) :
  ∀ n, T n = ∑ i in (range (n + 1)).erase 0, 1 / s i := 
  sorry

end general_formula_sum_of_terms_T_n_l457_457443


namespace max_a2_b2_l457_457903

theorem max_a2_b2 (a b c : ℝ) (h1 : a + b = c - 1) (h2 : ab = c^2 - 7c + 14) : a^2 + b^2 ≤ 8 := 
sorry

end max_a2_b2_l457_457903


namespace find_a_l457_457840

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (2 * x) - (1 / 3) * Real.sin (3 * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  2 * a * Real.cos (2 * x) - Real.cos (3 * x)

theorem find_a (a : ℝ) (h : f_prime a (Real.pi / 3) = 0) : a = 1 :=
by
  sorry

end find_a_l457_457840


namespace total_female_officers_on_police_force_l457_457584

theorem total_female_officers_on_police_force (F : ℕ) (H1 : 0.25 * F = 60) : F = 240 := by
  -- Placeholder for the proof
  sorry

end total_female_officers_on_police_force_l457_457584


namespace stock_quote_96_l457_457690

noncomputable def face_value (income: ℝ) (dividend_rate: ℝ) : ℝ :=
  (income * 100) / dividend_rate

noncomputable def stock_quote (market_value: ℝ) (face_value: ℝ) : ℝ :=
  (market_value / face_value) * 100

theorem stock_quote_96 (investment: ℝ) (dividend_rate: ℝ) (income: ℝ) (market_value: ℝ) (quote: ℝ) :
  investment = 1620 ∧ dividend_rate = 8 ∧ income = 135 ∧ market_value = 1620 ∧ quote = stock_quote market_value (face_value income dividend_rate) → quote = 96 :=
by
  intros h,
  sorry

end stock_quote_96_l457_457690


namespace multiply_expression_l457_457989

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l457_457989


namespace base5_to_base8_conversion_l457_457395

def base5_to_base10 (n : ℕ) : ℕ :=
  let digits := [4, 4, 4]  -- 444 in base-5 represented as a list of its digits
  digits.enum_from 0 |>.foldl (λ acc ⟨i, d⟩ => acc + d * 5 ^ i) 0

def base10_to_base8 (n : ℕ) : ℕ :=
  Nat.digits 8 n |>.reverse |>.enum_from 0 |>.foldl (λ acc ⟨i, d⟩ => acc + d * 10 ^ i) 0

theorem base5_to_base8_conversion : base10_to_base8 (base5_to_base10 444) = 174 :=
  sorry

end base5_to_base8_conversion_l457_457395


namespace pirate_rick_digging_time_l457_457591

theorem pirate_rick_digging_time :
  ∀ (initial_depth rate: ℕ) (storm_factor tsunami_added: ℕ),
  initial_depth = 8 →
  rate = 2 →
  storm_factor = 2 →
  tsunami_added = 2 →
  (initial_depth / storm_factor + tsunami_added) / rate = 3 := 
by
  intros
  sorry

end pirate_rick_digging_time_l457_457591


namespace minimum_value_of_expression_l457_457076

variable (a b c d : ℝ)

-- The given conditions:
def cond1 : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def cond2 : Prop := a^2 + b^2 = 4
def cond3 : Prop := c * d = 1

-- The minimum value:
def expression_value : ℝ := (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2)

theorem minimum_value_of_expression :
  cond1 a b c d → cond2 a b → cond3 c d → expression_value a b c d ≥ 16 :=
by
  sorry

end minimum_value_of_expression_l457_457076


namespace prove_OB_leq_OA1_div_4_l457_457071

noncomputable def four_lines_intersecting_at_O 
    (m1 m2 m3 m4 : Type) (O : Type) 
    (intersect : O = intersection_of_lines m1 m2 m3 m4) 
    (A1 : m1) 
    (A2 : Type) 
    (A2_condition : A2 = intersection_of_parallel_through_point m4 A1 m2)
    (A3 : Type)
    (A3_condition : A3 = intersection_of_parallel_through_point m1 A2 m3)
    (A4 : Type)
    (A4_condition : A4 = intersection_of_parallel_through_point m2 A3 m4)
    (B : Type)
    (B_condition : B = intersection_of_parallel_through_point m3 A4 m1) : Prop :=
  ∀ (OA1 OB : ℝ), OB ≤ OA1 / 4

variables 
  (m1 m2 m3 m4 : Type) (O : Type) 
  (intersect : O = intersection_of_lines m1 m2 m3 m4) 
  (A1 : m1) 
  (A2 : Type) 
  (OA1 : ℝ)
  (A2_condition : A2 = intersection_of_parallel_through_point m4 A1 m2)
  (A3 : Type)
  (A3_condition : A3 = intersection_of_parallel_through_point m1 A2 m3)
  (A4 : Type)
  (A4_condition : A4 = intersection_of_parallel_through_point m2 A3 m4)
  (B : Type)
  (OB : ℝ)
  (B_condition : B = intersection_of_parallel_through_point m3 A4 m1)

theorem prove_OB_leq_OA1_div_4 : four_lines_intersecting_at_O m1 m2 m3 m4 O intersect A1 A2 A2_condition A3 A3_condition A4 A4_condition B B_condition OA1 OB := 
  sorry

end prove_OB_leq_OA1_div_4_l457_457071


namespace jerry_money_from_aunt_l457_457552

-- Conditions translated to definitions
def from_friends : ℝ := 22 + 23 + 22 + 22
def from_sister : ℝ := 7
def total_from_friends_and_sister : ℝ := from_friends + from_sister
def mean (total : ℝ) (sources : ℝ) : ℝ := total / sources
def number_of_sources : ℝ := 7
def given_mean : ℝ := 16.3

-- Define the amount Jerry received from his aunt and uncle
variable (A : ℝ)

-- Proof problem statement
theorem jerry_money_from_aunt : 
  mean (total_from_friends_and_sister + 2 * A) number_of_sources = given_mean → 
  A = 9.05 := 
by 
-- placeholder for the proof
sorry

end jerry_money_from_aunt_l457_457552


namespace tv_cost_correct_l457_457572

-- Define Linda's total savings as a constant
def linda_savings : ℝ := 3000.0000000000005

-- The fraction of her savings spent on the TV
def fraction_tv : ℝ := 1 / 6

-- The cost of the TV
def cost_tv : ℝ := linda_savings * fraction_tv

theorem tv_cost_correct : cost_tv = 500.0000000000001 := 
  by
  sorry

end tv_cost_correct_l457_457572


namespace carter_cheesecakes_l457_457384

theorem carter_cheesecakes (C : ℕ) (nm : ℕ) (nr : ℕ) (increase : ℕ) (this_week_cakes : ℕ) (usual_cakes : ℕ) :
  nm = 5 → nr = 8 → increase = 38 → 
  this_week_cakes = 3 * C + 3 * nm + 3 * nr → 
  usual_cakes = C + nm + nr → 
  this_week_cakes = usual_cakes + increase → 
  C = 6 :=
by
  intros hnm hnr hinc htw husual hcakes
  sorry

end carter_cheesecakes_l457_457384


namespace distance_D_D_l457_457667

-- Definitions for points on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Distance function between two points
def distance (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)

-- Points D and D' before and after reflection over x-axis
def D : Point := { x := 2, y := 4 }
def D' : Point := { x := 2, y := -4 }

-- The main theorem stating the distance between D and D' is 8
theorem distance_D_D'_eq_8 : distance D D' = 8 := by sorry

end distance_D_D_l457_457667


namespace lambda_mu_relationship_l457_457136

theorem lambda_mu_relationship (a b : ℝ × ℝ) (λ μ : ℝ)
  (h1 : a = (1, 1))
  (h2 : b = (1, -1))
  (h3 : (1 + λ) * (1 + μ) + (1 - λ) * (1 - μ) = 0) :
  λ * μ = -1 :=
sorry

end lambda_mu_relationship_l457_457136


namespace exists_c_same_digit_occurrences_l457_457817

theorem exists_c_same_digit_occurrences (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ c : ℕ, c > 0 ∧ ∀ d : ℕ, d ≠ 0 → 
    (Nat.digits 10 (c * m)).count d = (Nat.digits 10 (c * n)).count d := sorry

end exists_c_same_digit_occurrences_l457_457817
