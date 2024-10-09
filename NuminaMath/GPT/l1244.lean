import Mathlib

namespace intersection_A_B_l1244_124485

-- define the set A
def A : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 3 * x - y = 7 }

-- define the set B
def B : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x + y = 3 }

-- Prove the intersection
theorem intersection_A_B :
  A ∩ B = { (2, -1) } :=
by
  -- We will insert the proof here
  sorry

end intersection_A_B_l1244_124485


namespace max_expression_l1244_124467

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem max_expression (x y : ℝ) (h : x + y = 5) :
  max_value x y ≤ 6084 / 17 :=
sorry

end max_expression_l1244_124467


namespace maximum_value_expression_maximum_value_expression_achieved_l1244_124443

theorem maximum_value_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  (1 / (x^2 - 4 * x + 9) + 1 / (y^2 - 4 * y + 9) + 1 / (z^2 - 4 * z + 9)) ≤ 7 / 18 :=
sorry

theorem maximum_value_expression_achieved :
  (1 / (0^2 - 4 * 0 + 9) + 1 / (0^2 - 4 * 0 + 9) + 1 / (1^2 - 4 * 1 + 9)) = 7 / 18 :=
sorry

end maximum_value_expression_maximum_value_expression_achieved_l1244_124443


namespace lines_parallel_distinct_l1244_124474

theorem lines_parallel_distinct (a : ℝ) : 
  (∀ x y : ℝ, (2 * x - a * y + 1 = 0) → ((a - 1) * x - y + a = 0)) ↔ 
  a = 2 := 
sorry

end lines_parallel_distinct_l1244_124474


namespace find_b_l1244_124449

theorem find_b (b p : ℝ) (h_factor : ∃ k : ℝ, 3 * (x^3 : ℝ) + b * x + 9 = (x^2 + p * x + 3) * (k * x + k)) :
  b = -6 :=
by
  obtain ⟨k, h_eq⟩ := h_factor
  sorry

end find_b_l1244_124449


namespace sum_of_roots_l1244_124450

theorem sum_of_roots (x : ℝ) : (x - 4)^2 = 16 → x = 8 ∨ x = 0 := by
  intro h
  have h1 : x - 4 = 4 ∨ x - 4 = -4 := by
    sorry
  cases h1
  case inl h2 =>
    rw [h2] at h
    exact Or.inl (by linarith)
  case inr h2 =>
    rw [h2] at h
    exact Or.inr (by linarith)

end sum_of_roots_l1244_124450


namespace lillian_candies_total_l1244_124412

variable (initial_candies : ℕ)
variable (candies_given_by_father : ℕ)

theorem lillian_candies_total (initial_candies : ℕ) (candies_given_by_father : ℕ) :
  initial_candies = 88 →
  candies_given_by_father = 5 →
  initial_candies + candies_given_by_father = 93 :=
by
  intros
  sorry

end lillian_candies_total_l1244_124412


namespace find_positive_root_l1244_124445

open Real

theorem find_positive_root 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (x : ℝ) :
  sqrt (a * b * x * (a + b + x)) + sqrt (b * c * x * (b + c + x)) + sqrt (c * a * x * (c + a + x)) = sqrt (a * b * c * (a + b + c)) →
  x = (a * b * c) / (a * b + b * c + c * a + 2 * sqrt (a * b * c * (a + b + c))) := 
sorry

end find_positive_root_l1244_124445


namespace area_of_region_l1244_124483

theorem area_of_region : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 6*x - 8*y = 16) → 
  (π * 41) = (π * 41) :=
by
  sorry

end area_of_region_l1244_124483


namespace fraction_simplification_l1244_124400

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end fraction_simplification_l1244_124400


namespace Paige_recycled_pounds_l1244_124405

-- Definitions based on conditions from step a)
def points_per_pound := 1 / 4
def friends_pounds_recycled := 2
def total_points := 4

-- The proof statement (no proof required)
theorem Paige_recycled_pounds :
  let total_pounds_recycled := total_points * 4
  let paige_pounds_recycled := total_pounds_recycled - friends_pounds_recycled
  paige_pounds_recycled = 14 :=
by
  sorry

end Paige_recycled_pounds_l1244_124405


namespace find_number_l1244_124499

theorem find_number (x : ℝ) (h : 0.50 * x = 0.30 * 50 + 13) : x = 56 :=
by
  sorry

end find_number_l1244_124499


namespace jake_balloons_bought_l1244_124408

theorem jake_balloons_bought (B : ℕ) (h : 6 = (2 + B) + 1) : B = 3 :=
by
  -- proof omitted
  sorry

end jake_balloons_bought_l1244_124408


namespace find_a_b_l1244_124470

theorem find_a_b (a b x y : ℝ) (h₀ : a + b = 10) (h₁ : a / x + b / y = 1) (h₂ : x + y = 16) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
    (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1) :=
by
  sorry

end find_a_b_l1244_124470


namespace find_real_solutions_l1244_124404

theorem find_real_solutions (x : ℝ) : 
  x^4 + (3 - x)^4 = 130 ↔ x = 1.5 + Real.sqrt 1.5 ∨ x = 1.5 - Real.sqrt 1.5 :=
sorry

end find_real_solutions_l1244_124404


namespace tetrahedron_area_theorem_l1244_124432

noncomputable def tetrahedron_faces_areas_and_angles
  (a b c d : ℝ) (α β γ : ℝ) : Prop :=
  d^2 = a^2 + b^2 + c^2 - 2 * a * b * Real.cos γ - 2 * b * c * Real.cos α - 2 * c * a * Real.cos β

theorem tetrahedron_area_theorem
  (a b c d : ℝ) (α β γ : ℝ) :
  tetrahedron_faces_areas_and_angles a b c d α β γ :=
sorry

end tetrahedron_area_theorem_l1244_124432


namespace hypotenuse_length_l1244_124473

-- Definitions derived from conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Proposed theorem
theorem hypotenuse_length (a c : ℝ) 
  (h1 : is_isosceles_right_triangle a a c) 
  (h2 : perimeter a a c = 8 + 8 * Real.sqrt 2) :
  c = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l1244_124473


namespace carter_total_drum_sticks_l1244_124446

def sets_per_show_used := 5
def sets_per_show_tossed := 6
def nights := 30

theorem carter_total_drum_sticks : 
  (sets_per_show_used + sets_per_show_tossed) * nights = 330 := by
  sorry

end carter_total_drum_sticks_l1244_124446


namespace average_cars_given_per_year_l1244_124482

/-- Definition of initial conditions and the proposition -/
def initial_cars : ℕ := 3500
def final_cars : ℕ := 500
def years : ℕ := 60

theorem average_cars_given_per_year : (initial_cars - final_cars) / years = 50 :=
by
  sorry

end average_cars_given_per_year_l1244_124482


namespace geometric_series_sum_l1244_124486

theorem geometric_series_sum :
  2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))))))) = 2046 := 
by sorry

end geometric_series_sum_l1244_124486


namespace congruence_solution_exists_l1244_124416

theorem congruence_solution_exists {p n a : ℕ} (hp : Prime p) (hn : n % p ≠ 0) (ha : a % p ≠ 0)
  (hx : ∃ x : ℕ, x^n % p = a % p) :
  ∀ r : ℕ, ∃ x : ℕ, x^n % (p^(r + 1)) = a % (p^(r + 1)) :=
by
  intros r
  sorry

end congruence_solution_exists_l1244_124416


namespace girls_count_in_leos_class_l1244_124415

def leo_class_girls_count (g b : ℕ) :=
  (g / b = 3 / 4) ∧ (g + b = 35) → g = 15

theorem girls_count_in_leos_class (g b : ℕ) :
  leo_class_girls_count g b :=
by
  sorry

end girls_count_in_leos_class_l1244_124415


namespace upward_shift_of_parabola_l1244_124421

variable (k : ℝ) -- Define k as a real number representing the vertical shift

def original_function (x : ℝ) : ℝ := -x^2 -- Define the original function

def shifted_function (x : ℝ) : ℝ := original_function x + 2 -- Define the shifted function by 2 units upwards

theorem upward_shift_of_parabola (x : ℝ) : shifted_function x = -x^2 + k :=
by
  sorry

end upward_shift_of_parabola_l1244_124421


namespace smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l1244_124441

theorem smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2 :
  ∀ (x y z w : ℝ),
    x = Real.sqrt 3 →
    y = -1 / 3 →
    z = -2 →
    w = 0 →
    (x > 1) ∧ (y < 0) ∧ (z < 0) ∧ (|y| = 1 / 3) ∧ (|z| = 2) ∧ (w = 0) →
    min (min (min x y) z) w = z :=
by
  intros x y z w hx hy hz hw hcond
  sorry

end smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l1244_124441


namespace cos_value_l1244_124414

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by sorry

end cos_value_l1244_124414


namespace eval_expression_in_second_quadrant_l1244_124452

theorem eval_expression_in_second_quadrant (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : Real.sin α > 0) (h3 : Real.cos α < 0) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / (Real.sin α) ^ 2 - 1) = -1 :=
by
  sorry

end eval_expression_in_second_quadrant_l1244_124452


namespace incorrect_statements_l1244_124440

-- Definitions based on conditions from the problem.

def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c < 0
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_inequality a b c x}
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Lean statements of the conditions and the final proof problem.
theorem incorrect_statements (a b c : ℝ) (M : Set ℝ) :
  (M = ∅ → (a < 0 ∧ discriminant a b c < 0) → false) ∧
  (M = {x | x ≠ x0} → a < b → (a + 4 * c) / (b - a) = 2 + 2 * Real.sqrt 2 → false) := sorry

end incorrect_statements_l1244_124440


namespace inequality_geq_l1244_124424

theorem inequality_geq (t : ℝ) (n : ℕ) (ht : t ≥ 1/2) : 
  t^(2*n) ≥ (t-1)^(2*n) + (2*t-1)^n := 
sorry

end inequality_geq_l1244_124424


namespace jackson_email_problem_l1244_124454

variables (E_0 E_1 E_2 E_3 X : ℕ)

/-- Jackson's email deletion and receipt problem -/
theorem jackson_email_problem
  (h1 : E_1 = E_0 - 50 + 15)
  (h2 : E_2 = E_1 - X + 5)
  (h3 : E_3 = E_2 + 10)
  (h4 : E_3 = 30) :
  X = 50 :=
sorry

end jackson_email_problem_l1244_124454


namespace minimum_k_value_l1244_124496

theorem minimum_k_value (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∀ a b, (1 / a + 1 / b + k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_value_l1244_124496


namespace problem_1_problem_2_l1244_124497

open BigOperators

-- Question 1
theorem problem_1 (a : Fin 2021 → ℝ) :
  (1 + 2 * x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  (∑ i in Finset.range 2021, (i * a i)) = 4040 * 3 ^ 2019 :=
sorry

-- Question 2
theorem problem_2 (a : Fin 2021 → ℝ) :
  (1 - x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  ((∑ i in Finset.range 2021, 1 / a i)) = 2021 / 1011 :=
sorry

end problem_1_problem_2_l1244_124497


namespace combined_weight_correct_l1244_124406

-- Define Jake's present weight
def Jake_weight : ℕ := 196

-- Define the weight loss
def weight_loss : ℕ := 8

-- Define Jake's weight after losing weight
def Jake_weight_after_loss : ℕ := Jake_weight - weight_loss

-- Define the relationship between Jake's weight after loss and his sister's weight
def sister_weight : ℕ := Jake_weight_after_loss / 2

-- Define the combined weight
def combined_weight : ℕ := Jake_weight + sister_weight

-- Prove that the combined weight is 290 pounds
theorem combined_weight_correct : combined_weight = 290 :=
by
  sorry

end combined_weight_correct_l1244_124406


namespace complete_the_square_l1244_124420

theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 = 6 := 
by {
  -- This is where you would provide the proof
  sorry
}

end complete_the_square_l1244_124420


namespace sufficient_necessary_condition_l1244_124428

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * a * x^3 + (1 / 2) * a * x^2 - 2 * a * x + 2 * a + 1

theorem sufficient_necessary_condition (a : ℝ) :
  (-6 / 5 < a ∧ a < -3 / 16) ↔
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧
   (∃ c₁ c₂ : ℝ, deriv (f a) c₁ = 0 ∧ deriv (f a) c₂ = 0 ∧
   deriv (deriv (f a)) c₁ < 0 ∧ deriv (deriv (f a)) c₂ > 0 ∧
   f a c₁ > 0 ∧ f a c₂ < 0)) := sorry

end sufficient_necessary_condition_l1244_124428


namespace find_x_l1244_124439

theorem find_x
  (x : ℝ)
  (h : 5^29 * x^15 = 2 * 10^29) :
  x = 4 :=
by
  sorry

end find_x_l1244_124439


namespace body_diagonal_length_l1244_124456

theorem body_diagonal_length (a b c : ℝ) (h1 : a * b = 6) (h2 : a * c = 8) (h3 : b * c = 12) :
  (a^2 + b^2 + c^2 = 29) :=
by
  sorry

end body_diagonal_length_l1244_124456


namespace percent_republicans_voting_for_A_l1244_124460

theorem percent_republicans_voting_for_A (V : ℝ) (percent_Democrats : ℝ) 
  (percent_Republicans : ℝ) (percent_D_voting_for_A : ℝ) 
  (percent_total_voting_for_A : ℝ) (R : ℝ) 
  (h1 : percent_Democrats = 0.60)
  (h2 : percent_Republicans = 0.40)
  (h3 : percent_D_voting_for_A = 0.85)
  (h4 : percent_total_voting_for_A = 0.59) :
  R = 0.2 :=
by 
  sorry

end percent_republicans_voting_for_A_l1244_124460


namespace total_house_rent_l1244_124434

theorem total_house_rent (P S R : ℕ)
  (h1 : S = 5 * P)
  (h2 : R = 3 * P)
  (h3 : R = 1800) : 
  S + P + R = 5400 :=
by
  sorry

end total_house_rent_l1244_124434


namespace at_least_half_team_B_can_serve_on_submarine_l1244_124459

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l1244_124459


namespace P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l1244_124498

-- Assume the definition of sum of digits of n and count of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum  -- Sum of digits in base 10 representation

def num_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).length  -- Number of digits in base 10 representation

def P (n : ℕ) : ℕ :=
  sum_of_digits n + num_of_digits n

-- Problem (a)
theorem P_2017 : P 2017 = 14 :=
sorry

-- Problem (b)
theorem P_eq_4 :
  {n : ℕ | P n = 4} = {3, 11, 20, 100} :=
sorry

-- Problem (c)
theorem exists_P_minus_P_succ_gt_50 : 
  ∃ n : ℕ, P n - P (n + 1) > 50 :=
sorry

end P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l1244_124498


namespace shrink_ray_coffee_l1244_124484

theorem shrink_ray_coffee (num_cups : ℕ) (ounces_per_cup : ℕ) (shrink_factor : ℝ) 
  (h1 : num_cups = 5) 
  (h2 : ounces_per_cup = 8) 
  (h3 : shrink_factor = 0.5) 
  : num_cups * ounces_per_cup * shrink_factor = 20 :=
by
  rw [h1, h2, h3]
  simp
  norm_num

end shrink_ray_coffee_l1244_124484


namespace taxi_speed_l1244_124495

theorem taxi_speed (v : ℕ) (h₁ : v > 30) (h₂ : ∃ t₁ t₂ : ℕ, t₁ = 3 ∧ t₂ = 3 ∧ 
                    v * t₁ = (v - 30) * (t₁ + t₂)) : 
                    v = 60 :=
by
  sorry

end taxi_speed_l1244_124495


namespace complex_fraction_value_l1244_124488

theorem complex_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 7 / 4 :=
sorry

end complex_fraction_value_l1244_124488


namespace algebra_square_formula_l1244_124472

theorem algebra_square_formula (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
sorry

end algebra_square_formula_l1244_124472


namespace winning_candidate_percentage_votes_l1244_124435

theorem winning_candidate_percentage_votes
  (total_votes : ℕ) (majority_votes : ℕ) (P : ℕ) 
  (h1 : total_votes = 6500) 
  (h2 : majority_votes = 1300) 
  (h3 : (P * total_votes) / 100 - ((100 - P) * total_votes) / 100 = majority_votes) : 
  P = 60 :=
sorry

end winning_candidate_percentage_votes_l1244_124435


namespace sin_6phi_l1244_124417

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * (Real.sqrt 8)) / 5) : 
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 :=
by
  sorry

end sin_6phi_l1244_124417


namespace stationery_store_sales_l1244_124477

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end stationery_store_sales_l1244_124477


namespace reciprocal_of_neg_three_l1244_124476

theorem reciprocal_of_neg_three : (1:ℝ) / (-3:ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg_three_l1244_124476


namespace min_value_inv_sum_l1244_124418

open Real

theorem min_value_inv_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  3 ≤ (1 / x) + (1 / y) + (1 / z) :=
sorry

end min_value_inv_sum_l1244_124418


namespace part_a_part_b_l1244_124433

-- Part a: Prove for specific numbers 2015 and 2017
theorem part_a : ∃ (x y : ℕ), (2015^2 + 2017^2) / 2 = x^2 + y^2 := sorry

-- Part b: Prove for any two different odd natural numbers
theorem part_b (a b : ℕ) (h1 : a ≠ b) (h2 : a % 2 = 1) (h3 : b % 2 = 1) :
  ∃ (x y : ℕ), (a^2 + b^2) / 2 = x^2 + y^2 := sorry

end part_a_part_b_l1244_124433


namespace positive_abc_l1244_124423

theorem positive_abc (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 := 
by
  sorry

end positive_abc_l1244_124423


namespace watch_cost_l1244_124493

variables (w s : ℝ)

theorem watch_cost (h1 : w + s = 120) (h2 : w = 100 + s) : w = 110 :=
by
  sorry

end watch_cost_l1244_124493


namespace has_two_distinct_roots_and_ordered_l1244_124413

-- Define the context and the conditions of the problem.
variables (a b c : ℝ) (h : a < b) (h2 : b < c)

-- Define the quadratic function derived from the problem.
def quadratic (x : ℝ) : ℝ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

-- State the main theorem.
theorem has_two_distinct_roots_and_ordered:
  ∃ x1 x2 : ℝ, quadratic a b c x1 = 0 ∧ quadratic a b c x2 = 0 ∧ a < x1 ∧ x1 < b ∧ b < x2 ∧ x2 < c :=
sorry

end has_two_distinct_roots_and_ordered_l1244_124413


namespace find_k_l1244_124491

theorem find_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 5)) (hk : k ≠ 0) : k = 5 :=
sorry

end find_k_l1244_124491


namespace plant_arrangement_count_l1244_124453

-- Define the count of identical plants
def basil_count := 3
def aloe_count := 2

-- Define the count of identical lamps in each color
def white_lamp_count := 3
def red_lamp_count := 3

-- Define the total ways to arrange the plants under the lamps.
def arrangement_ways := 128

-- Formalize the problem statement proving the arrangements count
theorem plant_arrangement_count :
  (∃ f : Fin (basil_count + aloe_count) → Fin (white_lamp_count + red_lamp_count), True) ↔
  arrangement_ways = 128 :=
sorry

end plant_arrangement_count_l1244_124453


namespace unique_primes_solution_l1244_124419

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_primes_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  p^3 - q^5 = (p + q)^2 ↔ (p = 7 ∧ q = 3) :=
by
  sorry

end unique_primes_solution_l1244_124419


namespace find_N_l1244_124489

theorem find_N (x y N : ℝ) (h1 : 2 * x + y = N) (h2 : x + 2 * y = 5) (h3 : (x + y) / 3 = 1) : N = 4 :=
by
  have h4 : x + y = 3 := by
    linarith [h3]
  have h5 : y = 3 - x := by
    linarith [h4]
  have h6 : x + 2 * (3 - x) = 5 := by
    linarith [h2, h5]
  have h7 : x = 1 := by
    linarith [h6]
  have h8 : y = 2 := by
    linarith [h4, h7]
  have h9 : 2 * x + y = 4 := by
    linarith [h7, h8]
  linarith [h1, h9]

end find_N_l1244_124489


namespace find_theta_interval_l1244_124487

theorem find_theta_interval (θ : ℝ) (x : ℝ) :
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (0 ≤ x ∧ x ≤ 1) →
  (∀ k, k = 0.5 → x^2 * Real.sin θ - k * x * (1 - x) + (1 - x)^2 * Real.cos θ ≥ 0) ↔
  (0 ≤ θ ∧ θ ≤ π / 12) ∨ (23 * π / 12 ≤ θ ∧ θ ≤ 2 * π) := 
sorry

end find_theta_interval_l1244_124487


namespace ferry_speed_difference_l1244_124463

variable (v_P v_Q d_P d_Q t_P t_Q x : ℝ)

-- Defining the constants and conditions provided in the problem
axiom h1 : v_P = 8 
axiom h2 : t_P = 2 
axiom h3 : d_P = t_P * v_P 
axiom h4 : d_Q = 3 * d_P 
axiom h5 : t_Q = t_P + 2
axiom h6 : d_Q = v_Q * t_Q 
axiom h7 : x = v_Q - v_P 

-- The theorem that corresponds to the solution
theorem ferry_speed_difference : x = 4 := by
  sorry

end ferry_speed_difference_l1244_124463


namespace physics_marks_l1244_124471

variables (P C M : ℕ)

theorem physics_marks (h1 : P + C + M = 195)
                      (h2 : P + M = 180)
                      (h3 : P + C = 140) : P = 125 :=
by
  sorry

end physics_marks_l1244_124471


namespace quadratic_coefficients_l1244_124464

theorem quadratic_coefficients (x1 x2 p q : ℝ)
  (h1 : x1 - x2 = 5)
  (h2 : x1 ^ 3 - x2 ^ 3 = 35) :
  (x1 + x2 = -p ∧ x1 * x2 = q ∧ (p = 1 ∧ q = -6) ∨ 
   x1 + x2 = p ∧ x1 * x2 = q ∧ (p = -1 ∧ q = -6)) :=
by
  sorry

end quadratic_coefficients_l1244_124464


namespace complete_the_square_l1244_124422

theorem complete_the_square (x : ℝ) : x^2 + 6 * x + 3 = 0 ↔ (x + 3)^2 = 6 := 
by
  sorry

end complete_the_square_l1244_124422


namespace range_of_t_l1244_124403

theorem range_of_t (a b c t: ℝ) 
  (h1 : 6 * a = 2 * b - 6)
  (h2 : 6 * a = 3 * c)
  (h3 : b ≥ 0)
  (h4 : c ≤ 2)
  (h5 : t = 2 * a + b - c) : 
  0 ≤ t ∧ t ≤ 6 :=
sorry

end range_of_t_l1244_124403


namespace find_point_symmetric_about_y_axis_l1244_124429

def point := ℤ × ℤ

def symmetric_about_y_axis (A B : point) : Prop :=
  B.1 = -A.1 ∧ B.2 = A.2

theorem find_point_symmetric_about_y_axis (A B : point) 
  (hA : A = (-5, 2)) 
  (hSym : symmetric_about_y_axis A B) : 
  B = (5, 2) := 
by
  -- We declare the proof but omit the steps for this exercise.
  sorry

end find_point_symmetric_about_y_axis_l1244_124429


namespace six_digit_palindromes_count_l1244_124490

theorem six_digit_palindromes_count : 
  ∃ n : ℕ, n = 27 ∧ 
  (∀ (A B C : ℕ), 
       (A = 6 ∨ A = 7 ∨ A = 8) ∧ 
       (B = 6 ∨ B = 7 ∨ B = 8) ∧ 
       (C = 6 ∨ C = 7 ∨ C = 8) → 
       ∃ p : ℕ, 
         p = (A * 10^5 + B * 10^4 + C * 10^3 + C * 10^2 + B * 10 + A) ∧ 
         (6 ≤ p / 10^5 ∧ p / 10^5 ≤ 8) ∧ 
         (6 ≤ (p / 10^4) % 10 ∧ (p / 10^4) % 10 ≤ 8) ∧ 
         (6 ≤ (p / 10^3) % 10 ∧ (p / 10^3) % 10 ≤ 8)) :=
  by sorry

end six_digit_palindromes_count_l1244_124490


namespace recurring_fraction_division_l1244_124469

noncomputable def recurring_833 := 5 / 6
noncomputable def recurring_1666 := 5 / 3

theorem recurring_fraction_division : 
  (recurring_833 / recurring_1666) = 1 / 2 := 
by 
  sorry

end recurring_fraction_division_l1244_124469


namespace am_gm_inequality_l1244_124455

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c) ^ 2 :=
by
  sorry

end am_gm_inequality_l1244_124455


namespace least_integer_x_l1244_124462

theorem least_integer_x (x : ℤ) (h : 3 * |x| - 2 * x + 8 < 23) : x = -3 :=
sorry

end least_integer_x_l1244_124462


namespace correct_exp_operation_l1244_124492

theorem correct_exp_operation (a : ℝ) : (a^2 * a = a^3) := 
by
  -- Leave the proof as an exercise
  sorry

end correct_exp_operation_l1244_124492


namespace volume_of_defined_region_l1244_124410

noncomputable def volume_of_region (x y z : ℝ) : ℝ :=
if x + y ≤ 5 ∧ z ≤ 5 ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x ≤ 2 then 15 else 0

theorem volume_of_defined_region :
  ∀ (x y z : ℝ),
  (0 ≤ x) → (0 ≤ y) → (0 ≤ z) → (x ≤ 2) →
  (|x + y + z| + |x + y - z| ≤ 10) →
  volume_of_region x y z = 15 :=
sorry

end volume_of_defined_region_l1244_124410


namespace sum_of_squares_l1244_124461

theorem sum_of_squares (x : ℚ) (h : x + 2 * x + 3 * x = 14) : 
  (x^2 + (2 * x)^2 + (3 * x)^2) = 686 / 9 :=
by
  sorry

end sum_of_squares_l1244_124461


namespace probability_of_two_non_defective_pens_l1244_124457

-- Definitions for conditions from the problem
def total_pens : ℕ := 16
def defective_pens : ℕ := 3
def selected_pens : ℕ := 2
def non_defective_pens : ℕ := total_pens - defective_pens

-- Function to calculate probability of drawing non-defective pens
noncomputable def probability_no_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  (non_defective_pens / total_pens) * ((non_defective_pens - 1) / (total_pens - 1))

-- Theorem stating the correct answer
theorem probability_of_two_non_defective_pens : 
  probability_no_defective total_pens defective_pens selected_pens = 13 / 20 :=
by
  sorry

end probability_of_two_non_defective_pens_l1244_124457


namespace dentist_age_is_32_l1244_124481

-- Define the conditions
def one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence (x : ℕ) : Prop :=
  (x - 8) / 6 = (x + 8) / 10

-- State the theorem
theorem dentist_age_is_32 : ∃ x : ℕ, one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence x ∧ x = 32 :=
by
  sorry

end dentist_age_is_32_l1244_124481


namespace find_y_coordinate_l1244_124451

theorem find_y_coordinate (x2 : ℝ) (y1 : ℝ) :
  (∃ m : ℝ, m = (y1 - 0) / (10 - 4) ∧ (-8 - y1) = m * (x2 - 10)) →
  y1 = -8 :=
by
  sorry

end find_y_coordinate_l1244_124451


namespace perimeter_paper_count_l1244_124442

theorem perimeter_paper_count (n : Nat) (h : n = 10) : 
  let top_side := n
  let right_side := n - 1
  let bottom_side := n - 1
  let left_side := n - 2
  top_side + right_side + bottom_side + left_side = 36 :=
by
  sorry

end perimeter_paper_count_l1244_124442


namespace product_of_repeating_decimal_and_22_l1244_124402

noncomputable def repeating_decimal_to_fraction : ℚ :=
  0.45 + 0.0045 * (10 ^ (-2 : ℤ))

theorem product_of_repeating_decimal_and_22 : (repeating_decimal_to_fraction * 22 = 10) :=
by
  sorry

end product_of_repeating_decimal_and_22_l1244_124402


namespace second_quadrant_implies_value_of_m_l1244_124475

theorem second_quadrant_implies_value_of_m (m : ℝ) : 4 - m < 0 → m = 5 := by
  intro h
  have ineq : m > 4 := by
    linarith
  sorry

end second_quadrant_implies_value_of_m_l1244_124475


namespace cyclists_meet_after_24_minutes_l1244_124479

noncomputable def meet_time (D : ℝ) (vm vb : ℝ) : ℝ :=
  D / (2.5 * D - 12)

theorem cyclists_meet_after_24_minutes
  (D vm vb : ℝ)
  (h_vm : 1/3 * vm + 2 = D/2)
  (h_vb : 1/2 * vb = D/2 - 3) :
  meet_time D vm vb = 24 :=
by
  sorry

end cyclists_meet_after_24_minutes_l1244_124479


namespace product_of_roots_l1244_124411

variable {x1 x2 : ℝ}

theorem product_of_roots (hx1 : x1 * Real.log x1 = 2006) (hx2 : x2 * Real.exp x2 = 2006) : x1 * x2 = 2006 :=
sorry

end product_of_roots_l1244_124411


namespace combined_cost_is_107_l1244_124401

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end combined_cost_is_107_l1244_124401


namespace roots_sum_and_product_l1244_124458

theorem roots_sum_and_product (p q : ℝ) (h_sum : p / 3 = 9) (h_prod : q / 3 = 24) : p + q = 99 :=
by
  -- We are given h_sum: p / 3 = 9
  -- We are given h_prod: q / 3 = 24
  -- We need to prove p + q = 99
  sorry

end roots_sum_and_product_l1244_124458


namespace count_negative_numbers_l1244_124494

theorem count_negative_numbers : 
  let n1 := abs (-2)
  let n2 := - abs (3^2)
  let n3 := - (3^2)
  let n4 := (-2)^(2023)
  (if n1 < 0 then 1 else 0) + (if n2 < 0 then 1 else 0) + (if n3 < 0 then 1 else 0) + (if n4 < 0 then 1 else 0) = 3 := 
by
  sorry

end count_negative_numbers_l1244_124494


namespace range_of_a_l1244_124465

open Set

variable (a : ℝ)

def P(a : ℝ) : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0

def Q(a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l1244_124465


namespace female_muscovy_ducks_l1244_124430

theorem female_muscovy_ducks :
  let total_ducks := 40
  let muscovy_percentage := 0.5
  let female_muscovy_percentage := 0.3
  let muscovy_ducks := total_ducks * muscovy_percentage
  let female_muscovy_ducks := muscovy_ducks * female_muscovy_percentage
  female_muscovy_ducks = 6 :=
by
  sorry

end female_muscovy_ducks_l1244_124430


namespace average_headcount_is_correct_l1244_124431

/-- The student headcount data for the specified semesters -/
def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

noncomputable def average_headcount : ℕ :=
  (student_headcount.sum) / student_headcount.length

theorem average_headcount_is_correct : average_headcount = 11029 := by
  sorry

end average_headcount_is_correct_l1244_124431


namespace sum_of_ages_l1244_124466

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
def condition1 := a = 16 + b + c
def condition2 := a^2 = 1632 + (b + c)^2

-- Define the theorem to prove the question
theorem sum_of_ages : condition1 a b c → condition2 a b c → a + b + c = 102 := 
by 
  intros h1 h2
  sorry

end sum_of_ages_l1244_124466


namespace length_of_GH_l1244_124437

def EF := 180
def IJ := 120

theorem length_of_GH (EF_parallel_GH : true) (GH_parallel_IJ : true) : GH = 72 := 
sorry

end length_of_GH_l1244_124437


namespace watch_correction_needed_l1244_124438

def watch_loses_rate : ℚ := 15 / 4  -- rate of loss per day in minutes
def initial_set_time : ℕ := 15  -- March 15th at 10 A.M.
def report_time : ℕ := 24  -- March 24th at 4 P.M.
def correction (loss_rate per_day min_hrs : ℚ) (days_hrs : ℚ) : ℚ :=
  (days_hrs * (loss_rate / (per_day * min_hrs)))

theorem watch_correction_needed :
  correction watch_loses_rate 24 60 (222) = 34.6875 := 
sorry

end watch_correction_needed_l1244_124438


namespace minimize_wood_frame_l1244_124444

noncomputable def min_wood_frame (x y : ℝ) : Prop :=
  let area_eq : Prop := x * y + x^2 / 4 = 8
  let length := 2 * (x + y) + Real.sqrt 2 * x
  let y_expr := 8 / x - x / 4
  let length_expr := (3 / 2 + Real.sqrt 2) * x + 16 / x
  let min_x := Real.sqrt (16 / (3 / 2 + Real.sqrt 2))
  area_eq ∧ y = y_expr ∧ length = length_expr ∧ x = 2.343 ∧ y = 2.828

theorem minimize_wood_frame : ∃ x y : ℝ, min_wood_frame x y :=
by
  use 2.343
  use 2.828
  unfold min_wood_frame
  -- we leave the proof of the properties as sorry
  sorry

end minimize_wood_frame_l1244_124444


namespace charlie_widgets_difference_l1244_124480

theorem charlie_widgets_difference (w t : ℕ) (hw : w = 3 * t) :
  w * t - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end charlie_widgets_difference_l1244_124480


namespace count_int_values_not_satisfying_ineq_l1244_124468

theorem count_int_values_not_satisfying_ineq :
  ∃ (s : Finset ℤ), (∀ x ∈ s, 3 * x^2 + 14 * x + 8 ≤ 17) ∧ (s.card = 10) :=
by
  sorry

end count_int_values_not_satisfying_ineq_l1244_124468


namespace dino_dolls_count_l1244_124447

theorem dino_dolls_count (T : ℝ) (H : 0.7 * T = 140) : T = 200 :=
sorry

end dino_dolls_count_l1244_124447


namespace science_book_pages_l1244_124427

def history_pages := 300
def novel_pages := history_pages / 2
def science_pages := novel_pages * 4

theorem science_book_pages : science_pages = 600 := by
  -- Given conditions:
  -- The novel has half as many pages as the history book, the history book has 300 pages,
  -- and the science book has 4 times as many pages as the novel.
  sorry

end science_book_pages_l1244_124427


namespace count_4_digit_divisible_by_45_l1244_124425

theorem count_4_digit_divisible_by_45 : 
  ∃ n, n = 11 ∧ (∀ a b : ℕ, a + b = 2 ∨ a + b = 11 → (20 + b * 10 + 5) % 45 = 0) :=
sorry

end count_4_digit_divisible_by_45_l1244_124425


namespace simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l1244_124478

theorem simplify_expression (a : ℤ) (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 :=
by
  sorry

theorem evaluate_expression_at_neg1 (h : (-1 : ℤ) ≠ 0) (h' : (-1 : ℤ) ≠ 1) : 
  (-1 - (2 * (-1) - 1) / (-1)) / ((-1 - 1) / (-1)) = -2 :=
by
  sorry

theorem evaluate_expression_at_2 (h : (2 : ℤ) ≠ 0) (h' : (2 : ℤ) ≠ 1) : 
  (2 - (2 * 2 - 1) / 2) / ((2 - 1) / 2) = 1 :=
by
  sorry

end simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l1244_124478


namespace cube_inequality_l1244_124407

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
by
  sorry

end cube_inequality_l1244_124407


namespace angle_between_bisectors_l1244_124448

theorem angle_between_bisectors (β γ : ℝ) (h_sum : β + γ = 130) : (β / 2) + (γ / 2) = 65 :=
by
  have h : β + γ = 130 := h_sum
  sorry

end angle_between_bisectors_l1244_124448


namespace range_of_m_l1244_124436

theorem range_of_m (m : ℝ) (x0 : ℝ)
  (h : (4^(-x0) - m * 2^(-x0 + 1)) = -(4^x0 - m * 2^(x0 + 1))) :
  m ≥ 1/2 :=
sorry

end range_of_m_l1244_124436


namespace max_min_values_l1244_124409

theorem max_min_values (x y : ℝ) 
  (h : (x - 3)^2 + 4 * (y - 1)^2 = 4) :
  ∃ (t u : ℝ), (∀ (z : ℝ), (x-3)^2 + 4*(y-1)^2 = 4 → t ≤ (x+y-3)/(x-y+1) ∧ (x+y-3)/(x-y+1) ≤ u) ∧ t = -1 ∧ u = 1 := 
by
  sorry

end max_min_values_l1244_124409


namespace number_of_mismatching_socks_l1244_124426

-- Define the conditions
def total_socks : Nat := 25
def pairs_of_matching_socks : Nat := 4
def socks_per_pair : Nat := 2
def matching_socks : Nat := pairs_of_matching_socks * socks_per_pair

-- State the theorem
theorem number_of_mismatching_socks : total_socks - matching_socks = 17 :=
by
  -- Skip the proof
  sorry

end number_of_mismatching_socks_l1244_124426
