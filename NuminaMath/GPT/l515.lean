import Mathlib

namespace NUMINAMATH_GPT_percentage_discount_proof_l515_51500

noncomputable def ticket_price : ℝ := 25
noncomputable def price_to_pay : ℝ := 18.75
noncomputable def discount_amount : ℝ := ticket_price - price_to_pay
noncomputable def percentage_discount : ℝ := (discount_amount / ticket_price) * 100

theorem percentage_discount_proof : percentage_discount = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_discount_proof_l515_51500


namespace NUMINAMATH_GPT_problem1_problem2_l515_51589

-- Problem 1 Proof Statement
theorem problem1 : Real.sin (30 * Real.pi / 180) + abs (-1) - (Real.sqrt 3 - Real.pi) ^ 0 = 1 / 2 := 
  by sorry

-- Problem 2 Proof Statement
theorem problem2 (x: ℝ) (hx : x ≠ 2) : (2 * x - 3) / (x - 2) - (x - 1) / (x - 2) = 1 := 
  by sorry

end NUMINAMATH_GPT_problem1_problem2_l515_51589


namespace NUMINAMATH_GPT_find_m_when_power_function_decreasing_l515_51542

theorem find_m_when_power_function_decreasing :
  ∃ m : ℝ, (m^2 - 2 * m - 2 = 1) ∧ (-4 * m - 2 < 0) ∧ (m = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_m_when_power_function_decreasing_l515_51542


namespace NUMINAMATH_GPT_total_spokes_is_60_l515_51587

def num_spokes_front : ℕ := 20
def num_spokes_back : ℕ := 2 * num_spokes_front
def total_spokes : ℕ := num_spokes_front + num_spokes_back

theorem total_spokes_is_60 : total_spokes = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_spokes_is_60_l515_51587


namespace NUMINAMATH_GPT_susannah_swims_more_than_camden_l515_51597

-- Define the given conditions
def camden_total_swims : ℕ := 16
def susannah_total_swims : ℕ := 24
def number_of_weeks : ℕ := 4

-- State the theorem
theorem susannah_swims_more_than_camden :
  (susannah_total_swims / number_of_weeks) - (camden_total_swims / number_of_weeks) = 2 :=
by
  sorry

end NUMINAMATH_GPT_susannah_swims_more_than_camden_l515_51597


namespace NUMINAMATH_GPT_x_is_half_l515_51560

theorem x_is_half (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : x = 0.5 :=
sorry

end NUMINAMATH_GPT_x_is_half_l515_51560


namespace NUMINAMATH_GPT_smallest_N_for_percentages_l515_51528

theorem smallest_N_for_percentages 
  (N : ℕ) 
  (h1 : ∃ N, ∀ f ∈ [1/10, 2/5, 1/5, 3/10], ∃ k : ℕ, N * f = k) :
  N = 10 := 
by
  sorry

end NUMINAMATH_GPT_smallest_N_for_percentages_l515_51528


namespace NUMINAMATH_GPT_chelsea_guaranteed_victory_l515_51555

noncomputable def minimum_bullseye_shots_to_win (k : ℕ) (n : ℕ) : ℕ :=
  if (k + 5 * n + 500 > k + 930) then n else sorry

theorem chelsea_guaranteed_victory (k : ℕ) :
  minimum_bullseye_shots_to_win k 87 = 87 :=
by
  sorry

end NUMINAMATH_GPT_chelsea_guaranteed_victory_l515_51555


namespace NUMINAMATH_GPT_min_value_of_M_l515_51566

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

noncomputable def M : ℝ :=
  (Real.rpow (a / (b + c)) (1 / 4)) + (Real.rpow (b / (c + a)) (1 / 4)) + (Real.rpow (c / (b + a)) (1 / 4)) +
  Real.sqrt ((b + c) / a) + Real.sqrt ((a + c) / b) + Real.sqrt ((a + b) / c)

theorem min_value_of_M : M a b c = 3 * Real.sqrt 2 + (3 * Real.rpow 8 (1 / 4)) / 2 := sorry

end NUMINAMATH_GPT_min_value_of_M_l515_51566


namespace NUMINAMATH_GPT_meat_pie_cost_l515_51580

variable (total_farthings : ℕ) (farthings_per_pfennig : ℕ) (remaining_pfennigs : ℕ)

def total_pfennigs (total_farthings farthings_per_pfennig : ℕ) : ℕ :=
  total_farthings / farthings_per_pfennig

def pie_cost (total_farthings farthings_per_pfennig remaining_pfennigs : ℕ) : ℕ :=
  total_pfennigs total_farthings farthings_per_pfennig - remaining_pfennigs

theorem meat_pie_cost
  (h1 : total_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7) :
  pie_cost total_farthings farthings_per_pfennig remaining_pfennigs = 2 :=
by
  sorry

end NUMINAMATH_GPT_meat_pie_cost_l515_51580


namespace NUMINAMATH_GPT_min_sum_ab_l515_51516

theorem min_sum_ab {a b : ℤ} (h : a * b = 36) : a + b ≥ -37 := sorry

end NUMINAMATH_GPT_min_sum_ab_l515_51516


namespace NUMINAMATH_GPT_solution_for_x_l515_51567

theorem solution_for_x (x : ℝ) : x^2 - x - 1 = (x + 1)^0 → x = 2 :=
by
  intro h
  have h_simp : x^2 - x - 1 = 1 := by simp [h]
  sorry

end NUMINAMATH_GPT_solution_for_x_l515_51567


namespace NUMINAMATH_GPT_positive_y_equals_32_l515_51593

theorem positive_y_equals_32 (y : ℝ) (h : y^2 = 1024) (hy : 0 < y) : y = 32 :=
sorry

end NUMINAMATH_GPT_positive_y_equals_32_l515_51593


namespace NUMINAMATH_GPT_find_number_l515_51531

theorem find_number (x : ℤ) (h : 5 * (x - 12) = 40) : x = 20 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l515_51531


namespace NUMINAMATH_GPT_area_of_region_l515_51539

-- Definitions from the problem's conditions.
def equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 10*y = -9

-- Statement of the theorem.
theorem area_of_region : 
  ∃ (area : ℝ), (∀ x y : ℝ, equation x y → True) ∧ area = 32 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_l515_51539


namespace NUMINAMATH_GPT_not_right_triangle_l515_51561

theorem not_right_triangle (a b c : ℝ) (h : a / b = 1 / 2 ∧ b / c = 2 / 3) :
  ¬(a^2 = b^2 + c^2) :=
by sorry

end NUMINAMATH_GPT_not_right_triangle_l515_51561


namespace NUMINAMATH_GPT_find_x_l515_51591

variable (x : ℕ)  -- we'll use natural numbers to avoid negative values

-- initial number of children
def initial_children : ℕ := 21

-- number of children who got off
def got_off : ℕ := 10

-- total children after some got on
def total_children : ℕ := 16

-- statement to prove x is the number of children who got on the bus
theorem find_x : initial_children - got_off + x = total_children → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l515_51591


namespace NUMINAMATH_GPT_decreasing_power_function_l515_51599

theorem decreasing_power_function (n : ℝ) (f : ℝ → ℝ) 
    (h : ∀ x > 0, f x = (n^2 - n - 1) * x^n) 
    (h_decreasing : ∀ x > 0, f x > f (x + 1)) : n = -1 :=
sorry

end NUMINAMATH_GPT_decreasing_power_function_l515_51599


namespace NUMINAMATH_GPT_complex_transformation_l515_51569

open Complex

theorem complex_transformation :
  let z := -1 + (7 : ℂ) * I
  let rotation := (1 / 2 + (Real.sqrt 3) / 2 * I)
  let dilation := 2
  (z * rotation * dilation = -22 - ((Real.sqrt 3) - 7) * I) :=
by
  sorry

end NUMINAMATH_GPT_complex_transformation_l515_51569


namespace NUMINAMATH_GPT_commentator_mistake_l515_51571

def round_robin_tournament : Prop :=
  ∀ (x y : ℝ),
    x + 2 * x + 13 * y = 105 ∧ x < y ∧ y < 2 * x → False

theorem commentator_mistake : round_robin_tournament :=
  by {
    sorry
  }

end NUMINAMATH_GPT_commentator_mistake_l515_51571


namespace NUMINAMATH_GPT_smallest_difference_l515_51544

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_difference_l515_51544


namespace NUMINAMATH_GPT_find_x_l515_51558

/-
If two minus the reciprocal of (3 - x) equals the reciprocal of (2 + x), 
then x equals (1 + sqrt(15)) / 2 or (1 - sqrt(15)) / 2.
-/
theorem find_x (x : ℝ) :
  (2 - (1 / (3 - x)) = (1 / (2 + x))) → 
  (x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2) :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l515_51558


namespace NUMINAMATH_GPT_min_AB_dot_CD_l515_51563

theorem min_AB_dot_CD (a b : ℝ) (h1 : 0 <= (a - 1)^2 + (b - 3 / 2)^2 - 13/4) :
  ∃ (a b : ℝ), (a-1)^2 + (b - 3 / 2)^2 - 13/4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_min_AB_dot_CD_l515_51563


namespace NUMINAMATH_GPT_throwing_skips_l515_51526

theorem throwing_skips :
  ∃ x y : ℕ, 
  y > x ∧ 
  (∃ z : ℕ, z = 2 * y ∧ 
  (∃ w : ℕ, w = z - 3 ∧ 
  (∃ u : ℕ, u = w + 1 ∧ u = 8))) ∧ 
  x + y + 2 * y + (2 * y - 3) + (2 * y - 2) = 33 ∧ 
  y - x = 2 :=
sorry

end NUMINAMATH_GPT_throwing_skips_l515_51526


namespace NUMINAMATH_GPT_simplify_sqrt_l515_51565

-- Define the domain and main trigonometric properties
open Real

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  sqrt (1 - 2 * sin x * cos x)

-- Define the main theorem with given conditions
theorem simplify_sqrt {x : ℝ} (h1 : (5 / 4) * π < x) (h2 : x < (3 / 2) * π) (h3 : cos x > sin x) :
  simplify_expression x = cos x - sin x :=
  sorry

end NUMINAMATH_GPT_simplify_sqrt_l515_51565


namespace NUMINAMATH_GPT_calculate_annual_rent_l515_51592

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end NUMINAMATH_GPT_calculate_annual_rent_l515_51592


namespace NUMINAMATH_GPT_number_of_intersections_l515_51577

def line₁ (x y : ℝ) := 2 * x - 3 * y + 6 = 0
def line₂ (x y : ℝ) := 5 * x + 2 * y - 10 = 0
def line₃ (x y : ℝ) := x - 2 * y + 1 = 0
def line₄ (x y : ℝ) := 3 * x - 4 * y + 8 = 0

theorem number_of_intersections : 
  ∃! (p₁ p₂ p₃ : ℝ × ℝ),
    (line₁ p₁.1 p₁.2 ∨ line₂ p₁.1 p₁.2) ∧ (line₃ p₁.1 p₁.2 ∨ line₄ p₁.1 p₁.2) ∧
    (line₁ p₂.1 p₂.2 ∨ line₂ p₂.1 p₂.2) ∧ (line₃ p₂.1 p₂.2 ∨ line₄ p₂.1 p₂.2) ∧
    (line₁ p₃.1 p₃.2 ∨ line₂ p₃.1 p₃.2) ∧ (line₃ p₃.1 p₃.2 ∨ line₄ p₃.1 p₃.2) ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ := 
sorry

end NUMINAMATH_GPT_number_of_intersections_l515_51577


namespace NUMINAMATH_GPT_problem_l515_51553

noncomputable def f(x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1
noncomputable def f_prime(x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + b

theorem problem (a b : ℝ) 
  (h₁ : f_prime 1 a b = 4) 
  (h₂ : f 1 a b = 3) : 
  a + b = 2 :=
sorry

end NUMINAMATH_GPT_problem_l515_51553


namespace NUMINAMATH_GPT_distinct_permutations_mathematics_l515_51506

theorem distinct_permutations_mathematics : 
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  (n.factorial / (freqM.factorial * freqA.factorial * freqT.factorial)) = 4989600 :=
by
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  sorry

end NUMINAMATH_GPT_distinct_permutations_mathematics_l515_51506


namespace NUMINAMATH_GPT_robis_savings_in_january_l515_51533

theorem robis_savings_in_january (x : ℕ) (h: (x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) = 126)) : x = 11 := 
by {
  -- By simplification, the lean equivalent proof would include combining like
  -- terms and solving the resulting equation. For now, we'll use sorry.
  sorry
}

end NUMINAMATH_GPT_robis_savings_in_january_l515_51533


namespace NUMINAMATH_GPT_correct_operation_l515_51536

theorem correct_operation (a : ℝ) : 2 * (a^2) * a = 2 * (a^3) := by sorry

end NUMINAMATH_GPT_correct_operation_l515_51536


namespace NUMINAMATH_GPT_geometric_sequence_sum_l515_51559

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a_n 1 + a_n 3 = 5) :
  a_n 3 + a_n 5 = 20 :=
by
  -- The proof would go here, but it is not required for this task.
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l515_51559


namespace NUMINAMATH_GPT_distance_between_chords_l515_51574

theorem distance_between_chords (R : ℝ) (AB CD : ℝ) (d : ℝ) : 
  R = 25 → AB = 14 → CD = 40 → (d = 39 ∨ d = 9) :=
by intros; sorry

end NUMINAMATH_GPT_distance_between_chords_l515_51574


namespace NUMINAMATH_GPT_craft_store_pricing_maximize_daily_profit_l515_51554

theorem craft_store_pricing (profit_per_item marked_price cost_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₂ : 8 * 0.85 * marked_price + 12 * (marked_price - 35) = 20 * cost_price)
  : cost_price = 155 ∧ marked_price = 200 := 
sorry

theorem maximize_daily_profit (profit_per_item cost_price marked_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₃ : ∀ p : ℝ, (100 + 4 * (200 - p)) * (p - cost_price) ≤ 4900)
  : p = 190 ∧ daily_profit = 4900 :=
sorry

end NUMINAMATH_GPT_craft_store_pricing_maximize_daily_profit_l515_51554


namespace NUMINAMATH_GPT_aeroplane_distance_l515_51530

theorem aeroplane_distance
  (speed : ℝ) (time : ℝ) (distance : ℝ)
  (h1 : speed = 590)
  (h2 : time = 8)
  (h3 : distance = speed * time) :
  distance = 4720 :=
by {
  -- The proof will contain the steps to show that distance = 4720
  sorry
}

end NUMINAMATH_GPT_aeroplane_distance_l515_51530


namespace NUMINAMATH_GPT_sequence_eventually_periodic_l515_51594

-- Definitions based on the conditions
def positive_int_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < a n

def satisfies_condition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)

-- Assertion to prove based on the question
theorem sequence_eventually_periodic (a : ℕ → ℕ) 
  (h1 : positive_int_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ p : ℕ, ∃ k : ℕ, ∀ n : ℕ, a (n + k) = a n :=
sorry

end NUMINAMATH_GPT_sequence_eventually_periodic_l515_51594


namespace NUMINAMATH_GPT_inequality_always_true_l515_51513

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) : a + c > b + d :=
by sorry

end NUMINAMATH_GPT_inequality_always_true_l515_51513


namespace NUMINAMATH_GPT_total_square_footage_after_expansion_l515_51583

-- Definitions from the conditions
def size_smaller_house_initial : ℕ := 5200
def size_larger_house : ℕ := 7300
def expansion_smaller_house : ℕ := 3500

-- The new size of the smaller house after expansion
def size_smaller_house_after_expansion : ℕ :=
  size_smaller_house_initial + expansion_smaller_house

-- The new total square footage
def new_total_square_footage : ℕ :=
  size_smaller_house_after_expansion + size_larger_house

-- Goal statement: Prove the total new square footage is 16000 sq. ft.
theorem total_square_footage_after_expansion : new_total_square_footage = 16000 := by
  sorry

end NUMINAMATH_GPT_total_square_footage_after_expansion_l515_51583


namespace NUMINAMATH_GPT_number_of_footballs_is_3_l515_51548

-- Define the variables and conditions directly from the problem

-- Let F be the cost of one football and S be the cost of one soccer ball
variable (F S : ℝ)

-- Condition 1: Some footballs and 1 soccer ball cost 155 dollars
variable (number_of_footballs : ℝ)
variable (H1 : F * number_of_footballs + S = 155)

-- Condition 2: 2 footballs and 3 soccer balls cost 220 dollars
variable (H2 : 2 * F + 3 * S = 220)

-- Condition 3: The cost of one soccer ball is 50 dollars
variable (H3 : S = 50)

-- Theorem: Prove that the number of footballs in the first set is 3
theorem number_of_footballs_is_3 (H1 H2 H3 : Prop) :
  number_of_footballs = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_footballs_is_3_l515_51548


namespace NUMINAMATH_GPT_zeros_in_square_of_nines_l515_51557

def num_zeros (n : ℕ) (m : ℕ) : ℕ :=
  -- Count the number of zeros in the decimal representation of m
sorry

theorem zeros_in_square_of_nines :
  num_zeros 6 ((10^6 - 1)^2) = 5 :=
sorry

end NUMINAMATH_GPT_zeros_in_square_of_nines_l515_51557


namespace NUMINAMATH_GPT_max_m_value_min_value_expression_l515_51585

-- Define the conditions for the inequality where the solution is the entire real line
theorem max_m_value (x m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
sorry

-- Define the conditions for a, b, c > 0 and their sum equal to 1
-- and prove the minimum value of 4a^2 + 9b^2 + c^2
theorem min_value_expression (a b c : ℝ) (hpos1 : a > 0) (hpos2 : b > 0) (hpos3 : c > 0) (hsum : a + b + c = 1) :
  4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧ (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 → a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
sorry

end NUMINAMATH_GPT_max_m_value_min_value_expression_l515_51585


namespace NUMINAMATH_GPT_elodie_rats_l515_51529

-- Define the problem conditions as hypotheses
def E (H : ℕ) : ℕ := H + 10
def K (H : ℕ) : ℕ := 3 * (E H + H)

-- The goal is to prove E = 30 given the conditions
theorem elodie_rats (H : ℕ) (h1 : E (H := H) + H + K (H := H) = 200) : E H = 30 :=
by
  sorry

end NUMINAMATH_GPT_elodie_rats_l515_51529


namespace NUMINAMATH_GPT_n_in_S_implies_n2_in_S_l515_51581

def S (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a ≥ b ∧ c ≥ d ∧ e ≥ f ∧
  n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2

theorem n_in_S_implies_n2_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end NUMINAMATH_GPT_n_in_S_implies_n2_in_S_l515_51581


namespace NUMINAMATH_GPT_commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l515_51598

def binary_star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

-- Statement (A): Commutativity
theorem commutative_star (x y : ℝ) : binary_star x y = binary_star y x := sorry

-- Statement (B): Distributivity (proving it's not distributive)
theorem not_distributive_star (x y z : ℝ) : ¬(binary_star x (y + z) = binary_star x y + binary_star x z) := sorry

-- Statement (C): Special case
theorem special_case_star (x : ℝ) : binary_star (x + 1) (x - 1) = binary_star x x - 1 := sorry

-- Statement (D): Identity element
theorem no_identity_star (x e : ℝ) : ¬(binary_star x e = x ∧ binary_star e x = x) := sorry

-- Statement (E): Associativity (proving it's not associative)
theorem not_associative_star (x y z : ℝ) : ¬(binary_star x (binary_star y z) = binary_star (binary_star x y) z) := sorry

end NUMINAMATH_GPT_commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l515_51598


namespace NUMINAMATH_GPT_triangle_is_isosceles_l515_51578

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_sum_angles : A + B + C = π)
  (h_condition : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l515_51578


namespace NUMINAMATH_GPT_typing_speed_in_6_minutes_l515_51517

theorem typing_speed_in_6_minutes (total_chars : ℕ) (chars_first_minute : ℕ) (chars_last_minute : ℕ) (chars_other_minutes : ℕ) :
  total_chars = 2098 →
  chars_first_minute = 112 →
  chars_last_minute = 97 →
  chars_other_minutes = 1889 →
  (1889 / 6 : ℝ) < 315 → 
  ¬(∀ n, 1 ≤ n ∧ n ≤ 14 - 6 + 1 → chars_other_minutes / 6 ≥ 946) :=
by
  -- Given that analyzing the content, 
  -- proof is skipped here, replace this line with the actual proof.
  sorry

end NUMINAMATH_GPT_typing_speed_in_6_minutes_l515_51517


namespace NUMINAMATH_GPT_steve_has_7_fewer_b_berries_l515_51588

-- Define the initial number of berries Stacy has
def stacy_initial_berries : ℕ := 32

-- Define the number of berries Steve takes from Stacy
def steve_takes : ℕ := 4

-- Define the initial number of berries Steve has
def steve_initial_berries : ℕ := 21

-- Using the given conditions, prove that Steve has 7 fewer berries compared to Stacy's initial amount
theorem steve_has_7_fewer_b_berries :
  stacy_initial_berries - (steve_initial_berries + steve_takes) = 7 := 
by
  sorry

end NUMINAMATH_GPT_steve_has_7_fewer_b_berries_l515_51588


namespace NUMINAMATH_GPT_bricklayer_team_size_l515_51550

/-- Problem: Prove the number of bricklayers in the team -/
theorem bricklayer_team_size
  (x : ℕ)
  (h1 : 432 = (432 * (x - 4) / x) + 9 * (x - 4)) :
  x = 16 :=
sorry

end NUMINAMATH_GPT_bricklayer_team_size_l515_51550


namespace NUMINAMATH_GPT_cos_third_quadrant_l515_51595

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end NUMINAMATH_GPT_cos_third_quadrant_l515_51595


namespace NUMINAMATH_GPT_fifth_plot_difference_l515_51540

-- Define the dimensions of the plots
def plot_width (n : Nat) : Nat := 3 + 2 * (n - 1)
def plot_length (n : Nat) : Nat := 4 + 3 * (n - 1)

-- Define the number of tiles in a plot
def tiles_in_plot (n : Nat) : Nat := plot_width n * plot_length n

-- The main theorem to prove the required difference
theorem fifth_plot_difference :
  tiles_in_plot 5 - tiles_in_plot 4 = 59 := sorry

end NUMINAMATH_GPT_fifth_plot_difference_l515_51540


namespace NUMINAMATH_GPT_total_votes_cast_l515_51545

theorem total_votes_cast (F A T : ℕ) (h1 : F = A + 70) (h2 : A = 2 * T / 5) (h3 : T = F + A) : T = 350 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l515_51545


namespace NUMINAMATH_GPT_price_reduction_for_2100_yuan_price_reduction_for_max_profit_l515_51520

-- Condition definitions based on the problem statement
def units_sold (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_unit (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

-- Statement to prove the price reduction for achieving a daily profit of 2100 yuan
theorem price_reduction_for_2100_yuan : ∃ x : ℝ, daily_profit x = 2100 ∧ x = 20 :=
  sorry

-- Statement to prove the price reduction to maximize the daily profit
theorem price_reduction_for_max_profit : ∀ x : ℝ, ∃ y : ℝ, (∀ z : ℝ, daily_profit z ≤ y) ∧ x = 17.5 :=
  sorry

end NUMINAMATH_GPT_price_reduction_for_2100_yuan_price_reduction_for_max_profit_l515_51520


namespace NUMINAMATH_GPT_sandy_carrots_l515_51505

-- Definitions and conditions
def total_carrots : ℕ := 14
def mary_carrots : ℕ := 6

-- Proof statement
theorem sandy_carrots : (total_carrots - mary_carrots) = 8 :=
by
  -- sorry is used to bypass the actual proof steps
  sorry

end NUMINAMATH_GPT_sandy_carrots_l515_51505


namespace NUMINAMATH_GPT_apples_purchased_by_danny_l515_51523

theorem apples_purchased_by_danny (pinky_apples : ℕ) (total_apples : ℕ) (danny_apples : ℕ) :
  pinky_apples = 36 → total_apples = 109 → danny_apples = total_apples - pinky_apples → danny_apples = 73 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_apples_purchased_by_danny_l515_51523


namespace NUMINAMATH_GPT_completing_the_square_l515_51556

theorem completing_the_square (x m n : ℝ) 
  (h : x^2 - 6 * x = 1) 
  (hm : (x - m)^2 = n) : 
  m + n = 13 :=
sorry

end NUMINAMATH_GPT_completing_the_square_l515_51556


namespace NUMINAMATH_GPT_count_integers_M_3_k_l515_51509

theorem count_integers_M_3_k (M : ℕ) (hM : M < 500) :
  (∃ k : ℕ, k ≥ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ M = 2 * k * (m + k - 1)) ∧
  (∃ k1 k2 k3 k4 : ℕ, k1 ≠ k2 ∧ k1 ≠ k3 ∧ k1 ≠ k4 ∧
    k2 ≠ k3 ∧ k2 ≠ k4 ∧ k3 ≠ k4 ∧
    (M / 2 = (k1 + k2 + k3 + k4) ∨ M / 2 = (k1 * k2 * k3 * k4))) →
  (∃ n : ℕ, n = 6) :=
by
  sorry

end NUMINAMATH_GPT_count_integers_M_3_k_l515_51509


namespace NUMINAMATH_GPT_sum_of_g1_l515_51518

-- Define the main conditions
variable {g : ℝ → ℝ}
variable (h_nonconst : ∀ a b : ℝ, a ≠ b → g a ≠ g b)
axiom main_condition : ∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x) ^ 2 / (2025 * x)

-- Define the goal
theorem sum_of_g1 :
  g 1 = 6075 :=
sorry

end NUMINAMATH_GPT_sum_of_g1_l515_51518


namespace NUMINAMATH_GPT_total_ways_to_buy_l515_51541

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end NUMINAMATH_GPT_total_ways_to_buy_l515_51541


namespace NUMINAMATH_GPT_average_children_with_children_l515_51590

theorem average_children_with_children (total_families : ℕ) (average_children : ℚ) (childless_families : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  (total_families * average_children) / (total_families - childless_families) = 4.0 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_average_children_with_children_l515_51590


namespace NUMINAMATH_GPT_logarithmic_inequality_l515_51572

theorem logarithmic_inequality (a : ℝ) (h : a > 1) : 
  1 / 2 + 1 / Real.log a ≥ 1 := 
sorry

end NUMINAMATH_GPT_logarithmic_inequality_l515_51572


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l515_51508

theorem number_of_ordered_pairs : 
  ∃ n, n = 325 ∧ ∀ (a b : ℤ), 
    1 ≤ a ∧ a ≤ 50 ∧ a % 2 = 1 ∧ 
    0 ≤ b ∧ b % 2 = 0 ∧ 
    ∃ r s : ℤ, r + s = -a ∧ r * s = b :=
sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l515_51508


namespace NUMINAMATH_GPT_sqrt_difference_square_l515_51584

theorem sqrt_difference_square (a b : ℝ) (h₁ : a = Real.sqrt 3 + Real.sqrt 2) (h₂ : b = Real.sqrt 3 - Real.sqrt 2) : a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_difference_square_l515_51584


namespace NUMINAMATH_GPT_sqrt_sum_equality_l515_51546

theorem sqrt_sum_equality :
  (Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_equality_l515_51546


namespace NUMINAMATH_GPT_laboratory_painting_area_laboratory_paint_needed_l515_51573

section
variable (l w h excluded_area : ℝ)
variable (paint_per_sqm : ℝ)

def painting_area (l w h excluded_area : ℝ) : ℝ :=
  let total_area := (l * w + w * h + h * l) * 2 - (l * w)
  total_area - excluded_area

def paint_needed (painting_area paint_per_sqm : ℝ) : ℝ :=
  painting_area * paint_per_sqm

theorem laboratory_painting_area :
  painting_area 12 8 6 28.4 = 307.6 :=
by
  simp [painting_area, *]
  norm_num

theorem laboratory_paint_needed :
  paint_needed 307.6 0.2 = 61.52 :=
by
  simp [paint_needed, *]
  norm_num

end

end NUMINAMATH_GPT_laboratory_painting_area_laboratory_paint_needed_l515_51573


namespace NUMINAMATH_GPT_find_number_l515_51576

theorem find_number (x : ℝ) (h : 0.65 * x = 0.8 * x - 21) : x = 140 := by
  sorry

end NUMINAMATH_GPT_find_number_l515_51576


namespace NUMINAMATH_GPT_blue_beads_l515_51515

-- Variables to denote the number of blue, red, white, and silver beads
variables (B R W S : ℕ)

-- Conditions derived from the problem statement
def conditions : Prop :=
  (R = 2 * B) ∧
  (W = B + R) ∧
  (S = 10) ∧
  (B + R + W + S = 40)

-- The theorem to prove
theorem blue_beads (B R W S : ℕ) (h : conditions B R W S) : B = 5 :=
by
  sorry

end NUMINAMATH_GPT_blue_beads_l515_51515


namespace NUMINAMATH_GPT_books_taken_out_on_Tuesday_l515_51510

theorem books_taken_out_on_Tuesday (T : ℕ) (initial_books : ℕ) (returned_books : ℕ) (withdrawn_books : ℕ) (final_books : ℕ) :
  initial_books = 250 ∧
  returned_books = 35 ∧
  withdrawn_books = 15 ∧
  final_books = 150 →
  T = 120 :=
by
  sorry

end NUMINAMATH_GPT_books_taken_out_on_Tuesday_l515_51510


namespace NUMINAMATH_GPT_find_m_n_and_sqrt_l515_51514

-- definitions based on conditions
def condition_1 (m : ℤ) : Prop := m + 3 = 1
def condition_2 (n : ℤ) : Prop := 2 * n - 12 = 64

-- the proof problem statement
theorem find_m_n_and_sqrt (m n : ℤ) (h1 : condition_1 m) (h2 : condition_2 n) : 
  m = -2 ∧ n = 38 ∧ Int.sqrt (m + n) = 6 := 
sorry

end NUMINAMATH_GPT_find_m_n_and_sqrt_l515_51514


namespace NUMINAMATH_GPT_third_smallest_triangular_square_l515_51502

theorem third_smallest_triangular_square :
  ∃ n : ℕ, n = 1225 ∧ 
           (∃ x y : ℕ, y^2 - 8 * x^2 = 1 ∧ 
                        y = 99 ∧ x = 35) :=
by
  sorry

end NUMINAMATH_GPT_third_smallest_triangular_square_l515_51502


namespace NUMINAMATH_GPT_euler_conjecture_disproof_l515_51537

theorem euler_conjecture_disproof :
    ∃ (n : ℕ), 133^4 + 110^4 + 56^4 = n^4 ∧ n = 143 :=
by {
  use 143,
  sorry
}

end NUMINAMATH_GPT_euler_conjecture_disproof_l515_51537


namespace NUMINAMATH_GPT_find_g2_l515_51543

variable {R : Type*} [Nonempty R] [Field R]

-- Define the function g
def g (x : R) : R := sorry

-- Given conditions
axiom condition1 : ∀ x y : R, x * g y = 2 * y * g x
axiom condition2 : g 10 = 5

-- The statement to be proved
theorem find_g2 : g 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_g2_l515_51543


namespace NUMINAMATH_GPT_blocks_fit_into_box_l515_51586

theorem blocks_fit_into_box :
  let box_height := 8
  let box_width := 10
  let box_length := 12
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let box_volume := box_height * box_width * box_length
  let block_volume := block_height * block_width * block_length
  let num_blocks := box_volume / block_volume
  num_blocks = 40 :=
by
  sorry

end NUMINAMATH_GPT_blocks_fit_into_box_l515_51586


namespace NUMINAMATH_GPT_line_quadrant_relationship_l515_51538

theorem line_quadrant_relationship
  (a b c : ℝ)
  (passes_first_second_fourth : ∀ x y : ℝ, (a * x + b * y + c = 0) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) :
  (a * b > 0) ∧ (b * c < 0) :=
sorry

end NUMINAMATH_GPT_line_quadrant_relationship_l515_51538


namespace NUMINAMATH_GPT_combined_stripes_eq_22_l515_51507

def stripes_olga_per_shoe : ℕ := 3
def shoes_per_person : ℕ := 2
def stripes_olga_total : ℕ := stripes_olga_per_shoe * shoes_per_person

def stripes_rick_per_shoe : ℕ := stripes_olga_per_shoe - 1
def stripes_rick_total : ℕ := stripes_rick_per_shoe * shoes_per_person

def stripes_hortense_per_shoe : ℕ := stripes_olga_per_shoe * 2
def stripes_hortense_total : ℕ := stripes_hortense_per_shoe * shoes_per_person

def total_stripes : ℕ := stripes_olga_total + stripes_rick_total + stripes_hortense_total

theorem combined_stripes_eq_22 : total_stripes = 22 := by
  sorry

end NUMINAMATH_GPT_combined_stripes_eq_22_l515_51507


namespace NUMINAMATH_GPT_largest_n_satisfying_expression_l515_51596

theorem largest_n_satisfying_expression :
  ∃ n < 100000, (n - 3)^5 - n^2 + 10 * n - 30 ≡ 0 [MOD 3] ∧ 
  (∀ m, m < 100000 → (m - 3)^5 - m^2 + 10 * m - 30 ≡ 0 [MOD 3] → m ≤ 99998) := sorry

end NUMINAMATH_GPT_largest_n_satisfying_expression_l515_51596


namespace NUMINAMATH_GPT_symmetric_about_line_l515_51575

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)
noncomputable def g (x a : ℝ) : ℝ := f (x + a)

theorem symmetric_about_line (a : ℝ) : (∀ x, g x a = x + 1) ↔ a = 0 :=
by sorry

end NUMINAMATH_GPT_symmetric_about_line_l515_51575


namespace NUMINAMATH_GPT_log_inequality_l515_51512

variable (a b : ℝ)

theorem log_inequality (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b :=
sorry

end NUMINAMATH_GPT_log_inequality_l515_51512


namespace NUMINAMATH_GPT_diff_set_Q_minus_P_l515_51527

def P (x : ℝ) : Prop := 1 - (2 / x) < 0
def Q (x : ℝ) : Prop := |x - 2| < 1
def diff_set (P Q : ℝ → Prop) (x : ℝ) : Prop := Q x ∧ ¬ P x

theorem diff_set_Q_minus_P :
  ∀ x : ℝ, diff_set Q P x ↔ (2 ≤ x ∧ x < 3) :=
by
  sorry

end NUMINAMATH_GPT_diff_set_Q_minus_P_l515_51527


namespace NUMINAMATH_GPT_least_integer_square_l515_51519

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_least_integer_square_l515_51519


namespace NUMINAMATH_GPT_proof_problem_l515_51532

theorem proof_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
sorry

end NUMINAMATH_GPT_proof_problem_l515_51532


namespace NUMINAMATH_GPT_original_decimal_l515_51521

theorem original_decimal (x : ℝ) (h : 1000 * x / 100 = 12.5) : x = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_original_decimal_l515_51521


namespace NUMINAMATH_GPT_min_quality_inspection_machines_l515_51511

theorem min_quality_inspection_machines (z x : ℕ) :
  (z + 30 * x) / 30 = 1 →
  (z + 10 * x) / 10 = 2 →
  (z + 5 * x) / 5 ≥ 4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_min_quality_inspection_machines_l515_51511


namespace NUMINAMATH_GPT_min_value_arith_seq_l515_51582

theorem min_value_arith_seq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + c = 2 * b) :
  (a + c) / b + b / (a + c) ≥ 5 / 2 := 
sorry

end NUMINAMATH_GPT_min_value_arith_seq_l515_51582


namespace NUMINAMATH_GPT_calculate_expression_l515_51504

theorem calculate_expression (x : ℝ) (h : x = 3) : (x^2 - 5 * x + 4) / (x - 4) = 2 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_calculate_expression_l515_51504


namespace NUMINAMATH_GPT_eq_determines_ratio_l515_51547

theorem eq_determines_ratio (a b x y : ℝ) (h : a * x^3 + b * x^2 * y + b * x * y^2 + a * y^3 = 0) :
  ∃ t : ℝ, t = x / y ∧ (a * t^3 + b * t^2 + b * t + a = 0) :=
sorry

end NUMINAMATH_GPT_eq_determines_ratio_l515_51547


namespace NUMINAMATH_GPT_max_light_window_l515_51501

noncomputable def max_window_light : Prop :=
  ∃ (x : ℝ), (4 - 2 * x) / 3 * x = -2 / 3 * (x - 1) ^ 2 + 2 / 3 ∧ x = 1 ∧ (4 - 2 * x) / 3 = 2 / 3

theorem max_light_window : max_window_light :=
by
  sorry

end NUMINAMATH_GPT_max_light_window_l515_51501


namespace NUMINAMATH_GPT_max_value_of_z_l515_51525

variable (x y z : ℝ)

def condition1 : Prop := 2 * x + y ≤ 4
def condition2 : Prop := x ≤ y
def condition3 : Prop := x ≥ 1 / 2
def objective_function : ℝ := 2 * x - y

theorem max_value_of_z :
  (∀ x y, condition1 x y ∧ condition2 x y ∧ condition3 x → z = objective_function x y) →
  z ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_z_l515_51525


namespace NUMINAMATH_GPT_selection_schemes_correct_l515_51534

-- Define the problem parameters
def number_of_selection_schemes (persons : ℕ) (cities : ℕ) (persons_cannot_visit : ℕ) : ℕ :=
  let choices_for_paris := persons - persons_cannot_visit
  let remaining_people := persons - 1
  choices_for_paris * remaining_people * (remaining_people - 1) * (remaining_people - 2)

-- Define the example constants
def total_people : ℕ := 6
def total_cities : ℕ := 4
def cannot_visit_paris : ℕ := 2

-- The statement to be proved
theorem selection_schemes_correct : 
  number_of_selection_schemes total_people total_cities cannot_visit_paris = 240 := by
  sorry

end NUMINAMATH_GPT_selection_schemes_correct_l515_51534


namespace NUMINAMATH_GPT_total_owed_proof_l515_51549

-- Define initial conditions
def initial_owed : ℕ := 20
def borrowed : ℕ := 8

-- Define the total amount owed
def total_owed : ℕ := initial_owed + borrowed

-- Prove the statement
theorem total_owed_proof : total_owed = 28 := 
by 
  -- Proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_total_owed_proof_l515_51549


namespace NUMINAMATH_GPT_teachers_per_grade_correct_l515_51551

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def parents_per_grade : ℕ := 2
def number_of_grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Total number of students
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders

-- Total number of parents
def total_parents : ℕ := parents_per_grade * number_of_grades

-- Total number of seats available on the buses
def total_seats : ℕ := buses * seats_per_bus

-- Seats left for teachers
def seats_for_teachers : ℕ := total_seats - total_students - total_parents

-- Teachers per grade
def teachers_per_grade : ℕ := seats_for_teachers / number_of_grades

theorem teachers_per_grade_correct : teachers_per_grade = 4 := sorry

end NUMINAMATH_GPT_teachers_per_grade_correct_l515_51551


namespace NUMINAMATH_GPT_probability_blue_point_l515_51579

-- Definitions of the random points
def is_random_point (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2

-- Definition of the condition for the probability problem
def condition (x y : ℝ) : Prop :=
  x < y ∧ y < 3 * x

-- Statement of the theorem
theorem probability_blue_point (x y : ℝ) (h1 : is_random_point x) (h2 : is_random_point y) :
  ∃ p : ℝ, (p = 1 / 3) ∧ (∃ (hx : x < y) (hy : y < 3 * x), x ≤ 2 ∧ 0 ≤ x ∧ y ≤ 2 ∧ 0 ≤ y) :=
by
  sorry

end NUMINAMATH_GPT_probability_blue_point_l515_51579


namespace NUMINAMATH_GPT_cost_of_adult_ticket_l515_51535

theorem cost_of_adult_ticket
    (child_ticket_cost : ℝ)
    (total_tickets : ℕ)
    (total_receipts : ℝ)
    (adult_tickets_sold : ℕ)
    (A : ℝ)
    (child_tickets_sold : ℕ := total_tickets - adult_tickets_sold)
    (total_revenue_adult : ℝ := adult_tickets_sold * A)
    (total_revenue_child : ℝ := child_tickets_sold * child_ticket_cost) :
    child_ticket_cost = 4 →
    total_tickets = 130 →
    total_receipts = 840 →
    adult_tickets_sold = 90 →
    total_revenue_adult + total_revenue_child = total_receipts →
    A = 7.56 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_l515_51535


namespace NUMINAMATH_GPT_residents_ticket_price_l515_51562

theorem residents_ticket_price
  (total_attendees : ℕ)
  (resident_count : ℕ)
  (non_resident_price : ℝ)
  (total_revenue : ℝ)
  (R : ℝ)
  (h1 : total_attendees = 586)
  (h2 : resident_count = 219)
  (h3 : non_resident_price = 17.95)
  (h4 : total_revenue = 9423.70)
  (total_residents_pay : ℝ := resident_count * R)
  (total_non_residents_pay : ℝ := (total_attendees - resident_count) * non_resident_price)
  (h5 : total_revenue = total_residents_pay + total_non_residents_pay) :
  R = 12.95 := by
  sorry

end NUMINAMATH_GPT_residents_ticket_price_l515_51562


namespace NUMINAMATH_GPT_sum_of_digits_l515_51503

-- Conditions setup
variables (a b c d : ℕ)
variables (h1 : a + c = 10) 
variables (h2 : b + c = 9) 
variables (h3 : a + d = 10)
variables (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

theorem sum_of_digits : a + b + c + d = 19 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l515_51503


namespace NUMINAMATH_GPT_find_function_l515_51522

-- Let f be a differentiable function over all real numbers
variable (f : ℝ → ℝ)
variable (k : ℝ)

-- Condition: f is differentiable over (-∞, ∞)
variable (h_diff : differentiable ℝ f)

-- Condition: f(0) = 1
variable (h_init : f 0 = 1)

-- Condition: for any x1, x2 in ℝ, f(x1 + x2) ≥ f(x1) f(x2)
variable (h_ineq : ∀ x1 x2 : ℝ, f (x1 + x2) ≥ f x1 * f x2)

-- We aim to prove: f(x) = e^(kx)
theorem find_function : ∃ k : ℝ, ∀ x : ℝ, f x = Real.exp (k * x) :=
sorry

end NUMINAMATH_GPT_find_function_l515_51522


namespace NUMINAMATH_GPT_fraction_of_recipe_l515_51570

theorem fraction_of_recipe 
  (recipe_sugar recipe_milk recipe_flour : ℚ)
  (have_sugar have_milk have_flour : ℚ)
  (h1 : recipe_sugar = 3/4) (h2 : recipe_milk = 2/3) (h3 : recipe_flour = 3/8)
  (h4 : have_sugar = 2/4) (h5 : have_milk = 1/2) (h6 : have_flour = 1/4) : 
  (min ((have_sugar / recipe_sugar)) (min ((have_milk / recipe_milk)) (have_flour / recipe_flour)) = 2/3) := 
by sorry

end NUMINAMATH_GPT_fraction_of_recipe_l515_51570


namespace NUMINAMATH_GPT_cost_price_eq_560_l515_51552

variables (C SP1 SP2 : ℝ)
variables (h1 : SP1 = 0.79 * C) (h2 : SP2 = SP1 + 140) (h3 : SP2 = 1.04 * C)

theorem cost_price_eq_560 : C = 560 :=
by 
  sorry

end NUMINAMATH_GPT_cost_price_eq_560_l515_51552


namespace NUMINAMATH_GPT_nat_solution_unique_l515_51524

theorem nat_solution_unique (x y : ℕ) (h : x + y = x * y) : (x, y) = (2, 2) :=
sorry

end NUMINAMATH_GPT_nat_solution_unique_l515_51524


namespace NUMINAMATH_GPT_geometric_series_sum_l515_51568

theorem geometric_series_sum :
  let a := 6
  let r := - (2 / 5)
  s = ∑' n, (a * r ^ n) ↔ s = 30 / 7 :=
sorry

end NUMINAMATH_GPT_geometric_series_sum_l515_51568


namespace NUMINAMATH_GPT_inequality_of_abc_l515_51564

variable {a b c : ℝ}

theorem inequality_of_abc 
    (h : 0 < a ∧ 0 < b ∧ 0 < c)
    (h₁ : abc * (a + b + c) = ab + bc + ca) :
    5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end NUMINAMATH_GPT_inequality_of_abc_l515_51564
