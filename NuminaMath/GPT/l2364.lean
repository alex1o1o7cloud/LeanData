import Mathlib

namespace NUMINAMATH_GPT_average_speed_entire_journey_l2364_236454

-- Define the average speed for the journey from x to y
def speed_xy := 60

-- Define the average speed for the journey from y to x
def speed_yx := 30

-- Definition for the distance (D) (it's an abstract value, so we don't need to specify)
variable (D : ℝ) (hD : D > 0)

-- Theorem stating that the average speed for the entire journey is 40 km/hr
theorem average_speed_entire_journey : 
  2 * D / ((D / speed_xy) + (D / speed_yx)) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_entire_journey_l2364_236454


namespace NUMINAMATH_GPT_parallel_vectors_solution_l2364_236488

theorem parallel_vectors_solution 
  (x : ℝ) 
  (a : ℝ × ℝ := (-1, 3)) 
  (b : ℝ × ℝ := (x, 1)) 
  (h : ∃ k : ℝ, a = k • b) :
  x = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_solution_l2364_236488


namespace NUMINAMATH_GPT_joey_average_speed_l2364_236491

noncomputable def average_speed_of_round_trip (distance_out : ℝ) (time_out : ℝ) (speed_return : ℝ) : ℝ :=
  let distance_return := distance_out
  let total_distance := distance_out + distance_return
  let time_return := distance_return / speed_return
  let total_time := time_out + time_return
  total_distance / total_time

theorem joey_average_speed :
  average_speed_of_round_trip 2 1 6.000000000000002 = 3 := by
  sorry

end NUMINAMATH_GPT_joey_average_speed_l2364_236491


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_for_extreme_value_l2364_236429

-- Defining the function f(x) = ax^3 + x + 1
def f (a x : ℝ) : ℝ := a * x^3 + x + 1

-- Defining the condition for f to have an extreme value
def has_extreme_value (a : ℝ) : Prop := ∃ x : ℝ, deriv (f a) x = 0

-- Stating the problem
theorem necessary_and_sufficient_condition_for_extreme_value (a : ℝ) :
  has_extreme_value a ↔ a < 0 := by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_for_extreme_value_l2364_236429


namespace NUMINAMATH_GPT_reduced_price_per_dozen_apples_l2364_236461

variables (P R : ℝ) 

theorem reduced_price_per_dozen_apples (h₁ : R = 0.70 * P) 
  (h₂ : (30 / P + 54) * R = 30) :
  12 * R = 2 := 
sorry

end NUMINAMATH_GPT_reduced_price_per_dozen_apples_l2364_236461


namespace NUMINAMATH_GPT_bakery_batches_per_day_l2364_236422

-- Definitions for the given problem's conditions
def baguettes_per_batch := 48
def baguettes_sold_batch1 := 37
def baguettes_sold_batch2 := 52
def baguettes_sold_batch3 := 49
def baguettes_left := 6

-- Theorem stating the number of batches made
theorem bakery_batches_per_day : 
  (baguettes_sold_batch1 + baguettes_sold_batch2 + baguettes_sold_batch3 + baguettes_left) / baguettes_per_batch = 3 :=
by 
  sorry

end NUMINAMATH_GPT_bakery_batches_per_day_l2364_236422


namespace NUMINAMATH_GPT_algebraic_expression_value_l2364_236480

theorem algebraic_expression_value (x : ℝ) 
  (h : 2 * x^2 + 3 * x + 7 = 8) : 
  4 * x^2 + 6 * x - 9 = -7 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2364_236480


namespace NUMINAMATH_GPT_spherical_triangle_area_correct_l2364_236412

noncomputable def spherical_triangle_area (R α β γ : ℝ) : ℝ :=
  R^2 * (α + β + γ - Real.pi)

theorem spherical_triangle_area_correct (R α β γ : ℝ) :
  spherical_triangle_area R α β γ = R^2 * (α + β + γ - Real.pi) := by
  sorry

end NUMINAMATH_GPT_spherical_triangle_area_correct_l2364_236412


namespace NUMINAMATH_GPT_car_speed_problem_l2364_236479

theorem car_speed_problem (S1 S2 : ℝ) (T : ℝ) (avg_speed : ℝ) (H1 : S1 = 70) (H2 : T = 2) (H3 : avg_speed = 80) :
  S2 = 90 :=
by
  have avg_speed_eq : avg_speed = (S1 + S2) / T := sorry
  have h : S2 = 90 := sorry
  exact h

end NUMINAMATH_GPT_car_speed_problem_l2364_236479


namespace NUMINAMATH_GPT_find_n_l2364_236427

variable (a b c n : ℤ)
variable (h1 : a + b + c = 100)
variable (h2 : a + b / 2 = 40)

theorem find_n : n = a - c := by
  sorry

end NUMINAMATH_GPT_find_n_l2364_236427


namespace NUMINAMATH_GPT_exists_infinite_diff_but_not_sum_of_kth_powers_l2364_236446

theorem exists_infinite_diff_but_not_sum_of_kth_powers (k : ℕ) (hk : k > 1) :
  ∃ (infinitely_many x : ℕ), (∃ (a b : ℕ), x = a^k - b^k) ∧ ¬ (∃ (c d : ℕ), x = c^k + d^k) :=
  sorry

end NUMINAMATH_GPT_exists_infinite_diff_but_not_sum_of_kth_powers_l2364_236446


namespace NUMINAMATH_GPT_smallest_repeating_block_length_of_7_over_13_l2364_236469

theorem smallest_repeating_block_length_of_7_over_13 : 
  ∀ k, (∃ a b, 7 / 13 = a + (b / 10^k)) → k = 6 := 
sorry

end NUMINAMATH_GPT_smallest_repeating_block_length_of_7_over_13_l2364_236469


namespace NUMINAMATH_GPT_polynomial_divisible_by_squared_root_l2364_236458

noncomputable def f (a1 a2 a3 a4 x : ℝ) : ℝ := 
  x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4

noncomputable def f_prime (a1 a2 a3 a4 x : ℝ) : ℝ := 
  4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_squared_root 
  (a1 a2 a3 a4 x0 : ℝ) 
  (h1 : f a1 a2 a3 a4 x0 = 0) 
  (h2 : f_prime a1 a2 a3 a4 x0 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x, f a1 a2 a3 a4 x = (x - x0)^2 * g x := 
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_squared_root_l2364_236458


namespace NUMINAMATH_GPT_function_values_l2364_236467

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem function_values (a b : ℝ) (h1 : f 1 a b = 2) (h2 : a = 2) : f 2 a b = 4 := by
  sorry

end NUMINAMATH_GPT_function_values_l2364_236467


namespace NUMINAMATH_GPT_height_of_fifth_tree_l2364_236438

theorem height_of_fifth_tree 
  (h₁ : tallest_tree = 108) 
  (h₂ : second_tallest_tree = 54 - 6) 
  (h₃ : third_tallest_tree = second_tallest_tree / 4) 
  (h₄ : fourth_shortest_tree = (second_tallest_tree + third_tallest_tree) - 2) 
  (h₅ : fifth_tree = 0.75 * (tallest_tree + second_tallest_tree + third_tallest_tree + fourth_shortest_tree)) : 
  fifth_tree = 169.5 :=
by
  sorry

end NUMINAMATH_GPT_height_of_fifth_tree_l2364_236438


namespace NUMINAMATH_GPT_range_of_a_l2364_236428

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) → ((x - a) ^ 2 < 1)) ↔ (1 ≤ a ∧ a ≤ 2) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l2364_236428


namespace NUMINAMATH_GPT_area_of_smaller_circle_l2364_236455

noncomputable def radius_large_circle (x : ℝ) : ℝ := 2 * x
noncomputable def radius_small_circle (y : ℝ) : ℝ := y

theorem area_of_smaller_circle 
(pa ab : ℝ)
(r : ℝ)
(area : ℝ) 
(h1 : pa = 5) 
(h2 : ab = 5) 
(h3 : radius_large_circle r = 2 * radius_small_circle r)
(h4 : 2 * radius_small_circle r + radius_large_circle r = 10)
(h5 : area = Real.pi * (radius_small_circle r)^2) 
: area = 6.25 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_smaller_circle_l2364_236455


namespace NUMINAMATH_GPT_percent_of_x_l2364_236447

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25 - x / 10 + x / 5) = (16 / 100) * x := by
  sorry

end NUMINAMATH_GPT_percent_of_x_l2364_236447


namespace NUMINAMATH_GPT_exam_max_incorrect_answers_l2364_236415

theorem exam_max_incorrect_answers :
  ∀ (c w b : ℕ),
  (c + w + b = 30) →
  (4 * c - w ≥ 85) → 
  (c ≥ 22) →
  (w ≤ 3) :=
by
  intros c w b h1 h2 h3
  sorry

end NUMINAMATH_GPT_exam_max_incorrect_answers_l2364_236415


namespace NUMINAMATH_GPT_discriminant_of_polynomial_l2364_236496

noncomputable def polynomial_discriminant (a b c : ℚ) : ℚ :=
b^2 - 4 * a * c

theorem discriminant_of_polynomial : polynomial_discriminant 2 (4 - (1/2 : ℚ)) 1 = 17 / 4 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_of_polynomial_l2364_236496


namespace NUMINAMATH_GPT_total_players_l2364_236430

def num_teams : Nat := 35
def players_per_team : Nat := 23

theorem total_players :
  num_teams * players_per_team = 805 :=
by
  sorry

end NUMINAMATH_GPT_total_players_l2364_236430


namespace NUMINAMATH_GPT_matinee_ticket_price_l2364_236497

theorem matinee_ticket_price
  (M : ℝ)  -- Denote M as the price of a matinee ticket
  (evening_ticket_price : ℝ := 12)  -- Price of an evening ticket
  (ticket_3D_price : ℝ := 20)  -- Price of a 3D ticket
  (matinee_tickets_sold : ℕ := 200)  -- Number of matinee tickets sold
  (evening_tickets_sold : ℕ := 300)  -- Number of evening tickets sold
  (tickets_3D_sold : ℕ := 100)  -- Number of 3D tickets sold
  (total_revenue : ℝ := 6600) -- Total revenue
  (h : matinee_tickets_sold * M + evening_tickets_sold * evening_ticket_price + tickets_3D_sold * ticket_3D_price = total_revenue) :
  M = 5 :=
by
  sorry

end NUMINAMATH_GPT_matinee_ticket_price_l2364_236497


namespace NUMINAMATH_GPT_correct_pythagorean_triple_l2364_236462

def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem correct_pythagorean_triple :
  (is_pythagorean_triple 1 2 3 = false) ∧ 
  (is_pythagorean_triple 4 5 6 = false) ∧ 
  (is_pythagorean_triple 6 8 9 = false) ∧ 
  (is_pythagorean_triple 7 24 25 = true) :=
by
  sorry

end NUMINAMATH_GPT_correct_pythagorean_triple_l2364_236462


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l2364_236478

theorem common_ratio_geometric_sequence (n : ℕ) :
  ∃ q : ℕ, (∀ k : ℕ, q = 4^(2*k+3) / 4^(2*k+1)) ∧ q = 16 :=
by
  use 16
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l2364_236478


namespace NUMINAMATH_GPT_integer_roots_of_polynomial_l2364_236490

theorem integer_roots_of_polynomial :
  {x : ℤ | x^3 - 4*x^2 - 14*x + 24 = 0} = {-4, -3, 3} := by
  sorry

end NUMINAMATH_GPT_integer_roots_of_polynomial_l2364_236490


namespace NUMINAMATH_GPT_probability_is_five_eleven_l2364_236409

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define a function to calculate combinations
def comb (n k : ℕ) : ℕ := n.choose k

-- Define the number of favorable outcomes for same letter and same color
def favorable_same_letter : ℕ := 4 * comb 3 2
def favorable_same_color : ℕ := 3 * comb 4 2

-- Total number of favorable outcomes
def total_favorable : ℕ := favorable_same_letter + favorable_same_color

-- Total number of ways to draw 2 cards from 12
def total_ways : ℕ := comb total_cards 2

-- Probability of drawing a winning pair
def probability_winning_pair : ℚ := total_favorable / total_ways

theorem probability_is_five_eleven : probability_winning_pair = 5 / 11 :=
by
  sorry

end NUMINAMATH_GPT_probability_is_five_eleven_l2364_236409


namespace NUMINAMATH_GPT_ratio_preference_l2364_236417

-- Definitions based on conditions
def total_respondents : ℕ := 180
def preferred_brand_x : ℕ := 150
def preferred_brand_y : ℕ := total_respondents - preferred_brand_x

-- Theorem statement to prove the ratio of preferences
theorem ratio_preference : preferred_brand_x / preferred_brand_y = 5 := by
  sorry

end NUMINAMATH_GPT_ratio_preference_l2364_236417


namespace NUMINAMATH_GPT_right_triangle_integral_sides_parity_l2364_236433

theorem right_triangle_integral_sides_parity 
  (a b c : ℕ) 
  (h : a^2 + b^2 = c^2) 
  (ha : a % 2 = 1 ∨ a % 2 = 0) 
  (hb : b % 2 = 1 ∨ b % 2 = 0) 
  (hc : c % 2 = 1 ∨ c % 2 = 0) : 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) := 
sorry

end NUMINAMATH_GPT_right_triangle_integral_sides_parity_l2364_236433


namespace NUMINAMATH_GPT_rotation_test_l2364_236437

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℝ) : Point ℝ :=
  Point.mk p.y (-p.x)

def A : Point ℝ := ⟨2, 3⟩
def B : Point ℝ := ⟨3, -2⟩

theorem rotation_test : rotate_90_clockwise A = B :=
by
  sorry

end NUMINAMATH_GPT_rotation_test_l2364_236437


namespace NUMINAMATH_GPT_distance_between_consecutive_trees_l2364_236405

-- Define the conditions as separate definitions
def num_trees : ℕ := 57
def yard_length : ℝ := 720
def spaces_between_trees := num_trees - 1

-- Define the target statement to prove
theorem distance_between_consecutive_trees :
  yard_length / spaces_between_trees = 12.857142857 := sorry

end NUMINAMATH_GPT_distance_between_consecutive_trees_l2364_236405


namespace NUMINAMATH_GPT_max_expression_value_l2364_236431

open Real

theorem max_expression_value (a b d x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : (x₁^4 - a * x₁^3 + b * x₁^2 - a * x₁ + d = 0))
  (h2 : (x₂^4 - a * x₂^3 + b * x₂^2 - a * x₂ + d = 0))
  (h3 : (x₃^4 - a * x₃^3 + b * x₃^2 - a * x₃ + d = 0))
  (h4 : (x₄^4 - a * x₄^3 + b * x₄^2 - a * x₄ + d = 0))
  (h5 : (1 / 2 ≤ x₁ ∧ x₁ ≤ 2))
  (h6 : (1 / 2 ≤ x₂ ∧ x₂ ≤ 2))
  (h7 : (1 / 2 ≤ x₃ ∧ x₃ ≤ 2))
  (h8 : (1 / 2 ≤ x₄ ∧ x₄ ≤ 2)) :
  ∃ (M : ℝ), M = 5 / 4 ∧
  (∀ (y₁ y₂ y₃ y₄ : ℝ),
    (y₁^4 - a * y₁^3 + b * y₁^2 - a * y₁ + d = 0) →
    (y₂^4 - a * y₂^3 + b * y₂^2 - a * y₂ + d = 0) →
    (y₃^4 - a * y₃^3 + b * y₃^2 - a * y₃ + d = 0) →
    (y₄^4 - a * y₄^3 + b * y₄^2 - a * y₄ + d = 0) →
    (1 / 2 ≤ y₁ ∧ y₁ ≤ 2) →
    (1 / 2 ≤ y₂ ∧ y₂ ≤ 2) →
    (1 / 2 ≤ y₃ ∧ y₃ ≤ 2) →
    (1 / 2 ≤ y₄ ∧ y₄ ≤ 2) →
    (y = (y₁ + y₂) * (y₁ + y₃) * y₄ / ((y₄ + y₂) * (y₄ + y₃) * y₁)) →
    y ≤ M) := 
sorry

end NUMINAMATH_GPT_max_expression_value_l2364_236431


namespace NUMINAMATH_GPT_abs_eq_k_solution_l2364_236474

theorem abs_eq_k_solution (k : ℝ) (h : k > 4014) :
  {x : ℝ | |x - 2007| + |x + 2007| = k} = (Set.Iio (-2007)) ∪ (Set.Ioi (2007)) :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_k_solution_l2364_236474


namespace NUMINAMATH_GPT_cost_of_building_fence_l2364_236475

-- Define the conditions
def area_of_circle := 289 -- Area in square feet
def price_per_foot := 58  -- Price in rupees per foot

-- Define the equations used in the problem
noncomputable def radius := Real.sqrt (area_of_circle / Real.pi)
noncomputable def circumference := 2 * Real.pi * radius
noncomputable def cost := circumference * price_per_foot

-- The statement to prove
theorem cost_of_building_fence : cost = 1972 :=
  sorry

end NUMINAMATH_GPT_cost_of_building_fence_l2364_236475


namespace NUMINAMATH_GPT_sum_proper_divisors_243_l2364_236494

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end NUMINAMATH_GPT_sum_proper_divisors_243_l2364_236494


namespace NUMINAMATH_GPT_polynomial_condition_l2364_236445

theorem polynomial_condition {P : Polynomial ℝ} :
  (∀ (a b c : ℝ), a * b + b * c + c * a = 0 → P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)) →
    ∃ α β : ℝ, P = Polynomial.C α * Polynomial.X^4 + Polynomial.C β * Polynomial.X^2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_condition_l2364_236445


namespace NUMINAMATH_GPT_solve_trig_problem_l2364_236419

open Real

theorem solve_trig_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := sorry

end NUMINAMATH_GPT_solve_trig_problem_l2364_236419


namespace NUMINAMATH_GPT_power_of_same_base_power_of_different_base_l2364_236432

theorem power_of_same_base (a n : ℕ) (h : ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m) :
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ a^n = (a^k)^m :=
  sorry

theorem power_of_different_base (a n : ℕ) : ∃ (b m : ℕ), a^n = b^m :=
  sorry

end NUMINAMATH_GPT_power_of_same_base_power_of_different_base_l2364_236432


namespace NUMINAMATH_GPT_eval_expr_l2364_236403

variable {x y : ℝ}

theorem eval_expr (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) - ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) = (2 * x^2) / (y^2) + (2 * y^2) / (x^2) := by
  sorry

end NUMINAMATH_GPT_eval_expr_l2364_236403


namespace NUMINAMATH_GPT_max_profit_l2364_236453

variables (x y : ℕ)

def steel_constraint := 10 * x + 70 * y ≤ 700
def non_ferrous_constraint := 23 * x + 40 * y ≤ 642
def non_negativity := x ≥ 0 ∧ y ≥ 0
def profit := 80 * x + 100 * y

theorem max_profit (h₁ : steel_constraint x y)
                   (h₂ : non_ferrous_constraint x y)
                   (h₃ : non_negativity x y):
  profit x y = 2180 := 
sorry

end NUMINAMATH_GPT_max_profit_l2364_236453


namespace NUMINAMATH_GPT_iterate_F_l2364_236413

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

theorem iterate_F (x : ℝ) : (Nat.iterate F 2017 x) = (x + 1)^(3^2017) - 1 :=
by
  sorry

end NUMINAMATH_GPT_iterate_F_l2364_236413


namespace NUMINAMATH_GPT_find_fraction_l2364_236468

variable (N : ℕ) (F : ℚ)
theorem find_fraction (h1 : N = 90) (h2 : 3 + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = F * N) : F = 1 / 15 :=
sorry

end NUMINAMATH_GPT_find_fraction_l2364_236468


namespace NUMINAMATH_GPT_area_enclosed_by_region_l2364_236485

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_area_enclosed_by_region_l2364_236485


namespace NUMINAMATH_GPT_max_sigma_squared_l2364_236484

theorem max_sigma_squared (c d : ℝ) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_c_ge_d : c ≥ d)
    (h : ∃ x y : ℝ, 0 ≤ x ∧ x < c ∧ 0 ≤ y ∧ y < d ∧ 
      c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x) ^ 2 + (d - y) ^ 2) : 
    σ^2 = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_max_sigma_squared_l2364_236484


namespace NUMINAMATH_GPT_butterfat_milk_mixing_l2364_236423

theorem butterfat_milk_mixing :
  ∀ (x : ℝ), 
  (0.35 * x + 0.10 * 12 = 0.20 * (x + 12)) → x = 8 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_butterfat_milk_mixing_l2364_236423


namespace NUMINAMATH_GPT_min_value_arithmetic_seq_l2364_236456

theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h_arith_seq : ∀ n, a n ≤ a (n + 1)) (h_pos : ∀ n, a n > 0) (h_cond : a 1 + a 2017 = 2) :
  ∃ (min_value : ℝ), min_value = 2 ∧ (∀ (x y : ℝ), x + y = 2 → x > 0 → y > 0 → x + y / (x * y) = 2) :=
  sorry

end NUMINAMATH_GPT_min_value_arithmetic_seq_l2364_236456


namespace NUMINAMATH_GPT_avg_height_first_30_girls_l2364_236408

theorem avg_height_first_30_girls (H : ℝ)
  (h1 : ∀ x : ℝ, 30 * x + 10 * 156 = 40 * 159) :
  H = 160 :=
by sorry

end NUMINAMATH_GPT_avg_height_first_30_girls_l2364_236408


namespace NUMINAMATH_GPT_exists_nat_numbers_satisfying_sum_l2364_236487

theorem exists_nat_numbers_satisfying_sum :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 :=
sorry

end NUMINAMATH_GPT_exists_nat_numbers_satisfying_sum_l2364_236487


namespace NUMINAMATH_GPT_negation_of_P_is_non_P_l2364_236435

open Real

/-- Proposition P: For any x in the real numbers, sin(x) <= 1 -/
def P : Prop := ∀ x : ℝ, sin x ≤ 1

/-- Negation of P: There exists x in the real numbers such that sin(x) >= 1 -/
def non_P : Prop := ∃ x : ℝ, sin x ≥ 1

theorem negation_of_P_is_non_P : ¬P ↔ non_P :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_P_is_non_P_l2364_236435


namespace NUMINAMATH_GPT_area_shaded_region_l2364_236451

theorem area_shaded_region :
  let r_s := 3   -- Radius of the smaller circle
  let r_l := 3 * r_s  -- Radius of the larger circle
  let A_l := π * r_l^2  -- Area of the larger circle
  let A_s := π * r_s^2  -- Area of the smaller circle
  A_l - A_s = 72 * π := 
by
  sorry

end NUMINAMATH_GPT_area_shaded_region_l2364_236451


namespace NUMINAMATH_GPT_number_of_hens_l2364_236434

theorem number_of_hens (H C G : ℕ) 
  (h1 : H + C + G = 120) 
  (h2 : 2 * H + 4 * C + 4 * G = 348) : 
  H = 66 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_hens_l2364_236434


namespace NUMINAMATH_GPT_smallest_integer_solution_l2364_236411

open Int

theorem smallest_integer_solution :
  ∃ x : ℤ, (⌊ (x : ℚ) / 8 ⌋ - ⌊ (x : ℚ) / 40 ⌋ + ⌊ (x : ℚ) / 240 ⌋ = 210) ∧ x = 2016 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_solution_l2364_236411


namespace NUMINAMATH_GPT_sum_of_angles_l2364_236499

theorem sum_of_angles 
    (ABC_isosceles : ∃ (A B C : Type) (angleBAC : ℝ), (AB = AC) ∧ (angleBAC = 25))
    (DEF_isosceles : ∃ (D E F : Type) (angleEDF : ℝ), (DE = DF) ∧ (angleEDF = 40)) 
    (AD_parallel_CE : Prop) : 
    ∃ (angleDAC angleADE : ℝ), angleDAC = 77.5 ∧ angleADE = 70 ∧ (angleDAC + angleADE = 147.5) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_angles_l2364_236499


namespace NUMINAMATH_GPT_m_minus_t_value_l2364_236495

-- Define the sum of squares of the odd integers from 1 to 215
def sum_squares_odds (n : ℕ) : ℕ := n * (4 * n^2 - 1) / 3

-- Define the sum of squares of the even integers from 2 to 100
def sum_squares_evens (n : ℕ) : ℕ := 2 * n * (n + 1) * (2 * n + 1) / 3

-- Number of odd terms from 1 to 215
def odd_terms_count : ℕ := (215 - 1) / 2 + 1

-- Number of even terms from 2 to 100
def even_terms_count : ℕ := (100 - 2) / 2 + 1

-- Define m and t
def m : ℕ := sum_squares_odds odd_terms_count
def t : ℕ := sum_squares_evens even_terms_count

-- Prove that m - t = 1507880
theorem m_minus_t_value : m - t = 1507880 :=
by
  -- calculations to verify the proof will be here, but are omitted for now
  sorry

end NUMINAMATH_GPT_m_minus_t_value_l2364_236495


namespace NUMINAMATH_GPT_mushrooms_collected_l2364_236404

theorem mushrooms_collected (x1 x2 x3 x4 : ℕ) 
  (h1 : x1 + x2 = 7) 
  (h2 : x1 + x3 = 9)
  (h3 : x2 + x3 = 10) : x1 = 3 ∧ x2 = 4 ∧ x3 = 6 ∧ x4 = 7 :=
by
  sorry

end NUMINAMATH_GPT_mushrooms_collected_l2364_236404


namespace NUMINAMATH_GPT_A_can_give_C_start_l2364_236425

def canGiveStart (total_distance start_A_B start_B_C start_A_C : ℝ) :=
  (total_distance - start_A_B) / total_distance * (total_distance - start_B_C) / total_distance = 
  (total_distance - start_A_C) / total_distance

theorem A_can_give_C_start :
  canGiveStart 1000 70 139.7849462365591 200 :=
by
  sorry

end NUMINAMATH_GPT_A_can_give_C_start_l2364_236425


namespace NUMINAMATH_GPT_question_d_l2364_236493

variable {x a : ℝ}

theorem question_d (h1 : x < a) (h2 : a < 0) : x^3 > a * x ∧ a * x < 0 :=
  sorry

end NUMINAMATH_GPT_question_d_l2364_236493


namespace NUMINAMATH_GPT_percentage_error_in_side_l2364_236444

theorem percentage_error_in_side {S S' : ℝ}
  (hs : S > 0)
  (hs' : S' > S)
  (h_area_error : (S'^2 - S^2) / S^2 * 100 = 90.44) :
  ((S' - S) / S * 100) = 38 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_in_side_l2364_236444


namespace NUMINAMATH_GPT_max_divisors_with_remainder_10_l2364_236466

theorem max_divisors_with_remainder_10 (m : ℕ) :
  (m > 0) → (∀ k, (2008 % k = 10) ↔ k < m) → m = 11 :=
by
  sorry

end NUMINAMATH_GPT_max_divisors_with_remainder_10_l2364_236466


namespace NUMINAMATH_GPT_find_m_l2364_236472

-- Define the points M and N and the normal vector n
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M (m : ℝ) : Point3D := { x := m, y := -2, z := 1 }
def N (m : ℝ) : Point3D := { x := 0, y := m, z := 3 }
def n : Point3D := { x := 3, y := 1, z := 2 }

-- Define the dot product
def dot_product (v1 v2 : Point3D) : ℝ :=
  (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

-- Define the vector MN
def MN (m : ℝ) : Point3D := { x := -(m), y := m + 2, z := 2 }

-- Prove the dot product condition is zero implies m = 3
theorem find_m (m : ℝ) (h : dot_product n (MN m) = 0) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2364_236472


namespace NUMINAMATH_GPT_percentage_less_than_l2364_236486

theorem percentage_less_than (x y : ℝ) (h : x = 8 * y) : ((x - y) / x) * 100 = 87.5 := 
by sorry

end NUMINAMATH_GPT_percentage_less_than_l2364_236486


namespace NUMINAMATH_GPT_positive_real_solutions_unique_l2364_236477

theorem positive_real_solutions_unique (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
(h : (a^2 - b * d) / (b + 2 * c + d) + (b^2 - c * a) / (c + 2 * d + a) + (c^2 - d * b) / (d + 2 * a + b) + (d^2 - a * c) / (a + 2 * b + c) = 0) : 
a = b ∧ b = c ∧ c = d :=
sorry

end NUMINAMATH_GPT_positive_real_solutions_unique_l2364_236477


namespace NUMINAMATH_GPT_minimum_work_to_remove_cube_l2364_236440

namespace CubeBuoyancy

def edge_length (ℓ : ℝ) := ℓ = 0.30 -- in meters
def wood_density (ρ : ℝ) := ρ = 750  -- in kg/m^3
def water_density (ρ₀ : ℝ) := ρ₀ = 1000 -- in kg/m^3

theorem minimum_work_to_remove_cube 
  {ℓ ρ ρ₀ : ℝ} 
  (h₁ : edge_length ℓ)
  (h₂ : wood_density ρ)
  (h₃ : water_density ρ₀) : 
  ∃ W : ℝ, W = 22.8 := 
sorry

end CubeBuoyancy

end NUMINAMATH_GPT_minimum_work_to_remove_cube_l2364_236440


namespace NUMINAMATH_GPT_net_change_is_12_l2364_236449

-- Definitions based on the conditions of the problem

def initial_investment : ℝ := 100
def first_year_increase_percentage : ℝ := 0.60
def second_year_decrease_percentage : ℝ := 0.30

-- Calculate the wealth at the end of the first year
def end_of_first_year_wealth : ℝ := initial_investment * (1 + first_year_increase_percentage)

-- Calculate the wealth at the end of the second year
def end_of_second_year_wealth : ℝ := end_of_first_year_wealth * (1 - second_year_decrease_percentage)

-- Calculate the net change
def net_change : ℝ := end_of_second_year_wealth - initial_investment

-- The target theorem to prove
theorem net_change_is_12 : net_change = 12 := by
  sorry

end NUMINAMATH_GPT_net_change_is_12_l2364_236449


namespace NUMINAMATH_GPT_sarah_class_choices_l2364_236418

-- Conditions 
def total_classes : ℕ := 10
def choose_classes : ℕ := 4
def specific_classes : ℕ := 2

-- Statement
theorem sarah_class_choices : 
  ∃ (n : ℕ), n = Nat.choose (total_classes - specific_classes) 3 ∧ n = 56 :=
by 
  sorry

end NUMINAMATH_GPT_sarah_class_choices_l2364_236418


namespace NUMINAMATH_GPT_single_rooms_booked_l2364_236439

noncomputable def hotel_problem (S D : ℕ) : Prop :=
  S + D = 260 ∧ 35 * S + 60 * D = 14000

theorem single_rooms_booked (S D : ℕ) (h : hotel_problem S D) : S = 64 :=
by
  sorry

end NUMINAMATH_GPT_single_rooms_booked_l2364_236439


namespace NUMINAMATH_GPT_common_remainder_proof_l2364_236424

def least_subtracted := 6
def original_number := 1439
def reduced_number := original_number - least_subtracted
def divisors := [5, 11, 13]
def common_remainder := 3

theorem common_remainder_proof :
  ∀ d ∈ divisors, reduced_number % d = common_remainder := by
  sorry

end NUMINAMATH_GPT_common_remainder_proof_l2364_236424


namespace NUMINAMATH_GPT_inequality_abc_equality_condition_l2364_236481

theorem inequality_abc (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) ≥ 12 :=
sorry

theorem equality_condition (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end NUMINAMATH_GPT_inequality_abc_equality_condition_l2364_236481


namespace NUMINAMATH_GPT_bowling_ball_weight_l2364_236492

theorem bowling_ball_weight (b k : ℝ) (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l2364_236492


namespace NUMINAMATH_GPT_odd_function_properties_l2364_236471

def f : ℝ → ℝ := sorry

theorem odd_function_properties 
  (H1 : ∀ x, f (-x) = -f x) -- f is odd
  (H2 : ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → f x ≤ f y) -- f is increasing on [1, 3]
  (H3 : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≥ 7) -- f has a minimum value of 7 on [1, 3]
  : (∀ x y, -3 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) -- f is increasing on [-3, -1]
    ∧ (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) -- f has a maximum value of -7 on [-3, -1]
:= sorry

end NUMINAMATH_GPT_odd_function_properties_l2364_236471


namespace NUMINAMATH_GPT_num_of_valid_m_vals_l2364_236448

theorem num_of_valid_m_vals : 
  (∀ m x : ℤ, (x + m ≤ 4 ∧ (x / 2 - (x - 1) / 4 > 1 → x > 3 → ∃ (c : ℚ), (x + 1)/4 > 1 )) ∧
  (∃ (x : ℤ), (x + m ≤ 4 ∧ (x > 3) ∧ (m < 1 ∧ m > -4)) ∧ 
  ∃ a b : ℚ, x^2 + a * x + b = 0) → 
  (∃ (count m : ℤ), count = 2)) :=
sorry

end NUMINAMATH_GPT_num_of_valid_m_vals_l2364_236448


namespace NUMINAMATH_GPT_election_winner_won_by_votes_l2364_236406

theorem election_winner_won_by_votes (V : ℝ) (winner_votes : ℝ) (loser_votes : ℝ)
    (h1 : winner_votes = 0.62 * V)
    (h2 : winner_votes = 930)
    (h3 : loser_votes = 0.38 * V)
    : winner_votes - loser_votes = 360 := 
  sorry

end NUMINAMATH_GPT_election_winner_won_by_votes_l2364_236406


namespace NUMINAMATH_GPT_solve_for_x_l2364_236442

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 8) * x = 12) : x = 168 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2364_236442


namespace NUMINAMATH_GPT_journey_time_difference_l2364_236464

theorem journey_time_difference :
  let t1 := (100:ℝ) / 60
  let t2 := (400:ℝ) / 40
  let T1 := t1 + t2
  let T2 := (500:ℝ) / 50
  let difference := (T1 - T2) * 60
  abs (difference - 100) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_journey_time_difference_l2364_236464


namespace NUMINAMATH_GPT_room_area_in_square_meters_l2364_236414

theorem room_area_in_square_meters :
  ∀ (length_ft width_ft : ℝ), 
  (length_ft = 15) → 
  (width_ft = 8) → 
  (1 / 9 * 0.836127 = 0.092903) → 
  (length_ft * width_ft * 0.092903 = 11.14836) :=
by
  intros length_ft width_ft h_length h_width h_conversion
  -- sorry to skip the proof steps.
  sorry

end NUMINAMATH_GPT_room_area_in_square_meters_l2364_236414


namespace NUMINAMATH_GPT_problem_statement_l2364_236420

theorem problem_statement : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2364_236420


namespace NUMINAMATH_GPT_Anton_thought_of_729_l2364_236459

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end NUMINAMATH_GPT_Anton_thought_of_729_l2364_236459


namespace NUMINAMATH_GPT_rate_per_meter_l2364_236436

theorem rate_per_meter (d : ℝ) (total_cost : ℝ) (rate_per_meter : ℝ) (h_d : d = 30)
    (h_total_cost : total_cost = 188.49555921538757) :
    rate_per_meter = 2 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_meter_l2364_236436


namespace NUMINAMATH_GPT_part_a_l2364_236463

theorem part_a (x y : ℝ) : x^2 - 2*y^2 = -((x + 2*y)^2 - 2*(x + y)^2) :=
sorry

end NUMINAMATH_GPT_part_a_l2364_236463


namespace NUMINAMATH_GPT_relationship_between_y_values_l2364_236483

-- Define the quadratic function given the constraints
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + abs b * x + c

-- Define the points (x1, y1), (x2, y2), (x3, y3)
def x1 := -14 / 3
def x2 := 5 / 2
def x3 := 3

def y1 (a b c : ℝ) : ℝ := quadratic_function a b c x1
def y2 (a b c : ℝ) : ℝ := quadratic_function a b c x2
def y3 (a b c : ℝ) : ℝ := quadratic_function a b c x3

theorem relationship_between_y_values 
  (a b c : ℝ) (h1 : - (abs b) / (2 * a) = -1) 
  (y1_value : ℝ := y1 a b c) 
  (y2_value : ℝ := y2 a b c) 
  (y3_value : ℝ := y3 a b c) : 
  y2_value < y1_value ∧ y1_value < y3_value := 
by 
  sorry

end NUMINAMATH_GPT_relationship_between_y_values_l2364_236483


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l2364_236452

variable (k x y1 y2 y3 : ℝ)
variable (h1 : k < 0)
variable (h2 : y1 = k / -4)
variable (h3 : y2 = k / 2)
variable (h4 : y3 = k / 3)

theorem relationship_y1_y2_y3 (k x y1 y2 y3 : ℝ) 
  (h1 : k < 0)
  (h2 : y1 = k / -4)
  (h3 : y2 = k / 2)
  (h4 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l2364_236452


namespace NUMINAMATH_GPT_smallest_value_a_b_l2364_236401

theorem smallest_value_a_b (a b : ℕ) (h : 2^6 * 3^9 = a^b) : a > 0 ∧ b > 0 ∧ (a + b = 111) :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_a_b_l2364_236401


namespace NUMINAMATH_GPT_Teresa_current_age_l2364_236443

-- Definitions of the conditions
def Morio_current_age := 71
def Morio_age_when_Michiko_born := 38
def Teresa_age_when_Michiko_born := 26

-- Definition of Michiko's current age
def Michiko_current_age := Morio_current_age - Morio_age_when_Michiko_born

-- The Theorem statement
theorem Teresa_current_age : Teresa_age_when_Michiko_born + Michiko_current_age = 59 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_Teresa_current_age_l2364_236443


namespace NUMINAMATH_GPT_prism_coloring_1995_prism_coloring_1996_l2364_236482

def prism_coloring_possible (n : ℕ) : Prop :=
  ∃ (color : ℕ → ℕ),
    (∀ i, 1 ≤ color i ∧ color i ≤ 3) ∧ -- Each color is within bounds
    (∀ i, color i ≠ color ((i + 1) % n)) ∧ -- Colors on each face must be different
    (n % 3 = 0 ∨ n ≠ 1996) -- Condition for coloring

theorem prism_coloring_1995 : prism_coloring_possible 1995 :=
sorry

theorem prism_coloring_1996 : ¬prism_coloring_possible 1996 :=
sorry

end NUMINAMATH_GPT_prism_coloring_1995_prism_coloring_1996_l2364_236482


namespace NUMINAMATH_GPT_Carlos_earnings_l2364_236473

theorem Carlos_earnings :
  ∃ (wage : ℝ), 
  (18 * wage) = (12 * wage + 36) ∧ 
  wage = 36 / 6 ∧ 
  (12 * wage + 18 * wage) = 180 :=
by
  sorry

end NUMINAMATH_GPT_Carlos_earnings_l2364_236473


namespace NUMINAMATH_GPT_M_identically_zero_l2364_236416

noncomputable def M (x y : ℝ) : ℝ := sorry

theorem M_identically_zero (a : ℝ) (h1 : a > 1) (h2 : ∀ x, M x (a^x) = 0) : ∀ x y, M x y = 0 :=
sorry

end NUMINAMATH_GPT_M_identically_zero_l2364_236416


namespace NUMINAMATH_GPT_totalKidsInLawrenceCounty_l2364_236410

-- Constants representing the number of kids in each category
def kidsGoToCamp : ℕ := 629424
def kidsStayHome : ℕ := 268627

-- Statement of the total number of kids in Lawrence county
theorem totalKidsInLawrenceCounty : kidsGoToCamp + kidsStayHome = 898051 := by
  sorry

end NUMINAMATH_GPT_totalKidsInLawrenceCounty_l2364_236410


namespace NUMINAMATH_GPT_find_fg_of_3_l2364_236426

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 4 * x - 5

theorem find_fg_of_3 : f (g 3) = 31 := by
  sorry

end NUMINAMATH_GPT_find_fg_of_3_l2364_236426


namespace NUMINAMATH_GPT_salary_increase_after_five_years_l2364_236400

theorem salary_increase_after_five_years (S : ℝ) : 
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  percent_increase = 76.23 :=
by
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  sorry

end NUMINAMATH_GPT_salary_increase_after_five_years_l2364_236400


namespace NUMINAMATH_GPT_next_volunteer_day_l2364_236498

-- Definitions based on conditions.
def Alison_schedule := 5
def Ben_schedule := 3
def Carla_schedule := 9
def Dave_schedule := 8

-- Main theorem
theorem next_volunteer_day : Nat.lcm Alison_schedule (Nat.lcm Ben_schedule (Nat.lcm Carla_schedule Dave_schedule)) = 360 := by
  sorry

end NUMINAMATH_GPT_next_volunteer_day_l2364_236498


namespace NUMINAMATH_GPT_product_of_fractions_l2364_236407

theorem product_of_fractions :
  (2 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) * (5 / 6 : ℚ) * (6 / 7 : ℚ) * (7 / 8 : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l2364_236407


namespace NUMINAMATH_GPT_magic_shop_purchase_l2364_236476

theorem magic_shop_purchase :
  let deck_price := 7
  let frank_decks := 3
  let friend_decks := 2
  let discount_rate := 0.1
  let tax_rate := 0.05
  let total_cost := (frank_decks + friend_decks) * deck_price
  let discount := discount_rate * total_cost
  let discounted_total := total_cost - discount
  let sales_tax := tax_rate * discounted_total
  let rounded_sales_tax := (sales_tax * 100).round / 100
  let final_amount := discounted_total + rounded_sales_tax
  final_amount = 33.08 :=
by
  sorry

end NUMINAMATH_GPT_magic_shop_purchase_l2364_236476


namespace NUMINAMATH_GPT_find_S20_l2364_236402

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry

axiom a_nonzero (n : ℕ) : a_seq n ≠ 0
axiom a1_eq : a_seq 1 = 1
axiom Sn_eq (n : ℕ) : S n = (a_seq n * a_seq (n + 1)) / 2

theorem find_S20 : S 20 = 210 := sorry

end NUMINAMATH_GPT_find_S20_l2364_236402


namespace NUMINAMATH_GPT_simplify_exponent_expression_l2364_236460

theorem simplify_exponent_expression (n : ℕ) :
  (3^(n+4) - 3 * 3^n) / (3 * 3^(n+3)) = 26 / 9 := by
  sorry

end NUMINAMATH_GPT_simplify_exponent_expression_l2364_236460


namespace NUMINAMATH_GPT_f_at_neg_one_l2364_236465

def f : ℝ → ℝ := sorry

theorem f_at_neg_one :
  (∀ x : ℝ, f (x / (1 + x)) = x) →
  f (-1) = -1 / 2 :=
by
  intro h
  -- proof omitted for clarity
  sorry

end NUMINAMATH_GPT_f_at_neg_one_l2364_236465


namespace NUMINAMATH_GPT_solution_set_for_inequality_l2364_236489

theorem solution_set_for_inequality :
  {x : ℝ | (1 / (x - 1) ≥ -1)} = {x : ℝ | x ≤ 0 ∨ x > 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l2364_236489


namespace NUMINAMATH_GPT_decrease_in_B_share_l2364_236421

theorem decrease_in_B_share (a b c : ℝ) (x : ℝ) 
  (h1 : c = 495)
  (h2 : a + b + c = 1010)
  (h3 : (a - 25) / 3 = (b - x) / 2)
  (h4 : (a - 25) / 3 = (c - 15) / 5) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_decrease_in_B_share_l2364_236421


namespace NUMINAMATH_GPT_simplify_expression_l2364_236457

variable (x : ℝ)

theorem simplify_expression : 2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2364_236457


namespace NUMINAMATH_GPT_sally_earnings_proof_l2364_236470

def sally_last_month_earnings : ℝ := 1000
def raise_percentage : ℝ := 0.10
def sally_this_month_earnings := sally_last_month_earnings * (1 + raise_percentage)
def sally_total_two_months_earnings := sally_last_month_earnings + sally_this_month_earnings

theorem sally_earnings_proof :
  sally_total_two_months_earnings = 2100 :=
by
  sorry

end NUMINAMATH_GPT_sally_earnings_proof_l2364_236470


namespace NUMINAMATH_GPT_emily_annual_income_l2364_236450

variables {q I : ℝ}

theorem emily_annual_income (h1 : (0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) = ((q + 0.75) * 0.01 * I)) : 
  I = 40000 := 
by
  sorry

end NUMINAMATH_GPT_emily_annual_income_l2364_236450


namespace NUMINAMATH_GPT_domain_g_l2364_236441

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 + 14 * x - 3)

theorem domain_g :
  {x : ℝ | -8 * x^2 + 14 * x - 3 ≥ 0} = { x : ℝ | x ≤ 1 / 4 ∨ x ≥ 3 / 2 } :=
by
  sorry

end NUMINAMATH_GPT_domain_g_l2364_236441
