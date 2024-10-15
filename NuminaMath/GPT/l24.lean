import Mathlib

namespace NUMINAMATH_GPT_sector_central_angle_l24_2438

-- Definitions and constants
def arc_length := 4 -- arc length of the sector in cm
def area := 2       -- area of the sector in cm²

-- The central angle of the sector we want to prove
def theta := 4      -- radian measure of the central angle

-- Main statement to prove
theorem sector_central_angle : 
  ∃ (r : ℝ), (1 / 2) * theta * r^2 = area ∧ theta * r = arc_length :=
by
  -- No proof is required as per the instruction
  sorry

end NUMINAMATH_GPT_sector_central_angle_l24_2438


namespace NUMINAMATH_GPT_find_number_l24_2413

theorem find_number 
  (n : ℤ)
  (h1 : n % 7 = 2)
  (h2 : n % 8 = 4)
  (quot_7 : ℤ)
  (quot_8 : ℤ)
  (h3 : n = 7 * quot_7 + 2)
  (h4 : n = 8 * quot_8 + 4)
  (h5 : quot_7 = quot_8 + 7) :
  n = 380 := by
  sorry

end NUMINAMATH_GPT_find_number_l24_2413


namespace NUMINAMATH_GPT_square_plot_area_l24_2486

theorem square_plot_area (s : ℕ) 
  (cost_per_foot : ℕ) 
  (total_cost : ℕ) 
  (H1 : cost_per_foot = 58) 
  (H2 : total_cost = 1624) 
  (H3 : total_cost = 232 * s) : 
  s * s = 49 := 
  by sorry

end NUMINAMATH_GPT_square_plot_area_l24_2486


namespace NUMINAMATH_GPT_bamboo_volume_l24_2434

theorem bamboo_volume :
  ∃ (a₁ d a₅ : ℚ), 
  (4 * a₁ + 6 * d = 5) ∧ 
  (3 * a₁ + 21 * d = 4) ∧ 
  (a₅ = a₁ + 4 * d) ∧ 
  (a₅ = 85 / 66) :=
sorry

end NUMINAMATH_GPT_bamboo_volume_l24_2434


namespace NUMINAMATH_GPT_books_for_sale_l24_2483

theorem books_for_sale (initial_books found_books : ℕ) (h1 : initial_books = 33) (h2 : found_books = 26) :
  initial_books + found_books = 59 :=
by
  sorry

end NUMINAMATH_GPT_books_for_sale_l24_2483


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l24_2405

theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), x^2 = 8 * y → ∃ F : ℝ × ℝ, F = (0, 2) :=
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l24_2405


namespace NUMINAMATH_GPT_find_length_of_rectangular_playground_l24_2437

def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

theorem find_length_of_rectangular_playground (P B : ℕ) (hP : P = 1200) (hB : B = 500) : ∃ L, perimeter L B = P ∧ L = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_rectangular_playground_l24_2437


namespace NUMINAMATH_GPT_speed_W_B_l24_2493

-- Definitions for the conditions
def distance_W_B (D : ℝ) := 2 * D
def average_speed := 36
def speed_B_C := 20

-- The problem statement to be verified in Lean
theorem speed_W_B (D : ℝ) (S : ℝ) (h1: distance_W_B D = 2 * D) (h2: S ≠ 0 ∧ D ≠ 0)
(h3: (3 * D) / ((2 * D) / S + D / speed_B_C) = average_speed) : S = 60 := by
sorry

end NUMINAMATH_GPT_speed_W_B_l24_2493


namespace NUMINAMATH_GPT_sin_cos_sum_l24_2475

theorem sin_cos_sum (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (h : Real.tan (θ + Real.pi / 4) = 1 / 7) : Real.sin θ + Real.cos θ = -1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_sum_l24_2475


namespace NUMINAMATH_GPT_probability_two_cards_diff_suits_l24_2419

def prob_two_cards_diff_suits {deck_size suits cards_per_suit : ℕ} (h1 : deck_size = 40) (h2 : suits = 4) (h3 : cards_per_suit = 10) : ℚ :=
  let total_cards := deck_size
  let cards_same_suit := cards_per_suit - 1
  let cards_diff_suit := total_cards - 1 - cards_same_suit 
  cards_diff_suit / (total_cards - 1)

theorem probability_two_cards_diff_suits (h1 : 40 = 40) (h2 : 4 = 4) (h3 : 10 = 10) :
  prob_two_cards_diff_suits h1 h2 h3 = 10 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_cards_diff_suits_l24_2419


namespace NUMINAMATH_GPT_color_theorem_l24_2426

/-- The only integers \( k \geq 1 \) such that if each integer is colored in one of these \( k \)
colors, there must exist integers \( a_1 < a_2 < \cdots < a_{2023} \) of the same color where the
differences \( a_2 - a_1, a_3 - a_2, \cdots, a_{2023} - a_{2022} \) are all powers of 2 are
\( k = 1 \) and \( k = 2 \). -/
theorem color_theorem : ∀ (k : ℕ), (k ≥ 1) →
  (∀ f : ℕ → Fin k,
    ∃ a : Fin 2023 → ℕ,
    (∀ i : Fin (2023 - 1), ∃ n : ℕ, 2^n = (a i.succ - a i)) ∧
    (∀ i j : Fin 2023, i < j → f (a i) = f (a j)))
  ↔ k = 1 ∨ k = 2 := by
  sorry

end NUMINAMATH_GPT_color_theorem_l24_2426


namespace NUMINAMATH_GPT_min_rice_proof_l24_2465

noncomputable def minRicePounds : ℕ := 2

theorem min_rice_proof (o r : ℕ) (h1 : o ≥ 8 + 3 * r / 4) (h2 : o ≤ 5 * r) :
  r ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_min_rice_proof_l24_2465


namespace NUMINAMATH_GPT_polynomial_solutions_l24_2406

theorem polynomial_solutions :
  (∀ x : ℂ, (x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = 0) ↔ (x = -1 ∨ x = Complex.I ∨ x = -Complex.I)) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solutions_l24_2406


namespace NUMINAMATH_GPT_values_of_x_l24_2461

def P (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x

theorem values_of_x (x : ℝ) :
  P x = P (x + 1) ↔ (x = 1 ∨ x = 4 / 3) :=
by sorry

end NUMINAMATH_GPT_values_of_x_l24_2461


namespace NUMINAMATH_GPT_watermelons_left_l24_2428

theorem watermelons_left (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (h1 : initial = 4) (h2 : eaten = 3) : remaining = 1 :=
by
  sorry

end NUMINAMATH_GPT_watermelons_left_l24_2428


namespace NUMINAMATH_GPT_factor_100_minus_16y2_l24_2425

theorem factor_100_minus_16y2 (y : ℝ) : 100 - 16 * y^2 = 4 * (5 - 2 * y) * (5 + 2 * y) := 
by sorry

end NUMINAMATH_GPT_factor_100_minus_16y2_l24_2425


namespace NUMINAMATH_GPT_polynomial_root_multiplicity_l24_2440

theorem polynomial_root_multiplicity (A B n : ℤ) (h1 : A + B + 1 = 0) (h2 : (n + 1) * A + n * B = 0) :
  A = n ∧ B = -(n + 1) :=
sorry

end NUMINAMATH_GPT_polynomial_root_multiplicity_l24_2440


namespace NUMINAMATH_GPT_identity_solution_l24_2433

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end NUMINAMATH_GPT_identity_solution_l24_2433


namespace NUMINAMATH_GPT_complement_A_possible_set_l24_2477

variable (U A B : Set ℕ)

theorem complement_A_possible_set (hU : U = {1, 2, 3, 4, 5, 6})
  (h_union : A ∪ B = {1, 2, 3, 4, 5}) 
  (h_inter : A ∩ B = {3, 4, 5}) :
  ∃ C, C = U \ A ∧ C = {6} :=
by
  sorry

end NUMINAMATH_GPT_complement_A_possible_set_l24_2477


namespace NUMINAMATH_GPT_extreme_points_l24_2427

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 / (2 * x)) - a * x^2 + x

theorem extreme_points (
  a : ℝ
) (h : 0 < a ∧ a < (1 : ℝ) / 8) :
  ∃ x1 x2 : ℝ, f a x1 + f a x2 > 3 - 4 * Real.log 2 :=
sorry

end NUMINAMATH_GPT_extreme_points_l24_2427


namespace NUMINAMATH_GPT_solve_system_l24_2416

theorem solve_system :
  ∃ a b c d e : ℤ, 
    (a * b + a + 2 * b = 78) ∧
    (b * c + 3 * b + c = 101) ∧
    (c * d + 5 * c + 3 * d = 232) ∧
    (d * e + 4 * d + 5 * e = 360) ∧
    (e * a + 2 * e + 4 * a = 192) ∧
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨ (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l24_2416


namespace NUMINAMATH_GPT_problem1_problem2_l24_2478

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (x : ℝ) (h : ∀ x, f x = abs (x - 1)) :
  f x ≥ (1/2) * (x + 1) ↔ (x ≤ 1/3) ∨ (x ≥ 3) :=
sorry

-- Problem 2
theorem problem2 (g : ℝ → ℝ) (A : Set ℝ) (a : ℝ) 
  (h1 : ∀ x, g x = abs (x - a) - abs (x - 2))
  (h2 : A ⊆ Set.Icc (-1 : ℝ) 3) :
  (1 ≤ a ∧ a < 2) ∨ (2 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l24_2478


namespace NUMINAMATH_GPT_number_of_arrangements_l24_2448

def basil_plants := 2
def aloe_plants := 1
def cactus_plants := 1
def white_lamps := 2
def red_lamps := 2
def total_plants := basil_plants + aloe_plants + cactus_plants
def total_lamps := white_lamps + red_lamps

theorem number_of_arrangements : total_plants = 4 ∧ total_lamps = 4 →
  ∃ n : ℕ, n = 28 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l24_2448


namespace NUMINAMATH_GPT_boys_in_school_l24_2459

theorem boys_in_school (x : ℕ) (boys girls : ℕ) (h1 : boys = 5 * x) 
  (h2 : girls = 13 * x) (h3 : girls - boys = 128) : boys = 80 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_school_l24_2459


namespace NUMINAMATH_GPT_number_of_intersections_of_lines_l24_2494

theorem number_of_intersections_of_lines : 
  let L1 := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 12}
  let L2 := {p : ℝ × ℝ | 5 * p.1 - 2 * p.2 = 10}
  let L3 := {p : ℝ × ℝ | p.1 = 3}
  let L4 := {p : ℝ × ℝ | p.2 = 1}
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ p1 ∈ L1 ∧ p1 ∈ L2 ∧ p2 ∈ L3 ∧ p2 ∈ L4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_intersections_of_lines_l24_2494


namespace NUMINAMATH_GPT_aerith_is_correct_l24_2488

theorem aerith_is_correct :
  ∀ x : ℝ, x = 1.4 → (x ^ (x ^ x)) < 2 → ∃ y : ℝ, y = x ^ (x ^ x) :=
by
  sorry

end NUMINAMATH_GPT_aerith_is_correct_l24_2488


namespace NUMINAMATH_GPT_clothing_order_equation_l24_2492

open Real

-- Definitions and conditions
def total_pieces : ℕ := 720
def initial_rate : ℕ := 48
def days_earlier : ℕ := 5

-- Statement that we need to prove
theorem clothing_order_equation (x : ℕ) :
    (720 / 48 : ℝ) - (720 / (x + 48) : ℝ) = 5 := 
sorry

end NUMINAMATH_GPT_clothing_order_equation_l24_2492


namespace NUMINAMATH_GPT_quadratic_equation_original_eq_l24_2400

theorem quadratic_equation_original_eq :
  ∃ (α β : ℝ), (α + β = 3) ∧ (α * β = -6) ∧ (∀ (x : ℝ), x^2 - 3 * x - 6 = 0 → (x = α ∨ x = β)) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_original_eq_l24_2400


namespace NUMINAMATH_GPT_elena_alex_total_dollars_l24_2402

theorem elena_alex_total_dollars :
  (5 / 6 : ℚ) + (7 / 15 : ℚ) = (13 / 10 : ℚ) :=
by
    sorry

end NUMINAMATH_GPT_elena_alex_total_dollars_l24_2402


namespace NUMINAMATH_GPT_rectangle_perimeter_eq_26_l24_2481

theorem rectangle_perimeter_eq_26 (a b c W : ℕ) (h_tri : a = 5 ∧ b = 12 ∧ c = 13)
  (h_right_tri : a^2 + b^2 = c^2) (h_W : W = 3) (h_area_eq : 1/2 * (a * b) = (W * L))
  (A L : ℕ) (hA : A = 30) (hL : L = A / W) :
  2 * (L + W) = 26 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_eq_26_l24_2481


namespace NUMINAMATH_GPT_miles_hiked_first_day_l24_2430

theorem miles_hiked_first_day (total_distance remaining_distance : ℕ)
  (h1 : total_distance = 36)
  (h2 : remaining_distance = 27) :
  total_distance - remaining_distance = 9 :=
by
  sorry

end NUMINAMATH_GPT_miles_hiked_first_day_l24_2430


namespace NUMINAMATH_GPT_ratio_of_second_to_first_l24_2414

theorem ratio_of_second_to_first (A1 A2 A3 : ℕ) (h1 : A1 = 600) (h2 : A3 = A1 + A2 - 400) (h3 : A1 + A2 + A3 = 3200) : A2 / A1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_second_to_first_l24_2414


namespace NUMINAMATH_GPT_product_prices_determined_max_product_A_pieces_l24_2403

theorem product_prices_determined (a b : ℕ) :
  (20 * a + 15 * b = 380) →
  (15 * a + 10 * b = 280) →
  a = 16 ∧ b = 4 :=
by sorry

theorem max_product_A_pieces (x : ℕ) :
  (16 * x + 4 * (100 - x) ≤ 900) →
  x ≤ 41 :=
by sorry

end NUMINAMATH_GPT_product_prices_determined_max_product_A_pieces_l24_2403


namespace NUMINAMATH_GPT_clock_angle_solution_l24_2412

theorem clock_angle_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 360) :
    (θ = 15) ∨ (θ = 165) :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_solution_l24_2412


namespace NUMINAMATH_GPT_Mr_Mayer_purchase_price_l24_2479

theorem Mr_Mayer_purchase_price 
  (P : ℝ) 
  (H1 : (1.30 * 2) * P = 2600) : 
  P = 1000 := 
by
  sorry

end NUMINAMATH_GPT_Mr_Mayer_purchase_price_l24_2479


namespace NUMINAMATH_GPT_price_first_oil_l24_2464

theorem price_first_oil (P : ℝ) (h1 : 10 * P + 5 * 66 = 15 * 58.67) : P = 55.005 :=
sorry

end NUMINAMATH_GPT_price_first_oil_l24_2464


namespace NUMINAMATH_GPT_interest_rate_difference_l24_2495

-- Definitions for given conditions
def principal : ℝ := 3000
def time : ℝ := 9
def additional_interest : ℝ := 1350

-- The Lean 4 statement for the equivalence
theorem interest_rate_difference 
  (R H : ℝ) 
  (h_interest_formula_original : principal * R * time / 100 = principal * R * time / 100) 
  (h_interest_formula_higher : principal * H * time / 100 = principal * R * time / 100 + additional_interest) 
  : (H - R) = 5 :=
sorry

end NUMINAMATH_GPT_interest_rate_difference_l24_2495


namespace NUMINAMATH_GPT_smallest_even_sum_l24_2489

theorem smallest_even_sum :
  ∃ (a b c : Int), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ b ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ c ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ (a + b + c) % 2 = 0 ∧ (a + b + c) = 14 := sorry

end NUMINAMATH_GPT_smallest_even_sum_l24_2489


namespace NUMINAMATH_GPT_find_a_of_perpendicular_lines_l24_2445

theorem find_a_of_perpendicular_lines (a : ℝ) :
  let line1 : ℝ := a * x + y - 1
  let line2 : ℝ := 4 * x + (a - 3) * y - 2
  (∀ x y : ℝ, (line1 = 0 → line2 ≠ 0 → line1 * line2 = -1)) → a = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_perpendicular_lines_l24_2445


namespace NUMINAMATH_GPT_hyperbola_condition_l24_2453

theorem hyperbola_condition (m n : ℝ) : 
  (mn < 0) ↔ (∀ x y : ℝ, ∃ k ∈ {a : ℝ | a ≠ 0}, (x^2 / m + y^2 / n = 1)) := sorry

end NUMINAMATH_GPT_hyperbola_condition_l24_2453


namespace NUMINAMATH_GPT_directrix_of_parabola_l24_2408

theorem directrix_of_parabola (p : ℝ) (hp : 0 < p) (h_point : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2)) :
  x = -1/2 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l24_2408


namespace NUMINAMATH_GPT_zero_in_interval_l24_2447

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ x0, f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
by
  have h_cont : Continuous f := sorry -- f is continuous (can be proven using the continuity of log and linear functions)
  have h_eval1 : f 2 < 0 := sorry -- f(2) = ln(2) - 6 + 4 < 0
  have h_eval2 : f 3 > 0 := sorry -- f(3) = ln(3) - 6 + 6 > 0
  -- By the Intermediate Value Theorem, since f is continuous and changes signs between (2, 3), there exists a zero x0 in (2, 3).
  exact sorry

end NUMINAMATH_GPT_zero_in_interval_l24_2447


namespace NUMINAMATH_GPT_estimate_expr_range_l24_2423

theorem estimate_expr_range :
  5 < (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) ∧
  (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) < 6 :=
  sorry

end NUMINAMATH_GPT_estimate_expr_range_l24_2423


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l24_2458

variable (A B C : Set α) (a : α)
variable [Nonempty α]
variable (H1 : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C))

theorem necessary_but_not_sufficient_condition :
  (a ∈ B → a ∈ A) ∧ ¬(a ∈ A → a ∈ B) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l24_2458


namespace NUMINAMATH_GPT_inequality_xyz_equality_condition_l24_2451

theorem inequality_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ 2 + x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  (x + y + z = 2 + x * y * z) ↔ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨
                                                  (x = 0 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 0 ∧ z = 1) ∨
                                                  (x = -1 ∧ y = 1 ∧ z = 0) :=
sorry

end NUMINAMATH_GPT_inequality_xyz_equality_condition_l24_2451


namespace NUMINAMATH_GPT_part_I_part_II_l24_2446

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2 * x|

-- Part I
theorem part_I : ∀ x : ℝ, g x > -4 → -5 < x ∧ x < -3 :=
by
  sorry

-- Part II
theorem part_II : 
  (∃ x1 x2 : ℝ, f x1 a = g x2) → -6 ≤ a ∧ a ≤ -4 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l24_2446


namespace NUMINAMATH_GPT_lines_intersect_at_l24_2436

theorem lines_intersect_at :
  ∃ t u : ℝ, (∃ (x y : ℝ),
    (x = 2 + 3 * t ∧ y = 4 - 2 * t) ∧
    (x = -1 + 6 * u ∧ y = 5 + u) ∧
    (x = 1/5 ∧ y = 26/5)) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_l24_2436


namespace NUMINAMATH_GPT_ab_value_in_triangle_l24_2429

theorem ab_value_in_triangle (a b c : ℝ) (C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_ab_value_in_triangle_l24_2429


namespace NUMINAMATH_GPT_hcf_of_three_numbers_l24_2497

def hcf (a b : ℕ) : ℕ := gcd a b

theorem hcf_of_three_numbers :
  let a := 136
  let b := 144
  let c := 168
  hcf (hcf a b) c = 8 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_three_numbers_l24_2497


namespace NUMINAMATH_GPT_correct_statements_l24_2417

theorem correct_statements (f : ℝ → ℝ)
  (h_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_pos : ∀ x : ℝ, x > 0 → f (x) > 0) :
  (f 0 ≠ 1) ∧
  (∀ x : ℝ, f (-x) = -f (x)) ∧
  ¬ (∀ x : ℝ, |f (x)| = |f (-x)|) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f (x₁) < f (x₂)) ∧
  ¬ (∀ x : ℝ, f (x) + 1 < f (x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l24_2417


namespace NUMINAMATH_GPT_total_cost_price_is_correct_l24_2415

noncomputable def selling_price_before_discount (sp_after_discount : ℝ) (discount_rate : ℝ) : ℝ :=
  sp_after_discount / (1 - discount_rate)

noncomputable def cost_price_from_profit (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

noncomputable def cost_price_from_loss (selling_price : ℝ) (loss_rate : ℝ) : ℝ :=
  selling_price / (1 - loss_rate)

noncomputable def total_cost_price : ℝ :=
  let CP1 := cost_price_from_profit (selling_price_before_discount 600 0.05) 0.25
  let CP2 := cost_price_from_loss 800 0.20
  let CP3 := cost_price_from_profit 1000 0.30 - 50
  CP1 + CP2 + CP3

theorem total_cost_price_is_correct : total_cost_price = 2224.49 :=
  by
  sorry

end NUMINAMATH_GPT_total_cost_price_is_correct_l24_2415


namespace NUMINAMATH_GPT_christmas_gift_count_l24_2441

theorem christmas_gift_count (initial_gifts : ℕ) (additional_gifts : ℕ) (gifts_to_orphanage : ℕ)
  (h1 : initial_gifts = 77)
  (h2 : additional_gifts = 33)
  (h3 : gifts_to_orphanage = 66) :
  (initial_gifts + additional_gifts - gifts_to_orphanage = 44) :=
by
  sorry

end NUMINAMATH_GPT_christmas_gift_count_l24_2441


namespace NUMINAMATH_GPT_perpendicular_plane_line_sum_l24_2474

theorem perpendicular_plane_line_sum (x y : ℝ)
  (h1 : ∃ k : ℝ, (2, -4 * x, 1) = (6 * k, 12 * k, -3 * k * y))
  : x + y = -2 :=
sorry

end NUMINAMATH_GPT_perpendicular_plane_line_sum_l24_2474


namespace NUMINAMATH_GPT_vector_dot_product_proof_l24_2462

variable (a b : ℝ × ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

theorem vector_dot_product_proof
  (h1 : a = (1, -3))
  (h2 : b = (3, 7)) :
  dot_product a b = -18 :=
by 
  sorry

end NUMINAMATH_GPT_vector_dot_product_proof_l24_2462


namespace NUMINAMATH_GPT_jerry_boxes_l24_2457

theorem jerry_boxes (boxes_sold boxes_left : ℕ) (h₁ : boxes_sold = 5) (h₂ : boxes_left = 5) : (boxes_sold + boxes_left = 10) :=
by
  sorry

end NUMINAMATH_GPT_jerry_boxes_l24_2457


namespace NUMINAMATH_GPT_union_complement_eq_complement_intersection_eq_l24_2435

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- Theorem 1: A ∪ (U \ B) = {2, 4, 5, 6}
theorem union_complement_eq : A ∪ (U \ B) = {2, 4, 5, 6} := by
  sorry

-- Theorem 2: U \ (A ∩ B) = {1, 2, 3, 4, 6, 7}
theorem complement_intersection_eq : U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by
  sorry

end NUMINAMATH_GPT_union_complement_eq_complement_intersection_eq_l24_2435


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l24_2411

theorem arithmetic_sequence_problem (q a₁ a₂ a₃ : ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : q > 1)
  (h2 : a₁ + a₂ + a₃ = 7)
  (h3 : a₁ + 3 + a₃ + 4 = 6 * a₂) :
  (∀ n : ℕ, a n = 2^(n-1)) ∧ (∀ n : ℕ, T n = (3 * n - 5) * 2^n + 5) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l24_2411


namespace NUMINAMATH_GPT_arithmetic_mean_eq_one_l24_2499

theorem arithmetic_mean_eq_one 
  (x a b : ℝ) 
  (hx : x ≠ 0) 
  (hb : b ≠ 0) : 
  (1 / 2 * ((x + a + b) / x + (x - a - b) / x)) = 1 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_eq_one_l24_2499


namespace NUMINAMATH_GPT_range_for_a_l24_2407

noncomputable def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 = 0 → (x ≥ 0 ∨ y ≥ 0)

theorem range_for_a (a : ℝ) :
  (line_not_in_second_quadrant a) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_GPT_range_for_a_l24_2407


namespace NUMINAMATH_GPT_elves_closed_eyes_l24_2467

theorem elves_closed_eyes :
  ∃ (age: ℕ → ℕ), -- Function assigning each position an age
  (∀ n, 1 ≤ n ∧ n ≤ 100 → (age n < age ((n % 100) + 1) ∧ age n < age (n - 1 % 100 + 1)) ∨
                          (age n > age ((n % 100) + 1) ∧ age n > age (n - 1 % 100 + 1))) :=
by
  sorry

end NUMINAMATH_GPT_elves_closed_eyes_l24_2467


namespace NUMINAMATH_GPT_remainder_expression_l24_2472

theorem remainder_expression (x y u v : ℕ) (hy_pos : y > 0) (h : x = u * y + v) (hv : 0 ≤ v) (hv_lt : v < y) :
  (x + 4 * u * y) % y = v :=
by
  sorry

end NUMINAMATH_GPT_remainder_expression_l24_2472


namespace NUMINAMATH_GPT_maximum_area_of_garden_l24_2468

theorem maximum_area_of_garden (w l : ℝ) 
  (h_perimeter : 2 * w + l = 400) : 
  ∃ (A : ℝ), A = 20000 ∧ A = w * l ∧ l = 400 - 2 * w ∧ ∀ (w' : ℝ) (l' : ℝ),
    2 * w' + l' = 400 → w' * l' ≤ 20000 :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_of_garden_l24_2468


namespace NUMINAMATH_GPT_table_height_l24_2496

theorem table_height (r s x y l : ℝ)
  (h1 : x + l - y = 32)
  (h2 : y + l - x = 28) :
  l = 30 :=
by
  sorry

end NUMINAMATH_GPT_table_height_l24_2496


namespace NUMINAMATH_GPT_smallest_common_multiple_of_9_and_6_l24_2455

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end NUMINAMATH_GPT_smallest_common_multiple_of_9_and_6_l24_2455


namespace NUMINAMATH_GPT_ab_greater_than_a_plus_b_l24_2432

variable {a b : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (h : a - b = a / b)

theorem ab_greater_than_a_plus_b : a * b > a + b :=
sorry

end NUMINAMATH_GPT_ab_greater_than_a_plus_b_l24_2432


namespace NUMINAMATH_GPT_sum_first_six_terms_l24_2466

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_six_terms :
  geometric_series_sum (1/4) (1/4) 6 = 4095 / 12288 :=
by 
  sorry

end NUMINAMATH_GPT_sum_first_six_terms_l24_2466


namespace NUMINAMATH_GPT_max_value_vector_sum_l24_2491

theorem max_value_vector_sum (α β : ℝ) :
  let a := (Real.cos α, Real.sin α)
  let b := (Real.sin β, -Real.cos β)
  |(a.1 + b.1, a.2 + b.2)| ≤ 2 := by
  sorry

end NUMINAMATH_GPT_max_value_vector_sum_l24_2491


namespace NUMINAMATH_GPT_division_and_multiplication_l24_2422

theorem division_and_multiplication (a b c d : ℝ) : (a / b / c * d) = 30 :=
by 
  let a := 120
  let b := 6
  let c := 2
  let d := 3
  sorry

end NUMINAMATH_GPT_division_and_multiplication_l24_2422


namespace NUMINAMATH_GPT_odd_periodic_function_l24_2470

theorem odd_periodic_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 5) = f x)
  (h_f1 : f 1 = 1)
  (h_f2 : f 2 = 2) :
  f 3 - f 4 = -1 :=
sorry

end NUMINAMATH_GPT_odd_periodic_function_l24_2470


namespace NUMINAMATH_GPT_parking_lot_perimeter_l24_2421

theorem parking_lot_perimeter (x y: ℝ) 
  (h1: x = (2 / 3) * y)
  (h2: x^2 + y^2 = 400)
  (h3: x * y = 120) :
  2 * (x + y) = 20 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_perimeter_l24_2421


namespace NUMINAMATH_GPT_find_min_value_l24_2490

theorem find_min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / a + 1 / b = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_min_value_l24_2490


namespace NUMINAMATH_GPT_Jamir_swims_more_l24_2480

def Julien_distance_per_day : ℕ := 50
def Sarah_distance_per_day (J : ℕ) : ℕ := 2 * J
def combined_distance_per_week (J S M : ℕ) : ℕ := 7 * (J + S + M)

theorem Jamir_swims_more :
  let J := Julien_distance_per_day
  let S := Sarah_distance_per_day J
  ∃ M, combined_distance_per_week J S M = 1890 ∧ (M - S = 20) := by
    let J := Julien_distance_per_day
    let S := Sarah_distance_per_day J
    use 120
    sorry

end NUMINAMATH_GPT_Jamir_swims_more_l24_2480


namespace NUMINAMATH_GPT_grape_juice_percentage_l24_2498

theorem grape_juice_percentage
  (original_mixture : ℝ)
  (percent_grape_juice : ℝ)
  (added_grape_juice : ℝ)
  (h1 : original_mixture = 50)
  (h2 : percent_grape_juice = 0.10)
  (h3 : added_grape_juice = 10)
  : (percent_grape_juice * original_mixture + added_grape_juice) / (original_mixture + added_grape_juice) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_grape_juice_percentage_l24_2498


namespace NUMINAMATH_GPT_pepperoni_crust_ratio_l24_2456

-- Define the conditions as Lean 4 statements
def L : ℕ := 50
def C : ℕ := 2 * L
def D : ℕ := 210
def S : ℕ := L + C + D
def S_E : ℕ := S / 4
def CR : ℕ := 600
def CH : ℕ := 400
def PizzaTotal (P : ℕ) : ℕ := CR + P + CH
def PizzaEats (P : ℕ) : ℕ := (PizzaTotal P) / 5
def JacksonEats : ℕ := 330

theorem pepperoni_crust_ratio (P : ℕ) (h1 : S_E + PizzaEats P = JacksonEats) : P / CR = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_pepperoni_crust_ratio_l24_2456


namespace NUMINAMATH_GPT_original_average_l24_2409

theorem original_average (A : ℝ) (h : (2 * (12 * A)) / 12 = 100) : A = 50 :=
by
  sorry

end NUMINAMATH_GPT_original_average_l24_2409


namespace NUMINAMATH_GPT_intersecting_lines_l24_2476

theorem intersecting_lines (m b : ℝ)
  (h1 : ∀ x, (9 : ℝ) = 2 * m * x + 3 → x = 3)
  (h2 : ∀ x, (9 : ℝ) = 4 * x + b → x = 3) :
  b + 2 * m = -1 :=
sorry

end NUMINAMATH_GPT_intersecting_lines_l24_2476


namespace NUMINAMATH_GPT_compounded_rate_of_growth_l24_2460

theorem compounded_rate_of_growth (k m : ℝ) :
  (1 + k / 100) * (1 + m / 100) - 1 = ((k + m + (k * m / 100)) / 100) :=
by
  sorry

end NUMINAMATH_GPT_compounded_rate_of_growth_l24_2460


namespace NUMINAMATH_GPT_monthly_income_of_A_l24_2469

theorem monthly_income_of_A (A B C : ℝ)
  (h1 : (A + B) / 2 = 5050)
  (h2 : (B + C) / 2 = 6250)
  (h3 : (A + C) / 2 = 5200) :
  A = 4000 :=
sorry

end NUMINAMATH_GPT_monthly_income_of_A_l24_2469


namespace NUMINAMATH_GPT_cone_radius_l24_2473

theorem cone_radius (h : ℝ) (V : ℝ) (π : ℝ) (r : ℝ)
    (h_def : h = 21)
    (V_def : V = 2199.114857512855)
    (volume_formula : V = (1/3) * π * r^2 * h) : r = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_cone_radius_l24_2473


namespace NUMINAMATH_GPT_minimal_abs_diff_l24_2410

theorem minimal_abs_diff (a b : ℤ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a * b - 3 * a + 7 * b = 222) : |a - b| = 54 :=
by
  sorry

end NUMINAMATH_GPT_minimal_abs_diff_l24_2410


namespace NUMINAMATH_GPT_xy_sum_143_l24_2404

theorem xy_sum_143 (x y : ℕ) (h1 : x < 30) (h2 : y < 30) (h3 : x + y + x * y = 143) (h4 : 0 < x) (h5 : 0 < y) :
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 :=
by
  sorry

end NUMINAMATH_GPT_xy_sum_143_l24_2404


namespace NUMINAMATH_GPT_age_difference_between_brother_and_cousin_l24_2454

-- Define the ages used in the problem 
def Lexie_age : ℕ := 8
def Grandma_age : ℕ := 68
def Brother_age : ℕ := Lexie_age - 6
def Sister_age : ℕ := 2 * Lexie_age
def Uncle_age : ℕ := Grandma_age - 12
def Cousin_age : ℕ := Brother_age + 5

-- The proof problem statement in Lean 4
theorem age_difference_between_brother_and_cousin : 
  Brother_age < Cousin_age ∧ Cousin_age - Brother_age = 5 :=
by
  -- Definitions and imports are done above. The statement below should prove the age difference.
  sorry

end NUMINAMATH_GPT_age_difference_between_brother_and_cousin_l24_2454


namespace NUMINAMATH_GPT_points_scored_fourth_game_l24_2431

-- Define the conditions
def avg_score_3_games := 18
def avg_score_4_games := 17
def games_played_3 := 3
def games_played_4 := 4

-- Calculate total points after 3 games
def total_points_3_games := avg_score_3_games * games_played_3

-- Calculate total points after 4 games
def total_points_4_games := avg_score_4_games * games_played_4

-- Define a theorem to prove the points scored in the fourth game
theorem points_scored_fourth_game :
  total_points_4_games - total_points_3_games = 14 :=
by
  sorry

end NUMINAMATH_GPT_points_scored_fourth_game_l24_2431


namespace NUMINAMATH_GPT_correct_option_l24_2439

variable (p q : Prop)

/-- If only one of p and q is true, then p or q is a true proposition. -/
theorem correct_option (h : (p ∧ ¬ q) ∨ (¬ p ∧ q)) : p ∨ q :=
by sorry

end NUMINAMATH_GPT_correct_option_l24_2439


namespace NUMINAMATH_GPT_value_of_A_l24_2487

theorem value_of_A {α : Type} [LinearOrderedSemiring α] 
  (L A D E : α) (L_value : L = 15) (LEAD DEAL DELL : α)
  (LEAD_value : LEAD = 50)
  (DEAL_value : DEAL = 55)
  (DELL_value : DELL = 60)
  (LEAD_condition : L + E + A + D = LEAD)
  (DEAL_condition : D + E + A + L = DEAL)
  (DELL_condition : D + E + L + L = DELL) :
  A = 25 :=
by
  sorry

end NUMINAMATH_GPT_value_of_A_l24_2487


namespace NUMINAMATH_GPT_intersection_height_correct_l24_2449

noncomputable def height_of_intersection (height1 height2 distance : ℝ) : ℝ :=
  let line1 (x : ℝ) := - (height1 / distance) * x + height1
  let line2 (x : ℝ) := - (height2 / distance) * x
  let x_intersect := - (height2 * distance) / (height1 - height2)
  line1 x_intersect

theorem intersection_height_correct :
  height_of_intersection 40 60 120 = 120 :=
by
  sorry

end NUMINAMATH_GPT_intersection_height_correct_l24_2449


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l24_2471

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x > 2 ∨ x < -2}
def expected_intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_M_and_N : M ∩ N = expected_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l24_2471


namespace NUMINAMATH_GPT_additional_rate_of_interest_l24_2463

variable (P A A' : ℝ) (T : ℕ) (R : ℝ)

-- Conditions
def principal_amount := (P = 8000)
def original_amount := (A = 9200)
def time_period := (T = 3)
def new_amount := (A' = 9440)

-- The Lean statement to prove the additional percentage of interest
theorem additional_rate_of_interest  (P A A' : ℝ) (T : ℕ) (R : ℝ)
    (h1 : principal_amount P)
    (h2 : original_amount A)
    (h3 : time_period T)
    (h4 : new_amount A') :
    (A' - P) / (P * T) * 100 - (A - P) / (P * T) * 100 = 1 :=
by
  sorry

end NUMINAMATH_GPT_additional_rate_of_interest_l24_2463


namespace NUMINAMATH_GPT_ellipse_distance_pf2_l24_2443

noncomputable def ellipse_focal_length := 2 * Real.sqrt 2
noncomputable def ellipse_equation (a : ℝ) (a_gt_one : a > 1)
  (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / a) + y^2 = 1

theorem ellipse_distance_pf2
  (a : ℝ) (a_gt_one : a > 1)
  (focus_distance : 2 * Real.sqrt (a - 1) = 2 * Real.sqrt 2)
  (F1 F2 P : ℝ × ℝ)
  (on_ellipse : ellipse_equation a a_gt_one P)
  (PF1_eq_two : dist P F1 = 2)
  (a_eq : a = 3) :
  dist P F2 = 2 * Real.sqrt 3 - 2 := 
sorry

end NUMINAMATH_GPT_ellipse_distance_pf2_l24_2443


namespace NUMINAMATH_GPT_prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l24_2452

variable {p a b : ℤ}

theorem prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes
  (hp : Prime p) (hp_ne_3 : p ≠ 3)
  (h1 : p ∣ (a + b)) (h2 : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) :=
sorry

end NUMINAMATH_GPT_prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l24_2452


namespace NUMINAMATH_GPT_find_p_q_l24_2418

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ :=
if x < -1 then p * x + q else 5 * x - 10

theorem find_p_q (p q : ℝ) (h : ∀ x, f p q (f p q x) = x) : p + q = 11 :=
sorry

end NUMINAMATH_GPT_find_p_q_l24_2418


namespace NUMINAMATH_GPT_combined_score_is_210_l24_2484

theorem combined_score_is_210 :
  ∀ (total_questions : ℕ) (marks_per_question : ℕ) (jose_wrong : ℕ) 
    (meghan_less_than_jose : ℕ) (jose_more_than_alisson : ℕ) (jose_total : ℕ),
  total_questions = 50 →
  marks_per_question = 2 →
  jose_wrong = 5 →
  meghan_less_than_jose = 20 →
  jose_more_than_alisson = 40 →
  jose_total = total_questions * marks_per_question - (jose_wrong * marks_per_question) →
  (jose_total - meghan_less_than_jose) + jose_total + (jose_total - jose_more_than_alisson) = 210 :=
by
  intros total_questions marks_per_question jose_wrong meghan_less_than_jose jose_more_than_alisson jose_total
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_combined_score_is_210_l24_2484


namespace NUMINAMATH_GPT_toy_position_from_left_l24_2444

/-- Define the total number of toys -/
def total_toys : ℕ := 19

/-- Define the position of toy (A) from the right -/
def position_from_right : ℕ := 8

/-- Prove the main statement: The position of toy (A) from the left is 12 given the conditions -/
theorem toy_position_from_left : total_toys - position_from_right + 1 = 12 := by
  sorry

end NUMINAMATH_GPT_toy_position_from_left_l24_2444


namespace NUMINAMATH_GPT_area_product_is_2_l24_2450

open Real

-- Definitions for parabola, points, and the condition of dot product
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2) = -4

def area (O F P : ℝ × ℝ) : ℝ :=
  0.5 * abs (O.1 * (F.2 - P.2) + F.1 * (P.2 - O.2) + P.1 * (O.2 - F.2))

-- Points A and B are on the parabola and the dot product condition holds
variables (A B : ℝ × ℝ)
variable (H_A_on_parabola : parabola A.1 A.2)
variable (H_B_on_parabola : parabola B.1 B.2)
variable (H_dot_product : dot_product_condition A B)

-- Focus of the parabola
def F : ℝ × ℝ := (1, 0)

-- Origin
def O : ℝ × ℝ := (0, 0)

-- Prove that the product of areas is 2
theorem area_product_is_2 : 
  area O F A * area O F B = 2 :=
sorry

end NUMINAMATH_GPT_area_product_is_2_l24_2450


namespace NUMINAMATH_GPT_flower_garden_mystery_value_l24_2442

/-- Prove the value of "花园探秘" given the arithmetic sum conditions and unique digit mapping. -/
theorem flower_garden_mystery_value :
  ∀ (shu_hua_hua_yuan : ℕ) (wo_ai_tan_mi : ℕ),
  shu_hua_hua_yuan + 2011 = wo_ai_tan_mi →
  (∃ (hua yuan tan mi : ℕ),
    0 ≤ hua ∧ hua < 10 ∧
    0 ≤ yuan ∧ yuan < 10 ∧
    0 ≤ tan ∧ tan < 10 ∧
    0 ≤ mi ∧ mi < 10 ∧
    hua ≠ yuan ∧ hua ≠ tan ∧ hua ≠ mi ∧
    yuan ≠ tan ∧ yuan ≠ mi ∧ tan ≠ mi ∧
    shu_hua_hua_yuan = hua * 1000 + yuan * 100 + tan * 10 + mi ∧
    wo_ai_tan_mi = 9713) := sorry

end NUMINAMATH_GPT_flower_garden_mystery_value_l24_2442


namespace NUMINAMATH_GPT_sum_f_sequence_l24_2420

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

theorem sum_f_sequence :
  f (1/10) + f (2/10) + f (3/10) + f (4/10) + f (5/10) + f (6/10) + f (7/10) + f (8/10) + f (9/10) = 9 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_f_sequence_l24_2420


namespace NUMINAMATH_GPT_Marge_savings_l24_2424

theorem Marge_savings
  (lottery_winnings : ℝ)
  (taxes_paid : ℝ)
  (student_loan_payment : ℝ)
  (amount_after_taxes : ℝ)
  (amount_after_student_loans : ℝ)
  (fun_money : ℝ)
  (investment : ℝ)
  (savings : ℝ)
  (h_win : lottery_winnings = 12006)
  (h_tax : taxes_paid = lottery_winnings / 2)
  (h_after_tax : amount_after_taxes = lottery_winnings - taxes_paid)
  (h_loans : student_loan_payment = amount_after_taxes / 3)
  (h_after_loans : amount_after_student_loans = amount_after_taxes - student_loan_payment)
  (h_fun : fun_money = 2802)
  (h_savings_investment : amount_after_student_loans - fun_money = savings + investment)
  (h_investment : investment = savings / 5)
  (h_left : amount_after_student_loans - fun_money = 1200) :
  savings = 1000 :=
by
  sorry

end NUMINAMATH_GPT_Marge_savings_l24_2424


namespace NUMINAMATH_GPT_complete_contingency_table_chi_square_test_certainty_l24_2401

-- Defining the initial conditions given in the problem
def total_students : ℕ := 100
def boys_dislike : ℕ := 10
def girls_like : ℕ := 20
def dislike_probability : ℚ := 0.4

-- Completed contingency table values based on given and inferred values
def boys_total : ℕ := 50
def girls_total : ℕ := 50
def boys_like : ℕ := boys_total - boys_dislike
def girls_dislike : ℕ := 30
def total_like : ℕ := boys_like + girls_like
def total_dislike : ℕ := boys_dislike + girls_dislike

-- Chi-square value from the solution
def K_squared : ℚ := 50 / 3

-- Declaring the proof problem for the completed contingency table
theorem complete_contingency_table :
  boys_total + girls_total = total_students ∧ 
  total_like + total_dislike = total_students ∧ 
  dislike_probability * total_students = total_dislike ∧ 
  boys_like = 40 ∧ 
  girls_dislike = 30 :=
sorry

-- Declaring the proof problem for the chi-square test
theorem chi_square_test_certainty :
  K_squared > 10.828 :=
sorry

end NUMINAMATH_GPT_complete_contingency_table_chi_square_test_certainty_l24_2401


namespace NUMINAMATH_GPT_number_of_shirts_that_weigh_1_pound_l24_2482

/-- 
Jon's laundry machine can do 5 pounds of laundry at a time. 
Some number of shirts weigh 1 pound. 
2 pairs of pants weigh 1 pound. 
Jon needs to wash 20 shirts and 20 pants. 
Jon has to do 3 loads of laundry. 
-/
theorem number_of_shirts_that_weigh_1_pound
    (machine_capacity : ℕ)
    (num_shirts : ℕ)
    (shirts_per_pound : ℕ)
    (pairs_of_pants_per_pound : ℕ)
    (num_pants : ℕ)
    (loads : ℕ)
    (weight_per_load : ℕ)
    (total_pants_weight : ℕ)
    (total_weight : ℕ)
    (shirt_weight_per_pound : ℕ)
    (shirts_weighing_one_pound : ℕ) :
  machine_capacity = 5 → 
  num_shirts = 20 → 
  pairs_of_pants_per_pound = 2 →
  num_pants = 20 →
  loads = 3 →
  weight_per_load = 5 → 
  total_pants_weight = (num_pants / pairs_of_pants_per_pound) →
  total_weight = (loads * weight_per_load) →
  shirts_weighing_one_pound = (total_weight - total_pants_weight) / num_shirts → 
  shirts_weighing_one_pound = 4 :=
by sorry

end NUMINAMATH_GPT_number_of_shirts_that_weigh_1_pound_l24_2482


namespace NUMINAMATH_GPT_compute_exponent_problem_l24_2485

noncomputable def exponent_problem : ℤ :=
  3 * (3^4) - (9^60) / (9^57)

theorem compute_exponent_problem : exponent_problem = -486 := by
  sorry

end NUMINAMATH_GPT_compute_exponent_problem_l24_2485
