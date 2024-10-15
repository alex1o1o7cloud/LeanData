import Mathlib

namespace NUMINAMATH_GPT_find_cost_price_l585_58539

theorem find_cost_price (SP : ℝ) (loss_percent : ℝ) (CP : ℝ) (h1 : SP = 1260) (h2 : loss_percent = 16) : CP = 1500 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l585_58539


namespace NUMINAMATH_GPT_convex_polygons_count_l585_58502

def binomial (n k : ℕ) : ℕ := if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def count_convex_polygons_with_two_acute_angles (m n : ℕ) : ℕ :=
  if 4 < m ∧ m < n then
    (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1))
  else 0

theorem convex_polygons_count (m n : ℕ) (h : 4 < m ∧ m < n) :
  count_convex_polygons_with_two_acute_angles m n = 
  (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1)) :=
by sorry

end NUMINAMATH_GPT_convex_polygons_count_l585_58502


namespace NUMINAMATH_GPT_books_problem_l585_58570

variable (L W : ℕ) -- L for Li Ming's initial books, W for Wang Hong's initial books

theorem books_problem (h1 : L = W + 26) (h2 : L - 14 = W + 14 - 2) : 14 = 14 :=
by
  sorry

end NUMINAMATH_GPT_books_problem_l585_58570


namespace NUMINAMATH_GPT_sum_of_coefficients_l585_58592

theorem sum_of_coefficients (a b : ℝ) (h1 : a = 1 * 5) (h2 : -b = 1 + 5) : a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l585_58592


namespace NUMINAMATH_GPT_find_Q_x_l585_58520

noncomputable def Q : ℝ → ℝ := sorry

variables (Q0 Q1 Q2 : ℝ)

axiom Q_def : ∀ x, Q x = Q0 + Q1 * x + Q2 * x^2
axiom Q_minus_2 : Q (-2) = -3

theorem find_Q_x : ∀ x, Q x = (3 / 5) * (1 + x - x^2) :=
by 
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_find_Q_x_l585_58520


namespace NUMINAMATH_GPT_total_points_scored_l585_58519

theorem total_points_scored (n m T : ℕ) 
  (h1 : T = 2 * n + 5 * m) 
  (h2 : n = m + 3 ∨ m = n + 3)
  : T = 20 :=
sorry

end NUMINAMATH_GPT_total_points_scored_l585_58519


namespace NUMINAMATH_GPT_max_val_4ab_sqrt3_12bc_l585_58571

theorem max_val_4ab_sqrt3_12bc {a b c : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
sorry

end NUMINAMATH_GPT_max_val_4ab_sqrt3_12bc_l585_58571


namespace NUMINAMATH_GPT_problem_statement_l585_58598

noncomputable def least_period (f : ℝ → ℝ) (P : ℝ) :=
  ∀ x : ℝ, f (x + P) = f x

theorem problem_statement (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 5) + f (x - 5) = f x) :
  least_period f 30 :=
sorry

end NUMINAMATH_GPT_problem_statement_l585_58598


namespace NUMINAMATH_GPT_units_digit_of_five_consecutive_product_is_zero_l585_58535

theorem units_digit_of_five_consecutive_product_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_five_consecutive_product_is_zero_l585_58535


namespace NUMINAMATH_GPT_tony_will_have_4_dollars_in_change_l585_58511

def tony_change : ℕ :=
  let bucket_capacity : ℕ := 2
  let sandbox_depth : ℕ := 2
  let sandbox_width : ℕ := 4
  let sandbox_length : ℕ := 5
  let sand_weight_per_cubic_foot : ℕ := 3
  let water_consumed_per_drink : ℕ := 3
  let trips_per_drink : ℕ := 4
  let bottle_capacity : ℕ := 15
  let bottle_cost : ℕ := 2
  let initial_money : ℕ := 10

  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let number_of_trips := total_sand_weight / bucket_capacity
  let number_of_drinks := number_of_trips / trips_per_drink
  let total_water_consumed := number_of_drinks * water_consumed_per_drink
  let number_of_bottles := total_water_consumed / bottle_capacity
  let total_water_cost := number_of_bottles * bottle_cost
  let change := initial_money - total_water_cost

  change

theorem tony_will_have_4_dollars_in_change : tony_change = 4 := by
  sorry

end NUMINAMATH_GPT_tony_will_have_4_dollars_in_change_l585_58511


namespace NUMINAMATH_GPT_cost_of_song_book_l585_58562

-- Define the given constants: cost of trumpet, cost of music tool, and total spent at the music store.
def cost_of_trumpet : ℝ := 149.16
def cost_of_music_tool : ℝ := 9.98
def total_spent_at_store : ℝ := 163.28

-- The goal is to prove that the cost of the song book is $4.14.
theorem cost_of_song_book :
  total_spent_at_store - (cost_of_trumpet + cost_of_music_tool) = 4.14 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_song_book_l585_58562


namespace NUMINAMATH_GPT_sum_interior_angles_of_regular_polygon_l585_58517

theorem sum_interior_angles_of_regular_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (n : ℝ)
  (h1 : exterior_angle = 45)
  (h2 : sum_exterior_angles = 360)
  (h3 : n = sum_exterior_angles / exterior_angle) :
  180 * (n - 2) = 1080 :=
by
  sorry

end NUMINAMATH_GPT_sum_interior_angles_of_regular_polygon_l585_58517


namespace NUMINAMATH_GPT_exam_question_correct_count_l585_58549

theorem exam_question_correct_count (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 110) : C = 34 :=
by
  sorry

end NUMINAMATH_GPT_exam_question_correct_count_l585_58549


namespace NUMINAMATH_GPT_problem1_problem2_l585_58560

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 / x

def is_increasing_on (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ < x₂ → f x₁ < f x₂

theorem problem1 : is_increasing_on f {x | 1 ≤ x} := 
by sorry

def is_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ > g x₂

theorem problem2 (g : ℝ → ℝ) (h_decreasing : is_decreasing g)
  (h_inequality : ∀ x : ℝ, 1 ≤ x → g (x^3 + 2) < g ((a^2 - 2 * a) * x)) :
  -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l585_58560


namespace NUMINAMATH_GPT_other_asymptote_of_hyperbola_l585_58521

theorem other_asymptote_of_hyperbola (a b : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) → 
  (∃ y : ℝ, x = -4) → 
  (∀ x : ℝ, y = - (1 / 2) * x - 7) := 
by {
  -- The proof will go here
  sorry
}

end NUMINAMATH_GPT_other_asymptote_of_hyperbola_l585_58521


namespace NUMINAMATH_GPT_correct_equation_after_moving_digit_l585_58555

theorem correct_equation_after_moving_digit :
  (101 - 102 = 1) →
  101 - 10^2 = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_equation_after_moving_digit_l585_58555


namespace NUMINAMATH_GPT_point_on_graph_l585_58587

variable (x y : ℝ)

-- Define the condition for a point to be on the graph of the function y = 6/x
def is_on_graph (x y : ℝ) : Prop :=
  x * y = 6

-- State the theorem to be proved
theorem point_on_graph : is_on_graph (-2) (-3) :=
  by
  sorry

end NUMINAMATH_GPT_point_on_graph_l585_58587


namespace NUMINAMATH_GPT_find_x_l585_58556

def f (x : ℝ) := 2 * x - 3

theorem find_x : ∃ x, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 :=
by 
  unfold f
  exists 5
  sorry

end NUMINAMATH_GPT_find_x_l585_58556


namespace NUMINAMATH_GPT_polynomial_factor_c_zero_l585_58596

theorem polynomial_factor_c_zero (c q : ℝ) :
    ∃ q : ℝ, (3*q + 6 = 0 ∧ c = 6*q + 12) ↔ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factor_c_zero_l585_58596


namespace NUMINAMATH_GPT_boys_count_eq_792_l585_58543

-- Definitions of conditions
variables (B G : ℤ)

-- Total number of students is 1443
axiom total_students : B + G = 1443

-- Number of girls is 141 fewer than the number of boys
axiom girls_fewer_than_boys : G = B - 141

-- Proof statement to show that the number of boys (B) is 792
theorem boys_count_eq_792 (B G : ℤ)
  (h1 : B + G = 1443)
  (h2 : G = B - 141) : B = 792 :=
by
  sorry

end NUMINAMATH_GPT_boys_count_eq_792_l585_58543


namespace NUMINAMATH_GPT_maximum_value_problem_l585_58546

theorem maximum_value_problem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b * c) * (b^2 - c * a) * (c^2 - a * b) ≤ 1 / 8 :=
sorry

end NUMINAMATH_GPT_maximum_value_problem_l585_58546


namespace NUMINAMATH_GPT_problem1_problem2_l585_58512

theorem problem1 : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

theorem problem2 : (Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l585_58512


namespace NUMINAMATH_GPT_bullseye_points_l585_58522

theorem bullseye_points (B : ℝ) (h : B + B / 2 = 75) : B = 50 :=
by
  sorry

end NUMINAMATH_GPT_bullseye_points_l585_58522


namespace NUMINAMATH_GPT_farm_horses_cows_l585_58589

variables (H C : ℕ)

theorem farm_horses_cows (H C : ℕ) (h1 : H = 6 * C) (h2 : (H - 15) = 3 * (C + 15)) : (H - 15) - (C + 15) = 70 :=
by {
  sorry
}

end NUMINAMATH_GPT_farm_horses_cows_l585_58589


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l585_58590

-- For the first expression
theorem simplify_expr1 (a b : ℝ) : 2 * a - 3 * b + a - 5 * b = 3 * a - 8 * b :=
by
  sorry

-- For the second expression
theorem simplify_expr2 (a : ℝ) : (a^2 - 6 * a) - 3 * (a^2 - 2 * a + 1) + 3 = -2 * a^2 :=
by
  sorry

-- For the third expression
theorem simplify_expr3 (x y : ℝ) : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l585_58590


namespace NUMINAMATH_GPT_cost_of_apples_and_bananas_l585_58593

variable (a b : ℝ) -- Assume a and b are real numbers.

theorem cost_of_apples_and_bananas (a b : ℝ) : 
  (3 * a + 2 * b) = 3 * a + 2 * b :=
by 
  sorry -- Proof placeholder

end NUMINAMATH_GPT_cost_of_apples_and_bananas_l585_58593


namespace NUMINAMATH_GPT_minimum_value_of_function_l585_58523

theorem minimum_value_of_function : ∀ x : ℝ, x ≥ 0 → (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8 / 3 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l585_58523


namespace NUMINAMATH_GPT_quadratic_equation_conditions_l585_58537

theorem quadratic_equation_conditions :
  ∃ (a b c : ℝ), a = 3 ∧ c = 1 ∧ (a * x^2 + b * x + c = 0 ↔ 3 * x^2 + 1 = 0) :=
by
  use 3, 0, 1
  sorry

end NUMINAMATH_GPT_quadratic_equation_conditions_l585_58537


namespace NUMINAMATH_GPT_number_of_days_l585_58504

noncomputable def days_to_lay_bricks (b c f : ℕ) : ℕ :=
(b * b) / f

theorem number_of_days (b c f : ℕ) (h_nonzero_f : f ≠ 0) (h_bc_pos : b > 0 ∧ c > 0) :
  days_to_lay_bricks b c f = (b * b) / f :=
by 
  sorry

end NUMINAMATH_GPT_number_of_days_l585_58504


namespace NUMINAMATH_GPT_total_cups_l585_58566

theorem total_cups (n : ℤ) (h_rainy_days : n = 8) :
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  total_cups = 26 :=
by
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  exact sorry

end NUMINAMATH_GPT_total_cups_l585_58566


namespace NUMINAMATH_GPT_probability_not_above_y_axis_l585_58507

-- Define the vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk (-1) 5
def Q := Point.mk 2 (-3)
def R := Point.mk (-5) (-3)
def S := Point.mk (-8) 5

-- Define predicate for being above the y-axis
def is_above_y_axis (p : Point) : Prop := p.y > 0

-- Define the parallelogram region (this is theoretical as defining a whole region 
-- can be complex, but we state the region as a property)
noncomputable def in_region_of_parallelogram (p : Point) : Prop := sorry

-- Define the probability calculation statement
theorem probability_not_above_y_axis (p : Point) :
  in_region_of_parallelogram p → ¬is_above_y_axis p := sorry

end NUMINAMATH_GPT_probability_not_above_y_axis_l585_58507


namespace NUMINAMATH_GPT_geometric_sequence_nine_l585_58569

theorem geometric_sequence_nine (a : ℕ → ℝ) (h_geo : ∀ n, a (n + 1) / a n = a 1 / a 0) 
  (h_a1 : a 1 = 2) (h_a5: a 5 = 4) : a 9 = 8 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_nine_l585_58569


namespace NUMINAMATH_GPT_find_number_of_girls_l585_58524

theorem find_number_of_girls (B G : ℕ) 
  (h1 : B + G = 604) 
  (h2 : 12 * B + 11 * G = 47 * 604 / 4) : 
  G = 151 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_girls_l585_58524


namespace NUMINAMATH_GPT_fraction_zero_if_abs_x_eq_one_l585_58527

theorem fraction_zero_if_abs_x_eq_one (x : ℝ) : 
  (|x| - 1) = 0 → (x^2 - 2 * x + 1 ≠ 0) → x = -1 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_zero_if_abs_x_eq_one_l585_58527


namespace NUMINAMATH_GPT_circle_O2_tangent_circle_O2_intersect_l585_58577

-- Condition: The equation of circle O_1 is \(x^2 + (y + 1)^2 = 4\)
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4

-- Condition: The center of circle O_2 is \(O_2(2, 1)\)
def center_O2 : (ℝ × ℝ) := (2, 1)

-- Prove the equation of circle O_2 if it is tangent to circle O_1
theorem circle_O2_tangent : 
  ∀ (x y : ℝ), circle_O1 x y → (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

-- Prove the equations of circle O_2 if it intersects circle O_1 and \(|AB| = 2\sqrt{2}\)
theorem circle_O2_intersect :
  ∀ (x y : ℝ), 
  circle_O1 x y → 
  (2 * Real.sqrt 2 = |(x - 2)^2 + (y - 1)^2 - 4| ∨ (x - 2)^2 + (y - 1)^2 = 20) :=
sorry

end NUMINAMATH_GPT_circle_O2_tangent_circle_O2_intersect_l585_58577


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l585_58599

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) (h2 : 2 = (Real.sqrt (a^2 + 3)) / a) : a = 1 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l585_58599


namespace NUMINAMATH_GPT_min_value_AP_AQ_l585_58510

noncomputable def min_distance (A P Q : ℝ × ℝ) : ℝ := dist A P + dist A Q

theorem min_value_AP_AQ :
  ∀ (A P Q : ℝ × ℝ),
    (∀ (x : ℝ), A = (x, 0)) →
    ((P.1 - 1) ^ 2 + (P.2 - 3) ^ 2 = 1) →
    ((Q.1 - 7) ^ 2 + (Q.2 - 5) ^ 2 = 4) →
    min_distance A P Q = 7 :=
by
  intros A P Q hA hP hQ
  -- Proof is to be provided here
  sorry

end NUMINAMATH_GPT_min_value_AP_AQ_l585_58510


namespace NUMINAMATH_GPT_rate_of_interest_first_year_l585_58531

-- Define the conditions
def principal : ℝ := 9000
def rate_second_year : ℝ := 0.05
def total_amount_after_2_years : ℝ := 9828

-- Define the problem statement which we need to prove
theorem rate_of_interest_first_year (R : ℝ) :
  (principal + (principal * R / 100)) + 
  ((principal + (principal * R / 100)) * rate_second_year) = 
  total_amount_after_2_years → 
  R = 4 := 
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_first_year_l585_58531


namespace NUMINAMATH_GPT_find_divisor_l585_58568

theorem find_divisor (n d k : ℤ) (h1 : n = k * d + 3) (h2 : n^2 % d = 4) : d = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l585_58568


namespace NUMINAMATH_GPT_part_I_solution_set_part_II_range_of_a_l585_58500

-- Given function definition
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 6

-- (I) Prove the solution set of f(x) < 0 when a = 5
theorem part_I_solution_set : 
  (∀ x : ℝ, f x 5 < 0 ↔ (-3 < x ∧ x < -2)) := by
  sorry

-- (II) Prove the range of a such that f(x) > 0 for all x ∈ ℝ 
theorem part_II_range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f x a > 0) ↔ (-2*Real.sqrt 6 < a ∧ a < 2*Real.sqrt 6)) := by
  sorry

end NUMINAMATH_GPT_part_I_solution_set_part_II_range_of_a_l585_58500


namespace NUMINAMATH_GPT_rectangular_garden_area_l585_58540

theorem rectangular_garden_area (w l : ℝ) 
  (h1 : l = 3 * w + 30) 
  (h2 : 2 * (l + w) = 800) : w * l = 28443.75 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_garden_area_l585_58540


namespace NUMINAMATH_GPT_a_eq_1_sufficient_not_necessary_l585_58516

theorem a_eq_1_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → |x - 1| ≤ |x - a|) ∧ ¬(∀ x : ℝ, x ≤ 1 → |x - 1| = |x - a|) :=
by
  sorry

end NUMINAMATH_GPT_a_eq_1_sufficient_not_necessary_l585_58516


namespace NUMINAMATH_GPT_parabola_expression_l585_58550

theorem parabola_expression:
  (∀ x : ℝ, y = a * (x + 3) * (x - 1)) →
  a * (0 + 3) * (0 - 1) = 2 →
  a = -2 / 3 →
  (∀ x : ℝ, y = -2 / 3 * x^2 - 4 / 3 * x + 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_expression_l585_58550


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l585_58505

theorem necessary_not_sufficient_condition (a : ℝ) :
  (a < 2) ∧ (a^2 - 4 < 0) ↔ (a < 2) ∧ (a > -2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l585_58505


namespace NUMINAMATH_GPT_find_values_l585_58578

theorem find_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) 
  (h2 : x + y^2 = 45) : 
  x = 7 ∧ y = Real.sqrt 38 :=
by
  sorry

end NUMINAMATH_GPT_find_values_l585_58578


namespace NUMINAMATH_GPT_min_value_of_f_l585_58558

def f (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 0 ∧ f 2 = 0 :=
  by sorry

end NUMINAMATH_GPT_min_value_of_f_l585_58558


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l585_58506

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = 12) (hS6 : S 6 = 42) 
  (h_arith_seq : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  a 10 + a 11 + a 12 = 66 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l585_58506


namespace NUMINAMATH_GPT_num_5_letter_words_with_at_least_two_consonants_l585_58561

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end NUMINAMATH_GPT_num_5_letter_words_with_at_least_two_consonants_l585_58561


namespace NUMINAMATH_GPT_simplify_tangent_expression_l585_58541

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_tangent_expression_l585_58541


namespace NUMINAMATH_GPT_sum_of_rationals_eq_l585_58501

theorem sum_of_rationals_eq (a1 a2 a3 a4 : ℚ)
  (h : {x : ℚ | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ x = a1 * a2 ∧ x = a1 * a3 ∧ x = a1 * a4 ∧ x = a2 * a3 ∧ x = a2 * a4 ∧ x = a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end NUMINAMATH_GPT_sum_of_rationals_eq_l585_58501


namespace NUMINAMATH_GPT_distance_to_plane_l585_58585

variable (V : ℝ) (A : ℝ) (r : ℝ) (d : ℝ)

-- Assume the volume of the sphere and area of the cross-section
def sphere_volume := V = 4 * Real.sqrt 3 * Real.pi
def cross_section_area := A = Real.pi

-- Define radius of sphere and cross-section
def sphere_radius := r = Real.sqrt 3
def cross_section_radius := Real.sqrt A = 1

-- Define distance as per Pythagorean theorem
def distance_from_center := d = Real.sqrt (r^2 - 1^2)

-- Main statement to prove
theorem distance_to_plane (V A : ℝ)
  (h1 : sphere_volume V) 
  (h2 : cross_section_area A) 
  (h3: sphere_radius r) 
  (h4: cross_section_radius A) : 
  distance_from_center r d :=
sorry

end NUMINAMATH_GPT_distance_to_plane_l585_58585


namespace NUMINAMATH_GPT_A_inter_B_complement_l585_58544

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem A_inter_B_complement :
  A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)} ∧
  {x | x ∉ A ∩ B} = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8 } :=
by
  sorry

end NUMINAMATH_GPT_A_inter_B_complement_l585_58544


namespace NUMINAMATH_GPT_percentage_HNO3_final_l585_58564

-- Define the initial conditions
def initial_volume_solution : ℕ := 60 -- 60 liters of solution
def initial_percentage_HNO3 : ℝ := 0.45 -- 45% HNO3
def added_pure_HNO3 : ℕ := 6 -- 6 liters of pure HNO3

-- Define the volume of HNO3 in the initial solution
def hno3_initial := initial_percentage_HNO3 * initial_volume_solution

-- Define the total volume of the final solution
def total_volume_final := initial_volume_solution + added_pure_HNO3

-- Define the total amount of HNO3 in the final solution
def total_hno3_final := hno3_initial + added_pure_HNO3

-- The main theorem: prove the final percentage is 50%
theorem percentage_HNO3_final :
  (total_hno3_final / total_volume_final) * 100 = 50 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_percentage_HNO3_final_l585_58564


namespace NUMINAMATH_GPT_second_number_is_72_l585_58547

theorem second_number_is_72 
  (sum_eq_264 : ∀ (x : ℝ), 2 * x + x + (2 / 3) * x = 264) 
  (first_eq_2_second : ∀ (x : ℝ), first = 2 * x)
  (third_eq_1_3_first : ∀ (first : ℝ), third = 1 / 3 * first) :
  second = 72 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_72_l585_58547


namespace NUMINAMATH_GPT_mixed_number_calculation_l585_58536

theorem mixed_number_calculation :
  47 * (4 + 3/7 - (5 + 1/3)) / (3 + 1/2 + (2 + 1/5)) = -7 - 119/171 := by
  sorry

end NUMINAMATH_GPT_mixed_number_calculation_l585_58536


namespace NUMINAMATH_GPT_even_n_ineq_l585_58509

theorem even_n_ineq (n : ℕ) (h : ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : Even n :=
  sorry

end NUMINAMATH_GPT_even_n_ineq_l585_58509


namespace NUMINAMATH_GPT_ball_distributions_l585_58532

theorem ball_distributions (p q : ℚ) (h1 : p = (Nat.choose 5 1 * Nat.choose 4 1 * Nat.choose 20 2 * Nat.choose 18 6 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20)
                            (h2 : q = (Nat.choose 20 4 * Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20) :
  p / q = 10 :=
by
  sorry

end NUMINAMATH_GPT_ball_distributions_l585_58532


namespace NUMINAMATH_GPT_complete_the_square_l585_58565

theorem complete_the_square :
  ∀ x : ℝ, (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_complete_the_square_l585_58565


namespace NUMINAMATH_GPT_smaller_angle_in_parallelogram_l585_58526

theorem smaller_angle_in_parallelogram 
  (opposite_angles : ∀ A B C D : ℝ, A = C ∧ B = D)
  (adjacent_angles_supplementary : ∀ A B : ℝ, A + B = π)
  (angle_diff : ∀ A B : ℝ, B = A + π/9) :
  ∃ θ : ℝ, θ = 4 * π / 9 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_in_parallelogram_l585_58526


namespace NUMINAMATH_GPT_ram_money_l585_58573

variable (R G K : ℕ)

theorem ram_money (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 2890) : R = 490 :=
by
  sorry

end NUMINAMATH_GPT_ram_money_l585_58573


namespace NUMINAMATH_GPT_distinct_names_impossible_l585_58588

-- Define the alphabet
inductive Letter
| a | u | o | e

-- Simplified form of words in the Mumbo-Jumbo language
def simplified_form : List Letter → List Letter
| [] => []
| (Letter.e :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.a :: xs) => simplified_form (Letter.a :: Letter.a :: xs)
| (Letter.o :: Letter.o :: Letter.o :: Letter.o :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.u :: xs) => simplified_form (Letter.u :: xs)
| (x :: xs) => x :: simplified_form xs

-- Number of possible names
def num_possible_names : ℕ := 343

-- Number of tribe members
def num_tribe_members : ℕ := 400

theorem distinct_names_impossible : num_possible_names < num_tribe_members :=
by
  -- Skipping the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_distinct_names_impossible_l585_58588


namespace NUMINAMATH_GPT_maximize_distance_l585_58533

noncomputable def maxTotalDistance (x : ℕ) (y : ℕ) (cityMPG highwayMPG : ℝ) (totalGallons : ℝ) : ℝ :=
  let cityDistance := cityMPG * ((x / 100.0) * totalGallons)
  let highwayDistance := highwayMPG * ((y / 100.0) * totalGallons)
  cityDistance + highwayDistance

theorem maximize_distance (x y : ℕ) (hx : x + y = 100) :
  maxTotalDistance x y 7.6 12.2 24.0 = 7.6 * (x / 100.0 * 24.0) + 12.2 * ((100.0 - x) / 100.0 * 24.0) :=
by
  sorry

end NUMINAMATH_GPT_maximize_distance_l585_58533


namespace NUMINAMATH_GPT_original_number_of_people_l585_58583

variable (x : ℕ)
-- Conditions
axiom one_third_left : x / 3 > 0
axiom half_dancing : 18 = x / 3

-- Theorem Statement
theorem original_number_of_people (x : ℕ) (one_third_left : x / 3 > 0) (half_dancing : 18 = x / 3) : x = 54 := sorry

end NUMINAMATH_GPT_original_number_of_people_l585_58583


namespace NUMINAMATH_GPT_range_of_a_l585_58567

noncomputable def f (x a : ℝ) : ℝ := (x^2 + (a - 1) * x + 1) * Real.exp x

theorem range_of_a :
  (∀ x, f x a + Real.exp 2 ≥ 0) ↔ (-2 ≤ a ∧ a ≤ Real.exp 3 + 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l585_58567


namespace NUMINAMATH_GPT_paper_thickness_after_2_folds_l585_58508

theorem paper_thickness_after_2_folds:
  ∀ (initial_thickness : ℝ) (folds : ℕ),
  initial_thickness = 0.1 →
  folds = 2 →
  (initial_thickness * 2^folds = 0.4) :=
by
  intros initial_thickness folds h_initial h_folds
  sorry

end NUMINAMATH_GPT_paper_thickness_after_2_folds_l585_58508


namespace NUMINAMATH_GPT_calculate_selling_price_l585_58538

theorem calculate_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) 
  (h1 : cost_price = 83.33) 
  (h2 : profit_percentage = 20) : 
  selling_price = 100 := by
  sorry

end NUMINAMATH_GPT_calculate_selling_price_l585_58538


namespace NUMINAMATH_GPT_p_add_inv_p_gt_two_l585_58530

theorem p_add_inv_p_gt_two {p : ℝ} (hp_pos : p > 0) (hp_neq_one : p ≠ 1) : p + 1 / p > 2 :=
by
  sorry

end NUMINAMATH_GPT_p_add_inv_p_gt_two_l585_58530


namespace NUMINAMATH_GPT_team_formation_problem_l585_58591

def num_team_formation_schemes : Nat :=
  let comb (n k : Nat) : Nat := Nat.choose n k
  (comb 5 1 * comb 4 2) + (comb 5 2 * comb 4 1)

theorem team_formation_problem :
  num_team_formation_schemes = 70 :=
sorry

end NUMINAMATH_GPT_team_formation_problem_l585_58591


namespace NUMINAMATH_GPT_speed_ratio_l585_58594

theorem speed_ratio (v_A v_B : ℝ) (L t : ℝ) 
  (h1 : v_A * t = (1 - 0.11764705882352941) * L)
  (h2 : v_B * t = L) : 
  v_A / v_B = 1.11764705882352941 := 
by 
  sorry

end NUMINAMATH_GPT_speed_ratio_l585_58594


namespace NUMINAMATH_GPT_trigonometric_identity_l585_58515

open Real

theorem trigonometric_identity (α : ℝ) (h : sin (α - (π / 12)) = 1 / 3) :
  cos (α + (17 * π / 12)) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l585_58515


namespace NUMINAMATH_GPT_min_value_is_four_l585_58552

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_is_four (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z h1 h2 h3 h4 = 4 :=
sorry

end NUMINAMATH_GPT_min_value_is_four_l585_58552


namespace NUMINAMATH_GPT_typing_speed_ratio_l585_58551

-- Define Tim's and Tom's typing speeds
variables (T t : ℝ)

-- Conditions from the problem
def condition1 : Prop := T + t = 15
def condition2 : Prop := T + 1.6 * t = 18

-- The proposition to prove: the ratio of Tom's typing speed to Tim's is 1:2
theorem typing_speed_ratio (h1 : condition1 T t) (h2 : condition2 T t) : t / T = 1 / 2 :=
sorry

end NUMINAMATH_GPT_typing_speed_ratio_l585_58551


namespace NUMINAMATH_GPT_triangle_area_0_0_0_5_7_12_l585_58581

theorem triangle_area_0_0_0_5_7_12 : 
    let base := 5
    let height := 7
    let area := (1 / 2) * base * height
    area = 17.5 := 
by
    sorry

end NUMINAMATH_GPT_triangle_area_0_0_0_5_7_12_l585_58581


namespace NUMINAMATH_GPT_sin_C_in_right_triangle_l585_58513

-- Triangle ABC with angle B = 90 degrees and tan A = 3/4
theorem sin_C_in_right_triangle (A C : ℝ) (h1 : A + C = π / 2) (h2 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_C_in_right_triangle_l585_58513


namespace NUMINAMATH_GPT_evaluate_expression_l585_58582

theorem evaluate_expression :
  (24^36) / (72^18) = 8^18 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l585_58582


namespace NUMINAMATH_GPT_center_of_circle_l585_58503

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : x + y = -1 := 
by 
  sorry

end NUMINAMATH_GPT_center_of_circle_l585_58503


namespace NUMINAMATH_GPT_orvin_max_balloons_l585_58559

variable (C : ℕ) (P : ℕ)

noncomputable def max_balloons (C P : ℕ) : ℕ :=
  let pair_cost := P + P / 2  -- Cost for two balloons
  let pairs := C / pair_cost  -- Maximum number of pairs
  pairs * 2 + (if C % pair_cost >= P then 1 else 0) -- Total balloons considering the leftover money

theorem orvin_max_balloons (hC : C = 120) (hP : P = 3) : max_balloons C P = 53 :=
by
  sorry

end NUMINAMATH_GPT_orvin_max_balloons_l585_58559


namespace NUMINAMATH_GPT_smallest_five_digit_int_equiv_mod_l585_58518

theorem smallest_five_digit_int_equiv_mod (n : ℕ) (h1 : 10000 ≤ n) (h2 : n % 9 = 4) : n = 10003 := 
sorry

end NUMINAMATH_GPT_smallest_five_digit_int_equiv_mod_l585_58518


namespace NUMINAMATH_GPT_quadratic_inequality_no_solution_l585_58553

theorem quadratic_inequality_no_solution (a b c : ℝ) (h : a ≠ 0)
  (hnsol : ∀ x : ℝ, ¬(a * x^2 + b * x + c ≥ 0)) :
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_no_solution_l585_58553


namespace NUMINAMATH_GPT_range_equality_of_f_and_f_f_l585_58528

noncomputable def f (x a : ℝ) := x * Real.log x - x + 2 * a

theorem range_equality_of_f_and_f_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → 1 < f x a) ∧ (∀ x : ℝ, 0 < x → f x a ≤ 1) →
  (∃ I : Set ℝ, (Set.range (λ x => f x a) = I) ∧ (Set.range (λ x => f (f x a) a) = I)) → 
  (1/2 < a ∧ a ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_range_equality_of_f_and_f_f_l585_58528


namespace NUMINAMATH_GPT_yao_ming_shots_l585_58597

-- Defining the conditions
def total_shots_made : ℕ := 14
def total_points_scored : ℕ := 28
def three_point_shots_made : ℕ := 3
def two_point_shots (x : ℕ) : ℕ := x
def free_throws_made (x : ℕ) : ℕ := total_shots_made - three_point_shots_made - x

-- The theorem we want to prove
theorem yao_ming_shots :
  ∃ (x y : ℕ),
    (total_shots_made = three_point_shots_made + x + y) ∧ 
    (total_points_scored = 3 * three_point_shots_made + 2 * x + y) ∧
    (x = 8) ∧
    (y = 3) :=
sorry

end NUMINAMATH_GPT_yao_ming_shots_l585_58597


namespace NUMINAMATH_GPT_eval_expression_l585_58574

theorem eval_expression : 5 + 4 - 3 + 2 - 1 = 7 :=
by
  -- Mathematically, this statement holds by basic arithmetic operations.
  sorry

end NUMINAMATH_GPT_eval_expression_l585_58574


namespace NUMINAMATH_GPT_mixed_number_multiplication_equiv_l585_58584

theorem mixed_number_multiplication_equiv :
  (-3 - 1 / 2) * (5 / 7) = -3.5 * (5 / 7) := 
by 
  sorry

end NUMINAMATH_GPT_mixed_number_multiplication_equiv_l585_58584


namespace NUMINAMATH_GPT_machine_shirts_per_minute_l585_58580

def shirts_made_yesterday : ℕ := 13
def shirts_made_today : ℕ := 3
def minutes_worked : ℕ := 2
def total_shirts_made : ℕ := shirts_made_yesterday + shirts_made_today
def shirts_per_minute : ℕ := total_shirts_made / minutes_worked

theorem machine_shirts_per_minute :
  shirts_per_minute = 8 := by
  sorry

end NUMINAMATH_GPT_machine_shirts_per_minute_l585_58580


namespace NUMINAMATH_GPT_salary_for_january_l585_58557

theorem salary_for_january (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8700)
  (h_may : May = 6500) :
  J = 3700 :=
by
  sorry

end NUMINAMATH_GPT_salary_for_january_l585_58557


namespace NUMINAMATH_GPT_train_speed_l585_58576

theorem train_speed
  (cross_time : ℝ := 5)
  (train_length : ℝ := 111.12)
  (conversion_factor : ℝ := 3.6)
  (speed : ℝ := (train_length / cross_time) * conversion_factor) :
  speed = 80 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l585_58576


namespace NUMINAMATH_GPT_range_of_k_l585_58575

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_f : ∀ x, f x = x^3 - 3 * x^2 - k)
  (h_f' : ∀ x, f' x = 3 * x^2 - 6 * x) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) ↔ -4 < k ∧ k < 0 :=
sorry

end NUMINAMATH_GPT_range_of_k_l585_58575


namespace NUMINAMATH_GPT_complete_square_solution_l585_58545

theorem complete_square_solution
  (x : ℝ)
  (h : x^2 + 4*x + 2 = 0):
  ∃ c : ℝ, (x + 2)^2 = c ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_solution_l585_58545


namespace NUMINAMATH_GPT_find_A_l585_58548

theorem find_A (A B C : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) (h4 : A < 10) (h5 : B < 10) (h6 : C < 10) (h7 : 10 * A + B + 10 * B + C = 101 * B + 10 * C) : A = 9 :=
sorry

end NUMINAMATH_GPT_find_A_l585_58548


namespace NUMINAMATH_GPT_decreasing_by_25_l585_58542

theorem decreasing_by_25 (n : ℕ) (k : ℕ) (y : ℕ) (hy : 0 ≤ y ∧ y < 10^k) : 
  (n = 6 * 10^k + y → n / 10 = y / 25) → (∃ m, n = 625 * 10^m) := 
sorry

end NUMINAMATH_GPT_decreasing_by_25_l585_58542


namespace NUMINAMATH_GPT_range_of_m_is_increasing_l585_58586

noncomputable def f (x : ℝ) (m: ℝ) := x^2 + m*x + m

theorem range_of_m_is_increasing :
  { m : ℝ // ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m } = {m | 4 ≤ m} :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_is_increasing_l585_58586


namespace NUMINAMATH_GPT_max_X_leq_ratio_XY_l585_58579

theorem max_X_leq_ratio_XY (x y z u : ℕ) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y z u : ℕ), (x + y = z + u) → (2 * x *y = z * u) → (x ≥ y) → m ≤ x / y :=
sorry

end NUMINAMATH_GPT_max_X_leq_ratio_XY_l585_58579


namespace NUMINAMATH_GPT_comparison_of_logs_l585_58563

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem comparison_of_logs : a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_logs_l585_58563


namespace NUMINAMATH_GPT_determinant_of_matrix_l585_58572

def mat : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![3, 0, 2],![8, 5, -2],![3, 3, 6]]

theorem determinant_of_matrix : Matrix.det mat = 90 := 
by 
  sorry

end NUMINAMATH_GPT_determinant_of_matrix_l585_58572


namespace NUMINAMATH_GPT_percentage_difference_between_maximum_and_minimum_changes_is_40_l585_58595

-- Definitions of initial and final survey conditions
def initialYesPercentage : ℝ := 0.40
def initialNoPercentage : ℝ := 0.60
def finalYesPercentage : ℝ := 0.80
def finalNoPercentage : ℝ := 0.20
def absenteePercentage : ℝ := 0.10

-- Main theorem stating the problem
theorem percentage_difference_between_maximum_and_minimum_changes_is_40 :
  let attendeesPercentage := 1 - absenteePercentage
  let adjustedFinalYesPercentage := finalYesPercentage / attendeesPercentage
  let minChange := adjustedFinalYesPercentage - initialYesPercentage
  let maxChange := initialYesPercentage + minChange
  maxChange - minChange = 0.40 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_percentage_difference_between_maximum_and_minimum_changes_is_40_l585_58595


namespace NUMINAMATH_GPT_calculate_expression_l585_58529

theorem calculate_expression : 
  let a := (-1 : Int) ^ 2023
  let b := (-8 : Int) / (-4)
  let c := abs (-5)
  a + b - c = -4 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l585_58529


namespace NUMINAMATH_GPT_quadratic_linear_term_l585_58514

theorem quadratic_linear_term (m : ℝ) 
  (h : 2 * m = 6) : -4 * (x : ℝ) + m * x = -x := by 
  sorry

end NUMINAMATH_GPT_quadratic_linear_term_l585_58514


namespace NUMINAMATH_GPT_arithmetic_sequence_difference_l585_58525

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

end NUMINAMATH_GPT_arithmetic_sequence_difference_l585_58525


namespace NUMINAMATH_GPT_painted_cells_l585_58554

open Int

theorem painted_cells : ∀ (m n : ℕ), (m = 20210) → (n = 1505) →
  let sub_rectangles := 215
  let cells_per_diagonal := 100
  let total_cells := sub_rectangles * cells_per_diagonal
  let total_painted_cells := 2 * total_cells
  let overlap_cells := sub_rectangles
  let unique_painted_cells := total_painted_cells - overlap_cells
  unique_painted_cells = 42785 := sorry

end NUMINAMATH_GPT_painted_cells_l585_58554


namespace NUMINAMATH_GPT_equivalent_sets_l585_58534

-- Definitions of the condition and expected result
def condition_set : Set ℕ := { x | x - 3 < 2 }
def expected_set : Set ℕ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem equivalent_sets : condition_set = expected_set := 
by
  sorry

end NUMINAMATH_GPT_equivalent_sets_l585_58534
