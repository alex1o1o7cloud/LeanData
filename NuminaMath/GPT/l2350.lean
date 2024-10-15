import Mathlib

namespace NUMINAMATH_GPT_largest_divisor_of_five_consecutive_odds_l2350_235005

theorem largest_divisor_of_five_consecutive_odds (n : ℕ) (hn : n % 2 = 0) :
    ∃ d, d = 15 ∧ ∀ m, (m = (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11)) → d ∣ m :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_five_consecutive_odds_l2350_235005


namespace NUMINAMATH_GPT_biscuits_afternoon_eq_40_l2350_235094

-- Define the initial conditions given in the problem.
def butter_cookies_afternoon : Nat := 10
def additional_biscuits : Nat := 30

-- Define the number of biscuits based on the initial conditions.
def biscuits_afternoon : Nat := butter_cookies_afternoon + additional_biscuits

-- The statement to prove according to the problem.
theorem biscuits_afternoon_eq_40 : biscuits_afternoon = 40 := by
  -- The proof is to be done, hence we use 'sorry'.
  sorry

end NUMINAMATH_GPT_biscuits_afternoon_eq_40_l2350_235094


namespace NUMINAMATH_GPT_inequality1_inequality2_l2350_235098

-- Problem 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem inequality1 (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := sorry

-- Problem 2
def g (x : ℝ) : ℝ := f x + f (-x)

theorem inequality2 (k : ℝ) (h : ∀ x : ℝ, |k - 1| < g x) : -3 < k ∧ k < 5 := sorry

end NUMINAMATH_GPT_inequality1_inequality2_l2350_235098


namespace NUMINAMATH_GPT_chicken_price_reaches_81_in_2_years_l2350_235026

theorem chicken_price_reaches_81_in_2_years :
  ∃ t : ℝ, (t / 12 = 2) ∧ (∃ n : ℕ, (3:ℝ)^(n / 6) = 81 ∧ n = t) :=
by
  sorry

end NUMINAMATH_GPT_chicken_price_reaches_81_in_2_years_l2350_235026


namespace NUMINAMATH_GPT_triangle_area_rational_l2350_235056

-- Define the conditions
def satisfies_eq (x y : ℤ) : Prop := x - y = 1

-- Define the points
variables (x1 y1 x2 y2 x3 y3 : ℤ)

-- Assume each point satisfies the equation
axiom point1 : satisfies_eq x1 y1
axiom point2 : satisfies_eq x2 y2
axiom point3 : satisfies_eq x3 y3

-- Statement that we need to prove
theorem triangle_area_rational :
  ∃ (area : ℚ), 
    ∃ (triangle_points : ∃ (x1 y1 x2 y2 x3 y3 : ℤ), satisfies_eq x1 y1 ∧ satisfies_eq x2 y2 ∧ satisfies_eq x3 y3), 
      true :=
sorry

end NUMINAMATH_GPT_triangle_area_rational_l2350_235056


namespace NUMINAMATH_GPT_octagon_area_inscribed_in_square_l2350_235030

noncomputable def side_length_of_square (perimeter : ℝ) : ℝ :=
  perimeter / 4

noncomputable def trisected_segment_length (side_length : ℝ) : ℝ :=
  side_length / 3

noncomputable def area_of_removed_triangle (segment_length : ℝ) : ℝ :=
  (segment_length * segment_length) / 2

noncomputable def total_area_removed_by_triangles (area_of_triangle : ℝ) : ℝ :=
  4 * area_of_triangle

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_of_octagon (area_of_square : ℝ) (total_area_removed : ℝ) : ℝ :=
  area_of_square - total_area_removed

theorem octagon_area_inscribed_in_square (perimeter : ℝ) (H : perimeter = 144) :
  area_of_octagon (area_of_square (side_length_of_square perimeter))
    (total_area_removed_by_triangles (area_of_removed_triangle (trisected_segment_length (side_length_of_square perimeter))))
  = 1008 :=
by
  rw [H]
  -- Intermediate steps would contain calculations for side_length_of_square, trisected_segment_length, area_of_removed_triangle, total_area_removed_by_triangles, and area_of_square based on the given perimeter.
  sorry

end NUMINAMATH_GPT_octagon_area_inscribed_in_square_l2350_235030


namespace NUMINAMATH_GPT_probability_of_females_right_of_males_l2350_235088

-- Defining the total and favorable outcomes
def total_outcomes : ℕ := Nat.factorial 5
def favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial 2

-- Defining the probability as a rational number
def probability_all_females_right : ℚ := favorable_outcomes / total_outcomes

-- Stating the theorem
theorem probability_of_females_right_of_males :
  probability_all_females_right = 1 / 10 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_probability_of_females_right_of_males_l2350_235088


namespace NUMINAMATH_GPT_chess_tournament_no_804_games_l2350_235055

/-- Statement of the problem: 
    Under the given conditions, prove that it is impossible for exactly 804 games to have been played in the chess tournament.
--/
theorem chess_tournament_no_804_games :
  ¬ ∃ n : ℕ, n * (n - 4) = 1608 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_no_804_games_l2350_235055


namespace NUMINAMATH_GPT_zoey_finishes_20th_book_on_wednesday_l2350_235049

theorem zoey_finishes_20th_book_on_wednesday :
  let days_spent := (20 * 21) / 2
  (days_spent % 7) = 0 → 
  (start_day : ℕ) → start_day = 3 → ((start_day + days_spent) % 7) = 3 :=
by
  sorry

end NUMINAMATH_GPT_zoey_finishes_20th_book_on_wednesday_l2350_235049


namespace NUMINAMATH_GPT_probability_of_odd_sum_given_even_product_l2350_235018

-- Define a function to represent the probability of an event given the conditions
noncomputable def conditional_probability_odd_sum_even_product (dice : Fin 5 → Fin 8) : ℚ :=
  if h : (∃ i, (dice i).val % 2 = 0)  -- At least one die is even (product is even)
  then (1/2) / (31/32)  -- Probability of odd sum given even product
  else 0  -- If product is not even (not possible under conditions)

theorem probability_of_odd_sum_given_even_product :
  ∀ (dice : Fin 5 → Fin 8),
  conditional_probability_odd_sum_even_product dice = 16/31 :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_probability_of_odd_sum_given_even_product_l2350_235018


namespace NUMINAMATH_GPT_continued_fraction_l2350_235009

theorem continued_fraction {w x y : ℕ} (hw : 0 < w) (hx : 0 < x) (hy : 0 < y)
  (h_eq : (97:ℚ) / 19 = w + 1 / (x + 1 / y)) : w + x + y = 16 :=
sorry

end NUMINAMATH_GPT_continued_fraction_l2350_235009


namespace NUMINAMATH_GPT_geometric_sequence_properties_l2350_235051

theorem geometric_sequence_properties 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  (∀ n, a n = 2^(n - 1)) ∧ (S 6 = 63) := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l2350_235051


namespace NUMINAMATH_GPT_triangle_area_example_l2350_235020

def point : Type := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example : 
  triangle_area (0, 0) (0, 6) (8, 10) = 24 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_example_l2350_235020


namespace NUMINAMATH_GPT_goldie_earnings_l2350_235041

theorem goldie_earnings
  (hourly_wage : ℕ := 5)
  (hours_last_week : ℕ := 20)
  (hours_this_week : ℕ := 30) :
  hourly_wage * hours_last_week + hourly_wage * hours_this_week = 250 :=
by
  sorry

end NUMINAMATH_GPT_goldie_earnings_l2350_235041


namespace NUMINAMATH_GPT_f_negative_l2350_235096

-- Let f be a function defined on the real numbers
variable (f : ℝ → ℝ)

-- Conditions: f is odd and given form for non-negative x
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom f_positive : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2 * x

theorem f_negative (x : ℝ) (hx : x < 0) : f x = -x^2 + 2 * x := by
  sorry

end NUMINAMATH_GPT_f_negative_l2350_235096


namespace NUMINAMATH_GPT_Randy_initial_money_l2350_235076

theorem Randy_initial_money (M : ℝ) (r1 : M + 200 - 1200 = 2000) : M = 3000 :=
by
  sorry

end NUMINAMATH_GPT_Randy_initial_money_l2350_235076


namespace NUMINAMATH_GPT_calculate_neg_pow_mul_l2350_235034

theorem calculate_neg_pow_mul (a : ℝ) : -a^4 * a^3 = -a^7 := by
  sorry

end NUMINAMATH_GPT_calculate_neg_pow_mul_l2350_235034


namespace NUMINAMATH_GPT_g_of_1_equals_3_l2350_235013

theorem g_of_1_equals_3 (f g : ℝ → ℝ)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hg_even : ∀ x, g (-x) = g x)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 :=
sorry

end NUMINAMATH_GPT_g_of_1_equals_3_l2350_235013


namespace NUMINAMATH_GPT_distance_between_home_and_retreat_l2350_235042

theorem distance_between_home_and_retreat (D : ℝ) 
  (h1 : D / 50 + D / 75 = 10) : D = 300 :=
sorry

end NUMINAMATH_GPT_distance_between_home_and_retreat_l2350_235042


namespace NUMINAMATH_GPT_order_large_pizzas_sufficient_l2350_235079

def pizza_satisfaction (gluten_free_slices_per_large : ℕ) (medium_slices : ℕ) (small_slices : ℕ) 
                       (gluten_free_needed : ℕ) (dairy_free_needed : ℕ) :=
  let slices_gluten_free := small_slices
  let slices_dairy_free := 2 * medium_slices
  (slices_gluten_free < gluten_free_needed) → 
  let additional_slices_gluten_free := gluten_free_needed - slices_gluten_free
  let large_pizzas_gluten_free := (additional_slices_gluten_free + gluten_free_slices_per_large - 1) / gluten_free_slices_per_large
  large_pizzas_gluten_free = 1

theorem order_large_pizzas_sufficient :
  pizza_satisfaction 14 10 8 15 15 :=
by
  unfold pizza_satisfaction
  sorry

end NUMINAMATH_GPT_order_large_pizzas_sufficient_l2350_235079


namespace NUMINAMATH_GPT_quadratic_equation_m_value_l2350_235053

-- Definition of the quadratic equation having exactly one solution with the given parameters
def quadratic_equation_has_one_solution (a b c : ℚ) : Prop :=
  b^2 - 4 * a * c = 0

-- Given constants in the problem
def a : ℚ := 3
def b : ℚ := -7

-- The value of m we aim to prove
def m_correct : ℚ := 49 / 12

-- The theorem stating the problem
theorem quadratic_equation_m_value (m : ℚ) (h : quadratic_equation_has_one_solution a b m) : m = m_correct :=
  sorry

end NUMINAMATH_GPT_quadratic_equation_m_value_l2350_235053


namespace NUMINAMATH_GPT_height_comparison_of_cylinder_and_rectangular_solid_l2350_235077

theorem height_comparison_of_cylinder_and_rectangular_solid
  (V : ℝ) (A : ℝ) (h_cylinder : ℝ) (h_rectangular_solid : ℝ)
  (equal_volume : V = V)
  (equal_base_areas : A = A)
  (height_cylinder_eq : h_cylinder = V / A)
  (height_rectangular_solid_eq : h_rectangular_solid = V / A)
  : ¬ (h_cylinder > h_rectangular_solid) :=
by {
  sorry
}

end NUMINAMATH_GPT_height_comparison_of_cylinder_and_rectangular_solid_l2350_235077


namespace NUMINAMATH_GPT_sugar_cups_used_l2350_235091

def ratio_sugar_water : ℕ × ℕ := (1, 2)
def total_cups : ℕ := 84

theorem sugar_cups_used (r : ℕ × ℕ) (tc : ℕ) (hsugar : r.1 = 1) (hwater : r.2 = 2) (htotal : tc = 84) :
  (tc * r.1) / (r.1 + r.2) = 28 :=
by
  sorry

end NUMINAMATH_GPT_sugar_cups_used_l2350_235091


namespace NUMINAMATH_GPT_ratio_lcm_gcf_l2350_235050

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 2^2 * 3^2 * 7) (h₂ : b = 2 * 3^2 * 5 * 7) :
  (Nat.lcm a b) / (Nat.gcd a b) = 10 := by
  sorry

end NUMINAMATH_GPT_ratio_lcm_gcf_l2350_235050


namespace NUMINAMATH_GPT_probability_of_spade_or_king_l2350_235021

open Classical

-- Pack of cards containing 52 cards
def total_cards := 52

-- Number of spades in the deck
def num_spades := 13

-- Number of kings in the deck
def num_kings := 4

-- Number of overlap (king of spades)
def num_king_of_spades := 1

-- Total favorable outcomes
def total_favorable_outcomes := num_spades + num_kings - num_king_of_spades

-- Probability of drawing a spade or a king
def probability_spade_or_king := (total_favorable_outcomes : ℚ) / total_cards

theorem probability_of_spade_or_king : probability_spade_or_king = 4 / 13 := by
  sorry

end NUMINAMATH_GPT_probability_of_spade_or_king_l2350_235021


namespace NUMINAMATH_GPT_actual_tax_equals_600_l2350_235087

-- Definition for the first condition: initial tax amount
variable (a : ℝ)

-- Define the first reduction: 25% reduction
def first_reduction (a : ℝ) : ℝ := 0.75 * a

-- Define the second reduction: further 20% reduction
def second_reduction (tax_after_first_reduction : ℝ) : ℝ := 0.80 * tax_after_first_reduction

-- Define the final reduction: combination of both reductions
def final_tax (a : ℝ) : ℝ := second_reduction (first_reduction a)

-- Proof that with a = 1000, the actual tax is 600 million euros
theorem actual_tax_equals_600 (a : ℝ) (h₁ : a = 1000) : final_tax a = 600 := by
    rw [h₁]
    simp [final_tax, first_reduction, second_reduction]
    sorry

end NUMINAMATH_GPT_actual_tax_equals_600_l2350_235087


namespace NUMINAMATH_GPT_pond_field_area_ratio_l2350_235054

theorem pond_field_area_ratio
  (l : ℝ) (w : ℝ) (A_field : ℝ) (A_pond : ℝ)
  (h1 : l = 2 * w)
  (h2 : l = 16)
  (h3 : A_field = l * w)
  (h4 : A_pond = 8 * 8) :
  A_pond / A_field = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_pond_field_area_ratio_l2350_235054


namespace NUMINAMATH_GPT_right_triangle_area_l2350_235047

theorem right_triangle_area (a b c : ℕ) (habc : a = 3 ∧ b = 4 ∧ c = 5) : 
  (a * a + b * b = c * c) → 
  1 / 2 * (a * b) = 6 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l2350_235047


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l2350_235093

variable {a : ℕ → ℝ}  -- Define the sequence as a function from natural numbers to real numbers.

-- Definition that the sequence is arithmetic.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- The given condition in the problem
axiom h1 : a 1 + a 5 = 6

-- The statement to prove
theorem arithmetic_sequence_a3 (h : is_arithmetic_sequence a) : a 3 = 3 :=
by {
  -- The proof is omitted.
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_a3_l2350_235093


namespace NUMINAMATH_GPT_abs_of_negative_l2350_235004

theorem abs_of_negative (a : ℝ) (h : a < 0) : |a| = -a :=
sorry

end NUMINAMATH_GPT_abs_of_negative_l2350_235004


namespace NUMINAMATH_GPT_f_12_eq_12_l2350_235072

noncomputable def f : ℕ → ℤ := sorry

axiom f_int (n : ℕ) (hn : 0 < n) : ∃ k : ℤ, f n = k
axiom f_2 : f 2 = 2
axiom f_mul (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n
axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m > n → f m > f n

theorem f_12_eq_12 : f 12 = 12 := sorry

end NUMINAMATH_GPT_f_12_eq_12_l2350_235072


namespace NUMINAMATH_GPT_remainder_six_pow_4032_mod_13_l2350_235097

theorem remainder_six_pow_4032_mod_13 : (6 ^ 4032) % 13 = 1 := 
by
  sorry

end NUMINAMATH_GPT_remainder_six_pow_4032_mod_13_l2350_235097


namespace NUMINAMATH_GPT_range_of_3x_minus_y_l2350_235083

-- Defining the conditions in Lean
variable (x y : ℝ)

-- Condition 1: -1 ≤ x + y ≤ 1
def cond1 : Prop := -1 ≤ x + y ∧ x + y ≤ 1

-- Condition 2: 1 ≤ x - y ≤ 3
def cond2 : Prop := 1 ≤ x - y ∧ x - y ≤ 3

-- The theorem statement to prove that the range of 3x - y is [1, 7]
theorem range_of_3x_minus_y (h1 : cond1 x y) (h2 : cond2 x y) : 1 ≤ 3 * x - y ∧ 3 * x - y ≤ 7 := by
  sorry

end NUMINAMATH_GPT_range_of_3x_minus_y_l2350_235083


namespace NUMINAMATH_GPT_triangle_side_relationship_l2350_235024

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem triangle_side_relationship 
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = 40 * Real.pi / 180)
  (hβ : β = 60 * Real.pi / 180)
  (hγ : γ = 80 * Real.pi / 180)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * (a + b + c) = b * (b + c) :=
sorry

end NUMINAMATH_GPT_triangle_side_relationship_l2350_235024


namespace NUMINAMATH_GPT_fraction_q_p_l2350_235089

theorem fraction_q_p (k : ℝ) (c p q : ℝ) (h : 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) :
  c = 8 ∧ p = -3/4 ∧ q = 31/2 → q / p = -62 / 3 :=
by
  intros hc_hp_hq
  sorry

end NUMINAMATH_GPT_fraction_q_p_l2350_235089


namespace NUMINAMATH_GPT_smallest_value_of_c_l2350_235031

theorem smallest_value_of_c :
  ∃ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c ∧ (∀ d : ℚ, (3 * d + 4) * (d - 2) = 9 * d → c ≤ d) ∧ c = -8 / 3 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_c_l2350_235031


namespace NUMINAMATH_GPT_always_real_roots_range_of_b_analytical_expression_parabola_l2350_235078

-- Define the quadratic equation with parameter m
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (5 * m - 1) * x + 4 * m - 4

-- Part 1: Prove the equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 := 
sorry

-- Part 2: Find the range of b such that the line intersects the parabola at two distinct points
theorem range_of_b (b : ℝ) : 
  (∀ m : ℝ, m = 1 → (b > -25/4 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = (x1 + b) ∧ quadratic_eq m x2 = (x2 + b)))) :=
sorry

-- Part 3: Find the analytical expressions of the parabolas given the distance condition
theorem analytical_expression_parabola (m : ℝ) : 
  (∀ x1 x2 : ℝ, (|x1 - x2| = 2 → quadratic_eq m x1 = 0 → quadratic_eq m x2 = 0) → 
  (m = -1 ∨ m = -1/5) → 
  ((quadratic_eq (-1) x = -x^2 + 6*x - 8) ∨ (quadratic_eq (-1/5) x = -1/5*x^2 + 2*x - 24/5))) :=
sorry

end NUMINAMATH_GPT_always_real_roots_range_of_b_analytical_expression_parabola_l2350_235078


namespace NUMINAMATH_GPT_value_of_x_when_z_is_32_l2350_235066

variables {x y z k : ℝ}
variable (m n : ℝ)

def directly_proportional (x y : ℝ) (m : ℝ) := x = m * y^2
def inversely_proportional (y z : ℝ) (n : ℝ) := y = n / z^2

-- Our main proof goal
theorem value_of_x_when_z_is_32 (h1 : directly_proportional x y m) 
  (h2 : inversely_proportional y z n) (h3 : z = 8) (hx : x = 5) : 
  x = 5 / 256 :=
by
  let k := x * z^4
  have k_value : k = 20480 := by sorry
  have x_new : x = k / z^4 := by sorry
  have z_new : z = 32 := by sorry
  have x_final : x = 5 / 256 := by sorry
  exact x_final

end NUMINAMATH_GPT_value_of_x_when_z_is_32_l2350_235066


namespace NUMINAMATH_GPT_remainder_of_n_plus_4500_l2350_235058

theorem remainder_of_n_plus_4500 (n : ℕ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_n_plus_4500_l2350_235058


namespace NUMINAMATH_GPT_certain_number_l2350_235017

theorem certain_number (a x : ℝ) (h1 : a / x * 2 = 12) (h2 : x = 0.1) : a = 0.6 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_l2350_235017


namespace NUMINAMATH_GPT_rectangular_eq_of_C_slope_of_l_l2350_235068

noncomputable section

/-- Parametric equations for curve C -/
def parametric_eq (θ : ℝ) : ℝ × ℝ :=
⟨4 * Real.cos θ, 3 * Real.sin θ⟩

/-- Question 1: Prove that the rectangular coordinate equation of curve C is (x^2)/16 + (y^2)/9 = 1. -/
theorem rectangular_eq_of_C (x y θ : ℝ) (h₁ : x = 4 * Real.cos θ) (h₂ : y = 3 * Real.sin θ) : 
  x^2 / 16 + y^2 / 9 = 1 := 
sorry

/-- Line passing through point M(2, 2) with parametric equations -/
def line_through_M (t α : ℝ) : ℝ × ℝ :=
⟨2 + t * Real.cos α, 2 + t * Real.sin α⟩ 

/-- Question 2: Prove that the slope of line l passing M(2, 2) which intersects curve C at points A and B is -9/16 -/
theorem slope_of_l (t₁ t₂ α : ℝ) (t₁_t₂_sum_zero : (9 * Real.sin α + 36 * Real.cos α) = 0) :
  Real.tan α = -9 / 16 :=
sorry

end NUMINAMATH_GPT_rectangular_eq_of_C_slope_of_l_l2350_235068


namespace NUMINAMATH_GPT_find_difference_l2350_235095

theorem find_difference (m n : ℕ) (hm : ∃ x, m = 111 * x) (hn : ∃ y, n = 31 * y) (h_sum : m + n = 2017) :
  n - m = 463 :=
sorry

end NUMINAMATH_GPT_find_difference_l2350_235095


namespace NUMINAMATH_GPT_number_of_rabbits_l2350_235039

theorem number_of_rabbits
  (dogs : ℕ) (cats : ℕ) (total_animals : ℕ)
  (joins_each_cat : ℕ → ℕ)
  (hares_per_rabbit : ℕ)
  (h_dogs : dogs = 1)
  (h_cats : cats = 4)
  (h_total : total_animals = 37)
  (h_hares_per_rabbit : hares_per_rabbit = 3)
  (H : total_animals = dogs + cats + 4 * joins_each_cat cats + 3 * 4 * joins_each_cat cats) :
  joins_each_cat cats = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rabbits_l2350_235039


namespace NUMINAMATH_GPT_janet_percentage_of_snowballs_l2350_235064

-- Define the number of snowballs made by Janet
def janet_snowballs : ℕ := 50

-- Define the number of snowballs made by Janet's brother
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage calculation function
def calculate_percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Proof statement
theorem janet_percentage_of_snowballs : calculate_percentage janet_snowballs total_snowballs = 25 := 
by
  sorry

end NUMINAMATH_GPT_janet_percentage_of_snowballs_l2350_235064


namespace NUMINAMATH_GPT_lunch_break_duration_l2350_235044

theorem lunch_break_duration :
  ∃ L : ℝ, 
    ∀ (p h : ℝ),
      (9 - L) * (p + h) = 0.4 ∧
      (7 - L) * h = 0.3 ∧
      (12 - L) * p = 0.3 →
      L = 0.5 := by
  sorry

end NUMINAMATH_GPT_lunch_break_duration_l2350_235044


namespace NUMINAMATH_GPT_problem_l2350_235043

noncomputable def f : ℝ → ℝ := sorry

theorem problem (f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x)
                (h : ∀ x : ℝ, 0 < x → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3 / 2 :=
sorry

end NUMINAMATH_GPT_problem_l2350_235043


namespace NUMINAMATH_GPT_even_n_condition_l2350_235063

theorem even_n_condition (x : ℝ) (n : ℕ) (h : ∀ x, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : n % 2 = 0 :=
sorry

end NUMINAMATH_GPT_even_n_condition_l2350_235063


namespace NUMINAMATH_GPT_shifted_parabola_eq_l2350_235028

def initial_parabola (x : ℝ) : ℝ := 5 * x^2

def shifted_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

theorem shifted_parabola_eq :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_shifted_parabola_eq_l2350_235028


namespace NUMINAMATH_GPT_david_marks_in_mathematics_l2350_235084

-- Define marks in individual subjects and the average
def marks_in_english : ℝ := 70
def marks_in_physics : ℝ := 78
def marks_in_chemistry : ℝ := 60
def marks_in_biology : ℝ := 65
def average_marks : ℝ := 66.6
def number_of_subjects : ℕ := 5

-- Define a statement to be proven
theorem david_marks_in_mathematics : 
    average_marks * number_of_subjects 
    - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 60 := 
by simp [average_marks, number_of_subjects, marks_in_english, marks_in_physics, marks_in_chemistry, marks_in_biology]; sorry

end NUMINAMATH_GPT_david_marks_in_mathematics_l2350_235084


namespace NUMINAMATH_GPT_prob_top_three_cards_all_hearts_l2350_235061

-- Define the total numbers of cards and suits
def total_cards := 52
def hearts_count := 13

-- Define the probability calculation as per the problem statement
def prob_top_three_hearts : ℚ :=
  (13 * 12 * 11 : ℚ) / (52 * 51 * 50 : ℚ)

-- The theorem states that the probability of the top three cards being all hearts is 11/850
theorem prob_top_three_cards_all_hearts : prob_top_three_hearts = 11 / 850 := by
  -- The proof details are not required, just stating the structure
  sorry

end NUMINAMATH_GPT_prob_top_three_cards_all_hearts_l2350_235061


namespace NUMINAMATH_GPT_min_value_expression_l2350_235012

theorem min_value_expression (x y z : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : z > 1) : ∃ C, C = 12 ∧
  ∀ (x y z : ℝ), x > 1 → y > 1 → z > 1 → (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ C := by
  sorry

end NUMINAMATH_GPT_min_value_expression_l2350_235012


namespace NUMINAMATH_GPT_sum_x_y_l2350_235035

theorem sum_x_y (x y : ℤ) (h1 : x - y = 40) (h2 : x = 32) : x + y = 24 := by
  sorry

end NUMINAMATH_GPT_sum_x_y_l2350_235035


namespace NUMINAMATH_GPT_multiplicative_inverse_modulo_l2350_235045

noncomputable def A := 123456
noncomputable def B := 153846
noncomputable def N := 500000

theorem multiplicative_inverse_modulo :
  (A * B * N) % 1000000 = 1 % 1000000 :=
by
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_modulo_l2350_235045


namespace NUMINAMATH_GPT_range_of_a_l2350_235015

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -(x^2 + 2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2 * x) →
  f (2 - a^2) > f a ↔ -2 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2350_235015


namespace NUMINAMATH_GPT_average_pages_per_book_l2350_235075

-- Conditions
def book_thickness_in_inches : ℕ := 12
def pages_per_inch : ℕ := 80
def number_of_books : ℕ := 6

-- Given these conditions, we need to prove the average number of pages per book is 160.
theorem average_pages_per_book (book_thickness_in_inches : ℕ) (pages_per_inch : ℕ) (number_of_books : ℕ)
  (h1 : book_thickness_in_inches = 12)
  (h2 : pages_per_inch = 80)
  (h3 : number_of_books = 6) :
  (book_thickness_in_inches * pages_per_inch) / number_of_books = 160 := by
  sorry

end NUMINAMATH_GPT_average_pages_per_book_l2350_235075


namespace NUMINAMATH_GPT_new_barbell_cost_l2350_235082

theorem new_barbell_cost (old_barbell_cost new_barbell_cost : ℝ) 
  (h1 : old_barbell_cost = 250)
  (h2 : new_barbell_cost = old_barbell_cost * 1.3) :
  new_barbell_cost = 325 := by
  sorry

end NUMINAMATH_GPT_new_barbell_cost_l2350_235082


namespace NUMINAMATH_GPT_range_of_a_l2350_235025

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), (0 < x1 ∧ x1 < x2 ∧ x2 < 1) → (a * x2 - x2^3) - (a * x1 - x1^3) > x2 - x1) : a ≥ 4 :=
sorry


end NUMINAMATH_GPT_range_of_a_l2350_235025


namespace NUMINAMATH_GPT_triangle_in_base_7_l2350_235019

theorem triangle_in_base_7 (triangle : ℕ) 
  (h1 : (triangle + 6) % 7 = 0) : 
  triangle = 1 := 
sorry

end NUMINAMATH_GPT_triangle_in_base_7_l2350_235019


namespace NUMINAMATH_GPT_multiplication_decomposition_l2350_235073

theorem multiplication_decomposition :
  100 * 3 = 100 + 100 + 100 :=
sorry

end NUMINAMATH_GPT_multiplication_decomposition_l2350_235073


namespace NUMINAMATH_GPT_percentage_of_bags_not_sold_l2350_235022

theorem percentage_of_bags_not_sold
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_wednesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : sold_monday = 25)
  (h_tuesday : sold_tuesday = 70)
  (h_wednesday : sold_wednesday = 100)
  (h_thursday : sold_thursday = 110)
  (h_friday : sold_friday = 145) : 
  (initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday)) * 100 / initial_stock = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_bags_not_sold_l2350_235022


namespace NUMINAMATH_GPT_find_original_number_l2350_235006

theorem find_original_number (h1 : 268 * 74 = 19732) (h2 : 2.68 * x = 1.9832) : x = 0.74 :=
sorry

end NUMINAMATH_GPT_find_original_number_l2350_235006


namespace NUMINAMATH_GPT_exists_geometric_arithmetic_progressions_l2350_235003

theorem exists_geometric_arithmetic_progressions (n : ℕ) (hn : n > 3) :
  ∃ (x y : ℕ → ℕ),
  (∀ m < n, x (m + 1) = (1 + ε)^m ∧ y (m + 1) = (1 + (m + 1) * ε - δ)) ∧
  ∀ m < n, x m < y m ∧ y m < x (m + 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_geometric_arithmetic_progressions_l2350_235003


namespace NUMINAMATH_GPT_percentage_selected_in_state_B_l2350_235062

theorem percentage_selected_in_state_B (appeared: ℕ) (selectedA: ℕ) (selected_diff: ℕ)
  (percentage_selectedA: ℝ)
  (h1: appeared = 8100)
  (h2: percentage_selectedA = 6.0)
  (h3: selectedA = appeared * (percentage_selectedA / 100))
  (h4: selected_diff = 81)
  (h5: selectedB = selectedA + selected_diff) :
  ((selectedB : ℝ) / appeared) * 100 = 7 := 
  sorry

end NUMINAMATH_GPT_percentage_selected_in_state_B_l2350_235062


namespace NUMINAMATH_GPT_robert_monthly_expenses_l2350_235065

def robert_basic_salary : ℝ := 1250
def robert_sales : ℝ := 23600
def first_tier_limit : ℝ := 10000
def second_tier_limit : ℝ := 20000
def first_tier_rate : ℝ := 0.10
def second_tier_rate : ℝ := 0.12
def third_tier_rate : ℝ := 0.15
def savings_rate : ℝ := 0.20

def first_tier_commission : ℝ :=
  first_tier_limit * first_tier_rate

def second_tier_commission : ℝ :=
  (second_tier_limit - first_tier_limit) * second_tier_rate

def third_tier_commission : ℝ :=
  (robert_sales - second_tier_limit) * third_tier_rate

def total_commission : ℝ :=
  first_tier_commission + second_tier_commission + third_tier_commission

def total_earnings : ℝ :=
  robert_basic_salary + total_commission

def savings : ℝ :=
  total_earnings * savings_rate

def monthly_expenses : ℝ :=
  total_earnings - savings

theorem robert_monthly_expenses :
  monthly_expenses = 3192 := by
  sorry

end NUMINAMATH_GPT_robert_monthly_expenses_l2350_235065


namespace NUMINAMATH_GPT_h_of_j_of_3_l2350_235038

def h (x : ℝ) : ℝ := 4 * x + 3
def j (x : ℝ) : ℝ := (x + 2) ^ 2

theorem h_of_j_of_3 : h (j 3) = 103 := by
  sorry

end NUMINAMATH_GPT_h_of_j_of_3_l2350_235038


namespace NUMINAMATH_GPT_remainder_division_1614_254_eq_90_l2350_235014

theorem remainder_division_1614_254_eq_90 :
  ∀ (x : ℕ) (R : ℕ),
    1614 - x = 1360 →
    x * 6 + R = 1614 →
    0 ≤ R →
    R < x →
    R = 90 := 
by
  intros x R h_diff h_div h_nonneg h_lt
  sorry

end NUMINAMATH_GPT_remainder_division_1614_254_eq_90_l2350_235014


namespace NUMINAMATH_GPT_remainder_division_l2350_235036

theorem remainder_division (k : ℤ) (N : ℤ) (h : N = 133 * k + 16) : N % 50 = 49 := by
  sorry

end NUMINAMATH_GPT_remainder_division_l2350_235036


namespace NUMINAMATH_GPT_beautifulEquations_1_find_n_l2350_235069

def isBeautifulEquations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ x y : ℝ, eq1 x ∧ eq2 y ∧ x + y = 1

def eq1a (x : ℝ) : Prop := 4 * x - (x + 5) = 1
def eq2a (y : ℝ) : Prop := -2 * y - y = 3

theorem beautifulEquations_1 : isBeautifulEquations eq1a eq2a :=
sorry

def eq1b (x : ℝ) (n : ℝ) : Prop := 2 * x - n + 3 = 0
def eq2b (x : ℝ) (n : ℝ) : Prop := x + 5 * n - 1 = 0

theorem find_n (n : ℝ) : (∀ x1 x2 : ℝ, eq1b x1 n ∧ eq2b x2 n ∧ x1 + x2 = 1) → n = -1 / 3 :=
sorry

end NUMINAMATH_GPT_beautifulEquations_1_find_n_l2350_235069


namespace NUMINAMATH_GPT_football_team_lineup_count_l2350_235067

theorem football_team_lineup_count :
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3

  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 39600 :=
by
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3
  
  exact sorry

end NUMINAMATH_GPT_football_team_lineup_count_l2350_235067


namespace NUMINAMATH_GPT_possible_values_a_l2350_235023

def A : Set ℝ := {-1, 2}
def B (a : ℝ) : Set ℝ := {x | a * x^2 = 2 ∧ a ≥ 0}

def whale_swallowing (S T : Set ℝ) : Prop :=
S ⊆ T ∨ T ⊆ S

def moth_eating (S T : Set ℝ) : Prop :=
(∃ x, x ∈ S ∧ x ∈ T) ∧ ¬(S ⊆ T) ∧ ¬(T ⊆ S)

def valid_a (a : ℝ) : Prop :=
whale_swallowing A (B a) ∨ moth_eating A (B a)

theorem possible_values_a :
  {a : ℝ | valid_a a} = {0, 1/2, 2} :=
sorry

end NUMINAMATH_GPT_possible_values_a_l2350_235023


namespace NUMINAMATH_GPT_find_a5_from_geometric_sequence_l2350_235090

def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  geo_seq a q ∧ 0 < a 1 ∧ 0 < q ∧ 
  (a 4 = (a 2) ^ 2) ∧ 
  (a 2 + a 4 = 5 / 16)

theorem find_a5_from_geometric_sequence :
  ∀ (a : ℕ → ℝ) (q : ℝ), geometric_sequence_property a q → 
  a 5 = 1 / 32 :=
by 
  sorry

end NUMINAMATH_GPT_find_a5_from_geometric_sequence_l2350_235090


namespace NUMINAMATH_GPT_find_c_l2350_235016

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 15 = 3) : c = 9 :=
by sorry

end NUMINAMATH_GPT_find_c_l2350_235016


namespace NUMINAMATH_GPT_reciprocal_of_neg_four_l2350_235070

def is_reciprocal (x y : ℚ) : Prop := x * y = 1

theorem reciprocal_of_neg_four : is_reciprocal (-4) (-1/4) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_four_l2350_235070


namespace NUMINAMATH_GPT_probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l2350_235052

noncomputable def germination_rate : ℝ := 0.9
noncomputable def non_germination_rate : ℝ := 1 - germination_rate
noncomputable def strong_seedling_rate : ℝ := 0.6
noncomputable def non_strong_seedling_rate : ℝ := 1 - strong_seedling_rate

theorem probability_two_seeds_missing_seedlings :
  (non_germination_rate ^ 2) = 0.01 := sorry

theorem probability_two_seeds_no_strong_seedlings :
  (non_strong_seedling_rate ^ 2) = 0.16 := sorry

theorem probability_three_seeds_having_seedlings :
  (1 - non_germination_rate ^ 3) = 0.999 := sorry

theorem probability_three_seeds_having_strong_seedlings :
  (1 - non_strong_seedling_rate ^ 3) = 0.936 := sorry

end NUMINAMATH_GPT_probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l2350_235052


namespace NUMINAMATH_GPT_perfect_square_trinomial_l2350_235007

theorem perfect_square_trinomial (m : ℝ) : (∃ b : ℝ, (x^2 - 6 * x + m) = (x + b) ^ 2) → m = 9 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l2350_235007


namespace NUMINAMATH_GPT_recurrence_relation_l2350_235027

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end NUMINAMATH_GPT_recurrence_relation_l2350_235027


namespace NUMINAMATH_GPT_find_ax_plus_a_negx_l2350_235002

theorem find_ax_plus_a_negx
  (a : ℝ) (x : ℝ)
  (h₁ : a > 0)
  (h₂ : a^(x/2) + a^(-x/2) = 5) :
  a^x + a^(-x) = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_ax_plus_a_negx_l2350_235002


namespace NUMINAMATH_GPT_perfect_square_adjacent_smaller_l2350_235086

noncomputable def is_perfect_square (n : ℕ) : Prop := 
    ∃ k : ℕ, k * k = n

theorem perfect_square_adjacent_smaller (m : ℕ) (hm : is_perfect_square m) : 
    ∃ k : ℕ, (k * k = m ∧ (k - 1) * (k - 1) = m - 2 * k + 1) := 
by 
  sorry

end NUMINAMATH_GPT_perfect_square_adjacent_smaller_l2350_235086


namespace NUMINAMATH_GPT_hyperbola_problem_l2350_235000

-- Given the conditions of the hyperbola
def hyperbola (x y: ℝ) (b: ℝ) : Prop := (x^2) / 4 - (y^2) / (b^2) = 1 ∧ b > 0

-- Asymptote condition
def asymptote (b: ℝ) : Prop := (b / 2) = (Real.sqrt 6 / 2)

-- Foci, point P condition
def foci_and_point (PF1 PF2: ℝ) : Prop := PF1 / PF2 = 3 / 1 ∧ PF1 - PF2 = 4

-- Math proof problem
theorem hyperbola_problem (b PF1 PF2: ℝ) (P: ℝ × ℝ) :
  hyperbola P.1 P.2 b ∧ asymptote b ∧ foci_and_point PF1 PF2 →
  |PF1 + PF2| = 2 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_problem_l2350_235000


namespace NUMINAMATH_GPT_inequality_solution_l2350_235059

theorem inequality_solution (x : ℝ) :
  (2 / (x + 2) + 9 / (x + 6) ≥ 2) ↔ (x ∈ Set.Ico (-6 : ℝ) (-3) ∪ Set.Ioc (-2) 3) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l2350_235059


namespace NUMINAMATH_GPT_average_age_decrease_l2350_235071

theorem average_age_decrease (N T : ℕ) (h₁ : (T : ℝ) / N - 3 = (T - 30 : ℝ) / N) : N = 10 :=
sorry

end NUMINAMATH_GPT_average_age_decrease_l2350_235071


namespace NUMINAMATH_GPT_equations_of_line_l2350_235008

variables (x y : ℝ)

-- Given conditions
def passes_through_point (P : ℝ × ℝ) (x y : ℝ) := (x, y) = P

def has_equal_intercepts_on_axes (f : ℝ → ℝ) :=
  ∃ z : ℝ, z ≠ 0 ∧ f z = 0 ∧ f 0 = z

-- The proof problem statement
theorem equations_of_line (P : ℝ × ℝ) (hP : passes_through_point P 2 (-3)) (h : has_equal_intercepts_on_axes (λ x => -x / (x / 2))) :
  (x + y + 1 = 0) ∨ (3 * x + 2 * y = 0) := 
sorry

end NUMINAMATH_GPT_equations_of_line_l2350_235008


namespace NUMINAMATH_GPT_divisibility_properties_l2350_235046

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬(a + b ∣ a^(2*k) + b^(2*k)) ∧ ¬(a - b ∣ a^(2*k) + b^(2*k))) ∧ 
  ((a + b ∣ a^(2*k) - b^(2*k)) ∧ (a - b ∣ a^(2*k) - b^(2*k))) ∧ 
  (a + b ∣ a^(2*k + 1) + b^(2*k + 1)) ∧ 
  (a - b ∣ a^(2*k + 1) - b^(2*k + 1)) := 
by sorry

end NUMINAMATH_GPT_divisibility_properties_l2350_235046


namespace NUMINAMATH_GPT_second_divisor_correct_l2350_235001

noncomputable def smallest_num: Nat := 1012
def known_divisors := [12, 18, 21, 28]
def lcm_divisors: Nat := 252 -- This is the LCM of 12, 18, 21, and 28.
def result: Nat := 14

theorem second_divisor_correct :
  ∃ (d : Nat), d ≠ 12 ∧ d ≠ 18 ∧ d ≠ 21 ∧ d ≠ 28 ∧ d ≠ 252 ∧ (smallest_num - 4) % d = 0 ∧ d = result :=
by
  sorry

end NUMINAMATH_GPT_second_divisor_correct_l2350_235001


namespace NUMINAMATH_GPT_extreme_points_inequality_l2350_235032

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) + 1 / 2 * x ^ 2 - x

theorem extreme_points_inequality 
  (a : ℝ)
  (ha : 0 < a ∧ a < 1)
  (alpha beta : ℝ)
  (h_eq_alpha : alpha = -Real.sqrt (1 - a))
  (h_eq_beta : beta = Real.sqrt (1 - a))
  (h_order : alpha < beta) :
  (f a beta / alpha) < (1 / 2) :=
sorry

end NUMINAMATH_GPT_extreme_points_inequality_l2350_235032


namespace NUMINAMATH_GPT_required_sticks_l2350_235085

variables (x y : ℕ)
variables (h1 : 2 * x + 3 * y = 96)
variables (h2 : x + y = 40)

theorem required_sticks (x y : ℕ) (h1 : 2 * x + 3 * y = 96) (h2 : x + y = 40) : 
  x = 24 ∧ y = 16 ∧ (96 - (x * 2 + y * 3) / 2) = 116 :=
by
  sorry

end NUMINAMATH_GPT_required_sticks_l2350_235085


namespace NUMINAMATH_GPT_asia_discount_problem_l2350_235060

theorem asia_discount_problem
  (originalPrice : ℝ)
  (storeDiscount : ℝ)
  (memberDiscount : ℝ)
  (finalPriceUSD : ℝ)
  (exchangeRate : ℝ)
  (finalDiscountPercentage : ℝ) :
  originalPrice = 300 →
  storeDiscount = 0.20 →
  memberDiscount = 0.10 →
  finalPriceUSD = 224 →
  exchangeRate = 1.10 →
  finalDiscountPercentage = 28 :=
by
  sorry

end NUMINAMATH_GPT_asia_discount_problem_l2350_235060


namespace NUMINAMATH_GPT_find_a_over_b_l2350_235033

variable (x y z a b : ℝ)
variable (h₁ : 4 * x - 2 * y + z = a)
variable (h₂ : 6 * y - 12 * x - 3 * z = b)
variable (h₃ : b ≠ 0)

theorem find_a_over_b : a / b = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_over_b_l2350_235033


namespace NUMINAMATH_GPT_constant_term_expansion_l2350_235057

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) : 
  let term (r : ℕ) : ℝ := (1 / 2) ^ (9 - r) * (-1) ^ r * Nat.choose 9 r * x ^ (3 / 2 * r - 9)
  term 6 = 21 / 2 :=
by
  sorry

end NUMINAMATH_GPT_constant_term_expansion_l2350_235057


namespace NUMINAMATH_GPT_fraction_of_70cm_ropes_l2350_235074

theorem fraction_of_70cm_ropes (R : ℕ) (avg_all : ℚ) (avg_70 : ℚ) (avg_85 : ℚ) (total_len : R * avg_all = 480) 
  (total_ropes : R = 6) : 
  ∃ f : ℚ, f = 1 / 3 ∧ f * R * avg_70 + (R - f * R) * avg_85 = R * avg_all :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_70cm_ropes_l2350_235074


namespace NUMINAMATH_GPT_james_music_BPM_l2350_235048

theorem james_music_BPM 
  (hours_per_day : ℕ)
  (beats_per_week : ℕ)
  (days_per_week : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_per_day : ℕ)
  (total_minutes_per_week : ℕ)
  (BPM : ℕ)
  (h1 : hours_per_day = 2)
  (h2 : beats_per_week = 168000)
  (h3 : days_per_week = 7)
  (h4 : minutes_per_hour = 60)
  (h5 : minutes_per_day = hours_per_day * minutes_per_hour)
  (h6 : total_minutes_per_week = minutes_per_day * days_per_week)
  (h7 : BPM = beats_per_week / total_minutes_per_week)
  : BPM = 200 :=
sorry

end NUMINAMATH_GPT_james_music_BPM_l2350_235048


namespace NUMINAMATH_GPT_inequality_holds_l2350_235029

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, a*x^2 + 2*a*x - 2 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) (0 : ℝ) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l2350_235029


namespace NUMINAMATH_GPT_triangle_side_length_l2350_235092

theorem triangle_side_length (a : ℝ) (h1 : 4 < a) (h2 : a < 8) : a = 6 :=
sorry

end NUMINAMATH_GPT_triangle_side_length_l2350_235092


namespace NUMINAMATH_GPT_find_h_l2350_235037

theorem find_h (h : ℝ) (r s : ℝ) (h_eq : ∀ x : ℝ, x^2 - 4 * h * x - 8 = 0)
  (sum_of_squares : r^2 + s^2 = 20) (roots_eq : x^2 - 4 * h * x - 8 = (x - r) * (x - s)) :
  h = 1 / 2 ∨ h = -1 / 2 := 
sorry

end NUMINAMATH_GPT_find_h_l2350_235037


namespace NUMINAMATH_GPT_identify_irrational_number_l2350_235080

theorem identify_irrational_number :
  (∀ a b : ℤ, (-1 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (0 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (1 : ℚ) / (2 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  ¬(∃ a b : ℤ, (Real.sqrt 3) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
sorry

end NUMINAMATH_GPT_identify_irrational_number_l2350_235080


namespace NUMINAMATH_GPT_custom_op_eval_l2350_235011

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 5 * a + 2 * b - 1

-- State the required proof problem
theorem custom_op_eval : custom_op (-4) 6 = -9 := 
by
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_custom_op_eval_l2350_235011


namespace NUMINAMATH_GPT_intersection_A_B_l2350_235099

def set_A : Set ℝ := { x | x ≥ 0 }
def set_B : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem intersection_A_B : set_A ∩ set_B = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2350_235099


namespace NUMINAMATH_GPT_pencils_left_l2350_235040

-- Define the initial quantities
def MondayPencils := 35
def TuesdayPencils := 42
def WednesdayPencils := 3 * TuesdayPencils
def WednesdayLoss := 20
def ThursdayPencils := WednesdayPencils / 2
def FridayPencils := 2 * MondayPencils
def WeekendLoss := 50

-- Define the total number of pencils Sarah has at the end of each day
def TotalMonday := MondayPencils
def TotalTuesday := TotalMonday + TuesdayPencils
def TotalWednesday := TotalTuesday + WednesdayPencils - WednesdayLoss
def TotalThursday := TotalWednesday + ThursdayPencils
def TotalFriday := TotalThursday + FridayPencils
def TotalWeekend := TotalFriday - WeekendLoss

-- The proof statement
theorem pencils_left : TotalWeekend = 266 :=
by
  sorry

end NUMINAMATH_GPT_pencils_left_l2350_235040


namespace NUMINAMATH_GPT_sum_mod_five_l2350_235010

theorem sum_mod_five {n : ℕ} (h_pos : 0 < n) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ ¬ (∃ k : ℕ, n = 4 * k) :=
sorry

end NUMINAMATH_GPT_sum_mod_five_l2350_235010


namespace NUMINAMATH_GPT_max_children_arrangement_l2350_235081

theorem max_children_arrangement (n : ℕ) (h1 : n = 49) 
  (h2 : ∀ i j, i ≠ j → 1 ≤ i ∧ i ≤ 49 → 1 ≤ j ∧ j ≤ 49 → (i * j < 100)) : 
  ∃ k, k = 18 :=
by
  sorry

end NUMINAMATH_GPT_max_children_arrangement_l2350_235081
