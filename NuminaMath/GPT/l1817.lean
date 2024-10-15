import Mathlib

namespace NUMINAMATH_GPT_sequence_increasing_l1817_181798

noncomputable def a (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ) (hn : 0 < n) : a n < a (n + 1) :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_sequence_increasing_l1817_181798


namespace NUMINAMATH_GPT_quadrilateral_area_l1817_181711

-- Define the number of interior and boundary points
def interior_points : ℕ := 5
def boundary_points : ℕ := 4

-- State the theorem to prove the area of the quadrilateral using Pick's Theorem
theorem quadrilateral_area : interior_points + (boundary_points / 2) - 1 = 6 := by sorry

end NUMINAMATH_GPT_quadrilateral_area_l1817_181711


namespace NUMINAMATH_GPT_disjoint_subsets_same_sum_l1817_181766

theorem disjoint_subsets_same_sum (s : Finset ℕ) (h₁ : s.card = 10) (h₂ : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 100) :
  ∃ A B : Finset ℕ, A ⊆ s ∧ B ⊆ s ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end NUMINAMATH_GPT_disjoint_subsets_same_sum_l1817_181766


namespace NUMINAMATH_GPT_find_e_l1817_181715

-- Definitions of the problem conditions
def Q (x : ℝ) (f d e : ℝ) := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) :
  (∀ x : ℝ, Q x f d e = 3 * x^3 + d * x^2 + e * x + f) →
  (f = 9) →
  ((∃ p q r : ℝ, p + q + r = - d / 3 ∧ p * q * r = - f / 3
    ∧ 1 / (p + q + r) = -3
    ∧ 3 + d + e + f = p * q * r) →
    e = -16) :=
by
  intros hQ hf hroots
  sorry

end NUMINAMATH_GPT_find_e_l1817_181715


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1817_181746

theorem arithmetic_sequence_sum (c d e : ℕ) (h1 : 10 - 3 = 7) (h2 : 17 - 10 = 7) (h3 : c - 17 = 7) (h4 : d - c = 7) (h5 : e - d = 7) : 
  c + d + e = 93 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1817_181746


namespace NUMINAMATH_GPT_seq_1964_l1817_181720

theorem seq_1964 (a : ℕ → ℤ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = -1)
  (h4 : ∀ n ≥ 4, a n = a (n - 1) * a (n - 3)) :
  a 1964 = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_seq_1964_l1817_181720


namespace NUMINAMATH_GPT_g_symmetry_value_h_m_interval_l1817_181792

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 12)) ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  1 + 1 / 2 * Real.sin (2 * x)

noncomputable def h (x : ℝ) : ℝ :=
  f x + g x

theorem g_symmetry_value (k : ℤ) : 
  g (k * Real.pi / 2 - Real.pi / 12) = (3 + (-1) ^ k) / 4 :=
by
  sorry

theorem h_m_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc (- Real.pi / 12) (5 * Real.pi / 12), |h x - m| ≤ 1) ↔ (1 ≤ m ∧ m ≤ 9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_g_symmetry_value_h_m_interval_l1817_181792


namespace NUMINAMATH_GPT_jeremy_total_earnings_l1817_181757

theorem jeremy_total_earnings :
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  steven_payment + mark_payment = 391 / 24 :=
by
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  sorry

end NUMINAMATH_GPT_jeremy_total_earnings_l1817_181757


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1817_181775

theorem sufficient_not_necessary_condition (x : ℝ) : (1 < x ∧ x < 2) → (x < 2) ∧ ((x < 2) → ¬(1 < x ∧ x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1817_181775


namespace NUMINAMATH_GPT_tangent_line_at_x_2_range_of_m_for_three_roots_l1817_181786

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

/-
Part 1: Proving the tangent line equation at x = 2
-/
theorem tangent_line_at_x_2 : ∃ k b, (k = 12) ∧ (b = -17) ∧ 
  (∀ x, 12 * x - f 2 - 17 = 0) :=
by
  sorry

/-
Part 2: Proving the range of m for three distinct real roots
-/
theorem range_of_m_for_three_roots (m : ℝ) :
  (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 + m = 0 ∧ f x2 + m = 0 ∧ f x3 + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_x_2_range_of_m_for_three_roots_l1817_181786


namespace NUMINAMATH_GPT_enumerate_set_l1817_181736

open Set

def is_positive_integer (x : ℕ) : Prop := x > 0

theorem enumerate_set :
  { p : ℕ × ℕ | p.1 + p.2 = 4 ∧ is_positive_integer p.1 ∧ is_positive_integer p.2 } =
  { (1, 3), (2, 2), (3, 1) } := by 
sorry

end NUMINAMATH_GPT_enumerate_set_l1817_181736


namespace NUMINAMATH_GPT_certain_number_is_gcd_l1817_181717

theorem certain_number_is_gcd (x : ℕ) (h1 : ∃ k : ℕ, 72 * 14 = k * x) (h2 : x = Nat.gcd 1008 72) : x = 72 :=
sorry

end NUMINAMATH_GPT_certain_number_is_gcd_l1817_181717


namespace NUMINAMATH_GPT_maximize_f_l1817_181771

noncomputable def f (x y z : ℝ) := x * y^2 * z^3

theorem maximize_f :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 →
  f x y z ≤ 1 / 432 ∧ (f x y z = 1 / 432 → x = 1/6 ∧ y = 1/3 ∧ z = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_maximize_f_l1817_181771


namespace NUMINAMATH_GPT_number_of_teams_in_BIG_N_l1817_181789

theorem number_of_teams_in_BIG_N (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_teams_in_BIG_N_l1817_181789


namespace NUMINAMATH_GPT_find_value_l1817_181707

theorem find_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 := 
sorry

end NUMINAMATH_GPT_find_value_l1817_181707


namespace NUMINAMATH_GPT_roof_length_width_difference_l1817_181702

variable (w l : ℕ)

theorem roof_length_width_difference (h1 : l = 7 * w) (h2 : l * w = 847) : l - w = 66 :=
by 
  sorry

end NUMINAMATH_GPT_roof_length_width_difference_l1817_181702


namespace NUMINAMATH_GPT_range_of_a_decreasing_l1817_181732

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a_decreasing (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Iic 4 → y ∈ Set.Iic 4 → x ≤ y → f x a ≥ f y a) ↔ a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_decreasing_l1817_181732


namespace NUMINAMATH_GPT_cos_graph_symmetric_l1817_181726

theorem cos_graph_symmetric :
  ∃ (x0 : ℝ), x0 = (Real.pi / 3) ∧ ∀ y, (∃ x, y = Real.cos (2 * x + Real.pi / 3)) ↔ (∃ x, y = Real.cos (2 * (2 * x0 - x) + Real.pi / 3)) :=
by
  -- Let x0 = π / 3
  let x0 := Real.pi / 3
  -- Show symmetry about x = π / 3
  exact ⟨x0, by norm_num, sorry⟩

end NUMINAMATH_GPT_cos_graph_symmetric_l1817_181726


namespace NUMINAMATH_GPT_complex_number_value_l1817_181777

theorem complex_number_value (i : ℂ) (h : i^2 = -1) : i^13 * (1 + i) = -1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_value_l1817_181777


namespace NUMINAMATH_GPT_spending_less_l1817_181727

-- Define the original costs in USD for each category.
def cost_A_usd : ℝ := 520
def cost_B_usd : ℝ := 860
def cost_C_usd : ℝ := 620

-- Define the budget cuts for each category.
def cut_A : ℝ := 0.25
def cut_B : ℝ := 0.35
def cut_C : ℝ := 0.30

-- Conversion rate from USD to EUR.
def conversion_rate : ℝ := 0.85

-- Sales tax rate.
def tax_rate : ℝ := 0.07

-- Calculate the reduced cost after budget cuts for each category.
def reduced_cost_A_usd := cost_A_usd * (1 - cut_A)
def reduced_cost_B_usd := cost_B_usd * (1 - cut_B)
def reduced_cost_C_usd := cost_C_usd * (1 - cut_C)

-- Convert costs from USD to EUR.
def reduced_cost_A_eur := reduced_cost_A_usd * conversion_rate
def reduced_cost_B_eur := reduced_cost_B_usd * conversion_rate
def reduced_cost_C_eur := reduced_cost_C_usd * conversion_rate

-- Calculate the total reduced cost in EUR before tax.
def total_reduced_cost_eur := reduced_cost_A_eur + reduced_cost_B_eur + reduced_cost_C_eur

-- Calculate the tax amount on the reduced cost.
def tax_reduced_cost := total_reduced_cost_eur * tax_rate

-- Total reduced cost in EUR after tax.
def total_reduced_cost_with_tax := total_reduced_cost_eur + tax_reduced_cost

-- Calculate the original costs in EUR without any cuts.
def original_cost_A_eur := cost_A_usd * conversion_rate
def original_cost_B_eur := cost_B_usd * conversion_rate
def original_cost_C_eur := cost_C_usd * conversion_rate

-- Calculate the total original cost in EUR before tax.
def total_original_cost_eur := original_cost_A_eur + original_cost_B_eur + original_cost_C_eur

-- Calculate the tax amount on the original cost.
def tax_original_cost := total_original_cost_eur * tax_rate

-- Total original cost in EUR after tax.
def total_original_cost_with_tax := total_original_cost_eur + tax_original_cost

-- Difference in spending.
def spending_difference := total_original_cost_with_tax - total_reduced_cost_with_tax

-- Prove the company must spend €561.1615 less.
theorem spending_less : spending_difference = 561.1615 := 
by 
  sorry

end NUMINAMATH_GPT_spending_less_l1817_181727


namespace NUMINAMATH_GPT_circle_radius_l1817_181782

theorem circle_radius (r A C : Real) (h1 : A = π * r^2) (h2 : C = 2 * π * r) (h3 : A + (Real.cos (π / 3)) * C = 56 * π) : r = 7 := 
by 
  sorry

end NUMINAMATH_GPT_circle_radius_l1817_181782


namespace NUMINAMATH_GPT_middle_number_is_nine_l1817_181764

theorem middle_number_is_nine (x : ℝ) (h : (2 * x)^2 + (4 * x)^2 = 180) : 3 * x = 9 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_is_nine_l1817_181764


namespace NUMINAMATH_GPT_regular_polygon_sides_l1817_181731

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1817_181731


namespace NUMINAMATH_GPT_gcd_subtract_ten_l1817_181718

theorem gcd_subtract_ten (a b : ℕ) (h₁ : a = 720) (h₂ : b = 90) : (Nat.gcd a b) - 10 = 80 := by
  sorry

end NUMINAMATH_GPT_gcd_subtract_ten_l1817_181718


namespace NUMINAMATH_GPT_find_m_set_l1817_181722

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
noncomputable def B (m : ℝ) : Set ℝ := if m = 0 then ∅ else {-1/m}

theorem find_m_set :
  { m : ℝ | A ∪ B m = A } = {0, -1/2, -1/3} :=
by
  sorry

end NUMINAMATH_GPT_find_m_set_l1817_181722


namespace NUMINAMATH_GPT_books_per_bookshelf_l1817_181744

theorem books_per_bookshelf (total_bookshelves total_books books_per_bookshelf : ℕ)
  (h1 : total_bookshelves = 23)
  (h2 : total_books = 621)
  (h3 : total_books = total_bookshelves * books_per_bookshelf) :
  books_per_bookshelf = 27 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_books_per_bookshelf_l1817_181744


namespace NUMINAMATH_GPT_dice_sum_not_18_l1817_181750

theorem dice_sum_not_18 (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) 
    (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (h_prod : d1 * d2 * d3 * d4 = 144) : 
    d1 + d2 + d3 + d4 ≠ 18 := 
sorry

end NUMINAMATH_GPT_dice_sum_not_18_l1817_181750


namespace NUMINAMATH_GPT_find_xiao_li_compensation_l1817_181788

-- Define the conditions
variable (total_days : ℕ) (extra_days : ℕ) (extra_compensation : ℕ)
variable (daily_work : ℕ) (daily_reward : ℕ) (xiao_li_days : ℕ)

-- Define the total compensation for Xiao Li
def xiao_li_compensation (xiao_li_days daily_reward : ℕ) : ℕ := xiao_li_days * daily_reward

-- The theorem statement asserting the final answer
theorem find_xiao_li_compensation
  (h1 : total_days = 12)
  (h2 : extra_days = 3)
  (h3 : extra_compensation = 2700)
  (h4 : daily_work = 1)
  (h5 : daily_reward = 225)
  (h6 : xiao_li_days = 2)
  (h7 : (total_days - extra_days) * daily_work = xiao_li_days * daily_work):
  xiao_li_compensation xiao_li_days daily_reward = 450 := 
sorry

end NUMINAMATH_GPT_find_xiao_li_compensation_l1817_181788


namespace NUMINAMATH_GPT_sufficient_condition_for_parallel_lines_l1817_181740

-- Define the condition for lines to be parallel
def lines_parallel (a b c d e f : ℝ) : Prop :=
(∃ k : ℝ, a = k * c ∧ b = k * d)

-- Define the specific lines given in the problem
def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 5

theorem sufficient_condition_for_parallel_lines (a : ℝ) :
  (lines_parallel (a) (1) (-1) (1) (-1) (1 + 5)) ↔ (a = -1) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_parallel_lines_l1817_181740


namespace NUMINAMATH_GPT_range_of_a_l1817_181753

theorem range_of_a (a : ℝ) :
  (∀ x, a * x^2 - x + (1 / 16 * a) > 0 → a > 2) →
  (0 < a - 3 / 2 ∧ a - 3 / 2 < 1 → 3 / 2 < a ∧ a < 5 / 2) →
  (¬ ((∀ x, a * x^2 - x + (1 / 16 * a) > 0) ∧ (0 < a - 3 / 2 ∧ a - 3 / 2 < 1))) →
  ((3 / 2 < a) ∧ (a ≤ 2)) ∨ (a ≥ 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1817_181753


namespace NUMINAMATH_GPT_nabla_value_l1817_181742

def nabla (a b c d : ℕ) : ℕ := a * c + b * d

theorem nabla_value : nabla 3 1 4 2 = 14 :=
by
  sorry

end NUMINAMATH_GPT_nabla_value_l1817_181742


namespace NUMINAMATH_GPT_minimize_distances_l1817_181762

/-- Given points P = (6, 7), Q = (3, 4), and R = (0, m),
    find the value of m that minimizes the sum of distances PR and QR. -/
theorem minimize_distances (m : ℝ) :
  let P := (6, 7)
  let Q := (3, 4)
  ∃ m : ℝ, 
    ∀ m' : ℝ, 
    (dist (6, 7) (0, m) + dist (3, 4) (0, m)) ≤ (dist (6, 7) (0, m') + dist (3, 4) (0, m'))
:= ⟨5, sorry⟩

end NUMINAMATH_GPT_minimize_distances_l1817_181762


namespace NUMINAMATH_GPT_point_P_in_second_quadrant_l1817_181767

-- Define what it means for a point to lie in a certain quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- The coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- Prove that the point P is in the second quadrant
theorem point_P_in_second_quadrant : in_second_quadrant (point_P.1) (point_P.2) :=
by
  sorry

end NUMINAMATH_GPT_point_P_in_second_quadrant_l1817_181767


namespace NUMINAMATH_GPT_garden_length_l1817_181795

theorem garden_length (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 240) : l = 80 :=
by
  sorry

end NUMINAMATH_GPT_garden_length_l1817_181795


namespace NUMINAMATH_GPT_calculate_actual_distance_l1817_181741

-- Definitions corresponding to the conditions
def map_scale : ℕ := 6000000
def map_distance_cm : ℕ := 5

-- The theorem statement corresponding to the proof problem
theorem calculate_actual_distance :
  (map_distance_cm * map_scale / 100000) = 300 := 
by
  sorry

end NUMINAMATH_GPT_calculate_actual_distance_l1817_181741


namespace NUMINAMATH_GPT_tigers_losses_l1817_181719

theorem tigers_losses (L T : ℕ) (h1 : 56 = 38 + L + T) (h2 : T = L / 2) : L = 12 :=
by sorry

end NUMINAMATH_GPT_tigers_losses_l1817_181719


namespace NUMINAMATH_GPT_number_added_multiplied_l1817_181708

theorem number_added_multiplied (x : ℕ) (h : (7/8 : ℚ) * x = 28) : ((x + 16) * (5/16 : ℚ)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_added_multiplied_l1817_181708


namespace NUMINAMATH_GPT_find_f_2_pow_2011_l1817_181729

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_positive (x : ℝ) : x > 0 → f x > 0

axiom f_initial_condition : f 1 + f 2 = 10

axiom f_functional_equation (a b : ℝ) : f a + f b = f (a+b) - 2 * Real.sqrt (f a * f b)

theorem find_f_2_pow_2011 : f (2^2011) = 2^4023 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_2_pow_2011_l1817_181729


namespace NUMINAMATH_GPT_simplify_sum1_simplify_sum2_l1817_181797

theorem simplify_sum1 : 296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200 := by
  sorry

theorem simplify_sum2 : 457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220 := by
  sorry

end NUMINAMATH_GPT_simplify_sum1_simplify_sum2_l1817_181797


namespace NUMINAMATH_GPT_find_four_digit_number_l1817_181748

theorem find_four_digit_number : ∃ x : ℕ, (1000 ≤ x ∧ x ≤ 9999) ∧ (x % 7 = 0) ∧ (x % 29 = 0) ∧ (19 * x % 37 = 3) ∧ x = 5075 :=
by
  sorry

end NUMINAMATH_GPT_find_four_digit_number_l1817_181748


namespace NUMINAMATH_GPT_problem_1_problem_2_l1817_181743

open Real

-- Step 1: Define the line and parabola conditions
def line_through_focus (k n : ℝ) : Prop := ∀ (x y : ℝ),
  y = k * (x - 1) ∧ (y = 0 → x = 1)
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Step 2: Prove x_1 x_2 = 1 if line passes through the focus
theorem problem_1 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k 1)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1))
  (h_non_zero : x1 * x2 ≠ 0) :
  x1 * x2 = 1 :=
sorry

-- Step 3: Prove n = 4 if x_1 x_2 + y_1 y_2 = 0
theorem problem_2 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k n)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - n) ∧ y2 = k * (x2 - n))
  (h_product_relate : x1 * x2 + y1 * y2 = 0) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1817_181743


namespace NUMINAMATH_GPT_impossible_to_achieve_target_l1817_181773

def initial_matchsticks := (1, 0, 0, 0)  -- Initial matchsticks at vertices (A, B, C, D)
def target_matchsticks := (1, 9, 8, 9)   -- Target matchsticks at vertices (A, B, C, D)

def S (a1 a2 a3 a4 : ℕ) : ℤ := a1 - a2 + a3 - a4

theorem impossible_to_achieve_target : 
  ¬∃ (f : ℕ × ℕ × ℕ × ℕ → ℕ × ℕ × ℕ × ℕ), 
    (f initial_matchsticks = target_matchsticks) ∧ 
    (∀ (a1 a2 a3 a4 : ℕ) k, 
      f (a1, a2, a3, a4) = (a1 - k, a2 + k, a3, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1, a2 - k, a3 + k, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1 + k, a2 - k, a3 - k, a4) ∨ 
      f (a1, a2, a3, a4) = (a1 - k, a2, a3 + k, a4 - k)) := sorry

end NUMINAMATH_GPT_impossible_to_achieve_target_l1817_181773


namespace NUMINAMATH_GPT_wall_height_correct_l1817_181791

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def wall_total_volume (num_bricks : ℕ) (brick_vol : ℝ) : ℝ := num_bricks * brick_vol

noncomputable def wall_height (total_volume : ℝ) (length : ℝ) (thickness : ℝ) : ℝ :=
  total_volume / (length * thickness)

theorem wall_height_correct :
  wall_height (wall_total_volume 7200 brick_volume) 900 22.5 = 600 := by
  sorry

end NUMINAMATH_GPT_wall_height_correct_l1817_181791


namespace NUMINAMATH_GPT_div_simplify_l1817_181706

theorem div_simplify (a b : ℝ) (h : a ≠ 0) : (8 * a * b) / (2 * a) = 4 * b :=
by
  sorry

end NUMINAMATH_GPT_div_simplify_l1817_181706


namespace NUMINAMATH_GPT_factor_y6_plus_64_l1817_181754

theorem factor_y6_plus_64 : (y^2 + 4) ∣ (y^6 + 64) :=
sorry

end NUMINAMATH_GPT_factor_y6_plus_64_l1817_181754


namespace NUMINAMATH_GPT_Compute_fraction_power_l1817_181721

theorem Compute_fraction_power :
  (81081 / 27027) ^ 4 = 81 :=
by
  -- We provide the specific condition as part of the proof statement
  have h : 27027 * 3 = 81081 := by norm_num
  sorry

end NUMINAMATH_GPT_Compute_fraction_power_l1817_181721


namespace NUMINAMATH_GPT_percentage_BCM_hens_l1817_181763

theorem percentage_BCM_hens (total_chickens : ℕ) (BCM_percentage : ℝ) (BCM_hens : ℕ) : 
  total_chickens = 100 → BCM_percentage = 0.20 → BCM_hens = 16 →
  ((BCM_hens : ℝ) / (total_chickens * BCM_percentage)) * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_percentage_BCM_hens_l1817_181763


namespace NUMINAMATH_GPT_total_books_is_177_l1817_181730

-- Define the number of books read (x), books yet to read (y), and the total number of books (T)
def x : Nat := 13
def y : Nat := 8
def T : Nat := x^2 + y

-- Prove that the total number of books in the series is 177
theorem total_books_is_177 : T = 177 :=
  sorry

end NUMINAMATH_GPT_total_books_is_177_l1817_181730


namespace NUMINAMATH_GPT_regular_tetrahedron_l1817_181787

-- Define the types for points and tetrahedrons
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Tetrahedron :=
(A B C D : Point)
(insphere : Point)

-- Conditions
def sphere_touches_at_angle_bisectors (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_altitudes (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_medians (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

-- Main theorem statement
theorem regular_tetrahedron (T : Tetrahedron)
  (h1 : sphere_touches_at_angle_bisectors T)
  (h2 : sphere_touches_at_altitudes T)
  (h3 : sphere_touches_at_medians T) :
  T.A = T.B ∧ T.A = T.C ∧ T.A = T.D := 
sorry

end NUMINAMATH_GPT_regular_tetrahedron_l1817_181787


namespace NUMINAMATH_GPT_find_e_m_l1817_181790

noncomputable def B (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![4, 5], ![7, e]]
noncomputable def B_inv (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := (1 / (4 * e - 35)) • ![![e, -5], ![-7, 4]]

theorem find_e_m (e m : ℝ) (B_inv_eq_mB : B_inv e = m • B e) : e = -4 ∧ m = 1 / 51 :=
sorry

end NUMINAMATH_GPT_find_e_m_l1817_181790


namespace NUMINAMATH_GPT_min_value_expression_l1817_181701

theorem min_value_expression (x y : ℝ) :
  ∃ m, (m = 104) ∧ (∀ x y : ℝ, (x + 3)^2 + 2 * (y - 2)^2 + 4 * (x - 7)^2 + (y + 4)^2 ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1817_181701


namespace NUMINAMATH_GPT_find_a_b_min_l1817_181733

def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_a_b_min (a b : ℝ) :
  (∃ a b, f 1 a b = 10 ∧ deriv (f · a b) 1 = 0) →
  a = 4 ∧ b = -11 ∧ ∀ x ∈ Set.Icc (-4:ℝ) 3, f x a b ≥ f 1 4 (-11) := 
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_find_a_b_min_l1817_181733


namespace NUMINAMATH_GPT_find_x_l1817_181781

theorem find_x (x : ℝ) (h : 3.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2800.0000000000005) : x = 0.3 :=
sorry

end NUMINAMATH_GPT_find_x_l1817_181781


namespace NUMINAMATH_GPT_cos_BHD_correct_l1817_181761

noncomputable def cos_BHD : ℝ :=
  let DB := 2
  let DC := 2 * Real.sqrt 2
  let AB := Real.sqrt 3
  let DH := DC
  let HG := DH * Real.sin (Real.pi / 6)  -- 30 degrees in radians
  let FB := AB
  let HB := FB * Real.sin (Real.pi / 4)  -- 45 degrees in radians
  let law_of_cosines :=
    DB^2 = DH^2 + HB^2 - 2 * DH * HB * Real.cos (Real.pi / 3)
  let expected_cos := (Real.sqrt 3) / 12
  expected_cos

theorem cos_BHD_correct :
  cos_BHD = (Real.sqrt 3) / 12 :=
by
  sorry

end NUMINAMATH_GPT_cos_BHD_correct_l1817_181761


namespace NUMINAMATH_GPT_men_required_l1817_181794

variable (m w : ℝ) -- Work done by one man and one woman in one day respectively
variable (x : ℝ) -- Number of men

-- Conditions from the problem
def condition1 (m w : ℝ) (x : ℝ) : Prop :=
  x * m = 12 * w

def condition2 (m w : ℝ) : Prop :=
  (6 * m + 11 * w) * 12 = 1

-- Proving that the number of men required to do the work in 20 days is x
theorem men_required (m w : ℝ) (x : ℝ) (h1 : condition1 m w x) (h2 : condition2 m w) : 
  (∃ x, condition1 m w x ∧ condition2 m w) := 
sorry

end NUMINAMATH_GPT_men_required_l1817_181794


namespace NUMINAMATH_GPT_mascots_arrangement_count_l1817_181774

-- Define the entities
def bing_dung_dung_mascots := 4
def xue_rong_rong_mascots := 3

-- Define the conditions
def xue_rong_rong_a_and_b_adjacent := true
def xue_rong_rong_c_not_adjacent_to_ab := true

-- Theorem stating the problem and asserting the answer
theorem mascots_arrangement_count : 
  (xue_rong_rong_a_and_b_adjacent ∧ xue_rong_rong_c_not_adjacent_to_ab) →
  (number_of_arrangements = 960) := by
  sorry

end NUMINAMATH_GPT_mascots_arrangement_count_l1817_181774


namespace NUMINAMATH_GPT_find_cube_edge_length_l1817_181723

-- Define parameters based on the problem conditions
def is_solution (n : ℕ) : Prop :=
  n > 4 ∧
  (6 * (n - 4)^2 = (n - 4)^3)

-- The main theorem statement
theorem find_cube_edge_length : ∃ n : ℕ, is_solution n ∧ n = 10 :=
by
  use 10
  sorry

end NUMINAMATH_GPT_find_cube_edge_length_l1817_181723


namespace NUMINAMATH_GPT_henley_initial_candies_l1817_181784

variables (C : ℝ)
variables (h1 : 0.60 * C = 180)

theorem henley_initial_candies : C = 300 :=
by sorry

end NUMINAMATH_GPT_henley_initial_candies_l1817_181784


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1817_181760

theorem simplify_and_evaluate_expression :
  let a := 2 * Real.sin (Real.pi / 3) + 3
  (a + 1) / (a - 3) - (a - 3) / (a + 2) / ((a^2 - 6 * a + 9) / (a^2 - 4)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1817_181760


namespace NUMINAMATH_GPT_percent_sales_other_l1817_181747

theorem percent_sales_other (percent_notebooks : ℕ) (percent_markers : ℕ) (h1 : percent_notebooks = 42) (h2 : percent_markers = 26) :
    100 - (percent_notebooks + percent_markers) = 32 := by
  sorry

end NUMINAMATH_GPT_percent_sales_other_l1817_181747


namespace NUMINAMATH_GPT_sum_of_cubes_l1817_181713

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 5) (h3 : abc = -6) : a^3 + b^3 + c^3 = -36 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_l1817_181713


namespace NUMINAMATH_GPT_find_x_l1817_181734

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x n : ℕ) (h₀ : n = 4) (h₁ : ¬(is_prime (2 * n + x))) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1817_181734


namespace NUMINAMATH_GPT_zoo_visitors_per_hour_l1817_181783

theorem zoo_visitors_per_hour 
    (h1 : ∃ V, 0.80 * V = 320)
    (h2 : ∃ H : Nat, H = 8)
    : ∃ N : Nat, N = 50 :=
by
  sorry

end NUMINAMATH_GPT_zoo_visitors_per_hour_l1817_181783


namespace NUMINAMATH_GPT_boys_more_than_girls_l1817_181725

def numGirls : ℝ := 28.0
def numBoys : ℝ := 35.0

theorem boys_more_than_girls : numBoys - numGirls = 7.0 := by
  sorry

end NUMINAMATH_GPT_boys_more_than_girls_l1817_181725


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1817_181749

noncomputable def isEllipseWithFociX (a b : ℝ) : Prop :=
  ∃ (C : ℝ → ℝ → Prop), (∀ (x y : ℝ), C x y ↔ (x^2 / a + y^2 / b = 1)) ∧ (a > b ∧ a > 0 ∧ b > 0)

theorem necessary_but_not_sufficient (a b : ℝ) :
  (∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1))
    → ((a > b ∧ a > 0 ∧ b > 0) → isEllipseWithFociX a b))
  ∧ ¬ (a > b → ∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1)) → isEllipseWithFociX a b) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1817_181749


namespace NUMINAMATH_GPT_field_day_difference_l1817_181759

def class_students (girls boys : ℕ) := girls + boys

def grade_students 
  (class1_girls class1_boys class2_girls class2_boys class3_girls class3_boys : ℕ) :=
  (class1_girls + class2_girls + class3_girls, class1_boys + class2_boys + class3_boys)

def diff_students (g1 b1 g2 b2 g3 b3 : ℕ) := 
  b1 + b2 + b3 - (g1 + g2 + g3)

theorem field_day_difference :
  let g3_1 := 10   -- 3rd grade first class girls
  let b3_1 := 14   -- 3rd grade first class boys
  let g3_2 := 12   -- 3rd grade second class girls
  let b3_2 := 10   -- 3rd grade second class boys
  let g3_3 := 11   -- 3rd grade third class girls
  let b3_3 :=  9   -- 3rd grade third class boys
  let g4_1 := 12   -- 4th grade first class girls
  let b4_1 := 13   -- 4th grade first class boys
  let g4_2 := 15   -- 4th grade second class girls
  let b4_2 := 11   -- 4th grade second class boys
  let g4_3 := 14   -- 4th grade third class girls
  let b4_3 := 12   -- 4th grade third class boys
  let g5_1 :=  9   -- 5th grade first class girls
  let b5_1 := 13   -- 5th grade first class boys
  let g5_2 := 10   -- 5th grade second class girls
  let b5_2 := 11   -- 5th grade second class boys
  let g5_3 := 11   -- 5th grade third class girls
  let b5_3 := 14   -- 5th grade third class boys
  diff_students (g3_1 + g3_2 + g3_3 + g4_1 + g4_2 + g4_3 + g5_1 + g5_2 + g5_3)
                (b3_1 + b3_2 + b3_3 + b4_1 + b4_2 + b4_3 + b5_1 + b5_2 + b5_3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_field_day_difference_l1817_181759


namespace NUMINAMATH_GPT_marta_should_buy_84_ounces_l1817_181712

/-- Definition of the problem's constants and assumptions --/
def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def bag_capacity : ℕ := 49
def num_bags : ℕ := 3

-- Marta wants to put the same number of apples and oranges in each bag
def equal_fruit (A O : ℕ) := A = O

-- Each bag should hold up to 49 ounces of fruit
def bag_limit (n : ℕ) := 4 * n + 3 * n ≤ 49

-- Marta's total apple weight based on the number of apples per bag and number of bags
def total_apple_weight (A : ℕ) : ℕ := (A * 3 * 4)

/-- Statement of the proof problem: 
Marta should buy 84 ounces of apples --/
theorem marta_should_buy_84_ounces : total_apple_weight 7 = 84 :=
by
  sorry

end NUMINAMATH_GPT_marta_should_buy_84_ounces_l1817_181712


namespace NUMINAMATH_GPT_find_divisor_l1817_181755

theorem find_divisor : ∃ (divisor : ℕ), ∀ (quotient remainder dividend : ℕ), quotient = 14 ∧ remainder = 7 ∧ dividend = 301 → (dividend = divisor * quotient + remainder) ∧ divisor = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1817_181755


namespace NUMINAMATH_GPT_base7_divisible_by_19_l1817_181716

theorem base7_divisible_by_19 (y : ℕ) (h : y ≤ 6) :
  (7 * y + 247) % 19 = 0 ↔ y = 0 :=
by sorry

end NUMINAMATH_GPT_base7_divisible_by_19_l1817_181716


namespace NUMINAMATH_GPT_age_difference_l1817_181739

theorem age_difference (O Y : ℕ) (h₀ : O = 38) (h₁ : Y + O = 74) : O - Y = 2 := by
  sorry

end NUMINAMATH_GPT_age_difference_l1817_181739


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_for_ax_square_pos_l1817_181703

variables (a x : ℝ)

theorem sufficient_but_not_necessary_for_ax_square_pos (h : a > 0) : 
  (a > 0 → ax^2 > 0) ∧ ((ax^2 > 0) → a > 0) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_for_ax_square_pos_l1817_181703


namespace NUMINAMATH_GPT_range_of_m_l1817_181752

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) 
(hf : ∀ x, f x = (Real.sqrt 3) * Real.sin ((Real.pi * x) / m))
(exists_extremum : ∃ x₀, (deriv f x₀ = 0) ∧ (x₀^2 + (f x₀)^2 < m^2)) :
(m > 2) ∨ (m < -2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1817_181752


namespace NUMINAMATH_GPT_bc_product_l1817_181778

theorem bc_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b * r - c = 0) → b * c = 30 :=
by
  sorry

end NUMINAMATH_GPT_bc_product_l1817_181778


namespace NUMINAMATH_GPT_sum_coordinates_D_l1817_181785

theorem sum_coordinates_D
    (M : (ℝ × ℝ))
    (C : (ℝ × ℝ))
    (D : (ℝ × ℝ))
    (H_M_midpoint : M = (5, 9))
    (H_C_coords : C = (11, 5))
    (H_M_def : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
    (D.1 + D.2) = 12 := 
by
  sorry
 
end NUMINAMATH_GPT_sum_coordinates_D_l1817_181785


namespace NUMINAMATH_GPT_max_value_y_on_interval_l1817_181709

noncomputable def y (x: ℝ) : ℝ := x^4 - 8 * x^2 + 2

theorem max_value_y_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y x = 11 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y z ≤ 11 := 
sorry

end NUMINAMATH_GPT_max_value_y_on_interval_l1817_181709


namespace NUMINAMATH_GPT_problem_a_even_triangles_problem_b_even_triangles_l1817_181728

-- Definition for problem (a)
def square_divided_by_triangles_3_4_even (a : ℕ) : Prop :=
  let area_triangle := 3 * 4 / 2
  let area_square := a * a
  let k := area_square / area_triangle
  (k % 2 = 0)

-- Definition for problem (b)
def rectangle_divided_by_triangles_1_2_even (l w : ℕ) : Prop :=
  let area_triangle := 1 * 2 / 2
  let area_rectangle := l * w
  let k := area_rectangle / area_triangle
  (k % 2 = 0)

-- Theorem for problem (a)
theorem problem_a_even_triangles {a : ℕ} (h : a > 0) :
  square_divided_by_triangles_3_4_even a :=
sorry

-- Theorem for problem (b)
theorem problem_b_even_triangles {l w : ℕ} (hl : l > 0) (hw : w > 0) :
  rectangle_divided_by_triangles_1_2_even l w :=
sorry

end NUMINAMATH_GPT_problem_a_even_triangles_problem_b_even_triangles_l1817_181728


namespace NUMINAMATH_GPT_vacation_cost_split_l1817_181776

theorem vacation_cost_split 
  (airbnb_cost : ℕ)
  (car_rental_cost : ℕ)
  (people : ℕ)
  (split_equally : Prop)
  (h1 : airbnb_cost = 3200)
  (h2 : car_rental_cost = 800)
  (h3 : people = 8)
  (h4 : split_equally)
  : (airbnb_cost + car_rental_cost) / people = 500 :=
by
  sorry

end NUMINAMATH_GPT_vacation_cost_split_l1817_181776


namespace NUMINAMATH_GPT_part_1_part_2_l1817_181796

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 4)

theorem part_1 (a : ℝ) (h : a = 3) :
  { x : ℝ | f x a ≥ 8 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | 1 ≤ x ∧ x ≤ 3 } ∪ { x : ℝ | x > 3 } := 
sorry

theorem part_2 (h : ∃ x : ℝ, f x a - abs (x + 2) ≤ 4) :
  -6 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l1817_181796


namespace NUMINAMATH_GPT_slices_with_both_toppings_l1817_181765

theorem slices_with_both_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (all_with_topping : total_slices = 15 ∧ pepperoni_slices = 8 ∧ mushroom_slices = 12 ∧ ∀ i, i < 15 → (i < 8 ∨ i < 12)) :
  ∃ n, (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_slices_with_both_toppings_l1817_181765


namespace NUMINAMATH_GPT_jessy_initial_earrings_l1817_181738

theorem jessy_initial_earrings (E : ℕ) (h₁ : 20 + E + (2 / 3 : ℚ) * E + (2 / 15 : ℚ) * E = 57) : E = 20 :=
by
  sorry

end NUMINAMATH_GPT_jessy_initial_earrings_l1817_181738


namespace NUMINAMATH_GPT_odd_number_as_diff_of_squares_l1817_181724

theorem odd_number_as_diff_of_squares :
    ∀ (x y : ℤ), 63 = x^2 - y^2 ↔ (x = 32 ∧ y = 31) ∨ (x = 12 ∧ y = 9) ∨ (x = 8 ∧ y = 1) := 
by
  sorry

end NUMINAMATH_GPT_odd_number_as_diff_of_squares_l1817_181724


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l1817_181769

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

theorem geometric_sequence_first_term (a_1 q : ℝ)
  (h1 : a_n a_1 q 2 * a_n a_1 q 3 * a_n a_1 q 4 = 27)
  (h2 : a_n a_1 q 6 = 27) 
  (h3 : a_1 > 0) : a_1 = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l1817_181769


namespace NUMINAMATH_GPT_seeds_total_l1817_181704

variable (seedsInBigGarden : Nat)
variable (numSmallGardens : Nat)
variable (seedsPerSmallGarden : Nat)

theorem seeds_total (h1 : seedsInBigGarden = 36) (h2 : numSmallGardens = 3) (h3 : seedsPerSmallGarden = 2) : 
  seedsInBigGarden + numSmallGardens * seedsPerSmallGarden = 42 := by
  sorry

end NUMINAMATH_GPT_seeds_total_l1817_181704


namespace NUMINAMATH_GPT_triangle_area_l1817_181770

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1817_181770


namespace NUMINAMATH_GPT_find_radius_l1817_181735

-- Define the given values
def arc_length : ℝ := 4
def central_angle : ℝ := 2

-- We need to prove this statement
theorem find_radius (radius : ℝ) : arc_length = radius * central_angle → radius = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_radius_l1817_181735


namespace NUMINAMATH_GPT_restaurant_sales_l1817_181714

theorem restaurant_sales (monday tuesday wednesday thursday : ℕ) 
  (h1 : monday = 40) 
  (h2 : tuesday = monday + 40) 
  (h3 : wednesday = tuesday / 2) 
  (h4 : monday + tuesday + wednesday + thursday = 203) : 
  thursday = wednesday + 3 := 
by sorry

end NUMINAMATH_GPT_restaurant_sales_l1817_181714


namespace NUMINAMATH_GPT_temp_product_l1817_181780

theorem temp_product (N : ℤ) (M D : ℤ)
  (h1 : M = D + N)
  (h2 : M - 8 = D + N - 8)
  (h3 : D + 5 = D + 5)
  (h4 : abs ((D + N - 8) - (D + 5)) = 3) :
  (N = 16 ∨ N = 10) →
  16 * 10 = 160 := 
by sorry

end NUMINAMATH_GPT_temp_product_l1817_181780


namespace NUMINAMATH_GPT_largest_time_for_77_degrees_l1817_181772

-- Define the initial conditions of the problem
def temperature_eqn (t : ℝ) : ℝ := -t^2 + 14 * t + 40

-- Define the proposition we want to prove
theorem largest_time_for_77_degrees : ∃ t, temperature_eqn t = 77 ∧ t = 11 := 
sorry

end NUMINAMATH_GPT_largest_time_for_77_degrees_l1817_181772


namespace NUMINAMATH_GPT_sin_ninety_degrees_l1817_181779

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end NUMINAMATH_GPT_sin_ninety_degrees_l1817_181779


namespace NUMINAMATH_GPT_probability_of_winning_l1817_181758

theorem probability_of_winning (P_lose P_tie P_win : ℚ) (h_lose : P_lose = 5/11) (h_tie : P_tie = 1/11)
  (h_total : P_lose + P_win + P_tie = 1) : P_win = 5/11 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_winning_l1817_181758


namespace NUMINAMATH_GPT_andrew_total_travel_time_l1817_181751

theorem andrew_total_travel_time :
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  subway_time + train_time + bike_time = 38 :=
by
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  sorry

end NUMINAMATH_GPT_andrew_total_travel_time_l1817_181751


namespace NUMINAMATH_GPT_third_trial_point_l1817_181768

variable (a b : ℝ) (x₁ x₂ x₃ : ℝ)

axiom experimental_range : a = 2 ∧ b = 4
axiom method_0618 : ∀ x1 x2, (x1 = 2 + 0.618 * (4 - 2) ∧ x2 = 2 + (4 - x1)) ∨ 
                              (x1 = (2 + (4 - 3.236)) ∧ x2 = 3.236)
axiom better_result (x₁ x₂ : ℝ) : x₁ > x₂  -- Assuming better means strictly greater

axiom x1_value : x₁ = 3.236 ∨ x₁ = 2.764
axiom x2_value : x₂ = 2.764 ∨ x₂ = 3.236
axiom x3_cases : (x₃ = 4 - 0.618 * (4 - x₁)) ∨ (x₃ = 2 + (4 - x₂))

theorem third_trial_point : x₃ = 3.528 ∨ x₃ = 2.472 :=
by
  sorry

end NUMINAMATH_GPT_third_trial_point_l1817_181768


namespace NUMINAMATH_GPT_seohyun_initial_marbles_l1817_181793

variable (M : ℤ)

theorem seohyun_initial_marbles (h1 : (2 / 3) * M = 12) (h2 : (1 / 2) * M + 12 = M) : M = 36 :=
sorry

end NUMINAMATH_GPT_seohyun_initial_marbles_l1817_181793


namespace NUMINAMATH_GPT_each_dog_food_intake_l1817_181710

theorem each_dog_food_intake (total_food : ℝ) (dog_count : ℕ) (equal_amount : ℝ) : total_food = 0.25 → dog_count = 2 → (total_food / dog_count) = equal_amount → equal_amount = 0.125 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_each_dog_food_intake_l1817_181710


namespace NUMINAMATH_GPT_find_m_l1817_181737

-- Defining the sets and conditions
def A (m : ℝ) : Set ℝ := {1, m-2}
def B : Set ℝ := {x | x = 2}

theorem find_m (m : ℝ) (h : A m ∩ B = {2}) : m = 4 := by
  sorry

end NUMINAMATH_GPT_find_m_l1817_181737


namespace NUMINAMATH_GPT_second_day_hike_ratio_l1817_181700

theorem second_day_hike_ratio (full_hike_distance first_day_distance third_day_distance : ℕ) 
(h_full_hike: full_hike_distance = 50)
(h_first_day: first_day_distance = 10)
(h_third_day: third_day_distance = 15) : 
(full_hike_distance - (first_day_distance + third_day_distance)) / full_hike_distance = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_second_day_hike_ratio_l1817_181700


namespace NUMINAMATH_GPT_problem_l1817_181756

theorem problem (a b c d e : ℝ) (h0 : a ≠ 0)
  (h1 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0)
  (h2 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h3 : 16 * a + 8 * b + 4 * c + 2 * d + e = 0) :
  (b + c + d) / a = -6 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1817_181756


namespace NUMINAMATH_GPT_graphs_intersect_exactly_eight_times_l1817_181745

theorem graphs_intersect_exactly_eight_times (A : ℝ) (hA : 0 < A) :
  ∃ (count : ℕ), count = 8 ∧ ∀ x y : ℝ, y = A * x ^ 4 → y ^ 2 + 5 = x ^ 2 + 6 * y :=
sorry

end NUMINAMATH_GPT_graphs_intersect_exactly_eight_times_l1817_181745


namespace NUMINAMATH_GPT_football_players_count_l1817_181705

-- Define the given conditions
def total_students : ℕ := 39
def long_tennis_players : ℕ := 20
def both_sports : ℕ := 17
def play_neither : ℕ := 10

-- Define a theorem to prove the number of football players is 26
theorem football_players_count : 
  ∃ (F : ℕ), F = 26 ∧ 
  (total_students - play_neither) = (F - both_sports) + (long_tennis_players - both_sports) + both_sports :=
by {
  sorry
}

end NUMINAMATH_GPT_football_players_count_l1817_181705


namespace NUMINAMATH_GPT_solution_set_l1817_181799

theorem solution_set (x : ℝ) : (2 : ℝ) ^ (|x-2| + |x-4|) > 2^6 ↔ x < 0 ∨ x > 6 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1817_181799
