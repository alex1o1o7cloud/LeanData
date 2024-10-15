import Mathlib

namespace NUMINAMATH_GPT_find_speed_of_man_l901_90168

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
(v_m + v_s = 6) ∧ (v_m - v_s = 8)

theorem find_speed_of_man :
  ∃ v_m v_s : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_man_l901_90168


namespace NUMINAMATH_GPT_orange_shells_correct_l901_90123

def total_shells : Nat := 65
def purple_shells : Nat := 13
def pink_shells : Nat := 8
def yellow_shells : Nat := 18
def blue_shells : Nat := 12
def orange_shells : Nat := total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)

theorem orange_shells_correct : orange_shells = 14 :=
by
  sorry

end NUMINAMATH_GPT_orange_shells_correct_l901_90123


namespace NUMINAMATH_GPT_ticket_cost_per_ride_l901_90182

theorem ticket_cost_per_ride
  (total_tickets: ℕ) 
  (spent_tickets: ℕ)
  (rides: ℕ)
  (remaining_tickets: ℕ)
  (ride_cost: ℕ)
  (h1: total_tickets = 79)
  (h2: spent_tickets = 23)
  (h3: rides = 8)
  (h4: remaining_tickets = total_tickets - spent_tickets)
  (h5: remaining_tickets = ride_cost * rides):
  ride_cost = 7 :=
by
  sorry

end NUMINAMATH_GPT_ticket_cost_per_ride_l901_90182


namespace NUMINAMATH_GPT_person_b_lap_time_l901_90184

noncomputable def lap_time_b (a_lap_time : ℕ) (meet_time : ℕ) : ℕ :=
  let combined_speed := 1 / meet_time
  let a_speed := 1 / a_lap_time
  let b_speed := combined_speed - a_speed
  1 / b_speed

theorem person_b_lap_time 
  (a_lap_time : ℕ) 
  (meet_time : ℕ) 
  (h1 : a_lap_time = 80) 
  (h2 : meet_time = 30) : 
  lap_time_b a_lap_time meet_time = 48 := 
by 
  rw [lap_time_b, h1, h2]
  -- Provided steps to solve the proof, skipped here only for statement
  sorry

end NUMINAMATH_GPT_person_b_lap_time_l901_90184


namespace NUMINAMATH_GPT_max_value_fraction_l901_90120

theorem max_value_fraction (a b : ℝ)
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  (a ≠ 0) → (b ≠ 0) →
  ∃ m, m = (a + 2 * b) / (2 * a + b) ∧ m ≤ 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_l901_90120


namespace NUMINAMATH_GPT_lunch_to_novel_ratio_l901_90174

theorem lunch_to_novel_ratio 
  (initial_amount : ℕ) 
  (novel_cost : ℕ) 
  (remaining_after_mall : ℕ) 
  (spent_on_lunch : ℕ)
  (h1 : initial_amount = 50) 
  (h2 : novel_cost = 7) 
  (h3 : remaining_after_mall = 29) 
  (h4 : spent_on_lunch = initial_amount - novel_cost - remaining_after_mall) :
  spent_on_lunch / novel_cost = 2 := 
  sorry

end NUMINAMATH_GPT_lunch_to_novel_ratio_l901_90174


namespace NUMINAMATH_GPT_k5_possibility_l901_90139

noncomputable def possible_k5 : Prop :=
  ∃ (intersections : Fin 5 → Fin 5 × Fin 10), 
    ∀ i j : Fin 5, i ≠ j → intersections i ≠ intersections j

theorem k5_possibility : possible_k5 := 
by
  sorry

end NUMINAMATH_GPT_k5_possibility_l901_90139


namespace NUMINAMATH_GPT_problem_statement_l901_90106

noncomputable def g (x : ℝ) : ℝ :=
  sorry

theorem problem_statement : (∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 + x^2) → g 3 = -201 / 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_statement_l901_90106


namespace NUMINAMATH_GPT_solve_inequality_solution_set_l901_90129

def solution_set (x : ℝ) : Prop := -x^2 + 5 * x > 6

theorem solve_inequality_solution_set :
  { x : ℝ | solution_set x } = { x : ℝ | 2 < x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_solve_inequality_solution_set_l901_90129


namespace NUMINAMATH_GPT_cos_product_inequality_l901_90162

theorem cos_product_inequality : (1 / 8 : ℝ) < (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) ∧
    (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_cos_product_inequality_l901_90162


namespace NUMINAMATH_GPT_determine_x_l901_90110

theorem determine_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^3) (h3 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l901_90110


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l901_90189

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ)
variable (h_arith : is_arithmetic_sequence a)
variable (h_sum : a 2 + a 3 + a 10 + a 11 = 48)

-- Goal
theorem arithmetic_sequence_sum : a 6 + a 7 = 24 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l901_90189


namespace NUMINAMATH_GPT_total_coins_l901_90152

theorem total_coins (q1 q2 q3 q4 : Nat) (d1 d2 d3 : Nat) (n1 n2 : Nat) (p1 p2 p3 p4 p5 : Nat) :
  q1 = 8 → q2 = 6 → q3 = 7 → q4 = 5 →
  d1 = 7 → d2 = 5 → d3 = 9 →
  n1 = 4 → n2 = 6 →
  p1 = 10 → p2 = 3 → p3 = 8 → p4 = 2 → p5 = 13 →
  q1 + q2 + q3 + q4 + d1 + d2 + d3 + n1 + n2 + p1 + p2 + p3 + p4 + p5 = 93 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_coins_l901_90152


namespace NUMINAMATH_GPT_parallelogram_base_length_l901_90156

theorem parallelogram_base_length (A : ℕ) (h b : ℕ) (h1 : A = b * h) (h2 : h = 2 * b) (h3 : A = 200) : b = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallelogram_base_length_l901_90156


namespace NUMINAMATH_GPT_zero_in_set_l901_90114

theorem zero_in_set : 0 ∈ ({0, 1, 2} : Set Nat) := 
sorry

end NUMINAMATH_GPT_zero_in_set_l901_90114


namespace NUMINAMATH_GPT_twelfth_term_arithmetic_sequence_l901_90108

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end NUMINAMATH_GPT_twelfth_term_arithmetic_sequence_l901_90108


namespace NUMINAMATH_GPT_hannah_bought_two_sets_of_measuring_spoons_l901_90143

-- Definitions of conditions
def number_of_cookies_sold : ℕ := 40
def price_per_cookie : ℝ := 0.8
def number_of_cupcakes_sold : ℕ := 30
def price_per_cupcake : ℝ := 2.0
def cost_per_measuring_spoon_set : ℝ := 6.5
def remaining_money : ℝ := 79

-- Definition of total money made from selling cookies and cupcakes
def total_money_made : ℝ := (number_of_cookies_sold * price_per_cookie) + (number_of_cupcakes_sold * price_per_cupcake)

-- Definition of money spent on measuring spoons
def money_spent_on_measuring_spoons : ℝ := total_money_made - remaining_money

-- Theorem statement
theorem hannah_bought_two_sets_of_measuring_spoons :
  (money_spent_on_measuring_spoons / cost_per_measuring_spoon_set) = 2 := by
  sorry

end NUMINAMATH_GPT_hannah_bought_two_sets_of_measuring_spoons_l901_90143


namespace NUMINAMATH_GPT_equivalent_problem_l901_90109

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem equivalent_problem
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h2 : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 := 
sorry

end NUMINAMATH_GPT_equivalent_problem_l901_90109


namespace NUMINAMATH_GPT_part_a_l901_90166

-- Define the sequences and their properties
variables {n : ℕ} (h1 : n ≥ 3)
variables (a b : ℕ → ℝ)
variables (h_arith : ∀ k, a (k+1) = a k + d)
variables (h_geom : ∀ k, b (k+1) = b k * q)
variables (h_a1_b1 : a 1 = b 1)
variables (h_an_bn : a n = b n)

-- State the theorem to be proven
theorem part_a (k : ℕ) (h_k : 2 ≤ k ∧ k ≤ n - 1) : a k > b k :=
  sorry

end NUMINAMATH_GPT_part_a_l901_90166


namespace NUMINAMATH_GPT_find_common_ratio_l901_90187

variable {a : ℕ → ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

theorem find_common_ratio (h : geometric_sequence_q a q) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l901_90187


namespace NUMINAMATH_GPT_mean_squared_sum_l901_90175

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem mean_squared_sum :
  (x + y + z = 30) ∧ 
  (xyz = 125) ∧ 
  ((1 / x + 1 / y + 1 / z) = 3 / 4) 
  → x^2 + y^2 + z^2 = 712.5 :=
by
  intros h
  have h₁ : x + y + z = 30 := h.1
  have h₂ : xyz = 125 := h.2.1
  have h₃ : (1 / x + 1 / y + 1 / z) = 3 / 4 := h.2.2
  sorry

end NUMINAMATH_GPT_mean_squared_sum_l901_90175


namespace NUMINAMATH_GPT_new_average_after_doubling_l901_90125

theorem new_average_after_doubling (n : ℕ) (avg : ℝ) (h_n : n = 12) (h_avg : avg = 50) :
  2 * avg = 100 :=
by
  sorry

end NUMINAMATH_GPT_new_average_after_doubling_l901_90125


namespace NUMINAMATH_GPT_sum_of_largest_and_smallest_odd_numbers_is_16_l901_90146

-- Define odd numbers between 5 and 12
def odd_numbers_set := {n | 5 ≤ n ∧ n ≤ 12 ∧ n % 2 = 1}

-- Define smallest odd number from the set
def min_odd := 5

-- Define largest odd number from the set
def max_odd := 11

-- The main theorem stating that the sum of the smallest and largest odd numbers is 16
theorem sum_of_largest_and_smallest_odd_numbers_is_16 :
  min_odd + max_odd = 16 := by
  sorry

end NUMINAMATH_GPT_sum_of_largest_and_smallest_odd_numbers_is_16_l901_90146


namespace NUMINAMATH_GPT_max_area_curves_intersection_l901_90165

open Real

def C₁ (x : ℝ) : ℝ := x^3 - x
def C₂ (x a : ℝ) : ℝ := (x - a)^3 - (x - a)

theorem max_area_curves_intersection (a : ℝ) (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ = C₂ x₁ a ∧ C₁ x₂ = C₂ x₂ a) :
  ∃ A_max : ℝ, A_max = 3 / 4 :=
by
  -- TODO: Provide the proof here
  sorry

end NUMINAMATH_GPT_max_area_curves_intersection_l901_90165


namespace NUMINAMATH_GPT_no_base_450_odd_last_digit_l901_90155

theorem no_base_450_odd_last_digit :
  ¬ ∃ b : ℕ, b^3 ≤ 450 ∧ 450 < b^4 ∧ (450 % b) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_no_base_450_odd_last_digit_l901_90155


namespace NUMINAMATH_GPT_total_number_of_workers_l901_90159

theorem total_number_of_workers (W N : ℕ) 
    (avg_all : ℝ) 
    (avg_technicians : ℝ) 
    (avg_non_technicians : ℝ)
    (h1 : avg_all = 8000)
    (h2 : avg_technicians = 20000)
    (h3 : avg_non_technicians = 6000)
    (h4 : 7 * avg_technicians + N * avg_non_technicians = (7 + N) * avg_all) :
  W = 49 := by
  sorry

end NUMINAMATH_GPT_total_number_of_workers_l901_90159


namespace NUMINAMATH_GPT_find_n_given_combination_l901_90199

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem find_n_given_combination : ∃ n : ℕ, binomial_coefficient (n+1) 2 = 21 ↔ n = 6 := by
  sorry

end NUMINAMATH_GPT_find_n_given_combination_l901_90199


namespace NUMINAMATH_GPT_sum_X_Y_Z_W_eq_156_l901_90190

theorem sum_X_Y_Z_W_eq_156 
  (X Y Z W : ℕ) 
  (h_arith_seq : Y - X = Z - Y)
  (h_geom_seq : Z / Y = 9 / 5)
  (h_W : W = Z^2 / Y) 
  (h_pos : 0 < X ∧ 0 < Y ∧ 0 < Z ∧ 0 < W) :
  X + Y + Z + W = 156 :=
sorry

end NUMINAMATH_GPT_sum_X_Y_Z_W_eq_156_l901_90190


namespace NUMINAMATH_GPT_petya_cannot_have_equal_coins_l901_90136

theorem petya_cannot_have_equal_coins
  (transact : ℕ → ℕ)
  (initial_two_kopeck : ℕ)
  (total_operations : ℕ)
  (insertion_machine : ℕ)
  (by_insert_two : ℕ)
  (by_insert_ten : ℕ)
  (odd : ℕ)
  :
  (initial_two_kopeck = 1) ∧ 
  (by_insert_two = 5) ∧ 
  (by_insert_ten = 5) ∧
  (∀ n, transact n = 1 + 4 * n) →
  (odd % 2 = 1) →
  (total_operations = transact insertion_machine) →
  (total_operations % 2 = 1) →
  (∀ x y, (x + y = total_operations) → (x = y) → False) :=
sorry

end NUMINAMATH_GPT_petya_cannot_have_equal_coins_l901_90136


namespace NUMINAMATH_GPT_simplify_fractions_l901_90171

-- Define the fractions and their product.
def fraction1 : ℚ := 14 / 3
def fraction2 : ℚ := 9 / -42

-- Define the product of the fractions with scalar multiplication by 5.
def product : ℚ := 5 * fraction1 * fraction2

-- The target theorem to prove the equivalence.
theorem simplify_fractions : product = -5 := 
sorry  -- Proof is omitted

end NUMINAMATH_GPT_simplify_fractions_l901_90171


namespace NUMINAMATH_GPT_determine_GH_l901_90141

-- Define a structure for a Tetrahedron with edge lengths as given conditions
structure Tetrahedron :=
  (EF FG EH FH EG GH : ℕ)

-- Instantiate the Tetrahedron with the given edge lengths
def tetrahedron_EFGH := Tetrahedron.mk 42 14 37 19 28 14

-- State the theorem
theorem determine_GH (t : Tetrahedron) (hEF : t.EF = 42) :
  t.GH = 14 :=
sorry

end NUMINAMATH_GPT_determine_GH_l901_90141


namespace NUMINAMATH_GPT_possible_last_three_digits_product_l901_90102

def lastThreeDigits (n : ℕ) : ℕ := n % 1000

theorem possible_last_three_digits_product (a b c : ℕ) (ha : a > 1000) (hb : b > 1000) (hc : c > 1000)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (a + c) % 10 = b % 10)
  (h3 : (b + c) % 10 = a % 10) :
  lastThreeDigits (a * b * c) = 0 ∨ lastThreeDigits (a * b * c) = 250 ∨ lastThreeDigits (a * b * c) = 500 ∨ lastThreeDigits (a * b * c) = 750 := 
sorry

end NUMINAMATH_GPT_possible_last_three_digits_product_l901_90102


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_cos_2x0_l901_90173

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sin x * Real.cos x + 2 * (Real.sqrt 3) * (Real.cos x)^2 - Real.sqrt 3

theorem smallest_positive_period_of_f :
  (∃ p > 0, ∀ x, f x = f (x + p)) ∧
  (∀ q > 0, (∀ x, f x = f (x + q)) -> q ≥ Real.pi) :=
sorry

theorem cos_2x0 (x0 : ℝ) (h0 : x0 ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h1 : f (x0 - Real.pi / 12) = 6 / 5) :
  Real.cos (2 * x0) = (3 - 4 * Real.sqrt 3) / 10 :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_cos_2x0_l901_90173


namespace NUMINAMATH_GPT_repeating_decimals_difference_l901_90140

theorem repeating_decimals_difference :
  let x := 234 / 999
  let y := 567 / 999
  let z := 891 / 999
  x - y - z = -408 / 333 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_difference_l901_90140


namespace NUMINAMATH_GPT_coordinates_of_D_l901_90163

-- Definitions of the points and translation conditions
def A : (ℝ × ℝ) := (-1, 4)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (4, 7)

theorem coordinates_of_D :
  ∃ (D : ℝ × ℝ), D = (1, 2) ∧
  ∀ (translate : ℝ × ℝ), translate = (C.1 - A.1, C.2 - A.2) → 
  D = (B.1 + translate.1, B.2 + translate.2) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_D_l901_90163


namespace NUMINAMATH_GPT_prove_p_l901_90124

variables {m n p : ℝ}

/-- Given points (m, n) and (m + p, n + 4) lie on the line 
   x = y / 2 - 2 / 5, prove p = 2.
-/
theorem prove_p (hmn : m = n / 2 - 2 / 5)
                (hmpn4 : m + p = (n + 4) / 2 - 2 / 5) : p = 2 := 
by
  sorry

end NUMINAMATH_GPT_prove_p_l901_90124


namespace NUMINAMATH_GPT_area_of_right_triangle_l901_90137

theorem area_of_right_triangle (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 10) (h_angle : angle = 45) :
  (1 / 2) * (5 * Real.sqrt 2) * (5 * Real.sqrt 2) = 25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_l901_90137


namespace NUMINAMATH_GPT_jan_skips_in_5_minutes_l901_90116

theorem jan_skips_in_5_minutes 
  (original_speed : ℕ)
  (time_in_minutes : ℕ)
  (doubled : ℕ)
  (new_speed : ℕ)
  (skips_in_5_minutes : ℕ) : 
  original_speed = 70 →
  doubled = 2 →
  new_speed = original_speed * doubled →
  time_in_minutes = 5 →
  skips_in_5_minutes = new_speed * time_in_minutes →
  skips_in_5_minutes = 700 :=
by
  intros 
  sorry

end NUMINAMATH_GPT_jan_skips_in_5_minutes_l901_90116


namespace NUMINAMATH_GPT_number_of_japanese_selectors_l901_90181

theorem number_of_japanese_selectors (F C J : ℕ) (h1 : J = 3 * C) (h2 : C = F + 15) (h3 : J + C + F = 165) : J = 108 :=
by
sorry

end NUMINAMATH_GPT_number_of_japanese_selectors_l901_90181


namespace NUMINAMATH_GPT_projection_v_w_l901_90138

noncomputable def vector_v : ℝ × ℝ := (3, 4)
noncomputable def vector_w : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := dot_product u v / dot_product v v
  (scalar * v.1, scalar * v.2)

theorem projection_v_w :
  proj vector_v vector_w = (4/5, -2/5) :=
sorry

end NUMINAMATH_GPT_projection_v_w_l901_90138


namespace NUMINAMATH_GPT_marble_distribution_l901_90158

theorem marble_distribution (a b c : ℚ) (h1 : a + b + c = 78) (h2 : a = 3 * b + 2) (h3 : b = c / 2) : 
  a = 40 ∧ b = 38 / 3 ∧ c = 76 / 3 :=
by
  sorry

end NUMINAMATH_GPT_marble_distribution_l901_90158


namespace NUMINAMATH_GPT_quadratic_root_in_interval_l901_90142

variable (a b c : ℝ)

theorem quadratic_root_in_interval 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_in_interval_l901_90142


namespace NUMINAMATH_GPT_tan_315_eq_neg1_l901_90192

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end NUMINAMATH_GPT_tan_315_eq_neg1_l901_90192


namespace NUMINAMATH_GPT_find_breadth_of_rectangle_l901_90113

theorem find_breadth_of_rectangle
  (L R S : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S ^ 2 = 625)
  (A : ℝ := 100)
  (h4 : A = L * B) :
  B = 10 := sorry

end NUMINAMATH_GPT_find_breadth_of_rectangle_l901_90113


namespace NUMINAMATH_GPT_ratio_female_democrats_l901_90180

theorem ratio_female_democrats (total_participants male_participants female_participants total_democrats female_democrats : ℕ)
  (h1 : total_participants = 750)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : total_democrats = total_participants / 3)
  (h4 : female_democrats = 125)
  (h5 : total_democrats = male_participants / 4 + female_democrats) :
  (female_democrats / female_participants : ℝ) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_female_democrats_l901_90180


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l901_90135

noncomputable section

-- Problem 1: Simplify the given expression
theorem simplify_expr1 (a b : ℝ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := 
by sorry

-- Problem 2: Simplify the given expression
theorem simplify_expr2 (x y : ℝ) : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y :=
by sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l901_90135


namespace NUMINAMATH_GPT_range_of_a_l901_90177

open Real

-- Definitions of the propositions p and q
def p (a : ℝ) : Prop := (2 - a > 0) ∧ (a + 1 > 0)

def discriminant (a : ℝ) : ℝ := 16 + 4 * a

def q (a : ℝ) : Prop := discriminant a ≥ 0

/--
Given propositions p and q defined above,
prove that the range of real number values for a 
such that ¬p ∧ q is true is
- 4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2
--/
theorem range_of_a (a : ℝ) : (¬ p a ∧ q a) → (-4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l901_90177


namespace NUMINAMATH_GPT_percentage_of_boy_scouts_with_signed_permission_slips_l901_90117

noncomputable def total_scouts : ℕ := 100 -- assume 100 scouts
noncomputable def total_signed_permission_slips : ℕ := 70 -- 70% of 100
noncomputable def boy_scouts : ℕ := 60 -- 60% of 100
noncomputable def girl_scouts : ℕ := 40 -- total_scouts - boy_scouts 

noncomputable def girl_scouts_signed_permission_slips : ℕ := girl_scouts * 625 / 1000 

theorem percentage_of_boy_scouts_with_signed_permission_slips :
  (boy_scouts * 75 / 100) = (total_signed_permission_slips - girl_scouts_signed_permission_slips) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_boy_scouts_with_signed_permission_slips_l901_90117


namespace NUMINAMATH_GPT_solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l901_90144

-- Definitions as conditions
def is_cone (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_cylinder (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_triangular_pyramid (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_rectangular_prism (solid : Type) : Prop := -- Definition placeholder
sorry 

-- Predicate to check if the front view of a solid is a quadrilateral
def front_view_is_quadrilateral (solid : Type) : Prop :=
  (is_cylinder solid ∨ is_rectangular_prism solid)

-- Theorem stating the problem
theorem solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism
    (s : Type) :
  front_view_is_quadrilateral s ↔ is_cylinder s ∨ is_rectangular_prism s :=
by
  sorry

end NUMINAMATH_GPT_solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l901_90144


namespace NUMINAMATH_GPT_janet_total_owed_l901_90188

def warehouseHourlyWage : ℝ := 15
def managerHourlyWage : ℝ := 20
def numWarehouseWorkers : ℕ := 4
def numManagers : ℕ := 2
def workDaysPerMonth : ℕ := 25
def workHoursPerDay : ℕ := 8
def ficaTaxRate : ℝ := 0.10

theorem janet_total_owed : 
  let warehouseWorkerMonthlyWage := warehouseHourlyWage * workDaysPerMonth * workHoursPerDay
  let managerMonthlyWage := managerHourlyWage * workDaysPerMonth * workHoursPerDay
  let totalMonthlyWages := (warehouseWorkerMonthlyWage * numWarehouseWorkers) + (managerMonthlyWage * numManagers)
  let ficaTaxes := totalMonthlyWages * ficaTaxRate
  let totalAmountOwed := totalMonthlyWages + ficaTaxes
  totalAmountOwed = 22000 := by
  sorry

end NUMINAMATH_GPT_janet_total_owed_l901_90188


namespace NUMINAMATH_GPT_volume_tetrahedral_region_is_correct_l901_90104

noncomputable def volume_of_tetrahedral_region (a : ℝ) : ℝ :=
  (81 - 8 * Real.pi) * a^3 / 486

theorem volume_tetrahedral_region_is_correct (a : ℝ) :
  volume_of_tetrahedral_region a = (81 - 8 * Real.pi) * a^3 / 486 :=
by
  sorry

end NUMINAMATH_GPT_volume_tetrahedral_region_is_correct_l901_90104


namespace NUMINAMATH_GPT_time_to_save_for_vehicle_l901_90157

def monthly_earnings : ℕ := 4000
def saving_factor : ℚ := 1 / 2
def vehicle_cost : ℕ := 16000

theorem time_to_save_for_vehicle : (vehicle_cost / (monthly_earnings * saving_factor)) = 8 := by
  sorry

end NUMINAMATH_GPT_time_to_save_for_vehicle_l901_90157


namespace NUMINAMATH_GPT_meaningful_fraction_l901_90178

theorem meaningful_fraction (x : ℝ) : (∃ (f : ℝ), f = 2 / x) ↔ x ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_l901_90178


namespace NUMINAMATH_GPT_total_lives_correct_l901_90105

-- Define the initial number of friends
def initial_friends : ℕ := 16

-- Define the number of lives each player has
def lives_per_player : ℕ := 10

-- Define the number of additional players that joined
def additional_players : ℕ := 4

-- Define the initial total number of lives
def initial_lives : ℕ := initial_friends * lives_per_player

-- Define the additional lives from the new players
def additional_lives : ℕ := additional_players * lives_per_player

-- Define the final total number of lives
def total_lives : ℕ := initial_lives + additional_lives

-- The proof goal
theorem total_lives_correct : total_lives = 200 :=
by
  -- This is where the proof would be written, but it is omitted.
  sorry

end NUMINAMATH_GPT_total_lives_correct_l901_90105


namespace NUMINAMATH_GPT_commission_percentage_l901_90195

theorem commission_percentage 
  (total_amount : ℝ) 
  (h1 : total_amount = 800) 
  (commission_first_500 : ℝ) 
  (h2 : commission_first_500 = 0.20 * 500) 
  (excess_amount : ℝ) 
  (h3 : excess_amount = (total_amount - 500)) 
  (commission_excess : ℝ) 
  (h4 : commission_excess = 0.25 * excess_amount) 
  (total_commission : ℝ) 
  (h5 : total_commission = commission_first_500 + commission_excess) 
  : (total_commission / total_amount) * 100 = 21.875 := 
by
  sorry

end NUMINAMATH_GPT_commission_percentage_l901_90195


namespace NUMINAMATH_GPT_f_has_one_zero_l901_90196

noncomputable def f (x : ℝ) : ℝ := 2 * x - 5 - Real.log x

theorem f_has_one_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_has_one_zero_l901_90196


namespace NUMINAMATH_GPT_range_of_f_l901_90193

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.cos x - 4 * Real.sin x

theorem range_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → f x ∈ Set.Icc (-5) 3) ∧
  (∀ y : ℝ, y ∈ Set.Icc (-5) 3 → ∃ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ f x = y) :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l901_90193


namespace NUMINAMATH_GPT_inequality_solution_absolute_inequality_l901_90191

-- Statement for Inequality Solution Problem
theorem inequality_solution (x : ℝ) : |x - 1| + |2 * x + 1| > 3 ↔ (x < -1 ∨ x > 1) := sorry

-- Statement for Absolute Inequality Problem with Bounds
theorem absolute_inequality (a b : ℝ) (ha : -1 ≤ a) (hb : a ≤ 1) (hc : -1 ≤ b) (hd : b ≤ 1) : 
  |1 + (a * b) / 4| > |(a + b) / 2| := sorry

end NUMINAMATH_GPT_inequality_solution_absolute_inequality_l901_90191


namespace NUMINAMATH_GPT_philip_paints_2_per_day_l901_90100

def paintings_per_day (initial_paintings total_paintings days : ℕ) : ℕ :=
  (total_paintings - initial_paintings) / days

theorem philip_paints_2_per_day :
  paintings_per_day 20 80 30 = 2 :=
by
  sorry

end NUMINAMATH_GPT_philip_paints_2_per_day_l901_90100


namespace NUMINAMATH_GPT_cookie_sheet_perimeter_l901_90161

theorem cookie_sheet_perimeter :
  let width_in_inches := 15.2
  let length_in_inches := 3.7
  let conversion_factor := 2.54
  let width_in_cm := width_in_inches * conversion_factor
  let length_in_cm := length_in_inches * conversion_factor
  2 * (width_in_cm + length_in_cm) = 96.012 :=
by
  sorry

end NUMINAMATH_GPT_cookie_sheet_perimeter_l901_90161


namespace NUMINAMATH_GPT_inequality_transitive_l901_90148

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by
  sorry

end NUMINAMATH_GPT_inequality_transitive_l901_90148


namespace NUMINAMATH_GPT_max_wx_plus_xy_plus_yz_l901_90112

theorem max_wx_plus_xy_plus_yz (w x y z : ℝ) (h1 : w ≥ 0) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) (h_sum : w + x + y + z = 200) : wx + xy + yz ≤ 10000 :=
sorry

end NUMINAMATH_GPT_max_wx_plus_xy_plus_yz_l901_90112


namespace NUMINAMATH_GPT_range_of_x_l901_90121

theorem range_of_x (x : ℝ) : (2 : ℝ)^(3 - 2 * x) < (2 : ℝ)^(3 * x - 4) → x > 7 / 5 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l901_90121


namespace NUMINAMATH_GPT_sine_triangle_sides_l901_90164

variable {α β γ : ℝ}

-- Given conditions: α, β, γ are angles of a triangle.
def is_triangle_angles (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi ∧
  0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi

-- The proof statement: Prove that there exists a triangle with sides sin α, sin β, sin γ
theorem sine_triangle_sides (h : is_triangle_angles α β γ) :
  ∃ (x y z : ℝ), x = Real.sin α ∧ y = Real.sin β ∧ z = Real.sin γ ∧
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x) := sorry

end NUMINAMATH_GPT_sine_triangle_sides_l901_90164


namespace NUMINAMATH_GPT_units_digit_of_pow_sum_is_correct_l901_90115

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end NUMINAMATH_GPT_units_digit_of_pow_sum_is_correct_l901_90115


namespace NUMINAMATH_GPT_general_formula_arithmetic_sequence_l901_90167

theorem general_formula_arithmetic_sequence :
  (∃ (a_n : ℕ → ℕ) (d : ℕ), d ≠ 0 ∧ 
    (a_2 = a_1 + d) ∧ 
    (a_4 = a_1 + 3 * d) ∧ 
    (a_2^2 = a_1 * a_4) ∧
    (a_5 = a_1 + 4 * d) ∧ 
    (a_6 = a_1 + 5 * d) ∧ 
    (a_5 + a_6 = 11) ∧ 
    ∀ n, a_n = a_1 + (n - 1) * d) → 
  ∀ n, a_n = n := 
sorry

end NUMINAMATH_GPT_general_formula_arithmetic_sequence_l901_90167


namespace NUMINAMATH_GPT_sin_theta_plus_2cos_theta_eq_zero_l901_90160

theorem sin_theta_plus_2cos_theta_eq_zero (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (1 + Real.sin (2 * θ)) / (Real.cos θ)^2 = 1 :=
  sorry

end NUMINAMATH_GPT_sin_theta_plus_2cos_theta_eq_zero_l901_90160


namespace NUMINAMATH_GPT_fresh_water_needed_l901_90134

noncomputable def mass_of_seawater : ℝ := 30
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def desired_salt_concentration : ℝ := 0.015

theorem fresh_water_needed :
  ∃ (fresh_water_mass : ℝ), 
    fresh_water_mass = 70 ∧ 
    (mass_of_seawater * initial_salt_concentration) / (mass_of_seawater + fresh_water_mass) = desired_salt_concentration :=
by
  sorry

end NUMINAMATH_GPT_fresh_water_needed_l901_90134


namespace NUMINAMATH_GPT_max_profit_jars_max_tax_value_l901_90145

-- Part a: Prove the optimal number of jars for maximum profit
theorem max_profit_jars (Q : ℝ) 
  (h : ∀ Q, Q >= 0 → (310 - 3 * Q) * Q - 10 * Q ≤ (310 - 3 * 50) * 50 - 10 * 50):
  Q = 50 :=
sorry

-- Part b: Prove the optimal tax for maximum tax revenue
theorem max_tax_value (t : ℝ) 
  (h : ∀ t, t >= 0 → ((300 * t - t^2) / 6) ≤ ((300 * 150 - 150^2) / 6)):
  t = 150 :=
sorry

end NUMINAMATH_GPT_max_profit_jars_max_tax_value_l901_90145


namespace NUMINAMATH_GPT_parallel_lines_condition_l901_90127

variable {a : ℝ}

theorem parallel_lines_condition (a_is_2 : a = 2) :
  (∀ x y : ℝ, a * x + 2 * y = 0 → x + y = 1) ∧ (∀ x y : ℝ, x + y = 1 → a * x + 2 * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l901_90127


namespace NUMINAMATH_GPT_equation_has_real_roots_for_all_K_l901_90101

open Real

noncomputable def original_equation (K x : ℝ) : ℝ :=
  x - K^3 * (x - 1) * (x - 3)

theorem equation_has_real_roots_for_all_K :
  ∀ K : ℝ, ∃ x : ℝ, original_equation K x = 0 :=
sorry

end NUMINAMATH_GPT_equation_has_real_roots_for_all_K_l901_90101


namespace NUMINAMATH_GPT_max_moves_440_l901_90172

-- Define the set of initial numbers
def initial_numbers : List ℕ := List.range' 1 22

-- Define what constitutes a valid move
def is_valid_move (a b : ℕ) : Prop := b ≥ a + 2

-- Perform the move operation
def perform_move (numbers : List ℕ) (a b : ℕ) : List ℕ :=
  (numbers.erase a).erase b ++ [a + 1, b - 1]

-- Define the maximum number of moves we need to prove
theorem max_moves_440 : ∃ m, m = 440 ∧
  ∀ (moves_done : ℕ) (numbers : List ℕ),
    moves_done <= m → ∃ a b, a ∈ numbers ∧ b ∈ numbers ∧
                             is_valid_move a b ∧
                             numbers = initial_numbers →
                             perform_move numbers a b ≠ numbers
 := sorry

end NUMINAMATH_GPT_max_moves_440_l901_90172


namespace NUMINAMATH_GPT_exponent_on_right_side_l901_90128

theorem exponent_on_right_side (n : ℕ) (k : ℕ) (h : n = 25) :
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^k → k = 26 :=
by
  sorry

end NUMINAMATH_GPT_exponent_on_right_side_l901_90128


namespace NUMINAMATH_GPT_total_dogs_l901_90119

-- Definitions of conditions
def brown_dogs : Nat := 20
def white_dogs : Nat := 10
def black_dogs : Nat := 15

-- Theorem to prove the total number of dogs
theorem total_dogs : brown_dogs + white_dogs + black_dogs = 45 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_total_dogs_l901_90119


namespace NUMINAMATH_GPT_probability_queen_in_center_after_2004_moves_l901_90170

def initial_probability (n : ℕ) : ℚ :=
if n = 0 then 1
else if n = 1 then 0
else if n % 2 = 0 then (1 : ℚ) / 2^(n / 2)
else (1 - (1 : ℚ) / 2^((n - 1) / 2)) / 2

theorem probability_queen_in_center_after_2004_moves :
  initial_probability 2004 = 1 / 3 + 1 / (3 * 2^2003) :=
sorry

end NUMINAMATH_GPT_probability_queen_in_center_after_2004_moves_l901_90170


namespace NUMINAMATH_GPT_vampire_daily_needs_l901_90198

theorem vampire_daily_needs :
  (7 * 8) / 2 / 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_vampire_daily_needs_l901_90198


namespace NUMINAMATH_GPT_trigonometric_identity_l901_90154

theorem trigonometric_identity :
  (1 / Real.cos (80 * (Real.pi / 180)) - Real.sqrt 3 / Real.sin (80 * (Real.pi / 180)) = 4) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l901_90154


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_l901_90111

theorem sum_of_number_and_reverse (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_l901_90111


namespace NUMINAMATH_GPT_function_property_l901_90133

theorem function_property (k a b : ℝ) (h : a ≠ b) (h_cond : k * a + 1 / b = k * b + 1 / a) : 
  k * (a * b) + 1 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_function_property_l901_90133


namespace NUMINAMATH_GPT_cloth_sales_worth_l901_90118

theorem cloth_sales_worth 
  (commission : ℝ) 
  (commission_rate : ℝ) 
  (commission_received : ℝ) 
  (commission_rate_of_sales : commission_rate = 2.5)
  (commission_received_rs : commission_received = 21) 
  : (commission_received / (commission_rate / 100)) = 840 :=
by
  sorry

end NUMINAMATH_GPT_cloth_sales_worth_l901_90118


namespace NUMINAMATH_GPT_sum_of_product_of_consecutive_numbers_divisible_by_12_l901_90183

theorem sum_of_product_of_consecutive_numbers_divisible_by_12 (a : ℤ) : 
  (a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1) % 12 = 0 :=
by sorry

end NUMINAMATH_GPT_sum_of_product_of_consecutive_numbers_divisible_by_12_l901_90183


namespace NUMINAMATH_GPT_arithmetic_progression_common_difference_and_first_terms_l901_90107

def sum (n : ℕ) : ℕ := 5 * n ^ 2
def Sn (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_common_difference_and_first_terms:
  ∀ n : ℕ, Sn 5 10 n = sum n :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_common_difference_and_first_terms_l901_90107


namespace NUMINAMATH_GPT_max_sin_a_l901_90103

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_max_sin_a_l901_90103


namespace NUMINAMATH_GPT_janet_saves_minutes_l901_90130

theorem janet_saves_minutes 
  (key_search_time : ℕ := 8) 
  (complaining_time : ℕ := 3) 
  (days_in_week : ℕ := 7) : 
  (key_search_time + complaining_time) * days_in_week = 77 := 
by
  sorry

end NUMINAMATH_GPT_janet_saves_minutes_l901_90130


namespace NUMINAMATH_GPT_value_of_v_star_star_l901_90185

noncomputable def v_star (v : ℝ) : ℝ :=
  v - v / 3
  
theorem value_of_v_star_star (v : ℝ) (h : v = 8.999999999999998) : v_star (v_star v) = 4.000000000000000 := by
  sorry

end NUMINAMATH_GPT_value_of_v_star_star_l901_90185


namespace NUMINAMATH_GPT_min_value_frac_inv_l901_90132

theorem min_value_frac_inv {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (∃ m, (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 2 → m ≤ (1 / x + 1 / y)) ∧ (m = 2)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_inv_l901_90132


namespace NUMINAMATH_GPT_different_books_l901_90131

-- Definitions for the conditions
def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def common_books_tony_dean : ℕ := 3
def common_books_all : ℕ := 1

-- Prove the total number of different books they have read is 47
theorem different_books : (tony_books + dean_books - common_books_tony_dean + breanna_books - 2 * common_books_all) = 47 :=
by
  sorry

end NUMINAMATH_GPT_different_books_l901_90131


namespace NUMINAMATH_GPT_variance_of_remaining_scores_l901_90153

def scores : List ℕ := [91, 89, 91, 96, 94, 95, 94]

def remaining_scores : List ℕ := [91, 91, 94, 95, 94]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_of_remaining_scores :
  variance remaining_scores = 2.8 := by
  sorry

end NUMINAMATH_GPT_variance_of_remaining_scores_l901_90153


namespace NUMINAMATH_GPT_total_peaches_l901_90147

variable {n m : ℕ}

-- conditions
def equal_subgroups (n : ℕ) := (n % 3 = 0)

def condition_1 (n m : ℕ) := (m - 27) % n = 0 ∧ (m - 27) / n = 5

def condition_2 (n m : ℕ) : Prop := 
  ∃ x : ℕ, 0 < x ∧ x < 7 ∧ (m - x) % n = 0 ∧ ((m - x) / n = 7) 

-- theorem to be proved
theorem total_peaches (n m : ℕ) (h1 : equal_subgroups n) (h2 : condition_1 n m) (h3 : condition_2 n m) : m = 102 := 
sorry

end NUMINAMATH_GPT_total_peaches_l901_90147


namespace NUMINAMATH_GPT_part_I_extreme_value_part_II_range_of_a_l901_90151

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + Real.log x + 1

theorem part_I_extreme_value (a : ℝ) (h1 : a = -1/4) :
  (∀ x > 0, f a x ≤ f a 2) ∧ f a 2 = 3/4 + Real.log 2 :=
sorry

theorem part_II_range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a x ≤ x) ↔ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_part_I_extreme_value_part_II_range_of_a_l901_90151


namespace NUMINAMATH_GPT_how_many_did_not_play_l901_90150

def initial_players : ℕ := 40
def first_half_starters : ℕ := 11
def first_half_substitutions : ℕ := 4
def second_half_extra_substitutions : ℕ := (first_half_substitutions * 3) / 4 -- 75% more substitutions
def injury_substitution : ℕ := 1
def total_second_half_substitutions : ℕ := first_half_substitutions + second_half_extra_substitutions + injury_substitution
def total_players_played : ℕ := first_half_starters + first_half_substitutions + total_second_half_substitutions
def players_did_not_play : ℕ := initial_players - total_players_played

theorem how_many_did_not_play : players_did_not_play = 17 := by
  sorry

end NUMINAMATH_GPT_how_many_did_not_play_l901_90150


namespace NUMINAMATH_GPT_max_value_of_expression_l901_90126

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 = z^2) :
  ∃ t, t = (3 * Real.sqrt 2) / 2 ∧ ∀ u, u = (x + 2 * y) / z → u ≤ t := by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l901_90126


namespace NUMINAMATH_GPT_solve_system_l901_90194

theorem solve_system :
  {p : ℝ × ℝ // 
    (p.1 + |p.2| = 3 ∧ 2 * |p.1| - p.2 = 3) ∧
    (p = (2, 1) ∨ p = (0, -3) ∨ p = (-6, 9))} :=
by { sorry }

end NUMINAMATH_GPT_solve_system_l901_90194


namespace NUMINAMATH_GPT_cricket_players_count_l901_90179

theorem cricket_players_count (hockey: ℕ) (football: ℕ) (softball: ℕ) (total: ℕ) : 
  hockey = 15 ∧ football = 21 ∧ softball = 19 ∧ total = 77 → ∃ cricket, cricket = 22 := by
  sorry

end NUMINAMATH_GPT_cricket_players_count_l901_90179


namespace NUMINAMATH_GPT_batsman_average_46_innings_l901_90122

variable (A : ℕ) (highest_score : ℕ) (lowest_score : ℕ) (average_excl : ℕ)
variable (n_innings n_without_highest_lowest : ℕ)

theorem batsman_average_46_innings
  (h_diff: highest_score - lowest_score = 190)
  (h_avg_excl: average_excl = 58)
  (h_highest: highest_score = 199)
  (h_innings: n_innings = 46)
  (h_innings_excl: n_without_highest_lowest = 44) :
  A = (44 * 58 + 199 + 9) / 46 := by
  sorry

end NUMINAMATH_GPT_batsman_average_46_innings_l901_90122


namespace NUMINAMATH_GPT_circle_radius_d_l901_90149

theorem circle_radius_d (d : ℝ) : ∀ (x y : ℝ), (x^2 + 8 * x + y^2 + 2 * y + d = 0) → (∃ r : ℝ, r = 5) → d = -8 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_d_l901_90149


namespace NUMINAMATH_GPT_smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l901_90176

theorem smallest_four_digit_palindrome_div_by_3_with_odd_first_digit :
  ∃ (n : ℕ), (∃ A B : ℕ, n = 1001 * A + 110 * B ∧ 1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ A % 2 = 1) ∧ 3 ∣ n ∧ n = 1221 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l901_90176


namespace NUMINAMATH_GPT_pencil_lead_loss_l901_90186

theorem pencil_lead_loss (L r : ℝ) (h : r = L * 1/10):
  ((9/10 * r^3) * (2/3)) / (r^3) = 3/5 := 
by
  sorry

end NUMINAMATH_GPT_pencil_lead_loss_l901_90186


namespace NUMINAMATH_GPT_at_least_one_equation_has_real_roots_l901_90169

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0)

theorem at_least_one_equation_has_real_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  has_two_distinct_real_roots a b c :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_equation_has_real_roots_l901_90169


namespace NUMINAMATH_GPT_men_sent_to_other_project_l901_90197

-- Let the initial number of men be 50
def initial_men : ℕ := 50
-- Let the time to complete the work initially be 10 days
def initial_days : ℕ := 10
-- Calculate the total work in man-days
def total_work : ℕ := initial_men * initial_days

-- Let the total time taken after sending some men to another project be 30 days
def new_days : ℕ := 30
-- Let the number of men sent to another project be x
variable (x : ℕ)
-- Let the new number of men be (initial_men - x)
def new_men : ℕ := initial_men - x

theorem men_sent_to_other_project (x : ℕ):
total_work = new_men x * new_days -> x = 33 :=
by
  sorry

end NUMINAMATH_GPT_men_sent_to_other_project_l901_90197
