import Mathlib

namespace NUMINAMATH_GPT_fraction_square_equality_l2183_218344

theorem fraction_square_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
    (h : a / b + c / d = 1) : 
    (a / b)^2 + c / d = (c / d)^2 + a / b :=
by
  sorry

end NUMINAMATH_GPT_fraction_square_equality_l2183_218344


namespace NUMINAMATH_GPT_inequality_correctness_l2183_218305

theorem inequality_correctness (a b c : ℝ) (h : c^2 > 0) : (a * c^2 > b * c^2) ↔ (a > b) := by 
sorry

end NUMINAMATH_GPT_inequality_correctness_l2183_218305


namespace NUMINAMATH_GPT_sum_expr_le_e4_l2183_218382

theorem sum_expr_le_e4
  (α β γ δ ε : ℝ) :
  (1 - α) * Real.exp α +
  (1 - β) * Real.exp (α + β) +
  (1 - γ) * Real.exp (α + β + γ) +
  (1 - δ) * Real.exp (α + β + γ + δ) +
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 :=
sorry

end NUMINAMATH_GPT_sum_expr_le_e4_l2183_218382


namespace NUMINAMATH_GPT_quadratic_roots_l2183_218340

-- Define the given conditions of the equation
def eqn (z : ℂ) : Prop := z^2 + 2 * z + (3 - 4 * Complex.I) = 0

-- State the theorem to prove that the roots of the equation are 2i and -2 + 2i.
theorem quadratic_roots :
  ∃ z1 z2 : ℂ, (z1 = 2 * Complex.I ∧ z2 = -2 + 2 * Complex.I) ∧ 
  (∀ z : ℂ, eqn z → z = z1 ∨ z = z2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l2183_218340


namespace NUMINAMATH_GPT_part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l2183_218364

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - Real.log x

theorem part1_min_value_of_f_when_a_is_1 : 
  (∃ x : ℝ, f 1 x = 1 / 2 ∧ (∀ y : ℝ, f 1 y ≥ f 1 x)) :=
sorry

theorem part2_range_of_a_for_f_ge_x :
  (∀ x : ℝ, x > 0 → f a x ≥ x) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_GPT_part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l2183_218364


namespace NUMINAMATH_GPT_chess_club_probability_l2183_218388

theorem chess_club_probability :
  let total_members := 20
  let boys := 12
  let girls := 8
  let total_ways := Nat.choose total_members 4
  let all_boys := Nat.choose boys 4
  let all_girls := Nat.choose girls 4
  total_ways ≠ 0 → 
  (1 - (all_boys + all_girls) / total_ways) = (4280 / 4845) :=
by
  sorry

end NUMINAMATH_GPT_chess_club_probability_l2183_218388


namespace NUMINAMATH_GPT_find_square_tiles_l2183_218328

theorem find_square_tiles (t s p : ℕ) (h1 : t + s + p = 35) (h2 : 3 * t + 4 * s + 5 * p = 140) (hp0 : p = 0) : s = 35 := by
  sorry

end NUMINAMATH_GPT_find_square_tiles_l2183_218328


namespace NUMINAMATH_GPT_scientific_notation_141260_million_l2183_218356

theorem scientific_notation_141260_million :
  (141260 * 10^6 : ℝ) = 1.4126 * 10^11 := 
sorry

end NUMINAMATH_GPT_scientific_notation_141260_million_l2183_218356


namespace NUMINAMATH_GPT_farmer_plant_beds_l2183_218321

theorem farmer_plant_beds :
  ∀ (bean_seedlings pumpkin_seeds radishes seedlings_per_row_pumpkin seedlings_per_row_radish radish_rows_per_bed : ℕ),
    bean_seedlings = 64 →
    seedlings_per_row_pumpkin = 7 →
    pumpkin_seeds = 84 →
    seedlings_per_row_radish = 6 →
    radish_rows_per_bed = 2 →
    (bean_seedlings / 8 + pumpkin_seeds / seedlings_per_row_pumpkin + radishes / seedlings_per_row_radish) / radish_rows_per_bed = 14 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_farmer_plant_beds_l2183_218321


namespace NUMINAMATH_GPT_fewest_four_dollar_frisbees_l2183_218371

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 60) (h2 : 3 * x + 4 * y = 200) : y = 20 :=
by 
  sorry  

end NUMINAMATH_GPT_fewest_four_dollar_frisbees_l2183_218371


namespace NUMINAMATH_GPT_line_through_parabola_no_intersection_l2183_218369

-- Definitions of the conditions
def parabola (x : ℝ) : ℝ := x^2 
def point_Q := (10, 5)

-- The main theorem statement
theorem line_through_parabola_no_intersection :
  ∃ r s : ℝ, (∀ (m : ℝ), (r < m ∧ m < s) ↔ ¬ ∃ x : ℝ, parabola x = m * (x - 10) + 5) ∧ r + s = 40 :=
sorry

end NUMINAMATH_GPT_line_through_parabola_no_intersection_l2183_218369


namespace NUMINAMATH_GPT_geometric_sequence_ninth_tenth_term_sum_l2183_218332

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^n

theorem geometric_sequence_ninth_tenth_term_sum (a₁ q : ℝ)
  (h1 : a₁ + a₁ * q = 2)
  (h5 : a₁ * q^4 + a₁ * q^5 = 4) :
  geometric_sequence a₁ q 8 + geometric_sequence a₁ q 9 = 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ninth_tenth_term_sum_l2183_218332


namespace NUMINAMATH_GPT_no_solution_absval_equation_l2183_218366

theorem no_solution_absval_equation (x : ℝ) : ¬ (|2*x - 5| = 3*x + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_absval_equation_l2183_218366


namespace NUMINAMATH_GPT_find_n_value_l2183_218335

theorem find_n_value : (15 * 25 + 20 * 5) = (10 * 25 + 45 * 5) := 
  sorry

end NUMINAMATH_GPT_find_n_value_l2183_218335


namespace NUMINAMATH_GPT_return_percentage_is_6_5_l2183_218370

def investment1 : ℤ := 16250
def investment2 : ℤ := 16250
def profit_percentage1 : ℚ := 0.15
def loss_percentage2 : ℚ := 0.05
def total_investment : ℤ := 25000
def net_income : ℚ := investment1 * profit_percentage1 - investment2 * loss_percentage2
def return_percentage : ℚ := (net_income / total_investment) * 100

theorem return_percentage_is_6_5 : return_percentage = 6.5 := by
  sorry

end NUMINAMATH_GPT_return_percentage_is_6_5_l2183_218370


namespace NUMINAMATH_GPT_sum_max_min_ratio_l2183_218361

theorem sum_max_min_ratio (x y : ℝ) 
  (h_ellipse : 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0) 
  : (∃ m_max m_min : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0 → y = m_max * x ∨ y = m_min * x) ∧ (m_max + m_min = 37 / 22)) :=
sorry

end NUMINAMATH_GPT_sum_max_min_ratio_l2183_218361


namespace NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_over_2_l2183_218380

theorem sin_600_eq_neg_sqrt_3_over_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_over_2_l2183_218380


namespace NUMINAMATH_GPT_golf_ratio_l2183_218377

-- Definitions based on conditions
def first_turn_distance : ℕ := 180
def excess_distance : ℕ := 20
def total_distance_to_hole : ℕ := 250

-- Derived definitions based on conditions
def second_turn_distance : ℕ := (total_distance_to_hole - first_turn_distance) + excess_distance

-- Lean proof problem statement
theorem golf_ratio : (second_turn_distance : ℚ) / first_turn_distance = 1 / 2 :=
by
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_golf_ratio_l2183_218377


namespace NUMINAMATH_GPT_sum_of_possible_values_l2183_218349

-- Define the triangle's base and height
def triangle_base (x : ℝ) : ℝ := x - 2
def triangle_height (x : ℝ) : ℝ := x - 2

-- Define the parallelogram's base and height
def parallelogram_base (x : ℝ) : ℝ := x - 3
def parallelogram_height (x : ℝ) : ℝ := x + 4

-- Define the areas
def triangle_area (x : ℝ) : ℝ := 0.5 * (triangle_base x) * (triangle_height x)
def parallelogram_area (x : ℝ) : ℝ := (parallelogram_base x) * (parallelogram_height x)

-- Statement to prove
theorem sum_of_possible_values (x : ℝ) (h : parallelogram_area x = 3 * triangle_area x) : x = 8 ∨ x = 3 →
  (x = 8 ∨ x = 3) → 8 + 3 = 11 :=
by sorry

end NUMINAMATH_GPT_sum_of_possible_values_l2183_218349


namespace NUMINAMATH_GPT_find_S_l2183_218319

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S) ^ 2) (h2 : S > 0) :
  S = 333332 := 
sorry

end NUMINAMATH_GPT_find_S_l2183_218319


namespace NUMINAMATH_GPT_solve_equation_l2183_218365

open Real

noncomputable def f (x : ℝ) := 2017 * x ^ 2017 - 2017 + x
noncomputable def g (x : ℝ) := (2018 - 2017 * x) ^ (1 / 2017 : ℝ)

theorem solve_equation :
  ∀ x : ℝ, 2017 * x ^ 2017 - 2017 + x = (2018 - 2017 * x) ^ (1 / 2017 : ℝ) → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2183_218365


namespace NUMINAMATH_GPT_mul_99_101_equals_9999_l2183_218383

theorem mul_99_101_equals_9999 : 99 * 101 = 9999 := by
  sorry

end NUMINAMATH_GPT_mul_99_101_equals_9999_l2183_218383


namespace NUMINAMATH_GPT_dice_faces_l2183_218341

theorem dice_faces (n : ℕ) (h : (1 / (n : ℝ)) ^ 5 = 0.0007716049382716049) : n = 10 := sorry

end NUMINAMATH_GPT_dice_faces_l2183_218341


namespace NUMINAMATH_GPT_total_tickets_sold_l2183_218363

theorem total_tickets_sold 
  (A D : ℕ) 
  (cost_adv cost_door : ℝ) 
  (revenue : ℝ)
  (door_tickets_sold total_tickets : ℕ) 
  (h1 : cost_adv = 14.50) 
  (h2 : cost_door = 22.00)
  (h3 : revenue = 16640) 
  (h4 : door_tickets_sold = 672) : 
  (total_tickets = 800) :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2183_218363


namespace NUMINAMATH_GPT_parabola_sum_vertex_point_l2183_218330

theorem parabola_sum_vertex_point
  (a b c : ℝ)
  (h_vertex : ∀ y : ℝ, y = -6 → x = a * (y + 6)^2 + 8)
  (h_point : x = a * ((-4) + 6)^2 + 8)
  (ha : a = 0.5)
  (hb : b = 6)
  (hc : c = 26) :
  a + b + c = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_parabola_sum_vertex_point_l2183_218330


namespace NUMINAMATH_GPT_students_participated_l2183_218324

theorem students_participated (like_dislike_sum : 383 + 431 = 814) : 
  383 + 431 = 814 := 
by exact like_dislike_sum

end NUMINAMATH_GPT_students_participated_l2183_218324


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_l2183_218396

-- State the main problem in Lean 4
theorem solve_arithmetic_sequence (y : ℝ) (h : y^2 = (4 + 16) / 2) (hy : y > 0) : y = Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_l2183_218396


namespace NUMINAMATH_GPT_simplest_form_option_l2183_218354

theorem simplest_form_option (x y : ℚ) :
  (∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (12 * (x - y) / (15 * (x + y)) ≠ 4 * (x - y) / 5 * (x + y))) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 + y^2) / (x + y) = a / b) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / ((x + y)^2) ≠ (x - y) / (x + y)) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / (x + y) ≠ x - y)) := sorry

end NUMINAMATH_GPT_simplest_form_option_l2183_218354


namespace NUMINAMATH_GPT_expected_absolute_deviation_greater_in_10_tosses_l2183_218342

variable {n m : ℕ}

def frequency_of_heads (m n : ℕ) : ℚ := m / n

def deviation_from_probability (m n : ℕ) : ℚ :=
  m / n - 0.5

def absolute_deviation (m n : ℕ) : ℚ :=
  |m / n - 0.5|

noncomputable def expectation_absolute_deviation (n : ℕ) : ℚ := 
  if n = 10 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 10 tosses
    sorry 
  else if n = 100 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 100 tosses
    sorry 
  else 0

theorem expected_absolute_deviation_greater_in_10_tosses (a b : ℚ) :
  expectation_absolute_deviation 10 > expectation_absolute_deviation 100 :=
by sorry

end NUMINAMATH_GPT_expected_absolute_deviation_greater_in_10_tosses_l2183_218342


namespace NUMINAMATH_GPT_negation_of_universal_prop_l2183_218314

theorem negation_of_universal_prop : 
  (¬ (∀ (x : ℝ), x ^ 2 ≥ 0)) ↔ (∃ (x : ℝ), x ^ 2 < 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l2183_218314


namespace NUMINAMATH_GPT_logical_contradiction_l2183_218350

-- Definitions based on the conditions
def all_destroying (x : Type) : Prop := ∀ y : Type, y ≠ x → y → false
def indestructible (x : Type) : Prop := ∀ y : Type, y = x → y → false

theorem logical_contradiction (x : Type) :
  (all_destroying x ∧ indestructible x) → false :=
by
  sorry

end NUMINAMATH_GPT_logical_contradiction_l2183_218350


namespace NUMINAMATH_GPT_Sarah_books_in_8_hours_l2183_218301

theorem Sarah_books_in_8_hours (pages_per_hour: ℕ) (pages_per_book: ℕ) (hours_available: ℕ) 
  (h_pages_per_hour: pages_per_hour = 120) (h_pages_per_book: pages_per_book = 360) (h_hours_available: hours_available = 8) :
  hours_available * pages_per_hour / pages_per_book = 2 := by
  sorry

end NUMINAMATH_GPT_Sarah_books_in_8_hours_l2183_218301


namespace NUMINAMATH_GPT_pebbles_difference_l2183_218302

def candy_pebbles : Nat := 4
def lance_pebbles : Nat := 3 * candy_pebbles

theorem pebbles_difference {candy_pebbles lance_pebbles : Nat} (h1 : candy_pebbles = 4) (h2 : lance_pebbles = 3 * candy_pebbles) : lance_pebbles - candy_pebbles = 8 := by
  sorry

end NUMINAMATH_GPT_pebbles_difference_l2183_218302


namespace NUMINAMATH_GPT_parabola_focus_distance_l2183_218347

noncomputable def PF (x₁ : ℝ) : ℝ := x₁ + 1
noncomputable def QF (x₂ : ℝ) : ℝ := x₂ + 1

theorem parabola_focus_distance 
  (x₁ x₂ : ℝ) (h₁ : x₂ = 3 * x₁ + 2) : 
  QF x₂ / PF x₁ = 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l2183_218347


namespace NUMINAMATH_GPT_value_of_fraction_l2183_218338

variable (m n : ℚ)

theorem value_of_fraction (h₁ : 3 * m + 2 * n = 0) (h₂ : m ≠ 0 ∧ n ≠ 0) :
  (m / n - n / m) = 5 / 6 := 
sorry

end NUMINAMATH_GPT_value_of_fraction_l2183_218338


namespace NUMINAMATH_GPT_solve_equation_l2183_218368

theorem solve_equation : ∀ x : ℝ, (x + 1 - 2 * (x - 1) = 1 - 3 * x) → x = 0 := 
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_equation_l2183_218368


namespace NUMINAMATH_GPT_number_of_parents_l2183_218390

theorem number_of_parents (P : ℕ) (h : P + 177 = 238) : P = 61 :=
by
  sorry

end NUMINAMATH_GPT_number_of_parents_l2183_218390


namespace NUMINAMATH_GPT_minimum_width_for_fence_l2183_218322

theorem minimum_width_for_fence (w : ℝ) (h : 0 ≤ 20) : 
  (w * (w + 20) ≥ 150) → w ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_minimum_width_for_fence_l2183_218322


namespace NUMINAMATH_GPT_compressor_stations_l2183_218315

/-- 
Problem: Given three compressor stations connected by straight roads and not on the same line,
with distances satisfying:
1. x + y = 4z
2. x + z + y = x + a
3. z + y + x = 85

Prove:
- The range of values for 'a' such that the described configuration of compressor stations is 
  possible is 60.71 < a < 68.
- The distances between the compressor stations for a = 5 are x = 70, y = 0, z = 15.
--/
theorem compressor_stations (x y z a : ℝ) 
  (h1 : x + y = 4 * z)
  (h2 : x + z + y = x + a)
  (h3 : z + y + x = 85) :
  (60.71 < a ∧ a < 68) ∧ (a = 5 → x = 70 ∧ y = 0 ∧ z = 15) :=
  sorry

end NUMINAMATH_GPT_compressor_stations_l2183_218315


namespace NUMINAMATH_GPT_lead_points_l2183_218318

-- Define final scores
def final_score_team : ℕ := 68
def final_score_green : ℕ := 39

-- Prove the lead
theorem lead_points : final_score_team - final_score_green = 29 :=
by
  sorry

end NUMINAMATH_GPT_lead_points_l2183_218318


namespace NUMINAMATH_GPT_find_k_value_l2183_218387

theorem find_k_value (x₁ x₂ x₃ x₄ : ℝ)
  (h1 : (x₁ + x₂ + x₃ + x₄) = 18)
  (h2 : (x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄) = k)
  (h3 : (x₁ * x₂ * x₃ + x₁ * x₂ * x₄ + x₁ * x₃ * x₄ + x₂ * x₃ * x₄) = -200)
  (h4 : (x₁ * x₂ * x₃ * x₄) = -1984)
  (h5 : x₁ * x₂ = -32) :
  k = 86 :=
by sorry

end NUMINAMATH_GPT_find_k_value_l2183_218387


namespace NUMINAMATH_GPT_jack_sugar_amount_l2183_218355

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jack_sugar_amount_l2183_218355


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2183_218374

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2 * x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2183_218374


namespace NUMINAMATH_GPT_gcd_min_value_l2183_218372

theorem gcd_min_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 :=
by
  sorry

end NUMINAMATH_GPT_gcd_min_value_l2183_218372


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l2183_218333

theorem cylinder_volume_ratio
  (h : ℝ)     -- height of cylinder B (radius of cylinder A)
  (r : ℝ)     -- radius of cylinder B (height of cylinder A)
  (VA : ℝ)    -- volume of cylinder A
  (VB : ℝ)    -- volume of cylinder B
  (cond1 : r = h / 3)
  (cond2 : VB = 3 * VA)
  (cond3 : VB = N * Real.pi * h^3) :
  N = 1 / 3 := 
sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l2183_218333


namespace NUMINAMATH_GPT_sector_to_cone_ratio_l2183_218398

noncomputable def sector_angle : ℝ := 135
noncomputable def sector_area (S1 : ℝ) : ℝ := S1
noncomputable def cone_surface_area (S2 : ℝ) : ℝ := S2

theorem sector_to_cone_ratio (S1 S2 : ℝ) :
  sector_area S1 = (3 / 8) * (π * 1^2) →
  cone_surface_area S2 = (3 / 8) * (π * 1^2) + (9 / 64 * π) →
  (S1 / S2) = (8 / 11) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sector_to_cone_ratio_l2183_218398


namespace NUMINAMATH_GPT_mike_total_spent_on_toys_l2183_218306

theorem mike_total_spent_on_toys :
  let marbles := 9.05
  let football := 4.95
  let baseball := 6.52
  marbles + football + baseball = 20.52 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_spent_on_toys_l2183_218306


namespace NUMINAMATH_GPT_find_bc_l2183_218316

theorem find_bc (A : ℝ) (a : ℝ) (area : ℝ) (b c : ℝ) :
  A = 60 * (π / 180) → a = Real.sqrt 7 → area = (3 * Real.sqrt 3) / 2 →
  ((b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3)) :=
by
  intros hA ha harea
  -- From the given area condition, derive bc = 6
  have h1 : b * c = 6 := sorry
  -- From the given conditions, derive b + c = 5
  have h2 : b + c = 5 := sorry
  -- Solve the system of equations to find possible values for b and c
  -- Using x² - S⋅x + P = 0 where x are roots, S = b + c, P = b⋅c
  have h3 : (b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3) := sorry
  exact h3

end NUMINAMATH_GPT_find_bc_l2183_218316


namespace NUMINAMATH_GPT_solve_inequality_l2183_218313

theorem solve_inequality (x : ℝ) (h : 3 - (1 / (3 * x + 4)) < 5) : 
  x ∈ { x : ℝ | x < -11/6 } ∨ x ∈ { x : ℝ | x > -4/3 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2183_218313


namespace NUMINAMATH_GPT_symmetric_points_origin_l2183_218327

theorem symmetric_points_origin (a b : ℝ) 
  (h1 : (-2, b) = (-a, -3)) : a - b = 5 := 
by
  -- solution steps are not included in the statement
  sorry

end NUMINAMATH_GPT_symmetric_points_origin_l2183_218327


namespace NUMINAMATH_GPT_second_year_selection_l2183_218353

noncomputable def students_from_first_year : ℕ := 30
noncomputable def students_from_second_year : ℕ := 40
noncomputable def selected_from_first_year : ℕ := 6
noncomputable def selected_from_second_year : ℕ := (selected_from_first_year * students_from_second_year) / students_from_first_year

theorem second_year_selection :
  students_from_second_year = 40 ∧ students_from_first_year = 30 ∧ selected_from_first_year = 6 →
  selected_from_second_year = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_second_year_selection_l2183_218353


namespace NUMINAMATH_GPT_johns_profit_l2183_218392

/-- Define the number of ducks -/
def numberOfDucks : ℕ := 30

/-- Define the cost per duck -/
def costPerDuck : ℤ := 10

/-- Define the weight per duck -/
def weightPerDuck : ℤ := 4

/-- Define the selling price per pound -/
def pricePerPound : ℤ := 5

/-- Define the total cost to buy the ducks -/
def totalCost : ℤ := numberOfDucks * costPerDuck

/-- Define the selling price per duck -/
def sellingPricePerDuck : ℤ := weightPerDuck * pricePerPound

/-- Define the total revenue from selling all the ducks -/
def totalRevenue : ℤ := numberOfDucks * sellingPricePerDuck

/-- Define the profit John made -/
def profit : ℤ := totalRevenue - totalCost

/-- The theorem stating the profit John made given the conditions is $300 -/
theorem johns_profit : profit = 300 := by
  sorry

end NUMINAMATH_GPT_johns_profit_l2183_218392


namespace NUMINAMATH_GPT_sin_cos_sum_l2183_218375

-- Let theta be an angle in the second quadrant
variables (θ : ℝ)
-- Given the condition tan(θ + π / 4) = 1 / 2
variable (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2)
-- Given θ is in the second quadrant
variable (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi)

-- Prove sin θ + cos θ = - sqrt(10) / 5
theorem sin_cos_sum (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2) (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi) :
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_GPT_sin_cos_sum_l2183_218375


namespace NUMINAMATH_GPT_sarah_jim_ratio_l2183_218317

theorem sarah_jim_ratio
  (Tim_toads : ℕ)
  (hTim : Tim_toads = 30)
  (Jim_toads : ℕ)
  (hJim : Jim_toads = Tim_toads + 20)
  (Sarah_toads : ℕ)
  (hSarah : Sarah_toads = 100) :
  Sarah_toads / Jim_toads = 2 :=
by
  sorry

end NUMINAMATH_GPT_sarah_jim_ratio_l2183_218317


namespace NUMINAMATH_GPT_star_example_l2183_218357

section star_operation

variables (x y z : ℕ) 

-- Define the star operation as a binary function
def star (a b : ℕ) : ℕ := a * b

-- Given conditions
axiom star_idempotent : ∀ x : ℕ, star x x = 0
axiom star_associative : ∀ x y z : ℕ, star x (star y z) = (star x y) + z

-- Main theorem to be proved
theorem star_example : star 1993 1935 = 58 :=
sorry

end star_operation

end NUMINAMATH_GPT_star_example_l2183_218357


namespace NUMINAMATH_GPT_find_d_l2183_218329

theorem find_d (d x y : ℝ) (H1 : x - 2 * y = 5) (H2 : d * x + y = 6) (H3 : x > 0) (H4 : y > 0) :
  -1 / 2 < d ∧ d < 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l2183_218329


namespace NUMINAMATH_GPT_muffin_combinations_l2183_218345

theorem muffin_combinations (k : ℕ) (n : ℕ) (h_k : k = 4) (h_n : n = 4) :
  (Nat.choose ((n + k - 1) : ℕ) ((k - 1) : ℕ)) = 35 :=
by
  rw [h_k, h_n]
  -- Simplifying Nat.choose (4 + 4 - 1) (4 - 1) = Nat.choose 7 3
  sorry

end NUMINAMATH_GPT_muffin_combinations_l2183_218345


namespace NUMINAMATH_GPT_tan_sum_formula_l2183_218309

open Real

theorem tan_sum_formula (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_cos_2α : cos (2 * α) = -3 / 5) :
  tan (π / 4 + 2 * α) = -1 / 7 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_tan_sum_formula_l2183_218309


namespace NUMINAMATH_GPT_min_g_l2183_218395

noncomputable def g (x : ℝ) : ℝ := (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem min_g : ∃ x : ℝ, g x = 2 :=
by
  use 0
  sorry

end NUMINAMATH_GPT_min_g_l2183_218395


namespace NUMINAMATH_GPT_original_number_l2183_218311

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end NUMINAMATH_GPT_original_number_l2183_218311


namespace NUMINAMATH_GPT_assignment_statement_correct_l2183_218339

def meaning_of_assignment_statement (N : ℕ) := N + 1

theorem assignment_statement_correct :
  meaning_of_assignment_statement N = N + 1 :=
sorry

end NUMINAMATH_GPT_assignment_statement_correct_l2183_218339


namespace NUMINAMATH_GPT_ratio_of_sum_of_terms_l2183_218334

variable {α : Type*}
variable [Field α]

def geometric_sequence (a : ℕ → α) := ∃ r, ∀ n, a (n + 1) = r * a n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) := S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem ratio_of_sum_of_terms (a : ℕ → α) (S : ℕ → α)
  (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h : S 8 / S 4 = 4) :
  S 12 / S 4 = 13 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sum_of_terms_l2183_218334


namespace NUMINAMATH_GPT_factorization_count_is_correct_l2183_218394

noncomputable def count_factorizations (n : Nat) (k : Nat) : Nat :=
  (Nat.choose (n + k - 1) (k - 1))

noncomputable def factor_count : Nat :=
  let alpha_count := count_factorizations 6 3
  let beta_count := count_factorizations 6 3
  let total_count := alpha_count * beta_count
  let unordered_factorizations := total_count - 15 * 3 - 1
  1 + 15 + unordered_factorizations / 6

theorem factorization_count_is_correct :
  factor_count = 139 := by
  sorry

end NUMINAMATH_GPT_factorization_count_is_correct_l2183_218394


namespace NUMINAMATH_GPT_find_f3_l2183_218385

variable (f : ℕ → ℕ)

axiom h : ∀ x : ℕ, f (x + 1) = x ^ 2

theorem find_f3 : f 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_f3_l2183_218385


namespace NUMINAMATH_GPT_cost_of_gas_per_gallon_l2183_218307

-- Definitions based on the conditions
def hours_driven_1 : ℕ := 2
def speed_1 : ℕ := 60
def hours_driven_2 : ℕ := 3
def speed_2 : ℕ := 50
def mileage_per_gallon : ℕ := 30
def total_gas_cost : ℕ := 18

-- An assumption to simplify handling dollars and gallons
noncomputable def cost_per_gallon : ℕ := total_gas_cost / (speed_1 * hours_driven_1 + speed_2 * hours_driven_2) * mileage_per_gallon

theorem cost_of_gas_per_gallon :
  cost_per_gallon = 2 := by
sorry

end NUMINAMATH_GPT_cost_of_gas_per_gallon_l2183_218307


namespace NUMINAMATH_GPT_gcd_m_n_l2183_218373

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l2183_218373


namespace NUMINAMATH_GPT_crop_fraction_brought_to_AD_l2183_218359

theorem crop_fraction_brought_to_AD
  (AD BC AB CD : ℝ)
  (h : ℝ)
  (angle : ℝ)
  (AD_eq_150 : AD = 150)
  (BC_eq_100 : BC = 100)
  (AB_eq_130 : AB = 130)
  (CD_eq_130 : CD = 130)
  (angle_eq_75 : angle = 75)
  (height_eq : h = (AB / 2) * Real.sin (angle * Real.pi / 180)) -- converting degrees to radians
  (area_trap : ℝ)
  (upper_area : ℝ)
  (total_area_eq : area_trap = (1 / 2) * (AD + BC) * h)
  (upper_area_eq : upper_area = (1 / 2) * (AD + (BC / 2)) * h)
  : (upper_area / area_trap) = 0.8 := 
sorry

end NUMINAMATH_GPT_crop_fraction_brought_to_AD_l2183_218359


namespace NUMINAMATH_GPT_one_third_of_recipe_l2183_218352

noncomputable def recipe_flour_required : ℚ := 7 + 3 / 4

theorem one_third_of_recipe : (1 / 3) * recipe_flour_required = (2 : ℚ) + 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_one_third_of_recipe_l2183_218352


namespace NUMINAMATH_GPT_rachel_minutes_before_bed_l2183_218331

-- Define the conditions in the Lean Lean.
def minutes_spent_solving_before_bed (m : ℕ) : Prop :=
  let problems_solved_before_bed := 5 * m
  let problems_finished_at_lunch := 16
  let total_problems_solved := 76
  problems_solved_before_bed + problems_finished_at_lunch = total_problems_solved

-- The statement we want to prove
theorem rachel_minutes_before_bed : ∃ m : ℕ, minutes_spent_solving_before_bed m ∧ m = 12 :=
sorry

end NUMINAMATH_GPT_rachel_minutes_before_bed_l2183_218331


namespace NUMINAMATH_GPT_greatest_divisor_of_product_of_four_consecutive_integers_l2183_218346

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, ∃ k : Nat, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_product_of_four_consecutive_integers_l2183_218346


namespace NUMINAMATH_GPT_totalPeaches_l2183_218378

-- Definitions based on the given conditions
def redPeaches : Nat := 13
def greenPeaches : Nat := 3

-- Problem statement
theorem totalPeaches : redPeaches + greenPeaches = 16 := by
  sorry

end NUMINAMATH_GPT_totalPeaches_l2183_218378


namespace NUMINAMATH_GPT_train_length_l2183_218362

-- Definitions and conditions
variable (L : ℕ)
def condition1 (L : ℕ) : Prop := L + 100 = 15 * (L + 100) / 15
def condition2 (L : ℕ) : Prop := L + 250 = 20 * (L + 250) / 20

-- Theorem statement
theorem train_length (h1 : condition1 L) (h2 : condition2 L) : L = 350 := 
by 
  sorry

end NUMINAMATH_GPT_train_length_l2183_218362


namespace NUMINAMATH_GPT_probability_of_drawing_red_or_green_l2183_218399

def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def total_marbles : ℕ := red_marbles + green_marbles + yellow_marbles
def favorable_marbles : ℕ := red_marbles + green_marbles
def probability_of_red_or_green : ℚ := favorable_marbles / total_marbles

theorem probability_of_drawing_red_or_green :
  probability_of_red_or_green = 7 / 13 := by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_red_or_green_l2183_218399


namespace NUMINAMATH_GPT_Marley_fruits_total_is_31_l2183_218358

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end NUMINAMATH_GPT_Marley_fruits_total_is_31_l2183_218358


namespace NUMINAMATH_GPT_solve_abs_inequality_l2183_218337

theorem solve_abs_inequality (x : ℝ) : abs ((7 - x) / 4) < 3 → 2 < x ∧ x < 19 :=
by 
  sorry

end NUMINAMATH_GPT_solve_abs_inequality_l2183_218337


namespace NUMINAMATH_GPT_a_profit_share_l2183_218320

/-- Definitions for the shares of capital -/
def a_share : ℚ := 1 / 3
def b_share : ℚ := 1 / 4
def c_share : ℚ := 1 / 5
def d_share : ℚ := 1 - (a_share + b_share + c_share)
def total_profit : ℚ := 2415

/-- The profit share for A, given the conditions on capital subscriptions -/
theorem a_profit_share : a_share * total_profit = 805 := by
  sorry

end NUMINAMATH_GPT_a_profit_share_l2183_218320


namespace NUMINAMATH_GPT_total_spent_snacks_l2183_218310

-- Define the costs and discounts
def cost_pizza : ℕ := 10
def boxes_robert_orders : ℕ := 5
def pizza_discount : ℝ := 0.15
def cost_soft_drink : ℝ := 1.50
def soft_drinks_robert : ℕ := 10
def cost_hamburger : ℕ := 3
def hamburgers_teddy_orders : ℕ := 6
def hamburger_discount : ℝ := 0.10
def soft_drinks_teddy : ℕ := 10

-- Calculate total costs
def total_cost_robert : ℝ := 
  let cost_pizza_total := (boxes_robert_orders * cost_pizza) * (1 - pizza_discount)
  let cost_soft_drinks_total := soft_drinks_robert * cost_soft_drink
  cost_pizza_total + cost_soft_drinks_total

def total_cost_teddy : ℝ :=
  let cost_hamburger_total := (hamburgers_teddy_orders * cost_hamburger) * (1 - hamburger_discount)
  let cost_soft_drinks_total := soft_drinks_teddy * cost_soft_drink
  cost_hamburger_total + cost_soft_drinks_total

-- The final theorem to prove the total spending
theorem total_spent_snacks : 
  total_cost_robert + total_cost_teddy = 88.70 := by
  sorry

end NUMINAMATH_GPT_total_spent_snacks_l2183_218310


namespace NUMINAMATH_GPT_divisibility_by_1956_l2183_218391

theorem divisibility_by_1956 (n : ℕ) (hn : n % 2 = 1) : 
  1956 ∣ (24 * 80^n + 1992 * 83^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_by_1956_l2183_218391


namespace NUMINAMATH_GPT_age_of_youngest_child_l2183_218393

theorem age_of_youngest_child
  (x : ℕ)
  (sum_of_ages : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) :
  x = 4 :=
sorry

end NUMINAMATH_GPT_age_of_youngest_child_l2183_218393


namespace NUMINAMATH_GPT_linear_function_equality_l2183_218312

theorem linear_function_equality (f : ℝ → ℝ) (hf : ∀ x, f (3 * (f x)⁻¹ + 5) = f x)
  (hf1 : f 1 = 5) : f 2 = 3 :=
sorry

end NUMINAMATH_GPT_linear_function_equality_l2183_218312


namespace NUMINAMATH_GPT_new_kite_area_l2183_218379

def original_base := 7
def original_height := 6
def scale_factor := 2
def side_length := 2

def new_base := original_base * scale_factor
def new_height := original_height * scale_factor
def half_new_height := new_height / 2

def area_triangle := (1 / 2 : ℚ) * new_base * half_new_height
def total_area := 2 * area_triangle

theorem new_kite_area : total_area = 84 := by
  sorry

end NUMINAMATH_GPT_new_kite_area_l2183_218379


namespace NUMINAMATH_GPT_money_spent_correct_l2183_218381

-- Define conditions
def spring_income : ℕ := 2
def summer_income : ℕ := 27
def amount_after_supplies : ℕ := 24

-- Define the resulting money spent on supplies
def money_spent_on_supplies : ℕ :=
  (spring_income + summer_income) - amount_after_supplies

theorem money_spent_correct :
  money_spent_on_supplies = 5 := by
  sorry

end NUMINAMATH_GPT_money_spent_correct_l2183_218381


namespace NUMINAMATH_GPT_area_of_large_rectangle_ABCD_l2183_218360

-- Definitions for conditions and given data
def shaded_rectangle_area : ℕ := 2
def area_of_rectangle_ABCD (a b c : ℕ) : ℕ := a + b + c

-- The theorem to prove
theorem area_of_large_rectangle_ABCD
  (a b c : ℕ) 
  (h1 : shaded_rectangle_area = a)
  (h2 : shaded_rectangle_area = b)
  (h3 : a + b + c = 8) : 
  area_of_rectangle_ABCD a b c = 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_large_rectangle_ABCD_l2183_218360


namespace NUMINAMATH_GPT_Kyle_papers_delivered_each_week_l2183_218343

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end NUMINAMATH_GPT_Kyle_papers_delivered_each_week_l2183_218343


namespace NUMINAMATH_GPT_value_x2012_l2183_218323

def f (x : ℝ) : ℝ := sorry

noncomputable def x (n : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom increasing_f : ∀ x y : ℝ, x < y → f x < f y
axiom arithmetic_seq : ∀ n : ℕ, x (n) = x (1) + (n-1) * 2
axiom condition : f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0

theorem value_x2012 : x 2012 = 4005 := 
by sorry

end NUMINAMATH_GPT_value_x2012_l2183_218323


namespace NUMINAMATH_GPT_moe_cannot_finish_on_time_l2183_218384

theorem moe_cannot_finish_on_time (lawn_length lawn_width : ℝ) (swath : ℕ) (overlap : ℕ) (speed : ℝ) (available_time : ℝ) :
  lawn_length = 120 ∧ lawn_width = 180 ∧ swath = 30 ∧ overlap = 6 ∧ speed = 4000 ∧ available_time = 2 →
  (lawn_width / (swath - overlap) * lawn_length / speed) > available_time :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_moe_cannot_finish_on_time_l2183_218384


namespace NUMINAMATH_GPT_avg_length_one_third_wires_l2183_218303

theorem avg_length_one_third_wires (x : ℝ) (L1 L2 L3 L4 L5 L6 : ℝ) 
  (h_total_wires : L1 + L2 + L3 + L4 + L5 + L6 = 6 * 80) 
  (h_avg_other_wires : (L3 + L4 + L5 + L6) / 4 = 85) 
  (h_avg_all_wires : (L1 + L2 + L3 + L4 + L5 + L6) / 6 = 80) :
  (L1 + L2) / 2 = 70 :=
by
  sorry

end NUMINAMATH_GPT_avg_length_one_third_wires_l2183_218303


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2183_218376

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0)
  (h_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / 16 = 1 ↔ true)
  (eccentricity : a^2 + 16 / a^2 = (5 / 3)^2) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2183_218376


namespace NUMINAMATH_GPT_subtract_mult_equal_l2183_218308

theorem subtract_mult_equal :
  2000000000000 - 1111111111111 * 1 = 888888888889 :=
by
  sorry

end NUMINAMATH_GPT_subtract_mult_equal_l2183_218308


namespace NUMINAMATH_GPT_find_n_l2183_218300

-- Definitions of the conditions
variables (x n : ℝ)
variable (h1 : (x / 4) * n + 10 - 12 = 48)
variable (h2 : x = 40)

-- Theorem statement
theorem find_n (x n : ℝ) (h1 : (x / 4) * n + 10 - 12 = 48) (h2 : x = 40) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2183_218300


namespace NUMINAMATH_GPT_concentration_after_removing_water_l2183_218351

theorem concentration_after_removing_water :
  ∀ (initial_volume : ℝ) (initial_percentage : ℝ) (water_removed : ℝ),
  initial_volume = 18 →
  initial_percentage = 0.4 →
  water_removed = 6 →
  (initial_percentage * initial_volume) / (initial_volume - water_removed) * 100 = 60 :=
by
  intros initial_volume initial_percentage water_removed h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_concentration_after_removing_water_l2183_218351


namespace NUMINAMATH_GPT_sum_cis_angles_l2183_218367

noncomputable def complex.cis (θ : ℝ) := Complex.exp (Complex.I * θ)

theorem sum_cis_angles :
  (complex.cis (80 * Real.pi / 180) + complex.cis (88 * Real.pi / 180) + complex.cis (96 * Real.pi / 180) + 
  complex.cis (104 * Real.pi / 180) + complex.cis (112 * Real.pi / 180) + complex.cis (120 * Real.pi / 180) + 
  complex.cis (128 * Real.pi / 180)) = r * complex.cis (104 * Real.pi / 180) := 
sorry

end NUMINAMATH_GPT_sum_cis_angles_l2183_218367


namespace NUMINAMATH_GPT_geometric_sequence_expression_l2183_218304

variable {a : ℕ → ℝ}

-- Define the geometric sequence property
def is_geometric (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_expression :
  is_geometric a q →
  a 3 = 2 →
  a 6 = 16 →
  ∀ n, a n = 2^(n-2) := by
  intros h_geom h_a3 h_a6
  sorry

end NUMINAMATH_GPT_geometric_sequence_expression_l2183_218304


namespace NUMINAMATH_GPT_number_of_attendants_writing_with_both_l2183_218397

-- Definitions for each of the conditions
def attendants_using_pencil : ℕ := 25
def attendants_using_pen : ℕ := 15
def attendants_using_only_one : ℕ := 20

-- Theorem that states the mathematically equivalent proof problem
theorem number_of_attendants_writing_with_both 
  (p : ℕ := attendants_using_pencil)
  (e : ℕ := attendants_using_pen)
  (o : ℕ := attendants_using_only_one) : 
  ∃ x, (p - x) + (e - x) = o ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_attendants_writing_with_both_l2183_218397


namespace NUMINAMATH_GPT_max_area_cross_section_rect_prism_l2183_218325

/-- The maximum area of the cross-sectional cut of a rectangular prism 
having its vertical edges parallel to the z-axis, with cross-section 
rectangle of sides 8 and 12, whose bottom side lies in the xy-plane 
centered at the origin (0,0,0), cut by the plane 3x + 5y - 2z = 30 
is approximately 118.34. --/
theorem max_area_cross_section_rect_prism :
  ∃ A : ℝ, abs (A - 118.34) < 0.01 :=
sorry

end NUMINAMATH_GPT_max_area_cross_section_rect_prism_l2183_218325


namespace NUMINAMATH_GPT_find_x0_l2183_218348

noncomputable def slopes_product_eq_three (x : ℝ) : Prop :=
  let y1 := 2 - 1 / x
  let y2 := x^3 - x^2 + 2 * x
  let dy1_dx := 1 / (x^2)
  let dy2_dx := 3 * x^2 - 2 * x + 2
  dy1_dx * dy2_dx = 3

theorem find_x0 : ∃ (x0 : ℝ), slopes_product_eq_three x0 ∧ x0 = 1 :=
by {
  use 1,
  sorry
}

end NUMINAMATH_GPT_find_x0_l2183_218348


namespace NUMINAMATH_GPT_wall_length_is_800_l2183_218326

def brick_volume : ℝ := 50 * 11.25 * 6
def total_brick_volume : ℝ := 3200 * brick_volume
def wall_volume (x : ℝ) : ℝ := x * 600 * 22.5

theorem wall_length_is_800 :
  ∀ (x : ℝ), total_brick_volume = wall_volume x → x = 800 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_wall_length_is_800_l2183_218326


namespace NUMINAMATH_GPT_conceived_number_is_seven_l2183_218386

theorem conceived_number_is_seven (x : ℕ) (h1 : x > 0) (h2 : (1 / 4 : ℚ) * (10 * x + 7 - x * x) - x = 0) : x = 7 := by
  sorry

end NUMINAMATH_GPT_conceived_number_is_seven_l2183_218386


namespace NUMINAMATH_GPT_distance_to_first_sign_l2183_218336

-- Definitions based on conditions
def total_distance : ℕ := 1000
def after_second_sign : ℕ := 275
def between_signs : ℕ := 375

-- Problem statement
theorem distance_to_first_sign 
  (D : ℕ := total_distance) 
  (a : ℕ := after_second_sign) 
  (d : ℕ := between_signs) : 
  (D - a - d = 350) :=
by
  sorry

end NUMINAMATH_GPT_distance_to_first_sign_l2183_218336


namespace NUMINAMATH_GPT_collinear_condition_perpendicular_condition_l2183_218389

-- Problem 1: Prove collinearity condition for k = -2
theorem collinear_condition (k : ℝ) : 
  (k - 5) * (-12) - (12 - k) * 6 = 0 ↔ k = -2 :=
sorry

-- Problem 2: Prove perpendicular condition for k = 2 or k = 11
theorem perpendicular_condition (k : ℝ) : 
  (20 + (k - 6) * (7 - k)) = 0 ↔ (k = 2 ∨ k = 11) :=
sorry

end NUMINAMATH_GPT_collinear_condition_perpendicular_condition_l2183_218389
