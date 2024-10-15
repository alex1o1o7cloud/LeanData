import Mathlib

namespace NUMINAMATH_GPT_damage_conversion_l1445_144595

def usd_to_cad_conversion_rate : ℝ := 1.25
def damage_in_usd : ℝ := 60000000
def damage_in_cad : ℝ := 75000000

theorem damage_conversion :
  damage_in_usd * usd_to_cad_conversion_rate = damage_in_cad :=
sorry

end NUMINAMATH_GPT_damage_conversion_l1445_144595


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l1445_144582

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 2 ≤ 3 ∧ 1 + x > -2) ↔ (-3 < x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l1445_144582


namespace NUMINAMATH_GPT_cat_catches_mouse_l1445_144540

-- Define the distances
def AB := 200
def BC := 140
def CD := 20

-- Define the speeds (in meters per minute)
def mouse_speed := 60
def cat_speed := 80

-- Define the total distances the mouse and cat travel
def mouse_total_distance := 320 -- The mouse path is along a zigzag route initially specified in the problem
def cat_total_distance := AB + BC + CD -- 360 meters as calculated

-- Define the times they take to reach point D
def mouse_time := mouse_total_distance / mouse_speed -- 5.33 minutes
def cat_time := cat_total_distance / cat_speed -- 4.5 minutes

-- Proof problem statement
theorem cat_catches_mouse : cat_time < mouse_time := 
by
  sorry

end NUMINAMATH_GPT_cat_catches_mouse_l1445_144540


namespace NUMINAMATH_GPT_find_y_arithmetic_mean_l1445_144519

theorem find_y_arithmetic_mean (y : ℝ) 
  (h : (8 + 15 + 20 + 7 + y + 9) / 6 = 12) : 
  y = 13 :=
sorry

end NUMINAMATH_GPT_find_y_arithmetic_mean_l1445_144519


namespace NUMINAMATH_GPT_find_angle_EHG_l1445_144589

noncomputable def angle_EHG (angle_EFG : ℝ) (angle_GHE : ℝ) : ℝ := angle_GHE - angle_EFG
 
theorem find_angle_EHG : 
  ∀ (EF GH : Prop) (angle_EFG angle_GHE : ℝ), (EF ∧ GH) → 
    EF ∧ GH ∧ angle_EFG = 50 ∧ angle_GHE = 80 → angle_EHG angle_EFG angle_GHE = 30 := 
by 
  intros EF GH angle_EFG angle_GHE h1 h2
  sorry

end NUMINAMATH_GPT_find_angle_EHG_l1445_144589


namespace NUMINAMATH_GPT_compute_a_l1445_144561

theorem compute_a (a : ℝ) (h : 2.68 * 0.74 = a) : a = 1.9832 :=
by
  -- Here skip the proof steps
  sorry

end NUMINAMATH_GPT_compute_a_l1445_144561


namespace NUMINAMATH_GPT_perimeter_of_rectangle_l1445_144549

theorem perimeter_of_rectangle (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  ∃ perimeter length, length = area / width ∧ perimeter = 2 * (length + width) ∧ perimeter = 110 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_l1445_144549


namespace NUMINAMATH_GPT_quad_roots_expression_l1445_144553

theorem quad_roots_expression (x1 x2 : ℝ) (h1 : x1 * x1 + 2019 * x1 + 1 = 0) (h2 : x2 * x2 + 2019 * x2 + 1 = 0) :
  x1 * x2 - x1 - x2 = 2020 :=
sorry

end NUMINAMATH_GPT_quad_roots_expression_l1445_144553


namespace NUMINAMATH_GPT_unique_zero_in_interval_l1445_144505

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x ^ 2

theorem unique_zero_in_interval
  (a : ℝ) (ha : a > 0)
  (x₀ : ℝ) (hx₀ : f a x₀ = 0)
  (h_interval : -1 < x₀ ∧ x₀ < 0) :
  Real.exp (-2) < x₀ + 1 ∧ x₀ + 1 < Real.exp (-1) :=
sorry

end NUMINAMATH_GPT_unique_zero_in_interval_l1445_144505


namespace NUMINAMATH_GPT_max_value_of_a_plus_b_l1445_144507

theorem max_value_of_a_plus_b (a b : ℕ) (h1 : 7 * a + 19 * b = 213) (h2 : a > 0) (h3 : b > 0) : a + b = 27 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_plus_b_l1445_144507


namespace NUMINAMATH_GPT_find_c_l1445_144571

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x : ℝ, g_inv (g x c) = x) -> c = 3 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_c_l1445_144571


namespace NUMINAMATH_GPT_digits_sum_unique_l1445_144541

variable (A B C D E F G H : ℕ)

theorem digits_sum_unique :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧
  F ≠ G ∧ F ≠ H ∧
  G ≠ H ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  0 ≤ E ∧ E ≤ 9 ∧ 0 ≤ F ∧ F ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ H ∧ H ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D) + (E * 1000 + F * 100 + G * 10 + H) = 10652 ∧
  A = 9 ∧ B = 5 ∧ C = 6 ∧ D = 7 ∧
  E = 1 ∧ F = 0 ∧ G = 8 ∧ H = 5 :=
sorry

end NUMINAMATH_GPT_digits_sum_unique_l1445_144541


namespace NUMINAMATH_GPT_gift_exchange_equation_l1445_144563

theorem gift_exchange_equation (x : ℕ) (h : x * (x - 1) = 40) : 
  x * (x - 1) = 40 :=
by
  exact h

end NUMINAMATH_GPT_gift_exchange_equation_l1445_144563


namespace NUMINAMATH_GPT_combined_selling_price_correct_l1445_144532

def cost_A : ℕ := 500
def cost_B : ℕ := 800
def cost_C : ℕ := 1200
def profit_A : ℕ := 25
def profit_B : ℕ := 30
def profit_C : ℕ := 20

def selling_price (cost profit_percentage : ℕ) : ℕ :=
  cost + (profit_percentage * cost / 100)

def combined_selling_price : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

theorem combined_selling_price_correct : combined_selling_price = 3105 := by
  sorry

end NUMINAMATH_GPT_combined_selling_price_correct_l1445_144532


namespace NUMINAMATH_GPT_milk_left_l1445_144543

theorem milk_left (initial_milk : ℝ) (milk_james : ℝ) (milk_maria : ℝ) :
  initial_milk = 5 → milk_james = 15 / 4 → milk_maria = 3 / 4 → 
  initial_milk - (milk_james + milk_maria) = 1 / 2 :=
by
  intros h_initial h_james h_maria
  rw [h_initial, h_james, h_maria]
  -- The calculation would be performed here.
  sorry

end NUMINAMATH_GPT_milk_left_l1445_144543


namespace NUMINAMATH_GPT_air_conditioning_price_november_l1445_144559

noncomputable def price_in_november : ℝ :=
  let january_price := 470
  let february_price := january_price * (1 - 0.12)
  let march_price := february_price * (1 + 0.08)
  let april_price := march_price * (1 - 0.10)
  let june_price := april_price * (1 + 0.05)
  let august_price := june_price * (1 - 0.07)
  let october_price := august_price * (1 + 0.06)
  october_price * (1 - 0.15)

theorem air_conditioning_price_november : price_in_november = 353.71 := by
  sorry

end NUMINAMATH_GPT_air_conditioning_price_november_l1445_144559


namespace NUMINAMATH_GPT_farmer_apples_l1445_144576

theorem farmer_apples : 127 - 39 = 88 := by
  -- Skipping proof details
  sorry

end NUMINAMATH_GPT_farmer_apples_l1445_144576


namespace NUMINAMATH_GPT_trap_speed_independent_of_location_l1445_144535

theorem trap_speed_independent_of_location 
  (h b a : ℝ) (v_mouse : ℝ) 
  (path_length : ℝ := Real.sqrt (a^2 + (3*h)^2)) 
  (T : ℝ := path_length / v_mouse) 
  (step_height : ℝ := h) 
  (v_trap : ℝ := step_height / T) 
  (h_val : h = 3) 
  (b_val : b = 1) 
  (a_val : a = 8) 
  (v_mouse_val : v_mouse = 17) : 
  v_trap = 8 := by
  sorry

end NUMINAMATH_GPT_trap_speed_independent_of_location_l1445_144535


namespace NUMINAMATH_GPT_students_attend_Purum_Elementary_School_l1445_144550
open Nat

theorem students_attend_Purum_Elementary_School (P N : ℕ) 
  (h1 : P + N = 41) (h2 : P = N + 3) : P = 22 :=
sorry

end NUMINAMATH_GPT_students_attend_Purum_Elementary_School_l1445_144550


namespace NUMINAMATH_GPT_polynomial_expansion_l1445_144592

theorem polynomial_expansion :
  (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
  35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l1445_144592


namespace NUMINAMATH_GPT_alice_total_spending_l1445_144545

theorem alice_total_spending :
  let book_price_gbp := 15
  let souvenir_price_eur := 20
  let gbp_to_usd_rate := 1.25
  let eur_to_usd_rate := 1.10
  let book_price_usd := book_price_gbp * gbp_to_usd_rate
  let souvenir_price_usd := souvenir_price_eur * eur_to_usd_rate
  let total_usd := book_price_usd + souvenir_price_usd
  total_usd = 40.75 :=
by
  sorry

end NUMINAMATH_GPT_alice_total_spending_l1445_144545


namespace NUMINAMATH_GPT_base8_addition_l1445_144533

theorem base8_addition : (234 : ℕ) + (157 : ℕ) = (4 * 8^2 + 1 * 8^1 + 3 * 8^0 : ℕ) :=
by sorry

end NUMINAMATH_GPT_base8_addition_l1445_144533


namespace NUMINAMATH_GPT_binomial_square_correct_k_l1445_144567

theorem binomial_square_correct_k (k : ℚ) : (∃ t u : ℚ, k = t^2 ∧ 28 = 2 * t * u ∧ 9 = u^2) → k = 196 / 9 :=
by
  sorry

end NUMINAMATH_GPT_binomial_square_correct_k_l1445_144567


namespace NUMINAMATH_GPT_fractions_ordered_l1445_144526

theorem fractions_ordered :
  (2 / 5 : ℚ) < (3 / 5) ∧ (3 / 5) < (4 / 6) ∧ (4 / 6) < (4 / 5) ∧ (4 / 5) < (6 / 5) ∧ (6 / 5) < (4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_fractions_ordered_l1445_144526


namespace NUMINAMATH_GPT_midpoint_distance_from_school_l1445_144562

def distance_school_kindergarten_km := 1
def distance_school_kindergarten_m := 700
def distance_kindergarten_house_m := 900

theorem midpoint_distance_from_school : 
  (1000 * distance_school_kindergarten_km + distance_school_kindergarten_m + distance_kindergarten_house_m) / 2 = 1300 := 
by
  sorry

end NUMINAMATH_GPT_midpoint_distance_from_school_l1445_144562


namespace NUMINAMATH_GPT_range_of_a_l1445_144558

variable {R : Type} [LinearOrderedField R]

def is_even (f : R → R) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_non_neg (f : R → R) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem range_of_a 
  (f : R → R) 
  (even_f : is_even f)
  (mono_f : is_monotone_increasing_on_non_neg f)
  (ineq : ∀ a, f (a + 1) ≤ f 4) : 
  ∀ a, -5 ≤ a ∧ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1445_144558


namespace NUMINAMATH_GPT_largest_constant_inequality_l1445_144523

theorem largest_constant_inequality :
  ∃ C : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧ C = Real.sqrt (4 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_constant_inequality_l1445_144523


namespace NUMINAMATH_GPT_prob_l1445_144542

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + 1 / (2 + 1 / x))

theorem prob (x1 x2 x3 : ℝ) (h1 : x1 = 0) 
  (h2 : 2 + 1 / x2 = 0) 
  (h3 : 2 + 1 / (2 + 1 / x3) = 0) : 
  x1 + x2 + x3 = -9 / 10 := 
sorry

end NUMINAMATH_GPT_prob_l1445_144542


namespace NUMINAMATH_GPT_trig_values_same_terminal_side_l1445_144502

-- Statement: The trigonometric function values of angles with the same terminal side are equal.
theorem trig_values_same_terminal_side (θ₁ θ₂ : ℝ) (h : ∃ k : ℤ, θ₂ = θ₁ + 2 * k * π) :
  (∀ f : ℝ -> ℝ, f θ₁ = f θ₂) :=
by
  sorry

end NUMINAMATH_GPT_trig_values_same_terminal_side_l1445_144502


namespace NUMINAMATH_GPT_rectangle_perimeter_l1445_144536

theorem rectangle_perimeter
  (L W : ℕ)
  (h1 : L * W = 360)
  (h2 : (L + 10) * (W - 6) = 360) :
  2 * L + 2 * W = 76 := 
sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1445_144536


namespace NUMINAMATH_GPT_solve_equation_l1445_144556

theorem solve_equation : ∀ x y : ℤ, x^2 + y^2 = 3 * x * y → x = 0 ∧ y = 0 := by
  intros x y h
  sorry

end NUMINAMATH_GPT_solve_equation_l1445_144556


namespace NUMINAMATH_GPT_rook_reaches_upper_right_in_expected_70_minutes_l1445_144569

section RookMoves

noncomputable def E : ℝ := 70

-- Definition of expected number of minutes considering the row and column moves.
-- This is a direct translation from the problem's correct answer.
def rook_expected_minutes_to_upper_right (E_0 E_1 : ℝ) : Prop :=
  E_0 = (70 : ℝ) ∧ E_1 = (70 : ℝ)

theorem rook_reaches_upper_right_in_expected_70_minutes : E = 70 := sorry

end RookMoves

end NUMINAMATH_GPT_rook_reaches_upper_right_in_expected_70_minutes_l1445_144569


namespace NUMINAMATH_GPT_translate_parabola_l1445_144548

theorem translate_parabola (x : ℝ) :
  (x^2 + 3) = (x - 5)^2 + 3 :=
sorry

end NUMINAMATH_GPT_translate_parabola_l1445_144548


namespace NUMINAMATH_GPT_total_expenditure_correct_l1445_144512

def length : ℝ := 50
def width : ℝ := 30
def cost_per_square_meter : ℝ := 100

def area (L W : ℝ) : ℝ := L * W
def total_expenditure (A C : ℝ) : ℝ := A * C

theorem total_expenditure_correct :
  total_expenditure (area length width) cost_per_square_meter = 150000 := by
  sorry

end NUMINAMATH_GPT_total_expenditure_correct_l1445_144512


namespace NUMINAMATH_GPT_cyclic_quadrilateral_ptolemy_l1445_144537

theorem cyclic_quadrilateral_ptolemy 
  (a b c d : ℝ) 
  (h : a + b + c + d = Real.pi) :
  Real.sin (a + b) * Real.sin (b + c) = Real.sin a * Real.sin c + Real.sin b * Real.sin d :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_ptolemy_l1445_144537


namespace NUMINAMATH_GPT_cost_of_15_brown_socks_is_3_dollars_l1445_144539

def price_of_brown_sock (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) : ℚ :=
  (price_white_socks - price_white_more_than_brown) / 2

def cost_of_15_brown_socks (price_brown_sock : ℚ) : ℚ :=
  15 * price_brown_sock

theorem cost_of_15_brown_socks_is_3_dollars
  (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) 
  (h1 : price_white_socks = 0.45) (h2 : price_white_more_than_brown = 0.25) :
  cost_of_15_brown_socks (price_of_brown_sock price_white_socks price_white_more_than_brown) = 3 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_15_brown_socks_is_3_dollars_l1445_144539


namespace NUMINAMATH_GPT_evaluate_expression_is_sixth_l1445_144573

noncomputable def evaluate_expression := (1 / Real.log 3000^4 / Real.log 8) + (4 / Real.log 3000^4 / Real.log 9)

theorem evaluate_expression_is_sixth:
  evaluate_expression = 1 / 6 :=
  by
  sorry

end NUMINAMATH_GPT_evaluate_expression_is_sixth_l1445_144573


namespace NUMINAMATH_GPT_shortest_side_of_right_triangle_l1445_144587

theorem shortest_side_of_right_triangle
  (a b c : ℝ)
  (h : a = 5) (k : b = 13) (rightangled : a^2 + c^2 = b^2) : c = 12 := 
sorry

end NUMINAMATH_GPT_shortest_side_of_right_triangle_l1445_144587


namespace NUMINAMATH_GPT_trains_crossing_time_l1445_144538

-- Definitions based on conditions
def train_length : ℕ := 120
def time_train1_cross_pole : ℕ := 10
def time_train2_cross_pole : ℕ := 15

-- Question reformulated as a proof goal
theorem trains_crossing_time :
  let v1 := train_length / time_train1_cross_pole  -- Speed of train 1
  let v2 := train_length / time_train2_cross_pole  -- Speed of train 2
  let relative_speed := v1 + v2                    -- Relative speed in opposite directions
  let total_distance := train_length + train_length -- Sum of both trains' lengths
  let time_to_cross := total_distance / relative_speed -- Time to cross each other
  time_to_cross = 12 := 
by
  -- The proof here is stated, but not needed in this task
  -- All necessary computation steps
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l1445_144538


namespace NUMINAMATH_GPT_weights_difference_l1445_144579

-- Definitions based on conditions
def A : ℕ := 36
def ratio_part : ℕ := A / 4
def B : ℕ := 5 * ratio_part
def C : ℕ := 6 * ratio_part

-- Theorem to prove
theorem weights_difference :
  (A + C) - B = 45 := by
  sorry

end NUMINAMATH_GPT_weights_difference_l1445_144579


namespace NUMINAMATH_GPT_polynomial_decomposition_l1445_144514

-- Define the given polynomial
def P (x y z : ℝ) : ℝ := x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2

-- Define the target decomposition
def Q (x y z : ℝ) : ℝ := (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2

theorem polynomial_decomposition (x y z : ℝ) : P x y z = Q x y z :=
  sorry

end NUMINAMATH_GPT_polynomial_decomposition_l1445_144514


namespace NUMINAMATH_GPT_smallest_of_three_consecutive_even_numbers_l1445_144580

def sum_of_three_consecutive_even_numbers (n : ℕ) : Prop :=
  n + (n + 2) + (n + 4) = 162

theorem smallest_of_three_consecutive_even_numbers (n : ℕ) (h : sum_of_three_consecutive_even_numbers n) : n = 52 :=
by
  sorry

end NUMINAMATH_GPT_smallest_of_three_consecutive_even_numbers_l1445_144580


namespace NUMINAMATH_GPT_max_time_for_taxiing_is_15_l1445_144599

-- Declare the function representing the distance traveled by the plane with respect to time
def distance (t : ℝ) : ℝ := 60 * t - 2 * t ^ 2

-- The main theorem stating the maximum time s the plane uses for taxiing
theorem max_time_for_taxiing_is_15 : ∃ s, ∀ t, distance t ≤ distance s ∧ s = 15 :=
by
  sorry

end NUMINAMATH_GPT_max_time_for_taxiing_is_15_l1445_144599


namespace NUMINAMATH_GPT_complex_fraction_simplify_l1445_144577

variable (i : ℂ)
variable (h : i^2 = -1)

theorem complex_fraction_simplify :
  (1 - i) / ((1 + i) ^ 2) = -1/2 - i/2 :=
by
  sorry

end NUMINAMATH_GPT_complex_fraction_simplify_l1445_144577


namespace NUMINAMATH_GPT_linda_savings_l1445_144524

theorem linda_savings (S : ℝ) (h1 : 1 / 4 * S = 150) : S = 600 :=
sorry

end NUMINAMATH_GPT_linda_savings_l1445_144524


namespace NUMINAMATH_GPT_sum_of_three_pairwise_relatively_prime_integers_l1445_144504

theorem sum_of_three_pairwise_relatively_prime_integers
  (a b c : ℕ)
  (h1 : a > 1)
  (h2 : b > 1)
  (h3 : c > 1)
  (h4 : a * b * c = 13824)
  (h5 : Nat.gcd a b = 1)
  (h6 : Nat.gcd b c = 1)
  (h7 : Nat.gcd a c = 1) :
  a + b + c = 144 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_pairwise_relatively_prime_integers_l1445_144504


namespace NUMINAMATH_GPT_numbers_lcm_sum_l1445_144583

theorem numbers_lcm_sum :
  ∃ A : List ℕ, A.length = 100 ∧
    (A.count 1 = 89 ∧ A.count 2 = 8 ∧ [4, 5, 6] ⊆ A) ∧
    A.sum = A.foldr lcm 1 :=
by
  sorry

end NUMINAMATH_GPT_numbers_lcm_sum_l1445_144583


namespace NUMINAMATH_GPT_value_of_x_y_l1445_144527

noncomputable def real_ln : ℝ → ℝ := sorry

theorem value_of_x_y (x y : ℝ) (h : 3 * x - y ≤ real_ln (x + 2 * y - 3) + real_ln (2 * x - 3 * y + 5)) :
  x + y = 16 / 7 :=
sorry

end NUMINAMATH_GPT_value_of_x_y_l1445_144527


namespace NUMINAMATH_GPT_initial_percentage_female_workers_l1445_144529

theorem initial_percentage_female_workers
(E : ℕ) (F : ℝ) 
(h1 : E + 30 = 360) 
(h2 : (F / 100) * E = (55 / 100) * (E + 30)) :
F = 60 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_initial_percentage_female_workers_l1445_144529


namespace NUMINAMATH_GPT_scott_awards_l1445_144511

theorem scott_awards (S : ℕ) 
  (h1 : ∃ J, J = 3 * S)
  (h2 : ∃ B, B = 2 * (3 * S) ∧ B = 24) : S = 4 := 
by 
  sorry

end NUMINAMATH_GPT_scott_awards_l1445_144511


namespace NUMINAMATH_GPT_number_of_hydrogen_atoms_l1445_144593

/-- 
A compound has a certain number of Hydrogen, 1 Chromium, and 4 Oxygen atoms. 
The molecular weight of the compound is 118. How many Hydrogen atoms are in the compound?
-/
theorem number_of_hydrogen_atoms
  (H Cr O : ℕ)
  (mw_H : ℕ := 1)
  (mw_Cr : ℕ := 52)
  (mw_O : ℕ := 16)
  (H_weight : ℕ := H * mw_H)
  (Cr_weight : ℕ := 1 * mw_Cr)
  (O_weight : ℕ := 4 * mw_O)
  (total_weight : ℕ := 118)
  (weight_without_H : ℕ := Cr_weight + O_weight) 
  (H_weight_calculated : ℕ := total_weight - weight_without_H) :
  H = 2 :=
  by
    sorry

end NUMINAMATH_GPT_number_of_hydrogen_atoms_l1445_144593


namespace NUMINAMATH_GPT_b_investment_months_after_a_l1445_144508

-- Definitions based on the conditions
def a_investment : ℕ := 100
def b_investment : ℕ := 200
def total_yearly_investment_period : ℕ := 12
def total_profit : ℕ := 100
def a_share_of_profit : ℕ := 50
def x (x_val : ℕ) : Prop := x_val = 6

-- Main theorem to prove
theorem b_investment_months_after_a (x_val : ℕ) 
  (h1 : a_investment = 100)
  (h2 : b_investment = 200)
  (h3 : total_yearly_investment_period = 12)
  (h4 : total_profit = 100)
  (h5 : a_share_of_profit = 50) :
  (100 * total_yearly_investment_period) = 200 * (total_yearly_investment_period - x_val) → 
  x x_val := 
by
  sorry

end NUMINAMATH_GPT_b_investment_months_after_a_l1445_144508


namespace NUMINAMATH_GPT_simple_fraction_pow_l1445_144590

theorem simple_fraction_pow : (66666^4 / 22222^4) = 81 := by
  sorry

end NUMINAMATH_GPT_simple_fraction_pow_l1445_144590


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1445_144501

-- Problem 1
theorem problem1 : (- (3 : ℝ) / 7) + (1 / 5) + (2 / 7) + (- (6 / 5)) = - (8 / 7) :=
by
  sorry

-- Problem 2
theorem problem2 : -(-1) + 3^2 / (1 - 4) * 2 = -5 :=
by
  sorry

-- Problem 3
theorem problem3 :  (-(1 / 6))^2 / ((1 / 2 - 1 / 3)^2) / (abs (-6))^2 = 1 / 36 :=
by
  sorry

-- Problem 4
theorem problem4 : (-1) ^ 1000 - 2.45 * 8 + 2.55 * (-8) = -39 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1445_144501


namespace NUMINAMATH_GPT_find_pairs_l1445_144551

def is_prime (p : ℕ) : Prop := (p ≥ 2) ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_pairs (a p : ℕ) (h_pos_a : a > 0) (h_prime_p : is_prime p) :
  (∀ m n : ℕ, 0 < m → 0 < n → (a ^ (2 ^ n) % p ^ n = a ^ (2 ^ m) % p ^ m ∧ a ^ (2 ^ n) % p ^ n ≠ 0))
  ↔ (∃ k : ℕ, a = 2 * k + 1 ∧ p = 2) :=
sorry

end NUMINAMATH_GPT_find_pairs_l1445_144551


namespace NUMINAMATH_GPT_find_constants_and_intervals_l1445_144521

open Real

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x^2 - 2 * x
def f' (x : ℝ) (a b : ℝ) := 3 * a * x^2 + 2 * b * x - 2

theorem find_constants_and_intervals :
  (f' (1 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (f' (-2 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) > 0 ↔ x < -2 ∨ x > 1) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) < 0 ↔ -2 < x ∧ x < 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_constants_and_intervals_l1445_144521


namespace NUMINAMATH_GPT_friends_truth_l1445_144588

-- Definitions for the truth values of the friends
def F₁_truth (a x₁ x₂ x₃ : Prop) : Prop := a ↔ ¬ (x₁ ∨ x₂ ∨ x₃)
def F₂_truth (b x₁ x₂ x₃ : Prop) : Prop := b ↔ (x₂ ∧ ¬ x₁ ∧ ¬ x₃)
def F₃_truth (c x₁ x₂ x₃ : Prop) : Prop := c ↔ x₃

-- Main theorem statement
theorem friends_truth (a b c x₁ x₂ x₃ : Prop) 
  (H₁ : F₁_truth a x₁ x₂ x₃) 
  (H₂ : F₂_truth b x₁ x₂ x₃) 
  (H₃ : F₃_truth c x₁ x₂ x₃)
  (H₄ : a ∨ b ∨ c) 
  (H₅ : ¬ (a ∧ b ∧ c)) : a ∧ ¬b ∧ ¬c ∨ ¬a ∧ b ∧ ¬c ∨ ¬a ∧ ¬b ∧ c :=
sorry

end NUMINAMATH_GPT_friends_truth_l1445_144588


namespace NUMINAMATH_GPT_side_of_beef_weight_after_processing_l1445_144566

theorem side_of_beef_weight_after_processing (initial_weight : ℝ) (lost_percentage : ℝ) (final_weight : ℝ) 
  (h1 : initial_weight = 400) 
  (h2 : lost_percentage = 0.4) 
  (h3 : final_weight = initial_weight * (1 - lost_percentage)) : 
  final_weight = 240 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_side_of_beef_weight_after_processing_l1445_144566


namespace NUMINAMATH_GPT_carla_total_time_l1445_144568

def total_time_spent (knife_time : ℕ) (peeling_time_multiplier : ℕ) : ℕ :=
  knife_time + peeling_time_multiplier * knife_time

theorem carla_total_time :
  total_time_spent 10 3 = 40 :=
by
  sorry

end NUMINAMATH_GPT_carla_total_time_l1445_144568


namespace NUMINAMATH_GPT_max_marks_for_test_l1445_144530

theorem max_marks_for_test (M : ℝ) (h1: (0.30 * M) = 180) : M = 600 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_for_test_l1445_144530


namespace NUMINAMATH_GPT_mean_equivalence_l1445_144534

theorem mean_equivalence {x : ℚ} :
  (8 + 15 + 21) / 3 = (18 + x) / 2 → x = 34 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_equivalence_l1445_144534


namespace NUMINAMATH_GPT_average_exp_Feb_to_Jul_l1445_144565

theorem average_exp_Feb_to_Jul (x y z : ℝ) 
    (h1 : 1200 + x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) = 6 * 4200) 
    (h2 : 0 ≤ x) 
    (h3 : 0 ≤ z) : 
    (x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) + 1500) / 6 = 4250 :=
by
    sorry

end NUMINAMATH_GPT_average_exp_Feb_to_Jul_l1445_144565


namespace NUMINAMATH_GPT_sum_of_constants_l1445_144598

variable (a b c : ℝ)

theorem sum_of_constants (h :  2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) :
  a + b + c = 9 := 
sorry

end NUMINAMATH_GPT_sum_of_constants_l1445_144598


namespace NUMINAMATH_GPT_tank_capacity_l1445_144513

theorem tank_capacity (one_third_full : ℚ) (added_water : ℚ) (capacity : ℚ) 
  (h1 : one_third_full = 1 / 3) 
  (h2 : 2 * one_third_full * capacity = 16) 
  (h3 : added_water = 16) 
  : capacity = 24 := 
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1445_144513


namespace NUMINAMATH_GPT_sum_non_prime_between_50_and_60_eq_383_l1445_144528

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def non_primes_between_50_and_60 : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58]

def sum_non_primes_between_50_and_60 : ℕ :=
  non_primes_between_50_and_60.sum

theorem sum_non_prime_between_50_and_60_eq_383 :
  sum_non_primes_between_50_and_60 = 383 :=
by
  sorry

end NUMINAMATH_GPT_sum_non_prime_between_50_and_60_eq_383_l1445_144528


namespace NUMINAMATH_GPT_roses_carnations_price_comparison_l1445_144552

variables (x y : ℝ)

theorem roses_carnations_price_comparison
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y :=
sorry

end NUMINAMATH_GPT_roses_carnations_price_comparison_l1445_144552


namespace NUMINAMATH_GPT_fraction_of_satisfactory_is_15_over_23_l1445_144506

def num_students_with_grade_A : ℕ := 6
def num_students_with_grade_B : ℕ := 5
def num_students_with_grade_C : ℕ := 4
def num_students_with_grade_D : ℕ := 2
def num_students_with_grade_F : ℕ := 6

def num_satisfactory_students : ℕ := 
  num_students_with_grade_A + num_students_with_grade_B + num_students_with_grade_C

def total_students : ℕ := 
  num_satisfactory_students + num_students_with_grade_D + num_students_with_grade_F

def fraction_satisfactory : ℚ := 
  (num_satisfactory_students : ℚ) / (total_students : ℚ)

theorem fraction_of_satisfactory_is_15_over_23 : 
  fraction_satisfactory = 15/23 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_fraction_of_satisfactory_is_15_over_23_l1445_144506


namespace NUMINAMATH_GPT_not_sufficient_nor_necessary_l1445_144522

theorem not_sufficient_nor_necessary (a b : ℝ) :
  ¬((a^2 > b^2) → (a > b)) ∧ ¬((a > b) → (a^2 > b^2)) :=
by
  sorry

end NUMINAMATH_GPT_not_sufficient_nor_necessary_l1445_144522


namespace NUMINAMATH_GPT_ratio_lcm_gcf_240_360_l1445_144554

theorem ratio_lcm_gcf_240_360 : Nat.lcm 240 360 / Nat.gcd 240 360 = 60 :=
by
  sorry

end NUMINAMATH_GPT_ratio_lcm_gcf_240_360_l1445_144554


namespace NUMINAMATH_GPT_mowing_time_l1445_144581

/-- 
Rena uses a mower to trim her "L"-shaped lawn which consists of two rectangular sections 
sharing one $50$-foot side. One section is $120$-foot by $50$-foot and the other is $70$-foot by 
$50$-foot. The mower has a swath width of $35$ inches with overlaps by $5$ inches. 
Rena walks at the rate of $4000$ feet per hour. 
Prove that it takes 0.95 hours for Rena to mow the entire lawn.
-/
theorem mowing_time 
  (length1 length2 width mower_swath overlap : ℝ) 
  (Rena_speed : ℝ) (effective_swath : ℝ) (total_area total_strips total_distance : ℝ)
  (h1 : length1 = 120)
  (h2 : length2 = 70)
  (h3 : width = 50)
  (h4 : mower_swath = 35 / 12)
  (h5 : overlap = 5 / 12)
  (h6 : effective_swath = mower_swath - overlap)
  (h7 : Rena_speed = 4000)
  (h8 : total_area = length1 * width + length2 * width)
  (h9 : total_strips = (length1 + length2) / effective_swath)
  (h10 : total_distance = total_strips * width) : 
  (total_distance / Rena_speed = 0.95) :=
by sorry

end NUMINAMATH_GPT_mowing_time_l1445_144581


namespace NUMINAMATH_GPT_min_value_ab_min_value_a_plus_2b_l1445_144572
open Nat

theorem min_value_ab (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 8 ≤ a * b :=
by
  sorry

theorem min_value_a_plus_2b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 9 ≤ a + 2 * b :=
by
  sorry

end NUMINAMATH_GPT_min_value_ab_min_value_a_plus_2b_l1445_144572


namespace NUMINAMATH_GPT_percentage_drop_l1445_144518

theorem percentage_drop (P N P' N' : ℝ) (h1 : N' = 1.60 * N) (h2 : P' * N' = 1.2800000000000003 * (P * N)) :
  P' = 0.80 * P :=
by
  sorry

end NUMINAMATH_GPT_percentage_drop_l1445_144518


namespace NUMINAMATH_GPT_cody_spent_tickets_l1445_144555

theorem cody_spent_tickets (initial_tickets lost_tickets remaining_tickets : ℝ) (h1 : initial_tickets = 49.0) (h2 : lost_tickets = 6.0) (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by
  sorry

end NUMINAMATH_GPT_cody_spent_tickets_l1445_144555


namespace NUMINAMATH_GPT_capsules_per_bottle_l1445_144584

-- Translating conditions into Lean definitions
def days := 180
def daily_serving_size := 2
def total_bottles := 6
def total_capsules_required := days * daily_serving_size

-- The statement to prove
theorem capsules_per_bottle : total_capsules_required / total_bottles = 60 :=
by
  sorry

end NUMINAMATH_GPT_capsules_per_bottle_l1445_144584


namespace NUMINAMATH_GPT_find_ratio_l1445_144546

noncomputable def decagon_area : ℝ := 12
noncomputable def area_below_PQ : ℝ := 6
noncomputable def unit_square_area : ℝ := 1
noncomputable def triangle_base : ℝ := 6
noncomputable def area_above_PQ : ℝ := 6
noncomputable def XQ : ℝ := 4
noncomputable def QY : ℝ := 2

theorem find_ratio {XQ QY : ℝ} (h1 : decagon_area = 12) (h2 : area_below_PQ = 6)
                   (h3 : unit_square_area = 1) (h4 : triangle_base = 6)
                   (h5 : area_above_PQ = 6) (h6 : XQ + QY = 6) :
  XQ / QY = 2 := by { sorry }

end NUMINAMATH_GPT_find_ratio_l1445_144546


namespace NUMINAMATH_GPT_no_lattice_points_on_hyperbola_l1445_144525

theorem no_lattice_points_on_hyperbola : ∀ x y : ℤ, x^2 - y^2 ≠ 2022 :=
by
  intro x y
  -- proof omitted
  sorry

end NUMINAMATH_GPT_no_lattice_points_on_hyperbola_l1445_144525


namespace NUMINAMATH_GPT_fraction_of_network_advertisers_l1445_144578

theorem fraction_of_network_advertisers 
  (total_advertisers : ℕ := 20) 
  (percentage_from_uni_a : ℝ := 0.75)
  (advertisers_from_uni_a := total_advertisers * percentage_from_uni_a) :
  (advertisers_from_uni_a / total_advertisers) = (3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_network_advertisers_l1445_144578


namespace NUMINAMATH_GPT_Chloe_second_round_points_l1445_144594

-- Conditions
def firstRoundPoints : ℕ := 40
def lastRoundPointsLost : ℕ := 4
def totalPoints : ℕ := 86
def secondRoundPoints : ℕ := 50

-- Statement to prove: Chloe scored 50 points in the second round
theorem Chloe_second_round_points :
  firstRoundPoints + secondRoundPoints - lastRoundPointsLost = totalPoints :=
by {
  -- Proof (not required, skipping with sorry)
  sorry
}

end NUMINAMATH_GPT_Chloe_second_round_points_l1445_144594


namespace NUMINAMATH_GPT_investment_amount_l1445_144560

noncomputable def total_investment (A T : ℝ) : Prop :=
  (0.095 * T = 0.09 * A + 2750) ∧ (T = A + 25000)

theorem investment_amount :
  ∃ T, ∀ A, total_investment A T ∧ T = 100000 :=
by
  sorry

end NUMINAMATH_GPT_investment_amount_l1445_144560


namespace NUMINAMATH_GPT_range_of_m_l1445_144591

theorem range_of_m (x m : ℝ)
  (h1 : (x + 2) / (10 - x) ≥ 0)
  (h2 : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (h3 : m < 0)
  (h4 : ∀ (x : ℝ), (x + 2) / (10 - x) ≥ 0 → (x^2 - 2 * x + 1 - m^2 ≤ 0)) :
  -3 ≤ m ∧ m < 0 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1445_144591


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1445_144597

noncomputable def a : ℚ := 0.66666667 -- Repeating decimal 0.666... corresponds to 2/3
noncomputable def b : ℚ := 0.22222223 -- Repeating decimal 0.222... corresponds to 2/9
noncomputable def c : ℚ := 0.44444445 -- Repeating decimal 0.444... corresponds to 4/9
noncomputable def d : ℚ := 0.99999999 -- Repeating decimal 0.999... corresponds to 1

theorem repeating_decimal_sum : a + b - c + d = 13 / 9 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1445_144597


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1445_144544

-- Given a quadratic inequality, prove the solution set in interval notation.
theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x ^ 2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1445_144544


namespace NUMINAMATH_GPT_determinant_of_roots_l1445_144557

noncomputable def determinant_expr (a b c d s p q r : ℝ) : ℝ :=
  by sorry

theorem determinant_of_roots (a b c d s p q r : ℝ)
    (h1 : a + b + c + d = -s)
    (h2 : abcd = r)
    (h3 : abc + abd + acd + bcd = -q)
    (h4 : ab + ac + bc = p) :
    determinant_expr a b c d s p q r = r - q + pq + p :=
  by sorry

end NUMINAMATH_GPT_determinant_of_roots_l1445_144557


namespace NUMINAMATH_GPT_triangle_is_obtuse_l1445_144531

theorem triangle_is_obtuse
  (α : ℝ)
  (h1 : α > 0 ∧ α < π)
  (h2 : Real.sin α + Real.cos α = 2 / 3) :
  ∃ β γ, β > 0 ∧ β < π ∧ γ > 0 ∧ γ < π ∧ β + γ + α = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2) :=
sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l1445_144531


namespace NUMINAMATH_GPT_train_travel_section_marked_l1445_144575

-- Definition of the metro structure with the necessary conditions.
structure Metro (Station : Type) :=
  (lines : List (Station × Station))
  (travel_time : Station → Station → ℕ)
  (terminal_turnaround : Station → Station)
  (transfer_station : Station → Station)

variable {Station : Type}

/-- The function that defines the bipolar coloring of the metro stations. -/
def station_color (s : Station) : ℕ := sorry  -- Placeholder for actual coloring function.

theorem train_travel_section_marked 
  (metro : Metro Station)
  (initial_station : Station)
  (end_station : Station)
  (travel_time : ℕ)
  (marked_section : Station × Station)
  (h_start : initial_station = marked_section.fst)
  (h_end : end_station = marked_section.snd)
  (h_travel_time : travel_time = 2016)
  (h_condition : ∀ s1 s2, (s1, s2) ∈ metro.lines → metro.travel_time s1 s2 = 1 ∧ 
                metro.terminal_turnaround s1 ≠ s1 ∧ metro.transfer_station s1 ≠ s2) :
  ∃ (time : ℕ), time = 2016 ∧ ∃ s1 s2, (s1, s2) = marked_section :=
sorry

end NUMINAMATH_GPT_train_travel_section_marked_l1445_144575


namespace NUMINAMATH_GPT_find_theta_l1445_144547

theorem find_theta (θ : ℝ) :
  (0 : ℝ) ≤ θ ∧ θ ≤ 2 * Real.pi →
  (∀ x, (0 : ℝ) ≤ x ∧ x ≤ 2 →
    x^2 * Real.cos θ - 2 * x * (1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intros hθ hx
  sorry

end NUMINAMATH_GPT_find_theta_l1445_144547


namespace NUMINAMATH_GPT_solve_for_x_l1445_144500

theorem solve_for_x (x y : ℕ) (h₁ : 9 ^ y = 3 ^ x) (h₂ : y = 6) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1445_144500


namespace NUMINAMATH_GPT_gwen_points_per_bag_l1445_144570

theorem gwen_points_per_bag : 
  ∀ (total_bags recycled_bags total_points_per_bag points_per_bag : ℕ),
  total_bags = 4 → 
  recycled_bags = total_bags - 2 →
  total_points_per_bag = 16 →
  points_per_bag = (total_points_per_bag / total_bags) →
  points_per_bag = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gwen_points_per_bag_l1445_144570


namespace NUMINAMATH_GPT_mass_of_added_water_with_temp_conditions_l1445_144516

theorem mass_of_added_water_with_temp_conditions
  (m_l : ℝ) (t_pi t_B t : ℝ) (c_B c_l lambda : ℝ) :
  m_l = 0.05 →
  t_pi = -10 →
  t_B = 10 →
  t = 0 →
  c_B = 4200 →
  c_l = 2100 →
  lambda = 3.3 * 10^5 →
  (0.0028 ≤ (2.1 * m_l * 10 + lambda * m_l) / (42 * 10) 
  ∧ (2.1 * m_l * 10) / (42 * 10) ≤ 0.418) :=
by
  sorry

end NUMINAMATH_GPT_mass_of_added_water_with_temp_conditions_l1445_144516


namespace NUMINAMATH_GPT_probability_team_A_3_points_probability_team_A_1_point_probability_combined_l1445_144596

namespace TeamProbabilities

noncomputable def P_team_A_3_points : ℚ :=
  (1 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_A_1_point : ℚ :=
  (1 / 3) * (2 / 3) * (2 / 3) + (2 / 3) * (1 / 3) * (2 / 3) + (2 / 3) * (2 / 3) * (1 / 3)

noncomputable def P_team_A_2_points : ℚ :=
  (1 / 3) * (1 / 3) * (2 / 3) + (1 / 3) * (2 / 3) * (1 / 3) + (2 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_B_1_point : ℚ :=
  (1 / 2) * (2 / 3) * (3 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (1 / 2) * (2 / 3) * (1 / 4) + (1 / 2) * (2 / 3) * (1 / 4) +
  (1 / 2) * (1 / 3) * (1 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (2 / 3) * (2 / 3) * (1 / 4) + (2 / 3) * (1 / 3) * (1 / 4)

noncomputable def combined_probability : ℚ :=
  P_team_A_2_points * P_team_B_1_point

theorem probability_team_A_3_points :
  P_team_A_3_points = 1 / 27 := by
  sorry

theorem probability_team_A_1_point :
  P_team_A_1_point = 4 / 9 := by
  sorry

theorem probability_combined :
  combined_probability = 11 / 108 := by
  sorry

end TeamProbabilities

end NUMINAMATH_GPT_probability_team_A_3_points_probability_team_A_1_point_probability_combined_l1445_144596


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l1445_144574

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : ∃ l : ℝ, l = 3 * b) (h2 : b * 3 * b = 675) : b = 15 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l1445_144574


namespace NUMINAMATH_GPT_cube_div_identity_l1445_144515

theorem cube_div_identity (a b : ℕ) (h1 : a = 6) (h2 : b = 3) : 
  (a^3 - b^3) / (a^2 + a * b + b^2) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_div_identity_l1445_144515


namespace NUMINAMATH_GPT_polynomial_root_cubic_sum_l1445_144564

theorem polynomial_root_cubic_sum
  (a b c : ℝ)
  (h : ∀ x : ℝ, (Polynomial.eval x (3 * Polynomial.X^3 + 5 * Polynomial.X^2 - 150 * Polynomial.X + 7) = 0)
    → x = a ∨ x = b ∨ x = c) :
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 :=
  sorry

end NUMINAMATH_GPT_polynomial_root_cubic_sum_l1445_144564


namespace NUMINAMATH_GPT_survey_total_people_l1445_144585

theorem survey_total_people (number_represented : ℕ) (percentage : ℝ) (h : number_represented = percentage * 200) : 
  (number_represented : ℝ) = 200 := 
by 
 sorry

end NUMINAMATH_GPT_survey_total_people_l1445_144585


namespace NUMINAMATH_GPT_total_cookies_eaten_l1445_144510

-- Definitions of the cookies eaten
def charlie_cookies := 15
def father_cookies := 10
def mother_cookies := 5

-- The theorem to prove the total number of cookies eaten
theorem total_cookies_eaten : charlie_cookies + father_cookies + mother_cookies = 30 := by
  sorry

end NUMINAMATH_GPT_total_cookies_eaten_l1445_144510


namespace NUMINAMATH_GPT_math_problem_l1445_144520

theorem math_problem (x : ℤ) :
  let a := 1990 * x + 1989
  let b := 1990 * x + 1990
  let c := 1990 * x + 1991
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1445_144520


namespace NUMINAMATH_GPT_reflect_origin_l1445_144517

theorem reflect_origin (x y : ℝ) (h₁ : x = 4) (h₂ : y = -3) : 
  (-x, -y) = (-4, 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_reflect_origin_l1445_144517


namespace NUMINAMATH_GPT_minimum_time_for_xiang_qing_fried_eggs_l1445_144509

-- Define the time taken for each individual step
def wash_scallions_time : ℕ := 1
def beat_eggs_time : ℕ := 1 / 2
def mix_egg_scallions_time : ℕ := 1
def wash_pan_time : ℕ := 1 / 2
def heat_pan_time : ℕ := 1 / 2
def heat_oil_time : ℕ := 1 / 2
def cook_dish_time : ℕ := 2

-- Define the total minimum time required
def minimum_time : ℕ := 5

-- The main theorem stating that the minimum time required is 5 minutes
theorem minimum_time_for_xiang_qing_fried_eggs :
  wash_scallions_time + beat_eggs_time + mix_egg_scallions_time + wash_pan_time + heat_pan_time + heat_oil_time + cook_dish_time = minimum_time := 
by sorry

end NUMINAMATH_GPT_minimum_time_for_xiang_qing_fried_eggs_l1445_144509


namespace NUMINAMATH_GPT_seokgi_jumped_furthest_l1445_144586

noncomputable def yooseung_jump : ℝ := 15 / 8
def shinyoung_jump : ℝ := 2
noncomputable def seokgi_jump : ℝ := 17 / 8

theorem seokgi_jumped_furthest :
  yooseung_jump < seokgi_jump ∧ shinyoung_jump < seokgi_jump :=
by
  sorry

end NUMINAMATH_GPT_seokgi_jumped_furthest_l1445_144586


namespace NUMINAMATH_GPT_sale_in_third_month_l1445_144503

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (average_sales : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 6470)
  (h_avg : average_sales = 6100) : 
  ∃ sale3, sale1 + sale2 + sale3 + sale4 + sale5 + sale6 = average_sales * 6 ∧ sale3 = 6200 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_third_month_l1445_144503
