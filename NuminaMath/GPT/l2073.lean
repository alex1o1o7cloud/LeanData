import Mathlib

namespace NUMINAMATH_GPT_john_spent_expected_amount_l2073_207313

-- Define the original price of each pin
def original_price : ℝ := 20

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the number of pins
def number_of_pins : ℝ := 10

-- Define the sales tax rate
def tax_rate : ℝ := 0.08

-- Calculate the discount on each pin
def discount_per_pin : ℝ := discount_rate * original_price

-- Calculate the discounted price per pin
def discounted_price_per_pin : ℝ := original_price - discount_per_pin

-- Calculate the total discounted price for all pins
def total_discounted_price : ℝ := discounted_price_per_pin * number_of_pins

-- Calculate the sales tax on the total discounted price
def sales_tax : ℝ := tax_rate * total_discounted_price

-- Calculate the total amount spent including sales tax
def total_amount_spent : ℝ := total_discounted_price + sales_tax

-- The theorem that John spent $183.60 on pins including the sales tax
theorem john_spent_expected_amount : total_amount_spent = 183.60 :=
by
  sorry

end NUMINAMATH_GPT_john_spent_expected_amount_l2073_207313


namespace NUMINAMATH_GPT_roots_inequality_l2073_207329

theorem roots_inequality (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) :
  -1 ≤ z ∧ z ≤ 13 / 3 :=
sorry

end NUMINAMATH_GPT_roots_inequality_l2073_207329


namespace NUMINAMATH_GPT_length_of_train_l2073_207336

theorem length_of_train (speed_kmh : ℕ) (time_seconds : ℕ) (h_speed : speed_kmh = 60) (h_time : time_seconds = 36) :
  let time_hours := (time_seconds : ℚ) / 3600
  let distance_km := (speed_kmh : ℚ) * time_hours
  let distance_m := distance_km * 1000
  distance_m = 600 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l2073_207336


namespace NUMINAMATH_GPT_nine_possible_xs_l2073_207395

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end NUMINAMATH_GPT_nine_possible_xs_l2073_207395


namespace NUMINAMATH_GPT_total_and_per_suitcase_profit_l2073_207386

theorem total_and_per_suitcase_profit
  (num_suitcases : ℕ)
  (purchase_price_per_suitcase : ℕ)
  (total_sales_revenue : ℕ)
  (total_profit : ℕ)
  (profit_per_suitcase : ℕ)
  (h_num_suitcases : num_suitcases = 60)
  (h_purchase_price : purchase_price_per_suitcase = 100)
  (h_total_sales : total_sales_revenue = 8100)
  (h_total_profit : total_profit = total_sales_revenue - num_suitcases * purchase_price_per_suitcase)
  (h_profit_per_suitcase : profit_per_suitcase = total_profit / num_suitcases) :
  total_profit = 2100 ∧ profit_per_suitcase = 35 := by
  sorry

end NUMINAMATH_GPT_total_and_per_suitcase_profit_l2073_207386


namespace NUMINAMATH_GPT_sum_of_squares_ge_one_third_l2073_207356

theorem sum_of_squares_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1/3 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_ge_one_third_l2073_207356


namespace NUMINAMATH_GPT_factorize_x4_minus_64_l2073_207383

theorem factorize_x4_minus_64 (x : ℝ) : (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by sorry

end NUMINAMATH_GPT_factorize_x4_minus_64_l2073_207383


namespace NUMINAMATH_GPT_central_park_trash_cans_more_than_half_l2073_207327

theorem central_park_trash_cans_more_than_half
  (C : ℕ)  -- Original number of trash cans in Central Park
  (V : ℕ := 24)  -- Original number of trash cans in Veteran's Park
  (V_now : ℕ := 34)  -- Number of trash cans in Veteran's Park after the move
  (H_move : (V_now - V) = C / 2)  -- Condition of trash cans moved
  (H_C : C = (1 / 2) * V + x)  -- Central Park had more than half trash cans as Veteran's Park, where x is an excess amount
  : C - (1 / 2) * V = 8 := 
sorry

end NUMINAMATH_GPT_central_park_trash_cans_more_than_half_l2073_207327


namespace NUMINAMATH_GPT_dice_probability_divisible_by_three_ge_one_fourth_l2073_207338

theorem dice_probability_divisible_by_three_ge_one_fourth
  (p q r : ℝ) 
  (h1 : 0 ≤ p) (h2 : 0 ≤ q) (h3 : 0 ≤ r) 
  (h4 : p + q + r = 1) : 
  p^3 + q^3 + r^3 + 6 * p * q * r ≥ 1 / 4 :=
sorry

end NUMINAMATH_GPT_dice_probability_divisible_by_three_ge_one_fourth_l2073_207338


namespace NUMINAMATH_GPT_part_length_proof_l2073_207339

-- Define the scale length in feet and inches
def scale_length_ft : ℕ := 6
def scale_length_inch : ℕ := 8

-- Define the number of equal parts
def num_parts : ℕ := 4

-- Calculate total length in inches
def total_length_inch : ℕ := scale_length_ft * 12 + scale_length_inch

-- Calculate the length of each part in inches
def part_length_inch : ℕ := total_length_inch / num_parts

-- Prove that each part is 1 foot 8 inches long
theorem part_length_proof :
  part_length_inch = 1 * 12 + 8 :=
by
  sorry

end NUMINAMATH_GPT_part_length_proof_l2073_207339


namespace NUMINAMATH_GPT_inequality_solution_l2073_207342

-- We define the problem
def interval_of_inequality : Set ℝ := { x : ℝ | (x + 1) * (2 - x) > 0 }

-- We define the expected solution set
def expected_solution_set : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

-- The theorem to be proved
theorem inequality_solution :
  interval_of_inequality = expected_solution_set := by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l2073_207342


namespace NUMINAMATH_GPT_non_empty_solution_set_range_l2073_207331

theorem non_empty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 :=
sorry

end NUMINAMATH_GPT_non_empty_solution_set_range_l2073_207331


namespace NUMINAMATH_GPT_sum_of_roots_gt_two_l2073_207322

noncomputable def f : ℝ → ℝ := λ x => Real.log x - x + 1

theorem sum_of_roots_gt_two (m : ℝ) (x1 x2 : ℝ) (hx1 : f x1 = m) (hx2 : f x2 = m) (hne : x1 ≠ x2) : x1 + x2 > 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_roots_gt_two_l2073_207322


namespace NUMINAMATH_GPT_min_angle_for_quadrilateral_l2073_207371

theorem min_angle_for_quadrilateral (d : ℝ) (h : ∀ (a b c d : ℝ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a + b + c + d = 360 → (a < d ∨ b < d)) :
  d = 120 :=
by
  sorry

end NUMINAMATH_GPT_min_angle_for_quadrilateral_l2073_207371


namespace NUMINAMATH_GPT_no_real_roots_of_polynomial_l2073_207346

noncomputable def p (x : ℝ) : ℝ := sorry

theorem no_real_roots_of_polynomial (p : ℝ → ℝ) (h_deg : ∃ n : ℕ, n ≥ 1 ∧ ∀ x: ℝ, p x = x^n) :
  (∀ x, p x * p (2 * x^2) = p (3 * x^3 + x)) →
  ¬ ∃ α : ℝ, p α = 0 := sorry

end NUMINAMATH_GPT_no_real_roots_of_polynomial_l2073_207346


namespace NUMINAMATH_GPT_center_of_circle_in_second_or_fourth_quadrant_l2073_207334

theorem center_of_circle_in_second_or_fourth_quadrant
  (α : ℝ) 
  (hyp1 : ∀ x y : ℝ, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0 → Real.cos α * Real.sin α > 0)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x*Real.cos α - 2*y*Real.sin α = 0) :
  (-Real.cos α > 0 ∧ Real.sin α > 0) ∨ (-Real.cos α < 0 ∧ Real.sin α < 0) :=
sorry

end NUMINAMATH_GPT_center_of_circle_in_second_or_fourth_quadrant_l2073_207334


namespace NUMINAMATH_GPT_value_of_b_l2073_207308

theorem value_of_b (y : ℝ) (b : ℝ) (h_pos : y > 0) (h_eqn : (7 * y) / b + (3 * y) / 10 = 0.6499999999999999 * y) : 
  b = 70 / 61.99999999999999 :=
sorry

end NUMINAMATH_GPT_value_of_b_l2073_207308


namespace NUMINAMATH_GPT_football_problem_l2073_207317

-- Definitions based on conditions
def total_balls (x y : Nat) : Prop := x + y = 200
def total_cost (x y : Nat) : Prop := 80 * x + 60 * y = 14400
def football_A_profit_per_ball : Nat := 96 - 80
def football_B_profit_per_ball : Nat := 81 - 60
def total_profit (x y : Nat) : Nat :=
  football_A_profit_per_ball * x + football_B_profit_per_ball * y

-- Lean statement proving the conditions lead to the solution
theorem football_problem
  (x y : Nat)
  (h1 : total_balls x y)
  (h2 : total_cost x y)
  (h3 : x = 120)
  (h4 : y = 80) :
  total_profit x y = 3600 := by
  sorry

end NUMINAMATH_GPT_football_problem_l2073_207317


namespace NUMINAMATH_GPT_common_fraction_l2073_207316

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end NUMINAMATH_GPT_common_fraction_l2073_207316


namespace NUMINAMATH_GPT_count_squares_3x3_grid_count_squares_5x5_grid_l2073_207388

/-- Define a mathematical problem: 
  Prove that the number of squares with all four vertices on the dots in a 3x3 grid is 4.
  Prove that the number of squares with all four vertices on the dots in a 5x5 grid is 50.
-/

def num_squares_3x3 : Nat := 4
def num_squares_5x5 : Nat := 50

theorem count_squares_3x3_grid : 
  ∀ (grid_size : Nat), grid_size = 3 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_3x3 = 4)) := 
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

theorem count_squares_5x5_grid : 
  ∀ (grid_size : Nat), grid_size = 5 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_5x5 = 50)) :=
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

end NUMINAMATH_GPT_count_squares_3x3_grid_count_squares_5x5_grid_l2073_207388


namespace NUMINAMATH_GPT_remainder_of_sum_divided_by_14_l2073_207377

def consecutive_odds : List ℤ := [12157, 12159, 12161, 12163, 12165, 12167, 12169]

def sum_of_consecutive_odds := consecutive_odds.sum

theorem remainder_of_sum_divided_by_14 :
  (sum_of_consecutive_odds % 14) = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_divided_by_14_l2073_207377


namespace NUMINAMATH_GPT_minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l2073_207359

theorem minimal_distance_ln_x_x :
  ∀ (x : ℝ), x > 0 → ∃ (d : ℝ), d = |Real.log x - x| → d ≥ 0 :=
by sorry

theorem minimal_distance_graphs_ex_ln_x :
  ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), ∃ (d : ℝ), y = d → d = 2 :=
by sorry

end NUMINAMATH_GPT_minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l2073_207359


namespace NUMINAMATH_GPT_circumscribed_circle_area_l2073_207384

/-- 
Statement: The area of the circle circumscribed about an equilateral triangle with side lengths of 9 units is 27π square units.
-/
theorem circumscribed_circle_area (s : ℕ) (h : s = 9) : 
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)
  area = 27 * Real.pi :=
by
  -- Axis and conditions definitions
  have := h

  -- Definition for the area based on the radius
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)

  -- Statement of the equality to be proven
  show area = 27 * Real.pi
  sorry

end NUMINAMATH_GPT_circumscribed_circle_area_l2073_207384


namespace NUMINAMATH_GPT_a_plus_b_plus_c_at_2_l2073_207385

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def maximum_value (a b c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic a b c x = 75

def passes_through (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  quadratic a b c p1.1 = p1.2 ∧ quadratic a b c p2.1 = p2.2

theorem a_plus_b_plus_c_at_2 
  (a b c : ℝ)
  (hmax : maximum_value a b c)
  (hpoints : passes_through a b c (-3, 0) (3, 0))
  (hvertex : ∀ x : ℝ, quadratic a 0 c x ≤ quadratic a (2 * b) c 0) : 
  quadratic a b c 2 = 125 / 3 :=
sorry

end NUMINAMATH_GPT_a_plus_b_plus_c_at_2_l2073_207385


namespace NUMINAMATH_GPT_solution_set_inequality_l2073_207335

open Set

theorem solution_set_inequality :
  {x : ℝ | (x+1)/(x-4) ≥ 3} = Iio 4 ∪ Ioo 4 (13/2) ∪ {13/2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l2073_207335


namespace NUMINAMATH_GPT_minimum_value_function_inequality_ln_l2073_207328

noncomputable def f (x : ℝ) := x * Real.log x

theorem minimum_value_function (t : ℝ) (ht : 0 < t) :
  ∃ (xmin : ℝ), xmin = if (0 < t ∧ t < 1 / Real.exp 1) then -1 / Real.exp 1 else t * Real.log t :=
sorry

theorem inequality_ln (x : ℝ) (hx : 0 < x) : 
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end NUMINAMATH_GPT_minimum_value_function_inequality_ln_l2073_207328


namespace NUMINAMATH_GPT_find_natural_triples_l2073_207325

theorem find_natural_triples (x y z : ℕ) : 
  (x+1) * (y+1) * (z+1) = 3 * x * y * z ↔ 
  (x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 2) ∨ (x, y, z) = (3, 2, 2) ∨
  (x, y, z) = (5, 1, 4) ∨ (x, y, z) = (5, 4, 1) ∨ (x, y, z) = (4, 1, 5) ∨ (x, y, z) = (4, 5, 1) ∨ 
  (x, y, z) = (1, 4, 5) ∨ (x, y, z) = (1, 5, 4) ∨ (x, y, z) = (8, 1, 3) ∨ (x, y, z) = (8, 3, 1) ∨
  (x, y, z) = (3, 1, 8) ∨ (x, y, z) = (3, 8, 1) ∨ (x, y, z) = (1, 3, 8) ∨ (x, y, z) = (1, 8, 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_natural_triples_l2073_207325


namespace NUMINAMATH_GPT_find_c_k_l2073_207397

theorem find_c_k (a b : ℕ → ℕ) (c : ℕ → ℕ) (k : ℕ) (d r : ℕ) 
  (h1 : ∀ n, a n = 1 + (n-1)*d)
  (h2 : ∀ n, b n = r^(n-1))
  (h3 : ∀ n, c n = a n + b n)
  (h4 : c (k-1) = 80)
  (h5 : c (k+1) = 500) :
  c k = 167 := sorry

end NUMINAMATH_GPT_find_c_k_l2073_207397


namespace NUMINAMATH_GPT_inverse_proposition_l2073_207360

theorem inverse_proposition (q_1 q_2 : ℚ) :
  (q_1 ^ 2 = q_2 ^ 2 → q_1 = q_2) ↔ (q_1 = q_2 → q_1 ^ 2 = q_2 ^ 2) :=
sorry

end NUMINAMATH_GPT_inverse_proposition_l2073_207360


namespace NUMINAMATH_GPT_tug_of_war_matches_l2073_207354

-- Define the number of classes
def num_classes : ℕ := 7

-- Define the number of matches Grade 3 Class 6 competes in
def matches_class6 : ℕ := num_classes - 1

-- Define the total number of matches
def total_matches : ℕ := (num_classes - 1) * num_classes / 2

-- Main theorem stating the problem
theorem tug_of_war_matches :
  matches_class6 = 6 ∧ total_matches = 21 := by
  sorry

end NUMINAMATH_GPT_tug_of_war_matches_l2073_207354


namespace NUMINAMATH_GPT_tan_neg_405_eq_neg_1_l2073_207301

theorem tan_neg_405_eq_neg_1 :
  (Real.tan (-405 * Real.pi / 180) = -1) ∧
  (∀ θ : ℝ, Real.tan (θ + 2 * Real.pi) = Real.tan θ) ∧
  (Real.tan θ = Real.sin θ / Real.cos θ) ∧
  (Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2) ∧
  (Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_GPT_tan_neg_405_eq_neg_1_l2073_207301


namespace NUMINAMATH_GPT_square_area_l2073_207304

theorem square_area (x : ℝ) (s : ℝ) 
  (h1 : s^2 + s^2 = (2 * x)^2) 
  (h2 : 4 * s = 16 * x) : s^2 = 16 * x^2 :=
by {
  sorry -- Proof not required
}

end NUMINAMATH_GPT_square_area_l2073_207304


namespace NUMINAMATH_GPT_CanVolume_l2073_207352

variable (X Y : Type) [Field X] [Field Y] (V W : X)

theorem CanVolume (mix_ratioX mix_ratioY drawn_volume new_ratioX new_ratioY : ℤ)
  (h1 : mix_ratioX = 5) (h2 : mix_ratioY = 7) (h3 : drawn_volume = 12) 
  (h4 : new_ratioX = 4) (h5 : new_ratioY = 7) :
  V = 72 ∧ W = 72 := 
sorry

end NUMINAMATH_GPT_CanVolume_l2073_207352


namespace NUMINAMATH_GPT_urn_contains_three_red_three_blue_after_five_operations_l2073_207333

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5

noncomputable def calculate_probability (initial_red: ℕ) (initial_blue: ℕ) (operations: ℕ) : ℚ :=
  sorry

theorem urn_contains_three_red_three_blue_after_five_operations :
  calculate_probability initial_red_balls initial_blue_balls total_operations = 8 / 105 :=
by sorry

end NUMINAMATH_GPT_urn_contains_three_red_three_blue_after_five_operations_l2073_207333


namespace NUMINAMATH_GPT_perimeter_of_star_is_160_l2073_207392

-- Define the radius of the circles
def radius := 5 -- in cm

-- Define the diameter based on radius
def diameter := 2 * radius

-- Define the side length of the square
def side_length_square := 2 * diameter

-- Define the side length of each equilateral triangle
def side_length_triangle := side_length_square

-- Define the perimeter of the four-pointed star
def perimeter_star := 8 * side_length_triangle

-- Statement: The perimeter of the star is 160 cm
theorem perimeter_of_star_is_160 :
  perimeter_star = 160 := by
    sorry

end NUMINAMATH_GPT_perimeter_of_star_is_160_l2073_207392


namespace NUMINAMATH_GPT_add_2001_1015_l2073_207381

theorem add_2001_1015 : 2001 + 1015 = 3016 := 
by
  sorry

end NUMINAMATH_GPT_add_2001_1015_l2073_207381


namespace NUMINAMATH_GPT_musical_chairs_l2073_207314

def is_prime_power (m : ℕ) : Prop :=
  ∃ (p k : ℕ), Nat.Prime p ∧ k > 0 ∧ m = p ^ k

theorem musical_chairs (n m : ℕ) (h1 : 1 < m) (h2 : m ≤ n) (h3 : ¬ is_prime_power m) :
  ∃ f : Fin n → Fin n, (∀ x, f x ≠ x) ∧ (∀ x, (f^[m]) x = x) :=
sorry

end NUMINAMATH_GPT_musical_chairs_l2073_207314


namespace NUMINAMATH_GPT_hyperbola_equation_l2073_207341

-- Definitions based on problem conditions
def asymptotes (x y : ℝ) : Prop :=
  y = (1/3) * x ∨ y = -(1/3) * x

def focus (p : ℝ × ℝ) : Prop :=
  p = (Real.sqrt 10, 0)

-- The main statement to prove
theorem hyperbola_equation :
  (∃ p, focus p) ∧ (∀ (x y : ℝ), asymptotes x y) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 = 1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l2073_207341


namespace NUMINAMATH_GPT_problem_ABCD_cos_l2073_207337

/-- In convex quadrilateral ABCD, angle A = 2 * angle C, AB = 200, CD = 200, the perimeter of 
ABCD is 720, and AD ≠ BC. Find the floor of 1000 * cos A. -/
theorem problem_ABCD_cos (A C : ℝ) (AB CD AD BC : ℝ) (h1 : AB = 200)
  (h2 : CD = 200) (h3 : AD + BC = 320) (h4 : A = 2 * C)
  (h5 : AD ≠ BC) : ⌊1000 * Real.cos A⌋ = 233 := 
sorry

end NUMINAMATH_GPT_problem_ABCD_cos_l2073_207337


namespace NUMINAMATH_GPT_canteen_consumption_l2073_207398

theorem canteen_consumption :
  ∀ (x : ℕ),
    (x + (500 - x) + (200 - x)) = 700 → 
    (500 - x) = 7 * (200 - x) →
    x = 150 :=
by
  sorry

end NUMINAMATH_GPT_canteen_consumption_l2073_207398


namespace NUMINAMATH_GPT_fraction_of_sophomores_attending_fair_l2073_207326

theorem fraction_of_sophomores_attending_fair
  (s j n : ℕ)
  (h1 : s = j)
  (h2 : j = n)
  (soph_attend : ℚ)
  (junior_attend : ℚ)
  (senior_attend : ℚ)
  (fraction_s : soph_attend = 4/5 * s)
  (fraction_j : junior_attend = 3/4 * j)
  (fraction_n : senior_attend = 1/3 * n) :
  soph_attend / (soph_attend + junior_attend + senior_attend) = 240 / 565 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_sophomores_attending_fair_l2073_207326


namespace NUMINAMATH_GPT_quadratic_expression_rewrite_l2073_207343

theorem quadratic_expression_rewrite :
  ∃ a b c : ℚ, (∀ k : ℚ, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧ c + 3 * b = -49/3 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_rewrite_l2073_207343


namespace NUMINAMATH_GPT_dividend_is_144_l2073_207396

theorem dividend_is_144 
  (Q : ℕ) (D : ℕ) (M : ℕ)
  (h1 : M = 6 * D)
  (h2 : D = 4 * Q) 
  (Q_eq_6 : Q = 6) : 
  M = 144 := 
sorry

end NUMINAMATH_GPT_dividend_is_144_l2073_207396


namespace NUMINAMATH_GPT_shortest_chord_through_point_l2073_207357

theorem shortest_chord_through_point
  (correct_length : ℝ)
  (h1 : correct_length = 2 * Real.sqrt 2)
  (circle_eq : ∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = 4)
  (passes_point : ∀ (p : ℝ × ℝ), p = (3, 1)) :
  correct_length = 2 * Real.sqrt 2 :=
by {
  -- the proof steps would go here
  sorry
}

end NUMINAMATH_GPT_shortest_chord_through_point_l2073_207357


namespace NUMINAMATH_GPT_boxes_needed_l2073_207348

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end NUMINAMATH_GPT_boxes_needed_l2073_207348


namespace NUMINAMATH_GPT_perfect_square_fraction_l2073_207376

theorem perfect_square_fraction (n : ℤ) : 
  n < 30 ∧ ∃ k : ℤ, (n / (30 - n)) = k^2 → ∃ cnt : ℕ, cnt = 4 :=
  by
  sorry

end NUMINAMATH_GPT_perfect_square_fraction_l2073_207376


namespace NUMINAMATH_GPT_second_odd_integer_is_72_l2073_207366

def consecutive_odd_integers (n : ℤ) : ℤ × ℤ × ℤ :=
  (n - 2, n, n + 2)

theorem second_odd_integer_is_72 (n : ℤ) (h : (n - 2) + (n + 2) = 144) : n = 72 :=
by {
  sorry
}

end NUMINAMATH_GPT_second_odd_integer_is_72_l2073_207366


namespace NUMINAMATH_GPT_masha_happy_max_l2073_207311

/-- Masha has 2021 weights, all with unique masses. She places weights one at a 
time on a two-pan balance scale without removing previously placed weights. 
Every time the scale balances, Masha feels happy. Prove that the maximum number 
of times she can find the scales in perfect balance is 673. -/
theorem masha_happy_max (weights : Finset ℕ) (h_unique : weights.card = 2021) : 
  ∃ max_happy_times : ℕ, max_happy_times = 673 := 
sorry

end NUMINAMATH_GPT_masha_happy_max_l2073_207311


namespace NUMINAMATH_GPT_no_solution_for_lcm_gcd_eq_l2073_207330

theorem no_solution_for_lcm_gcd_eq (n : ℕ) (h₁ : n ∣ 60) (h₂ : Nat.Prime n) :
  ¬(Nat.lcm n 60 = Nat.gcd n 60 + 200) :=
  sorry

end NUMINAMATH_GPT_no_solution_for_lcm_gcd_eq_l2073_207330


namespace NUMINAMATH_GPT_number_of_rows_is_ten_l2073_207389

-- Definition of the arithmetic sequence
def arithmetic_sequence_sum (n : ℕ) : ℕ :=
  n * (3 * n + 1) / 2

-- The main theorem to prove
theorem number_of_rows_is_ten :
  (∃ n : ℕ, arithmetic_sequence_sum n = 145) ↔ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rows_is_ten_l2073_207389


namespace NUMINAMATH_GPT_goods_train_speed_l2073_207369

def train_speed_km_per_hr (length_of_train length_of_platform time_to_cross : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_platform
  let speed_m_s := total_distance / time_to_cross
  speed_m_s * 36 / 10

-- Define the conditions given in the problem
def length_of_train : ℕ := 310
def length_of_platform : ℕ := 210
def time_to_cross : ℕ := 26

-- Define the target speed
def target_speed : ℕ := 72

-- The theorem proving the conclusion
theorem goods_train_speed :
  train_speed_km_per_hr length_of_train length_of_platform time_to_cross = target_speed := by
  sorry

end NUMINAMATH_GPT_goods_train_speed_l2073_207369


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2073_207380

theorem solution_set_of_inequality :
  { x : ℝ | (x - 3) * (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2073_207380


namespace NUMINAMATH_GPT_equilateral_triangle_on_parallel_lines_l2073_207323

theorem equilateral_triangle_on_parallel_lines 
  (l1 l2 l3 : ℝ → Prop)
  (h_parallel_12 : ∀ x y, l1 x → l2 y → ∀ z, l1 z → l2 z)
  (h_parallel_23 : ∀ x y, l2 x → l3 y → ∀ z, l2 z → l3 z) 
  (h_parallel_13 : ∀ x y, l1 x → l3 y → ∀ z, l1 z → l3 z) 
  (A : ℝ) (hA : l1 A)
  (B : ℝ) (hB : l2 B)
  (C : ℝ) (hC : l3 C):
  ∃ A B C : ℝ, l1 A ∧ l2 B ∧ l3 C ∧ (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_on_parallel_lines_l2073_207323


namespace NUMINAMATH_GPT_problem_1_part_1_proof_problem_1_part_2_proof_l2073_207300

noncomputable def problem_1_part_1 : Real :=
  2 * Real.sqrt 2 + (Real.sqrt 6) / 2

theorem problem_1_part_1_proof:
  let θ₀ := 3 * Real.pi / 4
  let ρ_A := 4 * Real.cos θ₀
  let ρ_B := Real.sqrt 3 * Real.sin θ₀
  |ρ_A - ρ_B| = 2 * Real.sqrt 2 + (Real.sqrt 6) / 2 :=
  sorry

theorem problem_1_part_2_proof :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x - (Real.sqrt 3)/2 * y = 0) :=
  sorry

end NUMINAMATH_GPT_problem_1_part_1_proof_problem_1_part_2_proof_l2073_207300


namespace NUMINAMATH_GPT_negation_of_exists_l2073_207340

theorem negation_of_exists : (¬ ∃ x : ℝ, x > 0 ∧ x^2 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_l2073_207340


namespace NUMINAMATH_GPT_lcm_two_primes_is_10_l2073_207364

theorem lcm_two_primes_is_10 (x y : ℕ) (h_prime_x : Nat.Prime x) (h_prime_y : Nat.Prime y) (h_lcm : Nat.lcm x y = 10) (h_gt : x > y) : 2 * x + y = 12 :=
sorry

end NUMINAMATH_GPT_lcm_two_primes_is_10_l2073_207364


namespace NUMINAMATH_GPT_inverse_of_g_l2073_207387

theorem inverse_of_g : 
  ∀ (g g_inv : ℝ → ℝ) (p q r s : ℝ),
  (∀ x, g x = (3 * x - 2) / (x + 4)) →
  (∀ x, g_inv x = (p * x + q) / (r * x + s)) →
  (∀ x, g (g_inv x) = x) →
  q / s = 2 / 3 :=
by
  intros g g_inv p q r s h_g h_g_inv h_g_ginv
  sorry

end NUMINAMATH_GPT_inverse_of_g_l2073_207387


namespace NUMINAMATH_GPT_maximum_xyzw_l2073_207324

theorem maximum_xyzw (x y z w : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_w : 0 < w)
(h : (x * y * z) + w = (x + w) * (y + w) * (z + w))
(h_sum : x + y + z + w = 1) :
  xyzw = 1 / 256 :=
sorry

end NUMINAMATH_GPT_maximum_xyzw_l2073_207324


namespace NUMINAMATH_GPT_white_tshirts_per_pack_l2073_207374

-- Define the given conditions
def packs_white := 5
def packs_blue := 3
def t_shirts_per_blue_pack := 9
def total_t_shirts := 57

-- Define the total number of blue t-shirts
def total_blue_t_shirts := packs_blue * t_shirts_per_blue_pack

-- Define the variable W for the number of white t-shirts per pack
variable (W : ℕ)

-- Define the total number of white t-shirts
def total_white_t_shirts := packs_white * W

-- State the theorem to prove
theorem white_tshirts_per_pack :
    total_white_t_shirts + total_blue_t_shirts = total_t_shirts → W = 6 :=
by
  sorry

end NUMINAMATH_GPT_white_tshirts_per_pack_l2073_207374


namespace NUMINAMATH_GPT_find_m_from_hyperbola_and_parabola_l2073_207378

theorem find_m_from_hyperbola_and_parabola (a m : ℝ) 
  (h_eccentricity : (Real.sqrt (a^2 + 4)) / a = 3 * Real.sqrt 5 / 5) 
  (h_focus_coincide : (m / 4) = -3) : m = -12 := 
  sorry

end NUMINAMATH_GPT_find_m_from_hyperbola_and_parabola_l2073_207378


namespace NUMINAMATH_GPT_even_abs_func_necessary_not_sufficient_l2073_207350

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem even_abs_func_necessary_not_sufficient (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, |f (-x)| = |f x|) ∧ (∃ g : ℝ → ℝ, (∀ x : ℝ, |g (-x)| = |g x|) ∧ ¬(∀ x : ℝ, g (-x) = -g x)) :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_even_abs_func_necessary_not_sufficient_l2073_207350


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2073_207312

open Complex

theorem sufficient_not_necessary_condition (a b : ℝ) (i := Complex.I) :
  (a = 1 ∧ b = 1) → ((a + b * i)^2 = 2 * i) ∧ ¬((a + b * i)^2 = 2 * i → a = 1 ∧ b = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2073_207312


namespace NUMINAMATH_GPT_least_sales_needed_not_lose_money_l2073_207347

noncomputable def old_salary : ℝ := 75000
noncomputable def new_salary_base : ℝ := 45000
noncomputable def commission_rate : ℝ := 0.15
noncomputable def sale_amount : ℝ := 750

theorem least_sales_needed_not_lose_money : 
  ∃ (n : ℕ), n * (commission_rate * sale_amount) ≥ (old_salary - new_salary_base) ∧ n = 267 := 
by
  -- The proof will show that n = 267 is the least number of sales needed to not lose money.
  existsi 267
  sorry

end NUMINAMATH_GPT_least_sales_needed_not_lose_money_l2073_207347


namespace NUMINAMATH_GPT_meaningful_domain_l2073_207349

def is_meaningful (x : ℝ) : Prop :=
  (x - 1) ≠ 0

theorem meaningful_domain (x : ℝ) : is_meaningful x ↔ (x ≠ 1) :=
  sorry

end NUMINAMATH_GPT_meaningful_domain_l2073_207349


namespace NUMINAMATH_GPT_max_points_of_intersection_l2073_207319

theorem max_points_of_intersection (circles : ℕ) (line : ℕ) (h_circles : circles = 3) (h_line : line = 1) : 
  ∃ points_of_intersection, points_of_intersection = 12 :=
by
  -- Proof here (omitted)
  sorry

end NUMINAMATH_GPT_max_points_of_intersection_l2073_207319


namespace NUMINAMATH_GPT_lion_room_is_3_l2073_207307

/-!
  A lion is hidden in one of three rooms. A note on the door of room 1 reads "The lion is here".
  A note on the door of room 2 reads "The lion is not here". A note on the door of room 3 reads "2+3=2×3".
  Only one of these notes is true. Prove that the lion is in room 3.
-/

def note1 (lion_room : ℕ) : Prop := lion_room = 1
def note2 (lion_room : ℕ) : Prop := lion_room ≠ 2
def note3 (lion_room : ℕ) : Prop := 2 + 3 = 2 * 3
def lion_is_in_room3 : Prop := ∀ lion_room, (note1 lion_room ∨ note2 lion_room ∨ note3 lion_room) ∧
  (note1 lion_room → note2 lion_room = false) ∧ (note1 lion_room → note3 lion_room = false) ∧
  (note2 lion_room → note1 lion_room = false) ∧ (note2 lion_room → note3 lion_room = false) ∧
  (note3 lion_room → note1 lion_room = false) ∧ (note3 lion_room → note2 lion_room = false) → lion_room = 3

theorem lion_room_is_3 : lion_is_in_room3 := 
  by
  sorry

end NUMINAMATH_GPT_lion_room_is_3_l2073_207307


namespace NUMINAMATH_GPT_solve_for_n_l2073_207372

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l2073_207372


namespace NUMINAMATH_GPT_binary_to_octal_equivalence_l2073_207318

theorem binary_to_octal_equivalence : (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) 
                                    = (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end NUMINAMATH_GPT_binary_to_octal_equivalence_l2073_207318


namespace NUMINAMATH_GPT_average_time_per_mile_l2073_207315

-- Define the conditions
def total_distance_miles : ℕ := 24
def total_time_hours : ℕ := 3
def total_time_minutes : ℕ := 36
def total_time_in_minutes : ℕ := (total_time_hours * 60) + total_time_minutes

-- State the theorem
theorem average_time_per_mile : total_time_in_minutes / total_distance_miles = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_time_per_mile_l2073_207315


namespace NUMINAMATH_GPT_marcia_wardrobe_cost_l2073_207344

theorem marcia_wardrobe_cost :
  let skirt_price := 20
  let blouse_price := 15
  let pant_price := 30
  let num_skirts := 3
  let num_blouses := 5
  let num_pants := 2
  let pant_offer := buy_1_get_1_half
  let skirt_cost := num_skirts * skirt_price
  let blouse_cost := num_blouses * blouse_price
  let pant_full_price := pant_price
  let pant_half_price := pant_price / 2
  let pant_cost := pant_full_price + pant_half_price
  let total_cost := skirt_cost + blouse_cost + pant_cost
  total_cost = 180 :=
by
  sorry -- proof is omitted

end NUMINAMATH_GPT_marcia_wardrobe_cost_l2073_207344


namespace NUMINAMATH_GPT_cricket_initial_avg_runs_l2073_207362

theorem cricket_initial_avg_runs (A : ℝ) (h : 11 * (A + 4) = 10 * A + 86) : A = 42 :=
sorry

end NUMINAMATH_GPT_cricket_initial_avg_runs_l2073_207362


namespace NUMINAMATH_GPT_weight_loss_clothes_percentage_l2073_207382

theorem weight_loss_clothes_percentage (W : ℝ) : 
  let initial_weight := W
  let weight_after_loss := 0.89 * initial_weight
  let final_weight_with_clothes := 0.9078 * initial_weight
  let added_weight_percentage := (final_weight_with_clothes / weight_after_loss - 1) * 100
  added_weight_percentage = 2 :=
by
  sorry

end NUMINAMATH_GPT_weight_loss_clothes_percentage_l2073_207382


namespace NUMINAMATH_GPT_parakeets_per_cage_l2073_207363

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (hcages : num_cages = 6) 
  (hparrots : parrots_per_cage = 6) 
  (htotal : total_birds = 48) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := 
  by
  sorry

end NUMINAMATH_GPT_parakeets_per_cage_l2073_207363


namespace NUMINAMATH_GPT_smallest_p_condition_l2073_207355

theorem smallest_p_condition (n p : ℕ) (hn1 : n % 2 = 1) (hn2 : n % 7 = 5) (hp : (n + p) % 10 = 0) : p = 1 := by
  sorry

end NUMINAMATH_GPT_smallest_p_condition_l2073_207355


namespace NUMINAMATH_GPT_problem_solution_l2073_207375

variable (a b : ℝ)

theorem problem_solution (h : 2 * a - 3 * b = 5) : 4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2073_207375


namespace NUMINAMATH_GPT_difference_is_2395_l2073_207379

def S : ℕ := 476
def L : ℕ := 6 * S + 15
def difference : ℕ := L - S

theorem difference_is_2395 : difference = 2395 :=
by
  sorry

end NUMINAMATH_GPT_difference_is_2395_l2073_207379


namespace NUMINAMATH_GPT_total_cupcakes_l2073_207303

-- Definitions of initial conditions
def cupcakes_initial : ℕ := 42
def cupcakes_sold : ℕ := 22
def cupcakes_made_after : ℕ := 39

-- Proof statement: Total number of cupcakes Robin would have
theorem total_cupcakes : 
  (cupcakes_initial - cupcakes_sold + cupcakes_made_after) = 59 := by
    sorry

end NUMINAMATH_GPT_total_cupcakes_l2073_207303


namespace NUMINAMATH_GPT_imaginary_part_of_conjugate_l2073_207309

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

theorem imaginary_part_of_conjugate :
  ∀ (z : ℂ), z = (1+i)^2 / (1-i) → (complex_conjugate z).im = -1 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_conjugate_l2073_207309


namespace NUMINAMATH_GPT_number_of_bowls_l2073_207351

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end NUMINAMATH_GPT_number_of_bowls_l2073_207351


namespace NUMINAMATH_GPT_shooting_guard_seconds_l2073_207332

-- Define the given conditions
def x_pg := 130
def x_sf := 85
def x_pf := 60
def x_c := 180
def avg_time_per_player := 120
def total_players := 5

-- Define the total footage
def total_footage : Nat := total_players * avg_time_per_player

-- Define the footage for four players
def footage_of_four : Nat := x_pg + x_sf + x_pf + x_c

-- Define the footage of the shooting guard, which is a variable we want to compute
def x_sg := total_footage - footage_of_four

-- The statement we want to prove
theorem shooting_guard_seconds :
  x_sg = 145 := by
  sorry

end NUMINAMATH_GPT_shooting_guard_seconds_l2073_207332


namespace NUMINAMATH_GPT_parking_lot_perimeter_l2073_207394

theorem parking_lot_perimeter (a b : ℝ) (h₁ : a^2 + b^2 = 625) (h₂ : a * b = 168) :
  2 * (a + b) = 62 :=
sorry

end NUMINAMATH_GPT_parking_lot_perimeter_l2073_207394


namespace NUMINAMATH_GPT_number_subtraction_l2073_207390

theorem number_subtraction
  (x : ℕ) (y : ℕ)
  (h1 : x = 30)
  (h2 : 8 * x - y = 102) : y = 138 :=
by 
  sorry

end NUMINAMATH_GPT_number_subtraction_l2073_207390


namespace NUMINAMATH_GPT_polynomial_sequence_symmetric_l2073_207345

def P : ℕ → ℝ → ℝ → ℝ → ℝ 
| 0, x, y, z => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_sequence_symmetric (m : ℕ) (x y z : ℝ) (σ : ℝ × ℝ × ℝ): 
  P m x y z = P m σ.1 σ.2.1 σ.2.2 :=
sorry

end NUMINAMATH_GPT_polynomial_sequence_symmetric_l2073_207345


namespace NUMINAMATH_GPT_haley_total_lives_l2073_207373

-- Define initial conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def lives_gained : ℕ := 36

-- Definition to calculate total lives
def total_lives (initial_lives lives_lost lives_gained : ℕ) : ℕ :=
  initial_lives - lives_lost + lives_gained

-- The theorem statement we want to prove
theorem haley_total_lives : total_lives initial_lives lives_lost lives_gained = 46 :=
by 
  sorry

end NUMINAMATH_GPT_haley_total_lives_l2073_207373


namespace NUMINAMATH_GPT_minimum_value_of_f_l2073_207320

noncomputable def f (x : ℝ) : ℝ := (Real.sin (Real.pi * x) - Real.cos (Real.pi * x) + 2) / Real.sqrt x

theorem minimum_value_of_f :
  ∃ x ∈ Set.Icc (1/4 : ℝ) (5/4 : ℝ), f x = (4 * Real.sqrt 5 / 5 - 2 * Real.sqrt 10 / 5) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2073_207320


namespace NUMINAMATH_GPT_line_through_two_points_l2073_207391

theorem line_through_two_points (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 4)) :
  ∃ k b : ℝ, (∀ x y : ℝ, (y = k * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ (k = 1) ∧ (b = 1) := 
by
  sorry

end NUMINAMATH_GPT_line_through_two_points_l2073_207391


namespace NUMINAMATH_GPT_total_floor_area_l2073_207365

theorem total_floor_area
    (n : ℕ) (a_cm : ℕ)
    (num_of_slabs : n = 30)
    (length_of_slab_cm : a_cm = 130) :
    (30 * ((130 * 130) / 10000)) = 50.7 :=
by
  sorry

end NUMINAMATH_GPT_total_floor_area_l2073_207365


namespace NUMINAMATH_GPT_calc_fractional_product_l2073_207310

theorem calc_fractional_product (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b :=
by
  sorry

end NUMINAMATH_GPT_calc_fractional_product_l2073_207310


namespace NUMINAMATH_GPT_total_sales_15_days_l2073_207353

def edgar_sales (n : ℕ) : ℕ := 3 * n - 1

def clara_sales (n : ℕ) : ℕ := 4 * n

def edgar_total_sales (d : ℕ) : ℕ := (d * (2 + (d * 3 - 1))) / 2

def clara_total_sales (d : ℕ) : ℕ := (d * (4 + (d * 4))) / 2

def total_sales (d : ℕ) : ℕ := edgar_total_sales d + clara_total_sales d

theorem total_sales_15_days : total_sales 15 = 810 :=
by
  sorry

end NUMINAMATH_GPT_total_sales_15_days_l2073_207353


namespace NUMINAMATH_GPT_residue_calculation_l2073_207361

theorem residue_calculation :
  (196 * 18 - 21 * 9 + 5) % 18 = 14 := 
by 
  sorry

end NUMINAMATH_GPT_residue_calculation_l2073_207361


namespace NUMINAMATH_GPT_split_payment_l2073_207370

noncomputable def Rahul_work_per_day := (1 : ℝ) / 3
noncomputable def Rajesh_work_per_day := (1 : ℝ) / 2
noncomputable def Ritesh_work_per_day := (1 : ℝ) / 4

noncomputable def total_work_per_day := Rahul_work_per_day + Rajesh_work_per_day + Ritesh_work_per_day

noncomputable def Rahul_proportion := Rahul_work_per_day / total_work_per_day
noncomputable def Rajesh_proportion := Rajesh_work_per_day / total_work_per_day
noncomputable def Ritesh_proportion := Ritesh_work_per_day / total_work_per_day

noncomputable def total_payment := 510

noncomputable def Rahul_share := Rahul_proportion * total_payment
noncomputable def Rajesh_share := Rajesh_proportion * total_payment
noncomputable def Ritesh_share := Ritesh_proportion * total_payment

theorem split_payment :
  Rahul_share + Rajesh_share + Ritesh_share = total_payment :=
by
  sorry

end NUMINAMATH_GPT_split_payment_l2073_207370


namespace NUMINAMATH_GPT_geometric_sum_l2073_207399

open BigOperators

noncomputable def geom_sequence (a q : ℚ) (n : ℕ) : ℚ := a * q ^ n

noncomputable def sum_geom_sequence (a q : ℚ) (n : ℕ) : ℚ := 
  if q = 1 then a * n
  else a * (1 - q ^ (n + 1)) / (1 - q)

theorem geometric_sum (a q : ℚ) (h_a : a = 1) (h_S3 : sum_geom_sequence a q 2 = 3 / 4) :
  sum_geom_sequence a q 3 = 5 / 8 :=
sorry

end NUMINAMATH_GPT_geometric_sum_l2073_207399


namespace NUMINAMATH_GPT_find_r_l2073_207393

theorem find_r (k r : ℝ) (h1 : (5 = k * 3^r)) (h2 : (45 = k * 9^r)) : r = 2 :=
  sorry

end NUMINAMATH_GPT_find_r_l2073_207393


namespace NUMINAMATH_GPT_probability_of_B_winning_is_correct_l2073_207306

noncomputable def prob_A_wins : ℝ := 0.2
noncomputable def prob_draw : ℝ := 0.5
noncomputable def prob_B_wins : ℝ := 1 - (prob_A_wins + prob_draw)

theorem probability_of_B_winning_is_correct : prob_B_wins = 0.3 := by
  sorry

end NUMINAMATH_GPT_probability_of_B_winning_is_correct_l2073_207306


namespace NUMINAMATH_GPT_carla_zoo_l2073_207367

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end NUMINAMATH_GPT_carla_zoo_l2073_207367


namespace NUMINAMATH_GPT_units_digit_lucas_L10_is_4_l2073_207305

def lucas : ℕ → ℕ 
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_lucas_L10_is_4 : units_digit (lucas (lucas 10)) = 4 := 
  sorry

end NUMINAMATH_GPT_units_digit_lucas_L10_is_4_l2073_207305


namespace NUMINAMATH_GPT_powderman_distance_approximates_275_yards_l2073_207302

noncomputable def distance_run (t : ℝ) : ℝ := 6 * t
noncomputable def sound_distance (t : ℝ) : ℝ := 1080 * (t - 45) / 3

theorem powderman_distance_approximates_275_yards : 
  ∃ t : ℝ, t > 45 ∧ 
  (distance_run t = sound_distance t) → 
  abs (distance_run t - 275) < 1 :=
by
  sorry

end NUMINAMATH_GPT_powderman_distance_approximates_275_yards_l2073_207302


namespace NUMINAMATH_GPT_clarence_oranges_after_giving_l2073_207321

def initial_oranges : ℝ := 5.0
def oranges_given : ℝ := 3.0

theorem clarence_oranges_after_giving : (initial_oranges - oranges_given) = 2.0 :=
by
  sorry

end NUMINAMATH_GPT_clarence_oranges_after_giving_l2073_207321


namespace NUMINAMATH_GPT_statement_B_l2073_207358

variable (Student : Type)
variable (nora : Student)
variable (correctly_answered_all_math_questions : Student → Prop)
variable (received_at_least_B : Student → Prop)

theorem statement_B :
  (∀ s : Student, correctly_answered_all_math_questions s → received_at_least_B s) →
  (¬ received_at_least_B nora → ∃ q : Student, ¬ correctly_answered_all_math_questions q) :=
by
  intros h hn
  sorry

end NUMINAMATH_GPT_statement_B_l2073_207358


namespace NUMINAMATH_GPT_remainder_of_cake_l2073_207368

theorem remainder_of_cake (John Emily : ℝ) (h1 : 0.60 ≤ John) (h2 : Emily = 0.50 * (1 - John)) :
  1 - John - Emily = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_cake_l2073_207368
