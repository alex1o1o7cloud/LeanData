import Mathlib

namespace NUMINAMATH_GPT_value_of_a_minus_b_l302_30282

theorem value_of_a_minus_b (a b : ℤ) 
  (h₁ : |a| = 7) 
  (h₂ : |b| = 5) 
  (h₃ : a < b) : 
  a - b = -12 ∨ a - b = -2 := 
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l302_30282


namespace NUMINAMATH_GPT_increasing_on_iff_decreasing_on_periodic_even_l302_30207

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f x = f (x + p)
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

theorem increasing_on_iff_decreasing_on_periodic_even :
  (is_even f ∧ is_periodic f 2 ∧ is_increasing_on f 0 1) ↔ is_decreasing_on f 3 4 := 
by
  sorry

end NUMINAMATH_GPT_increasing_on_iff_decreasing_on_periodic_even_l302_30207


namespace NUMINAMATH_GPT_parallel_lines_chords_distance_l302_30206

theorem parallel_lines_chords_distance
  (r d : ℝ)
  (h1 : ∀ (P Q : ℝ), P = Q + d / 2 → Q = P - d / 2)
  (h2 : ∀ (A B : ℝ), A = B + 3 * d / 2 → B = A - 3 * d / 2)
  (chords : ∀ (l1 l2 l3 l4 : ℝ), (l1 = 40 ∧ l2 = 40 ∧ l3 = 36 ∧ l4 = 36)) :
  d = 1.46 :=
sorry

end NUMINAMATH_GPT_parallel_lines_chords_distance_l302_30206


namespace NUMINAMATH_GPT_probability_even_sum_of_selected_envelopes_l302_30247

theorem probability_even_sum_of_selected_envelopes :
  let face_values := [5, 6, 8, 10]
  let possible_sum_is_even (s : ℕ) : Prop := s % 2 = 0
  let num_combinations := Nat.choose 4 2
  let favorable_combinations := 3
  (favorable_combinations / num_combinations : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_sum_of_selected_envelopes_l302_30247


namespace NUMINAMATH_GPT_min_games_to_achieve_98_percent_l302_30213

-- Define initial conditions
def initial_games : ℕ := 5
def initial_sharks_wins : ℕ := 2
def initial_tigers_wins : ℕ := 3

-- Define the total number of games and the total number of wins by the Sharks after additional games
def total_games (N : ℕ) : ℕ := initial_games + N
def total_sharks_wins (N : ℕ) : ℕ := initial_sharks_wins + N

-- Define the Sharks' winning percentage
def sharks_winning_percentage (N : ℕ) : ℚ := total_sharks_wins N / total_games N

-- Define the minimum number of additional games needed
def minimum_N : ℕ := 145

-- Theorem: Prove that the Sharks' winning percentage is at least 98% when N = 145
theorem min_games_to_achieve_98_percent :
  sharks_winning_percentage minimum_N ≥ 49 / 50 :=
sorry

end NUMINAMATH_GPT_min_games_to_achieve_98_percent_l302_30213


namespace NUMINAMATH_GPT_simplify_fraction_l302_30276

variables {a b c x y z : ℝ}

theorem simplify_fraction :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz) =
  a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l302_30276


namespace NUMINAMATH_GPT_parallel_line_distance_l302_30211

-- Definition of a line
structure Line where
  m : ℚ -- slope
  c : ℚ -- y-intercept

-- Given conditions
def given_line : Line :=
  { m := 3 / 4, c := 6 }

-- Prove that there exist lines parallel to the given line and 5 units away from it
theorem parallel_line_distance (L : Line)
  (h_parallel : L.m = given_line.m)
  (h_distance : abs (L.c - given_line.c) = 25 / 4) :
  (L.c = 12.25) ∨ (L.c = -0.25) :=
sorry

end NUMINAMATH_GPT_parallel_line_distance_l302_30211


namespace NUMINAMATH_GPT_nickels_used_for_notebook_l302_30274

def notebook_cost_dollars : ℚ := 1.30
def dollar_to_cents_conversion : ℤ := 100
def nickel_value_cents : ℤ := 5

theorem nickels_used_for_notebook : 
  (notebook_cost_dollars * dollar_to_cents_conversion) / nickel_value_cents = 26 := 
by 
  sorry

end NUMINAMATH_GPT_nickels_used_for_notebook_l302_30274


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l302_30299

variables {a : ℕ → ℕ} (d a1 : ℕ)

def arithmetic_sequence (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : arithmetic_sequence 1 + arithmetic_sequence 3 + arithmetic_sequence 9 = 20) :
  4 * arithmetic_sequence 5 - arithmetic_sequence 7 = 20 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l302_30299


namespace NUMINAMATH_GPT_grains_of_rice_in_teaspoon_is_10_l302_30231

noncomputable def grains_of_rice_per_teaspoon : ℕ :=
  let grains_per_cup := 480
  let tablespoons_per_half_cup := 8
  let teaspoons_per_tablespoon := 3
  grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)

theorem grains_of_rice_in_teaspoon_is_10 : grains_of_rice_per_teaspoon = 10 :=
by
  sorry

end NUMINAMATH_GPT_grains_of_rice_in_teaspoon_is_10_l302_30231


namespace NUMINAMATH_GPT_sin_diff_identity_l302_30202

variable (α β : ℝ)

def condition1 := (Real.sin α - Real.cos β = 3 / 4)
def condition2 := (Real.cos α + Real.sin β = -2 / 5)

theorem sin_diff_identity : 
  condition1 α β → 
  condition2 α β → 
  Real.sin (α - β) = 511 / 800 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sin_diff_identity_l302_30202


namespace NUMINAMATH_GPT_coprime_composite_lcm_l302_30241

theorem coprime_composite_lcm (a b : ℕ) (ha : a > 1) (hb : b > 1) (hcoprime : Nat.gcd a b = 1) (hlcm : Nat.lcm a b = 120) : 
  Nat.gcd a b = 1 ∧ min a b = 8 := 
by 
  sorry

end NUMINAMATH_GPT_coprime_composite_lcm_l302_30241


namespace NUMINAMATH_GPT_number_of_cds_on_shelf_l302_30230

-- Definitions and hypotheses
def cds_per_rack : ℕ := 8
def racks_per_shelf : ℕ := 4

-- Theorem statement
theorem number_of_cds_on_shelf :
  cds_per_rack * racks_per_shelf = 32 :=
by sorry

end NUMINAMATH_GPT_number_of_cds_on_shelf_l302_30230


namespace NUMINAMATH_GPT_number_of_ways_to_choose_marbles_l302_30237

theorem number_of_ways_to_choose_marbles 
  (total_marbles : ℕ) 
  (red_count green_count blue_count : ℕ) 
  (total_choice chosen_rgb_count remaining_choice : ℕ) 
  (h_total_marbles : total_marbles = 15) 
  (h_red_count : red_count = 2) 
  (h_green_count : green_count = 2) 
  (h_blue_count : blue_count = 2) 
  (h_total_choice : total_choice = 5) 
  (h_chosen_rgb_count : chosen_rgb_count = 2) 
  (h_remaining_choice : remaining_choice = 3) :
  ∃ (num_ways : ℕ), num_ways = 3300 :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_marbles_l302_30237


namespace NUMINAMATH_GPT_ratio_JL_JM_l302_30283

theorem ratio_JL_JM (s w h : ℝ) (shared_area_25 : 0.25 * s^2 = 0.4 * w * h) (jm_eq_s : h = s) :
  w / h = 5 / 8 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_ratio_JL_JM_l302_30283


namespace NUMINAMATH_GPT_find_point_D_l302_30273

structure Point :=
  (x : ℤ)
  (y : ℤ)

def translation_rule (A C : Point) : Point :=
{
  x := C.x - A.x,
  y := C.y - A.y
}

def translate (P delta : Point) : Point :=
{
  x := P.x + delta.x,
  y := P.y + delta.y
}

def A := Point.mk (-1) 4
def C := Point.mk 1 2
def B := Point.mk 2 1
def D := Point.mk 4 (-1)
def translation_delta : Point := translation_rule A C

theorem find_point_D : translate B translation_delta = D :=
by
  sorry

end NUMINAMATH_GPT_find_point_D_l302_30273


namespace NUMINAMATH_GPT_remainder_when_dividing_polynomial_by_x_minus_3_l302_30286

noncomputable def P (x : ℤ) : ℤ := 
  2 * x^8 - 3 * x^7 + 4 * x^6 - x^4 + 6 * x^3 - 5 * x^2 + 18 * x - 20

theorem remainder_when_dividing_polynomial_by_x_minus_3 :
  P 3 = 17547 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_polynomial_by_x_minus_3_l302_30286


namespace NUMINAMATH_GPT_discriminant_of_quadratic_equation_l302_30260

noncomputable def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_equation : discriminant 5 (-11) (-18) = 481 := by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_equation_l302_30260


namespace NUMINAMATH_GPT_inequalities_of_function_nonneg_l302_30221

theorem inequalities_of_function_nonneg (a b A B : ℝ)
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.sin (2 * θ) - B * Real.cos (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := sorry

end NUMINAMATH_GPT_inequalities_of_function_nonneg_l302_30221


namespace NUMINAMATH_GPT_cos_double_angle_l302_30209

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 2/3) : Real.cos (2 * θ) = -1/9 := 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l302_30209


namespace NUMINAMATH_GPT_exists_infinite_repeated_sum_of_digits_l302_30223

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence a_n which is the sum of digits of P(n)
def a (P : ℕ → ℤ) (n : ℕ) : ℕ :=
  sum_of_digits (P n).natAbs

theorem exists_infinite_repeated_sum_of_digits (P : ℕ → ℤ) (h_nat_coeffs : ∀ n, (P n) ≥ 0) :
  ∃ s : ℕ, ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a P n = s :=
sorry

end NUMINAMATH_GPT_exists_infinite_repeated_sum_of_digits_l302_30223


namespace NUMINAMATH_GPT_botanical_garden_path_length_l302_30281

theorem botanical_garden_path_length
  (scale : ℝ)
  (path_length_map : ℝ)
  (path_length_real : ℝ)
  (h_scale : scale = 500)
  (h_path_length_map : path_length_map = 6.5)
  (h_path_length_real : path_length_real = path_length_map * scale) :
  path_length_real = 3250 :=
by
  sorry

end NUMINAMATH_GPT_botanical_garden_path_length_l302_30281


namespace NUMINAMATH_GPT_probability_star_top_card_is_one_fifth_l302_30238

-- Define the total number of cards in the deck
def total_cards : ℕ := 65

-- Define the number of star cards in the deck
def star_cards : ℕ := 13

-- Define the probability calculation
def probability_star_top_card : ℚ := star_cards / total_cards

-- State the theorem regarding the probability
theorem probability_star_top_card_is_one_fifth :
  probability_star_top_card = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_star_top_card_is_one_fifth_l302_30238


namespace NUMINAMATH_GPT_rectangle_perimeter_given_square_l302_30232

-- Defining the problem conditions
def square_side_length (p : ℕ) : ℕ := p / 4

def rectangle_perimeter (s : ℕ) : ℕ := 2 * (s + (s / 2))

-- Stating the theorem: Given the perimeter of the square is 80, prove the perimeter of one of the rectangles is 60
theorem rectangle_perimeter_given_square (p : ℕ) (h : p = 80) : rectangle_perimeter (square_side_length p) = 60 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_given_square_l302_30232


namespace NUMINAMATH_GPT_seq_property_l302_30234

theorem seq_property (m : ℤ) (h1 : |m| ≥ 2)
  (a : ℕ → ℤ)
  (h2 : ¬ (a 1 = 0 ∧ a 2 = 0))
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) = a (n + 1) - m * a n)
  (r s : ℕ)
  (h4 : r > s ∧ s ≥ 2)
  (h5 : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| :=
by
  sorry

end NUMINAMATH_GPT_seq_property_l302_30234


namespace NUMINAMATH_GPT_cost_per_serving_in_cents_after_coupon_l302_30291

def oz_per_serving : ℝ := 1
def price_per_bag : ℝ := 25
def bag_weight : ℝ := 40
def coupon : ℝ := 5
def dollars_to_cents (d : ℝ) : ℝ := d * 100

theorem cost_per_serving_in_cents_after_coupon : 
  dollars_to_cents ((price_per_bag - coupon) / bag_weight) = 50 := by
  sorry

end NUMINAMATH_GPT_cost_per_serving_in_cents_after_coupon_l302_30291


namespace NUMINAMATH_GPT_butterfly_flutters_total_distance_l302_30257

-- Define the conditions
def start_pos : ℤ := 0
def first_move : ℤ := 4
def second_move : ℤ := -3
def third_move : ℤ := 7

-- Define a function that calculates the total distance
def total_distance (xs : List ℤ) : ℤ :=
  List.sum (List.map (fun ⟨x, y⟩ => abs (y - x)) (xs.zip xs.tail))

-- Create the butterfly's path
def path : List ℤ := [start_pos, first_move, second_move, third_move]

-- Define the proposition that we need to prove
theorem butterfly_flutters_total_distance : total_distance path = 21 := sorry

end NUMINAMATH_GPT_butterfly_flutters_total_distance_l302_30257


namespace NUMINAMATH_GPT_dealer_gross_profit_l302_30256

theorem dealer_gross_profit (P S G : ℝ) (hP : P = 150) (markup : S = P + 0.5 * S) :
  G = S - P → G = 150 :=
by
  sorry

end NUMINAMATH_GPT_dealer_gross_profit_l302_30256


namespace NUMINAMATH_GPT_middle_angle_range_l302_30294

theorem middle_angle_range (α β γ : ℝ) (h₀: α + β + γ = 180) (h₁: 0 < α) (h₂: 0 < β) (h₃: 0 < γ) (h₄: α ≤ β) (h₅: β ≤ γ) : 
  0 < β ∧ β < 90 :=
by
  sorry

end NUMINAMATH_GPT_middle_angle_range_l302_30294


namespace NUMINAMATH_GPT_tan_C_value_b_value_l302_30289

-- Define variables and conditions
variable (A B C a b c : ℝ)
variable (A_eq : A = Real.pi / 4)
variable (cond : b^2 - a^2 = 1 / 4 * c^2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 5 / 2)

-- First part: Prove tan(C) = 4 given the conditions
theorem tan_C_value : A = Real.pi / 4 ∧ b^2 - a^2 = 1 / 4 * c^2 → Real.tan C = 4 := by
  intro h
  sorry

-- Second part: Prove b = 5 / 2 given the area condition
theorem b_value : (1 / 2 * b * c * Real.sin (Real.pi / 4) = 5 / 2) → b = 5 / 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_tan_C_value_b_value_l302_30289


namespace NUMINAMATH_GPT_smaller_acute_angle_is_20_degrees_l302_30215

noncomputable def smaller_acute_angle (x : ℝ) : Prop :=
  let θ1 := 7 * x
  let θ2 := 2 * x
  θ1 + θ2 = 90 ∧ θ2 = 20

theorem smaller_acute_angle_is_20_degrees : ∃ x : ℝ, smaller_acute_angle x :=
  sorry

end NUMINAMATH_GPT_smaller_acute_angle_is_20_degrees_l302_30215


namespace NUMINAMATH_GPT_number_of_truthful_dwarfs_l302_30248

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end NUMINAMATH_GPT_number_of_truthful_dwarfs_l302_30248


namespace NUMINAMATH_GPT_find_T5_l302_30272

variables (a b x y : ℝ)

def T (n : ℕ) : ℝ := a * x^n + b * y^n

theorem find_T5
  (h1 : T a b x y 1 = 3)
  (h2 : T a b x y 2 = 7)
  (h3 : T a b x y 3 = 6)
  (h4 : T a b x y 4 = 42) :
  T a b x y 5 = -360 :=
sorry

end NUMINAMATH_GPT_find_T5_l302_30272


namespace NUMINAMATH_GPT_k_squared_geq_25_div_3_l302_30270

open Real

theorem k_squared_geq_25_div_3 
  (a₁ a₂ a₃ a₄ a₅ k : ℝ)
  (h₁₂ : abs (a₁ - a₂) ≥ 1) (h₁₃ : abs (a₁ - a₃) ≥ 1) (h₁₄ : abs (a₁ - a₄) ≥ 1) (h₁₅ : abs (a₁ - a₅) ≥ 1)
  (h₂₃ : abs (a₂ - a₃) ≥ 1) (h₂₄ : abs (a₂ - a₄) ≥ 1) (h₂₅ : abs (a₂ - a₅) ≥ 1)
  (h₃₄ : abs (a₃ - a₄) ≥ 1) (h₃₅ : abs (a₃ - a₅) ≥ 1)
  (h₄₅ : abs (a₄ - a₅) ≥ 1)
  (eq1 : a₁ + a₂ + a₃ + a₄ + a₅ = 2 * k)
  (eq2 : a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = 2 * k^2) :
  k^2 ≥ 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_k_squared_geq_25_div_3_l302_30270


namespace NUMINAMATH_GPT_probability_of_johns_8th_roll_l302_30280

noncomputable def probability_johns_8th_roll_is_last : ℚ :=
  (7/8)^6 * (1/8)

theorem probability_of_johns_8th_roll :
  probability_johns_8th_roll_is_last = 117649 / 2097152 := by
  sorry

end NUMINAMATH_GPT_probability_of_johns_8th_roll_l302_30280


namespace NUMINAMATH_GPT_bike_price_l302_30285

-- Definitions of the conditions
def maria_savings : ℕ := 120
def mother_offer : ℕ := 250
def amount_needed : ℕ := 230

-- Theorem statement
theorem bike_price (maria_savings mother_offer amount_needed : ℕ) : 
  maria_savings + mother_offer + amount_needed = 600 := 
by
  -- Sorry is used here to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_bike_price_l302_30285


namespace NUMINAMATH_GPT_transformed_parabola_l302_30292

theorem transformed_parabola (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 3) → (y = 2 * (x + 1)^2 + 2) :=
by
  sorry

end NUMINAMATH_GPT_transformed_parabola_l302_30292


namespace NUMINAMATH_GPT_yuna_has_biggest_number_l302_30279

-- Define the numbers assigned to each student
def Yoongi_num : ℕ := 7
def Jungkook_num : ℕ := 6
def Yuna_num : ℕ := 9
def Yoojung_num : ℕ := 8

-- State the main theorem that Yuna has the biggest number
theorem yuna_has_biggest_number : 
  (Yuna_num = 9) ∧ (Yuna_num > Yoongi_num) ∧ (Yuna_num > Jungkook_num) ∧ (Yuna_num > Yoojung_num) :=
sorry

end NUMINAMATH_GPT_yuna_has_biggest_number_l302_30279


namespace NUMINAMATH_GPT_min_value_reciprocals_l302_30201

theorem min_value_reciprocals (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h_sum : x + y = 8) (h_prod : x * y = 12) : 
  (1/x + 1/y) = 2/3 :=
sorry

end NUMINAMATH_GPT_min_value_reciprocals_l302_30201


namespace NUMINAMATH_GPT_airplane_altitude_l302_30244

theorem airplane_altitude (d_Alice_Bob : ℝ) (angle_Alice : ℝ) (angle_Bob : ℝ) (altitude : ℝ) : 
  d_Alice_Bob = 8 ∧ angle_Alice = 45 ∧ angle_Bob = 30 → altitude = 16 / 3 :=
by
  intros h
  rcases h with ⟨h1, ⟨h2, h3⟩⟩
  -- you may insert the proof here if needed
  sorry

end NUMINAMATH_GPT_airplane_altitude_l302_30244


namespace NUMINAMATH_GPT_sin_721_eq_sin_1_l302_30288

theorem sin_721_eq_sin_1 : Real.sin (721 * Real.pi / 180) = Real.sin (1 * Real.pi / 180) := 
by
  sorry

end NUMINAMATH_GPT_sin_721_eq_sin_1_l302_30288


namespace NUMINAMATH_GPT_triangle_inequality_l302_30236

theorem triangle_inequality (S R r : ℝ) (h : S^2 = 2 * R^2 + 8 * R * r + 3 * r^2) : 
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_inequality_l302_30236


namespace NUMINAMATH_GPT_total_legs_in_farm_l302_30227

def num_animals : Nat := 13
def num_chickens : Nat := 4
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

theorem total_legs_in_farm : 
  (num_chickens * legs_per_chicken) + ((num_animals - num_chickens) * legs_per_buffalo) = 44 :=
by
  sorry

end NUMINAMATH_GPT_total_legs_in_farm_l302_30227


namespace NUMINAMATH_GPT_square_of_cube_plus_11_l302_30210

def third_smallest_prime : ℕ := 5

theorem square_of_cube_plus_11 : (third_smallest_prime ^ 3)^2 + 11 = 15636 := by
  -- We will provide a proof later
  sorry

end NUMINAMATH_GPT_square_of_cube_plus_11_l302_30210


namespace NUMINAMATH_GPT_function_increasing_interval_l302_30267

theorem function_increasing_interval :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi),
  (2 * Real.sin ((Real.pi / 6) - 2 * x) : ℝ)
  ≤ 2 * Real.sin ((Real.pi / 6) - 2 * x + 1)) ↔ (x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6)) :=
sorry

end NUMINAMATH_GPT_function_increasing_interval_l302_30267


namespace NUMINAMATH_GPT_betsy_sewing_l302_30235

-- Definitions of conditions
def total_squares : ℕ := 16 + 16
def sewn_percentage : ℝ := 0.25
def sewn_squares : ℝ := sewn_percentage * total_squares
def squares_left : ℝ := total_squares - sewn_squares

-- Proof that Betsy needs to sew 24 more squares
theorem betsy_sewing : squares_left = 24 := by
  -- Sorry placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_betsy_sewing_l302_30235


namespace NUMINAMATH_GPT_will_money_left_l302_30296

def initial_money : ℝ := 74
def sweater_cost : ℝ := 9
def tshirt_cost : ℝ := 11
def shoes_cost : ℝ := 30
def hat_cost : ℝ := 5
def socks_cost : ℝ := 4
def refund_percentage : ℝ := 0.85
def discount_percentage : ℝ := 0.1
def tax_percentage : ℝ := 0.05

-- Total cost before returns and discounts
def total_cost_before : ℝ := 
  sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost

-- Refund for shoes
def shoes_refund : ℝ := refund_percentage * shoes_cost

-- New total cost after refund
def total_cost_after_refund : ℝ := total_cost_before - shoes_refund

-- Total cost of remaining items (excluding shoes)
def remaining_items_cost : ℝ := total_cost_before - shoes_cost

-- Discount on remaining items
def discount : ℝ := discount_percentage * remaining_items_cost

-- New total cost after discount
def total_cost_after_discount : ℝ := total_cost_after_refund - discount

-- Sales tax on the final purchase amount
def sales_tax : ℝ := tax_percentage * total_cost_after_discount

-- Final purchase amount with tax
def final_purchase_amount : ℝ := total_cost_after_discount + sales_tax

-- Money left after the final purchase
def money_left : ℝ := initial_money - final_purchase_amount

theorem will_money_left : money_left = 41.87 := by 
  sorry

end NUMINAMATH_GPT_will_money_left_l302_30296


namespace NUMINAMATH_GPT_sin2alpha_div_1_plus_cos2alpha_eq_3_l302_30250

theorem sin2alpha_div_1_plus_cos2alpha_eq_3 (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := 
  sorry

end NUMINAMATH_GPT_sin2alpha_div_1_plus_cos2alpha_eq_3_l302_30250


namespace NUMINAMATH_GPT_other_root_of_equation_l302_30220

theorem other_root_of_equation (m : ℤ) (h₁ : (2 : ℤ) ∈ {x : ℤ | x ^ 2 - 3 * x - m = 0}) : 
  ∃ x, x ≠ 2 ∧ (x ^ 2 - 3 * x - m = 0) ∧ x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_other_root_of_equation_l302_30220


namespace NUMINAMATH_GPT_animal_counts_l302_30212

-- Definitions based on given conditions
def ReptileHouse (R : ℕ) : ℕ := 3 * R - 5
def Aquarium (ReptileHouse : ℕ) : ℕ := 2 * ReptileHouse
def Aviary (Aquarium RainForest : ℕ) : ℕ := (Aquarium - RainForest) + 3

-- The main theorem statement
theorem animal_counts
  (R : ℕ)
  (ReptileHouse_eq : ReptileHouse R = 16)
  (A : ℕ := Aquarium 16)
  (V : ℕ := Aviary A R) :
  (R = 7) ∧ (A = 32) ∧ (V = 28) :=
by
  sorry

end NUMINAMATH_GPT_animal_counts_l302_30212


namespace NUMINAMATH_GPT_solution_1_solution_2_l302_30271

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem solution_1 :
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + Real.pi / 3)) :=
by sorry

theorem solution_2 (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (Real.pi / 2) Real.pi) :
  f (x0 / 2) = -3 / 8 → 
  Real.cos (x0 + Real.pi / 6) = - Real.sqrt 741 / 32 - 3 / 32 :=
by sorry

end NUMINAMATH_GPT_solution_1_solution_2_l302_30271


namespace NUMINAMATH_GPT_find_t_l302_30295

-- Given conditions 
variables (p j t : ℝ)

-- Condition 1: j is 25% less than p
def condition1 : Prop := j = 0.75 * p

-- Condition 2: j is 20% less than t
def condition2 : Prop := j = 0.80 * t

-- Condition 3: t is t% less than p
def condition3 : Prop := t = p * (1 - t / 100)

-- Final proof statement
theorem find_t (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t) : t = 6.25 :=
sorry

end NUMINAMATH_GPT_find_t_l302_30295


namespace NUMINAMATH_GPT_proof_problem_l302_30261

axiom is_line (m : Type) : Prop
axiom is_plane (α : Type) : Prop
axiom is_subset_of_plane (m : Type) (β : Type) : Prop
axiom is_perpendicular (a : Type) (b : Type) : Prop
axiom is_parallel (a : Type) (b : Type) : Prop

theorem proof_problem
  (m n : Type) 
  (α β : Type)
  (h1 : is_line m)
  (h2 : is_line n)
  (h3 : is_plane α)
  (h4 : is_plane β)
  (h_prop2 : is_parallel α β → is_subset_of_plane m α → is_parallel m β)
  (h_prop3 : is_perpendicular n α → is_perpendicular n β → is_perpendicular m α → is_perpendicular m β)
  : (is_subset_of_plane m β → is_perpendicular α β → ¬ (is_perpendicular m α)) ∧ 
    (is_parallel m α → is_parallel m β → ¬ (is_parallel α β)) :=
sorry

end NUMINAMATH_GPT_proof_problem_l302_30261


namespace NUMINAMATH_GPT_pencils_evenly_distributed_l302_30204

-- Define the initial number of pencils Eric had
def initialPencils : Nat := 150

-- Define the additional pencils brought by another teacher
def additionalPencils : Nat := 30

-- Define the total number of containers
def numberOfContainers : Nat := 5

-- Define the total number of pencils after receiving additional pencils
def totalPencils := initialPencils + additionalPencils

-- Define the number of pencils per container after even distribution
def pencilsPerContainer := totalPencils / numberOfContainers

-- Statement of the proof problem
theorem pencils_evenly_distributed :
  pencilsPerContainer = 36 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_pencils_evenly_distributed_l302_30204


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l302_30245

theorem geometric_sequence_ratio
  (a1 r : ℝ) (h_r : r ≠ 1)
  (h : (1 - r^6) / (1 - r^3) = 1 / 2) :
  (1 - r^9) / (1 - r^3) = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l302_30245


namespace NUMINAMATH_GPT_minimum_trucks_needed_l302_30275

theorem minimum_trucks_needed 
  (total_weight : ℕ) (box_weight: ℕ) (truck_capacity: ℕ) (min_trucks: ℕ)
  (h_total_weight : total_weight = 10)
  (h_box_weight_le : ∀ (w : ℕ), w <= box_weight → w <= 1)
  (h_truck_capacity : truck_capacity = 3)
  (h_min_trucks : min_trucks = 5) : 
  min_trucks >= (total_weight / truck_capacity) :=
sorry

end NUMINAMATH_GPT_minimum_trucks_needed_l302_30275


namespace NUMINAMATH_GPT_sum_of_coeffs_l302_30277

theorem sum_of_coeffs (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 2 * (0 : ℤ))^5 = a0)
  (h2 : (1 - 2 * (1 : ℤ))^5 = a0 + a1 + a2 + a3 + a4 + a5) :
  a1 + a2 + a3 + a4 + a5 = -2 := by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_l302_30277


namespace NUMINAMATH_GPT_tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l302_30203

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

theorem tangent_line_eq_at_1 : 
  ∃ c : ℝ, ∀ x y : ℝ, y = f x → (x = 1 → y = 0) → y = 2 * (x - 1) → 2 * x - y - 2 = 0 := 
by sorry

theorem max_value_on_interval :
  ∃ xₘ : ℝ, (0 ≤ xₘ ∧ xₘ ≤ 2) ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 2) → f x ≤ 6 :=
by sorry

theorem unique_solution_exists :
  ∃! x₀ : ℝ, f x₀ = g x₀ :=
by sorry

end NUMINAMATH_GPT_tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l302_30203


namespace NUMINAMATH_GPT_distinct_parallel_lines_l302_30266

theorem distinct_parallel_lines (k : ℝ) :
  (∃ (L1 L2 : ℝ × ℝ → Prop), 
    (∀ x y, L1 (x, y) ↔ x - 2 * y - 3 = 0) ∧ 
    (∀ x y, L2 (x, y) ↔ 18 * x - k^2 * y - 9 * k = 0)) → 
  (∃ slope1 slope2, 
    slope1 = 1/2 ∧ 
    slope2 = 18 / k^2 ∧
    (slope1 = slope2) ∧
    (¬ (∀ x y, x - 2 * y - 3 = 18 * x - k^2 * y - 9 * k))) → 
  k = -6 :=
by 
  sorry

end NUMINAMATH_GPT_distinct_parallel_lines_l302_30266


namespace NUMINAMATH_GPT_claire_shirts_proof_l302_30217

theorem claire_shirts_proof : 
  ∀ (brian_shirts andrew_shirts steven_shirts claire_shirts : ℕ),
    brian_shirts = 3 →
    andrew_shirts = 6 * brian_shirts →
    steven_shirts = 4 * andrew_shirts →
    claire_shirts = 5 * steven_shirts →
    claire_shirts = 360 := 
by
  intro brian_shirts andrew_shirts steven_shirts claire_shirts
  intros h_brian h_andrew h_steven h_claire
  sorry

end NUMINAMATH_GPT_claire_shirts_proof_l302_30217


namespace NUMINAMATH_GPT_diana_total_cost_l302_30255

noncomputable def shopping_total_cost := 
  let t_shirt_price := 10
  let sweater_price := 25
  let jacket_price := 100
  let jeans_price := 40
  let shoes_price := 70 

  let t_shirt_discount := 0.20
  let sweater_discount := 0.10
  let jacket_discount := 0.15
  let jeans_discount := 0.05
  let shoes_discount := 0.25

  let clothes_tax := 0.06
  let shoes_tax := 0.09

  let t_shirt_qty := 8
  let sweater_qty := 5
  let jacket_qty := 3
  let jeans_qty := 6
  let shoes_qty := 4

  let t_shirt_total := t_shirt_qty * t_shirt_price 
  let sweater_total := sweater_qty * sweater_price 
  let jacket_total := jacket_qty * jacket_price 
  let jeans_total := jeans_qty * jeans_price 
  let shoes_total := shoes_qty * shoes_price 

  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let sweater_discounted := sweater_total * (1 - sweater_discount)
  let jacket_discounted := jacket_total * (1 - jacket_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let shoes_discounted := shoes_total * (1 - shoes_discount)

  let t_shirt_final := t_shirt_discounted * (1 + clothes_tax)
  let sweater_final := sweater_discounted * (1 + clothes_tax)
  let jacket_final := jacket_discounted * (1 + clothes_tax)
  let jeans_final := jeans_discounted * (1 + clothes_tax)
  let shoes_final := shoes_discounted * (1 + shoes_tax)

  t_shirt_final + sweater_final + jacket_final + jeans_final + shoes_final

theorem diana_total_cost : shopping_total_cost = 927.97 :=
by sorry

end NUMINAMATH_GPT_diana_total_cost_l302_30255


namespace NUMINAMATH_GPT_inverse_true_l302_30293

theorem inverse_true : 
  (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop), 
    (∀ a b, supplementary a b → a = b) ∧ (∀ l1 l2, parallel l1 l2)) ↔ 
    (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop),
    (∀ l1 l2, parallel l1 l2) ∧ (∀ a b, supplementary a b → a = b)) :=
sorry

end NUMINAMATH_GPT_inverse_true_l302_30293


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l302_30249

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ a₃₀ : ℕ) (d : ℕ) (n : ℕ), a₁ = 3 → a₃₀ = 89 → n = 10 → 
  (a₃₀ - a₁) / 29 = d → a₁ + (n - 1) * d = 30 :=
by
  intros a₁ a₃₀ d n h₁ h₃₀ hn hd
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l302_30249


namespace NUMINAMATH_GPT_ann_hill_length_l302_30200

/-- Given the conditions:
1. Mary slides down a hill that is 630 feet long at a speed of 90 feet/minute.
2. Ann slides down a hill at a rate of 40 feet/minute.
3. Ann's trip takes 13 minutes longer than Mary's.
Prove that the length of the hill Ann slides down is 800 feet. -/
theorem ann_hill_length
    (distance_Mary : ℕ) (speed_Mary : ℕ) 
    (speed_Ann : ℕ) (time_diff : ℕ)
    (h1 : distance_Mary = 630)
    (h2 : speed_Mary = 90)
    (h3 : speed_Ann = 40)
    (h4 : time_diff = 13) :
    speed_Ann * ((distance_Mary / speed_Mary) + time_diff) = 800 := 
by
    sorry

end NUMINAMATH_GPT_ann_hill_length_l302_30200


namespace NUMINAMATH_GPT_dima_is_mistaken_l302_30246

theorem dima_is_mistaken :
  (∃ n : Nat, n > 0 ∧ ∀ n, 3 * n = 4 * n) → False :=
by
  intros h
  obtain ⟨n, hn1, hn2⟩ := h
  have hn := (hn2 n)
  linarith

end NUMINAMATH_GPT_dima_is_mistaken_l302_30246


namespace NUMINAMATH_GPT_sum_of_nonneg_reals_l302_30262

theorem sum_of_nonneg_reals (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_nonneg_reals_l302_30262


namespace NUMINAMATH_GPT_kim_average_round_correct_answers_l302_30254

theorem kim_average_round_correct_answers (x : ℕ) :
  (6 * 2) + (x * 3) + (4 * 5) = 38 → x = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_kim_average_round_correct_answers_l302_30254


namespace NUMINAMATH_GPT_weight_of_new_student_l302_30216

theorem weight_of_new_student (avg_decrease_per_student : ℝ) (num_students : ℕ) (weight_replaced_student : ℝ) (total_reduction : ℝ) 
    (h1 : avg_decrease_per_student = 5) (h2 : num_students = 8) (h3 : weight_replaced_student = 86) (h4 : total_reduction = num_students * avg_decrease_per_student) :
    ∃ (x : ℝ), x = weight_replaced_student - total_reduction ∧ x = 46 :=
by
  use 46
  simp [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_weight_of_new_student_l302_30216


namespace NUMINAMATH_GPT_solution_set_no_pos_ab_l302_30258

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem solution_set :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2 / 3 ≤ x ∧ x ≤ 4} :=
by sorry

theorem no_pos_ab :
  ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ 1 / a + 2 / b = 4 :=
by sorry

end NUMINAMATH_GPT_solution_set_no_pos_ab_l302_30258


namespace NUMINAMATH_GPT_negation_P_l302_30263

-- Define the original proposition P
def P (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- State the negation of P
theorem negation_P : ∀ (a b : ℝ), (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_P_l302_30263


namespace NUMINAMATH_GPT_solve_system_l302_30252

open Real

-- Define the system of equations as hypotheses
def eqn1 (x y z : ℝ) : Prop := x + y + 2 - 4 * x * y = 0
def eqn2 (x y z : ℝ) : Prop := y + z + 2 - 4 * y * z = 0
def eqn3 (x y z : ℝ) : Prop := z + x + 2 - 4 * z * x = 0

-- State the theorem
theorem solve_system (x y z : ℝ) :
  (eqn1 x y z ∧ eqn2 x y z ∧ eqn3 x y z) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by 
  sorry

end NUMINAMATH_GPT_solve_system_l302_30252


namespace NUMINAMATH_GPT_possible_integer_lengths_for_third_side_l302_30229

theorem possible_integer_lengths_for_third_side (x : ℕ) : (8 < x ∧ x < 19) ↔ (4 ≤ x ∧ x ≤ 18) :=
sorry

end NUMINAMATH_GPT_possible_integer_lengths_for_third_side_l302_30229


namespace NUMINAMATH_GPT_alyssa_went_to_13_games_last_year_l302_30251

theorem alyssa_went_to_13_games_last_year :
  ∀ (X : ℕ), (11 + X + 15 = 39) → X = 13 :=
by
  intros X h
  sorry

end NUMINAMATH_GPT_alyssa_went_to_13_games_last_year_l302_30251


namespace NUMINAMATH_GPT_marble_problem_l302_30253

theorem marble_problem : Nat.lcm (Nat.lcm (Nat.lcm 2 3) 5) 7 = 210 := by
  sorry

end NUMINAMATH_GPT_marble_problem_l302_30253


namespace NUMINAMATH_GPT_complement_U_A_l302_30218

-- Definitions based on conditions
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- Statement of the problem
theorem complement_U_A :
  (U \ A) = {4} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_l302_30218


namespace NUMINAMATH_GPT_man_speed_with_current_l302_30298

theorem man_speed_with_current
  (v : ℝ)  -- man's speed in still water
  (current_speed : ℝ) (against_current_speed : ℝ)
  (h1 : against_current_speed = v - 3.2)
  (h2 : current_speed = 3.2) :
  v = 12.8 → (v + current_speed = 16.0) :=
by
  sorry

end NUMINAMATH_GPT_man_speed_with_current_l302_30298


namespace NUMINAMATH_GPT_cubic_inequality_solution_l302_30228

theorem cubic_inequality_solution (x : ℝ) :
  (x^3 - 2 * x^2 - x + 2 > 0) ∧ (x < 3) ↔ (x < -1 ∨ (1 < x ∧ x < 3)) := 
sorry

end NUMINAMATH_GPT_cubic_inequality_solution_l302_30228


namespace NUMINAMATH_GPT_star_of_15_star_eq_neg_15_l302_30239

def y_star (y : ℤ) : ℤ := 10 - y
def star_y (y : ℤ) : ℤ := y - 10

theorem star_of_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by {
  -- applying given definitions;
  sorry
}

end NUMINAMATH_GPT_star_of_15_star_eq_neg_15_l302_30239


namespace NUMINAMATH_GPT_fraction_of_number_l302_30240

theorem fraction_of_number (x : ℕ) (f : ℚ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 :=
sorry

end NUMINAMATH_GPT_fraction_of_number_l302_30240


namespace NUMINAMATH_GPT_book_pages_l302_30297

theorem book_pages (P : ℕ) 
  (h1 : P / 2 + 11 + (P - (P / 2 + 11)) / 2 = 19)
  (h2 : P - (P / 2 + 11) = 2 * 19) : 
  P = 98 :=
by
  sorry

end NUMINAMATH_GPT_book_pages_l302_30297


namespace NUMINAMATH_GPT_new_class_mean_score_l302_30222

theorem new_class_mean_score : 
  let s1 := 68
  let n1 := 50
  let s2 := 75
  let n2 := 8
  let s3 := 82
  let n3 := 2
  (n1 * s1 + n2 * s2 + n3 * s3) / (n1 + n2 + n3) = 69.4 := by
  sorry

end NUMINAMATH_GPT_new_class_mean_score_l302_30222


namespace NUMINAMATH_GPT_range_of_x_l302_30205

variable (x : ℝ)

-- Conditions used in the problem
def sqrt_condition : Prop := x + 2 ≥ 0
def non_zero_condition : Prop := x + 1 ≠ 0

-- The statement to be proven
theorem range_of_x : sqrt_condition x ∧ non_zero_condition x ↔ (x ≥ -2 ∧ x ≠ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l302_30205


namespace NUMINAMATH_GPT_correct_calculation_result_l302_30284

theorem correct_calculation_result (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l302_30284


namespace NUMINAMATH_GPT_parabola_symmetry_l302_30278

-- Define the function f as explained in the problem
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Lean theorem to prove the inequality based on given conditions
theorem parabola_symmetry (b c : ℝ) (h : ∀ t : ℝ, f (3 + t) b c = f (3 - t) b c) :
  f 3 b c < f 1 b c ∧ f 1 b c < f 6 b c :=
by
  sorry

end NUMINAMATH_GPT_parabola_symmetry_l302_30278


namespace NUMINAMATH_GPT_sum_of_roots_l302_30219

theorem sum_of_roots (r p q : ℝ) 
  (h1 : (3 : ℝ) * r ^ 3 - (9 : ℝ) * r ^ 2 - (48 : ℝ) * r - (12 : ℝ) = 0)
  (h2 : (3 : ℝ) * p ^ 3 - (9 : ℝ) * p ^ 2 - (48 : ℝ) * p - (12 : ℝ) = 0)
  (h3 : (3 : ℝ) * q ^ 3 - (9 : ℝ) * q ^ 2 - (48 : ℝ) * q - (12 : ℝ) = 0)
  (roots_distinct : r ≠ p ∧ r ≠ q ∧ p ≠ q) :
  r + p + q = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_l302_30219


namespace NUMINAMATH_GPT_average_percent_increase_per_year_l302_30233

def initial_population : ℕ := 175000
def final_population : ℕ := 262500
def years : ℕ := 10

theorem average_percent_increase_per_year :
  ( ( ( ( final_population - initial_population ) / years : ℝ ) / initial_population ) * 100 ) = 5 := by
  sorry

end NUMINAMATH_GPT_average_percent_increase_per_year_l302_30233


namespace NUMINAMATH_GPT_problem1_problem2_l302_30242

variable {a b : ℝ}

theorem problem1 (h : a ≠ b) : 
  ((b / (a - b)) - (a / (a - b))) = -1 := 
by
  sorry

theorem problem2 (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) : 
  ((a^2 - a * b)/(a^2) / ((a / b) - (b / a))) = (b / (a + b)) := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l302_30242


namespace NUMINAMATH_GPT_rabbit_prob_top_or_bottom_l302_30225

-- Define the probability function for the rabbit to hit the top or bottom border from a given point
noncomputable def prob_reach_top_or_bottom (start : ℕ × ℕ) (board_end : ℕ × ℕ) : ℚ :=
  sorry -- Detailed probability computation based on recursive and symmetry argument

-- The proof statement for the starting point (2, 3) on a rectangular board extending to (6, 5)
theorem rabbit_prob_top_or_bottom : prob_reach_top_or_bottom (2, 3) (6, 5) = 17 / 24 :=
  sorry

end NUMINAMATH_GPT_rabbit_prob_top_or_bottom_l302_30225


namespace NUMINAMATH_GPT_sqrt_product_eq_225_l302_30268

theorem sqrt_product_eq_225 : (Real.sqrt (5 * 3) * Real.sqrt (3 ^ 3 * 5 ^ 3) = 225) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_product_eq_225_l302_30268


namespace NUMINAMATH_GPT_cos_value_of_inclined_line_l302_30224

variable (α : ℝ)
variable (l : ℝ) -- representing line as real (though we handle angles here)
variable (h_tan_line : ∃ α, tan α * (-1/2) = -1)

theorem cos_value_of_inclined_line (h_perpendicular : h_tan_line) :
  cos (2015 * Real.pi / 2 + 2 * α) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_cos_value_of_inclined_line_l302_30224


namespace NUMINAMATH_GPT_min_chord_length_l302_30214

variable (α : ℝ)

def curve_eq (x y α : ℝ) :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line_eq (x : ℝ) :=
  x = Real.pi / 4

theorem min_chord_length :
  ∃ d, (∀ α : ℝ, ∃ y1 y2 : ℝ, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ d = |y2 - y1|) ∧
  (∀ α : ℝ, ∃ y1 y2, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ |y2 - y1| ≥ d) :=
sorry

end NUMINAMATH_GPT_min_chord_length_l302_30214


namespace NUMINAMATH_GPT_prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l302_30269

noncomputable def total_outcomes := 24
noncomputable def outcomes_two_correct := 6
noncomputable def outcomes_at_least_two_correct := 7
noncomputable def outcomes_all_incorrect := 9

theorem prob_two_correct : (outcomes_two_correct : ℚ) / total_outcomes = 1 / 4 := by
  sorry

theorem prob_at_least_two_correct : (outcomes_at_least_two_correct : ℚ) / total_outcomes = 7 / 24 := by
  sorry

theorem prob_all_incorrect : (outcomes_all_incorrect : ℚ) / total_outcomes = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l302_30269


namespace NUMINAMATH_GPT_total_number_of_people_l302_30264

theorem total_number_of_people (c a : ℕ) (h1 : c = 2 * a) (h2 : c = 28) : c + a = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_people_l302_30264


namespace NUMINAMATH_GPT_prob_heart_club_spade_l302_30265

-- Definitions based on the conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13

-- Definitions based on the question
def prob_first_heart : ℚ := cards_per_suit / total_cards
def prob_second_club : ℚ := cards_per_suit / (total_cards - 1)
def prob_third_spade : ℚ := cards_per_suit / (total_cards - 2)

-- The main proof statement to be proved
theorem prob_heart_club_spade :
  prob_first_heart * prob_second_club * prob_third_spade = 169 / 10200 :=
by
  sorry

end NUMINAMATH_GPT_prob_heart_club_spade_l302_30265


namespace NUMINAMATH_GPT_base_eight_to_base_ten_l302_30243

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end NUMINAMATH_GPT_base_eight_to_base_ten_l302_30243


namespace NUMINAMATH_GPT_combined_weight_of_elephant_and_donkey_l302_30208

theorem combined_weight_of_elephant_and_donkey 
  (tons_to_pounds : ℕ → ℕ)
  (elephant_weight_tons : ℕ) 
  (donkey_percentage : ℕ) : 
  tons_to_pounds elephant_weight_tons * (1 + donkey_percentage / 100) = 6600 :=
by
  let tons_to_pounds (t : ℕ) := 2000 * t
  let elephant_weight_tons := 3
  let donkey_percentage := 10
  sorry

end NUMINAMATH_GPT_combined_weight_of_elephant_and_donkey_l302_30208


namespace NUMINAMATH_GPT_milk_quality_check_l302_30287

/-
Suppose there is a collection of 850 bags of milk numbered from 001 to 850. 
From this collection, 50 bags are randomly selected for testing by reading numbers 
from a random number table. Starting from the 3rd line and the 1st group of numbers, 
continuing to the right, we need to find the next 4 bag numbers after the sequence 
614, 593, 379, 242.
-/

def random_numbers : List Nat := [
  78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279,
  43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820,
  61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636,
  63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421,
  42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983
]

noncomputable def next_valid_numbers (nums : List Nat) (start_idx : Nat) : List Nat :=
  nums.drop start_idx |>.filter (λ n => n ≤ 850) |>.take 4

theorem milk_quality_check :
  next_valid_numbers random_numbers 18 = [203, 722, 104, 88] :=
sorry

end NUMINAMATH_GPT_milk_quality_check_l302_30287


namespace NUMINAMATH_GPT_max_n_divisor_l302_30290

theorem max_n_divisor (k n : ℕ) (h1 : 81849 % n = k) (h2 : 106392 % n = k) (h3 : 124374 % n = k) : n = 243 := by
  sorry

end NUMINAMATH_GPT_max_n_divisor_l302_30290


namespace NUMINAMATH_GPT_apps_difference_l302_30259

variable (initial_apps : ℕ) (added_apps : ℕ) (apps_left : ℕ)
variable (total_apps : ℕ := initial_apps + added_apps)
variable (deleted_apps : ℕ := total_apps - apps_left)
variable (difference : ℕ := added_apps - deleted_apps)

theorem apps_difference (h1 : initial_apps = 115) (h2 : added_apps = 235) (h3 : apps_left = 178) : 
  difference = 63 := by
  sorry

end NUMINAMATH_GPT_apps_difference_l302_30259


namespace NUMINAMATH_GPT_factorize_a3_minus_ab2_l302_30226

theorem factorize_a3_minus_ab2 (a b: ℝ) : 
  a^3 - a * b^2 = a * (a + b) * (a - b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_a3_minus_ab2_l302_30226
