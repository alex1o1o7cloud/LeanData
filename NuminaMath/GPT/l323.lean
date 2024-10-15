import Mathlib

namespace NUMINAMATH_GPT_last_digit_2008_pow_2008_l323_32340

theorem last_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := by
  -- Here, the proof would follow the understanding of the cyclic pattern of the last digits of powers of 2008
  sorry

end NUMINAMATH_GPT_last_digit_2008_pow_2008_l323_32340


namespace NUMINAMATH_GPT_total_games_played_in_league_l323_32310

theorem total_games_played_in_league (n : ℕ) (k : ℕ) (games_per_team : ℕ) 
  (h1 : n = 10) 
  (h2 : k = 4) 
  (h3 : games_per_team = n - 1) 
  : (k * (n * games_per_team) / 2) = 180 :=
by
  -- Definitions and transformations go here
  sorry

end NUMINAMATH_GPT_total_games_played_in_league_l323_32310


namespace NUMINAMATH_GPT_next_ten_winners_each_receive_160_l323_32302

def total_prize : ℕ := 2400
def first_winner_share : ℚ := 1 / 3 * total_prize
def remaining_after_first : ℚ := total_prize - first_winner_share
def next_ten_winners_share : ℚ := remaining_after_first / 10

theorem next_ten_winners_each_receive_160 :
  next_ten_winners_share = 160 := by
sorry

end NUMINAMATH_GPT_next_ten_winners_each_receive_160_l323_32302


namespace NUMINAMATH_GPT_complement_of_union_eq_l323_32311

-- Define the universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define the subset A
def A : Set ℤ := {-1, 0, 1}

-- Define the subset B
def B : Set ℤ := {0, 1, 2, 3}

-- Define the union of A and B
def A_union_B : Set ℤ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℤ := U \ A_union_B

-- State the theorem to be proved
theorem complement_of_union_eq {U A B : Set ℤ} :
  U = {-1, 0, 1, 2, 3, 4} →
  A = {-1, 0, 1} →
  B = {0, 1, 2, 3} →
  complement_U_A_union_B = {4} :=
by
  intros hU hA hB
  sorry

end NUMINAMATH_GPT_complement_of_union_eq_l323_32311


namespace NUMINAMATH_GPT_isosceles_triangle_l323_32323

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b^2 - 2 * b * c + c^2) = 0) : 
  (a = b) ∨ (b = c) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_l323_32323


namespace NUMINAMATH_GPT_sum_of_squares_l323_32324

theorem sum_of_squares (x y z : ℤ) (h1 : x + y + z = 3) (h2 : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l323_32324


namespace NUMINAMATH_GPT_max_expr_value_l323_32349

theorem max_expr_value (a b c d : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) (hc : 0 ≤ c) (hc1 : c ≤ 1) (hd : 0 ≤ d) (hd1 : d ≤ 1) : 
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_expr_value_l323_32349


namespace NUMINAMATH_GPT_sum_of_squares_of_coefficients_l323_32355

theorem sum_of_squares_of_coefficients :
  let p := 3 * (X^5 + 4 * X^3 + 2 * X + 1)
  let coeffs := [3, 12, 6, 3, 0, 0]
  let sum_squares := coeffs.map (λ c => c * c) |>.sum
  sum_squares = 198 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coefficients_l323_32355


namespace NUMINAMATH_GPT_find_d_e_f_l323_32326

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem find_d_e_f :
  ∃ (d e f : ℕ), (y ^ 50 = 3 * y ^ 48 + 10 * y ^ 45 + 9 * y ^ 43 - y ^ 25 + d * y ^ 21 + e * y ^ 19 + f * y ^ 15) 
    ∧ (d + e + f = 119) :=
sorry

end NUMINAMATH_GPT_find_d_e_f_l323_32326


namespace NUMINAMATH_GPT_kids_outside_l323_32377

theorem kids_outside (s t n c : ℕ)
  (h1 : s = 644997)
  (h2 : t = 893835)
  (h3 : n = 1538832)
  (h4 : (n - s) = t) : c = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_kids_outside_l323_32377


namespace NUMINAMATH_GPT_alicia_gumballs_l323_32321

theorem alicia_gumballs (A : ℕ) (h1 : 3 * A = 60) : A = 20 := sorry

end NUMINAMATH_GPT_alicia_gumballs_l323_32321


namespace NUMINAMATH_GPT_sum_of_money_l323_32317

theorem sum_of_money (jimin_100_won : ℕ) (jimin_50_won : ℕ) (seokjin_100_won : ℕ) (seokjin_10_won : ℕ) 
  (h1 : jimin_100_won = 5) (h2 : jimin_50_won = 1) (h3 : seokjin_100_won = 2) (h4 : seokjin_10_won = 7) :
  jimin_100_won * 100 + jimin_50_won * 50 + seokjin_100_won * 100 + seokjin_10_won * 10 = 820 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_money_l323_32317


namespace NUMINAMATH_GPT_partition_displacement_l323_32391

variables (l : ℝ) (R T : ℝ) (initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)

-- Conditions
def initial_conditions (initial_V1 initial_V2 : ℝ) : Prop :=
  initial_V1 + initial_V2 = l ∧
  initial_V2 = 2 * initial_V1 ∧
  initial_P1 * initial_V1 = R * T ∧
  initial_P2 * initial_V2 = 2 * R * T ∧
  initial_P1 = initial_P2

-- Final volumes
def final_volumes (final_Vleft final_Vright : ℝ) : Prop :=
  final_Vleft = l / 2 ∧ final_Vright = l / 2 

-- Displacement of the partition
def displacement (initial_position final_position : ℝ) : ℝ :=
  initial_position - final_position

-- Theorem statement: the displacement of the partition is l / 6
theorem partition_displacement (l R T initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)
  (h_initial_cond : initial_conditions l R T initial_V1 initial_V2 initial_P1 initial_P2)
  (h_final_vol : final_volumes l final_Vleft final_Vright) 
  (initial_position final_position : ℝ)
  (initial_position_def : initial_position = 2 * l / 3)
  (final_position_def : final_position = l / 2) :
  displacement initial_position final_position = l / 6 := 
by sorry

end NUMINAMATH_GPT_partition_displacement_l323_32391


namespace NUMINAMATH_GPT_max_monthly_profit_l323_32387

theorem max_monthly_profit (x : ℝ) (h : 0 < x ∧ x ≤ 15) :
  let C := 100 + 4 * x
  let p := 76 + 15 * x - x^2
  let L := p * x - C
  L = -x^3 + 15 * x^2 + 72 * x - 100 ∧
  (∀ x, 0 < x ∧ x ≤ 15 → L ≤ -12^3 + 15 * 12^2 + 72 * 12 - 100) :=
by
  sorry

end NUMINAMATH_GPT_max_monthly_profit_l323_32387


namespace NUMINAMATH_GPT_point_3_units_away_l323_32330

theorem point_3_units_away (x : ℤ) (h : abs (x + 1) = 3) : x = 2 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_point_3_units_away_l323_32330


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l323_32312

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (a_6 : a 6 = 2) : 
  (11 * (a 1 + (a 1 + 10 * ((a 6 - a 1) / 5))) / 2) = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l323_32312


namespace NUMINAMATH_GPT_injective_of_comp_injective_surjective_of_comp_surjective_l323_32373

section FunctionProperties

variables {X Y V : Type} (f : X → Y) (g : Y → V)

-- Proof for part (i) if g ∘ f is injective, then f is injective
theorem injective_of_comp_injective (h : Function.Injective (g ∘ f)) : Function.Injective f :=
  sorry

-- Proof for part (ii) if g ∘ f is surjective, then g is surjective
theorem surjective_of_comp_surjective (h : Function.Surjective (g ∘ f)) : Function.Surjective g :=
  sorry

end FunctionProperties

end NUMINAMATH_GPT_injective_of_comp_injective_surjective_of_comp_surjective_l323_32373


namespace NUMINAMATH_GPT_rect_area_perimeter_l323_32382

def rect_perimeter (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem rect_area_perimeter (Area Length : ℕ) (hArea : Area = 192) (hLength : Length = 24) :
  ∃ (Width Perimeter : ℕ), Width = Area / Length ∧ Perimeter = rect_perimeter Length Width ∧ Perimeter = 64 :=
by
  sorry

end NUMINAMATH_GPT_rect_area_perimeter_l323_32382


namespace NUMINAMATH_GPT_value_of_expression_l323_32306

theorem value_of_expression (x y : ℤ) (h1 : x = -6) (h2 : y = -3) : 4 * (x - y) ^ 2 - x * y = 18 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l323_32306


namespace NUMINAMATH_GPT_prob_log3_integer_l323_32314

theorem prob_log3_integer : 
  (∃ (N: ℕ), (100 ≤ N ∧ N ≤ 999) ∧ ∃ (k: ℕ), N = 3^k) → 
  (∃ (prob : ℚ), prob = 1 / 450) :=
sorry

end NUMINAMATH_GPT_prob_log3_integer_l323_32314


namespace NUMINAMATH_GPT_all_radii_equal_l323_32328
-- Lean 4 statement

theorem all_radii_equal (r : ℝ) (h : r = 2) : r = 2 :=
by
  sorry

end NUMINAMATH_GPT_all_radii_equal_l323_32328


namespace NUMINAMATH_GPT_percentage_decrease_of_y_compared_to_z_l323_32341

theorem percentage_decrease_of_y_compared_to_z (x y z : ℝ)
  (h1 : x = 1.20 * y)
  (h2 : x = 0.60 * z) :
  (y = 0.50 * z) → (1 - (y / z)) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_of_y_compared_to_z_l323_32341


namespace NUMINAMATH_GPT_quadratic_equation_even_coefficient_l323_32335

-- Define the predicate for a rational root
def has_rational_root (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), (q ≠ 0) ∧ (p.gcd q = 1) ∧ (a * p^2 + b * p * q + c * q^2 = 0)

-- Define the predicate for at least one being even
def at_least_one_even (a b c : ℤ) : Prop :=
  (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0)

theorem quadratic_equation_even_coefficient 
  (a b c : ℤ) (h_non_zero : a ≠ 0) (h_rational_root : has_rational_root a b c) :
  at_least_one_even a b c :=
sorry

end NUMINAMATH_GPT_quadratic_equation_even_coefficient_l323_32335


namespace NUMINAMATH_GPT_find_f_7_l323_32374

noncomputable def f (a b c d x : ℝ) : ℝ :=
  a * x^8 + b * x^7 + c * x^3 + d * x - 6

theorem find_f_7 (a b c d : ℝ) (h : f a b c d (-7) = 10) :
  f a b c d 7 = 11529580 * a - 22 :=
sorry

end NUMINAMATH_GPT_find_f_7_l323_32374


namespace NUMINAMATH_GPT_servings_in_container_l323_32339

def convert_to_improper_fraction (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

def servings (container : ℚ) (serving_size : ℚ) : ℚ :=
  container / serving_size

def mixed_number (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

theorem servings_in_container : 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  servings container serving_size = expected_servings :=
by 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  sorry

end NUMINAMATH_GPT_servings_in_container_l323_32339


namespace NUMINAMATH_GPT_total_hats_purchased_l323_32389

theorem total_hats_purchased (B G : ℕ) (h1 : G = 38) (h2 : 6 * B + 7 * G = 548) : B + G = 85 := 
by 
  sorry

end NUMINAMATH_GPT_total_hats_purchased_l323_32389


namespace NUMINAMATH_GPT_min_abc_sum_l323_32383

theorem min_abc_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 8) : a + b + c ≥ 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_abc_sum_l323_32383


namespace NUMINAMATH_GPT_number_of_DVDs_sold_l323_32342

theorem number_of_DVDs_sold (C D: ℤ) (h₁ : D = 16 * C / 10) (h₂ : D + C = 273) : D = 168 := 
sorry

end NUMINAMATH_GPT_number_of_DVDs_sold_l323_32342


namespace NUMINAMATH_GPT_proof_d_e_f_value_l323_32338

theorem proof_d_e_f_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 :=
sorry

end NUMINAMATH_GPT_proof_d_e_f_value_l323_32338


namespace NUMINAMATH_GPT_min_value_x3y3z2_is_1_over_27_l323_32367

noncomputable def min_value_x3y3z2 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h' : 1 / x + 1 / y + 1 / z = 9) : ℝ :=
  x^3 * y^3 * z^2

theorem min_value_x3y3z2_is_1_over_27 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z)
  (h' : 1 / x + 1 / y + 1 / z = 9) : min_value_x3y3z2 x y z h h' = 1 / 27 :=
sorry

end NUMINAMATH_GPT_min_value_x3y3z2_is_1_over_27_l323_32367


namespace NUMINAMATH_GPT_mostSuitableSampleSurvey_l323_32359

-- Conditions
def conditionA := "Security check for passengers before boarding a plane"
def conditionB := "Understanding the amount of physical exercise each classmate does per week"
def conditionC := "Interviewing job applicants for a company's recruitment process"
def conditionD := "Understanding the lifespan of a batch of light bulbs"

-- Define a predicate to determine the most suitable for a sample survey
def isMostSuitableForSampleSurvey (s : String) : Prop :=
  s = conditionD

-- Theorem statement
theorem mostSuitableSampleSurvey :
  isMostSuitableForSampleSurvey conditionD :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_mostSuitableSampleSurvey_l323_32359


namespace NUMINAMATH_GPT_cube_has_12_edges_l323_32304

-- Definition of the number of edges in a cube
def number_of_edges_of_cube : Nat := 12

-- The theorem that asserts the cube has 12 edges
theorem cube_has_12_edges : number_of_edges_of_cube = 12 := by
  -- proof to be filled later
  sorry

end NUMINAMATH_GPT_cube_has_12_edges_l323_32304


namespace NUMINAMATH_GPT_gumball_probability_l323_32356

theorem gumball_probability :
  let total_gumballs : ℕ := 25
  let orange_gumballs : ℕ := 10
  let green_gumballs : ℕ := 6
  let yellow_gumballs : ℕ := 9
  let total_gumballs_after_first : ℕ := total_gumballs - 1
  let total_gumballs_after_second : ℕ := total_gumballs - 2
  let orange_probability_first : ℚ := orange_gumballs / total_gumballs
  let green_or_yellow_probability_second : ℚ := (green_gumballs + yellow_gumballs) / total_gumballs_after_first
  let orange_probability_third : ℚ := (orange_gumballs - 1) / total_gumballs_after_second
  orange_probability_first * green_or_yellow_probability_second * orange_probability_third = 9 / 92 :=
by
  sorry

end NUMINAMATH_GPT_gumball_probability_l323_32356


namespace NUMINAMATH_GPT_z_value_l323_32375

theorem z_value (z : ℝ) (h : |z + 2| = |z - 3|) : z = 1 / 2 := 
sorry

end NUMINAMATH_GPT_z_value_l323_32375


namespace NUMINAMATH_GPT_quadrilateral_is_parallelogram_l323_32364

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : (a - c) ^ 2 + (b - d) ^ 2 = 0) : 
  -- The theorem states that if lengths a, b, c, d of a quadrilateral satisfy the given equation,
  -- then the quadrilateral must be a parallelogram.
  a = c ∧ b = d :=
by {
  sorry
}

end NUMINAMATH_GPT_quadrilateral_is_parallelogram_l323_32364


namespace NUMINAMATH_GPT_train_speed_A_to_B_l323_32316

-- Define the constants
def distance : ℝ := 480
def return_speed : ℝ := 120
def return_time_longer : ℝ := 1

-- Define the train's speed function on its way from A to B
noncomputable def train_speed : ℝ := distance / (4 - return_time_longer) -- This simplifies directly to 160 based on the provided conditions.

-- State the theorem
theorem train_speed_A_to_B :
  distance / train_speed + return_time_longer = distance / return_speed :=
by
  -- Result follows from the given conditions directly
  sorry

end NUMINAMATH_GPT_train_speed_A_to_B_l323_32316


namespace NUMINAMATH_GPT_trainB_speed_l323_32309

variable (v : ℕ)

def trainA_speed : ℕ := 30
def time_gap : ℕ := 2
def distance_overtake : ℕ := 360

theorem trainB_speed (h :  v > trainA_speed) : v = 42 :=
by
  sorry

end NUMINAMATH_GPT_trainB_speed_l323_32309


namespace NUMINAMATH_GPT_product_B_original_price_l323_32386

variable (a b : ℝ)

theorem product_B_original_price (h1 : a = 1.2 * b) (h2 : 0.9 * a = 198) : b = 183.33 :=
by
  sorry

end NUMINAMATH_GPT_product_B_original_price_l323_32386


namespace NUMINAMATH_GPT_tan_zero_l323_32398

theorem tan_zero : Real.tan 0 = 0 := 
by
  sorry

end NUMINAMATH_GPT_tan_zero_l323_32398


namespace NUMINAMATH_GPT_balancing_point_is_vertex_l323_32394

-- Define a convex polygon and its properties
structure ConvexPolygon (n : ℕ) :=
(vertices : Fin n → Point)

-- Define a balancing point for a convex polygon
def is_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  -- Placeholder for the actual definition that the areas formed by drawing lines from Q to vertices of P are equal
  sorry

-- Define the uniqueness of the balancing point
def unique_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  ∀ R : Point, is_balancing_point P R → R = Q

-- Main theorem statement
theorem balancing_point_is_vertex (P : ConvexPolygon n) (Q : Point) 
  (h_balance : is_balancing_point P Q) (h_unique : unique_balancing_point P Q) : 
  ∃ i : Fin n, Q = P.vertices i :=
sorry

end NUMINAMATH_GPT_balancing_point_is_vertex_l323_32394


namespace NUMINAMATH_GPT_sum_powers_mod_7_l323_32307

theorem sum_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_sum_powers_mod_7_l323_32307


namespace NUMINAMATH_GPT_avg_growth_rate_leq_half_sum_l323_32348

theorem avg_growth_rate_leq_half_sum (m n p : ℝ) (hm : 0 ≤ m) (hn : 0 ≤ n)
    (hp : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
    p ≤ (m + n) / 2 :=
by
  sorry

end NUMINAMATH_GPT_avg_growth_rate_leq_half_sum_l323_32348


namespace NUMINAMATH_GPT_time_to_traverse_nth_mile_l323_32381

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℝ, (∀ d : ℝ, d = n - 1 → (s_n = k / d)) ∧ (s_2 = 1 / 2)) → 
  t_n = 2 * (n - 1) :=
by 
  sorry

end NUMINAMATH_GPT_time_to_traverse_nth_mile_l323_32381


namespace NUMINAMATH_GPT_standard_deviation_does_not_require_repair_l323_32351

-- Definitions based on conditions
def greatest_deviation (d : ℝ) := d = 39
def nominal_mass (M : ℝ) := 0.1 * M = 39
def unreadable_measurement_deviation (d : ℝ) := d < 39

-- Theorems to be proved
theorem standard_deviation (σ : ℝ) (d : ℝ) (M : ℝ) :
  greatest_deviation d →
  nominal_mass M →
  unreadable_measurement_deviation d →
  σ ≤ 39 :=
by
  sorry

theorem does_not_require_repair (σ : ℝ) :
  σ ≤ 39 → ¬(machine_requires_repair) :=
by
  sorry

-- Adding an assumption that if σ ≤ 39, the machine does not require repair
axiom machine_requires_repair : Prop

end NUMINAMATH_GPT_standard_deviation_does_not_require_repair_l323_32351


namespace NUMINAMATH_GPT_neither_probability_l323_32357

-- Definitions of the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℝ := 0.63
def P_B : ℝ := 0.49
def P_A_and_B : ℝ := 0.32

-- Definition stating the probability of neither event
theorem neither_probability :
  (1 - (P_A + P_B - P_A_and_B)) = 0.20 := 
sorry

end NUMINAMATH_GPT_neither_probability_l323_32357


namespace NUMINAMATH_GPT_shift_sin_to_cos_l323_32363

open Real

theorem shift_sin_to_cos:
  ∀ x: ℝ, 3 * cos (2 * x) = 3 * sin (2 * (x + π / 6) - π / 6) :=
by 
  sorry

end NUMINAMATH_GPT_shift_sin_to_cos_l323_32363


namespace NUMINAMATH_GPT_determinant_equality_l323_32303

-- Given values p, q, r, s such that the determinant of the first matrix is 5
variables {p q r s : ℝ}

-- Define the determinant condition
def det_condition (p q r s : ℝ) : Prop := p * s - q * r = 5

-- State the theorem that we need to prove
theorem determinant_equality (h : det_condition p q r s) :
  p * (5*r + 2*s) - r * (5*p + 2*q) = 10 :=
sorry

end NUMINAMATH_GPT_determinant_equality_l323_32303


namespace NUMINAMATH_GPT_umbrella_cost_l323_32385

theorem umbrella_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) (h1 : house_umbrellas = 2) (h2 : car_umbrellas = 1) (h3 : cost_per_umbrella = 8) : 
  (house_umbrellas + car_umbrellas) * cost_per_umbrella = 24 := 
by
  sorry

end NUMINAMATH_GPT_umbrella_cost_l323_32385


namespace NUMINAMATH_GPT_minimum_value_nine_l323_32358

noncomputable def min_value (a b c k : ℝ) : ℝ :=
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a

theorem minimum_value_nine (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  min_value a b c k ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_nine_l323_32358


namespace NUMINAMATH_GPT_liquid_X_percentage_in_B_l323_32372

noncomputable def percentage_of_solution_B (X_A : ℝ) (w_A w_B total_X : ℝ) : ℝ :=
  let X_B := (total_X - (w_A * (X_A / 100))) / w_B 
  X_B * 100

theorem liquid_X_percentage_in_B :
  percentage_of_solution_B 0.8 500 700 19.92 = 2.274 := by
  sorry

end NUMINAMATH_GPT_liquid_X_percentage_in_B_l323_32372


namespace NUMINAMATH_GPT_geometric_series_sum_l323_32337

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l323_32337


namespace NUMINAMATH_GPT_circle_center_radius_sum_l323_32329

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), c = -6 ∧ d = -7 ∧ s = Real.sqrt 13 ∧
  (x^2 + 14 * y + 72 = -y^2 - 12 * x → c + d + s = -13 + Real.sqrt 13) :=
sorry

end NUMINAMATH_GPT_circle_center_radius_sum_l323_32329


namespace NUMINAMATH_GPT_largest_term_quotient_l323_32318

theorem largest_term_quotient (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n, S n = (n * (a 0 + a n)) / 2)
  (h_S15_pos : S 15 > 0)
  (h_S16_neg : S 16 < 0) :
  ∃ m, 1 ≤ m ∧ m ≤ 15 ∧
       ∀ k, (1 ≤ k ∧ k ≤ 15) → (S m / a m) ≥ (S k / a k) ∧ m = 8 := 
sorry

end NUMINAMATH_GPT_largest_term_quotient_l323_32318


namespace NUMINAMATH_GPT_problem_acute_angles_l323_32362

theorem problem_acute_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h1 : 3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1)
  (h2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := 
by 
  sorry

end NUMINAMATH_GPT_problem_acute_angles_l323_32362


namespace NUMINAMATH_GPT_percentage_increase_second_year_l323_32390

theorem percentage_increase_second_year 
  (initial_population : ℝ)
  (first_year_increase : ℝ) 
  (population_after_2_years : ℝ) 
  (final_population : ℝ)
  (H_initial_population : initial_population = 800)
  (H_first_year_increase : first_year_increase = 0.22)
  (H_population_after_2_years : final_population = 1220) :
  ∃ P : ℝ, P = 25 := 
by
  -- Define the population after the first year
  let population_after_first_year := initial_population * (1 + first_year_increase)
  -- Define the equation relating populations and solve for P
  let second_year_increase := (final_population / population_after_first_year - 1) * 100
  -- Show P equals 25
  use second_year_increase
  sorry

end NUMINAMATH_GPT_percentage_increase_second_year_l323_32390


namespace NUMINAMATH_GPT_car_bus_initial_speed_l323_32376

theorem car_bus_initial_speed {d : ℝ} {t : ℝ} {s_c : ℝ} {s_b : ℝ}
    (h1 : t = 4) 
    (h2 : s_c = s_b + 8) 
    (h3 : d = 384)
    (h4 : ∀ t, 0 ≤ t → t ≤ 2 → d = s_c * t + s_b * t) 
    (h5 : ∀ t, 2 < t → t ≤ 4 → d = (s_c - 10) * (t - 2) + s_b * (t - 2)) 
    : s_b = 46.5 ∧ s_c = 54.5 := 
by 
    sorry

end NUMINAMATH_GPT_car_bus_initial_speed_l323_32376


namespace NUMINAMATH_GPT_fraction_identity_l323_32327

variable {n : ℕ}

theorem fraction_identity
  (h1 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1)))
  (h2 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → n ≠ 2 → 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))))
  : 1 / (n * (n + 1) * (n + 2) * (n + 3)) = 1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end NUMINAMATH_GPT_fraction_identity_l323_32327


namespace NUMINAMATH_GPT_negation_of_universal_statement_l323_32350

theorem negation_of_universal_statement:
  (∀ x : ℝ, x ≥ 2) ↔ ¬ (∃ x : ℝ, x < 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_universal_statement_l323_32350


namespace NUMINAMATH_GPT_sum_ages_l323_32354

theorem sum_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 := 
by 
  sorry

end NUMINAMATH_GPT_sum_ages_l323_32354


namespace NUMINAMATH_GPT_polynomial_degree_l323_32392

noncomputable def polynomial1 : Polynomial ℤ := 3 * Polynomial.monomial 5 1 + 2 * Polynomial.monomial 4 1 - Polynomial.monomial 1 1 + Polynomial.C 5
noncomputable def polynomial2 : Polynomial ℤ := 4 * Polynomial.monomial 11 1 - 2 * Polynomial.monomial 8 1 + 5 * Polynomial.monomial 5 1 - Polynomial.C 9
noncomputable def polynomial3 : Polynomial ℤ := (Polynomial.monomial 2 1 - Polynomial.C 3) ^ 9

theorem polynomial_degree :
  (polynomial1 * polynomial2 - polynomial3).degree = 18 := by
  sorry

end NUMINAMATH_GPT_polynomial_degree_l323_32392


namespace NUMINAMATH_GPT_daniel_practices_each_school_day_l323_32368

-- Define the conditions
def total_minutes : ℕ := 135
def school_days : ℕ := 5
def weekend_days : ℕ := 2

-- Define the variables
def x : ℕ := 15

-- Define the practice time equations
def school_week_practice_time (x : ℕ) := school_days * x
def weekend_practice_time (x : ℕ) := weekend_days * 2 * x
def total_practice_time (x : ℕ) := school_week_practice_time x + weekend_practice_time x

-- The proof goal
theorem daniel_practices_each_school_day :
  total_practice_time x = total_minutes := by
  sorry

end NUMINAMATH_GPT_daniel_practices_each_school_day_l323_32368


namespace NUMINAMATH_GPT_sequence_formula_l323_32300

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 1) (h_recurrence : ∀ n : ℕ, 2 * n * a n + 1 = (n + 1) * a n) :
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_formula_l323_32300


namespace NUMINAMATH_GPT_value_of_algebraic_expression_l323_32322

variable {a b : ℝ}

theorem value_of_algebraic_expression (h : b = 4 * a + 3) : 4 * a - b - 2 = -5 := 
by
  sorry

end NUMINAMATH_GPT_value_of_algebraic_expression_l323_32322


namespace NUMINAMATH_GPT_solve_fraction_equation_l323_32370

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 0 ↔ x = -3 :=
sorry

end NUMINAMATH_GPT_solve_fraction_equation_l323_32370


namespace NUMINAMATH_GPT_solve_a_minus_b_l323_32336

theorem solve_a_minus_b (a b : ℝ) (h1 : 2010 * a + 2014 * b = 2018) (h2 : 2012 * a + 2016 * b = 2020) : a - b = -3 :=
sorry

end NUMINAMATH_GPT_solve_a_minus_b_l323_32336


namespace NUMINAMATH_GPT_sqrt_expression_equals_l323_32365

theorem sqrt_expression_equals : Real.sqrt (5^2 * 7^4) = 245 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_equals_l323_32365


namespace NUMINAMATH_GPT_mailman_total_pieces_l323_32397

def piecesOfMailFirstHouse := 6 + 5 + 3 + 4 + 2
def piecesOfMailSecondHouse := 4 + 7 + 2 + 5 + 3
def piecesOfMailThirdHouse := 8 + 3 + 4 + 6 + 1

def totalPiecesOfMail := piecesOfMailFirstHouse + piecesOfMailSecondHouse + piecesOfMailThirdHouse

theorem mailman_total_pieces : totalPiecesOfMail = 63 := by
  sorry

end NUMINAMATH_GPT_mailman_total_pieces_l323_32397


namespace NUMINAMATH_GPT_reserved_fraction_l323_32308

variable (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ)
variable (f : ℚ)

def mrSalazarFractionReserved (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ) : ℚ :=
  1 - (leftover_oranges + rotten_oranges) * sold_fraction / initial_oranges

theorem reserved_fraction (h1 : initial_oranges = 84) (h2 : sold_fraction = 3 / 7) (h3 : rotten_oranges = 4) (h4 : leftover_oranges = 32) :
  (mrSalazarFractionReserved initial_oranges sold_fraction rotten_oranges leftover_oranges) = 1 / 4 :=
  by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_reserved_fraction_l323_32308


namespace NUMINAMATH_GPT_smallest_result_l323_32344

-- Define the given set of numbers
def given_set : Set Nat := {3, 4, 7, 11, 13, 14}

-- Define the condition for prime numbers greater than 10
def is_prime_gt_10 (n : Nat) : Prop :=
  Nat.Prime n ∧ n > 10

-- Define the property of choosing three different numbers and computing the result
def compute (a b c : Nat) : Nat :=
  (a + b) * c

-- The main theorem stating the problem and its solution
theorem smallest_result : ∃ (a b c : Nat), 
  a ∈ given_set ∧ b ∈ given_set ∧ c ∈ given_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (is_prime_gt_10 a ∨ is_prime_gt_10 b ∨ is_prime_gt_10 c) ∧
  compute a b c = 77 ∧
  ∀ (a' b' c' : Nat), 
    a' ∈ given_set ∧ b' ∈ given_set ∧ c' ∈ given_set ∧
    a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
    (is_prime_gt_10 a' ∨ is_prime_gt_10 b' ∨ is_prime_gt_10 c') →
    compute a' b' c' ≥ 77 :=
by
  -- Proof is not required, hence sorry
  sorry

end NUMINAMATH_GPT_smallest_result_l323_32344


namespace NUMINAMATH_GPT_sum_first_5n_eq_630_l323_32366

theorem sum_first_5n_eq_630 (n : ℕ)
  (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 300) :
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end NUMINAMATH_GPT_sum_first_5n_eq_630_l323_32366


namespace NUMINAMATH_GPT_rectangular_prism_total_count_l323_32319

-- Define the dimensions of the rectangular prism
def length : ℕ := 4
def width : ℕ := 3
def height : ℕ := 5

-- Define the total count of edges, corners, and faces
def total_count : ℕ := 12 + 8 + 6

-- The proof statement that the total count is 26
theorem rectangular_prism_total_count : total_count = 26 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_total_count_l323_32319


namespace NUMINAMATH_GPT_tennis_tournament_matches_l323_32388

theorem tennis_tournament_matches (n : ℕ) (h₁ : n = 128) (h₂ : ∃ m : ℕ, m = 32) (h₃ : ∃ k : ℕ, k = 96) (h₄ : ∀ i : ℕ, i > 1 → i ≤ n → ∃ j : ℕ, j = 1 + (i - 1)) :
  ∃ total_matches : ℕ, total_matches = 127 := 
by 
  sorry

end NUMINAMATH_GPT_tennis_tournament_matches_l323_32388


namespace NUMINAMATH_GPT_calc_problem_l323_32361

def odot (a b : ℕ) : ℕ := a * b - (a + b)

theorem calc_problem : odot 6 (odot 5 4) = 49 :=
by
  sorry

end NUMINAMATH_GPT_calc_problem_l323_32361


namespace NUMINAMATH_GPT_moon_land_value_l323_32346

theorem moon_land_value (surface_area_earth : ℕ) (surface_area_moon : ℕ) (total_value_earth : ℕ) (worth_factor : ℕ)
  (h_moon_surface_area : surface_area_moon = surface_area_earth / 5)
  (h_surface_area_earth : surface_area_earth = 200) 
  (h_worth_factor : worth_factor = 6) 
  (h_total_value_earth : total_value_earth = 80) : (total_value_earth / 5) * worth_factor = 96 := 
by 
  -- Simplify using the given conditions
  -- total_value_earth / 5 is the value of the moon's land if it had the same value per square acre as Earth's land
  -- multiplying by worth_factor to get the total value on the moon
  sorry

end NUMINAMATH_GPT_moon_land_value_l323_32346


namespace NUMINAMATH_GPT_sqrt_ab_is_integer_l323_32331

theorem sqrt_ab_is_integer
  (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
  (h_eq : a * (b^2 + n^2) = b * (a^2 + n^2)) :
  ∃ k : ℕ, k * k = a * b :=
by
  sorry

end NUMINAMATH_GPT_sqrt_ab_is_integer_l323_32331


namespace NUMINAMATH_GPT_product_of_three_consecutive_integers_is_square_l323_32379

theorem product_of_three_consecutive_integers_is_square (x : ℤ) : 
  ∃ n : ℤ, x * (x + 1) * (x + 2) = n^2 → x = 0 ∨ x = -1 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_consecutive_integers_is_square_l323_32379


namespace NUMINAMATH_GPT_total_sampled_papers_l323_32352

-- Define the conditions
variables {A B C c : ℕ}
variable (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50)
variable (stratified_sampling : true)   -- We simply denote that stratified sampling method is used

-- Theorem to prove the total number of exam papers sampled
theorem total_sampled_papers {T : ℕ} (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50) (stratified_sampling : true) :
  T = (1260 + 720 + 900) * (50 / 900) := sorry

end NUMINAMATH_GPT_total_sampled_papers_l323_32352


namespace NUMINAMATH_GPT_half_product_two_consecutive_integers_mod_3_l323_32305

theorem half_product_two_consecutive_integers_mod_3 (A : ℤ) : 
  (A * (A + 1) / 2) % 3 = 0 ∨ (A * (A + 1) / 2) % 3 = 1 :=
sorry

end NUMINAMATH_GPT_half_product_two_consecutive_integers_mod_3_l323_32305


namespace NUMINAMATH_GPT_correct_equation_l323_32360

-- Define the conditions
variables {x : ℝ}

-- Condition 1: The unit price of a notebook is 2 yuan less than that of a water-based pen.
def notebook_price (water_pen_price : ℝ) : ℝ := water_pen_price - 2

-- Condition 2: Xiaogang bought 5 notebooks and 3 water-based pens for exactly 14 yuan.
def total_cost (notebook_price water_pen_price : ℝ) : ℝ :=
  5 * notebook_price + 3 * water_pen_price

-- Question restated as a theorem: Verify the given equation is correct
theorem correct_equation (water_pen_price : ℝ) (h : total_cost (notebook_price water_pen_price) water_pen_price = 14) :
  5 * (water_pen_price - 2) + 3 * water_pen_price = 14 :=
  by
    -- Introduce the assumption
    intros
    -- Sorry to skip the proof
    sorry

end NUMINAMATH_GPT_correct_equation_l323_32360


namespace NUMINAMATH_GPT_exists_unique_decomposition_l323_32313

theorem exists_unique_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ),
    (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n, z n ≥ z (n-1)) ∧
    (∀ n, y n * (z n - z (n-1)) = 0) ∧
    z 0 = 0 :=
sorry

end NUMINAMATH_GPT_exists_unique_decomposition_l323_32313


namespace NUMINAMATH_GPT_coloring_ways_l323_32332

-- Define the function that checks valid coloring
noncomputable def valid_coloring (colors : Fin 6 → Fin 3) : Prop :=
  colors 0 = 0 ∧ -- The central pentagon is colored red
  (colors 1 ≠ colors 0 ∧ colors 2 ≠ colors 1 ∧ 
   colors 3 ≠ colors 2 ∧ colors 4 ≠ colors 3 ∧ 
   colors 5 ≠ colors 4 ∧ colors 1 ≠ colors 5) -- No two adjacent polygons have the same color

-- Define the main theorem
theorem coloring_ways (f : Fin 6 → Fin 3) (h : valid_coloring f) : 
  ∃! (f : Fin 6 → Fin 3), valid_coloring f := by
  sorry

end NUMINAMATH_GPT_coloring_ways_l323_32332


namespace NUMINAMATH_GPT_kathryn_gave_56_pencils_l323_32353

-- Define the initial and total number of pencils
def initial_pencils : ℕ := 9
def total_pencils : ℕ := 65

-- Define the number of pencils Kathryn gave to Anthony
def pencils_given : ℕ := total_pencils - initial_pencils

-- Prove that Kathryn gave Anthony 56 pencils
theorem kathryn_gave_56_pencils : pencils_given = 56 :=
by
  -- Proof is omitted as per the requirement
  sorry

end NUMINAMATH_GPT_kathryn_gave_56_pencils_l323_32353


namespace NUMINAMATH_GPT_swimmer_distance_l323_32320

theorem swimmer_distance :
  let swimmer_speed : ℝ := 3
  let current_speed : ℝ := 1.7
  let time : ℝ := 2.3076923076923075
  let effective_speed := swimmer_speed - current_speed
  let distance := effective_speed * time
  distance = 3 := by
sorry

end NUMINAMATH_GPT_swimmer_distance_l323_32320


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l323_32315

theorem arithmetic_sequence_ratio (S T : ℕ → ℕ) (a b : ℕ → ℕ)
  (h : ∀ n, S n / T n = (7 * n + 3) / (n + 3)) :
  a 8 / b 8 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l323_32315


namespace NUMINAMATH_GPT_sequence_a_2011_l323_32384

noncomputable def sequence_a : ℕ → ℕ
| 0       => 2
| 1       => 3
| (n+2)   => (sequence_a (n+1) * sequence_a n) % 10

theorem sequence_a_2011 : sequence_a 2010 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_2011_l323_32384


namespace NUMINAMATH_GPT_area_ratio_l323_32371

variables {A B C D: Type} [LinearOrderedField A]
variables {AB AD AR AE : A}

-- Conditions
axiom cond1 : AR = (2 / 3) * AB
axiom cond2 : AE = (1 / 3) * AD

theorem area_ratio (h : A) (h1 : A) (S_ABCD : A) (S_ARE : A)
  (h_eq : S_ABCD = AD * h)
  (h1_eq : S_ARE = (1 / 2) * AE * h1)
  (ratio_heights : h / h1 = 3 / 2) :
  S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_area_ratio_l323_32371


namespace NUMINAMATH_GPT_KeatonAnnualEarnings_l323_32393

-- Keaton's conditions for oranges
def orangeHarvestInterval : ℕ := 2
def orangeSalePrice : ℕ := 50

-- Keaton's conditions for apples
def appleHarvestInterval : ℕ := 3
def appleSalePrice : ℕ := 30

-- Annual earnings calculation
def annualEarnings (monthsInYear : ℕ) : ℕ :=
  let orangeEarnings := (monthsInYear / orangeHarvestInterval) * orangeSalePrice
  let appleEarnings := (monthsInYear / appleHarvestInterval) * appleSalePrice
  orangeEarnings + appleEarnings

-- Prove the total annual earnings is 420
theorem KeatonAnnualEarnings : annualEarnings 12 = 420 :=
  by 
    -- We skip the proof details here.
    sorry

end NUMINAMATH_GPT_KeatonAnnualEarnings_l323_32393


namespace NUMINAMATH_GPT_smallest_positive_e_l323_32380

-- Define the polynomial and roots condition
def polynomial (a b c d e : ℤ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

def has_integer_roots (p : ℝ → ℝ) (roots : List ℝ) : Prop :=
  ∀ r ∈ roots, p r = 0

def polynomial_with_given_roots (a b c d e : ℤ) : Prop :=
  has_integer_roots (polynomial a b c d e) [-3, 4, 11, -(1/4)]

-- Main theorem to prove the smallest positive integer e
theorem smallest_positive_e (a b c d : ℤ) :
  ∃ e : ℤ, e > 0 ∧ polynomial_with_given_roots a b c d e ∧
            (∀ e' : ℤ, e' > 0 ∧ polynomial_with_given_roots a b c d e' → e ≤ e') :=
  sorry

end NUMINAMATH_GPT_smallest_positive_e_l323_32380


namespace NUMINAMATH_GPT_distance_between_wheels_l323_32301

theorem distance_between_wheels 
  (D : ℕ) 
  (back_perimeter : ℕ) (front_perimeter : ℕ) 
  (more_revolutions : ℕ)
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : more_revolutions = 10)
  (h4 : D / front_perimeter = D / back_perimeter + more_revolutions) : 
  D = 315 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_wheels_l323_32301


namespace NUMINAMATH_GPT_joshInitialMarbles_l323_32347

-- Let n be the number of marbles Josh initially had
variable (n : ℕ)

-- Condition 1: Jack gave Josh 20 marbles
def jackGaveJoshMarbles : ℕ := 20

-- Condition 2: Now Josh has 42 marbles
def joshCurrentMarbles : ℕ := 42

-- Theorem: prove that the number of marbles Josh had initially was 22
theorem joshInitialMarbles : n + jackGaveJoshMarbles = joshCurrentMarbles → n = 22 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_joshInitialMarbles_l323_32347


namespace NUMINAMATH_GPT_integer_mod_105_l323_32399

theorem integer_mod_105 (x : ℤ) :
  (4 + x ≡ 2 * 2 [ZMOD 3^3]) →
  (6 + x ≡ 3 * 3 [ZMOD 5^3]) →
  (8 + x ≡ 5 * 5 [ZMOD 7^3]) →
  x % 105 = 3 :=
by
  sorry

end NUMINAMATH_GPT_integer_mod_105_l323_32399


namespace NUMINAMATH_GPT_speech_competition_sequences_l323_32369

theorem speech_competition_sequences
    (contestants : Fin 5 → Prop)
    (girls boys : Fin 5 → Prop)
    (girl_A : Fin 5)
    (not_girl_A_first : ¬contestants 0)
    (no_consecutive_boys : ∀ i, boys i → ¬boys (i + 1))
    (count_girls : ∀ x, girls x → x = girl_A ∨ (contestants x ∧ ¬boys x))
    (count_boys : ∀ x, (boys x) → contestants x)
    (total_count : Fin 5 → Fin 5 → ℕ)
    (correct_answer : total_count = 276) : 
    ∃ seq_count, seq_count = 276 := 
sorry

end NUMINAMATH_GPT_speech_competition_sequences_l323_32369


namespace NUMINAMATH_GPT_range_of_a_l323_32333

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 ≤ 1) (f_def : ∀ x, f x = a * x - x^3)
  (condition : f x2 - f x1 > x2 - x1) :
  a ≥ 4 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l323_32333


namespace NUMINAMATH_GPT_smallest_positive_integer_div_conditions_l323_32395

theorem smallest_positive_integer_div_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3) → x ≤ y :=
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_div_conditions_l323_32395


namespace NUMINAMATH_GPT_Basel_series_l323_32396

theorem Basel_series :
  (∑' (n : ℕ+), 1 / (n : ℝ)^2) = π^2 / 6 := by sorry

end NUMINAMATH_GPT_Basel_series_l323_32396


namespace NUMINAMATH_GPT_horse_distribution_l323_32378

variable (b₁ b₂ b₃ : ℕ) 
variable (a : Matrix (Fin 3) (Fin 3) ℝ)
variable (h1 : a 0 0 > a 0 1 ∧ a 0 0 > a 0 2)
variable (h2 : a 1 1 > a 1 0 ∧ a 1 1 > a 1 2)
variable (h3 : a 2 2 > a 2 0 ∧ a 2 2 > a 2 1)

theorem horse_distribution :
  ∃ n : ℕ, ∀ (b₁ b₂ b₃ : ℕ), min b₁ (min b₂ b₃) > n → 
  ∃ (x1 y1 x2 y2 x3 y3 : ℕ), 3*x1 + y1 = b₁ ∧ 3*x2 + y2 = b₂ ∧ 3*x3 + y3 = b₃ ∧
  y1*a 0 0 > y2*a 0 1 ∧ y1*a 0 0 > y3*a 0 2 ∧
  y2*a 1 1 > y1*a 1 0 ∧ y2*a 1 1 > y3*a 1 2 ∧
  y3*a 2 2 > y1*a 2 0 ∧ y3*a 2 2 > y2*a 2 1 :=
sorry

end NUMINAMATH_GPT_horse_distribution_l323_32378


namespace NUMINAMATH_GPT_only_element_in_intersection_l323_32343

theorem only_element_in_intersection :
  ∃! (n : ℕ), n = 2500 ∧ ∃ (r : ℚ), r ≠ 2 ∧ r ≠ -2 ∧ 404 / (r^2 - 4) = n := sorry

end NUMINAMATH_GPT_only_element_in_intersection_l323_32343


namespace NUMINAMATH_GPT_knight_tour_impossible_49_squares_l323_32334

-- Define the size of the chessboard
def boardSize : ℕ := 7

-- Define the total number of squares on the chessboard
def totalSquares : ℕ := boardSize * boardSize

-- Define the condition for a knight's tour on the 49-square board
def knight_tour_possible (n : ℕ) : Prop :=
  n = totalSquares ∧ 
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 
  -- add condition representing knight's tour and ending
  -- adjacent condition can be mathematically proved here 
  -- but we'll skip here as we asked just to state the problem not the proof.
  sorry -- Placeholder for the precise condition

-- Define the final theorem statement
theorem knight_tour_impossible_49_squares : ¬ knight_tour_possible totalSquares :=
by sorry

end NUMINAMATH_GPT_knight_tour_impossible_49_squares_l323_32334


namespace NUMINAMATH_GPT_raju_working_days_l323_32325

theorem raju_working_days (x : ℕ) 
  (h1: (1 / 10 : ℚ) + 1 / x = 1 / 8) : x = 40 :=
by sorry

end NUMINAMATH_GPT_raju_working_days_l323_32325


namespace NUMINAMATH_GPT_set_B_correct_l323_32345

-- Define the set A
def A : Set ℤ := {-1, 0, 1, 2}

-- Define the set B using the given formula
def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2 * x}

-- State the theorem that B is equal to the given set {-1, 0, 3}
theorem set_B_correct : B = {-1, 0, 3} := 
by 
  sorry

end NUMINAMATH_GPT_set_B_correct_l323_32345
