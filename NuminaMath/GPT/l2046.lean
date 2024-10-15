import Mathlib

namespace NUMINAMATH_GPT_find_value_l2046_204684

theorem find_value : (100 + (20 / 90)) * 90 = 120 := by
  sorry

end NUMINAMATH_GPT_find_value_l2046_204684


namespace NUMINAMATH_GPT_find_m_n_l2046_204628

theorem find_m_n : ∃ (m n : ℕ), m > n ∧ m^3 - n^3 = 999 ∧ ((m = 10 ∧ n = 1) ∨ (m = 12 ∧ n = 9)) :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_l2046_204628


namespace NUMINAMATH_GPT_point_in_plane_region_l2046_204654

theorem point_in_plane_region :
  let P := (0, 0)
  let Q := (2, 4)
  let R := (-1, 4)
  let S := (1, 8)
  (P.1 + P.2 - 1 < 0) ∧ ¬(Q.1 + Q.2 - 1 < 0) ∧ ¬(R.1 + R.2 - 1 < 0) ∧ ¬(S.1 + S.2 - 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_in_plane_region_l2046_204654


namespace NUMINAMATH_GPT_correct_quotient_l2046_204663

-- Define number N based on given conditions
def N : ℕ := 9 * 8 + 6

-- Prove that the correct quotient when N is divided by 6 is 13
theorem correct_quotient : N / 6 = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_correct_quotient_l2046_204663


namespace NUMINAMATH_GPT_seven_points_unit_distance_l2046_204698

theorem seven_points_unit_distance :
  ∃ (A B C D E F G : ℝ × ℝ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
     E ≠ F ∧ E ≠ G ∧
     F ≠ G) ∧
    (∀ (P Q R : ℝ × ℝ),
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F ∨ P = G) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F ∨ Q = G) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E ∨ R = F ∨ R = G) →
      P ≠ Q → P ≠ R → Q ≠ R →
      (dist P Q = 1 ∨ dist P R = 1 ∨ dist Q R = 1)) :=
sorry

end NUMINAMATH_GPT_seven_points_unit_distance_l2046_204698


namespace NUMINAMATH_GPT_legendre_polynomial_expansion_l2046_204608

noncomputable def f (α β γ : ℝ) (θ : ℝ) : ℝ := α + β * Real.cos θ + γ * Real.cos θ ^ 2

noncomputable def P0 (x : ℝ) : ℝ := 1
noncomputable def P1 (x : ℝ) : ℝ := x
noncomputable def P2 (x : ℝ) : ℝ := (3 * x ^ 2 - 1) / 2

theorem legendre_polynomial_expansion (α β γ : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
    f α β γ θ = (α + γ / 3) * P0 (Real.cos θ) + β * P1 (Real.cos θ) + (2 * γ / 3) * P2 (Real.cos θ) := by
  sorry

end NUMINAMATH_GPT_legendre_polynomial_expansion_l2046_204608


namespace NUMINAMATH_GPT_largest_y_coordinate_of_degenerate_ellipse_l2046_204643

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ x y : ℝ, (x^2 / 49 + (y - 3)^2 / 25 = 0) → y ≤ 3 := by
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_of_degenerate_ellipse_l2046_204643


namespace NUMINAMATH_GPT_sodium_bicarbonate_moles_needed_l2046_204659

-- Definitions for the problem.
def balanced_reaction : Prop := 
  ∀ (NaHCO₃ HCl NaCl H₂O CO₂ : Type) (moles_NaHCO₃ moles_HCl moles_NaCl moles_H₂O moles_CO₂ : Nat),
  (moles_NaHCO₃ = moles_HCl) → 
  (moles_NaCl = moles_HCl) → 
  (moles_H₂O = moles_HCl) → 
  (moles_CO₂ = moles_HCl)

-- Given condition: 3 moles of HCl
def moles_HCl : Nat := 3

-- The theorem statement
theorem sodium_bicarbonate_moles_needed : 
  balanced_reaction → moles_HCl = 3 → ∃ moles_NaHCO₃, moles_NaHCO₃ = 3 :=
by 
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_sodium_bicarbonate_moles_needed_l2046_204659


namespace NUMINAMATH_GPT_find_y_l2046_204671

theorem find_y (y : ℕ) (hy_mult_of_7 : ∃ k, y = 7 * k) (hy_pos : 0 < y) (hy_square : y^2 > 225) (hy_upper_bound : y < 30) : y = 21 :=
sorry

end NUMINAMATH_GPT_find_y_l2046_204671


namespace NUMINAMATH_GPT_soda_cost_l2046_204662

theorem soda_cost (S P W : ℝ) (h1 : P = 3 * S) (h2 : W = 3 * P) (h3 : 3 * S + 2 * P + W = 18) : S = 1 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_l2046_204662


namespace NUMINAMATH_GPT_algebraic_expression_value_l2046_204611

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : 2 * a^2 - 4 * a + 2022 = 2024 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2046_204611


namespace NUMINAMATH_GPT_minimum_stamps_l2046_204687

theorem minimum_stamps (c f : ℕ) (h : 3 * c + 4 * f = 50) : c + f = 13 :=
sorry

end NUMINAMATH_GPT_minimum_stamps_l2046_204687


namespace NUMINAMATH_GPT_problem_solution_l2046_204664

theorem problem_solution :
  50000 - ((37500 / 62.35) ^ 2 + Real.sqrt 324) = -311752.222 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2046_204664


namespace NUMINAMATH_GPT_pumpkin_weight_difference_l2046_204647

variable (Brad_weight Jessica_weight Betty_weight : ℕ)

theorem pumpkin_weight_difference :
  Brad_weight = 54 →
  Jessica_weight = Brad_weight / 2 →
  Betty_weight = 4 * Jessica_weight →
  Betty_weight - Jessica_weight = 81 := by
  sorry

end NUMINAMATH_GPT_pumpkin_weight_difference_l2046_204647


namespace NUMINAMATH_GPT_find_a_l2046_204668

theorem find_a (a : ℝ) : (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 12 * x + a = (2 * x + b) ^ 2)) → a = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l2046_204668


namespace NUMINAMATH_GPT_ascending_order_l2046_204695

theorem ascending_order : (3 / 8 : ℝ) < 0.75 ∧ 
                          0.75 < (1 + 2 / 5 : ℝ) ∧ 
                          (1 + 2 / 5 : ℝ) < 1.43 ∧
                          1.43 < (13 / 8 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_ascending_order_l2046_204695


namespace NUMINAMATH_GPT_variance_of_binomial_distribution_l2046_204691

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_binomial_distribution :
  binomial_variance 10 (2/5) = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_binomial_distribution_l2046_204691


namespace NUMINAMATH_GPT_range_of_a_l2046_204622

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t ≥ 1 → f (2 * t - 1) a ≥ 2 * f t a - 3) ↔ a < 2 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l2046_204622


namespace NUMINAMATH_GPT_sum_of_distinct_integers_l2046_204638

theorem sum_of_distinct_integers 
  (p q r s : ℕ) 
  (h1 : p * q = 6) 
  (h2 : r * s = 8) 
  (h3 : p * r = 4) 
  (h4 : q * s = 12) 
  (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) : 
  p + q + r + s = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_l2046_204638


namespace NUMINAMATH_GPT_gauravi_walks_4500m_on_tuesday_l2046_204693

def initial_distance : ℕ := 500
def increase_per_day : ℕ := 500
def target_distance : ℕ := 4500

def distance_after_days (n : ℕ) : ℕ :=
  initial_distance + n * increase_per_day

def day_of_week_after (start_day : ℕ) (n : ℕ) : ℕ :=
  (start_day + n) % 7

def monday : ℕ := 0 -- Represent Monday as 0

theorem gauravi_walks_4500m_on_tuesday :
  distance_after_days 8 = target_distance ∧ day_of_week_after monday 8 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_gauravi_walks_4500m_on_tuesday_l2046_204693


namespace NUMINAMATH_GPT_normal_cost_of_car_wash_l2046_204689

-- Conditions
variables (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180)

-- Theorem to be proved
theorem normal_cost_of_car_wash (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180) : C = 15 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_normal_cost_of_car_wash_l2046_204689


namespace NUMINAMATH_GPT_pants_original_price_l2046_204666

theorem pants_original_price (P : ℝ) (h1 : P * 0.6 = 50.40) : P = 84 :=
sorry

end NUMINAMATH_GPT_pants_original_price_l2046_204666


namespace NUMINAMATH_GPT_determinant_scalar_multiplication_l2046_204609

theorem determinant_scalar_multiplication (x y z w : ℝ) (h : abs (x * w - y * z) = 10) :
  abs (3*x * 3*w - 3*y * 3*z) = 90 :=
by
  sorry

end NUMINAMATH_GPT_determinant_scalar_multiplication_l2046_204609


namespace NUMINAMATH_GPT_two_digit_number_is_24_l2046_204621

-- Defining the two-digit number conditions

variables (x y : ℕ)

noncomputable def condition1 := y = x + 2
noncomputable def condition2 := (10 * x + y) * (x + y) = 144

-- The statement of the proof problem
theorem two_digit_number_is_24 (h1 : condition1 x y) (h2 : condition2 x y) : 10 * x + y = 24 :=
sorry

end NUMINAMATH_GPT_two_digit_number_is_24_l2046_204621


namespace NUMINAMATH_GPT_total_books_l2046_204642

-- Define the number of books each person has
def books_beatrix : ℕ := 30
def books_alannah : ℕ := books_beatrix + 20
def books_queen : ℕ := books_alannah + (books_alannah / 5)

-- State the theorem to be proved
theorem total_books (h_beatrix : books_beatrix = 30)
                    (h_alannah : books_alannah = books_beatrix + 20)
                    (h_queen : books_queen = books_alannah + (books_alannah / 5)) :
  books_alannah + books_beatrix + books_queen = 140 :=
sorry

end NUMINAMATH_GPT_total_books_l2046_204642


namespace NUMINAMATH_GPT_find_f2009_l2046_204665

noncomputable def f : ℝ → ℝ :=
sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (2 + x) = -f (2 - x)
axiom initial_condition : f (-3) = -2

theorem find_f2009 : f 2009 = 2 :=
sorry

end NUMINAMATH_GPT_find_f2009_l2046_204665


namespace NUMINAMATH_GPT_ways_to_make_50_cents_without_dimes_or_quarters_l2046_204639

theorem ways_to_make_50_cents_without_dimes_or_quarters : 
  ∃ (n : ℕ), n = 1024 := 
by
  let num_ways := (2 ^ 10)
  existsi num_ways
  sorry

end NUMINAMATH_GPT_ways_to_make_50_cents_without_dimes_or_quarters_l2046_204639


namespace NUMINAMATH_GPT_monochromatic_triangle_probability_l2046_204617

-- Define the coloring of the edges
inductive Color
| Red : Color
| Blue : Color

-- Define an edge
structure Edge :=
(v1 v2 : Nat)
(color : Color)

-- Define the hexagon with its sides and diagonals
def hexagonEdges : List Edge := [
  -- Sides of the hexagon
  { v1 := 1, v2 := 2, color := sorry }, { v1 := 2, v2 := 3, color := sorry },
  { v1 := 3, v2 := 4, color := sorry }, { v1 := 4, v2 := 5, color := sorry },
  { v1 := 5, v2 := 6, color := sorry }, { v1 := 6, v2 := 1, color := sorry },
  -- Diagonals of the hexagon
  { v1 := 1, v2 := 3, color := sorry }, { v1 := 1, v2 := 4, color := sorry },
  { v1 := 1, v2 := 5, color := sorry }, { v1 := 2, v2 := 4, color := sorry },
  { v1 := 2, v2 := 5, color := sorry }, { v1 := 2, v2 := 6, color := sorry },
  { v1 := 3, v2 := 5, color := sorry }, { v1 := 3, v2 := 6, color := sorry },
  { v1 := 4, v2 := 6, color := sorry }
]

-- Define what a triangle is
structure Triangle :=
(v1 v2 v3 : Nat)

-- List all possible triangles formed by vertices of the hexagon
def hexagonTriangles : List Triangle := [
  { v1 := 1, v2 := 2, v3 := 3 }, { v1 := 1, v2 := 2, v3 := 4 },
  { v1 := 1, v2 := 2, v3 := 5 }, { v1 := 1, v2 := 2, v3 := 6 },
  { v1 := 1, v2 := 3, v3 := 4 }, { v1 := 1, v2 := 3, v3 := 5 },
  { v1 := 1, v2 := 3, v3 := 6 }, { v1 := 1, v2 := 4, v3 := 5 },
  { v1 := 1, v2 := 4, v3 := 6 }, { v1 := 1, v2 := 5, v3 := 6 },
  { v1 := 2, v2 := 3, v3 := 4 }, { v1 := 2, v2 := 3, v3 := 5 },
  { v1 := 2, v2 := 3, v3 := 6 }, { v1 := 2, v2 := 4, v3 := 5 },
  { v1 := 2, v2 := 4, v3 := 6 }, { v1 := 2, v2 := 5, v3 := 6 },
  { v1 := 3, v2 := 4, v3 := 5 }, { v1 := 3, v2 := 4, v3 := 6 },
  { v1 := 3, v2 := 5, v3 := 6 }, { v1 := 4, v2 := 5, v3 := 6 }
]

-- Define the probability calculation, with placeholders for terms that need proving
noncomputable def probabilityMonochromaticTriangle : ℚ :=
  1 - (3 / 4) ^ 20

-- The theorem to prove the probability matches the given answer
theorem monochromatic_triangle_probability :
  probabilityMonochromaticTriangle = 253 / 256 :=
by sorry

end NUMINAMATH_GPT_monochromatic_triangle_probability_l2046_204617


namespace NUMINAMATH_GPT_smaller_circle_radius_is_6_l2046_204612

-- Define the conditions of the problem
def large_circle_radius : ℝ := 2

def smaller_circles_touching_each_other (r : ℝ) : Prop :=
  let oa := large_circle_radius + r
  let ob := large_circle_radius + r
  let ab := 2 * r
  (oa^2 + ob^2 = ab^2)

def problem_statement : Prop :=
  ∃ r : ℝ, smaller_circles_touching_each_other r ∧ r = 6

theorem smaller_circle_radius_is_6 : problem_statement :=
sorry

end NUMINAMATH_GPT_smaller_circle_radius_is_6_l2046_204612


namespace NUMINAMATH_GPT_rehabilitation_centers_total_l2046_204600

noncomputable def jane_visits (han_visits : ℕ) : ℕ := 2 * han_visits + 6
noncomputable def han_visits (jude_visits : ℕ) : ℕ := 2 * jude_visits - 2
noncomputable def jude_visits (lisa_visits : ℕ) : ℕ := lisa_visits / 2
def lisa_visits : ℕ := 6

def total_visits (jane_visits han_visits jude_visits lisa_visits : ℕ) : ℕ :=
  jane_visits + han_visits + jude_visits + lisa_visits

theorem rehabilitation_centers_total :
  total_visits (jane_visits (han_visits (jude_visits lisa_visits))) 
               (han_visits (jude_visits lisa_visits))
               (jude_visits lisa_visits) 
               lisa_visits = 27 :=
by
  sorry

end NUMINAMATH_GPT_rehabilitation_centers_total_l2046_204600


namespace NUMINAMATH_GPT_triangles_congruence_l2046_204603

theorem triangles_congruence (A_1 B_1 C_1 A_2 B_2 C_2 : ℝ)
  (angle_A1 angle_B1 angle_C1 angle_A2 angle_B2 angle_C2 : ℝ)
  (h_side1 : A_1 = A_2) 
  (h_side2 : B_1 = B_2)
  (h_angle1 : angle_A1 = angle_A2)
  (h_angle2 : angle_B1 = angle_B2)
  (h_angle3 : angle_C1 = angle_C2) : 
  ¬((A_1 = C_1) ∧ (B_1 = C_2) ∧ (angle_A1 = angle_B2) ∧ (angle_B1 = angle_A2) ∧ (angle_C1 = angle_B2) → 
     (A_1 = A_2) ∧ (B_1 = B_2) ∧ (C_1 = C_2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_triangles_congruence_l2046_204603


namespace NUMINAMATH_GPT_chess_games_l2046_204606

theorem chess_games (n : ℕ) (total_games : ℕ) (players : ℕ) (games_per_player : ℕ)
  (h1 : players = 9)
  (h2 : total_games = 36)
  (h3 : ∀ i : ℕ, i < players → games_per_player = players - 1)
  (h4 : 2 * total_games = players * games_per_player) :
  games_per_player = 1 :=
by
  rw [h1, h2] at h4
  sorry

end NUMINAMATH_GPT_chess_games_l2046_204606


namespace NUMINAMATH_GPT_first_math_festival_divisibility_largest_ordinal_number_divisibility_l2046_204616

-- Definition of the conditions for part (a)
def first_math_festival_year : ℕ := 1990
def first_ordinal_number : ℕ := 1

-- Statement for part (a)
theorem first_math_festival_divisibility : first_math_festival_year % first_ordinal_number = 0 :=
sorry

-- Definition of the conditions for part (b)
def nth_math_festival_year (N : ℕ) : ℕ := 1989 + N

-- Statement for part (b)
theorem largest_ordinal_number_divisibility : ∀ N : ℕ, 
  (nth_math_festival_year N) % N = 0 → N ≤ 1989 :=
sorry

end NUMINAMATH_GPT_first_math_festival_divisibility_largest_ordinal_number_divisibility_l2046_204616


namespace NUMINAMATH_GPT_valid_reasonings_l2046_204604

-- Define the conditions as hypotheses
def analogical_reasoning (R1 : Prop) : Prop := R1
def inductive_reasoning (R2 R4 : Prop) : Prop := R2 ∧ R4
def invalid_generalization (R3 : Prop) : Prop := ¬R3

-- Given the conditions, prove that the valid reasonings are (1), (2), and (4)
theorem valid_reasonings
  (R1 : Prop) (R2 : Prop) (R3 : Prop) (R4 : Prop)
  (h1 : analogical_reasoning R1) 
  (h2 : inductive_reasoning R2 R4) 
  (h3 : invalid_generalization R3) : 
  R1 ∧ R2 ∧ R4 :=
by 
  sorry

end NUMINAMATH_GPT_valid_reasonings_l2046_204604


namespace NUMINAMATH_GPT_Chloe_total_score_l2046_204677

-- Definitions
def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

-- Statement of the theorem
theorem Chloe_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 81 := by
  sorry

end NUMINAMATH_GPT_Chloe_total_score_l2046_204677


namespace NUMINAMATH_GPT_cos_A_and_sin_2B_minus_A_l2046_204699

variable (A B C a b c : ℝ)
variable (h1 : a * Real.sin A = 4 * b * Real.sin B)
variable (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2))

theorem cos_A_and_sin_2B_minus_A :
  Real.cos A = -Real.sqrt 5 / 5 ∧ Real.sin (2 * B - A) = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_A_and_sin_2B_minus_A_l2046_204699


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_12_l2046_204636

theorem tan_alpha_plus_pi_over_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + π / 6)) :
  Real.tan (α + π / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_12_l2046_204636


namespace NUMINAMATH_GPT_find_third_angle_l2046_204661

variable (A B C : ℝ)

theorem find_third_angle
  (hA : A = 32)
  (hB : B = 3 * A)
  (hC : C = 2 * A - 12) :
  C = 52 := by
  sorry

end NUMINAMATH_GPT_find_third_angle_l2046_204661


namespace NUMINAMATH_GPT_find_real_numbers_l2046_204653

theorem find_real_numbers (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 6) 
  (h3 : x^3 + y^3 + z^3 = 8) :
  (x = 1 ∧ y = 2 ∧ z = -1) ∨ 
  (x = 1 ∧ y = -1 ∧ z = 2) ∨
  (x = 2 ∧ y = 1 ∧ z = -1) ∨ 
  (x = 2 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 2) ∨
  (x = -1 ∧ y = 2 ∧ z = 1) := 
sorry

end NUMINAMATH_GPT_find_real_numbers_l2046_204653


namespace NUMINAMATH_GPT_car_sharing_problem_l2046_204675

theorem car_sharing_problem 
  (x : ℕ)
  (cond1 : ∃ c : ℕ, x = 4 * c + 4)
  (cond2 : ∃ c : ℕ, x = 3 * c + 9):
  (x / 4 + 1 = (x - 9) / 3) :=
by sorry

end NUMINAMATH_GPT_car_sharing_problem_l2046_204675


namespace NUMINAMATH_GPT_factor_expression_l2046_204688

theorem factor_expression (y : ℤ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) :=
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l2046_204688


namespace NUMINAMATH_GPT_intervals_union_l2046_204680

open Set

noncomputable def I (a b : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < b}

theorem intervals_union {I1 I2 I3 : Set ℝ} (h1 : ∃ (a1 b1 : ℝ), I1 = I a1 b1)
  (h2 : ∃ (a2 b2 : ℝ), I2 = I a2 b2) (h3 : ∃ (a3 b3 : ℝ), I3 = I a3 b3)
  (h_non_empty : (I1 ∩ I2 ∩ I3).Nonempty) (h_not_contained : ¬ (I1 ⊆ I2) ∧ ¬ (I1 ⊆ I3) ∧ ¬ (I2 ⊆ I1) ∧ ¬ (I2 ⊆ I3) ∧ ¬ (I3 ⊆ I1) ∧ ¬ (I3 ⊆ I2)) :
  I1 ⊆ (I2 ∪ I3) ∨ I2 ⊆ (I1 ∪ I3) ∨ I3 ⊆ (I1 ∪ I2) :=
sorry

end NUMINAMATH_GPT_intervals_union_l2046_204680


namespace NUMINAMATH_GPT_students_in_class_l2046_204626

theorem students_in_class (N : ℕ) 
  (avg_age_class : ℕ) (avg_age_4 : ℕ) (avg_age_10 : ℕ) (age_15th : ℕ) 
  (total_age_class : ℕ) (total_age_4 : ℕ) (total_age_10 : ℕ)
  (h1 : avg_age_class = 15)
  (h2 : avg_age_4 = 14)
  (h3 : avg_age_10 = 16)
  (h4 : age_15th = 9)
  (h5 : total_age_class = avg_age_class * N)
  (h6 : total_age_4 = 4 * avg_age_4)
  (h7 : total_age_10 = 10 * avg_age_10)
  (h8 : total_age_class = total_age_4 + total_age_10 + age_15th) :
  N = 15 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l2046_204626


namespace NUMINAMATH_GPT_fraction_of_ponies_with_horseshoes_l2046_204613

variable (P H : ℕ)
variable (F : ℚ)

theorem fraction_of_ponies_with_horseshoes 
  (h1 : H = P + 3)
  (h2 : P + H = 163)
  (h3 : (5/8 : ℚ) * F * P = 5) :
  F = 1/10 :=
  sorry

end NUMINAMATH_GPT_fraction_of_ponies_with_horseshoes_l2046_204613


namespace NUMINAMATH_GPT_problem_l2046_204652

theorem problem (x : ℝ) : (x^2 + 2 * x - 3 ≤ 0) → ¬(abs x > 3) :=
by sorry

end NUMINAMATH_GPT_problem_l2046_204652


namespace NUMINAMATH_GPT_unique_positive_integer_solution_l2046_204644

theorem unique_positive_integer_solution (p : ℕ) (hp : Nat.Prime p) (hop : p % 2 = 1) :
  ∃! (x y : ℕ), x^2 + p * x = y^2 ∧ x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_unique_positive_integer_solution_l2046_204644


namespace NUMINAMATH_GPT_BC_work_time_l2046_204692

-- Definitions
def rateA : ℚ := 1 / 4 -- A's rate of work
def rateB : ℚ := 1 / 4 -- B's rate of work
def rateAC : ℚ := 1 / 3 -- A and C's combined rate of work

-- To prove
theorem BC_work_time : 1 / (rateB + (rateAC - rateA)) = 3 := by
  sorry

end NUMINAMATH_GPT_BC_work_time_l2046_204692


namespace NUMINAMATH_GPT_solve_case1_solve_case2_l2046_204669

variables (a b c A B C x y z : ℝ)

-- Define the conditions for the first special case
def conditions_case1 := (A = b + c) ∧ (B = c + a) ∧ (C = a + b)

-- State the proposition to prove for the first special case
theorem solve_case1 (h : conditions_case1 a b c A B C) :
  z = 0 ∧ y = -1 ∧ x = A + b := by
  sorry

-- Define the conditions for the second special case
def conditions_case2 := (A = b * c) ∧ (B = c * a) ∧ (C = a * b)

-- State the proposition to prove for the second special case
theorem solve_case2 (h : conditions_case2 a b c A B C) :
  z = 1 ∧ y = -(a + b + c) ∧ x = a * b * c := by
  sorry

end NUMINAMATH_GPT_solve_case1_solve_case2_l2046_204669


namespace NUMINAMATH_GPT_fifth_dog_is_older_than_fourth_l2046_204660

theorem fifth_dog_is_older_than_fourth :
  ∀ (age_1 age_2 age_3 age_4 age_5 : ℕ),
  (age_1 = 10) →
  (age_2 = age_1 - 2) →
  (age_3 = age_2 + 4) →
  (age_4 = age_3 / 2) →
  (age_5 = age_4 + 20) →
  ((age_1 + age_5) / 2 = 18) →
  (age_5 - age_4 = 20) :=
by
  intros age_1 age_2 age_3 age_4 age_5 h1 h2 h3 h4 h5 h_avg
  sorry

end NUMINAMATH_GPT_fifth_dog_is_older_than_fourth_l2046_204660


namespace NUMINAMATH_GPT_system_solution_l2046_204681

theorem system_solution (x y : ℝ) 
  (h1 : 0 < x + y) 
  (h2 : x + y ≠ 1) 
  (h3 : 2 * x - y ≠ 0)
  (eq1 : (x + y) * 2^(y - 2 * x) = 6.25) 
  (eq2 : (x + y) * (1 / (2 * x - y)) = 5) :
x = 9 ∧ y = 16 := 
sorry

end NUMINAMATH_GPT_system_solution_l2046_204681


namespace NUMINAMATH_GPT_min_value_of_f_at_sqrt2_l2046_204615

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x + (1 / x)))

theorem min_value_of_f_at_sqrt2 :
  f (Real.sqrt 2) = (11 * Real.sqrt 2) / 6 :=
sorry

end NUMINAMATH_GPT_min_value_of_f_at_sqrt2_l2046_204615


namespace NUMINAMATH_GPT_triangle_area_l2046_204670

theorem triangle_area (area_WXYZ : ℝ) (side_small_squares : ℝ) 
  (AB_eq_AC : (AB = AC)) (A_on_center : (A = O)) :
  area_WXYZ = 64 ∧ side_small_squares = 2 →
  ∃ (area_triangle_ABC : ℝ), area_triangle_ABC = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_triangle_area_l2046_204670


namespace NUMINAMATH_GPT_eval_expression_l2046_204696

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l2046_204696


namespace NUMINAMATH_GPT_total_balloons_after_gift_l2046_204602

-- Definitions for conditions
def initial_balloons := 26
def additional_balloons := 34

-- Proposition for the total number of balloons
theorem total_balloons_after_gift : initial_balloons + additional_balloons = 60 := 
by
  -- Proof omitted, adding sorry
  sorry

end NUMINAMATH_GPT_total_balloons_after_gift_l2046_204602


namespace NUMINAMATH_GPT_cos_inequality_for_triangle_l2046_204685

theorem cos_inequality_for_triangle (A B C : ℝ) (h : A + B + C = π) :
  (1 / 3) * (Real.cos A + Real.cos B + Real.cos C) ≤ (1 / 2) ∧
  (1 / 2) ≤ Real.sqrt ((1 / 3) * (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2)) :=
by
  sorry

end NUMINAMATH_GPT_cos_inequality_for_triangle_l2046_204685


namespace NUMINAMATH_GPT_small_stick_length_l2046_204650

theorem small_stick_length 
  (x : ℝ) 
  (hx1 : 3 < x) 
  (hx2 : x < 9) 
  (hx3 : 3 + 6 > x) : 
  x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_small_stick_length_l2046_204650


namespace NUMINAMATH_GPT_problem_1_problem_2_l2046_204646

-- Problem 1 proof statement
theorem problem_1 (x : ℝ) (h : x = -1) : 
  (1 * (-x^2 + 5 * x) - (x - 3) - 4 * x) = 2 := by
  -- Placeholder for the proof
  sorry

-- Problem 2 proof statement
theorem problem_2 (m n : ℝ) (h_m : m = -1/2) (h_n : n = 1/3) : 
  (5 * (3 * m^2 * n - m * n^2) - (m * n^2 + 3 * m^2 * n)) = 4/3 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2046_204646


namespace NUMINAMATH_GPT_contrapositive_example_contrapositive_proof_l2046_204656

theorem contrapositive_example (x : ℝ) (h : x > 1) : x^2 > 1 := 
sorry

theorem contrapositive_proof (x : ℝ) (h : x^2 ≤ 1) : x ≤ 1 :=
sorry

end NUMINAMATH_GPT_contrapositive_example_contrapositive_proof_l2046_204656


namespace NUMINAMATH_GPT_trapezoid_area_l2046_204641

theorem trapezoid_area 
  (area_ABE area_ADE : ℝ)
  (DE BE : ℝ)
  (h1 : area_ABE = 40)
  (h2 : area_ADE = 30)
  (h3 : DE = 2 * BE) : 
  area_ABE + area_ADE + area_ADE + 4 * area_ABE = 260 :=
by
  -- sorry admits the goal without providing the actual proof
  sorry

end NUMINAMATH_GPT_trapezoid_area_l2046_204641


namespace NUMINAMATH_GPT_price_of_each_brownie_l2046_204683

variable (B : ℝ)

theorem price_of_each_brownie (h : 4 * B + 10 + 28 = 50) : B = 3 := by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_price_of_each_brownie_l2046_204683


namespace NUMINAMATH_GPT_possible_case_l2046_204649

-- Define the logical propositions P and Q
variables (P Q : Prop)

-- State the conditions given in the problem
axiom h1 : P ∨ Q     -- P ∨ Q is true
axiom h2 : ¬ (P ∧ Q) -- P ∧ Q is false

-- Formulate the proof problem in Lean
theorem possible_case : P ∧ ¬Q :=
by
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_possible_case_l2046_204649


namespace NUMINAMATH_GPT_cost_of_soccer_ball_l2046_204618

theorem cost_of_soccer_ball
  (F S : ℝ)
  (h1 : 3 * F + S = 155)
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 :=
sorry

end NUMINAMATH_GPT_cost_of_soccer_ball_l2046_204618


namespace NUMINAMATH_GPT_remainder_problem_l2046_204679

theorem remainder_problem : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_problem_l2046_204679


namespace NUMINAMATH_GPT_total_earnings_l2046_204627

theorem total_earnings :
  (15 * 2) + (12 * 1.5) = 48 := by
  sorry

end NUMINAMATH_GPT_total_earnings_l2046_204627


namespace NUMINAMATH_GPT_monotonicity_x_pow_2_over_3_l2046_204674

noncomputable def x_pow_2_over_3 (x : ℝ) : ℝ := x^(2/3)

theorem monotonicity_x_pow_2_over_3 : ∀ x y : ℝ, 0 < x → x < y → x_pow_2_over_3 x < x_pow_2_over_3 y :=
by
  intros x y hx hxy
  sorry

end NUMINAMATH_GPT_monotonicity_x_pow_2_over_3_l2046_204674


namespace NUMINAMATH_GPT_businessmen_neither_coffee_nor_tea_l2046_204678

/-- Definitions of conditions -/
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def both_drinkers : ℕ := 6

/-- Statement of the problem -/
theorem businessmen_neither_coffee_nor_tea : 
  (total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end NUMINAMATH_GPT_businessmen_neither_coffee_nor_tea_l2046_204678


namespace NUMINAMATH_GPT_fish_disappeared_l2046_204655

theorem fish_disappeared (g : ℕ) (c : ℕ) (left : ℕ) (disappeared : ℕ) (h₁ : g = 7) (h₂ : c = 12) (h₃ : left = 15) (h₄ : g + c - left = disappeared) : disappeared = 4 :=
by
  sorry

end NUMINAMATH_GPT_fish_disappeared_l2046_204655


namespace NUMINAMATH_GPT_gcd_8251_6105_l2046_204645

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 :=
by
  sorry

end NUMINAMATH_GPT_gcd_8251_6105_l2046_204645


namespace NUMINAMATH_GPT_area_fraction_of_square_hole_l2046_204657

theorem area_fraction_of_square_hole (A B C M N : ℝ)
  (h1 : B = C)
  (h2 : M = 0.5 * A)
  (h3 : N = 0.5 * A) :
  (M * N) / (B * C) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_fraction_of_square_hole_l2046_204657


namespace NUMINAMATH_GPT_sum_of_decimals_l2046_204637

-- Defining the specific decimal values as constants
def x : ℝ := 5.47
def y : ℝ := 4.26

-- Noncomputable version for addition to allow Lean to handle real number operations safely
noncomputable def sum : ℝ := x + y

-- Theorem statement asserting the sum of x and y
theorem sum_of_decimals : sum = 9.73 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l2046_204637


namespace NUMINAMATH_GPT_winning_candidate_percentage_l2046_204658

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) (h1 : votes1 = 1256) (h2 : votes2 = 7636) (h3 : votes3 = 11628) 
    : (votes3 : ℝ) / (votes1 + votes2 + votes3) * 100 = 56.67 := by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l2046_204658


namespace NUMINAMATH_GPT_problem_l2046_204624

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x >= 0 then Real.log x / Real.log 3 + m else 1 / 2017

theorem problem (m := -2) (h_root : f 3 m = 0):
  f (f 6 m - 2) m = 1 / 2017 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2046_204624


namespace NUMINAMATH_GPT_unique_geometric_sequence_l2046_204690

theorem unique_geometric_sequence (a : ℝ) (q : ℝ) (a_n b_n : ℕ → ℝ) 
    (h1 : a > 0) 
    (h2 : a_n 1 = a) 
    (h3 : b_n 1 - a_n 1 = 1) 
    (h4 : b_n 2 - a_n 2 = 2) 
    (h5 : b_n 3 - a_n 3 = 3) 
    (h6 : ∀ n, a_n (n + 1) = a_n n * q) 
    (h7 : ∀ n, b_n (n + 1) = b_n n * q) : 
    a = 1 / 3 := sorry

end NUMINAMATH_GPT_unique_geometric_sequence_l2046_204690


namespace NUMINAMATH_GPT_production_equation_l2046_204629

-- Define the conditions as per the problem
variables (workers : ℕ) (x : ℕ) 

-- The number of total workers is fixed
def total_workers := 44

-- Production rates per worker
def bodies_per_worker := 50
def bottoms_per_worker := 120

-- The problem statement as a Lean theorem
theorem production_equation (h : workers = total_workers) (hx : x ≤ workers) :
  2 * bottoms_per_worker * (total_workers - x) = bodies_per_worker * x :=
by
  sorry

end NUMINAMATH_GPT_production_equation_l2046_204629


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2046_204619

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  (x : ℝ) -> x^2 + m * x + 1 = 0 → (m < -2 ∨ m > 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2046_204619


namespace NUMINAMATH_GPT_ordering_of_a_b_c_l2046_204634

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 4 / 4

-- We need to prove that the ordering is a > b > c.

theorem ordering_of_a_b_c : a > b ∧ b > c :=
by 
  sorry

end NUMINAMATH_GPT_ordering_of_a_b_c_l2046_204634


namespace NUMINAMATH_GPT_best_trip_representation_l2046_204623

structure TripConditions where
  initial_walk_moderate : Prop
  main_road_speed_up : Prop
  bird_watching : Prop
  return_same_route : Prop
  coffee_stop : Prop
  final_walk_moderate : Prop

theorem best_trip_representation (conds : TripConditions) : 
  conds.initial_walk_moderate →
  conds.main_road_speed_up →
  conds.bird_watching →
  conds.return_same_route →
  conds.coffee_stop →
  conds.final_walk_moderate →
  True := 
by 
  intros 
  exact True.intro

end NUMINAMATH_GPT_best_trip_representation_l2046_204623


namespace NUMINAMATH_GPT_current_selling_price_is_correct_profit_per_unit_is_correct_l2046_204605

variable (a : ℝ)

def original_selling_price (a : ℝ) : ℝ :=
  a * 1.22

def current_selling_price (a : ℝ) : ℝ :=
  original_selling_price a * 0.85

def profit_per_unit (a : ℝ) : ℝ :=
  current_selling_price a - a

theorem current_selling_price_is_correct : current_selling_price a = 1.037 * a :=
by
  unfold current_selling_price original_selling_price
  sorry

theorem profit_per_unit_is_correct : profit_per_unit a = 0.037 * a :=
by
  unfold profit_per_unit current_selling_price original_selling_price
  sorry

end NUMINAMATH_GPT_current_selling_price_is_correct_profit_per_unit_is_correct_l2046_204605


namespace NUMINAMATH_GPT_zack_group_size_l2046_204640

theorem zack_group_size (total_students : Nat) (groups : Nat) (group_size : Nat)
  (H1 : total_students = 70)
  (H2 : groups = 7)
  (H3 : total_students = group_size * groups) :
  group_size = 10 := by
  sorry

end NUMINAMATH_GPT_zack_group_size_l2046_204640


namespace NUMINAMATH_GPT_compute_expression_l2046_204620

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end NUMINAMATH_GPT_compute_expression_l2046_204620


namespace NUMINAMATH_GPT_number_of_8_digit_increasing_integers_mod_1000_l2046_204697

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_8_digit_increasing_integers_mod_1000 :
  let M := choose 9 8
  M % 1000 = 9 :=
by
  let M := choose 9 8
  show M % 1000 = 9
  sorry

end NUMINAMATH_GPT_number_of_8_digit_increasing_integers_mod_1000_l2046_204697


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l2046_204633

theorem ratio_of_larger_to_smaller (x y : ℝ) (h_pos : 0 < y) (h_ineq : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l2046_204633


namespace NUMINAMATH_GPT_minimize_sum_of_squares_l2046_204632

noncomputable def sum_of_squares (x : ℝ) : ℝ := x^2 + (18 - x)^2

theorem minimize_sum_of_squares : ∃ x : ℝ, x = 9 ∧ (18 - x) = 9 ∧ ∀ y : ℝ, sum_of_squares y ≥ sum_of_squares 9 :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_l2046_204632


namespace NUMINAMATH_GPT_banknotes_sum_divisible_by_101_l2046_204635

theorem banknotes_sum_divisible_by_101 (a b : ℕ) (h₀ : a ≠ b % 101) : 
  ∃ (m n : ℕ), m + n = 100 ∧ ∃ k l : ℕ, k ≤ m ∧ l ≤ n ∧ (k * a + l * b) % 101 = 0 :=
sorry

end NUMINAMATH_GPT_banknotes_sum_divisible_by_101_l2046_204635


namespace NUMINAMATH_GPT_exists_nat_expressed_as_sum_of_powers_l2046_204607

theorem exists_nat_expressed_as_sum_of_powers 
  (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : ℕ, (∀ p ∈ P, ∃ a b : ℕ, x = a^p + b^p) ∧ (∀ p : ℕ, Nat.Prime p → p ∉ P → ¬∃ a b : ℕ, x = a^p + b^p) :=
by
  let x := 2^(P.val.prod + 1)
  use x
  sorry

end NUMINAMATH_GPT_exists_nat_expressed_as_sum_of_powers_l2046_204607


namespace NUMINAMATH_GPT_polygon_area_correct_l2046_204651

def AreaOfPolygon : Real := 37.5

def polygonVertices : List (Real × Real) :=
  [(0, 0), (5, 0), (5, 5), (0, 5), (5, 10), (0, 10), (0, 0)]

theorem polygon_area_correct :
  (∃ (A : Real) (verts : List (Real × Real)),
    verts = polygonVertices ∧ A = AreaOfPolygon ∧ 
    A = 37.5) := by
  sorry

end NUMINAMATH_GPT_polygon_area_correct_l2046_204651


namespace NUMINAMATH_GPT_fraction_relation_l2046_204614

theorem fraction_relation (a b : ℝ) (h : a / b = 2 / 3) : (a - b) / b = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_relation_l2046_204614


namespace NUMINAMATH_GPT_selling_price_40_percent_profit_l2046_204601

variable (C L : ℝ)

-- Condition: the profit earned by selling at $832 is equal to the loss incurred when selling at some price "L".
axiom eq_profit_loss : 832 - C = C - L

-- Condition: the desired profit price for a 40% profit on the cost price is $896.
axiom forty_percent_profit : 1.40 * C = 896

-- Theorem: the selling price for making a 40% profit is $896.
theorem selling_price_40_percent_profit : 1.40 * C = 896 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_40_percent_profit_l2046_204601


namespace NUMINAMATH_GPT_fraction_of_weight_kept_l2046_204648

-- Definitions of the conditions
def hunting_trips_per_month := 6
def months_in_season := 3
def deers_per_trip := 2
def weight_per_deer := 600
def weight_kept_per_year := 10800

-- Definition calculating total weight caught in the hunting season
def total_trips := hunting_trips_per_month * months_in_season
def weight_per_trip := deers_per_trip * weight_per_deer
def total_weight_caught := total_trips * weight_per_trip

-- The theorem to prove the fraction
theorem fraction_of_weight_kept : (weight_kept_per_year : ℚ) / (total_weight_caught : ℚ) = 1 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_of_weight_kept_l2046_204648


namespace NUMINAMATH_GPT_find_k_l2046_204630

theorem find_k (k : ℤ)
  (h : ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ ∀ x, ((k^2 - 1) * x^2 - 3 * (3 * k - 1) * x + 18 = 0) ↔ (x = x₁ ∨ x = x₂)
       ∧ x₁ > 0 ∧ x₂ > 0) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2046_204630


namespace NUMINAMATH_GPT_fraction_zero_numerator_l2046_204631

theorem fraction_zero_numerator (x : ℝ) (h₁ : (x^2 - 9) / (x + 3) = 0) (h₂ : x + 3 ≠ 0) : x = 3 :=
sorry

end NUMINAMATH_GPT_fraction_zero_numerator_l2046_204631


namespace NUMINAMATH_GPT_benny_birthday_money_l2046_204667

def money_spent_on_gear : ℕ := 34
def money_left_over : ℕ := 33

theorem benny_birthday_money : money_spent_on_gear + money_left_over = 67 :=
by
  sorry

end NUMINAMATH_GPT_benny_birthday_money_l2046_204667


namespace NUMINAMATH_GPT_woodworker_tables_l2046_204610

theorem woodworker_tables (L C_leg C T_leg : ℕ) (hL : L = 40) (hC_leg : C_leg = 4) (hC : C = 6) (hT_leg : T_leg = 4) :
  T = (L - C * C_leg) / T_leg := by
  sorry

end NUMINAMATH_GPT_woodworker_tables_l2046_204610


namespace NUMINAMATH_GPT_exchange_rate_decrease_l2046_204625

theorem exchange_rate_decrease
  (x y z : ℝ)
  (hx : 0 < |x| ∧ |x| < 1)
  (hy : 0 < |y| ∧ |y| < 1)
  (hz : 0 < |z| ∧ |z| < 1)
  (h_eq : (1 + x) * (1 + y) * (1 + z) = (1 - x) * (1 - y) * (1 - z)) :
  (1 - x^2) * (1 - y^2) * (1 - z^2) < 1 :=
by
  sorry

end NUMINAMATH_GPT_exchange_rate_decrease_l2046_204625


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2046_204682

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h : x > 1) : x > 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2046_204682


namespace NUMINAMATH_GPT_son_age_is_14_l2046_204672

-- Definition of Sandra's age and the condition about the ages 3 years ago.
def Sandra_age : ℕ := 36
def son_age_3_years_ago (son_age_now : ℕ) : ℕ := son_age_now - 3 
def Sandra_age_3_years_ago := 36 - 3
def condition_3_years_ago (son_age_now : ℕ) : Prop := Sandra_age_3_years_ago = 3 * (son_age_3_years_ago son_age_now)

-- The goal: proving Sandra's son's age is 14
theorem son_age_is_14 (son_age_now : ℕ) (h : condition_3_years_ago son_age_now) : son_age_now = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_son_age_is_14_l2046_204672


namespace NUMINAMATH_GPT_snack_cost_inequality_l2046_204676

variables (S : ℝ)

def cost_water : ℝ := 0.50
def cost_fruit : ℝ := 0.25
def bundle_price : ℝ := 4.60
def special_price : ℝ := 2.00

theorem snack_cost_inequality (h : bundle_price = 4.60 ∧ special_price = 2.00 ∧
  cost_water = 0.50 ∧ cost_fruit = 0.25) : S < 15.40 / 16 := sorry

end NUMINAMATH_GPT_snack_cost_inequality_l2046_204676


namespace NUMINAMATH_GPT_first_train_speed_l2046_204686

noncomputable def speed_of_first_train (length_train1 : ℕ) (speed_train2 : ℕ) (length_train2 : ℕ) (time_cross : ℕ) : ℕ :=
  let relative_speed_m_s := (500 : ℕ) / time_cross
  let relative_speed_km_h := relative_speed_m_s * 18 / 5
  relative_speed_km_h - speed_train2

theorem first_train_speed :
  speed_of_first_train 270 80 230 9 = 920 := by
  sorry

end NUMINAMATH_GPT_first_train_speed_l2046_204686


namespace NUMINAMATH_GPT_geometric_sequence_fraction_l2046_204673

open Classical

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_fraction {a : ℕ → ℝ} {q : ℝ}
  (h₀ : ∀ n, 0 < a n)
  (h₁ : geometric_seq a q)
  (h₂ : 2 * (1 / 2 * a 2) = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fraction_l2046_204673


namespace NUMINAMATH_GPT_sequence_general_term_l2046_204694

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2046_204694
