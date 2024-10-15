import Mathlib

namespace NUMINAMATH_GPT_trig_identity_l1712_171247

theorem trig_identity (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 2 / 3) : 
  Real.cos (2 * α + Real.pi / 3) = -1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1712_171247


namespace NUMINAMATH_GPT_find_distance_BC_l1712_171230

variables {d_AB d_AC d_BC : ℝ}

theorem find_distance_BC
  (h1 : d_AB = d_AC + d_BC - 200)
  (h2 : d_AC = d_AB + d_BC - 300) :
  d_BC = 250 := 
sorry

end NUMINAMATH_GPT_find_distance_BC_l1712_171230


namespace NUMINAMATH_GPT_part1_part2_l1712_171236

-- Part (1)
theorem part1 (a : ℝ) (A B : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hB : B = { x : ℝ | x^2 - a * x + a - 1 = 0 }) 
  (hUnion : A ∪ B = A) : 
  a = 2 ∨ a = 3 := 
sorry

-- Part (2)
theorem part2 (m : ℝ) (A C : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hC : C = { x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 5 = 0 }) 
  (hInter : A ∩ C = C) : 
  m ∈ Set.Iic (-3) := 
sorry

end NUMINAMATH_GPT_part1_part2_l1712_171236


namespace NUMINAMATH_GPT_relative_error_approximation_l1712_171272

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  (1 / (1 + y) - (1 - y)) / (1 / (1 + y)) = y^2 :=
by
  sorry

end NUMINAMATH_GPT_relative_error_approximation_l1712_171272


namespace NUMINAMATH_GPT_find_k_values_l1712_171261

theorem find_k_values :
    ∀ (k : ℚ),
    (∀ (a b : ℚ), (5 * a^2 + 7 * a + k = 0) ∧ (5 * b^2 + 7 * b + k = 0) ∧ |a - b| = a^2 + b^2 → k = 21 / 25 ∨ k = -21 / 25) :=
by
  sorry

end NUMINAMATH_GPT_find_k_values_l1712_171261


namespace NUMINAMATH_GPT_fraction_arithmetic_l1712_171297

theorem fraction_arithmetic :
  (3 / 4) / (5 / 8) + (1 / 8) = 53 / 40 :=
by
  sorry

end NUMINAMATH_GPT_fraction_arithmetic_l1712_171297


namespace NUMINAMATH_GPT_regular_polygon_perimeter_is_28_l1712_171266

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_is_28_l1712_171266


namespace NUMINAMATH_GPT_subtracted_value_l1712_171253

theorem subtracted_value (N V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 :=
by
  sorry

end NUMINAMATH_GPT_subtracted_value_l1712_171253


namespace NUMINAMATH_GPT_field_trip_classrooms_count_l1712_171262

variable (students : ℕ) (seats_per_bus : ℕ) (number_of_buses : ℕ) (total_classrooms : ℕ)

def fieldTrip 
    (students := 58)
    (seats_per_bus := 2)
    (number_of_buses := 29)
    (total_classrooms := 2) : Prop :=
  students = seats_per_bus * number_of_buses  ∧ total_classrooms = students / (students / total_classrooms)

theorem field_trip_classrooms_count : fieldTrip := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_field_trip_classrooms_count_l1712_171262


namespace NUMINAMATH_GPT_part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l1712_171204

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem part1_f0_f1 : f 0 + f 1 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg1_f2 : f (-1) + f 2 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg2_f3 : f (-2) + f 3 = Real.sqrt 3 / 3 := sorry

theorem part2_conjecture (x1 x2 : ℝ) (h : x1 + x2 = 1) : f x1 + f x2 = Real.sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l1712_171204


namespace NUMINAMATH_GPT_target_hit_prob_l1712_171258

-- Probability definitions for A, B, and C
def prob_A := 1 / 2
def prob_B := 1 / 3
def prob_C := 1 / 4

-- Theorem to prove the probability of the target being hit
theorem target_hit_prob :
  (1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_target_hit_prob_l1712_171258


namespace NUMINAMATH_GPT_average_rate_of_change_l1712_171227

variable {α : Type*} [LinearOrderedField α]
variable (f : α → α)
variable (x x₁ : α)
variable (h₁ : x ≠ x₁)

theorem average_rate_of_change : 
  (f x₁ - f x) / (x₁ - x) = (f x₁ - f x) / (x₁ - x) :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_l1712_171227


namespace NUMINAMATH_GPT_find_min_value_l1712_171286

noncomputable def min_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem find_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (cond : 2/x + 3/y + 5/z = 10) : min_value x y z = 390625 / 1296 :=
sorry

end NUMINAMATH_GPT_find_min_value_l1712_171286


namespace NUMINAMATH_GPT_area_relationship_l1712_171275

theorem area_relationship (P Q R : ℝ) (h_square : 10 * 10 = 100)
  (h_triangle1 : P + R = 50)
  (h_triangle2 : Q + R = 50) :
  P - Q = 0 :=
by
  sorry

end NUMINAMATH_GPT_area_relationship_l1712_171275


namespace NUMINAMATH_GPT_solve_digits_l1712_171267

theorem solve_digits : ∃ A B C : ℕ, (A = 1 ∧ B = 0 ∧ (C = 9 ∨ C = 1)) ∧ 
  (∃ (X : ℕ), X ≥ 2 ∧ (C = X - 1 ∨ C = 1)) ∧ 
  (A * 1000 + B * 100 + B * 10 + C) * (C * 100 + C * 10 + A) = C * 100000 + C * 10000 + C * 1000 + C * 100 + A * 10 + C :=
by sorry

end NUMINAMATH_GPT_solve_digits_l1712_171267


namespace NUMINAMATH_GPT_focus_of_parabola_l1712_171213

theorem focus_of_parabola : 
  (∃ p : ℝ, y^2 = 4 * p * x ∧ p = 1 ∧ ∃ c : ℝ × ℝ, c = (1, 0)) :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_l1712_171213


namespace NUMINAMATH_GPT_percentage_reduction_in_women_l1712_171294

theorem percentage_reduction_in_women
    (total_people : Nat) (men_in_office : Nat) (women_in_office : Nat)
    (men_in_meeting : Nat) (women_in_meeting : Nat)
    (even_men_women : men_in_office = women_in_office)
    (total_people_condition : total_people = men_in_office + women_in_office)
    (meeting_condition : total_people = 60)
    (men_meeting_condition : men_in_meeting = 4)
    (women_meeting_condition : women_in_meeting = 6) :
    ((women_in_meeting * 100) / women_in_office) = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_in_women_l1712_171294


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l1712_171255

variable (V_b V_s t_up t_down : ℝ)

theorem speed_of_boat_in_still_water (h1 : t_up = 2 * t_down)
  (h2 : V_s = 18) 
  (h3 : ∀ d : ℝ, d = (V_b - V_s) * t_up ∧ d = (V_b + V_s) * t_down) : V_b = 54 :=
sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l1712_171255


namespace NUMINAMATH_GPT_total_pints_l1712_171292

variables (Annie Kathryn Ben Sam : ℕ)

-- Conditions
def condition1 := Annie = 16
def condition2 (Annie : ℕ) := Kathryn = 2 * Annie + 2
def condition3 (Kathryn : ℕ) := Ben = Kathryn / 2 - 3
def condition4 (Ben Kathryn : ℕ) := Sam = 2 * (Ben + Kathryn) / 3

-- Statement to prove
theorem total_pints (Annie Kathryn Ben Sam : ℕ) 
  (h1 : condition1 Annie) 
  (h2 : condition2 Annie Kathryn) 
  (h3 : condition3 Kathryn Ben) 
  (h4 : condition4 Ben Kathryn Sam) : 
  Annie + Kathryn + Ben + Sam = 96 :=
sorry

end NUMINAMATH_GPT_total_pints_l1712_171292


namespace NUMINAMATH_GPT_product_of_possible_values_l1712_171260

theorem product_of_possible_values (b : ℝ) (side_length : ℝ) (square_condition : (b - 2) = side_length ∨ (2 - b) = side_length) : 
  (b = -3 ∨ b = 7) → (-3 * 7 = -21) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_product_of_possible_values_l1712_171260


namespace NUMINAMATH_GPT_age_discrepancy_l1712_171243

theorem age_discrepancy (R G M F A : ℕ)
  (hR : R = 12)
  (hG : G = 7 * R)
  (hM : M = G / 2)
  (hF : F = M + 5)
  (hA : A = G - 8)
  (hDiff : A - F = 10) :
  false :=
by
  -- proofs and calculations leading to contradiction go here
  sorry

end NUMINAMATH_GPT_age_discrepancy_l1712_171243


namespace NUMINAMATH_GPT_point_A_in_third_quadrant_l1712_171222

-- Defining the point A with its coordinates
structure Point :=
  (x : Int)
  (y : Int)

def A : Point := ⟨-1, -3⟩

-- The definition of quadrants in Cartesian coordinate system
def quadrant (p : Point) : String :=
  if p.x > 0 ∧ p.y > 0 then "first"
  else if p.x < 0 ∧ p.y > 0 then "second"
  else if p.x < 0 ∧ p.y < 0 then "third"
  else if p.x > 0 ∧ p.y < 0 then "fourth"
  else "boundary"

-- The theorem we want to prove
theorem point_A_in_third_quadrant : quadrant A = "third" :=
by 
  sorry

end NUMINAMATH_GPT_point_A_in_third_quadrant_l1712_171222


namespace NUMINAMATH_GPT_pieces_equality_l1712_171274

-- Define the pieces of chocolate and their areas.
def piece1_area : ℝ := 6 -- Area of triangle EBC
def piece2_area : ℝ := 6 -- Area of triangle AEC
def piece3_area : ℝ := 6 -- Area of polygon AHGFD
def piece4_area : ℝ := 6 -- Area of polygon CFGH

-- State the problem: proving the equality of the areas.
theorem pieces_equality : piece1_area = piece2_area ∧ piece2_area = piece3_area ∧ piece3_area = piece4_area :=
by
  sorry

end NUMINAMATH_GPT_pieces_equality_l1712_171274


namespace NUMINAMATH_GPT_roots_cubic_polynomial_l1712_171246

theorem roots_cubic_polynomial (a b c : ℝ) 
  (h1 : a^3 - 2*a - 2 = 0) 
  (h2 : b^3 - 2*b - 2 = 0) 
  (h3 : c^3 - 2*c - 2 = 0) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -18 :=
by
  sorry

end NUMINAMATH_GPT_roots_cubic_polynomial_l1712_171246


namespace NUMINAMATH_GPT_buying_beams_l1712_171283

theorem buying_beams (x : ℕ) (h : 3 * (x - 1) * x = 6210) :
  3 * (x - 1) * x = 6210 :=
by {
  sorry
}

end NUMINAMATH_GPT_buying_beams_l1712_171283


namespace NUMINAMATH_GPT_boaster_guarantee_distinct_balls_l1712_171208

noncomputable def canGuaranteeDistinctBallCounts (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → boxes i ≠ boxes j

theorem boaster_guarantee_distinct_balls :
  ∃ (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)),
  canGuaranteeDistinctBallCounts boxes pairs :=
sorry

end NUMINAMATH_GPT_boaster_guarantee_distinct_balls_l1712_171208


namespace NUMINAMATH_GPT_plums_total_correct_l1712_171242

-- Define the number of plums picked by Melanie, Dan, and Sally
def plums_melanie : ℕ := 4
def plums_dan : ℕ := 9
def plums_sally : ℕ := 3

-- Define the total number of plums picked
def total_plums : ℕ := plums_melanie + plums_dan + plums_sally

-- Theorem stating the total number of plums picked
theorem plums_total_correct : total_plums = 16 := by
  sorry

end NUMINAMATH_GPT_plums_total_correct_l1712_171242


namespace NUMINAMATH_GPT_area_of_square_field_l1712_171217

-- Define side length
def sideLength : ℕ := 14

-- Define the area function for a square
def area_of_square (side : ℕ) : ℕ := side * side

-- Prove that the area of the square with side length 14 meters is 196 square meters
theorem area_of_square_field : area_of_square sideLength = 196 := by
  sorry

end NUMINAMATH_GPT_area_of_square_field_l1712_171217


namespace NUMINAMATH_GPT_congruence_a_b_mod_1008_l1712_171284

theorem congruence_a_b_mod_1008
  (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : a ^ b - b ^ a = 1008) : a ≡ b [MOD 1008] :=
sorry

end NUMINAMATH_GPT_congruence_a_b_mod_1008_l1712_171284


namespace NUMINAMATH_GPT_baking_completion_time_l1712_171229

theorem baking_completion_time (start_time : ℕ) (partial_bake_time : ℕ) (fraction_baked : ℕ) :
  start_time = 9 → partial_bake_time = 3 → fraction_baked = 4 →
  (start_time + (partial_bake_time * fraction_baked)) = 21 :=
by
  intros h_start h_partial h_fraction
  sorry

end NUMINAMATH_GPT_baking_completion_time_l1712_171229


namespace NUMINAMATH_GPT_find_m_in_hyperbola_l1712_171264

-- Define the problem in Lean 4
theorem find_m_in_hyperbola (m : ℝ) (x y : ℝ) (e : ℝ) (a_sq : ℝ := 9) (h_eq : e = 2) (h_hyperbola : x^2 / a_sq - y^2 / m = 1) : m = 27 :=
sorry

end NUMINAMATH_GPT_find_m_in_hyperbola_l1712_171264


namespace NUMINAMATH_GPT_set_intersection_l1712_171221

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

noncomputable def complement_U_A := U \ A
noncomputable def intersection := B ∩ complement_U_A

theorem set_intersection :
  intersection = ({3, 4} : Set ℕ) := by
  sorry

end NUMINAMATH_GPT_set_intersection_l1712_171221


namespace NUMINAMATH_GPT_sum_three_circles_l1712_171215

theorem sum_three_circles (a b : ℚ) 
  (h1 : 5 * a + 2 * b = 27)
  (h2 : 2 * a + 5 * b = 29) :
  3 * b = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_three_circles_l1712_171215


namespace NUMINAMATH_GPT_solve_equation_l1712_171201

theorem solve_equation : ∀ x : ℝ, (2 * x - 8 = 0) ↔ (x = 4) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1712_171201


namespace NUMINAMATH_GPT_simple_interest_amount_l1712_171224

noncomputable def simple_interest (P r t : ℝ) : ℝ := (P * r * t) / 100
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r / 100)^t - P

theorem simple_interest_amount:
  ∀ (P : ℝ), compound_interest P 5 2 = 51.25 → simple_interest P 5 2 = 50 :=
by
  intros P h
  -- this is where the proof would go
  sorry

end NUMINAMATH_GPT_simple_interest_amount_l1712_171224


namespace NUMINAMATH_GPT_sum_of_first_five_primes_with_units_digit_3_l1712_171239

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_five_primes_with_units_digit_3_l1712_171239


namespace NUMINAMATH_GPT_mixture_replacement_l1712_171265

theorem mixture_replacement:
  ∀ (A B x : ℝ),
    A = 64 →
    B = A / 4 →
    (A - (4/5) * x) / (B + (4/5) * x) = 2 / 3 →
    x = 40 :=
by
  intros A B x hA hB hRatio
  sorry

end NUMINAMATH_GPT_mixture_replacement_l1712_171265


namespace NUMINAMATH_GPT_average_salary_of_all_workers_l1712_171212

theorem average_salary_of_all_workers :
  let technicians := 7
  let technicians_avg_salary := 20000
  let rest := 49 - technicians
  let rest_avg_salary := 6000
  let total_workers := 49
  let total_tech_salary := technicians * technicians_avg_salary
  let total_rest_salary := rest * rest_avg_salary
  let total_salary := total_tech_salary + total_rest_salary
  (total_salary / total_workers) = 8000 := by
  sorry

end NUMINAMATH_GPT_average_salary_of_all_workers_l1712_171212


namespace NUMINAMATH_GPT_sum_of_radical_conjugates_l1712_171288

theorem sum_of_radical_conjugates : 
  (8 - Real.sqrt 1369) + (8 + Real.sqrt 1369) = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_radical_conjugates_l1712_171288


namespace NUMINAMATH_GPT_find_t_l1712_171270

theorem find_t
  (x y t : ℝ)
  (h1 : 2 ^ x = t)
  (h2 : 5 ^ y = t)
  (h3 : 1 / x + 1 / y = 2)
  (h4 : t ≠ 1) : 
  t = Real.sqrt 10 := 
by
  sorry

end NUMINAMATH_GPT_find_t_l1712_171270


namespace NUMINAMATH_GPT_part1_part2_l1712_171256

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B (a : ℝ) := {x : ℝ | (x - a) * (x - a - 1) < 0}

theorem part1 (a : ℝ) : (1 ∈ set_B a) → 0 < a ∧ a < 1 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, x ∈ set_B a → x ∈ set_A) ∧ (∃ x, x ∉ set_B a ∧ x ∈ set_A) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1712_171256


namespace NUMINAMATH_GPT_pow2_gt_square_for_all_n_ge_5_l1712_171296

theorem pow2_gt_square_for_all_n_ge_5 (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
by
  sorry

end NUMINAMATH_GPT_pow2_gt_square_for_all_n_ge_5_l1712_171296


namespace NUMINAMATH_GPT_polygon_problem_l1712_171299

theorem polygon_problem
  (sum_interior_angles : ℕ → ℝ)
  (sum_exterior_angles : ℝ)
  (condition : ∀ n, sum_interior_angles n = (3 * sum_exterior_angles) - 180) :
  (∃ n : ℕ, sum_interior_angles n = 180 * (n - 2) ∧ n = 7) ∧
  (∃ n : ℕ, n = 7 → (n * (n - 3) / 2) = 14) :=
by
  sorry

end NUMINAMATH_GPT_polygon_problem_l1712_171299


namespace NUMINAMATH_GPT_scooter_value_depreciation_l1712_171207

theorem scooter_value_depreciation (V0 Vn : ℝ) (rate : ℝ) (n : ℕ) 
  (hV0 : V0 = 40000) 
  (hVn : Vn = 9492.1875) 
  (hRate : rate = 3 / 4) 
  (hValue : Vn = V0 * rate ^ n) : 
  n = 5 := 
by 
  -- Conditions are set up, proof needs to be constructed.
  sorry

end NUMINAMATH_GPT_scooter_value_depreciation_l1712_171207


namespace NUMINAMATH_GPT_geometric_sequence_third_term_and_sum_l1712_171203

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ :=
  b1 * r^(n - 1)

theorem geometric_sequence_third_term_and_sum (b2 b5 : ℝ) (h1 : b2 = 24.5) (h2 : b5 = 196) :
  (∃ b1 r : ℝ, r ≠ 0 ∧ geometric_sequence b1 r 2 = b2 ∧ geometric_sequence b1 r 5 = b5 ∧
  geometric_sequence b1 r 3 = 49 ∧
  b1 * (r^4 - 1) / (r - 1) = 183.75) :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_and_sum_l1712_171203


namespace NUMINAMATH_GPT_profit_no_discount_l1712_171268

theorem profit_no_discount (CP SP ASP : ℝ) (discount profit : ℝ) (h1 : discount = 4 / 100) (h2 : profit = 38 / 100) (h3 : SP = CP + CP * profit) (h4 : ASP = SP - SP * discount) :
  ((SP - CP) / CP) * 100 = 38 :=
by
  sorry

end NUMINAMATH_GPT_profit_no_discount_l1712_171268


namespace NUMINAMATH_GPT_locus_of_P_l1712_171219

theorem locus_of_P
  (a b x y : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (x ≠ 0 ∧ y ≠ 0))
  (h4 : x^2 / a^2 - y^2 / b^2 = 1) :
  (x / a)^2 - (y / b)^2 = ((a^2 + b^2) / (a^2 - b^2))^2 := by
  sorry

end NUMINAMATH_GPT_locus_of_P_l1712_171219


namespace NUMINAMATH_GPT_price_of_one_rose_l1712_171210

theorem price_of_one_rose
  (tulips1 tulips2 tulips3 roses1 roses2 roses3 : ℕ)
  (price_tulip : ℕ)
  (total_earnings : ℕ)
  (R : ℕ) :
  tulips1 = 30 →
  roses1 = 20 →
  tulips2 = 2 * tulips1 →
  roses2 = 2 * roses1 →
  tulips3 = 10 * tulips2 / 100 →  -- simplification of 0.1 * tulips2
  roses3 = 16 →
  price_tulip = 2 →
  total_earnings = 420 →
  (96 * price_tulip + 76 * R) = total_earnings →
  R = 3 :=
by
  intros
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_price_of_one_rose_l1712_171210


namespace NUMINAMATH_GPT_range_of_2a_minus_b_l1712_171291

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : 2 < b) (h4 : b < 4) : 
  -2 < 2 * a - b ∧ 2 * a - b < 4 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_2a_minus_b_l1712_171291


namespace NUMINAMATH_GPT_prime_arithmetic_sequence_l1712_171293

theorem prime_arithmetic_sequence {p1 p2 p3 d : ℕ} 
  (hp1 : Nat.Prime p1) 
  (hp2 : Nat.Prime p2) 
  (hp3 : Nat.Prime p3)
  (h3_p1 : 3 < p1)
  (h3_p2 : 3 < p2)
  (h3_p3 : 3 < p3)
  (h_seq1 : p2 = p1 + d)
  (h_seq2 : p3 = p1 + 2 * d) : 
  d % 6 = 0 :=
by sorry

end NUMINAMATH_GPT_prime_arithmetic_sequence_l1712_171293


namespace NUMINAMATH_GPT_cubed_gt_if_gt_l1712_171220

theorem cubed_gt_if_gt {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cubed_gt_if_gt_l1712_171220


namespace NUMINAMATH_GPT_set_intersection_complement_l1712_171252
open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}
def comp_T : Set ℕ := U \ T

theorem set_intersection_complement :
  S ∩ comp_T = {1, 5} := by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1712_171252


namespace NUMINAMATH_GPT_a3_equals_1_div_12_l1712_171298

-- Definition of the sequence
def seq (n : Nat) : Rat :=
  1 / (n * (n + 1))

-- Assertion to be proved
theorem a3_equals_1_div_12 : seq 3 = 1 / 12 := 
sorry

end NUMINAMATH_GPT_a3_equals_1_div_12_l1712_171298


namespace NUMINAMATH_GPT_hexagon_largest_angle_l1712_171295

variable (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ)
theorem hexagon_largest_angle (h : a₁ = 3)
                             (h₀ : a₂ = 3)
                             (h₁ : a₃ = 3)
                             (h₂ : a₄ = 4)
                             (h₃ : a₅ = 5)
                             (h₄ : a₆ = 6)
                             (sum_angles : 3*a₁ + 3*a₀ + 3*a₁ + 4*a₂ + 5*a₃ + 6*a₄ = 720) :
                             6 * 30 = 180 := by
    sorry

end NUMINAMATH_GPT_hexagon_largest_angle_l1712_171295


namespace NUMINAMATH_GPT_count_divisible_2_3_or_5_lt_100_l1712_171259
-- We need the Mathlib library for general mathematical functions

-- The main theorem statement
theorem count_divisible_2_3_or_5_lt_100 : 
  let A2 := Nat.floor (100 / 2)
  let A3 := Nat.floor (100 / 3)
  let A5 := Nat.floor (100 / 5)
  let A23 := Nat.floor (100 / 6)
  let A25 := Nat.floor (100 / 10)
  let A35 := Nat.floor (100 / 15)
  let A235 := Nat.floor (100 / 30)
  (A2 + A3 + A5 - A23 - A25 - A35 + A235) = 74 :=
by
  sorry

end NUMINAMATH_GPT_count_divisible_2_3_or_5_lt_100_l1712_171259


namespace NUMINAMATH_GPT_soap_bars_problem_l1712_171263

theorem soap_bars_problem :
  ∃ (N : ℤ), 200 < N ∧ N < 300 ∧ 2007 % N = 5 :=
sorry

end NUMINAMATH_GPT_soap_bars_problem_l1712_171263


namespace NUMINAMATH_GPT_menkara_index_card_area_l1712_171271

theorem menkara_index_card_area :
  ∀ (length width: ℕ), 
  length = 5 → width = 7 → (length - 2) * width = 21 → 
  (length * (width - 2) = 25) :=
by
  intros length width h_length h_width h_area
  sorry

end NUMINAMATH_GPT_menkara_index_card_area_l1712_171271


namespace NUMINAMATH_GPT_star_running_back_yardage_l1712_171248

-- Definitions
def total_yardage : ℕ := 150
def catching_passes_yardage : ℕ := 60
def running_yardage (total_yardage catching_passes_yardage : ℕ) : ℕ :=
  total_yardage - catching_passes_yardage

-- Statement to prove
theorem star_running_back_yardage :
  running_yardage total_yardage catching_passes_yardage = 90 := 
sorry

end NUMINAMATH_GPT_star_running_back_yardage_l1712_171248


namespace NUMINAMATH_GPT_intersection_point_of_lines_l1712_171289

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 
    2 * x + y - 7 = 0 ∧ 
    x + 2 * y - 5 = 0 ∧ 
    x = 3 ∧ 
    y = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_point_of_lines_l1712_171289


namespace NUMINAMATH_GPT_fourth_root_12960000_eq_60_l1712_171225

theorem fourth_root_12960000_eq_60 :
  (6^4 = 1296) →
  (10^4 = 10000) →
  (60^4 = 12960000) →
  (Real.sqrt (Real.sqrt 12960000) = 60) := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_fourth_root_12960000_eq_60_l1712_171225


namespace NUMINAMATH_GPT_range_of_k_l1712_171228

theorem range_of_k :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1712_171228


namespace NUMINAMATH_GPT_minimize_travel_time_l1712_171269

-- Definitions and conditions
def grid_size : ℕ := 7
def mid_point : ℕ := (grid_size + 1) / 2
def is_meeting_point (p : ℕ × ℕ) : Prop := 
  p = (mid_point, mid_point)

-- Main theorem statement to be proven
theorem minimize_travel_time : 
  ∃ (p : ℕ × ℕ), is_meeting_point p ∧
  (∀ (q : ℕ × ℕ), is_meeting_point q → p = q) :=
sorry

end NUMINAMATH_GPT_minimize_travel_time_l1712_171269


namespace NUMINAMATH_GPT_tank_capacity_l1712_171233

theorem tank_capacity (x : ℝ) (h : (5/12) * x = 150) : x = 360 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1712_171233


namespace NUMINAMATH_GPT_rotated_curve_eq_l1712_171290

theorem rotated_curve_eq :
  let θ := Real.pi / 4  -- Rotation angle 45 degrees in radians
  let cos_theta := Real.sqrt 2 / 2
  let sin_theta := Real.sqrt 2 / 2
  let x' := cos_theta * x - sin_theta * y
  let y' := sin_theta * x + cos_theta * y
  x + y^2 = 1 → x' ^ 2 + y' ^ 2 - 2 * x' * y' + Real.sqrt 2 * x' + Real.sqrt 2 * y' - 2 = 0 := 
sorry  -- Proof to be provided.

end NUMINAMATH_GPT_rotated_curve_eq_l1712_171290


namespace NUMINAMATH_GPT_value_of_f_two_l1712_171232

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_two :
  (∀ x : ℝ, f (1 / x) = 1 / (x + 1)) → f 2 = 2 / 3 := by
  intro h
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_value_of_f_two_l1712_171232


namespace NUMINAMATH_GPT_probability_of_different_colors_l1712_171211

theorem probability_of_different_colors :
  let total_chips := 12
  let prob_blue_then_yellow_red := ((6 / total_chips) * ((4 + 2) / total_chips))
  let prob_yellow_then_blue_red := ((4 / total_chips) * ((6 + 2) / total_chips))
  let prob_red_then_blue_yellow := ((2 / total_chips) * ((6 + 4) / total_chips))
  prob_blue_then_yellow_red + prob_yellow_then_blue_red + prob_red_then_blue_yellow = 11 / 18 := by
    sorry

end NUMINAMATH_GPT_probability_of_different_colors_l1712_171211


namespace NUMINAMATH_GPT_classes_after_drop_remaining_hours_of_classes_per_day_l1712_171244

def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_classes : ℕ := 1

theorem classes_after_drop 
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ) :
  initial_classes - dropped_classes = 3 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

theorem remaining_hours_of_classes_per_day
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ)
  (h : initial_classes - dropped_classes = 3) :
  hours_per_class * (initial_classes - dropped_classes) = 6 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

end NUMINAMATH_GPT_classes_after_drop_remaining_hours_of_classes_per_day_l1712_171244


namespace NUMINAMATH_GPT_nat_add_ge_3_implies_at_least_one_ge_2_l1712_171209

theorem nat_add_ge_3_implies_at_least_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_nat_add_ge_3_implies_at_least_one_ge_2_l1712_171209


namespace NUMINAMATH_GPT_quadratic_solutions_l1712_171206

theorem quadratic_solutions:
  (2 * (x : ℝ)^2 - 5 * x + 2 = 0) ↔ (x = 2 ∨ x = 1 / 2) :=
sorry

end NUMINAMATH_GPT_quadratic_solutions_l1712_171206


namespace NUMINAMATH_GPT_problem_1_problem_2_l1712_171251

-- Definitions for sets A and B
def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 6
def B (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Problem (1): What is A ∩ B when m = 3
theorem problem_1 : ∀ (x : ℝ), A x → B x 3 → (-1 ≤ x ∧ x ≤ 4) := by
  intro x hA hB
  sorry

-- Problem (2): What is the range of m if A ⊆ B and m > 0
theorem problem_2 (m : ℝ) : m > 0 → (∀ x, A x → B x m) → (m ≥ 5) := by
  intros hm hAB
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1712_171251


namespace NUMINAMATH_GPT_sum_of_squares_l1712_171205

theorem sum_of_squares (R r r1 r2 r3 d d1 d2 d3 : ℝ) 
  (h1 : d^2 = R^2 - 2 * R * r)
  (h2 : d1^2 = R^2 + 2 * R * r1)
  (h3 : d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2) :
  d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1712_171205


namespace NUMINAMATH_GPT_equilateral_sector_area_l1712_171238

noncomputable def area_of_equilateral_sector (r : ℝ) : ℝ :=
  if h : r = r then (1/2) * r^2 * 1 else 0

theorem equilateral_sector_area (r : ℝ) : r = 2 → area_of_equilateral_sector r = 2 :=
by
  intros hr
  rw [hr]
  unfold area_of_equilateral_sector
  split_ifs
  · norm_num
  · contradiction

end NUMINAMATH_GPT_equilateral_sector_area_l1712_171238


namespace NUMINAMATH_GPT_find_fraction_l1712_171281

theorem find_fraction (f : ℝ) (n : ℝ) (h : n = 180) (eqn : f * ((1 / 3) * (1 / 5) * n) + 6 = (1 / 15) * n) : f = 1 / 2 :=
by
  -- Definitions and assumptions provided above will be used here.
  sorry

end NUMINAMATH_GPT_find_fraction_l1712_171281


namespace NUMINAMATH_GPT_solve_fraction_equation_l1712_171226

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 5 * x - 14) - 1 / (x^2 - 15 * x - 18) = 0) →
  x = 2 ∨ x = -9 ∨ x = 6 ∨ x = -3 :=
sorry

end NUMINAMATH_GPT_solve_fraction_equation_l1712_171226


namespace NUMINAMATH_GPT_avg_height_students_l1712_171276

theorem avg_height_students 
  (x : ℕ)  -- number of students in the first group
  (avg_height_first_group : ℕ)  -- average height of the first group
  (avg_height_second_group : ℕ)  -- average height of the second group
  (avg_height_combined_group : ℕ)  -- average height of the combined group
  (h1 : avg_height_first_group = 20)
  (h2 : avg_height_second_group = 20)
  (h3 : avg_height_combined_group = 20)
  (h4 : 20*x + 20*11 = 20*31) :
  x = 20 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_avg_height_students_l1712_171276


namespace NUMINAMATH_GPT_find_x_value_l1712_171278

theorem find_x_value (x : ℝ) (h1 : x^2 + x = 6) (h2 : x^2 - 2 = 1) : x = 2 := sorry

end NUMINAMATH_GPT_find_x_value_l1712_171278


namespace NUMINAMATH_GPT_fraction_of_girls_l1712_171249

variable (total_students : ℕ) (number_of_boys : ℕ)

theorem fraction_of_girls (h1 : total_students = 160) (h2 : number_of_boys = 60) :
    (total_students - number_of_boys) / total_students = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_of_girls_l1712_171249


namespace NUMINAMATH_GPT_tangent_line_at_point_l1712_171223

noncomputable def tangent_line_equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  x - f 0 + 2 = 0

theorem tangent_line_at_point (f : ℝ → ℝ)
  (h_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y)
  (h_eq : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) :
  tangent_line_equation f 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l1712_171223


namespace NUMINAMATH_GPT_evaluate_expressions_for_pos_x_l1712_171235

theorem evaluate_expressions_for_pos_x :
  (∀ x : ℝ, x > 0 → 6^x * x^3 = 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (3 * x)^(3 * x) ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → 3^x * x^6 ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (6 * x)^x ≠ 6^x * x^3) →
  ∃ n : ℕ, n = 1 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expressions_for_pos_x_l1712_171235


namespace NUMINAMATH_GPT_john_pin_discount_l1712_171216

theorem john_pin_discount :
  ∀ (n_pins price_per_pin amount_spent discount_rate : ℝ),
    n_pins = 10 →
    price_per_pin = 20 →
    amount_spent = 170 →
    discount_rate = ((n_pins * price_per_pin - amount_spent) / (n_pins * price_per_pin)) * 100 →
    discount_rate = 15 :=
by
  intros n_pins price_per_pin amount_spent discount_rate h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_john_pin_discount_l1712_171216


namespace NUMINAMATH_GPT_total_area_of_field_l1712_171200

theorem total_area_of_field (A1 A2 : ℝ) (h1 : A1 = 225)
    (h2 : A2 - A1 = (1 / 5) * ((A1 + A2) / 2)) :
  A1 + A2 = 500 := by
  sorry

end NUMINAMATH_GPT_total_area_of_field_l1712_171200


namespace NUMINAMATH_GPT_max_profit_thousand_rubles_l1712_171234

theorem max_profit_thousand_rubles :
  ∃ x y : ℕ, 
    (80 * x + 100 * y = 2180) ∧ 
    (10 * x + 70 * y ≤ 700) ∧ 
    (23 * x + 40 * y ≤ 642) := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_max_profit_thousand_rubles_l1712_171234


namespace NUMINAMATH_GPT_james_profit_l1712_171202

theorem james_profit
  (tickets_bought : ℕ)
  (cost_per_ticket : ℕ)
  (percentage_winning : ℕ)
  (winning_tickets_percentage_5dollars : ℕ)
  (grand_prize : ℕ)
  (average_other_prizes : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (winning_tickets : ℕ)
  (tickets_prize_5dollars : ℕ)
  (amount_won_5dollars : ℕ)
  (other_winning_tickets : ℕ)
  (other_tickets_prize : ℕ)
  (total_winning_amount : ℕ)
  (profit : ℕ) :

  tickets_bought = 200 →
  cost_per_ticket = 2 →
  percentage_winning = 20 →
  winning_tickets_percentage_5dollars = 80 →
  grand_prize = 5000 →
  average_other_prizes = 10 →
  total_tickets = tickets_bought →
  total_cost = total_tickets * cost_per_ticket →
  winning_tickets = (percentage_winning * total_tickets) / 100 →
  tickets_prize_5dollars = (winning_tickets_percentage_5dollars * winning_tickets) / 100 →
  amount_won_5dollars = tickets_prize_5dollars * 5 →
  other_winning_tickets = winning_tickets - 1 →
  other_tickets_prize = (other_winning_tickets - tickets_prize_5dollars) * average_other_prizes →
  total_winning_amount = amount_won_5dollars + grand_prize + other_tickets_prize →
  profit = total_winning_amount - total_cost →
  profit = 4830 := 
sorry

end NUMINAMATH_GPT_james_profit_l1712_171202


namespace NUMINAMATH_GPT_average_mark_of_excluded_students_l1712_171254

theorem average_mark_of_excluded_students (N A E A_R A_E : ℝ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hAR : A_R = 90) 
  (h_eq : N * A - E * A_E = (N - E) * A_R) : 
  A_E = 40 := 
by 
  sorry

end NUMINAMATH_GPT_average_mark_of_excluded_students_l1712_171254


namespace NUMINAMATH_GPT_bowls_total_marbles_l1712_171240

theorem bowls_total_marbles :
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  C1 = 450 ∧ C3 = 225 ∧ (C1 + C2 + C3 = 1275) := 
by
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  have hC1 : C1 = 450 := by norm_num
  have hC3 : C3 = 225 := by norm_num
  have hTotal : C1 + C2 + C3 = 1275 := by norm_num
  exact ⟨hC1, hC3, hTotal⟩

end NUMINAMATH_GPT_bowls_total_marbles_l1712_171240


namespace NUMINAMATH_GPT_janice_typing_proof_l1712_171237

noncomputable def janice_typing : Prop :=
  let initial_speed := 6
  let error_speed := 8
  let corrected_speed := 5
  let typing_duration_initial := 20
  let typing_duration_corrected := 15
  let erased_sentences := 40
  let typing_duration_after_lunch := 18
  let total_sentences_end_of_day := 536

  let sentences_initial_typing := typing_duration_initial * error_speed
  let sentences_post_error_typing := typing_duration_corrected * initial_speed
  let sentences_final_typing := typing_duration_after_lunch * corrected_speed

  let sentences_total_typed := sentences_initial_typing + sentences_post_error_typing - erased_sentences + sentences_final_typing

  let sentences_started_with := total_sentences_end_of_day - sentences_total_typed

  sentences_started_with = 236

theorem janice_typing_proof : janice_typing := by
  sorry

end NUMINAMATH_GPT_janice_typing_proof_l1712_171237


namespace NUMINAMATH_GPT_max_value_of_quadratic_l1712_171257

-- Define the quadratic function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the main theorem of finding the maximum value
theorem max_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 11 := sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l1712_171257


namespace NUMINAMATH_GPT_abs_neg_seven_l1712_171287

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end NUMINAMATH_GPT_abs_neg_seven_l1712_171287


namespace NUMINAMATH_GPT_odd_function_behavior_l1712_171285

variable {f : ℝ → ℝ}

theorem odd_function_behavior (h1 : ∀ x : ℝ, f (-x) = -f x) 
                             (h2 : ∀ x : ℝ, 0 < x → f x = x * (1 + x)) 
                             (x : ℝ)
                             (hx : x < 0) : 
  f x = x * (1 - x) :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_odd_function_behavior_l1712_171285


namespace NUMINAMATH_GPT_impossible_permuted_sum_l1712_171280

def isPermutation (X Y : ℕ) : Prop :=
  -- Define what it means for two numbers to be permutations of each other.
  sorry

theorem impossible_permuted_sum (X Y : ℕ) (h1 : isPermutation X Y) (h2 : X + Y = (10^1111 - 1)) : false :=
  sorry

end NUMINAMATH_GPT_impossible_permuted_sum_l1712_171280


namespace NUMINAMATH_GPT_problem1_problem2_l1712_171245

-- Definitions of sets A and B
def A : Set ℝ := { x | x > 1 }
def B (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Problem 1:
theorem problem1 (a : ℝ) : B a ⊆ A → 1 ≤ a :=
  sorry

-- Problem 2:
theorem problem2 (a : ℝ) : (A ∩ B a).Nonempty → 0 < a :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1712_171245


namespace NUMINAMATH_GPT_largest_power_of_two_dividing_7_pow_2048_minus_1_l1712_171214

theorem largest_power_of_two_dividing_7_pow_2048_minus_1 :
  ∃ n : ℕ, 2^n ∣ (7^2048 - 1) ∧ n = 14 :=
by
  use 14
  sorry

end NUMINAMATH_GPT_largest_power_of_two_dividing_7_pow_2048_minus_1_l1712_171214


namespace NUMINAMATH_GPT_sequence_general_term_l1712_171241

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, a n = S n - S (n-1) :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1712_171241


namespace NUMINAMATH_GPT_intersection_complement_l1712_171282

def real_set_M : Set ℝ := {x | 1 < x}
def real_set_N : Set ℝ := {x | x > 4}

theorem intersection_complement (x : ℝ) : x ∈ (real_set_M ∩ (real_set_Nᶜ)) ↔ 1 < x ∧ x ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1712_171282


namespace NUMINAMATH_GPT_race_outcomes_l1712_171279

-- Definition of participants
inductive Participant
| Abe 
| Bobby
| Charles
| Devin
| Edwin
| Frank
deriving DecidableEq

open Participant

def num_participants : ℕ := 6

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Proving the number of different 1st-2nd-3rd outcomes
theorem race_outcomes : factorial 6 / factorial 3 = 120 := by
  sorry

end NUMINAMATH_GPT_race_outcomes_l1712_171279


namespace NUMINAMATH_GPT_find_x_l1712_171273

/-- Let r be the result of doubling both the base and exponent of a^b, 
and b does not equal to 0. If r equals the product of a^b by x^b,
then x equals 4a. -/
theorem find_x (a b x: ℝ) (h₁ : b ≠ 0) (h₂ : (2*a)^(2*b) = a^b * x^b) : x = 4*a := 
  sorry

end NUMINAMATH_GPT_find_x_l1712_171273


namespace NUMINAMATH_GPT_find_integer_of_divisors_l1712_171250

theorem find_integer_of_divisors:
  ∃ (N : ℕ), (∀ (l m n : ℕ), N = (2^l) * (3^m) * (5^n) → 
  (2^120) * (3^60) * (5^90) = (2^l * 3^m * 5^n)^( ((l+1)*(m+1)*(n+1)) / 2 ) ) → 
  N = 18000 :=
sorry

end NUMINAMATH_GPT_find_integer_of_divisors_l1712_171250


namespace NUMINAMATH_GPT_statue_of_liberty_model_height_l1712_171218

theorem statue_of_liberty_model_height :
  let scale_ratio : Int := 30
  let actual_height : Int := 305
  round (actual_height / scale_ratio) = 10 := by
  sorry

end NUMINAMATH_GPT_statue_of_liberty_model_height_l1712_171218


namespace NUMINAMATH_GPT_minimum_value_l1712_171231

variable (a b : ℝ)
variable (ab_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (circle1 : ∀ x y, x^2 + y^2 + 2 * a * x + a^2 - 9 = 0)
variable (circle2 : ∀ x y, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0)
variable (centers_distance : a^2 + 4 * b^2 = 16)

theorem minimum_value :
  (4 / a^2 + 1 / b^2) = 1 := sorry

end NUMINAMATH_GPT_minimum_value_l1712_171231


namespace NUMINAMATH_GPT_find_C_share_l1712_171277

-- Definitions
variable (A B C : ℝ)
variable (H1 : A + B + C = 585)
variable (H2 : 4 * A = 6 * B)
variable (H3 : 6 * B = 3 * C)

-- Problem statement
theorem find_C_share (A B C : ℝ) (H1 : A + B + C = 585) (H2 : 4 * A = 6 * B) (H3 : 6 * B = 3 * C) : C = 260 :=
by
  sorry

end NUMINAMATH_GPT_find_C_share_l1712_171277
