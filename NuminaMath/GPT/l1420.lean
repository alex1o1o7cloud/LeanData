import Mathlib

namespace geometric_sequence_condition_l1420_142056

variable (a b c : ℝ)

-- Condition: For a, b, c to form a geometric sequence.
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ≠ 0) ∧ (b^2 = a * c)

-- Given that a, b, c are real numbers
-- Prove that ac = b^2 is a necessary but not sufficient condition for a, b, c to form a geometric sequence.
theorem geometric_sequence_condition (a b c : ℝ) (h : a * c = b^2) :
  ¬ (∃ b : ℝ, b^2 = a * c → (is_geometric_sequence a b c)) :=
sorry

end geometric_sequence_condition_l1420_142056


namespace partial_fraction_sum_zero_l1420_142098

theorem partial_fraction_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_zero_l1420_142098


namespace john_amount_share_l1420_142015

theorem john_amount_share {total_amount : ℕ} {total_parts john_share : ℕ} (h1 : total_amount = 4200) (h2 : total_parts = 2 + 4 + 6) (h3 : john_share = 2) :
  john_share * (total_amount / total_parts) = 700 :=
by
  sorry

end john_amount_share_l1420_142015


namespace fenced_yard_area_l1420_142001

theorem fenced_yard_area :
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  yard - cutout1 - cutout2 = 343 := by
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  have h : yard - cutout1 - cutout2 = 343 := sorry
  exact h

end fenced_yard_area_l1420_142001


namespace evaluate_expression_l1420_142064

theorem evaluate_expression :
  let a := 3^1005
  let b := 4^1006
  (a + b)^2 - (a - b)^2 = 160 * 10^1004 :=
by
  sorry

end evaluate_expression_l1420_142064


namespace mass_percentage_O_in_Al2_CO3_3_l1420_142084

-- Define the atomic masses
def atomic_mass_Al : Float := 26.98
def atomic_mass_C : Float := 12.01
def atomic_mass_O : Float := 16.00

-- Define the formula of aluminum carbonate
def Al_count : Nat := 2
def C_count : Nat := 3
def O_count : Nat := 9

-- Define the molar mass calculation
def molar_mass_Al2_CO3_3 : Float :=
  (Al_count.toFloat * atomic_mass_Al) + 
  (C_count.toFloat * atomic_mass_C) + 
  (O_count.toFloat * atomic_mass_O)

-- Define the mass of oxygen in aluminum carbonate
def mass_O_in_Al2_CO3_3 : Float := O_count.toFloat * atomic_mass_O

-- Define the mass percentage of oxygen in aluminum carbonate
def mass_percentage_O : Float := (mass_O_in_Al2_CO3_3 / molar_mass_Al2_CO3_3) * 100

-- Proof statement
theorem mass_percentage_O_in_Al2_CO3_3 :
  mass_percentage_O = 61.54 := by
  sorry

end mass_percentage_O_in_Al2_CO3_3_l1420_142084


namespace matt_total_score_l1420_142018

-- Definitions from the conditions
def num_2_point_shots : ℕ := 4
def num_3_point_shots : ℕ := 2
def score_per_2_point_shot : ℕ := 2
def score_per_3_point_shot : ℕ := 3

-- Proof statement
theorem matt_total_score : 
  (num_2_point_shots * score_per_2_point_shot) + 
  (num_3_point_shots * score_per_3_point_shot) = 14 := 
by 
  sorry  -- placeholder for the actual proof

end matt_total_score_l1420_142018


namespace lcm_20_45_36_l1420_142063

-- Definitions from the problem
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 36

-- Statement of the proof problem
theorem lcm_20_45_36 : Nat.lcm (Nat.lcm num1 num2) num3 = 180 := by
  sorry

end lcm_20_45_36_l1420_142063


namespace find_number_l1420_142029

theorem find_number (x : ℝ) : (x * 12) / (180 / 3) + 80 = 81 → x = 5 :=
by
  sorry

end find_number_l1420_142029


namespace cos_360_eq_one_l1420_142092

theorem cos_360_eq_one : Real.cos (2 * Real.pi) = 1 :=
by sorry

end cos_360_eq_one_l1420_142092


namespace poly_value_at_two_l1420_142055

def f (x : ℝ) : ℝ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem poly_value_at_two : f 2 = 216 :=
by
  unfold f
  norm_num
  sorry

end poly_value_at_two_l1420_142055


namespace factors_of_48_multiples_of_8_l1420_142007

theorem factors_of_48_multiples_of_8 : 
  ∃ count : ℕ, count = 4 ∧ (∀ d ∈ {d | d ∣ 48 ∧ (∃ k, d = 8 * k)}, true) :=
by {
  sorry  -- This is a placeholder for the actual proof
}

end factors_of_48_multiples_of_8_l1420_142007


namespace correct_operation_l1420_142033

theorem correct_operation :
  (∀ a : ℝ, (a^4)^2 ≠ a^6) ∧
  (∀ a b : ℝ, (a - b)^2 ≠ a^2 - ab + b^2) ∧
  (∀ a b : ℝ, 6 * a^2 * b / (2 * a * b) = 3 * a) ∧
  (∀ a : ℝ, a^2 + a^4 ≠ a^6) :=
by {
  sorry
}

end correct_operation_l1420_142033


namespace length_AB_l1420_142065

noncomputable def parabola_p := 3
def x1_x2_sum := 6

theorem length_AB (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : x1 + x2 = x1_x2_sum)
  (h2 : (y1^2 = 6 * x1) ∧ (y2^2 = 6 * x2))
  : abs (x1 + parabola_p / 2 - (x2 + parabola_p / 2)) = 9 := by
  sorry

end length_AB_l1420_142065


namespace no_integer_n_exists_l1420_142051

theorem no_integer_n_exists (n : ℤ) : ¬(∃ n : ℤ, ∃ k : ℤ, ∃ m : ℤ, (n - 6) = 15 * k ∧ (n - 5) = 24 * m) :=
by
  sorry

end no_integer_n_exists_l1420_142051


namespace find_n_l1420_142090

theorem find_n (x n : ℝ) (h : x > 0) 
  (h_eq : x / 10 + x / n = 0.14000000000000002 * x) : 
  n = 25 :=
by
  sorry

end find_n_l1420_142090


namespace simple_interest_difference_l1420_142035

theorem simple_interest_difference :
  let P : ℝ := 900
  let R1 : ℝ := 4
  let R2 : ℝ := 4.5
  let T : ℝ := 7
  let SI1 := P * R1 * T / 100
  let SI2 := P * R2 * T / 100
  SI2 - SI1 = 31.50 := by
  sorry

end simple_interest_difference_l1420_142035


namespace completing_the_square_l1420_142042

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l1420_142042


namespace expand_product_l1420_142013

theorem expand_product (x : ℝ) : 
  5 * (x + 6) * (x^2 + 2 * x + 3) = 5 * x^3 + 40 * x^2 + 75 * x + 90 := 
by 
  sorry

end expand_product_l1420_142013


namespace isosceles_triangle_base_length_l1420_142040

noncomputable def equilateral_side_length (p_eq : ℕ) : ℕ := p_eq / 3

theorem isosceles_triangle_base_length (p_eq p_iso s b : ℕ) 
  (h1 : p_eq = 45)
  (h2 : p_iso = 40)
  (h3 : s = equilateral_side_length p_eq)
  (h4 : p_iso = s + s + b)
  : b = 10 :=
by
  simp [h1, h2, h3] at h4
  -- steps to solve for b would be written here
  sorry

end isosceles_triangle_base_length_l1420_142040


namespace food_remaining_l1420_142021

-- Definitions for conditions
def first_week_donations : ℕ := 40
def second_week_donations := 2 * first_week_donations
def total_donations := first_week_donations + second_week_donations
def percentage_given_out : ℝ := 0.70
def amount_given_out := percentage_given_out * total_donations

-- Proof goal
theorem food_remaining (h1 : first_week_donations = 40)
                      (h2 : second_week_donations = 2 * first_week_donations)
                      (h3 : percentage_given_out = 0.70) :
                      total_donations - amount_given_out = 36 := by
  sorry

end food_remaining_l1420_142021


namespace find_b_c_l1420_142026

theorem find_b_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6)
  (h3 : a * b + b * c + c * d + d * a = 28) : 
  b + c = 17 / 3 := 
by
  sorry

end find_b_c_l1420_142026


namespace midpoint_distance_l1420_142095

theorem midpoint_distance (a b c d : ℝ) :
  let m := (a + c) / 2
  let n := (b + d) / 2
  let m' := m - 0.5
  let n' := n - 0.5
  dist (m, n) (m', n') = (Real.sqrt 2) / 2 := 
by 
  sorry

end midpoint_distance_l1420_142095


namespace total_swordfish_caught_correct_l1420_142099

-- Define the parameters and the catch per trip
def shelly_catch_per_trip : ℕ := 5 - 2
def sam_catch_per_trip (shelly_catch : ℕ) : ℕ := shelly_catch - 1
def total_catch_per_trip (shelly_catch sam_catch : ℕ) : ℕ := shelly_catch + sam_catch
def total_trips : ℕ := 5

-- Define the total number of swordfish caught over the trips
def total_swordfish_caught : ℕ :=
  let shelly_catch := shelly_catch_per_trip
  let sam_catch := sam_catch_per_trip shelly_catch
  let total_catch := total_catch_per_trip shelly_catch sam_catch
  total_catch * total_trips

-- The theorem states that the total swordfish caught is 25
theorem total_swordfish_caught_correct : total_swordfish_caught = 25 := by
  sorry

end total_swordfish_caught_correct_l1420_142099


namespace tangent_line_parabola_l1420_142012

theorem tangent_line_parabola (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  ∀ x y : ℝ, (y^2 = 4 * x) ∧ (P = (-1, 0)) → (x + y + 1 = 0) ∨ (x - y + 1 = 0) := by
  sorry

end tangent_line_parabola_l1420_142012


namespace probability_y_eq_2x_l1420_142032

/-- Two fair cubic dice each have six faces labeled with the numbers 1, 2, 3, 4, 5, and 6. 
Rolling these dice sequentially, find the probability that the number on the top face 
of the second die (y) is twice the number on the top face of the first die (x). --/
noncomputable def dice_probability : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem probability_y_eq_2x : dice_probability = 1 / 12 :=
  by sorry

end probability_y_eq_2x_l1420_142032


namespace y_relationship_l1420_142088

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (hA : y1 = -7 * x1 + 14) 
  (hB : y2 = -7 * x2 + 14) 
  (hC : y3 = -7 * x3 + 14) 
  (hx : x1 > x3 ∧ x3 > x2) : y1 < y3 ∧ y3 < y2 :=
by
  sorry

end y_relationship_l1420_142088


namespace sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l1420_142009

def row_10_pascals_triangle : List ℕ := [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]

theorem sum_of_row_10_pascals_triangle :
  (List.sum row_10_pascals_triangle) = 1024 := by
  sorry

theorem sum_of_squares_of_row_10_pascals_triangle :
  (List.sum (List.map (fun x => x * x) row_10_pascals_triangle)) = 183756 := by
  sorry

end sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l1420_142009


namespace new_group_size_l1420_142074

theorem new_group_size (N : ℕ) (h1 : 20 < N) (h2 : N < 50) (h3 : (N - 5) % 6 = 0) (h4 : (N - 5) % 7 = 0) (h5 : (N % (N - 7)) = 7) : (N - 7).gcd (N) = 8 :=
by
  sorry

end new_group_size_l1420_142074


namespace sequence_arith_l1420_142030

theorem sequence_arith {a : ℕ → ℕ} (h_initial : a 2 = 2) (h_recursive : ∀ n ≥ 2, a (n + 1) = a n + 1) :
  ∀ n ≥ 2, a n = n :=
by
  sorry

end sequence_arith_l1420_142030


namespace hardcover_volumes_l1420_142072

theorem hardcover_volumes (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 25 * h + 15 * p = 240) : h = 6 :=
by
  -- omitted proof steps for brevity
  sorry

end hardcover_volumes_l1420_142072


namespace local_minimum_at_neg_one_l1420_142091

noncomputable def f (x : ℝ) := x * Real.exp x

theorem local_minimum_at_neg_one : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 1) < δ → f x > f (-1)) :=
sorry

end local_minimum_at_neg_one_l1420_142091


namespace arithmetic_seq_a2_a8_a5_l1420_142041

-- Define the sequence and sum conditions
variable {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Define the given conditions
axiom seq_condition (n : ℕ) : (1 - q) * S n + q * a n = 1
axiom q_nonzero : q * (q - 1) ≠ 0
axiom geom_seq : ∀ n, a n = q^(n - 1)

-- Main theorem (consistent with both parts (Ⅰ) and (Ⅱ) results)
theorem arithmetic_seq_a2_a8_a5 (S_arith : S 3 + S 6 = 2 * S 9) : a 2 + a 5 = 2 * a 8 :=
by
    sorry

end arithmetic_seq_a2_a8_a5_l1420_142041


namespace amitabh_avg_expenditure_feb_to_jul_l1420_142077

variable (expenditure_avg_jan_to_jun expenditure_jan expenditure_jul : ℕ)

theorem amitabh_avg_expenditure_feb_to_jul (h1 : expenditure_avg_jan_to_jun = 4200) 
  (h2 : expenditure_jan = 1200) (h3 : expenditure_jul = 1500) :
  (expenditure_avg_jan_to_jun * 6 - expenditure_jan + expenditure_jul) / 6 = 4250 := by
  -- Using the given conditions
  sorry

end amitabh_avg_expenditure_feb_to_jul_l1420_142077


namespace james_ride_time_l1420_142037

theorem james_ride_time :
  let distance := 80 
  let speed := 16 
  distance / speed = 5 := 
by
  -- sorry to skip the proof
  sorry

end james_ride_time_l1420_142037


namespace area_of_triangle_ABC_l1420_142057

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 2 }
def B : Point := { x := 6, y := 0 }
def C : Point := { x := 4, y := 7 }

def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * abs (P1.x * (P2.y - P3.y) +
             P2.x * (P3.y - P1.y) +
             P3.x * (P1.y - P2.y))

theorem area_of_triangle_ABC : triangle_area A B C = 19 :=
by
  sorry

end area_of_triangle_ABC_l1420_142057


namespace sampling_methods_l1420_142014
-- Import the necessary library

-- Definitions for the conditions of the problem:
def NumberOfFamilies := 500
def HighIncomeFamilies := 125
def MiddleIncomeFamilies := 280
def LowIncomeFamilies := 95
def SampleSize := 100

def FemaleStudentAthletes := 12
def NumberToChoose := 3

-- Define the appropriate sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Stating the proof problem in Lean 4
theorem sampling_methods :
  SamplingMethod.Stratified = SamplingMethod.Stratified ∧
  SamplingMethod.SimpleRandom = SamplingMethod.SimpleRandom :=
by
  -- Proof is omitted in this theorem statement
  sorry

end sampling_methods_l1420_142014


namespace production_volume_bounds_l1420_142049

theorem production_volume_bounds:
  ∀ (x : ℕ),
  (10 * x ≤ 800 * 2400) ∧ 
  (10 * x ≤ 4000000 + 16000000) ∧
  (x ≥ 1800000) →
  (1800000 ≤ x ∧ x ≤ 1920000) :=
by
  sorry

end production_volume_bounds_l1420_142049


namespace zero_in_interval_l1420_142004

noncomputable def f (x : ℝ) := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end zero_in_interval_l1420_142004


namespace library_shelves_l1420_142073

theorem library_shelves (S : ℕ) (h_books : 4305 + 11 = 4316) :
  4316 % S = 0 ↔ S = 11 :=
by 
  have h_total_books := h_books
  sorry

end library_shelves_l1420_142073


namespace annulus_area_l1420_142096

theorem annulus_area (r_inner r_outer : ℝ) (h_inner : r_inner = 8) (h_outer : r_outer = 2 * r_inner) :
  π * r_outer ^ 2 - π * r_inner ^ 2 = 192 * π :=
by
  sorry

end annulus_area_l1420_142096


namespace second_container_clay_l1420_142068

theorem second_container_clay :
  let h1 := 3
  let w1 := 5
  let l1 := 7
  let clay1 := 105
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let V1 := h1 * w1 * l1
  let V2 := h2 * w2 * l2
  V1 = clay1 →
  V2 = 6 * V1 →
  V2 = 630 :=
by
  intros
  sorry

end second_container_clay_l1420_142068


namespace abs_neg_five_l1420_142082

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l1420_142082


namespace saturated_function_2014_l1420_142046

def saturated (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f^[f^[f n] n] n = n

theorem saturated_function_2014 (f : ℕ → ℕ) (m : ℕ) (h : saturated f) :
  (m ∣ 2014) ↔ (f^[2014] m = m) :=
sorry

end saturated_function_2014_l1420_142046


namespace simplify_expression_l1420_142019

theorem simplify_expression (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) = 
  (y^6 - y^5 + 2 * y^4 + y^3 - 2) :=
by 
  sorry

end simplify_expression_l1420_142019


namespace remainder_relation_l1420_142087

theorem remainder_relation (P P' D R R' : ℕ) (hP : P > P') (h1 : P % D = R) (h2 : P' % D = R') :
  ∃ C : ℕ, ((P + C) * P') % D ≠ (P * P') % D ∧ ∃ C : ℕ, ((P + C) * P') % D = (P * P') % D :=
by sorry

end remainder_relation_l1420_142087


namespace arithmetic_sequence_geometric_l1420_142062

theorem arithmetic_sequence_geometric (a : ℕ → ℤ) (d : ℤ) (m n : ℕ)
  (h1 : ∀ n, a (n+1) = a 1 + n * d)
  (h2 : a 1 = 1)
  (h3 : (a 3 - 2)^2 = a 1 * a 5)
  (h_d_nonzero : d ≠ 0)
  (h_mn : m - n = 10) :
  a m - a n = 30 := 
by
  sorry

end arithmetic_sequence_geometric_l1420_142062


namespace part_a_l1420_142076

theorem part_a (a b c : Int) (h1 : a + b + c = 0) : 
  ¬(a ^ 1999 + b ^ 1999 + c ^ 1999 = 2) :=
by
  sorry

end part_a_l1420_142076


namespace cylinder_height_relation_l1420_142070

variables (r1 h1 r2 h2 : ℝ)
variables (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2) (r2_eq_1_2_r1 : r2 = 1.2 * r1)

theorem cylinder_height_relation : h1 = 1.44 * h2 :=
by
  sorry

end cylinder_height_relation_l1420_142070


namespace square_mirror_side_length_l1420_142048

theorem square_mirror_side_length :
  ∃ (side_length : ℝ),
  let wall_width := 42
  let wall_length := 27.428571428571427
  let wall_area := wall_width * wall_length
  let mirror_area := wall_area / 2
  (side_length * side_length = mirror_area) → side_length = 24 :=
by
  use 24
  intro h
  sorry

end square_mirror_side_length_l1420_142048


namespace largest_number_of_minerals_per_shelf_l1420_142058

theorem largest_number_of_minerals_per_shelf (d : ℕ) :
  d ∣ 924 ∧ d ∣ 1386 ∧ d ∣ 462 ↔ d = 462 :=
by
  sorry

end largest_number_of_minerals_per_shelf_l1420_142058


namespace reciprocal_of_one_is_one_l1420_142025

def is_reciprocal (x y : ℝ) : Prop := x * y = 1

theorem reciprocal_of_one_is_one : is_reciprocal 1 1 := 
by
  sorry

end reciprocal_of_one_is_one_l1420_142025


namespace simplify_expression_l1420_142022

theorem simplify_expression (x : ℝ) (h : x^2 - x - 1 = 0) :
  ( ( (x - 1) / x - (x - 2) / (x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) ) = 1 := 
by
  sorry

end simplify_expression_l1420_142022


namespace grill_run_time_l1420_142031

def time_burn (coals : ℕ) (burn_rate : ℕ) (interval : ℕ) : ℚ :=
  (coals / burn_rate) * interval

theorem grill_run_time :
  let time_a1 := time_burn 60 15 20
  let time_a2 := time_burn 75 12 20
  let time_a3 := time_burn 45 15 20
  let time_b1 := time_burn 50 10 30
  let time_b2 := time_burn 70 8 30
  let time_b3 := time_burn 40 10 30
  let time_b4 := time_burn 80 8 30
  time_a1 + time_a2 + time_a3 + time_b1 + time_b2 + time_b3 + time_b4 = 1097.5 := sorry

end grill_run_time_l1420_142031


namespace problem_proof_l1420_142067

theorem problem_proof (M N : ℕ) 
  (h1 : 4 * 63 = 7 * M) 
  (h2 : 4 * N = 7 * 84) : 
  M + N = 183 :=
sorry

end problem_proof_l1420_142067


namespace exists_q_lt_1_l1420_142080

variable {a : ℕ → ℝ}

theorem exists_q_lt_1 (h_nonneg : ∀ n, 0 ≤ a n)
  (h_rec : ∀ k m, a (k + m) ≤ a (k + m + 1) + a k * a m)
  (h_large_n : ∃ n₀, ∀ n ≥ n₀, n * a n < 0.2499) :
  ∃ q, 0 < q ∧ q < 1 ∧ (∃ n₀, ∀ n ≥ n₀, a n < q ^ n) :=
by
  sorry

end exists_q_lt_1_l1420_142080


namespace solution_set_inequality_l1420_142054

theorem solution_set_inequality (a x : ℝ) :
  (12 * x^2 - a * x > a^2) ↔
  ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
   (a = 0 ∧ x ≠ 0) ∨
   (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
sorry


end solution_set_inequality_l1420_142054


namespace shaded_area_possible_values_l1420_142086

variable (AB BC PQ SC : ℕ)

-- Conditions:
def dimensions_correct : Prop := AB * BC = 33 ∧ AB < 7 ∧ BC < 7
def length_constraint : Prop := PQ < SC

-- Theorem statement
theorem shaded_area_possible_values (h1 : dimensions_correct AB BC) (h2 : length_constraint PQ SC) :
  (AB = 3 ∧ BC = 11 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17) ∨
                      (33 - 2 * 3 - 1 * 6 = 21) ∨
                      (33 - 2 * 4 - 1 * 5 = 20))) ∨ 
  (AB = 11 ∧ BC = 3 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17))) :=
sorry

end shaded_area_possible_values_l1420_142086


namespace median_on_hypotenuse_length_l1420_142050

theorem median_on_hypotenuse_length
  (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) (right_triangle : (a ^ 2 + b ^ 2) = c ^ 2) :
  (1 / 2) * c = 5 :=
  sorry

end median_on_hypotenuse_length_l1420_142050


namespace race_time_difference_l1420_142069

-- Define Malcolm's speed, Joshua's speed, and the distance
def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 7 -- minutes per mile
def race_distance := 15 -- miles

-- Statement of the theorem
theorem race_time_difference :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 15 :=
by sorry

end race_time_difference_l1420_142069


namespace rectangle_dimension_l1420_142038

theorem rectangle_dimension (x : ℝ) (h : (x^2) * (x + 5) = 3 * (2 * (x^2) + 2 * (x + 5))) : x = 3 :=
by
  have eq1 : (x^2) * (x + 5) = x^3 + 5 * x^2 := by ring
  have eq2 : 3 * (2 * (x^2) + 2 * (x + 5)) = 6 * x^2 + 6 * x + 30 := by ring
  rw [eq1, eq2] at h
  sorry  -- Proof details omitted

end rectangle_dimension_l1420_142038


namespace negation_statement_l1420_142020

variable {α : Type} (S : Set α)

theorem negation_statement (P : α → Prop) :
  (∀ x ∈ S, ¬ P x) ↔ (∃ x ∈ S, P x) :=
by
  sorry

end negation_statement_l1420_142020


namespace proof_problem_l1420_142010

-- Given conditions
variables {a b c : ℕ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a > b) (h5 : a^2 - a * b - a * c + b * c = 7)

-- Statement to prove
theorem proof_problem : a - c = 1 ∨ a - c = 7 :=
sorry

end proof_problem_l1420_142010


namespace polygon_sides_and_diagonals_l1420_142027

theorem polygon_sides_and_diagonals (n : ℕ) (h : (n-2) * 180 / 360 = 13 / 2) : 
  n = 15 ∧ (n * (n - 3) / 2 = 90) :=
by {
  sorry
}

end polygon_sides_and_diagonals_l1420_142027


namespace segment_length_at_1_point_5_l1420_142097

-- Definitions for the conditions
def Point := ℝ × ℝ
def Triangle (A B C : Point) := ∃ a b c : ℝ, a = 4 ∧ b = 3 ∧ c = 5 ∧ (A = (0, 0)) ∧ (B = (4, 0)) ∧ (C = (0, 3)) ∧ (c^2 = a^2 + b^2)

noncomputable def length_l (x : ℝ) : ℝ := (4 * (abs ((3/4) * x + 3))) / 5

theorem segment_length_at_1_point_5 (A B C : Point) (h : Triangle A B C) : 
  length_l 1.5 = 3.3 := by 
  sorry

end segment_length_at_1_point_5_l1420_142097


namespace whitney_spent_179_l1420_142094

def total_cost (books_whales books_fish magazines book_cost magazine_cost : ℕ) : ℕ :=
  (books_whales + books_fish) * book_cost + magazines * magazine_cost

theorem whitney_spent_179 :
  total_cost 9 7 3 11 1 = 179 :=
by
  sorry

end whitney_spent_179_l1420_142094


namespace base_of_minus4_pow3_l1420_142052

theorem base_of_minus4_pow3 : ∀ (x : ℤ) (n : ℤ), (x, n) = (-4, 3) → x = -4 :=
by intros x n h
   cases h
   rfl

end base_of_minus4_pow3_l1420_142052


namespace circle_O₁_equation_sum_of_squares_constant_l1420_142008

-- Given conditions
def circle_O (x y : ℝ) := x^2 + y^2 = 25
def center_O₁ (m : ℝ) : ℝ × ℝ := (m, 0) 
def intersect_point := (3, 4)
def is_intersection (x y : ℝ) := circle_O x y ∧ (x - intersect_point.1)^2 + (y - intersect_point.2)^2 = 0
def line_passing_P (k : ℝ) (x y : ℝ) := y - intersect_point.2 = k * (x - intersect_point.1)
def point_on_circle (circle : ℝ × ℝ → Prop) (x y : ℝ) := circle (x, y)
def distance_squared (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Problem statements
theorem circle_O₁_equation (k : ℝ) (m : ℝ) (x y : ℝ) (h : k = 1) (h_intersect: is_intersection 3 4)
  (h_BP_distance : distance_squared (3, 4) (x, y) = (7 * Real.sqrt 2)^2) : 
  (x - 14)^2 + y^2 = 137 := sorry

theorem sum_of_squares_constant (k m : ℝ) (h : k ≠ 0) (h_perpendicular : line_passing_P (-1/k) 3 4)
  (A B C D : ℝ × ℝ) (h_AB_distance : distance_squared A B = 4 * m^2 / (1 + k^2)) 
  (h_CD_distance : distance_squared C D = 4 * m^2 * k^2 / (1 + k^2)) : 
  distance_squared A B + distance_squared C D = 4 * m^2 := sorry

end circle_O₁_equation_sum_of_squares_constant_l1420_142008


namespace price_of_n_kilograms_l1420_142066

theorem price_of_n_kilograms (m n : ℕ) (hm : m ≠ 0) (h : 9 = m) : (9 * n) / m = (9 * n) / m :=
by
  sorry

end price_of_n_kilograms_l1420_142066


namespace xiao_cong_math_score_l1420_142023

theorem xiao_cong_math_score :
  ∀ (C M E : ℕ),
    (C + M + E) / 3 = 122 → C = 118 → E = 125 → M = 123 :=
by
  intros C M E h1 h2 h3
  sorry

end xiao_cong_math_score_l1420_142023


namespace minimize_f_at_a_l1420_142036

def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f_at_a (a : ℝ) (h : a = 82 / 43) :
  ∃ x, ∀ y, f x a ≤ f y a :=
sorry

end minimize_f_at_a_l1420_142036


namespace num_five_letter_words_correct_l1420_142003

noncomputable def num_five_letter_words : ℕ := 1889568

theorem num_five_letter_words_correct :
  let a := 3
  let e := 4
  let i := 2
  let o := 5
  let u := 4
  (a + e + i + o + u) ^ 5 = num_five_letter_words :=
by
  sorry

end num_five_letter_words_correct_l1420_142003


namespace physics_majors_consecutive_probability_l1420_142075

open Nat

-- Define the total number of seats and the specific majors
def totalSeats : ℕ := 10
def mathMajors : ℕ := 4
def physicsMajors : ℕ := 3
def chemistryMajors : ℕ := 2
def biologyMajors : ℕ := 1

-- Assuming a round table configuration
def probabilityPhysicsMajorsConsecutive : ℚ :=
  (3 * (Nat.factorial (totalSeats - physicsMajors))) / (Nat.factorial (totalSeats - 1))

-- Declare the theorem
theorem physics_majors_consecutive_probability : 
  probabilityPhysicsMajorsConsecutive = 1 / 24 :=
by
  sorry

end physics_majors_consecutive_probability_l1420_142075


namespace purchasing_options_count_l1420_142044

theorem purchasing_options_count : ∃ (s : Finset (ℕ × ℕ)), s.card = 4 ∧
  ∀ (a : ℕ × ℕ), a ∈ s ↔ 
    (80 * a.1 + 120 * a.2 = 1000) 
    ∧ (a.1 > 0) ∧ (a.2 > 0) :=
by
  sorry

end purchasing_options_count_l1420_142044


namespace stadium_length_l1420_142059

theorem stadium_length
  (W : ℝ) (H : ℝ) (P : ℝ) (L : ℝ)
  (h1 : W = 18)
  (h2 : H = 16)
  (h3 : P = 34)
  (h4 : P^2 = L^2 + W^2 + H^2) :
  L = 24 :=
by
  sorry

end stadium_length_l1420_142059


namespace arithmetic_geometric_inequality_l1420_142060

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
by
  sorry

end arithmetic_geometric_inequality_l1420_142060


namespace extreme_value_0_at_minus_1_l1420_142061

theorem extreme_value_0_at_minus_1 (m n : ℝ)
  (h1 : (-1) + 3 * m - n + m^2 = 0)
  (h2 : 3 - 6 * m + n = 0) :
  m + n = 11 :=
sorry

end extreme_value_0_at_minus_1_l1420_142061


namespace single_elimination_games_l1420_142043

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ g : ℕ, g = n - 1 :=
by
  sorry

end single_elimination_games_l1420_142043


namespace initial_men_in_camp_l1420_142006

theorem initial_men_in_camp (M F : ℕ) 
  (h1 : F = M * 50)
  (h2 : F = (M + 10) * 25) : 
  M = 10 :=
by
  sorry

end initial_men_in_camp_l1420_142006


namespace exists_separating_line_l1420_142085

noncomputable def f1 (x : ℝ) (a1 b1 c1 : ℝ) : ℝ := a1 * x^2 + b1 * x + c1
noncomputable def f2 (x : ℝ) (a2 b2 c2 : ℝ) : ℝ := a2 * x^2 + b2 * x + c2

theorem exists_separating_line (a1 b1 c1 a2 b2 c2 : ℝ) (h_intersect : ∀ x, f1 x a1 b1 c1 ≠ f2 x a2 b2 c2)
  (h_neg : a1 * a2 < 0) : ∃ α β : ℝ, ∀ x, f1 x a1 b1 c1 < α * x + β ∧ α * x + β < f2 x a2 b2 c2 :=
sorry

end exists_separating_line_l1420_142085


namespace find_tan_G_l1420_142028

def right_triangle (FG GH FH : ℕ) : Prop :=
  FG^2 = GH^2 + FH^2

def tan_ratio (GH FH : ℕ) : ℚ :=
  FH / GH

theorem find_tan_G
  (FG GH : ℕ)
  (H1 : FG = 13)
  (H2 : GH = 12)
  (FH : ℕ)
  (H3 : right_triangle FG GH FH) :
  tan_ratio GH FH = 5 / 12 :=
by
  sorry

end find_tan_G_l1420_142028


namespace lcm_of_12_and_15_l1420_142002
-- Import the entire Mathlib library

-- Define the given conditions
def HCF (a b : ℕ) : ℕ := gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / (gcd a b)

-- Given the values
def a := 12
def b := 15
def hcf := 3

-- State the proof problem
theorem lcm_of_12_and_15 : LCM a b = 60 :=
by
  -- Proof goes here (skipped)
  sorry

end lcm_of_12_and_15_l1420_142002


namespace prime_divides_3np_minus_3n1_l1420_142045

theorem prime_divides_3np_minus_3n1 (p n : ℕ) (hp : Prime p) : p ∣ (3^(n + p) - 3^(n + 1)) :=
sorry

end prime_divides_3np_minus_3n1_l1420_142045


namespace sequence_a2002_l1420_142089

theorem sequence_a2002 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (a 2 = 2) → 
  (∀ n, 2 ≤ n → a (n + 1) = 3 * a n - 2 * a (n - 1)) → 
  a 2002 = 2 ^ 2001 :=
by
  intros a ha1 ha2 hrecur
  sorry

end sequence_a2002_l1420_142089


namespace problem_1_problem_2_l1420_142024

def is_in_solution_set (x : ℝ) : Prop := -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0

variables {a b : ℝ}

theorem problem_1 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |(1 / 3) * a + (1 / 6) * b| < 1 / 4 :=
sorry

theorem problem_2 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end problem_1_problem_2_l1420_142024


namespace original_cost_price_l1420_142005

-- Define the conditions
def selling_price : ℝ := 24000
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.02
def profit_rate : ℝ := 0.12

-- Define the necessary calculations
def discounted_price (sp : ℝ) (dr : ℝ) : ℝ := sp * (1 - dr)
def total_tax (sp : ℝ) (tr : ℝ) : ℝ := sp * tr
def profit (c : ℝ) (pr : ℝ) : ℝ := c * (1 + pr)

-- The problem is to prove that the original cost price is $17,785.71
theorem original_cost_price : 
  ∃ (C : ℝ), C = 17785.71 ∧ 
  selling_price * (1 - discount_rate - tax_rate) = (1 + profit_rate) * C :=
sorry

end original_cost_price_l1420_142005


namespace sufficient_but_not_necessary_condition_l1420_142081

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a < b) : 
  ((a - b) * a^2 < 0) ↔ (a < b) :=
sorry

end sufficient_but_not_necessary_condition_l1420_142081


namespace trajectory_of_center_l1420_142078

theorem trajectory_of_center :
  ∃ (x y : ℝ), (x + 1) ^ 2 + y ^ 2 = 49 / 4 ∧ (x - 1) ^ 2 + y ^ 2 = 1 / 4 ∧ ( ∀ P, (P = (x, y) → (P.1^2) / 4 + (P.2^2) / 3 = 1) ) := sorry

end trajectory_of_center_l1420_142078


namespace option_C_correct_l1420_142071

theorem option_C_correct (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := 
by
  sorry

end option_C_correct_l1420_142071


namespace no_third_quadrant_l1420_142093

def quadratic_no_real_roots (b : ℝ) : Prop :=
  16 - 4 * b < 0

def passes_through_third_quadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = -2 * x + b ∧ x < 0 ∧ y < 0

theorem no_third_quadrant (b : ℝ) (h : quadratic_no_real_roots b) : ¬ passes_through_third_quadrant b := 
by {
  sorry
}

end no_third_quadrant_l1420_142093


namespace find_x_l1420_142016

theorem find_x (x y : ℕ) (h1 : x / y = 6 / 3) (h2 : y = 27) : x = 54 :=
sorry

end find_x_l1420_142016


namespace percentage_area_covered_by_pentagons_l1420_142034

theorem percentage_area_covered_by_pentagons :
  ∀ (a : ℝ), (∃ (large_square_area small_square_area pentagon_area : ℝ),
    large_square_area = 16 * a^2 ∧
    small_square_area = a^2 ∧
    pentagon_area = 10 * small_square_area ∧
    (pentagon_area / large_square_area) * 100 = 62.5) :=
sorry

end percentage_area_covered_by_pentagons_l1420_142034


namespace problem_solution_l1420_142053

noncomputable def solution_set : Set ℝ :=
  { x : ℝ | x ∈ (Set.Ioo 0 (5 - Real.sqrt 10)) ∨ x ∈ (Set.Ioi (5 + Real.sqrt 10)) }

theorem problem_solution (x : ℝ) : (x^3 - 10*x^2 + 15*x > 0) ↔ x ∈ solution_set :=
by
  sorry

end problem_solution_l1420_142053


namespace negation_of_proposition_l1420_142000

theorem negation_of_proposition :
    (¬ ∃ (x : ℝ), (Real.exp x - x - 1 < 0)) ↔ (∀ (x : ℝ), Real.exp x - x - 1 ≥ 0) :=
by
  sorry

end negation_of_proposition_l1420_142000


namespace unique_ordered_triple_l1420_142017

theorem unique_ordered_triple (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ab : Nat.lcm a b = 500) (h_bc : Nat.lcm b c = 2000) (h_ca : Nat.lcm c a = 2000) :
  (a = 100 ∧ b = 2000 ∧ c = 2000) :=
by
  sorry

end unique_ordered_triple_l1420_142017


namespace equivalent_contrapositive_l1420_142011

-- Given definitions
variables {Person : Type} (possess : Person → Prop) (happy : Person → Prop)

-- The original statement: "If someone is happy, then they possess it."
def original_statement : Prop := ∀ p : Person, happy p → possess p

-- The contrapositive: "If someone does not possess it, then they are not happy."
def contrapositive_statement : Prop := ∀ p : Person, ¬ possess p → ¬ happy p

-- The theorem to prove logical equivalence
theorem equivalent_contrapositive : original_statement possess happy ↔ contrapositive_statement possess happy := 
by sorry

end equivalent_contrapositive_l1420_142011


namespace solve_for_x_l1420_142079

noncomputable def n : ℝ := Real.sqrt (7^2 + 24^2)
noncomputable def d : ℝ := Real.sqrt (49 + 16)
noncomputable def x : ℝ := n / d

theorem solve_for_x : x = 5 * Real.sqrt 65 / 13 := by
  sorry

end solve_for_x_l1420_142079


namespace arc_length_of_octagon_side_l1420_142047

-- Define the conditions
def is_regular_octagon (side_length : ℝ) (angle_subtended : ℝ) := side_length = 5 ∧ angle_subtended = 2 * Real.pi / 8

-- Define the property to be proved
theorem arc_length_of_octagon_side :
  ∀ (side_length : ℝ) (angle_subtended : ℝ), 
    is_regular_octagon side_length angle_subtended →
    (angle_subtended / (2 * Real.pi)) * (2 * Real.pi * side_length) = 5 * Real.pi / 4 :=
by
  intros side_length angle_subtended h
  unfold is_regular_octagon at h
  sorry

end arc_length_of_octagon_side_l1420_142047


namespace molly_age_l1420_142083

theorem molly_age
  (S M : ℕ)
  (h1 : S / M = 4 / 3)
  (h2 : S + 6 = 30) :
  M = 18 :=
sorry

end molly_age_l1420_142083


namespace rectangle_area_l1420_142039

theorem rectangle_area (side_of_square := 45)
  (radius_of_circle := side_of_square)
  (length_of_rectangle := (2/5 : ℚ) * radius_of_circle)
  (breadth_of_rectangle := 10) :
  breadth_of_rectangle * length_of_rectangle = 180 := 
by
  sorry

end rectangle_area_l1420_142039
