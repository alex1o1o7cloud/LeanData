import Mathlib

namespace triangle_angle_82_82_degrees_l809_809370

noncomputable def triangle_angle_sides (a b c : ℝ) : ℝ := 
  real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) * (180 / real.pi)

theorem triangle_angle_82_82_degrees :
  triangle_angle_sides 2 2 (real.sqrt 7) = 82.82 :=
by 
  -- We fill in the proof here or leave it as sorry for now
  sorry

end triangle_angle_82_82_degrees_l809_809370


namespace angle_KSN_eq_angle_MSL_l809_809772

open EuclideanGeometry

variables {A B C D K L M N S : Point}

-- Definitions of the midpoints and given conditions
def is_midpoint (P X Y : Point) : Prop := dist P X = dist P Y ∧ on_line_segment P X Y

def is_parallelogram (P Q R S : Point) : Prop :=
  par P Q R S ∨ par P R Q S

-- The theorem to be proved
theorem angle_KSN_eq_angle_MSL
  (hK : is_midpoint K A B)
  (hL : is_midpoint L B C)
  (hM : is_midpoint M C D)
  (hN : is_midpoint N D A)
  (hKSLS : dist K S = dist L S)
  (hNSMS : dist N S = dist M S) :
  angle K S N = angle M S L :=
begin
  sorry
end

end angle_KSN_eq_angle_MSL_l809_809772


namespace angle_between_hour_and_minute_hand_at_5_oclock_l809_809728

theorem angle_between_hour_and_minute_hand_at_5_oclock : 
  let degrees_in_circle := 360
  let hours_in_clock := 12
  let angle_per_hour := degrees_in_circle / hours_in_clock
  let hour_hand_position := 5
  let minute_hand_position := 0
  let angle := (hour_hand_position - minute_hand_position) * angle_per_hour
  angle = 150 :=
by sorry

end angle_between_hour_and_minute_hand_at_5_oclock_l809_809728


namespace radius_B_l809_809243

noncomputable def radius_A := 2
noncomputable def radius_D := 4

theorem radius_B (r_B : ℝ) (x y : ℝ) 
  (h1 : (2 : ℝ) + y = x + (x^2 / 4)) 
  (h2 : y = 2 - (x^2 / 8)) 
  (h3 : x = (4: ℝ) / 3) 
  (h4 : y = x + (x^2 / 4)) : r_B = 20 / 9 :=
sorry

end radius_B_l809_809243


namespace fox_catches_hares_if_speed_sufficient_l809_809993

open Real

noncomputable def fox_can_catch_hares (v : ℝ) : Prop :=
  ∀ (t : ℝ), t ≥ 0 →
    (∃ x_B y_B x_D y_D : ℝ, 
      (x_B = 0 ∧ 0 < y_B ∧ y_B ≤ 1 + t) ∨ 
      (y_D = 0 ∧ 0 < x_D ∧ x_D ≤ 1 + t))
  
theorem fox_catches_hares_if_speed_sufficient (v : ℝ) (h1 : v ≥ 1 + √2) : fox_can_catch_hares v :=
sorry

end fox_catches_hares_if_speed_sufficient_l809_809993


namespace volume_of_right_pyramid_l809_809213

noncomputable def square_base_area : ℝ := 120
noncomputable def surface_area : ℝ := 600
noncomputable def num_sides : ℝ := 5
noncomputable def side_length (area : ℝ) : ℝ := real.sqrt (area)
noncomputable def tri_area (num_sides : ℝ) (square_base_area : ℝ) : ℝ := 2 * square_base_area
noncomputable def height_pm (tri_area side_length : ℝ) : ℝ := (2 * tri_area) / side_length
noncomputable def height_pf (height_pm side_length : ℝ) : ℝ := real.sqrt (height_pm ^ 2 - (side_length / 2) ^ 2)
noncomputable def pyramid_volume (square_base_area height_pf : ℝ) : ℝ := (1 / 3) * square_base_area * height_pf

theorem volume_of_right_pyramid : 
    pyramid_volume square_base_area (height_pf (height_pm (tri_area num_sides square_base_area) (side_length square_base_area)) (side_length square_base_area)) = 40 * real.sqrt 210 := 
    by sorry

end volume_of_right_pyramid_l809_809213


namespace number_of_sets_C_l809_809701

def A : Set ℕ := {x : ℕ | x^2 - 3 * x + 2 = 0}
def B : Set ℕ := {x : ℕ | 0 < x ∧ x < 6}

theorem number_of_sets_C (hA : A = {1, 2}) (hB : B = {1, 2, 3, 4, 5}) :
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B}.card = 8 :=
by sorry

end number_of_sets_C_l809_809701


namespace find_a_l809_809817

def f (x : ℝ) : ℝ := if x ≤ 0 then -4 * x else x^2

theorem find_a (a : ℝ) : f a = 4 → (a = -1 ∨ a = 2) := by
  sorry

end find_a_l809_809817


namespace davida_hours_difference_l809_809629

theorem davida_hours_difference : 
  let hours_week_1 := 35 in
  let hours_week_2 := 35 in
  let total_hours_weeks_1_2 := hours_week_1 + hours_week_2 in
  let hours_week_3 := 48 in
  let hours_week_4 := 48 in
  let total_hours_weeks_3_4 := hours_week_3 + hours_week_4 in
  total_hours_weeks_3_4 - total_hours_weeks_1_2 = 26 := 
by
  sorry

end davida_hours_difference_l809_809629


namespace number_of_terms_in_arithmetic_sequence_l809_809636

-- Define the first term, common difference, and the nth term of the sequence
def a : ℤ := -3
def d : ℤ := 4
def a_n : ℤ := 45

-- Define the number of terms in the arithmetic sequence
def num_of_terms : ℤ := 13

-- The theorem states that for the given arithmetic sequence, the number of terms n satisfies the sequence equation
theorem number_of_terms_in_arithmetic_sequence :
  a + (num_of_terms - 1) * d = a_n :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l809_809636


namespace find_a_l809_809343

def orthogonal (v w : List ℝ) : Prop := 
  v.zipWith (λ x y => x * y) w |>.sum = 0

def vec_minus (v w : List ℝ) : List ℝ :=
  v.zipWith (λ x y => x - y) w

def vec_scalar_mul (a : ℝ) (v : List ℝ) : List ℝ :=
  v.map (λ x => a * x)

def vec_add (v w : List ℝ) : List ℝ :=
  v.zipWith (λ x y => x + y) w

theorem find_a : 
  let m := [-2, 1]
  let n := [1, 1]
  orthogonal (vec_minus m (vec_scalar_mul 2 n)) (vec_add (vec_scalar_mul a m) n) → a = 5/7 := 
by
  let m := [-2, 1]
  let n := [1, 1]
  sorry

end find_a_l809_809343


namespace chocolate_bar_cost_l809_809023

theorem chocolate_bar_cost :
  ∀ (total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips : ℕ),
  total = 150 →
  gummy_bear_cost = 2 →
  chocolate_chip_cost = 5 →
  num_chocolate_bars = 10 →
  num_gummy_bears = 10 →
  num_chocolate_chips = 20 →
  ((total - (num_gummy_bears * gummy_bear_cost + num_chocolate_chips * chocolate_chip_cost)) / num_chocolate_bars = 3) := 
by
  intros total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips 
  intros htotal hgb_cost hcc_cost hncb hngb hncc
  sorry

end chocolate_bar_cost_l809_809023


namespace no_adjacent_girls_arrangement_l809_809193

theorem no_adjacent_girls_arrangement :
  let boys := 4 in
  let girls := 4 in
  let totalWaysToArrangeBoys := Nat.factorial boys in
  let totalWaysToInsertGirls := (Nat.factorial (boys + 1)) / (Nat.factorial (boys + 1 - girls)) in
  let totalArrangements := totalWaysToArrangeBoys * totalWaysToInsertGirls in
  totalArrangements = 2880 :=
by
  sorry

end no_adjacent_girls_arrangement_l809_809193


namespace part1_part2_l809_809691

-- Define the linear function
def linear_function (m x : ℝ) : ℝ := (m - 2) * x + 6

-- Prove part 1: If y increases as x increases, then m > 2
theorem part1 (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → linear_function m x1 < linear_function m x2) → m > 2 :=
sorry

-- Prove part 2: When -2 ≤ x ≤ 4, and y ≤ 10, the range of m is (2, 3] or [0, 2)
theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → linear_function m x ≤ 10) →
  (2 < m ∧ m ≤ 3) ∨ (0 ≤ m ∧ m < 2) :=
sorry

end part1_part2_l809_809691


namespace cars_produced_per_month_l809_809573

-- Define the necessary variables and conditions
variables (C : ℕ) -- The number of cars produced per month
def material_cost_cars : ℕ := 100
def price_per_car : ℕ := 50
def material_cost_motorcycles : ℕ := 250
def motorcycle_revenue : ℕ := 8 * 50
def additional_profit : ℕ := 50

-- Formalize the problem in a proof statement
theorem cars_produced_per_month :
  let cars_profit := price_per_car * C - material_cost_cars in
  let motorcycles_profit := motorcycle_revenue - material_cost_motorcycles in
  motorcycles_profit - cars_profit = additional_profit →
  C = 4 :=
by
  intros h
  sorry

end cars_produced_per_month_l809_809573


namespace total_surface_area_two_parts_l809_809219

def AI (cube_side : ℝ) : ℝ := 4
def DL (cube_side : ℝ) : ℝ := 4
def JF (cube_side : ℝ) : ℝ := 3
def KG (cube_side : ℝ) : ℝ := 3
def IJ (cube_side MI MJ : ℝ) : ℝ := Real.sqrt (MI^2 + MJ^2)
def rectangle_area (length width : ℝ) : ℝ := length * width

theorem total_surface_area_two_parts
  (cube_side : ℝ)
  (h_cube_side : cube_side = 12)
  (AI_eq_DL : AI cube_side = 4)
  (DL_eq_consts : DL cube_side = 4)
  (JF_eq_consts : JF cube_side = 3)
  (KG_eq_consts : KG cube_side = 3)
  (IJKL_rectangle : True) :
  6 * cube_side^2 + 2 * (rectangle_area 12 (IJ 12 12 5)) = 1176 :=
by
  sorry

end total_surface_area_two_parts_l809_809219


namespace complex_div_l809_809285

theorem complex_div (a b c d : ℂ) (h : a = 2 ∧ b = -1 ∧ c = 2 ∧ d = 1) :
  (a + b * complex.i) / (c + d * complex.i) = 3/5 - 4/5 * complex.i :=
by
  rcases h with ⟨ha, hb, hc, hd⟩
  sorry

end complex_div_l809_809285


namespace inequality_proof_l809_809718

noncomputable def f (x m : ℝ) : ℝ := 2 * m * x - Real.log x

theorem inequality_proof (m x₁ x₂ : ℝ) (hm : m ≥ -1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hineq : (f x₁ m + f x₂ m) / 2 ≤ x₁ ^ 2 + x₂ ^ 2 + (3 / 2) * x₁ * x₂) :
  x₁ + x₂ ≥ (Real.sqrt 3 - 1) / 2 := 
sorry

end inequality_proof_l809_809718


namespace gordon_total_cost_l809_809518

def discount_30 (price : ℝ) : ℝ :=
  price - 0.30 * price

def discount_20 (price : ℝ) : ℝ :=
  price - 0.20 * price

def discounted_price (price : ℝ) : ℝ :=
  if price > 22.00 then discount_30 price
  else if price < 20.00 then discount_20 price
  else price

def total_cost : ℝ :=
  discounted_price 25.00 +
  discounted_price 18.00 +
  discounted_price 21.00 +
  discounted_price 35.00 +
  discounted_price 12.00 +
  discounted_price 10.00

theorem gordon_total_cost : total_cost = 95.00 := 
  by
  sorry

end gordon_total_cost_l809_809518


namespace num_real_solutions_frac_sine_l809_809656

theorem num_real_solutions_frac_sine :
  (∃ n : ℕ, ∀ x : ℝ, x ∈ Icc (-150) 150 → (x/150 = Real.sin x) ↔ (n = 95)) := 
sorry

end num_real_solutions_frac_sine_l809_809656


namespace four_digit_sum_divisible_by_5_l809_809349

theorem four_digit_sum_divisible_by_5 :
  ∃ (a b c d : ℕ), 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    1000 * a + 100 * b + 10 * c + d ≥ 1000 ∧
    a + b + c + d = 12 ∧ 
    d ∈ {0, 5} :=
by
  sorry

end four_digit_sum_divisible_by_5_l809_809349


namespace floored_square_of_T_l809_809047

-- Define the summation condition T
def T : ℝ := ∑ j in Finset.range (1007 + 1) \ Finset.range 2, real.sqrt (1 + (1 : ℝ) / (j : ℝ) ^ 2 + (1 : ℝ) / (j + 1) ^ 2)

-- State and prove the main theorem
theorem floored_square_of_T : int.floor (T ^ 2) = 1013008 :=
by
  sorry

end floored_square_of_T_l809_809047


namespace count_values_of_x_satisfying_g_g_x_2_l809_809118

noncomputable def g : ℝ → ℝ := sorry

-- Given conditions: 
axiom g_at_neg3 : g (-3) = 2
axiom g_at_0 : g (0) = 2
axiom g_at_4 : g (4) = 2

-- Additional conditions inferred from the solution:
axiom g_no_x_for_neg3 : ∀ x, g x ≠ -3
axiom g_at_neg1 : g (-1) = 0
axiom g_at_3 : g (3) = 4

-- Prove that there are exactly 2 values of x such that g(g(x)) = 2
theorem count_values_of_x_satisfying_g_g_x_2 : 
  (∃ a b : ℝ, a ≠ b ∧ g (g a) = 2 ∧ g (g b) = 2) ∧ (∀ y ≠ a ∧ y ≠ b, g (g y) ≠ 2) :=
sorry

end count_values_of_x_satisfying_g_g_x_2_l809_809118


namespace equal_distances_l809_809812

-- Definitions as per the problem statement
variables {ABC A'B'C' : Type} [triangle ABC] [triangle A'B'C']
variables {A B C A' B' C' O H H' : Point}

-- Conditions
axiom directly_similar (ABC A'B'C' : Type) : Prop
axiom A_on_B'C' : A ∈ line_segment B' C'
axiom B_on_C'A' : B ∈ line_segment C' A'
axiom C_on_A'B' : C ∈ line_segment A' B'
axiom circumcenter (t : Type) : Point
axiom orthocenter (t : Type) : Point
axiom O_is_circumcenter : O = circumcenter ABC
axiom H_is_orthocenter : H = orthocenter ABC
axiom H'_is_orthocenter : H' = orthocenter A'B'C'

-- Theorem to be proven
theorem equal_distances (t1 t2 : Type) 
  [h1 : triangle t1] [h2 : triangle t2] 
  (H_t1 : t1 = ABC) (H_t2 : t2 = A'B'C')
  (O : Point) (H : Point) (H' : Point)
  (O_def : O = circumcenter ABC)
  (H_def : H = orthocenter ABC)
  (H'_def : H' = orthocenter A'B'C') :
  dist O H = dist O H' :=
by
  sorry

end equal_distances_l809_809812


namespace angle_RIS_acute_l809_809806

variable {A B C I K L M R S : Point}
variable {α β γ δ : Angle}

-- Definitions of points and tangencies
def incenter (A B C I : Point) := circle_tangent_to_triangle_sides A B C I K L M
def parallel (P Q R S : Point) := line_through P translate Q R S

-- Core statement to be proved
theorem angle_RIS_acute
  (h_incenter : incenter A B C I)
  (h_tangent1 : tangent_point I K B C)
  (h_tangent2 : tangent_point I L C A)
  (h_tangent3 : tangent_point I M A B)
  (h_parallel_RS_MK : parallel B R S M K) :
  ∠ R I S < 90 :=
sorry

end angle_RIS_acute_l809_809806


namespace find_b_l809_809753

-- Define the conditions of the problem
def cosA := 5 / 13
def sinB := 3 / 5
def a := 20

-- Prove that b = 13 given the conditions
theorem find_b :
  ∃ b, cosA = 5 / 13 ∧ sinB = 3 / 5 ∧ a = 20 ∧ b = 13 :=
by
  -- Placeholder for the proof
  use 13
  simp [cosA, sinB, a]
  split; exact rfl

end find_b_l809_809753


namespace goods_train_length_l809_809987

theorem goods_train_length (speed_kmph : ℕ) (platform_length : ℕ) (crossing_time : ℕ)
  (H1 : speed_kmph = 72)
  (H2 : platform_length = 270)
  (H3 : crossing_time = 26) :
  let speed_mps := (speed_kmph * 5) / 18 in
  let total_distance := speed_mps * crossing_time in
  let train_length := total_distance - platform_length in
  train_length = 250 := by
  rw [H1, H2, H3]
  -- Intermediate steps can be shown here
  sorry

end goods_train_length_l809_809987


namespace common_internal_tangent_length_l809_809473

theorem common_internal_tangent_length
  (d r1 r2 : ℝ) 
  (h1 : d = 45)
  (h2 : r1 = 7)
  (h3 : r2 = 6) : 
  sqrt(d^2 - (r1 + r2)^2) = 43 := 
by {
  sorry
}

end common_internal_tangent_length_l809_809473


namespace percentage_of_children_with_both_colors_l809_809986

theorem percentage_of_children_with_both_colors (F : ℕ) (h_even : F % 2 = 0) :
  let C := F / 2 in
  60 * C / 100 + 50 * C / 100 - 100 * C / 100 = 10 * C / 100 :=
by
  -- This 'sorry' represents where the proof would go
  sorry

end percentage_of_children_with_both_colors_l809_809986


namespace find_k_l809_809012

variables (P Q R S : Type) [trapezium : Trapezium P Q R S]

def angle_PQR : ℝ
def angle_SPQ : ℝ := 2 * angle_PQR
def angle_RSP : ℝ := 2 * angle_SPQ
def angle_QRS : ℝ := k * angle_PQR

theorem find_k :
  angle_PQR + angle_SPQ + angle_RSP + angle_QRS = 360 ∧
  6 * angle_PQR = 180 ∧
  angle_PQR ≠ 0 → 
  k = 5 :=
by
  sorry

end find_k_l809_809012


namespace no_positive_integer_has_product_as_perfect_square_l809_809647

theorem no_positive_integer_has_product_as_perfect_square:
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n * (n + 1) = k * k :=
by
  sorry

end no_positive_integer_has_product_as_perfect_square_l809_809647


namespace possible_dice_labels_l809_809183

theorem possible_dice_labels : 
  ∃ (die1 : Fin 6 → Nat) (die2 : Fin 6 → Nat), 
  (∀ k ∈ (Finset.range 1 37), ∃ i j, k = die1 i + die2 j) :=
by
  sorry

end possible_dice_labels_l809_809183


namespace prob_first_red_third_expected_num_red_lights_l809_809049

variables (p1 p2 p3 : ℝ) (X : ℕ → ℝ)
variables (cond1 : p1 = 1/2)
variables (cond2 : p2 > p1 ∧ p3 > p2)
variables (cond3 : (1 - p1) * (1 - p2) * (1 - p3) = 1/24)
variables (cond4 : p1 * p2 * p3 = 1/4)
variables (cond5 : ∀ n m, n ≠ m → ∃ (EventIndependent n m))

-- Part 1
theorem prob_first_red_third : (p1 * (1 - p2) * p3) = 1/8 := 
by {
  sorry
}

-- Part 2
theorem expected_num_red_lights : (Expectation X) = 23/12 := 
by {
  sorry
}

end prob_first_red_third_expected_num_red_lights_l809_809049


namespace smallest_area_triangle_l809_809309

-- Definition of the angular domain and point inside it
variables {a b : ℝ → Prop} -- sides of the angle
variables (P : {x : ℝ × ℝ // x ∈ a ∧ x ∈ b}) -- point inside the angular domain

-- Statement that the line through P which bisects the segment
-- between intersection points with sides a and b creates the triangle with the smallest area
theorem smallest_area_triangle :
  ∃ (l : ℝ → ℝ), (P ∈ l) ∧
  (∀ l', (P ∈ l') → area (triangle (a ∩ l') (b ∩ l')) ≥ area (triangle (a ∩ l) (b ∩ l))) :=
sorry

end smallest_area_triangle_l809_809309


namespace dusty_change_l809_809223

theorem dusty_change :
  let single_layer_price := 4
  let double_layer_price := 7
  let num_single_layers := 7
  let num_double_layers := 5
  let total_payment := 100
  let total_cost := (single_layer_price * num_single_layers) + (double_layer_price * num_double_layers)
  let change := total_payment - total_cost
  change = 37 :=
by
  -- Definitions only, proof not required.
  let single_layer_price := 4
  let double_layer_price := 7
  let num_single_layers := 7
  let num_double_layers := 5
  let total_payment := 100
  let total_cost := (single_layer_price * num_single_layers) + (double_layer_price * num_double_layers)
  let change := total_payment - total_cost
  have h₁ : total_cost = 63, by sorry
  have h₂ : change = 37, by sorry
  exact h₂

end dusty_change_l809_809223


namespace sum_of_cosines_l809_809889

theorem sum_of_cosines (x1 x2 x3 p q r : ℝ) (h_root : polynomial.eval x1 (polynomial.C r + polynomial.X * (polynomial.C q + polynomial.X * (polynomial.C (-p) + polynomial.X * polynomial.C 1))) = 0)
  (h_root2 : polynomial.eval x2 (polynomial.C r + polynomial.X * (polynomial.C q + polynomial.X * (polynomial.C (-p) + polynomial.X * polynomial.C 1))) = 0)
  (h_root3 : polynomial.eval x3 (polynomial.C r + polynomial.X * (polynomial.C q + polynomial.X * (polynomial.C (-p) + polynomial.X * polynomial.C 1))) = 0)
  (h_triangle : x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ (x1 + x2 > x3) ∧ (x2 + x3 > x1) ∧ (x3 + x1 > x2)) :
  (cos (real.arccos ((x2^2 + x3^2 - x1^2) / (2 * x2 * x3))) + 
   cos (real.arccos ((x1^2 + x3^2 - x2^2) / (2 * x1 * x3))) + 
   cos (real.arccos ((x1^2 + x2^2 - x3^2) / (2 * x1 * x2)))) = 
   (4 * p * q - 6 * r - p^3) / (2 * r) :=
sorry

end sum_of_cosines_l809_809889


namespace benjamin_earns_more_l809_809426

noncomputable def additional_earnings : ℝ :=
  let P : ℝ := 75000
  let r : ℝ := 0.05
  let t_M : ℝ := 3
  let r_m : ℝ := r / 12
  let t_B : ℝ := 36
  let A_M : ℝ := P * (1 + r)^t_M
  let A_B : ℝ := P * (1 + r_m)^t_B
  A_B - A_M

theorem benjamin_earns_more : additional_earnings = 204 := by
  sorry

end benjamin_earns_more_l809_809426


namespace radius_of_inscribed_circle_eq_r_l809_809912

-- Define the circles and their properties
noncomputable def ωx (r : ℝ) : circle := sorry  -- definition of circle ωx with radius r
noncomputable def ωy (ry : ℝ) (r : ℝ) : circle := sorry  -- definition of circle ωy with radius ry > r
noncomputable def ωz (r : ℝ) : circle := sorry  -- definition of circle ωz with radius r

-- Define the tangents and their relations
def tangent_point (c : circle) : point := sorry  -- tangent point definition

def t : line := sorry  -- common tangent line definition
def X := tangent_point (ωx r)
def Y := midpoint (tangent_point (ωx r)) (tangent_point (ωz r))
def Z := tangent_point (ωz r)

def p : line := common_internal_tangent (ωx r) (ωy ry r)
def q : line := common_internal_tangent (ωy ry r) (ωz r)

-- Define the triangle and its properties
def triangle_formed := triangle p q t

-- Define the proof statement
theorem radius_of_inscribed_circle_eq_r (r : ℝ) (ry : ℝ) (h1 : ry > r) :
  radius (inscribed_circle (triangle_formed)) = r :=
sorry

end radius_of_inscribed_circle_eq_r_l809_809912


namespace third_number_is_58_l809_809107

theorem third_number_is_58 
  (numbers : List ℕ) 
  (h1 : numbers = [54, 55, 58, 59, 62, 62, 63, 65, 65])
  (h2 : numbers.Average = 60) : 
  numbers.nth 2 = some 58 := 
by 
  rw [h1] 
  simp 
  sorry

end third_number_is_58_l809_809107


namespace dusty_change_l809_809224

noncomputable def single_layer_price : ℕ := 4
noncomputable def double_layer_price : ℕ := 7
noncomputable def single_layer_quantity : ℕ := 7
noncomputable def double_layer_quantity : ℕ := 5
noncomputable def payment : ℕ := 100

theorem dusty_change : 
  let total_cost := (single_layer_price * single_layer_quantity) + (double_layer_price * double_layer_quantity)
  in payment - total_cost = 37 := by
  sorry

end dusty_change_l809_809224


namespace factorization_of_polynomial_solve_quadratic_equation_l809_809559

-- Problem 1: Factorization
theorem factorization_of_polynomial : ∀ y : ℝ, 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) :=
by
  intro y
  sorry

-- Problem 2: Solving the quadratic equation
theorem solve_quadratic_equation : ∀ x : ℝ, x^2 + 4 * x + 3 = 0 ↔ x = -1 ∨ x = -3 :=
by
  intro x
  sorry

end factorization_of_polynomial_solve_quadratic_equation_l809_809559


namespace possible_dice_labels_l809_809181

theorem possible_dice_labels : 
  ∃ (die1 : Fin 6 → Nat) (die2 : Fin 6 → Nat), 
  (∀ k ∈ (Finset.range 1 37), ∃ i j, k = die1 i + die2 j) :=
by
  sorry

end possible_dice_labels_l809_809181


namespace total_students_l809_809375

theorem total_students (females : ℕ) (ratio : ℕ) (males := ratio * females) (total := females + males) :
  females = 13 → ratio = 3 → total = 52 :=
by
  intros h_females h_ratio
  rw [h_females, h_ratio]
  simp [total, males]
  sorry

end total_students_l809_809375


namespace sheilas_hourly_wage_l809_809452

-- Definitions based on conditions
def hours_per_day_mwf : ℕ := 8
def days_mwf : ℕ := 3

def hours_per_day_tt : ℕ := 6
def days_tt : ℕ := 2

def weekly_hours : ℕ := (hours_per_day_mwf * days_mwf) + (hours_per_day_tt * days_tt)

def weekly_earnings : ℕ := 396

-- The theorem proving the question == answer given the conditions
theorem sheilas_hourly_wage : (weekly_earnings / weekly_hours) = 11 :=
by
  unfold hours_per_day_mwf days_mwf hours_per_day_tt days_tt weekly_hours weekly_earnings
  sorry

end sheilas_hourly_wage_l809_809452


namespace polynomial_derivative_bound_l809_809984

open Real

noncomputable def p (x : ℝ) : ℝ := -- define polynomial p, we assume p is bounded as given

theorem polynomial_derivative_bound {p : ℝ → ℝ} (h₁ : ∀ x ∈ [-1, 1], |p x| ≤ 1)
  (h₂ : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) :
  ∀ x ∈ [-1, 1], |(deriv p) x| ≤ 4 :=
sorry

end polynomial_derivative_bound_l809_809984


namespace find_angle_BXY_l809_809005

-- Define angles AXE and CYX
variables (AXE CYX : ℝ)

-- Define the conditions
axiom parallel_lines : (AB CD : Prop)
axiom angle_relation : AXE = 4 * CYX - 120
axiom alt_interior_angles : AXE = CYX

-- Lean Theorem Statement
theorem find_angle_BXY (AXE CYX : ℝ) 
  (parallel_lines : AB CD) 
  (angle_relation : AXE = 4 * CYX - 120) 
  (alt_interior_angles : AXE = CYX) : 
  CYX = 40 :=
by
  sorry

end find_angle_BXY_l809_809005


namespace mode_and_median_of_scores_l809_809900

theorem mode_and_median_of_scores :
  let scores := [82, 95, 82, 76, 76, 82]
  in (mode scores = 82) ∧ (median scores = 82) :=
by
  -- mode is the most frequent element
  -- median is the middle value(s)
  sorry

end mode_and_median_of_scores_l809_809900


namespace circumcenter_of_triangle_A_A_C_on_line_A_l809_809092

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

-- Conditions
variables (A A' B B' C C' D : α)
variable  (circumcenter_triangle_A_A_C : α)
variables (circumcircle_of_triangle_A_A_C : set α)
variables (right_triangle_ABC : ∀ (A B C : α), angle A B C = π / 2)
variables (right_triangle_A'B'C' : ∀ (A' B' C' : α), angle A' B' C' = π / 2)
variable  (similar : ∀ (A B C A' B' C' : α), triangle.similar (A, B, C) (A', B', C'))
variable  (A_eq_C' : A = C')
variable  (A'_on_ray_BC : ∃ (k : ℝ), k > 1 ∧ A' = B + k • (C - B))

-- Question / Theorem to prove
theorem circumcenter_of_triangle_A_A_C_on_line_A'_B' :
  circumcenter_triangle_A_A_C ∈ line_through A' B' :=
sorry

end circumcenter_of_triangle_A_A_C_on_line_A_l809_809092


namespace not_prime_4k4_plus_1_not_prime_k4_plus_4_l809_809582

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_4k4_plus_1 (k : ℕ) (hk : k > 0) : ¬ is_prime (4 * k^4 + 1) :=
by sorry

theorem not_prime_k4_plus_4 (k : ℕ) (hk : k > 0) : ¬ is_prime (k^4 + 4) :=
by sorry

end not_prime_4k4_plus_1_not_prime_k4_plus_4_l809_809582


namespace unique_positive_integer_A_l809_809273

theorem unique_positive_integer_A :
  ∀ (m n p : ℕ), 1 ≤ p ∧ p ≤ n ∧ n ≤ m → 
  let A := (m - 1 / n) * (n - 1 / p) * (p - 1 / m) in 
  (∃! A : ℕ, A = 21) :=
by
  intros m n p h
  let A := (m - 1 / n) * (n - 1 / p) * (p - 1 / m)
  sorry

end unique_positive_integer_A_l809_809273


namespace sufficient_not_necessary_zero_point_l809_809997

noncomputable def f (x m : ℝ) : ℝ :=
  m + log x / log 2

theorem sufficient_not_necessary_zero_point (m : ℝ) :
  (∃ x : ℝ, x ≥ 1 ∧ f x m = 0) ↔ (m < 0 ∨ ∃ y : ℝ, y ≥ 1 ∧ f y m = 0) :=
by
  sorry

end sufficient_not_necessary_zero_point_l809_809997


namespace g_at_4_l809_809819

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

theorem g_at_4 : g 4 = -2 :=
by
  -- Proof would go here
  sorry

end g_at_4_l809_809819


namespace find_a_l809_809366

theorem find_a (a : ℤ) :
  (∃! x : ℤ, |a * x + a + 2| < 2) ↔ a = 3 ∨ a = -3 := 
sorry

end find_a_l809_809366


namespace mark_pages_per_week_l809_809061

theorem mark_pages_per_week
    (initial_hours_per_day : ℕ)
    (increase_percentage : ℕ)
    (initial_pages_per_day : ℕ) :
    initial_hours_per_day = 2 →
    increase_percentage = 150 →
    initial_pages_per_day = 100 →
    (initial_pages_per_day * (1 + increase_percentage / 100)) * 7 = 1750 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have reading_speed := 100 / 2 -- 50 pages per hour
  have increased_time := 2 * 1.5  -- 3 more hours
  have new_total_time := 2 + 3    -- 5 hours per day
  have pages_per_day := 5 * 50    -- 250 pages per day
  have pages_per_week := 250 * 7  -- 1750 pages per week
  exact eq.refl 1750

end mark_pages_per_week_l809_809061


namespace simplify_expression_correct_l809_809093

-- Defining the problem conditions and required proof
def simplify_expression (x : ℝ) (h : x ≠ 2) : Prop :=
  (x / (x - 2) + 2 / (2 - x) = 1)

-- Stating the theorem
theorem simplify_expression_correct (x : ℝ) (h : x ≠ 2) : simplify_expression x h :=
  by sorry

end simplify_expression_correct_l809_809093


namespace top_square_after_folds_is_7_l809_809583

-- Define the initial 3x3 grid
def initial_grid: List (List ℕ) := [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

-- Define the folding conditions
def fold_right (grid: List (List ℕ)) : List (List ℕ) :=
  [ [grid[0][2], grid[0][1], grid[0][0]],
    [grid[1][2], grid[1][1], grid[1][0]],
    [grid[2][2], grid[2][1], grid[2][0]] ]

def fold_left (grid: List (List ℕ)) : List (List ℕ) := grid

def fold_bottom (grid: List (List ℕ)) : List (List ℕ) :=
  [ grid[2], grid[1], grid[0] ]

-- Prove the top square number after three folds is 7
theorem top_square_after_folds_is_7 : 
  let g1 := fold_right initial_grid in
  let g2 := fold_left g1 in
  let g3 := fold_bottom g2 in
  g3.head.head = 7 := 
by {
  -- This is the theorem statement only
  sorry
}

end top_square_after_folds_is_7_l809_809583


namespace length_AB_min_OA_OB_l809_809793

-- Conditions for the problem
def parabola (A : ℝ × ℝ) : Prop := A.2^2 = 2 * A.1
def focus : ℝ × ℝ := (1 / 2, 1)
def origin : ℝ × ℝ := (0, 0)

-- First question: Length of AB
theorem length_AB 
  (A B : ℝ × ℝ) 
  (hA : parabola A) 
  (hB : parabola B) 
  (slope_AB : (B.2 - A.2) / (B.1 - A.1) = 2) 
  (line_through_focus : (B.2 - A.2) = 2 * (B.1 - 1/2)) :
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 / 2 :=
sorry

-- Second question: Minimum value of |OA| * |OB|
theorem min_OA_OB
  (A B : ℝ × ℝ)
  (hA : parabola A)
  (hB : parabola B)
  (O : ℝ × ℝ := origin)
  (perpendicular : A.1 * B.1 + A.2 * B.2 = 0) :
  (real.sqrt (A.1^2 + A.2^2) * real.sqrt (B.1^2 + B.2^2)) ≥ 8 :=
sorry

end length_AB_min_OA_OB_l809_809793


namespace davida_hours_difference_l809_809628

theorem davida_hours_difference : 
  let hours_week_1 := 35 in
  let hours_week_2 := 35 in
  let total_hours_weeks_1_2 := hours_week_1 + hours_week_2 in
  let hours_week_3 := 48 in
  let hours_week_4 := 48 in
  let total_hours_weeks_3_4 := hours_week_3 + hours_week_4 in
  total_hours_weeks_3_4 - total_hours_weeks_1_2 = 26 := 
by
  sorry

end davida_hours_difference_l809_809628


namespace lana_extra_flowers_l809_809788

noncomputable def tulips_total := 120
noncomputable def roses_total := 74
noncomputable def lilies_total := 56

noncomputable def tulips_used := (0.45 * tulips_total).toInt
noncomputable def roses_used := (0.60 * roses_total).toInt
noncomputable def lilies_used := (0.70 * lilies_total).toInt

noncomputable def tulips_extra := tulips_total - tulips_used
noncomputable def roses_extra := roses_total - roses_used
noncomputable def lilies_extra := lilies_total - lilies_used

noncomputable def total_extra := tulips_extra + roses_extra + lilies_extra

theorem lana_extra_flowers : total_extra = 113 := 
by
  sorry

end lana_extra_flowers_l809_809788


namespace sum_of_coefficients_l809_809048

def u (n : ℕ) : ℕ := 
  match n with
  | 0 => 6 -- Assume the sequence starts at u_0 for easier indexing
  | n + 1 => u n + 5 + 2 * n

theorem sum_of_coefficients (u : ℕ → ℕ) : 
  (∀ n, u (n + 1) = u n + 5 + 2 * n) ∧ u 1 = 6 → 
  (∃ a b c : ℕ, (∀ n, u n = a * n^2 + b * n + c) ∧ a + b + c = 6) := 
by
  sorry

end sum_of_coefficients_l809_809048


namespace symmetric_axis_of_transformed_function_l809_809335

def original_function (x : ℝ) : ℝ :=
  sqrt 3 * sin (x - π / 6) + cos (x - π / 6)

noncomputable def transformed_function (x : ℝ) : ℝ :=
  2 * sin (2 * (x + π / 6))

theorem symmetric_axis_of_transformed_function :
  ∃ k : ℤ, (2 * k * π / 2 + π / 12 = π / 12) →
  k = 0 :=
sorry

end symmetric_axis_of_transformed_function_l809_809335


namespace convex_100_gon_intersection_50_triangles_l809_809282

def smallest_n_intersection_of_triangles (n : ℕ) : Prop :=
∀ (P : Polytope ℝ 2), P.is_convex ∧ P.num_sides = 100 → 
∃ (T : ℕ), T ≤ n ∧ ∀ i < T, is_triangle (get_triangle P i) ∧ 
    (P = ⋂ i < T, get_triangle P i)

theorem convex_100_gon_intersection_50_triangles :
    smallest_n_intersection_of_triangles 50 :=
sorry

end convex_100_gon_intersection_50_triangles_l809_809282


namespace list_count_first_10_positive_integers_l809_809033

theorem list_count_first_10_positive_integers :
  let is_valid (l : List ℕ) : Prop :=
    (∀ i, 2 ≤ i ∧ i ≤ 10 → ((l[i-1] + 1 < i ∧ l.take (i-1).contains (l[i-1] + 1)) ∨ (l[i-1] - 1 < i ∧ l.take (i-1).contains (l[i-1] - 1))))
  List.filter (λ l, l.length = 10 ∧ (∀ n, n ∈ l → n ∈ {1, 2, ..., 10}) ∧ is_valid l) (List.permutations [1, 2, ..., 10]).length = 512 :=
sorry

end list_count_first_10_positive_integers_l809_809033


namespace find_area_of_park_l809_809891

-- Define the variables and conditions
variable (x : ℝ) 

-- Conditions identified earlier
def length := 3 * x
def width := 2 * x
def cost_per_meter_dollar := 1 / 150
def total_fence_cost_dollar := 100

-- Use the conditions to redefine the needed values for proof
def perimeter := 2 * (length + width)
def area := length * width

-- Lean theorem statement for the problem
theorem find_area_of_park
  (h1 : cost_per_meter_dollar * perimeter = total_fence_cost_dollar) 
  (h2 : length = 3 * x) 
  (h3 : width = 2 * x) :
  area = 13500000 := 
by
  sorry

end find_area_of_park_l809_809891


namespace f_g_5_l809_809359

def g (x : ℕ) : ℕ := 4 * x + 10

def f (x : ℕ) : ℕ := 6 * x - 12

theorem f_g_5 : f (g 5) = 168 := by
  sorry

end f_g_5_l809_809359


namespace carols_weight_l809_809896

variables (a c : ℝ)

theorem carols_weight (h1 : a + c = 220) (h2 : c - a = c / 3 + 10) : c = 138 :=
by
  sorry

end carols_weight_l809_809896


namespace triangle_division_l809_809355

theorem triangle_division (A B C P Q : Type) (h1 : ∃ P, point_in_triangle A B C P)
  (h2 : ∃ Q, point_on_side B C Q) : 
  ∃ APQ BPQ CPQ PAB, 
    is_triangle_division A B C APQ BPQ CPQ PAB ∧ 
    (∀ T1 T2, T1 ≠ T2 → ¬(shares_entire_side T1 T2)) :=
sorry

end triangle_division_l809_809355


namespace students_who_like_yellow_l809_809431

theorem students_who_like_yellow (total_students girls students_like_green girls_like_pink students_like_yellow : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_green = total_students / 2)
  (h3 : girls_like_pink = girls / 3)
  (h4 : girls = 18)
  (h5 : students_like_yellow = total_students - (students_like_green + girls_like_pink)) :
  students_like_yellow = 9 :=
by
  sorry

end students_who_like_yellow_l809_809431


namespace average_marks_of_class_l809_809991

theorem average_marks_of_class : 
  let total_students := 25 
    in let students_scored_95 := 5 
      in let students_scored_zero := 3 
        in let avg_score_rest := 45 
          in (students_scored_95 * 95 + students_scored_zero * 0 + (total_students - students_scored_95 - students_scored_zero) * avg_score_rest) / total_students = 49.6 := 
by
  sorry

end average_marks_of_class_l809_809991


namespace find_a_l809_809720

noncomputable def f (a : ℝ) (x : ℝ) := a * x * Real.sin x - (3 / 2 : ℝ)
def interval := Set.Icc (0 : ℝ) (Real.pi / 2)
def max_f_value := (Real.pi - 3) / 2

theorem find_a (a : ℝ) : (∃ x ∈ interval, ∀ y ∈ interval, f a y ≤ f a x) ∧ f a (Real.pi / 2) = max_f_value → a = 1 :=
by
  sorry

end find_a_l809_809720


namespace find_x_coordinate_of_C_l809_809440

theorem find_x_coordinate_of_C :
  let A := (-3 : ℝ, 2 : ℝ),
      B := (5 : ℝ, 10 : ℝ),
      y_C := 8 in
  ∃ x_C : ℝ, 
    (∃ C : ℝ × ℝ, C = (x_C, y_C) ∧ 
    dist A C = 2 * dist A B / 3 ∧ 
    dist B C = dist A B / 3 ∧ 
    A.1 <= C.1 ∧ C.1 <= B.1) ∧ 
    x_C = 7 / 3 :=
begin
  sorry
end

end find_x_coordinate_of_C_l809_809440


namespace clock_angle_at_3_15_l809_809221

noncomputable def degrees_angle_at_3_15 : ℝ := 7.5

theorem clock_angle_at_3_15 :
  let minute_angle := (15 / 60) * 360,
      hour_movement := (15 / 60) * 30,
      hour_angle := 90 + hour_movement,
      smaller_angle := abs (minute_angle - hour_angle)
  in smaller_angle = degrees_angle_at_3_15 := 
by {
  let minute_angle := (15 / 60) * 360,
  let hour_movement := (15 / 60) * 30,
  let hour_angle := 90 + hour_movement,
  let smaller_angle := abs (minute_angle - hour_angle),
  have h1 : minute_angle = 90, by norm_num,
  have h2 : hour_movement = 7.5, by norm_num,
  have h3 : hour_angle = 97.5, by linarith [h1, h2],
  have h4 : smaller_angle = abs (90 - 97.5), by rw [h1, h3],
  have h5 : smaller_angle = 7.5, by norm_num [h4],
  exact h5,
  sorry
}

end clock_angle_at_3_15_l809_809221


namespace largest_base_5_three_digit_in_base_10_l809_809949

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l809_809949


namespace gordon_bookstore_cost_l809_809516

def price (p : ℝ) (d : ℝ) : ℝ := p - (p * d)

def total_cost (books : List ℝ) (d_over_22 : ℝ) (d_under_20 : ℝ) : ℝ :=
  (books.filter (λ p => p > 22)).map (λ p => price p d_over_22) |>.sum
  + (books.filter (λ p => p < 20)).map (λ p => price p d_under_20) |>.sum
  + (books.filter (λ p => p == 21)).sum

theorem gordon_bookstore_cost :
  let books := [25.0, 18.0, 21.0, 35.0, 12.0, 10.0]
  let d_over_22 := 0.30
  let d_under_20 := 0.20
  total_cost books d_over_22 d_under_20 = 95.00 := 
by
  sorry

end gordon_bookstore_cost_l809_809516


namespace square_perimeter_calculation_l809_809595

noncomputable def perimeter_of_square (radius: ℝ) : ℝ := 
  if radius = 4 then 64 * Real.sqrt 2 else 0

theorem square_perimeter_calculation :
  perimeter_of_square 4 = 64 * Real.sqrt 2 :=
by
  sorry

end square_perimeter_calculation_l809_809595


namespace no_such_function_l809_809272

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f(x + y) = x * f(x) + y :=
by
  sorry

end no_such_function_l809_809272


namespace correct_product_equality_l809_809403

def geom_sequence_terms (a : ℕ → ℝ) : Prop :=
∀ n, a n = a 0 * (q ^ n)

def product_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = ∏ i in finset.range n, a (i + 1)

theorem correct_product_equality
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h_seq : geom_sequence_terms a)
  (h_prod : product_of_first_n_terms a S)
  (h_nine : a 9 = 1) :
  S 5 = S 12 :=
sorry

end correct_product_equality_l809_809403


namespace JackRan_distance_l809_809434

noncomputable def Jack_distance (avg1 avg2 : ℚ) (n1 n2 : ℕ) (di : ℚ) : ℚ :=
di * (n2 * avg2 - n1 * avg1) / (n2 - n1)

theorem JackRan_distance :
  let avg1 := 3
  let avg2 := 3.1
  let n1 := 20
  let n2 := 21
  let di := 20 * avg1
  in Jack_distance avg1 avg2 n1 n2 di = 5.1 :=
by {
  -- Proof would go here, but we're using sorry to skip the actual proof steps.
  sorry
}

end JackRan_distance_l809_809434


namespace necessary_but_not_sufficient_for_q_implies_range_of_a_l809_809097

variable (a : ℝ)

def p (x : ℝ) := |4*x - 3| ≤ 1
def q (x : ℝ) := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem necessary_but_not_sufficient_for_q_implies_range_of_a :
  (∀ x : ℝ, q a x → p x) → (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end necessary_but_not_sufficient_for_q_implies_range_of_a_l809_809097


namespace find_y_angle_l809_809006

-- Define points and angles
structure Point :=
  (x : ℝ) (y : ℝ)

structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

def angle_at (p : Point) (l1 l2 : Line) : ℝ := sorry

-- Given conditions
variables (p q r s : Line)
variable (B : Point)

-- Assuming p and q are parallel and r is a transversal
axiom parallel_p_q : p.a = q.a ∧ p.b = q.b
axiom transversal_r : ∃ x, r.a * x + r.b * x + r.c = 0

-- Angles above point B on line p and just below on line q
axiom angle_above_B_on_p : angle_at B p r = 45
axiom angle_below_B_on_q : angle_at B q s = 15

-- The final property to prove
theorem find_y_angle : (angle_at B q s + angle_at B p r) = 60 → 180 - (angle_at B p r + angle_at B q s) = 120
:= by
  -- Given angle_above_B_on_p and angle_below_B_on_q, calculate y
  intro h
  rw h
  exact rfl

end find_y_angle_l809_809006


namespace boat_speed_in_still_water_l809_809130

theorem boat_speed_in_still_water (b : ℝ) (current_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : current_speed = 5) (h2 : distance = 10) (h3 : time = 24 / 60) 
  (downstream_speed : ℝ) (h4 : downstream_speed = b + current_speed) 
  (h5 : distance = downstream_speed * time) : b = 20 :=
by
  have : 24 / 60 = 0.4 := by norm_num
  rw [← this] at h3
  have : downstream_speed = b + 5 := by rw h1; exact h4
  have : 10 = (b + 5) * 0.4 := by rw [h2, h3, this]; exact h5
  have : 10 = 0.4 * b + 2 := by linarith
  have : 8 = 0.4 * b := by linarith
  have : b = 20 := by linarith
  exact this

end boat_speed_in_still_water_l809_809130


namespace probability_all_selected_l809_809140

theorem probability_all_selected (P_Ram P_Ravi P_Ritu : ℚ) 
  (h1 : P_Ram = 3 / 7) 
  (h2 : P_Ravi = 1 / 5) 
  (h3 : P_Ritu = 2 / 9) : 
  P_Ram * P_Ravi * P_Ritu = 2 / 105 := 
by
  sorry

end probability_all_selected_l809_809140


namespace salamander_decline_l809_809267

theorem salamander_decline :
  ∀ N : ℝ, (N > 0) → ∃ t : ℕ, N * (0.8 : ℝ)^t < 0.05 * N ∧ t = 14 :=
begin
  intro N,
  intro hN,
  use 14,
  split,
  { exact calc
      N * (0.8 : ℝ)^14
        = N * 0.8^14 : by rw [pow_nat_cast 0.8 14]
    ... < 0.05 * N : by sorry },
  { refl }
end

end salamander_decline_l809_809267


namespace find_n_l809_809601

-- Define the values of quarters and dimes in cents
def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10

-- Define the number of quarters and dimes
def num_quarters : ℕ := 15
def num_dimes : ℕ := 25

-- Define the total value in cents corresponding to the quarters
def total_value_quarters : ℕ := num_quarters * value_of_quarter

-- Define the condition where total value by quarters equals total value by n dimes
def equivalent_dimes (n : ℕ) : Prop := total_value_quarters = n * value_of_dime

-- The theorem to prove
theorem find_n : ∃ n : ℕ, equivalent_dimes n ∧ n = 38 := 
by {
  use 38,
  sorry
}

end find_n_l809_809601


namespace correct_option_C_l809_809407

variable {α β : Type} [Plane α] [Plane β]
variable {m n : Type} [Line m] [Line n]

-- Define perpendicular and parallel relationships between lines and planes.
def perpendicular (l1 l2 : Type) [Line l1] [Line l2] : Prop := sorry
def parallel (l1 l2 : Type) [Line l1] [Line l2] : Prop := sorry
def line_in_plane (l : Type) [Line l] (p : Type) [Plane p] : Prop := sorry

theorem correct_option_C (h1 : perpendicular m β) (h2 : parallel n β) : perpendicular m n :=
sorry

end correct_option_C_l809_809407


namespace PB_length_l809_809775

/-- In a square ABCD with area 1989 cm², with the center O, and
a point P inside such that ∠OPB = 45° and PA : PB = 5 : 14,
prove that PB = 42 cm. -/
theorem PB_length (s PA PB : ℝ) (h₁ : s^2 = 1989) 
(h₂ : PA / PB = 5 / 14) 
(h₃ : 25 * (PA / PB)^2 + 196 * (PB / PA)^2 = s^2) :
  PB = 42 := 
by sorry

end PB_length_l809_809775


namespace base10_to_base3_121_l809_809928

open Nat

theorem base10_to_base3_121 : (show String from 121) = "11111" := by
  sorry

end base10_to_base3_121_l809_809928


namespace combined_wealth_of_a_c_f_l809_809584

-- Assuming a structure to represent grandchildren wealth distribution
structure WealthDistribution where
  a b c d e f g : ℕ

-- Conditions extracted from the problem
def totalWealth : ℕ := 100000
def ratio : WealthDistribution := ⟨7, 5, 4, 3, 5, 7, 6⟩
def totalParts : ℕ := ratio.a + ratio.b + ratio.c + ratio.d + ratio.e + ratio.f + ratio.g

-- Defining the value of one part
def valuePerPart : ℕ := totalWealth / totalParts

-- Combined wealth calculation
def combinedWealthACF : ℕ := (ratio.a + ratio.c + ratio.f) * valuePerPart

-- Proof statement
theorem combined_wealth_of_a_c_f : combinedWealthACF = 48648.60 := by
  -- The proof goes here
  sorry

end combined_wealth_of_a_c_f_l809_809584


namespace a_plus_b_equals_109_l809_809316

theorem a_plus_b_equals_109
  (h1 : sqrt (2 + 2 / 3) = 2 * sqrt (2 / 3))
  (h2 : sqrt (3 + 3 / 8) = 3 * sqrt (3 / 8))
  (h3 : sqrt (4 + 4 / 15) = 4 * sqrt (4 / 15))
  (h4 : sqrt (5 + 5 / 24) = 5 * sqrt (5 / 24))
  (h5 : sqrt (10 + 10 / 99) = 10 * sqrt (10 / 99)) :
  10 + 99 = 109 := 
sorry

end a_plus_b_equals_109_l809_809316


namespace least_positive_n_is_not_integer_l809_809401

noncomputable def find_least_positive_n 
  (p : ℕ) (r : ℕ) [Fact (Nat.Prime p)] (h_r : r > 0) : ℕ :=
(p-2) * p^(r-1) + 1 + (r-1) * (p-1) * p^(r-1)

theorem least_positive_n_is_not_integer
  (p : ℕ) (r : ℕ) [Fact (Nat.Prime p)] (h_r : r > 0) :
  ∃ n, n = find_least_positive_n p r ∧
  ∑ k in Finset.filter (λ k, Nat.gcd k p = 1) (Finset.range (p^r).succ), 
    (1 / (1 - complex.exp (2 * real.pi * complex.I / (p^r)) ^ k) ^ n) ∉ ℤ  :=
sorry

end least_positive_n_is_not_integer_l809_809401


namespace f_at_neg_8_5_pi_eq_pi_div_2_l809_809485

def f (x : Real) : Real := sorry

axiom functional_eqn (x : Real) : f (x + (3 * Real.pi / 2)) = -1 / f x
axiom f_interval (x : Real) (h : x ∈ Set.Icc (-Real.pi) Real.pi) : f x = x * Real.sin x

theorem f_at_neg_8_5_pi_eq_pi_div_2 : f (-8.5 * Real.pi) = Real.pi / 2 := 
  sorry

end f_at_neg_8_5_pi_eq_pi_div_2_l809_809485


namespace liquid_level_ratio_l809_809144

/-- 
Given two right circular cones with radii of the tops of the liquid surfaces as 4 cm and 8 cm, 
and filled with the same amount of liquid, show that after sinking a spherical marble of radius 2 cm 
into each cone, the ratio of the rise in the liquid level in the narrower cone to the rise 
in the wider cone is 4:1. 
-/
theorem liquid_level_ratio
  (r1 r2 : ℝ) (h1 h2 : ℝ) (V : ℝ)
  (initial_height_same : (1/3) * π * r1^2 * h1 = V)
  (initial_height_same_wide : (1/3) * π * r2^2 * h2 = V)
  (r1_val : r1 = 4) (r2_val : r2 = 8)
  (marble_radius : ℝ) (marble_volume : ℝ) 
  (marble_radius_val : marble_radius = 2)
  (marble_volume_val : marble_volume = (4/3) * π * marble_radius^3) :
  let Δh1 := 2 in
  let Δh2 := 0.5 in
  Δh1 / Δh2 = 4 := sorry

end liquid_level_ratio_l809_809144


namespace find_z2015_l809_809326

noncomputable def complex_seq : ℕ → ℂ 
| 1       := 1
| (n + 1) := complex.conj (complex_seq n) + 1 + (n : ℂ) * complex.I

theorem find_z2015 : complex_seq 2015 = 2015 + 1007 * complex.I :=
by      
      sorry

end find_z2015_l809_809326


namespace least_positive_nine_n_square_twelve_n_cube_l809_809533

theorem least_positive_nine_n_square_twelve_n_cube :
  ∃ (n : ℕ), 0 < n ∧ (∃ (k1 k2 : ℕ), 9 * n = k1^2 ∧ 12 * n = k2^3) ∧ n = 144 :=
by
  sorry

end least_positive_nine_n_square_twelve_n_cube_l809_809533


namespace total_length_of_rope_l809_809589

theorem total_length_of_rope (x : ℝ) : (∃ r1 r2 : ℝ, r1 / r2 = 2 / 3 ∧ r1 = 16 ∧ x = r1 + r2) → x = 40 :=
by
  intro h
  cases' h with r1 hr
  cases' hr with r2 hs
  sorry

end total_length_of_rope_l809_809589


namespace complex_series_sum_eq_zero_l809_809673

noncomputable def z_complex_series : ℂ :=
  ∑ k in (finset.range 12).map (function.embedding.coe finset.nat_embedding), (complex.i ^ (k + 1))

theorem complex_series_sum_eq_zero : z_complex_series = 0 := 
  sorry

end complex_series_sum_eq_zero_l809_809673


namespace equal_perpendicular_distances_l809_809388

theorem equal_perpendicular_distances {C1 C2 : Type} [circle C1] [circle C2] 
  (h_congruent : congruent C1 C2) 
  (θ₁ θ₂ : central_angle) 
  (h_equal_angles : θ₁ = θ₂) : 
  perpendicular_distance_to_chord θ₁ = perpendicular_distance_to_chord θ₂ :=
  sorry

end equal_perpendicular_distances_l809_809388


namespace number_of_valid_house_numbers_l809_809639

theorem number_of_valid_house_numbers : 
  ∃ k : ℕ, ∀ ABCD : ℕ, 
  (ABCD / 100 = p ∧ ABCD % 100 = q ∧ p ∈ {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59} ∧ 
  q ∈ {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59} ∧ p ≠ q ∧ ∀ m : ℕ, 
  ¬ ((p * q : ℕ) = m^3)) ↔ ABCD = k :=
sorry

end number_of_valid_house_numbers_l809_809639


namespace cannot_navigate_within_700m_can_navigate_within_800m_l809_809556

-- Condition: The width of the river is 1000 meters
def river_width := 1000

-- Condition: The banks consist of line segments and arcs of circles
-- We will represent this as a generic structure without specific implementation details
structure Bank :=
  (segments: Set (segment ℝ))
  (arcs: Set (arc ℝ))

-- Condition: The river is defined with two banks and a width
structure River :=
  (left_bank : Bank)
  (right_bank : Bank)
  (width : ℝ := river_width)

-- The main problem statements:
-- Question a)
theorem cannot_navigate_within_700m (R : River) : ¬ (∃ (route : Route), ∀ p ∈ route, 
  distance_to_bank p R.left_bank ≤ 700 ∧ distance_to_bank p R.right_bank ≤ 700) := sorry

-- Question b)
theorem can_navigate_within_800m (R : River) : ∃ (route : Route), ∀ p ∈ route, 
  distance_to_bank p R.left_bank ≤ 800 ∧ distance_to_bank p R.right_bank ≤ 800 := sorry

end cannot_navigate_within_700m_can_navigate_within_800m_l809_809556


namespace major_premise_error_l809_809138

theorem major_premise_error (f : ℝ → ℝ) (x₀ : ℝ) 
  (h₁ : ∀ (f : ℝ → ℝ) (x₀ : ℝ), (f'' x₀ = 0) → is_extreme_point f x₀)
  (h₂ : f = λ x, x^3)
  (h₃ : f'' 0 = 0) :
  ¬ is_extreme_point (λ x, x^3) 0 :=
by sorry

def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop := sorry

end major_premise_error_l809_809138


namespace largest_base5_three_digit_to_base10_l809_809959

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l809_809959


namespace ellipse_same_foci_l809_809709

-- Definitions related to the problem
variables {x y p q : ℝ}

-- Condition
def represents_hyperbola (p q : ℝ) : Prop :=
  (p * q > 0) ∧ (∀ x y : ℝ, (x^2 / -p + y^2 / q = 1))

-- Proof Statement
theorem ellipse_same_foci (p q : ℝ) (hpq : p * q > 0)
  (h : ∀ x y : ℝ, x^2 / -p + y^2 / q = 1) :
  (∀ x y : ℝ, x^2 / (2*p + q) + y^2 / p = -1) :=
sorry -- Proof goes here

end ellipse_same_foci_l809_809709


namespace rectangle_side_length_l809_809602

noncomputable theory

variables (a b : ℝ) -- Side length of square and AE
-- Define the concept of a square and the relationship between variables
def is_square (ABCD : Type) := ∀ (x y : ℝ), (∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D ∧ ABCD = x * y)
def is_rectangle (AEFG : Type) := ∀ (x y: ℝ), (∀ (A E F G : ℝ), A = E ∧ E = F ∧ F = G ∧ AEFG = x * y)

-- Theorem statement
theorem rectangle_side_length (a b : ℝ) (Ha : 0 < a) (Hb : 0 < b) :
  (∃ EF : ℝ, EF = a^2 / b) :=
  sorry

end rectangle_side_length_l809_809602


namespace sequence_fourth_term_l809_809385

theorem sequence_fourth_term (r x : ℝ)
  (h1 : 0.02 = 0.001 * r)
  (h2 : 0.4 = 0.02 * r)
  (h3 : x = 0.4 * r) :
  x = 8 :=
by
  have r_value : r = 20 := by
    linarith
  rw [r_value] at h3
  rw [h3]
  norm_num

end sequence_fourth_term_l809_809385


namespace compound_interest_rate_l809_809172

theorem compound_interest_rate
  (P A : ℝ) (t n : ℕ) (r : ℝ)
  (hP : P = 1200)
  (hA : A = 1348.32)
  (ht : t = 2)
  (hn : n = 1)
  (formula : A = P * (1 + r / n) ^ (n * t)) :
  r = (ℝ.sqrt (A / P) - 1) :=
by
  rw [hP, hA, ht, hn] at formula
  sorry

end compound_interest_rate_l809_809172


namespace circle_equation_trajectory_midpoint_l809_809303

-- Problem 1: Equation of the circle
theorem circle_equation (A B : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (hA : A = (1, 2)) (hB : B = (1, 10)) (O : ℝ × ℝ)
  (hO1 : O.1 > 0) (hO2 : O.2 > 0)
  (hTangent : tangent O 2 1 1) :
  ∃ (a r : ℝ), (x - a) ^ 2 + (y - 6) ^ 2 = r ^ 2 ∧ a = 3 ∧ r = 2 * sqrt 5 ∧ 
  (x − 3)^2 + (y − 6)^2 = 20 := by
  sorry

-- Problem 2: Trajectory of the midpoint
theorem trajectory_midpoint (P Q M : ℝ × ℝ)
  (hQ : Q = (-3, -6))
  (hCircle : (P.1 - 3) ^ 2 + (P.2 - 6) ^ 2 = 20)
  (hMid : M = ((P.1 - 3) / 2, (P.2 - 6) / 2)) :
  M.1^2 + M.2^2 = 5 := by
  sorry

end circle_equation_trajectory_midpoint_l809_809303


namespace car_downhill_distance_l809_809198

/-- Given conditions:
  - speed uphill: 30 km/hr
  - speed downhill: 70 km/hr
  - uphill distance: 100 km
  - average speed: 37.05882352941177 km/hr
Prove that the distance travelled downhill is 50 km. -/
theorem car_downhill_distance :
  ∃ (D : ℝ), let speed_uphill := 30,
    let speed_downhill := 70,
    let distance_uphill := 100,
    let average_speed := 37.05882352941177,
    let time_uphill := distance_uphill / speed_uphill,
    let time_downhill := D / speed_downhill,
    let total_distance := distance_uphill + D,
    let total_time := time_uphill + time_downhill,
  average_speed = total_distance / total_time ∧ D = 50 :=
sorry

end car_downhill_distance_l809_809198


namespace circumscribed_sphere_radius_l809_809890

noncomputable def radius_of_circumscribed_sphere (a : ℝ) (α : ℝ) : ℝ :=
  a / (3 * Real.sin α)

theorem circumscribed_sphere_radius (a α : ℝ) :
  radius_of_circumscribed_sphere a α = a / (3 * Real.sin α) :=
by
  sorry

end circumscribed_sphere_radius_l809_809890


namespace initial_seashells_l809_809022

-- Definitions based on the problem conditions
def gave_to_joan : ℕ := 6
def left_with_jessica : ℕ := 2

-- Theorem statement to prove the number of seashells initially found by Jessica
theorem initial_seashells : gave_to_joan + left_with_jessica = 8 := by
  -- Proof goes here
  sorry

end initial_seashells_l809_809022


namespace real_part_implies_value_of_a_l809_809715

theorem real_part_implies_value_of_a (a b : ℝ) (h : a = 2 * b) (hb : b = 1) : a = 2 := by
  sorry

end real_part_implies_value_of_a_l809_809715


namespace largest_base5_three_digit_to_base10_l809_809970

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l809_809970


namespace recurring_decimal_to_fraction_l809_809537

theorem recurring_decimal_to_fraction : ∃ x : ℕ, (0.37 + (0.246 / 999)) = (x / 99900) ∧ x = 371874 :=
by
  sorry

end recurring_decimal_to_fraction_l809_809537


namespace fourth_term_binomial_expansion_l809_809532

def binomial_expansion_step (n : ℕ) (a x : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * (a / x^2)^(n - k) * (x^2 / a)^k

theorem fourth_term_binomial_expansion (a x : ℝ) :
  binomial_expansion_step 7 a x 3 = 35 * (a / x^2) :=
by
  sorry

end fourth_term_binomial_expansion_l809_809532


namespace inverse_matrix_of_A_l809_809692

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![(-1 : ℚ), 0], ![0, 1/2]]

theorem inverse_matrix_of_A :
  let B := ![![1, (2 : ℚ)], ![0, 6]]
  let A := fun m : Matrix (Fin 2) (Fin 2) ℚ =>
    m ⬝ B = ![![(-1 : ℚ), (-2)], ![0, 3]]
  A matrix_A → matrix_A⁻¹ = ![![(-1 : ℚ), 0], ![0, 2]] := by
  sorry

end inverse_matrix_of_A_l809_809692


namespace largest_base5_three_digit_to_base10_l809_809969

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l809_809969


namespace length_MN_radius_sphere_l809_809112

-- Define the conditions of the cube
def is_on_segment_BD (M : ℝ × ℝ × ℝ) (a : ℝ) : Prop := 
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ M = (k * a, a - k * a, 0)

def is_on_segment_CC1 (N : ℝ × ℝ × ℝ) (a : ℝ) : Prop := 
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ N = (a, a, k * a)

def angle_with_plane_π_over_4 (M N : ℝ × ℝ × ℝ) (a : ℝ) : Prop := 
  let d_vec := (a - M.1, M.1, N.3)
  (N.3 / (real.sqrt ((a - M.1) ^ 2 + (M.1) ^ 2 + (N.3) ^ 2)) = real.sqrt 2 / 2)

def angle_with_plane_π_over_6 (M N : ℝ × ℝ × ℝ) (a : ℝ) : Prop := 
  let d_vec := (a - M.1, M.1, N.3)
  ((a - M.1) / (real.sqrt ((a - M.1) ^ 2 + (M.1) ^ 2 + (N.3) ^ 2)) = real.sqrt 3 / 2)

-- Part (a): Prove length of MN is a
theorem length_MN (a : ℝ) (M N : ℝ × ℝ × ℝ) :
  is_on_segment_BD M a → is_on_segment_CC1 N a →
  angle_with_plane_π_over_4 M N a → angle_with_plane_π_over_6 M N a →
  real.sqrt ((N.1 - M.1) ^ 2 + (N.2 - M.2) ^ 2 + (N.3 - M.3) ^ 2) = a :=
by
  sorry

-- Part (b): Prove radius of the sphere is (a*(2 - √2)) / 2
theorem radius_sphere (a : ℝ) (M N : ℝ × ℝ × ℝ) :
  is_on_segment_BD M a → is_on_segment_CC1 N a →
  angle_with_plane_π_over_4 M N a → angle_with_plane_π_over_6 M N a →
  let R := (a * (2 - real.sqrt 2)) / 2 in
  ∃ (P : ℝ × ℝ × ℝ), (P ∈ line M N) ∧ 
  (real.dist P.abcd_plane = R) ∧ (real.dist P.bb1c1c_plane = R) :=
by
  sorry

end length_MN_radius_sphere_l809_809112


namespace ellipse_focal_length_l809_809114

theorem ellipse_focal_length (m : ℝ) : 
  (∃ c : ℝ, ∀ (x y : ℝ), (x^2 / 16) + (y^2 / m) = 1 ∧ 2 * c = 2 * Real.sqrt 7) →
  (m = 9 ∨ m = 23) :=
begin
  sorry
end

end ellipse_focal_length_l809_809114


namespace part1_part2_l809_809719

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (x - Real.pi / 3))

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 :
  {x : ℝ | f x < 1 / 4} = {x : ℝ | ∃ k : ℤ, x ∈ Set.Ioo (k * Real.pi - 7 * Real.pi / 12) (k * Real.pi - Real.pi / 12)} :=
by
  sorry

end part1_part2_l809_809719


namespace round_robin_games_count_l809_809510

theorem round_robin_games_count (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = 15 := by
  sorry

end round_robin_games_count_l809_809510


namespace solid_of_revolution_arcsin_arccos_l809_809236

noncomputable def volume_of_solid_of_revolution (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  π * ∫ y in a..b, (f y)^2

theorem solid_of_revolution_arcsin_arccos :
  volume_of_solid_of_revolution 0 (π / 2) (λ x, let y := if x < π / 4 then cos x else sin x in y) = 
  (π * (π + 2)) / 4 :=
by
  sorry

end solid_of_revolution_arcsin_arccos_l809_809236


namespace collinear_intersection_points_l809_809834

noncomputable theory
open_locale classical

variables {Ω : Type*} [euclidean_space Ω]

-- Define points and triangles
variables (A B C A1 B1 C1 P M : Ω)
variables (circumcircle_feuerbach : circle Ω)
variables (circumcircle_abc : circle Ω)
variables (thales_circle : circle Ω)

-- Definitions expressing conditions
def pedal_triangle (A B C P : Ω) : triangle Ω := sorry
def orthocenter (A B C : Ω) : Ω := sorry
def radical_axis (k1 k2 : circle Ω) : line Ω := sorry

-- The statement we want to prove
theorem collinear_intersection_points :
  let A0 := line_intersection (line_through B C) (line_through B1 C1),
      B0 := line_intersection (line_through A C) (line_through A1 C1),
      C0 := line_intersection (line_through A B) (line_through A1 B1) in
  collinear ({A0, B0, C0} : set Ω) :=
begin
  sorry
end

end collinear_intersection_points_l809_809834


namespace sum_of_digits_M_eq_8_l809_809251

noncomputable def M : ℕ := 9 + 99 + 999 + 9999 + 99999

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_M_eq_8 : sum_of_digits M = 8 := by
  sorry

end sum_of_digits_M_eq_8_l809_809251


namespace p_value_k_value_m_range_l809_809412

noncomputable def f (x : ℝ) (p : ℝ) := 2^x + (p - 1) * 2^(-x)

-- Prove that p = 2 given that f(x) is even.
theorem p_value : 
  (∀ x : ℝ, f x p = f (-x) p) → p = 2 :=
  by
  intro h
  sorry

noncomputable def g (x : ℝ) (p k : ℝ) := f (2*x) p - 2*k*(2^x - 2^(-x))

-- Prove that k = sqrt 6 given that g(x) has a minimum value of -4 on [1, +∞).
theorem k_value :
  (∀ x ∈ Ici 1, g x 2 k ≥ -4) → k = Real.sqrt 6 :=
  by
  intro h
  sorry

-- Prove the range of m is (-∞, 3) given that f(2x) > m*f(x) - 4 for any real number x.
theorem m_range :
  (∀ x : ℝ, f (2*x) 2 > m * f x 2 - 4) → m < 3 :=
  by
  intro h
  sorry

end p_value_k_value_m_range_l809_809412


namespace red_light_first_time_at_third_intersection_expected_value_X_l809_809051

/- Definitions based on provided conditions -/
def p0 : ℝ := 1 / 2

variable (p1 p2 : ℝ)

def independent_encounters :=
  (1 - p0) * (1 - p1) * (1 - p2) = 1 / 24 ∧
  p0 * p1 * p2 = 1 / 4

/- Proof goal 1: Probability that Li Ping encounters a red light for the first time at the third intersection -/
theorem red_light_first_time_at_third_intersection (h : independent_encounters p1 p2) :
  p0 * (1 - p1) * p2 = 1 / 8 := sorry

/- Proof goal 2: Mathematical expectation E(X) -/
def p_not_all_red (p : ℝ) : ℝ := 1 / 24

def p1_red (p0 p1 p2 : ℝ) : ℝ := p0 * (1 - p1) * (1 - p2) + (1 - p0) * p1 * (1 - p2) + (1 - p0) * (1 - p1) * p2

def p2_red (p0 p1 p2 : ℝ) : ℝ := p0 * p1 * (1 - p2) + p0 * (1 - p1) * p2 + (1 - p0) * p1 * p2

def p3_red (p : ℝ) : ℝ := 1 / 4

def distribution_X (p0 p1 p2 : ℝ) :=
  [p_not_all_red 1 / 24, p1_red p0 p1 p2, p2_red p0 p1 p2, p3_red 1 / 4]

def E_X (X : list ℝ) : ℝ := 0 * X.head! + 1 * X.tail!.head! + 2 * X.tail!.tail!.head! + 3 * X.tail!.tail!.tail!.head!

theorem expected_value_X (h : independent_encounters p1 p2) :
  E_X (distribution_X p0 p1 p2) = 23 / 12 := sorry

end red_light_first_time_at_third_intersection_expected_value_X_l809_809051


namespace proportional_value_l809_809191

theorem proportional_value :
  ∃ (x : ℝ), 18 / 60 / (12 / 60) = x / 6 ∧ x = 9 := sorry

end proportional_value_l809_809191


namespace date_analysis_l809_809429

noncomputable def mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def median (data : List ℕ) : ℚ :=
  let sorted := data.qsort (· < ·)
  let len := sorted.length
  if len % 2 == 1 then
    sorted[len / 2]
  else
    (sorted[len / 2 - 1] + sorted[len / 2]) / 2

def modes (data : List ℕ) : List ℕ :=
  let freq_map := data.foldl (λ m x => m.insert x (m.find x + 1)) (Std.RBMap.empty _ _)
  let max_freq := freq_map.fold (λ _ f acc => max acc f) 0
  freq_map.fold (λ k f acc => if f = max_freq then k :: acc else acc) []

def median_of_modes (data : List ℕ) : ℚ :=
  let mode_list := modes data
  median mode_list

theorem date_analysis (data : List ℕ)
  (h_data : data = List.concat (List.replicate 12 [1, 2, 3, ..., 29]) (List.concat (List.replicate 11 [30]) [31] ++ (List.replicate 7 [31]))) :
  let μ := mean data
  let M := median data
  let d := median_of_modes data
  d < μ ∧ μ < M := by
  sorry

end date_analysis_l809_809429


namespace packs_to_purchase_l809_809830

theorem packs_to_purchase {n m k : ℕ} (h : 8 * n + 15 * m + 30 * k = 135) : n + m + k = 5 :=
sorry

end packs_to_purchase_l809_809830


namespace bob_refund_amount_l809_809232

def total_packs : ℕ := 80
def expired_percentage : ℝ := 40
def price_per_pack : ℝ := 12
def refund_amount : ℝ := (expired_percentage / 100) * total_packs * price_per_pack

theorem bob_refund_amount : refund_amount = 384 := by
  -- the proof goes here
  sorry

end bob_refund_amount_l809_809232


namespace ratio_of_average_to_median_l809_809105

open_locale classical

variables (a b c : ℝ)

theorem ratio_of_average_to_median (h1 : a < b) (h2 : b < c) (h3 : a = 0) (h4 : c = 1.5 * b) :
  (a + b + c) / 3 / b = 5 / 6 :=
by {
  -- Start by substituting a, and using the given conditions
  have h : c / b = 1.5, from (eq_div_iff ((ne_of_lt h1).symm)).mpr h4,
  calc
    -- Rewrite a to 0
    (a + b + c) / 3 / b
        = (0 + b + c) / 3 / b : by rw h3
    ... = (b + c) / 3 / b : by simp
    ... = (b + 1.5 * b) / 3 / b : by rw h4
    ... = (2.5 * b) / 3 / b : by norm_num
    ... = 5 / 6 : by field_simp [mul_comm]
}

end ratio_of_average_to_median_l809_809105


namespace largest_base_5_three_digit_in_base_10_l809_809950

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l809_809950


namespace sum_of_digits_of_N_l809_809216

theorem sum_of_digits_of_N :
  ∃ (N : ℕ), (N * (N + 1)) / 2 = 5050 ∧ (N.digits 10).sum = 1 :=
begin
  sorry
end

end sum_of_digits_of_N_l809_809216


namespace tangent_line_equation_range_of_a_l809_809422

noncomputable def f (x a : ℝ) : ℝ := (2 * x ^ 2 - 4 * a * x) * Real.log x

-- Part 1: Prove the tangent line equation for a = 1 at x = 1
theorem tangent_line_equation (x y : ℝ) (h : x = 1) : 
    let a := 1 in
    f(a, 1) = 0 ∧ (f'(a, 1) = -2) →
    (2 * x + y - 2) = 0 :=
sorry

-- Part 2: Prove the range of a such that ∀ x ∈ [1, +∞), f(x) + x^2 - a > 0
theorem range_of_a (a : ℝ) :
    (∀ x : ℝ, 1 ≤ x → f(x, a) + x^2 - a > 0) → a < 1 :=
sorry

end tangent_line_equation_range_of_a_l809_809422


namespace total_dog_legs_l809_809921

theorem total_dog_legs (total_animals : ℕ) (frac_cats : ℚ) (legs_per_dog : ℕ)
  (h1 : total_animals = 300)
  (h2 : frac_cats = 2 / 3)
  (h3 : legs_per_dog = 4) :
  ∃ (num_cats num_dogs : ℕ), 
    (num_cats = (frac_cats * total_animals).to_nat) ∧
    (num_dogs = total_animals - num_cats) ∧
    (num_dogs * legs_per_dog = 400) := 
by
  sorry

end total_dog_legs_l809_809921


namespace largest_base5_three_digit_in_base10_l809_809954

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l809_809954


namespace mark_reads_1750_pages_per_week_l809_809064

def initialReadingHoursPerDay := 2
def increasePercentage := 150
def initialPagesPerDay := 100

def readingHoursPerDayAfterIncrease : Nat := initialReadingHoursPerDay + (initialReadingHoursPerDay * increasePercentage) / 100
def readingSpeedPerHour := initialPagesPerDay / initialReadingHoursPerDay
def pagesPerDayNow := readingHoursPerDayAfterIncrease * readingSpeedPerHour
def pagesPerWeekNow : Nat := pagesPerDayNow * 7

theorem mark_reads_1750_pages_per_week :
  pagesPerWeekNow = 1750 :=
sorry -- Proof omitted

end mark_reads_1750_pages_per_week_l809_809064


namespace negation_proposition_l809_809495

theorem negation_proposition : ¬ (∀ x : ℝ, (1 < x) → x^3 > x^(1/3)) ↔ ∃ x : ℝ, (1 < x) ∧ x^3 ≤ x^(1/3) := by
  sorry

end negation_proposition_l809_809495


namespace solution_of_inequality_system_l809_809749

theorem solution_of_inequality_system (a b : ℝ) 
    (h1 : 4 - 2 * a = 0)
    (h2 : (3 + b) / 2 = 1) : a + b = 1 := 
by 
  sorry

end solution_of_inequality_system_l809_809749


namespace sequence_formula_l809_809800

def f (x : ℝ) : ℝ := 2 * x / (2 + x)

noncomputable def a : ℕ+ → ℝ
| ⟨1, _⟩   := 1
| ⟨n+1, h⟩ := f (a ⟨n, Nat.succ_pos _⟩)

theorem sequence_formula (n : ℕ+) : a n = 2 / (n + 1 : ℕ) := 
sorry

end sequence_formula_l809_809800


namespace largest_base5_three_digits_is_124_l809_809933

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l809_809933


namespace minDistanceParabolaToLine_l809_809207

theorem minDistanceParabolaToLine :
  let L := fun x y => x - y + 2
  let parabola := fun x y => y^2 = 4 * x
  ∀ M : ℝ × ℝ, 
    (parabola M.1 M.2) →
    ∃ min_dist : ℝ, min_dist = sqrt 2 / 2 :=
by
  sorry

end minDistanceParabolaToLine_l809_809207


namespace smallest_solution_to_fractional_equation_l809_809670

theorem smallest_solution_to_fractional_equation :
  let eq := (fun x : ℝ => (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 12) in
  ∃ x : ℝ, eq x ∧ (∀ y : ℝ, eq y → (1 - real.sqrt 10 ≤ y)) → x = 1 - real.sqrt 10 :=
by
  let eq := (fun x : ℝ => (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 12)
  existsi (1 - real.sqrt 10)
  split
  sorry
  intro y hy
  sorry

end smallest_solution_to_fractional_equation_l809_809670


namespace find_x2_plus_y2_l809_809705

-- Given conditions as definitions in Lean
variable {x y : ℝ}
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x * y + x + y = 71)
variable (h4 : x^2 * y + x * y^2 = 880)

-- The statement to be proved
theorem find_x2_plus_y2 : x^2 + y^2 = 146 :=
by
  sorry

end find_x2_plus_y2_l809_809705


namespace determine_phi_l809_809918

theorem determine_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) (even_g : ∀ x, sin(2 * x + π/3 + φ) = sin(-2 * x + π/3 + φ)) : φ = π/6 :=
sorry

end determine_phi_l809_809918


namespace tumblers_count_correct_l809_809069

section MrsPetersonsTumblers

-- Define the cost of one tumbler
def tumbler_cost : ℕ := 45

-- Define the amount paid in total by Mrs. Petersons
def total_paid : ℕ := 5 * 100

-- Define the change received by Mrs. Petersons
def change_received : ℕ := 50

-- Calculate the total amount spent
def total_spent : ℕ := total_paid - change_received

-- Calculate the number of tumblers bought
def tumblers_bought : ℕ := total_spent / tumbler_cost

-- Prove the number of tumblers bought is 10
theorem tumblers_count_correct : tumblers_bought = 10 :=
  by
    -- Proof steps will be filled here
    sorry

end MrsPetersonsTumblers

end tumblers_count_correct_l809_809069


namespace simplify_sqrt_expression_correct_l809_809845

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x))

theorem simplify_sqrt_expression_correct (x : ℝ) : 
  simplify_sqrt_expression x = 120 * x * sqrt (x) := 
by 
  sorry

end simplify_sqrt_expression_correct_l809_809845


namespace non_empty_proper_subsets_count_l809_809256

theorem non_empty_proper_subsets_count (s : Finset ℕ) (h : s.card = 4) : 
  (2^s.card - 2 = 14) :=
by 
  -- s := {a, b, c, d}
  have h_card : s.card = 4 := h,
  rw h_card,
  sorry

end non_empty_proper_subsets_count_l809_809256


namespace largest_base5_three_digits_is_124_l809_809930

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l809_809930


namespace additional_charge_per_minute_atlantic_call_l809_809528

def base_rate_U : ℝ := 11.0
def rate_per_minute_U : ℝ := 0.25
def base_rate_A : ℝ := 12.0
def call_duration : ℝ := 20.0
variable (rate_per_minute_A : ℝ)

theorem additional_charge_per_minute_atlantic_call :
  base_rate_U + rate_per_minute_U * call_duration = base_rate_A + rate_per_minute_A * call_duration →
  rate_per_minute_A = 0.20 := by
  sorry

end additional_charge_per_minute_atlantic_call_l809_809528


namespace triangle_perimeter_proof_l809_809773

-- Definitions for the centers of the circles
variable (P Q R S T : Type*)

-- Assume the circles are given and their properties
def circles_tangent (radius : ℝ) (P Q R S T : Type*) : Prop :=
  ∀ (x : Type*), x ∈ {P, Q, R, S, T} → (radius = 2) ∧
  (∃ side_of_triangle, tangent_to_side x side_of_triangle) ∧
  (pairs_coords_tangent P Q) ∧
  (pairs_coords_tangent P R) ∧
  (pairs_coords_tangent P S) ∧
  (pairs_coords_tangent Q R) ∧
  (pairs_coords_tangent Q S) ∧
  (pairs_coords_tangent R S) ∧
  (tangent_to_two_sides T P)

-- Definition of the triangle perimeter
def triangle_perimeter (A B C : Type*) : ℝ :=
  side_length A B + side_length B C + side_length C A

theorem triangle_perimeter_proof
  (A B C : Type*)
  (P Q R S T : Type*)
  (h1 : P Q R S T)
  (h2 : circles_tangent 2 P Q R S T) :
  triangle_perimeter A B C = 4 * Real.sqrt 5 + 8 :=
sorry

end triangle_perimeter_proof_l809_809773


namespace cannot_be_sum_of_three_naturals_l809_809645

theorem cannot_be_sum_of_three_naturals (a : ℕ) : 
  (¬ ∃ x y z : ℕ, x + y + z = a ∧ (x * y * z).sqrt * (x * y * z).sqrt = x * y * z) ↔ 
  (a = 1 ∨ a = 2 ∨ a = 4) :=
begin
  sorry
end

end cannot_be_sum_of_three_naturals_l809_809645


namespace there_exist_at_least_three_elements_in_S_l809_809419
noncomputable def harmonic_sum_frac (n : ℕ) : ℚ :=
  (∑ i in Finset.range (n + 1), 1 / (i + 1))

def are_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def coprime_frac_decomposition (n : ℕ) (an bn : ℕ) : Prop :=
  harmonic_sum_frac n = (an : ℚ) / (bn : ℚ) ∧ are_coprime an bn

def S (p : ℕ) : Set ℕ :=
  {n | ∃ an bn, coprime_frac_decomposition n an bn ∧ p ∣ an}

theorem there_exist_at_least_three_elements_in_S (p : ℕ) (hp : Nat.Prime p) (hp5 : p ≥ 5) :
  ∃ n1 n2 n3 : ℕ, n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ∧ n1 ∈ S p ∧ n2 ∈ S p ∧ n3 ∈ S p :=
sorry

end there_exist_at_least_three_elements_in_S_l809_809419


namespace f_image_eq_T_l809_809045

variable {S : Type} [fintype S] 
variable (f : S → S)
variable (g : S → S)
variable (b : set (S → S)) (hb: b = {h : S → S | true})
variable (A : set (S → S)) (hA: A ∈ b)
variable (hf_not_g: g ∈ b ∧ g ≠ f → f ∘ g ∘ f ≠ g ∘ f ∘ g)

theorem f_image_eq_T (T : set S) (hT : T = f '' univ) :
  f '' T = T :=
sorry -- Proof omitted

end f_image_eq_T_l809_809045


namespace max_sum_product_four_numbers_l809_809505

theorem max_sum_product_four_numbers (a b c d : ℕ) (h : {a, b, c, d} = {2, 3, 4, 5}) :
    ab + bc + cd + da ≤ 48 :=
sorry

end max_sum_product_four_numbers_l809_809505


namespace remaining_problems_to_grade_l809_809557

-- Define the conditions
def problems_per_worksheet : ℕ := 3
def total_worksheets : ℕ := 15
def graded_worksheets : ℕ := 7

-- The remaining worksheets to grade
def remaining_worksheets : ℕ := total_worksheets - graded_worksheets

-- Theorems stating the amount of problems left to grade
theorem remaining_problems_to_grade : problems_per_worksheet * remaining_worksheets = 24 :=
by
  sorry

end remaining_problems_to_grade_l809_809557


namespace a_lt_c_lt_b_l809_809799

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.sqrt 2 * Real.sin (30.5 * Real.pi / 180) * Real.cos (30.5 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem a_lt_c_lt_b : a < c ∧ c < b := by
  sorry

end a_lt_c_lt_b_l809_809799


namespace find_points_l809_809277

open Real

def line (t : ℝ) : ℝ × ℝ := (-2 - sqrt 2 * t, 3 + sqrt 2 * t)

def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem find_points : ∃ t1 t2 : ℝ, 
  (t1 = 1 ∨ t1 = -1) ∧ (t2 = 1 ∨ t2 = -1) ∧ 
  distance (-2, 3) (line t1) = sqrt 2 ∧ distance (-2, 3) (line t2) = sqrt 2 ∧
  (line t1 = (-3, 4) ∨ line t1 = (-1, 2)) ∧ 
  (line t2 = (-3, 4) ∨ line t2 = (-1, 2)) := 
by 
sorרי

end find_points_l809_809277


namespace solve_thought_of_number_l809_809513

def thought_of_number (x : ℝ) : Prop :=
  (x / 6) + 5 = 17

theorem solve_thought_of_number :
  ∃ x, thought_of_number x ∧ x = 72 :=
by
  sorry

end solve_thought_of_number_l809_809513


namespace log_comparison_l809_809682

variable (a b : ℝ) (h : 0 < a ∧ a < b ∧ b < 1)

theorem log_comparison (h : 0 < a ∧ a < b ∧ b < 1) : 
  Real.log a⁻¹ 3 > Real.log b⁻¹ 3 :=
by
  sorry

end log_comparison_l809_809682


namespace fraction_spent_at_arcade_l809_809215

theorem fraction_spent_at_arcade 
  (weekly_allowance : ℝ) (x : ℝ)
  (arcade_spent : x * weekly_allowance)
  (toy_store_spent : (1 / 3) * (weekly_allowance - arcade_spent))
  (candy_store_spent : weekly_allowance - arcade_spent - toy_store_spent = 1.20)
  (weekly_allowance_eq : weekly_allowance = 4.50) :
  x = 3 / 5 :=
by
  sorry

end fraction_spent_at_arcade_l809_809215


namespace largest_base5_three_digit_in_base10_l809_809952

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l809_809952


namespace largest_base5_three_digits_is_124_l809_809932

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l809_809932


namespace days_after_power_of_two_two_2016_mod_7_day_of_week_after_2_2016_days_l809_809916

theorem days_after_power_of_two (n : ℕ) (h : n ≡ 1 [MOD 7]) : 
  (n + 4) % 7 = 5 := 
by
  sorry

theorem two_2016_mod_7 : (2^2016 % 7) = 1 := 
by
  sorry

theorem day_of_week_after_2_2016_days : 
  let today := 4 in
  (2^2016 + today) % 7 = 5 :=
by
  have h : (2^2016 % 7) = 1 := two_2016_mod_7
  exact days_after_power_of_two (2^2016) h

end days_after_power_of_two_two_2016_mod_7_day_of_week_after_2_2016_days_l809_809916


namespace hyperbola_condition_l809_809870

theorem hyperbola_condition (a b c : ℝ) :
  (ab < 0 ↔ ∃ x y : ℝ, ax^2 + by^2 = c ∧ (ax = 0 ∨ by = 0)) ∧ ¬ (∀ x y : ℝ, ax^2 + by^2 = c → ab < 0) :=
by sorry

end hyperbola_condition_l809_809870


namespace two_digit_reverse_square_count_l809_809126

theorem two_digit_reverse_square_count :
  let two_digit_num (a b : ℕ) := a * 10 + b
  let reverse_num (a b : ℕ) := b * 10 + a
  let valid_sum (a b : ℕ) := 11 * (a + b)
  ∃ (l : List (Σ a b : ℕ, a + b = 11)),
  List.length (List.filter (λ (p : Σ a b, a + b = 11),
    let ab := two_digit_num p.1 p.2
    let ba := reverse_num p.1 p.2
    (∃ c : ℕ, valid_sum p.1 p.2 = c * c)
  ) l) = 8 :=
begin
  sorry
end

end two_digit_reverse_square_count_l809_809126


namespace rotated_angle_170_degrees_l809_809881

theorem rotated_angle_170_degrees (theta : ℝ) (h_theta : theta = 70) : 
  ∃ new_theta : ℝ, new_theta = 170 ∧
  (∃ k : ℤ, 360 * k + 70 + 960 = new_theta) :=
by {
  use 170,
  split,
  exact rfl,
  use 2,
  linarith,
}

end rotated_angle_170_degrees_l809_809881


namespace translated_parabola_expression_correct_l809_809520

-- Definitions based on the conditions
def original_parabola (x : ℝ) : ℝ := x^2 - 1
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- The theorem to prove
theorem translated_parabola_expression_correct :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 1 :=
by
  sorry

end translated_parabola_expression_correct_l809_809520


namespace algebra_geometry_probabilities_l809_809762

theorem algebra_geometry_probabilities :
  let total := 5
  let algebra := 2
  let geometry := 3
  let prob_first_algebra := algebra / total
  let prob_second_geometry_after_algebra := geometry / (total - 1)
  let prob_both := prob_first_algebra * prob_second_geometry_after_algebra
  let total_after_first_algebra := total - 1
  let remaining_geometry := geometry
  prob_both = 3 / 10 ∧ remaining_geometry / total_after_first_algebra = 3 / 4 :=
by
  sorry

end algebra_geometry_probabilities_l809_809762


namespace zeros_of_f_l809_809135

def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 4 else x^2 - 4

theorem zeros_of_f : {x : ℝ | f x = 0} = {-4, 2} :=
sorry

end zeros_of_f_l809_809135


namespace restaurant_ratio_l809_809606

theorem restaurant_ratio (W : ℕ) (h1 : 45 = W + 12) :
    ∃ W', 9 / W' = 1 / 3 :=
by {
  have h2 : W = 33,
  {
    linarith,
  },
  exact ⟨33, by norm_num⟩
}


end restaurant_ratio_l809_809606


namespace part1_part2_l809_809013

variables (A B C : ℝ) (a b c : ℝ)
variables (cosC cosB : ℝ) (D : ℝ)

-- Definitions using conditions
def is_midpoint (x y z : ℝ) : Prop := (x + y) / 2 = z
def is_triangle_ABC (a b c : ℝ) (cosC cosB : ℝ) : Prop := 
  a + b = 5 ∧ (2 * a + b) * cosC + c * cosB = 0

def area_triangle (a b : ℝ) : ℝ := (a * b * (sqrt 3)) / 4

-- Proof statements
theorem part1 (h₁ : is_triangle_ABC a b c cosC cosB) 
              (h₂ : area_triangle a b = sqrt 3 / 2) : 
              c = sqrt 23 :=
sorry

theorem part2 (h₁ : is_triangle_ABC a b c cosC cosB) 
              (h₂ : is_midpoint a b D) (h₃ : D = 30) : 
              a = 5 / 3 ∧ b = 10 / 3 :=
sorry

end part1_part2_l809_809013


namespace equidecomposable_transitive_l809_809445

-- Definition of equidecomposable (to be further refined before the final proof)
def equidecomposable (P Q : Type) : Prop :=
  ∃ pieces : List P, ∀ piece ∈ pieces, ∃ rearrangement : List Q, rearrangement = pieces

-- Given conditions
variables (P Q R : Type)
variable (h1 : equidecomposable P R)
variable (h2 : equidecomposable Q R)

-- Proving the main statement
theorem equidecomposable_transitive : equidecomposable P Q :=
sorry

end equidecomposable_transitive_l809_809445


namespace max_good_points_l809_809906

/-- 
  There are 7 lines in the plane. A point is called a *good* point if it is 
  contained on at least three of these seven lines. Let's prove that the 
  maximum number of good points is 6.
-/
theorem max_good_points (L : Finset (Set (EuclideanSpace ℝ 2))) (hL : L.card = 7) :
  ∃ max_good_points, max_good_points ≤ 6 ∧ 
  (∀ P : EuclideanSpace ℝ 2, (∃ T : Finset (Set (EuclideanSpace ℝ 2)), T ⊆ L ∧ 
  T.card ≥ 3 ∧ ∀ l ∈ T, P ∈ l) → ¬∃ T' : Finset (Set (EuclideanSpace ℝ 2)), 
  T' ⊆ L ∧ T'.card ≥ 7 ∧ ∀ l ∈ T', P ∈ l) := 
sorry

end max_good_points_l809_809906


namespace operations_equivalent_l809_809362

theorem operations_equivalent (x : ℚ) : 
  ((x * (5 / 6)) / (2 / 3) - 2) = (x * (5 / 4) - 2) :=
sorry

end operations_equivalent_l809_809362


namespace balance_balls_possible_l809_809564

variable {ι : Type} [Fintype ι] [DecidableEq ι]

def m : Fin 10 → ℝ
def x (i : Fin 10) : ℝ := |m i - m ((i + 1) % 10)|

theorem balance_balls_possible :
  ∃ (s : Fin 10 → bool),
  ∑ i, (if s i then x i else 0) = ∑ i, (if s i = false then x i else 0) :=
by
suffices h : ∑ i, x i = 2 * ∑ i, (if s i then x i else 0),
{ use s,
  simp_rw [mul_eq_mul_left_iff, Ne.def, one_ne_zero, or_false] at h,
  exact h },
sorry

end balance_balls_possible_l809_809564


namespace EF_vector_in_parallelogram_l809_809706

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

def midpoint (A B : V) : V := (A + B) / 2

def parallelogram (A B C D : V) : Prop :=
  (B - A = D - C) ∧ (C - B = A - D)

noncomputable def EF_vector (A B C D E F : V) 
  (h_parallelogram : parallelogram A B C D)
  (hE : E = midpoint C D)
  (hF : F = midpoint B C) : V := 
  (E - F)

theorem EF_vector_in_parallelogram (A B C D E F : V) 
  (h_parallelogram : parallelogram A B C D)
  (hE : E = midpoint C D)
  (hF : F = midpoint B C) :
  EF_vector A B C D E F h_parallelogram hE hF = 
  (1/2 : ℝ) • (B - A) - (1/2 : ℝ) • (D - A) :=
sorry

end EF_vector_in_parallelogram_l809_809706


namespace count_numbers_neither_divisible_by_6_nor_8_l809_809610

theorem count_numbers_neither_divisible_by_6_nor_8 :
  let total := 499
  let divisible_by_6 := Nat.floor (499 / 6)
  let divisible_by_8 := Nat.floor (499 / 8)
  let divisible_by_6_and_8 := Nat.floor (499 / 24)
  let total_divisible := divisible_by_6 + divisible_by_8 - divisible_by_6_and_8
  let not_divisible := total - total_divisible
  not_divisible = 374 := by
  let total := 499
  let divisible_by_6 := 83
  let divisible_by_8 := 62
  let divisible_by_6_and_8 := 20
  let total_divisible := 83 + 62 - 20
  let not_divisible := 499 - total_divisible
  show not_divisible = 374
  sorry

end count_numbers_neither_divisible_by_6_nor_8_l809_809610


namespace divides_14n_eq_n_l809_809747

noncomputable def has_three_prime_factors (n : ℕ) : Prop :=
  ∃ p1 p2 p3 : ℕ, p1.prime ∧ p2.prime ∧ p3.prime ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3

theorem divides_14n_eq_n (n : ℕ) (h1 : n < 200) (h2 : has_three_prime_factors n) (d : ℕ) (h3 : 14 * n % d = 0) : d = n :=
sorry

end divides_14n_eq_n_l809_809747


namespace inscribed_circle_radius_l809_809878

noncomputable def radius_of_inscribed_circle (a b c : ℝ) : ℝ :=
  - (b + Real.sqrt (b^2 - 2 * a * c)) / (2 * a)

theorem inscribed_circle_radius (a b c : ℝ) (h_a : a ≠ 0) (h_b2_2ac_ge_0 : b^2 - 2 * a * c ≥ 0) :
  ∃ r : ℝ, r = radius_of_inscribed_circle a b c :=
begin
  use radius_of_inscribed_circle a b c,
  simp [radius_of_inscribed_circle],
  sorry
end

end inscribed_circle_radius_l809_809878


namespace simple_interest_rate_l809_809173

theorem simple_interest_rate (P A T : ℝ) (R : ℝ) (hP : P = 750) (hA : A = 900) (hT : T = 5) :
    (A - P) = (P * R * T) / 100 → R = 4 := by
  sorry

end simple_interest_rate_l809_809173


namespace part1_part2_l809_809818

noncomputable def f (x a : ℝ) := (x + 1) * Real.log x - a * (x - 1)

theorem part1 : (∀ x a : ℝ, (x + 1) * Real.log x - a * (x - 1) = x - 1 → a = 1) := 
by sorry

theorem part2 (x : ℝ) (h : 1 < x ∧ x < 2) : 
  ( 1 / Real.log x - 1 / Real.log (x - 1) < 1 / ((x - 1) * (2 - x))) :=
by sorry

end part1_part2_l809_809818


namespace total_distance_walked_l809_809455

theorem total_distance_walked (area_of_each_square : ℝ) (num_squares_per_side : ℕ)
    (total_squares : ℕ) (side_length : ℝ)
    (outer_edges_length : ℝ) (inner_edges_length : ℝ)
    (total_distance : ℝ) :
    area_of_each_square = 400 → 
    num_squares_per_side = 4 →  
    total_squares = 16 → 
    side_length = 20 →  
    outer_edges_length = 4 * num_squares_per_side * side_length → 
    inner_edges_length = 4 * (num_squares_per_side - 1) * 2 * side_length →
    total_distance = outer_edges_length + inner_edges_length → 
    total_distance = 800 :=
by { intros, sorry }

end total_distance_walked_l809_809455


namespace students_not_wearing_glasses_l809_809904

theorem students_not_wearing_glasses (total_students : ℕ) (students_wear_glasses : ℕ → ℕ) (h : students_wear_glasses (total_students - students_wear_glasses total_students) = (3 / 5) * (total_students - students_wear_glasses total_students)) : 
  total_students = 240 → ∃ x, x = total_students - students_wear_glasses total_students ∧ x = 150 :=
by 
  intros ht
  use 150
  split
  sorry
  sorry

end students_not_wearing_glasses_l809_809904


namespace terry_lunch_combos_l809_809464

def num_lettuce : ℕ := 2
def num_tomatoes : ℕ := 3
def num_olives : ℕ := 4
def num_soups : ℕ := 2

theorem terry_lunch_combos : num_lettuce * num_tomatoes * num_olives * num_soups = 48 :=
by
  sorry

end terry_lunch_combos_l809_809464


namespace algae_difference_l809_809825

theorem algae_difference :
  let original_algae := 809
  let current_algae := 3263
  current_algae - original_algae = 2454 :=
by
  sorry

end algae_difference_l809_809825


namespace largest_base5_three_digit_to_base10_l809_809963

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l809_809963


namespace amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l809_809663

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 2) + 1

theorem amplitude_of_f : (∀ x y : ℝ, |f x - f y| ≤ 2 * |x - y|) := sorry

theorem phase_shift_of_f : (∃ φ : ℝ, φ = -Real.pi / 8) := sorry

theorem vertical_shift_of_f : (∃ v : ℝ, v = 1) := sorry

end amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l809_809663


namespace sum_of_coefficients_l809_809737

theorem sum_of_coefficients {x : ℝ} :
  let a := (3 * x - 1)^7
  in (a.coeff 7 + a.coeff 6 + a.coeff 5 + a.coeff 4 + a.coeff 3 + a.coeff 2 + a.coeff 1) = 2186 := by
sorry

end sum_of_coefficients_l809_809737


namespace trent_keeps_tadpoles_l809_809820

variable (x : ℕ)

-- Define the conditions
def tadpoles_caught (x : ℕ) : ℕ := x
def percentage_let_go : ℝ := 82.5
def percentage_kept : ℝ := 100 - percentage_let_go
def kept_tadpoles (x : ℕ) : ℝ := (percentage_kept / 100) * x

-- Prove the number of tadpoles kept is 0.175 * x
theorem trent_keeps_tadpoles (x : ℕ) : kept_tadpoles x = 0.175 * x :=
by
  sorry

end trent_keeps_tadpoles_l809_809820


namespace interpretation_of_neg_two_pow_six_l809_809880

theorem interpretation_of_neg_two_pow_six :
  - (2^6) = -(6 * 2) :=
by
  sorry

end interpretation_of_neg_two_pow_six_l809_809880


namespace red_light_first_time_at_third_intersection_expected_value_X_l809_809052

/- Definitions based on provided conditions -/
def p0 : ℝ := 1 / 2

variable (p1 p2 : ℝ)

def independent_encounters :=
  (1 - p0) * (1 - p1) * (1 - p2) = 1 / 24 ∧
  p0 * p1 * p2 = 1 / 4

/- Proof goal 1: Probability that Li Ping encounters a red light for the first time at the third intersection -/
theorem red_light_first_time_at_third_intersection (h : independent_encounters p1 p2) :
  p0 * (1 - p1) * p2 = 1 / 8 := sorry

/- Proof goal 2: Mathematical expectation E(X) -/
def p_not_all_red (p : ℝ) : ℝ := 1 / 24

def p1_red (p0 p1 p2 : ℝ) : ℝ := p0 * (1 - p1) * (1 - p2) + (1 - p0) * p1 * (1 - p2) + (1 - p0) * (1 - p1) * p2

def p2_red (p0 p1 p2 : ℝ) : ℝ := p0 * p1 * (1 - p2) + p0 * (1 - p1) * p2 + (1 - p0) * p1 * p2

def p3_red (p : ℝ) : ℝ := 1 / 4

def distribution_X (p0 p1 p2 : ℝ) :=
  [p_not_all_red 1 / 24, p1_red p0 p1 p2, p2_red p0 p1 p2, p3_red 1 / 4]

def E_X (X : list ℝ) : ℝ := 0 * X.head! + 1 * X.tail!.head! + 2 * X.tail!.tail!.head! + 3 * X.tail!.tail!.tail!.head!

theorem expected_value_X (h : independent_encounters p1 p2) :
  E_X (distribution_X p0 p1 p2) = 23 / 12 := sorry

end red_light_first_time_at_third_intersection_expected_value_X_l809_809052


namespace count_valid_numbers_l809_809345

-- Defining the conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def sum_of_digits_is (n sum : ℕ) : Prop :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  a + b + c + d = sum

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The main theorem statement
theorem count_valid_numbers : 
  (number_of_valid_numbers : ℕ) = 
  (finset.card (finset.filter (λ n, four_digit_number n ∧ sum_of_digits_is n 12 ∧ divisible_by_5 n) (finset.range 10000))) :=
127

end count_valid_numbers_l809_809345


namespace circle_chord_intersect_zero_l809_809381

noncomputable def circle_product (r : ℝ) : ℝ :=
  let O := (0, 0)
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B)

theorem circle_chord_intersect_zero (r : ℝ) :
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B) = 0 :=
by sorry

end circle_chord_intersect_zero_l809_809381


namespace fractional_sum_l809_809289

-- Defining the greatest integer function
def greatest_integer (x : ℝ) : ℤ := int.floor x

-- Defining the fractional part
def fractional_part (x : ℝ) : ℝ := x - greatest_integer x

theorem fractional_sum :
  ∑ n in Finset.range 2014, fractional_part ((2014 : ℝ)^((n + 1) : ℕ) / 2015) = 1007 :=
by
  sorry

end fractional_sum_l809_809289


namespace largest_base5_three_digit_in_base10_l809_809955

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l809_809955


namespace sufficient_but_not_necessary_l809_809700

theorem sufficient_but_not_necessary (a b : ℝ) : (ab >= 2) -> a^2 + b^2 >= 4 ∧ ∃ a b : ℝ, a^2 + b^2 >= 4 ∧ ab < 2 := by
  sorry

end sufficient_but_not_necessary_l809_809700


namespace expected_number_of_letters_in_mailbox_A_l809_809525

def prob_xi_0 : ℚ := 4 / 9
def prob_xi_1 : ℚ := 4 / 9
def prob_xi_2 : ℚ := 1 / 9

def expected_xi := 0 * prob_xi_0 + 1 * prob_xi_1 + 2 * prob_xi_2

theorem expected_number_of_letters_in_mailbox_A :
  expected_xi = 2 / 3 := by
  sorry

end expected_number_of_letters_in_mailbox_A_l809_809525


namespace dice_labeling_possible_l809_809186

theorem dice_labeling_possible : 
  ∃ (die1 : Fin 6 → ℕ) (die2 : Fin 6 → ℕ), 
  (∀ x1 x2 : Fin 6, let sums := {s | ∃ (a b : ℕ), a = die1 x1 ∧ b = die2 x2 ∧ s = a + b} in sums = (Finset.range 36).image (λ n, n + 1)) :=
sorry

end dice_labeling_possible_l809_809186


namespace coefficient_x_5_2_in_expansion_of_2x_minus_sqrt_inv_x_l809_809748

noncomputable def binomialCoefficient (n k : ℕ) : ℚ := (Nat.factorial n : ℚ) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem coefficient_x_5_2_in_expansion_of_2x_minus_sqrt_inv_x (n : ℕ) (h : n = 4) :
  let term := λ r, binomialCoefficient 4 r * (2 ^ (4 - r)) * ((-1) ^ r) * (x ^ (4 - (3 / 2) * r)) in
  term 1 = -32 * (x ^ (5 / 2)) :=
by
  sorry

end coefficient_x_5_2_in_expansion_of_2x_minus_sqrt_inv_x_l809_809748


namespace repeating_decimal_fraction_l809_809540

theorem repeating_decimal_fraction :
  let x := (37/100) + (246 / 99900)
  in x = 37245 / 99900 :=
by
  let x := (37/100) + (246 / 99900)
  show x = 37245 / 99900
  sorry

end repeating_decimal_fraction_l809_809540


namespace prime_divisor_congruence_l809_809810

theorem prime_divisor_congruence (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_div : q ∣ (1 + p + p^2 + ... + p^(p-1))) : q ≡ 1 [MOD p] :=
sorry

end prime_divisor_congruence_l809_809810


namespace prob_first_red_third_expected_num_red_lights_l809_809050

variables (p1 p2 p3 : ℝ) (X : ℕ → ℝ)
variables (cond1 : p1 = 1/2)
variables (cond2 : p2 > p1 ∧ p3 > p2)
variables (cond3 : (1 - p1) * (1 - p2) * (1 - p3) = 1/24)
variables (cond4 : p1 * p2 * p3 = 1/4)
variables (cond5 : ∀ n m, n ≠ m → ∃ (EventIndependent n m))

-- Part 1
theorem prob_first_red_third : (p1 * (1 - p2) * p3) = 1/8 := 
by {
  sorry
}

-- Part 2
theorem expected_num_red_lights : (Expectation X) = 23/12 := 
by {
  sorry
}

end prob_first_red_third_expected_num_red_lights_l809_809050


namespace car_stops_at_three_seconds_l809_809864

theorem car_stops_at_three_seconds (t : ℝ) (h : -3 * t^2 + 18 * t = 0) : t = 3 := 
sorry

end car_stops_at_three_seconds_l809_809864


namespace log_inequality_l809_809832

theorem log_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a)) 
    ≥ 9 / (a + b + c) :=
by
  sorry

end log_inequality_l809_809832


namespace abundant_numbers_count_below_35_l809_809633

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d, d ≠ n ∧ n % d = 0)

def is_abundant (n : ℕ) : Bool :=
  n < (proper_divisors n).sum

def abundant_numbers_below_35 : List ℕ :=
  (List.range 35).filter is_abundant

theorem abundant_numbers_count_below_35 : abundant_numbers_below_35.length = 5 := by
  sorry

end abundant_numbers_count_below_35_l809_809633


namespace smallest_positive_period_f_max_min_values_f_l809_809328

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * (cos x) ^ 2

theorem smallest_positive_period_f :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧
  (∀ q > 0, (∀ x, f (x + q) = f x) → p ≤ q ∧ (p < q → false)) :=
sorry

theorem max_min_values_f : 
  ∀ x, x ∈ set.Icc (-π / 3) (π / 3) →
  ∃ a b, f x ≤ a ∧ b ≤ f x ∧
  a = 2 + sqrt 3 ∧ b = 0 :=
sorry

end smallest_positive_period_f_max_min_values_f_l809_809328


namespace solve_inequality_l809_809650

noncomputable def inequality_solution : Set ℝ :=
  { x | x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4 }

theorem solve_inequality :
  inequality_solution = { x | -2 < x ∧ x < 2 } ∪ { x | 3 ≤ x } :=
by
  sorry

end solve_inequality_l809_809650


namespace inverse_proportion_equation_l809_809876

def inverse_proportion_function (k x : ℝ) : ℝ := k / x

theorem inverse_proportion_equation (k : ℝ) (x y : ℝ) (hx : x ≠ 0) (h : inverse_proportion_function k x = y) : 
    (inverse_proportion_function k (-2) = 3) → 
    inverse_proportion_function (-6) x = y := 
by
   sorry

end inverse_proportion_equation_l809_809876


namespace find_v_l809_809630

def A := ![![1, 2], ![2, 1]] : Matrix (Fin 2) (Fin 2) ℚ

theorem find_v :
  let v := ![1/17, 4/23] : Fin 2 → ℚ,
      I := 1,
      A2 := matMul A A,
      A4 := matMul A2 A2,
      A6 := matMul A4 A2,
      A8 := matMul A4 A4,
      A10 := matMul A8 A2 in
  (A10 + 2 * A8 + 3 * A6 + 4 * A4 + 5 * A2 + I) ⬝ v = ![13, 5] :=
by
  sorry

end find_v_l809_809630


namespace largest_base_5_three_digit_in_base_10_l809_809947

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l809_809947


namespace animals_per_aquarium_l809_809344

theorem animals_per_aquarium (total_animals : ℕ) (number_of_aquariums : ℕ) (h1 : total_animals = 40) (h2 : number_of_aquariums = 20) : 
  total_animals / number_of_aquariums = 2 :=
by
  sorry

end animals_per_aquarium_l809_809344


namespace mark_pages_per_week_l809_809067

theorem mark_pages_per_week :
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100) * initial_reading_hours
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  increased_pages_per_day * days_per_week = 1750 :=
by
  -- Definitions
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  
  -- Calculate increased reading hours per day
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100.0) * initial_reading_hours
  -- Calculate increased pages per day
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  
  -- Calculate pages per week
  have h : increased_pages_per_day * days_per_week = 1750 := by
    sorry

  exact h

end mark_pages_per_week_l809_809067


namespace smallest_int_ending_in_9_divisible_by_11_l809_809156

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end smallest_int_ending_in_9_divisible_by_11_l809_809156


namespace largest_base5_eq_124_l809_809941

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l809_809941


namespace conditional_expectation_property_l809_809409

variables {Ω : Type*} [measurable_space Ω] {P : probability_measure Ω}

variables {η ξ : Ω → ℝ}
variables {𝒢 𝒢' : measurable_space Ω}
variables [sub measurable_space Ω 𝒢] [sub measurable_space Ω 𝒢']

def integrable (f : Ω → ℝ) : Prop :=
  ∫⁻ x, |f x| ∂P < ⊤

noncomputable
def p_exponent : Prop := 
  ∃ (p q : ℝ), 1 < p ∧ 1 < q ∧ 1 / p + 1 / q = 1

theorem conditional_expectation_property
  (h_measurable_η : measurable[𝒢] η)
  (h_measurable_ξ : measurable[𝒢'] ξ)
  (h_integrable_η : integrable (λ x, |η x|^q)) 
  (h_integrable_ξ : integrable (λ x, |ξ x|^p))
  (h_exponents : p_exponent) :
  ∀ᵖ x ∂P, condexp P 𝒢 (ξ * η) x = η x * condexp P 𝒢 ξ x :=
sorry

end conditional_expectation_property_l809_809409


namespace count_positive_even_less_than_100_contains_5_or_9_l809_809990

def contains_5_or_9 (n : ℕ) : Prop :=
  n.to_digits 10 = List.filter (λ d, d = 5 ∨ d = 9) n.to_digits 10 ≠ []

def even (n : ℕ) : Prop :=
  n % 2 = 0

def less_than_100 (n : ℕ) : Prop :=
  n < 100

def positive (n : ℕ) : Prop :=
  n > 0

theorem count_positive_even_less_than_100_contains_5_or_9 :
  ∃! S : Finset ℕ, S.card = 10 ∧ ∀ x, x ∈ S ↔ positive x ∧ even x ∧ less_than_100 x ∧ contains_5_or_9 x := by
  sorry

end count_positive_even_less_than_100_contains_5_or_9_l809_809990


namespace largest_base5_eq_124_l809_809942

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l809_809942


namespace analogy_reasoning_conducts_electricity_l809_809887

theorem analogy_reasoning_conducts_electricity (Gold Silver Copper Iron : Prop) (conducts : Prop)
  (h1 : Gold) (h2 : Silver) (h3 : Copper) (h4 : Iron) :
  (Gold ∧ Silver ∧ Copper ∧ Iron → conducts) → (conducts → !CompleteInductive ∧ !Inductive ∧ !Deductive ∧ Analogical) :=
by
  sorry

end analogy_reasoning_conducts_electricity_l809_809887


namespace dima_numbers_l809_809863

theorem dima_numbers (a b : ℕ) (h_cond1 : a ∈ {1, 2, 3, 4, 6}) (h_cond2 : b ∈ {1, 2, 3, 4, 6}) :
  (a ≠ b ∧ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (3, 6) ∨ (a, b) = (6, 3)) ∨ (a = b ∧ (a, b) = (1, 1) ∨ (a, b) = (4, 4))
:=
sorry

end dima_numbers_l809_809863


namespace largest_base5_three_digit_to_base10_l809_809965

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l809_809965


namespace no_solution_inequalities_l809_809751

theorem no_solution_inequalities (m : ℝ) : 
  (∀ x : ℝ, ¬ (x - m > 2 ∧ x - 2m < -1)) ↔ (m ≤ 3) :=
by
  sorry

end no_solution_inequalities_l809_809751


namespace max_m_for_factored_polynomial_l809_809635

theorem max_m_for_factored_polynomial :
  ∃ m, (∀ A B : ℤ, (5 * x ^ 2 + m * x + 45 = (5 * x + A) * (x + B) → AB = 45) → 
    m = 226) :=
sorry

end max_m_for_factored_polynomial_l809_809635


namespace conditional_probability_A_given_B_l809_809467

noncomputable def P (A B : Prop) : ℝ := sorry -- Placeholder for the probability function

variables (A B : Prop)

axiom P_A_def : P A = 4/15
axiom P_B_def : P B = 2/15
axiom P_AB_def : P (A ∧ B) = 1/10

theorem conditional_probability_A_given_B : P (A ∧ B) / P B = 3/4 :=
by
  rw [P_AB_def, P_B_def]
  norm_num
  sorry

end conditional_probability_A_given_B_l809_809467


namespace polynomial_product_mod_p_l809_809398

theorem polynomial_product_mod_p
  (f : Polynomial ℤ) (p : ℕ) (hp : Nat.Prime p)
  (z : Fin (p - 1) → ℂ)
  (hz : ∀ i, (z i)^(p - 1) = 1 ∧ (∀ j, i ≠ j → z i ≠ z j)) :
  (∏ i in Finset.range (p - 1), f.eval (z i)) % p = (∏ k in Finset.range (p - 1), f.eval (k : ℤ)) % p := sorry

end polynomial_product_mod_p_l809_809398


namespace a_general_term_c_seq_sum_gt_m_a_fraction_ineq_l809_809723

noncomputable def f (x : ℝ) : ℝ := 4 * x + 1
noncomputable def g (x : ℝ) : ℝ := 2 * x

def a_seq (n : ℕ) : ℝ := if n = 0 then 1 else 2^n - 1
def b_seq (n : ℕ) : ℝ := a_seq n / 2
def c_seq (n : ℕ) : ℝ := 1 / ((2 * n + 1) * (2 * n + 3))

def T (n : ℕ) : ℝ := (List.range n).sum c_seq

theorem a_general_term : ∀ n : ℕ, a_seq n = 2^n - 1 :=
by sorry

theorem c_seq_sum_gt_m (m : ℕ) : T <| m > m / 150 :=
by sorry

theorem a_fraction_ineq (n : ℕ) : 
  (List.range n).sum (λ k, a_seq k / a_seq (k + 1)) > n / 2 - 1 / 3 :=
by sorry

end a_general_term_c_seq_sum_gt_m_a_fraction_ineq_l809_809723


namespace cos_alpha_identity_l809_809827

theorem cos_alpha_identity :
  ∀ (α : ℝ) (m n p : ℝ),
    (cos (2 * α) = 2 * (cos α)^2 - 1) →
    (cos (4 * α) = 8 * (cos α)^4 - 8 * (cos α)^2 + 1) →
    (cos (6 * α) = 32 * (cos α)^6 - 48 * (cos α)^4 + 18 * (cos α)^2 - 1) →
    (cos (8 * α) = 128 * (cos α)^8 - 256 * (cos α)^6 + 160 * (cos α)^4 - 32 * (cos α)^2 + 1) →
    (cos (10 * α) = m * (cos α)^10 - 1280 * (cos α)^8 + 1120 * (cos α)^6 + n * (cos α)^4 + p * (cos α)^2 - 1) →
    m - n + p = 962 :=
by
  intros α m n p h1 h2 h3 h4 h5
  sorry

end cos_alpha_identity_l809_809827


namespace circle_center_l809_809871

theorem circle_center :
  ∀ x y : ℝ, (x^2 + y^2 - 2*x + 2*y = 0) → (1, -1) :=
by
  intro x y h
  sorry

end circle_center_l809_809871


namespace minimum_even_numbers_l809_809507

theorem minimum_even_numbers (n : ℕ) (h : n = 101)
  (even_count_cond : ∀ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → (∀ j, 0 ≤ j ∧ j < 3 → even (a ((i + j - 1) % n) ))) → ∃ k, even (a k)) :
  ∃ m, m = 34 :=
by {
  sorry
}

end minimum_even_numbers_l809_809507


namespace balearic_1989_q6_l809_809809

noncomputable def polynomial_irreducible (n : ℕ) (a : fin (n+1) → ℕ) : Prop :=
  ∀ (P g h : polynomial ℤ),
    (P = polynomial.sum fin.val (λ i, polynomial.C (a i) * polynomial.X ^ i)) →
    (polynomial.degree g > 0 ∧ polynomial.degree h > 0) →
    (P = g * h) → false

theorem balearic_1989_q6 (n : ℕ) (a : fin (n+1) → ℕ) (h1 : 1 < n) (h2 : 1 < a ⟨n, nat.lt_succ_self n⟩)
    (h3 : nat.prime (∑ i, a ⟨i, (finset.mem_fin (n+1) i).elim⟩ * 10 ^ i)) :
  polynomial_irreducible n a :=
sorry

end balearic_1989_q6_l809_809809


namespace g_neg_1002_l809_809418

noncomputable def g : ℝ → ℝ := -- definition of the function g

axiom g_function : ∀ x y : ℝ, g (x * y) + x = x * g y + g x

axiom g_minus_one : g (-1) = 5

theorem g_neg_1002 : g (-1002) = 2007 :=
sorry

end g_neg_1002_l809_809418


namespace total_surface_area_correct_l809_809287

-- Defining the volumes of the cubes
def volumes : List ℕ := [1, 27, 125, 343, 729]

-- Function to calculate the side length from the volume of a cube
def side_length (v : ℕ) : ℕ := Int.natAbs ⟨Real.cbrt v⟩

-- Function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ := 6 * s * s

-- Individual side lengths of cubes
def sides : List ℕ := volumes.map side_length

-- Surface areas without overlaps
def surface_areas : List ℕ := sides.map surface_area

-- Adjusting for overlaps
def adjusted_surface_areas : List ℕ := 
  [surface_areas.nthLe 0 sorry,                               -- Cube with side length 1
   surface_areas.nthLe 1 sorry - sides.nthLe 1 sorry ^ 2,      -- Cube with side length 3
   surface_areas.nthLe 2 sorry - (sides.nthLe 2 sorry ^ 2 / 2),-- Cube with side length 5
   surface_areas.nthLe 3 sorry - sides.nthLe 3 sorry ^ 2,      -- Cube with side length 7
   surface_areas.nthLe 4 sorry - sides.nthLe 4 sorry ^ 2]      -- Cube with side length 9

-- Total surface area of the tower
def total_surface_area : ℕ := adjusted_surface_areas.sum

-- The proof problem statement:
theorem total_surface_area_correct : total_surface_area = 838.5 := by
  sorry -- Skip the actual proof

end total_surface_area_correct_l809_809287


namespace min_value_expr_l809_809340

theorem min_value_expr (x y z : ℝ) (h1 : x * y ≠ 0) (h2 : x + y ≠ 0) :
  ∃ m : ℝ, m = 5 ∧ ∀ x y z, x * y ≠ 0 → x + y ≠ 0 → 
    (let e := ( (y + z) / x + 2 )^2 + ( z / y + 2 )^2 + ( z / (x + y) - 1 )^2 in
    e ≥ m) :=
sorry

end min_value_expr_l809_809340


namespace addition_in_base4_l809_809284

def base4_add (a b c : ℕ) : option ℕ :=
  let a4 := a.to_digits 4
  let b4 := b.to_digits 4
  let c4 := c.to_digits 4
  let sum4 := (a4.zip_with (+) b4).zip_with (+) c4
  sum4.if_some (fun v => some (v.from_digits 4))

theorem addition_in_base4 :
  base4_add (nat.of_digits 4 [3, 0, 2]) (nat.of_digits 4 [2, 1, 1]) (nat.of_digits 4 [1, 2, 3]) = some (nat.of_digits 4 [0, 4, 0, 1]) :=
by sorry

end addition_in_base4_l809_809284


namespace calculateRequiredMonthlyRent_l809_809208

noncomputable def requiredMonthlyRent (purchase_price : ℝ) (annual_return_rate : ℝ) (annual_taxes : ℝ) (repair_percentage : ℝ) : ℝ :=
  let annual_return := annual_return_rate * purchase_price
  let total_annual_need := annual_return + annual_taxes
  let monthly_requirement := total_annual_need / 12
  let monthly_rent := monthly_requirement / (1 - repair_percentage)
  monthly_rent

theorem calculateRequiredMonthlyRent : requiredMonthlyRent 20000 0.06 450 0.10 = 152.78 := by
  sorry

end calculateRequiredMonthlyRent_l809_809208


namespace interest_rate_eq_ten_l809_809873

theorem interest_rate_eq_ten (R : ℝ) (P : ℝ) (SI CI : ℝ) :
  P = 1400 ∧
  SI = 14 * R ∧
  CI = 1400 * ((1 + R / 200) ^ 2 - 1) ∧
  CI - SI = 3.50 → 
  R = 10 :=
by
  sorry

end interest_rate_eq_ten_l809_809873


namespace croissant_baking_time_l809_809430

theorem croissant_baking_time :
  (mixing_time : ℕ) (folding_time_per_fold : ℕ) (number_of_folds : ℕ)
  (resting_time_per_rest : ℕ) (number_of_rests : ℕ) (baking_time : ℕ)
  (total_minutes : ℕ) (total_hours : ℕ) :
  mixing_time = 10 →
  folding_time_per_fold = 5 →
  number_of_folds = 4 →
  resting_time_per_rest = 75 →
  number_of_rests = 4 →
  baking_time = 30 →
  total_minutes = mixing_time + number_of_folds * folding_time_per_fold
                      + number_of_rests * resting_time_per_rest + baking_time →
  total_hours = total_minutes / 60 →
  total_hours = 6 :=
by
  intros
  sorry

end croissant_baking_time_l809_809430


namespace greatest_prime_factor_is_5_l809_809147

-- Define the expression
def expr : Nat := (3^8 + 9^5)

-- State the theorem
theorem greatest_prime_factor_is_5 : ∃ p : Nat, Prime p ∧ p = 5 ∧ ∀ q : Nat, Prime q ∧ q ∣ expr → q ≤ 5 := by
  sorry

end greatest_prime_factor_is_5_l809_809147


namespace ratio_CDEF_to_ABC_l809_809390

-- Lean 4 statement to prove the problem
theorem ratio_CDEF_to_ABC (A B C E D F: Type*)
    [triangle A B C]
    (h1: E ∈ line_segment A B)
    (h2: D ∈ line_segment A C)
    (h3: F ∈ line_segment B C)
    (h4: AE = 1)
    (h5: EB = 2)
    (h6: parallel DE BC)
    (h7: parallel EF AC) :
    area_ratio C D E F A B C = 4 / 9 :=
    sorry

end ratio_CDEF_to_ABC_l809_809390


namespace households_subscribing_B_and_C_l809_809588

/-- Each household subscribes to 2 different newspapers.
Residents only subscribe to newspapers A, B, and C.
There are 30 subscriptions for newspaper A.
There are 34 subscriptions for newspaper B.
There are 40 subscriptions for newspaper C.
Thus, the number of households that subscribe to both
newspaper B and newspaper C is 22. -/
theorem households_subscribing_B_and_C (subs_A subs_B subs_C households : ℕ) 
    (hA : subs_A = 30) (hB : subs_B = 34) (hC : subs_C = 40) (h_total : households = (subs_A + subs_B + subs_C) / 2) :
  (households - subs_A) = 22 :=
by
  -- Substitute the values to demonstrate equality based on the given conditions.
  sorry

end households_subscribing_B_and_C_l809_809588


namespace minimum_elements_l809_809400

variable {F : Finset ℤ}
variable {n : ℕ}

-- Conditions
def finite_nonempty_set_of_integers (F : Finset ℤ) : Prop :=
  F.nonempty ∧ ∀ x ∈ F, ∃ y z ∈ F, x = y + z

def positive_integer (n : ℕ) : Prop := n > 0

def sum_non_zero (F : Finset ℤ) (n : ℕ) : Prop :=
  ∀ (k : ℕ) (hk : 1 ≤ k ∧ k ≤ n) (x : Fin n → ℤ), (∀ i, x i ∈ F) → (Finset.univ.sum x) ≠ 0

-- Theorem statement
theorem minimum_elements (hmF : finite_nonempty_set_of_integers F) (hn : positive_integer n) (hF_sum : sum_non_zero F n) :
  F.card ≥ 2 * n + 2 := sorry

end minimum_elements_l809_809400


namespace bottles_total_l809_809600

def bottles_problem (C B M : ℕ) (T : ℕ) : Prop :=
  C = 40 ∧ B = 80 ∧ (1/2 : ℚ) * (C + B + M) = 90 ∧ T = C + B + M → T = 180

-- To formally state the theorem
theorem bottles_total : ∃ (C B M T : ℕ), bottles_problem C B M T :=
begin
  use [40, 80, 60, 180],
  unfold bottles_problem,
  split; try {refl}, -- C = 40 and B = 80
  split,
  { exact (by norm_num : (1/2 : ℚ) * (40 + 80 + 60) = 90)},
  { exact (by norm_num : 40 + 80 + 60 = 180)},
end

end bottles_total_l809_809600


namespace jake_watched_friday_l809_809017

theorem jake_watched_friday
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (thursday_hours : ℕ)
  (total_hours : ℕ)
  (day_hours : ℕ := 24) :
  monday_hours = (day_hours / 2) →
  tuesday_hours = 4 →
  wednesday_hours = (day_hours / 4) →
  thursday_hours = ((monday_hours + tuesday_hours + wednesday_hours) / 2) →
  total_hours = 52 →
  (total_hours - (monday_hours + tuesday_hours + wednesday_hours + thursday_hours)) = 19 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jake_watched_friday_l809_809017


namespace line_passes_through_point_l809_809492

theorem line_passes_through_point (k : ℝ) :
  (1 + 4 * k) * 2 - (2 - 3 * k) * 2 + 2 - 14 * k = 0 :=
by
  sorry

end line_passes_through_point_l809_809492


namespace dice_visible_dots_total_l809_809438

/-- Define the conditions for the sum of visible dots on the configured dice -/
def dice_visible_dots_sum (faces : ℕ → ℕ) (configuration : list (ℕ × list ℕ)) : ℕ :=
  configuration.foldr (λ (die : ℕ × list ℕ) (acc : ℕ), acc + die.1 + die.2.foldr (· + ·) 0) 0

theorem dice_visible_dots_total :
  ∃ (configuration : list (ℕ × list ℕ)),
  configuration = [(1, [6, 3, 2, 5]), 
                   (4, [5, 1, 2, 6]),
                   (6, [4, 1, 2, 3]),
                   (2, [1,3])] →
  dice_visible_dots_sum id configuration = 57 :=
begin
  sorry
end

end dice_visible_dots_total_l809_809438


namespace lydia_total_plants_l809_809054

theorem lydia_total_plants (T F : ℕ) (h1 : 0.4 * T = F) (h2 : (F / 4) * 5 = 40) : T = 80 :=
by sorry

end lydia_total_plants_l809_809054


namespace find_x_l809_809360

theorem find_x (h₁ : 2994 / 14.5 = 175) (h₂ : 29.94 / x = 17.5) : x = 29.94 / 17.5 :=
by
  -- skipping proofs
  sorry

end find_x_l809_809360


namespace acute_triangle_area_bound_l809_809764

theorem acute_triangle_area_bound (ABC : Triangle)
  (h_acute : ABC.isAcute)
  (AM : Segment)
  (BK : Segment)
  (CH : Segment)
  (M' K' H' : Point)
  (h_AM : AM.isMedian ABC A B C)
  (h_BK : BK.isAngleBisector ABC B K)
  (h_CH : CH.isAltitude ABC C H)
  (h_intersections : IntersectingSegments AM BK CH M' K' H') :
  Area (Triangle.mk M' K' H') > 0.499 * Area ABC :=
by
  sorry

end acute_triangle_area_bound_l809_809764


namespace area_ratio_of_quadrilateral_ADGJ_to_decagon_l809_809080

noncomputable def ratio_of_areas (k : ℝ) : ℝ :=
  (2 * k^2 * Real.sin (72 * Real.pi / 180)) / (5 * Real.sqrt (5 + 2 * Real.sqrt 5))

theorem area_ratio_of_quadrilateral_ADGJ_to_decagon
  (k : ℝ) :
  ∃ (n m : ℝ), m / n = ratio_of_areas k :=
  sorry

end area_ratio_of_quadrilateral_ADGJ_to_decagon_l809_809080


namespace simplify_radical_expression_l809_809855

theorem simplify_radical_expression (x : ℝ) :
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x)) = 120 * x * sqrt (2 * x) := by
sorry

end simplify_radical_expression_l809_809855


namespace parabola_focus_vector_dot_product_l809_809339

theorem parabola_focus_vector_dot_product :
  let C := {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1},
      F := (1, 0), -- Focus of the parabola
      M := (3, 2 * Real.sqrt 3), -- Since x = 3
      Q := (-1, 2 * Real.sqrt 3), -- Directrix related point
      FQ := (Q.1 - F.1, Q.2 - F.2),
      FM := (M.1 - F.1, M.2 - F.2)
  in FQ.1 * FM.1 + FQ.2 * FM.2 = 8 := 
by 
  -- Assume that the triangle conditions and coordinates are valid, omitted steps
  sorry

end parabola_focus_vector_dot_product_l809_809339


namespace chocolate_bar_cost_l809_809025

-- Define the quantities Jessica bought
def chocolate_bars := 10
def gummy_bears_packs := 10
def chocolate_chips_bags := 20

-- Define the costs
def total_cost := 150
def gummy_bears_pack_cost := 2
def chocolate_chips_bag_cost := 5

-- Define what we want to prove (the cost of one chocolate bar)
theorem chocolate_bar_cost : 
  ∃ chocolate_bar_cost, 
    chocolate_bars * chocolate_bar_cost + 
    gummy_bears_packs * gummy_bears_pack_cost + 
    chocolate_chips_bags * chocolate_chips_bag_cost = total_cost ∧
    chocolate_bar_cost = 3 :=
by
  -- Proof goes here
  sorry

end chocolate_bar_cost_l809_809025


namespace sum_distances_l809_809908

variable (n : ℕ)
variable (x y : Fin n → ℝ)

def A_n (n : ℕ) (x y : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.range n, ∑ j in Finset.range i, (|x i - x j| + |y i - y j|)

def B_n (n : ℕ) (x y : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.range n, ∑ j in Finset.range n, |x i - y j|

theorem sum_distances (n : ℕ) (x y : Fin n → ℝ) : B_n n x y ≥ A_n n x y := 
  sorry

end sum_distances_l809_809908


namespace combined_molecular_weight_l809_809929

theorem combined_molecular_weight :
  let CaO_molecular_weight := 56.08
  let CO2_molecular_weight := 44.01
  let HNO3_molecular_weight := 63.01
  let moles_CaO := 5
  let moles_CO2 := 3
  let moles_HNO3 := 2
  moles_CaO * CaO_molecular_weight + moles_CO2 * CO2_molecular_weight + moles_HNO3 * HNO3_molecular_weight = 538.45 :=
by sorry

end combined_molecular_weight_l809_809929


namespace alternating_sum_sequence_l809_809235

theorem alternating_sum_sequence : 
  (2 - 5 + 8 - 11 + 14 - 17 + 20 - 23 + 26 - 29 + 
   32 - 35 + 38 - 41 + 44 - 47 + 50 - 53 + 56 - 
   59 + 62 - 65 + 68) = 35 :=
begin
  sorry
end

end alternating_sum_sequence_l809_809235


namespace first_year_students_sampled_equals_40_l809_809915

-- Defining the conditions
def num_first_year_students := 800
def num_second_year_students := 600
def num_third_year_students := 500
def num_sampled_third_year_students := 25
def total_students := num_first_year_students + num_second_year_students + num_third_year_students

-- Proving the number of first-year students sampled
theorem first_year_students_sampled_equals_40 :
  (num_first_year_students * num_sampled_third_year_students) / num_third_year_students = 40 := by
  sorry

end first_year_students_sampled_equals_40_l809_809915


namespace sum_of_positive_integers_m_for_positive_coordinates_P_l809_809493

theorem sum_of_positive_integers_m_for_positive_coordinates_P :
  (∑ m in {m : ℕ | ∃ x : ℕ, x = y ∧ x = ((-4 : ℤ) / (1 - (m : ℤ))) ∧ y = ((-4 : ℤ) / (1 - (m : ℤ))) ∧ (y > 0)}.toFinset) = 10 :=
by
  sorry

end sum_of_positive_integers_m_for_positive_coordinates_P_l809_809493


namespace proof_equations_l809_809794

variable {S : Type*} [Inhabited S] [Nonempty S]

variable (op : S → S → S)

def condition (a b : S) : Prop := a * (op b a) = b

theorem proof_equations (a b : S) (h : ∀ a b, condition op a b) :
  ( ∀ a b, (a * b) * (op a b) = b) ∧
  ( ∀ a b, (op b a) * (op a b) = a) ∧
  ( ∀ a b, (a * b) * (op (op b a) b) = b) :=
by
  sorry

end proof_equations_l809_809794


namespace part1_part2_l809_809331

-- Let f(x) = a^x - a^(-x) for some a > 0 and a ≠ 1
def f (a : ℝ) (x : ℝ) : ℝ := a^x - a^(-x)

-- Given that f(1) < 0, then we need to prove 0 < a < 1
theorem part1 (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a 1 < 0) : 0 < a ∧ a < 1 :=
sorry

-- g(x) = a^(2x) + a^(-2x) - 2m * f(a, x)
def g (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := a^(2 * x) + a^(-2 * x) - 2 * m * f a x

-- Given f(1) = 3 / 2, and the minimum value of g(x) on [1,+∞) is -2, 
-- we need to prove that m = 2
theorem part2 (a : ℝ) (m : ℤ) (h1 : f a 1 = 3 / 2) 
(h2 : ∀ x ∈ set.Ici (1 : ℝ), g a m x ≥ -2) 
(h3 : ∃ x ∈ set.Ici (1 : ℝ), g a m x = -2) : m = 2 :=
sorry

end part1_part2_l809_809331


namespace dusty_change_l809_809222

theorem dusty_change :
  let single_layer_price := 4
  let double_layer_price := 7
  let num_single_layers := 7
  let num_double_layers := 5
  let total_payment := 100
  let total_cost := (single_layer_price * num_single_layers) + (double_layer_price * num_double_layers)
  let change := total_payment - total_cost
  change = 37 :=
by
  -- Definitions only, proof not required.
  let single_layer_price := 4
  let double_layer_price := 7
  let num_single_layers := 7
  let num_double_layers := 5
  let total_payment := 100
  let total_cost := (single_layer_price * num_single_layers) + (double_layer_price * num_double_layers)
  let change := total_payment - total_cost
  have h₁ : total_cost = 63, by sorry
  have h₂ : change = 37, by sorry
  exact h₂

end dusty_change_l809_809222


namespace remainder_sum_binomial_mod_l809_809618

noncomputable def binomial_mod_sum (n : ℕ) (m : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), if i % 3 = 0 then Nat.choose n i else 0

theorem remainder_sum_binomial_mod {n : ℕ} (h : n = 2007) : 
  (binomial_mod_sum 2007 3) % 1000 = 42 := 
by
  rw [h]
  sorry

end remainder_sum_binomial_mod_l809_809618


namespace exists_smallest_k_l809_809129

noncomputable def b : ℕ → ℝ
| 0       := 1
| 1       := real.root 7 3
| (n + 2) := b (n + 1) * (b n)^2

theorem exists_smallest_k : ∃ k : ℕ, 0 < k ∧ ∀ n < k, ∃ m : ℤ, b (fin.succ n) ^ (n + 1) = (m : ℝ) :=
by
  sorry

end exists_smallest_k_l809_809129


namespace perimeter_square_l809_809594

-- Definition of the side length
def side_length : ℝ := 9

-- Definition of the perimeter calculation
def perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem stating that the perimeter of a square with side length 9 cm is 36 cm
theorem perimeter_square : perimeter side_length = 36 := 
by sorry

end perimeter_square_l809_809594


namespace largest_base5_three_digit_to_base10_l809_809961

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l809_809961


namespace sum_of_solutions_eqn_l809_809671

theorem sum_of_solutions_eqn :
  ∑ y in {y : ℚ | -4 * y / (y ^ 2 - 1) = 3 * y / (y + 1) - 8 / (y - 1)}, y = 7 / 3 :=
sorry

end sum_of_solutions_eqn_l809_809671


namespace find_line_equation_l809_809640

noncomputable def perpendicular_origin_foot := 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ y = 2 * x + 5) ∧
    l (-2) 1

theorem find_line_equation : 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ 2 * x - y + 5 = 0) ∧
    l (-2) 1 ∧
    ∀ p q : ℝ, p = 0 → q = 0 → ¬ (l p q)
:= sorry

end find_line_equation_l809_809640


namespace sequence_eventually_periodic_l809_809288

-- Definitions based on the conditions
def positive_int_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < a n

def satisfies_condition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)

-- Assertion to prove based on the question
theorem sequence_eventually_periodic (a : ℕ → ℕ) 
  (h1 : positive_int_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ p : ℕ, ∃ k : ℕ, ∀ n : ℕ, a (n + k) = a n :=
sorry

end sequence_eventually_periodic_l809_809288


namespace find_a_for_domain_odd_function_solution_set_l809_809717

def question1_dom (x a : ℝ) : Prop := x < 1/3 ∨ x > 1
def question1_func (x a : ℝ) : ℝ := log (2 / (x - 1) + a)

theorem find_a_for_domain :
  (∀ x, question1_dom x 3) →
  (∀ x, ∃ a, a = 3) :=
sorry

def question2_odd (x a : ℝ) : Prop := (log (2 / (x - 1) + a) = -log (2 / (-(x - 1)) + a))
def question2_ineq (x a : ℝ) : Prop := log (2 / (x - 1) + a) > 0

theorem odd_function_solution_set :
  (∀ x, question2_odd x 1) →
  (∀ x, question2_ineq x 1 → x > 1) :=
sorry

end find_a_for_domain_odd_function_solution_set_l809_809717


namespace square_count_eq_nine_l809_809404

noncomputable def num_squares_sharing_vertices_with_triangle (A B C : ℝ × ℝ) (h_eq : ∀ x y z : ℝ × ℝ, (x - y) = (z - x) ∨ (y - z) = (x - y) ∧ (dist x y = dist y z ∧ dist y z = dist z x))
: Nat := 9

theorem square_count_eq_nine (A B C : ℝ × ℝ) (h_eq : (dist A B = dist B C) ∧ (dist B C = dist C A)) : 
  num_squares_sharing_vertices_with_triangle A B C h_eq = 9 :=
sorry

end square_count_eq_nine_l809_809404


namespace even_number_of_z_quad_dominoes_l809_809996

theorem even_number_of_z_quad_dominoes (P : set (ℤ × ℤ)) 
(H1 : ∀ (S : set (ℤ × ℤ)), covers(P, S_quad_domino)) 
(H2 : ∀ (S : set (ℤ × ℤ)), even (count_black_squares(S_quad_domino))) 
(H3 : ∀ (Z : set (ℤ × ℤ)), odd (count_black_squares(Z_quad_domino))) :
∃ Z, covers (P, Z_quad_domino) ∧ even (cardinality Z) := 
sorry

end even_number_of_z_quad_dominoes_l809_809996


namespace k1_k2_is_constant_l809_809383

-- Define the ellipse C with given equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 6 + y^2 / 3 = 1)

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- P lies on the ellipse
def P_on_ellipse : Prop :=
  ellipse P.1 P.2

-- Define the points A and B on the ellipse
variables {A B : ℝ × ℝ}

def A_on_ellipse : Prop :=
  ellipse A.1 A.2

def B_on_ellipse : Prop :=
  ellipse B.1 B.2

-- Define the midpoint D of segment AB
def midpoint_D : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The slope of the line OD is 1
def slope_OD : Prop :=
  (midpoint_D.2 / midpoint_D.1) = 1

-- The slopes of lines PA and PB
def slope_PA : ℝ :=
  (A.2 - P.2) / (A.1 - P.1)

def slope_PB : ℝ :=
  (B.2 - P.2) / (B.1 - P.1)

-- The slopes multiplied together
def k1_k2 : ℝ :=
  slope_PA * slope_PB

-- Given the above conditions, prove that k1_k2 is 1/2
theorem k1_k2_is_constant (hP : P_on_ellipse) (hA : A_on_ellipse) (hB : B_on_ellipse)
  (hSlope : slope_OD) : k1_k2 = 1 / 2 := by
  sorry

end k1_k2_is_constant_l809_809383


namespace surface_area_of_box_l809_809450

theorem surface_area_of_box
  (l w c : ℕ)
  (hl : l = 25)
  (hw : w = 35)
  (hc : c = 6) :
  (l * w) - 4 * (c * c) = 731 :=
by
  rw [hl, hw, hc]
  sorry

end surface_area_of_box_l809_809450


namespace find_deeper_depth_l809_809597

noncomputable def swimming_pool_depth_proof 
  (width : ℝ) (length : ℝ) (shallow_depth : ℝ) (volume : ℝ) : Prop :=
  volume = (1 / 2) * (shallow_depth + 4) * width * length

theorem find_deeper_depth
  (h : width = 9)
  (l : length = 12)
  (a : shallow_depth = 1)
  (V : volume = 270) :
  swimming_pool_depth_proof 9 12 1 270 := by
  sorry

end find_deeper_depth_l809_809597


namespace evaluate_expression_l809_809974

theorem evaluate_expression (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 + 7 * x = 696 :=
by
  have hx : x = 3 := h
  sorry

end evaluate_expression_l809_809974


namespace find_area_triangle_l809_809745

variables (A_square A_overlap A_triangle : ℝ)

def side_length_square : ℝ := 8
def area_square : ℝ := side_length_square * side_length_square
def area_overlap : ℝ := (3 / 4) * area_square
def area_triangle : ℝ := 2 * area_overlap

theorem find_area_triangle :
  A_square = side_length_square * side_length_square →
  A_overlap = (3 / 4) * A_square →
  A_overlap = (1 / 2) * A_triangle →
  A_triangle = 96 :=
by
  intros h1 h2 h3
  rw [h1] at h2 ⊢
  have h : A_overlap = 48 := by linarith
  rw [h] at h3
  linarith

end find_area_triangle_l809_809745


namespace smallest_integer_ending_in_9_divisible_by_11_is_99_l809_809155

noncomputable def smallest_positive_integer_ending_in_9_and_divisible_by_11 : ℕ :=
  99

theorem smallest_integer_ending_in_9_divisible_by_11_is_99 :
  ∃ n : ℕ, n > 0 ∧ (n % 10 = 9) ∧ (n % 11 = 0) ∧
          (∀ m : ℕ, m > 0 → (m % 10 = 9) → (m % 11 = 0) → n ≤ m) :=
begin
  use smallest_positive_integer_ending_in_9_and_divisible_by_11,
  split,
  { -- n > 0
    exact nat.zero_lt_bit1 nat.zero_lt_one },
  split,
  { -- n % 10 = 9
    exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { -- n % 11 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 99) },
  { -- ∀ m > 0, m % 10 = 9, m % 11 = 0 → n ≤ m
    intros m hm1 hm2 hm3,
    change 99 ≤ m,
    -- m % 99 = 0 → 99 ≤ m since 99 > 0
    sorry
  }
end

end smallest_integer_ending_in_9_divisible_by_11_is_99_l809_809155


namespace largest_base_5_three_digit_in_base_10_l809_809948

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l809_809948


namespace solution_of_inequality_is_correct_l809_809503

-- Inequality condition (x-1)/(2x+1) ≤ 0
def inequality (x : ℝ) : Prop := (x - 1) / (2 * x + 1) ≤ 0 

-- Conditions
def condition1 (x : ℝ) : Prop := (x - 1) * (2 * x + 1) ≤ 0
def condition2 (x : ℝ) : Prop := 2 * x + 1 ≠ 0

-- Combined condition
def combined_condition (x : ℝ) : Prop := condition1 x ∧ condition2 x

-- Solution set
def solution_set : Set ℝ := { x | -1/2 < x ∧ x ≤ 1 }

-- Theorem statement
theorem solution_of_inequality_is_correct :
  ∀ x : ℝ, inequality x ↔ combined_condition x ∧ x ∈ solution_set :=
by
  sorry

end solution_of_inequality_is_correct_l809_809503


namespace prob_two_red_cards_l809_809763

noncomputable def total_cards := 60
noncomputable def suits := 5
noncomputable def cards_per_suit := 12
noncomputable def red_suits := 3
noncomputable def black_suits := 2

def number_of_red_cards := red_suits * cards_per_suit
def number_of_black_cards := black_suits * cards_per_suit

theorem prob_two_red_cards :
  let total_cards := total_cards,
      number_of_red_cards := number_of_red_cards in
  (number_of_red_cards / total_cards) * ((number_of_red_cards - 1) / (total_cards - 1)) = 21 / 59 :=
by
  sorry

end prob_two_red_cards_l809_809763


namespace price_decrease_percent_approx_l809_809170

theorem price_decrease_percent_approx :
  let original_price := 79.95
  let sale_price := 59.95
  let amount_decrease := original_price - sale_price
  let percent_decrease := (amount_decrease / original_price) * 100
  abs (percent_decrease - 25) < 0.05 :=
by
  sorry

end price_decrease_percent_approx_l809_809170


namespace cubic_function_sum_l809_809678

def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12)

theorem cubic_function_sum :
  ∑ k in finset.range 2016, f (↑k / 2017) = 2016 :=
by
  sorry

end cubic_function_sum_l809_809678


namespace find_c_value_l809_809336

theorem find_c_value 
  (b : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + b * x + 3 ≥ 0) 
  (h2 : ∀ m c : ℝ, (∀ x : ℝ, x^2 + b * x + 3 < c ↔ m - 8 < x ∧ x < m)) 
  : c = 16 :=
sorry

end find_c_value_l809_809336


namespace compound_interest_l809_809363

theorem compound_interest
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) :
  P = 1000 → r = 0.10 → n = 2 → t = 1 →
  P * (1 + r / n)^(n * t) = 1102.50 :=
by
  intros hP hr hn ht
  rw [hP, hr, ht, hn]
  norm_num
  sorry

end compound_interest_l809_809363


namespace min_distance_C2_to_line_l_l809_809768

noncomputable def line_l_param : ℝ → ℝ × ℝ :=
  λ t, (-2 + t / 2, (sqrt 3) / 2 * t)

noncomputable def curve_C1_polar : ℝ → ℝ × ℝ :=
  λ θ, (sqrt 6 * cos θ, sqrt 6 * sin θ)

noncomputable def curve_C2_param : ℝ → ℝ × ℝ :=
  λ θ, (cos θ, sqrt 3 * sin θ)

noncomputable def distance_from_point_to_line (p : ℝ × ℝ) : ℝ :=
  abs (sqrt 3 * p.1 - p.2 + 2 * sqrt 3) / sqrt (3 + 1)

theorem min_distance_C2_to_line_l :
  ∃ θ, distance_from_point_to_line (curve_C2_param θ) = (2 * sqrt 3 - sqrt 6) / 2 :=
sorry

end min_distance_C2_to_line_l_l809_809768


namespace positive_difference_of_solutions_eq_l809_809857

theorem positive_difference_of_solutions_eq (m : ℝ) (h : 27 - m > 0) :
  let x1 := 3 + real.sqrt (27 - m)
  let x2 := 3 - real.sqrt (27 - m)
  abs (x1 - x2) = 2 * real.sqrt (27 - m) :=
by
  sorry

end positive_difference_of_solutions_eq_l809_809857


namespace part_a_l809_809662

theorem part_a (a b : ℤ) (x : ℤ) :
  (x % 5 = a) ∧ (x % 6 = b) → x = 6 * a + 25 * b :=
by
  sorry

end part_a_l809_809662


namespace vertex_parabola_l809_809476

theorem vertex_parabola (h k : ℝ) : 
  (∀ x : ℝ, -((x - 2)^2) + 3 = k) → (h = 2 ∧ k = 3) :=
by 
  sorry

end vertex_parabola_l809_809476


namespace product_sum_condition_l809_809443

theorem product_sum_condition (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c > (1/a) + (1/b) + (1/c)) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end product_sum_condition_l809_809443


namespace Jake_watched_hours_on_Friday_l809_809019

theorem Jake_watched_hours_on_Friday :
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  total_show_hours - total_hours_before_Friday = 19 :=
by
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  sorry

end Jake_watched_hours_on_Friday_l809_809019


namespace determine_c_values_l809_809632

noncomputable def isSolution (c x y : ℝ) : Prop :=
  (x * y = c^c) ∧ (Real.log c (x^(Real.log c y * c)) + Real.log c (y^(Real.log c x * c)) = 5 * c^5)

theorem determine_c_values :
  ∀ c : ℝ, (∃ x y : ℝ, isSolution c x y) ↔ c > 0 ∧ c ≤ Real.cbrt (1 / 5) := by
  sorry

end determine_c_values_l809_809632


namespace proof_f_of_2_add_g_of_3_l809_809043

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x^2 + 2 * x - 1

theorem proof_f_of_2_add_g_of_3 : f (2 + g 3) = 44 :=
by
  sorry

end proof_f_of_2_add_g_of_3_l809_809043


namespace pyramid_inscribed_sphere_radius_l809_809919

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := 
a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3))

theorem pyramid_inscribed_sphere_radius (a : ℝ) (h1 : a > 0) : 
  inscribed_sphere_radius a = a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3)) :=
by
  sorry

end pyramid_inscribed_sphere_radius_l809_809919


namespace range_of_b_l809_809312

-- Define the points M and N
def M : (ℝ × ℝ) := (1, 0)
def N : (ℝ × ℝ) := (-1, 0)

-- Define the line segment MN
def on_segment (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ (p.1 ≥ -1 ∧ p.1 ≤ 1)

-- Define the intersecting line
def line (b : ℝ) (p : ℝ × ℝ) : Prop :=
  2 * p.1 + p.2 = b

-- Prove the range of b for intersection
theorem range_of_b : ∀ (b : ℝ), 
  (∃ (p : ℝ × ℝ), on_segment p ∧ line b p) ↔ (b ≥ -2 ∧ b ≤ 2) :=
by
  intro b
  split
  {
    rintro ⟨p, ⟨h1, ⟨h2, h3⟩⟩, h4⟩
    have h5 := h4
    rw [mul_assoc] at h5
    rw [add_assoc] at h5
    rw [←h1] at h5
    exact ⟨by linarith, by linarith⟩
  }
  {
    rintro ⟨h1, h2⟩
    use (b / 2, 0)
    split
    {
      exact ⟨rfl, by linarith⟩
    }
    {
      nlinarith
    }
  }
  sorry

end range_of_b_l809_809312


namespace largest_base5_eq_124_l809_809940

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l809_809940


namespace max_real_roots_of_P_l809_809790

open Polynomial

noncomputable def P (x : ℝ) : ℝ := sorry -- P(x) is a nonzero polynomial

theorem max_real_roots_of_P :
  (∀ x : ℝ, P(x ^ 2 - 1) = P(x) * P(-x)) → 
  ∃ (n : ℕ), (∀ x : ℝ, Polynomial.eval x P = 0 → n ≤ 4) :=
sorry

end max_real_roots_of_P_l809_809790


namespace sequence_periodicity_l809_809631

theorem sequence_periodicity :
  let t : ℕ → ℚ :=
    λ n, if n = 1 then 20 else if n = 2 then 21 else (5 * t (n - 1) + 1) / (25 * t (n - 2))
  in (t 2020 = 101 / 525) ∧ Nat.gcd 101 525 = 1 ∧ (101 + 525 = 626) :=
by
  sorry

end sequence_periodicity_l809_809631


namespace largest_base5_eq_124_l809_809939

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l809_809939


namespace hyperbola_range_k_l809_809365

theorem hyperbola_range_k (k : ℝ) : (4 + k) * (1 - k) < 0 ↔ k ∈ (Set.Iio (-4) ∪ Set.Ioi 1) := 
by
  sorry

end hyperbola_range_k_l809_809365


namespace zero_order_l809_809802

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (x : ℝ) : ℝ := x * Real.log x - 1
noncomputable def h (x : ℝ) : ℝ := 1 - 1 / x + x / 2 + x^2 / 3

theorem zero_order (a b c : ℝ)
  (hf : ∃ a, 0 < a ∧ f(a) = 0)
  (hg : ∃ b, 0 < b ∧ g(b) = 0)
  (hh : ∃ c, 0 < c ∧ h(c) = 0)
  (ha : a ∈ (1/2 : ℝ, 1))
  (hb : b ∈ (1, 2))
  (hc : c ∈ (1/2, 1))
  (hcp : c > 5/8)
  (hap : a < 5/8) : b > c ∧ c > a :=
sorry

end zero_order_l809_809802


namespace cost_calculation_l809_809477

variables (H M F : ℝ)

theorem cost_calculation 
  (h1 : 3 * H + 5 * M + F = 23.50) 
  (h2 : 5 * H + 9 * M + F = 39.50) : 
  2 * H + 2 * M + 2 * F = 15.00 :=
sorry

end cost_calculation_l809_809477


namespace sum_of_lengths_le_half_l809_809549

open Set

theorem sum_of_lengths_le_half (M : Set ℝ) :
    (∀ x y ∈ M, |x - y| ≠ 1/10) → 
    ∀ s, (∀ t ∈ s, t ⊆ (Icc 0 1)) → s ⊆ M → 
    ∑ t in s, (↑(Sup t) - Inf t) ≤ 1 / 2 := by
  sorry

end sum_of_lengths_le_half_l809_809549


namespace move_sine_to_cosine_l809_809914

theorem move_sine_to_cosine (x : ℝ) : 
  shift_left (λ x, 3 * sin (2 * x + π / 6)) (π / 6) = (λ x, 3 * cos (2 * x)) :=
sorry

end move_sine_to_cosine_l809_809914


namespace count_negative_expressions_l809_809115

theorem count_negative_expressions : 
  let e1 := -(-2)
  let e2 := -| -2 |
  let e3 := -2 * 5
  let e4 := -(-2) - 3
(e1 < 0).toNat + (e2 < 0).toNat + (e3 < 0).toNat + (e4 < 0).toNat = 3 :=
by
  let e1 := -(-2)
  let e2 := -| -2 |
  let e3 := -2 * 5
  let e4 := -(-2) - 3
  sorry

end count_negative_expressions_l809_809115


namespace count_special_ordered_quadruples_l809_809255

theorem count_special_ordered_quadruples : 
  let special_quadruple (a b c d : ℕ) := 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 12 ∧ a + b < c + d
  nat.card {p : ℕ × ℕ × ℕ × ℕ // special_quadruple p.1 p.2.1 p.2.2.1 p.2.2.2} = 247 :=
by
  sorry

end count_special_ordered_quadruples_l809_809255


namespace monotonically_increasing_on_interval_l809_809330

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem monotonically_increasing_on_interval :
  ∀ x y : ℝ, (x ∈ Set.Icc (3 * Real.pi / 4) Real.pi) → 
             (y ∈ Set.Icc (3 * Real.pi / 4) Real.pi) → 
             (x ≤ y) → 
             (f x ≤ f y) :=
begin
  sorry
end

end monotonically_increasing_on_interval_l809_809330


namespace initial_deposit_l809_809861

theorem initial_deposit (P R : ℝ) (h1 : 8400 = P + (P * R * 2) / 100) (h2 : 8760 = P + (P * (R + 4) * 2) / 100) : 
  P = 2250 :=
  sorry

end initial_deposit_l809_809861


namespace triangle_shape_l809_809318

-- Define the sides of the triangle and the angles
variables {a b c : ℝ}
variables {A B C : ℝ} 
-- Assume that angles are in radians and 0 < A, B, C < π
-- Also assume that the sum of angles in the triangle is π
axiom angle_sum_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : a^2 * Real.cos A * Real.sin B = b^2 * Real.sin A * Real.cos B

-- Conclusion: The shape of triangle ABC is either isosceles or right triangle
theorem triangle_shape : 
  (A = B) ∨ (A + B = (Real.pi / 2)) := 
by sorry

end triangle_shape_l809_809318


namespace set_d_forms_triangle_l809_809542

theorem set_d_forms_triangle (a b c : ℕ) : a = 5 ∧ b = 6 ∧ c = 10 → (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  intros h
  cases h with h₁ h₂
  cases h₂ with h₃ h₄
  rw [h₁, h₃, h₄]
  exact ⟨by decide, by decide, by decide⟩

end set_d_forms_triangle_l809_809542


namespace exists_unique_x_l809_809674

-- Define the largest power of 2 that divides n
def v2 (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.find_greatest (λ k, 2^k ∣ n) n

-- Define the predicate and the function
variable (f : ℕ → ℕ)

-- First condition: f(x) ≤ 3x for all natural numbers x
axiom f_le_3x : ∀ x : ℕ, f(x) ≤ 3 * x

-- Second condition: v2(f(x) + f(y)) = v2(x + y) for all natural numbers x and y
axiom v2_condition : ∀ x y : ℕ, v2(f(x) + f(y)) = v2(x + y)

-- Proof statement: for every natural number a, there exists exactly one natural number x such that f(x) = 3a
theorem exists_unique_x (a : ℕ) : ∃! x : ℕ, f(x) = 3 * a :=
  by 
    sorry

end exists_unique_x_l809_809674


namespace max_distance_from_curve_to_line_l809_809770

noncomputable def curve_param_x (α : ℝ) := (√6) * Real.cos α
noncomputable def curve_param_y (α : ℝ) := (√2) * Real.sin α

-- Distance formula from a point (x₁, y₁) to the line Ax + By + C = 0
noncomputable def distance_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
  (abs (A * x₁ + B * y₁ + C)) / (√ (A^2 + B^2))

-- Prove that max distance from points on the parametric curve to the line is 2 + √3
theorem max_distance_from_curve_to_line :
  ∃ α : ℝ, distance_to_line (curve_param_x α) (curve_param_y α) 1 (√3) 4 = 2 + √3 :=
begin
  sorry
end

end max_distance_from_curve_to_line_l809_809770


namespace midpoints_form_square_l809_809304

open EuclideanGeometry

def quadrilateral : Type :=
  {A B C D : Point ℝ // A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A
                  ∧ ∠ A B C = 45 ∧ ∠ B C D = 45 ∧ ∠ C D A = 45 
                  ∧ non-convex (A B C D)
                  ∧ non-self-intersecting (A B C D)}

theorem midpoints_form_square (q : quadrilateral) :
  let (A, B, C, D) := (q.1, q.2, q.3, q.4) in
  is_square (midpoint A B, midpoint B C, midpoint C D, midpoint D A) := by
  sorry

end midpoints_form_square_l809_809304


namespace smallest_integer_ending_in_9_and_divisible_by_11_l809_809152

theorem smallest_integer_ending_in_9_and_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
  sorry

end smallest_integer_ending_in_9_and_divisible_by_11_l809_809152


namespace evaluate_statements_l809_809544

-- Definitions (Conditions)
def equal_arcs_correspond_to_equal_chords := ∀ (C : Type) [metric_space C] [normed_group C], 
  ∀ (arc₁ arc₂ : set C), is_arc C arc₁ ∧ is_arc C arc₂ ∧ arc₁ = arc₂ → corresponding_chords_equal arc₁ arc₂

def angles_formed_by_equal_chords := ∀ (C : Circle), ∀ (chord₁ chord₂ : Chord C), 
  chord₁ = chord₂ → equal_angles_formed_by_chords chord₁ chord₂

def diameter_perpendicular_to_chord_bisects_chord := ∀ (C : Circle) (d : Diameter C) 
 (chord : Chord C), is_perpendicular d chord → bisects d chord

def line_passing_through_point_of_tangency := ∀ (C : Type) [metric_space C] [normed_group C], 
  ∀ (tangent_line : set C) (P : C), is_point_of_tangency P tangent_line → is_tangent tangent_line

-- Goal (Proof Problem)
theorem evaluate_statements : 
  equal_arcs_correspond_to_equal_chords ∧ ¬angles_formed_by_equal_chords ∧ 
  ¬diameter_perpendicular_to_chord_bisects_chord ∧ ¬line_passing_through_point_of_tangency :=
by 
  sorry

end evaluate_statements_l809_809544


namespace mark_hours_per_week_l809_809427

theorem mark_hours_per_week (w_historical : ℕ) (w_spring : ℕ) (h_spring : ℕ) (e_spring : ℕ) (e_goal : ℕ) (w_goal : ℕ) (h_goal : ℚ) :
  (e_spring : ℚ) / (w_historical * w_spring) = h_spring / w_spring →
  e_goal = 21000 →
  w_goal = 50 →
  h_spring = 35 →
  w_spring = 15 →
  e_spring = 4200 →
  (h_goal : ℚ) = 2625 / w_goal →
  h_goal = 52.5 :=
sorry

end mark_hours_per_week_l809_809427


namespace simplify_imaginary_expr_l809_809999

def imaginary_unit_expr : ℂ :=
  let i : ℂ := Complex.i in
  i + 2 * i^2 + 3 * i^3 + 4 * i^4

theorem simplify_imaginary_expr :
  imaginary_unit_expr = 2 - 2 * Complex.i :=
by
  sorry

end simplify_imaginary_expr_l809_809999


namespace George_oranges_3150_l809_809264

variable (Betty_oranges : ℕ)
variable (Sandra_oranges : ℕ)
variable (Emily_oranges : ℕ)
variable (Frank_oranges : ℕ)
variable (George_oranges : ℕ)

-- Definitions based on problem conditions
def Sandra_oranges_def := Sandra_oranges = 3 * Betty_oranges
def Emily_oranges_def := Emily_oranges = 7 * Sandra_oranges
def Frank_oranges_def := Frank_oranges = 5 * Emily_oranges
def George_oranges_def := George_oranges = 2.5 * Frank_oranges

-- Betty's oranges
def Betty_oranges_value := Betty_oranges = 12

-- Main statement to prove
theorem George_oranges_3150 (H1 : Betty_oranges_value)
                            (H2 : Sandra_oranges_def)
                            (H3 : Emily_oranges_def)
                            (H4 : Frank_oranges_def)
                            (H5 : George_oranges_def) : 
  George_oranges = 3150 := 
  by sorry

end George_oranges_3150_l809_809264


namespace solve_exponential_l809_809361

noncomputable def eq_exponential (z : ℝ) : Prop :=
  8 ^ (3 * z) = 64

noncomputable def negative_exponential (z : ℝ) : ℝ :=
  8 ^ (-2 * z)

theorem solve_exponential (z : ℝ) (h : 8 ^ (3 * z) = 64) :
  negative_exponential z = 1 / 16 :=
by
  sorry

end solve_exponential_l809_809361


namespace rate_ratio_l809_809205

theorem rate_ratio
  (rate_up : ℝ) (time_up : ℝ) (distance_up : ℝ)
  (distance_down : ℝ) (time_down : ℝ) :
  rate_up = 4 → time_up = 2 → distance_up = rate_up * time_up →
  distance_down = 12 → time_down = 2 →
  (distance_down / time_down) / rate_up = 3 / 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rate_ratio_l809_809205


namespace num_real_solutions_frac_sine_l809_809657

theorem num_real_solutions_frac_sine :
  (∃ n : ℕ, ∀ x : ℝ, x ∈ Icc (-150) 150 → (x/150 = Real.sin x) ↔ (n = 95)) := 
sorry

end num_real_solutions_frac_sine_l809_809657


namespace probability_distance_at_least_a_l809_809032

open Real

theorem probability_distance_at_least_a (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_a_lt_b : a < b) :
  let p := (b - a)^2 / b^2 in
  p = (b - a)^2 / b^2 := by
sorry

end probability_distance_at_least_a_l809_809032


namespace DyckPathsWithoutEvenReturn_corresponds_l809_809566

def isDyckPath (steps : List (ℤ × ℤ)) : Prop :=
  steps.length / 2 = n ∧ steps.all (λ step, step = (1, 1) ∨ step = (1, -1)) ∧
  (steps.scanl (λ acc step => acc + step.2) 0).all (λ x => x ≥ 0)

def hasNoEvenReturn (steps : List (ℤ × ℤ)) : Prop :=
  ∀ i, i < steps.length → 
  steps.nthLe i = (1, -1) →
  ∃ j, j > i ∧ j ≤ steps.length ∧ steps.nthLe j = (1, 1) →
  (j - i).succ % 2 = 1

noncomputable def DyckPaths := { steps : List (ℤ × ℤ) // isDyckPath steps }

noncomputable def DyckPathsWithoutEvenReturn (n : ℕ) :=
{ steps : DyckPaths // hasNoEvenReturn steps.val }

noncomputable def DyckPaths_n (n : ℕ) := { steps : List (ℤ × ℤ) // isDyckPath steps n }

noncomputable def DyckPathsWithoutEvenReturn_n (n : ℕ) :=
{ steps : DyckPaths_n n // hasNoEvenReturn steps.val }

theorem DyckPathsWithoutEvenReturn_corresponds (n : ℕ) :
  (DyckPathsWithoutEvenReturn_n n).card = (DyckPaths_n (n-1)).card :=
sorry

end DyckPathsWithoutEvenReturn_corresponds_l809_809566


namespace radius_of_circle_B_l809_809248

theorem radius_of_circle_B :
  ∀ {A B C D : Type} 
  [has_radius A] [has_radius B] [has_radius C] [has_radius D]
  (externally_tangent : A ⟶ B) (externally_tangent_2 : A ⟶ C) (externally_tangent_3 : B ⟶ C)
  (internally_tangent : A ⟶ D) (internally_tangent_2 : B ⟶ D) (internally_tangent_3 : C ⟶ D)
  (congruent_BC : congruent B C)
  (radius_A : radius A = 2)
  (passes_through_center : ∃ F: center D, passes_through_center A F) :
  radius B = 16/9 :=
by
  sorry

end radius_of_circle_B_l809_809248


namespace total_rotated_cycles_l809_809994

def is_rotated_cycle (π : Fin 11 → Fin 11) (i j k : Fin 11) : Prop :=
  i < j ∧ j < k ∧ π j < π k ∧ π k < π i

theorem total_rotated_cycles : 
  (∑ π in (Equiv.Perm.fintype (Fin 11)).elems, 
     ∑ i in (Finset.range 10).attach, ∑ j in (Finset.Icc (i+1) 10).attach, 
       ∑ k in (Finset.Icc (j+1) 10).attach, if is_rotated_cycle π i j k then 1 else 0) = 72576000 :=
sorry

end total_rotated_cycles_l809_809994


namespace problem1_problem2_l809_809358

noncomputable theory

open Real

def a : ℝ := sqrt 5 + 1
def b : ℝ := sqrt 5 - 1

theorem problem1 : a^2 * b + a * b^2 = 8 * sqrt 5 :=
by
  -- proof goes here
  sorry

theorem problem2 : a^2 - a * b + b^2 = 8 :=
by
  -- proof goes here
  sorry

end problem1_problem2_l809_809358


namespace ronald_profit_fraction_l809_809085

theorem ronald_profit_fraction:
  let initial_units : ℕ := 200
  let total_investment : ℕ := 3000
  let selling_price_per_unit : ℕ := 20
  let total_selling_price := initial_units * selling_price_per_unit
  let total_profit := total_selling_price - total_investment
  (total_profit : ℚ) / total_investment = (1 : ℚ) / 3 :=
by
  -- here we will put the steps needed to prove the theorem.
  sorry

end ronald_profit_fraction_l809_809085


namespace bake_sale_revenue_l809_809228

theorem bake_sale_revenue :
  let betty_choc_chip := 4 * 12,
      betty_oatmeal_raisin := 6 * 12,
      betty_brownies := 2 * 12,
      paige_sugar_cookies := 6 * 12,
      paige_blondies := 3 * 12,
      paige_ccsw_brownies := 5 * 12,
      price_per_cookie := 1,
      price_per_brownie := 2,
      total_cookies := betty_choc_chip + betty_oatmeal_raisin + paige_sugar_cookies,
      total_brownies := betty_brownies + paige_blondies + paige_ccsw_brownies,
      money_from_cookies := total_cookies * price_per_cookie,
      money_from_brownies := total_brownies * price_per_brownie,
      total_money := money_from_cookies + money_from_brownies
  in total_money = 432 := 
by
  sorry

end bake_sale_revenue_l809_809228


namespace y_coordinate_of_equidistant_point_l809_809531

theorem y_coordinate_of_equidistant_point :
  ∃ y : ℝ, (y = -8/3) ∧ 
    (Real.sqrt ((3 - 0)^2 + (0 - y)^2) = Real.sqrt ((4 - 0)^2 + (-3 - y)^2)) :=
by {
  use -8/3,
  split,
  { refl },
  { sorry }
}

end y_coordinate_of_equidistant_point_l809_809531


namespace log_inequalities_A_log_inequalities_D_l809_809541

theorem log_inequalities_A :
  log 8 4 > log 9 4 ∧ log 9 4 > log 10 4 :=
by {
  have h1 : log 8 4 = Real.log 4 / Real.log 8, sorry,
  have h2 : log 9 4 = Real.log 4 / Real.log 9, sorry,
  have h3 : log 10 4 = Real.log 4 / Real.log 10, sorry,
  have h4 : Real.log 8 < Real.log 9, sorry,
  have h5 : (Real.log 4 / Real.log 8) > (Real.log 4 / Real.log 9), sorry,
  have h6 : (Real.log 4 / Real.log 9) > (Real.log 4 / Real.log 10), sorry,
  exact ⟨h5, h6⟩,
}

theorem log_inequalities_D :
  log 0.3 4 < 0.3^2 ∧ 0.3^2 < 2^0.4 :=
by {
  have h1 : log 0.3 4 < 0, sorry,
  have h2 : 0.3^2 = 0.09, sorry,
  have h3 : 2^0.4 < Real.sqrt 2, sorry,
  have h4 : 0.09 < 2^0.4, sorry,
  exact ⟨h1, h4⟩,
}

end log_inequalities_A_log_inequalities_D_l809_809541


namespace num_four_digit_numbers_l809_809730

/- Definitions established from conditions. -/
def digits : List Nat := [2, 3, 3, 5]
def countPermutations (l : List Nat) : Nat :=
  (l.length.factorial / l.foldr (λ d acc, acc * (l.count d).factorial) 1)

/- The target theorem statement. -/
theorem num_four_digit_numbers : countPermutations digits = 12 := by
  sorry

end num_four_digit_numbers_l809_809730


namespace sqrt_product_simplified_l809_809852

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l809_809852


namespace number_greater_than_10_l809_809512

theorem number_greater_than_10 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n + 9 ≤ 9999) :
  ∃ m, m ∈ (list.range' n 10).map (λ k, (nat.min_fac k)) ∧ m > 10 :=
by
  sorry

end number_greater_than_10_l809_809512


namespace tangent_line_at_one_l809_809337

noncomputable def f (x : ℝ) := Real.log x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  let slope := (1/x + 4*x - 4) 
  let y_val := -2 
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), A = 1 ∧ B = -1 ∧ C = -3 ∧ (∀ (x y : ℝ), f x = y → A * x + B * y + C = 0) :=
by
  sorry

end tangent_line_at_one_l809_809337


namespace chocolate_bar_cost_l809_809024

theorem chocolate_bar_cost :
  ∀ (total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips : ℕ),
  total = 150 →
  gummy_bear_cost = 2 →
  chocolate_chip_cost = 5 →
  num_chocolate_bars = 10 →
  num_gummy_bears = 10 →
  num_chocolate_chips = 20 →
  ((total - (num_gummy_bears * gummy_bear_cost + num_chocolate_chips * chocolate_chip_cost)) / num_chocolate_bars = 3) := 
by
  intros total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips 
  intros htotal hgb_cost hcc_cost hncb hngb hncc
  sorry

end chocolate_bar_cost_l809_809024


namespace relationship_between_a_b_c_l809_809741

theorem relationship_between_a_b_c (a b c : ℝ) (ha : a = real.cbrt 7) (hb : b = real.sqrt 5) (hc : c = 2) : a < c ∧ c < b :=
by {
    sorry
}

end relationship_between_a_b_c_l809_809741


namespace simplify_sqrt_expression_correct_l809_809846

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x))

theorem simplify_sqrt_expression_correct (x : ℝ) : 
  simplify_sqrt_expression x = 120 * x * sqrt (x) := 
by 
  sorry

end simplify_sqrt_expression_correct_l809_809846


namespace vanya_first_place_l809_809608

theorem vanya_first_place {n : ℕ} {E A : Finset ℕ} (e_v : ℕ) (a_v : ℕ)
  (he_v : e_v = n)
  (h_distinct_places : E.card = (E ∪ A).card)
  (h_all_worse : ∀ e_i ∈ E, e_i ≠ e_v → ∃ a_i ∈ A, a_i > e_i)
  : a_v = 1 := 
sorry

end vanya_first_place_l809_809608


namespace length_segment_AB_polar_l809_809777

theorem length_segment_AB_polar :
  let A := (4, Real.pi / 6)
  let B := (2, Real.pi / 2)
  let r1 := A.1
  let θ1 := A.2
  let r2 := B.1
  let θ2 := B.2
  let d := real.sqrt (r1^2 + r2^2 - 2 * r1 * r2 * real.cos (θ2 - θ1))
  d = 2 * real.sqrt 3 := by
  sorry

end length_segment_AB_polar_l809_809777


namespace probability_calc_l809_809480

noncomputable def probability_no_exceed_10_minutes : ℝ :=
  let arrival_times := {x : ℝ | 7 + 50 / 60 ≤ x ∧ x ≤ 8 + 30 / 60}
  let favorable_times := {x : ℝ | (7 + 50 / 60 ≤ x ∧ x ≤ 8) ∨ (8 + 20 / 60 ≤ x ∧ x ≤ 8 + 30 / 60)}
  (favorable_times.count.to_real) / (arrival_times.count.to_real)

theorem probability_calc : probability_no_exceed_10_minutes = 1 / 2 :=
  sorry

end probability_calc_l809_809480


namespace ratio_of_faces_l809_809209

-- Define a polyhedron with the given conditions
structure Polyhedron where
  faces : Type
  is_triangle_or_square : faces → Prop
  no_two_squares_share_edge : Prop
  no_two_triangles_share_edge : Prop
  edge_shared : faces → faces → Prop

-- Define the problem with the given number of triangular faces and square faces
def num_faces (poly : Polyhedron) (is_triangle : poly.faces → Prop) (is_square : poly.faces → Prop) : ℕ × ℕ :=
  (poly.faces.filter is_triangle).card, (poly.faces.filter is_square).card

-- The main theorem to be proven
theorem ratio_of_faces {polyhedron : Polyhedron}
  (triangle_faces : polyhedron.faces → Prop)
  (square_faces : polyhedron.faces → Prop)
  (h1 : polyhedron.is_triangle_or_square = λ f, triangle_faces f ∨ square_faces f)
  (h2 : polyhedron.no_two_squares_share_edge)
  (h3 : polyhedron.no_two_triangles_share_edge)
  (h4 : ∀ f1 f2, polyhedron.edge_shared f1 f2 → (triangle_faces f1 ∧ square_faces f2) ∨ (square_faces f1 ∧ triangle_faces f2)) :
  let (t, s) := num_faces polyhedron triangle_faces square_faces in
  (4 * s = 3 * t) → (t : ℚ) / s = 4 / 3 :=
by {
  sorry
}

end ratio_of_faces_l809_809209


namespace count_valid_numbers_l809_809346

-- Defining the conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def sum_of_digits_is (n sum : ℕ) : Prop :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  a + b + c + d = sum

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The main theorem statement
theorem count_valid_numbers : 
  (number_of_valid_numbers : ℕ) = 
  (finset.card (finset.filter (λ n, four_digit_number n ∧ sum_of_digits_is n 12 ∧ divisible_by_5 n) (finset.range 10000))) :=
127

end count_valid_numbers_l809_809346


namespace reduced_price_per_kg_l809_809587

variable (P R Q : ℝ)

theorem reduced_price_per_kg :
  R = 0.75 * P →
  1200 = (Q + 5) * R →
  Q * P = 1200 →
  R = 60 :=
by
  intro h₁ h₂ h₃
  sorry

end reduced_price_per_kg_l809_809587


namespace Jana_new_walking_speed_l809_809020

variable (minutes : ℕ) (distance1 distance2 : ℝ)

-- Given conditions
def minutes_taken_to_walk := 30
def current_distance := 2
def new_distance := 3
def time_in_hours := minutes / 60

-- Define outcomes
def current_speed_per_minute := current_distance / minutes
def current_speed_per_hour := current_speed_per_minute * 60
def required_speed_per_minute := new_distance / minutes
def required_speed_per_hour := required_speed_per_minute * 60

-- Final statement to prove
theorem Jana_new_walking_speed : required_speed_per_hour = 6 := by
  sorry

end Jana_new_walking_speed_l809_809020


namespace computation_l809_809615

theorem computation :
  ( ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * 
    ( (7^3 - 1) / (7^3 + 1) ) * ( (8^3 - 1) / (8^3 + 1) ) 
  ) = (73 / 312) :=
by
  sorry

end computation_l809_809615


namespace smallest_integer_ending_in_9_and_divisible_by_11_l809_809150

theorem smallest_integer_ending_in_9_and_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
  sorry

end smallest_integer_ending_in_9_and_divisible_by_11_l809_809150


namespace tour_groups_and_savings_minimum_people_for_savings_l809_809642

theorem tour_groups_and_savings (x y : ℕ) (m : ℕ):
  (x + y = 102) ∧ (45 * x + 50 * y - 40 * 102 = 730) → 
  (x = 58 ∧ y = 44) :=
by
  sorry

theorem minimum_people_for_savings (m : ℕ):
  (∀ m, m < 50 → 50 * m > 45 * 51) → 
  (m ≥ 46) :=
by
  sorry

end tour_groups_and_savings_minimum_people_for_savings_l809_809642


namespace inequality_with_integrals_l809_809783

variable {f : ℝ → ℝ} {A B : ℝ}

theorem inequality_with_integrals
  (h_continuous : ContinuousOn f (Set.Icc 0 1))
  (h_bounds : ∀ x ∈ Set.Icc 0 1, 0 < A ∧ A ≤ f x ∧ f x ≤ B) :
  A * B * ∫ x in 0..1, (1 / f x) ≤ A + B - ∫ x in 0..1, f x :=
by
  sorry

end inequality_with_integrals_l809_809783


namespace store_profit_l809_809078

variables (m n : ℝ)

def total_profit (m n : ℝ) : ℝ :=
  110 * m - 50 * n

theorem store_profit (m n : ℝ) : total_profit m n = 110 * m - 50 * n :=
  by
  -- sorry indicates that the proof is skipped
  sorry

end store_profit_l809_809078


namespace largest_base5_three_digit_in_base10_l809_809957

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l809_809957


namespace bisects_AX_l809_809410

theorem bisects_AX 
  (ABC : Type*) [EuclideanGeometry ABC]
  (A B C P Q X Y : ABC)
  (hABC : Triangle A B C)
  (hAB_AC : distance A B > distance A C)
  (hP_foot : foot C A B = P)
  (hQ_foot : foot B A C = Q)
  (hX_intersect : line P Q ∩ line B C = X)
  (hC_circumcircles : (circumcircle A X C) ∩ (circumcircle P Q C) = {C, Y}) :
  midpoint P Y A X :=
sorry

end bisects_AX_l809_809410


namespace items_per_baggie_l809_809056

def num_pretzels : ℕ := 64
def num_suckers : ℕ := 32
def num_kids : ℕ := 16
def num_goldfish : ℕ := 4 * num_pretzels
def total_items : ℕ := num_pretzels + num_goldfish + num_suckers

theorem items_per_baggie : total_items / num_kids = 22 :=
by
  -- Calculation proof
  sorry

end items_per_baggie_l809_809056


namespace sum_possible_N_eq_zero_l809_809620

noncomputable def cubic_root_sum (R : ℝ) : ℝ :=
  let eqn := ∀ N : ℝ, N ≠ 0 → N^2 - 2 / N = R → true
  in 0

theorem sum_possible_N_eq_zero (R : ℝ) : cubic_root_sum R = 0 :=
by
  sorry

end sum_possible_N_eq_zero_l809_809620


namespace find_constants_l809_809424

variables (a b c p : ℝ^3) 
variables (s t u : ℝ)

def condition1 : Prop :=
  ∥p - b∥ = 3 * ∥p - a∥

def condition2 : Prop :=
  ∥p - (s • a + t • b + u • c)∥ = ∥p - c∥

theorem find_constants 
  (h1 : condition1 a b p)
  (h2 : condition2 a b c p s t u) :
  s = 9 / 2 ∧ t = 1 / 4 ∧ u = 0 :=
sorry

end find_constants_l809_809424


namespace eccentricity_of_ellipse_l809_809315

-- Definitions based on given conditions
variables {a b c : ℝ} (e : ℝ)

-- Given conditions
def ellipse := ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

-- Define the distances as per the problem conditions
def PF (P : ℝ) (F : ℝ) := b^2 / a
def AF (A : ℝ) (F : ℝ) := a - c

-- The relationship given in the problem
def condition := PF P F = (3/4) * AF A F

-- Question: Prove the eccentricity
theorem eccentricity_of_ellipse (h : ellipse) (h1 : condition) : e = 1 / 4 :=
sorry

end eccentricity_of_ellipse_l809_809315


namespace problem1_problem2_l809_809815

-- Definitions and conditions
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Problem 1
theorem problem1 (a : ℝ) (h : a = 1) (hpq : p a ∧ q) : ∃ x : ℝ, 2 < x ∧ x < 3 := sorry

-- Negations
def not_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 ≥ 0
def not_q (x : ℝ) : Prop := x ≤ 2 ∨ x > 3

-- Problem 2
theorem problem2 (h : ∀ x : ℝ, not_p a x → not_q x) : 1 < a ∧ a ≤ 2 := sorry

end problem1_problem2_l809_809815


namespace total_students_l809_809374

theorem total_students (females : ℕ) (ratio : ℕ) (males := ratio * females) (total := females + males) :
  females = 13 → ratio = 3 → total = 52 :=
by
  intros h_females h_ratio
  rw [h_females, h_ratio]
  simp [total, males]
  sorry

end total_students_l809_809374


namespace log_eight_tenth_l809_809257

theorem log_eight_tenth : logBase 10 (0.8) = -0.097 :=
by
  have hlog2 : logBase 10 (2) ≈ 0.301 := sorry
  have hlog5 : logBase 10 (5) ≈ 0.699 := sorry
  sorry

end log_eight_tenth_l809_809257


namespace problem_I_problem_II_problem_III_l809_809801

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + a * x - real.log x

-- Problem (I)
theorem problem_I (a : ℝ) (a_eq_one : a = 1) :
  (∀ x, 0 < x ∧ x < 1/2 → deriv (λ x, f x a) x < 0) ∧
  (∀ x, 1/2 < x → deriv (λ x, f x a) x > 0) :=
sorry

-- Problem (II)
theorem problem_II (a : ℝ) :
  (∀ x, 0 < x ∧ x ≤ 1 → deriv (λ x, f x a) x ≤ 0) ↔ a ≤ -1 :=
sorry

-- Problem (III)
theorem problem_III (a : ℝ) :
  (∃ t, t = 1 ∧
    let k := deriv (λ x, f x a) t in
    k = (f t a) / t) :=
sorry

end problem_I_problem_II_problem_III_l809_809801


namespace cover_all_positive_integers_l809_809835

noncomputable def sequence (d a : ℕ) : ℕ → ℕ :=
  λ n, a + n * d

def sequences := 
  [
    (3, 1), (4, 2), (6, 5), (8, 4), (9, 3), (12, 9), 
    (16, 8), (18, 15), (32, 0), (36, 27), (48, 0), 
    (64, 16), (72, 45), (96, 16), (192, 176)
  ]

theorem cover_all_positive_integers 
  (S : list (ℕ × ℕ)) 
  (h : S = sequences) :
  ∀ n : ℕ, n > 0 → ∃ (d a : ℕ), (d, a) ∈ S ∧ ∃ k : ℕ, n = sequence d a k :=
sorry

end cover_all_positive_integers_l809_809835


namespace find_b_l809_809511

open Matrix

def vec1 : Fin 3 → ℝ := ![2, 3, -1]
def vec_sum : Fin 3 → ℝ := ![8, 0, -4]
def vec_b : Fin 3 → ℝ := ![36 / 7, -30 / 7, -18 / 7]

theorem find_b :
  ∃ k : ℝ, (k • vec1) + vec_b = vec_sum ∧ dot_product vec_b vec1 = 0 :=
by
  sorry

end find_b_l809_809511


namespace rationalize_denominator_l809_809448

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
  (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11)) = 
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
  A + B + C + D + E + F = 136 := 
sorry

end rationalize_denominator_l809_809448


namespace find_a_l809_809497

theorem find_a (a : ℝ) (h : -2 * a + 1 = -1) : a = 1 :=
by sorry

end find_a_l809_809497


namespace percentage_born_in_september_l809_809120

theorem percentage_born_in_september (total famous : ℕ) (born_in_september : ℕ) (h1 : total = 150) (h2 : born_in_september = 12) :
  (born_in_september * 100 / total) = 8 :=
by
  sorry

end percentage_born_in_september_l809_809120


namespace time_at_constant_speed_l809_809596

-- We declare the constants and variables used in the conditions
def total_distance : ℝ := 21 -- km
def total_time : ℝ := 16 / 60 -- hours
def constant_speed : ℝ := 90 -- km/h

-- We state the property to prove: the time at constant speed is 12 minutes (1/5 hours)
theorem time_at_constant_speed :
  ∃ t_acc t_dec t : ℝ,
    t_acc + t + t_dec = total_time ∧
    (t_acc + t_dec) * (constant_speed / 2) + t * constant_speed = total_distance ∧
    t = 1 / 5 :=
by sorry

end time_at_constant_speed_l809_809596


namespace equivalent_polar_point_l809_809001

-- Definitions for matching Lean's syntax for polar coordinates and conditions
def standard_polar_point (r θ : ℝ) : ℝ × ℝ :=
  if r < 0 then (real.abs r, θ + real.pi) else (r, θ)

noncomputable def standard_equivalent_point : ℝ × ℝ :=
  standard_polar_point (-3) (5 * real.pi / 6)

theorem equivalent_polar_point :
  standard_equivalent_point = (3, 11 * real.pi / 6) :=
by
  sorry

end equivalent_polar_point_l809_809001


namespace values_of_m_and_n_l809_809294

theorem values_of_m_and_n (m n : ℕ) : 
  3 * 10 * m * n = 9! ↔ m * n = 6048 := 
by
  sorry

end values_of_m_and_n_l809_809294


namespace horner_v3_value_l809_809924

def polynomial := λ x : ℝ, 2 + 0.35 * x + 1.8 * x^2 - 3.66 * x^3 + 6 * x^4 - 5.2 * x^5 + x^6

def horner_v3 (a0 a1 a2 a3 a4 a5 a6 x : ℝ) : ℝ :=
let v0 := a6 in
let v1 := v0 * x + a5 in
let v2 := v1 * x + a4 in
let v3 := v2 * x + a3 in
v3

theorem horner_v3_value :
  horner_v3 2 0.35 1.8 (-3.66) 6 (-5.2) 1 (-1) = -15.86 := by
  sorry

end horner_v3_value_l809_809924


namespace Lenny_pens_left_l809_809397

theorem Lenny_pens_left :
  let boxes := 20
  let pens_per_box := 5
  let total_pens := boxes * pens_per_box
  let pens_given_to_friends := 0.4 * total_pens
  let pens_left_after_friends := total_pens - pens_given_to_friends
  let pens_given_to_classmates := (1/4) * pens_left_after_friends
  let pens_left := pens_left_after_friends - pens_given_to_classmates
  pens_left = 45 :=
by
  repeat { sorry }

end Lenny_pens_left_l809_809397


namespace line_through_points_l809_809672

theorem line_through_points:
  ∀ (m b : ℝ), (∃ (m b : ℝ), m = (0 - 2) / (3 - 1) ∧ (2 = m * 1 + b ∧ 0 = m * 3 + b)) → m + b = 2 :=
by 
  intros m b h,
  obtain ⟨m, b, h_slope, ⟨h_point1, h_point2⟩⟩ := h,
  sorry

end line_through_points_l809_809672


namespace cost_price_of_article_l809_809550

variable (C : ℝ)
variable (h1 : (0.18 * C - 0.09 * C = 72))

theorem cost_price_of_article : C = 800 :=
by
  sorry

end cost_price_of_article_l809_809550


namespace amount_paid_Y_l809_809523

theorem amount_paid_Y (X Y : ℝ) (h1 : X + Y = 330) (h2 : X = 1.2 * Y) : Y = 150 := 
by
  sorry

end amount_paid_Y_l809_809523


namespace obtuse_triangle_x_range_l809_809697

theorem obtuse_triangle_x_range (x : ℝ) : 
  (3 < x ∧ x < 7 ∧ 9 + 16 - x^2 < 0) ∨ (1 < x ∧ x < real.sqrt 7) :=
by
  have hx1: 9 + 16 - x^2 > 0 ↔ x < real.sqrt 25 := sorry
  have hx2: 7 - x^2 < 0 ↔ x > 1 := sorry
  exact sorry

end obtuse_triangle_x_range_l809_809697


namespace dusty_change_l809_809225

noncomputable def single_layer_price : ℕ := 4
noncomputable def double_layer_price : ℕ := 7
noncomputable def single_layer_quantity : ℕ := 7
noncomputable def double_layer_quantity : ℕ := 5
noncomputable def payment : ℕ := 100

theorem dusty_change : 
  let total_cost := (single_layer_price * single_layer_quantity) + (double_layer_price * double_layer_quantity)
  in payment - total_cost = 37 := by
  sorry

end dusty_change_l809_809225


namespace rectangle_width_decrease_l809_809489

theorem rectangle_width_decrease (L W : ℝ) (A : ℝ) 
  (hA : A = L * W) :
  let L_new := 1.3 * L in
  let W_new := A / L_new in
  (W_new / W) = (1 / 1.3) →
  (1 - W_new / W) * 100 ≈ 23.08 :=
by
  assume h : (W_new / W) = (1 / 1.3)
  have h1 : A = L * W, from hA
  have h2 : L_new = 1.3 * L, by rfl
  have h3 : (A / (1.3 * L)) / W = 1 / 1.3, from h
  exact sorry

end rectangle_width_decrease_l809_809489


namespace difference_sweaters_Monday_Tuesday_l809_809786

-- Define conditions
def sweaters_knit_on_Monday : ℕ := 8
def sweaters_knit_on_Tuesday (T : ℕ) : Prop := T > 8
def sweaters_knit_on_Wednesday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Thursday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Friday : ℕ := 4

-- Define total sweaters knit in the week
def total_sweaters_knit (T : ℕ) : ℕ :=
  sweaters_knit_on_Monday + T + sweaters_knit_on_Wednesday T + sweaters_knit_on_Thursday T + sweaters_knit_on_Friday

-- Lean Theorem Statement
theorem difference_sweaters_Monday_Tuesday : ∀ T : ℕ, sweaters_knit_on_Tuesday T → total_sweaters_knit T = 34 → T - sweaters_knit_on_Monday = 2 :=
by
  intros T hT_total
  sorry

end difference_sweaters_Monday_Tuesday_l809_809786


namespace radius_of_B_l809_809240

theorem radius_of_B {A B C D : Type} (r_A : ℝ) (r_D : ℝ) (r_B : ℝ) (r_C : ℝ)
  (center_A : A) (center_B : B) (center_C : C) (center_D : D)
  (h_cong_BC : r_B = r_C)
  (h_A_D : r_D = 2 * r_A)
  (h_r_A : r_A = 2)
  (h_tangent_A_D : (dist center_A center_D) = r_A) :
  r_B = 32/25 := sorry

end radius_of_B_l809_809240


namespace arithmetic_sequence_problem_l809_809380

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ) (d : ℕ) (h_d_ne_zero : d ≠ 0) 
  (h1 : a 1 + a 3 = 8) 
  (h2 : (a 4)^2 = (a 2) * (a 9)) :
  a 5 = 13 :=
by {
  -- Definitions for arithmetic sequence
  let a (n : ℕ) := a 1 + (n - 1) * d,
  -- Applying the given conditions
  have h_a1_3: a 1 + a 3 = a 1 + (a 1 + 2 * d) := h1,
  have h_a4_geom: (a 1 + 3 * d)^2 = (a 1 + d) * (a 1 + 8 * d) := h2,
  -- Calculating values of a1 and d
  -- Solving the system of equations
  have h_d: d = 3, 
  have h_a1: a 1 = 1,
  -- Conclusion
  have h_a5: a 5 = 1 + 4 * 3,
  exact h_a5,
  have a_5_value: a 5 = 13,
  exact a_5_value,
}

end arithmetic_sequence_problem_l809_809380


namespace slower_bike_speed_l809_809142

theorem slower_bike_speed:
  ∃ v : ℚ, (∀ d : ℚ, d = 960 →
    ∀ v_fast : ℚ, v_fast = 64 →
    ∀ t_fast : ℚ, t_fast = d / v_fast →
    t_fast + 1 = d / v →
    v = 60) :=
begin
  use 60,
  intros d hd v_fast hv_fast t_fast ht_fast h,
  rw [hd, hv_fast] at *,
  exact h,
  sorry,
end

end slower_bike_speed_l809_809142


namespace B_profit_correct_l809_809592

-- Declare the necessary conditions
variables (A_cp B_sp : ℝ)
variable (A_profit_percent : ℝ)

-- Cost price for A is defined
def A_cost_price : ℝ := 120

-- Profit percentage for A is defined
def A_profit : ℝ := A_profit_percent / 100 * A_cost_price

-- Selling price for B is cost price for B
def B_cost_price : ℝ := A_cost_price + A_profit

-- Final selling price of the bicycle by B to C
def B_selling_price : ℝ := 225

-- Calculate B's profit
def B_profit : ℝ := B_selling_price - B_cost_price

-- Calculate B's profit percentage
def B_profit_percent : ℝ := B_profit / B_cost_price * 100

-- The theorem to prove
theorem B_profit_correct : B_profit_percent = 50 :=
sorry

end B_profit_correct_l809_809592


namespace perpendicular_and_parallel_implies_perpendicular_l809_809406

variables {α β : Type*}
variables [plane α] [plane β]
variables (m n : line)

theorem perpendicular_and_parallel_implies_perpendicular
  (h1 : m ⊥ β) (h2 : n // β) : m ⊥ n :=
sorry

end perpendicular_and_parallel_implies_perpendicular_l809_809406


namespace product_modulo_6_l809_809616

theorem product_modulo_6 :
  (2017 * 2018 * 2019 * 2020) % 6 = 0 :=
by
  -- Conditions provided:
  have h1 : 2017 ≡ 5 [MOD 6] := by sorry
  have h2 : 2018 ≡ 0 [MOD 6] := by sorry
  have h3 : 2019 ≡ 1 [MOD 6] := by sorry
  have h4 : 2020 ≡ 2 [MOD 6] := by sorry
  -- Proof of the theorem:
  sorry

end product_modulo_6_l809_809616


namespace triangle_tangent_condition_l809_809391

noncomputable def is_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 

theorem triangle_tangent_condition (A B C a b c : ℝ) (h1 : is_triangle A B C) (h2 : a = b) : 
  a * tan A + b * tan B = (a + b) * tan ((A + B) / 2) :=
sorry

end triangle_tangent_condition_l809_809391


namespace largest_three_digit_product_l809_809124

theorem largest_three_digit_product : 
    ∃ (n : ℕ), 
    (n = 336) ∧ 
    (n > 99 ∧ n < 1000) ∧ 
    (∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = x * y * (5 * x + 2 * y) ∧ 
        ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ k * m = (5 * x + 2 * y)) :=
by
  sorry

end largest_three_digit_product_l809_809124


namespace frustumLateralSurfaceArea_l809_809575

-- Define the given radii and height
def r1 : ℝ := 4
def r2 : ℝ := 8
def h  : ℝ := 6

-- Define the slant height l using the Pythagorean theorem
def l : ℝ := Real.sqrt ((r2 - r1)^2 + h^2)

-- Define the lateral surface area of the frustum
def lateralSurfaceArea (π : ℝ) (r1 r2 : ℝ) (l : ℝ) : ℝ := π * (r1 + r2) * l

-- Prove the lateral surface area given the conditions
theorem frustumLateralSurfaceArea : 
  lateralSurfaceArea Real.pi r1 r2 l = 12 * Real.pi * Real.sqrt 52 := 
by sorry

end frustumLateralSurfaceArea_l809_809575


namespace find_f_prime_one_l809_809300

-- Given function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * (deriv f 1) - 6

-- The statement we want to prove
-- Prove that the derivative of f at 1 is -2
theorem find_f_prime_one : (deriv f 1) = -2 := by
  -- Insert proof here
  sorry

end find_f_prime_one_l809_809300


namespace sequence_converges_to_fixed_point_l809_809807

open Function Real

def sequence (a : ℕ → ℝ) (α : ℝ) : Prop :=
  a 1 = α ∧ ∀ n, a (n + 1) = cos (a n)

theorem sequence_converges_to_fixed_point (a : ℕ → ℝ) (α : ℝ) (h_seq : sequence a α) :
  ∃ h, (∀ ε > 0, ∃ N, ∀ n ≥ N, abs (a n - h) < ε) ∧ h = cos h :=
sorry

end sequence_converges_to_fixed_point_l809_809807


namespace time_interval_between_glows_l809_809122

noncomputable def time_period : ℕ :=
  (2 * 60 + 2) + (1 * 3600) + (20 * 60 + 47)

def number_of_glows : ℚ :=
  382.2307692307692

theorem time_interval_between_glows :
  ((time_period : ℚ) / number_of_glows) ≈ 13 :=
by
  sorry

end time_interval_between_glows_l809_809122


namespace cubic_sum_of_roots_l809_809417

theorem cubic_sum_of_roots (a b c : ℝ) 
  (h1 : a + b + c = -1)
  (h2 : a * b + b * c + c * a = -333)
  (h3 : a * b * c = 1001) :
  a^3 + b^3 + c^3 = 2003 :=
sorry

end cubic_sum_of_roots_l809_809417


namespace circle_cut_hexagon_possible_l809_809784

-- Definition of a circle with center O and radius r and the associated conditions.

theorem circle_cut_hexagon_possible :
  ∀ (O : Point) (r : ℝ), (r > 0) → 
  (∃ (parts : Set (Set Point)),
    (∀ part ∈ parts, O ∈ boundary part) ∧
    (∃ (hex : Set Point), is_regular_hexagon hex ∧ ⊆ hex circle)) :=
by
  sorry

end circle_cut_hexagon_possible_l809_809784


namespace contractor_daily_wage_l809_809201

constant total_days : ℕ := 30
constant absent_days : ℕ := 12
constant fine_per_absent_day : ℚ := 7.50
constant total_amount_received : ℚ := 360

noncomputable def worked_days := total_days - absent_days
noncomputable def fine_total := absent_days * fine_per_absent_day

theorem contractor_daily_wage :
  ∃ (daily_wage : ℚ), worked_days * daily_wage - fine_total = total_amount_received ∧ daily_wage = 25 :=
by
  -- Conditions
  let worked_days := total_days - absent_days
  have h1 : worked_days = 18 := rfl
  let fine_total := (12 : ℚ) * 7.50
  have h2 : fine_total = 90 := rfl

  -- Equation Setup
  let eqn := worked_days * 25 - fine_total = total_amount_received
  
  -- Proving the theorem
  use 25
  split
  . rw [←h1, ←h2]
    linarith
  . rfl

end contractor_daily_wage_l809_809201


namespace avg_speed_correct_l809_809603

def distance1 : ℝ := 18  -- Distance covered in the first part of the journey (in kilometers)
def time1_minutes : ℝ := 24  -- Time taken in the first part of the journey (in minutes)
def speed2 : ℝ := 72  -- Speed during the second part of the journey (in kilometers per hour)
def time2_minutes : ℝ := 35  -- Time taken in the second part of the journey (in minutes)

def time1 : ℝ := time1_minutes / 60  -- Time in hours for the first part
def time2 : ℝ := time2_minutes / 60  -- Time in hours for the second part
def distance2 : ℝ := speed2 * time2  -- Distance covered in the second part (in kilometers)
def total_distance : ℝ := distance1 + distance2  -- Total distance covered (in kilometers)
def total_time_hours : ℝ := (time1_minutes + time2_minutes) / 60  -- Total time taken for the whole journey (in hours)
def avg_speed_per_hour : ℝ := total_distance / total_time_hours  -- Average speed (in km per hour)
def avg_speed_per_minute : ℝ := avg_speed_per_hour / 60  -- Average speed (in km per minute)

theorem avg_speed_correct : avg_speed_per_minute = 1.02 := by
  -- Proof steps would go here
  sorry

end avg_speed_correct_l809_809603


namespace sum_of_areas_of_tangent_circles_l809_809506

theorem sum_of_areas_of_tangent_circles :
  ∀ (r_A r_B r_C : ℝ), 
  (r_A + r_B = 5) ∧ (r_A + r_C = 12) ∧ (r_B + r_C = 13) →
  (π * (r_A^2 + r_B^2 + r_C^2)) = 113 * π :=
by
  intros r_A r_B r_C,
  sorry

end sum_of_areas_of_tangent_circles_l809_809506


namespace uma_income_l809_809501

theorem uma_income
  (x y : ℝ)
  (h1 : 8 * x - 7 * y = 2000)
  (h2 : 7 * x - 6 * y = 2000) :
  8 * x = 16000 := by
  sorry

end uma_income_l809_809501


namespace transformation_is_rotation_l809_809160

-- Define the 90 degree rotation matrix
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- Define the transformation matrix to be proven
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- The theorem that proves they are equivalent
theorem transformation_is_rotation :
  transformation_matrix = rotation_matrix :=
by
  sorry

end transformation_is_rotation_l809_809160


namespace k_lt_half_plus_sqrt_two_n_l809_809308

theorem k_lt_half_plus_sqrt_two_n
  {S : set (ℝ × ℝ)} {n k : ℕ}
  (hS_size : S.size = n)
  (hS_collinear : ∀ P₁ P₂ P₃ ∈ S, ¬ collinear P₁ P₂ P₃)
  (hS_equidistant : ∀ P ∈ S, ∃ T ⊆ S, T.size ≥ k ∧ ∀ Q₁ Q₂ ∈ T, dist P Q₁ = dist P Q₂) :
  k < (1/2 : ℝ) + real.sqrt (2 * n) :=
by
  sorry

end k_lt_half_plus_sqrt_two_n_l809_809308


namespace largest_base5_three_digit_in_base10_l809_809956

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l809_809956


namespace lawn_chair_sale_price_l809_809194

theorem lawn_chair_sale_price
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (discount_amount : ℝ)
  (sale_price : ℝ)
  (h1 : original_price = 79.95)
  (h2 : discount_percentage = 25.01563477173233)
  (h3 : discount_amount = original_price * discount_percentage / 100)
  (h4 : sale_price = original_price - discount_amount) :
  sale_price ≈ 59.96 := 
sorry

end lawn_chair_sale_price_l809_809194


namespace find_all_n_l809_809469

theorem find_all_n (n : ℕ) (h : n ≤ 1996) : 
  (∃ (m k : ℕ), 
    k ≤ 17 ∧ 
    (∀ i, i ≤ k → ∃ size, size = m ∨ size = m + 1) ∧ 
    (∀ j, j ≤ 17 - k → ∃ size, size = m ∨ size = m + 1) ∧ 
    ∑ (i : ℕ) in finset.range k, size_of i = n ∧ 
    ∑ (j : ℕ) in finset.range (17 - k), size_of j =  n
  ) ↔ 
  n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12 ∨ n = 13 ∨ n = 14 ∨ 
  n = 15 ∨ n = 16 ∨ n = 18 ∨ n = 19 ∨ n = 20 ∨ 
  n = 21 ∨ n = 22 ∨ n = 23 ∨ n = 24 :=
sorry

end find_all_n_l809_809469


namespace smallest_positive_integer_divisible_by_14_15_18_l809_809668

theorem smallest_positive_integer_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ n = 630 :=
sorry

end smallest_positive_integer_divisible_by_14_15_18_l809_809668


namespace assignment_plans_l809_809905

theorem assignment_plans (students locations : ℕ) (library science_museum nursing_home : ℕ) 
  (students_eq : students = 5) (locations_eq : locations = 3) 
  (lib_gt0 : library > 0) (sci_gt0 : science_museum > 0) (nur_gt0 : nursing_home > 0) 
  (lib_science_nursing : library + science_museum + nursing_home = students) : 
  ∃ (assignments : ℕ), assignments = 150 :=
by
  sorry

end assignment_plans_l809_809905


namespace unique_quadruple_exists_l809_809654

theorem unique_quadruple_exists :
  ∃! (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
  a + b + c + d = 2 ∧
  a^2 + b^2 + c^2 + d^2 = 3 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 := by
  sorry

end unique_quadruple_exists_l809_809654


namespace bonds_return_rate_is_six_percent_l809_809425

-- Define the parameters and conditions
def total_investment : ℝ := 10000
def bank_investment : ℝ := 6000
def bank_interest_rate : ℝ := 0.05
def total_annual_income : ℝ := 660
def bonds_investment : ℝ := 6000

-- Define the interest earned from the bank
def bank_interest : ℝ := bank_investment * bank_interest_rate

-- Define the income from the bonds
def bonds_income : ℝ := total_annual_income - bank_interest

-- Define the return rate of the bonds
def bonds_return_rate : ℝ := bonds_income / bonds_investment

-- Prove the expected annual return rate of the bonds
theorem bonds_return_rate_is_six_percent : bonds_return_rate = 0.06 :=
by 
  -- Placeholder for the actual proof
  sorry

end bonds_return_rate_is_six_percent_l809_809425


namespace largest_real_solution_l809_809415

theorem largest_real_solution : 
  (∃ (a b c : ℕ), ∀ (x : ℝ), 
  ( (∀ (x : ℝ), (4 / (x-4)) + (6 / (x-6)) + (18 / (x-18)) + (20 / (x-20)) = x^2 - 12*x - 5 
    → (x = 20))
   → ∃ (a b c : ℕ), a = 20 ∧ b = 0 ∧ c = 0 ∧ 20 = a + √(b + √c))) := 
begin
  -- This is essentially the skeletal Lean statement for the proof problem
  -- We are asserting the existence of a, b, c where the conditions are satisfied
  -- We leave the proof part with 'sorry'
  sorry
end

end largest_real_solution_l809_809415


namespace largest_base5_three_digits_is_124_l809_809931

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l809_809931


namespace ratio_a6_b6_l809_809104

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence a
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence b
noncomputable def S_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence a
noncomputable def T_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence b

axiom condition (n : ℕ) : S_n n / T_n n = (2 * n) / (3 * n + 1)

theorem ratio_a6_b6 : a_n 6 / b_n 6 = 11 / 17 :=
by
  sorry

end ratio_a6_b6_l809_809104


namespace number_of_elements_total_l809_809106

theorem number_of_elements_total 
  (avg_total : ℝ) (avg_pair1 : ℝ) (avg_pair2 : ℝ) (avg_pair3 : ℝ)
  (h_avg_total : avg_total = 3.95)
  (h_avg_pair1 : avg_pair1 = 3.4)
  (h_avg_pair2 : avg_pair2 = 3.85)
  (h_avg_pair3 : avg_pair3 = 4.600000000000001) :
  ∃ (N : ℕ), N = 6 :=
by
  -- Definitions corresponding to provided conditions
  let sum_pair1 := 2 * avg_pair1
  let sum_pair2 := 2 * avg_pair2
  let sum_pair3 := 2 * avg_pair3
  let total_sum := sum_pair1 + sum_pair2 + sum_pair3
  
  have h_sum_pair1 := (2 : ℝ) * h_avg_pair1
  have h_sum_pair2 := (2 : ℝ) * h_avg_pair2
  have h_sum_pair3 := (2 : ℝ) * h_avg_pair3
  have total_sum_eq : total_sum = 23.7 := by sorry
  
  existsi (total_sum / avg_total).natAbs
  have h_avg_total_pos : avg_total > 0 := by linarith
  field_simp at *
  rw h_avg_total
  rw total_sum_eq
  norm_num
  sorry

end number_of_elements_total_l809_809106


namespace geometric_sequence_expression_l809_809322

theorem geometric_sequence_expression (a : ℝ) (a_n: ℕ → ℝ)
  (h1 : a_n 1 = a - 1)
  (h2 : a_n 2 = a + 1)
  (h3 : a_n 3 = a + 4)
  (hn : ∀ n, a_n (n + 1) = a_n n * (a_n 2 / a_n 1)) :
  a_n n = 4 * (3/2)^(n-1) :=
sorry

end geometric_sequence_expression_l809_809322


namespace fraction_approximation_l809_809087

def fraction := 7 / 9

theorem fraction_approximation : Real.toRat (Float.toReal (Float.roundTo (fraction: Float) 2)) = Real.toRat 0.78 :=
by
  -- Add necessary conditions and intermediate steps here
  sorry

end fraction_approximation_l809_809087


namespace installation_quantities_l809_809132

theorem installation_quantities :
  ∃ x1 x2 x3 : ℕ, x1 = 22 ∧ x2 = 88 ∧ x3 = 22 ∧
  (x1 + x2 + x3 ≥ 100) ∧
  (x2 = 4 * x1) ∧
  (∃ k : ℕ, x3 = k * x1) ∧
  (5 * x3 = x2 + 22) :=
  by {
    -- We are simply stating the equivalence and supporting conditions.
    -- Here, we will use 'sorry' as a placeholder.
    sorry
  }

end installation_quantities_l809_809132


namespace complementary_implies_right_triangle_l809_809121

theorem complementary_implies_right_triangle (A B C : ℝ) (h : A + B = 90 ∧ A + B + C = 180) :
  C = 90 :=
by
  sorry

end complementary_implies_right_triangle_l809_809121


namespace polynomial_possible_integer_roots_l809_809210

theorem polynomial_possible_integer_roots :
  ∀(x : ℤ), (x^3 + 2*x^2 - 5*x + 30 = 0) → (x ∈ {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}) :=
by
  sorry

end polynomial_possible_integer_roots_l809_809210


namespace tangent_line_at_1_l809_809323

noncomputable def f (x : ℝ) : ℝ := sorry -- since we don't know the exact form

axiom f_eq : ∀ x : ℝ, f(x) = 2 * f(2 - x) - x^2 + 8 * x - 8

theorem tangent_line_at_1 : ∀ x : ℝ, (1, 1) (x, 2 * x - 1) :=
by
  sorry

end tangent_line_at_1_l809_809323


namespace area_of_extended_quadrilateral_l809_809446

def quadrilateral (α : Type) [EuclideanGeometry α] (a b c d e f g h e' f' g' h' : α) (P : ℝ) : Prop :=
  let EF := dist a b in
  let FF' := dist b f' in
  let FG := dist b c in
  let GG' := dist c g' in
  let GH := dist c d in
  let HH' := dist d h' in
  let HE := dist d e in
  let EE' := dist e e' in
  (EF = 5) ∧ (FF' = 5) ∧ (FG = 7) ∧ (GG' = 7) ∧ (GH = 9) ∧ (HH' = 9) ∧ (HE = 8) ∧ (EE' = 8) ∧ 
  let area_EFGH := 25 in 
  let area_total := 75 in
  P = area_total

theorem area_of_extended_quadrilateral (α : Type) [EuclideanGeometry α]
  (a b c d e f g h e' f' g' h' : α) :
  quadrilateral α a b c d e f g h e' f' g' h' 75 := by
  sorry

end area_of_extended_quadrilateral_l809_809446


namespace dice_labeling_possible_l809_809184

theorem dice_labeling_possible : 
  ∃ (die1 : Fin 6 → ℕ) (die2 : Fin 6 → ℕ), 
  (∀ x1 x2 : Fin 6, let sums := {s | ∃ (a b : ℕ), a = die1 x1 ∧ b = die2 x2 ∧ s = a + b} in sums = (Finset.range 36).image (λ n, n + 1)) :=
sorry

end dice_labeling_possible_l809_809184


namespace ticket_count_l809_809371

theorem ticket_count (x y : ℕ) 
  (h1 : x + y = 35)
  (h2 : 24 * x + 18 * y = 750) : 
  x = 20 ∧ y = 15 :=
by
  sorry

end ticket_count_l809_809371


namespace converse_example_l809_809475

theorem converse_example (x : ℝ) (h : x^2 = 1) : x = 1 :=
sorry

end converse_example_l809_809475


namespace blueberry_jelly_amount_l809_809840

theorem blueberry_jelly_amount (total_jelly : ℕ) (strawberry_jelly : ℕ) 
  (h_total : total_jelly = 6310) 
  (h_strawberry : strawberry_jelly = 1792) 
  : total_jelly - strawberry_jelly = 4518 := 
by 
  sorry

end blueberry_jelly_amount_l809_809840


namespace simplify_sqrt_expression_correct_l809_809847

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x))

theorem simplify_sqrt_expression_correct (x : ℝ) : 
  simplify_sqrt_expression x = 120 * x * sqrt (x) := 
by 
  sorry

end simplify_sqrt_expression_correct_l809_809847


namespace four_digit_sum_divisible_by_5_l809_809350

theorem four_digit_sum_divisible_by_5 :
  ∃ (a b c d : ℕ), 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    1000 * a + 100 * b + 10 * c + d ≥ 1000 ∧
    a + b + c + d = 12 ∧ 
    d ∈ {0, 5} :=
by
  sorry

end four_digit_sum_divisible_by_5_l809_809350


namespace domain_h_l809_809634

def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 6*x - 8) / (x^2 - 5*x + 6)

theorem domain_h : {x : ℝ | h x ∈ ℝ} = {x : ℝ | x ≠ 2 ∧ x ≠ 3} :=
by
  sorry

end domain_h_l809_809634


namespace voucher_c_saves_more_l809_809641

-- Definitions for the problem conditions
def voucher_a_saving (x : ℝ) : ℝ :=
  if x > 100 then 0.10 * x else 0

def voucher_b_saving (x : ℝ) : ℝ :=
  if x > 200 then 30 else 0

def voucher_c_saving (x : ℝ) : ℝ :=
  if x > 200 then 0.20 * (x - 200) else 0

-- Theorem to prove
theorem voucher_c_saves_more (x : ℝ) : 
  (voucher_c_saving x > voucher_a_saving x) ∧ 
  (voucher_c_saving x > voucher_b_saving x) → 
  x > 400 :=
by
  sorry

end voucher_c_saves_more_l809_809641


namespace triangle_O_l809_809014

/-- In triangle ABC, point D is on segment BC. Let O₁ and O₂ be the circumcenters of triangles ABD and ACD respectively. 
Let O' be the center of the circle passing through A, O₁, and O₂. We then conclude that O'D is perpendicular to BC 
if and only if AD passes through the nine-point center of triangle ABC. -/
theorem triangle_O'D_perp_BC_iff_AD_passes_nine_point_center (A B C D O1 O2 O' : Point)
  (h₁ : D ∈ Line B C) 
  (h₂ : Circle.center (circumcircle A B D) = O1)
  (h₃ : Circle.center (circumcircle A C D) = O2)
  (h₄ : Circle.center (circumcircle A O1 O2) = O') 
  (nine_point_center : Point) 
  (h₅ : nine_point_center = ninePointCircleCenter A B C) :
  Line.perpendicular (Line O' D) (Line B C) ↔ Segment.passesThrough (Segment A D) nine_point_center :=
sorry

end triangle_O_l809_809014


namespace sequence_an_general_formula_sequence_bn_sum_l809_809693

theorem sequence_an_general_formula 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_sum : ∀ n, S n = ∑ i in finset.range (n + 1), a i)
  (h_Sn : ∀ n, 4 * S n = (a n + 1) ^ 2)
  : (∀ n, a n = 2 * n - 1) :=
by
  sorry

theorem sequence_bn_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_sum : ∀ n, S n = ∑ i in finset.range (n + 1), a i)
  (h_Sn : ∀ n, 4 * S n = (a n + 1) ^ 2)
  (h_an : ∀ n, a n = 2 * n - 1)
  (h_bn : ∀ n, b n = 1 / (a n * a (n + 1)))
  : (∀ n, ∑ i in finset.range n, b i < 1 / 2) :=
by
  sorry

end sequence_an_general_formula_sequence_bn_sum_l809_809693


namespace rotated_curve_eq_l809_809086

theorem rotated_curve_eq :
  let θ := Real.pi / 4  -- Rotation angle 45 degrees in radians
  let cos_theta := Real.sqrt 2 / 2
  let sin_theta := Real.sqrt 2 / 2
  let x' := cos_theta * x - sin_theta * y
  let y' := sin_theta * x + cos_theta * y
  x + y^2 = 1 → x' ^ 2 + y' ^ 2 - 2 * x' * y' + Real.sqrt 2 * x' + Real.sqrt 2 * y' - 2 = 0 := 
sorry  -- Proof to be provided.

end rotated_curve_eq_l809_809086


namespace mark_reads_1750_pages_per_week_l809_809063

def initialReadingHoursPerDay := 2
def increasePercentage := 150
def initialPagesPerDay := 100

def readingHoursPerDayAfterIncrease : Nat := initialReadingHoursPerDay + (initialReadingHoursPerDay * increasePercentage) / 100
def readingSpeedPerHour := initialPagesPerDay / initialReadingHoursPerDay
def pagesPerDayNow := readingHoursPerDayAfterIncrease * readingSpeedPerHour
def pagesPerWeekNow : Nat := pagesPerDayNow * 7

theorem mark_reads_1750_pages_per_week :
  pagesPerWeekNow = 1750 :=
sorry -- Proof omitted

end mark_reads_1750_pages_per_week_l809_809063


namespace minimum_value_of_expression_l809_809811

theorem minimum_value_of_expression (p q r s t u : ℝ) 
  (hpqrsu_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ 0 < t ∧ 0 < u) 
  (sum_eq : p + q + r + s + t + u = 8) : 
  98 ≤ (2 / p + 4 / q + 9 / r + 16 / s + 25 / t + 36 / u) :=
sorry

end minimum_value_of_expression_l809_809811


namespace optionD_is_quad_eq_in_one_var_l809_809163

/-- Define a predicate for being a quadratic equation in one variable --/
def is_quad_eq_in_one_var (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ ∀ x : ℕ, eq a b c

/-- Options as given predicates --/
def optionA (a b c : ℕ) : Prop := 3 * a^2 - 6 * b + 2 = 0
def optionB (a b c : ℕ) : Prop := a * a^2 - b * a + c = 0
def optionC (a b c : ℕ) : Prop := (1 / a^2) + b = c
def optionD (a b c : ℕ) : Prop := a^2 = 0

/-- Prove that Option D is a quadratic equation in one variable --/
theorem optionD_is_quad_eq_in_one_var : is_quad_eq_in_one_var optionD :=
sorry

end optionD_is_quad_eq_in_one_var_l809_809163


namespace main_l809_809042

def f (x : ℝ) := 2 * x + 3
def g (x : ℝ) := 5 * x + 4

theorem main : f(g(f(3))) = 101 := by
  sorry

end main_l809_809042


namespace Martha_time_spent_l809_809428

theorem Martha_time_spent
  (x : ℕ)
  (h1 : 6 * x = 6 * x) -- Time spent on hold with Comcast is 6 times the time spent turning router off and on again
  (h2 : 3 * x = 3 * x) -- Time spent yelling at the customer service rep is half of time spent on hold, which is still 3x
  (h3 : x + 6 * x + 3 * x = 100) -- Total time spent is 100 minutes
  : x = 10 := 
by
  -- skip the proof steps
  sorry

end Martha_time_spent_l809_809428


namespace probability_heads_penny_dime_halfdollar_l809_809102

def possible_outcomes (n : ℕ) : ℕ := 2 ^ n

def successful_outcomes : ℕ := 8

theorem probability_heads_penny_dime_halfdollar :
  let total_outcomes := possible_outcomes 6 in
  let probability := (successful_outcomes : ℚ) / (total_outcomes : ℚ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_dime_halfdollar_l809_809102


namespace initial_number_2008_l809_809829

theorem initial_number_2008 (x : ℕ) (h : x = 2008 ∨ (∃ y: ℕ, (x = 2*y + 1 ∨ (x = y / (y + 2))))): x = 2008 :=
by
  cases h with
  | inl h2008 => exact h2008
  | inr hexists => cases hexists with
    | intro y hy =>
        cases hy
        case inl h2y => sorry
        case inr hdiv => sorry

end initial_number_2008_l809_809829


namespace right_building_shorter_l809_809836

-- Define the conditions as hypotheses
def middle_building_height : ℕ := 100
def left_building_height : ℕ := (80 * middle_building_height) / 100
def combined_height_left_middle : ℕ := left_building_height + middle_building_height
def total_height : ℕ := 340
def right_building_height : ℕ := total_height - combined_height_left_middle

-- Define the statement we need to prove
theorem right_building_shorter :
  combined_height_left_middle - right_building_height = 20 :=
by sorry

end right_building_shorter_l809_809836


namespace mark_pages_per_week_l809_809059

theorem mark_pages_per_week
    (initial_hours_per_day : ℕ)
    (increase_percentage : ℕ)
    (initial_pages_per_day : ℕ) :
    initial_hours_per_day = 2 →
    increase_percentage = 150 →
    initial_pages_per_day = 100 →
    (initial_pages_per_day * (1 + increase_percentage / 100)) * 7 = 1750 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have reading_speed := 100 / 2 -- 50 pages per hour
  have increased_time := 2 * 1.5  -- 3 more hours
  have new_total_time := 2 + 3    -- 5 hours per day
  have pages_per_day := 5 * 50    -- 250 pages per day
  have pages_per_week := 250 * 7  -- 1750 pages per week
  exact eq.refl 1750

end mark_pages_per_week_l809_809059


namespace midpoint_equidistant_l809_809379

theorem midpoint_equidistant (A B C A1 B1 L K M : Point)
  (h1 : AcuteAngled T ABC)
  (h2 : Altitude T AA1)
  (h3 : Altitude T BB1)
  (h4 : AngleBisectorIntersectsAltitude T ACB L AA1)
  (h5 : AngleBisectorIntersectsAltitude T ACB K BB1)
  (h6 : Midpoint M KL) :
  dist M A1 = dist M B1 :=
sorry

end midpoint_equidistant_l809_809379


namespace collinear_points_in_triangle_l809_809680

theorem collinear_points_in_triangle
  (A B C P M N Q : Point)
  (hABC : triangle ABC)
  (hEquilateral : equilateral_triangle ABC)
  (hCircumcircle : on_circumcircle P ABC)
  (hParallel1 : parallel (line_through P P.offset(B - C)) (line_through M C))
  (hParallel2 : parallel (line_through P P.offset(C - A)) (line_through N B))
  (hParallel3 : parallel (line_through P P.offset(A - B)) (line_through Q C)) :
  collinear {M, N, Q} :=
sorry

end collinear_points_in_triangle_l809_809680


namespace Jonathan_typing_time_l809_809393

theorem Jonathan_typing_time
  (J : ℝ)
  (HJ : 0 < J)
  (rate_Jonathan : ℝ := 1 / J)
  (rate_Susan : ℝ := 1 / 30)
  (rate_Jack : ℝ := 1 / 24)
  (combined_rate : ℝ := 1 / 10)
  (combined_rate_eq : rate_Jonathan + rate_Susan + rate_Jack = combined_rate)
  : J = 40 :=
sorry

end Jonathan_typing_time_l809_809393


namespace correct_option_A_l809_809319

variables {α β : Plane} {m n : Line}

-- Given conditions
def perpendicular_line_plane (m : Line) (α : Plane) : Prop := ⊥
def line_in_plane (n : Line) (β : Plane) : Prop := n ∈ β
def parallel_planes (α β : Plane) : Prop := α ∥ β
def perpendicular_lines (m n : Line) : Prop := m ⊥ n

theorem correct_option_A :
  (perpendicular_line_plane m α) →
  (line_in_plane n β) →
  (parallel_planes α β) →
  (perpendicular_lines m n) :=
by sorry

end correct_option_A_l809_809319


namespace solve_problem_1_solve_problem_2_l809_809611

noncomputable def problem_1 : ℝ :=
  log 500 / log 10 + log (8 / 5) / log 10 - (1 / 2) * (log 64 / log 10) + (log 3 / log 2) * (log 4 / log 3)

theorem solve_problem_1 : problem_1 = 4 := 
by
  sorry

noncomputable def problem_2 : ℝ :=
  0.0081^(-1 / 4) - (3 * (7 / 8)^0)^(-1) * (81^(-0.25) + (3 * (3 / 8))^(-1 / 3))^(-1 / 2)

theorem solve_problem_2 : problem_2 = 3 := 
by
  sorry

end solve_problem_1_solve_problem_2_l809_809611


namespace possible_dice_labels_l809_809182

theorem possible_dice_labels : 
  ∃ (die1 : Fin 6 → Nat) (die2 : Fin 6 → Nat), 
  (∀ k ∈ (Finset.range 1 37), ∃ i j, k = die1 i + die2 j) :=
by
  sorry

end possible_dice_labels_l809_809182


namespace find_n_in_arithmetic_sequence_l809_809798

noncomputable def arithmetic_sequence (a1 d n : ℕ) := a1 + (n - 1) * d

theorem find_n_in_arithmetic_sequence (a1 d an : ℕ) (h1 : a1 = 1) (h2 : d = 5) (h3 : an = 2016) :
  ∃ n : ℕ, an = arithmetic_sequence a1 d n :=
  by
  sorry

end find_n_in_arithmetic_sequence_l809_809798


namespace mean_of_three_numbers_l809_809868

-- Defining the conditions from the problem.
def mean_four_numbers (x y z a : ℝ) : ℝ :=
  (x + y + z + a) / 4

-- Given conditions
def condition1 := (mean_four_numbers x y z 108 = 92)
def condition2 := (true) -- Largest number is 108, already considered

-- The proof goal in Lean statement
theorem mean_of_three_numbers (x y z : ℝ) (h1 : mean_four_numbers x y z 108 = 92) : 
  (x + y + z) / 3 = 260 / 3 :=
begin
  -- Sorry to skip the proof
  sorry,
end

end mean_of_three_numbers_l809_809868


namespace max_ratio_PB_PA_l809_809767

-- Definitions of points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := -2 }
def B : Point := { x := 1, y := -1 }

-- Definition of a point P lying on the circle
def on_circle (P : Point) : Prop :=
  P.x ^ 2 + P.y ^ 2 = 2

-- Function to compute the distance between two points
def distance (P1 P2 : Point) : ℝ :=
  real.sqrt ((P1.x - P2.x) ^ 2 + (P1.y - P2.y) ^ 2)

-- Function to compute the ratio PB/PA
def ratio_PB_PA (P : Point) : ℝ :=
  distance P B / distance P A

-- Theorem stating the maximum value of ratio PB/PA
theorem max_ratio_PB_PA : ∀ (P : Point), on_circle P → ratio_PB_PA P ≤ 2 :=
by
  intro P hP
  sorry

end max_ratio_PB_PA_l809_809767


namespace smallest_abundant_gt_20_l809_809281

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d < n ∧ n % d = 0) (List.range n)

def is_abundant (n : ℕ) : Prop :=
  List.sum (proper_divisors n) > n

theorem smallest_abundant_gt_20 : ∃ n, n > 20 ∧ is_abundant n ∧ ∀ m, (m > 20 ∧ is_abundant m) → n ≤ m :=
by sorry

end smallest_abundant_gt_20_l809_809281


namespace smallest_n_for_p_lt_1_over_3000_l809_809902

theorem smallest_n_for_p_lt_1_over_3000 (n : ℕ) (h₁ : ∀ k : ℕ, (1 ≤ k ∧ k ≤ 3000) → (P k = 1 / (k^2 + 1)))
    (h₂ : ∀ n : ℕ, (P n < 1 / 3000) ↔ (n^2 > 2999)) 
    : n = 55 :=
by
  -- Add the necessary logical transformations that bridge 
  -- the assumptions to the conclusion.
  sorry

noncomputable def P (n : ℕ) : ℚ := 1 / (n^2 + 1)

-- To automatically satisfy necessary conditions
example : ∀ k : ℕ, (1 ≤ k ∧ k ≤ 3000) → (P k = 1 / (k^2 + 1)) := by tautology
example : ∀ n : ℕ, (P n < 1 / 3000) ↔ (n^2 > 2999) := by tautology

end smallest_n_for_p_lt_1_over_3000_l809_809902


namespace tangent_slope_at_pi_over_four_l809_809893

theorem tangent_slope_at_pi_over_four :
  deriv (fun x => Real.tan x) (Real.pi / 4) = 2 :=
sorry

end tangent_slope_at_pi_over_four_l809_809893


namespace find_minimum_values_l809_809302

noncomputable def problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : Prop :=
  (xy_min (x * y ≥ 64) ∧ xy_sum_min (x + y ≥ 18))

theorem find_minimum_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) :
  problem x y hx hy h := by
  sorry

end find_minimum_values_l809_809302


namespace number_of_correct_propositions_l809_809721

variable {b c d : ℝ}
noncomputable def f (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem number_of_correct_propositions
  (h1 : ∀ k, (k < 0 ∨ k > 4) → ∃! x, f(x) = k)
  (h2 : ∀ k, (0 < k ∧ k < 4) → ∃ x1 x2 x3, f(x1) = k ∧ f(x2) = k ∧ f(x3) = k ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :
  (∃ x, f(x) = 4 ∧ f''(x) = 0) ∧
  (∃ x, f(x) = 0 ∧ f''(x) = 0) ∧
  (∀ x1 x2, f(x1) + 3 = 0 → f(x2) - 1 = 0 → x1 > x2) ∧
  (∀ x1 x2, f(x1) + 5 = 0 → f(x2) - 2 = 0 → x1 < x2) :=
sorry

end number_of_correct_propositions_l809_809721


namespace car_stops_at_three_seconds_l809_809865

theorem car_stops_at_three_seconds (t : ℝ) (h : -3 * t^2 + 18 * t = 0) : t = 3 := 
sorry

end car_stops_at_three_seconds_l809_809865


namespace perpendicular_lines_l809_809561

theorem perpendicular_lines (m : ℝ) : 
  (m = -2 → (2-m) * (-(m+3)/(2-m)) + m * (m-3) / (-(m+3)) = 0) → 
  (m = -2 ∨ m = 1) := 
sorry

end perpendicular_lines_l809_809561


namespace rotation_scaling_matrix_l809_809149

noncomputable def transformation_matrix (θ : ℝ) (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let cosine := Real.cos θ
  let sine := Real.sin θ
  s * Matrix.vecCons (Vector.cons cosine (Vector.cons (-sine) Vector.nil))
                    (Matrix.vecCons (Vector.cons sine (Vector.cons cosine Vector.nil)) Matrix.empty)

theorem rotation_scaling_matrix :
  transformation_matrix (Real.pi / 3) 2 = ![![1, -Real.sqrt 3], ![Real.sqrt 3, 1]] :=
by
  sorry

end rotation_scaling_matrix_l809_809149


namespace probability_final_color_green_l809_809137

/-- 
There are initially 7 green amoeba and 3 blue amoeba.
Every minute, each amoeba splits into two identical copies.
After splitting, we randomly remove half of the amoeba so that there are always 10 amoeba.
This process continues until all amoeba are the same color.
This theorem proves the probability that the final color of the amoeba is green is 0.7.
-/
theorem probability_final_color_green 
  (initial_green : ℕ := 7) 
  (initial_blue : ℕ := 3) 
  (total_amoeba : ℕ := 10) 
  : (7 : ℛ)/10 = 0.7 := 
by sorry

end probability_final_color_green_l809_809137


namespace total_number_of_workers_l809_809869

theorem total_number_of_workers (W : ℕ) (R : ℕ) 
  (h1 : (7 + R) * 8000 = 7 * 18000 + R * 6000) 
  (h2 : W = 7 + R) : W = 42 :=
by
  -- Proof steps will go here
  sorry

end total_number_of_workers_l809_809869


namespace length_of_other_bullet_train_l809_809565

-- Definitions of conditions
def length_first_train : ℝ := 270
def speed_first_train_kmph : ℝ := 120
def speed_second_train_kmph : ℝ := 80
def time_to_cross : ℝ := 9

-- Conversion of speeds from kmph to m/s
def speed_first_train_mps : ℝ := speed_first_train_kmph * (1000 / 3600)
def speed_second_train_mps : ℝ := speed_second_train_kmph * (1000 / 3600)

-- Relative speed when trains are moving in opposite directions
def relative_speed : ℝ := speed_first_train_mps + speed_second_train_mps

-- Total distance covered when they cross each other
def total_distance : ℝ := relative_speed * time_to_cross

-- Length of the second train
def length_second_train : ℝ := total_distance - length_first_train

-- Theorem statement
theorem length_of_other_bullet_train : length_second_train = 229.95 := by
  sorry

end length_of_other_bullet_train_l809_809565


namespace mei_age_l809_809548

theorem mei_age (
  li_age : ℕ,
  zhang_age : ℕ,
  jung_age : ℕ,
  mei_age : ℕ
) 
  (h1 : zhang_age = 2 * li_age)
  (h2 : li_age = 12)
  (h3 : jung_age = zhang_age + 2)
  (h4 : mei_age = jung_age / 2)
: mei_age = 13 := 
sorry

end mei_age_l809_809548


namespace calculate_value_l809_809174

theorem calculate_value (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 :=
by
  sorry

end calculate_value_l809_809174


namespace inequality_problem_l809_809813

theorem inequality_problem (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  (a / real.sqrt (a^2 + b^2) + b / real.sqrt (b^2 + c^2) + c / real.sqrt (c^2 + a^2))
  ≤ 3 * real.sqrt 2 / 2 :=
by sorry

end inequality_problem_l809_809813


namespace cuboid_division_l809_809975

-- Given conditions
variables {a b c : ℝ}
hypothesis h1 : a ≤ b
hypothesis h2 : b ≤ c

-- Definition of the ratio for the edges
def edge_ratio (a b c : ℝ) : Prop :=
  a / b = 1 / real.cbrt 2 ∧ b / c = real.cbrt 2 / real.cbrt (4 : ℝ)

-- Lean 4 statement to prove the given problem
theorem cuboid_division (h1 : a ≤ b) (h2 : b ≤ c) :
  ∃ k : ℝ, a = k ∧ b = k * real.cbrt 2 ∧ c = k * real.cbrt 4 :=
by
  sorry

end cuboid_division_l809_809975


namespace rate_equivalence_l809_809226

-- Definitions of principal amounts, rates and times
def P1 : ℕ := 200
def P2 : ℕ := 400
def R1 : ℕ := 10
def T1 : ℕ := 12
def T2 : ℕ := 5

-- Interest calculation function using simple interest formula
def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ :=
(P * R * T) / 100

-- Prove that Rs 400 will produce the same interest in 5 years at the rate of 1.2% per annum as Rs 200 produces in 12 years at 10% per annum.
theorem rate_equivalence : simple_interest P1 R1 T1 = simple_interest P2 1.2 T2 := by
  sorry

end rate_equivalence_l809_809226


namespace linear_function_difference_l809_809411

variable (f : ℝ → ℝ)
variable (h_linear : ∀ x y z w : ℝ, (f x - f y) / (x - y) = (f z - f w) / (z - w))
variable (h_condition : f 8 - f 3 = 15)

theorem linear_function_difference : f 18 - f 5 = 39 :=
by
  have h_slope_consistent := h_linear 8 3 18 5
  rw [h_condition] at h_slope_consistent
  linarith

end linear_function_difference_l809_809411


namespace two_cars_meet_in_l809_809521

noncomputable def distance_traveled (speed : ℕ) (time : ℕ) : ℕ := speed * time

theorem two_cars_meet_in {
  (total_distance : ℕ) (speed1 : ℕ) (speed2 : ℕ) (t : ℚ) 
  (h_total_distance : total_distance = 45)
  (h_speed1 : speed1 = 14)
  (h_speed2 : speed2 = 16)
  (h_sum_speeds : speed1 + speed2 = 30)
  (h_distance_eq : (speed1 + speed2) * t = total_distance)
:
  t = 3 / 2 := by
  rw [h_speed1, h_speed2, h_total_distance, h_sum_speeds] at h_distance_eq
  sorry

end two_cars_meet_in_l809_809521


namespace marriage_problem_l809_809139

def persons : Type := { John, Peter, Alexis, Mary, Kitty, Jenny }
def men : Set persons := { John, Peter, Alexis }
def women : Set persons := { Mary, Kitty, Jenny }
def bought_items : persons → ℕ := λ p, sorry
def spent_pence : persons → ℕ := λ p, (bought_items p) * (bought_items p)

def marriage : persons → persons := λ p, sorry

theorem marriage_problem :
  (∀ p ∈ men, spent_pence p = spent_pence (marriage p) + 63) → 
  (bought_items John = bought_items Kitty + 23) →
  (bought_items Peter = bought_items Mary + 11) →
  (marriage John = Jenny ∧ marriage Peter = Kitty ∧ marriage Alexis = Mary) :=
begin
  sorry -- Proof to be filled in
end

end marriage_problem_l809_809139


namespace solve_sqrt_equation_l809_809275

theorem solve_sqrt_equation (z : ℚ) : sqrt (5 - 5 * z) = 7 → z = -44 / 5 :=
by
  intro h
  -- Proof goes here
  sorry

end solve_sqrt_equation_l809_809275


namespace fill_tank_without_leak_l809_809204

theorem fill_tank_without_leak (T : ℕ) : 
  (1 / T - 1 / 110 = 1 / 11) ↔ T = 10 :=
by 
  sorry

end fill_tank_without_leak_l809_809204


namespace max_cards_with_digit_three_l809_809909

/-- There are ten cards each of the digits "3", "4", and "5". Choose any 8 cards such that their sum is 27. 
Prove that the maximum number of these cards that can be "3" is 6. -/
theorem max_cards_with_digit_three (c3 c4 c5 : ℕ) (hc3 : c3 + c4 + c5 = 8) (h_sum : 3 * c3 + 4 * c4 + 5 * c5 = 27) :
  c3 ≤ 6 :=
sorry

end max_cards_with_digit_three_l809_809909


namespace Lopez_family_seating_arrangements_l809_809468

theorem Lopez_family_seating_arrangements :
  let family := ["Mr. Lopez", "Mrs. Lopez", "eldest child", "child 2", "child 3"]
  let seats := ["driver", "front passenger", "back 1", "back 2", "back 3"]
  let driver_choices := 3
  let front_passenger_choices := 4
  let back_seat_arrangements := 3!
  (driver_choices * front_passenger_choices * back_seat_arrangements) = 72 :=
by
  let family := ["Mr. Lopez", "Mrs. Lopez", "eldest child", "child 2", "child 3"]
  let seats := ["driver", "front passenger", "back 1", "back 2", "back 3"]
  let driver_choices := 3
  let front_passenger_choices := 4
  let back_seat_arrangements := 3!
  show (driver_choices * front_passenger_choices * back_seat_arrangements) = 72
  sorry

end Lopez_family_seating_arrangements_l809_809468


namespace polar_equation_is_circle_of_radius_five_l809_809652

theorem polar_equation_is_circle_of_radius_five :
  ∀ θ : ℝ, (3 * Real.sin θ + 4 * Real.cos θ) ^ 2 = 25 :=
by
  sorry

end polar_equation_is_circle_of_radius_five_l809_809652


namespace trigonometric_evaluation_l809_809356

theorem trigonometric_evaluation
  (θ : ℝ)
  (h : sin θ + 2 * cos θ = 1) :
  (sin θ - cos θ) / (sin θ + cos θ) = -7 ∨ (sin θ - cos θ) / (sin θ + cos θ) = 1 :=
by
  sorry

end trigonometric_evaluation_l809_809356


namespace descent_time_l809_809249

-- Define conditions from the problem
def stationary_time (x y : ℕ) (h1 : 80 * x = y) : Prop := 80 * x = y
def moving_time (x y k : ℕ) (h2 : 40 * (x + k) = y) : Prop := 40 * (x + k) = y
def k_equals_x (x k : ℕ) (h3 : k = x) : Prop := k = x

-- Main statement of the problem
theorem descent_time (x y k : ℕ) (h1 : stationary_time x y (by rfl))
  (h2 : moving_time x y k (by rfl)) (h3 : k_equals_x x k (by rfl)) :
  20 + (y - 20 * x) / (2 * x) = 50 := by
  sorry

end descent_time_l809_809249


namespace polynomial_solution_l809_809646

open Polynomial

noncomputable def P : Polynomial ℚ := sorry
def f (x : ℚ) : ℚ := x^2 + 1/4

theorem polynomial_solution (P : Polynomial ℚ) :
  (∀ n : ℕ, P.eval n ^ 2 + 1/4 = P.eval (n^2 + 1/4)) ↔ 
  ∃ n : ℕ, ∀ x, P.eval x = (nat.iterate (λ t, f t) n x) :=
sorry

end polynomial_solution_l809_809646


namespace maximum_a_is_9_l809_809644

def is_digit (n : ℕ) := n < 10

def is_valid_number (a d e : ℕ) := is_digit a ∧ is_digit d ∧ is_digit e

def last_three_digits (e : ℕ) := 524 + e

def sum_of_digits (a d e : ℕ) := 16 + a + d + e

def divisible_by_8 (n : ℕ) := n % 8 = 0

def divisible_by_3 (s : ℕ) := s % 3 = 0

noncomputable def find_max_a : ℕ :=
  let possible_values := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  possible_values.reverse.find (λ a, ∃ d e, is_valid_number a d e ∧ 
                                          divisible_by_8 (last_three_digits e) ∧ 
                                          divisible_by_3 (sum_of_digits a d e)) | 9

theorem maximum_a_is_9 : find_max_a = 9 := 
  sorry

end maximum_a_is_9_l809_809644


namespace sum_f_1_to_10_l809_809679

-- Define the function f with the properties given.

def f (x : ℝ) : ℝ := sorry

-- Specify the conditions of the problem
local notation "R" => ℝ

axiom odd_function : ∀ (x : R), f (-x) = -f (x)
axiom periodicity : ∀ (x : R), f (x + 3) = f (x)
axiom f_neg1 : f (-1) = 1

-- State the theorem to be proved
theorem sum_f_1_to_10 : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry
end sum_f_1_to_10_l809_809679


namespace proof_problem_l809_809792

open EuclideanGeometry

variables (ABC : Triangle) (O H P Q : Point)

def is_circumcenter (O : Point) (ABC : Triangle) : Prop := 
  Circles.circumcenter O ABC

def is_orthocenter (H : Point) (ABC : Triangle) : Prop :=
  Triangles.orthocenter H ABC

def is_diameter_pt (P : Point) (A O : Point) : Prop :=
  Circles.on_diameter P A O

def is_isogonal_conjugate (P Q : Point) (ABC : Triangle) : Prop :=
  Triangles.isogonal_conjugate P Q ABC

def intersects_1 (A Q R : Point) (BC : Line) : Prop :=
  Lines.intersects A Q BC = R

def intersects_2 (A Q S : Point) (ABC : Triangle) : Prop :=
  Circles.intersects A Q (Triangles.circumcircle ABC) = S

theorem proof_problem
  (h1 : is_circumcenter O ABC)
  (h2 : is_orthocenter H ABC)
  (h3 : is_diameter_pt P A O)
  (h4 : is_isogonal_conjugate P Q ABC)
  (h5 : intersects_1 A Q R BC)
  (h6 : intersects_2 A Q S ABC) :
  QR = RS ∧ ∠ ORB = ∠ SQH :=
sorry

end proof_problem_l809_809792


namespace common_chord_eqv_l809_809771

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
noncomputable def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

def common_chord_parametric (x y : ℝ) : Prop := x = 1 ∧ -real.sqrt 3 ≤ y ∧ y ≤ real.sqrt 3

theorem common_chord_eqv :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord_parametric x y :=
by
  sorry -- The proof will involve showing that (1, √3) and (1, -√3) lie on both circles

end common_chord_eqv_l809_809771


namespace PQ_over_PR_l809_809598

noncomputable def ratio_PQ_PR (a : ℝ) (b : ℝ) (c : ℝ) (Q P R : ℝ × ℝ) : ℝ :=
  let PQ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  let PR := (P.1 - R.1)^2 + (P.2 - R.2)^2
  PQ / PR

theorem PQ_over_PR (a b x1 y1 x2 y2 : ℝ) 
  (h1 : x1^2 / a^2 + y1^2 / b^2 = 1)
  (h2 : x2^2 / a^2 + y2^2 / b^2 = 1)
  (hQ : b = sqrt (a^2 - (3 / 5 * a)^2))
  (hPR_parallel_x : y1 = 0 ∧ y2 = 0)
  (hQ_at_origin : Q = (0, b))
  (hP : P = (-0.3 * a, 0))
  (hR : R = (0.3 * a, 0)) :
  ratio_PQ_PR a b (3 / 5 * a) Q P R = 5 / 3 := by
  sorry

end PQ_over_PR_l809_809598


namespace pow_mod_equiv_l809_809167

theorem pow_mod_equiv (h : 5^500 ≡ 1 [MOD 1250]) : 5^15000 ≡ 1 [MOD 1250] := 
by 
  sorry

end pow_mod_equiv_l809_809167


namespace length_of_AC_l809_809782

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AB BC AC : ℝ)
variables (right_triangle : AB ^ 2 + BC ^ 2 = AC ^ 2)
variables (tan_A : BC / AB = 4 / 3)
variable (AB_val : AB = 4)

theorem length_of_AC :
  AC = 20 / 3 :=
sorry

end length_of_AC_l809_809782


namespace ratio_pq_l809_809416

theorem ratio_pq (p q : ℝ) (h_nonzero_p : p ≠ 0) (h_nonzero_q : q ≠ 0)
  (h_pure_real : (3 - 4 * complex.I) * (complex.of_real p + q * complex.I)).im = 0 :
  p / q = 3 / 4 :=
sorry

end ratio_pq_l809_809416


namespace eq_iff_eq_log_log_nec_not_suff_l809_809687

theorem eq_iff_eq_log {x y : ℝ} (h1 : x > 0) (h2 : y > 0) : (ln x = ln y ↔ x = y) := sorry

theorem log_nec_not_suff {x y : ℝ} : (ln x = ln y → x = y) ∧ ¬(x = y → ln x = ln y) := sorry

end eq_iff_eq_log_log_nec_not_suff_l809_809687


namespace volunteer_arrangements_l809_809571

-- Define the set of volunteers as a parameter
def volunteers : List Char := ['A', 'B', 'C', 'D', 'E', 'F']

-- Condition that A is not on the first day
def notFirstDay (arrangement : List Char) : Prop :=
  arrangement.head ≠ 'A'

-- Condition that B and C are consecutive
def consecutiveBC (arrangement : List Char) : Prop :=
  let pairs := arrangement.zip arrangement.tail
  pairs.contains ('B', 'C') ∨ pairs.contains ('C', 'B')

-- Define the main problem as a theorem
theorem volunteer_arrangements : 
  ∃ (arrangements : Finset (List Char)), 
    arrangements.card = 192 ∧
    (∀ a ∈ arrangements, notFirstDay a) ∧ 
    (∀ a ∈ arrangements, consecutiveBC a) := 
sorry

end volunteer_arrangements_l809_809571


namespace simplify_sqrt_expression_correct_l809_809848

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x))

theorem simplify_sqrt_expression_correct (x : ℝ) : 
  simplify_sqrt_expression x = 120 * x * sqrt (x) := 
by 
  sorry

end simplify_sqrt_expression_correct_l809_809848


namespace amount_paid_l809_809392

def cost_cat_toy : ℝ := 8.77
def cost_cage : ℝ := 10.97
def change_received : ℝ := 0.26

theorem amount_paid : (cost_cat_toy + cost_cage + change_received) = 20.00 := by
  sorry

end amount_paid_l809_809392


namespace compare_powers_inequality_l809_809614

noncomputable def compare_powers : Prop :=
  ∀ (n : ℕ), n ≥ 3 → (n + 1)^2 < 3^n

theorem compare_powers_inequality : compare_powers :=
begin
  sorry -- Proof omitted as specified
end

end compare_powers_inequality_l809_809614


namespace lucy_select_prob_l809_809822

theorem lucy_select_prob : 
  (choose(5, 3) = 10) ∧ (choose(5, 2) = 10) ∧ P1 = (1/10) ∧ P2 = (1/10) ∧ P3 = (1/4) ∧ 
  P_total = (P1 * P2 * P3) → 
  P_total = (1/400) :=
by
  sorry

end lucy_select_prob_l809_809822


namespace general_formula_anan_l809_809010

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n > 0 → 3 * (Finset.range n.succ).sum a = (n + 2) * a n.succ

theorem general_formula_anan : 
  ∃ a : ℕ → ℕ, sequence a ∧ ∀ n : ℕ, n > 0 → a n = n * (n + 1) := 
by
  sorry

end general_formula_anan_l809_809010


namespace davida_worked_more_hours_l809_809626

theorem davida_worked_more_hours :
  ∀ (week1 week2 week3 week4 : ℕ),
  week1 = 35 →
  week2 = 35 →
  week3 = 48 →
  week4 = 48 →
  (week3 + week4) - (week1 + week2) = 26 :=
by
  intros week1 week2 week3 week4
  simp [week1, week2, week3, week4]
  sorry

end davida_worked_more_hours_l809_809626


namespace Holly_throws_5_times_l809_809227

def Bess.throw_distance := 20
def Bess.throw_times := 4
def Holly.throw_distance := 8
def total_distance := 200

theorem Holly_throws_5_times : 
  (total_distance - Bess.throw_times * 2 * Bess.throw_distance) / Holly.throw_distance = 5 :=
by 
  sorry

end Holly_throws_5_times_l809_809227


namespace length_of_chord_formed_by_line_and_circle_l809_809580

theorem length_of_chord_formed_by_line_and_circle :
  ∀ (x y : ℝ), 
    (y = √3 * x) → 
    (x^2 + y^2 - 4 * y = 0) → 
    ∃ (chord_length : ℝ), chord_length = 2 * √3 :=
by
  sorry

end length_of_chord_formed_by_line_and_circle_l809_809580


namespace smallest_nat_mod_5_6_7_l809_809283

theorem smallest_nat_mod_5_6_7 (n : ℕ) :
  n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → n = 209 :=
sorry

end smallest_nat_mod_5_6_7_l809_809283


namespace angle_PKC_eq_90_l809_809766

open EuclideanGeometry

variables (A B C P Q K M : Point)
variables [RightTriangle (Triangle.mk A B C)] [Circumcircle (Triangle.mk A B C) Γ]
variables [IsMidpointOfArc M A B C]
variables [TangentIntersection Γ A P BC] [Intersection PM Γ Q]
variables [TangentIntersection Γ Q A K AC]

theorem angle_PKC_eq_90 (A B C : Point) (P Q K M: Point):
  RightTriangle (Triangle.mk A B C) ∧ 
  Circumcircle (Triangle.mk A B C) ∧ 
  IsMidpointOfArc M (Arc.mk A B C) ∧ 
  TangentIntersection (Circumcircle (Triangle.mk A B C)) A P BC ∧ 
  Intersection PM (Circumcircle (Triangle.mk A B C)) Q ∧ 
  TangentIntersection (Circumcircle (Triangle.mk A B C)) Q K AC 
→ ∠PKC = 90° :=
sorry

end angle_PKC_eq_90_l809_809766


namespace incorrect_statement_among_ABCD_l809_809307

theorem incorrect_statement_among_ABCD
  (x : ℝ)
  (f : ℝ → ℝ)
  (h_f_def : ∀ x, f x = x - ⌊x⌋) :
  ∃ D_correct : Prop, (¬ D_correct ∧
    ∀ A_correct B_correct C_correct : Prop,
    (A_correct → f x ≥ 0) ∧
    (B_correct → ∀ y, y ∈ Ico 0 1 → ¬ (∃ y_max, ∀ y, y < y_max)) ∧
    (C_correct → ∀ k : ℤ, f x = f (x + k)) ∧
    (D_correct → ∀ x, f x = f (-x)) ∧ ¬ D_correct) :=
by
  sorry

end incorrect_statement_among_ABCD_l809_809307


namespace triangle_construction_existence_l809_809624

noncomputable section

open EuclideanGeometry

variables {A B C : Point}
variables (c : ℝ) (h_c : ℝ)
variables (angle_diff : ℝ) -- angle_diff corresponds to |A| - |B|

-- Define the side AB
def Side_AB : Real :=
  dist A B

-- Define the height from C to AB
def Height_from_C_to_AB : Real :=
  by sorry

-- Define the difference in angles A and B
def Angle_difference : Real :=
  angle (A, B, C) - angle (B, C, A)

-- Prove the existence of triangle with given conditions
theorem triangle_construction_existence :
  ∃ (A B C : Point), 
    (dist A B = c) ∧ 
    (Height_from_C_to_AB = h_c) ∧
    (Angle_difference = angle_diff) :=
by
  sorry

end triangle_construction_existence_l809_809624


namespace mark_walk_distance_l809_809058

-- Defining the conditions
def speed : ℝ := 3 -- Speed of both Mark and Chris (miles per hour)
def distance_to_school : ℝ := 9 -- Distance from home to school (miles)
def extra_time_mark_spent : ℝ := 2 -- Extra time Mark spent compared to Chris (hours)

-- Definitions of time taken by Chris and Mark
def time_chris := distance_to_school / speed -- Time taken for Chris to walk to school (hours)
def time_mark := time_chris + extra_time_mark_spent -- Time Mark spends walking (hours)

-- Definition of total distance walked by Mark
def total_distance_mark := speed * time_mark -- Total distance walked by Mark (miles)

-- Distance Mark walked before turning around
def distance_before_turning_around := total_distance_mark / 2

-- Statement to prove
theorem mark_walk_distance :
  distance_before_turning_around = 7.5 := 
by
  sorry

end mark_walk_distance_l809_809058


namespace last_digit_probability_l809_809100

theorem last_digit_probability :
  let m_set := {7, 9, 12, 18, 21}
  let n_set := (finset.range 21).image (2005 + ·)
  let valid_m_set := {2, 8}  -- because only 2 and 8 can end in 6 via cycles
  let valid_n = n_set.filter (λ n, n % 4 = 0) -- only n ≡ 4 (mod 4)
  m_set.card = 5 → n_set.card = 21 →
  let total_pairs := m_set.card * n_set.card
  let successful_pairs := valid_m_set.card * valid_n.card
  total_pairs = 105 → successful_pairs = 8 →
  probability := successful_pairs / total_pairs :
  probability = 8 / 105 :=
by {
  intros,
  sorry
}

end last_digit_probability_l809_809100


namespace ceiling_square_range_l809_809742

theorem ceiling_square_range {x : ℝ} (h : ⌊x⌋ = -11) : 
  (finset.Icc 100 121).card = 22 :=
by
  sorry

end ceiling_square_range_l809_809742


namespace farmer_field_area_l809_809203

theorem farmer_field_area
  (initial_daily_productivity : ℕ := 120)
  (initial_days : ℕ := 12)
  (increased_productivity : ℕ := 150)
  (days_ahead : ℕ := 2) :
  let A := initial_daily_productivity * initial_days in
  A = 1440 := 
by
  -- Problem context and setup
  let days_worked_with_initial_productivity := 2
  let days_worked_with_increased_productivity := initial_days - days_worked_with_initial_productivity - days_ahead
  let area_first_days := days_worked_with_initial_productivity * initial_daily_productivity
  let area_remaining_days := days_worked_with_increased_productivity * increased_productivity
  let total_area := area_first_days + area_remaining_days
  have : A = total_area, by sorry -- Remaining calculations step
  sorry

end farmer_field_area_l809_809203


namespace mod_sum_remainder_l809_809234

theorem mod_sum_remainder :
  (4283 % 5 + 4284 % 5 + 4285 % 5 + 4286 % 5 + 4287 % 5) % 5 = 0 := 
by 
  calc
    4283 % 5 + 4284 % 5 + 4285 % 5 + 4286 % 5 + 4287 % 5
      = 3 + 4 + 0 + 1 + 2 : by norm_num
  ... = 10 : by norm_num
  ... % 5 = 0 : by norm_num

end mod_sum_remainder_l809_809234


namespace no_three_have_same_acquaintances_l809_809562

theorem no_three_have_same_acquaintances (n : ℕ) (h : n ≥ 1) :
    ∃ f : fin n → fin n → bool,
      (∀ i j : fin n, f i j = f j i) ∧
      (∀ i : fin n, f i i = ff) ∧
      (∃ lst : list (fin n), lst.nodup ∧ 
         ∀ i : fin n, count (λ x, f i x = tt) lst = lst.length)
:= sorry

end no_three_have_same_acquaintances_l809_809562


namespace largest_base5_three_digits_is_124_l809_809936

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l809_809936


namespace unique_positive_solution_l809_809655

theorem unique_positive_solution : 
  ∃! x : ℝ, 0 < x ∧ sin (arccos (tanh (arcsin x))) = x ∧ x = (real.sqrt 2) / 2 :=
begin
 sorry
end

end unique_positive_solution_l809_809655


namespace maximal_subset_size_number_of_such_subsets_l809_809031

-- Definition of the set M
def M := {1, 2, ..., 49}

-- Theorem statement about the maximum subset size with no 6 consecutive integers
theorem maximal_subset_size : ∃ S ⊆ M, (|S| = 41) ∧ (∀ {a b}, (a ∈ S ∧ b ∈ S ∧ |a - b| < 6) → false) := sorry

-- Corollary statement about the number of such subsets
theorem number_of_such_subsets : ∃ S ⊆ M, (|S| = 41) ∧
  ∀ T ⊆ M, (|T| = 41) ∧ (∀ {a b}, (a ∈ T ∧ b ∈ T ∧ |a - b| < 6) → false) → 
  S = T ∧ 
  (number_of_such_subsets = 495) := sorry

end maximal_subset_size_number_of_such_subsets_l809_809031


namespace davida_worked_more_hours_l809_809627

theorem davida_worked_more_hours :
  ∀ (week1 week2 week3 week4 : ℕ),
  week1 = 35 →
  week2 = 35 →
  week3 = 48 →
  week4 = 48 →
  (week3 + week4) - (week1 + week2) = 26 :=
by
  intros week1 week2 week3 week4
  simp [week1, week2, week3, week4]
  sorry

end davida_worked_more_hours_l809_809627


namespace length_of_base_l809_809263

-- Define the conditions of the problem
def base_of_triangle (b : ℕ) : Prop :=
  ∃ c : ℕ, b + 3 + c = 12 ∧ 9 + b*b = c*c

-- Statement to prove
theorem length_of_base : base_of_triangle 4 :=
  sorry

end length_of_base_l809_809263


namespace largest_base5_three_digit_to_base10_l809_809966

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l809_809966


namespace probability_satisfied_yogurt_young_population_distribution_expectation_X_max_satisfaction_increase_l809_809546

-- Condition definitions
def total_sample : ℕ := 500
def satisfied_yogurt_elderly : ℕ := 100
def satisfied_yogurt_middle_aged : ℕ := 120
def satisfied_yogurt_young : ℕ := 150

-- Question 1
theorem probability_satisfied_yogurt :
  (satisfied_yogurt_elderly + satisfied_yogurt_middle_aged + satisfied_yogurt_young) / total_sample = 37 / 50 :=
sorry

-- Question 2
def p : ℚ := 3 / 4

def binomial_distribution (n : ℕ) (p : ℚ) := sorry -- Need a proper definition here
def E (X : Type) := sorry -- Need a proper definition here

theorem young_population_distribution :
  binomial_distribution 3 p = sorry := 
sorry

theorem expectation_X :
  E X = 9 / 4 := 
sorry

-- Question 3
def satisfaction_increase (age_group : Type) : ℚ := sorry  -- Definition for increases in different groups

theorem max_satisfaction_increase :
  satisfaction_increase young_population > satisfaction_increase elderly_population ∧ satisfaction_increase young_population > satisfaction_increase middle_aged_population :=
sorry

end probability_satisfied_yogurt_young_population_distribution_expectation_X_max_satisfaction_increase_l809_809546


namespace seniors_count_l809_809760

theorem seniors_count (J S j s : ℕ) 
  (h1 : j = 0.4 * J) 
  (h2 : s = 0.2 * S) 
  (h3 : J + S = 50) 
  (h4 : j = 2 * s) : S = 25 := 
by 
  sorry

end seniors_count_l809_809760


namespace range_of_ac_over_b2_l809_809000

theorem range_of_ac_over_b2 (a b c : ℝ) (A B C : ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : A + B + C = π) 
  (h4 : B = π / 3)
  (h5 : C = 2 * π / 3 - A)
  (h6 : ∃ (t : Triangle ℝ), t = ⟨a, b, c, A, B, C⟩ ∧ t.is_obtuse) :
  ∃ (r : Set ℝ), r = Set.Ioo 0 (2 / 3) ∧ ((ac / (b * b)) ∈ r) :=
sorry

end range_of_ac_over_b2_l809_809000


namespace lines_intersect_on_CD_l809_809386

variables (A B C D K : Point)
variables (AB AD BC CD KC KD : Line)
variables [is_parallel AD BC] [in_AB K]

-- Define the lines through A and B parallel to KC and KD respectively
def line_A_parallel_KC := parallel_line_through A KC
def line_B_parallel_KD := parallel_line_through B KD

-- Define the intersection points of these lines with CD
def L := intersection_point line_A_parallel_KC CD
def L' := intersection_point line_B_parallel_KD CD

theorem lines_intersect_on_CD :
  L = L' :=
by
  -- The proof will be populated here
  sorry

end lines_intersect_on_CD_l809_809386


namespace min_value_vector_prod_exist_equation_line_l_l809_809003

variable (p : ℝ)
variable (A B C N : ℝ × ℝ)

-- Conditions
variable (h1 : A = (x1, y1)) 
variable (h2 : B = (x2, y2))
variable (h3 : C = (p, 0))
variable (h4 : parabola : ((y1 ^ 2 = 2 * p * x1) ∧ (y2 ^ 2 = 2 * p * x2)))
variable (h5 : N = (-p, 0))

-- Proving the minimum value of vector product
theorem min_value_vector_prod (h : p > 0) : 
  ∃ A B C N, h4 →
  ∃ m, ∃ y1 y2, (y1 + y2 = 2 * p * m) ∧ (y1 * y2 = -2 * p^2) →
  ∃ minVal, minVal = 2 * p^2 ∧ 
  ∀ m, parameter (m = 0) → 
  (x1 + p) * (x2 + p) + y1 * y2 = minVal :=
sorry

-- Proving the existence and equation of line l
theorem exist_equation_line_l (h : p > 0) :
  ∃ A C, h1 ∧ h3  →
  ∃ l, (line_perpendicular x_axis l) ∧
  (length_chord_circle_diameter_ac l C = constant) ∧ 
  equation (l = x =  1 / 2 * p) :=
sorry

end min_value_vector_prod_exist_equation_line_l_l809_809003


namespace two_disjoint_paths_l809_809577

variable {G : Type*} [graph : SimpleGraph G]

/-- A condition stating that for any three vertices, there exists a path from A to B that does not pass through C -/
def path_avoiding (G : SimpleGraph G) := 
  ∀ (A B C : G), A ≠ B ∧ A ≠ C ∧ B ≠ C → ∃ p : G.path A B, p.vertices.all (≠ C)

/-- To prove that there exist two disjoint paths from A to B -/
theorem two_disjoint_paths (h : ∀ (A B C : G), A ≠ B ∧ A ≠ C ∧ B ≠ C → ∃ p : G.path A B, p.vertices.all (≠ C)) 
  (A B : G) (H : ∃ (A B C : G), A ≠ B ∧ A ≠ C ∧ B ≠ C) : ∃ p₁ p₂ : G.path A B, p₁ ≠ p₂ ∧ p₁.vertices.disjoint p₂.vertices :=
sorry

end two_disjoint_paths_l809_809577


namespace total_elements_in_C_l809_809451

-- Definitions of the sets C and D and their properties
variables (C D : Set ℕ)
variables (c d : ℕ)

-- Conditions given in the problem
def condition1 := c = 3 * d
def condition2 := (C ∪ D).card = 4500
def condition3 := (C ∩ D).card = 800

-- The statement to prove: the total number of elements in set C is 3975
theorem total_elements_in_C : c = 3975 := 
by {
  have h₁ : c = 3 * d := condition1,
  have h₂ : (C ∪ D).card = 4500 := condition2,
  have h₃ : (C ∩ D).card = 800 := condition3,
  sorry
}

end total_elements_in_C_l809_809451


namespace imag_part_of_Z_l809_809324

-- Given condition: Conjugate of Z
def conjZ : ℂ := (1 - complex.i) / (1 + 2 * complex.i)

-- Prove statement: The imaginary part of Z is 3/5
theorem imag_part_of_Z : ∃ Z : ℂ, complex.conj Z = conjZ ∧ complex.imag Z = 3/5 :=
by
  sorry

end imag_part_of_Z_l809_809324


namespace unique_peg_placement_l809_809621

-- Definitions based on the problem's conditions
def peg_board : Type := { rows : ℕ // rows ∈ [1, 2, 3, 4, 5, 6] }
def colors : Type := {color : String // color ∈ ["yellow", "red", "green", "blue", "orange", "violet"]}
def pegs_per_color : colors → ℕ
| ⟨"yellow", _⟩ := 6
| ⟨"red", _⟩ := 5
| ⟨"green", _⟩ := 4
| ⟨"blue", _⟩ := 3
| ⟨"orange", _⟩ := 2
| ⟨"violet", _⟩ := 1
| _ := 0 -- Fallback, should ideally never hit

-- The theorem statement
theorem unique_peg_placement : 
  ∃ (placement : peg_board → colors), 
    (∀ r1 r2 (pr1 pr2 : peg_board), r1 ≠ r2 → (placement pr1) ≠ (placement pr2)) ∧ 
    (∃! (placement : peg_board → colors), 
      (∀ r (pr : peg_board), pegs_per_color (placement pr) ≤ pr.val)) :=
sorry

end unique_peg_placement_l809_809621


namespace ff_neg_two_eq_two_l809_809413

def f (x : ℝ) : ℝ :=
  if x >= 0 then x - 2 else x ^ 2

theorem ff_neg_two_eq_two : f (f (-2)) = 2 := by
  sorry

end ff_neg_two_eq_two_l809_809413


namespace cars_meet_time_in_minutes_cars_meet_time_in_minutes_proof_l809_809612

-- Definitions of the speeds and times involved in the problem.
def train_speed_kmh : ℝ := 60
def train_length_m : ℝ := 180
def time_train_passes_car_a_s : ℝ := 30
def time_train_passes_car_b_s : ℝ := 6
def encounter_time_min : ℝ := 5

-- Converting speeds and times to consistent units (SI units, i.e., meters and seconds)
def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
def encounter_time_s : ℝ := encounter_time_min * 60

-- Derived speeds for the cars based on the given conditions
def relative_speed_train_car_a : ℝ := train_length_m / time_train_passes_car_a_s
def car_a_speed_ms : ℝ := train_speed_ms - relative_speed_train_car_a

def relative_speed_train_car_b : ℝ := train_length_m / time_train_passes_car_b_s
def car_b_speed_ms : ℝ := relative_speed_train_car_b - train_speed_ms

-- Distance between cars A and B when the train meets car B
def distance_a_b_m : ℝ := encounter_time_s * (train_speed_ms - car_a_speed_ms)

-- Relative speed of cars A and B when moving toward each other
def relative_speed_cars_a_b : ℝ := car_a_speed_ms + car_b_speed_ms

-- Time for cars A and B to meet after the train passes car B
def meeting_time_s : ℝ := distance_a_b_m / relative_speed_cars_a_b

-- Conversion of meeting time from seconds to minutes
def meeting_time_min : ℝ := meeting_time_s / 60

theorem cars_meet_time_in_minutes :
  meeting_time_min = 1.25 :=
by
  rw [meeting_time_min, meeting_time_s, distance_a_b_m, relative_speed_cars_a_b,
      encounter_time_s, car_a_speed_ms, car_b_speed_ms, train_speed_ms,
      relative_speed_train_car_a, relative_speed_train_car_b, train_speed_kmh, train_length_m]
  norm_num

-- The actual proof steps would involve the calculated value for 1.25 minutes, but we use sorry to skip the detailed proof
theorem cars_meet_time_in_minutes_proof :
  meeting_time_min = 1.25 := sorry

end cars_meet_time_in_minutes_cars_meet_time_in_minutes_proof_l809_809612


namespace finding_ratio_l809_809055

-- Define the problem
variables {X Y M P Q : Point ℝ} (d : ℝ)  -- Assumes the points are in ℝ

-- M is the midpoint of XY
def is_midpoint (M X Y : Point ℝ) : Prop :=
  M = (X + Y) / 2

-- Points P and Q lie on a line through Y on opposite sides of Y
def is_on_opposite_sides (P Q Y : Point ℝ) : Prop :=
  ∃ t k : ℝ, t > 0 ∧ k > 0 ∧ P = Y + t * d ∧ Q = Y - k * d

-- |XQ| = 2|MP|
def distance_condition (X Q M P : Point ℝ) : Prop :=
  dist X Q = 2 * dist M P

-- |XY|/2 < |MP| < 3|XY|/2
def midpoint_condition (X Y M P : Point ℝ) : Prop :=
  dist X Y / 2 < dist M P ∧ dist M P < 3 * dist X Y / 2

-- Finding the value of |PY|/|QY| that minimizes |PQ|
theorem finding_ratio (X Y M P Q : Point ℝ) (d : ℝ)
  (hM : is_midpoint M X Y)
  (hOpp : is_on_opposite_sides P Q Y)
  (hDistC : distance_condition X Q M P)
  (hMidC : midpoint_condition X Y M P) :
  ∃ ratio : ℝ, is_minimal_ratio ratio → ratio = ∞ := sorry

end finding_ratio_l809_809055


namespace mark_should_leave_at_9_am_l809_809084

noncomputable theory

def travel_time_rob : ℕ := 1
def travel_time_mark : ℕ := 3 * travel_time_rob
def rob_leaves_at : ℕ := 11 -- representing 11 a.m. in a 24-hour format

def mark_leaves_at : ℕ := rob_leaves_at - (travel_time_mark - travel_time_rob)

theorem mark_should_leave_at_9_am : mark_leaves_at = 9 := 
by
  sorry

end mark_should_leave_at_9_am_l809_809084


namespace lambda_power_eq_one_l809_809250

noncomputable def polynomial (n : ℕ) (a : Fin n → ℝ) (x : ℂ) : ℂ :=
  (Finset.range n).sum (λ i, a i * x ^ (n - i))

theorem lambda_power_eq_one
  (n : ℕ)
  (a : Fin n → ℝ)
  (a0_pos : 0 < a ⟨0, sorry⟩)
  (a_ordered : ∀ i j : Fin n, i ≤ j → a i ≤ a j)
  (a_le_one : ∀ i : Fin n, a i ≤ 1)
  (λ : ℂ)
  (λ_root : polynomial n a λ = 0)
  (λ_norm_ge_one : 1 ≤ |λ|) :
  λ ^ (n + 1) = 1 := by
  sorry

end lambda_power_eq_one_l809_809250


namespace Lenny_pens_left_l809_809396

theorem Lenny_pens_left :
  let boxes := 20
  let pens_per_box := 5
  let total_pens := boxes * pens_per_box
  let pens_given_to_friends := 0.4 * total_pens
  let pens_left_after_friends := total_pens - pens_given_to_friends
  let pens_given_to_classmates := (1/4) * pens_left_after_friends
  let pens_left := pens_left_after_friends - pens_given_to_classmates
  pens_left = 45 :=
by
  repeat { sorry }

end Lenny_pens_left_l809_809396


namespace dave_gets_fewest_pies_l809_809560

noncomputable def abby_pies_area : ℝ := 12
noncomputable def bob_pies_area : ℝ := 9
noncomputable def clara_pies_area : ℝ := 10
noncomputable def dave_pies_area : ℝ := 15
noncomputable def emma_pies_area : ℝ := 8

theorem dave_gets_fewest_pies (D : ℝ) (h : D > 0) :
  let abby_pies := D / abby_pies_area in
  let bob_pies := D / bob_pies_area in
  let clara_pies := D / clara_pies_area in
  let dave_pies := D / dave_pies_area in
  let emma_pies := D / emma_pies_area in
  dave_pies < abby_pies ∧ dave_pies < bob_pies ∧ dave_pies < clara_pies ∧ dave_pies < emma_pies :=
by
  sorry

end dave_gets_fewest_pies_l809_809560


namespace candidates_appeared_in_each_state_equals_7900_l809_809175

theorem candidates_appeared_in_each_state_equals_7900 (x : ℝ) (h : 0.07 * x = 0.06 * x + 79) : x = 7900 :=
sorry

end candidates_appeared_in_each_state_equals_7900_l809_809175


namespace number_of_truthful_monkeys_l809_809004

-- Define the conditions of the problem
def num_tigers : ℕ := 100
def num_foxes : ℕ := 100
def num_monkeys : ℕ := 100
def total_groups : ℕ := 100
def animals_per_group : ℕ := 3
def yes_tiger : ℕ := 138
def yes_fox : ℕ := 188

-- Problem statement to be proved
theorem number_of_truthful_monkeys :
  ∃ m : ℕ, m = 76 ∧
  ∃ (x y z m n : ℕ),
    -- The number of monkeys mixed with tigers
    x + 2 * (74 - y) = num_monkeys ∧

    -- Given constraints
    m ∈ {n : ℕ | n ≤ x} ∧
    n ∈ {n : ℕ | n ≤ (num_foxes - x)} ∧

    -- Equation setup and derived equations
    (x - m) + (num_foxes - y) + n = yes_tiger ∧
    m + (num_tigers - x - n) + (num_tigers - z) = yes_fox ∧
    y + z = 74 ∧
    
    -- ensuring the groups are valid
    2 * (74 - y) = z :=

sorry

end number_of_truthful_monkeys_l809_809004


namespace distance_between_lines_is_four_l809_809514

noncomputable def pointP : ℝ × ℝ := (-2, 4)

noncomputable def circleC (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 25

noncomputable def lineM (a x y : ℝ) : Prop := a * x - 3 * y = 0

noncomputable def lineL (x y : ℝ) : Prop := 4 * x - 3 * y + 20 = 0

theorem distance_between_lines_is_four (a : ℝ) (l_eq_m : a = 4) :
  ∀ (x y : ℝ), lineL x y → lineM a x y → real.abs(20 - 0) / real.sqrt ((4:ℝ)^2 + (-3)^2) = 4 :=
by
  sorry

end distance_between_lines_is_four_l809_809514


namespace chocolate_bar_cost_l809_809026

-- Define the quantities Jessica bought
def chocolate_bars := 10
def gummy_bears_packs := 10
def chocolate_chips_bags := 20

-- Define the costs
def total_cost := 150
def gummy_bears_pack_cost := 2
def chocolate_chips_bag_cost := 5

-- Define what we want to prove (the cost of one chocolate bar)
theorem chocolate_bar_cost : 
  ∃ chocolate_bar_cost, 
    chocolate_bars * chocolate_bar_cost + 
    gummy_bears_packs * gummy_bears_pack_cost + 
    chocolate_chips_bags * chocolate_chips_bag_cost = total_cost ∧
    chocolate_bar_cost = 3 :=
by
  -- Proof goes here
  sorry

end chocolate_bar_cost_l809_809026


namespace area_triangle_ABC_l809_809110

-- Definitions of points and lengths based on conditions
def O := (0, 0)  -- center of the circle
def A := (3, 0)  -- one end of the diameter
def B := (-3, 0) -- other end of the diameter
def D := (-7, 0) -- point D such that BD = 4
def E := (-7, 6) -- point E such that ED = 6 and ED ⊥ AD

-- Translation of the problem into Lean
theorem area_triangle_ABC :
  let C := ((-7 + 3) / 2, (6 + 0) / 2) in  -- C is the intersection point on AE
  let AB := abs(A.1 - B.1) in  -- length of AB, diameter of the circle
  let AC := ((-7 + 3) / 2 - 3) in  -- using the midpoint coordinate of AE for calculations
  let BC := sqrt((A.1 - C.1)^2 + (A.2 - C.2)^2) in  -- length of BC using distance formula
  1 / 2 * BC * AC = 28 / 3 :=     -- area of triangle ABC equals 28/3

sorry

end area_triangle_ABC_l809_809110


namespace triangle_inequality_internal_point_l809_809509

theorem triangle_inequality_internal_point {A B C P : Type} 
  (x y z p q r : ℝ) 
  (h_distances_from_vertices : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distances_from_sides : p > 0 ∧ q > 0 ∧ r > 0)
  (h_x_y_z_triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_p_q_r_triangle_ineq : p + q > r ∧ q + r > p ∧ r + p > q) :
  x * y * z ≥ (q + r) * (r + p) * (p + q) :=
sorry

end triangle_inequality_internal_point_l809_809509


namespace divisors_diff_not_22_l809_809688

theorem divisors_diff_not_22 (n : ℕ) (a b c d : ℕ) (h1 : nat.prime a) (h2 : nat.prime b) (h3 : nat.prime c) (h4 : nat.prime d) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h5 : n = a * b * c * d) (h6 : n < 1995) 
  (divisors : finset ℕ) (h7 : divisors = (nat.divisors n)) (h8 : (divisors.sort (≤)).card = 16) :
  (divisors.sort (≤)).nth 8 - (divisors.sort (≤)).nth 7 ≠ 22 :=
sorry

end divisors_diff_not_22_l809_809688


namespace dice_labeling_possible_l809_809178

theorem dice_labeling_possible :
  ∃ (A B : set ℕ), (A = {1, 2, 3, 4, 5, 6}) ∧ (B = {0, 6, 12, 18, 24, 30}) ∧
  (∀ n ∈ set.range (λ (a ∈ A) (b ∈ B), a + b), 1 ≤ n ∧ n ≤ 36 ∧ set.Surjective (λ (a ∈ A) (b ∈ B), a + b)) :=
by
  sorry

end dice_labeling_possible_l809_809178


namespace ott_fractional_part_l809_809433

theorem ott_fractional_part (M L N O x : ℝ)
  (hM : M = 6 * x)
  (hL : L = 5 * x)
  (hN : N = 4 * x)
  (hO : O = 0)
  (h_each : O + M + L + N = x + x + x) :
  (3 * x) / (M + L + N) = 1 / 5 :=
by
  sorry

end ott_fractional_part_l809_809433


namespace points_scored_per_treasure_l809_809837

-- Define the problem conditions
def treasures_first_level := 5
def treasures_second_level := 2
def total_score := 63

-- Define the points per treasure
def points_per_treasure (p : ℕ) : Prop := (treasures_first_level + treasures_second_level) * p = total_score

-- Theorem to prove the points per treasure
theorem points_scored_per_treasure (p : ℕ) (h : points_per_treasure p) : p = 9 :=
by {
    have h₁ : (treasures_first_level + treasures_second_level) = 7,
    exact rfl,
    have h₂ : 7 * p = total_score,
    rwa [←h₁] at h,
    have h₃ : p = total_score / 7,
    exact (nat.div_eq_of_eq_mul h₂.symm),
    rw [total_score, nat.div_eq_of_eq_mul (eq.refl _)],
    exact (nat.div_eq_of_eq_mul (eq.refl 63)).symm,
    exact rfl,
    exact eq.refl 9,
}

end points_scored_per_treasure_l809_809837


namespace gordon_total_cost_l809_809517

def discount_30 (price : ℝ) : ℝ :=
  price - 0.30 * price

def discount_20 (price : ℝ) : ℝ :=
  price - 0.20 * price

def discounted_price (price : ℝ) : ℝ :=
  if price > 22.00 then discount_30 price
  else if price < 20.00 then discount_20 price
  else price

def total_cost : ℝ :=
  discounted_price 25.00 +
  discounted_price 18.00 +
  discounted_price 21.00 +
  discounted_price 35.00 +
  discounted_price 12.00 +
  discounted_price 10.00

theorem gordon_total_cost : total_cost = 95.00 := 
  by
  sorry

end gordon_total_cost_l809_809517


namespace only_prop_4_true_l809_809039

-- Define types for lines and planes
constant Line : Type
constant Plane : Type

-- Define perpendicular and parallel relationships
constant perp : Line → Line → Prop
constant perp_plane : Line → Plane → Prop
constant parallel : Line → Line → Prop
constant subset : Line → Plane → Prop

-- Definitions and propositions
constants a b c : Line
constants alpha beta gamma : Plane

def prop_1 := perp_plane alpha beta ∧ perp_plane beta gamma → parallel_plane alpha gamma
def prop_2 := perp a b ∧ perp b c → parallel a c ∨ perp a c
def prop_3 := subset a alpha ∧ subset b beta ∧ subset c beta ∧ perp a b ∧ perp a c → perp_plane alpha beta
def prop_4 := perp_plane a alpha ∧ subset b beta ∧ parallel a b → perp_plane alpha beta

-- Theorem stating only prop_4 is true
theorem only_prop_4_true : ¬prop_1 ∧ ¬prop_2 ∧ ¬prop_3 ∧ prop_4 :=
by
  sorry

end only_prop_4_true_l809_809039


namespace complex_number_quadrant_l809_809125

theorem complex_number_quadrant (a : ℝ) : 
  let z := 3 - (a^2 + 1 : ℂ) * I in
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_number_quadrant_l809_809125


namespace sum_of_ages_3_years_ago_l809_809568

noncomputable def siblings_age_3_years_ago (R D S J : ℕ) : Prop :=
  R = D + 6 ∧
  D = S + 8 ∧
  J = R - 5 ∧
  R + 8 = 2 * (S + 8) ∧
  J + 10 = (D + 10) / 2 + 4 ∧
  S + 24 + J = 60 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43

theorem sum_of_ages_3_years_ago (R D S J : ℕ) :
  siblings_age_3_years_ago R D S J :=
by
  intros
  sorry

end sum_of_ages_3_years_ago_l809_809568


namespace symmetric_function_l809_809877

-- Define the function g(x) = log_2(x) for x > 0
def g (x : ℝ) (hx : x > 0) : ℝ := Real.log x / Real.log 2

-- Define the symmetry condition for f
def symmetric_origin (f g : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (-x) = - g x

-- State that f(x) = -log_2(-x) for x < 0 is symmetric with respect to origin
theorem symmetric_function :
  symmetric_origin (λ x, - Real.log (-x) / Real.log 2) g :=
by
  sorry

end symmetric_function_l809_809877


namespace simplify_sqrt_product_l809_809841

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l809_809841


namespace units_digit_of_fraction_l809_809161

theorem units_digit_of_fraction :
  let numer := 30 * 31 * 32 * 33 * 34 * 35
  let denom := 1000
  (numer / denom) % 10 = 6 :=
by
  sorry

end units_digit_of_fraction_l809_809161


namespace estimate_flight_time_l809_809101

-- Define the constants used in the problem.
def radius_of_earth : ℝ := 3950
def speed_of_jet : ℝ := 550

-- Define the formula for the circumference of a circle.
def circumference_of_earth (r : ℝ) : ℝ := 2 * Real.pi * r

-- Define the formula for the time taken to travel the circumference at a constant speed.
def time_to_fly_around_earth (C : ℝ) (v : ℝ) : ℝ := C / v

-- The given estimate calculation of 14.364 * π.
def approximate_time (t : ℝ) : ℝ := t * 3.14

-- The theorem that states the estimated time of flight around the Earth, rounded to the nearest choice.
theorem estimate_flight_time :
  approximate_time (time_to_fly_around_earth (circumference_of_earth radius_of_earth) speed_of_jet) ≈ 45 :=
sorry

end estimate_flight_time_l809_809101


namespace dice_labeling_possible_l809_809185

theorem dice_labeling_possible : 
  ∃ (die1 : Fin 6 → ℕ) (die2 : Fin 6 → ℕ), 
  (∀ x1 x2 : Fin 6, let sums := {s | ∃ (a b : ℕ), a = die1 x1 ∧ b = die2 x2 ∧ s = a + b} in sums = (Finset.range 36).image (λ n, n + 1)) :=
sorry

end dice_labeling_possible_l809_809185


namespace range_of_m_l809_809441

def positive_numbers (a b : ℝ) : Prop := a > 0 ∧ b > 0

def equation_condition (a b : ℝ) : Prop := 9 * a + b = a * b

def inequality_for_any_x (a b m : ℝ) : Prop := ∀ x : ℝ, a + b ≥ -x^2 + 2 * x + 18 - m

theorem range_of_m :
  ∀ (a b m : ℝ),
    positive_numbers a b →
    equation_condition a b →
    inequality_for_any_x a b m →
    m ≥ 3 :=
by
  sorry

end range_of_m_l809_809441


namespace minimum_k_for_two_coloring_l809_809667

theorem minimum_k_for_two_coloring (k : ℕ) :
  (∀ coloring : {x // x ∈ Fin (k + 1)} → ℕ, ∃ a b c d e f g h i j,
    (coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨b, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨c, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨d, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨e, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨f, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨g, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨h, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨i, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     coloring ⟨a, Nat.lt_succ_of_le (Nat.le_refl k)⟩ = 
     coloring ⟨j, Nat.lt_succ_of_le (Nat.le_refl k)⟩ ∧
     a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j) ↔ k = 109 := sorry

end minimum_k_for_two_coloring_l809_809667


namespace sqrt_product_simplified_l809_809851

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l809_809851


namespace shaded_region_area_eq_3_l809_809651

noncomputable def area_of_shaded_region (f g : ℝ → ℝ) (a b : ℝ) :=
  ∫ x in a..b, (g x - f x)

theorem shaded_region_area_eq_3 : 
  area_of_shaded_region (λ x, -1/2 * x + 5) (λ x, -3/4 * x + 7) 2 6 = 3 :=
by
  sorry

end shaded_region_area_eq_3_l809_809651


namespace solve_system_l809_809858

theorem solve_system 
    (x y z : ℝ) 
    (h1 : x + y - 2 + 4 * x * y = 0) 
    (h2 : y + z - 2 + 4 * y * z = 0) 
    (h3 : z + x - 2 + 4 * z * x = 0) :
    (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
sorry

end solve_system_l809_809858


namespace aprilPriceChange_l809_809759

noncomputable def priceChangeInApril : ℕ :=
  let P0 := 100
  let P1 := P0 + (20 / 100) * P0
  let P2 := P1 - (20 / 100) * P1
  let P3 := P2 + (25 / 100) * P2
  let P4 := P3 - x / 100 * P3
  17

theorem aprilPriceChange (x : ℕ) : x = priceChangeInApril := by
  sorry

end aprilPriceChange_l809_809759


namespace valid_expression_l809_809292

theorem valid_expression (x : ℝ) : 
  (x - 1 ≥ 0 ∧ x - 2 ≠ 0) ↔ (x ≥ 1 ∧ x ≠ 2) := 
by
  sorry

end valid_expression_l809_809292


namespace mary_cups_of_flour_l809_809823

/-- 
Given that the recipe calls for 3 cups of sugar,
Mary needs 5 more cups of flour than the amount of sugar,
and she has already put in 2 cups of flour,
prove that the total number of cups of flour the recipe calls for is 8.
-/
theorem mary_cups_of_flour (sugar flour_put in_more : ℕ) (h1 : sugar = 3) 
  (h2 : flour_put = 2) (h3 : in_more = 5) : (sugar + in_more) = 8 :=
by {
  rw [h1, h3],
  norm_num,
}

end mary_cups_of_flour_l809_809823


namespace distinct_values_count_l809_809384

theorem distinct_values_count : 
  ∃ (S : Finset ℕ), S.card = 3 ∧
  ∀ a b c d : ℕ, 
    a ∈ {5, 6, 7, 8} ∧ b ∈ {5, 6, 7, 8} ∧ c ∈ {5, 6, 7, 8} ∧ d ∈ {5, 6, 7, 8} ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ((a * b) + (c * d) + 5) ∈ S :=
begin
  sorry
end

end distinct_values_count_l809_809384


namespace gasoline_price_april_l809_809756

theorem gasoline_price_april (P₀ : ℝ) (P₁ P₂ P₃ P₄ : ℝ) (x : ℝ)
  (h₁ : P₁ = P₀ * 1.20)  -- Price after January's increase
  (h₂ : P₂ = P₁ * 0.80)  -- Price after February's decrease
  (h₃ : P₃ = P₂ * 1.25)  -- Price after March's increase
  (h₄ : P₄ = P₃ * (1 - x / 100))  -- Price after April's decrease
  (h₅ : P₄ = P₀)  -- Price at the end of April equals the initial price
  : x = 17 := 
by
  sorry

end gasoline_price_april_l809_809756


namespace golden_ratio_division_l809_809543

-- Definition: A point C divides a line segment AB at the golden ratio if the ratio of the whole
-- segment (AB) to the larger segment (BC) is the same as the ratio of the larger segment (BC)
-- to the smaller segment (AC), i.e., AB / BC = BC / AC, which simplifies to the golden ratio φ.
-- Let's denote φ = (1 + sqrt(5)) / 2

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

def divides_at_golden_ratio (A B C: ℝ) : Prop :=
  let AB := B - A
  let AC := C - A
  let BC := B - C
  AB / BC = phi ∧ BC / AC = phi

-- Theorem: The golden ratio point divides a line segment into two line segments, with the longer one being approximately 0.618 times the length of the original segment.
theorem golden_ratio_division (A B C: ℝ) (h: divides_at_golden_ratio A B C):
  let AB := B - A in
  let BC := B - C in
  BC ≈ 0.618 * AB :=
sorry

end golden_ratio_division_l809_809543


namespace find_a_l809_809677

-- Define the polynomial and its properties
noncomputable def poly := λ (a b : ℝ) (x : ℝ), 9 * x^3 + 5 * a * x^2 + 4 * b * x + a

-- Define the roots of the polynomial
variable {r s t : ℝ}

-- Define the conditions
variable (hrst : r ≠ s ∧ s ≠ t ∧ r ≠ t)
variable (hroots : poly a b r = 0 ∧ poly a b s = 0 ∧ poly a b t = 0)
variable (hlog_sum : real.log2 r + real.log2 s + real.log2 t = 4)

-- State the goal
theorem find_a (a b : ℝ) : 
  (∃ r s t : ℝ, r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
  poly a b r = 0 ∧ poly a b s = 0 ∧ poly a b t = 0 ∧
  real.log2 r + real.log2 s + real.log2 t = 4) → 
  a = -144 :=
begin
  sorry
end

end find_a_l809_809677


namespace distance_mn_l809_809755

theorem distance_mn :
  let A := (0 : ℝ, 0 : ℝ, 0 : ℝ),
      B := (4 : ℝ, 0 : ℝ, 0 : ℝ),
      C := (6 : ℝ, 2 : ℝ, 0 : ℝ),
      D := (2 : ℝ, 2 : ℝ, 0 : ℝ),
      A' := (0 : ℝ, 0 : ℝ, 12 : ℝ),
      B' := (4 : ℝ, 0 : ℝ, 10 : ℝ),
      C' := (6 : ℝ, 2 : ℝ, 20 : ℝ),
      D' := (2 : ℝ, 2 : ℝ, 24 : ℝ),
      M := ((A'.1 + C'.1) / 2, (A'.2 + C'.2) / 2, (A'.3 + C'.3) / 2),
      N := ((B'.1 + D'.1) / 2, (B'.2 + D'.2) / 2, (B'.3 + D'.3) / 2)
  in dist M N = 1 :=
by
  sorry

end distance_mn_l809_809755


namespace num_math_books_l809_809552

theorem num_math_books (total_books total_cost math_book_cost history_book_cost : ℕ) (M H : ℕ)
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 368)
  (h5 : M + H = total_books)
  (h6 : math_book_cost * M + history_book_cost * H = total_cost) :
  M = 32 :=
by
  sorry

end num_math_books_l809_809552


namespace largest_base5_three_digit_in_base10_l809_809953

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l809_809953


namespace tom_mowing_lawn_l809_809917

theorem tom_mowing_lawn (hours_to_mow : ℕ) (time_worked : ℕ) (fraction_mowed_per_hour : ℚ) : 
  (hours_to_mow = 6) → 
  (time_worked = 3) → 
  (fraction_mowed_per_hour = (1 : ℚ) / hours_to_mow) → 
  (1 - (time_worked * fraction_mowed_per_hour) = (1 : ℚ) / 2) :=
by
  intros h1 h2 h3
  sorry

end tom_mowing_lawn_l809_809917


namespace absolute_comparative_advantage_trade_benefits_no_trade_harms_l809_809177

-- Definitions based on given conditions
def yield_A := (zucchinis : ℕ → ℕ) | cauliflower : ℕ → ℕ
def yield_B := (zucchinis : ℕ → ℕ) | cauliflower : ℕ → ℕ

-- Scenario for absolute and comparative advantage
theorem absolute_comparative_advantage :
  (yield_B.zucchinis > yield_A.zucchinis) ∧ 
  (yield_B.cauliflower > yield_A.cauliflower) ∧ 
  ((yield_A.zucchinis / yield_A.cauliflower < yield_B.zucchinis / yield_B.cauliflower) ∧ 
   (yield_B.cauliflower / yield_B.zucchinis < yield_A.cauliflower / yield_A.zucchinis)) := sorry

-- Scenario for trade benefits
theorem trade_benefits :
  let A_trade := (yield_A.cauliflower, yield_A.cauliflower * (yield_B.zucchinis / yield_B.cauliflower)) in
  let B_trade := (yield_B.zucchinis * (yield_A.cauliflower / yield_A.zucchinis), yield_B.zucchinis) in
  (A_trade.1 + A_trade.2 > 16) ∧ (B_trade.1 + B_trade.2 > 36) := sorry

-- Scenario for unified country without trade
theorem no_trade_harms :
  let total_production := (yield_A.cauliflower + yield_B.cauliflower, yield_A.zucchinis + yield_B.zucchinis) in
  let balanced_production := (total_production.1 / 2, total_production.2 / 2) in
  (balanced_production.1 < total_production.1) ∧ (balanced_production.2 < total_production.2) := sorry

end absolute_comparative_advantage_trade_benefits_no_trade_harms_l809_809177


namespace pens_count_l809_809168

theorem pens_count : 
    let start_pens := 25 in
    let mikes_pens := 22 in
    let cindys_double := (start_pens + mikes_pens) * 2 in
    let sharons_pens := 19 in
    cindys_double - sharons_pens = 75 :=
by 
  sorry

end pens_count_l809_809168


namespace centroid_positions_count_l809_809862

noncomputable def is_non_collinear (P Q R : (ℝ × ℝ)) : Prop :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xR, yR) := R
  (xQ - xP) * (yR - yP) ≠ (xR - xP) * (yQ - yP)

noncomputable def points_on_hexagon : List (ℝ × ℝ) :=
  -- Assume this function returns the 60 points equally spaced around a regular hexagon
  sorry

theorem centroid_positions_count :
  let points := points_on_hexagon
  let centroids := { ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3) | P Q R ∈ points, is_non_collinear P Q R }
  centroids.size = 961 :=
sorry

end centroid_positions_count_l809_809862


namespace largest_base5_three_digit_to_base10_l809_809964

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l809_809964


namespace not_decreasing_in_interval_l809_809423

def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

theorem not_decreasing_in_interval : ∃ (x₁ x₂ : ℝ), (Real.pi / 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi) ∧ (f x₁ < f x₂ ∨ f x₁ > f x₂) :=
by
  sorry

end not_decreasing_in_interval_l809_809423


namespace rational_statements_correctness_l809_809217

theorem rational_statements_correctness :
  let S1 := (∀ (r : ℚ), ∃ (x : ℝ), x = r) in
  let S2 := (∀ (x : ℝ), ∃ (r : ℚ), x = r) in
  let S3 := (∀ (r : ℚ), abs r >= 0) in
  let S4 := (∀ (r : ℚ), ∃ (r' : ℚ), r' = -r) in
  (S1 ∧ ¬S2 ∧ S3 ∧ S4) → 3 :=
by
  intros
  sorry

end rational_statements_correctness_l809_809217


namespace N_eq_P_l809_809314

def N : Set ℝ := {x | ∃ n : ℤ, x = (n : ℝ) / 2 - 1 / 3}
def P : Set ℝ := {x | ∃ p : ℤ, x = (p : ℝ) / 2 + 1 / 6}

theorem N_eq_P : N = P :=
  sorry

end N_eq_P_l809_809314


namespace car_stops_at_3_seconds_l809_809866

-- Define the distance function S(t)
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

-- The proof statement
theorem car_stops_at_3_seconds :
    (∃ t : ℝ, S t = 27 ∧ t = 3) :=
begin
  use 3,
  split,
  { unfold S,
    simp,
    exact eq.refl 27 },
  { exact eq.refl 3 },
end

end car_stops_at_3_seconds_l809_809866


namespace find_ellipse_equation_l809_809338

noncomputable def hyperbola := ∀ x y: ℝ, (x^2 / 9) - (y^2 / 3) = 1

noncomputable def ellipse_equation :=
  ∃ a b: ℝ, a = 2 * Real.sqrt 3 ∧ b^2 = a^2 - 3 ∧ elliptic_eq(x, y, a, b)

def elliptic_eq (x y a b : ℝ) : Prop := (x^2 / (a^2 - 3)) + (y^2 / (b^2)) = 1

theorem find_ellipse_equation : 
  ∀ (x y: ℝ), hyperbola x y → ellipse_equation :=
sorry

end find_ellipse_equation_l809_809338


namespace smaller_sphere_radius_l809_809572

theorem smaller_sphere_radius (r R : ℝ) (h : ℝ) (V_cone V_sphere_small V_sphere_large : ℝ) :
  R = 2 * r →
  h = 8 * r →
  V_cone = (1 / 3) * Real.pi * (sqrt 8 * r)^2 * h →
  V_sphere_small = (4 / 3) * Real.pi * r^3 →
  V_sphere_large = (4 / 3) * Real.pi * (2 * r)^3 →
  V_cone - (V_sphere_small + V_sphere_large) = 2016 * Real.pi →
  r = 6 :=
by
  sorry

end smaller_sphere_radius_l809_809572


namespace meal_ticket_probability_l809_809591

/-- Xiaoming has certain meal tickets with specified denominations.
Given he randomly draws 2 tickets from his pocket,
this theorem proves that the probability that the sum of the values of the drawn tickets
is at least 4 yuan is 1/2. -/
theorem meal_ticket_probability :
  let tickets := [1, 1, 2, 2, 5] in
  let total_outcomes := Nat.choose (List.length tickets) 2 in
  let favorable_outcomes :=
    ((Nat.choose 2 1) * (Nat.choose 1 1)) +   -- (2) ~ (2, 5)
    ((Nat.choose 2 1) * (Nat.choose 1 1)) +   -- (1) ~ (2, 2)
    (Nat.choose 2 2)                         -- (2) ~ (5, _)
  in
  favorable_outcomes / total_outcomes = 1 / 2 := sorry

end meal_ticket_probability_l809_809591


namespace measure_angle_PMN_is_60_l809_809774

-- Assume points P, M, N, Q, and R and angles are given.
variable (P M N Q R : Point)
variable (angle : Point → Point → Point → ℝ)

-- Define the conditions
def is_isosceles_PMN : Prop := distance P M = distance P N
def PR_equals_RQ : Prop := distance P R = distance R Q
def angle_PQR_60 : Prop := angle P Q R = 60

-- Define the target statement
theorem measure_angle_PMN_is_60
  (h1 : is_isosceles_PMN P M N)
  (h2 : PR_equals_RQ P R Q)
  (h3 : angle_PQR_60 P Q R) :
  angle P M N = 60 :=
by
  sorry -- Proof is omitted

end measure_angle_PMN_is_60_l809_809774


namespace find_k_l809_809491

theorem find_k (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_parabola_A : y₁ = x₁^2)
  (h_parabola_B : y₂ = x₂^2)
  (h_line_A : y₁ = x₁ - k)
  (h_line_B : y₂ = x₂ - k)
  (h_midpoint : (y₁ + y₂) / 2 = 1) 
  (h_sum_x : x₁ + x₂ = 1) :
  k = -1 / 2 :=
by sorry

end find_k_l809_809491


namespace rationalizing_factor_sqrt2_minus_1_eliminate_sqrt_denominators_compare_sqrt_expressions_l809_809449

-- The rationalizing factor of (sqrt 2 - 1) is (sqrt 2 + 1)
theorem rationalizing_factor_sqrt2_minus_1 : 
  let x := Real.sqrt 2 in 
  ((x - 1) * (x + 1) = x^2 - 1) := 
sorry

-- Eliminating square roots in the denominators of given expressions
theorem eliminate_sqrt_denominators : 
  (2 / (3 * Real.sqrt 2) = Real.sqrt 2 / 3) ∧
  (3 / (3 - Real.sqrt 6) = 3 + Real.sqrt 6) := 
sorry

-- Comparing sqrt(2023) - sqrt(2022) and sqrt(2022) - sqrt(2021)
theorem compare_sqrt_expressions : 
  (Real.sqrt 2023 - Real.sqrt 2022 < Real.sqrt 2022 - Real.sqrt 2021) := 
sorry

end rationalizing_factor_sqrt2_minus_1_eliminate_sqrt_denominators_compare_sqrt_expressions_l809_809449


namespace slope_OA_l809_809009

-- Definitions for the given conditions
def ellipse (a b : ℝ) := {P : ℝ × ℝ | (P.1^2) / a^2 + (P.2^2) / b^2 = 1}

def C1 := ellipse 2 1  -- ∑(x^2 / 4 + y^2 = 1)
def C2 := ellipse 2 4  -- ∑(y^2 / 16 + x^2 / 4 = 1)

variable {P₁ P₂ : ℝ × ℝ}  -- Points A and B
variable (h1 : P₁ ∈ C1)
variable (h2 : P₂ ∈ C2)
variable (h_rel : P₂.1 = 2 * P₁.1 ∧ P₂.2 = 2 * P₁.2)  -- ∑(x₂ = 2x₁, y₂ = 2y₁)

-- Proof that the slope of ray OA is ±1
theorem slope_OA : ∃ (m : ℝ), (m = 1 ∨ m = -1) :=
sorry

end slope_OA_l809_809009


namespace pears_sold_l809_809590

theorem pears_sold (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : a = 240) : total = 360 :=
by
  sorry

end pears_sold_l809_809590


namespace mark_pages_per_week_l809_809066

theorem mark_pages_per_week :
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100) * initial_reading_hours
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  increased_pages_per_day * days_per_week = 1750 :=
by
  -- Definitions
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  
  -- Calculate increased reading hours per day
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100.0) * initial_reading_hours
  -- Calculate increased pages per day
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  
  -- Calculate pages per week
  have h : increased_pages_per_day * days_per_week = 1750 := by
    sorry

  exact h

end mark_pages_per_week_l809_809066


namespace solve_sqrt_equation_l809_809649

theorem solve_sqrt_equation (z : ℤ) (h : sqrt (5 - 4 * z) = 7) : z = -11 :=
by
  sorry

end solve_sqrt_equation_l809_809649


namespace rahul_and_sham_together_complete_task_in_35_days_l809_809079

noncomputable def rahul_rate (W : ℝ) : ℝ := W / 60
noncomputable def sham_rate (W : ℝ) : ℝ := W / 84
noncomputable def combined_rate (W : ℝ) := rahul_rate W + sham_rate W

theorem rahul_and_sham_together_complete_task_in_35_days (W : ℝ) :
  (W / combined_rate W) = 35 :=
by
  sorry

end rahul_and_sham_together_complete_task_in_35_days_l809_809079


namespace smallest_integer_ending_in_9_and_divisible_by_11_l809_809151

theorem smallest_integer_ending_in_9_and_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
  sorry

end smallest_integer_ending_in_9_and_divisible_by_11_l809_809151


namespace min_employees_birthday_Wednesday_l809_809613

theorem min_employees_birthday_Wednesday (W D : ℕ) (h_eq : W + 6 * D = 50) (h_gt : W > D) : W = 8 :=
sorry

end min_employees_birthday_Wednesday_l809_809613


namespace find_angle_C_max_perimeter_l809_809696

-- Conditions for the triangle and given equations
variables {A B C : ℝ} (a b c : ℝ)
variables (h1 : (a - 2 * b) * Real.cos C + c * Real.cos A = 0)
variables (h2 : c = 2 * Real.sqrt 3)

-- Statement 1: Finding angle C
theorem find_angle_C (h1 : (a - 2 * b) * Real.cos C + c * Real.cos A = 0) : C = Real.pi / 3 :=
by
  sorry

-- Statement 2: Finding the maximum perimeter of triangle ABC
theorem max_perimeter (h1 : (a - 2 * b) * Real.cos C + c * Real.cos A = 0) (h2 : c = 2 * Real.sqrt 3) :
  let perimeter := a + b + c in
  perimeter ≤ 6 * Real.sqrt 3 :=
by
  sorry

end find_angle_C_max_perimeter_l809_809696


namespace midline_parallel_and_equal_l809_809143

open Set

variable {Point : Type} [AffineSpace Point]

variables (A B C D A1 B1 C1 D1 M N M1 N1 : Point)
variable (midpoint : Point → Point → Point) 
variables [midpoint_def : ∀ (x y : Point), midpoint x y = (x + y) / 2]

theorem midline_parallel_and_equal 
    (h1 : (B - A) = (B1 - A1))
    (h2 : (C - D) = (C1 - D1))
    (hM : midpoint A D = M)
    (hN : midpoint B C = N)
    (hM1 : midpoint A1 D1 = M1)
    (hN1 : midpoint B1 C1 = N1) :
  (N - M) = (N1 - M1) := 
  sorry

end midline_parallel_and_equal_l809_809143


namespace license_plate_count_correct_l809_809578

-- Define the number of choices for digits and letters
def num_digit_choices : ℕ := 10^3
def num_letter_block_choices : ℕ := 26^3
def num_position_choices : ℕ := 4

-- Compute the total number of distinct license plates
def total_license_plates : ℕ := num_position_choices * num_digit_choices * num_letter_block_choices

-- The proof statement
theorem license_plate_count_correct : total_license_plates = 70304000 := by
  -- This proof is left as an exercise
  sorry

end license_plate_count_correct_l809_809578


namespace propositions_correct_l809_809563

variables {α β : Type} [plane α] [plane β]
variable (a : line)

-- condition: line a is contained in plane α
noncomputable def line_in_plane (a : line) (α : plane) : Prop := sorry

theorem propositions_correct :
  (∀ (α β) [plane α] [plane β], (line_in_plane a α) → 
  ( (α ∥ β) → (a ∥ β) ) ) ∧
  (∃ (α β) [plane α] [plane β], (line_in_plane a α) → 
  ( ¬ (a ∥ β) → ¬ (α ∥ β) ) ) :=
sorry

end propositions_correct_l809_809563


namespace evaluate_expression_l809_809265

theorem evaluate_expression : (1023 * 1023) - (1022 * 1024) = 1 := by
  sorry

end evaluate_expression_l809_809265


namespace parallelogram_angles_l809_809791

noncomputable def parallelogram (A B C D : Point) : Prop :=
  parallelogram_prop A B C D

noncomputable def triangle (A B C : Point) : Prop :=
  triangle_prop A B C

noncomputable def intersection (A B C D : Point) : Point

noncomputable def inside (N : Point) (A B : Point) : Prop :=
  inside_prop N A B

noncomputable def angle (A B C : Point) : Angle

theorem parallelogram_angles (A B C D M N : Point) :
  parallelogram A B C D →
  M = intersection A C B D →
  inside N A B → 
  angle A N D = angle B N C →
  angle M N C = angle N D A ∧ angle M N D = angle N C B :=
by
  intros h_parallelogram h_intersection h_inside h_angles
  sorry

end parallelogram_angles_l809_809791


namespace exists_distinct_indices_divisible_l809_809622

theorem exists_distinct_indices_divisible (p : ℕ) (m : ℕ → ℕ) (σ : Perm (Fin p))
  (h_prime : Nat.Prime p) (h_odd : p % 2 = 1)
  (h_consecutive : ∀ i : Fin p, m (i + 1) = m i + 1) :
  ∃ (k l : Fin p), k ≠ l ∧ p ∣ (m k * m (σ k) - m l * m (σ l)) :=
by
  sorry

end exists_distinct_indices_divisible_l809_809622


namespace solve_trig_equation_l809_809457

noncomputable def sin_3x (x : Real) : Real := 3 * Real.sin x - 4 * (Real.sin x)^3

theorem solve_trig_equation (x : Real) (k : Int) :
  (Real.abs (Real.sin x) - sin_3x x) / (Real.cos x * Real.cos (2 * x)) = 2 * Real.sqrt 3 ↔
  ∃ k₁ k₂ : Int, x = (2 * Real.pi / 3) + (2 * k₁ * Real.pi) ∨ 
                 x = - (2 * Real.pi / 3) + (2 * k₁ * Real.pi) ∨ 
                 x = - (Real.pi / 6) + (2 * k₂ * Real.pi) := sorry

end solve_trig_equation_l809_809457


namespace knight_expected_moves_l809_809522

theorem knight_expected_moves :
  let m := 12
  let n := 5
  let gcd := Nat.gcd m n
  gcd = 1 →
  100 * m + n = 1205 :=
by
  intro gcd_eq_one
  have : 100 * 12 + 5 = 1205 := by norm_num
  exact this

end knight_expected_moves_l809_809522


namespace largest_base5_three_digit_to_base10_l809_809971

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l809_809971


namespace four_digit_numbers_sum_12_div_by_5_l809_809351

theorem four_digit_numbers_sum_12_div_by_5 : 
  (∃ n : ℕ, (n >= 1000 ∧ n < 10000) ∧ (∃ a b c d : ℕ, n = a * 1000 + b * 100 + c * 10 + d ∧ a + b + c + d = 12 ∧ d ∈ {0, 5}) ∧ (n % 5 = 0))
  = 127 := 
sorry

end four_digit_numbers_sum_12_div_by_5_l809_809351


namespace sum_of_coefficients_l809_809734

-- Define a polynomial
noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (3*x - 1)^7

-- Define the coefficients of the polynomial
variables {a_7 a_6 a_5 a_4 a_3 a_2 a_1 a_0 : ℝ}

-- Equivocate the expanded form with coefficients
axiom polynomial_coefficients :
  polynomial_expansion x = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

-- Prove that the sum of specific coefficients is correct
theorem sum_of_coefficients :
  a_7 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1 = 129 :=
sorry

end sum_of_coefficients_l809_809734


namespace bikeShopProfit_correct_l809_809027

noncomputable def bikeShopProfit : ℕ :=
  let tireIncome := 20 * 300
  let chainIncome := 75 * 50
  let bikeIncome := 300 * 8
  let retailIncome := 2000
  let totalIncome := tireIncome + chainIncome + bikeIncome + retailIncome 

  let tireCost := 5 * 300
  let chainCost := 25 * 50
  let bikeCost := 50 * 8
  let totalPartsCost := tireCost + chainCost + bikeCost

  let discount := if totalPartsCost ≥ 2500 then 0.10 * totalPartsCost else 0
  let finalPartsCost := totalPartsCost - discount

  let retailProfit := 2000 - 1200
  let totalProfitBeforeTaxes := totalIncome - finalPartsCost - retailProfit

  let taxes := 0.06 * totalIncome
  let profitAfterTaxes := totalProfitBeforeTaxes - taxes

  let fixedExpenses := 4000
  let finalProfit := profitAfterTaxes - fixedExpenses

  finalProfit

theorem bikeShopProfit_correct : bikeShopProfit = 8206 := by
  sorry

end bikeShopProfit_correct_l809_809027


namespace function_passes_through_point_l809_809070

-- Lean 4 Statement
theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧ (a^(x-1) + 4) = y :=
by
  use 1
  use 5
  sorry

end function_passes_through_point_l809_809070


namespace willie_final_stickers_l809_809165

-- Conditions
def willie_start_stickers : ℝ := 36.0
def emily_gives_willie : ℝ := 7.0

-- Theorem
theorem willie_final_stickers : willie_start_stickers + emily_gives_willie = 43.0 :=
by
  sorry

end willie_final_stickers_l809_809165


namespace inner_product_sum_is_neg_27_5_l809_809038

variables (u v w : ℝ^3) -- or ℝ^n if appropriate
variable h₀ : ∥u∥ = 5
variable h₁ : ∥v∥ = 3
variable h₂ : ∥w∥ = 7
variable h₃ : u + 2 • v + w = 0

theorem inner_product_sum_is_neg_27_5 :
  (u • v + u • w + v • w) = -27.5 :=
sorry

end inner_product_sum_is_neg_27_5_l809_809038


namespace four_digit_sum_divisible_by_5_l809_809348

theorem four_digit_sum_divisible_by_5 :
  ∃ (a b c d : ℕ), 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    1000 * a + 100 * b + 10 * c + d ≥ 1000 ∧
    a + b + c + d = 12 ∧ 
    d ∈ {0, 5} :=
by
  sorry

end four_digit_sum_divisible_by_5_l809_809348


namespace problem1_problem2_l809_809189

-- Problem 1
theorem problem1 (α β : ℝ) (h1 : tan (α + β) = 2 / 5) (h2 : tan (β - π/4) = 1 / 4) :
  (cos α + sin α) / (cos α - sin α) = 3 / 22 :=
sorry

-- Problem 2
theorem problem2 (α β : ℝ) (acute_α : 0 < α ∧ α < π / 2) (acute_β : 0 < β ∧ β < π / 2)
  (h1 : cos (α + β) = sqrt 5 / 5) (h2 : sin (α - β) = sqrt 10 / 10) :
  β = π / 8 :=
sorry

end problem1_problem2_l809_809189


namespace simplify_expression_eval_at_2_l809_809420

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (x^2 + a)^2 / ((a - b) * (a - c)) + (x^2 + b)^2 / ((b - a) * (b - c)) + (x^2 + c)^2 / ((c - a) * (c - b)) =
    x^4 + x^2 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

theorem eval_at_2 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (2^2 + a)^2 / ((a - b) * (a - c)) + (2^2 + b)^2 / ((b - a) * (b - c)) + (2^2 + c)^2 / ((c - a) * (c - b)) =
    16 + 4 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

end simplify_expression_eval_at_2_l809_809420


namespace minimum_value_expression_l809_809291

noncomputable def f (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2 + 26*a + 86*b + 2018)

theorem minimum_value_expression : 
  ∃ (a b : ℝ), f(a, b) + f(a, -b) + f(-a, b) + f(-a, -b) = 4 * real.sqrt 2018 :=
begin
  sorry
end

end minimum_value_expression_l809_809291


namespace intersection_range_l809_809123

noncomputable def function_f (x: ℝ) : ℝ := abs (x^2 - 4 * x + 3)

theorem intersection_range (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ function_f x1 = b ∧ function_f x2 = b ∧ function_f x3 = b) ↔ (0 < b ∧ b ≤ 1) := 
sorry

end intersection_range_l809_809123


namespace quadratic_root_p_q_sum_l809_809188

theorem quadratic_root_p_q_sum (p q : ℝ) (h_root : 2 * (-3 + 2*complex.I)^2 + p * (-3 + 2*complex.I) + q = 0) : 
  p + q = 38 :=
sorry

end quadratic_root_p_q_sum_l809_809188


namespace grasshopper_flea_adjacency_l809_809860

-- Define the types of cells
inductive CellColor
| Red
| White

-- Define the infinite grid as a function from ℤ × ℤ to CellColor
def InfiniteGrid : Type := ℤ × ℤ → CellColor

-- Define the positions of the grasshopper and the flea
variables (g_start f_start : ℤ × ℤ)

-- The conditions for the grid and movement rules
axiom grid_conditions (grid : InfiniteGrid) :
  ∃ g_pos f_pos : ℤ × ℤ, 
  (g_pos = g_start ∧ f_pos = f_start) ∧
  (∀ x y : ℤ × ℤ, grid x = CellColor.Red ∨ grid x = CellColor.White) ∧
  (∀ x y : ℤ × ℤ, grid y = CellColor.Red ∨ grid y = CellColor.White)

-- Define the movement conditions for grasshopper and flea
axiom grasshopper_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.Red ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

axiom flea_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.White ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

-- The main theorem statement
theorem grasshopper_flea_adjacency (grid : InfiniteGrid)
    (g_start f_start : ℤ × ℤ) :
    ∃ pos1 pos2 pos3 : ℤ × ℤ,
    (pos1 = g_start ∨ pos1 = f_start) ∧ 
    (pos2 = g_start ∨ pos2 = f_start) ∧ 
    (abs (pos3.1 - g_start.1) + abs (pos3.2 - g_start.2) ≤ 1 ∧ 
    abs (pos3.1 - f_start.1) + abs (pos3.2 - f_start.2) ≤ 1) :=
sorry

end grasshopper_flea_adjacency_l809_809860


namespace rate_of_interest_l809_809988

theorem rate_of_interest (R : ℝ) (h : 5000 * 2 * R / 100 + 3000 * 4 * R / 100 = 2200) : R = 10 := by
  sorry

end rate_of_interest_l809_809988


namespace shared_foci_ellipse_hyperbola_l809_809206

theorem shared_foci_ellipse_hyperbola (a b : ℝ) (P F1 F2 : ℝ × ℝ)
  (hF1 : F1 = (-5, 0)) (hF2 : F2 = (5, 0)) (hP : P = (4, 3)) :
  (∃ a, a^2 = 40 ∧ ∃ b, b^2 = 16 ∧ (∀ x y, 
    (x, y) ∈ ellipse_eqn a (P, F1, F2) ↔ (x, y) ∈ hyperbola_eqn b (P, F1, F2))) :=
  sorry

def ellipse_eqn (a : ℝ) (centers : ℝ × ℝ × ℝ × ℝ) (c : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (F1, F2) := centers in 
  let b2 := a^2 - 25 in
  x^2 / a^2 + y^2 / b2 = 1

def hyperbola_eqn (b : ℝ) (centers : ℝ × ℝ × ℝ × ℝ) (c : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (F1, F2) := centers in
  let b2 := 25 - b^2 in
  x^2 / b^2 - y^2 / b2 = 1

end shared_foci_ellipse_hyperbola_l809_809206


namespace dot_product_computation_l809_809617

-- Define scalar multiples of the vectors as given in the conditions.
def v1 : ℝ × ℝ := (3 * -2, 3 * 0)
def v2 : ℝ × ℝ := (4 * 3, 4 * -5)

-- Define the dot product function.
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Formalize the problem statement.
theorem dot_product_computation :
  dot_product v1 v2 = -72 :=
by 
  -- Definitions based on conditions
  have v1_eq : v1 = (-6, 0) := by simp [v1]
  have v2_eq : v2 = (12, -20) := by simp [v2]
  -- Using the definitions and computing the dot product
  rw [v1_eq, v2_eq]
  done

end dot_product_computation_l809_809617


namespace magicStack_cardCount_l809_809108

-- Define the conditions and question based on a)
def isMagicStack (n : ℕ) : Prop :=
  let totalCards := 2 * n
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range totalCards) ∧
    (∀ x ∈ A, x < n) ∧ (∀ x ∈ B, x ≥ n) ∧
    (∀ i ∈ A, i % 2 = 1) ∧ (∀ j ∈ B, j % 2 = 0) ∧
    (151 ∈ A) ∧
    ∃ (newStack : Finset ℕ), (newStack = A ∪ B) ∧
    (∀ k ∈ newStack, k ∈ A ∨ k ∈ B) ∧
    (151 = 151)

-- The theorem that states the number of cards, when card 151 retains its position, is 452.
theorem magicStack_cardCount :
  isMagicStack 226 → 2 * 226 = 452 :=
by
  sorry

end magicStack_cardCount_l809_809108


namespace value_of_e_over_f_l809_809317

variables {a b c d e f : ℝ}

-- Conditions
def condition1 := a / b = 1 / 3
def condition2 := b / c = 2
def condition3 := c / d = 1 / 2
def condition4 := d / e = 3
def condition5 := a * b * c / (d * e * f) = 0.1875

-- Theorem to prove the question
theorem value_of_e_over_f
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  (h5 : condition5) :
  e / f = 0.125 :=
sorry

end value_of_e_over_f_l809_809317


namespace length_of_major_axis_l809_809311

theorem length_of_major_axis
  (a b : ℝ) (h_pos : a > b) (h_nonneg : b > 0)
  (F1 F2 : ℝ × ℝ) (h_foci : F1 = (-1, 0) ∧ F2 = (1, 0))
  (h_ellipse : ∀ x y, (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 → (x = 0 ∨ x ≠ 0))
  (h_parabola : ∀ x y, y ^ 2 = 4 * x → x = 0 ∨ y ≠ 0)
  (P : ℝ × ℝ) (h_P : ∃ x y, P = (x, y) ∧ (y ^ 2 = 4 * x ∧ ((x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1)) ∧ x > 0 ∧ y > 0)
  (h_tangent : ∀ k : ℝ, P = (1, 2) ∧ k = 1 → y = k * (x + 1)) :
  length_of_major_axis = 2 * sqrt 2 + 2 := sorry

end length_of_major_axis_l809_809311


namespace gcd_polynomial_l809_809703

theorem gcd_polynomial (b : ℤ) (h : 570 ∣ b) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 :=
sorry

end gcd_polynomial_l809_809703


namespace eccentricity_of_ellipse_l809_809488

open Real

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) (angle_MPN : ∠ MPN = 120) : ℝ :=
  let e := sqrt (1 - (b^2) / (a^2))
  in e

theorem eccentricity_of_ellipse (a b : ℝ) (h: a > b ∧ b > 0) (h_angle: MPN = 120) : 
  ellipse_eccentricity a b h h_angle = (sqrt 6) / 3 :=
by
  sorry

end eccentricity_of_ellipse_l809_809488


namespace distinct_three_digit_numbers_l809_809136

theorem distinct_three_digit_numbers :
  let cards := [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
      create_numbers :=
        (list_permutations : (List (Nat × Nat))) →
        List.map (λ (c ∷ cs), (10 * (c.2) + cs.head!.2 * 10 + cs.tail!.head!.2)) list_permutations,
      valid_numbers := List.size (List.toFinset (create_numbers (List.permutations cards))),
  valid_numbers = 96 :=
by
  sorry

end distinct_three_digit_numbers_l809_809136


namespace largest_base_5_three_digit_in_base_10_l809_809945

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l809_809945


namespace range_of_m_integer_value_of_m_l809_809342

open Real

theorem range_of_m (m x y : ℝ) (h₁ : x + y = -m - 7) (h₂ : x - y = 3m + 1) (hx : x ≤ 0) (hy : y < 0) :
  -2 < m ∧ m ≤ 3 :=
by
  sorry

theorem integer_value_of_m (m x : ℝ) (hx1 : x > 1) (hx2 : 2m * x + x < 2m + 1) (range_m : -2 < m ∧ m ≤ 3) :
  m = -1 :=
by
  sorry

end range_of_m_integer_value_of_m_l809_809342


namespace other_diagonal_of_rhombus_l809_809470

noncomputable def calculate_other_diagonal (area d1 : ℝ) : ℝ :=
  (area * 2) / d1

theorem other_diagonal_of_rhombus {a1 a2 : ℝ} (area_eq : a1 = 21.46) (d1_eq : a2 = 7.4) : calculate_other_diagonal a1 a2 = 5.8 :=
by
  rw [area_eq, d1_eq]
  norm_num
  -- The next step would involve proving that (21.46 * 2) / 7.4 = 5.8 in a formal proof.
  sorry

end other_diagonal_of_rhombus_l809_809470


namespace area_removed_to_nearest_integer_is_4_l809_809295

def side_length_square := 5
def side_length_triangle (x : ℝ) := x

theorem area_removed_to_nearest_integer_is_4 (x : ℝ) 
  (h : x * real.sqrt 2 = 5 - 2 * x) : 
  let area_removed := 75 - 50 * real.sqrt 2 in
  | area_removed - 4 | < 1 :=
sorry

end area_removed_to_nearest_integer_is_4_l809_809295


namespace probability_odd_result_l809_809171

-- The conditions as definitions
def initial_display : ℕ := 0
def has_operations (ops : List (ℕ → ℕ → ℕ)) : Prop := ops.contains (+)
def last_key_press_idx (ops : List (ℕ → ℕ → ℕ)) : Nat := ops.length - 1

-- The main theorem to prove
theorem probability_odd_result (ops : List (ℕ → ℕ → ℕ))
  (h_initial : initial_display = 0)
  (h_operations : has_operations ops)
  (h_last_press : ∀ i < last_key_press_idx ops, ops[i] = (+)) :
  (∃ n : ℕ, ops.foldl (λ acc op, op acc n) initial_display % 2 = 1) →
  ∀ (p : ℝ), p = 0.5 :=
sorry

end probability_odd_result_l809_809171


namespace term_is_18_minimum_value_l809_809779

-- Define the sequence a_n
def a_n (n : ℕ) : ℤ := n^2 - 5 * n + 4

-- Prove that a_n = 18 implies n = 7
theorem term_is_18 (n : ℕ) (h : a_n n = 18) : n = 7 := 
by 
  sorry

-- Prove that the minimum value of a_n is -2 and it occurs at n = 2 or n = 3
theorem minimum_value (n : ℕ) : n = 2 ∨ n = 3 ∧ a_n n = -2 :=
by 
  sorry

end term_is_18_minimum_value_l809_809779


namespace sqrt_product_simplified_l809_809849

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l809_809849


namespace find_a_l809_809329

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 1 then x^2 + 1 else 2^x + a * x

theorem find_a (a : ℝ) (h : f a (f a 1) = 4 * a) : a = 2 := by
  sorry

end find_a_l809_809329


namespace range_of_a_l809_809098

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|4 * x - 3| ≤ 1)) → 
  (∀ x : ℝ, (x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
begin
  sorry
end

end range_of_a_l809_809098


namespace solve_system_l809_809458

theorem solve_system (x y : ℝ) :
  (2 * y = (abs (2 * x + 3)) - (abs (2 * x - 3))) ∧ 
  (4 * x = (abs (y + 2)) - (abs (y - 2))) → 
  (-1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x) := 
by
  sorry

end solve_system_l809_809458


namespace bead_necklace_no_consecutive_same_color_l809_809146

theorem bead_necklace_no_consecutive_same_color (n k : ℕ) (colors : Fin n → ℕ) :
  (∀ color, ∃ m, (∑ i in Finset.univ.filter (λ j, colors j = color), 1) ≤ m) →
  k ≤ n / 2 → 
  ∃ arrangement : Fin n → ℕ, (∀ i, arrangement i ≠ arrangement ((i + 1) % n)) :=
by
  intros h_sum h_ratio
  sorry

end bead_necklace_no_consecutive_same_color_l809_809146


namespace height_difference_l809_809088

def pine_tree_height : ℚ := 12 + 1 / 4
def maple_tree_height : ℚ := 18 + 1 / 2

theorem height_difference :
  maple_tree_height - pine_tree_height = 6 + 1 / 4 :=
by sorry

end height_difference_l809_809088


namespace total_students_in_class_l809_809372

theorem total_students_in_class (female_students : ℕ) (male_students : ℕ) (total_students : ℕ) 
  (h1 : female_students = 13) 
  (h2 : male_students = 3 * female_students) 
  (h3 : total_students = female_students + male_students) : 
    total_students = 52 := 
by
  sorry

end total_students_in_class_l809_809372


namespace length_of_AE_l809_809814

-- Define points A, B, D, E, M in a 2D plane (or use an abstract representation if needed)
variables (A B D E M : ℝ)
-- Define lengths of the segments
variables (x : ℝ)

-- Conditions:
-- B and D divide AE into four equal segments
def quadrisection : Prop := (A < B) ∧ (B < D) ∧ (D < E) ∧ (B - A = x) ∧ (D - B = x) ∧ (E - D = x) ∧ (M = A + E) / 2

-- M is the centroid of triangle ADE
-- MD = 10
def centroid_condition : Prop := @dist ℝ _ (⟨D.x, D.y⟩) (⟨M.x, M.y⟩) = 10

-- Value of AE
def length_AE : ℝ := 4 * x

-- Proof statement:
theorem length_of_AE (h1 : quadrisection A B D E x) (h2 : centroid_condition A D M) : length_AE A E x = 15 :=
sorry

end length_of_AE_l809_809814


namespace N_vector_3_eq_result_vector_l809_809037

noncomputable def matrix_N : Matrix (Fin 2) (Fin 2) ℝ :=
-- The matrix N is defined such that:
-- N * (vector 3 -2) = (vector 4 1)
-- N * (vector -2 3) = (vector 1 2)
sorry

def vector_1 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 3 | ⟨1,_⟩ => -2
def vector_2 : Fin 2 → ℝ := fun | ⟨0,_⟩ => -2 | ⟨1,_⟩ => 3
def vector_3 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 7 | ⟨1,_⟩ => 0
def result_vector : Fin 2 → ℝ := fun | ⟨0,_⟩ => 14 | ⟨1,_⟩ => 7

theorem N_vector_3_eq_result_vector :
  matrix_N.mulVec vector_3 = result_vector := by
  -- Given conditions:
  -- matrix_N.mulVec vector_1 = vector_4
  -- and matrix_N.mulVec vector_2 = vector_5
  sorry

end N_vector_3_eq_result_vector_l809_809037


namespace magazine_purchase_ways_l809_809567

theorem magazine_purchase_ways :
  let M := 8
  let N := 3
  let C := Nat.choose
  ∑ (C(N, 2) * C(M, 4)) + C(M, 5) = 266
by
  sorry

end magazine_purchase_ways_l809_809567


namespace problem_quadratic_inequality_l809_809367

theorem problem_quadratic_inequality
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : 0 < a)
  (h2 : a ≤ 4/9)
  (h3 : b = -a)
  (h4 : c = -2*a + 1)
  (h5 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 1) :
  3*a + 2*b + c ≠ 1/3 ∧ 3*a + 2*b + c ≠ 5/4 :=
by
  sorry

end problem_quadratic_inequality_l809_809367


namespace radius_of_B_l809_809242

theorem radius_of_B {A B C D : Type} (r_A : ℝ) (r_D : ℝ) (r_B : ℝ) (r_C : ℝ)
  (center_A : A) (center_B : B) (center_C : C) (center_D : D)
  (h_cong_BC : r_B = r_C)
  (h_A_D : r_D = 2 * r_A)
  (h_r_A : r_A = 2)
  (h_tangent_A_D : (dist center_A center_D) = r_A) :
  r_B = 32/25 := sorry

end radius_of_B_l809_809242


namespace sparrow_grains_l809_809436

theorem sparrow_grains (x : ℤ) : 9 * x < 1001 ∧ 10 * x > 1100 → x = 111 :=
by
  sorry

end sparrow_grains_l809_809436


namespace reciprocal_roots_l809_809625

theorem reciprocal_roots (p q : ℝ) (h1 : q ≠ 0) :
  let x1 x2 := roots of the equation x^2 + p*x + q = 0,
  (x1 + x2 = -p) ∧ (x1 * x2 = q) → 
  (roots of the equation q*x^2 + p*x + 1 = 0 are the reciprocals of x1 and x2) :=
by { sorry }

end reciprocal_roots_l809_809625


namespace factor_tree_example_l809_809925

theorem factor_tree_example : 
  let F := 2 * 5,
  let G := 3 * 7,
  let Y := 7 * F,
  let Z := 11 * G,
  let X := Y * Z
  in X = 16170 := by
  sorry

end factor_tree_example_l809_809925


namespace sum_of_coefficients_l809_809735

-- Define a polynomial
noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (3*x - 1)^7

-- Define the coefficients of the polynomial
variables {a_7 a_6 a_5 a_4 a_3 a_2 a_1 a_0 : ℝ}

-- Equivocate the expanded form with coefficients
axiom polynomial_coefficients :
  polynomial_expansion x = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

-- Prove that the sum of specific coefficients is correct
theorem sum_of_coefficients :
  a_7 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1 = 129 :=
sorry

end sum_of_coefficients_l809_809735


namespace problem_statement_l809_809816

-- Letting Lean know that we will use the real numbers
noncomputable theory
open Real

-- Definition of the function f
def f (x a : ℝ) := abs (x - 4) + abs (x - a)

-- Lean statement for the mathematically equivalent proof problem
theorem problem_statement (a : ℝ) (h : 1 < a) :
  (∀ x : ℝ, f x a ≥ 3) → a = 7 ∧ ∀ x : ℝ, f x 7 ≤ 5 → 3 ≤ x ∧ x ≤ 8 :=
by {
  sorry
}

end problem_statement_l809_809816


namespace probability_y_eq_2x_l809_809524

/-- Two fair cubic dice each have six faces labeled with the numbers 1, 2, 3, 4, 5, and 6. 
Rolling these dice sequentially, find the probability that the number on the top face 
of the second die (y) is twice the number on the top face of the first die (x). --/
noncomputable def dice_probability : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem probability_y_eq_2x : dice_probability = 1 / 12 :=
  by sorry

end probability_y_eq_2x_l809_809524


namespace find_remainder_l809_809665

open Polynomial

noncomputable def remainder_poly : Polynomial ℤ := 149 * X^2 - 447 * X + 301
noncomputable def dividend_poly : Polynomial ℤ := X^6 - X^5 - X^4 + X^3 + 2 * X^2 + X
noncomputable def divisor_poly : Polynomial ℤ := (X^2 - 9) * (X - 1)

theorem find_remainder :
  let q := dividend_poly /ₚ divisor_poly,
      r := dividend_poly %ₚ divisor_poly in
  r = remainder_poly := by
  sorry

end find_remainder_l809_809665


namespace bobby_truck_gasoline_consumption_rate_l809_809233

theorem bobby_truck_gasoline_consumption_rate :
  (let total_miles := 7 * 2 + 3 * 2 + 10 + 5 + 8 + 8 in
   let total_gallons := 15 - 2 in
   total_miles / total_gallons = 3.9231) :=
by
  let total_miles := 7 * 2 + 3 * 2 + 10 + 5 + 8 + 8
  let total_gallons := 15 - 2
  have h : total_miles = 51 := by norm_num
  have h₁ : total_gallons = 13 := by norm_num
  rw [h, h₁]
  norm_num
  sorry

end bobby_truck_gasoline_consumption_rate_l809_809233


namespace num_sets_B_l809_809421

open Set

noncomputable def A : Set (Set ℕ) := { {0, 1} }

theorem num_sets_B (c : ℕ) (hc : c ∉ {0, 1}) :
  ∃ B : Set (Set ℕ), {0, 1} ∪ B = {0, 1, c} ∧ B.card = 4 := by
  sorry

end num_sets_B_l809_809421


namespace decimal_representation_of_fraction_infinite_digits_l809_809731

theorem decimal_representation_of_fraction_infinite_digits :
  ∀ (n : ℕ), n = 1 / (2^3 * 5^4 * 3^2) → decimal_places_to_right_of_point n = ∞ :=
by
  sorry

end decimal_representation_of_fraction_infinite_digits_l809_809731


namespace largest_base5_three_digit_to_base10_l809_809962

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l809_809962


namespace problem_solution_l809_809619

theorem problem_solution : 
  (∑ k in Finset.range 20, Real.logb (5^k) (3^(k^2)))
  * (∑ k in Finset.range 100, Real.logb 9 (25^k)) = 21000 := 
sorry

end problem_solution_l809_809619


namespace oranges_less_per_student_l809_809200

-- Define the given conditions
def total_students := 25
def total_oranges := 240
def bad_oranges := 65
def oranges_per_student_without_bad := total_oranges / total_students
def good_oranges := total_oranges - bad_oranges
def oranges_per_student_with_bad_removed := good_oranges / total_students

-- Theorem to be proven
theorem oranges_less_per_student : (oranges_per_student_without_bad - oranges_per_student_with_bad_removed) = 2.6 := by
  sorry

end oranges_less_per_student_l809_809200


namespace parallelogram_perimeter_eq_l809_809369

noncomputable def perimeter_parallelogram (A B C D E F : ℝ) (AB AC BC DE EF : ℝ)
  (h1 : AB = AC) (h2 : AB = 24) (h3 : AC = 24) (h4 : BC = 20) 
  (h5 : DE = D - E) (h6 : EF = E - F) (h7 : DE.parallel AC) (h8 : EF.parallel AB) 
  (h9 : A - B = D - E) : ℝ :=
AD + DE + EF + AF

theorem parallelogram_perimeter_eq :
  ∃ (A B C D E F : ℝ) (h1 : AB = AC) (h2 : AB = 24) (h3 : AC = 24) (h4 : BC = 20) 
    (h7 : DE.parallel AC) (h8 : EF.parallel AB),
  perimeter_parallelogram A B C D E F = 48 :=
by sorry

end parallelogram_perimeter_eq_l809_809369


namespace right_triangle_median_slope_l809_809253

theorem right_triangle_median_slope (a b h k : ℝ) (n : ℝ) :
  (∃ (a b h k : ℝ), 
    let mid1 := (a, b + k),
        mid2 := (a - h, b) in
    (2 * k) = 2 * h ∧ 
    (- (2 * k / h)) = n ∧ 
    ((y = 2 * x + 1) ∧ (y = n * x - 1))) → n = -4 :=
begin
  sorry
end

end right_triangle_median_slope_l809_809253


namespace a_finishes_job_in_60_days_l809_809196

theorem a_finishes_job_in_60_days (A B : ℝ)
  (h1 : A + B = 1 / 30)
  (h2 : 20 * (A + B) = 2 / 3)
  (h3 : 20 * A = 1 / 3) :
  1 / A = 60 :=
by sorry

end a_finishes_job_in_60_days_l809_809196


namespace no_positive_integers_ap_bq_prime_for_large_primes_l809_809444

theorem no_positive_integers_ap_bq_prime_for_large_primes :
  ¬(∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ ∀ (p q : ℕ), prime p → prime q → 1000 < p → 1000 < q → p ≠ q → prime (a * p + b * q)) :=
by
  sorry

end no_positive_integers_ap_bq_prime_for_large_primes_l809_809444


namespace smallest_int_ending_in_9_divisible_by_11_l809_809158

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end smallest_int_ending_in_9_divisible_by_11_l809_809158


namespace Jake_watched_hours_on_Friday_l809_809018

theorem Jake_watched_hours_on_Friday :
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  total_show_hours - total_hours_before_Friday = 19 :=
by
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  sorry

end Jake_watched_hours_on_Friday_l809_809018


namespace determinant_of_matrixA_l809_809266

variable (x y z : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![2, x, y + z],
  ![2, x + y, y],
  ![2, x, x + z]
]

theorem determinant_of_matrixA : matrixA.det = 2 * x^2 + 2 * x * z + 2 * y * z - 2 * y^2 := by
  sorry

end determinant_of_matrixA_l809_809266


namespace probability_no_two_adjacent_stand_l809_809463

-- Define b_n based on the recursive relation provided in the solution
def b : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| n + 1 := b n + b (n - 1)

-- Total number of outcomes
def total_outcomes : ℕ := 2 ^ 10

-- b_10 as calculated in the solution is 123
def b_10 : ℕ := b 10

-- Probability as per calculated solution
def desired_probability : ℚ := b_10 / total_outcomes

-- The main theorem stating the required probability
theorem probability_no_two_adjacent_stand :
  desired_probability = 123 / 1024 :=
by
  sorry

end probability_no_two_adjacent_stand_l809_809463


namespace correct_option_C_l809_809408

variable {α β : Type} [Plane α] [Plane β]
variable {m n : Type} [Line m] [Line n]

-- Define perpendicular and parallel relationships between lines and planes.
def perpendicular (l1 l2 : Type) [Line l1] [Line l2] : Prop := sorry
def parallel (l1 l2 : Type) [Line l1] [Line l2] : Prop := sorry
def line_in_plane (l : Type) [Line l] (p : Type) [Plane p] : Prop := sorry

theorem correct_option_C (h1 : perpendicular m β) (h2 : parallel n β) : perpendicular m n :=
sorry

end correct_option_C_l809_809408


namespace sheila_paintings_l809_809605

theorem sheila_paintings (a b : ℕ) (h1 : a = 9) (h2 : b = 9) : a + b = 18 :=
by
  sorry

end sheila_paintings_l809_809605


namespace percentage_of_boys_toast_marshmallows_l809_809607

theorem percentage_of_boys_toast_marshmallows
  (total_campers : ℕ) (boys_fraction girls_fraction : ℚ)
  (girls_toasting_percentage : ℚ) (total_marshmallows : ℕ) :
  total_campers = 96 ∧
  boys_fraction = 2/3 ∧
  girls_fraction = 1/3 ∧
  girls_toasting_percentage = 0.75 ∧
  total_marshmallows = 56 →
  let boys := boys_fraction * total_campers,
      girls := girls_fraction * total_campers,
      girls_toasting := girls_toasting_percentage * girls,
      boys_toasting_needed := total_marshmallows - girls_toasting,
      boys_toasting_percentage := (boys_toasting_needed * 100) / boys in
  boys_toasting_percentage = 50 :=
by
  intros h
  rcases h with ⟨h_total_campers, h_boys_fraction, h_girls_fraction, h_girls_toasting_percentage, h_total_marshmallows⟩
  sorry

end percentage_of_boys_toast_marshmallows_l809_809607


namespace popsicle_sticks_each_boy_brought_l809_809462

theorem popsicle_sticks_each_boy_brought
  (B : ℕ)
  (total_sticks_girls : ℕ := 12 * 12)
  (total_sticks_boys : ℕ := total_sticks_girls + 6)
  (boys : ℕ := 10)
  (girls : ℕ := 12)
  (sticks_each_girl : ℕ := 12) :
  total_sticks_boys / boys = 15 :=
by
  unfold total_sticks_girls total_sticks_boys boys girls sticks_each_girl
  sorry

end popsicle_sticks_each_boy_brought_l809_809462


namespace slope_of_line_l809_809892

theorem slope_of_line : (tan (5 * Real.pi / 6) = - Real.sqrt 3 / 3) := by
  sorry

end slope_of_line_l809_809892


namespace part_two_l809_809332

noncomputable def f (x : ℝ) : ℝ := x - 1 / x - Real.log x

theorem part_two {x1 x2 : ℝ} (h1 : x1 ≠ x2) (h2 : f' x1 = f' x2) :
  f(x1) + f(x2) > 3 - 2 * Real.log 2 := 
sorry

end part_two_l809_809332


namespace P_not_sqrt_30_l809_809498

noncomputable def P (x a : ℝ) : ℝ := x^3 + a * x + 1

theorem P_not_sqrt_30 (a : ℝ) (h1 : ∃! x ∈ Ico (-2 : ℝ) 0, P x a = 0)
    (h2 : ∃! x ∈ Ioc (0 : ℝ) 1, P x a = 0) : P 2 a ≠ Real.sqrt 30 :=
by
  sorry

end P_not_sqrt_30_l809_809498


namespace repeat_12_remainder_99_l809_809553

theorem repeat_12_remainder_99 (n : ℕ) (h : n = 150) : 
  let number := 12 * (10 ^ (2 * n) - 1) / 99 in
  number % 99 = 18 :=
by
  sorry

end repeat_12_remainder_99_l809_809553


namespace last_two_digits_of_7_pow_2017_l809_809071

theorem last_two_digits_of_7_pow_2017 :
  (7 ^ 2017) % 100 = 7 :=
sorry

end last_two_digits_of_7_pow_2017_l809_809071


namespace simplify_sqrt_product_l809_809842

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l809_809842


namespace universality_eq_nonatomic_measure_l809_809454

noncomputable theory

variables {Ω : Type*} {ℱ : measurable_space Ω} {P : measure Ω}

-- Define the property of universality for \((Ω, ℱ, P)\).
def universal (P : measure Ω) : Prop :=
  ∀ u : Ω → set.Icc (0: ℝ) 1, measure_univ P P

-- Define nonatomic measure for \(P\)
def nonatomic_measure (P : measure Ω) : Prop :=
  ∀ A ∈ ℱ, P(A) > 0 → ∃ B ⊆ A, measurable_set B ∧ P(B) > 0 ∧ P(B) < P(A)

-- Theorem: Universality property is equivalent to being a nonatomic measure.
theorem universality_eq_nonatomic_measure :
  universal P ↔ nonatomic_measure P :=
sorry

end universality_eq_nonatomic_measure_l809_809454


namespace stewarts_theorem_degenerate_quad_l809_809145

theorem stewarts_theorem_degenerate_quad (a b c d e f : ℝ) 
  (h1 : e^2 * f^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (180 : ℝ)) :
  e^2 * f^2 = a^2 * c^2 + b^2 * d^2 + 2 * a * b * c * d :=
by
  have h_cos : Real.cos 180 = -1 := by sorry -- Known fact
  rw [h_cos] at h1
  exact h1

end stewarts_theorem_degenerate_quad_l809_809145


namespace find_xy_yz_xz_l809_809995

noncomputable def xy_yz_xz (x y z : ℝ) : ℝ := x * y + y * z + x * z

theorem find_xy_yz_xz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 48) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + x * z + x^2 = 64) :
  xy_yz_xz x y z = 32 :=
sorry

end find_xy_yz_xz_l809_809995


namespace total_dog_legs_l809_809920

theorem total_dog_legs (total_animals : ℕ) (frac_cats : ℚ) (legs_per_dog : ℕ)
  (h1 : total_animals = 300)
  (h2 : frac_cats = 2 / 3)
  (h3 : legs_per_dog = 4) :
  ∃ (num_cats num_dogs : ℕ), 
    (num_cats = (frac_cats * total_animals).to_nat) ∧
    (num_dogs = total_animals - num_cats) ∧
    (num_dogs * legs_per_dog = 400) := 
by
  sorry

end total_dog_legs_l809_809920


namespace eccentricity_of_conic_section_l809_809712

-- Definitions as per the problem statement

-- Given condition: 4, m, 9 form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Defining the conic section and its respective eccentricities
def ellipse_eccentricity (m : ℝ) : ℝ := 
  if h : m = 6 then real.sqrt(1 - (1 / 6))
  else if h : m = -6 then real.sqrt(1 + 6)
  else 0

-- The mathematically equivalent proof problem
theorem eccentricity_of_conic_section (m : ℝ) (h_geom : is_geometric_sequence 4 m 9) :
  ellipse_eccentricity m = real.sqrt 7 ∨ ellipse_eccentricity m = real.sqrt 30 / 6 :=
by
  sorry


end eccentricity_of_conic_section_l809_809712


namespace arithmetic_geometric_sequence_problem_l809_809310

noncomputable def a (n : ℕ) : ℝ := 2^(n-1)

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)

noncomputable def b (n : ℕ) : ℝ := (5/2) * (Real.log (a n) / Real.log 2)

def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem arithmetic_geometric_sequence_problem
  (h₁ : a 1 + a 3 = 5)
  (h₂ : S 4 = 15) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, T n = 5 * n * (n - 1) / 4) :=
by
  sorry

end arithmetic_geometric_sequence_problem_l809_809310


namespace ratio_probabilities_l809_809262

-- Definitions for the conditions
def num_ways_to_distribute_balls (total_balls bins : ℕ) :=
  (total_balls.factorial) / (List.product (List.range (bins + 1)).map factorial)

def set_C := (binom 4 2) * (binom 25 6) * (binom 19 6) * (binom 13 5)
def set_D := 4 * 25 * (binom 24 6) * (binom 18 6) * (binom 12 6)

-- The main statement to prove
theorem ratio_probabilities (r s : ℝ) (X : ℝ) :
  r = set_C / num_ways_to_distribute_balls 25 4 → 
  s = set_D / num_ways_to_distribute_balls 25 4 → 
  r / s = X :=
by 
  sorry

end ratio_probabilities_l809_809262


namespace jake_watched_friday_l809_809016

theorem jake_watched_friday
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (thursday_hours : ℕ)
  (total_hours : ℕ)
  (day_hours : ℕ := 24) :
  monday_hours = (day_hours / 2) →
  tuesday_hours = 4 →
  wednesday_hours = (day_hours / 4) →
  thursday_hours = ((monday_hours + tuesday_hours + wednesday_hours) / 2) →
  total_hours = 52 →
  (total_hours - (monday_hours + tuesday_hours + wednesday_hours + thursday_hours)) = 19 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jake_watched_friday_l809_809016


namespace sum_not_divisible_by_5_l809_809091

theorem sum_not_divisible_by_5 (a : ℤ) : 
  (1^a + 2^a + 3^a + 4^a + 5^a + 6^a + 7^a + 8^a) % 5 ≠ 0 := 
sorry

end sum_not_divisible_by_5_l809_809091


namespace angle_between_p_q_l809_809797

noncomputable def angle_between_vectors
  (p q : ℝ → ℝ) (unit_p : ∥p∥ = 1) (unit_q : ∥q∥ = 1)
  (orthogonal : (3 • p - q) • (2 • p + 5 • q) = 0) : ℝ :=
  real.arccos (-(1 / 13))

theorem angle_between_p_q (p q : ℝ → ℝ)
  (unit_p : ∥p∥ = 1) (unit_q : ∥q∥ = 1)
  (orthogonal : (3 • p - q) • (2 • p + 5 • q) = 0) :
  angle_between_vectors p q unit_p unit_q orthogonal = real.arccos (-(1 / 13)) :=
sorry

end angle_between_p_q_l809_809797


namespace sequence_sum_less_than_4_l809_809623

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 / 2 ∧ ∀ k ≥ 1, a (k + 2) = a k + 0.5 * a (k + 1) + 1 / (4 * a k * a (k + 1))

theorem sequence_sum_less_than_4 : 
  ∃ a : ℕ → ℝ, sequence a → (∑ k in finset.range 98, 1 / (a k * a (k + 2))) < 4 :=
sorry

end sequence_sum_less_than_4_l809_809623


namespace imaginary_part_of_complex_expr_is_neg2_l809_809487

-- Define the two key complex numbers
def z1 : ℂ := 1 + I
def z2 : ℂ := 1 - I

-- Define the complex number in question
def complex_expr := (z1^4) / z2

-- Define the target imaginary part
def imag_part (z : ℂ) : ℝ := z.im

-- Assert that the imaginary part of the complex_expr is -2
theorem imaginary_part_of_complex_expr_is_neg2 : imag_part complex_expr = -2 := by
  sorry

end imaginary_part_of_complex_expr_is_neg2_l809_809487


namespace integral_equals_zero_l809_809638

theorem integral_equals_zero :
  ∫ x in (Real.pi / 4) .. (9 * Real.pi / 4), (sqrt 2) * cos (2 * x + Real.pi / 4) = 0 := by
sor

end integral_equals_zero_l809_809638


namespace direction_and_magnitude_of_maximal_growth_l809_809278

def function_z (x y : ℝ) : ℝ := 3 * x^2 - 2 * y^2

def point_M : ℝ × ℝ := (1, 2)

theorem direction_and_magnitude_of_maximal_growth :
  let grad_z := (6 * point_M.1, -4 * point_M.2) in
  grad_z = (6, -8) ∧ (real.sqrt ((6:ℝ)^2 + (-8:ℝ)^2) = 10) :=
by
  let grad_z := (6 * point_M.1, -4 * point_M.2)
  split
  { sorry }
  { sorry }

end direction_and_magnitude_of_maximal_growth_l809_809278


namespace middle_odd_number_is_26_l809_809897

theorem middle_odd_number_is_26 (x : ℤ) 
  (h : (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 130) : x = 26 := 
by 
  sorry

end middle_odd_number_is_26_l809_809897


namespace probability_event_B_after_transfer_l809_809765

theorem probability_event_B_after_transfer (
  red_balls_A : ℕ, white_balls_A : ℕ, black_balls_A : ℕ,
  red_balls_B : ℕ, white_balls_B : ℕ, black_balls_B : ℕ) :
  red_balls_A = 3 → white_balls_A = 2 → black_balls_A = 5 →
  red_balls_B = 3 → white_balls_B = 3 → black_balls_B = 4 →
  (let total_balls_A := red_balls_A + white_balls_A + black_balls_A in
   let total_balls_B := red_balls_B + white_balls_B + black_balls_B in
   let prob_A1 := red_balls_A / total_balls_A in
   let prob_A2 := white_balls_A / total_balls_A in
   let prob_A3 := black_balls_A / total_balls_A in
   let prob_B_given_A1 := (red_balls_B + 1) / (total_balls_B + 1) in
   let prob_B_given_A2 := red_balls_B / (total_balls_B + 1) in
   let prob_B_given_A3 := red_balls_B / (total_balls_B + 1) in
   let prob_B := prob_A1 * prob_B_given_A1 + prob_A2 * prob_B_given_A2 + prob_A3 * prob_B_given_A3 in
   prob_B = 3 / 10) :=
begin
  intros hred_a hwhite_a hblack_a hred_b hwhite_b hblack_b,
  simp only [hred_a, hwhite_a, hblack_a, hred_b, hwhite_b, hblack_b],
  norm_num,
end

end probability_event_B_after_transfer_l809_809765


namespace sum_of_coefficients_l809_809736

theorem sum_of_coefficients {x : ℝ} :
  let a := (3 * x - 1)^7
  in (a.coeff 7 + a.coeff 6 + a.coeff 5 + a.coeff 4 + a.coeff 3 + a.coeff 2 + a.coeff 1) = 2186 := by
sorry

end sum_of_coefficients_l809_809736


namespace car_stops_at_3_seconds_l809_809867

-- Define the distance function S(t)
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

-- The proof statement
theorem car_stops_at_3_seconds :
    (∃ t : ℝ, S t = 27 ∧ t = 3) :=
begin
  use 3,
  split,
  { unfold S,
    simp,
    exact eq.refl 27 },
  { exact eq.refl 3 },
end

end car_stops_at_3_seconds_l809_809867


namespace contrapositive_true_l809_809474

theorem contrapositive_true (h : ∀ x : ℝ, x < 0 → x^2 > 0) : 
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by 
  sorry

end contrapositive_true_l809_809474


namespace find_p_b_l809_809320

variables (p : Set ℝ → ℝ)
variables (a b : Set ℝ)

-- Definitions for the conditions given in the problem
def p_a := (2 / 15 : ℝ)
def p_b_given_a := (6 / 15 : ℝ)
def p_a_union_b := (12 / 15 : ℝ)

-- Hypotheses from the problem 
hypothesis (H1 : p a = p_a)
hypothesis (H2 : p b = ∃ (x : ℝ), p b = x)
hypothesis (H3 : p (a ∪ b) = p_a_union_b)
hypothesis (H4 : p (a ∩ b) = p_b_given_a * p_a)

-- The goal is to prove p(b) = 11/15
theorem find_p_b (H1 : p a = p_a) 
                 (H3 : p (a ∪ b) = p_a_union_b) 
                 (H4 : p (a ∩ b) = p_b_given_a * p_a) : 
                 ∃ (x : ℝ), p b = (11 / 15 : ℝ) :=
by 
  sorry

end find_p_b_l809_809320


namespace problem_sol_l809_809661

open Real Trig

noncomputable def numberOfRealSolutions : ℝ := 95

theorem problem_sol : ∃ x : ℝ, -150 ≤ x ∧ x ≤ 150 ∧ (x / 150 = sin x) ∧ finset.card (finset.filter (λx, x / 150 = sin x) (finset.range 301 - 151)) = 95 :=
by
  sorry

end problem_sol_l809_809661


namespace log_base2_condition_l809_809683

theorem log_base2_condition (a b : ℝ) (h : log 2 a = 0.5 ^ a ∧ 0.5 ^ a = 0.2 ^ b) : b < 1 ∧ 1 < a :=
sorry

end log_base2_condition_l809_809683


namespace debate_panel_probability_l809_809127
open Nat

/-- The probability that a randomly selected 4-person debate panel from 20 members (8 boys and 12 girls) includes at least one boy and one girl is 856/969. -/
theorem debate_panel_probability :
  let total_members := 20
  let boys := 8
  let girls := 12
  let panel_size := 4
  let total_ways := binomial total_members panel_size
  let all_boys_ways := binomial boys panel_size
  let all_girls_ways := binomial girls panel_size
  let at_least_one_of_each := (total_ways - (all_boys_ways + all_girls_ways))
  let probability := at_least_one_of_each / total_ways
  probability = 856 / 969 :=
by
  sorry

end debate_panel_probability_l809_809127


namespace max_and_min_of_expression_l809_809306

variable {x y : ℝ}

theorem max_and_min_of_expression (h : |5 * x + y| + |5 * x - y| = 20) : 
  (∃ (maxQ minQ : ℝ), maxQ = 124 ∧ minQ = 3 ∧ 
  (∀ z, z = x^2 - x * y + y^2 → z <= 124 ∧ z >= 3)) :=
sorry

end max_and_min_of_expression_l809_809306


namespace expressInScientificNotation_l809_809643

def excessivePackagingReduction := True

def carbonDioxideEmissionsReduced (x : ℕ) : Prop :=
  x = 3120000

def scientificNotation (x : ℕ) : Prop :=
  x = 3.12 * (10 ^ 6)

theorem expressInScientificNotation :
  ∀ x : ℕ, carbonDioxideEmissionsReduced x → scientificNotation x := 
by
  intros x hx
  rw hx
  -- Proof omitted
  sorry

end expressInScientificNotation_l809_809643


namespace eventually_constant_l809_809776

noncomputable theory

def f : ℕ → ℕ := sorry

axiom f_pos : ∀ x : ℕ, 0 < f x
axiom f_bound : ∀ x : ℕ, f x < 2000
axiom f_ineq : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)

theorem eventually_constant :
  ∃ m : ℕ, ∀ x : ℕ, x ≥ m → f x = f m :=
sorry

end eventually_constant_l809_809776


namespace shopkeeper_percentage_profit_l809_809989

variable {x : ℝ} -- cost price per kg of apples

theorem shopkeeper_percentage_profit 
  (total_weight : ℝ)
  (first_half_sold_at : ℝ)
  (second_half_sold_at : ℝ)
  (first_half_profit : ℝ)
  (second_half_profit : ℝ)
  (total_cost_price : ℝ)
  (total_selling_price : ℝ)
  (total_profit : ℝ)
  (percentage_profit : ℝ) :
  total_weight = 100 →
  first_half_sold_at = 0.5 * total_weight →
  second_half_sold_at = 0.5 * total_weight →
  first_half_profit = 25 →
  second_half_profit = 30 →
  total_cost_price = x * total_weight →
  total_selling_price = (first_half_sold_at * (1 + first_half_profit / 100) * x) + (second_half_sold_at * (1 + second_half_profit / 100) * x) →
  total_profit = total_selling_price - total_cost_price →
  percentage_profit = (total_profit / total_cost_price) * 100 →
  percentage_profit = 27.5 := by
  sorry

end shopkeeper_percentage_profit_l809_809989


namespace three_digit_numbers_satisfying_f_l809_809803

def f (a b c : ℕ) : ℕ :=
  (a + b + c) + (a * b + b * c + c * a) + a * b * c

theorem three_digit_numbers_satisfying_f :
  {n : ℕ | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = 100 * a + 10 * b + c ∧ f a b c = 100 * a + 10 * b + c}.to_finset.card = 9 :=
sorry

end three_digit_numbers_satisfying_f_l809_809803


namespace largest_base_not_digit_sum_9_l809_809976

theorem largest_base_not_digit_sum_9 :
  ∃ b : ℕ, (∀ (d : ℕ), (d < b) → (d > 1) → ( (digits b (12^3)).sum < 9) ) :=
begin
  -- proof
  sorry
end

end largest_base_not_digit_sum_9_l809_809976


namespace valid_net_B_l809_809113

open Classical

inductive Color where
| black : Color
| white : Color
| grey : Color

structure CubeNet where
  faces : Fin 6 → Color
  (opposite_face : Fin 6 → Fin 6)
  (opposite_spec : ∀ i, opposite_face (opposite_face i) = i)
  (color_consistent : ∀ i, faces i = faces (opposite_face i))

namespace CubeNet

def possible_net : CubeNet → Prop :=
  λ net, ∀ i j,
    i ≠ j →
    net.faces i ≠ net.faces j →
    (i, j) ∉ {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}

theorem valid_net_B (nets : List CubeNet) (netB : CubeNet) :
  netB ∈ nets → possible_net netB := by
  sorry

end CubeNet

end valid_net_B_l809_809113


namespace solve_system_inequalities_l809_809859

theorem solve_system_inequalities (x : ℝ) :
  (x - 7 < 5 * (x - 1)) ∧ ((4/3) * x + 3 ≥ 1 - (2/3) * x) →
  x > -1/2 :=
by
  intro h
  cases h with h1 h2
  have h1' : x > -1/2 := sorry
  exact h1'

end solve_system_inequalities_l809_809859


namespace max_distance_complex_l809_809804

open Complex

theorem max_distance_complex (z : ℂ) (h : ∥z∥ = 3) :
  ∥(2 + 3 * I) * z ^ 4 - z ^ 6∥ = 729 + 162 * Real.sqrt 13 :=
sorry

end max_distance_complex_l809_809804


namespace volume_relation_l809_809694

-- Define the radius of the cone, cylinder, and sphere as r (ℝ)
variables (r : ℝ)

-- Define the height of the cone and cylinder as 2r
def height := 2 * r

-- Define the volume of the cone as A
def volume_cone : ℝ := (1 / 3) * π * r^2 * height

-- Define the volume of the cylinder as M
def volume_cylinder : ℝ := π * r^2 * height

-- Define the volume of the sphere as C
def volume_sphere : ℝ := (4 / 3) * π * r^3

-- Theorem statement
theorem volume_relation (r : ℝ) :
  2 * volume_cone r + volume_cylinder r = 2.5 * volume_sphere r :=
by sorry

end volume_relation_l809_809694


namespace mixed_doubles_count_l809_809907

theorem mixed_doubles_count : 
  let males := 5
  let females := 4
  ∃ (ways : ℕ), (ways = (Nat.choose males 2) * (Nat.choose females 2) * 2) ∧ ways = 120 := 
by
  sorry

end mixed_doubles_count_l809_809907


namespace fifteenth_term_ratio_l809_809036

noncomputable def U (n : ℕ) (c f : ℚ) := n * (2 * c + (n - 1) * f) / 2
noncomputable def V (n : ℕ) (g h : ℚ) := n * (2 * g + (n - 1) * h) / 2

theorem fifteenth_term_ratio (c f g h : ℚ)
  (h1 : ∀ n : ℕ, (n > 0) → (U n c f) / (V n g h) = (5 * (n * n) + 3 * n + 2) / (3 * (n * n) + 2 * n + 30)) :
  (c + 14 * f) / (g + 14 * h) = 125 / 99 :=
by
  sorry

end fifteenth_term_ratio_l809_809036


namespace sequence_term_l809_809676

theorem sequence_term (S : ℕ → ℕ) (h : ∀ (n : ℕ), S n = 5 * n + 2 * n^2) (r : ℕ) : 
  (S r - S (r - 1) = 4 * r + 3) :=
by {
  sorry
}

end sequence_term_l809_809676


namespace bob_refund_amount_l809_809231

def total_packs : ℕ := 80
def expired_percentage : ℝ := 40
def price_per_pack : ℝ := 12
def refund_amount : ℝ := (expired_percentage / 100) * total_packs * price_per_pack

theorem bob_refund_amount : refund_amount = 384 := by
  -- the proof goes here
  sorry

end bob_refund_amount_l809_809231


namespace actual_time_when_watch_reads_five_pm_l809_809239

def time_loss_rate : ℝ := (111 + 48/60) / 120

theorem actual_time_when_watch_reads_five_pm :
  ∀ (start_time actual_10am watch_10am: ℝ), 
  start_time = 0 ∧ actual_10am = 2 ∧ watch_10am = (51 + 48/60) →
  let actual_minutes := 540 * (1 / time_loss_rate) in
  actual_minutes = 579 + 15/60 :=
by
  intros start_time actual_10am watch_10am h
  calc
    time_loss_rate = (111.8 : ℝ) / 120 : by norm_num
    _ → actual_minutes : ℝ := 540 * (120 / 111.8) : by norm_num
    _ = 579 + 15/60 : by sorry

end actual_time_when_watch_reads_five_pm_l809_809239


namespace mark_departure_time_l809_809081

-- Defining the conditions as constants
def rob_travel_time : ℕ := 1
def mark_travel_time : ℕ := 3 * rob_travel_time
def rob_departure_time : ℕ := 11 -- 11 a.m. in hours

-- The statement to be proved
theorem mark_departure_time :
  ∃ mark_departure_time : ℕ, mark_departure_time = 9 := 
by {
  have rob_arrival_time : ℕ := rob_departure_time + rob_travel_time,
  have mark_arrival_time : ℕ := rob_arrival_time,
  have mark_departure_time := mark_arrival_time - mark_travel_time,
  use mark_departure_time,
  sorry
}

end mark_departure_time_l809_809081


namespace positive_integer_solutions_l809_809274

theorem positive_integer_solutions :
  ∀ m n : ℕ, 0 < m ∧ 0 < n ∧ 3^m - 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by
  sorry

end positive_integer_solutions_l809_809274


namespace perfect_square_exists_l809_809034

theorem perfect_square_exists (A : Finset ℕ) (hA : A.card = 2022)
  (h_prime_divisors : ∀ a ∈ A, ∀ p ∈ (nat.prime_divisors a), p ≤ 30) :
  ∃ (a b c d : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a * b * c * d).isSquare :=
by
  sorry

end perfect_square_exists_l809_809034


namespace m_value_for_positive_root_eq_l809_809293

-- We start by defining the problem:
-- Given the condition that the equation (3x - 1)/(x + 1) - m/(x + 1) = 1 has a positive root,
-- we need to prove that m = -4.

theorem m_value_for_positive_root_eq (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 :=
by
  sorry

end m_value_for_positive_root_eq_l809_809293


namespace sin_x_intersect_ratio_l809_809478

theorem sin_x_intersect_ratio :
  ∃ r s : ℕ, r < s ∧ Nat.coprime r s ∧ (∀ n : ℤ, ∃ x1 x2 : ℝ, 
    (x1 = 30 + 360 * n ∧ x2 = 150 + 360 * n) ∧ (∃ k : ℤ, y = sin (k * 360 + 30) ∧ y = sin (k * 360 + 150)) ∧
    ((x2 - x1 = 120) ∧ (x1 + 360 - x2 = 240)) ∧ (r : ℝ) / (s : ℝ) = 1 / 2) :=
⟨1, 2, by decide, Nat.coprime_one_right _, by sorry⟩

end sin_x_intersect_ratio_l809_809478


namespace probability_red_then_blue_l809_809197

theorem probability_red_then_blue :
  let total_marbles := 4 + 3 + 6 in
  let prob_red := 4 / total_marbles in
  let remaining_marbles := total_marbles - 1 in
  let prob_blue_after_red := 3 / remaining_marbles in
  prob_red * prob_blue_after_red = 0.076923 :=
by
  sorry

end probability_red_then_blue_l809_809197


namespace intersection_complement_A_U_and_B_l809_809035

def U := {-1, 0, 1, 2, 3}
def A := {-1, 0}
def B := {0, 1, 2}

theorem intersection_complement_A_U_and_B : (U \ A) ∩ B = {1, 2} := by
  sorry

end intersection_complement_A_U_and_B_l809_809035


namespace dogs_legs_count_l809_809922

def animals : ℕ := 300
def fraction_dogs : ℚ := 1 / 3
def dogs_per_animal (animals : ℕ) (fraction_dogs : ℚ) : ℕ :=
  animals * fraction_dogs

def legs_per_dog : ℕ := 4
def total_dog_legs (dogs : ℕ) (legs_per_dog : ℕ) : ℕ :=
  dogs * legs_per_dog

theorem dogs_legs_count :
  total_dog_legs (dogs_per_animal animals fraction_dogs) legs_per_dog = 400 :=
  by sorry

end dogs_legs_count_l809_809922


namespace probability_of_shaded_section_l809_809377

theorem probability_of_shaded_section 
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (H1 : total_sections = 8)
  (H2 : shaded_sections = 4)
  : (shaded_sections / total_sections : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_shaded_section_l809_809377


namespace mark_departure_time_l809_809082

-- Defining the conditions as constants
def rob_travel_time : ℕ := 1
def mark_travel_time : ℕ := 3 * rob_travel_time
def rob_departure_time : ℕ := 11 -- 11 a.m. in hours

-- The statement to be proved
theorem mark_departure_time :
  ∃ mark_departure_time : ℕ, mark_departure_time = 9 := 
by {
  have rob_arrival_time : ℕ := rob_departure_time + rob_travel_time,
  have mark_arrival_time : ℕ := rob_arrival_time,
  have mark_departure_time := mark_arrival_time - mark_travel_time,
  use mark_departure_time,
  sorry
}

end mark_departure_time_l809_809082


namespace cube_root_of_neg_64_l809_809258

-- Define the function for the cube of a number
def cube (x : ℝ) : ℝ := x^3

-- Given that (-4)^3 = -64
lemma cube_neg_four : cube (-4) = -64 :=
  by 
  calc
    cube (-4) = (-4)^3 : by rfl
           ... = -64 : by ring

-- We need to prove that the cube root of -64 is -4
theorem cube_root_of_neg_64 : ∃ (x : ℝ), cube x = -64 ∧ x = -4 :=
  by 
  use -4
  split
  . exact cube_neg_four
  . by rfl

end cube_root_of_neg_64_l809_809258


namespace find_constant_l809_809875

def f (x : ℝ) : ℝ := x + 4

theorem find_constant :
  (3 * f (0.4 - 2) / f 0 + c = f (2 * 0.4 + 1)) → c = 4 :=
by
  intro h
  have h1 : f 0 = 4, by exact (f 0).symm
  have h2 : f (2 * 0.4 + 1) = 2 * 0.4 + 5, by exact (f (2 * 0.4 + 1)).symm
  have h3 : f (0.4 - 2) = 0.4 + 2, by exact (f (0.4 - 2)).symm
  sorry

end find_constant_l809_809875


namespace area_of_square_with_corners_l809_809437

theorem area_of_square_with_corners 
  (A B : ℝ × ℝ)
  (hA : A = (4, -1))
  (hB : B = (-1, 3))
  (h_adj : ∃C D : ℝ × ℝ, (A, B, C, D) are_corners_of_square) :
  ∃ (s : ℝ), (s = dist A B) ∧ (s ^ 2 = 41) :=
by
  sorry

end area_of_square_with_corners_l809_809437


namespace sum_geometric_sequence_divisibility_l809_809483

theorem sum_geometric_sequence_divisibility (n : ℕ) (h_pos: n > 0) :
  (n % 2 = 1 ↔ (3^(n+1) - 2^(n+1)) % 5 = 0) :=
sorry

end sum_geometric_sequence_divisibility_l809_809483


namespace bob_refund_amount_l809_809230

theorem bob_refund_amount 
  (total_packs : ℕ) (expired_percentage : ℝ) (cost_per_pack : ℝ) 
  (h_total_packs : total_packs = 80) 
  (h_expired_percentage : expired_percentage = 0.40) 
  (h_cost_per_pack : cost_per_pack = 12) : 
  real.of_nat (total_packs * expired_percentage.to_nat * cost_per_pack) = 384 :=
by 
  sorry

end bob_refund_amount_l809_809230


namespace largest_base_5_three_digit_in_base_10_l809_809946

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l809_809946


namespace cross_section_area_of_tetrahedron_l809_809585

-- Define a regular tetrahedron and necessary points
structure Tetrahedron :=
  (A B C D : Point)
  (edge_length : ℝ)
  (regular : RegularTetrahedron A B C D edge_length)

-- Conditions from the problem
variable (T : Tetrahedron)
def K := midpoint T.A T.D
def plane : Plane := {
  point := K,
  normal_vector := vector_perp_to (line_segment T.C T.D)
}

-- Define the cross-section and its area
noncomputable def cross_section_area (T : Tetrahedron) (plane : Plane) : ℝ := sorry

-- The theorem statement
theorem cross_section_area_of_tetrahedron (T : Tetrahedron) (h : T.edge_length = a) :
  cross_section_area T plane = (a^2 * real.sqrt 2) / 16 := 
sorry

end cross_section_area_of_tetrahedron_l809_809585


namespace geometric_progression_difference_l809_809479

variable {n : ℕ}
variable {a : ℕ → ℝ} -- assuming the sequence is indexed by natural numbers
variable {a₁ : ℝ}
variable {r : ℝ} (hr : r = (1 + Real.sqrt 5) / 2)

def geometric_progression (a : ℕ → ℝ) (a₁ : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a₁ * (r ^ n)

theorem geometric_progression_difference
  (a₁ : ℝ)
  (hr : r = (1 + Real.sqrt 5) / 2)
  (hg : geometric_progression a a₁ r) :
  ∀ n, n ≥ 2 → a n = a (n-1) - a (n-2) :=
by
  sorry

end geometric_progression_difference_l809_809479


namespace locus_of_point_property_l809_809305

noncomputable def locus_of_points (A : Point) (B C : Point) : Set Point :=
  let sphere_AB := {P : Point | dist P A = dist P B / 2} -- Sphere with diameter AB
  let sphere_AC := {P : Point | dist P A = dist P C / 2} -- Sphere with diameter AC
  let inside_spheres := (sphere_AB ∪ sphere_AC)
  let intersection_spheres := (sphere_AB ∩ sphere_AC)
  inside_spheres \ intersection_spheres

theorem locus_of_point_property (A B C : Point) :
  (∀ P : Point, (∃ X ∈ seg A B C, ∠ (A P X) = 90) ↔ P ∈ (locus_of_points A B C)) :=
by
  sorry

end locus_of_point_property_l809_809305


namespace corrected_observation_l809_809879

theorem corrected_observation:
  (initial_mean : ℝ) (n : ℕ) (incorrect_value new_mean correct_value : ℝ) 
  (h_initial_mean : initial_mean = 36) 
  (h_n : n = 20) 
  (h_incorrect_value : incorrect_value = 40) 
  (h_new_mean : new_mean = 34.9) 
  (h_correct_value : correct_value = 18) :
  let initial_sum := n * initial_mean in
  let sum_of_19 := initial_sum - incorrect_value in
  let corrected_sum := n * new_mean in
  correct_value = corrected_sum - sum_of_19 :=
by {
  sorry
}

end corrected_observation_l809_809879


namespace sqrt_product_simplified_l809_809850

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l809_809850


namespace line_symmetric_point_eq_l809_809504

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 4, y := 5 }
def B : Point := { x := -2, y := 7 }

-- Define the perpendicular bisector condition
def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

-- Define the slope computation
def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

theorem line_symmetric_point_eq (l : ℝ → ℝ) (h : ∀ (p : Point), l p.x = p.y - 6) :
  ∃ a b c : ℝ, a * 4 + b * 5 + c = 0 ∧ a * -2 + b * 7 + c = 0 ∧ ∀ x y, y = a * x + b * x + c :=
  sorry

end line_symmetric_point_eq_l809_809504


namespace max_value_condition_l809_809727

variable {x y a : ℝ}

theorem max_value_condition (h1 : 1 ≤ x + y)
                           (h2 : x + y ≤ 4)
                           (h3 : -2 ≤ x - y)
                           (h4 : x - y ≤ 2)
                           (h5 : a > 0)
                           (h6 : ∀ (x y : ℝ), ax + y ≤ 3a + 1) :
  a > 1 := 
sorry

end max_value_condition_l809_809727


namespace complex_plane_second_quadrant_l809_809321

theorem complex_plane_second_quadrant (x : ℝ) :
  (x ^ 2 - 6 * x + 5 < 0 ∧ x - 2 > 0) ↔ (2 < x ∧ x < 5) :=
by
  -- The proof is to be completed.
  sorry

end complex_plane_second_quadrant_l809_809321


namespace celsius_to_fahrenheit_l809_809716

theorem celsius_to_fahrenheit (C F : ℤ) (h1 : C = 50) (h2 : C = 5 / 9 * (F - 32)) : F = 122 :=
by
  sorry

end celsius_to_fahrenheit_l809_809716


namespace gasoline_price_april_l809_809757

theorem gasoline_price_april (P₀ : ℝ) (P₁ P₂ P₃ P₄ : ℝ) (x : ℝ)
  (h₁ : P₁ = P₀ * 1.20)  -- Price after January's increase
  (h₂ : P₂ = P₁ * 0.80)  -- Price after February's decrease
  (h₃ : P₃ = P₂ * 1.25)  -- Price after March's increase
  (h₄ : P₄ = P₃ * (1 - x / 100))  -- Price after April's decrease
  (h₅ : P₄ = P₀)  -- Price at the end of April equals the initial price
  : x = 17 := 
by
  sorry

end gasoline_price_april_l809_809757


namespace exercise_counts_l809_809382

-- Definitions according to the conditions in the problem
def zachary_pushups := 44
def zachary_crunches := 17
def zachary_pullups := 23

def david_pushups := zachary_pushups + 29
def david_crunches := zachary_crunches - 13
def david_pullups := zachary_pullups + 10

def alyssa_pushups := 2 * zachary_pushups
def alyssa_crunches := 17 / 2 -- ambiguous interpretation; we'll consider only the whole number part
def alyssa_pullups := zachary_pullups - 8

-- The final tuple we aim to prove
noncomputable def final_counts := 
  ((zachary_pushups, zachary_crunches, zachary_pullups),
   (david_pushups, david_crunches, david_pullups),
   (alyssa_pushups, alyssa_crunches, alyssa_pullups))

-- Theorem statement with placeholder for proof
theorem exercise_counts :
  final_counts = 
  ((44, 17, 23),
   (73, 4, 33),
   (88, 8.5, 15)) :=
by sorry -- Placeholder for the proof

end exercise_counts_l809_809382


namespace min_value_expression_l809_809357

theorem min_value_expression (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) 
  (h₃ : 2 * log a b + 3 * log b a = 7) : 
  a + 1 / (b^2 - 1) ≥ 3 :=
sorry

end min_value_expression_l809_809357


namespace product_is_correct_l809_809169

-- Define the variables and conditions
variables {a b c d : ℚ}

-- State the conditions
def conditions (a b c d : ℚ) :=
  3 * a + 2 * b + 4 * c + 6 * d = 36 ∧
  4 * (d + c) = b ∧
  4 * b + 2 * c = a ∧
  c - 2 = d

-- The theorem statement
theorem product_is_correct (a b c d : ℚ) (h : conditions a b c d) :
  a * b * c * d = -315 / 32 :=
sorry

end product_is_correct_l809_809169


namespace radius_B_l809_809244

noncomputable def radius_A := 2
noncomputable def radius_D := 4

theorem radius_B (r_B : ℝ) (x y : ℝ) 
  (h1 : (2 : ℝ) + y = x + (x^2 / 4)) 
  (h2 : y = 2 - (x^2 / 8)) 
  (h3 : x = (4: ℝ) / 3) 
  (h4 : y = x + (x^2 / 4)) : r_B = 20 / 9 :=
sorry

end radius_B_l809_809244


namespace area_relation_l809_809073

def Point := ℝ × ℝ
def Triangle (A B C : Point) := True
def LineSegment (P Q : Point) := True
def Parallel (l1 l2 : LineSegment) := True
def OnLineSegment (P : Point) (l : LineSegment) := True
def Area (X : Type) := ℝ

variables (A B C D E F G H I O : Point)
variables (S₁ S₂ : ℝ)

noncomputable def is_interior (O A B C : Point) := ∃ α β γ : ℝ, α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧ O = (α • A + β • B + γ • C)
noncomputable def hexagon_area (D G H E F I : Point) := S₁
noncomputable def triangle_area (A B C : Point) := S₂

theorem area_relation
  (h_interior : is_interior O A B C)
  (h_parallel1 : Parallel (LineSegment D E) (LineSegment B C))
  (h_parallel2 : Parallel (LineSegment F G) (LineSegment C A))
  (h_parallel3 : Parallel (LineSegment H I) (LineSegment A B))
  (h_on_AB_D : OnLineSegment D (LineSegment A B))
  (h_on_AB_G : OnLineSegment G (LineSegment A B))
  (h_on_BC_I : OnLineSegment I (LineSegment B C))
  (h_on_BC_F : OnLineSegment F (LineSegment B C))
  (h_on_CA_E : OnLineSegment E (LineSegment C A))
  (h_on_CA_H : OnLineSegment H (LineSegment C A))
  (h_hex_area : hexagon_area D G H E F I = S₁)
  (h_triangle_area : triangle_area A B C = S₂) :
  S₁ ≥ (2/3) * S₂ :=
sorry

end area_relation_l809_809073


namespace steve_matching_pairs_l809_809460

/-- Steve's total number of socks -/
def total_socks : ℕ := 25

/-- Number of Steve's mismatching socks -/
def mismatching_socks : ℕ := 17

/-- Number of Steve's matching socks -/
def matching_socks : ℕ := total_socks - mismatching_socks

/-- Number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := matching_socks / 2

/-- Proof that Steve has 4 pairs of matching socks -/
theorem steve_matching_pairs : matching_pairs = 4 := by
  sorry

end steve_matching_pairs_l809_809460


namespace minimum_distance_C1_C2_l809_809778

theorem minimum_distance_C1_C2:
  let C1_x (α: ℝ) := 4 * Real.cos α,
      C1_y (α: ℝ) := 2 * Real.sqrt (2) * Real.sin α,
      C2_x (θ: ℝ) := -1 + Real.sqrt (2) * Real.cos θ,
      C2_y (θ: ℝ) := Real.sqrt (2) * Real.sin θ in
  ∃ α θ, 
    let M := (C1_x α, C1_y α),
        N := (C2_x θ, C2_y θ),
        distance := Real.sqrt ((C1_x α - (-1))^2 + (C1_y α)^2) in
    ∀ (α θ: ℝ), distance ≥ Real.sqrt 7 :=
begin
  let C1_x := λ α : ℝ, 4 * Real.cos α,
  let C1_y := λ α : ℝ, 2 * Real.sqrt (2) * Real.sin α,
  let C2_x := λ θ : ℝ, -1 + Real.sqrt (2) * Real.cos θ,
  let C2_y := λ θ : ℝ, Real.sqrt (2) * Real.sin θ,
  use [-π / 3, π / 2],
  sorry
end

end minimum_distance_C1_C2_l809_809778


namespace largest_base5_eq_124_l809_809943

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l809_809943


namespace count_valid_numbers_l809_809347

-- Defining the conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def sum_of_digits_is (n sum : ℕ) : Prop :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  a + b + c + d = sum

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The main theorem statement
theorem count_valid_numbers : 
  (number_of_valid_numbers : ℕ) = 
  (finset.card (finset.filter (λ n, four_digit_number n ∧ sum_of_digits_is n 12 ∧ divisible_by_5 n) (finset.range 10000))) :=
127

end count_valid_numbers_l809_809347


namespace a_n_formula_b_n_formula_T_n_formula_l809_809128

def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := 3 * a_seq n

def b_seq (n : ℕ) : ℕ := 3 * n - 6

def c_seq (n : ℕ) : ℚ := b_seq (n + 2) / a_seq (n + 2)

def T (n : ℕ) : ℚ := (Finset.range n).sum (λ k, c_seq k)

theorem a_n_formula (n : ℕ) : a_seq n = 3^(n-1) :=
by sorry

theorem b_n_formula (n : ℕ) : b_seq n = 3 * n - 6 :=
by sorry

theorem T_n_formula (n : ℕ) : T n = (3/4) - (3 + 2 * n) / (4 * 3^n) :=
by sorry

end a_n_formula_b_n_formula_T_n_formula_l809_809128


namespace perpendicular_and_parallel_implies_perpendicular_l809_809405

variables {α β : Type*}
variables [plane α] [plane β]
variables (m n : line)

theorem perpendicular_and_parallel_implies_perpendicular
  (h1 : m ⊥ β) (h2 : n // β) : m ⊥ n :=
sorry

end perpendicular_and_parallel_implies_perpendicular_l809_809405


namespace lenny_pens_left_l809_809395

def total_pens (boxes : ℕ) (pens_per_box : ℕ) : ℕ := boxes * pens_per_box

def pens_to_friends (total : ℕ) (percentage : ℚ) : ℚ := total * percentage

def remaining_after_friends (total : ℕ) (given : ℚ) : ℚ := total - given

def pens_to_classmates (remaining : ℚ) (fraction : ℚ) : ℚ := remaining * fraction

def final_remaining (remaining : ℚ) (given : ℚ) : ℚ := remaining - given

theorem lenny_pens_left :
  let total := total_pens 20 5 in
  let given_to_friends := pens_to_friends total (40 / 100) in
  let remaining1 := remaining_after_friends total given_to_friends in
  let given_to_classmates := pens_to_classmates remaining1 (1 / 4) in
  let remaining2 := final_remaining remaining1 given_to_classmates in
  remaining2 = 45 :=
by
  sorry

end lenny_pens_left_l809_809395


namespace total_students_in_class_l809_809373

theorem total_students_in_class (female_students : ℕ) (male_students : ℕ) (total_students : ℕ) 
  (h1 : female_students = 13) 
  (h2 : male_students = 3 * female_students) 
  (h3 : total_students = female_students + male_students) : 
    total_students = 52 := 
by
  sorry

end total_students_in_class_l809_809373


namespace simplify_sqrt_product_l809_809844

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l809_809844


namespace matching_pair_probability_l809_809268

theorem matching_pair_probability :
  ∀ (blue green total : ℕ), blue = 12 → green = 10 → total = blue + green →
  (∃ (ways_total ways_blue ways_green : ℕ),
    ways_total = Nat.choose total 2 ∧
    ways_blue = Nat.choose blue 2 ∧
    ways_green = Nat.choose green 2 ∧
    ((ways_blue + ways_green) : ℚ) / ways_total = 111 / 231) :=
by
  intros blue green total hblue hgreen htotal
  use [Nat.choose total 2, Nat.choose blue 2, Nat.choose green 2]
  split
  { rw htotal }
  split
  { rw hblue }
  split
  { rw hgreen }
  sorry

end matching_pair_probability_l809_809268


namespace probability_unit_sphere_in_cube_l809_809586

noncomputable def cube_volume : ℝ := (4:ℝ) ^ 3

noncomputable def sphere_volume : ℝ := (4 * Real.pi) / 3

theorem probability_unit_sphere_in_cube:
  let probability := sphere_volume / cube_volume in
  probability = Real.pi / 48 := by
  sorry

end probability_unit_sphere_in_cube_l809_809586


namespace probability_red_or_blue_l809_809992

theorem probability_red_or_blue :
  ∀ (total_marbles white_marbles green_marbles red_blue_marbles : ℕ),
    total_marbles = 90 →
    (white_marbles : ℝ) / total_marbles = 1 / 6 →
    (green_marbles : ℝ) / total_marbles = 1 / 5 →
    white_marbles = 15 →
    green_marbles = 18 →
    red_blue_marbles = total_marbles - (white_marbles + green_marbles) →
    (red_blue_marbles : ℝ) / total_marbles = 19 / 30 :=
by
  intros total_marbles white_marbles green_marbles red_blue_marbles
  intros h_total_marbles h_white_prob h_green_prob h_white_count h_green_count h_red_blue_count
  sorry

end probability_red_or_blue_l809_809992


namespace vectors_orthogonal_l809_809698

variables (a b : ℝ^2)

def angle_between_vectors : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2)))

theorem vectors_orthogonal (a b : ℝ^2) (h : |a + b| = |a - b|) : angle_between_vectors a b = π / 2 := 
by
  sorry

end vectors_orthogonal_l809_809698


namespace largest_integer_x_l809_809148

theorem largest_integer_x (x : ℤ) : (x / 4 + 3 / 5 < 7 / 4) → x ≤ 4 := sorry

end largest_integer_x_l809_809148


namespace count_congruent_to_three_mod_eleven_l809_809354

theorem count_congruent_to_three_mod_eleven : 
  (finset.filter (λ x, x % 11 = 3) (finset.range 300)).card = 28 := 
by 
  sorry

end count_congruent_to_three_mod_eleven_l809_809354


namespace coeff_x4y3_in_expansion_l809_809007

/-- In the expansion of (2 * x + y) * (x + 2 * y) ^ 6, the coefficient of the term x^4 * y^3 is 380. -/
theorem coeff_x4y3_in_expansion :
  let poly := (2 * x + y) * (x + 2 * y) ^ 6 in
  coeff poly (4, 3) = 380 :=
begin
  sorry
end

end coeff_x4y3_in_expansion_l809_809007


namespace quadratic_has_one_solution_positive_value_of_n_l809_809664

theorem quadratic_has_one_solution_positive_value_of_n :
  ∃ n : ℝ, (4 * x ^ 2 + n * x + 1 = 0 → n ^ 2 - 16 = 0) ∧ n > 0 ∧ n = 4 :=
sorry

end quadratic_has_one_solution_positive_value_of_n_l809_809664


namespace locus_of_right_angle_view_l809_809653

section
  variables {α : Type*} [euclidean_space α]
  variables {A B M : α}

  def on_circle_of_diameter (A B : α) (M : α) : Prop :=
    ∃ C : α, C = mid_point A B ∧ dist M C = dist A C

  def right_angle_at_M (A B M : α) : Prop :=
    ∠A M B = 90

  theorem locus_of_right_angle_view (A B : α) :
    ∀ M, right_angle_at_M A B M ↔ (on_circle_of_diameter A B M ∧ M ≠ A ∧ M ≠ B) :=
  sorry
end

end locus_of_right_angle_view_l809_809653


namespace max_Q_value_l809_809675

noncomputable def Q (a : ℝ) (x y : ℝ) : ℝ := 
  if a ≥ 0 ∧ a ≤ 1 ∧ x ∈ set.Icc 0 a ∧ y ∈ set.Icc 0 (1 - a) then 
    (if sin (π * x) ^ 2 + sin (π * y) ^ 2 > 1 then 1 else 0) 
  else 0

theorem max_Q_value : ∃ (a : ℝ), a = 0.5 ∧ Q a = 0.9 :=
by
  sorry

end max_Q_value_l809_809675


namespace min_value_inequality_l809_809040

theorem min_value_inequality (a b c : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h : a + b + c = 2) : 
  (1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a)) ≥ 27 / 8 :=
sorry

end min_value_inequality_l809_809040


namespace no_real_roots_composition_l809_809554

theorem no_real_roots_composition (a b c : ℝ) (h : (b - 1)^2 < 4 * a * c) : ¬ ∃ x : ℝ, p (p x) = x :=
by
  let p := λ x : ℝ, a * x^2 + b * x + c
  have h1 : (b - 1) ^ 2 < 4 * a * c := h
  sorry

end no_real_roots_composition_l809_809554


namespace angle_calculation_l809_809805

-- Define a regular nonadecagon
def regular_nonadecagon (A : ℕ → Point) :=
  (∃ O : Point, 
    ∃ r : ℝ,
    ∀ i : ℕ, 
      (1 ≤ i ∧ i ≤ 19) → (dist O (A i) = r) ∧ 
      ∀ j k : ℕ, 1 ≤ j ∧ j ≤ k ∧ k ≤ 19 → 
      ∠O (A j) (A k) = (2 * π * (k - j) / 19)
  )

-- Define points and the intersection
variable {A : ℕ → Point}

axiom intersection (X : Point) : 
  line_through (A 1) (A 5) ∩ line_through (A 3) (A 4) = {X}

-- The goal statement encapsulating the problem
theorem angle_calculation {X : Point} :
  regular_nonadecagon A →
  intersection X →
  ∠(A 7) X (A 5) = 1170 * π / 180 / 19 :=
sorry

end angle_calculation_l809_809805


namespace angle_bisector_larger_segment_l809_809472

theorem angle_bisector_larger_segment (A B C D : Type) [LinearOrder A] :
  ∀ (AB BC AD DC : A) (BD_line : A), (AB > BC) → 
  (BD is_angle_bisector_of_triangle ABC) → 
  AD > DC := 
by 
  intro AB BC AD DC BD_line hAB_gt_BC hBD_angle_bisector
  sorry

end angle_bisector_larger_segment_l809_809472


namespace sin_cos_value_l809_809684

theorem sin_cos_value (a : ℝ) :
  sin(π - a) = -2 * sin(π / 2 + a) →
  sin a * cos a = -2 / 5 :=
by
  intro h1
  -- proof steps go here
  sorry

end sin_cos_value_l809_809684


namespace graph_E_correct_l809_809368

-- Define the percentage data for each year.
def percentage_1960 : ℤ := 5
def percentage_1970 : ℤ := 8
def percentage_1980 : ℤ := 15
def percentage_1990 : ℤ := 30

-- Define the points that graph E should have
def graph_E_points : list (ℕ × ℤ) := [(1960, 5), (1970, 8), (1980, 15), (1990, 30)]

-- Proposition stating that graph (E) correctly represents the data points.
theorem graph_E_correct : graph_E_points = [(1960, percentage_1960), (1970, percentage_1970), (1980, percentage_1980), (1990, percentage_1990)] :=
by
  -- Placeholder for the actual proof
  sorry

end graph_E_correct_l809_809368


namespace repeating_sum_of_1_over_9801_l809_809484

theorem repeating_sum_of_1_over_9801 : 
  let period_digits : List ℕ := [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 2, 0, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7, 2, 8, 2, 9, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 9, 4, 0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7, 5, 8, 5, 9, 6, 0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 6, 6, 6, 7, 6, 8, 6, 9, 7, 0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7, 7, 7, 8, 7, 9, 8, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7, 8, 8, 8, 9, 9, 0, 9, 1, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 9, 8, 9, 9] 
  in period_digits.sum = 4950 := sorry

end repeating_sum_of_1_over_9801_l809_809484


namespace maximize_total_profit_l809_809500

-- Definitions based on the problem conditions
variables (t x : ℝ)

def profit_P (t : ℝ) : ℝ := 3 * real.sqrt t
def profit_Q (t : ℝ) : ℝ := t

-- The total profit y as a function of x (question (1))
def total_profit (x : ℝ) : ℝ := profit_P x + profit_Q (3 - x)

-- Statement to prove the equality for total profit and maximization (question (2))
theorem maximize_total_profit (h : 0 ≤ x ∧ x ≤ 3) : 
  total_profit x = 3 * real.sqrt x + 3 - x ∧ 
  (∃ (x_opt : ℝ), x_opt = 9 / 4 ∧ total_profit x_opt = 3 / 4) :=
sorry

end maximize_total_profit_l809_809500


namespace square_of_sum_possible_l809_809780

theorem square_of_sum_possible (a b c : ℝ) : 
  ∃ d : ℝ, d = (a + b + c)^2 :=
sorry

end square_of_sum_possible_l809_809780


namespace find_square_sum_l809_809261

variable (a b c : ℝ)
variable (p q r : ℝ)

-- Conditions from the problem:
axiom tetrahedron_sides_eq (h1 : p^2 + q^2 = a^2)
axiom tetrahedron_sides_eq2 (h2 : p^2 + r^2 = b^2)
axiom tetrahedron_sides_eq3 (h3 : q^2 + r^2 = c^2)
axiom tetrahedron_circumradius (h4 : p^2 + q^2 + r^2 = 4)

-- Goal to prove:
theorem find_square_sum (a b c : ℝ) (p q r : ℝ) 
    (h1 : p^2 + q^2 = a^2) (h2 : p^2 + r^2 = b^2) 
    (h3 : q^2 + r^2 = c^2) (h4 : p^2 + q^2 + r^2 = 4) : 
    a^2 + b^2 + c^2 = 8 :=
by 
    sorry

end find_square_sum_l809_809261


namespace tutors_next_meeting_day_l809_809015

open Nat

theorem tutors_next_meeting_day : 
  let isaac_days := 5
  let jaclyn_days := 4
  let marcelle_days := 3
  let susanna_days := 6
  let wanda_days := 8
  lcm (lcm isaac_days (lcm jaclyn_days (lcm marcelle_days susanna_days))) wanda_days = 120 := 
by
  -- Proof not required
  sorry

end tutors_next_meeting_day_l809_809015


namespace pages_revised_twice_l809_809886

theorem pages_revised_twice
  (x : ℕ)
  (h1 : ∀ x, x > 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h2 : ∀ x, x < 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h3 : 1000 + 100 + 10 * 30 = 1400) :
  x = 30 :=
by
  sorry

end pages_revised_twice_l809_809886


namespace braeburn_money_l809_809068

def mrJonathan := 75

def cortland := mrJonathan / 2

def braeburn := cortland - 5

theorem braeburn_money : braeburn = 32.5 := 
by
  sorry

end braeburn_money_l809_809068


namespace total_amount_l809_809199

noncomputable def A : ℝ := 360.00000000000006
noncomputable def B : ℝ := (3/2) * A
noncomputable def C : ℝ := 4 * B

theorem total_amount (A B C : ℝ)
  (hA : A = 360.00000000000006)
  (hA_B : A = (2/3) * B)
  (hB_C : B = (1/4) * C) :
  A + B + C = 3060.0000000000007 := by
  sorry

end total_amount_l809_809199


namespace greater_fraction_is_correct_l809_809131

noncomputable def greater_fraction (x y : ℚ) : ℚ :=
if x > y then x else y

theorem greater_fraction_is_correct :
  ∃ (x y : ℚ), x + y = 5 / 6 ∧ x * y = 1 / 8 ∧ greater_fraction x y = (5 + real.sqrt 7) / 12 := by
  sorry

end greater_fraction_is_correct_l809_809131


namespace vasya_can_interfere_with_petya_goal_l809_809831

theorem vasya_can_interfere_with_petya_goal :
  ∃ (evens odds : ℕ), evens + odds = 50 ∧ (evens + odds) % 2 = 1 :=
sorry

end vasya_can_interfere_with_petya_goal_l809_809831


namespace volume_proportionality_l809_809901

variable (W V : ℕ)
variable (k : ℚ)

-- Given conditions
theorem volume_proportionality (h1 : V = k * W) (h2 : W = 112) (h3 : k = 3 / 7) :
  V = 48 := by
  sorry

end volume_proportionality_l809_809901


namespace marc_trip_equation_l809_809057

theorem marc_trip_equation (t : ℝ) 
  (before_stop_speed : ℝ := 90)
  (stop_time : ℝ := 0.5)
  (after_stop_speed : ℝ := 110)
  (total_distance : ℝ := 300)
  (total_trip_time : ℝ := 3.5) :
  before_stop_speed * t + after_stop_speed * (total_trip_time - stop_time - t) = total_distance :=
by 
  sorry

end marc_trip_equation_l809_809057


namespace rectangle_area_expectation_rectangle_area_standard_deviation_cm2_l809_809482

noncomputable def expected_area (E_X : ℝ) (E_Y : ℝ) : ℝ := E_X * E_Y

noncomputable def variance_of_product (E_X : ℝ) (E_Y : ℝ) (Var_X : ℝ) (Var_Y : ℝ) : ℝ :=
  (E_X^2 * Var_Y) + (E_Y^2 * Var_X) + (Var_X * Var_Y)

noncomputable def standard_deviation (variance : ℝ) : ℝ := real.sqrt variance

theorem rectangle_area_expectation :
  let E_X := 1  -- Expected width in meters
  let E_Y := 2  -- Expected length in meters
  in expected_area E_X E_Y = 2 := by
  sorry

theorem rectangle_area_standard_deviation_cm2 :
  let E_X := 1  -- Expected width in meters
  let E_Y := 2  -- Expected length in meters
  let Var_X := 0.003 ^ 2  -- Variance of width in square meters
  let Var_Y := 0.002 ^ 2  -- Variance of length in square meters
  let Var_A := variance_of_product E_X E_Y Var_X Var_Y
  let SD_A_m2 := standard_deviation Var_A
  let SD_A_cm2 := SD_A_m2 * (100 ^ 2)  -- Conversion to square centimeters
  in SD_A_cm2 ≈ 63 := by
  sorry

end rectangle_area_expectation_rectangle_area_standard_deviation_cm2_l809_809482


namespace angle_sum_star_area_ratio_intersection_pentagon_l809_809481

/- Definitions required by the problem -/
def is_convex_pentagon (ABCDE : List Point)(ABCDE.length = 5) : Prop :=
∀ (i < 5), angle (ABCDE[i]) (ABCDE[(i+1)%5]) (ABCDE[(i+2)%5]) < 180

def is_regular_pentagon (ABCDE : List Point)(ABCDE.length = 5) : Prop :=
∀ (i < 5), distance (ABCDE[i]) (ABCDE[(i+1)%5]) = distance (ABCDE[(i+1)%5]) (ABCDE[(i+2)%5]) ∧ ∀ (i < 5), angle (ABCDE[i]) (ABCDE[(i+1)%5]) (ABCDE[(i+2)%5]) = 108

/- The sum of the angles of the five-pointed star. -/
theorem angle_sum_star {ABCDE : List Point} (h_convex: is_convex_pentagon ABCDE) :
  (angle (ABCDE[0]) + angle (ABCDE[1]) + angle (ABCDE[2]) + angle (ABCDE[3]) + angle (ABCDE[4])) = 180 :=
sorry

/- The ratio of the area of the intersection pentagon to the original regular pentagon. -/
theorem area_ratio_intersection_pentagon {ABCDE : List Point} (h_regular: is_regular_pentagon ABCDE) :
  (area_intersection_pentagon (ABCDE[0]) (ABCDE[1]) (ABCDE[2]) (ABCDE[3]) (ABCDE[4])) / (area_regular_pentagon ABCDE[0] ABCDE[1] ABCDE[2] ABCDE[3] ABCDE[4]) = (7 - 3 * (sqrt 5)) / 2 :=
sorry

end angle_sum_star_area_ratio_intersection_pentagon_l809_809481


namespace jill_basket_total_weight_l809_809785

def jill_basket_capacity : ℕ := 24
def type_a_weight : ℕ := 150
def type_b_weight : ℕ := 170
def jill_basket_type_a_count : ℕ := 12
def jill_basket_type_b_count : ℕ := 12

theorem jill_basket_total_weight :
  (jill_basket_type_a_count * type_a_weight + jill_basket_type_b_count * type_b_weight) = 3840 :=
by
  -- We provide the calculations for clarification; not essential to the theorem statement
  -- (12 * 150) + (12 * 170) = 1800 + 2040 = 3840
  -- Started proof to provide context; actual proof steps are omitted
  sorry

end jill_basket_total_weight_l809_809785


namespace pipe_A_fill_time_l809_809570

noncomputable def pipe_A_fill_rate (t : ℝ) : ℝ := 1 / t
def pipe_B_empty_rate : ℝ := 1 / 15
def net_rate_with_both_pipes (t : ℝ) : ℝ := pipe_A_fill_rate t - pipe_B_empty_rate
def net_rate_given : ℝ := 1 / 30

theorem pipe_A_fill_time (t : ℝ) : net_rate_with_both_pipes t = net_rate_given → t = 10 :=
by
  sorry

end pipe_A_fill_time_l809_809570


namespace prove_inequality_l809_809259

noncomputable def inequality_holds (x y : ℝ) : Prop :=
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2)

theorem prove_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : inequality_holds x y :=
  sorry

end prove_inequality_l809_809259


namespace problem_inequality_l809_809041

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable h : ∀ i, i ≤ n → a i > 0

theorem problem_inequality (h : ∀ i, i ≤ n → a i > 0) :
  (∑ k in Finset.range n, (k + 1) / (∑ i in Finset.range (k + 1), a (i + 1))) ≤
  2 * ∑ i in Finset.range n, 1 / (a (i + 1)) :=
sorry

end problem_inequality_l809_809041


namespace Timi_has_five_ears_l809_809911

theorem Timi_has_five_ears (seeing_ears_Imi seeing_ears_Dimi seeing_ears_Timi : ℕ)
  (H1 : seeing_ears_Imi = 8)
  (H2 : seeing_ears_Dimi = 7)
  (H3 : seeing_ears_Timi = 5)
  (total_ears : ℕ := (seeing_ears_Imi + seeing_ears_Dimi + seeing_ears_Timi) / 2) :
  total_ears - seeing_ears_Timi = 5 :=
by
  sorry -- Proof not required.

end Timi_has_five_ears_l809_809911


namespace minimum_custodians_needed_l809_809496

noncomputable def min_custodians (n : ℕ) (h : n > 1) : ℕ :=
  ⌊n / 2⌋

theorem minimum_custodians_needed (n : ℕ) (h : n > 1)
  (chequered_figure : ∀ (i j : ℕ), chequered_figure i j → chequered_figure (i + 1) j ∨ chequered_figure i (j + 1))
  (adjacent_reachable : ∀ (i j : ℕ), room_reachable i j → adjacent_by_side i j)
  (custodian_reach : ∀ (i j : ℕ), custodian_reach i j → move_like_chess_rook i j)
  : (min_custodians n h) = ⌊n / 2⌋ := 
by 
  sorry

end minimum_custodians_needed_l809_809496


namespace min_value_of_translated_function_l809_809486

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

theorem min_value_of_translated_function :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ (Real.pi / 2) → ∀ (ϕ : ℝ), |ϕ| < (Real.pi / 2) →
  ∀ (k : ℤ), f (x + (Real.pi / 6)) (ϕ + (Real.pi / 3) + k * Real.pi) = f x ϕ →
  ∃ y : ℝ, y = - Real.sqrt 3 / 2 := sorry

end min_value_of_translated_function_l809_809486


namespace one_million_minutes_later_l809_809029

open Nat

def initial_datetime : DateTime :=
  { year := 2007, month := 4, day := 15, hour := 12, minute := 0, second := 0 }

def one_million_minutes : Nat := 1000000

def target_datetime : DateTime :=
  { year := 2009, month := 3, day := 10, hour := 10, minute := 40, second := 0 }

theorem one_million_minutes_later :
  let one_million_minutes_in_seconds := one_million_minutes * 60
  let added_seconds := initial_datetime.toSecond.toNat + one_million_minutes_in_seconds 
  DateTime.ofSecond added_seconds == target_datetime :=
by
  sorry

end one_million_minutes_later_l809_809029


namespace exists_rational_not_nGood_l809_809077

-- Definitions
def isNGood (n : ℕ) (q : ℚ) : Prop :=
  ∃ (a : Fin n.succ → ℕ) (h : ∀ i, a i > 0), q = (Finset.univ : Finset (Fin n.succ)).sum (λ i, (1 : ℚ) / a i)

-- Theorem statement
theorem exists_rational_not_nGood (n : ℕ) (hn : n ≥ 1) : 
  ∃ q : ℚ, q < 1 ∧ q > 0 ∧ ¬ isNGood n q :=
sorry

end exists_rational_not_nGood_l809_809077


namespace mark_pages_per_week_l809_809060

theorem mark_pages_per_week
    (initial_hours_per_day : ℕ)
    (increase_percentage : ℕ)
    (initial_pages_per_day : ℕ) :
    initial_hours_per_day = 2 →
    increase_percentage = 150 →
    initial_pages_per_day = 100 →
    (initial_pages_per_day * (1 + increase_percentage / 100)) * 7 = 1750 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have reading_speed := 100 / 2 -- 50 pages per hour
  have increased_time := 2 * 1.5  -- 3 more hours
  have new_total_time := 2 + 3    -- 5 hours per day
  have pages_per_day := 5 * 50    -- 250 pages per day
  have pages_per_week := 250 * 7  -- 1750 pages per week
  exact eq.refl 1750

end mark_pages_per_week_l809_809060


namespace sum_of_m_l809_809750

-- Define the conditions
variable {x y m : Int}

-- The inequality system
def inequality1 : Prop := (x - 2) / 4 < (x - 1) / 3
def inequality2 : Prop := 3 * x - m ≤ 3 - x

-- The system of equations
def equation1 : Prop := m * x + y = 4
def equation2 : Prop := 3 * x - y = 0

-- Combined conditions
def conditions : Prop := 
  inequality1 ∧ 
  inequality2 ∧ 
  Int.sqrt (equation1 ∧ equation2)

-- Statement to prove
theorem sum_of_m : conditions → (∑ m, m ∈ {-3, -2, -1, 0} ∧ equation1 ∧ equation2 = -3) :=
by sorry

end sum_of_m_l809_809750


namespace leak_drain_time_l809_809212

-- The condition that the pump can fill the tank in 2 hours
def pump_fill_rate := 1 / 2 -- tanks per hour

-- The condition that it took 2 1/8 hours to fill the tank with the leak
def combined_time : ℝ := 2 + 1/8
def combined_fill_rate := 1 / combined_time -- tanks per hour

-- Proving the time taken by the leak to drain the tank
theorem leak_drain_time :
  let L := pump_fill_rate - combined_fill_rate in
  1 / L = 34 :=
by
  -- All the proof steps can be skipped with sorry
  sorry

end leak_drain_time_l809_809212


namespace zero_point_in_range_l809_809334

theorem zero_point_in_range (a : ℝ) (x1 x2 x3 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : x1 < x2) (h4 : x2 < x3)
  (hx1 : (x1^3 - 4*x1 + a) = 0) (hx2 : (x2^3 - 4*x2 + a) = 0) (hx3 : (x3^3 - 4*x3 + a) = 0) :
  0 < x2 ∧ x2 < 1 :=
by
  sorry

end zero_point_in_range_l809_809334


namespace dogs_legs_count_l809_809923

def animals : ℕ := 300
def fraction_dogs : ℚ := 1 / 3
def dogs_per_animal (animals : ℕ) (fraction_dogs : ℚ) : ℕ :=
  animals * fraction_dogs

def legs_per_dog : ℕ := 4
def total_dog_legs (dogs : ℕ) (legs_per_dog : ℕ) : ℕ :=
  dogs * legs_per_dog

theorem dogs_legs_count :
  total_dog_legs (dogs_per_animal animals fraction_dogs) legs_per_dog = 400 :=
  by sorry

end dogs_legs_count_l809_809923


namespace minimum_value_of_u_l809_809299

noncomputable def minimum_value_lemma (x y : ℝ) (hx : Real.sin x + Real.sin y = 1 / 3) : Prop :=
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m

theorem minimum_value_of_u
  (x y : ℝ)
  (hx : Real.sin x + Real.sin y = 1 / 3) :
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m :=
sorry

end minimum_value_of_u_l809_809299


namespace largest_base5_three_digit_to_base10_l809_809960

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l809_809960


namespace mark_pages_per_week_l809_809065

theorem mark_pages_per_week :
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100) * initial_reading_hours
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  increased_pages_per_day * days_per_week = 1750 :=
by
  -- Definitions
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  
  -- Calculate increased reading hours per day
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100.0) * initial_reading_hours
  -- Calculate increased pages per day
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  
  -- Calculate pages per week
  have h : increased_pages_per_day * days_per_week = 1750 := by
    sorry

  exact h

end mark_pages_per_week_l809_809065


namespace triangle_equilateral_l809_809453

theorem triangle_equilateral
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : (a^3 + b^3 + c^3) / (a + b + c) = c^2)
  (h2 : sin(α) * sin(β) = (sin(γ))^2) :
  a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_l809_809453


namespace burrito_count_l809_809459

def burrito_orders (wraps beef_fillings chicken_fillings : ℕ) :=
  if wraps = 5 ∧ beef_fillings >= 4 ∧ chicken_fillings >= 3 then 25 else 0

theorem burrito_count : burrito_orders 5 4 3 = 25 := by
  sorry

end burrito_count_l809_809459


namespace largest_base5_three_digit_in_base10_l809_809951

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l809_809951


namespace probability_sum_even_is_two_fifths_l809_809297

variable (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6})

theorem probability_sum_even_is_two_fifths (h : s = {1, 2, 3, 4, 5, 6}) :
  (s.filter (λ x, (x % 2 = 0)).card = 3 ∧ s.filter (λ x, (x % 2 = 1)).card = 3 ∧
  (Finset.card s).choose 2 = 15 → 
  (3 + 3) / 15 = 2 / 5) :=
by sorry

end probability_sum_even_is_two_fifths_l809_809297


namespace regular_hexagon_angle_l809_809074

theorem regular_hexagon_angle {A B C D E F : Point} (hex : regular_hexagon A B C D E F) : 
  angle A B F = 30 :=
by
  sorry

end regular_hexagon_angle_l809_809074


namespace remainder_of_8_pow_6_plus_7_pow_7_plus_6_pow_8_div_5_l809_809973

theorem remainder_of_8_pow_6_plus_7_pow_7_plus_6_pow_8_div_5 :
  let a := 8
  let b := 7
  let c := 6
  a ≡ -2 [MOD 5] ∧ b ≡ 2 [MOD 5] ∧ c ≡ 1 [MOD 5] →
  (a^6 + b^7 + c^8) % 5 = 3 :=
by
  sorry

end remainder_of_8_pow_6_plus_7_pow_7_plus_6_pow_8_div_5_l809_809973


namespace number_of_different_ways_is_18_l809_809439

-- Define the problem conditions
def number_of_ways_to_place_balls : ℕ :=
  let total_balls := 9
  let boxes := 3
  -- Placeholder function to compute the requirement
  -- The actual function would involve combinatorial logic
  -- Let us define it as an axiom for now.
  sorry

-- The theorem to be proven
theorem number_of_different_ways_is_18 :
  number_of_ways_to_place_balls = 18 :=
sorry

end number_of_different_ways_is_18_l809_809439


namespace triangle_regions_1000_2000_l809_809192

theorem triangle_regions_1000_2000 {n : ℕ} (h₀ : n = 3000)
  (h₁ : ∀ i j : ℕ, i ≠ j → ¬ parallel (line i) (line j))
  (h₂ : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ i → ¬ concurrent (line i) (line j) (line k)) :
  ∃ (triangles : ℕ), triangles ≥ 1000 ∧ triangles ≥ 2000 := by
  sorry

end triangle_regions_1000_2000_l809_809192


namespace problem_sol_l809_809659

open Real Trig

noncomputable def numberOfRealSolutions : ℝ := 95

theorem problem_sol : ∃ x : ℝ, -150 ≤ x ∧ x ≤ 150 ∧ (x / 150 = sin x) ∧ finset.card (finset.filter (λx, x / 150 = sin x) (finset.range 301 - 151)) = 95 :=
by
  sorry

end problem_sol_l809_809659


namespace log_base_change_l809_809796

theorem log_base_change (x k : ℝ) (h₁ : Real.logBase 8 3 = x) (h₂ : Real.logBase 2 81 = k * x) : k = 12 :=
by
  sorry

end log_base_change_l809_809796


namespace parabola_intersect_sum_l809_809579

theorem parabola_intersect_sum (x1 y1 x2 y2 : ℝ) 
    (h1 : y1^2 = 4 * x1)
    (h2 : y2^2 = 4 * x2)
    (h3 : real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 8) :
  x1 + x2 = 6 :=
sorry

end parabola_intersect_sum_l809_809579


namespace angle_B_in_ABC_l809_809389

theorem angle_B_in_ABC (A B C D : Type) [triangle A B C] 
  (h1 : AB = AC) (h2 : D ∈ AB) (h3 : ∠BD bisects ∠ACB) (h4 : BD = BC) : 
  ∠B = 90 :=
sorry

end angle_B_in_ABC_l809_809389


namespace effective_days_4_units_minimum_value_a_l809_809260

noncomputable def concentration_function (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 4 then 16 / (8 - x) - 1 else
if h : 4 < x ∧ x ≤ 10 then 5 - (1 / 2) * x else 0

def effective_concentration := 4

theorem effective_days_4_units :
  ∀ x: ℝ, 0 ≤ x ∧ x ≤ 8 → 4 * concentration_function x ≥ effective_concentration :=
sorry

theorem minimum_value_a :
  ∃ a : ℝ, (1 ≤ a ∧ a ≤ 4) ∧
    a = 24 - 16 * real.sqrt 2 ∧
     ∀ x : ℝ, 6 ≤ x ∧ x ≤ 10 → 
       2 * concentration_function (x - 6) + 
        a * (concentration_function x - 2) ≥ effective_concentration :=
sorry

end effective_days_4_units_minimum_value_a_l809_809260


namespace primes_in_Q_plus_n_sequence_l809_809166

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_in_Q_plus_n_sequence :
  let Q := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53,
      sequence := { n | 2 ≤ n ∧ n ≤ 54 }
  in
  ∀ n ∈ sequence, ¬ is_prime (Q + n) :=
by
  sorry

end primes_in_Q_plus_n_sequence_l809_809166


namespace n_divisibility_and_factors_l809_809884

open Nat

theorem n_divisibility_and_factors (n : ℕ) (h1 : 1990 ∣ n) (h2 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n):
  n = 4 * 5 * 199 ∨ n = 2 * 25 * 199 ∨ n = 2 * 5 * 39601 := 
sorry

end n_divisibility_and_factors_l809_809884


namespace find_some_number_l809_809752

theorem find_some_number :
  (let '+' := (· * ·) in
   let '-' := (· + ·) in
   let '×' := (· / ·) in
   let '÷' := (· - ·) in
   6 - 9 + 8 × 3 ÷ some_number = 5) → some_number = 25 :=
by
  intros
  sorry

end find_some_number_l809_809752


namespace interval_of_monotonic_decrease_l809_809117

noncomputable def log_monotonic_decrease_interval (a : ℝ) : Set ℝ :=
  {x : ℝ | y = real.log a (x^2 + 2 * x - 3) ∧ x ∈ Iio (-3)}

theorem interval_of_monotonic_decrease (a : ℝ) (h₀ : ∀ x : ℝ, 2 = x → real.log a (x^2 + 2 * x - 3) > 0) :
  log_monotonic_decrease_interval a = Iio (-3) :=
by
  sorry

end interval_of_monotonic_decrease_l809_809117


namespace expression_divisible_by_16_l809_809076

theorem expression_divisible_by_16 (m n : ℤ) : 
  ∃ k : ℤ, (5 * m + 3 * n + 1)^5 * (3 * m + n + 4)^4 = 16 * k :=
sorry

end expression_divisible_by_16_l809_809076


namespace unique_zero_function_l809_809271

theorem unique_zero_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l809_809271


namespace simplify_radical_expression_l809_809853

theorem simplify_radical_expression (x : ℝ) :
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x)) = 120 * x * sqrt (2 * x) := by
sorry

end simplify_radical_expression_l809_809853


namespace problem_a_problem_b_problem_d_l809_809685

open Real

theorem problem_a (a b : ℝ) (h : exp a + exp b = 4) : a + b ≤ 2 * ln 2 := 
sorry

theorem problem_b (a b : ℝ) (h : exp a + exp b = 4) : exp a + b ≤ 3 := 
sorry

theorem problem_d (a b : ℝ) (h : exp a + exp b = 4) : exp (2 * a) + exp (2 * b) ≥ 8 := 
sorry

end problem_a_problem_b_problem_d_l809_809685


namespace bob_refund_amount_l809_809229

theorem bob_refund_amount 
  (total_packs : ℕ) (expired_percentage : ℝ) (cost_per_pack : ℝ) 
  (h_total_packs : total_packs = 80) 
  (h_expired_percentage : expired_percentage = 0.40) 
  (h_cost_per_pack : cost_per_pack = 12) : 
  real.of_nat (total_packs * expired_percentage.to_nat * cost_per_pack) = 384 :=
by 
  sorry

end bob_refund_amount_l809_809229


namespace probability_two_digit_greater_than_20_l809_809681

open Finset

def two_digit_numbers (s : Finset ℕ) : Finset ℕ :=
  (s.product s).filter (λ p, p.1 ≠ p.2).image (λ p, 10 * p.1 + p.2)

def favorable_outcomes (s : Finset ℕ) : ℕ :=
  (two_digit_numbers s).filter (λ n, n > 20).card

def total_outcomes (s : Finset ℕ) : ℕ :=
  (two_digit_numbers s).card

theorem probability_two_digit_greater_than_20 : 
  let s := {1, 2, 3, 4}
  in (favorable_outcomes s : ℚ) / (total_outcomes s : ℚ) = 3 / 4 :=
by
  let s := {1, 2, 3, 4}
  have total := total_outcomes s
  have favorable := favorable_outcomes s
  have h_div : (favorable : ℚ) / (total : ℚ) = 3 / 4 := sorry
  exact h_div

end probability_two_digit_greater_than_20_l809_809681


namespace total_sweaters_calculated_l809_809787

def monday_sweaters := 8
def tuesday_sweaters := monday_sweaters + 2
def wednesday_sweaters := tuesday_sweaters - 4
def thursday_sweaters := tuesday_sweaters - 4
def friday_sweaters := monday_sweaters / 2

def total_sweaters := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem total_sweaters_calculated : total_sweaters = 34 := 
by sorry

end total_sweaters_calculated_l809_809787


namespace isosceles_triangle_angle_l809_809008

theorem isosceles_triangle_angle (A B C : Type) [triangle A B C] (isosceles : is_isosceles A B C) (angle_A : ∠ A = 110) : 
  ∠ B = 35 :=
sorry

end isosceles_triangle_angle_l809_809008


namespace william_travel_time_l809_809982

noncomputable def total_travel_time (start_time_missouri : ℕ) (arrival_time_hometown : ℕ) (time_diff : ℕ) (stops : List ℕ) (traffic_delay : ℕ) : ℕ :=
  let travel_time_without_stops := arrival_time_hometown - (start_time_missouri + time_diff)
  let total_stops_time := stops.sum
  let total_delay_time := total_stops_time + traffic_delay
  travel_time_without_stops + (total_delay_time / 60) -- converting minutes to hours

theorem william_travel_time :
  total_travel_time 7 20 2 [25, 10, 25] 45 = 12.75 :=
by
  sorry

end william_travel_time_l809_809982


namespace sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l809_809159

theorem sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0 :
  (9^25 + 11^25) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l809_809159


namespace dot_product_ABC_l809_809726

-- Defining vectors as pairs of real numbers
def vector := (ℝ × ℝ)

-- Defining the vectors AB and AC
def AB : vector := (1, 0)
def AC : vector := (-2, 3)

-- Definition of vector subtraction
def vector_sub (v1 v2 : vector) : vector := (v1.1 - v2.1, v1.2 - v2.2)

-- Definition of dot product
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define vector BC using the given vectors AB and AC
def BC : vector := vector_sub AC AB

-- The theorem stating the desired dot product result
theorem dot_product_ABC : dot_product AB BC = -3 := by
  sorry

end dot_product_ABC_l809_809726


namespace sin_pi_add_x_pos_l809_809133

open Real
open Int

theorem sin_pi_add_x_pos (k : ℤ) (x : ℝ) :
  (2 * k * π + π < x ∧ x < 2 * k * π + 2 * π) ↔ sin (π + x) > 0 :=
sorry

end sin_pi_add_x_pos_l809_809133


namespace probability_at_least_one_succeeds_l809_809526

variable (p1 p2 : ℝ)

theorem probability_at_least_one_succeeds : 
  0 ≤ p1 ∧ p1 ≤ 1 → 0 ≤ p2 ∧ p2 ≤ 1 → (1 - (1 - p1) * (1 - p2)) = 1 - (1 - p1) * (1 - p2) :=
by 
  intro h1 h2
  sorry

end probability_at_least_one_succeeds_l809_809526


namespace dogs_not_doing_anything_l809_809508

def total_dogs : ℕ := 500
def dogs_running : ℕ := 18 * total_dogs / 100
def dogs_playing_with_toys : ℕ := (3 * total_dogs) / 20
def dogs_barking : ℕ := 7 * total_dogs / 100
def dogs_digging_holes : ℕ := total_dogs / 10
def dogs_competing : ℕ := 12
def dogs_sleeping : ℕ := (2 * total_dogs) / 25
def dogs_eating_treats : ℕ := total_dogs / 5

def dogs_doing_anything : ℕ := dogs_running + dogs_playing_with_toys + dogs_barking + dogs_digging_holes + dogs_competing + dogs_sleeping + dogs_eating_treats

theorem dogs_not_doing_anything : total_dogs - dogs_doing_anything = 98 :=
by
  -- proof steps would go here
  sorry

end dogs_not_doing_anything_l809_809508


namespace repeating_decimal_fraction_l809_809539

theorem repeating_decimal_fraction :
  let x := (37/100) + (246 / 99900)
  in x = 37245 / 99900 :=
by
  let x := (37/100) + (246 / 99900)
  show x = 37245 / 99900
  sorry

end repeating_decimal_fraction_l809_809539


namespace largest_base5_three_digit_to_base10_l809_809958

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l809_809958


namespace simplify_radical_expression_l809_809854

theorem simplify_radical_expression (x : ℝ) :
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x)) = 120 * x * sqrt (2 * x) := by
sorry

end simplify_radical_expression_l809_809854


namespace total_cows_is_108_l809_809202

-- Definitions of the sons' shares and the number of cows the fourth son received
def first_son_share : ℚ := 2 / 3
def second_son_share : ℚ := 1 / 6
def third_son_share : ℚ := 1 / 9
def fourth_son_cows : ℕ := 6

-- The total number of cows in the herd
def total_cows (n : ℕ) : Prop :=
  first_son_share + second_son_share + third_son_share + (fourth_son_cows / n) = 1

-- Prove that given the number of cows the fourth son received, the total number of cows in the herd is 108
theorem total_cows_is_108 : total_cows 108 :=
by
  sorry

end total_cows_is_108_l809_809202


namespace min_distance_PQ_l809_809387

noncomputable def curveC_parametric (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π) : ℝ × ℝ :=
(1 + Real.cos α, Real.sin α)

def curveC_cartesian (x y : ℝ) : Prop :=
(x - 1)^2 + y^2 = 1 ∧ y ≥ 0

noncomputable def lineL_polar (θ : ℝ) : ℝ :=
4 / (Real.sqrt 2 * Real.sin(θ - π / 4))

def lineL_cartesian (x y : ℝ) : Prop :=
x - y + 4 = 0

def distancePQ (P Q : ℝ × ℝ) : ℝ :=
Real.sqrt ((P.fst - Q.fst)^2 + (P.snd - Q.snd)^2)

theorem min_distance_PQ :
  ∀ (P Q : ℝ × ℝ),
    (∃ α (hα : 0 ≤ α ∧ α ≤ π), P = curveC_parametric α hα)
    ∧ (∃ θ, Q = (lineL_polar θ, θ)) →
    distancePQ P Q ≥ (5 * Real.sqrt 2 / 2 - 1) :=
sorry

end min_distance_PQ_l809_809387


namespace complex_z_squared_l809_809325

theorem complex_z_squared (z : ℂ) (i : ℂ) (hz : 𝕜.exp(𝕜.pi * 𝕜.I / 2) = i) (h : 1 / z = 1 + i) : z^2 = -1 / 2 * i :=
by
  sorry

end complex_z_squared_l809_809325


namespace transformation_l809_809555

noncomputable def Q (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

theorem transformation 
  (a b c d e f x y x₀ y₀ x' y' : ℝ)
  (h : a * c - b^2 ≠ 0)
  (hQ : Q a b c x y + 2 * d * x + 2 * e * y = f)
  (hx : x' = x + x₀)
  (hy : y' = y + y₀) :
  ∃ f' : ℝ, (a * x'^2 + 2 * b * x' * y' + c * y'^2 = f' ∧ 
             f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end transformation_l809_809555


namespace correct_statement_l809_809981

def synthetic_method_is_direct : Prop := -- define the synthetic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

def analytic_method_is_direct : Prop := -- define the analytic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

theorem correct_statement : synthetic_method_is_direct ∧ analytic_method_is_direct → 
                             "Synthetic method and analytic method are direct proof methods" = "A" :=
by
  intros h
  cases h
  -- This is where you would provide the proof steps. We skip this with sorry.
  sorry

end correct_statement_l809_809981


namespace a_b_nature_l809_809519

def equation_satisfied (a b : ℂ) : Prop :=
  (a + b) / a = b / (a + b) ∧ a ≠ 0 ∧ a + b ≠ 0

theorem a_b_nature (a b : ℂ) (h : equation_satisfied a b) : 
  ¬(a ∈ ℝ ∧ b ∈ ℝ) :=
sorry

end a_b_nature_l809_809519


namespace alcohol_mix_problem_l809_809094

theorem alcohol_mix_problem
  (x_volume : ℕ) (y_volume : ℕ)
  (x_percentage : ℝ) (y_percentage : ℝ)
  (target_percentage : ℝ)
  (x_volume_eq : x_volume = 200)
  (x_percentage_eq : x_percentage = 0.10)
  (y_percentage_eq : y_percentage = 0.30)
  (target_percentage_eq : target_percentage = 0.14)
  (y_solution : ℝ)
  (h : y_volume = 50) :
  (20 + 0.3 * y_solution) / (200 + y_solution) = target_percentage := by sorry

end alcohol_mix_problem_l809_809094


namespace polynomial_roots_geometric_progression_q_l809_809116

theorem polynomial_roots_geometric_progression_q :
    ∃ (a r : ℝ), (a ≠ 0) ∧ (r ≠ 0) ∧
    (a + a * r + a * r ^ 2 + a * r ^ 3 = 0) ∧
    (a ^ 4 * r ^ 6 = 16) ∧
    (a ^ 2 + (a * r) ^ 2 + (a * r ^ 2) ^ 2 + (a * r ^ 3) ^ 2 = 16) :=
by
    sorry

end polynomial_roots_geometric_progression_q_l809_809116


namespace smallest_value_div_by_13_l809_809211

theorem smallest_value_div_by_13 : 
  ∃ (A B : ℕ), 
    (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ 
    A ≠ B ∧ 
    1001 * A + 110 * B = 1771 ∧ 
    (1001 * A + 110 * B) % 13 = 0 :=
by
  sorry

end smallest_value_div_by_13_l809_809211


namespace polynomial_expansion_l809_809689

theorem polynomial_expansion :
  let x := 1 
  let y := -1 
  let a_0 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_1 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_2 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_3 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_4 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_5 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3 + a_5)^2 = 3125 := by
sorry

end polynomial_expansion_l809_809689


namespace exists_infinite_t_l809_809442

def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (n % 10) + digit_sum (n / 10)

theorem exists_infinite_t (k : ℕ) : ∃ (inf_many_t : ℕ → Prop), 
  (∀ t, inf_many_t t → digit_sum t = digit_sum (k * t)) ∧ 
  (∀ n, ∃ t, t > n ∧ inf_many_t t ∧ (∀ d, t ≠ 0 → (t % 10) = 9)) :=
sorry

end exists_infinite_t_l809_809442


namespace range_of_m_l809_809341

noncomputable def A (x : ℝ) : ℝ := x^2 - (3/2) * x + 1

def in_interval (x : ℝ) : Prop := (3/4 ≤ x) ∧ (x ≤ 2)

def B (y : ℝ) (m : ℝ) : Prop := y ≥ 1 - m^2

theorem range_of_m (m : ℝ) :
  (∀ x, in_interval x → B (A x) m) ↔ (m ≤ - (3/4) ∨ m ≥ (3/4)) := 
sorry

end range_of_m_l809_809341


namespace zero_in_interval_l809_809134

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x + 2 * x - 8

theorem zero_in_interval : (f 3 < 0) ∧ (f 4 > 0) → ∃ c, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  sorry

end zero_in_interval_l809_809134


namespace largest_base5_three_digits_is_124_l809_809935

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l809_809935


namespace mark_reads_1750_pages_per_week_l809_809062

def initialReadingHoursPerDay := 2
def increasePercentage := 150
def initialPagesPerDay := 100

def readingHoursPerDayAfterIncrease : Nat := initialReadingHoursPerDay + (initialReadingHoursPerDay * increasePercentage) / 100
def readingSpeedPerHour := initialPagesPerDay / initialReadingHoursPerDay
def pagesPerDayNow := readingHoursPerDayAfterIncrease * readingSpeedPerHour
def pagesPerWeekNow : Nat := pagesPerDayNow * 7

theorem mark_reads_1750_pages_per_week :
  pagesPerWeekNow = 1750 :=
sorry -- Proof omitted

end mark_reads_1750_pages_per_week_l809_809062


namespace smallest_angle_is_60_l809_809972

open Real

noncomputable def smallest_positive_angle (x : ℝ) : Prop :=
  4^(sin x ^ 2) * 2^(cos x ^ 2) = 2 * 8 ^ (1 / 4) ∧ 0 < x ∧ x < 2 * π

theorem smallest_angle_is_60 :
  ∃ x, smallest_positive_angle x ∧ x = π / 3 :=
by
  sorry

end smallest_angle_is_60_l809_809972


namespace chord_length_cube_l809_809111

noncomputable def diameter : ℝ := 1
noncomputable def AC (a : ℝ) : ℝ := a
noncomputable def AD (b : ℝ) : ℝ := b
noncomputable def AE (a b : ℝ) : ℝ := (a^2 + b^2).sqrt / 2
noncomputable def AF (b : ℝ) : ℝ := b^2

theorem chord_length_cube (a b : ℝ) (h : AE a b = b^2) : a = b^3 :=
by
  sorry

end chord_length_cube_l809_809111


namespace slant_asymptote_degree_l809_809119

theorem slant_asymptote_degree (q : ℝ[X]) (hq : q.degree = 6) :
    let f := (3 * X^7 + 5 * X^4 - 6) / q in 
    (∃ d : ℝ[X], d.degree = 6) ∧ (∃ m : ℝ, ∀ x : ℝ, f = m * x + O(1 / x)) :=
by
  sorry

end slant_asymptote_degree_l809_809119


namespace members_of_set_U_are_in_set_A_l809_809551

theorem members_of_set_U_are_in_set_A (U A B : Finset α) (total : U.card = 193)
  (members_of_B : B.card = 41) (neither : (U.filter (λ x, ¬ (x ∈ A ∨ x ∈ B))).card = 59)
  (both : (A ∩ B).card = 23) : A.card = 116 := by
  sorry

end members_of_set_U_are_in_set_A_l809_809551


namespace train_speed_l809_809581

-- Declare the conditions
def time : ℕ := 10
def length_goods_train : ℕ := 240
def speed_goods_train_kmh : ℝ := 50.4

-- Define the function to convert km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

-- Prove the statement
theorem train_speed (v : ℝ) : 
  kmh_to_ms speed_goods_train_kmh + v = length_goods_train / time → 
  v = 10 → 
  v * 3600 / 1000 = 36 :=
by
  -- No proof needed, just state the theorem
  sorry

end train_speed_l809_809581


namespace smallest_positive_theta_l809_809637

theorem smallest_positive_theta :
  ∃ (θ : ℝ), θ > 0 ∧ θ = π / 14 ∧ sin (3 * θ) * sin (4 * θ) = cos (3 * θ) * cos (4 * θ) :=
sorry

end smallest_positive_theta_l809_809637


namespace sum_of_numbers_using_0_to_9_is_not_100_l809_809530

theorem sum_of_numbers_using_0_to_9_is_not_100 :
  ∀ (numbers : List ℕ), (∀ number ∈ numbers, 0 < number) → 
                            (∀ d ∈ (List.range 10), ∃! n ∈ numbers, d ∈ (n.toString.toList.map (λ c => c.toNat - '0'.toNat))) → 
                            (numbers.sum ≠ 100) :=
by
  -- The statement asserts there is no way to sum numbers made out of unique digits 0-9 to get 100.
  sorry

end sum_of_numbers_using_0_to_9_is_not_100_l809_809530


namespace expected_and_variance_l809_809725

variables (xi1 xi2 : ℕ → ℕ)
variables (p1 p2 : ℝ)

def P (ξ : ℕ → ℕ) (x : ℕ) : ℝ := if x = 1 then p1 else 1 - p1

axiom xi1_constraints : 0 < p1 ∧ p1 < 1 / 2
axiom xi2_constraints : p1 < p2 ∧ p2 < 1 / 2

noncomputable def E (ξ : ℕ → ℕ) : ℝ := 1 * P ξ 1 + 0 * P ξ 0

noncomputable def D (ξ : ℕ → ℕ) : ℝ := (1 - E ξ) * P ξ 1 + (0 - E ξ) * P ξ 0

theorem expected_and_variance:
  E xi1 < E xi2 ∧ D xi1 < D xi2 :=
sorry

end expected_and_variance_l809_809725


namespace count_values_of_g50_eq_18_l809_809290

def d (n : ℕ) : ℕ := finset.card (finset.filter (λ (k : ℕ), k > 0 ∧ n % k = 0) (finset.range (n + 1)))
def g1 (n : ℕ) : ℕ := 3 * d n
def g (j : ℕ) (n : ℕ) : ℕ :=
  if j = 1 then g1 n else g1 (g (j - 1) n)

theorem count_values_of_g50_eq_18 : (finset.filter (λ n, g 50 n = 18) (finset.range 51)).card = 3 :=
sorry

end count_values_of_g50_eq_18_l809_809290


namespace distance_to_center_from_point_l809_809296

-- Definitions based on conditions
variable {R : ℝ} (A B D C O : Point)
variable [circle O R]
variable (tangent_line : Tangent A B)
variable (secant_line : Secant A D C)

-- Hypotheses
def key_conditions : Prop :=
  TangentAt O B A B ∧ -- B is the point of tangency
  OnCircle O D ∧ OnCircle O C ∧ -- D and C are on the circle
  Between A D C ∧ -- D is between A and C on the secant
  AngleBisector B D (Angle B A C) ∧ -- BD is the angle bisector
  BD.length = R -- BD has length equal to R

-- Theorem statement
theorem distance_to_center_from_point (h : key_conditions A B D C O tangent_line secant_line) :
  distance A O = (R * sqrt 7) / 2 :=
sorry

end distance_to_center_from_point_l809_809296


namespace prove_range_of_xyz_l809_809220

variable (x y z a : ℝ)

theorem prove_range_of_xyz 
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2 / 2)
  (ha : 0 < a) :
  (0 ≤ x ∧ x ≤ 2 * a / 3) ∧ (0 ≤ y ∧ y ≤ 2 * a / 3) ∧ (0 ≤ z ∧ z ≤ 2 * a / 3) :=
sorry

end prove_range_of_xyz_l809_809220


namespace small_paintings_completed_l809_809218

variable (S : ℕ)

def uses_paint : Prop :=
  3 * 3 + 2 * S = 17

theorem small_paintings_completed : uses_paint S → S = 4 := by
  intro h
  sorry

end small_paintings_completed_l809_809218


namespace num_real_solutions_frac_sine_l809_809658

theorem num_real_solutions_frac_sine :
  (∃ n : ℕ, ∀ x : ℝ, x ∈ Icc (-150) 150 → (x/150 = Real.sin x) ↔ (n = 95)) := 
sorry

end num_real_solutions_frac_sine_l809_809658


namespace sequence_geometric_and_formula_l809_809695

-- Define the sequence a_n based on the given conditions
def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * a n + 1

-- State the theorem to be proven
theorem sequence_geometric_and_formula :
  ∃ a : ℕ → ℝ, sequence a ∧ ∀ n, a n = (1 / 2) * (3^n - 1) :=
by
  sorry

end sequence_geometric_and_formula_l809_809695


namespace first_digits_of_five_sequence_in_two_sequence_reversed_l809_809252

noncomputable def first_digit_of_pow (base : ℕ) (exp : ℕ) : ℕ :=
  (base ^ exp) / 10 ^ (Nat.log10 (base ^ exp))

theorem first_digits_of_five_sequence_in_two_sequence_reversed :
  ∀ (segment : List ℕ),
    (∀ (i : ℕ), ∃ (n : ℕ), segment.get? i = first_digit_of_pow 5 n) →
    ∀ (j : ℕ), ∃ (m : ℕ), segment.reverse.get? j = first_digit_of_pow 2 m := by
  sorry

end first_digits_of_five_sequence_in_two_sequence_reversed_l809_809252


namespace four_digit_numbers_sum_12_div_by_5_l809_809353

theorem four_digit_numbers_sum_12_div_by_5 : 
  (∃ n : ℕ, (n >= 1000 ∧ n < 10000) ∧ (∃ a b c d : ℕ, n = a * 1000 + b * 100 + c * 10 + d ∧ a + b + c + d = 12 ∧ d ∈ {0, 5}) ∧ (n % 5 = 0))
  = 127 := 
sorry

end four_digit_numbers_sum_12_div_by_5_l809_809353


namespace sebastian_total_payment_l809_809839

theorem sebastian_total_payment 
  (cost_per_ticket : ℕ) (number_of_tickets : ℕ) (service_fee : ℕ) (total_paid : ℕ)
  (h1 : cost_per_ticket = 44)
  (h2 : number_of_tickets = 3)
  (h3 : service_fee = 18)
  (h4 : total_paid = (number_of_tickets * cost_per_ticket) + service_fee) :
  total_paid = 150 :=
by
  sorry

end sebastian_total_payment_l809_809839


namespace minimum_value_of_f_range_of_x_l809_809333

noncomputable def f (x : ℝ) := |2*x + 1| + |2*x - 1|

-- Problem 1
theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 2 :=
by
  intro x
  sorry

-- Problem 2
theorem range_of_x (a b : ℝ) (h : |2*a + b| + |a| - (1/2) * |a + b| * f x ≥ 0) : 
  - (1/2) ≤ x ∧ x ≤ 1/2 :=
by
  sorry

end minimum_value_of_f_range_of_x_l809_809333


namespace AB_distance_l809_809894

/-- The side length of the smaller square given its perimeter. --/
def smaller_square_side_length (perimeter : ℝ) : ℝ :=
  perimeter / 4

/-- The side length of the larger square given its area. --/
def larger_square_side_length (area : ℝ) : ℝ :=
  Real.sqrt area

/-- The distance from point A to point B. --/
def distance_AB {perimeter smaller_area larger_area : ℝ}
  (h1: perimeter = 8) (h2 : smaller_area = 2) (h3 : larger_area = 25) : ℝ :=
  let smaller_side := smaller_square_side_length perimeter
  let larger_side := larger_square_side_length larger_area
  Real.sqrt ((smaller_side + larger_side)^2 + (larger_side - smaller_side)^2)

theorem AB_distance {perimeter smaller_area larger_area : ℝ}
  (h1: perimeter = 8) (h2 : smaller_area = 2) (h3 : larger_area = 25) :
  distance_AB h1 h2 h3 = 7.6 :=
  sorry

end AB_distance_l809_809894


namespace walnut_trees_total_l809_809903

variable (current_trees : ℕ) (new_trees : ℕ)

theorem walnut_trees_total (h1 : current_trees = 22) (h2 : new_trees = 55) : current_trees + new_trees = 77 :=
by
  sorry

end walnut_trees_total_l809_809903


namespace branches_after_six_weeks_branches_after_seven_weeks_branches_after_thirteen_weeks_l809_809075

-- Definitions according to problem conditions
inductive TreeGrowth
| first_week : TreeGrowth
| after_two_weeks : TreeGrowth
| continues : TreeGrowth

def num_branches : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 1
| n => num_branches (n - 1) + num_branches (n - 2)

-- Proof Statements for each part
theorem branches_after_six_weeks : num_branches 6 = 8 := by
  sorry

theorem branches_after_seven_weeks : num_branches 7 = 13 := by
  sorry

theorem branches_after_thirteen_weeks : num_branches 13 = 233 := by
  sorry

end branches_after_six_weeks_branches_after_seven_weeks_branches_after_thirteen_weeks_l809_809075


namespace find_three_numbers_l809_809286

theorem find_three_numbers (x : ℤ) (a b c : ℤ) :
  a + b + c = (x + 1)^2 ∧ a + b = x^2 ∧ b + c = (x - 1)^2 ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by {
  sorry
}

end find_three_numbers_l809_809286


namespace smallest_integer_ending_in_9_divisible_by_11_is_99_l809_809153

noncomputable def smallest_positive_integer_ending_in_9_and_divisible_by_11 : ℕ :=
  99

theorem smallest_integer_ending_in_9_divisible_by_11_is_99 :
  ∃ n : ℕ, n > 0 ∧ (n % 10 = 9) ∧ (n % 11 = 0) ∧
          (∀ m : ℕ, m > 0 → (m % 10 = 9) → (m % 11 = 0) → n ≤ m) :=
begin
  use smallest_positive_integer_ending_in_9_and_divisible_by_11,
  split,
  { -- n > 0
    exact nat.zero_lt_bit1 nat.zero_lt_one },
  split,
  { -- n % 10 = 9
    exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { -- n % 11 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 99) },
  { -- ∀ m > 0, m % 10 = 9, m % 11 = 0 → n ≤ m
    intros m hm1 hm2 hm3,
    change 99 ≤ m,
    -- m % 99 = 0 → 99 ≤ m since 99 > 0
    sorry
  }
end

end smallest_integer_ending_in_9_divisible_by_11_is_99_l809_809153


namespace f_of_3_l809_809874

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ (x : ℝ), x ≠ 1 → f x + f ((x + 2) / (2 - 2 * x)) = x^2

theorem f_of_3 : f 3 = 119 / 16 :=
by
  have := f_condition 3 (by norm_num),
  sorry

end f_of_3_l809_809874


namespace maximum_distance_theorem_l809_809761

-- Given conditions
variables {A B C A' B' C' : Type}
variable (distance_eq_one : ∀ {X Y : Type}, (parallel X Y) → (dist X Y = 1))

-- Lengths of the sides
variable (AC BC AB : ℝ)

-- Right angle condition at ∠ABC
variable (right_angle_ABC : ∠ABC = 90)

-- Length values
variables (ac_eq_ten : AC = 10)
variables (ab_eq_eight : AB = 8)
variables (bc_eq_six : BC = 6)

-- Distance from a point on A'B'C' to the sides of ABC
axiom maximum_distance {P : A' B' C'} : (dist P AB) + (dist P BC) + (dist P AC) ≤ 7

theorem maximum_distance_theorem :
  ∀ {P : A' B' C'},
  (dist P AB) + (dist P BC) + (dist P AC) = 7 :=
sorry

end maximum_distance_theorem_l809_809761


namespace sum_of_reciprocals_of_unit_circle_roots_l809_809046

noncomputable def sum_of_reciprocals (a b c d : ℝ) : ℂ :=
  let p : Polynomial ℂ := Polynomial.X ^ 4 + Polynomial.C a * Polynomial.X ^ 3 +
                            Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d
  let roots := (p.roots.map fun z => if z = 0 then 1 else (z ^ -1))
  roots.sum

theorem sum_of_reciprocals_of_unit_circle_roots 
  (a b c d : ℝ)
  (h : ∀ z : ℂ, z ∈ (Polynomial.X ^ 4 + Polynomial.C a * Polynomial.X ^ 3 +
                      Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X +
                      Polynomial.C d).roots → |z| = 1) :
  sum_of_reciprocals a b c d = -↑a := sorry

end sum_of_reciprocals_of_unit_circle_roots_l809_809046


namespace geometric_seq_sum_identity_l809_809714

noncomputable def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, a (n + 1) = q * a n

theorem geometric_seq_sum_identity (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0)
  (hgeom : is_geometric_seq a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end geometric_seq_sum_identity_l809_809714


namespace line_passes_through_1_neg1_l809_809739

-- Define the conditions given in the problem
variables (A B C x y : ℝ)
def condition1 := A - B + C = 0
def line_eq := A * x + B * y + C = 0

-- The theorem to prove that the line passes through the point (1, -1)
theorem line_passes_through_1_neg1 (A B C : ℝ) (h : condition1 A B C) :
  line_eq A B C 1 (-1) :=
by
  assume h : A - B + C = 0
  sorry

end line_passes_through_1_neg1_l809_809739


namespace equations_create_24_l809_809072

theorem equations_create_24 :
  ∃ (eq1 eq2 : ℤ),
  ((eq1 = 3 * (-6 + 4 + 10) ∧ eq1 = 24) ∧ 
   (eq2 = 4 - (-6 / 3) * 10 ∧ eq2 = 24)) ∧ 
   eq1 ≠ eq2 := 
by
  sorry

end equations_create_24_l809_809072


namespace log_inequality_range_l809_809298

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log a (4/5) < 1) → (a ∈ (set.Ioo 0 (4/5)) ∨ a ∈ (set.Ioi 1)) :=
  sorry

end log_inequality_range_l809_809298


namespace dice_labeling_possible_l809_809180

theorem dice_labeling_possible :
  ∃ (A B : set ℕ), (A = {1, 2, 3, 4, 5, 6}) ∧ (B = {0, 6, 12, 18, 24, 30}) ∧
  (∀ n ∈ set.range (λ (a ∈ A) (b ∈ B), a + b), 1 ≤ n ∧ n ≤ 36 ∧ set.Surjective (λ (a ∈ A) (b ∈ B), a + b)) :=
by
  sorry

end dice_labeling_possible_l809_809180


namespace simplest_square_root_l809_809164

-- Defining the options as square roots
def option_A := Real.sqrt 1.5
def option_B := Real.sqrt 3
def option_C := Real.sqrt (5 / 2)
def option_D := Real.sqrt 8

-- Proving that option_B is the simplest
theorem simplest_square_root : option_B = Real.sqrt 3 :=
by
  sorry

end simplest_square_root_l809_809164


namespace cooking_time_remaining_l809_809187

def time_to_cook_remaining (n_total n_cooked t_per : ℕ) : ℕ := (n_total - n_cooked) * t_per

theorem cooking_time_remaining :
  ∀ (n_total n_cooked t_per : ℕ), n_total = 13 → n_cooked = 5 → t_per = 6 → time_to_cook_remaining n_total n_cooked t_per = 48 :=
by
  intros n_total n_cooked t_per h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end cooking_time_remaining_l809_809187


namespace max_xy_l809_809704

theorem max_xy (x y : ℝ) (h1 : x < 0) (h2 : y < 0) (h3 : 3 * x + y = -2) : 
  xy <= 1/3 ∧ (∀ x y, xy = 1/3) :=
by
  sorry

end max_xy_l809_809704


namespace ratio_of_M_to_R_l809_809740

variable (M Q P N R : ℝ)

theorem ratio_of_M_to_R :
      M = 0.40 * Q →
      Q = 0.25 * P →
      N = 0.60 * P →
      R = 0.30 * N →
      M / R = 5 / 9 := by
  sorry

end ratio_of_M_to_R_l809_809740


namespace valid_initial_distributions_count_valid_distributions_l809_809599

theorem valid_initial_distributions (x : Fin 8 → Fin 2) (f : Fin 8 → ℤ) :
  (∀ i, f i = x (Fin.cast_succ i) ∨ f i = 0) →
  ( ∑ i, (choose 7 i : ℤ) • f i % 5 = 0) ↔ ( ∑ i in Finset.range 8, choose 7 i • x i - (1 + 1) * x 0 - 2 * (x 1 + x 6) - x 7 = 0) :=
by
  sorry

theorem count_valid_distributions : 
  ∃ count : ℕ, count = 32 ∧ 
  (∀ x : Fin 8 → Fin 2, ( ∑ i in Finset.range 8, choose 7 i • x i % 5 = 0 ) ↔ (valid_initial_distributions x) → count = 32) :=
by
  sorry

end valid_initial_distributions_count_valid_distributions_l809_809599


namespace largest_base5_three_digit_to_base10_l809_809968

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l809_809968


namespace radius_B_l809_809245

noncomputable def radius_A := 2
noncomputable def radius_D := 4

theorem radius_B (r_B : ℝ) (x y : ℝ) 
  (h1 : (2 : ℝ) + y = x + (x^2 / 4)) 
  (h2 : y = 2 - (x^2 / 8)) 
  (h3 : x = (4: ℝ) / 3) 
  (h4 : y = x + (x^2 / 4)) : r_B = 20 / 9 :=
sorry

end radius_B_l809_809245


namespace inequality_solution_l809_809276

noncomputable def solution_set : Set ℝ :=
  {x : ℝ | x < -2} ∪
  {x : ℝ | -2 < x ∧ x ≤ -1} ∪
  {x : ℝ | 1 ≤ x}

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x + 2)^2 ≥ 0} = solution_set := by
  sorry

end inequality_solution_l809_809276


namespace remainder_of_polynomial_division_l809_809280

def polynomial : ℝ → ℝ := λ x, x^3 - 4 * x^2 + 5 * x - 3

theorem remainder_of_polynomial_division : polynomial 2 = -1 := 
  by sorry

end remainder_of_polynomial_division_l809_809280


namespace flower_shop_l809_809435

-- Define the problem in Lean 4
theorem flower_shop (n_roses : ℕ) (ratio : ℚ) (cost_per_rose : ℚ) (total_spent : ℚ)
  (h_roses : n_roses = 20) (h_ratio : ratio = 3/4) (h_cost_per_rose : cost_per_rose = 5) (h_total_spent : total_spent = 250) :
  let n_lilies := ratio * n_roses,
      cost_per_lily := (total_spent - n_roses * cost_per_rose) / n_lilies in
  cost_per_lily / cost_per_rose = 2 :=
 by
 -- Define and initialize variables
 have n_lilies_def : n_lilies = ratio * n_roses := by rw [h_ratio, h_roses]; exact rfl,
 have cost_roses : n_roses * cost_per_rose = 100 := by rw [h_roses, h_cost_per_rose]; exact rfl,
 have cost_lilies : total_spent - n_roses * cost_per_rose = 150 := by rw [h_total_spent, cost_roses]; exact rfl,
 have cost_per_lily_def : cost_per_lily = (total_spent - n_roses * cost_per_rose) / n_lilies := by exact rfl,
 have cost_per_lily_value : cost_per_lily = 10 := by rw [cost_per_lily_def, cost_lilies, n_lilies_def]; exact rfl,
 have cost_per_rose_value : cost_per_rose = 5 := by exact rfl,
 show cost_per_lily / cost_per_rose = 2 by rw [cost_per_lily_value, cost_per_rose_value]; exact rfl,
 sorry

end flower_shop_l809_809435


namespace tetrahedron_circumsphere_radius_l809_809011

theorem tetrahedron_circumsphere_radius
  (P A B C : Type*)
  [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (PA BC PB AC PC AB : ℝ)
  (hPA : PA = BC)
  (hPB : PB = AC)
  (hPC : PC = AB)
  (hPA_val : PA = √6)
  (hPB_val : PB = √8)
  (hPC_val : PC = √10) :
  ∃ R : ℝ, R = √3 :=
by
  sorry

end tetrahedron_circumsphere_radius_l809_809011


namespace solution_set_x2f_l809_809710

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def inequality_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → (x * f' x + f x) / x^2 > 0

theorem solution_set_x2f (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_f1 : f 1 = 0)
  (h_ineq : inequality_condition f) :
  {x : ℝ | x^2 * f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 1 < x} :=
sorry

end solution_set_x2f_l809_809710


namespace planet_combinations_l809_809733

theorem planet_combinations :
  let a : ℕ := 3 in
  let b : ℕ := 6 in
  let c : ℕ := 4 in
  let d : ℕ := 4 in
  let e : ℕ := 5 in
  let f : ℕ := 2 in
  let combination_a := Nat.choose 5 3 in
  let combination_b := (Nat.choose 5 4) * (Nat.choose 6 4) in
  let combination_c := (Nat.choose 5 5) * (Nat.choose 6 2) in
  let total_combinations := combination_a + combination_b + combination_c in
  total_combinations = 100 :=
by
  sorry

end planet_combinations_l809_809733


namespace infinite_n_divisible_by_p_l809_809833

theorem infinite_n_divisible_by_p (p : ℕ) (hp : Nat.Prime p) : 
  ∃ᶠ n in Filter.atTop, p ∣ (2^n - n) :=
by
  sorry

end infinite_n_divisible_by_p_l809_809833


namespace rectangle_dimensions_length_of_rectangle_l809_809888

theorem rectangle_dimensions (x : ℝ) (h1 : (18 + x)^2 = 28^2 + (10 + x)^2) : 
  28 + 10 + x = 45 :=
by 
  have h2 : 18^2 + 36 * x + x^2 = 784 + 10^2 + 20 * x + x^2 := sorry
  have h3 : 18^2 + 36 * x + x^2 = 784 + 100 + 20 * x + x^2 := sorry
  have h4 : 36 * x = 884 - 324 + 20 * x := sorry
  have h5 : 36 * x - 20 * x = 560 := sorry
  have h6 : 16 * x = 560 := sorry
  have h7 : x = 35 := by sorry
  sorry

theorem length_of_rectangle (x : ℝ) (h1 : 10 + x = 10 + 35) : 
  45 :=
by
  sorry

end rectangle_dimensions_length_of_rectangle_l809_809888


namespace largest_base5_eq_124_l809_809937

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l809_809937


namespace simplify_sqrt_product_l809_809843

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l809_809843


namespace number_of_correct_statements_l809_809882

theorem number_of_correct_statements :
  let x̄ := 5
  let conditions :=
    (∀ x1 x2 ... xn, (2 * x̄ + 1 = 10) → False) ∧
    (∀ c x1 x2 ..., (average (λxi, xi - c) = average (λxi, xi)) → False) ∧ 
    (∀ n1 n2 n3 n4 n5, systematic_sampling [5, 16, 27, 38, 49] n → (n = 60) → False) ∧
    (∀ x, linear_regression 2 (-1.5) x → x ↑ 1 = ↓ 2 y → False)
  in 
    count_correct_statements conditions = 0 :=
sorry

end number_of_correct_statements_l809_809882


namespace frequency_of_fourth_group_l809_809593

theorem frequency_of_fourth_group :
  let n := 50 in
  let f1 := 10 in
  let f2 := 8 in
  let f3 := 11 in
  let f5 := 0.18 * n in
  n - (f1 + f2 + f3 + f5) = 12 :=
by
  let n := 50
  let f1 := 10
  let f2 := 8
  let f3 := 11
  let f5 := 0.18 * n
  have h : n - (f1 + f2 + f3 + f5) = 12 := sorry
  exact h

end frequency_of_fourth_group_l809_809593


namespace copper_price_l809_809499

theorem copper_price (c : ℕ) (hzinc : ℕ) (zinc_weight : ℕ) (brass_weight : ℕ) (price_brass : ℕ) (used_copper : ℕ) :
  hzinc = 30 →
  zinc_weight = brass_weight - used_copper →
  brass_weight = 70 →
  price_brass = 45 →
  used_copper = 30 →
  (used_copper * c + zinc_weight * hzinc) = brass_weight * price_brass →
  c = 65 :=
by
  sorry

end copper_price_l809_809499


namespace sum_of_lengths_geq_inv_k_l809_809502

theorem sum_of_lengths_geq_inv_k 
  (k : ℕ) 
  (M : set (ℝ × ℝ)) 
  (hM : ∀ (a b : ℝ), (a, b) ∈ M → a < b) 
  (h_non_overlap : ∀ (a1 a2 b1 b2 : ℝ), (a1, b1) ∈ M → (a2, b2) ∈ M → (b1 ≤ a2 ∨ b2 ≤ a1) → (a1, b1) ≠ (a2, b2)) 
  (h_unit_seg : ∀ (a b : ℝ), abs (a - b) ≤ 1 → ∃ (x y : ℝ), (x, y) ∈ M ∧ a = x ∧ b = y) :
  ∑ (a b : ℝ) in M.to_finset, (b - a) ≥ 1 / k := sorry

end sum_of_lengths_geq_inv_k_l809_809502


namespace parallelogram_diagonals_properties_l809_809913

open_locale big_operators

variables (A B C D E F M S : Type)
variables [affine_space A] [affine_space B] [affine_space C]
variables [affine_space D] [affine_space E] [affine_space F]
variables [affine_space M] [affine_space S]

-- Given a trapezoid ABCD with AB || CD
axiom trapezoid (A B C D : Type) [affine_space A] [affine_space B] [affine_space C] [affine_space D]
  (AB_parallel_CD : ∃ (a b : line), a ∥ b)

-- Lines are drawn through each vertex of the trapezoid parallel to the diagonals
axiom lines_through_vertices (A B C D : Type) [affine_space A] [affine_space B] [affine_space C] [affine_space D]
  (lines_parallel_diagonals : ∀ (a b c d: line), 
    (a ∥ c) ∧ (b ∥ d))

-- We want to prove that one of the diagonals of the formed parallelogram is parallel to the sides of the trapezoid, 
-- and the other diagonal passes through the intersection point of the non-parallel sides.
theorem parallelogram_diagonals_properties (A B C D : Type) [affine_space A] [affine_space B] [affine_space C] [affine_space D] 
  [affine_space M] [affine_space S] (AB_parallel_CD : ∃ (a b : line), a ∥ b) 
  (lines_parallel_diagonals : ∀ (a b c d: line), 
    (a ∥ c) ∧ (b ∥ d)) :
    (∃ (p : Type), p ∥ line.from_points A B ∧ p ∥ line.from_points C D)
    ∧ 
    (∀ (q : Type), ∃ (intersection : point), intersection ∈ line.from_points A D ∧ intersection ∈ line.from_points B C) :=
sorry

end parallelogram_diagonals_properties_l809_809913


namespace repeating_decimal_fraction_l809_809538

theorem repeating_decimal_fraction :
  let x := (37/100) + (246 / 99900)
  in x = 37245 / 99900 :=
by
  let x := (37/100) + (246 / 99900)
  show x = 37245 / 99900
  sorry

end repeating_decimal_fraction_l809_809538


namespace radius_of_circle_B_l809_809246

theorem radius_of_circle_B :
  ∀ {A B C D : Type} 
  [has_radius A] [has_radius B] [has_radius C] [has_radius D]
  (externally_tangent : A ⟶ B) (externally_tangent_2 : A ⟶ C) (externally_tangent_3 : B ⟶ C)
  (internally_tangent : A ⟶ D) (internally_tangent_2 : B ⟶ D) (internally_tangent_3 : C ⟶ D)
  (congruent_BC : congruent B C)
  (radius_A : radius A = 2)
  (passes_through_center : ∃ F: center D, passes_through_center A F) :
  radius B = 16/9 :=
by
  sorry

end radius_of_circle_B_l809_809246


namespace aprilPriceChange_l809_809758

noncomputable def priceChangeInApril : ℕ :=
  let P0 := 100
  let P1 := P0 + (20 / 100) * P0
  let P2 := P1 - (20 / 100) * P1
  let P3 := P2 + (25 / 100) * P2
  let P4 := P3 - x / 100 * P3
  17

theorem aprilPriceChange (x : ℕ) : x = priceChangeInApril := by
  sorry

end aprilPriceChange_l809_809758


namespace simplify_radical_expression_l809_809856

theorem simplify_radical_expression (x : ℝ) :
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x)) = 120 * x * sqrt (2 * x) := by
sorry

end simplify_radical_expression_l809_809856


namespace solve_sqrt_equation_l809_809648

theorem solve_sqrt_equation (z : ℤ) (h : sqrt (5 - 4 * z) = 7) : z = -11 :=
by
  sorry

end solve_sqrt_equation_l809_809648


namespace frog_final_position_probability_l809_809574

noncomputable def frogJumpProbability : ℝ :=
  let n := 5
  let jumpLength := 1
  let maxDistance := 1
  let directions := (finite_sub (orthonormal_basis_from_matrix (std_matrix 3 3)))^n
  let random_walk := λ n directions, iterate n (λ (pos : ℝ × ℝ × ℝ) dir, pos + (jumpLength * dir)) (0, 0, 0) directions
  let final_position := random_walk n directions
  if (norm final_position ≤ maxDistance) then 1 else 0

theorem frog_final_position_probability : frogJumpProbability = 1 / 10 := 
  sorry

end frog_final_position_probability_l809_809574


namespace set_subtraction_M_N_l809_809702

-- Definitions
def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def B : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }
def M : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

-- Statement
theorem set_subtraction_M_N : (M \ N) = { x | x < 0 } := by
  sorry

end set_subtraction_M_N_l809_809702


namespace inverse_sum_eq_minus_five_l809_809270

def f (x : ℝ) : ℝ := x * abs x

noncomputable def f_inv (y : ℝ) : ℝ :=
if 0 ≤ y then sqrt y else -sqrt (-y)

theorem inverse_sum_eq_minus_five : f_inv 9 + f_inv (-64) = -5 := by
  sorry

end inverse_sum_eq_minus_five_l809_809270


namespace scaleneTriangleDistinctLinesCount_l809_809378

noncomputable def countDistinctLinesInScaleneTriangle (triangle : Type) [Scalene triangle] : Nat :=
  let altitudes := 3
  let medians := 3
  let angleBisectors := 3
  altitudes + medians + angleBisectors

theorem scaleneTriangleDistinctLinesCount (triangle : Type) [Scalene triangle] :
  countDistinctLinesInScaleneTriangle triangle = 9 := by
  sorry

end scaleneTriangleDistinctLinesCount_l809_809378


namespace range_of_m_l809_809301

theorem range_of_m 
    (m : ℝ) (x : ℝ)
    (p : x^2 - 8 * x - 20 > 0)
    (q : (x - (1 - m)) * (x - (1 + m)) > 0)
    (h : ∀ x, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
    0 < m ∧ m ≤ 3 := by
  sorry

end range_of_m_l809_809301


namespace douglas_votes_in_Y_is_46_l809_809176

variable (V : ℝ)
variable (P : ℝ)

def percentage_won_in_Y :=
  let total_voters_X := 2 * V
  let total_voters_Y := V
  let votes_in_X := 0.64 * total_voters_X
  let votes_in_Y := P / 100 * total_voters_Y
  let total_votes := 1.28 * V + (P / 100 * V)
  let combined_voters := 3 * V
  let combined_votes_percentage := 0.58 * combined_voters
  P = 46

theorem douglas_votes_in_Y_is_46
  (V_pos : V > 0)
  (H : 1.28 * V + (P / 100 * V) = 0.58 * 3 * V) :
  percentage_won_in_Y V P := by
  sorry

end douglas_votes_in_Y_is_46_l809_809176


namespace max_value_expr_l809_809707

def point_on_line (m n : ℝ) : Prop :=
  3 * m + n = -1

def mn_positive (m n : ℝ) : Prop :=
  m * n > 0

theorem max_value_expr (m n : ℝ) (h1 : point_on_line m n) (h2 : mn_positive m n) :
  (3 / m + 1 / n) = -16 :=
sorry

end max_value_expr_l809_809707


namespace number_of_diagonals_in_hexagon_l809_809729

-- Define the number of sides of the hexagon
def sides_of_hexagon : ℕ := 6

-- Define the formula for the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem we want to prove
theorem number_of_diagonals_in_hexagon : number_of_diagonals sides_of_hexagon = 9 :=
by
  sorry

end number_of_diagonals_in_hexagon_l809_809729


namespace find_uv_non_integer_l809_809789

noncomputable def q (x y : ℝ) (b : ℕ → ℝ) := 
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_uv_non_integer (b : ℕ → ℝ) 
  (h0 : q 0 0 b = 0) 
  (h1 : q 1 0 b = 0) 
  (h2 : q (-1) 0 b = 0) 
  (h3 : q 0 1 b = 0) 
  (h4 : q 0 (-1) b = 0) 
  (h5 : q 1 1 b = 0) 
  (h6 : q 1 (-1) b = 0) 
  (h7 : q 3 3 b = 0) : 
  ∃ u v : ℝ, q u v b = 0 ∧ u = 17/19 ∧ v = 18/19 := 
  sorry

end find_uv_non_integer_l809_809789


namespace sum_inequality_l809_809808

variable {n : ℕ}
variable (a : ℕ → ℝ)

noncomputable def sum_a_eq_n (h1 : (∀ m, m ≤ n → a m > 0))
  (h2 : (∑ m in finset.range n, a m) = n) :
  ℝ :=
  ∑ m in finset.range n, a m

theorem sum_inequality (h1 : ∀ m, m < n → a m > 0) 
  (h2 : (∑ m in finset.range n, a m) = n) :
  (∑ m in finset.range n, a m /
    (∏ k in finset.range (m + 1), (1 + a k))) ≤ 1 - (1 / 2 ^ n) :=
  sorry

end sum_inequality_l809_809808


namespace mark_should_leave_at_9_am_l809_809083

noncomputable theory

def travel_time_rob : ℕ := 1
def travel_time_mark : ℕ := 3 * travel_time_rob
def rob_leaves_at : ℕ := 11 -- representing 11 a.m. in a 24-hour format

def mark_leaves_at : ℕ := rob_leaves_at - (travel_time_mark - travel_time_rob)

theorem mark_should_leave_at_9_am : mark_leaves_at = 9 := 
by
  sorry

end mark_should_leave_at_9_am_l809_809083


namespace goldfish_added_per_day_is_7_l809_809838

def initial_koi_fish : ℕ := 227 - 2
def initial_goldfish : ℕ := 280 - initial_koi_fish
def added_goldfish : ℕ := 200 - initial_goldfish
def days_in_three_weeks : ℕ := 3 * 7
def goldfish_added_per_day : ℕ := (added_goldfish + days_in_three_weeks - 1) / days_in_three_weeks -- rounding to nearest integer 

theorem goldfish_added_per_day_is_7 : goldfish_added_per_day = 7 :=
by 
-- sorry to skip the proof
sorry

end goldfish_added_per_day_is_7_l809_809838


namespace min_cubes_for_views_l809_809214

-- Definitions: front and side views represented as lists of heights
def front_view := [3, 2, 2]
def side_view := [2, 2, 3]

-- Minimum number of cubes needed to construct the figure with the given views
def min_cubes := 5

-- Main theorem stating the problem
theorem min_cubes_for_views :
  ∀ (figure : list (ℕ × ℕ × ℕ)), 
    (∀ cube ∈ figure, ∃ n, cube = (n, 1, 0) ∨ cube = (1, n, 0) ∨ cube = (1, 0, n) ∧
    (∀ x y z, (x, y, z) ∈ figure → ∃ dx dy dz, (x + dx, y + dy, z + dz) ∈ figure ∨
                                (x - dx, y - dy, z - dz) ∈ figure)) →
    list.length figure ≥ min_cubes :=
by
  intro figure h
  sorry

end min_cubes_for_views_l809_809214


namespace tom_crab_selling_price_l809_809141

theorem tom_crab_selling_price :
  ∀ (crab_buckets daily_crabs per_week_earnings : ℕ),
    crab_buckets = 8 →
    daily_crabs = 12 →
    per_week_earnings = 3360 →
    let crabs_per_day := crab_buckets * daily_crabs in
    let crabs_per_week := crabs_per_day * 7 in
    (per_week_earnings / crabs_per_week) = 5 :=
begin
  intros crab_buckets daily_crabs per_week_earnings h1 h2 h3,
  let crabs_per_day := crab_buckets * daily_crabs,
  let crabs_per_week := crabs_per_day * 7,
  have : crabs_per_week = 672, by calc
    crabs_per_week = crab_buckets * daily_crabs * 7 : by rw [h1, h2]
               ... = 8 * 12 * 7             : rfl
               ... = 672                    : by norm_num,
  rw this at *,
  rw h3,
  norm_num,
end

end tom_crab_selling_price_l809_809141


namespace shaded_region_area_is_48pi_l809_809569

open Real

noncomputable def small_circle_radius : ℝ := 4
noncomputable def small_circle_area : ℝ := π * small_circle_radius^2
noncomputable def large_circle_radius : ℝ := 2 * small_circle_radius
noncomputable def large_circle_area : ℝ := π * large_circle_radius^2
noncomputable def shaded_region_area : ℝ := large_circle_area - small_circle_area

theorem shaded_region_area_is_48pi :
  shaded_region_area = 48 * π := by
    sorry

end shaded_region_area_is_48pi_l809_809569


namespace arithmetic_grid_cell_value_l809_809103

theorem arithmetic_grid_cell_value :
  ∃ d x y z w, 
  (d = 13) ∧
  (x = 74) ∧
  (((0 + d) = 13) ∧ ((0 + 2 * d) = 26) ∧ ((0 + 3 * d) = 39) ∧ ((0 + 4 * d) = 52)) ∧
  ((3 * d + x = 2 * 74) → (148 - 3 * d = y)) ∧
  (y = 109) ∧
  (186 = w) ∧ 
  (w = z) ∧ 
  (z = 186) →

  -- Finally, the value of the cell marked with *.
  (0 + 11 * d = 142) :=
begin
  -- Proof is omitted.
  sorry
end

end arithmetic_grid_cell_value_l809_809103


namespace radius_of_B_l809_809241

theorem radius_of_B {A B C D : Type} (r_A : ℝ) (r_D : ℝ) (r_B : ℝ) (r_C : ℝ)
  (center_A : A) (center_B : B) (center_C : C) (center_D : D)
  (h_cong_BC : r_B = r_C)
  (h_A_D : r_D = 2 * r_A)
  (h_r_A : r_A = 2)
  (h_tangent_A_D : (dist center_A center_D) = r_A) :
  r_B = 32/25 := sorry

end radius_of_B_l809_809241


namespace trigonometric_generalization_l809_809327

theorem trigonometric_generalization
    (α β : ℝ) :
    sin α * sin β = sin^2 ((α + β) / 2) - sin^2 ((α - β) / 2) :=
by
  sorry

end trigonometric_generalization_l809_809327


namespace find_m_range_l809_809722

noncomputable def f (x m : ℝ) : ℝ :=
  if x > -1 ∧ x ≤ 1 then m * real.sqrt (1 - x^2)
  else if x > 1 ∧ x ≤ 3 then 1 - |x - 2|
  else 0

theorem find_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 3 * f x m = x) ∧ m > 0 → m > real.sqrt (15)/3 ∧ m < real.sqrt 7 :=
begin
  sorry
end

end find_m_range_l809_809722


namespace find_x_and_compute_fraction_l809_809669

theorem find_x_and_compute_fraction (x a b c d : ℝ) 
  (h1: a = 4)
  (h2: b = -4)
  (h3: c = 15)
  (h4: d = 7) 
  (h5: x = (4 - 4 * real.sqrt 15) / 7) 
  (h6: 7*x/8 - 1 = 4/x) : 
  a * c * d / b = -105 := 
by sorry

end find_x_and_compute_fraction_l809_809669


namespace latest_start_time_for_liz_l809_809053

def latest_start_time (weight : ℕ) (roast_time_per_pound : ℕ) (num_turkeys : ℕ) (dinner_time : ℕ) : ℕ :=
  dinner_time - (num_turkeys * weight * roast_time_per_pound) / 60

theorem latest_start_time_for_liz : 
  latest_start_time 16 15 2 18 = 10 := by
  sorry

end latest_start_time_for_liz_l809_809053


namespace inverse_var_y_l809_809998

theorem inverse_var_y (k : ℝ) (y x : ℝ)
  (h1 : 5 * y = k / x^2)
  (h2 : y = 16) (h3 : x = 1) (h4 : k = 80) :
  y = 1 / 4 :=
by
  sorry

end inverse_var_y_l809_809998


namespace tangent_line_problem_l809_809711

noncomputable def f (x : ℝ) : ℝ := sorry

theorem tangent_line_problem :
  (∀ (x : ℝ), y = (1 / 2) * x + 2 → y = f(x)) →
  (∀ (x : ℝ), (f'(1) = 1 / 2)) →
  f(1) + f'(1) = 3 :=
by
  intros h1 h2,
  sorry

end tangent_line_problem_l809_809711


namespace number_of_pizzas_ordered_l809_809926

-- Definitions from conditions
def slices_per_pizza : Nat := 2
def total_slices : Nat := 28

-- Proof that the number of pizzas ordered is 14
theorem number_of_pizzas_ordered : total_slices / slices_per_pizza = 14 := by
  sorry

end number_of_pizzas_ordered_l809_809926


namespace smallest_int_ending_in_9_divisible_by_11_l809_809157

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end smallest_int_ending_in_9_divisible_by_11_l809_809157


namespace rahul_matches_played_l809_809447

theorem rahul_matches_played (avg new_avg runs_today : ℕ) 
  (h_average : avg = 46)
  (h_new_average : new_avg = 54)
  (h_runs : runs_today = 78) :
  ∃ m : ℕ, 46 * m + 78 = 54 * (m + 1) ∧ m = 3 :=
by
  use 3
  split
  sorry
  rfl

end rahul_matches_played_l809_809447


namespace filling_sum_l809_809547

/-- Initial sequence on the blackboard: 2, 0, 2, 3.
    Each filling consists of subtracting the left number from the right number 
    for each pair of adjacent numbers and inserting the result between them.
    After 2023 fillings, the sum of all numbers on the blackboard is 2030. -/
theorem filling_sum (init_seq : List ℤ) (num_fillings : ℕ) (final_sum : ℤ) 
  (h_init_seq : init_seq = [2, 0, 2, 3])
  (h_num_fillings : num_fillings = 2023)
  (fill_increase : ∀ (n : ℕ), n ≥ 0 → ∑ b in (init_seq ++ List.repeat 1 n), b = 7 + n) :
  final_sum = 2030 := 
by
  have h_init_sum : ∑ i in init_seq, i = 7 := by simp [h_init_seq]
  have fill_7_n : ∑ i in (init_seq ++ List.repeat 1 2023), i = 7 + 2023 := fill_increase 2023 (by norm_num)
  exact Eq.trans fill_7_n (by norm_num)

end filling_sum_l809_809547


namespace distance_from_focus_to_asymptote_l809_809002

noncomputable def focus_parabola := (0, 2)
def asymptote1 (x : ℝ) := 3 * x
def asymptote2 (x : ℝ) := -3 * x
def distance (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ := |p.2 - l p.1| / real.sqrt (3^2 + 1^2)

theorem distance_from_focus_to_asymptote :
  distance focus_parabola asymptote1 = (real.sqrt 10) / 5 :=
sorry

end distance_from_focus_to_asymptote_l809_809002


namespace points_on_x_axis_circles_intersect_l809_809708

theorem points_on_x_axis_circles_intersect (a b : ℤ)
  (h1 : 3 * a - b = 9)
  (h2 : 2 * a + 3 * b = -5) : (a : ℝ)^b = 1/8 :=
by
  sorry

end points_on_x_axis_circles_intersect_l809_809708


namespace smallest_integer_ending_in_9_divisible_by_11_is_99_l809_809154

noncomputable def smallest_positive_integer_ending_in_9_and_divisible_by_11 : ℕ :=
  99

theorem smallest_integer_ending_in_9_divisible_by_11_is_99 :
  ∃ n : ℕ, n > 0 ∧ (n % 10 = 9) ∧ (n % 11 = 0) ∧
          (∀ m : ℕ, m > 0 → (m % 10 = 9) → (m % 11 = 0) → n ≤ m) :=
begin
  use smallest_positive_integer_ending_in_9_and_divisible_by_11,
  split,
  { -- n > 0
    exact nat.zero_lt_bit1 nat.zero_lt_one },
  split,
  { -- n % 10 = 9
    exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { -- n % 11 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 99) },
  { -- ∀ m > 0, m % 10 = 9, m % 11 = 0 → n ≤ m
    intros m hm1 hm2 hm3,
    change 99 ≤ m,
    -- m % 99 = 0 → 99 ≤ m since 99 > 0
    sorry
  }
end

end smallest_integer_ending_in_9_divisible_by_11_is_99_l809_809154


namespace parallel_vectors_lambda_l809_809699

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 2)

-- Define vector a
def a (λ : ℝ) : ℝ × ℝ := (2, λ)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Main theorem
theorem parallel_vectors_lambda (λ : ℝ) (h : a λ ∥ AB) : λ = 2 / 3 :=
by
  sorry

end parallel_vectors_lambda_l809_809699


namespace p_plus_q_divides_pq_plus_qp_l809_809743

-- Definitions reflecting the conditions 1. and 2.
def is_prime (n : ℕ) : Prop := Nat.Prime n

variables (p q : ℕ)

-- Prime constraints and the relationship between p and q
axiom p_is_prime : is_prime p
axiom q_is_prime : is_prime q
axiom q_equals_p_plus_2 : q = p + 2

-- The main theorem to prove the question
theorem p_plus_q_divides_pq_plus_qp (p q : ℕ)
  [Nat.Prime p] [Nat.Prime q]
  (h : q = p + 2) : (p + q) ∣ (p^q + q^p) := by
sorry

end p_plus_q_divides_pq_plus_qp_l809_809743


namespace extremum_of_function_l809_809494

theorem extremum_of_function : 
  ∀ (x : ℝ), (0 < x) → (∀ (f g : ℝ → ℝ), (f x = x) ∧ (g x = 1 / x) → 
  let y := (f x)^ (g x) in 
  let y' := y * ((g' x) * (ln (f x)) + (g x) * (1 / (f x)) * (f' x)) in 
  y' = x^ (1 / x) * (1 - (ln x)) / (x^ 2) → 
  x = e → y = exp (1 / e)) := 
by 
  sorry

end extremum_of_function_l809_809494


namespace correct_differentiation_option_l809_809162

open Real

theorem correct_differentiation_option :
  ¬((deriv cos x = sin x) ∧ (deriv (λ x => 3^x) x = 3^x) ∧ 
  (deriv (λ x => log x / x) x = (1 - log x) / x) ∧ 
  (deriv (λ x => x * exp x) x = (x + 1) * exp x)) ∧
  deriv (λ x => cos x) x ≠ sin x ∧ 
  deriv (λ x => 3^x) x ≠ 3^x ∧ 
  deriv (λ x => (log x) / x) x ≠ (1 - log x) / x ∧ 
  deriv (λ x => x * exp x) x = (x + 1) * exp x := 
by
  sorry

end correct_differentiation_option_l809_809162


namespace dice_labeling_possible_l809_809179

theorem dice_labeling_possible :
  ∃ (A B : set ℕ), (A = {1, 2, 3, 4, 5, 6}) ∧ (B = {0, 6, 12, 18, 24, 30}) ∧
  (∀ n ∈ set.range (λ (a ∈ A) (b ∈ B), a + b), 1 ≤ n ∧ n ≤ 36 ∧ set.Surjective (λ (a ∈ A) (b ∈ B), a + b)) :=
by
  sorry

end dice_labeling_possible_l809_809179


namespace induction_term_l809_809529

theorem induction_term (k : ℕ) (h : 0 < k) :
  let lhs_k := (list.range k).map (λ i, i + k + 1),
      rhs_k := (2^k) * (list.range k).map (λ i, 2 * i + 1),
      lhs_k_plus_1 := (list.range (k + 1)).map (λ i, i + k + 2),
      rhs_k_plus_1 := (2^(k + 1)) * (list.range (k + 1)).map (λ i, 2 * i + 1)
  in
  lhs_k.prod = rhs_k → lhs_k_plus_1.prod = rhs_k_plus_1 :=
by sorry

end induction_term_l809_809529


namespace problem_l809_809713
noncomputable def a (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b (n : ℕ) : ℕ := 2 * n - 1
def S (n : ℕ) : ℕ := (n * (b 1 + b n)) / 2
noncomputable def c (n : ℕ) : ℚ := 1 / (4 * S n - 1)
noncomputable def T (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), c i

theorem problem : 
  (∀ n, a n = 2^(n-1)) ∧ 
  (∀ n, b n = 2 * n - 1) ∧
  (∀ n, S n = n^2) ∧
  (∀ n, c n = 1 / (4 * S n - 1)) ∧
  (∀ n, T n = n / (2 * n + 1)) := 
sorry

end problem_l809_809713


namespace calculate_c_minus_d_l809_809609

-- Define that g is an invertible function
variable {α β : Type} [Inhabited α] [Inhabited β]
variable (g : α → β) (g_inv : β → α)
variable (h_inv_left : ∀ y, g (g_inv y) = y)
variable (h_inv_right : ∀ x, g_inv (g x) = x)

-- Define the conditions given
variable (c d : α) (h_cd : g c = d) (h_d5 : g d = 5)

-- The theorem statement
theorem calculate_c_minus_d : c - d = -2 := by
  sorry

end calculate_c_minus_d_l809_809609


namespace nested_g_of_2_l809_809044

def g (x : ℤ) : ℤ := x^2 - 4*x + 3

theorem nested_g_of_2 : g (g (g (g (g (g 2))))) = 1394486148248 := by
  sorry

end nested_g_of_2_l809_809044


namespace all_points_on_regression_line_l809_809527

variables {X Y : Type} [LinearOrderedField X] [LinearOrderedField Y] -- Types for variables X and Y
variables (points : List (X × Y))  -- List of sample points
variables (a b : X)  -- Parameters for the linear relationship Y = aX + b

def sum_of_squared_residuals (a b : X) (points : List (X × Y)) : X :=
  points.sum (fun p => let (x, y) := p; (y - (a * x + b))^2)

theorem all_points_on_regression_line
  (h_linear_relationship : ∀ x, x ∈ points -> ∃ a b, ∀ (x y : X), (x, y) ∈ points -> y = a * x + b)
  (h_ssr_zero : sum_of_squared_residuals a b points = 0) :
  ∀ (x y : X), (x, y) ∈ points -> y = a * x + b :=
sorry

end all_points_on_regression_line_l809_809527


namespace range_of_a_condition_l809_809724

noncomputable def f (x : ℝ) : ℝ := x - 1 / (x + 1)
def g (x a : ℝ) : ℝ := x ^ 2 - 2 * a * x + 4

theorem range_of_a_condition (a : ℝ) :
  (∀ x1 ∈ Icc (0 : ℝ) 1, ∃ x2 ∈ Icc (1 : ℝ) 2, f x1 ≥ g x2 a) → a ≥ 9 / 4 :=
by
  -- Proof will be provided here
  sorry

end range_of_a_condition_l809_809724


namespace jensen_miles_city_l809_809828

theorem jensen_miles_city (total_gallons : ℕ) (highway_miles : ℕ) (highway_mpg : ℕ)
  (city_mpg : ℕ) (highway_gallons : ℕ) (city_gallons : ℕ) (city_miles : ℕ) :
  total_gallons = 9 ∧ highway_miles = 210 ∧ highway_mpg = 35 ∧ city_mpg = 18 ∧
  highway_gallons = highway_miles / highway_mpg ∧
  city_gallons = total_gallons - highway_gallons ∧
  city_miles = city_gallons * city_mpg → city_miles = 54 :=
by
  sorry

end jensen_miles_city_l809_809828


namespace melt_brown_fabric_scientific_notation_l809_809824

theorem melt_brown_fabric_scientific_notation :
  0.000156 = 1.56 * 10^(-4) :=
sorry

end melt_brown_fabric_scientific_notation_l809_809824


namespace x_intercept_of_perpendicular_line_l809_809927

-- Define the original line and its perpendicular line with the given y-intercept
def original_line (x y : ℝ) := 4 * x + 5 * y = 20
def perpendicular_line (x y : ℝ) := y = (5/4) * x + 4

-- Prove that the x-intercept of the perpendicular line is -16/5
theorem x_intercept_of_perpendicular_line : 
  ∃ x : ℝ, perpendicular_line x 0 ∧ x = -16/5 :=
by 
  use -16 / 5
  split
  { -- Prove the intercept condition
    sorry },
  { -- Prove the value of x
    refl }

end x_intercept_of_perpendicular_line_l809_809927


namespace largest_base5_eq_124_l809_809938

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l809_809938


namespace length_AB_distance_PQ_l809_809769

noncomputable def parametric_equation_l (t : ℝ) :=
  (x : ℝ, y : ℝ) := (-2 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

def circle_eq (x y : ℝ) := x^2 + (y - 2)^2 = 2

theorem length_AB : 
  ∃ t1 t2 : ℝ, 
  (circle_eq (-2 + (Real.sqrt 3 / 2) * t1) (2 + (1 / 2) * t1)) ∧ 
  (circle_eq (-2 + (Real.sqrt 3 / 2) * t2) (2 + (1 / 2) * t2)) ∧ 
  (t1 + t2 = 2 * Real.sqrt 3) ∧ 
  (t1 * t2 = 2) ∧ 
  (|t1 - t2| = 2) :=
  sorry

def convert_polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := 
  (rho * Real.cos theta, rho * Real.sin theta)

theorem distance_PQ :
  ∀ P Q : ℝ × ℝ, 
  P = convert_polar_to_cartesian (2 * Real.sqrt 2) (3 / 4 * Real.pi) →
  ∃ t1 t2 : ℝ, 
  (circle_eq (-2 + (Real.sqrt 3 / 2) * t1) (2 + (1 / 2) * t1)) ∧ 
  (circle_eq (-2 + (Real.sqrt 3 / 2) * t2) (2 + (1 / 2) * t2)) ∧ 
  let Q := ((-2 + (Real.sqrt 3 / 2) * t1 + (-2 + (Real.sqrt 3 / 2) * t2)) / 2, (2 + (1 / 2) * t1 + (2 + (1 / 2) * t2)) / 2) in
  |((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)| = Real.sqrt 3 :=
  sorry

end length_AB_distance_PQ_l809_809769


namespace necessary_but_not_sufficient_for_q_implies_range_of_a_l809_809096

variable (a : ℝ)

def p (x : ℝ) := |4*x - 3| ≤ 1
def q (x : ℝ) := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem necessary_but_not_sufficient_for_q_implies_range_of_a :
  (∀ x : ℝ, q a x → p x) → (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end necessary_but_not_sufficient_for_q_implies_range_of_a_l809_809096


namespace sum_of_squares_l809_809090

theorem sum_of_squares (n : ℕ) : ∑ k in Finset.range (n + 1), k^2 = n * (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

end sum_of_squares_l809_809090


namespace xiaolong_average_speed_l809_809983

noncomputable def averageSpeed (dist_home_store : ℕ) (time_home_store : ℕ) 
                               (speed_store_playground : ℕ) (time_store_playground : ℕ) 
                               (dist_playground_school : ℕ) (speed_playground_school : ℕ) 
                               (total_time : ℕ) : ℕ :=
  let dist_store_playground := speed_store_playground * time_store_playground
  let time_playground_school := dist_playground_school / speed_playground_school
  let total_distance := dist_home_store + dist_store_playground + dist_playground_school
  total_distance / total_time

theorem xiaolong_average_speed :
  averageSpeed 500 7 80 8 300 60 20 = 72 := by
  sorry

end xiaolong_average_speed_l809_809983


namespace recurring_decimal_to_fraction_l809_809536

theorem recurring_decimal_to_fraction : ∃ x : ℕ, (0.37 + (0.246 / 999)) = (x / 99900) ∧ x = 371874 :=
by
  sorry

end recurring_decimal_to_fraction_l809_809536


namespace vitamin_C_relationship_l809_809826

variables (A O G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + O + G = 275
def condition2 : Prop := 2 * A + 3 * O + 4 * G = 683

-- Rewrite the math proof problem statement
theorem vitamin_C_relationship (h1 : condition1 A O G) (h2 : condition2 A O G) : O + 2 * G = 133 :=
by {
  sorry
}

end vitamin_C_relationship_l809_809826


namespace radius_of_circle_B_l809_809247

theorem radius_of_circle_B :
  ∀ {A B C D : Type} 
  [has_radius A] [has_radius B] [has_radius C] [has_radius D]
  (externally_tangent : A ⟶ B) (externally_tangent_2 : A ⟶ C) (externally_tangent_3 : B ⟶ C)
  (internally_tangent : A ⟶ D) (internally_tangent_2 : B ⟶ D) (internally_tangent_3 : C ⟶ D)
  (congruent_BC : congruent B C)
  (radius_A : radius A = 2)
  (passes_through_center : ∃ F: center D, passes_through_center A F) :
  radius B = 16/9 :=
by
  sorry

end radius_of_circle_B_l809_809247


namespace range_of_a_l809_809313

variables {a : ℝ}

def p := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x
def q := ∃ x : ℝ, x^2 - 4 * x + a ≤ 0

theorem range_of_a (hpq : p ∧ q) : Real.exp 1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l809_809313


namespace additional_element_is_H_l809_809279

noncomputable def Br_weight : ℝ := 79.904
noncomputable def O_weight : ℝ := 15.999
noncomputable def BrO3_weight : ℝ := Br_weight + 3 * O_weight
noncomputable def total_weight : ℝ := 129
noncomputable def additional_element_weight : ℝ := total_weight - BrO3_weight

theorem additional_element_is_H : additional_element_weight ≈ 1.008 :=
by
  let H_weight : ℝ := 1.008
  -- Given the definitions and the atomic weight of Hydrogen, the proof follows.
  sorry

end additional_element_is_H_l809_809279


namespace sum_sequence_six_l809_809795

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry

theorem sum_sequence_six :
  (∀ n, S n = 2 * a n + 1) → S 6 = 63 :=
by
  sorry

end sum_sequence_six_l809_809795


namespace range_of_a_l809_809099

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|4 * x - 3| ≤ 1)) → 
  (∀ x : ℝ, (x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
begin
  sorry
end

end range_of_a_l809_809099


namespace number_of_solutions_l809_809744

noncomputable def log2 := Real.log2

def f (x : ℝ) : ℝ :=
  log2 (log2 (2 * x + 2)) + 2 ^ (2 * x + 2)

theorem number_of_solutions :
  (finset.filter (fun x => ∃ n : ℤ, f x = n) (finset.Icc 0 1)).card = 14 :=
sorry

end number_of_solutions_l809_809744


namespace triangle_isosceles_l809_809754

theorem triangle_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) :
  b = c → IsoscelesTriangle := 
by
  sorry

end triangle_isosceles_l809_809754


namespace value_of_expression_l809_809490

noncomputable def line_does_not_pass_through_third_quadrant (k b : ℝ) : Prop :=
k < 0 ∧ b ≥ 0

theorem value_of_expression 
  (k b a e m n c d : ℝ) 
  (h_line : line_does_not_pass_through_third_quadrant k b)
  (h_a_gt_e : a > e)
  (hA : a * k + b = m)
  (hB : e * k + b = n)
  (hC : -m * k + b = c)
  (hD : -n * k + b = d) :
  (m - n) * (c - d) ^ 3 > 0 :=
sorry

end value_of_expression_l809_809490


namespace solve_r_l809_809738

theorem solve_r (k r : ℝ) (h1 : 3 = k * 2^r) (h2 : 15 = k * 4^r) : 
  r = Real.log 5 / Real.log 2 := 
sorry

end solve_r_l809_809738


namespace four_digit_numbers_sum_12_div_by_5_l809_809352

theorem four_digit_numbers_sum_12_div_by_5 : 
  (∃ n : ℕ, (n >= 1000 ∧ n < 10000) ∧ (∃ a b c d : ℕ, n = a * 1000 + b * 100 + c * 10 + d ∧ a + b + c + d = 12 ∧ d ∈ {0, 5}) ∧ (n % 5 = 0))
  = 127 := 
sorry

end four_digit_numbers_sum_12_div_by_5_l809_809352


namespace correct_option_is_C_l809_809979

theorem correct_option_is_C (A B C D : Prop) : 
  (A ↔ \sqrt{9} = \pm 3) ∧ 
  (B ↔ \sqrt{(-2)^2} = -2) ∧ 
  (C ↔ \sqrt[3]{-125} = -5) ∧ 
  (D ↔ \pm\sqrt{16} = 4) → 
  (∃ c : Prop, c = C) ∧ (∀ a : Prop, a ≠ C) :=
by
  sorry

end correct_option_is_C_l809_809979


namespace zeros_of_f_in_interval_l809_809254

noncomputable def f : ℝ → ℝ :=
  λ x => if x ∈ Icc (-1:ℝ) 4 then x^2 - 2*x else sorry

theorem zeros_of_f_in_interval :
  (∀ x : ℝ, f(x) + f(x + 5) = 16) ∧ 
  (∀ x : ℝ, x ∈ Icc (-1) 4 -> f(x) = x^2 - 2*x) ∧ 
  (∃ num_zeros : ℕ, num_zeros = 604 ∧
    (∀ a b : ℝ, (0 ≤ a ∧ b ≤ 2013) →
    { x : ℝ | f x = 0 ∧ a ≤ x ∧ x ≤ b }.card = num_zeros)) :=
sorry

end zeros_of_f_in_interval_l809_809254


namespace ferris_wheel_travel_time_19_seconds_l809_809195

noncomputable def ferris_wheel_time_to_height (R : ℝ) (T : ℝ) (h : ℝ) : ℝ :=
  let A := R
  let D := R
  let B := real.pi / T
  (30 / real.pi) * real.acos ((h - D) / A)

theorem ferris_wheel_travel_time_19_seconds :
  ferris_wheel_time_to_height 25 60 15 ≈ 19 :=
by
  sorry

end ferris_wheel_travel_time_19_seconds_l809_809195


namespace factorization_of_polynomial_l809_809269

-- Given conditions:
variables {α β : Type*} [field α] [field β] (a b c : α) (ω : β)
def omega_conditions : Prop := ω^3 = 1 ∧ 1 + ω + ω^2 = 0

-- Problem statement:
theorem factorization_of_polynomial (h : omega_conditions ω) : 
  (a^3 + b^3 + c^3 - 3 * a * b * c) = 
  (a + b + c) * (a + ω * b + ω^2 * c) * 
  (a + ω^2 * b + ω * c) :=
sorry

end factorization_of_polynomial_l809_809269


namespace conference_messages_l809_809883

universe u

variables {N F : Type u} 
variables (messages_to_foreign : N → F) (messages_to_native : F → N)
variable [finite : nonempty_fintype N]

theorem conference_messages 
  (h1 : ∃ x : N, ∀ y : F, messages_to_native y ≠ x) : 
  ∃ S : set N, 
    (∀ x ∈ S, ∀ y : F, messages_to_native y ≠ x) ∧
    disjoint S { x | ∃ y : F, messages_to_native y = x ∧ messages_to_foreign x ∉ S} ∧
    S ∪ { x | ∃ y : F, messages_to_native y = x ∧ messages_to_foreign x ∉ S} = @set.univ N :=
by
  sorry

end conference_messages_l809_809883


namespace sum_of_a_b_l809_809885

theorem sum_of_a_b (a b : ℝ) (h1 : a * b = 1) (h2 : (3 * a + 2 * b) * (3 * b + 2 * a) = 295) : a + b = 7 :=
by
  sorry

end sum_of_a_b_l809_809885


namespace current_intensity_solution_l809_809899

noncomputable def current_intensity 
  (n : ℕ)                -- Number of cells
  (E : ℝ)                -- Electromotive force of one cell in Volts
  (R_b : ℝ)              -- Internal resistance of one cell in Ohms
  (d_wire : ℝ)           -- Distance to Miskolc in km
  (D_wire : ℝ)           -- Diameter of iron wire in mm
  (R_unit : ℝ)           -- Resistance of a 1 mm length and 1 mm diameter iron wire in Ohms
  (d_morse : ℝ)          -- Equivalent distance of Morse machine in km
: ℝ :=
let R_wire := (d_wire + d_morse) * 10^3 * R_unit / (let D_base := 1 in (D_wire / D_base)^2), -- Total resistance of the wire
    R_total := n * R_b + R_wire, -- Total resistance considering internal and external resistances
    V_total := n * E -- Total voltage
in
V_total / R_total

#eval current_intensity 60 1.079 0.62 79 5 0.2 16 -- Should approximately match 0.572

theorem current_intensity_solution 
  (n : ℕ)                  -- Number of cells
  (E : ℝ)                  -- Electromotive force of one cell in Volts
  (R_b : ℝ)                -- Internal resistance of one cell in Ohms
  (d_wire : ℝ)             -- Distance to Miskolc in km
  (D_wire : ℝ)             -- Diameter of iron wire in mm
  (R_unit : ℝ)             -- Resistance of a 1 mm length and 1 mm diameter iron wire in Ohms
  (d_morse : ℝ)            -- Equivalent distance of Morse machine in km
  (h_n : n = 60)           -- Number of cells is 60
  (h_e : E = 1.079)        -- Electromotive force is 1.079 Volts
  (h_rb : R_b = 0.62)      -- Internal resistance is 0.62 Ohms
  (h_d_wire : d_wire = 79) -- Distance to Miskolc is 79 km
  (h_d_wire_d : D_wire = 5)-- Diameter of iron wire is 5 mm
  (h_r_unit : R_unit = 0.2)-- Resistance of iron wire is 0.2 Ohms
  (h_d_morse : d_morse = 16)-- Morse machine equivalent distance is 16 km
  : current_intensity n E R_b d_wire D_wire R_unit d_morse ≈ 0.572 := sorry

end current_intensity_solution_l809_809899


namespace parabola_focus_l809_809109

-- Define the equation of the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = -8 * x

-- Define the coordinates of the focus
def focus (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- The Lean statement that needs to be proved
theorem parabola_focus : ∀ (x y : ℝ), parabola_eq x y → focus x y :=
by
  intros x y h
  sorry

end parabola_focus_l809_809109


namespace most_suitable_for_census_l809_809545

-- Define the options
inductive Survey
| quality_of_batch_of_milk : Survey
| sleep_patterns_of_students : Survey
| chang_e_5_components : Survey
| water_quality_of_yangtze : Survey

-- Define the question: Which survey is suitable for a census?
def is_census_suitable (s : Survey) : Prop :=
  s = Survey.chang_e_5_components

-- Statement of the proof problem
theorem most_suitable_for_census : is_census_suitable Survey.chang_e_5_components :=
by {
  -- Add statements indicating other options are not suitable for census
  have hA : ¬ is_census_suitable Survey.quality_of_batch_of_milk := sorry,
  have hB : ¬ is_census_suitable Survey.sleep_patterns_of_students := sorry,
  have hD : ¬ is_census_suitable Survey.water_quality_of_yangtze := sorry,
  -- The correct option is Survey.chang_e_5_components
  exact rfl
}

end most_suitable_for_census_l809_809545


namespace trapezoid_diagonal_length_l809_809781

theorem trapezoid_diagonal_length (AB BC CD DA : ℝ) (h1 : AB = 7) (h2 : BC = 19) (h3 : CD = 7) (h4 : DA = 11) (h5 : ∠BCD = 90) : BD = 20 := by
  sorry

end trapezoid_diagonal_length_l809_809781


namespace calculation_1_calculation_2_calculation_3_calculation_4_l809_809237

theorem calculation_1 :
  (3 + 13/15 - (2 + 13/14) + 5 + 2/15 - (1 + 1/14) = 5) :=
sorry

theorem calculation_2 :
  (1/9 ÷ (2 ÷ (3/4 - 2/3)) = 1/216) :=
sorry

theorem calculation_3 :
  (99 * 78.6 + 786 * 0.3 - 7.86 * 20 = 7860) :=
sorry

theorem calculation_4 :
  (2015 ÷ (2015 * (1 + 1/2016)) = 2016/2017) :=
sorry

end calculation_1_calculation_2_calculation_3_calculation_4_l809_809237


namespace coordinates_in_new_basis_l809_809364

def vector_in_new_basis 
  (a b c p : ℝ^3) 
  (ha : p = (1 * a + 3 * b + 2 * c)) 
  (new_basis_1 : ℝ^3 := (a + b)) 
  (new_basis_2 : ℝ^3 := (a - b)) 
  (new_basis_3 : ℝ^3 := c) : Prop :=
  ∃ (x y z : ℝ), p = (x * new_basis_1 + y * new_basis_2 + z * new_basis_3) 
  ∧ (x = 2 ∧ y = -1 ∧ z = 2)

theorem coordinates_in_new_basis 
  (a b c p : ℝ^3) 
  (ha : p = (1 * a + 3 * b + 2 * c)) : 
  vector_in_new_basis a b c p ha :=
sorry

end coordinates_in_new_basis_l809_809364


namespace transformed_area_l809_809414

variables {α β γ : Type*} [Field α] [AddCommGroup β] [module α β] [linear_order α]

def is_triangle_area (area : α) (x1 y1 x2 y2 x3 y3 : α) : Prop :=
  ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs / 2 = area

variables {x_a x_b x_c : α} {g : α → β}

-- Initial area assumption
axiom initial_area : 
  is_triangle_area 50 
                   x_a (g x_a) 
                   x_b (g x_b) 
                   x_c (g x_c)

-- Proof of the area under transformation y = 3g(3x)
theorem transformed_area :
  is_triangle_area 50 
                   (x_a / 3) (3 * g x_a)
                   (x_b / 3) (3 * g x_b)
                   (x_c / 3) (3 * g x_c) :=
sorry

end transformed_area_l809_809414


namespace petya_mistake_l809_809898

def astonishing_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [8, 0, 3] → (∃ k : ℕ, n = 2023 * digit_sum d k)

theorem petya_mistake (a : ℕ) (h : a > 3) : 
  ¬ astonishing_number (a^2 - 12) :=
by
  sorry

end petya_mistake_l809_809898


namespace largest_base5_three_digit_to_base10_l809_809967

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l809_809967


namespace solve_equation_l809_809095

theorem solve_equation (x : ℝ) : 32 = 2 * 16^(x-2) → x = 3 :=
by
  intros h
  sorry

end solve_equation_l809_809095


namespace correct_option_B_l809_809978

theorem correct_option_B (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := 
  sorry

end correct_option_B_l809_809978


namespace recurring_decimal_to_fraction_l809_809535

theorem recurring_decimal_to_fraction : ∃ x : ℕ, (0.37 + (0.246 / 999)) = (x / 99900) ∧ x = 371874 :=
by
  sorry

end recurring_decimal_to_fraction_l809_809535


namespace domain_of_h_l809_809461

variable {α : Type} [LinearOrder α] (f : α → α)
variable (x : α)
noncomputable def h (x : α) := f (3 * x)

theorem domain_of_h (f : ℝ → ℝ)
  (domain_f : ∀ x, -6 ≤ x ∧ x ≤ 9 → f x = f x) :
  ∀ x, -2 ≤ x ∧ x ≤ 3 ↔ ∃ y, y = 3 * x ∧ y ∈ Icc (-6 : ℝ) (9 : ℝ) := 
by
  intro x
  constructor
  case mp =>
    intro hx
    use 3 * x
    sorry
  case mpr =>
    intro h
    obtain ⟨y, hy⟩ := h
    rw ← hy
    split
    · linarith
    · linarith

end domain_of_h_l809_809461


namespace gordon_bookstore_cost_l809_809515

def price (p : ℝ) (d : ℝ) : ℝ := p - (p * d)

def total_cost (books : List ℝ) (d_over_22 : ℝ) (d_under_20 : ℝ) : ℝ :=
  (books.filter (λ p => p > 22)).map (λ p => price p d_over_22) |>.sum
  + (books.filter (λ p => p < 20)).map (λ p => price p d_under_20) |>.sum
  + (books.filter (λ p => p == 21)).sum

theorem gordon_bookstore_cost :
  let books := [25.0, 18.0, 21.0, 35.0, 12.0, 10.0]
  let d_over_22 := 0.30
  let d_under_20 := 0.20
  total_cost books d_over_22 d_under_20 = 95.00 := 
by
  sorry

end gordon_bookstore_cost_l809_809515


namespace smallest_N_divisible_l809_809534

theorem smallest_N_divisible (N x : ℕ) (H: N - 24 = 84 * Nat.lcm x 60) : N = 5064 :=
by
  sorry

end smallest_N_divisible_l809_809534


namespace chocolate_squares_Jenny_ate_l809_809021

variable (Mike_chocolate_squares : ℕ)
variable (Mike_candies : ℕ)
variable (Mike_friend_candies : ℕ)
variable (Jenny_chocolates : ℕ)
variable (Jenny_candies : ℕ)

-- Conditions
def Mike_ate_20_chocolates (h1 : Mike_chocolate_squares = 20) : Prop := 
  Mike_chocolate_squares = 20

def Mike_friend_ate_10_less_candies (h2 : Mike_friend_candies = Mike_candies - 10) : Prop := 
  Mike_friend_candies = Mike_candies - 10

def Jenny_ate_5_more_than_thrice_Mike (h3 : Jenny_chocolates = 3 * Mike_chocolate_squares + 5) : Prop := 
  Jenny_chocolates = 3 * Mike_chocolate_squares + 5

def Jenny_ate_twice_candies_as_friend (h4 : Jenny_candies = 2 * Mike_friend_candies) : Prop := 
  Jenny_candies = 2 * Mike_friend_candies

-- Proof statement
theorem chocolate_squares_Jenny_ate 
  (h1 : Mike_ate_20_chocolates Mike_chocolate_squares)
  (h2 : Mike_friend_ate_10_less_candies Mike_friend_candies Mike_candies)
  (h3 : Jenny_ate_5_more_than_thrice_Mike Jenny_chocolates Mike_chocolate_squares)
  (h4 : Jenny_ate_twice_candies_as_friend Jenny_candies Mike_friend_candies)
  (h5 : Mike_candies = 20) (h6 : Mike_friend_candies = 10) :
  Jenny_chocolates = 65 := 
  by
    sorry

end chocolate_squares_Jenny_ate_l809_809021


namespace find_k_for_binomial_square_l809_809190

theorem find_k_for_binomial_square :
  ∃ k : ℝ, (∀ x : ℝ, x^2 - 20 * x + k = (x - 10) ^ 2) ∧ k = 100 :=
by
  use 100
  split
  · intro x
    calc
      x^2 - 20 * x + 100
      = x^2 - 20 * x + 10^2 : by rw [pow_two]
      ... = (x - 10) ^ 2     : by rw [sub_self];
        sorry

end find_k_for_binomial_square_l809_809190


namespace athlete_target_heart_rate_30_years_old_l809_809604

def target_heart_rate (age : ℕ) (consumed_caffeine : Bool) : ℕ :=
  let max_heart_rate := if consumed_caffeine then (220 - age) * 11 / 10 else (220 - age)
  Float.toNat (0.85 * max_heart_rate).toFloat

theorem athlete_target_heart_rate_30_years_old :
  target_heart_rate 30 true = 178 :=
by
  sorry

end athlete_target_heart_rate_30_years_old_l809_809604


namespace john_total_spent_l809_809028

noncomputable def computer_cost : ℝ := 1500
noncomputable def peripherals_cost : ℝ := (1 / 4) * computer_cost
noncomputable def base_video_card_cost : ℝ := 300
noncomputable def upgraded_video_card_cost : ℝ := 2.5 * base_video_card_cost
noncomputable def discount_on_video_card : ℝ := 0.12 * upgraded_video_card_cost
noncomputable def video_card_cost_after_discount : ℝ := upgraded_video_card_cost - discount_on_video_card
noncomputable def sales_tax_on_peripherals : ℝ := 0.05 * peripherals_cost
noncomputable def total_spent : ℝ := computer_cost + peripherals_cost + video_card_cost_after_discount + sales_tax_on_peripherals

theorem john_total_spent : total_spent = 2553.75 := by
  sorry

end john_total_spent_l809_809028


namespace angle_QRS_determination_l809_809376

theorem angle_QRS_determination (PQ_parallel_RS : ∀ (P Q R S T : Type) 
  (angle_PTQ : ℝ) (angle_SRT : ℝ), 
  PQ_parallel_RS → (angle_PTQ = angle_SRT) → (angle_PTQ = 4 * angle_SRT - 120)) 
  (angle_SRT : ℝ) (angle_QRS : ℝ) 
  (h : angle_SRT = 4 * angle_SRT - 120) : angle_QRS = 40 :=
by 
  sorry

end angle_QRS_determination_l809_809376


namespace trig_identity_l809_809985

theorem trig_identity (α : ℝ) (h : sin α ^ 2 + cos α ^ 2 = 1) : 
  sin α ^ 6 + cos α ^ 6 + 3 * sin α ^ 2 * cos α ^ 2 = 1 :=
by
  sorry

end trig_identity_l809_809985


namespace tenth_term_sequence_l809_809465

-- Define the sequence given its general formula
def sequence (n : ℕ) : ℚ := 1 / (n + 2)

-- Prove that the 10th term of the sequence is 1/12
theorem tenth_term_sequence : sequence 10 = 1 / 12 :=
by
  sorry

end tenth_term_sequence_l809_809465


namespace classical_prob_l809_809980

def classical_probability_model_correct_statements (finite_events : Prop) (elementary_events_eq_prob : Prop) 
(event_prob_formula : ∀ {n k : ℕ}, P (event A) = k / n) : Prop :=
finite_events ∧ elementary_events_eq_prob ∧ event_prob_formula

theorem classical_prob (finite_events elementary_events_eq_prob event_prob_formula) :
  classical_probability_model_correct_statements finite_events elementary_events_eq_prob event_prob_formula :=
by
  sorry

end classical_prob_l809_809980


namespace negative_binomial_expectation_tau_l809_809471

variable (p q : ℝ) (h0 : q = 1 - p) (h1 : 0 < p) (h2 : p ≤ 1)

def bernoulli (ξ : ℕ → ℤ) (n : ℕ) : Prop := 
  (∀ k, ξ k = 1 ∨ ξ k = 0) ∧ 
  (∀ k, (ξ k = 1 → ℙ (ξ k = 1) = p) ∧ (ξ k = 0 → ℙ (ξ k = 0) = q))

def sum_trials (ξ : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in finset.range n, ξ i

theorem negative_binomial (ξ : ℕ → ℤ)
  (h_bernoulli : bernoulli p q ξ)
  (r k : ℕ) (hrk : r ≤ k) :
  ℙ (sum_trials ξ (k - 1) = r - 1 ∧ ξ k = 1) = nat.choose (k - 1) (r - 1) * p ^ r * q ^ (k - r) := sorry

theorem expectation_tau (ξ : ℕ → ℤ)
  (h_bernoulli : bernoulli p q ξ)
  (r : ℕ) :
  E (λ k, if sum_trials ξ (k - 1) = r - 1 ∧ ξ k = 1 then k else 0) = r / p := sorry

end negative_binomial_expectation_tau_l809_809471


namespace solution_set_of_inequality_system_l809_809895

theorem solution_set_of_inequality_system :
  (6 - 2 * x ≥ 0) ∧ (2 * x + 4 > 0) ↔ (-2 < x ∧ x ≤ 3) := 
sorry

end solution_set_of_inequality_system_l809_809895


namespace calculation_solve_system_l809_809558

-- Part (1)
theorem calculation : (cbrt (-8) - real.sqrt 2 + (real.sqrt 3) ^ 2 + abs (1 - real.sqrt 2) - (-1) ^ 2023) = 1 := 
by sorry

-- Part (2)
theorem solve_system (x y : ℝ) (h₁ : (1 / 2) * x - (3 / 2) * y = -1) (h₂ : 2 * x + y = 3) : (x = 1) ∧ (y = 1) := 
by sorry

end calculation_solve_system_l809_809558


namespace travel_cost_minimized_l809_809821

noncomputable def distance_de (distance_df distance_ef : ℕ) : ℕ :=
  (Math.sqrt (distance_ef^2 - distance_df^2)).to_nat

def cost_by_airplane (distance : ℕ) : ℕ :=
  120 + distance * 12 / 100

def cost_by_bus (distance : ℕ) : ℕ :=
  distance * 20 / 100

def min_cost (cost1 cost2 : ℕ) : ℕ :=
  if cost1 < cost2 then cost1 else cost2

def total_min_travel_cost (distance_df distance_ef : ℕ) : ℕ :=
  let distance_de := distance_de distance_df distance_ef
  let cost_de := min_cost (cost_by_airplane distance_de) (cost_by_bus distance_de)
  let cost_ef := min_cost (cost_by_airplane distance_ef) (cost_by_bus distance_ef)
  let cost_fd := min_cost (cost_by_airplane distance_df) (cost_by_bus distance_df)
  cost_de + cost_ef + cost_fd

theorem travel_cost_minimized : total_min_travel_cost 4000 4500 = 1680 :=
  by
    sorry

end travel_cost_minimized_l809_809821


namespace arrangement_volunteers_l809_809089

theorem arrangement_volunteers (A B : Finset ℕ) (hA : A.card = 3) (hB : B.card = 3) :
  (∀ row : Finset ℕ, row.card = 3 → ∀ x y ∈ row, x ∈ A ∧ y ∈ A ∨ x ∈ B ∧ y ∈ B → x ≠ y) →
  (∃ arr : ℕ, arr = 72) :=
sorry

end arrangement_volunteers_l809_809089


namespace hexagon_area_correct_problem_solution_l809_809030

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  8 * b + 8 * real.sqrt (b ^ 2 - 12)

noncomputable def b_value : ℝ := 10 / real.sqrt 3

theorem hexagon_area_correct :
  hexagon_area b_value = 48 * real.sqrt 3 :=
by
  sorry

def m : ℕ := 48
def n : ℕ := 3

theorem problem_solution :
  m + n = 51 :=
by
  rw [m, n]
  norm_num

end hexagon_area_correct_problem_solution_l809_809030


namespace remainder_x1012_div_x2p1_xm1_l809_809666

theorem remainder_x1012_div_x2p1_xm1 :
  let x := Polynomial ℤ,
      dividend := x^1012,
      divisor := (x^2 + 1) * (x - 1)
  in dividend % divisor = 1 :=
by
  sorry

end remainder_x1012_div_x2p1_xm1_l809_809666


namespace count_six_digit_integers_l809_809432

def is_valid_six_digit_integer (n : Nat) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ 
  -- Convert Nat to List of digits
  let digits := to_digits n in
  -- Check that exactly 2 digits are 3
  digits.count 3 = 2 ∧ 
  -- Check if removing 3s results in 2022
  to_nat(digits.filter (≠ 3)) = 2022

theorem count_six_digit_integers : 
  (Finset.filter is_valid_six_digit_integer (Finset.range 1000000)).card = 15 :=
sorry

-- Helper functions to convert Nat to List of digits and back
def to_digits (n : Nat) : List Nat := sorry
def to_nat (digits : List Nat) : Nat := sorry

end count_six_digit_integers_l809_809432


namespace solve_for_x_l809_809456

theorem solve_for_x (x : ℝ) : 3^x + 12 = 5 * 3^x - 40 → x = Real.log 13 / Real.log 3 := by
  sorry

end solve_for_x_l809_809456


namespace quantities_purchased_min_discount_percentage_l809_809466

section
variables (x y : ℕ) 

-- Conditions for Part 1
def cost_eq : 40 * x + 30 * y = 25000 := by sorry
def profit_eq : 18 * x + 15 * y = 11700 := by sorry

-- Proof of quantities
theorem quantities_purchased : x = 400 ∧ y = 300 :=
begin
  -- prove that x = 400 and y = 300 satisfy the equations
  have h1 : cost_eq x y := sorry,
  have h2 : profit_eq x y := sorry,
  sorry
end

-- Variables and constraints for Part 2
variables (m : ℕ)
def ornaments_purchased : x = 400 := by sorry
def pendants_purchased : y = 300 := by sorry
def new_pendants_purchased : 2 * y = 600 := by sorry
def profit_constraint : 7200 + 600 * m - 18000 ≥ 10800 := by sorry

-- Proof for minimum discount percentage
theorem min_discount_percentage : m ≥ 36 :=
begin
  -- prove that the minimum m value that satisfies the profit constraint is 36
  have h1 : new_pendants_purchased y := sorry,
  have h2 : profit_constraint m := sorry,
  sorry
end

end

end quantities_purchased_min_discount_percentage_l809_809466


namespace lenny_pens_left_l809_809394

def total_pens (boxes : ℕ) (pens_per_box : ℕ) : ℕ := boxes * pens_per_box

def pens_to_friends (total : ℕ) (percentage : ℚ) : ℚ := total * percentage

def remaining_after_friends (total : ℕ) (given : ℚ) : ℚ := total - given

def pens_to_classmates (remaining : ℚ) (fraction : ℚ) : ℚ := remaining * fraction

def final_remaining (remaining : ℚ) (given : ℚ) : ℚ := remaining - given

theorem lenny_pens_left :
  let total := total_pens 20 5 in
  let given_to_friends := pens_to_friends total (40 / 100) in
  let remaining1 := remaining_after_friends total given_to_friends in
  let given_to_classmates := pens_to_classmates remaining1 (1 / 4) in
  let remaining2 := final_remaining remaining1 given_to_classmates in
  remaining2 = 45 :=
by
  sorry

end lenny_pens_left_l809_809394


namespace sufficient_condition_l809_809402

variable {α : Type*} (A B : Set α)

theorem sufficient_condition (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
by
  sorry

end sufficient_condition_l809_809402


namespace math_proof_problem_l809_809238

noncomputable def a : ℝ := Real.sqrt 18
noncomputable def b : ℝ := (-1 / 3) ^ (-2 : ℤ)
noncomputable def c : ℝ := abs (-3 * Real.sqrt 2)
noncomputable def d : ℝ := (1 - Real.sqrt 2) ^ 0

theorem math_proof_problem : a - b - c - d = -10 := by
  -- Sorry is used to skip the proof, as the proof steps are not required for this problem.
  sorry

end math_proof_problem_l809_809238


namespace meaning_of_negative_distance_l809_809746

theorem meaning_of_negative_distance (distance : ℝ) (positive_direction : Prop) 
  (h : positive_direction ↔ "east") (d : distance = -50) : 
  "movement in the opposite direction" = "west" :=
sorry

end meaning_of_negative_distance_l809_809746


namespace problem_sol_l809_809660

open Real Trig

noncomputable def numberOfRealSolutions : ℝ := 95

theorem problem_sol : ∃ x : ℝ, -150 ≤ x ∧ x ≤ 150 ∧ (x / 150 = sin x) ∧ finset.card (finset.filter (λx, x / 150 = sin x) (finset.range 301 - 151)) = 95 :=
by
  sorry

end problem_sol_l809_809660


namespace count_base8_numbers_with_4_or_6_l809_809732

theorem count_base8_numbers_with_4_or_6 :
  let n := 8 in 
  let m := n^4 in
  let restricted_digits := {0, 1, 2, 3, 5, 7} in
  let qualifying_count := m - (6^ 4) in
  qualifying_count = 1105
  := by
  -- Definitions
  let n := 8
  let m := n^4
  let restricted_digits := {0, 1, 2, 3, 5, 7}
  let qualifying_count := m - (6^4)
  -- Proof
  sorry

end count_base8_numbers_with_4_or_6_l809_809732


namespace largest_base5_three_digits_is_124_l809_809934

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l809_809934


namespace tetrahedron_three_right_planar_angles_l809_809910

noncomputable def right_dihedral_angle_tetrahedron
  (T : Tetrahedron)
  (A B C D : T.V) : Prop :=
dihedral_angle T A B C = π/2 ∧ 
dihedral_angle T B C D = π/2 ∧ 
dihedral_angle T C D A = π/2

noncomputable def right_planar_angles_tetrahedron
  (T : Tetrahedron)
  (A B C D : T.V) : Prop :=
planar_angle T A B C = π/2 ∧ 
planar_angle T B C D = π/2 ∧ 
planar_angle T C D A = π/2 

theorem tetrahedron_three_right_planar_angles
  (T : Tetrahedron)
  (A B C D : T.V)
  (h : right_dihedral_angle_tetrahedron T A B C D) :
  right_planar_angles_tetrahedron T A B C D :=
sorry

end tetrahedron_three_right_planar_angles_l809_809910


namespace square_of_sum_l809_809977

theorem square_of_sum (x y : ℝ) (A B C D : ℝ) :
  A = 2 * x^2 + y^2 →
  B = 2 * (x + y)^2 →
  C = 2 * x + y^2 →
  D = (2 * x + y)^2 →
  D = (2 * x + y)^2 :=
by intros; exact ‹D = (2 * x + y)^2›

end square_of_sum_l809_809977


namespace largest_base_5_three_digit_in_base_10_l809_809944

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l809_809944


namespace problem_equivalent_proof_l809_809399

variables {A B C D M N O : Type*}
variables [has_mem A (set O)] [has_mem B (set O)] [has_mem C (set O)] [has_mem D (set O)]
variables 
  (ABCD_is_trapezium : parallel A B C D) 
  (circle_DAD : circle (diameter A D))
  (circle_BCB : circle (diameter B C))
  (M_is_intersection : ∃ M, circle_DAD.intersect M ∧ circle_BCB.intersect M)
  (N_is_intersection : ∃ N, circle_DAD.intersect N ∧ circle_BCB.intersect N)
  (O_is_intersection : ∃ O, line_AD A C.intersect B D ∧ line_BD)

theorem problem_equivalent_proof : O ∈ set.line M N :=
sorry

end problem_equivalent_proof_l809_809399


namespace tomato_seed_cost_l809_809576

theorem tomato_seed_cost (T : ℝ) 
  (h1 : 3 * 2.50 + 4 * T + 5 * 0.90 = 18) : 
  T = 1.50 := 
by
  sorry

end tomato_seed_cost_l809_809576


namespace find_square_side_length_l809_809872

noncomputable def side_length_PQRS (x : ℝ) : Prop :=
  let PT := 1
  let QU := 2
  let RV := 3
  let SW := 4
  let PQRS_area := x^2
  let TUVW_area := 1 / 2 * x^2
  let triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height
  PQRS_area = x^2 ∧ TUVW_area = 1 / 2 * x^2 ∧
  triangle_area 1 (x - 4) + (x - 1) + 
  triangle_area 3 (x - 2) + 2 * (x - 3) = 1 / 2 * x^2

theorem find_square_side_length : ∃ x : ℝ, side_length_PQRS x ∧ x = 6 := 
  sorry

end find_square_side_length_l809_809872


namespace place_rooks_4x4x4_l809_809690

def valid_rook_position (n : ℕ) (positions : Fin n → Fin n × Fin n × Fin n) : Prop :=
  ∀ (i j : Fin n),
    i ≠ j →
    (positions i).1 ≠ (positions j).1 ∧
    (positions i).2 ≠ (positions j).2 ∧
    (positions i).3 ≠ (positions j).3

theorem place_rooks_4x4x4 : ∃ positions : Fin 16 → Fin 4 × Fin 4 × Fin 4,
  valid_rook_position 16 positions :=
sorry

end place_rooks_4x4x4_l809_809690


namespace find_k_l809_809686

theorem find_k (k : ℕ) (h₀ : 0 < k) (h₁ : k < log 2 3 + log 3 4) (h₂ : log 2 3 + log 3 4 < k + 1) : k = 2 :=
sorry

end find_k_l809_809686
