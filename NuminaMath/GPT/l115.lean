import Mathlib

namespace abs_opposite_numbers_l115_11576

theorem abs_opposite_numbers (m n : ℤ) (h : m + n = 0) : |m + n - 1| = 1 := by
  sorry

end abs_opposite_numbers_l115_11576


namespace simplify_tan_expression_l115_11594

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end simplify_tan_expression_l115_11594


namespace garden_ratio_l115_11518

theorem garden_ratio 
  (P : ℕ) (L : ℕ) (W : ℕ) 
  (h1 : P = 900) 
  (h2 : L = 300) 
  (h3 : P = 2 * (L + W)) : 
  L / W = 2 :=
by 
  sorry

end garden_ratio_l115_11518


namespace minimum_value_l115_11548

noncomputable def min_value (a b c d : ℝ) : ℝ :=
(a - c) ^ 2 + (b - d) ^ 2

theorem minimum_value (a b c d : ℝ) (hab : a * b = 3) (hcd : c + 3 * d = 0) :
  min_value a b c d ≥ (18 / 5) :=
by
  sorry

end minimum_value_l115_11548


namespace directrix_of_parabola_l115_11541

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end directrix_of_parabola_l115_11541


namespace find_z_l115_11512

theorem find_z (a z : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * z) : z = 49 :=
sorry

end find_z_l115_11512


namespace original_square_area_l115_11578

noncomputable def area_of_original_square (x : ℝ) (w : ℝ) (remaining_area : ℝ) : Prop :=
  x * x = remaining_area + x * w

theorem original_square_area : ∃ (x : ℝ), area_of_original_square x 3 40 → x * x = 64 :=
sorry

end original_square_area_l115_11578


namespace angle_solution_l115_11579

/-!
  Given:
  k + 90° = 360°

  Prove:
  k = 270°
-/

theorem angle_solution (k : ℝ) (h : k + 90 = 360) : k = 270 :=
by
  sorry

end angle_solution_l115_11579


namespace combinations_of_balls_and_hats_l115_11597

def validCombinations (b h : ℕ) : Prop :=
  6 * b + 4 * h = 100 ∧ h ≥ 2

theorem combinations_of_balls_and_hats : 
  (∃ (n : ℕ), n = 8 ∧ (∀ b h : ℕ, validCombinations b h → validCombinations b h)) :=
by
  sorry

end combinations_of_balls_and_hats_l115_11597


namespace four_distinct_real_roots_l115_11506

theorem four_distinct_real_roots (m : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 4 * |x| + 5 - m) ∧ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) ↔ (1 < m ∧ m < 5) :=
by
  sorry

end four_distinct_real_roots_l115_11506


namespace problem_solution_l115_11567

theorem problem_solution (b : ℝ) (i : ℂ) (h : i^2 = -1) (h_cond : (2 - i) * (4 * i) = 4 + b * i) : 
  b = 8 := 
by 
  sorry

end problem_solution_l115_11567


namespace tunnel_length_l115_11505

theorem tunnel_length (L L_1 L_2 v v_new t t_new : ℝ) (H1: L_1 = 6) (H2: L_2 = 12) 
  (H3: v_new = 0.8 * v) (H4: t = (L + L_1) / v) (H5: t_new = 1.5 * t)
  (H6: t_new = (L + L_2) / v_new) : 
  L = 24 :=
by
  sorry

end tunnel_length_l115_11505


namespace rongrong_bike_speed_l115_11523

theorem rongrong_bike_speed :
  ∃ (x : ℝ), (15 / x - 15 / (4 * x) = 45 / 60) → x = 15 :=
by
  sorry

end rongrong_bike_speed_l115_11523


namespace bottles_from_Shop_C_l115_11556

theorem bottles_from_Shop_C (TotalBottles ShopA ShopB ShopC : ℕ) 
  (h1 : TotalBottles = 550) 
  (h2 : ShopA = 150) 
  (h3 : ShopB = 180) 
  (h4 : TotalBottles = ShopA + ShopB + ShopC) : 
  ShopC = 220 := 
by
  sorry

end bottles_from_Shop_C_l115_11556


namespace correct_exponential_calculation_l115_11574

theorem correct_exponential_calculation (a : ℝ) (ha : a ≠ 0) : 
  (a^4)^4 = a^16 :=
by sorry

end correct_exponential_calculation_l115_11574


namespace right_triangle_largest_side_l115_11516

theorem right_triangle_largest_side (b d : ℕ) (h_triangle : (b - d)^2 + b^2 = (b + d)^2)
  (h_arith_seq : (b - d) < b ∧ b < (b + d))
  (h_perimeter : (b - d) + b + (b + d) = 840) :
  (b + d = 350) :=
by sorry

end right_triangle_largest_side_l115_11516


namespace point_D_sum_is_ten_l115_11532

noncomputable def D_coordinates_sum_eq_ten : Prop :=
  ∃ (D : ℝ × ℝ), (5, 5) = ( (7 + D.1) / 2, (3 + D.2) / 2 ) ∧ (D.1 + D.2 = 10)

theorem point_D_sum_is_ten : D_coordinates_sum_eq_ten :=
  sorry

end point_D_sum_is_ten_l115_11532


namespace sin_330_correct_l115_11583

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l115_11583


namespace quadratic_roots_equal_l115_11526

theorem quadratic_roots_equal {k : ℝ} (h : (2 * k) ^ 2 - 4 * 1 * (k^2 + k + 3) = 0) : k^2 + k + 3 = 9 :=
by
  sorry

end quadratic_roots_equal_l115_11526


namespace shortest_side_of_triangle_with_medians_l115_11550

noncomputable def side_lengths_of_triangle_with_medians (a b c m_a m_b m_c : ℝ) : Prop :=
  m_a = 3 ∧ m_b = 4 ∧ m_c = 5 →
  a^2 = 2*b^2 + 2*c^2 - 36 ∧
  b^2 = 2*a^2 + 2*c^2 - 64 ∧
  c^2 = 2*a^2 + 2*b^2 - 100

theorem shortest_side_of_triangle_with_medians :
  ∀ (a b c : ℝ), side_lengths_of_triangle_with_medians a b c 3 4 5 → 
  min a (min b c) = c :=
sorry

end shortest_side_of_triangle_with_medians_l115_11550


namespace minimum_negative_factors_l115_11500

theorem minimum_negative_factors (a b c d : ℝ) (h1 : a * b * c * d < 0) (h2 : a + b = 0) (h3 : c * d > 0) : 
    (∃ x ∈ [a, b, c, d], x < 0) :=
by
  sorry

end minimum_negative_factors_l115_11500


namespace min_positive_announcements_l115_11573

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 110) 
  (h2 : y * (y - 1) + (x - y) * (x - 1 - (y - 1)) = 50) : 
  y >= 5 := 
sorry

end min_positive_announcements_l115_11573


namespace evaluate_expression_l115_11582

theorem evaluate_expression (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 :=
by
  sorry

end evaluate_expression_l115_11582


namespace factorization1_factorization2_l115_11558

theorem factorization1 (x y : ℝ) : 4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3 * x + 3 * y)^2 :=
by
  sorry

theorem factorization2 (x : ℝ) (a : ℝ) : 2 * a * (x^2 + 1)^2 - 8 * a * x^2 = 2 * a * (x - 1)^2 * (x + 1)^2 :=
by
  sorry

end factorization1_factorization2_l115_11558


namespace number_multiplied_by_3_l115_11584

variable (A B C D E : ℝ) -- Declare the five numbers

theorem number_multiplied_by_3 (h1 : (A + B + C + D + E) / 5 = 6.8) 
    (h2 : ∃ X : ℝ, (A + B + C + D + E + 2 * X) / 5 = 9.2) : 
    ∃ X : ℝ, X = 6 := 
  sorry

end number_multiplied_by_3_l115_11584


namespace perpendicular_vectors_l115_11590

theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) (c : ℝ × ℝ) 
  (h1 : a = (1, 2)) (h2 : b = (1, 1)) 
  (h3 : c = (1 + k, 2 + k))
  (h4 : b.1 * c.1 + b.2 * c.2 = 0) : 
  k = -3 / 2 :=
by
  sorry

end perpendicular_vectors_l115_11590


namespace probability_red_then_white_l115_11528

-- Define the total number of balls and the probabilities
def total_balls : ℕ := 9
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probabilities
def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

-- Define the combined probability of drawing a red and then a white ball 
theorem probability_red_then_white : (prob_red * prob_white) = 2/27 :=
by
  sorry

end probability_red_then_white_l115_11528


namespace truth_values_l115_11538

-- Define the region D as a set
def D (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 ≤ 4

-- Define propositions p and q
def p : Prop := ∀ x y, D x y → 2 * x + y ≤ 8
def q : Prop := ∃ x y, D x y ∧ 2 * x + y ≤ -1

-- State the propositions to be proven
def prop1 : Prop := p ∨ q
def prop2 : Prop := ¬p ∨ q
def prop3 : Prop := p ∧ ¬q
def prop4 : Prop := ¬p ∧ ¬q

-- State the main theorem asserting the truth values of the propositions
theorem truth_values : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
by
  sorry

end truth_values_l115_11538


namespace harmon_high_voting_l115_11557

theorem harmon_high_voting
  (U : Finset ℝ) -- Universe of students
  (A B : Finset ℝ) -- Sets of students favoring proposals
  (hU : U.card = 215)
  (hA : A.card = 170)
  (hB : B.card = 142)
  (hAcBc : (U \ (A ∪ B)).card = 38) :
  (A ∩ B).card = 135 :=
by {
  sorry
}

end harmon_high_voting_l115_11557


namespace math_problem_l115_11559

theorem math_problem (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1)^2 = 0) : (x + 2 * y)^3 = 125 / 8 := 
sorry

end math_problem_l115_11559


namespace equal_money_distribution_l115_11547

theorem equal_money_distribution (y : ℝ) : 
  ∃ z : ℝ, z = 0.1 * (1.25 * y) ∧ (1.25 * y) - z = y + z - y :=
by
  sorry

end equal_money_distribution_l115_11547


namespace four_digit_integer_5533_l115_11581

theorem four_digit_integer_5533
  (a b c d : ℕ)
  (h1 : a + b + c + d = 16)
  (h2 : b + c = 8)
  (h3 : a - d = 2)
  (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  1000 * a + 100 * b + 10 * c + d = 5533 :=
by {
  sorry
}

end four_digit_integer_5533_l115_11581


namespace student_weight_l115_11537

theorem student_weight (S R : ℕ) (h1 : S - 5 = 2 * R) (h2 : S + R = 116) : S = 79 :=
sorry

end student_weight_l115_11537


namespace total_gain_percentage_combined_l115_11533

theorem total_gain_percentage_combined :
  let CP1 := 20
  let CP2 := 35
  let CP3 := 50
  let SP1 := 25
  let SP2 := 44
  let SP3 := 65
  let totalCP := CP1 + CP2 + CP3
  let totalSP := SP1 + SP2 + SP3
  let totalGain := totalSP - totalCP
  let gainPercentage := (totalGain / totalCP) * 100
  gainPercentage = 27.62 :=
by sorry

end total_gain_percentage_combined_l115_11533


namespace earn_2800_probability_l115_11525

def total_outcomes : ℕ := 7 ^ 4

def favorable_outcomes : ℕ :=
  (1 * 3 * 2 * 1) * 4 -- For each combination: \$1000, \$600, \$600, \$600; \$1000, \$1000, \$400, \$400; \$800, \$800, \$600, \$600; \$800, \$800, \$800, \$400

noncomputable def probability_of_earning_2800 : ℚ := favorable_outcomes / total_outcomes

theorem earn_2800_probability : probability_of_earning_2800 = 96 / 2401 := by
  sorry

end earn_2800_probability_l115_11525


namespace sum_of_reciprocals_of_roots_eq_17_div_8_l115_11517

theorem sum_of_reciprocals_of_roots_eq_17_div_8 :
  ∀ p q : ℝ, (p + q = 17) → (p * q = 8) → (1 / p + 1 / q = 17 / 8) :=
by
  intros p q h1 h2
  sorry

end sum_of_reciprocals_of_roots_eq_17_div_8_l115_11517


namespace solve_congruences_l115_11530

theorem solve_congruences :
  ∃ x : ℤ, 
  x ≡ 3 [ZMOD 7] ∧ 
  x^2 ≡ 44 [ZMOD 49] ∧ 
  x^3 ≡ 111 [ZMOD 343] ∧ 
  x ≡ 17 [ZMOD 343] :=
sorry

end solve_congruences_l115_11530


namespace four_digit_numbers_with_property_l115_11508

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l115_11508


namespace cards_per_box_l115_11513

-- Define the conditions
def total_cards : ℕ := 75
def cards_not_in_box : ℕ := 5
def boxes_given_away : ℕ := 2
def boxes_left : ℕ := 5

-- Calculating the total number of boxes initially
def initial_boxes : ℕ := boxes_given_away + boxes_left

-- Define the number of cards in each box
def num_cards_per_box (number_of_cards : ℕ) (number_of_boxes : ℕ) : ℕ :=
  (number_of_cards - cards_not_in_box) / number_of_boxes

-- The proof problem statement
theorem cards_per_box :
  num_cards_per_box total_cards initial_boxes = 10 :=
by
  -- Proof is omitted with sorry
  sorry

end cards_per_box_l115_11513


namespace area_in_terms_of_diagonal_l115_11552

variables (l w d : ℝ)

-- Given conditions
def length_to_width_ratio := l / w = 5 / 2
def diagonal_relation := d^2 = l^2 + w^2

-- Proving the area is kd^2 with k = 10 / 29
theorem area_in_terms_of_diagonal 
    (ratio : length_to_width_ratio l w)
    (diag_rel : diagonal_relation l w d) :
  ∃ k, k = 10 / 29 ∧ (l * w = k * d^2) :=
sorry

end area_in_terms_of_diagonal_l115_11552


namespace sin_double_angle_l115_11560

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the angle α such that its terminal side passes through point P
noncomputable def α : ℝ := sorry -- The exact definition of α is not needed for this statement

-- Define r as the distance from the origin to the point P
noncomputable def r : ℝ := Real.sqrt ((P.1 ^ 2) + (P.2 ^ 2))

-- Define sin(α) and cos(α)
noncomputable def sin_α : ℝ := P.2 / r
noncomputable def cos_α : ℝ := P.1 / r

-- The proof statement
theorem sin_double_angle : 2 * sin_α * cos_α = -4 / 5 := by
  sorry

end sin_double_angle_l115_11560


namespace inequality_solution_set_l115_11545

theorem inequality_solution_set :
  {x : ℝ | (x^2 - 4) / (x^2 - 9) > 0} = {x : ℝ | x < -3 ∨ x > 3} :=
sorry

end inequality_solution_set_l115_11545


namespace fraction_of_power_l115_11540

theorem fraction_of_power (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end fraction_of_power_l115_11540


namespace hypotenuse_length_l115_11592

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l115_11592


namespace expected_value_of_win_is_correct_l115_11522

noncomputable def expected_value_of_win : ℝ :=
  (1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) +
  (1 / 8) * (8 - 4) + (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) +
  (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8)

theorem expected_value_of_win_is_correct :
  expected_value_of_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_correct_l115_11522


namespace combined_solid_sum_faces_edges_vertices_l115_11599

noncomputable def prism_faces : ℕ := 6
noncomputable def prism_edges : ℕ := 12
noncomputable def prism_vertices : ℕ := 8
noncomputable def new_pyramid_faces : ℕ := 4
noncomputable def new_pyramid_edges : ℕ := 4
noncomputable def new_pyramid_vertex : ℕ := 1

theorem combined_solid_sum_faces_edges_vertices :
  prism_faces - 1 + new_pyramid_faces + prism_edges + new_pyramid_edges + prism_vertices + new_pyramid_vertex = 34 :=
by
  -- proof would go here
  sorry

end combined_solid_sum_faces_edges_vertices_l115_11599


namespace least_number_to_add_l115_11593

theorem least_number_to_add (n : ℕ) (h : n = 17 * 23 * 29) : 
  ∃ k, k + 1024 ≡ 0 [MOD n] ∧ 
       (∀ m, (m + 1024) ≡ 0 [MOD n] → k ≤ m) ∧ 
       k = 10315 :=
by 
  sorry

end least_number_to_add_l115_11593


namespace solve_system_of_equations_l115_11501

theorem solve_system_of_equations :
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℤ), 
    x1 + x2 + x3 = 6 ∧
    x2 + x3 + x4 = 9 ∧
    x3 + x4 + x5 = 3 ∧
    x4 + x5 + x6 = -3 ∧
    x5 + x6 + x7 = -9 ∧
    x6 + x7 + x8 = -6 ∧
    x7 + x8 + x1 = -2 ∧
    x8 + x1 + x2 = 2 ∧
    (x1, x2, x3, x4, x5, x6, x7, x8) = (1, 2, 3, 4, -4, -3, -2, -1) :=
by
  -- solution will be here
  sorry

end solve_system_of_equations_l115_11501


namespace pascal_no_divisible_by_prime_iff_form_l115_11591

theorem pascal_no_divisible_by_prime_iff_form (p : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) :
  (∀ k ≤ n, Nat.choose n k % p ≠ 0) ↔ ∃ s q : ℕ, s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by
  sorry

end pascal_no_divisible_by_prime_iff_form_l115_11591


namespace white_square_area_l115_11561

theorem white_square_area
  (edge_length : ℝ)
  (total_green_area : ℝ)
  (faces : ℕ)
  (green_per_face : ℝ)
  (total_surface_area : ℝ)
  (white_area_per_face : ℝ) :
  edge_length = 12 ∧ total_green_area = 432 ∧ faces = 6 ∧ total_surface_area = 864 ∧ green_per_face = total_green_area / faces ∧ white_area_per_face = total_surface_area / faces - green_per_face → white_area_per_face = 72 :=
by
  sorry

end white_square_area_l115_11561


namespace number_of_weavers_is_4_l115_11571

theorem number_of_weavers_is_4
  (mats1 days1 weavers1 mats2 days2 weavers2 : ℕ)
  (h1 : mats1 = 4)
  (h2 : days1 = 4)
  (h3 : weavers2 = 10)
  (h4 : mats2 = 25)
  (h5 : days2 = 10)
  (h_rate_eq : (mats1 / (weavers1 * days1)) = (mats2 / (weavers2 * days2))) :
  weavers1 = 4 :=
by
  sorry

end number_of_weavers_is_4_l115_11571


namespace floor_e_eq_two_l115_11562

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l115_11562


namespace determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l115_11595

-- Cost price per souvenir
def cost_price : ℕ := 40

-- Minimum selling price
def min_selling_price : ℕ := 44

-- Maximum selling price
def max_selling_price : ℕ := 60

-- Units sold if selling price is min_selling_price
def units_sold_at_min_price : ℕ := 300

-- Units sold decreases by 10 for every 1 yuan increase in selling price
def decrease_in_units (increase : ℕ) : ℕ := 10 * increase

-- Daily profit for a given increase in selling price
def daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase)

-- Maximum profit calculation
def maximizing_daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase) 

-- Statement for Problem Part 1
theorem determine_selling_price_for_daily_profit : ∃ P, P = 52 ∧ daily_profit (P - min_selling_price) = 2640 := 
sorry

-- Statement for Problem Part 2
theorem determine_max_profit_and_selling_price : ∃ P, P = 57 ∧ maximizing_daily_profit (P - min_selling_price) = 2890 := 
sorry

end determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l115_11595


namespace benny_spent_amount_l115_11536

-- Definitions based on given conditions
def initial_amount : ℕ := 79
def amount_left : ℕ := 32

-- Proof problem statement
theorem benny_spent_amount :
  initial_amount - amount_left = 47 :=
sorry

end benny_spent_amount_l115_11536


namespace find_z_when_y_is_6_l115_11519

variable {y z : ℚ}

/-- Condition: y^4 varies inversely with √[4]{z}. -/
def inverse_variation (k : ℚ) (y z : ℚ) : Prop :=
  y^4 * z^(1/4) = k

/-- Given constant k based on y = 3 and z = 16. -/
def k_value : ℚ := 162

theorem find_z_when_y_is_6
  (h_inv : inverse_variation k_value 3 16)
  (h_y : y = 6) :
  z = 1 / 4096 := 
sorry

end find_z_when_y_is_6_l115_11519


namespace expected_amoebas_after_one_week_l115_11575

section AmoebaProblem

-- Definitions from conditions
def initial_amoebas : ℕ := 1
def split_probability : ℝ := 0.8
def days : ℕ := 7

-- Function to calculate expected amoebas
def expected_amoebas (n : ℕ) : ℝ :=
  initial_amoebas * ((2 : ℝ) ^ n) * (split_probability ^ n)

-- Theorem statement
theorem expected_amoebas_after_one_week :
  expected_amoebas days = 26.8435456 :=
by sorry

end AmoebaProblem

end expected_amoebas_after_one_week_l115_11575


namespace cos_alpha_solution_l115_11577

open Real

theorem cos_alpha_solution
  (α : ℝ)
  (h1 : π < α)
  (h2 : α < 3 * π / 2)
  (h3 : tan α = 2) :
  cos α = -sqrt (1 / (1 + 2^2)) :=
by
  sorry

end cos_alpha_solution_l115_11577


namespace oldest_bride_age_l115_11502

theorem oldest_bride_age (B G : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) :
  B = 102 :=
by
  sorry

end oldest_bride_age_l115_11502


namespace perimeter_division_l115_11572

-- Define the given conditions
def is_pentagon (n : ℕ) : Prop := n = 5
def side_length (s : ℕ) : Prop := s = 25
def perimeter (P : ℕ) (n s : ℕ) : Prop := P = n * s

-- Define the Lean statement to prove
theorem perimeter_division (n s P x : ℕ) 
  (h1 : is_pentagon n) 
  (h2 : side_length s) 
  (h3 : perimeter P n s) 
  (h4 : P = 125) 
  (h5 : s = 25) : 
  P / x = s → x = 5 := 
by
  sorry

end perimeter_division_l115_11572


namespace evaluate_f_at_5_l115_11549

def f (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 38*x^2 - 35*x - 40

theorem evaluate_f_at_5 : f 5 = 110 :=
by
  sorry

end evaluate_f_at_5_l115_11549


namespace yanna_change_l115_11585

theorem yanna_change :
  let shirt_cost := 5
  let sandal_cost := 3
  let num_shirts := 10
  let num_sandals := 3
  let given_amount := 100
  (given_amount - (num_shirts * shirt_cost + num_sandals * sandal_cost)) = 41 :=
by
  sorry

end yanna_change_l115_11585


namespace book_discount_l115_11507

theorem book_discount (a b : ℕ) (x y : ℕ) (h1 : x = 10 * a + b) (h2 : y = 10 * b + a) (h3 : (3 / 8) * x = y) :
  x - y = 45 := 
sorry

end book_discount_l115_11507


namespace oil_vinegar_new_ratio_l115_11587

theorem oil_vinegar_new_ratio (initial_oil initial_vinegar new_vinegar : ℕ) 
    (h1 : initial_oil / initial_vinegar = 3 / 1)
    (h2 : new_vinegar = (2 * initial_vinegar)) :
    initial_oil / new_vinegar = 3 / 2 :=
by
  sorry

end oil_vinegar_new_ratio_l115_11587


namespace num_common_points_l115_11515

noncomputable def curve (x : ℝ) : ℝ := 3 * x ^ 4 - 2 * x ^ 3 - 9 * x ^ 2 + 4

noncomputable def tangent_line (x : ℝ) : ℝ :=
  -12 * (x - 1) - 4

theorem num_common_points :
  ∃ (x1 x2 x3 : ℝ), curve x1 = tangent_line x1 ∧
                    curve x2 = tangent_line x2 ∧
                    curve x3 = tangent_line x3 ∧
                    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
sorry

end num_common_points_l115_11515


namespace total_pages_in_book_l115_11521

-- Define the conditions
def pagesDay1To5 : Nat := 5 * 25
def pagesDay6To9 : Nat := 4 * 40
def pagesLastDay : Nat := 30

-- Total calculation
def totalPages (p1 p2 pLast : Nat) : Nat := p1 + p2 + pLast

-- The proof problem statement
theorem total_pages_in_book :
  totalPages pagesDay1To5 pagesDay6To9 pagesLastDay = 315 :=
  by
    sorry

end total_pages_in_book_l115_11521


namespace graph_properties_l115_11553

theorem graph_properties (x : ℝ) :
  (∃ p : ℝ × ℝ, p = (1, -7) ∧ y = -7 * x) ∧
  (x ≠ 0 → y * x < 0) ∧
  (x > 0 → y < 0) :=
by
  sorry

end graph_properties_l115_11553


namespace chair_and_desk_prices_l115_11544

theorem chair_and_desk_prices (c d : ℕ) 
  (h1 : c + d = 115)
  (h2 : d - c = 45) :
  c = 35 ∧ d = 80 := 
by
  sorry

end chair_and_desk_prices_l115_11544


namespace floor_width_l115_11588

theorem floor_width (W : ℕ) (hAreaFloor: 10 * W - 64 = 16) : W = 8 :=
by
  -- the proof should be added here
  sorry

end floor_width_l115_11588


namespace squares_difference_l115_11565

theorem squares_difference :
  1010^2 - 994^2 - 1008^2 + 996^2 = 8016 :=
by
  sorry

end squares_difference_l115_11565


namespace travel_same_direction_time_l115_11551

variable (A B : Type) [MetricSpace A] (downstream_speed upstream_speed : ℝ)
  (H_A_downstream_speed : downstream_speed = 8)
  (H_A_upstream_speed : upstream_speed = 4)
  (H_B_downstream_speed : downstream_speed = 8)
  (H_B_upstream_speed : upstream_speed = 4)
  (H_equal_travel_time : (∃ x : ℝ, x * downstream_speed + (3 - x) * upstream_speed = 3)
                      ∧ (∃ x : ℝ, x * upstream_speed + (3 - x) * downstream_speed = 3))

theorem travel_same_direction_time (A_α_downstream B_β_upstream A_α_upstream B_β_downstream : ℝ)
  (H_travel_time : (∃ x : ℝ, x = 1) ∧ (A_α_upstream = 3 - A_α_downstream) ∧ (B_β_downstream = 3 - B_β_upstream)) :
  A_α_downstream = 1 → A_α_upstream = 3 - 1 → B_β_downstream = 1 → B_β_upstream = 3 - 1 → ∃ t, t = 1 :=
by
  sorry

end travel_same_direction_time_l115_11551


namespace find_B_l115_11524

theorem find_B (B: ℕ) (h1: 5457062 % 2 = 0 ∧ 200 * B % 4 = 0) (h2: 5457062 % 5 = 0 ∧ B % 5 = 0) (h3: 5450062 % 8 = 0 ∧ 100 * B % 8 = 0) : B = 0 :=
sorry

end find_B_l115_11524


namespace percentage_cost_for_overhead_l115_11564

theorem percentage_cost_for_overhead
  (P M N : ℝ)
  (hP : P = 48)
  (hM : M = 50)
  (hN : N = 12) :
  (P + M - P - N) / P * 100 = 79.17 := by
  sorry

end percentage_cost_for_overhead_l115_11564


namespace multiplication_factor_l115_11555

theorem multiplication_factor 
  (avg1 : ℕ → ℕ → ℕ)
  (avg2 : ℕ → ℕ → ℕ)
  (sum1 : ℕ)
  (num1 : ℕ)
  (num2 : ℕ)
  (sum2 : ℕ)
  (factor : ℚ) :
  avg1 sum1 num1 = 7 →
  avg2 sum2 num2 = 84 →
  sum1 = 10 * 7 →
  sum2 = 10 * 84 →
  factor = sum2 / sum1 →
  factor = 12 :=
by
  sorry

end multiplication_factor_l115_11555


namespace no_solution_to_equation_l115_11589

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, 8 / (x ^ 2 - 4) + 1 = x / (x - 2) :=
by
  sorry

end no_solution_to_equation_l115_11589


namespace average_pages_per_book_deshaun_l115_11554

-- Definitions related to the conditions
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def person_closest_percentage : ℚ := 0.75
def second_person_daily_pages : ℕ := 180

-- Derived definitions
def second_person_total_pages : ℕ := second_person_daily_pages * summer_days
def deshaun_total_pages : ℚ := second_person_total_pages / person_closest_percentage

-- The final proof statement
theorem average_pages_per_book_deshaun : 
  deshaun_total_pages / deshaun_books = 320 := 
by
  -- We would provide the proof here
  sorry

end average_pages_per_book_deshaun_l115_11554


namespace find_n_l115_11529

theorem find_n :
  ∀ (n : ℕ),
    2^200 * 2^203 + 2^163 * 2^241 + 2^126 * 2^277 = 32^n →
    n = 81 :=
by
  intros n h
  sorry

end find_n_l115_11529


namespace yvonnes_probability_l115_11563

open Classical

variables (P_X P_Y P_Z : ℝ)

theorem yvonnes_probability
  (h1 : P_X = 1/5)
  (h2 : P_Z = 5/8)
  (h3 : P_X * P_Y * (1 - P_Z) = 0.0375) :
  P_Y = 0.5 :=
by
  sorry

end yvonnes_probability_l115_11563


namespace find_triple_l115_11570

theorem find_triple (A B C : ℕ) (h1 : A^2 + B - C = 100) (h2 : A + B^2 - C = 124) : 
  (A, B, C) = (12, 13, 57) := 
  sorry

end find_triple_l115_11570


namespace XiaoYing_minimum_water_usage_l115_11509

-- Definitions based on the problem's conditions
def first_charge_rate : ℝ := 2.8
def excess_charge_rate : ℝ := 3
def initial_threshold : ℝ := 5
def minimum_bill : ℝ := 29

-- Main statement for the proof based on the derived inequality
theorem XiaoYing_minimum_water_usage (x : ℝ) (h1 : 2.8 * initial_threshold + 3 * (x - initial_threshold) ≥ 29) : x ≥ 10 := by
  sorry

end XiaoYing_minimum_water_usage_l115_11509


namespace gcd_lcm_sum_eq_90_l115_11596

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem gcd_lcm_sum_eq_90 : 
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  A + B = 90 :=
by
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  sorry

end gcd_lcm_sum_eq_90_l115_11596


namespace tan_sum_product_l115_11539

theorem tan_sum_product (tan : ℝ → ℝ) : 
  (1 + tan 23) * (1 + tan 22) = 2 + tan 23 * tan 22 := by sorry

end tan_sum_product_l115_11539


namespace find_modulus_z_l115_11546

open Complex

noncomputable def z_w_condition1 (z w : ℂ) : Prop := abs (3 * z - w) = 17
noncomputable def z_w_condition2 (z w : ℂ) : Prop := abs (z + 3 * w) = 4
noncomputable def z_w_condition3 (z w : ℂ) : Prop := abs (z + w) = 6

theorem find_modulus_z (z w : ℂ) (h1 : z_w_condition1 z w) (h2 : z_w_condition2 z w) (h3 : z_w_condition3 z w) :
  abs z = 5 :=
by
  sorry

end find_modulus_z_l115_11546


namespace range_of_a_l115_11510

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 - 2 * x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic a x > 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l115_11510


namespace total_initial_candles_l115_11543

-- Define the conditions
def used_candles : ℕ := 32
def leftover_candles : ℕ := 12

-- State the theorem
theorem total_initial_candles : used_candles + leftover_candles = 44 := by
  sorry

end total_initial_candles_l115_11543


namespace mul_112_54_l115_11569

theorem mul_112_54 : 112 * 54 = 6048 :=
by
  sorry

end mul_112_54_l115_11569


namespace total_price_of_property_l115_11514

theorem total_price_of_property (price_per_sq_ft: ℝ) (house_size barn_size: ℝ) (house_price barn_price total_price: ℝ) :
  price_per_sq_ft = 98 ∧ house_size = 2400 ∧ barn_size = 1000 → 
  house_price = price_per_sq_ft * house_size ∧
  barn_price = price_per_sq_ft * barn_size ∧
  total_price = house_price + barn_price →
  total_price = 333200 :=
by
  sorry

end total_price_of_property_l115_11514


namespace boxes_with_no_items_l115_11566

-- Definitions of each condition as given in the problem
def total_boxes : Nat := 15
def pencil_boxes : Nat := 8
def pen_boxes : Nat := 5
def marker_boxes : Nat := 3
def pen_pencil_boxes : Nat := 4
def all_three_boxes : Nat := 1

-- The theorem to prove
theorem boxes_with_no_items : 
     (total_boxes - ((pen_pencil_boxes - all_three_boxes)
                     + (pencil_boxes - pen_pencil_boxes - all_three_boxes)
                     + (pen_boxes - pen_pencil_boxes - all_three_boxes)
                     + (marker_boxes - all_three_boxes)
                     + all_three_boxes)) = 5 := 
by 
  -- This is where the proof would go, but we'll use sorry to indicate it's skipped.
  sorry

end boxes_with_no_items_l115_11566


namespace find_f_of_7_6_l115_11527

-- Definitions from conditions
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x k : ℤ, f (x + T * (k : ℝ)) = f x

def f_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = x

-- The periodic function f with period 4
def f : ℝ → ℝ := sorry

-- Hypothesis
axiom f_periodic : periodic_function f 4
axiom f_on_interval : f_in_interval f

-- Theorem to prove
theorem find_f_of_7_6 : f 7.6 = 3.6 :=
by
  sorry

end find_f_of_7_6_l115_11527


namespace rectangular_solid_surface_area_l115_11503

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hvol : a * b * c = 455) : 
  let surface_area := 2 * (a * b + b * c + c * a)
  surface_area = 382 := by
-- proof
sorry

end rectangular_solid_surface_area_l115_11503


namespace simplify_expression_l115_11586

theorem simplify_expression (r : ℝ) : (2 * r^2 + 5 * r - 3) + (3 * r^2 - 4 * r + 2) = 5 * r^2 + r - 1 := 
by
  sorry

end simplify_expression_l115_11586


namespace cost_of_blue_cap_l115_11580

theorem cost_of_blue_cap (cost_tshirt cost_backpack cost_cap total_spent discount: ℝ) 
  (h1 : cost_tshirt = 30) 
  (h2 : cost_backpack = 10) 
  (h3 : discount = 2)
  (h4 : total_spent = 43) 
  (h5 : total_spent = cost_tshirt + cost_backpack + cost_cap - discount) : 
  cost_cap = 5 :=
by sorry

end cost_of_blue_cap_l115_11580


namespace triangle_area_l115_11568

variable (a b c k : ℝ)
variable (h1 : a = 2 * k)
variable (h2 : b = 3 * k)
variable (h3 : c = k * Real.sqrt 13)

theorem triangle_area (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 * a * b) = 3 * k^2 := 
by 
  sorry

end triangle_area_l115_11568


namespace exists_multiple_with_odd_digit_sum_l115_11542

theorem exists_multiple_with_odd_digit_sum (M : Nat) :
  ∃ N : Nat, N % M = 0 ∧ (Nat.digits 10 N).sum % 2 = 1 :=
by
  sorry

end exists_multiple_with_odd_digit_sum_l115_11542


namespace ratio_of_other_triangle_l115_11531

noncomputable def ratioAreaOtherTriangle (m : ℝ) : ℝ := 1 / (4 * m)

theorem ratio_of_other_triangle (m : ℝ) (h : m > 0) : ratioAreaOtherTriangle m = 1 / (4 * m) :=
by
  -- Proof will be provided here
  sorry

end ratio_of_other_triangle_l115_11531


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l115_11504

theorem problem_1 : 286 = 200 + 80 + 6 := sorry
theorem problem_2 : 7560 = 7000 + 500 + 60 := sorry
theorem problem_3 : 2048 = 2000 + 40 + 8 := sorry
theorem problem_4 : 8009 = 8000 + 9 := sorry
theorem problem_5 : 3070 = 3000 + 70 := sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l115_11504


namespace probability_red_side_first_on_third_roll_l115_11535

noncomputable def red_side_probability_first_on_third_roll : ℚ :=
  let p_non_red := 7 / 10
  let p_red := 3 / 10
  (p_non_red * p_non_red * p_red)

theorem probability_red_side_first_on_third_roll :
  red_side_probability_first_on_third_roll = 147 / 1000 := 
sorry

end probability_red_side_first_on_third_roll_l115_11535


namespace smallest_distance_l115_11511

open Complex

variable (z w : ℂ)

def a : ℂ := -2 - 4 * I
def b : ℂ := 5 + 6 * I

-- Conditions
def cond1 : Prop := abs (z + 2 + 4 * I) = 2
def cond2 : Prop := abs (w - 5 - 6 * I) = 4

-- Problem
theorem smallest_distance (h1 : cond1 z) (h2 : cond2 w) : abs (z - w) = Real.sqrt 149 - 6 :=
sorry

end smallest_distance_l115_11511


namespace eval_expression_l115_11534

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

theorem eval_expression : 2 * f 3 + 3 * f (-3) = 147 := by
  sorry

end eval_expression_l115_11534


namespace ellipse_equation_point_M_exists_l115_11520

-- Condition: Point (1, sqrt(2)/2) lies on the ellipse
def point_lies_on_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (a_gt_b : a > b) : Prop :=
  (1, Real.sqrt 2 / 2).fst^2 / a^2 + (1, Real.sqrt 2 / 2).snd^2 / b^2 = 1

-- Condition: Eccentricity of the ellipse is sqrt(2)/2
def eccentricity_condition (a b : ℝ) (c : ℝ) : Prop :=
  c / a = Real.sqrt 2 / 2 ∧ a^2 = b^2 + c^2

-- Question (I): Equation of ellipse should be (x^2 / 2 + y^2 = 1)
theorem ellipse_equation (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (a_gt_b : a > b) (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : a = Real.sqrt 2 ∧ b = 1 := 
sorry

-- Question (II): There exists M such that MA · MB is constant
theorem point_M_exists (a b c x0 : ℝ)
    (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b) 
    (a_val : a = Real.sqrt 2) (b_val : b = 1) 
    (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : 
    ∃ (M : ℝ × ℝ), M.fst = 5 / 4 ∧ M.snd = 0 ∧ -7 / 16 = -7 / 16 := 
sorry

end ellipse_equation_point_M_exists_l115_11520


namespace inequality_proof_l115_11598

theorem inequality_proof (x : ℝ) (n : ℕ) (h : 3 * x ≥ -1) : (1 + x) ^ n ≥ 1 + n * x :=
sorry

end inequality_proof_l115_11598
