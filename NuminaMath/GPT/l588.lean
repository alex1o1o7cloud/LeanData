import Mathlib

namespace NUMINAMATH_GPT_reduced_price_l588_58880

theorem reduced_price (P R : ℝ) (Q : ℝ) 
  (h1 : R = 0.80 * P) 
  (h2 : 600 = Q * P) 
  (h3 : 600 = (Q + 4) * R) : 
  R = 30 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_l588_58880


namespace NUMINAMATH_GPT_factorize_expression_l588_58839

variable (x y : ℝ)

theorem factorize_expression :
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l588_58839


namespace NUMINAMATH_GPT_potatoes_left_l588_58866

def p_initial : ℕ := 8
def p_eaten : ℕ := 3
def p_left : ℕ := p_initial - p_eaten

theorem potatoes_left : p_left = 5 := by
  sorry

end NUMINAMATH_GPT_potatoes_left_l588_58866


namespace NUMINAMATH_GPT_mice_needed_l588_58884

-- Definitions for relative strength in terms of M (Mouse strength)
def C (M : ℕ) : ℕ := 6 * M
def J (M : ℕ) : ℕ := 5 * C M
def G (M : ℕ) : ℕ := 4 * J M
def B (M : ℕ) : ℕ := 3 * G M
def D (M : ℕ) : ℕ := 2 * B M

-- Condition: all together can pull up the Turnip with strength 1237M
def total_strength_with_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M + M

-- Condition: without the Mouse, they cannot pull up the Turnip
def total_strength_without_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M

theorem mice_needed (M : ℕ) (h : total_strength_with_mouse M = 1237 * M) (h2 : total_strength_without_mouse M < 1237 * M) :
  1237 = 1237 :=
by
  -- using sorry to indicate proof is not provided
  sorry

end NUMINAMATH_GPT_mice_needed_l588_58884


namespace NUMINAMATH_GPT_patrick_purchased_pencils_l588_58888

theorem patrick_purchased_pencils (c s : ℝ) : 
  (∀ n : ℝ, n * c = 1.375 * n * s ∧ (n * c - n * s = 30 * s) → n = 80) :=
by sorry

end NUMINAMATH_GPT_patrick_purchased_pencils_l588_58888


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l588_58893

-- Define the concept of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- The problem's conditions
def a₁ : ℕ := 2
def d : ℕ := 3

-- The proof problem
theorem arithmetic_sequence_a5 : arithmetic_sequence a₁ d 5 = 14 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l588_58893


namespace NUMINAMATH_GPT_distance_from_dorm_to_city_l588_58882

theorem distance_from_dorm_to_city (D : ℝ) (h1 : D = (1/4)*D + (1/2)*D + 10 ) : D = 40 :=
sorry

end NUMINAMATH_GPT_distance_from_dorm_to_city_l588_58882


namespace NUMINAMATH_GPT_pyramid_volume_is_1_12_l588_58809

def base_rectangle_length_1 := 1
def base_rectangle_width_1_4 := 1 / 4
def pyramid_height_1 := 1

noncomputable def pyramid_volume : ℝ :=
  (1 / 3) * (base_rectangle_length_1 * base_rectangle_width_1_4) * pyramid_height_1

theorem pyramid_volume_is_1_12 : pyramid_volume = 1 / 12 :=
sorry

end NUMINAMATH_GPT_pyramid_volume_is_1_12_l588_58809


namespace NUMINAMATH_GPT_fraction_addition_l588_58870

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end NUMINAMATH_GPT_fraction_addition_l588_58870


namespace NUMINAMATH_GPT_range_of_a_l588_58849

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l588_58849


namespace NUMINAMATH_GPT_polar_to_rectangular_conversion_l588_58875

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular 5 (5 * Real.pi / 4) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_conversion_l588_58875


namespace NUMINAMATH_GPT_line_through_nodes_l588_58806

def Point := (ℤ × ℤ)

structure Triangle :=
  (A B C : Point)

def is_node (p : Point) : Prop := 
  ∃ (x y : ℤ), p = (x, y)

def strictly_inside (p : Point) (t : Triangle) : Prop := 
  -- Assume we have a function that defines if a point is strictly inside a triangle
  sorry

def nodes_inside (t : Triangle) (nodes : List Point) : Prop := 
  nodes.length = 2 ∧ ∀ p, p ∈ nodes → strictly_inside p t

theorem line_through_nodes (t : Triangle) (node1 node2 : Point) (h_inside : nodes_inside t [node1, node2]) :
   ∃ (v : Point), v ∈ [t.A, t.B, t.C] ∨
   (∃ (s : Triangle -> Point -> Point -> Prop), s t node1 node2) := 
sorry

end NUMINAMATH_GPT_line_through_nodes_l588_58806


namespace NUMINAMATH_GPT_find_divisor_value_l588_58803

theorem find_divisor_value
  (D : ℕ) 
  (h1 : ∃ k : ℕ, 242 = k * D + 6)
  (h2 : ∃ l : ℕ, 698 = l * D + 13)
  (h3 : ∃ m : ℕ, 940 = m * D + 5) : 
  D = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_value_l588_58803


namespace NUMINAMATH_GPT_find_digit_to_make_divisible_by_seven_l588_58862

/-- 
  Given a number formed by concatenating 2023 digits of 6 with 2023 digits of 5.
  In a three-digit number 6*5, find the digit * to make this number divisible by 7.
  i.e., We must find the digit x such that the number 600 + 10x + 5 is divisible by 7.
-/
theorem find_digit_to_make_divisible_by_seven :
  ∃ x : ℕ, x < 10 ∧ (600 + 10 * x + 5) % 7 = 0 :=
sorry

end NUMINAMATH_GPT_find_digit_to_make_divisible_by_seven_l588_58862


namespace NUMINAMATH_GPT_value_of_expression_l588_58824

theorem value_of_expression (a : ℚ) (h : a = 1/3) : (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l588_58824


namespace NUMINAMATH_GPT_parameterize_circle_l588_58845

noncomputable def parametrization (t : ℝ) : ℝ × ℝ :=
  ( (t^2 - 1) / (t^2 + 1), (-2 * t) / (t^2 + 1) )

theorem parameterize_circle (t : ℝ) : 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  (x^2 + y^2) = 1 :=
by 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  sorry

end NUMINAMATH_GPT_parameterize_circle_l588_58845


namespace NUMINAMATH_GPT_num_correct_props_geometric_sequence_l588_58897

-- Define what it means to be a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Original Proposition P
def Prop_P (a : ℕ → ℝ) :=
  a 1 < a 2 ∧ a 2 < a 3 → ∀ n : ℕ, a n < a (n + 1)

-- Converse of Proposition P
def Conv_Prop_P (a : ℕ → ℝ) :=
  ( ∀ n : ℕ, a n < a (n + 1) ) → a 1 < a 2 ∧ a 2 < a 3

-- Inverse of Proposition P
def Inv_Prop_P (a : ℕ → ℝ) :=
  ¬(a 1 < a 2 ∧ a 2 < a 3) → ¬( ∀ n : ℕ, a n < a (n + 1) )

-- Contrapositive of Proposition P
def Contra_Prop_P (a : ℕ → ℝ) :=
  ¬( ∀ n : ℕ, a n < a (n + 1) ) → ¬(a 1 < a 2 ∧ a 2 < a 3)

-- Main theorem to be proved
theorem num_correct_props_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a → 
  Prop_P a ∧ Conv_Prop_P a ∧ Inv_Prop_P a ∧ Contra_Prop_P a := by
  sorry

end NUMINAMATH_GPT_num_correct_props_geometric_sequence_l588_58897


namespace NUMINAMATH_GPT_parabola_fixed_point_l588_58828

theorem parabola_fixed_point (t : ℝ) : ∃ y, y = 4 * 3^2 + 2 * t * 3 - 3 * t ∧ y = 36 :=
by
  exists 36
  sorry

end NUMINAMATH_GPT_parabola_fixed_point_l588_58828


namespace NUMINAMATH_GPT_probability_three_consecutive_heads_four_tosses_l588_58886

theorem probability_three_consecutive_heads_four_tosses :
  let total_outcomes := 16
  let favorable_outcomes := 2
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 1 / 8 := by
    sorry

end NUMINAMATH_GPT_probability_three_consecutive_heads_four_tosses_l588_58886


namespace NUMINAMATH_GPT_sean_whistles_l588_58838

def charles_whistles : ℕ := 13
def extra_whistles : ℕ := 32

theorem sean_whistles : charles_whistles + extra_whistles = 45 := by
  sorry

end NUMINAMATH_GPT_sean_whistles_l588_58838


namespace NUMINAMATH_GPT_martian_angle_conversion_l588_58801

-- Defines the full circle measurements
def full_circle_clerts : ℕ := 600
def full_circle_degrees : ℕ := 360
def angle_degrees : ℕ := 60

-- The main statement to prove
theorem martian_angle_conversion : 
    (full_circle_clerts * angle_degrees) / full_circle_degrees = 100 :=
by
  sorry  

end NUMINAMATH_GPT_martian_angle_conversion_l588_58801


namespace NUMINAMATH_GPT_muirheadable_decreasing_columns_iff_l588_58831

def isMuirheadable (n : ℕ) (grid : List (List ℕ)) : Prop :=
  -- Placeholder definition; the actual definition should specify the conditions
  sorry

theorem muirheadable_decreasing_columns_iff (n : ℕ) (h : n > 0) :
  (∃ grid : List (List ℕ), isMuirheadable n grid) ↔ n ≠ 3 :=
by 
  sorry

end NUMINAMATH_GPT_muirheadable_decreasing_columns_iff_l588_58831


namespace NUMINAMATH_GPT_eggs_broken_l588_58810

theorem eggs_broken (brown_eggs white_eggs total_pre total_post broken_eggs : ℕ) 
  (h1 : brown_eggs = 10)
  (h2 : white_eggs = 3 * brown_eggs)
  (h3 : total_pre = brown_eggs + white_eggs)
  (h4 : total_post = 20)
  (h5 : broken_eggs = total_pre - total_post) : broken_eggs = 20 :=
by
  sorry

end NUMINAMATH_GPT_eggs_broken_l588_58810


namespace NUMINAMATH_GPT_b_catches_a_distance_l588_58844

-- Define the initial conditions
def a_speed : ℝ := 10  -- A's speed in km/h
def b_speed : ℝ := 20  -- B's speed in km/h
def start_delay : ℝ := 3  -- B starts cycling 3 hours after A in hours

-- Define the target distance to prove
theorem b_catches_a_distance : ∃ (d : ℝ), d = 60 := 
by 
  sorry

end NUMINAMATH_GPT_b_catches_a_distance_l588_58844


namespace NUMINAMATH_GPT_ratio_of_sums_l588_58869

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2
axiom a4_eq_2a3 : a 4 = 2 * a 3

theorem ratio_of_sums (a : ℕ → ℝ) (S : ℕ → ℝ)
                      (arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2)
                      (a4_eq_2a3 : a 4 = 2 * a 3) :
  S 7 / S 5 = 14 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_of_sums_l588_58869


namespace NUMINAMATH_GPT_number_of_donuts_finished_l588_58898

-- Definitions from conditions
def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def total_spent : ℕ := 18

-- Theorem statement
theorem number_of_donuts_finished (H1 : ounces_per_donut = 2)
                                   (H2 : ounces_per_pot = 12)
                                   (H3 : cost_per_pot = 3)
                                   (H4 : total_spent = 18) : 
  ∃ n : ℕ, n = 36 :=
  sorry

end NUMINAMATH_GPT_number_of_donuts_finished_l588_58898


namespace NUMINAMATH_GPT_distance_from_plate_to_bottom_edge_l588_58850

theorem distance_from_plate_to_bottom_edge :
  ∀ (W T d : ℕ), W = 73 ∧ T = 20 ∧ (T + d = W) → d = 53 :=
by
  intros W T d
  rintro ⟨hW, hT, h⟩
  rw [hW, hT] at h
  linarith

end NUMINAMATH_GPT_distance_from_plate_to_bottom_edge_l588_58850


namespace NUMINAMATH_GPT_lindsey_savings_l588_58832

theorem lindsey_savings
  (september_savings : Nat := 50)
  (october_savings : Nat := 37)
  (november_savings : Nat := 11)
  (additional_savings : Nat := 25)
  (video_game_cost : Nat := 87)
  (total_savings := september_savings + october_savings + november_savings)
  (mom_bonus : Nat := if total_savings > 75 then additional_savings else 0)
  (final_amount := total_savings + mom_bonus - video_game_cost) :
  final_amount = 36 := by
  sorry

end NUMINAMATH_GPT_lindsey_savings_l588_58832


namespace NUMINAMATH_GPT_glorias_ratio_l588_58836

variable (Q : ℕ) -- total number of quarters
variable (dimes : ℕ) -- total number of dimes, given as 350
variable (quarters_left : ℕ) -- number of quarters left

-- Given conditions
def conditions (Q dimes quarters_left : ℕ) : Prop :=
  dimes = 350 ∧
  quarters_left = (3 * Q) / 5 ∧
  (dimes + quarters_left = 392)

-- The ratio of dimes to quarters left
def ratio_of_dimes_to_quarters_left (dimes quarters_left : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd dimes quarters_left
  (dimes / gcd, quarters_left / gcd)

theorem glorias_ratio (Q : ℕ) (quarters_left : ℕ) : conditions Q 350 quarters_left → ratio_of_dimes_to_quarters_left 350 quarters_left = (25, 3) := by 
  sorry

end NUMINAMATH_GPT_glorias_ratio_l588_58836


namespace NUMINAMATH_GPT_common_chord_circle_eq_l588_58843

theorem common_chord_circle_eq {a b : ℝ} (hb : b ≠ 0) :
  ∃ x y : ℝ, 
    (x^2 + y^2 - 2 * a * x = 0) ∧ 
    (x^2 + y^2 - 2 * b * y = 0) ∧ 
    (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0 :=
by sorry

end NUMINAMATH_GPT_common_chord_circle_eq_l588_58843


namespace NUMINAMATH_GPT_sin_330_eq_neg_half_l588_58827

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sin_330_eq_neg_half_l588_58827


namespace NUMINAMATH_GPT_complex_magnitude_of_3_minus_4i_l588_58848

open Complex

theorem complex_magnitude_of_3_minus_4i : Complex.abs ⟨3, -4⟩ = 5 := sorry

end NUMINAMATH_GPT_complex_magnitude_of_3_minus_4i_l588_58848


namespace NUMINAMATH_GPT_class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l588_58823

noncomputable def average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + pitch + innovation) / 3

noncomputable def weighted_average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + 7 * pitch + 2 * innovation) / 10

theorem class_7th_grade_1_has_higher_average_score :
  average_score 90 77 85 > average_score 74 95 80 :=
by sorry

theorem class_7th_grade_2_has_higher_weighted_score :
  weighted_average_score 74 95 80 > weighted_average_score 90 77 85 :=
by sorry

end NUMINAMATH_GPT_class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l588_58823


namespace NUMINAMATH_GPT_average_marbles_of_other_colors_l588_58808

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_marbles_of_other_colors_l588_58808


namespace NUMINAMATH_GPT_nonnegative_solution_exists_l588_58861

theorem nonnegative_solution_exists
  (a b c d n : ℕ)
  (h_npos : 0 < n)
  (h_gcd_abc : Nat.gcd (Nat.gcd a b) c = 1)
  (h_gcd_ab : Nat.gcd a b = d)
  (h_conds : n > a * b / d + c * d - a - b - c) :
  ∃ x y z : ℕ, a * x + b * y + c * z = n := 
by
  sorry

end NUMINAMATH_GPT_nonnegative_solution_exists_l588_58861


namespace NUMINAMATH_GPT_initial_books_l588_58804

variable (B : ℤ)

theorem initial_books (h1 : 4 / 6 * B = B - 3300) (h2 : 3300 = 2 / 6 * B) : B = 9900 :=
by
  sorry

end NUMINAMATH_GPT_initial_books_l588_58804


namespace NUMINAMATH_GPT_jim_juice_amount_l588_58877

def susan_juice : ℚ := 3 / 8
def jim_fraction : ℚ := 5 / 6

theorem jim_juice_amount : jim_fraction * susan_juice = 5 / 16 := by
  sorry

end NUMINAMATH_GPT_jim_juice_amount_l588_58877


namespace NUMINAMATH_GPT_seeds_sum_l588_58817

def Bom_seeds : ℕ := 300

def Gwi_seeds : ℕ := Bom_seeds + 40

def Yeon_seeds : ℕ := 3 * Gwi_seeds

def total_seeds : ℕ := Bom_seeds + Gwi_seeds + Yeon_seeds

theorem seeds_sum : total_seeds = 1660 := by
  sorry

end NUMINAMATH_GPT_seeds_sum_l588_58817


namespace NUMINAMATH_GPT_range_of_b_for_local_minimum_l588_58878

variable {x : ℝ}
variable (b : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ :=
  x^3 - 6 * b * x + 3 * b

def f' (x : ℝ) (b : ℝ) : ℝ :=
  3 * x^2 - 6 * b

theorem range_of_b_for_local_minimum
  (h1 : f' 0 b < 0)
  (h2 : f' 1 b > 0) :
  0 < b ∧ b < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_for_local_minimum_l588_58878


namespace NUMINAMATH_GPT_calculate_product_l588_58864

theorem calculate_product : (3 * 5 * 7 = 38) → (13 * 15 * 17 = 268) → 1 * 3 * 5 = 15 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_calculate_product_l588_58864


namespace NUMINAMATH_GPT_pear_price_is_6300_l588_58890

def price_of_pear (P : ℕ) : Prop :=
  P + (P + 2400) = 15000

theorem pear_price_is_6300 : ∃ (P : ℕ), price_of_pear P ∧ P = 6300 :=
by
  sorry

end NUMINAMATH_GPT_pear_price_is_6300_l588_58890


namespace NUMINAMATH_GPT_relay_race_total_distance_l588_58867

theorem relay_race_total_distance
  (Sadie_speed : ℝ) (Sadie_time : ℝ) (Ariana_speed : ℝ) (Ariana_time : ℝ) (Sarah_speed : ℝ) (total_race_time : ℝ)
  (h1 : Sadie_speed = 3) (h2 : Sadie_time = 2)
  (h3 : Ariana_speed = 6) (h4 : Ariana_time = 0.5)
  (h5 : Sarah_speed = 4) (h6 : total_race_time = 4.5) :
  (Sadie_speed * Sadie_time + Ariana_speed * Ariana_time + Sarah_speed * (total_race_time - (Sadie_time + Ariana_time))) = 17 :=
by
  sorry

end NUMINAMATH_GPT_relay_race_total_distance_l588_58867


namespace NUMINAMATH_GPT_min_value_m_l588_58863

theorem min_value_m (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + a 1)
  (h_geometric : ∀ n, b (n + 1) = b 1 * (b 1 ^ n))
  (h_b1_mean : 2 * b 1 = a 1 + a 2)
  (h_a3 : a 3 = 5)
  (h_b3 : b 3 = a 4 + 1)
  (h_S_formula : ∀ n, S n = n^2)
  (h_S_le_b : ∀ n ≥ 4, S n ≤ b n) :
  ∃ m, ∀ n, (n ≥ m → S n ≤ b n) ∧ m = 4 := sorry

end NUMINAMATH_GPT_min_value_m_l588_58863


namespace NUMINAMATH_GPT_greatest_possible_length_l588_58807

theorem greatest_possible_length :
  ∃ (g : ℕ), g = Nat.gcd 700 (Nat.gcd 385 1295) ∧ g = 35 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_length_l588_58807


namespace NUMINAMATH_GPT_determine_dress_and_notebooks_l588_58856

structure Girl :=
  (name : String)
  (dress_color : String)
  (notebook_color : String)

def colors := ["red", "yellow", "blue"]

def Sveta : Girl := ⟨"Sveta", "red", "red"⟩
def Ira : Girl := ⟨"Ira", "blue", "yellow"⟩
def Tania : Girl := ⟨"Tania", "yellow", "blue"⟩

theorem determine_dress_and_notebooks :
  (Sveta.dress_color = Sveta.notebook_color) ∧
  (¬ Tania.dress_color = "red") ∧
  (¬ Tania.notebook_color = "red") ∧
  (Ira.notebook_color = "yellow") ∧
  (Sveta ∈ [Sveta, Ira, Tania]) ∧
  (Ira ∈ [Sveta, Ira, Tania]) ∧
  (Tania ∈ [Sveta, Ira, Tania]) →
  ([Sveta, Ira, Tania] = 
   [{name := "Sveta", dress_color := "red", notebook_color := "red"},
    {name := "Ira", dress_color := "blue", notebook_color := "yellow"},
    {name := "Tania", dress_color := "yellow", notebook_color := "blue"}])
:=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_dress_and_notebooks_l588_58856


namespace NUMINAMATH_GPT_find_value_expression_l588_58812

theorem find_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 4)
  (h3 : z^2 + x * z + x^2 = 79) :
  x * y + y * z + x * z = 20 := 
sorry

end NUMINAMATH_GPT_find_value_expression_l588_58812


namespace NUMINAMATH_GPT_find_maximum_value_of_f_φ_has_root_l588_58883

open Set Real

noncomputable section

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := -6 * (sin x + cos x) - 3

-- Definition of the function φ(x)
def φ (x : ℝ) : ℝ := f x + 10

-- The assumptions on the interval
def interval := Icc 0 (π / 4)

-- Statement to prove that the maximum value of f(x) is -9
theorem find_maximum_value_of_f : ∀ x ∈ interval, f x ≤ -9 ∧ ∃ x_0 ∈ interval, f x_0 = -9 := sorry

-- Statement to prove that φ(x) has a root in the interval
theorem φ_has_root : ∃ x ∈ interval, φ x = 0 := sorry

end NUMINAMATH_GPT_find_maximum_value_of_f_φ_has_root_l588_58883


namespace NUMINAMATH_GPT_volume_of_region_l588_58876

theorem volume_of_region (r1 r2 : ℝ) (h : r1 = 5) (h2 : r2 = 8) : 
  let V_sphere (r : ℝ) := (4 / 3) * Real.pi * r^3
  let V_cylinder (r : ℝ) := Real.pi * r^2 * r
  (V_sphere r2) - (V_sphere r1) - (V_cylinder r1) = 391 * Real.pi :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_volume_of_region_l588_58876


namespace NUMINAMATH_GPT_bricks_in_wall_l588_58857

theorem bricks_in_wall (x : ℕ) (r₁ r₂ combined_rate : ℕ) :
  (r₁ = x / 8) →
  (r₂ = x / 12) →
  (combined_rate = r₁ + r₂ - 15) →
  (6 * combined_rate = x) →
  x = 360 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_bricks_in_wall_l588_58857


namespace NUMINAMATH_GPT_axis_of_symmetry_shift_l588_58872

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem axis_of_symmetry_shift (f : ℝ → ℝ) (hf : is_even_function f) :
  (∃ a : ℝ, ∀ x : ℝ, f (x + 1) = f (-(x + 1))) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_shift_l588_58872


namespace NUMINAMATH_GPT_possible_sums_of_digits_l588_58879

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_nonzero (A : ℕ) : Prop :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def reverse_number (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  1000 * d + 100 * c + 10 * b + a

def sum_of_digits (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a + b + c + d

theorem possible_sums_of_digits (A B : ℕ) 
  (h_four_digit : is_four_digit_number A) 
  (h_nonzero_digits : all_digits_nonzero A) 
  (h_reverse : B = reverse_number A) 
  (h_divisible : (A + B) % 109 = 0) : 
  sum_of_digits A = 14 ∨ sum_of_digits A = 23 ∨ sum_of_digits A = 28 := 
sorry

end NUMINAMATH_GPT_possible_sums_of_digits_l588_58879


namespace NUMINAMATH_GPT_intersection_P_Q_l588_58891

def P (k : ℤ) (α : ℝ) : Prop := 2 * k * Real.pi ≤ α ∧ α ≤ (2 * k + 1) * Real.pi
def Q (α : ℝ) : Prop := -4 ≤ α ∧ α ≤ 4

theorem intersection_P_Q :
  (∃ k : ℤ, P k α) ∧ Q α ↔ (-4 ≤ α ∧ α ≤ -Real.pi) ∨ (0 ≤ α ∧ α ≤ Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l588_58891


namespace NUMINAMATH_GPT_greatest_multiple_of_30_less_than_800_l588_58815

theorem greatest_multiple_of_30_less_than_800 : 
    ∃ n : ℤ, (n % 30 = 0) ∧ (n < 800) ∧ (∀ m : ℤ, (m % 30 = 0) ∧ (m < 800) → m ≤ n) ∧ n = 780 :=
by
  sorry

end NUMINAMATH_GPT_greatest_multiple_of_30_less_than_800_l588_58815


namespace NUMINAMATH_GPT_square_of_1023_l588_58821

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end NUMINAMATH_GPT_square_of_1023_l588_58821


namespace NUMINAMATH_GPT_stamps_on_last_page_l588_58854

theorem stamps_on_last_page (total_books : ℕ) (pages_per_book : ℕ) (stamps_per_page_initial : ℕ) (stamps_per_page_new : ℕ)
    (full_books_new : ℕ) (pages_filled_seventh_book : ℕ) (total_stamps : ℕ) (stamps_in_seventh_book : ℕ) 
    (remaining_stamps : ℕ) :
    total_books = 10 →
    pages_per_book = 50 →
    stamps_per_page_initial = 8 →
    stamps_per_page_new = 12 →
    full_books_new = 6 →
    pages_filled_seventh_book = 37 →
    total_stamps = total_books * pages_per_book * stamps_per_page_initial →
    stamps_in_seventh_book = 4000 - (600 * full_books_new) →
    remaining_stamps = stamps_in_seventh_book - (pages_filled_seventh_book * stamps_per_page_new) →
    remaining_stamps = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_stamps_on_last_page_l588_58854


namespace NUMINAMATH_GPT_inequality_solution_set_compare_mn_and_2m_plus_2n_l588_58852

def f (x : ℝ) : ℝ := |x| + |x - 3|

theorem inequality_solution_set :
  {x : ℝ | f x - 5 ≥ x} = { x : ℝ | x ≤ -2 / 3 } ∪ { x : ℝ | x ≥ 8 } :=
sorry

theorem compare_mn_and_2m_plus_2n (m n : ℝ) (hm : ∃ x, m = f x) (hn : ∃ x, n = f x) :
  2 * (m + n) < m * n + 4 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_compare_mn_and_2m_plus_2n_l588_58852


namespace NUMINAMATH_GPT_alice_instructors_l588_58889

noncomputable def num_students : ℕ := 40
noncomputable def num_life_vests_Alice_has : ℕ := 20
noncomputable def percent_students_with_their_vests : ℕ := 20
noncomputable def num_additional_life_vests_needed : ℕ := 22

-- Constants based on calculated conditions
noncomputable def num_students_with_their_vests : ℕ := (percent_students_with_their_vests * num_students) / 100
noncomputable def num_students_without_their_vests : ℕ := num_students - num_students_with_their_vests
noncomputable def num_life_vests_needed_for_students : ℕ := num_students_without_their_vests - num_life_vests_Alice_has
noncomputable def num_life_vests_needed_for_instructors : ℕ := num_additional_life_vests_needed - num_life_vests_needed_for_students

theorem alice_instructors : num_life_vests_needed_for_instructors = 10 := 
by
  sorry

end NUMINAMATH_GPT_alice_instructors_l588_58889


namespace NUMINAMATH_GPT_sum_of_two_numbers_l588_58814

theorem sum_of_two_numbers (a b : ℝ) (h1 : a + b = 25) (h2 : a * b = 144) (h3 : |a - b| = 7) : a + b = 25 := 
  by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l588_58814


namespace NUMINAMATH_GPT_total_flowers_l588_58800

def tulips : ℕ := 3
def carnations : ℕ := 4

theorem total_flowers : tulips + carnations = 7 := by
  sorry

end NUMINAMATH_GPT_total_flowers_l588_58800


namespace NUMINAMATH_GPT_hall_width_l588_58895

theorem hall_width
  (L H E C : ℝ)
  (hL : L = 20)
  (hH : H = 5)
  (hE : E = 57000)
  (hC : C = 60) :
  ∃ w : ℝ, (w * 50 + 100) * C = E ∧ w = 17 :=
by
  use 17
  simp [hL, hH, hE, hC]
  sorry

end NUMINAMATH_GPT_hall_width_l588_58895


namespace NUMINAMATH_GPT_perimeter_ratio_of_divided_square_l588_58859

theorem perimeter_ratio_of_divided_square
  (S_ΔADE : ℝ) (S_EDCB : ℝ)
  (S_ratio : S_ΔADE / S_EDCB = 5 / 19)
  : ∃ (perim_ΔADE perim_EDCB : ℝ),
  perim_ΔADE / perim_EDCB = 15 / 22 :=
by
  -- Let S_ΔADE = 5x and S_EDCB = 19x
  -- x can be calculated based on the given S_ratio = 5/19
  -- Apply geometric properties and simplifications analogous to the described solution.
  sorry

end NUMINAMATH_GPT_perimeter_ratio_of_divided_square_l588_58859


namespace NUMINAMATH_GPT_glass_volume_l588_58802

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end NUMINAMATH_GPT_glass_volume_l588_58802


namespace NUMINAMATH_GPT_cherries_in_mix_l588_58829

theorem cherries_in_mix (total_fruit : ℕ) (blueberries : ℕ) (raspberries : ℕ) (cherries : ℕ) 
  (H1 : total_fruit = 300)
  (H2: raspberries = 3 * blueberries)
  (H3: cherries = 5 * blueberries)
  (H4: total_fruit = blueberries + raspberries + cherries) : cherries = 167 :=
by
  sorry

end NUMINAMATH_GPT_cherries_in_mix_l588_58829


namespace NUMINAMATH_GPT_poly_coeff_sum_l588_58816

theorem poly_coeff_sum :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ,
  (∀ x : ℤ, ((x^2 + 1) * (x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 + a_11 * x^11))
  ∧ a_0 = -512) →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510) :=
by
  sorry

end NUMINAMATH_GPT_poly_coeff_sum_l588_58816


namespace NUMINAMATH_GPT_lewis_earnings_during_harvest_l588_58805

-- Define the conditions
def regular_earnings_per_week : ℕ := 28
def overtime_earnings_per_week : ℕ := 939
def number_of_weeks : ℕ := 1091

-- Define the total earnings per week
def total_earnings_per_week := regular_earnings_per_week + overtime_earnings_per_week

-- Define the total earnings during the harvest season
def total_earnings_during_harvest := total_earnings_per_week * number_of_weeks

-- Theorem statement
theorem lewis_earnings_during_harvest : total_earnings_during_harvest = 1055497 := by
  sorry

end NUMINAMATH_GPT_lewis_earnings_during_harvest_l588_58805


namespace NUMINAMATH_GPT_octagon_non_intersecting_diagonals_l588_58853

-- Define what an octagon is
def octagon : Type := { vertices : Finset (Fin 8) // vertices.card = 8 }

-- Define non-intersecting diagonals in an octagon
def non_intersecting_diagonals (oct : octagon) : ℕ :=
  8  -- Given the cyclic pattern and star formation, we know the number is 8

-- The theorem we want to prove
theorem octagon_non_intersecting_diagonals (oct : octagon) : non_intersecting_diagonals oct = 8 :=
by sorry

end NUMINAMATH_GPT_octagon_non_intersecting_diagonals_l588_58853


namespace NUMINAMATH_GPT_find_m_l588_58841

theorem find_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 3, 2*m - 1}) (hB: B = {3, m^2}) (h_subset: B ⊆ A) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l588_58841


namespace NUMINAMATH_GPT_fraction_of_calls_processed_by_team_B_l588_58834

theorem fraction_of_calls_processed_by_team_B
  (C_B : ℕ) -- the number of calls processed by each member of team B
  (B : ℕ)  -- the number of call center agents in team B
  (C_A : ℕ := C_B / 5) -- each member of team A processes 1/5 the number of calls as each member of team B
  (A : ℕ := 5 * B / 8) -- team A has 5/8 as many agents as team B
: 
  (B * C_B) / ((A * C_A) + (B * C_B)) = (8 / 9 : ℚ) :=
sorry

end NUMINAMATH_GPT_fraction_of_calls_processed_by_team_B_l588_58834


namespace NUMINAMATH_GPT_problem_statement_l588_58892

variables {a b c p q r : ℝ}

-- Given conditions
axiom h1 : 19 * p + b * q + c * r = 0
axiom h2 : a * p + 29 * q + c * r = 0
axiom h3 : a * p + b * q + 56 * r = 0
axiom h4 : a ≠ 19
axiom h5 : p ≠ 0

-- Statement to prove
theorem problem_statement : 
  (a / (a - 19)) + (b / (b - 29)) + (c / (c - 56)) = 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l588_58892


namespace NUMINAMATH_GPT_geometric_sequence_general_term_arithmetic_sequence_sum_l588_58830

variable {n : ℕ}

-- Defining sequences and sums
def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℕ := sorry
def T (n : ℕ) : ℕ := sorry
def b (n : ℕ) : ℕ := sorry

-- Given conditions
axiom h1 : 2 * S n = 3 * a n - 3
axiom h2 : b 1 = a 1
axiom h3 : b 7 = b 1 * b 2
axiom a1_value : a 1 = 3
axiom d_value : ∃ d : ℕ, b 2 = b 1 + d ∧ b 7 = b 1 + 6 * d

theorem geometric_sequence_general_term : a n = 3 ^ n :=
by sorry

theorem arithmetic_sequence_sum : T n = n^2 + 2*n :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_arithmetic_sequence_sum_l588_58830


namespace NUMINAMATH_GPT_simultaneous_in_Quadrant_I_l588_58874

def in_Quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem simultaneous_in_Quadrant_I (c x y : ℝ) : 
  (2 * x - y = 5) ∧ (c * x + y = 4) ↔ in_Quadrant_I x y ∧ (-2 < c ∧ c < 8 / 5) :=
sorry

end NUMINAMATH_GPT_simultaneous_in_Quadrant_I_l588_58874


namespace NUMINAMATH_GPT_sum_3x_4y_l588_58865

theorem sum_3x_4y (x y N : ℝ) (H1 : 3 * x + 4 * y = N) (H2 : 6 * x - 4 * y = 12) (H3 : x * y = 72) : 3 * x + 4 * y = 60 := 
sorry

end NUMINAMATH_GPT_sum_3x_4y_l588_58865


namespace NUMINAMATH_GPT_quadratic_range_l588_58847

noncomputable def quadratic_condition (a m : ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (- (1 + 1 / m) > 0) ∧
  (3 * m^2 - 2 * m - 1 ≤ 0)

theorem quadratic_range (a m : ℝ) :
  quadratic_condition a m → - (1 / 3) ≤ m ∧ m < 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_range_l588_58847


namespace NUMINAMATH_GPT_grade_point_average_one_third_classroom_l588_58813

theorem grade_point_average_one_third_classroom
  (gpa1 : ℝ) -- grade point average of one third of the classroom
  (gpa_rest : ℝ) -- grade point average of the rest of the classroom
  (gpa_whole : ℝ) -- grade point average of the whole classroom
  (h_rest : gpa_rest = 45)
  (h_whole : gpa_whole = 48) :
  gpa1 = 54 :=
by
  sorry

end NUMINAMATH_GPT_grade_point_average_one_third_classroom_l588_58813


namespace NUMINAMATH_GPT_max_songs_played_l588_58887

theorem max_songs_played (n m t : ℕ) (h1 : n = 50) (h2 : m = 50) (h3 : t = 180) :
  3 * n + 5 * (m - ((t - 3 * n) / 5)) = 56 :=
by
  sorry

end NUMINAMATH_GPT_max_songs_played_l588_58887


namespace NUMINAMATH_GPT_number_of_3digit_even_numbers_divisible_by_9_l588_58825

theorem number_of_3digit_even_numbers_divisible_by_9 : 
    ∃ n : ℕ, (n = 50) ∧
    (∀ k, (108 + (k - 1) * 18 = 990) ↔ (108 ≤ 108 + (k - 1) * 18 ∧ 108 + (k - 1) * 18 ≤ 999)) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_3digit_even_numbers_divisible_by_9_l588_58825


namespace NUMINAMATH_GPT_least_common_multiple_l588_58822

theorem least_common_multiple (x : ℕ) (hx : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end NUMINAMATH_GPT_least_common_multiple_l588_58822


namespace NUMINAMATH_GPT_mike_took_23_green_marbles_l588_58885

-- Definition of the conditions
def original_green_marbles : ℕ := 32
def remaining_green_marbles : ℕ := 9

-- Definition of the statement we want to prove
theorem mike_took_23_green_marbles : original_green_marbles - remaining_green_marbles = 23 := by
  sorry

end NUMINAMATH_GPT_mike_took_23_green_marbles_l588_58885


namespace NUMINAMATH_GPT_cube_inverse_sum_l588_58873

theorem cube_inverse_sum (x : ℂ) (h : x + 1/x = -3) : x^3 + (1/x)^3 = -18 :=
by
  sorry

end NUMINAMATH_GPT_cube_inverse_sum_l588_58873


namespace NUMINAMATH_GPT_algebraic_expression_value_l588_58899

theorem algebraic_expression_value (a b : ℝ) 
  (h : |a + 2| + (b - 1)^2 = 0) : (a + b) ^ 2005 = -1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l588_58899


namespace NUMINAMATH_GPT_xyz_final_stock_price_l588_58826

def initial_stock_price : ℝ := 120
def first_year_increase_rate : ℝ := 0.80
def second_year_decrease_rate : ℝ := 0.30

def final_stock_price_after_two_years : ℝ :=
  (initial_stock_price * (1 + first_year_increase_rate)) * (1 - second_year_decrease_rate)

theorem xyz_final_stock_price :
  final_stock_price_after_two_years = 151.2 := by
  sorry

end NUMINAMATH_GPT_xyz_final_stock_price_l588_58826


namespace NUMINAMATH_GPT_find_y_l588_58811

theorem find_y (x y : ℕ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : y = 1 :=
sorry

end NUMINAMATH_GPT_find_y_l588_58811


namespace NUMINAMATH_GPT_abs_x_equals_4_l588_58896

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_x_equals_4_l588_58896


namespace NUMINAMATH_GPT_function_pair_solution_l588_58818

-- Define the conditions for f and g
variables (f g : ℝ → ℝ)

-- Define the main hypothesis
def main_hypothesis : Prop := 
∀ (x y : ℝ), 
  x ≠ 0 → y ≠ 0 → 
  f (x + y) = g (1/x + 1/y) * (x * y) ^ 2008

-- The theorem that proves f and g are of the given form
theorem function_pair_solution (c : ℝ) (h : main_hypothesis f g) : 
  (∀ x, f x = c * x ^ 2008) ∧ 
  (∀ x, g x = c * x ^ 2008) :=
sorry

end NUMINAMATH_GPT_function_pair_solution_l588_58818


namespace NUMINAMATH_GPT_math_proof_l588_58881

def problem_statement : Prop :=
  ∃ x : ℕ, (2 * x + 3 = 19) ∧ (x + (2 * x + 3) = 27)

theorem math_proof : problem_statement :=
  sorry

end NUMINAMATH_GPT_math_proof_l588_58881


namespace NUMINAMATH_GPT_number_of_best_friends_l588_58855

-- Constants and conditions
def initial_tickets : ℕ := 37
def tickets_per_friend : ℕ := 5
def tickets_left : ℕ := 2

-- Problem statement
theorem number_of_best_friends : (initial_tickets - tickets_left) / tickets_per_friend = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_best_friends_l588_58855


namespace NUMINAMATH_GPT_jar_initial_water_fraction_l588_58894

theorem jar_initial_water_fraction (C W : ℝ) (hC : C > 0) (hW : W + C / 4 = 0.75 * C) : W / C = 0.5 :=
by
  -- necessary parameters and sorry for the proof 
  sorry

end NUMINAMATH_GPT_jar_initial_water_fraction_l588_58894


namespace NUMINAMATH_GPT_number_of_routes_A_to_B_l588_58837

theorem number_of_routes_A_to_B :
  (∃ f : ℕ × ℕ → ℕ,
  (∀ n m, f (n + 1, m) = f (n, m) + f (n + 1, m - 1)) ∧
  f (0, 0) = 1 ∧ 
  (∀ i, f (i, 0) = 1) ∧ 
  (∀ j, f (0, j) = 1) ∧ 
  f (3, 5) = 23) :=
sorry

end NUMINAMATH_GPT_number_of_routes_A_to_B_l588_58837


namespace NUMINAMATH_GPT_color_triplet_exists_l588_58840

theorem color_triplet_exists (color : ℕ → Prop) :
  (∀ n, color n ∨ ¬ color n) → ∃ x y z : ℕ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ color x = color y ∧ color y = color z ∧ x * y = z ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_color_triplet_exists_l588_58840


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l588_58819

-- Definitions of the sets M and N
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Statement of the theorem proving the intersection of M and N
theorem intersection_of_M_and_N :
  M ∩ N = {2, 3} :=
by sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l588_58819


namespace NUMINAMATH_GPT_number_of_people_l588_58871

theorem number_of_people (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 2 * x + y / 2 + z / 4 = 12) : 
  x = 5 ∧ y = 1 ∧ z = 6 := 
by
  sorry

end NUMINAMATH_GPT_number_of_people_l588_58871


namespace NUMINAMATH_GPT_jacob_walked_8_miles_l588_58820

theorem jacob_walked_8_miles (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 := by
  -- conditions
  have hr : rate = 4 := h_rate
  have ht : time = 2 := h_time
  -- problem
  sorry

end NUMINAMATH_GPT_jacob_walked_8_miles_l588_58820


namespace NUMINAMATH_GPT_part_one_part_two_l588_58860

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- (1) Prove that if a = 2, then ∀ x, f(x, 2) ≤ 6 implies -1 ≤ x ≤ 3
theorem part_one (x : ℝ) : f x 2 ≤ 6 → -1 ≤ x ∧ x ≤ 3 :=
by sorry

-- (2) Prove that ∀ a ∈ ℝ, ∀ x ∈ ℝ, (f(x, a) + g(x) ≥ 3 → a ∈ [2, +∞))
theorem part_two (a x : ℝ) : f x a + g x ≥ 3 → 2 ≤ a :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l588_58860


namespace NUMINAMATH_GPT_triangles_in_pentadecagon_l588_58858

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end NUMINAMATH_GPT_triangles_in_pentadecagon_l588_58858


namespace NUMINAMATH_GPT_euler_phi_divisibility_l588_58846

def euler_phi (n : ℕ) : ℕ := sorry -- Placeholder for the Euler phi-function

theorem euler_phi_divisibility (n : ℕ) (hn : n > 0) :
    2^(n * (n + 1)) ∣ 32 * euler_phi (2^(2^n) - 1) :=
sorry

end NUMINAMATH_GPT_euler_phi_divisibility_l588_58846


namespace NUMINAMATH_GPT_smaller_number_is_180_l588_58851

theorem smaller_number_is_180 (a b : ℕ) (h1 : a = 3 * b) (h2 : a + 4 * b = 420) :
  a = 180 :=
sorry

end NUMINAMATH_GPT_smaller_number_is_180_l588_58851


namespace NUMINAMATH_GPT_find_point_on_line_and_distance_l588_58833

def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem find_point_on_line_and_distance :
  ∃ P : ℝ × ℝ, (2 * P.1 - 3 * P.2 + 5 = 0) ∧ (distance P (2, 3) = 13) →
  (P = (5, 5) ∨ P = (-1, 1)) :=
by
  sorry

end NUMINAMATH_GPT_find_point_on_line_and_distance_l588_58833


namespace NUMINAMATH_GPT_x_plus_y_value_l588_58868

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem x_plus_y_value :
  let x := sum_of_integers 50 70
  let y := count_even_integers 50 70
  x + y = 1271 := by
    let x := sum_of_integers 50 70
    let y := count_even_integers 50 70
    sorry

end NUMINAMATH_GPT_x_plus_y_value_l588_58868


namespace NUMINAMATH_GPT_inequality_a_b_c_l588_58835

noncomputable def a := Real.log (Real.pi / 3)
noncomputable def b := Real.log (Real.exp 1 / 3)
noncomputable def c := Real.exp (0.5)

theorem inequality_a_b_c : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_inequality_a_b_c_l588_58835


namespace NUMINAMATH_GPT_f_1993_of_3_l588_58842

def f (x : ℚ) := (1 + x) / (1 - 3 * x)

def f_n (x : ℚ) : ℕ → ℚ
| 0 => x
| (n + 1) => f (f_n x n)

theorem f_1993_of_3 :
  f_n 3 1993 = 1 / 5 :=
sorry

end NUMINAMATH_GPT_f_1993_of_3_l588_58842
