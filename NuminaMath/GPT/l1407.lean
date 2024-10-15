import Mathlib

namespace NUMINAMATH_GPT_ratio_sum_product_is_constant_l1407_140708

variables {p a : ℝ} (h_a : 0 < a)
theorem ratio_sum_product_is_constant
    (k : ℝ) (h_k : k ≠ 0)
    (x₁ x₂ : ℝ) (h_intersection : x₁ * (2 * p * (x₂ - a)) = 2 * p * (x₁ - a) ∧ x₂ * (2 * p * (x₁ - a)) = 2 * p * (x₂ - a)) :
  (x₁ + x₂) / (x₁ * x₂) = 1 / a := by
  sorry

end NUMINAMATH_GPT_ratio_sum_product_is_constant_l1407_140708


namespace NUMINAMATH_GPT_min_value_of_diff_squares_l1407_140763

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))

theorem min_value_of_diff_squares (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  ∃ minimum_value, minimum_value = 36 ∧ ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → (C x y z)^2 - (D x y z)^2 ≥ minimum_value :=
sorry

end NUMINAMATH_GPT_min_value_of_diff_squares_l1407_140763


namespace NUMINAMATH_GPT_range_of_a_l1407_140714

theorem range_of_a (x : ℝ) (a : ℝ) (hx : 0 < x ∧ x < 4) : |x - 1| < a → a ≥ 3 := sorry

end NUMINAMATH_GPT_range_of_a_l1407_140714


namespace NUMINAMATH_GPT_set_intersection_A_B_l1407_140785

def A := {x : ℝ | 2 * x - x^2 > 0}
def B := {x : ℝ | x > 1}
def I := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_A_B :
  A ∩ B = I :=
sorry

end NUMINAMATH_GPT_set_intersection_A_B_l1407_140785


namespace NUMINAMATH_GPT_allison_total_supply_items_is_28_l1407_140778

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end NUMINAMATH_GPT_allison_total_supply_items_is_28_l1407_140778


namespace NUMINAMATH_GPT_triangle_area_not_twice_parallelogram_l1407_140773

theorem triangle_area_not_twice_parallelogram (b h : ℝ) :
  (1 / 2) * b * h ≠ 2 * b * h :=
sorry

end NUMINAMATH_GPT_triangle_area_not_twice_parallelogram_l1407_140773


namespace NUMINAMATH_GPT_range_of_x_minus_cos_y_l1407_140712

theorem range_of_x_minus_cos_y {x y : ℝ} (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (a b : ℝ), ∀ z, z = x - Real.cos y → a ≤ z ∧ z ≤ b ∧ a = -1 ∧ b = 1 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_minus_cos_y_l1407_140712


namespace NUMINAMATH_GPT_percentage_of_water_in_mixture_l1407_140711

-- Conditions
def percentage_water_LiquidA : ℝ := 0.10
def percentage_water_LiquidB : ℝ := 0.15
def percentage_water_LiquidC : ℝ := 0.25

def volume_LiquidA (v : ℝ) : ℝ := 4 * v
def volume_LiquidB (v : ℝ) : ℝ := 3 * v
def volume_LiquidC (v : ℝ) : ℝ := 2 * v

-- Proof
theorem percentage_of_water_in_mixture (v : ℝ) :
  (percentage_water_LiquidA * volume_LiquidA v + percentage_water_LiquidB * volume_LiquidB v + percentage_water_LiquidC * volume_LiquidC v) / (volume_LiquidA v + volume_LiquidB v + volume_LiquidC v) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_water_in_mixture_l1407_140711


namespace NUMINAMATH_GPT_inequality_proof_equality_case_l1407_140744

variables (x y z : ℝ)
  
theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) : 
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 := 
sorry

theorem equality_case 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) 
  (h_eq : (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1) :
  x = 1 ∧ y = 1 ∧ z = 1 := 
sorry

end NUMINAMATH_GPT_inequality_proof_equality_case_l1407_140744


namespace NUMINAMATH_GPT_find_a_and_x_range_l1407_140702

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a_and_x_range :
  (∃ a, (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3)) →
  (∀ x, ∃ a, f x a ≤ 5 → 
    ((a = 1 → (0 ≤ x ∧ x ≤ 5)) ∧
     (a = 7 → (3 ≤ x ∧ x ≤ 8)))) :=
by sorry

end NUMINAMATH_GPT_find_a_and_x_range_l1407_140702


namespace NUMINAMATH_GPT_factor_polynomial_l1407_140762

def p (x y z : ℝ) : ℝ := x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2)

theorem factor_polynomial (x y z : ℝ) : 
  p x y z = (x - y) * (y - z) * (z - x) * -(x * y + x * z + y * z) :=
by 
  simp [p]
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1407_140762


namespace NUMINAMATH_GPT_cricket_player_average_l1407_140741

theorem cricket_player_average (A : ℕ)
  (H1 : 10 * A + 62 = 11 * (A + 4)) : A = 18 :=
by {
  sorry -- The proof itself
}

end NUMINAMATH_GPT_cricket_player_average_l1407_140741


namespace NUMINAMATH_GPT_equality_of_coefficients_l1407_140794

open Real

theorem equality_of_coefficients (a b c x : ℝ)
  (h1 : a * x^2 - b * x - c = b * x^2 - c * x - a)
  (h2 : b * x^2 - c * x - a = c * x^2 - a * x - b)
  (h3 : c * x^2 - a * x - b = a * x^2 - b * x - c):
  a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_equality_of_coefficients_l1407_140794


namespace NUMINAMATH_GPT_roots_of_polynomial_l1407_140730

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^3 - 3 * x^2 + 2 * x) * (x - 5) = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l1407_140730


namespace NUMINAMATH_GPT_isosceles_base_l1407_140703

theorem isosceles_base (s b : ℕ) 
  (h1 : 3 * s = 45) 
  (h2 : 2 * s + b = 40) 
  (h3 : s = 15): 
  b = 10 :=
  sorry

end NUMINAMATH_GPT_isosceles_base_l1407_140703


namespace NUMINAMATH_GPT_graph_does_not_pass_second_quadrant_l1407_140774

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h₀ : 1 < a) (h₁ : b < -1) : 
∀ x : ℝ, ¬ (y = a^x + b ∧ y > 0 ∧ x < 0) :=
by
  sorry

end NUMINAMATH_GPT_graph_does_not_pass_second_quadrant_l1407_140774


namespace NUMINAMATH_GPT_random_event_proof_l1407_140758

def is_certain_event (event: Prop) : Prop := ∃ h: event → true, ∃ h': true → event, true
def is_impossible_event (event: Prop) : Prop := event → false
def is_random_event (event: Prop) : Prop := ¬is_certain_event event ∧ ¬is_impossible_event event

def cond1 : Prop := sorry -- Yingying encounters a green light
def cond2 : Prop := sorry -- A non-transparent bag contains one ping-pong ball and two glass balls of the same size, and a ping-pong ball is drawn from it.
def cond3 : Prop := sorry -- You are currently answering question 12 of this test paper.
def cond4 : Prop := sorry -- The highest temperature in our city tomorrow will be 60°C.

theorem random_event_proof : 
  is_random_event cond1 ∧ 
  ¬is_random_event cond2 ∧ 
  ¬is_random_event cond3 ∧ 
  ¬is_random_event cond4 :=
by
  sorry

end NUMINAMATH_GPT_random_event_proof_l1407_140758


namespace NUMINAMATH_GPT_correct_statements_for_sequence_l1407_140799

theorem correct_statements_for_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  -- Statement 1
  (S_n = n^2 + n → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 2
  (S_n = 2^n - 1 → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1)) ∧
  -- Statement 3
  (∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1) → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 4
  (¬(∀ n, n ≥ 2 → a n^2 = a (n + 1) * a (n - 1) → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1))) :=
sorry

end NUMINAMATH_GPT_correct_statements_for_sequence_l1407_140799


namespace NUMINAMATH_GPT_plane_equation_correct_l1407_140782

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p q : Point3D) : Point3D :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def plane_eq (n : Point3D) (A : Point3D) : Point3D → ℝ :=
  fun P => n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

def is_perpendicular_plane (A B C : Point3D) (D : Point3D → ℝ) : Prop :=
  let BC := vector_sub C B
  D = plane_eq BC A

theorem plane_equation_correct :
  let A := { x := 7, y := -5, z := 1 }
  let B := { x := 5, y := -1, z := -3 }
  let C := { x := 3, y := 0, z := -4 }
  is_perpendicular_plane A B C (fun P => -2 * P.x + P.y - P.z + 20) :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_correct_l1407_140782


namespace NUMINAMATH_GPT_number_of_yogurts_l1407_140718

def slices_per_yogurt : Nat := 8
def slices_per_banana : Nat := 10
def number_of_bananas : Nat := 4

theorem number_of_yogurts (slices_per_yogurt slices_per_banana number_of_bananas : Nat) : 
  slices_per_yogurt = 8 → 
  slices_per_banana = 10 → 
  number_of_bananas = 4 → 
  (number_of_bananas * slices_per_banana) / slices_per_yogurt = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_of_yogurts_l1407_140718


namespace NUMINAMATH_GPT_solve_inequality_l1407_140796

-- Defining the inequality
def inequality (x : ℝ) : Prop := 1 / (x - 1) ≤ 1

-- Stating the theorem
theorem solve_inequality :
  { x : ℝ | inequality x } = { x : ℝ | x < 1 } ∪ { x : ℝ | 2 ≤ x } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1407_140796


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1407_140720

namespace Proofs

theorem solve_equation_1 (x : ℝ) :
  (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 :=
by
  sorry

end Proofs

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1407_140720


namespace NUMINAMATH_GPT_original_number_l1407_140752

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_list (n : ℕ) (a b c d e : ℕ) : Prop :=
  n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e

def four_digit_variant (N n : ℕ) (a b c d e : ℕ) : Prop :=
  (n = 10^3 * b + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d)

theorem original_number (N : ℕ) (a b c d e : ℕ) 
  (h1 : is_five_digit N) 
  (h2 : digit_list N a b c d e)
  (h3 : ∃ n, is_five_digit n ∧ four_digit_variant N n a b c d e ∧ N + n = 54321) :
  N = 49383 := 
sorry

end NUMINAMATH_GPT_original_number_l1407_140752


namespace NUMINAMATH_GPT_find_weight_of_B_l1407_140740

theorem find_weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) : B = 33 :=
by 
  sorry

end NUMINAMATH_GPT_find_weight_of_B_l1407_140740


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l1407_140725

theorem inequality_proof (a : ℝ) : (a^2 + 5)^2 + 4 * a * (10 - a) ≥ 8 * a^3  :=
by sorry

theorem equality_condition (a : ℝ) : ((a^2 + 5)^2 + 4 * a * (10 - a) = 8 * a^3) ↔ (a = 5 ∨ a = -1) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_equality_condition_l1407_140725


namespace NUMINAMATH_GPT_ratio_of_rectangle_to_square_l1407_140739

theorem ratio_of_rectangle_to_square (s w h : ℝ) 
  (hs : h = s / 2)
  (shared_area_ABCD_EFGH_1 : 0.25 * s^2 = 0.4 * w * h)
  (shared_area_ABCD_EFGH_2 : 0.25 * s^2 = 0.4 * w * h) :
  w / h = 2.5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_rectangle_to_square_l1407_140739


namespace NUMINAMATH_GPT_locus_of_D_l1407_140757

theorem locus_of_D 
  (a b : ℝ)
  (hA : 0 ≤ a ∧ a ≤ (2 * Real.sqrt 3 / 3))
  (hB : 0 ≤ b ∧ b ≤ (2 * Real.sqrt 3 / 3))
  (AB_eq : Real.sqrt ((b - 2 * a)^2 + (Real.sqrt 3 * b)^2)  = 2) :
  3 * (b - a / 2)^2 + (Real.sqrt 3 / 2 * (a + b))^2 / 3 = 1 :=
sorry

end NUMINAMATH_GPT_locus_of_D_l1407_140757


namespace NUMINAMATH_GPT_ratio_problem_l1407_140747

-- Given condition: a, b, c are in the ratio 2:3:4
theorem ratio_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : a / c = 2 / 4) : 
  (a - b + c) / b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_ratio_problem_l1407_140747


namespace NUMINAMATH_GPT_combination_of_students_l1407_140732

-- Define the conditions
def num_boys := 4
def num_girls := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Calculate possible combinations
def two_boys_one_girl : ℕ :=
  combination num_boys 2 * combination num_girls 1

def one_boy_two_girls : ℕ :=
  combination num_boys 1 * combination num_girls 2

-- Total combinations
def total_combinations : ℕ :=
  two_boys_one_girl + one_boy_two_girls

-- Lean statement to be proven
theorem combination_of_students :
  total_combinations = 30 :=
by sorry

end NUMINAMATH_GPT_combination_of_students_l1407_140732


namespace NUMINAMATH_GPT_cubic_identity_l1407_140777

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 11) (h3 : abc = -6) : a^3 + b^3 + c^3 = 94 :=
by
  sorry

end NUMINAMATH_GPT_cubic_identity_l1407_140777


namespace NUMINAMATH_GPT_quinton_cupcakes_l1407_140728

theorem quinton_cupcakes (students_Delmont : ℕ) (students_Donnelly : ℕ)
                         (num_teachers_nurse_principal : ℕ) (leftover : ℕ) :
  students_Delmont = 18 → students_Donnelly = 16 →
  num_teachers_nurse_principal = 4 → leftover = 2 →
  students_Delmont + students_Donnelly + num_teachers_nurse_principal + leftover = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_quinton_cupcakes_l1407_140728


namespace NUMINAMATH_GPT_ball_count_proof_l1407_140770

noncomputable def valid_ball_count : ℕ :=
  150

def is_valid_ball_count (N : ℕ) : Prop :=
  80 < N ∧ N ≤ 200 ∧
  (∃ y b w r : ℕ,
    y = Nat.div (12 * N) 100 ∧
    b = Nat.div (20 * N) 100 ∧
    w = 2 * Nat.div N 3 ∧
    r = N - (y + b + w) ∧
    r.mod N = 0 )

theorem ball_count_proof : is_valid_ball_count valid_ball_count :=
by
  -- The proof would be inserted here.
  sorry

end NUMINAMATH_GPT_ball_count_proof_l1407_140770


namespace NUMINAMATH_GPT_num_pass_students_is_85_l1407_140734

theorem num_pass_students_is_85 (T P F : ℕ) (avg_all avg_pass avg_fail : ℕ) (weight_pass weight_fail : ℕ) 
  (h_total_students : T = 150)
  (h_avg_all : avg_all = 40)
  (h_avg_pass : avg_pass = 45)
  (h_avg_fail : avg_fail = 20)
  (h_weight_ratio : weight_pass = 3 ∧ weight_fail = 1)
  (h_total_marks : (weight_pass * avg_pass * P + weight_fail * avg_fail * F) / (weight_pass * P + weight_fail * F) = avg_all)
  (h_students_sum : P + F = T) :
  P = 85 :=
by
  sorry

end NUMINAMATH_GPT_num_pass_students_is_85_l1407_140734


namespace NUMINAMATH_GPT_n_divisible_by_100_l1407_140707

theorem n_divisible_by_100 
    (n : ℕ) 
    (h_pos : 0 < n) 
    (h_div : 100 ∣ n^3) : 
    100 ∣ n := 
sorry

end NUMINAMATH_GPT_n_divisible_by_100_l1407_140707


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l1407_140772

theorem solve_quadratic_inequality (x : ℝ) (h : x^2 - 7 * x + 6 < 0) : 1 < x ∧ x < 6 :=
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l1407_140772


namespace NUMINAMATH_GPT_large_block_dimension_ratio_l1407_140733

theorem large_block_dimension_ratio
  (V_normal V_large : ℝ) 
  (k : ℝ)
  (h1 : V_normal = 4)
  (h2 : V_large = 32) 
  (h3 : V_large = k^3 * V_normal) :
  k = 2 := by
  sorry

end NUMINAMATH_GPT_large_block_dimension_ratio_l1407_140733


namespace NUMINAMATH_GPT_minimum_cuts_for_polygons_l1407_140751

theorem minimum_cuts_for_polygons (initial_pieces desired_pieces : ℕ) (sides : ℕ)
    (h_initial_pieces : initial_pieces = 1) (h_desired_pieces : desired_pieces = 100)
    (h_sides : sides = 20) :
    ∃ (cuts : ℕ), cuts = 1699 ∧
    (∀ current_pieces, current_pieces < desired_pieces → current_pieces + cuts ≥ desired_pieces) :=
by
    sorry

end NUMINAMATH_GPT_minimum_cuts_for_polygons_l1407_140751


namespace NUMINAMATH_GPT_train_speed_from_clicks_l1407_140726

theorem train_speed_from_clicks (speed_mph : ℝ) (rail_length_ft : ℝ) (clicks_heard : ℝ) :
  rail_length_ft = 40 →
  clicks_heard = 1 →
  (60 * rail_length_ft * clicks_heard * speed_mph / 5280) = 27 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_train_speed_from_clicks_l1407_140726


namespace NUMINAMATH_GPT_baker_initial_cakes_cannot_be_determined_l1407_140775

theorem baker_initial_cakes_cannot_be_determined (initial_pastries sold_cakes sold_pastries remaining_pastries : ℕ)
  (h1 : initial_pastries = 148)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : sold_pastries + remaining_pastries = initial_pastries) :
  True :=
by
  sorry

end NUMINAMATH_GPT_baker_initial_cakes_cannot_be_determined_l1407_140775


namespace NUMINAMATH_GPT_min_value_of_fraction_l1407_140742

noncomputable def problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  problem_statement a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_fraction_l1407_140742


namespace NUMINAMATH_GPT_factorization_a4_plus_4_l1407_140731

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 - 2*a + 2) * (a^2 + 2*a + 2) :=
by sorry

end NUMINAMATH_GPT_factorization_a4_plus_4_l1407_140731


namespace NUMINAMATH_GPT_cubes_and_quartics_sum_l1407_140724

theorem cubes_and_quartics_sum (a b : ℝ) (h1 : a + b = 2) (h2 : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_cubes_and_quartics_sum_l1407_140724


namespace NUMINAMATH_GPT_smallest_N_proof_l1407_140704

theorem smallest_N_proof (N c1 c2 c3 c4 : ℕ)
  (h1 : N + c1 = 4 * c3 - 2)
  (h2 : N + c2 = 4 * c1 - 3)
  (h3 : 2 * N + c3 = 4 * c4 - 1)
  (h4 : 3 * N + c4 = 4 * c2) :
  N = 12 :=
sorry

end NUMINAMATH_GPT_smallest_N_proof_l1407_140704


namespace NUMINAMATH_GPT_num_red_balls_l1407_140745

theorem num_red_balls (x : ℕ) (h : 4 / (4 + x) = 1 / 5) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_num_red_balls_l1407_140745


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_l1407_140797

theorem arithmetic_sequence_a4 (a : ℕ → ℤ) (a2 a4 a3 : ℤ) (S5 : ℤ)
  (h₁ : S5 = 25)
  (h₂ : a 2 = 3)
  (h₃ : S5 = a 1 + a 2 + a 3 + a 4 + a 5)
  (h₄ : a 3 = (a 1 + a 5) / 2)
  (h₅ : ∀ n : ℕ, (a (n+1) - a n) = (a 2 - a 1)) :
  a 4 = 7 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_l1407_140797


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1407_140736

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) (h1 : area = 600) (h2 : d1 = 30) :
  d2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1407_140736


namespace NUMINAMATH_GPT_mary_turnips_grown_l1407_140727

variable (sally_turnips : ℕ)
variable (total_turnips : ℕ)
variable (mary_turnips : ℕ)

theorem mary_turnips_grown (h_sally : sally_turnips = 113)
                          (h_total : total_turnips = 242) :
                          mary_turnips = total_turnips - sally_turnips := by
  sorry

end NUMINAMATH_GPT_mary_turnips_grown_l1407_140727


namespace NUMINAMATH_GPT_det_B_eq_2_l1407_140791

theorem det_B_eq_2 {x y : ℝ}
  (hB : ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), B = ![![x, 2], ![-3, y]])
  (h_eqn : ∃ (B_inv : Matrix (Fin 2) (Fin 2) ℝ),
    B_inv = (1 / (x * y + 6)) • ![![y, -2], ![3, x]] ∧
    ![![x, 2], ![-3, y]] + 2 • B_inv = 0) : 
  Matrix.det ![![x, 2], ![-3, y]] = 2 :=
by
  sorry

end NUMINAMATH_GPT_det_B_eq_2_l1407_140791


namespace NUMINAMATH_GPT_count_valid_prime_pairs_l1407_140715

theorem count_valid_prime_pairs (x y : ℕ) (h₁ : Prime x) (h₂ : Prime y) (h₃ : x ≠ y) (h₄ : (621 * x * y) % (x + y) = 0) : 
  ∃ p, p = 6 := by
  sorry

end NUMINAMATH_GPT_count_valid_prime_pairs_l1407_140715


namespace NUMINAMATH_GPT_connectivity_within_square_l1407_140764

theorem connectivity_within_square (side_length : ℝ) (highway1 highway2 : ℝ) 
  (A1 A2 A3 A4 : ℝ → ℝ → Prop) : 
  side_length = 10 → 
  highway1 ≠ highway2 → 
  (∀ x y, (0 ≤ x ∧ x ≤ side_length ∧ 0 ≤ y ∧ y ≤ side_length) → 
    (A1 x y ∨ A2 x y ∨ A3 x y ∨ A4 x y)) →
  ∃ (road_length : ℝ), road_length ≤ 25 := 
sorry

end NUMINAMATH_GPT_connectivity_within_square_l1407_140764


namespace NUMINAMATH_GPT_correct_statement_four_l1407_140749

variable {α : Type*} (A B S : Set α) (U : Set α)

theorem correct_statement_four (h1 : U = Set.univ) (h2 : A ∩ B = U) : A = U ∧ B = U := by
  sorry

end NUMINAMATH_GPT_correct_statement_four_l1407_140749


namespace NUMINAMATH_GPT_count_valid_age_pairs_l1407_140784

theorem count_valid_age_pairs :
  ∃ (d n : ℕ) (a b : ℕ), 10 * a + b ≥ 30 ∧
                       10 * b + a ≥ 35 ∧
                       b > a ∧
                       ∃ k : ℕ, k = 10 := 
sorry

end NUMINAMATH_GPT_count_valid_age_pairs_l1407_140784


namespace NUMINAMATH_GPT_max_weight_of_crates_on_trip_l1407_140760

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 150

theorem max_weight_of_crates_on_trip : max_crates * min_crate_weight = 750 := by
  sorry

end NUMINAMATH_GPT_max_weight_of_crates_on_trip_l1407_140760


namespace NUMINAMATH_GPT_nonneg_int_solution_coprime_l1407_140735

theorem nonneg_int_solution_coprime (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : c ≥ (a - 1) * (b - 1)) :
  ∃ (x y : ℕ), c = a * x + b * y :=
sorry

end NUMINAMATH_GPT_nonneg_int_solution_coprime_l1407_140735


namespace NUMINAMATH_GPT_problem_1_problem_2_l1407_140721

noncomputable def complete_residue_system (n : ℕ) (as : Fin n → ℕ) :=
  ∀ i j : Fin n, i ≠ j → as i % n ≠ as j % n

theorem problem_1 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) := 
sorry

theorem problem_2 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) ∧ complete_residue_system n (λ i => as i - i) := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1407_140721


namespace NUMINAMATH_GPT_first_discount_percentage_l1407_140753

/-
  Prove that under the given conditions:
  1. The price before the first discount is $33.78.
  2. The final price after the first and second discounts is $19.
  3. The second discount is 25%.
-/
theorem first_discount_percentage (x : ℝ) :
  (33.78 * (1 - x / 100) * (1 - 25 / 100) = 19) →
  x = 25 :=
by
  -- Proof steps (to be filled)
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l1407_140753


namespace NUMINAMATH_GPT_remainder_101_pow_50_mod_100_l1407_140771

theorem remainder_101_pow_50_mod_100 : (101 ^ 50) % 100 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_101_pow_50_mod_100_l1407_140771


namespace NUMINAMATH_GPT_total_combined_grapes_l1407_140723

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end NUMINAMATH_GPT_total_combined_grapes_l1407_140723


namespace NUMINAMATH_GPT_min_students_l1407_140780

theorem min_students (b g : ℕ) (hb : 1 ≤ b) (hg : 1 ≤ g)
    (h1 : b = (4/3) * g) 
    (h2 : (1/2) * b = 2 * ((1/3) * g)) 
    : b + g = 7 :=
by sorry

end NUMINAMATH_GPT_min_students_l1407_140780


namespace NUMINAMATH_GPT_no_integer_points_on_circle_l1407_140750

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, ¬ ((x - 3)^2 + (x + 1 + 2)^2 ≤ 64) := by
  sorry

end NUMINAMATH_GPT_no_integer_points_on_circle_l1407_140750


namespace NUMINAMATH_GPT_divisible_by_133_l1407_140719

theorem divisible_by_133 (n : ℕ) : (11^(n + 2) + 12^(2*n + 1)) % 133 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_133_l1407_140719


namespace NUMINAMATH_GPT_proof_q_values_proof_q_comparison_l1407_140759

-- Definitions of the conditions given.
def q : ℝ → ℝ := 
  sorry -- The definition is not required to be constructed, as we are only focusing on the conditions given.

-- Conditions
axiom cond1 : q 2 = 5
axiom cond2 : q 1.5 = 3

-- Statements to prove
theorem proof_q_values : (q 2 = 5) ∧ (q 1.5 = 3) := 
  by sorry

theorem proof_q_comparison : q 2 > q 1.5 :=
  by sorry

end NUMINAMATH_GPT_proof_q_values_proof_q_comparison_l1407_140759


namespace NUMINAMATH_GPT_total_boys_in_camp_l1407_140787

theorem total_boys_in_camp (T : ℝ) 
  (h1 : 0.20 * T = number_of_boys_from_school_A)
  (h2 : 0.30 * number_of_boys_from_school_A = number_of_boys_study_science_from_school_A)
  (h3 : number_of_boys_from_school_A - number_of_boys_study_science_from_school_A = 42) :
  T = 300 := 
sorry

end NUMINAMATH_GPT_total_boys_in_camp_l1407_140787


namespace NUMINAMATH_GPT_right_isosceles_triangle_areas_l1407_140716

theorem right_isosceles_triangle_areas :
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  A + B = C :=
by
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  sorry

end NUMINAMATH_GPT_right_isosceles_triangle_areas_l1407_140716


namespace NUMINAMATH_GPT_Mary_sleep_hours_for_avg_score_l1407_140700

def sleep_score_inverse_relation (sleep1 score1 sleep2 score2 : ℝ) : Prop :=
  sleep1 * score1 = sleep2 * score2

theorem Mary_sleep_hours_for_avg_score (h1 s1 s2 : ℝ) (h_eq : h1 = 6) (s1_eq : s1 = 60)
  (avg_score_cond : (s1 + s2) / 2 = 75) :
  ∃ h2 : ℝ, sleep_score_inverse_relation h1 s1 h2 s2 ∧ h2 = 4 := 
by
  sorry

end NUMINAMATH_GPT_Mary_sleep_hours_for_avg_score_l1407_140700


namespace NUMINAMATH_GPT_julia_total_watches_l1407_140701

-- Definitions based on conditions.
def silver_watches : Nat := 20
def bronze_watches : Nat := 3 * silver_watches
def total_silver_bronze_watches : Nat := silver_watches + bronze_watches
def gold_watches : Nat := total_silver_bronze_watches / 10

-- The final proof statement without providing the proof.
theorem julia_total_watches : (silver_watches + bronze_watches + gold_watches) = 88 :=
by 
  -- Since we don't need to provide the actual proof, we use sorry
  sorry

end NUMINAMATH_GPT_julia_total_watches_l1407_140701


namespace NUMINAMATH_GPT_correct_proposition_is_D_l1407_140755

theorem correct_proposition_is_D (A B C D : Prop) :
  (∀ (H : Prop), (H = A ∨ H = B ∨ H = C) → ¬H) → D :=
by
  -- We assume that A, B, and C are false.
  intro h
  -- Now we need to prove that D is true.
  sorry

end NUMINAMATH_GPT_correct_proposition_is_D_l1407_140755


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1407_140798

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes_intersect : ∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 = -1 ∧ B.1 = -1 ∧
    ∀ (A B : ℝ × ℝ), ∃ x y : ℝ, (A.2 = y ∧ B.2 = y ∧ x^2 / a^2 - y^2 / b^2 = 1))
  (triangle_area : ∃ A B : ℝ × ℝ, 1 / 2 * abs (A.1 * B.2 - A.2 * B.1) = 2 * Real.sqrt 3) :
  ∃ e : ℝ, e = Real.sqrt 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_eccentricity_l1407_140798


namespace NUMINAMATH_GPT_blue_notebook_cost_l1407_140783

theorem blue_notebook_cost
    (total_spent : ℕ)
    (total_notebooks : ℕ)
    (red_notebooks : ℕ) (red_cost : ℕ)
    (green_notebooks : ℕ) (green_cost : ℕ)
    (blue_notebooks : ℕ) (blue_total_cost : ℕ) 
    (blue_cost : ℕ) :
    total_spent = 37 →
    total_notebooks = 12 →
    red_notebooks = 3 →
    red_cost = 4 →
    green_notebooks = 2 →
    green_cost = 2 →
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks →
    blue_total_cost = total_spent - red_notebooks * red_cost - green_notebooks * green_cost →
    blue_cost = blue_total_cost / blue_notebooks →
    blue_cost = 3 := 
    by sorry

end NUMINAMATH_GPT_blue_notebook_cost_l1407_140783


namespace NUMINAMATH_GPT_units_digit_17_pow_2023_l1407_140748

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_17_pow_2023_l1407_140748


namespace NUMINAMATH_GPT_rectangle_perimeter_l1407_140713

-- Defining the given conditions
def rectangleArea := 4032
noncomputable def ellipseArea := 4032 * Real.pi
noncomputable def b := Real.sqrt 2016
noncomputable def a := 2 * Real.sqrt 2016

-- Problem statement: the perimeter of the rectangle
theorem rectangle_perimeter (x y : ℝ) (h1 : x * y = rectangleArea)
  (h2 : x + y = 2 * a) : 2 * (x + y) = 8 * Real.sqrt 2016 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1407_140713


namespace NUMINAMATH_GPT_solve_congruence_l1407_140737

theorem solve_congruence :
  ∃ n : ℤ, 19 * n ≡ 13 [ZMOD 47] ∧ n ≡ 25 [ZMOD 47] :=
by
  sorry

end NUMINAMATH_GPT_solve_congruence_l1407_140737


namespace NUMINAMATH_GPT_find_fraction_of_original_flow_rate_l1407_140767

noncomputable def fraction_of_original_flow_rate (f : ℚ) : Prop :=
  let original_flow_rate := 5
  let reduced_flow_rate := 2
  reduced_flow_rate = f * original_flow_rate - 1

theorem find_fraction_of_original_flow_rate : ∃ (f : ℚ), fraction_of_original_flow_rate f ∧ f = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_of_original_flow_rate_l1407_140767


namespace NUMINAMATH_GPT_value_of_expr_l1407_140756

noncomputable def verify_inequality (x a b c : ℝ) : Prop :=
  (x - a) * (x - b) / (x - c) ≥ 0

theorem value_of_expr (a b c : ℝ) :
  (∀ x : ℝ, verify_inequality x a b c ↔ (x < -6 ∨ abs (x - 30) ≤ 2)) →
  a < b →
  a = 28 →
  b = 32 →
  c = -6 →
  a + 2 * b + 3 * c = 74 := by
  sorry

end NUMINAMATH_GPT_value_of_expr_l1407_140756


namespace NUMINAMATH_GPT_roots_cubic_polynomial_l1407_140705

theorem roots_cubic_polynomial (r s t : ℝ)
  (h₁ : 8 * r^3 + 1001 * r + 2008 = 0)
  (h₂ : 8 * s^3 + 1001 * s + 2008 = 0)
  (h₃ : 8 * t^3 + 1001 * t + 2008 = 0)
  (h₄ : r + s + t = 0) :
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := 
sorry

end NUMINAMATH_GPT_roots_cubic_polynomial_l1407_140705


namespace NUMINAMATH_GPT_increase_in_sides_of_polygon_l1407_140710

theorem increase_in_sides_of_polygon (n n' : ℕ) (h : (n' - 2) * 180 - (n - 2) * 180 = 180) : n' = n + 1 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_sides_of_polygon_l1407_140710


namespace NUMINAMATH_GPT_polynomial_roots_l1407_140738

theorem polynomial_roots :
  ∀ x : ℝ, (4 * x^4 - 28 * x^3 + 53 * x^2 - 28 * x + 4 = 0) ↔ (x = 4 ∨ x = 2 ∨ x = 1/4 ∨ x = 1/2) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l1407_140738


namespace NUMINAMATH_GPT_exist_infinitely_many_coprime_pairs_l1407_140776

theorem exist_infinitely_many_coprime_pairs (a b : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : Nat.gcd a b = 1) : 
  ∃ (a b : ℕ), (a + b).mod (a^b + b^a) = 0 :=
sorry

end NUMINAMATH_GPT_exist_infinitely_many_coprime_pairs_l1407_140776


namespace NUMINAMATH_GPT_average_age_of_contestants_l1407_140768

theorem average_age_of_contestants :
  let numFemales := 12
  let avgAgeFemales := 25
  let numMales := 18
  let avgAgeMales := 40
  let sumAgesFemales := avgAgeFemales * numFemales
  let sumAgesMales := avgAgeMales * numMales
  let totalSumAges := sumAgesFemales + sumAgesMales
  let totalContestants := numFemales + numMales
  (totalSumAges / totalContestants) = 34 := by
  sorry

end NUMINAMATH_GPT_average_age_of_contestants_l1407_140768


namespace NUMINAMATH_GPT_largest_multiple_of_7_less_than_100_l1407_140793

theorem largest_multiple_of_7_less_than_100 : ∃ (n : ℕ), n * 7 < 100 ∧ ∀ (m : ℕ), m * 7 < 100 → m * 7 ≤ n * 7 :=
  by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_less_than_100_l1407_140793


namespace NUMINAMATH_GPT_product_d_e_l1407_140788

-- Define the problem: roots of the polynomial x^2 + x - 2
def roots_of_quadratic : Prop :=
  ∃ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0)

-- Define the condition that both roots are also roots of another polynomial
def roots_of_higher_poly (α β : ℚ) : Prop :=
  (α^7 - 7 * α^3 - 10 = 0 ) ∧ (β^7 - 7 * β^3 - 10 = 0)

-- The final proposition to prove
theorem product_d_e :
  ∀ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0) → (α^7 - 7 * α^3 - 10 = 0) ∧ (β^7 - 7 * β^3 - 10 = 0) → 7 * 10 = 70 := 
by sorry

end NUMINAMATH_GPT_product_d_e_l1407_140788


namespace NUMINAMATH_GPT_water_current_speed_l1407_140766

-- Definitions based on the conditions
def swimmer_speed : ℝ := 4  -- The swimmer's speed in still water (km/h)
def swim_time : ℝ := 2  -- Time taken to swim against the current (hours)
def swim_distance : ℝ := 6  -- Distance swum against the current (km)

-- The effective speed against the current
noncomputable def effective_speed_against_current (v : ℝ) : ℝ := swimmer_speed - v

-- Lean statement that formalizes proving the speed of the current
theorem water_current_speed (v : ℝ) (h : effective_speed_against_current v = swim_distance / swim_time) : v = 1 :=
by
  sorry

end NUMINAMATH_GPT_water_current_speed_l1407_140766


namespace NUMINAMATH_GPT_sara_has_8_balloons_l1407_140709

-- Define the number of yellow balloons Tom has.
def tom_balloons : ℕ := 9 

-- Define the total number of yellow balloons.
def total_balloons : ℕ := 17

-- Define the number of yellow balloons Sara has.
def sara_balloons : ℕ := total_balloons - tom_balloons

-- Theorem stating that Sara has 8 yellow balloons.
theorem sara_has_8_balloons : sara_balloons = 8 := by
  -- Proof goes here. Adding sorry for now to skip the proof.
  sorry

end NUMINAMATH_GPT_sara_has_8_balloons_l1407_140709


namespace NUMINAMATH_GPT_pizza_slices_l1407_140789

theorem pizza_slices (P T S : ℕ) (h1 : P = 2) (h2 : T = 16) : S = 8 :=
by
  -- to be filled in
  sorry

end NUMINAMATH_GPT_pizza_slices_l1407_140789


namespace NUMINAMATH_GPT_three_distinct_roots_condition_l1407_140761

noncomputable def k_condition (k : ℝ) : Prop :=
  ∀ (x : ℝ), (x / (x - 1) + x / (x - 3)) = k * x → 
    (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

theorem three_distinct_roots_condition (k : ℝ) : k ≠ 0 ↔ k_condition k :=
by
  sorry

end NUMINAMATH_GPT_three_distinct_roots_condition_l1407_140761


namespace NUMINAMATH_GPT_wendys_sales_are_205_l1407_140717

def price_of_apple : ℝ := 1.5
def price_of_orange : ℝ := 1.0
def apples_sold_morning : ℕ := 40
def oranges_sold_morning : ℕ := 30
def apples_sold_afternoon : ℕ := 50
def oranges_sold_afternoon : ℕ := 40

/-- Wendy's total sales for the day are $205 given the conditions about the prices of apples and oranges,
and the number of each sold in the morning and afternoon. -/
def wendys_total_sales : ℝ :=
  let total_apples_sold := apples_sold_morning + apples_sold_afternoon
  let total_oranges_sold := oranges_sold_morning + oranges_sold_afternoon
  let sales_from_apples := total_apples_sold * price_of_apple
  let sales_from_oranges := total_oranges_sold * price_of_orange
  sales_from_apples + sales_from_oranges

theorem wendys_sales_are_205 : wendys_total_sales = 205 := by
  sorry

end NUMINAMATH_GPT_wendys_sales_are_205_l1407_140717


namespace NUMINAMATH_GPT_sequence_property_l1407_140769

-- Conditions as definitions
def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 = -(2 / 3)) ∧ (∀ n ≥ 2, S n + (1 / S n) + 2 = a n)

-- The desired property of the sequence
def S_property (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = -((n + 1) / (n + 2))

-- The main theorem
theorem sequence_property (a S : ℕ → ℝ) (h_seq : seq a S) : S_property S := sorry

end NUMINAMATH_GPT_sequence_property_l1407_140769


namespace NUMINAMATH_GPT_find_science_books_l1407_140795

theorem find_science_books
  (S : ℕ)
  (h1 : 2 * 3 + 3 * 2 + 3 * S = 30) :
  S = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_science_books_l1407_140795


namespace NUMINAMATH_GPT_sum_of_numbers_l1407_140722

theorem sum_of_numbers : ∃ (a b : ℕ), (a + b = 21) ∧ (a / b = 3 / 4) ∧ (max a b = 12) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1407_140722


namespace NUMINAMATH_GPT_total_profit_at_100_max_profit_price_l1407_140746

noncomputable def sales_volume (x : ℝ) : ℝ := 15 - 0.1 * x
noncomputable def floating_price (S : ℝ) : ℝ := 10 / S
noncomputable def supply_price (x : ℝ) : ℝ := 30 + floating_price (sales_volume x)
noncomputable def profit_per_set (x : ℝ) : ℝ := x - supply_price x
noncomputable def total_profit (x : ℝ) : ℝ := profit_per_set x * sales_volume x

-- Theorem 1: Total profit when each set is priced at 100 yuan is 340 ten thousand yuan
theorem total_profit_at_100 : total_profit 100 = 340 := by
  sorry

-- Theorem 2: The price per set that maximizes profit per set is 140 yuan
theorem max_profit_price : ∃ x, profit_per_set x = 100 ∧ x = 140 := by
  sorry

end NUMINAMATH_GPT_total_profit_at_100_max_profit_price_l1407_140746


namespace NUMINAMATH_GPT_commute_times_absolute_difference_l1407_140781

theorem commute_times_absolute_difference
  (x y : ℝ)
  (H_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (H_var : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  abs (x - y) = 4 :=
by
  -- proof steps are omitted
  sorry

end NUMINAMATH_GPT_commute_times_absolute_difference_l1407_140781


namespace NUMINAMATH_GPT_number_of_bicycles_l1407_140779

theorem number_of_bicycles (B T : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 := by
  sorry

end NUMINAMATH_GPT_number_of_bicycles_l1407_140779


namespace NUMINAMATH_GPT_number_of_paths_l1407_140743

open Nat

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| x, 0 => 1
| 0, y => 1
| (x + 1), (y + 1) => f x (y + 1) + f (x + 1) y

theorem number_of_paths (n : ℕ) : f n 2 = (n^2 + 3 * n + 2) / 2 := by sorry

end NUMINAMATH_GPT_number_of_paths_l1407_140743


namespace NUMINAMATH_GPT_range_of_a_l1407_140754

variable {a x : ℝ}

theorem range_of_a (h : ∀ x, (a - 5) * x > a - 5 ↔ x < 1) : a < 5 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1407_140754


namespace NUMINAMATH_GPT_nonneg_or_nonpos_l1407_140790

theorem nonneg_or_nonpos (n : ℕ) (h : n ≥ 2) (c : Fin n → ℝ)
  (h_eq : (n - 1) * (Finset.univ.sum (fun i => c i ^ 2)) = (Finset.univ.sum c) ^ 2) :
  (∀ i, c i ≥ 0) ∨ (∀ i, c i ≤ 0) := 
  sorry

end NUMINAMATH_GPT_nonneg_or_nonpos_l1407_140790


namespace NUMINAMATH_GPT_minimum_time_to_cook_l1407_140765

def wash_pot_fill_water : ℕ := 2
def wash_vegetables : ℕ := 3
def prepare_noodles_seasonings : ℕ := 2
def boil_water : ℕ := 7
def cook_noodles_vegetables : ℕ := 3

theorem minimum_time_to_cook : wash_pot_fill_water + boil_water + cook_noodles_vegetables = 12 :=
by
  sorry

end NUMINAMATH_GPT_minimum_time_to_cook_l1407_140765


namespace NUMINAMATH_GPT_temperature_decrease_l1407_140729

theorem temperature_decrease (current_temp : ℝ) (future_temp : ℝ) : 
  current_temp = 84 → future_temp = (3 / 4) * current_temp → (current_temp - future_temp) = 21 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_temperature_decrease_l1407_140729


namespace NUMINAMATH_GPT_functional_eq_solution_l1407_140792

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (x) ^ 2 + f (y)) = x * f (x) + y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := sorry

end NUMINAMATH_GPT_functional_eq_solution_l1407_140792


namespace NUMINAMATH_GPT_tan_half_sum_l1407_140786

variable (p q : ℝ)

-- Given conditions
def cos_condition : Prop := (Real.cos p + Real.cos q = 1 / 3)
def sin_condition : Prop := (Real.sin p + Real.sin q = 4 / 9)

-- Prove the target expression
theorem tan_half_sum (h1 : cos_condition p q) (h2 : sin_condition p q) : 
  Real.tan ((p + q) / 2) = 4 / 3 :=
sorry

-- For better readability, I included variable declarations and definitions separately

end NUMINAMATH_GPT_tan_half_sum_l1407_140786


namespace NUMINAMATH_GPT_Micah_words_per_minute_l1407_140706

-- Defining the conditions
def Isaiah_words_per_minute : ℕ := 40
def extra_words : ℕ := 1200

-- Proving the statement that Micah can type 20 words per minute
theorem Micah_words_per_minute (Isaiah_wpm : ℕ) (extra_w : ℕ) : Isaiah_wpm = 40 → extra_w = 1200 → (Isaiah_wpm * 60 - extra_w) / 60 = 20 :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_Micah_words_per_minute_l1407_140706
