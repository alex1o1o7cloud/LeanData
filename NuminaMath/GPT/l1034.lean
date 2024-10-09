import Mathlib

namespace smallest_sum_l1034_103468

theorem smallest_sum (a b c : ℕ) (h : (13 * a + 11 * b + 7 * c = 1001)) :
    a / 77 + b / 91 + c / 143 = 1 → a + b + c = 79 :=
by
  sorry

end smallest_sum_l1034_103468


namespace base8_to_base10_conversion_l1034_103402

theorem base8_to_base10_conversion : 
  (6 * 8^3 + 3 * 8^2 + 7 * 8^1 + 5 * 8^0) = 3325 := 
by 
  sorry

end base8_to_base10_conversion_l1034_103402


namespace find_second_offset_l1034_103436

-- Define the given constants
def diagonal : ℝ := 30
def offset1 : ℝ := 10
def area : ℝ := 240

-- The theorem we want to prove
theorem find_second_offset : ∃ (offset2 : ℝ), area = (1 / 2) * diagonal * (offset1 + offset2) ∧ offset2 = 6 :=
sorry

end find_second_offset_l1034_103436


namespace morgan_change_l1034_103471

theorem morgan_change:
  let hamburger := 5.75
  let onion_rings := 2.50
  let smoothie := 3.25
  let side_salad := 3.75
  let cake := 4.20
  let total_cost := hamburger + onion_rings + smoothie + side_salad + cake
  let payment := 50
  let change := payment - total_cost
  ℝ := by
    exact sorry

end morgan_change_l1034_103471


namespace Sarah_l1034_103415

variable (s g : ℕ)

theorem Sarah's_score_130 (h1 : s = g + 50) (h2 : (s + g) / 2 = 105) : s = 130 :=
by
  sorry

end Sarah_l1034_103415


namespace quadratic_completion_l1034_103431

theorem quadratic_completion (b c : ℝ) (h : (x : ℝ) → x^2 + 1600 * x + 1607 = (x + b)^2 + c) (hb : b = 800) (hc : c = -638393) : 
  c / b = -797.99125 := by
  sorry

end quadratic_completion_l1034_103431


namespace problem_statement_l1034_103403

-- Define operations "※" and "#"
def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

-- Define the proof statement
theorem problem_statement : hash 4 (star (star 6 8) (hash 3 5)) = 103 := by
  sorry

end problem_statement_l1034_103403


namespace parabolic_arch_height_l1034_103423

noncomputable def arch_height (a : ℝ) : ℝ :=
  a * (0 : ℝ)^2

theorem parabolic_arch_height :
  ∃ (a : ℝ), (∫ x in (-4 : ℝ)..4, a * x^2) = (160 : ℝ) ∧ arch_height a = 30 :=
by
  sorry

end parabolic_arch_height_l1034_103423


namespace chord_length_sqrt_10_l1034_103426

/-
  Given a line L: 3x - y - 6 = 0 and a circle C: x^2 + y^2 - 2x - 4y = 0,
  prove that the length of the chord AB formed by their intersection is sqrt(10).
-/

noncomputable def line_L : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ 3 * x - y - 6 = 0}

noncomputable def circle_C : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 + y^2 - 2 * x - 4 * y = 0}

noncomputable def chord_length (L C : Set (ℝ × ℝ)) : ℝ :=
  let center := (1, 2)
  let r := Real.sqrt 5
  let d := |3 * 1 - 2 - 6| / Real.sqrt (1 + 3^2)
  2 * Real.sqrt (r^2 - d^2)

theorem chord_length_sqrt_10 : chord_length line_L circle_C = Real.sqrt 10 := sorry

end chord_length_sqrt_10_l1034_103426


namespace pushups_total_l1034_103474

theorem pushups_total (x melanie david karen john : ℕ) 
  (hx : x = 51)
  (h_melanie : melanie = 2 * x - 7)
  (h_david : david = x + 22)
  (h_avg : (x + melanie + david) / 3 = (x + (2 * x - 7) + (x + 22)) / 3)
  (h_karen : karen = (x + (2 * x - 7) + (x + 22)) / 3 - 5)
  (h_john : john = (x + 22) - 4) :
  john + melanie + karen = 232 := by
  sorry

end pushups_total_l1034_103474


namespace find_x_of_series_eq_16_l1034_103464

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x ^ n

theorem find_x_of_series_eq_16 (x : ℝ) (h : series_sum x = 16) : x = (4 - Real.sqrt 2) / 4 :=
by
  sorry

end find_x_of_series_eq_16_l1034_103464


namespace number_of_even_factors_of_n_l1034_103457

noncomputable def n := 2^3 * 3^2 * 7^3

theorem number_of_even_factors_of_n : 
  (∃ (a : ℕ), (1 ≤ a ∧ a ≤ 3)) ∧ 
  (∃ (b : ℕ), (0 ≤ b ∧ b ≤ 2)) ∧ 
  (∃ (c : ℕ), (0 ≤ c ∧ c ≤ 3)) → 
  (even_nat_factors_count : ℕ) = 36 :=
by
  sorry

end number_of_even_factors_of_n_l1034_103457


namespace local_value_proof_l1034_103417

-- Definitions based on the conditions
def face_value_7 : ℕ := 7
def local_value_6_in_7098060 : ℕ := 6000
def product_of_face_value_and_local_value : ℕ := face_value_7 * local_value_6_in_7098060
def local_value_6_in_product : ℕ := 6000

-- Theorem statement
theorem local_value_proof : local_value_6_in_product = 6000 :=
by
  -- Direct restatement of the condition in Lean
  sorry

end local_value_proof_l1034_103417


namespace tan_alpha_value_l1034_103447

open Real

theorem tan_alpha_value (α : ℝ) (h1 : sin α + cos α = -1 / 2) (h2 : 0 < α ∧ α < π) : tan α = -1 / 3 :=
sorry

end tan_alpha_value_l1034_103447


namespace number_of_cities_from_group_B_l1034_103440

theorem number_of_cities_from_group_B
  (total_cities : ℕ)
  (cities_in_A : ℕ)
  (cities_in_B : ℕ)
  (cities_in_C : ℕ)
  (sampled_cities : ℕ)
  (h1 : total_cities = cities_in_A + cities_in_B + cities_in_C)
  (h2 : total_cities = 24)
  (h3 : cities_in_A = 4)
  (h4 : cities_in_B = 12)
  (h5 : cities_in_C = 8)
  (h6 : sampled_cities = 6) :
  cities_in_B * sampled_cities / total_cities = 3 := 
  by 
    sorry

end number_of_cities_from_group_B_l1034_103440


namespace det_B_squared_minus_3B_l1034_103409

theorem det_B_squared_minus_3B (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B = ![![2, 4], ![3, 2]]) : 
  Matrix.det (B * B - 3 • B) = 88 := by
  sorry

end det_B_squared_minus_3B_l1034_103409


namespace smallest_d_l1034_103442

theorem smallest_d (d : ℝ) : 
  (∃ d, d > 0 ∧ (4 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d - 2)^2))) → d = 2 :=
sorry

end smallest_d_l1034_103442


namespace transformation_is_rotation_l1034_103441

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

end transformation_is_rotation_l1034_103441


namespace mixture_ratio_l1034_103495

variables (p q V W : ℝ)

-- Condition summaries:
-- - First jar has volume V, ratio of alcohol to water is p:1.
-- - Second jar has volume W, ratio of alcohol to water is q:2.

theorem mixture_ratio (hp : p > 0) (hq : q > 0) (hV : V > 0) (hW : W > 0) : 
  (p * V * (p + 2) + q * W * (p + 1)) / ((p + 1) * (q + 2) * (V + 2 * W)) =
  (p * V) / (p + 1) + (q * W) / (q + 2) :=
sorry

end mixture_ratio_l1034_103495


namespace quadratic_complete_square_l1034_103404

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 + 2 * x + 3) = ((x + 1)^2 + 2) :=
by
  intro x
  sorry

end quadratic_complete_square_l1034_103404


namespace greg_total_earnings_correct_l1034_103488

def charge_per_dog := 20
def charge_per_minute := 1

def earnings_one_dog := charge_per_dog + charge_per_minute * 10
def earnings_two_dogs := 2 * (charge_per_dog + charge_per_minute * 7)
def earnings_three_dogs := 3 * (charge_per_dog + charge_per_minute * 9)

def total_earnings := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

theorem greg_total_earnings_correct : total_earnings = 171 := by
  sorry

end greg_total_earnings_correct_l1034_103488


namespace greatest_possible_value_of_q_minus_r_l1034_103490

theorem greatest_possible_value_of_q_minus_r :
  ∃ q r : ℕ, 0 < q ∧ 0 < r ∧ 852 = 21 * q + r ∧ q - r = 28 :=
by
  -- Proof goes here
  sorry

end greatest_possible_value_of_q_minus_r_l1034_103490


namespace total_trees_after_planting_l1034_103459

theorem total_trees_after_planting
  (initial_walnut_trees : ℕ) (initial_oak_trees : ℕ) (initial_maple_trees : ℕ)
  (plant_walnut_trees : ℕ) (plant_oak_trees : ℕ) (plant_maple_trees : ℕ) :
  (initial_walnut_trees = 107) →
  (initial_oak_trees = 65) →
  (initial_maple_trees = 32) →
  (plant_walnut_trees = 104) →
  (plant_oak_trees = 79) →
  (plant_maple_trees = 46) →
  initial_walnut_trees + plant_walnut_trees +
  initial_oak_trees + plant_oak_trees +
  initial_maple_trees + plant_maple_trees = 433 :=
by
  intros
  sorry

end total_trees_after_planting_l1034_103459


namespace solve_inner_circle_radius_l1034_103425

noncomputable def isosceles_trapezoid_radius := 
  let AB := 8
  let BC := 7
  let DA := 7
  let CD := 6
  let radiusA := 4
  let radiusB := 4
  let radiusC := 3
  let radiusD := 3
  let r := (-72 + 60 * Real.sqrt 3) / 26
  r

theorem solve_inner_circle_radius :
  let k := 72
  let m := 60
  let n := 3
  let p := 26
  gcd k p = 1 → -- explicit gcd calculation between k and p 
  (isosceles_trapezoid_radius = (-k + m * Real.sqrt n) / p) ∧ (k + m + n + p = 161) :=
by
  sorry

end solve_inner_circle_radius_l1034_103425


namespace membership_percentage_change_l1034_103413

-- Definitions required based on conditions
def membersFallChange (initialMembers : ℝ) : ℝ := initialMembers * 1.07
def membersSpringChange (fallMembers : ℝ) : ℝ := fallMembers * 0.81
def membersSummerChange (springMembers : ℝ) : ℝ := springMembers * 1.15

-- Prove the total change in percentage from fall to the end of summer
theorem membership_percentage_change :
  let initialMembers := 100
  let fallMembers := membersFallChange initialMembers
  let springMembers := membersSpringChange fallMembers
  let summerMembers := membersSummerChange springMembers
  ((summerMembers - initialMembers) / initialMembers) * 100 = -0.33 := by
  sorry

end membership_percentage_change_l1034_103413


namespace exponential_function_decreasing_l1034_103493

theorem exponential_function_decreasing {a : ℝ} 
  (h : ∀ x y : ℝ, x > y → (a-1)^x < (a-1)^y) : 1 < a ∧ a < 2 :=
by sorry

end exponential_function_decreasing_l1034_103493


namespace find_a_in_triangle_l1034_103418

theorem find_a_in_triangle (b c : ℝ) (cos_B_minus_C : ℝ) (a : ℝ) 
  (hb : b = 7) (hc : c = 6) (hcos : cos_B_minus_C = 15 / 16) :
  a = 5 * Real.sqrt 3 :=
by
  sorry

end find_a_in_triangle_l1034_103418


namespace dog_food_packages_l1034_103472

theorem dog_food_packages
  (packages_cat_food : Nat := 9)
  (cans_per_package_cat_food : Nat := 10)
  (cans_per_package_dog_food : Nat := 5)
  (more_cans_cat_food : Nat := 55)
  (total_cans_cat_food : Nat := packages_cat_food * cans_per_package_cat_food)
  (total_cans_dog_food : Nat := d * cans_per_package_dog_food)
  (h : total_cans_cat_food = total_cans_dog_food + more_cans_cat_food) :
  d = 7 :=
by
  sorry

end dog_food_packages_l1034_103472


namespace gcd_40_120_80_l1034_103469

-- Given numbers
def n1 := 40
def n2 := 120
def n3 := 80

-- The problem we want to prove:
theorem gcd_40_120_80 : Int.gcd (Int.gcd n1 n2) n3 = 40 := by
  sorry

end gcd_40_120_80_l1034_103469


namespace intersecting_lines_l1034_103446

theorem intersecting_lines
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h_p1 : p1 = (3, 2))
  (h_line1 : ∀ x : ℝ, p1.2 = 3 * p1.1 + 4)
  (h_line2 : ∀ x : ℝ, p2.2 = - (1 / 3) * p2.1 + 3) :
  (∃ p : ℝ × ℝ, p = (-3 / 10, 31 / 10)) :=
by
  sorry

end intersecting_lines_l1034_103446


namespace work_together_l1034_103458

theorem work_together (A_days B_days : ℕ) (hA : A_days = 8) (hB : B_days = 4)
  (A_work : ℚ := 1 / A_days)
  (B_work : ℚ := 1 / B_days) :
  (A_work + B_work = 3 / 8) :=
by
  rw [hA, hB]
  sorry

end work_together_l1034_103458


namespace binary_div_four_remainder_l1034_103480

theorem binary_div_four_remainder (n : ℕ) (h : n = 0b111001001101) : n % 4 = 1 := 
sorry

end binary_div_four_remainder_l1034_103480


namespace ken_house_distance_condition_l1034_103492

noncomputable def ken_distance_to_dawn : ℕ := 4 -- This is the correct answer

theorem ken_house_distance_condition (K M : ℕ) (h1 : K = 2 * M) (h2 : K + M + M + K = 12) :
  K = ken_distance_to_dawn :=
  by
  sorry

end ken_house_distance_condition_l1034_103492


namespace find_non_zero_real_x_satisfies_equation_l1034_103434

theorem find_non_zero_real_x_satisfies_equation :
  ∃! x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 - (18 * x) ^ 9 = 0 ∧ x = 2 :=
by
  sorry

end find_non_zero_real_x_satisfies_equation_l1034_103434


namespace smallest_number_from_digits_l1034_103414

theorem smallest_number_from_digits : 
  ∀ (d1 d2 d3 d4 : ℕ), (d1 = 2) → (d2 = 0) → (d3 = 1) → (d4 = 6) →
  ∃ n : ℕ, (n = 1026) ∧ 
  ((n = d1 * 1000 + d2 * 100 + d3 * 10 + d4) ∨ 
   (n = d1 * 1000 + d2 * 100 + d4 * 10 + d3) ∨ 
   (n = d1 * 1000 + d3 * 100 + d2 * 10 + d4) ∨ 
   (n = d1 * 1000 + d3 * 100 + d4 * 10 + d2) ∨ 
   (n = d1 * 1000 + d4 * 100 + d2 * 10 + d3) ∨ 
   (n = d1 * 1000 + d4 * 100 + d3 * 10 + d2) ∨ 
   (n = d2 * 1000 + d1 * 100 + d3 * 10 + d4) ∨ 
   (n = d2 * 1000 + d1 * 100 + d4 * 10 + d3) ∨ 
   (n = d2 * 1000 + d3 * 100 + d1 * 10 + d4) ∨ 
   (n = d2 * 1000 + d3 * 100 + d4 * 10 + d1) ∨ 
   (n = d2 * 1000 + d4 * 100 + d1 * 10 + d3) ∨ 
   (n = d2 * 1000 + d4 * 100 + d3 * 10 + d1) ∨ 
   (n = d3 * 1000 + d1 * 100 + d2 * 10 + d4) ∨ 
   (n = d3 * 1000 + d1 * 100 + d4 * 10 + d2) ∨ 
   (n = d3 * 1000 + d2 * 100 + d1 * 10 + d4) ∨ 
   (n = d3 * 1000 + d2 * 100 + d4 * 10 + d1) ∨ 
   (n = d3 * 1000 + d4 * 100 + d1 * 10 + d2) ∨ 
   (n = d3 * 1000 + d4 * 100 + d2 * 10 + d1) ∨ 
   (n = d4 * 1000 + d1 * 100 + d2 * 10 + d3) ∨ 
   (n = d4 * 1000 + d1 * 100 + d3 * 10 + d2) ∨ 
   (n = d4 * 1000 + d2 * 100 + d1 * 10 + d3) ∨ 
   (n = d4 * 1000 + d2 * 100 + d3 * 10 + d1) ∨ 
   (n = d4 * 1000 + d3 * 100 + d1 * 10 + d2) ∨ 
   (n = d4 * 1000 + d3 * 100 + d2 * 10 + d1)) := sorry

end smallest_number_from_digits_l1034_103414


namespace ratio_of_sum_and_difference_l1034_103496

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (x + y) / (x - y) = x / y) : x / y = 1 + Real.sqrt 2 :=
sorry

end ratio_of_sum_and_difference_l1034_103496


namespace sector_area_l1034_103481

theorem sector_area (theta : ℝ) (r : ℝ) (h_theta : theta = 2 * π / 3) (h_r : r = 3) : 
    (theta / (2 * π) * π * r^2) = 3 * π :=
by 
  -- Placeholder for the actual proof
  sorry

end sector_area_l1034_103481


namespace score_difference_l1034_103491

theorem score_difference 
  (x y z w : ℝ)
  (h1 : x = 2 + (y + z + w) / 3)
  (h2 : y = (x + z + w) / 3 - 3)
  (h3 : z = 3 + (x + y + w) / 3) :
  (x + y + z) / 3 - w = 2 :=
by {
  sorry
}

end score_difference_l1034_103491


namespace ellipse_sum_l1034_103450

theorem ellipse_sum (h k a b : ℤ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 7) (b_val : b = 4) : 
  h + k + a + b = 9 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l1034_103450


namespace spatial_relationship_l1034_103407

variables {a b c : Type}          -- Lines a, b, c
variables {α β γ : Type}          -- Planes α, β, γ

-- Parallel relationship between planes
def plane_parallel (α β : Type) : Prop := sorry
-- Perpendicular relationship between planes
def plane_perpendicular (α β : Type) : Prop := sorry
-- Parallel relationship between lines and planes
def line_parallel_plane (a α : Type) : Prop := sorry
-- Perpendicular relationship between lines and planes
def line_perpendicular_plane (a α : Type) : Prop := sorry
-- Parallel relationship between lines
def line_parallel (a b : Type) : Prop := sorry
-- The angle formed by a line and a plane
def angle (a : Type) (α : Type) : Type := sorry

theorem spatial_relationship :
  (plane_parallel α γ ∧ plane_parallel β γ → plane_parallel α β) ∧
  ¬ (line_parallel_plane a α ∧ line_parallel_plane b α → line_parallel a b) ∧
  ¬ (plane_perpendicular α γ ∧ plane_perpendicular β γ → plane_parallel α β) ∧
  ¬ (line_perpendicular_plane a c ∧ line_perpendicular_plane b c → line_parallel a b) ∧
  (line_parallel a b ∧ plane_parallel α β → angle a α = angle b β) :=
sorry

end spatial_relationship_l1034_103407


namespace base_of_first_term_l1034_103444

theorem base_of_first_term (e : ℕ) (b : ℝ) (h : e = 35) :
  b^e * (1/4)^18 = 1/(2 * 10^35) → b = 1/5 :=
by
  sorry

end base_of_first_term_l1034_103444


namespace soaps_in_one_package_l1034_103455

theorem soaps_in_one_package (boxes : ℕ) (packages_per_box : ℕ) (total_packages : ℕ) (total_soaps : ℕ) : 
  boxes = 2 → packages_per_box = 6 → total_packages = boxes * packages_per_box → total_soaps = 2304 → (total_soaps / total_packages) = 192 :=
by
  intros h_boxes h_packages_per_box h_total_packages h_total_soaps
  sorry

end soaps_in_one_package_l1034_103455


namespace S6_is_48_l1034_103461

-- Define the first term and common difference
def a₁ : ℕ := 3
def d : ℕ := 2

-- Define the formula for sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) : ℕ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

-- Prove that the sum of the first 6 terms is 48
theorem S6_is_48 : sum_of_arithmetic_sequence 6 = 48 := by
  sorry

end S6_is_48_l1034_103461


namespace real_root_quadratic_complex_eq_l1034_103437

open Complex

theorem real_root_quadratic_complex_eq (a : ℝ) :
  ∀ x : ℝ, a * (1 + I) * x^2 + (1 + a^2 * I) * x + (a^2 + I) = 0 →
  a = -1 :=
by
  intros x h
  -- We need to prove this, but we're skipping the proof for now.
  sorry

end real_root_quadratic_complex_eq_l1034_103437


namespace cement_tesss_street_l1034_103430

-- Definitions of the given conditions
def cement_lexis_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Proof statement to show the amount of cement used to pave Tess's street
theorem cement_tesss_street : total_cement_used - cement_lexis_street = 5.1 :=
by 
  -- Add proof steps to show the theorem is valid.
  sorry

end cement_tesss_street_l1034_103430


namespace sum_of_eight_terms_l1034_103473

theorem sum_of_eight_terms :
  (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) = 3125000 :=
by
  sorry

end sum_of_eight_terms_l1034_103473


namespace additional_charge_per_segment_l1034_103477

theorem additional_charge_per_segment :
  ∀ (initial_fee total_charge distance : ℝ), 
    initial_fee = 2.35 →
    total_charge = 5.5 →
    distance = 3.6 →
    (total_charge - initial_fee) / (distance / (2 / 5)) = 0.35 :=
by
  intros initial_fee total_charge distance h_initial_fee h_total_charge h_distance
  sorry

end additional_charge_per_segment_l1034_103477


namespace casper_initial_candies_l1034_103485

theorem casper_initial_candies : 
  ∃ x : ℕ, 
    (∃ y1 : ℕ, y1 = x / 2 - 3) ∧
    (∃ y2 : ℕ, y2 = y1 / 2 - 5) ∧
    (∃ y3 : ℕ, y3 = y2 / 2 - 2) ∧
    (y3 = 10) ∧
    x = 122 := 
sorry

end casper_initial_candies_l1034_103485


namespace lowest_possible_number_of_students_l1034_103438

theorem lowest_possible_number_of_students :
  ∃ n : ℕ, (n % 12 = 0 ∧ n % 24 = 0) ∧ ∀ m : ℕ, ((m % 12 = 0 ∧ m % 24 = 0) → n ≤ m) :=
sorry

end lowest_possible_number_of_students_l1034_103438


namespace henry_age_l1034_103401

theorem henry_age (H J : ℕ) (h1 : H + J = 43) (h2 : H - 5 = 2 * (J - 5)) : H = 27 :=
by
  -- This is where we would prove the theorem based on the given conditions
  sorry

end henry_age_l1034_103401


namespace kitten_weight_l1034_103411

theorem kitten_weight :
  ∃ (x y z : ℝ), x + y + z = 36 ∧ x + z = 3 * y ∧ x + y = 1 / 2 * z ∧ x = 3 := 
by
  sorry

end kitten_weight_l1034_103411


namespace common_root_for_permutations_of_coeffs_l1034_103451

theorem common_root_for_permutations_of_coeffs :
  ∀ (a b c d : ℤ), (a = -7 ∨ a = 4 ∨ a = -3 ∨ a = 6) ∧ 
                   (b = -7 ∨ b = 4 ∨ b = -3 ∨ b = 6) ∧
                   (c = -7 ∨ c = 4 ∨ c = -3 ∨ c = 6) ∧
                   (d = -7 ∨ d = 4 ∨ d = -3 ∨ d = 6) ∧
                   (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (a * 1^3 + b * 1^2 + c * 1 + d = 0) :=
by
  intros a b c d h
  sorry

end common_root_for_permutations_of_coeffs_l1034_103451


namespace find_c_l1034_103470

-- Define that \( r \) and \( s \) are roots of \( 2x^2 - 4x - 5 \)
variables (r s : ℚ)
-- Condition: sum of roots \( r + s = 2 \)
axiom sum_of_roots : r + s = 2
-- Condition: product of roots \( rs = -5/2 \)
axiom product_of_roots : r * s = -5 / 2

-- Definition of \( c \) based on the roots \( r-3 \) and \( s-3 \)
def c : ℚ := (r - 3) * (s - 3)

-- The theorem to be proved
theorem find_c : c = 1 / 2 :=
by
  sorry

end find_c_l1034_103470


namespace total_surface_area_of_modified_cube_l1034_103422

-- Define the side length of the original cube
def side_length_cube := 3

-- Define the side length of the holes
def side_length_hole := 1

-- Define the condition of the surface area calculation
def total_surface_area_including_internal (side_length_cube side_length_hole : ℕ) : ℕ :=
  let original_surface_area := 6 * (side_length_cube * side_length_cube)
  let reduction_area := 6 * (side_length_hole * side_length_hole)
  let remaining_surface_area := original_surface_area - reduction_area
  let interior_surface_area := 6 * (4 * side_length_hole * side_length_cube)
  remaining_surface_area + interior_surface_area

-- Statement for the proof
theorem total_surface_area_of_modified_cube : total_surface_area_including_internal 3 1 = 72 :=
by
  -- This is the statement; the proof is omitted as "sorry"
  sorry

end total_surface_area_of_modified_cube_l1034_103422


namespace sum_of_two_relatively_prime_integers_l1034_103449

theorem sum_of_two_relatively_prime_integers (x y : ℕ) : 0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧
  gcd x y = 1 ∧ x * y + x + y = 119 ∧ x + y = 20 :=
by
  sorry

end sum_of_two_relatively_prime_integers_l1034_103449


namespace closing_price_l1034_103460

theorem closing_price
  (opening_price : ℝ)
  (increase_percentage : ℝ)
  (h_opening_price : opening_price = 15)
  (h_increase_percentage : increase_percentage = 6.666666666666665) :
  opening_price * (1 + increase_percentage / 100) = 16 :=
by
  sorry

end closing_price_l1034_103460


namespace coloringBooks_shelves_l1034_103494

variables (initialStock soldBooks shelves : ℕ)

-- Given conditions
def initialBooks : initialStock = 87 := sorry
def booksSold : soldBooks = 33 := sorry
def numberOfShelves : shelves = 9 := sorry

-- Number of coloring books per shelf
def coloringBooksPerShelf (remainingBooksResult : ℕ) (booksPerShelfResult : ℕ) : Prop :=
  remainingBooksResult = initialStock - soldBooks ∧ booksPerShelfResult = remainingBooksResult / shelves

-- Prove the number of coloring books per shelf is 6
theorem coloringBooks_shelves (remainingBooksResult booksPerShelfResult : ℕ) : 
  coloringBooksPerShelf initialStock soldBooks shelves remainingBooksResult booksPerShelfResult →
  booksPerShelfResult = 6 :=
sorry

end coloringBooks_shelves_l1034_103494


namespace fruit_vendor_l1034_103483

theorem fruit_vendor (x y a b : ℕ) (C1 : 60 * x + 40 * y = 3100) (C2 : x + y = 60) 
                     (C3 : 15 * a + 20 * b = 600) (C4 : 3 * a + 4 * b = 120)
                     (C5 : 3 * a + 4 * b + 3 * (x - a) + 4 * (y - b) = 250) :
  (x = 35 ∧ y = 25) ∧ (820 - 12 * a - 16 * b = 340) ∧ (a + b = 52 ∨ a + b = 53) :=
by
  sorry

end fruit_vendor_l1034_103483


namespace find_a5_l1034_103482

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, ∃ q : ℝ, a (n + m) = a n * q ^ m

theorem find_a5
  (h : geometric_sequence a)
  (h3 : a 3 = 2)
  (h7 : a 7 = 8) :
  a 5 = 4 :=
sorry

end find_a5_l1034_103482


namespace num_real_numbers_l1034_103419

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l1034_103419


namespace range_of_m_l1034_103448

theorem range_of_m (m : ℝ) :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + m ≤ 0) → 1 < m := by
  sorry

end range_of_m_l1034_103448


namespace problem1_problem2_l1034_103424

theorem problem1 : -1 + (-6) - (-4) + 0 = -3 := by
  sorry

theorem problem2 : 24 * (-1 / 4) / (-3 / 2) = 4 := by
  sorry

end problem1_problem2_l1034_103424


namespace relationship_between_y1_y2_l1034_103445

variable (k b y1 y2 : ℝ)

-- Let A = (-3, y1) and B = (4, y2) be points on the line y = kx + b, with k < 0
axiom A_on_line : y1 = k * -3 + b
axiom B_on_line : y2 = k * 4 + b
axiom k_neg : k < 0

theorem relationship_between_y1_y2 : y1 > y2 :=
by sorry

end relationship_between_y1_y2_l1034_103445


namespace simplify_sqrt_expr_l1034_103408

/-- Simplify the given radical expression and prove its equivalence to the expected result. -/
theorem simplify_sqrt_expr :
  (Real.sqrt (5 * 3) * Real.sqrt ((3 ^ 4) * (5 ^ 2)) = 225 * Real.sqrt 15) := 
by
  sorry

end simplify_sqrt_expr_l1034_103408


namespace min_value_a_b_inv_a_inv_b_l1034_103454

theorem min_value_a_b_inv_a_inv_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 :=
sorry

end min_value_a_b_inv_a_inv_b_l1034_103454


namespace compute_58_sq_pattern_l1034_103439

theorem compute_58_sq_pattern : (58 * 58 = 56 * 60 + 4) :=
by
  sorry

end compute_58_sq_pattern_l1034_103439


namespace simplify_division_l1034_103498

noncomputable def a := 5 * 10 ^ 10
noncomputable def b := 2 * 10 ^ 4 * 10 ^ 2

theorem simplify_division : a / b = 25000 := by
  sorry

end simplify_division_l1034_103498


namespace john_read_bible_in_weeks_l1034_103453

-- Given Conditions
def reads_per_hour : ℕ := 50
def reads_per_day_hours : ℕ := 2
def bible_length_pages : ℕ := 2800

-- Calculated values based on the given conditions
def reads_per_day : ℕ := reads_per_hour * reads_per_day_hours
def days_to_finish : ℕ := bible_length_pages / reads_per_day
def days_per_week : ℕ := 7

-- The proof statement
theorem john_read_bible_in_weeks : days_to_finish / days_per_week = 4 := by
  sorry

end john_read_bible_in_weeks_l1034_103453


namespace lcm_1230_924_l1034_103433

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 :=
by
  /- Proof steps skipped -/
  sorry

end lcm_1230_924_l1034_103433


namespace necessary_not_sufficient_condition_l1034_103466

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the condition for the problem
def condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the proof problem: Prove that the interval is a necessary but not sufficient condition for f(x) < 0
theorem necessary_not_sufficient_condition : 
  ∀ x : ℝ, condition x → ¬ (∀ y : ℝ, condition y → f y < 0) :=
sorry

end necessary_not_sufficient_condition_l1034_103466


namespace find_k_l1034_103462

theorem find_k (k : ℝ) :
  (∃ x y : ℝ, y = x + 2 * k ∧ y = 2 * x + k + 1 ∧ x^2 + y^2 = 4) ↔
  (k = 1 ∨ k = -1/5) := 
sorry

end find_k_l1034_103462


namespace numberOfBookshelves_l1034_103497

-- Define the conditions as hypotheses
def numBooks : ℕ := 23
def numMagazines : ℕ := 61
def totalItems : ℕ := 2436

-- Define the number of items per bookshelf
def itemsPerBookshelf : ℕ := numBooks + numMagazines

-- State the theorem to be proven
theorem numberOfBookshelves (bookshelves : ℕ) :
  itemsPerBookshelf * bookshelves = totalItems → 
  bookshelves = 29 :=
by
  -- placeholder for proof
  sorry

end numberOfBookshelves_l1034_103497


namespace tank_capacity_l1034_103412

variable (c w : ℕ)

-- Conditions
def initial_fraction (w c : ℕ) : Prop := w = c / 7
def final_fraction (w c : ℕ) : Prop := (w + 2) = c / 5

-- The theorem statement
theorem tank_capacity : 
  initial_fraction w c → 
  final_fraction w c → 
  c = 35 := 
by
  sorry  -- indicates that the proof is not provided

end tank_capacity_l1034_103412


namespace ratio_of_areas_l1034_103421

-- Definitions of conditions
def side_length (s : ℝ) : Prop := s > 0
def original_area (A s : ℝ) : Prop := A = s^2

-- Definition of the new area after folding
def new_area (B A s : ℝ) : Prop := B = (7/8) * s^2

-- The proof statement to show the ratio B/A is 7/8
theorem ratio_of_areas (s A B : ℝ) (h_side : side_length s) (h_area : original_area A s) (h_B : new_area B A s) : 
  B / A = 7 / 8 := 
by 
  sorry

end ratio_of_areas_l1034_103421


namespace karen_total_cost_l1034_103487

noncomputable def calculate_total_cost (burger_price sandwich_price smoothie_price : ℝ) (num_smoothies : ℕ)
  (discount_rate tax_rate : ℝ) (order_time : ℕ) : ℝ :=
  let total_cost_before_discount := burger_price + sandwich_price + (num_smoothies * smoothie_price)
  let discount := if total_cost_before_discount > 15 ∧ order_time ≥ 1400 ∧ order_time ≤ 1600 then total_cost_before_discount * discount_rate else 0
  let reduced_price := total_cost_before_discount - discount
  let tax := reduced_price * tax_rate
  reduced_price + tax

theorem karen_total_cost :
  calculate_total_cost 5.75 4.50 4.25 2 0.20 0.12 1545 = 16.80 :=
by
  sorry

end karen_total_cost_l1034_103487


namespace range_of_m_l1034_103432

-- Define the function g as an even function on the interval [-2, 2] 
-- and monotonically decreasing on [0, 2]

variable {g : ℝ → ℝ}

axiom even_g : ∀ x, g x = g (-x)
axiom mono_dec_g : ∀ {x y}, 0 ≤ x → x ≤ y → g y ≤ g x
axiom domain_g : ∀ x, -2 ≤ x ∧ x ≤ 2

theorem range_of_m (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) (h : g (1 - m) < g m) : -1 ≤ m ∧ m < 1 / 2 :=
sorry

end range_of_m_l1034_103432


namespace function_inequality_m_l1034_103435

theorem function_inequality_m (m : ℝ) : (∀ x : ℝ, (1 / 2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) ↔ m ≥ (3 / 2) := sorry

end function_inequality_m_l1034_103435


namespace penny_canoe_l1034_103486

theorem penny_canoe (P : ℕ)
  (h1 : 140 * (2/3 : ℚ) * P + 35 = 595) : P = 6 :=
sorry

end penny_canoe_l1034_103486


namespace arithm_prog_diff_max_l1034_103484

noncomputable def find_most_common_difference (a b c : Int) : Prop :=
  let d := a - b
  (b = a - d) ∧ (c = a - 2 * d) ∧
  (2 * a * 2 * a - 4 * 2 * a * c ≥ 0) ∧
  (2 * a * 2 * b - 4 * 2 * a * c ≥ 0) ∧
  (2 * b * 2 * b - 4 * 2 * b * c ≥ 0) ∧
  (2 * b * c - 4 * 2 * b * a ≥ 0) ∧
  (c * c - 4 * c * 2 * b ≥ 0) ∧
  ((2 * a * c - 4 * 2 * c * b) ≥ 0)

theorem arithm_prog_diff_max (a b c Dmax: Int) : 
  find_most_common_difference 4 (-1) (-6) ∧ Dmax = -5 :=
by 
  sorry

end arithm_prog_diff_max_l1034_103484


namespace triangle_area_tangent_log2_l1034_103467

open Real

noncomputable def log_base_2 (x : ℝ) : ℝ := log x / log 2

theorem triangle_area_tangent_log2 :
  let y := log_base_2
  let f := fun x : ℝ => y x
  let deriv := (deriv f 1)
  let tangent_line := fun x : ℝ => deriv * (x - 1) + f 1
  let x_intercept := 1
  let y_intercept := tangent_line 0
  
  (1 : ℝ) * (abs y_intercept) / 2 = 1 / (2 * log 2) := by
  sorry

end triangle_area_tangent_log2_l1034_103467


namespace Murtha_pebble_collection_l1034_103410

def sum_of_first_n_natural_numbers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem Murtha_pebble_collection : sum_of_first_n_natural_numbers 20 = 210 := by
  sorry

end Murtha_pebble_collection_l1034_103410


namespace skateboard_weight_is_18_l1034_103456

def weight_of_canoe : Nat := 45
def weight_of_four_canoes := 4 * weight_of_canoe
def weight_of_ten_skateboards := weight_of_four_canoes
def weight_of_one_skateboard := weight_of_ten_skateboards / 10

theorem skateboard_weight_is_18 : weight_of_one_skateboard = 18 := by
  sorry

end skateboard_weight_is_18_l1034_103456


namespace Joe_team_wins_eq_1_l1034_103489

-- Definition for the points a team gets for winning a game.
def points_per_win := 3
-- Definition for the points a team gets for a tie game.
def points_per_tie := 1

-- Given conditions
def Joe_team_draws := 3
def first_place_wins := 2
def first_place_ties := 2
def points_difference := 2

def first_place_points := (first_place_wins * points_per_win) + (first_place_ties * points_per_tie)

def Joe_team_total_points := first_place_points - points_difference
def Joe_team_points_from_ties := Joe_team_draws * points_per_tie
def Joe_team_points_from_wins := Joe_team_total_points - Joe_team_points_from_ties

-- To prove: number of games Joe's team won
theorem Joe_team_wins_eq_1 : (Joe_team_points_from_wins / points_per_win) = 1 :=
by
  sorry

end Joe_team_wins_eq_1_l1034_103489


namespace express_as_scientific_notation_l1034_103427

-- Define the question and condition
def trillion : ℝ := 1000000000000
def num := 6.13 * trillion

-- The main statement to be proven
theorem express_as_scientific_notation : num = 6.13 * 10^12 :=
by
  sorry

end express_as_scientific_notation_l1034_103427


namespace max_value_y_eq_neg10_l1034_103405

open Real

theorem max_value_y_eq_neg10 (x : ℝ) (hx : x > 0) : 
  ∃ y, y = 2 - 9 * x - 4 / x ∧ (∀ z, (∃ (x' : ℝ), x' > 0 ∧ z = 2 - 9 * x' - 4 / x') → z ≤ y) ∧ y = -10 :=
by
  sorry

end max_value_y_eq_neg10_l1034_103405


namespace book_pages_l1034_103478

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
  pages_per_day = 8 → days = 12 → total_pages = pages_per_day * days → total_pages = 96 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end book_pages_l1034_103478


namespace length_of_first_train_solution_l1034_103420

noncomputable def length_of_first_train (speed1_kmph speed2_kmph : ℝ) (length2_m time_s : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (5 / 18)
  let speed2_mps := speed2_kmph * (5 / 18)
  let relative_speed_mps := speed1_mps + speed2_mps
  let combined_length_m := relative_speed_mps * time_s
  combined_length_m - length2_m

theorem length_of_first_train_solution 
  (speed1_kmph : ℝ) 
  (speed2_kmph : ℝ) 
  (length2_m : ℝ) 
  (time_s : ℝ) 
  (h₁ : speed1_kmph = 42) 
  (h₂ : speed2_kmph = 30) 
  (h₃ : length2_m = 120) 
  (h₄ : time_s = 10.999120070394369) : 
  length_of_first_train speed1_kmph speed2_kmph length2_m time_s = 99.98 :=
by 
  sorry

end length_of_first_train_solution_l1034_103420


namespace turtles_remaining_on_log_l1034_103443

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l1034_103443


namespace initial_oranges_count_l1034_103428

theorem initial_oranges_count 
  (O : ℕ)
  (h1 : 10 = O - 13) : 
  O = 23 := 
sorry

end initial_oranges_count_l1034_103428


namespace range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l1034_103452

theorem range_of_x_if_p_and_q_true (a : ℝ) (p q : ℝ → Prop) (h_a : a = 1) (h_p : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (h_q : ∀ x, q x ↔ (x-3)^2 < 1) (h_pq : ∀ x, p x ∧ q x) :
  ∀ x, 2 < x ∧ x < 3 :=
by
  sorry

theorem range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q (p q : ℝ → Prop) (h_neg : ∀ x, ¬p x → ¬q x) : 
  ∀ a : ℝ, a > 0 → (a ≥ 4/3 ∧ a ≤ 2) :=
by
  sorry

end range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l1034_103452


namespace isosceles_triangles_perimeter_l1034_103406

theorem isosceles_triangles_perimeter (c d : ℕ) 
  (h1 : ¬(7 = c ∧ 10 = d) ∧ ¬(7 = d ∧ 10 = c))
  (h2 : 2 * c + d = 24) :
  d = 2 :=
sorry

end isosceles_triangles_perimeter_l1034_103406


namespace problem1_problem2_l1034_103400

variable (x : ℝ)

theorem problem1 : 
  (3 * x + 1) * (3 * x - 1) - (3 * x + 1)^2 = -6 * x - 2 :=
sorry

theorem problem2 : 
  (6 * x^4 - 8 * x^3) / (-2 * x^2) - (3 * x + 2) * (1 - x) = 3 * x - 2 :=
sorry

end problem1_problem2_l1034_103400


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l1034_103429

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l1034_103429


namespace square_perimeter_l1034_103499

-- Define the area of the square
def square_area := 720

-- Define the side length of the square
noncomputable def side_length := Real.sqrt square_area

-- Define the perimeter of the square
noncomputable def perimeter := 4 * side_length

-- Statement: Prove that the perimeter is 48 * sqrt(5)
theorem square_perimeter : perimeter = 48 * Real.sqrt 5 :=
by
  -- The proof is omitted as instructed
  sorry

end square_perimeter_l1034_103499


namespace favouring_more_than_one_is_39_l1034_103463

def percentage_favouring_more_than_one (x : ℝ) : Prop :=
  let sum_two : ℝ := 8 + 6 + 4 + 2 + 7 + 5 + 3 + 5 + 3 + 2
  let sum_three : ℝ := 1 + 0.5 + 0.3 + 0.8 + 0.2 + 0.1 + 1.5 + 0.7 + 0.3 + 0.4
  let all_five : ℝ := 0.2
  x = sum_two - sum_three - all_five

theorem favouring_more_than_one_is_39 : percentage_favouring_more_than_one 39 := 
by
  sorry

end favouring_more_than_one_is_39_l1034_103463


namespace find_A_l1034_103416

theorem find_A (A B C D: ℕ) (h1: A ≠ B) (h2: A ≠ C) (h3: A ≠ D) (h4: B ≠ C) (h5: B ≠ D) (h6: C ≠ D)
  (hAB: A * B = 72) (hCD: C * D = 72) (hDiff: A - B = C + D + 2) : A = 6 :=
sorry

end find_A_l1034_103416


namespace correct_average_l1034_103479

theorem correct_average (incorrect_avg : ℝ) (n : ℕ) (wrong_num correct_num : ℝ)
  (h_avg : incorrect_avg = 23)
  (h_n : n = 10)
  (h_wrong : wrong_num = 26)
  (h_correct : correct_num = 36) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 24 :=
by
  -- Proof goes here
  sorry

end correct_average_l1034_103479


namespace positive_integer_in_base_proof_l1034_103475

noncomputable def base_conversion_problem (A B : ℕ) (n : ℕ) : Prop :=
  n = 9 * A + B ∧ n = 8 * B + A ∧ A < 9 ∧ B < 8 ∧ A ≠ 0 ∧ B ≠ 0

theorem positive_integer_in_base_proof (A B n : ℕ) (h : base_conversion_problem A B n) : n = 0 :=
sorry

end positive_integer_in_base_proof_l1034_103475


namespace common_ratio_l1034_103476

theorem common_ratio
  (a b : ℝ)
  (h_arith : 2 * a = 1 + b)
  (h_geom : (a + 2) ^ 2 = 3 * (b + 5))
  (h_non_zero_a : a + 2 ≠ 0)
  (h_non_zero_b : b + 5 ≠ 0) :
  (a = 4 ∧ b = 7) ∧ (b + 5) / (a + 2) = 2 :=
by {
  sorry
}

end common_ratio_l1034_103476


namespace functional_equation_l1034_103465

noncomputable def f : ℝ → ℝ :=
  sorry

theorem functional_equation (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end functional_equation_l1034_103465
