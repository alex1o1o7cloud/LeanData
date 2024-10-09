import Mathlib

namespace maximum_fraction_l2173_217332

theorem maximum_fraction (a b h : ℝ) (d : ℝ) (h_d_def : d = Real.sqrt (a^2 + b^2 + h^2)) :
  (a + b + h) / d ≤ Real.sqrt 3 :=
sorry

end maximum_fraction_l2173_217332


namespace no_x2_term_a_eq_1_l2173_217328

theorem no_x2_term_a_eq_1 (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a * x + 1) * (x^2 - 3 * a + 2) = x^4 + bx^3 + cx + d) →
  c = 0 →
  a = 1 :=
sorry

end no_x2_term_a_eq_1_l2173_217328


namespace sqrt_product_eq_six_l2173_217351

theorem sqrt_product_eq_six (sqrt24 sqrtThreeOverTwo: ℝ)
    (h1 : sqrt24 = Real.sqrt 24)
    (h2 : sqrtThreeOverTwo = Real.sqrt (3 / 2))
    : sqrt24 * sqrtThreeOverTwo = 6 := by
  sorry

end sqrt_product_eq_six_l2173_217351


namespace transylvanian_sanity_l2173_217343

theorem transylvanian_sanity (sane : Prop) (belief : Prop) (h1 : sane) (h2 : sane → belief) : belief :=
by
  sorry

end transylvanian_sanity_l2173_217343


namespace value_of_a_plus_b_l2173_217396

theorem value_of_a_plus_b :
  ∀ (a b x y : ℝ), x = 3 → y = -2 → 
  a * x + b * y = 2 → b * x + a * y = -3 → 
  a + b = -1 := 
by
  intros a b x y hx hy h1 h2
  subst hx
  subst hy
  sorry

end value_of_a_plus_b_l2173_217396


namespace fabian_cards_l2173_217315

theorem fabian_cards : ∃ (g y b r : ℕ),
  (g > 0 ∧ g < 10) ∧ (y > 0 ∧ y < 10) ∧ (b > 0 ∧ b < 10) ∧ (r > 0 ∧ r < 10) ∧
  (g * y = g) ∧
  (b = r) ∧
  (b * r = 10 * g + y) ∧ 
  (g = 8) ∧
  (y = 1) ∧
  (b = 9) ∧
  (r = 9) :=
by
  sorry

end fabian_cards_l2173_217315


namespace find_f_of_4_l2173_217362

def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem find_f_of_4 {a b c : ℝ} (h1 : f a b c 1 = 3) (h2 : f a b c 2 = 12) (h3 : f a b c 3 = 27) :
  f a b c 4 = 48 := 
sorry

end find_f_of_4_l2173_217362


namespace age_ratios_l2173_217392

variable (A B : ℕ)

-- Given conditions
theorem age_ratios :
  (A / B = 2 / 1) → (A - 4 = B + 4) → ((A + 4) / (B - 4) = 5 / 1) :=
by
  intro h1 h2
  sorry

end age_ratios_l2173_217392


namespace mark_sprint_distance_l2173_217353

theorem mark_sprint_distance (t v : ℝ) (ht : t = 24.0) (hv : v = 6.0) : 
  t * v = 144.0 := 
by
  -- This theorem is formulated with the conditions that t = 24.0 and v = 6.0,
  -- we need to prove that the resulting distance is 144.0 miles.
  sorry

end mark_sprint_distance_l2173_217353


namespace restock_quantities_correct_l2173_217326

-- Definition for the quantities of cans required
def cans_peas : ℕ := 810
def cans_carrots : ℕ := 954
def cans_corn : ℕ := 675

-- Definition for the number of cans per box, pack, and case.
def cans_per_box_peas : ℕ := 4
def cans_per_pack_carrots : ℕ := 6
def cans_per_case_corn : ℕ := 5

-- Define the expected order quantities.
def order_boxes_peas : ℕ := 203
def order_packs_carrots : ℕ := 159
def order_cases_corn : ℕ := 135

-- Proof statement for the quantities required to restock exactly.
theorem restock_quantities_correct :
  (order_boxes_peas = Nat.ceil (cans_peas / cans_per_box_peas))
  ∧ (order_packs_carrots = cans_carrots / cans_per_pack_carrots)
  ∧ (order_cases_corn = cans_corn / cans_per_case_corn) :=
by
  sorry

end restock_quantities_correct_l2173_217326


namespace solution_interval_l2173_217331

theorem solution_interval:
  ∃ x : ℝ, (x^3 = 2^(2-x)) ∧ 1 < x ∧ x < 2 :=
by
  sorry

end solution_interval_l2173_217331


namespace divisor_of_930_l2173_217373

theorem divisor_of_930 : ∃ d > 1, d ∣ 930 ∧ ∀ e, e ∣ 930 → e > 1 → d ≤ e :=
by
  sorry

end divisor_of_930_l2173_217373


namespace parabola_ratio_l2173_217377

noncomputable def AF_over_BF (p : ℝ) (h_p : p > 0) : ℝ :=
  let AF := 4 * p
  let x := (4 / 7) * p -- derived from solving the equation in the solution
  AF / x

theorem parabola_ratio (p : ℝ) (h_p : p > 0) : AF_over_BF p h_p = 7 :=
  sorry

end parabola_ratio_l2173_217377


namespace arithmetic_mean_of_sequence_beginning_at_5_l2173_217366

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

def sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def arithmetic_mean (a d n : ℕ) : ℚ :=
  sequence_sum a d n / n

theorem arithmetic_mean_of_sequence_beginning_at_5 : 
  arithmetic_mean 5 1 60 = 34.5 :=
by
  sorry

end arithmetic_mean_of_sequence_beginning_at_5_l2173_217366


namespace zero_extreme_points_l2173_217395

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

theorem zero_extreme_points : ∀ x : ℝ, 
  ∃! (y : ℝ), deriv f y = 0 → y = x :=
by
  sorry

end zero_extreme_points_l2173_217395


namespace area_of_fifteen_sided_figure_l2173_217327

noncomputable def figure_area : ℝ :=
  let full_squares : ℝ := 6
  let num_triangles : ℝ := 10
  let triangles_to_rectangles : ℝ := num_triangles / 2
  let triangles_area : ℝ := triangles_to_rectangles
  full_squares + triangles_area

theorem area_of_fifteen_sided_figure :
  figure_area = 11 := by
  sorry

end area_of_fifteen_sided_figure_l2173_217327


namespace sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l2173_217336

open Real

theorem sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq :
  (∀ α : ℝ, sin α = cos α → ∃ k : ℤ, α = (k : ℝ) * π + π / 4) ∧
  (¬ ∀ k : ℤ, ∀ α : ℝ, α = (k : ℝ) * π + π / 4 → sin α = cos α) :=
by
  sorry

end sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l2173_217336


namespace find_k_l2173_217313

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

-- Given that the union of sets A and B is {1, 2, 3, 5}, prove that k = 3.
theorem find_k (k : ℕ) (h : A k ∪ B = {1, 2, 3, 5}) : k = 3 :=
by
  sorry

end find_k_l2173_217313


namespace quadratic_translation_l2173_217357

theorem quadratic_translation (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c = (x - 3)^2 - 2)) →
  b = 4 ∧ c = 6 :=
by
  sorry

end quadratic_translation_l2173_217357


namespace intersection_in_fourth_quadrant_l2173_217364

variable {a : ℝ} {x : ℝ}

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x / Real.log a
noncomputable def g (x : ℝ) (a : ℝ) := (1 - a) * x

theorem intersection_in_fourth_quadrant (h : a > 1) :
  ∃ x : ℝ, x > 0 ∧ f x a < 0 ∧ f x a = g x a :=
sorry

end intersection_in_fourth_quadrant_l2173_217364


namespace trapezoid_perimeter_calc_l2173_217310

theorem trapezoid_perimeter_calc 
  (EF GH : ℝ) (d : ℝ)
  (h_parallel : EF = 10) 
  (h_eq : GH = 22) 
  (h_distance : d = 5) 
  (h_parallel_cond : EF = 10 ∧ GH = 22 ∧ d = 5) 
: 32 + 2 * Real.sqrt 61 = (10 : ℝ) + 2 * (Real.sqrt ((12 / 2)^2 + 5^2)) + 22 := 
by {
  -- The proof goes here, but for now it's omitted
  sorry
}

end trapezoid_perimeter_calc_l2173_217310


namespace solve_for_x_l2173_217383

theorem solve_for_x (x : ℝ) (h : 4 * x + 45 ≠ 0) :
  (8 * x^2 + 80 * x + 4) / (4 * x + 45) = 2 * x + 3 → x = -131 / 22 := 
by 
  sorry

end solve_for_x_l2173_217383


namespace square_side_length_l2173_217384

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end square_side_length_l2173_217384


namespace dodecahedron_path_count_l2173_217348

/-- A regular dodecahedron with constraints on movement between faces. -/
def num_ways_dodecahedron_move : Nat := 810

/-- Proving the number of different ways to move from the top face to the bottom face of a regular dodecahedron via a series of adjacent faces, such that each face is visited at most once, and movement from the lower ring to the upper ring is not allowed is 810. -/
theorem dodecahedron_path_count :
  num_ways_dodecahedron_move = 810 :=
by
  -- Proof goes here
  sorry

end dodecahedron_path_count_l2173_217348


namespace volumes_relation_l2173_217347

-- Definitions and conditions based on the problem
variables {a b c : ℝ} (h_triangle : a > b) (h_triangle2 : b > c) (h_acute : 0 < θ ∧ θ < π)

-- The heights from vertices
variables (AD BE CF : ℝ)

-- Volumes of the tetrahedrons formed after folding
variables (V1 V2 V3 : ℝ)

-- The heights are given:
noncomputable def height_AD (BC : ℝ) (theta : ℝ) := AD
noncomputable def height_BE (CA : ℝ) (theta : ℝ) := BE
noncomputable def height_CF (AB : ℝ) (theta : ℝ) := CF

-- Using these heights and the acute nature of the triangle
noncomputable def volume_V1 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V1
noncomputable def volume_V2 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V2
noncomputable def volume_V3 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V3

-- The theorem stating the relationship between volumes
theorem volumes_relation
  (h_triangle: a > b)
  (h_triangle2: b > c)
  (h_acute: 0 < θ ∧ θ < π)
  (h_volumes: V1 > V2 ∧ V2 > V3):
  V1 > V2 ∧ V2 > V3 :=
sorry

end volumes_relation_l2173_217347


namespace correct_option_B_l2173_217386

theorem correct_option_B (x y a b : ℝ) :
  (3 * x + 2 * x^2 ≠ 5 * x) →
  (-y^2 * x + x * y^2 = 0) →
  (-a * b - a * b ≠ 0) →
  (3 * a^3 * b^2 - 2 * a^3 * b^2 ≠ 1) →
  (-y^2 * x + x * y^2 = 0) :=
by
  intros hA hB hC hD
  exact hB

end correct_option_B_l2173_217386


namespace eggs_at_park_l2173_217338

-- Define the number of eggs found at different locations
def eggs_at_club_house : Nat := 40
def eggs_at_town_hall : Nat := 15
def total_eggs_found : Nat := 80

-- Prove that the number of eggs found at the park is 25
theorem eggs_at_park :
  ∃ P : Nat, eggs_at_club_house + P + eggs_at_town_hall = total_eggs_found ∧ P = 25 := 
by
  sorry

end eggs_at_park_l2173_217338


namespace surface_area_sphere_dihedral_l2173_217322

open Real

theorem surface_area_sphere_dihedral (R a : ℝ) (hR : 0 < R) (haR : 0 < a ∧ a < R) (α : ℝ) :
  2 * R^2 * arccos ((R * cos α) / sqrt (R^2 - a^2 * sin α^2)) 
  - 2 * R * a * sin α * arccos ((a * cos α) / sqrt (R^2 - a^2 * sin α^2)) = sorry :=
sorry

end surface_area_sphere_dihedral_l2173_217322


namespace three_digit_numbers_eq_11_sum_squares_l2173_217352

theorem three_digit_numbers_eq_11_sum_squares :
  ∃ (N : ℕ), 
    (N = 550 ∨ N = 803) ∧
    (∃ (a b c : ℕ), 
      N = 100 * a + 10 * b + c ∧ 
      100 * a + 10 * b + c = 11 * (a ^ 2 + b ^ 2 + c ^ 2) ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9) :=
sorry

end three_digit_numbers_eq_11_sum_squares_l2173_217352


namespace point_in_second_quadrant_l2173_217379

theorem point_in_second_quadrant (m n : ℝ)
  (h_translation : ∃ A' : ℝ × ℝ, A' = (m+2, n+3) ∧ (A'.1 < 0) ∧ (A'.2 > 0)) :
  m < -2 ∧ n > -3 :=
by
  sorry

end point_in_second_quadrant_l2173_217379


namespace total_pieces_of_clothing_l2173_217376

-- Define the conditions:
def boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

-- Define the target statement:
theorem total_pieces_of_clothing : (boxes * (scarves_per_box + mittens_per_box)) = 32 :=
by
  sorry

end total_pieces_of_clothing_l2173_217376


namespace exist_infinite_a_l2173_217325

theorem exist_infinite_a (n : ℕ) (a : ℕ) (h₁ : ∃ k : ℕ, k > 0 ∧ (n^6 + 3 * a = (n^2 + 3 * k)^3)) : 
  ∃ f : ℕ → ℕ, ∀ m : ℕ, (∃ k : ℕ, k > 0 ∧ f m = 9 * k^3 + 3 * n^2 * k * (n^2 + 3 * k)) :=
by 
  sorry

end exist_infinite_a_l2173_217325


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l2173_217301

-- Define the basic conditions of the figures
def regular_pentagon (side_length : ℕ) : ℝ := 5 * side_length

-- Define ink length of a figure n
def ink_length (n : ℕ) : ℝ :=
  if n = 1 then regular_pentagon 1 else
  regular_pentagon (n-1) + (3 * (n - 1) + 2)

-- Part (a): Ink length of Figure 4
theorem ink_length_figure_4 : ink_length 4 = 38 := 
  by sorry

-- Part (b): Difference between ink length of Figure 9 and Figure 8
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 29 :=
  by sorry

-- Part (c): Ink length of Figure 100
theorem ink_length_figure_100 : ink_length 100 = 15350 :=
  by sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l2173_217301


namespace tea_set_costs_l2173_217361
noncomputable section

-- Definition for the conditions of part 1
def cost_condition1 (x y : ℝ) : Prop := x + 2 * y = 250
def cost_condition2 (x y : ℝ) : Prop := 3 * x + 4 * y = 600

-- Definition for the conditions of part 2
def cost_condition3 (a : ℝ) : ℝ := 108 * a + 60 * (80 - a)

-- Definition for the conditions of part 3
def profit (a b : ℝ) : ℝ := 30 * a + 20 * b

theorem tea_set_costs (x y : ℝ) (a : ℕ) :
  cost_condition1 x y →
  cost_condition2 x y →
  x = 100 ∧ y = 75 ∧ a ≤ 30 ∧ profit 30 50 = 1900 := by
  sorry

end tea_set_costs_l2173_217361


namespace card_draw_probability_l2173_217305

theorem card_draw_probability:
  let hearts := 13
  let diamonds := 13
  let clubs := 13
  let total_cards := 52
  let first_draw_probability := hearts / (total_cards : ℝ)
  let second_draw_probability := diamonds / (total_cards - 1 : ℝ)
  let third_draw_probability := clubs / (total_cards - 2 : ℝ)
  first_draw_probability * second_draw_probability * third_draw_probability = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l2173_217305


namespace find_least_skilled_painter_l2173_217307

-- Define the genders
inductive Gender
| Male
| Female

-- Define the family members
inductive Member
| Grandmother
| Niece
| Nephew
| Granddaughter

-- Define a structure to hold the properties of each family member
structure Properties where
  gender : Gender
  age : Nat
  isTwin : Bool

-- Assume the properties of each family member as given
def grandmother : Properties := { gender := Gender.Female, age := 70, isTwin := false }
def niece : Properties := { gender := Gender.Female, age := 20, isTwin := false }
def nephew : Properties := { gender := Gender.Male, age := 20, isTwin := true }
def granddaughter : Properties := { gender := Gender.Female, age := 20, isTwin := true }

-- Define the best painter
def bestPainter := niece

-- Conditions based on the problem (rephrased to match formalization)
def conditions (least_skilled : Member) : Prop :=
  (bestPainter.gender ≠ (match least_skilled with
                          | Member.Grandmother => grandmother
                          | Member.Niece => niece
                          | Member.Nephew => nephew
                          | Member.Granddaughter => granddaughter ).gender) ∧
  ((match least_skilled with
    | Member.Grandmother => grandmother
    | Member.Niece => niece
    | Member.Nephew => nephew
    | Member.Granddaughter => granddaughter ).isTwin) ∧
  (bestPainter.age = (match least_skilled with
                      | Member.Grandmother => grandmother
                      | Member.Niece => niece
                      | Member.Nephew => nephew
                      | Member.Granddaughter => granddaughter ).age)

-- Statement of the problem
theorem find_least_skilled_painter : ∃ m : Member, conditions m ∧ m = Member.Granddaughter :=
by
  sorry

end find_least_skilled_painter_l2173_217307


namespace base_angle_of_isosceles_triangle_l2173_217304

theorem base_angle_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : a = 50) (h₂ : a + b + c = 180) (h₃ : a = b ∨ b = c ∨ c = a) : 
  b = 50 ∨ b = 65 :=
by sorry

end base_angle_of_isosceles_triangle_l2173_217304


namespace sum_of_coefficients_eq_one_l2173_217393

theorem sum_of_coefficients_eq_one :
  ∀ x y : ℤ, (x - 2 * y) ^ 18 = (1 - 2 * 1) ^ 18 → (x - 2 * y) ^ 18 = 1 :=
by
  intros x y h
  sorry

end sum_of_coefficients_eq_one_l2173_217393


namespace minimum_a_for_cube_in_tetrahedron_l2173_217390

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  (Real.sqrt 6 / 12) * a

theorem minimum_a_for_cube_in_tetrahedron (a : ℝ) (r : ℝ) 
  (h_radius : r = radius_of_circumscribed_sphere a)
  (h_diag : Real.sqrt 3 = 2 * r) :
  a = 3 * Real.sqrt 2 :=
by
  sorry

end minimum_a_for_cube_in_tetrahedron_l2173_217390


namespace shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l2173_217374

-- (a) Prove that the area of the shaded region is 36 cm^2
theorem shaded_area_a (AB EF : ℕ) (h1 : AB = 10) (h2 : EF = 8) : (AB ^ 2) - (EF ^ 2) = 36 :=
by
  sorry

-- (b) Prove that the length of EF is 7 cm
theorem length_EF_b (AB : ℕ) (shaded_area : ℕ) (h1 : AB = 13) (h2 : shaded_area = 120)
  : ∃ EF, (AB ^ 2) - (EF ^ 2) = shaded_area ∧ EF = 7 :=
by
  sorry

-- (c) Prove that the length of EF is 9 cm
theorem length_EF_c (AB : ℕ) (h1 : AB = 18)
  : ∃ EF, (AB ^ 2) - ((1 / 4) * AB ^ 2) = (3 / 4) * AB ^ 2 ∧ EF = 9 :=
by
  sorry

-- (d) Prove that a / b = 5 / 3
theorem ratio_ab_d (a b : ℕ) (shaded_percent : ℚ) (h1 : shaded_percent = 0.64)
  : (a ^ 2) - ((0.36) * a ^ 2) = (a ^ 2) * shaded_percent ∧ (a / b) = (5 / 3) :=
by
  sorry

end shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l2173_217374


namespace sqrt_9_is_pm3_l2173_217319

theorem sqrt_9_is_pm3 : {x : ℝ | x ^ 2 = 9} = {3, -3} := sorry

end sqrt_9_is_pm3_l2173_217319


namespace cheat_buying_percentage_l2173_217333

-- Definitions for the problem
def profit_margin := 0.5
def cheat_selling := 0.2

-- Prove that the cheating percentage while buying is 20%
theorem cheat_buying_percentage : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ x = 0.2 := by
  sorry

end cheat_buying_percentage_l2173_217333


namespace simplify_expression_l2173_217355

theorem simplify_expression : 
  (((5 + 7 + 3) * 2 - 4) / 2 - (5 / 2) = 21 / 2) :=
by
  sorry

end simplify_expression_l2173_217355


namespace penguin_fish_consumption_l2173_217341

-- Definitions based on the conditions
def initial_penguins : ℕ := 158
def total_fish_per_day : ℕ := 237
def fish_per_penguin_per_day : ℚ := 1.5

-- Lean statement for the conditional problem
theorem penguin_fish_consumption
  (P : ℕ)
  (h_initial_penguins : P = initial_penguins)
  (h_total_fish_per_day : total_fish_per_day = 237)
  (h_current_penguins : P * 2 * 3 + 129 = 1077)
  : total_fish_per_day / P = fish_per_penguin_per_day := by
  sorry

end penguin_fish_consumption_l2173_217341


namespace range_of_a_neg_p_true_l2173_217382

theorem range_of_a_neg_p_true :
  (∀ x : ℝ, x ∈ Set.Ioo (-2:ℝ) 0 → x^2 + (2*a - 1)*x + a ≠ 0) →
  ∀ a : ℝ, a ∈ Set.Icc 0 ((2 + Real.sqrt 3) / 2) :=
sorry

end range_of_a_neg_p_true_l2173_217382


namespace chocolate_cost_in_promotion_l2173_217309

/-!
Bernie buys two chocolates every week at a local store, where one chocolate costs $3.
In a different store with a promotion, each chocolate costs some amount and Bernie would save $6 
in three weeks if he bought his chocolates there. Prove that the cost of one chocolate 
in the store with the promotion is $2.
-/

theorem chocolate_cost_in_promotion {n p_local savings : ℕ} (weeks : ℕ) (p_promo : ℕ)
  (h_n : n = 2)
  (h_local : p_local = 3)
  (h_savings : savings = 6)
  (h_weeks : weeks = 3)
  (h_promo : p_promo = (p_local * n * weeks - savings) / (n * weeks)) :
  p_promo = 2 :=
by {
  -- Proof would go here
  sorry
}

end chocolate_cost_in_promotion_l2173_217309


namespace determine_n_l2173_217306

theorem determine_n (n : ℕ) (h1 : 0 < n) 
(h2 : ∃ (sols : Finset (ℕ × ℕ × ℕ)), 
  (∀ (x y z : ℕ), (x, y, z) ∈ sols ↔ 3 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  ∧ sols.card = 55) : 
  n = 36 := 
by 
  sorry 

end determine_n_l2173_217306


namespace expected_value_is_100_cents_l2173_217312

-- Definitions for the values of the coins
def value_quarter : ℕ := 25
def value_half_dollar : ℕ := 50
def value_dollar : ℕ := 100

-- Define the total value of all coins
def total_value : ℕ := 2 * value_quarter + value_half_dollar + value_dollar

-- Probability of heads for a single coin
def p_heads : ℚ := 1 / 2

-- Expected value calculation
def expected_value : ℚ := p_heads * ↑total_value

-- The theorem we need to prove
theorem expected_value_is_100_cents : expected_value = 100 :=
by
  -- This is where the proof would go, but we are omitting it
  sorry

end expected_value_is_100_cents_l2173_217312


namespace cubic_function_value_l2173_217320

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

theorem cubic_function_value (p q r s : ℝ) (h : g (-3) p q r s = -2) :
  12 * p - 6 * q + 3 * r - s = 2 :=
sorry

end cubic_function_value_l2173_217320


namespace least_number_of_stamps_is_11_l2173_217356

theorem least_number_of_stamps_is_11 (s t : ℕ) (h : 5 * s + 6 * t = 60) : s + t = 11 := 
  sorry

end least_number_of_stamps_is_11_l2173_217356


namespace distance_to_fourth_side_l2173_217334

-- Let s be the side length of the square.
variable (s : ℝ) (d1 d2 d3 d4 : ℝ)

-- The given conditions:
axiom h1 : d1 = 4
axiom h2 : d2 = 7
axiom h3 : d3 = 13
axiom h4 : d1 + d2 + d3 + d4 = s
axiom h5 : 0 < d4

-- The statement to prove:
theorem distance_to_fourth_side : d4 = 10 ∨ d4 = 16 :=
by
  sorry

end distance_to_fourth_side_l2173_217334


namespace find_prime_c_l2173_217316

-- Define the statement of the problem
theorem find_prime_c (c : ℕ) (hc : Nat.Prime c) (h : ∃ m : ℕ, (m > 0) ∧ (11 * c + 1 = m^2)) : c = 13 :=
by
  sorry

end find_prime_c_l2173_217316


namespace min_sum_nonpos_l2173_217375

theorem min_sum_nonpos (a b : ℤ) (h_nonpos_a : a ≤ 0) (h_nonpos_b : b ≤ 0) (h_prod : a * b = 144) : 
  a + b = -30 :=
sorry

end min_sum_nonpos_l2173_217375


namespace greatest_sum_consecutive_integers_lt_500_l2173_217354

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l2173_217354


namespace determine_y_l2173_217363

theorem determine_y (y : ℝ) (y_nonzero : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := 
sorry

end determine_y_l2173_217363


namespace find_m_l2173_217367

-- Let m be a real number such that m > 1 and
-- \sum_{n=1}^{\infty} \frac{3n+2}{m^n} = 2.
theorem find_m (m : ℝ) (h1 : m > 1) 
(h2 : ∑' n : ℕ, (3 * (n + 1) + 2) / m^(n + 1) = 2) : 
  m = 3 :=
sorry

end find_m_l2173_217367


namespace greatest_line_segment_length_l2173_217371

theorem greatest_line_segment_length (r : ℝ) (h : r = 4) : 
  ∃ d : ℝ, d = 2 * r ∧ d = 8 :=
by
  sorry

end greatest_line_segment_length_l2173_217371


namespace point_outside_circle_l2173_217321

theorem point_outside_circle (a b : ℝ)
  (h_line_intersects_circle : ∃ (x1 y1 x2 y2 : ℝ), 
     x1^2 + y1^2 = 1 ∧ 
     x2^2 + y2^2 = 1 ∧ 
     a * x1 + b * y1 = 1 ∧ 
     a * x2 + b * y2 = 1 ∧ 
     (x1, y1) ≠ (x2, y2)) : 
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l2173_217321


namespace percentage_of_400_that_results_in_224_point_5_l2173_217359

-- Let x be the unknown percentage of 400
variable (x : ℝ)

-- Condition: x% of 400 plus 45% of 250 equals 224.5
def condition (x : ℝ) : Prop := (400 * x / 100) + (250 * 45 / 100) = 224.5

theorem percentage_of_400_that_results_in_224_point_5 : condition 28 :=
by
  -- proof goes here
  sorry

end percentage_of_400_that_results_in_224_point_5_l2173_217359


namespace mold_growth_problem_l2173_217385

/-- Given the conditions:
    - Initial mold spores: 50 at 9:00 a.m.
    - Colony doubles in size every 10 minutes.
    - Time elapsed: 70 minutes from 9:00 a.m. to 10:10 a.m.,

    Prove that the number of mold spores at 10:10 a.m. is 6400 -/
theorem mold_growth_problem : 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  final_population = 6400 :=
by 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  sorry

end mold_growth_problem_l2173_217385


namespace _l2173_217339

def triangle (A B C : Type) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def angles_not_equal_sides_not_equal (A B C : Type) (angleB angleC : ℝ) (sideAC sideAB : ℝ) : Prop :=
  triangle A B C →
  (angleB ≠ angleC → sideAC ≠ sideAB)
  
lemma xiaoming_theorem {A B C : Type} 
  (hTriangle : triangle A B C)
  (angleB angleC : ℝ)
  (sideAC sideAB : ℝ) :
  angleB ≠ angleC → sideAC ≠ sideAB := 
sorry

end _l2173_217339


namespace no_such_function_exists_l2173_217370

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l2173_217370


namespace equation_of_plane_l2173_217303

/--
The equation of the plane passing through the points (2, -2, 2) and (0, 0, 2),
and which is perpendicular to the plane 2x - y + 4z = 8, is given by:
Ax + By + Cz + D = 0 where A, B, C, D are integers such that A > 0 and gcd(|A|,|B|,|C|,|D|) = 1.
-/
theorem equation_of_plane :
  ∃ (A B C D : ℤ),
    A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
    (∀ x y z : ℤ, A * x + B * y + C * z + D = 0 ↔ x + y = 0) :=
sorry

end equation_of_plane_l2173_217303


namespace sum_of_positive_x_and_y_is_ten_l2173_217329

theorem sum_of_positive_x_and_y_is_ten (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^3 + y^3 + (x + y)^3 + 30 * x * y = 2000) : 
  x + y = 10 :=
sorry

end sum_of_positive_x_and_y_is_ten_l2173_217329


namespace checkerboard_sum_is_328_l2173_217314

def checkerboard_sum : Nat :=
  1 + 2 + 9 + 8 + 73 + 74 + 81 + 80

theorem checkerboard_sum_is_328 : checkerboard_sum = 328 := by
  sorry

end checkerboard_sum_is_328_l2173_217314


namespace cute_pairs_count_l2173_217372

def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

def max_cute_pairs : Prop :=
  ∀ (ages : Finset ℕ), 
  (∀ x ∈ ages, 1 ≤ x ∧ x ≤ 100) →
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ pair ∈ pairs, is_cute_pair pair.1 pair.2) ∧
    (∀ x ∈ pairs, ∀ y ∈ pairs, x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) ∧
    pairs.card = 43)

theorem cute_pairs_count : max_cute_pairs := 
sorry

end cute_pairs_count_l2173_217372


namespace remainder_83_pow_89_times_5_mod_11_l2173_217378

theorem remainder_83_pow_89_times_5_mod_11 : 
  (83^89 * 5) % 11 = 10 := 
by
  have h1 : 83 % 11 = 6 := by sorry
  have h2 : 6^10 % 11 = 1 := by sorry
  have h3 : 89 = 8 * 10 + 9 := by sorry
  sorry

end remainder_83_pow_89_times_5_mod_11_l2173_217378


namespace behavior_of_f_in_interval_l2173_217388

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- Define the property of even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- The theorem statement
theorem behavior_of_f_in_interval (m : ℝ) (hf_even : is_even_function (f m)) :
  m = 0 → (∀ x : ℝ, -4 < x ∧ x < 0 → f 0 x < f 0 (-x)) ∧ (∀ x : ℝ, 0 < x ∧ x < 2 → f 0 (-x) > f 0 x) :=
by 
  sorry

end behavior_of_f_in_interval_l2173_217388


namespace count_letters_with_both_l2173_217389

theorem count_letters_with_both (a b c x : ℕ) 
  (h₁ : a = 24) 
  (h₂ : b = 7) 
  (h₃ : c = 40) 
  (H : a + b + x = c) : 
  x = 9 :=
by {
  -- Proof here
  sorry
}

end count_letters_with_both_l2173_217389


namespace simplify_fraction_l2173_217349

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 :=
by sorry

end simplify_fraction_l2173_217349


namespace total_cookies_is_58_l2173_217397

noncomputable def total_cookies : ℝ :=
  let M : ℝ := 5
  let T : ℝ := 2 * M
  let W : ℝ := T + 0.4 * T
  let Th : ℝ := W - 0.25 * W
  let F : ℝ := Th - 0.25 * Th
  let Sa : ℝ := F - 0.25 * F
  let Su : ℝ := Sa - 0.25 * Sa
  M + T + W + Th + F + Sa + Su

theorem total_cookies_is_58 : total_cookies = 58 :=
by
  sorry

end total_cookies_is_58_l2173_217397


namespace total_value_l2173_217350

/-- 
The total value of the item V can be determined based on the given conditions.
- The merchant paid an import tax of $109.90.
- The tax rate is 7%.
- The tax is only on the portion of the value above $1000.

Given these conditions, prove that the total value V is 2567.
-/
theorem total_value {V : ℝ} (h1 : 0.07 * (V - 1000) = 109.90) : V = 2567 :=
by
  sorry

end total_value_l2173_217350


namespace triangle_area_of_integral_sides_with_perimeter_8_l2173_217365

theorem triangle_area_of_integral_sides_with_perimeter_8 :
  ∃ (a b c : ℕ), a + b + c = 8 ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ 
  ∃ (area : ℝ), area = 2 * Real.sqrt 2 := by
  sorry

end triangle_area_of_integral_sides_with_perimeter_8_l2173_217365


namespace find_g3_l2173_217344

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^2 + b * x^3 + c * x + d

theorem find_g3 (a b c d : ℝ) (h : g (-3) a b c d = 2) : g 3 a b c d = 0 := 
by 
  sorry

end find_g3_l2173_217344


namespace solve_trig_equation_proof_l2173_217360

noncomputable def solve_trig_equation (θ : ℝ) : Prop :=
  2 * Real.cos θ ^ 2 - 5 * Real.cos θ + 2 = 0 ∧ (θ = 60 / 180 * Real.pi)

theorem solve_trig_equation_proof (θ : ℝ) :
  solve_trig_equation θ :=
sorry

end solve_trig_equation_proof_l2173_217360


namespace total_interest_calculation_l2173_217346

-- Define the total investment
def total_investment : ℝ := 20000

-- Define the fractional part of investment at 9 percent rate
def fraction_higher_rate : ℝ := 0.55

-- Define the investment amounts based on the fractional part
def investment_higher_rate : ℝ := fraction_higher_rate * total_investment
def investment_lower_rate : ℝ := total_investment - investment_higher_rate

-- Define interest rates
def rate_lower : ℝ := 0.06
def rate_higher : ℝ := 0.09

-- Define time period (in years)
def time_period : ℝ := 1

-- Define interest calculations
def interest_lower : ℝ := investment_lower_rate * rate_lower * time_period
def interest_higher : ℝ := investment_higher_rate * rate_higher * time_period

-- Define the total interest
def total_interest : ℝ := interest_lower + interest_higher

-- Theorem stating the total interest earned
theorem total_interest_calculation : total_interest = 1530 := by
  -- skip proof using sorry
  sorry

end total_interest_calculation_l2173_217346


namespace sum_powers_divisible_by_13_l2173_217308

-- Statement of the problem in Lean
theorem sum_powers_divisible_by_13 (a b p : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : p = 13) :
  (a^1974 + b^1974) % p = 0 := 
by
  sorry

end sum_powers_divisible_by_13_l2173_217308


namespace square_area_from_diagonal_l2173_217380

theorem square_area_from_diagonal :
  ∀ (d : ℝ), d = 10 * Real.sqrt 2 → (d / Real.sqrt 2) ^ 2 = 100 :=
by
  intros d hd
  sorry -- Skipping the proof

end square_area_from_diagonal_l2173_217380


namespace no_pairs_satisfy_equation_l2173_217387

theorem no_pairs_satisfy_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a^2 + 1 / b^2 = 1 / (a^2 + b^2)) → False :=
by
  sorry

end no_pairs_satisfy_equation_l2173_217387


namespace trig_expr_value_l2173_217398

theorem trig_expr_value :
  (Real.cos (7 * Real.pi / 24)) ^ 4 +
  (Real.sin (11 * Real.pi / 24)) ^ 4 +
  (Real.sin (17 * Real.pi / 24)) ^ 4 +
  (Real.cos (13 * Real.pi / 24)) ^ 4 = 3 / 2 :=
by
  sorry

end trig_expr_value_l2173_217398


namespace B_age_l2173_217399

-- Define the conditions
variables (x y : ℕ)
variable (current_year : ℕ)
axiom h1 : 10 * x + y + 4 = 43
axiom reference_year : current_year = 1955

-- Define the relationship between the digit equation and the year
def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

-- Birth year calculation
def age (current_year birth_year : ℕ) : ℕ := current_year - birth_year

-- Final theorem: Age of B
theorem B_age (x y : ℕ) (current_year : ℕ) (h1 : 10 * x  + y + 4 = 43) (reference_year : current_year = 1955) :
  age current_year (birth_year x y) = 16 :=
by
  sorry

end B_age_l2173_217399


namespace problem_220_l2173_217324

variables (x y : ℝ)

theorem problem_220 (h1 : x + y = 10) (h2 : (x * y) / (x^2) = -3 / 2) :
  x = -20 ∧ y = 30 :=
by
  sorry

end problem_220_l2173_217324


namespace macaroon_weight_l2173_217368

theorem macaroon_weight (bakes : ℕ) (packs : ℕ) (bags_after_eat : ℕ) (remaining_weight : ℕ) (macaroons_per_bag : ℕ) (weight_per_bag : ℕ)
  (H1 : bakes = 12) 
  (H2 : packs = 4)
  (H3 : bags_after_eat = 3)
  (H4 : remaining_weight = 45)
  (H5 : macaroons_per_bag = bakes / packs) 
  (H6 : weight_per_bag = remaining_weight / bags_after_eat) :
  ∀ (weight_per_macaroon : ℕ), weight_per_macaroon = weight_per_bag / macaroons_per_bag → weight_per_macaroon = 5 :=
by
  sorry -- Proof will come here, not required as per instructions

end macaroon_weight_l2173_217368


namespace cody_steps_away_from_goal_l2173_217335

def steps_in_week (daily_steps : ℕ) : ℕ :=
  daily_steps * 7

def total_steps_in_4_weeks (initial_steps : ℕ) : ℕ :=
  steps_in_week initial_steps +
  steps_in_week (initial_steps + 1000) +
  steps_in_week (initial_steps + 2000) +
  steps_in_week (initial_steps + 3000)

theorem cody_steps_away_from_goal :
  let goal := 100000
  let initial_daily_steps := 1000
  let total_steps := total_steps_in_4_weeks initial_daily_steps
  goal - total_steps = 30000 :=
by
  sorry

end cody_steps_away_from_goal_l2173_217335


namespace sandy_goal_hours_l2173_217330

def goal_liters := 3 -- The goal in liters
def liters_to_milliliters := 1000 -- Conversion rate from liters to milliliters
def goal_milliliters := goal_liters * liters_to_milliliters -- Total milliliters to drink
def drink_rate_milliliters := 500 -- Milliliters drunk every interval
def interval_hours := 2 -- Interval in hours

def sets_to_goal := goal_milliliters / drink_rate_milliliters -- The number of drink sets to reach the goal
def total_hours := sets_to_goal * interval_hours -- Total time in hours to reach the goal

theorem sandy_goal_hours : total_hours = 12 := by
  -- Proof steps would go here
  sorry

end sandy_goal_hours_l2173_217330


namespace person_b_worked_alone_days_l2173_217337

theorem person_b_worked_alone_days :
  ∀ (x : ℕ), 
  (x / 10 + (12 - x) / 20 = 1) → x = 8 :=
by
  sorry

end person_b_worked_alone_days_l2173_217337


namespace find_root_D_l2173_217323

/-- Given C and D are roots of the polynomial k x^2 + 2 x + 5 = 0, 
    and k = -1/4 and C = 10, then D must be -2. -/
theorem find_root_D 
  (k : ℚ) (C D : ℚ)
  (h1 : k = -1/4)
  (h2 : C = 10)
  (h3 : C^2 * k + 2 * C + 5 = 0)
  (h4 : D^2 * k + 2 * D + 5 = 0) : 
  D = -2 :=
by
  sorry

end find_root_D_l2173_217323


namespace given_statements_l2173_217369

def addition_is_associative (x y z : ℝ) : Prop := (x + y) + z = x + (y + z)

def averaging_is_commutative (x y : ℝ) : Prop := (x + y) / 2 = (y + x) / 2

def addition_distributes_over_averaging (x y z : ℝ) : Prop := 
  x + (y + z) / 2 = (x + y + x + z) / 2

def averaging_distributes_over_addition (x y z : ℝ) : Prop := 
  (x + (y + z)) / 2 = ((x + y) / 2) + ((x + z) / 2)

def averaging_has_identity_element (x e : ℝ) : Prop := 
  (x + e) / 2 = x

theorem given_statements (x y z e : ℝ) :
  addition_is_associative x y z ∧ 
  averaging_is_commutative x y ∧ 
  addition_distributes_over_averaging x y z ∧ 
  ¬averaging_distributes_over_addition x y z ∧ 
  ¬∃ e, averaging_has_identity_element x e :=
by
  sorry

end given_statements_l2173_217369


namespace bridge_length_proof_l2173_217340

open Real

def train_length : ℝ := 100
def train_speed_kmh : ℝ := 45
def crossing_time_s: ℝ := 30

noncomputable def bridge_length : ℝ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem bridge_length_proof : bridge_length = 275 := 
by
  sorry

end bridge_length_proof_l2173_217340


namespace y_star_definition_l2173_217342

def y_star (y : Real) : Real := y - 1

theorem y_star_definition (y : Real) : (5 : Real) - y_star 5 = 1 :=
  by sorry

end y_star_definition_l2173_217342


namespace linear_function_quadrants_l2173_217300

theorem linear_function_quadrants
  (k : ℝ) (h₀ : k ≠ 0) (h₁ : ∀ x : ℝ, x > 0 → k*x < 0) :
  (∃ x > 0, 2*x + k > 0) ∧
  (∃ x > 0, 2*x + k < 0) ∧
  (∃ x < 0, 2*x + k < 0) :=
  by
  sorry

end linear_function_quadrants_l2173_217300


namespace ratio_of_daily_wages_l2173_217318

-- Definitions for daily wages and conditions
def daily_wage_man : ℝ := sorry
def daily_wage_woman : ℝ := sorry

axiom condition_for_men (M : ℝ) : 16 * M * 25 = 14400
axiom condition_for_women (W : ℝ) : 40 * W * 30 = 21600

-- Theorem statement for the ratio of daily wages
theorem ratio_of_daily_wages 
  (M : ℝ) (W : ℝ) 
  (hM : 16 * M * 25 = 14400) 
  (hW : 40 * W * 30 = 21600) :
  M / W = 2 := 
  sorry

end ratio_of_daily_wages_l2173_217318


namespace factorize_expression_l2173_217345

-- The problem is about factorizing the expression x^3y - xy
theorem factorize_expression (x y : ℝ) : x^3 * y - x * y = x * y * (x - 1) * (x + 1) := 
by sorry

end factorize_expression_l2173_217345


namespace actual_plot_area_in_acres_l2173_217317

-- Condition Definitions
def base_cm : ℝ := 8
def height_cm : ℝ := 12
def scale_cm_to_miles : ℝ := 1  -- 1 cm = 1 mile
def miles_to_acres : ℝ := 320  -- 1 square mile = 320 acres

-- Theorem Statement
theorem actual_plot_area_in_acres (A : ℝ) :
  A = 15360 :=
by
  sorry

end actual_plot_area_in_acres_l2173_217317


namespace number_of_students_l2173_217302

theorem number_of_students 
  (n : ℕ)
  (h1: 108 - 36 = 72)
  (h2: ∀ n > 0, 108 / n - 72 / n = 3) :
  n = 12 :=
sorry

end number_of_students_l2173_217302


namespace inequality_problem_l2173_217358

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 / (b * (a + b)) + 2 / (c * (b + c)) + 2 / (a * (c + a))) ≥ (27 / (a + b + c)^2) :=
by
  sorry

end inequality_problem_l2173_217358


namespace plane_triangle_coverage_l2173_217394

noncomputable def percentage_triangles_covered (a : ℝ) : ℝ :=
  let total_area := (4 * a) ^ 2
  let triangle_area := 10 * (1 / 2 * a^2)
  (triangle_area / total_area) * 100

theorem plane_triangle_coverage (a : ℝ) :
  abs (percentage_triangles_covered a - 31.25) < 0.75 :=
  sorry

end plane_triangle_coverage_l2173_217394


namespace minimum_value_analysis_l2173_217391

theorem minimum_value_analysis
  (a : ℝ) (m n : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : 2 * m + n = 2)
  (h4 : m > 0)
  (h5 : n > 0) :
  (2 / m + 1 / n) ≥ 9 / 2 :=
sorry

end minimum_value_analysis_l2173_217391


namespace gcd_f_50_51_l2173_217381

-- Define f(x)
def f (x : ℤ) : ℤ := x^3 - x^2 + 2 * x + 2000

-- State the problem: Prove gcd(f(50), f(51)) = 8
theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 8 := by
  sorry

end gcd_f_50_51_l2173_217381


namespace pen_ratio_l2173_217311

theorem pen_ratio 
  (Dorothy_pens Julia_pens Robert_pens : ℕ)
  (pen_cost total_cost : ℚ)
  (h1 : Dorothy_pens = Julia_pens / 2)
  (h2 : Robert_pens = 4)
  (h3 : pen_cost = 1.5)
  (h4 : total_cost = 33)
  (h5 : total_cost / pen_cost = Dorothy_pens + Julia_pens + Robert_pens) :
  (Julia_pens / Robert_pens : ℚ) = 3 :=
  sorry

end pen_ratio_l2173_217311
