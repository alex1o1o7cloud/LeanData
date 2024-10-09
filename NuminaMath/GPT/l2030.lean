import Mathlib

namespace jack_sugar_l2030_203028

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l2030_203028


namespace arithmetic_geometric_seq_l2030_203077

theorem arithmetic_geometric_seq (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_diff : d = 2)
  (h_geom : (a 1)^2 = a 0 * (a 0 + 6)) :
  a 1 = -6 :=
by 
  sorry

end arithmetic_geometric_seq_l2030_203077


namespace total_heads_l2030_203071

/-- There are H hens and C cows. Each hen has 1 head and 2 feet, and each cow has 1 head and 4 feet.
Given that the total number of feet is 140 and there are 26 hens, prove that the total number of heads is 48. -/
theorem total_heads (H C : ℕ) (h1 : 2 * H + 4 * C = 140) (h2 : H = 26) : H + C = 48 := by
  sorry

end total_heads_l2030_203071


namespace classroom_needs_more_money_l2030_203003

theorem classroom_needs_more_money 
    (goal : ℕ) 
    (raised_from_two_families : ℕ) 
    (raised_from_eight_families : ℕ) 
    (raised_from_ten_families : ℕ) 
    (H : goal = 200) 
    (H1 : raised_from_two_families = 2 * 20) 
    (H2 : raised_from_eight_families = 8 * 10) 
    (H3 : raised_from_ten_families = 10 * 5) 
    (total_raised : ℕ := raised_from_two_families + raised_from_eight_families + raised_from_ten_families) : 
    (goal - total_raised) = 30 := 
by 
  sorry

end classroom_needs_more_money_l2030_203003


namespace middle_joints_capacity_l2030_203087

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def bamboo_tube_capacity (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 4.5 ∧ a 6 + a 7 + a 8 = 2.5 ∧ arithmetic_seq a (a 1 - a 0)

theorem middle_joints_capacity (a : ℕ → ℝ) (d : ℝ) (h : bamboo_tube_capacity a) : 
  a 3 + a 4 + a 5 = 3.5 :=
by
  sorry

end middle_joints_capacity_l2030_203087


namespace multiplier_for_ab_to_equal_1800_l2030_203034

variable (a b m : ℝ)
variable (h1 : 4 * a = 30)
variable (h2 : 5 * b = 30)
variable (h3 : a * b = 45)
variable (h4 : m * (a * b) = 1800)

theorem multiplier_for_ab_to_equal_1800 (h1 : 4 * a = 30) (h2 : 5 * b = 30) (h3 : a * b = 45) (h4 : m * (a * b) = 1800) :
  m = 40 :=
sorry

end multiplier_for_ab_to_equal_1800_l2030_203034


namespace if_a_gt_abs_b_then_a2_gt_b2_l2030_203049

theorem if_a_gt_abs_b_then_a2_gt_b2 (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by sorry

end if_a_gt_abs_b_then_a2_gt_b2_l2030_203049


namespace length_of_short_pieces_l2030_203061

def total_length : ℕ := 27
def long_piece_length : ℕ := 4
def number_of_long_pieces : ℕ := total_length / long_piece_length
def remainder_length : ℕ := total_length % long_piece_length
def number_of_short_pieces : ℕ := 3

theorem length_of_short_pieces (h1 : remainder_length = 3) : (remainder_length / number_of_short_pieces) = 1 :=
by
  sorry

end length_of_short_pieces_l2030_203061


namespace count_ways_to_write_2010_l2030_203089

theorem count_ways_to_write_2010 : ∃ N : ℕ, 
  (∀ (a_3 a_2 a_1 a_0 : ℕ), a_0 ≤ 99 ∧ a_1 ≤ 99 ∧ a_2 ≤ 99 ∧ a_3 ≤ 99 → 
    2010 = a_3 * 10^3 + a_2 * 10^2 + a_1 * 10 + a_0) ∧ 
    N = 202 :=
sorry

end count_ways_to_write_2010_l2030_203089


namespace average_age_of_coaches_l2030_203023

variables 
  (total_members : ℕ) (avg_age_total : ℕ) 
  (num_girls : ℕ) (num_boys : ℕ) (num_coaches : ℕ) 
  (avg_age_girls : ℕ) (avg_age_boys : ℕ)

theorem average_age_of_coaches 
  (h1 : total_members = 50) 
  (h2 : avg_age_total = 18)
  (h3 : num_girls = 25) 
  (h4 : num_boys = 20) 
  (h5 : num_coaches = 5)
  (h6 : avg_age_girls = 16)
  (h7 : avg_age_boys = 17) : 
  (900 - (num_girls * avg_age_girls + num_boys * avg_age_boys)) / num_coaches = 32 :=
by
  sorry

end average_age_of_coaches_l2030_203023


namespace triangle_angle_bisectors_l2030_203080

theorem triangle_angle_bisectors (α β γ : ℝ) 
  (h1 : α + β + γ = 180)
  (h2 : α = 100) 
  (h3 : β = 30) 
  (h4 : γ = 50) :
  ∃ α' β' γ', α' = 40 ∧ β' = 65 ∧ γ' = 75 :=
sorry

end triangle_angle_bisectors_l2030_203080


namespace parallelogram_sticks_l2030_203047

theorem parallelogram_sticks (a : ℕ) (h₁ : ∃ l₁ l₂, l₁ = 5 ∧ l₂ = 5 ∧ 
                                (l₁ = l₂) ∧ (a = 7)) : a = 7 :=
by sorry

end parallelogram_sticks_l2030_203047


namespace area_of_triangle_l2030_203037

theorem area_of_triangle {A B C : ℝ} {a b c : ℝ}
  (h1 : b = 2) (h2 : c = 2 * Real.sqrt 2) (h3 : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - C - (1 / 2 * Real.pi / 3)) = Real.sqrt 3 + 1 :=
by
  sorry

end area_of_triangle_l2030_203037


namespace isosceles_triangle_vertex_angle_l2030_203006

theorem isosceles_triangle_vertex_angle (B : ℝ) (V : ℝ) (h1 : B = 70) (h2 : B = B) (h3 : V + 2 * B = 180) : V = 40 ∨ V = 70 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l2030_203006


namespace letters_per_large_envelope_l2030_203033

theorem letters_per_large_envelope
  (total_letters : ℕ)
  (small_envelope_letters : ℕ)
  (large_envelopes : ℕ)
  (large_envelopes_count : ℕ)
  (h1 : total_letters = 80)
  (h2 : small_envelope_letters = 20)
  (h3 : large_envelopes_count = 30)
  (h4 : total_letters - small_envelope_letters = large_envelopes)
  : large_envelopes / large_envelopes_count = 2 :=
by
  sorry

end letters_per_large_envelope_l2030_203033


namespace quadratic_monotonic_range_l2030_203095

theorem quadratic_monotonic_range {t : ℝ} (h : ∀ x1 x2 : ℝ, (1 < x1 ∧ x1 < 3) → (1 < x2 ∧ x2 < 3) → x1 < x2 → (x1^2 - 2 * t * x1 + 1 ≤ x2^2 - 2 * t * x2 + 1)) : 
  t ≤ 1 ∨ t ≥ 3 :=
by
  sorry

end quadratic_monotonic_range_l2030_203095


namespace tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l2030_203054

open Real

-- Part 1
theorem tan_x_min_x_div_x_min_sin_x_gt_two (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) :
  (tan x - x) / (x - sin x) > 2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x + 2 * sin x - a * x > 0) → a ≤ 3 :=
sorry

end tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l2030_203054


namespace similar_triangle_longest_side_length_l2030_203002

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end similar_triangle_longest_side_length_l2030_203002


namespace tan_pi_div_4_sub_theta_l2030_203035

theorem tan_pi_div_4_sub_theta (theta : ℝ) (h : Real.tan theta = 1 / 2) : 
  Real.tan (π / 4 - theta) = 1 / 3 := 
sorry

end tan_pi_div_4_sub_theta_l2030_203035


namespace combined_shoe_size_l2030_203091

-- Definitions based on conditions
def Jasmine_size : ℕ := 7
def Alexa_size : ℕ := 2 * Jasmine_size
def Clara_size : ℕ := 3 * Jasmine_size

-- Statement to prove
theorem combined_shoe_size : Jasmine_size + Alexa_size + Clara_size = 42 :=
by
  sorry

end combined_shoe_size_l2030_203091


namespace sufficient_but_not_necessary_l2030_203069

def quadratic_real_roots (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 - 2 * x + a = 0)

theorem sufficient_but_not_necessary (a : ℝ) :
  (quadratic_real_roots 1) ∧ (∀ a > 1, ¬ quadratic_real_roots a) :=
sorry

end sufficient_but_not_necessary_l2030_203069


namespace total_cost_correct_l2030_203048

def shirt_price : ℕ := 5
def hat_price : ℕ := 4
def jeans_price : ℕ := 10
def jacket_price : ℕ := 20
def shoes_price : ℕ := 15

def num_shirts : ℕ := 4
def num_jeans : ℕ := 3
def num_hats : ℕ := 4
def num_jackets : ℕ := 3
def num_shoes : ℕ := 2

def third_jacket_discount : ℕ := jacket_price / 2
def discount_per_two_shirts : ℕ := 2
def free_hat : ℕ := if num_jeans ≥ 3 then 1 else 0
def shoes_discount : ℕ := (num_shirts / 2) * discount_per_two_shirts

def total_cost : ℕ :=
  (num_shirts * shirt_price) +
  (num_jeans * jeans_price) +
  ((num_hats - free_hat) * hat_price) +
  ((num_jackets - 1) * jacket_price + third_jacket_discount) +
  (num_shoes * shoes_price - shoes_discount)

theorem total_cost_correct : total_cost = 138 := by
  sorry

end total_cost_correct_l2030_203048


namespace treasure_chest_age_l2030_203066

theorem treasure_chest_age (n : ℕ) (h : n = 3 * 8^2 + 4 * 8^1 + 7 * 8^0) : n = 231 :=
by
  sorry

end treasure_chest_age_l2030_203066


namespace range_of_a_l2030_203019

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp (x-2) + (1/3) * x^3 - (3/2) * x^2 + 2 * x - Real.log (x-1) + a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, (1 < x → f a x = y) ↔ ∃ z : ℝ, 1 < z → f a (f a z) = y) →
  a ≤ 1/3 :=
sorry

end range_of_a_l2030_203019


namespace number_of_terms_arithmetic_sequence_l2030_203031

theorem number_of_terms_arithmetic_sequence
  (a₁ d n : ℝ)
  (h1 : a₁ + (a₁ + d) + (a₁ + 2 * d) = 34)
  (h2 : (a₁ + (n-3) * d) + (a₁ + (n-2) * d) + (a₁ + (n-1) * d) = 146)
  (h3 : n / 2 * (2 * a₁ + (n-1) * d) = 390) :
  n = 11 :=
by sorry

end number_of_terms_arithmetic_sequence_l2030_203031


namespace set_representation_l2030_203050

def is_nat_star (n : ℕ) : Prop := n > 0
def satisfies_eqn (x y : ℕ) : Prop := y = 6 / (x + 3)

theorem set_representation :
  {p : ℕ × ℕ | is_nat_star p.fst ∧ is_nat_star p.snd ∧ satisfies_eqn p.fst p.snd } = { (3, 1) } :=
by
  sorry

end set_representation_l2030_203050


namespace unique_intersection_of_A_and_B_l2030_203079

-- Define the sets A and B with their respective conditions
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = 4 }

def B (r : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - 3)^2 + (y - 4)^2 = r^2 ∧ r > 0 }

-- Define the main theorem statement
theorem unique_intersection_of_A_and_B (r : ℝ) (h : r > 0) : 
  (∃! p, p ∈ A ∧ p ∈ B r) ↔ r = 3 ∨ r = 7 :=
sorry

end unique_intersection_of_A_and_B_l2030_203079


namespace a_five_minus_a_divisible_by_five_l2030_203046

theorem a_five_minus_a_divisible_by_five (a : ℤ) : 5 ∣ (a^5 - a) :=
by
  -- proof steps
  sorry

end a_five_minus_a_divisible_by_five_l2030_203046


namespace area_of_trapezoid_RSQT_l2030_203099

theorem area_of_trapezoid_RSQT
  (PR PQ : ℝ)
  (PR_eq_PQ : PR = PQ)
  (small_triangle_area : ℝ)
  (total_area : ℝ)
  (num_of_small_triangles : ℕ)
  (num_of_triangles_in_trapezoid : ℕ)
  (area_of_trapezoid : ℝ)
  (is_isosceles_triangle : ∀ (a b c : ℝ), a = b → b = c → a = c)
  (are_similar_triangles : ∀ {A B C D E F : ℝ}, 
    A / B = D / E → A / C = D / F → B / A = E / D → C / A = F / D)
  (smallest_triangle_areas : ∀ {n : ℕ}, n = 9 → small_triangle_area = 2 → num_of_small_triangles = 9)
  (triangle_total_area : ∀ (a : ℝ), a = 72 → total_area = 72)
  (contains_3_small_triangles : ∀ (n : ℕ), n = 3 → num_of_triangles_in_trapezoid = 3)
  (parallel_ST_to_PQ : ∀ {x y z : ℝ}, x = z → y = z → x = y)
  : area_of_trapezoid = 39 :=
sorry

end area_of_trapezoid_RSQT_l2030_203099


namespace solve_fraction_zero_l2030_203082

theorem solve_fraction_zero (x : ℕ) (h : x ≠ 0) (h_eq : (x - 1) / x = 0) : x = 1 := by 
  sorry

end solve_fraction_zero_l2030_203082


namespace arrangement_plans_count_l2030_203009

noncomputable def number_of_arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
if num_teachers = 2 ∧ num_students = 4 then 12 else 0

theorem arrangement_plans_count :
  number_of_arrangement_plans 2 4 = 12 :=
by 
  sorry

end arrangement_plans_count_l2030_203009


namespace star_angle_sum_l2030_203051

-- Define variables and angles for Petya's and Vasya's stars.
variables {α β γ δ ε : ℝ}
variables {φ χ ψ ω : ℝ}
variables {a b c d e : ℝ}

-- Conditions
def all_acute (a b c d e : ℝ) : Prop := a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90 ∧ e < 90
def one_obtuse (a b c d e : ℝ) : Prop := (a > 90 ∨ b > 90 ∨ c > 90 ∨ d > 90 ∨ e > 90)

-- Question: Prove the sum of the angles at the vertices of both stars is equal
theorem star_angle_sum : all_acute α β γ δ ε → one_obtuse φ χ ψ ω α → 
  α + β + γ + δ + ε = φ + χ + ψ + ω + α := 
by sorry

end star_angle_sum_l2030_203051


namespace cannot_afford_laptop_l2030_203090

theorem cannot_afford_laptop (P_0 : ℝ) : 56358 < P_0 * (1.06)^2 :=
by
  sorry

end cannot_afford_laptop_l2030_203090


namespace johnny_fishes_l2030_203045

theorem johnny_fishes (total_fishes sony_multiple j : ℕ) (h1 : total_fishes = 120) (h2 : sony_multiple = 7) (h3 : total_fishes = j + sony_multiple * j) : j = 15 :=
by sorry

end johnny_fishes_l2030_203045


namespace paving_cost_l2030_203065

def length : Real := 5.5
def width : Real := 3.75
def rate : Real := 700
def area : Real := length * width
def cost : Real := area * rate

theorem paving_cost :
  cost = 14437.50 :=
by
  -- Proof steps go here
  sorry

end paving_cost_l2030_203065


namespace range_of_m_l2030_203000

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) (h : ¬ (p m ∨ q m)) : m ≥ 2 :=
by
  sorry

end range_of_m_l2030_203000


namespace base_eight_conversion_l2030_203060

theorem base_eight_conversion :
  (1 * 8^2 + 3 * 8^1 + 2 * 8^0 = 90) := by
  sorry

end base_eight_conversion_l2030_203060


namespace find_order_amount_l2030_203097

noncomputable def unit_price : ℝ := 100

def discount_rate (x : ℕ) : ℝ :=
  if x < 250 then 0
  else if x < 500 then 0.05
  else if x < 1000 then 0.10
  else 0.15

theorem find_order_amount (T : ℝ) (x : ℕ)
    (hx : x = 980) (hT : T = 88200) :
  T = unit_price * x * (1 - discount_rate x) :=
by
  rw [hx, hT]
  sorry

end find_order_amount_l2030_203097


namespace robert_ate_7_chocolates_l2030_203064

-- Define the number of chocolates Nickel ate
def nickel_chocolates : ℕ := 5

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := nickel_chocolates + 2

-- Prove that Robert ate 7 chocolates
theorem robert_ate_7_chocolates : robert_chocolates = 7 := by
    sorry

end robert_ate_7_chocolates_l2030_203064


namespace average_marks_l2030_203032

-- Conditions
def marks_english : ℕ := 73
def marks_mathematics : ℕ := 69
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 64
def marks_biology : ℕ := 82
def number_of_subjects : ℕ := 5

-- Problem Statement
theorem average_marks :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / number_of_subjects = 76 :=
by
  sorry

end average_marks_l2030_203032


namespace simplify_arithmetic_expr1_simplify_arithmetic_expr2_l2030_203015

-- Problem 1 Statement
theorem simplify_arithmetic_expr1 (x y : ℝ) : 
  (x - 3 * y) - (y - 2 * x) = 3 * x - 4 * y :=
sorry

-- Problem 2 Statement
theorem simplify_arithmetic_expr2 (a b : ℝ) : 
  5 * a * b^2 - 3 * (2 * a^2 * b - 2 * (a^2 * b - 2 * a * b^2)) = -7 * a * b^2 :=
sorry

end simplify_arithmetic_expr1_simplify_arithmetic_expr2_l2030_203015


namespace bob_weight_l2030_203036

variable (j b : ℕ)

theorem bob_weight :
  j + b = 210 →
  b - j = b / 3 →
  b = 126 :=
by
  intros h1 h2
  sorry

end bob_weight_l2030_203036


namespace cost_price_of_article_l2030_203074

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 :=
by sorry

end cost_price_of_article_l2030_203074


namespace car_payment_months_l2030_203008

theorem car_payment_months 
    (total_price : ℕ) 
    (initial_payment : ℕ)
    (monthly_payment : ℕ) 
    (h_total_price : total_price = 13380) 
    (h_initial_payment : initial_payment = 5400) 
    (h_monthly_payment : monthly_payment = 420) 
    : total_price - initial_payment = 7980 
    ∧ (total_price - initial_payment) / monthly_payment = 19 := 
by 
  sorry

end car_payment_months_l2030_203008


namespace factor_expression_l2030_203038

noncomputable def numerator (a b c : ℝ) : ℝ := 
(|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3)

noncomputable def denominator (a b c : ℝ) : ℝ := 
(|a + b|^3 + |b + c|^3 + |c + a|^3)

theorem factor_expression (a b c : ℝ) : 
  (denominator a b c) ≠ 0 → 
  (numerator a b c) / (denominator a b c) = 1 :=
by
  sorry

end factor_expression_l2030_203038


namespace area_of_triangle_is_18_l2030_203001

-- Define the vertices of the triangle
def point1 : ℝ × ℝ := (1, 4)
def point2 : ℝ × ℝ := (7, 4)
def point3 : ℝ × ℝ := (1, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

-- Statement of the problem
theorem area_of_triangle_is_18 :
  triangle_area point1 point2 point3 = 18 :=
by
  -- skipping the proof
  sorry

end area_of_triangle_is_18_l2030_203001


namespace pounds_of_fish_to_ship_l2030_203007

theorem pounds_of_fish_to_ship (crates_weight : ℕ) (cost_per_crate : ℝ) (total_cost : ℝ) :
  crates_weight = 30 → cost_per_crate = 1.5 → total_cost = 27 → 
  (total_cost / cost_per_crate) * crates_weight = 540 :=
by
  intros h1 h2 h3
  sorry

end pounds_of_fish_to_ship_l2030_203007


namespace car_second_hour_speed_l2030_203075

theorem car_second_hour_speed (x : ℝ) 
  (first_hour_speed : ℝ := 20)
  (average_speed : ℝ := 40) 
  (total_time : ℝ := 2)
  (total_distance : ℝ := first_hour_speed + x) 
  : total_distance / total_time = average_speed → x = 60 :=
by
  intro h
  sorry

end car_second_hour_speed_l2030_203075


namespace correct_option_l2030_203026

-- Defining the conditions for each option
def optionA (m n : ℝ) : Prop := (m / n)^7 = m^7 * n^(1/7)
def optionB : Prop := (4)^(4/12) = (-3)^(1/3)
def optionC (x y : ℝ) : Prop := ((x^3 + y^3)^(1/4)) = (x + y)^(3/4)
def optionD : Prop := (9)^(1/6) = 3^(1/3)

-- Asserting that option D is correct
theorem correct_option : optionD :=
by
  sorry

end correct_option_l2030_203026


namespace sum_of_solutions_l2030_203012

theorem sum_of_solutions (x : ℝ) :
  (4 * x + 6) * (3 * x - 12) = 0 → (x = -3 / 2 ∨ x = 4) →
  (-3 / 2 + 4) = 5 / 2 :=
by
  intros Hsol Hsols
  sorry

end sum_of_solutions_l2030_203012


namespace find_number_of_3cm_books_l2030_203025

-- Define the conditions
def total_books : ℕ := 46
def total_thickness : ℕ := 200
def thickness_3cm : ℕ := 3
def thickness_5cm : ℕ := 5

-- Let x be the number of 3 cm thick books, y be the number of 5 cm thick books
variable (x y : ℕ)

-- Define the system of equations based on the given conditions
axiom total_books_eq : x + y = total_books
axiom total_thickness_eq : thickness_3cm * x + thickness_5cm * y = total_thickness

-- The theorem to prove: x = 15
theorem find_number_of_3cm_books : x = 15 :=
by
  sorry

end find_number_of_3cm_books_l2030_203025


namespace find_base_l2030_203081

-- Definitions based on the conditions of the problem
def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) := ∃ m : ℕ, m * m * m = n
def is_perfect_fourth (n : ℕ) := ∃ m : ℕ, m * m * m * m = n

-- Define the number A in terms of base a
def A (a : ℕ) : ℕ := 4 * a * a + 4 * a + 1

-- Problem statement: find a base a > 4 such that A is both a perfect cube and a perfect fourth power
theorem find_base (a : ℕ)
  (ha : a > 4)
  (h_square : is_perfect_square (A a)) :
  is_perfect_cube (A a) ∧ is_perfect_fourth (A a) :=
sorry

end find_base_l2030_203081


namespace tens_digit_of_23_pow_1987_l2030_203076

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l2030_203076


namespace max_chocolates_l2030_203073

theorem max_chocolates (b c k : ℕ) (h1 : b + c = 36) (h2 : c = k * b) (h3 : k > 0) : b ≤ 18 :=
sorry

end max_chocolates_l2030_203073


namespace coloring_circle_impossible_l2030_203044

theorem coloring_circle_impossible (n : ℕ) (h : n = 2022) : 
  ¬ (∃ (coloring : ℕ → ℕ), (∀ i, 0 ≤ coloring i ∧ coloring i < 3) ∧ (∀ i, coloring ((i + 1) % n) ≠ coloring i)) :=
sorry

end coloring_circle_impossible_l2030_203044


namespace pairs_satisfy_inequality_l2030_203024

section inequality_problem

variables (a b : ℝ)

-- Conditions
variable (hb1 : b ≠ -1)
variable (hb2 : b ≠ 0)

-- Inequalities to check
def inequality (a b : ℝ) : Prop :=
  (1 + a) ^ 2 / (1 + b) ≤ 1 + a ^ 2 / b

-- Main theorem
theorem pairs_satisfy_inequality :
  (b > 0 ∨ b < -1 → ∀ a, a ≠ b → inequality a b) ∧
  (∀ a, a ≠ -1 ∧ a ≠ 0 → inequality a a) :=
by
  sorry

end inequality_problem

end pairs_satisfy_inequality_l2030_203024


namespace sphere_radius_ratio_l2030_203086

theorem sphere_radius_ratio (R r : ℝ) (h₁ : (4 / 3) * Real.pi * R ^ 3 = 450 * Real.pi) (h₂ : (4 / 3) * Real.pi * r ^ 3 = 0.25 * 450 * Real.pi) :
  r / R = 1 / 2 :=
sorry

end sphere_radius_ratio_l2030_203086


namespace profit_growth_rate_and_expected_profit_l2030_203004

theorem profit_growth_rate_and_expected_profit
  (profit_April : ℕ)
  (profit_June : ℕ)
  (months : ℕ)
  (avg_growth_rate : ℝ)
  (profit_July : ℕ) :
  profit_April = 6000 ∧ profit_June = 7260 ∧ months = 2 ∧ 
  (profit_April : ℝ) * (1 + avg_growth_rate)^months = profit_June →
  avg_growth_rate = 0.1 ∧ 
  (profit_June : ℝ) * (1 + avg_growth_rate) = profit_July →
  profit_July = 7986 := 
sorry

end profit_growth_rate_and_expected_profit_l2030_203004


namespace assertion1_false_assertion2_true_assertion3_false_assertion4_false_l2030_203068

section

-- Assertion 1: ∀ x ∈ ℝ, x ≥ 1 is false
theorem assertion1_false : ¬(∀ x : ℝ, x ≥ 1) := 
sorry

-- Assertion 2: ∃ x ∈ ℕ, x ∈ ℝ is true
theorem assertion2_true : ∃ x : ℕ, (x : ℝ) = x := 
sorry

-- Assertion 3: ∀ x ∈ ℝ, x > 2 → x ≥ 3 is false
theorem assertion3_false : ¬(∀ x : ℝ, x > 2 → x ≥ 3) := 
sorry

-- Assertion 4: ∃ n ∈ ℤ, ∀ x ∈ ℝ, n ≤ x < n + 1 is false
theorem assertion4_false : ¬(∃ n : ℤ, ∀ x : ℝ, n ≤ x ∧ x < n + 1) := 
sorry

end

end assertion1_false_assertion2_true_assertion3_false_assertion4_false_l2030_203068


namespace factor_product_l2030_203072

theorem factor_product : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end factor_product_l2030_203072


namespace intersection_of_sets_l2030_203085

open Set Real

theorem intersection_of_sets :
  let M := {x : ℝ | x ≤ 4}
  let N := {x : ℝ | x > 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_of_sets_l2030_203085


namespace num_balls_box_l2030_203005

theorem num_balls_box (n : ℕ) (balls : Fin n → ℕ) (red blue : Fin n → Prop)
  (h_colors : ∀ i, red i ∨ blue i)
  (h_constraints : ∀ i j k,  red i ∨ red j ∨ red k ∧ blue i ∨ blue j ∨ blue k) : 
  n = 4 := 
sorry

end num_balls_box_l2030_203005


namespace min_sum_y1_y2_l2030_203088

theorem min_sum_y1_y2 (y : ℕ → ℕ) (h_seq : ∀ n ≥ 1, y (n+2) = (y n + 2013)/(1 + y (n+1))) : 
  ∃ y1 y2, y1 + y2 = 94 ∧ (∀ n, y n > 0) ∧ (y 1 = y1) ∧ (y 2 = y2) := 
sorry

end min_sum_y1_y2_l2030_203088


namespace power_function_passes_point_l2030_203027

noncomputable def f (k α x : ℝ) : ℝ := k * x^α

theorem power_function_passes_point (k α : ℝ) (h1 : f k α (1/2) = (Real.sqrt 2)/2) : 
  k + α = 3/2 :=
sorry

end power_function_passes_point_l2030_203027


namespace find_e_l2030_203070

variables (j p t b a : ℝ) (e : ℝ)

theorem find_e
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end find_e_l2030_203070


namespace determine_a_l2030_203017

open Real

theorem determine_a :
  (∃ a : ℝ, |x^2 + a*x + 4*a| ≤ 3 → x^2 + a*x + 4*a = 3) ↔ (a = 8 + 2*sqrt 13 ∨ a = 8 - 2*sqrt 13) :=
by
  sorry

end determine_a_l2030_203017


namespace smallest_digit_not_in_odd_units_l2030_203059

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l2030_203059


namespace range_of_m_l2030_203058

open Real

theorem range_of_m (m : ℝ) : (¬ ∃ x₀ : ℝ, m * x₀^2 + m * x₀ + 1 ≤ 0) ↔ (0 ≤ m ∧ m < 4) := by
  sorry

end range_of_m_l2030_203058


namespace square_rectangle_area_ratio_l2030_203067

theorem square_rectangle_area_ratio (l1 l2 : ℕ) (h1 : l1 = 32) (h2 : l2 = 64) (p : ℕ) (s : ℕ) 
  (h3 : p = 256) (h4 : s = p / 4)  :
  (s * s) / (l1 * l2) = 2 := 
by
  sorry

end square_rectangle_area_ratio_l2030_203067


namespace probability_of_team_A_winning_is_11_over_16_l2030_203096

noncomputable def prob_A_wins_series : ℚ :=
  let total_games := 5
  let wins_needed_A := 2
  let wins_needed_B := 3
  -- Assuming equal probability for each game being won by either team
  let equal_chance_of_winning := 0.5
  -- Calculation would follow similar steps omitted for brevity
  -- Assuming the problem statement proven by external logical steps
  11 / 16

theorem probability_of_team_A_winning_is_11_over_16 :
  prob_A_wins_series = 11 / 16 := 
  sorry

end probability_of_team_A_winning_is_11_over_16_l2030_203096


namespace model_to_statue_scale_l2030_203053

theorem model_to_statue_scale
  (statue_height_ft : ℕ)
  (model_height_in : ℕ)
  (ft_to_in : ℕ)
  (statue_height_in : ℕ)
  (scale : ℕ)
  (h1 : statue_height_ft = 120)
  (h2 : model_height_in = 6)
  (h3 : ft_to_in = 12)
  (h4 : statue_height_in = statue_height_ft * ft_to_in)
  (h5 : scale = (statue_height_in / model_height_in) / ft_to_in) : scale = 20 := 
  sorry

end model_to_statue_scale_l2030_203053


namespace carlos_and_dana_rest_days_l2030_203062

structure Schedule where
  days_of_cycle : ℕ
  work_days : ℕ
  rest_days : ℕ

def carlos : Schedule := ⟨7, 5, 2⟩
def dana : Schedule := ⟨13, 9, 4⟩

def days_both_rest (days_count : ℕ) (sched1 sched2 : Schedule) : ℕ :=
  let lcm_cycle := Nat.lcm sched1.days_of_cycle sched2.days_of_cycle
  let coincidences_in_cycle := 2  -- As derived from the solution
  let full_cycles := days_count / lcm_cycle
  coincidences_in_cycle * full_cycles

theorem carlos_and_dana_rest_days :
  days_both_rest 1500 carlos dana = 32 := by
  sorry

end carlos_and_dana_rest_days_l2030_203062


namespace perpendicular_lines_parallel_lines_l2030_203092

-- Define the given lines
def l1 (m : ℝ) (x y : ℝ) : ℝ := (m-2)*x + 3*y + 2*m
def l2 (m x y : ℝ) : ℝ := x + m*y + 6

-- The slope conditions for the lines to be perpendicular
def slopes_perpendicular (m : ℝ) : Prop :=
  (m - 2) * m = 3

-- The slope conditions for the lines to be parallel
def slopes_parallel (m : ℝ) : Prop :=
  m = -1

-- Perpendicular lines proof statement
theorem perpendicular_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_perpendicular m :=
sorry

-- Parallel lines proof statement
theorem parallel_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_parallel m :=
sorry

end perpendicular_lines_parallel_lines_l2030_203092


namespace rectangle_perimeter_of_divided_square_l2030_203020

theorem rectangle_perimeter_of_divided_square
  (s : ℝ)
  (hs : 4 * s = 100) :
  let l := s
  let w := s / 2
  2 * (l + w) = 75 :=
by
  let l := s
  let w := s / 2
  sorry

end rectangle_perimeter_of_divided_square_l2030_203020


namespace remainder_of_3_pow_108_plus_5_l2030_203063

theorem remainder_of_3_pow_108_plus_5 :
  (3^108 + 5) % 10 = 6 := by
  sorry

end remainder_of_3_pow_108_plus_5_l2030_203063


namespace volume_conversion_l2030_203022

theorem volume_conversion (V_ft : ℕ) (h_V : V_ft = 216) (conversion_factor : ℕ) (h_cf : conversion_factor = 27) :
  V_ft / conversion_factor = 8 :=
by
  sorry

end volume_conversion_l2030_203022


namespace train_speed_is_60_0131_l2030_203043

noncomputable def train_speed (speed_of_man_kmh : ℝ) (length_of_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_of_man_ms := speed_of_man_kmh * 1000 / 3600
  let relative_speed := length_of_train_m / time_s
  let train_speed_ms := relative_speed - speed_of_man_ms
  train_speed_ms * 3600 / 1000

theorem train_speed_is_60_0131 :
  train_speed 6 330 17.998560115190788 = 60.0131 := by
  sorry

end train_speed_is_60_0131_l2030_203043


namespace triangle_DEF_area_10_l2030_203083

-- Definitions of vertices and line
def D : ℝ × ℝ := (4, 0)
def E : ℝ × ℝ := (0, 4)
def line (x y : ℝ) : Prop := x + y = 9

-- Definition of point F lying on the given line
axiom F_on_line (F : ℝ × ℝ) : line (F.1) (F.2)

-- The proof statement of the area of triangle DEF being 10
theorem triangle_DEF_area_10 : ∃ F : ℝ × ℝ, line F.1 F.2 ∧ 
  (1 / 2) * abs (D.1 - F.1) * abs E.2 = 10 :=
by
  sorry

end triangle_DEF_area_10_l2030_203083


namespace other_root_l2030_203084

-- Define the condition that one root of the quadratic equation is -3
def is_root (a b c : ℤ) (x : ℚ) : Prop := a * x^2 + b * x + c = 0

-- Define the quadratic equation 7x^2 + mx - 6 = 0
def quadratic_eq (m : ℤ) (x : ℚ) : Prop := is_root 7 m (-6) x

-- Prove that the other root is 2/7 given that one root is -3
theorem other_root (m : ℤ) (h : quadratic_eq m (-3)) : quadratic_eq m (2 / 7) :=
by
  sorry

end other_root_l2030_203084


namespace tim_took_rulers_l2030_203011

theorem tim_took_rulers (initial_rulers : ℕ) (remaining_rulers : ℕ) (rulers_taken : ℕ) :
  initial_rulers = 46 → remaining_rulers = 21 → rulers_taken = initial_rulers - remaining_rulers → rulers_taken = 25 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end tim_took_rulers_l2030_203011


namespace find_line_equation_of_ellipse_intersection_l2030_203041

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 2 = 1

-- Defining the line intersects points
def line_intersects (A B : ℝ × ℝ) : Prop := 
  ∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ 
  (ellipse x1 y1) ∧ (ellipse x2 y2) ∧ 
  ((x1 + x2) / 2 = 1 / 2) ∧ ((y1 + y2) / 2 = -1)

-- Statement to prove the equation of the line
theorem find_line_equation_of_ellipse_intersection (A B : ℝ × ℝ)
  (h : line_intersects A B) : 
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ x - 4*y - (9/2) = 0) :=
sorry

end find_line_equation_of_ellipse_intersection_l2030_203041


namespace Z_real_Z_imaginary_Z_pure_imaginary_l2030_203039

-- Definitions

def Z (a : ℝ) : ℂ := (a^2 - 9 : ℝ) + (a^2 - 2 * a - 15 : ℂ)

-- Statement for the proof problems

theorem Z_real (a : ℝ) : 
  (Z a).im = 0 ↔ a = 5 ∨ a = -3 := sorry

theorem Z_imaginary (a : ℝ) : 
  (Z a).re = 0 ↔ a ≠ 5 ∧ a ≠ -3 := sorry

theorem Z_pure_imaginary (a : ℝ) : 
  (Z a).re = 0 ∧ (Z a).im ≠ 0 ↔ a = 3 := sorry

end Z_real_Z_imaginary_Z_pure_imaginary_l2030_203039


namespace count_valid_Q_l2030_203018

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 5)

def Q_degree (Q : Polynomial ℝ) : Prop :=
  Q.degree = 2

def R_degree (R : Polynomial ℝ) : Prop :=
  R.degree = 3

def P_Q_relation (Q R : Polynomial ℝ) : Prop :=
  ∀ x, P (Q.eval x) = P x * R.eval x

theorem count_valid_Q : 
  (∃ Qs : Finset (Polynomial ℝ), ∀ Q ∈ Qs, Q_degree Q ∧ (∃ R, R_degree R ∧ P_Q_relation Q R) 
    ∧ Qs.card = 22) :=
sorry

end count_valid_Q_l2030_203018


namespace problem1_problem2_l2030_203042

-- Problem 1
theorem problem1 (a b : ℝ) (h : 2 * (a + 1) * (b + 1) = (a + b) * (a + b + 2)) : a^2 + b^2 = 2 := sorry

-- Problem 2
theorem problem2 (a b c : ℝ) (h : a^2 + c^2 = 2 * b^2) : (a + b) * (a + c) + (c + a) * (c + b) = 2 * (b + a) * (b + c) := sorry

end problem1_problem2_l2030_203042


namespace ship_navigation_avoid_reefs_l2030_203021

theorem ship_navigation_avoid_reefs (a : ℝ) (h : a > 0) :
  (10 * a) * 40 / Real.sqrt ((10 * a) ^ 2 + 40 ^ 2) > 20 ↔
  a > (4 * Real.sqrt 3 / 3) :=
by
  sorry

end ship_navigation_avoid_reefs_l2030_203021


namespace master_bedroom_and_bath_area_l2030_203056

-- Definitions of the problem conditions
def guest_bedroom_area : ℕ := 200
def two_guest_bedrooms_area : ℕ := 2 * guest_bedroom_area
def kitchen_guest_bath_living_area : ℕ := 600
def total_rent : ℕ := 3000
def cost_per_sq_ft : ℕ := 2
def total_area_of_house : ℕ := total_rent / cost_per_sq_ft
def expected_master_bedroom_and_bath_area : ℕ := 500

-- Theorem statement to prove the desired area
theorem master_bedroom_and_bath_area :
  total_area_of_house - (two_guest_bedrooms_area + kitchen_guest_bath_living_area) = expected_master_bedroom_and_bath_area :=
by
  sorry

end master_bedroom_and_bath_area_l2030_203056


namespace reciprocal_of_fraction_subtraction_l2030_203040

theorem reciprocal_of_fraction_subtraction : (1 / ((2 / 3) - (3 / 4))) = -12 := by
  sorry

end reciprocal_of_fraction_subtraction_l2030_203040


namespace inequality_proof_l2030_203098

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hx1y1 : x1 * y1 - z1^2 > 0) (hx2y2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 - z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l2030_203098


namespace minimum_value_of_f_l2030_203093

noncomputable def f (x : ℝ) : ℝ := sorry

theorem minimum_value_of_f :
  (∀ x : ℝ, f (x + 1) + f (x - 1) = 2 * x^2 - 4 * x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = -2 :=
by
  sorry

end minimum_value_of_f_l2030_203093


namespace circumference_base_of_cone_l2030_203078

-- Define the given conditions
def radius_circle : ℝ := 6
def angle_sector : ℝ := 300

-- Define the problem to prove the circumference of the base of the resulting cone in terms of π
theorem circumference_base_of_cone :
  (angle_sector / 360) * (2 * π * radius_circle) = 10 * π := by
sorry

end circumference_base_of_cone_l2030_203078


namespace sum_lent_is_1050_l2030_203029

-- Define the variables for the problem
variable (P : ℝ) -- Sum lent
variable (r : ℝ) -- Interest rate
variable (t : ℝ) -- Time period
variable (I : ℝ) -- Interest

-- Define the conditions
def conditions := 
  r = 0.06 ∧ 
  t = 6 ∧ 
  I = P - 672 ∧ 
  I = P * (r * t)

-- Define the main theorem
theorem sum_lent_is_1050 (P r t I : ℝ) (h : conditions P r t I) : P = 1050 :=
  sorry

end sum_lent_is_1050_l2030_203029


namespace integer_values_satisfying_square_root_condition_l2030_203010

theorem integer_values_satisfying_square_root_condition :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 := sorry

end integer_values_satisfying_square_root_condition_l2030_203010


namespace intersection_A_B_l2030_203013

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {y : ℝ | ∃ x : ℝ, y = 2^x}

theorem intersection_A_B :
  A ∩ {x : ℝ | x > 0} = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_B_l2030_203013


namespace hiking_trip_rate_ratio_l2030_203030

theorem hiking_trip_rate_ratio 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down : ℝ)
  (h1 : rate_up = 7) 
  (h2 : time_up = 2) 
  (h3 : distance_down = 21) 
  (h4 : time_down = 2) : 
  (distance_down / time_down) / rate_up = 1.5 :=
by
  -- skip the proof as per instructions
  sorry

end hiking_trip_rate_ratio_l2030_203030


namespace sequence_converges_and_limit_l2030_203016

theorem sequence_converges_and_limit {a : ℝ} (m : ℕ) (h_a_pos : 0 < a) (h_m_pos : 0 < m) :
  (∃ (x : ℕ → ℝ), 
  (x 1 = 1) ∧ 
  (x 2 = a) ∧ 
  (∀ n : ℕ, x (n + 2) = (x (n + 1) ^ m * x n) ^ (↑(1 : ℕ) / (m + 1))) ∧ 
  ∃ l : ℝ, (∀ ε > 0, ∃ N, ∀ n > N, |x n - l| < ε) ∧ l = a ^ (↑(m + 1) / ↑(m + 2))) :=
sorry

end sequence_converges_and_limit_l2030_203016


namespace max_gcd_of_consecutive_terms_seq_b_l2030_203052

-- Define the sequence b_n
def sequence_b (n : ℕ) : ℕ := n.factorial + 3 * n

-- Define the gcd function for two terms in the sequence
def gcd_two_terms (n : ℕ) : ℕ := Nat.gcd (sequence_b n) (sequence_b (n + 1))

-- Define the condition of n being greater than or equal to 0
def n_ge_zero (n : ℕ) : Prop := n ≥ 0

-- The theorem statement
theorem max_gcd_of_consecutive_terms_seq_b : ∃ n : ℕ, n_ge_zero n ∧ gcd_two_terms n = 14 := 
sorry

end max_gcd_of_consecutive_terms_seq_b_l2030_203052


namespace differentiable_function_zero_l2030_203055

noncomputable def f : ℝ → ℝ := sorry

theorem differentiable_function_zero (f : ℝ → ℝ) (h_diff : ∀ x ≥ 0, DifferentiableAt ℝ f x)
  (h_f0 : f 0 = 0) (h_fun : ∀ x ≥ 0, ∀ y ≥ 0, (x = y^2) → deriv f x = f y) : 
  ∀ x ≥ 0, f x = 0 :=
by
  sorry

end differentiable_function_zero_l2030_203055


namespace set_intersection_complement_l2030_203014

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 3}
noncomputable def B : Set ℕ := {2, 3}

theorem set_intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end set_intersection_complement_l2030_203014


namespace cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l2030_203094

def cube (n : ℕ) : Type := ℕ × ℕ × ℕ

-- Define a 4x4x4 cube and the painting conditions
def four_by_four_cube := cube 4

-- Determine the number of small cubes with exactly one face painted
theorem cubes_with_one_face_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Determine the number of small cubes with exactly two faces painted
theorem cubes_with_two_faces_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Given condition and find the size of the new cube
theorem size_of_new_cube (n : ℕ) : 
  (n - 2) ^ 3 = 3 * 12 * (n - 2) → n = 8 :=
by
  -- proof goes here
  sorry

end cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l2030_203094


namespace arithmetic_seq_sum_l2030_203057

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 4 + a 7 = 39) (h2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end arithmetic_seq_sum_l2030_203057
