import Mathlib

namespace NUMINAMATH_GPT_rectangle_ratio_ratio_simplification_l469_46960

theorem rectangle_ratio (w : ℕ) (h : w + 10 = 10) (p : 2 * w + 2 * 10 = 30) :
  w = 5 := by
  sorry

theorem ratio_simplification (x y : ℕ) (h : x * 10 = y * 5) (rel_prime : Nat.gcd x y = 1) :
  (x, y) = (1, 2) := by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_ratio_simplification_l469_46960


namespace NUMINAMATH_GPT_paul_mowing_money_l469_46957

theorem paul_mowing_money (M : ℝ) 
  (h1 : 2 * M = 6) : 
  M = 3 :=
by 
  sorry

end NUMINAMATH_GPT_paul_mowing_money_l469_46957


namespace NUMINAMATH_GPT_find_frac_sin_cos_l469_46944

theorem find_frac_sin_cos (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin (3 * Real.pi / 2 + α)) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_frac_sin_cos_l469_46944


namespace NUMINAMATH_GPT_percentage_sales_other_l469_46993

theorem percentage_sales_other (p_pens p_pencils p_markers p_other : ℕ)
(h_pens : p_pens = 25)
(h_pencils : p_pencils = 30)
(h_markers : p_markers = 20)
(h_other : p_other = 100 - (p_pens + p_pencils + p_markers)): p_other = 25 :=
by
  rw [h_pens, h_pencils, h_markers] at h_other
  exact h_other


end NUMINAMATH_GPT_percentage_sales_other_l469_46993


namespace NUMINAMATH_GPT_sym_coords_origin_l469_46991

theorem sym_coords_origin (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) :
  (-a, -b) = (-3, 4) :=
sorry

end NUMINAMATH_GPT_sym_coords_origin_l469_46991


namespace NUMINAMATH_GPT_man_born_in_1892_l469_46968

-- Define the conditions and question
def man_birth_year (x : ℕ) : ℕ :=
x^2 - x

-- Conditions:
variable (x : ℕ)
-- 1. The man was born in the first half of the 20th century
variable (h1 : man_birth_year x < 1950)
-- 2. The man's age x and the conditions in the problem
variable (h2 : x^2 - x < 1950)

-- The statement we aim to prove
theorem man_born_in_1892 (x : ℕ) (h1 : man_birth_year x < 1950) (h2 : x = 44) : man_birth_year x = 1892 := by
  sorry

end NUMINAMATH_GPT_man_born_in_1892_l469_46968


namespace NUMINAMATH_GPT_range_of_m_l469_46901

theorem range_of_m (x : ℝ) (h₁ : 1/2 ≤ x) (h₂ : x ≤ 2) :
  2 - Real.log 2 ≤ -Real.log x + 3*x - x^2 ∧ -Real.log x + 3*x - x^2 ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l469_46901


namespace NUMINAMATH_GPT_inverse_proposition_true_l469_46954

theorem inverse_proposition_true (x : ℝ) (h : x > 1 → x^2 > 1) : x^2 ≤ 1 → x ≤ 1 :=
by
  intros h₂
  sorry

end NUMINAMATH_GPT_inverse_proposition_true_l469_46954


namespace NUMINAMATH_GPT_min_cost_and_ways_l469_46975

-- Define the cost of each package
def cost_A : ℕ := 10
def cost_B : ℕ := 5

-- Define a function to calculate the total cost given the number of each package
def total_cost (nA nB : ℕ) : ℕ := nA * cost_A + nB * cost_B

-- Define the number of friends
def num_friends : ℕ := 4

-- Prove the minimum cost is 15 yuan and there are 28 ways
theorem min_cost_and_ways :
  (∃ nA nB : ℕ, total_cost nA nB = 15 ∧ (
    (nA = 1 ∧ nB = 1 ∧ 12 = 12) ∨ 
    (nA = 0 ∧ nB = 3 ∧ 12 = 12) ∨
    (nA = 0 ∧ nB = 3 ∧ 4 = 4) → 28 = 28)) :=
sorry

end NUMINAMATH_GPT_min_cost_and_ways_l469_46975


namespace NUMINAMATH_GPT_average_of_remaining_two_l469_46949

theorem average_of_remaining_two (a1 a2 a3 a4 a5 : ℚ)
  (h1 : (a1 + a2 + a3 + a4 + a5) / 5 = 11)
  (h2 : (a1 + a2 + a3) / 3 = 4) :
  ((a4 + a5) / 2 = 21.5) :=
sorry

end NUMINAMATH_GPT_average_of_remaining_two_l469_46949


namespace NUMINAMATH_GPT_simplify_expression_l469_46987

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l469_46987


namespace NUMINAMATH_GPT_find_g_four_l469_46900

theorem find_g_four (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 11 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_g_four_l469_46900


namespace NUMINAMATH_GPT_determine_phi_l469_46912

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem determine_phi 
  (φ : ℝ)
  (H1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|)
  (H2 : f (π / 3) φ > f (π / 2) φ) :
  φ = π / 6 :=
sorry

end NUMINAMATH_GPT_determine_phi_l469_46912


namespace NUMINAMATH_GPT_five_consecutive_product_div_24_l469_46995

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end NUMINAMATH_GPT_five_consecutive_product_div_24_l469_46995


namespace NUMINAMATH_GPT_theater_revenue_l469_46918

theorem theater_revenue 
  (seats : ℕ)
  (capacity_percentage : ℝ)
  (ticket_price : ℝ)
  (days : ℕ)
  (H1 : seats = 400)
  (H2 : capacity_percentage = 0.8)
  (H3 : ticket_price = 30)
  (H4 : days = 3)
  : (seats * capacity_percentage * ticket_price * days = 28800) :=
by
  sorry

end NUMINAMATH_GPT_theater_revenue_l469_46918


namespace NUMINAMATH_GPT_max_k_value_l469_46937

theorem max_k_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = a * b + b * c + c * a →
  (a + b + c) * (1 / (a + b) + 1 / (b + c) + 1 / (c + a) - 1) ≥ 1 :=
by
  intros a b c ha hb hc habc_eq
  sorry

end NUMINAMATH_GPT_max_k_value_l469_46937


namespace NUMINAMATH_GPT_sum_of_distinct_integers_l469_46907

noncomputable def a : ℤ := 11
noncomputable def b : ℤ := 9
noncomputable def c : ℤ := 4
noncomputable def d : ℤ := 2
noncomputable def e : ℤ := 1

def condition : Prop := (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120
def distinct_integers : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem sum_of_distinct_integers (h1 : condition) (h2 : distinct_integers) : a + b + c + d + e = 27 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_l469_46907


namespace NUMINAMATH_GPT_num_multiples_of_three_in_ap_l469_46977

variable (a : ℕ → ℚ)  -- Defining the arithmetic sequence

def first_term (a1 : ℚ) := a 1 = a1
def eighth_term (a8 : ℚ) := a 8 = a8
def general_term (d : ℚ) := ∀ n : ℕ, a n = 9 + (n - 1) * d
def multiple_of_three (n : ℕ) := ∃ k : ℕ, a n = 3 * k

theorem num_multiples_of_three_in_ap 
  (a : ℕ → ℚ)
  (h1 : first_term a 9)
  (h2 : eighth_term a 12) :
  ∃ n : ℕ, n = 288 ∧ ∃ l : ℕ → Prop, ∀ k : ℕ, l k → multiple_of_three a (k * 7 + 1) :=
sorry

end NUMINAMATH_GPT_num_multiples_of_three_in_ap_l469_46977


namespace NUMINAMATH_GPT_fraction_of_number_is_one_fifth_l469_46996

theorem fraction_of_number_is_one_fifth (N : ℕ) (f : ℚ) 
    (hN : N = 90) 
    (h : 3 + (1 / 2) * (1 / 3) * f * N = (1 / 15) * N) : 
  f = 1 / 5 := by 
  sorry

end NUMINAMATH_GPT_fraction_of_number_is_one_fifth_l469_46996


namespace NUMINAMATH_GPT_find_expression_l469_46992

variables {x y : ℝ}

theorem find_expression
  (h1: 3 * x + y = 5)
  (h2: x + 3 * y = 6)
  : 10 * x^2 + 13 * x * y + 10 * y^2 = 97 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_l469_46992


namespace NUMINAMATH_GPT_intersection_point_parallel_line_through_intersection_l469_46980

-- Definitions for the problem
def l1 (x y : ℝ) : Prop := x + 8 * y + 7 = 0
def l2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x + y + 1 = 0
def parallel (x y c : ℝ) : Prop := x + y + c = 0
def point (x y : ℝ) : Prop := x = 1 ∧ y = -1

-- (1) Proof that the intersection point of l1 and l2 is (1, -1)
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ point x y :=
by 
  sorry

-- (2) Proof that the line passing through the intersection point of l1 and l2
-- which is parallel to l3 is x + y = 0
theorem parallel_line_through_intersection : ∃ (c : ℝ), parallel 1 (-1) c ∧ c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_intersection_point_parallel_line_through_intersection_l469_46980


namespace NUMINAMATH_GPT_problem_b_problem_c_problem_d_l469_46961

variable (a b : ℝ)

theorem problem_b (h : a * b > 0) :
  2 * (a^2 + b^2) ≥ (a + b)^2 :=
sorry

theorem problem_c (h : a * b > 0) :
  (b / a) + (a / b) ≥ 2 :=
sorry

theorem problem_d (h : a * b > 0) :
  (a + 1 / a) * (b + 1 / b) ≥ 4 :=
sorry

end NUMINAMATH_GPT_problem_b_problem_c_problem_d_l469_46961


namespace NUMINAMATH_GPT_min_abs_ab_l469_46958

theorem min_abs_ab (a b : ℤ) (h : 1009 * a + 2 * b = 1) : ∃ k : ℤ, |a * b| = 504 :=
by
  sorry

end NUMINAMATH_GPT_min_abs_ab_l469_46958


namespace NUMINAMATH_GPT_find_constants_a_b_l469_46919

def M : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![2, -2]
]

theorem find_constants_a_b :
  ∃ (a b : ℚ), (M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  a = 1/8 ∧ b = -1/8 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_a_b_l469_46919


namespace NUMINAMATH_GPT_first_other_factor_of_lcm_l469_46990

theorem first_other_factor_of_lcm (A B hcf lcm : ℕ) (h1 : A = 368) (h2 : hcf = 23) (h3 : lcm = hcf * 16 * X) :
  X = 1 :=
by
  sorry

end NUMINAMATH_GPT_first_other_factor_of_lcm_l469_46990


namespace NUMINAMATH_GPT_marie_distance_biked_l469_46951

def biking_speed := 12.0 -- Speed in miles per hour
def biking_time := 2.583333333 -- Time in hours

theorem marie_distance_biked : biking_speed * biking_time = 31 := 
by 
  -- The proof steps go here
  sorry

end NUMINAMATH_GPT_marie_distance_biked_l469_46951


namespace NUMINAMATH_GPT_nancy_total_spent_l469_46965

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end NUMINAMATH_GPT_nancy_total_spent_l469_46965


namespace NUMINAMATH_GPT_eggs_per_omelet_l469_46994

theorem eggs_per_omelet:
  let small_children_tickets := 53
  let older_children_tickets := 35
  let adult_tickets := 75
  let senior_tickets := 37
  let smallChildrenOmelets := small_children_tickets * 0.5
  let olderChildrenOmelets := older_children_tickets
  let adultOmelets := adult_tickets * 2
  let seniorOmelets := senior_tickets * 1.5
  let extra_omelets := 25
  let total_omelets := smallChildrenOmelets + olderChildrenOmelets + adultOmelets + seniorOmelets + extra_omelets
  let total_eggs := 584
  total_eggs / total_omelets = 2 := 
by
  sorry

end NUMINAMATH_GPT_eggs_per_omelet_l469_46994


namespace NUMINAMATH_GPT_part_1_part_2_l469_46903

-- Definitions based on given conditions
def a : ℕ → ℝ := λ n => 2 * n + 1
noncomputable def b : ℕ → ℝ := λ n => 1 / ((2 * n + 1)^2 - 1)
noncomputable def S : ℕ → ℝ := λ n => n ^ 2 + 2 * n
noncomputable def T : ℕ → ℝ := λ n => n / (4 * (n + 1))

-- Lean statement for proving the problem
theorem part_1 (n : ℕ) :
  ∀ a_3 a_5 a_7 : ℝ, 
  a 3 = a_3 → 
  a_3 = 7 →
  a_5 = a 5 →
  a_7 = a 7 →
  a_5 + a_7 = 26 →
  ∃ a_1 d : ℝ,
    (a 1 = a_1 + 0 * d) ∧
    (a 2 = a_1 + 1 * d) ∧
    (a 3 = a_1 + 2 * d) ∧
    (a 4 = a_1 + 3 * d) ∧
    (a 5 = a_1 + 4 * d) ∧
    (a 7 = a_1 + 6 * d) ∧
    (a n = a_1 + (n - 1) * d) ∧
    (S n = n^2 + 2*n) := sorry

theorem part_2 (n : ℕ) :
  ∀ a_n b_n : ℝ,
  b n = b_n →
  a n = a_n →
  1 / b n = a_n^2 - 1 →
  T n = τ →
  (T n = n / (4 * (n + 1))) := sorry

end NUMINAMATH_GPT_part_1_part_2_l469_46903


namespace NUMINAMATH_GPT_solve_for_x_l469_46926

theorem solve_for_x (x : ℤ) (h : (3012 + x)^2 = x^2) : x = -1506 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l469_46926


namespace NUMINAMATH_GPT_total_cost_of_apples_l469_46966

variable (num_apples_per_bag cost_per_bag num_apples : ℕ)
#check num_apples_per_bag = 50
#check cost_per_bag = 8
#check num_apples = 750

theorem total_cost_of_apples : 
  (num_apples_per_bag = 50) → 
  (cost_per_bag = 8) → 
  (num_apples = 750) → 
  (num_apples / num_apples_per_bag * cost_per_bag = 120) :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_cost_of_apples_l469_46966


namespace NUMINAMATH_GPT_quadrilateral_is_parallelogram_l469_46974

theorem quadrilateral_is_parallelogram
  (AB BC CD DA : ℝ)
  (K L M N : ℝ)
  (H₁ : K = (AB + BC) / 2)
  (H₂ : L = (BC + CD) / 2)
  (H₃ : M = (CD + DA) / 2)
  (H₄ : N = (DA + AB) / 2)
  (H : K + M + L + N = (AB + BC + CD + DA) / 2)
  : ∃ P Q R S : ℝ, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P ∧ 
    (P + R = AB) ∧ (Q + S = CD)  := 
sorry

end NUMINAMATH_GPT_quadrilateral_is_parallelogram_l469_46974


namespace NUMINAMATH_GPT_rectangle_measurement_error_l469_46988

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : 0 < L) (h2 : 0 < W) 
  (h3 : A = L * W)
  (h4 : A' = L * (1 + x / 100) * W * (1 - 4 / 100))
  (h5 : A' = A * (100.8 / 100)) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_measurement_error_l469_46988


namespace NUMINAMATH_GPT_alyssa_cut_11_roses_l469_46920

theorem alyssa_cut_11_roses (initial_roses cut_roses final_roses : ℕ) 
  (h1 : initial_roses = 3) 
  (h2 : final_roses = 14) 
  (h3 : initial_roses + cut_roses = final_roses) : 
  cut_roses = 11 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_alyssa_cut_11_roses_l469_46920


namespace NUMINAMATH_GPT_cadence_total_earnings_l469_46905

noncomputable def total_earnings (old_years : ℕ) (old_monthly : ℕ) (new_increment : ℤ) (extra_months : ℕ) : ℤ :=
  let old_months := old_years * 12
  let old_earnings := old_monthly * old_months
  let new_monthly := old_monthly + ((old_monthly * new_increment) / 100)
  let new_months := old_months + extra_months
  let new_earnings := new_monthly * new_months
  old_earnings + new_earnings

theorem cadence_total_earnings :
  total_earnings 3 5000 20 5 = 426000 :=
by
  sorry

end NUMINAMATH_GPT_cadence_total_earnings_l469_46905


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l469_46916

theorem solve_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, (a * x ^ 2 - (2 * a + 1) * x + 2 > 0 ↔
    if a = 0 then
      x < 2
    else if a > 0 then
      if a >= 1 / 2 then
        x < 1 / a ∨ x > 2
      else
        x < 2 ∨ x > 1 / a
    else
      x > 1 / a ∧ x < 2)) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l469_46916


namespace NUMINAMATH_GPT_smaller_number_eq_l469_46997

variable (m n t s : ℝ)
variable (h_ratio : m / n = t)
variable (h_sum : m + n = s)
variable (h_t_gt_one : t > 1)

theorem smaller_number_eq : n = s / (1 + t) :=
by sorry

end NUMINAMATH_GPT_smaller_number_eq_l469_46997


namespace NUMINAMATH_GPT_geom_sequence_product_l469_46924

noncomputable def geom_seq (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_product (a : ℕ → ℝ) (h1 : geom_seq a) (h2 : a 0 * a 4 = 4) :
  a 0 * a 1 * a 2 * a 3 * a 4 = 32 ∨ a 0 * a 1 * a 2 * a 3 * a 4 = -32 :=
by
  sorry

end NUMINAMATH_GPT_geom_sequence_product_l469_46924


namespace NUMINAMATH_GPT_diving_assessment_l469_46948

theorem diving_assessment (total_athletes : ℕ) (selected_athletes : ℕ) (not_meeting_standard : ℕ) 
  (first_level_sample : ℕ) (first_level_total : ℕ) (athletes : Set ℕ) :
  total_athletes = 56 → 
  selected_athletes = 8 → 
  not_meeting_standard = 2 → 
  first_level_sample = 3 → 
  (∀ (A B C D E : ℕ), athletes = {A, B, C, D, E} → first_level_total = 5 → 
  (∃ proportion_standard number_first_level probability_E, 
    proportion_standard = (8 - 2) / 8 ∧  -- first part: proportion of athletes who met the standard
    number_first_level = 56 * (3 / 8) ∧ -- second part: number of first-level athletes
    probability_E = 4 / 10))           -- third part: probability of athlete E being chosen
:= sorry

end NUMINAMATH_GPT_diving_assessment_l469_46948


namespace NUMINAMATH_GPT_total_area_at_stage_4_l469_46914

/-- Define the side length of the square at a given stage -/
def side_length (n : ℕ) : ℕ := n + 2

/-- Define the area of the square at a given stage -/
def area (n : ℕ) : ℕ := (side_length n) ^ 2

/-- State the theorem -/
theorem total_area_at_stage_4 : 
  (area 0) + (area 1) + (area 2) + (area 3) = 86 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_area_at_stage_4_l469_46914


namespace NUMINAMATH_GPT_circle_hyperbola_intersection_l469_46986

def hyperbola_equation (x y a : ℝ) : Prop := x^2 - y^2 = a^2
def circle_equation (x y c d r : ℝ) : Prop := (x - c)^2 + (y - d)^2 = r^2

theorem circle_hyperbola_intersection (a r : ℝ) (P Q R S : ℝ × ℝ):
  (∃ c d: ℝ, 
    circle_equation P.1 P.2 c d r ∧ 
    circle_equation Q.1 Q.2 c d r ∧ 
    circle_equation R.1 R.2 c d r ∧ 
    circle_equation S.1 S.2 c d r ∧ 
    hyperbola_equation P.1 P.2 a ∧ 
    hyperbola_equation Q.1 Q.2 a ∧ 
    hyperbola_equation R.1 R.2 a ∧ 
    hyperbola_equation S.1 S.2 a
  ) →
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_hyperbola_intersection_l469_46986


namespace NUMINAMATH_GPT_max_slope_of_circle_l469_46952

theorem max_slope_of_circle (x y : ℝ) 
  (h : x^2 + y^2 - 6 * x - 6 * y + 12 = 0) : 
  ∃ k : ℝ, k = 3 + 2 * Real.sqrt 2 ∧ ∀ k' : ℝ, (x = 0 → k' = 0) ∧ (x ≠ 0 → y = k' * x → k' ≤ k) :=
sorry

end NUMINAMATH_GPT_max_slope_of_circle_l469_46952


namespace NUMINAMATH_GPT_shooting_competition_probabilities_l469_46947

theorem shooting_competition_probabilities (p_A_not_losing p_B_losing : ℝ)
  (h₁ : p_A_not_losing = 0.59)
  (h₂ : p_B_losing = 0.44) :
  (1 - p_B_losing = 0.56) ∧ (p_A_not_losing - p_B_losing = 0.15) :=
by
  sorry

end NUMINAMATH_GPT_shooting_competition_probabilities_l469_46947


namespace NUMINAMATH_GPT_perfect_square_iff_l469_46984

theorem perfect_square_iff (A : ℕ) : (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, n > 0 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n ∣ ((A + k)^2 - A)) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_iff_l469_46984


namespace NUMINAMATH_GPT_fraction_is_one_over_three_l469_46979

variable (x : ℚ) -- Let the fraction x be a rational number
variable (num : ℚ) -- Let the number be a rational number

theorem fraction_is_one_over_three (h1 : num = 45) (h2 : x * num - 5 = 10) : x = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_is_one_over_three_l469_46979


namespace NUMINAMATH_GPT_max_min_fraction_l469_46959

-- Given condition
def circle_condition (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 1 = 0

-- Problem statement
theorem max_min_fraction (x y : ℝ) (h : circle_condition x y) :
  -20 / 21 ≤ y / (x - 4) ∧ y / (x - 4) ≤ 0 :=
sorry

end NUMINAMATH_GPT_max_min_fraction_l469_46959


namespace NUMINAMATH_GPT_sum_of_products_l469_46913

theorem sum_of_products : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end NUMINAMATH_GPT_sum_of_products_l469_46913


namespace NUMINAMATH_GPT_geometric_sequence_properties_l469_46967

noncomputable def geometric_sequence (a2 a5 : ℕ) (n : ℕ) : ℕ :=
  3 ^ (n - 1)

noncomputable def sum_first_n_terms (n : ℕ) : ℕ :=
  (3^n - 1) / 2

def T10_sum_of_sequence : ℚ := 10/11

theorem geometric_sequence_properties :
  (geometric_sequence 3 81 2 = 3) ∧
  (geometric_sequence 3 81 5 = 81) ∧
  (sum_first_n_terms 2 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2)) ∧
  (sum_first_n_terms 5 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2 + geometric_sequence 3 81 3 + geometric_sequence 3 81 4 + geometric_sequence 3 81 5)) ∧
  T10_sum_of_sequence = 10/11 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l469_46967


namespace NUMINAMATH_GPT_parking_lot_wheels_l469_46981

-- Define the conditions
def num_cars : Nat := 10
def num_bikes : Nat := 2
def wheels_per_car : Nat := 4
def wheels_per_bike : Nat := 2

-- Define the total number of wheels
def total_wheels : Nat := (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike)

-- State the theorem
theorem parking_lot_wheels : total_wheels = 44 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_wheels_l469_46981


namespace NUMINAMATH_GPT_john_marbles_l469_46902

theorem john_marbles : ∃ m : ℕ, (m ≡ 3 [MOD 7]) ∧ (m ≡ 2 [MOD 4]) ∧ m = 10 := by
  sorry

end NUMINAMATH_GPT_john_marbles_l469_46902


namespace NUMINAMATH_GPT_neg_of_forall_sin_ge_neg_one_l469_46942

open Real

theorem neg_of_forall_sin_ge_neg_one :
  (¬ (∀ x : ℝ, sin x ≥ -1)) ↔ (∃ x0 : ℝ, sin x0 < -1) := by
  sorry

end NUMINAMATH_GPT_neg_of_forall_sin_ge_neg_one_l469_46942


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l469_46953

-- Definitions
def is_quadratic_eq (a b c x : ℝ) (fx : ℝ) := a * x^2 + b * x + c = fx

-- Theorem statement
theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_quadratic_eq 1 (-2) m x₁ 0 ∧ is_quadratic_eq 1 (-2) m x₂ 0) → m < 1 :=
sorry -- Proof omitted

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l469_46953


namespace NUMINAMATH_GPT_magic_triangle_max_sum_l469_46941

/-- In a magic triangle, each of the six consecutive whole numbers 11 to 16 is placed in one of the circles. 
    The sum, S, of the three numbers on each side of the triangle is the same. One of the sides must contain 
    three consecutive numbers. Prove that the largest possible value for S is 41. -/
theorem magic_triangle_max_sum :
  ∀ (a b c d e f : ℕ), 
  (a = 11 ∨ a = 12 ∨ a = 13 ∨ a = 14 ∨ a = 15 ∨ a = 16) ∧
  (b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16) ∧
  (c = 11 ∨ c = 12 ∨ c = 13 ∨ c = 14 ∨ c = 15 ∨ c = 16) ∧
  (d = 11 ∨ d = 12 ∨ d = 13 ∨ d = 14 ∨ d = 15 ∨ d = 16) ∧
  (e = 11 ∨ e = 12 ∨ e = 13 ∨ e = 14 ∨ e = 15 ∨ e = 16) ∧
  (f = 11 ∨ f = 12 ∨ f = 13 ∨ f = 14 ∨ f = 15 ∨ f = 16) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
  (a + b + c = S) ∧ (c + d + e = S) ∧ (e + f + a = S) ∧
  (∃ k, a = k ∧ b = k+1 ∧ c = k+2 ∨ b = k ∧ c = k+1 ∧ d = k+2 ∨ c = k ∧ d = k+1 ∧ e = k+2 ∨ d = k ∧ e = k+1 ∧ f = k+2) →
  S = 41 :=
by
  sorry

end NUMINAMATH_GPT_magic_triangle_max_sum_l469_46941


namespace NUMINAMATH_GPT_smallest_number_divisible_l469_46928

theorem smallest_number_divisible (n : ℕ) :
  (∀ d ∈ [4, 6, 8, 10, 12, 14, 16], (n - 16) % d = 0) ↔ n = 3376 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_divisible_l469_46928


namespace NUMINAMATH_GPT_both_firms_participate_social_optimality_l469_46950

variables (α V IC : ℝ)

-- Conditions definitions
def expected_income_if_both_participate (α V : ℝ) : ℝ :=
  α * (1 - α) * V + 0.5 * (α^2) * V

def condition_for_both_participation (α V IC : ℝ) : Prop :=
  expected_income_if_both_participate α V - IC ≥ 0

-- Values for specific case
noncomputable def V_specific : ℝ := 24
noncomputable def α_specific : ℝ := 0.5
noncomputable def IC_specific : ℝ := 7

-- Proof problem statement
theorem both_firms_participate : condition_for_both_participation α_specific V_specific IC_specific := by
  sorry

-- Definitions for social welfare considerations
def total_profit_if_both_participate (α V IC : ℝ) : ℝ :=
  2 * (expected_income_if_both_participate α V - IC)

def expected_income_if_one_participates (α V IC : ℝ) : ℝ :=
  α * V - IC

def social_optimal (α V IC : ℝ) : Prop :=
  total_profit_if_both_participate α V IC < expected_income_if_one_participates α V IC

theorem social_optimality : social_optimal α_specific V_specific IC_specific := by
  sorry

end NUMINAMATH_GPT_both_firms_participate_social_optimality_l469_46950


namespace NUMINAMATH_GPT_angle_sum_of_roots_of_complex_eq_32i_l469_46935

noncomputable def root_angle_sum : ℝ :=
  let θ1 := 22.5
  let θ2 := 112.5
  let θ3 := 202.5
  let θ4 := 292.5
  θ1 + θ2 + θ3 + θ4

theorem angle_sum_of_roots_of_complex_eq_32i :
  root_angle_sum = 630 := by
  sorry

end NUMINAMATH_GPT_angle_sum_of_roots_of_complex_eq_32i_l469_46935


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l469_46963

theorem common_difference_arithmetic_sequence (a b : ℝ) :
  ∃ d : ℝ, b = a + 6 * d ∧ d = (b - a) / 6 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l469_46963


namespace NUMINAMATH_GPT_probability_of_green_l469_46936

open Classical

-- Define the total number of balls in each container
def balls_A := 12
def balls_B := 14
def balls_C := 12

-- Define the number of green balls in each container
def green_balls_A := 7
def green_balls_B := 6
def green_balls_C := 9

-- Define the probability of selecting each container
def prob_select_container := (1:ℚ) / 3

-- Define the probability of drawing a green ball from each container
def prob_green_A := green_balls_A / balls_A
def prob_green_B := green_balls_B / balls_B
def prob_green_C := green_balls_C / balls_C

-- Define the total probability of drawing a green ball
def total_prob_green := prob_select_container * prob_green_A +
                        prob_select_container * prob_green_B +
                        prob_select_container * prob_green_C

-- Create the proof statement
theorem probability_of_green : total_prob_green = 127 / 252 := 
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_probability_of_green_l469_46936


namespace NUMINAMATH_GPT_book_arrangement_l469_46983

theorem book_arrangement (math_books : ℕ) (english_books : ℕ) (science_books : ℕ)
  (math_different : math_books = 4) 
  (english_different : english_books = 5) 
  (science_different : science_books = 2) :
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books) = 34560 := 
by
  sorry

end NUMINAMATH_GPT_book_arrangement_l469_46983


namespace NUMINAMATH_GPT_function_passes_through_point_l469_46906

theorem function_passes_through_point :
  (∃ (a : ℝ), a = 1 ∧ (∀ (x y : ℝ), y = a * x + a → y = x + 1)) →
  ∃ x y : ℝ, x = -2 ∧ y = -1 ∧ y = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_function_passes_through_point_l469_46906


namespace NUMINAMATH_GPT_possible_values_of_d_l469_46917

theorem possible_values_of_d (r s : ℝ) (c d : ℝ)
  (h1 : ∃ u, u = -r - s ∧ r * s + r * u + s * u = c)
  (h2 : ∃ v, v = -r - s - 8 ∧ (r - 3) * (s + 5) + (r - 3) * (u - 8) + (s + 5) * (u - 8) = c)
  (u_eq : u = -r - s)
  (v_eq : v = -r - s - 8)
  (polynomial_relation : d + 156 = -((r - 3) * (s + 5) * (u - 8))) : 
  d = -198 ∨ d = 468 := 
sorry

end NUMINAMATH_GPT_possible_values_of_d_l469_46917


namespace NUMINAMATH_GPT_josh_money_left_l469_46998

theorem josh_money_left (initial_amount : ℝ) (first_spend : ℝ) (second_spend : ℝ) 
  (h1 : initial_amount = 9) 
  (h2 : first_spend = 1.75) 
  (h3 : second_spend = 1.25) : 
  initial_amount - first_spend - second_spend = 6 := 
by 
  sorry

end NUMINAMATH_GPT_josh_money_left_l469_46998


namespace NUMINAMATH_GPT_geometric_sequence_term_number_l469_46929

theorem geometric_sequence_term_number 
  (a_n : ℕ → ℝ)
  (a1 : ℝ) (q : ℝ) (n : ℕ)
  (h1 : a1 = 1/2)
  (h2 : q = 1/2)
  (h3 : a_n n = 1/32)
  (h4 : ∀ n, a_n n = a1 * (q^(n-1))) :
  n = 5 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_number_l469_46929


namespace NUMINAMATH_GPT_compute_expression_l469_46943

noncomputable def a : ℝ := 125^(1/3)
noncomputable def b : ℝ := (-2/3)^0
noncomputable def c : ℝ := Real.log 8 / Real.log 2

theorem compute_expression : a - b - c = 1 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l469_46943


namespace NUMINAMATH_GPT_horse_rent_problem_l469_46931

theorem horse_rent_problem (total_rent : ℝ) (b_payment : ℝ) (a_horses b_horses c_horses : ℝ) 
  (a_months b_months c_months : ℝ) (h_total_rent : total_rent = 870) (h_b_payment : b_payment = 360)
  (h_a_horses : a_horses = 12) (h_b_horses : b_horses = 16) (h_c_horses : c_horses = 18) 
  (h_b_months : b_months = 9) (h_c_months : c_months = 6) : 
  ∃ (a_months : ℝ), (a_horses * a_months * 2.5 + b_payment + c_horses * c_months * 2.5 = total_rent) :=
by
  use 8
  sorry

end NUMINAMATH_GPT_horse_rent_problem_l469_46931


namespace NUMINAMATH_GPT_find_angle_l469_46964

theorem find_angle (A : ℝ) (deg_to_rad : ℝ) :
  (1/2 * Real.sin (A / 2 * deg_to_rad) + Real.cos (A / 2 * deg_to_rad) = 1) →
  (A = 360) :=
sorry

end NUMINAMATH_GPT_find_angle_l469_46964


namespace NUMINAMATH_GPT_muffin_cost_ratio_l469_46982

theorem muffin_cost_ratio (m b : ℝ) 
  (h1 : 5 * m + 4 * b = 20)
  (h2 : 3 * (5 * m + 4 * b) = 60)
  (h3 : 3 * m + 18 * b = 60) :
  m / b = 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_muffin_cost_ratio_l469_46982


namespace NUMINAMATH_GPT_minimum_distance_square_l469_46938

/-- Given the equation of a circle centered at (2,3) with radius 1, find the minimum value of 
the function z = x^2 + y^2 -/
theorem minimum_distance_square (x y : ℝ) 
  (h : (x - 2)^2 + (y - 3)^2 = 1) : ∃ (z : ℝ), z = x^2 + y^2 ∧ z = 14 - 2 * Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_minimum_distance_square_l469_46938


namespace NUMINAMATH_GPT_right_triangle_acute_angle_le_45_l469_46945

theorem right_triangle_acute_angle_le_45
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hright : a^2 + b^2 = c^2):
  ∃ θ φ : ℝ, θ + φ = 90 ∧ (θ ≤ 45 ∨ φ ≤ 45) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_acute_angle_le_45_l469_46945


namespace NUMINAMATH_GPT_barrel_capacity_is_16_l469_46922

noncomputable def capacity_of_barrel (midway_tap_rate bottom_tap_rate used_bottom_tap_early_time assistant_use_time : Nat) : Nat :=
  let midway_draw := used_bottom_tap_early_time / midway_tap_rate
  let bottom_draw_assistant := assistant_use_time / bottom_tap_rate
  let total_extra_draw := midway_draw + bottom_draw_assistant
  2 * total_extra_draw

theorem barrel_capacity_is_16 :
  capacity_of_barrel 6 4 24 16 = 16 :=
by
  sorry

end NUMINAMATH_GPT_barrel_capacity_is_16_l469_46922


namespace NUMINAMATH_GPT_length_of_nylon_cord_l469_46978

-- Definitions based on the conditions
def tree : ℝ := 0 -- Tree as the center point (assuming a 0 for simplicity)
def distance_ran : ℝ := 30 -- Dog ran approximately 30 feet

-- The theorem to prove
theorem length_of_nylon_cord : (distance_ran / 2) = 15 := by
  -- Assuming the dog ran along the diameter of the circle
  -- and the length of the cord is the radius of that circle.
  sorry

end NUMINAMATH_GPT_length_of_nylon_cord_l469_46978


namespace NUMINAMATH_GPT_expected_value_is_correct_l469_46904

noncomputable def expected_winnings : ℝ :=
  (1/12 : ℝ) * (9 + 8 + 7 + 6 + 5 + 1 + 2 + 3 + 4 + 5 + 6 + 7)

theorem expected_value_is_correct : expected_winnings = 5.25 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_correct_l469_46904


namespace NUMINAMATH_GPT_distance_to_Tianbo_Mountain_l469_46970

theorem distance_to_Tianbo_Mountain : ∀ (x y : ℝ), 
  (x ≠ 0) ∧ 
  (y = 3) ∧ 
  (∀ v, v = (4 * y + x) * ((2 * x - 8) / v)) ∧ 
  (2 * (y * x) = 8 * y + x^2 - 4 * x) 
  → 
  (x + y = 9) := 
by
  sorry

end NUMINAMATH_GPT_distance_to_Tianbo_Mountain_l469_46970


namespace NUMINAMATH_GPT_abc_inequality_l469_46927

theorem abc_inequality (a b c : ℝ) : a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end NUMINAMATH_GPT_abc_inequality_l469_46927


namespace NUMINAMATH_GPT_rowing_upstream_speed_l469_46915

theorem rowing_upstream_speed (V_m V_down V_up V_s : ℝ) 
  (hVm : V_m = 40) 
  (hVdown : V_down = 60) 
  (hVdown_eq : V_down = V_m + V_s) 
  (hVup_eq : V_up = V_m - V_s) : 
  V_up = 20 := 
by
  sorry

end NUMINAMATH_GPT_rowing_upstream_speed_l469_46915


namespace NUMINAMATH_GPT_find_unique_f_l469_46934

theorem find_unique_f (f : ℝ → ℝ) (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f (x) * f (y * z) + 1) : 
    ∀ x : ℝ, f x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_unique_f_l469_46934


namespace NUMINAMATH_GPT_range_of_k_l469_46908

noncomputable section

open Classical

variables {A B C k : ℝ}

def is_acute_triangle (A B C : ℝ) := A < 90 ∧ B < 90 ∧ C < 90

theorem range_of_k (hA : A = 60) (hBC : BC = 6) (h_acute : is_acute_triangle A B C) : 
  2 * Real.sqrt 3 < k ∧ k < 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_range_of_k_l469_46908


namespace NUMINAMATH_GPT_last_years_rate_per_mile_l469_46973

-- Definitions from the conditions
variables (m : ℕ) (x : ℕ)

-- Condition 1: This year, walkers earn $2.75 per mile
def amount_per_mile_this_year : ℝ := 2.75

-- Condition 2: Last year's winner collected $44
def last_years_total_amount : ℕ := 44

-- Condition 3: Elroy will walk 5 more miles than last year's winner
def elroy_walks_more_miles (m : ℕ) : ℕ := m + 5

-- The main goal is to prove that last year's rate per mile was $4 given the conditions
theorem last_years_rate_per_mile (h1 : last_years_total_amount = m * x)
  (h2 : last_years_total_amount = (elroy_walks_more_miles m) * amount_per_mile_this_year) :
  x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_last_years_rate_per_mile_l469_46973


namespace NUMINAMATH_GPT_min_odd_integers_l469_46933

theorem min_odd_integers 
  (a b c d e f : ℤ)
  (h1 : a + b = 30)
  (h2 : c + d = 15)
  (h3 : e + f = 17)
  (h4 : c + d + e + f = 32) :
  ∃ n : ℕ, (n = 2) ∧ (∃ odd_count, 
  odd_count = (if (a % 2 = 0) then 0 else 1) + 
                     (if (b % 2 = 0) then 0 else 1) + 
                     (if (c % 2 = 0) then 0 else 1) + 
                     (if (d % 2 = 0) then 0 else 1) + 
                     (if (e % 2 = 0) then 0 else 1) + 
                     (if (f % 2 = 0) then 0 else 1) ∧
  odd_count = 2) := sorry

end NUMINAMATH_GPT_min_odd_integers_l469_46933


namespace NUMINAMATH_GPT_lex_coins_total_l469_46923

def value_of_coins (dimes quarters : ℕ) : ℕ :=
  10 * dimes + 25 * quarters

def more_quarters_than_dimes (dimes quarters : ℕ) : Prop :=
  quarters > dimes

theorem lex_coins_total (dimes quarters : ℕ) (h : value_of_coins dimes quarters = 265) (h_more : more_quarters_than_dimes dimes quarters) : dimes + quarters = 13 :=
sorry

end NUMINAMATH_GPT_lex_coins_total_l469_46923


namespace NUMINAMATH_GPT_intersection_complement_l469_46921

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}

theorem intersection_complement :
  A ∩ (U \ B) = {4, 5} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l469_46921


namespace NUMINAMATH_GPT_undefined_value_of_expression_l469_46939

theorem undefined_value_of_expression (a : ℝ) : (a^3 - 8 = 0) → (a = 2) := by
  sorry

end NUMINAMATH_GPT_undefined_value_of_expression_l469_46939


namespace NUMINAMATH_GPT_slices_with_all_toppings_l469_46989

theorem slices_with_all_toppings (p m o a b c x total : ℕ) 
  (pepperoni_slices : p = 8)
  (mushrooms_slices : m = 12)
  (olives_slices : o = 14)
  (total_slices : total = 16)
  (inclusion_exclusion : p + m + o - a - b - c - 2 * x = total) :
  x = 4 := 
by
  rw [pepperoni_slices, mushrooms_slices, olives_slices, total_slices] at inclusion_exclusion
  sorry

end NUMINAMATH_GPT_slices_with_all_toppings_l469_46989


namespace NUMINAMATH_GPT_determine_X_with_7_gcd_queries_l469_46955

theorem determine_X_with_7_gcd_queries : 
  ∀ (X : ℕ), (X ≤ 100) → ∃ (f : Fin 7 → ℕ × ℕ), 
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) ∧ (∃ (Y : Fin 7 → ℕ), 
      (∀ i, Y i = Nat.gcd (X + (f i).1) (f i).2) → 
        (∀ (X' : ℕ), (X' ≤ 100) → ((∀ i, Y i = Nat.gcd (X' + (f i).1) (f i).2) → X' = X))) :=
sorry

end NUMINAMATH_GPT_determine_X_with_7_gcd_queries_l469_46955


namespace NUMINAMATH_GPT_mike_taller_than_mark_l469_46925

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end NUMINAMATH_GPT_mike_taller_than_mark_l469_46925


namespace NUMINAMATH_GPT_mean_equal_implication_l469_46976

theorem mean_equal_implication (y : ℝ) :
  (7 + 10 + 15 + 23 = 55) →
  (55 / 4 = 13.75) →
  (18 + y + 30 = 48 + y) →
  (48 + y) / 3 = 13.75 →
  y = -6.75 :=
by 
  intros h1 h2 h3 h4
  -- The steps would be applied here to prove y = -6.75
  sorry

end NUMINAMATH_GPT_mean_equal_implication_l469_46976


namespace NUMINAMATH_GPT_gopi_salary_turbans_l469_46932

-- Define the question and conditions as statements
def total_salary (turbans : ℕ) : ℕ := 90 + 30 * turbans
def servant_receives : ℕ := 60 + 30
def fraction_annual_salary : ℚ := 3 / 4

-- The theorem statement capturing the equivalent proof problem
theorem gopi_salary_turbans (T : ℕ) 
  (salary_eq : total_salary T = 90 + 30 * T)
  (servant_eq : servant_receives = 60 + 30)
  (fraction_eq : fraction_annual_salary = 3 / 4)
  (received_after_9_months : ℚ) :
  fraction_annual_salary * (90 + 30 * T : ℚ) = received_after_9_months → 
  received_after_9_months = 90 →
  T = 1 :=
sorry

end NUMINAMATH_GPT_gopi_salary_turbans_l469_46932


namespace NUMINAMATH_GPT_polynomial_remainder_l469_46946

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def rem (x : ℝ) : ℝ := 14 * x - 14

theorem polynomial_remainder :
  ∀ x : ℝ, p x % d x = rem x := 
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l469_46946


namespace NUMINAMATH_GPT_ratio_Jake_sister_l469_46910

theorem ratio_Jake_sister (Jake_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (expected_ratio : ℕ) :
  Jake_weight = 113 →
  total_weight = 153 →
  weight_loss = 33 →
  expected_ratio = 2 →
  (Jake_weight - weight_loss) / (total_weight - Jake_weight) = expected_ratio :=
by
  intros hJake hTotal hLoss hRatio
  sorry

end NUMINAMATH_GPT_ratio_Jake_sister_l469_46910


namespace NUMINAMATH_GPT_ratio_of_M_to_N_l469_46999

theorem ratio_of_M_to_N 
  (M Q P N : ℝ) 
  (h1 : M = 0.4 * Q) 
  (h2 : Q = 0.25 * P) 
  (h3 : N = 0.75 * P) : 
  M / N = 2 / 15 := 
sorry

end NUMINAMATH_GPT_ratio_of_M_to_N_l469_46999


namespace NUMINAMATH_GPT_abs_w_unique_l469_46972

theorem abs_w_unique (w : ℂ) (h : w^2 - 6 * w + 40 = 0) : ∃! x : ℝ, x = Complex.abs w ∧ x = Real.sqrt 40 := by
  sorry

end NUMINAMATH_GPT_abs_w_unique_l469_46972


namespace NUMINAMATH_GPT_consecutive_integer_cubes_sum_l469_46956

theorem consecutive_integer_cubes_sum : 
  ∀ (a : ℕ), 
  (a > 2) → 
  (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2)) →
  ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3) = 224 :=
by
  intro a ha h
  sorry

end NUMINAMATH_GPT_consecutive_integer_cubes_sum_l469_46956


namespace NUMINAMATH_GPT_rented_room_percentage_l469_46909

theorem rented_room_percentage (total_rooms : ℕ) (h1 : 3 * total_rooms / 4 = 3 * total_rooms / 4) 
                               (h2 : 3 * total_rooms / 5 = 3 * total_rooms / 5) 
                               (h3 : 2 * (3 * total_rooms / 5) / 3 = 2 * (3 * total_rooms / 5) / 3) :
  (1 * (3 * total_rooms / 5) / 5) / (1 * total_rooms / 4) * 100 = 80 := by
  sorry

end NUMINAMATH_GPT_rented_room_percentage_l469_46909


namespace NUMINAMATH_GPT_smallest_even_n_l469_46911

theorem smallest_even_n (n : ℕ) :
  (∃ n, 0 < n ∧ n % 2 = 0 ∧ (∀ k, 1 ≤ k → k ≤ n / 2 → k = 2213 ∨ k = 3323 ∨ k = 6121) ∧ (2^k * (k!)) % (2213 * 3323 * 6121) = 0) → n = 12242 :=
sorry

end NUMINAMATH_GPT_smallest_even_n_l469_46911


namespace NUMINAMATH_GPT_parabola_has_one_x_intercept_l469_46930

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end NUMINAMATH_GPT_parabola_has_one_x_intercept_l469_46930


namespace NUMINAMATH_GPT_no_solution_for_xx_plus_yy_eq_9z_l469_46971

theorem no_solution_for_xx_plus_yy_eq_9z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ¬ (x^x + y^y = 9^z) :=
sorry

end NUMINAMATH_GPT_no_solution_for_xx_plus_yy_eq_9z_l469_46971


namespace NUMINAMATH_GPT_greatest_third_side_l469_46962

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end NUMINAMATH_GPT_greatest_third_side_l469_46962


namespace NUMINAMATH_GPT_train_speed_l469_46985

theorem train_speed (distance time : ℕ) (h1 : distance = 180) (h2 : time = 9) : distance / time = 20 := by
  sorry

end NUMINAMATH_GPT_train_speed_l469_46985


namespace NUMINAMATH_GPT_prism_base_shape_l469_46940

theorem prism_base_shape (n : ℕ) (hn : 3 * n = 12) : n = 4 := by
  sorry

end NUMINAMATH_GPT_prism_base_shape_l469_46940


namespace NUMINAMATH_GPT_integer_roots_condition_l469_46969

theorem integer_roots_condition (a : ℝ) (h_pos : 0 < a) :
  (∀ x y : ℤ, (a ^ 2 * x ^ 2 + a * x + 1 - 13 * a ^ 2 = 0) ∧ (a ^ 2 * y ^ 2 + a * y + 1 - 13 * a ^ 2 = 0)) ↔
  (a = 1 ∨ a = 1/3 ∨ a = 1/4) :=
by sorry

end NUMINAMATH_GPT_integer_roots_condition_l469_46969
