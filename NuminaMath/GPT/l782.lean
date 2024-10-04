import Mathlib

namespace faye_pencils_l782_782846

theorem faye_pencils (rows : ℕ) (pencils_per_row : ℕ) (h_rows : rows = 30) (h_pencils_per_row : pencils_per_row = 24) :
  rows * pencils_per_row = 720 :=
by
  sorry

end faye_pencils_l782_782846


namespace sum_of_roots_l782_782976

theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hroots : ∀ x : ℝ, x^2 - p*x + 2*q = 0) :
  p + q = p :=
by sorry

end sum_of_roots_l782_782976


namespace power_of_power_evaluation_l782_782452

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l782_782452


namespace eval_exp_l782_782423

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l782_782423


namespace tan_domain_correct_l782_782684

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4}

def is_domain_correct : Prop :=
  ∀ x : ℝ, x ∈ domain_tan ↔ (∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4)

-- Statement of the problem in Lean 4
theorem tan_domain_correct : is_domain_correct :=
  sorry

end tan_domain_correct_l782_782684


namespace number_of_bowls_l782_782721

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l782_782721


namespace entrance_ticket_cost_l782_782786

theorem entrance_ticket_cost
  (students teachers : ℕ)
  (total_cost : ℕ)
  (students_count : students = 20)
  (teachers_count : teachers = 3)
  (cost : total_cost = 115) :
  total_cost / (students + teachers) = 5 := by
  sorry

end entrance_ticket_cost_l782_782786


namespace segment_length_is_15_l782_782959

theorem segment_length_is_15 : 
  ∀ (x : ℝ), 
  ∀ (y1 y2 : ℝ), 
  x = 3 → 
  y1 = 5 → 
  y2 = 20 → 
  abs (y2 - y1) = 15 := by 
sorry

end segment_length_is_15_l782_782959


namespace prob_2022_2023_l782_782571

theorem prob_2022_2023 (n : ℤ) (h : (n - 2022)^2 + (2023 - n)^2 = 1) : (n - 2022) * (2023 - n) = 0 :=
sorry

end prob_2022_2023_l782_782571


namespace man_l782_782343

variable (V_m V_stream V_up V_down : ℝ)

def proof_problem : Prop :=
  (V_up = 8) ∧
  (V_stream = 3.5) →
  (V_m = 11.5) →
  (V_down = V_m + V_stream) →
  (V_down = 15)

theorem man's_speed_downstream
  (h1 : V_up = 8)
  (h2 : V_stream = 3.5)
  (hVm : V_m = 11.5)
  (hVd : V_down = V_m + V_stream) :
  V_down = 15 :=
begin
  sorry
end

end man_l782_782343


namespace no_infinite_family_of_lines_exists_l782_782840

theorem no_infinite_family_of_lines_exists :
  ¬ ∃ (l : ℕ → ℝ × ℝ → ℝ), 
    (∀ n : ℕ, (1, 1) ∈ l n) ∧
    (∀ n : ℕ, let k_n := (snd (l n) - fst (l n)) / (1 : ℝ - fst (l n)),
                  a_n := 1 - 1 / k_n,
                  b_n := 1 - k_n in 
                  (snd (l (n+1)) - fst (l (n+1))) / (1 : ℝ - fst (l (n+1))) = a_n - b_n) ∧
    (∀ n : ℕ, k_n k_(n+1) ≥ 0)
:=
begin
  sorry
end

end no_infinite_family_of_lines_exists_l782_782840


namespace find_k_l782_782016

variables {α : Type*} [CommRing α]

theorem find_k (a b c : α) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c :=
by sorry

end find_k_l782_782016


namespace ratio_of_investments_l782_782362

theorem ratio_of_investments (A B : ℤ) (h1 : A = 2000) (h2 : B = 2000) (h3 : 0.05 * A + 0.06 * B = 520) :
  A / B = 1 :=
by {
  sorry
}

end ratio_of_investments_l782_782362


namespace tiger_distance_traveled_l782_782361

theorem tiger_distance_traveled :
  let distance1 := 25 * 1
  let distance2 := 35 * 2
  let distance3 := 20 * 1.5
  let distance4 := 10 * 1
  let distance5 := 50 * 0.5
  distance1 + distance2 + distance3 + distance4 + distance5 = 160 := by
sorry

end tiger_distance_traveled_l782_782361


namespace find_alpha_l782_782693

noncomputable def isochronous_growth (k α x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  y₁ = k * x₁^α ∧
  y₂ = k * x₂^α ∧
  x₂ = 16 * x₁ ∧
  y₂ = 8 * y₁

theorem find_alpha (k x₁ x₂ y₁ y₂ : ℝ) (h : isochronous_growth k (3/4) x₁ x₂ y₁ y₂) : 3/4 = 3/4 :=
by 
  sorry

end find_alpha_l782_782693


namespace eval_expr_l782_782427

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l782_782427


namespace cubic_solutions_l782_782470

theorem cubic_solutions (x : ℝ) :
  (∃ x, (18 * x - 3) ^ (1 / 3 : ℝ) + (12 * x + 3) ^ (1 / 3 : ℝ) = 5 * x ^ (1 / 3 : ℝ)) ↔
    x = 0 ∨ x = (-27 + real.sqrt 18477) / 1026 ∨ x = (-27 - real.sqrt 18477) / 1026 :=
begin
  sorry
end

end cubic_solutions_l782_782470


namespace today_is_Thursday_l782_782813

-- Define days of the week as an inductive type
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq, Repr

open Day

-- Define who lies on which days
def A_lies (d : Day) : Prop :=
  d = Monday ∨ d = Tuesday ∨ d = Wednesday

def B_lies (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday

-- Define the statements made by A and B
def A_says (d : Day) : Prop :=
  d = Monday ∧ A_lies Sunday ∨ d = Thursday ∧ A_lies Wednesday

def B_says (d : Day) : Prop :=
  d = Thursday ∧ B_lies Wednesday ∨ d = Sunday ∧ B_lies Saturday

-- The main statement we need to prove:
theorem today_is_Thursday : ∃ d : Day, A_says d ∧ B_says d ∧ (d = Thursday) :=
by
  exists Thursday
  split
  . sorry -- Prove this step

  . split
  . sorry -- Prove this step

  . rfl -- d = Thursday

end today_is_Thursday_l782_782813


namespace interior_angle_second_quadrant_l782_782537

theorem interior_angle_second_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin α * Real.tan α < 0) : 
  π / 2 < α ∧ α < π :=
by
  sorry

end interior_angle_second_quadrant_l782_782537


namespace rabbit_shape_area_l782_782818

theorem rabbit_shape_area (A_ear : ℝ) (h1 : A_ear = 10) (h2 : A_ear = (1/8) * A_total) :
  A_total = 80 :=
by
  sorry

end rabbit_shape_area_l782_782818


namespace logarithmic_solution_l782_782409

theorem logarithmic_solution (x : ℝ) (h : log (10 : ℝ) / log (x^2) + log 10 / log (x^4) + log 10 / log (9*x^5) = 0) : 
  1 / x ^ 18 = 9 ^ 93 := 
by 
  sorry

end logarithmic_solution_l782_782409


namespace value_at_zero_eq_sixteen_l782_782467

-- Define the polynomial P(x)
def P (x : ℚ) : ℚ := x ^ 4 - 20 * x ^ 2 + 16

-- Theorem stating the value of P(0)
theorem value_at_zero_eq_sixteen :
  P 0 = 16 :=
by
-- We know the polynomial P(x) is x^4 - 20x^2 + 16
-- When x = 0, P(0) = 0^4 - 20 * 0^2 + 16 = 16
sorry

end value_at_zero_eq_sixteen_l782_782467


namespace find_k_l782_782298

theorem find_k :
  ∀ k : ℝ, (1 / 2) ^ 25 * (1 / 81) ^ k = 1 / 18 ^ 25 → k = -12.5 :=
by
  intro k h
  sorry

end find_k_l782_782298


namespace solution_set_of_inequality_l782_782055

noncomputable def f_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f y < f x 

noncomputable def point_A_on_graph (f : ℝ → ℝ) := f (-1) = 3

noncomputable def point_B_on_graph (f : ℝ → ℝ) := f 1 = 1

theorem solution_set_of_inequality (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h_decreasing : f_decreasing f)
  (h_point_A : point_A_on_graph f)
  (h_point_B : point_B_on_graph f) :
  {x : ℝ | |2008 * f_inv (Real.log x / Real.log 2)| < 2008} = Ioo 2 8 :=
begin
  sorry -- Proof omitted
end

end solution_set_of_inequality_l782_782055


namespace choose_3_out_of_10_l782_782117

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 := by
  sorry

end choose_3_out_of_10_l782_782117


namespace circle_radii_touch_ext_l782_782751

theorem circle_radii_touch_ext (A B C O1 O2 : Point) (r1 r2 : ℝ) 
  (h_touch_ext : circles_touch_ext A O1 O2) (h_AB : dist A B = 8) (h_AC : dist A C = 6) 
  (h_tangent : is_tangent B C O1 O2) : 
  r1 = 15 / 4 ∧ r2 = 20 / 3 := 
by 
  sorry

end circle_radii_touch_ext_l782_782751


namespace sqrt_fraction_expression_eq_one_l782_782688

theorem sqrt_fraction_expression_eq_one :
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 6) = 1 := 
by
  sorry

end sqrt_fraction_expression_eq_one_l782_782688


namespace roots_of_quadratic_and_square_sum_l782_782189

variables {p q : ℝ}

theorem roots_of_quadratic_and_square_sum :
  (∀ p q, (∃r s, r = p ∧ s = q ∧ (r + s = 5) ∧ (r * s = 6))) → p^2 + q^2 = 13 :=
by
  intro h
  obtain ⟨r, s, hr, hs, hr_sum, hr_prod⟩ := h p q
  rw [←hr, ←hs] at *
  have : p + q = 5 := hr_sum
  have : p * q = 6 := hr_prod
  calc
    p^2 + q^2 = (p + q)^2 - 2 * (p * q) : by ring
          ... = 5^2 - 2 * 6 : by rw [this, this]
          ... = 25 - 12 : by norm_num
          ... = 13 : by norm_num

end roots_of_quadratic_and_square_sum_l782_782189


namespace sum_of_possible_values_l782_782962

theorem sum_of_possible_values (θ φ : ℝ) 
  (h : (cos θ)^6 / (cos φ)^2 + (sin θ)^6 / (sin φ)^2 = 1) : 
  (sin φ)^6 / (sin θ)^2 + (cos φ)^6 / (cos θ)^2 = 1 :=
  sorry

end sum_of_possible_values_l782_782962


namespace evaluate_power_l782_782459

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l782_782459


namespace plane_intersects_interior_l782_782349

theorem plane_intersects_interior (vertices : Finset (ℝ × ℝ × ℝ)) 
  (h_dims : vertices.card = 8)
  (dim1 dim2 dim3 : ℝ) (h_dim1 : dim1 = 3) (h_dim2 : dim2 = 4) (h_dim3 : dim3 = 5) :
  (∃ (v1 v2 v3 : ℝ × ℝ × ℝ), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ 
    Probability (v1, v2, v3) = 4 / 7) :=
sorry

end plane_intersects_interior_l782_782349


namespace range_of_a_l782_782076

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) 
  (h2 : log_base a (a^2 + 1) < log_base a (2 * a))
  (h3 : log_base a (2 * a) < 0) : a ∈ Set.Ioo (0.5) 1 := 
sorry

end range_of_a_l782_782076


namespace calculate_product1_calculate_square_l782_782393

theorem calculate_product1 : 100.2 * 99.8 = 9999.96 :=
by
  sorry

theorem calculate_square : 103^2 = 10609 :=
by
  sorry

end calculate_product1_calculate_square_l782_782393


namespace arrangement_of_books_l782_782960

theorem arrangement_of_books : 
  let history_books := 4
  let science_books := 6
  let ways_to_arrange_on_shelf := (2! * 4! * 6!) in
  ways_to_arrange_on_shelf = 34560 :=
by 
  sorry

end arrangement_of_books_l782_782960


namespace part_1_part_2_l782_782940

variable (a b t : ℝ)
variable (x y : ℝ)

def E := x^2 / a^2 - y^2 / b^2 = 1
def A := (2 : ℝ, 0 : ℝ)
def P := (4 : ℝ, 0 : ℝ)
def L (x : ℝ) := y = b / a * (x - 4)
def dist (p q : ℝ × ℝ) := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem part_1 (a_pos : 0 < a) (b_pos : 0 < b)
(h_vertex : (2:ℝ, 0:ℝ) ∈ E) 
(h_dist : dist A (4, 0) = 2 * real.sqrt 5 / 5)
: E := 
begin 
  sorry
end

theorem part_2 (h_intersect : ∀ t, intersects (L t) E)
(h_angle : ∀ M N, angle M Q P = angle N Q P)
: t = 1 := 
begin
  sorry
end

end part_1_part_2_l782_782940


namespace part1_part2_l782_782556

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x + 5

def domain_range_condition (a : ℝ) := 
  (∀ x, 1 ≤ x ∧ x ≤ a → 1 ≤ f x a ∧ f x a ≤ a)

def decreasing_interval_condition (a : ℝ) :=
  (∀ x, x ∈ Icc (-∞ : ℝ) 2 → f x a ≤ f 2 a)

def bounded_difference_condition (a : ℝ) :=
  (∀ x1 x2, x1 ∈ Icc 1 (a + 1) → x2 ∈ Icc 1 (a + 1) → abs (f x1 a - f x2 a) ≤ 4)

theorem part1 (a : ℝ) (h : domain_range_condition a) : a = 2 := 
  sorry

theorem part2 (a : ℝ) 
  (h_decreasing : decreasing_interval_condition a) 
  (h_bounded : bounded_difference_condition a) : 
  2 ≤ a ∧ a ≤ 3 := 
  sorry

end part1_part2_l782_782556


namespace equal_opposite_planar_angles_l782_782217

def inscribed_in_tetrahedral_angle (P A B C D K L M N : Point) : Prop :=
  -- A predicate stating that a sphere is inscribed in the tetrahedral angle PABCD with tangency points defined
  inscribed_in_tetrahedral_angle P A B C D K L M N 

theorem equal_opposite_planar_angles 
  (P A B C D : Point) (K L M N : Point) 
  (h_inscribed : inscribed_in_tetrahedral_angle P A B C D K L M N) 
  : 
  angle A P D + angle C P D = angle B P C + angle A P D := 
sorry

end equal_opposite_planar_angles_l782_782217


namespace number_of_bowls_l782_782710

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l782_782710


namespace buratino_loss_l782_782374

def buratino_dollars_lost (x y : ℕ) : ℕ := 5 * y - 3 * x

theorem buratino_loss :
  ∃ (x y : ℕ), x + y = 50 ∧ 3 * y - 2 * x = 0 ∧ buratino_dollars_lost x y = 10 :=
by {
  sorry
}

end buratino_loss_l782_782374


namespace sum_of_first_five_terms_arithmetic_sequence_l782_782239

theorem sum_of_first_five_terms_arithmetic_sequence (a4 a5 a6 : Int) (d : Int) (a3 a2 a1 : Int) :
  a4 = 11 →
  a5 = 15 →
  a6 = 19 →
  d = a5 - a4 →
  a3 = a4 - d →
  a2 = a3 - d →
  a1 = a2 - d →
  a1 + a2 + a3 + a4 + a5 = 35 :=
by
  intros h1 h2 h3 hd ha3 ha2 ha1
  rw [h1, h2, h3] at *
  rw [hd, ha3, ha2, ha1]
  sorry

end sum_of_first_five_terms_arithmetic_sequence_l782_782239


namespace sin_cos_inequality_l782_782672

theorem sin_cos_inequality (x : ℝ) : 
    1 ≤ sin x ^ 10 + 10 * (sin x ^ 2 * cos x ^ 2) + cos x ^ 10 ∧ 
    sin x ^ 10 + 10 * (sin x ^ 2 * cos x ^ 2) + cos x ^ 10 ≤ 41 / 16 :=
by sorry

end sin_cos_inequality_l782_782672


namespace tv_price_comparison_l782_782290

def area (width height : ℕ) : ℕ := width * height

def cost_per_square_inch (cost area : ℕ) : ℚ := cost.toRat / area.toRat

def price_difference (cost1 cost2 area1 area2 : ℕ) : ℚ :=
  cost_per_square_inch(cost1, area1) - cost_per_square_inch(cost2, area2)

theorem tv_price_comparison :
  price_difference 672 1152 (area 24 16) (area 48 32) = 1 := by
  sorry

end tv_price_comparison_l782_782290


namespace perimeter_triangle_ABC_is_correct_l782_782523

noncomputable def semicircle_perimeter_trianlge_ABC : ℝ :=
  let BE := (1 : ℝ)
  let EF := (24 : ℝ)
  let FC := (3 : ℝ)
  let BC := BE + EF + FC
  let r := EF / 2
  let x := 71.5
  let AB := x + BE
  let AC := x + FC
  AB + BC + AC

theorem perimeter_triangle_ABC_is_correct : semicircle_perimeter_trianlge_ABC = 175 := by
  sorry

end perimeter_triangle_ABC_is_correct_l782_782523


namespace common_divisors_90_105_l782_782957

def divisors (n : ℕ) : Finset ℤ :=
  (Finset.range (n+1)).filter (λ d, n % d = 0).image (λ d, d : ℤ ∈ d) ∪ -(Finset.range (n+1)).filter (λ d, n % d = 0).image (λ d, -(d : ℤ ∈ d))

def common_divisors_count (a b : ℕ) : Nat :=
  (divisors a) ∩ (divisors b).card

theorem common_divisors_90_105 : common_divisors_count 90 105 = 8 :=
by
  sorry

end common_divisors_90_105_l782_782957


namespace perfect_square_partition_l782_782869

open Nat

-- Define the condition of a number being a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- Define the main theorem statement
theorem perfect_square_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n+1)) → (A ∩ B = ∅) →
  ∃ a b ∈ A, a ≠ b ∧ is_perfect_square (a + b)
:= by
  sorry

end perfect_square_partition_l782_782869


namespace domain_f_monotonicity_f_range_a_l782_782551

noncomputable def f (a : ℝ) (x : ℝ) := (ln (a * x)) / (x + 1) - ln (a * x) + ln (x + 1)

-- Representation of the domain of f(x)
theorem domain_f {a : ℝ} (ha : a ≠ 0) : 
  (a > 0 → ∀ x, x > 0 → ∀ x, f a x = (0, +∞)) ∧
  (a < 0 → ∀ x, -1 < x ∧ x < 0 → ∀ x, f a x = (-1, 0)) :=
sorry

-- Representation of the monotonicity of f(x)
theorem monotonicity_f (a : ℝ) (ha : a ≠ 0) :
  ∃ x, 
  (a > 0 → (∀ x, x ∈ (0, 1/a) → ∃ x, increasing f a x) ∧ 
            (∀ x, x ∈ (1/a, +∞) → ∃ x, decreasing f a x)) ∧
  (-1 ≤ a ∧ a < 0 → ∀ x, x ∈ (-1, 0) → ∃ x, increasing f a x) ∧
  (a < -1 → (∀ x, x ∈ (-1, 1/a) → ∃ x, decreasing f a x) ∧ 
           (∀ x, x ∈ (1/a, 0) → ∃ x, increasing f a x)) :=
sorry

-- Representation of range of values of a when f(x) >= ln(2a)
theorem range_a (a : ℝ) (ha_pos : a > 0) :
  (∃ x, f a x ≥ ln (2 * a)) → (0 < a ∧ a ≤ 1) :=
sorry

end domain_f_monotonicity_f_range_a_l782_782551


namespace count_satisfying_numbers_l782_782488

def R (n : ℕ) : ℕ :=
(2 to 12).map (n % ·).sum

def delta (n k : ℕ) : ℤ :=
if n % k = k - 1 then -((k : ℤ) - 1) else 1

noncomputable def satisfies_condition (n : ℕ) : Prop :=
(2 to 12).sum (delta n) = 0

theorem count_satisfying_numbers : 
    (finset.filter satisfies_condition (finset.Icc 100 999)).card = 2 :=
sorry

end count_satisfying_numbers_l782_782488


namespace chickens_on_farm_are_120_l782_782707

-- Given conditions
def Number_of_hens : ℕ := 52
def Difference_hens_roosters : ℕ := 16

-- Define the number of roosters based on the conditions
def Number_of_roosters : ℕ := Number_of_hens + Difference_hens_roosters

-- The total number of chickens is the sum of hens and roosters
def Total_number_of_chickens : ℕ := Number_of_hens + Number_of_roosters

-- Prove that the total number of chickens is 120
theorem chickens_on_farm_are_120 : Total_number_of_chickens = 120 := by
  -- leave this part unimplemented for proof.
  -- The steps would involve computing the values based on definitions
  sorry

end chickens_on_farm_are_120_l782_782707


namespace count_four_digit_numbers_l782_782923

theorem count_four_digit_numbers :
  ∀ (a b c d : ℕ),
    a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} ∧ c ∈ {1, 2, 3, 4} ∧ d ∈ {1, 2, 3, 4} ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    a = min a (min b (min c d)) →
  (∃ n, n = 28) :=
by
  intros a b c d
  intro h
  have : ∃ n, n = 28 := sorry
  exact this

end count_four_digit_numbers_l782_782923


namespace total_snakes_l782_782777

def People (n : ℕ) : Prop := n = 59
def OnlyDogs (n : ℕ) : Prop := n = 15
def OnlyCats (n : ℕ) : Prop := n = 10
def OnlyCatsAndDogs (n : ℕ) : Prop := n = 5
def CatsDogsSnakes (n : ℕ) : Prop := n = 3

theorem total_snakes (n_people n_dogs n_cats n_catsdogs n_catdogsnsnakes : ℕ)
  (h_people : People n_people) 
  (h_onlyDogs : OnlyDogs n_dogs)
  (h_onlyCats : OnlyCats n_cats)
  (h_onlyCatsAndDogs : OnlyCatsAndDogs n_catsdogs)
  (h_catsDogsSnakes : CatsDogsSnakes n_catdogsnsnakes) :
  n_catdogsnsnakes >= 3 :=
by
  -- Proof goes here
  sorry

end total_snakes_l782_782777


namespace min_distance_circle_tangent_l782_782563

theorem min_distance_circle_tangent
  (P : ℝ × ℝ)
  (hP: 3 * P.1 + 4 * P.2 = 11) :
  ∃ d : ℝ, d = 11 / 5 := 
sorry

end min_distance_circle_tangent_l782_782563


namespace train_speed_including_stoppages_l782_782845

noncomputable def trainSpeedExcludingStoppages : ℝ := 45
noncomputable def stoppageTimePerHour : ℝ := 20 / 60 -- 20 minutes per hour converted to hours
noncomputable def runningTimePerHour : ℝ := 1 - stoppageTimePerHour

theorem train_speed_including_stoppages (speed : ℝ) (stoppage : ℝ) (running_time : ℝ) : 
  speed = 45 → stoppage = 20 / 60 → running_time = 1 - stoppage → 
  (speed * running_time) / 1 = 30 :=
by sorry

end train_speed_including_stoppages_l782_782845


namespace value_expression_eq_zero_l782_782167

theorem value_expression_eq_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
    a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 :=
by
  sorry

end value_expression_eq_zero_l782_782167


namespace bees_multiple_l782_782199

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end bees_multiple_l782_782199


namespace area_BFEC_l782_782601

variable (A B C D E F : Type)
variable (height_A_to_CD AB BE DF_AF : ℝ)
variable [Parallelogram ABCD]
variable [OnLine E BC]
variable [OnLine F AD]
variable [DF_AF = 1/3]
variable [AB = 12]
variable [height_A_to_CD = 7]
variable [BE = 4]

theorem area_BFEC : 
  let area_ABCD := AB * height_A_to_CD
  let area_ABF := (1/2) * (2/3 * 12) * height_A_to_CD
  let area_CED := (1/2) * (AB - BE) * height_A_to_CD
  area_BFEC = area_ABCD - (area_ABF + area_CED) → area_BFEC = 28 :=
by
  sorry

end area_BFEC_l782_782601


namespace sum_abs_coeffs_expansion_l782_782883

theorem sum_abs_coeffs_expansion (x : ℝ) :
  (|1 - 0 * x| + |1 - 3 * x| + |1 - 3^2 * x^2| + |1 - 3^3 * x^3| + |1 - 3^4 * x^4| + |1 - 3^5 * x^5| = 1024) :=
sorry

end sum_abs_coeffs_expansion_l782_782883


namespace coefficient_x7_expansion_l782_782875

theorem coefficient_x7_expansion :
  (∀ (x : ℂ), coeff (1 - x + 2 * x^2)^5 x^7 = -200) :=
by sorry

end coefficient_x7_expansion_l782_782875


namespace max_real_roots_l782_782406

noncomputable def P (x : ℝ) (n : ℕ) : ℝ := 
  ∑ i in finset.range (2 * n + 2), x ^ i

theorem max_real_roots (n : ℕ) : 
  (∑ i in finset.range (2 * n + 2), (1 : ℝ) ^ i = 2 * n + 2 ↔ (2 : ℝ) * n + 1 = 0) ∧ 
  (∑ i in finset.range (2 * n + 2), (-1 : ℝ) ^ i = 2 * (n + 1)) → 
  ∃ x : ℝ, P x n = 0 ∧ (x = 1 ∨ x = -1) → 
  x = -1 := 
begin
  sorry,
end

end max_real_roots_l782_782406


namespace evaluate_exponent_l782_782447

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l782_782447


namespace nearest_divisible_by_11_l782_782880

theorem nearest_divisible_by_11 (n : ℕ) (h1 : n = 457) (h2 : ∃ k : ℤ, k * 11 = n): ∃ m : ℕ, m = 462 ∧ (m % 11 = 0) ∧ (abs (m - n) = abs (462 - n)) :=
by
  sorry

end nearest_divisible_by_11_l782_782880


namespace x_sq_plus_3x_eq_1_l782_782536

theorem x_sq_plus_3x_eq_1 (x : ℝ) (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : x^2 + 3*x = 1 :=
sorry

end x_sq_plus_3x_eq_1_l782_782536


namespace find_B_l782_782228

theorem find_B (A B C : ℝ) (h : ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -1 → 
    2 / ((x-7)*(x+1)^2) = A / (x-7) + B / (x+1) + C / (x+1)^2) : 
  B = 1 / 16 :=
sorry

end find_B_l782_782228


namespace sequence_seventh_term_l782_782144

theorem sequence_seventh_term : 
  let seq := (fun n => √(3 * n - 1)) in
  seq 7 = 2 * √5 :=
by sorry

end sequence_seventh_term_l782_782144


namespace sum_of_digits_2000_pow_2000_l782_782641

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_2000_pow_2000:
  sum_of_digits (sum_of_digits (sum_of_digits (2000 ^ 2000))) = 4 :=
by
  sorry

end sum_of_digits_2000_pow_2000_l782_782641


namespace hyperbola_asymptote_solution_l782_782086

theorem hyperbola_asymptote_solution (b : ℝ) (hb : b > 0)
  (h_asym : ∀ x y, (∀ y : ℝ, y = (1 / 2) * x ∨ y = - (1 / 2) * x) → (x^2 / 4 - y^2 / b^2 = 1)) :
  b = 1 :=
sorry

end hyperbola_asymptote_solution_l782_782086


namespace candy_distribution_invariant_l782_782102

-- Definitions to reflect the problem conditions
def candies : ℕ := 1000
def children : Type
def Child (g: children) : Prop
constant is_boy : children → Prop
constant is_girl : children → Prop
constant round_up : ℕ → ℕ
constant round_down : ℕ → ℕ

-- Invariant property to be demonstrated
theorem candy_distribution_invariant (k : ℕ) (c : children) (g : Child c) : 
  ∀ c1 c2 : list children,
  (∀ (x : children), (x ∈ c1 ∧ x ∈ c2) → c1 = c2 ) → 
  let boys := filter is_boy (c1.append c2) 
  in 
  (∀ turn order t1 t2 : list children, 
  (∀ (x : children), ((x ∈ t1.append g) → 
   candies - round_up (candies / k) = 
   candies - round_down (candies / k)))
  ) → 
  (∑ x in boys, round_up (candies / k)) = 
  (∑ x in boys, round_up (candies / k)) :=
begin
  sorry
end

end candy_distribution_invariant_l782_782102


namespace not_prime_n4_plus_2n2_plus_3_l782_782216

theorem not_prime_n4_plus_2n2_plus_3 (n : ℤ) : ¬ prime (n^4 + 2 * n^2 + 3) := 
sorry

end not_prime_n4_plus_2n2_plus_3_l782_782216


namespace clara_biked_more_l782_782989

def clara_speed : ℕ := 18
def denise_speed : ℕ := 16
def race_duration : ℕ := 5

def clara_distance := clara_speed * race_duration
def denise_distance := denise_speed * race_duration
def distance_difference := clara_distance - denise_distance

theorem clara_biked_more : distance_difference = 10 := by
  sorry

end clara_biked_more_l782_782989


namespace sum_reciprocals_lt_two_l782_782628

theorem sum_reciprocals_lt_two (N : ℕ) (n : ℕ) (x : ℕ → ℕ) 
  (h1 : ∀ i, i < n → x i < N) 
  (h2 : ∀ i j, i < n → j < n → i ≠ j → Nat.lcm (x i) (x j) > N) : 
    ∑ i in Finset.range n, (1 : ℚ) / x i < 2 := 
by 
  sorry

end sum_reciprocals_lt_two_l782_782628


namespace lowest_point_in_fourth_quadrant_l782_782093

theorem lowest_point_in_fourth_quadrant (k : ℝ) (h : k < -1) :
    let x := - (k + 1) / 2
    let y := (4 * k - (k + 1) ^ 2) / 4
    y < 0 ∧ x > 0 :=
by
  let x := - (k + 1) / 2
  let y := (4 * k - (k + 1) ^ 2) / 4
  sorry

end lowest_point_in_fourth_quadrant_l782_782093


namespace find_k_l782_782695

-- We define the conditions
def line := λ k x, k * x - 2
def parabola := λ y, y ^ 2 = 8 * (y + 2 * exp (2 / y))

-- We state the main proof goal
theorem find_k : 
  (∀ x1 x2 : ℝ, (line k x1 = parabola y) ∧ (line k x2 = parabola y)  ∧ ((x1 + x2) / 2 = 2)) → k = 2 := 
  sorry

end find_k_l782_782695


namespace peter_savings_time_l782_782660

def required_euros : ℝ := 5000.0
def current_savings_usd : ℝ := 2900.0
def exchange_rate : ℝ := 1.10
def monthly_saving_first_month : ℝ := 700.0
def savings_decrease_percentage : ℝ := 0.80
def transaction_fee_percentage : ℝ := 0.015

-- Total USD needed considering both exchange rate and transaction fee
def total_usd_needed : ℝ := (required_euros * exchange_rate) * (1 + transaction_fee_percentage)

-- Additional amount needed
def additional_usd_needed : ℝ := total_usd_needed - current_savings_usd

-- Savings in a two-month period
def two_month_savings : ℝ := monthly_saving_first_month + (monthly_saving_first_month * savings_decrease_percentage)

-- Number of two-month periods required
def required_two_month_periods : ℝ := (additional_usd_needed / two_month_savings).ceil

-- Total months required
def total_months_required : ℕ := (required_two_month_periods * 2).to_nat

theorem peter_savings_time :
  total_months_required = 6 :=
by
  sorry

end peter_savings_time_l782_782660


namespace round_trip_ticket_percentage_l782_782204

variable (P : ℕ) -- Total number of passengers
variable (x : ℝ) -- Percentage of passengers with round-trip tickets who took their cars
variable (h : 0 < x ∧ x <= 100) -- Constraint for x to be within proper percentage range

-- Let's declare definitions using provided conditions
def passengers_with_round_trip_tickets_with_cars : ℝ := x / 100 * P
def passengers_with_round_trip_tickets_without_cars : ℝ := 0.6 * P

theorem round_trip_ticket_percentage (P : ℕ) (x : ℝ) (h : 0 < x ∧ x <= 100) :
  let y := x / 0.4 in y = x / 0.4 := 
  by
  sorry

end round_trip_ticket_percentage_l782_782204


namespace number_of_matches_in_first_set_l782_782234

theorem number_of_matches_in_first_set
  (x : ℕ)
  (h1 : (30 : ℚ) * x + 15 * 10 = 25 * (x + 10)) :
  x = 20 :=
by
  -- The proof will be filled in here
  sorry

end number_of_matches_in_first_set_l782_782234


namespace polynomial_real_roots_abs_c_geq_2_l782_782686

-- Definition of the polynomial P(x)
def P (x : ℝ) (a b c : ℝ) : ℝ := x^6 + a*x^5 + b*x^4 + c*x^3 + b*x^2 + a*x + 1

-- Statement of the problem: Given that P(x) has six distinct real roots, prove |c| ≥ 2
theorem polynomial_real_roots_abs_c_geq_2 (a b c : ℝ) :
  (∃ r1 r2 r3 r4 r5 r6 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
                           r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
                           r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
                           r4 ≠ r5 ∧ r4 ≠ r6 ∧
                           r5 ≠ r6 ∧
                           P r1 a b c = 0 ∧ P r2 a b c = 0 ∧ P r3 a b c = 0 ∧
                           P r4 a b c = 0 ∧ P r5 a b c = 0 ∧ P r6 a b c = 0) →
  |c| ≥ 2 := by
  sorry

end polynomial_real_roots_abs_c_geq_2_l782_782686


namespace smallest_z_minus_w_l782_782642

noncomputable def smallest_value_of_z_minus_w (z w : ℂ) : ℝ :=
  if (|z - (2 + 4 * Complex.i)| = 2) ∧ (|w - (-5 - 6 * Complex.i)| = 4) then
    sqrt 149 - 6
  else
    0   -- This else branch is needed syntactically, but will never be used given the problem statement.

theorem smallest_z_minus_w {z w : ℂ} 
  (hz : |z - (2 + 4 * Complex.i)| = 2) 
  (hw : |w - (-5 - 6 * Complex.i)| = 4) :
  |z - w| = sqrt 149 - 6 :=
sorry  -- Proof goes here.

end smallest_z_minus_w_l782_782642


namespace range_of_f_strictly_increasing_f_10_1_div_2_eq_1_div_12_l782_782691

def f (x : ℝ) : ℝ := x / (1 + abs x)

theorem range_of_f : set.Icc (-1) 1 = {y | ∃ x, f x = y} :=
sorry

theorem strictly_increasing (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) : 
  (f x₁ - f x₂) / (x₁ - x₂) > 0 :=
sorry

noncomputable def f_n : ℕ+ → (ℝ → ℝ)
| ⟨1, _⟩ := f
| ⟨n+1, h⟩ := f ∘ f_n ⟨n, nat.succ_pos n⟩

theorem f_10_1_div_2_eq_1_div_12 : f_n ⟨10, by norm_num⟩ (1/2) = 1/12 :=
sorry

end range_of_f_strictly_increasing_f_10_1_div_2_eq_1_div_12_l782_782691


namespace evaluate_exp_power_l782_782431

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l782_782431


namespace exists_proper_bicoloration_l782_782515

variable (V : Type) [Inhabited V] (E : V → V → Prop)

def is_proper_bicoloration (χ : V → Prop) : Prop :=
  ∀ v, (χ v → (∑ w in {w | E v w}, ¬χ w) ≥ ∑ w in {w | E v w}, χ w) ∧
       (¬χ v → (∑ w in {w | E v w}, χ w) ≥ ∑ w in {w | E v w}, ¬χ w)

theorem exists_proper_bicoloration : ∃ χ : V → Prop, is_proper_bicoloration V E χ :=
sorry

end exists_proper_bicoloration_l782_782515


namespace jack_cookie_sales_l782_782615

def brownies := 4
def brownies_price := 3
def lemon_squares := 5
def lemon_squares_price := 2
def goal := 50
def cookie_price := 4
def pack_size := 5
def pack_price := 17

theorem jack_cookie_sales :
  let total_money_from_previous_sales := brownies * brownies_price + lemon_squares * lemon_squares_price in
  let remaining_money_needed := goal - total_money_from_previous_sales in
  let cookies_needed_individually := remaining_money_needed / cookie_price in
  let cookies_needed_with_bulk := 
    let bulk_sales := pack_price in
    let remaining_after_bulk := remaining_money_needed - bulk_sales in
    pack_size + (remaining_after_bulk / cookie_price).ceil in
  cookies_needed_with_bulk = 8 :=
by
  sorry

end jack_cookie_sales_l782_782615


namespace score_difference_proof_l782_782888

variable (α β γ δ : ℝ)

theorem score_difference_proof
  (h1 : α + β = γ + δ + 17)
  (h2 : α = β - 4)
  (h3 : γ = δ + 5) :
  β - δ = 13 :=
by
  -- proof goes here
  sorry

end score_difference_proof_l782_782888


namespace larger_segment_length_l782_782104

open Real

theorem larger_segment_length (a b c : ℝ) (h : a = 50 ∧ b = 110 ∧ c = 120) :
  ∃ x : ℝ, x = 100 ∧ (∃ h : ℝ, a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2) :=
by
  sorry

end larger_segment_length_l782_782104


namespace no_solution_for_xx_plus_yy_eq_9z_l782_782224

theorem no_solution_for_xx_plus_yy_eq_9z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ¬ (x^x + y^y = 9^z) :=
sorry

end no_solution_for_xx_plus_yy_eq_9z_l782_782224


namespace power_of_power_evaluation_l782_782455

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l782_782455


namespace solve_z_six_minus_six_z_four_plus_nine_z_sq_eq_zero_l782_782874

theorem solve_z_six_minus_six_z_four_plus_nine_z_sq_eq_zero :
  ∀ z : ℂ, z^6 - 6 * z^4 + 9 * z^2 = 0 ↔ z = 0 ∨ z = -Complex.sqrt 3 ∨ z = Complex.sqrt 3 :=
by
  intro z
  constructor
  sorry

end solve_z_six_minus_six_z_four_plus_nine_z_sq_eq_zero_l782_782874


namespace number_of_ways_l782_782629

theorem number_of_ways (n k : ℕ) (A B : ℕ) (h1 : n ≥ 2) (h2 : k ≥ 1) (h3 : A ≠ B) :
  (∃ a_k : ℕ, a_k = (1/n) * ((n - 1)^k - (-1)^k)) :=
by
  sorry

end number_of_ways_l782_782629


namespace monic_quadratic_with_root_l782_782478

theorem monic_quadratic_with_root (p : Polynomial ℝ) : 
  Polynomial.monic p ∧ (p.eval (2 - 3 * Complex.I) = 0) ∧ ∀ c : ℂ, is_real_root p c → is_real_root p (Conj c) → p = Polynomial.X^2 - 4 * Polynomial.X + 13 :=
by 
  sorry

end monic_quadratic_with_root_l782_782478


namespace sum_first_five_terms_arith_seq_l782_782690

theorem sum_first_five_terms_arith_seq (a : ℕ → ℤ)
  (h4 : a 4 = 3) (h5 : a 5 = 7) (h6 : a 6 = 11) :
  a 1 + a 2 + a 3 + a 4 + a 5 = -5 :=
by
  sorry

end sum_first_five_terms_arith_seq_l782_782690


namespace find_AM_l782_782668

-- Definitions (conditions)
variables {A M B : ℝ}
variable  (collinear : A ≤ M ∧ M ≤ B ∨ B ≤ M ∧ M ≤ A ∨ A ≤ B ∧ B ≤ M)
          (h1 : abs (M - A) = 2 * abs (M - B)) 
          (h2 : abs (A - B) = 6)

-- Proof problem statement
theorem find_AM : (abs (M - A) = 4) ∨ (abs (M - A) = 12) :=
by 
  sorry

end find_AM_l782_782668


namespace area_of_triangle_NAB_constant_l782_782548

theorem area_of_triangle_NAB_constant (a b : ℝ) (h_ab : a > b > 0) :
  let C1 := { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }
  let C2 := { p : ℝ × ℝ | p.1^2 / (3 * a^2) + p.2^2 / (3 * b^2) = 1 }
  let Eccentricity_C1 := (sqrt 6) / 3
  let Point_on_C2 := (sqrt 3 / 2, sqrt 3 / 2)
  -- Standard equation of ellipse C1 is given by solving for a and b
  let a_sq_eq : a^2 = 1 := sorry
  let b_sq_eq : b^2 = 1 / 3 := sorry
  
  -- Proof that the area of △NAB is constant:
  -- Here M is any point on C1, and N is the intersection of the ray MO with C2
  -- Line l intersects ellipse C2 at two points A and B
  ∀ (M : ℝ × ℝ) (N A B : ℝ × ℝ),
    M ∈ C1 →
    N ∈ C2 →
    line_through M N ∩ C1 = {M} →
    line_through M N ∩ C2 = {A, B} →
  
    let area_NAB : ℝ :=
      (1 / 2) * abs ((M.1 - N.1) * (A.2 - B.2) - (M.2 - N.2) * (A.1 - B.1))
    area_NAB = (sqrt 2 + sqrt 6 / 3) :=
sorry

end area_of_triangle_NAB_constant_l782_782548


namespace sin_smallest_angle_geometric_seq_l782_782998

theorem sin_smallest_angle_geometric_seq (A B C : ℝ) (q : ℝ) 
  (h_right_angle: C = 90) 
  (h_geom_seq: (sin C = 1) ∧ (sin B = q) ∧ (sin A = q^2) ∧ (sin^2 A + sin^2 B = 1)) : 
  sin A = (sqrt 5 - 1) / 2 :=
by sorry

end sin_smallest_angle_geometric_seq_l782_782998


namespace train_speed_64_kmh_l782_782785

noncomputable def speed_of_train (distance time : ℝ) : ℝ :=
  distance / time * 3.6

theorem train_speed_64_kmh :
  speed_of_train 160 9 = 64 :=
by
  unfold speed_of_train
  norm_num
  sorry

end train_speed_64_kmh_l782_782785


namespace proof_MrLalandeInheritance_l782_782650

def MrLalandeInheritance : Nat := 18000
def initialPayment : Nat := 3000
def monthlyInstallment : Nat := 2500
def numInstallments : Nat := 6

theorem proof_MrLalandeInheritance :
  initialPayment + numInstallments * monthlyInstallment = MrLalandeInheritance := 
by 
  sorry

end proof_MrLalandeInheritance_l782_782650


namespace number_of_bowls_l782_782717

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l782_782717


namespace equation_not_satisfied_by_more_than_half_sets_l782_782670

theorem equation_not_satisfied_by_more_than_half_sets
  (n : ℕ)
  (X : Fin n → Bool)
  (A : Fin n → ℤ)
  (B : ℤ)
  (h_not_all_zero : ¬ (∀ i, A i = 0) ∨ B ≠ 0) :
  ¬ (∃ S : Fin n → Bool, (∑ i, if S i then A i else 0) = B ∧
       ∀ T : Fin n → Bool, (∑ j, if T j then A j else 0) = B → S = T) :=
sorry

end equation_not_satisfied_by_more_than_half_sets_l782_782670


namespace exists_polygon_with_3_or_4_sides_l782_782407

-- Definitions
structure RegularNGon (n : ℕ) :=
  (gon : Type) -- represents the n-gon

-- Acceptable lines and condition
def acceptable_line {n : ℕ} (gon : RegularNGon n) (line : gon → gon → Prop) :=
  ∀ A B : gon, ¬ (line A B → is_side A B)

-- Existence statement of m depending on n
theorem exists_polygon_with_3_or_4_sides
  (n : ℕ) (H_n_gt_3 : n > 3) :
  ∃ (m : ℕ), (m = n - 4) → ∀ (gon : RegularNGon n) 
    (lines : list (gon → gon → Prop)), 
    (∀ l ∈ lines, acceptable_line gon l) → 
    ∃ (small_polygon : gon → Prop), 
    small_polygon gon & (bounded_between (small_polygon.edges_count [3, 4])) :=
    sorry

end exists_polygon_with_3_or_4_sides_l782_782407


namespace solve_for_b_l782_782570

theorem solve_for_b (a b c m : ℚ) (h : m = c * a * b / (a - b)) : b = (m * a) / (m + c * a) :=
by
  sorry

end solve_for_b_l782_782570


namespace distinct_flags_count_l782_782338

theorem distinct_flags_count :
  let colors := 5 in
  let first_strip_ways := colors in
  let second_strip_ways := 4 in
  let third_strip_ways := 4 in
  let fourth_strip_ways := 4 in
  first_strip_ways * second_strip_ways * third_strip_ways * fourth_strip_ways = 320 :=
by
  sorry

end distinct_flags_count_l782_782338


namespace bill_miles_sunday_l782_782775

variables (B : ℕ)
def miles_ran_Bill_Saturday := B
def miles_ran_Bill_Sunday := B + 4
def miles_ran_Julia_Sunday := 2 * (B + 4)
def total_miles_ran := miles_ran_Bill_Saturday + miles_ran_Bill_Sunday + miles_ran_Julia_Sunday

theorem bill_miles_sunday (h1 : total_miles_ran B = 32) : 
  miles_ran_Bill_Sunday B = 9 := 
by sorry

end bill_miles_sunday_l782_782775


namespace function_additive_of_tangential_property_l782_782677

open Set

variable {f : ℝ → ℝ}

def is_tangential_quadrilateral_sides (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ (a + c = b + d)

theorem function_additive_of_tangential_property
  (h : ∀ (a b c d : ℝ), is_tangential_quadrilateral_sides a b c d → f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x + y) = f x + f y :=
by
  sorry

end function_additive_of_tangential_property_l782_782677


namespace shortest_remaining_side_l782_782357

theorem shortest_remaining_side (a b c : ℝ) (h₁ : a = 5) (h₂ : c = 13) (h₃ : a^2 + b^2 = c^2) : b = 12 :=
by
  rw [h₁, h₂] at h₃
  sorry

end shortest_remaining_side_l782_782357


namespace quadratic_real_roots_and_value_l782_782059

theorem quadratic_real_roots_and_value (m x1 x2: ℝ) 
  (h1: ∀ (a: ℝ), ∃ (b c: ℝ), a = x^2 - 4 * x - 2 * m + 5) 
  (h2: x1 * x2 + x1 + x2 = m^2 + 6):
  m ≥ 1/2 ∧ m = 1 := 
by
  sorry

end quadratic_real_roots_and_value_l782_782059


namespace ellipse_eccentricity_m_l782_782492

theorem ellipse_eccentricity_m (m : ℝ) (e : ℝ) (h1 : ∀ x y : ℝ, x^2 / m + y^2 = 1) (h2 : e = Real.sqrt 3 / 2) :
  m = 4 ∨ m = 1 / 4 :=
by sorry

end ellipse_eccentricity_m_l782_782492


namespace num_subsets_of_A_l782_782567

variable A : Set Nat := {1, 2}

theorem num_subsets_of_A : (A.toFinset.powerset.card = 4) := by
  sorry

end num_subsets_of_A_l782_782567


namespace complex_point_quadrant_l782_782139

theorem complex_point_quadrant :
  let z := complex.mk (Real.sin 3) (Real.cos 3) in
  (0 < Real.sin 3 ∧ Real.cos 3 < 0) → 
  -- z lies in the fourth quadrant
  (Real.sin 3 > 0 ∧ Real.cos 3 < 0)
:=
by 
  intros z,
  sorry

end complex_point_quadrant_l782_782139


namespace smallest_ab_41503_539_l782_782484

noncomputable def find_smallest_ab : (ℕ × ℕ) :=
  let a := 41503
  let b := 539
  (a, b)

theorem smallest_ab_41503_539 (a b : ℕ) (h : 7 * a^3 = 11 * b^5) (ha : a > 0) (hb : b > 0) :
  (a = 41503 ∧ b = 539) :=
  by
    -- Add sorry to skip the proof
    sorry

end smallest_ab_41503_539_l782_782484


namespace tangent_line_circle_l782_782090

theorem tangent_line_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, x + y = 2 ↔ x^2 + y^2 = m) → m = 2 :=
by
  intro h_tangent
  sorry

end tangent_line_circle_l782_782090


namespace smallest_five_divisible_gt_2000_l782_782018

def is_five_divisible (n : ℕ) : Prop :=
  (1 ≤ n ∧ n mod 1 = 0) + 
  (2 ≤ n ∧ n mod 2 = 0) +
  (3 ≤ n ∧ n mod 3 = 0) +
  (4 ≤ n ∧ n mod 4 = 0) +
  (5 ≤ n ∧ n mod 5 = 0) +
  (6 ≤ n ∧ n mod 6 = 0) +
  (7 ≤ n ∧ n mod 7 = 0) +
  (8 ≤ n ∧ n mod 8 = 0) +
  (9 ≤ n ∧ n mod 9 = 0) ≥ 5

theorem smallest_five_divisible_gt_2000 : ∃ N : ℕ, N > 2000 ∧ is_five_divisible N ∧ (∀ M : ℕ, M > 2000 → is_five_divisible M → N ≤ M) :=
by
  -- We state the problem but do not provide the proof
  sorry

end smallest_five_divisible_gt_2000_l782_782018


namespace bartender_overcharge_l782_782410

theorem bartender_overcharge (W p : ℝ) (H_w : W = 3) (charge : ℝ) (H_charge : charge = 11.80) :
  ¬ ∃ k : ℤ, charge = 3 * k :=
by
  -- Given initial constraints
  have H_W_div_3 : W % 3 = 0 := by rw [H_w]; norm_num
  
  -- Total cost using given conditions
  let total_cost := W + 6 * p
  
  -- Hypothesis for bartender's charge
  have H_total_cost_charge : total_cost = charge := sorry -- To be proved
  
  -- Check for divisibility by 3
  have div_check : ¬ ∃ k : ℤ, 11.80 = 3 * k := by
  { intro h,
    rcases h with ⟨k, hk⟩,
    norm_num at hk,
    exact hk }

  exact div_check


end bartender_overcharge_l782_782410


namespace number_of_bowls_l782_782743

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l782_782743


namespace max_angle_A_30_degrees_l782_782750

-- Definitions of the conditions from a)
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

def BC := 1
def AC := 2

-- Theorem statement using the conditions and correct answer identified
theorem max_angle_A_30_degrees : 
  ∃(A B C : Type) (hBC : BC = 1) (hAC : AC = 2), ∠A = 30 :=
sorry

end max_angle_A_30_degrees_l782_782750


namespace real_number_solution_pure_imaginary_solution_zero_solution_l782_782486

noncomputable def real_number_condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 2 = 0

noncomputable def pure_imaginary_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ ¬(m^2 - 3 * m + 2 = 0)

noncomputable def zero_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 3 * m + 2 = 0)

theorem real_number_solution (m : ℝ) : real_number_condition m ↔ (m = 1 ∨ m = 2) := 
sorry

theorem pure_imaginary_solution (m : ℝ) : pure_imaginary_condition m ↔ (m = -1 / 2) :=
sorry

theorem zero_solution (m : ℝ) : zero_condition m ↔ (m = 2) :=
sorry

end real_number_solution_pure_imaginary_solution_zero_solution_l782_782486


namespace invitations_per_package_l782_782398

theorem invitations_per_package (total_friends : ℕ) (total_packs : ℕ) (invitations_per_pack : ℕ) 
  (h1 : total_friends = 10) (h2 : total_packs = 5)
  (h3 : invitations_per_pack * total_packs = total_friends) : 
  invitations_per_pack = 2 :=
by
  sorry

end invitations_per_package_l782_782398


namespace AK_squared_eq_KL_times_KM_l782_782306

variable (A B C D K L M : Type) 
variable [EuclideanGeometry A B C D K L M]

-- Assuming the following given conditions
axiom parallelogram_ABCD : parallelogram A B C D
axiom K_on_BD : collinear {B, D, K}
axiom AK_intersects_CD_at_L : ∃ (pt : Type), pt = L ∧ collinear {A, K, L} ∧ collinear {C, D, L}
axiom AK_intersects_BC_at_M : ∃ (pt : Type), pt = M ∧ collinear {A, K, M} ∧ collinear {B, C, M}

theorem AK_squared_eq_KL_times_KM : AK^2 = KL * KM :=
sorry

end AK_squared_eq_KL_times_KM_l782_782306


namespace cyclic_quadrilateral_KRLQ_l782_782178

noncomputable def is_cyclic_quadrilateral (a b c d : Point) : Prop :=
∃ (circle : Circle), a ∈ circle ∧ b ∈ circle ∧ c ∈ circle ∧ d ∈ circle

theorem cyclic_quadrilateral_KRLQ
  (A S T X Y R P Q K L : Point)
  (circle_omega : Circle)
  (hA : ¬ A ∈ circle_omega)
  (hS : S ∈ circle_omega)
  (hT : T ∈ circle_omega)
  (hST : tangent A S ∧ tangent A T)
  (hX : midpoint X A T)
  (hY : midpoint Y A S)
  (hR : tangent X R ∧ R ∈ circle_omega)
  (hP : midpoint P X T)
  (hQ : midpoint Q X R)
  (hK : line_intersection P Q X Y = some K)
  (hL : line_intersection S X T K = some L) :
  is_cyclic_quadrilateral K R L Q :=
sorry

end cyclic_quadrilateral_KRLQ_l782_782178


namespace quadratic_roots_condition_l782_782092

theorem quadratic_roots_condition (m : ℝ) :
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1^2 - 3 * x1 + 2 * m = 0 ∧ x2^2 - 3 * x2 + 2 * m = 0) →
  m < 9 / 8 :=
by
  sorry

end quadratic_roots_condition_l782_782092


namespace part1_part2_part3_l782_782237

-- Define the daily deviations and planned sales volume
def daily_deviations : list ℤ := [3, -5, -2, 11, -7, 13, 5]
def planned_sales_volume := 100

-- Question 1: Difference in sales volumes between highest and lowest sales days
theorem part1 :
  max (3 :: -5 :: -2 :: 11 :: -7 :: 13 :: 5 :: list.nil) - min (3 :: -5 :: -2 :: 11 :: -7 :: 13 :: 5 :: list.nil) = 20 :=
begin
  sorry
end

-- Question 2: Total kilograms of pomelos sold in the first week
theorem part2 :
  list.sum daily_deviations + planned_sales_volume * 7 = 718 :=
begin
  sorry
end

-- Question 3: Total profit from selling pomelos in the first week
theorem part3 :
  (list.sum daily_deviations + planned_sales_volume * 7) * 5 = 3590 :=
begin
  sorry
end

end part1_part2_part3_l782_782237


namespace range_of_alpha_l782_782665

-- Define the function y = x^3 - x + 2/3
def f (x : ℝ) : ℝ := x^3 - x + 2 / 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- The task is to prove the range of alpha where f' gives the slope of the tangent line
theorem range_of_alpha :
  ∀ (α : ℝ), (∃ x : ℝ, f' x = tan α) ↔ (α ∈ set.Ico 0 (π / 2) ∨ α ∈ set.Ico (3 * π / 4) π) :=
by
  sorry

end range_of_alpha_l782_782665


namespace problem_1_problem_2_problem_3_l782_782049

def f (x : ℝ) : ℝ := 2^x

-- Proof Problem 1
theorem problem_1 : ∃ x : ℝ, f(2 * x) - f(x + 1) = 8 ↔ x = 2 :=
sorry

-- Proof Problem 2
def g (x : ℝ) (a : ℝ) : ℝ := f(x) + a * 4^x

theorem problem_2 (a : ℝ) : ∃ M : ℝ, ∀ x ∈ set.Icc (0 : ℝ) 1, g x a ≤ M :=
sorry

-- Proof Problem 3
theorem problem_3 (x1 x2 : ℝ) : 
    (f(x1) + f(x2) = f(x1) * f(x2)) ∧ 
    (f(x1) + f(x2) + f(x3) = f(x1) * f(x2) * f(x3)) → 
    ∃ x3 : ℝ, x3 = 2 - real.logb 2 3 :=
sorry

end problem_1_problem_2_problem_3_l782_782049


namespace determine_a_l782_782934

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - a * x + 3

-- Define the condition that f(x) >= a for all x in the interval [-1, +∞)
def condition (a : ℝ) : Prop := ∀ x : ℝ, x ≥ -1 → f x a ≥ a

-- The theorem to prove:
theorem determine_a : ∀ a : ℝ, condition a ↔ a ≤ 2 :=
by
  sorry

end determine_a_l782_782934


namespace find_E_coordinates_l782_782207

structure Point :=
(x : ℚ)
(y : ℚ)

def A : Point := { x := -2, y := 1 }
def B : Point := { x := 1, y := 4 }
def C : Point := { x := 4, y := -3 }

def D : Point := 
  let m : ℚ := 1
  let n : ℚ := 2
  let x1 := A.x
  let y1 := A.y
  let x2 := B.x
  let y2 := B.y
  { x := (m * x2 + n * x1) / (m + n), y := (m * y2 + n * y1) / (m + n) }

theorem find_E_coordinates : 
  let k : ℚ := 4
  let x_E : ℚ := (k * C.x + D.x) / (k + 1)
  let y_E : ℚ := (k * C.y + D.y) / (k + 1)
  ∃ E : Point, E.x = (17:ℚ) / 3 ∧ E.y = -(14:ℚ) / 3 :=
sorry

end find_E_coordinates_l782_782207


namespace reciprocal_relation_l782_782572

theorem reciprocal_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := 
by
  sorry

end reciprocal_relation_l782_782572


namespace choose_3_out_of_10_l782_782115

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 := by
  sorry

end choose_3_out_of_10_l782_782115


namespace problem1_problem2_l782_782929

-- Definition of the function and given conditions for problem 1
def f (a b c x : ℝ) := (a * x^2 + b * x + c) * Real.exp x

-- Problem 1
theorem problem1 (a b c : ℝ) (h0 : f a b c 0 = 1) (h1 : f a b c 1 = 0) :
  (∀ x y ∈ set.Icc 0 1, x < y → f a b c x ≥ f a b c y) → a ∈ set.Icc 0 1 := sorry

-- Definition of the function and given conditions for problem 2
def f_a0 (b c x : ℝ) := (b * x + c) * Real.exp x

-- Problem 2
theorem problem2 (b c : ℝ) (m : ℝ)
  (h0 : f_a0 b c 0 = 1) (h1 : f_a0 b c 1 = 0)
  (h2 : ∀ x : ℝ, 2 * f_a0 b c x + 4 * x * Real.exp x ≥ m * x + 1 ∧ m * x + 1 ≥ -x^2 + 4*x + 1) :
  m = 4 := sorry

end problem1_problem2_l782_782929


namespace radius_is_12_of_given_height_and_area_l782_782253

noncomputable def cylinder_radius_of_lateral_surface_area :
  (h : ℝ) → (A : ℝ) → (r : ℝ) → Prop :=
  λ h A r, h = 21 ∧ A = 1583.3626974092558 ∧ A = 2 * Real.pi * r * h → r = 12

theorem radius_is_12_of_given_height_and_area :
  ∀ (h A r : ℝ), cylinder_radius_of_lateral_surface_area h A r :=
by
  intros h A r
  sorry

end radius_is_12_of_given_height_and_area_l782_782253


namespace triangle_is_isosceles_l782_782318

variables {A B C M N : Type} 
variable [MetricSpace A]
variables [MetricSpace B]
variables [MetricSpace C]
variables [MetricSpace M]
variables [MetricSpace N]
variables (triangle_ABC : Triangle B C A)
variables (circle_diameter_BC : Circle B C)
variables (M_on_side_AB : IsOn M A B)
variables (N_on_side_AC : IsOn N A C)
variables (BM_eq_CN : MetricSpace.dist B M = MetricSpace.dist C N)

theorem triangle_is_isosceles : MetricSpace.dist A B = MetricSpace.dist A C :=
by
  sorry

end triangle_is_isosceles_l782_782318


namespace prove_Tn_ineq_l782_782906

variable (a : ℝ)
variable (a_ne_zero : a ≠ 0)
variable (a_ne_one : a ≠ 1)
variable (a_frac : a = 1/3)

noncomputable def S (a : ℝ) (n : ℕ) (a_n : ℝ) := (a / (a - 1)) * (a_n - 1)

theorem prove_Tn_ineq (n : ℕ) (a_n : ℕ → ℝ)
  (h_an_geometric : ∀ n, a_n n = (1/3)^n)
  (c_n := λ n, 1 / (1 + a_n n) + 1 / (1 - a_n (n + 1)))
  (T_n := ∑ i in finset.range (n + 1), c_n i)
  (a := 1/3) :
  T_n > 2 * n - 1 / 3 :=
by
  sorry

end prove_Tn_ineq_l782_782906


namespace tan_ratio_l782_782172

theorem tan_ratio (p q : Real) (hpq1 : Real.sin (p + q) = 0.6) (hpq2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := 
by
  sorry

end tan_ratio_l782_782172


namespace initial_pieces_l782_782886

-- Definitions of the conditions
def pieces_eaten : ℕ := 7
def pieces_given : ℕ := 21
def pieces_now : ℕ := 37

-- The proposition to prove
theorem initial_pieces (C : ℕ) (h : C - pieces_eaten + pieces_given = pieces_now) : C = 23 :=
by
  -- Proof would go here
  sorry

end initial_pieces_l782_782886


namespace general_term_and_sum_minimum_l782_782047

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem general_term_and_sum_minimum (a : ℕ → ℝ) (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = -1) (h_a5 : a 5 = 5) :
  (∀ n, a n = 2 * n - 5) ∧ (∀ n, S n = n * (n - 4) ∧ S (4) = 0) :=
begin
  sorry,
end

end general_term_and_sum_minimum_l782_782047


namespace product_xy_l782_782174

theorem product_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h : ∀ m n : ℤ, ∃ (m n : ℤ), 
  (sqrt (log x) = m ∧ sqrt (log y) = n ∧ 
  m + n + 2 * log (sqrt x ) + 2 * log (sqrt y) = 84)) : 
  x * y = 10 ^ 72 := 
by
  sorry

end product_xy_l782_782174


namespace betty_age_l782_782296

theorem betty_age {A M B : ℕ} (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 14) : B = 7 :=
sorry

end betty_age_l782_782296


namespace original_price_per_tire_l782_782891

-- Definitions derived from the problem
def number_of_tires : ℕ := 4
def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36

-- Goal to prove the original price of each tire
theorem original_price_per_tire :
  (sale_price_per_tire + total_savings / number_of_tires) = 84 :=
by sorry

end original_price_per_tire_l782_782891


namespace red_blue_regions_l782_782778

theorem red_blue_regions (
    a b : Nat, -- number of red regions and blue regions respectively
    λ : List Nat, -- a list representing λ(P) for various intersection points
    h_colored : ∀ (r1 r2 : Nat), r1 ≠ r2 → color(r1) ≠ color(r2) -- adjacent regions have different colors
) : a ≤ 2 * b - 2 - λ.sum (λ x, x - 2) :=
by
  sorry

end red_blue_regions_l782_782778


namespace leading_coefficient_g_l782_782696

noncomputable def g (x : ℝ) : ℝ := sorry

theorem leading_coefficient_g :
  (∀ x : ℝ, g(x + 1) - g(x) = 10*x + 3) →
  (∃ c : ℝ, ∀ x : ℝ, g(x) = 5*x^2 - 2*x + c) :=
by
  -- Proof outline will be added here
  sorry

end leading_coefficient_g_l782_782696


namespace chess_game_problem_l782_782661

-- Mathematical definitions based on the conditions
def petr_wins : ℕ := 6
def petr_draws : ℕ := 2
def karel_points : ℤ := 9
def points_for_win : ℕ := 3
def points_for_loss : ℕ := 2
def points_for_draw : ℕ := 0

-- Defining the final statement to prove
theorem chess_game_problem :
    ∃ (total_games : ℕ) (leader : String), total_games = 15 ∧ leader = "Karel" := 
by
  -- Placeholder for proof
  sorry

end chess_game_problem_l782_782661


namespace evaluate_power_l782_782465

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l782_782465


namespace ratio_Eustace_Milford_l782_782844

variable (E M : ℕ)

theorem ratio_Eustace_Milford (hE : E + 3 = 39) (hM : M + 3 = 21) : E / M = 2 :=
by
  have hE_calc : E = 36 := by linarith
  have hM_calc : M = 18 := by linarith
  rw [hE_calc, hM_calc]
  norm_num

#check ratio_Eustace_Milford

end ratio_Eustace_Milford_l782_782844


namespace no_winning_strategy_exceeds_half_probability_l782_782331

-- Provided conditions
def well_shuffled_standard_deck : Type := sorry -- Placeholder for the deck type

-- Statement of the problem
theorem no_winning_strategy_exceeds_half_probability :
  ∀ strategy : (well_shuffled_standard_deck → ℕ → bool),
    let r := 26 in -- Assuming a standard deck half red cards (26 red)
    let b := 26 in -- and half black cards (26 black)
    let P_win := (r : ℝ) / (r + b) in       
    P_win ≤ 0.5 :=
by
  sorry

end no_winning_strategy_exceeds_half_probability_l782_782331


namespace find_a_l782_782914

noncomputable def f (a x : ℝ) : ℝ := a * real.sqrt (1 - x^2) + real.sqrt(1 + x) + real.sqrt(1 - x)

def t (x : ℝ) : ℝ := real.sqrt(1 + x) + real.sqrt(1 - x)

def m (a t : ℝ) : ℝ := 1/2 * a * t^2 + t - a

noncomputable def g (a : ℝ) : ℝ :=
  if a > -1/2 then a + 2
  else if -real.sqrt 2 / 2 < a && a ≤ -1/2 then -a - 1/(2*a)
  else real.sqrt 2

theorem find_a (a : ℝ) : g a = g (1 / a) ↔ (a ∈ Icc(-real.sqrt 2, -real.sqrt 2 / 2) ∨ a = 1) := sorry

end find_a_l782_782914


namespace natural_numbers_partition_l782_782854

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l782_782854


namespace total_number_of_animals_is_304_l782_782098

theorem total_number_of_animals_is_304
    (dogs frogs : ℕ) 
    (h1 : frogs = 160) 
    (h2 : frogs = 2 * dogs) 
    (cats : ℕ) 
    (h3 : cats = dogs - (dogs / 5)) :
  cats + dogs + frogs = 304 :=
by
  sorry

end total_number_of_animals_is_304_l782_782098


namespace probability_distance_ge_one_l782_782160

theorem probability_distance_ge_one (S : set ℝ) (side_length_S : ∀ x ∈ S, x = 2)
  (P : ℝ) : 
  -- Assuming two points are chosen independently at random on the sides of a square S of side length 2
  let prob := (26 - Real.pi) / 32 in
    P = prob := 
sorry

end probability_distance_ge_one_l782_782160


namespace number_of_bowls_l782_782729

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l782_782729


namespace Fedya_prevents_four_digits_l782_782345

theorem Fedya_prevents_four_digits (n_0 : ℕ) (h0 : n_0 = 123) (h1 : ∀ t : ℕ, ∃ k : ℕ, (nat.digits 10 (n_0 + 102 * t)).erase_leading_zeros < 1000 := sorry

  sorry

end Fedya_prevents_four_digits_l782_782345


namespace option_B_option_D_l782_782514

-- Definitions and conditions as per the original problem
variable {f : ℝ → ℝ}
variable {x y : ℝ}

-- Condition 1: \(f\) is defined on \((0, +\infty)\)
def func_defined : Prop := ∀ x > 0, f x ≠ 0

-- Condition 2: \(\frac{x_2 f(x_1) - x_1 f(x_2)}{x_1 - x_2} > 0\) for \(x_1 \neq x_2\)
def cond_inequality : Prop := ∀ (x1 x2 : ℝ), x1 ≠ x2 → (x2 * f x1 - x1 * f x2) / (x1 - x2) > 0

-- Option B: The function \(y = \frac{f(x)}{x}\) is increasing on \((0, +\infty)\)
theorem option_B (h₁ : func_defined) (h₂ : cond_inequality) : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 < x2 → (f x2 / x2) > (f x1 / x1) := sorry

-- Option D: \(f(2x₁ + x₂) + f(x₁ + 2x₂) > 3f(x₁ + x₂)\)
theorem option_D (h₁ : func_defined) (h₂ : cond_inequality) : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → 
  f (2 * x1 + x2) + f (x1 + 2 * x2) > 3 * f (x1 + x2) := sorry


end option_B_option_D_l782_782514


namespace metropolis_city_taxi_fare_l782_782987

theorem metropolis_city_taxi_fare
  (base_fare : ℝ)
  (initial_miles : ℝ)
  (additional_mile_rate : ℝ)
  (additional_mile_charge : ℝ)
  (total_amount : ℝ)
  (tip : ℝ)
  (total_miles : ℝ) :
  base_fare = 3.00 →
  initial_miles = 0.75 →
  additional_mile_rate = 0.1 →
  additional_mile_charge = 0.25 →
  total_amount = 15 →
  tip = 3 →
  (base_fare + additional_mile_charge * ((total_miles - initial_miles) / additional_mile_rate) = total_amount - tip)
  → total_miles = 4.35 :=
begin
  sorry
end

end metropolis_city_taxi_fare_l782_782987


namespace principal_amount_correct_l782_782881

noncomputable def calculate_principal (A r n t : ℝ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem principal_amount_correct : 
  calculate_principal 1008 0.05 1 2.4 ≈ 894.21 :=
by 
  sorry

end principal_amount_correct_l782_782881


namespace binomial_identity_example_l782_782913

theorem binomial_identity_example
  (h1 : nat.choose 20 12 = 125970)
  (h2 : nat.choose 19 11 = 75582)
  (h3 : nat.choose 18 11 = 31824) :
  nat.choose 19 12 = 50388 := by
  sorry

end binomial_identity_example_l782_782913


namespace bus_costs_unique_min_buses_cost_A_l782_782772

-- Defining the main conditions
def condition1 (x y : ℕ) : Prop := x + 2 * y = 300
def condition2 (x y : ℕ) : Prop := 2 * x + y = 270

-- Part 1: Proving individual bus costs
theorem bus_costs_unique (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 110 := 
by 
  sorry

-- Part 2: Minimum buses of type A and total cost constraint
def total_buses := 10
def total_cost (x y a : ℕ) : Prop := 
  x * a + y * (total_buses - a) ≤ 1000

theorem min_buses_cost_A (x y : ℕ) (hx : x = 80) (hy : y = 110) :
  ∃ a cost, total_cost x y a ∧ a >= 4 ∧ cost = x * 4 + y * (total_buses - 4) ∧ cost = 980 :=
by
  sorry

end bus_costs_unique_min_buses_cost_A_l782_782772


namespace distance_from_gable_to_citadel_l782_782653

def citadel_position : ℂ := 0
def gable_position : ℂ := 1600 + 1200 * complex.i

theorem distance_from_gable_to_citadel : complex.abs (gable_position - citadel_position) = 2000 :=
by
  sorry

end distance_from_gable_to_citadel_l782_782653


namespace price_equivalence_l782_782575

/-- Define the relationship between the cost of apples, bananas, and carrots. -/
def cost_equiv (cost : String → ℕ) : Prop :=
  (cost "10 apples" = cost "5 bananas") ∧
  (cost "2 bananas" = cost "5 carrots")

/-- The main theorem to prove: 12 apples cost the same as 15 carrots. -/
theorem price_equivalence (cost : String → ℕ) 
  (h : cost_equiv cost) : 
  cost "12 apples" = cost "15 carrots" :=
begin
  sorry
end

end price_equivalence_l782_782575


namespace number_of_bowls_l782_782715

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l782_782715


namespace hours_per_batch_l782_782146

noncomputable section

def gallons_per_batch : ℕ := 3 / 2   -- 1.5 gallons expressed as a rational number
def ounces_per_gallon : ℕ := 128
def jack_consumption_per_2_days : ℕ := 96
def total_days : ℕ := 24
def time_spent_hours : ℕ := 120

def total_ounces : ℕ := gallons_per_batch * ounces_per_gallon
def total_ounces_consumed_24_days : ℕ := jack_consumption_per_2_days * (total_days / 2)
def number_of_batches : ℕ := total_ounces_consumed_24_days / total_ounces

theorem hours_per_batch :
  (time_spent_hours / number_of_batches) = 20 := by
  sorry

end hours_per_batch_l782_782146


namespace evaluate_exp_power_l782_782436

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l782_782436


namespace intersection_of_M_and_N_l782_782560

def setM (x : ℝ) := y = Real.log (1 - x)
def setN (x : ℝ) := y = x^2 - 2*x + 1

theorem intersection_of_M_and_N :
  {x : ℝ | ∃ y : ℝ, setM x} ∩ {x : ℝ | ∃ y : ℝ, setN x} = {x : ℝ | x < 1} := sorry

end intersection_of_M_and_N_l782_782560


namespace rank_logarithmic_expressions_l782_782899

noncomputable def a : ℝ := Real.log π / Real.log 3
noncomputable def b : ℝ := Real.log π / Real.log (1 / 3)
noncomputable def c : ℝ := π ^ (-3)

theorem rank_logarithmic_expressions (h_pi_gt_one : π > 1) (h_pi_gt_three : π > 3) : 
  a > c ∧ c > b := 
by
  sorry

end rank_logarithmic_expressions_l782_782899


namespace find_f_l782_782534

def f (x : ℝ) : ℝ := 3 * x + 2

theorem find_f (x : ℝ) : f x = 3 * x + 2 :=
  sorry

end find_f_l782_782534


namespace reflected_quad_area_l782_782546

-- Defining a convex quadrilateral with area 1
def convex_quadrilateral (A B C D : Point) : Prop :=
  convex A B C D ∧ area A B C D = 1

-- The problem statement to be proved
theorem reflected_quad_area (A B C D : Point) (h: convex_quadrilateral A B C D) :
  let A' := reflect_over(B, A)
      B' := reflect_over(C, B)
      C' := reflect_over(D, C)
      D' := reflect_over(A, D)
  in area A' B' C' D' = 5 :=
begin
  sorry
end

end reflected_quad_area_l782_782546


namespace union_complements_eq_l782_782703

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complements_eq :
  U = {0, 1, 3, 5, 6, 8} →
  A = {1, 5, 8} →
  B = {2} →
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  -- Prove that (U \ A) ∪ B = {0, 2, 3, 6}
  sorry

end union_complements_eq_l782_782703


namespace condition_iff_inequality_l782_782029

theorem condition_iff_inequality (a b : ℝ) (h : a * b ≠ 0) : (0 < a ∧ 0 < b) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
by
  -- Proof goes here
  sorry 

end condition_iff_inequality_l782_782029


namespace average_word_count_l782_782951

-- Definitions for the conditions provided
def word_counts := [8, 24, 34, 20, 8, 6]
def midpoints := [3, 8, 13, 18, 23, 28]
def total_sentences := 100

-- Mathematics proof problem to prove average_words == 13.7 given the conditions
theorem average_word_count :
  (∑ i, word_counts[i] * midpoints[i]) / total_sentences.toFloat = 13.7 :=
  by
  sorry

end average_word_count_l782_782951


namespace smallest_positive_period_and_monotonicity_l782_782900

noncomputable def f (x : ℝ) : ℝ := 6 * (Real.cos x)^2 - 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

theorem smallest_positive_period_and_monotonicity :
  (∀ x, f (x + Real.pi) = f x) ∧ ∀ k : ℤ, (∀ x ∈ Set.Icc (-7 * Real.pi / 12 + k * Real.pi) (-Real.pi / 12 + k * Real.pi), f' x > 0) := by
  sorry

end smallest_positive_period_and_monotonicity_l782_782900


namespace semi_circle_radius_l782_782254

noncomputable def perimeter : ℝ := 15.93893722612836
noncomputable def π_value : ℝ := 3.141592653589793

theorem semi_circle_radius (r : ℝ) 
    (h : perimeter = π_value * r + 2 * r) : r ≈ 3.099 := 
sorry

end semi_circle_radius_l782_782254


namespace power_of_power_evaluation_l782_782457

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l782_782457


namespace point_outside_circle_range_l782_782085

theorem point_outside_circle_range (m : ℝ) : 
  let P := (-1, -1)
  let circle_eq := λ m x y => x^2 + y^2 + 4*m*x - 2*y + 5*m
  let point_outside := circle_eq m (-1) (-1) > 0
  let represents_circle := 4 + 16*m^2 - 20*m > 0
  (point_outside ∧ represents_circle) → (m ∈ Ioo (-4) (1/4) ∨ m > 1) :=
sorry

end point_outside_circle_range_l782_782085


namespace sum_of_perpendiculars_of_point_in_square_l782_782346

theorem sum_of_perpendiculars_of_point_in_square (s : ℝ) (P : ℝ × ℝ) 
  (hP : 0 ≤ P.1 ∧ P.1 ≤ s ∧ 0 ≤ P.2 ∧ P.2 ≤ s) :
  let d1 := P.2,
      d2 := s - P.1,
      d3 := s - P.2,
      d4 := P.1 in
  d1 + d2 + d3 + d4 = 2 * s :=
by
  sorry

end sum_of_perpendiculars_of_point_in_square_l782_782346


namespace estimate_3_sqrt_2_range_l782_782000

theorem estimate_3_sqrt_2_range :
  4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 :=
by
  sorry

end estimate_3_sqrt_2_range_l782_782000


namespace acute_triangles_in_prism_l782_782369

theorem acute_triangles_in_prism :
  (set.univ.filter (λ (V : finset (fin 8)), V.card = 3)).card -
  (set.univ.filter (λ (T : finset (fin 8)), ∃ (a b c : fin 8), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
                    is_right_triangle a.val b.val c.val)).card = 8 := 
sorry

end acute_triangles_in_prism_l782_782369


namespace domain_of_f_eq_R_l782_782982

noncomputable def f (x m : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

theorem domain_of_f_eq_R (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x + 3 ≠ 0) ↔ (0 ≤ m ∧ m < 3 / 4) :=
by
  sorry

end domain_of_f_eq_R_l782_782982


namespace square_prob_distance_l782_782158

noncomputable def probability_distance_ge_1 (S : set (ℝ × ℝ)) (side_len : ℝ) : ℝ :=
  let a := 28 in
  let b := 1 in
  let c := 32 in
  (a - b * Real.pi) / c

theorem square_prob_distance (side_len : ℝ) (hS : side_len = 2) :
  probability_distance_ge_1 {p | p.1 ≥ 0 ∧ p.1 ≤ side_len ∧ p.2 ≥ 0 ∧ p.2 ≤ side_len} side_len = (28 - Real.pi) / 32 :=
by {
  rw hS,
  unfold probability_distance_ge_1,
  sorry
}

end square_prob_distance_l782_782158


namespace wire_length_ratio_l782_782384

theorem wire_length_ratio :
  ∀ (Bonnie_wire_pieces Roark_wire_lengths : ℕ)
    (Bonnie_piece_length Roark_piece_length : ℕ)
    (total_volume_Bonnie total_volume_Roark_cube unit_cube_volume : ℕ),
  Bonnie_wire_pieces = 12 →
  Bonnie_piece_length = 8 →
  let volume_Bonnie_cube := Bonnie_piece_length ^ 3 in
  total_volume_Bonnie = volume_Bonnie_cube →
  Roark_wire_lengths / unit_cube_volume = total_volume_Bonnie / Roark_piece_length ^ 3 →
  let length_Bonnie := Bonnie_wire_pieces * Bonnie_piece_length in
  let cubes_needed := total_volume_Bonnie / unit_cube_volume in
  let length_Roark_per_cube := 12 * Roark_piece_length in
  let total_length_Roark := cubes_needed * length_Roark_per_cube in
  (length_Bonnie : ℚ) / (total_length_Roark : ℚ) = 1 / 16 :=
begin
  intros,
  sorry
end

end wire_length_ratio_l782_782384


namespace eccentricity_range_l782_782694

def ellipse_eq (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def circle_eq (x y c : ℝ) : Prop := x^2 + y^2 = c^2
def eccentricity (a c : ℝ) := (c^2 / a^2)⁰^(1/2)

theorem eccentricity_range (a b c e : ℝ) 
  (h_ellipse: a > b > 0) 
  (h_eq_c: c = (a^2 - b^2)⁰^(1/2)) 
  (h_ellipse_eq : ∀ x y, ellipse_eq x y a b → circle_eq x y c → (have e = eccentricity a c, from true)): 
  (h_intersection: ∃ x y, ellipse_eq x y a b ∧ circle_eq x y c)
  : (sqrt 2 / 2) ≤ e ∧ e < 1 :=
sorry

end eccentricity_range_l782_782694


namespace quadrilateral_inequality_equality_condition_iff_parallelogram_l782_782529

variables {V : Type*} [InnerProductSpace ℝ V]

-- Define vectors representing the sides of the quadrilateral
variables (a b c : V)

-- Main theorem statement
theorem quadrilateral_inequality :
  ∥a∥^2 + ∥b - a∥^2 + ∥c - b∥^2 + ∥c - a∥^2 ≥ ∥b∥^2 + ∥b - a + a - c∥^2 :=
begin
  -- Proof is omitted
  sorry
end

-- Equality condition statement
theorem equality_condition_iff_parallelogram :
  ∥a∥^2 + ∥b - a∥^2 + ∥c - b∥^2 + ∥c - a∥^2 = ∥b∥^2 + ∥b - a + a - c∥^2 ↔ b = a + c :=
begin
  -- Proof is omitted
  sorry
end

end quadrilateral_inequality_equality_condition_iff_parallelogram_l782_782529


namespace eval_exp_l782_782422

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l782_782422


namespace domain_of_f_l782_782904

noncomputable def f (a x : ℝ) : ℝ := (Real.log (3 - x) / Real.log a) / (x - 2)

theorem domain_of_f (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  {x : ℝ | f a x ≠ ⊥} = {x : ℝ | x < 3 ∧ x ≠ 2} :=
by sorry

end domain_of_f_l782_782904


namespace distinct_real_roots_range_of_m_l782_782033

theorem distinct_real_roots_range_of_m (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + x₁ - m = 0) ∧ (x₂^2 + x₂ - m = 0)) → m > -1/4 := 
sorry

end distinct_real_roots_range_of_m_l782_782033


namespace most_likely_num_acceptable_bearings_l782_782372

/--
Let X be a normally distributed random variable with standard deviation σ = 0.4 mm.
The deviation X from the design size is considered acceptable if |X| ≤ 0.77 mm.
We take a sample of 100 bearings.
Prove that the most likely number of acceptable bearings in the sample is 95.
-/
theorem most_likely_num_acceptable_bearings :
  ∀ (X : ℝ → ℝ), is_normal_dist X 0 0.4 →
  (∀ x, X x ∈ set.Icc (-0.77) 0.77) →
  let n := 100 in
  let p := 0.9464 in
  n * p ≈ 95 :=
begin
  sorry
end

end most_likely_num_acceptable_bearings_l782_782372


namespace largest_angle_l782_782520

-- Define the sequence where the sum of the first n terms is n^2
def sequence (n : ℕ) : ℕ := sorry

-- Define a_2, a_3, and a_4 in terms of the sequence
def a_2 : ℕ := 2 * 2 - 1 * 1
def a_3 : ℕ := 3 * 3 - 2 * 2
def a_4 : ℕ := 4 * 4 - 3 * 3

-- Largest angle of triangle
theorem largest_angle (θ : ℝ) (k : ℝ) (h_pos : k > 0) (h_ratio : 3 * k = a_2 ∧ 5 * k = a_3 ∧ 7 * k = a_4) :
  θ = 2 * Real.pi / 3 :=
  sorry

end largest_angle_l782_782520


namespace least_subtraction_divisible_l782_782758

theorem least_subtraction_divisible (n : ℕ) (h : n = 3830) (lcm_val : ℕ) (hlcm : lcm_val = Nat.lcm (Nat.lcm 3 7) 11) 
(largest_multiple : ℕ) (h_largest : largest_multiple = (n / lcm_val) * lcm_val) :
  ∃ x : ℕ, x = n - largest_multiple ∧ x = 134 := 
by
  sorry

end least_subtraction_divisible_l782_782758


namespace monic_quadratic_with_root_l782_782479

theorem monic_quadratic_with_root (p : Polynomial ℝ) : 
  Polynomial.monic p ∧ (p.eval (2 - 3 * Complex.I) = 0) ∧ ∀ c : ℂ, is_real_root p c → is_real_root p (Conj c) → p = Polynomial.X^2 - 4 * Polynomial.X + 13 :=
by 
  sorry

end monic_quadratic_with_root_l782_782479


namespace f_log_value_l782_782558

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

noncomputable def f_value_at (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  if h : 0 < x ∧ x < 1 then 2^x else 0

theorem f_log_value (f : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : is_periodic_function f 2)
  (h3 : ∀ x, 0 < x ∧ x < 1 → f x = 2^x) : 
  f (Real.log 0.5 23) = -23 / 16 := sorry

end f_log_value_l782_782558


namespace combined_distance_week_l782_782147

def distance_julien (time: ℕ) (speed: ℕ) : ℕ := speed * time
def distance_sarah (distance_julien: ℕ) : ℕ := 2 * distance_julien
def distance_jamir (distance_sarah: ℕ) : ℕ := distance_sarah + 20
def distance_lily (time: ℕ) (speed_julien: ℕ) : ℕ := 4 * speed_julien * time
def combined_distance_sunny (d_julien d_sarah d_jamir d_lily: ℕ) : ℕ := d_julien + d_sarah + d_jamir + d_lily
def combined_distance_cloudy (d_julien d_sarah d_jamir d_lily: ℕ) : ℕ := (d_julien + d_sarah + d_jamir + d_lily) / 2

theorem combined_distance_week :
  let speed_julien := 50 / 20,
      distance_julien := distance_julien 30 speed_julien,
      distance_sarah := distance_sarah distance_julien,
      distance_jamir := distance_jamir distance_sarah,
      distance_lily := distance_lily 30 speed_julien,
      total_sunny := 5 * combined_distance_sunny distance_julien distance_sarah distance_jamir distance_lily,
      total_cloudy := 2 * combined_distance_cloudy distance_julien distance_sarah distance_jamir distance_lily
  in total_sunny + total_cloudy = 4170 := 
  by
    let speed_julien := 50 / 20
    let distance_julien := distance_julien 30 speed_julien
    let distance_sarah := distance_sarah distance_julien
    let distance_jamir := distance_jamir distance_sarah
    let distance_lily := distance_lily 30 speed_julien
    let total_sunny := 5 * combined_distance_sunny distance_julien distance_sarah distance_jamir distance_lily
    let total_cloudy := 2 * combined_distance_cloudy distance_julien distance_sarah distance_jamir distance_lily
    sorry

end combined_distance_week_l782_782147


namespace valveOperationTime_l782_782801

theorem valveOperationTime (a b t : ℚ) (h1 : a * (1 / 10) + b * (1 / 15) + (t - a - b) * (1 / 6) = 1) (h2 : t = 7) : 
  t - a - b = 5 := 
by
  sorry

end valveOperationTime_l782_782801


namespace remainder_of_smallest_multiple_of_8_with_unique_digits_l782_782155

theorem remainder_of_smallest_multiple_of_8_with_unique_digits
  (M : ℕ) (h1 : M % 8 = 0) (h2 : ∀ i j : ℕ, i ≠ j → ¬(M.digit_in_position i = M.digit_in_position j)) :
  M % 1000 = 120 := 
sorry

end remainder_of_smallest_multiple_of_8_with_unique_digits_l782_782155


namespace books_taken_out_on_monday_l782_782706

-- Define total number of books initially
def total_books_init := 336

-- Define books taken out on Monday
variable (x : ℕ)

-- Define books brought back on Tuesday
def books_brought_back := 22

-- Define books present after Tuesday
def books_after_tuesday := 234

-- Theorem statement
theorem books_taken_out_on_monday :
  total_books_init - x + books_brought_back = books_after_tuesday → x = 124 :=
by sorry

end books_taken_out_on_monday_l782_782706


namespace parabola_focus_l782_782474

-- Definition of the parabola equation
def parabola (x : ℝ) : ℝ := 2 * x^2 + 6 * x - 5

-- Assertion about the focus of the parabola
theorem parabola_focus : 
  ∃ h k : ℝ, h = -3/2 ∧ k = -45/8 ∧ (parabola_focus_condition parabola h k) := 
sorry

-- Helper function for the focus condition used in the theorem
def parabola_focus_condition (p : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∃ a b : ℝ, (∀ x : ℝ, p x = a * (x + h)^2 + b) ∧ b = k + 1 / (4 * a)

end parabola_focus_l782_782474


namespace triangle_to_20gon_l782_782411

theorem triangle_to_20gon (a : ℝ) :
  ∃ (p q : set (ℝ × ℝ)), 
  is_triangle (triangle_with_side a) p q ∧
  can_rearrange_to_20gon p q :=
sorry

end triangle_to_20gon_l782_782411


namespace wanda_brought_45_pieces_of_bread_l782_782617

theorem wanda_brought_45_pieces_of_bread
  (T : ℝ)             -- The number of treats Jane brings
  (HB : 0.75 * T)     -- Jane brings 0.75 times T pieces of bread
  (TW : T / 2)        -- Wanda brings T/2 treats
  (BW : 1.5 * (T / 2))-- Wanda brings 1.5 times T/2 pieces of bread
  (Total : 0.75 * T + T + 1.5 * (T / 2) + T / 2 = 225) :  -- Total count is 225
  BW = 45 :=          -- We need to prove that BW = 45
begin
  sorry
end

end wanda_brought_45_pieces_of_bread_l782_782617


namespace minimal_abs_diff_l782_782077

theorem minimal_abs_diff (x y : ℕ) (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h_eq : x * y - 4 * x + 5 * y = 221) : 
  ∃ x y, |x - y| = 66 := 
sorry

end minimal_abs_diff_l782_782077


namespace eval_exp_l782_782421

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l782_782421


namespace solution_set_of_inequality_l782_782051

-- Define the function f(x)
def f (x: ℝ) : ℝ := 2017^x + Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 2017 - 2017^(-x)

-- State the theorem
theorem solution_set_of_inequality : 
  {x : ℝ | f (2 * x + 3) + f x > 0} = Ioi (-1) :=
by
  sorry

end solution_set_of_inequality_l782_782051


namespace min_first_row_sum_l782_782747

noncomputable def grid := array (9 * 2004) ℕ -- Define the grid as an array of natural numbers

-- Define each condition outlined in the problem
def filled_nine_times (g : grid) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 2004 → count (λ x, x = n) g.to_list = 9

def diff_not_exceed_three (g : grid) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 9 ∧ 0 ≤ j ∧ j < 2004 →
  ∀ k, 0 ≤ k ∧ k < 9 ∧ 0 ≤ l ∧ l < 2004 →
  |(g.to_list.nth ((i * 2004) + j)).get_or_else 0 - (g.to_list.nth ((k * 2004) + l)).get_or_else 0| ≤ 3

-- Define the sum of the first row
def first_row_sum (g : grid) : ℕ :=
  (list.map (λ i, (g.to_list.nth i).get_or_else 0) (list.range (2004))).sum

-- The main theorem
theorem min_first_row_sum (g : grid) :
  filled_nine_times g →
  diff_not_exceed_three g →
  first_row_sum g = 2005004 :=
begin
  sorry
end

end min_first_row_sum_l782_782747


namespace largest_radius_circle_equation_l782_782603

-- Define the line equation parameter
def line_eq (m : ℝ) : ℝ → ℝ → Prop := λ x y, mx - y - 2m + 1 = 0

-- Define the circle
def circle_eq (r : ℝ) : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = r^2

-- Proof statement: proving the largest radius circle equation
theorem largest_radius_circle_equation : 
  (∃ r : ℝ, ∀ m : ℝ, ∀ x y : ℝ, line_eq m x y → circle_eq r x y) → 
  (∃ (r : ℝ), r^2 = 5) :=
by 
  sorry

end largest_radius_circle_equation_l782_782603


namespace competition_result_l782_782583

def fishing_season_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def first_period_days : ℕ := 30
def second_period_days : ℕ := 60

theorem competition_result :
  let first_fisherman_total := fishing_season_days * first_fisherman_rate in
  let second_fisherman_total := 
    (second_fisherman_rate1 * first_period_days) +
    (second_fisherman_rate2 * second_period_days) +
    (second_fisherman_rate3 * (fishing_season_days - (first_period_days + second_period_days))) in
  second_fisherman_total - first_fisherman_total = 3 :=
by
  sorry

end competition_result_l782_782583


namespace smallest_b_correct_l782_782012

noncomputable def smallest_b : ℤ :=
  if h : ∃ (r s : ℤ), r * s = 1800 ∧ r + s > 0 then
    let ⟨r, s, hrs, hsum⟩ := Classical.choose h in
    min (r + s)
  else
    0

theorem smallest_b_correct : smallest_b = 85 :=
sorry

end smallest_b_correct_l782_782012


namespace no_strategy_wins_more_than_half_probability_l782_782327

theorem no_strategy_wins_more_than_half_probability
  (deck : Finset Card)
  (red_count black_count : ℕ)
  (well_shuffled : deck.shuffled)
  (player_strategy : (Finset Card) → Bool) :
  ∀ r b, red_count = r ∧ black_count = b →
    (∀ red black : ℕ, (red + black = r + b) → 
      (red / (red + black + 1) ≤ 0.5)) :=
sorry

end no_strategy_wins_more_than_half_probability_l782_782327


namespace shape_is_cone_l782_782600

open_locale real_inner_product_space

structure CylindricalCoordinates where
  r : ℝ
  θ : ℝ
  z : ℝ

def shape (k : ℝ) (coords : CylindricalCoordinates) : Prop :=
  coords.z = k * coords.r

theorem shape_is_cone (k : ℝ) : 
  ∀ (coords : CylindricalCoordinates), shape k coords → 
  ∃ apex, coords.r = apex * coords.z := 
  sorry

end shape_is_cone_l782_782600


namespace largest_prime_value_l782_782145

theorem largest_prime_value (p : ℕ) (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hp : Nat.Prime p)
  (h : x^3 + y^3 - 3 * x * y = p - 1) : p ≤ 5 := 
sorry

example : (∃ p x y : ℕ, 0 < x ∧ 0 < y ∧ Nat.Prime p ∧ (x^3 + y^3 - 3 * x * y = p - 1) ∧ p = 5) := 
begin
  use [5, 2, 2],
  repeat { split },
  { exact nat.succ_pos 1 },
  { exact nat.succ_pos 1 },
  { norm_num },
  { norm_num }
end

end largest_prime_value_l782_782145


namespace f_odd_function_f_range_neg2_to_2_l782_782918

-- Define the conditions for the function f
variable {f : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, x ∈ [-2,2] → y ∈ [-2,2] → f(x + y) = f(x) + f(y))
variable (h2 : ∀ x : ℝ, x > 0 → x ∈ [-2,2] → f(x) > 0)
variable (h3 : ∀ x : ℝ, x ∈ [-2,2] → f(1) = 3)

-- Part I: Prove that f is an odd function
theorem f_odd_function : ∀ x : ℝ, x ∈ [-2,2] → f(-x) = -f(x) :=
by
  sorry

-- Part II: Find the range of f on [-2, 2] given f(1) = 3
theorem f_range_neg2_to_2 : Set.range (λ x, f x) = [-6,6] :=
by
  sorry

end f_odd_function_f_range_neg2_to_2_l782_782918


namespace star_commutative_l782_782142

variable {ℝ : Type*} [AddCommGroup ℝ] [Mul ℝ] [DistribLeft ℝ]

-- Define the operation * and the given condition
def star (x y : ℝ) := sorry 
axiom operation_property : ∀ x y z : ℝ, star x (y + z) = (star y x) + (star z x)

-- Prove that the operation is commutative
theorem star_commutative : ∀ u v : ℝ, star u v = star v u :=
  sorry

end star_commutative_l782_782142


namespace no_repeated_stock_value_l782_782614

theorem no_repeated_stock_value (n : ℕ) :
  ∀ P : ℝ, ∃ k l : ℕ, P * (1 + (n : ℝ) / 100) ^ k * (1 - (n : ℝ) / 100) ^ l = P → false :=
by
  intro P k l h
  sorry

end no_repeated_stock_value_l782_782614


namespace incorrect_unit_vector_statement_l782_782293

-- Definitions of the conditions
def length_eq (v w : ℝ^3) : Prop :=
  ‖v‖ = ‖w‖

def magnitude_comparable (v w : ℝ^3) : Prop :=
  true  -- This implies magnitudes can be compared, we assume it as always true

def nonzero_vector_translatable (v : ℝ^3) : Prop :=
  v ≠ 0  -- Non-zero vectors can be translated parallelly

-- The main theorem: statement B is incorrect
theorem incorrect_unit_vector_statement (v : ℝ^3) (u w : ℝ^2) (x y z : ℝ^3) :
  length_eq x y ∧ magnitude_comparable v w ∧ nonzero_vector_translatable z → 
  ¬ (u = v → ‖u‖ = 1 ∧ ‖v‖ = 1 → u = v) :=
by sorry

end incorrect_unit_vector_statement_l782_782293


namespace skiing_ratio_l782_782709

theorem skiing_ratio (S : ℕ) (H1 : 4000 ≤ 12000) (H2 : S + 4000 = 12000) : S / 4000 = 2 :=
by {
  sorry
}

end skiing_ratio_l782_782709


namespace hildas_age_l782_782223

theorem hildas_age :
  ∃ x, x ∈ {25, 29, 31, 33, 37, 39, 42, 45, 48, 50} ∧
       ((∃ a b, a, b ∈ {25, 29, 31, 33, 37, 39, 42, 45, 48, 50} ∧ (a = x - 2 ∧ b = x + 2)) ∧
        x % 2 ≠ 0 ∧ -- not even
        (∃ d > 1, d < x ∧ d ∣ x) ∧ -- composite
        (∃ l, l = {n ∈ {25, 29, 31, 33, 37, 39, 42, 45, 48, 50} | n > x} ∧ l.card * 4 ≥ ({25, 29, 31, 33, 37, 39, 42, 45, 48, 50}).card)) :=
sorry

end hildas_age_l782_782223


namespace max_S_at_10_l782_782920

-- Define the sum of the first n terms of an arithmetic sequence
def S (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Given conditions
variables (a₁ d : ℝ)
axiom S20_gt_0 : S a₁ d 20 > 0
axiom S21_lt_0 : S a₁ d 21 < 0

-- Proof declaration
theorem max_S_at_10 : ∃ (n : ℕ), n = 10 ∧ ∀ m : ℕ, S a₁ d m ≤ S a₁ d 10 := by
  sorry

end max_S_at_10_l782_782920


namespace max_value_of_f_l782_782019

-- Define the function f(x)
def f (x : ℝ) : ℝ := min (min (4 * x + 1) (x + 2)) (-2 * x + 4)

-- State the theorem with our conditions and the correct answer
theorem max_value_of_f : ∃ x : ℝ, f x = 8 / 3 :=
by
  sorry

end max_value_of_f_l782_782019


namespace relationship_among_a_b_c_l782_782168

noncomputable def a : ℝ := (Real.sqrt 2) / 2 * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b : ℝ := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c : ℝ := (Real.sqrt 3) / 2

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l782_782168


namespace calculate_OP_from_centroid_property_l782_782156

theorem calculate_OP_from_centroid_property
  (OQ BP : ℝ) (hOQ : OQ = 5) (hBP : BP = 18) :
  let OP := (2 / 3) * BP in
  OP = 12 :=
by
  -- Skip the proof
  sorry

end calculate_OP_from_centroid_property_l782_782156


namespace circle_radius_is_one_l782_782319

-- Defining the conditions of the problem
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (AC BC : ℝ) (r : ℝ)

-- Assume the given conditions
axiom AC_val : AC = 5
axiom BC_val : BC = 3
axiom r_is_int : ∃ (r : ℤ), (r : ℝ) = r

-- Define the statement to prove
theorem circle_radius_is_one (h : AC = 5) (hB : BC = 3) (r_is_int : ∃ (r : ℤ), (r : ℝ) = r) : r = 1 :=
sorry

end circle_radius_is_one_l782_782319


namespace acute_angle_3_30_acute_angle_3_30_correct_l782_782761

theorem acute_angle_3_30 :
  let degree_per_hour := 360 / 12
  let minute_angle := 6 * degree_per_hour
  let hour_angle := 3 * degree_per_hour + degree_per_hour / 2
  let angle_between_hands := minute_angle - hour_angle
  abs angle_between_hands < 180

theorem acute_angle_3_30_correct :
  let degree_per_hour := 360 / 12
  let minute_angle := 6 * degree_per_hour
  let hour_angle := 3 * degree_per_hour + degree_per_hour / 2
  let angle_between_hands := minute_angle - hour_angle
  abs angle_between_hands = 75 :=
by
  let degree_per_hour := 360 / 12
  let minute_angle := 6 * degree_per_hour
  let hour_angle := 3 * degree_per_hour + degree_per_hour / 2
  let angle_between_hands := minute_angle - hour_angle
  have h1 : abs angle_between_hands = 75 := by
    sorry
  exact h1

end acute_angle_3_30_acute_angle_3_30_correct_l782_782761


namespace johns_electric_fan_usage_l782_782151

-- Definitions based on the given conditions
def power_rating : ℕ := 75  -- watts
def hours_per_day : ℕ := 8  -- hours
def total_energy_consumed : ℕ := 18 -- kWh

-- We need to prove that:
-- Given the power rating, hours per day, and total energy consumed,
-- the number of days considered is 30.
theorem johns_electric_fan_usage : 
  let daily_consumption := (power_rating * hours_per_day) / 1000 in
  (total_energy_consumed / daily_consumption) = 30 :=
by
  -- The proof would be constructed here
  sorry

end johns_electric_fan_usage_l782_782151


namespace problem_a_range_l782_782836

theorem problem_a_range (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x^2 - 2 * (a - 1) * x - 2 < 0) ↔ (-1 < a ∧ a ≤ 1) :=
by
  sorry

end problem_a_range_l782_782836


namespace max_value_frac_l782_782759

open Real

theorem max_value_frac (t : ℝ) : 
  (∃ t, (∀ t, (3^t - 4*t) * t / 9^t ≤ 1/16)) ∧ 
  (∃ t, (3^t - 4*t) * t / 9^t = 1/16) := 
by 
  sorry

end max_value_frac_l782_782759


namespace payment_duration_l782_782294

-- Defining conditions
def monthly_subscription_cost : ℝ := 14
def monthly_payment : ℝ := monthly_subscription_cost / 2
def total_paid : ℝ := 84

-- Theorem stating the proof problem
theorem payment_duration (h1 : monthly_payment = 7) (h2 : total_paid = 84) :
  (total_paid / monthly_payment = 12) :=
by
  -- Provide sorry as we only need the statement
  sorry

end payment_duration_l782_782294


namespace find_abs_z_l782_782185

open Complex

variable {z w : ℂ}

def cond1 := abs (3 * z - w) = 30
def cond2 := abs (z + 3 * w) = 15
def cond3 := abs (z - w) = 10

theorem find_abs_z (hz : cond1) (hw : cond2) (hv : cond3) : abs z = 9 := sorry

end find_abs_z_l782_782185


namespace intervals_of_monotonicity_maximum_area_of_acute_triangle_l782_782170

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - cos (x + π / 4)^2

-- Problem 1: Intervals of monotonicity
theorem intervals_of_monotonicity : 
  (∀ k ∈ ℤ, ∀ x ∈ [k * π - π / 4, k * π + π / 4], f' x > 0) ∧
  (∀ k ∈ ℤ, ∀ x ∈ [k * π + π / 4, k * π + 3 * π / 4], f' x < 0) :=
sorry

-- Problem 2: Maximum area of an acute triangle ABC
theorem maximum_area_of_acute_triangle 
  (A B C : ℝ) (a b c : ℝ) (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (sides : a = 1) (fx_zero : f (A / 2) = 0) : 
  let area := 1/2 * b * c * sin A in
  area ≤ (2 + sqrt 3) / 4 :=
sorry

end intervals_of_monotonicity_maximum_area_of_acute_triangle_l782_782170


namespace hyperbola_eccentricity_sqrt_two_l782_782576

-- Given the hyperbola equation and conditions
variables (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_asymptotes : b / a = 1)

-- Define the eccentricity formula for the hyperbola
def eccentricity (a b : ℝ) : ℝ := sqrt (1 + (b / a) ^ 2)

-- Lean statement to prove that the eccentricity is sqrt(2)
theorem hyperbola_eccentricity_sqrt_two :
  eccentricity a b = sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_sqrt_two_l782_782576


namespace factor_expression_l782_782402

theorem factor_expression (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) :=
by
  sorry

end factor_expression_l782_782402


namespace isosceles_triangle_AUV_l782_782096

-- Setup the geometric definitions for the problem.
variables {A B C X Y O₁ O₂ U V : Type*}

-- Required conditions from provided hypothesis.
variables [AffineSpace ℝ ℝ] [Plane ℝ ℝ ℝ]

-- Definitions
def points_on_line_BC (B C X Y : Type*) : Prop := collinear ℝ {B, C, X} ∧ collinear ℝ {B, C, Y}

def power_of_point_equation (B C X Y A : Type*) : Prop :=
  (distance B X) * (distance A C) = (distance C Y) * (distance A B)

def circumcenter (O : Type*) (A B C : Type*) : Prop :=
  is_center_of_circumcircle O ({A, B, C} : set Type*)

def intersects_at (L₁ L₂ : Type*) (P : Type*) := are_intersecting ℝ L₁ L₂ ∧ point_on_line P L₂

-- Formalize the proof to show triangle AUV is isosceles under the given conditions.
theorem isosceles_triangle_AUV (h1: points_on_line_BC B C X Y)
(h2: power_of_point_equation B C X Y A)
(h3: circumcenter O₁ {A, C, X}) 
(h4: circumcenter O₂ {A, B, Y})
(h5: intersects_at (line_through O₁ O₂) (line_through A B) U)
(h6: intersects_at (line_through O₁ O₂) (line_through A C) V) : 
is_isosceles_triangle A U V :=
sorry

end isosceles_triangle_AUV_l782_782096


namespace probability_exactly_3_tails_l782_782622

noncomputable def binomial_probability_3_tails : ℚ :=
  let n := 8
  let k := 3
  let p := 3/5
  let q := 2/5
  (nat.choose n k : ℚ) * p^k * q^(n-k)

theorem probability_exactly_3_tails : 
  binomial_probability_3_tails = 48624 / 390625 := 
by
  -- Expected result: prob_exactly_3_tails = 48624 / 390625
  sorry

end probability_exactly_3_tails_l782_782622


namespace constant_sequence_value_bound_seq_an_monotone_decreasing_seq_l782_782256

noncomputable section

-- Problem (I)
def constant_a (a : ℝ) : Prop :=
∀ n : ℕ, a = 3 * a - 3 * a^2

theorem constant_sequence_value :
  constant_a a -> a = 2 / 3 :=
sorry

-- Problem (II)
def seq_an (a1 : ℝ) : ℕ → ℝ
| 0       := a1
| (n + 1) := 3 * seq_an a1 n - 3 * (seq_an a1 n)^2

theorem bound_seq_an (a1 : ℝ) (h : a1 = 1 / 2) (n : ℕ) :
  2 / 3 < seq_an a1 (2 * n) ∧ seq_an a1 (2 * n) ≤ 3 / 4 :=
sorry

-- Problem (III)
theorem monotone_decreasing_seq (a1 : ℝ) (h : a1 = 1 / 2) :
  ∀ n : ℕ, seq_an a1 (2 * (n + 1)) < seq_an a1 (2 * n) :=
sorry

end constant_sequence_value_bound_seq_an_monotone_decreasing_seq_l782_782256


namespace tan_of_cos_alpha_l782_782499

open Real

theorem tan_of_cos_alpha (α : ℝ) (h1 : cos α = 3 / 5) (h2 : -π < α ∧ α < 0) : tan α = -4 / 3 :=
sorry

end tan_of_cos_alpha_l782_782499


namespace total_highlighters_l782_782986

theorem total_highlighters :
  let pink_highlighters := 47
  let yellow_highlighters := 36
  let blue_highlighters := 21
  let orange_highlighters := 15
  let green_highlighters := 27
  pink_highlighters + yellow_highlighters + blue_highlighters + orange_highlighters + green_highlighters = 146 :=
by
  let pink_highlighters := 47
  let yellow_highlighters := 36
  let blue_highlighters := 21
  let orange_highlighters := 15
  let green_highlighters := 27
  show pink_highlighters + yellow_highlighters + blue_highlighters + orange_highlighters + green_highlighters = 146,
  sorry

end total_highlighters_l782_782986


namespace choose_3_out_of_10_l782_782116

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 := by
  sorry

end choose_3_out_of_10_l782_782116


namespace kernel_count_in_final_bag_l782_782659

namespace PopcornKernelProblem

def percentage_popped (popped total : ℕ) : ℤ := ((popped : ℤ) * 100) / (total : ℤ)

def first_bag_percentage := percentage_popped 60 75
def second_bag_percentage := percentage_popped 42 50
def final_bag_percentage (x : ℕ) : ℤ := percentage_popped 82 x

theorem kernel_count_in_final_bag :
  (first_bag_percentage + second_bag_percentage + final_bag_percentage 100) / 3 = 82 := 
sorry

end PopcornKernelProblem

end kernel_count_in_final_bag_l782_782659


namespace pure_water_needed_l782_782071

theorem pure_water_needed :
  ∃ w : ℝ, 
    50 * 0.4 / (50 + w) = 0.25 ∧
    w = 30 :=
by
  have h₁ : 0.4 * 50 = 20 := by norm_num
  have h₂ : (20 : ℝ) / (50 + 30) = 0.25 := by norm_num
  have h₃ : 30 = (20 - 12.5) / 0.25 := by norm_num
  use 30
  simp [h₁, h₂, h₃]
  norm_num
  sorry

end pure_water_needed_l782_782071


namespace rain_all_three_days_is_six_percent_l782_782249

-- Definitions based on conditions from step a)
def P_rain_friday : ℚ := 2 / 5
def P_rain_saturday : ℚ := 1 / 2
def P_rain_sunday : ℚ := 3 / 10

-- The probability it will rain on all three days
def P_rain_all_three_days : ℚ := P_rain_friday * P_rain_saturday * P_rain_sunday

-- The Lean 4 theorem statement
theorem rain_all_three_days_is_six_percent : P_rain_all_three_days * 100 = 6 := by
  sorry

end rain_all_three_days_is_six_percent_l782_782249


namespace isosceles_triangle_solution_l782_782597

noncomputable def isosceles_sides : (a b : ℝ) → (a = 25) ∧ (b = 10) → (sides : list ℝ)
| a b ⟨h₁, h₂⟩ := [a, a, b]

theorem isosceles_triangle_solution
(perimeter : ℝ) (iso_condition : ∃ a b, a = 25 ∧ b = 10 ∧ 2 * a + b = perimeter)
(intersect_uninscribed_boundary : ∀ (O M : ℝ) (ratio : ℝ), O = 2 / 3 * M) : 
sides 
       ∃ a b, (a = 25) ∧ (b = 10) ∧ (perimeter = 60)  :=
begin
  sorry
end

end isosceles_triangle_solution_l782_782597


namespace KatieMarbles_l782_782153

variable {O P : ℕ}

theorem KatieMarbles :
  13 + O + P = 33 → P = 4 * O → 13 - O = 9 :=
by
  sorry

end KatieMarbles_l782_782153


namespace competition_result_l782_782582

def fishing_season_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def first_period_days : ℕ := 30
def second_period_days : ℕ := 60

theorem competition_result :
  let first_fisherman_total := fishing_season_days * first_fisherman_rate in
  let second_fisherman_total := 
    (second_fisherman_rate1 * first_period_days) +
    (second_fisherman_rate2 * second_period_days) +
    (second_fisherman_rate3 * (fishing_season_days - (first_period_days + second_period_days))) in
  second_fisherman_total - first_fisherman_total = 3 :=
by
  sorry

end competition_result_l782_782582


namespace backpacking_trip_cooks_l782_782131

theorem backpacking_trip_cooks :
  nat.choose 10 3 = 120 :=
sorry

end backpacking_trip_cooks_l782_782131


namespace inequality_solution_set_range_of_expr_l782_782939

-- Definition of the function f
def f (x : ℝ) : ℝ := abs (x + 1) + 2 * abs (x - 1)

-- Lean 4 statement for problem 1
theorem inequality_solution_set : { x : ℝ | f x ≤ 4 } = set.Icc (-1) (5/3) :=
sorry

-- Lean 4 statement for problem 2
theorem range_of_expr (a b : ℝ) (h₁ : 1 * a + 2 * b = 4) (h₂ : 0 < a) (h₃ : 0 < b) : 
  2 / a + 1 / b ∈ set.Ici (2 : ℝ) :=
sorry

end inequality_solution_set_range_of_expr_l782_782939


namespace real_product_iff_imag_product_iff_l782_782480

section ComplexProductConditions

variables {a b c d : ℝ}

-- Define the complex numbers
def z1 := complex.mk a b
def z2 := complex.mk c d

-- Define the product of the two complex numbers
noncomputable def z1_mul_z2 := complex.mk (a * c - b * d) (a * d + b * c)

-- Proof problems stated as Lean theorems
theorem real_product_iff : (a * d + b * c = 0) ↔ (z1_mul_z2.im = 0) :=
sorry

theorem imag_product_iff : (a * c - b * d = 0) ↔ (z1_mul_z2.re = 0) :=
sorry

end ComplexProductConditions

end real_product_iff_imag_product_iff_l782_782480


namespace triangle_sides_arith_prog_sum_l782_782679

theorem triangle_sides_arith_prog_sum (y : ℝ) (d e f : ℕ) (h_d : d = 8) (h_e : e = 39) (h_f : f = 0) :
  (∀ a b c : ℝ, a + b + c = 180 ∧ (a + k) = b ∧ (b + k) = c) →
  (5^2 + 7^2 - 5 * y * real.cos 60° = y^2 ∨ 7^2 + 5^2 - 7 * 5 * real.cos 60° = y^2) →
  d + e + f = 47 :=
by
  sorry

end triangle_sides_arith_prog_sum_l782_782679


namespace largest_possible_n_l782_782230

noncomputable def largest_n_base_10 : ℕ :=
  let n : ℕ := 64 * 1 + 8 * 7 + 0 -- Expected solution steps directly encoded
  n

theorem largest_possible_n :
  ∃ (A B C : ℕ),
    A ∈ {0, 1, 2, 3, 4, 5, 6, 7} ∧ -- Conditions for base 8 valid digits
    B ∈ {0, 1, 2, 3, 4, 5, 6, 7} ∧
    C ∈ {0, 2, 4, 6} ∧ -- C is even, base 8 digit
    (64 * A + 8 * B + C = 81 * C + 9 * B + A) ∧
    (1 ∈ {0, 1, 2, 3, 4, 5, 6, 7} ∧ 7 ∈ {0, 1, 2, 3, 4, 5, 6, 7}) ∧
    64 * A + 8 * (63 * A - 80 * C) + C = n ∧ -- Correct transformation for B
    n = 120 :=
begin
  use [1, 7, 0],
  split,
  {
    simp,
  },
  split,
  {
    simp,
  },
  split,
  {
    simp,
  },
  split,
  {
    simp,
  },
  split,
  {
    simp,
  },
  {
    simp,
  },
  sorry
end

end largest_possible_n_l782_782230


namespace intersection_point_moves_straight_line_l782_782528

-- Define the points and conditions
variables {A B C D P : Type} [EuclideanGeometry A B C D P]
variable (M N : EuclideanMidpoint A B C D P)


-- Define the proof problem
theorem intersection_point_moves_straight_line
  (AB_parallel : parallel AB CD)
  (AB_fixed : fixed_length AB)
  (CD_parallel : parallel CD AB)
  (CD_fixed : fixed_length CD)
  (BC_length_fixed : fixed_length BC)
  (BC_pivot_at_B : pivot_at BC B)
  (D_moves_parallel_AB : moves_parallel D (line_through AB)) :
  moves_on_straight_line (intersection_point AC BD) :=
sorry

end intersection_point_moves_straight_line_l782_782528


namespace mf_perp_cd_l782_782610

variables {A B C D M F : Point}
variables {AD BD BC CD : Line}

-- Given definitions based on conditions
def is_trapezoid (A B C D : Point) (AD BC : Line) := 
  -- Define a trapezoid condition
  sorry

def is_midpoint (M : Point) (A D : Line) := 
  -- Define the midpoint condition
  sorry

def right_angle (A B D : Point) := 
  -- Define the right angle condition
  sorry

def is_isosceles (B C D : Point) := 
  -- Define isosceles triangle condition
  sorry

def perpendicular (A B C : Point) := 
  -- Define perpendicular condition
  sorry

-- Lean 4 statement
theorem mf_perp_cd 
  (H1 : is_trapezoid A B C D<AD BC>) 
  (H2 : is_midpoint M AD) 
  (H3 : right_angle A B D) 
  (H4 : is_isosceles B C D) 
  (H5 : perpendicular B C F) : 
  perpendicular M F C :=
by 
  sorry

end mf_perp_cd_l782_782610


namespace nat_with_8_divisors_l782_782848

def has_8_divisors (n : ℕ) : Prop :=
  ∃ (p : ℕ) (hp : Prime p), n = p^7 ∨ 
  ∃ (p1 p2 : ℕ) (hp1 : Prime p1) (hp2 : Prime p2) (hp1p2 : p1 ≠ p2), n = p1 * p2^3 ∨ 
  ∃ (p1 p2 p3 : ℕ) (hp1 : Prime p1) (hp2 : Prime p2) (hp3 : Prime p3) (hp1p2 : p1 ≠ p2) (hp1p3 : p1 ≠ p3) (hp2p3 : p2 ≠ p3), n = p1 * p2 * p3

theorem nat_with_8_divisors (a : ℕ) (h : number_of_divisors a = 8) : has_8_divisors a :=
sorry

end nat_with_8_divisors_l782_782848


namespace quartic_sum_of_roots_l782_782248

theorem quartic_sum_of_roots :
  ∀ (Q : Polynomial ℂ) (φ : ℂ) 
  (hQ_monic : Q.monic) 
  (hQ_degree : Q.natDegree = 4)
  (hQ_realCoeffs : Q.coeffs ∈ Set ℝ) 
  (hz1 : Q.hasRoot (exp (Complex.I * φ)))
  (hz2 : Q.hasRoot (cos φ + sin (2 * φ) * Complex.I))
  (h0_lt_phi_and_phi_lt_pi_six : 0 < φ ∧ φ < π / 6) 
  (hQuadrilateral : -- some condition that relates the area of the quadrilateral of roots to Q(0)),
  ∑ i in (Q.roots : Multiset ℂ), i = sqrt 3 :=
by
  sorry

end quartic_sum_of_roots_l782_782248


namespace simplify_expression_l782_782028

theorem simplify_expression (a : ℝ) (h : a > 1) : 
  (\frac{2 * log (log (a ^ 100))}{2 + log (log a)} + (1 / 9) ^ (-1 / 2)) = 5 :=
sorry

end simplify_expression_l782_782028


namespace val_total_money_l782_782754

theorem val_total_money : 
  ∀ (nickels_initial dimes_initial nickels_found : ℕ),
    nickels_initial = 20 →
    dimes_initial = 3 * nickels_initial →
    nickels_found = 2 * nickels_initial →
    (nickels_initial * 5 + dimes_initial * 10 + nickels_found * 5) / 100 = 9 :=
by
  intros nickels_initial dimes_initial nickels_found h1 h2 h3
  sorry

end val_total_money_l782_782754


namespace gain_percent_l782_782776

variable {MP : ℝ}

noncomputable def CP : ℝ := 0.64 * MP
noncomputable def SP : ℝ := MP * 0.86

theorem gain_percent (MP : ℝ) : ( ((SP - CP) / CP) * 100 ) = 34.375 := by
  sorry

end gain_percent_l782_782776


namespace contradiction_assumption_l782_782769

theorem contradiction_assumption (x y : ℝ) (h1 : x > y) : ¬ (x^3 ≤ y^3) := 
by
  sorry

end contradiction_assumption_l782_782769


namespace sum_of_integers_eq_17_l782_782251

theorem sum_of_integers_eq_17 (a b : ℕ) (h1 : a * b + a + b = 87) 
  (h2 : Nat.gcd a b = 1) (h3 : a < 15) (h4 : b < 15) (h5 : Even a ∨ Even b) :
  a + b = 17 := 
sorry

end sum_of_integers_eq_17_l782_782251


namespace tan_pi_over_4_minus_alpha_eq_neg_3_l782_782901

theorem tan_pi_over_4_minus_alpha_eq_neg_3 (α : ℝ) 
  (h : sin α = 2 * sin (3 * π / 2 - α)) : tan (π / 4 - α) = -3 := 
by 
  sorry

end tan_pi_over_4_minus_alpha_eq_neg_3_l782_782901


namespace bees_multiple_l782_782200

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end bees_multiple_l782_782200


namespace find_values_l782_782949

noncomputable def vector_dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

theorem find_values 
  (A B C : ℝ × ℝ)
  (P : ℝ × ℝ)
  (λ μ : ℝ)
  (hA : A = (1, -1))
  (hB : B = (3, 0))
  (hC : C = (2, 1))
  (hP : P = (2*λ + μ, λ + 2*μ))
  (hAP_AB : vector_dot_product P (B.1 - A.1, B.2 - A.2) = 0)
  (hAP_AC : vector_dot_product P (C.1 - A.1, C.2 - A.2) = 3)
  : 
  (vector_dot_product (B.1 - A.1, B.2 - A.2) (C.1 - A.1, C.2 - A.2) = 4) ∧ (λ + μ = 1 / 3) :=
by
  sorry

end find_values_l782_782949


namespace total_arrangements_problem1_total_arrangements_problem2_total_arrangements_problem3_total_arrangements_problem4_total_arrangements_problem5_total_arrangements_problem6_l782_782032

def problem1 : Nat :=
  let total_groups := Nat.choose 7 5
  let permutations := (5!).val
  total_groups * permutations

def problem2 : Nat :=
  let front_row := Finset.range 7
  let front_permutations := front_row.card.factorial
  let back_permutations := (4!).val
  front_permutations * back_permutations

def problem3 (person : Fin 7) : Nat :=
  let remaining_permutations := (6!).val
  let internal_positions := 5
  remaining_permutations * internal_positions

def problem4 : Nat :=
  let female_group_permutations := (4!).val
  let total_permutations := (4!).val
  female_group_permutations * total_permutations

def problem5 : Nat :=
  let female_permutations := (4!).val
  let slots := Finset.range 5
  let male_arrangements := slots.card.factorial
  female_permutations * male_arrangements

def problem6 (A B : Fin 7) : Nat :=
  let middle_group_permutations := (5!).val
  let flip_arrangements := 2
  let remaining_permutations := (2!).val
  middle_group_permutations * flip_arrangements * remaining_permutations

theorem total_arrangements_problem1 : problem1 = 2520 := sorry

theorem total_arrangements_problem2 : problem2 = 5040 := sorry

theorem total_arrangements_problem3 (person : Fin 7) : problem3 person = 3600 := sorry

theorem total_arrangements_problem4 : problem4 = 576 := sorry

theorem total_arrangements_problem5 : problem5 = 1440 := sorry

theorem total_arrangements_problem6 (A B : Fin 7) : problem6 A B = 720 := sorry

end total_arrangements_problem1_total_arrangements_problem2_total_arrangements_problem3_total_arrangements_problem4_total_arrangements_problem5_total_arrangements_problem6_l782_782032


namespace choose_3_from_10_is_120_l782_782112

theorem choose_3_from_10_is_120 :
  nat.choose 10 3 = 120 :=
by {
  -- proof would go here
  sorry
}

end choose_3_from_10_is_120_l782_782112


namespace stock_yield_percentage_l782_782787

noncomputable def annual_dividend (par_value : ℝ) (rate : ℝ) : ℝ :=
  rate * par_value / 100

noncomputable def yield_percentage (annual_dividend : ℝ) (market_value : ℝ) : ℝ :=
  (annual_dividend / market_value) * 100

theorem stock_yield_percentage :
  let par_value := 100
  let market_value := 75
  let rate := 6
  annual_dividend par_value rate / market_value * 100 = 8 :=
by
  let par_value := 100
  let market_value := 75
  let rate := 6
  have h1 : annual_dividend par_value rate = 6 := by
    unfold annual_dividend
    sorry
  have h2 : annual_dividend par_value rate / market_value * 100 = 8 := by
    unfold yield_percentage
    rw [h1]
    sorry
  exact h2

end stock_yield_percentage_l782_782787


namespace count_positive_integers_in_given_numbers_l782_782674

def is_positive_integer (n : ℚ) : Prop := n > 0 ∧ n.den = 1

def given_rational_numbers := [-2, -1, 0, -1/2, 2, 1/3]

theorem count_positive_integers_in_given_numbers :
  (finset.filter is_positive_integer (finset.of_list given_rational_numbers)).card = 1 :=
by
  sorry

end count_positive_integers_in_given_numbers_l782_782674


namespace daily_wage_each_working_day_l782_782814

variable (x : ℝ)
variable (idle_days working_days : ℕ)
variable (total_deduction total_earnings : ℝ)

-- Conditions as definitions
def number_idle_days := 40
def number_working_days := 20
def deduction_per_idle_day := 3
def total_earnings_after_60_days := 280
def total_deduction := number_idle_days * deduction_per_idle_day

-- Lean 4 statement for proof problem
theorem daily_wage_each_working_day (h : total_earnings = total_earnings_after_60_days) :
  20 * x - 120 = total_earnings → x = 20 := 
by 
  sorry

end daily_wage_each_working_day_l782_782814


namespace no_winning_strategy_exceeds_half_probability_l782_782330

-- Provided conditions
def well_shuffled_standard_deck : Type := sorry -- Placeholder for the deck type

-- Statement of the problem
theorem no_winning_strategy_exceeds_half_probability :
  ∀ strategy : (well_shuffled_standard_deck → ℕ → bool),
    let r := 26 in -- Assuming a standard deck half red cards (26 red)
    let b := 26 in -- and half black cards (26 black)
    let P_win := (r : ℝ) / (r + b) in       
    P_win ≤ 0.5 :=
by
  sorry

end no_winning_strategy_exceeds_half_probability_l782_782330


namespace geometry_proof_l782_782604

noncomputable def part1 : Prop :=
  ∀ (x y : ℝ), (x + 1) * (x - 1) + y^2 = 1 → x^2 + y^2 = 1

noncomputable def part2 : Prop :=
  ∃ (T : ℝ × ℝ), T.1^2 = 2 * T.2 ∧ T.2 ≥ 0 ∧
  ∀ (A B : ℝ × ℝ), A ≠ B ∧ (A^2 + A.2^2 = 1) ∧ (B^2 + B.2^2 = 1) ∧ tangent_line A B T →
    ∃ (D : ℝ × ℝ), D = (1 / 2, 0) ∧
    maximum_area_triangle D A B = sqrt(2) / 2

-- properties to be defined within your scope
axiom tangent_line : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop
axiom maximum_area_triangle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ

-- Main theorem combining both parts
theorem geometry_proof : part1 ∧ part2 :=
by {
  -- Skip the proof
  sorry
}

end geometry_proof_l782_782604


namespace average_books_per_student_l782_782993

theorem average_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (h_total : total_students = 40)
  (h_0_books : students_0_books = 2)
  (h_1_book : students_1_book = 12)
  (h_2_books : students_2_books = 14)
  (h_at_least_3_books : students_at_least_3_books = total_students - (students_0_books + students_1_book + students_2_books))
  (h_nonnegative : 0 ≤ students_at_least_3_books) :
  let total_books := (students_0_books * 0) + (students_1_book * 1) + (students_2_books * 2) + (students_at_least_3_books * 3)
  in (total_books : ℝ) / total_students = 1.9 :=
by
  have h_students_28 := by
    rw [h_0_books, h_1_book, h_2_books]
    exact rfl
  have h_students_12 := by
    rw [h_total, h_students_28, ←Nat.sub_eq_iff_eq_add]
    apply h_at_least_3_books
    exact h_nonnegative
  have total_books := (students_0_books * 0) + (students_1_book * 1) + (students_2_books * 2) + (students_at_least_3_books * 3)
  have h_total_books_76 : total_books = 76 :=
    by rw [h_0_books, h_1_book, h_2_books, h_students_12]
    exact rfl
  show (total_books : ℝ) / total_students = 1.9,
  by
    rw [h_total, h_total_books_76],
    norm_num

end average_books_per_student_l782_782993


namespace first_number_in_sequence_l782_782265

noncomputable def sequence : ℕ → ℕ
| 0 := 5
| 1 := 15 
| 2 := 17
| 3 := 51
| 4 := 53
| 5 := 159
| 6 := 161
| (n + 7) := sequence n * 3 -- Defining further terms to follow the pattern, though not required for our proof

theorem first_number_in_sequence : sequence 0 = 5 :=
by
  -- The proof goes here
  sorry

end first_number_in_sequence_l782_782265


namespace measure_angle_PYV_l782_782141

def problem_conditions (P Q R S V W Y Q' R': Type) [AffineSpace P Q R S] [AffineSpace V W Y] : Prop :=
  is_rectangle P Q R S 
  ∧ angle V W Q = 125
  ∧ moved_to R R'
  ∧ moved_to Q Q'
  ∧ intersects R' V (P W) Y

theorem measure_angle_PYV (P Q R S V W Y Q' R': Type) [AffineSpace P Q R S] [AffineSpace V W Y]
  (h : problem_conditions P Q R S V W Y Q' R') :
  angle P Y V = 110 :=
sorry

end measure_angle_PYV_l782_782141


namespace complement_A_in_B_l782_782911

-- Define the sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

-- Define the complement of A in B
def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Statement to prove
theorem complement_A_in_B :
  complement B A = {0, 1, 4} := by
  sorry

end complement_A_in_B_l782_782911


namespace probability_dart_in_center_pentagon_l782_782792

theorem probability_dart_in_center_pentagon (s : ℝ) : 
  let p := s * Real.cos (36 * Real.pi / 180)
  let A_p := (1 / 4) * Real.sqrt(5 * (5 + 2 * Real.sqrt(5))) * p^2
  let A_d := (5 / 2) * s^2 * Real.sin (72 * Real.pi / 180)
  let P := A_p / A_d
  P = (Real.sqrt(5 * (5 + 2 * Real.sqrt(5))) * (Real.cos (36 * Real.pi / 180))^2) / (10 * Real.sin (72 * Real.pi / 180)) :=
by 
  sorry

end probability_dart_in_center_pentagon_l782_782792


namespace probability_bernardo_larger_l782_782382

-- Define the sets from which Bernardo and Silvia are picking numbers
def set_B : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def set_S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to calculate the probability as described in the problem statement
def bernardo_larger_probability : ℚ := sorry -- The step by step calculations will be inserted here

-- Main theorem stating what needs to be proved
theorem probability_bernardo_larger : bernardo_larger_probability = 61 / 80 := 
sorry

end probability_bernardo_larger_l782_782382


namespace choose_3_from_10_is_120_l782_782109

theorem choose_3_from_10_is_120 :
  nat.choose 10 3 = 120 :=
by {
  -- proof would go here
  sorry
}

end choose_3_from_10_is_120_l782_782109


namespace trig_identity_proof_l782_782136

open Real

constants (θ : ℝ)
noncomputable def sin_θ := sin θ
noncomputable def tan_2θ := tan (2 * θ)

theorem trig_identity_proof
  (x y : ℝ) (h : x = 3/5 ∧ y = 4/5 ∧ sqrt (x^2 + y^2) = 1)
  (r : ℝ) (hr : r = sqrt (x^2 + y^2)) :
  sin θ = y / r ∧ tan (2 * θ) = -24/7 := by
  sorry

end trig_identity_proof_l782_782136


namespace explicit_formula_l782_782038

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem explicit_formula (x1 x2 : ℝ) (h1 : x1 ∈ Set.Icc (-1 : ℝ) 1) (h2 : x2 ∈ Set.Icc (-1 : ℝ) 1) :
  f x = x^3 - 3 * x ∧ |f x1 - f x2| ≤ 4 :=
by
  sorry

end explicit_formula_l782_782038


namespace tv_price_comparison_l782_782289

def area (width height : ℕ) : ℕ := width * height

def cost_per_square_inch (cost area : ℕ) : ℚ := cost.toRat / area.toRat

def price_difference (cost1 cost2 area1 area2 : ℕ) : ℚ :=
  cost_per_square_inch(cost1, area1) - cost_per_square_inch(cost2, area2)

theorem tv_price_comparison :
  price_difference 672 1152 (area 24 16) (area 48 32) = 1 := by
  sorry

end tv_price_comparison_l782_782289


namespace find_y_l782_782245

theorem find_y (y : ℝ) : (∃ y : ℝ, (4, y) ≠ (2, -3) ∧ ((-3 - y) / (2 - 4) = 1)) → y = -1 :=
by
  sorry

end find_y_l782_782245


namespace area_of_smaller_circle_l782_782269

noncomputable theory
open Real

variables {r R : ℝ}
variables (P A A' B B' : ℝ)

theorem area_of_smaller_circle
  (h1 : r > 0)
  (h2 : R = 3 * r)
  (h3 : dist P A = 6)
  (h4 : dist A B = 6)
  (h5 : dist P B = dist P A + dist A B) :
  π * r^2 = 36 * π :=
by
  -- Assume r is the radius of the smaller circle
  -- Assume R is the radius of the larger circle
  -- Assume dist P A = 6 (tangent segment)
  -- Assume dist A B = 6 (common tangent)
  -- Assume dist P B = 12 (total distance)
  sorry

end area_of_smaller_circle_l782_782269


namespace probability_correct_l782_782763

/-- 
The set of characters in "HMMT2005".
-/
def characters : List Char := ['H', 'M', 'M', 'T', '2', '0', '0', '5']

/--
The number of ways to choose 4 positions out of 8.
-/
def choose_4_from_8 : ℕ := Nat.choose 8 4

/-- 
The factorial of an integer n.
-/
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

/-- 
The number of ways to arrange "HMMT".
-/
def arrangements_hmmt : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of ways to arrange "2005".
-/
def arrangements_2005 : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of arrangements where both "HMMT" and "2005" appear.
-/
def arrangements_both : ℕ := choose_4_from_8

/-- 
The total number of possible arrangements of "HMMT2005".
-/
def total_arrangements : ℕ := factorial 8 / (factorial 2 * factorial 2)

/-- 
The number of desirable arrangements using inclusion-exclusion.
-/
def desirable_arrangements : ℕ := arrangements_hmmt + arrangements_2005 - arrangements_both

/-- 
The probability of being able to read either "HMMT" or "2005" 
in a random arrangement of "HMMT2005".
-/
def probability : ℚ := (desirable_arrangements : ℚ) / (total_arrangements : ℚ)

/-- 
Prove that the computed probability is equal to 23/144.
-/
theorem probability_correct : probability = 23 / 144 := sorry

end probability_correct_l782_782763


namespace candy_distribution_l782_782651

-- Define the required parameters and conditions.
def num_distinct_candies : ℕ := 9
def num_bags : ℕ := 3

-- The result that we need to prove
theorem candy_distribution :
  (3 ^ num_distinct_candies) - 3 * (2 ^ (num_distinct_candies - 1) - 2) = 18921 := by
  sorry

end candy_distribution_l782_782651


namespace trapezoid_segment_length_is_six_l782_782341

noncomputable def trapezoid_MN_length : ℝ :=
  let BC := 4
  let AD := 12
  let O := (diagonals_intersection A B C D BC AD)
  let MN := (line_through O parallel_to (BC AD))
  segment_length MN (non_parallel_sides A B C D O BC AD)

theorem trapezoid_segment_length_is_six (A B C D : Type) : trapezoid_MN_length = 6 := by
  sorry

end trapezoid_segment_length_is_six_l782_782341


namespace intersection_points_range_l782_782602

theorem intersection_points_range (P : ℝ × ℝ) (α : ℝ) (rho_eq : ℝ → ℝ) (inter_range : set ℝ)
  (P_def : P = (4, 2))
  (line_through_P : ∀ t, ∃ x y, x = 4 + t * Real.cos α ∧ y = 2 + t * Real.sin α)
  (rho_eq_def : rho_eq = λ θ, 4 * Real.cos θ)
  (polar_to_cartesian : ∀ θ, (rho_eq θ)^2 = (ρ θ * Real.cos θ)^2 + (ρ θ * Real.sin θ)^2)
  (curve_cart_eq : ∀ x y, x^2 + y^2 - 4 * x = 0)
  : inter_range = Ioo (4 : ℝ) (4 * Real.sqrt 2) :=
sorry

end intersection_points_range_l782_782602


namespace no_winning_strategy_l782_782333

theorem no_winning_strategy (r b : ℕ) (h1 : r + b = 52) (strategy : ℕ → bool) :
  ∀ k, (strategy k → (r / (r + b : ℝ) ≤ 0.5)) :=
by 
  sorry

end no_winning_strategy_l782_782333


namespace stop_at_quarter_X_l782_782654

theorem stop_at_quarter_X
  (circumference : ℕ)
  (distance_run : ℕ)
  (starting_point : char)
  (quarters : list char)
  (start_quarter : char) 
  (num_of_laps : ℕ)
  (runs_full_laps : bool) 
  : starting_point = start_quarter → circumference = 200 → distance_run = 3000 → starting_point = 'S' → quarters = ['X', 'Y', 'Z', 'W'] → distance_run / circumference = num_of_laps → num_of_laps = 15 → runs_full_laps → start_quarter = 'X' := 
by
  assume h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end stop_at_quarter_X_l782_782654


namespace isosceles_triangle_sides_l782_782591

theorem isosceles_triangle_sides (A B C : Type) [plus : A → A → A] [le : A → A → Prop] [mul : A → A → A] [div : A → A → A] [zero : A] [one : A]
  (triangle_iso : ∀ (x y z : A), plus x y z = 60 → le (plus x z) y → le (plus y z) x → le (plus x y) z → x = y)
  (per_60 : plus (plus A B) C = 60)
  (medians_inter_inscribed : A → A → A → A)
  (centroid_on_incircle : (medians_inter_inscribed A B C) = Inscribed_circle)
  (A : A)
  (B : A)
  (C : A) : 
  A = 25 ∧ B = 25 ∧ C = 10 := 
sorry

end isosceles_triangle_sides_l782_782591


namespace cos_double_angle_l782_782043

variable (α : Real)
variable h : Real.sin (α + Real.pi / 2) = 2 / 3

theorem cos_double_angle :
  Real.cos (2 * α) = -1 / 9 := 
sorry

end cos_double_angle_l782_782043


namespace evaluate_three_cubed_squared_l782_782441

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l782_782441


namespace triangle_side_y_values_l782_782811

theorem triangle_side_y_values (y : ℕ) : (4 < y^2 ∧ y^2 < 20) ↔ (y = 3 ∨ y = 4) :=
by
  sorry

end triangle_side_y_values_l782_782811


namespace is_optimal_number_l782_782753

noncomputable def digits := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Assume n1 and n2 are the two five-digit numbers formed
def n1 := 96420
def n2 := 87531

-- Statement to show that 96420 can be one of the numbers
theorem is_optimal_number (n1 n2 : ℕ) (h1 : digits = nat.digits 10 n1 ++ nat.digits 10 n2)
(h2 : ∀ a b, (nat.digits 10 n1) ⊕ (nat.digits 10 n2) = digits ∧ n1 ≠ n2) :
n1 = 96420 ∨ n2 = 96420 := sorry

end is_optimal_number_l782_782753


namespace product_of_distinct_divisors_count_l782_782634

-- Define the set of positive integer divisors of 90,000
def T : Set ℕ := { d | d ∣ 90000 ∧ d > 0 }

-- We need to establish the statement about the number of products of two distinct elements
-- of the set T. We use the prime factorization of 90,000 to determine the set T.
theorem product_of_distinct_divisors_count :
  ∃ (count : ℕ), count = 401 ∧ ∀ a b ∈ T, a ≠ b → (a * b ∈ T) :=
sorry

end product_of_distinct_divisors_count_l782_782634


namespace lines_perpendicular_l782_782526

noncomputable def l1 (a : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), 2 * p.1 - a * p.2 - 1 = 0
def l : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 + 2 * p.2 = 0
def point : ℝ × ℝ := (1, 1)
def a_val : ℝ := 1

theorem lines_perpendicular (h1 : l1 a_val point) : 
  (∀ (p : ℝ × ℝ), l1 a_val p → (2 * (-1/2) = -1 ∧ l p)) :=
by
  sorry

end lines_perpendicular_l782_782526


namespace volume_of_rectangular_solid_l782_782519

theorem volume_of_rectangular_solid (a b c : ℝ) (h1 : a * b = Real.sqrt 2) (h2 : b * c = Real.sqrt 3) (h3 : c * a = Real.sqrt 6) : a * b * c = Real.sqrt 6 :=
sorry

end volume_of_rectangular_solid_l782_782519


namespace min_other_side_of_rectangle_l782_782657

theorem min_other_side_of_rectangle (a b c d : ℝ)
  (h1 : a + b = 1)
  (h2 : a * c ≥ 1)
  (h3 : a * d ≥ 1)
  (h4 : b * c ≥ 1)
  (h5 : b * d ≥ 2) :
  c + d ≥ 3 + 2 * real.sqrt 2 :=
begin
  sorry
end

end min_other_side_of_rectangle_l782_782657


namespace find_f1987_l782_782186

noncomputable theory

variables {f : ℝ → ℝ}
variable x : ℝ
variable y : ℝ
variable v : ℝ
variable z : ℝ

-- Conditions provided in the problem
axiom condition_i : ∀ (x y v : ℝ), x > y → f(y) - y ≥ v ∧ v ≥ f(x) - x → ∃ z : ℝ, z ∈ set.Icc y x ∧ f(z) = v + z
axiom condition_ii : (∃ x : ℝ, f(x) = 0) ∧ (∀ x : ℝ, f(x) = 0 → x ≤ 0)
axiom condition_iii : f 0 = 1
axiom condition_iv : f 1987 ≤ 1988
axiom condition_v : ∀ x y : ℝ, f(x) * f(y) = f(x * f(y) + y * f(x) - x * y)

-- Prove that f(1987) = 1988
theorem find_f1987 : f 1987 = 1988 :=
sorry

end find_f1987_l782_782186


namespace vector_projection_eq_minus_two_l782_782188

open Real

variables (a b : ℝ^2)

noncomputable def vector_projection (v w : ℝ^2) : ℝ :=
  (v • w) / ∥v∥

theorem vector_projection_eq_minus_two 
  (h1 : (a + b) • (2 • a - b) = -12)
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 4) :
  vector_projection a b = -2 := 
sorry

end vector_projection_eq_minus_two_l782_782188


namespace sequence_100th_term_is_4_l782_782274

theorem sequence_100th_term_is_4 :
  (∃ (f : ℕ → ℕ), (∀ n, f n = match (n + 4) % 5 with
                                 | 0 := n // 5 + 1
                                 | k := (n + 5 - k) // 5 
                                end) ∧ f 100 = 4) :=
sorry

end sequence_100th_term_is_4_l782_782274


namespace tv_price_comparison_l782_782288

def area (width height : ℕ) : ℕ := width * height

def cost_per_square_inch (cost area : ℕ) : ℚ := cost.toRat / area.toRat

def price_difference (cost1 cost2 area1 area2 : ℕ) : ℚ :=
  cost_per_square_inch(cost1, area1) - cost_per_square_inch(cost2, area2)

theorem tv_price_comparison :
  price_difference 672 1152 (area 24 16) (area 48 32) = 1 := by
  sorry

end tv_price_comparison_l782_782288


namespace probability_selecting_A_l782_782276

theorem probability_selecting_A :
  let total_people := 4
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_people
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_A_l782_782276


namespace standard_eq_hyperbola_l782_782340

theorem standard_eq_hyperbola (a b : ℝ) (h₁ : b = (3/4) * a) (h₂ : (4, 6) ∈ { p | p.1^2/a^2 - p.2^2/b^2 = 1 }) :
  ∃ a b, (a^2 = 48) ∧ (b^2 = 27) ∧ (∀ (x y : ℝ), x^2 / 48 - y^2 / 27 = 1) :=
  begin
  sorry,
end

end standard_eq_hyperbola_l782_782340


namespace maximize_angle_acb_l782_782655

theorem maximize_angle_acb
  (O A B C : Point)
  (α β : Line)
  (hα : On α A)
  (hα' : On α B)
  (hβ : On β C)
  (hA_O_not_eq : A ≠ O)
  (hB_O_not_eq : B ≠ O)
  (acute_angle : ∠O α < π / 2) :
  ∃ C, On β C ∧ is_Tangent (circle_through A B) β :=
sorry

end maximize_angle_acb_l782_782655


namespace distinct_paintings_l782_782138

theorem distinct_paintings (n : ℕ) (blue red yellow : ℕ) (f : Fin n → ℕ) :
  n = 8 →
  blue = 4 →
  red = 3 →
  yellow = 1 →
  (∀ i, f i ∈ {blue, red, yellow}) →
  (∃ g : ℕ → Fin n → Fin n, 
      (∀ i, g i = i ∨ ∃ j, g j = symmetric g i) → 
      (∀ i, f (g i) = f i)) →
  ∃ (count : ℕ), count = 72 :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  apply Exists.intro 72
  sorry

end distinct_paintings_l782_782138


namespace evaluate_exp_power_l782_782435

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l782_782435


namespace minimum_value_of_function_l782_782006

noncomputable def target_function (x : ℝ) := x^2 + 12 * x + 108 / x^4

theorem minimum_value_of_function : ∃ x > 0, target_function x = 36 :=
by
  use 3
  split
  -- Prove x > 0
  linarith
  -- Prove target_function x = 36
  unfold target_function
  calc
    3^2 + 12 * 3 + 108 / 3^4 = 9 + 36 + 108 / 81 : by norm_num
    ...                       = 9 + 36 + 108 / 81 : by norm_num
    ...                       = 45                : by norm_num
    ...                       = 45                : by norm_num
    ...                       = 36                : sorry

end minimum_value_of_function_l782_782006


namespace log_identity_l782_782671

theorem log_identity (a x b : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (h₁ : a ≠ 1) :
  (log x / log (x) / log (a * b)) = 1 + (log b / log x) := 
begin
  sorry
end

end log_identity_l782_782671


namespace simple_interest_years_l782_782352

noncomputable def principal : ℝ := 2100
noncomputable def interestDiff : ℝ := 63

theorem simple_interest_years (R N : ℝ) (h1 : principal * N * (R + 1) - principal * N * R = interestDiff) : N = 3 :=
by {
  have h2 : principal * N * (R + 1) = principal * N * R + interestDiff,
  rw [mul_add, ← add_mul] at h2,
  have : principal * N = 2100 * N := rfl,
  have eq₁ : 2100 * N * (R + 1) = 2100 * N * R + 63 := by rwa h2,
  have eq₂ : 2100 * N = 2100 * N := by rfl,
  rw [this, add_sub_cancel'_right] at eq₁,
  replace h1 := eq₁,
  simp [mul_eq_mul_right_iff] at h1,
  cases h1,
  case or.inl {
    exact h1,
  },
  case or.inr {
    exfalso,
    linarith,
  },
}

end simple_interest_years_l782_782352


namespace part1_part2_l782_782887

-- Define P_n as the product of factorials up to n
def P (n : ℕ) : ℕ := ∏ i in (range n).map (λ x, x + 1), factorial i

-- Define the perfect_square predicate
def perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

-- Statement 1: There exists an m such that P_2020 / m! is a perfect square.
theorem part1 : ∃ m : ℕ, perfect_square (P 2020 / factorial m) := sorry

-- Statement 2: There are at least two m such that there are infinitely many n for which P_n / m! is a perfect square.
theorem part2 : ∃ m1 m2 : ℕ, (m1 ≠ m2) ∧ (∃ f1 f2 : ℕ → ℕ, (∀ k1 k2, f1 k1 ≠ f2 k2) ∧ 
∀ n, perfect_square (P n / factorial (m1 + f1 n))) ∧ 
∀ n, perfect_square (P n / factorial (m2 + f2 n)) := sorry

end part1_part2_l782_782887


namespace smallest_distance_z_w_l782_782638

theorem smallest_distance_z_w (z w : ℂ) 
  (h1 : abs (z + (3 + 4 * I)) = 2) 
  (h2 : abs (w - (6 + 10 * I)) = 4) : 
  abs (z - w) ≥ Real.sqrt 277 - 6 :=
sorry

end smallest_distance_z_w_l782_782638


namespace minimum_soldiers_l782_782107

theorem minimum_soldiers (m n : ℕ)
  (h1 : 1 ≤ (m * n) / 100)
  (h2 : 0.3 * m ≤ (m * n) / 100)
  (h3 : 0.4 * n ≤ (m * n) / 100) :
  m ≥ 40 ∧ n ≥ 30 → m * n = 1200 :=
by
  sorry

end minimum_soldiers_l782_782107


namespace problem_l782_782521

-- Definitions of sets and their properties
variable (A : ℕ → Set (ℝ × ℝ)) -- Sequence of sets of points in the plane
variable (A0 : Set (ℝ × ℝ)) -- Initial set of points
variable (H : ℕ → Set (ℝ × ℝ)) -- Set of points in the plane

-- Intersection point definition
def is_intersection_point (P : ℝ × ℝ) (H : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B C D : ℝ × ℝ), A ≠ B ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧
  Line_through A B ≠ Line_through C D ∧
  Line_through A B ∩ Line_through C D = {P}

-- Sequence definition
def seq_union (A : Set (ℝ × ℝ)) : ℕ → Set (ℝ × ℝ)
| 0 => A
| j+1 => (seq_union A j) ∪ {P | is_intersection_point P (seq_union A j)}

-- Problem statement
theorem problem (A0 : Set (ℝ × ℝ))
  (finite_union : Finite (⋃ j, seq_union A0 j))
  (A1 : Set (ℝ × ℝ)) :
  (∀ i ≥ 1, seq_union A0 i = seq_union A0 1) := by
  sorry

end problem_l782_782521


namespace sequence_property_lim_ratio_l782_782905

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / Real.sqrt n) * (Finset.sum (Finset.range n) (λ k : ℕ, Real.sqrt (k+1)))

theorem sequence_property (n : ℕ) (hn1 : 1 ≤ n) :
  (a 1 = 1) ∧
  (a 2 = 1 + Real.sqrt 2 / 2) ∧
  ∀ n ≥ 1, (a n / (a (n + 1) - 1)) ^ 2 + ((a n - 1) / a (n - 1)) ^ 2 = 2 :=
by
  sorry

theorem lim_ratio : (Real.liminf (λ n : ℕ, a n / n)) = 2 / 3 ∧
                   (Real.limsup (λ n : ℕ, a n / n)) = 2 / 3 := 
by
  sorry

end sequence_property_lim_ratio_l782_782905


namespace domain_v_l782_782280

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt x + Real.cbrt x)

theorem domain_v : ∀ x : ℝ, (0 < x) ↔ (Real.sqrt x + Real.cbrt x ≠ 0) :=
by
  intro x
  apply iff.intro
  · intro h
    sorry -- Proof that if 0 < x, then Real.sqrt x + Real.cbrt x ≠ 0
  · intro h
    sorry -- Proof that if Real.sqrt x + Real.cbrt x ≠ 0, then 0 < x

end domain_v_l782_782280


namespace number_of_bowls_l782_782725

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l782_782725


namespace problem_solution_l782_782073

theorem problem_solution :
  (∑ n in Finset.range 1000 + 1, n * (1001 - n)) = 1000 * 500 * (2 / 3) :=
by
  sorry

end problem_solution_l782_782073


namespace min_value_of_function_l782_782008

/-- Prove that the minimum value of the function y = cos^2(x) - sin^2(x) + 2sin(x)cos(x) is -√2 -/
theorem min_value_of_function : 
    ∃ x ∈ ℝ, (cos x * cos x - sin x * sin x + 2 * sin x * cos x) = -√2 :=
sorry

end min_value_of_function_l782_782008


namespace volume_ratio_of_tetrahedrons_l782_782564

variables (s1 s2 : ℝ) (edge_ratio : ℝ := 2) -- defining side length variables and edge length ratio
variables (A₁ A₂ V₁ V₂ : ℝ) (h : ℝ) -- defining area, volume, and height variables

-- Define the conditions
def side_length_ratio : Prop := s1 / s2 = 1 / edge_ratio
def area_ratio : Prop := A₁ / A₂ = (s1 / s2) ^ 2
def height_ratio : Prop := h * (s1 / s2) = h / edge_ratio
def volume_of_tetrahedron (s : ℝ) (A : ℝ) (h : ℝ) : ℝ := (1 / 3) * A * h

-- Theorem to prove the volume ratio
theorem volume_ratio_of_tetrahedrons (h : ℝ) :
  side_length_ratio s1 s2 edge_ratio →
  area_ratio s1 s2 A₁ A₂ →
  height_ratio s1 s2 h →
  (volume_of_tetrahedron s1 A₁ h) / (volume_of_tetrahedron s2 A₂ (h / edge_ratio)) = 1 / (edge_ratio ^ 3)
:=
λ hs ha hh, sorry

end volume_ratio_of_tetrahedrons_l782_782564


namespace tangent_ellipse_hyperbola_l782_782685

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ∧ x^2 - n * (y - 1)^2 = 1) → n = 2 :=
by
  intro h
  sorry

end tangent_ellipse_hyperbola_l782_782685


namespace total_percentage_of_women_l782_782377

theorem total_percentage_of_women
    (initial_employees : ℕ)
    (men_fraction : ℚ)
    (new_women : ℕ)
    (initial_employees = 90)
    (men_fraction = 2/3)
    (new_women = 10) :
  let women_fraction := 1 - men_fraction in
  let initial_women := women_fraction * initial_employees in
  let total_women := initial_women + new_women in
  let total_employees := initial_employees + new_women in
  let percentage_of_women := (total_women / total_employees) * 100 in
  percentage_of_women = 40 :=
by
  sorry

end total_percentage_of_women_l782_782377


namespace perfect_square_partition_l782_782871

open Nat

-- Define the condition of a number being a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- Define the main theorem statement
theorem perfect_square_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n+1)) → (A ∩ B = ∅) →
  ∃ a b ∈ A, a ≠ b ∧ is_perfect_square (a + b)
:= by
  sorry

end perfect_square_partition_l782_782871


namespace term_formula_an_sum_bn_l782_782632

variable {n : ℕ}
variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Condition 1: a_n > 0
def pos_an (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

-- Condition 2: a_n^2 + 2a_n = 4S_n + 3
def relationship_an_Sn (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ^ 2 + 2 * a n = 4 * S n + 3

-- General term formula for {a_n}
theorem term_formula_an (a : ℕ → ℝ) (S : ℕ → ℝ) (ha_pos : pos_an a) (hrel : relationship_an_Sn a S) :
  ∀ n : ℕ, a n = 2 * n + 1 := sorry

-- Sum of first n terms of {b_n}
def bn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (a n * a (n+1))

def partial_sum_bn (a : ℕ → ℝ) (n : ℕ) : ℕ → ℝ
| 0       := bn a 0
| (k + 1) := partial_sum_bn a k + bn a (k + 1)

theorem sum_bn (a : ℕ → ℝ) (n : ℕ) (ha_pos : pos_an a) (hrel : relationship_an_Sn a S) (hterm : ∀ n, a n = 2 * n + 1) :
  partial_sum_bn a n = n / (3 * (2 * n + 3)) := sorry

end term_formula_an_sum_bn_l782_782632


namespace number_of_bowls_l782_782745

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l782_782745


namespace eval_exp_l782_782419

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l782_782419


namespace calculate_s_l782_782584

variable (r : ℝ) (s a : ℝ)
variable (h_pos : r > 0) (h_series_start : a > 0)
variable (h_s : s = 1 / (Real.sqrt 2))
variable (u₃ := s * a)
variable (u₄ := a * u₃)
variable (u₅ := u₃ * u₄)
variable (u₈ := (u₅ * (u₃ * (u₄ * (u₅ * (u₃ * u₄))))))

-- Conditions
variable (h₅ : u₅ = 1 / r)
variable (h₈ : u₈ = 1 / r ^ 4)

theorem calculate_s (h₁ : r = 2) : s = 1 / sqrt 2 :=
  by
  sorry

end calculate_s_l782_782584


namespace luke_payment_difference_l782_782645

noncomputable def plan1_payment (principal: ℝ) (rate: ℝ) (n: ℝ) (t: ℝ) : ℝ :=
  let accumulated = principal * (1 + rate / n) ^ (n * t)
  let half_payment = accumulated / 2
  let remaining = half_payment * (1 + rate / n) ^ (n * t)
  half_payment + remaining

noncomputable def plan2_payment (principal: ℝ) (rate: ℝ) (t: ℝ) (fee: ℝ) : ℝ :=
  let accumulated = principal * (1 + rate) ^ t
  accumulated + fee

theorem luke_payment_difference :
  let plan1 = plan1_payment 12000 0.08 4 7.5
  let plan2 = plan2_payment 12000 0.08 15 500
  abs (plan2 - plan1) = 10142.23 := sorry

end luke_payment_difference_l782_782645


namespace evaluate_exponent_l782_782449

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l782_782449


namespace monic_quadratic_with_complex_root_l782_782476

theorem monic_quadratic_with_complex_root :
  ∃ P : Polynomial ℝ, (P = Polynomial.X^2 - 4 * Polynomial.X + 13) ∧ P.monic ∧ P.eval (2 - 3 * Complex.I) = 0 :=
by
  sorry

end monic_quadratic_with_complex_root_l782_782476


namespace min_value_on_interval_l782_782246

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 1

theorem min_value_on_interval : ∃ (c : ℝ), (c ∈ set.Icc 0 3) ∧ (∀ x ∈ set.Icc 0 3, f c ≤ f x) ∧ f c = -3 := 
by
  use 2
  simp [f]
  split
  { exact set.left_mem_Icc.mpr zero_le_two }
  { split
    { intros x hx,
      have p1 : 0 ≤ (x - 2)^2 := sq_nonneg (x - 2),
      have p2 : (x - 2)^2 = x^2 - 4 * x + 4 := by ring,
      linarith [p1, p2] }
    { simp } }

end min_value_on_interval_l782_782246


namespace shaded_area_represents_correct_set_l782_782063

theorem shaded_area_represents_correct_set :
  ∀ (U A B : Set ℕ), 
    U = {0, 1, 2, 3, 4} → 
    A = {1, 2, 3} → 
    B = {2, 4} → 
    (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} :=
by
  intros U A B hU hA hB
  -- The rest of the proof would go here
  sorry

end shaded_area_represents_correct_set_l782_782063


namespace total_number_of_people_in_room_l782_782101

theorem total_number_of_people_in_room (P M : ℕ) (hP : 0.02 * P = 1) (hM : 0.05 * M = 1) :
  P + M - 1 = 69 :=
by
  -- Convert the given conditions into expressions in real number domain
  have h_condition1 : P = 50,
  { calc
      P = 1 / 0.02 : by rw ← hP
      ... = 50 : by norm_num
  },
  have h_condition2 : M = 20,
  { calc
      M = 1 / 0.05 : by rw ← hM
      ... = 20 : by norm_num
  },
  -- Use the derived values to compute the final answer
  rw [h_condition1, h_condition2],
  norm_num,
  -- Result
  sorry

end total_number_of_people_in_room_l782_782101


namespace area_triangle_EDH_l782_782380

-- Given conditions
def ABCD_square (A B C D : Point) : Prop := 
  square A B C D ∧ side_length A B = 11

def CEFG_square (C E F G : Point) : Prop := 
  square C E F G ∧ side_length C E = 9 ∧ G ∈ line_segment C D

def H_on_AB (H A B : Point) : Prop := 
  H ∈ line_segment A B

def right_angle_EDH (E D H : Point) : Prop := 
  right_angle E D H

-- Theorem to prove the area of triangle EDH is 101 square centimeters
theorem area_triangle_EDH (A B C D E F G H : Point) 
  (h1 : ABCD_square A B C D)
  (h2 : CEFG_square C E F G)
  (h3 : H_on_AB H A B)
  (h4 : right_angle_EDH E D H) :
  area_triangle E D H = 101 :=
by
  sorry

end area_triangle_EDH_l782_782380


namespace evaluate_power_l782_782462

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l782_782462


namespace ab_bc_ca_max_le_l782_782183

theorem ab_bc_ca_max_le (a b c : ℝ) :
  ab + bc + ca + max (abs (a - b)) (max (abs (b - c)) (abs (c - a))) ≤
  1 + (1 / 3) * (a + b + c)^2 :=
sorry

end ab_bc_ca_max_le_l782_782183


namespace monic_quadratic_with_complex_root_l782_782477

theorem monic_quadratic_with_complex_root :
  ∃ P : Polynomial ℝ, (P = Polynomial.X^2 - 4 * Polynomial.X + 13) ∧ P.monic ∧ P.eval (2 - 3 * Complex.I) = 0 :=
by
  sorry

end monic_quadratic_with_complex_root_l782_782477


namespace choose_three_cooks_from_ten_l782_782126

theorem choose_three_cooks_from_ten : 
  (nat.choose 10 3) = 120 := 
by
  sorry

end choose_three_cooks_from_ten_l782_782126


namespace loss_percentage_l782_782773

-- Definitions of cost price (C) and selling price (S)
def cost_price : ℤ := sorry
def selling_price : ℤ := sorry

-- Given condition: Cost price of 40 articles equals selling price of 25 articles
axiom condition : 40 * cost_price = 25 * selling_price

-- Statement to prove: The merchant made a loss of 20%
theorem loss_percentage (C S : ℤ) (h : 40 * C = 25 * S) : 
  ((S - C) * 100) / C = -20 := 
sorry

end loss_percentage_l782_782773


namespace transform_z1_z2_l782_782910

noncomputable def z1 := 2 + 3 * Complex.I
noncomputable def z2 := 3 + 2 * Complex.I
noncomputable def w := Complex.exp (-Complex.I * Real.pi / 2) * ·

theorem transform_z1_z2:
  w z1 = 3 - 2 * Complex.I ∧
  w z2 = 2 - 3 * Complex.I :=
by
  sorry

end transform_z1_z2_l782_782910


namespace find_ABCD_l782_782607

theorem find_ABCD :
  ∃ (A B C D : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
    A = 9 ∧ B = 8 ∧ C = 0 ∧ D = 1 :=
by 
  existsi 9, 8, 0, 1
  repeat { split }; 
  try {norm_num}
  sorry

end find_ABCD_l782_782607


namespace brooke_math_time_l782_782385

variable (x : ℝ)
variable (n_math : ℕ) (n_social : ℕ) (n_science : ℕ) (t_social_seconds : ℕ) (t_science_minutes : ℝ) (total_time : ℝ)

-- Definition of problem conditions
def math_problems := n_math = 15
def social_problems := n_social = 6
def science_problems := n_science = 10
def social_problem_time := t_social_seconds = 30
def science_problem_time := t_science_minutes = 1.5
def total_homework_time := total_time = 48

-- The statement we aim to prove
theorem brooke_math_time : 
  math_problems ∧ social_problems ∧ science_problems ∧ social_problem_time ∧ science_problem_time ∧ total_homework_time 
  → 15 * x + n_social * (t_social_seconds / 60) + n_science * t_science_minutes = 48 → x = 2 :=
by
  sorry

end brooke_math_time_l782_782385


namespace eval_expr_l782_782426

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l782_782426


namespace integer_part_of_x_l782_782082

noncomputable def x : ℝ := 1 + ∑ k in (finset.range 10^6).filter (λ k, k ≥ 2), (1 / real.sqrt k)

theorem integer_part_of_x : real.floor x = 1998 := 
by
  sorry

end integer_part_of_x_l782_782082


namespace max_possible_points_l782_782261

-- Define the predicate for a triangle having angles in natural degrees
def natural_number_degrees (P1 P2 P3 : ℝ × ℝ) : Prop :=
  ∃ a b c : ℕ, a + b + c = 180 ∧
  ∃ α β γ : ℝ, 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = 180 ∧
  α = a ∧ β = b ∧ γ = c ∧
  α = degrees (angle P1 P2 P3) ∧ β = degrees (angle P2 P3 P1) ∧ γ = degrees (angle P3 P1 P2)

-- Define the main theorem
theorem max_possible_points (N : ℕ) (points : fin N → ℝ × ℝ)
  (h1 : ∀ (i j k : fin N), i ≠ j ∧ j ≠ k ∧ k ≠ i → ¬collinear (points i) (points j) (points k))
  (h2 : ∀ (i j k : fin N), i ≠ j ∧ j ≠ k ∧ k ≠ i → natural_number_degrees (points i) (points j) (points k)) :
  N ≤ 180 :=
sorry

end max_possible_points_l782_782261


namespace max_sum_of_products_l782_782517

theorem max_sum_of_products (n : ℕ) (k : ℝ) (h_pos : 0 < k) (x : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) (h_sum : (Finset.univ : Finset (Fin n)).sum x = k) :
  ∃ x, x₁ x₂ + ∑ i in Finset.range (n-1), x i * x (i+1) ≤ k^2 / 4 :=
begin
  sorry
end

end max_sum_of_products_l782_782517


namespace log_base3_x_condition_l782_782075

theorem log_base3_x_condition (x : ℝ) (hx : 2^(Real.log x / Real.log 3) = 1 / 8) : x = 1 / 27 :=
by
  sorry

end log_base3_x_condition_l782_782075


namespace pipe_A_fill_time_l782_782271

theorem pipe_A_fill_time (x : ℝ) (h₁ : x > 0) (h₂ : 1 / x + 1 / 15 = 1 / 6) : x = 10 :=
by
  sorry

end pipe_A_fill_time_l782_782271


namespace no_conclusion_term_area_even_l782_782358

structure Point :=
  (x : ℤ)
  (y : ℤ)

structure Triangle :=
  (p1 : Point)
  (p2 : Point)
  (p3 : Point)

def area (t : Triangle) : ℤ :=
  (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y)).natAbs / 2

theorem no_conclusion_term_area_even (p1 p2 p3 : Point) (h1 : ∃ k : Int, k * 2 * 2 = ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)))) :
  (¬(is_right_angle (Triangle.mk p1 p2 p3)) ∧ ¬(is_equilateral (Triangle.mk p1 p2 p3)) ∧ ¬(is_isosceles (Triangle.mk p1 p2 p3)) ∧ ¬(is_scalene (Triangle.mk p1 p2 p3))) :=
by
  sorry

-- Definitions for is_right_angle, is_equilateral, is_isosceles, and is_scalene need to be defined here or assumed from a pre-loaded library.

end no_conclusion_term_area_even_l782_782358


namespace find_a_l782_782532

theorem find_a (a : ℝ) :
  let A := {0, 2, 3}
  let B := {2, a^2 + 1}
  B ⊆ A →
  a = sqrt 2 ∨ a = -sqrt 2 :=
by
  sorry

end find_a_l782_782532


namespace days_of_earning_l782_782698

theorem days_of_earning (T D d : ℕ) (hT : T = 165) (hD : D = 33) (h : d = T / D) :
  d = 5 :=
by sorry

end days_of_earning_l782_782698


namespace problem1_problem2_l782_782190

noncomputable def a (x : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.sin x, real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (real.cos x, real.sin x)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

-- First part: proving x = π/6 given |a| = |b| and x ∈ [0, π/2]
theorem problem1 (x : ℝ) (h1 : magnitude (a x) = magnitude (b x)) (h2 : 0 ≤ x ∧ x ≤ real.pi / 2) :
  x = real.pi / 6 :=
sorry

-- Second part: finding intervals where f(x) is monotonically increasing
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem problem2 (k : ℤ) : 
  ∀ x : ℝ, f'(x) > 0 → (x ∈ set.Icc (k * real.pi - real.pi / 6) (k * real.pi + real.pi / 3)) :=
sorry

end problem1_problem2_l782_782190


namespace john_change_sum_l782_782623

theorem john_change_sum :
  (∑ c in {c | ∃ n m : ℕ, 5 * n + 4 = c ∧ 10 * m + 6 = c ∧ c < 100}, c) = 540 :=
by
  -- Proof steps would go here
  sorry

end john_change_sum_l782_782623


namespace jelly_beans_initial_counts_l782_782267

variable (A B C : ℕ)

/-- Conditions of the initial jelly beans for A, B, and C. -/
def initial_conditions (A_init B_init C_init : ℕ) : Prop :=
  -- First operation: A gives B and C the number of jelly beans equal to what B and C originally had
  let A1 := A_init - B_init - C_init
  let B1 := 2 * B_init
  let C1 := 2 * C_init in
  -- Second operation: B gives A and C the number of jelly beans equal to what A and C currently have
  let A2 := 2 * A1
  let B2 := B1 - A1 - C1
  let C2 := 2 * C1 in
  -- Third operation: C gives A and B the number of jelly beans equal to what A and B currently have
  let A3 := A2 + C2
  let B3 := B2 + C2
  let C3 := C2 - A2 - B2 in
  -- Final condition
  A3 = 64 ∧ B3 = 64 ∧ C3 = 64

/-- Theorem stating that initially A, B, and C had 104, 56, and 32 jelly beans respectively. -/
theorem jelly_beans_initial_counts :
  ∃ (A B C : ℕ), initial_conditions A B C ∧ A = 104 ∧ B = 56 ∧ C = 32 :=
by sorry

end jelly_beans_initial_counts_l782_782267


namespace probability_sum_of_dice_eq_3_l782_782768

theorem probability_sum_of_dice_eq_3 : 
  ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → 
  (a + b + c = 3) → 
  prob_event (λ (a b c : ℕ), a + b + c = 3) = 1 / 216 :=
by
  sorry

end probability_sum_of_dice_eq_3_l782_782768


namespace evaluate_three_cubed_squared_l782_782438

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l782_782438


namespace car_speed_l782_782980

theorem car_speed (revolutions_per_minute circumference : ℕ) (rpm_eq : revolutions_per_minute = 400) (circ_eq : circumference = 3) : 
  let distance_per_min := revolutions_per_minute * circumference in
  let speed_km_per_hour := (distance_per_min * 60) / 1000 in
  speed_km_per_hour = 72 := 
by
  sorry

end car_speed_l782_782980


namespace natural_numbers_partition_l782_782857

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l782_782857


namespace equalize_shares_l782_782648

theorem equalize_shares :
  ∀ (m j : ℤ),
  let total_cost := 150 + 90 + 210 in
  let fair_share := total_cost / 3 in
  let m_contribution := 150 - fair_share in
  let j_contribution := 90 - fair_share in
  let c_contribution := 210 - fair_share in
  (m = m_contribution) → (j = j_contribution) → m - j = -60 :=
by
  intros m j total_cost fair_share m_contribution j_contribution c_contribution h_m h_j
  sorry

end equalize_shares_l782_782648


namespace value_of_a_plus_3b_l782_782635

noncomputable def f (a b : ℝ) : ℝ → ℝ
| x => if x < 0 then a * x + 1 else (b * x + 2) / (x + 1)

theorem value_of_a_plus_3b (a b : ℝ) (hf_periodic : ∀ x, f a b x = f a b (x+2))
  (h1 : ∃ x : ℝ, -1 ≤ x ∧ x < 0 → f a b x = a * x + 1)
  (h2 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a b x = (b * x + 2) / (x + 1))
  (h3 : a ∈ ℝ) (h4 : b ∈ ℝ)
  (h5 : f a b (1/2) = f a b (3/2)) :
  a + 3 * b = -10 := 
sorry

end value_of_a_plus_3b_l782_782635


namespace sunflower_height_A_l782_782194

-- Define the height of sunflowers from Packet B
def height_B : ℝ := 160

-- Define that Packet A sunflowers are 20% taller than Packet B sunflowers
def height_A : ℝ := 1.2 * height_B

-- State the theorem to show that height_A equals 192 inches
theorem sunflower_height_A : height_A = 192 := by
  sorry

end sunflower_height_A_l782_782194


namespace problem_solution_l782_782260

-- Define the arithmetic sequence a_n
def a_n (n : ℕ) : ℕ := n

-- Define the geometric sequence b_n
def b_n (n : ℕ) : ℕ := 2^(n - 1)

-- Define the sum of the first n terms of the product sequence a_n * b_n
noncomputable def T_n (n : ℕ) : ℕ :=
  finset.range n |>.sum (λ k, (k + 1) * 2^k)

-- Define the sum of the geometric series
def sum_geometric_series (n : ℕ) : ℕ :=
  (finset.range n |>.sum (λ k, 2^(k+1))) - n * 2^n

-- Main theorem
theorem problem_solution (n : ℕ) : T_n n = n - 2 - (n - 1) * 2^n :=
sorry

end problem_solution_l782_782260


namespace tangent_line_at_1_l782_782687

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x

theorem tangent_line_at_1:
  let x := (1 : ℝ)
  let y := 0
  f(x) = y →
  let slope := deriv f 1 in
  slope = Real.exp 1 →
  ∀ (t : ℝ), (slope * (t - 1) + y = Real.exp 1 * (t - 1)) →
  ∀ (t : ℝ), f t = Real.exp t * Real.log t →
  (∀ t : ℝ, t * Real.exp 1 - Real.exp 1 = (Real.exp 1) * (t - 1)) :=
by
  intro x y hx slope hs
  intro t ht ht'
  sorry

end tangent_line_at_1_l782_782687


namespace exists_distinct_group_and_country_selection_l782_782278

theorem exists_distinct_group_and_country_selection 
  (n m : ℕ) 
  (h_nm1 : n > m) 
  (h_m1 : m > 1) 
  (groups : Fin n → Fin m → Fin n → Prop) 
  (group_conditions : ∀ i j : Fin n, ∀ k : Fin m, ∀ l : Fin m, (i ≠ j) → (groups i k j = false)) 
  : 
  ∃ (selected : Fin n → Fin (m * n)), 
    (∀ i j: Fin n, i ≠ j → selected i ≠ selected j) ∧ 
    (∀ i j: Fin n, selected i / m ≠ selected j / m) := sorry

end exists_distinct_group_and_country_selection_l782_782278


namespace matrix_product_l782_782389

theorem matrix_product :
  let M : Fin 51 → Matrix (Fin 2) (Fin 2) ℝ := λ i =>
    match i with
    | ⟨0, _⟩ => ![![1, 2], ![0, 1]]
    | ⟨n+1, _⟩ => ![![1, 2 * (n+1) + 2], ![0, 1]]
  ∏ i in Finset.range 51, M i = ![![1, 2550], ![0, 1]] :=
by
  sorry

end matrix_product_l782_782389


namespace range_of_m_l782_782540

noncomputable def f (x : ℝ) : ℝ := (Real.log x - 1) / x

theorem range_of_m {a b : ℝ} (ha : Real.norm a = 1) (hb : Real.norm b = 1) (angle_ab : Real.angle a b = π / 3)
  (cond : ∀ (x1 x2 : ℝ), x1 > e^2 → x2 > e^2 → x1 < x2 →
    (x1 * Real.log x2 - x2 * Real.log x1) / (x1 - x2) > Real.norm (a - b)) :
  (∀ m : ℝ, m ≥ e^2 → (∀ x1 x2 : ℝ, x1 ∈ Set.Ioi m → x2 ∈ Set.Ioi m → x1 < x2 →
    f x2 < f x1)) :=
begin
  sorry
end

end range_of_m_l782_782540


namespace range_of_m_l782_782054

noncomputable def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m : {m : ℝ | ∀ x : ℝ, quadratic_function m x < 0} = set.Icc (-4 : ℝ) 0 :=
by
  sorry

end range_of_m_l782_782054


namespace samantha_birth_year_l782_782689

-- Define conditions
def first_amc8_year : ℕ := 1985
def amc8_interval : ℕ := 2
def sam_age_at_fourth_amc8 : ℕ := 12
def fourth_amc8_year : ℕ := first_amc8_year + 3 * amc8_interval

-- Define the main theorem to prove
theorem samantha_birth_year : ∃ y : ℕ, y = (fourth_amc8_year : ℕ) - sam_age_at_fourth_amc8 := 
begin
  -- Calculation and proof steps would go here
  sorry
end

end samantha_birth_year_l782_782689


namespace evaluate_power_l782_782460

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l782_782460


namespace max_gcd_of_sequence_l782_782255

-- Step definitions and conditions
def a (n : ℕ) := n ^ 2 + 1000

def d (n : ℕ) := Nat.gcd (a n) (a (n + 1))

-- Step to assert maximum value of d_n is 4001
theorem max_gcd_of_sequence : ∃ n ∈ (ℕ+), d n = 4001 := sorry

end max_gcd_of_sequence_l782_782255


namespace percentage_of_women_in_company_l782_782375

theorem percentage_of_women_in_company (initial_workers : ℕ) (men_ratio women_ratio : ℚ) (new_women : ℕ) 
  (h_total : initial_workers = 90) 
  (h_men_ratio : men_ratio = 2/3) 
  (h_women_ratio : women_ratio = 1/3)
  (h_new_women : new_women = 10) :
  let men := (men_ratio * initial_workers).natAbs in
  let women := (women_ratio * initial_workers).natAbs in
  let total_women := women + new_women in
  let total_workers := initial_workers + new_women in
  (total_women * 100 / total_workers) = 40 := 
by
  sorry

end percentage_of_women_in_company_l782_782375


namespace final_number_after_operations_l782_782203

theorem final_number_after_operations :
  let numbers := {1, 2, ..., 20}
  ∀ a b, a ∈ numbers → b ∈ numbers →
  ∃ c, c = a*b + a + b :=
  let final_number := 21! - 1
  final_number = c :=
begin
  sorry
end

end final_number_after_operations_l782_782203


namespace tony_normal_temperature_l782_782268

theorem tony_normal_temperature (fever_threshold current_temp rise_due_to_sickness normal_temp : ℝ)
  (h1 : fever_threshold = 100)
  (h2 : current_temp = fever_threshold + 5)
  (h3 : rise_due_to_sickness = 10)
  (h4 : normal_temp = current_temp - rise_due_to_sickness) :
  normal_temp = 95 :=
by 
  rw [h1, h2, h3, h4]
  simp

end tony_normal_temperature_l782_782268


namespace part1_part2_l782_782041

-- Definitions
def A (x : ℝ) : Prop := (x + 2) / (x - 3 / 2) < 0
def B (x : ℝ) (m : ℝ) : Prop := x^2 - (m + 1) * x + m ≤ 0

-- Part (1): when m = 2, find A ∪ B
theorem part1 :
  (∀ x, A x ∨ B x 2) ↔ ∀ x, -2 < x ∧ x ≤ 2 := sorry

-- Part (2): find the range of real number m
theorem part2 :
  (∀ x, A x → B x m) ↔ (-2 < m ∧ m < 3 / 2) := sorry

end part1_part2_l782_782041


namespace evaluate_exponent_l782_782448

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l782_782448


namespace problem_statement_l782_782505

variable (a b m : ℝ)

theorem problem_statement (h : a < b) : ∃ m, m ≤ 0 ∧ ¬(ma > mb) :=
by
  sorry

end problem_statement_l782_782505


namespace garden_length_to_perimeter_ratio_l782_782794

theorem garden_length_to_perimeter_ratio :
  ∀ (length width : ℕ), length = 24 → width = 18 → (length / (2 * (length + width)) = 2 / 7) :=
by
  intros length width h_length h_width
  rw [h_length, h_width]
  -- Reasoning steps will go here
  sorry

end garden_length_to_perimeter_ratio_l782_782794


namespace log_mantissa_and_inequalities_l782_782091

theorem log_mantissa_and_inequalities (
  {a b : ℝ}
  (H1 : ∃ (k : ℤ), log a b = k)
  (H2 : log a (1 / b) > log a (sqrt b))
  (H3 : log a (sqrt b) > log b (a^2))
) : (ab - 1 = 0 ∧ ¬(1 / b > sqrt b > a^2) ∧ 
   ¬(log a b + log b a = 0) ∧ ¬(0 < a ∧ a < b ∧ b < 1)) :=
sorry

end log_mantissa_and_inequalities_l782_782091


namespace area_of_good_points_region_l782_782587

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

structure Point where
  a : ℝ
  b : ℝ
  condition : ¬(hyperbola a b)

def Omega_P (P : Point) (l : ℝ → ℝ × ℝ) : Prop :=
  ∃ (t1 t2 : ℝ), let (x1, y1) := l t1; let (x2, y2) := l t2 in
  hyperbola x1 y1 ∧ hyperbola x2 y2

def f_P (P : Point) (l : ℝ → ℝ × ℝ) : ℝ :=
  let ⟨t1, t2, _⟩ := Omega_P.exists_two_points { a := P.a, b := P.b, condition := P.condition } l;
  let (x1, y1) := l t1; let (x2, y2) := l t2 in
  ((P.a - x1)^2 + (P.b - y1)^2) ^ 0.5 * ((P.a - x2)^2 + (P.b - y2)^2) ^ 0.5

def is_good_point (P : Point) : Prop :=
  ∃ (l0 : ℝ → ℝ × ℝ) (hl0 : Omega_P P l0), 
  let ⟨t1, t2, ht0⟩ := hl0 in
  (l0 t1).1 * (l0 t2).1 < 0 ∧ ∀ (l : ℝ → ℝ × ℝ), Omega_P P l → l ≠ l0 → f_P P l > f_P P l0 

def good_points_region : Set Point :=
  {P : Point | is_good_point P}

theorem area_of_good_points_region : ∃ (area : ℝ), area = 4 :=
sorry

end area_of_good_points_region_l782_782587


namespace perpendicular_vectors_l782_782950

-- Define the vectors a and b.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (-2, x)

-- Define the dot product function.
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition that a is perpendicular to b.
def perp_condition (x : ℝ) : Prop :=
  dot_product vector_a (vector_b x) = 0

-- Main theorem stating that if a is perpendicular to b, then x = -1.
theorem perpendicular_vectors (x : ℝ) (h : perp_condition x) : x = -1 :=
by sorry

end perpendicular_vectors_l782_782950


namespace product_less_than_3_l782_782220

open BigOperators

def seq (n : ℕ) : ℝ := 1 + 1 / (n^3 : ℝ)

def product_up_to (n : ℕ) : ℝ := ∏ i in Finset.range (n + 1), seq (i + 1)

theorem product_less_than_3 : product_up_to 2013 < 3 := sorry

end product_less_than_3_l782_782220


namespace eval_expr_l782_782424

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l782_782424


namespace find_value_of_2a_minus_3b_l782_782678

theorem find_value_of_2a_minus_3b
  (a b : ℝ)
  (f g h : ℝ → ℝ)
  (hf : ∀ x, f(x) = a * x + b)
  (hg : ∀ x, g(x) = -4 * x + 6)
  (hh : ∀ x, h(x) = f(g(x)))
  (hinv : ∀ y, h(y) - 9 = y) :
  2 * a - 3 * b = 22 :=
sorry

end find_value_of_2a_minus_3b_l782_782678


namespace evaluate_three_cubed_squared_l782_782443

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l782_782443


namespace find_natural_number_n_l782_782469

def is_terminating_decimal (x : ℚ) : Prop :=
  ∃ (a b : ℕ), x = (a / b) ∧ (∃ (m n : ℕ), b = 2 ^ m * 5 ^ n)

theorem find_natural_number_n (n : ℕ) (h₁ : is_terminating_decimal (1 / n)) (h₂ : is_terminating_decimal (1 / (n + 1))) : n = 4 :=
by sorry

end find_natural_number_n_l782_782469


namespace alyssa_games_this_year_l782_782368

theorem alyssa_games_this_year : 
    ∀ (X: ℕ), 
    (13 + X + 15 = 39) → 
    X = 11 := 
by
  intros X h
  have h₁ : 13 + 15 = 28 := by norm_num
  have h₂ : X + 28 = 39 := by linarith
  have h₃ : X = 11 := by linarith
  exact h₃

end alyssa_games_this_year_l782_782368


namespace value_expression_eq_zero_l782_782166

theorem value_expression_eq_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
    a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 :=
by
  sorry

end value_expression_eq_zero_l782_782166


namespace eval_expr_l782_782430

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l782_782430


namespace exists_constants_l782_782040

noncomputable def f (a b c n : ℕ) : ℕ :=
  (sorry : ℕ) -- Placeholder for the actual function computation

theorem exists_constants (a b c : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_coprime: ∀ {a b c}, nat.coprime a b ∧ nat.coprime b c ∧ nat.coprime c a) :
  ∃ (α β γ : ℝ), ∀ n : ℕ, |f a b c n - (α * n^2 + β * n + γ)| < (1 / 12) * (a + b + c) :=
by
  sorry

end exists_constants_l782_782040


namespace lemonade_glasses_l782_782618

theorem lemonade_glasses (total_lemons : ℝ) (lemons_per_glass : ℝ) (glasses : ℝ) :
  total_lemons = 18.0 → lemons_per_glass = 2.0 → glasses = total_lemons / lemons_per_glass → glasses = 9 :=
by
  intro h_total_lemons h_lemons_per_glass h_glasses
  sorry

end lemonade_glasses_l782_782618


namespace mixed_doubles_pairing_l782_782264

def num_ways_to_pair (men women : ℕ) (select_men select_women : ℕ) : ℕ :=
  (Nat.choose men select_men) * (Nat.choose women select_women) * 2

theorem mixed_doubles_pairing : num_ways_to_pair 5 4 2 2 = 120 := by
  sorry

end mixed_doubles_pairing_l782_782264


namespace choose_3_out_of_10_l782_782121

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 :=
by
  sorry

end choose_3_out_of_10_l782_782121


namespace max_x2_plus_4y_plus_3_l782_782902

theorem max_x2_plus_4y_plus_3 
  (x y : ℝ) 
  (h : x^2 + y^2 = 1) : 
  x^2 + 4*y + 3 ≤ 7 := sorry

end max_x2_plus_4y_plus_3_l782_782902


namespace shingles_per_sq_ft_is_8_l782_782149

-- Declare the variables and constants
def num_roofs : ℕ := 3
def length_side : ℕ := 20
def width_side : ℕ := 40
def total_shingles : ℕ := 38400

-- Compute the area of one side of the roof
def area_one_side (length : ℕ) (width : ℕ) : ℕ :=
  length * width

-- Compute the total area for one roof (with two sides)
def total_area_one_roof (length : ℕ) (width : ℕ) : ℕ :=
  2 * area_one_side length width

-- Compute the total area for three roofs
def total_area_three_roofs (num_roofs : ℕ) (length : ℕ) (width : ℕ) : ℕ :=
  num_roofs * total_area_one_roof length, width

-- Given the total number of shingles, compute the number of shingles per square foot
def shingles_per_square_foot (total_shingles : ℕ) (total_area : ℕ) : ℕ :=
  total_shingles / total_area

-- The theorem we need to prove
theorem shingles_per_sq_ft_is_8 :
  shingles_per_square_foot total_shingles (total_area_three_roofs num_roofs length_side width_side) = 8 :=
by
  sorry

end shingles_per_sq_ft_is_8_l782_782149


namespace range_of_alpha_l782_782667

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + (2 / 3)

noncomputable def slope_of_tangent_line (x : ℝ) : ℝ := 3 * x^2 - 1

theorem range_of_alpha :
  ∃α : set ℝ, 
    {α ∈ set.univ | (∃ x, α = slope_of_tangent_line x)} = 
    {α | (0 <= α ∧ α < π/2) ∨ (3*π/4 <= α ∧ α < π)} :=
sorry

end range_of_alpha_l782_782667


namespace partition_with_sum_square_l782_782851

def sum_is_square (a b : ℕ) : Prop := ∃ k : ℕ, a + b = k * k

theorem partition_with_sum_square (n : ℕ) (h : n ≥ 15) :
  ∀ (s₁ s₂ : finset ℕ), (∅ ⊂ s₁ ∪ s₂ ∧ s₁ ∩ s₂ = ∅ ∧ (∀ x ∈ s₁ ∪ s₂, x ∈ finset.range (n + 1))) →
  (∃ a b : ℕ, a ≠ b ∧ (a ∈ s₁ ∧ b ∈ s₁ ∨ a ∈ s₂ ∧ b ∈ s₂) ∧ sum_is_square a b) :=
by sorry

end partition_with_sum_square_l782_782851


namespace number_of_bowls_l782_782740

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l782_782740


namespace function_inequality_l782_782793

theorem function_inequality
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f(x) * f(y) ≤ x^2 * f(y/2) + y^2 * f(x/2))
  (M : ℝ)
  (h2 : M > 0)
  (h3 : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → |f(x)| ≤ M) :
  ∀ x : ℝ, x ≥ 0 → f(x) ≤ x^2 :=
by
  sorry

end function_inequality_l782_782793


namespace sin_BC_identity_maximum_area_l782_782304

-- Part (1)
theorem sin_BC_identity (A B C : ℝ) (h₁ : sin A = (4 * sqrt 5) / 9) 
  (h₃ : A + B + C = π) : 
  sin (2 * (B + C)) + (sin (B + C) / 2)^2 = (45 - 8 * sqrt 5) / 81 :=
sorry

-- Part (2)
theorem maximum_area (A B C a b c : ℝ) (h₁ : sin A = (4 * sqrt 5) / 9)
  (h_ac : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) -- Acute triangle condition
  (h₂ : a = 4) 
  (h₃ : a^2 = b^2 + c^2 - 2 * b * c * cos A) 
  (h₄ : a = 4) :
  let bc := 9,
      S := (1 / 2) * b * c * sin A
  in S = 2 * sqrt 5 :=
sorry

end sin_BC_identity_maximum_area_l782_782304


namespace evaluate_power_l782_782461

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l782_782461


namespace repeating_decimal_eq_one_div_three_one_minus_repeating_decimal_l782_782386

/-- Define the repeating decimal 0.333... as a limit of a sequence. -/
def repeating_decimal : ℝ := (∑'n:ℕ, (3:ℝ) / 10^(n + 1))

/-- Prove that 0.\overline{3} is equal to 1/3 -/
theorem repeating_decimal_eq_one_div_three : repeating_decimal = 1 / 3 := sorry

/-- Prove that 1 - 0.\overline{3} = 2/3 -/
theorem one_minus_repeating_decimal : 1 - repeating_decimal = 2 / 3 := by
  rw [repeating_decimal_eq_one_div_three]
  norm_num

end repeating_decimal_eq_one_div_three_one_minus_repeating_decimal_l782_782386


namespace ce_sq_plus_de_sq_eq_196_l782_782992

theorem ce_sq_plus_de_sq_eq_196 
  (radius : ℝ) (A B C D E O : Point) 
  (h_circle : Circle O radius)
  (h_AB_diameter : diameter O A B)
  (h_CD_chord : chord O C D)
  (h_E_intersection : intersect E (segment A B) (segment C D))
  (h_radius : radius = 7)
  (h_BE : dist B E = 3)
  (h_angle_AEC : ∠ A E C = 30) :
  (dist C E) ^ 2 + (dist D E) ^ 2 = 196 :=
sorry

end ce_sq_plus_de_sq_eq_196_l782_782992


namespace sine_beta_sum_angles_l782_782903

theorem sine_beta (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
                  (h3 : 0 < β) (h4 : β < π / 2)
                  (h5 : cos α = 4 / 5)
                  (h6 : cos (α + β) = 3 / 5) :
  sin β = 7 / 25 :=
sorry

theorem sum_angles (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
                   (h3 : 0 < β) (h4 : β < π / 2)
                   (h5 : cos α = 4 / 5)
                   (h6 : cos (α + β) = 3 / 5) :
  2 * α + β = π / 2 :=
sorry

end sine_beta_sum_angles_l782_782903


namespace solve_for_y_l782_782225

theorem solve_for_y (y : ℝ) (h : (5 - 1 / y)^(1/3) = -3) : y = 1 / 32 :=
by
  sorry

end solve_for_y_l782_782225


namespace gcd_72_120_168_l782_782243

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  -- Each step would be proven individually here.
  sorry

end gcd_72_120_168_l782_782243


namespace complement_union_l782_782898

open Set

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement relative to U
def complement (A B : Set ℕ) : Set ℕ := { x ∈ B | x ∉ A }

-- The theorem we need to prove
theorem complement_union :
  complement (M ∪ N) U = {4} :=
by
  sorry

end complement_union_l782_782898


namespace min_jumps_inequality_l782_782797

theorem min_jumps_inequality (k : ℕ) (i : ℕ) (hk : 2 ≤ k) :
  min_jumps (2^i * k) > min_jumps (2^i) :=
by
  -- proof goes here
  sorry

end min_jumps_inequality_l782_782797


namespace second_fisherman_more_fish_l782_782580

-- Defining the conditions
def total_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def days_rate1 : ℕ := 30
def days_rate2 : ℕ := 60
def days_rate3 : ℕ := total_days - (days_rate1 + days_rate2)

-- Calculating the total number of fish caught by both fishermen
def total_fish_first_fisherman : ℕ := first_fisherman_rate * total_days
def total_fish_second_fisherman : ℕ := (second_fisherman_rate1 * days_rate1) + 
                                        (second_fisherman_rate2 * days_rate2) + 
                                        (second_fisherman_rate3 * days_rate3)

-- Theorem stating the difference in the number of fish caught
theorem second_fisherman_more_fish : (total_fish_second_fisherman - total_fish_first_fisherman) = 3 := 
by
  sorry

end second_fisherman_more_fish_l782_782580


namespace arcsin_sqrt2_div2_l782_782825

theorem arcsin_sqrt2_div2 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end arcsin_sqrt2_div2_l782_782825


namespace seq_inequality_l782_782625

noncomputable def a_seq : ℕ → ℝ
| 0       := 2
| (n + 1) := (a_seq n) ^ 2 - a_seq n + 1 

theorem seq_inequality :
  ∀ {n : ℕ}, n ≥ 1 → 
  ∃ (a : ℕ → ℝ) (h₁ : a 0 = 2)
    (h₂ : ∀ n, a (n+1) = (a n) ^ 2 - a n + 1),
  1 - 1/(2003 ^ 2003) < ∑ i in Finset.range 2003, (1 / a i) ∧ ∑ i in Finset.range 2003, (1 / a i) < 1 :=
begin
  sorry
end

end seq_inequality_l782_782625


namespace hare_tortoise_meeting_time_l782_782339

-- Define the given speeds as constants
def speed_tortoise : ℝ := 5 / 3
def speed_hare : ℝ := 5 / 2

-- Define the initial distance the hare is behind the tortoise
def initial_distance : ℝ := 20

-- Define the relative speed of the hare catching up to the tortoise
def relative_speed : ℝ := speed_hare - speed_tortoise

-- Define the time when the hare and the tortoise meet
def meeting_time : ℝ := initial_distance / relative_speed

-- Goal: Prove that the meeting time is 24 minutes
theorem hare_tortoise_meeting_time : meeting_time = 24 := by
  sorry

end hare_tortoise_meeting_time_l782_782339


namespace find_n_largest_binomial_coefficient_term_constant_term_in_expansion_l782_782466

noncomputable def sum_binomial_first_three_terms (n : ℕ) : ℕ :=
  nat.choose n 0 + nat.choose n 1 + nat.choose n 2

theorem find_n (n : ℕ) (hn1 : sum_binomial_first_three_terms n = 22) : n = 6 := by
  sorry

theorem largest_binomial_coefficient_term (n : ℕ) (hn : n = 6) : 
  (nat.choose n 3) * 2^6 * (2^3) * x^(3 / 2) = 1280 * x^(3 / 2) := by
  sorry

theorem constant_term_in_expansion (n : ℕ) (hn : n = 6) :
  (nat.choose n 4) * 2^6 = 960 := by
  sorry

end find_n_largest_binomial_coefficient_term_constant_term_in_expansion_l782_782466


namespace number_of_correct_propositions_l782_782021

variables {a b : Line} {M N : Plane}

def P1 : Prop := (a ∥ M ∧ b ∥ M) → (a ∥ b)
def P2 : Prop := (a ∥ M ∧ b ⟂ M) → (a ⟂ b)
def P3 : Prop := (a ∥ b ∧ b ∥ M) → (a ∥ M)
def P4 : Prop := (a ⟂ M ∧ a ∥ N) → (M ⟂ N)

theorem number_of_correct_propositions : 
  (if P1 then 1 else 0) + 
  (if P2 then 1 else 0) + 
  (if P3 then 1 else 0) + 
  (if P4 then 1 else 0) = 2 :=
begin
  sorry
end

end number_of_correct_propositions_l782_782021


namespace pyramid_surface_area_l782_782942

theorem pyramid_surface_area (a : ℝ) : 
  let h := sqrt(3) / 3 * a in 
  let slant_height := sqrt(h^2 + (a / 2)^2) in 
  2 * (a * slant_height) + a^2 = 2 * sqrt(3) * a^2 :=
by {
  sorry
}

end pyramid_surface_area_l782_782942


namespace no_real_solutions_iff_k_gt_4_l782_782023

theorem no_real_solutions_iff_k_gt_4 (k : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + k ≠ 0) ↔ k > 4 :=
sorry

end no_real_solutions_iff_k_gt_4_l782_782023


namespace area_of_figure_l782_782829

theorem area_of_figure : 
  ∀ (x y : ℝ), |3 * x + 4| + |4 * y - 3| ≤ 12 → area_of_rhombus = 24 := 
by
  sorry

end area_of_figure_l782_782829


namespace cyclic_implies_angle_eq_l782_782035

-- Definitions of points and their relations
variables {A B C M P Q D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- A, B, C form a triangle
axiom is_triangle_ABC : ¬ collinear {A, B, C}

-- M is midpoint of AB
def midpoint (A B : Type) [MetricSpace A] [MetricSpace B] : Type :=
  sorry  -- Explicit midpoint construction provided by the user

def M_is_mid_A_B (M : Type) [MetricSpace M] (A B : Type) [MetricSpace A] [MetricSpace B] : Prop :=
  midpoint A B M

-- P is inside triangle ABC
axiom P_in_triangle : ∃ (interior : Prop), interior

-- Q is the reflection of P across M
def reflection (P M Q : Type) [MetricSpace P] [MetricSpace M] [MetricSpace Q] : Prop :=
  sorry  -- Explicit reflection construction provided by the user

def Q_is_reflection_P_M (Q : Type) [MetricSpace Q] (P M : Type) [MetricSpace P] [MetricSpace M] : Prop :=
  reflection P M Q

-- D, E are intersection points of lines AP, BP with sides BC, AC respectively
axiom D_Intersection : ∃ (intersection : Prop), intersection
axiom E_Intersection : ∃ (intersection : Prop), intersection

-- Definitions and Axoms required for the proof
def cyclic_points (A B D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace D] [MetricSpace E] : Prop :=
  ∃ (cyclic : Prop), cyclic

def angle_equality (P Q C A : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace C] [MetricSpace A] : Prop :=
  ∃ (angle_equal : Prop), angle_equal

theorem cyclic_implies_angle_eq (A B C M P Q D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace M] [MetricSpace P] [MetricSpace Q] [MetricSpace D] [MetricSpace E] :
  (M_is_mid_A_B M A B) ∧ (P_in_triangle) ∧ (Q_is_reflection_P_M Q P M) ∧ (D_Intersection Ap Bc D) ∧ (E_Intersection Bp Ac E)
  → (cyclic_points A B D E ↔ angle_equality P Q C A) :=
sorry

end cyclic_implies_angle_eq_l782_782035


namespace dave_tickets_left_l782_782381

-- Define the initial conditions
def initial_tickets : ℕ := 25
def tickets_won : ℕ := 127
def tickets_spent_first : ℕ := 84
def bonus_tickets : ℕ := 45
def tickets_spent_second : ℕ := 56

-- The problem statement to prove
theorem dave_tickets_left : 
  let total_tickets := initial_tickets + tickets_won 
  let after_first_spent := total_tickets - tickets_spent_first
  let after_bonus := after_first_spent + bonus_tickets
  let final_tickets := after_bonus - tickets_spent_second
  in final_tickets = 57 :=
by {
  -- you can add the proof steps here
  -- sorry is used to mean that the proof is not complete
  sorry 
}

end dave_tickets_left_l782_782381


namespace y_coordinate_of_vertex_C_l782_782213

-- Given points and conditions
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 6)
def D : ℝ × ℝ := (6, 6)
def E : ℝ × ℝ := (6, 0)
def C (h : ℝ) : ℝ × ℝ := (3, h)

-- The area of the squares ABDE and BCD
def square_area (side : ℝ) : ℝ := side * side
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

-- Total area of the pentagon
def pentagon_area (h : ℝ) : ℝ := square_area 6 + triangle_area 6 (h - 6)

-- Prove that the y-coordinate of vertex C is 24 when the total area of the pentagon is 90
theorem y_coordinate_of_vertex_C : (∃ h : ℝ, pentagon_area h = 90) → (∃ h : ℝ, C h = (3, 24)) :=
by 
  sorry

end y_coordinate_of_vertex_C_l782_782213


namespace evaluate_three_cubed_squared_l782_782442

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l782_782442


namespace min_max_f_on_interval_f_below_g_der_power_diff_inequality_l782_782936

noncomputable def f (x : ℝ) := (1/2) * x^2 + log x
noncomputable def g (x : ℝ) := (2/3) * x^3
noncomputable def F (x : ℝ) := (1/2) * x^2 + log x - (2/3) * x^3
noncomputable def f' (x : ℝ) := x + 1 / x

theorem min_max_f_on_interval : 
  f 1 = 1 / 2 ∧ f real.exp(1) = 1 / 2 * real.exp(1)^2 + 1 := sorry

theorem f_below_g (x : ℝ) (h : 1 ≤ x) : f x < g x := sorry

theorem der_power_diff_inequality (x : ℝ) (n : ℕ) (hn : n > 0):
  (f' x)^n - f' (x^n) ≥ 2^n - 2 := sorry

end min_max_f_on_interval_f_below_g_der_power_diff_inequality_l782_782936


namespace square_prob_distance_l782_782159

noncomputable def probability_distance_ge_1 (S : set (ℝ × ℝ)) (side_len : ℝ) : ℝ :=
  let a := 28 in
  let b := 1 in
  let c := 32 in
  (a - b * Real.pi) / c

theorem square_prob_distance (side_len : ℝ) (hS : side_len = 2) :
  probability_distance_ge_1 {p | p.1 ≥ 0 ∧ p.1 ≤ side_len ∧ p.2 ≥ 0 ∧ p.2 ≤ side_len} side_len = (28 - Real.pi) / 32 :=
by {
  rw hS,
  unfold probability_distance_ge_1,
  sorry
}

end square_prob_distance_l782_782159


namespace apple_capacity_l782_782212

/-- Question: What is the largest possible number of apples that can be held by the 6 boxes and 4 extra trays?
 Conditions:
 - Paul has 6 boxes.
 - Each box contains 12 trays.
 - Paul has 4 extra trays.
 - Each tray can hold 8 apples.
 Answer: 608 apples
-/
theorem apple_capacity :
  let boxes := 6
  let trays_per_box := 12
  let extra_trays := 4
  let apples_per_tray := 8
  let total_trays := (boxes * trays_per_box) + extra_trays
  let total_apples_capacity := total_trays * apples_per_tray
  total_apples_capacity = 608 := 
by
  sorry

end apple_capacity_l782_782212


namespace backpacking_trip_cooks_l782_782129

theorem backpacking_trip_cooks :
  nat.choose 10 3 = 120 :=
sorry

end backpacking_trip_cooks_l782_782129


namespace no_winning_strategy_l782_782322

noncomputable def probability_of_winning_after_stop (r b : ℕ) : ℚ :=
  r / (r + b : ℚ)

theorem no_winning_strategy (r b : ℕ) (h : r = 26 ∧ b = 26) : 
  ¬ (∃ strategy : (ℕ → Bool) → ℚ, strategy (λ x, true) > 0.5) := 
by
  sorry

end no_winning_strategy_l782_782322


namespace count_valid_4_digit_integers_l782_782953

theorem count_valid_4_digit_integers :
  ∃ n : ℕ, n = 54 ∧ (∀ (d1 d2 d3 d4 : ℕ),
    d1 ∈ {1, 4, 5, 9} ∧ d2 ∈ {1, 4, 5, 9} ∧ (d1 * d2) % 2 = 0 ∧
    d3 ∈ {5, 6, 7} ∧ d4 ∈ {5, 6, 7} ∧ d3 ≠ d4 →
    (10^3 * d1 + 10^2 * d2 + 10 * d3 + d4) ∈ {x : ℕ | x < 10000 ∧ x ≥ 1000}) :=
begin
  sorry
end

end count_valid_4_digit_integers_l782_782953


namespace union_of_sets_l782_782533

noncomputable def A (a : ℝ) := {-1, a}
noncomputable def B (a b : ℝ) := {2^a, b}

theorem union_of_sets (a b : ℝ) (h : A a ∩ B a b = {1}) : A a ∪ B a b = {-1, 1, 2} := by
  sorry

end union_of_sets_l782_782533


namespace hyperbola_range_b_squared_l782_782941

theorem hyperbola_range_b_squared 
  (b : ℝ) (hb : b > 0)
  (hyp : ∀ (x y : ℝ), x^2 - (y^2) / (b^2) = 1)
  (focus : ∃ c : ℝ, c^2 = b^2 + 1)
  (P : ∃ (x y : ℝ), x = - focus.some / 2 ∧ y = (b * focus.some) / 2)
  (circle_condition : x^2 + y^2 < 4 * b^2) :
  7 - 4 * real.sqrt 3 < b^2 ∧ b^2 < 7 + 4 * real.sqrt 3 :=
by
  sorry

end hyperbola_range_b_squared_l782_782941


namespace smallest_positive_integer_divisible_l782_782882

theorem smallest_positive_integer_divisible (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 15 = 0 ∧ m % 13 = 0 ∧ m % 18 = 0 ∧ m = 1170) :=
by {
  use 1170,
  split,
  { exact nat.zero_lt_succ 1169 },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_lcm_of_dvd 15 (and.intro (by norm_num : 15 ∣ 45) (by norm_num : 45 ∣ 1170))) },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_lcm_of_dvd 13 (and.intro (by norm_num : 13 ∣ 1170) (by norm_num : 13 ∣ 1170))) },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_lcm_of_dvd 18 (and.intro (by norm_num : 18 ∣ 18) (by norm_num : 18 ∣ 1170))) },
  refl
}

end smallest_positive_integer_divisible_l782_782882


namespace coordinate_plane_is_cartesian_l782_782205

theorem coordinate_plane_is_cartesian :
  (∀ x, x^2 - x^3 = 0 → (x = 0 ∨ x = 1)) →
  ∃ plane : Type, plane = euclidean_space ℝ 2 :=
sorry

end coordinate_plane_is_cartesian_l782_782205


namespace min_value_x2_y2_z2_l782_782175

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  4 ≤ x^2 + y^2 + z^2 :=
sorry

end min_value_x2_y2_z2_l782_782175


namespace sum_sequence_excluding_90_l782_782822

theorem sum_sequence_excluding_90 : 
  (Finset.sum (Finset.filter (λ n, n ≠ 90) (Finset.range 101 \ Finset.range 80)) id) = 1800 := 
by
  sorry

end sum_sequence_excluding_90_l782_782822


namespace original_price_correct_l782_782893

def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36
def number_of_tires : ℕ := 4
def saving_per_tire : ℝ := total_savings / number_of_tires
def original_price_per_tire : ℝ := sale_price_per_tire + saving_per_tire

theorem original_price_correct :
  original_price_per_tire = 84 :=
by
  sorry

end original_price_correct_l782_782893


namespace power_of_power_evaluation_l782_782458

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l782_782458


namespace max_degree_poly_l782_782979

-- Definition of the polynomial P(x)
def P (x : Real) (d : Nat) : Real :=
  ∑ i in (Finset.range d), (x ^ i : Real) + (x ^ d : Real)

-- The proof statement
theorem max_degree_poly :
  (∃ P : Real → Real, (∀ x : Real, (P x = ∑ i in (Finset.range 5), (x ^ i : Real) + (x ^ 4 + x ^ 3 + x ^ 2 + x + 1)) ∧ ∃ d : Nat, d = 4)) :=
sorry

end max_degree_poly_l782_782979


namespace equal_share_of_marbles_l782_782885

-- Define the number of marbles bought by each friend based on the conditions
def wolfgang_marbles : ℕ := 16
def ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
def michael_marbles : ℕ := 2 * (wolfgang_marbles + ludo_marbles) / 3
def shania_marbles : ℕ := 2 * ludo_marbles
def gabriel_marbles : ℕ := (wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles) - 1
def total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles + gabriel_marbles
def marbles_per_friend : ℕ := total_marbles / 5

-- Mathematical equivalent proof problem
theorem equal_share_of_marbles : marbles_per_friend = 39 := by
  sorry

end equal_share_of_marbles_l782_782885


namespace relationship_between_a_b_c_l782_782169

noncomputable def a : ℝ := log 2 (1 / 5)
noncomputable def b : ℝ := log 3 (1 / 5)
noncomputable def c : ℝ := 2 ^ (-0.1)

theorem relationship_between_a_b_c : c > b ∧ b > a :=
by
  sorry

end relationship_between_a_b_c_l782_782169


namespace min_value_3x_9y_l782_782530

theorem min_value_3x_9y (x y : ℝ) (h : x + 2 * y = 2) : 3^x + 9^y ≥ 6 :=
sorry

end min_value_3x_9y_l782_782530


namespace coefficient_x_squared_l782_782097

theorem coefficient_x_squared (a b : ℝ) (h : (∃ x:ℝ, x = 150 ∧ a * x + b = p) ∧ b = 7500 ∧ p = (λ x:ℝ, - 25 * x^2 + 7500 * x)) :
  a = -25 :=
by
  sorry

end coefficient_x_squared_l782_782097


namespace analytic_expression_and_monotonic_interval_l782_782554

def f (ω φ : ℝ) (x : ℝ) := Real.sin (ω * x + φ)

def g (ω φ π_6 : ℝ) (x : ℝ) := f ω φ (x + π_6)

theorem analytic_expression_and_monotonic_interval 
  (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π)
  (π_2 : ℝ) (h_distance : ∀ x1 x2, f ω φ x1 = 0 
                  → f ω φ x2 = 0 → x1 ≠ x2 → |x1 - x2| = π_2)
  (π_6 : ℝ)
  (h_even : ∀ x, g ω φ π_6 x = g ω φ π_6 (-x)) :
  (∃ φ', φ' = π_6 ∧ ω = 2 ∧ f ω φ' x = Real.sin (2 * x + π_6) ∧ 
          ∀ k : ℤ, real.range_increasing (f ω φ') 
          (Icc (k * π - π/3) (k * π + π/6))) := sorry

end analytic_expression_and_monotonic_interval_l782_782554


namespace multiple_of_bees_l782_782195

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end multiple_of_bees_l782_782195


namespace evaluate_exponent_l782_782450

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l782_782450


namespace problem1_problem2_l782_782931

noncomputable def f (a b c x : ℝ) := (a * x^2 + b * x + c) * Real.exp x

def problem1_statement (a : ℝ) : Prop :=
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → 
    a * x^2 + (a - 1) * x - a ≤ 0) → 0 ≤ a ∧ a ≤ 1

def problem2_statement (a m : ℝ) : Prop :=
  a = 0 → 
  (∃ m, ∀ (x : ℝ), 
    2 * ((1 - x) * Real.exp x) + 4 * x * Real.exp x ≥ m * x + 1 ∧ 
    m * x + 1 ≥ -x^2 + 4 * x + 1) → m = 4

theorem problem1 (a : ℝ) (h0 : f a (-(1 + a)) 1 0 = 1) (h1 : f a (-(1 + a)) 1 1 = 0) : 
  problem1_statement a := 
begin
  intros, 
  sorry
end

theorem problem2 (a : ℝ) (h0 : f 0 (-1) 1 0 = 1) (h1 : f 0 (-1) 1 1 = 0) : 
  problem2_statement 0 4 := 
begin
  intros, 
  sorry
end

end problem1_problem2_l782_782931


namespace ammonium_chloride_potassium_hydroxide_ammonia_l782_782879

theorem ammonium_chloride_potassium_hydroxide_ammonia
  (moles_KOH : ℕ) (moles_NH3 : ℕ) (moles_NH4Cl : ℕ) 
  (reaction : moles_KOH = 3 ∧ moles_NH3 = moles_KOH ∧ moles_NH4Cl >= moles_KOH) : 
  moles_NH3 = 3 :=
by
  sorry

end ammonium_chloride_potassium_hydroxide_ammonia_l782_782879


namespace square_prob_distance_l782_782157

noncomputable def probability_distance_ge_1 (S : set (ℝ × ℝ)) (side_len : ℝ) : ℝ :=
  let a := 28 in
  let b := 1 in
  let c := 32 in
  (a - b * Real.pi) / c

theorem square_prob_distance (side_len : ℝ) (hS : side_len = 2) :
  probability_distance_ge_1 {p | p.1 ≥ 0 ∧ p.1 ≤ side_len ∧ p.2 ≥ 0 ∧ p.2 ≤ side_len} side_len = (28 - Real.pi) / 32 :=
by {
  rw hS,
  unfold probability_distance_ge_1,
  sorry
}

end square_prob_distance_l782_782157


namespace number_of_bowls_l782_782718

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l782_782718


namespace third_number_in_sequence_l782_782700

theorem third_number_in_sequence (n : ℕ) (h_sum : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 63) : n + 2 = 8 :=
by
  -- the proof would be written here
  sorry

end third_number_in_sequence_l782_782700


namespace lines_parallel_l782_782163

-- Definitions of circles and points
variable (Γ Γ' : Set Point)
variable (P Q A B A' B' : Point)
variable (lineAP : Line A P)
variable (lineBQ : Line B Q)
variable (lineAB : Line A B)
variable (lineAprimeP : Line A' P)
variable (lineBprimeQ : Line B' Q)

-- Conditions
axiom circles_intersect : Γ ∩ Γ' = {P, Q}
axiom points_on_circle_Gamma : A ∈ Γ ∧ B ∈ Γ
axiom intersection_Aprime : A' ∈ Γ' ∧ A' ∈ lineAP
axiom intersection_Bprime : B' ∈ Γ' ∧ B' ∈ lineBQ

-- Statement to prove
theorem lines_parallel : lineAB.parallel (Line.mk A' B') :=
sorry

end lines_parallel_l782_782163


namespace area_of_R_l782_782755

open Real

/-- Prove that the area of region R is 1/4 given the conditions:
 1. Vertex E of right-isosceles △ABE is inside unit square ABCD with hypotenuse AB.
 2. Region R is defined as all points inside ABCD and outside △ABE whose distance from AD is between 1/4 and 3/4.
-/
theorem area_of_R (E : Point) (h₁ : E ∈ interior (unit_square ABCD))
  (h₂ : isosceles (△ABE) (right (∠AB = ∠BE))) :
  area (R ABCD ABE) = 1 / 4 :=
sorry

end area_of_R_l782_782755


namespace choose_three_cooks_from_ten_l782_782124

theorem choose_three_cooks_from_ten : 
  (nat.choose 10 3) = 120 := 
by
  sorry

end choose_three_cooks_from_ten_l782_782124


namespace arc_length_of_circle_l782_782991

theorem arc_length_of_circle (r : ℝ) (θ_peripheral : ℝ) (h_r : r = 5) (h_θ : θ_peripheral = 2/3 * π) :
  r * (2/3 * θ_peripheral) = 20 * π / 3 := 
by sorry

end arc_length_of_circle_l782_782991


namespace ratio_of_triangle_to_quad_area_l782_782803

structure Hexagon :=
  (A B C D E F : Type)

structure EquilateralTriangle :=
  (vertices : Set Hexagon)

def area (triangle : EquilateralTriangle) : ℝ := sorry

noncomputable def quad_area
  (triangles : Set EquilateralTriangle) (h : triangles.size = 4) : ℝ :=
  triangles.sum (λ t, area t)

theorem ratio_of_triangle_to_quad_area (hex : Hexagon)
  (triangle : EquilateralTriangle)
  (quad : Set EquilateralTriangle)
  (h1 : quad.size = 4)
  (h2 : triangle ∈ quad)
  (h3 : ∀ t ∈ quad, area t = area triangle)
  : area triangle / quad_area quad h1 = 1 / 4 :=
by sorry

end ratio_of_triangle_to_quad_area_l782_782803


namespace intersection_setA_setB_l782_782561

def setA := {x : ℝ | |x| < 1}
def setB := {x : ℝ | x^2 - 2 * x ≤ 0}

theorem intersection_setA_setB :
  {x : ℝ | 0 ≤ x ∧ x < 1} = setA ∩ setB :=
by
  sorry

end intersection_setA_setB_l782_782561


namespace parallel_suff_nec_not_l782_782972

open Set

variables {Point : Type} [MetricSpace Point]

-- Definitions and assumptions
variables (a b : Set Point) (α : Set Point)
variables [IsLine a] [IsLine b] [IsPlane α]
variables (is_subset : a ⊆ α)
variables (parallel_a_b : IsParallel a b)
variables (parallel_b_alpha : IsParallel b α)

-- Theorem Statement
theorem parallel_suff_nec_not (a b : Set Point) (α : Set Point)
  [IsLine a] [IsLine b] [IsPlane α]
  (is_subset : a ⊆ α) : ¬((IsParallel a b) → (IsParallel b α)) ∧ ¬((IsParallel b α) → (IsParallel a b)) :=
by
  sorry

end parallel_suff_nec_not_l782_782972


namespace partition_contains_square_sum_l782_782860

-- Define a natural number n
def is_square (x : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = x

theorem partition_contains_square_sum (n : ℕ) (hn : n ≥ 15) :
  ∀ (A B : fin n → Prop), (∀ x, A x ∨ B x) ∧ (∀ x, ¬ (A x ∧ B x)) → (∃ a b, a ≠ b ∧ A a ∧ A b ∧ is_square (a + b)) ∨ (∃ c d, c ≠ d ∧ B c ∧ B d ∧ is_square (c + d)) :=
by
  sorry

end partition_contains_square_sum_l782_782860


namespace number_of_bowls_l782_782734

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l782_782734


namespace log_div_sqrt_defined_iff_l782_782493

theorem log_div_sqrt_defined_iff (x : ℝ) : 
  (∃ y, y = (log 5 (5-x)) / sqrt(x+2)) ↔ (-2 < x) ∧ (x < 5) :=
by sorry

end log_div_sqrt_defined_iff_l782_782493


namespace choose_3_out_of_10_l782_782114

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 := by
  sorry

end choose_3_out_of_10_l782_782114


namespace mary_total_spent_l782_782647

-- The conditions given in the problem
def cost_berries : ℝ := 11.08
def cost_apples : ℝ := 14.33
def cost_peaches : ℝ := 9.31

-- The theorem to prove the total cost
theorem mary_total_spent : cost_berries + cost_apples + cost_peaches = 34.72 := 
by
  sorry

end mary_total_spent_l782_782647


namespace pizza_payment_difference_l782_782824

theorem pizza_payment_difference
  (total_slices : ℕ := 12)
  (plain_cost : ℝ := 12)
  (onion_cost : ℝ := 3)
  (jack_onion_slices : ℕ := 4)
  (jack_plain_slices : ℕ := 3)
  (carl_plain_slices : ℕ := 5) :
  let total_cost := plain_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jack_onion_payment := jack_onion_slices * cost_per_slice
  let jack_plain_payment := jack_plain_slices * cost_per_slice
  let jack_total_payment := jack_onion_payment + jack_plain_payment
  let carl_total_payment := carl_plain_slices * cost_per_slice
  jack_total_payment - carl_total_payment = 2.5 :=
by
  sorry

end pizza_payment_difference_l782_782824


namespace james_total_time_l782_782616

def vacuuming_time : ℝ := 3
def cleaning_time : ℝ := 3 * vacuuming_time
def laundry_time : ℝ := (1 / 2) * cleaning_time
def initial_combined_time : ℝ := vacuuming_time + cleaning_time + laundry_time
def organizing_time : ℝ := 2 * initial_combined_time
def total_time : ℝ := vacuuming_time + cleaning_time + laundry_time + organizing_time

theorem james_total_time : total_time = 49.5 := by
  sorry

end james_total_time_l782_782616


namespace point_on_y_axis_is_0_2_l782_782663

def is_on_y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

theorem point_on_y_axis_is_0_2 (M : ℝ × ℝ) :
  M ∈ [(-4,-4), (4,4), (-2,0), (0,2)] → is_on_y_axis M → M = (0,2) :=
by
  intros h_mem h_y_axis
  cases h_mem
  rw [h_mem, is_on_y_axis] at h_y_axis
  contradiction
  -- Similar steps for the other cases, depending on the successful matching of Lean's case analysis on h_mem
sorry

end point_on_y_axis_is_0_2_l782_782663


namespace locus_of_points_is_straight_line_l782_782099

theorem locus_of_points_is_straight_line 
  (a R1 R2 : ℝ) 
  (h_nonzero_a : a ≠ 0)
  (h_positive_R1 : R1 > 0)
  (h_positive_R2 : R2 > 0) :
  ∃ x : ℝ, ∀ (y : ℝ),
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ 
  x = (R1^2 - R2^2) / (4 * a) :=
by
  sorry

end locus_of_points_is_straight_line_l782_782099


namespace problem1_max_min_problem2_min_value_l782_782555

noncomputable def f1 (x : ℝ) : ℝ := -x^2 + 3*x - Real.log x

theorem problem1_max_min :
  (∀ x ∈ set.Icc (1/2 : ℝ) 2, f1 x ≤ 2) ∧
  (∀ x ∈ set.Icc (1/2 : ℝ) 2, f1 x ≥ Real.log 2 + 5/4) :=
sorry

noncomputable def f2 (x : ℝ) (b : ℝ) : ℝ := b * x - Real.log x

theorem problem2_min_value (b : ℝ) :
  (b = Real.exp 2) ↔ (∃ x ∈ set.Ioo 0 Real.exp 1, ∀ y ∈ set.Icc 0 Real.exp (1 : ℝ), f2 y b ≥ 3 ∧ f2 x b = 3) :=
sorry

end problem1_max_min_problem2_min_value_l782_782555


namespace students_behind_hoseok_l782_782313

theorem students_behind_hoseok : 
  ∀ (N : ℕ) (S₁ S₂ : ℕ), N = 20 → S₁ = 11 → S₂ = S₁ + 2 → N - S₂ = 7 :=
by
  intro N S₁ S₂ hN hS₁ hS₂
  rw [hN, hS₁] at hS₂
  simp at hS₂
  rw [hS₂]
  simp
  sorry

end students_behind_hoseok_l782_782313


namespace partition_perfect_square_l782_782866

theorem partition_perfect_square (n : ℕ) (h : n ≥ 15) :
  ∀ A B : finset ℕ, disjoint A B → A ∪ B = finset.range (n + 1) →
  ∃ x y ∈ A ∨ ∃ x y ∈ B, x ≠ y ∧ (∃ k : ℕ, x + y = k^2) :=
begin
  sorry
end

end partition_perfect_square_l782_782866


namespace coefficient_x3_in_expansion_l782_782236

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_x3_in_expansion
    (n : ℕ) 
    (a b : ℝ) 
    (x : ℝ) 
    (h_n : n = 9)
    (h_a : a = x^2)
    (h_b : b = -1 / x) :
    ∃ (r : ℕ), r = 5 ∧ binomial 9 5 * (-1)^5 = -126 :=
by
  sorry

end coefficient_x3_in_expansion_l782_782236


namespace problem1_problem2_l782_782309

/-- Problem 1: Prove the solution to the system of equations is x = 1/2 and y = 5 -/
theorem problem1 (x y : ℚ) (h1 : 2 * x - y = -4) (h2 : 4 * x - 5 * y = -23) : 
  x = 1 / 2 ∧ y = 5 := 
sorry

/-- Problem 2: Prove the value of the expression (x-3y)^{2} - (2x+y)(y-2x) when x = 2 and y = -1 is 40 -/
theorem problem2 (x y : ℚ) (h1 : x = 2) (h2 : y = -1) : 
  (x - 3 * y) ^ 2 - (2 * x + y) * (y - 2 * x) = 40 := 
sorry

end problem1_problem2_l782_782309


namespace y_days_to_complete_work_l782_782774

theorem y_days_to_complete_work:
  (forall (y : ℕ), 
    (forall (m r : ℕ) (m_days r_days : ℕ) (m_work r_work : ℚ),
      m = 45 →
      m_days = 20 →
      r_days = 30 →
      m_work = m_days / m →
      r_work = 1 - m_work →
      let r_rate := r_work / r_days in
      let y_rate := 1 / y in
      r_rate = y_rate → 
      y = 54) →
    y = 54) :=
by
  intros y H
  sorry

end y_days_to_complete_work_l782_782774


namespace find_side_f_l782_782612

variable (D E : ℝ)

theorem find_side_f (d e f : ℝ) (cos_DE : ℝ) 
  (hd : d = 7) (he : e = 3) (hcos : cos_DE = 7 / 8) :
  f = 6.5 :=
by
  rw [hd, he, hcos]
  sorry

end find_side_f_l782_782612


namespace sin_squared_analytic_log_z_analytic_power_formula_analytic_l782_782277

noncomputable def sin_squared (z : ℂ) : ℂ := complex.sin(z)^2
noncomputable def log_z (z : ℂ) : ℂ := complex.log(z)
noncomputable def power_formula (z : ℂ) : ℂ := 4^(z^2 + 2*z*complex.I)

theorem sin_squared_analytic (z : ℂ) : complex.analytic_at sin_squared z := sorry

theorem log_z_analytic (z : ℂ): complex.analytic_at log_z z := sorry

theorem power_formula_analytic (z : ℂ): complex.analytic_at power_formula z := sorry

end sin_squared_analytic_log_z_analytic_power_formula_analytic_l782_782277


namespace isosceles_triangle_sides_l782_782595

/-
  Given: 
  - An isosceles triangle with a perimeter of 60 cm.
  - The intersection point of the medians lies on the inscribed circle.
  Prove:
  - The sides of the triangle are 25 cm, 25 cm, and 10 cm.
-/

theorem isosceles_triangle_sides (AB BC AC : ℝ) 
  (h1 : AB = BC)
  (h2 : AB + BC + AC = 60) 
  (h3 : ∃ r : ℝ, r > 0 ∧ 6 * r = AC ∧ 3 * r * AC = 30 * r) :
  AB = 25 ∧ BC = 25 ∧ AC = 10 :=
sorry

end isosceles_triangle_sides_l782_782595


namespace probability_third_shiny_coin_appears_on_sixth_draw_l782_782788

theorem probability_third_shiny_coin_appears_on_sixth_draw :
  let shiny_coins := 4 in
  let dull_coins := 3 in
  ∑ x in (finset.range 5), ↑(nat.choose 5 x) = 10 → -- ways to pick 2 shiny coins in first 5 draws
  (10 / 35 : ℚ) = (2 / 7 : ℚ) → -- Probability calculation
  (2 + 7 = 9) := 
begin
  intro shiny_coins dull_coins,
  sorry
end

end probability_third_shiny_coin_appears_on_sixth_draw_l782_782788


namespace f_f_neg2_eq_14_l782_782509

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 else 2^x - 2

theorem f_f_neg2_eq_14 : f (f (-2)) = 14 := 
sorry

end f_f_neg2_eq_14_l782_782509


namespace partition_contains_square_sum_l782_782862

-- Define a natural number n
def is_square (x : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = x

theorem partition_contains_square_sum (n : ℕ) (hn : n ≥ 15) :
  ∀ (A B : fin n → Prop), (∀ x, A x ∨ B x) ∧ (∀ x, ¬ (A x ∧ B x)) → (∃ a b, a ≠ b ∧ A a ∧ A b ∧ is_square (a + b)) ∨ (∃ c d, c ≠ d ∧ B c ∧ B d ∧ is_square (c + d)) :=
by
  sorry

end partition_contains_square_sum_l782_782862


namespace perimeter_triangle_APR_l782_782272

variable (A B C P Q R : Point)
variable (x y : ℝ)
variable (circle : Circle)

-- Conditions
axiom tangent_1 : IsTangent A B circle
axiom tangent_2 : IsTangent A C circle
axiom tangent_3 : IsTangent P Q circle
axiom intersects_AB : Between P A B
axiom intersects_AC : Between R A C
axiom BP_PQ_x : (dist B P) = x ∧ (dist P Q) = x
axiom QR_CR_y : (dist Q R) = y ∧ (dist C R) = y
axiom AB_24 : (dist A B) = 24
axiom AC_24 : (dist A C) = 24
axiom x_y_sum_12 : x + y = 12

-- The perimeter of triangle APR
theorem perimeter_triangle_APR : (dist A P) + (dist P R) + (dist A R) = 48 :=
by 
  sorry

end perimeter_triangle_APR_l782_782272


namespace liquid_in_cylinders_l782_782704

theorem liquid_in_cylinders (n : ℕ) (a : ℝ) (h1 : 2 ≤ n) :
  (∃ x : ℕ → ℝ, ∀ (k : ℕ), (1 ≤ k ∧ k ≤ n) → 
    (if k = 1 then 
      x k = a * n * (n - 2) / (n - 1) ^ 2 
    else if k = 2 then 
      x k = a * (n^2 - 2*n + 2) / (n - 1) ^ 2 
    else 
      x k = a)) :=
sorry

end liquid_in_cylinders_l782_782704


namespace initial_violet_balloons_l782_782620

-- Define initial conditions and variables
def red_balloons := 4
def violet_balloons_lost := 3
def current_violet_balloons := 4

-- Define the theorem we want to prove
theorem initial_violet_balloons (red_balloons : ℕ) (violet_balloons_lost : ℕ) (current_violet_balloons : ℕ) : 
  red_balloons = 4 → violet_balloons_lost = 3 → current_violet_balloons = 4 → (current_violet_balloons + violet_balloons_lost) = 7 :=
by
  intros
  sorry

end initial_violet_balloons_l782_782620


namespace digit_as_exponent_correct_l782_782088

theorem digit_as_exponent_correct (n : ℤ) (exponent : ℤ) (h1 : n = 101) (h2 : exponent = 2) : 
  n - 10 ^ exponent = 1 :=
by
  rw [h1, h2]
  calc 101 - 10 ^ 2 = 101 - 100 := by norm_num
                  ... = 1 := by norm_num
                  
-- The above proof uses rewrites to substitute values of n and exponent followed by calculations.

end digit_as_exponent_correct_l782_782088


namespace no_strategy_wins_more_than_half_probability_l782_782326

theorem no_strategy_wins_more_than_half_probability
  (deck : Finset Card)
  (red_count black_count : ℕ)
  (well_shuffled : deck.shuffled)
  (player_strategy : (Finset Card) → Bool) :
  ∀ r b, red_count = r ∧ black_count = b →
    (∀ red black : ℕ, (red + black = r + b) → 
      (red / (red + black + 1) ≤ 0.5)) :=
sorry

end no_strategy_wins_more_than_half_probability_l782_782326


namespace evaluate_exponent_l782_782451

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l782_782451


namespace range_for_m_l782_782062

def A := { x : ℝ | x^2 - 3 * x - 10 < 0 }
def B (m : ℝ) := { x : ℝ | m + 1 < x ∧ x < 1 - 3 * m }

theorem range_for_m (m : ℝ) (h : ∀ x, x ∈ A ∪ B m ↔ x ∈ B m) : m ≤ -3 := sorry

end range_for_m_l782_782062


namespace solve_for_a_l782_782577

theorem solve_for_a (a : ℝ) 
  (h : ∃ x > 0, sqrt x = 2 * a + 1 ∧ sqrt x = a + 5) :
  a = 4 := 
by
  sorry

end solve_for_a_l782_782577


namespace isosceles_triangle_solution_l782_782599

noncomputable def isosceles_sides : (a b : ℝ) → (a = 25) ∧ (b = 10) → (sides : list ℝ)
| a b ⟨h₁, h₂⟩ := [a, a, b]

theorem isosceles_triangle_solution
(perimeter : ℝ) (iso_condition : ∃ a b, a = 25 ∧ b = 10 ∧ 2 * a + b = perimeter)
(intersect_uninscribed_boundary : ∀ (O M : ℝ) (ratio : ℝ), O = 2 / 3 * M) : 
sides 
       ∃ a b, (a = 25) ∧ (b = 10) ∧ (perimeter = 60)  :=
begin
  sorry
end

end isosceles_triangle_solution_l782_782599


namespace part_I_part_II_l782_782053

noncomputable def f (x : ℝ) := log x - (1/2) * x^2
noncomputable def g (x : ℝ) (m : ℝ) := (1/2) * m * x^2 + x
noncomputable def F (x : ℝ) (m : ℝ) := f x + g x m

-- Part (Ⅰ)
theorem part_I :
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv f x > 0) ∧
  (∀ x : ℝ, x > 1 → deriv f x < 0) ∧
  (f 1 = -1 / 2) :=
sorry

-- Part (Ⅱ)
theorem part_II (F : ℝ → ℝ → ℝ) :
  (∀ x : ℝ, F x 2 ≤ 2 * x - 1) →
  (∀ m : ℝ, (∀ x : ℝ, F x m ≤ m * x - 1) → 2 ≤ m) :=
sorry

end part_I_part_II_l782_782053


namespace calculate_tangent_lines_and_length_l782_782039

variable (P : ℝ × ℝ)
variable (C : ℝ × ℝ)
variable (r : ℝ)
variable (k1 k2 : ℝ)
variable (l1 l2 : ℝ → ℝ → Prop)
noncomputable def circle (C : ℝ × ℝ) (r : ℝ) : ℝ → ℝ → Prop := λ x y, (x - C.1)^2 + (y - C.2)^2 = r^2

noncomputable def line (P : ℝ × ℝ) (k : ℝ) : ℝ → ℝ → Prop := λ x y, y + P.2 = k * (x - P.1)

noncomputable def tangent_length (C P : ℝ × ℝ) (r : ℝ) : ℝ :=
  let PC_dist := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  (Real.sqrt (PC_dist^2 - r^2))

theorem calculate_tangent_lines_and_length :
  let P := (2, -1)
  let C := (1, 2)
  let r := Real.sqrt 2
  let k1 := -1
  let k2 := 7
  let l1 := λ x y, x + y - 1 = 0
  let l2 := λ x y, 7*x - y - 15 = 0
  (line P k1 = l1) ∧ (line P k2 = l2) ∧ tangent_length C P r = 2 * Real.sqrt 2 :=
by
  sorry

end calculate_tangent_lines_and_length_l782_782039


namespace problem1_problem2_l782_782396

theorem problem1 (a b : ℝ) :
  5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b := 
by sorry

theorem problem2 (m n : ℝ) :
  -5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2 := 
by sorry

end problem1_problem2_l782_782396


namespace marks_obtained_l782_782373

-- Definitions based on the conditions
def max_marks : ℕ := 400
def passing_percentage : ℕ := 36
def fail_by : ℕ := 14

-- Calculations based on provided conditions
def passing_marks : ℕ := (passing_percentage * max_marks) / 100

-- The theorem to be proved
theorem marks_obtained : ℕ :=
  passing_marks - fail_by = 130 := by sorry

end marks_obtained_l782_782373


namespace isosceles_triangle_similar_perimeter_l782_782106

theorem isosceles_triangle_similar_perimeter :
  ∀ (s1 s2 s3 : ℝ) (s1' : ℝ) (ratio : ℝ),
    s1 = 18 ∧ s2 = 24 ∧ s3 = 24 →
    s1' = 45 →
    ratio = s1' / s1 →
    let s2' := s2 * ratio in
    let s3' := s3 * ratio in
    let perimeter := s1' + s2' + s3' in
    perimeter = 165 :=
by
  intros s1 s2 s3 s1' ratio
  rintros ⟨h1, h2, h3⟩ h4 h5
  let s2' := s2 * ratio
  let s3' := s3 * ratio
  let perimeter := s1' + s2' + s3'
  sorry

end isosceles_triangle_similar_perimeter_l782_782106


namespace perpendicular_PA_PI_l782_782626

noncomputable theory

open set

variables {A B C O I D M P : Type} [innermost_circle (A B C) (O)] [incenter (I) (triangle A B C)]
  [incircle_tangent_point (D) (side B C)] [second_intersection (M) (A I) (O)]
  [intersection_point (P) (line D M) (O)]

theorem perpendicular_PA_PI : perp PA PI :=
sorry

end perpendicular_PA_PI_l782_782626


namespace bob_daily_work_hours_l782_782002

theorem bob_daily_work_hours
  (total_hours_in_month : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_working_days : ℕ)
  (daily_working_hours : ℕ)
  (h1 : total_hours_in_month = 200)
  (h2 : days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_working_days = days_per_week * weeks_per_month)
  (h5 : daily_working_hours = total_hours_in_month / total_working_days) :
  daily_working_hours = 10 := 
sorry

end bob_daily_work_hours_l782_782002


namespace tan_at_max_value_l782_782089

theorem tan_at_max_value : 
  ∃ x₀, (∀ x, 3 * Real.sin x₀ - 4 * Real.cos x₀ ≥ 3 * Real.sin x - 4 * Real.cos x) → Real.tan x₀ = 3/4 := 
sorry

end tan_at_max_value_l782_782089


namespace area_trapezoid_EFGH_l782_782767

-- Definition of points based on the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def E := Point.mk 3 (-3)
def F := Point.mk 3 2
def G := Point.mk 8 10
def H := Point.mk 8 4

-- Definition of distance function for vertical line segments
def vertical_distance (p1 p2 : Point) : ℝ :=
  abs (p2.y - p1.y)

-- Definition of height of the trapezoid, since the x-coordinates of the bases are the same
def height : ℝ :=
  abs (G.x - E.x)

-- Area of the trapezoid function
def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

-- Proof of trapezoid area
theorem area_trapezoid_EFGH : 
  vertical_distance E F = 5 → 
  vertical_distance G H = 6 → 
  height = 5 → 
  trapezoid_area (vertical_distance E F) (vertical_distance G H) height = 27.5 := 
by
  intros h1 h2 h3
  sorry

end area_trapezoid_EFGH_l782_782767


namespace solution_set_l782_782257

noncomputable def inequality_solution_set (x : ℝ) : Prop :=
  2^(2 * x) ≤ 3 * 2^(x + sqrt x) + 4 * 2^(2 * sqrt x)

theorem solution_set (x : ℝ) : 
  inequality_solution_set x ↔ (0 ≤ x ∧ x ≤ 4) := 
by
  sorry

end solution_set_l782_782257


namespace number_of_common_divisors_90_105_l782_782955

theorem number_of_common_divisors_90_105 : 
  let divisors_90 := {d | d ∣ 90} in
  let divisors_105 := {d | d ∣ 105} in
  let common_divisors := divisors_90 ∩ divisors_105 in
  common_divisors.to_finset.card = 8 :=
by {
  let divisors_90 := {d | d ∣ 90},
  let divisors_105 := {d | d ∣ 105},
  let common_divisors := divisors_90 ∩ divisors_105,
  exact 8
}

end number_of_common_divisors_90_105_l782_782955


namespace min_people_like_mozart_and_bach_but_not_beethoven_l782_782656

theorem min_people_like_mozart_and_bach_but_not_beethoven 
  (total : ℕ) (likes_mozart : ℕ) (likes_bach : ℕ) (likes_beethoven : ℕ)
  (h_total : total = 120)
  (h_likes_mozart : likes_mozart = 95)
  (h_likes_bach : likes_bach = 80)
  (h_likes_beethoven : likes_beethoven = 75) :
  ∃ (min_overlap : ℕ), min_overlap = 45 :=
by
  use 45
  sorry

end min_people_like_mozart_and_bach_but_not_beethoven_l782_782656


namespace abs_inequality_solution_l782_782483

theorem abs_inequality_solution (x : ℝ) : 
  3 ≤ |x - 3| ∧ |x - 3| ≤ 7 ↔ (-4 ≤ x ∧ x ≤ 0) ∨ (6 ≤ x ∧ x ≤ 10) := 
by {
  sorry
}

end abs_inequality_solution_l782_782483


namespace triangle_angle_proof_l782_782585

theorem triangle_angle_proof
  (A B C D : Point)
  (h1 : is_triangle A B C) 
  (h2 : has_segment DC 2 BD)
  (h3 : angle ABC = 45)
  (h4 : angle ADC = 60) : 
  angle ACB = 75 := 
sorry

end triangle_angle_proof_l782_782585


namespace probability_difference_is_zero_l782_782784

noncomputable def boys_ages : List ℕ := [3, 4, 6, 7]
noncomputable def girls_ages : List ℕ := [5, 8, 9, 11]

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_odd (n : ℕ) : Prop :=
  ¬(is_even n)

def probability_difference (boys_ages girls_ages : List ℕ) : ℕ :=
  let combined_ages := do
    boys <- boys_ages.combinations 3
    girls <- girls_ages.combinations 3
    pure (boys.sum + girls.sum)
  let multiples_of_three := combined_ages.filter is_multiple_of_three
  let even_multiples := multiples_of_three.filter is_even
  let odd_multiples := multiples_of_three.filter is_odd
  even_multiples.length - odd_multiples.length

theorem probability_difference_is_zero : probability_difference boys_ages girls_ages = 0 := by
  sorry

end probability_difference_is_zero_l782_782784


namespace cevian_ratios_sum_to_one_l782_782179

variables {A B C M D E F : Point}

theorem cevian_ratios_sum_to_one
  (hM : inside_triangle M A B C)
  (hD : point_intersection (line_through A M) (line_through B C) = D)
  (hE : point_intersection (line_through B M) (line_through C A) = E)
  (hF : point_intersection (line_through C M) (line_through A B) = F) :
  (MD / AD) + (ME / BE) + (MF / CF) = 1 :=
sorry

end cevian_ratios_sum_to_one_l782_782179


namespace count_bijections_with_fixed_points_l782_782878

def setA : Finset ℕ := {1, 2, 3, 4, 5}

noncomputable def fixed_points_bijections (f: {g: ℕ → ℕ // ∀ a ∈ setA, g a ∈ setA ∧ Function.Bijective g}) : ℕ :=
(setA.filter (λ x, f.1 x = x)).card

theorem count_bijections_with_fixed_points :
  (Finset.filter (λ f: {g: ℕ → ℕ // ∀ a ∈ setA, g a ∈ setA ∧ Function.Bijective g}, fixed_points_bijections f = 2)
     (Finset.univ : Finset {g: ℕ → ℕ // ∀ a ∈ setA, g a ∈ setA ∧ Function.Bijective g})).card = 20 := 
sorry

end count_bijections_with_fixed_points_l782_782878


namespace equal_segments_through_H_l782_782627

-- Definition of cyclic quadrilateral and its circumcircle
def isCyclicQuadrilateral (A B C D O : Point) (ω : Circle) (H : Point) (O1 O2 : Point) 
(M1 M2 N1 N2 : Point) : Prop := 
  CyclicQuadrilateral A B C D ω ∧ 
  Center ω = O ∧ 
  IntersectsAt (Diagonals A B C D) H ∧
  Circumcenter (Triangle.mk A H D) = O1 ∧
  Circumcenter (Triangle.mk B H C) = O2 ∧
  IntersectsAt (LineThrough H) ω M1 M2 ∧
  IntersectsAt (Circumcircle.mk O1 H O) (LineThrough H) N1 ∧
  IntersectsAt (Circumcircle.mk O2 H O) (LineThrough H) N2 ∧
  InsideCircumcircle ω N1 ∧
  InsideCircumcircle ω N2

-- The theorem statement proving M1N1 = M2N2 given the conditions
theorem equal_segments_through_H {A B C D O : Point} {ω : Circle} {H : Point} {O1 O2 : Point} 
{M1 M2 N1 N2 : Point} :
  isCyclicQuadrilateral A B C D O ω H O1 O2 M1 M2 N1 N2 → 
  distance M1 N1 = distance M2 N2 :=
begin
  intro h,
  sorry
end

end equal_segments_through_H_l782_782627


namespace evaluate_exponent_l782_782445

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l782_782445


namespace number_of_bowls_l782_782731

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l782_782731


namespace david_english_marks_l782_782412

theorem david_english_marks (avg math phys chem bio : ℕ) (h_avg : avg = 76)
  (h_math : math = 65) (h_phys : phys = 82) (h_chem : chem = 67) (h_bio : bio = 85) :
  let eng := (avg * 5 - (math + phys + chem + bio)) in eng = 81 := 
by 
  have sum := math + phys + chem + bio
  have eng := avg * 5 - sum
  have h_sum : sum = 65 + 82 + 67 + 85 := by rw [h_math, h_phys, h_chem, h_bio]
  have h_sum_calc : sum = 299 := by norm_num
  have h_eng : eng = 76 * 5 - 299 := by rw [h_avg, h_sum_calc]
  have h_eng_calc : eng = 81 := by norm_num
  
  exact h_eng_calc

end david_english_marks_l782_782412


namespace lateral_surface_area_of_truncated_cone_l782_782876

-- Definitions for the given conditions
variable (α r : ℝ)

-- Assume alpha is an acute angle
axiom acute_angle (hα : 0 < α) (hα_lt_pi : α < π / 2)

-- Proof statement that needs to be shown
theorem lateral_surface_area_of_truncated_cone 
  (hα : 0 < α) (hα_lt_pi : α < π / 2) :
  let 
    S_lateral := (8 * Real.sqrt 3 * Real.pi * r^2) / (3 * Real.sin α^2)
  in
  S_lateral = (8 * Real.sqrt 3 * Real.pi * r^2) / (3 * Real.sin α^2) := 
by 
  -- Insert proof here
  sorry

end lateral_surface_area_of_truncated_cone_l782_782876


namespace range_of_m_l782_782943

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := 
  (2 * m - 3)^2 - 4 > 0

def q (m : ℝ) : Prop := 
  2 * m > 3

-- Theorem statement
theorem range_of_m (m : ℝ) : ¬ (p m ∧ q m) ∧ (p m ∨ q m) ↔ (m < 1 / 2 ∨ 3 / 2 < m ∧ m ≤ 5 / 2) :=
  sorry

end range_of_m_l782_782943


namespace alpha_sufficient_not_necessary_for_beta_l782_782191

noncomputable def statement_alpha : Prop := (x = 1) ∧ (y = 2)
noncomputable def statement_beta : Prop := (x + y = 3)

theorem alpha_sufficient_not_necessary_for_beta :
  (statement_alpha -> statement_beta) ∧ ¬(statement_beta -> statement_alpha) :=
by sorry

end alpha_sufficient_not_necessary_for_beta_l782_782191


namespace tiling_remainder_l782_782315

theorem tiling_remainder (N : ℕ) :
  (let tilings : ℕ := 21 * 15 + 35 * 54 + 35 * 180 + 21 * 570 + 7 * 1776 + 1 * 5436 in
   N = tilings) →
  N % 1000 = 343 := by
  sorry

end tiling_remainder_l782_782315


namespace Tara_books_to_sell_l782_782231

theorem Tara_books_to_sell 
  (initial_savings : ℕ)
  (clarinet_cost : ℕ)
  (book_price : ℕ)
  (accessory_cost : ℕ)
  (halfway_loss_savings : bool)
  (h_initial_savings : initial_savings = 10)
  (h_clarinet_cost : clarinet_cost = 90)
  (h_book_price : book_price = 4)
  (h_accessory_cost : accessory_cost = 20)
  (h_halfway_loss_savings : halfway_loss_savings = tt)
  : let halfway_goal := (clarinet_cost - initial_savings) / 2 in
    let total_goal := (clarinet_cost - initial_savings) + accessory_cost in
    let books_to_halfway := halfway_goal / book_price in
    let books_to_total := total_goal / book_price in
    let total_books := books_to_halfway + books_to_total
    in total_books = 35 :=
by
  -- Proof goes here.
  sorry

end Tara_books_to_sell_l782_782231


namespace backpacking_trip_cooks_l782_782133

theorem backpacking_trip_cooks :
  nat.choose 10 3 = 120 :=
sorry

end backpacking_trip_cooks_l782_782133


namespace common_limit_exists_seq_relationship_l782_782181

variables (a b k : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a < b)
noncomputable def a_seq : ℕ → ℝ
| 0       := a
| (n + 1) := 2 * a_seq n * b_seq n / (a_seq n + b_seq n)

noncomputable def b_seq : ℕ → ℝ
| 0       := b
| (n + 1) := (a_seq n + b_seq n) / 2

theorem common_limit_exists :
  ∃ L, (tendsto (a_seq a b) at_top (𝓝 L)) ∧ (tendsto (b_seq a b) at_top (𝓝 L)) ∧ L = sqrt (a * b) := sorry

theorem seq_relationship :
  a = 1 →
  ∃ k, b = k → ∀ n, b_seq 1 k n = x_seq n := sorry

end common_limit_exists_seq_relationship_l782_782181


namespace second_fisherman_more_fish_l782_782581

-- Defining the conditions
def total_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def days_rate1 : ℕ := 30
def days_rate2 : ℕ := 60
def days_rate3 : ℕ := total_days - (days_rate1 + days_rate2)

-- Calculating the total number of fish caught by both fishermen
def total_fish_first_fisherman : ℕ := first_fisherman_rate * total_days
def total_fish_second_fisherman : ℕ := (second_fisherman_rate1 * days_rate1) + 
                                        (second_fisherman_rate2 * days_rate2) + 
                                        (second_fisherman_rate3 * days_rate3)

-- Theorem stating the difference in the number of fish caught
theorem second_fisherman_more_fish : (total_fish_second_fisherman - total_fish_first_fisherman) = 3 := 
by
  sorry

end second_fisherman_more_fish_l782_782581


namespace num_valid_constants_m_l782_782613

theorem num_valid_constants_m : 
  ∃ (m1 m2 : ℝ), 
  m1 ≠ m2 ∧ 
  (∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∧ 8 = m1 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∧ (1 / 2) = m2 ∨ 2 * c / d = 2)) ∧
  (∀ (m : ℝ), 
    (m = m1 ∨ m = m2) →
    ∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∨ 2 * c / d = 2)) :=
sorry

end num_valid_constants_m_l782_782613


namespace range_of_m_l782_782552

noncomputable def F (x m : ℝ) : ℝ := -x^2 - m*x + 1

theorem range_of_m (m : ℝ) (h1 : ∀ x : ℝ, m ≤ x ∧ x ≤ m + 1 → F x m > 0) : 
  -real.sqrt 2 / 2 < m ∧ m < 0 :=
by sorry

end range_of_m_l782_782552


namespace decagon_diagonals_l782_782832

theorem decagon_diagonals : 
  let n := 10 in 
  (n * (n - 3)) / 2 = 35 :=
by 
  let n := 10
  exact sorry

end decagon_diagonals_l782_782832


namespace min_benches_l782_782351
-- Import the necessary library

-- Defining the problem in Lean statement
theorem min_benches (N : ℕ) :
  (∀ a c : ℕ, (8 * N = a) ∧ (12 * N = c) ∧ (a = c)) → N = 6 :=
by
  sorry

end min_benches_l782_782351


namespace main_problem_l782_782143

-- Define the polar equations of C1 and C2
def polar_C1 (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def polar_C2 (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 4 * Real.cos θ

-- Define the parametric equation of curve C
def parametric_C (t : ℝ) : ℝ × ℝ :=
  (2 + 1 / 2 * t, (Real.sqrt 3) / 2 * t)

-- Define the rectangular coordinates equations of C1 and C2
def rect_C1 (x y : ℝ) : Prop := x^2 + y^2 = 2 * x
def rect_C2 (x y : ℝ) : Prop := y^2 = 4 * x

-- Main theorem combining all statements
theorem main_problem
  (ρ θ t : ℝ) :
  (polar_C1 ρ θ → rect_C1 (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (polar_C2 ρ θ → rect_C2 (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  let P := (parametric_C t).fst,
  let Q := (parametric_C (t - 1)).snd, -- Example points for PQRS adjustment
  let R := (parametric_C (t + 1)).fst,
  let S := (parametric_C (t + 2)).snd in
  ∃ t1 t2 t3 t4 : ℝ, (parametric_C t1).fst = (parametric_C t2).fst ∧
                   (parametric_C t3).snd = (parametric_C t4).snd ∧
                   |(t2 - t1) - (t4 - t3)| = 11 / 3 :=
by
  sorry

end main_problem_l782_782143


namespace sum_of_roots_l782_782964

theorem sum_of_roots (x : ℝ) (h : (2 * x + 3) * (x - 5) = 27) :
  (let root_sum := -(-7) / 2 in root_sum) = 7 / 2 :=
by
  sorry

end sum_of_roots_l782_782964


namespace value_of_a_l782_782983

theorem value_of_a (a b : ℚ) (h₁ : b = 3 * a) (h₂ : b = 12 - 5 * a) : a = 3 / 2 :=
by
  sorry

end value_of_a_l782_782983


namespace number_of_people_is_8_l782_782235

noncomputable def find_number_of_people (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ) (n : ℕ) :=
  avg_increase = weight_diff / n ∧ old_weight = 70 ∧ new_weight = 90 ∧ weight_diff = new_weight - old_weight → n = 8

theorem number_of_people_is_8 :
  ∃ n : ℕ, find_number_of_people 2.5 70 90 20 n :=
by
  use 8
  sorry

end number_of_people_is_8_l782_782235


namespace maximum_terms_l782_782944

def seq (n : ℕ) : ℝ := (2 / 3)^(n - 1) * (n - 8)

theorem maximum_terms (n : ℕ) (hn : n ∈ [10, 11]) :
  seq n = max (seq 10) (seq 11) := sorry

end maximum_terms_l782_782944


namespace combined_shape_area_l782_782356

noncomputable def trapezium_area (a b h : ℝ) : ℝ := 1/2 * (a + b) * h

noncomputable def triangle_area (b h : ℝ) : ℝ := 1/2 * b * h

theorem combined_shape_area :
  let a₁ := 20 -- longer parallel side of the trapezium
      a₂ := 18 -- shorter parallel side of the trapezium
      h₁ := 5  -- height (distance between parallel sides) of the trapezium
      b := 20  -- base of the right-angled triangle
      h₂ := 7  -- height (longer leg) of the right-angled triangle
  in trapezium_area a₁ a₂ h₁ + triangle_area b h₂ = 165 :=
by
  sorry

end combined_shape_area_l782_782356


namespace isosceles_triangle_sides_l782_782593

theorem isosceles_triangle_sides (A B C : Type) [plus : A → A → A] [le : A → A → Prop] [mul : A → A → A] [div : A → A → A] [zero : A] [one : A]
  (triangle_iso : ∀ (x y z : A), plus x y z = 60 → le (plus x z) y → le (plus y z) x → le (plus x y) z → x = y)
  (per_60 : plus (plus A B) C = 60)
  (medians_inter_inscribed : A → A → A → A)
  (centroid_on_incircle : (medians_inter_inscribed A B C) = Inscribed_circle)
  (A : A)
  (B : A)
  (C : A) : 
  A = 25 ∧ B = 25 ∧ C = 10 := 
sorry

end isosceles_triangle_sides_l782_782593


namespace identify_false_propositions_l782_782636

-- Definitions corresponding to the conditions
variables (m n : Type) [Field m] [Field n] (α β γ : Type) [Field α] [Field β] [Field γ]

-- We need to define the relationship:
def parallel (a b : Type) [Field a] [Field b] : Prop := sorry
def perpendicular (a b : Type) [Field a] [Field b] : Prop := sorry
def subset (a b : Type) [Field a] [Field b] : Prop := sorry

-- Defining the propositions
def Prop1 : Prop := (parallel α β ∧ parallel α γ) → parallel β γ
def Prop2 : Prop := (perpendicular α β ∧ parallel m α) → perpendicular m β
def Prop3 : Prop := (perpendicular m α ∧ parallel m β) → perpendicular α β
def Prop4 : Prop := (parallel m n ∧ subset n α) → parallel m α

-- The final statement should check the false propositions
theorem identify_false_propositions : (¬Prop2) ∧ (¬Prop4) := by
  sorry

end identify_false_propositions_l782_782636


namespace angle_FYZ_l782_782140

open Set Classical

variables {P Q R S T U : Type*} [LinearOrder P]

theorem angle_FYZ (AB CD EF GH XF : Line P) (F Y Z : P)
  (h1: Parallel AB CD)
  (h2: Parallel EF GH)
  (h3: Angle XF F = 135)
  (h4: lies_on Y CD ∧ lies_on Y XF)
  (h5: lies_on Z GH ∧ lies_on Z XF) :
  Angle F Y Z = 45 :=
by sorry

end angle_FYZ_l782_782140


namespace solve_x_eq_l782_782074

theorem solve_x_eq : ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 ∧ x = -7 / 3 :=
by 
  sorry

end solve_x_eq_l782_782074


namespace coords_of_P_max_PA_distance_l782_782048

open Real

noncomputable def A : (ℝ × ℝ) := (0, -5)

def on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, x = P.1 ∧ y = P.2 ∧ (x - 2)^2 + (y + 3)^2 = 2

def max_PA_distance (P : (ℝ × ℝ)) : Prop :=
  dist P A = max (dist (3, -2) A) (dist (1, -4) A)

theorem coords_of_P_max_PA_distance (P : (ℝ × ℝ)) :
  on_circle P →
  max_PA_distance P →
  P = (3, -2) :=
  sorry

end coords_of_P_max_PA_distance_l782_782048


namespace cone_to_prism_volume_ratio_l782_782805

theorem cone_to_prism_volume_ratio (a h : ℝ) (ha : a > 0) (hh : h > 0) :
  let r := a,
      V_cone := (1 / 3) * π * r^2 * h,
      V_prism := (2 * a) * (3 * a) * h in
  (V_cone / V_prism) = (π / 18) :=
by
  let r := a,
      V_cone := (1 / 3) * π * r^2 * h,
      V_prism := (2 * a) * (3 * a) * h
  sorry

end cone_to_prism_volume_ratio_l782_782805


namespace construct_sqrt_ab_l782_782531

-- Given segments a and b of positive real lengths, construct a segment of length sqrt(a * b)
theorem construct_sqrt_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ∃ (x : ℝ), x = sqrt (a * b) :=
by
  sorry

end construct_sqrt_ab_l782_782531


namespace sum_of_squares_eq_two_l782_782968

theorem sum_of_squares_eq_two {a b : ℝ} (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 := sorry

end sum_of_squares_eq_two_l782_782968


namespace partition_with_sum_square_l782_782853

def sum_is_square (a b : ℕ) : Prop := ∃ k : ℕ, a + b = k * k

theorem partition_with_sum_square (n : ℕ) (h : n ≥ 15) :
  ∀ (s₁ s₂ : finset ℕ), (∅ ⊂ s₁ ∪ s₂ ∧ s₁ ∩ s₂ = ∅ ∧ (∀ x ∈ s₁ ∪ s₂, x ∈ finset.range (n + 1))) →
  (∃ a b : ℕ, a ≠ b ∧ (a ∈ s₁ ∧ b ∈ s₁ ∨ a ∈ s₂ ∧ b ∈ s₂) ∧ sum_is_square a b) :=
by sorry

end partition_with_sum_square_l782_782853


namespace third_number_exists_l782_782187

theorem third_number_exists (n a b c r x : ℕ)
  (h1 : 1305 = n * a + r)
  (h2 : 4665 = n * b + r)
  (h3 : x = n * c + r)
  (h4 : digit_sum n = 4)
  (h5 : ∀ m, m ∣ 3360 → digit_sum m = 4 → m ≤ n) :
  x = 4705 := 
by
  sorry

end third_number_exists_l782_782187


namespace eval_expr_l782_782428

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l782_782428


namespace qrs_company_profit_increase_l782_782301

theorem qrs_company_profit_increase :
  ∀ (P : ℝ), let april_profit := 1.10 * P,
                 may_profit := 0.88 * P,
                 june_profit := 1.32 * P in
  ((june_profit - P) / P) * 100 = 32 := by
  sorry

end qrs_company_profit_increase_l782_782301


namespace de_morgan_logic_l782_782083

theorem de_morgan_logic (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
by
  split
  all_goals { intros h3; cases h2 (or.inl h3); cases h2 (or.inr h3) }

end de_morgan_logic_l782_782083


namespace probability_distance_ge_one_l782_782161

theorem probability_distance_ge_one (S : set ℝ) (side_length_S : ∀ x ∈ S, x = 2)
  (P : ℝ) : 
  -- Assuming two points are chosen independently at random on the sides of a square S of side length 2
  let prob := (26 - Real.pi) / 32 in
    P = prob := 
sorry

end probability_distance_ge_one_l782_782161


namespace find_c_of_quadratic_roots_l782_782884

theorem find_c_of_quadratic_roots (c : ℚ) :
  (∀ x : ℂ, x^2 * (9 : ℂ) + x * (-5 : ℂ) + (c : ℂ) = 0 ↔
    (x = ((-5 : ℂ) + complex.i * complex.sqrt 415) / 18 ∨
     x = ((-5 : ℂ) - complex.i * complex.sqrt 415) / 18)) →
  c = 110 / 9 :=
by
  sorry

end find_c_of_quadratic_roots_l782_782884


namespace no_winning_strategy_l782_782321

noncomputable def probability_of_winning_after_stop (r b : ℕ) : ℚ :=
  r / (r + b : ℚ)

theorem no_winning_strategy (r b : ℕ) (h : r = 26 ∧ b = 26) : 
  ¬ (∃ strategy : (ℕ → Bool) → ℚ, strategy (λ x, true) > 0.5) := 
by
  sorry

end no_winning_strategy_l782_782321


namespace position_of_x21_in_P3_l782_782026

def C_transformation (perm : List ℕ) : List ℕ :=
  let odds := List.filter (fun i => i % 2 = 1) perm
  let evens := List.filter (fun i => i % 2 = 0) perm
  odds ++ evens

def P_i (N : ℕ) (i : ℕ) : List ℕ :=
  if i = 0 then List.range' 1 N
  else
    let prev := P_i N (i - 1)
    let segment_size := N / (2 ^ i)
    List.bind (List.init ((N / segment_size)) (fun k => C_transformation (prev.drop (k * segment_size)).take segment_size))

example : P_i 32 3 = [
    1, 9, 17, 25, 5, 13, 21, 29,
    3, 11, 19, 27, 7, 15, 23, 31,
    2, 10, 18, 26, 6, 14, 22, 30,
    4, 12, 20, 28, 8, 16, 24, 32] :=
by sorry

theorem position_of_x21_in_P3 : (P_i 32 3).indexOf 21 = 6 :=
by sorry

end position_of_x21_in_P3_l782_782026


namespace calc_f_a_f_2a_f_3a_l782_782640

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x < 1 then 2^x - x else
  if h : ∃ k : ℤ, x = k + (x - ↑k) ∧ (0 ≤ x - ↑k ∧ x - ↑k < 1) then
    let ⟨k, ⟨hx_eq, ⟨hx_lb, hx_ub⟩⟩⟩ := h in
    1 - f (x - 1)
  else 0 -- this handles invalid inputs, but should never actually occur under our constraints

lemma f_add_f_succ_eq_one (x : ℝ) : f x + f (x + 1) = 1 :=
sorry

def a : ℝ := Real.log 3 / Real.log 2

theorem calc_f_a_f_2a_f_3a : f a + f (2 * a) + f (3 * a) = (17 : ℚ) / 16 :=
sorry

end calc_f_a_f_2a_f_3a_l782_782640


namespace find_angle_A_find_area_S_l782_782501

-- The angles A, B, C of triangle ABC
variable {A B C : ℝ}
-- The sides of triangle ABC, BC = 7 and AC = 5
variable {BC AC : ℝ}
-- Equivalent condition for angles: 2sin^2(B+C) = sqrt(3)sin(2A)
variable (h₁ : 2 * (Real.sin (B + C))^2 = Real.sqrt 3 * Real.sin (2 * A))

-- Prove A = 60 degrees
theorem find_angle_A (h : A + B + C = Real.pi) : A = Real.pi / 3 := by
  sorry

-- Prove area S of triangle ABC given A = 60 degrees, BC = 7, AC = 5
theorem find_area_S (h₂ : BC = 7) (h₃ : AC = 5) (h₄ : A = Real.pi / 3) : 
  let AB := 8 in -- Solve quadratic equation to find AB = 8
  let S := (1/2) * AB * AC * (Real.sin A) in
  S = 10 * Real.sqrt 3 := by
  sorry

end find_angle_A_find_area_S_l782_782501


namespace moles_of_calcium_hydroxide_l782_782009

theorem moles_of_calcium_hydroxide (CaOH2 CO2 H2O : ℕ) (h1 : CO2 = 2) (h2 : H2O = 2) :
  CaOH2 * H2O = 2 * 2 :=
by
  have h3 : CaOH2 = 2 := sorry
  have h4 : H2O = 2 := sorry
  rw [h3, h4]
  norm_num
  sorry

end moles_of_calcium_hydroxide_l782_782009


namespace complement_intersection_l782_782498

open Set

variable (U P Q : Set ℕ)

def U := {x : ℕ | x < 6}
def P := {2, 4}
def Q := {1, 3, 4, 6}

theorem complement_intersection : (U \ P) ∩ Q = {1, 3} := by
  sorry

end complement_intersection_l782_782498


namespace exists_root_in_interval_l782_782244

theorem exists_root_in_interval {f : ℝ → ℝ} (hf : ∀ x, f x = Real.exp x - x - 2) :
  ∃ (c : ℝ), c ∈ set.Ioo 1 2 ∧ f c = 0 :=
by
  sorry

end exists_root_in_interval_l782_782244


namespace gcd_of_three_numbers_l782_782241

theorem gcd_of_three_numbers (a b c d : ℕ) (ha : a = 72) (hb : b = 120) (hc : c = 168) (hd : d = 24) : 
  Nat.gcd (Nat.gcd a b) c = d :=
by
  rw [ha, hb, hc, hd]
  -- Placeholder for the actual proof
  exact sorry

end gcd_of_three_numbers_l782_782241


namespace pentagon_triangle_area_percentage_l782_782800

theorem pentagon_triangle_area_percentage (s : ℝ) :
  let triangle_area := (sqrt 3 / 4) * s^2
  let square_area := (2 * s)^2
  let pentagon_area := square_area + triangle_area
  (triangle_area / pentagon_area) * 100 ≈ 9.77 :=
by 
  sorry

end pentagon_triangle_area_percentage_l782_782800


namespace modular_expression_problem_l782_782171

theorem modular_expression_problem
  (m : ℕ)
  (hm : 0 ≤ m ∧ m < 29)
  (hmod : 4 * m % 29 = 1) :
  (5^m % 29)^4 - 3 % 29 = 13 % 29 :=
by
  sorry

end modular_expression_problem_l782_782171


namespace area_of_circumcircle_l782_782544

-- Define the context and conditions
variables {a b c S : ℝ} (R : ℝ) (C : ℝ)
def U := ∀ (a b : ℝ) (c : ℝ) (S : ℝ), 
  (a^2 + b^2 - c^2 = 4 * √3 * S) →
  (c = 1) →
  (C = π / 6) →
  (R = 1) →
  (π * R^2 = π)

-- Prove the area of the circumcircle given the conditions
theorem area_of_circumcircle (h1 : a^2 + b^2 - c^2 = 4 * √3 * S)
                            (h2 : c = 1)
                            (hC : C = π / 6)
                            (hR : R = 1):
  π * R^2 = π := 
begin
  -- proof omitted
  sorry
end

end area_of_circumcircle_l782_782544


namespace angle_between_CK_and_AB_l782_782232

theorem angle_between_CK_and_AB 
  (ABC : Type) 
  [triangle ABC] 
  (α β γ δ : ℝ) 
  (circumcenter : point ABC) 
  (hαβγ : α + β + γ = 180) 
  (hα : α > 0 ∧ α < 180) 
  (hβ : β > 0 ∧ β < 180) 
  (hγ : γ > 0 ∧ γ < 180) 
  (hδ : δ = |90 - (α - β)|) : 
  angle_between (line_through circumcenter vertex_C) side_AB = δ :=
sorry

end angle_between_CK_and_AB_l782_782232


namespace minimum_value_of_expression_l782_782877

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) : 6 * x + 1 / x ^ 6 ≥ 7 :=
sorry

end minimum_value_of_expression_l782_782877


namespace integer_part_of_x_l782_782080

noncomputable def x : ℝ := 
  1 + ∑ k in Finset.range 10^6, 1 / Real.sqrt (k + 1)

theorem integer_part_of_x : ⌊x⌋ = 1998 :=
by
  sorry

end integer_part_of_x_l782_782080


namespace number_of_bowls_l782_782724

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l782_782724


namespace distinct_real_numbers_condition_l782_782164

theorem distinct_real_numbers_condition (a b c : ℝ) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a / (b - c)) + (b / (c - a)) + (c / (a - b)) = 1) :
  (a / (b - c)^2) + (b / (c - a)^2) + (c / (a - b)^2) = 1 := 
by sorry

end distinct_real_numbers_condition_l782_782164


namespace mean_and_variance_comparison_l782_782348

-- Define the original heights.
def heights_original : List ℕ := [160, 165, 170, 163, 167]

-- Define the new member height.
def height_new_member : ℕ := 165

-- Calculate the mean of a list of heights.
noncomputable def mean (heights : List ℕ) : ℚ :=
  heights.sum / heights.length

-- Calculate the variance of a list of heights.
noncomputable def variance (heights : List ℕ) : ℚ :=
  let mean_val := mean heights
  heights.map (λ x => (x - mean_val) ^ 2).sum / heights.length

-- Calculate the original mean and variance.
noncomputable def mean_original : ℚ := mean heights_original
noncomputable def variance_original : ℚ := variance heights_original

-- Calculate the new heights list after adding the new member.
def heights_new : List ℕ := height_new_member :: heights_original

-- Calculate the new mean and variance.
noncomputable def mean_new : ℚ := mean heights_new
noncomputable def variance_new : ℚ := variance heights_new

-- Problem statement to prove:
theorem mean_and_variance_comparison :
  mean_original = mean_new ∧ variance_original > variance_new :=
by
  sorry

end mean_and_variance_comparison_l782_782348


namespace shaded_area_l782_782379

-- Definitions based on conditions in a)
def square (s : ℝ) : ℝ := s * s

def half (s : ℝ) : ℝ := s / 2

-- The given area of square ABCD
def area_ABCD : ℝ := 144

-- Problem statement: proving the area of the shaded region
theorem shaded_area (ABCD_area : ℝ) (E F G H : ℝ) 
  (cond1 : ABCD_area = square 12)
  (cond2 : E = half 12 ∧ F = half 12 ∧ G = half 12 ∧ H = half 12)
  : ABCD_area - (square (half 12)) = 108 :=
by
  sorry

end shaded_area_l782_782379


namespace adopted_dogs_count_l782_782371

-- Definitions for conditions
def vet_donation (D C : ℕ) : ℝ := (15 * D + 13 * C) / 3

-- Theorem statement
theorem adopted_dogs_count : 
  ∃ D : ℕ, ∃ C : ℕ, C = 3 ∧ vet_donation D C = 53 ∧ D = 8 := 
by
  sorry -- Proof is not required

end adopted_dogs_count_l782_782371


namespace mutually_exclusive_of_additivity_l782_782771

-- Define events A and B
variables {Ω : Type*} {P : Ω → Prop}

-- Define the probability measure P
variables {μ : Measure Ω}

-- Define mutually exclusive property
def mutually_exclusive (A B : Ω → Prop) : Prop :=
  μ (A ∩ B) = 0

-- Define additivity condition
def additivity (A B : Ω → Prop) : Prop :=
  μ (A ∪ B) = μ A + μ B

-- Given conditions and the property to prove
theorem mutually_exclusive_of_additivity (A B : Ω → Prop) (h : additivity A B) : mutually_exclusive A B :=
begin
  sorry
end

end mutually_exclusive_of_additivity_l782_782771


namespace value_of_a2_l782_782573

theorem value_of_a2 (a0 a1 a2 a3 a4 : ℝ) (x : ℝ) 
  (h : x^4 = a0 + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3 + a4 * (x - 2)^4) :
  a2 = 24 :=
sorry

end value_of_a2_l782_782573


namespace partition_perfect_square_l782_782865

theorem partition_perfect_square (n : ℕ) (h : n ≥ 15) :
  ∀ A B : finset ℕ, disjoint A B → A ∪ B = finset.range (n + 1) →
  ∃ x y ∈ A ∨ ∃ x y ∈ B, x ≠ y ∧ (∃ k : ℕ, x + y = k^2) :=
begin
  sorry
end

end partition_perfect_square_l782_782865


namespace Macau_Math_Olympiad_1993_problem_l782_782176

theorem Macau_Math_Olympiad_1993_problem (x : Fin 1993 → ℝ)
  (h : ∑ i in Finset.range 1992, |x i - x (i + 1)| = 1993)
  (y : Fin 1993 → ℝ := λ k, (∑ i in Finset.range (k + 1), x i) / (k + 1)) :
  ∑ i in Finset.range 1992, |y i - y (i + 1)| ≤ 1992 :=
sorry

end Macau_Math_Olympiad_1993_problem_l782_782176


namespace fish_population_l782_782316

theorem fish_population (x : ℕ) : 
  (1: ℝ) / 45 = (100: ℝ) / ↑x -> x = 1125 :=
by
  sorry

end fish_population_l782_782316


namespace find_a_l782_782238

-- Define the parabola equation y² = 2ax with a > 0
def parabola_equation (a : ℝ) (a_pos : a > 0) : Prop :=
  ∃ (F : ℝ × ℝ), F = (a / 2, 0) 

-- Define the hyperbola equation (y² / 4) - (x² / 9) = 1
def hyperbola_equation (M N : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), (y^2 / 4) - (x^2 / 9) = 1 ∧ M = (-a/2, y) ∧ N = (-a/2, -y)

-- Define the angle condition ∠MFN = 120°
def angle_MFN (M F N : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t/2 = 60 * π / 180 ∧ tan t = sqrt 3 -- Conversion of degrees to radians and tangential relationship

-- Define the overall problem as proving a = 3√26 / 13
theorem find_a (a : ℝ) (a_pos : a > 0) : a = 3 * sqrt 26 / 13 :=
by
  have parabola_conditions := parabola_equation a a_pos
  have hyperbola_conditions := hyperbola_equation ((-a / 2, sqrt (4 + a^2 / 9)) (-a / 2, -sqrt (4 + a^2 / 9)))
  have angle_condition := angle_MFN ((-a / 2, sqrt (4 + a^2 / 9)) ((a / 2, 0)) (-a / 2, -sqrt (4 + a^2 / 9)))
  sorry

end find_a_l782_782238


namespace choose_three_cooks_from_ten_l782_782125

theorem choose_three_cooks_from_ten : 
  (nat.choose 10 3) = 120 := 
by
  sorry

end choose_three_cooks_from_ten_l782_782125


namespace choose_3_out_of_10_l782_782118

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 := by
  sorry

end choose_3_out_of_10_l782_782118


namespace min_distance_proof_l782_782547

-- Define the curve in polar coordinates
def curve_polar (θ : ℝ) : ℝ := 2 / (Real.sqrt (1 + 3 * (Real.sin θ) ^ 2))

-- Define the rectangular form of the curve
def curve_rect (x y : ℝ) : Prop := (x ^ 2) / 4 + y ^ 2 = 1

-- Define the distance function
def distance_to_line (x y : ℝ) : ℝ :=
  (Real.abs (x - 2 * y - 4 * Real.sqrt 2)) / (Real.sqrt 5)

-- Define the minimum distance
def min_distance : ℝ := 2 * Real.sqrt 10 / 5

-- Theorem statement proving the minimum distance
theorem min_distance_proof : ∃ (θ : ℝ), distance_to_line (2 * Real.cos θ) (2 * Real.sin θ) = min_distance := sorry

end min_distance_proof_l782_782547


namespace concurrency_of_three_lines_l782_782219

theorem concurrency_of_three_lines
  (A B C A_1 C_1 A_0 B_0 : Point)
  (triangle_ABC : Triangle A B C)
  (incircle_tangent_BC : TangentPoint incircle A_1 (Side BC))
  (incircle_tangent_BA : TangentPoint incircle C_1 (Side BA))
  (midpoint_A0 : Midpoint A_0 (Segment BC))
  (midpoint_B0 : Midpoint B_0 (Segment AC))
  : Exists (N : Point), LineThrough N (Bisector A (Angle A B C)) ∧ LineThrough N (ParallelThrough A_0 (Line B A)) ∧ LineThrough N (LineThrough A_1 C_1) :=
sorry

end concurrency_of_three_lines_l782_782219


namespace eval_exp_l782_782418

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l782_782418


namespace number_of_bowls_l782_782727

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l782_782727


namespace number_of_bowls_l782_782742

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l782_782742


namespace evaluate_power_l782_782463

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l782_782463


namespace supplemental_tank_time_l782_782383

-- Define the given conditions as assumptions
def primary_tank_time : Nat := 2
def total_time_needed : Nat := 8
def supplemental_tanks : Nat := 6
def additional_time_needed : Nat := total_time_needed - primary_tank_time

-- Define the theorem to prove
theorem supplemental_tank_time :
  additional_time_needed / supplemental_tanks = 1 :=
by
  -- Here we would provide the proof, but it is omitted with "sorry"
  sorry

end supplemental_tank_time_l782_782383


namespace number_of_bowls_l782_782730

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l782_782730


namespace find_a_max_value_b_leq_0_max_value_0_lt_b_leq_1_max_value_1_lt_b_leq_upper_bound_l782_782052

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1/3) * x^3 - 2 * x^2 + 3 * x + a

theorem find_a (a : ℝ) (hx : f 1 a = 2) : a = 2/3 :=
  sorry

theorem max_value_b_leq_0 (a b : ℝ) (hx : a = 2/3) (h : b ≤ 0 ∨ b > (9 + real.sqrt 33) / 6) :
  ∀ x ∈ set.Icc b (b+1), f x a ≤ f (b + 1) a :=
  sorry

theorem max_value_0_lt_b_leq_1 (a b : ℝ) (hx : a = 2/3) (h : 0 < b ∧ b ≤ 1) :
  ∀ x ∈ set.Icc b (b+1), f x a ≤ 2 :=
  sorry

theorem max_value_1_lt_b_leq_upper_bound (a b : ℝ) (hx : a = 2/3) (h : 1 < b ∧ b ≤ (9 + real.sqrt 33) / 6) :
  ∀ x ∈ set.Icc b (b+1), f x a ≤ f b a :=
  sorry

end find_a_max_value_b_leq_0_max_value_0_lt_b_leq_1_max_value_1_lt_b_leq_upper_bound_l782_782052


namespace find_question_mark_l782_782297

theorem find_question_mark (x : ℝ) (h : sqrt x / 15 = 4) : x = 3600 :=
by
  sorry

end find_question_mark_l782_782297


namespace sum_of_solutions_eq_six_l782_782015

theorem sum_of_solutions_eq_six :
  (∑ x in (Finset.filter (λ x, 2^(x^2 - 4*x + 1) = 4^(x - 3)) (Finset.range 100)), x) = 6 :=
by
  -- Here we should provide the proof steps, but for now, we skip it.
  sorry

end sum_of_solutions_eq_six_l782_782015


namespace lock_probability_l782_782522

/-- The probability of correctly guessing the last digit of a three-digit combination lock,
given that the first two digits are correctly set and each digit ranges from 0 to 9. -/
theorem lock_probability : 
  ∀ (d1 d2 : ℕ), 
  (0 ≤ d1 ∧ d1 < 10) ∧ (0 ≤ d2 ∧ d2 < 10) →
  (0 ≤ d3 ∧ d3 < 10) → 
  (1/10 : ℝ) = (1 : ℝ) / (10 : ℝ) :=
by
  sorry

end lock_probability_l782_782522


namespace exists_longest_edge_with_acute_adjacent_angles_l782_782995

structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ
  edges : Fin 6 → Fin 2 → ℝ × ℝ × ℝ

def is_adjacent (e : ℝ × ℝ × ℝ) (angle : ℝ) : Prop :=
  -- Definition of an angle being adjacent to an edge
  sorry

def acute (angle : ℝ) : Prop :=
  angle > 0 ∧ angle < π / 2

theorem exists_longest_edge_with_acute_adjacent_angles (T : Tetrahedron) :
  ∃ e, (∀ α, is_adjacent e α → acute α) :=
by
  -- We need to prove there exists a longest edge in T such that all adjacent angles are acute
  sorry

end exists_longest_edge_with_acute_adjacent_angles_l782_782995


namespace omega_range_monotonically_decreasing_function_l782_782935

theorem omega_range_monotonically_decreasing_function :
  (∀ (x ∈ Ioo (real.pi / 2) real.pi), 
    let f := λ x, (real.sqrt 2) * real.sin (ω * x) * real.cos (ω * x) + 
                    (real.sqrt 2) * (real.cos (ω * x))^2 - (real.sqrt 2) / 2
    (∀ x y, (x < y) → f x ≥ f y)) 
    ↔ (real.sqrt (2:real) * real.sin (ω * x) * real.cos (ω * x) + 
        real.sqrt (2:real) * (real.cos (ω * x)) ^ 2 - (real.sqrt (2:real)) / 2) 
    ∈ [1/4, 5/8] :=
sorry

end omega_range_monotonically_decreasing_function_l782_782935


namespace n_digit_numbers_composition_l782_782952

theorem n_digit_numbers_composition (n : ℕ) : 
  let eligible_numbers := {x : vector ℕ n // ∀ i, x.nth i ∈ {1, 2, 3}} 
  in (eligible_numbers.card = 3^n - 3 * 2^n + 3) :=
sorry

end n_digit_numbers_composition_l782_782952


namespace matrix_product_arithmetic_sequence_l782_782391

open Matrix

def mat (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, n], ![0, 1]]

theorem matrix_product_arithmetic_sequence :
  (List.range' 2 2 50).foldl (λ acc n, acc ⬝ mat n) (1 : Matrix _ _ ℤ) = 
  ![![1, 2550], ![0, 1]] := by
  sorry

end matrix_product_arithmetic_sequence_l782_782391


namespace number_of_bowls_l782_782726

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l782_782726


namespace perfect_square_partition_l782_782873

open Nat

-- Define the condition of a number being a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- Define the main theorem statement
theorem perfect_square_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n+1)) → (A ∩ B = ∅) →
  ∃ a b ∈ A, a ≠ b ∧ is_perfect_square (a + b)
:= by
  sorry

end perfect_square_partition_l782_782873


namespace convert_to_spherical_l782_782830

-- Definitions
def x : ℝ := 1
def y : ℝ := -4
def z : ℝ := 2 * Real.sqrt 3

-- The hypothesis (conditions)
axiom h_x : x = 1
axiom h_y : y = -4
axiom h_z : z = 2 * Real.sqrt 3

-- The proof problem statement in Lean 4
theorem convert_to_spherical :
  let ρ := Real.sqrt (x^2 + y^2 + z^2),
      θ := Real.pi - Real.arctan 4,
      φ := Real.arccos (2 * Real.sqrt 3 / Real.sqrt 29)
  in (ρ, θ, φ) = (Real.sqrt 29, Real.pi - Real.arctan 4, Real.arccos (2 * Real.sqrt 3 / Real.sqrt 29)) :=
by {
  sorry
}

end convert_to_spherical_l782_782830


namespace natural_numbers_partition_l782_782855

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l782_782855


namespace largest_even_integer_sum_l782_782258

theorem largest_even_integer_sum (sum_eq : ∑ i in finset.range 30, (2 * 371) + 2 * i = 12000) :
  429 = (2 * 371 + 58) :=
by
  sorry

end largest_even_integer_sum_l782_782258


namespace degree_measure_correct_l782_782472

noncomputable def find_degree_measure (deg : ℝ) : Prop :=
  deg = Real.arccos (((Finset.range (6142 - 2541)).sum (λ i, Real.sin ((2541 + i) * (Real.pi / 180)))) 
    * Real.cos (2520 * (Real.pi / 180)) + (Finset.range (6121 - 2521)).sum (λ j, Real.cos ((2521 + j) * (Real.pi / 180))))

theorem degree_measure_correct : find_degree_measure 69 :=
by 
  -- Proof to be provided here.
  sorry

end degree_measure_correct_l782_782472


namespace chess_games_played_l782_782302

theorem chess_games_played (n k : ℕ) (h_n : n = 14) (h_k : k = 2) : 
  nat.choose n k = 91 :=
by {
  rw [h_n, h_k],
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)
}

end chess_games_played_l782_782302


namespace width_of_field_l782_782802

-- Defining the lengths and widths
def length_field : ℕ := 60
def width_path : ℝ := 2.5
def area_path : ℝ := 1200
def cost_per_sq_meter : ℝ := 2

-- Defining the width of the field
noncomputable def width_field : ℝ :=
let total_length := length_field + 2 * width_path in
let total_width := λ w : ℝ, w + 2 * width_path in
let area_total := λ w : ℝ, total_length * total_width w in
let area_field := λ w : ℝ, length_field * w in
  if (area_total 175 - area_field 175 = area_path) then 175 else 0 -- This ensures uniqueness and correctness for w = 175

theorem width_of_field : width_field = 175 := by
  -- assuming that all units and calculated operations are correct
  unfold width_field
  have length_correct : total_length = length_field + 2 * width_path := rfl
  have width_correct : total_width 175 = 175 + 2 * width_path := rfl
  have area_correct := calc area_total 175
    = total_length * (175 + 2 * width_path) : by simp [width_correct, total_length]
    ... = 65 * 180 : by simp [total_length, width_correct]
    ... = 11700 : by simp
  have field_correct := calc area_field 175
    = length_field * 175 : by simp [length_field]
    ... = 10500 : by simp
  have path_correct := calc area_total 175 - area_field 175
    = 1200 : by simp [area_correct, field_correct]
  exact if_pos path_correct

end width_of_field_l782_782802


namespace Megan_finish_all_problems_in_8_hours_l782_782489

theorem Megan_finish_all_problems_in_8_hours :
  ∀ (math_problems spelling_problems problems_per_hour : ℕ),
    math_problems = 36 →
    spelling_problems = 28 →
    problems_per_hour = 8 →
    (math_problems + spelling_problems) / problems_per_hour = 8 :=
by
  intros
  sorry

end Megan_finish_all_problems_in_8_hours_l782_782489


namespace trajectory_equation_l782_782042

noncomputable def ellipse_trajectory (x y : ℝ) : Prop :=
  (x^2 / 25) + (y^2 / 16) = 1

def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def satisfies_condition (M : ℝ × ℝ) : Prop :=
  dist M F1 + dist M F2 = 10

theorem trajectory_equation (M : ℝ × ℝ) :
  satisfies_condition M → ellipse_trajectory M.1 M.2 := by
  sorry

end trajectory_equation_l782_782042


namespace incenter_of_triangle_BMC_l782_782683

theorem incenter_of_triangle_BMC
  (A B C D O M : Point)
  (h1 : inscribed_quadrilateral A B C D)
  (h2 : diagonals_intersect_at A C B D O)
  (h3 : circumcircles_intersect_at A O B C O D M (on_side M A D)) :
  incenter O B M C := sorry

end incenter_of_triangle_BMC_l782_782683


namespace graphs_with_inverses_l782_782068

-- Define the types of the functions as per the conditions
def graphA (x : ℝ) : ℝ := -- piecewise linear function
  if x < -1 then x + 6 
  else if x < 1 then -2 * x 
  else x - 6

def graphB (x : ℝ) : ℝ := -- piecewise constant function
  if x < -3 then -3 
  else if x < 3 then 3 
  else 3

def graphC (x : ℝ) : ℝ := x -- straight diagonal line

def graphD (x : ℝ) : ℝ := if abs x <= 4 then sqrt (16 - x^2) else 0 -- semicircle

def graphE (x : ℝ) : ℝ := x^3 / 30 - x -- cubic function

def graphF (x : ℝ) : ℝ := -- V-shaped graph
  if x < 0 then -4 * x 
  else 4 * x

-- Define the theorem stating which graphs have inverses
theorem graphs_with_inverses : 
  (∃ g : ℝ → ℝ, ∀ x, g (graphB x) = x) ∧ 
  (∃ g : ℝ → ℝ, ∀ x, g (graphC x) = x) ∧ 
  (∃ g : ℝ → ℝ, ∀ x, g (graphF x) = x) :=
by
  sorry --proof goes here

end graphs_with_inverses_l782_782068


namespace steven_owes_jeremy_l782_782148

theorem steven_owes_jeremy (rate_per_room rooms_cleaned : ℚ) (h1 : rate_per_room = 11 / 2) (h2 : rooms_cleaned = 7 / 3) :
    (rate_per_room * rooms_cleaned) = 77 / 6 :=
by
  rw [h1, h2]
  norm_num

end steven_owes_jeremy_l782_782148


namespace outdoor_section_width_l782_782804

theorem outdoor_section_width (Length Area Width : ℝ) (h1 : Length = 6) (h2 : Area = 24) : Width = 4 :=
by
  -- We'll use "?" to represent the parts that need to be inferred by the proof assistant. 
  sorry

end outdoor_section_width_l782_782804


namespace term_with_largest_binomial_coefficient_term_with_largest_absolute_value_l782_782605

noncomputable def term_largest_binomial_coefficient (x : ℝ) : ℝ := (8.choose 4) * (sqrt x)^4 * ((-2 / x^2)^4) 
noncomputable def term_largest_absolute_value (x : ℝ) : ℝ := max (8.choose 5 * (-2)^5 * x^(-10) * x^(3/2)) (8.choose 6 * (-2)^6 * x^(-12) * x^(1))

theorem term_with_largest_binomial_coefficient (x : ℝ) (hx : x ≠ 0): 
  term_largest_binomial_coefficient x = (1120 / x^6) :=
  sorry

theorem term_with_largest_absolute_value (x : ℝ) (hx : x ≠ 0):
  term_largest_absolute_value x = max (-1792 * x^(-17/2)) (1792 * x^(-11)) :=
  sorry

end term_with_largest_binomial_coefficient_term_with_largest_absolute_value_l782_782605


namespace sqrt_p_mul_sqrt_q_inv_2_l782_782182

theorem sqrt_p_mul_sqrt_q_inv_2 (p q : ℚ) (hp : p = 4/7) (hq : q = 3/4) :
  (Real.sqrt p) * (Real.sqrt q)⁻² = 8 * Real.sqrt 7 / 21 := 
by
  have pValue : p = 4 / 7 := hp
  have qValue : q = 3 / 4 := hq
  sorry

end sqrt_p_mul_sqrt_q_inv_2_l782_782182


namespace no_winning_strategy_exceeds_half_probability_l782_782328

-- Provided conditions
def well_shuffled_standard_deck : Type := sorry -- Placeholder for the deck type

-- Statement of the problem
theorem no_winning_strategy_exceeds_half_probability :
  ∀ strategy : (well_shuffled_standard_deck → ℕ → bool),
    let r := 26 in -- Assuming a standard deck half red cards (26 red)
    let b := 26 in -- and half black cards (26 black)
    let P_win := (r : ℝ) / (r + b) in       
    P_win ≤ 0.5 :=
by
  sorry

end no_winning_strategy_exceeds_half_probability_l782_782328


namespace cost_difference_per_square_inch_l782_782287

theorem cost_difference_per_square_inch (width1 height1 width2 height2 : ℕ) (cost1 cost2 : ℕ)
  (h_size1 : width1 = 24 ∧ height1 = 16)
  (h_cost1 : cost1 = 672)
  (h_size2 : width2 = 48 ∧ height2 = 32)
  (h_cost2 : cost2 = 1152) :
  (cost1 / (width1 * height1) : ℚ) - (cost2 / (width2 * height2) : ℚ) = 1 := 
by
  sorry

end cost_difference_per_square_inch_l782_782287


namespace correct_expression_l782_782770

noncomputable def checkExpressionA : Prop := sqrt 25 = 5
noncomputable def checkExpressionB : Prop := (sqrt 25 = 5 ∨ sqrt 25 = -5)
noncomputable def checkExpressionC : Prop := sqrt ((-5) * (-5)) = -5
noncomputable def checkExpressionD : Prop := real.cbrt (-125) = -5

theorem correct_expression : ¬checkExpressionA ∧ ¬checkExpressionB ∧ ¬checkExpressionC ∧ checkExpressionD := 
by
  sorry

end correct_expression_l782_782770


namespace min_value_expression_l782_782637

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : y = Real.sqrt x) :
  ∃ c, c = 2 ∧ ∀ u v : ℝ, 0 < u → v = Real.sqrt u → (u^2 + v^4) / (u * v^2) = c :=
by
  sorry

end min_value_expression_l782_782637


namespace sum_of_possible_a_l782_782837

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x + 3 * a

theorem sum_of_possible_a :
  (∑ a in {a : ℝ | ∃ r s : ℝ, f a r = 0 ∧ f a s = 0 ∧ r + s = a ∧ r * s = 3 * a}.toFinset, a) = 24 := by
  sorry

end sum_of_possible_a_l782_782837


namespace number_of_bowls_l782_782713

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l782_782713


namespace impossible_to_form_rectangle_l782_782263

-- Define the problem conditions
def piece := List (ℕ × ℕ) -- A piece is a list of coordinates
def config := List piece -- Config is a list of pieces

-- Assume we have five such pieces
variable (P1 P2 P3 P4 P5 : piece)
variable (pieces : config := [P1, P2, P3, P4, P5])

-- Condition: each piece composed of 4 squares
def valid_piece (p : piece) : Prop := p.length = 4

-- Condition: the 4x5 rectangle
def rectangle4x5 : List (ℕ × ℕ) := 
  [(x, y) | x in [0, 1, 2, 3], y in [0, 1, 2, 3, 4]] -- 4 rows, 5 columns

-- The main theorem
theorem impossible_to_form_rectangle : 
  (∀ p ∈ pieces, valid_piece p) → 
  (∀ p₁ p₂ ∈ pieces, p₁ ≠ p₂ → disjoint p₁ p₂) → 
  (∪ pieces = rectangle4x5) → False := by
 sorry

end impossible_to_form_rectangle_l782_782263


namespace tangent_slope_positive_l782_782513

-- Define the function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the point (2, 3) on the curve y = f(x)
def P1 : ℝ × ℝ := (2, f 2)

-- Define the point (-1, 2) that the tangent line passes through
def P2 : ℝ × ℝ := (-1, 2)

-- Define the condition that the tangent line at x=2 passes through the point (-1, 2)
def tangent_condition : Prop :=
  let k := (2 - f 2) / (-1 - 2) in 
  f'(2) = k

-- The theorem to prove: the slope of the tangent line at (2, f 2) is positive
theorem tangent_slope_positive 
  (h : tangent_condition f) : f'(2) > 0 :=
sorry

end tangent_slope_positive_l782_782513


namespace difference_in_girls_and_boys_l782_782990

-- Given conditions as definitions
def boys : ℕ := 40
def ratio_boys_to_girls (b g : ℕ) : Prop := 5 * g = 13 * b

-- Statement of the problem
theorem difference_in_girls_and_boys (g : ℕ) (h : ratio_boys_to_girls boys g) : g - boys = 64 :=
by
  sorry

end difference_in_girls_and_boys_l782_782990


namespace jason_borrowed_amount_l782_782619

theorem jason_borrowed_amount : 
  (∑ i in range 9, (i + 1)) * (27 / 9) = 135 :=
by sorry

end jason_borrowed_amount_l782_782619


namespace evaluate_power_l782_782464

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l782_782464


namespace sum_series_x_2008_l782_782897

theorem sum_series_x_2008 (x : ℂ) (h : 1 + x + x^2 + x^3 = 0) : (∑ i in Finset.range 2009, x^i) = 1 :=
by
  sorry

end sum_series_x_2008_l782_782897


namespace logical_word_used_is_or_l782_782252

-- Defining the absolute value equation condition
def abs_eq_one (x : ℝ) : Prop := |x| = 1

-- Proposition stating the solutions to the equation
def solutions_to_abs_eq_one : Prop :=
  ∀ x : ℝ, abs_eq_one x ↔ (x = 1 ∨ x = -1)

-- The corresponding logical connective used in the proposition
def logical_connective_used_is_or : Prop :=
  (λ P Q : Prop, P ∨ Q) = solutions_to_abs_eq_one

-- Main theorem stating the condition
theorem logical_word_used_is_or : logical_connective_used_is_or :=
sorry

end logical_word_used_is_or_l782_782252


namespace number_of_bowls_l782_782720

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l782_782720


namespace bleach_contains_chlorine_l782_782005

noncomputable def element_in_bleach (mass_percentage : ℝ) (substance : String) : String :=
  if mass_percentage = 31.08 ∧ substance = "sodium hypochlorite" then "Chlorine"
  else "unknown"

theorem bleach_contains_chlorine : element_in_bleach 31.08 "sodium hypochlorite" = "Chlorine" :=
by
  sorry

end bleach_contains_chlorine_l782_782005


namespace tan_b_range_max_f_l782_782311

-- Part 1
theorem tan_b_range (A B C : ℝ) (h1 : 2 * log (tan B) = log (tan A) + log (tan C)) (h2 : A + B + C = π) :
  B ∈ set.Ico (π / 3) (π / 2) :=
sorry 

-- Part 2
def f (x : ℝ) := 7 - 4 * sin x * cos x + 4 * cos x ^ 2 - 4 * (cos x) ^ 4

theorem max_f : (∀ x, f x ≤ 10) ∧ ∃ x, f x = 10 :=
sorry

end tan_b_range_max_f_l782_782311


namespace arc_mtn_range_l782_782997

theorem arc_mtn_range (ABC : Triangle) (B : Point) (C : Point) (A : Point)
  (h1 : right_triangle ABC)
  (h2 : ∠BAC = 90)
  (h3 : ∠ABC = 30)
  (Circle : circle)
  (r : ℝ)
  (h4 : r = (1 / 2) * (length (segment B C)))
  (h5 : Circle.radius = r)
  (h6 : Circle.tangent_to (segment A B))
  (T : Point)
  (M : Point)
  (N : Point)
  (h7 : T ∈ (segment A B))
  (h8 : M ∈ (intersection Circle C))
  (h9 : N ∈ (intersection Circle C)) :
  (0 ≤ arc MTN.degrees) ∧ (arc MTN.degrees ≤ 180) :=
sorry

end arc_mtn_range_l782_782997


namespace max_volume_of_pyramid_l782_782608

theorem max_volume_of_pyramid
  (a b c : ℝ)
  (h1 : a + b + c = 9)
  (h2 : ∀ (α β : ℝ), α = 30 ∧ β = 45)
  : ∃ V, V = (9 * Real.sqrt 2) / 4 ∧ V = (1/6) * (Real.sqrt 2 / 2) * a * b * c :=
by
  sorry

end max_volume_of_pyramid_l782_782608


namespace no_positive_integers_satisfying_inequality_l782_782491

def positive_integers := {x : ℕ // x > 0}

theorem no_positive_integers_satisfying_inequality : ∀ x : positive_integers, 
  (30 < x.val) → (x.val < 90) → (log10 (x.val - 30) + log10 (90 - x.val) < 1.5) → false :=
by
  intro x h1 h2 h3
  -- Here is where the proof steps would go
  sorry

end no_positive_integers_satisfying_inequality_l782_782491


namespace total_percentage_of_women_l782_782378

theorem total_percentage_of_women
    (initial_employees : ℕ)
    (men_fraction : ℚ)
    (new_women : ℕ)
    (initial_employees = 90)
    (men_fraction = 2/3)
    (new_women = 10) :
  let women_fraction := 1 - men_fraction in
  let initial_women := women_fraction * initial_employees in
  let total_women := initial_women + new_women in
  let total_employees := initial_employees + new_women in
  let percentage_of_women := (total_women / total_employees) * 100 in
  percentage_of_women = 40 :=
by
  sorry

end total_percentage_of_women_l782_782378


namespace repeating_decimal_difference_l782_782631

theorem repeating_decimal_difference :
  let G := 0.726726726 in
  let frac_726_999 := (726 : ℚ) / 999 in
  let G_frac := frac_726_999 in
  let simplest_form_frac := (242 : ℚ) / 333 in
  let numerator := 242 in
  let denominator := 333 in
  numerator - denominator = -91 := by
sorry

end repeating_decimal_difference_l782_782631


namespace multiple_of_bees_l782_782196

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end multiple_of_bees_l782_782196


namespace number_of_bowls_l782_782712

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l782_782712


namespace Euler_line_l782_782303

-- Definitions based on given conditions
variables {A B C O G H D E F : Type*}
variables (Triangle : Type*) (Medians : Type*) (Altitudes : Type*)
variables [InCircle A B C] [Centeroid G A B C] [OrthoCenter H A B C]
variables [MidPoints A' B' C'] [FooterPoints D E F]

-- Conditions and questions in Lean statement
theorem Euler_line
  (triangle_condition : Triangle ABC)
  (circumcenter_condition : Circumcenter O ABC)
  (centroid_condition : Centroid G ABC)
  (orthocenter_condition : Orthocenter H ABC)
  (midpoints_condition : Midpoints A' B' C' ABC)
  (feet_of_altitudes_condition : FeetOfAltitudes D E F ABC):
  -- Show G is at 2/3 way along medians
  splits_medians_in_2_1_ratio G A' B' C' ∧ 
  homothety_centered_at_G G ABC A'B'C' ∧ 
  collinear O G H :=
begin
  sorry,
end

-- Helper statements to capture the essence of the given problem
class InCircle (A B C : Type*) := (circ_center : O)
class Centeroid (G A B C : Type*) := (med_split_2_1 : Prop)
class OrthoCenter (H A B C : Type*) := (orthocenter_property : Prop)
class MidPoints (A' B' C' : Type*) := (mid_points : Prop)
class FooterPoints (D E F : Type*) := (footer_points : Prop)

variables {ABC : Triangle} 

-- Define that the centroid splits the medians in a 2:1 ratio
def splits_medians_in_2_1_ratio (G : Type*) (A' B' C' : Type*) := ∀ (M : Type*), G.1 / 3 = M

-- Define the homothety centered at G
def homothety_centered_at_G (G : Type*) (ABC : Triangle) (A'B'C' : Type*) := ∀ (T : Triangle), T.1 = T.2 / 2

-- Define collinearity of O, G, H
def collinear (O G H : Type*) := ∃ (X Y : Type*), O.1 + X.1 * G.1 + 2 = Y.1 

end Euler_line_l782_782303


namespace δ_can_be_arbitrarily_small_l782_782841

-- Define δ(r) as the distance from the circle to the nearest point with integer coordinates.
def δ (r : ℝ) : ℝ := sorry -- exact definition would depend on the implementation details

-- The main theorem to be proven.
theorem δ_can_be_arbitrarily_small (ε : ℝ) (hε : ε > 0) : ∃ r : ℝ, r > 0 ∧ δ r < ε :=
sorry

end δ_can_be_arbitrarily_small_l782_782841


namespace chips_placement_l782_782779

theorem chips_placement (grid : fin 6 → fin 6) :
  ∃! placements : finset (fin 6 × fin 6), 
    placements.card = 4 ∧ 
    (∀ sq : fin 6 × fin 6, ∃ chip ∈ placements, 
      (chip.fst = sq.fst ∨ chip.snd = sq.snd ∨ 
       (chip.fst - chip.snd = sq.fst - sq.snd ∨
        chip.fst + chip.snd = sq.fst + sq.snd))) ∧ 
    (\@finset.sizeof (fin 6 × fin 6) _ placements = 48) :=
by
  sorry

end chips_placement_l782_782779


namespace sequence_sum_formula_l782_782034

def sequence_sum (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / (4 * (i + 1)^2 - 1)

theorem sequence_sum_formula (n : ℕ) (h₁ : a_2 = 12) 
  (h₂ : ∀ (k : ℝ), S_n = k * n^2 - 1) (h₃ : k = 4) : 
  sequence_sum n = n / (2 * n + 1) :=
  sorry

end sequence_sum_formula_l782_782034


namespace number_of_bowls_l782_782741

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l782_782741


namespace problem1_problem2_l782_782930

-- Definition of the function and given conditions for problem 1
def f (a b c x : ℝ) := (a * x^2 + b * x + c) * Real.exp x

-- Problem 1
theorem problem1 (a b c : ℝ) (h0 : f a b c 0 = 1) (h1 : f a b c 1 = 0) :
  (∀ x y ∈ set.Icc 0 1, x < y → f a b c x ≥ f a b c y) → a ∈ set.Icc 0 1 := sorry

-- Definition of the function and given conditions for problem 2
def f_a0 (b c x : ℝ) := (b * x + c) * Real.exp x

-- Problem 2
theorem problem2 (b c : ℝ) (m : ℝ)
  (h0 : f_a0 b c 0 = 1) (h1 : f_a0 b c 1 = 0)
  (h2 : ∀ x : ℝ, 2 * f_a0 b c x + 4 * x * Real.exp x ≥ m * x + 1 ∧ m * x + 1 ≥ -x^2 + 4*x + 1) :
  m = 4 := sorry

end problem1_problem2_l782_782930


namespace find_b_condition_l782_782752

theorem find_b_condition (b : ℚ) :
  let v1 := ⟨4, -9⟩
  let v2 := ⟨b, 3⟩
  v1.1 * v2.1 + v1.2 * v2.2 = 0 →
  b = 27 / 4 :=
sorry

end find_b_condition_l782_782752


namespace original_price_correct_l782_782894

def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36
def number_of_tires : ℕ := 4
def saving_per_tire : ℝ := total_savings / number_of_tires
def original_price_per_tire : ℝ := sale_price_per_tire + saving_per_tire

theorem original_price_correct :
  original_price_per_tire = 84 :=
by
  sorry

end original_price_correct_l782_782894


namespace cos_B_eq_1_over_12_l782_782908

-- Given conditions
variables {A B C : ℝ} -- Internal angles of triangle ABC
variables {GA GB GC : ℝ} -- Distances from centroid G to vertices A, B, and C respectively
variables {a b c : ℝ} -- Sides opposite to angles A, B, and C respectively

-- Given vectors sum to zero
axiom vec_condition : 2 * sin A * GA + sqrt 3 * sin B * GB + 3 * sin C * GC = 0

-- The conclusion we need to prove
theorem cos_B_eq_1_over_12 (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hG : GA = 1 / 3 * (a + b + c))
    (cond1 : 2 * a = sqrt 3 * b) 
    (cond2 : sqrt 3 * b = 3 * c) : 
  cos B = 1 / 12 :=
sorry

end cos_B_eq_1_over_12_l782_782908


namespace find_y_l782_782229

theorem find_y (y : ℚ) (h : Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9) : y = 68 / 3 := 
by
  sorry

end find_y_l782_782229


namespace min_sum_areas_of_squares_l782_782588

noncomputable def right_triangle := triangle (pt.mk 0 0) (pt.mk 10 0) (pt.mk 0 15)

def angle_C_is_right (ABC : right_triangle) : Prop := ∠ (pt.mk 0 0) (pt.mk 10 0) (pt.mk 0 15) = 90

def BC_value (ABC : right_triangle) : Prop := dist (pt.mk 0 0) (pt.mk 10 0) = 10

def AC_value (ABC : right_triangle) : Prop := dist (pt.mk 0 0) (pt.mk 0 15) = 15

theorem min_sum_areas_of_squares (ABC : right_triangle) (h1 : angle_C_is_right ABC) (h2 : BC_value ABC) (h3 : AC_value ABC) :
    ∃ s₁ s₂ : ℝ, (s₁ >= 0) ∧ (s₂ >= 0) ∧ (s₁^2 + s₂^2 = 36) := sorry

end min_sum_areas_of_squares_l782_782588


namespace find_angle_C_find_side_c_l782_782948

noncomputable section

-- Definitions and conditions for Part 1
def vectors_dot_product_sin_2C (A B C : ℝ) (m : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  m = (Real.sin A, Real.cos A) ∧ n = (Real.cos B, Real.sin B) ∧ 
  ((m.1 * n.1 + m.2 * n.2) = Real.sin (2 * C))

def angles_of_triangle (A B C : ℝ) : Prop := 
  A + B + C = Real.pi

theorem find_angle_C (A B C : ℝ) (m n : ℝ × ℝ) :
  vectors_dot_product_sin_2C A B C m n → angles_of_triangle A B C → C = Real.pi / 3 :=
sorry

-- Definitions and conditions for Part 2
def sin_in_arithmetic_sequence (x y z : ℝ) : Prop :=
  x + z = 2 * y

def product_of_sides_cos_C (a b c : ℝ) (C : ℝ) : Prop :=
  (a * b * Real.cos C = 18) ∧ (Real.cos C = 1 / 2)

theorem find_side_c (A B C a b c : ℝ) (m n : ℝ × ℝ) :
  sin_in_arithmetic_sequence (Real.sin A) (Real.sin C) (Real.sin B) → 
  angles_of_triangle A B C → 
  product_of_sides_cos_C a b c C → 
  C = Real.pi / 3 → 
  c = 6 :=
sorry

end find_angle_C_find_side_c_l782_782948


namespace coprime_count_l782_782958

theorem coprime_count (n : ℕ) (h : n = 56700000) : 
  ∃ m, m = 12960000 ∧ ∀ i < n, Nat.gcd i n = 1 → i < m :=
by
  sorry

end coprime_count_l782_782958


namespace problem1_problem2_problem3_l782_782922

variable {x : ℝ}

-- Problem 1: Explicit expression for f(x)
def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem problem1 (b : ℝ) (a : ℝ)
  (h1 : b * Real.exp 1 - a = e - 1)
  (h2 : b * Real.exp 1 = e) :
  f x = Real.exp x - Real.log x :=
sorry

-- Problem 2: Number of zeros of the derivative y = f'(x)
def f' (x : ℝ) : ℝ := Real.exp x - 1 / x

theorem problem2 : ∃! x : ℝ, 0 < x ∧ f' x = 0 :=
sorry

-- Problem 3: Prove that f(x) > 2
theorem problem3 : ∀ x > 0, f x > 2 :=
sorry

end problem1_problem2_problem3_l782_782922


namespace sum_of_squares_eq_2_l782_782966

theorem sum_of_squares_eq_2 (a b : ℝ) 
  (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 :=
by sorry

end sum_of_squares_eq_2_l782_782966


namespace angle_e1_e2_f_properties_l782_782971

variables {R : Type*} [LinearOrder R] [Field R] [NormedSpace ℝ R] [RealVectorSpace ℝ R] {x : ℝ} {n : ℕ+}

-- Define the unit vectors and their conditions
variables (e₁ e₂ e₃ : R)
variables (h1 : ∥e₁∥ = 1) (h2 : ∥e₂∥ = 1) (h3 : ∥e₃∥ = 1)
variables (h4 : e₁ + e₂ + e₃ = 0)
variables (a : R := x • e₁ + (n / x) • e₂ + (x + n / x) • e₃) 

-- Prove the angle between e₁ and e₂ is 2π/3
theorem angle_e1_e2 : 
  real.angle e₁ e₂ = 2 * real.pi / 3 :=
sorry

-- Define f and show its properties
noncomputable def f (x : ℝ) : ℝ := ∥a∥

-- Prove critical points and minimum of f(x)
theorem f_properties :
  ∃ x_min : ℝ, f x_min = sqrt n ∧ 
  (∀ x, |x| = sqrt n → f x = sqrt n) :=
sorry

end angle_e1_e2_f_properties_l782_782971


namespace train_length_is_150_meters_l782_782354

def train_speed_kmph : ℝ := 68
def man_speed_kmph : ℝ := 8
def passing_time_sec : ℝ := 8.999280057595392

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph - man_speed_kmph
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * passing_time_sec

theorem train_length_is_150_meters (train_speed_kmph man_speed_kmph passing_time_sec : ℝ) :
  train_speed_kmph = 68 → man_speed_kmph = 8 → passing_time_sec = 8.999280057595392 →
  length_of_train = 150 :=
by
  intros h1 h2 h3
  simp [length_of_train, h1, h2, h3]
  sorry

end train_length_is_150_meters_l782_782354


namespace sin_value_of_angle_l782_782496

theorem sin_value_of_angle (α : ℝ) (h1 : - real.pi / 6 < α) (h2 : α < real.pi / 6)
    (h3 : real.cos (α + real.pi / 6) = 4 / 5) :
    real.sin (2 * α + real.pi / 12) = 17 * real.sqrt 2 / 50 :=
by
  sorry

end sin_value_of_angle_l782_782496


namespace line_through_intersection_perpendicular_l782_782473

theorem line_through_intersection_perpendicular :
  ∃ (a b c : ℝ), 
  (a = 2) ∧ (b = 3) ∧ (c = -2) ∧
  (∀ (x y : ℝ), 2 * x - 3 * y + 10 = 0 ∧ 3 * x + 4 * y - 2 = 0 → a * x + b * y + c = 0) ∧
  (∀ (x0 y0 : ℝ), a * x0 + b * y0 + c = 0 → 3 * x0 - 2 * y0 + 4 ≠ 0) :=
begin
  use [2, 3, -2],
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  split,
  {
    intros x y h,
    sorry
  },
  {
    intros x0 y0 h,
    sorry
  }
end

end line_through_intersection_perpendicular_l782_782473


namespace number_of_bowls_l782_782722

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l782_782722


namespace jon_website_hours_per_day_l782_782152

theorem jon_website_hours_per_day :
  ∀ (dollars_per_visit visits_per_hour total_earnings days_in_month visits_per_day earnings_per_day : ℝ),
  dollars_per_visit = 0.10 → visits_per_hour = 50 → total_earnings = 3600 → days_in_month = 30 →
  visits_per_day = (total_earnings / dollars_per_visit) / days_in_month →
  earnings_per_day = visits_per_day * dollars_per_visit →
  (earnings_per_day / dollars_per_visit) / visits_per_hour = 24 := 
by
  intros dollars_per_visit visits_per_hour total_earnings days_in_month visits_per_day earnings_per_day
  assume h1 : dollars_per_visit = 0.10
  assume h2 : visits_per_hour = 50
  assume h3 : total_earnings = 3600
  assume h4 : days_in_month = 30
  assume h5 : visits_per_day = (total_earnings / dollars_per_visit) / days_in_month
  assume h6 : earnings_per_day = visits_per_day * dollars_per_visit
  sorry

end jon_website_hours_per_day_l782_782152


namespace find_number_l782_782766

-- Define the unknown number
variable (x : ℝ)

-- Define the components described in the conditions
def one_fifth_of_one_fourth  := (1/5) * (1/4) * x
def five_percent            := (5/100) * x
def one_third_minus_one_seventh := (1/3) * x - (1/7) * x
def one_tenth_minus_twelve  := (1/10) * x - 12

-- Relate all conditions to reach the final question
theorem find_number : 
  (one_fifth_of_one_fourth - five_percent) + one_third_minus_one_seventh = one_tenth_minus_twelve 
  → x = -132
  := sorry

end find_number_l782_782766


namespace integer_part_of_x_l782_782081

noncomputable def x : ℝ := 1 + ∑ k in (finset.range 10^6).filter (λ k, k ≥ 2), (1 / real.sqrt k)

theorem integer_part_of_x : real.floor x = 1998 := 
by
  sorry

end integer_part_of_x_l782_782081


namespace perfect_square_partition_l782_782870

open Nat

-- Define the condition of a number being a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- Define the main theorem statement
theorem perfect_square_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n+1)) → (A ∩ B = ∅) →
  ∃ a b ∈ A, a ≠ b ∧ is_perfect_square (a + b)
:= by
  sorry

end perfect_square_partition_l782_782870


namespace replace_batteries_in_December_16_years_later_l782_782069

theorem replace_batteries_in_December_16_years_later :
  ∀ (n : ℕ), n = 30 → ∃ (years : ℕ) (months : ℕ), years = 16 ∧ months = 11 :=
by
  sorry

end replace_batteries_in_December_16_years_later_l782_782069


namespace isosceles_triangle_sides_l782_782592

theorem isosceles_triangle_sides (A B C : Type) [plus : A → A → A] [le : A → A → Prop] [mul : A → A → A] [div : A → A → A] [zero : A] [one : A]
  (triangle_iso : ∀ (x y z : A), plus x y z = 60 → le (plus x z) y → le (plus y z) x → le (plus x y) z → x = y)
  (per_60 : plus (plus A B) C = 60)
  (medians_inter_inscribed : A → A → A → A)
  (centroid_on_incircle : (medians_inter_inscribed A B C) = Inscribed_circle)
  (A : A)
  (B : A)
  (C : A) : 
  A = 25 ∧ B = 25 ∧ C = 10 := 
sorry

end isosceles_triangle_sides_l782_782592


namespace derivative_at_zero_l782_782938

def f (x : ℝ) : ℝ := Real.exp (2 * x + 1) - 3 * x

theorem derivative_at_zero (e : ℝ) : 
  let f' (x : ℝ) : ℝ := 2 * Real.exp (2 * x + 1) - 3
  f' 0 = 2 * Real.exp 1 - 3 := 
sorry

end derivative_at_zero_l782_782938


namespace max_volume_height_l782_782947

noncomputable def height_at_max_volume (S : ℝ) : ℝ :=
  let R := sqrt (S / (6 * real.pi))
  in (sqrt (6 * real.pi * S)) / (3 * real.pi)

theorem max_volume_height (S : ℝ) (hS : 0 < S) : 
  (H : ℝ) (R : ℝ) (V : ℝ) (1 : ℝ) :
    (H = (S / (2 * real.pi * R)) - R)
    → (R = sqrt(S / (6 * real.pi)))
    → (V = (pi * R^2 * H) = (S * R / 2 - pi * R^3)) 
    → H = height_at_max_volume S :=
by
  sorry

end max_volume_height_l782_782947


namespace min_value_inequality_l782_782507

theorem min_value_inequality (a b : ℝ) (h : a * b = 1) : 4 * a^2 + 9 * b^2 ≥ 12 :=
by sorry

end min_value_inequality_l782_782507


namespace number_of_bowls_l782_782736

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l782_782736


namespace remainder_of_multiple_of_n_mod_7_l782_782981

theorem remainder_of_multiple_of_n_mod_7
  (n m : ℤ)
  (h1 : n % 7 = 1)
  (h2 : m % 7 = 3) :
  (m * n) % 7 = 3 :=
by
  sorry

end remainder_of_multiple_of_n_mod_7_l782_782981


namespace maci_blue_pens_l782_782193

theorem maci_blue_pens :
  ∃ (B : ℕ),
  (∀ (red_pens : ℕ) (blue_pen_cost red_pen_cost total_cost : ℝ),
    red_pens = 15 ∧ blue_pen_cost = 0.10 ∧ red_pen_cost = 2 * blue_pen_cost ∧ total_cost = 4 ∧
    total_cost = red_pens * red_pen_cost + B * blue_pen_cost) → B = 10 :=
begin
  sorry
end

end maci_blue_pens_l782_782193


namespace fold_paper_crease_length_l782_782017

theorem fold_paper_crease_length 
    (w l : ℝ) (w_pos : w = 12) (l_pos : l = 16) 
    (F G : ℝ × ℝ) (F_on_AD : F = (0, 12))
    (G_on_BC : G = (16, 12)) :
    dist F G = 20 := 
by
  sorry

end fold_paper_crease_length_l782_782017


namespace Dean_shorter_than_Ron_l782_782364

def Dean_height (water_depth : ℝ) : ℝ := water_depth / 2
-- Given conditions
def water_depth := 12
def Ron_height := 14

-- Given Dean's height
def Dean_height_value := Dean_height water_depth

-- Prove the difference in height
theorem Dean_shorter_than_Ron : (Ron_height - Dean_height_value) = 8 := by
  simp [Ron_height, Dean_height_value, Dean_height, water_depth]
  sorry

end Dean_shorter_than_Ron_l782_782364


namespace find_b_find_C_l782_782064

-- Define the given conditions
variables (A B C : ℝ) (a b c : ℝ)
hypothesis ha : a = 4
hypothesis hc : c = Real.sqrt 13
hypothesis hsinA : Real.sin A = 4 * Real.sin B

-- Proof problem to find b
theorem find_b (ha : a = 4) (hsinA : Real.sin A = 4 * Real.sin B)
  (hb : b = 1) : b = 1 :=
by
  sorry

-- Proof problem to find angle C
theorem find_C (ha : a = 4) (hb : b = 1) (hc : c = Real.sqrt 13) 
  (hcosC : Real.cos C = (a * a + b * b - c * c) / (2 * a * b))
  (hC : C = Real.pi / 3) : C = Real.pi / 3 :=
by
  sorry

end find_b_find_C_l782_782064


namespace shop_owner_pricing_l782_782809

theorem shop_owner_pricing (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : S = 1.3 * C)
  (h3 : S = 0.75 * M) : 
  M = 1.3 * L := 
sorry

end shop_owner_pricing_l782_782809


namespace triangle_area_values_count_l782_782415

/-
Distinct points \(A\), \(B\), \(C\), \(D\), and \(G\) lie on a line, with \(AB = BC = CD = 1\) and \(DG = 2\). 
Points \(E\) and \(F\) lie on another line, parallel to the first, with \(EF = 2.5\). 
Line \(HE\) is perpendicular to both lines containing \(A\), \(B\), \(C\), \(D\), \(E\), \(F\), and \(G\).
\(HE = 1\). 
A triangle with positive area must use three of the eight points (A, B, C, D, G, E, F, H) as vertices.
-/

noncomputable def number_of_possible_triangle_areas : ℕ := 4

theorem triangle_area_values_count :
  let points : Set (ℝ × ℝ) := {
    (0,0), (1,0), (2,0), (3,0), (5,0), -- Points A, B, C, D, G
    (0,2.5), (2.5,2.5), -- Points E, F
    (0,1) -- Point H, projected onto y-axis
  },
  bases := {1, 2, 3, 2.5},
  height := 1 in
  {0.5 * base * height | base ∈ bases}.card = number_of_possible_triangle_areas := by
sorry

end triangle_area_values_count_l782_782415


namespace vector_subtraction_l782_782566

def a : ℝ × ℝ × ℝ := (5, -3, 2)
def b : ℝ × ℝ × ℝ := (-1, 4, -2)

theorem vector_subtraction : 
  let four_b := (4 * -1, 4 * 4, 4 * -2) in
  let result := (5 + 4, -3 - 16, 2 + 8) in
  a - 4 * b = result := 
sorry

end vector_subtraction_l782_782566


namespace arithmetic_sequence_sum_l782_782907

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d) 
  (h2 : ∀ n, S_n n = n * (a 0 + a n) / 2) 
  (h3 : 2 * a 6 = 5 + a 8) :
  S_n 9 = 45 := 
by 
  sorry

end arithmetic_sequence_sum_l782_782907


namespace find_h_l782_782969

theorem find_h (h : ℤ) (root_condition : (-3)^3 + h * (-3) - 18 = 0) : h = -15 :=
by
  sorry

end find_h_l782_782969


namespace partition_with_sum_square_l782_782850

def sum_is_square (a b : ℕ) : Prop := ∃ k : ℕ, a + b = k * k

theorem partition_with_sum_square (n : ℕ) (h : n ≥ 15) :
  ∀ (s₁ s₂ : finset ℕ), (∅ ⊂ s₁ ∪ s₂ ∧ s₁ ∩ s₂ = ∅ ∧ (∀ x ∈ s₁ ∪ s₂, x ∈ finset.range (n + 1))) →
  (∃ a b : ℕ, a ≠ b ∧ (a ∈ s₁ ∧ b ∈ s₁ ∨ a ∈ s₂ ∧ b ∈ s₂) ∧ sum_is_square a b) :=
by sorry

end partition_with_sum_square_l782_782850


namespace integer_part_of_x_l782_782079

noncomputable def x : ℝ := 
  1 + ∑ k in Finset.range 10^6, 1 / Real.sqrt (k + 1)

theorem integer_part_of_x : ⌊x⌋ = 1998 :=
by
  sorry

end integer_part_of_x_l782_782079


namespace triangle_circumscribed_circle_diameter_l782_782084

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem triangle_circumscribed_circle_diameter :
  let a := 16
  let A := Real.pi / 4   -- 45 degrees in radians
  circumscribed_circle_diameter a A = 16 * Real.sqrt 2 :=
by
  sorry

end triangle_circumscribed_circle_diameter_l782_782084


namespace tangency_point_of_parabolas_l782_782010

theorem tangency_point_of_parabolas :
  ∃ (x y : ℝ), y = x^2 + 17 * x + 40 ∧ x = y^2 + 51 * y + 650 ∧ x = -7 ∧ y = -25 :=
by
  sorry

end tangency_point_of_parabolas_l782_782010


namespace same_incenter_l782_782639

open EuclideanGeometry

-- Define the points A, B, C, and their reflections B' and C'
variables (A B C B' C' : Point)
-- Define the angle bisector of ∠A
variable (l : LineBisector (∠ A B C))

-- Reflections definitions
def B' := Reflection B l
def C' := Reflection C l

-- Assume the points B and C have been reflected appropriately
axiom ABC_reflection : Triangle ABC ∧ B' = Reflection B l ∧ C' = Reflection C l

-- Define the incenter of triangle ABC
noncomputable def incenter_ABC : Point := incenter A B C

-- Define the incenter of triangle AB'C'
noncomputable def incenter_AB'C' : Point := incenter A B' C'

-- The goal: showing that the incenter of ABC is the same as the incenter of AB'C'
theorem same_incenter (h : ABC_reflection) : incenter_ABC A B C = incenter_AB'C' A B' C' :=
sorry

end same_incenter_l782_782639


namespace total_seating_arrangements_l782_782890

-- Definitions for the problem conditions
def pairs : Type := Fin 4
def individuals : Type := Fin 8
def seats : Type := Fin 12

structure BusSeating :=
  (rows : pairs → Fin 3) -- function defining the row assignment for each brother-sister pair
  (cols : individuals → Fin 4) -- function defining the column assignment for each individual
  -- Constraints ensuring valid seating arrangements
  (no_adjacent_in_row : ∀ (p : pairs), (cols (p * 2) ≠ (cols (p * 2 + 1))))
  (no_directly_in_front_or_behind : ∀ (i : individuals), rows (i / 2) ≠ rows (i / 2) + 1)

-- The main theorem to prove
theorem total_seating_arrangements : ∃ (s : BusSeating), 1944 arrangements valid :=
sorry

end total_seating_arrangements_l782_782890


namespace sum_digits_n_l782_782482

theorem sum_digits_n (n : ℕ) (h: (n + 1)! + 2 * (n + 2)! = n! * 871) : 
  (n = 19) → (1 + 9 = 10) := by
    sorry

end sum_digits_n_l782_782482


namespace gabe_is_in_seat_2_l782_782843

-- Define the seats
def Seat := {1, 2, 3, 4, 5}

-- Define the people
inductive Person
| Ella | Fiona | Gabe | Harry | Ivan

open Person

-- Conditions as stated in the problem
def fiona_in_seat_3 (fiona_seat : Seat) : Prop :=
  fiona_seat = 3

def fiona_next_to_gabe (fiona_seat gabe_seat : Seat) : Prop :=
  gabe_seat = fiona_seat + 1 ∨ gabe_seat = fiona_seat - 1

def ella_not_between_fiona_and_gabe (ella_seat fiona_seat gabe_seat : Seat) : Prop :=
  ¬ ((ella_seat < fiona_seat ∧ ella_seat > gabe_seat) ∨ (ella_seat > fiona_seat ∧ ella_seat < gabe_seat))

-- The final Lean statement to prove that Gabe is in seat #2
theorem gabe_is_in_seat_2 : 
  ∀ (fiona_seat gabe_seat ella_seat : Seat),
    fiona_in_seat_3 fiona_seat ∧
    fiona_next_to_gabe fiona_seat gabe_seat ∧
    ella_not_between_fiona_and_gabe ella_seat fiona_seat gabe_seat →
    gabe_seat = 2 :=
by
  intros
  sorry

end gabe_is_in_seat_2_l782_782843


namespace benjamin_franklin_gathering_handshakes_l782_782819

theorem benjamin_franklin_gathering_handshakes :
  ∃ (n m : ℕ), n = 15 ∧ m = 15 ∧ 
  let total_handshakes := (n * (n - 1)) / 2 + n * (m - 1)
  in total_handshakes = 315 :=
by
  sorry

end benjamin_franklin_gathering_handshakes_l782_782819


namespace find_x_l782_782350
-- Import all necessary libraries

-- Define the conditions
variables (x : ℝ) (log5x log6x log15x : ℝ)

-- Assume the edge lengths of the prism are logs with different bases
def edge_lengths (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  log5x = Real.logb 5 x ∧ log6x = Real.logb 6 x ∧ log15x = Real.logb 15 x

-- Define the ratio of Surface Area to Volume
def ratio_SA_to_V (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  let SA := 2 * (log5x * log6x + log5x * log15x + log6x * log15x)
  let V  := log5x * log6x * log15x
  SA / V = 10

-- Prove the value of x
theorem find_x (h1 : edge_lengths x log5x log6x log15x) (h2 : ratio_SA_to_V x log5x log6x log15x) :
  x = Real.rpow 450 (1/5) := 
sorry

end find_x_l782_782350


namespace BS_gt_CS_l782_782799

noncomputable section

open EuclideanGeometry

-- Definitions of the points in the Euclidean Plane
variables {A B C D E S : Point}

-- Definitions of the conditions
variables 
  (h1 : InscribedPentagon A B C D E)
  (h2 : dist B C < dist C D)
  (h3 : dist A B < dist D E)
  (h4 : ∀ P, dist S A ≥ dist S P)

-- The theorem to be proven
theorem BS_gt_CS 
  (h1 : InscribedPentagon A B C D E)
  (h2 : dist B C < dist C D)
  (h3 : dist A B < dist D E)
  (h4 : ∀ P, dist S A ≥ dist S P) : 
  dist S B > dist S C :=
sorry

end BS_gt_CS_l782_782799


namespace redistribution_amount_l782_782487

theorem redistribution_amount
    (earnings : Fin 5 → ℕ)
    (h : earnings = ![18, 22, 30, 35, 45]) :
    (earnings 4 - ((earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5)) = 15 :=
by
  sorry

end redistribution_amount_l782_782487


namespace problem_statement_l782_782180

def is_prime (p : ℕ) : Prop :=
  ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

def polynomial (R : Type*) [comm_ring R] : Type* := R[X]

def degree {R : Type*} [comm_ring R] (f : polynomial R) : ℕ := nat.find_greatest (λ n, f.coeff n ≠ 0) (polynomial.nat_degree f)

theorem problem_statement
  (p : ℕ) (f : polynomial ℤ)
  (hp : is_prime p)
  (hf0 : f.eval 0 = 0)
  (hf1 : f.eval 1 = 1)
  (hfn : ∀ n : ℤ, (f.eval n) % p = 0 ∨ (f.eval n) % p = 1) :
  degree f ≥ p - 1 :=
sorry

end problem_statement_l782_782180


namespace tangent_line_curves_l782_782984

theorem tangent_line_curves (a : ℝ) :
  (∃ m b : ℝ, ∀ x : ℝ, m * x + b = x ^ 3 ∧ m = 3 * x ^ 2 ∧ (1, 0) ∈ set_of (λ p : ℝ × ℝ, p.1 = 1 ∧ p.2 = 0)) ∧
  (∃ m b : ℝ, ∀ x : ℝ, m * x + b = a * x ^ 2 + (15 / 4) * x - 9 ∧ (1, 0) ∈ set_of (λ p : ℝ × ℝ, p.1 = 1 ∧ p.2 = 0))
  → a = -1 ∨ a = -25 / 64 :=
begin
  sorry -- Proof required
end

end tangent_line_curves_l782_782984


namespace problem1_problem2_l782_782578

variable {A B C a b c : ℝ}

-- Problem (1)
theorem problem1 (h : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B) : b = 2 * c := 
sorry

-- Problem (2)
theorem problem2 (a_eq : a = 1) (tanA_eq : Real.tan A = 2 * Real.sqrt 2) (b_eq_c : b = 2 * c): 
  Real.sqrt (c^2 * (1 - (Real.cos (A + B)))) = 2 * Real.sqrt 2 * b :=
sorry

end problem1_problem2_l782_782578


namespace percentage_of_women_in_company_l782_782376

theorem percentage_of_women_in_company (initial_workers : ℕ) (men_ratio women_ratio : ℚ) (new_women : ℕ) 
  (h_total : initial_workers = 90) 
  (h_men_ratio : men_ratio = 2/3) 
  (h_women_ratio : women_ratio = 1/3)
  (h_new_women : new_women = 10) :
  let men := (men_ratio * initial_workers).natAbs in
  let women := (women_ratio * initial_workers).natAbs in
  let total_women := women + new_women in
  let total_workers := initial_workers + new_women in
  (total_women * 100 / total_workers) = 40 := 
by
  sorry

end percentage_of_women_in_company_l782_782376


namespace matrix_inverse_l782_782839

theorem matrix_inverse (a b : ℝ) :
  (\begin{pmatrix} 4 & -2 \\ a & b \end{pmatrix} * \begin{pmatrix} 4 & -2 \\ a & b \end{pmatrix} = 1) → 
  a = 15/2 ∧ b = -4 := 
by 
  sorry

end matrix_inverse_l782_782839


namespace problem_solution_l782_782003

theorem problem_solution :
  { x : ℝ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) } 
  = { x : ℝ | x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end problem_solution_l782_782003


namespace f_increasing_f_at_2_solve_inequality_l782_782557

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a + f b - 1
axiom f_pos (x : ℝ) (h : x > 0) : f x > 1
axiom f_at_4 : f 4 = 5

theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

theorem f_at_2 : f 2 = 3 :=
sorry

theorem solve_inequality (m : ℝ) : f (3 * m^2 - m - 2) < 3 ↔ -1 < m ∧ m < 4 / 3 :=
sorry

end f_increasing_f_at_2_solve_inequality_l782_782557


namespace simple_interest_calculation_l782_782087

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t) - P

noncomputable def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem simple_interest_calculation :
  ∀ (P r : ℝ) (n t : ℕ),
  compound_interest P r n t = 993 →
  r = 0.10 →
  n = 1 →
  t = 4 →
  simple_interest P r t = 856.19 :=
begin
  intros P r n t h_compound_interest h_r h_n h_t,
  sorry
end

end simple_interest_calculation_l782_782087


namespace standard_equation_and_directrix_of_parabola_l782_782516

theorem standard_equation_and_directrix_of_parabola (vertex : ℝ × ℝ) (M : ℝ × ℝ)
  (h_vertex : vertex = (0, 0)) (h_M : M = (1, -2)) :
  (∃ m : ℝ, y² = m * x ∧ m ≠ 0) ∨ (∃ n : ℝ, x² = n * y ∧ n ≠ 0) ∧
  (∃ m : ℝ, y² = 4x ∨ x² = - (1 / 2) * y) ∧
  (∃ k : ℝ, x = -1 ∨ y = 1 / 8) :=
by
  sorry

end standard_equation_and_directrix_of_parabola_l782_782516


namespace charity_event_raised_1080_l782_782342

noncomputable def total_money_raised : ℝ :=
  let a_tickets := 100 * 3.00
  let b_tickets := 50 * 5.50
  let c_tickets := 25 * 10.00
  let donations := 30.00 + 30.00 + 50.00 + 45.00 + 100.00
  a_tickets + b_tickets + c_tickets + donations

theorem charity_event_raised_1080 :
  total_money_raised = 1080.00 :=
by 
  have h_a : a_tickets = 300.00 := rfl
  have h_b : b_tickets = 275.00 := rfl
  have h_c : c_tickets = 250.00 := rfl
  have h_d : donations = 255.00 := rfl
  have h_total : total_money_raised = 1080.00 := 
    calc
      a_tickets + b_tickets + c_tickets + donations = 300.00 + 275.00 + 250.00 + 255.00 : by rw [h_a, h_b, h_c, h_d]
      ... = 1080.00 : by norm_num
  exact h_total

end charity_event_raised_1080_l782_782342


namespace perfect_square_partition_l782_782872

open Nat

-- Define the condition of a number being a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- Define the main theorem statement
theorem perfect_square_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n+1)) → (A ∩ B = ∅) →
  ∃ a b ∈ A, a ≠ b ∧ is_perfect_square (a + b)
:= by
  sorry

end perfect_square_partition_l782_782872


namespace largest_integer_2010_divides_2010_factorial_square_l782_782404

noncomputable def largest_integer_dividing_factorial_square (n : ℕ) : ℕ :=
  let prime_factors_2010 := [2, 3, 5, 67]
  let k := prime_factors_2010.map (λ p, (Nat.floor (2010/ p)) + (Nat.floor (2010/ (p * p))))
  2 * k.foldr Nat.add 0

theorem largest_integer_2010_divides_2010_factorial_square : largest_integer_dividing_factorial_square 2010 = 60 :=
by sorry

end largest_integer_2010_divides_2010_factorial_square_l782_782404


namespace xiaoqiang_matches_l782_782621

def num_of_matches_jia : Nat := 5
def num_of_matches_yi : Nat := 4
def num_of_matches_bing : Nat := 3
def num_of_matches_ding : Nat := 2
def num_of_matches_wu : Nat := 1

theorem xiaoqiang_matches (num_of_matches_jia = 5) 
                          (num_of_matches_yi = 4) 
                          (num_of_matches_bing = 3) 
                          (num_of_matches_ding = 2) 
                          (num_of_matches_wu = 1) : 
                          ∃ n, n = 3 := sorry

end xiaoqiang_matches_l782_782621


namespace intersection_points_relation_l782_782510

noncomputable def num_intersections (k : ℕ) : ℕ :=
  k * (k - 1) / 2

theorem intersection_points_relation (k : ℕ) :
  num_intersections (k + 1) = num_intersections k + k := by
sorry

end intersection_points_relation_l782_782510


namespace no_even_threes_in_circle_l782_782815

theorem no_even_threes_in_circle (arr : ℕ → ℕ) (h1 : ∀ i, 1 ≤ arr i ∧ arr i ≤ 2017)
  (h2 : ∀ i, (arr i + arr ((i + 1) % 2017) + arr ((i + 2) % 2017)) % 2 = 0) : false :=
sorry

end no_even_threes_in_circle_l782_782815


namespace evaluate_three_cubed_squared_l782_782440

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l782_782440


namespace shelves_count_l782_782307

theorem shelves_count (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) : total_books / books_per_shelf = 1780 := 
by 
  rw [h1, h2]
  norm_num
  sorry

end shelves_count_l782_782307


namespace bees_multiple_l782_782198

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end bees_multiple_l782_782198


namespace bonus_relationship_correct_sales_profit_xiaojiang_l782_782791

noncomputable def bonus_relation (x : ℝ) : ℝ :=
  if x ≤ 8 then 0.15 * x
  else 1.2 + log (5 : ℝ) (2 * x - 15)

theorem bonus_relationship_correct (x : ℝ) (hx : 0 ≤ x) : bonus_relation x = 
  if x ≤ 8 then 0.15 * x
  else 1.2 + log (5 : ℝ) (2 * x - 15) :=
sorry

theorem sales_profit_xiaojiang (x : ℝ) (hx : 0 ≤ x) (h_bonus : bonus_relation x = 3.2) : 
  x = 20 :=
sorry

end bonus_relationship_correct_sales_profit_xiaojiang_l782_782791


namespace calculate_expression_l782_782395

theorem calculate_expression : 3 * Real.sqrt 2 - abs (Real.sqrt 2 - Real.sqrt 3) = 4 * Real.sqrt 2 - Real.sqrt 3 :=
  by sorry

end calculate_expression_l782_782395


namespace number_of_pairs_l782_782835

noncomputable def num_pairs (x y : ℕ) : ℕ :=
  if x ≤ y ∧ gcd x y = nat.factorial 5 ∧ nat.lcm x y = nat.factorial 50
    then 1
    else 0

theorem number_of_pairs : 
  (finset.univ.filter (λ p : ℕ × ℕ, num_pairs p.fst p.snd = 1)).card = 2 ^ 14 :=
sorry

end number_of_pairs_l782_782835


namespace sum_of_squares_eq_2_l782_782965

theorem sum_of_squares_eq_2 (a b : ℝ) 
  (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 :=
by sorry

end sum_of_squares_eq_2_l782_782965


namespace length_of_segment_EC_l782_782535

-- Given conditions as Lean definitions
def mAngleA := 45
def BC := 8
def BD_perp_AC := True
def CE_perp_AB := True
def mAngleDBC_eq_2mAngleECB (angleECB : ℤ) := mAngleDBC = 2 * angleECB

-- The statement to be proven
theorem length_of_segment_EC (a b c : ℤ) (angleECB : ℤ) :
  mAngleA = 45 ∧ BC = 8 ∧ BD_perp_AC ∧ CE_perp_AB ∧ mAngleDBC_eq_2mAngleECB angleECB ∧
  EC = a * (Int.sqrt b + Int.sqrt c) ∧ ¬∃ (n : ℤ), n * n = b ∧ ¬∃ (n : ℤ), n * n = c → 
  a + b + c = 7 :=
by
  sorry

end length_of_segment_EC_l782_782535


namespace value_of_expression_at_minus_two_l782_782765

theorem value_of_expression_at_minus_two : 
  (x = -2) → (x^3 + x^2 + 3*x - 6 = -16) :=
by
  intro h
  rw h
  sorry

end value_of_expression_at_minus_two_l782_782765


namespace power_of_power_evaluation_l782_782456

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l782_782456


namespace find_k_of_division_property_l782_782970

theorem find_k_of_division_property (k : ℝ) :
  (3 * (1 / 3)^3 - k * (1 / 3)^2 + 4) % (3 * (1 / 3) - 1) = 5 → k = -8 :=
by sorry

end find_k_of_division_property_l782_782970


namespace log_product_simplification_l782_782078

theorem log_product_simplification :
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) *
  (Real.log 6 / Real.log 5) * ... * (Real.log 50 / Real.log 49) +
  (Real.log 100 / Real.log 50) = 
  3 + 2 * (Real.log 5 / Real.log 2) :=
sorry

end log_product_simplification_l782_782078


namespace marble_distribution_count_l782_782108

-- Let's define the problem step by step:

-- The distinct integers that sum to 18 for 5 people:
def distinct_marble_distribution : List ℕ := [1, 2, 3, 4, 8]

-- Verifying the sum:
lemma sum_of_distinct_integers : distinct_marble_distribution.sum = 18 :=
by
  simp [distinct_marble_distribution]
  norm_num

-- Number of ways to permute 5 distinct integers:
lemma permutations_count : (distinct_marble_distribution.permutations.length : ℕ) = 120 :=
by
  simp [List.permutations_length_eq_factorial]
  norm_num

-- Formal statement of the problem:
theorem marble_distribution_count :
  (∃ (a b c d e : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
     c ≠ d ∧ c ≠ e ∧
     d ≠ e) ∧
    a + b + c + d + e = 18 ∧
    (List.length (List.permutations [a, b, c, d, e]) = 120)) :=
by
  use [1, 2, 3, 4, 8]
  repeat { split }
  all_goals
  { try { norm_num } }
  sorry

end marble_distribution_count_l782_782108


namespace f_gt_g_l782_782508

open Real

def f (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), 1 / sqrt (i + 1)

def g (n : ℕ) : ℝ :=
  2 * (sqrt (n + 1) - 1)

theorem f_gt_g (n : ℕ) (h : 0 < n) : f n > g n :=
  sorry

end f_gt_g_l782_782508


namespace power_of_power_evaluation_l782_782454

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l782_782454


namespace probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l782_782227

def probability_of_excellence_A : ℚ := 2/5
def probability_of_excellence_B1 : ℚ := 1/4
def probability_of_excellence_B2 : ℚ := 2/5
def probability_of_excellence_B3 (n : ℚ) : ℚ := n

def one_excellence_A : ℚ := 3 * (2/5) * (3/5)^2
def one_excellence_B (n : ℚ) : ℚ := 
    (probability_of_excellence_B1 * (3/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (2/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (3/5) * n)

theorem probability_one_excellence_A : one_excellence_A = 54/125 := sorry

theorem probability_one_excellence_B (n : ℚ) (hn : n = 1/3) : one_excellence_B n = 9/20 := sorry

def expected_excellence_A : ℚ := 3 * (2/5)
def expected_excellence_B (n : ℚ) : ℚ := (13/20) + n

theorem range_n_for_A (n : ℚ) (hn1 : 0 < n) (hn2 : n < 11/20): 
    expected_excellence_A > expected_excellence_B n := sorry

end probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l782_782227


namespace number_of_bowls_l782_782728

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l782_782728


namespace polynomial_solution_l782_782481

noncomputable def p (x : ℝ) : ℝ := (7 / 4) * x^2 + 1

theorem polynomial_solution :
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) ∧ p 2 = 8 :=
by
  sorry

end polynomial_solution_l782_782481


namespace closest_point_to_given_point_l782_782011

noncomputable def closest_point_on_line_to_point (s : ℝ) : affine_space ℝ ℝ^3 :=
  (smul s ⟨ 3, -1, 4 ⟩) +ᵥ ⟨ 5, -2, 3 ⟩

def given_point : affine_space ℝ ℝ^3 := ⟨ 1, 0, 2 ⟩

theorem closest_point_to_given_point :
  ∃ s : ℝ, 
    let point := closest_point_on_line_to_point s in
    point = ⟨ 38/13, -17/13, 3/13 ⟩ ∧
    dist point given_point = sqrt 1667 / 13 :=
begin
  sorry
end

end closest_point_to_given_point_l782_782011


namespace range_of_alpha_l782_782664

-- Define the function y = x^3 - x + 2/3
def f (x : ℝ) : ℝ := x^3 - x + 2 / 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- The task is to prove the range of alpha where f' gives the slope of the tangent line
theorem range_of_alpha :
  ∀ (α : ℝ), (∃ x : ℝ, f' x = tan α) ↔ (α ∈ set.Ico 0 (π / 2) ∨ α ∈ set.Ico (3 * π / 4) π) :=
by
  sorry

end range_of_alpha_l782_782664


namespace sum_of_digits_large_integer_l782_782828

noncomputable def large_integer_sum_digits : ℕ :=
  let series := (list.range 321).map (λ n, 10^(n+1) - 1)
  series.sum

theorem sum_of_digits_large_integer :
  let N := large_integer_sum_digits
  sumDigits N = 342 := 
by
  sorry

end sum_of_digits_large_integer_l782_782828


namespace integral_solution_set_l782_782545

theorem integral_solution_set (a : ℝ) (a_value : a = 2)
  (h : ∀ (x : ℝ), 1 - 3 / (x + a) < 0 ↔ x ∈ (-1 : ℝ, 2)) :
  ∫ x in 0..2, (1 - 3 / (x + 2)) = 2 - 3 * real.log 3 := 
by
  rw a_value at h
  sorry

end integral_solution_set_l782_782545


namespace last_digit_fib_mod_9_l782_782475

def fibonacci_mod_9 : ℕ → ℕ 
| 0       => 0
| 1       => 1
| n       => (fibonacci_mod_9 (n-1) + fibonacci_mod_9 (n-2)) % 9 

theorem last_digit_fib_mod_9 : 
∃ n, (∀ m < n, ∃ k, fibonacci_mod_9 k % 9 = m) ∧ fibonacci_mod_9 n % 9 = 6 :=
sorry

end last_digit_fib_mod_9_l782_782475


namespace perimeter_of_triangle_ACE_lt_one_l782_782643

open_locale classical

variables {A B C D E : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E]

-- Definitions of the distances between points
variable {distance : A → B → ℝ}

-- Conditions of the problem
variable (convex_pentagon : true) -- placeholder for convexity condition
variable (perimeter_eq_one : distance A B + distance B C + distance C D + distance D E + distance E A = 1)

-- Theorem to be proved
theorem perimeter_of_triangle_ACE_lt_one :
  distance A C + distance C E + distance E A < 1 :=
sorry

end perimeter_of_triangle_ACE_lt_one_l782_782643


namespace max_points_on_unit_sphere_l782_782708

theorem max_points_on_unit_sphere (n : ℕ) 
  (h_points : ∀ (i j : ℕ), i ≠ j → dist (points i) (points j) ≥ 1) : n ≤ 14 := sorry

end max_points_on_unit_sphere_l782_782708


namespace relation_between_zero_two_l782_782808

variable {P : Set Int}

axiom P_contains_pos_neg : ∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0
axiom P_contains_odd_even : ∃ x ∈ P, ∃ y ∈ P, IsOdd x ∧ IsEven y
axiom P_excludes_one : ∀ x, x ∈ P → x ≠ 1
axiom P_closed_under_addition : ∀ x y, x ∈ P → y ∈ P → x + y ∈ P

theorem relation_between_zero_two :
  (0 ∈ P) ∧ (2 ∉ P) :=
sorry

end relation_between_zero_two_l782_782808


namespace partition_with_sum_square_l782_782849

def sum_is_square (a b : ℕ) : Prop := ∃ k : ℕ, a + b = k * k

theorem partition_with_sum_square (n : ℕ) (h : n ≥ 15) :
  ∀ (s₁ s₂ : finset ℕ), (∅ ⊂ s₁ ∪ s₂ ∧ s₁ ∩ s₂ = ∅ ∧ (∀ x ∈ s₁ ∪ s₂, x ∈ finset.range (n + 1))) →
  (∃ a b : ℕ, a ≠ b ∧ (a ∈ s₁ ∧ b ∈ s₁ ∨ a ∈ s₂ ∧ b ∈ s₂) ∧ sum_is_square a b) :=
by sorry

end partition_with_sum_square_l782_782849


namespace man_l782_782344

theorem man's_speed_against_current (Vcurrent Vwith_current : ℝ) (h1 : Vcurrent = 2.5) (h2 : Vwith_current = 21) :
  let V_m := Vwith_current - Vcurrent in
  let Vagainst_current := V_m - Vcurrent in
  Vagainst_current = 16 := 
by 
  sorry

end man_l782_782344


namespace min_value_function_l782_782045

theorem min_value_function (x : ℝ) (h : x > 2) : 
  ∃ m, (∀ y, (y = 4 / (x - 2) + x) → (y ≥ m)) ∧ m = 6 :=
begin
  sorry
end

end min_value_function_l782_782045


namespace room_width_to_perimeter_ratio_l782_782807

theorem room_width_to_perimeter_ratio (L W : ℕ) (hL : L = 25) (hW : W = 15) :
  let P := 2 * (L + W)
  let ratio := W / P
  ratio = 3 / 16 :=
by
  sorry

end room_width_to_perimeter_ratio_l782_782807


namespace calculate_expr_l782_782394

theorem calculate_expr : (125 : ℝ)^(2/3) * 2 = 50 := sorry

end calculate_expr_l782_782394


namespace sqrt_sum_of_powers_l782_782764

theorem sqrt_sum_of_powers :
  sqrt (4^4 + 4^4 + 4^4) = 16 * real.sqrt 3 :=
by {
  sorry
}

end sqrt_sum_of_powers_l782_782764


namespace common_ratio_is_63_98_l782_782834

/-- Define the terms of the geometric series -/
def term (n : Nat) : ℚ := 
  match n with
  | 0 => 4 / 7
  | 1 => 18 / 49
  | 2 => 162 / 343
  | _ => sorry  -- For simplicity, we can define more terms if needed, but it's irrelevant for our proof

/-- Define the common ratio of the geometric series -/
def common_ratio (a b : ℚ) : ℚ := b / a

/-- The problem states that the common ratio of first two terms of the given series is equal to 63/98 -/
theorem common_ratio_is_63_98 : common_ratio (term 0) (term 1) = 63 / 98 :=
by
  -- leave the proof as sorry for now
  sorry

end common_ratio_is_63_98_l782_782834


namespace count_perfect_square_integers_l782_782490

-- Define the necessary functions and checks
def is_square (m : ℤ) : Prop := ∃ k : ℤ, k^2 = m

def num_perfect_square_integers (a : ℤ) (b : ℤ) : ℕ := 
  Finset.card ((Finset.filter (λ n, is_square (n / (b - n))) (Finset.range (b + 1))): Finset ℤ)

theorem count_perfect_square_integers : 
  num_perfect_square_integers 0 25 = 2 :=
sorry

end count_perfect_square_integers_l782_782490


namespace fixed_point_of_line_minimized_triangle_area_l782_782919

theorem fixed_point_of_line (a : ℝ) : ∃ x y : ℝ, (a + 1) * x + y - 5 - 2 * a = 0 ∧ x = 2 ∧ y = 3 :=
by
  sorry

theorem minimized_triangle_area : ∃ (a b : ℝ) (h1 : a > 0) (h2 : b > 0), (2 / a + 3 / b = 1) ∧ (a * b = 24) ∧ (3 * 4 / a + 2 * 6 / b - 12 = 0) :=
by
  sorry

end fixed_point_of_line_minimized_triangle_area_l782_782919


namespace triangle_at_most_one_obtuse_l782_782999

theorem triangle_at_most_one_obtuse 
  (A B C : ℝ)
  (h_sum : A + B + C = 180) 
  (h_obtuse_A : A > 90) 
  (h_obtuse_B : B > 90) 
  (h_obtuse_C : C > 90) :
  false :=
by 
  sorry

end triangle_at_most_one_obtuse_l782_782999


namespace isosceles_triangle_sides_l782_782594

/-
  Given: 
  - An isosceles triangle with a perimeter of 60 cm.
  - The intersection point of the medians lies on the inscribed circle.
  Prove:
  - The sides of the triangle are 25 cm, 25 cm, and 10 cm.
-/

theorem isosceles_triangle_sides (AB BC AC : ℝ) 
  (h1 : AB = BC)
  (h2 : AB + BC + AC = 60) 
  (h3 : ∃ r : ℝ, r > 0 ∧ 6 * r = AC ∧ 3 * r * AC = 30 * r) :
  AB = 25 ∧ BC = 25 ∧ AC = 10 :=
sorry

end isosceles_triangle_sides_l782_782594


namespace exists_point_with_no_more_than_three_nearest_points_l782_782337

def finite_set (α : Type) [MetricSpace α] := Set α

theorem exists_point_with_no_more_than_three_nearest_points (M : finite_set ℝ) 
  (h_finite : Set.Finite M) :
  ∃ x ∈ M, ∃ Sl : finite_set ℝ, Sl ⊆ M ∧ Sl.card ≤ 3 ∧ ∀ y ∈ Sl, dist x y = dist x (Set.closest_point x M) :=
sorry

end exists_point_with_no_more_than_three_nearest_points_l782_782337


namespace angle5_measure_l782_782192

-- Given definitions
variables (m n : Line) (h_parallel : m ∥ n)
variables (angle1 angle2 angle5 : ℕ)
variable (h_angle1_angle2 : measure angle1 = (1 / 6) * measure angle2)

-- The main theorem to prove
theorem angle5_measure (m n : Line) (h_parallel : m ∥ n)
  (angle1 angle2 angle5 : ℕ)
  (h_angle1_angle2 : measure angle1 = (1 / 6) * measure angle2) :
  measure angle5 = 180 / 7 :=
by
  sorry

end angle5_measure_l782_782192


namespace price_per_foot_of_fence_l782_782762

theorem price_per_foot_of_fence (area : ℝ) (total_cost : ℝ) (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) 
  (h1 : area = 289) (h2 : total_cost = 3672) (h3 : side_length = Real.sqrt area) (h4 : perimeter = 4 * side_length) (h5 : price_per_foot = total_cost / perimeter) :
  price_per_foot = 54 := by
  sorry

end price_per_foot_of_fence_l782_782762


namespace choose_3_out_of_10_l782_782119

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 :=
by
  sorry

end choose_3_out_of_10_l782_782119


namespace ABC_collinear_l782_782896

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]

-- Given conditions as definitions
def AB (a b : V) : V := a + 5 • b
def BC (a b : V) : V := -2 • a + 8 • b
def CD (a b : V) : V := 4 • a + 2 • b
def BD (a b : V) : V := BC a b + CD a b

-- Proof that points A, B, D are collinear
theorem ABC_collinear (a b : V) (λ : ℝ) :
  AB a b = λ • BD a b ↔ λ = 1 / 2 := sorry

end ABC_collinear_l782_782896


namespace owen_hours_on_chores_l782_782658

def hours_in_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_sleeping : ℕ := 11

def hours_on_chores : ℕ := hours_in_day - hours_at_work - hours_sleeping

theorem owen_hours_on_chores : hours_on_chores = 7 :=
by
  simp [hours_on_chores, hours_in_day, hours_at_work, hours_sleeping]
  sorry

end owen_hours_on_chores_l782_782658


namespace seq_not_arithmetic_l782_782702

def sum_seq (n : ℕ) : ℤ := n^2 - 7*n + 6

def first_terms : list ℤ := [0, -4, -2, 0]

def is_arithmetic (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, (n < m → a (m+1) - a m = a (n+1) - a n)

theorem seq_not_arithmetic : ¬ is_arithmetic (λ n, match n with
  | 0     => 0
  | 1     => -4
  | 2     => -2
  | 3     => 0
  | (n+1) => (2 * (n+1)) - 8
  end) := 
by sorry


def a_n : ℕ → ℤ
| 0 := 0
| n := 2*n - 8

lemma seq_general_term :
  ∀ n, a_n n = 
  if n = 1 then 0 
  else 2 * n - 8 := 
by sorry

lemma first_four_terms :
  list.map a_n [1, 2, 3, 4] = [0, -4, -2, 0] :=
by sorry

end seq_not_arithmetic_l782_782702


namespace complex_quadrant_l782_782925

example (z : ℂ) (h : (1 + 2 * ℂ.I) * z = 3 + ℂ.I * z) : (z = 3 / 2 - (3 / 2) * ℂ.I) :=
by sorry

theorem complex_quadrant (z : ℂ) (h : (1 + 2 * ℂ.I) * z = 3 + ℂ.I * z) : z.re > 0 ∧ z.im < 0 :=
by sorry

end complex_quadrant_l782_782925


namespace positive_root_correct_l782_782630

open Real

noncomputable def solve_equation (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : ℝ :=
  (a * b * c) / (a * b + b * c + c * a + 2 * sqrt (a * b * c * (a + b + c)))

theorem positive_root_correct (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  (∑ z in [a, b, c], sqrt (z * a * b * solve_equation a b c (by assumption) (by assumption) (by assumption)))
  = sqrt (a * b * c * (a + b + c)) :=
sorry

end positive_root_correct_l782_782630


namespace number_of_rice_packets_l782_782756

theorem number_of_rice_packets
  (initial_balance : ℤ) 
  (price_per_rice_packet : ℤ)
  (num_wheat_flour_packets : ℤ) 
  (price_per_wheat_flour_packet : ℤ)
  (price_soda : ℤ) 
  (remaining_balance : ℤ)
  (spent : ℤ)
  (eqn : initial_balance - (price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda) = remaining_balance) :
  price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda = spent 
    → initial_balance - spent = remaining_balance
    → 2 = 2 :=
by 
  sorry

end number_of_rice_packets_l782_782756


namespace find_original_wage_l782_782363

-- Definitions based on problem conditions
def original_wage (W : ℝ) : Prop :=
  let increased_wage := 1.5 * W in
  let tax := 0.2 * increased_wage in
  let deduction := 3 in
  let take_home_pay := increased_wage - tax - deduction in
  take_home_pay = 42

-- Theorem to be proven
theorem find_original_wage : ∃ W : ℝ, original_wage W ∧ W = 37.5 :=
begin
  use 37.5,
  unfold original_wage,
  norm_num,
end

end find_original_wage_l782_782363


namespace no_winning_strategy_l782_782335

theorem no_winning_strategy (r b : ℕ) (h1 : r + b = 52) (strategy : ℕ → bool) :
  ∀ k, (strategy k → (r / (r + b : ℝ) ≤ 0.5)) :=
by 
  sorry

end no_winning_strategy_l782_782335


namespace isosceles_triangle_sides_l782_782596

/-
  Given: 
  - An isosceles triangle with a perimeter of 60 cm.
  - The intersection point of the medians lies on the inscribed circle.
  Prove:
  - The sides of the triangle are 25 cm, 25 cm, and 10 cm.
-/

theorem isosceles_triangle_sides (AB BC AC : ℝ) 
  (h1 : AB = BC)
  (h2 : AB + BC + AC = 60) 
  (h3 : ∃ r : ℝ, r > 0 ∧ 6 * r = AC ∧ 3 * r * AC = 30 * r) :
  AB = 25 ∧ BC = 25 ∧ AC = 10 :=
sorry

end isosceles_triangle_sides_l782_782596


namespace cost_difference_per_square_inch_l782_782286

theorem cost_difference_per_square_inch (width1 height1 width2 height2 : ℕ) (cost1 cost2 : ℕ)
  (h_size1 : width1 = 24 ∧ height1 = 16)
  (h_cost1 : cost1 = 672)
  (h_size2 : width2 = 48 ∧ height2 = 32)
  (h_cost2 : cost2 = 1152) :
  (cost1 / (width1 * height1) : ℚ) - (cost2 / (width2 * height2) : ℚ) = 1 := 
by
  sorry

end cost_difference_per_square_inch_l782_782286


namespace no_winning_strategy_l782_782320

noncomputable def probability_of_winning_after_stop (r b : ℕ) : ℚ :=
  r / (r + b : ℚ)

theorem no_winning_strategy (r b : ℕ) (h : r = 26 ∧ b = 26) : 
  ¬ (∃ strategy : (ℕ → Bool) → ℚ, strategy (λ x, true) > 0.5) := 
by
  sorry

end no_winning_strategy_l782_782320


namespace partition_perfect_square_l782_782868

theorem partition_perfect_square (n : ℕ) (h : n ≥ 15) :
  ∀ A B : finset ℕ, disjoint A B → A ∪ B = finset.range (n + 1) →
  ∃ x y ∈ A ∨ ∃ x y ∈ B, x ≠ y ∧ (∃ k : ℕ, x + y = k^2) :=
begin
  sorry
end

end partition_perfect_square_l782_782868


namespace range_of_n_l782_782060

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

variable {a b n y1 y2 : ℝ}

theorem range_of_n (h_a : a > 0) 
  (hA : parabola a b (2*n + 3) = y1) 
  (hB : parabola a b (n - 1) = y2)
  (h_sym : y1 < y2) 
  (h_opposite_sides : (2*n + 3 - 1) * (n - 1 - 1) < 0) :
  -1 < n ∧ n < 0 :=
sorry

end range_of_n_l782_782060


namespace sin_30_eq_one_half_l782_782403

noncomputable def sin_thirty_degree : Prop :=
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) ∧
  Q.2 = 1 / 2

theorem sin_30_eq_one_half : sin_thirty_degree :=
sorry

end sin_30_eq_one_half_l782_782403


namespace spider_distance_l782_782810

/--
A spider crawls along a number line, starting at -3.
It crawls to -7, then turns around and crawls to 8.
--/
def spiderCrawl (start : ℤ) (point1 : ℤ) (point2 : ℤ): ℤ :=
  let dist1 := abs (point1 - start)
  let dist2 := abs (point2 - point1)
  dist1 + dist2

theorem spider_distance :
  spiderCrawl (-3) (-7) 8 = 19 :=
by
  sorry

end spider_distance_l782_782810


namespace partition_perfect_square_l782_782867

theorem partition_perfect_square (n : ℕ) (h : n ≥ 15) :
  ∀ A B : finset ℕ, disjoint A B → A ∪ B = finset.range (n + 1) →
  ∃ x y ∈ A ∨ ∃ x y ∈ B, x ≠ y ∧ (∃ k : ℕ, x + y = k^2) :=
begin
  sorry
end

end partition_perfect_square_l782_782867


namespace probability_of_specific_marble_selection_l782_782025

theorem probability_of_specific_marble_selection :
  ∃ (bag : Finset ℕ) (red blue green : Finset ℕ) (total : ℕ),
  (bag = {0, 1, 2, 3, 4, 5, 6, 7, 8}) ∧
  (red = {0, 1, 2}) ∧
  (blue = {3, 4, 5}) ∧
  (green = {6, 7, 8}) ∧
  (total = bag.card) ∧
  (total = 9) ∧
  let ways_to_choose_4 := (Finset.card (Finset.powersetLen 4 bag)) in
  let ways_to_choose_specific := (Finset.card (Finset.powersetLen 2 red) *
                                  Finset.card (Finset.powersetLen 1 blue) *
                                  Finset.card (Finset.powersetLen 1 green)) in
  (ways_to_choose_4 = 126) ∧
  (ways_to_choose_specific = 27) ∧
  (ways_to_choose_specific / ways_to_choose_4 = 3 / 14) :=
begin
  use [{0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9],
  repeat {split},
  { exact rfl, }, -- bag = {0, 1, 2, 3, 4, 5, 6, 7, 8}
  { exact rfl, }, -- red = {0, 1, 2}
  { exact rfl, }, -- blue = {3, 4, 5}
  { exact rfl, }, -- green = {6, 7, 8}
  { exact rfl, }, -- total = 9
  { exact nat.eq_refl 9, }, -- total = 9
  have ways_to_choose_4 := Finset.card (Finset.powersetLen 4 {0, 1, 2, 3, 4, 5, 6, 7, 8}),
  have ways_to_choose_specific := Finset.card (Finset.powersetLen 2 {0, 1, 2}) *
                                   Finset.card (Finset.powersetLen 1 {3, 4, 5}) *
                                   Finset.card (Finset.powersetLen 1 {6, 7, 8}),
  { exact nat.eq_refl 126, }, -- ways_to_choose_4 = 126
  { exact nat.eq_refl 27, }, --  ways_to_choose_specific = 27
  norm_num,
  sorry -- ways_to_choose_specific / ways_to_choose_4 = 3 / 14
end

end probability_of_specific_marble_selection_l782_782025


namespace min_value_expression_l782_782014

theorem min_value_expression (n : ℕ) (h : 0 < n) : 
  ∃ (m : ℕ), (m = n) ∧ (∀ k > 0, (k = n) -> (n / 3 + 27 / n) = 6) := 
sorry

end min_value_expression_l782_782014


namespace median_siblings_l782_782988

theorem median_siblings (students : ℕ) (siblings : list ℕ)
  (h_students : students = 15) (h_siblings_distribution : siblings = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6])
  (h_siblings_len : siblings.length = students) :
  list.nth_le (list.sort (≤) siblings) ((students + 1) / 2 - 1) sorry = 3 :=
by {
  sorry
}

end median_siblings_l782_782988


namespace common_divisors_90_105_l782_782956

def divisors (n : ℕ) : Finset ℤ :=
  (Finset.range (n+1)).filter (λ d, n % d = 0).image (λ d, d : ℤ ∈ d) ∪ -(Finset.range (n+1)).filter (λ d, n % d = 0).image (λ d, -(d : ℤ ∈ d))

def common_divisors_count (a b : ℕ) : Nat :=
  (divisors a) ∩ (divisors b).card

theorem common_divisors_90_105 : common_divisors_count 90 105 = 8 :=
by
  sorry

end common_divisors_90_105_l782_782956


namespace monotonicity_extrema_l782_782550

noncomputable def f (a x : ℝ) : ℝ :=
  (a + 1/a) * real.log x + 1/x - x

theorem monotonicity (a : ℝ) (ha : 1 < a) :
  ∀ x : ℝ, (0 < x ∧ x ≤ 1/a) → 
           (f a x < f a (x + 1)) ∧
           (∀ y ∈ set.Ioo (1/a) 1, f a y > f a (y - 1)) :=
sorry

theorem extrema (a : ℝ) (ha : 0 < a) :
  ∀ x : ℝ, 
    (f a (1/a) = -(a + 1/a) * real.log a + a - 1/a) ∧
    (f a a = (a + 1/a) * real.log a + 1/a - a) :=
sorry

end monotonicity_extrema_l782_782550


namespace eval_expr_l782_782429

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l782_782429


namespace jesse_carpet_problem_l782_782150

noncomputable def area_rectangle (length width : ℝ) : ℝ :=
  length * width

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

noncomputable def area_circle (radius : ℝ) : ℝ :=
  Real.pi * radius^2

noncomputable def total_area (rect_area tri_area circ_area carpet_bought : ℝ) : ℝ :=
  rect_area + tri_area + circ_area - carpet_bought

noncomputable def carpet_needed (total_area : ℝ) : ℝ :=
  total_area

noncomputable def max_area (budget price_per_sqft : ℝ) : ℝ :=
  budget / price_per_sqft

theorem jesse_carpet_problem :
  let rect_area := area_rectangle 11 15 in
  let tri_area := area_triangle 12 8 in
  let circ_area := area_circle 6 in
  let initial_carpet := 16 in
  let budget := 800 in
  let total_needed := total_area rect_area tri_area circ_area initial_carpet in
  let needed_carpet := carpet_needed total_needed in
  let max_regular := max_area budget 5 in
  let max_deluxe := max_area budget 7.5 in
  let max_luxury := max_area budget 10 in
  (needed_carpet ≈ 310.097) ∧ (max_regular < 310.097) ∧ (max_deluxe < 310.097) ∧ (max_luxury < 310.097) :=
sorry

end jesse_carpet_problem_l782_782150


namespace eval_exp_l782_782417

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l782_782417


namespace partition_contains_square_sum_l782_782863

-- Define a natural number n
def is_square (x : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = x

theorem partition_contains_square_sum (n : ℕ) (hn : n ≥ 15) :
  ∀ (A B : fin n → Prop), (∀ x, A x ∨ B x) ∧ (∀ x, ¬ (A x ∧ B x)) → (∃ a b, a ≠ b ∧ A a ∧ A b ∧ is_square (a + b)) ∨ (∃ c d, c ≠ d ∧ B c ∧ B d ∧ is_square (c + d)) :=
by
  sorry

end partition_contains_square_sum_l782_782863


namespace shortest_paths_cube_l782_782817

-- Definitions based on conditions
variables (A B : Type) [fintype A] [fintype B]
variables (G : Type) [graph G A] [graph G B]

-- Mathematical statement to be proved
theorem shortest_paths_cube :
  (number_of_shortest_paths G A B) = 6 :=
sorry

end shortest_paths_cube_l782_782817


namespace sum_first_9_terms_l782_782701

variables {a : ℕ → ℝ}  -- Define the arithmetic sequence

-- Condition: the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

-- Condition: 2a8 = 6 + a11
def condition (a : ℕ → ℝ) : Prop :=
2 * a 8 = 6 + a 11

-- Statement: S9 is 54
theorem sum_first_9_terms (h_arith : is_arithmetic_sequence a) (h_cond : condition a) :
  (∑ k in Finset.range 9, a k) = 54 :=
sorry

end sum_first_9_terms_l782_782701


namespace g_is_increasing_l782_782692

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + a

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x + (a / x) - 2 * a

theorem g_is_increasing (a : ℝ) (h : a < 1) :
  ∀ x y : ℝ, 1 < x → 1 < y → x < y → g x a < g y a := by
  sorry

end g_is_increasing_l782_782692


namespace james_percentage_votes_l782_782100

noncomputable def totalVotes : ℕ := 2000
noncomputable def votesNeededToWin : ℕ := totalVotes / 2  -- 50% of 2000 is 1000
noncomputable def additionalVotesNeeded : ℕ := 991

theorem james_percentage_votes (votesReceived : ℕ) :
  votesReceived + additionalVotesNeeded = votesNeededToWin + 1 →
  (votesReceived: ℚ) / totalVotes * 100 = 0.5 := by
  intro h1
  have h2 : votesReceived = 10 := by linarith
  rw h2
  norm_num
  sorry

end james_percentage_votes_l782_782100


namespace value_of_a_l782_782933

def f (x : ℝ) : ℝ :=
if x > 1 then 1 + 1 / x else if -1 ≤ x ∧ x ≤ 1 then x^2 + 1 else 2 * x + 3

theorem value_of_a {a : ℝ} (h : f a = 3 / 2) : a = 2 ∨ a = sqrt 2 / 2 ∨ a = - (sqrt 2) / 2 :=
by
  sorry

end value_of_a_l782_782933


namespace sum_of_solutions_l782_782135

theorem sum_of_solutions (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 169) : x + -x = 0 :=
by 
  have h : x^2 + 81 = 169, from h2.subst h1.symm, sorry

end sum_of_solutions_l782_782135


namespace no_photos_count_l782_782994

theorem no_photos_count (n A B C A_B A_C B_C A_B_C : ℕ) 
(Hn : n = 42) 
(HA : A = 21) 
(HB : B = 20) 
(HC : C = 18) 
(HA_B : A_B = 7) 
(HA_C : A_C = 10) 
(HB_C : B_C = 11) 
(HA_B_C : A_B_C = 6) : 
  n - (A + B + C - A_B - A_C - B_C + A_B_C) = 5 :=
by
  rw [Hn, HA, HB, HC, HA_B, HA_C, HB_C, HA_B_C]
  norm_num
  -- The total number of students with at least one photo
  have hABC : A + B + C - A_B - A_C - B_C + A_B_C = 37 by norm_num
  rw hABC
  norm_num

end no_photos_count_l782_782994


namespace gcd_lcm_product_360_distinct_gcd_values_l782_782292

/-- 
  Given two integers a and b, such that the product of their gcd and lcm is 360,
  we need to prove that the number of distinct possible values for their gcd is 9.
--/
theorem gcd_lcm_product_360_distinct_gcd_values :
  ∀ (a b : ℕ), gcd a b * lcm a b = 360 → 
  (∃ gcd_values : Finset ℕ, gcd_values.card = 9 ∧ ∀ g, g ∈ gcd_values ↔ g = gcd a b) :=
by
  sorry

end gcd_lcm_product_360_distinct_gcd_values_l782_782292


namespace no_strategy_wins_more_than_half_probability_l782_782325

theorem no_strategy_wins_more_than_half_probability
  (deck : Finset Card)
  (red_count black_count : ℕ)
  (well_shuffled : deck.shuffled)
  (player_strategy : (Finset Card) → Bool) :
  ∀ r b, red_count = r ∧ black_count = b →
    (∀ red black : ℕ, (red + black = r + b) → 
      (red / (red + black + 1) ≤ 0.5)) :=
sorry

end no_strategy_wins_more_than_half_probability_l782_782325


namespace final_mixture_alcohol_concentration_l782_782273

def alcohol_concentration (x y : ℝ) : ℝ :=
  (0.6 + 2.7 + x * (y / 100)) / (8 + x)

theorem final_mixture_alcohol_concentration (x y : ℝ) :
  alcohol_concentration x y = (0.6 + 2.7 + x * (y / 100)) / (8 + x) :=
by
  -- proof to be filled in
  sorry

end final_mixture_alcohol_concentration_l782_782273


namespace specially_monotonous_count_l782_782831

open Finset

def is_special_monotonous (n : ℕ) : Prop :=
  if n ≤ 8 then True
  else let digits := nat.digits 10 n in
       (∀ i j, i < j ∧ j < digits.length → digits.nth i < digits.nth j) ∨
       (∀ i j, i < j ∧ j < digits.length → digits.nth i > digits.nth j)

theorem specially_monotonous_count : 
  {n : ℕ | is_special_monotonous n}.to_finset.card = 193 := 
sorry

end specially_monotonous_count_l782_782831


namespace digit_at_100_l782_782780

-- Define the sequence construction pattern
def sequence_digit (n : ℕ) : ℕ :=
  let sum_to_n (k : ℕ) := k * (k + 1) / 2
  let group := (List.range n).find (λ k => sum_to_n k ≥ n).get_or_else 0
  if group % 5 = 0 then 5 else group % 5

-- Proof that the 100th digit in the sequence is 4
theorem digit_at_100 : sequence_digit 100 = 4 :=
  sorry

end digit_at_100_l782_782780


namespace power_of_power_evaluation_l782_782453

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l782_782453


namespace smallest_y_divisible_l782_782291

theorem smallest_y_divisible (y : ℕ) : 
  (y % 3 = 2) ∧ (y % 5 = 4) ∧ (y % 7 = 6) → y = 104 :=
by
  sorry

end smallest_y_divisible_l782_782291


namespace no_winning_strategy_l782_782323

noncomputable def probability_of_winning_after_stop (r b : ℕ) : ℚ :=
  r / (r + b : ℚ)

theorem no_winning_strategy (r b : ℕ) (h : r = 26 ∧ b = 26) : 
  ¬ (∃ strategy : (ℕ → Bool) → ℚ, strategy (λ x, true) > 0.5) := 
by
  sorry

end no_winning_strategy_l782_782323


namespace partition_contains_square_sum_l782_782859

-- Define a natural number n
def is_square (x : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = x

theorem partition_contains_square_sum (n : ℕ) (hn : n ≥ 15) :
  ∀ (A B : fin n → Prop), (∀ x, A x ∨ B x) ∧ (∀ x, ¬ (A x ∧ B x)) → (∃ a b, a ≠ b ∧ A a ∧ A b ∧ is_square (a + b)) ∨ (∃ c d, c ≠ d ∧ B c ∧ B d ∧ is_square (c + d)) :=
by
  sorry

end partition_contains_square_sum_l782_782859


namespace oliver_learning_vowels_l782_782202

theorem oliver_learning_vowels : 
  let learn := 5
  let rest_days (n : Nat) := n
  let total_days :=
    (learn + rest_days 1) + -- For 'A'
    (learn + rest_days 2) + -- For 'E'
    (learn + rest_days 3) + -- For 'I'
    (learn + rest_days 4) + -- For 'O'
    (rest_days 5 + learn)  -- For 'U' and 'Y'
  total_days = 40 :=
by
  sorry

end oliver_learning_vowels_l782_782202


namespace sum_of_squares_eq_two_l782_782967

theorem sum_of_squares_eq_two {a b : ℝ} (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 := sorry

end sum_of_squares_eq_two_l782_782967


namespace maximum_value_of_y_l782_782827

noncomputable def y (x : ℝ) : ℝ := -3 * (Real.cos x)^2 + 4 * Real.sin x + 5

theorem maximum_value_of_y :
  (∃ x : ℝ, 0 < x ∧ x < Real.pi ∧ y x = 9) :=
by {
  use Real.pi / 2,
  split,
  { norm_num, apply True.intro, },
  split,
  { refine Real.pi_pos.trans_le _, apply half_le_self, apply Real.pi_pos, },
  { dsimp [y],
    norm_num,
    simp,
    sorry,
}

end maximum_value_of_y_l782_782827


namespace avg_first_six_results_l782_782233

theorem avg_first_six_results (average_11 : ℕ := 52) (average_last_6 : ℕ := 52) (sixth_result : ℕ := 34) :
  ∃ A : ℕ, (6 * A + 6 * average_last_6 - sixth_result = 11 * average_11) ∧ A = 49 :=
by
  sorry

end avg_first_six_results_l782_782233


namespace gcd_72_120_168_l782_782242

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  -- Each step would be proven individually here.
  sorry

end gcd_72_120_168_l782_782242


namespace setB_is_correct_l782_782497

def setA : Set ℤ := {1, 0, -1, 2}
def setB : Set ℤ := { y | ∃ x ∈ setA, y = Int.natAbs x }

theorem setB_is_correct : setB = {0, 1, 2} := by
  sorry

end setB_is_correct_l782_782497


namespace evaluate_exp_power_l782_782433

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l782_782433


namespace probability_not_next_to_each_other_l782_782103

theorem probability_not_next_to_each_other : 
  let available_chairs := {1, 2, 3, 4, 6, 7, 8, 9, 10} in
  let total_choices := Nat.choose 9 2 in
  let adjacent_pairs := [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10)] in
  let adjacent_count := 8 in
  let prob_adjacent := (adjacent_count:ℚ) / (total_choices:ℚ) in
  let prob_not_adjacent := 1 - prob_adjacent in
  prob_not_adjacent = 7/9 := 
begin
  sorry
end

end probability_not_next_to_each_other_l782_782103


namespace problem1_problem2_l782_782512

noncomputable def f (x a b c : ℝ) : ℝ := abs (x + a) + abs (x - b) + c

theorem problem1 (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : ∃ x, f x a b c = 4) : a + b + c = 4 :=
sorry

theorem problem2 (a b c : ℝ) (h : a + b + c = 4) : (1 / a) + (1 / b) + (1 / c) ≥ 9 / 4 :=
sorry

end problem1_problem2_l782_782512


namespace range_of_a_l782_782543

theorem range_of_a:
  (∀ x ∈ ℝ, f (a^2 - sin x) ≤ f (a + 1 + cos x ^ 2))
  ∧ (∀ x ∈ ℝ, f x ∈ Iic (3 : ℝ))
  → (a ∈ set.Icc (-(real.sqrt 2)) ((1 - real.sqrt 10) / 2)) := by
  sorry

end range_of_a_l782_782543


namespace university_diploma_percentage_l782_782299

def total_population : ℝ := 100
def job_choice_percentage : ℝ := 40 / 100
def no_diploma_job_choice_percentage : ℝ := 10 / 100
def no_job_choice_have_diploma_percentage : ℝ := 30 / 100

theorem university_diploma_percentage :
  let total_population := (1 : ℝ)
  let job_choice_percentage := (0.40 : ℝ)
  let no_diploma_job_choice_percentage := (0.10 : ℝ)
  let no_job_choice_have_diploma_percentage := (0.30 : ℝ)
  total_population = 1 →
  job_choice_percentage = 0.40 →
  no_diploma_job_choice_percentage = 0.10 →
  no_job_choice_have_diploma_percentage = 0.30 →
  let people_with_job_choice_and_diploma := 
    job_choice_percentage - no_diploma_job_choice_percentage
  let people_without_job_choice_but_diploma := 
    no_job_choice_have_diploma_percentage * (1 - job_choice_percentage)
  people_with_job_choice_and_diploma + people_without_job_choice_but_diploma = 0.48
  := by sorry

end university_diploma_percentage_l782_782299


namespace find_natural_numbers_l782_782468

theorem find_natural_numbers (n : ℕ) (p q : ℕ) (hp : p.Prime) (hq : q.Prime)
  (h : q = p + 2) (h1 : (2^n + p).Prime) (h2 : (2^n + q).Prime) :
    n = 1 ∨ n = 3 :=
by
  sorry

end find_natural_numbers_l782_782468


namespace seq_eq_exp_l782_782946

theorem seq_eq_exp (a : ℕ → ℕ) 
  (h₀ : a 1 = 2) 
  (h₁ : ∀ n ≥ 2, a n = 2 * a (n - 1) - 1) :
  ∀ n ≥ 2, a n = 2^(n-1) + 1 := 
  by 
  sorry

end seq_eq_exp_l782_782946


namespace taxi_fare_proportional_l782_782366

theorem taxi_fare_proportional (fare_50 : ℝ) (distance_50 distance_70 : ℝ) (proportional : Prop) (h_fare_50 : fare_50 = 120) (h_distance_50 : distance_50 = 50) (h_distance_70 : distance_70 = 70) :
  fare_70 = 168 :=
by {
  sorry
}

end taxi_fare_proportional_l782_782366


namespace general_term_of_seq_a_l782_782945

def seq_a : ℕ → ℚ
| 1     := 2
| (n+1) := 2 * (n+1) * seq_a n / (seq_a n + n)

theorem general_term_of_seq_a (n : ℕ) (h : n ≠ 0) :
  seq_a (n+1) = n+1 * 2^(n+1) / (2^(n+1) - 1) := sorry

end general_term_of_seq_a_l782_782945


namespace false_proposition_C_is_false_l782_782549

open Real

theorem false_proposition_C_is_false : ¬ ∀ x : ℝ, 2^x - 1 > 0 :=
by
  -- Assume the opposite to derive a contradiction
  intro h
  -- Take x = 0, for instance
  have h0 := h 0
  -- Compute the value 2^0 - 1 which is 0
  have h0_calc : 2^0 - 1 = 0 := by norm_num
  -- Prove the contradiction
  rw h0_calc at h0
  -- Show 0 > 0 is false
  exact lt_irrefl 0 h0
  sorry

end false_proposition_C_is_false_l782_782549


namespace length_of_AD_l782_782134

theorem length_of_AD (AB BC CD : ℝ) (h1 : AB = 4) (h2 : BC = 3.5) (h3 : CD = 2)
  (h4 : ∠(AB,BC) = 90) (h5 : ∠(BC,CD) = 90) 
  (area_eq : (1 / 2) * AB * BC = (1 / 2) * BC * CD) : AD = 6.8 := by
  sorry

end length_of_AD_l782_782134


namespace union_of_two_triangles_not_13gon_l782_782397

theorem union_of_two_triangles_not_13gon 
  (triangle1 : Triangle) 
  (triangle2 : Triangle) 
  (h1 : triangle1.vertices = 3) 
  (h2 : triangle2.vertices = 3) 
  (h3 : triangle1.sides = 3) 
  (h4 : triangle2.sides = 3) 
  (h5 : ∀ s1 ∈ triangle1.sides, ∀ s2 ∈ triangle2.sides, s1 ∩ s2 ≤ 2) : 
  ¬(polygon formed by union_of_two_triangles vertices = 13) :=
by
  -- Proof goes here
  sorry

end union_of_two_triangles_not_13gon_l782_782397


namespace f_increasing_on_nonnegative_max_min_values_on_interval_l782_782553

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (x + 1)

-- Prove that the function f(x) is increasing on [0, +∞)
theorem f_increasing_on_nonnegative :
  ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
sorry

-- Prove that the maximum and minimum values of the function f(x) on [2, 9] are 3/2 and 1/3 respectively
theorem max_min_values_on_interval :
  ∃ max min : ℝ, max = f 9 ∧ min = f 2 ∧ (∀ x ∈ set.Icc 2 9, f x ≤ max ∧ min ≤ f x) :=
sorry

end f_increasing_on_nonnegative_max_min_values_on_interval_l782_782553


namespace seedling_prices_max_type_A_seedlings_l782_782579

theorem seedling_prices (x y : ℕ) : 
  (15 * x + 5 * y = 190) ∧ (25 * x + 15 * y = 370) → 
  x = 10 ∧ y = 8 := 
by 
  sorry

theorem max_type_A_seedlings (x y : ℕ) (m : ℕ) :
  x = 10 → 
  y = 8 → 
  (9 * m + 7.2 * (100 - m) ≤ 828) → 
  m ≤ 60 := 
by 
  sorry

end seedling_prices_max_type_A_seedlings_l782_782579


namespace sum_alternating_binomial_l782_782001

-- Define the problem
theorem sum_alternating_binomial :
  (∑ k in Finset.range 51, (-1:ℤ)^k * Nat.choose 100 (2 * k)) = 2^50 := sorry

end sum_alternating_binomial_l782_782001


namespace calc_complex_square_l782_782392

/-- Define the imaginary unit i such that i^2 = -1 -/
def i : ℂ := complex.I

/-- State the theorem that we need to prove -/
theorem calc_complex_square : (1 + i)^2 = 2 * i :=
by
  sorry

end calc_complex_square_l782_782392


namespace B_work_days_l782_782295

theorem B_work_days (a b : ℝ) (h1 : a + b = 1/4) (h2 : a = 1/14) : 1 / b = 5.6 :=
by
  sorry

end B_work_days_l782_782295


namespace unique_zero_condition_l782_782050

-- Define the function f
def f (x m : ℝ) : ℝ := 4^x + m * 2^x + 1

-- The Lean statement for the proof problem
theorem unique_zero_condition : (∀ x : ℝ, f x m = 0 → x = 0) ↔ (m = -2) := by
  sorry

end unique_zero_condition_l782_782050


namespace smallest_k_divides_l782_782013

-- Define the polynomial f(z) = z^{11} + z^{10} + z^7 + z^6 + z^5 + z^2 + 1
noncomputable def f (z : Complex) := z^11 + z^10 + z^7 + z^6 + z^5 + z^2 + 1

-- Define the polynomial g(z, k) = z^k - 1
noncomputable def g (z : Complex) (k : ℕ) := z^k - 1

-- The main statement to prove that the smallest k such that f(z) divides g(z, k) is 24
theorem smallest_k_divides : ∃ k : ℕ, (k > 0) ∧ (∀ z : Complex, f(z) ∣ g(z, k)) ∧ k = 24 :=
by
  sorry

end smallest_k_divides_l782_782013


namespace polygon_perimeter_l782_782748

/-- A polygon with n sides, each of length l, has perimeter n * l. -/
theorem polygon_perimeter (n : ℕ) (l : ℕ) : ∀ (n = 9) (l = 2), (n * l = 18) :=
by
  sorry

end polygon_perimeter_l782_782748


namespace backpacking_trip_cooks_l782_782132

theorem backpacking_trip_cooks :
  nat.choose 10 3 = 120 :=
sorry

end backpacking_trip_cooks_l782_782132


namespace find_c_l782_782413

open Real

-- Definition of the quadratic expression in question
def expr (x y c : ℝ) : ℝ := 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

-- The theorem to prove that the minimum value of this expression being 0 over all (x, y) implies c = 4
theorem find_c :
  (∀ x y : ℝ, expr x y c ≥ 0) → (∃ x y : ℝ, expr x y c = 0) → c = 4 := 
by 
  sorry

end find_c_l782_782413


namespace vector_addition_result_l782_782066

-- Define the given vectors
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-3, 4)

-- Statement to prove that the sum of the vectors is (-1, 5)
theorem vector_addition_result : vector_a + vector_b = (-1, 5) :=
by
  -- Use the fact that vector addition in ℝ^2 is component-wise
  sorry

end vector_addition_result_l782_782066


namespace correct_options_l782_782502

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c : V)
hypothesis h_basis : LinearIndependent ℝ ![a, b, c] ∧ Submodule.span ℝ ![a, b, c] = ⊤

/- Prove that:
1. The vectors a + b, b + c, c + a can form a basis.
2. For any vector p in the space, there exist real numbers x, y, z such that p = x * a + y * b + z * c.
-/

theorem correct_options (p : V) : 
  (LinearIndependent ℝ ![a + b, b + c, c + a]) ∧ 
  (∃ x y z : ℝ, p = x • a + y • b + z • c) :=
sorry

end correct_options_l782_782502


namespace min_ab_value_l782_782504

theorem min_ab_value (a b : Real) (h_a : 1 < a) (h_b : 1 < b)
  (h_geom_seq : ∀ (x₁ x₂ x₃ : Real), x₁ = (1/4) * Real.log a → x₂ = 1/4 → x₃ = Real.log b →  x₂^2 = x₁ * x₃) : 
  a * b ≥ Real.exp 1 := by
  sorry

end min_ab_value_l782_782504


namespace angle_A_is_pi_over_3_area_of_ABC_l782_782105

theorem angle_A_is_pi_over_3 (A B C : ℝ) (a b c : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_relation : 2 * a * sin B = sqrt 3 * b) : A = π / 3 :=
sorry

theorem area_of_ABC (a b c : ℝ) (A : ℝ) (h_a : a = 6) (h_bc_eq_8 : b + c = 8) (h_A : A = π / 3) : 
1/2 * b * c * sin A = 7 * sqrt 3 / 3 :=
sorry

end angle_A_is_pi_over_3_area_of_ABC_l782_782105


namespace abs_eq_zero_sum_is_neg_two_l782_782977

theorem abs_eq_zero_sum_is_neg_two (x y : ℝ) (h : |x - 1| + |y + 3| = 0) : x + y = -2 := 
by 
  sorry

end abs_eq_zero_sum_is_neg_two_l782_782977


namespace number_of_bowls_l782_782723

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l782_782723


namespace count_valid_three_digit_numbers_l782_782072

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_invalid (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ = d₃ ∧ d₁ ≠ d₂

theorem count_valid_three_digit_numbers :
  let count_invalid := (fun n => if is_invalid n then 1 else 0) in
  let total_invalid := (Finset.range (999 + 1)).sum count_invalid in
  900 - total_invalid = 747 :=
by
  sorry

end count_valid_three_digit_numbers_l782_782072


namespace regular_octagon_diagonal_ratio_l782_782996

theorem regular_octagon_diagonal_ratio :
  ∀ (s : ℝ), 
  let AC := s * Real.sqrt (2 + Real.sqrt 2) in 
  s / AC = 1 / (Real.sqrt (2 + Real.sqrt 2)) :=
by
  sorry

end regular_octagon_diagonal_ratio_l782_782996


namespace total_strawberries_l782_782211

theorem total_strawberries (initial : ℕ) (picked : ℕ) (h_initial : initial = 42) (h_picked : picked = 78) : initial + picked = 120 :=
by
  rw [h_initial, h_picked]
  exact rfl

end total_strawberries_l782_782211


namespace distance_between_foci_hyperbola_l782_782681

open Real

-- Conditions and definitions 
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 1
def hyperbola_eq (x y : ℝ) (a b : ℝ) : Prop :=
  ((x + 1/2)^2 / a^2) - ((y - 2)^2 / b^2) = 1
def point : ℝ × ℝ := (4, 5)
def sum_of_squares := (45 / 4 : ℝ)
def c_squared (a b : ℝ) : ℝ := a^2 + b^2

-- Statement to prove
theorem distance_between_foci_hyperbola :
  ∀ a b : ℝ, 
  hyperbola_eq (point.1) (point.2) a b →
  a^2 = sum_of_squares →
  b^2 = sum_of_squares →
  2 * Real.sqrt(c_squared a b) = 3 * Real.sqrt 10 :=
by
  intros a b ha hb1 hb2
  sorry

end distance_between_foci_hyperbola_l782_782681


namespace y_coordinate_of_vertex_l782_782833

-- Define the quadratic equation
def parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 4

-- State the theorem
theorem y_coordinate_of_vertex : 
  let x_vertex := -((-6) / (2 * 3)) in 
  parabola x_vertex = 1 :=
by
  sorry

end y_coordinate_of_vertex_l782_782833


namespace carrie_pays_199_27_l782_782399

noncomputable def carrie_payment : ℝ :=
  let shirts := 8 * 12
  let pants := 4 * 25
  let jackets := 4 * 75
  let skirts := 3 * 30
  let shoes := 2 * 50
  let shirts_discount := 0.20 * shirts
  let jackets_discount := 0.20 * jackets
  let skirts_discount := 0.10 * skirts
  let total_cost := shirts + pants + jackets + skirts + shoes
  let discounted_cost := (shirts - shirts_discount) + (pants) + (jackets - jackets_discount) + (skirts - skirts_discount) + shoes
  let mom_payment := 2 / 3 * discounted_cost
  let carrie_payment := discounted_cost - mom_payment
  carrie_payment

theorem carrie_pays_199_27 : carrie_payment = 199.27 :=
by
  sorry

end carrie_pays_199_27_l782_782399


namespace lines_parallel_if_perpendicular_planes_l782_782044

variables {α β : Type*}
variables {m n : α} {α β : β}

noncomputable def non_coincident_lines (m n : α) : Prop := m ≠ n
noncomputable def non_coincident_planes (α β : β) : Prop := α ≠ β
noncomputable def line_perpendicular_plane (m : α) (α : β) : Prop := sorry -- Define m ⊥ α
noncomputable def line_parallel (m n : α) : Prop := sorry -- Define m ∥ n
noncomputable def plane_parallel (α β : β) : Prop := sorry -- Define α ∥ β

theorem lines_parallel_if_perpendicular_planes
  (m n : α) (α β : β)
  (hncl : non_coincident_lines m n)
  (hncp : non_coincident_planes α β)
  (hmpa : line_perpendicular_plane m α)
  (hnpb : line_perpendicular_plane n β)
  (hppb : plane_parallel α β) :
  line_parallel m n :=
sorry

end lines_parallel_if_perpendicular_planes_l782_782044


namespace sequence_sum_eq_sixteen_over_nine_l782_782173

noncomputable def r : ℝ := by
  sorry -- Definition of r as the positive real root of x^3 + (3/4)x - 1 = 0

theorem sequence_sum_eq_sixteen_over_nine :
  let S := r^2 + 2 * r^5 + 3 * r^8 + 4 * r^11 + ...
  in S = 16 / 9 :=
by
  sorry

end sequence_sum_eq_sixteen_over_nine_l782_782173


namespace sin_cos_product_sin_minus_cos_l782_782027

variable (x : ℝ)

-- Assumptions
axiom h1 : sin (x + Real.pi) + cos (x - Real.pi) = 1 / 2
axiom h2 : 0 < x ∧ x < Real.pi

theorem sin_cos_product : sin x * cos x = -3 / 8 :=
by
  sorry

theorem sin_minus_cos : sin x - cos x = real.sqrt 7 / 2 :=
by
  sorry

end sin_cos_product_sin_minus_cos_l782_782027


namespace george_has_2_boxes_l782_782495

theorem george_has_2_boxes (blocks_per_box : ℕ) (total_blocks : ℕ) (h1 : blocks_per_box = 6) (h2 : total_blocks = 12) :
    total_blocks / blocks_per_box = 2 :=
by
  rw [h1, h2]
  norm_num

end george_has_2_boxes_l782_782495


namespace find_quotient_l782_782586

def dividend : ℕ := 55053
def divisor : ℕ := 456
def remainder : ℕ := 333

theorem find_quotient (Q : ℕ) (h : dividend = (divisor * Q) + remainder) : Q = 120 := by
  sorry

end find_quotient_l782_782586


namespace sin_theta_add_pi_over_3_l782_782500

theorem sin_theta_add_pi_over_3 (θ : ℝ) (h : Real.cos (π / 6 - θ) = 2 / 3) : 
  Real.sin (θ + π / 3) = 2 / 3 :=
sorry

end sin_theta_add_pi_over_3_l782_782500


namespace root_of_cubic_eqn_l782_782699

theorem root_of_cubic_eqn : 
  ∃ x : ℝ, (1/2) * x^3 + 4 = 0 ∧ x = -2 :=
by {
  use -2,
  split,
  {
    have h1: (1/2 : ℝ) = (1/2), from rfl,
    have h2: (-2)^3 = -8, by norm_num,
    rw [h2, h1],
    ring,
    norm_num,
  },
  { 
    refl,  -- x = -2 is trivial here after we have used -2
  }
}

end root_of_cubic_eqn_l782_782699


namespace traditionalThoughtInfluenceValueConcepts_l782_782247

-- Definitions of the conditions as hypotheses
def FromAnalects : Prop := ∃ (TH : Prop), TH = "Traditional Chinese thought comes from The Analects"
def CoreOfConfuciusThought : Prop := ∃ (TH : Prop), TH = "Traditional Chinese thought is at the core of Confucius' thought"
def EnhancesCohesionAndHarmony : Prop := ∃ (TH : Prop), TH = "Traditional Chinese thought is important today for enhancing national cohesion and building a harmonious society"

-- The proof problem
theorem traditionalThoughtInfluenceValueConcepts : 
  FromAnalects ∧ CoreOfConfuciusThought ∧ EnhancesCohesionAndHarmony →
  ∃ (TH : Prop), TH = "Traditional Chinese thought profoundly influences the value concepts of contemporary Chinese" := by
  sorry

end traditionalThoughtInfluenceValueConcepts_l782_782247


namespace no_strategy_wins_more_than_half_probability_l782_782324

theorem no_strategy_wins_more_than_half_probability
  (deck : Finset Card)
  (red_count black_count : ℕ)
  (well_shuffled : deck.shuffled)
  (player_strategy : (Finset Card) → Bool) :
  ∀ r b, red_count = r ∧ black_count = b →
    (∀ red black : ℕ, (red + black = r + b) → 
      (red / (red + black + 1) ≤ 0.5)) :=
sorry

end no_strategy_wins_more_than_half_probability_l782_782324


namespace perpendicular_proof_l782_782310

variables {Plane Line : Type}
variables (α β : Plane) (m n : Line)

def perp (x y : Type) : Prop := sorry -- Definition for orthogonality/perpendicularity

theorem perpendicular_proof:
  (perp α β ∧ perp m β ∧ perp n α → perp m n) ∨
  (perp m n ∧ perp m β ∧ perp n α → perp α β) :=
sorry

end perpendicular_proof_l782_782310


namespace tap_emptying_time_l782_782790

theorem tap_emptying_time
  (fill_rate1 : ℝ)
  (fill_rate_both : ℝ) 
  (T : ℝ)
  (h1 : fill_rate1 = 1 / 3) 
  (h2 : fill_rate_both = 1 / 6) :
  (1 / T = 1 / 3 - fill_rate_both) → T = 6 :=
begin
  sorry,
end

end tap_emptying_time_l782_782790


namespace matrix_product_l782_782388

theorem matrix_product :
  let M : Fin 51 → Matrix (Fin 2) (Fin 2) ℝ := λ i =>
    match i with
    | ⟨0, _⟩ => ![![1, 2], ![0, 1]]
    | ⟨n+1, _⟩ => ![![1, 2 * (n+1) + 2], ![0, 1]]
  ∏ i in Finset.range 51, M i = ![![1, 2550], ![0, 1]] :=
by
  sorry

end matrix_product_l782_782388


namespace candy_bar_cost_l782_782646

def cost_of_candy_bars :=
  let m := 35
  let t := 3 * m
  let delta := 140
  let c := 2
  105 * c = 35 * c + delta

theorem candy_bar_cost 
  (m t delta : ℕ) (c : ℕ) 
  (hm : m = 35) 
  (ht : t = 3 * m) 
  (hdelta : delta = 140) 
  (hTc : 105 * c = 35 * c + delta) :
  c = 2 :=
by
  subst hm
  subst ht
  subst hdelta
  calc
    105 * 2 = 210 : by norm_num
    35 * 2 + 140 = 70 + 140 : by norm_num
            ... = 210 : by norm_num
  sorry

end candy_bar_cost_l782_782646


namespace sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l782_782312

-- Proof for Problem 1
theorem sin_of_cos_in_third_quadrant (α : ℝ) 
  (hcos : Real.cos α = -4 / 5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3 / 5 :=
by
  sorry

-- Proof for Problem 2
theorem ratio_of_trig_functions (α : ℝ) 
  (htan : Real.tan α = -3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 7 / 2 :=
by
  sorry

end sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l782_782312


namespace area_of_PQRS_l782_782215

theorem area_of_PQRS (P Q R S E F G H : ℝ × ℝ)
  (h1: dist E F = 5) (h2: dist F G = 5) (h3: dist G H = 5) (h4: dist H E = 5)
  (h5: ∃ (P : ℝ × ℝ), dist E P = 5 ∧ dist F P = 5 ∧ dist (E, F, P) = equilateral_triangle)
  (h6: ∃ (Q : ℝ × ℝ), dist F Q = 5 ∧ dist G Q = 5 ∧ dist (F, G, Q) = equilateral_triangle)
  (h7: ∃ (R : ℝ × ℝ), dist G R = 5 ∧ dist H R = 5 ∧ dist (G, H, R) = equilateral_triangle)
  (h8: ∃ (S : ℝ × ℝ), dist H S = 5 ∧ dist E S = 5 ∧ dist (H, E, S) = equilateral_triangle)
  (h9: area_of_square(E, F, G, H) = 25)
: area_of_square(P, Q, R, S) = 50 + 25 * real.sqrt(3) := sorry

end area_of_PQRS_l782_782215


namespace choose_3_from_10_is_120_l782_782111

theorem choose_3_from_10_is_120 :
  nat.choose 10 3 = 120 :=
by {
  -- proof would go here
  sorry
}

end choose_3_from_10_is_120_l782_782111


namespace pyramid_volume_isoceles_right_triangle_base_2_l782_782542

theorem pyramid_volume_isoceles_right_triangle_base_2 :
  let s := 2 in
  let base_area := (sqrt 3) / 4 * s^2 in
  let h := sqrt 2 in
  let volume := (1 / 3) * base_area * h in
  volume = sqrt 2 / 3 :=
by
  sorry

end pyramid_volume_isoceles_right_triangle_base_2_l782_782542


namespace original_price_per_tire_l782_782892

-- Definitions derived from the problem
def number_of_tires : ℕ := 4
def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36

-- Goal to prove the original price of each tire
theorem original_price_per_tire :
  (sale_price_per_tire + total_savings / number_of_tires) = 84 :=
by sorry

end original_price_per_tire_l782_782892


namespace number_of_bowls_l782_782744

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l782_782744


namespace max_int_solutions_edge_centered_polynomial_l782_782347

-- State the conditions and the theorem
theorem max_int_solutions_edge_centered_polynomial :
  ∀ (p : ℤ[X]), p.coeffs ∈ (λ c, ∀ i, c i ∈ (set.univ : set ℤ)) ∧ p.eval 50 = 50 →
  ∃ (s : finset ℤ), (∀ x ∈ s, p.eval x = x^2) ∧ (s.card = 6) :=
begin
  intros p hp,
  sorry, -- proof would go here
end

end max_int_solutions_edge_centered_polynomial_l782_782347


namespace number_of_bowls_l782_782711

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l782_782711


namespace ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l782_782782

theorem ten_times_ten_thousand : 10 * 10000 = 100000 :=
by sorry

theorem ten_times_one_million : 10 * 1000000 = 10000000 :=
by sorry

theorem ten_times_ten_million : 10 * 10000000 = 100000000 :=
by sorry

theorem tens_of_thousands_in_hundred_million : 100000000 / 10000 = 10000 :=
by sorry

end ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l782_782782


namespace number_of_bowls_l782_782716

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l782_782716


namespace find_smallest_integer_l782_782283

theorem find_smallest_integer (x : ℤ) (h : x + 5 < 3x - 9) : x ≥ 8 :=
by 
  -- The proof steps would typically follow here, but are omitted as per the instructions.
  sorry

end find_smallest_integer_l782_782283


namespace non_adjacent_girls_arrangement_l782_782821

theorem non_adjacent_girls_arrangement : 
  let boys := 4
  let girls := 2
  let arrangements := boys.factorial * (5.choose girls) * girls.factorial
  arrangements = 480 :=
by
  let boys := 4
  let girls := 2
  let arrangements := boys.factorial * (5.choose girls) * girls.factorial
  have factorial_4 : boys.factorial = 24 := by sorry
  have arrangements : arrangements = 24 * 20 := by
    rw [factorial_4, ← mul_assoc, mul_comm 5 4, mul_assoc]
    have girls_factorial : girls.factorial = 2 := by sorry
    have choose_5_2 : (5.choose girls) = 10 := by sorry
    have factor_comb : (5.choose girls) * girls.factorial = 20 := by
      rw [choose_5_2, girls_factorial]
    rw factor_comb
  exact sorry

end non_adjacent_girls_arrangement_l782_782821


namespace line_divides_perimeter_into_ratio_l782_782214

def divides_square_diagonal (A B C D M : Point) (s : ℝ) (AM_MC_ratio : ℝ) : Prop :=
  dist A M = 3/5 * dist A C ∧ dist M C = 2/5 * dist A C

def area_ratio (A B C D M : Point) (ratio : ℝ) : Prop :=
  area_partition A B C D M (9/20) ∧ area_partition A B C D M (11/20)

theorem line_divides_perimeter_into_ratio 
  (A B C D M : Point) (s : ℝ)
  (h1 : divides_square_diagonal A B C D M s 3/2)
  (h2 : area_ratio A B C D M 9/11)
  (is_square : is_square A B C D s) :
  divides_perimeter A B C D M (19/21) :=
by
  sorry

end line_divides_perimeter_into_ratio_l782_782214


namespace exists_zero_in_interval_l782_782937

def f (x : ℝ) : ℝ := x^3 - (1 / 2)^(x - 2)

theorem exists_zero_in_interval :
  ∃ c ∈ set.Ioo 1 2, f c = 0 :=
by
  have f1 : f 1 < 0 := by sorry
  have f2 : f 2 > 0 := by sorry
  exact intermediate_value_Theorem f (1 : ℝ) 2 f1 f2

end exists_zero_in_interval_l782_782937


namespace quadratic_formula_correct_solve_quadratic_by_completing_square_l782_782284

theorem quadratic_formula_correct (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b^2 - 4 * a * c > 0) :
  ∃ x : ℝ, x^2 + (b / a) * x + (c / a) = 0 ∧ 
  x = (-b + real.sqrt (b^2 - 4 * a * c)) / (2 * a) ∨
  x = (-b - real.sqrt (b^2 - 4 * a * c)) / (2 * a) :=
sorry

theorem solve_quadratic_by_completing_square :
  ∃ x₁ x₂ : ℝ, (x₁ = 6 ∧ x₂ = -4) ∨ (x₁ = -4 ∧ x₂ = 6) ∧ 
  (x₁ - 1) ^ 2 = 25 ∧ (x₂ - 1) ^ 2 = 25 :=
sorry

end quadratic_formula_correct_solve_quadratic_by_completing_square_l782_782284


namespace distance_focus_asymptote_l782_782538

noncomputable def focus := (Real.sqrt 6 / 2, 0)
def asymptote (x y : ℝ) := x - Real.sqrt 2 * y = 0
def hyperbola (x y : ℝ) := x^2 - 2 * y^2 = 1

theorem distance_focus_asymptote :
  let d := (Real.sqrt 6 / 2, 0)
  let A := 1
  let B := -Real.sqrt 2
  let C := 0
  let numerator := abs (A * d.1 + B * d.2 + C)
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator = Real.sqrt 2 / 2 :=
sorry

end distance_focus_asymptote_l782_782538


namespace imaginary_part_of_z_times_conj_z_plus_i_l782_782031

-- Defining the given condition.
def z : ℂ := (2 : ℝ) - (1 : ℝ) * Complex.i

-- Define the proof problem.
theorem imaginary_part_of_z_times_conj_z_plus_i : Complex.im (z * (Complex.conj z + Complex.i)) = 2 :=
by
  sorry

end imaginary_part_of_z_times_conj_z_plus_i_l782_782031


namespace enclosed_area_calculation_l782_782405

def side_length_of_pentagon := 2
def arc_length := (3 * Real.pi) / 4
def area_of_enclosed_figure : ℝ := 4.8284 + 3 * Real.pi

theorem enclosed_area_calculation :
  let s := side_length_of_pentagon in
  let L := arc_length in
  let θ := arc_length in
  let r := 1 in -- derived as L / θ
  let sector_area := θ / (2 * Real.pi) * Real.pi * r ^ 2 in
  let total_sector_area := 8 * sector_area in
  let pentagon_area := 4.8284 in -- precomputed
  pentagon_area + total_sector_area = area_of_enclosed_figure :=
by
  sorry

end enclosed_area_calculation_l782_782405


namespace choose_three_cooks_from_ten_l782_782128

theorem choose_three_cooks_from_ten : 
  (nat.choose 10 3) = 120 := 
by
  sorry

end choose_three_cooks_from_ten_l782_782128


namespace variance_same_for_shifted_samples_l782_782926

def sampleA : List ℝ := [72, 73, 76, 76, 77, 78, 78]
def sampleB : List ℝ := sampleA.map (λ x => x + 2)

def variance (l : List ℝ) : ℝ :=
  let mean := l.sum / l.length
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

theorem variance_same_for_shifted_samples :
  variance sampleA = variance sampleB :=
by
  sorry

end variance_same_for_shifted_samples_l782_782926


namespace not_divisible_by_97_l782_782975

theorem not_divisible_by_97 (k : ℤ) (h : k ∣ (99^3 - 99)) : k ≠ 97 :=
sorry

end not_divisible_by_97_l782_782975


namespace distance_circle_center_to_line_l782_782137

def line := {p : ℝ × ℝ | p.1 + p.2 = 6}
def circle_center := (0, 2)
def distance_from_center_to_line := Real.sqrt ((circle_center.1 + circle_center.2 - 6)^2 / ((1 : ℝ)^2 + (1 : ℝ)^2))

theorem distance_circle_center_to_line : distance_from_center_to_line = 2 * Real.sqrt 2 := by
  sorry

end distance_circle_center_to_line_l782_782137


namespace sam_found_seashells_l782_782675

def seashells_given : Nat := 18
def seashells_left : Nat := 17
def seashells_found : Nat := seashells_given + seashells_left

theorem sam_found_seashells : seashells_found = 35 := by
  sorry

end sam_found_seashells_l782_782675


namespace number_of_bowls_l782_782733

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l782_782733


namespace cost_difference_per_square_inch_l782_782285

theorem cost_difference_per_square_inch (width1 height1 width2 height2 : ℕ) (cost1 cost2 : ℕ)
  (h_size1 : width1 = 24 ∧ height1 = 16)
  (h_cost1 : cost1 = 672)
  (h_size2 : width2 = 48 ∧ height2 = 32)
  (h_cost2 : cost2 = 1152) :
  (cost1 / (width1 * height1) : ℚ) - (cost2 / (width2 * height2) : ℚ) = 1 := 
by
  sorry

end cost_difference_per_square_inch_l782_782285


namespace christian_age_in_eight_years_l782_782401

theorem christian_age_in_eight_years (b c : ℕ)
  (h1 : c = 2 * b)
  (h2 : b + 8 = 40) :
  c + 8 = 72 :=
sorry

end christian_age_in_eight_years_l782_782401


namespace nonnegative_difference_roots_eq_12_l782_782760

theorem nonnegative_difference_roots_eq_12 :
  ∀ (x : ℝ), (x^2 + 40 * x + 300 = -64) →
  ∃ (r₁ r₂ : ℝ), (x^2 + 40 * x + 364 = 0) ∧ 
  (r₁ = -26 ∧ r₂ = -14)
  ∧ (|r₁ - r₂| = 12) :=
by
  sorry

end nonnegative_difference_roots_eq_12_l782_782760


namespace production_increase_l782_782336

theorem production_increase (h1 : ℝ) (h2 : ℝ) (h3 : h1 = 0.75) (h4 : h2 = 0.5) :
  (h1 + h2 - 1) = 0.25 := by
  sorry

end production_increase_l782_782336


namespace solution_set_inequality_l782_782916

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_non_neg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x
def f_neg_half_eq_zero (f : ℝ → ℝ) : Prop := f (-1/2) = 0

-- Problem statement
theorem solution_set_inequality (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_decreasing : is_decreasing_on_non_neg f) 
  (hf_neg_half_zero : f_neg_half_eq_zero f) : 
  {x : ℝ | f (Real.logb (1/4) x) < 0} = {x | x > 2} ∪ {x | 0 < x ∧ x < 1/2} :=
  sorry

end solution_set_inequality_l782_782916


namespace sylvia_carla_together_time_l782_782662

-- Define the conditions
def sylviaRate := 1 / 45
def carlaRate := 1 / 30

-- Define the combined work rate and the time taken to complete the job together
def combinedRate := sylviaRate + carlaRate
def timeTogether := 1 / combinedRate

-- Theorem stating the desired result
theorem sylvia_carla_together_time : timeTogether = 18 := by
  sorry

end sylvia_carla_together_time_l782_782662


namespace unique_digit_sum_is_21_l782_782606

theorem unique_digit_sum_is_21
  (Y E M T : ℕ)
  (YE ME : ℕ)
  (HT0 : YE = 10 * Y + E)
  (HT1 : ME = 10 * M + E)
  (H1 : YE * ME = 999)
  (H2 : Y ≠ E)
  (H3 : Y ≠ M)
  (H4 : Y ≠ T)
  (H5 : E ≠ M)
  (H6 : E ≠ T)
  (H7 : M ≠ T)
  (H8 : Y < 10)
  (H9 : E < 10)
  (H10 : M < 10)
  (H11 : T < 10) :
  Y + E + M + T = 21 :=
sorry

end unique_digit_sum_is_21_l782_782606


namespace no_winning_strategy_exceeds_half_probability_l782_782329

-- Provided conditions
def well_shuffled_standard_deck : Type := sorry -- Placeholder for the deck type

-- Statement of the problem
theorem no_winning_strategy_exceeds_half_probability :
  ∀ strategy : (well_shuffled_standard_deck → ℕ → bool),
    let r := 26 in -- Assuming a standard deck half red cards (26 red)
    let b := 26 in -- and half black cards (26 black)
    let P_win := (r : ℝ) / (r + b) in       
    P_win ≤ 0.5 :=
by
  sorry

end no_winning_strategy_exceeds_half_probability_l782_782329


namespace domain_of_f_l782_782974

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3) + Real.log (x + 1)

theorem domain_of_f :
  ∀ x : ℝ, (0 ≤ x + 3) → (0 < x + 1) → (x ∈ Ioi -1) :=
by
  intros x hx1 hx2
  sorry

end domain_of_f_l782_782974


namespace distance_parallel_lines_l782_782065

noncomputable def distance_between_parallel_lines (
  m : Real
) : Real :=
  |((m - 2) - 8) / Real.sqrt(1 + 4)|

theorem distance_parallel_lines (m : Real) (h1 : 1 / m = (m + 1) / 2) (h2 : m = 1) :
  distance_between_parallel_lines m = 9 * Real.sqrt 5 / 5 :=
by
  sorry

end distance_parallel_lines_l782_782065


namespace choose_3_out_of_10_l782_782123

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 :=
by
  sorry

end choose_3_out_of_10_l782_782123


namespace evaluate_exp_power_l782_782434

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l782_782434


namespace domain_of_function_l782_782004

def quadratic_inequality (x : ℝ) : Prop := -8 * x^2 - 14 * x + 9 ≥ 0

theorem domain_of_function :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 9 / 8} :=
by
  -- The detailed proof would go here, but we're focusing on the statement structure.
  sorry

end domain_of_function_l782_782004


namespace eval_expr_l782_782425

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l782_782425


namespace age_of_first_replaced_man_is_157_l782_782682

noncomputable def age_of_first_replaced_man 
    (avg_age_before: ℝ) 
    (avg_age_of_new_men: ℝ) 
    (age_second_replaced_man: ℝ) 
    (num_men: ℕ) 
    (increase_in_avg_age: ℝ) 
    (total_age_after: ℝ) : ℝ :=
    let total_age_before := avg_age_before * ↑num_men in
    let total_age_increase := increase_in_avg_age * ↑num_men in
    let age_of_new_men := avg_age_of_new_men * 2 in
    let total_age_of_replaced_men := total_age_after - age_of_new_men in
    total_age_of_replaced_men - age_second_replaced_man

theorem age_of_first_replaced_man_is_157 :
  age_of_first_replaced_man 28 30 23 8 2 240 = 157 := 
by 
  sorry

end age_of_first_replaced_man_is_157_l782_782682


namespace number_of_combinations_of_planets_is_1141_l782_782961

def number_of_combinations_of_planets : ℕ :=
  (if 7 ≥ 7 ∧ 8 ≥2 then Nat.choose 7 7 * Nat.choose 8 2 else 0) + 
  (if 7 ≥ 6 ∧ 8 ≥ 4 then Nat.choose 7 6 * Nat.choose 8 4 else 0) + 
  (if 7 ≥ 5 ∧ 8 ≥ 6 then Nat.choose 7 5 * Nat.choose 8 6 else 0) +
  (if 7 ≥ 4 ∧ 8 ≥ 8 then Nat.choose 7 4 * Nat.choose 8 8 else 0)

theorem number_of_combinations_of_planets_is_1141 :
  number_of_combinations_of_planets = 1141 :=
by
  sorry

end number_of_combinations_of_planets_is_1141_l782_782961


namespace terminating_fraction_count_l782_782020

theorem terminating_fraction_count : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 529 ∧ n % 53 = 0 }.card = 9 := 
sorry

end terminating_fraction_count_l782_782020


namespace problem_1_problem_2_problem_3_l782_782057

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)
noncomputable def g (x a : ℝ) : ℝ := a * x
noncomputable def F (x a : ℝ) : ℝ := g x a * f x a

theorem problem_1 (a : ℝ) (h : a = 1) :
  ∃ x, f x a = g x a ∧ x = 1 / 2 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x, f x a = g x a → x ∈ set.Ioi 0) →
  a ∈ set.Icc (-1:ℝ) 0 ∪ set.Ioc -a_proof0:ℝ a x _0(0:ℝ ?) i e o α Xe : i ) 0 0 c  ( : \ . propos .hhh{) 1 ℝset.Ioo sorry 

theorem problem_3 (y : ℝ) (a : ℝ) (h : a > 0) :
  (∀ x ∈ set.Icc (1:ℝ) (2:ℝ), F y x = F y x ∧ 
    (if 0 < a ∧ a ≤ 1 then ∀ x ∈ set.Icc (1:ℝ) (2:ℝ), F x a ≤ 4 * a - 2 * a ^ 2 else 
    if 1 < a ∧ a ≤ 2 then ∀ x ∈ set.Icc (1:ℝ) (2:ℝ), F x a ≤ if 1 < a ∧ a < 5 / 3 then 4 * a - 2 * a ^ 2 else a ^ 2 - a else 
    if 2 < a ∧ a ≤ 4 then ∀ x ∈ set.Icc (1:ℝ) (2:ℝ), F x a ≤ 3 * a ^ 3 / 4 else 
    if a > 4 then ∀ x ∈ set.Icc (1:ℝ) (2:ℝ), F x a ≤ 2 * a ^ 2 - 4 * a)) :=
sorry

end problem_1_problem_2_problem_3_l782_782057


namespace find_matrix_and_curve_transformation_l782_782058

variables {a b : ℝ}

def matrix_transform (A : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  A a 1 0 b

def point_transform (A : ℝ → ℝ → ℝ → ℝ → Prop) (x1 y1 x2 y2 : ℝ) : Prop :=
  ∃ (a b : ℝ), A a 1 0 b ∧ (a * x1 + 1 * y1 = x2) ∧ (0 * x1 + b * y1 = y2)

theorem find_matrix_and_curve_transformation :
  (matrix_transform (λ a 1 0 b, True)) →
  point_transform (λ a 1 0 b, True) 1 1 2 2 →
  ∃ (a b : ℝ), a = 1 ∧ b = 2 ∧
  (∀ (x y : ℝ), (x^2 + y^2 = 1) → ((x - y/2) - y/2 * (x - y/2) + (y/2)^2 = 1)) :=
by
  intro h_A h_transform
  use 1, 2
  split
  { refl }
  split
  { refl }
  intro x y h_curve
  sorry

end find_matrix_and_curve_transformation_l782_782058


namespace number_of_bowls_l782_782739

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l782_782739


namespace population_decreases_l782_782746

theorem population_decreases (P_0 : ℝ) (k : ℝ) (n : ℕ) (hP0 : P_0 > 0) (hk : -1 < k ∧ k < 0) : 
  P_0 * (1 + k)^n * k < 0 → P_0 * (1 + k)^(n + 1) < P_0 * (1 + k)^n := by
  sorry

end population_decreases_l782_782746


namespace find_central_angle_l782_782680

open Real

-- Given conditions as Lean definitions
def area_of_sector : ℝ := 47.77142857142857
def radius : ℝ := 12
def sector_area_formula (θ : ℝ) : ℝ := (θ / 360) * π * radius^2

-- Lean statement version of the proof
theorem find_central_angle (θ : ℝ) (h : sector_area_formula θ = area_of_sector) : θ ≈ 38.197 :=
sorry

end find_central_angle_l782_782680


namespace correct_propositions_l782_782928

-- Definitions of the propositions
def prop1 (p q : Prop) : Prop := ¬(p ∧ q) → ¬p ∧ ¬q
def prop2 (a b : ℝ) : Prop := ¬(a > b → 3^a > 3^b - 1) = (a ≤ b → 3^a ≤ 3^b - 1)
def prop3 : Prop := ¬(∀ x : ℝ, x^2 + 1 ≥ 0) = ∃ x₀ : ℝ, x₀^2 + 1 < 0
def prop4 (a : ℝ) : Prop := (a ≥ 0) ↔ (∃ x : ℝ, a*x^2 + x + 1 ≥ 0)

-- The representation of the problem statement: prove the correct propositions are prop2 and prop3
theorem correct_propositions : ({2, 3} = {i | let ps := [prop1 = λ (p q : Prop), prop2 = λ (a b : ℝ), prop3, prop4 = λ (a : ℝ)] in ps[i]}) :=
by {
  sorry
}

end correct_propositions_l782_782928


namespace parabola_vertex_on_x_axis_l782_782095

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + c = 0) → c = 1 := by
  sorry

end parabola_vertex_on_x_axis_l782_782095


namespace arithmetic_sequence_difference_l782_782676

noncomputable def solve_common_difference (b d : ℕ) : Prop :=
  ∑ i in finset.range 50, (b + i * d) = 150 ∧
  ∑ i in (finset.range 50).map (embedding.of_preorder_inc f_pred 51), (b + (i + 50) * d) = 300

theorem arithmetic_sequence_difference (b d: ℕ) (h₁: ∑ i in finset.range 50, (b + i * d) = 150)
  (h₂: ∑ i in finset.range 51 101, (b + (i + 50) * d) = 300) : b + d = 1 / 224 :=
  sorry

end arithmetic_sequence_difference_l782_782676


namespace quadratic_sum_r_s_l782_782226

/-- Solve the quadratic equation and identify the sum of r and s 
from the equivalent completed square form (x + r)^2 = s. -/
theorem quadratic_sum_r_s (r s : ℤ) :
  (∃ r s : ℤ, (x - r)^2 = s → r + s = 11) :=
sorry

end quadratic_sum_r_s_l782_782226


namespace find_m_l782_782609

noncomputable def m : ℤ := sorry

-- Conditions as definitions
def c1 : ℤ := 5
def c2 : ℤ := 9
def c3 : ℤ := 7
def c4 : ℤ := 12
def b3_1 : ℤ := m + c1
def b3_2 : ℤ := c1 + c2
def b3_3 : ℤ := c2 + c3
def b3_4 : ℤ := c3 + c4
def b2_1 : ℤ := b3_1 + b3_2
def b2_2 : ℤ := b3_2 + b3_3
def b2_3 : ℤ := b3_3 + b3_4
def b1 : ℤ := b2_1 + b2_3

-- Theorem to prove m == 1 given conditions
theorem find_m : b1 = 54 → m = 1 :=
by
  sorry

end find_m_l782_782609


namespace triangle_AC_length_l782_782611

theorem triangle_AC_length (A B C : ℝ) (hB : ∠ABC = 90) (h_tanA : Real.tan A = 4/3) (hAB : AB = 3) : AC = 5 := 
by 
  sorry

end triangle_AC_length_l782_782611


namespace sum_of_possible_values_l782_782184

theorem sum_of_possible_values (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end sum_of_possible_values_l782_782184


namespace minimum_value_expression_l782_782007

theorem minimum_value_expression : ∃ (x y : ℝ), ∀ x' y' : ℝ, x' ^ 2 + 4 * x' * sin y' - 4 * cos y' ^ 2 ≥ x ^ 2 + 4 * x * sin y - 4 * cos y ^ 2 ∧ x ^ 2 + 4 * x * sin y - 4 * cos y ^ 2 = -4 :=
by {
  sorry
}

end minimum_value_expression_l782_782007


namespace perimeter_of_rhombus_in_rectangle_l782_782697

noncomputable def perimeter_AFCE : ℝ :=
  89.44

theorem perimeter_of_rhombus_in_rectangle :
  ∀ (A B C D F E : Type → Prop),
    rhombus AFCE ∧ rectangle ABCD ∧
    width_AB_AD == 20 ∧ length_AD_AB == 25 →
    perimeter_AFCE == 89.44 :=
begin
  sorry
end

end perimeter_of_rhombus_in_rectangle_l782_782697


namespace ellipse_properties_l782_782524

noncomputable theory
open_locale classical

-- Define the constants a, b assuming the condition a > b > 0 and their values.
def a : ℝ := sqrt 2
def b : ℝ := 1
def c : ℝ := 1

-- Define the equations
def ellipse_eq(x y : ℝ) : Prop := x^2 / (a^2) + y^2 / (b^2) = 1

def circle_eq(x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 18

def distance_cond : Prop := 2 * c < 4

-- Left focus of the ellipse F(-1, 0)
def left_focus : ℝ × ℝ := (-1, 0)

-- Define |AB| condition
def ab_length(m : ℝ) (y1 y2 : ℝ) : ℝ := 
  (sqrt (1 + m^2)) * (sqrt ((4 * m^2) / ((m^2 + 2)^2) + (4 / (m^2 + 2))))

theorem ellipse_properties :
  (a = sqrt 2) ∧ (b = 1) ∧ (c = 1) ∧ distance_cond →
  (∀ x y, ellipse_eq x y) ∧ (|AB| = (3 * sqrt 2) / 2) :=
sorry

end ellipse_properties_l782_782524


namespace evaluate_exp_power_l782_782437

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l782_782437


namespace cross_correlation_XY_cross_correlation_YX_l782_782669

noncomputable def K_X : (ℝ × ℝ) → ℝ := sorry
def X : ℝ → ℝ := sorry
def Y (t : ℝ) : ℝ := ∫ s in 0..t, X s

theorem cross_correlation_XY (t1 t2 : ℝ) :
  (∫ s in 0..t2, K_X (t1, s)) = ∫ s in 0..t2, K_X (t1, s) := sorry

theorem cross_correlation_YX (t1 t2 : ℝ) :
  (∫ s in 0..t1, K_X (s, t2)) = ∫ s in 0..t1, K_X (s, t2) := sorry

end cross_correlation_XY_cross_correlation_YX_l782_782669


namespace cos_C_value_l782_782985

variables (A B C : ℝ) (a b c : ℝ)
variable (triangle_ABC : Triangle A B C a b c)

open Triangle

def sin_geometric_sequence (triangle_ABC : Triangle A B C a b c) : Prop :=
  let sin_A := real.sin A
  let sin_B := real.sin B
  let sin_C := real.sin C
  (sin_A * sin_C = sin_A * sin_A) ∧ (sin_C * sin_B = sin_A * sin_B)
  
theorem cos_C_value
  (triangle_ABC : Triangle A B C a b c)
  (b_eq_2a : b = 2*a)
  (sin_geom_seq : sin_geometric_sequence triangle_ABC):
  cos C = 3/4 :=
  sorry

end cos_C_value_l782_782985


namespace explicit_expression_of_odd_function_l782_782559

/-- f is an odd function, and for x > 0, f(x) = x^2 - x - 1.
    We need to prove that for x < 0, f(x) = -x^2 - x + 1. -/
theorem explicit_expression_of_odd_function (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 < x → f x = x^2 - x - 1) :
  ∀ x, x < 0 → f x = -x^2 - x + 1 :=
begin
  sorry
end

end explicit_expression_of_odd_function_l782_782559


namespace inverse_function_log_base2_l782_782973

theorem inverse_function_log_base2 (f : ℝ → ℝ) (hf : ∀ x, x > 5 → f x = log (x - 1) / log 2) :
  ∀ y, y > 2 → (∃ x, (x > 5 ∧ f x = y) ∧ x = 2^y + 1) :=
by
  sorry

end inverse_function_log_base2_l782_782973


namespace octagon_area_l782_782282

-- Define the octagon's side information and surrounding conditions
def side_length : ℝ := 1
def height : ℝ := 4
def num_triangles : ℕ := 4
def rect_width : ℝ := 5
def rect_length : ℝ := 8

-- Define the areas of individual components
def triangle_area (b h : ℝ) : ℝ := (1/2) * b * h
def triangles_area : ℝ := num_triangles * triangle_area side_length height
def rect_area (w l : ℝ) : ℝ := w * l

-- State the main theorem
theorem octagon_area : rect_area rect_width rect_length - triangles_area = 32 :=
by
  sorry

end octagon_area_l782_782282


namespace number_of_bowls_l782_782735

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l782_782735


namespace constant_term_of_expansion_eq_80_l782_782503

theorem constant_term_of_expansion_eq_80 (a : ℝ) (h1 : 0 < a)
  (h2 : (∑ k in Finset.range 6, Nat.choose 5 k * real.pow a k * real.pow (2.5 : ℝ) (10 - 2.5 * k : ℝ)) = 80) :
  a = 2 :=
by
  sorry

end constant_term_of_expansion_eq_80_l782_782503


namespace find_alpha_and_beta_l782_782067

theorem find_alpha_and_beta
  (α β : ℝ)
  (hα : α ∈ Ioo (π / 2) π)
  (hα_perp : (sin α * 1 + cos α * sqrt 3) = 0)
  (hβ : β ∈ Ioo (π / 6) (π / 2))
  (h_sin : sin (α - β) = 3 / 5) :
  α = 2 * π / 3 ∧ β = Real.arcsin ((4 * sqrt 3 + 3) / 10) :=
by
  sorry

end find_alpha_and_beta_l782_782067


namespace events_mutually_exclusive_l782_782266

-- Definitions for the people and balls
inductive Person : Type
| A : Person
| B : Person
| C : Person

inductive Ball : Type
| Red : Ball
| Black : Ball
| White : Ball

def receives_white_ball (p : Person) (assignment : Person → Ball) : Prop :=
  assignment p = Ball.White

theorem events_mutually_exclusive (assignment : Person → Ball) :
  ∀ (p1 p2 : Person), p1 ≠ p2 → (receives_white_ball p1 assignment) → ¬(receives_white_ball p2 assignment) := 
by
  intros p1 p2 hne h1 h2
  rw [receives_white_ball] at h1 h2
  have h : p1 = p2 := by
    rw [←h1, ←h2]
  contradiction

end events_mutually_exclusive_l782_782266


namespace right_triangle_area_l782_782806

theorem right_triangle_area (a b : ℝ) (H₁ : a = 3) (H₂ : b = 5) : 
  1 / 2 * a * b = 7.5 := by
  rw [H₁, H₂]
  norm_num

end right_triangle_area_l782_782806


namespace exists_fixed_point_L_collinear_l782_782539

-- Definitions based on conditions provided in the problem.
variables {Γ : Type*} [metric_space Γ] [normed_group Γ] [normed_space ℝ Γ]
variables (A B C P D E U V M N L : Γ)

-- Assumptions based on the problem's conditions.
variables (outside_circle_A : metric.dinstance A Γ > 0)
variables (tangent_AB : ∀ (x : Γ), x ∈ metric.d_ball (metric.add A B) r.solve)
variables (tangent_AC : ∀ (x : Γ), x ∈ metric.d_ball (metric.add A C) r.solve)
variables (arc_P : P ∈ metric.d_ball (metric.add B C) r.solve)
variables (tangent_DP : D ∈ metric.d_ball (metric.add P B) r.solve ∧ E ∈ metric.d_ball (metric.add P C) r.solve)
variables (intersect_UV : U ∈ metric.d_ball (metric.add B P) r.solve ∧ V ∈ metric.d_ball (metric.add C P) r.solve)
variables (perpendicular_PM : M ∈ affine_span ℝ P (metric.d_ball A B))
variables (perpendicular_PN : N ∈ affine_span ℝ P (metric.d_ball A C))

-- Statement to prove collinearity based on the conditions.
theorem exists_fixed_point_L_collinear (M N L : Γ) : ∃ L, collinear ℝ {M, N, L} :=
begin
  sorry
end

end exists_fixed_point_L_collinear_l782_782539


namespace circle_numbers_opposite_l782_782201

def is_opposite (n : ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k ∧ (b = a + k ∨ a = b + k)

theorem circle_numbers_opposite :
  ∃ n : ℕ, is_opposite n 5 14 ∧ n = 18 :=
by
  use 18
  split
  { unfold is_opposite,
    use 9,
    split; linarith }
  { refl }

end circle_numbers_opposite_l782_782201


namespace largest_number_is_A_l782_782414

noncomputable def numA : ℝ := 4.25678
noncomputable def numB : ℝ := 4.2567777 -- repeating 7
noncomputable def numC : ℝ := 4.25676767 -- repeating 67
noncomputable def numD : ℝ := 4.25675675 -- repeating 567
noncomputable def numE : ℝ := 4.25672567 -- repeating 2567

theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE := by
  sorry

end largest_number_is_A_l782_782414


namespace find_center_of_circle_l782_782471

noncomputable def circle_eq : ℝ → ℝ → ℝ := λ x y, x^2 - 8 * x + y^2 + 4 * y + 3

theorem find_center_of_circle :
  ∃ h k : ℝ, (∀ x y : ℝ, circle_eq x y = (x - h)^2 + (y - k)^2 - 17) ∧ (h = 4) ∧ (k = -2) :=
sorry

end find_center_of_circle_l782_782471


namespace natural_numbers_partition_l782_782858

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l782_782858


namespace eval_exp_l782_782420

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l782_782420


namespace partition_with_sum_square_l782_782852

def sum_is_square (a b : ℕ) : Prop := ∃ k : ℕ, a + b = k * k

theorem partition_with_sum_square (n : ℕ) (h : n ≥ 15) :
  ∀ (s₁ s₂ : finset ℕ), (∅ ⊂ s₁ ∪ s₂ ∧ s₁ ∩ s₂ = ∅ ∧ (∀ x ∈ s₁ ∪ s₂, x ∈ finset.range (n + 1))) →
  (∃ a b : ℕ, a ≠ b ∧ (a ∈ s₁ ∧ b ∈ s₁ ∨ a ∈ s₂ ∧ b ∈ s₂) ∧ sum_is_square a b) :=
by sorry

end partition_with_sum_square_l782_782852


namespace AliBaba_coin_distribution_l782_782367

-- Defining the cave as an m x n grid
variable (m n : ℕ) (cave : Finₓ m → Finₓ n → Bool) -- cave as a function from indices to Bool representing black (True) or white (False)

-- Condition definitions
def isCheckerboard (cave : Finₓ m → Finₓ n → Bool) := 
  ∀ i j, cave i j = (if (i + j) % 2 = 0 then true else false)

def walksToAnyAdjacentCell (i j : Finₓ m) (c k : Finₓ n) : Prop :=
  (i < m - 1 ∧ i + 1 = c ∧ j = k) ∨ 
  (i > 0 ∧ i - 1 = c ∧ j = k) ∨ 
  (j < n - 1 ∧ j + 1 = k ∧ i = c) ∨ 
  (j > 0 ∧ j - 1 = k ∧ i = c)

-- Initial state (Ali Baba starts in a corner)
def initialState (cave : Finₓ m → Finₓ n → Bool) : Prop :=
  cave 0 0 -- Starting at (0, 0)

-- Walks and coin adjustments
def canPlaceOrPickCoin (coins : Finₓ m → Finₓ n → ℕ) (i j : Finₓ m) : Prop := 
  coins i j = coins i j + 1 ∨ coins i j = coins i j - 1
  
theorem AliBaba_coin_distribution (m n : ℕ) (cave : Finₓ m → Finₓ n → Bool) 
(h_check : isCheckerboard cave) 
(initial : initialState cave) 
(walk_adj : ∀ i j c k, walksToAnyAdjacentCell i j c k) 
(adj_move : ∀ coins i j, canPlaceOrPickCoin coins i j) :
∃ coins : Finₓ m → Finₓ n → ℕ, 
  (∀ i j, cave i j → coins i j = 1) ∧ 
  (∀ i j, ¬ cave i j → coins i j = 0) := 
sorry

end AliBaba_coin_distribution_l782_782367


namespace locus_of_Q_on_OA_l782_782917

variables {A B C O P Q : Type} [Geometry A B C O] [Circumcenter O A B C]

-- Definitions and conditions:
def PointOnExtensionOA (P : Type) : Prop := ∃ (A : Type), P ∈ line A O -- P is on the extension of OA
def LineSymmetricPB (l : Type) (P B A : Type) : Prop := is_symmetric l (line P B) (line B A) -- l is symmetric to PB wrt BA
def LineSymmetricPC (h : Type) (P C A : Type) : Prop := is_symmetric h (line P C) (line C A) -- h is symmetric to PC wrt AC
def IntersectionLines (l h : Type) (Q : Type) : Prop := Q ∈ l ∧ Q ∈ h -- Q is intersection of l and h
def LocusOfQ (Q : Type) : Set Point := {q | q ∈ line O A} -- The locus of Q is on the segment OA

-- Main theorem:
theorem locus_of_Q_on_OA (P : Type) (l h : Type) (Q : Type) [PointOnExtensionOA P]
    [LineSymmetricPB l P B A] [LineSymmetricPC h P C A] [IntersectionLines l h Q] :
    LocusOfQ Q :=
sorry

end locus_of_Q_on_OA_l782_782917


namespace minimum_difference_b_a_l782_782511

theorem minimum_difference_b_a (a b : ℝ) (h : ∀ x ∈ Ioo 0 (π / 2), a * x < sin x ∧ sin x < b * x) :
  (1 - 2 / π) = b - a :=
sorry

end minimum_difference_b_a_l782_782511


namespace number_of_bowls_l782_782714

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l782_782714


namespace zeros_at_end_of_factorial_base_16_l782_782820

theorem zeros_at_end_of_factorial_base_16 :
  let fact_eight := (nat.factorial 8)
  in let base_sixteen := 16
  in let zeros := 2
  in (
    ∃ (k : ℕ), fact_eight = base_sixteen^zeros * k ∧ (k % base_sixteen ≠ 0)
  ) :=
by
  sorry

end zeros_at_end_of_factorial_base_16_l782_782820


namespace solve_estate_problem_l782_782649

def estate_problem (E : ℝ) : Prop :=
  ∃ (x : ℝ), 
  let daughter_share := 3 * x in
  let son_share := 2 * x in
  let wife_share := 3 * daughter_share in
  let cook_share := 800 in
  (daughter_share + son_share = 0.5 * E) ∧
  (wife_share = 3 * daughter_share) ∧
  (E = wife_share + daughter_share + son_share + cook_share) ∧
  (daughter_share + son_share = 0.5 * E) →
  E = 2000

theorem solve_estate_problem : estate_problem 2000 :=
by {
  -- proof goes here
  sorry
}

end solve_estate_problem_l782_782649


namespace theta_is_15_deg_l782_782568

open Real

-- Define the condition that for some θ, √3 * sin 15° = cos θ + sin θ holds true
def exists_theta_satisfying_condition : Prop := 
  ∃ θ : ℝ, (0 < θ ∧ θ < 90) ∧ (sqrt 3 * sin (15 * pi / 180) = cos (θ * pi / 180) + sin (θ * pi / 180))

-- State the theorem where we need to prove θ = 15° if the condition holds
theorem theta_is_15_deg (h : exists_theta_satisfying_condition) : ∃ θ : ℝ, θ = 15 :=
by
  cases h with θ hθ,
  use θ,
  sorry

end theta_is_15_deg_l782_782568


namespace race_finish_orders_count_l782_782070

theorem race_finish_orders_count :
  (∀ participants : Finset String, participants.card = 4 → 
    participants = { "Harry", "Ron", "Neville", "Hermione" }) →
  ∃ n : ℕ, n = 24 :=
by
  intros
  have fact_4 : 4! = 24 := rfl
  use 4!
  exact fact_4
  sorry

end race_finish_orders_count_l782_782070


namespace case_1_case_2_l782_782061

variable {n : ℕ}
variable {a : Fin n → ℕ}  -- For sequence of positive integers \(a_i\) where \(1 ≤ a₁ ≤ a₂ ≤ … ≤ aₙ\)
variable {x : Fin n → ℝ}  -- For sequence of real numbers \(x_i\)

noncomputable def Q (a : Fin n → ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ i, a i * (x i)^2 + 2 * ∑ i in Finset.range (n - 1), (x ⟨i⟩) * (x ⟨i + 1⟩)

-- Problem Statement 1: If \(a_2 \ge 2\), show \(Q > 0\) for all \(x_i \ne 0\)
theorem case_1 (h_ordered : ∀ i j, i ≤ j → a i ≤ a j) (h1 : a ⟨1⟩ ≥ 2) (hx : ∃ i, x i ≠ 0) : 
  Q a x > 0 :=
by
  sorry

-- Problem Statement 2: If \(a_2 < 2\), show there exists \(x_i\) not all zero for which \(Q \le 0\)
theorem case_2 (h_ordered : ∀ i j, i ≤ j → a i ≤ a j) (h2 : a ⟨1⟩ < 2) : 
  ∃ x : Fin n → ℝ, (∃ i, x i ≠ 0) ∧ Q a x ≤ 0 :=
by
  sorry

end case_1_case_2_l782_782061


namespace problem1_problem2_problem3_problem4_l782_782308

-- (1) Prove that the value of the expression is equal to 5/2
theorem problem1 : sqrt 9 + cbrt (-1) - sqrt 0 + sqrt (1 / 4) = 5 / 2 := by
  sorry

-- (2) Prove that the value of the expression is equal to sqrt 6 + 2sqrt 2
theorem problem2 : 3 * sqrt 6 + sqrt 2 - (2 * sqrt 6 - sqrt 2) = sqrt 6 + 2 * sqrt 2 := by
  sorry

-- (3) Prove that the given system of equations results in x = -1 and y = -4
theorem problem3 (x y : ℝ) (h1 : x = 3 + y) (h2 : 3 * x - 2 * y = 5) : x = -1 ∧ y = -4 := by
  sorry

-- (4) Prove that the given system of equations results in x = 1 and y = -3/2
theorem problem4 (x y : ℝ) (h1 : 5 * x + 2 * y = 2) (h2 : 3 * x + 4 * y = -3) : x = 1 ∧ y = -3 / 2 := by
  sorry

end problem1_problem2_problem3_problem4_l782_782308


namespace f_neg_three_l782_782177

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f(x) = f(-x)

axiom even_function_f : even_function f
axiom condition_two : ∀ x > 0, f(x + 2) = -2 * f(2 - x)
axiom f_neg_one : f(-1) = 4

theorem f_neg_three : f(-3) = -8 :=
by
  sorry

end f_neg_three_l782_782177


namespace product_inequality_l782_782305

theorem product_inequality 
  (n : ℕ) 
  (a : ℝ) 
  (x : Fin n → ℝ) 
  (s : ℝ)
  (hx : ∀ i, 0 < x i)
  (ha : 0 < a)
  (h_sum : (Finset.univ.sum x) = s)
  (h_ineq : s ≤ a) :
  (∏ i, (a + x i) / (a - x i)) ≥ ((n * a + s) / (n * a - s)) ^ n := 
by 
  sorry

end product_inequality_l782_782305


namespace sum_boundary_values_of_range_l782_782838

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 3 * x^2 + 6 * x)

theorem sum_boundary_values_of_range : 
  let c := 0
  let d := 1
  c + d = 1 :=
by
  sorry

end sum_boundary_values_of_range_l782_782838


namespace part1_part1_decreasing_part2_part3_l782_782030

noncomputable def f (x : ℝ) (a : ℝ) := (Real.log (1-x) + a * x ^ 2 + x)

theorem part1 (x : ℝ) : f x (1/2) = Real.log (1-x) + (1/2) * x ^ 2 + x := by
  sorry

theorem part1_decreasing (x : ℝ) : x ∈ Ioo (-∞) 1 → ((Real.log (1-x) + (1/2) * x ^ 2 + x) < (Real.log (1-y) + (1/2) * y ^ 2 + y)) -> x > y := by
  sorry

theorem part2 (a : ℝ) (x : ℝ) : 0 < a ∧ a ≤ 1 / 2 ∧ x ∈ Ioo (0 : ℝ) (1 : ℝ) → (Real.log (1-x) + a * x ^ 2 + x < 0) := by
  sorry

theorem part3 (n : ℕ) : 0 < n → (Real.log (1 + n) - (Finset.sum (Finset.range (n + 1)) (λ i, 1 / (i + 1 : ℝ))) > 1 - 1 / (2 * n)) := by
  sorry

end part1_part1_decreasing_part2_part3_l782_782030


namespace arc_product_l782_782206

/-- On a circle with n points, colored either red or white,
    the product of values assigned to the arcs depends only
    on the number of red points and white points. Specifically,
    if there are m red points, then the product is 2^(2m - n) -/
theorem arc_product (n m : ℕ) (h : m ≤ n) :
  ∃ product : ℝ, 
    (∀ order : (finset.univ.filter (λ i, i < m)).perm (finset.univ.filter (λ i, i ≥ m)),
     product = 2 ^ (2 * m - n)) :=
  sorry

end arc_product_l782_782206


namespace find_a_l782_782518

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a (a : ℝ) (h : a < 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
by
  -- Define f(1-a) and f(1+a)
  have h1 : f a (1 - a) = -(1 - a) - 2 * a :=
    if_neg ((by linarith : 1 - a ≥ 1))
  have h2 : f a (1 + a) = 2 * (1 + a) + a :=
    if_pos ((by linarith : 1 + a < 1))
  -- Equate f(1-a) and f(1+a)
  rw [h1, h2] at h_eq
  -- Proceed with solving the equation as per the problem's solution steps
  sorry

end find_a_l782_782518


namespace general_term_a_n_sum_S_n_l782_782921

variable (bn : ℕ → ℕ) (an : ℕ → ℕ)

-- Given conditions
axiom geom_seq : ∀ n : ℕ, bn (n + 1) = bn n * 2
axiom bn_def : ∀ n : ℕ, bn n = 2 ^ (an n - 1)
axiom a1_def : an 1 = 2
axiom a3_def : an 3 = 4

-- Proving the general term formula
theorem general_term_a_n : ∀ n : ℕ, an n = n + 1 := sorry

-- Proving the sum of the first n terms
theorem sum_S_n (n : ℕ) : 
  let S := (λ m, (an m) / (bn m)) in 
  ∑ i in Finset.range n, S (i + 1) = 3 - (n + 3) / (2 ^ n) := sorry

end general_term_a_n_sum_S_n_l782_782921


namespace liking_pattern_count_l782_782889

-- Define the problem conditions
def students : Type := {Kim, Alex, Nina, Mike}
def games : fin 5 -- Define the five games with a finite type of size 5 (0,1,2,3,4)

def likes (s : students) (g : games) : Prop := sorry -- Define the liking relation (TBD)
def not_all_liked (g : games) : Prop := ¬(likes Kim g ∧ likes Alex g ∧ likes Nina g ∧ likes Mike g)

def at_least_one_game_liked_by_three (s1 s2 s3 : students) (g : games) : Prop :=
  (likes s1 g ∧ likes s2 g ∧ likes s3 g) ∧ 
  ∀ s4 : students, (s4 ≠ s1 ∧ s4 ≠ s2 ∧ s4 ≠ s3) → ¬ likes s4 g

-- Main statement
theorem liking_pattern_count : 
  (∀ g : games, not_all_liked g) → 
  (∀ (s1 s2 s3 : students), ∃ g : games, at_least_one_game_liked_by_three s1 s2 s3 g) →
  ∃! (n : ℕ), n = 124 :=
sorry

end liking_pattern_count_l782_782889


namespace mod_inv_64_equality_l782_782912

theorem mod_inv_64_equality :
  (4 ^ -1 ≡ 57 [MOD 119]) → (64 ^ -1 ≡ 29 [MOD 119]) :=
by
  sorry

end mod_inv_64_equality_l782_782912


namespace sum_of_first_3n_terms_l782_782036

theorem sum_of_first_3n_terms (n a d : ℝ) 
  (h1 : n > 0)
  (S_n : ∑ i in finset.range n, (a + i * d) = 48)
  (S_2n : ∑ i in finset.range (2 * n), (a + i * d) = 60) :
  ∑ i in finset.range (3 * n), (a + i * d) = 36 :=
by
  sorry

end sum_of_first_3n_terms_l782_782036


namespace ellipse_equation_lambda_range_l782_782046

theorem ellipse_equation :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a^2 = 4 * b^2) ∧ (a * b = 1 / 2) ∧ (∀ x y : ℝ, 
  x^2 + 4 * y^2 = 1 ↔ (x / a)^2 + (y / b)^2 = 1) :=
sorry

theorem lambda_range (λ : ℝ) :
  (∃ m y1 y2 : ℝ, y1 = -2 * y2 ∧ (2 * (λ * m) * y2) = ((λ^2 - 1)(m^2 + 4)) ∧ 
  (8 * λ^2 * m^2 + 4 * λ^2 = 4)) ↔ (λ ∈ set.Icc(-1,-1/3) ∪ set.Icc(1/3,1)) :=
sorry

end ellipse_equation_lambda_range_l782_782046


namespace exponentiation_calculation_l782_782387

theorem exponentiation_calculation : 3000 * (3000 ^ 3000) ^ 2 = 3000 ^ 6001 := by
  sorry

end exponentiation_calculation_l782_782387


namespace winner_percentage_l782_782590

theorem winner_percentage (V_winner V_margin V_total : ℕ) (h_winner: V_winner = 806) (h_margin: V_margin = 312) (h_total: V_total = V_winner + (V_winner - V_margin)) :
  ((V_winner: ℚ) / V_total) * 100 = 62 := by
  sorry

end winner_percentage_l782_782590


namespace area_of_triangle_l782_782359

noncomputable def radius_of_circle := 8 / Real.pi
noncomputable def angle_theta := 22.5 * Real.pi / 180
noncomputable def angle_alpha := 5 * angle_theta * 180 / Real.pi
noncomputable def angle_beta := 5 * angle_theta * 180 / Real.pi
noncomputable def angle_gamma := 6 * angle_theta * 180 / Real.pi
noncomputable def a := (16 / Real.pi) * Real.cos (angle_theta)
noncomputable def b := (16 / Real.pi) * Real.cos (angle_theta)
noncomputable def c := (16 * Real.sqrt 2) / (2 * Real.pi)

theorem area_of_triangle : 
  let area := 1 / 2 * a * b * Real.sin angle_gamma in
  area = 32 * (2 * Real.sqrt 2 + 2) / (Real.pi * Real.pi) :=
sorry

end area_of_triangle_l782_782359


namespace partition_perfect_square_l782_782864

theorem partition_perfect_square (n : ℕ) (h : n ≥ 15) :
  ∀ A B : finset ℕ, disjoint A B → A ∪ B = finset.range (n + 1) →
  ∃ x y ∈ A ∨ ∃ x y ∈ B, x ≠ y ∧ (∃ k : ℕ, x + y = k^2) :=
begin
  sorry
end

end partition_perfect_square_l782_782864


namespace multiple_of_bees_l782_782197

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end multiple_of_bees_l782_782197


namespace six_circles_distance_relation_l782_782842

/--
Prove that for any pair of non-touching circles (among six circles where each touches four of the remaining five),
their radii \( r_1 \) and \( r_2 \) and the distance \( d \) between their centers satisfy 

\[ d^{2}=r_{1}^{2}+r_{2}^{2} \pm 6r_{1}r_{2} \]

("plus" if the circles do not lie inside one another, "minus" otherwise).
-/
theorem six_circles_distance_relation 
  (r1 r2 d : ℝ) 
  (h : ∀ i : Fin 6, i < 6 → ∃ c : ℝ, (c = r1 ∨ c = r2) ∧ ∀ j : Fin 6, j ≠ i → abs (c - j) ≠ d ) :
  d^2 = r1^2 + r2^2 + 6 * r1 * r2 ∨ d^2 = r1^2 + r2^2 - 6 * r1 * r2 := 
  sorry

end six_circles_distance_relation_l782_782842


namespace round_robin_victory_interval_l782_782589

open Int Nat

theorem round_robin_victory_interval
  (n k : ℕ)
  (players : Fin (2 * n + 1) → (Fin (2 * n + 1) → ℕ))
  (distinct_strengths : ∀ i j : Fin (2 * n + 1), i ≠ j → players i ≠ players j)
  (weaker_wins : ℕ)
  (h_weaker_wins : weaker_wins = k)
  (total_matches : (Fin (2 * n + 1)) → ℕ → ℕ → Prop)
  (h_total_matches : ∀ i j : Fin (2 * n + 1), total_matches i j 1 ∨ total_matches j i 1)
  (h_unique_result : ∀ i j : Fin (2 * n + 1), (total_matches i j 1 ∧ ¬ total_matches j i 1) ∨ (total_matches j i 1 ∧ ¬ total_matches i j 1))
  : ∃ (i : Fin (2 * n + 1)), (n : ℤ) - Int.sqrt (2 * k) ≤ (sum (λ j, if total_matches i j 1 then 1 else 0 : ℕ → ℕ).toℤ i) ∧ (sum (λ j, if total_matches i j 1 then 1 else 0 : ℕ → ℕ).toℤ i) ≤ (n : ℤ) + Int.sqrt (2 * k) :=
  sorry

end round_robin_victory_interval_l782_782589


namespace binom_sum_eq_l782_782673

noncomputable def binom := Nat.choose

theorem binom_sum_eq (m n : ℕ) (h : m ≤ n) :
  (∑ k in Finset.range (n + 1), if k ≥ m then binom n k * binom k m else 0) = 2^(n - m) * binom n m :=
by
  sorry

end binom_sum_eq_l782_782673


namespace total_eggs_l782_782262

noncomputable def total_eggs_in_all_containers (n : ℕ) (f l : ℕ) : ℕ :=
  n * (f * l)

theorem total_eggs (f l : ℕ) :
  (f = 14 + 20 - 1) →
  (l = 3 + 2 - 1) →
  total_eggs_in_all_containers 28 f l = 3696 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end total_eggs_l782_782262


namespace max_profit_l782_782365

def fixed_cost : ℝ := 5    -- Fixed annual cost in million yuan

def c (x : ℝ) : ℝ :=
if h : x < 80 then
  (1/2) * x^2 + 40 * x
else
  101 * x + 8100 / x - 2180

def revenue (x : ℝ) : ℝ := 100 * x    -- Revenue per unit

def profit (x : ℝ) : ℝ :=
if h : x < 80 then
  revenue x - c x - fixed_cost
else
  revenue x - c x - fixed_cost

theorem max_profit : profit 90 = 1500 :=
sorry

end max_profit_l782_782365


namespace integral_evaluation_l782_782416

noncomputable def integral_problem : ℝ :=
  ∫ x in -π/4 .. π/4, (2 * cos (x / 2) ^ 2 + tan x)

theorem integral_evaluation : integral_problem = π / 2 + sqrt 2 :=
  sorry

end integral_evaluation_l782_782416


namespace total_remaining_tomatoes_l782_782360

-- Definitions for initial tomato counts and picking rates.
def initial_stage1 : ℕ := 150
def initial_stage2 : ℕ := 200
def picked_initially_stage1 : ℕ := (1 / 5) * initial_stage1
def picked_initially_stage2 : ℕ := (1 / 4) * initial_stage2
def picked_after_week_stage1 : ℕ := 15
def picked_after_week_stage2 : ℕ := 25
def picked_following_week_stage1 : ℕ := 2 * picked_after_week_stage1
def picked_following_week_stage2 : ℕ := 2 * picked_after_week_stage2

-- Proof that the remaining tomatoes on the plant are 150.
theorem total_remaining_tomatoes : 
  let total_picked_stage1 := picked_initially_stage1 + picked_after_week_stage1 + picked_following_week_stage1 in
  let total_picked_stage2 := picked_initially_stage2 + picked_after_week_stage2 + picked_following_week_stage2 in
  let remaining_stage1 := initial_stage1 - total_picked_stage1 in
  let remaining_stage2 := initial_stage2 - total_picked_stage2 in
  (remaining_stage1 + remaining_stage2) = 150 := 
by 
  sorry

end total_remaining_tomatoes_l782_782360


namespace choose_three_cooks_from_ten_l782_782127

theorem choose_three_cooks_from_ten : 
  (nat.choose 10 3) = 120 := 
by
  sorry

end choose_three_cooks_from_ten_l782_782127


namespace natural_numbers_partition_l782_782856

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l782_782856


namespace problem_ab_cd_l782_782574

theorem problem_ab_cd
    (a b c d : ℝ)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (habcd : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
    (h2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (ab)^2012 - (cd)^2012 = -2012 := 
sorry

end problem_ab_cd_l782_782574


namespace distinguishable_large_triangles_count_l782_782749

/-- Problem Statement:
   Given an unlimited supply of congruent equilateral triangles made of colored paper,
   each having one of eight different solid colors with the same color on both sides,
   construct a large equilateral triangle using nine small equilateral triangles.
   Considering the translation, rotation, and reflection symmetries,
   prove that the number of distinguishable large equilateral triangles is 3584.
-/
theorem distinguishable_large_triangles_count :
  let colors := 8  -- number of different colors
  (∃ n : ℕ, n = 9 ∧ ∃ f : fin n → fin colors, ∃ h : is_distinguishable f, 
    count_distinguishable f h = 3584) :=
sorry

end distinguishable_large_triangles_count_l782_782749


namespace no_winning_strategy_l782_782332

theorem no_winning_strategy (r b : ℕ) (h1 : r + b = 52) (strategy : ℕ → bool) :
  ∀ k, (strategy k → (r / (r + b : ℝ) ≤ 0.5)) :=
by 
  sorry

end no_winning_strategy_l782_782332


namespace choose_3_out_of_10_l782_782122

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 :=
by
  sorry

end choose_3_out_of_10_l782_782122


namespace problem1_problem2_l782_782932

noncomputable def f (a b c x : ℝ) := (a * x^2 + b * x + c) * Real.exp x

def problem1_statement (a : ℝ) : Prop :=
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → 
    a * x^2 + (a - 1) * x - a ≤ 0) → 0 ≤ a ∧ a ≤ 1

def problem2_statement (a m : ℝ) : Prop :=
  a = 0 → 
  (∃ m, ∀ (x : ℝ), 
    2 * ((1 - x) * Real.exp x) + 4 * x * Real.exp x ≥ m * x + 1 ∧ 
    m * x + 1 ≥ -x^2 + 4 * x + 1) → m = 4

theorem problem1 (a : ℝ) (h0 : f a (-(1 + a)) 1 0 = 1) (h1 : f a (-(1 + a)) 1 1 = 0) : 
  problem1_statement a := 
begin
  intros, 
  sorry
end

theorem problem2 (a : ℝ) (h0 : f 0 (-1) 1 0 = 1) (h1 : f 0 (-1) 1 1 = 0) : 
  problem2_statement 0 4 := 
begin
  intros, 
  sorry
end

end problem1_problem2_l782_782932


namespace a_days_to_complete_work_l782_782317

theorem a_days_to_complete_work (
    B_days : ℕ,
    total_payment : ℕ,
    B_share : ℕ
) : 
    B_days = 20 → 
    total_payment = 1000 → 
    B_share = 600 → 
    A_days = 30 
:=
begin
  intros hB_days htotal_payment hB_share,
  have hB_rate := (1/20 : ℚ),
  have hshare_ratio := (400/600 : ℚ),
  have hwork_rate_ratio := (20 / A_days : ℚ),
  calc A_days
      = 30 : by sorry
end

end a_days_to_complete_work_l782_782317


namespace x_minus_p_eq_2_minus_2p_l782_782978

theorem x_minus_p_eq_2_minus_2p (x p : ℝ) (h1 : |x - 3| = p + 1) (h2 : x < 3) : x - p = 2 - 2 * p := 
sorry

end x_minus_p_eq_2_minus_2p_l782_782978


namespace equal_profits_l782_782209

theorem equal_profits (p q : ℝ) : 
  (1 + p/200)^2 = (1 + q/1200)^12 ↔ True :=
begin
  sorry
end

end equal_profits_l782_782209


namespace choose_3_out_of_10_l782_782120

theorem choose_3_out_of_10 : nat.choose 10 3 = 120 :=
by
  sorry

end choose_3_out_of_10_l782_782120


namespace ratio_of_volumes_l782_782314

noncomputable def volume_ratio_of_cylinders
: ℚ :=
  let height_A := 12
  let height_B := 7
  let radius_A := 7 / (2 * Real.pi)
  let volume_A := 12 * Real.pi * (radius_A ^ 2)  -- height_A * π * r_A^2
  let radius_B := 12 / (2 * Real.pi)
  let volume_B := 7 * Real.pi * (radius_B ^ 2)  -- height_B * π * r_B^2

  (volume_B / volume_A)

theorem ratio_of_volumes :
  volume_ratio_of_cylinders = (56 / 33) :=
by
  sorry

end ratio_of_volumes_l782_782314


namespace choose_3_from_10_is_120_l782_782113

theorem choose_3_from_10_is_120 :
  nat.choose 10 3 = 120 :=
by {
  -- proof would go here
  sorry
}

end choose_3_from_10_is_120_l782_782113


namespace digit_sum_inequality_l782_782259

theorem digit_sum_inequality (n : ℕ) (h : (digit_sum n = digit_sum (2 * n + 1))) :
  (digit_sum (3 * n - 3) ≠ digit_sum (n - 2)) :=
begin
  sorry  -- Proof will be provided here.
end

end digit_sum_inequality_l782_782259


namespace LCM_4_6_15_is_60_l782_782281

def prime_factors (n : ℕ) : List ℕ :=
  [] -- placeholder, definition of prime_factor is not necessary for the problem statement, so we leave it abstract

def LCM (a b : ℕ) : ℕ := 
  sorry -- placeholder, definition of LCM not directly necessary for the statement

theorem LCM_4_6_15_is_60 : LCM (LCM 4 6) 15 = 60 := 
  sorry

end LCM_4_6_15_is_60_l782_782281


namespace direction_and_distance_farthest_distance_total_fuel_consumption_l782_782210

def travel_records : List Int := [+10, -9, +7, -15, +6, -14, +4, -2]

-- Part 1: Direction and Distance from the Guard Post to Point A
theorem direction_and_distance (records : List Int) :
  let displacement := records.sum
  displacement = -13 → abs displacement = 13 :=
sorry

-- Part 2: Farthest Distance from the Starting Point
theorem farthest_distance (records : List Int) :
  let cumulative_sums := records.scanl (+) 0
  cumulative_sums.maximum = some 10 :=
sorry

-- Part 3: Total Fuel Consumption
theorem total_fuel_consumption (records : List Int) (fuel_per_km : Float) :
  let total_distance := records.map abs |>.sum
  let total_fuel := total_distance * fuel_per_km
  total_distance = 67 → total_fuel = 33.5 :=
sorry

-- Provide the default travel records list to use in theorems
#eval direction_and_distance travel_records
#eval farthest_distance travel_records
#eval total_fuel_consumption travel_records 0.5

end direction_and_distance_farthest_distance_total_fuel_consumption_l782_782210


namespace distinct_real_numbers_condition_l782_782165

theorem distinct_real_numbers_condition (a b c : ℝ) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a / (b - c)) + (b / (c - a)) + (c / (a - b)) = 1) :
  (a / (b - c)^2) + (b / (c - a)^2) + (c / (a - b)^2) = 1 := 
by sorry

end distinct_real_numbers_condition_l782_782165


namespace isogonally_conjugated_parallel_to_tangents_meet_l782_782644

open_locale classical

noncomputable def isogonal_conjugate (D : Point) (ABC : Triangle) : Point := sorry -- Placeholder for isogonal conjugate definition

structure Triangle where
  A B C : Point

structure Point where
  x y : ℝ

def is_median (A M D : Point) (ABC : Triangle) : Prop := sorry -- Placeholder for median definition

def is_tangent (circ : Circle) (P : Point) : Line := sorry -- Placeholder for tangent definition

def parallel (l1 l2 : Line) : Prop := sorry -- Placeholder for parallel definition

def circumcircle (ABC : Triangle) : Circle := sorry -- Placeholder for circumcircle definition

def tangents_meet_at (circ : Circle) (B C : Point) : Point := sorry -- Placeholder for tangents meet point definition

theorem isogonally_conjugated_parallel_to_tangents_meet (
  (A B C D : Point)
  (ABC : Triangle)
  (hD : D lies_on_median AM in ABC)
  (circ : Circle := circumcircle ⟨A, B, C⟩)
  (K : Point := tangents_meet_at circ B C)
  (D' : Point := isogonal_conjugate D ABC)
) : parallel (line_through D D') (line_through A K) :=
begin
  -- Proof would go here
  sorry,
end

end isogonally_conjugated_parallel_to_tangents_meet_l782_782644


namespace no_winning_strategy_l782_782334

theorem no_winning_strategy (r b : ℕ) (h1 : r + b = 52) (strategy : ℕ → bool) :
  ∀ k, (strategy k → (r / (r + b : ℝ) ≤ 0.5)) :=
by 
  sorry

end no_winning_strategy_l782_782334


namespace pages_with_same_units_digit_l782_782789

theorem pages_with_same_units_digit (book_pages : ℕ → ℕ) :
  (∀ x, (1 ≤ x ∧ x ≤ 73) → book_pages x = 74 - x) →
  (finset.card {x | (1 ≤ x ∧ x ≤ 73 ∧ (x % 10) = ((74 - x) % 10))}.to_finset = 15) :=
by
  sorry

end pages_with_same_units_digit_l782_782789


namespace number_of_bowls_l782_782738

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l782_782738


namespace probability_multiple_of_4_l782_782624

/-- Juan rolls a fair regular decagonal die marked with the numbers 1 through 10. 
Amal rolls a fair 12-sided die. The probability that the product of their rolls 
is a multiple of 4 is 2/5. -/
theorem probability_multiple_of_4 :
  let probability_multiple_of_4 := 2/5 in
  ∃ (prob_a : ℚ) (prob_b : ℚ), (prob_a * prob_b = probability_multiple_of_4) :=
sorry

end probability_multiple_of_4_l782_782624


namespace radius_of_symmetric_circle_l782_782525

theorem radius_of_symmetric_circle 
  (m : ℝ)
  (h_symmetric : ∀ x y : ℝ, ((x^2 + y^2 - 2*x + m*y - 4 = 0) ↔ (∃ (x' y' : ℝ), (x'^2 + y'^2 - 2*x' + m*y' - 4 = 0) ∧ (2*x + y = 0) ∧ (2*x' + y' = 0))) :
  let center : ℝ × ℝ := (1, -m/2) in
  let r := (1/2 * real.sqrt ((-2)^2 + 4^2 + 4 * 4)) in
  r = 3 :=
by
  sorry

end radius_of_symmetric_circle_l782_782525


namespace arithmetic_progression_of_polynomials_l782_782218

variable (a b c d : ℝ)

-- Conditions
variable (h1 : b = a + d)
variable (h2 : c = a + 2d)

-- Statement to prove
theorem arithmetic_progression_of_polynomials 
  (a b c d : ℝ) 
  (h1 : b = a + d) 
  (h2 : c = a + 2d) : 
  (a^2 + a*b + b^2) = (2*(a^2 + a*c + c^2) - (a^2 + a*b + b^2 + b^2 + b*c + c^2)) :=
sorry

end arithmetic_progression_of_polynomials_l782_782218


namespace median_and_mode_l782_782927

theorem median_and_mode (data : List ℕ) (h_data : data = [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]) :
  let sorted_data := data.qsort (<=)
  let n := sorted_data.length
  let median := (sorted_data.get! (n / 2 - 1) + sorted_data.get! (n / 2)) / 2
  let mode := (sorted_data.foldl (λ acc x, if sorted_data.count x > sorted_data.count acc then x else acc) 0)
  median = 15 ∧ mode = 17 :=
by 
  sorry

end median_and_mode_l782_782927


namespace isosceles_triangle_solution_l782_782598

noncomputable def isosceles_sides : (a b : ℝ) → (a = 25) ∧ (b = 10) → (sides : list ℝ)
| a b ⟨h₁, h₂⟩ := [a, a, b]

theorem isosceles_triangle_solution
(perimeter : ℝ) (iso_condition : ∃ a b, a = 25 ∧ b = 10 ∧ 2 * a + b = perimeter)
(intersect_uninscribed_boundary : ∀ (O M : ℝ) (ratio : ℝ), O = 2 / 3 * M) : 
sides 
       ∃ a b, (a = 25) ∧ (b = 10) ∧ (perimeter = 60)  :=
begin
  sorry
end

end isosceles_triangle_solution_l782_782598


namespace range_3a_2b_l782_782506

theorem range_3a_2b (a b : ℝ) (h : a^2 + b^2 = 4) : 
  -2 * Real.sqrt 13 ≤ 3 * a + 2 * b ∧ 3 * a + 2 * b ≤ 2 * Real.sqrt 13 := 
by 
  sorry

end range_3a_2b_l782_782506


namespace evaluate_exp_power_l782_782432

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l782_782432


namespace richard_second_day_distance_l782_782221

-- Define the conditions
def total_distance := 70
def day1_distance := 20
def day3_distance := 10
def remaining_distance := 36

-- Proof statement
theorem richard_second_day_distance : 
  ∃ x, (x = 4 ∧ (day1_distance / 2 - x = 6)) ∧ (day1_distance + x + day3_distance + remaining_distance = total_distance) :=
by {
  use 4,
  split,
  { split,
    { refl },
    { norm_num,
      ring }
  },
  { ring }
}

end richard_second_day_distance_l782_782221


namespace polynomial_coeffs_even_odd_condition_l782_782565

/-- 
  Given two polynomials P(x) and Q(x) with integer coefficients, 
  such that their product P(x) * Q(x) is a polynomial with even 
  coefficients, not all of which are divisible by 4. 
  Prove that one polynomial has all even coefficients, 
  and the other has at least one odd coefficient.
-/
theorem polynomial_coeffs_even_odd_condition
  (P Q : Polynomial ℤ)
  (h : ∀ k : ℕ, even (coeff (P * Q) k) ∧ ¬(∀ k : ℕ, 4 ∣ (coeff (P * Q) k))) :
  (∀ k : ℕ, even (coeff P k)) ∨ (∃ k : ℕ, odd (coeff Q k)) :=
by
  sorry

end polynomial_coeffs_even_odd_condition_l782_782565


namespace diffMaxMinPlanes_l782_782781

-- Define the geometric structures and the conditions
structure Tetrahedron :=
(F : Set Point)         -- Union of the faces of the tetrahedron
(vertices : Set Point)  -- Set of vertices of the tetrahedron

def Midpoint (p1 p2 : Point) : Point := sorry -- Define midpoint function

-- Define plane and its interaction with the tetrahedron, forming segments from vertices to edge midpoints
def PlaneIntersect (t : Tetrahedron) (plane : Set Point) : Prop :=
  ∀ v ∈ t.vertices, ∃ e1 e2 ∈ t.vertices, plane ∩ (segment v (Midpoint e1 e2)) ≠ ∅

-- Define the maximum and minimum number of intersecting planes
def maxNumberOfPlanes : ℕ := 4
def minNumberOfPlanes : ℕ := 2

-- Prove the difference between maximum and minimum numbers of planes
theorem diffMaxMinPlanes : maxNumberOfPlanes - minNumberOfPlanes = 2 :=
by {
  sorry
}

end diffMaxMinPlanes_l782_782781


namespace hypotenuse_segment_ratio_l782_782094

theorem hypotenuse_segment_ratio (x : ℝ) (h : 0 < x) :
  let AB := 3 * x,
      BC := x,
      AC := Real.sqrt (AB^2 + BC^2),
      AD := AC / 9 in
  AD * AC = (1 / 9) * (AC^2) :=
by 
  sorry

end hypotenuse_segment_ratio_l782_782094


namespace problem1_problem2_l782_782485

theorem problem1 : (2 + 1 / 4) ^ (1 / 2) - 0.3 ^ 0 - 16 ^ (-3 / 4) = 3 / 8 := 
by
  sorry

theorem problem2 : 4 ^ Real.logb 4 5 - Real.log 5 + Real.log10 500 + Real.log10 2 = 3 := 
by
  sorry

end problem1_problem2_l782_782485


namespace equation_of_the_line_l782_782795

theorem equation_of_the_line (a b : ℝ) :
    ((a - b = 5) ∧ (9 / a + 4 / b = 1)) → 
    ( (2 * 9 + 3 * 4 - 30 = 0) ∨ (2 * 9 - 3 * 4 - 6 = 0) ∨ (9 - 4 - 5 = 0)) :=
  by
    sorry

end equation_of_the_line_l782_782795


namespace garbage_decomposition_l782_782494

theorem garbage_decomposition (a b : ℝ) (t : ℕ) 
  (h1 : 0.05 = a * b^6) (h2 : 0.1 = a * b^12) : t = 32 :=
by
  let b := 2^(1/6)
  let a := 0.025
  have hb : b^6 = 2 := by sorry
  have ha : a * b^6 = 0.05 := by sorry
  have ha2 : a * b^12 = 0.1 := by sorry
  have v_expr : ∀ t, 0.025 * 2^(t / 6) = 1 → t = 32 := by
    intro t
    intro h3
    have h4 : 40 = 2^(t / 6) := by sorry
    have h5 : log 2 40 = t / 6 := by sorry
    have ht : t = 32 := by sorry
    exact ht
  exact v_expr t sorry

end garbage_decomposition_l782_782494


namespace choose_3_from_10_is_120_l782_782110

theorem choose_3_from_10_is_120 :
  nat.choose 10 3 = 120 :=
by {
  -- proof would go here
  sorry
}

end choose_3_from_10_is_120_l782_782110


namespace car_grid_probability_l782_782652

theorem car_grid_probability:
  let m := 11
  let n := 48
  100 * m + n = 1148 := by
  sorry

end car_grid_probability_l782_782652


namespace part1_part2_l782_782541

structure Vector3D (α : Type _) :=
  (x : α) (y : α) (z : α)

def magnitude {α : Type _} [field α] [has_sqrt α] (v : Vector3D α) : α :=
  real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def dot_product {α : Type _} [field α] (a b : Vector3D α) : α :=
  a.x * b.x + a.y * b.y + a.z * b.z

def cosine_deg (θ : ℝ) : ℝ :=
  real.cos (θ * real.pi / 180)

noncomputable def vec_a : Vector3D ℝ := ⟨4, 0, 0⟩ -- arbitrary example vector
noncomputable def vec_b : Vector3D ℝ := ⟨-1, sqrt 3, 0⟩ -- arbitrary example vector to form 120-degree angle

theorem part1 :
  (dot_product (vec_a - 2 • vec_b) (vec_a + vec_b) = 12) :=
sorry

theorem part2 :
  (magnitude (vec_a + vec_b) = 2 * real.sqrt 3) :=
sorry

end part1_part2_l782_782541


namespace evaluate_three_cubed_squared_l782_782444

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l782_782444


namespace number_of_bowls_l782_782719

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l782_782719


namespace distinct_equilateral_triangles_in_decagon_l782_782562

theorem distinct_equilateral_triangles_in_decagon :
  let vertices := {B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_{10}} in
  let polygon := regular_polygon 10 vertices in
  count_equilateral_triangles_with_two_vertices_in_set polygon vertices = 82 :=
sorry

end distinct_equilateral_triangles_in_decagon_l782_782562


namespace binomial_parameters_l782_782569

-- Define the problem as per given conditions
theorem binomial_parameters (n p : ℝ) (ξ : ℝ → Prop) (h₁ : ∀ ξ, ξ ∼ B(n, p)) (h₂ : Eξ ξ = 6) (h₃ : Dξ ξ = 3.6) :
  n = 15 :=
by
  sorry -- Proof goes here

end binomial_parameters_l782_782569


namespace length_of_tunnel_l782_782355

theorem length_of_tunnel (time : ℝ) (speed : ℝ) (train_length : ℝ) (total_distance : ℝ) (tunnel_length : ℝ) 
  (h1 : time = 30) (h2 : speed = 100 / 3) (h3 : train_length = 400) (h4 : total_distance = speed * time) 
  (h5 : tunnel_length = total_distance - train_length) : 
  tunnel_length = 600 :=
by
  sorry

end length_of_tunnel_l782_782355


namespace number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l782_782154

-- Define the conditions
def total_matches := 16
def played_matches := 9
def lost_matches := 2
def current_points := 19
def max_points_per_win := 3
def draw_points := 1
def remaining_matches := total_matches - played_matches
def required_points := 34

-- Statements to prove
theorem number_of_wins_in_first_9_matches :
  ∃ wins_in_first_9, 3 * wins_in_first_9 + draw_points * (played_matches - lost_matches - wins_in_first_9) = current_points :=
sorry

theorem highest_possible_points :
  current_points + remaining_matches * max_points_per_win = 40 :=
sorry

theorem minimum_wins_in_remaining_matches :
  ∃ min_wins_in_remaining_7, (min_wins_in_remaining_7 = 4 ∧ 3 * min_wins_in_remaining_7 + current_points + (remaining_matches - min_wins_in_remaining_7) * draw_points ≥ required_points) :=
sorry

end number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l782_782154


namespace matrix_product_arithmetic_sequence_l782_782390

open Matrix

def mat (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, n], ![0, 1]]

theorem matrix_product_arithmetic_sequence :
  (List.range' 2 2 50).foldl (λ acc n, acc ⬝ mat n) (1 : Matrix _ _ ℤ) = 
  ![![1, 2550], ![0, 1]] := by
  sorry

end matrix_product_arithmetic_sequence_l782_782390


namespace cloth_length_l782_782270

theorem cloth_length (L : ℕ) (x : ℕ) :
  32 + x = L ∧ 20 + 3 * x = L → L = 38 :=
by
  sorry

end cloth_length_l782_782270


namespace probability_of_A_l782_782250

variables {Ω : Type} [ProbabilitySpace Ω]
variables (A B : Event Ω)

-- Given conditions
def P_A_and_B : ℝ := 0.25
def P_A_or_B : ℝ := 0.6
def P_B : ℝ := 0.45

-- Goal
theorem probability_of_A (h1 : prob (A ∩ B) = P_A_and_B)
                         (h2 : prob (A ∪ B) = P_A_or_B)
                         (h3 : prob B = P_B) :
  prob A = 0.4 :=
sorry

end probability_of_A_l782_782250


namespace parrot_initial_phrases_l782_782895

theorem parrot_initial_phrases (current_phrases : ℕ) (days_with_parrot : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) :
  current_phrases = 17 →
  days_with_parrot = 49 →
  phrases_per_week = 2 →
  initial_phrases = current_phrases - phrases_per_week * (days_with_parrot / 7) :=
by
  sorry

end parrot_initial_phrases_l782_782895


namespace probability_distance_ge_one_l782_782162

theorem probability_distance_ge_one (S : set ℝ) (side_length_S : ∀ x ∈ S, x = 2)
  (P : ℝ) : 
  -- Assuming two points are chosen independently at random on the sides of a square S of side length 2
  let prob := (26 - Real.pi) / 32 in
    P = prob := 
sorry

end probability_distance_ge_one_l782_782162


namespace underline_at_most_500_numbers_l782_782705

theorem underline_at_most_500_numbers (
    A : Finset ℕ 
    (h_hash : A.card = 1000)
    (h_prime_diff : ∀ a ∈ A, ∃ b ∈ A, a ≠ b ∧ Nat.Prime (abs (a - b)))
) : ∃ B ⊆ A, B.card ≤ 500 ∧ ∀ a ∈ A \ B, ∃ b ∈ B, Nat.Prime (abs (a - b)) :=
sorry

end underline_at_most_500_numbers_l782_782705


namespace corrected_mean_l782_782300

theorem corrected_mean (mean : ℝ) (n : ℕ) (wrong_ob : ℝ) (correct_ob : ℝ) 
(h1 : mean = 36) (h2 : n = 50) (h3 : wrong_ob = 23) (h4 : correct_ob = 34) : 
(mean * n + (correct_ob - wrong_ob)) / n = 36.22 :=
by
  sorry

end corrected_mean_l782_782300


namespace tarun_left_after_four_days_l782_782816

def work_done (x : ℕ) (rate : ℕ) : ℚ := x * (1/rate)

theorem tarun_left_after_four_days :
  ∀ (W : ℚ) 
  (combined_rate_days : ℕ) 
  (arun_alone_days : ℕ) 
  (arun_leftover_days : ℕ),
  (work_done 1 combined_rate_days) = (work_done 1 arun_alone_days) + (work_done 1 arun_leftover_days) →
  ∃ (x : ℕ), 
  (work_done x combined_rate_days) + (work_done arun_leftover_days arun_alone_days) = 1 ∧ 
  x = 4 :=
by {
  intros W combined_rate_days arun_alone_days arun_leftover_days h,
  use 4,
  sorry
}

end tarun_left_after_four_days_l782_782816


namespace grades_assignment_l782_782353

theorem grades_assignment (students grades : ℕ) (h_students : students = 12) (h_grades : grades = 4) : 
  (grades ^ students) = 16777216 :=
by
  rw [h_students, h_grades]
  -- Simplify 4^12
  have h1 : 4 = 2^2 := rfl
  have h2 : 4^12 = (2^2)^12 := by rw h1
  rw pow_pow at h2
  have h3 : (2^2)^12 = 2^(2*12) := h2
  rw [mul_comm, ← pow_mul] at h3
  norm_num at h3
  exact h3

end grades_assignment_l782_782353


namespace backpacking_trip_cooks_l782_782130

theorem backpacking_trip_cooks :
  nat.choose 10 3 = 120 :=
sorry

end backpacking_trip_cooks_l782_782130


namespace same_terminal_side_l782_782279

theorem same_terminal_side (k : ℤ) : ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 6) = 11 * Real.pi / 6 := by
  sorry

end same_terminal_side_l782_782279


namespace number_of_bowls_l782_782732

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l782_782732


namespace maximize_profit_l782_782796

def sales_volume (x : ℝ) : ℝ := 5 - 12 / (x + 3)

def input_cost (t : ℝ) : ℝ := 10 + 2 * t

def selling_price (t : ℝ) : ℝ := 5 + 20 / t

def profit (x : ℝ) (a : ℝ) : ℝ :=
  let t := sales_volume x in
  t * selling_price t - input_cost t - x

theorem maximize_profit (a : ℝ) (h_a_pos : a > 0) (h_a_in : 0 ≤ a):
  (∀ x, 0 ≤ x ∧ x ≤ a → profit x a ≤ 25 - (36 / (x + 3) + x)) ∧ 
  (a ≥ 3 → ∀ x, (x = 3)) ∧ 
  (0 < a < 3 → ∀ x, (x = a)) :=
sorry

end maximize_profit_l782_782796


namespace area_of_T_prime_l782_782633

-- Given conditions
def AreaBeforeTransformation : ℝ := 9

def TransformationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4],![-2, 5]]

def AreaAfterTransformation (M : Matrix (Fin 2) (Fin 2) ℝ) (area_before : ℝ) : ℝ :=
  (M.det) * area_before

-- Problem statement
theorem area_of_T_prime : 
  AreaAfterTransformation TransformationMatrix AreaBeforeTransformation = 207 :=
by
  sorry

end area_of_T_prime_l782_782633


namespace merchant_initial_spent_l782_782798

-- Defining the initial amounts and conditions
variable (x : ℝ)

-- Conditions given in the problem
-- 1. The merchant sold salt in Moscow with a profit of 100 rubles on the first sale.
-- 2. The merchant used all the money earned from the first sale to buy more salt in Tver.
-- 3. He sold this salt in Moscow with a profit of 120 rubles on the second sale.
def initial_amount_spent {x : ℝ} (h1 : x + 100 ≥ 0) (h2 : (x + 100) + 120 ≥ 0) : Prop :=
  x = 500

theorem merchant_initial_spent :
  ∀ x : ℝ, (x + 100 ≥ 0) ∧ ((x + 100) + 120 ≥ 0) → initial_amount_spent x :=
  by
    intros x h
    sorry

end merchant_initial_spent_l782_782798


namespace range_of_alpha_l782_782666

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + (2 / 3)

noncomputable def slope_of_tangent_line (x : ℝ) : ℝ := 3 * x^2 - 1

theorem range_of_alpha :
  ∃α : set ℝ, 
    {α ∈ set.univ | (∃ x, α = slope_of_tangent_line x)} = 
    {α | (0 <= α ∧ α < π/2) ∨ (3*π/4 <= α ∧ α < π)} :=
sorry

end range_of_alpha_l782_782666


namespace liters_to_quarts_l782_782915

theorem liters_to_quarts (h : 0.75 * 2 * x ≈ 1.58 * 2 → x ≈ 1.58) :
  (1.5 * x * (1 / 2)) ≈ 1.6 := by
sorry

end liters_to_quarts_l782_782915


namespace range_of_set_W_l782_782222

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∈ (Finset.range (n - 1)).filter (λ x, 1 < x), n % m ≠ 0

def set_W : Finset ℕ := 
  (Finset.range 100).filter is_prime

theorem range_of_set_W : set_W.max' - set_W.min' = 95 := by
  sorry

end range_of_set_W_l782_782222


namespace correct_choice_l782_782527

open Real

def p : Prop := ∃ x : ℝ, 2 ^ x + 2 ^ (-x) = 1

def q : Prop := ∀ x : ℝ, log (x ^ 2 + 2 * x + 3) > 0

theorem correct_choice :
  (¬p ∧ q) ∧ ¬(p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∧ q) := by
  sorry

end correct_choice_l782_782527


namespace systematic_sampling_first_group_draw_l782_782275

noncomputable def index_drawn_from_group (x n : ℕ) : ℕ := x + 8 * (n - 1)

theorem systematic_sampling_first_group_draw (k : ℕ) (fifteenth_group : index_drawn_from_group k 15 = 116) :
  index_drawn_from_group k 1 = 4 := 
sorry

end systematic_sampling_first_group_draw_l782_782275


namespace billiard_ball_radius_unique_l782_782783

noncomputable def radius_of_billiard_balls (r : ℝ) : Prop :=
  let side_length := 292
  let lhs := (8 + 2 * Real.sqrt 3) * r
  lhs = side_length

theorem billiard_ball_radius_unique (r : ℝ) : radius_of_billiard_balls r → r = (146 / 13) * (4 - Real.sqrt 3 / 3) :=
by
  intro h1
  sorry

end billiard_ball_radius_unique_l782_782783


namespace number_of_functions_l782_782826

open Nat

theorem number_of_functions (f : Fin 15 → Fin 15)
  (h : ∀ x, (f (f x) - 2 * f x + x : Int) % 15 = 0) :
  ∃! n : Nat, n = 375 := sorry

end number_of_functions_l782_782826


namespace volume_space_within_cube_l782_782823

open Real

-- Define the cubic container and the sphere
def edge_length_cube : ℝ := 4
def radius_sphere : ℝ := 1

-- Volume of the cube of edge length 2 (due to the sphere's diameter being 2)
def volume_inner_cube (e : ℝ) : ℝ := e^3
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem volume_space_within_cube :
  volume_inner_cube 2 - volume_sphere 1 = 8 - (4 / 3) * π := sorry

end volume_space_within_cube_l782_782823


namespace determine_m_value_l782_782024

def quadratic_function_opens_downwards (m : ℝ) : Prop :=
  (m + 1) * (m^2 - 4) < 0

theorem determine_m_value :
  ∀ m : ℝ, (m^2 - 2 = 2) ∧ (m + 1 < 0) → m = -2 :=
by
  intros m h
  cases h with h_deg h_lead
  have h_sqr : m^2 = 4 := by linarith
  cases sqr_eq_four h_sqr with h_pos h_neg
  { simp [*, -sub_eq_add_neg] }
  sorry

/-- Auxiliary lemma to handle the square root cases --/
lemma sqr_eq_four {m : ℝ} (h : m^2 = 4) : m = 2 ∨ m = -2 :=
by
  -- solve the positive and negative solutions for m^2 = 4
  left; linarith; right; linarith

end determine_m_value_l782_782024


namespace evaluate_exponent_l782_782446

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l782_782446


namespace gcd_of_three_numbers_l782_782240

theorem gcd_of_three_numbers (a b c d : ℕ) (ha : a = 72) (hb : b = 120) (hc : c = 168) (hd : d = 24) : 
  Nat.gcd (Nat.gcd a b) c = d :=
by
  rw [ha, hb, hc, hd]
  -- Placeholder for the actual proof
  exact sorry

end gcd_of_three_numbers_l782_782240


namespace number_of_correct_relations_l782_782812

theorem number_of_correct_relations :
  ( ( ∀ a b, ({a, b} ⊆ {b, a}) ) ∧
    ( ∀ a b, ({a, b} = {b, a}) ) ∧
    ( {0} ≠ ∅ ) ∧
    ( 0 ∈ ({0} : Set Nat) ) ∧
    ( ∅ ∉ ({0} : Set Nat) ) ∧
    ( ∅ ⊆ ({0} : Set Nat) ) 
  ) → (4 = 4)
:= by
  intros
  sorry

end number_of_correct_relations_l782_782812


namespace min_chord_length_through_point_l782_782924

def circle_eq (x y : ℝ) := x^2 + y^2 - 6*x = 0

def point_lies_on_line (p : ℝ × ℝ) : Prop := ∃ m b : ℝ, p.2 = m * p.1 + b

def chord_length (p : ℝ × ℝ) : ℝ :=
  let d := real.sqrt ((3 - p.1)^2 + (p.2 - 0)^2) in
    2 * real.sqrt 9 - d^2

theorem min_chord_length_through_point (p : ℝ × ℝ) (h : p = (1, 2)) :
  ∃ (L : ℝ), point_lies_on_line p → circle_eq p.1 p.2 → L >= 2 :=
begin
  sorry
end

end min_chord_length_through_point_l782_782924


namespace t_shirt_cost_l782_782400

theorem t_shirt_cost (n_tshirts : ℕ) (total_cost : ℝ) (cost_per_tshirt : ℝ)
  (h1 : n_tshirts = 25)
  (h2 : total_cost = 248) :
  cost_per_tshirt = 9.92 :=
by
  sorry

end t_shirt_cost_l782_782400


namespace calculate_expression_l782_782022

namespace MathProof

def operation (x y : ℝ) : ℝ := (x - y) ^ 2

theorem calculate_expression (x y z : ℝ) : 
  operation (x - y) (y - z) = (x - 2y + z) ^ 2 :=
by
  sorry

end MathProof

end calculate_expression_l782_782022


namespace partition_contains_square_sum_l782_782861

-- Define a natural number n
def is_square (x : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = x

theorem partition_contains_square_sum (n : ℕ) (hn : n ≥ 15) :
  ∀ (A B : fin n → Prop), (∀ x, A x ∨ B x) ∧ (∀ x, ¬ (A x ∧ B x)) → (∃ a b, a ≠ b ∧ A a ∧ A b ∧ is_square (a + b)) ∨ (∃ c d, c ≠ d ∧ B c ∧ B d ∧ is_square (c + d)) :=
by
  sorry

end partition_contains_square_sum_l782_782861


namespace propositions_true_or_false_l782_782408

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem propositions_true_or_false :
  ¬(∀ x, (x ∈ (-∞ : ℝ, -1) ∪ (1, ∞)) → f x) ∧
  (∀ x, f (-x) = -f x) ∧
  ¬(∀ x₁ x₂, (x₁, x₂ ∈ (-1, 1)) → f x₁ + f x₂ = f ((x₁ + x₂) / (1 + x₁ * x₂))) ∧
  ¬(∀ x, ∀ y, x < y → f x < f y) :=
begin
  sorry
end

end propositions_true_or_false_l782_782408


namespace boy_reaches_school_early_l782_782757

-- Definitions and conditions from the problem
def usual_time : ℕ := 14
def speed_factor : ℚ := 7 / 6

-- Define the function that calculates new time
def new_time (T : ℕ) (s : ℚ) : ℚ := T / s

-- Prove that the boy reaches school 2 minutes early
theorem boy_reaches_school_early :
  usual_time - (new_time usual_time speed_factor).toNat = 2 :=
by
  -- The actual Lean proof would go here
  sorry

end boy_reaches_school_early_l782_782757


namespace slope_of_AB_l782_782909

noncomputable def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

def pointA : ℝ × ℝ := (2, 1)
def pointB : ℝ × ℝ := (3, 3)

theorem slope_of_AB :
  slope pointA pointB = 2 :=
by
  sorry

end slope_of_AB_l782_782909


namespace evaluate_three_cubed_squared_l782_782439

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l782_782439


namespace number_of_bowls_l782_782737

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l782_782737


namespace length_of_BC_l782_782208

axiom Rectangle (A B C D : Type) [rect : RectangleShape A B C D] (AB AD : ℝ)
axiom E (AD : ℝ) (ED : ℝ)
axiom M (EC : ℝ)
axiom AB_eq_BM : AB = BM
axiom AE_eq_EM : AE = EM
axiom ED_val : ED = 16
axiom CD_val : CD = 12
axiom AB_val_eq_BC (BC : ℝ) : AB = BC

theorem length_of_BC {A B C D : Type} [rect : RectangleShape A B C D]
(E : AD -> ℝ -> Type) (M : EC -> ℝ -> Type)
(AB_eq_BM : AB = BM) (AE_eq_EM : AE = EM) (ED_val : ED = 16) (CD_val : CD = 12) : BC = 20 :=
by
sorry

end length_of_BC_l782_782208


namespace diagonals_bisect_each_other_for_shapes_l782_782370

-- Definitions for the shapes
def is_parallelogram (P : Type*) [affine_space P] (A B C D : P) : Prop :=
  -- define the condition for a quadrilateral to be a parallelogram
  sorry 

def is_rectangle (P : Type*) [affine_space P] (A B C D : P) : Prop :=
  -- define the condition for a quadrilateral to be a rectangle
  sorry

def is_rhombus (P : Type*) [affine_space P] (A B C D : P) : Prop :=
  -- define the condition for a quadrilateral to be a rhombus
  sorry

def is_square (P : Type*) [affine_space P] (A B C D : P) : Prop :=
  -- define the condition for a quadrilateral to be a square
  sorry

-- Statement that diagonals bisect each other for all four shapes
theorem diagonals_bisect_each_other_for_shapes (P : Type*) [affine_space P] (A B C D : P) :
  (is_parallelogram P A B C D ∨ is_rectangle P A B C D ∨ is_rhombus P A B C D ∨ is_square P A B C D) →
  bisecting_diagonals P A B C D :=
begin
  -- proof body
  sorry
end

end diagonals_bisect_each_other_for_shapes_l782_782370


namespace trigonometric_identity_l782_782963

theorem trigonometric_identity (x : ℝ) (h : (1 + sin x) / cos x = 2) : (1 - sin x) / cos x = 1 / 2 :=
sorry

end trigonometric_identity_l782_782963


namespace problem_1_and_2_l782_782056

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin (2 * x) + Real.cos (2 * x)

theorem problem_1_and_2
  (a : ℝ)
  (h : f a (Real.pi / 3) = (Real.sqrt 3 - 1) / 2) :
  a = 1 ∧
  (∀ x, f 1 x ≤ Real.sqrt 2) ∧
  (∀ k : ℤ,
    ∀ x ∈ Set.Icc (k * Real.pi + Real.pi / 4) (k * Real.pi + 3 * Real.pi / 4),
      f 1 x < 0 := sorry

end problem_1_and_2_l782_782056


namespace find_special_four_digit_number_l782_782847

theorem find_special_four_digit_number :
  ∃ (N : ℕ), 
  (N % 131 = 112) ∧ 
  (N % 132 = 98) ∧ 
  (1000 ≤ N) ∧ 
  (N < 10000) ∧ 
  (N = 1946) :=
sorry

end find_special_four_digit_number_l782_782847


namespace arithmetic_geometric_sum_l782_782037

open Nat

theorem arithmetic_geometric_sum (d : ℕ) (n : ℕ) (hn : 0 < n) (hd : d ≠ 0) 
  (h1 : ∀ m, a (m + 1) = a m + d) 
  (h2 : a 1 = 1) 
  (h3 : let a_2 := a 1 + d; let a_4 := a 1 + 3 * d in a_2 * a_2 = a 1 * a_4) :
  a 4  + a 8 + a 12 + ... + a (4 * n + 4) = 2 * n^2 + 6 * n + 4 := 
by 
  sorry

end arithmetic_geometric_sum_l782_782037


namespace number_of_common_divisors_90_105_l782_782954

theorem number_of_common_divisors_90_105 : 
  let divisors_90 := {d | d ∣ 90} in
  let divisors_105 := {d | d ∣ 105} in
  let common_divisors := divisors_90 ∩ divisors_105 in
  common_divisors.to_finset.card = 8 :=
by {
  let divisors_90 := {d | d ∣ 90},
  let divisors_105 := {d | d ∣ 105},
  let common_divisors := divisors_90 ∩ divisors_105,
  exact 8
}

end number_of_common_divisors_90_105_l782_782954
