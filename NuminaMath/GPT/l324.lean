import Mathlib

namespace minimum_a_l324_324635

noncomputable def func (t a : ℝ) := 5 * (t + 1) ^ 2 + a / (t + 1) ^ 5

theorem minimum_a (a : ℝ) (h: ∀ t ≥ 0, func t a ≥ 24) :
  a = 2 * Real.sqrt ((24 / 7) ^ 7) :=
sorry

end minimum_a_l324_324635


namespace count_six_digit_flippy_numbers_divisible_by_18_l324_324780

def is_flippy (n : ℕ) : Prop :=
  n.to_digits.length = 6 ∧
  ∀i < 5, n.to_digits.get i ≠ n.to_digits.get (i + 1)

def divisible_by_2_and_9 (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n.to_digits.sum % 9 = 0

def six_digit_flippy_numbers := {n : ℕ | 10^5 ≤ n ∧ n < 10^6 ∧ is_flippy n ∧ divisible_by_2_and_9 n}

theorem count_six_digit_flippy_numbers_divisible_by_18 :
  Fintype.card six_digit_flippy_numbers = 4 :=
sorry

end count_six_digit_flippy_numbers_divisible_by_18_l324_324780


namespace inequality_solutions_l324_324320

theorem inequality_solutions (a : ℝ) (h_pos : 0 < a) 
  (h_ineq_1 : ∃! x : ℕ, 10 < a ^ x ∧ a ^ x < 100) : ∃! x : ℕ, 100 < a ^ x ∧ a ^ x < 1000 :=
by
  sorry

end inequality_solutions_l324_324320


namespace convex_polygons_with_A_l324_324118

theorem convex_polygons_with_A
  {n : ℕ}
  (h_n : 3 ≤ n) :
  let polygon_not_containing_A := ∑ k in finset.Ico 3 n, nat.choose (n - 1) k,
      polygon_containing_A := ∑ k in finset.Ico 2 n, nat.choose (n - 1) k in
  polygon_containing_A > polygon_not_containing_A :=
begin
  -- Proof should be provided here
  sorry
end

end convex_polygons_with_A_l324_324118


namespace PT_value_l324_324817

noncomputable def find_PT (PQ QR PR RS PT : ℚ) (h1 : PQ = 14) (h2 : QR = 16) (h3 : PR = 15) (h4 : RS = 7)
  (condition : Prop) := PT = 21952 / 3397 

theorem PT_value : 
  ∃ PT, find_PT 14 16 15 7 PT (∃ T S α, α = ∠(PQ T) ∧ α = ∠(PS T)) :=
begin
  use (21952 / 3397),
  -- Further proof would show that this PT meets all conditions.
  sorry
end

end PT_value_l324_324817


namespace number_of_rallies_l324_324697

open Nat

def X_rallies : Nat := 10
def O_rallies : Nat := 100
def sequence_Os : Nat := 3
def sequence_Xs : Nat := 7

theorem number_of_rallies : 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≤ 379) ∧ 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≥ 370) := 
by
  sorry

end number_of_rallies_l324_324697


namespace marie_distance_biked_l324_324847

def biking_speed := 12.0 -- Speed in miles per hour
def biking_time := 2.583333333 -- Time in hours

theorem marie_distance_biked : biking_speed * biking_time = 31 := 
by 
  -- The proof steps go here
  sorry

end marie_distance_biked_l324_324847


namespace right_triangle_AB_l324_324803

theorem right_triangle_AB
  (A B C : Point)
  (h_triangle : right_triangle A B C)
  (h_angleC : angle C = 90)
  (h_tanA : tan (angle A) = 7 / 2)
  (h_AC : dist A C = 12) :
  dist A B = 2 * sqrt 477 := sorry

end right_triangle_AB_l324_324803


namespace find_days_A_alone_works_l324_324989

-- Given conditions
def A_is_twice_as_fast_as_B (a b : ℕ) : Prop := a = b / 2
def together_complete_in_12_days (a b : ℕ) : Prop := (1 / b + 1 / a) = 1 / 12

-- We need to prove that A alone can finish the work in 18 days.
def A_alone_in_18_days (a : ℕ) : Prop := a = 18

theorem find_days_A_alone_works :
  ∃ (a b : ℕ), A_is_twice_as_fast_as_B a b ∧ together_complete_in_12_days a b ∧ A_alone_in_18_days a :=
sorry

end find_days_A_alone_works_l324_324989


namespace height_water_in_cylinder_l324_324664

-- Define the given conditions
def base_radius_cone := 12  -- Radius of the base of the cone in cm
def height_cone := 18       -- Height of the cone in cm
def base_radius_cylinder := 24  -- Radius of the base of the cylinder in cm

-- Volume formula for a cone
def volume_cone (r : ℝ) (h : ℝ) := (1/3) * real.pi * r^2 * h

-- Volume formula for a cylinder
def volume_cylinder (r : ℝ) (h : ℝ) := real.pi * r^2 * h

-- Problem statement
theorem height_water_in_cylinder :
  volume_cone base_radius_cone height_cone = volume_cylinder base_radius_cylinder 1.5 :=
by
  sorry

end height_water_in_cylinder_l324_324664


namespace dividend_calculation_l324_324932

theorem dividend_calculation (Divisor Quotient Remainder : ℕ) (h1 : Divisor = 15) (h2 : Quotient = 8) (h3 : Remainder = 5) : 
  (Divisor * Quotient + Remainder) = 125 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end dividend_calculation_l324_324932


namespace part1_part2_l324_324331

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x * (Real.sin x + Real.cos x)) - 1 / 2

theorem part1 (α : ℝ) (hα1 : 0 < α ∧ α < Real.pi / 2) (hα2 : Real.sin α = Real.sqrt 2 / 2) :
  f α = 1 / 2 :=
sorry

theorem part2 :
  ∀ (k : ℤ), ∀ (x : ℝ),
  -((3 : ℝ) * Real.pi / 8) + k * Real.pi ≤ x ∧ x ≤ (Real.pi / 8) + k * Real.pi →
  MonotoneOn f (Set.Icc (-((3 : ℝ) * Real.pi / 8) + k * Real.pi) ((Real.pi / 8) + k * Real.pi)) :=
sorry

end part1_part2_l324_324331


namespace size_of_second_drink_l324_324824

theorem size_of_second_drink
  (ounces_first_drink caffeine_first_drink : ℝ)
  (caffeine_ratio : ℝ)
  (total_caffeine : ℝ) :
  ounces_first_drink = 12 →
  caffeine_first_drink = 250 →
  caffeine_ratio = 3 →
  total_caffeine = 750 →
  ∃ (x : ℝ), caffeine_first_drink / 12 = 20.83 ∧
            caffeine_ratio * 20.83 = 62.5 ∧
            62.5 * x + caffeine_first_drink = 375 ∧
            total_caffeine / 2 = 375 ∧
            375 - caffeine_first_drink = 125 ∧
            62.5 * x = 125 ∧
            x = 2 :=
by
  intros h1 h2 h3 h4
  use 2
  rw [h1, h2, h3, h4]
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  · norm_num

end size_of_second_drink_l324_324824


namespace convex_polyhedron_faces_same_edges_l324_324444

-- Define convexity of polyhedron
def is_convex_polyhedron (P : Type) [Polyhedron P] : Prop :=
  ∀ x y ∈ polyhedron.interior P, segment x y ⊆ polyhedron.interior P

-- State the main theorem
theorem convex_polyhedron_faces_same_edges (P : Type) [Polyhedron P] (hP : is_convex_polyhedron P) (f : ℕ) (m : ℕ) :
  (∃ f_1 f_2 : Face P, f_1 ≠ f_2 ∧ Face.num_edges f_1 = Face.num_edges f_2) :=
sorry

end convex_polyhedron_faces_same_edges_l324_324444


namespace cos_sum_to_product_l324_324704

theorem cos_sum_to_product (a b : ℝ) : 
  cos (a + b) + cos (a - b) = 2 * cos a * cos b :=
sorry

end cos_sum_to_product_l324_324704


namespace find_f_of_3_l324_324479

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l324_324479


namespace B_work_days_l324_324970

theorem B_work_days (W : ℝ) (A_work_days : ℝ) (B_work_days_remaining : ℝ) :
  A_work_days = 40 → B_work_days_remaining = 45 → let A_work_done := 10 * (W / A_work_days) in
  let Remaining_work := W - A_work_done in
  (Remaining_work / B_work_days_remaining * 60) = W :=
by
  intros hA hB
  let A_work_done := 10 * (W / A_work_days)
  let Remaining_work := W - A_work_done
  have : Remaining_work = 3 * W / 4 := sorry
  have : B_complete_work := W = (Remaining_work / B_work_days_remaining) * 60
  exact this

end B_work_days_l324_324970


namespace find_two_different_weight_coins_l324_324197

theorem find_two_different_weight_coins :
  ∃ (weighings : ℕ), weighings ≤ 2 ∧
  ∃ (coins : fin 8 → ℝ), 
    (∃ i1 i2 : fin 8, coins i1 ≠ coins i2) ∧ 
    (∀ (groups : list (finset (fin 8))),
      ∀ g ∈ groups, g.card = 4 →
        let f := λ g : finset (fin 8), ∑ i in g, coins i,
            balance := λ g1 g2 : finset (fin 8), f g1 = f g2 in 
        ∃ g1 g2, g1 ∈ groups ∧ g2 ∈ groups ∧ balance g1 g2) :=
begin
  sorry
end

end find_two_different_weight_coins_l324_324197


namespace daughter_age_in_3_years_l324_324964

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end daughter_age_in_3_years_l324_324964


namespace math_proof_problem_l324_324333

noncomputable def parabola_equation (p : ℝ) (y₀ : ℝ) : ℝ → ℝ → Prop := 
  λ x y, x ^ 2 = 2 * p * y

noncomputable def is_parabola_focus (p : ℝ) (F : ℝ × ℝ) : Prop := 
  F = (0, p / 2)

def is_on_parabola (p y₀ : ℝ) (A : ℝ × ℝ) : Prop :=
  A.1 = 2 * Real.sqrt 3 ∧ parabola_equation p y₀ A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def is_distance_four (A F : ℝ × ℝ) : Prop :=
  distance A F = 4

def line_eq (k b : ℝ) : ℝ → ℝ := 
  λ x, k * x + b

def intersects_parabola (k b p : ℝ) : Prop :=
  ∃ D E : ℝ × ℝ, D ≠ E ∧ D.1^2 = 2 * p * D.2 ∧ E.1^2 = 2 * p * E.2 ∧
  D.2 = line_eq k b D.1 ∧ E.2 = line_eq k b E.1 ∧
  D.1 * E.1 + D.2 * E.2 = -4

def passes_fixed_point_P (k b : ℝ) : Prop :=
  line_eq k b 0 = 2

def line_l2 (m : ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, x - m * y + 3 * m + 2 = 0

def passes_fixed_point_Q (m : ℝ) : Prop :=
  line_l2 m (-2, 3)

def area_triangle (F P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := F;
  let (x2, y2) := P;
  let (x3, y3) := Q;
  Real.abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem math_proof_problem (p y₀ : ℝ) (F A : ℝ × ℝ)
  (h_p_pos : 0 < p) (h_y₀_large : p < y₀) (h_on_parabola : is_on_parabola p y₀ A)
  (h_distance : is_distance_four A F) : 
  parabola_equation 2 (2 * Real.sqrt 3) A.1 A.2 ∧ 
  (∀ (k b : ℝ), intersects_parabola k b 4 → passes_fixed_point_P k b ∧
    ∃ Q, passes_fixed_point_Q 1 ∧ area_triangle F (0, 2) Q = 1) :=
by
  sorry

end math_proof_problem_l324_324333


namespace resulting_parabola_is_correct_l324_324170

-- Conditions
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x - a)

def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ y, f y + b

-- The equivalent proof problem
theorem resulting_parabola_is_correct :
  (translate_up (translate_right initial_parabola 1) 3) = (λ x, (x - 1)^2 + 1) :=
sorry

end resulting_parabola_is_correct_l324_324170


namespace three_layers_rug_area_l324_324203

theorem three_layers_rug_area :
  ∀ (A B C D E : ℝ),
    A + B + C = 212 →
    (A + B + C) - D - 2 * E = 140 →
    D = 24 →
    E = 24 :=
by
  intros A B C D E h1 h2 h3
  sorry

end three_layers_rug_area_l324_324203


namespace cylinder_surface_area_l324_324981

theorem cylinder_surface_area (h : ℝ) (r : ℝ) (h_eq : h = 12) (r_eq : r = 4) : 
  2 * π * r * (r + h) = 128 * π := 
by
  rw [h_eq, r_eq]
  sorry

end cylinder_surface_area_l324_324981


namespace tile_border_ratio_l324_324217

theorem tile_border_ratio (n : ℕ) (t w : ℝ) (H1 : n = 30)
  (H2 : 900 * t^2 / (30 * t + 30 * w)^2 = 0.81) :
  w / t = 1 / 9 :=
by
  sorry

end tile_border_ratio_l324_324217


namespace snail_displacement_at_55_days_l324_324649

theorem snail_displacement_at_55_days :
  ∑ n in finset.range 55, (1 / (n + 1) - 1 / (n + 2)) = 55 / 56 := 
sorry

end snail_displacement_at_55_days_l324_324649


namespace complex_point_quadrant_l324_324900

theorem complex_point_quadrant :
  let z := Complex.cos (2 * Real.pi / 3) + Complex.sin (2 * Real.pi / 3) * Complex.I in
  z.re < 0 ∧ z.im > 0 :=
by
  let z := Complex.cos (2 * Real.pi / 3) + Complex.sin (2 * Real.pi / 3) * Complex.I
  sorry

end complex_point_quadrant_l324_324900


namespace probability_of_sequence_l324_324711

open Finset

noncomputable def count_arrangements (n k : ℕ) : ℕ :=
  (finset.range n).powerset.filter (λ s, s.card = k).card

theorem probability_of_sequence:
  (count_arrangements 8 5) = 56 ∧
  (∃! s: finset ℕ, s = {0, 1, 2, 5, 7}) →
  (1 / 56 : ℝ) = 1 / count_arrangements 8 5 :=
sorry

end probability_of_sequence_l324_324711


namespace flower_pots_total_cost_l324_324848

theorem flower_pots_total_cost :
  ∃ (x : ℝ), 
    let a := x in
    let b := x + 0.3 in
    let c := x + 0.6 in
    let d := x + 0.9 in
    let e := x + 1.2 in
    let f := x + 1.5 in
    f = 2.125 ∧ (a + b + c + d + e + f = 8.25) :=
by
  sorry

end flower_pots_total_cost_l324_324848


namespace hexagons_form_square_l324_324877

theorem hexagons_form_square (a b y : ℕ) (h_rect : a = 8 ∧ b = 18) 
    (h_hexagons : 2 * (a * b) = s * s) 
    (h_square : s = 12) : 
    y = s / 2 := 
by
  -- Rectangle area
  have h1 : a * b = 144 := by
    calc 8 * 18 = 144
  -- Square's side length squared
  have h2 : s^2 = 144 := by
    calc s^2 = 144 
  -- Translate from the problem's geometric constraints
  have h3 : s = 12 := by 
    exact h_square
  -- Conclude that y must be half the side of the square's side length
  have h4 : y = 12 / 2 := by
    calc y = 12 / 2
  -- Therefore y = 6
  exact h4

end hexagons_form_square_l324_324877


namespace oldest_child_age_l324_324133

theorem oldest_child_age 
  (avg_age : ℕ) (child1 : ℕ) (child2 : ℕ) (child3 : ℕ) (child4 : ℕ)
  (h_avg : avg_age = 8) 
  (h_child1 : child1 = 5) 
  (h_child2 : child2 = 7) 
  (h_child3 : child3 = 10)
  (h_avg_eq : (child1 + child2 + child3 + child4) / 4 = avg_age) :
  child4 = 10 := 
by 
  sorry

end oldest_child_age_l324_324133


namespace dice_sum_to_10_probability_l324_324555

theorem dice_sum_to_10_probability : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 10) →
  (calculate_probability (3, 6, 10) = 1 / 9) :=
begin
  sorry
end

-- Defining calculate_probability (used in the theorem) to represent the probability calculation
def calculate_probability (n_dice : ℕ, n_faces : ℕ, target_sum : ℕ) : ℚ :=
  let outcomes := (finset.powerset (finset.range (n_faces*2)).map(λ x, (x.val + 1)).filter(x.val <= n_faces)) in
  let valid_outcomes := outcomes.filter(λ l, l.sum = target_sum) in
  valid_outcomes.card / outcomes.card.to_rat

end dice_sum_to_10_probability_l324_324555


namespace max_min_values_interval_l324_324277

def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x

theorem max_min_values_interval :
  ∃ (x_max x_min : ℝ), x_max ∈ Icc (-3 : ℝ) (3 : ℝ) ∧ x_min ∈ Icc (-3 : ℝ) (3 : ℝ) ∧
  f x_max = (16 / 3 : ℝ) ∧ f x_min = - (16 / 3 : ℝ) := 
by
  sorry

end max_min_values_interval_l324_324277


namespace equations_of_motion_l324_324682

noncomputable def omega := 10 -- angular velocity (rad/s)
noncomputable def OA := 90 -- length OA in cm
noncomputable def AB := 90 -- length AB in cm
noncomputable def AL := 30 -- length AL in cm (1/3 of AB)

def Ax (θ : ℝ) := OA * Real.cos θ
def Ay (θ : ℝ) := OA * Real.sin θ

def Bx (θ : ℝ) (xB : ℝ) := (xB - OA * Real.cos θ)^2 + (yB - OA * Real.sin θ)^2 = AB^2

def Lx (θ : ℝ) (xB : ℝ) := Ax θ + (1/3) * (xB - Ax θ)
def Ly (θ : ℝ) (yB : ℝ) := Ay θ + (1/3) * (yB - Ay θ)

def vLx (θ : ℝ) (dxB_dt : ℝ) := (- OA * omega * Real.sin θ) + (1/3) * (dxB_dt + OA * omega * Real.sin θ)
def vLy (θ : ℝ) (dyB_dt : ℝ) := (OA * omega * Real.cos θ) + (1/3) * (dyB_dt - OA * omega * Real.cos θ)

theorem equations_of_motion (θ : ℝ) (xB yB dxB_dt dyB_dt : ℝ) : 
  Lx θ xB = Ax θ + (1/3) * (xB - Ax θ) ∧
  Ly θ yB = Ay θ + (1/3) * (yB - Ay θ) ∧
  vLx θ dxB_dt = (- OA * omega * Real.sin θ) + (1/3) * (dxB_dt + OA * omega * Real.sin θ) ∧
  vLy θ dyB_dt = (OA * omega * Real.cos θ) + (1/3) * (dyB_dt - OA * omega * Real.cos θ) :=
by
  sorry

end equations_of_motion_l324_324682


namespace FDI_in_rural_AndhraPradesh_l324_324233

-- Definitions from conditions
def total_FDI : ℝ := 300 -- Total FDI calculated
def FDI_Gujarat : ℝ := 0.30 * total_FDI
def FDI_Gujarat_Urban : ℝ := 0.80 * FDI_Gujarat
def FDI_AndhraPradesh : ℝ := 0.20 * total_FDI
def FDI_AndhraPradesh_Rural : ℝ := 0.50 * FDI_AndhraPradesh 

-- Given the conditions, prove the size of FDI in rural Andhra Pradesh is 30 million
theorem FDI_in_rural_AndhraPradesh :
  FDI_Gujarat_Urban = 72 → FDI_AndhraPradesh_Rural = 30 :=
by
  sorry

end FDI_in_rural_AndhraPradesh_l324_324233


namespace perfect_square_trinomial_k_l324_324028

theorem perfect_square_trinomial_k :
  ∀ (k : ℤ), (∃ (a b : ℤ), a*x² + b*x + c = (a*x + b)²) ↔ (k = 6 ∨ k = -6) :=
by
  -- proof skipped
  sorry

end perfect_square_trinomial_k_l324_324028


namespace binary_to_decimal_l324_324254

theorem binary_to_decimal : 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0) = 54 :=
by 
  sorry

end binary_to_decimal_l324_324254


namespace gain_in_transaction_per_year_l324_324950

variable (P : ℝ) (R_borrow R_lend : ℝ) (T : ℝ)
variable (interest_borrow interest_lend gain_per_year : ℝ) 

def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / 100

def total_interest_earned : ℝ :=
  simple_interest P R_lend T

def total_interest_paid : ℝ :=
  simple_interest P R_borrow T

def total_gain_in_2_years : ℝ :=
  total_interest_earned - total_interest_paid

def gain_per_year_computed : ℝ :=
  total_gain_in_2_years / T

theorem gain_in_transaction_per_year 
  (hP : P = 6000) 
  (hR_borrow : R_borrow = 4) 
  (hR_lend : R_lend = 6) 
  (hT : T = 2) 
  (h_interest_borrow : interest_borrow = simple_interest P R_borrow T) 
  (h_interest_lend : interest_lend = simple_interest P R_lend T) 
  (h_gain_per_year : gain_per_year = gain_per_year_computed) :
  gain_per_year = 120 := 
by 
  simp [gain_per_year_computed, total_gain_in_2_years, total_interest_earned, total_interest_paid, simple_interest]
  sorry

end gain_in_transaction_per_year_l324_324950


namespace john_payment_correct_l324_324825

noncomputable def camera_value : ℝ := 5000
noncomputable def base_rental_fee_per_week : ℝ := 0.10 * camera_value
noncomputable def high_demand_fee_per_week : ℝ := base_rental_fee_per_week + 0.03 * camera_value
noncomputable def low_demand_fee_per_week : ℝ := base_rental_fee_per_week - 0.02 * camera_value
noncomputable def total_rental_fee : ℝ :=
  high_demand_fee_per_week + low_demand_fee_per_week + high_demand_fee_per_week + low_demand_fee_per_week
noncomputable def insurance_fee : ℝ := 0.05 * camera_value
noncomputable def pre_tax_total_cost : ℝ := total_rental_fee + insurance_fee
noncomputable def tax : ℝ := 0.08 * pre_tax_total_cost
noncomputable def total_cost : ℝ := pre_tax_total_cost + tax

noncomputable def mike_contribution : ℝ := 0.20 * total_cost
noncomputable def sarah_contribution : ℝ := min (0.30 * total_cost) 1000
noncomputable def alex_contribution : ℝ := min (0.10 * total_cost) 700
noncomputable def total_friends_contributions : ℝ := mike_contribution + sarah_contribution + alex_contribution

noncomputable def john_final_payment : ℝ := total_cost - total_friends_contributions

theorem john_payment_correct : john_final_payment = 1015.20 :=
by
  sorry

end john_payment_correct_l324_324825


namespace daughter_age_in_3_years_l324_324956

variable (mother_age_now : ℕ) (gap_years : ℕ) (ratio : ℕ)

theorem daughter_age_in_3_years
  (h1 : mother_age_now = 41) 
  (h2 : gap_years = 5)
  (h3 : ratio = 2) :
  let mother_age_then := mother_age_now - gap_years in
  let daughter_age_then := mother_age_then / ratio in
  let daughter_age_now := daughter_age_then + gap_years in
  let daughter_age_in_3_years := daughter_age_now + 3 in
  daughter_age_in_3_years = 26 :=
  by
    sorry

end daughter_age_in_3_years_l324_324956


namespace find_triangle_area_GCD_l324_324454

noncomputable def square_area_144 (ABCD : Set Point) : Prop :=
  shape ABCD = square ∧ area ABCD = 144

noncomputable def point_E_on_BC (B C E : Point) : Prop :=
  line_segment B C E ∧ length_segment B E = 3 * length_segment E C

noncomputable def midpoint_FG (A E D F G : Point) : Prop :=
  midpoint A E F ∧ midpoint D E G

noncomputable def quad_area_26 (B E G F : Set Point) : Prop :=
  quad B E G F ∧ area (shape B E G F) = 26

noncomputable def triangle_area_GCD (G C D : Set Point) : Prop :=
  triangle G C D ∧ area (shape G C D) = 28

theorem find_triangle_area_GCD (A B C D E F G : Point) (ABCD : Set Point) (BEGF : Set Point) :
  square_area_144 ABCD →
  point_E_on_BC B C E →
  midpoint_FG A E D F G →
  quad_area_26 BEGF →
  triangle_area_GCD G C D :=
by
  intros h_square h_point_E h_midpoint h_quad_area
  sorry

end find_triangle_area_GCD_l324_324454


namespace std_deviation_eq_l324_324363

def sampleA : list ℝ := [66, 67, 65, 68, 64, 62, 69, 66, 65, 63]
def sampleB : list ℝ := sampleA.map (λ x, x + 2)

theorem std_deviation_eq (A B : list ℝ) (h : B = A.map (λ x, x + 2)) :
  (stddev A) = (stddev B) := 
by
  sorry

end std_deviation_eq_l324_324363


namespace find_f_g_3_l324_324890

def f : ℕ → ℕ
| 1 := 2
| 2 := 4
| 3 := 3
| 4 := 1
| _ := 0 -- Default case to handle values not in the table

def g : ℕ → ℕ
| 1 := 3
| 2 := 1
| 3 := 2
| 4 := 4
| _ := 0 -- Default case to handle values not in the table

theorem find_f_g_3 : f (g 3) = 4 :=
by
  sorry

end find_f_g_3_l324_324890


namespace correct_option_is_E_l324_324579

def optionA : Prop := (1 = "red") ∧ (2 = "blue") ∧ (3 = "green")
def optionB : Prop := (1 = "red") ∧ (2 = "green") ∧ (3 = "blue")
def optionC : Prop := (1 = "blue") ∧ (2 = "red") ∧ (3 = "green")
def optionD : Prop := (1 = "blue") ∧ (2 = "green") ∧ (3 = "red")
def optionE : Prop := (1 = "green") ∧ (2 = "blue") ∧ (3 = "red")

theorem correct_option_is_E : optionE :=
by {
   -- Proof required here
   sorry
}

end correct_option_is_E_l324_324579


namespace intercepts_equal_implies_a_neg_3_l324_324308

theorem intercepts_equal_implies_a_neg_3 (a : ℝ) :
  (∃ b : ℝ, b ≠ 0 ∧ (line_through (0, 1) (4, a) ≠ line_through_origin ∧ 
  x_intercept (line_through (0, 1) (4, a)) = y_intercept (line_through (0, 1) (4, a)))) → 
  a = -3 :=
sorry

end intercepts_equal_implies_a_neg_3_l324_324308


namespace product_positive_l324_324109

variables {x y : ℝ}

noncomputable def non_zero (z : ℝ) := z ≠ 0

theorem product_positive (hx : non_zero x) (hy : non_zero y) 
(h1 : x^2 - x > y^2) (h2 : y^2 - y > x^2) : x * y > 0 :=
by
  sorry

end product_positive_l324_324109


namespace rabbit_reaches_0_6_from_1_3_l324_324997

noncomputable def Q : ℕ × ℕ → ℚ
| (0, 6) := 1
| (0, 0) := 0
| (6, 0) := 0
| (6, 6) := 0
| (1, 3) := 1/4 * Q (0, 3) + 1/4 * Q (2, 3) + 1/4 * Q (1, 2) + 1/4 * Q (1, 4)
| (0, y) := sorry -- Other points that would need to be defined for a complete solution
| (x, 0) := sorry -- Similarly, other unknown starting points
| (x, 6) := sorry
| (6, y) := sorry
| (x, y) := sorry

theorem rabbit_reaches_0_6_from_1_3 :
  Q (1, 3) = 1 / 4 :=
sorry

end rabbit_reaches_0_6_from_1_3_l324_324997


namespace find_fixed_point_on_ellipse_l324_324304

theorem find_fixed_point_on_ellipse (a b c : ℝ) (h_gt_zero : a > b ∧ b > 0)
    (h_ellipse : ∀ (P : ℝ × ℝ), (P.1 ^ 2 / a ^ 2) + (P.2 ^ 2 / b ^ 2) = 1)
    (A1 A2 : ℝ × ℝ)
    (h_A1 : A1 = (-a, 0))
    (h_A2 : A2 = (a, 0))
    (MC : ℝ) (h_MC : MC = (a^2 + b^2) / c) :
  ∃ (M : ℝ × ℝ), M = (MC, 0) := 
sorry

end find_fixed_point_on_ellipse_l324_324304


namespace vector_subtraction_l324_324338

theorem vector_subtraction :
  let p := ⟨-2, 3, 4⟩ : ℝ^3
  let q := ⟨1, -2, 5⟩ : ℝ^3
  p - 5 • q = ⟨-7, 13, -21⟩ :=
by
  sorry

end vector_subtraction_l324_324338


namespace solve_quadratic_l324_324869

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 / 2 ∨ x = 1 :=
by
  intro x
  construct sorry

end solve_quadratic_l324_324869


namespace sum_eq_sum_l324_324529

theorem sum_eq_sum {a b c d : ℝ} (h1 : a + b = c + d) (h2 : ac = bd) (h3 : a + b ≠ 0) : a + c = b + d := 
by
  sorry

end sum_eq_sum_l324_324529


namespace length_segment_AB_l324_324321

-- Definitions of the objects and conditions in the problem
def circle (x y : ℝ) := (x - 2)^2 + y^2 = 3

def line1 (x y m : ℝ) := x - m * y - 1 = 0

def line2 (x y m : ℝ) := m * x + y - m = 0

def slope_cd := -1

-- The final theorem statement
theorem length_segment_AB (m : ℝ) (A C B D : ℝ × ℝ)
  (h1 : circle (A.1) (A.2))
  (h2 : circle (C.1) (C.2))
  (h3 : circle (B.1) (B.2))
  (h4 : circle (D.1) (D.2))
  (h5 : line1 (A.1) (A.2) m)
  (h6 : line1 (C.1) (C.2) m)
  (h7 : line2 (B.1) (B.2) m)
  (h8 : line2 (D.1) (D.2) m)
  (h9 : (C.2 - D.2) / (C.1 - D.1) = slope_cd)
  : dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end length_segment_AB_l324_324321


namespace probability_rain_at_most_3_days_l324_324150

noncomputable def probability_of_rain := (1:ℝ)/5
noncomputable def days_in_april := 30

theorem probability_rain_at_most_3_days : 
  (∑ k in finset.range 4, (nat.choose days_in_april k * probability_of_rain ^ k * (1 - probability_of_rain) ^ (days_in_april - k))) ≈ 0.616 :=
by sorry

end probability_rain_at_most_3_days_l324_324150


namespace problem1_problem2_l324_324250

-- Problem 1: Prove \( \sqrt{10} \times \sqrt{2} + \sqrt{15} \div \sqrt{3} = 3\sqrt{5} \)
theorem problem1 : Real.sqrt 10 * Real.sqrt 2 + Real.sqrt 15 / Real.sqrt 3 = 3 * Real.sqrt 5 := 
by sorry

-- Problem 2: Prove \( \sqrt{27} - (\sqrt{12} - \sqrt{\frac{1}{3}}) = \frac{4\sqrt{3}}{3} \)
theorem problem2 : Real.sqrt 27 - (Real.sqrt 12 - Real.sqrt (1 / 3)) = (4 * Real.sqrt 3) / 3 :=
by sorry

end problem1_problem2_l324_324250


namespace equivalent_operation_l324_324221

theorem equivalent_operation (x : ℚ) :
  (x * (2/3)) / (5/6) = x * (4/5) :=
by
  -- Normal proof steps might follow here
  sorry

end equivalent_operation_l324_324221


namespace find_f_3_l324_324509

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l324_324509


namespace exists_pow_two_sub_one_divisible_by_odd_l324_324860

theorem exists_pow_two_sub_one_divisible_by_odd {a : ℕ} (h_odd : a % 2 = 1) 
  : ∃ b : ℕ, (2^b - 1) % a = 0 :=
sorry

end exists_pow_two_sub_one_divisible_by_odd_l324_324860


namespace find_f_of_3_l324_324478

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l324_324478


namespace batsman_average_after_17th_l324_324208

def runs_17th_inning : ℕ := 87
def increase_in_avg : ℕ := 4
def num_innings : ℕ := 17

theorem batsman_average_after_17th (A : ℕ) (H : A + increase_in_avg = (16 * A + runs_17th_inning) / num_innings) : 
  (A + increase_in_avg) = 23 := sorry

end batsman_average_after_17th_l324_324208


namespace sum_of_first_n_terms_l324_324746

noncomputable def sum_of_arithmetic_sequence (n a_1 d : ℕ) : ℕ :=
  n * a_1 + (n * (n - 1)) * d / 2

theorem sum_of_first_n_terms (a_2 a_3 a_6 : ℕ) (h_arithmetic : a_2 = a_3 - 2)
  (h_arithmetic2 : a_6 = a_3 + 6)
  (h_geometric : a_3 ^ 2 = a_2 * a_6)
  (n : ℕ) :
  sum_of_arithmetic_sequence n (-1) 2 = n * (n - 2) :=
begin
  sorry
end

end sum_of_first_n_terms_l324_324746


namespace daughter_age_in_3_years_l324_324963

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end daughter_age_in_3_years_l324_324963


namespace height_of_water_in_cylinder_l324_324666

-- Define the parameters of the cone
def r_cone : ℝ := 12
def h_cone : ℝ := 18

-- Define the volume formula for a cone
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

-- Define the parameters of the cylinder
def r_cylinder : ℝ := 24

-- Define the volume formula for a cylinder
def volume_cylinder (r h : ℝ) : ℝ := real.pi * r^2 * h

-- Noncomptuable context for the proof
noncomputable def h_cylinder : ℝ :=
  let V_cone := volume_cone r_cone h_cone
  in (V_cone / (real.pi * r_cylinder^2))

-- Theorem to prove the height of water in the cylinder
theorem height_of_water_in_cylinder : h_cylinder = 1.5 :=
  by
  -- Placeholder for the actual proof steps
  sorry

end height_of_water_in_cylinder_l324_324666


namespace measure_8_liters_possible_l324_324004

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l324_324004


namespace pizza_toppings_l324_324630

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l324_324630


namespace num_ways_to_pay_l324_324282

theorem num_ways_to_pay (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (n / 2) + 1 :=
sorry

end num_ways_to_pay_l324_324282


namespace find_n_l324_324199

theorem find_n (n : ℕ) 
  (hM : ∀ M, M = n - 7 → 1 ≤ M)
  (hA : ∀ A, A = n - 2 → 1 ≤ A)
  (hT : ∀ M A, M = n - 7 → A = n - 2 → M + A < n) :
  n = 8 :=
by
  sorry

end find_n_l324_324199


namespace quadratic_solution_l324_324872

theorem quadratic_solution (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 → (x = 1 / 2 ∨ x = 1) :=
by sorry

end quadratic_solution_l324_324872


namespace good_numbers_fraction_l324_324996

/-- A positive integer is 'good' if each digit is 1 or 2 and there are neither four consecutive 1's nor three consecutive 2's. -/
def is_good (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → (digit k = 1 ∨ digit k = 2) ∧
  ¬ (four_consecutive_ones n ∨ three_consecutive_twos n)

/-- Define the number of n-digit 'good' positive integers -/
def a : ℕ → ℕ
| 0      := sorry
| 1      := sorry
| 2      := sorry
| 3      := sorry
| 4      := sorry
| 5      := sorry
| (n + 5) := a n + 2 * a (n + 2) + 2 * a (n + 1) + a (n + 3)

/-- Prove the given expression equals 2. -/
theorem good_numbers_fraction (a : ℕ → ℕ) (h : ∀ n, a (n + 5) = a n + 2 * a (n + 2) + 2 * a (n + 1) + a (n + 3)) :
  (a 10 - a 8 - a 5) / (a 7 + a 6) = 2 :=
sorry

end good_numbers_fraction_l324_324996


namespace units_digit_sum_factorials_l324_324942

theorem units_digit_sum_factorials : 
  let units_digit (n : Nat) := n % 10 in
  (units_digit (1!) + units_digit (2!) + units_digit (3!) + units_digit (4!) + units_digit (Sum (List.init (500-4) (fun n => (n+5) !)))) % 10 = 3 :=
by
  let units_digit (n : Nat) := n % 10
  sorry

end units_digit_sum_factorials_l324_324942


namespace number_between_neg2_and_0_l324_324659

theorem number_between_neg2_and_0 : 
  ∃ (x : ℤ), x ∈ ({3, 1, -3, -1} : Set ℤ) ∧ -2 < x ∧ x < 0 := 
by 
  use -1
  simp
  split
  · exact (by norm_num : -1 ∈ {3, 1, -3, -1})
  · split
    · exact (by norm_num : -2 < -1)
    · exact (by norm_num : -1 < 0)
  · sorry

end number_between_neg2_and_0_l324_324659


namespace ratio_of_medians_range_l324_324786

noncomputable def median_length (a b : ℝ) : ℝ :=
  real.sqrt ((a * a / 4) + (b * b))

noncomputable def ratio_of_medians (x y : ℝ) : ℝ :=
  (median_length x y) / (median_length y x)

theorem ratio_of_medians_range (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let m := ratio_of_medians x y in
  1 / 2 < m ∧ m < 2 :=
sorry

end ratio_of_medians_range_l324_324786


namespace quadratic_min_value_unique_l324_324253

theorem quadratic_min_value_unique {a b c : ℝ} (h : a > 0) :
  (∀ x : ℝ, 3 * x^2 - 8 * x + 7 ≥ 3 * (4 / 3)^2 - 8 * (4 / 3) + 7) → 
  ∃ x : ℝ, x = 4 / 3 :=
by
  sorry

end quadratic_min_value_unique_l324_324253


namespace measure_8_liters_with_buckets_l324_324005

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l324_324005


namespace similar_triangles_PAABB_l324_324077

variable {A B A' B' P : EuclideanGeometry.Point}
variable (C : EuclideanGeometry.Circle)
variable (hA : A ∈ C)
variable (hB : B ∈ C)
variable (hPA' : Line (P, A) ∩ C = {A, A'}) 
variable (hPB' : Line (P, B) ∩ C = {B, B'}) 

theorem similar_triangles_PAABB'_if_circ_inter_sec
  (h : ∀ A B A' B' : EuclideanGeometry.Point, 
    A ∈ C ∧ B ∈ C ∧ Line (P, A) ∩ C = {A, A'} ∧ Line (P, B) ∩ C = {B, B'} → 
    SimilarTriangle P A B P A' B') : 
    SimilarTriangle P A B P A' B' :=
h sorry

end similar_triangles_PAABB_l324_324077


namespace ab_root_of_Q_l324_324457

theorem ab_root_of_Q (a b : ℝ) (h : a ≠ b) (ha : a^4 + a^3 - 1 = 0) (hb : b^4 + b^3 - 1 = 0) :
  (ab : ℝ)^6 + (ab : ℝ)^4 + (ab : ℝ)^3 - (ab : ℝ)^2 - 1 = 0 := 
sorry

end ab_root_of_Q_l324_324457


namespace a_seq_unbounded_l324_324155

noncomputable def a_seq : ℕ → ℝ
| 0       := 1
| (n + 1) := a_seq n + 1 / (Real.sqrt n * a_seq n)

theorem a_seq_unbounded : ¬∃ M : ℝ, ∀ n : ℕ, a_seq n ≤ M := 
sorry

end a_seq_unbounded_l324_324155


namespace find_CD_values_l324_324111

theorem find_CD_values :
  ∃ (CD : ℤ), CD ∈ {3, 1, 19, 15} ∧
  (∀ (A B C D : ℤ), 
    A = -3 → 
    B = 6 → 
    (C = A + 8 ∨ C = A - 8) → 
    (D = B + 2 ∨ D = B - 2) → 
    CD = |C - D|)
:= sorry

end find_CD_values_l324_324111


namespace find_f_of_3_l324_324489

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l324_324489


namespace unique_perpendicular_line_l324_324744

variable {α β : Plane}
variable {A : Point}

-- Conditions: Given plane α is perpendicular to plane β and point A lies in plane α
def conditions (h1 : α.perpendicular β) (h2 : A ∈ α) : Prop := true

theorem unique_perpendicular_line (h1 : α.perpendicular β) (h2 : A ∈ α) :
  ∃! (L : Line), (L.perpendicular β) ∧ (A ∈ L) ∧ (L ⊆ α) :=
by
  sorry

end unique_perpendicular_line_l324_324744


namespace tangent_circles_count_l324_324912

theorem tangent_circles_count {a b : ℝ} :
  let circle1 := (λ x y : ℝ, (x - a)^2 - (y - b)^2 - 4 * (a^2 + b^2) = 0)
  let circle2 := (λ x y : ℝ, (x + a)^2 + (y + b)^2 - 4 * (a^2 + b^2) = 0)
  let radius := Real.sqrt (a^2 + b^2)
  (∃ circles : List (ℝ × ℝ × ℝ), circles.enum.count (λ (x y r : ℝ), (circle1 x y = 0 ∧ circle2 x y = 0 ∧ r = radius)) = 5 :=
sorry

end tangent_circles_count_l324_324912


namespace shoes_to_polish_l324_324243

-- Definitions based on conditions
def total_pairs := 10
def polished_percentage := 45 / 100

-- Goal: Number of shoes Austin needs to polish
theorem shoes_to_polish (total_pairs : ℕ) (polished_percentage : ℚ) : 
  let total_shoes := 2 * total_pairs in
  let polished_shoes := polished_percentage * total_shoes in
  total_shoes - polished_shoes = 11 :=
  by 
  let total_shoes := 2 * total_pairs
  let polished_shoes := polished_percentage * total_shoes
  sorry

end shoes_to_polish_l324_324243


namespace distances_from_P_to_l_are_correct_l324_324062

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.Cos θ, Real.sin θ)

def line_cartesian_equation (x y : ℝ) : Prop :=
  x + y = 1

def distance_from_point_to_line (x y : ℝ) (a b c : ℝ) : ℝ :=
  Real.abs (a * x + b * y + c) / Real.sqrt (a * a + b * b)

theorem distances_from_P_to_l_are_correct :
  ∀ (θ : ℝ), 
  line_cartesian_equation (fst (curve_C θ)) (snd (curve_C θ)) →
  Real.sqrt ((distance_from_point_to_line (fst (curve_C θ)) (snd (curve_C θ)) 1 1 (-1)) = Real.sqrt 2 * (2 - 1)) ∧
  Real.sqrt ((distance_from_point_to_line (fst (curve_C θ)) (snd (curve_C θ)) 1 1 (-1)) = Real.sqrt 2 * (-2 - 1)) :=
by
  sorry

end distances_from_P_to_l_are_correct_l324_324062


namespace sqrt_series_fixed_point_l324_324093

theorem sqrt_series_fixed_point : ∃ x : ℝ, x = 2 ∧ x = real.sqrt (2 + x) :=
by
  use 2
  split
  · rfl
  sorry

end sqrt_series_fixed_point_l324_324093


namespace tetrahedron_spheres_bound_l324_324115

theorem tetrahedron_spheres_bound (S_1 S_2 S_3 S_4 V : ℝ) :
  (∃ (ε1 ε2 ε3 ε4 : {x : ℝ // x = 1 ∨ x = -1}), 
    let r := 3 * V / (ε1.val * S_1 + ε2.val * S_2 + ε3.val * S_3 + ε4.val * S_4) in
    r > 0) → 
  (5 ≤ (∑ i in ({s | s = 1 ∨ s = -1 }.to_finset : finset ℝ), 
    if 3 * V / (i * S_1 + i * S_2 + i * S_3 + i * S_4) > 0 then 1 else 0) ∧ 
    (∑ i in ({s | s = 1 ∨ s = -1 }.to_finset : finset ℝ), 
    if 3 * V / (i * S_1 + i * S_2 + i * S_3 + i * S_4) > 0 then 1 else 0) ≤ 8) :=
sorry

end tetrahedron_spheres_bound_l324_324115


namespace prob_soldier_shooting_l324_324798

variables (P₉ P₈₋₉ P₇₋₈ P₆₋₇ : ℝ)

-- Conditions
def prob_9_or_above : Prop := P₉ = 0.18
def prob_8_to_9 : Prop := P₈₋₉ = 0.51
def prob_7_to_8 : Prop := P₇₋₈ = 0.15
def prob_6_to_7 : Prop := P₆₋₇ = 0.09

-- Questions to prove
def prob_8_or_above (P₉ P₈₋₉ : ℝ) : ℝ := P₉ + P₈₋₉
def prob_pass (P₉ P₈₋₉ P₇₋₈ P₆₋₇ : ℝ) : ℝ := P₉ + P₈₋₉ + P₇₋₈ + P₆₋₇

theorem prob_soldier_shooting :
  prob_9_or_above P₉ ∧
  prob_8_to_9 P₈₋₉ ∧
  prob_7_to_8 P₇₋₈ ∧
  prob_6_to_7 P₆₋₇ →
  prob_8_or_above P₉ P₈₋₉ = 0.69 ∧ prob_pass P₉ P₈₋₉ P₇₋₈ P₆₋₇ = 0.93 :=
by
sor ry  -- Proof should be done here, this placeholder skips the proof

end prob_soldier_shooting_l324_324798


namespace find_y_l324_324144

theorem find_y (y : ℝ) (h : 9 * y^2 + 36 * y^2 + 9 * y^2 = 1300) : 
  y = Real.sqrt 1300 / Real.sqrt 54 :=
by 
  sorry

end find_y_l324_324144


namespace sum_less_than_518_l324_324965

variable (a : ℕ → ℕ → ℕ)
variable (sum_rows : ℕ → ℕ)
variable (sum_diagonals : ℕ)
variable (n m : ℕ)

-- Given conditions
def condition1 : Prop := ∑ i in range 8, ∑ j in range 8, a i j = 1956
def condition2 : Prop := sum_diagonals = 112
def condition3 : Prop := ∀ i j, a i j = a (7 - i) (7 - j)
def condition4 : Prop := ∀ i, sum_rows i = ∑ j in range 8, a i j

-- Proof goal
theorem sum_less_than_518 
  (cond1 : condition1 a)
  (cond2 : condition2 sum_diagonals)
  (cond3 : condition3 a)
  (cond4 : condition4 a sum_rows) :
  ∀ i, sum_rows i < 518 := 
sorry

end sum_less_than_518_l324_324965


namespace polynomial_divisibility_l324_324042

theorem polynomial_divisibility (k : ℝ) : (∃ q : ℝ[X], X^2 + 2 * k * X - 3 * k = (X - 1) * q) → k = 1 :=
by
  sorry

end polynomial_divisibility_l324_324042


namespace expected_value_two_consecutive_red_balls_l324_324359

noncomputable def expected_draws_until_consecutive_red : ℝ :=
  let E : ℝ := sorry
  in E

theorem expected_value_two_consecutive_red_balls : expected_draws_until_consecutive_red = 4 := sorry

end expected_value_two_consecutive_red_balls_l324_324359


namespace decreasing_interval_implies_a_ge_two_l324_324889

-- The function f is given
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 3

-- Defining the condition for f(x) being decreasing in the interval (-8, 2)
def is_decreasing_in_interval (a : ℝ) : Prop :=
  ∀ x y : ℝ, (-8 < x ∧ x < y ∧ y < 2) → f x a > f y a

-- The proof statement
theorem decreasing_interval_implies_a_ge_two (a : ℝ) (h : is_decreasing_in_interval a) : a ≥ 2 :=
sorry

end decreasing_interval_implies_a_ge_two_l324_324889


namespace ratio_of_seconds_l324_324152

theorem ratio_of_seconds (x : ℕ) :
  (12 : ℕ) / 8 = x / 240 → x = 360 :=
by
  sorry

end ratio_of_seconds_l324_324152


namespace measure_8_liters_possible_l324_324001

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l324_324001


namespace car_speed_is_80_l324_324585

def speed_of_car (distance speed_train_ratio train_stop_time train_time_same_distance result : ℝ) : Prop :=
  let car_speed := result in
  let train_speed := speed_train_ratio * car_speed in
  let travel_time := distance / car_speed in
  car_speed = result ∧ 
  75 = train_speed * (travel_time - train_stop_time)

theorem car_speed_is_80 :
  speed_of_car 75 1.5 (12.5 / 60) 1.0 80 :=
by 
  sorry

end car_speed_is_80_l324_324585


namespace pizza_topping_count_l324_324612

theorem pizza_topping_count (n : ℕ) (hn : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by
  sorry

end pizza_topping_count_l324_324612


namespace area_of_pentagon_eq_fraction_l324_324520

theorem area_of_pentagon_eq_fraction (w : ℝ) (h : ℝ) (fold_x : ℝ) (fold_y : ℝ)
    (hw3 : h = 3 * w)
    (hfold : fold_x = fold_y)
    (hx : fold_x ^ 2 + fold_y ^ 2 = 3 ^ 2)
    (hx_dist : fold_x = 4 / 3) :
  (3 * (1 / 2) + fold_x / 2) / (3 * w) = 13 / 18 := 
by 
  sorry

end area_of_pentagon_eq_fraction_l324_324520


namespace thomas_blocks_total_l324_324919

theorem thomas_blocks_total 
  (first_stack : ℕ)
  (second_stack : ℕ)
  (third_stack : ℕ)
  (fourth_stack : ℕ)
  (fifth_stack : ℕ) 
  (h1 : first_stack = 7)
  (h2 : second_stack = first_stack + 3)
  (h3 : third_stack = second_stack - 6)
  (h4 : fourth_stack = third_stack + 10)
  (h5 : fifth_stack = 2 * second_stack) :
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack = 55 :=
by
  sorry

end thomas_blocks_total_l324_324919


namespace a_share_is_1400_l324_324190

-- Definitions for the conditions
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def share_B : ℕ := 2200

-- Definition for the ratios
def ratio_A : ℚ := investment_A / 1000
def ratio_B : ℚ := investment_B / 1000
def ratio_C : ℚ := investment_C / 1000

-- Sum of ratios
def sum_ratios : ℚ := ratio_A + ratio_B + ratio_C

-- Total profit P can be deduced from B's share
def total_profit : ℚ := share_B * sum_ratios / ratio_B

-- Goal: Prove that A's share is $1400
def share_A : ℚ := ratio_A * total_profit / sum_ratios

theorem a_share_is_1400 : share_A = 1400 :=
sorry

end a_share_is_1400_l324_324190


namespace triangles_with_two_equal_sides_count_l324_324565

theorem triangles_with_two_equal_sides_count :
  let lengths := [2, 3, 5, 7, 11]
  in ∃ n, n = 14 ∧ (∀ a b ∈ lengths, (2 * a > b ∧ ∃! (a1, a2, a3: ℕ), a1 = a ∧ a2 = a ∧ a3 = b ∨
                                       2 * b > a ∧ ∃! (b1, b2, b3: ℕ), b1 = b ∧ b2 = b ∧ b3 = a)) :=
sorry

end triangles_with_two_equal_sides_count_l324_324565


namespace num_8tuples_satisfying_condition_l324_324709

theorem num_8tuples_satisfying_condition :
  (∃! (y : Fin 8 → ℝ),
    (2 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + 
    (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + 
    (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 4 / 9) :=
sorry

end num_8tuples_satisfying_condition_l324_324709


namespace matrix_problem_l324_324082

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)
variable (I : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = !![2, 1; 4, 3]) :
  B * A = !![2, 1; 4, 3] :=
sorry

end matrix_problem_l324_324082


namespace n_pow_b_eq_9_l324_324035

noncomputable def n : ℝ := 3 ^ 0.15
noncomputable def b : ℝ := 13.33333333333333

theorem n_pow_b_eq_9 : n ^ b = 9 := by
  -- Proof omitted
  sorry

end n_pow_b_eq_9_l324_324035


namespace find_f_of_3_l324_324516

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l324_324516


namespace largest_angle_of_triangle_l324_324896

theorem largest_angle_of_triangle
  (a b y : ℝ)
  (h1 : a = 60)
  (h2 : b = 70)
  (h3 : a + b + y = 180) :
  max a (max b y) = b :=
by
  sorry

end largest_angle_of_triangle_l324_324896


namespace rooks_in_quadrants_l324_324854

theorem rooks_in_quadrants (board : fin 100 × fin 100 → bool)
  (h1 : ∀ i : fin 100, ∃! j : fin 100, board (i, j) = true)
  (h2 : ∀ j : fin 100, ∃! i : fin 100, board (i, j) = true) :
  (∑ i in finset.finRange 50, ∑ j in finset.finRange 50, if board (i, j + 50) then 1 else 0) =
  (∑ i in finset.finRange 50, ∑ j in finset.finRange 50, if board (i + 50, j) then 1 else 0) :=
sorry

end rooks_in_quadrants_l324_324854


namespace probability_all_quitters_same_tribe_l324_324902

-- Definitions of the problem conditions
def total_contestants : ℕ := 20
def tribe_size : ℕ := 10
def quitters : ℕ := 3

-- Definition of the binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem probability_all_quitters_same_tribe :
  (choose tribe_size quitters + choose tribe_size quitters) * 
  (total_contestants.choose quitters) = 240 
  ∧ ((choose tribe_size quitters + choose tribe_size quitters) / (total_contestants.choose quitters)) = 20 / 95 :=
by
  sorry

end probability_all_quitters_same_tribe_l324_324902


namespace time_to_extinguish_fire_l324_324214

def hose_delivery_before_pressure_drop : ℕ := 18 + 22 + 16 + 20 + 24
def total_water_required : ℕ := 6000
def initial_delivery_time : ℕ := 20
def reduced_hose_delivery_after_pressure_drop : ℕ := (18 / 2) + (22 / 2) + (16 / 2) + (20 / 2) + (24 / 2)
def water_delivered_in_initial_time : ℕ := hose_delivery_before_pressure_drop * initial_delivery_time
def remaining_water_after_initial_time : ℕ := total_water_required - water_delivered_in_initial_time

theorem time_to_extinguish_fire :
  hose_delivery_before_pressure_drop = 100 ∧
  reduced_hose_delivery_after_pressure_drop = 50 ∧
  total_water_required = 6000 ∧
  water_delivered_in_initial_time = 2000 ∧
  remaining_water_after_initial_time = 4000 ∧
  20 + (4000 / 50) = 100 := by
  split
  all_goals { sorry }

end time_to_extinguish_fire_l324_324214


namespace number_of_ordered_quadruples_summing_88_l324_324095

def positive_odd_integer (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

theorem number_of_ordered_quadruples_summing_88 :
  let n := { quadruple : (ℕ × ℕ × ℕ × ℕ) // (∀ x ∈ {quadruple.1, quadruple.2, quadruple.3, quadruple.4}, positive_odd_integer x) ∧ (quadruple.1 + quadruple.2 + quadruple.3 + quadruple.4 = 88) },
  n.card = 14190 :=
by
  sorry

end number_of_ordered_quadruples_summing_88_l324_324095


namespace system_solution_b_l324_324762

theorem system_solution_b (x y b : ℚ) 
  (h1 : 4 * x + 2 * y = b) 
  (h2 : 3 * x + 7 * y = 3 * b) 
  (hy : y = 3) : 
  b = 22 / 3 := 
by
  sorry

end system_solution_b_l324_324762


namespace ordered_triples_count_l324_324286

theorem ordered_triples_count :
  {n : ℕ // n = 4} :=
sorry

end ordered_triples_count_l324_324286


namespace smallest_two_digit_integer_l324_324575

theorem smallest_two_digit_integer (n a b : ℕ) (h1 : n = 10 * a + b) (h2 : 2 * n = 10 * b + a + 5) (h3 : 1 ≤ a) (h4 : a ≤ 9) (h5 : 0 ≤ b) (h6 : b ≤ 9) : n = 69 := 
by 
  sorry

end smallest_two_digit_integer_l324_324575


namespace snail_displacement_at_55_days_l324_324647

theorem snail_displacement_at_55_days :
  ∑ n in Finset.range 55, (1 / (n + 1 : ℝ) - 1 / (n + 2 : ℝ)) = 55 / 56 :=
by
  sorry

end snail_displacement_at_55_days_l324_324647


namespace count_primes_in_sequence_l324_324013

-- Define the list as a sequence of numbers starting with 47, and each subsequent 
-- number is 47 concatenated to the previous ones forming a sequence of the form 47*(10^n + 10^(n-2) + ... + 10^0).
def sequence (n : ℕ) : ℕ :=
  47 * (list.range ((n + 1) * 2)).map (λ m, if m % 2 = 0 then 10^(m/2) else 0).sum

theorem count_primes_in_sequence : ∃! n, n = 1 ∧ prime (sequence 0)
  ∧ ∀ k > 0, ¬ prime (sequence k) :=
by
  sorry

end count_primes_in_sequence_l324_324013


namespace median_name_length_is_5_l324_324368

theorem median_name_length_is_5
  (names_lengths : List ℕ) :
  names_lengths = List.replicate 9 4 ++ List.replicate 6 5 ++ List.replicate 5 6 ++ List.replicate 3 7 ++ List.replicate 2 8 →
  List.median names_lengths = 5 :=
by 
  intro h
  sorry

end median_name_length_is_5_l324_324368


namespace find_line_eq_find_dot_product_l324_324334

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 8 + y^2 / 4 = 1)

def point_B1 : ℝ × ℝ := (0, -2)
def point_B2 : ℝ × ℝ := (0, 2)
def point_A : ℝ × ℝ := (0, -2 * Real.sqrt 2)
def line_l (k : ℝ) (x y : ℝ) : Prop := (y = k * x - 2 * Real.sqrt 2)

def intersects_ellipse (k : ℝ) : Prop := ∃ x1 x2 y1 y2, 
  (line_l k x1 y1 ∧ ellipse_eq x1 y1) ∧ 
  (line_l k x2 y2 ∧ ellipse_eq x2 y2) ∧ 
  (x1, y1) ≠ point_B1 ∧ 
  (x2, y2) ≠ point_B2 ∧ 
  (tan_angle (point_B1 fst y1) (point_B2 fst) = 2 * tan_angle (point_B1 x1 y1) (point_B2 x2 y2))

theorem find_line_eq (k : ℝ) (h : intersects_ellipse k) :
  k = 3 * Real.sqrt 14 / 14 ∨ k = -3 * Real.sqrt 14 / 14 :=
sorry

def vector_AR_R (x_R y_R : ℝ) : ℝ × ℝ := (x_R, y_R - (- 2 * Real.sqrt 2))
def vector_B1_B2 : ℝ × ℝ := (0, 4)

theorem find_dot_product (x_R y_R : ℝ) (hR : is_intersection point_B1 point_B2 x_R y_R) :
  vector_AR_R x_R y_R *= vector_B1_B2 = 4 * Real.sqrt 2 :=
sorry

end find_line_eq_find_dot_product_l324_324334


namespace perfect_square_trinomial_k_l324_324030

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a b : ℤ, (a = 1 ∨ a = -1) ∧ (b = 3 ∨ b = -3) ∧ (x : ℤ) → x^2 - k*x + 9 = (a*x + b)^2) ↔ (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l324_324030


namespace distance_difference_l324_324376

-- Definition of parametric equations of curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Definition of polar coordinate equation of line l
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin θ + 2 * ρ * Real.cos θ = 2

-- Given point P
def P : ℝ × ℝ := (1, 0)

-- Prove ||PA|-|PB|| = 2√5 / 5
theorem distance_difference {A B : ℝ × ℝ} (hA : A ∈ {p : ℝ × ℝ | curve_C_c (Real.arg p) p})
                            (hB : B ∈ {p : ℝ × ℝ | curve_C_c (Real.arg p) p}) :
  |(Real.dist P A) - (Real.dist P B)| = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end distance_difference_l324_324376


namespace cot_alpha_value_l324_324313

variable (α : ℝ)

noncomputable def sin := Real.sin
noncomputable def cos := Real.cos
noncomputable def cot (x : ℝ) := cos x / sin x

theorem cot_alpha_value :
  sin (2 * α) = -sin α ∧ α ∈ Ioo (π / 2) π → cot α = -√3 / 3 :=
by
  sorry

end cot_alpha_value_l324_324313


namespace longer_segment_dc_l324_324904
-- Import the Mathlib to leverage the necessary definitions and theorems

-- Definitions as per the conditions
variables {A B C D : Type} [is_triangle_const : triangle_const A B C]

-- Condition: the ratio of the sides
def side_ratio (a b c : ℝ) : Prop := a / b = 3 / 4 ∧ a / c = 3 / 5 ∧ b / c = 4 / 5

-- Given: the angle bisector and lengths
def angle_bisector_ratio (ad dc : ℝ) (ratio : ℝ) : Prop := ad / dc = ratio

-- Given the length of side AC
def length_AC (a : ℝ) : Prop := a = 15

-- The proof problem to show that DC is the longer segment
theorem longer_segment_dc {a b c ad dc : ℝ}
  (h_side_ratio : side_ratio a b c)
  (h_ratio : angle_bisector_ratio ad dc (3 / 4))
  (h_ac_length : length_AC a) :
  dc = (60 / 7) :=
sorry

end longer_segment_dc_l324_324904


namespace solve_for_g_l324_324346

def f : ℝ → ℝ := λ x, x^4 - 4 * x^2 + 3 * x - 1
def polynomial : ℝ → ℝ := λ x, x^2 - 3 * x + 5

theorem solve_for_g : ∃ g : (ℝ → ℝ), ∀ x : ℝ, f(x) + g(x) = polynomial x ∧ 
  g(x) = -x^4 + 5 * x^2 - 6 * x + 6 :=
begin
  use λ x, -x^4 + 5 * x^2 - 6 * x + 6,
  sorry,
end

end solve_for_g_l324_324346


namespace no_real_root_in_interval_l324_324831

def P (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem no_real_root_in_interval (a b c d : ℝ) 
    (h1 : min d (b + d) > max (abs c) (abs (a + c))) :
    ∀ x : ℝ, x ∈ Icc (-1) 1 → P a b c d x ≠ 0 := by
sorry

end no_real_root_in_interval_l324_324831


namespace shapes_final_positions_l324_324297

-- Definitions for shapes and their initial positions within the large circle
variables (A B C : Type) -- Triangle, smaller circle, rectangle
variables (InitialPosition : A → B → C → (ℝ × ℝ)) -- positions within the coordinate plane

-- Given transformations
def rotate_clockwise (θ : ℝ) (pos : (ℝ × ℝ)) : (ℝ × ℝ) := sorry -- to define the rotation
def reflect_vertical (pos : (ℝ × ℝ)) : (ℝ × ℝ) := sorry -- to define the reflection

-- Initial conditions
axiom initial_triangle_pos : (ℝ × ℝ)
axiom initial_smaller_circle_pos : (ℝ × ℝ)
axiom initial_rectangle_pos : (ℝ × ℝ)

-- Transformed positions after operations
def final_pos_triangle  := reflect_vertical (rotate_clockwise (150 * (π / 180)) initial_triangle_pos)
def final_pos_smaller_circle := reflect_vertical (rotate_clockwise (150 * (π / 180)) initial_smaller_circle_pos)
def final_pos_rectangle := reflect_vertical (rotate_clockwise (150 * (π / 180)) initial_rectangle_pos)

-- Expected final positions for proof
axiom expected_triangle_pos : (ℝ × ℝ) := initial_smaller_circle_pos
axiom expected_smaller_circle_pos : (ℝ × ℝ) := initial_triangle_pos
axiom expected_rectangle_pos : (ℝ × ℝ) := (initial_rectangle_pos.1, -initial_rectangle_pos.2) -- Assuming reflection

-- Theorem statement (with sketch proof for now)
theorem shapes_final_positions :
  final_pos_triangle = expected_triangle_pos ∧
  final_pos_smaller_circle = expected_smaller_circle_pos ∧
  final_pos_rectangle = expected_rectangle_pos :=
by
  -- Your proving steps would go here
  sorry

end shapes_final_positions_l324_324297


namespace diagonals_in_30_sided_polygon_l324_324692

theorem diagonals_in_30_sided_polygon : 
  let n := 30 in n * (n - 3) / 2 = 405 :=
by
  let n := 30
  calc
    n * (n - 3) / 2 = 30 * 27 / 2 := by rfl
    ... = 405 := by norm_num

end diagonals_in_30_sided_polygon_l324_324692


namespace max_distance_two_spheres_l324_324934

-- Define the centers and radii of the spheres
def center1 : ℝ × ℝ × ℝ := (5, 15, -7)
def radius1 : ℝ := 23
def center2 : ℝ × ℝ × ℝ := (-12, 3, 18)
def radius2 : ℝ := 91

-- Calculate the distance between centers
def dist_centers (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

-- Define the proof statement
theorem max_distance_two_spheres :
  let d := dist_centers center1 center2
  in d = Real.sqrt 1058 → 
     23 + d + 91 = 114 + Real.sqrt 1058 :=
by
  simp [dist_centers, center1, center2, radius1, radius2]
  intro h
  rw [h]
  simp
  sorry

end max_distance_two_spheres_l324_324934


namespace team_average_typing_speed_l324_324466

-- Definitions of typing speeds of each team member
def typing_speed_rudy := 64
def typing_speed_joyce := 76
def typing_speed_gladys := 91
def typing_speed_lisa := 80
def typing_speed_mike := 89

-- Number of team members
def number_of_team_members := 5

-- Total typing speed calculation
def total_typing_speed := typing_speed_rudy + typing_speed_joyce + typing_speed_gladys + typing_speed_lisa + typing_speed_mike

-- Average typing speed calculation
def average_typing_speed := total_typing_speed / number_of_team_members

-- Theorem statement
theorem team_average_typing_speed : average_typing_speed = 80 := by
  sorry

end team_average_typing_speed_l324_324466


namespace domino_arrangement_paths_l324_324100

open Nat

-- Define conditions as inputs
def gridWidth : ℕ := 6
def gridHeight : ℕ := 5
def dominoWidth : ℕ := 1
def dominoHeight : ℕ := 2
def dominoCount : ℕ := 5
def movesRight : ℕ := 4
def movesDown : ℕ := 5
def totalMoves : ℕ := movesRight + movesDown

-- State the main theorem to be proven
theorem domino_arrangement_paths : 
  (∃ (C D : Fin gridWidth × Fin gridHeight), 
  C = (Fin.mk 0 (by simp)) ∧ 
  D = (Fin.mk (gridWidth - 1) (by simp), 
  (number_of_paths : ℕ) = Nat.choose totalMoves movesRight)) = 126 := 
by 
  sorry

end domino_arrangement_paths_l324_324100


namespace distinct_real_solutions_l324_324837

noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem distinct_real_solutions : 
  { c : ℝ | g (g (g (g c))) = 5 }.finite.to_finset.card = 16 :=
by
  sorry

end distinct_real_solutions_l324_324837


namespace parabola_equation_l324_324906

theorem parabola_equation (P : ℝ × ℝ) (hp : P = (4, -2)) : 
  ∃ m : ℝ, (∀ x y : ℝ, (y^2 = m * x) → (x, y) = P) ∧ (m = 1) :=
by
  have m_val : 1 = 1 := rfl
  sorry

end parabola_equation_l324_324906


namespace term_1004_is_3036_l324_324840

-- Definition of the sequence using the pattern provided in conditions
noncomputable def sequence : ℕ → ℕ 
| n :=
  let k := (nat.find (λ k, k*(k+1)/2 > n - 1)) - 1
  in if n - (k * (k + 1) / 2 + 1) < 0 then
       1
     else
       let offset := n - (k * (k + 1) / 2) - 1
       in 2 * k^2 - 7 * k + 6 + offset * 4

-- The property to prove
theorem term_1004_is_3036 : sequence 1004 = 3036 := 
begin
  sorry
end

end term_1004_is_3036_l324_324840


namespace no_8_consecutive_integers_exception_l324_324864

def f (x y : ℤ) : ℤ := 7 * x^2 + 9 * x * y - 5 * y^2

theorem no_8_consecutive_integers_exception (x y n : ℤ) :
  12 ≤ n ∧ n ≤ 19 → (f x y).natAbs ≠ n :=
by sorry

end no_8_consecutive_integers_exception_l324_324864


namespace cube_construction_distinct_ways_l324_324564

theorem cube_construction_distinct_ways : 
  let n_white := 6
  let n_black := 2
  let n_total := 8
  let rotations := 24
  (n_white + n_black = n_total) ∧ 
  rotations = 24 -> 
  distinct_constructions n_white n_black = 3 :=
by
  sorry

end cube_construction_distinct_ways_l324_324564


namespace virus_memory_occupation_l324_324226

-- Define the initial conditions
def initial_memory_kb : ℕ := 2
def doubling_interval_minutes : ℕ := 3

-- Define constants for memory size conversion
def mb_to_kb (mb : ℕ) : ℕ := mb * 1024

-- Problem statement
theorem virus_memory_occupation :
  ∃ (n : ℕ), 2 ^ (n + 1) * initial_memory_kb = mb_to_kb 64 ∧ n * doubling_interval_minutes = 45 :=
begin
  -- This is where the proof would go, but it's skipped for now
  sorry
end

end virus_memory_occupation_l324_324226


namespace quadrilateral_area_relation_l324_324128

variables {Point : Type*} [add_comm_group Point] [vector_space ℝ Point]
variables A B C D A' B' C' D' O : Point
variables (S S' : ℝ)

-- Lean statement to prove that S = 2S'
theorem quadrilateral_area_relation
  (hOA : O - A = A' - B')
  (hOB : O - B = B' - C')
  (hOC : O - C = C' - D')
  (hOD : O - D = D' - A')
  (area_ABCD : ∀ (p1 p2 p3 p4 : Point), fit_area p1 p2 p3 p4 → S)
  (area_A'B'C'D' : ∀ (p1 p2 p3 p4 : Point), fit_area p1 p2 p3 p4 → S'):
  S = 2 * S' :=
sorry

end quadrilateral_area_relation_l324_324128


namespace geometric_sequence_first_term_l324_324907

theorem geometric_sequence_first_term (a_3 a_4 : ℝ) (h3 : a_3 = 36) (h4 : a_4 = 54) :
  ∃ a r : ℝ, a * r^2 = a_3 ∧ a * r^3 = a_4 ∧ r = 3/2 ∧ a = 16 := by
  use 16, 3/2
  split
  -- Show that 16 * (3/2)^2 = a_3
  { calc 
      16 * (3/2)^2
          = 16 * (9/4) : by sorry
      ... = 36 : by sorry },
  split
  -- Show that 16 * (3/2)^3 = a_4
  { calc 
      16 * (3/2)^3
          = 16 * (27/8) : by sorry
      ... = 54 : by sorry },
  -- Show that r = 3/2
  { rfl },
  -- Show that a = 16
  { rfl }

end geometric_sequence_first_term_l324_324907


namespace factory_toys_per_day_l324_324192

theorem factory_toys_per_day (toys_per_week : ℤ) (days_per_week : ℤ) 
    (h_toys_per_week : toys_per_week = 5505) 
    (h_days_per_week : days_per_week = 5) :
    (toys_per_day : ℤ) (h_toys_per_day : toys_per_day = toys_per_week / days_per_week) := sorry

end factory_toys_per_day_l324_324192


namespace initial_bees_l324_324915

theorem initial_bees (B : ℕ) (h : B + 7 = 23) : B = 16 :=
by {
  sorry
}

end initial_bees_l324_324915


namespace exam_question_combinations_l324_324373

theorem exam_question_combinations : 
  let questions := Finset.range 9 in
  let first_five := Finset.range 5 in
  let count_ways_to_choose := 
    (@Finset.choose (Fin 5) _ _ ⟨3, by norm_num⟩).card * 
    (@Finset.choose (Fin 4) _ _ ⟨3, by norm_num⟩).card +
    (@Finset.choose (Fin 5) _ _ ⟨4, by norm_num⟩).card * 
    (@Finset.choose (Fin 4) _ _ ⟨2, by norm_num⟩).card +
    (@Finset.choose (Fin 5) _ _ ⟨5, by norm_num⟩).card * 
    (@Finset.choose (Fin 4) _ _ ⟨1, by norm_num⟩).card
  in count_ways_to_choose = 74 :=
by sorry

end exam_question_combinations_l324_324373


namespace cindy_added_pens_l324_324189

-- Define the initial number of pens
def initial_pens : ℕ := 5

-- Define the number of pens given by Mike
def pens_from_mike : ℕ := 20

-- Define the number of pens given to Sharon
def pens_given_to_sharon : ℕ := 10

-- Define the final number of pens
def final_pens : ℕ := 40

-- Formulate the theorem regarding the pens added by Cindy
theorem cindy_added_pens :
  final_pens = initial_pens + pens_from_mike - pens_given_to_sharon + 25 :=
by
  sorry

end cindy_added_pens_l324_324189


namespace least_possible_integer_for_friends_statements_l324_324604

theorem least_possible_integer_for_friends_statements 
    (M : Nat)
    (statement_divisible_by : Nat → Prop)
    (h1 : ∀ n, 1 ≤ n ∧ n ≤ 30 → statement_divisible_by n = (M % n = 0))
    (h2 : ∃ m, 1 ≤ m ∧ m < 30 ∧ (statement_divisible_by m = false ∧ 
                                    statement_divisible_by (m + 1) = false)) :
    M = 12252240 :=
by
  sorry

end least_possible_integer_for_friends_statements_l324_324604


namespace sum_invested_eq_additional_amount_l324_324228

variable (P x R : ℝ)

theorem sum_invested_eq_additional_amount :
  (P * (5 / 6) = x) ↔
  let SI₁ := (P * R * 15) / 100 in
  let SI₂ := (P * (R + 8) * 15) / 100 in
  SI₂ - SI₁ = x :=
by
  sorry

end sum_invested_eq_additional_amount_l324_324228


namespace enclosed_area_is_correct_l324_324603

noncomputable def area_between_curves : ℝ := 
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let cubic_parabola (x : ℝ) := - 1 / 2 * x^3 + 2 * x
  let x1 : ℝ := -2
  let x2 : ℝ := Real.sqrt 2
  -- Properly calculate the area between the two curves
  sorry

theorem enclosed_area_is_correct :
  area_between_curves = 3 * ( Real.pi + 1 ) / 2 :=
sorry

end enclosed_area_is_correct_l324_324603


namespace det_of_matrix_2x2_l324_324681

-- The given matrix
def matrix_2x2 := ![![5, -2], ![4, 3]]

-- The statement we want to prove
theorem det_of_matrix_2x2 : matrix.det matrix_2x2 = 23 := by 
  sorry

end det_of_matrix_2x2_l324_324681


namespace find_f_of_3_l324_324512

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l324_324512


namespace max_inscribed_triangle_area_sum_l324_324224

noncomputable def inscribed_triangle_area (a b : ℝ) (h_a : a = 12) (h_b : b = 13) : ℝ :=
  let s := min (a / (Real.sqrt 3 / 2)) (b / (1 / 2))
  (Real.sqrt 3 / 4) * s^2

theorem max_inscribed_triangle_area_sum :
  inscribed_triangle_area 12 13 (by rfl) (by rfl) = 48 * Real.sqrt 3 - 0 :=
by
  sorry

#eval 48 + 3 + 0
-- Expected Result: 51

end max_inscribed_triangle_area_sum_l324_324224


namespace nickel_probability_l324_324990

theorem nickel_probability (dimes nickels pennies : ℕ) 
  (ValueDimes ValueNickels ValuePennies : ℚ) 
  (WorthDime WorthNickel WorthPenny : ℚ) 
  (hDimes : ValueDimes = 5) 
  (hWorthDime : WorthDime = 0.10) 
  (hNickels : ValueNickels = 2.50) 
  (hWorthNickel : WorthNickel = 0.05) 
  (hPennies : ValuePennies = 1) 
  (hWorthPenny : WorthPenny = 0.01)
  (hNumDimes : dimes = (ValueDimes / WorthDime).toNat)
  (hNumNickels : nickels = (ValueNickels / WorthNickel).toNat)
  (hNumPennies : pennies = (ValuePennies / WorthPenny).toNat) :
  (nickels : ℚ) / (dimes + nickels + pennies) = 1 / 4 := 
sorry

end nickel_probability_l324_324990


namespace mod_of_complex_l324_324322

theorem mod_of_complex (z : ℂ) (h : (conj z) * Complex.I = 3 + 4 * Complex.I) : Complex.abs z = 5 := 
sorry

end mod_of_complex_l324_324322


namespace magnitude_F3_angle_F2_F3_is_90_l324_324045

-- Define the vectors and their properties
variables {V : Type*} [inner_product_space ℝ V] {F1 F2 F3 : V}
variables (magF1 magF2 angleF1F2 : ℝ)

-- Given conditions
def forces_in_equilibrium : Prop :=
  F1 + F2 + F3 = 0
def mag_F1 : Prop := 
  ∥F1∥ = 4
def mag_F2 : Prop := 
  ∥F2∥ = 2
def angle_F1_F2 : Prop :=
  real.angle.cos (F1.angle_with F2) = -0.5

-- Theorems to prove
theorem magnitude_F3 (h1 : forces_in_equilibrium) (h2 : mag_F1) (h3 : mag_F2) (h4 : angle_F1_F2) :
  ∥F3∥ = 2 * real.sqrt 3 :=
sorry

theorem angle_F2_F3_is_90 (h1 : forces_in_equilibrium) (h2 : mag_F1) (h3 : mag_F2) (h4 : angle_F1_F2) :
  real.angle.cos (F2.angle_with F3) = 0 :=
sorry

end magnitude_F3_angle_F2_F3_is_90_l324_324045


namespace real_number_satisfies_condition_l324_324043

theorem real_number_satisfies_condition (a : ℝ) (h : (a * complex.I) / (2 - complex.I) = 1 - 2 * complex.I) : a = -5 :=
by sorry

end real_number_satisfies_condition_l324_324043


namespace max_t_value_l324_324545

variable (students teachers : ℕ)
variable (knows_st: students → teachers → Prop)
variable (ai bj : teachers → ℕ) -- ai is the number of students known by teacher i, bj is the number of teachers known by student j
variable (t : ℕ)

-- Conditions
variable (cond1 : students = 2006)
variable (cond2 : teachers = 14)
variable (cond3 : ∀ j, ∃ i, knows_st j i)
variable (cond4 : ∀ i j, knows_st j i → (ai i) / (bj j) ≥ t)

-- Statement to prove
theorem max_t_value : t ≤ 143 :=
sorry

end max_t_value_l324_324545


namespace present_age_of_son_l324_324194

theorem present_age_of_son (S F : ℕ)
  (h1 : F = S + 24)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  -- The proof is omitted, as per instructions.
  sorry
}

end present_age_of_son_l324_324194


namespace snail_displacement_at_55_days_l324_324648

theorem snail_displacement_at_55_days :
  ∑ n in finset.range 55, (1 / (n + 1) - 1 / (n + 2)) = 55 / 56 := 
sorry

end snail_displacement_at_55_days_l324_324648


namespace investment_difference_l324_324826

noncomputable def calc_investment (P r t : ℕ) : ℝ := 
  P * (1 + r : ℝ) ^ t

noncomputable def calc_monthly_investment (P : ℕ) (r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investment_difference :
  let P := 50000
  let r := 0.05
  let y := 2
  let m := 12
  let jose_final_amount := calc_investment P r y
  let patricia_final_amount := calc_monthly_investment P r m y
  (Int.round patricia_final_amount) - (Int.round jose_final_amount) = 111 :=
by
  sorry

end investment_difference_l324_324826


namespace pizza_topping_count_l324_324627

theorem pizza_topping_count (n : ℕ) (h : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by {
  rw h,
  simp [nat.choose],
  sorry,
}

end pizza_topping_count_l324_324627


namespace find_f_4_l324_324091

noncomputable def f : ℕ → ℝ
-- Definition of f

axiom functional_eq (f : ℕ → ℝ) (x y : ℕ) : f(x + y) = f(x) * f(y)

axiom initial_value (f : ℕ → ℝ) (k : ℝ) : f 19 = 524288 * k

theorem find_f_4 (f : ℕ → ℝ) (k : ℝ) (functional_eq : ∀ x y : ℕ, f(x + y) = f(x) * f(y)) (initial_value : f 19 = 524288 * k) : 
  f 4 = 16 * k^(4 / 19) :=
sorry

end find_f_4_l324_324091


namespace only_prime_in_list_is_47_l324_324019

def is_concatenation_of_4747 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 47 * (4747 ^ k)

theorem only_prime_in_list_is_47 : 
  (∃ n : ℕ, is_prime n ∧ is_concatenation_of_4747 n) → n = 47 := 
sorry

end only_prime_in_list_is_47_l324_324019


namespace problem_statement_l324_324258

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem problem_statement : star A B = {1, 7} := by
  sorry

end problem_statement_l324_324258


namespace dad_additional_money_l324_324673

-- Define the conditions in Lean
def daily_savings : ℕ := 35
def days : ℕ := 7
def total_savings_before_doubling := daily_savings * days
def doubled_savings := 2 * total_savings_before_doubling
def total_amount_after_7_days : ℕ := 500

-- Define the theorem to prove
theorem dad_additional_money : (total_amount_after_7_days - doubled_savings) = 10 := by
  sorry

end dad_additional_money_l324_324673


namespace problem_statement_l324_324283

-- Define z * x as the greatest positive even integer less than or equal to y + x
def custom_mul (z x y : ℝ) : ℝ := 
  if even (floor (y + x)) then floor (y + x)
  else floor (y + x) - 1

-- Given y = 6, prove that (6.25 * 3.5) - (6.25 * 3.5) = 0
theorem problem_statement (y : ℝ) (h_y : y = 6) :
  (custom_mul 6.25 3.5 y) - (custom_mul 6.25 3.5 y) = 0 :=
by
  sorry

end problem_statement_l324_324283


namespace units_digit_of_k3_plus_5k_l324_324086

def k : ℕ := 2024^2 + 3^2024

theorem units_digit_of_k3_plus_5k (k := 2024^2 + 3^2024) : 
  ((k^3 + 5^k) % 10) = 8 := 
by 
  sorry

end units_digit_of_k3_plus_5k_l324_324086


namespace bahs_equivalent_l324_324032

noncomputable def bahs_to_rahs (bahs : ℕ) : ℕ := (2 * bahs) / 3
noncomputable def rahs_to_yahs (rahs : ℕ) : ℕ := (3 * rahs) / 2
noncomputable def yahs_to_rahs (yahs : ℕ) : ℕ := (2 * yahs) / 3
noncomputable def rahs_to_bahs (rahs : ℕ) : ℕ := (3 * rahs) / 2

theorem bahs_equivalent : 
  (bahs_to_rahs 18 = 27) → (rahs_to_yahs 12 = 18) → yahs_to_rahs 1200 = 800 → rahs_to_bahs 800 = 1600 / 3 →
  ∃ bahs : ℕ, bahs = 1600 / 3 :=
by
  intros h1 h2 h3 h4
  use 1600 / 3
  sorry

end bahs_equivalent_l324_324032


namespace div_poly_iff_even_l324_324446

noncomputable def g (x : ℂ) (n : ℕ) : ℂ := ∑ i in Finset.range (n + 1), x^(2 * i)

noncomputable def f (x : ℂ) (n : ℕ) : ℂ := ∑ i in Finset.range (n + 1), x^(4 * i)

theorem div_poly_iff_even (n : ℕ) : 
  (∀ x : ℂ, g x n ∣ f x n) ↔ Even n :=
sorry

end div_poly_iff_even_l324_324446


namespace gcd_pow_sub_l324_324568

theorem gcd_pow_sub (h1001 h1012 : ℕ) (h : 1001 ≤ 1012) : 
  (Nat.gcd (2 ^ 1001 - 1) (2 ^ 1012 - 1)) = 2047 := sorry

end gcd_pow_sub_l324_324568


namespace hyperbola_eccentricity_l324_324148

-- Defining the hyperbola and the conditions
variables {a b c x0 y0 : ℝ} (ha : a > 0) (hb : b > 0)
-- The hyperbola definition
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
-- Left focus of the hyperbola
def focus := -c
-- Symmetric point A of F with respect to the given line
def symmetric_point (x0 y0 : ℝ) := x0 = c / 2 ∧ y0 = (sqrt 3 * c) / 2

-- Main theorem to prove that the eccentricity 'e' of the hyperbola is sqrt(3) + 1
theorem hyperbola_eccentricity (h0 : c = sqrt (a^2 + b^2))
  (h1 : symmetric_point x0 y0)
  (h2 : hyperbola x0 y0) :
  c / a = sqrt 3 + 1 :=
begin
  sorry
end

end hyperbola_eccentricity_l324_324148


namespace right_triangle_third_side_l324_324787

theorem right_triangle_third_side (a b : ℝ) 
  (h_eq : sqrt (a^2 - 6 * a + 9) + |b - 4| = 0) : 
  (∃ c : ℝ, (c = sqrt (a^2 + b^2) ∨ c = sqrt (a^2 - b^2) ∨ c = sqrt (b^2 - a^2)) ∧ (c = 5 ∨ c = sqrt 7)) :=
by {
  sorry
}

end right_triangle_third_side_l324_324787


namespace number_of_balls_l324_324597

theorem number_of_balls (x : ℕ) (h : x - 20 = 30 - x) : x = 25 :=
sorry

end number_of_balls_l324_324597


namespace arithmetic_sequence_condition_l324_324806

theorem arithmetic_sequence_condition (a : ℕ → ℕ) 
(h1 : a 4 = 4) 
(h2 : a 3 + a 8 = 5) : 
a 7 = 1 := 
sorry

end arithmetic_sequence_condition_l324_324806


namespace orthogonal_OH_MN_l324_324592

noncomputable theory

open_locale classical

variables {A B C D E F H O M N : Type} [metric_space A] [metric_space B] 
  [metric_space C] [metric_space D] [metric_space E] [metric_space F] 
  [metric_space H] [metric_space O] [metric_space M] [metric_space N]

/-- 
  Let \( \triangle ABC \) be a triangle with orthocenter \( H \).
  \( AD, BE, CF \) are altitudes of the triangle.
  The extensions of \( ED \) and \( AB \) intersect at point \( M \).
  The extensions of \( DF \) and \( CA \) intersect at point \( N \).
  Prove that \( OH \perp MN \).
 -/
theorem orthogonal_OH_MN
  (h_triangle : metric_space A ∧ metric_space B ∧ metric_space C)
  (h_altitudes : metric_space D ∧ metric_space E ∧ metric_space F)
  (h_orthocenter : metric_space H)
  (h_extensions : metric_space M ∧ metric_space N)
  (H_is_orthocenter : is_orthocenter A B C H)
  (D_altitude : altitude A D H)
  (E_altitude : altitude B E H)
  (F_altitude : altitude C F H)
  (M_intersection : intersection (extension D E) (extension A B) M)
  (N_intersection : intersection (extension D F) (extension C A) N)
  (O_point : metric_space O) :
  orthogonal (line O H) (line M N) :=
begin
  sorry
end

end orthogonal_OH_MN_l324_324592


namespace count_prime_numbers_in_list_only_prime_in_list_is_47_number_of_primes_in_list_l324_324016

theorem count_prime_numbers_in_list : 
  ∀ (n : ℕ), (∃ k : ℕ, n = 47 * ((10^k - 1) / 9)) → n ≠ 47 =→ n % 47 = 0 → isPrime n → False :=
by
  assume n hn h_eq_prime_47 h_divisible it_is_prime
  sorry

theorem only_prime_in_list_is_47 : ∀ (n : ℕ), n ∈ { num | ∃ k : ℕ, num = 47 * ((10^k - 1) / 9) } → (isPrime n ↔ n = 47) := 
by
  assume n hn
  split
    assume it_is_prime
    by_cases h_case : n = 47
      case inl => exact h_case
      case inr =>
        obtain ⟨k, hk⟩ := hn
        have h_mod : n % 47 = 0 := by rw [hk, nat.mul_mod_right]
        apply_false_from (count_prime_numbers_in_list n) hk h_case h_mod it_is_prime
        contradiction
    assume it_is_47
    exact by norm_num [it_is_47]
    sorry

theorem number_of_primes_in_list : ∀ (l : List ℕ), (∀ (n : ℕ), n ∈ l → ∃ k : ℕ, n = 47 * ((10^k - 1) / 9)) → l.filter isPrime = [47] :=
by
  sorry

end count_prime_numbers_in_list_only_prime_in_list_is_47_number_of_primes_in_list_l324_324016


namespace simpsons_hats_l324_324244

variable (S : ℕ)
variable (O : ℕ)

-- Define the conditions: O'Brien's hats before losing one
def obriens_hats_before : Prop := O = 2 * S + 5

-- Define the current number of O'Brien's hats
def obriens_current_hats : Prop := O = 34 + 1

-- Main theorem statement
theorem simpsons_hats : obriens_hats_before S O ∧ obriens_current_hats O → S = 15 := 
by
  sorry

end simpsons_hats_l324_324244


namespace bruce_purchased_grapes_l324_324674

-- Define the assumptions as constants or hypotheses
constants (G : ℕ) -- the kilograms of grapes
constants (cost_grapes cost_mangoes total_amount : ℕ)

-- Specify the conditions
axiom cost_grapes_def : cost_grapes = 70 * G
axiom cost_mangoes_def : cost_mangoes = 9 * 55
axiom total_amount_def : total_amount = cost_grapes + cost_mangoes
axiom total_paid : total_amount = 985

-- The theorem we need to prove
theorem bruce_purchased_grapes : G = 7 :=
by
  have h_cost_mangoes : cost_mangoes = 495, from calc
    cost_mangoes = 9 * 55 : cost_mangoes_def
    ... = 495 : rfl,
  have h_total_amount : total_amount = 70 * G + 495, from calc
    total_amount = cost_grapes + cost_mangoes : total_amount_def
    ... = 70 * G + 495 : by rw [cost_grapes_def, h_cost_mangoes],
  have h_total_eq : 70 * G + 495 = 985, from calc
    70 * G + 495 = total_amount : h_total_amount.symm
    ... = 985 : total_paid,
  have h_solve_G : 70 * G = 490, from calc
    70 * G = 985 - 495 : by linarith
    ... = 490 : rfl,
  calc
    G = 490 / 70 : by rw [←Nat.div_eq_of_eq_mul (Nat.eq_mul_of_div h_solve_G)]
    ... = 7 : by norm_num

-- Sorry is used here just to illustrate
-- REAL PROOF:

end bruce_purchased_grapes_l324_324674


namespace marbles_each_friend_is_16_l324_324773

-- Define the initial condition
def initial_marbles : ℕ := 100

-- Define the marbles Harold kept for himself
def kept_marbles : ℕ := 20

-- Define the number of friends Harold shared the marbles with
def num_friends : ℕ := 5

-- Define the marbles each friend receives
def marbles_per_friend (initial kept : ℕ) (friends : ℕ) : ℕ :=
  (initial - kept) / friends

-- Prove that each friend gets 16 marbles
theorem marbles_each_friend_is_16 : marbles_per_friend initial_marbles kept_marbles num_friends = 16 :=
by
  unfold initial_marbles kept_marbles num_friends marbles_per_friend
  exact Nat.mk_eq Nat.zero 16 sorry

end marbles_each_friend_is_16_l324_324773


namespace a_parallel_b_l324_324733

variable {Line : Type} (a b c : Line)

-- Definition of parallel lines
def parallel (x y : Line) : Prop := sorry

-- Conditions
axiom a_parallel_c : parallel a c
axiom b_parallel_c : parallel b c

-- Theorem to prove a is parallel to b given the conditions
theorem a_parallel_b : parallel a b :=
by
  sorry

end a_parallel_b_l324_324733


namespace sought_circle_equation_l324_324274

def circle_passing_through_point (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x = 0) → (y = 2) → x^2 + y^2 + D * x + E * y + F = 0

def chord_lies_on_line (D E F : ℝ) : Prop :=
  (D + 1) / 5 = (E - 2) / 2 ∧ (D + 1) / 5 = (F + 3)

theorem sought_circle_equation :
  ∃ (D E F : ℝ), 
  circle_passing_through_point D E F ∧ 
  chord_lies_on_line D E F ∧
  (D = -6) ∧ (E = 0) ∧ (F = -4) ∧ 
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 6 * x - 4 = 0 :=
by
  sorry

end sought_circle_equation_l324_324274


namespace sticks_left_sticks_left_l324_324187

theorem sticks_left (original totalSticks : ℕ) (pickedSticks : ℕ) (H_original : totalSticks = 99) (H_picked : pickedSticks = 38) : totalSticks - pickedSticks = 61 :=
by
  rw [H_original, H_picked]
  rfl

-- This leaves a final clean theorem statement as follows:

theorem sticks_left {totalSticks pickedSticks : ℕ} (H_original : totalSticks = 99) (H_picked : pickedSticks = 38) : totalSticks - pickedSticks = 61 := by
  rw [H_original, H_picked]
  rfl

end sticks_left_sticks_left_l324_324187


namespace ratio_of_areas_l324_324874

theorem ratio_of_areas (y : ℝ) (hy : y > 0) : 
  let area_A := y^2,
      area_B := (3 * y)^2
  in area_A / area_B = 1 / 9 :=
by 
  let area_A := y^2,
      area_B := (3 * y)^2
  show area_A / area_B = 1 / 9
  sorry

end ratio_of_areas_l324_324874


namespace radio_cost_price_l324_324885

theorem radio_cost_price (SP : ℝ) (Loss : ℝ) (CP : ℝ) (h1 : SP = 1110) (h2 : Loss = 0.26) (h3 : SP = CP * (1 - Loss)) : CP = 1500 :=
  by
  sorry

end radio_cost_price_l324_324885


namespace directrix_of_parabola_l324_324708

theorem directrix_of_parabola : 
  ∀ y : ℝ, let x := - (1 / 4) * y^2 in
  x = 1 :=
  sorry

end directrix_of_parabola_l324_324708


namespace sum_of_distances_to_sides_eq_height_l324_324862

noncomputable def equilateral_triangle_height (s : ℝ) : ℝ :=
  (real.sqrt 3 / 2) * s

theorem sum_of_distances_to_sides_eq_height
  (A B C P : ℝ × ℝ)
  (h : ℝ)
  (equilateral : dist A B = dist A C ∧ dist B C = dist A B)
  (height_def : h = equilateral_triangle_height (dist A B))
  (d_a d_b d_c : ℝ)
  (d_a_def : ∃ H_a : ℝ × ℝ, collinear {B, C, H_a} ∧ dist P H_a = d_a)
  (d_b_def : ∃ H_b : ℝ × ℝ, collinear {C, A, H_b} ∧ dist P H_b = d_b)
  (d_c_def : ∃ H_c : ℝ × ℝ, collinear {A, B, H_c} ∧ dist P H_c = d_c) :
  d_a + d_b + d_c = h := sorry

end sum_of_distances_to_sides_eq_height_l324_324862


namespace coeff_x4_expansion_l324_324384

theorem coeff_x4_expansion : 
  let p := (1 + x - x^2) * (1 + x^2) ^ 10 in
  polynomial.coeff p 4 = 35 :=
by
  sorry

end coeff_x4_expansion_l324_324384


namespace total_bulbs_is_118_l324_324105

-- Define the number of medium lights
def medium_lights : Nat := 12

-- Define the number of large and small lights based on the given conditions
def large_lights : Nat := 2 * medium_lights
def small_lights : Nat := medium_lights + 10

-- Define the number of bulbs required for each type of light
def bulbs_needed_for_medium : Nat := 2 * medium_lights
def bulbs_needed_for_large : Nat := 3 * large_lights
def bulbs_needed_for_small : Nat := 1 * small_lights

-- Define the total number of bulbs needed
def total_bulbs_needed : Nat := bulbs_needed_for_medium + bulbs_needed_for_large + bulbs_needed_for_small

-- The theorem that represents the proof problem
theorem total_bulbs_is_118 : total_bulbs_needed = 118 := by 
  sorry

end total_bulbs_is_118_l324_324105


namespace latest_race_time_l324_324799

def times : List ℝ := [12.5, 11.9, 12.2, 11.7, 12.0]
def new_median : ℝ := 11.95

theorem latest_race_time : ∃ x : ℝ, 
  (let sorted_times := times.insertion_sort (≤)
   let new_times := sorted_times.insert x
   let median := (new_times[(new_times.length / 2) - 1] + new_times[new_times.length / 2]) / 2
   median = new_median ∧ x = 11.9) :=
by 
  sorry

end latest_race_time_l324_324799


namespace find_m_l324_324747

-- Define the focus of the parabola
def focus_parabola := (0, 1/2)

-- Define the focus of the ellipse
def focus_ellipse (m : ℝ) : ℝ × ℝ := (0, Real.sqrt (m - 2))

-- Define the problem statement
theorem find_m (m : ℝ) (h1 : focus_parabola.snd = focus_ellipse m).snd :
  m = 9/4 :=
by
  sorry

end find_m_l324_324747


namespace find_friendly_pairs_l324_324022

def is_friendly_pair (R S : ℕ × ℕ) : Prop :=
  let (a, b) := R
  let (c, d) := S
  2 * (a + b) = c * d ∧ 2 * (c + d) = a * b

def all_friendly_pairs : Finset (ℕ × ℕ × ℕ × ℕ) :=
  {[⟨22, 5, 54, 1⟩, ⟨13, 6, 38, 1⟩, ⟨10, 7, 34, 1⟩, ⟨10, 3, 13, 2⟩, ⟨6, 4, 10, 2⟩, ⟨6, 3, 6, 3⟩, ⟨4, 4, 4, 4⟩]}

theorem find_friendly_pairs :
  {⟨a, b, c, d⟩ | (a, b) ∈ ℕ × ℕ ∧ (c, d) ∈ ℕ × ℕ ∧ is_friendly_pair (a, b) (c, d)} = all_friendly_pairs :=
sorry

end find_friendly_pairs_l324_324022


namespace find_n_l324_324257

def Sn (n : ℕ) : Set ℕ := { x | x < n }

def valid_bijection (n : ℕ) (f : ℕ → ℕ) : Prop :=
  Function.Bijective f ∧ ∀ a b c, a < n ∧ b < n ∧ c < n →
    n ∣ (a + b - c) → n ∣ (f a + f b - f c)

def set_of_valid_bijections (n : ℕ) : Set (ℕ → ℕ) :=
  { f | valid_bijection n f }

theorem find_n (n : ℕ) (hn : Sn n) (T : Set (Sn n → Sn n))
  : |set_of_valid_bijections n| = 60 →
    n = 61 ∨ n = 77 ∨ n = 93 ∨ n = 99 ∨ n = 122 ∨ n = 124 ∨ n = 154 ∨ n = 186 ∨ n = 198 :=
sorry

end find_n_l324_324257


namespace motorist_gas_problem_l324_324220

noncomputable def original_price_per_gallon (P : ℝ) : Prop :=
  12 * P = 10 * (P + 0.30)

def fuel_efficiency := 25

def new_distance_travelled (P : ℝ) : ℝ :=
  10 * fuel_efficiency

theorem motorist_gas_problem :
  ∃ P : ℝ, original_price_per_gallon P ∧ P = 1.5 ∧ new_distance_travelled P = 250 :=
by
  use 1.5
  sorry

end motorist_gas_problem_l324_324220


namespace find_f_3_l324_324506

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l324_324506


namespace part_a_part_b_l324_324952

open Set

variables {X : Type*} {n k : ℕ}

-- Part (a)
def condition1 (𝓕 : Finset (Finset X)) : Prop :=
  ∀ {A B C : Finset X}, A ∈ 𝓕 → B ∈ 𝓕 → C ∈ 𝓕 → A ≠ B → B ≠ C → A ≠ C → ¬ (A ∩ B ⊆ C)

theorem part_a (𝓕 : Finset (Finset X)) (h𝓕 : condition1 𝓕) (hX : Fintype.card X = n) :
  𝓕.card ≤ Nat.choose k (k / 2) + 1 :=
sorry

-- Part (b)
def condition2 (𝓕 : Finset (Finset X)) : Prop :=
  ∀ {A B C : Finset X}, A ∈ 𝓕 → B ∈ 𝓕 → C ∈ 𝓕 → A ≠ B → B ≠ C → A ≠ C → ¬ (A ∩ B ⊆ C)

theorem part_b (𝓕 : Finset (Finset X)) (h𝓕 : condition2 𝓕) (hX : Fintype.card X = n) :
  𝓕.card ≤ Nat.choose n ((n - 2) / 3).ceil + 2 :=
sorry

end part_a_part_b_l324_324952


namespace Lena_stops_in_X_l324_324440

def circumference : ℕ := 60
def distance_run : ℕ := 7920
def starting_point : String := "T"
def quarter_stops : String := "X"

theorem Lena_stops_in_X :
  (distance_run / circumference) * circumference + (distance_run % circumference) = distance_run →
  distance_run % circumference = 0 →
  (distance_run % circumference = 0 → starting_point = quarter_stops) →
  quarter_stops = "X" :=
sorry

end Lena_stops_in_X_l324_324440


namespace radius_of_circles_l324_324164

noncomputable section

def enclosed_area : ℝ := 1.975367389481267
def reuleaux_coefficient : ℝ := (3 * Real.sqrt 3 / 2) - (Real.pi / 2)

theorem radius_of_circles :
  sqrt (enclosed_area / reuleaux_coefficient) ≈ 1.3867 := sorry

end radius_of_circles_l324_324164


namespace number_one_seventh_equals_five_l324_324361

theorem number_one_seventh_equals_five (n : ℕ) (h : n / 7 = 5) : n = 35 :=
sorry

end number_one_seventh_equals_five_l324_324361


namespace preferred_sum_2026_l324_324335

def is_preferred (n : ℕ) : Prop := 
  ∃ k : ℕ, k ≥ 2 ∧ n + 2 = 2^k

def preferred_sum (upper_bound : ℕ) : ℕ := 
  (∑ n in finset.range upper_bound, if is_preferred n then n else 0)

theorem preferred_sum_2026 : preferred_sum 2012 = 2026 := 
by sorry

end preferred_sum_2026_l324_324335


namespace optimal_winding_times_to_maximize_overtakes_top_deeper_than_bottom_l324_324657

variables {t : ℝ} -- time in hours
variables {N1 N2 : ℝ → ℝ} -- time functions for the positions of the weights

-- Conditions from the problem
def condition1 (h : ℝ) : Prop := (h = 2)
def condition2 (N1 : ℝ → ℝ) : Prop := ∀ t, N1(t) = t * (2/24)
def condition3 (N2 : ℝ → ℝ) : Prop := ∀ t, f := (λ t:ℝ, if floor t = t then {h = t; N2(h) = h * (2/24)} else N2 (t) = (N2(floor t)) )

-- Question 1: Optimal winding times 
def optimal_times := [(3 + 50/60), (3 + 57/60), (4 + 22/60), (4 + 37/60), (4 + 10/60), (5 + 17/60), (6 + 6/60), (6 + 13/60), (7 + 10/60), (7 + 17/60), (8 + 22/60), (8 + 37/60), (9 + 50/60), (9 + 57/60)]

-- Statement for optimal winding times and catch-ups
theorem optimal_winding_times_to_maximize_overtakes (h : ℝ) (N1 N2 : ℝ → ℝ) (t : ℝ) :
  condition1 h ∧ condition2 N1 ∧ condition3 N2 →
  ∃ t ∈ optimal_times, N1(t) = N2(t) :=
by { sorry }

-- Question 2: Top of one cylinder deeper than the bottom of the other
theorem top_deeper_than_bottom (h : ℝ) (N1 N2 : ℝ → ℝ) (t1 t2 : ℝ) :
  condition1 h ∧ condition2 N1 ∧ condition3 N2 →
  ∃ t1 t2, t1 ≠ t2 ∧ N1(t1) < N2(t2) :=
by { sorry }

end optimal_winding_times_to_maximize_overtakes_top_deeper_than_bottom_l324_324657


namespace diagonal_products_eq_parallel_l324_324469

section Quadrilateral

variables (A B C D O : Point)

-- Conditions: quadrilateral ABCD with diagonals AC and BD intersecting at O
variables (quad : between A C O ∧ between B D O)

theorem diagonal_products_eq_parallel (O : Point) : 
  (segment_length A O * segment_length B O = segment_length C O * segment_length D O) ↔
  parallel (line B C) (line A D) :=
sorry

end Quadrilateral

end diagonal_products_eq_parallel_l324_324469


namespace scores_increase_properties_l324_324580

noncomputable def original_scores := [42, 47, 53, 53, 58, 58, 58, 61, 64, 65, 73]
noncomputable def new_score := 80

def mean (scores : List ℕ) : ℕ :=
  (scores.sum) / scores.length

def range (scores : List ℕ) : ℕ :=
  scores.maximum - scores.minimum

def std_dev (scores : List ℕ) : ℝ :=
  let μ := (scores.sum : ℝ) / scores.length
  let variance := (scores.map (λ x => ((x : ℝ) - μ)^2)).sum / scores.length
  Real.sqrt variance

theorem scores_increase_properties :
  let original_mean := mean original_scores
  let new_mean := mean (original_scores ++ [new_score])
  let original_range := range original_scores
  let new_range := range (original_scores ++ [new_score])
  let original_std_dev := std_dev original_scores
  let new_std_dev := std_dev (original_scores ++ [new_score])
  new_mean > original_mean ∧ new_range > original_range ∧ new_std_dev > original_std_dev :=
by
  sorry

end scores_increase_properties_l324_324580


namespace best_approximation_of_x_squared_l324_324809

theorem best_approximation_of_x_squared
  (x : ℝ) (A B C D E : ℝ)
  (h1 : -2 < -1)
  (h2 : -1 < 0)
  (h3 : 0 < 1)
  (h4 : 1 < 2)
  (hx : -1 < x ∧ x < 0)
  (hC : 0 < C ∧ C < 1) :
  x^2 = C :=
sorry

end best_approximation_of_x_squared_l324_324809


namespace hyperbola_eccentricity_range_l324_324332

theorem hyperbola_eccentricity_range
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c^2 = a^2 + b^2)
  (h4 : ∃ x y : ℝ, (x - √3)^2 + y^2 = 1 ∧ b * x - a * y = 0) :
  1 < (c / a) ∧ (c / a) ≤ (√6 / 2) :=
by
  sorry

end hyperbola_eccentricity_range_l324_324332


namespace proof_correct_answer_l324_324294

variables (x y : ℝ)

def p : Prop := x > abs y → x > y
def q : Prop := x + y > 0 → x^2 > y^2

def proposition1 : Prop := p ∨ q
def proposition2 : Prop := ¬p ∧ ¬q
def proposition3 : Prop := p ∧ ¬q
def proposition4 : Prop := p ∧ q

def number_of_true_propositions : ℕ :=
  [proposition1, proposition2, proposition3, proposition4].count (λ prop, prop)

theorem proof_correct_answer : number_of_true_propositions x y = 2 := by
  sorry

end proof_correct_answer_l324_324294


namespace even_function_a_zero_max_value_at_minus_one_three_zero_points_l324_324328

-- Problem 1: Prove that if f(x) is an even function, then a = 0.
theorem even_function_a_zero (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x : ℝ, f x = -x^2 + 2 * real.abs (x - a)) 
  (hf_even : ∀ x : ℝ, f (-x) = f x) : a = 0 :=
sorry

-- Problem 2: Prove that if f(x) reaches its maximum value at x = -1, the range of a is [0, ∞).
theorem max_value_at_minus_one (f : ℝ → ℝ) (a : ℝ)
  (h : ∀ x : ℝ, f x = -x^2 + 2 * real.abs (x - a))
  (hf_max : ∀ y : ℝ, f (-1) ≥ f y) : 0 ≤ a :=
sorry

-- Problem 3: Prove that if f(x) has three zero points, a ∈ {-1/2, 0, 1/2}.
theorem three_zero_points (f : ℝ → ℝ) (a : ℝ)
  (h : ∀ x : ℝ, f x = -x^2 + 2 * real.abs (x - a))
  (hf_zeros : ∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  a = -1/2 ∨ a = 0 ∨ a = 1/2 :=
sorry

end even_function_a_zero_max_value_at_minus_one_three_zero_points_l324_324328


namespace center_of_circle_l324_324278

theorem center_of_circle (x y : ℝ) 
  (h : ∀ x y : ℝ, x^2 + y^2 = 12 * x - 4 * y + 10 → (x, y) = (6, -2)) : 
  x + y = 4 :=
by
  have h_center : (6, -2) = (6, -2), from rfl
  have h_sum : 6 + (-2) = 4, by ring
  exact h_sum

end center_of_circle_l324_324278


namespace pizza_topping_count_l324_324625

theorem pizza_topping_count (n : ℕ) (h : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by {
  rw h,
  simp [nat.choose],
  sorry,
}

end pizza_topping_count_l324_324625


namespace mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l324_324290

def card_is_heart (c : ℕ) := c ≥ 1 ∧ c ≤ 13

def card_is_diamond (c : ℕ) := c ≥ 14 ∧ c ≤ 26

def card_is_red (c : ℕ) := c ≥ 1 ∧ c ≤ 26

def card_is_black (c : ℕ) := c ≥ 27 ∧ c ≤ 52

def card_is_face_234610 (c : ℕ) := c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6 ∨ c = 10

def card_is_face_2345678910 (c : ℕ) :=
  c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10

def card_is_face_AKQJ (c : ℕ) :=
  c = 1 ∨ c = 11 ∨ c = 12 ∨ c = 13

def card_is_ace_king_queen_jack (c : ℕ) := c = 1 ∨ (c ≥ 11 ∧ c ≤ 13)

theorem mutually_exclusive_pair2 : ∀ c : ℕ, card_is_red c ≠ card_is_black c := by
  sorry

theorem complementary_pair2 : ∀ c : ℕ, card_is_red c ∨ card_is_black c := by
  sorry

theorem mutually_exclusive_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ≠ card_is_ace_king_queen_jack c := by
  sorry

theorem complementary_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ∨ card_is_ace_king_queen_jack c := by
  sorry

end mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l324_324290


namespace height_of_water_in_cylinder_l324_324665

-- Define the parameters of the cone
def r_cone : ℝ := 12
def h_cone : ℝ := 18

-- Define the volume formula for a cone
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

-- Define the parameters of the cylinder
def r_cylinder : ℝ := 24

-- Define the volume formula for a cylinder
def volume_cylinder (r h : ℝ) : ℝ := real.pi * r^2 * h

-- Noncomptuable context for the proof
noncomputable def h_cylinder : ℝ :=
  let V_cone := volume_cone r_cone h_cone
  in (V_cone / (real.pi * r_cylinder^2))

-- Theorem to prove the height of water in the cylinder
theorem height_of_water_in_cylinder : h_cylinder = 1.5 :=
  by
  -- Placeholder for the actual proof steps
  sorry

end height_of_water_in_cylinder_l324_324665


namespace find_f_3_l324_324508

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l324_324508


namespace dan_money_left_l324_324255

theorem dan_money_left
  (initial_amount : ℝ := 45)
  (cost_per_candy_bar : ℝ := 4)
  (num_candy_bars : ℕ := 4)
  (price_toy_car : ℝ := 15)
  (discount_rate_toy_car : ℝ := 0.10)
  (sales_tax_rate : ℝ := 0.05) :
  initial_amount - ((num_candy_bars * cost_per_candy_bar) + ((price_toy_car - (price_toy_car * discount_rate_toy_car)) * (1 + sales_tax_rate))) = 14.02 :=
by
  sorry

end dan_money_left_l324_324255


namespace point_in_fourth_quadrant_l324_324378

-- Define the point P and the quadrants
structure Point :=
  (x : ℝ)
  (y : ℝ)

inductive Quadrant
| first
| second
| third
| fourth

-- Define the conditions and the conclusion proof problem
theorem point_in_fourth_quadrant (P : Point) (hx : P.x = 1) (hy : P.y = -2) : Quadrant :=
by
  sorry

def point_P := { x := 1, y := -2 }

example : point_in_fourth_quadrant point_P rfl rfl = Quadrant.fourth :=
by
  sorry

end point_in_fourth_quadrant_l324_324378


namespace solution_set_of_inequality_l324_324145

noncomputable def f : ℝ → ℝ := sorry
def f_prime (x : ℝ) : ℝ := (deriv f) x

theorem solution_set_of_inequality :
  (f 1 = 2) →
  (∀ x, f_prime x - 3 > 0) →
  { x : ℝ | f (log 3 x) < 3 * log 3 x - 1 } = {x : ℝ | 0 < x ∧ x < 3} :=
begin
  sorry
end

end solution_set_of_inequality_l324_324145


namespace only_prime_in_list_is_47_l324_324017

def is_concatenation_of_4747 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 47 * (4747 ^ k)

theorem only_prime_in_list_is_47 : 
  (∃ n : ℕ, is_prime n ∧ is_concatenation_of_4747 n) → n = 47 := 
sorry

end only_prime_in_list_is_47_l324_324017


namespace angle_ABC_in_shared_square_and_octagon_l324_324650

/-- Definitions based on the given conditions. --/
structure Square where
  A B C D : Type
  has_common_side : Prop
  
structure RegularOctagon where
  A B C D E F G H : Type
  interior_angle : ℕ → ℝ
  
def angle_measure (n : ℕ) : ℕ := 
  (n - 2) * 180 / n

/-- The Proof Statement corresponding to the problem --/
theorem angle_ABC_in_shared_square_and_octagon 
  (S : Square) 
  (O : RegularOctagon) 
  (A B C : O) 
  (D E : S) 
  (h_common_side : S.has_common_side) 
  (h_interior_angle : O.interior_angle 8 = 135) 
  : ∠ABC = 67.5 :=
by
  sorry

end angle_ABC_in_shared_square_and_octagon_l324_324650


namespace least_common_multiple_135_195_l324_324935

def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem least_common_multiple_135_195 : leastCommonMultiple 135 195 = 1755 := by
  sorry

end least_common_multiple_135_195_l324_324935


namespace tangent_slope_l324_324835

open Topology

variable {α : Type*} [TopologicalSpace α] {β : Type*} [LinearOrderedField β]

theorem tangent_slope (f : β → β) (h_diff : Differentiable β f)
  (h_limit : Tendsto (λ x, (f 1 - f (1 - x)) / (2 * x)) (𝓝 0) (𝓝 (-1))) :
  deriv f 1 = -2 :=
by sorry

end tangent_slope_l324_324835


namespace pizza_toppings_count_l324_324621

theorem pizza_toppings_count :
  let toppings := 8 in
  let one_topping_pizzas := toppings in
  let two_topping_pizzas := (toppings.choose 2) in
  let three_topping_pizzas := (toppings.choose 3) in
  one_topping_pizzas + two_topping_pizzas + three_topping_pizzas = 92 :=
by sorry

end pizza_toppings_count_l324_324621


namespace count_b_of_quadratic_inequality_l324_324260

theorem count_b_of_quadratic_inequality : 
  ∃ (b_vals : Set ℤ), (∀ b ∈ b_vals, ∃! (s : Set ℤ), (s.card = 3 ∧ ∀ x ∈ s, x^2 + b * x + 6 ≤ 0)) ∧ b_vals.card = 2 :=
by
  sorry

end count_b_of_quadratic_inequality_l324_324260


namespace part_a_part_b_l324_324815

-- Define the conditions
variables {ABC : Type*} [Metric_Space ABC]
variables {A B C O I L : ABC}
variables {A' : ABC} (symm_of_A: A' = symmetric_point A O)
variables {R r : ℝ}  -- Radii of circumcircle and incircle
variables (circumcircle: metric.sphere O R) (incircle: metric.sphere I r)
variables (A_tangent_to_incircle : tangent A incircle)
variables (A'_tangent_to_incircle : tangent A' incircle)

-- Define the theorem
theorem part_a : 
  power_of_point A incircle + power_of_point A' incircle = 
  4 * R^2 - 4 * R * r - 2 * r^2 :=
sorry

-- Additional conditions for part (b)
variables (intersection_of_circles : intersect_circle_in_point circumcircle (metric.sphere A' (dist A' I)) L)
variables (A_I_square : (dist A I) ^ 2 + (dist A' I) ^ 2 = 4 * R^2 - 4 * R * r)
variables (A_A' : dist A A' = 2 * R)

-- Define the theorem
theorem part_b :
  dist A L = real.sqrt (dist A B * dist A C) :=
sorry

end part_a_part_b_l324_324815


namespace num_distinct_real_solutions_l324_324085

def f (x : ℝ) := x^3 - 3*x + 1

theorem num_distinct_real_solutions :
  {c : ℝ | f (f (f (f c))) = 2}.to_finset.card = 81 :=
sorry

end num_distinct_real_solutions_l324_324085


namespace angle_PBA_25_l324_324386

theorem angle_PBA_25 (A B C D P : Type) [square A B C D] [point_inside_square P A B C D] 
  (h1 : angle D C P = 25) (h2 : angle C A P = 25) : angle P B A = 25 := 
sorry

end angle_PBA_25_l324_324386


namespace find_f_of_3_l324_324513

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l324_324513


namespace intersection_of_M_and_N_l324_324761

-- Definitions of the sets
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- The theorem to prove
theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_of_M_and_N_l324_324761


namespace angle_C_measure_phi_not_exist_l324_324394

section triangle_problem

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
def sides_opposite_angles (a b c A B C : ℝ) : Prop :=
  ∃ (θA θB θC : ℝ), 
      (θA = θA.mod (2 * Real.pi)) ∧ (θB = θB.mod (2 * Real.pi)) ∧ (θC = θC.mod (2 * Real.pi)) ∧ 
      (a = is length of sides opposite to θA) ∧
      (b = is length of sides opposite to θB) ∧
      (c = is length of sides opposite to θC) 

def sine_squared_eq (A B C : ℝ) : Prop :=
  Real.sin(A)^2 + Real.sin(B)^2 + 4 * Real.sin(A) * Real.sin(B) * Real.cos(C) = 0

def c_squared_eq (a b c : ℝ) : Prop :=
  c^2 = 3 * a * b

-- Question 1 answer
theorem angle_C_measure (h1: sides_opposite_angles a b c A B C) (h2: sine_squared_eq A B C) (h3: c_squared_eq a b c) :
  C = 2 * Real.pi / 3 := by 
  sorry

-- Variables for question 2
variable {x : ℝ}
def f_question2 (ω ϕ : ℝ) (x : ℝ) : ℝ := Real.sin(ω * x + ϕ)

-- Conditions Question 2
def f_monotonic (ω ϕ : ℝ) : Prop :=
  ∃ (C : ℝ), (2 * Real.pi / 3 = C ∧ f_question2 ω ϕ C = - 1 / 2) ∧ (∀ (x1 x2 ∈ Ioo (Real.pi / 7) (Real.pi / 2)), f_question2 ω ϕ x1 ≤ f_question2 ω ϕ x2 ∨ f_question2 ω ϕ x1 ≥ f_question2 ω ϕ x2 )

-- Question 2 answer
theorem phi_not_exist (h1: ∃ ω ∈ {1, 2}, ω > 0) (h2: ∀ ω, f_monotonic ω ϕ):
  ¬(∃ (ϕ : ℝ), ϕ.abs < Real.pi / 2) := by 
  sorry

end triangle_problem

end angle_C_measure_phi_not_exist_l324_324394


namespace problem_extraneous_root_l324_324349

theorem problem_extraneous_root (m : ℤ) :
  (∃ x, x = -4 ∧ (x + 4 = 0) ∧ ((x-1)/(x+4) = m/(x+4)) ∧ (m = -5)) :=
sorry

end problem_extraneous_root_l324_324349


namespace rearrange_princeton_correct_l324_324819

-- Define the set of vowels and consonants
def vowels : Set Char := {'I', 'E', 'O'}
def consonants : Set Char := {'P', 'R', 'N', 'C', 'T', 'N'}

-- Define the string "PRINCETON"
def princeton : List Char := ['P', 'R', 'I', 'N', 'C', 'E', 'T', 'O', 'N']

--Total number of ways to rearrange the letters
noncomputable def rearrange_princeton_count : Nat :=
  let total_ways_case1 := 2 * 3! * 6! / 2!
  let total_ways_case2 := 6 * 3! * 6! / 2!
  total_ways_case1 + total_ways_case2
  
theorem rearrange_princeton_correct :
  rearrange_princeton_count = 17280 := sorry

end rearrange_princeton_correct_l324_324819


namespace probability_A_and_B_l324_324151

variable (P : Set α → ℝ)

-- Conditions
def A := {a : α | event_a a}
def B := {a : α | event_b a}

axiom P_A : P A = 0.4
axiom P_A_union_B : P (A ∪ B) = 0.6
axiom P_B : P B = 0.45

-- Proof Problem
theorem probability_A_and_B : P (A ∩ B) = 0.25 := by
  sorry

end probability_A_and_B_l324_324151


namespace area_enclosed_by_cosine_curve_l324_324127

-- Define the function cos
def f (x : ℝ) : ℝ := cos x

-- Define the bounds for the x values
def x₁ : ℝ := π / 2
def x₂ : ℝ := 3 * π / 2

-- State the theorem we want to prove
theorem area_enclosed_by_cosine_curve :
  ∫ x in x₁..x₂, -f x = 2 := by
  sorry

end area_enclosed_by_cosine_curve_l324_324127


namespace measure_of_angle_C_phi_no_solution_l324_324391

open Real

-- Definition and condition for finding angle C
def sin_squared_condition (A B C : ℝ) : Prop :=
  sin A ^ 2 + sin B ^ 2 + 4 * sin A * sin B * cos C = 0

def side_condition (a b c : ℝ) : Prop :=
  c ^ 2 = 3 * a * b

-- Theorem for finding angle C
theorem measure_of_angle_C (A B C a b c : ℝ)
  (h1 : sin_squared_condition A B C)
  (h2 : side_condition a b c) :
  cos C = -1 / 2 :=
sorry

-- Definition and conditions for the φ problem
def f (ω φ x : ℝ) : ℝ :=
  sin (ω * x + φ)

def monotonic_interval (ω φ : ℝ) : Prop :=
  ∀ x y, (π / 7 < x ∧ x < π / 2) → ((π / 7 < y ∧ y < π / 2) → x ≤ y → f ω φ x ≤ f ω φ y)

-- Theorem for the φ problem
theorem phi_no_solution (C φ : ℝ)
  (ω : ℕ)
  (h1 : |φ| < π / 2)
  (h2 : ω ≠ 0)
  (h3 : f ω φ C = -1 / 2)
  (h4 : monotonic_interval (ω.to_real) φ) :
  false :=
sorry

end measure_of_angle_C_phi_no_solution_l324_324391


namespace exactly_one_valid_N_l324_324010

def four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

def condition (N x a : ℕ) : Prop := 
  N = 1000 * a + x ∧ x = N / 7

theorem exactly_one_valid_N : 
  ∃! N : ℕ, ∃ x a : ℕ, four_digit_number N ∧ condition N x a :=
sorry

end exactly_one_valid_N_l324_324010


namespace hexagons_form_square_l324_324878

theorem hexagons_form_square (a b y : ℕ) (h_rect : a = 8 ∧ b = 18) 
    (h_hexagons : 2 * (a * b) = s * s) 
    (h_square : s = 12) : 
    y = s / 2 := 
by
  -- Rectangle area
  have h1 : a * b = 144 := by
    calc 8 * 18 = 144
  -- Square's side length squared
  have h2 : s^2 = 144 := by
    calc s^2 = 144 
  -- Translate from the problem's geometric constraints
  have h3 : s = 12 := by 
    exact h_square
  -- Conclude that y must be half the side of the square's side length
  have h4 : y = 12 / 2 := by
    calc y = 12 / 2
  -- Therefore y = 6
  exact h4

end hexagons_form_square_l324_324878


namespace vertical_angles_are_equal_l324_324531

theorem vertical_angles_are_equal (A B C D : Type) (angle1 : angle A B C) (angle2 : angle C D A) 
  (h : vertical_angles angle1 angle2) : angle1 = angle2 := 
sorry

end vertical_angles_are_equal_l324_324531


namespace thomas_total_blocks_l324_324920

theorem thomas_total_blocks :
  let stack1 := 7 in
  let stack2 := stack1 + 3 in
  let stack3 := stack2 - 6 in
  let stack4 := stack3 + 10 in
  let stack5 := stack2 * 2 in
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 :=
by
  let stack1 := 7
  let stack2 := stack1 + 3
  let stack3 := stack2 - 6
  let stack4 := stack3 + 10
  let stack5 := stack2 * 2
  have : stack1 + stack2 + stack3 + stack4 + stack5 = 7 + 10 + 4 + 14 + 20 := by rfl
  rw [this]
  norm_num
  sorry

end thomas_total_blocks_l324_324920


namespace find_f_3_l324_324507

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l324_324507


namespace separating_line_exists_l324_324206

def point : Type := ℝ × ℝ

noncomputable def pairwise_distance_sum (ps : List point) : ℝ :=
  ps.pairwise (λ p1 p2, (Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))).sum

def red_points (ps : List point) : Prop :=
  ps.length = 101 ∧ pairwise_distance_sum ps = 1

def blue_points (ps : List point) : Prop :=
  ps.length = 101 ∧ pairwise_distance_sum ps = 1

def mixed_pairs_distance_sum (red : List point) (blue : List point) : ℝ :=
  (List.product red blue).map (λ p, Real.sqrt ((p.1.1 - p.2.1)^2 + (p.1.2 - p.2.2)^2)).sum

theorem separating_line_exists
  (R B : List point)
  (hR : red_points R)
  (hB : blue_points B)
  (hMixed : mixed_pairs_distance_sum R B = 400) 
  : ∃ l : point → Prop, (∀ r ∈ R, l r) ∧ (∀ b ∈ B, ¬ l b) :=
sorry

end separating_line_exists_l324_324206


namespace sufficient_but_not_necessary_condition_l324_324814

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a < 1 / b) ∧ ¬ (1 / a < 1 / b → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l324_324814


namespace cyclic_points_l324_324405

open PlanarGeometry

theorem cyclic_points 
  (DFGE : CyclicQuadrilateral) 
  (C : Point) (H : Point) (J : Point) (I : Point)
  (intersect1 : ∃ A B, line_line_intersect DFGE.DF DFGE.EG = some (C, A, B))
  (intersect2 : ∃ A B, line_line_intersect DFGE.FE DFGE.DG = some (H, A, B))
  (midpointJ : J = Point.midpoint DFGE.FG)
  (reflection_ell : ∃ ell, reflection_line_elliptic DFGE.DE (line_through C H) = ell)
  (intersect_ell_Gf : ∃ A B, line_line_intersect (reflection_line_elliptic DFGE.DE (line_through C H)) DFGE.GF = some (I, A, B)) :
  concyclic {C, J, H, I} :=
sorry

end cyclic_points_l324_324405


namespace distance_to_plane_is_6cm_l324_324110

noncomputable def distance_from_center_to_plane (O A B C : EuclideanGeometry.Point)
  (radius : ℝ) (distance_AB : ℝ) (angle_ACB : ℝ) :
  ℝ :=
if h : radius = 10 ∧ distance_AB = 8 * Real.sqrt 3 ∧ angle_ACB = 60 then
  6
else
  0 -- default value in case the conditions do not hold

theorem distance_to_plane_is_6cm :
  let O := EuclideanGeometry.center, 
  A := EuclideanGeometry.Point.mk 8 (sqrt 3),
  B := EuclideanGeometry.Point.mk 8 (-sqrt 3),
  C := EuclideanGeometry.Point.mk 1 0 in
  distance_from_center_to_plane O A B C 10 (8 * (Real.sqrt 3)) 60 = 6 :=
begin
  -- Proof is omitted
  sorry
end

end distance_to_plane_is_6cm_l324_324110


namespace jane_doe_investment_l324_324823

theorem jane_doe_investment (total_investment mutual_funds real_estate : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total_investment = mutual_funds + real_estate) :
  real_estate = 187500 :=
by
  sorry

end jane_doe_investment_l324_324823


namespace equilibrium_properties_l324_324064

noncomputable def SO₂ : Type := ℝ
noncomputable def O₂  : Type := ℝ
noncomputable def SO₃ : Type := ℝ

variables (T₁ T₂ : ℝ) (H : ℝ) (V : ℝ) (SO₂_init O₂_init SO₃_eq SO₂_new O₂_new : ℝ)

-- Conditions
axiom reaction_eq : 2 * SO₂ + O₂ = 2 * SO₃
axiom delta_H : H = -190 -- kJ/mol
axiom temp_450C : T₁ = 450 -- C
axiom temp_500C : T₂ = 500 -- C
axiom init_amounts : SO₂_init = 0.20 ∧ O₂_init = 0.10 ∧ V = 5
axiom eq_conc_SO₃ : SO₃_eq = 0.18

-- Added reactants
axiom added_SO₂ : SO₂_new = 0.30
axiom added_O₂ : O₂_new = 0.15

-- Proof statements
theorem equilibrium_properties : 
  let K_450 := (SO₃_eq^2) / (SO₂_init^2 * O₂_init),
      K_500 := (SO₃_eq^2) / ((SO₂_init + SO₂_new)^2 * (O₂_init + O₂_new)) in
  (K_450 > K_500) ∧ 
  (0.036 = (0.09 / (5 * 0.5))) ∧ 
  (0.36 < SO₃_eq ∧ SO₃_eq < 0.50) ∧
  ∃ rate_eq : Prop, 
  rate_eq ⇔ 
  (velocity_ratio := some_ratio_expression) ∧ 
  (indicator_density_constant := density_constant) ∧ 
  (indicator_molecular_weight_constant := molecular_weight_constant) :=
sorry

end equilibrium_properties_l324_324064


namespace problem_not_true_equation_l324_324293

theorem problem_not_true_equation
  (a b : ℝ)
  (h : a / b = 2 / 3) : a / b ≠ (a + 2) / (b + 2) := 
sorry

end problem_not_true_equation_l324_324293


namespace domain_correct_l324_324691

noncomputable def domain_function (x : ℝ) : Prop :=
  (4 * x - 3 > 0) ∧ (Real.log (4 * x - 3) / Real.log 0.5 > 0)

theorem domain_correct : {x : ℝ | domain_function x} = {x : ℝ | (3 / 4 : ℝ) < x ∧ x < 1} :=
by
  sorry

end domain_correct_l324_324691


namespace B_completes_in_10_days_l324_324954

variables (A_efficiency : ℝ) (B_efficiency : ℝ)
variables (A_days : ℝ := 12) (B_days : ℝ)

-- Define that A completes the work in 12 days
def A_work_rate := 1 / A_days

-- Define that B is 20% more efficient than A
def B_work_rate := 1.2 * A_work_rate

-- Given that B's work rate is 1.2 times A's work rate, conclude the number of days B takes
theorem B_completes_in_10_days (h1 : A_efficiency = 1 / A_days) (h2 : B_efficiency = 1.2 * A_efficiency) :
  B_days = 10 :=
by 
  sorry

end B_completes_in_10_days_l324_324954


namespace find_f_of_3_l324_324491

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l324_324491


namespace total_students_in_school_l324_324049

theorem total_students_in_school (C1 C2 C3 C4 C5 : ℕ) 
  (h1 : C1 = 23)
  (h2 : C2 = C1 - 2)
  (h3 : C3 = C2 - 2)
  (h4 : C4 = C3 - 2)
  (h5 : C5 = C4 - 2)
  : C1 + C2 + C3 + C4 + C5 = 95 := 
by 
  -- proof details skipped with sorry
  sorry

end total_students_in_school_l324_324049


namespace number_of_ordered_quadruples_l324_324693

theorem number_of_ordered_quadruples :
  {a b c d : ℝ // 
     (exists (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ), 
         (∀ i₁ i₂, (i₁ = (2 * a * a) ∧ i₂ = ( 0 * c )) :=
         (i₁ = 1 ∧ i₂ = 0)) 
         ∧ (i₁ = (2 * a * b) ∧ i₂ = ( 0 * d )) :=
         (i₁ = 0 ∧ i₂ = 1))
} 
= 4 := by sorry

end number_of_ordered_quadruples_l324_324693


namespace measure_8_liters_possible_l324_324003

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l324_324003


namespace bulbs_needed_l324_324107

theorem bulbs_needed (M : ℕ) (hM : M = 12) : 
  let large := 2 * M in
  let small := M + 10 in
  let bulbs_medium := 2 * M in
  let bulbs_large := 3 * large in
  let bulbs_small := small in
  bulbs_medium + bulbs_large + bulbs_small = 118 :=
by
  sorry

end bulbs_needed_l324_324107


namespace find_special_integer_l324_324833

-- Definition of number of positive divisors
def d(n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, n % d = 0).card

-- Main statement
theorem find_special_integer (n : ℕ) (h1 : 4 ≤ d(n)) :
  (∃ (a : Fin n) (h : (∀ i j : Fin (d(n) - 1), i < j → gcd (a i) n ≠ gcd (a j) n)), true) ↔ 
  (n = 8 ∨ n = 12 ∨ (∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ n = p * q)) :=
sorry

end find_special_integer_l324_324833


namespace distinct_solutions_eq_2_l324_324528

theorem distinct_solutions_eq_2 : 
  ∃ S : set ℝ, (∀ x ∈ S, abs (x - abs (2 * x + 1)) = 3) ∧ S.card = 2 :=
begin
  sorry
end

end distinct_solutions_eq_2_l324_324528


namespace oldest_sibling_age_difference_l324_324158

theorem oldest_sibling_age_difference 
  (D : ℝ) 
  (avg_age : ℝ) 
  (hD : D = 25.75) 
  (h_avg : avg_age = 30) :
  ∃ A : ℝ, (A - D ≥ 17) :=
by
  sorry

end oldest_sibling_age_difference_l324_324158


namespace complex_modulus_l324_324323

noncomputable def z : ℂ := Complex.inv (1 + Complex.i) * 2 * Complex.i

theorem complex_modulus (z : ℂ) (hz : z * (1 + Complex.i) = 2 * Complex.i) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end complex_modulus_l324_324323


namespace number_of_true_propositions_is_2_l324_324420

-- Definitions of lines and planes
variable {Line Plane : Type}

-- Definitions of parallel and perpendicular relationships
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (lineInPlane : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- Assume conditions as hypotheses
variables (m n : Line)
variables (α β : Plane)

def proposition1 := planeParallel α β → lineInPlane m β → lineInPlane n α → parallel m n
def proposition2 := planeParallel α β → perpendicular m β → parallel n α → perpendicular m n
def proposition3 := planePerpendicular α β → perpendicular m α → parallel n β → parallel m n
def proposition4 := planePerpendicular α β → perpendicular m α → perpendicular n β → perpendicular m n

-- Statement of the problem in Lean, asserting the number of true propositions is 2
theorem number_of_true_propositions_is_2
  (h1 : ¬ proposition1)
  (h2 : proposition2)
  (h3 : ¬ proposition3)
  (h4 : proposition4) :
  (if proposition1 then 1 else 0) + (if proposition2 then 1 else 0) + 
  (if proposition3 then 1 else 0) + (if proposition4 then 1 else 0) = 2 :=
by
  sorry

end number_of_true_propositions_is_2_l324_324420


namespace sqrt_9_eq_pm3_l324_324939

theorem sqrt_9_eq_pm3 : ∃ x : ℤ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end sqrt_9_eq_pm3_l324_324939


namespace smallest_4_digit_divisible_by_44_l324_324176

theorem smallest_4_digit_divisible_by_44 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 44 = 0 ∧ n = 1023 :=
by
  -- Problem conditions
  existsi 1023
  -- Prove that 1023 is within the four-digit number range
  simp
  -- Check if the number is divisible by 44
  have h44 : 1023 % 44 = 0 := sorry
  -- Check if this is the smallest such number
  have hmin : ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 44 = 0 → 1023 ≤ m := sorry
  -- Combining all together
  exact ⟨h44, hmin⟩

end smallest_4_digit_divisible_by_44_l324_324176


namespace greg_total_earnings_correct_l324_324769

def charge_per_dog := 20
def charge_per_minute := 1

def earnings_one_dog := charge_per_dog + charge_per_minute * 10
def earnings_two_dogs := 2 * (charge_per_dog + charge_per_minute * 7)
def earnings_three_dogs := 3 * (charge_per_dog + charge_per_minute * 9)

def total_earnings := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

theorem greg_total_earnings_correct : total_earnings = 171 := by
  sorry

end greg_total_earnings_correct_l324_324769


namespace function_monotonic_range_k_l324_324350

theorem function_monotonic_range_k (k : ℝ) :
  (∀ x y ∈ (Ici 1 : Set ℝ), x < y → 4 * x^2 - k * x - 8 <= 4 * y^2 - k * y - 8 ∨
                            4 * x^2 - k * x - 8 >= 4 * y^2 - k * y - 8) ↔ k ≤ 8 :=
by
  sorry

end function_monotonic_range_k_l324_324350


namespace value_of_q_g_l324_324743

variable {Ω : Type}
variable [ProbabilitySpace Ω]
variable {g h : Event Ω}

-- Given conditions
def q_h : ℝ := 0.9
def q_g_and_h : ℝ := 0.3
def q_g_given_h : ℝ := 0.3333333333333333
def q_h_given_g : ℝ := 0.3333333333333333

-- The theorem to prove
theorem value_of_q_g : Prob g = 0.9 :=
by
  -- Corresponding Lean formal statement
  sorry

end value_of_q_g_l324_324743


namespace min_clicks_to_uniform_color_l324_324601

noncomputable def chessboard := Fin 98 × Fin 98 → Prop

-- Chessboard is assumed alternating, let color be false for white and true for black
def initial_chessboard : chessboard :=
  λ ⟨i, _⟩ ⟨j, _⟩, (i + j) % 2 = 0

-- A click on a rectangle on the chessboard
def click (b : chessboard) (x1 x2 y1 y2 : Fin 98) : chessboard :=
  λ (i : Fin 98) (j : Fin 98), if x1 ≤ i ∧ i ≤ x2 ∧ y1 ≤ j ∧ j ≤ y2 then ¬b i j else b i j

-- Minimum number of clicks to make the board all one color
theorem min_clicks_to_uniform_color : 
  ∃ n, n = 98 ∧ ∀ b : chessboard, if ∃ i j, b i j ≠ initial_chessboard i j 
    then ∃ (mouse_clicks : list (Fin 98 × Fin 98 × Fin 98 × Fin 98)), mouse_clicks.length = n ∧
      (mouse_clicks.foldl (λ (acc : chessboard) (rect : Fin 98 × Fin 98 × Fin 98 × Fin 98), click acc rect.1 rect.2 rect.3 rect.4) b) = λ _ _, true ∨ λ _ _, false
  else false :=
sorry

end min_clicks_to_uniform_color_l324_324601


namespace marbles_each_friend_is_16_l324_324772

-- Define the initial condition
def initial_marbles : ℕ := 100

-- Define the marbles Harold kept for himself
def kept_marbles : ℕ := 20

-- Define the number of friends Harold shared the marbles with
def num_friends : ℕ := 5

-- Define the marbles each friend receives
def marbles_per_friend (initial kept : ℕ) (friends : ℕ) : ℕ :=
  (initial - kept) / friends

-- Prove that each friend gets 16 marbles
theorem marbles_each_friend_is_16 : marbles_per_friend initial_marbles kept_marbles num_friends = 16 :=
by
  unfold initial_marbles kept_marbles num_friends marbles_per_friend
  exact Nat.mk_eq Nat.zero 16 sorry

end marbles_each_friend_is_16_l324_324772


namespace area_AEF_l324_324667

-- Given conditions about the areas of triangles within rectangle ABCD
def area_CEF := 3
def area_ABE := 4
def area_ADF := 5

-- We need to prove the area of triangle AEF
theorem area_AEF : area_AEF = 8 :=
by
  sorry

end area_AEF_l324_324667


namespace find_f_three_l324_324485

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l324_324485


namespace plus_recurring_decimal_l324_324246

noncomputable def recurring_decimal (n : ℕ) (d : ℕ) : ℚ :=
  let frac := n / d
  frac

theorem plus_recurring_decimal : 2 + recurring_decimal 1 3 = 7 / 3 :=
by
  have b := recurring_decimal 1 3
  have b_eq : b = 1 / 3 := by sorry
  rw [b_eq]
  have eq1 : 2 + 1 / 3 = (6 + 1) / 3 := by sorry
  have eq2 : (6 + 1) / 3 = 7 / 3 := by sorry
  rw [eq1, eq2]
  exact sorry

end plus_recurring_decimal_l324_324246


namespace product_sin_val_l324_324092

theorem product_sin_val (n : ℕ) (h : n ≥ 2) :
  ∏ k in finset.range (n - 1), Real.sin (π * (n * (k + 1) + (k + 1)) / (n * (n + 1))) = 1 / 2 ^ (n - 1) :=
sorry

end product_sin_val_l324_324092


namespace smallest_k_remainder_1_l324_324177

theorem smallest_k_remainder_1
  (k : ℤ) : 
  (k > 1) ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 4 = 1)
  ↔ k = 105 :=
by
  sorry

end smallest_k_remainder_1_l324_324177


namespace probability_of_1_in_first_20_rows_of_pascals_triangle_l324_324661

theorem probability_of_1_in_first_20_rows_of_pascals_triangle :
  (let total_elements := (20 * 21) / 2 in
   let number_of_ones := 1 + 2 * 19 in
   number_of_ones / total_elements = 13 / 70) := by
{
  let total_elements := (20 * 21) / 2;
  let number_of_ones := 1 + 2 * 19;
  have h1 : total_elements = 210 := by
  {
    sorry, -- place the computation steps for total_elements here
  },
  have h2 : number_of_ones = 39 := by
  {
    sorry, -- place the computation steps for number_of_ones here
  },
  have h3 : 39 / 210 = 13 / 70 := by
  {
    sorry, -- place the simplification steps for the division here
  },
  exact h3,
}

end probability_of_1_in_first_20_rows_of_pascals_triangle_l324_324661


namespace unique_equidistant_point_in_africa_l324_324396

structure Coordinates where
  latitude : ℝ
  longitude : ℝ

noncomputable def equidistant_point (K F D : Coordinates) : Coordinates :=
  sorry -- This is where the calculation would occur, omitted for this task

def is_point_equidistant (K F D : Coordinates) (P : Coordinates) : Prop :=
  let dist (P1 P2 : Coordinates) : ℝ :=
    real.cos (P2.latitude * real.pi / 180) *
    real.cos (P1.latitude * real.pi / 180) *
    real.cos ((P2.longitude - P1.longitude) * real.pi / 180) +
    real.sin (P2.latitude * real.pi / 180) * 
    real.sin (P1.latitude * real.pi / 180)
  dist P K = dist P F ∧
  dist P F = dist P D

theorem unique_equidistant_point_in_africa : 
  let K := ⟨30, 31.25⟩        -- Cairo (K)
  let F := ⟨-34, 18.5⟩        -- Cape Town (F)
  let D := ⟨14.45, -17.25⟩    -- Dakar (D)
  is_point_equidistant K F D ⟨0, 14⟩ := 
sorry

end unique_equidistant_point_in_africa_l324_324396


namespace circle_equation_l324_324142

theorem circle_equation
  (C : ℝ × ℝ)
  (A : ℝ × ℝ)
  (hC : C = (-2, 1))
  (hA : A = (2, -2)) :
  ∃ r : ℝ, (x y : ℝ), (x + 2) ^ 2 + (y - 1) ^ 2 = r^2 ∧ r = 5 := 
sorry

end circle_equation_l324_324142


namespace problem1_problem2_l324_324412

-- Definition of points on a hyperbola
def hyperbola (a b : ℝ) (P : ℝ × ℝ) := (P.1 ^ 2 / a ^ 2) - (P.2 ^ 2 / b ^ 2) = 1

-- Definitions of given points
def F₁ : ℝ × ℝ := (-real.sqrt 3, 0)
def F₂ : ℝ × ℝ := (real.sqrt 3, 0)
def P₁ : ℝ × ℝ := (2, 1)

-- Condition: Slope of OM equals 3/2
def slope_OM_exists (M : ℝ × ℝ) : Prop := M.1 ≠ 0 ∧ (M.2 / M.1) = 3 / 2

-- Problem 1: Prove k₁k₂ = b² / a² given the conditions
theorem problem1 (a b : ℝ) (P₁ P₂ M : ℝ × ℝ) (k₁ k₂ : ℝ) :
  hyperbola a b P₁ ∧ hyperbola a b P₂ ∧ M = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2) ∧
  slope_OM_exists M ∧ ¬((P₁.1 + P₂.1) = 0) ∧ P₁.1 ≠ P₂.1 ∧ P₂.2 ≠ P₂.2 ∧
  k₁ = (P₁.2 - P₂.2) / (P₁.1 - P₂.1) ∧ k₂ = (M.2 / M.1) - k₁ → 
  k₁ * k₂ = (b ^ 2 / a ^ 2) :=
sorry

-- Problem 2: Find area of quadrilateral P₁F₁P₂F₂ given conditions
theorem problem2 (a b : ℝ) (P₂ : ℝ × ℝ) (M : ℝ × ℝ) :
  hyperbola a b P₁ ∧ hyperbola a b P₂ ∧
  M = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2) ∧ slope_OM_exists M ∧
  P₂.1 = -10 / 7 ∧ P₂.2 = -1 / 7 →
  let area := (real.sqrt 3) * (8 / 7) / 2 in area = 8 * (real.sqrt 3) / 7 :=
sorry

end problem1_problem2_l324_324412


namespace correct_statement_is_c_l324_324185

-- Definitions corresponding to conditions
def lateral_surface_of_cone_unfolds_into_isosceles_triangle : Prop :=
  false -- This is false because it unfolds into a sector.

def prism_with_two_congruent_bases_other_faces_rectangles : Prop :=
  false -- This is false because the bases are congruent and parallel, and all other faces are parallelograms.

def frustum_complemented_with_pyramid_forms_new_pyramid : Prop :=
  true -- This is true, as explained in the solution.

def point_on_lateral_surface_of_truncated_cone_has_countless_generatrices : Prop :=
  false -- This is false because there is exactly one generatrix through such a point.

-- The main proof statement
theorem correct_statement_is_c :
  ¬lateral_surface_of_cone_unfolds_into_isosceles_triangle ∧
  ¬prism_with_two_congruent_bases_other_faces_rectangles ∧
  frustum_complemented_with_pyramid_forms_new_pyramid ∧
  ¬point_on_lateral_surface_of_truncated_cone_has_countless_generatrices :=
by
  -- The proof involves evaluating all the conditions above.
  sorry

end correct_statement_is_c_l324_324185


namespace total_leaves_l324_324827

theorem total_leaves (ferns fronds leaves : ℕ) (h1 : ferns = 12) (h2 : fronds = 15) (h3 : leaves = 45) :
  ferns * fronds * leaves = 8100 :=
by
  sorry

end total_leaves_l324_324827


namespace solution_exists_l324_324125

theorem solution_exists (x y : ℝ) (hx : |x| + x + y = 15) (hy : x + |y| - y = 9) (hxy : y = 3x - 7) : x + y = 10.6 :=
sorry

end solution_exists_l324_324125


namespace pizza_combination_count_l324_324617

open_locale nat

-- Given condition: the number of different toppings
def num_toppings : ℕ := 8

-- Calculating the total number of one-topping, two-topping, and three-topping pizzas
theorem pizza_combination_count : ((nat.choose num_toppings 1) + (nat.choose num_toppings 2) + (nat.choose num_toppings 3)) = 92 :=
by
  sorry

end pizza_combination_count_l324_324617


namespace ratio_of_books_on_each_table_l324_324212

-- Define the conditions
variables (number_of_tables number_of_books : ℕ)
variables (R : ℕ) -- Ratio we need to find

-- State the conditions
def conditions := (number_of_tables = 500) ∧ (number_of_books = 100000)

-- Mathematical Problem Statement
theorem ratio_of_books_on_each_table (h : conditions number_of_tables number_of_books) :
    100000 = 500 * R → R = 200 :=
by
  sorry

end ratio_of_books_on_each_table_l324_324212


namespace spinner_points_east_l324_324395

-- Definitions for the conditions
def initial_direction := "north"

-- Clockwise and counterclockwise movements as improper fractions
def clockwise_move := (7 : ℚ) / 2
def counterclockwise_move := (17 : ℚ) / 4

-- Compute the net movement (negative means counterclockwise)
def net_movement := clockwise_move - counterclockwise_move

-- Translate net movement into a final direction (using modulo arithmetic with 1 revolution = 360 degrees equivalent)
def final_position : ℚ := (net_movement + 1) % 1

-- The goal is to prove that the final direction is east (which corresponds to 1/4 revolution)
theorem spinner_points_east :
  final_position = (1 / 4 : ℚ) :=
by
  sorry

end spinner_points_east_l324_324395


namespace friends_sum_equality_l324_324875

theorem friends_sum_equality (a : Fin 100 → ℕ) (c : Fin 100 → ℕ) :
  (∀ i : Fin 100, ∃ ai, a i = ai) ∧  -- Each student has some number of friends
  (∀ j : Fin 100, c j = finset.card {i : Fin 100 | a i > j}) →  -- c_j is the number of students with more than j friends
  finset.sum (finset.univ : finset (Fin 100)) a = finset.sum (finset.range 100) c :=
sorry

end friends_sum_equality_l324_324875


namespace parallelogram_area_l324_324261

-- Definition of the vertices
def V1 := (0, 0)
def V2 := (7, 0)
def V3 := (3, 5)
def V4 := (10, 5)

-- Definition of the base and height of the parallelogram
def base := (V2.1 - V1.1 : ℕ)
def height := (V3.2 - V1.2 : ℕ)

-- Theorem statement
theorem parallelogram_area : (base * height) = 35 :=
by
  -- We use the values directly, the proof itself is not required.
  sorry

end parallelogram_area_l324_324261


namespace daughter_age_in_3_years_l324_324957

variable (mother_age_now : ℕ) (gap_years : ℕ) (ratio : ℕ)

theorem daughter_age_in_3_years
  (h1 : mother_age_now = 41) 
  (h2 : gap_years = 5)
  (h3 : ratio = 2) :
  let mother_age_then := mother_age_now - gap_years in
  let daughter_age_then := mother_age_then / ratio in
  let daughter_age_now := daughter_age_then + gap_years in
  let daughter_age_in_3_years := daughter_age_now + 3 in
  daughter_age_in_3_years = 26 :=
  by
    sorry

end daughter_age_in_3_years_l324_324957


namespace cos_trig_identity_l324_324721

variable (α : ℝ)

theorem cos_trig_identity : cos (π / 6 - α) = 2 / 3 → cos (5 * π / 3 + 2 * α) = -1 / 9 := by
  sorry

end cos_trig_identity_l324_324721


namespace consecutive_sum_21_sets_l324_324779

theorem consecutive_sum_21_sets :
  (∃ (S : Set (Set ℕ)), (∀ s ∈ S, ∃ (a n : ℕ), n ≥ 3 ∧ s = {a, a+1, ..., a+n-1} ∧ (s.sum) = 21) ∧ S.card = 2) :=
sorry

end consecutive_sum_21_sets_l324_324779


namespace lambda_value_l324_324358

variable (a b : ℝ)
variable (OA OB : ℝ → ℝ)
variable (lambda : ℝ)

-- Conditions
variable (h1 : |OA| = a)
variable (h2 : |OB| = b)
variable (AD : ℝ → ℝ)
variable (AB : ℝ → ℝ)
variable (OD_ne_b : OA + lambda * (OB - OA) ⊥ AD)
variable (h3 : AD = lambda * (OB - OA))

-- Theorem statement
theorem lambda_value 
  : lambda = (a • (a - b)) / |a - b|^2 := 
sorry

end lambda_value_l324_324358


namespace sum_of_consecutive_odd_integers_mod_8_l324_324175

theorem sum_of_consecutive_odd_integers_mod_8 :
  let nums := [22277, 22279, 22281, 22283, 22285, 22287, 22289, 22291] in
  (nums.sum % 8) = 0 :=
by
  let nums := [22277, 22279, 22281, 22283, 22285, 22287, 22289, 22291]
  have h : nums = [22277, 22279, 22281, 22283, 22285, 22287, 22289, 22291] := rfl
  have h_sum : nums.sum = 22277 + 22279 + 22281 + 22283 + 22285 + 22287 + 22289 + 22291 := by
    rw h
    simp only [List.sum_cons, List.sum_nil]
  have h_mod : (22277 + 22279 + 22281 + 22283 + 22285 + 22287 + 22289 + 22291) % 8 = 0 := sorry
  show (nums.sum % 8) = 0
  rw [h_sum, h_mod]
  exact h_mod

end sum_of_consecutive_odd_integers_mod_8_l324_324175


namespace cot_alpha_solution_l324_324311

theorem cot_alpha_solution 
  (α : Real) 
  (h1 : Real.sin (2 * α) = - Real.sin α)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.cot α = -Real.sqrt 3 / 3 :=
by
  sorry

end cot_alpha_solution_l324_324311


namespace nth_term_divisible_by_10_pow_8_l324_324096

theorem nth_term_divisible_by_10_pow_8 :
  ∀ (n : ℕ), (let a1 := (5 : ℚ) / 3;
                  a2 := (25 : ℚ);
                  r := a2 / a1;
                  a_n := (r^(n - 1) * a1)
              in 10^8 ∣ a_n) 
              → n ≥ 9 := 
begin
  intros n hn,
  -- Using logic to show the provided statement will require the actual proof.
  sorry
end

end nth_term_divisible_by_10_pow_8_l324_324096


namespace surprise_shop_daily_revenue_l324_324265

def closed_days_per_year : ℕ := 3
def years_active : ℕ := 6
def total_revenue_lost : ℚ := 90000

def total_closed_days : ℕ :=
  closed_days_per_year * years_active

def daily_revenue : ℚ :=
  total_revenue_lost / total_closed_days

theorem surprise_shop_daily_revenue :
  daily_revenue = 5000 := by
  sorry

end surprise_shop_daily_revenue_l324_324265


namespace compare_fractions_l324_324741

theorem compare_fractions (a : ℝ) : 
  (a = 0 → (1 / (1 - a)) = (1 + a)) ∧ 
  (0 < a ∧ a < 1 → (1 / (1 - a)) > (1 + a)) ∧ 
  (a > 1 → (1 / (1 - a)) < (1 + a)) := by
  sorry

end compare_fractions_l324_324741


namespace square_root_of_9_l324_324941

theorem square_root_of_9 : {x : ℝ // x^2 = 9} = {x : ℝ // x = 3 ∨ x = -3} :=
by
  sorry

end square_root_of_9_l324_324941


namespace Justin_reads_130_pages_to_pass_l324_324400

theorem Justin_reads_130_pages_to_pass :
  ∀ (d1 : ℕ) (d2 : ℕ), 
  d1 = 10 → 
  d2 = 2 * d1 →
  (6 * d2 + d1) = 130 →
  (∃ p : ℕ, p = 130) :=
by
  intros d1 d2 h1 h2 h3
  use 130
  sorry

end Justin_reads_130_pages_to_pass_l324_324400


namespace marble_count_l324_324548

noncomputable def total_marbles : ℕ := 24

def red_marbles (total : ℕ) : ℕ := total / 4

def blue_marbles (red : ℕ) : ℕ := red + 6

def yellow_marbles (total red blue : ℕ) : ℕ := total - (red + blue)

theorem marble_count (total : ℕ) :
  let red := red_marbles total,
      blue := blue_marbles red,
      yellow := yellow_marbles total red blue
  in blue > red ∧ blue > yellow :=
by
  sorry

end marble_count_l324_324548


namespace vertical_angles_equal_l324_324533

theorem vertical_angles_equal {α β : Type} [LinearOrderedField α] [LinearOrder β] (a1 a2 : β) 
  (h : a1 = a2) : "If two angles are vertical angles, then these two angles are equal." :=
sorry

end vertical_angles_equal_l324_324533


namespace semicircle_radius_approx_l324_324641

noncomputable def semicircle_radius (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem semicircle_radius_approx (h : 46.27433388230814 = (Real.pi * r + 2 * r)) : 
  semicircle_radius 46.27433388230814 ≈ 8.998883928 :=
by
  sorry

end semicircle_radius_approx_l324_324641


namespace james_bought_400_fish_l324_324397

theorem james_bought_400_fish
  (F : ℝ)
  (h1 : 0.80 * F = 320)
  (h2 : F / 0.80 = 400) :
  F = 400 :=
by
  sorry

end james_bought_400_fish_l324_324397


namespace conjugate_is_correct_l324_324808

noncomputable def z : ℂ := (1 : ℂ) - (2 : ℂ) * complex.I

theorem conjugate_is_correct : complex.conj z = 1 + 2 * complex.I := by
  -- The proof goes here
  sorry

end conjugate_is_correct_l324_324808


namespace tim_score_l324_324645

theorem tim_score :
  let single_line_points := 1000
  let tetris_points := 8 * single_line_points
  let singles_scored := 6
  let tetrises_scored := 4
  in singles_scored * single_line_points + tetrises_scored * tetris_points = 38000 := by
  sorry

end tim_score_l324_324645


namespace pizza_toppings_count_l324_324619

theorem pizza_toppings_count :
  let toppings := 8 in
  let one_topping_pizzas := toppings in
  let two_topping_pizzas := (toppings.choose 2) in
  let three_topping_pizzas := (toppings.choose 3) in
  one_topping_pizzas + two_topping_pizzas + three_topping_pizzas = 92 :=
by sorry

end pizza_toppings_count_l324_324619


namespace restaurant_sodas_l324_324225

theorem restaurant_sodas (M : ℕ) (h1 : M + 19 = 96) : M = 77 :=
by
  sorry

end restaurant_sodas_l324_324225


namespace find_f_3_l324_324503

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l324_324503


namespace solve_quadratic_l324_324870

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 / 2 ∨ x = 1 :=
by
  intro x
  construct sorry

end solve_quadratic_l324_324870


namespace probability_sum_greater_than_product_l324_324237

theorem probability_sum_greater_than_product : 
  let s := {1, 2, 3, 4}
  let number_pairs := {p : ℕ × ℕ | p.fst ∈ s ∧ p.snd ∈ s ∧ p.fst ≠ p.snd}
  let valid_pairs := {p ∈ number_pairs | p.fst + p.snd > p.fst * p.snd}
  valid_pairs.to_finset.card.to_real / number_pairs.to_finset.card.to_real = 1 / 2
  := 
sorry

end probability_sum_greater_than_product_l324_324237


namespace find_f_3_l324_324500

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l324_324500


namespace fill_cistern_time_l324_324191

theorem fill_cistern_time :
  let r_fill := 1 / 4
  let r_empty := 1 / 9
  let r_net := r_fill - r_empty
  r_net > 0
  (1 / r_net) = 36 / 5 :=
by
  sorry

end fill_cistern_time_l324_324191


namespace locus_of_P_l324_324403

-- Define the regular tetrahedron, point types, and orthogonality conditions
variables {Point : Type} [metric_space Point]
noncomputable def regular_tetrahedron (A B C D : Point) : Prop := sorry
noncomputable def orthogonal (P Q R : Point) : Prop := sorry

-- Main theorem stating the loci of P
theorem locus_of_P (A B C D M P : Point) 
  (h_tetrahedron : regular_tetrahedron A B C D)
  (h_M_on_CD : M ∈ segment ℝ C D)
  (h_intersection : ∃ H K, orthogonal A B M → P = intersection (line_through A H) (line_through B K)
                     ∧ orthogonal C D K → orthogonal B A M) :
  (locus P (λ M, M ∈ line_through C D) = circumcircle_of_triangle (midpoint A B) center_of_triangle A B C center_of_triangle A B D \ {midpoint A B}) ∧
  (locus P (λ M, M ∈ segment ℝ C D) = arc_of_circumcircle (center_of_triangle A B C) (center_of_triangle A B D)) :=
  sorry

end locus_of_P_l324_324403


namespace trapezoid_height_l324_324136

-- Definitions of the problem conditions
def is_isosceles_trapezoid (a b : ℝ) : Prop :=
  ∃ (AB CD BM CN h : ℝ), a = 24 ∧ b = 10 ∧ AB = 25 ∧ CD = 25 ∧ BM = h ∧ CN = h ∧
  BM ^ 2 + ((24 - 10) / 2) ^ 2 = AB ^ 2

-- The theorem to prove
theorem trapezoid_height (a b : ℝ) (h : ℝ) 
  (H : is_isosceles_trapezoid a b) : h = 24 :=
sorry

end trapezoid_height_l324_324136


namespace num_odd_digits_base4_437_l324_324710

theorem num_odd_digits_base4_437 : 
  let base4_representation := toBaseN 4 437
  let num_odd_digits := (base4_representation.data.count (fun x => x % 2 = 1))
  num_odd_digits = 4 := 
by 
  let base4_representation := 12311 -- Base-4 representation of 437
  let num_odd_digits := List.count (fun x => x % 2 = 1) [1, 2, 3, 1, 1]
  show num_odd_digits = 4 from sorry

end num_odd_digits_base4_437_l324_324710


namespace minimum_distance_between_points_l324_324113

noncomputable def circle_O (P : ℝ × ℝ) : Prop :=
P.1^2 + P.2^2 = 1

noncomputable def circle_C (Q : ℝ × ℝ) : Prop :=
(Q.1 - 3)^2 + Q.2^2 = 1

theorem minimum_distance_between_points (P Q : ℝ × ℝ) :
  circle_O P →
  circle_C Q →
  ∃ (min_dist : ℝ), min_dist = 1 ∧ ∀ (P Q : ℝ × ℝ), circle_O P → circle_C Q → |(P.1 - Q.1)^2 + (P.2 - Q.2)^2|^(1/2) ≥ min_dist :=
begin
  intros hP hQ,
  use 1,
  split,
  { refl, },
  { intros P' Q' hP' hQ',
    sorry
  }
end

end minimum_distance_between_points_l324_324113


namespace triangle_is_acute_l324_324301

theorem triangle_is_acute
  (ABC : Triangle)
  (O : Point)
  (I : Point)
  (circumcenter: is_circumcenter O ABC)
  (incenter: is_incenter I ABC)
  (O_inside_incircle: O ∈ incircle I (triangle_sides_length ABC))
  : is_acute_triangle ABC :=
sorry

end triangle_is_acute_l324_324301


namespace derivative_of_f_at_minus_2_l324_324037

theorem derivative_of_f_at_minus_2 (f : ℝ → ℝ) (h : f = λ x, x^3) : deriv (λ x, f (-2)) = 0 :=
by
  rw h
  unfold f
  -- Unfolding the definition and simplifying the problem
  simp
  exact deriv_const (-8)
  -- Alternatively, following traditional steps:
  -- have h1 : f(-2) = -8 := by
  --   rw h
  --   simp
  -- exact deriv_const (-8)

end derivative_of_f_at_minus_2_l324_324037


namespace sequence_property_l324_324834

noncomputable def floor (x : ℝ) : ℕ := ⌊x⌋₊

theorem sequence_property (a : ℕ → ℕ) (c : ℕ) (h_c_pos : c > 0)
  (h_sequence : ∀ n, (∑ i in finset.range n.succ, a (floor (n.to_real / i.to_real))) = n^10) :
  ∀ n : ℕ, n > 0 → (c^a n - c^a (n-1)) % n = 0 :=
by { sorry }

end sequence_property_l324_324834


namespace remainder_is_approx_9_l324_324578

noncomputable def remainder_approx {x y : ℕ} (hcond1 : x ≥ 0) (hcond2 : y > 0)
    (hdiv : (x : ℝ) / (y : ℝ) = 96.45) (hy_approx : y ≈ 20) : ℕ :=
  let decimal_part := 0.45 * (y : ℝ) in
  round decimal_part

theorem remainder_is_approx_9 {x y : ℕ} (hcond1 : x ≥ 0) (hcond2 : y > 0)
    (hdiv : (x : ℝ) / (y : ℝ) = 96.45) (hy_approx : y ≈ 20) :
  remainder_approx hcond1 hcond2 hdiv hy_approx = 9 :=
sorry

end remainder_is_approx_9_l324_324578


namespace equivalent_discount_calculation_l324_324222

theorem equivalent_discount_calculation :
  ∃ d : ℚ, d = 40.94 / 100 ∧ 
  let original_price := 120 in
  let coupon_price := original_price - 15 in
  let first_discount_price := coupon_price * 0.75 in
  let final_price := first_discount_price * 0.90 in
  final_price = original_price * (1 - d) :=
by
  sorry

end equivalent_discount_calculation_l324_324222


namespace acute_angle_inclination_range_l324_324040

/-- 
For the line passing through points P(1-a, 1+a) and Q(3, 2a), 
prove that the range of the real number a such that the line has an acute angle of inclination is (-∞, 1) ∪ (1, 4).
-/
theorem acute_angle_inclination_range (a : ℝ) : 
  (a < 1 ∨ (1 < a ∧ a < 4)) ↔ (0 < (a - 1) / (4 - a)) :=
sorry

end acute_angle_inclination_range_l324_324040


namespace intersection_eq_l324_324432

def M : set ℝ := {x : ℝ | x^2 - x ≥ 0}
def N : set ℝ := {x : ℝ | x < 2}
def intersection : set ℝ := {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)}

theorem intersection_eq : M ∩ N = intersection := 
by
  -- Proof steps here
  sorry

end intersection_eq_l324_324432


namespace right_triangle_hypotenuse_angle_45_degrees_l324_324671

noncomputable def right_triangle_angle (a b : ℝ) (h₀ : a + b + (real.sqrt (a^2 + b^2)) = 2) : ℝ :=
  let c := real.sqrt (a^2 + b^2) in
  let P := (1 / real.sqrt 2, 1 / real.sqrt 2) in
  real.arctan ((1 + a / (real.sqrt 2)) / (1 + b / (real.sqrt 2))) * (180 / real.pi) / 2

theorem right_triangle_hypotenuse_angle_45_degrees : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0
  ∧ a + b + (real.sqrt (a^2 + b^2)) = 2 
  ∧ right_triangle_angle (a b (by sorry)) = 45 :=
begin
  sorry
end

end right_triangle_hypotenuse_angle_45_degrees_l324_324671


namespace remaining_days_to_complete_job_l324_324949

-- Define the given conditions
def in_10_days (part_of_job_done : ℝ) (days : ℕ) : Prop :=
  part_of_job_done = 1 / 8 ∧ days = 10

-- Define the complete job condition
def complete_job (total_days : ℕ) : Prop :=
  total_days = 80

-- Define the remaining days to finish the job
def remaining_days (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) : Prop :=
  total_days_worked = 80 ∧ days_worked = 10 ∧ remaining = 70

-- The theorem statement
theorem remaining_days_to_complete_job (part_of_job_done : ℝ) (days : ℕ) (total_days : ℕ) (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) :
  in_10_days part_of_job_done days → complete_job total_days → remaining_days total_days_worked days_worked remaining :=
sorry

end remaining_days_to_complete_job_l324_324949


namespace quadratic_equation_roots_l324_324530

theorem quadratic_equation_roots :
  ∃ (b c : ℝ), (∀ (x : ℝ), (x - (1 + sqrt 2)) * (x - (1 - sqrt 2)) = x^2 + b * x + c) ∧
  b = -2 ∧
  c = -1 :=
begin
  use [-2, -1],
  split,
  { intro x,
    ring,
    norm_num,
  },
  split;
  refl,
end

end quadratic_equation_roots_l324_324530


namespace product_series_simplified_l324_324120

theorem product_series_simplified :
  (∏ n in Finset.range 125, (8 * n + 7) / (8 * n - 1)) = 333 :=
by
  sorry

end product_series_simplified_l324_324120


namespace lawn_length_l324_324638

theorem lawn_length :
  ∃ L : ℝ, (∀ (width : ℝ) (road_width : ℝ) (cost : ℝ), 
            width = 60 ∧ road_width = 10 ∧ cost = 2600 ∧ 
            2 * ((road_width * L + road_width * width - road_width * road_width)) = cost)
  → L = 80 :=
by {
  intros L exists_conditions,
  use 80,
  intros width road_width cost conditions,
  obtain ⟨width_eq, road_width_eq, cost_eq, equation⟩ := conditions,
  rw [width_eq, road_width_eq, cost_eq] at equation,
  have sim_eq : 2 * (10 * 80 + 10 * 60 - 10 * 10) = 2600,
  { calc 2 * (10 * 80 + 10 * 60 - 10 * 10)
      = 2 * (800 + 600 - 100) : by ring
  ... = 2 * 1300 : by ring
  ... = 2600 : by ring },
  exact equation
},
sorry

end lawn_length_l324_324638


namespace pizza_topping_count_l324_324626

theorem pizza_topping_count (n : ℕ) (h : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by {
  rw h,
  simp [nat.choose],
  sorry,
}

end pizza_topping_count_l324_324626


namespace green_balls_removal_l324_324048

theorem green_balls_removal : 
  ∀ (initial_total : ℕ) (initial_green_percentage : ℚ) (desired_green_percentage : ℚ),
  initial_total = 800 →
  initial_green_percentage = 0.7 →
  desired_green_percentage = 0.6 →
  let initial_green := initial_total * initial_green_percentage in
  let initial_yellow := initial_total - initial_green in
  let x := initial_green - 200 in
  let remaining_total := initial_total - 200 in
  (x / remaining_total) = desired_green_percentage :=
by
  sorry

end green_balls_removal_l324_324048


namespace measure_8_liters_with_buckets_l324_324006

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l324_324006


namespace geom_seq_general_term_sum_b_terms_l324_324416

-- Define the geometric sequence and conditions
def geom_seq (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = sum (finset.range (n + 1)) a + 1

-- Define the sequence b_n from a_n
def seq_b (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = (-1 : ℤ) ^ n * (a n + n)

-- State the properties of the geometric sequence 
theorem geom_seq_general_term (a : ℕ → ℕ) (n : ℕ) (h : geom_seq a n) : 
  a n = 2^(n - 1) :=
sorry

theorem sum_b_terms (a b : ℕ → ℕ) (n : ℕ) (h_geom : geom_seq a n) (h_b : seq_b a b) :
  (finset.range (2 * n)).sum b = (4^n - 1) / 3 + n :=
sorry

end geom_seq_general_term_sum_b_terms_l324_324416


namespace min_moves_no_further_moves_possible_l324_324052

theorem min_moves_no_further_moves_possible (n : ℕ) (n_pos : n > 0) :
  ∃ m, m ≥ ⌊(n * n) / 3⌋ ∧ no_further_moves_possible n m :=
sorry

end min_moves_no_further_moves_possible_l324_324052


namespace volume_of_prism_l324_324463

theorem volume_of_prism (a : ℝ) (h_pos : 0 < a) (h_lat : ∀ S_lat, S_lat = a ^ 2) : 
  ∃ V, V = (a ^ 3 * (Real.sqrt 2 - 1)) / 4 :=
by
  sorry

end volume_of_prism_l324_324463


namespace triangle_side_ratio_l324_324816

noncomputable theory

variables {A B C : ℝ} {a b c : ℝ}
-- a, b, and c are the sides opposite to angles A, B, and C respectively in triangle ABC.

theorem triangle_side_ratio (h : b * Real.cos C + c * Real.cos B = 2 * b) : a / b = 2 := 
sorry

end triangle_side_ratio_l324_324816


namespace mean_problem_l324_324130

noncomputable def arithmetic_mean (x y z : ℝ) := (x + y + z) / 3
noncomputable def geometric_mean (x y z : ℝ) := (x * y * z) ^ (1 / 3)
noncomputable def harmonic_mean (x y z : ℝ) := 3 / ((1 / x) + (1 / y) + (1 / z))

theorem mean_problem (x y z : ℝ)
  (h0 : arithmetic_mean x y z = 10)
  (h1 : geometric_mean x y z = 6)
  (h2 : harmonic_mean x y z = 4) :
  x^2 + y^2 + z^2 = 576 := 
by
  -- Since x, y, z are real, their sums and products can be manipulated directly
  have h_sum : x + y + z = 30, from sorry,
  have h_prod : x * y * z = 216, from sorry,
  have h_harm_sum : (x * y + y * z + z * x) / (x * y * z) = 3 / 4, from sorry,
  have h_mul_sum : x * y + y * z + z * x = 162, from sorry,

  -- Calculate the square of the sum and subtract twice the product sum
  calc 
    x^2 + y^2 + z^2 
        = (x + y + z)^2 - 2 * (x * y + y * z + z * x) : by sorry
    ... = 30^2 - 2 * 162       : by rw [h_sum, h_mul_sum]
    ... = 900 - 324            : by norm_num
    ... = 576                  : by norm_num

-- Placeholder sorry to satisfy the Lean compiler
persist sorry

end mean_problem_l324_324130


namespace work_together_days_l324_324598

theorem work_together_days (hA : ∃ d : ℝ, d > 0 ∧ d = 15)
                          (hB : ∃ d : ℝ, d > 0 ∧ d = 20)
                          (hfrac : ∃ f : ℝ, f = (23 / 30)) :
  ∃ d : ℝ, d = 2 := by
  sorry

end work_together_days_l324_324598


namespace range_of_b_l324_324330

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x + 2
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (f x b) b

theorem range_of_b (b : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f x b = y) → (∀ z : ℝ, ∃ x : ℝ, g x b = z) → b ≥ 4 ∨ b ≤ -2 :=
sorry

end range_of_b_l324_324330


namespace H_functions_l324_324728

noncomputable def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1

def f1 (x : ℝ) : ℝ := Real.exp x + x
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := 3 * x - Real.sin x
noncomputable def f4 (x : ℝ) : ℝ :=
  if x = 0 then 0 else Real.log (Real.abs x)

theorem H_functions :
  is_H_function f1 ∧ ¬ is_H_function f2 ∧ is_H_function f3 ∧ ¬ is_H_function f4 :=
by
  sorry

end H_functions_l324_324728


namespace min_marked_squares_l324_324303

theorem min_marked_squares (n : ℕ) (h : n ≥ 2) : 
  ∃ N : ℕ, N = 1 / 4 * n * ( n + 2) ∧
           ∀ (board : fin n × fin n → bool), 
             (∀ i j, (i : fin n) < n → (j : fin n) < n → board (i, j) = true → 
              (∃ x y, abs (x - i) + abs (y - j) = 1 ∧ board (x, y) = true)) → 
             ∀ i j, (i : fin n) < n → (j : fin n) < n → 
              (∃ x y, abs (x - i) + abs (y - j) = 1 ∧ board (x, y) = true) :=
begin
  sorry
end

end min_marked_squares_l324_324303


namespace angle_AFD_equilateral_congruent_triangles_l324_324305

theorem angle_AFD_equilateral_congruent_triangles 
  (A B C D E F : Point)
  (h1 : EquilateralTriangle A B C)
  (h2 : EquilateralTriangle B D E)
  (h3 : collinear A B D)
  (h4 : same_halfplane (Line.mk A B D) C E)
  (h5 : F = intersection (Line.mk C D) (Line.mk A E)) :
  angle A F D = 120 :=
sorry

end angle_AFD_equilateral_congruent_triangles_l324_324305


namespace pizza_combination_count_l324_324614

open_locale nat

-- Given condition: the number of different toppings
def num_toppings : ℕ := 8

-- Calculating the total number of one-topping, two-topping, and three-topping pizzas
theorem pizza_combination_count : ((nat.choose num_toppings 1) + (nat.choose num_toppings 2) + (nat.choose num_toppings 3)) = 92 :=
by
  sorry

end pizza_combination_count_l324_324614


namespace max_value_correct_l324_324422

noncomputable def max_value_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) : Prop :=
  x ^ 4 * y ^ 3 * z ^ 2 ≤ 1 / 372008

theorem max_value_correct (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) :
  max_value_ineq x y z h1 h2 h3 h4 :=
sorry

end max_value_correct_l324_324422


namespace mean_problem_l324_324129

noncomputable def arithmetic_mean (x y z : ℝ) := (x + y + z) / 3
noncomputable def geometric_mean (x y z : ℝ) := (x * y * z) ^ (1 / 3)
noncomputable def harmonic_mean (x y z : ℝ) := 3 / ((1 / x) + (1 / y) + (1 / z))

theorem mean_problem (x y z : ℝ)
  (h0 : arithmetic_mean x y z = 10)
  (h1 : geometric_mean x y z = 6)
  (h2 : harmonic_mean x y z = 4) :
  x^2 + y^2 + z^2 = 576 := 
by
  -- Since x, y, z are real, their sums and products can be manipulated directly
  have h_sum : x + y + z = 30, from sorry,
  have h_prod : x * y * z = 216, from sorry,
  have h_harm_sum : (x * y + y * z + z * x) / (x * y * z) = 3 / 4, from sorry,
  have h_mul_sum : x * y + y * z + z * x = 162, from sorry,

  -- Calculate the square of the sum and subtract twice the product sum
  calc 
    x^2 + y^2 + z^2 
        = (x + y + z)^2 - 2 * (x * y + y * z + z * x) : by sorry
    ... = 30^2 - 2 * 162       : by rw [h_sum, h_mul_sum]
    ... = 900 - 324            : by norm_num
    ... = 576                  : by norm_num

-- Placeholder sorry to satisfy the Lean compiler
persist sorry

end mean_problem_l324_324129


namespace sum_divisible_by_9_l324_324059

theorem sum_divisible_by_9 {
  a b c d e f : ℕ
  (h_sum : a + b + c = c + d + e ∧ c + d + e = e + f + a) :
  (a + b + c + d + e + f) % 9 = 0 :=
sorry

end sum_divisible_by_9_l324_324059


namespace jessica_cut_21_roses_l324_324163

def initial_roses : ℕ := 2
def thrown_roses : ℕ := 4
def final_roses : ℕ := 23

theorem jessica_cut_21_roses : (final_roses - initial_roses) = 21 :=
by
  sorry

end jessica_cut_21_roses_l324_324163


namespace find_f_of_3_l324_324493

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l324_324493


namespace find_f_3_l324_324505

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l324_324505


namespace tetrahedron_face_condition_nonempty_l324_324423

noncomputable def regular_tetrahedron := Type

def isometry (F : regular_tetrahedron → regular_tetrahedron) : Prop :=
∀ x y : regular_tetrahedron, dist (F x) (F y) = dist x y

variables (A B C D : regular_tetrahedron)
variables (Z : regular_tetrahedron → regular_tetrahedron)
variable (t : ℝ)

def is_regular_tetrahedron (A B C D : regular_tetrahedron) : Prop :=
dist A B = dist A C ∧ dist A C = dist A D ∧
dist B C = dist B D ∧ dist C D = dist A B

def is_Z_isometry (Z A B C D : regular_tetrahedron) : Prop :=
Z A = B ∧ Z B = C ∧ Z C = D ∧ Z D = A

def is_on_face (X : regular_tetrahedron) (A B C : regular_tetrahedron) : Prop :=
∃ α β : ℝ, 0 ≤ α ∧ 0 ≤ β ∧ α + β ≤ 1 ∧ X = α * A + β * B + (1 - α - β) * C

def distance_condition (X : regular_tetrahedron) (Z : regular_tetrahedron → regular_tetrahedron) (t : ℝ) : Prop :=
dist X (Z X) = t

noncomputable def face_points_nonempty (A B C D : regular_tetrahedron) (Z : regular_tetrahedron → regular_tetrahedron) (t : ℝ) : Prop :=
∀ X : regular_tetrahedron, (is_on_face X A B C) → distance_condition X Z t

theorem tetrahedron_face_condition_nonempty
  (A B C D : regular_tetrahedron)
  (Z : regular_tetrahedron → regular_tetrahedron)
  (t : ℝ)
  (h1 : is_regular_tetrahedron A B C D)
  (h2 : is_Z_isometry Z A B C D)
  : (face_points_nonempty A B C D Z t) ↔ 1 / real.sqrt 10 ≤ t ∧ t ≤ 1 := sorry

end tetrahedron_face_condition_nonempty_l324_324423


namespace intersection_polar_coordinates_l324_324805

variable {t : ℝ}

def C1 (x y : ℝ) : Prop := x^2 + y^2 = 2

def C2 (t : ℝ) : ℝ × ℝ := (2 - t, t)

theorem intersection_polar_coordinates :
  ∃ ρ θ, (ρ = real.sqrt 2 ∧ θ = real.pi / 4) ∧ 
  (∃ t, (2 - t)^2 + t^2 = 2 ∧ ρ = real.sqrt ((2 - t)^2 + t^2) ∧ θ = real.atan (t / (2 - t))) :=
by
  -- Proof has been omitted.
  sorry

end intersection_polar_coordinates_l324_324805


namespace simple_interest_rate_l324_324034

theorem simple_interest_rate (P R: ℝ) (T : ℝ) (hT : T = 8) (h : 2 * P = P + (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for proof steps
  sorry

end simple_interest_rate_l324_324034


namespace numWaysToCutAndArrange_is_465_l324_324966

/-- The length of the wire -/
def wireLength : ℕ := 25

/-- The length of the 2-meter pieces -/
def lengthTwoMeterPiece : ℕ := 2

/-- The length of the 3-meter pieces -/
def lengthThreeMeterPiece : ℕ := 3

/-- The total number of ways to cut and arrange pieces -/
def numWaysToCutAndArrange (totalLength : ℕ) (length2 : ℕ) (length3 : ℕ) : ℕ :=
  let rec countWays (total : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
    if total < 0 then 0
    else if total = 0 then nat.factorial (x + y) / (nat.factorial x * nat.factorial y)
    else countWays (total - length2) (x + 1) y + countWays (total - length3) x (y + 1)
  countWays totalLength 0 0

theorem numWaysToCutAndArrange_is_465 : numWaysToCutAndArrange wireLength lengthTwoMeterPiece lengthThreeMeterPiece = 465 := sorry

end numWaysToCutAndArrange_is_465_l324_324966


namespace problem_solution_l324_324310

theorem problem_solution (a b : ℝ) (h₁ : {1, a + b, a} = {0, b / a, b}) : b - a = 2 := 
by 
  sorry

end problem_solution_l324_324310


namespace tangent_line_intersection_l324_324750

noncomputable def a_tangent_line_intersection : ℝ := 8

theorem tangent_line_intersection :
  let y1 := λ x : ℝ, x + Real.log x,
      y2 := λ x : ℝ, a_tangent_line_intersection * x^2 + (a_tangent_line_intersection + 2) * x + 1,
      tangent := λ x : ℝ, 2 * x - 1 in
  (tangent 1 = y1 1) ∧
  (∃ x₀ : ℝ, y2 x₀ = tangent x₀) ∧
  ∀ x : ℝ, y1 x = tangent x ↔ y2 x = tangent x :=
by sorry

end tangent_line_intersection_l324_324750


namespace num_proper_irreducible_fractions_l324_324343
open Nat

theorem num_proper_irreducible_fractions :
  {n : Nat | n < 100 ∧ ∃ m : Nat, m < 100 ∧ n + m = 100 ∧ gcd n m = 1}.card = 20 :=
by
  sorry

end num_proper_irreducible_fractions_l324_324343


namespace bryan_has_more_candies_l324_324675

theorem bryan_has_more_candies (bryans_skittles : ℕ) (bens_mms : ℕ)
  (hbryan : bryans_skittles = 50) (hben : bens_mms = 20) :
  bryans_skittles - bens_mms = 30 :=
by {
  rw [hbryan, hben],
  exact rfl,
}

end bryan_has_more_candies_l324_324675


namespace right_triangle_similar_side_length_l324_324999

theorem right_triangle_similar_side_length
  (a b c d : ℝ)
  (h1 : a^2 + b^2 = c^2)
  (h2 : b = 15)
  (h3 : c = 17)
  (similar_hypotenuse : 3 * c = d)
  (d = 51) :
  ∃ e, (a * 3 = e) ∧ e = 24 :=
by
  -- Use the given conditions and definitions to construct the proof.
  sorry

end right_triangle_similar_side_length_l324_324999


namespace distance_to_larger_cross_section_l324_324562

theorem distance_to_larger_cross_section (a₁ a₂ : ℝ) (d : ℝ) : 
  a₁ = 144 * Real.sqrt 3 → 
  a₂ = 324 * Real.sqrt 3 → 
  d = 6 → 
  (h : ℝ), h - (2 / 3) * h = 6 → 
  h = 18 := 
by
  intros
  sorry

end distance_to_larger_cross_section_l324_324562


namespace distance_preserved_l324_324402

-- Let S be a set of n concentric circles.
noncomputable def S : Type := { s : ℝ // s ≥ 0 }
-- Define the Euclidean distance function between radii.
def d (A B : S) : ℝ := abs (A.val - B.val)

-- Declare the function f mapping S to S.
noncomputable def f : S → S := sorry

-- Assume the given property: d(f(A), f(B)) ≥ d(A, B) for all A, B in S.
axiom f_property (A B : S) : d (f A) (f B) ≥ d A B

-- Prove the target property: d(f(A), f(B)) = d(A, B) for all A, B in S.
theorem distance_preserved (A B : S) : d (f A) (f B) = d A B :=
begin
  sorry
end

end distance_preserved_l324_324402


namespace problem_statement_l324_324882

noncomputable def r (C: ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def A (r: ℝ) : ℝ := Real.pi * r^2

noncomputable def combined_area_difference (C1 C2 C3: ℝ) : ℝ :=
  let r1 := r C1
  let r2 := r C2
  let r3 := r C3
  let A1 := A r1
  let A2 := A r2
  let A3 := A r3
  (A3 - A1) - A2

theorem problem_statement : combined_area_difference 528 704 880 = -9.76 :=
by
  sorry

end problem_statement_l324_324882


namespace pizza_toppings_l324_324631

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l324_324631


namespace equilibrium_is_unstable_l324_324821

-- Definitions corresponding to conditions
def system_ode (x y : ℝ) : (ℝ × ℝ) :=
  (2 * x + y - 5 * y^2, 3 * x^6 + y + (x^3) / 2)

def equilibrium_point : (ℝ × ℝ) := (0, 0)

-- Main theorem statement
theorem equilibrium_is_unstable : 
  ∃ (λ : ℝ), λ = (3 + Real.sqrt 13) / 2 ∧ λ >= 0 :=
sorry

end equilibrium_is_unstable_l324_324821


namespace team_average_typing_speed_l324_324465

-- Definitions of typing speeds of each team member
def typing_speed_rudy := 64
def typing_speed_joyce := 76
def typing_speed_gladys := 91
def typing_speed_lisa := 80
def typing_speed_mike := 89

-- Number of team members
def number_of_team_members := 5

-- Total typing speed calculation
def total_typing_speed := typing_speed_rudy + typing_speed_joyce + typing_speed_gladys + typing_speed_lisa + typing_speed_mike

-- Average typing speed calculation
def average_typing_speed := total_typing_speed / number_of_team_members

-- Theorem statement
theorem team_average_typing_speed : average_typing_speed = 80 := by
  sorry

end team_average_typing_speed_l324_324465


namespace solution_set_of_inequality_l324_324538

theorem solution_set_of_inequality :
  { x : ℝ | x * (x - 1) ≤ 0 } = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
by
sorry

end solution_set_of_inequality_l324_324538


namespace subsets_contain_5_eq_4_l324_324336

-- Define the set A
def A : Set ℤ := {-1, 5, 1}

-- Definition of the problem: The number of subsets of A that contain the element 5
def num_subsets_containing_5 : ℕ :=
  Set.card { B : Set ℤ | B ⊆ A ∧ 5 ∈ B }

-- Prove that this number is 4
theorem subsets_contain_5_eq_4 : num_subsets_containing_5 = 4 :=
by
  -- The proof will be provided here
  sorry

end subsets_contain_5_eq_4_l324_324336


namespace parabola_translation_l324_324169

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

-- Define the transformation: translate one unit to the right
def translate_right (x : ℝ) : ℝ := initial_parabola (x - 1)

-- Define the transformation: move up three units
def move_up (y : ℝ) : ℝ := y + 3

-- Define the resulting equation after the transformations
def resulting_parabola (x : ℝ) : ℝ := move_up (translate_right x)

-- Define the target equation
def target_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Formalize the proof problem
theorem parabola_translation :
  ∀ x : ℝ, resulting_parabola x = target_parabola x :=
by
  -- Proof steps go here
  sorry

end parabola_translation_l324_324169


namespace find_first_term_l324_324081

theorem find_first_term (S_n : ℕ → ℝ) (a d : ℝ) (n : ℕ) (h₁ : ∀ n > 0, S_n n = n * (2 * a + (n - 1) * d) / 2)
  (h₂ : d = 3) (h₃ : ∃ c, ∀ n > 0, S_n (3 * n) / S_n n = c) : a = 3 / 2 :=
by
  sorry

end find_first_term_l324_324081


namespace mean_temperature_is_minus_two_l324_324126

-- Define the list of temperatures
def temperatures : List Int := [-7, -4, -4, -5, 1, 3, 2]

-- Define the function to calculate the mean of a list of integers
def mean (l : List Int) : Float :=
  (l.sum : Float) / (l.length : Float)

-- State the theorem to prove
theorem mean_temperature_is_minus_two : mean temperatures = -2 := by
  sorry

end mean_temperature_is_minus_two_l324_324126


namespace slices_per_pizza_l324_324828

-- Definitions based on the conditions
def num_pizzas : Nat := 3
def total_cost : Nat := 72
def cost_per_5_slices : Nat := 10

-- To find the number of slices per pizza
theorem slices_per_pizza (num_pizzas : Nat) (total_cost : Nat) (cost_per_5_slices : Nat): 
  (total_cost / num_pizzas) / (cost_per_5_slices / 5) = 12 :=
by
  sorry

end slices_per_pizza_l324_324828


namespace model2_has_better_fit_l324_324439

variable (x : ℝ)

def Model1 (x : ℝ) : ℝ := 3.5 * x - 2
def Model2 (x : ℝ) : ℝ := Real.sqrt x - 3

def R1_squared : ℝ := 0.87
def R2_squared : ℝ := 0.9

theorem model2_has_better_fit (h1 : R1_squared = 0.87) (h2 : R2_squared = 0.9) : R1_squared < R2_squared :=
by
  rw [h1, h2]
  exact sorry

end model2_has_better_fit_l324_324439


namespace remainder_of_sum_of_factorials_l324_324249

def factorial : ℕ → ℕ
| 0        := 1
| (n + 1)  := (n + 1) * factorial n

theorem remainder_of_sum_of_factorials :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 20 = 13 :=
by
  sorry  -- proof omitted

end remainder_of_sum_of_factorials_l324_324249


namespace total_cost_sufficient_funds_l324_324651

noncomputable def f (x : ℕ) : ℝ := (144 / x) + 4 * x

theorem total_cost (x : ℕ) (h_pos : 0 < x) (h_le : x ≤ 36) :
  f x = (144 / x) + 4 * x := by sorry

theorem sufficient_funds (x : ℕ) (h_pos : 0 < x) (h_natural : x = 6) :
  f x ≤ 480 :=
begin
  have h_f : f x = (144 / x) + 4 * x := by sorry,
  rw h_natural at h_f,
  norm_num at h_f,
  sorry
end

end total_cost_sufficient_funds_l324_324651


namespace find_f_of_3_l324_324476

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l324_324476


namespace midpoint_product_coordinates_l324_324080

theorem midpoint_product_coordinates :
  ∃ (x y : ℝ), (4 : ℝ) = (-2 + x) / 2 ∧ (-3 : ℝ) = (-7 + y) / 2 ∧ x * y = 10 := by
  sorry

end midpoint_product_coordinates_l324_324080


namespace right_triangle_area_l324_324560

theorem right_triangle_area (P Q R : Type) 
  (angle_P : angle (PQR) = 90)
  (PQ PR : ℝ)
  (hPQ : PQ = 8)
  (hPR : PR = 10)
  : area (triangle P Q R) = 40 :=
by
  sorry

end right_triangle_area_l324_324560


namespace decreasing_function_on_real_l324_324242

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x + f y
axiom f_negative (x : ℝ) : x > 0 → f x < 0
axiom f_not_identically_zero : ∃ x, f x ≠ 0

theorem decreasing_function_on_real :
  ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
sorry

end decreasing_function_on_real_l324_324242


namespace abc_value_l324_324084

variables (a b c : ℂ)

theorem abc_value :
  (a * b + 4 * b = -16) →
  (b * c + 4 * c = -16) →
  (c * a + 4 * a = -16) →
  a * b * c = 64 :=
by
  intros h1 h2 h3
  sorry

end abc_value_l324_324084


namespace arithmetic_sequence_expression_l324_324380

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

theorem arithmetic_sequence_expression
  (h_arith_seq : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -3) :
  ∀ n : ℕ, a n = -2 * n + 3 :=
  sorry

end arithmetic_sequence_expression_l324_324380


namespace pizza_toppings_l324_324629

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l324_324629


namespace probability_units_digit_less_than_three_l324_324986

theorem probability_units_digit_less_than_three :
  let even_digits := {0, 2, 4, 6, 8}
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ (d ∈ even_digits), d < 3) →
  (∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 2 = 0 ∧ (n % 10) < 3) → 
  (∃ (p : ℚ), p = 2 / 5) :=
by
  sorry

end probability_units_digit_less_than_three_l324_324986


namespace angle_DAB_is_45_degrees_l324_324357

-- Setup the problem space
def Point : Type := ℝ × ℝ

def is_square (A C D E : Point) : Prop :=
  dist A C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E A ∧ 
  ∠ A C D = 90 ∧ ∠ C D E = 90 ∧ ∠ D E A = 90 ∧ ∠ E A C = 90

-- Definitions based on conditions given
def right_isosceles_triangle (A B C : Point) : Prop :=
  dist A C = dist C B ∧ ∠ C = 90

-- Main theorem to prove
theorem angle_DAB_is_45_degrees (A B C D E : Point)
  (h₁ : right_isosceles_triangle A B C)
  (h₂ : is_square A C D E)
  (h₃ : on_same_line A C D E) :
  ∠ D A B = 45 :=
sorry

end angle_DAB_is_45_degrees_l324_324357


namespace relationship_among_abc_l324_324844

noncomputable def a (f : ℝ → ℝ) : ℝ := (1/2) * f (Real.pi / 3)
def b : ℝ := 0
noncomputable def c (f : ℝ → ℝ) : ℝ := -(Real.sqrt 3 / 2) * f (5 * Real.pi / 6)

-- The main theorem statement with conditions and required proof
theorem relationship_among_abc (f : ℝ → ℝ) (f'' : ℝ → ℝ) (h : ∀ x ∈ Ioo 0 Real.pi, f x * Real.sin x - f'' x * Real.cos x < 0) :
  a f < b ∧ b < c f := 
sorry

end relationship_among_abc_l324_324844


namespace last_two_digits_sum_factorial_l324_324275

theorem last_two_digits_sum_factorial (h : ∀ n, n ≥ 10 → (n! % 100 = 0)) : 
  (25! + 50! + 75! + 100! + 125!) % 100 = 0 :=
by
  sorry

end last_two_digits_sum_factorial_l324_324275


namespace circumference_semicircle_is_correct_l324_324590

/-- Define the length and breadth of the rectangle -/
def rectangle_length : ℝ := 8
def rectangle_breadth : ℝ := 6

/-- Define the perimeter of the rectangle -/
def perimeter_rectangle : ℝ := 2 * (rectangle_length + rectangle_breadth)

/-- Define the perimeter of the square -/
def perimeter_square : ℝ := perimeter_rectangle

/-- Define the side length of the square -/
def side_square : ℝ := perimeter_square / 4

/-- Define the diameter of the semicircle -/
def diameter_semicircle : ℝ := side_square

/-- Define the radius of the semicircle -/
def radius_semicircle : ℝ := diameter_semicircle / 2

/-- Define the circumference of the semicircle -/
def circumference_semicircle : ℝ := (Real.pi * diameter_semicircle) / 2 + diameter_semicircle

/-- Theorem statement: Prove the circumference of the semicircle is 17.99 cm -/
theorem circumference_semicircle_is_correct :
  circumference_semicircle = 17.99 := 
sorry

end circumference_semicircle_is_correct_l324_324590


namespace smallest_positive_integer_l324_324180

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 60606 * n = 273 :=
sorry

end smallest_positive_integer_l324_324180


namespace find_f_of_3_l324_324492

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l324_324492


namespace padic_quadratic_solutions_l324_324445

open Classical

-- p-adic numbers defined for the context of this problem
noncomputable def padic_numbers := sorry

-- Definition of the proof problem
theorem padic_quadratic_solutions (a : padic_numbers) (ha : a ≠ 0) : 
  ∃ (x1 x2 : padic_numbers), (x1 ≠ x2 ∧ x1 ^ 2 = a ∧ x2 ^ 2 = a) ∨ 
  (∀ x : padic_numbers, x ^ 2 ≠ a) :=
sorry

end padic_quadratic_solutions_l324_324445


namespace calculate_bmw_sales_and_revenue_l324_324982

variable (total_cars : ℕ) (percentage_ford percentage_toyota percentage_nissan percentage_audi : ℕ) (avg_price_bmw : ℕ)
variable (h_total_cars : total_cars = 300) (h_percentage_ford : percentage_ford = 10)
variable (h_percentage_toyota : percentage_toyota = 25) (h_percentage_nissan : percentage_nissan = 20)
variable (h_percentage_audi : percentage_audi = 15) (h_avg_price_bmw : avg_price_bmw = 35000)

theorem calculate_bmw_sales_and_revenue :
  let percentage_non_bmw := percentage_ford + percentage_toyota + percentage_nissan + percentage_audi
  let percentage_bmw := 100 - percentage_non_bmw
  let number_bmw := total_cars * percentage_bmw / 100
  let total_revenue := number_bmw * avg_price_bmw
  (number_bmw = 90) ∧ (total_revenue = 3150000) := by
  -- Definitions are taken from conditions and used directly in the theorem statement
  sorry

end calculate_bmw_sales_and_revenue_l324_324982


namespace robotics_club_neither_l324_324853

theorem robotics_club_neither (total_students cs_students e_students both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 52)
  (h3 : e_students = 45)
  (h4 : both_students = 32) :
  total_students - (cs_students - both_students + e_students - both_students + both_students) = 15 :=
by
  sorry

end robotics_club_neither_l324_324853


namespace num_values_g_50_eq_9_l324_324715

-- Definitions based on the conditions provided
open Nat

def num_divisors_less_than (n : ℕ) : ℕ :=
  (range (n - 1)).filter (λ d, d + 1 ∣ n).length

def g_1 (n : ℕ) : ℕ :=
  3 * num_divisors_less_than n

def g_j : ℕ → ℕ → ℕ
| 1, n := g_1 n
| (j+1), n := g_1 (g_j j n)

-- Main statement: There are exactly 3 values of n ≤ 60 such that g_50(n) = 9
theorem num_values_g_50_eq_9 : (finset.range 60).filter (λ n, g_j 50 n = 9).card = 3 :=
sorry

end num_values_g_50_eq_9_l324_324715


namespace quadrilateral_not_parallelogram_l324_324183

-- Definitions used in the conditions
def opposite_sides_parallel_and_equal (q : Quadrilateral) : Prop :=
  ∃ p1 p2 p3 p4, q.points = [p1, p2, p3, p4] ∧
  (p1 - p2 = p3 - p4 ∧ p1 - p4 ∥ p2 - p3) 

-- The statement to prove
theorem quadrilateral_not_parallelogram (q : Quadrilateral) :
  ¬ opposite_sides_parallel_and_equal q :=
sorry

end quadrilateral_not_parallelogram_l324_324183


namespace sum_of_oscillatory_sequence_is_zero_l324_324456

def oscillatory_property (seq : ℕ → ℤ) : Prop :=
∀ n, seq (n+1) = seq n + seq (n+2)

def is_period_six_zero_sum (seq : ℕ → ℤ) : Prop :=
∀ n, seq n + seq (n+2) + seq (n+4) = 0

theorem sum_of_oscillatory_sequence_is_zero
  (seq : ℕ → ℤ)
  (hosc : oscillatory_property seq)
  (hlen : ∀ n, n < 2016 → seq (n+1) = seq n + seq (n+2)) :
  (finset.range 2016).sum seq = 0 :=
sorry

end sum_of_oscillatory_sequence_is_zero_l324_324456


namespace daughterAgeThreeYearsFromNow_l324_324960

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end daughterAgeThreeYearsFromNow_l324_324960


namespace units_digit_sum_factorials_l324_324943

theorem units_digit_sum_factorials : 
  let units_digit (n : Nat) := n % 10 in
  (units_digit (1!) + units_digit (2!) + units_digit (3!) + units_digit (4!) + units_digit (Sum (List.init (500-4) (fun n => (n+5) !)))) % 10 = 3 :=
by
  let units_digit (n : Nat) := n % 10
  sorry

end units_digit_sum_factorials_l324_324943


namespace find_tangent_line_equation_l324_324753

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
def tangent_point (x0 : ℝ) : Prop := ∃ l : ℝ, f' x0 = l ∧ l = 1 + Real.log x0
def is_tangent (l : ℝ → ℝ) : Prop :=
  ∃ x0, l = λ x, 1 * (x - x0) + f x0 ∧ l 0 = -1

theorem find_tangent_line_equation : 
  ∃ l, is_tangent l ∧ l = λ x, x - 1 :=
sorry

end find_tangent_line_equation_l324_324753


namespace golden_ratio_expression_l324_324218

variable (w l : ℝ) (R : ℝ)

-- Conditions
def ratio_condition : Prop :=
  w / l = l / (w + l)

def R_definition : Prop :=
  R = w / l

-- Theorem statement
theorem golden_ratio_expression :
  ratio_condition w l →
  R_definition w l R →
  R = (sqrt 5 - 1) / 2 →
  R^(R^(R^2 + R⁻¹) + R⁻¹) + R⁻¹ = 2 :=
by
  intros
  sorry

end golden_ratio_expression_l324_324218


namespace roxy_total_plants_remaining_l324_324866

def initial_flowering_plants : Nat := 7
def initial_fruiting_plants : Nat := 2 * initial_flowering_plants
def flowering_plants_bought : Nat := 3
def fruiting_plants_bought : Nat := 2
def flowering_plants_given_away : Nat := 1
def fruiting_plants_given_away : Nat := 4

def total_remaining_plants : Nat :=
  let flowering_plants_now := initial_flowering_plants + flowering_plants_bought - flowering_plants_given_away
  let fruiting_plants_now := initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given_away
  flowering_plants_now + fruiting_plants_now

theorem roxy_total_plants_remaining
  : total_remaining_plants = 21 := by
  sorry

end roxy_total_plants_remaining_l324_324866


namespace solve_equation_l324_324452

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ -1) :
  -x^2 = (4 * x + 2) / (x^2 + 3 * x + 2) ↔ x = -1 :=
by
  sorry

end solve_equation_l324_324452


namespace pizza_combination_count_l324_324618

open_locale nat

-- Given condition: the number of different toppings
def num_toppings : ℕ := 8

-- Calculating the total number of one-topping, two-topping, and three-topping pizzas
theorem pizza_combination_count : ((nat.choose num_toppings 1) + (nat.choose num_toppings 2) + (nat.choose num_toppings 3)) = 92 :=
by
  sorry

end pizza_combination_count_l324_324618


namespace polynomial_divisibility_l324_324718

theorem polynomial_divisibility (n : ℕ) (hn : n ≥ 1) : 
  (1 + ∑ i in finset.range n, x^(2 * i)) ∣ (1 + ∑ i in finset.range n, x^i) ↔ odd n := by
  sorry

end polynomial_divisibility_l324_324718


namespace workload_increase_l324_324055

theorem workload_increase (a b c d p : ℕ) (h : p ≠ 0) :
  let total_workload := a + b + c + d
  let workload_per_worker := total_workload / p
  let absent_workers := p / 4
  let remaining_workers := p - absent_workers
  let workload_per_remaining_worker := total_workload / (3 * p / 4)
  workload_per_remaining_worker = (a + b + c + d) * 4 / (3 * p) :=
by
  sorry

end workload_increase_l324_324055


namespace angle_A_is_60_degrees_l324_324356

theorem angle_A_is_60_degrees
  (a b c : ℝ)
  (h : b^2 + c^2 - a^2 = b * c) :
  ∠A = 60 :=
begin
  sorry
end

end angle_A_is_60_degrees_l324_324356


namespace ellipse_properties_l324_324987

theorem ellipse_properties :
  ∃ a b : ℝ, 0 < b ∧ b < a ∧ (a = 2) ∧ (b = 1) ∧
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1) ∧
  ∃ (k : ℝ) (A M N : ℝ × ℝ), k > 0 ∧
  let A := (0, 1) in
  let M :=⟨(-8 * k) / (1 + 4 * k^2), (1 - 4 * k^2) / (1 + 4 * k^2)⟩ in
  let N := ⟨(8 * k) / (k^2 + 4), (k^2 - 4) / (k^2 + 4)⟩ in
  ∀ (Q : ℝ × ℝ), Q = (0, -5 / 3) → collinear {M, N, Q} :=
begin
  sorry
end

end ellipse_properties_l324_324987


namespace initial_price_of_phone_l324_324658

theorem initial_price_of_phone (P : ℝ) (h : 0.20 * P = 480) : P = 2400 :=
sorry

end initial_price_of_phone_l324_324658


namespace extra_men_needed_l324_324586

theorem extra_men_needed (length_road : ℝ) (total_days : ℕ) (initial_workers : ℕ) (completed_road : ℝ) (days_spent : ℕ) (remaining_days : ℕ) : 
  length_road = 10 ∧ total_days = 60 ∧ initial_workers = 30 ∧ completed_road = 2 ∧ days_spent = 20 ∧ remaining_days = 60 - 20 →
  ∃ n2 : ℕ, n2 = 30 :=
begin
  intros h,
  sorry
end

end extra_men_needed_l324_324586


namespace min_force_required_l324_324573

def V : ℝ := 10 * 10^-6 -- Volume of the cube in cubic meters
def ρ_cube : ℝ := 700 -- Density of the cube's material in kg/m^3
def ρ_water : ℝ := 1000 -- Density of water in kg/m^3
def g : ℝ := 10 -- Acceleration due to gravity in m/s^2

noncomputable def F_g : ℝ := (ρ_cube * V) * g -- Gravitational force
noncomputable def F_b : ℝ := (ρ_water * V) * g -- Buoyant force
noncomputable def F_add : ℝ := F_b - F_g -- Additional force required to submerge

theorem min_force_required : F_add = 0.03 := by
  sorry

end min_force_required_l324_324573


namespace item_prices_l324_324543

noncomputable def total_cost_eq (B P S Sh : ℝ) : Prop :=
  B + P + S + Sh = 205.93

noncomputable def price_conditions (B : ℝ) : Prop :=
  let P := B - 2.93 in
  let S := 1.5 * P in
  let Sh := 3 * S in
  total_cost_eq B P S Sh

theorem item_prices (B P S Sh : ℝ) (h₁ : P = B - 2.93) (h₂ : S = 1.5 * P) (h₃ : Sh = 3 * S) (h₄ : total_cost_eq B P S Sh) :
  B = 28.305 ∧ P = 25.375 ∧ S = 38.0625 ∧ Sh = 114.1875 :=
sorry

end item_prices_l324_324543


namespace num_possible_sets_l324_324344

theorem num_possible_sets (M : Set ℕ) :
  {2, 3} ⊂ M ∧ M ⊂ {1, 2, 3, 4, 5} → (M \ {2, 3} ⊆ {1, 4, 5} ∧ |M| = 6) :=
by
  intros h
  unfold Set
  sorry

end num_possible_sets_l324_324344


namespace dave_mass_per_unit_length_l324_324181

theorem dave_mass_per_unit_length (mass length : ℝ) (h_mass : mass = 26) (h_length : length = 40) :
  mass / length = 0.65 :=
by {
  rw [h_mass, h_length],
  norm_num,
  sorry
}

end dave_mass_per_unit_length_l324_324181


namespace find_f_of_3_l324_324494

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l324_324494


namespace lally_internet_days_l324_324198

-- Definitions based on the conditions
def cost_per_day : ℝ := 0.5
def debt_limit : ℝ := 5
def initial_payment : ℝ := 7
def initial_balance : ℝ := 0

-- Proof problem statement
theorem lally_internet_days : ∀ (d : ℕ), 
  (initial_balance + initial_payment - cost_per_day * d ≤ debt_limit) -> (d = 14) :=
sorry

end lally_internet_days_l324_324198


namespace divisor_problem_l324_324123

/-
Problem Statement:
Suppose a, b, and c are integers such that 4b = 10 - 3a + c. Prove that the number of guaranteed divisors from the set {1, 2, 3, 4, 5, 6} of the expression 3b + 15 - c is exactly 1.
-/

theorem divisor_problem (a b c : ℤ) (h : 4 * b = 10 - 3 * a + c) : 
  ∃ n ∈ {1, 2, 3, 4, 5, 6}, n = 1 ∧ ∀ m ∈ {1, 2, 3, 4, 5, 6}, m ≠ n → ¬ (m ∣ (3 * b + 15 - c)) :=
by
  sorry

end divisor_problem_l324_324123


namespace area_of_triangle_is_correct_l324_324135

noncomputable def area_of_triangle (base : ℝ) (median1 : ℝ) (median2 : ℝ) : ℝ :=
  -- Define the centroid distances based on given medians
  let BM := (2 / 3) * median1 in
  let CM := (2 / 3) * median2 in
  -- Calculate the area of the smaller right triangle
  let area_BMC := (1 / 2) * BM * CM in
  -- The area of the original triangle is three times the area of the smaller right triangle
  3 * area_BMC

theorem area_of_triangle_is_correct :
  area_of_triangle 20 18 24 = 288 :=
by
  sorry

end area_of_triangle_is_correct_l324_324135


namespace curves_intersect_and_max_value_l324_324812

def curve_C1 (t : ℝ) : ℝ × ℝ :=
  (2 - real.sqrt 2 * t, -1 + real.sqrt 2 * t)

def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * (real.cos (θ + real.pi / 4))
  (ρ * real.cos θ, ρ * real.sin θ)

theorem curves_intersect_and_max_value :
  (∃ t θ, curve_C1 t = curve_C2 θ) ∧ 
  (∃ θ, 2 * (curve_C2 θ).1 + (curve_C2 θ).2 = (real.sqrt 2) / 2 + real.sqrt 5) :=
by
  sorry

end curves_intersect_and_max_value_l324_324812


namespace area_of_angle_ABC_l324_324364

-- Definitions for the problem conditions
def radius : ℝ := 1
def AB : ℝ := real.sqrt 2
def BC : ℝ := 10 / 7
def angle_BAC : Real.Angle := sorry -- we assume it's given that BAC is acute

-- Required area calculation corresponding to the given problem conditions
theorem area_of_angle_ABC {O A B C : Point}
  (circle_radius : ∀ A, distance O A = radius)
  (chord_AB : distance A B = AB)
  (chord_BC : distance B C = BC)
  (angle_BAC_acute : Acute angle_BAC) :
  let area := (1/2) + (10 * real.sqrt 6 / 49) + (3 * real.pi / 4) - real.arcsin (5 / 7) in
  part_of_circle_inside_angle O A B C = area :=
sorry

end area_of_angle_ABC_l324_324364


namespace range_of_a_l324_324089

theorem range_of_a 
  (a b x1 x2 x3 x4 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a^2 ≠ 0)
  (hx1 : a * x1^2 + b * x1 + 1 = 0) 
  (hx2 : a * x2^2 + b * x2 + 1 = 0) 
  (hx3 : a^2 * x3^2 + b * x3 + 1 = 0) 
  (hx4 : a^2 * x4^2 + b * x4 + 1 = 0)
  (h_order : x3 < x1 ∧ x1 < x2 ∧ x2 < x4) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l324_324089


namespace perfect_square_trinomial_k_l324_324027

theorem perfect_square_trinomial_k :
  ∀ (k : ℤ), (∃ (a b : ℤ), a*x² + b*x + c = (a*x + b)²) ↔ (k = 6 ∨ k = -6) :=
by
  -- proof skipped
  sorry

end perfect_square_trinomial_k_l324_324027


namespace ab_cd_zero_l324_324843

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : ac + bd = 0) : 
  ab + cd = 0 := 
sorry

end ab_cd_zero_l324_324843


namespace Melody_reading_plan_l324_324101

-- Given conditions
def english_pages : ℕ := 60
def math_pages : ℝ := 35.5
def history_pages : ℕ := 48
def chinese_chapters : ℕ := 25
def art_illustrations : ℕ := 22
def audiobook_minutes : ℕ := 150

-- Fractions to be read
def english_fraction : ℚ := 1 / 3
def math_percentage : ℝ := 46.5 / 100
def history_fraction : ℚ := 3 / 8
def chinese_percentage : ℚ := 40 / 100
def art_percentage : ℚ := 27.3 / 100
def audiobook_percentage : ℚ := 75 / 100

-- Desired Outputs
def english_tomorrow_reading : ℕ := 20
def math_tomorrow_reading : ℕ := 17
def history_tomorrow_reading : ℕ := 18
def chinese_tomorrow_reading : ℕ := 10
def art_tomorrow_reading : ℕ := 6
def audiobook_tomorrow_reading : ℕ := 113

-- Lean 4 Statement
theorem Melody_reading_plan :
  english_pages * english_fraction = english_tomorrow_reading ∧
  round (math_pages * math_percentage) = math_tomorrow_reading ∧
  history_pages * history_fraction = history_tomorrow_reading ∧
  (chinese_chapters : ℚ) * chinese_percentage = chinese_tomorrow_reading ∧
  round ((art_illustrations : ℚ) * art_percentage) = art_tomorrow_reading ∧
  round ((audiobook_minutes : ℚ) * audiobook_percentage) = audiobook_tomorrow_reading :=
by
  sorry

end Melody_reading_plan_l324_324101


namespace angle_between_a_b_l324_324766

variables (a b : EuclideanSpace ℝ (Fin 2)) -- assuming 2-dimensional Euclidean space for vectors a and b

-- Given conditions
axiom magnitude_a : ∥a∥ = Real.sqrt 2
axiom magnitude_b : ∥b∥ = 2
axiom orthogonal_condition : (a - b) ⬝ a = 0

-- To prove: The angle θ between vectors a and b is π/4
theorem angle_between_a_b : 
  let θ := Real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)) in θ = Real.pi / 4 :=
by sorry

end angle_between_a_b_l324_324766


namespace sarahs_next_monday_birthday_l324_324449

def is_leap_year (year : ℤ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def next_monday_year (start_year : ℤ) (start_day : Day) : ℤ := 
  let rec day_of_week (year : ℤ) (day : Day) : Day :=
    if year == start_year then day else
      let next_day := if is_leap_year year then (day + 2) % 7 else (day + 1) % 7
      day_of_week (year - 1) next_day
  if day_of_week 2025 start_day == Day.monday then 2025 else sorry

theorem sarahs_next_monday_birthday :
  next_monday_year 2017 Day.friday = 2025 := by 
  sorry

end sarahs_next_monday_birthday_l324_324449


namespace fraction_of_girls_at_dance_l324_324159

theorem fraction_of_girls_at_dance :
  (270 : ℚ) * (4 / 9) + (180 : ℚ) * (5 / 9) = 220 ∧
  450 = 450 →
  (220 : ℚ) / 450 = 22 / 45 := 
by sorry

end fraction_of_girls_at_dance_l324_324159


namespace number_of_rectangles_4x4_grid_l324_324778

theorem number_of_rectangles_4x4_grid : 
  let r := 4 in
  let c := 4 in
  (Nat.choose r 2) * (Nat.choose c 2) = 36 := 
by 
  let r := 4
  let c := 4
  have h1 : Nat.choose r 2 = 6 := Nat.choose_succ_succ_eq r 1
  have h2 : Nat.choose c 2 = 6 := Nat.choose_succ_succ_eq c 1
  calc (Nat.choose r 2) * (Nat.choose c 2)
      = 6 * 6 : by rw [h1, h2]
  ... = 36   : by norm_num

end number_of_rectangles_4x4_grid_l324_324778


namespace sum_first_13_terms_eq_26_l324_324372

open Real

variables {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a n + d

theorem sum_first_13_terms_eq_26
  (h_arith_seq : is_arithmetic_sequence a_n)
  (h_condition : 3 * (a_n 2 + a_n 4) + 2 * (a_n 6 + a_n 9 + a_n 12) = 24) :
  (∑ i in Finset.range 13, a_n i) = 26 :=
sorry

end sum_first_13_terms_eq_26_l324_324372


namespace min_value_expression_l324_324936

theorem min_value_expression (x y : ℝ) :
  ∃ (x_min y_min : ℝ), 
    (x_min = -5 ∧ y_min = 4) ∧
    (∀ (x y : ℝ), x^2 + y^2 + 10*x - 8*y + 34 ≥ -7) :=
by
  use [-5, 4]
  split
  { simp }
  { intro x y
    sorry }

end min_value_expression_l324_324936


namespace problem_l324_324131

noncomputable
def harmonic_mean_condition (x y z : ℝ) : Prop :=
  (1 / x) + (1 / y) + (1 / z) = 0.75

noncomputable
theorem problem (x y z : ℝ) 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 216)
  (h3 : harmonic_mean_condition x y z) :
  x^2 + y^2 + z^2 = 576 := 
by 
  -- Proof goes here
  sorry

end problem_l324_324131


namespace find_function_l324_324724

theorem find_function {f : ℝ → ℝ} (h : ∀ x : ℝ, f (real.sqrt x + 1) = x + 2 * real.sqrt x) :
  ∀ y : ℝ, y ≥ 1 → f y = y^2 - 1 :=
by sorry

end find_function_l324_324724


namespace bulbs_needed_l324_324108

theorem bulbs_needed (M : ℕ) (hM : M = 12) : 
  let large := 2 * M in
  let small := M + 10 in
  let bulbs_medium := 2 * M in
  let bulbs_large := 3 * large in
  let bulbs_small := small in
  bulbs_medium + bulbs_large + bulbs_small = 118 :=
by
  sorry

end bulbs_needed_l324_324108


namespace find_incorrect_value_of_observation_l324_324524

noncomputable def incorrect_observation_value (mean1 : ℝ) (mean2 : ℝ) (n : ℕ) : ℝ :=
  let old_sum := mean1 * n
  let new_sum := mean2 * n
  let correct_value := 45
  let incorrect_value := (old_sum - new_sum + correct_value)
  (incorrect_value / -1)

theorem find_incorrect_value_of_observation :
  incorrect_observation_value 36 36.5 50 = 20 :=
by
  -- By the problem setup, incorrect_observation_value 36 36.5 50 is as defined in the proof steps.
  -- As per the proof steps and calculation, incorrect_observation_value 36 36.5 50 should evaluate to 20.
  sorry

end find_incorrect_value_of_observation_l324_324524


namespace find_f_three_l324_324484

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l324_324484


namespace solve_eq1_solve_eq2_l324_324453

-- Proof problem 1: Prove that under the condition 6x - 4 = 3x + 2, x = 2
theorem solve_eq1 : ∀ x : ℝ, 6 * x - 4 = 3 * x + 2 → x = 2 :=
by
  intro x
  intro h
  sorry

-- Proof problem 2: Prove that under the condition (x / 4) - (3 / 5) = (x + 1) / 2, x = -22/5
theorem solve_eq2 : ∀ x : ℝ, (x / 4) - (3 / 5) = (x + 1) / 2 → x = -(22 / 5) :=
by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l324_324453


namespace hyperbola_eccentricity_l324_324755

theorem hyperbola_eccentricity (m : ℝ) (h : m > 0) 
(hyperbola_eq : ∀ (x y : ℝ), x^2 / 9 - y^2 / m = 1) 
(eccentricity : ∀ (e : ℝ), e = 2) 
: m = 27 :=
sorry

end hyperbola_eccentricity_l324_324755


namespace y_increase_by_8_implication_l324_324103

theorem y_increase_by_8_implication :
  (∀ {x y : ℝ}, x = 2 → y = 5 → (∃ ratio : ℝ, ratio = 5 / 2 → y = ratio * x)) →
  (∃ Δy : ℝ, Δy = 20 → ∃ Δx : ℝ, Δx = 8 → Δy = 5 / 2 * Δx) :=
begin
  intros h,
  use 20,
  split,
  { refl, },
  intros Δx h₁,
  rw h₁,
  have : (5 : ℝ) / 2 * 8 = 20,
  { norm_num, },
  rw this,
  use 8,
  split,
  { refl, },
  { exact this, }
end

end y_increase_by_8_implication_l324_324103


namespace product_of_possible_b_values_l324_324522

theorem product_of_possible_b_values : 
  ∀ b : ℝ, 
    (abs (b - 2) = 2 * (4 - 1)) → 
    (b = 8 ∨ b = -4) → 
    (8 * (-4) = -32) := by
  sorry

end product_of_possible_b_values_l324_324522


namespace trapezoid_areas_l324_324369

open Real

theorem trapezoid_areas :
  ∀ (ABCD : Type) [Trapezoid ABCD] (A B C D O M N: Point)
  (h_base_lengths : (distance A D) = 84 ∧ (distance B C) = 42)
  (h_side_lengths : (distance A B) = 39 ∧ (distance C D) = 45)
  (h_diagonals_intersect : O = intersection (diagonal A C) (diagonal B D))
  (h_parallel_through_O : is_parallel (line_through O M) (line_through D A))
  (h_parallel_bases : is_parallel (line_through A D) (line_through B C)),
  area_trapezoid M B C N = 588 ∧ area_trapezoid A M N D = 1680 := sorry

end trapezoid_areas_l324_324369


namespace sum_of_20_consecutive_integers_from_neg_9_l324_324576

theorem sum_of_20_consecutive_integers_from_neg_9 :
  (List.range 20).map (λ i, -9 + i).sum = 10 := by
  -- The proof will go here
  sorry

end sum_of_20_consecutive_integers_from_neg_9_l324_324576


namespace total_pupils_l324_324207

theorem total_pupils (girls boys : ℕ) (h_girls : girls = 542) (h_boys : boys = 387) : girls + boys = 929 := 
by
  rw [h_girls, h_boys]
  exact Nat.add_comm 542 387
  sorry

end total_pupils_l324_324207


namespace marbles_each_friend_gets_l324_324771

-- Definitions for the conditions
def total_marbles : ℕ := 100
def marbles_kept : ℕ := 20
def number_of_friends : ℕ := 5

-- The math proof problem
theorem marbles_each_friend_gets :
  (total_marbles - marbles_kept) / number_of_friends = 16 :=
by
  -- We include the proof steps within a by notation but stop at sorry for automated completion skipping proof steps
  sorry

end marbles_each_friend_gets_l324_324771


namespace subtract_one_div_by_five_l324_324947

theorem subtract_one_div_by_five : ∃ k : ℕ, 5026 - 1 = 5 * k :=
by
sory

end subtract_one_div_by_five_l324_324947


namespace smallest_possible_last_digit_l324_324473

-- Statement of the problem in Lean
theorem smallest_possible_last_digit :
  ∃ (s : List ℕ), 
    (List.length s = 1001) ∧
    (List.head s = some 2) ∧
    (∀ (i : ℕ), (i < 1000 → let n := (s.nth i).getD 0 * 10 + (s.nth (i + 1)).getD 0 in n % 17 = 0 ∨ n % 29 = 0)) ∧
    (List.last s = some 1) :=
sorry

end smallest_possible_last_digit_l324_324473


namespace subset_neg1_of_leq3_l324_324023

theorem subset_neg1_of_leq3 :
  {x | x = -1} ⊆ {x | x ≤ 3} :=
sorry

end subset_neg1_of_leq3_l324_324023


namespace probability_sum_of_three_dice_is_10_l324_324553

theorem probability_sum_of_three_dice_is_10 :
  (∃ (dice_rolls : finset (list ℕ)), 
    dice_rolls = {l | l.perm = [[1, 3, 6], [1, 4, 5], [2, 2, 6], [2, 3, 5], [2, 4, 4], [3, 3, 4]]} ∧
    ∑ roll in dice_rolls, roll.sum = 10 ∧
    ∑ roll in dice_rolls, 1 = 27) →
  let total_outcomes := 6 * 6 * 6 in
  (27 / total_outcomes) = (1 / 8) :=
begin
  sorry
end

end probability_sum_of_three_dice_is_10_l324_324553


namespace wall_width_l324_324519

-- Definitions for the problem
def height (W : ℝ) := 6 * W
def length (H : ℝ) := 7 * H
def volume (W H L : ℝ) := W * H * L

-- The proof problem statement
theorem wall_width (W : ℝ) (H := height W) (L := length H) 
  (V : ℝ := 86436) (hV : volume W H L = V) : W = 7 :=
by
  sorry

end wall_width_l324_324519


namespace problem_a_problem_b_l324_324072

-- Define the sequence as a real sequence
def real_sequence (a : ℕ → ℝ) : Prop := true

-- Define the series convergence condition
def series_converges (a : ℕ → ℝ) : Prop := 
  ∃ l : ℝ, has_sum a l

-- Problem (a): Prove that given a convergent series, the rearranged series also converges
theorem problem_a (a : ℕ → ℝ) 
  (h_converges : series_converges a) :
  series_converges (λ n, if n < 2 then a n else if n < 4 then a (3 - n + 2 * (n / 4)) else sorry) :=
sorry

-- Problem (b): Prove that given a convergent series, the rearranged series does not necessarily converge
theorem problem_b (a : ℕ → ℝ) 
  (h_converges : series_converges a) :
  ¬ series_converges (λ n, if n < 2 then a n else if n < 4 then a n else sorry) :=
sorry

end problem_a_problem_b_l324_324072


namespace tim_total_points_l324_324642

theorem tim_total_points :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6
  let tetrises := 4
  let total_points := singles * single_points + tetrises * tetris_points
  total_points = 38000 :=
by
  sorry

end tim_total_points_l324_324642


namespace cubic_no_roots_in_Qn_l324_324689

def Q : ℕ → Set ℚ := sorry

noncomputable def Q0 := (setOf rational)

noncomputable def Qn (n : ℕ) : Set ℚ := 
  { x | ∃ (p q r : ℚ) (H1 : p ∈ Q(n-1)) (H2 : q ∈ Q(n-1)) (H3 : r ∈ Q(n-1)), x = p + q * sqrt r ∧ sqrt r ∉ Q(n-1) }

theorem cubic_no_roots_in_Qn (a b c : ℚ) :
  (∀ (x : ℚ), x^3 + a * x^2 + b * x + c ≠ 0) → ∀ n, ¬ ∃ x ∈ Q n, x^3 + a * x^2 + b * x + c = 0 :=
sorry

end cubic_no_roots_in_Qn_l324_324689


namespace unscreened_percentage_l324_324195

/-- Given:
1. The dimensions of the TV are 6 by 5.
2. The dimensions of the screen are 5 by 4.

Proof:
The percentage of the unscreened part of the TV is approximately 33.33%.
-/
theorem unscreened_percentage {length_TV width_TV length_screen width_screen : ℕ}
  (h1 : length_TV = 6) (h2 : width_TV = 5) (h3 : length_screen = 5) (h4 : width_screen = 4) :
  ((length_TV * width_TV - length_screen * width_screen : ℚ) / (length_TV * width_TV) * 100 : ℚ) ≈ 33.33 := 
by
  rw [h1, h2, h3, h4]
  have h_area_TV : length_TV * width_TV = 30 := rfl
  have h_area_screen : length_screen * width_screen = 20 := rfl
  have h_area_unscreened : length_TV * width_TV - length_screen * width_screen = 10 := rfl
  have h_percentage_unscreened : ((length_TV * width_TV - length_screen * width_screen : ℚ) / (length_TV * width_TV) * 100 : ℚ) = 33.33 :=
    by norm_num
  exact h_percentage_unscreened


end unscreened_percentage_l324_324195


namespace translation_equivalence_l324_324167

noncomputable def initial_function (x : ℝ) : ℝ := sin (x + π / 3)

noncomputable def translated_function (x : ℝ) : ℝ := sin (x + π / 3 - π / 6)

theorem translation_equivalence :
  ∀ x : ℝ, translated_function x = sin (x + π / 6) := 
by
  -- Proof steps would go here
  sorry

end translation_equivalence_l324_324167


namespace count_primes_in_sequence_l324_324012

-- Define the list as a sequence of numbers starting with 47, and each subsequent 
-- number is 47 concatenated to the previous ones forming a sequence of the form 47*(10^n + 10^(n-2) + ... + 10^0).
def sequence (n : ℕ) : ℕ :=
  47 * (list.range ((n + 1) * 2)).map (λ m, if m % 2 = 0 then 10^(m/2) else 0).sum

theorem count_primes_in_sequence : ∃! n, n = 1 ∧ prime (sequence 0)
  ∧ ∀ k > 0, ¬ prime (sequence k) :=
by
  sorry

end count_primes_in_sequence_l324_324012


namespace expression_of_a_n_find_a0_l324_324636

noncomputable def a_sequence (a0 : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a0 else (λ m, 2^(m-1) - 3 * a_sequence a0 (m - 1)) n

theorem expression_of_a_n (a0 : ℝ) (n : ℕ) :
  a_sequence a0 n = (1/5) * (2^n + (-1:ℝ)^(n-1) * 3^n) + (-1:ℝ)^n * 3^n * a0 :=
sorry

theorem find_a0 (a0 : ℝ) :
  (∀ n : ℕ, n > 0 → a_sequence a0 (n + 1) > a_sequence a0 n) → a0 = (1/5) :=
sorry

end expression_of_a_n_find_a0_l324_324636


namespace triangle_area_l324_324567

theorem triangle_area (a b c : ℝ) (ha : a = 5) (hb : b = 6) (hc : c = 7) (a_pos: a > 0) (b_pos: b > 0) (c_pos: c > 0) (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  let s := (a + b + c) / 2
  in sqrt (s * (s - a) * (s - b) * (s - c)) = 6 * sqrt 6 :=
by
  sorry

end triangle_area_l324_324567


namespace find_square_side_length_l324_324668

theorem find_square_side_length
  (a CF AE : ℝ)
  (h_CF : CF = 2 * a)
  (h_AE : AE = 3.5 * a)
  (h_sum : CF + AE = 91) :
  a = 26 := by
  sorry

end find_square_side_length_l324_324668


namespace g_strictly_decreasing_l324_324075

noncomputable def g : ℝ → ℝ := sorry
axiom g_diff : differentiable_on ℝ g (Set.Ioi 0)
axiom g_cont_diff : continuous_on (deriv g) (Set.Ioi 0)
axiom g_self_inv : ∀ x > 0, g (g x) = x
axiom g_not_identity : ∃ x > 0, g x ≠ x

theorem g_strictly_decreasing : ∀ x y > 0, x < y → g y < g x := 
by
  sorry

end g_strictly_decreasing_l324_324075


namespace range_of_a_l324_324355

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Icc 0 1 ∧ 2^x * (3 * x + a) < 1) → a < 1 :=
by
  intros h
  sorry

end range_of_a_l324_324355


namespace difference_between_two_numbers_l324_324542

theorem difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) 
  (h3 : x^2 - y^2 = 200) : 
  x - y = 10 :=
by 
  sorry

end difference_between_two_numbers_l324_324542


namespace angle_same_terminal_side_l324_324459

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 290 = k * 360 - 70 :=
by
  sorry

end angle_same_terminal_side_l324_324459


namespace point_between_lines_implies_b_value_l324_324353

variable (b : ℤ)

def line1 (x y : ℝ) := 6 * x - 8 * y + 1
def line2 (x y : ℝ) := 3 * x - 4 * y + 5
def parallel_condition (x y : ℝ) := (line1 x y = 0) ∨ (line2 x y = 0)
def point_between_lines : Prop := ∀ (x y : ℝ), parallel_condition x y → 
  line2 x y < 4 * b - 15 + line1 x y ∧ 4 * b - 15 < 4 * line1 x y

theorem point_between_lines_implies_b_value :
  point_between_lines (5 : ℝ) (b : ℝ) →
  (b = 4) :=
sorry

end point_between_lines_implies_b_value_l324_324353


namespace pizza_toppings_count_l324_324623

theorem pizza_toppings_count :
  let toppings := 8 in
  let one_topping_pizzas := toppings in
  let two_topping_pizzas := (toppings.choose 2) in
  let three_topping_pizzas := (toppings.choose 3) in
  one_topping_pizzas + two_topping_pizzas + three_topping_pizzas = 92 :=
by sorry

end pizza_toppings_count_l324_324623


namespace stools_count_l324_324551

theorem stools_count : ∃ x y : ℕ, 3 * x + 4 * y = 39 ∧ x = 3 := 
by
  sorry

end stools_count_l324_324551


namespace daughter_age_in_3_years_l324_324958

variable (mother_age_now : ℕ) (gap_years : ℕ) (ratio : ℕ)

theorem daughter_age_in_3_years
  (h1 : mother_age_now = 41) 
  (h2 : gap_years = 5)
  (h3 : ratio = 2) :
  let mother_age_then := mother_age_now - gap_years in
  let daughter_age_then := mother_age_then / ratio in
  let daughter_age_now := daughter_age_then + gap_years in
  let daughter_age_in_3_years := daughter_age_now + 3 in
  daughter_age_in_3_years = 26 :=
  by
    sorry

end daughter_age_in_3_years_l324_324958


namespace count_prime_numbers_in_list_only_prime_in_list_is_47_number_of_primes_in_list_l324_324015

theorem count_prime_numbers_in_list : 
  ∀ (n : ℕ), (∃ k : ℕ, n = 47 * ((10^k - 1) / 9)) → n ≠ 47 =→ n % 47 = 0 → isPrime n → False :=
by
  assume n hn h_eq_prime_47 h_divisible it_is_prime
  sorry

theorem only_prime_in_list_is_47 : ∀ (n : ℕ), n ∈ { num | ∃ k : ℕ, num = 47 * ((10^k - 1) / 9) } → (isPrime n ↔ n = 47) := 
by
  assume n hn
  split
    assume it_is_prime
    by_cases h_case : n = 47
      case inl => exact h_case
      case inr =>
        obtain ⟨k, hk⟩ := hn
        have h_mod : n % 47 = 0 := by rw [hk, nat.mul_mod_right]
        apply_false_from (count_prime_numbers_in_list n) hk h_case h_mod it_is_prime
        contradiction
    assume it_is_47
    exact by norm_num [it_is_47]
    sorry

theorem number_of_primes_in_list : ∀ (l : List ℕ), (∀ (n : ℕ), n ∈ l → ∃ k : ℕ, n = 47 * ((10^k - 1) / 9)) → l.filter isPrime = [47] :=
by
  sorry

end count_prime_numbers_in_list_only_prime_in_list_is_47_number_of_primes_in_list_l324_324015


namespace find_f_3_l324_324502

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l324_324502


namespace find_f_of_3_l324_324515

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l324_324515


namespace necessary_folds_to_verify_square_l324_324774

-- Defining the properties of a square
def is_square {α : Type} [metric_space α] (quadrilateral : α → Prop) : Prop :=
  ∃ a b c d : α, quadrilateral a ∧ quadrilateral b ∧ quadrilateral c ∧ quadrilateral d ∧
  dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a ∧
  angle a b c = 90 ∧ angle b c d = 90 ∧ angle c d a = 90 ∧ angle d a b = 90

-- Proving the number of folds required
theorem necessary_folds_to_verify_square (quadrilateral : Type) [metric_space quadrilateral] (q : quadrilateral → Prop) :
  (is_square q) → (∃ n, n ≥ 3 ∧ required_folds q n) :=
sorry

end necessary_folds_to_verify_square_l324_324774


namespace quadratic_reciprocity_law_l324_324205

theorem quadratic_reciprocity_law 
  (p q : ℕ) (h1 : p.Prime) (h2 : q.Prime) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) : 
  (legendre_sym q p) * (legendre_sym p q) = (-1) ^ ((p - 1) * (q - 1) / 4) :=
by
  sorry

end quadratic_reciprocity_law_l324_324205


namespace surface_area_of_circumscribed_sphere_l324_324315

theorem surface_area_of_circumscribed_sphere 
  (a b c s : ℝ) 
  (h_perpendicular : a ⊥ b ∧ b ⊥ c ∧ c ⊥ a) 
  (h_length : a = 1 ∧ b = 1 ∧ c = 1)
  :
  let R := (Real.sqrt 3) / 2 in
  4 * Real.pi * R^2 = 3 * Real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_l324_324315


namespace most_cost_effective_l324_324654

def fare (d : ℕ) : ℕ :=
if d ≤ 3 then 12
else if d ≤ 7 then 12 + (d - 3) * 26 / 10
else 12 + 4 * 26 / 10 + (d - 7) * 35 / 10

noncomputable def cost_option1 : ℕ :=
2 * fare 15

noncomputable def cost_option2 : ℕ :=
3 * fare 10

noncomputable def cost_option3 (seg1 seg2 seg3 : ℕ) : ℕ :=
if seg1 > 3 ∧ seg2 > 3 ∧ seg3 > 3 ∧ seg1 + seg2 + seg3 = 30 then 
  fare seg1 + fare seg2 + fare seg3
else 
  0

theorem most_cost_effective (h1 : cost_option1 = 10080 / 100) 
                            (h2 : cost_option2 = 9870 / 100) 
                            (h3 : ∀ seg1 seg2 seg3, cost_option3 seg1 seg2 seg3 ≥ (10290 / 100)) :
  cost_option2 = 9870 / 100 :=
begin
  sorry
end

end most_cost_effective_l324_324654


namespace unique_solution_l324_324259

theorem unique_solution (n : ℕ) (h1 : n > 0) (h2 : n^2 ∣ 3^n + 1) : n = 1 :=
sorry

end unique_solution_l324_324259


namespace min_value_of_fraction_l324_324345

open Real

theorem min_value_of_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) : 
  ∃ c, c = 4 * sqrt 3 + 7 ∧ (∀ x y : ℝ, (0 < x) → (0 < y) → (x + 2 * y = 1) → ( (x + y + 1) / (x * y) ≥ c )) :=
by
  use 4 * sqrt 3 + 7
  intro x y hx hy hxy
  have : (x + y + 1) / (x * y) ≥ 4 * sqrt 3 + 7 := sorry
  exact this

end min_value_of_fraction_l324_324345


namespace ratio_PM_MQ_1_1_l324_324820

-- Define the square and the given points
structure Square (side : ℝ) :=
(A B C D : ℝ × ℝ)
(E M P Q : ℝ × ℝ)
(side_length : side)
(is_square : ∀(A B C D E M P Q), side_length = 10 ∧ 
             A = (0, 10) ∧ B = (10, 10) ∧ C = (10, 0) ∧ D = (0,0) ∧ 
             E = (6, 0) ∧ 
             M = ((0+6)/2, (10+0)/2) ∧ 
             (∀ x y, y - 5 = (3/5) * (x - 3)) [Perpendicular Bisector] ∧
             P = (10, 10) ∧ Q = (-20/3, 0))

-- Define the lengths PM and MQ
def length (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def square_PQ := {s : Square // 
   length s.P s.M = length s.M s.Q}

-- Prove the ratio PM:MQ is 1:1
theorem ratio_PM_MQ_1_1 (s : Square) (h : square_PQ) : 
  length s.P s.M = length s.M s.Q :=
sorry -- This would here be the place where the actual proof is constructed.


end ratio_PM_MQ_1_1_l324_324820


namespace ben_weekly_eggs_l324_324851

-- Definitions for the conditions
def weekly_saly_eggs : ℕ := 10
def weekly_ben_eggs (B : ℕ) : ℕ := B
def weekly_ked_eggs (B : ℕ) : ℕ := B / 2

def weekly_production (B : ℕ) : ℕ :=
  weekly_saly_eggs + weekly_ben_eggs B + weekly_ked_eggs B

def monthly_production (B : ℕ) : ℕ := 4 * weekly_production B

-- Theorem for the proof
theorem ben_weekly_eggs (B : ℕ) (h : monthly_production B = 124) : B = 14 :=
sorry

end ben_weekly_eggs_l324_324851


namespace most_suitable_for_sampling_survey_l324_324186

-- Definitions of the conditions
def question := "Which of the following surveys is most suitable for a sampling survey?"

def optionA := "Security check of passengers before boarding a plane"
def optionB := "School recruiting teachers, corresponding to the interview of applicants"
def optionC := "Survey of the quality of a batch of masks"
def optionD := "Survey of the vision of students in Class 7-1 of a certain school"

-- The problem as a proposition to prove
theorem most_suitable_for_sampling_survey : (optionA -> False) ∧ (optionB -> False) ∧ (optionD -> False) -> optionC = "Survey of the quality of a batch of masks" :=
by
  sorry

end most_suitable_for_sampling_survey_l324_324186


namespace F_f_log2_13_eq_neg_5_6_l324_324757

def F : ℝ → ℝ :=
  λ x, if x > 0 then (1/2)^x - 4/3 else -(1/2)^(-x) + 4/3

def f : ℝ → ℝ :=
  λ x, -(1/2)^(-x) + 4/3

theorem F_f_log2_13_eq_neg_5_6 : F (f (log 2 (1/3))) = -5/6 := by
  sorry

end F_f_log2_13_eq_neg_5_6_l324_324757


namespace max_PA_PB_product_l324_324088

noncomputable def distance (p1 p2: ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem max_PA_PB_product :
  ∀ (P: ℝ × ℝ) (m: ℝ), 
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 3) 
  (x + m * y + 2 = 0 → ∃ y, y = P.2) ∧ (m * x - y - 2 * m + 3 = 0 → ∃ x, x = P.1) →
  (distance P A * distance P B ≤ 25 / 2) := sorry

end max_PA_PB_product_l324_324088


namespace pizza_topping_count_l324_324628

theorem pizza_topping_count (n : ℕ) (h : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by {
  rw h,
  simp [nat.choose],
  sorry,
}

end pizza_topping_count_l324_324628


namespace find_f_of_3_l324_324510

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l324_324510


namespace range_of_f_l324_324327

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 2^x - 1 else 1 + Real.log x / Real.log 2

theorem range_of_f : set.range f = {y | -1 < y} :=
by
  sorry

end range_of_f_l324_324327


namespace min_tan_sum_in_acute_triangle_l324_324053

theorem min_tan_sum_in_acute_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A) (h2 : A < π / 2) 
  (h3 : 0 < B) (h4 : B < π / 2) 
  (h5 : 0 < C) (h6 : C < π / 2)
  (h7 : A + B + C = π)
  (h8 : a = (sin A) * k) (h9 : b = (sin B) * k) (h10 : c = (sin C) * k) 
  (h11 : 2 * a ^ 2 = 2 * b ^ 2 + c ^ 2) : 
  (tan A + tan B + tan C) = 6 := 
sorry

end min_tan_sum_in_acute_triangle_l324_324053


namespace min_f_value_l324_324408

open Real

theorem min_f_value (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) :
    ∃ (x : ℝ), (∀ y : ℝ, (|y - a| + |y - b| + |y - c| + |y - d| + |y - e|) ≥ -a - b + d + e) ∧ 
    (|x - a| + |x - b| + |x - c| + |x - d| + |x - e| = -a - b + d + e) :=
sorry

end min_f_value_l324_324408


namespace no_real_roots_of_quadratic_l324_324156

theorem no_real_roots_of_quadratic (a b c : ℝ) (ha : a = 1) (hb : b = -1) (hc : c = 2) :
  b^2 - 4 * a * c < 0 :=
by {
  -- Given the specific values of a, b, and c, we substitute them directly:
  rw [ha, hb, hc],
  -- Simplifying the expression for the discriminant.
  simp,
  -- Showing explicitly that the discriminant is negative.
  norm_num,
  exact dec_trivial,
}

end no_real_roots_of_quadratic_l324_324156


namespace smallest_number_is_neg1_l324_324236

-- Defining the list of numbers
def numbers := [0, -1, 1, 2]

-- Theorem statement to prove that the smallest number in the list is -1
theorem smallest_number_is_neg1 :
  ∀ x ∈ numbers, x ≥ -1 := 
sorry

end smallest_number_is_neg1_l324_324236


namespace hotel_guest_movements_l324_324988

def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

def number_of_movements (n : ℕ) : ℕ :=
(fibonacci (n + 1))^2

theorem hotel_guest_movements : number_of_movements 8 = 3025 :=
by
  simp [number_of_movements, fibonacci],
  -- Fibonacci numbers up to F_9 :=
  have F_9 := fibonacci 9,
  have F_9_value : fibonacci 9 = 55,
  { 
    -- By computation or known results:
    -- F_2 = 2
    -- F_3 = 3
    -- F_4 = 5
    -- F_5 = 8
    -- F_6 = 13
    -- F_7 = 21
    -- F_8 = 34
    -- F_9 = 55
    sorry
  },
  simp [F_9, F_9_value],
  norm_num, -- 55^2 = 3025
  exact F_9_value²

end hotel_guest_movements_l324_324988


namespace mode_of_scores_is_75_l324_324537

def scores : List ℕ :=
  [45, 45, 45,
   52, 56, 56,
   61, 63, 63, 63, 63,
   72, 74, 75, 75, 75, 75, 75,
   80, 83, 86,
   91, 91, 94, 97]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ(highest : (ℕ × ℕ)) (n : ℕ),
    let count := l.count n
    if count > highest.2 then (n, count) else highest)
    (0, 0)

theorem mode_of_scores_is_75 : mode scores = 75 := 
  sorry

end mode_of_scores_is_75_l324_324537


namespace treaty_signed_on_friday_l324_324903

/-- The revolutionary documentation was signed on Monday, January 7, 1765. 
  The treaty was signed 1024 days later, on February 5, 1768.
  Prove that the treaty was signed on a Friday. -/
theorem treaty_signed_on_friday
  (documentation_day : Nat) -- Monday as start day
  (days_later : Nat) -- 1024 days later
  (start_date : Nat) -- January 7, 1765 as start date
  (end_date : Nat) -- February 5, 1768 as end date
  (documentation_day = 1) -- 1 represents Monday 
  (days_later = 1024) 
  (start_date = 17650107) 
  (end_date = 17680205) 
  :
  (documentation_day + (days_later % 7)) % 7 = 5 :=
by
  sorry -- Proof is omitted.

end treaty_signed_on_friday_l324_324903


namespace right_angle_triangle_l324_324413

noncomputable def is_right_triangle {α : Type*} [TopologicalSpace α] [AddGroup α] [MetricSpace α] 
  (P T Q : α) : Prop :=
  ∀ (A B C : α), A = P → B = T → C = Q → ∃ t : ℝ, angle A B C = 90

theorem right_angle_triangle 
  {α : Type*} [TopologicalSpace α] [AddGroup α] [MetricSpace α]
  (Γ₁ Γ₂ : set α) (T P Q : α) 
  (tangent_T : tangent_to Γ₁ Γ₂ T)
  (common_tangent : common_tangent_to Γ₁ Γ₂ P Q)
  : is_right_triangle P T Q :=
  sorry

end right_angle_triangle_l324_324413


namespace scores_fraction_difference_l324_324165

theorem scores_fraction_difference (y : ℕ) (white_ratio : ℕ) (black_ratio : ℕ) (total : ℕ) 
(h1 : white_ratio = 7) (h2 : black_ratio = 6) (h3 : total = 78) 
(h4 : y = white_ratio + black_ratio) : 
  ((white_ratio * total / y) - (black_ratio * total / y)) / total = 1 / 13 :=
by
 sorry

end scores_fraction_difference_l324_324165


namespace units_digit_sum_factorials_500_l324_324945

-- Define the unit digit computation function
def unit_digit (n : ℕ) : ℕ := n % 10

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * fact n

-- Define the sum of factorials from 1 to n
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => fact i)

-- Define the problem statement
theorem units_digit_sum_factorials_500 : unit_digit (sum_factorials 500) = 3 :=
sorry

end units_digit_sum_factorials_500_l324_324945


namespace chess_games_won_l324_324210

theorem chess_games_won (W L : ℕ) (h1 : W + L = 44) (h2 : 4 * L = 7 * W) : W = 16 :=
by
  sorry

end chess_games_won_l324_324210


namespace new_person_weight_l324_324134

def average_weight_increase (old_weight : ℝ) (new_weight : ℝ) (num_persons : ℝ) : ℝ :=
  (new_weight - old_weight) / num_persons

theorem new_person_weight :
  ∀ (old_weight : ℝ), ∀ (num_persons : ℝ), ∀ (increase : ℝ), 
  old_weight = 55 → num_persons = 8 → increase = 2.5 →
  new_weight = old_weight + (num_persons * increase) :=
by
  intros
  sorry

end new_person_weight_l324_324134


namespace part_one_part_two_l324_324097

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- (1) If for any \( x \in [0, 3] \), it holds that \( f(x) \geq 0 \), find the range of values for \( m \).
theorem part_one (m : ℝ) (h : ∀ x ∈ set.Icc (0:ℝ) 3, f x m ≥ 0) : m ≥ 1 := 
sorry

-- (2) If there exists \( x \in [0, 3] \) such that \( f(x) \geq 0 \), find the range of values for \( m \).
theorem part_two (m : ℝ) (h : ∃ x ∈ set.Icc (0:ℝ) 3, f x m ≥ 0) : m ≥ -3 :=
sorry

end part_one_part_two_l324_324097


namespace exists_irreducible_poly_l324_324838

theorem exists_irreducible_poly
  (p : ℕ) (hp : Nat.Prime p)
  (N : Set ℕ)
  (A : Fin p → Set ℕ)
  (hN : {1..} = ⋃ i, A i)
  (hA_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j)):
  ∃ i, ∃ f : ℕ[X], Irreducible f ∧ f.degree = p - 1 ∧ 
    ∀ (a : ℕ), a ∈ f.coeffs → a ∈ A i ∧ ∀ (a₁ a₂ : ℕ), a₁ ≠ a₂ → a₁ ∈ f.coeffs → a₂ ∈ f.coeffs → a₁ ≠ a₂ := by
    sorry

end exists_irreducible_poly_l324_324838


namespace symmetric_point_correct_minimum_value_correct_l324_324307

open Real

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -3, y := 5 }
def B : Point := { x := 2, y := 15 }

def line (P : Point) : Prop := P.x - P.y + 1 = 0

noncomputable def distance (P Q : Point) : ℝ := sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

noncomputable def symmetric_point (A : Point) : Point :=
  let x0 := 4
  let y0 := -2
  { x := x0, y := y0 }

def minimum_value (P : Point) : ℝ := distance P A + distance P B

theorem symmetric_point_correct :
  symmetric_point A = { x := 4, y := -2 } := sorry

theorem minimum_value_correct :
  (∀ P, line P → minimum_value P) = sqrt 293 := sorry

end symmetric_point_correct_minimum_value_correct_l324_324307


namespace max_consecutive_positive_terms_l324_324712

def sequence (a : ℕ → ℝ) : ℕ → ℝ := λ n, a(n)

theorem max_consecutive_positive_terms (a : ℕ → ℝ)
  (h_recurrence: ∀ n ≥ 2, a n = a (n - 1) + a (n + 2)) :
  ∃ k, 5 = k ∧ ∀ n, (a n > 0) ∧ (a (n + 1) > 0) ∧ (a (n + 2) > 0) ∧ (a (n + 3) > 0) ∧ (a (n + 4) > 0) ∧ ¬((a (n + 5) > 0) ∧ (a (n + 6) > 0)) :=
sorry

end max_consecutive_positive_terms_l324_324712


namespace number_of_sides_on_die_l324_324971

theorem number_of_sides_on_die (n : ℕ) 
  (h1 : n ≥ 6) 
  (h2 : (∃ k : ℕ, k = 5) → (5 : ℚ) / (n ^ 2 : ℚ) = (5 : ℚ) / (36 : ℚ)) 
  : n = 6 :=
sorry

end number_of_sides_on_die_l324_324971


namespace minimum_faces_combined_l324_324925

noncomputable def least_possible_faces_on_dice (a b : ℕ) : ℕ :=
if h1 : a ≥ 7 
   ∧ b ≥ 5 
   ∧ (1 / (a * b) * 12) = 2 * (1 / (a * b) * 7)
   ∧ (1 / (a * b) * 7) = 1 / 20 
then a + b else 0

theorem minimum_faces_combined (a b : ℕ) : (a ≥ 7) → (b ≥ 5) →
  (1 / (a * b) * 12 = 2 * (1 / (a * b) * 7)) →
  (1 / (a * b) * 7 = 1 / 20) →
  least_possible_faces_on_dice a b = 24 :=
by
  intros h1 h2 h3 h4
  unfold least_possible_faces_on_dice
  rw [if_pos]
  exact sorry
  -- Given conditions, the Lean statement must hold.
  finish

end minimum_faces_combined_l324_324925


namespace mr_kishore_groceries_expense_l324_324234

theorem mr_kishore_groceries_expense
  (S : ℝ) (G : ℝ) 
  (h1 : S * 0.10 = 2000)
  (h2 : 5000 + 1500 + G + 2500 + 2000 + 2500 = 0.90 * S) :
  G = 6500 :=
by
s --proof goes here

end mr_kishore_groceries_expense_l324_324234


namespace angle_C_max_perimeter_l324_324401

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def circumradius_2 (r : ℝ) : Prop :=
  r = 2

def satisfies_condition (a b c A B C : ℝ) : Prop :=
  (a - c)*(Real.sin A + Real.sin C) = b*(Real.sin A - Real.sin B)

theorem angle_C (A B C a b c : ℝ) (h₁ : triangle_ABC A B C a b c) 
                 (h₂ : satisfies_condition a b c A B C)
                 (h₃ : circumradius_2 (2 : ℝ)) : 
  C = Real.pi / 3 :=
sorry

theorem max_perimeter (A B C a b c r : ℝ) (h₁ : triangle_ABC A B C a b c)
                      (h₂ : satisfies_condition a b c A B C)
                      (h₃ : circumradius_2 r) : 
  4 * Real.sqrt 3 + 2 * Real.sqrt 3 = 6 * Real.sqrt 3 :=
sorry

end angle_C_max_perimeter_l324_324401


namespace solve_for_phi_l324_324752

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * φ)

theorem solve_for_phi (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π / 2)
    (h_min_diff : |x1 - x2| = π / 6)
    (h_condition : |f x1 - g x2 φ| = 4) :
    φ = π / 3 := 
    sorry

end solve_for_phi_l324_324752


namespace find_extrema_l324_324706

noncomputable def y (x : ℝ) := (Real.sin (3 * x))^2

theorem find_extrema : 
  ∃ (x : ℝ), (0 < x ∧ x < 0.6) ∧ (∀ ε > 0, ε < 0.6 - x → y (x + ε) ≤ y x ∧ y (x - ε) ≤ y x) ∧ x = Real.pi / 6 :=
by
  sorry

end find_extrema_l324_324706


namespace polygon_sides_l324_324044

theorem polygon_sides
  (n : ℕ)
  (h1 : 180 * (n - 2) - (2 * (2790 / (n - 1)) - 20) = 2790) :
  n = 18 := sorry

end polygon_sides_l324_324044


namespace greatest_alpha_coloring_l324_324734

theorem greatest_alpha_coloring (a b : ℕ) (h₁ : a < b) (h₂ : b < 2 * a)
    (h₃ : ∀ {A B : ℕ}, ∀ (H : ∀ (x : ℕ), x = A ∨ x = B), ∃ (x y ∈ finset.range (A * B)), colored (x, y)) : 
    ∃ α : ℝ, α = 1 / (a^2 + (b - a)^2) ∧ ∀ (N : ℕ), ∃ (T : finset (ℕ × ℕ)), T.card ≥ α * N^2 :=
by
  sorry

end greatest_alpha_coloring_l324_324734


namespace volume_in_cubic_yards_l324_324998

-- Define the conditions
def volume_in_cubic_feet : ℕ := 162
def cubic_feet_per_cubic_yard : ℕ := 27

-- Problem statement in Lean 4
theorem volume_in_cubic_yards : volume_in_cubic_feet / cubic_feet_per_cubic_yard = 6 := 
  by
    sorry

end volume_in_cubic_yards_l324_324998


namespace cos2_add_2sin2_eq_64_over_25_l324_324782

theorem cos2_add_2sin2_eq_64_over_25 (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
sorry

end cos2_add_2sin2_eq_64_over_25_l324_324782


namespace surface_area_of_cube_l324_324140

theorem surface_area_of_cube (edge : ℝ) (h : edge = 5) : 6 * (edge * edge) = 150 := by
  have h_square : edge * edge = 25 := by
    rw [h]
    norm_num
  rw [h_square]
  norm_num

end surface_area_of_cube_l324_324140


namespace y_gets_for_each_rupee_x_gets_l324_324652

variables (a x y z : ℝ)

theorem y_gets_for_each_rupee_x_gets :
  (∃ (a : ℝ), (x + a * x + 0.5 * x = 273) ∧ (y = a * x) ∧ (y = 63)) -> a = 0.45 :=
by {
  assume h,
  obtain ⟨a, ha1, ha2, hy⟩ := h,
  have hx : x ≠ 0,
  { sorry },
  have ha : a = 0.45,
  { sorry },
  exact ha,
}

end y_gets_for_each_rupee_x_gets_l324_324652


namespace angle_bisector_theorem_l324_324842

noncomputable theory
open_locale classical

variables {A B C D P Q : Type*}

-- Points are in the Euclidean plane
variables [Euclidean_plane A] [Euclidean_plane B] [Euclidean_plane C] [Euclidean_plane D]
variables (P : Euclidean_plane P) (Q : Euclidean_plane Q)

-- Quadrilateral ABCD is convex
def is_convex_quadrilateral (A B C D : Euclidean_plane) : Prop :=
  convex (A ∪ B ∪ C ∪ D)

-- Given P is interior to ABCD and the angle relationships
def angle_relations (A B C D P : Euclidean_plane) : Prop :=
  let φ := ∠PAD in
  let ψ := ∠CBP in
  ∠PBA = 2 * φ ∧ ∠DPA = 3 * φ ∧
  ∠BAP = 2 * ψ ∧ ∠BPC = 3 * ψ

-- Internal bisectors of ∠ADP and ∠PCB intersect at Q inside ABP
def is_bisector_intersection (A B C D P Q : Euclidean_plane) : Prop :=
  let bisect_ADP := bisector_of ∠ADP in
  let bisect_PCB := bisector_of ∠PCB in
  incircle_intersection (A ∪ B ∪ P) bisect_ADP bisect_PCB Q

theorem angle_bisector_theorem :
  is_convex_quadrilateral A B C D ∧ P.interior (A ∪ B ∪ C ∪ D) ∧
  angle_relations A B C D P ∧ is_bisector_intersection A B C D P Q →
  (dist A Q = dist B Q) :=
sorry

end angle_bisector_theorem_l324_324842


namespace total_students_school_l324_324549

theorem total_students_school 
  (n : Nat)
  (h1 : 6 * n ≥ 150)
  (h2 : 12.9 * n ≤ 400)
  (h3 : n ≥ 25)
  (h4 : n ≤ 31)
  (h5 : ∃ k, n = 10 * k) : 
  let boys := 6 * n
  let girls := 6 * 9 / 10 * n
  let total_students := boys + girls
  total_students = 387 := by
  let boys := 6 * n
  let girls := 6.9 * n
  let total_students := boys + girls
  have h1 : boys ≥ 150 := by exact h1
  have h2 : total_students ≤ 400 := by exact h2
  simp at total_students
  sorry

end total_students_school_l324_324549


namespace possible_arrangements_count_l324_324600

-- Define students as a type
inductive Student
| A | B | C | D | E | F

open Student

-- Define Club as a type
inductive Club
| A | B | C

open Club

-- Define the arrangement constraints
structure Arrangement :=
(assignment : Student → Club)
(club_size : Club → Nat)
(A_and_B_same_club : assignment A = assignment B)
(C_and_D_diff_clubs : assignment C ≠ assignment D)
(club_A_size : club_size A = 3)
(all_clubs_nonempty : ∀ c : Club, club_size c > 0)

-- Define the possible number of arrangements
def arrangement_count (a : Arrangement) : Nat := sorry

-- Theorem stating the number of valid arrangements
theorem possible_arrangements_count : ∃ a : Arrangement, arrangement_count a = 24 := sorry

end possible_arrangements_count_l324_324600


namespace valid_combinations_correct_l324_324231

def herbs : ℕ := 4
def stones : ℕ := 6
def incompatible_combinations : ℕ := 3
def total_combinations : ℕ := herbs * stones
def valid_combinations : ℕ := total_combinations - incompatible_combinations

theorem valid_combinations_correct : valid_combinations = 21 := by
  -- We assume these values are given based on the problem's conditions
  have h1 : herbs = 4 := rfl
  have h2 : stones = 6 := rfl
  have h3 : incompatible_combinations = 3 := rfl
  have h4 : total_combinations = herbs * stones := rfl
  have h5 : valid_combinations = total_combinations - incompatible_combinations := rfl
  -- From these values, it follows:
  calc
    valid_combinations
      = total_combinations - incompatible_combinations : by rw [h5]
  ... = herbs * stones - incompatible_combinations  : by rw [h4]
  ... = 4 * 6 - 3                                  : by rw [h1, h2, h3]
  ... = 24 - 3                                     : by rfl
  ... = 21                                         : by rfl

end valid_combinations_correct_l324_324231


namespace ellipse_equation_point_d_abscissa_l324_324737

-- Definitions for the problem conditions
def C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def is_right_focus (a c : ℝ) : Prop := c = 1
def point_on_ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop := ∃ x y, P = (x, y) ∧ C a b x y
def perpendicular_to_x_axis (F P : ℝ × ℝ) : Prop := F.1 = P.1 ∧ F.2 = 0
def line_l (m : ℝ) (F : ℝ × ℝ) : ℝ → ℝ × ℝ := λ x, (x, m * (x - F.1) + F.2)
def slopes (P A D B : ℝ × ℝ) : ℝ → ℝ → ℝ → ℝ := λ k₁ k₂ k₃, k₁ + k₃ = 2 * k₂

-- Theorem statements based on the conditions and the correct answer
theorem ellipse_equation {a b : ℝ} (h₁ : a > b > 0) (P : ℝ × ℝ)
  (h₂ : point_on_ellipse a b P) (h₃ : perpendicular_to_x_axis (1, 0) P) : 
  C 2 (√3) 1 (3/2) :=
sorry

theorem point_d_abscissa {a b : ℝ} (h₁ : a > b > 0) (P A B D : ℝ × ℝ)
  (h₂ : point_on_ellipse a b P) (h₃ : point_on_ellipse a b A) (h₄ : point_on_ellipse a b B)
  (h₅ : line_l m (1, 0) D) (h₆ : slopes P A D B (λ k₁ k₂ k₃, k₁ + k₃ = 2 * k₂)) :
  D.1 = 4 :=
sorry

end ellipse_equation_point_d_abscissa_l324_324737


namespace probability_correct_l324_324792

variable (new_balls old_balls total_balls : ℕ)

-- Define initial conditions
def initial_conditions (new_balls old_balls : ℕ) : Prop :=
  new_balls = 4 ∧ old_balls = 2

-- Define total number of balls in the box
def total_balls_condition (new_balls old_balls total_balls : ℕ) : Prop :=
  total_balls = new_balls + old_balls ∧ total_balls = 6

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of picking one new ball and one old ball
def probability_one_new_one_old (new_balls old_balls total_balls : ℕ) : ℚ :=
  (combination new_balls 1 * combination old_balls 1) / (combination total_balls 2)

-- The theorem to prove the probability
theorem probability_correct (new_balls old_balls total_balls : ℕ)
  (h_initial : initial_conditions new_balls old_balls)
  (h_total : total_balls_condition new_balls old_balls total_balls) :
  probability_one_new_one_old new_balls old_balls total_balls = 8 / 15 := by
  sorry

end probability_correct_l324_324792


namespace find_f_three_l324_324488

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l324_324488


namespace integral_f1_l324_324430

def f (x : ℝ) : ℝ := 1 / x

def f_1 (x : ℝ) : ℝ :=
  if x ≥ 1 then 1
  else 1 / x

theorem integral_f1 : (∫ x in (1/4)..2, f_1 x) = 1 + 2 * Real.log 2 := by
  sorry

end integral_f1_l324_324430


namespace length_of_CY_l324_324389

theorem length_of_CY (A B C X Y Z : Type) [Points A B C X Y Z]
  (AB : Length A B = 12) (XYZ_parallel_AB : Parallel XYZ AB)
  (XY_length : Length X Y = 6) (AX_bisects_BYZ : Bisects (Extension A X) (Angle B Y Z)) :
  ∃ (CY : Length C Y), CY = 12 :=
begin
  sorry
end

end length_of_CY_l324_324389


namespace students_neither_art_nor_music_l324_324797

def total_students := 75
def art_students := 45
def music_students := 50
def both_art_and_music := 30

theorem students_neither_art_nor_music : 
  total_students - (art_students - both_art_and_music + music_students - both_art_and_music + both_art_and_music) = 10 :=
by 
  sorry

end students_neither_art_nor_music_l324_324797


namespace max_b_value_l324_324428

noncomputable def f (x a : ℝ) := (3 / 2) * x^2 - 2 * a * x
noncomputable def g (x a b : ℝ) := a^2 * Real.log x + b

theorem max_b_value (a : ℝ) (ha: a > 0) (b : ℝ) :
  (∃ x0 y0 : ℝ, f x0 a = y0 ∧ g x0 a b = y0 ∧ (Deriv f x0 = Deriv g x0))
  → b ≤ 1 / (2 * Real.e^2) :=
by
  sorry

end max_b_value_l324_324428


namespace find_f_of_3_l324_324514

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l324_324514


namespace sum_log_series_l324_324252

noncomputable def log_base (a x : ℝ) := real.log x / real.log a

theorem sum_log_series :
  ∑ k in finset.range 98 \ finset.range 2,
    log_base 3 (1 + 1 / (k + 3)) * log_base k (3 : ℝ) * log_base (k + 4) 3
  = 0.7689 :=
by
  sorry

end sum_log_series_l324_324252


namespace integer_remainder_18_l324_324662

theorem integer_remainder_18 (n : ℤ) (h : n ∈ ({14, 15, 16, 17, 18} : Set ℤ)) : n % 7 = 4 :=
by
  sorry

end integer_remainder_18_l324_324662


namespace first_player_wins_l324_324550

theorem first_player_wins (initial_piles : list ℕ)
  (h1 : initial_piles = [50, 60, 70])
  (move_rule : ∀ pile, pile > 1 → ∃ p q, pile = p + q ∧ p, q ≥ 1)
  (winning_condition : ∀ piles, (∀ pile ∈ piles, pile = 1) → player_wins): 
  player_wins :=
by
  cases h1
  rename_var p1 p2 p3
  sorry

end first_player_wins_l324_324550


namespace repetitive_decimals_subtraction_correct_l324_324279

noncomputable def repetitive_decimals_subtraction : Prop :=
  let a : ℚ := 4567 / 9999
  let b : ℚ := 1234 / 9999
  let c : ℚ := 2345 / 9999
  a - b - c = 988 / 9999

theorem repetitive_decimals_subtraction_correct : repetitive_decimals_subtraction :=
  by sorry

end repetitive_decimals_subtraction_correct_l324_324279


namespace work_duration_l324_324200

/-- p and q can complete a work in 40 days and 24 days respectively.
    p works alone for 16 days, and then q joins p until the work is completed.
    Show that the total work duration is 25 days. -/
theorem work_duration {W : ℕ} : 
  let p_work_rate := W / 40,
      q_work_rate := W / 24,
      work_done_by_p_alone := 16 * (W / 40),
      remaining_work := W - work_done_by_p_alone,
      combined_work_rate := (W / 40) + (W / 24) in
  (16 + (remaining_work / combined_work_rate)) = 25 :=
  sorry

end work_duration_l324_324200


namespace sqrt_9_eq_pm3_l324_324938

theorem sqrt_9_eq_pm3 : ∃ x : ℤ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end sqrt_9_eq_pm3_l324_324938


namespace pizza_topping_count_l324_324610

theorem pizza_topping_count (n : ℕ) (hn : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by
  sorry

end pizza_topping_count_l324_324610


namespace snail_displacement_at_55_days_l324_324646

theorem snail_displacement_at_55_days :
  ∑ n in Finset.range 55, (1 / (n + 1 : ℝ) - 1 / (n + 2 : ℝ)) = 55 / 56 :=
by
  sorry

end snail_displacement_at_55_days_l324_324646


namespace unique_prime_digit_B_l324_324905

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → ¬(m ∣ n)

theorem unique_prime_digit_B :
  ∃ T B : ℕ, (B = 2) ∧ (199 * 10^3 + T * 10^2 + 0 + B) is_prime ∧
  ∀ B' : ℕ, B' ≠ 2 → ¬ is_prime (199 * 10^3 + T * 10^2 + 0 + B') :=
by
  sorry

end unique_prime_digit_B_l324_324905


namespace largest_angle_in_triangle_l324_324897

theorem largest_angle_in_triangle (y : ℝ) (h : 60 + 70 + y = 180) :
    70 > 60 ∧ 70 > y :=
by {
  sorry
}

end largest_angle_in_triangle_l324_324897


namespace four_digit_numbers_with_conditions_l324_324340

theorem four_digit_numbers_with_conditions :
  ∃ n : ℕ, n = 126 ∧
    (∃ a b c d : ℕ,
      1000 ≤ a * 10^3 + b * 10^2 + c * 10 + d ∧ 
      a * 10^3 + b * 10^2 + c * 10 + d < 10000 ∧
      a ∈ {2, 6, 8} ∧ 
      d = 4 ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
      b ≠ c ∧ b ≠ d ∧
      c ≠ d) := 
begin
  use 126,
  split,
  { refl },
  { sorry }
end

end four_digit_numbers_with_conditions_l324_324340


namespace perimeter_of_triangle_PQR_l324_324383

-- Define the geometric entities P, Q, and R
variables {P Q R : Type} [metric_space P] [metric_space Q] [metric_space R]

-- Define the conditions of the isosceles triangle PQR
def is_isosceles_triangle (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] (QR PR PQ : ℝ) : Prop :=
  ∠ P Q R = ∠ P R Q ∧ QR = 8 ∧ PR = 10

-- Define the problem to calculate the perimeter of the triangle
def triangle_perimeter (Q P R : Type) [metric_space P] [metric_space Q] [metric_space R] (QR PR PQ : ℝ) : ℝ :=
  PQ + QR + PR

-- The main theorem to prove the perimeter
theorem perimeter_of_triangle_PQR (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] :
  is_isosceles_triangle P Q R 8 10 10 → triangle_perimeter Q P R 8 10 10 = 28 :=
begin
  sorry
end

end perimeter_of_triangle_PQR_l324_324383


namespace range_of_a_l324_324427

def p (x : ℝ) : Prop := abs (2 * x - 1) ≤ 3

def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ x a, (¬ q x a) → (¬ p x))
  ∧ (∃ x a, (¬ q x a) ∧ (¬ p x))
  → (-1 : ℝ) ≤ a ∧ a ≤ (1 : ℝ) :=
sorry

end range_of_a_l324_324427


namespace tate_initial_tickets_l324_324458

theorem tate_initial_tickets (T : ℕ) (h1 : T + 2 + (T + 2)/2 = 51) : T = 32 := 
by
  sorry

end tate_initial_tickets_l324_324458


namespace value_of_c_l324_324031

theorem value_of_c :
  let c : ℕ := 3 in
  (5 ^ 5) * (9 ^ 3) = c * (15 ^ 5) :=
by
  sorry

end value_of_c_l324_324031


namespace ratio_F1F2_V1V2_l324_324415

-- Define the parabola \(y = (x-1)^2 + 1\)
def parabola_P (x : ℝ) : ℝ := (x - 1) ^ 2 + 1

-- Define vertex and focus of the parabola
def vertex_V1 : ℝ × ℝ := (1, 1)
def focus_F1 : ℝ × ℝ := (1, 1.25)

-- Define points A and B on the parabola
def point_A (a : ℝ) : ℝ × ℝ := (a, parabola_P a)
def point_B (b : ℝ) : ℝ × ℝ := (b, parabola_P b)

-- Define the condition for perpendicular tangents
def perp_tangents_condition (a b : ℝ) : Prop := 4 * (a - 1) * (b - 1) = -1

-- Define the midpoint of line segment AB
def midpoint_AB (a b : ℝ) : ℝ × ℝ :=
  let mid_x := (a + b) / 2
  let mid_y := ((a - 1) ^ 2 + 1 + (b - 1) ^ 2 + 1) / 2
  (mid_x, mid_y)

-- Define the locus equation of the midpoint (Q)
def locus_Q (α : ℝ) : ℝ := α^2 / 2 + 1

-- Define vertex and focus of the locus Q
def vertex_V2 : ℝ × ℝ := (0, 1)
def focus_F2 : ℝ × ℝ := (0, 1.25)

-- Prove the ratio of distances F1F2 and V1V2 is 1
theorem ratio_F1F2_V1V2 : 
  let F1 := focus_F1
  let F2 := focus_F2
  let V1 := vertex_V1
  let V2 := vertex_V2
  real.sqrt ((F1.1 - F2.1) ^ 2 + (F1.2 - F2.2) ^ 2) / 
  real.sqrt ((V1.1 - V2.1) ^ 2 + (V1.2 - V2.2) ^ 2) = 1 :=
by
  sorry

end ratio_F1F2_V1V2_l324_324415


namespace pizza_combination_count_l324_324616

open_locale nat

-- Given condition: the number of different toppings
def num_toppings : ℕ := 8

-- Calculating the total number of one-topping, two-topping, and three-topping pizzas
theorem pizza_combination_count : ((nat.choose num_toppings 1) + (nat.choose num_toppings 2) + (nat.choose num_toppings 3)) = 92 :=
by
  sorry

end pizza_combination_count_l324_324616


namespace not_possible_divisible_columns_l324_324047

theorem not_possible_divisible_columns (grid : Fin 100 → Fin 100 → ℕ) 
  (h : ∀ i : Fin 100, (∑ j : Fin 100, grid i j) % 11 = 0)
  (h_nonzero : ∀ i j : Fin 100, grid i j ≠ 0) : 
  ¬ (∃ cols : Finset (Fin 100), cols.card = 99 ∧ ∀ j ∈ cols, (∑ i : Fin 100, grid i j) % 11 = 0) :=
sorry

end not_possible_divisible_columns_l324_324047


namespace adelaide_ducks_l324_324655

variable (A E K : ℕ)

theorem adelaide_ducks (h1 : A = 2 * E) (h2 : E = K - 45) (h3 : (A + E + K) / 3 = 35) :
  A = 30 := by
  sorry

end adelaide_ducks_l324_324655


namespace distance_PQ_is_12_miles_l324_324442

-- Define the conditions
def average_speed_PQ := 40 -- mph
def average_speed_QP := 45 -- mph
def time_difference := 2 -- minutes

-- Main proof statement to show that the distance is 12 miles
theorem distance_PQ_is_12_miles 
    (x : ℝ) 
    (h1 : average_speed_PQ > 0) 
    (h2 : average_speed_QP > 0) 
    (h3 : abs ((x / average_speed_PQ * 60) - (x / average_speed_QP * 60)) = time_difference) 
    : x = 12 := 
by
  sorry

end distance_PQ_is_12_miles_l324_324442


namespace a2_and_a3_values_geometric_sequence_general_formula_sum_of_first_n_terms_l324_324813

noncomputable def sequence (n : ℕ) : ℕ → ℤ
| 1       := 3
| (n + 1) := - sequence n - 2 * (n + 1) + 1

theorem a2_and_a3_values :
  sequence 2 = -6 ∧ sequence 3 = 1 :=
sorry

theorem geometric_sequence_general_formula :
  ∃ r : ℤ, r = -1 ∧ ∀ n : ℕ, n > 0 → sequence n + n = 4 * (-1) ^ (n - 1) :=
sorry

theorem sum_of_first_n_terms (n : ℕ) :
  ∑ i in finset.range n, sequence (i + 1) = -(n^2 + n - 4) / 2 - 2 * (-1) ^ n :=
sorry

end a2_and_a3_values_geometric_sequence_general_formula_sum_of_first_n_terms_l324_324813


namespace pizza_topping_count_l324_324624

theorem pizza_topping_count (n : ℕ) (h : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by {
  rw h,
  simp [nat.choose],
  sorry,
}

end pizza_topping_count_l324_324624


namespace not_on_graph_ln_l324_324041

theorem not_on_graph_ln {a b : ℝ} (h : b = Real.log a) : ¬ (1 + b = Real.log (a + Real.exp 1)) :=
by
  sorry

end not_on_graph_ln_l324_324041


namespace sum_of_integers_l324_324470

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 288) : x + y = 35 :=
sorry

end sum_of_integers_l324_324470


namespace angle_B_is_acute_l324_324443

theorem angle_B_is_acute (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (triangle : Triangle A B C) (right_angle_C : angle C = 90) : angle B < 90 :=
by
  sorry

end angle_B_is_acute_l324_324443


namespace incorrect_value_of_observation_l324_324526

theorem incorrect_value_of_observation
  (mean_initial : ℝ) (n : ℕ) (sum_initial: ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mean_corrected : ℝ)
  (h1 : mean_initial = 36) 
  (h2 : n = 50) 
  (h3 : sum_initial = n * mean_initial) 
  (h4 : correct_value = 45) 
  (h5 : mean_corrected = 36.5) 
  (sum_corrected : ℝ) 
  (h6 : sum_corrected = n * mean_corrected) : 
  incorrect_value = 20 := 
by 
  sorry

end incorrect_value_of_observation_l324_324526


namespace evaluate_expression_l324_324284

def greatest_integer_le (y : ℝ) : ℤ := int.floor y

theorem evaluate_expression : 
  ∀ x : ℝ, x = 2 →
  ([greatest_integer_le 6.5 * greatest_integer_le (2 / 3)] + ([greatest_integer_le x] * 7.2) + [greatest_integer_le 8.3] - 6.6 = 15.8) :=
by
  intros x hx
  have H1 : greatest_integer_le 6.5 = 6 := by sorry
  have H2 : greatest_integer_le (2 / 3) = 0 := by sorry
  have H3 : greatest_integer_le x = 2 := by sorry
  have H4 : greatest_integer_le 8.3 = 8 := by sorry
  rw [H1, H2, H3, H4]
  norm_num
  sorry

end evaluate_expression_l324_324284


namespace find_a_l324_324326

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then (1/2)^x - 7 else x^2

theorem find_a (a : ℝ) (h : f a = 1) : a = -3 ∨ a = 1 := 
by
  sorry

end find_a_l324_324326


namespace area_of_shaded_region_l324_324066

open Real

-- Define points and squares
structure Point (α : Type*) := (x : α) (y : α)

def A := Point.mk 0 12 -- top-left corner of large square
def G := Point.mk 0 0  -- bottom-left corner of large square
def F := Point.mk 4 0  -- bottom-right corner of small square
def E := Point.mk 4 4  -- top-right corner of small square
def C := Point.mk 12 0 -- bottom-right corner of large square
def D := Point.mk 3 0  -- intersection of AF extended with the bottom edge

-- Define the length of sides
def side_small_square : ℝ := 4
def side_large_square : ℝ := 12

-- Areas calculation
def area_square (side : ℝ) : ℝ := side * side

def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Theorem statement
theorem area_of_shaded_region : area_square side_small_square - area_triangle 3 side_small_square = 10 :=
by
  rw [area_square, area_triangle]
  -- Plug in values: 4^2 - 0.5 * 3 * 4
  norm_num
  sorry

end area_of_shaded_region_l324_324066


namespace solve_trig_eq_l324_324873

theorem solve_trig_eq 
  (x : ℝ)
  (h₁ : cos x ≠ 0)
  (h₂ : sin x ≠ 0) :
  3 * sqrt 3 * tan x * sin x - cot x * cos x + 9 * sin x - 3 * sqrt 3 * cos x = 0 ↔
    ∃ k n l : ℤ,
    x = π / 6 * (6 * k + 1) ∨ 
    x = arctan ((-2 * sqrt 3 - 3) / 3) + π * n ∨
    x = arctan ((-2 * sqrt 3 + 3) / 3) + π * l := 
sorry

end solve_trig_eq_l324_324873


namespace option_A_not_equal_option_B_equal_option_C_not_equal_option_D_not_equal_final_answer_l324_324235

noncomputable def f_A (x : ℝ) : ℝ := x - 1
noncomputable def g_A (x : ℝ) : ℝ := if x = -1 then 0 else (x^2 - 1) / (x + 1)

noncomputable def f_B (x : ℝ) : ℝ := abs (x + 1)
noncomputable def g_B (x : ℝ) : ℝ := if x ≥ -1 then x + 1 else -x - 1

noncomputable def f_C (x : ℝ) : ℝ := 1
noncomputable def g_C (x : ℝ) : ℝ := if x = -1 then 0 else (x + 1)^0

noncomputable def f_D (x : ℝ) : ℝ := x
noncomputable def g_D (x : ℝ) : ℝ := if x < 0 then 0 else (sqrt x) ^ 2

theorem option_A_not_equal : ∃ x : ℝ, f_A x ≠ g_A x := sorry
theorem option_B_equal : ∀ x : ℝ, f_B x = g_B x := sorry
theorem option_C_not_equal : ∃ x : ℝ, f_C x ≠ g_C x := sorry
theorem option_D_not_equal : ∃ x : ℝ, f_D x ≠ g_D x := sorry

theorem final_answer : set_not eq AC : {option_A_not_equal, option_C_not_equal, option_D_not_equal} = true := sorry

end option_A_not_equal_option_B_equal_option_C_not_equal_option_D_not_equal_final_answer_l324_324235


namespace new_person_weight_l324_324201

noncomputable def average_weight_increase (current_weight replaced_weight increase_in_average : ℝ) : ℝ :=
  current_weight + 2 * increase_in_average - replaced_weight

theorem new_person_weight (increase_in_average : ℝ) (replaced_weight : ℝ) (new_weight : ℝ) :
  increase_in_average = 4.5 → replaced_weight = 65 → new_weight = 74 :=
by
  intros h_increase h_replaced
  have total_increase : ℝ := 2 * increase_in_average
  have h1 : total_increase = 9 := by linarith
  have new_weight_def := replaced_weight + total_increase
  have h2: new_weight_def = new_weight := by linarith
  exact h2

end new_person_weight_l324_324201


namespace maria_chairs_l324_324438

variable (C : ℕ) -- Number of chairs Maria bought
variable (tables : ℕ := 2) -- Number of tables Maria bought is 2
variable (time_per_furniture : ℕ := 8) -- Time spent on each piece of furniture in minutes
variable (total_time : ℕ := 32) -- Total time spent assembling furniture

theorem maria_chairs :
  (time_per_furniture * C + time_per_furniture * tables = total_time) → C = 2 :=
by
  intro h
  sorry

end maria_chairs_l324_324438


namespace possible_table_sum_80_l324_324807

variable {f : ℕ → ℕ → ℕ}

def valid_table (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ i < 9, ∀ j < 9, f i j ≥ 0

def row_sum_constraint (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ i < 8, (∑ j in finset.range 9, f i j + f (i + 1) j) ≥ 20

def column_sum_constraint (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ j < 8, (∑ i in finset.range 9, f i j + f i (j + 1)) ≤ 16

theorem possible_table_sum_80 :
  (∃ f : ℕ → ℕ → ℕ, valid_table f ∧ row_sum_constraint f ∧ column_sum_constraint f) →
  (∑ i in finset.range 9, ∑ j in finset.range 9, f i j = 80) :=
by
  sorry

end possible_table_sum_80_l324_324807


namespace find_m_l324_324317

theorem find_m (m : ℕ) (h_m_pos : m > 0) (h_m_decomp : ∃ smallest, (smallest = 91) ∧ 
  (m^3 = ∑ i in finset.range m, (smallest + 2 * i))) : m = 10 :=
by {
  sorry
}

end find_m_l324_324317


namespace intersection_A_B_l324_324760

def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | -1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_A_B_l324_324760


namespace find_number_of_elements_l324_324461

theorem find_number_of_elements (n S : ℕ) (h1: (S + 26) / n = 15) (h2: (S + 36) / n = 16) : n = 10 := by
  sorry

end find_number_of_elements_l324_324461


namespace range_of_m_l324_324296

/--
Given a circle C with the equation (x - 1)² + (y - √3)² = 1, 
and two points A(0, m) and B(0, -m) (m > 0), 
if there exists a point P on circle C such that ∠APB = 90 degrees, 
then the range of values for the real number m is [1, 3].
-/
theorem range_of_m (C : set (ℝ × ℝ)) (A B : ℝ × ℝ) (m : ℝ) (hC : ∀ (x y : ℝ), (x - 1)^2 + (y - real.sqrt 3)^2 = 1 → (x, y) ∈ C)
  (hA : A = (0, m)) (hB : B = (0, -m)) (m_pos : m > 0) :
  (∃ P ∈ C, ∠ A P B = 90°) → 1 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l324_324296


namespace circle_centers_connection_line_eq_l324_324472

-- Define the first circle equation
def circle1 (x y : ℝ) := (x^2 + y^2 - 4*x + 6*y = 0)

-- Define the second circle equation
def circle2 (x y : ℝ) := (x^2 + y^2 - 6*x = 0)

-- Given the centers of the circles, prove the equation of the line connecting them
theorem circle_centers_connection_line_eq (x y : ℝ) :
  (∀ (x y : ℝ), circle1 x y → (x = 2 ∧ y = -3)) →
  (∀ (x y : ℝ), circle2 x y → (x = 3 ∧ y = 0)) →
  (3 * x - y - 9 = 0) :=
by
  -- Here we would sketch the proof but skip it with sorry
  sorry

end circle_centers_connection_line_eq_l324_324472


namespace intersection_P_Q_l324_324687

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l324_324687


namespace area_twice_l324_324065

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def area_of_quadrilateral (A1 A2 A3 A4 : ℝ × ℝ) :=
  (1 / 2) * abs ((fst A1 * snd A2 + fst A2 * snd A3 + fst A3 * snd A4 + fst A4 * snd A1) - 
                (snd A1 * fst A2 + snd A2 * fst A3 + snd A3 * fst A4 + snd A4 * fst A1))

noncomputable def is_parallel_and_equal (A1 A2 B O : ℝ × ℝ) : Prop :=
  (fst B - fst O = fst A2 - fst A1) ∧ (snd B - snd O = snd A2 - snd A1)

def quadrilateral_parity (A1 A2 A3 A4 B1 B2 B3 B4 O : ℝ × ℝ) (h1 : is_parallel_and_equal A1 A2 B1 O)
  (h2 : is_parallel_and_equal A2 A3 B2 O) (h3 : is_parallel_and_equal A3 A4 B3 O) (h4 : is_parallel_and_equal A4 A1 B4 O) :
  Prop := area_of_quadrilateral B1 B2 B3 B4 = 2 * area_of_quadrilateral A1 A2 A3 A4

theorem area_twice (A1 A2 A3 A4 B1 B2 B3 B4 O : ℝ × ℝ) (h1 : is_parallel_and_equal A1 A2 B1 O)
  (h2 : is_parallel_and_equal A2 A3 B2 O) (h3 : is_parallel_and_equal A3 A4 B3 O) (h4 : is_parallel_and_equal A4 A1 B4 O) :
  quadrilateral_parity A1 A2 A3 A4 B1 B2 B3 B4 O h1 h2 h3 h4 :=
sorry

end area_twice_l324_324065


namespace cos_2theta_plus_pi_over_3_l324_324723

theorem cos_2theta_plus_pi_over_3 (θ : ℝ) (hθ1 : θ ∈ Ioo (π / 2) π) (hθ2 : (1 / (Real.sin θ) + 1 / (Real.cos θ)) = 2 * Real.sqrt 2) :
  Real.cos (2 * θ + π / 3) = Real.sqrt 3 / 2 :=
by
  have h1 : Real.sin θ > 0 := sorry
  have h2 : Real.cos θ < 0 := sorry
  have h3 : 2 * θ ∈ Ioo (3 * π / 2) (2 * π) := sorry
  have h4 : Real.sin (2 * θ) = -1 / 2 := sorry
  have h5 : Real.cos (2 * θ) = Real.sqrt 3 / 2 := sorry
  rw [Real.cos_add 2 * θ (π / 3)],
  rw [h5, Real.cos_pi_div_three, Real.sin_pi_div_three, mul_div_cancel_left, ← neg_div],
  simp,
  norm_num,
  sorry

end cos_2theta_plus_pi_over_3_l324_324723


namespace line_segment_param_square_sum_l324_324893

theorem line_segment_param_square_sum :
  (∃ a b c d : ℝ, (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    ((a * t + b = at t) ∧ (c * t + d = yt t))) ∧
    (a * 0 + b = -3) ∧ (c * 0 + d = 9) ∧
    (a * 1 + b = 2) ∧ (c * 1 + d = 12))
  → (∃ a b c d : ℝ, a^2 + b^2 + c^2 + d^2 = 124) :=
by
  sorry

end line_segment_param_square_sum_l324_324893


namespace michael_truck_meet_once_l324_324102

-- Michael's walking speed.
def michael_speed := 4 -- feet per second

-- Distance between trash pails.
def pail_distance := 100 -- feet

-- Truck's speed.
def truck_speed := 8 -- feet per second

-- Time truck stops at each pail.
def truck_stop_time := 20 -- seconds

-- Prove how many times Michael and the truck will meet given the initial condition.
theorem michael_truck_meet_once :
  ∃ n : ℕ, michael_truck_meet_count == 1 :=
sorry

end michael_truck_meet_once_l324_324102


namespace part1_part2_part3_l324_324729

-- Step 1: Conditions
def a (n : ℕ) : ℕ
| 1       := 1
| (n + 1) := 2 * a n + 1

-- Step 1: Prove that {a_n + 1} is a geometric sequence
theorem part1 (n : ℕ) : ∀ n, ∃ r b : ℕ, (a (n+1) + 1) = r * (a n + 1) :=
sorry

-- Step 2: Prove the general formula for the sequence a_n
theorem part2 (n : ℕ) : a n = 2^n - 1 :=
sorry

-- Step 3: Prove the range of values for the sum T_n
def c (n : ℕ) : ℚ := (a n + 1) / (n * (n + 1) * 2^n)

def T (n : ℕ) : ℚ := ∑ i in finset.range n, c i

theorem part3 (n : ℕ) : ∀ n, 1/2 ≤ T n ∧ T n < 1 :=
sorry

end part1_part2_part3_l324_324729


namespace sequence_range_k_l324_324300

theorem sequence_range_k (a_1 a_2 a_3 a_4 k : ℝ)
    (h_geom : a_1 * a_3 = a_2 ^ 2)
    (sum_geom : a_1 + a_2 + a_3 = k)
    (h_arith : 2 * a_3 = a_2 + a_4)
    (sum_arith : a_2 + a_3 + a_4 = 15)
    (h_not_zero : a_4 - a_3 ≠ a_3 - a_2) :
    (∃a_1 a_2 a_3 a_4, (a_1 * a_3 = a_2^2) ∧ (a_1 + a_2 + a_3 = k) ∧ (2 * a_3 = a_2 + a_4) ∧ (a_2 + a_3 + a_4 = 15) ∧ (a_4 - a_3 ≠ a_3 - a_2)) ↔
    k ∈ set.Icc (15/4) 5 ∪ set.Ioc 5 15 ∪ set.Ioi 15 := sorry

end sequence_range_k_l324_324300


namespace probability_favorite_track_before_eighth_l324_324953

/-- Pete's favorite track is the 8th track on an 11 track CD.
    When the CD is in random mode, what is the probability
    that he will reach his favorite track with fewer than 8 button presses? --/
theorem probability_favorite_track_before_eighth :
  let n := 11 in
  let favorite := 8 in
  (1 / n) + (1 / n) + (1 / n) + (1 / n) + (1 / n) + (1 / n) + (1 / n) = 7 / 11 :=
by sorry

end probability_favorite_track_before_eighth_l324_324953


namespace john_bought_3_croissants_l324_324702

variable (c k : ℕ)

theorem john_bought_3_croissants
  (h1 : c + k = 5)
  (h2 : ∃ n : ℕ, 88 * c + 44 * k = 100 * n) :
  c = 3 :=
by
-- Proof omitted
sorry

end john_bought_3_croissants_l324_324702


namespace greatest_number_dividing_1642_and_1856_l324_324569

theorem greatest_number_dividing_1642_and_1856 (a b r1 r2 k : ℤ) (h_intro : a = 1642) (h_intro2 : b = 1856) 
    (h_r1 : r1 = 6) (h_r2 : r2 = 4) (h_k1 : k = Int.gcd (a - r1) (b - r2)) :
    k = 4 :=
by
  sorry

end greatest_number_dividing_1642_and_1856_l324_324569


namespace fixed_point_l324_324474

theorem fixed_point (a : ℝ) : ∃ y : ℝ, y = log 2 (a * 0 + 1) + 1 ∧ (0, 1) = (0, y) := 
by
  use 1
  sorry

end fixed_point_l324_324474


namespace subproblem1_l324_324790

theorem subproblem1 (a b c q : ℝ) (h1 : c = b * q) (h2 : c = a * q^2) : 
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 := 
sorry

end subproblem1_l324_324790


namespace gravitational_force_at_new_distance_l324_324891

-- Define the inverse proportionality constant
def inverse_proportionality_constant (f d : ℝ) : ℝ := f * d^2

-- Give the conditions
def initial_distance : ℝ := 5000
def initial_force : ℝ := 500

-- Calculate the constant k from the initial conditions
def k := inverse_proportionality_constant initial_force initial_distance

-- Define the second distance and expected force
def second_distance : ℝ := 200000
def expected_force : ℝ := 5 / 16

-- State the problem as proving the gravitational force at a new distance
theorem gravitational_force_at_new_distance : 
  (inverse_proportionality_constant expected_force second_distance) = k :=
by
  -- Replace with actual proof if necessary
  sorry

end gravitational_force_at_new_distance_l324_324891


namespace a_and_b_complete_work_in_7_days_l324_324948

def work_rate_B := (1 / 21 : ℝ)
def work_rate_A := 2 * work_rate_B
def combined_work_rate := work_rate_A + work_rate_B
def time_to_complete := 1 / combined_work_rate

theorem a_and_b_complete_work_in_7_days :
  time_to_complete = 7 := sorry

end a_and_b_complete_work_in_7_days_l324_324948


namespace thomas_blocks_total_l324_324918

theorem thomas_blocks_total 
  (first_stack : ℕ)
  (second_stack : ℕ)
  (third_stack : ℕ)
  (fourth_stack : ℕ)
  (fifth_stack : ℕ) 
  (h1 : first_stack = 7)
  (h2 : second_stack = first_stack + 3)
  (h3 : third_stack = second_stack - 6)
  (h4 : fourth_stack = third_stack + 10)
  (h5 : fifth_stack = 2 * second_stack) :
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack = 55 :=
by
  sorry

end thomas_blocks_total_l324_324918


namespace grading_options_count_l324_324795

theorem grading_options_count :
  (4 ^ 15) = 1073741824 :=
by
  sorry

end grading_options_count_l324_324795


namespace AE_eq_DE_l324_324068

noncomputable theory
open_locale classical

variables {A B C F D E : Type*} [metric_space A]
variables (triangle : triangle A B C)
variables (AF : segment A F) (median_AF : is_median AF C B)
variables (midpoint_DF: D = midpoint F A)
variables (intersection_E: E ∈ line A B ∧ E ∈ line C D)
variables (BD_FC : dist B D = dist B F)
variables (BD = dist midpoint B F)

-- Define the segments involved
def is_median (seg : segment A F) (C B : point) : Prop :=
  is_midpoint F C && is_midpoint F B

-- Define midpoint
def midpoint (p1 p2 : point) : point :=
  sorry -- (Just a placeholder, you need to define midpoint correctly)

-- Define a theorem to prove the equality of segments AE and DE
theorem AE_eq_DE :
  dist A E = dist D E :=
sorry

end AE_eq_DE_l324_324068


namespace find_a_range_l324_324756

theorem find_a_range (a : ℝ) (x : ℝ) (h1 : a * x < 6) (h2 : (3 * x - 6 * a) / 2 > a / 3 - 1) :
  a ≤ -3 / 2 :=
sorry

end find_a_range_l324_324756


namespace sum_of_func_values_l324_324727

def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem sum_of_func_values : 
  ∑ k in finset.range 4025 + 1, f (k / 2013) = -8050 :=
by
  let f := λ x : ℝ, x^3 - 3 * x^2
  sorry

end sum_of_func_values_l324_324727


namespace power_neg8_equality_l324_324930

theorem power_neg8_equality :
  (1 / ((-8 : ℤ) ^ 2)^3) * (-8 : ℤ)^7 = 8 :=
by
  sorry

end power_neg8_equality_l324_324930


namespace find_number_l324_324347

theorem find_number (n : ℝ) (x : ℕ) (h1 : x = 4) (h2 : n^(2*x) = 3^(12-x)) : n = 3 := by
  sorry

end find_number_l324_324347


namespace fixed_points_of_quadratic_function_l324_324713

noncomputable def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = x

noncomputable def has_exactly_one_fixed_point (f : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, is_fixed_point f x

noncomputable def quadratic_function (a : ℝ) : ℝ → ℝ :=
  λ x, a * x ^ 2 + (2 * a - 3) * x + 1

theorem fixed_points_of_quadratic_function (a : ℝ) :
  has_exactly_one_fixed_point (quadratic_function a) ↔ a = 0 ∨ a = 1 ∨ a = 4 := 
sorry

end fixed_points_of_quadratic_function_l324_324713


namespace nth_equation_pattern_l324_324852

theorem nth_equation_pattern (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 :=
by
  sorry

end nth_equation_pattern_l324_324852


namespace find_c_l324_324707

theorem find_c (k n d : ℝ) (h₁ : d ≠ 0) (h₂ : (0, 3) ∈ setOf (λ p : ℝ × ℝ, p.snd = n * p.fst + d)) :
  ∃ c : ℝ, c = 6 ∧ (∃ k : ℝ, abs (k^2 + 4*k + 3 - (n * k + d)) = c) :=
by
  sorry

end find_c_l324_324707


namespace meeting_integer_l324_324232

theorem meeting_integer
  (n : ℕ)
  (A1 : n < 12 ∨ ¬(n < 12))
  (A2 : 7 ∣ n ∨ ¬(7 ∣ n))
  (A3 : 5 * n < 70 ∨ ¬(5 * n < 70))
  (B1 : 12 * n > 1000 ∨ ¬(12 * n > 1000))
  (B2 : 10 ∣ n ∨ ¬(10 ∣ n))
  (B3 : n > 100 ∨ ¬(n > 100))
  (C1 : 4 ∣ n ∨ ¬(4 ∣ n))
  (C2 : 11 * n < 1000 ∨ ¬(11 * n < 1000))
  (C3 : 9 ∣ n ∨ ¬(9 ∣ n))
  (D1 : n < 20 ∨ ¬(n < 20))
  (D2 : (nat.prime n) ∨ ¬(nat.prime n))
  (D3 : 7 ∣ n ∨ ¬(7 ∣ n))
  (A_one_true : A1 ∨ A2 ∨ A3)
  (A_one_false : ¬A1 ∨ ¬A2 ∨ ¬A3)
  (B_one_true : B1 ∨ B2 ∨ B3)
  (B_one_false : ¬B1 ∨ ¬B2 ∨ ¬B3)
  (C_one_true : C1 ∨ C2 ∨ C3)
  (C_one_false : ¬C1 ∨ ¬C2 ∨ ¬C3)
  (D_one_true : D1 ∨ D2 ∨ D3)
  (D_one_false : ¬D1 ∨ ¬D2 ∨ ¬D3)
  : n = 89 := by
  sorry

end meeting_integer_l324_324232


namespace columns_contain_1_to_n_l324_324731

theorem columns_contain_1_to_n 
  (n k m : ℕ) 
  (hk : k < n) 
  (hmk : k < m) 
  (h_coprime : Nat.coprime m (n - k))
  (h_mpos : 0 < m - k)
  (h_ordered_1st_row : ∀ i, 1 ≤ i ∧ i ≤ n → i = (i - 1) % n + 1)
  (h_transformation : ∀ r j, 
    1 ≤ j ∧ j ≤ n →
    (if j ≤ k then r + (n - k) + j
    else if j ≤ m then r + (j - k)
    else r + (j - m - k)))
  : (∀ c, ∀ i, 1 ≤ i ∧ i ≤ n → ∃ r, 1 ≤ r ∧ r ≤ n ∧ h_transformation r c = i) :=
sorry

end columns_contain_1_to_n_l324_324731


namespace general_term_formula_l324_324517

theorem general_term_formula (n : Nat) :
  let seq := λ (n : Nat), (n^2 + 2 * n) / (n + 1)
  (seq 1 = 3 / 2 ∧
   seq 2 = 8 / 3 ∧
   seq 3 = 15 / 4 ∧
   seq 4 = 24 / 5 ∧
   seq 5 = 35 / 6 ∧
   seq 6 = 48 / 7) :=
by
  sorry

end general_term_formula_l324_324517


namespace part1_part2_l324_324083

-- Conditions: a triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- Given a * sin A + c * sin C = b
variable (a b c A B C : ℝ)
variable (sin_A sin_C : ℝ)
variable (S k : ℝ)

-- Part (1): Prove sin B + (2ac / (a^2 + c^2)) cos B = 1
theorem part1 (h1: a * sin_A + c * sin_C = b)
    (h2: sin_A = sin A)
    (h3: sin_C = sin C)
    (cos_B : ℝ) :
    sin B + (2 * a * c / (a^2 + c^2)) * cos B = 1 := sorry

-- Part (2): Prove max value of k = 1/4 and that triangle is right-angled when k = 1/4
theorem part2 (h4: S = k * b^2)
    (h5: S = (1 / 2) * a * c * sin B)
    (h6: k > 0) :
    k <= 1/4 ∧ (k = 1/ 4 → angle B = π / 2) := sorry 

end part1_part2_l324_324083


namespace right_triangle_hypotenuse_l324_324857

theorem right_triangle_hypotenuse (h : ℝ) :
  (∀ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = b ∧ a = 12 ∧ ∃ θ : ℝ, θ = real.pi / 4 → c = h) → h = 12 * real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l324_324857


namespace wall_length_to_height_ratio_l324_324147

theorem wall_length_to_height_ratio (W H L V : ℝ) (h1 : H = 6 * W) (h2 : V = W * H * L) (h3 : W = 4) (h4 : V = 16128) :
  L / H = 7 :=
by
  -- Note: The proof steps are omitted as per the problem's instructions.
  sorry

end wall_length_to_height_ratio_l324_324147


namespace solve_for_z_l324_324451

-- Define the problem context
variables {z : ℂ}

-- The given condition
def given_condition : Prop := 4 - 2 * complex.I * z = 3 + 5 * complex.I * z

-- The proof problem to solve for z
theorem solve_for_z (h : given_condition) : z = (complex.I / 7) := 
sorry

end solve_for_z_l324_324451


namespace curve_C_equation_polar_of_diameter_l324_324166

-- Definition of the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Transformation conditions
def transform_x (x1 : ℝ) : ℝ := 2 * x1
def transform_y (y1 : ℝ) : ℝ := y1

-- Definition of curve C after transformation
def curve_C (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 4) = 1

-- Definition of the line l
def line_l (x y : ℝ) : Prop :=
  x - 2 * y + 4 = 0

-- Intersection points with curve C
def P1 : ℝ × ℝ := (-4, 0)
def P2 : ℝ × ℝ := (0, 2)

-- Midpoint of P1 and P2
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  ( (P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2 )

-- Radius of the circle with diameter P1P2
def radius (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ( (P1.1 - P2.1)^2 + (P1.2 - P2.2)^2 ) / 2

-- Polar equation of the circle
def polar_eqn (ρ θ : ℝ) : Prop :=
  ρ = -4 * real.cos θ + 2 * real.sin θ

-- Theorems to prove
theorem curve_C_equation (x y x1 y1 : ℝ) (h_trans_x : x = transform_x x1) (h_trans_y : y = transform_y y1) (h_orig_circle : original_circle x1 y1) :
  curve_C x y := sorry

theorem polar_of_diameter (ρ θ : ℝ) (h_P1 : line_l P1.1 P1.2) (h_P2 : line_l P2.1 P2.2) :
  polar_eqn ρ θ := sorry

end curve_C_equation_polar_of_diameter_l324_324166


namespace range_of_function_l324_324535

theorem range_of_function : 
  (set.range (λ x : ℝ, x^2 / (x^2 + 1)) = set.Ico 0 1) := 
sorry

end range_of_function_l324_324535


namespace tan_subtraction_formula_l324_324025

theorem tan_subtraction_formula 
  (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := 
by
  sorry

end tan_subtraction_formula_l324_324025


namespace max_x_satisfies_inequality_l324_324748

theorem max_x_satisfies_inequality (k : ℝ) :
    (∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) → k = 8 :=
by
  intros h
  /- The proof goes here. -/
  sorry

end max_x_satisfies_inequality_l324_324748


namespace math_problem_l324_324726

theorem math_problem (x : ℤ) :
  let a := 1990 * x + 1989
  let b := 1990 * x + 1990
  let c := 1990 * x + 1991
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end math_problem_l324_324726


namespace range_of_k_for_circle_l324_324348

theorem range_of_k_for_circle (x y : ℝ) (k : ℝ) : 
  (x^2 + y^2 - 4*x + 2*y + 5*k = 0) → k < 1 :=
by 
  sorry

end range_of_k_for_circle_l324_324348


namespace function_form_l324_324273

theorem function_form (f : ℕ → ℕ) (H : ∀ (x y z : ℕ), x ≠ y → y ≠ z → z ≠ x → (∃ k : ℕ, x + y + z = k^2 ↔ ∃ m : ℕ, f x + f y + f z = m^2)) : ∃ k : ℕ, ∀ n : ℕ, f n = k^2 * n :=
by
  sorry

end function_form_l324_324273


namespace find_f_3_l324_324501

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l324_324501


namespace sum_of_star_tip_angles_l324_324280

-- All points are evenly spaced on a circle, connected to form a star, with one tip angle labeled as beta.
theorem sum_of_star_tip_angles :
  (let beta : ℝ := 108 in 5 * beta = 540) :=
by
  sorry

end sum_of_star_tip_angles_l324_324280


namespace angle_SPR_equals_twice_angle_P_l324_324390

open Real

noncomputable def triangle_PQR (P Q R : Point) :=
  triangle P Q R ∧ QR = diameter_of_some_circle ∧ PQ_is_hypotenuse_of_right_triangle_PQR

theorem angle_SPR_equals_twice_angle_P (P Q R S : Point) :
  triangle_PQR P Q R →
  meets_tangent_at_R_and_intersects_hypotenuse_at_PQ P Q R S →
  angle SPR = 2 * angle P := by
  sorry

end angle_SPR_equals_twice_angle_P_l324_324390


namespace count_primes_in_sequence_l324_324011

-- Define the list as a sequence of numbers starting with 47, and each subsequent 
-- number is 47 concatenated to the previous ones forming a sequence of the form 47*(10^n + 10^(n-2) + ... + 10^0).
def sequence (n : ℕ) : ℕ :=
  47 * (list.range ((n + 1) * 2)).map (λ m, if m % 2 = 0 then 10^(m/2) else 0).sum

theorem count_primes_in_sequence : ∃! n, n = 1 ∧ prime (sequence 0)
  ∧ ∀ k > 0, ¬ prime (sequence k) :=
by
  sorry

end count_primes_in_sequence_l324_324011


namespace upper_limit_of_x_l324_324788

theorem upper_limit_of_x :
  ∀ x : ℤ, (0 < x ∧ x < 7) ∧ (0 < x ∧ x < some_upper_limit) ∧ (5 > x ∧ x > -1) ∧ (3 > x ∧ x > 0) ∧ (x + 2 < 4) →
  some_upper_limit = 2 :=
by
  intros x h
  sorry

end upper_limit_of_x_l324_324788


namespace find_f_15_l324_324836

noncomputable def f : ℝ → ℝ := sorry

lemma even_function (x : ℝ) : f x = f (-x) := sorry

lemma functional_equation (x : ℝ) : f (x + 2) = -f (x) := sorry

theorem find_f_15 : f 15 = 0 :=
by 
  have h1 : f 15 = -f 13, from functional_equation 13,
  have h2 : f 13 = f 11, from even_function 13,
  have h3 : f 11 = -f 9, from functional_equation 9,
  have h4 : f 9 = f 7, from even_function 9,
  have h5 : f 7 = -f 5, from functional_equation 5,
  have h6 : f 5 = f 3, from even_function 5,
  have h7 : f 3 = -f 1, from functional_equation 1,
  have h8 : f 1 = f (-1), from even_function 1,
  have h9 : -f 1 = f 1, from calc
    -f 1 = -f (-1) : by rw [even_function]
  ... = f 1 : by rw [neg_neg],
  have h10 : f 1 = 0, from eq_zero_of_neg_eq_self h9,
  calc
  f 15 = -f 13 : h1
  ... = f 11 : h2
  ... = -f 9 : h3
  ... = f 7 : h4
  ... = -f 5 : h5
  ... = f 3 : h6
  ... = -f 1 : h7
  ... = f (-1) : h8
  ... = f 1 : by rw [even_function]
  ... = 0 : h10

end find_f_15_l324_324836


namespace cost_per_meter_l324_324521

def length_of_plot : ℝ := 75
def cost_of_fencing : ℝ := 5300

-- Define breadth as a variable b
def breadth_of_plot (b : ℝ) : Prop := length_of_plot = b + 50

-- Calculate the perimeter given the known breadth
def perimeter (b : ℝ) : ℝ := 2 * length_of_plot + 2 * b

-- Define the proof problem
theorem cost_per_meter (b : ℝ) (hb : breadth_of_plot b) : 5300 / (perimeter b) = 26.5 := by
  -- Given hb: length_of_plot = b + 50, perimeter calculation follows
  sorry

end cost_per_meter_l324_324521


namespace parallelogram_area_l324_324993

noncomputable def area_parallelogram (b s θ : ℝ) : ℝ := b * (s * Real.sin θ)

theorem parallelogram_area : area_parallelogram 20 10 (Real.pi / 6) = 100 := by
  sorry

end parallelogram_area_l324_324993


namespace winner_beats_by_16_secons_l324_324219

-- Definitions of the times for mathematician and physicist
variables (x y : ℕ)

-- Conditions based on the given problem
def condition1 := 2 * y - x = 24
def condition2 := 2 * x - y = 72

-- The statement to prove
theorem winner_beats_by_16_secons (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x - 2 * y = 16 := 
sorry

end winner_beats_by_16_secons_l324_324219


namespace linda_lock_combinations_l324_324433

-- Define the sets of even and odd digits
def even_digits := {2, 4, 6}
def odd_digits := {1, 3, 5}

-- Define the conditions for a valid combination
def valid_combination (combo : List ℕ) : Prop :=
  combo.length = 6 ∧
  (∀ i, i < 5 → combo[i + 1] ≠ combo[i]) ∧
  (∀ i, i < 5 → (combo[i] ∈ even_digits → combo[i + 1] ∈ odd_digits))

-- Define the number of valid combinations
def num_valid_combinations : ℕ :=
  (even_digits.card + odd_digits.card) ^ 6

-- The theorem to prove
theorem linda_lock_combinations : num_valid_combinations = 1458 :=
  sorry

end linda_lock_combinations_l324_324433


namespace linear_function_quadrant_l324_324061

theorem linear_function_quadrant (x y : ℝ) : 
  y = 2 * x - 3 → ¬ ((x < 0 ∧ y > 0)) := 
sorry

end linear_function_quadrant_l324_324061


namespace complement_intersection_l324_324337

def U : Set ℕ := {x | -2 < x ∧ x < 6}

def A : Set ℕ := {2, 4}

def B : Set ℕ := {1, 3, 4}

theorem complement_intersection :
  ((U \ A) ∩ B) = {1, 3} :=
by
  sorry

end complement_intersection_l324_324337


namespace negation_of_P_l324_324114

theorem negation_of_P : ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x :=
by sorry

end negation_of_P_l324_324114


namespace Duke_three_pointers_impossible_l324_324811

theorem Duke_three_pointers_impossible (old_record : ℤ)
  (points_needed_to_tie : ℤ)
  (points_broken_record : ℤ)
  (free_throws : ℕ)
  (regular_baskets : ℕ)
  (three_pointers : ℕ)
  (normal_three_pointers_per_game : ℕ)
  (max_attempts : ℕ)
  (last_minutes : ℕ)
  (points_per_free_throw : ℤ)
  (points_per_regular_basket : ℤ)
  (points_per_three_pointer : ℤ) :
  free_throws = 5 → regular_baskets = 4 → normal_three_pointers_per_game = 2 → max_attempts = 10 → 
  points_per_free_throw = 1 → points_per_regular_basket = 2 → points_per_three_pointer = 3 →
  old_record = 257 → points_needed_to_tie = 17 → points_broken_record = 5 →
  (free_throws + regular_baskets + three_pointers ≤ max_attempts) →
  last_minutes = 6 → 
  ¬(free_throws + regular_baskets + (points_needed_to_tie + points_broken_record - 
  (free_throws * points_per_free_throw + regular_baskets * points_per_regular_basket)) / points_per_three_pointer ≤ max_attempts) := sorry

end Duke_three_pointers_impossible_l324_324811


namespace complex_exponent_evaluation_l324_324725

-- Given problem statement and conditions
def complex_num (x y : ℝ) : ℂ := x + y * I

theorem complex_exponent_evaluation (x y : ℝ) (i : ℂ) (i_unit : i = I) (h1 : complex_num x y = (1 + I)^2) : (i : ℂ) ^ (x + y) = -1 :=
by
  sorry

end complex_exponent_evaluation_l324_324725


namespace sufficient_condition_for_m_perp_beta_l324_324414

def plane : Type := sorry
def line : Type := sorry
def perp (a b : plane) : Prop := sorry
def perp (a b : line) : Prop := sorry

variables (α β : plane) (m n : line)

theorem sufficient_condition_for_m_perp_beta 
    (h1 : perp n α) (h2 : perp n β) (h3 : perp m α) 
    : perp m β := 
sorry

end sufficient_condition_for_m_perp_beta_l324_324414


namespace focus_and_directrix_l324_324036

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def parabola_directrix (p : ℝ) : ℝ := -(p / 2)

theorem focus_and_directrix (p : ℝ) (H : parabola_focus p = (2, 0)) : 
  p = 4 ∧ parabola_directrix p = -2 :=
begin
  sorry,
end

end focus_and_directrix_l324_324036


namespace fraction_simplification_l324_324582

theorem fraction_simplification (a : ℝ) (h1 : a > 1) (h2 : a ≠ 2 / Real.sqrt 3) : 
  (a^3 - 3 * a^2 + 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) / 
  (a^3 + 3 * a^2 - 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) = 
  ((a - 2) * Real.sqrt (a + 1)) / ((a + 2) * Real.sqrt (a - 1)) :=
by
  sorry

end fraction_simplification_l324_324582


namespace concatenated_square_is_perfect_l324_324251

theorem concatenated_square_is_perfect :
  ∃ a b : ℕ, a = 15 ∧ b = 25 ∧
  (a^2 = 225 ∧ b^2 = 625 ∧ (225625 = 475^2)) :=
by
  use 15, 25
  simp
  exact ⟨rfl, rfl, ⟨by norm_num, by norm_num, by norm_num⟩⟩

end concatenated_square_is_perfect_l324_324251


namespace rectangle_y_coordinate_l324_324910

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

theorem rectangle_y_coordinate (y : ℝ) :
  let p1 := (1 : ℝ, 0 : ℝ),
      p2 := (5 : ℝ, 0 : ℝ),
      p3 := (1 : ℝ, y),
      p4 := (5 : ℝ, y)
  in
  (∃ l : ℝ → ℝ, l 0 = 0 ∧ slope 0 0 3 1 = 1 / 3) →
  (∃ y_mid : ℝ, slope 0 0 3 y_mid = 1 / 3 ∧ y_mid = 1) →
  y = 1 :=
begin
  intros,
  sorry
end

end rectangle_y_coordinate_l324_324910


namespace product_greater_than_sum_l324_324861

theorem product_greater_than_sum (n : ℕ) 
    (x : Fin n → ℝ) 
    (h_pos : ∀ i, 0 < x i) 
    (h_lt_one : ∀ i, x i < 1) 
    (h_n : 2 ≤ n) :
    (∏ i, (1 - x i)) > 1 - (∑ i, x i) :=
  sorry

end product_greater_than_sum_l324_324861


namespace triangle_side_lengths_l324_324975

noncomputable def solve_triangle : ℕ × ℕ × ℕ :=
  let BN := 3;
  let NC := 4;
  let r := 3;
  let BC := BN + NC;
  let x := 21 in
  let AB := x + BN;
  let AC := x + NC;
  (AB, AC, BC)

theorem triangle_side_lengths :
  solve_triangle = (24, 25, 7) :=
by
  sorry

end triangle_side_lengths_l324_324975


namespace pen_cost_is_2_25_l324_324607

variables (p i : ℝ)

def total_cost (p i : ℝ) : Prop := p + i = 2.50
def pen_more_expensive (p i : ℝ) : Prop := p = 2 + i

theorem pen_cost_is_2_25 (p i : ℝ) 
  (h1 : total_cost p i) 
  (h2 : pen_more_expensive p i) : 
  p = 2.25 := 
by
  sorry

end pen_cost_is_2_25_l324_324607


namespace find_n_l324_324379

noncomputable def sum_first_terms {α : Type*} [ordered_ring α] (a d : α) (num_terms : ℕ) : α :=
  let first_terms := list.map (λ i, a + i * d) (list.range num_terms)
  in (list.sum first_terms)

noncomputable def sum_last_terms {α : Type*} [ordered_ring α] (a d : α) (n num_terms : ℕ) : α :=
  let last_terms := list.map (λ i, a + (n-1 - i) * d) (list.range num_terms)
  in (list.sum last_terms)

theorem find_n (a d : ℚ) :
  sum_first_terms a d 4 = 21 ∧ sum_last_terms a d 26 4 = 67 ∧ 
  (26 * (2 * a + 25 * d)) / 2 = 286 → 26 = 26 :=
by {
  sorry
}

end find_n_l324_324379


namespace permutation_moves_l324_324653

theorem permutation_moves (m n : ℕ) : (∀ k, 
  (∀ σ : Fin (m * n) → Fin (m * n), 
    (∃ H_moves V_moves : Fin m → (Fin n → Fin n), 
      σ = λ x, (Fin.mk (V_moves x / n)
        (by sorry)) (Fin.mk (H_moves (V_moves x / n) % n)
        (by sorry)) →
        ∃ i < k, application_of_moves (H_moves, V_moves, σ))
          → (∀ i < 2, ¬(∃ H_moves V_moves : Fin m → (Fin n → Fin n), 
            σ = λ x, (Fin.mk (V_moves x / n)
              (by sorry)) (Fin.mk (H_moves (V_moves x / n) % n)
              (by sorry))) ∨ (m = 1 ∨ n = 1)) ∧ k = 3) ∨ 
     (m = 1 ∨ n = 1) → k = 1)
:= 
begin 
  sorry
end

end permutation_moves_l324_324653


namespace area_of_shaded_region_l324_324887

theorem area_of_shaded_region (ABCD_is_square : ∀ (A B C D : ℝ × ℝ), 
  (ABCD ⟨(3, 0)⟩ ⟨(0, 3)⟩ ∧ ∀ x y z, (is_right_triangle x y z) := 
  let side_length := 3
  let BE := 4
  let shaded_area := 5 + 5 / 8 in
  ∃ A B E : ℝ × ℝ, 
  A = (0, 0) ∧ B = (3, 0) ∧ E.x ≠ B.x ∧ 
  is_square ABCD ∧ 
  is_right_triangle A B E ∧ 
  length_of_line B C = side_length ∧ 
  length_of_line B E = BE ∧
  area_of_shaded_part ABCD A E = shaded_area := sorry

end area_of_shaded_region_l324_324887


namespace find_f_3_l324_324497

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l324_324497


namespace xyz_value_l324_324557

theorem xyz_value (x y z : ℝ) (h1 : 2 * x + 3 * y + z = 13) 
                              (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := 
by 
  sorry

end xyz_value_l324_324557


namespace discontinuity_conditions_l324_324173

variable {α : Type*} [TopologicalSpace α] {β : Type*} [TopologicalSpace β]

def is_discontinuous_at (f : α → β) (M₀ : α) : Prop :=
¬ContinuousAt f M₀

theorem discontinuity_conditions (f : α → β) (M₀ : α) :
  is_discontinuous_at f M₀ ↔
  (¬∃ U : Set α, M₀ ∈ U ∧ ∀ x ∈ U, M₀ ≠ x → f x ≠ f M₀) ∨
  (∃ U : Set α, M₀ ∈ U ∧ ∀ x ∈ U, x ≠ M₀ → ¬ContinuousAt f x) ∨
  (∃ L, Tendsto f (𝓝 M₀) (𝓝 L) ∧ f M₀ ≠ L) ∨
  (∃ C : Set α, M₀ ∈ C ∧ ∀ M ∈ C, ¬ContinuousAt f M) := sorry

end discontinuity_conditions_l324_324173


namespace impossible_to_fill_boxes_with_condition_l324_324690

theorem impossible_to_fill_boxes_with_condition
  (a : ℕ → ℕ)
  (H : ∀ n ≥ 1, a n > (a (n - 1) + a (n + 1)) / 2) :
  false :=
begin
  sorry
end

end impossible_to_fill_boxes_with_condition_l324_324690


namespace coefficient_x3_l324_324810

theorem coefficient_x3 (x : ℕ) : 
  (nat.choose 5 3 - nat.choose 6 3) = -10 :=
by sorry

end coefficient_x3_l324_324810


namespace value_of_m_for_power_function_l324_324351

theorem value_of_m_for_power_function (m : ℝ) : (f : ℝ → ℝ) 
  (h1 : f = λ x, (2 * m + 3) * x ^ (m ^ 2 - 3))
  (h2 : ∃ k n : ℝ, f = λ x, k * x ^ n) :
  m = -1 :=
by
  sorry

end value_of_m_for_power_function_l324_324351


namespace exists_harmonic_conjugate_l324_324830

theorem exists_harmonic_conjugate (n : ℕ) (h : n = 2013) (A : Fin n → Point) (P : Point) (O : Point)
  (hP : P ≠ O) (circumcircle : Circle O) (hA : ∀ i, A i ∈ circumcircle) :
  ∃ Q : Point, Q ≠ P ∧ (∀ i, (distance (A i) P) / (distance (A i) Q) = (distance (A 0) P) / (distance (A 0) Q)) := 
sorry

end exists_harmonic_conjugate_l324_324830


namespace trig_ordering_l324_324418

-- Definitions
def a : ℝ := Real.sin (Real.pi * 33 / 180)
def b : ℝ := Real.cos (Real.pi * 55 / 180)
def c : ℝ := Real.tan (Real.pi * 35 / 180)

-- Theorem statement
theorem trig_ordering : c > b ∧ b > a :=
by
  -- Proof steps are omitted
  sorry

end trig_ordering_l324_324418


namespace probability_sum_of_three_dice_is_10_l324_324554

theorem probability_sum_of_three_dice_is_10 :
  (∃ (dice_rolls : finset (list ℕ)), 
    dice_rolls = {l | l.perm = [[1, 3, 6], [1, 4, 5], [2, 2, 6], [2, 3, 5], [2, 4, 4], [3, 3, 4]]} ∧
    ∑ roll in dice_rolls, roll.sum = 10 ∧
    ∑ roll in dice_rolls, 1 = 27) →
  let total_outcomes := 6 * 6 * 6 in
  (27 / total_outcomes) = (1 / 8) :=
begin
  sorry
end

end probability_sum_of_three_dice_is_10_l324_324554


namespace sum_series_eq_one_l324_324832

noncomputable def f (x : ℝ) : ℝ := ∫ t in 0..x, exp t * (cos t + sin t)
noncomputable def g (x : ℝ) : ℝ := ∫ t in 0..x, exp t * (cos t - sin t)

theorem sum_series_eq_one (a : ℝ) :
  (∑' n : ℕ, 1 ≤ n → 
    (∑' n in ∞, (e^(2 * a) / 
      ((f^[n] a)^2 + (g^[n] a)^2))) = 1 :=
begin
  sorry
end

end sum_series_eq_one_l324_324832


namespace leo_third_part_time_l324_324829

-- Definitions to represent the conditions
def total_time : ℕ := 120
def first_part_time : ℕ := 25
def second_part_time : ℕ := 2 * first_part_time

-- Proposition to prove
theorem leo_third_part_time :
  total_time - (first_part_time + second_part_time) = 45 :=
by
  sorry

end leo_third_part_time_l324_324829


namespace common_complex_root_l324_324581

theorem common_complex_root (m n : ℕ) (h_diff : m ≠ n) 
  (h_eq1 : ∀ (x : ℂ), x^(m+1) - x^n + 1 = 0)
  (h_eq2 : ∀ (x : ℂ), x^(n+1) - x^m + 1 = 0) :
  ∃ (x : ℂ), x = (1/2) + (real.sqrt 3)/2 * complex.I ∨ x = (1/2) - (real.sqrt 3)/2 * complex.I :=
by
  sorry

end common_complex_root_l324_324581


namespace negative_exponent_reciprocal_power_of_two_neg_two_l324_324591

theorem negative_exponent_reciprocal (a : ℝ) (n : ℕ) : a ≠ 0 → a ^ (-n : ℤ) = 1 / a ^ n := by
  sorry

theorem power_of_two_neg_two : (2 : ℝ) ^ (-2 : ℤ) = 1 / 4 := by
  have h := negative_exponent_reciprocal 2 2
  -- We need to show the assumption 2 ≠ 0 holds
  have h_nonzero : 2 ≠ 0 := by norm_num
  specialize h h_nonzero
  exact h

end negative_exponent_reciprocal_power_of_two_neg_two_l324_324591


namespace min_passengers_on_vehicle_with_no_adjacent_seats_l324_324266

-- Define the seating arrangement and adjacency rules

structure Seat :=
(row : Fin 2) (col : Fin 5)

def adjacent (a b : Seat) : Prop :=
(a.row = b.row ∧ (a.col = b.col + 1 ∨ a.col + 1 = b.col)) ∨
(a.col = b.col ∧ (a.row = b.row + 1 ∨ a.row + 1 = b.row))

def valid_seating (seated : List Seat) : Prop :=
∀ (i j : Seat), i ∈ seated → j ∈ seated → adjacent i j → false

def min_passengers : ℕ :=
5

theorem min_passengers_on_vehicle_with_no_adjacent_seats :
∃ seated : List Seat, valid_seating seated ∧ List.length seated = min_passengers :=
sorry

end min_passengers_on_vehicle_with_no_adjacent_seats_l324_324266


namespace triangle_bisection_l324_324995

theorem triangle_bisection
    (A B C P Q R S T L : Point)
    (triangle_ABC : Triangle A B C)
    (P_on_AB : PointOnSegment P A B)
    (S_on_AC : PointOnSegment S A C)
    (T_on_BC : PointOnSegment T B C)
    (AP_eq_AS : dist A P = dist A S)
    (BP_eq_BT : dist B P = dist B T)
    (circumcircle_PST : Circle (circumcenter P S T) (circumradius P S T))
    (Q_on_AB' : PointOnCircle Q circumcircle_PST)
    (R_on_BC' : PointOnCircle R circumcircle_PST)
    (L_on_PS_QR : meets_at L (line_through P S) (line_through Q R))
    (C_on_CL : PointOnLine C L) :
    bisects (line_through C L) P Q := 
sorry

end triangle_bisection_l324_324995


namespace lizard_eyes_l324_324822

theorem lizard_eyes (E W S : Nat) 
  (h1 : W = 3 * E) 
  (h2 : S = 7 * W) 
  (h3 : E = S + W - 69) : 
  E = 3 := 
by
  sorry

end lizard_eyes_l324_324822


namespace find_f_3_l324_324496

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l324_324496


namespace hyperbola_equation1_hyperbola_equation2_l324_324694

section Problem1

open Real

-- Define the constants and conditions related to the first problem
def focus_hyperbola1 := (± 2 * sqrt 5, 0)
def passes_through := (3 * sqrt 2, 2)

-- Define the target equation of the hyperbola for the first problem
theorem hyperbola_equation1 : 
  ∃ a b, (a = 12) ∧ (b = 8) ∧ (a > 0) ∧ (b > 0) ∧
  (∀ x y, (x, y) = passes_through → ((x^2 / a) - (y^2 / b) = 1)) ∧
  (∀ x y, (x^2 / 16) - (y^2 / 4) = 1 → abs y < foo) := sorry

end Problem1

section Problem2

open Real

-- Define the constants and conditions related to the second problem
def ellipse_equation := (3 * x^2 + 13 * y^2 = 39)
def asymptotes := (y = ± (x / 2))

-- Define the target equation of the hyperbola for the second problem
theorem hyperbola_equation2 :
  ∃ a b, (a = 8) ∧ (b = 2) ∧ (a > 0) ∧ (b > 0) ∧
  (∀ x y, (3*x^2 + 13*y^2 = 39) → (focus x y = ± sqrt 10, 0)) ∧
  (∀ x y, ((y = x / 2) ∨ (y = -x / 2)) → ((x^2 / a) - (y^2 / b)) = 1) := sorry

end Problem2

end hyperbola_equation1_hyperbola_equation2_l324_324694


namespace sector_angle_l324_324289

noncomputable def cone_angle : Float := 360 - ((2 * 10 * 180) / (5 * Real.sqrt 13))

theorem sector_angle:
  (volume_cone : ℝ) 
  (radius_base_cone : ℝ) 
  (height_cone : ℝ)
  (slant_height : ℝ) :
  volume_cone = 500 * Real.pi ∧
  radius_base_cone = 10 ∧
  height_cone = 15 ∧
  slant_height = 5 * Real.sqrt 13 →
  cone_angle = 130.817 :=
sorry

end sector_angle_l324_324289


namespace vertical_angles_are_equal_l324_324532

theorem vertical_angles_are_equal (A B C D : Type) (angle1 : angle A B C) (angle2 : angle C D A) 
  (h : vertical_angles angle1 angle2) : angle1 = angle2 := 
sorry

end vertical_angles_are_equal_l324_324532


namespace measure_8_liters_with_buckets_l324_324007

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l324_324007


namespace first_year_payment_l324_324951

theorem first_year_payment (X : ℝ) (second_year : ℝ) (third_year : ℝ) (fourth_year : ℝ) 
    (total_payments : ℝ) 
    (h1 : second_year = X + 2)
    (h2 : third_year = X + 5)
    (h3 : fourth_year = X + 9)
    (h4 : total_payments = X + second_year + third_year + fourth_year) :
    total_payments = 96 → X = 20 :=
by
    sorry

end first_year_payment_l324_324951


namespace tom_gas_spending_l324_324924

-- Defining the conditions given in the problem
def miles_per_gallon := 50
def miles_per_day := 75
def gas_price := 3
def number_of_days := 10

-- Defining the main theorem to be proven
theorem tom_gas_spending : 
  (miles_per_day * number_of_days) / miles_per_gallon * gas_price = 45 := 
by 
  sorry

end tom_gas_spending_l324_324924


namespace focus_of_parabola_l324_324884

-- Definitions for the problem
def parabola_eq (x y : ℝ) : Prop := y = 2 * x^2

def general_parabola_form (x y h k p : ℝ) : Prop :=
  4 * p * (y - k) = (x - h)^2

def vertex_origin (h k : ℝ) : Prop := h = 0 ∧ k = 0

-- Lean statement asserting that the focus of the given parabola is (0, 1/8)
theorem focus_of_parabola : ∃ p : ℝ, parabola_eq x y → general_parabola_form x y 0 0 p ∧ p = 1/8 := by
  sorry

end focus_of_parabola_l324_324884


namespace competition_scores_days_l324_324883

theorem competition_scores_days (n k : ℕ) (h_n : n ≥ 2) (h_k : k ≥ 2) 
  (h_scores : ∀ i, 1 ≤ i ∧ i ≤ n → 1 ≤ ∑ i in finset.range k, i ∧ (1 ≤ ∑ i in finset.range k, (26 / k))):
  (n, k) = (25, 2) ∨ (n, k) = (12, 4) ∨ (n, k) = (3, 13) :=
by
  sorry

end competition_scores_days_l324_324883


namespace lcm_hcf_product_l324_324202

theorem lcm_hcf_product (lcm hcf a b : ℕ) (hlcm : lcm = 2310) (hhcf : hcf = 30) (ha : a = 330) (eq : lcm * hcf = a * b) : b = 210 :=
by {
  sorry
}

end lcm_hcf_product_l324_324202


namespace right_triangle_not_one_altitude_l324_324796

-- Define what an altitude is formally.
def is_altitude (triangle : Type) (a b c : triangle) (h_ab : a ≠ b) 
  (line_bc : Set (triangle)) (is_perpendicular : ∀ (l : Set (triangle)), perpendicular l line_bc) : Prop :=
  ∃ (p : triangle), p ∈ line_bc ∧ is_perpendicular a p

-- Definition of a right triangle and its properties
structure RightTriangle (triangle : Type) :=
  (a b c : triangle)
  (right_angle : ∃ (line_bc : Set (triangle)), is_perpendicular c line_bc)

-- Statement: In a right triangle, show that there are not only one altitude
theorem right_triangle_not_one_altitude (triangle : Type) (T : RightTriangle triangle) : 
  ¬(∃ (alt : Set (triangle)), ∀ (a b c : triangle), alt = {is_altitude T.a T.b T.c} → alt.cardinality = 1) :=
by 
  sorry

end right_triangle_not_one_altitude_l324_324796


namespace trapezoid_area_ratio_l324_324182

-- Define the original trapezoid with bases a, b and height h
variables (a b h : ℝ)

-- Define the area of the original trapezoid
def trapezoid_area (a b h : ℝ) : ℝ :=
  (a + b) / 2 * h

-- Define the height of the intuitive diagram
def intuitive_diagram_height (h : ℝ) : ℝ :=
  (h / 2) * (Real.sin (Real.pi / 4))

-- Define the area of the intuitive diagram
def intuitive_diagram_area (a b h : ℝ) : ℝ :=
  (a + b) / 2 * (intuitive_diagram_height h)

-- Prove the ratio of the areas is sqrt(2) / 4
theorem trapezoid_area_ratio :
  ∀ a b h : ℝ, 
    h > 0 → 
    intuitive_diagram_area a b h / trapezoid_area a b h = Real.sqrt 2 / 4 :=
by
  intros a b h h_pos
  unfold intuitive_diagram_area trapezoid_area intuitive_diagram_height
  calc
    ((a + b) / 2 * ((h / 2) * Real.sin (Real.pi / 4))) / ((a + b) / 2 * h)
      = ((a + b) / 2 * ((h / 2) * (Real.sqrt 2 / 2))) / ((a + b) / 2 * h) : by rw [Real.sin_pi_div_four]
  ... = ((a + b) / 2 * (h * (Real.sqrt 2 / 4))) / ((a + b) / 2 * h) : by ring
  ... = (h * (Real.sqrt 2 / 4)) / h : by rw [mul_div_mul_left (a + b) 2 h h_pos, mul_comm  (Real.sqrt 2) 4]
  ... = Real.sqrt 2 / 4 : by rw [h_pos, mul_div_cancel_left ((Real.sqrt 2) / 4) h h_pos]

end trapezoid_area_ratio_l324_324182


namespace Verify_N_l324_324276

-- Definitions for the problem
def N : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![1, 2]]
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![6, 12], ![3, 6]]

-- Statement to be proved
theorem Verify_N :
  N^3 - (3:N).smul N^2 + (3:N).smul N = M :=
by
  sorry

end Verify_N_l324_324276


namespace angle_measure_in_triangle_l324_324791

theorem angle_measure_in_triangle
  {A B C a b c : ℝ} (h1 : a = c / (2 * b) / (2 * tan B))
  (h2 : 1 + tan A / tan B + 2 * c / b = 0) :
  A = 2 * Real.pi / 3 := 
sorry

end angle_measure_in_triangle_l324_324791


namespace speed_of_train_is_72_km_per_hr_l324_324230

-- Definition of the problem parameters
def length_of_train : ℝ := 200
def time_to_cross_bridge : ℝ := 16.5986721062315
def length_of_bridge : ℝ := 132
def conversions : ℝ := 3.6

-- Definition of the total distance covered by the train
def total_distance := length_of_train + length_of_bridge

-- Definition of the speed in meters per second
def speed_m_per_s := total_distance / time_to_cross_bridge

-- Proof statement that speed in km/hr is equal to 72
theorem speed_of_train_is_72_km_per_hr :
  (speed_m_per_s * conversions) = 72 := 
sorry

end speed_of_train_is_72_km_per_hr_l324_324230


namespace max_license_plates_l324_324360
open Nat

theorem max_license_plates (k : ℕ) (h1 : k ≤ 10^6)
(h2 : ∀ (plate1 plate2 : Fin 1000000) (hplate1 : plate1 < k) (hplate2 : plate2 < k), 
  (plate1 ≠ plate2 → (∃ i : Fin 6, (plate1 % 10^(i+1) / 10^i) ≠ (plate2 % 10^(i+1) / 10^i)))) :
  k ≤ 100000 :=
by
  sorry

end max_license_plates_l324_324360


namespace behavior_of_function_l324_324685

noncomputable def is_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f y < f x

noncomputable def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x < f y

theorem behavior_of_function {n : ℕ} :
  (∀ x : ℝ, x > 0 → is_decreasing (λ x, x^(-n : ℤ)) {x | x > 0}) ∧
  (∀ x : ℝ, x < 0 → is_increasing (λ x, x^(-n : ℤ)) {x | x < 0}) ∧
  (∀ x : ℝ, x > 0 → is_decreasing (λ x, x^(-n : ℤ)) {x | x > 0}) ∧
  (∀ x : ℝ, x < 0 → is_increasing (λ x, x^(-n : ℤ)) {x | x < 0}) :=
sorry

end behavior_of_function_l324_324685


namespace largest_angle_in_triangle_l324_324898

theorem largest_angle_in_triangle (y : ℝ) (h : 60 + 70 + y = 180) :
    70 > 60 ∧ 70 > y :=
by {
  sorry
}

end largest_angle_in_triangle_l324_324898


namespace polygons_sides_sum_l324_324688

theorem polygons_sides_sum : 
  let triangle := 3 in
  let square := 4 in
  let pentagon := 5 in
  let hexagon := 6 in
  let heptagon := 7 in
  let octagon := 8 in
  let nonagon := 9 in
  triangle + square + pentagon + hexagon + heptagon + octagon + nonagon 
  - 2 * 1 - 5 * 2 = 30 :=
by 
  let triangle := 3 
  let square := 4 
  let pentagon := 5 
  let hexagon := 6 
  let heptagon := 7 
  let octagon := 8 
  let nonagon := 9 
  apply sorry

end polygons_sides_sum_l324_324688


namespace quadratic_roots_real_coefficients_complex_conjugate_cubed_real_l324_324785

theorem quadratic_roots_real_coefficients_complex_conjugate_cubed_real
  (a b c : ℝ) (x₁ x₂ : ℂ)
  (h_eq : a ≠ 0)
  (h_root1 : a * x₁^2 + b * x₁ + c = 0)
  (h_root2 : a * x₂^2 + b * x₂ + c = 0)
  (h_conjugate : x₂ = complex.conj x₁)
  (h_cube_real : (x₁^3).im = 0) :
  a * c / b^2 = 1 := 
sorry

end quadratic_roots_real_coefficients_complex_conjugate_cubed_real_l324_324785


namespace flour_per_new_crust_l324_324069

-- Definitions based on conditions
def num_initial_pie_crusts := 40
def flour_per_initial_crust := (1 / 8 : ℚ)
def total_flour_used := 5
def num_new_pie_crusts := 25

-- Theorem to prove the amount of flour per new pie crust
theorem flour_per_new_crust :
  (num_initial_pie_crusts * flour_per_initial_crust = total_flour_used) →
  (num_new_pie_crusts * flour_per_initial_crust = total_flour_used / num_new_pie_crusts) → 
  (flour_per_initial_crust = 1 / 5) :=
begin
  intros h1 h2,
  have h : total_flour_used / num_new_pie_crusts = 1 / 5, sorry,
  exact h,
end

end flour_per_new_crust_l324_324069


namespace limit_exp_sequence_l324_324839

noncomputable theory

open Filter
open_locale Topology

theorem limit_exp_sequence (a : ℝ) (hx : ∀ n : ℕ, x n ∈ ℚ) (hlim : tendsto x at_top (𝓝 0)) (ha : 0 < a) :
  tendsto (λ n, a ^ (x n)) at_top (𝓝 1) :=
sorry

end limit_exp_sequence_l324_324839


namespace find_f_of_3_l324_324511

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l324_324511


namespace half_abs_sum_diff_squares_cubes_l324_324931

theorem half_abs_sum_diff_squares_cubes (a b : ℤ) (h1 : a = 21) (h2 : b = 15) :
  (|a^2 - b^2| + |a^3 - b^3|) / 2 = 3051 := by
  sorry

end half_abs_sum_diff_squares_cubes_l324_324931


namespace percentage_green_ducks_in_smaller_pond_l324_324366

theorem percentage_green_ducks_in_smaller_pond
  (total_ducks_smaller_pond : ℕ)
  (total_ducks_larger_pond : ℕ)
  (percentage_green_ducks_larger_pond : ℕ)
  (percentage_green_ducks_total : ℕ)
  (total_green_ducks : ℕ)
  (h1 : total_ducks_smaller_pond = 20)
  (h2 : total_ducks_larger_pond = 80)
  (h3 : percentage_green_ducks_larger_pond = 15)
  (h4 : percentage_green_ducks_total = 16)
  (h5 : total_green_ducks = 16) :
  ∃ x : ℕ, (x / 100) * total_ducks_smaller_pond + (percentage_green_ducks_larger_pond / 100) * total_ducks_larger_pond = total_green_ducks ∧ x = 20 := 
by {
  let x := 20,
  use x,
  split,
  {
    calc
      (x / 100) * total_ducks_smaller_pond + (percentage_green_ducks_larger_pond / 100) * total_ducks_larger_pond
      = (20 / 100) * 20 + (15 / 100) * 80 : by rw [h1, h2, h3]
      ... = 4 + 12
      ... = 16,
  },
  {
    exact rfl,
  },
}

end percentage_green_ducks_in_smaller_pond_l324_324366


namespace max_three_numbers_condition_l324_324441

theorem max_three_numbers_condition (n : ℕ) 
  (x : Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → (x i)^2 > (x j) * (x k)) : n ≤ 3 := 
sorry

end max_three_numbers_condition_l324_324441


namespace find_f_three_l324_324486

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l324_324486


namespace sin_neg_270_eq_one_l324_324701

theorem sin_neg_270_eq_one : Real.sin (-(270 : ℝ) * (Real.pi / 180)) = 1 := by
  sorry

end sin_neg_270_eq_one_l324_324701


namespace problem_statement_l324_324124

section
variable (f : ℝ → ℝ)
variable (h : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)

def g (x : ℝ) := (1 / x) * f x

theorem problem_statement :
  g f 0.25 > g f 2 ∧ g f 2 > g f 5 :=
begin
  sorry
end

end

end problem_statement_l324_324124


namespace lassis_from_12_mangoes_l324_324678

-- Conditions as definitions in Lean 4
def total_mangoes : ℕ := 12
def damaged_mango_ratio : ℕ := 1 / 6
def lassis_per_pair_mango : ℕ := 11

-- Equation to calculate the lassis
theorem lassis_from_12_mangoes : (total_mangoes - total_mangoes / 6) / 2 * lassis_per_pair_mango = 55 :=
by
  -- calculation steps should go here, but are omitted as per instructions
  sorry

end lassis_from_12_mangoes_l324_324678


namespace range_a_l324_324955

variable (x a : ℝ)

def p : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_a (h : ¬p → ¬q) : 0 ≤ a ∧ a ≤ 1/2 :=
  sorry

end range_a_l324_324955


namespace average_typing_speed_l324_324468

theorem average_typing_speed :
  let rudy := 64
  let joyce := 76
  let gladys := 91
  let lisa := 80
  let mike := 89
  let total := rudy + joyce + gladys + lisa + mike
  let average := total / 5
  in average = 80 := by
  sorry

end average_typing_speed_l324_324468


namespace seq_geometric_and_formula_find_min_n_l324_324730

-- Define the sequence a_n
def S (n : ℕ) : ℕ := sorry -- Sum of the first n terms

-- Condition given in the problem
axiom condition (n : ℕ) (h : n > 0) : S n + n = 2 * (a n)

-- Sequence a_n derived formula
def a (n : ℕ) : ℕ := 2^n - 1

-- Define b_n
def b (n : ℕ) : ℕ := (2 * n + 1) * a n + 2 * n + 1

-- Sum of the first n terms of sequence {b_n}
def T (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), b i

-- Statement to prove the sequence {a_n + 1} is geometric and find the general formula for a_n
theorem seq_geometric_and_formula :
  (∀ n : ℕ, n > 0 → a (n + 1) + 1 = 2 * (a n + 1)) ∧ (∀ (n : ℕ), n > 0 → a n = 2^n - 1) :=
sorry

-- Statement to find the minimum value of n satisfying the inequality
theorem find_min_n : ∃ n : ℕ, ∀ (h : n > 0), 2 ^ (n + 1) > 2010 ∧ (∀ m : ℕ, m < n → 2 ^ (m + 1) ≤ 2010) :=
sorry

end seq_geometric_and_formula_find_min_n_l324_324730


namespace prob_pass_exactly_once_l324_324227

-- Defining the probability of passing a single test
def prob_single_pass : ℚ := 1 / 3

-- Definition of passing exactly once in three tests
theorem prob_pass_exactly_once :
  -- Probability of passing exactly once in three tests
  ((3.choose 1) * prob_single_pass * (1 - prob_single_pass) ^ 2) = 4 / 9 := by
  sorry

end prob_pass_exactly_once_l324_324227


namespace james_hours_worked_l324_324268

variable (x : ℝ) (y : ℝ)

theorem james_hours_worked (h1: 18 * x + 16 * (1.5 * x) = 40 * x + (y - 40) * (2 * x)) : y = 41 :=
by
  sorry

end james_hours_worked_l324_324268


namespace radius_of_sphere_tangent_to_cube_faces_and_edges_correct_l324_324264

noncomputable def radius_of_sphere_tangent_to_cube_faces_and_edges : ℝ :=
  2 - Real.sqrt 2

theorem radius_of_sphere_tangent_to_cube_faces_and_edges_correct :
    let A := (0, 0, 0) : ℝ × ℝ × ℝ;
    let B := (1, 1, 1) : ℝ × ℝ × ℝ;
    ∃ r : ℝ, 
      (∀ (x y z : ℝ), x ∈ {0, 1} → y ∈ {0, 1} → z ∈ {0, 1} → 
        (x = 0 → |(x, y, z).1 - r| = r) ∧
        (y = 0 → |(x, y, z).2 - r| = r) ∧
        (z = 0 → |(x, y, z).3 - r| = r) ∧
        (x = 1 → |(x, y, z).1 + r - 1| = r) ∧
        (y = 1 → |(x, y, z).2 + r - 1| = r) ∧
        (z = 1 → |(x, y, z).3 + r - 1| = r)) ∧
      r = radius_of_sphere_tangent_to_cube_faces_and_edges :=
  by
    let A := (0, 0, 0) : ℝ × ℝ × ℝ
    let B := (1, 1, 1) : ℝ × ℝ × ℝ 
    use 2 - Real.sqrt 2
    sorry

end radius_of_sphere_tangent_to_cube_faces_and_edges_correct_l324_324264


namespace staffing_ways_l324_324677

def total_resumes : ℕ := 30
def unsuitable_resumes : ℕ := 10
def suitable_resumes : ℕ := total_resumes - unsuitable_resumes
def position_count : ℕ := 5

theorem staffing_ways :
  20 * 19 * 18 * 17 * 16 = 1860480 := by
  sorry

end staffing_ways_l324_324677


namespace quadratic_has_real_root_l324_324559

theorem quadratic_has_real_root (a b : ℝ) : ¬ (∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 := 
by
  sorry

end quadratic_has_real_root_l324_324559


namespace xyz_inequality_l324_324411

-- Definitions for the conditions and the statement of the problem
theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h_ineq : x * y * z ≥ x * y + y * z + z * x) : 
  x * y * z ≥ 3 * (x + y + z) :=
by
  sorry

end xyz_inequality_l324_324411


namespace toy_car_wheel_circumference_ratio_approx_l324_324656

noncomputable def toy_car_wheel_circumference_ratio (d : ℝ) (C_actual : ℝ) : ℝ :=
  C_actual / d

theorem toy_car_wheel_circumference_ratio_approx :
  toy_car_wheel_circumference_ratio 30 94.2 ≈ 3.14 :=
by
  sorry

end toy_car_wheel_circumference_ratio_approx_l324_324656


namespace problem1_minimum_m_problem2_sin_C_l324_324325

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x^2 - sqrt 3

-- Problem 1: Prove minimum value of m is -π/3
theorem problem1_minimum_m :
  ∀ m : ℝ,
  (∀ x : ℝ, m ≤ x → x ≤ π/2 → f x ∈ set.Icc (-sqrt 3) 2) → m = -π/3 :=
sorry

-- Problem 2: Prove sin C = sqrt(21)/7 in the given triangle conditions
theorem problem2_sin_C (A B C : ℝ) :
  2 * sin (A / 2 + π / 3) = 2 →
  sin B = (3 * sqrt 3 / 4) * cos C →
  sin C = sqrt 21 / 7 :=
sorry

end problem1_minimum_m_problem2_sin_C_l324_324325


namespace domain_of_v_l324_324933

theorem domain_of_v :
  { p : ℝ × ℝ | let x := p.1, y := p.2 in v x y = 1 / (Real.sqrt (x + y)) } =
  { p : ℝ × ℝ | let x := p.1, y := p.2 in y > -x } :=
sorry

end domain_of_v_l324_324933


namespace projection_of_right_triangle_l324_324038

variable {α : Type*} [LinearOrderedField α]

/--
Given a right-angled triangle ABC with hypotenuse BC lying in a plane α,
and the vertex A outside of plane α, the projection of triangle ABC on plane α
is a line segment or an obtuse triangle.
-/
theorem projection_of_right_triangle (ABC : Triangle α) (BC_in_plane : lies_in_plane BC α) (A_not_in_plane : ¬ lies_in_plane A α) :
  is_projection_on_plane α ABC (line_segment) ∨ is_projection_on_plane α ABC (obtuse_triangle) :=
sorry

end projection_of_right_triangle_l324_324038


namespace robber_moves_10000_police_catches_robber_l324_324223

-- Define the structure of the board
structure Board :=
  (size : ℕ)
  (initial_police_pos : ℕ × ℕ)
  (initial_robber_pos : ℕ × ℕ)
  (police_special_move : ((ℕ × ℕ) → (ℕ × ℕ)))

def move := (x y : ℕ × ℕ) → Prop

-- Define properties of the game
axiom move_down : ∀ (x y : ℕ × ℕ), move (x, y + 1)
axiom move_right : ∀ (x y : ℕ × ℕ), move (x + 1, y)
axiom move_diagonal : ∀ (x y : ℕ × ℕ), move (x - 1, y - 1)

axiom police_special_move :
  (police_position : ℕ × ℕ) → 
  (police_position = (2001, 2001)) →
  (2001 × 2001 → 1 × 1)

noncomputable def initial_positions (board : Board) : Prop :=
  board.initial_police_pos = (1001, 1001) ∧
  board.initial_robber_pos = (1001, 1002)

-- Define the conditions for capturing the robber
def police_captures_robber (board : Board) (police_pos : ℕ × ℕ) (robber_pos : ℕ × ℕ) : Prop :=
  police_pos = robber_pos

-- Prove that the robber can move at least 10,000 times before being captured
theorem robber_moves_10000 (board : Board) :
  ∃ moves : ℕ, moves ≥ 10000 ∧ ∀ p_pos r_pos, p_pos ≠ r_pos :=
sorry

-- Prove that the policeman has a strategy to eventually catch the robber
theorem police_catches_robber (board : Board) :
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), ∀ police_pos, ∃ n, police_captures_robber board (strategy police_pos) board.initial_robber_pos :=
sorry

end robber_moves_10000_police_catches_robber_l324_324223


namespace sum_of_x_coordinates_on_parabola_l324_324316

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1

-- Define the points P and Q on the parabola
variables {x1 x2 : ℝ}

-- The Lean theorem statement: 
theorem sum_of_x_coordinates_on_parabola 
  (h1 : parabola x1 = 1) 
  (h2 : parabola x2 = 1) : 
  x1 + x2 = 2 :=
sorry

end sum_of_x_coordinates_on_parabola_l324_324316


namespace pizza_toppings_count_l324_324620

theorem pizza_toppings_count :
  let toppings := 8 in
  let one_topping_pizzas := toppings in
  let two_topping_pizzas := (toppings.choose 2) in
  let three_topping_pizzas := (toppings.choose 3) in
  one_topping_pizzas + two_topping_pizzas + three_topping_pizzas = 92 :=
by sorry

end pizza_toppings_count_l324_324620


namespace max_F_l324_324409

def f (x a b : ℝ) : ℝ := (x + a) * (x + b)

def sum_condition (x : ℕ → ℝ) (n : ℕ) : Prop := (∀ i, 0 ≤ x i) ∧ ((Finset.range n).sum x = 1)

def F (x : ℕ → ℝ) (a b : ℝ) (n : ℕ) : ℝ := 
  (Finset.range n).sum (λ i, (Finset.range i).sum (λ j, min (f (x i) a b) (f (x j) a b)))

theorem max_F (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (x : ℕ → ℝ) (n : ℕ) (h₃ : sum_condition x n) : 
  F x a b n ≤ (n - 1) / (2 * n) * (n * a + 1) * (n * b + 1) := sorry

end max_F_l324_324409


namespace daughterAgeThreeYearsFromNow_l324_324959

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end daughterAgeThreeYearsFromNow_l324_324959


namespace coin_game_goal_l324_324204

theorem coin_game_goal (a b : ℕ) (h_diff : a ≤ 3 * b ∧ b ≤ 3 * a) (h_sum : (a + b) % 4 = 0) :
  ∃ x y p q : ℕ, (a + 2 * x - 2 * y = 3 * (b + 2 * p - 2 * q)) ∨ (a + 2 * y - 2 * x = 3 * (b + 2 * q - 2 * p)) :=
sorry

end coin_game_goal_l324_324204


namespace set_elements_equality_problem_solution_l324_324917

open Set

theorem set_elements_equality (a b : ℝ) (h : {a, b / a, 1} = {a ^ 2, a + b, 0}) :
  a ≠ 0 ∧ a = -1 ∧ b = 0 :=
by
  sorry

theorem problem_solution (a b : ℝ) (h : {a, b / a, 1} = {a ^ 2, a + b, 0}) :
  a ^ 2013 + b ^ 2013 = -1 :=
by
  have h₁ : a ≠ 0 ∧ a = -1 ∧ b = 0 := set_elements_equality a b h
  cases h₁ with a_nonzero ha_bo
  cases ha_bo with a_neg_one b_zero
  rw [a_neg_one, b_zero]
  norm_num

end set_elements_equality_problem_solution_l324_324917


namespace number_of_sixes_l324_324967

-- Definitions of the conditions
def total_runs : ℕ := 120
def boundaries : ℕ := 3
def boundary_runs : ℕ := boundaries * 4  -- Each boundary gives 4 runs
def running_percentage : ℚ := 50 / 100   -- 50%

-- Definition of the question as a theorem statement
theorem number_of_sixes (runs_from_wickets : ℕ := (total_runs * (running_percentage : ℚ)).natAbs) 
                        (runs_from_boundaries_and_sixes : ℕ := total_runs - runs_from_wickets)
                        (sixes_runs : ℕ := runs_from_boundaries_and_sixes - boundary_runs)
                        (num_sixes : ℕ := sixes_runs / 6) : 
  num_sixes = 8 :=
sorry

end number_of_sixes_l324_324967


namespace number_of_valid_partitions_l324_324295

-- Define the condition to check if a list of integers has all elements same or exactly differ by 1
def validPartition (l : List ℕ) : Prop :=
  l ≠ [] ∧ (∀ (a b : ℕ), a ∈ l → b ∈ l → a = b ∨ a = b + 1 ∨ b = a + 1)

-- Count valid partitions of n (integer partitions meeting the given condition)
noncomputable def countValidPartitions (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

-- Main theorem
theorem number_of_valid_partitions (n : ℕ) : countValidPartitions n = n :=
by
  sorry

end number_of_valid_partitions_l324_324295


namespace percentage_profit_is_35_l324_324992

-- Define the conditions
def initial_cost_price : ℝ := 100
def markup_percentage : ℝ := 0.5
def discount_percentage : ℝ := 0.1
def marked_price : ℝ := initial_cost_price * (1 + markup_percentage)
def selling_price : ℝ := marked_price * (1 - discount_percentage)

-- Define the statement/proof problem
theorem percentage_profit_is_35 :
  (selling_price - initial_cost_price) / initial_cost_price * 100 = 35 := by 
  sorry

end percentage_profit_is_35_l324_324992


namespace find_f_of_3_l324_324475

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l324_324475


namespace number_of_basic_events_l324_324288

-- Define the elements in the bag
def elements : List ℕ := [1, 2, 3]

-- Define what it means to draw two balls successively without replacement
def draw_two_balls (s : List ℕ) : List (ℕ × ℕ) :=
  (s.product s).filter (λ p, p.1 ≠ p.2)

-- Define the main theorem statement
theorem number_of_basic_events : (draw_two_balls elements).length = 6 := by
  sorry

end number_of_basic_events_l324_324288


namespace find_f_three_l324_324487

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l324_324487


namespace simplest_quadratic_radical_l324_324184

-- Define variables used in the proof
variables (x a : ℝ)

def is_simplest_quadratic_radical (e : ℝ) : Prop :=
  ∀ (x a : ℝ), e = sqrt (x^2 + 1) → 
                sqrt(0.2) > e ∧
                sqrt(8 * a) > e ∧
                sqrt(9) > e

theorem simplest_quadratic_radical : is_simplest_quadratic_radical (sqrt (x^2 + 1)) :=
by sorry

end simplest_quadratic_radical_l324_324184


namespace find_number_l324_324972

theorem find_number (x : ℝ) (h : x = 0.17999999999999997) (hx : ∃ number, number / x = 0.05) : ∃ number, number = 0.009 :=
by
  cases hx with number hnumber
  have hx_mul : number = 0.05 * x := by
    rw [h] at hnumber
    linarith
  use number
  rw [hx_mul, ← h]
  norm_num
  intros
  sorry

end find_number_l324_324972


namespace one_number_is_zero_l324_324162

variable {a b c : ℤ}
variable (cards : Fin 30 → ℤ)

theorem one_number_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (h_cards : ∀ i : Fin 30, cards i = a ∨ cards i = b ∨ cards i = c)
    (h_sum_zero : ∀ (S : Finset (Fin 30)) (hS : S.card = 5),
        ∃ T : Finset (Fin 30), T.card = 5 ∧ (S ∪ T).sum cards = 0) :
    b = 0 := 
sorry

end one_number_is_zero_l324_324162


namespace max_a4_a5_product_l324_324054

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a_pos : ∀ n, a n > 0)
variable (sum_eight : ∑ i in Finset.range 8, a i = 40)
variable (arith_seq : ∀ n, a (n + 1) = a n + d)

theorem max_a4_a5_product : 
  (∃ a : ℕ → ℝ, 
   ∃ d : ℝ,
   (∀ n, a n > 0) ∧
   (∑ i in Finset.range 8, a i = 40) ∧ 
   (∀ n, a (n + 1) = a n + d) ∧
   ∀ x y : ℝ, x = a 3 ∧ y = a 4 → x * y ≤ 25) :=
sorry

end max_a4_a5_product_l324_324054


namespace Dongdong_test_score_l324_324593

theorem Dongdong_test_score (a b c : ℕ) (h1 : a + b + c = 280) : a ≥ 94 ∨ b ≥ 94 ∨ c ≥ 94 :=
by
  sorry

end Dongdong_test_score_l324_324593


namespace ratio_B_to_C_l324_324867

/-- Shares of A, B, and C -/
def A_share : ℕ := 408
def B_share : ℕ := 102
def C_share : ℕ := 68

/-- Equation representing the total sum of shares -/
def total_share : ℕ := A_share + B_share + C_share

/-- A's share is 2/3 of B's share -/
def A_share_eq : Prop := A_share = 2 * B_share / 3

/-- Given conditions -/
variables (h1 : total_share = 578) (h2 : A_share_eq)

theorem ratio_B_to_C : B_share / Nat.gcd B_share C_share = 3 ∧ C_share / Nat.gcd B_share C_share = 2 :=
by
  sorry

end ratio_B_to_C_l324_324867


namespace distance_between_points_l324_324139

theorem distance_between_points : 
  ∀ (t : ℝ), 
  let x := 1 + 3 * t,
      y := 1 + t in 
  (dist (x, y).(t=0) (x, y).(t=1) = real.sqrt 10) := 
by
  sorry

end distance_between_points_l324_324139


namespace fill_bathtub_time_l324_324464

def rate_cold_water : ℚ := 3 / 20
def rate_hot_water : ℚ := 1 / 8
def rate_drain : ℚ := 3 / 40
def net_rate : ℚ := rate_cold_water + rate_hot_water - rate_drain

theorem fill_bathtub_time :
  net_rate = 1/5 → (1 / net_rate) = 5 := by
  sorry

end fill_bathtub_time_l324_324464


namespace smallest_angle_triangle_HAD_l324_324841

def hexagon_cond (ABCDEF : ConvexEquilateralHexagon) (parallel : (BC || AD) ∧ (AD || EF)) (ortho_triangle : Orthocenter ABD H) (smallest_angle_hexagon : degrees 4) : Prop :=
  ∃ H : Point, (isOrthocenter H A B D) ∧
               (smallest_angle_hexagon = 4)

theorem smallest_angle_triangle_HAD (ABCDEF : ConvexEquilateralHexagon)
  (h1 : parallel (BC || AD))
  (h2 : parallel (AD || EF))
  (H : Point)
  (h3 : isOrthocenter H A B D)
  (smallest_angle_hexagon : degrees 4) :
  smallest_angle (triangle HAD) = degrees 3 :=
by
  sorry

end smallest_angle_triangle_HAD_l324_324841


namespace total_revenue_is_correct_l324_324070

-- Joan decided to sell all of her old books.
-- She had 33 books in total.
-- She sold 15 books at $4 each.
-- She sold 6 books at $7 each.
-- The rest of the books were sold at $10 each.
-- We need to prove that the total revenue is $222.

def totalBooks := 33
def booksAt4 := 15
def priceAt4 := 4
def booksAt7 := 6
def priceAt7 := 7
def priceAt10 := 10
def remainingBooks := totalBooks - (booksAt4 + booksAt7)
def revenueAt4 := booksAt4 * priceAt4
def revenueAt7 := booksAt7 * priceAt7
def revenueAt10 := remainingBooks * priceAt10
def totalRevenue := revenueAt4 + revenueAt7 + revenueAt10

theorem total_revenue_is_correct : totalRevenue = 222 := by
  sorry

end total_revenue_is_correct_l324_324070


namespace problem_Ⅰ_problem_Ⅱ_problem_Ⅲ_l324_324976

open Real

-- Definition of the circle given the conditions
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Problem Ⅰ
theorem problem_Ⅰ :
    ∃ m ∈ Int, 5 = (abs (4 * m.toReal - 29)) / 5 → circle x y :=
sorry

-- Problem Ⅱ
theorem problem_Ⅱ (a : ℝ) (h : a > 0) :
    (∀ x y, circle x y → ax - y + 5 = 0) → a > 5/12 :=
sorry

-- Problem Ⅲ
theorem problem_Ⅲ :
    (∃ (a : ℝ), a > 5/12 ∧ ∃ (x y : ℝ), (circle x y) ∧
    (((x-1)^2 + y^2 = 25) ∧ (ax - y + 5 = 0)) ∧
    (x + a * y + 2 - 4 * a = 0) ∧ (1 + 0 + 2 - 4a = 0) ∧ a = 3/4) :=
sorry

end problem_Ⅰ_problem_Ⅱ_problem_Ⅲ_l324_324976


namespace stratified_sampling_third_year_students_l324_324050

theorem stratified_sampling_third_year_students :
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  (third_year_students : ℚ) * sampling_ratio = 20 :=
by 
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  show (third_year_students : ℚ) * sampling_ratio = 20
  sorry

end stratified_sampling_third_year_students_l324_324050


namespace Rahul_batting_average_l324_324865

theorem Rahul_batting_average :
  ∃ A : ℝ, (∀ (matches_played_before runs_scored_today batting_average_after : ℕ),
  matches_played_before = 8 → 
  runs_scored_today = 78 →
  batting_average_after = 54 →
  let total_runs_before := matches_played_before * A in
  let total_runs_after := total_runs_before + runs_scored_today in
  let matches_played_after := matches_played_before + 1 in
  matches_played_after * batting_average_after = total_runs_after
  ) → A = 51 :=
begin
  existsi (51 : ℝ),
  intros _ _ _ h_matches h_runs h_avg,
  simp [h_matches, h_runs, h_avg],
  sorry
end

end Rahul_batting_average_l324_324865


namespace M_plus_m_eq_2_l324_324739

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 5 - b * x + Real.sin x + 1

noncomputable def g (a b x : ℝ) : ℝ := a * x ^ 5 - b * x + Real.sin x

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

theorem M_plus_m_eq_2 (a b : ℝ) 
  (hf : ∀ x, f a b (-x) = f a b x) 
  (hg_odd : is_odd_function (g a b)):
  let M := (λ x, g a b x) (a) + 1,
      m := (λ x, - (g a b x)) (-a) + 1 in
  M + m = 2 := by
  sorry

end M_plus_m_eq_2_l324_324739


namespace probability_of_heads_l324_324589

-- Definitions of conditions
def has_two_sides (coin : Type) : Prop :=
  ∃ (heads tails : coin), heads ≠ tails

def is_fair (coin : Type) [measurable_space coin] (P : measure coin) : Prop :=
  ∀ (A : set coin), measurable_set A → P A = P (univ \ A)

-- Given a coin with two sides
variable (coin : Type) [measurable_space coin] (P : measure coin)
-- The coin has two sides: heads and tails
variable (heads tails : coin)
-- The coin is fair and not biased
hypothesis h1 : has_two_sides coin
hypothesis h2 : is_fair coin P

-- Question: What is the probability of getting a head in a fair coin toss?
theorem probability_of_heads : P {heads} = 1 / 2 :=
by
  sorry

end probability_of_heads_l324_324589


namespace biking_meet_time_l324_324098

noncomputable def distance (rate time : ℝ) := rate * time

theorem biking_meet_time {t : ℝ} :
  (Maria_start_time = 7 + 45 / 60) ∧ (Maria_rate = 15) ∧
  (Joey_start_time = 8 + 15 / 60) ∧ (Joey_rate = 18) ∧
  ∃ t : ℝ, (distance Maria_rate t + distance Joey_rate (t - 0.5) = 75) ∧
  (Maria_start_time + t ≈ 10 + 15 / 60) :=
by
  sorry

end biking_meet_time_l324_324098


namespace parallel_vectors_sum_is_six_l324_324736

theorem parallel_vectors_sum_is_six (x y : ℝ) :
  let a := (4, -1, 1)
  let b := (x, y, 2)
  (x / 4 = 2) ∧ (y / -1 = 2) →
  x + y = 6 :=
by
  intros
  sorry

end parallel_vectors_sum_is_six_l324_324736


namespace circle_angles_sum_l324_324056

theorem circle_angles_sum
  (ABCD_is_circle : ∃ O, ∀ P ∈ {A, B, C, D}, dist O P = r)
  (A B C D : Point)
  (hBAC : ∠ BAC = 20)
  (hBDC : ∠ BDC = 50)
  : ∠ BCA + ∠ BAD = 220 := 
sorry

end circle_angles_sum_l324_324056


namespace similar_triangles_l324_324241

-- Define the geometric setup
variables {O1 O2 P A B C E : Type} 
        [geometry.Circle O1] [geometry.Circle O2]
        (r1 r2 : ℝ)
        (O1_O2_collinear : collinear O1 A O2)
        (PB_tangent : tangent P B O1)
        (PC_tangent : tangent P C O2)
        (PB_PC_ratio : PB_tangent.ratio PC_tangent = r1 / r2)
        (PA_intersect : intersects PA O2 E)

-- Prove the similarity of the triangles
theorem similar_triangles : similar (triangle P A B) (triangle P E C) :=
sorry

end similar_triangles_l324_324241


namespace integral_result_integral_ln2_l324_324698

open Real

noncomputable def integral_eval (a : ℝ) : ℝ :=
  ∫ x in 0..a, 1 / (1 + cos x + sin x)

theorem integral_result :
  -π / 2 < a ∧ a < π →
  integral_eval a = log (1 + tan (a / 2)) :=
by
  intro h
  sorry

theorem integral_ln2 : 
  integral_eval (π / 2) = log 2 :=
by
  have h : -π / 2 < π / 2 ∧ π / 2 < π := by linarith [real.pi_pos]
  rw [integral_result h]
  have h2 : tan (π / 4) = 1 := by simp [real.tan_pi_div_four]
  simp [h2]
  sorry

end integral_result_integral_ln2_l324_324698


namespace find_f_3_l324_324499

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l324_324499


namespace isosceles_triangle_area_triangle_l324_324046

variables (A B C a b c : ℝ)
variables (S : ℝ)
variables (sin_A sin_B : ℝ)

/-- Basic conditions and vectors --/
def triangle_conditions (a b : ℝ) (sin_A sin_B : ℝ) : Prop :=
  let m := (a, b)
  let n := (sin_B, sin_A)
  let p := (b - 2, a - 2)
  m.1 * n.2 = m.2 * n.1 ∧ m.1 * p.1 + m.2 * p.2 = 0

/-- Part (Ⅰ): Prove if m is parallel to n, the triangle is isosceles. --/
theorem isosceles_triangle (h₁ : ∃ a b sin_A sin_B, triangle_conditions a b sin_A sin_B ∧ a * sin_A = b * sin_B) :
  ∃ a b, a = b :=
by
  sorry

/-- Part (Ⅱ): Given conditions, find the area of the triangle. --/
theorem area_triangle (h₂ : ∃ a b, triangle_conditions a b (sin B) (sin A) ∧ a + b = a * b ∧ a^2 + b^2 - 2 * a * b * (1/2) = 4) :
  S = sqrt 3 :=
by
  sorry

end isosceles_triangle_area_triangle_l324_324046


namespace point_not_in_transformed_plane_l324_324087

noncomputable def A : ℝ × ℝ × ℝ := (-2, 4, 1)

def plane (x y z : ℝ) : Prop := 3 * x + y + 2 * z + 2 = 0

def similarity_transform_coeff : ℝ := 3

theorem point_not_in_transformed_plane :
  ¬ plane A.1 A.2 A.3 := by
  sorry

end point_not_in_transformed_plane_l324_324087


namespace semicircle_curve_l324_324886

theorem semicircle_curve (x y : ℝ) :
  x - 1 = real.sqrt (1 - (y - 1)^2) ↔ (x - 1)^2 + (y - 1)^2 = 1 ∧ x ≥ 1 :=
by sorry

end semicircle_curve_l324_324886


namespace find_f_three_l324_324483

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l324_324483


namespace triangle_determines_plane_l324_324804

theorem triangle_determines_plane (A B C : Point) (h : ¬ collinear A B C) : determines_plane (triangle A B C) :=
sorry

end triangle_determines_plane_l324_324804


namespace isosceles_trapezoid_ratio_ab_cd_l324_324078

theorem isosceles_trapezoid_ratio_ab_cd (AB CD : ℝ) (P : ℝ → ℝ → Prop)
  (area1 area2 area3 area4 : ℝ)
  (h1 : AB > CD)
  (h2 : area1 = 5)
  (h3 : area2 = 7)
  (h4 : area3 = 3)
  (h5 : area4 = 9) :
  AB / CD = 1 + 2 * Real.sqrt 2 :=
sorry

end isosceles_trapezoid_ratio_ab_cd_l324_324078


namespace true_propositions_l324_324660

-- Definitions for the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (a b : ℝ) : Prop := a^2 > b^2 → |a| > |b|
def proposition3 (a b c : ℝ) : Prop := (a > b ↔ a + c > b + c)

-- Theorem to state the true propositions
theorem true_propositions (a b c : ℝ) :
  -- Proposition 3 is true
  (proposition3 a b c) →
  -- Assert that the serial number of the true propositions is 3
  {3} = { i | (i = 1 ∧ proposition1 a b) ∨ (i = 2 ∧ proposition2 a b) ∨ (i = 3 ∧ proposition3 a b c)} :=
by
  sorry

end true_propositions_l324_324660


namespace exists_k_with_large_prime_l324_324154

def a : ℕ → ℕ
| 0     => 2
| (n+1) => a n * a n + a n

def greatest_prime_divisor (n : ℕ) : option ℕ :=
  Nat.prime_divisors n |>.maximum

def condition (n : ℕ) : Prop := 
  match greatest_prime_divisor (a n) with
  | none => false
  | some p => p > 1000 ^ 1000

theorem exists_k_with_large_prime :
  ∃ k, condition k :=
sorry

end exists_k_with_large_prime_l324_324154


namespace infinite_solutions_l324_324714

def f (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), n / 2^k

theorem infinite_solutions (a : ℕ) (ha : a > 0) : ∃^∞ n, n - f n = a :=
sorry

end infinite_solutions_l324_324714


namespace unique_b_for_smallest_a_l324_324552

noncomputable def smallest_a := ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ (∀ x : ℝ, (x^4 - a * x^3 + a * x^2 - b * x + b = 0) → real_roots x) ∧ a = 4 ∧ b = 1

theorem unique_b_for_smallest_a : smallest_a x := 
  begin
    sorry
  end

end unique_b_for_smallest_a_l324_324552


namespace solve_system_l324_324122

noncomputable
def system_of_equations (x y z t : ℝ) : Prop :=
  (x * y * z = x + y + z) ∧
  (y * z * t = y + z + t) ∧
  (z * t * x = z + t + x) ∧
  (t * x * y = t + x + y)

noncomputable
def solution_set : set (ℝ × ℝ × ℝ × ℝ) :=
  { (0, 0, 0, 0), (sqrt 3, sqrt 3, sqrt 3, sqrt 3), (-sqrt 3, -sqrt 3, -sqrt 3, -sqrt 3) }

theorem solve_system :
  {p : ℝ × ℝ × ℝ × ℝ | system_of_equations p.1 p.2 p.3 p.4} = solution_set :=
by
  sorry

end solve_system_l324_324122


namespace factorize_expression_l324_324271

variable (x y : ℝ)

theorem factorize_expression : xy^2 + 6*xy + 9*x = x*(y + 3)^2 := by
  sorry

end factorize_expression_l324_324271


namespace ordering_of_a_b_c_l324_324417

theorem ordering_of_a_b_c :
  let a := 6^0.7
  let b := Real.log 0.6 / Real.log 7
  let c := Real.log 0.7 / Real.log 0.6
  a > c ∧ c > b := by
    sorry

end ordering_of_a_b_c_l324_324417


namespace avg_five_probability_l324_324719

/- Define the set of natural numbers from 1 to 9. -/
def S : Finset ℕ := Finset.range 10 \ {0}

/- Define the binomial coefficient for choosing 7 out of 9. -/
def choose_7_9 : ℕ := Nat.choose 9 7

/- Define the condition for the sum of chosen numbers to be 35. -/
def sum_is_35 (s : Finset ℕ) : Prop := s.sum id = 35

/- Number of ways to choose 3 pairs that sum to 10 and include number 5 - means sum should be 35-/
def ways_3_pairs_and_5 : ℕ := 4

/- Probability calculation. -/
def prob_sum_is_35 : ℚ := (ways_3_pairs_and_5: ℚ) / (choose_7_9: ℚ)

theorem avg_five_probability : prob_sum_is_35 = 1 / 9 := by
  sorry

end avg_five_probability_l324_324719


namespace A_completes_work_in_18_days_l324_324193

-- Define the conditions
def efficiency_A_twice_B (A B : ℕ → ℕ) : Prop := ∀ w, A w = 2 * B w
def same_work_time (A B C D : ℕ → ℕ) : Prop := 
  ∀ w t, A w + B w = C w + D w ∧ C t = 1 / 20 ∧ D t = 1 / 30

-- Define the key quantity to be proven
theorem A_completes_work_in_18_days (A B C D : ℕ → ℕ) 
  (h1 : efficiency_A_twice_B A B) 
  (h2 : same_work_time A B C D) : 
  ∀ w, A w = 1 / 18 :=
sorry

end A_completes_work_in_18_days_l324_324193


namespace right_triangle_hypotenuse_l324_324856

theorem right_triangle_hypotenuse (h : ℝ) :
  (∀ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = b ∧ a = 12 ∧ ∃ θ : ℝ, θ = real.pi / 4 → c = h) → h = 12 * real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l324_324856


namespace solution_set_eq_l324_324868

theorem solution_set_eq :
  {x : ℕ | nat.choose 28 x = nat.choose 28 (3 * x - 8)} = {4, 9} :=
by
  sorry

end solution_set_eq_l324_324868


namespace aluminum_cans_total_volume_l324_324267

theorem aluminum_cans_total_volume : 
  ∃ (x : ℕ), x + 1.5 * x + (64 / 3) * x < 30 ∧ x * (6/6 + 9/6 + 128/6) = 23 := 
by
  sorry

end aluminum_cans_total_volume_l324_324267


namespace number_of_unique_intersections_l324_324354

-- Definitions for the given lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 3
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 5 * x - 3 * y = 6

-- The problem is to show the number of unique intersection points is 2
theorem number_of_unique_intersections : ∃ p1 p2 : ℝ × ℝ,
  p1 ≠ p2 ∧
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line1 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (p1 ≠ p2 → ∀ p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2 ∨ line3 p.1 p.2) →
    (p = p1 ∨ p = p2)) :=
sorry

end number_of_unique_intersections_l324_324354


namespace length_DF_l324_324801

variables {A B C D E F : Type}
variables (DC EB DE : ℝ)
variables [∃ (AB BC : ℝ), AB = BC]

theorem length_DF 
  (h1 : DC = 15)
  (h2 : EB = 5)
  (h3 : DE = 7)
  (hAB : ∀ (A B : Type), AB = DC) :
  ∃ DF : ℝ, DF = 7 :=
by
  sorry

end length_DF_l324_324801


namespace sample_size_l324_324599

-- Definitions for the conditions
def ratio_A : Nat := 2
def ratio_B : Nat := 3
def ratio_C : Nat := 4
def stratified_sample_size : Nat := 9 -- Total parts in the ratio sum
def products_A_sample : Nat := 18 -- Sample contains 18 Type A products

-- We need to tie these conditions together and prove the size of the sample n
theorem sample_size (n : Nat) (ratio_A ratio_B ratio_C stratified_sample_size products_A_sample : Nat) :
  ratio_A = 2 → ratio_B = 3 → ratio_C = 4 → stratified_sample_size = 9 → products_A_sample = 18 → n = 81 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof body here
  sorry -- Placeholder for the proof

end sample_size_l324_324599


namespace B_time_l324_324587

open Nat

variable (A B : ℕ) -- A is the time it takes for A to finish the task, and B is the time for B to finish alone.
variable (h1 : A + 12 = B) -- Condition 1: B takes 12 more days than A
variable (h2 : 60 * B / 100 = 12 + A - 12) -- Condition 2: B completes 60% of the task

theorem B_time (B : ℕ) (h1 : A + 12 = B) (h2 : 0.4 * B = 12) : B = 30 := by
  -- proof goes here
  sorry

end B_time_l324_324587


namespace only_prime_in_list_is_47_l324_324018

def is_concatenation_of_4747 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 47 * (4747 ^ k)

theorem only_prime_in_list_is_47 : 
  (∃ n : ℕ, is_prime n ∧ is_concatenation_of_4747 n) → n = 47 := 
sorry

end only_prime_in_list_is_47_l324_324018


namespace units_digit_sum_factorials_500_l324_324944

-- Define the unit digit computation function
def unit_digit (n : ℕ) : ℕ := n % 10

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * fact n

-- Define the sum of factorials from 1 to n
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => fact i)

-- Define the problem statement
theorem units_digit_sum_factorials_500 : unit_digit (sum_factorials 500) = 3 :=
sorry

end units_digit_sum_factorials_500_l324_324944


namespace pizza_topping_count_l324_324611

theorem pizza_topping_count (n : ℕ) (hn : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by
  sorry

end pizza_topping_count_l324_324611


namespace lcm_5_7_10_14_l324_324570

theorem lcm_5_7_10_14 : Nat.lcm (Nat.lcm 5 7) (Nat.lcm 10 14) = 70 := by
  sorry

end lcm_5_7_10_14_l324_324570


namespace minimum_mushrooms_l324_324855

variable {x y k : ℕ}

-- Define the conditions
def masha_first_day (x : ℕ) : ℝ := (3 / 4) * x
def masha_second_day (y : ℕ) : ℝ := (6 / 5) * y

-- Total mushrooms collected by Masha and Vasya
def total_vasya (x y : ℕ) : ℕ := x + y
def total_masha (x y : ℕ) : ℝ := masha_first_day x + masha_second_day y

-- Condition: Masha collected 10% more over two days
def condition (x y : ℕ) : Prop := 1.1 * (total_vasya x y) = total_masha x y

-- Given the smallest natural numbers such that y = (7/2)x
def smallest_numbers : Prop := 
  ∃ (k : ℕ) (x y : ℕ), x = 2 * k ∧ y = 7 * k ∧ condition x y

-- Prove the minimum total number of mushrooms collected together
theorem minimum_mushrooms : smallest_numbers → (total_vasya 2 7 + total_masha 2 7 = 189) :=
by
  sorry

end minimum_mushrooms_l324_324855


namespace incorrect_value_of_observation_l324_324525

theorem incorrect_value_of_observation
  (mean_initial : ℝ) (n : ℕ) (sum_initial: ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mean_corrected : ℝ)
  (h1 : mean_initial = 36) 
  (h2 : n = 50) 
  (h3 : sum_initial = n * mean_initial) 
  (h4 : correct_value = 45) 
  (h5 : mean_corrected = 36.5) 
  (sum_corrected : ℝ) 
  (h6 : sum_corrected = n * mean_corrected) : 
  incorrect_value = 20 := 
by 
  sorry

end incorrect_value_of_observation_l324_324525


namespace machine_A_sprockets_per_hour_l324_324436

-- Definitions based on the problem conditions
def MachineP_time (A : ℝ) (T : ℝ) : ℝ := T + 10
def MachineQ_rate (A : ℝ) : ℝ := 1.1 * A
def MachineP_sprockets (A : ℝ) (T : ℝ) : ℝ := A * (T + 10)
def MachineQ_sprockets (A : ℝ) (T : ℝ) : ℝ := 1.1 * A * T

-- Lean proof statement to prove that Machine A produces 8 sprockets per hour
theorem machine_A_sprockets_per_hour :
  ∀ A T : ℝ, 
  880 = MachineP_sprockets A T ∧
  880 = MachineQ_sprockets A T →
  A = 8 :=
by
  intros A T h
  have h1 : 880 = MachineP_sprockets A T := h.left
  have h2 : 880 = MachineQ_sprockets A T := h.right
  sorry

end machine_A_sprockets_per_hour_l324_324436


namespace area_of_triangle_l324_324991

open Real

noncomputable def triangle_area (P S R : Point) : ℝ :=
  0.5 * abs ((S.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (S.2 - P.2))

def point_P : Point := (2, 9)
def point_S : Point := (4, 0)
def point_R : Point := let x_R := -1 in (x_R, 0)

theorem area_of_triangle : triangle_area point_P point_S point_R = 22.5 :=
  sorry

end area_of_triangle_l324_324991


namespace hyperbola_eccentricity_correct_l324_324754

noncomputable def hyperbola_eccentricity (a b : ℝ) (x y : ℝ) (asymp : x - 2 * y = 0) (h : x^2 / a^2 - y^2 / b^2 = 1) : ℝ :=
  let e := real.sqrt (1 + (b^2 / a^2))
  in e

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ), (∃ x y : ℝ, x - 2 * y = 0 ∧ x^2 / a^2 - y^2 / b^2 = 1) →
  hyperbola_eccentricity a b = real.sqrt 5 / 2 :=
sorry

end hyperbola_eccentricity_correct_l324_324754


namespace four_digit_numbers_count_l324_324009

theorem four_digit_numbers_count :
  let digits := {1, 2, 3, 4, 5}
  in (∀ (n : ℕ), n ∈ digits) → (cardinal.mk (finset.perm_4 digits) = 120) :=
by
  let digits := {1, 2, 3, 4, 5}
  have h1: (∀ (n : ℕ), n ∈ digits), from sorry
  have h2: cardinal.mk (finset.perm_4 digits) = 120, from sorry
  exact h2


end four_digit_numbers_count_l324_324009


namespace shaded_area_between_circles_l324_324561

theorem shaded_area_between_circles
  (r_small_circle r_large_circle : ℝ)
  (chord_length : ℝ)
  -- Conditions
  (h1 : r_small_circle = 40)
  (h2 : r_large_circle = 60)
  (h3 : chord_length = 100)
  (h4 : ∃ Q, OQ = r_small_circle ∧ ∀ C, OC = r_large_circle ∧ OQ ⊥ CD) :
  -- Conclusion
  (π * r_large_circle^2 - π * r_small_circle^2 = 2000 * π) :=
by
  sorry

end shaded_area_between_circles_l324_324561


namespace sum_of_first_15_terms_l324_324381

variable {a: ℕ → ℤ} -- Assume a is an arithmetic sequence with integer terms

-- Definition: a is an arithmetic sequence if there exists a common difference d
def is_arithmetic_seq (a: ℕ → ℤ) (d: ℤ): Prop :=
  ∀ n: ℕ, a (n + 1) = a n + d

-- Given condition
variable (h_condition: a 1 - a 4 - a 8 - a 12 + a 15 = 2)

-- Proof goal: We want to prove that the sum of the first 15 terms is -30
theorem sum_of_first_15_terms
  (h_arith_seq: is_arithmetic_seq a d) 
: (Finset.range 15).sum (λ n, a (n + 1)) = -30 :=
sorry

end sum_of_first_15_terms_l324_324381


namespace intersection_M_complement_N_l324_324079

open Set

def M : Set ℝ := {x | x ≥ -2}
def N : Set ℝ := {x | 2 ^ x - 1 > 0}
def CR_N : Set ℝ := {x | ¬(2 ^ x - 1 > 0)}

theorem intersection_M_complement_N : M ∩ CR_N = {x | -2 ≤ x ∧ x ≤ 0} := by
  sorry

end intersection_M_complement_N_l324_324079


namespace geom_seq_thm_l324_324298

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
a 1 ≠ 0 ∧ ∀ n, a (n + 1) = (a n ^ 2) / (a (n - 1))

theorem geom_seq_thm (a : ℕ → ℝ) (h : geom_seq a) (h_neg : ∀ n, a n < 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) : a 3 + a 5 = -6 :=
by
  sorry

end geom_seq_thm_l324_324298


namespace problem_l324_324424

theorem problem (n : ℕ) (a : Fin n → ℕ) (h : ∀ (i j : Fin n), i ≠ j → a i ≠ a j) 
  (h_pos : ∀ (i : Fin n), 0 < a i) : (∑ i : Fin n, a i / (i + 1)) ≥ n := 
by
  sorry

end problem_l324_324424


namespace rectangle_area_correct_l324_324892

noncomputable def rectangle_area (x: ℚ) : ℚ :=
  let length := 5 * x - 18
  let width := 25 - 4 * x
  length * width

theorem rectangle_area_correct (x: ℚ) (h1: 3.6 < x) (h2: x < 6.25) :
  rectangle_area (43 / 9) = (2809 / 81) := 
  by
    sorry

end rectangle_area_correct_l324_324892


namespace midpoint_traces_line_perpendicular_l324_324973

variable (k : Type) [MetricSpace k]
variable (A B C : k)
variable (r c : ℝ)

-- conditions
variable (circle : MetricSpace k) (fixed_point_on_circumference : k)
variable (A: circle) (B C : circle)

-- AB^2 + AC^2 remains constant
axiom AB_AC_constant : ∀ (B C : circle), dist A B ^ 2 + dist A C ^ 2 = c ^ 2

theorem midpoint_traces_line_perpendicular (F : k) :
  ∀ (BC_midpoint : k), (BC_midpoint = midpoint B C) → 
  (BC_midpoint trace_path = line_perpendicular AD) :=
sorry

end midpoint_traces_line_perpendicular_l324_324973


namespace total_spent_on_clothing_l324_324398

-- Define the individual costs
def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

-- Define the proof problem to show the total cost
theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  sorry

end total_spent_on_clothing_l324_324398


namespace opposite_of_neg_three_is_three_l324_324899

theorem opposite_of_neg_three_is_three : ∃ x : ℤ, (-3) + x = 0 ∧ x = 3 :=
by
  use 3
  split
  · linarith
  · rfl

end opposite_of_neg_three_is_three_l324_324899


namespace complex_inequalities_equiv_l324_324426

variable (a b c : ℂ)

theorem complex_inequalities_equiv :
  (Complex.re ((a - c) * (Complex.conj c - Complex.conj b)) ≥ 0) ↔ 
  (Complex.abs (c - (a + b) / 2) ≤ (1 / 2) * Complex.abs (a - b)) :=
sorry

end complex_inequalities_equiv_l324_324426


namespace part1_part2_l324_324767

variable (x : ℝ)
def A : ℝ := 2 * x^2 - 3 * x + 2
def B : ℝ := x^2 - 3 * x - 2

theorem part1 : A x - B x = x^2 + 4 := sorry

theorem part2 (h : x = -2) : A x - B x = 8 := sorry

end part1_part2_l324_324767


namespace triangle_fraction_of_grid_l324_324683

def point := (ℝ × ℝ)
def A : point := (2, 4)
def B : point := (7, 2)
def C : point := (6, 6)
def grid_width : ℝ := 8
def grid_height : ℝ := 6

theorem triangle_fraction_of_grid :
  let area_triangle := 1 / 2 * (abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2)))) in
  let area_grid := grid_width * grid_height in
  area_triangle / area_grid = 3 / 16 :=
by
  sorry

end triangle_fraction_of_grid_l324_324683


namespace sin_squared_sum_to_cos_product_l324_324684

variables {x : ℝ}

theorem sin_squared_sum_to_cos_product (h : sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 3) :
  ∃ a b c : ℕ, a = 2 ∧ b = 4 ∧ c = 8 ∧ (cos (a * x) * cos (b * x) * cos (c * x) = 0) ∧ a + b + c = 14 :=
begin
  sorry
end

end sin_squared_sum_to_cos_product_l324_324684


namespace theresa_sons_count_l324_324160

theorem theresa_sons_count (total_meat_left : ℕ) (meat_per_plate : ℕ) (frac_left : ℚ) (s : ℕ) :
  total_meat_left = meat_per_plate ∧ meat_per_plate * frac_left * s = 3 → s = 9 :=
by sorry

end theresa_sons_count_l324_324160


namespace sum_of_divisors_divisible_by_24_l324_324410

theorem sum_of_divisors_divisible_by_24 (n : ℕ) (h : n + 1 % 24 = 0) :
  24 ∣ (∑ d in (finset.filter (λ x, n % x = 0) (finset.range (n+1))), d) :=
  sorry

end sum_of_divisors_divisible_by_24_l324_324410


namespace smallest_rational_number_l324_324238

theorem smallest_rational_number 
  (a b c d : ℚ) 
  (h1 : a = -1) 
  (h2 : b = 0) 
  (h3 : c = 3) 
  (h4 : d = -1/3) 
  : min (min a (min b (min c d))) = a := by
sorry

end smallest_rational_number_l324_324238


namespace play_earning_l324_324994

-- Conditions
def rows := 30
def seats_per_row := 15
def total_seats := rows * seats_per_row
def seats_per_section := total_seats / 3

def section_A_rows := 10
def section_B_rows := 10
def section_C_rows := 10

def ticket_price_A := 20
def ticket_price_B := 15
def ticket_price_C := 10

def percentage_sold_A := 0.8
def percentage_sold_B := 0.7
def percentage_sold_C := 0.9

def seats_sold_A := seats_per_section * percentage_sold_A
def seats_sold_B := seats_per_section * percentage_sold_B
def seats_sold_C := seats_per_section * percentage_sold_C

def earnings_A := seats_sold_A * ticket_price_A
def earnings_B := seats_sold_B * ticket_price_B
def earnings_C := seats_sold_C * ticket_price_C
def total_earnings := earnings_A + earnings_B + earnings_C

-- The proof statement
theorem play_earning : total_earnings = 5325 := by
  sorry

end play_earning_l324_324994


namespace enclosed_area_abs_eq_20_l324_324566

theorem enclosed_area_abs_eq_20 : ∀ (x y : ℝ),
  (|x| + |3 * y| + |x - y| = 20) →
  (enclosed_area_eq_200_over_3 : ℝ) := sorry

end enclosed_area_abs_eq_20_l324_324566


namespace total_students_l324_324051

theorem total_students (S : ℕ) (h1 : 0.45 * S + 0.23 * S + 0.15 * S + 119 = S) : S = 700 := sorry

end total_students_l324_324051


namespace total_wheels_computation_probability_bicycle_or_tricycle_l324_324285

def transportation_data := {
  cars : ℕ,
  bicycles : ℕ,
  trucks : ℕ,
  tricycles : ℕ,
  motorcycles : ℕ,
  skateboards : ℕ,
  unicycles : ℕ
}

def wheels_per_vehicle : transportation_data → ℕ
| ⟨cars, bicycles, trucks, tricycles, motorcycles, skateboards, unicycles⟩ :=
  (cars * 4) + (bicycles * 2) + (trucks * 4) + (tricycles * 3) + (motorcycles * 2) + (skateboards * 4) + (unicycles * 1)

theorem total_wheels_computation:
  let t := ⟨15, 3, 8, 1, 4, 2, 1⟩ in wheels_per_vehicle t = 118 :=
by {
  intro t,
  unfold wheels_per_vehicle,
  norm_num,
  have h : ((((15 * 4) + (3 * 2) + (8 * 4)) + (1 * 3) + (4 * 2)) + (2 * 4) + (1 * 1) = 118), by norm_num,
  exact h,
  sorry
}

theorem probability_bicycle_or_tricycle:
  let total_units := 15 + 3 + 8 + 1 + 4 + 2 + 1 in
  let bicycles_and_tricycles := 3 + 1 in
  (bicycles_and_tricycles : rat) / (total_units : rat) ≈ 11.76 / 100 :=
by {
  rw [total_units, bicycles_and_tricycles],
  norm_num,
  have h : ((4 : rat) / (34 : rat)) ≈ (11.76 / 100), sorry,
  exact h,
  sorry
}

end total_wheels_computation_probability_bicycle_or_tricycle_l324_324285


namespace count_prime_numbers_in_list_only_prime_in_list_is_47_number_of_primes_in_list_l324_324014

theorem count_prime_numbers_in_list : 
  ∀ (n : ℕ), (∃ k : ℕ, n = 47 * ((10^k - 1) / 9)) → n ≠ 47 =→ n % 47 = 0 → isPrime n → False :=
by
  assume n hn h_eq_prime_47 h_divisible it_is_prime
  sorry

theorem only_prime_in_list_is_47 : ∀ (n : ℕ), n ∈ { num | ∃ k : ℕ, num = 47 * ((10^k - 1) / 9) } → (isPrime n ↔ n = 47) := 
by
  assume n hn
  split
    assume it_is_prime
    by_cases h_case : n = 47
      case inl => exact h_case
      case inr =>
        obtain ⟨k, hk⟩ := hn
        have h_mod : n % 47 = 0 := by rw [hk, nat.mul_mod_right]
        apply_false_from (count_prime_numbers_in_list n) hk h_case h_mod it_is_prime
        contradiction
    assume it_is_47
    exact by norm_num [it_is_47]
    sorry

theorem number_of_primes_in_list : ∀ (l : List ℕ), (∀ (n : ℕ), n ∈ l → ∃ k : ℕ, n = 47 * ((10^k - 1) / 9)) → l.filter isPrime = [47] :=
by
  sorry

end count_prime_numbers_in_list_only_prime_in_list_is_47_number_of_primes_in_list_l324_324014


namespace pond_length_l324_324057

-- Define the dimensions and volume of the pond
def pond_width : ℝ := 15
def pond_depth : ℝ := 5
def pond_volume : ℝ := 1500

-- Define the length variable
variable (L : ℝ)

-- State that the volume relationship holds and L is the length we're solving for
theorem pond_length :
  pond_volume = L * pond_width * pond_depth → L = 20 :=
by
  sorry

end pond_length_l324_324057


namespace problem_one_solution_set_problem_two_range_m_l324_324450

def f (x m : ℝ) : ℝ := |x - 1| + |x - m|

-- Exercise (1)
theorem problem_one_solution_set :
  (set_of (λ x : ℝ, f x 3 ≥ 5) = { x | x ≤ -1/2 } ∪ { x | 9/2 ≤ x }) :=
sorry

-- Exercise (2)
theorem problem_two_range_m (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 2 * m - 1) → (m ≤ 2 / 3) :=
sorry

end problem_one_solution_set_problem_two_range_m_l324_324450


namespace exists_negative_root_of_P_l324_324263

def P(x : ℝ) : ℝ := x^7 - 2 * x^6 - 7 * x^4 - x^2 + 10

theorem exists_negative_root_of_P : ∃ x : ℝ, x < 0 ∧ P x = 0 :=
sorry

end exists_negative_root_of_P_l324_324263


namespace find_f_of_3_l324_324481

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l324_324481


namespace efficiency_ratio_l324_324605

theorem efficiency_ratio (r : ℚ) (work_B : ℚ) (work_AB : ℚ) (B_alone : ℚ) (AB_together : ℚ) (efficiency_A : ℚ) (B_efficiency : ℚ) :
  B_alone = 30 ∧ AB_together = 20 ∧ B_efficiency = (1/B_alone) ∧ efficiency_A = (r * B_efficiency) ∧ (efficiency_A + B_efficiency) = (1 / AB_together) → r = 1 / 2 :=
by
  sorry

end efficiency_ratio_l324_324605


namespace std_dev_same_l324_324362

variables {n : ℕ} {x : Fin n → ℝ}

/- Given the data set x_i, we define another set y_i such that y_i = x_i + 2. -/
def y (i : Fin n) : ℝ := x i + 2

/- Prove that the standard deviation of x and y are the same. -/
theorem std_dev_same (x : Fin n → ℝ) : 
  Real.stddev (Fintype.elems (Fin n) (x)) = Real.stddev (Fintype.elems (Fin n) (y)) :=
sorry

end std_dev_same_l324_324362


namespace emily_sixth_quiz_score_l324_324270

theorem emily_sixth_quiz_score (
  scores : List ℕ,
  required_mean : ℚ,
  existing_scores_len : scores.length = 5,
  existing_scores : scores = [91, 94, 86, 88, 101],
  required_mean_val : required_mean = 94
) : ∃ sixth_score : ℚ, (scores.sum + sixth_score) / 6 = required_mean ∧ sixth_score = 104 :=
by
  sorry

end emily_sixth_quiz_score_l324_324270


namespace find_vertex_C_l324_324894

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)
def euler_line (x y : ℝ) : Prop := x - y + 2 = 0

theorem find_vertex_C 
  (C : ℝ × ℝ)
  (h_centroid : (2 + C.1) / 3 = (4 + C.2) / 3)
  (h_euler_line : euler_line ((2 + C.1) / 3) ((4 + C.2) / 3))
  (h_circumcenter : (C.1 + 1)^2 + (C.2 - 1)^2 = 10) :
  C = (-4, 0) :=
sorry

end find_vertex_C_l324_324894


namespace length_of_second_train_l324_324927

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (meeting_time_seconds : ℝ)
  (speed_conversion_factor : ℝ := 1000 / 3600)
  (length_second_train : ℝ := ((speed_first_train_kmph * speed_conversion_factor
                                + speed_second_train_kmph * speed_conversion_factor)
                               * meeting_time_seconds) - length_first_train) :
  length_second_train ≈ 299.97 := by
  have speed_first_train := speed_first_train_kmph * speed_conversion_factor
  have speed_second_train := speed_second_train_kmph * speed_conversion_factor
  have relative_speed := speed_first_train + speed_second_train
  have total_distance := relative_speed * meeting_time_seconds
  have length_second_train_calc := total_distance - length_first_train
  show length_second_train_calc ≈ 299.97 from sorry

end length_of_second_train_l324_324927


namespace square_root_of_9_l324_324940

theorem square_root_of_9 : {x : ℝ // x^2 = 9} = {x : ℝ // x = 3 ∨ x = -3} :=
by
  sorry

end square_root_of_9_l324_324940


namespace vector_dot_scalar_mul_l324_324680

theorem vector_dot_scalar_mul (v w : Fin 2 → ℤ) (hv : v = ![-3, 2]) (hw : w = ![5, -4]) :
  (v ⬝ (2 • w)) = -46 :=
by sorry

end vector_dot_scalar_mul_l324_324680


namespace rectangle_length_l324_324595

theorem rectangle_length (x : ℕ) :
    let w := 6 in
    let swept_out_area := 45 * Real.pi in
    (1 / 4) * Real.pi * (x^2 + w^2) = swept_out_area →
    x = 12 :=
by 
  sorry

end rectangle_length_l324_324595


namespace cyclist_speed_ratio_l324_324172

-- Define the conditions
def speeds_towards_each_other (v1 v2 : ℚ) : Prop :=
  v1 + v2 = 25

def speeds_apart_with_offset (v1 v2 : ℚ) : Prop :=
  v1 - v2 = 10 / 3

-- The proof problem to show the required ratio of speeds
theorem cyclist_speed_ratio (v1 v2 : ℚ) (h1 : speeds_towards_each_other v1 v2) (h2 : speeds_apart_with_offset v1 v2) :
  v1 / v2 = 17 / 13 :=
sorry

end cyclist_speed_ratio_l324_324172


namespace cot_alpha_solution_l324_324312

theorem cot_alpha_solution 
  (α : Real) 
  (h1 : Real.sin (2 * α) = - Real.sin α)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.cot α = -Real.sqrt 3 / 3 :=
by
  sorry

end cot_alpha_solution_l324_324312


namespace average_daily_net_income_is_correct_l324_324969

-- Define net income for each day
def net_income_day_1 := 600 - 50
def net_income_day_2 := 250 - 70
def net_income_day_3 := 450 - 100
def net_income_day_4 := 400 - 30
def net_income_day_5 := 800 - 60
def net_income_day_6 := 450 - 40
def net_income_day_7 := 350
def net_income_day_8 := 600 - 55
def net_income_day_9 := 270 - 80
def net_income_day_10 := 500 - 90

-- Total net income
def total_net_income := net_income_day_1 + net_income_day_2 + net_income_day_3 + net_income_day_4 +
                        net_income_day_5 + net_income_day_6 + net_income_day_7 + net_income_day_8 +
                        net_income_day_9 + net_income_day_10

-- Number of days
def num_days := 10

-- Average daily net income
def average_daily_net_income := total_net_income / num_days

-- Theorem to prove
theorem average_daily_net_income_is_correct :
  average_daily_net_income = 399.50 :=
by
  sorry

end average_daily_net_income_is_correct_l324_324969


namespace sector_area_correct_l324_324881

-- Define the conditions
def radius : ℝ := 12
def angle_degrees : ℝ := 39
def π : ℝ := Real.pi

-- Define the function to calculate the area of the sector
def area_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * π * (r ^ 2)

-- Assert the theorem to prove
theorem sector_area_correct : area_of_sector radius angle_degrees ≈ 48.943 :=
sorry

end sector_area_correct_l324_324881


namespace min_force_required_l324_324574

def V : ℝ := 10 * 10^-6 -- Volume of the cube in cubic meters
def ρ_cube : ℝ := 700 -- Density of the cube's material in kg/m^3
def ρ_water : ℝ := 1000 -- Density of water in kg/m^3
def g : ℝ := 10 -- Acceleration due to gravity in m/s^2

noncomputable def F_g : ℝ := (ρ_cube * V) * g -- Gravitational force
noncomputable def F_b : ℝ := (ρ_water * V) * g -- Buoyant force
noncomputable def F_add : ℝ := F_b - F_g -- Additional force required to submerge

theorem min_force_required : F_add = 0.03 := by
  sorry

end min_force_required_l324_324574


namespace cos_of_right_triangle_l324_324802

theorem cos_of_right_triangle :
  ∀ (A B C : Type) [InnerProductSpace ℝ A]
  (AB BC : ℝ) (hA : ∠A = 90°) (h_length_AB : AB = 6) (h_length_BC : BC = 10),
  ∃ AC : ℝ, AC = real.sqrt (BC^2 - AB^2) → real.cos (real.atan2 (real.sqrt (BC^2 - AB^2)) AB) = 4 / 5 := 
by {
  intros,
  sorry
}

end cos_of_right_triangle_l324_324802


namespace number_of_citroens_submerged_is_zero_l324_324670

-- Definitions based on the conditions
variables (x y : ℕ) -- Define x as the number of Citroen and y as the number of Renault submerged
variables (r p c vr vp : ℕ) -- Define r as the number of Renault, p as the number of Peugeot, c as the number of Citroën

-- Given conditions translated
-- Condition 1: There were twice as many Renault cars as there were Peugeot cars
def condition1 (r p : ℕ) : Prop := r = 2 * p
-- Condition 2: There were twice as many Peugeot cars as there were Citroens
def condition2 (p c : ℕ) : Prop := p = 2 * c
-- Condition 3: As many Citroens as Renaults were submerged in the water
def condition3 (x y : ℕ) : Prop := y = x
-- Condition 4: Three times as many Renaults were in the water as there were Peugeots
def condition4 (r y : ℕ) : Prop := r = 3 * y
-- Condition 5: As many Peugeots visible in the water as there were Citroens
def condition5 (vp c : ℕ) : Prop := vp = c

-- The question to prove: The number of Citroen cars submerged is 0
theorem number_of_citroens_submerged_is_zero
  (h1 : condition1 r p) 
  (h2 : condition2 p c)
  (h3 : condition3 x y)
  (h4 : condition4 r y)
  (h5 : condition5 vp c) :
  x = 0 :=
sorry

end number_of_citroens_submerged_is_zero_l324_324670


namespace find_f_three_l324_324482

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l324_324482


namespace line_perpendicular_to_CD_l324_324447

structure CyclicQuadrilateral (A B C D : Type) where
  inscribed_in_circle : Prop

structure QuadrilateralMidpoint (A B K : Type) where
  midpoint_AB : Prop

structure PerpendicularDiagonals (A C B D Q : Type) where
  perpendicular_at_Q : Prop

theorem line_perpendicular_to_CD (A B C D Q K : Type)
  [CyclicQuadrilateral A B C D]
  [QuadrilateralMidpoint A B K]
  [PerpendicularDiagonals A C B D Q] :
  ∃ (KQ : Type), KQ = K → ∀ CD : Type, CD = D → Perpendicular KQ CD :=
sorry

end line_perpendicular_to_CD_l324_324447


namespace find_y_given_x_and_line_passing_two_points_l324_324606

theorem find_y_given_x_and_line_passing_two_points :
  ∃ y : ℝ, ∀ x : ℝ, (x = 4) → let m := (17 - 5) / (-7 - 3) in
  let y1 := 5 in let x1 := 3 in
  y - y1 = m * (x - x1) → y = 3.8 :=
by
  sorry

end find_y_given_x_and_line_passing_two_points_l324_324606


namespace average_speed_return_journey_l324_324104

theorem average_speed_return_journey
    (time_to_work : ℕ) (time_to_home : ℕ) (speed_to_work : ℕ)
    (distance_to_work : ℕ) (distance_to_home : ℕ)
    (distance_are_equal : distance_to_work = distance_to_home)
    (morning_conditions : time_to_work = 1 ∧ speed_to_work = 30 ∧ distance_to_work = speed_to_work * time_to_work)
    (evening_conditions : time_to_home = 1.5):
  (distance_to_work / time_to_home) = 20 := sorry

end average_speed_return_journey_l324_324104


namespace gcd_14568_78452_l324_324262

theorem gcd_14568_78452 : Nat.gcd 14568 78452 = 4 :=
sorry

end gcd_14568_78452_l324_324262


namespace largest_and_smallest_decimal_difference_l324_324174

-- Definitions of the numbers provided
def numbers : List ℕ := [3, 0, 4, 8]

-- Definitions of the largest and smallest decimals formed
def largest_decimal : ℝ := 8.430
def smallest_decimal : ℝ := 0.348

-- Statement that needs to be proved
theorem largest_and_smallest_decimal_difference :
  ∃ (largest smallest diff : ℝ), 
    largest = 8.430 ∧ 
    smallest = 0.348 ∧ 
    diff = largest - smallest ∧ 
    diff = 8.082 :=
by
  use 8.430, 0.348, 8.082
  split
  repeat {ring_nf}
  sorry

end largest_and_smallest_decimal_difference_l324_324174


namespace total_bulbs_is_118_l324_324106

-- Define the number of medium lights
def medium_lights : Nat := 12

-- Define the number of large and small lights based on the given conditions
def large_lights : Nat := 2 * medium_lights
def small_lights : Nat := medium_lights + 10

-- Define the number of bulbs required for each type of light
def bulbs_needed_for_medium : Nat := 2 * medium_lights
def bulbs_needed_for_large : Nat := 3 * large_lights
def bulbs_needed_for_small : Nat := 1 * small_lights

-- Define the total number of bulbs needed
def total_bulbs_needed : Nat := bulbs_needed_for_medium + bulbs_needed_for_large + bulbs_needed_for_small

-- The theorem that represents the proof problem
theorem total_bulbs_is_118 : total_bulbs_needed = 118 := by 
  sorry

end total_bulbs_is_118_l324_324106


namespace sin_double_angle_zero_l324_324429

-- Definitions of given conditions
def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

-- Statement to prove
theorem sin_double_angle_zero (α : ℝ) (h : f α = 1) : Real.sin (2 * α) = 0 := by
  sorry

end sin_double_angle_zero_l324_324429


namespace coefficient_ratio_is_4_l324_324720

noncomputable def coefficient_x3 := 
  let a := 60 -- Coefficient of x^3 in the expansion
  let b := Nat.choose 6 2 -- Binomial coefficient \binom{6}{2}
  a / b

theorem coefficient_ratio_is_4 : coefficient_x3 = 4 := by
  sorry

end coefficient_ratio_is_4_l324_324720


namespace measure_8_liters_with_buckets_l324_324008

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l324_324008


namespace quad_sin_theorem_l324_324375

-- Define the necessary entities in Lean
structure Quadrilateral (A B C D : Type) :=
(angleB : ℝ)
(angleD : ℝ)
(angleA : ℝ)

-- Define the main theorem
theorem quad_sin_theorem {A B C D : Type} (quad : Quadrilateral A B C D) (AC AD : ℝ) (α : ℝ) :
  quad.angleB = 90 ∧ quad.angleD = 90 ∧ quad.angleA = α → AD = AC * Real.sin α := 
sorry

end quad_sin_theorem_l324_324375


namespace min_process_improvements_correct_l324_324800

noncomputable def log := Real.log

noncomputable def min_process_improvements 
  (r_0 r_1 : ℝ)
  (lg2 lg3 : ℝ) : ℕ :=
  let t := -0.25
  let rn (n : ℕ) := r_0 + (r_1 - r_0) * 3^(0.25 * n + t)
  Nat.ceiling ((4 * (log 40) / log 3) + 1)

theorem min_process_improvements_correct : 
  ∀ (r_0 r_1 : ℝ) (lg2 lg3 : ℝ),
  r_0 = 2.25 →
  r_1 = 2.2 →
  lg2 = log 2 →
  lg3 = log 3 →
  min_process_improvements r_0 r_1 lg2 lg3 = 16 :=
by
  sorry

end min_process_improvements_correct_l324_324800


namespace distance_to_x_axis_of_point_on_hyperbola_l324_324738

noncomputable def distance_from_point_to_x_axis 
  (x y : ℝ) (h : x^2 - y^2 = 1) (angle : real.angle) 
  (angle_eq : angle = real.angle.pi / 3) : ℝ :=
|y|

theorem distance_to_x_axis_of_point_on_hyperbola 
  {x y : ℝ} (h : x^2 - y^2 = 1) (angle : real.angle)
  (angle_eq : angle = real.angle.pi / 3) : 
  distance_from_point_to_x_axis x y h angle angle_eq = sqrt 6 / 2 :=
sorry

end distance_to_x_axis_of_point_on_hyperbola_l324_324738


namespace tan_subtraction_formula_l324_324024

theorem tan_subtraction_formula 
  (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := 
by
  sorry

end tan_subtraction_formula_l324_324024


namespace coin_flips_probability_l324_324033

section 

-- Definition for the probability of heads in a single flip
def prob_heads : ℚ := 1 / 2

-- Definition for flipping the coin 5 times and getting heads on the first 4 flips and tails on the last flip
def prob_specific_sequence (n : ℕ) (k : ℕ) : ℚ := (prob_heads) ^ k * (prob_heads) ^ (n - k)

-- The main theorem which states the probability of the desired outcome
theorem coin_flips_probability : 
  prob_specific_sequence 5 4 = 1 / 32 :=
sorry

end

end coin_flips_probability_l324_324033


namespace range_of_x_l324_324784

theorem range_of_x (x : ℝ) : (∀ t : ℝ, -1 ≤ t ∧ t ≤ 3 → x^2 - (t^2 + t - 3) * x + t^2 * (t - 3) > 0) ↔ (x < -4 ∨ x > 9) :=
by
  sorry

end range_of_x_l324_324784


namespace median_length_of_pieces_is_198_l324_324594

   -- Define the conditions
   variables (A B C D E : ℕ)
   variables (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E)
   variables (avg_length : (A + B + C + D + E) = 640)
   variables (h_A_max : A ≤ 110)

   -- Statement of the problem (proof stub)
   theorem median_length_of_pieces_is_198 :
     C = 198 :=
   by
   sorry
   
end median_length_of_pieces_is_198_l324_324594


namespace factorize_polynomial_l324_324705

theorem factorize_polynomial {x : ℝ} : x^3 + 2 * x^2 - 3 * x = x * (x + 3) * (x - 1) :=
by sorry

end factorize_polynomial_l324_324705


namespace measure_of_angle_C_phi_no_solution_l324_324392

open Real

-- Definition and condition for finding angle C
def sin_squared_condition (A B C : ℝ) : Prop :=
  sin A ^ 2 + sin B ^ 2 + 4 * sin A * sin B * cos C = 0

def side_condition (a b c : ℝ) : Prop :=
  c ^ 2 = 3 * a * b

-- Theorem for finding angle C
theorem measure_of_angle_C (A B C a b c : ℝ)
  (h1 : sin_squared_condition A B C)
  (h2 : side_condition a b c) :
  cos C = -1 / 2 :=
sorry

-- Definition and conditions for the φ problem
def f (ω φ x : ℝ) : ℝ :=
  sin (ω * x + φ)

def monotonic_interval (ω φ : ℝ) : Prop :=
  ∀ x y, (π / 7 < x ∧ x < π / 2) → ((π / 7 < y ∧ y < π / 2) → x ≤ y → f ω φ x ≤ f ω φ y)

-- Theorem for the φ problem
theorem phi_no_solution (C φ : ℝ)
  (ω : ℕ)
  (h1 : |φ| < π / 2)
  (h2 : ω ≠ 0)
  (h3 : f ω φ C = -1 / 2)
  (h4 : monotonic_interval (ω.to_real) φ) :
  false :=
sorry

end measure_of_angle_C_phi_no_solution_l324_324392


namespace daughterAgeThreeYearsFromNow_l324_324961

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end daughterAgeThreeYearsFromNow_l324_324961


namespace factorial_divisibility_inequality_l324_324407

theorem factorial_divisibility_inequality
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h : (a! + b!) ∣ (a! * b!)) : 3 * a ≥ 2 * b + 2 := 
sorry

end factorial_divisibility_inequality_l324_324407


namespace prob_playerA_wins_4_to_1_prob_playerB_wins_and_match_lasts_more_than_5_games_dist_of_number_of_games_played_l324_324229

-- Define the conditions as assumptions
def playerA_wins_4_to_1 (p_win : ℝ) : Prop := comb 3 1 * (p_win ^ 4) * ((1 - p_win) ^ 1) = 1 / 8

def playerB_wins_and_match_lasts_more_than_5_games (p_win : ℝ) : Prop :=
  (comb 4 2 * (p_win ^ 4) * ((1 - p_win) ^ 2) + comb 5 3 * (p_win ^ 4) * ((1 - p_win) ^ 3)) = 15 / 64

def distribution_of_number_of_games (p_win : ℝ) : Prop :=
  (comb 3 0 * (p_win ^ 4) = 1 / 16) ∧
  (comb 4 1 * (p_win ^ 4) * ((1 - p_win) ^ 1) = 1 / 8) ∧
  (comb 5 2 * (p_win ^ 4) * ((1 - p_win) ^ 2) = 5 / 32) ∧
  (comb 6 3 * (p_win ^ 4) * ((1 - p_win) ^ 3) = 5 / 32)

-- Lean 4 statements that need proving
theorem prob_playerA_wins_4_to_1 (p_win : ℝ) (h : p_win = 1 / 2) : playerA_wins_4_to_1 p_win :=
by
  sorry

theorem prob_playerB_wins_and_match_lasts_more_than_5_games (p_win : ℝ) (h : p_win = 1 / 2) : 
  playerB_wins_and_match_lasts_more_than_5_games p_win :=
by 
  sorry

theorem dist_of_number_of_games_played (p_win : ℝ) (h : p_win = 1 / 2) : 
  distribution_of_number_of_games p_win :=
by 
  sorry

end prob_playerA_wins_4_to_1_prob_playerB_wins_and_match_lasts_more_than_5_games_dist_of_number_of_games_played_l324_324229


namespace determine_y_l324_324371

-- Definitions
noncomputable def acute_triangle (A B C : Type*) : Prop := sorry -- Define properties for acute triangle.
noncomputable def altitudes_divide_segments (AC_segment BE_segment EC_segment AD_segment DC_segment : ℕ) : Prop :=
  AC_segment = 7 ∧ BE_segment = 3 ∧ DC_segment = 3 ∧ AD_segment = 4

-- Theorem statement
theorem determine_y
  (A B C D E : Type*) [acute_triangle A B C]
  (AC BE EC AD DC : ℕ)
  (h1 : altitudes_divide_segments AC BE EC AD DC)
  (h2 : AC = 7)
  (h3 : AD = 4)
  (h4 : DC = 3)
  (y : ℕ)
  (h5 : BE = 3)
  (h6 : EC = y)
  (h7 : BC = y + 3)
  (proportion : 4 / 3 = 4 / y) :
  y = 3 :=
sorry

end determine_y_l324_324371


namespace find_f_3_l324_324498

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l324_324498


namespace expression_evaluation_l324_324699

noncomputable def evaluate_expression (a b c : ℚ) : ℚ :=
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7)

theorem expression_evaluation : 
  ∀ (a b c : ℚ), c = b - 11 → b = a + 3 → a = 5 → 
  (a + 2) ≠ 0 → (b - 3) ≠ 0 → (c + 7) ≠ 0 → 
  evaluate_expression a b c = 72 / 35 :=
by
  intros a b c hc hb ha h1 h2 h3
  rw [ha, hb, hc, evaluate_expression]
  -- The proof is not required.
  sorry

end expression_evaluation_l324_324699


namespace equilateral_triangle_perimeter_isosceles_triangle_leg_length_l324_324239

-- Definitions for equilateral triangle problem
def side_length_equilateral : ℕ := 12
def perimeter_equilateral := side_length_equilateral * 3

-- Definitions for isosceles triangle problem
def perimeter_isosceles : ℕ := 72
def base_length_isosceles : ℕ := 28
def leg_length_isosceles := (perimeter_isosceles - base_length_isosceles) / 2

-- Theorem statement
theorem equilateral_triangle_perimeter : perimeter_equilateral = 36 := 
by
  sorry

theorem isosceles_triangle_leg_length : leg_length_isosceles = 22 := 
by
  sorry

end equilateral_triangle_perimeter_isosceles_triangle_leg_length_l324_324239


namespace integral_evaluation_l324_324700

noncomputable def integral_value : ℝ :=
  ∫ x in -4..4, real.sqrt (16 - x^2)

theorem integral_evaluation : integral_value = 8 * π :=
by sorry

end integral_evaluation_l324_324700


namespace integer_solutions_count_l324_324341

theorem integer_solutions_count : 
  ∃ S : Set ℤ, S = {n : ℤ | (n-3)*(n+5) < 0} ∧ S.card = 7 :=
by
  sorry

end integer_solutions_count_l324_324341


namespace find_value_of_k_l324_324377

noncomputable theory

-- Definitions to match the conditions
def circle1 (m r1 : ℝ) : set ℝ := {p : ℝ × ℝ | (p.1 - m * r1)^2 + (p.2 - r1)^2 = r1^2}
def circle2 (m r2 : ℝ) : set ℝ := {p : ℝ × ℝ | (p.1 - m * r2)^2 + (p.2 - r2)^2 = r2^2}
def pointP : ℝ × ℝ := (3, 2)
def product_of_radii (r1 r2 : ℝ) : Prop := r1 * r2 = 13 / 2
def tangent_line (k : ℝ) : set ℝ := {p : ℝ × ℝ | p.2 = k * p.1}

-- The proof statement
theorem find_value_of_k 
  (m r1 r2 k : ℝ)
  (P_intersects : P ∈ circle1 m r1 ∧ P ∈ circle2 m r2)
  (radii_product : product_of_radii r1 r2)
  (tangent_to_x_axis : ∀ P ∈ tangent_line k, P.2 = 0) :
  k = 2 * real.sqrt 2 := 
sorry

end find_value_of_k_l324_324377


namespace cube_surface_area_l324_324602

/-
  We need to prove that the total surface area of the rearranged slices of a cube
  with given cuts is 10 square feet.
-/

-- Define the total surface area calculation given the heights from conditions
theorem cube_surface_area (hA hB hC : ℝ) (v : hA + hB + hC = 1) : 
  let top_and_bottom := (3 : ℝ),
      sides := (2 : ℝ),
      front_and_back := (2 : ℝ) * (hA + hB + hC) in
  (top_and_bottom + sides + front_and_back = 10) :=
by
  -- Replace heights with given values
  have hA_def : hA = 1/4 := by sorry
  have hB_def : hB = 1/6 := by sorry
  have hC_def : hC = 7/12 := by sorry
  sorry

end cube_surface_area_l324_324602


namespace count_numbers_containing_5_or_7_l324_324775

theorem count_numbers_containing_5_or_7 :
  let N := (1 : ℕ) ..
  let numbers := List.range' 1 (701)
  let contains_digit_5_or_7 (n : ℕ) : Prop :=
    (n.digits 10).contains 5 ∨ (n.digits 10).contains 7
  ∃ nums_subset : Finset ℕ, nums_subset.card = 244 ∧ ∀ n ∈ nums_subset, contains_digit_5_or_7 n :=
begin
  sorry
end

end count_numbers_containing_5_or_7_l324_324775


namespace race_graph_correct_l324_324387

-- Definitions based on the problem conditions
def tortoise_speed_constant : Prop := ∀ t1 t2 : ℝ, (distance_tortoise t2 - distance_tortoise t1) = c * (t2 - t1)
def hare_speed_changes : Prop := ∃ t1 t2 t3 t4 : ℝ, t1 < t2 < t3 < t4 ∧
  (distance_hare t2 - distance_hare t1) > (distance_hare t3 - distance_hare t2) ∧
  (distance_hare t4 - distance_hare t3) > 0
def tortoise_wins : Prop := ∃ t : ℝ, distance_tortoise t = finish_line ∧ distance_hare t < finish_line

-- Combined problem statement asserting that the graph of Option B correctly describes the race
theorem race_graph_correct (distance_tortoise distance_hare : ℝ → ℝ) 
  (finish_line c : ℝ) :
  tortoise_speed_constant distance_tortoise c →
  hare_speed_changes distance_hare →
  tortoise_wins distance_tortoise distance_hare finish_line →
  correct_graph OptionB :=
sorry

end race_graph_correct_l324_324387


namespace curve_max_value_ratio_l324_324324

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem curve_max_value_ratio (a b : ℝ) 
  (h1 : f a b 1 = 10) 
  (h2 : deriv (f a b) 1 = 0) : 
  a / b = -2/3 := 
by 
  sorry

end curve_max_value_ratio_l324_324324


namespace pizza_toppings_l324_324633

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l324_324633


namespace cylinder_surface_area_l324_324978

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (r_radius : r = 4) : 
  2 * π * r * (r + h) = 128 * π :=
by
  -- providing the proof steps is beyond the scope of this task
  sorry

end cylinder_surface_area_l324_324978


namespace triangle_angle_bisector_l324_324137

theorem triangle_angle_bisector
  (A B C D E : Type)
  [Plane (Triangle A B C)]
  (bisect_A : AngleBisector A (Side B C D))
  (bisect_B : AngleBisector B (Side A C E))
  (h : distance A E + distance B D = distance A B) :
  angle C = 60 :=
sorry

end triangle_angle_bisector_l324_324137


namespace pizza_toppings_count_l324_324622

theorem pizza_toppings_count :
  let toppings := 8 in
  let one_topping_pizzas := toppings in
  let two_topping_pizzas := (toppings.choose 2) in
  let three_topping_pizzas := (toppings.choose 3) in
  one_topping_pizzas + two_topping_pizzas + three_topping_pizzas = 92 :=
by sorry

end pizza_toppings_count_l324_324622


namespace sum_digits_base_9_of_1944_l324_324577

theorem sum_digits_base_9_of_1944 : 
  ∑ d in (Nat.digits 9 1944), d = 8 := 
sorry

end sum_digits_base_9_of_1944_l324_324577


namespace marbles_each_friend_gets_l324_324770

-- Definitions for the conditions
def total_marbles : ℕ := 100
def marbles_kept : ℕ := 20
def number_of_friends : ℕ := 5

-- The math proof problem
theorem marbles_each_friend_gets :
  (total_marbles - marbles_kept) / number_of_friends = 16 :=
by
  -- We include the proof steps within a by notation but stop at sorry for automated completion skipping proof steps
  sorry

end marbles_each_friend_gets_l324_324770


namespace tim_total_points_l324_324643

theorem tim_total_points :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6
  let tetrises := 4
  let total_points := singles * single_points + tetrises * tetris_points
  total_points = 38000 :=
by
  sorry

end tim_total_points_l324_324643


namespace problem1_problem2_l324_324740

variable (a b : ℝ)

-- (1) Prove a + b = 2 given the conditions
theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x : ℝ, abs (x - a) + abs (x + b) ≥ 2) : a + b = 2 :=
sorry

-- (2) Prove it is not possible for both a^2 + a > 2 and b^2 + b > 2 to hold simultaneously
theorem problem2 (h1: a + b = 2) (h2 : a^2 + a > 2) (h3 : b^2 + b > 2) : False :=
sorry

end problem1_problem2_l324_324740


namespace find_f_of_3_l324_324477

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l324_324477


namespace hyperbola_eccentricity_is_sqrt2_l324_324149

noncomputable theory

-- Definition of the parabola C1 and its focus F
def parabola (p : ℝ) : ℝ × ℝ → Prop :=
λ (P : ℝ × ℝ), P.1 ^ 2 = 2 * p * P.2

-- Definition of the hyperbola C2 and its foci F1, F2
def hyperbola (a b : ℝ) : ℝ × ℝ → Prop :=
λ (P: ℝ × ℝ), (P.1 ^ 2 / a ^ 2) - (P.2 ^ 2 / b ^ 2) = 1

-- The eccentricity of a hyperbola
def hyperbola_eccentricity (a b : ℝ) : ℝ :=
sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_sqrt2
    (p a b : ℝ)
    (F P F1 F2 : ℝ × ℝ)
    (hP1 : parabola p P)
    (hP2 : hyperbola a b P)
    (hF : F = (0, p / 2))
    (hF1 : F1 = (c, 0))
    (hF2 : F2 = (-c, 0))
    (hline : P.1 / P.2 = F1.1 / F1.2)
    (hc : c = b^2 / a)
    (hcollinear : ∀ {x y z : ℝ × ℝ}, 
        function.LinearlyIndependent ℝ ![[x, y], [x, z], [y, z]])
    (htangent : ∀ P', tangent parabola p P' = tangent hyperbola a b P')
    : hyperbola_eccentricity a b = sqrt 2 := sorry

end hyperbola_eccentricity_is_sqrt2_l324_324149


namespace screws_per_pile_l324_324850

theorem screws_per_pile :
  let screws_initial := 8 in
  let screws_to_buy := 2 * screws_initial in
  let total_screws := screws_initial + screws_to_buy in
  let screws_sections := 4 in
  total_screws / screws_sections = 6 :=
by
  sorry

end screws_per_pile_l324_324850


namespace lateral_area_is_24pi_l324_324319

-- Define the parameters: height h and base area A
def h : ℝ := 4
def A : ℝ := 9 * Real.pi

-- Define the radius based on base area
def R : ℝ := Real.sqrt 9

-- Now state the theorem for the lateral surface area
theorem lateral_area_is_24pi : 
  let r := Real.sqrt (A / Real.pi) in 
  (2 * Real.pi * r * h) = 24 * Real.pi :=
by
  sorry

end lateral_area_is_24pi_l324_324319


namespace intersection_of_BE_and_CF_eq_P_l324_324818

variables {A B C P E F : Type}
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C]
variables [AddCommGroup V] [Module ℝ V]

noncomputable def vec_E (a c : V) : V :=
  (3/5 : ℝ) • a + (2/5 : ℝ) • c

noncomputable def vec_F (a b : V) : V :=
  (2/5 : ℝ) • a + (3/5 : ℝ) • b

theorem intersection_of_BE_and_CF_eq_P 
  (hE : vec_E (A : V) (C : V) = E : V)
  (hF : vec_F (A : V) (B : V) = F : V)
  (hP : P = intersection_of (B : V) (E : V) (C : V) (F : V)):
  ∃ (x y z : ℝ), x + y + z = 1 ∧
  (x • (A : V) + y • (B : V) + z • (C : V)) = (15 / 22 : ℝ) • (A : V) + (3 / 22 : ℝ) • (B : V) + (4 / 22 : ℝ) • (C : V) :=
sorry

end intersection_of_BE_and_CF_eq_P_l324_324818


namespace probability_greater_than_4_l324_324749

open MeasureTheory

-- Define the probability density function of a normal distribution
def f (x : ℝ) : ℝ := (1 / real.sqrt (2 * real.pi)) * real.exp (-(x - 2)^2 / 2)

-- Define the integral condition
def integral_condition : Prop :=
  ∫ x in 0..2, f x = 1 / 3

-- Theorem statement to be proved
theorem probability_greater_than_4 (h : integral_condition) : 
  ∫ x in 4..real.infinity, f x = 1 / 6 :=
sorry

end probability_greater_than_4_l324_324749


namespace quadratic_inequality_sufficient_necessary_condition_l324_324157

theorem quadratic_inequality_sufficient_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 2 > 0) ↔ -2 * real.sqrt 2 < m ∧ m < 2 * real.sqrt 2 :=
by
  sorry

end quadratic_inequality_sufficient_necessary_condition_l324_324157


namespace find_front_axle_wheel_count_l324_324908

theorem find_front_axle_wheel_count 
  (t x f : ℕ) 
  (h_toll_formula : t = 1.50 + 1.50 * (x - 2)) 
  (h_toll : t = 6) 
  (h_total_wheels : 18 = f + 4 * (x - 1)) : 
  f = 2 :=
  by
  sorry

end find_front_axle_wheel_count_l324_324908


namespace algorithm_description_method_filled_l324_324527

-- Definitions for the conditions
def methods_to_describe_algorithm := ["Natural language", "Flowcharts", "Pseudocode"]

-- The proof statement
theorem algorithm_description_method_filled :
  methods_to_describe_algorithm.nth 1 = some "Flowcharts" := by
  -- proof to be completed
  sorry

end algorithm_description_method_filled_l324_324527


namespace difference_of_numbers_l324_324541

theorem difference_of_numbers (x y : ℕ) (h₁ : x + y = 50) (h₂ : Nat.gcd x y = 5) :
  (x - y = 20 ∨ y - x = 20 ∨ x - y = 40 ∨ y - x = 40) :=
sorry

end difference_of_numbers_l324_324541


namespace cylinder_surface_area_l324_324980

theorem cylinder_surface_area (h : ℝ) (r : ℝ) (h_eq : h = 12) (r_eq : r = 4) : 
  2 * π * r * (r + h) = 128 * π := 
by
  rw [h_eq, r_eq]
  sorry

end cylinder_surface_area_l324_324980


namespace minimum_force_to_submerge_cube_l324_324572

noncomputable def volume_cm3_to_m3 (v_cm: ℝ) : ℝ := v_cm * 10 ^ (-6)
noncomputable def mass (density: ℝ) (volume: ℝ): ℝ := density * volume
noncomputable def gravitational_force (mass: ℝ) (g: ℝ): ℝ := mass * g
noncomputable def buoyant_force (density_water: ℝ) (volume: ℝ) (g: ℝ): ℝ := density_water * volume * g

theorem minimum_force_to_submerge_cube 
    (V_cm: ℝ) (ρ_cube: ℝ) (ρ_water: ℝ) (g: ℝ) 
    (hV: V_cm = 10) (hρ_cube: ρ_cube = 700) (hρ_water: ρ_water = 1000) (hg: g = 10) :
    (buoyant_force ρ_water (volume_cm3_to_m3 V_cm) g - gravitational_force (mass ρ_cube (volume_cm3_to_m3 V_cm)) g) = 0.03 :=
by
    sorry

end minimum_force_to_submerge_cube_l324_324572


namespace volume_of_rectangular_solid_l324_324639

theorem volume_of_rectangular_solid 
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : c * a = 6) :
  a * b * c = 30 := 
by
  -- sorry placeholder for the proof
  sorry

end volume_of_rectangular_solid_l324_324639


namespace compare_game_x_and_game_y_l324_324209

noncomputable def prob_heads : ℝ := 3 / 4
noncomputable def prob_tails : ℝ := 1 / 4

def prob_game_x_win : ℝ :=
  4 * (prob_heads^4 * prob_tails)

def prob_game_y_win : ℝ :=
  prob_heads^6 + prob_tails^6

theorem compare_game_x_and_game_y :
  prob_game_x_win - prob_game_y_win = 298 / 2048 :=
by
  unfold prob_game_x_win prob_game_y_win prob_heads prob_tails
  norm_num
  sorry

end compare_game_x_and_game_y_l324_324209


namespace expand_product_l324_324703

theorem expand_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7 * x^3 + 12 := 
  sorry

end expand_product_l324_324703


namespace volume_sphere_same_radius_l324_324911

noncomputable theory

-- Define the volume of the cylinder
def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Define the volume of the sphere
def volume_of_sphere (r : ℝ) : ℝ := (4/3) * π * r^3

-- Given the volume relation for a cylinder
def cylinder_condition (Vc : ℝ) (r : ℝ) (h : ℝ) : Prop := Vc = π * r^2 * h

-- Prove the volume of sphere for the same radius
theorem volume_sphere_same_radius (r h : ℝ) (Vc : ℝ) (h_eq_r : h = r)
    (cylinder_volume_condition : cylinder_condition Vc r h) : 
    volume_of_sphere r = 96 * π := 
by
    -- Given volume Vc is 72π
    have Vc_eq : Vc = 72 * π, from cylinder_volume_condition
    -- From h = r and simplifying we know r = (72)^(1/3) from above we can conclude the volume
    sorry

end volume_sphere_same_radius_l324_324911


namespace height_water_in_cylinder_l324_324663

-- Define the given conditions
def base_radius_cone := 12  -- Radius of the base of the cone in cm
def height_cone := 18       -- Height of the cone in cm
def base_radius_cylinder := 24  -- Radius of the base of the cylinder in cm

-- Volume formula for a cone
def volume_cone (r : ℝ) (h : ℝ) := (1/3) * real.pi * r^2 * h

-- Volume formula for a cylinder
def volume_cylinder (r : ℝ) (h : ℝ) := real.pi * r^2 * h

-- Problem statement
theorem height_water_in_cylinder :
  volume_cone base_radius_cone height_cone = volume_cylinder base_radius_cylinder 1.5 :=
by
  sorry

end height_water_in_cylinder_l324_324663


namespace probability_of_r25_to_r35_l324_324299

/-- 
Given a sequence of 50 distinct real numbers initially in random order,
the probability that the number initially in position r_25 will end up in
the 35th position after one bubble pass is 1 / 1190. 
-/
theorem probability_of_r25_to_r35
  (seq : Fin 50 → ℝ) (distinct : Function.Injective seq) : 
  let event := { s : Fin 50 → ℝ | in_bubble_position seq s 25 35} in
  probability event = 1 / 1190 :=
sorry

end probability_of_r25_to_r35_l324_324299


namespace find_f_of_3_l324_324495

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l324_324495


namespace line_equation_l324_324063

def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let m := (y₂ - y₁) / (x₂ - x₁)
  y - y₁ = m * (x - x₁)

noncomputable def is_trisection_point (A B QR : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (qx, qy) := QR
  (qx = (2 * x₂ + x₁) / 3 ∧ qy = (2 * y₂ + y₁) / 3) ∨
  (qx = (x₂ + 2 * x₁) / 3 ∧ qy = (y₂ + 2 * y₁) / 3)

theorem line_equation (A B P Q : ℝ × ℝ)
  (hA : A = (3, 4))
  (hB : B = (-4, 5))
  (hP : is_trisection_point B A P)
  (hQ : is_trisection_point B A Q) :
  line_through A P 1 3 ∨ line_through A P 2 1 → 
  (line_through A P 3 4 → P = (1, 3)) ∧ 
  (line_through A P 2 1 → P = (2, 1)) ∧ 
  (line_through A P x y → x - 4 * y + 13 = 0) := 
by 
  sorry

end line_equation_l324_324063


namespace identify_multiple_l324_324914

-- Define the digits
def digits : Set ℕ := {1, 2, 3, 4, 5}

-- Define the property of forming 3-digit numbers without repetition
def valid_3_digit_numbers (d : Set ℕ) : Nat :=
  if h: ∀ x ∈ d, x ∈ digits ∧ (d.card = 3) ∧ Function.Injective (λ n : ℕ, n) 
  then d.prod else 0

-- Define the condition of having exactly 8 different multiples
def count_valid_multiples : ℕ := 8

-- Define the property of these multiples
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

-- The main theorem to prove
theorem identify_multiple (h1 : digits.card = 5)
                          (h2 : Function.Injective (λ n, n ∈ digits))
                          (h3 : valid_3_digit_numbers digits = count_valid_multiples)
: ∃ n, ∀ m ∈ {x | x  = 5 * m} ∧ valid_3_digit_numbers {x | x ≠ m ∨ (m ∈ digits ∧ m < 10)}, is_multiple_of m n := sorry

end identify_multiple_l324_324914


namespace resulting_parabola_is_correct_l324_324171

-- Conditions
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x - a)

def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ y, f y + b

-- The equivalent proof problem
theorem resulting_parabola_is_correct :
  (translate_up (translate_right initial_parabola 1) 3) = (λ x, (x - 1)^2 + 1) :=
sorry

end resulting_parabola_is_correct_l324_324171


namespace number_of_interesting_sums_l324_324686

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

noncomputable def prime_sequence : ℕ → ℕ 
| 0     := 3
| (n + 1) := prime_sequence n + (nth_prime (n + 1))

def is_interesting_sum (n : ℕ) : Prop := 
  is_prime (prime_sequence n) ∧ prime_sequence n > 50

theorem number_of_interesting_sums : 
  (finset.range 15).filter is_interesting_sum).card = 2 :=
by sorry

end number_of_interesting_sums_l324_324686


namespace sine_axis_translation_l324_324518

open Real

def translated_sine_axis_sym :=
  ∃ k : ℤ, ∀ x : ℝ, x = k * (π / 2) + (π / 4)

theorem sine_axis_translation (x : ℝ) :
  translated_sine_axis_sym →
  ∃ k : ℤ, x = (k * (π / 2) + π / 4) →
  x = π / 4 :=
begin
  sorry
end

end sine_axis_translation_l324_324518


namespace sets_are_equal_l324_324846

def setA : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def setB : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l324_324846


namespace worked_days_proof_l324_324216

theorem worked_days_proof (W N : ℕ) (hN : N = 24) (h0 : 100 * W = 25 * N) : W + N = 30 :=
by
  sorry

end worked_days_proof_l324_324216


namespace residual_is_correct_l324_324536

-- Define the regression equation and sample point, and calculate the residual
def regression_equation (x : ℝ) : ℝ := 2.5 * x + 0.31

def actual_value : ℝ := 1.2

def sample_point : ℝ := 4.0

def predicted_value : ℝ := regression_equation sample_point

def residual : ℝ := actual_value - predicted_value

theorem residual_is_correct : residual = -9.11 := by
  -- Proof goes here
  sorry

end residual_is_correct_l324_324536


namespace sam_initial_balloons_l324_324448

theorem sam_initial_balloons:
  ∀ (S : ℝ), (S - 5.0 + 7.0 = 8) → S = 6.0 :=
by
  intro S h
  sorry

end sam_initial_balloons_l324_324448


namespace smallest_number_of_brownies_l324_324768

noncomputable def total_brownies (m n : ℕ) : ℕ := m * n
def perimeter_brownies (m n : ℕ) : ℕ := 2 * m + 2 * n - 4
def interior_brownies (m n : ℕ) : ℕ := (m - 2) * (n - 2)

theorem smallest_number_of_brownies : 
  ∃ (m n : ℕ), 2 * interior_brownies m n = perimeter_brownies m n ∧ total_brownies m n = 36 :=
by
  sorry

end smallest_number_of_brownies_l324_324768


namespace area_of_triangle_l324_324302

theorem area_of_triangle (a b : ℝ) (C : ℝ) (ha : a = 4) (hb : b = 5) (hC : C = real.pi / 3) : 
  (1 / 2) * a * b * real.sin C = 5 * real.sqrt 3 :=
by
  subst ha
  subst hb
  subst hC
  sorry

end area_of_triangle_l324_324302


namespace avg_weights_b_c_l324_324462

noncomputable def weights : Type := ℝ

variables (A B C : weights)

theorem avg_weights_b_c
  (h1 : (A + B + C) / 3 = 43)
  (h2 : (A + B) / 2 = 40)
  (h3 : B = 37) :
  (B + C) / 2 = 43 :=
sorry

end avg_weights_b_c_l324_324462


namespace problem_l324_324132

noncomputable
def harmonic_mean_condition (x y z : ℝ) : Prop :=
  (1 / x) + (1 / y) + (1 / z) = 0.75

noncomputable
theorem problem (x y z : ℝ) 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 216)
  (h3 : harmonic_mean_condition x y z) :
  x^2 + y^2 + z^2 = 576 := 
by 
  -- Proof goes here
  sorry

end problem_l324_324132


namespace unique_frog_arrangement_l324_324916

def frog_colors := {green, red, blue, yellow}
def frog_count : frog_colors → ℕ
| green := 4
| red := 3
| blue := 2
| yellow := 1

-- Conditions
def validate_frog_arrangement (arrangement : list frog_colors) : Prop :=
  let adjacent_pairs := arrangement.zip arrangement.tail
  (all (λ (pair : frog_colors × frog_colors), pair.1 ≠ pair.2) adjacent_pairs) ∧
  ∀(a b : frog_colors), (a = green ∧ b = red) ∨ (a = red ∧ b = green) → ¬((a, b) ∈ adjacent_pairs)

theorem unique_frog_arrangement :
  ∃ arrangement : list frog_colors, validate_frog_arrangement arrangement :=
sorry

end unique_frog_arrangement_l324_324916


namespace pizza_varieties_l324_324196

-- Definition of the problem conditions
def base_flavors : ℕ := 4
def topping_options : ℕ := 4  -- No toppings, extra cheese, mushrooms, both

-- The math proof problem statement
theorem pizza_varieties : base_flavors * topping_options = 16 := by 
  sorry

end pizza_varieties_l324_324196


namespace events_a_b_mutually_exclusive_events_a_c_not_independent_l324_324245

section
variables (A B : set ℕ)
def balls_A := {1, 2, 3, 4}
def balls_B := {5, 6, 7, 8}

def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0
def event_B (x y : ℕ) : Prop := (x + y) = 9
def event_C (x y : ℕ) : Prop := (x + y) > 9

theorem events_a_b_mutually_exclusive :
  ∀ x y, x ∈ balls_A → y ∈ balls_B → event_A x y → ¬event_B x y :=
begin
  intros x y hx hy hA hB,
  rw event_A at hA,
  rw event_B at hB,
  have h1 : (x + y) % 2 = 0 := hA,
  have h2 : x + y = 9 := hB,
  have h3 : 9 % 2 = 1 := dec_trivial,
  rw h2 at h1,
  rw h3 at h1,
  exact ne_of_lt zero_lt_one h1,
end

theorem events_a_c_not_independent :
  ∃ x y z w,
    x ∈ balls_A ∧ y ∈ balls_B ∧
    z ∈ balls_A ∧ w ∈ balls_B ∧
    event_A x y ∧ event_C x y ∧
    event_A z w ∧ ¬event_C z w :=
begin
  use [2, 8, 1, 7],
  split; try {split; try {split; try {split}}},
  { dec_trivial },
  { dec_trivial },
  { dec_trivial },
  { dec_trivial },
  { rw event_A, exact dec_trivial },
  { rw event_C, exact dec_trivial },
  { rw event_A, exact dec_trivial },
  { rw event_C, exact sorry } -- This part will depend on the specific pairs examined
end
end

end events_a_b_mutually_exclusive_events_a_c_not_independent_l324_324245


namespace hypotenuse_of_45_45_90_triangle_l324_324859

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h₁ : a = 12) (h₂ : ∠ABC = 45) : hypotenuse = a * sqrt 2 := 
by 
  sorry

end hypotenuse_of_45_45_90_triangle_l324_324859


namespace probability_B_second_shot_probability_A_i_th_shot_expected_shots_A_l324_324563

theorem probability_B_second_shot (p_A p_B p_first_A p_first_B : ℝ) 
  (h1 : p_A = 0.6)
  (h2 : p_B = 0.8)
  (h3 : p_first_A = 0.5)
  (h4 : p_first_B = 0.5) :
  (p_first_A * (1 - p_A) + p_first_B * p_B) = 0.6 :=
by 
  simp [h1, h2, h3, h4]
  sorry

theorem probability_A_i_th_shot (i : ℕ) (p_A p_B p_first_A p_first_B : ℝ)
  (h1 : p_A = 0.6)
  (h2 : p_B = 0.8)
  (h3 : p_first_A = 0.5)
  (h4 : p_first_B = 0.5) :
  let P : ℕ → ℝ := λ n, (1/3) + (1/6) * (2/5)^(n-1)
  P i = (1/3) + (1/6) * (2/5)^(i-1) :=
by 
  simp [h1, h2, h3, h4]
  sorry

theorem expected_shots_A (n : ℕ) (p_A p_B p_first_A p_first_B : ℝ)
  (h1 : p_A = 0.6)
  (h2 : p_B = 0.8)
  (h3 : p_first_A = 0.5)
  (h4 : p_first_B = 0.5) :
  let E_Y : ℕ → ℝ := λ n, (5/18) * (1 - (2/5)^n) + (n/3)
  E_Y n = (5/18) * (1 - (2/5)^n) + (n/3) :=
by 
  simp [h1, h2, h3, h4]
  sorry

end probability_B_second_shot_probability_A_i_th_shot_expected_shots_A_l324_324563


namespace magnitude_of_complex_number_l324_324679

theorem magnitude_of_complex_number (a b : ℝ) (z1 z2 : ℂ) 
  (h1 : z1 = a + 4 * complex.I) (h2 : z2 = 3 + b * complex.I)
  (h3 : complex.abs (z1 + z2).im = 0) (h4 : complex.abs (z1 - z2).re = 0) :
  complex.abs (a + b * complex.I) = 5 :=
by
  sorry

end magnitude_of_complex_number_l324_324679


namespace perfect_square_trinomial_k_l324_324029

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a b : ℤ, (a = 1 ∨ a = -1) ∧ (b = 3 ∨ b = -3) ∧ (x : ℤ) → x^2 - k*x + 9 = (a*x + b)^2) ↔ (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l324_324029


namespace transmit_data_time_l324_324696

def total_chunks (blocks: ℕ) (chunks_per_block: ℕ) : ℕ := blocks * chunks_per_block

def transmit_time (total_chunks: ℕ) (chunks_per_second: ℕ) : ℕ := total_chunks / chunks_per_second

def time_in_minutes (transmit_time_seconds: ℕ) : ℕ := transmit_time_seconds / 60

theorem transmit_data_time :
  ∀ (blocks chunks_per_block chunks_per_second : ℕ),
    blocks = 150 →
    chunks_per_block = 256 →
    chunks_per_second = 200 →
    time_in_minutes (transmit_time (total_chunks blocks chunks_per_block) chunks_per_second) = 3 := by
  intros
  sorry

end transmit_data_time_l324_324696


namespace average_typing_speed_l324_324467

theorem average_typing_speed :
  let rudy := 64
  let joyce := 76
  let gladys := 91
  let lisa := 80
  let mike := 89
  let total := rudy + joyce + gladys + lisa + mike
  let average := total / 5
  in average = 80 := by
  sorry

end average_typing_speed_l324_324467


namespace number_of_digits_of_m_times_n_l324_324425

def m := 10 ^ 77 - 1
def n := 7 * (10 ^ 98 + 10 ^ 97 + 10 ^ 96 + ... + 10 + 1) -- This represents a number consisting of 99 sevens.

theorem number_of_digits_of_m_times_n : (number_of_digits (m * n) = 176) := sorry

end number_of_digits_of_m_times_n_l324_324425


namespace max_omega_l324_324329

open Real

-- Define the function f(x) = sin(ωx + φ)
noncomputable def f (ω φ x : ℝ) := sin (ω * x + φ)

-- ω > 0 and |φ| ≤ π / 2
def condition_omega_pos (ω : ℝ) := ω > 0
def condition_phi_bound (φ : ℝ) := abs φ ≤ π / 2

-- x = -π/4 is a zero of f(x)
def condition_zero (ω φ : ℝ) := f ω φ (-π/4) = 0

-- x = π/4 is the axis of symmetry for the graph of y = f(x)
def condition_symmetry (ω φ : ℝ) := 
  ∀ x : ℝ, f ω φ (π/4 - x) = f ω φ (π/4 + x)

-- f(x) is monotonic in the interval (π/18, 5π/36)
def condition_monotonic (ω φ : ℝ) := 
  ∀ x₁ x₂ : ℝ, π/18 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * π / 36 
  → f ω φ x₁ ≤ f ω φ x₂

-- Prove that the maximum value of ω satisfying all the conditions is 9
theorem max_omega (ω : ℝ) (φ : ℝ)
  (h1 : condition_omega_pos ω)
  (h2 : condition_phi_bound φ)
  (h3 : condition_zero ω φ)
  (h4 : condition_symmetry ω φ)
  (h5 : condition_monotonic ω φ) :
  ω ≤ 9 :=
sorry

end max_omega_l324_324329


namespace product_evaluation_l324_324946

theorem product_evaluation : 
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_evaluation_l324_324946


namespace area_triangle_RTO_l324_324058

theorem area_triangle_RTO
  (E F G H R S T U O : Type)
  (parallelogram_EFGH : Parallelogram E F G H)
  (area_EFGH : Real := m)
  (line_ER : Line E R)
  (bisect_ER_S: Bisects line_ER (F, G) S)
  (line_HT : Line H T)
  (bisect_HT_U: Bisects line_HT (E, F) U)
  (extend_ER_R: Meets line_ER (Extend EH) R)
  (extend_HT_T: Meets line_HT (Extend EH) T)
  (intersect_OR_HT: Intersect ER HT O) :
  area Triangle R T O = m / 2 :=
sorry

end area_triangle_RTO_l324_324058


namespace intersections_case_a_intersections_case_b_intersections_case_c_l324_324117

open scoped real

-- Define the regular thirty-sided polygon
def regularThirtySidedPolygon : Type := ℝ

-- Define the property of line intersections
def intersects_at_single_point (a b c : regularThirtySidedPolygon) : Prop := sorry

-- Case (a): Prove diagonals intersect at a single point
theorem intersections_case_a (A1 A2 A4 A7 A9 A23 : regularThirtySidedPolygon) :
  intersects_at_single_point A1 A7 A2 ∧
  intersects_at_single_point A2 A9 A4 ∧
  intersects_at_single_point A4 A23 A1 :=
sorry

-- Case (b): Prove diagonals intersect at a single point
theorem intersections_case_b (A1 A2 A4 A7 A15 A29 : regularThirtySidedPolygon) :
  intersects_at_single_point A1 A7 A2 ∧
  intersects_at_single_point A2 A15 A4 ∧
  intersects_at_single_point A4 A29 A1 :=
sorry

-- Case (c): Prove diagonals intersect at a single point
theorem intersections_case_c (A1 A2 A10 A13 A15 A29 : regularThirtySidedPolygon) :
  intersects_at_single_point A1 A13 A2 ∧
  intersects_at_single_point A2 A15 A10 ∧
  intersects_at_single_point A10 A29 A1 :=
sorry

-- Seamlessly importing all necessary math libraries for lean validation
open scoped mathlib

end intersections_case_a_intersections_case_b_intersections_case_c_l324_324117


namespace lisa_total_miles_flown_l324_324435

-- Definitions based on given conditions
def distance_per_trip : ℝ := 256.0
def number_of_trips : ℝ := 32.0
def total_miles_flown : ℝ := 8192.0

-- Lean statement asserting the equivalence
theorem lisa_total_miles_flown : 
    (distance_per_trip * number_of_trips = total_miles_flown) :=
by 
    sorry

end lisa_total_miles_flown_l324_324435


namespace find_f_of_3_l324_324490

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l324_324490


namespace candy_cost_proof_l324_324213

theorem candy_cost_proof (x : ℝ) (h1 : 10 ≤ 30) (h2 : 0 ≤ 5) (h3 : 0 ≤ 6) 
(h4 : 10 * x + 20 * 5 = 6 * 30) : x = 8 := by
  sorry

end candy_cost_proof_l324_324213


namespace books_in_identical_bags_l324_324669

theorem books_in_identical_bags :
  let n_books := 5
  let n_bags := 4
  let ways := 
    -- Case 1: All 5 books in one bag.
    1
    -- Case 2: Four books in one bag, one book in another.
    + 5
    -- Case 3: Three books in one bag, and two in another.
    + 5
    -- Case 4: Three books in one bag, one in another, and one in a third.
    + 10
    -- Case 5: Two books in each of two bags, and one in a third.
    + 10 * 3
  in
  ways = 51 :=
begin
  sorry
end

end books_in_identical_bags_l324_324669


namespace minimum_dot_product_l324_324309

   variable {ℝ : Type*} [NormedField ℝ] [InnerProductSpace ℝ ℝ] -- To handle real numbers and dot products.
   variable (e1 e2 : ℝ) (t : ℝ)

   -- We assume e1 and e2 are unit vectors and the angle between them is π/3.
   axiom unit_vectors (u1 u2 : ℝ) : ∥u1∥ = 1 ∧ ∥u2∥ = 1 
   axiom angle_between_unit_vectors (u1 u2 : ℝ) : real_inner u1 u2 = 1/2

   theorem minimum_dot_product : 
     ∃ t : ℝ, let dp := ((e1 + t • e2) ⬝ (t • e1 + e2)) in 
     dp = -3/2 :=
   sorry
   
end minimum_dot_product_l324_324309


namespace tim_score_l324_324644

theorem tim_score :
  let single_line_points := 1000
  let tetris_points := 8 * single_line_points
  let singles_scored := 6
  let tetrises_scored := 4
  in singles_scored * single_line_points + tetrises_scored * tetris_points = 38000 := by
  sorry

end tim_score_l324_324644


namespace grid_diagonal_intersections_l324_324365

open Nat

theorem grid_diagonal_intersections (m n : ℕ) (hm : m = 12) (hn : n = 17) :
  (m + n - gcd m n - 1) = 29 :=
by
  rw [hm, hn]
  rfl

end grid_diagonal_intersections_l324_324365


namespace vector_dot_product_problem_l324_324745

variable (a b : EuclideanSpace ℝ (Fin 3))
variable (θ : ℝ)

noncomputable def angle := Real.pi / 4
noncomputable def norm_a := 3 * Real.sqrt 2
noncomputable def norm_b := 2

axiom angle_condition : θ = Real.pi / 4
axiom norm_a_condition : ‖a‖ = 3 * Real.sqrt 2
axiom norm_b_condition : ‖b‖ = 2

theorem vector_dot_product_problem
  (ha : ‖a‖ = 3 * Real.sqrt 2)
  (hb : ‖b‖ = 2)
  (hθ : θ = Real.pi / 4) :
  ((2 • a + b) • b) = 16 :=
by
  sorry

end vector_dot_product_problem_l324_324745


namespace hypotenuse_of_45_45_90_triangle_l324_324858

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h₁ : a = 12) (h₂ : ∠ABC = 45) : hypotenuse = a * sqrt 2 := 
by 
  sorry

end hypotenuse_of_45_45_90_triangle_l324_324858


namespace derivative_at_one_l324_324455

theorem derivative_at_one (f : ℝ → ℝ) (h_differentiable : Differentiable ℝ f)
  (h_limit : ∀ Δx : ℝ, Δx ≠ 0 → (tendsto (λ n : ℕ, (f (1 + 2 * Δx) - f (1)) / Δx) at_top (𝓝 (-2)))) :
  deriv f 1 = -1 :=
sorry

end derivative_at_one_l324_324455


namespace find_incorrect_value_of_observation_l324_324523

noncomputable def incorrect_observation_value (mean1 : ℝ) (mean2 : ℝ) (n : ℕ) : ℝ :=
  let old_sum := mean1 * n
  let new_sum := mean2 * n
  let correct_value := 45
  let incorrect_value := (old_sum - new_sum + correct_value)
  (incorrect_value / -1)

theorem find_incorrect_value_of_observation :
  incorrect_observation_value 36 36.5 50 = 20 :=
by
  -- By the problem setup, incorrect_observation_value 36 36.5 50 is as defined in the proof steps.
  -- As per the proof steps and calculation, incorrect_observation_value 36 36.5 50 should evaluate to 20.
  sorry

end find_incorrect_value_of_observation_l324_324523


namespace dot_product_result_l324_324765

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 1)

theorem dot_product_result : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_result_l324_324765


namespace evaluate_expression_l324_324026

theorem evaluate_expression (x : ℝ) (h : x > 0) : 
    |x + (real.sqrt ((x + 2)^2)) - 3| =
    if x ≥ 0.5 then 2 * x - 1 else 1 - 2 * x :=
by
  sorry

end evaluate_expression_l324_324026


namespace oscar_cookie_baking_time_l324_324269

theorem oscar_cookie_baking_time : 
  (1 / 5) + (1 / 6) + (1 / o) - (1 / 4) = (1 / 8) → o = 120 := by
  sorry

end oscar_cookie_baking_time_l324_324269


namespace coeff_a_neg_one_in_binomial_expansion_is_zero_l324_324385

theorem coeff_a_neg_one_in_binomial_expansion_is_zero :
  let f := fun (a : ℕ) => ∑ k in finset.range 9, (binomial 8 k) * (-1)^k * a^(8 - 2*k)
  ∀ a : ℕ, polynomial.C (f a) = 0 := by
sorry

end coeff_a_neg_one_in_binomial_expansion_is_zero_l324_324385


namespace tangent_line_t_l324_324039

theorem tangent_line_t (x0 t : ℝ) (h_tangent : ∀ x y : ℝ, y = x + t ↔ y = exp x) :
  t = 1 :=
by
  have h1 : exp x0 = x0 + t := h_tangent x0 (exp x0)
  have h2 : exp x0 = 1 := by sorry
  have h3 : x0 = 0 := by sorry
  have h4 : t = 1 := by sorry
  exact h4

end tangent_line_t_l324_324039


namespace angle_C_measure_phi_not_exist_l324_324393

section triangle_problem

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
def sides_opposite_angles (a b c A B C : ℝ) : Prop :=
  ∃ (θA θB θC : ℝ), 
      (θA = θA.mod (2 * Real.pi)) ∧ (θB = θB.mod (2 * Real.pi)) ∧ (θC = θC.mod (2 * Real.pi)) ∧ 
      (a = is length of sides opposite to θA) ∧
      (b = is length of sides opposite to θB) ∧
      (c = is length of sides opposite to θC) 

def sine_squared_eq (A B C : ℝ) : Prop :=
  Real.sin(A)^2 + Real.sin(B)^2 + 4 * Real.sin(A) * Real.sin(B) * Real.cos(C) = 0

def c_squared_eq (a b c : ℝ) : Prop :=
  c^2 = 3 * a * b

-- Question 1 answer
theorem angle_C_measure (h1: sides_opposite_angles a b c A B C) (h2: sine_squared_eq A B C) (h3: c_squared_eq a b c) :
  C = 2 * Real.pi / 3 := by 
  sorry

-- Variables for question 2
variable {x : ℝ}
def f_question2 (ω ϕ : ℝ) (x : ℝ) : ℝ := Real.sin(ω * x + ϕ)

-- Conditions Question 2
def f_monotonic (ω ϕ : ℝ) : Prop :=
  ∃ (C : ℝ), (2 * Real.pi / 3 = C ∧ f_question2 ω ϕ C = - 1 / 2) ∧ (∀ (x1 x2 ∈ Ioo (Real.pi / 7) (Real.pi / 2)), f_question2 ω ϕ x1 ≤ f_question2 ω ϕ x2 ∨ f_question2 ω ϕ x1 ≥ f_question2 ω ϕ x2 )

-- Question 2 answer
theorem phi_not_exist (h1: ∃ ω ∈ {1, 2}, ω > 0) (h2: ∀ ω, f_monotonic ω ϕ):
  ¬(∃ (ϕ : ℝ), ϕ.abs < Real.pi / 2) := by 
  sorry

end triangle_problem

end angle_C_measure_phi_not_exist_l324_324393


namespace pizza_topping_count_l324_324613

theorem pizza_topping_count (n : ℕ) (hn : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by
  sorry

end pizza_topping_count_l324_324613


namespace pizza_topping_count_l324_324609

theorem pizza_topping_count (n : ℕ) (hn : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by
  sorry

end pizza_topping_count_l324_324609


namespace min_knights_in_village_l324_324370

theorem min_knights_in_village :
  ∃ (K L : ℕ), K + L = 7 ∧ 2 * K * L = 24 ∧ K ≥ 3 :=
by
  sorry

end min_knights_in_village_l324_324370


namespace probability_heads_at_least_twice_in_five_tosses_l324_324985

noncomputable def fair_coin := Prob.one_half

def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_heads_at_least_twice_in_five_tosses :
  let p := fair_coin in
  1 - (binomialProbability 5 0 p) - (binomialProbability 5 1 p) = 0.8125 := by
  sorry

end probability_heads_at_least_twice_in_five_tosses_l324_324985


namespace bakery_difference_l324_324672

theorem bakery_difference (cakes_sold pastries_sold : ℕ) (h1 : cakes_sold = 158) (h2 : pastries_sold = 147) : cakes_sold - pastries_sold = 11 := by
  rw [h1, h2]
  exact Nat.sub_self (158 - 147:=11)

# Now the Lean 4 theorem is equivalent to the given problem, with conditions and expected results translated directly.

end bakery_difference_l324_324672


namespace correct_propositions_l324_324287

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

-- Proposition 2: Symmetry about the line x = -3π/4
def proposition_2 : Prop := ∀ x, f (x + 3 * Real.pi / 4) = f (-x)

-- Proposition 3: There exists φ ∈ ℝ, such that the graph of the function f(x + φ) is centrally symmetric about the origin
def proposition_3 : Prop := ∃ φ : ℝ, ∀ x, f (x + φ) = -f (-x)

theorem correct_propositions :
  (proposition_2 ∧ proposition_3) := by
  sorry

end correct_propositions_l324_324287


namespace proper_divisors_condition_l324_324984

theorem proper_divisors_condition (N : ℕ) :
  ∀ x : ℕ, (x ∣ N ∧ x ≠ 1 ∧ x ≠ N) → 
  (∀ L : ℕ, (L ∣ N ∧ L ≠ 1 ∧ L ≠ N) → (L = x^3 + 3 ∨ L = x^3 - 3)) → 
  (N = 10 ∨ N = 22) :=
by
  sorry

end proper_divisors_condition_l324_324984


namespace pizza_toppings_l324_324632

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l324_324632


namespace tangent_line_eqn_for_all_a_for_f_l324_324431

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem tangent_line_eqn :
  let x := 1;
  let y := f x;
  let f' (x : ℝ) := (Real.exp x * x - (Real.exp x - 1)) / x^2;
  f' 1 = 1 ∧ f 1 = Real.exp 1 - 1 →
  x - y + Real.exp 1 - 2 = 0 := by
  sorry

theorem for_all_a_for_f :
  ∀ a > 0, ∀ x, 0 < |x| < Real.log (1 + a) → |f x - 1| < a := by
  sorry

end tangent_line_eqn_for_all_a_for_f_l324_324431


namespace solutions_of_quadratic_l324_324781

theorem solutions_of_quadratic (c : ℝ) (h : ∀ α β : ℝ, 
  (α^2 - 3*α + c = 0 ∧ β^2 - 3*β + c = 0) → 
  ( (-α)^2 + 3*(-α) - c = 0 ∨ (-β)^2 + 3*(-β) - c = 0 ) ) :
  ∃ α β : ℝ, (α = 0 ∧ β = 3) ∨ (α = 3 ∧ β = 0) :=
by
  sorry

end solutions_of_quadratic_l324_324781


namespace quadrilateral_AD_eq_CD_l324_324281

theorem quadrilateral_AD_eq_CD (A B C D : Point)
  (h1 : ∠ CBD = 2 * ∠ ADB)
  (h2 : ∠ ABD = 2 * ∠ CDB)
  (h3 : dist A B = dist C B) :
  dist A D = dist C D := 
sorry

end quadrilateral_AD_eq_CD_l324_324281


namespace prime_divisor_of_digits_in_set_l324_324404

theorem prime_divisor_of_digits_in_set {B : ℕ} (hB : 10 < B) (hDigits : ∀ d, d ∈ nat.digits 10 B → d ∈ {1, 3, 7, 9}) :
  ∃ p, nat.prime p ∧ p ∣ B ∧ 11 ≤ p :=
begin
  sorry
end

end prime_divisor_of_digits_in_set_l324_324404


namespace mentorship_arrangement_is_90_l324_324558

def mentorship_arrangement_ways : Nat :=
  let total_students := 5
  let teachers := 3
  -- Calculation of groups arrangement
  let ways_to_group := (Nat.choose total_students 2 * Nat.choose (total_students - 2) 2 * Nat.choose (total_students - 4) 1) / Nat.factorial 2
  -- Ways to assign the groups to the teachers
  let ways_to_assign := ways_to_group * Nat.factorial teachers
  ways_to_assign

theorem mentorship_arrangement_is_90 : mentorship_arrangement_ways = 90 := by
  sorry

end mentorship_arrangement_is_90_l324_324558


namespace chalkboard_area_l324_324849

def width : Float := 3.5
def length : Float := 2.3 * width
def area : Float := length * width

theorem chalkboard_area : area = 28.175 :=
by 
  sorry

end chalkboard_area_l324_324849


namespace circle_ratio_l324_324974

theorem circle_ratio (s : ℝ) (A X : ℝ) :
  let r := s * Real.sqrt 3 / 6 in
  let r1 := r / 3 in
  let area := (π * r^2) / 12 in
  let total_other_areas := area / 8 in
  A = π * r^2 →
  X = π * r1^2 / (1 - 1 / 9)  →
  A / X = 8 / 3 :=
by
  intros hA hX
  calc
    A / X = sorry

end circle_ratio_l324_324974


namespace largest_angle_of_triangle_l324_324895

theorem largest_angle_of_triangle
  (a b y : ℝ)
  (h1 : a = 60)
  (h2 : b = 70)
  (h3 : a + b + y = 180) :
  max a (max b y) = b :=
by
  sorry

end largest_angle_of_triangle_l324_324895


namespace cot_alpha_value_l324_324314

variable (α : ℝ)

noncomputable def sin := Real.sin
noncomputable def cos := Real.cos
noncomputable def cot (x : ℝ) := cos x / sin x

theorem cot_alpha_value :
  sin (2 * α) = -sin α ∧ α ∈ Ioo (π / 2) π → cot α = -√3 / 3 :=
by
  sorry

end cot_alpha_value_l324_324314


namespace necessary_and_sufficient_condition_l324_324382

noncomputable def shaded_areas_equal (θ : ℝ) (r : ℝ) : Prop :=
  (0 < θ ∧ θ < π / 2) →
  let left_shaded_area := (r^2 * tan θ / 2) - (θ * r^2 / 2)
  let right_shaded_area := θ * r^2 / 2
  left_shaded_area = right_shaded_area ↔ tan θ = 2 * θ

-- Statement without proof:
theorem necessary_and_sufficient_condition (θ : ℝ) (r : ℝ) : shaded_areas_equal θ r := sorry

end necessary_and_sufficient_condition_l324_324382


namespace pizza_problem_l324_324596

noncomputable def pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) : ℕ := 
  let slices_with_both := total_slices - (pepperoni_slices + mushroom_slices - total_slices)
  slices_with_both

theorem pizza_problem 
  (total_slices pepperoni_slices mushroom_slices : ℕ)
  (h_total: total_slices = 16)
  (h_pepperoni: pepperoni_slices = 8)
  (h_mushrooms: mushroom_slices = 12)
  (h_at_least_one: pepperoni_slices + mushroom_slices - total_slices ≥ 0)
  (h_no_three_toppings: total_slices = pepperoni_slices + mushroom_slices - 
   (total_slices - (pepperoni_slices + mushroom_slices - total_slices))) : 
  pizza_slices total_slices pepperoni_slices mushroom_slices = 4 :=
by 
  rw [h_total, h_pepperoni, h_mushrooms]
  sorry

end pizza_problem_l324_324596


namespace jellybeans_left_l324_324161

theorem jellybeans_left :
  let initial_jellybeans := 500
  let total_kindergarten := 10
  let total_firstgrade := 10
  let total_secondgrade := 10
  let sick_kindergarten := 2
  let sick_secondgrade := 3
  let jellybeans_sick_kindergarten := 5
  let jellybeans_sick_secondgrade := 10
  let jellybeans_remaining_kindergarten := 3
  let jellybeans_firstgrade := 5
  let jellybeans_secondgrade_per_firstgrade := 5 / 2 * total_firstgrade
  let consumed_by_sick := sick_kindergarten * jellybeans_sick_kindergarten + sick_secondgrade * jellybeans_sick_secondgrade
  let remaining_kindergarten := total_kindergarten - sick_kindergarten
  let consumed_by_remaining := remaining_kindergarten * jellybeans_remaining_kindergarten + total_firstgrade * jellybeans_firstgrade + total_secondgrade * jellybeans_secondgrade_per_firstgrade
  let total_consumed := consumed_by_sick + consumed_by_remaining
  initial_jellybeans - total_consumed = 176 := by 
  sorry

end jellybeans_left_l324_324161


namespace ellipse_minor_axis_length_l324_324732

noncomputable def minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ) :=
  if (a > b ∧ b > 0 ∧ eccentricity = (Real.sqrt 5) / 3 ∧ sum_distances = 12) then
    2 * b
  else
    0

theorem ellipse_minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : eccentricity = (Real.sqrt 5) / 3) (h4 : sum_distances = 12) :
  minor_axis_length a b eccentricity sum_distances = 8 :=
sorry

end ellipse_minor_axis_length_l324_324732


namespace average_reciprocal_sequence_l324_324256

theorem average_reciprocal_sequence (a : ℕ → ℝ) (n : ℕ) (h : ∀ n, n > 0 → (n / finset.sum (finset.range n.succ) a) = (1 / (3 * n + 2))) : ∀ n, a (n + 1) = 6 * (n + 1) - 1 :=
by
  sorry

end average_reciprocal_sequence_l324_324256


namespace num_perfect_squares_and_cubes_l324_324776

theorem num_perfect_squares_and_cubes (n : ℕ) (h : n < 555) : 
  (count (λ x, x < n ∧ (∃ k, x = k^2) ∨ (∃ k, x = k^3))) = 29 :=
sorry

end num_perfect_squares_and_cubes_l324_324776


namespace find_m_ge_T_m_l324_324406

noncomputable def T (n : ℕ) : ℕ :=
  if h : ∃ k, n ∣ k * (k + 1) / 2 then Nat.find h else 0

theorem find_m_ge_T_m :
  ∀ (m : ℕ),
    (m ≤ 1 ∨ ∃ k n, k ≥ 1 ∧ n ≥ 3 ∧ n % 2 = 1 ∧ m = 2^k * n) →
    m ≥ T m :=
by
  intro m h
  sorry

end find_m_ge_T_m_l324_324406


namespace find_total_income_l324_324584

theorem find_total_income (I : ℝ)
  (h1 : 0.6 * I + 0.3 * I + 0.005 * (I - (0.6 * I + 0.3 * I)) + 50000 = I) : 
  I = 526315.79 :=
by
  sorry

end find_total_income_l324_324584


namespace total_bags_sold_l324_324977

theorem total_bags_sold (first_week second_week third_week fourth_week total : ℕ) 
  (h1 : first_week = 15) 
  (h2 : second_week = 3 * first_week) 
  (h3 : third_week = 20) 
  (h4 : fourth_week = 20) 
  (h5 : total = first_week + second_week + third_week + fourth_week) : 
  total = 100 := 
sorry

end total_bags_sold_l324_324977


namespace sum_ineq_l324_324094

theorem sum_ineq 
  (n : ℕ) 
  (a b : fin n → ℝ) 
  (h_ab : ∀ i, 1 ≤ a i ∧ a i ≤ 2 ∧ 1 ≤ b i ∧ b i ≤ 2) 
  (h_sum_sq : ∑ i, (a i)^2 = ∑ i, (b i)^2) 
: 
  (∑ i, (a i)^3 / (b i)) ≤ (17 / 10) * ∑ i, (a i)^2 := 
sorry

end sum_ineq_l324_324094


namespace hexagons_to_square_l324_324880

-- Definitions of conditions
def rectangle_length := 18
def rectangle_width := 8
def rectangle_area := rectangle_length * rectangle_width

-- Definition related to hexagons forming a square
def hexagons_form_square : Prop :=
  ∃ s : ℕ, (s * s = rectangle_area) ∧ (y = s / 2)

-- Statement of the problem
theorem hexagons_to_square : hexagons_form_square := by
  sorry

end hexagons_to_square_l324_324880


namespace boat_speed_in_still_water_l324_324539

/--
The speed of the stream is 6 kmph.
The boat can cover 48 km downstream or 32 km upstream in the same time.
We want to prove that the speed of the boat in still water is 30 kmph.
-/
theorem boat_speed_in_still_water (x : ℝ)
  (h1 : ∃ t : ℝ, t = 48 / (x + 6) ∧ t = 32 / (x - 6)) : x = 30 :=
by
  sorry

end boat_speed_in_still_water_l324_324539


namespace at_least_one_equals_a_l324_324116

theorem at_least_one_equals_a (x y z a : ℝ) (hx_ne_0 : x ≠ 0) (hy_ne_0 : y ≠ 0) (hz_ne_0 : z ≠ 0) (ha_ne_0 : a ≠ 0)
  (h1 : x + y + z = a) (h2 : 1/x + 1/y + 1/z = 1/a) : x = a ∨ y = a ∨ z = a :=
  sorry

end at_least_one_equals_a_l324_324116


namespace original_price_l324_324983

theorem original_price (x : ℝ) (h1 : x > 0) (h2 : 1.12 * x - x = 270) : x = 2250 :=
by
  sorry

end original_price_l324_324983


namespace appropriate_sampling_method_l324_324640

theorem appropriate_sampling_method
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (survey_size : ℕ)
  (diff_interests : Prop)
  (h1 : total_students = 1000)
  (h2 : male_students = 500)
  (h3 : female_students = 500)
  (h4 : survey_size = 100)
  (h5 : diff_interests) : 
  sampling_method = "stratified sampling" :=
by
  sorry

end appropriate_sampling_method_l324_324640


namespace sum_abs_eq_sixty_six_l324_324717

-- Given sum of the first n terms
def S (n : ℕ) : ℤ :=
  n^2 - 4*n + 2

-- Define the sequence {a_n} such that a_1 = S_1 and for n >= 2, a_n = S_n - S_(n-1)
def a (n : ℕ) : ℤ :=
  if h : n = 1 then S 1
  else S n - S (n - 1)

-- Statement to prove: |a_1| + |a_2| + ⋯ + |a_{10}| = 66
theorem sum_abs_eq_sixty_six : (Finset.range 10).sum (λ n, Int.natAbs (a (n+1))) = 66 := by
  sorry

end sum_abs_eq_sixty_six_l324_324717


namespace shaded_area_approx_21_l324_324211

def rectangle_area (length width : ℝ) : ℝ := length * width

def circle_area (diameter : ℝ) : ℝ :=
  let r := diameter / 2
  in π * r ^ 2

def shaded_area (length width diameter : ℝ) : ℝ :=
  rectangle_area length width - circle_area diameter
  
theorem shaded_area_approx_21 :
  shaded_area 4 6 2 ≈ 21 :=
by 
  sorry

end shaded_area_approx_21_l324_324211


namespace statement_c_not_always_true_l324_324764

noncomputable def vector_op (a b : ℝ × ℝ) : ℝ :=
  let θ := real.angle.to_real (real.angle.of_vector a - real.angle.of_vector b)
  in nat.abs (a.fst * b.snd - b.fst * a.snd)

theorem statement_c_not_always_true (a b c : ℝ × ℝ) : ¬ ∀ (a b c : ℝ × ℝ),
  vector_op (a + b) c = vector_op a c + vector_op b c :=
by
  sorry

end statement_c_not_always_true_l324_324764


namespace marcy_pets_cat_time_l324_324437

theorem marcy_pets_cat_time (P : ℝ) (h1 : P + (1/3)*P = 16) : P = 12 :=
by
  sorry

end marcy_pets_cat_time_l324_324437


namespace lionsAfterOneYear_l324_324546

-- Definitions based on problem conditions
def initialLions : Nat := 100
def birthRate : Nat := 5
def deathRate : Nat := 1
def monthsInYear : Nat := 12

-- Theorem statement
theorem lionsAfterOneYear :
  initialLions + birthRate * monthsInYear - deathRate * monthsInYear = 148 :=
by
  sorry

end lionsAfterOneYear_l324_324546


namespace dice_sum_to_10_probability_l324_324556

theorem dice_sum_to_10_probability : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 10) →
  (calculate_probability (3, 6, 10) = 1 / 9) :=
begin
  sorry
end

-- Defining calculate_probability (used in the theorem) to represent the probability calculation
def calculate_probability (n_dice : ℕ, n_faces : ℕ, target_sum : ℕ) : ℚ :=
  let outcomes := (finset.powerset (finset.range (n_faces*2)).map(λ x, (x.val + 1)).filter(x.val <= n_faces)) in
  let valid_outcomes := outcomes.filter(λ l, l.sum = target_sum) in
  valid_outcomes.card / outcomes.card.to_rat

end dice_sum_to_10_probability_l324_324556


namespace base_of_second_term_l324_324588

theorem base_of_second_term (h : ℕ) (a b c : ℕ) (H1 : h > 0) 
  (H2 : 225 ∣ h) (H3 : 216 ∣ h) 
  (H4 : h = (2^a) * (some_number^b) * (5^c)) 
  (H5 : a + b + c = 8) : some_number = 3 :=
by
  sorry

end base_of_second_term_l324_324588


namespace smallest_b_for_hexagon_l324_324121

-- Define the regular hexagon with side length of 2
structure Hexagon :=
(side_length : ℝ := 2)

-- Define a predicate that checks if a set of points is within the hexagon
def inside_or_on_hexagon (hex : Hexagon) (points : set (ℝ × ℝ)) : Prop :=
  -- Placeholder for actual implementation
  sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the condition for the problem
def six_points_in_hexagon (hex : Hexagon) (points : set (ℝ × ℝ)) : Prop :=
  points.finite ∧ points.card = 6 ∧ inside_or_on_hexagon hex points

-- Define the smallest possible distance b
def smallest_distance_b (b : ℝ) (hex : Hexagon) : Prop :=
  ∀ points : set (ℝ × ℝ), six_points_in_hexagon hex points → 
  ∃ p1 p2 ∈ points, p1 ≠ p2 ∧ distance p1 p2 ≤ b

-- The statement of the theorem
theorem smallest_b_for_hexagon : smallest_distance_b (2 / real.sqrt 3) ⟨2⟩ :=
by
  sorry

end smallest_b_for_hexagon_l324_324121


namespace power_product_eq_nine_l324_324247

theorem power_product_eq_nine (x : ℝ) (hx : x = 81) : (x^0.25) * (x^0.20) = 9 := by 
  sorry

end power_product_eq_nine_l324_324247


namespace linda_savings_l324_324434

theorem linda_savings (S : ℝ) (h : (1 / 2) * S = 300) : S = 600 :=
sorry

end linda_savings_l324_324434


namespace tangent_line_eq_l324_324888

noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Definition of the tangent line to the curve f at point A(0, f(0))
theorem tangent_line_eq {f : ℝ → ℝ} (h : f = λ x, Real.exp x) :
  ∃ k b, (∀ x, y = k * x + b ↔ x - y + 1 = 0) :=
begin
  sorry
end

end tangent_line_eq_l324_324888


namespace transformation_correct_l324_324722

theorem transformation_correct (a b : ℝ) (h₁ : 3 * a = 2 * b) (h₂ : a ≠ 0) (h₃ : b ≠ 0) :
  a / 2 = b / 3 :=
sorry

end transformation_correct_l324_324722


namespace simplify_sqrt_l324_324119

theorem simplify_sqrt (sin cos : ℝ → ℝ) : 
  sqrt (1 - 2 * sin 4 * cos 4) = cos 4 - sin 4 := 
by sorry

end simplify_sqrt_l324_324119


namespace smallest_positive_n_l324_324179

theorem smallest_positive_n (n : ℕ) : n > 0 → (3 * n ≡ 1367 [MOD 26]) → n = 5 :=
by
  intros _ _
  sorry

end smallest_positive_n_l324_324179


namespace part_a_part_b_part_c_l324_324076

-- Definitions
def lamp_step (n : ℕ) (state : ℕ → bool) (j : ℕ) : (ℕ → bool) :=
  λ i, if i = j then ¬ state j else state i

def all_on (n : ℕ) (state : ℕ → bool) : Prop :=
  ∀ i : ℕ, i < n → state i = true

-- Statements

-- Part (a)
theorem part_a (n : ℕ) (h : n > 1) :
  ∃ M : ℕ, ∀ state : ℕ → bool, all_on n state →
  all_on n (nat.iterate (lamp_step n) M state) := 
sorry

-- Part (b)
theorem part_b (k : ℕ) (n := 2 ^ k) (h : 2 ^ k > 1) :
  ∀ state : ℕ → bool, all_on n state →
  all_on n (nat.iterate (lamp_step n) (n^2 - 1) state) :=
sorry

-- Part (c)
theorem part_c (k : ℕ) (n := 2 ^ k + 1) (h : 2 ^ k + 1 > 1) :
  ∀ state : ℕ → bool, all_on n state →
  all_on n (nat.iterate (lamp_step n) (n^2 - n + 1) state) :=
sorry

end part_a_part_b_part_c_l324_324076


namespace find_x2_y2_l324_324608

variable (x y : ℝ)

-- Given conditions
def average_commute_time (x y : ℝ) := (x + y + 10 + 11 + 9) / 5 = 10
def variance_commute_time (x y : ℝ) := ( (x - 10) ^ 2 + (y - 10) ^ 2 + (10 - 10) ^ 2 + (11 - 10) ^ 2 + (9 - 10) ^ 2 ) / 5 = 2

-- The theorem to prove
theorem find_x2_y2 (hx_avg : average_commute_time x y) (hx_var : variance_commute_time x y) : 
  x^2 + y^2 = 208 :=
sorry

end find_x2_y2_l324_324608


namespace total_distinct_values_S_l324_324090

noncomputable def z := Complex.exp (Complex.pi * Complex.I / 3)
def S (n : Int) := z^n + z^(-n)

theorem total_distinct_values_S : 
  Set.card (Set.ofFunction S (Finset.range 6)) = 4 :=
by
  sorry

end total_distinct_values_S_l324_324090


namespace measure_of_angle_DSO_l324_324067

variables (D O G S : Type) [Inhabited D] [Inhabited O] [Inhabited G] [Inhabited S]
variables (∠DOG ∠DGO ∠OGD ∠DOS ∠DSO : ℝ)

-- Given conditions
axiom angle_equality1 : ∠DGO = 40
axiom angle_equality2 : ∠DOG = 40
axiom angle_bisector : ∠DOS = (1 / 2) * ∠DOG

-- Defining the measures of the angles using the sum of angles in a triangle
def angle_sum_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Theorem statement
theorem measure_of_angle_DSO : ∠DSO = 60 :=
by
  have angle_equality3 : ∠OGD = 180 - 2 * ∠DOG := sorry
  have angle_dos_value : ∠DOS = 20 := sorry
  have angle_sum_for_DOS : angle_sum_triangle ∠DOS ∠OGD ∠DSO := sorry
  show ∠DSO = 60 from sorry

end measure_of_angle_DSO_l324_324067


namespace odd_function_expression_l324_324318

noncomputable def f : ℝ → ℝ
| x := if x ≥ 0 then x^2 - 2 * x else x^2 + 2 * x

theorem odd_function_expression (x : ℝ) :
  f(x) = x * (|x| - 2) :=
by 
  sorry -- Proof is omitted

end odd_function_expression_l324_324318


namespace sequence_a2002_l324_324759

theorem sequence_a2002 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (a 2 = 2) → 
  (∀ n, 2 ≤ n → a (n + 1) = 3 * a n - 2 * a (n - 1)) → 
  a 2002 = 2 ^ 2001 :=
by
  intros a ha1 ha2 hrecur
  sorry

end sequence_a2002_l324_324759


namespace values_of_r_l324_324306

theorem values_of_r (a : Finₓ n → ℝ) (r : Finₓ n → ℝ) 
  (h : ∀ (x : Finₓ n → ℝ), (∑ i, r i * (x i - a i)) ≤ (Real.sqrt (∑ i, (x i)^2) - Real.sqrt (∑ i, (a i)^2))) :
  ∀ k, r k = a k / Real.sqrt (∑ i, (a i)^2) :=
by
  sorry

end values_of_r_l324_324306


namespace pet_store_animals_l324_324153

theorem pet_store_animals (cats dogs birds : ℕ) 
    (ratio_cats_dogs_birds : 2 * birds = 4 * cats ∧ 3 * cats = 2 * dogs) 
    (num_cats : cats = 20) : dogs = 30 ∧ birds = 40 :=
by 
  -- This is where the proof would go, but we can skip it for this problem statement.
  sorry

end pet_store_animals_l324_324153


namespace num_polynomials_is_27_l324_324342

def polynomial_has_form (n : ℕ) (coeffs : Fin (n + 1) → ℝ) : Prop :=
  n + (Finset.univ.sum fun i => abs (coeffs i)) = 5

def num_polynomials_matching_criteria : ℕ :=
  Nat.card { p : Σ n, Fin (n + 1) → ℝ // polynomial_has_form p.1 p.2 }

theorem num_polynomials_is_27 : num_polynomials_matching_criteria = 27 := by
  sorry

end num_polynomials_is_27_l324_324342


namespace find_ellipse_equation_verify_relationship_l324_324060

-- Definitions and conditions
def ellipse (a b : ℝ) := λ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1
def circle (r : ℝ) := λ (x y : ℝ), x^2 + y^2 = r^2
def tangent_line (k m : ℝ) := λ (x y : ℝ), y = k * x + m

-- Conditions
variables (a b r : ℝ) (ha : a > 0) (hb : b > 0) (hr : r > 0) (hb_r : b > r)  

-- Question (I) and its answer
theorem find_ellipse_equation (k : ℝ) (r : ℝ) (hk : k = -1/2) (hr : r = 1) (x y : ℝ) :
  (circle r x y → tangent_line k (sqrt 5 / 2) x y → x = 0 → y = sqrt 5 / 2 → a = sqrt 5 → b = sqrt 5 / 2) →
  ellipse a b x y = (x^2 / 5 + 4 * y^2 / 5 = 1) := sorry

-- Question (II) and its verification
theorem verify_relationship (a b r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0) (h : 0 < r < b) 
  (h_circle : ∀ (x y : ℝ), circle r x y) 
  (h_origin : ∀ (x_A y_A x_B y_B : ℝ), (tangent_line k m x_A y_A) ∧ (tangent_line k m x_B y_B) → (x_A = 0) ∧ (y_A = sqrt 5 / 2) ∧ (x_B = sqrt 5) ∧ (y_B = 0)) :
  1 / a^2 + 1 / b^2 = 1 / r^2 := sorry

end find_ellipse_equation_verify_relationship_l324_324060


namespace positive_integers_condition_1000000_l324_324716

def r (a N : ℕ) : ℕ := a % N

theorem positive_integers_condition_1000000 :
  (∑ n in Finset.range 1000001, if r n 1000 > r n 1001 then 1 else 0) = 499501 :=
by
  sorry

end positive_integers_condition_1000000_l324_324716


namespace find_k_l324_324419

theorem find_k (a b c k : ℤ)
  (g : ℤ → ℤ)
  (h1 : ∀ x, g x = a * x^2 + b * x + c)
  (h2 : g 2 = 0)
  (h3 : 60 < g 6 ∧ g 6 < 70)
  (h4 : 90 < g 9 ∧ g 9 < 100)
  (h5 : 10000 * k < g 50 ∧ g 50 < 10000 * (k + 1)) :
  k = 0 :=
sorry

end find_k_l324_324419


namespace divide_plot_l324_324634

theorem divide_plot (plot_length plot_width fence_count fence_length : ℕ)
  (h1 : plot_length = 80)
  (h2 : plot_width = 50)
  (h3 : fence_count = 5)
  (h4 : ∀ (a : ℕ), (plot_length * plot_width) / fence_count = a → a = 800)
  (h5 : fence_length = 40) :
  ∃ (fences : Finset (ℕ × ℕ)), 
    fences.card = fence_count ∧ 
    (∀ fence ∈ fences, fence.2 = fence_length) ∧ 
    (∀ partitions : Finset (ℕ × ℕ), partitions.card = fence_count → 
      (∃ (length width : ℕ), 
        (length * width = 800) ∧ 
        (length = fence.1 ∧ width = plot_width)) ∨
      ∃ (length width : ℕ), 
        (width * plot_width = 800) ∧ 
        (width ∈ partitions ∧ length = fence.1)) :=
by {
  sorry
}

end divide_plot_l324_324634


namespace students_taking_geometry_or_history_but_not_both_l324_324272

theorem students_taking_geometry_or_history_but_not_both (s1 s2 s3 : ℕ) :
  (s1 = 15) → (s2 = 30) → (s3 = 18) → s2 - s1 + s3 = 33 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  exact sorry,
end

end students_taking_geometry_or_history_but_not_both_l324_324272


namespace quadratic_equal_roots_iff_a_eq_4_l324_324352

theorem quadratic_equal_roots_iff_a_eq_4 (a : ℝ) (h : ∃ x : ℝ, (a * x^2 - 4 * x + 1 = 0) ∧ (a * x^2 - 4 * x + 1 = 0)) :
  a = 4 :=
by
  sorry

end quadratic_equal_roots_iff_a_eq_4_l324_324352


namespace number_of_white_dogs_l324_324913

noncomputable def number_of_brown_dogs : ℕ := 20
noncomputable def number_of_black_dogs : ℕ := 15
noncomputable def total_number_of_dogs : ℕ := 45

theorem number_of_white_dogs : total_number_of_dogs - (number_of_brown_dogs + number_of_black_dogs) = 10 := by
  sorry

end number_of_white_dogs_l324_324913


namespace find_a_l324_324421

theorem find_a (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_a_l324_324421


namespace dirichlet_properties_l324_324143

-- Define the Dirichlet function
def dirichlet (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

-- Prove that the number of true propositions is 3
theorem dirichlet_properties :
  (¬ (∀ x : ℝ, dirichlet (dirichlet x) = 0)) ∧
  (∀ x : ℝ, dirichlet (-x) = dirichlet x) ∧
  (∀ (T : ℝ), T ≠ 0 ∧ T ∈ ℚ → ∀ x : ℝ, dirichlet (x + T) = dirichlet x) ∧
  (∃ (x1 x2 x3 : ℝ), 
    let A := (x1, dirichlet x1),
        B := (x2, dirichlet x2),
        C := (x3, dirichlet x3) in
    (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ∧
    (dirichlet x1 = 0) ∧
    (dirichlet x2 = 1) ∧
    (dirichlet x3 = 0) ∧
    ((x2 - x1)^2 + (dirichlet x2 - dirichlet x1)^2 = (x3 - x2)^2 + (dirichlet x3 - dirichlet x2)^2 ∧
    (x3 - x1)^2 + (dirichlet x3 - dirichlet x1)^2 = (x3 - x2)^2 + (dirichlet x3 - dirichlet x2)^2))
  = 3 := sorry

end dirichlet_properties_l324_324143


namespace arithmetic_mean_of_sequence_60_eq_32_5_l324_324248

-- Define the arithmetic sequence
def sequence (n : ℕ) : ℕ := n + 2

-- Define the sum of first 60 terms of the sequence
def sum_sequence_60 : ℕ := 1950

-- Define the arithmetic mean calculation
def arithmetic_mean (sum : ℕ) (terms : ℕ) : ℚ := sum / terms

-- The main theorem stating the arithmetic mean is 32.5
theorem arithmetic_mean_of_sequence_60_eq_32_5 : 
  arithmetic_mean sum_sequence_60 60 = 32.5 := by
  sorry

end arithmetic_mean_of_sequence_60_eq_32_5_l324_324248


namespace BANANA_distinct_arrangements_l324_324777

theorem BANANA_distinct_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 1) * (Nat.factorial 3) * (Nat.factorial 2)) = 60 := 
by
  sorry

end BANANA_distinct_arrangements_l324_324777


namespace number_of_perfect_square_factors_l324_324020

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem number_of_perfect_square_factors :
  let n := (2^12) * (3^18) * (5^20) in
  let factors := ∏ i in [2, 3, 5], i, factors.count is_perfect_square = 770 :=
by sorry

end number_of_perfect_square_factors_l324_324020


namespace donuts_percentage_missing_l324_324021

noncomputable def missing_donuts_percentage (initial_donuts : ℕ) (remaining_donuts : ℕ) : ℝ :=
  ((initial_donuts - remaining_donuts : ℕ) : ℝ) / initial_donuts * 100

theorem donuts_percentage_missing
  (h_initial : ℕ := 30)
  (h_remaining : ℕ := 9) :
  missing_donuts_percentage h_initial h_remaining = 70 :=
by
  sorry

end donuts_percentage_missing_l324_324021


namespace percentage_of_students_wearing_red_shirts_l324_324367

/-- In a school of 700 students:
    - 45% of students wear blue shirts.
    - 15% of students wear green shirts.
    - 119 students wear shirts of other colors.
    We are proving that the percentage of students wearing red shirts is 23%. --/
theorem percentage_of_students_wearing_red_shirts:
  let total_students := 700
  let blue_shirt_percentage := 45 / 100
  let green_shirt_percentage := 15 / 100
  let other_colors_students := 119
  let students_with_blue_shirts := blue_shirt_percentage * total_students
  let students_with_green_shirts := green_shirt_percentage * total_students
  let students_with_other_colors := other_colors_students
  let students_with_blue_green_or_red_shirts := total_students - students_with_other_colors
  let students_with_red_shirts := students_with_blue_green_or_red_shirts - students_with_blue_shirts - students_with_green_shirts
  (students_with_red_shirts / total_students) * 100 = 23 := by
  sorry

end percentage_of_students_wearing_red_shirts_l324_324367


namespace vertical_angles_equal_l324_324534

theorem vertical_angles_equal {α β : Type} [LinearOrderedField α] [LinearOrder β] (a1 a2 : β) 
  (h : a1 = a2) : "If two angles are vertical angles, then these two angles are equal." :=
sorry

end vertical_angles_equal_l324_324534


namespace thomas_total_blocks_l324_324921

theorem thomas_total_blocks :
  let stack1 := 7 in
  let stack2 := stack1 + 3 in
  let stack3 := stack2 - 6 in
  let stack4 := stack3 + 10 in
  let stack5 := stack2 * 2 in
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 :=
by
  let stack1 := 7
  let stack2 := stack1 + 3
  let stack3 := stack2 - 6
  let stack4 := stack3 + 10
  let stack5 := stack2 * 2
  have : stack1 + stack2 + stack3 + stack4 + stack5 = 7 + 10 + 4 + 14 + 20 := by rfl
  rw [this]
  norm_num
  sorry

end thomas_total_blocks_l324_324921


namespace total_books_l324_324071

theorem total_books (joan_books : ℕ) (tom_books : ℕ) (h1 : joan_books = 10) (h2 : tom_books = 38) : joan_books + tom_books = 48 :=
by
  -- insert proof here
  sorry

end total_books_l324_324071


namespace range_of_f_l324_324909

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem range_of_f : set.range (λ (x : ℝ), f x) = set.Icc (-4) 0 := 
by sorry

end range_of_f_l324_324909


namespace perimeter_ratio_of_quadrilateral_cut_by_line_l324_324540

variable (b : ℝ)

def square_vertices := {(-2*b, -2*b), (2*b, -2*b), (-2*b, 2*b), (2*b, 2*b)}
def cutting_line (x : ℝ) := x / 3

theorem perimeter_ratio_of_quadrilateral_cut_by_line : 
  let intersection_points := {(2*b, 2*b/3), (-2*b, -2*b/3)}
  let side_length := 4 * b
  let vertical_height := 4 * b / 3
  let diagonal_length := (real.sqrt 160) * b / 3
  let perimeter := 2 * vertical_height + side_length + diagonal_length in
  (perimeter / b = (4 * (4 + real.sqrt 10) / 3)) :=
by
  sorry

end perimeter_ratio_of_quadrilateral_cut_by_line_l324_324540


namespace sum_of_possible_values_l324_324901

variable {M : ℝ}

theorem sum_of_possible_values (h : M * (M - 8) = 7) : M = 4 + 2 * real.sqrt 3 ∨ M = 4 - 2 * real.sqrt 3 := sorry

end sum_of_possible_values_l324_324901


namespace base_s_is_8_l324_324793

-- Define a noncomputable field (necessary for algebraic manipulations)
noncomputable theory

-- Here is the Lean statement
theorem base_s_is_8 (s : ℕ) (h₁ : 5 * s^2 + 3 * s + s^3 + 2 * s^2 + 3 * s = 2 * s^3) : s = 8 :=
by sorry

end base_s_is_8_l324_324793


namespace log_sum_squared_l324_324845

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_sum_squared :
  ∀ (a : ℝ) (x : ℕ → ℝ),
  (0 < a ∧ a ≠ 1) →
  log_base a (∏ i in Finset.range 2017, x i) = 8 →
  (∑ i in Finset.range 2017, log_base a (x i ^ 2)) = 16 :=
by
  sorry

end log_sum_squared_l324_324845


namespace pizza_combination_count_l324_324615

open_locale nat

-- Given condition: the number of different toppings
def num_toppings : ℕ := 8

-- Calculating the total number of one-topping, two-topping, and three-topping pizzas
theorem pizza_combination_count : ((nat.choose num_toppings 1) + (nat.choose num_toppings 2) + (nat.choose num_toppings 3)) = 92 :=
by
  sorry

end pizza_combination_count_l324_324615


namespace lines_perpendicular_l324_324138

noncomputable def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem lines_perpendicular {m : ℝ} :
  is_perpendicular (m + 2) (1 - m) (m - 1) (2 * m + 3) ↔ m = 1 :=
by
  sorry

end lines_perpendicular_l324_324138


namespace measure_8_liters_possible_l324_324002

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l324_324002


namespace probability_ln_ineq_l324_324928

noncomputable def is_in_interval (a : ℝ) : Prop := 0 < a ∧ a < 1

noncomputable def ln_ineq (a : ℝ) : Prop := log (3 * a - 1) < 0

theorem probability_ln_ineq :
  ∀ (μ : measure_theory.measure ℝ),
    (∀ (a : ℝ), 0 < a ∧ a < 1 → μ {a} = 1) →
    (μ (set.Ioc (1/3 : ℝ) (2/3)) = 1/3) →
      μ {a | 0 < a ∧ a < 1 ∧ log (3 * a - 1) < 0} = 1/3 :=
begin
  intros μ hμ1 hμ2,
  -- Proof steps would go here
  sorry

end probability_ln_ineq_l324_324928


namespace graph_passes_through_point_l324_324146

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 3

theorem graph_passes_through_point (a : ℝ) : f a 1 = 4 := by
  sorry

end graph_passes_through_point_l324_324146


namespace oak_grove_libraries_total_books_l324_324547

theorem oak_grove_libraries_total_books :
  let publicLibraryBooks := 1986
  let schoolLibrariesBooks := 5106
  let communityCollegeLibraryBooks := 3294.5
  let medicalLibraryBooks := 1342.25
  let lawLibraryBooks := 2785.75
  publicLibraryBooks + schoolLibrariesBooks + communityCollegeLibraryBooks + medicalLibraryBooks + lawLibraryBooks = 15514.5 :=
by
  sorry

end oak_grove_libraries_total_books_l324_324547


namespace can_only_arrange_for_n_1_and_2_l324_324929

theorem can_only_arrange_for_n_1_and_2 (n : ℕ) : 
  (n = 1 ∨ n = 2) ↔ ∃ L : list ℕ, 
    (L.perm [1, 1, 2, 2, ..., n, n] ∧ 
      ∀ k ≤ n, ∃ i j : ℕ, i < j ∧ list.nth L i = some k ∧ list.nth L j = some k ∧ j - i - 1 = k) :=
sorry

end can_only_arrange_for_n_1_and_2_l324_324929


namespace range_of_x_l324_324735

theorem range_of_x (x : ℝ) (p : log (x^2 - 2*x - 2) ≥ 0) (q_false : ¬ (0 < x ∧ x < 4)) : x ≥ 4 ∨ x ≤ -1 := sorry

end range_of_x_l324_324735


namespace find_f_2013_l324_324763

def f : ℕ → ℤ
def g : ℕ → ℤ

axiom cond1 : ∀ n, f (f n) + g n = 2 * n + 3
axiom cond2 : f 0 = 1
axiom cond3 : g 0 = 2
axiom cond4 : ∀ n > 0, f n = 2 * f (n - 1) - g (n - 1)
axiom cond5 : ∀ n > 0, g n = f n + g (n - 1)

theorem find_f_2013 : f 2013 = -4024 := sorry

end find_f_2013_l324_324763


namespace system_solution_l324_324583

theorem system_solution (x y : ℝ) 
  (h1 : 0 < x + y) 
  (h2 : x + y ≠ 1) 
  (h3 : 2 * x - y ≠ 0)
  (eq1 : (x + y) * 2^(y - 2 * x) = 6.25) 
  (eq2 : (x + y) * (1 / (2 * x - y)) = 5) :
x = 9 ∧ y = 16 := 
sorry

end system_solution_l324_324583


namespace cylinder_surface_area_l324_324979

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (r_radius : r = 4) : 
  2 * π * r * (r + h) = 128 * π :=
by
  -- providing the proof steps is beyond the scope of this task
  sorry

end cylinder_surface_area_l324_324979


namespace smallest_x_for_pow_l324_324178

theorem smallest_x_for_pow (x : ℕ) (h : 27 = 3 ^ 3) : x = 9 → 27 ^ x > 3 ^ 24 :=
by
  intro hx
  rw hx
  rw h
  sorry

end smallest_x_for_pow_l324_324178


namespace number_of_balanced_integers_l324_324240

def is_balanced (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (a + b) - (c + d) = 2

def balanced_integers (low high : ℕ) : ℕ :=
  (List.range' low (high - low + 1)).count is_balanced

theorem number_of_balanced_integers : balanced_integers 2000 9999 = 343 := 
  sorry

end number_of_balanced_integers_l324_324240


namespace mixed_numbers_sum_l324_324544

-- Declare the mixed numbers as fraction equivalents
def mixed1 : ℚ := 2 + 1/10
def mixed2 : ℚ := 3 + 11/100
def mixed3 : ℚ := 4 + 111/1000

-- Assert that the sum of mixed1, mixed2, and mixed3 is equal to 9.321
theorem mixed_numbers_sum : mixed1 + mixed2 + mixed3 = 9321 / 1000 := by
  sorry

end mixed_numbers_sum_l324_324544


namespace remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l324_324937

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def sum_first_seven_primes : ℕ := first_seven_primes.sum

def eighth_prime : ℕ := 19

theorem remainder_when_multiplied_by_three_and_divided_by_eighth_prime :
  ((sum_first_seven_primes * 3) % eighth_prime = 3) :=
by
  sorry

end remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l324_324937


namespace quadratic_at_most_two_roots_l324_324863

theorem quadratic_at_most_two_roots (a b c x1 x2 x3 : ℝ) (ha : a ≠ 0) 
(h1 : a * x1^2 + b * x1 + c = 0)
(h2 : a * x2^2 + b * x2 + c = 0)
(h3 : a * x3^2 + b * x3 + c = 0)
(h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : 
false :=
sorry

end quadratic_at_most_two_roots_l324_324863


namespace area_two_layers_l324_324923

variables (total_wallpaper_area : ℝ) (wall_area : ℝ) (three_layer_area : ℝ)

-- Conditions
def conditions := total_wallpaper_area = 300 ∧ wall_area = 180 ∧ three_layer_area = 40

-- Question and Proof
theorem area_two_layers (h : conditions) : 
  let overlapping_area := total_wallpaper_area - wall_area in
  let two_layer_area := overlapping_area - three_layer_area in
  two_layer_area = 80 :=
by
  cases h with hw1 hw_rest, 
  cases hw_rest with hw2 hw3,
  simp only [hw1, hw2, hw3, sub_eq_add_neg] at *,
  let overlapping_area := 300 - 180,
  let two_layer_area := overlapping_area - 40,
  exact eq.refl (80 : ℝ)

end area_two_layers_l324_324923


namespace area_of_triangle_ABC_l324_324789

-- Given triangle ABC with a right angle at C
variables {A B C H M : Type}
variables [right_triangle {A B C}] (right_angle_C : is_right_angle {∠ACB})
variables (CH : altitude {A B C})
variables (CM : median {A B C})
variables (bisect_right_angle : bisects {CM ∠ACB})
variables (bisect_right_angle : bisects {CH ∠ACB})
variables (area_CHM : K)
 
-- Prove the area of triangle ABC
theorem area_of_triangle_ABC (K : ℝ) :
  area {A B C} = 2 * sqrt 2 * K :=
sorry

end area_of_triangle_ABC_l324_324789


namespace hexagons_to_square_l324_324879

-- Definitions of conditions
def rectangle_length := 18
def rectangle_width := 8
def rectangle_area := rectangle_length * rectangle_width

-- Definition related to hexagons forming a square
def hexagons_form_square : Prop :=
  ∃ s : ℕ, (s * s = rectangle_area) ∧ (y = s / 2)

-- Statement of the problem
theorem hexagons_to_square : hexagons_form_square := by
  sorry

end hexagons_to_square_l324_324879


namespace total_cost_l324_324099

variable (puppy_cost : ℕ) (weeks : ℕ) (food_per_day : ℚ)
variable (bag_capacity : ℚ) (bag_cost : ℚ)
variable (leash_cost : ℕ) (collar_cost : ℕ) (bed_cost : ℕ)

theorem total_cost
  (h1 : puppy_cost = 150)
  (h2 : weeks = 6)
  (h3 : food_per_day = 1/3)
  (h4 : bag_capacity = 3.5)
  (h5 : bag_cost = 2)
  (h6 : leash_cost = 15)
  (h7 : collar_cost = 12)
  (h8 : bed_cost = 25) :
  puppy_cost + (weeks * 7 * food_per_day / bag_capacity * bag_cost).to_nat
  + leash_cost + collar_cost + bed_cost = 210 := by
  sorry

end total_cost_l324_324099


namespace count_obtainable_results_from_2016_operations_l324_324112

theorem count_obtainable_results_from_2016_operations :
  let digits := [20, 16, 201, 6]
  let operations := [(+), (-), (*)]
  let results := [36, 195, 207, 320]
  (results.filter (λ r, (∃ d1 d2 ∈ digits, ∃ op ∈ operations, op d1 d2 = r))).length = 4 :=
by 
  sorry

end count_obtainable_results_from_2016_operations_l324_324112


namespace measure_8_liters_with_two_buckets_l324_324000

def bucket_is_empty (B : ℕ) : Prop :=
  B = 0

def bucket_has_capacity (B : ℕ) (c : ℕ) : Prop :=
  B ≤ c

def fill_bucket (B : ℕ) (c : ℕ) : ℕ :=
  c

def empty_bucket (B : ℕ) : ℕ :=
  0

def pour_bucket (B1 B2 : ℕ) (c1 c2 : ℕ) : (ℕ × ℕ) :=
  if B1 + B2 <= c2 then (0, B1 + B2)
  else (B1 - (c2 - B2), c2)

theorem measure_8_liters_with_two_buckets (B10 B6 : ℕ) (c10 c6 : ℕ) :
  bucket_has_capacity B10 c10 ∧ bucket_has_capacity B6 c6 ∧
  c10 = 10 ∧ c6 = 6 →
  ∃ B10' B6', B10' = 8 ∧ B6' ≤ 6 :=
by
  intros h
  have h1 : ∃ B1, bucket_is_empty B1,
    from ⟨0, rfl⟩
  let B10 := fill_bucket 0 c10
  let ⟨B10, B6⟩ := pour_bucket B10 0 c10 c6
  let B6 := empty_bucket B6
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  let B10 := fill_bucket B10 c10
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  exact ⟨B10, B6, rfl, le_refl 6⟩

end measure_8_liters_with_two_buckets_l324_324000


namespace julia_tuesday_kids_l324_324399

-- Definitions based on the given conditions in the problem.
def monday_kids : ℕ := 15
def monday_tuesday_kids : ℕ := 33

-- The problem statement to prove the number of kids played with on Tuesday.
theorem julia_tuesday_kids :
  (∃ tuesday_kids : ℕ, tuesday_kids = monday_tuesday_kids - monday_kids) →
  18 = monday_tuesday_kids - monday_kids :=
by
  intro h
  sorry

end julia_tuesday_kids_l324_324399


namespace distinct_acute_angles_l324_324188

theorem distinct_acute_angles (A₁ A₂ A₃ B₁ B₂ B₃ : ℝ) 
  (h₁ : A₁ = 90) (h₂ : A₂ = 75) (h₃ : A₃ = 15)
  (h₄ : B₁ = 90) (h₅ : B₂ = 54) (h₆ : B₃ = 36) :
  ∃ n, n = 29 ∧ ∀ θ, (3 ≤ θ ∧ θ < 90 ∧ θ % 3 = 0) ↔ (θ ∈ {3, 6, 9, ..., 87}) :=
by
  sorry

end distinct_acute_angles_l324_324188


namespace average_income_correct_l324_324968

/-- Define the daily earnings, commissions, and expenses. -/
def day1_income : ℝ := 250
def day1_commission : ℝ := 0.10 * day1_income

def day2_income : ℝ := 400
def day2_expense : ℝ := 50

def day3_income : ℝ := 750
def day3_commission : ℝ := 0.15 * day3_income

def day4_income : ℝ := 400
def day4_expense : ℝ := 40

def day5_income : ℝ := 500
def day5_commission : ℝ := 0.20 * day5_income

/-- Define the net income for each day. -/
def day1_net_income : ℝ := day1_income - day1_commission
def day2_net_income : ℝ := day2_income - day2_expense
def day3_net_income : ℝ := day3_income - day3_commission
def day4_net_income : ℝ := day4_income - day4_expense
def day5_net_income : ℝ := day5_income - day5_commission

/-- Sum of net incomes for all 5 days. -/
def total_net_income : ℝ := day1_net_income + day2_net_income + day3_net_income + day4_net_income + day5_net_income

/-- Calculate the average daily income. -/
def average_daily_income : ℝ := total_net_income / 5

/-- The theorem we aim to prove. -/
theorem average_income_correct : average_daily_income = 394.50 := by 
  sorry

end average_income_correct_l324_324968


namespace parabola_translation_l324_324168

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

-- Define the transformation: translate one unit to the right
def translate_right (x : ℝ) : ℝ := initial_parabola (x - 1)

-- Define the transformation: move up three units
def move_up (y : ℝ) : ℝ := y + 3

-- Define the resulting equation after the transformations
def resulting_parabola (x : ℝ) : ℝ := move_up (translate_right x)

-- Define the target equation
def target_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Formalize the proof problem
theorem parabola_translation :
  ∀ x : ℝ, resulting_parabola x = target_parabola x :=
by
  -- Proof steps go here
  sorry

end parabola_translation_l324_324168


namespace functional_equation_l324_324073

noncomputable def noncomputable_function (f : ℝ⁺ → ℝ⁺) := 
  ∀ x1 x2 : ℝ⁺, f (real.sqrt (x1 * x2)) = real.sqrt (f x1 * f x2)

theorem functional_equation (f : ℝ⁺ → ℝ⁺) 
  (h : noncomputable_function f) : 
∀ n : ℕ, ∀ x1 x2 ... xn : ℝ⁺, f (real.pow (list.prod (x1::x2::...::xn::[])) (1 / n)) = real.pow (list.prod (list.map f (x1::x2::...::xn::[]))) (1 / n) :=
sorry

end functional_equation_l324_324073


namespace range_of_magnitudes_l324_324292

theorem range_of_magnitudes (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 5) :
  3 ≤ BC ∧ BC ≤ 13 :=
by
  sorry

end range_of_magnitudes_l324_324292


namespace negation_of_universal_proposition_l324_324758

theorem negation_of_universal_proposition :
  (∀ x : ℝ, x > 1 → log 2 x + 4 * log x 2 > 4) ↔ ¬(∃ x : ℝ, x > 1 ∧ log 2 x + 4 * log x 2 ≤ 4) := by
  sorry

end negation_of_universal_proposition_l324_324758


namespace percent_of_class_received_50_to_59_l324_324215

-- Define the frequencies for each score range
def freq_90_to_100 := 5
def freq_80_to_89 := 7
def freq_70_to_79 := 9
def freq_60_to_69 := 8
def freq_50_to_59 := 4
def freq_below_50 := 3

-- Define the total number of students
def total_students := freq_90_to_100 + freq_80_to_89 + freq_70_to_79 + freq_60_to_69 + freq_50_to_59 + freq_below_50

-- Define the frequency of students scoring in the 50%-59% range
def freq_50_to_59_ratio := (freq_50_to_59 : ℚ) / total_students

-- Define the percentage calculation
def percent_50_to_59 := freq_50_to_59_ratio * 100

theorem percent_of_class_received_50_to_59 :
  percent_50_to_59 = 100 / 9 := 
by {
  sorry
}

end percent_of_class_received_50_to_59_l324_324215


namespace rank_from_last_l324_324783

theorem rank_from_last (total_students : ℕ) (rank_from_top : ℕ) (rank_from_last : ℕ) : 
  total_students = 35 → 
  rank_from_top = 14 → 
  rank_from_last = (total_students - rank_from_top + 1) → 
  rank_from_last = 22 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end rank_from_last_l324_324783


namespace quadratic_solution_l324_324871

theorem quadratic_solution (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 → (x = 1 / 2 ∨ x = 1) :=
by sorry

end quadratic_solution_l324_324871


namespace exist_pairs_l324_324751

theorem exist_pairs (n : ℕ) (z : ℕ → ℂ) (h1 : (∑ i in finset.range n, z i) = 0) (h2 : ∀ i, i < n → |z i| < 1) :
  ∃ i j, i < j ∧ j < n ∧ |z i + z j| < 1 :=
by {
  sorry
}

end exist_pairs_l324_324751


namespace daughter_age_in_3_years_l324_324962

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end daughter_age_in_3_years_l324_324962


namespace S8_value_l324_324388

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 5 / 5 + S 11 / 11 = 12) (h2 : S 11 = S 8 + 1 / a 9 + 1 / a 10 + 1 / a 11) : S 8 = 48 :=
sorry

end S8_value_l324_324388


namespace max_distance_between_points_on_curve_l324_324374

theorem max_distance_between_points_on_curve (theta : ℝ) :
  ∃ P Q : ℝ × ℝ, 
    (P.exists (λ θ, P = (4 * real.cos θ, 4 * θ))) ∧ 
    (Q.exists (λ θ, Q = (4 * real.cos θ, 4 * θ))) ∧ 
    dist P Q = 4 :=
sorry

end max_distance_between_points_on_curve_l324_324374


namespace wrapping_paper_area_l324_324637

variable (w h : ℝ)

theorem wrapping_paper_area : ∃ A, A = 4 * (w + h) ^ 2 :=
by
  sorry

end wrapping_paper_area_l324_324637


namespace minimum_force_to_submerge_cube_l324_324571

noncomputable def volume_cm3_to_m3 (v_cm: ℝ) : ℝ := v_cm * 10 ^ (-6)
noncomputable def mass (density: ℝ) (volume: ℝ): ℝ := density * volume
noncomputable def gravitational_force (mass: ℝ) (g: ℝ): ℝ := mass * g
noncomputable def buoyant_force (density_water: ℝ) (volume: ℝ) (g: ℝ): ℝ := density_water * volume * g

theorem minimum_force_to_submerge_cube 
    (V_cm: ℝ) (ρ_cube: ℝ) (ρ_water: ℝ) (g: ℝ) 
    (hV: V_cm = 10) (hρ_cube: ρ_cube = 700) (hρ_water: ρ_water = 1000) (hg: g = 10) :
    (buoyant_force ρ_water (volume_cm3_to_m3 V_cm) g - gravitational_force (mass ρ_cube (volume_cm3_to_m3 V_cm)) g) = 0.03 :=
by
    sorry

end minimum_force_to_submerge_cube_l324_324571


namespace trainB_reaches_in_3_hours_l324_324926

variable (trainA_speed trainB_speed : ℕ) (x t : ℝ)

-- Given conditions
axiom h1 : trainA_speed = 70
axiom h2 : trainB_speed = 105
axiom h3 : ∀ x t, 70 * x + 70 * 9 = 105 * x + 105 * t

-- Prove that train B takes 3 hours to reach destination after meeting
theorem trainB_reaches_in_3_hours : t = 3 :=
by
  sorry

end trainB_reaches_in_3_hours_l324_324926


namespace problem_a_problem_b_problem_c_l324_324922

-- Define the conditions
def probability_of_winning : ℕ → ℚ := λ n, (1/2)^n
def probability_of_needing_fifth_game : ℚ := 3/4
def probability_of_C_winning : ℚ := 7/16

-- State the proof problem
theorem problem_a : probability_of_winning 4 = 1/16 :=
sorry

theorem problem_b : 1 - (4 * probability_of_winning 4) = probability_of_needing_fifth_game :=
sorry

theorem problem_c : 1 - 2 * (9/32) = probability_of_C_winning :=
sorry

end problem_a_problem_b_problem_c_l324_324922


namespace kite_area_is_40_l324_324291

open Real

-- Define the vertices of the kite
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (4, 10)
def C : ℝ × ℝ := (8, 6)
def D : ℝ × ℝ := (4, 0)

-- Noncomputable since we are dealing with real numbers and skipping the proof
noncomputable def vertical_diagonal_length (B D : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - D.1) ^ 2 + (B.2 - D.2) ^ 2)

noncomputable def horizontal_diagonal_length (A C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2)

-- The main theorem for the area
theorem kite_area_is_40 :
  let vertical_length := vertical_diagonal_length B D,
      horizontal_length := horizontal_diagonal_length A C,
      base := horizontal_length,
      height := vertical_length / 2,
      area_of_one_triangle := (1 / 2) * base * height,
      total_area := 2 * area_of_one_triangle
  in total_area = 40 := by
  sorry

end kite_area_is_40_l324_324291


namespace xy_system_solution_l324_324742

theorem xy_system_solution (x y : ℝ) (h₁ : x + 5 * y = 6) (h₂ : 3 * x - y = 2) : x + y = 2 := 
by 
  sorry

end xy_system_solution_l324_324742


namespace perimeter_of_square_garden_l324_324460

-- Define the conditions
def is_square (x : ℝ) : Prop := x = (sqrt x) ^ 2
def has_perimeter (p : ℝ) : Prop := ∃ s : ℝ, p = 4 * s
def has_area (q p : ℝ) : Prop := q = p + 21

-- Define the problem statement
theorem perimeter_of_square_garden (p q : ℝ) (h1 : is_square q) (h2 : has_perimeter p) (h3 : has_area q p) : 
  p = 28 :=
sorry

end perimeter_of_square_garden_l324_324460


namespace total_time_paint_room_l324_324695

def doug_rate : ℝ := 1/4
def dave_rate : ℝ := 1/6
def combined_rate : ℝ := doug_rate + dave_rate
def work_done : ℝ := 1
def break_time : ℝ := 0.5

theorem total_time_paint_room (t : ℝ) (h : (combined_rate * (t - break_time)) = work_done) : 
  t = 29 / 10 := 
sorry

end total_time_paint_room_l324_324695


namespace find_x_l324_324339

open Real

-- Definition of the vectors
def a : ℝ × ℝ := (3, sqrt 3)
def b (x : ℝ) : ℝ × ℝ := (0, x)

-- Definition of the dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Definition of the magnitude of a vector
def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 * u.1 + u.2 * u.2)

-- The main theorem that needs to be proved
theorem find_x (x : ℝ) (h : dot_product a (b x) = magnitude a) : x = 2 :=
by
  sorry

end find_x_l324_324339


namespace minimal_degree_eq_2_pow_n_minus_1_l324_324074

-- Define the function f
def f (x : ℝ) : ℝ := x - (1 / x)

-- Define the iterative process for f
def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  (nat.iterate (λ y, f y)) n x

-- Proposition to prove the minimal degree dn is 2^n - 1
theorem minimal_degree_eq_2_pow_n_minus_1 (n : ℕ) : 
  ∃ p q : polynomial ℝ, 
  (∀ x : ℝ, q ≠ 0 → f_iter n x = (p.eval x) / (q.eval x)) ∧ 
  (q.degree = ↑(2^n - 1)) :=
sorry

end minimal_degree_eq_2_pow_n_minus_1_l324_324074


namespace find_f_3_l324_324504

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l324_324504


namespace projectile_reaches_35m_first_at_1_571_l324_324471

theorem projectile_reaches_35m_first_at_1_571 :
  ∃ t : ℝ, -4.9 * t^2 + 30 * t = 35 ∧ t ≈ 1.571 := by
  sorry

end projectile_reaches_35m_first_at_1_571_l324_324471


namespace value_of_f_at_6_over_7_l324_324876

variable {f : ℝ → ℝ}

theorem value_of_f_at_6_over_7
  (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(1 - x) = 1 - f(x))
  (h2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(1/3 * x) = 1/2 * f(x))
  (h3 : ∀ a b, 0 ≤ a ∧ a ≤ b ∧ b ≤ 1 → f(a) ≤ f(b)) :
  f (6/7) = 3/4 :=
sorry

end value_of_f_at_6_over_7_l324_324876


namespace three_distinct_real_roots_l324_324141

noncomputable def f : ℝ → ℝ := λ x, x^3 - 6 * x^2 + 9 * x + m

theorem three_distinct_real_roots (m : ℝ) :
  -4 < m ∧ m < 0 ↔ ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 :=
by
  sorry

end three_distinct_real_roots_l324_324141


namespace log_sum_l324_324676

theorem log_sum : Real.logb 2 1 + Real.logb 3 9 = 2 := by
  sorry

end log_sum_l324_324676


namespace find_f_of_3_l324_324480

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l324_324480


namespace ratio_above_8_to_8_l324_324794

variable (total_students : ℕ)
variable (students_below_8_percent : ℝ)
variable (students_of_8_years : ℕ)

def students_below_8 (total_students : ℕ) (students_below_8_percent : ℝ) : ℕ :=
  (students_below_8_percent * total_students).toNat

def students_above_8 (total_students : ℕ) (students_below_8 : ℕ) (students_of_8_years : ℕ) : ℕ :=
  total_students - (students_below_8 + students_of_8_years)

theorem ratio_above_8_to_8 (h1 : total_students = 100) 
                           (h2 : students_below_8_percent = 0.20)
                           (h3 : students_of_8_years = 48)
                           (h4 : students_below_8 total_students students_below_8_percent = 20) :
  (students_above_8 total_students (students_below_8 total_students students_below_8_percent) students_of_8_years).toRat /
  students_of_8_years.toRat = 2 / 3 :=
by
  sorry

end ratio_above_8_to_8_l324_324794
