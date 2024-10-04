import Mathlib

namespace inverse_four_times_l764_764845

def g : ℕ → ℕ
| 1 := 4
| 2 := 2
| 3 := 6
| 4 := 5
| 5 := 3
| _ := 0 -- defining g(x) for values outside 1 to 5, though not necessarily

noncomputable def g_inv (y : ℕ) : ℕ :=
if h : y ∈ set.range g then classical.some (classical.some_spec h) else 0

theorem inverse_four_times :
  g_inv (g_inv (g_inv (g_inv 2))) = 2 := by
sorry

end inverse_four_times_l764_764845


namespace ratio_of_areas_l764_764748

theorem ratio_of_areas (A B C D : Type) [ordered_semiring ℝ]
  (BD : ℝ) (DC : ℝ) 
  (hBD : BD = 8) (hDC : DC = 12) :
  (BD / DC) = (2 / 3) := 
by
  sorry

end ratio_of_areas_l764_764748


namespace number_of_pigs_l764_764892

variable (cows pigs : Nat)

theorem number_of_pigs (h1 : 2 * (7 + pigs) = 32) : pigs = 9 := by
  sorry

end number_of_pigs_l764_764892


namespace probability_problem_l764_764433

noncomputable def tamika_results : Finset ℕ :=
  {17, 18, 19, 20, 21}

noncomputable def carlos_results : Finset ℕ :=
  {13, 16, 19, 28, 33, 40}

noncomputable def probability_tamika_greater : ℚ := 13 / 18

theorem probability_problem :
  (λ tamika_results carlos_results, 
    ((tamika_results.product carlos_results).filter (λ p, p.1 > p.2)).card
    = 26
  ) → 
  (13 : ℚ) / 18 = probability_tamika_greater := sorry

end probability_problem_l764_764433


namespace students_spring_outing_l764_764147

theorem students_spring_outing (n : ℕ) (h1 : n = 5) : 2^n = 32 :=
  by {
    sorry
  }

end students_spring_outing_l764_764147


namespace range_of_a_l764_764693

def f (x : ℝ) (a : ℝ) : ℝ := 
  if x < 0 then x^2 + x + a
  else -1 / x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  if x < 0 then 2 * x + 1
  else 1 / x^2

theorem range_of_a (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ f' x₁ a = f' x₂ a) → (-2 < a ∧ a < 1 / 4) :=
by
  sorry

end range_of_a_l764_764693


namespace min_third_side_length_l764_764350

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end min_third_side_length_l764_764350


namespace coffee_shop_total_revenue_l764_764201

def regular_coffee_price := 5
def large_coffee_price := 6
def regular_tea_price := 4
def large_tea_price := 5

def tax_rate_coffee := 0.08
def tax_rate_tea := 0.06

def regular_coffee_customers := 7
def large_coffee_customers := 3
def regular_tea_customers := 4
def large_tea_customers := 4

noncomputable def total_revenue_with_tax : ℝ := 
  let revenue_regular_coffee := regular_coffee_customers * regular_coffee_price
  let revenue_large_coffee := large_coffee_customers * large_coffee_price
  let revenue_regular_tea := regular_tea_customers * regular_tea_price
  let revenue_large_tea := large_tea_customers * large_tea_price

  let tax_regular_coffee := revenue_regular_coffee * tax_rate_coffee
  let tax_large_coffee := revenue_large_coffee * tax_rate_coffee
  let tax_regular_tea := revenue_regular_tea * tax_rate_tea
  let tax_large_tea := revenue_large_tea * tax_rate_tea

  let total_revenue_regular_coffee := revenue_regular_coffee + tax_regular_coffee
  let total_revenue_large_coffee := revenue_large_coffee + tax_large_coffee
  let total_revenue_regular_tea := revenue_regular_tea + tax_regular_tea
  let total_revenue_large_tea := revenue_large_tea + tax_large_tea

  total_revenue_regular_coffee + total_revenue_large_coffee + total_revenue_regular_tea + total_revenue_large_tea

theorem coffee_shop_total_revenue : total_revenue_with_tax = 95.40 := 
  by
    sorry

end coffee_shop_total_revenue_l764_764201


namespace rachel_essay_time_l764_764806

/-- Rachel spends 9.69 hours to complete her essay given the described durations. --/

def total_hours_spent (researching:ℕ) (outline:ℕ) (brainstorming:ℕ) (first_draft_pages:ℕ)
  (first_draft_time_per_page:ℕ) (breaks:ℕ) (break_time:ℕ) (second_draft_pages:ℕ)
  (second_draft_time_per_page:ℕ) (editing:ℕ) (proofreading:ℕ) : ℝ :=
  (researching + outline + brainstorming 
  + first_draft_pages * first_draft_time_per_page
  + breaks * break_time 
  + second_draft_pages * second_draft_time_per_page 
  + editing + proofreading) / 3600.0

theorem rachel_essay_time :
  total_hours_spent 2700 900 1200 6 1800 5 600 6 1500 4500 1800 = 9.69 :=
by sorry

end rachel_essay_time_l764_764806


namespace sufficient_but_not_necessary_condition_l764_764654

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 > 1) ∧ ¬(x^2 > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l764_764654


namespace max_f_l764_764456

open Real

noncomputable def f (x : ℝ) : ℝ := 3 + log x + 4 / log x

theorem max_f (h : 0 < x ∧ x < 1) : f x ≤ -1 :=
sorry

end max_f_l764_764456


namespace product_of_solutions_eq_neg_35_l764_764870

theorem product_of_solutions_eq_neg_35 :
  ∀ (x : ℝ), -35 = -x^2 - 2 * x → ∃ (p : ℝ), p = -35 :=
by
  intro x h
  sorry

end product_of_solutions_eq_neg_35_l764_764870


namespace greatest_number_divides_with_remainders_l764_764884

theorem greatest_number_divides_with_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end greatest_number_divides_with_remainders_l764_764884


namespace cylinder_volume_l764_764720

theorem cylinder_volume (h : ℝ) (H1 : π * h ^ 2 = 4 * π) : (π * (h / 2) ^ 2 * h) = 2 * π :=
by
  sorry

end cylinder_volume_l764_764720


namespace total_yen_l764_764932

/-- 
Abe's family has a checking account with 6359 yen
and a savings account with 3485 yen.
-/
def checking_account : ℕ := 6359
def savings_account : ℕ := 3485

/-- 
Prove that the total amount of yen Abe's family has
is equal to 9844 yen.
-/
theorem total_yen : checking_account + savings_account = 9844 :=
by
  sorry

end total_yen_l764_764932


namespace number_of_pupils_in_class_l764_764916

variable (n : ℕ)

def marks_wrongly_entered := 73 - 45 = 28
def average_increase_per_pupil := n * (1 / 2) = 28

theorem number_of_pupils_in_class (h1 : marks_wrongly_entered) (h2 : average_increase_per_pupil) : n = 56 := 
sorry

end number_of_pupils_in_class_l764_764916


namespace girls_count_in_leos_class_l764_764728

def leo_class_girls_count (g b : ℕ) :=
  (g / b = 3 / 4) ∧ (g + b = 35) → g = 15

theorem girls_count_in_leos_class (g b : ℕ) :
  leo_class_girls_count g b :=
by
  sorry

end girls_count_in_leos_class_l764_764728


namespace transmitter_finding_probability_l764_764159

/-- 
  A license plate in the country Kerrania consists of 4 digits followed by two letters.
  The letters A, B, and C are used only by government vehicles while the letters D through Z are used by non-government vehicles.
  Kerrania's intelligence agency has recently captured a message from the country Gonzalia indicating that an electronic transmitter 
  has been installed in a Kerrania government vehicle with a license plate starting with 79. 
  In addition, the message reveals that the last three digits of the license plate form a palindromic sequence (meaning that they are 
  the same forward and backward), and the second digit is either a 3 or a 5. 
  If it takes the police 10 minutes to inspect each vehicle, what is the probability that the police will find the transmitter 
  within 3 hours, considering the additional restrictions on the possible license plate combinations?
-/
theorem transmitter_finding_probability :
  0.1 = 18 / 180 :=
by
  sorry

end transmitter_finding_probability_l764_764159


namespace find_a_max_min_l764_764457

theorem find_a_max_min (a M m x : ℝ) (f : ℝ → ℝ) (Hf : ∀ x ∈ Icc 0 6, f x = (x^2 - 6*x) * Real.sin(x - 3) + x + a)
  (Hmax_min : ∃ M m, ∀ x ∈ Icc 0 6, f(x) ≤ M ∧ f(x) ≥ m ∧ M + m = 8) : a = 1 := 
sorry

end find_a_max_min_l764_764457


namespace probability_of_point_in_sphere_l764_764551

noncomputable def probability_inside_sphere : ℝ :=
  let cube_volume := 4 ^ 3 in
  let sphere_volume := (4 / 3) * Real.pi * (2 ^ 3) in
  sphere_volume / cube_volume

theorem probability_of_point_in_sphere :
  ∀ (x y z : ℝ), 
    (-2 ≤ x ∧ x ≤ 2) ∧ 
    (-2 ≤ y ∧ y ≤ 2) ∧ 
    (-2 ≤ z ∧ z ≤ 2) → 
    (probability_inside_sphere = (Real.pi / 6)) := by
  sorry

end probability_of_point_in_sphere_l764_764551


namespace triangle_count_l764_764219

def count_triangles (smallest intermediate larger even_larger whole_structure : Nat) : Nat :=
  smallest + intermediate + larger + even_larger + whole_structure

theorem triangle_count :
  count_triangles 2 6 6 6 12 = 32 :=
by
  sorry

end triangle_count_l764_764219


namespace triangle_reflection_similarity_l764_764473

theorem triangle_reflection_similarity
  (ABC : Triangle)
  (H : Point)
  (H'_1 H'_2 H'_3 : Point)
  (is_orthocenter : is_orthocenter(ABC, H))
  (feet_of_perpendiculars : feet_of_perpendiculars(ABC, H, H'_1, H'_2, H'_3))
  (cyclic_HH'_1H'_2H'_3 : cyclic (H, H'_1, H'_2, H'_3))
  (parallel_HH'_1_BC : HH'_1 ∥ BC)
  (parallel_HH'_2_CA : HH'_2 ∥ CA)
  (parallel_HH'_3_AB : HH'_3 ∥ AB) :
  similar (Triangle.mk H'_1 H'_2 H'_3) ABC :=
sorry

end triangle_reflection_similarity_l764_764473


namespace smallest_t_satisfies_equation_l764_764636

def satisfies_equation (t x y : ℤ) : Prop :=
  (x^2 + y^2)^2 + 2 * t * x * (x^2 + y^2) = t^2 * y^2

theorem smallest_t_satisfies_equation : ∃ t x y : ℤ, t > 0 ∧ x > 0 ∧ y > 0 ∧ satisfies_equation t x y ∧
  ∀ t' x' y' : ℤ, t' > 0 ∧ x' > 0 ∧ y' > 0 ∧ satisfies_equation t' x' y' → t' ≥ t :=
sorry

end smallest_t_satisfies_equation_l764_764636


namespace investor_difference_l764_764575

def investment_A : ℝ := 300
def investment_B : ℝ := 200
def rate_A : ℝ := 0.30
def rate_B : ℝ := 0.50

theorem investor_difference :
  ((investment_A * (1 + rate_A)) - (investment_B * (1 + rate_B))) = 90 := 
by
  sorry

end investor_difference_l764_764575


namespace triangle_side_length_l764_764747

theorem triangle_side_length (A : ℝ) (b : ℝ) (S : ℝ) (hA : A = 120) (hb : b = 4) (hS: S = 2 * Real.sqrt 3) : 
  ∃ c : ℝ, c = 2 := 
by 
  sorry

end triangle_side_length_l764_764747


namespace angle_sum_eq_114_l764_764951

noncomputable def sum_of_cis (angles : List ℝ) : ℂ :=
  angles.foldl (λ acc x, acc + Complex.cis x) 0

theorem angle_sum_eq_114 (r : ℝ) (θ : ℝ)
  (h_r_pos : r > 0)
  (h_theta : 0 ≤ θ ∧ θ < 360)
  (h_sum : sum_of_cis [70, 78, 86, 94, 102, 110, 118, 126, 134, 142, 150, 158] = r * Complex.cis θ) :
  θ = 114 :=
sorry

end angle_sum_eq_114_l764_764951


namespace problem_statement_l764_764268

noncomputable theory
open Classical

variable {f : ℝ → ℝ}

-- Given conditions
def decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y < f x

def is_even_fn_shifted (f : ℝ → ℝ) : Prop := ∀ x, f(x + 1) = f(-x + 1)

-- Statement to prove
theorem problem_statement (h1 : decreasing_on f {x | 1 < x})
                           (h2 : is_even_fn_shifted f) :
  f(0) > f(3) :=
sorry

end problem_statement_l764_764268


namespace product_ABCD_is_9_l764_764312

noncomputable def A : ℝ := Real.sqrt 2018 + Real.sqrt 2019 + 1
noncomputable def B : ℝ := -Real.sqrt 2018 - Real.sqrt 2019 - 1
noncomputable def C : ℝ := Real.sqrt 2018 - Real.sqrt 2019 + 1
noncomputable def D : ℝ := Real.sqrt 2019 - Real.sqrt 2018 + 1

theorem product_ABCD_is_9 : A * B * C * D = 9 :=
by sorry

end product_ABCD_is_9_l764_764312


namespace trajectory_of_point_l764_764340

theorem trajectory_of_point (x y : ℝ) :
  (∀ z : ℂ, ∃ x y : ℝ, z = (x^2 + y^2 - 4) + (x - y) * Complex.I 
      → (RealPart z = 0 → x ≠ y)) 
  → (x^2 + y^2 = 4 ∧ x ≠ y ↔ (x, y) ∈ {p : ℝ × ℝ | p ∈ (λ q, q.1 ^ 2 + q.2 ^ 2 = 4 ∧ q.1 ≠ q.2)({(x, y) | x^2 + y^2 = 4 \ {⟨√2, √2⟩, ⟨-√2, -√2⟩}})) :=
sorry

end trajectory_of_point_l764_764340


namespace median_room_number_l764_764581

theorem median_room_number (occupied_rooms : List ℕ) (n : ℕ) (not_arrived : List ℕ) (h1 : occupied_rooms = List.filter (λ x, ¬ List.elem x not_arrived) [1, 2, .., n])
  (h2 : n = 24) (h3 : not_arrived = [15, 16, 17]) : List.nthLe (List.sort (≤) occupied_rooms) (List.length occupied_rooms / 2) sorry = 11 :=
by
  sorry

end median_room_number_l764_764581


namespace james_car_purchase_l764_764751

/-- 
James sold his $20,000 car for 80% of its value, 
then bought a $30,000 sticker price car, 
and he was out of pocket $11,000. 
James bought the new car for 90% of its value. 
-/
theorem james_car_purchase (V_1 P_1 V_2 O P : ℝ)
  (hV1 : V_1 = 20000)
  (hP1 : P_1 = 80)
  (hV2 : V_2 = 30000)
  (hO : O = 11000)
  (hSaleOld : (P_1 / 100) * V_1 = 16000)
  (hDiff : 16000 + O = 27000)
  (hPurchase : (P / 100) * V_2 = 27000) :
  P = 90 := 
sorry

end james_car_purchase_l764_764751


namespace equal_black_white_area_l764_764105

def chessboard := fin 8 × fin 8

def is_black (pos : chessboard) : Prop :=
  (pos.1.val + pos.2.val) % 2 = 0

def polygonal_line (ps : list chessboard) : Prop :=
  ps.nodup ∧ (∀ i, i < ps.length - 1 → 
    (abs (ps[i].1.val - ps[i+1].1.val) ≤ 1 ∧ abs (ps[i].2.val - ps[i+1].2.val) ≤ 1))

theorem equal_black_white_area (ps : list chessboard) (h : polygonal_line ps) :
  let enclosed_area := list.prod (λ p, if is_black p then 1 else -1) ps
  in enclosed_area = 0 :=
by {
  sorry
}

end equal_black_white_area_l764_764105


namespace minute_hand_length_l764_764832

noncomputable def length_of_minute_hand (A : ℝ) : ℝ :=
  let θ := Real.pi / 3
  let one_over_two := 1 / 2
  let area_formula := A = one_over_two * (r * r) * θ
  let r_squared := (A * 6) / Real.pi
  Real.sqrt r_squared

theorem minute_hand_length :
    ∀ (A : ℝ), 
    A = 15.845238095238093 
    → length_of_minute_hand A ≈ 5.5 := 
  by
  sorry

end minute_hand_length_l764_764832


namespace evaluate_expression_l764_764614

noncomputable def absoluteValue (x : ℝ) : ℝ := |x|

noncomputable def ceilingFunction (x : ℝ) : ℤ := ⌈x⌉

theorem evaluate_expression : ceilingFunction (absoluteValue (-52.7)) = 53 :=
by
  sorry

end evaluate_expression_l764_764614


namespace least_possible_third_side_l764_764351

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) : ∃ c, c = 24 ∧ a^2 - c^2 = 527  :=
by
  use (√527)
  sorry

end least_possible_third_side_l764_764351


namespace circumscribed_sphere_exists_l764_764902

-- Define the problem conditions
def ConvexSolid (side_length : ℝ) :=
  (bounded_by_12_equilateral_triangles : Bool) ∧
  (bounded_by_2_regular_hexagons : Bool) ∧
  (hexagons_planes_parallel : Bool)

-- Define the proof problem
theorem circumscribed_sphere_exists (side_length : ℝ) (solid : ConvexSolid side_length) :
  solid.bounded_by_12_equilateral_triangles = true
  → solid.bounded_by_2_regular_hexagons = true
  → solid.hexagons_planes_parallel = true
  → ∃ (r : ℝ), r = (1/2) * real.sqrt (3 + real.sqrt 3) :=
by
  sorry

end circumscribed_sphere_exists_l764_764902


namespace measure_of_A_area_of_triangle_l764_764277

-- Given definitions
variable (A B C a b c : ℝ)

-- Triangle vertices on the unit circle implies sides and angle constraints
def on_unit_circle : Prop :=
  ∀ (A B C : ℝ), A^2 + B^2 = 1 ∧ B^2 + C^2 = 1 ∧ C^2 + A^2 = 1

-- Given condition: b^2 + c^2 = a^2 + bc
axiom sides_condition : b^2 + c^2 = a^2 + bc

-- Proof Problem 1: Measure of angle A
theorem measure_of_A (h_circle : on_unit_circle A B C) (h_condition : sides_condition) :
  A = π / 3 :=
sorry

-- Proof Problem 2: Area of triangle ABC
theorem area_of_triangle (h_circle : on_unit_circle A B C) (h_condition : sides_condition) (h_bc_sum : b^2 + c^2 = 4) :
  let bc := 1 in
  (1 / 2) * bc * (√3 / 2) = √3 / 4 :=
sorry

end measure_of_A_area_of_triangle_l764_764277


namespace calculate_discount_l764_764405

def original_price := 22
def sale_price := 16

theorem calculate_discount : original_price - sale_price = 6 := 
by
  sorry

end calculate_discount_l764_764405


namespace sum_of_arithmetic_sequence_6_7_8_l764_764688

theorem sum_of_arithmetic_sequence_6_7_8 {a : ℕ → ℝ} (a1 : ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_13 : ∑ i in Finset.range 13, a (i + 1) = 39) : 
  a 6 + a 7 + a 8 = 39 :=
sorry

end sum_of_arithmetic_sequence_6_7_8_l764_764688


namespace four_digit_integer_l764_764075

theorem four_digit_integer (a b c d : ℕ) (h1 : a + b + c + d = 17) (h2 : b + c = 8) (h3 : a - d = 3) (h4 : 6 ≤ a ∧ a ≤ 9) (h5 : 0 ≤ b ∧ b ≤ 9) (h6 : 0 ≤ c ∧ c ≤ 9) (h7 : 0 ≤ d ∧ d ≤ 3) : (1000 * a + 100 * b + 10 * c + d = 6443) :=
by
  have h8 : a + d = 9, from
    calc
      a + d = (a + b + c + d) - (b + c) : by sorry
      ... = 17 - 8 : by rw [h1, h2]
      ... = 9 : by norm_num,
  have h9 : 2 * a = 12, from
    calc
      2 * a = (a + a) : by norm_num
      ... = (a + d) + (a - d) : by rw [←h3]
      ... = 9 + 3 : by rw [h8]
      ... = 12 : by norm_num,
  have h10 : a = 6, from by linarith only [h9],
  have h11 : d = 3, from by linarith only [h8, h10],
  have h12 : (b + c = 8) from h2,
  have h13 : 7 ∣ (1000 * a + 100 * b + 10 * c + d), from by sorry,
  sorry

end four_digit_integer_l764_764075


namespace jet_ski_travel_time_l764_764157

variables (t r p : ℝ)

theorem jet_ski_travel_time :
  let p := r + 10 in
  let total_distance := 60 in
  (p + r) * t = total_distance →
  r * 8 = total_distance + (p - r) * (8 - t) →
  t = 3 :=
by
  intros
  sorry

end jet_ski_travel_time_l764_764157


namespace total_heartbeats_during_race_l764_764539

-- Definitions for conditions
def heart_rate_per_minute : ℕ := 120
def pace_minutes_per_km : ℕ := 4
def race_distance_km : ℕ := 120

-- Lean statement of the proof problem
theorem total_heartbeats_during_race :
  120 * (4 * 120) = 57600 := by
  sorry

end total_heartbeats_during_race_l764_764539


namespace probability_computation_l764_764552

noncomputable def probability_inside_sphere : ℝ :=
  let volume_of_cube : ℝ := 64
  let volume_of_sphere : ℝ := (4/3) * Real.pi * (2^3)
  volume_of_sphere / volume_of_cube

theorem probability_computation :
  probability_inside_sphere = Real.pi / 6 :=
by
  sorry

end probability_computation_l764_764552


namespace value_of_z_plus_nine_over_z_l764_764779

noncomputable def complex_condition (z : ℂ) : Prop :=
  15 * (complex.abs z)^2 = 3 * (complex.abs (z + 3))^2 + (complex.abs (z^2 + 2))^2 + 36

theorem value_of_z_plus_nine_over_z (z : ℂ) (h : complex_condition z) : z + 9/z = 1 := by
  sorry

end value_of_z_plus_nine_over_z_l764_764779


namespace bonus_distributed_correctly_l764_764439

def amount_received (A B C D E F : ℝ) :=
  -- Conditions
  (A = 2 * B) ∧ 
  (B = C) ∧ 
  (D = 2 * B - 1500) ∧ 
  (E = C + 2000) ∧ 
  (F = 1/2 * (A + D)) ∧ 
  -- Total bonus amount
  (A + B + C + D + E + F = 25000)

theorem bonus_distributed_correctly :
  ∃ (A B C D E F : ℝ), 
    amount_received A B C D E F ∧ 
    A = 4950 ∧ 
    B = 2475 ∧ 
    C = 2475 ∧ 
    D = 3450 ∧ 
    E = 4475 ∧ 
    F = 4200 :=
sorry

end bonus_distributed_correctly_l764_764439


namespace lateral_area_of_cylinder_l764_764455

theorem lateral_area_of_cylinder (d h : ℝ) (π : ℝ) (diam_eq : d = 4) (height_eq : h = 4) : 
  (lateral_area : ℝ) := π * d * h = 16 * π :=
by sorry

end lateral_area_of_cylinder_l764_764455


namespace sqrt_mixed_number_simplified_l764_764984

theorem sqrt_mixed_number_simplified :
  (sqrt (10 + 1/9)) = (sqrt 91) / 3 :=
sorry

end sqrt_mixed_number_simplified_l764_764984


namespace at_least_n_squared_div_2_towers_l764_764853

-- Definition of the problem
def has_at_least_n_towers (n : ℕ) (t : ℕ → ℕ → Prop) :=
  ∀ i j, ¬t i j → (∑ i', t i' j) ≥ n ∧ (∑ j', t i j') ≥ n

-- Main theorem
theorem at_least_n_squared_div_2_towers {n : ℕ} (t : ℕ → ℕ → Prop) (h : has_at_least_n_towers n t) :
  ∑ i j, t i j ≥ n^2 / 2 :=
sorry

end at_least_n_squared_div_2_towers_l764_764853


namespace one_fourth_of_8_point_8_is_fraction_l764_764998

theorem one_fourth_of_8_point_8_is_fraction:
  (1 / 4) * 8.8 = 11 / 5 :=
by sorry

end one_fourth_of_8_point_8_is_fraction_l764_764998


namespace eagles_panthers_first_half_points_sum_eq_59_l764_764730

theorem eagles_panthers_first_half_points_sum_eq_59
  (a k b d : ℕ) 
  (h1 : a + 1 * k = a + k) 
  (h2 : a + 2 * k = a + 4 * k) 
  (h3 : a + 3 * k = a + 9 * k) 
  (h4 : b + 1 * d = b + d) 
  (h5 : b + 2 * d = b + 2 * d) 
  (h6 : b + 3 * d = b + 3 * d)
  (eagles_score_eq : (4 * a + 14 * k))
  (panthers_score_eq : (4 * b + 6 * d))
  (win_by_two : 4 * a + 14 * k = 4 * b + 6 * d + 2) :
  a + (a + k) + b + (b + d) = 59 :=
by
  sorry

end eagles_panthers_first_half_points_sum_eq_59_l764_764730


namespace correct_calculation_l764_764506

theorem correct_calculation :
  \(\sqrt{(-5)^2}\) = 5 :=
by
  let a := (-5)
  have h1 : a^2 = 25 :=
    by calc
      a^2 = (-5) * (-5) : by rw [sq]
        ... = 25 : by norm_num
  have h2 : sqrt a^2 = sqrt 25 :=
    by rw [h1]
  have h3 : sqrt 25 = 5 :=
    by norm_num
  show sqrt a^2 = 5 :=
    by rw [h2, h3]

end correct_calculation_l764_764506


namespace fraction_of_work_left_correct_l764_764135

-- Define the conditions for p, q, and r
def p_one_day_work : ℚ := 1 / 15
def q_one_day_work : ℚ := 1 / 20
def r_one_day_work : ℚ := 1 / 30

-- Define the total work done in one day by p, q, and r
def total_one_day_work : ℚ := p_one_day_work + q_one_day_work + r_one_day_work

-- Define the work done in 4 days
def work_done_in_4_days : ℚ := total_one_day_work * 4

-- Define the fraction of work left after 4 days
def fraction_of_work_left : ℚ := 1 - work_done_in_4_days

-- Statement to prove
theorem fraction_of_work_left_correct : fraction_of_work_left = 2 / 5 := by
  sorry

end fraction_of_work_left_correct_l764_764135


namespace expected_value_of_weighted_dice_l764_764184

theorem expected_value_of_weighted_dice :
  let p2 := (1:ℚ) / 4,
      p5 := (1:ℚ) / 2,
      p_other := (1:ℚ) / 12,
      v2 := (4:ℚ),
      v5 := (6:ℚ),
      v_other := (-3:ℚ) in
  let E := p2 * v2 + p5 * v5 + 4 * p_other * v_other in
  E = 3 :=
by
  sorry

end expected_value_of_weighted_dice_l764_764184


namespace coloring_problem_l764_764960

theorem coloring_problem (n : ℕ) (h : n ≥ 2) : 
  ∃ (a : Fin n → ℕ), 
  (∀ i j, i < j → a i < a j) ∧ 
  (∀ (k : Fin (n-1)), a (⟨k.val+1, Nat.lt_of_succ_lt_succ k.isLt⟩) - a k ≤ a ⟨k.val+2, sorry⟩ - a ⟨k.val+1, sorry⟩) :=
sorry

end coloring_problem_l764_764960


namespace g_is_odd_l764_764002

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  -- The proof can be filled in here
  sorry

end g_is_odd_l764_764002


namespace binom_20_17_l764_764590

theorem binom_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end binom_20_17_l764_764590


namespace convex_pentagon_exists_l764_764038

theorem convex_pentagon_exists (A : Fin 9 → ℝ × ℝ) 
    (h_no_collinear : ∀ (i j k : Fin 9), i ≠ j → j ≠ k → i ≠ k → ¬ collinear {A i, A j, A k}) : 
    ∃ (S : Finset (Fin 9)), S.card = 5 ∧ convex_hull ℝ (S.image A).points ∧ is_convex_polygon (S.image A).points :=
sorry

end convex_pentagon_exists_l764_764038


namespace median_books_bought_l764_764084

theorem median_books_bought (books : List ℕ) (h : books = [6, 3, 7, 1, 8, 5, 9, 2]) : 
  let sorted_books := books.qsort (· ≤ ·)
  let n := sorted_books.length
  let m := (sorted_books[(n / 2).pred] + sorted_books[n / 2]) / 2
  m = 5.5 :=
by 
suffices sorted_if : sorted_books = [1, 2, 3, 5, 6, 7, 8, 9] from
have mid_vals : sorted_books = [1, 2, 3, 5, 6, 7, 8, 9] ∧ n = 8 ∧ m = 5.5 := sorry
exact sorry

end median_books_bought_l764_764084


namespace area_triangle_MSQ_l764_764375

-- Given definitions and conditions
variables (M N P Q R S : Type) [point : Type] (circumcircle : circle M N P)
variables (MQ NR : line) (MN : length) (intersects : extends NR intersects circumcircle at S)
variables (MQ_length : length MQ = 15) (NR_length : length NR = 20) (MN_length : length MN = 30)
variables (Q_midpoint_np : midpoint Q N P) (R_midpoint_mp : midpoint R M P)

-- Statement of the problem
theorem area_triangle_MSQ:
  area (triangle M S Q) = 15 * sqrt 143 :=
sorry

end area_triangle_MSQ_l764_764375


namespace perp_condition_vector_difference_magnitude_l764_764325

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))
noncomputable def vector_c : ℝ × ℝ := (Real.sqrt 3, -1)

-- Condition for perpendicular vectors
theorem perp_condition (x : ℝ) :
  vector_a x.1 * vector_b x.1 + vector_a x.2 * vector_b x.2 = 0 ↔
  ∃ k : ℤ, x = k * (Real.pi / 2) + (Real.pi / 4) := sorry

-- Magnitude bounds for vector difference
theorem vector_difference_magnitude (x : ℝ) :
  1 ≤ Real.sqrt ((vector_a x).1 - vector_c.1)^2 + ((vector_a x).2 - vector_c.2)^2 ≤ 3 := sorry

end perp_condition_vector_difference_magnitude_l764_764325


namespace find_x_l764_764256

theorem find_x (x : ℕ) (h1 : 8 = 2 ^ 3) (h2 : 32 = 2 ^ 5) :
  (2^(x+2) * 8^(x-1) = 32^3) ↔ (x = 4) :=
by
  sorry

end find_x_l764_764256


namespace find_angle_C_range_of_ab_l764_764283

noncomputable def acute_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ (triangle : ∡) (acos_A : polynomial.to_fun_of_is_root A) 
                     (acos_B : polynomial.to_fun_of_is_root B)
                     (acos_C : polynomial.to_fun_of_is_root C),
    is_angle_acute_triangle ∧ triangle A B C ∧
    angle_opposite A a ∧ angle_opposite B b ∧ angle_opposite C c ∧
    tan C = (a * b) / (a^2 + b^2 - c^2)

theorem find_angle_C (a b c A B C : ℝ) : 
  acute_triangle a b c A B C →
  C = 30 := by sorry

theorem range_of_ab (a b : ℝ) (A B C : ℝ) :
  acute_triangle a b 1 A B C →
  2 * sqrt 3 < a * b ∧ a * b ≤ 2 + sqrt 3 := by sorry

end find_angle_C_range_of_ab_l764_764283


namespace sum_of_selected_terms_l764_764687

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function from natural numbers to rational numbers

noncomputable def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_of_selected_terms (h₁ : sum_first_n_terms a 13 = 39) : a 6 + a 7 + a 8 = 13 :=
sorry

end sum_of_selected_terms_l764_764687


namespace speedboat_travel_time_l764_764101

-- Define constants and variables based on the given conditions
def length_of_river : ℝ := sorry
def time_first_segment : ℝ := 50
def time_second_segment : ℝ := 20

-- Define the speeds based on the given conditions
def speed_still_water : ℝ := (length_of_river / 3) / time_first_segment
def speed_water_flow : ℝ := ((length_of_river / 3) / time_second_segment) - speed_still_water

theorem speedboat_travel_time : (50 + 20 + (length_of_river / 3) / speed_water_flow) = 100 / 3 := by
  sorry

end speedboat_travel_time_l764_764101


namespace number_of_piglets_born_l764_764541

-- Define the conditions
def sell_price (pig: Type) : ℕ := 300
def feed_cost_per_month (pig: Type) : ℕ := 10
def pigs_sold_12_months : ℕ := 3
def pigs_sold_16_months : ℕ := 3
def total_profit : ℕ := 960

theorem number_of_piglets_born :
  (∀ (P: Type), sell_price P * 6 - ((feed_cost_per_month P * 12 * pigs_sold_12_months) + (feed_cost_per_month P * 16 * pigs_sold_16_months)) = total_profit) → 6 :=
by intro h; exact 6 -- sorry, proof is skipped

end number_of_piglets_born_l764_764541


namespace least_positive_t_l764_764963

noncomputable def arcsin_sin_gp (α t : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧
  (arcsin (sin (3 * α)) = if 0 < 3 * α ∧ 3 * α ≤ π / 2 then 3 * α else π - 3 * α) ∧
  (arcsin (sin (5 * α)) = if 0 < 5 * α ∧ 5 * α ≤ π / 2 then 5 * α else π - 5 * α) ∧
  (arcsin (sin (t * α)) = if 0 < t * α ∧ t * α ≤ π / 2 then t * α else π - t * α) ∧
  (arcsin (sin (3 * α)) / arcsin (sin α) = arcsin (sin (5 * α)) / arcsin (sin (3 * α))) ∧
  (arcsin (sin (5 * α)) / arcsin (sin (3 * α)) = arcsin (sin (t * α)) / arcsin (sin (5 * α)))

theorem least_positive_t (α : ℝ) :
  0 < α ∧ α < π / 2 →
  ∃ t : ℝ, t > 0 ∧ arcsin_sin_gp α t ∧ t = 3 * (π - 5 * α) / (π - 3 * α) :=
sorry

end least_positive_t_l764_764963


namespace cost_to_paint_cube_l764_764136

def side_length := 30 -- in feet
def cost_per_kg := 40 -- Rs. per kg
def coverage_per_kg := 20 -- sq. ft. per kg

def area_of_one_face := side_length * side_length
def total_surface_area := 6 * area_of_one_face
def paint_required := total_surface_area / coverage_per_kg
def total_cost := paint_required * cost_per_kg

theorem cost_to_paint_cube : total_cost = 10800 := 
by
  -- proof here would follow the solution steps provided in the solution part, which are omitted
  sorry

end cost_to_paint_cube_l764_764136


namespace problem_statement_l764_764282

open Nat

theorem problem_statement (n a : ℕ) 
  (hn : n > 1) 
  (ha : a > n^2)
  (H : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ k, a + i = (n^2 + i) * k) :
  a > n^4 - n^3 := 
sorry

end problem_statement_l764_764282


namespace num_program_orders_l764_764729

theorem num_program_orders :
  let programs := ["singing", "dancing", "skit", "cross_talk", "recitation", "game"],
  (game := "game"),
  (paired := {programs // ∀ p q : String, (p = "singing" ∧ q = "dancing") ∨ (p = "dancing" ∧ q = "singing")}),
  |paired| = 1 →
  ∃ orders : Set (List String),
    (∀ order ∈ orders, game ≠ order.head) ∧          -- Game is not the first program
    (∀ order ∈ orders, paired ⊆ 
        {⟨x, y⟩ | (x, y) = ("singing", "dancing") ∨ (x, y) = ("dancing", "singing")}) →  -- Singing and dancing are adjacent
    orders.card = 192 := sorry

end num_program_orders_l764_764729


namespace no_base_450_odd_last_digit_l764_764253

theorem no_base_450_odd_last_digit :
  ¬ ∃ b : ℕ, b^3 ≤ 450 ∧ 450 < b^4 ∧ (450 % b) % 2 = 1 :=
sorry

end no_base_450_odd_last_digit_l764_764253


namespace example_arrangement_l764_764943

noncomputable def arrangement (A B C D E : ℕ) : Prop :=
  (B = 3 * A ∨ B = 9 * A) ∧
  (C = 3 * A ∨ C = 9 * A) ∧
  (D = 3 * B ∨ D = 9 * B) ∧
  (E = 3 * C ∨ E = 9 * C) ∧
  (E = 3 * D ∨ E = 9 * D) ∧
  (C ≠ 3 * B) ∧ (C ≠ 9 * B) ∧
  (D ≠ 3 * A) ∧ (D ≠ 9 * A) ∧
  (E ≠ 3 * A) ∧ (E ≠ 9 * A) ∧
  (E ≠ 3 * B) ∧ (E ≠ 9 * B)

theorem example_arrangement : ∃ (A B C D E : ℕ),
  arrangement A B C D E :=
begin
  use [1, 3, 9, 27, 81],
  unfold arrangement,
  by {
    split,
    { left, refl },
    split,
    { right, norm_num },
    split,
    { right, norm_num },
    split,
    { right, norm_num },
    split,
    { left, norm_num },
    repeat { split; norm_num; linarith }
  }
end

end example_arrangement_l764_764943


namespace length_of_second_train_is_319_95_l764_764895

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kph : ℝ) (speed_second_train_kph : ℝ) (time_to_cross_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kph * 1000 / 3600
  let speed_second_train_mps := speed_second_train_kph * 1000 / 3600
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross_seconds
  let length_second_train := total_distance_covered - length_first_train
  length_second_train

theorem length_of_second_train_is_319_95 :
  length_of_second_train 180 120 80 9 = 319.95 :=
sorry

end length_of_second_train_is_319_95_l764_764895


namespace pyramid_sine_correct_k_valid_range_l764_764435

noncomputable def pyramid_sine (α k : ℝ): ℝ :=
  if k > 5 * (Real.sin α) then Real.sin α / (k - 4 * (Real.sin α)) else 0

theorem pyramid_sine_correct (α k : ℝ) (hα : 0 < α ∧ α < π / 2) (hk : k > 5 * (Real.sin α)) :
  pyramid_sine α k = Real.sin α / (k - 4 * (Real.sin α)) :=
by {
  simp [pyramid_sine, hk],
  sorry
}

theorem k_valid_range (α k : ℝ) (hα : 0 < α ∧ α < π / 2) :
  k > 5 * (Real.sin α) :=
by {
  sorry
}

end pyramid_sine_correct_k_valid_range_l764_764435


namespace problem_1_problem_2_l764_764486

def dice := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

def probability {α : Type*} (s : Finset α) (p : α → Prop) : ℚ :=
  (s.filter p).card / s.card

theorem problem_1 :
  probability ({(m, n) | m ∈ dice ∧ n ∈ dice}.to_finset) (λ pair, let (m, n) := pair in m + n ≤ 4)
    = 1 / 6 :=
by
  sorry

theorem problem_2 :
  probability ({(m, n) | m ∈ dice ∧ n ∈ dice}.to_finset) (λ pair, let (m, n) := pair in m < n + 2)
    = 13 / 18 :=
by
  sorry

end problem_1_problem_2_l764_764486


namespace γ_n_between_zero_and_one_γ_converges_and_bounded_l764_764701

-- Define γ_n sequence
def γ_n (n : ℕ) : ℝ := (Finset.range (n-1)).sum (λ k => 1 / (k + 1)) - Real.log n

-- Statement for part (a)
theorem γ_n_between_zero_and_one (n : ℕ) (hn : n > 0) : 0 < γ_n n ∧ γ_n n < 1 :=
sorry

-- Statement for part (b)
theorem γ_converges_and_bounded : ∃ γ : ℝ, (0 < γ ∧ γ < 1) ∧ Tendsto γ_n at_top (𝓝 γ) :=
sorry

end γ_n_between_zero_and_one_γ_converges_and_bounded_l764_764701


namespace positive_difference_of_two_numbers_l764_764094

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℤ), (x + y = 40) ∧ (3 * y - 2 * x = 8) ∧ (|y - x| = 4) :=
by
  sorry

end positive_difference_of_two_numbers_l764_764094


namespace perpendicular_tangent_line_l764_764625

theorem perpendicular_tangent_line (a b : ℝ) : 
  let line_slope := 1 / 3,
      perp_slope := -3,
      curve := λ x, x^3 + 3 * x^2 - 5,
      derivative_curve := λ x, 3 * x^2 + 6 * x,
      a_eqn := a = -1,
      b_eqn := b = -3
  in
  (2 * a - 6 * b + 1 = 0) ∧ (a, b) = P ∧ (derivative_curve a = -3) → 
  3 * x + y + 6 = 0 :=
by
  sorry

end perpendicular_tangent_line_l764_764625


namespace intercept_sum_l764_764141

theorem intercept_sum (x0 y0 : ℤ) 
  (hx : 0 ≤ x0 ∧ x0 < 35) 
  (hy : 0 ≤ y0 ∧ y0 < 35) 
  (hx_intercept : 3 * x0 ≡ -1 [MOD 35])
  (hy_intercept : 4 * y0 ≡ 1 [MOD 35]) : 
  x0 + y0 = 32 := 
by
  sorry

end intercept_sum_l764_764141


namespace percentage_exceeds_self_l764_764548

theorem percentage_exceeds_self (N : ℕ) (P : ℝ) (h1 : N = 150) (h2 : N = (P / 100) * N + 126) : P = 16 := by
  sorry

end percentage_exceeds_self_l764_764548


namespace tim_balloons_l764_764220

theorem tim_balloons (dan_balloons : ℕ) (h : dan_balloons = 29) 
    (times_more : ℕ) (h' : times_more = 7) : 
    let tim_balloons := times_more * dan_balloons in 
    tim_balloons = 203 := 
by 
  -- Define the number of balloons Tim has 
  let tim_balloons := times_more * dan_balloons 
  -- Substitute the conditions
  rw [h, h'], 
  -- Calculate the number of balloons
  simp,
  -- This is the result 
  exact rfl

end tim_balloons_l764_764220


namespace find_x_l764_764333

-- Define the function h
def h (x : ℝ) := (x + 5) ^ (1 / 3) / 5 ^ (1 / 3)

-- State the theorem
theorem find_x (x : ℝ) (hx : h (2 * x) = 4 * h (x)) : x = -315 / 62 := by
  sorry

end find_x_l764_764333


namespace phantom_needs_more_money_l764_764418

def amount_phantom_has : ℤ := 50
def cost_black : ℤ := 11
def count_black : ℕ := 2
def cost_red : ℤ := 15
def count_red : ℕ := 3
def cost_yellow : ℤ := 13
def count_yellow : ℕ := 2

def total_cost : ℤ := cost_black * count_black + cost_red * count_red + cost_yellow * count_yellow
def additional_amount_needed : ℤ := total_cost - amount_phantom_has

theorem phantom_needs_more_money : additional_amount_needed = 43 := by
  sorry

end phantom_needs_more_money_l764_764418


namespace pq_range_l764_764313

noncomputable def parabola (y : ℝ) := y ^ 2 = 4 * x
noncomputable def ellipse (x y : ℝ) := x ^ 2 / 4 + y^2 / 3 = 1

theorem pq_range (x y k λ : ℝ) (h₀ : parabola (λ * y)) (h₁ : λ ∈ set.Ico (1 / 2 : ℝ) 1)  :
  ∃ a b, (∀ P Q : ℝ, P ≠ Q → (a = 4 ∧ b = 2) → | |P - Q| ∈ set.Icc 0 (Real.sqrt 17 / 2)) :=
sorry

end pq_range_l764_764313


namespace sum_of_areas_eq_l764_764659

-- Let us define the concepts and conditions in the problem
variables (ABCD : Type) [convex_quadrilateral ABCD]
variables (k : ℕ) [fact (0 < k)]
variables (area : ABCD → ℝ)
variables (division_points : ABCD → ℕ → list (ABCD → ℝ))

-- Define the function that fetches the k chosen quadrilaterals meeting the separation conditions
noncomputable def chosen_quadrilaterals (k : ℕ) : set (ABCD → ℝ) :=
{ q | ∃ (l1 l2 : list (ABCD → ℝ)), 
    l1.length = k ∧ l2.length = k ∧
    (∀ i j, i ≠ j → (q ∈ l1 → q ∉ l2)) }

-- Define the proof statement
theorem sum_of_areas_eq (ABCD : Type) [convex_quadrilateral ABCD]
 (k : ℕ) [fact (0 < k)]
 (S_ABCD : ℝ) (chosen_areas : set (ℝ)) :
   (∀ q ∈ chosen_quadrilaterals k, ∃ area_q, area q = some area_q) → 
    ∑ a ∈ chosen_areas, a = (1 : ℝ) / k * S_ABCD :=
by
  sorry

end sum_of_areas_eq_l764_764659


namespace y_intercept_not_z_l764_764370

variable (a b z x y : ℝ)

-- Given conditions
def objective_function (a b x y z : ℝ) : Prop := z = a * x + b * y ∧ b ≠ 0

-- The proof problem: Prove that the y-intercept of the line ax + by - z = 0 is z / b
theorem y_intercept_not_z (h : objective_function a b x y z) : let y_intercept := z / b in y_intercept ≠ z :=
by
  sorry

end y_intercept_not_z_l764_764370


namespace sum_of_T_elements_l764_764756

-- Define T to represent the set of numbers 0.abcd with a, b, c, d being distinct digits.
def is_valid_abcd (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def T : Set ℝ := { x | ∃ (a b c d : ℕ), is_valid_abcd a b c d ∧ x = ((1000 * a + 100 * b + 10 * c + d : ℝ) / 9999) }

-- The main theorem statement.
theorem sum_of_T_elements : ∑ x in T, x = 2520 := by
  sorry

end sum_of_T_elements_l764_764756


namespace net_change_in_price_l764_764721

-- Definitions for initial price, percentage decrements, and increments
variable (P : ℝ)

def price_after_decrease : ℝ := P * 0.70
def price_after_increase : ℝ := price_after_decrease P * 1.40

theorem net_change_in_price (P : ℝ) : price_after_increase P - P = -0.02 * P := by
  simp [price_after_decrease, price_after_increase]
  sorry  -- Proof goes here

end net_change_in_price_l764_764721


namespace fraction_remain_same_l764_764335

theorem fraction_remain_same (x y : ℝ) : (2 * x + y) / (3 * x + y) = (2 * (10 * x) + (10 * y)) / (3 * (10 * x) + (10 * y)) :=
by sorry

end fraction_remain_same_l764_764335


namespace sum_proper_divisors_of_24_l764_764587

-- Definitions/Conditions
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d, d < n ∧ n % d = 0)

def sum_list (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

-- Statement expressing the problem and the expected outcome
theorem sum_proper_divisors_of_24 : sum_list (proper_divisors 24) = 36 := by
  sorry

end sum_proper_divisors_of_24_l764_764587


namespace solution_inequalities_l764_764874

theorem solution_inequalities (x : ℝ) :
  (x^2 - 12 * x + 32 > 0) ∧ (x^2 - 13 * x + 22 < 0) → 2 < x ∧ x < 4 :=
by
  intro h
  sorry

end solution_inequalities_l764_764874


namespace rahul_share_is_42_l764_764046

def work_share_rahul (rahul_days : ℕ) (rajesh_days : ℕ) (total_payment : ℕ) : ℕ :=
  let rahul_work_per_day := 1 / rahul_days.toRational
  let rajesh_work_per_day := 1 / rajesh_days.toRational
  let total_work_per_day := rahul_work_per_day + rajesh_work_per_day
  let rahul_share := (rahul_work_per_day / total_work_per_day) * total_payment
  rahul_share.toNat

theorem rahul_share_is_42 (rahul_days : ℕ) (rajesh_days : ℕ) (total_payment : ℕ) :
  rahul_days = 3 → rajesh_days = 2 → total_payment = 105 → work_share_rahul rahul_days rajesh_days total_payment = 42 :=
by
  intros h1 h2 h3
  simp [work_share_rahul]
  rw [h1, h2, h3]
  norm_num
  sorry

end rahul_share_is_42_l764_764046


namespace negation_of_proposition_l764_764082

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x > 0 → (x - 2) / x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ (0 ≤ x ∧ x < 2) := 
sorry

end negation_of_proposition_l764_764082


namespace find_x_l764_764866

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : (∀ (a b c d : ℝ), balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 := 
by
  sorry

end find_x_l764_764866


namespace cycle_with_sum_divisible_by_prime_l764_764397

theorem cycle_with_sum_divisible_by_prime (p : ℕ) (hp : Prime p) (G : CompleteGraph (1000 * p))
  (label : (Sym2 G.Vertices) → ℤ) :
  ∃ (c : Cycle G.Vertices), (∑ e in c.edges, label e) % p = 0 := 
sorry

end cycle_with_sum_divisible_by_prime_l764_764397


namespace area_of_right_triangle_l764_764174

theorem area_of_right_triangle (h : ℝ) 
  (a b : ℝ) 
  (h_a_triple : b = 3 * a)
  (h_hypotenuse : h ^ 2 = a ^ 2 + b ^ 2) : 
  (1 / 2) * a * b = (3 * h ^ 2) / 20 :=
by
  sorry

end area_of_right_triangle_l764_764174


namespace find_point_P_on_parabola_l764_764168

theorem find_point_P_on_parabola (x y : ℝ) : 
  let V := (0 : ℝ, 0 : ℝ) in
  let F := (0 : ℝ, 2 : ℝ) in
  (V.1 = 0 ∧ V.2 = 0) ∧ 
  (F.1 = 0 ∧ F.2 = 2) ∧ 
  (sqrt (x^2 + (y - 2)^2) = 50) ∧ 
  (x > 0) ∧ (y > 0) ∧
  (sqrt (x^2 + (y - F.2)^2) = y + F.2) → 
  (x = 8 * sqrt 6 ∧ y = 48) :=
by
  intros
  -- Add proof here
  sorry

end find_point_P_on_parabola_l764_764168


namespace ursula_change_l764_764861

theorem ursula_change : 
  let hot_dog_cost := 1.50
  let salad_cost := 2.50
  let hot_dogs_count := 5
  let salads_count := 3
  let bill_count := 2
  let bill_value := 10.00
  let total_hot_dog_cost := hot_dogs_count * hot_dog_cost
  let total_salad_cost := salads_count * salad_cost
  let total_purchase_cost := total_hot_dog_cost + total_salad_cost
  let total_money := bill_count * bill_value
  let change_received := total_money - total_purchase_cost
  in change_received = 5.00 :=
by
  sorry

end ursula_change_l764_764861


namespace smallest_value_square_l764_764799

theorem smallest_value_square (z : ℂ) (hz : z.re > 0) (A : ℝ) :
  (A = 24 / 25) →
  abs ((Complex.abs z + 1 / Complex.abs z)^2 - (2 - 14 / 25)) = 0 :=
by
  sorry

end smallest_value_square_l764_764799


namespace one_fourth_of_8_point8_simplified_l764_764993

noncomputable def one_fourth_of (x : ℚ) : ℚ := x / 4

def convert_to_fraction (x : ℚ) : ℚ := 
  let num := 22
  let denom := 10
  num / denom

def simplify_fraction (num denom : ℚ) (gcd : ℚ) : ℚ := 
  (num / gcd) / (denom / gcd)

theorem one_fourth_of_8_point8_simplified : one_fourth_of 8.8 = (11 / 5) := 
by
  have h : one_fourth_of 8.8 = 2.2 := by sorry
  have h_frac : 2.2 = (22 / 10) := by sorry
  have h_simplified : (22 / 10) = (11 / 5) := by sorry
  rw [h, h_frac, h_simplified]
  exact rfl

end one_fourth_of_8_point8_simplified_l764_764993


namespace triangle_isosceles_or_right_l764_764727

theorem triangle_isosceles_or_right {A B C : ℝ} (hA : A > 0) (hB : B > 0) (hC : C > 0) (hTriangle : A + B + C = π) (hSin : sin (2 * A) = sin (2 * B)) :
  (A = B) ∨ (A + B = π / 2) :=
by
  sorry

end triangle_isosceles_or_right_l764_764727


namespace intersect_sets_l764_764027

def A := {x : ℝ | x > -1}
def B := {x : ℝ | x ≤ 5}

theorem intersect_sets : (A ∩ B) = {x : ℝ | -1 < x ∧ x ≤ 5} := 
by 
  sorry

end intersect_sets_l764_764027


namespace Ursula_change_l764_764859

theorem Ursula_change : 
  let cost_hot_dogs := 5 * 1.50
  let cost_salads := 3 * 2.50
  let total_cost := cost_hot_dogs + cost_salads
  let amount_ursula_had := 2 * 10
  let change := amount_ursula_had - total_cost
  change = 5 := 
by
  let cost_hot_dogs := 5 * 1.50
  let cost_salads := 3 * 2.50
  let total_cost := cost_hot_dogs + cost_salads
  let amount_ursula_had := 2 * 10
  let change := amount_ursula_had - total_cost
  have h1 : cost_hot_dogs = 7.50 := sorry
  have h2 : cost_salads = 7.50 := sorry
  have h3 : total_cost = 15.00 := sorry
  have h4 : amount_ursula_had = 20.00 := sorry
  have h5 : change = 5.00 := sorry
  exact h5

end Ursula_change_l764_764859


namespace rectangle_area_inscribed_in_triangle_l764_764047

theorem rectangle_area_inscribed_in_triangle
  (F E G A B C D : Point)
  (h : Collinear [E, G, A, D])
  (h1 : Collinear [E, G, C, D])
  (h2 : Collinear [E, G, E, F])
  (h3 : Altitude F E G A D 12)
  (h4 : LineSegment E G = 15)
  (h5 : LineSegment A B = (1 / 3) * LineSegment A D) :
  AreaOfRectangle A B C D = 10800 / 289 :=
by
  sorry

end rectangle_area_inscribed_in_triangle_l764_764047


namespace min_third_side_of_right_triangle_l764_764345

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end min_third_side_of_right_triangle_l764_764345


namespace location_determined_by_D_l764_764564

theorem location_determined_by_D 
  (A : Prop := "The 2nd row of Rui'an Guangda Cinema")
  (B : Prop := "Hongqiao Road, Rui'an City")
  (C : Prop := "45° northeast")
  (D : Prop := "Longitude 119°E, Latitude 42°N")
  : ∃ location : String, location = "Longitude 119°E, Latitude 42°N" := 
sorry

end location_determined_by_D_l764_764564


namespace perpendicular_line_equation_l764_764178

theorem perpendicular_line_equation (x y : ℝ) :
  (2, -1) ∈ ({ p : ℝ × ℝ | p.1 * 2 + p.2 * 1 - 3 = 0 }) ∧ 
  (∀ p : ℝ × ℝ, (p.1 * 2 + p.2 * (-4) + 5 = 0) → (p.2 * 1 + p.1 * 2 = 0)) :=
sorry

end perpendicular_line_equation_l764_764178


namespace S2_def_limit_S2_S1_l764_764755

noncomputable def S1 (a : ℝ) : ℝ := a * Real.log a - a + 1

noncomputable def S2 (a : ℝ) : ℝ :=
  let b := (a - 1) / Real.log a
  0.5 * (a * Real.log a - a + 1 + (a - 1) * (Real.log (a - 1) - Real.log (Real.log a)))

theorem S2_def (a : ℝ) (ha : a > 1) : 
  S2 a = 0.5 * (a * Real.log a - a + 1 + (a - 1) * (Real.log (a - 1) - Real.log (Real.log a))) :=
sorry

theorem limit_S2_S1 (a : ℝ) (ha : a > 1) : 
  filter.tendsto (λ a, S2 a / S1 a) filter.at_top (nhds (0.5)) :=
sorry

end S2_def_limit_S2_S1_l764_764755


namespace jack_can_measure_2kg_case1_l764_764326

theorem jack_can_measure_2kg_case1 (gold_dust : ℕ) (w1 w2 : ℕ) (target : ℕ) :
  gold_dust = 9000 → w1 = 200 → w2 = 50 → target = 2000 →
  (∃ steps : list ℕ → Prop, steps [gold_dust, w1, w2] ∧ steps [target]) := by
  sorry

end jack_can_measure_2kg_case1_l764_764326


namespace distance_between_points_l764_764599

theorem distance_between_points (x y : ℝ)
  (h₁ : x^2 + y^2 = 29)
  (h₂ : x + y = 11) :
  let p1 := (5 : ℝ, 6 : ℝ),
      p2 := (6 : ℝ, 5 : ℝ),
      d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) in
  d = Real.sqrt 2 :=
by {
  sorry
}

end distance_between_points_l764_764599


namespace max_travel_within_budget_l764_764899

noncomputable def rental_cost_per_day : ℝ := 30
noncomputable def insurance_fee_per_day : ℝ := 10
noncomputable def mileage_cost_per_mile : ℝ := 0.18
noncomputable def budget : ℝ := 75
noncomputable def minimum_required_travel : ℝ := 100

theorem max_travel_within_budget : ∀ (rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel), 
  rental_cost_per_day = 30 → 
  insurance_fee_per_day = 10 → 
  mileage_cost_per_mile = 0.18 → 
  budget = 75 →
  minimum_required_travel = 100 →
  (minimum_required_travel + (budget - rental_cost_per_day - insurance_fee_per_day - mileage_cost_per_mile * minimum_required_travel) / mileage_cost_per_mile) = 194 := 
by
  intros rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end max_travel_within_budget_l764_764899


namespace total_initial_passengers_l764_764152

theorem total_initial_passengers (M W : ℕ) 
  (h1 : W = M / 3) 
  (h2 : M - 24 = W + 12) : 
  M + W = 72 :=
sorry

end total_initial_passengers_l764_764152


namespace triangle_side_length_l764_764287

noncomputable def sin_from_cos (cos_x : ℝ) : ℝ :=
  real.sqrt (1 - cos_x ^ 2)

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ)
  (cos_B : ℝ) (area_ABC : ℝ)
  (h₁ : a = 10)
  (h₂ : cos_B = 4 / 5)
  (h₃ : area_ABC = 42) :
  c = 14 :=
by
  sorry

end triangle_side_length_l764_764287


namespace number_of_pits_less_than_22222_l764_764542

def is_pit (n : ℕ) : Prop :=
  let digits := n.digits 10;
  digits.length = 5 ∧
  digits.take 3 = (digits.take 3).sort (· > ·) ∧
  digits.drop 2 = (digits.drop 2).sort (· < ·)

def is_less_than_22222 (n : ℕ) : Prop := n < 22222

theorem number_of_pits_less_than_22222 : { n // is_pit n ∧ is_less_than_22222 n }.card = 36 :=
by 
  sorry

end number_of_pits_less_than_22222_l764_764542


namespace find_a_l764_764700

noncomputable theory
open_locale classical

structure Point :=
(x : ℝ)
(y : ℝ)

def line_intersects_circle (a : ℝ) : Prop :=
∃ (A B : Point), (A.x + A.y = 1) ∧ (A.x^2 + A.y^2 = a) ∧ (B.x + B.y = 1) ∧ (B.x^2 + B.y^2 = a) ∧ 
(∀ O C : Point, O.x = 0 ∧ O.y = 0 ∧ C.x^2 + C.y^2 = a ∧ (O.x + A.x + B.x = C.x) ∧ (O.y + A.y + B.y = C.y))

theorem find_a (a : ℝ) (h : line_intersects_circle a) : a = 2 :=
sorry

end find_a_l764_764700


namespace one_fourth_of_8_8_is_11_over_5_l764_764991

theorem one_fourth_of_8_8_is_11_over_5 :
  ∀ (a b : ℚ), a = 8.8 → b = 1/4 → b * a = 11/5 :=
by
  assume a b : ℚ,
  assume ha : a = 8.8,
  assume hb : b = 1/4,
  sorry

end one_fourth_of_8_8_is_11_over_5_l764_764991


namespace solution_l764_764818

noncomputable def problem_statement : Prop :=
  ∃ (O : AffineSpace) (P : O),
  let r := 8
  ∃ (A B C D : O),
  ∃ (AC BD : Line O),
  let OP := dist O P
  ∃ (a b : ℕ),
    2 * (sqrt a) = 2 * sqrt 61 ∧
    6 * (sqrt b) = 6 * sqrt 7 ∧
    BD ⊥ AC ∧
    Circumcircle Tangeant (Triangle A B P) (Triangle C D P) ∧
    Circumcircle Tangeant (Triangle A D P) (Triangle B C P) ∧
    (OP = sqrt a - sqrt b) ∧
    (100 * a + b = 103360)

theorem solution : problem_statement := sorry

end solution_l764_764818


namespace correct_option_l764_764871

-- Definitions
def option_A (a : ℕ) : Prop := a^2 * a^3 = a^5
def option_B (a : ℕ) : Prop := a^6 / a^2 = a^3
def option_C (a b : ℕ) : Prop := (a * b^3) ^ 2 = a^2 * b^9
def option_D (a : ℕ) : Prop := 5 * a - 2 * a = 3

-- Theorem statement
theorem correct_option :
  (∃ (a : ℕ), option_A a) ∧
  (∀ (a : ℕ), ¬option_B a) ∧
  (∀ (a b : ℕ), ¬option_C a b) ∧
  (∀ (a : ℕ), ¬option_D a) :=
by
  sorry

end correct_option_l764_764871


namespace proof_problem_l764_764319

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x| > 1}

def B : Set ℝ := {x | (0 : ℝ) < x ∧ x ≤ 2}

def complement_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def intersection (s1 s2 : Set ℝ) : Set ℝ := s1 ∩ s2

theorem proof_problem : (complement_A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
by {
  sorry
}

end proof_problem_l764_764319


namespace sum_of_numbers_base5_reverse_base9_l764_764953

theorem sum_of_numbers_base5_reverse_base9 :
  let valid_numbers := { n : ℕ | n.digits 5 = n.digits 9.reverse ∧ 0 < n } in
  (valid_numbers.sum : ℕ) = 10 :=
by sorry

end sum_of_numbers_base5_reverse_base9_l764_764953


namespace find_equation_of_ellipse_intersection_at_fixed_point_l764_764192

-- Define the relevant conditions and properties
def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x y : ℝ) : Prop := y = sqrt 3

def eccentric (a c : ℝ) : Prop := c / a = 1 / 2

def right_focus (a : ℝ) : ℝ := a / 2

noncomputable def proj_coords (x : ℝ) : ℝ := 4

def fixed_intersect (P : ℚ) : Prop := P = (5 / 2 : ℚ, 0 : ℚ)

-- Formalize the statements as Lean 4 theorems
theorem find_equation_of_ellipse (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : passes_through 0 (sqrt 3))
    (h₄ : eccentric a (right_focus a)) : ellipse x y 2 (sqrt 3) := by
    sorry

theorem intersection_at_fixed_point (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : passes_through 0 (sqrt 3))
    (h₄ : eccentric a (right_focus a)) : fixed_intersect (5 / 2, 0) := by
    sorry

end find_equation_of_ellipse_intersection_at_fixed_point_l764_764192


namespace arithmetic_sequence_minimum_sum_m_l764_764683

theorem arithmetic_sequence_minimum_sum_m
    (a : ℕ → ℤ) (S : ℕ → ℤ)
    (h1 : a 1 = -19)
    (h2 : a 7 - a 4 = 6)
    (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
    (h_sum : ∀ n, ∑ i in finset.range n, a i = S n)
    (h_nonneg : ∀ n, S n ≥ S m) :
  m = 10 :=
by
  sorry

end arithmetic_sequence_minimum_sum_m_l764_764683


namespace neg_P_implies_neg_Q_l764_764656

variable (x : ℝ)

def P : Prop := |2 * x - 3| > 1
def Q : Prop := x^2 - 3 * x + 2 ≥ 0

theorem neg_P_implies_neg_Q : ¬ P → ¬ Q := 
begin
  intro h,
  unfold P at h,
  unfold Q,
  sorry,
end

end neg_P_implies_neg_Q_l764_764656


namespace find_valid_pairs_l764_764524

def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem find_valid_pairs :
  ∀ (a b : ℕ), (0 < a ∧ 0 < b ∧ a ^ 2 * b ^ 2 + 208 = 4 * (lcm a b + gcd a b) ^ 2) ↔ 
  ((a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) ∨ (a = 2 ∧ b = 12) ∨ (a = 12 ∧ b = 2)) :=
by sorry

end find_valid_pairs_l764_764524


namespace middle_dimension_of_crate_l764_764538

theorem middle_dimension_of_crate (middle_dimension : ℝ) : 
    (∀ r : ℝ, r = 5 → ∃ w h l : ℝ, w = 5 ∧ h = 12 ∧ l = middle_dimension ∧
        (diameter = 2 * r ∧ diameter ≤ middle_dimension ∧ h ≥ 12)) → 
    middle_dimension = 10 :=
by
  sorry

end middle_dimension_of_crate_l764_764538


namespace area_of_quadrilateral_ABDF_l764_764554

theorem area_of_quadrilateral_ABDF :
  let AC : ℝ := 40
  let AE : ℝ := 25
  let AB : ℝ := 20
  let BC : ℝ := 20
  let AF : ℝ := 15
  let FE : ℝ := 10
  let A_ACDE := AC * AE
  let A_BCD := (1 / 2) * BC * AE
  let A_EFD := (1 / 2) * FE * AC
  A_ACDE - A_BCD - A_EFD = 550 :=
by
  let AC := 40
  let AE := 25
  let AB := 20
  let BC := 20
  let AF := 15
  let FE := 10
  let A_ACDE := AC * AE
  let A_BCD := (1 / 2) * BC * 25
  let A_EFD := (1 / 2) * FE * 40
  have h : A_ACDE - A_BCD - A_EFD = 550 := by sorry
  exact h

end area_of_quadrilateral_ABDF_l764_764554


namespace tank_filling_time_l764_764179

-- Define the conditions
variables (T X Y Z : ℝ)

-- Conditions given in the problem
def condition1 : Prop := T = 3 * (X + Y)
def condition2 : Prop := T = 6 * (X + Z)
def condition3 : Prop := T = 4.5 * (Y + Z)

-- The goal is to prove the time taken by all three pipes working together is 3.27 hours
def proof_problem : Prop :=
  condition1 T X Y Z ∧
  condition2 T X Y Z ∧
  condition3 T X Y Z →
  108 / (33 : ℝ) = 3.27

-- Provide the statement without the proof
theorem tank_filling_time :
  ∀ (T X Y Z : ℝ),
  condition1 T X Y Z →
  condition2 T X Y Z →
  condition3 T X Y Z →
  108 / 33 = 3.27 :=
begin
  intros T X Y Z h₁ h₂ h₃,
  sorry
end

end tank_filling_time_l764_764179


namespace maximize_area_DEF_l764_764278

-- Define angles
constant α β γ : ℝ

-- Assume they form a triangle
axiom triangle_angles : α + β + γ = π

-- Define cotangent and arctangent functions for the given angles
def ctg (x : ℝ) := 1 / tan x
def arctan (x : ℝ) := Real.arctan x

-- Define the problem to be proved
theorem maximize_area_DEF :
  ∃ φ : ℝ, φ = arctan (ctg α + ctg β + ctg γ) :=
sorry

end maximize_area_DEF_l764_764278


namespace max_chocolate_pieces_10x10_chessboard_l764_764491

-- Define the chessboard and pieces
def Chessboard := Fin 10 × Fin 10
inductive Piece
| bishop : Chessboard -> Piece
| rook : Chessboard -> Piece

def attacks (p1 p2 : Piece) : Bool :=
  match p1, p2 with
  | Piece.bishop (r1, c1), Piece.bishop (r2, c2) =>
    (r1 - c1 = r2 - c2) ∨ (r1 + c1 = r2 + c2)
  | Piece.rook (r1, c1), Piece.rook (r2, c2) =>
    (r1 = r2) ∨ (c1 = c2)
  | Piece.bishop _, Piece.rook _ => false
  | Piece.rook _, Piece.bishop _ => false

def chocolate (pieces : List Piece) (p : Piece) : Bool :=
  ∀ q ∈ pieces, q ≠ p → ¬ attacks q p

noncomputable def max_chocolate_pieces : Nat := 50

theorem max_chocolate_pieces_10x10_chessboard : 
  ∃ pieces : List Piece, 
    List.forall pieces (chocolate pieces) ∧ 
    pieces.length = max_chocolate_pieces :=
sorry

end max_chocolate_pieces_10x10_chessboard_l764_764491


namespace number_of_true_propositions_is_three_l764_764571

-- Definitions based on the conditions provided
variables (α β r : Plane) (l m : Line)

-- Propositions to evaluate
def prop1 := (α ⊥ r) ∧ (β ⊥ r) → (α ∥ β)
def prop2 := (α ∥ r) ∧ (β ∥ r) → (α ∥ β)
def prop3 := (α ⊥ l) ∧ (β ⊥ l) → (α ∥ β)
def prop4 := (skew l m) ∧ (l ∥ α) ∧ (m ∥ α) ∧ (l ∥ β) ∧ (m ∥ β) → (α ∥ β)

-- The theorem statement
theorem number_of_true_propositions_is_three :
  (prop1 → false) ∧ prop2 ∧ prop3 ∧ prop4 :=
sorry

end number_of_true_propositions_is_three_l764_764571


namespace circle_params_l764_764390

noncomputable def center_and_radius (x y : ℝ) : Prop :=
  x^2 + 6 * x + 36 = -y^2 - 8 * y + 45

theorem circle_params :
  ∃ a b r, center_and_radius a b ∧ a = -3 ∧ b = -4 ∧ r = Real.sqrt 34 ∧ a + b + r = -7 + Real.sqrt 34 := 
by
  use -3, -4, Real.sqrt 34
  split
  -- Condition for the circle equation
  sorry
  -- Verify the center coordinates
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  -- Compute a + b + r and verify the value
  rw [add_assoc, add_left_comm (-4), add_comm (-3)]
  exact rfl
  
end circle_params_l764_764390


namespace value_of_k_l764_764255

theorem value_of_k (k : ℝ) : 
  (∃ x y : ℝ, x = 1/3 ∧ y = -8 ∧ -3/4 - 3 * k * x = 7 * y) → k = 55.25 :=
by
  intro h
  sorry

end value_of_k_l764_764255


namespace average_marks_of_all_students_l764_764437

theorem average_marks_of_all_students (n1 n2 a1 a2 : ℕ) (n1_eq : n1 = 12) (a1_eq : a1 = 40) 
  (n2_eq : n2 = 28) (a2_eq : a2 = 60) : 
  ((n1 * a1 + n2 * a2) / (n1 + n2) : ℕ) = 54 := 
by
  sorry

end average_marks_of_all_students_l764_764437


namespace sequence_sum_l764_764403

theorem sequence_sum :
  (∃ (a : ℕ → ℕ), a 1 = 7 ∧ (∀ n, a n + a (n + 1) = 20) ∧ (∑ i in finset.range 50, a (i + 1)) = 500) :=
sorry

end sequence_sum_l764_764403


namespace min_distance_sum_l764_764800

-- Defining relevant points and parabola properties
def parabola (P : ℝ × ℝ) : Prop := P.2 ^ 2 = 4 * P.1
def point_A : ℝ × ℝ := (0, -1)
def line_x_neg1 (P : ℝ × ℝ) : ℝ := abs (P.1 + 1)
def distance (P1 P2 : ℝ × ℝ) : ℝ := real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

-- Problem statement in Lean
theorem min_distance_sum (P : ℝ × ℝ) (h : parabola P) : 
  distance P point_A + line_x_neg1 P = real.sqrt 2 := sorry

end min_distance_sum_l764_764800


namespace parallel_translation_homothety_l764_764095

-- Definitions and statements
variables (A B A' B' : Point)
variable (f : Point → Point)
variable (k : ℝ)

-- Hypotheses/Conditions
hypothesis hf : ∀ A B, f(A') = A' ∧ f(B') = B' → \overrightarrow{A'B'} = k \overrightarrow{AB}

-- Statement for part a)
theorem parallel_translation (h1 : k = 1) : 
  (∀ (A B : Point), \overrightarrow{A'B'} = \overrightarrow{AB}) → 
    (∀ (A B : Point), \overrightarrow{BB'} = \overrightarrow{AA'}) := begin
  sorry
end

-- Statement for part b)
theorem homothety (h2 : k ≠ 1) :
  ∃ O : Point, (∀ (X : Point), \overrightarrow{OX'} = k \overrightarrow{OX}) := begin
  sorry
end

end parallel_translation_homothety_l764_764095


namespace mean_variance_correct_l764_764557

noncomputable def mean_variance_of_scores (scores: list ℝ) : ℝ × ℝ :=
  let sorted_scores := scores.qsort (≤)
  let trimmed_scores := sorted_scores.drop 1 |>.take (sorted_scores.length - 2)
  let mean := trimmed_scores.sum / trimmed_scores.length
  let variance := (trimmed_scores.map (λ x, (x - mean)^2)).sum / trimmed_scores.length
  (mean, variance)

theorem mean_variance_correct :
  mean_variance_of_scores [90, 89, 90, 95, 93, 94, 93] = (92, 2.8) :=
by
  sorry

end mean_variance_correct_l764_764557


namespace seating_arrangement_cond_l764_764886

theorem seating_arrangement_cond {M D G : Type} :
  let initial := [X, Y, M] in
  (∃ X Y, initial = [D, G, M] ∨ initial = [G, D, M]) ∧
  (all_positions_covered : (
    set.univ = set.of_list [{[G, D, M], [M, D, G], [G, M, D]} ∨
                           {[M, D, G], [G, D, M], [G, M, D]} ∨
                           {[G, D, M], [D, G, M], [M, D, G]} ∨
                           {[D, G, M], [M, G, D], [G, D, M]}
  ) ∧ ∃ final_arrangement = [G, M, D] :=
by exact sorry

end seating_arrangement_cond_l764_764886


namespace evaluate_expression_l764_764981

theorem evaluate_expression : 
  (Int.floor (Real.ceil ((11 / 5 : ℚ) ^ 2) + 19 / 3)) = 11 := 
by 
  sorry

end evaluate_expression_l764_764981


namespace segments_of_square_l764_764921

theorem segments_of_square (a b : ℕ) (side : ℕ) (perimeter : ℕ) (group1 group2 : ℕ) 
  (h_side : side = 20) (h_perimeter : perimeter = 4 * side) 
  (h_group1 : group1 = 3) (h_group2 : group2 = 4)
  (h_sum_segments : group1 + group2 = 7)
  (h_segments1 : a = side) (h_segments2 : b = side / group2) :
  let total_length := group1 * a + group2 * b
  in total_length = perimeter :=
by
  intros
  sorry

end segments_of_square_l764_764921


namespace max_sphere_radius_in_cone_l764_764154

theorem max_sphere_radius_in_cone {c : ℝ} : 
  ∃ r : ℝ, 
  (∀ (s1 s2 s3 : sphere) 
    (cone : cone), 
    s1.radius = r ∧ s2.radius = r ∧ s3.radius = r ∧ 
    s1.touches_externally s2 ∧ s2.touches_externally s3 ∧ s1.touches_externally s3 ∧ 
    s1.touches_lateral_surface cone ∧ s2.touches_lateral_surface cone ∧ s3.touches_lateral_surface cone ∧ 
    s1.touches_base cone ∧ s2.touches_base cone ∧ 
    cone.base_radius = 2 ∧ cone.slant_height = c) → 
    r = √3 - 1) :=
sorry

end max_sphere_radius_in_cone_l764_764154


namespace pastries_calculation_l764_764958

theorem pastries_calculation 
    (G : ℕ) (C : ℕ) (P : ℕ) (F : ℕ)
    (hG : G = 30) 
    (hC : C = G - 5)
    (hP : P = G - 5)
    (htotal : C + P + F + G = 97) :
    C - F = 8 ∧ P - F = 8 :=
by
  sorry

end pastries_calculation_l764_764958


namespace seating_arrangement_l764_764567

def num_ways_to_seat (A B C D E F : Type) (chairs : List (Option Type)) : Nat := sorry

theorem seating_arrangement {A B C D E F : Type} :
  ∀ (chairs : List (Option Type)),
    (A ≠ B ∧ A ≠ C ∧ F ≠ B) → num_ways_to_seat A B C D E F chairs = 28 :=
by
  sorry

end seating_arrangement_l764_764567


namespace point_in_fourth_quadrant_l764_764837

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := i / (i + 2)

-- Define the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- Define the x-coordinate of the conjugate of z
def x_coord : ℝ := z_conjugate.re

-- Define the y-coordinate of the conjugate of z
def y_coord : ℝ := z_conjugate.im

-- Define a predicate that checks if a point is in the fourth quadrant
def in_fourth_quadrant (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- The statement of the problem in Lean 4
theorem point_in_fourth_quadrant : in_fourth_quadrant x_coord y_coord :=
by {
  -- The proof is omitted
  sorry
}

end point_in_fourth_quadrant_l764_764837


namespace cos_alpha_minus_beta_cos_alpha_cos_beta_l764_764670

-- Definitions of points A and B, and given distance condition
def point_A (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def point_B (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)
def distance (α β : ℝ) : ℝ := Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2)

-- Problem 1: Given conditions, prove cos(α - β) = 4/5
theorem cos_alpha_minus_beta (α β : ℝ) (hα : α > 0 ∧ α < π/2) (hβ : β > 0 ∧ β < π/2) 
  (h_dist : distance α β = Real.sqrt 10 / 5) : Real.cos (α - β) = 4 / 5 :=
by
  sorry

-- Problem 2: Given tan(α/2) = 1/2, prove cos(α) = 3/5 and cos(β) = 24/25
theorem cos_alpha_cos_beta (α β : ℝ) (hα : α > 0 ∧ α < π/2) (hβ : β > 0 ∧ β < π/2)
  (h_tan : Real.tan (α / 2) = 1 / 2) : Real.cos α = 3 / 5 ∧ Real.cos β = 24 / 25 :=
by
  sorry

end cos_alpha_minus_beta_cos_alpha_cos_beta_l764_764670


namespace no_such_function_in_open_interval_l764_764519

noncomputable def f : ℝ → ℝ := sorry

def finite_union_of_points_and_intervals (f : ℝ → ℝ) (a b : ℝ) : Prop := sorry

theorem no_such_function_in_open_interval :
  ¬ ∃ f : ℝ → ℝ, 
    (∀ x ∈ Ioo (-1 : ℝ) 1, f (f x) = -x)
    ∧ finite_union_of_points_and_intervals f (-1) 1 :=
by
  sorry

end no_such_function_in_open_interval_l764_764519


namespace equifacial_tetrahedron_iff_congruent_faces_l764_764802

variable {A B C D : Point}

-- Conditions: a tetrahedron with vertices A, B, C, D
def tetrahedron (A B C D : Point) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- The assertion: All faces of the tetrahedron have the same area if and only if they are congruent
theorem equifacial_tetrahedron_iff_congruent_faces (h : tetrahedron A B C D) :
  (∀ (x y z : Point), {x, y, z} ⊆ {A, B, C, D} → ∃ a : ℝ, area x y z = a) ↔
  (∀ (x1 y1 z1 x2 y2 z2 : Point), {x1, y1, z1} ⊆ {A, B, C, D} → {x2, y2, z2} ⊆ {A, B, C, D} → (x1 ≠ z1 ∧ y2 ≠ z2) → congruent_faces x1 y1 z1 x2 y2 z2) :=
sorry

end equifacial_tetrahedron_iff_congruent_faces_l764_764802


namespace area_of_hexagon_is_9_sqrt_3_div_4_l764_764369

namespace HexagonArea

noncomputable def hexagon_area : ℝ :=
  let side_length := 1
  let longer_side_length := 2
  let area_large_triangle := Real.sqrt 3 / 4 * (longer_side_length ^ 2)
  let area_small_triangle := Real.sqrt 3 / 4 * (side_length ^ 2)
  2 * area_large_triangle + area_small_triangle

theorem area_of_hexagon_is_9_sqrt_3_div_4 : 
  hexagon_area = 9 * Real.sqrt 3 / 4 :=
sorry

end HexagonArea

end area_of_hexagon_is_9_sqrt_3_div_4_l764_764369


namespace ratio_dog_to_cat_video_l764_764206

-- Conditions as definitions
def cat_video_length : ℕ := 4
def dog_video_length : ℕ
def gorilla_video_length := 2 * (cat_video_length + dog_video_length)
def total_time_watching_videos := cat_video_length + dog_video_length + gorilla_video_length

-- Correct answer
theorem ratio_dog_to_cat_video (D : ℕ)
  (h1 : total_time_watching_videos = 36)
  (h2 : dog_video_length = D) :
  D / cat_video_length = 2 :=
by
  sorry

end ratio_dog_to_cat_video_l764_764206


namespace sum_of_coordinates_of_intersection_l764_764205

def h(x : ℝ) : ℝ := 4.125 - (x + 0.5)^2 / 2

theorem sum_of_coordinates_of_intersection :
  ∃ a b : ℝ, h(a) = h(a + 2) ∧ a + b = 4.125 :=
by
  sorry

end sum_of_coordinates_of_intersection_l764_764205


namespace ab_le_1_e2_l764_764653

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.log x - a * x - b

theorem ab_le_1_e2 {a b : ℝ} (h : 0 < a) (hx : ∃ x : ℝ, 0 < x ∧ f x a b ≥ 0) : a * b ≤ 1 / Real.exp 2 :=
sorry

end ab_le_1_e2_l764_764653


namespace chemistry_textbook_weight_l764_764387

theorem chemistry_textbook_weight (G C : ℝ) (h1 : G = 0.62) (h2 : C = G + 6.5) : C = 7.12 :=
by
  sorry

end chemistry_textbook_weight_l764_764387


namespace find_extreme_values_largest_value_smallest_value_l764_764621

theorem find_extreme_values {x : ℝ} (h : abs (x - 3) = 10) : (x = 13 ∨ x = -7) :=
sorry

theorem largest_value {x : ℝ} (h : abs (x - 3) = 10) : x ≤ 13 :=
by
  have := find_extreme_values h
  cases this
  · rw [this]
    exact le_rfl
  · right
    linarith

theorem smallest_value {x : ℝ} (h : abs (x - 3) = 10) : -7 ≤ x :=
by
  have := find_extreme_values h
  cases this
  · left
    linarith
  · rw [this]
    exact le_rfl

end find_extreme_values_largest_value_smallest_value_l764_764621


namespace ratio_of_A_to_B_l764_764463

variable (people : Type)

noncomputable def numOnlyA : ℕ := 1000
noncomputable def numBothAB : ℕ := 500
noncomputable def numOnlyB : ℕ := 250

def numA := numOnlyA + numBothAB
def numB := numOnlyB + numBothAB

theorem ratio_of_A_to_B : numA / numB = 2 :=
by
  -- Place the actual proof here, which is omitted. 
  sorry

end ratio_of_A_to_B_l764_764463


namespace subset_count_eq_60_l764_764474

theorem subset_count_eq_60 :
  let n := 12 in
  ∃ subsets : list (set (fin n)), 
    (∀ s ∈ subsets, 
      (cardinality s ≥ 3) ∧ 
      (cardinality s ≤ 7) ∧ 
      (adjacent_chairs s)) ∧
    (length subsets = 60) :=
by {
  sorry
}

end subset_count_eq_60_l764_764474


namespace grid_number_matching_cells_l764_764849

theorem grid_number_matching_cells :
  ∃ (N : ℕ), N = 7 ∧
    let rows := 31;
    let cols := 67;
    let method1 (i j : ℕ) := 67 * (i - 1) + j;
    let method2 (i j : ℕ) := 31 * (j - 1) + i;
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs.length = N ∧
    (∀ (i j : ℕ), (i, j) ∈ pairs ↔ 1 ≤ i ∧ i ≤ rows ∧ 1 ≤ j ∧ j ≤ cols ∧ method1 i j = method2 i j) :=
by sorry

end grid_number_matching_cells_l764_764849


namespace tangency_condition_l764_764322

def functions_parallel (a b c : ℝ) (f g: ℝ → ℝ)
       (parallel: ∀ x, f x = a * x + b ∧ g x = a * x + c) := 
  ∀ x, f x = a * x + b ∧ g x = a * x + c

theorem tangency_condition (a b c A : ℝ)
    (h_parallel : a ≠ 0)
    (h_tangency : (∀ x, (a * x + b)^2 = 7 * (a * x + c))) :
  A = 0 ∨ A = -7 :=
sorry

end tangency_condition_l764_764322


namespace terminating_decimal_count_l764_764650

theorem terminating_decimal_count (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 719) :
  (∃ k : ℕ, n = 9 * k) → (set.finite { n | 1 ≤ n ∧ n ≤ 719 ∧ (∃ k : ℕ, n = 9 * k) } → set.finite.to_finset _).card = 79 :=
by sorry

end terminating_decimal_count_l764_764650


namespace second_player_can_ensure_divisibility_by_13_l764_764743

theorem second_player_can_ensure_divisibility_by_13 :
  ∀ (a : Fin 8 → ℤ),
    (∀ i, a i = 1 ∨ a i = -1) →
    (∃ b : Fin 8 → ℤ, 0 ≤ b 0 ∧ b 0 < 2 ∧ (∑ i, (a i) * 8^i) % 13 = 0) :=
begin
  sorry
end

end second_player_can_ensure_divisibility_by_13_l764_764743


namespace contrapositive_geometric_sequence_l764_764442

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (b^2 ≠ a * c) → ¬geometric_sequence a b c :=
by
  intros h
  unfold geometric_sequence
  assumption

end contrapositive_geometric_sequence_l764_764442


namespace total_apples_eaten_l764_764198

-- Define the variables based on the conditions
variable (tuesday_apples : ℕ)
variable (wednesday_apples : ℕ)
variable (thursday_apples : ℕ)
variable (total_apples : ℕ)

-- Define the conditions
def cond1 : Prop := tuesday_apples = 4
def cond2 : Prop := wednesday_apples = 2 * tuesday_apples
def cond3 : Prop := thursday_apples = tuesday_apples / 2

-- Define the total apples
def total : Prop := total_apples = tuesday_apples + wednesday_apples + thursday_apples

-- Prove the equivalence
theorem total_apples_eaten : 
  cond1 → cond2 → cond3 → total_apples = 14 :=
by 
  sorry

end total_apples_eaten_l764_764198


namespace maria_towels_l764_764510

-- Define the initial total towels
def initial_total : ℝ := 124.5 + 67.7

-- Define the towels given to her mother
def towels_given : ℝ := 85.35

-- Define the remaining towels (this is what we need to prove)
def towels_remaining : ℝ := 106.85

-- The theorem that states Maria ended up with the correct number of towels
theorem maria_towels :
  initial_total - towels_given = towels_remaining :=
by
  -- Here we would provide the proof, but we use sorry for now
  sorry

end maria_towels_l764_764510


namespace true_propositions_l764_764422

variables (m n : Line) (α β : Plane)

-- Condition: (m ⊥ α) ∧ (n ⊥ β) ∧ (α ⊥ β)
def condition2 : Prop := m ⊥ α ∧ n ⊥ β ∧ α ⊥ β

-- Proposition (2): m ⊥ n
def proposition2 (h : condition2 m n α β) : Prop := m ⊥ n

-- Condition: (m ⊥ α) ∧ (n ∥ β) ∧ (α ∥ β)
def condition3 : Prop := m ⊥ α ∧ n ∥ β ∧ α ∥ β

-- Proposition (3): m ⊥ n
def proposition3 (h : condition3 m n α β) : Prop := m ⊥ n

-- Theorem: (proposition2 is true) ∧ (proposition3 is true)
theorem true_propositions : proposition2 m n α β ∧ proposition3 m n α β :=
by
  sorry

end true_propositions_l764_764422


namespace right_triangle_exists_l764_764379

theorem right_triangle_exists (a b m : ℝ) (h : m = 1 / 5 * real.sqrt (9 * b^2 - 16 * a^2)) :
  ∃ (a b m c : ℝ), a^2 + b^2 = c^2 ∧ m = a * b / c ∧ b = 2 * a :=
sorry

end right_triangle_exists_l764_764379


namespace student_sequence_count_l764_764733

theorem student_sequence_count (n_students n_sessions : ℕ) (h1 : n_students = 12) (h2 : n_sessions = 5) :
  (n_students ^ n_sessions) = 248832 :=
by
  rw [h1, h2]
  simp
  norm_num
  sorry

end student_sequence_count_l764_764733


namespace cost_prices_max_profit_l764_764071

theorem cost_prices (a b : ℝ) (x : ℝ) (y : ℝ)
    (h1 : a - b = 500)
    (h2 : 40000 / a = 30000 / b)
    (h3 : 0 ≤ x ∧ x ≤ 20)
    (h4 : 2000 * x + 1500 * (20 - x) ≤ 36000) :
    a = 2000 ∧ b = 1500 := sorry

theorem max_profit (x : ℝ) (y : ℝ)
    (h1 : 0 ≤ x ∧ x ≤ 12) :
    y = 200 * x + 6000 ∧ y ≤ 8400 := sorry

end cost_prices_max_profit_l764_764071


namespace figure_perimeter_l764_764954

theorem figure_perimeter (BD BC : ℝ) 
  (h1 : BD = 115) 
  (h2 : BC = 115) 
  (h3 : ∠DBC = 60)
  (h4 : ∠BCD = 60) 
  (AB AE : ℝ) 
  (h5 : AB = 120) 
  (h6 : AE = 120) 
  (DE : ℝ) 
  (h7 : DE = 226) 
  : BD + BC + AB + AE + DE + CD = 696 := 
  by
    sorry

end figure_perimeter_l764_764954


namespace binomial_distribution_probability_l764_764285

noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  nat.choose n k * p^k * (1 - p)^(n - k)

theorem binomial_distribution_probability :
  ∃ (n : ℕ) (p : ℝ), (E : ℝ := n * p) = 6 ∧ (D : ℝ := n * p * (1 - p)) = 3 ∧
  binomial_prob n 1 p = 3 * 2^(-10) :=
by
  sorry

end binomial_distribution_probability_l764_764285


namespace difference_in_profit_percentage_l764_764191

-- Constants
def selling_price1 : ℝ := 350
def selling_price2 : ℝ := 340
def cost_price : ℝ := 200

-- Profit calculations
def profit1 : ℝ := selling_price1 - cost_price
def profit2 : ℝ := selling_price2 - cost_price

-- Profit percentage calculations
def profit_percentage1 : ℝ := (profit1 / cost_price) * 100
def profit_percentage2 : ℝ := (profit2 / cost_price) * 100

-- Statement of the problem: Difference in profit percentage
theorem difference_in_profit_percentage : profit_percentage1 - profit_percentage2 = 5 := by
  sorry

end difference_in_profit_percentage_l764_764191


namespace solution_set_for_inequality_l764_764281

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

noncomputable def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

theorem solution_set_for_inequality
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_decreasing : decreasing_on f (Set.Iio 0))
  (h_f1 : f 1 = 0) :
  {x : ℝ | x^3 * f x > 0} = {x : ℝ | x > 1 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l764_764281


namespace find_x_given_perpendicular_vectors_l764_764706

-- Define the vectors a and b
def vec_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

-- Define the dot product of two 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Given that a and b are perpendicular and defining the value of x
theorem find_x_given_perpendicular_vectors : ∃ x : ℝ, dot_product vec_a (vec_b x) = 0 ∧ x = 14 :=
by
  sorry

end find_x_given_perpendicular_vectors_l764_764706


namespace max_area_of_triangle_ABC_l764_764742

variable {A B C Q : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace Q]

variable {QA QB QC BC : ℝ}

noncomputable def maximum_area_triangle_ABC
  (hQA : QA = 3)
  (hQB : QB = 4)
  (hQC : QC = 5)
  (hBC : BC = 6) : ℝ :=
  18.921

theorem max_area_of_triangle_ABC
  (hQA : QA = 3)
  (hQB : QB = 4)
  (hQC : QC = 5)
  (hBC : BC = 6) : maximum_area_triangle_ABC hQA hQB hQC hBC = 18.921 :=
begin
  sorry
end

end max_area_of_triangle_ABC_l764_764742


namespace sum_of_intersections_l764_764610

noncomputable def quartic : Polynomial ℝ := Polynomial.C 3 * Polynomial.X^4 - Polynomial.C 12 * Polynomial.X^3 + Polynomial.C 10 * Polynomial.X + Polynomial.C 9

theorem sum_of_intersections :
  let roots := Polynomial.roots quartic,
      x_sum := roots.sum,
      y_sum := roots.sum (λ x, (2 * x) / 3 - 2)
  in x_sum = 3 ∧ y_sum = -6 :=
by
  sorry

end sum_of_intersections_l764_764610


namespace isosceles_triangle_area_ratio_l764_764578

-- Definitions and conditions
variables (a alpha beta : ℝ)
-- Define isosceles triangle with sides AB=BC=a and angle ABC = alpha

-- Assume the line passing through vertex A and point D on base BC forming an angle beta with the base BC

-- We need to prove the ratio of the areas
theorem isosceles_triangle_area_ratio (h₁ : 0 < alpha ∧ alpha < π) (h₂ : 0 < beta ∧ beta < π) :
  ∀ (a : ℝ), 
  α = 2 * β  →
  let S_BAD := λ x, a * (cos ((α / 2) + β)) / (2 * sin (α / 2) * sin β)
  in
  ∃ x, S_BAD x = 1 - (cos (α / 2 + β)) / (cos (α / 2 - β)) :=
sorry

end isosceles_triangle_area_ratio_l764_764578


namespace parallelogram_area_example_l764_764113

open Real

noncomputable def parallelogram_area (a b θ : ℝ) : ℝ :=
a * b * sin θ

theorem parallelogram_area_example :
  abs (parallelogram_area 15 20 (35 * (π / 180)) - 172.08) < 1 :=
by
  -- Conditions
  let a := 15
  let b := 20
  let θ := 35 * (π / 180)

  -- Calculate area
  let area := parallelogram_area a b θ

  -- Approximation in the theorem statement
  have h := abs (area - 172.08)
  sorry

end parallelogram_area_example_l764_764113


namespace airplane_altitude_l764_764565

noncomputable theory

-- Define the main problem in Lean
theorem airplane_altitude
  (h : ℝ) -- altitude of the airplane
  (d : ℝ) -- distance between Alice and Bob
  (45_degrees_to_radians : ℝ := real.pi / 4) -- 45 degrees in radians
  (30_degrees_to_radians : ℝ := real.pi / 6) -- 30 degrees in radians
  (ha : d = 15) -- given condition: distance between Alice and Bob
  (a45 : tan 45_degrees_to_radians = 1) -- tan(45°) = 1
  (b30 : tan 30_degrees_to_radians = 1 / real.sqrt 3) -- tan(30°) = 1 / sqrt(3))
  (eq1 : h + h * real.sqrt 3 = d) -- combining the two horizontal distances from Alice and Bob
  : h = 5.5 := -- proving altitude
sorry

end airplane_altitude_l764_764565


namespace correct_statement_l764_764873

-- Define the conditions stated in the question
def condition_A : Prop :=
  ∃ (α β γ : ℝ), (α + β + γ = π) ∧ ((α ≤ π/2) ∨ (α ≥ π/2) ∧ (β ≤ π/2) ∨ (β ≥ π/2) ∧ (γ ≤ π/2) ∨ (γ ≥ π/2))

def condition_B : Prop :=
  ∃ (θ : ℝ), (0 < θ < π/2) ∧ (θ > 0)

def condition_C : Prop :=
  ∃ (θ1 θ2 : ℝ), (π/2 < θ1 < π) ∧ (0 < θ2 < π/2) ∧ (θ1 > θ2)

def condition_D : Prop :=
  ∀ (α : ℝ) (k : ℤ), (2 * k * π - π / 2 < α ∧ α < 2 * k * π) → (3 * π / 2 < α ∧ α < 2 * π)

-- The proposition that only condition D is correct
theorem correct_statement :
  ¬condition_A ∧ ¬condition_B ∧ ¬condition_C ∧ condition_D :=
by
  sorry

end correct_statement_l764_764873


namespace fraction_subtraction_l764_764867

theorem fraction_subtraction (a b : ℚ) (h₁ : a = 15) (h₂ : b = 12) :
    (a / b) - (b / a) = 9 / 20 := 
by 
  rw [h₁, h₂] 
  calc 
    (15 / 12) - (12 / 15) 
        = 5 / 4 - 4 / 5 : by norm_num 
    ... 
        = 25 / 20 - 16 / 20 : by norm_num 
    ... 
        = 9 / 20 : by norm_num

end fraction_subtraction_l764_764867


namespace max_t_lt_3_sum_of_sides_l764_764592

-- Let's define the conditions first
variables {A B C D P A' B' C' D' : Type} 
variables (p q r s : ℝ)

-- Ensure the side lengths are in the required order
axiom h_side_lengths : p ≤ q ∧ q ≤ r ∧ r ≤ s

-- Define the segments from point P
noncomputable def t (AA' BB' CC' DD' : ℝ) : ℝ := AA' + BB' + CC' + DD'

-- State the theorem to be proved
theorem max_t_lt_3_sum_of_sides 
  (h_convex : convex_quadrilateral A B C D) 
  (h_point_inside : interior_point P A B C D)
  (h_AA' : segment_from_point_to_opposite_side P A A')
  (h_BB' : segment_from_point_to_opposite_side P B B')
  (h_CC' : segment_from_point_to_opposite_side P C C')
  (h_DD' : segment_from_point_to_opposite_side P D D')
  (AA' BB' CC' DD' : ℝ) : 
  t AA' BB' CC' DD' < 3 * (p + q + r + s) :=
sorry

end max_t_lt_3_sum_of_sides_l764_764592


namespace series_converges_l764_764598

-- Define the series S
noncomputable def S := ∑' n : ℕ, 3 * (1/3)^n

-- State the conditions
def is_finite (x : ℝ) := ∃ l : ℝ, tendsto (λ n : ℕ, x) atTop (𝓝 l)

-- Mathematical Equivalent Proof Problem
theorem series_converges :
  is_finite S :=
begin
  sorry
end

end series_converges_l764_764598


namespace race_distance_is_214_l764_764734

-- Define the speeds of participants P, Q, and R
def speed_R (v : ℝ) : ℝ := v
def speed_Q (v : ℝ) : ℝ := 1.5 * v
def speed_P (v : ℝ) : ℝ := 1.875 * v

-- Define the distances run by P, Q, and R
def distance_P (d : ℝ) : ℝ := d
def distance_Q (d : ℝ) : ℝ := d - 60
def distance_R (d : ℝ) : ℝ := d - 100

-- Define the times taken by P, Q, and R to finish the race
def time_P (d v : ℝ) : ℝ := distance_P d / speed_P v
def time_Q (d v : ℝ) : ℝ := distance_Q d / speed_Q v
def time_R (d v : ℝ) : ℝ := distance_R d / speed_R v

-- The race ends in a three-way tie
def three_way_tie_condition (d v : ℝ) : Prop :=
  time_P d v = time_Q d v ∧ time_Q d v = time_R d v

-- Prove that the distance P runs, d, is 214 meters
theorem race_distance_is_214 (v : ℝ) : three_way_tie_condition 214 v :=
by
  sorry

end race_distance_is_214_l764_764734


namespace probability_complement_l764_764102

theorem probability_complement (P_A : ℝ) (h : P_A = 0.992) : 1 - P_A = 0.008 := by
  sorry

end probability_complement_l764_764102


namespace value_independent_of_b_value_for_d_zero_l764_764029

theorem value_independent_of_b
  (c b d h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (h1 : x1 = b - d - h)
  (h2 : x2 = b - d)
  (h3 : x3 = b + d)
  (h4 : x4 = b + d + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h * (2 * d + h) :=
by
  sorry

theorem value_for_d_zero
  (c b h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (d : ℝ := 0)
  (h1 : x1 = b - h)
  (h2 : x2 = b)
  (h3 : x3 = b)
  (h4 : x4 = b + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h^2 :=
by
  sorry

end value_independent_of_b_value_for_d_zero_l764_764029


namespace even_sum_probability_l764_764112

-- Definitions for the given problem
def wheel_a_numbers : List ℕ := [1, 1, 2, 2, 3]
def wheel_b_numbers : List ℕ := [2, 3, 4, 4, 5, 5]

-- Define the function to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Probability of an event given the list of possible outcomes
def probability (event : ℕ → Prop) (outcomes : List ℕ) : ℚ :=
  (outcomes.countp event : ℚ) / (outcomes.length : ℚ)

-- Proving the probability that the sum of numbers from both wheels is even
theorem even_sum_probability :
  let prob_even_a := probability is_even wheel_a_numbers in
  let prob_odd_a := probability (λ x => ¬ is_even x) wheel_a_numbers in
  let prob_even_b := probability is_even wheel_b_numbers in
  let prob_odd_b := probability (λ x => ¬ is_even x) wheel_b_numbers in
  let prob_even_sum := prob_even_a * prob_even_b + prob_odd_a * prob_odd_b in
  prob_even_sum = 1 / 2 :=
by
  sorry

end even_sum_probability_l764_764112


namespace tan_sum_identity_l764_764447

theorem tan_sum_identity :
  (∀ (a b : ℝ), tan (a + b) = (tan a + tan b) / (1 - tan a * tan b)) →
  (∀ (x : ℝ), x = (tan 12 + tan 18) / (1 - tan 12 * tan 18)) →
  (∀ (x : ℝ), x = tan 30) →
  (tan 30 = sqrt 3 / 3) →
  (tan 12 + tan 18) / (1 - tan 12 * tan 18) = sqrt 3 / 3 :=
by
  intros H1 H2 H3 H4
  sorry

end tan_sum_identity_l764_764447


namespace simplify_product_l764_764054

theorem simplify_product : 
  (∀ (n: ℕ), n ≥ 1 →
    let term := (∏ (k: ℕ) in (finset.range n).filter (λ k, k > 0), ((3 * k + 6) / (3 * k))) in
    term = (3003 / 3)) → 
  ∑ (k : ℕ) in finset.range 1001, k = 1001 :=
sorry

end simplify_product_l764_764054


namespace number_of_valid_lines_l764_764413

-- Definition of a prime number less than 10
def is_valid_x_intercept (a : ℕ) : Prop :=
  a ∈ {2, 3, 5, 7}

-- Definition of a power of 2
def is_valid_y_intercept (b : ℕ) : Prop :=
  ∃ (k : ℕ), b = 2 ^ k

-- Definition of passing through the point (5, 4)
def passes_through_5_4 (a b : ℕ) : Prop :=
  5 * b + 4 * a = a * b

-- Prove the main statement
theorem number_of_valid_lines : 
  ∃! (a b : ℕ), is_valid_x_intercept a ∧ is_valid_y_intercept b ∧ passes_through_5_4 a b :=
by
  sorry

end number_of_valid_lines_l764_764413


namespace median_room_number_l764_764202

theorem median_room_number (rooms : Finset ℕ) :
  rooms = (Finset.range 30).erase 15 .erase 16 .erase 17 →
  rooms.card = 27 →
  ∃ median, median = 14 := by
  sorry

end median_room_number_l764_764202


namespace bucky_savings_excess_l764_764949

def cost_of_game := 60
def saved_amount := 15
def fish_earnings_weekends (fish : String) : ℕ :=
  match fish with
  | "trout" => 5
  | "bluegill" => 4
  | "bass" => 7
  | "catfish" => 6
  | _ => 0

def fish_earnings_weekdays (fish : String) : ℕ :=
  match fish with
  | "trout" => 10
  | "bluegill" => 8
  | "bass" => 14
  | "catfish" => 12
  | _ => 0

def sunday_fish := 10
def weekday_fish := 3
def weekdays := 2

def sunday_fish_distribution := [
  ("trout", 3),
  ("bluegill", 2),
  ("bass", 4),
  ("catfish", 1)
]

noncomputable def sunday_earnings : ℕ :=
  sunday_fish_distribution.foldl (λ acc (fish, count) =>
    acc + count * fish_earnings_weekends fish) 0

noncomputable def weekday_earnings : ℕ :=
  weekdays * weekday_fish * (
    fish_earnings_weekdays "trout" +
    fish_earnings_weekdays "bluegill" +
    fish_earnings_weekdays "bass")

noncomputable def total_earnings : ℕ :=
  sunday_earnings + weekday_earnings

noncomputable def total_savings : ℕ :=
  total_earnings + saved_amount

theorem bucky_savings_excess :
  total_savings - cost_of_game = 76 :=
by sorry

end bucky_savings_excess_l764_764949


namespace number_of_true_propositions_l764_764568

noncomputable def proposition1 (α β r : Plane) : Prop := 
  (α ⊥ r ∧ β ⊥ r) → (α ∥ β)

noncomputable def proposition2 (α β r : Plane) : Prop := 
  (α ∥ r ∧ β ∥ r) → (α ∥ β)

noncomputable def proposition3 (α β : Plane) (l : Line) : Prop := 
  (α ⊥ l ∧ β ⊥ l) → (α ∥ β)

noncomputable def proposition4 (α β : Plane) (l m : Line) : Prop := 
  (skew l m ∧ l ∥ α ∧ m ∥ α ∧ l ∥ β ∧ m ∥ β) → (α ∥ β)

theorem number_of_true_propositions (α β r : Plane) (l m : Line) :
  (¬proposition1 α β r) ∧ (proposition2 α β r) ∧ (proposition3 α β l) ∧
  (proposition4 α β l m) → 
  3 = countp (λ p: Prop, p = proposition2 α β r ∨ p = proposition3 α β l ∨ p = proposition4 α β l m) [true, true, true, false] := sorry

end number_of_true_propositions_l764_764568


namespace fraction_value_l764_764714

variable (x y : ℝ)

theorem fraction_value (hx : x = 4) (hy : y = -3) : (x - 2 * y) / (x + y) = 10 := by
  sorry

end fraction_value_l764_764714


namespace sum_even_not_divisible_by_10_lt_233_l764_764646

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_not_divisible_by_10 (n : ℕ) : Prop := n % 10 ≠ 0

def sum_even_not_divisible_by_10 (N : ℕ) : ℕ :=
  ∑ n in (Finset.range N).filter (λ n, is_even n ∧ is_not_divisible_by_10 n), n

theorem sum_even_not_divisible_by_10_lt_233 : 
  sum_even_not_divisible_by_10 233 = 10812 := 
by 
  sorry

end sum_even_not_divisible_by_10_lt_233_l764_764646


namespace percent_of_srp_bob_paid_l764_764081

theorem percent_of_srp_bob_paid (SRP MP PriceBobPaid : ℝ) 
  (h1 : MP = 0.60 * SRP)
  (h2 : PriceBobPaid = 0.60 * MP) :
  (PriceBobPaid / SRP) * 100 = 36 := by
  sorry

end percent_of_srp_bob_paid_l764_764081


namespace sine_of_ACB_l764_764737

noncomputable def sin_angle_ACB : ℝ :=
  4 - real.sqrt 2 / 6

theorem sine_of_ACB (A B C L K : ℝ) (h1 : ∃ α, α > π/2 ∧ α < π ∧ α ∈ {angle A C B}) 
  (h2 : dist A B = 14) (h3 : ∃ p, p ∈ {L} ∧ is_equidistant_from_line p (line A C) (line B C)) 
  (h4 : ∃ p, p ∈ interior (line A L) ∧ is_equidistant p {A} {B})
  (h5 : dist K L = 1) (h6 : ∠ A C B = 45 * (π / 180)) :
  sin (angle A C B) = sin_angle_ACB := 
  sorry 

end sine_of_ACB_l764_764737


namespace dacid_weighted_average_l764_764602

def weighted_average_score
  (marks: List ℕ)
  (weights: List ℕ) :=
  (List.sum (List.zipWith (· * ·) marks weights): ℝ) / (List.sum weights: ℝ)

theorem dacid_weighted_average :
  weighted_average_score
    [51, 65, 82, 67, 85, 63, 78, 90, 72, 68]
    [2, 3, 2, 1, 1, 1, 1, 3, 1, 2] = 72.47 :=
by
  sorry

end dacid_weighted_average_l764_764602


namespace equal_cardinality_of_Sk_l764_764187

theorem equal_cardinality_of_Sk (n : ℕ) (h_pos : n > 0) :
  ∀ k : ℕ, k < 2 * n - 1 →
  card { p : {X : Finset ℕ // X.card = n ∧ (n - X.sum) % (2 * n - 1) = k}} =
  card { p : {X : Finset ℕ // X.card = n ∧ (n - X.sum) % (2 * n - 1) = 0}} :=
by
  sorry

end equal_cardinality_of_Sk_l764_764187


namespace train_length_proof_l764_764927

-- Given conditions and definitions
def initial_speed_kmh : ℝ := 60 -- initial speed in km/hr
def acceleration_kmh2 : ℝ := 4 -- acceleration in km/hr²
def deceleration_kmh2 : ℝ := 3.5 -- deceleration due to slope in km/hr²
def time_secs : ℕ := 3 -- time to cross the pole in seconds

-- Convert km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * 1000 / 3600

-- Convert km/hr² to m/s²
def kmh2_to_ms2 (a : ℝ) : ℝ := a * 1000 / (3600 * 3600)

-- Calculate the net acceleration in m/s²
def net_acceleration : ℝ := kmh2_to_ms2 (acceleration_kmh2 - deceleration_kmh2)

-- Convert initial speed to m/s
def initial_speed_ms : ℝ := kmh_to_ms initial_speed_kmh

-- Calculate the final velocity in m/s
def final_velocity : ℝ := initial_speed_ms + net_acceleration * time_secs

-- Calculate the length of the train using uniformly accelerated motion formula
def train_length (u : ℝ) (a : ℝ) (t : ℝ) : ℝ :=
  u * t + (1 / 2) * a * t^2

-- The target length based on the given conditions
def expected_length : ℝ := 50.17

-- Proof statement
theorem train_length_proof : train_length initial_speed_ms net_acceleration time_secs = expected_length :=
by
  sorry

end train_length_proof_l764_764927


namespace anne_total_bottle_caps_l764_764933

/-- 
Anne initially has 10 bottle caps 
and then finds another 5 bottle caps.
-/
def anne_initial_bottle_caps : ℕ := 10
def anne_found_bottle_caps : ℕ := 5

/-- 
Prove that the total number of bottle caps
Anne ends with is equal to 15.
-/
theorem anne_total_bottle_caps : 
  anne_initial_bottle_caps + anne_found_bottle_caps = 15 :=
by 
  sorry

end anne_total_bottle_caps_l764_764933


namespace probability_of_green_tile_l764_764897

theorem probability_of_green_tile : 
  (let tiles := filter (λ n, n % 7 = 3) (finset.range 101) in
    (tiles.card / 100) = (7 / 50)) :=
by {
  sorry
}

end probability_of_green_tile_l764_764897


namespace expand_product_l764_764983

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := 
by
  sorry

end expand_product_l764_764983


namespace fraction_sum_l764_764952

theorem fraction_sum : (1 / 3 : ℚ) + (5 / 9 : ℚ) = (8 / 9 : ℚ) :=
by
  sorry

end fraction_sum_l764_764952


namespace mul_composition_l764_764839

theorem mul_composition (a b c d : ℕ) (h₁ : a * b = 2021) (h₂ : a = 43) (h₃ : b = 47) (h₄ : c = a + 10) (h₅ : d = b + 10) :
  c * d = 3021 := by
  have : c = 53 := by
    rw [h₂, h₄]
    rfl
  have : d = 57 := by
    rw [h₃, h₅]
    rfl
  rw [this, this]
  sorry

end mul_composition_l764_764839


namespace solve_equation_for_x_l764_764063

theorem solve_equation_for_x : 
  {x : ℝ | (Real.root 4 (61 - 3 * x) + Real.root 4 (17 + 3 * x) = 6)} = {7, -23} :=
sorry

end solve_equation_for_x_l764_764063


namespace one_fourth_of_8_8_is_11_over_5_l764_764990

theorem one_fourth_of_8_8_is_11_over_5 :
  ∀ (a b : ℚ), a = 8.8 → b = 1/4 → b * a = 11/5 :=
by
  assume a b : ℚ,
  assume ha : a = 8.8,
  assume hb : b = 1/4,
  sorry

end one_fourth_of_8_8_is_11_over_5_l764_764990


namespace max_yellow_apples_max_total_apples_l764_764139

theorem max_yellow_apples (total_apples green_apples yellow_apples red_apples : ℕ)
  (h_total: total_apples = 35)
  (h_green: green_apples = 8)
  (h_yellow: yellow_apples = 11)
  (h_red: red_apples = 16)
  (condition: (∀ g y r : ℕ, g < y ∧ y < r → true)) :
  ∃ y_max : ℕ, y_max = 11 := 
begin
  use 11,
  sorry
end

theorem max_total_apples (total_apples green_apples yellow_apples red_apples : ℕ)
  (h_total: total_apples = 35)
  (h_green: green_apples = 8)
  (h_yellow: yellow_apples = 11)
  (h_red: red_apples = 16)
  (condition: (∀ g y r : ℕ, g < y ∧ y < r → true)) :
  ∃ t_max : ℕ, t_max = 33 := 
begin
  use 33,
  sorry
end

end max_yellow_apples_max_total_apples_l764_764139


namespace part1_part2_l764_764677

noncomputable def geometric_seq (a : ℝ) (n : ℕ) : ℝ := 
  if n = 0 then a - 2 else
  if n = 1 then 4 else 
  if n = 2 then 2 * a else 
  sorry -- sequence definition for other n

noncomputable def Sn (a : ℝ) (k : ℕ) : ℝ := 
  ∑ i in (range k), geometric_seq a i

noncomputable def bn (a : ℝ) (n : ℕ) : ℝ :=
  (2 * (n + 1) - 1) * geometric_seq a n

noncomputable def Tn (a : ℝ) (n : ℕ) : ℝ :=
  ∑ i in (range n), bn a i

theorem part1 (a : ℝ) (k : ℕ) (h : Sn a k = 62) : a = 4 ∧ k = 5 :=
  sorry

theorem part2 (n : ℕ) : Tn 4 n = (2 * n - 3) * 2^(n + 1) + 6 :=
  sorry

end part1_part2_l764_764677


namespace total_apples_eaten_l764_764196

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end total_apples_eaten_l764_764196


namespace average_age_of_three_l764_764798

theorem average_age_of_three (Kimiko_age : ℕ) (Omi_age : ℕ) (Arlette_age : ℕ) 
  (h1 : Omi_age = 2 * Kimiko_age) 
  (h2 : Arlette_age = (3 * Kimiko_age) / 4) 
  (h3 : Kimiko_age = 28) : 
  (Kimiko_age + Omi_age + Arlette_age) / 3 = 35 := 
  by sorry

end average_age_of_three_l764_764798


namespace gcd_a_b_l764_764772

def a : ℕ := 333333333
def b : ℕ := 555555555

theorem gcd_a_b : Nat.gcd a b = 111111111 := 
by
  sorry

end gcd_a_b_l764_764772


namespace one_eighth_of_two_power_36_equals_two_power_x_l764_764337

theorem one_eighth_of_two_power_36_equals_two_power_x (x : ℕ) :
  (1 / 8) * (2 : ℝ) ^ 36 = (2 : ℝ) ^ x → x = 33 :=
by
  intro h
  sorry

end one_eighth_of_two_power_36_equals_two_power_x_l764_764337


namespace one_eighth_of_power_l764_764339

theorem one_eighth_of_power (x : ℕ) (h : (1 / 8) * (2 ^ 36) = 2 ^ x) : x = 33 :=
by 
  -- Proof steps are not needed, so we leave it as sorry.
  sorry

end one_eighth_of_power_l764_764339


namespace circle_intersection_problem_l764_764851

-- Definitions of the conditions
variables {k1 k2 k3 : Circle} {O A B C A' B' C' : Point}

-- Given conditions
def condition_1 (k1 k2 k3 : Circle) (O : Point) : Prop :=
intersect_circle_at_two_points k1 k2 k3 O

def condition_2 (k2 k3 : Circle) (O A : Point) : Prop :=
second_intersection_point k2 k3 O A

def condition_3 (k1 k3 : Circle) (O B : Point) : Prop :=
second_intersection_point k1 k3 O B

def condition_4 (k1 k2 : Circle) (O C : Point) : Prop :=
second_intersection_point k1 k2 O C

def condition_5 (O : Point) (A B C : Point) : Prop :=
point_inside_triangle O A B C

def condition_6 (AO BO CO : Line) (k1 k2 k3 : Circle) (A' B' C' : Point) : Prop :=
lines_intersect_circles A O k1 A' B O k2 B' C O k3 C'

-- Final statement to prove
theorem circle_intersection_problem
  (k1 k2 k3 : Circle) (O A B C A' B' C' : Point)
  (AO BO CO : Line) 
  (h1 : condition_1 k1 k2 k3 O)
  (h2 : condition_2 k2 k3 O A)
  (h3 : condition_3 k1 k3 O B)
  (h4 : condition_4 k1 k2 O C)
  (h5 : condition_5 O A B C)
  (h6 : condition_6 AO BO CO k1 k2 k3 A' B' C') :
  \frac{|AO|}{|AA'|} + \frac{|BO|}{|BB'|} + \frac{|CO|}{|CC'|} = 1 := 
sorry

end circle_intersection_problem_l764_764851


namespace investor_more_money_in_A_l764_764576

noncomputable def investment_difference 
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ) :
  ℝ :=
investment_A * (1 + yield_A) - investment_B * (1 + yield_B)

theorem investor_more_money_in_A
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ)
  (hA : investment_A = 300)
  (hB : investment_B = 200)
  (hYA : yield_A = 0.3)
  (hYB : yield_B = 0.5)
  :
  investment_difference investment_A investment_B yield_A yield_B = 90 := 
by
  sorry

end investor_more_money_in_A_l764_764576


namespace range_of_b_sub_a_over_a_l764_764672

theorem range_of_b_sub_a_over_a (a b : ℝ) (h : ∀ x : ℝ, f x = exp x - a * x + b) (hf : ∀ x : ℝ, f x ≥ 1) :
    ∃ r, r = (b - a) / a ∧ r ∈ Set.Ici (-1) :=
by
  sorry

end range_of_b_sub_a_over_a_l764_764672


namespace product_sequence_value_l764_764955

theorem product_sequence_value (n : ℕ) : (List.prod (List.map (λ k, 2^(2^k) + 1) (List.range (n + 1)))) = 2^(4*n) - 1 :=
  sorry

end product_sequence_value_l764_764955


namespace water_in_conical_tank_l764_764097

theorem water_in_conical_tank :
  let radius := 8
  let height := 64
  let tank_volume := (1 / 3) * Real.pi * (radius^2) * height
  let water_volume := 0.4 * tank_volume
  let x := (water_volume / tank_volume) ^ (1 / 3)
  let water_height := height * x
  let a := 64
  let b := 2
  a + b = 66 :=
by
  let radius := 8
  let height := 64
  let tank_volume := (1 / 3) * Real.pi * (radius^2) * height
  let water_volume := 0.4 * tank_volume
  let x := (water_volume / tank_volume) ^ (1 / 3)
  let water_height := height * x
  let a := 64
  let b := 2
  have h : water_height = a * Real.cbrt (b / 1) := sorry
  exact Eq.trans (by rw [a, b]; refl) h

end water_in_conical_tank_l764_764097


namespace outlet_pipe_emptying_time_l764_764490

theorem outlet_pipe_emptying_time :
  let rate1 := 1 / 18
  let rate2 := 1 / 20
  let fill_time := 0.08333333333333333
  ∃ x : ℝ, (rate1 + rate2 - 1 / x = 1 / fill_time) → x = 45 :=
by
  intro rate1 rate2 fill_time
  use 45
  intro h
  sorry

end outlet_pipe_emptying_time_l764_764490


namespace expansion_gameplay_hours_l764_764005

theorem expansion_gameplay_hours :
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  expansion_hours = 30 :=
by
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  show expansion_hours = 30
  sorry

end expansion_gameplay_hours_l764_764005


namespace train_speed_l764_764924

theorem train_speed (L : ℝ) (t : ℝ) (m_per_s_to_km_per_hr : ℝ) (hL : L = 100) (ht : t = 4.99960003199744) (hc : m_per_s_to_km_per_hr = 3.6) :
  (L / t * m_per_s_to_km_per_hr) ≈ 72.01 :=
by
  sorry

end train_speed_l764_764924


namespace tim_balloons_l764_764221

theorem tim_balloons (dan_balloons : ℕ) (h : dan_balloons = 29) 
    (times_more : ℕ) (h' : times_more = 7) : 
    let tim_balloons := times_more * dan_balloons in 
    tim_balloons = 203 := 
by 
  -- Define the number of balloons Tim has 
  let tim_balloons := times_more * dan_balloons 
  -- Substitute the conditions
  rw [h, h'], 
  -- Calculate the number of balloons
  simp,
  -- This is the result 
  exact rfl

end tim_balloons_l764_764221


namespace map_distance_representation_l764_764412

-- Define the conditions and the question as a Lean statement
theorem map_distance_representation :
  (∀ (length_cm : ℕ), (length_cm : ℕ) = 23 → (length_cm * 50 / 10 : ℕ) = 115) :=
by
  sorry

end map_distance_representation_l764_764412


namespace minimum_eccentricities_l764_764678

-- Definitions
variables (F1 F2 P : ℝ) (e1 e2 : ℝ)

-- Conditions
def sharedFoci := (F1, F2)

def intersectionPoint (P : ℝ) := true

def angleCondition : Prop := ∠ (F1, P, F2) = 60

-- Statement to prove
theorem minimum_eccentricities (hshared : sharedFoci) (hpoint : intersectionPoint P) (hangle : angleCondition) :
  ∃ (e1 e2 : ℝ), e1^2 + e2^2 = 1 + sqrt 3 / 2 :=
by sorry

end minimum_eccentricities_l764_764678


namespace calculate_tank_length_l764_764555

noncomputable def tank_length (w h cost_per_sqft total_cost : ℝ) : ℝ :=
  let A := total_cost / cost_per_sqft in
  let l := (A - 2 * w * h) / (2 * w + 2 * h) in
  l

theorem calculate_tank_length :
  (tank_length 6 2 20 1440) = 3 :=
by
  sorry

end calculate_tank_length_l764_764555


namespace find_z_l764_764317

open Complex

theorem find_z (z : ℂ) (i : ℂ) (H_imaginary_unit : i * i = -1) 
  (H_M : ∃ z : ℂ, M = {1, 2, z * i}) 
  (H_N : N = {3, 4}) 
  (H_inter : M ∩ N = {4}) : 
  z = -4 * i := 
by
  sorry

end find_z_l764_764317


namespace original_acid_concentration_l764_764153

theorem original_acid_concentration (P : ℝ) (h1 : 0.5 * P + 0.5 * 20 = 35) : P = 50 :=
by
  sorry

end original_acid_concentration_l764_764153


namespace balance_diff_proof_l764_764959

noncomputable def A_C : ℝ :=
  15000 * (1 + 0.06 / 2)^(2 * 10)

noncomputable def A_D : ℝ :=
  15000 * (1 + 0.08 * 10)

def balance_difference : ℝ :=
  A_C - A_D

theorem balance_diff_proof : balance_difference = 92 := by
  sorry

end balance_diff_proof_l764_764959


namespace product_lmn_eq_one_l764_764782

noncomputable def distinct_complex_numbers : Prop := sorry
noncomputable def distinct_non_zero_constants : Prop := sorry

theorem product_lmn_eq_one
  (p q r : ℂ)
  (l m n : ℂ)
  (h1 : distinct_complex_numbers p q r)
  (h2 : distinct_non_zero_constants l m n)
  (h3 : l = p / (1 - q))
  (h4 : m = q / (1 - r))
  (h5 : n = r / (1 - p)) :
  l * m * n = 1 := 
sorry

end product_lmn_eq_one_l764_764782


namespace unique_valid_quintuple_l764_764640

theorem unique_valid_quintuple :
  ∃! (a b c d e : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
    a^2 + b^2 + c^3 + d^3 + e^3 = 5 ∧
    (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25 :=
sorry

end unique_valid_quintuple_l764_764640


namespace f_ordering_l764_764269

open Real

variables {f : ℝ → ℝ}

-- Conditions
def periodic (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x + 2) = f(x)
def strictly_decreasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f x1 > f x2
def f_shift_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x + 1) = f (x + 1)

theorem f_ordering
  (hf1 : periodic f)
  (hf2 : strictly_decreasing_on_unit_interval f)
  (hf3 : f_shift_even f) :
  f 5.5 < f 7.8 ∧ f 7.8 < f (-2) :=
sorry

end f_ordering_l764_764269


namespace largest_divisor_of_n_plus_8_dividing_n_cube_plus_64_l764_764975

theorem largest_divisor_of_n_plus_8_dividing_n_cube_plus_64 :
  ∃ n : ℕ, (∀ m : ℕ, (m + 8 ∣ m^3 + 64) → m ≤ 440) ∧ (440 + 8 ∣ 440^3 + 64) :=
begin
  sorry
end

end largest_divisor_of_n_plus_8_dividing_n_cube_plus_64_l764_764975


namespace find_certain_number_multiplied_by_24_l764_764815

-- Define the conditions
theorem find_certain_number_multiplied_by_24 :
  (∃ x : ℤ, 37 - x = 24) →
  ∀ x : ℤ, (37 - x = 24) → (x * 24 = 312) :=
by
  intros h x hx
  -- Here we will have the proof using the assumption and the theorem.
  sorry

end find_certain_number_multiplied_by_24_l764_764815


namespace sum_of_squares_of_special_numbers_l764_764647

theorem sum_of_squares_of_special_numbers :
  let is_special (n : ℕ) := (7 ≤ n) ∧ (n ≤ 49) ∧ (n % 6 = 0) ∧ (n % 5 = 3) ∧ (Nat.Prime n) in
  (∑ n in Finset.filter is_special (Finset.range 50), n * n) = 0 :=
by
  sorry

end sum_of_squares_of_special_numbers_l764_764647


namespace no_geometric_progression_for_1_2_5_l764_764088

theorem no_geometric_progression_for_1_2_5 (m n p : ℕ) (u : ℕ → ℝ) (q : ℝ) 
(h1 : u m = 1) (h2 : u n = 2) (h3 : u p = 5) 
(geom_prog : ∀ k : ℕ, u k = u 1 * q ^ (k - 1)) : 
  false :=
by 
  intros,
  have h4 : 1 = u 1 * q ^ (m - 1), from h1.symm ▸ geom_prog m,
  have h5 : 2 = u 1 * q ^ (n - 1), from h2.symm ▸ geom_prog n,
  have h6 : 5 = u 1 * q ^ (p - 1), from h3.symm ▸ geom_prog p,
  have h7 : 2 = q ^ (n - m), from (h5 / h4).symm,
  have h8 : 5 = q ^ (p - m), from (h6 / h4).symm,
  have h9 : 2 ^ (p - m) = 5 ^ (n - m), from sorry, -- Derived from raising to appropriate powers
  have h10 : 2 ≠ 5, from sorry, -- Because 2 is always even and 5 is always odd
  contradiction

end no_geometric_progression_for_1_2_5_l764_764088


namespace reduction_when_fifth_runner_twice_as_fast_l764_764248

theorem reduction_when_fifth_runner_twice_as_fast (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h_T1 : (T1 / 2 + T2 + T3 + T4 + T5) = 0.95 * T)
  (h_T2 : (T1 + T2 / 2 + T3 + T4 + T5) = 0.90 * T)
  (h_T3 : (T1 + T2 + T3 / 2 + T4 + T5) = 0.88 * T)
  (h_T4 : (T1 + T2 + T3 + T4 / 2 + T5) = 0.85 * T)
  : (T1 + T2 + T3 + T4 + T5 / 2) = 0.92 * T := 
sorry

end reduction_when_fifth_runner_twice_as_fast_l764_764248


namespace min_value_fraction_expr_l764_764639

theorem min_value_fraction_expr : ∀ (x : ℝ), x > 0 → (4 + x) * (1 + x) / x ≥ 9 :=
by
  sorry

end min_value_fraction_expr_l764_764639


namespace calculate_selling_price_l764_764918

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 20
noncomputable def profit_percent : ℝ := 22.448979591836732

noncomputable def total_cost : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := (profit_percent / 100) * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem calculate_selling_price : selling_price = 300 := by
  sorry

end calculate_selling_price_l764_764918


namespace SarahCansYesterday_l764_764875

variable (S : ℕ)
variable (LaraYesterday : ℕ := S + 30)
variable (SarahToday : ℕ := 40)
variable (LaraToday : ℕ := 70)
variable (YesterdayTotal : ℕ := LaraYesterday + S)
variable (TodayTotal : ℕ := SarahToday + LaraToday)

theorem SarahCansYesterday : 
  TodayTotal + 20 = YesterdayTotal -> 
  S = 50 :=
by
  sorry

end SarahCansYesterday_l764_764875


namespace total_apples_eaten_l764_764197

-- Define the variables based on the conditions
variable (tuesday_apples : ℕ)
variable (wednesday_apples : ℕ)
variable (thursday_apples : ℕ)
variable (total_apples : ℕ)

-- Define the conditions
def cond1 : Prop := tuesday_apples = 4
def cond2 : Prop := wednesday_apples = 2 * tuesday_apples
def cond3 : Prop := thursday_apples = tuesday_apples / 2

-- Define the total apples
def total : Prop := total_apples = tuesday_apples + wednesday_apples + thursday_apples

-- Prove the equivalence
theorem total_apples_eaten : 
  cond1 → cond2 → cond3 → total_apples = 14 :=
by 
  sorry

end total_apples_eaten_l764_764197


namespace sum_of_zeros_gt_two_l764_764305

noncomputable def f (a x : ℝ) := 2 * a * Real.log x + x ^ 2 - 2 * (a + 1) * x

theorem sum_of_zeros_gt_two (a x1 x2 : ℝ) (h_a : -0.5 < a ∧ a < 0)
  (h_fx_zeros : f a x1 = 0 ∧ f a x2 = 0) (h_x_order : x1 < x2) : x1 + x2 > 2 := 
sorry

end sum_of_zeros_gt_two_l764_764305


namespace tan_periodic_n_solution_l764_764635

open Real

theorem tan_periodic_n_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ tan (n * (π / 180)) = tan (1540 * (π / 180)) ∧ n = 40 :=
by
  sorry

end tan_periodic_n_solution_l764_764635


namespace sum_of_T_l764_764763

def is_repeating_decimal (x : ℝ) :=
  ∃ (a b c d : ℕ), (∀ i j, {i, j} ⊆ {a, b, c, d} → i ≠ j) ∧
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : set ℝ := { x | is_repeating_decimal x }

theorem sum_of_T : ∑ x in T, x = 280 := sorry

end sum_of_T_l764_764763


namespace g_inv_undefined_at_1_l764_764697

noncomputable def g (x : ℝ) : ℝ := (x - 3) / (x - 5)

noncomputable def g_inv (x : ℝ) : ℝ := (5 * x - 3) / (x - 1)

theorem g_inv_undefined_at_1 : ∀ x : ℝ, (g_inv x) = g_inv 1 → x = 1 :=
by
  intro x h
  sorry

end g_inv_undefined_at_1_l764_764697


namespace Ryan_time_to_entrance_l764_764808

-- Define the conditions given in the problem
def rate (d t : ℕ) : ℕ := d / t
def distance_in_yards (y : ℕ) : ℕ := y * 3

-- Given conditions
def Ryan_rate := rate 80 20
def remaining_distance := distance_in_yards 100

-- The statement we want to prove
theorem Ryan_time_to_entrance : (remaining_distance / Ryan_rate) = 75 :=
 by
   simp [Ryan_rate, remaining_distance, rate, distance_in_yards] 
   sorry

end Ryan_time_to_entrance_l764_764808


namespace sum_of_repeating_decimals_l764_764768

def repeating_decimal_sum := 
  let T := {x : ℝ | ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999}
  ∑ x in T, x = 908.208208208208

theorem sum_of_repeating_decimals :
  repeating_decimal_sum := sorry

end sum_of_repeating_decimals_l764_764768


namespace inequality_sum_l764_764816

theorem inequality_sum {n : ℕ} (a : Fin n → ℝ) (b : Fin n → ℝ) 
  (h_pos : ∀ i, a i > 0) (h_n : n > 1)
  (h_sum_a : ∑ i, a i = 1)
  (h_b : ∀ i, b i = (a i)^2 / ∑ j, (a j)^2) :
  (∑ i, a i / (1 - a i)) ≤ (∑ i, b i / (1 - b i)) :=
sorry

end inequality_sum_l764_764816


namespace min_distance_l764_764674

open Complex

theorem min_distance (z : ℂ) (h : abs (z + 2 - 2 * I) = 1) : ∃ w : ℂ, abs (w - 2 - 2 * I) = 3 :=
begin
  sorry
end

end min_distance_l764_764674


namespace correct_propositions_count_l764_764301

theorem correct_propositions_count :
  (∃ (α β : plane) (m n l : line) (A : point),
   m ⊆ α ∧ l ∩ α = {A} ∧ A ∉ m ∧
   l.parallel_to α ∧ m.parallel_to α ∧
   n ⊥ l ∧ n ⊥ m ∧ n ⊥ α ∧
   α.parallel_to β ∧ 
   l ⊆ α ∧ m ⊆ α ∧ l ∩ m = {A} ∧ 
   l.parallel_to β ∧ m.parallel_to β ∧
   l ⊥ α ∧ l ⊥ n ∧
   (∃ (skew_lines: l.skew_with m), true) ∧
   (∃ (parallel_planes: α.parallel_to β), true)) →
  (∃ (num_correct : ℕ), num_correct = 3) :=
sorry

end correct_propositions_count_l764_764301


namespace jodie_walking_speed_l764_764173

theorem jodie_walking_speed :
  let pool_perimeter := 2 * (50 + 30)
  let walkway_perimeter := 2 * (58 + 38)
  let slippery_distance := 2 * 25
  let time_difference := 80
  let pool_time (s : ℝ) := pool_perimeter / s + slippery_distance / (s / 2)
  let walkway_time (s : ℝ) := walkway_perimeter / s + slippery_distance / (s / 2)
  ∃ s : ℝ, (walkway_time s - pool_time s = 80) ∧ s = 32 / 5 :=
begin
  sorry
end

end jodie_walking_speed_l764_764173


namespace campers_rowing_afternoon_l764_764146

theorem campers_rowing_afternoon (morning_rowing morning_hiking total : ℕ) 
  (h1 : morning_rowing = 41) 
  (h2 : morning_hiking = 4) 
  (h3 : total = 71) : 
  total - (morning_rowing + morning_hiking) = 26 :=
by
  sorry

end campers_rowing_afternoon_l764_764146


namespace smallest_positive_n_l764_764120

theorem smallest_positive_n (n : ℕ) (h1 : 0 < n) (h2 : gcd (8 * n - 3) (6 * n + 4) > 1) : n = 1 :=
sorry

end smallest_positive_n_l764_764120


namespace find_lines_intercept_length_minimum_distance_l764_764888

-- Define the curve and the circle
def curve (m : ℝ) (x y : ℝ) := 4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y = 20 - 24 * m^2

def circle (x y : ℝ) := x^2 + y^2 - 10 * x + 4 * real.sqrt 6 * y = -30

-- Part 1
theorem find_lines_intercept_length (m : ℝ) :
  ∃ b : ℝ, (b = 2 ∨ b = -2) ∧ (∀ (x₁ x₂ : ℝ), y = 2 * x + b) ∧ (length_intercepted_segment x₁ x₂ = 5 * real.sqrt 5 / 3) := sorry

-- Part 2
theorem minimum_distance (m : ℝ) :
  ∀ (P Q : ℝ × ℝ),
    curve m P.1 P.2 →
    circle Q.1 Q.2 →
    ∃ (d : ℝ), (d = 2 * real.sqrt 5 - 1) :=
sorry

end find_lines_intercept_length_minimum_distance_l764_764888


namespace students_selected_milk_l764_764947

theorem students_selected_milk
    (total_students : ℕ)
    (students_soda students_milk students_juice : ℕ)
    (soda_percentage : ℚ)
    (milk_percentage : ℚ)
    (juice_percentage : ℚ)
    (h1 : soda_percentage = 0.7)
    (h2 : milk_percentage = 0.2)
    (h3 : juice_percentage = 0.1)
    (h4 : students_soda = 84)
    (h5 : total_students = students_soda / soda_percentage)
    : students_milk = total_students * milk_percentage :=
by
    sorry

end students_selected_milk_l764_764947


namespace local_minimum_at_2_l764_764243

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

def f' (x : ℝ) : ℝ := 3 * x^2 - 12

theorem local_minimum_at_2 :
  (∀ x : ℝ, -2 < x ∧ x < 2 → f' x < 0) →
  (∀ x : ℝ, x > 2 → f' x > 0) →
  (∃ ε > 0, ∀ x : ℝ, abs (x - 2) < ε → f x > f 2) :=
by
  sorry

end local_minimum_at_2_l764_764243


namespace knights_adjacent_probability_sum_l764_764481

open Nat

theorem knights_adjacent_probability_sum : 
  let total_knights := 30
  let chosen_knights := 4
  let total_ways := Nat.choose total_knights chosen_knights
  let no_adjacent_count := 30 * 27 * 26 * 25
  let P_no_adjacent := no_adjacent_count / total_ways
  let P_adjacent := 1 - P_no_adjacent 
  let P_adjacent_numerator := 53
  let P_adjacent_denominator := 183
  P_adjacent = P_adjacent_numerator / P_adjacent_denominator →
  P_adjacent_numerator + P_adjacent_denominator = 236 := 
by
  -- Definitions and given conditions
  let total_knights := 30
  let chosen_knights := 4
  -- Total ways to choose 4 out of 30 knights
  let total_ways := Nat.choose total_knights chosen_knights
  -- Ways to place knights such that none are adjacent
  let no_adjacent_count := 30 * 27 * 26 * 25
  -- Calculate the probability of no adjacent knights
  let P_no_adjacent := (no_adjacent_count : ℚ) / total_ways
  -- Calculate the probability of at least one pair of adjacent knights
  let P_adjacent := 1 - P_no_adjacent
  -- Given the final fraction
  let P_adjacent_numerator := 53
  let P_adjacent_denominator := 183
  -- Assert the final condition
  have P_eq : P_adjacent = P_adjacent_numerator / P_adjacent_denominator := sorry
  show P_adjacent_numerator + P_adjacent_denominator = 236 from by
    rw [←P_eq]
    exact rfl

end knights_adjacent_probability_sum_l764_764481


namespace vasya_tolya_no_meet_at_O_l764_764266

variables {A B C D O : Type} [linear_ordered_field A]
variables (AB BC CD DA AC BD : A) (constant_speed_Petya constant_speed_Vasya constant_speed_Tolya : A)

-- Definitions of paths and convexity
def convex_quadrilateral (A B C D : Type) : Prop :=
  -- Add conditions for convexity of the quadrilateral ABCD here (detail omitted for brevity)
  sorry

def constant_speeds (A B C D : Type) (constant_speed_Petya constant_speed_Vasya constant_speed_Tolya : A) : Prop :=
  -- Each pedestrian walks at constant speed
  sorry

def intersect_at_O (A B C D O : Type) : Prop :=
  -- Condition for Vasya and Tolya intersecting at O
  sorry

noncomputable def problem_statement : Prop :=
  convex_quadrilateral A B C D ∧ 
  constant_speeds A B C D constant_speed_Petya constant_speed_Vasya constant_speed_Tolya →
  ¬intersect_at_O A B C D O

-- Now declare the theorem that states Vasya and Tolya cannot meet at O simultaneously
theorem vasya_tolya_no_meet_at_O (A B C D O : Type) 
    (h1 : convex_quadrilateral A B C D)
    (h2 : constant_speeds A B C D constant_speed_Petya constant_speed_Vasya constant_speed_Tolya) :
    ¬intersect_at_O A B C D O :=
begin
  sorry -- Proof to be filled in
end

end vasya_tolya_no_meet_at_O_l764_764266


namespace cost_price_of_ball_is_60_l764_764512

-- Definitions based on the problem conditions
def cost_price_per_ball (x : ℝ) : Prop :=
  let total_cost_price := 17 * x in
  let selling_price := 720 in
  let loss := 5 * x in
  total_cost_price - loss = selling_price

-- The theorem to prove
theorem cost_price_of_ball_is_60 : 
  ∃ x : ℝ, cost_price_per_ball x ∧ x = 60 :=
sorry

end cost_price_of_ball_is_60_l764_764512


namespace second_investment_value_l764_764894

theorem second_investment_value :
  ∃ (x : ℝ), (0.07 * 500 + 0.11 * x = 0.10 * (500 + x)) → x = 1500 := by
{
  use 1500,
  intro h,
  sorry
}

end second_investment_value_l764_764894


namespace minimize_expression_l764_764254

theorem minimize_expression : ∃ c : ℝ, (∀ x : ℝ, (1/3 * x^2 + 7*x - 4) ≥ (1/3 * c^2 + 7*c - 4)) ∧ (c = -21/2) :=
sorry

end minimize_expression_l764_764254


namespace production_in_four_minutes_l764_764050

-- Definitions for conditions
def production_rate_per_machine := 270 / 6
def total_production_in_a_minute := (n : ℕ) → n * production_rate_per_machine

-- Theorem statement
theorem production_in_four_minutes :
  total_production_in_a_minute 16 * 4 = 2880 :=
by
  sorry

end production_in_four_minutes_l764_764050


namespace unique_positive_integer_x_l764_764130

def length (x : ℕ) := x + 6
def width (x : ℕ) := x - 6
def height (x : ℕ) := x^2 + 36
def volume (x : ℕ) := length x * width x * height x

theorem unique_positive_integer_x :
  ∃! x : ℕ, (0 < x) ∧ (volume x < 800) :=
sorry

end unique_positive_integer_x_l764_764130


namespace coach_recommendation_l764_764073

def shots_A : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def shots_B : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) (mean : ℚ) : ℚ :=
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

noncomputable def recommendation (shots_A shots_B : List ℕ) : String :=
  let avg_A := average shots_A
  let avg_B := average shots_B
  let var_A := variance shots_A avg_A
  let var_B := variance shots_B avg_B
  if avg_A = avg_B ∧ var_A > var_B then "player B" else "player A"

theorem coach_recommendation : recommendation shots_A shots_B = "player B" :=
  by
  sorry

end coach_recommendation_l764_764073


namespace cylindrical_to_cartesian_coords_l764_764297

theorem cylindrical_to_cartesian_coords (ρ θ z : ℝ) (hρ : ρ = sqrt 2) (hθ : θ = 5 * Real.pi / 4) (hz : z = sqrt 2) :
    let x := ρ * Real.cos θ
    let y := ρ * Real.sin θ
    (x, y, z) = (-1, -1, sqrt 2) :=
by
    sorry

end cylindrical_to_cartesian_coords_l764_764297


namespace valid_license_plates_count_l764_764930

-- Defining the total number of choices for letters and digits
def num_letter_choices := 26
def num_digit_choices := 10

-- Function to calculate the total number of valid license plates
def total_license_plates := num_letter_choices ^ 3 * num_digit_choices ^ 4

-- The proof statement
theorem valid_license_plates_count : total_license_plates = 175760000 := 
by 
  -- The placeholder for the proof
  sorry

end valid_license_plates_count_l764_764930


namespace min_value_reciprocal_sum_l764_764774

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  3 ≤ (1 / a) + (1 / b) + (1 / c) :=
by sorry

end min_value_reciprocal_sum_l764_764774


namespace sum_elements_T_l764_764767

noncomputable def T := {x : ℝ | ∃ a b c d : ℕ, 0 ≤ a ∧ a < 10 ∧ 
                                   0 ≤ b ∧ b < 10 ∧ 
                                   0 ≤ c ∧ c < 10 ∧ 
                                   0 ≤ d ∧ d < 10 ∧ 
                                   a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 
                                   d ≠ b ∧ d ≠ c ∧ x = a * 10^3 + b * 10^2 + c * 10 + d }

theorem sum_elements_T : ∑ x in T, x = 28028 / 111 :=
by
    sorry

end sum_elements_T_l764_764767


namespace benny_lost_books_l764_764051

-- Define the initial conditions
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def total_books : ℕ := sandy_books + tim_books
def remaining_books : ℕ := 19

-- Define the proof problem to find out the number of books Benny lost
theorem benny_lost_books : total_books - remaining_books = 24 :=
by
  sorry -- Insert proof here

end benny_lost_books_l764_764051


namespace tax_diminished_by_16_percent_l764_764843

variables (T X : ℝ)

-- Condition: The new revenue is 96.6% of the original revenue
def new_revenue_effect : Prop :=
  (1.15 * (T - X) / 100) = (T / 100) * 0.966

-- Target: Prove that X is 16% of T
theorem tax_diminished_by_16_percent (h : new_revenue_effect T X) : X = 0.16 * T :=
sorry

end tax_diminished_by_16_percent_l764_764843


namespace first_payment_amount_l764_764383

theorem first_payment_amount 
  (P : ℕ)
  (r : ℕ := 3)
  (n : ℕ := 10)
  (total_amount : ℕ := 2952400)
  (payment_series_sum : ℕ := P * (1 - r ^ n) / (1 - r)) 
  (h : total_amount = payment_series_sum) :
  P = 100 :=
by {
  have payment_series_sum_val : P * (1 - 3 ^ 10) / (1 - 3) = 2952400 := h,
  sorry
}

end first_payment_amount_l764_764383


namespace find_general_formula_find_sum_l764_764666

def sequence_a (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → 2 * S n = (a n)^2 + a n

def sequence_b (b : ℕ → ℝ) :=
  b 1 = 1 ∧ (∀ n : ℕ, n > 0 → 2 * b (n + 1) - b n = 0)

def sequence_c (a b c : ℕ → ℝ) :=
  ∀ n : ℕ, c n = a n * b n

def sum_terms (c : ℕ → ℝ) (T : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → T n = ∑ k in finset.range n, c (k + 1)

theorem find_general_formula
  {a : ℕ → ℝ} {S : ℕ → ℝ}
  (h_seq_a : sequence_a a S) :
  ∀ n : ℕ, n > 0 → a n = n := sorry

theorem find_sum
  {a b c : ℕ → ℝ} {T : ℕ → ℝ}
  (h_seq_a : sequence_a a (λ n, a 1 + n * (n - 1) / 2))
  (h_seq_b : sequence_b b)
  (h_seq_c : sequence_c a b c)
  (h_sum_terms : sum_terms c T) :
  ∀ n : ℕ, n > 0 → T n = 4 - (n + 2) * (1 / 2)^(n - 1) := sorry

end find_general_formula_find_sum_l764_764666


namespace bug_crawl_distance_l764_764533

-- Define the positions visited by the bug
def start_position := -3
def first_stop := 0
def second_stop := -8
def final_stop := 10

-- Define the function to calculate the total distance crawled by the bug
def total_distance : ℤ :=
  abs (first_stop - start_position) + abs (second_stop - first_stop) + abs (final_stop - second_stop)

-- Prove that the total distance is 29 units
theorem bug_crawl_distance : total_distance = 29 :=
by
  -- Definitions are used here to validate the statement
  sorry

end bug_crawl_distance_l764_764533


namespace equal_profit_for_Robi_and_Rudy_l764_764048

theorem equal_profit_for_Robi_and_Rudy
  (robi_contrib : ℕ)
  (rudy_extra_contrib : ℕ)
  (profit_percent : ℚ)
  (share_profit_equally : Prop)
  (total_profit: ℚ)
  (each_share: ℕ) :
  robi_contrib = 4000 →
  rudy_extra_contrib = (1/4) * robi_contrib →
  profit_percent = 0.20 →
  share_profit_equally →
  total_profit = profit_percent * (robi_contrib + robi_contrib + rudy_extra_contrib) →
  each_share = (total_profit / 2) →
  each_share = 900 :=
by {
  sorry
}

end equal_profit_for_Robi_and_Rudy_l764_764048


namespace roots_polynomial_equation_l764_764489

noncomputable def rootsEquation (x y : ℝ) := x + y = 10 ∧ |x - y| = 12

theorem roots_polynomial_equation : ∃ (x y : ℝ), rootsEquation x y ∧ (x^2 - 10 * x - 11 = 0) := sorry

end roots_polynomial_equation_l764_764489


namespace find_c_l764_764776

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x : ℝ, g_inv (g x c) = x) -> c = 3 :=
by 
  intro h
  sorry

end find_c_l764_764776


namespace sum_elements_T_l764_764764

noncomputable def T := {x : ℝ | ∃ a b c d : ℕ, 0 ≤ a ∧ a < 10 ∧ 
                                   0 ≤ b ∧ b < 10 ∧ 
                                   0 ≤ c ∧ c < 10 ∧ 
                                   0 ≤ d ∧ d < 10 ∧ 
                                   a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 
                                   d ≠ b ∧ d ≠ c ∧ x = a * 10^3 + b * 10^2 + c * 10 + d }

theorem sum_elements_T : ∑ x in T, x = 28028 / 111 :=
by
    sorry

end sum_elements_T_l764_764764


namespace area_ratio_equilateral_extension_l764_764015

theorem area_ratio_equilateral_extension
  (A B C A' B' C' : Type)
  (ABC_equilateral : equilateral_triangle A B C)
  (BB'_length : dist B B' = 2 * dist A B)
  (CC'_length : dist C C' = 2 * dist B C)
  (AA'_length : dist A A' = 2 * dist C A)
  (area_tri_ABC : ℝ)
  (area_tri_A'B'C' : ℝ) :
  area_tri_A'B'C' / area_tri_ABC = 9 :=
sorry

end area_ratio_equilateral_extension_l764_764015


namespace find_x_l764_764233

theorem find_x (x : ℝ) (hx_pos : x > 0) (hx_ceil_eq : ⌈x⌉ = 15) : x = 14 :=
by
  -- Define the condition
  have h_eq : ⌈x⌉ * x = 210 := sorry
  -- Prove that the only solution is x = 14
  sorry

end find_x_l764_764233


namespace min_third_side_of_right_triangle_l764_764346

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end min_third_side_of_right_triangle_l764_764346


namespace hyperbola_equation_l764_764310

-- Define the conditions
def hyperbola_eq := ∀ (x y a b : ℝ), a > 0 ∧ b > 0 → x^2 / a^2 - y^2 / b^2 = 1
def parabola_eq := ∀ (x y : ℝ), y^2 = (2 / 5) * x
def intersection_point_M := ∃ (x : ℝ), ∀ (y : ℝ), y = 1 → y^2 = (2 / 5) * x
def line_intersect_N := ∀ (F₁ M N : ℝ × ℝ), 
  (N.1 = -1 / 10) ∧ (F₁.1 ≠ M.1) ∧ (N.2 = 0)

-- State the proof problem
theorem hyperbola_equation 
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyp_eq : hyperbola_eq)
  (par_eq : parabola_eq)
  (int_pt_M : intersection_point_M)
  (line_int_N : line_intersect_N) :
  ∀ (x y : ℝ), x^2 / 5 - y^2 / 4 = 1 :=
by sorry

end hyperbola_equation_l764_764310


namespace num_black_haired_girls_l764_764790

-- Definitions representing the given conditions
def initial_choir_size : ℕ := 80
def initial_blonde : ℕ := 30
def added_blonde : ℕ := 10
def total_blonde := initial_blonde + added_blonde
def total_choir_size := initial_choir_size + added_blonde

-- Statement representing the proof objective
theorem num_black_haired_girls (initial_choir_size initial_blonde added_blonde : ℕ)
  (init_ch_sz : initial_choir_size = 80)
  (init_bl : initial_blonde = 30)
  (add_bl : added_blonde = 10)
  (total_ch_sz : initial_choir_size + added_blonde = 90)
  : (total_choir_size - total_blonde = 50) :=
by {
  -- Declaration of intermediate steps
  let total_blonde := initial_blonde + added_blonde,
  let total_choir_size := initial_choir_size + added_blonde,
  -- Proof obligations
  have h1 : total_blonde = 40 := by rw [init_bl, add_bl],
  have h2 : total_choir_size = 90 := by rw [init_ch_sz, add_bl],
  -- Conclusion
  show total_choir_size - total_blonde = 50,
  rw [h1, h2],
  exact rfl
}

end num_black_haired_girls_l764_764790


namespace horses_months_b_l764_764132

theorem horses_months_b (a_horses : ℕ) (a_months : ℕ) (b_horses : ℕ) (b_payment : ℕ)
(c_horses : ℕ) (c_months : ℕ) (total_payment : ℕ) : 
a_horses = 12 →
a_months = 8 →
b_horses = 16 →
b_payment = 180 →
c_horses = 18 →
c_months = 6 →
total_payment = 435 →
∃ (b_months : ℕ), b_months = 9 :=
by 
  intro ha hm hb hbp hc hcm ht
  use 9
  sorry

end horses_months_b_l764_764132


namespace population_net_change_l764_764923

theorem population_net_change
  (initial_population : ℝ)
  (year1_increase : initial_population * (6/5) = year1_population)
  (year2_increase : year1_population * (6/5) = year2_population)
  (year3_decrease : year2_population * (4/5) = year3_population)
  (year4_decrease : year3_population * (4/5) = final_population) :
  ((final_population - initial_population) / initial_population) * 100 = -8 :=
  sorry

end population_net_change_l764_764923


namespace unknown_score_is_66_l764_764796

theorem unknown_score_is_66 
  (scores : List ℕ)
  (h_scores : scores = [65, 70, 78, 85, 92])
  (h_avg_int : ∀ i < 6, ∃ k : ℤ, (List.take (i + 1) scores).sum % (i + 1) = 0) :
  ∃ x, x = 66 ∧ (h_scores ++ [x]).sum % 6 = 0 := 
sorry

end unknown_score_is_66_l764_764796


namespace one_fourth_of_8_8_is_11_over_5_l764_764989

theorem one_fourth_of_8_8_is_11_over_5 :
  ∀ (a b : ℚ), a = 8.8 → b = 1/4 → b * a = 11/5 :=
by
  assume a b : ℚ,
  assume ha : a = 8.8,
  assume hb : b = 1/4,
  sorry

end one_fourth_of_8_8_is_11_over_5_l764_764989


namespace arithmetic_square_root_of_9_l764_764823

theorem arithmetic_square_root_of_9 : ∃ y : ℕ, y^2 = 9 ∧ y = 3 :=
by
  sorry

end arithmetic_square_root_of_9_l764_764823


namespace euler_line_parallel_l764_764426

notation "ℝ" => Real

/-!
  Prove that the Euler line of triangle ABC is parallel to side BC if and only if:
  2a^4 = (b^2 - c^2)^2 + (b^2 + c^2)a^2
-/

theorem euler_line_parallel (a b c : ℝ) (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  (∃ (A B C : Type) [triangle ABC], (euler_line_parallel_to_side ABC BC)) ↔ 2 * a^4 = (b^2 - c^2)^2 + (b^2 + c^2) * a^2 :=
  sorry

end euler_line_parallel_l764_764426


namespace fundamental_system_diff_eq_l764_764257

-- Conditions
def y1 (x : ℝ) : ℝ := exp x
def y2 (x : ℝ) : ℝ := exp (-x)

-- Statement of the proof problem
theorem fundamental_system_diff_eq :
  (∀ x, y1 x = exp x) ∧ (∀ x, y2 x = exp (-x)) →
  (∃ a b c, (∀ y x, a * y'' x + b * y' x + c * y x = 
    (differentiation twice_on y x) - y x = 0)) :=
by
  sorry

end fundamental_system_diff_eq_l764_764257


namespace find_ciphertext_word_l764_764364

namespace Cryptography

def letter_of_number (n : ℕ) : Option Char :=
  if n ≥ 0 ∧ n ≤ 25 then
    some (Char.ofNat (n + 97))  -- ASCII code for 'a' is 97
  else
    none

theorem find_ciphertext_word :
  ∃ (x1 x2 x3 x4 : ℕ),
  (0 ≤ x1 ∧ x1 ≤ 25) ∧
  (0 ≤ x2 ∧ x2 ≤ 25) ∧
  (0 ≤ x3 ∧ x3 ≤ 25) ∧
  (0 ≤ x4 ∧ x4 ≤ 25) ∧
  ((x1 + 2 * x2) % 26 = 9) ∧
  ((3 * x2) % 26 = 16) ∧
  ((x3 + 2 * x4) % 26 = 23) ∧
  ((3 * x4) % 26 = 12) ∧
  letter_of_number x1 = some 'h' ∧
  letter_of_number x2 = some 'o' ∧
  letter_of_number x3 = some 'p' ∧
  letter_of_number x4 = some 'e' := 
by
  use 7, 14, 15, 4
  repeat {
    split,
    all_goals {sorry}
  }

end Cryptography

end find_ciphertext_word_l764_764364


namespace angela_initial_action_figures_l764_764942

theorem angela_initial_action_figures (X : ℕ) (h1 : X - (1/4 : ℚ) * X - (1/3 : ℚ) * (3/4 : ℚ) * X = 12) : X = 24 :=
sorry

end angela_initial_action_figures_l764_764942


namespace value_of_x_l764_764741

variable (a b x : ℝ)
variable (h₁ : a + b = 30)
variable (h₂ : 30a - a^2 = 15x)

theorem value_of_x : x = 15 :=
sorry

end value_of_x_l764_764741


namespace num_valid_lines_l764_764483

def is_valid_line (a b : ℕ) : Prop :=
  ∃ (l : ℝ → ℝ), ∀ (x y : ℝ), (x = 1 ∧ y = 3) → (y = l x) ∧
    (a ≠ 0) ∧ (b ≠ 0) ∧ (1 / a + 3 / b = 1)

theorem num_valid_lines : ∃! (a b : ℕ), is_valid_line a b :=
begin
  sorry
end

end num_valid_lines_l764_764483


namespace sin_x_bisects_circle_area_l764_764298

-- Definition of the circle
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Definition of the bisecting function
def bisecting_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, circle x (f x) → f x = sin x

-- Theorem statement
theorem sin_x_bisects_circle_area :
  bisecting_function (λ x, sin x) :=
sorry

end sin_x_bisects_circle_area_l764_764298


namespace original_jellybeans_correct_l764_764611

-- Define the constants and the equation
def num_jellybeans : Real := 50

-- Define the condition that Jenny eats 20% of the jellybeans each day
def remaining_after_one_day (x : Real) : Real := 0.8 * x

-- Define that at the end of the second day, 32 jellybeans remain
def remaining_after_two_days (x : Real) : Real := remaining_after_one_day(remaining_after_one_day(x))

theorem original_jellybeans_correct : remaining_after_two_days num_jellybeans = 32 :=
by {
  unfold remaining_after_two_days remaining_after_one_day,
  norm_num,
  sorry -- Proof goes here.
}

end original_jellybeans_correct_l764_764611


namespace polynomial_non_negative_for_all_real_iff_l764_764355

theorem polynomial_non_negative_for_all_real_iff (a : ℝ) :
  (∀ x : ℝ, x^4 + (a - 1) * x^2 + 1 ≥ 0) ↔ a ≥ -1 :=
by sorry

end polynomial_non_negative_for_all_real_iff_l764_764355


namespace range_of_a_l764_764356

variable {a x : ℝ}

theorem range_of_a (h : ∀ x, (a - 5) * x > a - 5 ↔ x < 1) : a < 5 := 
sorry

end range_of_a_l764_764356


namespace average_speed_car_y_l764_764210

-- Defining the constants based on the problem conditions
def speedX : ℝ := 35
def timeDifference : ℝ := 1.2  -- This is 72 minutes converted to hours
def distanceFromStartOfY : ℝ := 294

-- Defining the main statement
theorem average_speed_car_y : 
  ( ∀ timeX timeY distanceX distanceY : ℝ, 
      timeX = timeY + timeDifference ∧
      distanceX = speedX * timeX ∧
      distanceY = distanceFromStartOfY ∧
      distanceX = distanceFromStartOfY + speedX * timeDifference
  → distanceY / timeX = 30.625) :=
sorry

end average_speed_car_y_l764_764210


namespace triangle_inequality_l764_764754

noncomputable def p (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def r (a b c : ℝ) : ℝ := 
  let p := p a b c
  let x := p - a
  let y := p - b
  let z := p - c
  Real.sqrt ((x * y * z) / (x + y + z))

noncomputable def x (a b c : ℝ) : ℝ := p a b c - a
noncomputable def y (a b c : ℝ) : ℝ := p a b c - b
noncomputable def z (a b c : ℝ) : ℝ := p a b c - c

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 / (x a b c)^2 + 1 / (y a b c)^2 + 1 / (z a b c)^2 ≥ (x a b c + y a b c + z a b c) / ((x a b c) * (y a b c) * (z a b c)) := by
    sorry

end triangle_inequality_l764_764754


namespace number_of_boys_from_other_communities_l764_764363

-- Define the initial conditions
variables (initial_boys : ℕ)
variables (percent_muslims percent_hindus percent_sikhs : ℝ)

-- Define growth rates and dropout rates for other communities
variables (growth_others dropout_others : ℝ)

-- Assume initial conditions
def problem_conditions := 
  let percent_others := 1 - (percent_muslims + percent_hindus + percent_sikhs) in
  let initial_others := percent_others * initial_boys in
  let growth := growth_others * initial_others in
  let dropouts := dropout_others * initial_others in
  let final_others := initial_others + growth - dropouts in
  final_others = 124

theorem number_of_boys_from_other_communities (
  h_initial : initial_boys = 650)
  (h_perc_muslims : percent_muslims = 0.44) 
  (h_perc_hindus : percent_hindus = 0.28) 
  (h_perc_sikhs : percent_sikhs = 0.10)
  (h_growth_others : growth_others = 0.10) 
  (h_dropout_others : dropout_others = 0.04) :
  problem_conditions initial_boys percent_muslims percent_hindus percent_sikhs growth_others dropout_others :=
by 
  sorry

end number_of_boys_from_other_communities_l764_764363


namespace sum_of_repeating_decimals_l764_764769

def repeating_decimal_sum := 
  let T := {x : ℝ | ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999}
  ∑ x in T, x = 908.208208208208

theorem sum_of_repeating_decimals :
  repeating_decimal_sum := sorry

end sum_of_repeating_decimals_l764_764769


namespace find_perpendicular_tangent_line_l764_764627

noncomputable def line_eq (a b c x y : ℝ) : Prop :=
a * x + b * y + c = 0

def perp_line (a b c d e f : ℝ) (x y : ℝ) : Prop :=
b * x - a * y = 0  -- Perpendicular condition

def tangent_line (f : ℝ → ℝ) (a b c x : ℝ) : Prop :=
∃ t, f t = a * t + b * (f t) + c ∧ (deriv f t) = -a / b  -- Tangency condition with derivative

theorem find_perpendicular_tangent_line :
  let f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 5 in
  ∃ a b c d e f: ℝ, perp_line 2 (-6) 1 a b c ∧ tangent_line f a b c ∧ line_eq 3 1 6 (x : ℝ) (f x) :=
sorry

end find_perpendicular_tangent_line_l764_764627


namespace simplify_expression_l764_764052

-- Define the variables and conditions
variables {a b x y : ℝ}
variable (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
variable (h2 : x ≠ -(a * y) / b)
variable (h3 : x ≠ (b * y) / a)

-- The Theorem to prove
theorem simplify_expression
  (a b x y : ℝ)
  (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
  (h2 : x ≠ -(a * y) / b)
  (h3 : x ≠ (b * y) / a) :
  (a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) *
  ((a * x + b * y)^2 - 4 * a * b * x * y) /
  (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = 
  a^2 * x^2 - b^2 * y^2 :=
sorry

end simplify_expression_l764_764052


namespace pits_less_than_22222_l764_764545

def isDescendingTriple (a b c : ℕ) : Prop := a > b ∧ b > c
def isAscendingPair (d e : ℕ) : Prop := d < e

def isPit (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 = n ∧
  isDescendingTriple d1 d2 d3 ∧ isAscendingPair d4 d5

def countPitsUnder (maxNum : ℕ) : ℕ :=
  (List.range maxNum).countp (λ n => isPit n)

theorem pits_less_than_22222 : countPitsUnder 22222 = 36 := by
  sorry

end pits_less_than_22222_l764_764545


namespace proof_smallest_integer_proof_sum_of_integers_l764_764504

def smallest_integer (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ n = 98

def sum_of_integers (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ a + b + c + d + e = 510

theorem proof_smallest_integer : ∃ n : Int, smallest_integer n := by
  sorry

theorem proof_sum_of_integers : ∃ n : Int, sum_of_integers n := by
  sorry

end proof_smallest_integer_proof_sum_of_integers_l764_764504


namespace part1_part2_l764_764320

-- Proving part (1)
theorem part1 (x : ℝ) :
  let a := (sqrt 3 * sin x, -1 : ℝ)
  let b := (cos x, sqrt 3 : ℝ)
  (Parall(a, b) → 3 * sin x - cos x = -3 * (sin x + cos x)) :=
sorry

-- Proving part (2)
theorem part2 (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/2 ∧ 
  2 * (sqrt 3 * sin x + cos x, m - 1 : ℝ) • (cos x, m) - 2 * m ^ 2 - 1 = 0)
  → m ∈ Icc (-1 / 2) 1 :=
sorry

end part1_part2_l764_764320


namespace largest_n_exists_ints_l764_764974

theorem largest_n_exists_ints (n : ℤ) :
  (∃ x y z : ℤ, n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 10 :=
sorry

end largest_n_exists_ints_l764_764974


namespace sum_of_T_elements_l764_764757

-- Define T to represent the set of numbers 0.abcd with a, b, c, d being distinct digits.
def is_valid_abcd (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def T : Set ℝ := { x | ∃ (a b c d : ℕ), is_valid_abcd a b c d ∧ x = ((1000 * a + 100 * b + 10 * c + d : ℝ) / 9999) }

-- The main theorem statement.
theorem sum_of_T_elements : ∑ x in T, x = 2520 := by
  sorry

end sum_of_T_elements_l764_764757


namespace max_sugar_in_cup_l764_764009

theorem max_sugar_in_cup (h1 : ∀ n : ℕ, n ≤ 127 → ∃ f : Fin 7 → ℕ, (∀ i, f i = 2 ^ i) ∧ ∑ i, f i * (n / 2^i % 2) = n) :
  ∃ i : Fin 7, (2 ^ i) = 64 := by
  -- The proof steps will go here
  sorry

end max_sugar_in_cup_l764_764009


namespace smallest_n_value_l764_764180

theorem smallest_n_value :
  ∃ n, (∀ (sheets : Fin 2000 → Fin 4 → Fin 4),
        (∀ (n : Nat) (h : n ≤ 2000) (a b c d : Fin n) (h' : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d),
          ∃ (i j k : Fin 5), sheets a i = sheets b i ∧ sheets a j = sheets b j ∧ sheets a k = sheets b k → ¬ sheets a i = sheets c i ∧ ¬ sheets b j = sheets c j ∧ ¬ sheets a k = sheets c k)) ↔ n = 25 :=
sorry

end smallest_n_value_l764_764180


namespace find_m_parallel_l764_764354

noncomputable def is_parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  -(A1 / B1) = -(A2 / B2)

theorem find_m_parallel : ∃ m : ℝ, is_parallel (m-1) 3 m 1 (m+1) 2 ∧ m = -2 :=
by
  unfold is_parallel
  exists (-2 : ℝ)
  sorry

end find_m_parallel_l764_764354


namespace parallelogram_diagonals_not_equal_l764_764937

-- Defining a quadrilateral and a parallelogram
structure Quadrilateral :=
(vertices : ℕ) (sides : ℕ)

def is_parallelogram (q : Quadrilateral) : Prop :=
q.vertices = 4 ∧
∃ (a b c d : ℝ),
-- Representing the parallel and equal pairs of sides
a = c ∧ b = d ∧ -- Opposite sides are equal
(a + b = c + d)

structure Parallelogram extends Quadrilateral :=
(parallel_pairs : vertices = 4 ∧ sides = 4)

-- Define properties
def property_A (p : Parallelogram) : Prop :=
p.parallel_pairs

def property_B (p : Parallelogram) : Prop :=
∃ (a b c d : ℝ),
a = c ∧ b = d ∧ -- Opposite sides are equal
(a + b = c + d)

def property_C (p : Parallelogram) : Prop :=
∃ (d1 d2 : ℝ),
d1 = d2 -- Diagonals are equal

def property_D (p : Parallelogram) : Prop :=
∃ (d1 d2 : ℝ),
d1 / 2 = d2 / 2 -- Diagonals bisect each other

theorem parallelogram_diagonals_not_equal :
  ∀ (p : Parallelogram), property_A p ∧ property_B p ∧ property_D p → ¬ property_C p :=
by sorry

end parallelogram_diagonals_not_equal_l764_764937


namespace inscribed_square_area_of_right_triangle_l764_764216

noncomputable def hypotenuse_length (AB AC : ℝ) : ℝ :=
  Real.sqrt (AB^2 + AC^2)

noncomputable def inscribed_square_area (AB AC : ℝ) : real :=
  let BC := hypotenuse_length AB AC
  let s := BC^2 / (AB + AC)
  s^2

theorem inscribed_square_area_of_right_triangle :
  inscribed_square_area 63 16 = 2867 :=
by
  let AB := 63
  let AC := 16
  have h : 2867 = (hypotenuse_length AB AC) ^ 2 / (AB + AC) ^ 2 :=
    sorry
  exact h

end inscribed_square_area_of_right_triangle_l764_764216


namespace arithmetic_sequence_scaled_l764_764573

variable {α : Type*} [AddGroup α] [HasSmul ℝ α] 

theorem arithmetic_sequence_scaled (a : ℕ → α) (d : α) (c : ℝ)
    (h_arith : ∀ n, a (n+1) - a n = d)
    (hc : c ≠ 0) :
    ∀ n, c • a (n+1) - c • a n = c • d :=
by
  intro n
  calc
    c • a (n+1) - c • a n
        = c • (a (n+1) - a n) : sorry
    ... = c • d : sorry

end arithmetic_sequence_scaled_l764_764573


namespace max_coloured_regions_proof_l764_764464

noncomputable def max_coloured_regions (n : ℕ) : ℝ := n * (n + 1) / 3

theorem max_coloured_regions_proof (n : ℕ) (hn : n ≥ 3)
  (h1 : ∀ i j, i ≠ j → ¬Parallel (lines i) (lines j))
  (h2 : ∀ i j k, i < j < k → ¬Concurrent (lines i) (lines j) (lines k))
  (h3 : ∀ r1 r2, Coloured r1 → Coloured r2 → ¬SharesSegment r1 r2)
  : ∃ m, m ≤ max_coloured_regions n := 
sorry

end max_coloured_regions_proof_l764_764464


namespace max_min_page_difference_l764_764828

-- Define the number of pages in each book
variables (Poetry Documents Rites Changes SpringAndAutumn : ℤ)

-- Define the conditions as given in the problem
axiom h1 : abs (Poetry - Documents) = 24
axiom h2 : abs (Documents - Rites) = 17
axiom h3 : abs (Rites - Changes) = 27
axiom h4 : abs (Changes - SpringAndAutumn) = 19
axiom h5 : abs (SpringAndAutumn - Poetry) = 15

-- Assertion to prove
theorem max_min_page_difference : 
  ∃ a b c d e : ℤ, a = Poetry ∧ b = Documents ∧ c = Rites ∧ d = Changes ∧ e = SpringAndAutumn ∧ 
  abs (a - b) = 24 ∧ abs (b - c) = 17 ∧ abs (c - d) = 27 ∧ abs (d - e) = 19 ∧ abs (e - a) = 15 ∧ 
  (max a (max b (max c (max d e))) - min a (min b (min c (min d e)))) = 34 :=
by {
  sorry
}

end max_min_page_difference_l764_764828


namespace a6_value_l764_764316

theorem a6_value
  (a : ℕ → ℤ)
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (geo : ∃ q : ℤ, ∀ n : ℕ, n > 0 → a (n + 1) = q^n * (a 1 + 1) - 1):
  a 6 = 63 :=
by
  sorry

end a6_value_l764_764316


namespace sum_of_lengths_of_intervals_l764_764021

def f (x : ℝ) : ℝ := (Int.floor x : ℝ) * ((2013 : ℝ) ^ (x - Int.floor x) - 1)

theorem sum_of_lengths_of_intervals : 
  (∑ k in finset.range (2013 - 1), real.log ((k + 1) / k) / real.log 2013) = 1 := sorry

end sum_of_lengths_of_intervals_l764_764021


namespace candles_number_problem_l764_764212

theorem candles_number_problem :
  ∃ c : ℕ, (choose c 2) * (choose 9 8) = 54 ∧ c = 4 :=
by {
  have h_comb_candles := @choose_eq_factorial_div_factorial (λ c, choose c 2),
  have h_comb_flowers := @choose_eq_factorial_div_factorial (λ _, choose 9 8),
  have h_comb_flower := 9,
  use 4,
  split,
  { rw [h_comb_candles, ←mul_eq_mul_left, h_comb_flowers, ←mul_assoc, choose_one_mul],
    linarith },
  { refl }
}

end candles_number_problem_l764_764212


namespace number_of_true_propositions_l764_764976

theorem number_of_true_propositions :
  (¬ (∀ (a b : ℝ), (a > b ↔ a^2 > b^2)) ∧
   (∀ (a b : ℝ), a > b ↔ a^3 > b^3) ∧
   (¬ (∀ (a b : ℝ), a > b → |a| > |b|)) ∧
   (¬ (∀ (a b c : ℝ), a > b ↔ ac^2 ≤ bc^2))) →
  1 :=
by
  sorry

end number_of_true_propositions_l764_764976


namespace square_area_is_45_l764_764482

-- Define the basic setup of the problem
variables (L1 L2 L3 : ℝ) -- Represent the y-coordinates of the lines L1, L2, L3
variables (A B C D : ℝ × ℝ) -- Points A, B, C, D in the plane

-- Conditions on the lines
def lines_are_parallel : Prop := L1 ≠ L2 ∧ L1 ≠ L3 ∧ L2 ≠ L3

def distances_between_lines : Prop := 
  (L2 - L1 = 3) ∧ (L3 - L2 = 3)

-- Conditions on the square ABCD
def square_conditions : Prop := 
  A.2 = L1 ∧ B.2 = L3 ∧ C.2 = L2

-- Definition of a square in terms of equal side lengths and right angles
def is_square (A B C D : ℝ × ℝ) : Prop := 
  (∥A - B∥ = ∥B - C∥) ∧ (∥B - C∥ = ∥C - D∥) ∧ (∥C - D∥ = ∥D - A∥) ∧
  (angle A B C = π / 2) ∧ (angle B C D = π / 2) ∧ (angle C D A = π / 2) ∧ (angle D A B = π / 2)

-- Main theorem statement
theorem square_area_is_45 :
  lines_are_parallel L1 L2 L3 ∧ distances_between_lines L1 L2 L3 ∧ square_conditions L1 L2 L3 A B C D ∧ is_square A B C D →
  let side_length := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) in
  side_length^2 = 45 :=
sorry

end square_area_is_45_l764_764482


namespace ellipse_distance_l764_764595

theorem ellipse_distance :
  ∀ x y : ℝ,
    4 * (x - 2) ^ 2 + 16 * y ^ 2 = 64 →
    let C := (2 + 4, 0); let D := (2, 2) in
    dist C D = 2 * Real.sqrt 5 :=
by
  intros x y h
  let C := (2 + 4, 0)
  let D := (2, 2)
  rw [dist_eq, (id @eq), sq, sq]
  -- The proof steps would proceed here, constructing the distance CD.
  sorry

end ellipse_distance_l764_764595


namespace system_solution_is_unique_l764_764093

theorem system_solution_is_unique
  (a b : ℝ)
  (h1 : 2 - a * 5 = -1)
  (h2 : b + 3 * 5 = 8) :
  (∃ m n : ℝ, 2 * (m + n) - a * (m - n) = -1 ∧ b * (m + n) + 3 * (m - n) = 8 ∧ m = 3 ∧ n = -2) :=
by
  sorry

end system_solution_is_unique_l764_764093


namespace determine_x_l764_764448

theorem determine_x
  (total_area : ℝ)
  (side_length_square1 : ℝ)
  (side_length_square2 : ℝ)
  (h1 : total_area = 1300)
  (h2 : side_length_square1 = 3 * x)
  (h3 : side_length_square2 = 7 * x) :
    x = Real.sqrt (2600 / 137) :=
by
  sorry

end determine_x_l764_764448


namespace smallest_N_conditions_l764_764165

theorem smallest_N_conditions:
  ∃N : ℕ, N % 9 = 8 ∧
           N % 8 = 7 ∧
           N % 7 = 6 ∧
           N % 6 = 5 ∧
           N % 5 = 4 ∧
           N % 4 = 3 ∧
           N % 3 = 2 ∧
           N % 2 = 1 ∧
           N = 2519 :=
sorry

end smallest_N_conditions_l764_764165


namespace necessary_but_not_sufficient_condition_l764_764936

theorem necessary_but_not_sufficient_condition (a b : ℝ) : (a > b) → (a - b > -2) ∧ ¬(a - b > -2 → a > b) :=
by
  intros h
  split
  { linarith }
  {
    intro h1
    have h2 : b > a,
    {
      -- Provide a counterexample where (a - b > -2) is true but (a > b) is false
      sorry
    }
    linarith
  }

end necessary_but_not_sufficient_condition_l764_764936


namespace f_even_f_increasing_l764_764661

-- Define the function f with its domain and properties
noncomputable def f : ℝ → ℝ := sorry
axiom domain_f : ∀ x, x ≠ 0 → f x ≠ -∞ ∧ f x ≠ ∞
axiom func_prop : ∀ x₁ x₂, x₁ ≠ 0 → x₂ ≠ 0 → f (x₁ * x₂) = f x₁ + f x₂
axiom positive_prop : ∀ x, x > 1 → f x > 0

-- Statement 1: Prove that f is an even function.
theorem f_even : ∀ x : ℝ, x ≠ 0 → f x = f (-x) := sorry

-- Statement 2: Prove that f is increasing on (0, +∞).
theorem f_increasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := sorry

end f_even_f_increasing_l764_764661


namespace not_possible_divide_44_balls_into_9_piles_l764_764523

theorem not_possible_divide_44_balls_into_9_piles :
    ¬ ∃ (a : Fin 9 → ℕ), (∀ i j : Fin 9, i ≠ j → a i ≠ a j) ∧ (∑ i, a i = 44) :=
by
  sorry

end not_possible_divide_44_balls_into_9_piles_l764_764523


namespace sheep_to_cow_water_ratio_l764_764795

-- Set up the initial conditions
def number_of_cows := 40
def water_per_cow_per_day := 80
def number_of_sheep := 10 * number_of_cows
def total_water_per_week := 78400

-- Calculate total water consumption of cows per week
def water_cows_per_week := number_of_cows * water_per_cow_per_day * 7

-- Calculate total water consumption of sheep per week
def water_sheep_per_week := total_water_per_week - water_cows_per_week

-- Calculate daily water consumption per sheep
def water_sheep_per_day := water_sheep_per_week / 7
def daily_water_per_sheep := water_sheep_per_day / number_of_sheep

-- Define the target ratio
def target_ratio := 1 / 4

-- Statement to prove
theorem sheep_to_cow_water_ratio :
  (daily_water_per_sheep / water_per_cow_per_day) = target_ratio :=
sorry

end sheep_to_cow_water_ratio_l764_764795


namespace number_of_commonly_used_structures_is_3_l764_764414

def commonly_used_algorithm_structures : Nat := 3
theorem number_of_commonly_used_structures_is_3 
  (structures : Nat)
  (h : structures = 1 ∨ structures = 2 ∨ structures = 3 ∨ structures = 4) :
  commonly_used_algorithm_structures = 3 :=
by
  -- Proof to be added
  sorry

end number_of_commonly_used_structures_is_3_l764_764414


namespace annual_interest_rate_of_second_investment_l764_764432

def total_investment := 10000
def interest_rate_first := 0.06
def amount_invested_first := 7200
def total_interest := 684

theorem annual_interest_rate_of_second_investment : 
  ∃ r, 
    let interest_first := amount_invested_first * interest_rate_first in
    let interest_second := total_interest - interest_first in
    let amount_invested_second := total_investment - amount_invested_first in
    interest_second = amount_invested_second * r ∧
    r = 0.09 :=
by
  existsi (0.09 : ℝ)
  sorry

end annual_interest_rate_of_second_investment_l764_764432


namespace no_rational_roots_of_polynomial_l764_764987

theorem no_rational_roots_of_polynomial :
  ¬ ∃ (x : ℚ), (3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3 = 0) :=
by
  sorry

end no_rational_roots_of_polynomial_l764_764987


namespace cards_example_l764_764100

def card := (ℕ, ℕ)

def cards : set card := {(1, 2), (1, 3), (2, 3)}

variables (A B C : card)
variables (hA_card : A ∈ cards)
variables (hB_card : B ∈ cards)
variables (hC_card : C ∈ cards)
variables (hA_B : 2 ∉ A ∨ 2 ∉ B)
variables (hB_C : 1 ∉ B ∨ 1 ∉ C)
variables (hC_sum : C.1 + C.2 ≠ 5)

theorem cards_example : A = (1, 3) :=
by sorry

end cards_example_l764_764100


namespace real_sum_of_coefficients_of_g_l764_764025

noncomputable def is_real (z : ℂ) : Prop := z.im = 0

theorem real_sum_of_coefficients_of_g
  (n : ℕ)
  (a : ℕ → ℂ)
  (b : ℕ → ℂ)
  (h_sum1 : is_real ((list.sum (list.of_fn (λ i, if i % 2 = 1 then a i else 0)))))
  (h_sum2 : is_real ((list.sum (list.of_fn (λ i, if i % 2 = 0 then a i else 0)))))
  (h_roots : ∀ x : ℂ, (polynomial.map (algebra_map ℂ ℂ) (polynomial.sum (polynomial.C ∘ a) (λ i, polynomial.X ^ (n - i)))) x = 0 ↔ 
                      (polynomial.map (algebra_map ℂ ℂ) (polynomial.sum (polynomial.C ∘ b) (λ i, polynomial.X ^ (n - i)))) (x^2) = 0) :
  is_real (polynomial.sum (λ i, b i)) := sorry

end real_sum_of_coefficients_of_g_l764_764025


namespace ratio_r_R_range_l764_764971

noncomputable def R (a b c : ℝ) : ℝ :=
1 / 2 * sqrt (a^2 + b^2 + c^2)

noncomputable def r (a b c : ℝ) : ℝ :=
sqrt ((b^2 + c^2) * (c^2 + a^2) * (a^2 + b^2)) / (2 * sqrt (b^2 * c^2 + c^2 * a^2 + a^2 * b^2))

theorem ratio_r_R_range (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  ∀ (ratio : ℝ), ratio = r a b c / R a b c → (2 * sqrt 2 / 3) ≤ ratio ∧ ratio < 1 :=
sorry

end ratio_r_R_range_l764_764971


namespace translate_point_B_to_B1_l764_764324

theorem translate_point_B_to_B1 :
  ∀ (A B A1 B1 : ℝ × ℝ), 
  A = (-1, 0) → 
  B = (1, 2) → 
  A1 = (2, -1) → 
  B1 = (4, 1) → 
  B1 = (B.1 + (A1.1 - A.1), B.2 + (A1.2 - A.2)) :=
by {
  intros A B A1 B1 hA hB hA1 hB1,
  rw [hA, hB, hA1, hB1],
  exact rfl,
}

end translate_point_B_to_B1_l764_764324


namespace perpendicular_tangent_line_exists_and_correct_l764_764630

theorem perpendicular_tangent_line_exists_and_correct :
  ∃ L : ℝ → ℝ → Prop,
    (∀ x y, L x y ↔ 3 * x + y + 6 = 0) ∧
    (∀ x y, 2 * x - 6 * y + 1 = 0 → 3 * x + y + 6 ≠ 0) ∧
    (∃ a b : ℝ, 
       b = a^3 + 3*a^2 - 5 ∧ 
       (a, b) ∈ { p : ℝ × ℝ | ∃ f' : ℝ → ℝ, f' a = 3 * a^2 + 6 * a ∧ f' a * 3 + 1 = 0 } ∧
       L a b)
:= 
sorry

end perpendicular_tangent_line_exists_and_correct_l764_764630


namespace impossible_to_equalize_l764_764857

-- Define the problem conditions
section
  variable (P : ℕ) (initialDistribution : Fin 20 → ℕ)

  -- Total number of pies condition
  def total_pies := ∑ i, initialDistribution i = 40

  -- Operation condition: Move 2 pies from one plate to two neighboring plates
  def valid_move (d : Fin 20 → ℕ) (i : Fin 20) :=
    d (i - 1) = initialDistribution (i - 1) + 1 ∧
    d i = initialDistribution i - 2 ∧
    d (i + 1) = initialDistribution (i + 1) + 1

  -- Prove that it is not possible to achieve equal pies on all plates in any configuration
  theorem impossible_to_equalize (initialDistribution : Fin 20 → ℕ)
    (h_total : total_pies initialDistribution) :
    ¬ ∃ k : Fin 20 → ℕ, (∀ i, k i = 2) ∧ (∃ d, valid_move initialDistribution d) :=
  sorry
end

end impossible_to_equalize_l764_764857


namespace sum_of_roots_of_quadratic_l764_764645

theorem sum_of_roots_of_quadratic :
  let a := 1
  let b := -5
  let c := 6
  (root_sum : Float := (-b) / (a)) in
  root_sum = 5 :=
by
  let a := 1
  let b := -5
  let c := 6
  have h1 : (root_sum : Float := (-b) / a) := sorry
  exact h1

end sum_of_roots_of_quadratic_l764_764645


namespace hike_up_time_eq_l764_764819

variable (t : ℝ)
variable (h_rate_up : ℝ := 4)
variable (h_rate_down : ℝ := 6)
variable (total_time : ℝ := 3)

theorem hike_up_time_eq (h_rate_up_eq : h_rate_up = 4) 
                        (h_rate_down_eq : h_rate_down = 6) 
                        (total_time_eq : total_time = 3) 
                        (dist_eq : h_rate_up * t = h_rate_down * (total_time - t)) :
  t = 9 / 5 := by
  sorry

end hike_up_time_eq_l764_764819


namespace increase_by_75_percent_l764_764526

theorem increase_by_75_percent
  (original_num : ℕ)
  (increase_percent : ℕ)
  (h1 : original_num = 120)
  (h2 : increase_percent = 75) : 
  original_num + (increase_percent * original_num / 100) = 210 := 
by
  rw [h1, h2]
  sorry

end increase_by_75_percent_l764_764526


namespace quadrilateral_diagonals_inequality_l764_764043

theorem quadrilateral_diagonals_inequality (a b c d e f : ℝ) :
  e^2 + f^2 ≤ b^2 + d^2 + 2 * a * c :=
by
  sorry

end quadrilateral_diagonals_inequality_l764_764043


namespace min_third_side_of_right_triangle_l764_764347

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end min_third_side_of_right_triangle_l764_764347


namespace mike_total_payment_l764_764407

def camera_initial_cost : ℝ := 4000
def camera_increase_rate : ℝ := 0.30
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200
def sales_tax_rate : ℝ := 0.08

def new_camera_cost := camera_initial_cost * (1 + camera_increase_rate)
def discounted_lens_cost := lens_initial_cost - lens_discount
def total_purchase_before_tax := new_camera_cost + discounted_lens_cost
def sales_tax := total_purchase_before_tax * sales_tax_rate
def total_purchase_with_tax := total_purchase_before_tax + sales_tax

theorem mike_total_payment : total_purchase_with_tax = 5832 := by
  sorry

end mike_total_payment_l764_764407


namespace calculate_profit_per_meter_l764_764181

/--
A trader sells 85 meters of cloth for Rs. 8925. The cost price of one meter of cloth is Rs. 100.
We need to prove that the profit per meter of cloth is Rs. 5.
-/

def meters_sold : ℕ := 85
def selling_price : ℕ := 8925
def cost_price_per_meter : ℕ := 100
def profit_per_meter : ℕ := 5

theorem calculate_profit_per_meter :
  let total_cost_price := cost_price_per_meter * meters_sold in
  let total_selling_price := selling_price in
  let total_profit := total_selling_price - total_cost_price in
  profit_per_meter * meters_sold = total_profit :=
by
  sorry

end calculate_profit_per_meter_l764_764181


namespace base8_problem_l764_764014

/--
Let A, B, and C be non-zero and distinct digits in base 8 such that
ABC_8 + BCA_8 + CAB_8 = AAA0_8 and A + B = 2C.
Prove that B + C = 14 in base 8.
-/
theorem base8_problem (A B C : ℕ) 
    (h1 : A > 0 ∧ B > 0 ∧ C > 0)
    (h2 : A < 8 ∧ B < 8 ∧ C < 8)
    (h3 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (bcd_sum : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) 
        = 8^3 * A + 8^2 * A + 8 * A)
    (sum_condition : A + B = 2 * C) :
    B + C = A + B := by {
  sorry
}

end base8_problem_l764_764014


namespace eldora_boxes_paper_clips_l764_764613

theorem eldora_boxes_paper_clips (x y : ℝ)
  (h1 : 1.85 * x + 7 * y = 55.40)
  (h2 : 1.85 * 12 + 10 * y = 61.70)
  (h3 : 1.85 = 1.85) : -- Given && Asserting the constant price of one box

  x = 15 :=
by
  sorry

end eldora_boxes_paper_clips_l764_764613


namespace smallest_nonprime_with_no_prime_factors_less_than_20_l764_764022

theorem smallest_nonprime_with_no_prime_factors_less_than_20 :
  ∃ (n : ℕ), n = 529 ∧ (∀ p : ℕ, prime p → p ∣ n → p ≥ 20) :=
sorry

end smallest_nonprime_with_no_prime_factors_less_than_20_l764_764022


namespace smallest_N_exists_l764_764167

theorem smallest_N_exists : ∃ N : ℕ, 
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  N = 503 :=
by {
  sorry
}

end smallest_N_exists_l764_764167


namespace part_a_part_b_l764_764498

-- Definition of "remarkable" polygons
def remarkable (p : Set (ℕ × ℕ)) : Prop :=
  ¬ (∃ a b c d : ℕ, p = { (i, j) | a ≤ i < a + c ∧ b ≤ j < b + d }) ∧
  ∃ s : ℕ, s ≠ 1 ∧ ∃ ps : Set (Set (ℕ × ℕ)), (∀ q ∈ ps, q = s • p) ∧ p ∈ ps

-- Problem Part (a)
theorem part_a :
  ∃ p : Set (ℕ × ℕ), p.card = 4 ∧ remarkable p :=
sorry

-- Problem Part (b)
theorem part_b :
  ∀ n > 4, ∃ p : Set (ℕ × ℕ), p.card = n ∧ remarkable p :=
sorry

end part_a_part_b_l764_764498


namespace area_U_property_l764_764018

noncomputable def ceil (x : ℝ) : ℝ := ⌈x⌉₊

def T (t : ℝ) : ℝ := real.sqrt (t - (t.floor))

def U (t : ℝ) : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.1 - T t)^2 + (p.2 - T t)^2 ≤ 4 * (T t)^2}

theorem area_U_property (t : ℝ) (ht : t ≥ 0) :
  0 ≤ measure_theory.volume (U t) ∧ measure_theory.volume (U t) ≤ 4 * real.pi :=
sorry

end area_U_property_l764_764018


namespace describe_graph_of_equation_l764_764508

theorem describe_graph_of_equation :
  (∀ x y : ℝ, (x + y)^3 = x^3 + y^3 → (x = 0 ∨ y = 0 ∨ y = -x)) :=
by
  intros x y h
  sorry

end describe_graph_of_equation_l764_764508


namespace smallest_points_set_l764_764176

-- Define the conditions of symmetry about the origin, x-axis, y-axis, y = x, and y = -x
def symmetric_origin (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (-a, -b) ∈ T

def symmetric_x_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, -b) ∈ T → (a, b) ∈ T 

def symmetric_y_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (-a, b) ∈ T → (a, b) ∈ T 

def symmetric_y_equals_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (b, a) ∈ T → (a, b) ∈ T 

def symmetric_y_equals_neg_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (-b, -a) ∈ T → (a, b) ∈ T 

-- The problem condition encoded in Lean
theorem smallest_points_set (T : Set (ℝ × ℝ)) (h_origin : symmetric_origin T) (h_x_axis : symmetric_x_axis T)
    (h_y_axis : symmetric_y_axis T) (h_y_x : symmetric_y_equals_x T) (h_y_neg_x : symmetric_y_equals_neg_x T)
    (h_point : (1, 4) ∈ T) : T.size = 8 :=
sorry

end smallest_points_set_l764_764176


namespace sum_of_arithmetic_series_l764_764143

-- Define the conditions
def first_term := 1
def last_term := 12
def number_of_terms := 12

-- Prop statement that the sum of the arithmetic series equals 78
theorem sum_of_arithmetic_series : (number_of_terms / 2) * (first_term + last_term) = 78 := 
by
  sorry

end sum_of_arithmetic_series_l764_764143


namespace plane_equation_l764_764633

theorem plane_equation (A B C D : ℤ) 
  (h1 : A = 2) (h2 : B = -1) (h3 : C = 3) 
  (h4 : A > 0) 
  (h5 : gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1)
  (point : ℤ × ℤ × ℤ)
  (h6 : point = (2, 3, -4))
  (parallel_plane : ℤ → ℤ → ℤ → ℤ)
  (h7 : ∀ x y z, parallel_plane x y z = 2 * x - y + 3 * z + 5) :
  (parallel_plane (point.1) (point.2) (point.3) = 0) → D = 11 :=
by
  sorry

end plane_equation_l764_764633


namespace cat_trip_distance_l764_764087

variable (A B : Type) [metric_space A] (cat : A → A → ℝ → ℝ → Type)

-- Definitions and conditions
def uphill_speed := 4 -- km/h
def downhill_speed := 5 -- km/h
def time_A_to_B := 132 -- minutes (2 hours and 12 minutes)
def time_B_to_A := 138 -- minutes (132 + 6)

-- Define the distances in terms of x (uphill distance)
variable (x : ℝ) -- distance uphill in km
variable (x₁ x₂ : ℝ)

-- Definitions for time taken to travel uphill and downhill
def time_uphill (x₁ : ℝ) := (x₁ / uphill_speed) * 60
def time_downhill (x₂ : ℝ) := (x₂ / downhill_speed) * 60

-- Define the total distance
def total_distance (x₁ x₂ : ℝ) := x₁ + x₂

-- Theorem statement
theorem cat_trip_distance : total_distance x (x + 2) = 10 :=
by
  have equation : 15 * x + 12 * (x + 2) = 132 := sorry
  exact sorry

end cat_trip_distance_l764_764087


namespace remaining_surface_area_l764_764903

theorem remaining_surface_area (a b : ℝ) : 
  let SA_original := 6 * (3 * a) ^ 2,
      SA_exposed := 3 * b ^ 2,
      SA_contact := 3 * b ^ 2
  in SA_original - SA_contact + SA_exposed = 54 * a ^ 2 :=
by
  sorry

end remaining_surface_area_l764_764903


namespace right_triangle_hypotenuse_len_l764_764362

theorem right_triangle_hypotenuse_len (a b : ℕ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 3) 
  (h₃ : a^2 + b^2 = c^2) : c = Real.sqrt 10 := by
  sorry

end right_triangle_hypotenuse_len_l764_764362


namespace sequence_geom_sum_bn_l764_764273

-- Definition and condition for the first part, proving that {S_n - 2} is a geometric sequence
theorem sequence_geom (S : ℕ → ℚ) (a : ℕ → ℚ) (q : ℚ) (hq : q > 0) 
  (hS4 : S 4 = 15/8)
  (h_arith : a 1 + a 2 = 6 * a 3)
  (h_geom : ∀ n, S (n) = (a 1 * (1 - q^n)) / (1 - q)) :
  ∃ q1 > 0, ∀ n > 0, S n - 2 = (-q1)^n :=
by sorry

-- Definition and condition for the second part, finding the sum of the first n terms of {b_n}
theorem sum_bn (a b : ℕ → ℚ) (q : ℚ) (hq : q > 0) 
  (ha : ∀ n, a n = (1/2)^(n-1))
  (hb : ∀ n, b n = a n * (Real.log (2 : ℚ) (a n) - 1)) :
  ∀ n, ∑ i in finset.range n, b (i + 1) = (n + 2) * (1/2)^(n-1) - 4 :=
by sorry

end sequence_geom_sum_bn_l764_764273


namespace log_equation_solution_l764_764878

theorem log_equation_solution (x : ℝ) (h_pos: x > 0)
    (h1 : log 2 (2 * x^2) = 1 + 2 * log 2 x)
    (h2 : log 4 (16 * x) = 2 + (1/2) * log 2 x)
    (h3 : log 4 (x^3) = (3/2) * log 2 x) :
    7.333 * sqrt ((log 2 (2 * x^2)) * (log 4 (16 * x))) = log 4 (x^3) → x = 16 := 
sorry

end log_equation_solution_l764_764878


namespace perp_case_parallel_distance_l764_764321

open Real

-- Define the line equations
def l1 (x y : ℝ) := 2 * x + y + 4 = 0
def l2 (a x y : ℝ) := a * x + 4 * y + 1 = 0

-- Perpendicular condition between l1 and l2
def perpendicular (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ (2 * -a) / 4 = -1)

-- Parallel condition between l1 and l2
def parallel (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ a = 8)

noncomputable def intersection_point : (ℝ × ℝ) := (-3/2, -1)

noncomputable def distance_between_lines : ℝ := (3 * sqrt 5) / 4

-- Statement for the intersection point when perpendicular
theorem perp_case (a : ℝ) : perpendicular a → ∃ x y, l1 x y ∧ l2 (-2) x y := 
by
  sorry

-- Statement for the distance when parallel
theorem parallel_distance {a : ℝ} : parallel a → distance_between_lines = (3 * sqrt 5) / 4 :=
by
  sorry

end perp_case_parallel_distance_l764_764321


namespace exchange_rate_decrease_l764_764887

-- Define the conditions for the problem
variables (x : ℕ → ℝ) (h_bound : ∀ i, 0 < x i ∧ x i < 1)
-- Given the conditions, the predicted changes are inverted
-- and the product of actual changes equals the product of predicted changes
def overall_change (prod_actual_changes prod_predicted_changes : ℝ): Prop :=
  (prod_actual_changes = prod_predicted_changes)

-- The actual monthly changes
def actual_multiplier := (∏ i in finset.range 12, (1 - x i))
def predicted_multiplier := (∏ i in finset.range 12, (1 + x i))

-- The statement we are proving
theorem exchange_rate_decrease 
  (h_eq : (∏ i in finset.range 12, (1 - x i)) = (∏ i in finset.range 12, (1 + x i))):
  (∏ i in finset.range 12, (1 - x i)) < 1 := by
  sorry

end exchange_rate_decrease_l764_764887


namespace ursula_change_l764_764862

theorem ursula_change : 
  let hot_dog_cost := 1.50
  let salad_cost := 2.50
  let hot_dogs_count := 5
  let salads_count := 3
  let bill_count := 2
  let bill_value := 10.00
  let total_hot_dog_cost := hot_dogs_count * hot_dog_cost
  let total_salad_cost := salads_count * salad_cost
  let total_purchase_cost := total_hot_dog_cost + total_salad_cost
  let total_money := bill_count * bill_value
  let change_received := total_money - total_purchase_cost
  in change_received = 5.00 :=
by
  sorry

end ursula_change_l764_764862


namespace number_of_true_propositions_l764_764569

noncomputable def proposition1 (α β r : Plane) : Prop := 
  (α ⊥ r ∧ β ⊥ r) → (α ∥ β)

noncomputable def proposition2 (α β r : Plane) : Prop := 
  (α ∥ r ∧ β ∥ r) → (α ∥ β)

noncomputable def proposition3 (α β : Plane) (l : Line) : Prop := 
  (α ⊥ l ∧ β ⊥ l) → (α ∥ β)

noncomputable def proposition4 (α β : Plane) (l m : Line) : Prop := 
  (skew l m ∧ l ∥ α ∧ m ∥ α ∧ l ∥ β ∧ m ∥ β) → (α ∥ β)

theorem number_of_true_propositions (α β r : Plane) (l m : Line) :
  (¬proposition1 α β r) ∧ (proposition2 α β r) ∧ (proposition3 α β l) ∧
  (proposition4 α β l m) → 
  3 = countp (λ p: Prop, p = proposition2 α β r ∨ p = proposition3 α β l ∨ p = proposition4 α β l m) [true, true, true, false] := sorry

end number_of_true_propositions_l764_764569


namespace intersections_of_altitudes_locus_l764_764847

variables (A B C : Point) (p1 p2 : Line)
hypothesis (parallel_lines : p1 ∥ p2)
hypothesis (A_on_p1 : A ∈ p1)
hypothesis (B_on_p1 : B ∈ p1)
hypothesis (C_on_p2 : C ∈ p2)

theorem intersections_of_altitudes_locus :
  ∃ l : Line, is_perpendicular l BC ∧ A ∈ l := 
sorry 

end intersections_of_altitudes_locus_l764_764847


namespace determine_coefficients_l764_764889

noncomputable def polynomial_coefficients (A B C D : ℝ) : Prop :=
  let P := (λ x, x^6 + 4 * x^5 + A * x^4 + B * x^3 + C * x^2 + D * x + 1)
  let Q := (λ x, x^6 - 4 * x^5 + A * x^4 - B * x^3 + C * x^2 - D * x + 1)
  ∃ b : ℝ, P = (λ x, (x^3 + 2 * x^2 + b * x + 1)^2) ∧ Q = (λ x, (x^3 - 2 * x^2 + b * x - 1)^2)

theorem determine_coefficients (A B C D : ℝ) :
  polynomial_coefficients A B C D → 
  (A = 8 ∧ B = 10 ∧ C = 8 ∧ D = 4) := 
sorry

end determine_coefficients_l764_764889


namespace magnitude_of_z_l764_764715

theorem magnitude_of_z (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = 1 / Real.sqrt 5 := by
  sorry

end magnitude_of_z_l764_764715


namespace estimated_number_of_red_balls_l764_764738

theorem estimated_number_of_red_balls (total_balls : ℕ) (red_draws : ℕ) (total_draws : ℕ)
    (h_total_balls : total_balls = 8) (h_red_draws : red_draws = 75) (h_total_draws : total_draws = 100) :
    total_balls * (red_draws / total_draws : ℚ) = 6 := 
by
  sorry

end estimated_number_of_red_balls_l764_764738


namespace change_received_l764_764408

def totalCostBeforeDiscount : ℝ :=
  5.75 + 2.50 + 3.25 + 3.75 + 4.20

def discount : ℝ :=
  (3.75 + 4.20) * 0.10

def totalCostAfterDiscount : ℝ :=
  totalCostBeforeDiscount - discount

def salesTax : ℝ :=
  totalCostAfterDiscount * 0.06

def finalTotalCost : ℝ :=
  totalCostAfterDiscount + salesTax

def amountPaid : ℝ :=
  50.00

def change : ℝ :=
  amountPaid - finalTotalCost

theorem change_received (h : change = 30.34) : change = 30.34 := by
  sorry

end change_received_l764_764408


namespace profit_with_discount_is_23_5_percent_l764_764920

def calculate_percentage_profit_with_discount (CP SP SP_discount : ℝ) (CP_positive : CP > 0) (SP_eq : SP = CP * 1.30) (SP_discount_eq : SP_discount = SP * 0.95) : ℝ :=
  (SP_discount - CP) / CP * 100

theorem profit_with_discount_is_23_5_percent (CP : ℝ) (CP_positive : CP > 0) (assumed_CP : CP = 100)
  (SP : ℝ) (SP_eq : SP = CP * 1.30)
  (SP_discount : ℝ) (SP_discount_eq : SP_discount = SP * 0.95) :
  calculate_percentage_profit_with_discount CP SP SP_discount CP_positive SP_eq SP_discount_eq = 23.5 :=
by
  have h1 : SP = CP * 1.30 := SP_eq
  have h2 : SP_discount = SP * 0.95 := SP_discount_eq
  have : calculate_percentage_profit_with_discount CP SP SP_discount CP_positive SP_eq SP_discount_eq = ((SP_discount - CP) / CP) * 100 := by
    apply rfl
  sorry

end profit_with_discount_is_23_5_percent_l764_764920


namespace integral_result_l764_764011

noncomputable def integral_expression (θ : ℝ) : ℝ := 
  ∫ x in (0 : ℝ)..(2 * Real.pi), sin (8 * x) * |sin (x - θ)|

theorem integral_result (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi) : 
  integral_expression θ = -(4 / 63) * sin (8 * θ) :=
by
  sorry

end integral_result_l764_764011


namespace smallest_n_l764_764607

theorem smallest_n (n : ℕ) : n > 1 ∧ (∃ a : ℕ → ℕ, (∀ i, a i > 0) ∧ (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2 + (a 6)^2 + (a 7)^2 + (a 8)^2 + (a 9)^2 ∣ ((a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) + (a 9))^2 - 1) :=
begin
  have h_n : n = 9,
  { -- Prove n = 9
    sorry
  },
  have h_exists : ∃ a : ℕ → ℕ, (∀ i, a i > 0) ∧ (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2 + (a 6)^2 + (a 7)^2 + (a 8)^2 + (a 9)^2 ∣ ((a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) + (a 9))^2 - 1,
  { -- Provide the integers a_1, ..., a_9
    sorry
  },
  split,
  { exact h_n,
  },
  { exact h_exists,
  },
end

end smallest_n_l764_764607


namespace calculate_selling_price_l764_764074

theorem calculate_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1500 → 
  loss_percentage = 0.17 →
  selling_price = cost_price - (loss_percentage * cost_price) →
  selling_price = 1245 :=
by 
  intros hc hl hs
  rw [hc, hl] at hs
  norm_num at hs
  exact hs

end calculate_selling_price_l764_764074


namespace roots_equiv_l764_764395

open Real

noncomputable def f (x c : ℝ) := x^2 + 4 * x + c

theorem roots_equiv (c : ℝ) : 
  (∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ ∀ x : ℝ, f (f x c) c = 0 → (x = r1 ∨ x = r2)) ↔ c = 1 - sqrt (13 : ℝ) := 
begin
  sorry
end

end roots_equiv_l764_764395


namespace remaining_dogs_after_adoption_l764_764472

variable (initial_dogs additional_dogs adopted_first_week adopted_later : ℕ)

theorem remaining_dogs_after_adoption :
  initial_dogs = 200 →
  additional_dogs = 100 →
  adopted_later = 60 →
  (300 - (adopted_first_week + adopted_later) = 240 - adopted_first_week) := by
    intros h_initial h_additional h_later
    calc
      300 - (adopted_first_week + adopted_later)
          = 300 - adopted_first_week - adopted_later : by rw [sub_sub_assoc _ _ _ rfl]
      ... = 240 - adopted_first_week : by rw [← h_later]; ring


end remaining_dogs_after_adoption_l764_764472


namespace solve_problem_l764_764961

noncomputable def problem_statement : ℝ :=
  limit (fun h => 
    (sin (π / 3 + 4 * h) - 4 * sin (π / 3 + 3 * h) + 6 * sin (π / 3 + 2 * h) 
    - 4 * sin (π / 3 + h) + sin (π / 3)) / h^4) (0)

theorem solve_problem : problem_statement = sqrt 3 / 2 :=
by
  sorry

end solve_problem_l764_764961


namespace sum_of_repeating_decimals_l764_764771

def repeating_decimal_sum := 
  let T := {x : ℝ | ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999}
  ∑ x in T, x = 908.208208208208

theorem sum_of_repeating_decimals :
  repeating_decimal_sum := sorry

end sum_of_repeating_decimals_l764_764771


namespace sum_of_T_l764_764760

def is_repeating_decimal (x : ℝ) :=
  ∃ (a b c d : ℕ), (∀ i j, {i, j} ⊆ {a, b, c, d} → i ≠ j) ∧
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : set ℝ := { x | is_repeating_decimal x }

theorem sum_of_T : ∑ x in T, x = 280 := sorry

end sum_of_T_l764_764760


namespace trig_proof_l764_764711

noncomputable def alpha : ℝ := sorry

theorem trig_proof (h₀ : 0 < alpha ∧ alpha < π / 2) 
                   (h₁ : sin(alpha)^2 + cos(2 * alpha) = 1 / 4) :
  tan(alpha) = sqrt(3) := sorry

end trig_proof_l764_764711


namespace sum_of_T_l764_764762

def is_repeating_decimal (x : ℝ) :=
  ∃ (a b c d : ℕ), (∀ i j, {i, j} ⊆ {a, b, c, d} → i ≠ j) ∧
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : set ℝ := { x | is_repeating_decimal x }

theorem sum_of_T : ∑ x in T, x = 280 := sorry

end sum_of_T_l764_764762


namespace rope_touching_tower_sum_is_813_l764_764929

-- Definitions based on conditions
def radius_tower : ℝ := 8
def height_unicorn : ℝ := 4
def distance_rope_end_to_tower : ℝ := 4
def total_rope_length : ℝ := 20

-- Main statement to prove
theorem rope_touching_tower_sum_is_813 (a b c : ℕ) (prime_c : Nat.Prime c) :
  let rope_touching_tower := (total_rope_length : ℝ) - (Real.sqrt (radius_tower^2 + (total_rope_length^2 - (radius_tower + distance_rope_end_to_tower)^2)) / c) in
  rope_touching_tower = (a - Real.sqrt b) / c →
  a + b + c = 813 :=
begin
  sorry
end

end rope_touching_tower_sum_is_813_l764_764929


namespace actual_percent_profit_l764_764880

-- Define the cost price (CP)
def CP : ℝ := 100

-- Define the labeled price (LP) as 50% profit on CP
def LP : ℝ := CP + 0.50 * CP

-- Define the selling price (SP) as LP with a 10% discount
def SP : ℝ := LP - 0.10 * LP

-- Define the actual profit
def Profit : ℝ := SP - CP

-- Define the actual percent profit
def PercentProfit : ℝ := (Profit / CP) * 100

-- The theorem to prove
theorem actual_percent_profit : PercentProfit = 35 := by
  sorry

end actual_percent_profit_l764_764880


namespace negation_of_universal_proposition_trigonometric_identity_projection_of_vector_interval_of_decrease_l764_764525

-- Problem 1
theorem negation_of_universal_proposition (p : Prop) (H : ¬ (∀ n : ℕ, n!^2 < 2^n)) :
  ∃ n0 : ℕ, n0!^2 ≥ 2^n0 := sorry

-- Problem 2
theorem trigonometric_identity : cos (75 * π / 180) * cos (15 * π / 180) - sin (255 * π / 180) * sin (165 * π / 180) = 1 / 2 := sorry

-- Problem 3
def vector_project (m n : ℝ × ℝ) : ℝ :=
  let ⟨m1, m2⟩ := m
  let ⟨n1, n2⟩ := n
  (m1 * n1 + m2 * n2) / (real.sqrt (n1^2 + n2^2))

theorem projection_of_vector :
  vector_project (1, 2) (2, 3) = (8 * real.sqrt 13) / 13 := sorry

-- Problem 4
def decreasing_interval (f : ℝ → ℝ) : set ℝ := 
  {x | x > 0 ∧ (f x)' < 0}

theorem interval_of_decrease :
  decreasing_interval (fun x => 2*x^2 - real.log x) = set.Ioc 0 (1/2) := sorry

end negation_of_universal_proposition_trigonometric_identity_projection_of_vector_interval_of_decrease_l764_764525


namespace roots_of_quadratic_eq_l764_764467

theorem roots_of_quadratic_eq : (x : ℝ) → x * (x - 1) = 0 → x = 0 ∨ x = 1 :=
by
  intro x h
  have h₁ := eq_zero_or_eq_zero_of_mul_eq_zero h
  cases h₁
  · exact Or.inl h₁
  · exact Or.inr (eq_of_sub_eq_zero h₁)
  sorry

end roots_of_quadratic_eq_l764_764467


namespace sum_of_nine_pointed_star_tips_l764_764409

open Float Real

def evenly_spaced_points_on_circle (n : ℕ) (total_degrees : ℝ) := 
  (0 : ℕ) < n ∧ total_degrees = 360 ∧ (total_degrees / n : ℝ) = 40

def nine_point_star := (total_points : ℕ) := 
  evenly_spaced_points_on_circle total_points 360 ∧ total_points = 9 

theorem sum_of_nine_pointed_star_tips 
  (h : nine_point_star 9): 
  (9 * ((360 : ℝ) - 2 * (40 : ℝ * 4))) = 1440 := 
by 
  sorry

end sum_of_nine_pointed_star_tips_l764_764409


namespace ratio_youngest_sister_to_yvonne_l764_764511

def laps_yvonne := 10
def laps_joel := 15
def joel_ratio := 3

theorem ratio_youngest_sister_to_yvonne
  (laps_yvonne : ℕ)
  (laps_joel : ℕ)
  (joel_ratio : ℕ)
  (H_joel : laps_joel = 3 * (laps_yvonne / joel_ratio))
  : (laps_joel / joel_ratio) = laps_yvonne / 2 :=
by
  sorry

end ratio_youngest_sister_to_yvonne_l764_764511


namespace min_third_side_length_l764_764348

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end min_third_side_length_l764_764348


namespace find_initial_population_l764_764527

-- Definitions based on the given conditions
def initial_population (P : ℕ) : Prop :=
  let pop_after_bombardment := 0.95 * P in
  let final_population := 0.85 * pop_after_bombardment in
  final_population = 3294

-- Statement to prove
theorem find_initial_population : ∃ P : ℕ, initial_population P ∧ P = 4080 :=
by
  use 4080
  unfold initial_population
  have pop_after_bombardment_eq : 0.95 * 4080 = 3876 := by
    sorry
  have final_population_eq : 0.85 * 3876 = 3294 := by 
    sorry
  split
  · exact final_population_eq
  · rfl

end find_initial_population_l764_764527


namespace shortest_distance_curve_line_l764_764090

theorem shortest_distance_curve_line :
  let curve := λ x : ℝ, x^2 + x - Real.log x in
  let line := λ x y : ℝ, 2 * x - y - 2 = 0 in
  ∀ P : ℝ × ℝ, P.2 = curve P.1 →
    ∃ Q : ℝ × ℝ, Q ∈ metric.closest_point {Q : ℝ × ℝ | line Q.1 Q.2} P ∧
    dist P Q = 2 * Real.sqrt 5 / 5 :=
sorry

end shortest_distance_curve_line_l764_764090


namespace series_pattern_l764_764797

theorem series_pattern :
    (3 / (1 * 2) * (1 / 2) + 4 / (2 * 3) * (1 / 2^2) + 5 / (3 * 4) * (1 / 2^3) + 6 / (4 * 5) * (1 / 2^4) + 7 / (5 * 6) * (1 / 2^5)) 
    = (1 - 1 / (6 * 2^5)) :=
  sorry

end series_pattern_l764_764797


namespace smallest_perimeter_correct_l764_764965

noncomputable def smallest_perimeter_obtuse_triangle : ℕ × ℕ × ℕ :=
  (7, 9, 12)

theorem smallest_perimeter_correct :
  ∀ (a b c : ℕ), 
    a < b ∧ b < c ∧ c^2 = b * (a + b) ∧
    2 * (real.arccos (a^2 + c^2 - b^2) / (2 * a * c)) > real.pi / 2 ∧
    (a + b + c).nat_abs ≤ 28 → (a, b, c) = (7, 9, 12) :=
by sorry

end smallest_perimeter_correct_l764_764965


namespace max_radius_sphere_in_prism_l764_764833

noncomputable def maxSphereRadiusInPrism (height : ℝ) (edge : ℝ) : ℝ :=
  (edge * Real.sqrt 3) / 6

theorem max_radius_sphere_in_prism :
  ∀ (height edge : ℝ), height = 5 ∧ edge = 4 * Real.sqrt 3 
  → maxSphereRadiusInPrism height edge = 2 :=
by
  intros height edge h
  cases h with height_eq edge_eq
  sorry

end max_radius_sphere_in_prism_l764_764833


namespace natural_numbers_with_2019_divisors_l764_764605

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop := ∃ (a : ℕ), n = a * a
def num_divisors (n : ℕ) : ℕ := (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card

-- State the proof problem
theorem natural_numbers_with_2019_divisors (n : ℕ) :
  n < 2^679 ∧ num_divisors n = 2019 ↔ 
  n = 2^672 * 3^2 ∨ n = 2^672 * 5^2 ∨ n = 2^672 * 7^2 ∨ n = 2^672 * 11^2 := 
by 
  sorry

end natural_numbers_with_2019_divisors_l764_764605


namespace unknown_rate_of_blankets_l764_764907

theorem unknown_rate_of_blankets (x : ℝ) :
  2 * 100 + 5 * 150 + 2 * x = 9 * 150 → x = 200 :=
by
  sorry

end unknown_rate_of_blankets_l764_764907


namespace sum_elements_T_l764_764765

noncomputable def T := {x : ℝ | ∃ a b c d : ℕ, 0 ≤ a ∧ a < 10 ∧ 
                                   0 ≤ b ∧ b < 10 ∧ 
                                   0 ≤ c ∧ c < 10 ∧ 
                                   0 ≤ d ∧ d < 10 ∧ 
                                   a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 
                                   d ≠ b ∧ d ≠ c ∧ x = a * 10^3 + b * 10^2 + c * 10 + d }

theorem sum_elements_T : ∑ x in T, x = 28028 / 111 :=
by
    sorry

end sum_elements_T_l764_764765


namespace find_height_BF_l764_764651

variables (A B C E F : Point)
variables (α β : Angle)
variables (x y : ℝ)

-- Defining the key conditions
def center_on_ray_BE (O : Point) : Prop := on_ray B E O
def product_AF_FE : Prop := AF * FE = 5
def cot_ratio_condition : Prop := cot α / cot β = 3 / 4

-- Angle assignments
def angle_EBC := α
def angle_BEC := β

-- Target height to prove
def height_BF : ℝ := sqrt(15) / 2

-- Formal statement of the theorem/problem
theorem find_height_BF
    (h_center : ∃ (O : Point), center_on_ray_BE O)
    (h_product : product_AF_FE)
    (h_cot_ratio : cot_ratio_condition)
    : distance B F = height_BF :=
sorry

end find_height_BF_l764_764651


namespace problem1_problem2_l764_764318

-- Definitions from the conditions
def A (x : ℝ) : Prop := -1 < x ∧ x < 3

def B (x m : ℝ) : Prop := x^2 - 2 * m * x + m^2 - 1 < 0

-- Intersection problem
theorem problem1 (h₁ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₂ : ∀ x, B x 3 ↔ (2 < x ∧ x < 4)) :
  ∀ x, (A x ∧ B x 3) ↔ (2 < x ∧ x < 3) := by
  sorry

-- Union problem
theorem problem2 (h₃ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₄ : ∀ x m, B x m ↔ ((x - m)^2 < 1)) :
  ∀ m, (0 ≤ m ∧ m ≤ 2) ↔ (∀ x, A x ∨ B x m → A x) := by
  sorry

end problem1_problem2_l764_764318


namespace november_has_five_mondays_l764_764430

open Nat 

variable (M : ℕ)

def days_in_november : ℕ := 30

def friday_dates_in_october : List (List ℕ) :=
  [[1, 8, 15, 22, 29], [2, 9, 16, 23, 30], [3, 10, 17, 24, 31]]

def date_of_october : List ℕ → ℕ
| [1, 8, 15, 22, 29] := 31 -- Sunday
| [2, 9, 16, 23, 30] := 30 -- Saturday
| [3, 10, 17, 24, 31] := 31 -- Friday
| _ := 0

def day_of_november (oct_date : ℕ) : ℕ :=
oct_date % 7 -- Assuming Sunday as 0 and so on

def count_mondays (start_day: ℕ) : ℕ :=
if start_day = 1 ∨ start_day = 0 then 5 else 4

theorem november_has_five_mondays (M : ℕ) (h : ∃ d ∈ friday_dates_in_october, date_of_october d = 31 → day_of_november 31 % 7 ∈ [0, 1]) :
  count_mondays ((day_of_november (date_of_october 31)) % 7) = 5 :=
by sorry

end november_has_five_mondays_l764_764430


namespace scientific_notation_conversion_l764_764848

theorem scientific_notation_conversion (h : ℝ) (h_def : h = 0.0000046) : h = 4.6 * 10^(-6) :=
sorry

end scientific_notation_conversion_l764_764848


namespace jenny_research_time_l764_764381

noncomputable def time_spent_on_research (total_hours : ℕ) (proposal_hours : ℕ) (report_hours : ℕ) : ℕ :=
  total_hours - proposal_hours - report_hours

theorem jenny_research_time : time_spent_on_research 20 2 8 = 10 := by
  sorry

end jenny_research_time_l764_764381


namespace sphere_surface_area_of_solid_l764_764172

theorem sphere_surface_area_of_solid (l w h : ℝ) (hl : l = 2) (hw : w = 1) (hh : h = 2) 
: 4 * Real.pi * ((Real.sqrt (l^2 + w^2 + h^2) / 2)^2) = 9 * Real.pi := 
by 
  sorry

end sphere_surface_area_of_solid_l764_764172


namespace monkey_climb_time_l764_764163

theorem monkey_climb_time : 
  ∀ (height hop slip : ℕ), 
    height = 22 ∧ hop = 3 ∧ slip = 2 → 
    ∃ (time : ℕ), time = 20 := 
by
  intros height hop slip h
  rcases h with ⟨h_height, ⟨h_hop, h_slip⟩⟩
  sorry

end monkey_climb_time_l764_764163


namespace find_total_reactions_l764_764532

def total_reactions (x : ℝ) (n : ℕ) : Prop :=
  2 + 2.1 + 2 + 2.2 + x = 2 * n

theorem find_total_reactions : ∃ (n : ℕ) (x : ℝ), 2 + 2.1 + 2 + 2.2 + x = 2 * n ∧ n = 5 :=
by
  -- Define x
  let x := 1.7
  -- Define sum of known readings
  have sum_known_readings : ℝ := 2 + 2.1 + 2 + 2.2
  -- Prove that total sum matches 2 * n when n = 5 and x = 1.7
  have : sum_known_readings + x = 2 * 5 :=
    calc
      sum_known_readings + x = 8.3 + 1.7 : by sorry
      ... = 10 : by sorry
      ... = 2 * 5 : by sorry
  -- Provide the required proof
  exact ⟨5, 1.7, by sorry, rfl⟩

end find_total_reactions_l764_764532


namespace hall_area_l764_764846

theorem hall_area 
  (L W : ℝ)
  (h1 : W = 1/2 * L)
  (h2 : L - W = 10) : 
  L * W = 200 := 
sorry

end hall_area_l764_764846


namespace first_term_and_general_formula_sum_of_b_seq_range_of_a_l764_764663

variable {ℕ : Type*}
variable {ℝ : Type*}

-- Define the sequences and their conditions
noncomputable def a_seq (n : ℕ) : ℝ := n
def S (n : ℕ) : ℝ := ∑ k in range(n+1), a_seq k

-- Given sequences and conditions
noncomputable def b_seq (n : ℕ) : ℝ := (2 * a_seq n - 1) * 2^n

noncomputable def T (n : ℕ) : ℝ :=
  ∑ k in range(n+1), b_seq k

noncomputable def c_seq (n : ℕ) : ℝ :=
  (4 * n - 6) / (T n - 6) - (1 / (a_seq n * a_seq (n + 1)))

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) * x^2 + (1 / 2) * x

-- The problem is to prove that these correct answers hold
theorem first_term_and_general_formula :
  a_seq 1 = 1 ∧ (∀ n, a_seq n = n) :=
by sorry

theorem sum_of_b_seq :
  ∀ n, T n = 6 + (2 * n - 3) * 2^(n + 1) :=
by sorry

theorem range_of_a :
  ∀ n x (h : x ∈ Icc (-1/2) (1/2)),
  ∃ a, (∑ k in range(n+1), c_seq k ≤ f x - a) ↔ a ≤ 19 / 80 :=
by sorry

end first_term_and_general_formula_sum_of_b_seq_range_of_a_l764_764663


namespace solveProblem_l764_764668

/-
Proof Problem: 
Prove the following:
1. Given an ellipse described by \(\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1\) where \(a > b > 0\), with the left focal point at \(F_1(-1, 0)\) and a vertex \(P\) on the ellipse such that \(\angle PF_1O = 45^\circ\), we can determine the values of \(a\) and \(b\).
2. Given line \(l_1 : y = kx + m_1\) intersects the ellipse at points \(A, B\), and line \(l_2 : y = kx + m_2\) (where \(m_1 \neq m_2\)) intersects the ellipse at points \(C, D\), and \(|AB| = |CD|\),
  (i) \(m_1 + m_2 = 0\).
  (ii) The maximum area \(S\) of quadrilateral \(ABCD\) is \(2\sqrt{2}\).
-/

noncomputable def problem (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (f1 : ℝ × ℝ := (-1, 0)) (θ : ℝ := 45) (hθ : θ = (π / 4)) : Prop :=
  (∃ (c : ℝ), c = 1 ∧ b = c ∧ a = sqrt 2) ∧
  (∀ (k m1 m2 : ℝ) (h_m1_ne_m2 : m1 ≠ m2) (AB CD : ℝ),
    (k ≠ 0 → (|AB| = |CD| → (m1 + m2 = 0 ∧ ∃ d : ℝ, d = (2 * sqrt 2))))
    
theorem solveProblem (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (f1 : ℝ × ℝ := (-1, 0)) (θ : ℝ := 45) (hθ : θ = (π / 4)) (k m1 m2 : ℝ) (h_m1_ne_m2 : m1 ≠ m2) (AB CD : ℝ) :
  problem a b a_pos b_pos f1 θ hθ :=
begin
  sorry,
end

end solveProblem_l764_764668


namespace range_of_a_for_monotonic_f_l764_764302

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 0 then a * x^2 + 1 else (a + 3) * real.exp(a * x)

theorem range_of_a_for_monotonic_f :
  ∀ (a : ℝ), (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ∨
             (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f a x₁ ≥ f a x₂) →
  a ∈ set.Ico (-2 : ℝ) (0 : ℝ) :=
by 
  sorry

end range_of_a_for_monotonic_f_l764_764302


namespace bg_fg_ratio_l764_764744

open Real

-- Given the lengths AB, BD, AF, DF, BE, CF
def AB : ℝ := 15
def BD : ℝ := 18
def AF : ℝ := 15
def DF : ℝ := 12
def BE : ℝ := 24
def CF : ℝ := 17

-- Prove that the ratio BG : FG = 27 : 17
theorem bg_fg_ratio (BG FG : ℝ)
  (h_BG_FG : BG / FG = 27 / 17) :
  BG / FG = 27 / 17 := by
  sorry

end bg_fg_ratio_l764_764744


namespace length_of_rectangle_l764_764444

-- Define the conditions as given in the problem
variables (width : ℝ) (perimeter : ℝ) (length : ℝ)

-- The conditions provided
def conditions : Prop :=
  width = 15 ∧ perimeter = 70 ∧ perimeter = 2 * (length + width)

-- The statement to prove: the length of the rectangle is 20 feet
theorem length_of_rectangle {width perimeter length : ℝ} (h : conditions width perimeter length) : length = 20 :=
by 
  -- This is where the proof steps would go
  sorry

end length_of_rectangle_l764_764444


namespace unique_real_solution_system_l764_764214

/-- There is exactly one real solution (x, y, z, w) to the given system of equations:
  x + 1 = z + w + z * w * x,
  y - 1 = w + x + w * x * y,
  z + 2 = x + y + x * y * z,
  w - 2 = y + z + y * z * w
-/
theorem unique_real_solution_system :
  let eq1 (x y z w : ℝ) := x + 1 = z + w + z * w * x
  let eq2 (x y z w : ℝ) := y - 1 = w + x + w * x * y
  let eq3 (x y z w : ℝ) := z + 2 = x + y + x * y * z
  let eq4 (x y z w : ℝ) := w - 2 = y + z + y * z * w
  ∃! (x y z w : ℝ), eq1 x y z w ∧ eq2 x y z w ∧ eq3 x y z w ∧ eq4 x y z w := by {
  sorry
}

end unique_real_solution_system_l764_764214


namespace solve_linear_eq_l764_764938

theorem solve_linear_eq (x y : ℤ) : 2 * x + 3 * y = 0 ↔ (x, y) = (3, -2) := sorry

end solve_linear_eq_l764_764938


namespace rectangle_diagonals_equal_l764_764142

-- Condition: Given quadrilateral ABCD is a rectangle
def is_rectangle (A B C D : Type) [has_diagonal A B C D] : Prop :=
  -- Assuming the quadrilateral is a rectangle
  A.rectangle ∧ B.rectangle ∧ C.rectangle ∧ D.rectangle

-- Theorem: Prove that the diagonals of the rectangle are equal
theorem rectangle_diagonals_equal (A B C D : Type) [has_diagonal A B C D] (h : is_rectangle A B C D) : 
  diagonals_equal A B C D :=
sorry

end rectangle_diagonals_equal_l764_764142


namespace Isabel_paper_left_l764_764380

theorem Isabel_paper_left (total_paper : ℕ) (used_paper : ℕ) (remaining_paper : ℕ) 
    (h1 : total_paper = 900) (h2 : used_paper = 156) : remaining_paper = total_paper - used_paper :=
sorry

example : Isabel_paper_left 900 156 744 rfl rfl :=
sorry

end Isabel_paper_left_l764_764380


namespace students_left_in_classroom_l764_764882

theorem students_left_in_classroom : 
  let total_students := 50
  let students_painting := (3 / 5) * total_students
  let students_playing := (1 / 5) * total_students
  students_painting + students_playing = total_students - 10 := 
by
  have h1 : students_painting = 30 := by sorry
  have h2 : students_playing = 10 := by sorry
  have h3 : total_students = 50 := rfl
  have h4 : students_painting + students_playing = 40 := by
    rw [h1, h2]
    exact (show 30 + 10 = 40 by rfl)
  show total_students - (students_painting + students_playing) = 10 from
    calc
      total_students - (students_painting + students_playing)
        = 50 - 40 : by rw [h3, ←h4]
    ... = 10 : rfl

end students_left_in_classroom_l764_764882


namespace number_of_points_on_ellipse_l764_764671

open Real

theorem number_of_points_on_ellipse
  (F1 F2 : ℝ × ℝ)
  (E : Set (ℝ × ℝ))
  {r : ℝ} (hmr : 2 * π * r = 3 * π) :
  let c := 3 in
  E = {p | (p.1^2 / 25 + p.2^2 / 16 = 1)} ∧
  |F1.1 - F2.1| = 6 → -- Distance between F1 and F2 is 6
  ∃! (M : ℝ × ℝ), M ∈ E ∧ 
                  by let MF1 := dist M F1
                     let MF2 := dist M F2
                     have h_sum : MF1 + MF2 = 10
                     have area_main_eq : 4 * r = 6 := -- Twice the semi-minor axis
                         by sorry
                     M.2 = 4 := -- y_M = 4
                     sorry :=

end number_of_points_on_ellipse_l764_764671


namespace perpendicular_tangent_line_l764_764626

theorem perpendicular_tangent_line (a b : ℝ) : 
  let line_slope := 1 / 3,
      perp_slope := -3,
      curve := λ x, x^3 + 3 * x^2 - 5,
      derivative_curve := λ x, 3 * x^2 + 6 * x,
      a_eqn := a = -1,
      b_eqn := b = -3
  in
  (2 * a - 6 * b + 1 = 0) ∧ (a, b) = P ∧ (derivative_curve a = -3) → 
  3 * x + y + 6 = 0 :=
by
  sorry

end perpendicular_tangent_line_l764_764626


namespace cranberries_picked_l764_764377

theorem cranberries_picked (C : ℕ) : 
  (let total_berries := 30 + C + 10 in
   let fresh_berries := (2 / 3 : ℚ) * total_berries in
   let berries_to_sell := (1 / 2 : ℚ) * fresh_berries in
   berries_to_sell = 20) → 
  C = 20 :=
by
  -- Let total_berries = 40 + C
  let total_berries := 30 + C + 10
  -- Let fresh_berries = 2 / 3 * total_berries
  let fresh_berries := (2 / 3 : ℚ) * total_berries
  -- Let berries_to_sell = 1 / 2 * fresh_berries
  let berries_to_sell := (1 / 2 : ℚ) * fresh_berries
  have h : berries_to_sell = 20 := by sorry
  sorry

end cranberries_picked_l764_764377


namespace one_fourth_of_8_point8_simplified_l764_764994

noncomputable def one_fourth_of (x : ℚ) : ℚ := x / 4

def convert_to_fraction (x : ℚ) : ℚ := 
  let num := 22
  let denom := 10
  num / denom

def simplify_fraction (num denom : ℚ) (gcd : ℚ) : ℚ := 
  (num / gcd) / (denom / gcd)

theorem one_fourth_of_8_point8_simplified : one_fourth_of 8.8 = (11 / 5) := 
by
  have h : one_fourth_of 8.8 = 2.2 := by sorry
  have h_frac : 2.2 = (22 / 10) := by sorry
  have h_simplified : (22 / 10) = (11 / 5) := by sorry
  rw [h, h_frac, h_simplified]
  exact rfl

end one_fourth_of_8_point8_simplified_l764_764994


namespace john_hiking_probability_l764_764008

theorem john_hiking_probability :
  let P_rain := 0.3
  let P_sunny := 0.7
  let P_hiking_if_rain := 0.1
  let P_hiking_if_sunny := 0.9

  let P_hiking := P_rain * P_hiking_if_rain + P_sunny * P_hiking_if_sunny

  P_hiking = 0.66 := by
    sorry

end john_hiking_probability_l764_764008


namespace tim_tasks_per_day_l764_764484

theorem tim_tasks_per_day (earnings_per_task : ℝ) (days_per_week : ℕ) (weekly_earnings : ℝ) :
  earnings_per_task = 1.2 ∧ days_per_week = 6 ∧ weekly_earnings = 720 → (weekly_earnings / days_per_week / earnings_per_task = 100) :=
by
  sorry

end tim_tasks_per_day_l764_764484


namespace identify_roles_l764_764496

def Role := { knight : Prop // knight = true
            ∨ knight = false }

structure Person :=
(role : Role)
(statement : Prop)

constant A B C : Person
constant knight liar normal : Person → Prop

axiom A_statement : A.statement = (normal A)
axiom B_statement : B.statement = true
axiom C_statement : C.statement = ¬ (normal C)

axiom unique_roles : 
  (knight A ∧ liar B ∧ normal C)
  ∨ (knight A ∧ liar C ∧ normal B)
  ∨ (knight B ∧ liar A ∧ normal C)
  ∨ (knight B ∧ liar C ∧ normal A)
  ∨ (knight C ∧ liar A ∧ normal B)
  ∨ (knight C ∧ liar B ∧ normal A)

theorem identify_roles : 
  (liar A ∧ knight B ∧ normal C) :=
by {
  -- Proof steps would go here.
  sorry
}

end identify_roles_l764_764496


namespace simplify_product_l764_764055

theorem simplify_product : 
  (∀ (n: ℕ), n ≥ 1 →
    let term := (∏ (k: ℕ) in (finset.range n).filter (λ k, k > 0), ((3 * k + 6) / (3 * k))) in
    term = (3003 / 3)) → 
  ∑ (k : ℕ) in finset.range 1001, k = 1001 :=
sorry

end simplify_product_l764_764055


namespace newborn_members_approximation_l764_764360

-- Defining the conditions
def survival_prob_first_month : ℚ := 7/8
def survival_prob_second_month : ℚ := 7/8
def survival_prob_third_month : ℚ := 7/8
def survival_prob_three_months : ℚ := (7/8) ^ 3
def expected_survivors : ℚ := 133.984375

-- Statement to prove that the number of newborn members, N, approximates to 200
theorem newborn_members_approximation (N : ℚ) : 
  N * survival_prob_three_months = expected_survivors → 
  N = 200 :=
by
  sorry

end newborn_members_approximation_l764_764360


namespace tangent_line_at_1_range_of_a_l764_764696

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - Real.log x - 1

-- Part 1: Prove the tangent line equation at x = 1
theorem tangent_line_at_1 : ∀ (x : ℝ), (2 * Real.exp 1 - 1) * x - Real.exp 1 = f 1 + (x - 1) * f.deriv.eval 1 := by
  sorry

-- Part 2: Prove the range of a for f(x) ≥ ax for all x ≥ 0
theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 0 → f x ≥ a * x) ↔ a ≤ 1 := by
  sorry

end tangent_line_at_1_range_of_a_l764_764696


namespace find_fraction_identity_l764_764786

variable (x y z : ℝ)

theorem find_fraction_identity
 (h1 : 16 * y^2 = 15 * x * z)
 (h2 : y = 2 * x * z / (x + z)) :
 x / z + z / x = 34 / 15 := by
-- proof skipped
sorry

end find_fraction_identity_l764_764786


namespace teddy_uses_2_pounds_per_pillow_l764_764820

/-- Teddy uses a certain amount of fluffy foam material to make each pillow. 
 1. Teddy has three tons of fluffy foam material.
 2. There are 2,000 pounds in a ton.
 3. Teddy can make 3,000 pillows with the fluffy foam material.
 Prove that Teddy uses 2 pounds of fluffy foam material to make each pillow. -/
theorem teddy_uses_2_pounds_per_pillow :
  ∀ (tons_of_material : ℕ) (pounds_per_ton : ℕ) (number_of_pillows : ℕ),
  tons_of_material = 3 →
  pounds_per_ton = 2000 →
  number_of_pillows = 3000 →
  (tons_of_material * pounds_per_ton) / number_of_pillows = 2 :=
begin
  intros tons_of_material pounds_per_ton number_of_pillows h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry,
end

end teddy_uses_2_pounds_per_pillow_l764_764820


namespace find_minimum_value_l764_764294

noncomputable def f (x : ℝ) : ℝ := x * 2^x

theorem find_minimum_value : ∃ x : ℝ, (f x = inf (set.range f)) ∧ x = -real.logb 2 (real.exp 1) :=
sorry

end find_minimum_value_l764_764294


namespace maria_average_speed_l764_764035

theorem maria_average_speed:
  let distance1 := 180
  let time1 := 4.5
  let distance2 := 270
  let time2 := 5.25
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time = 46.15 := by
  -- Sorry to skip the proof
  sorry

end maria_average_speed_l764_764035


namespace ratio_jl_jm_l764_764807

-- Define the side length of the square NOPQ as s
variable (s : ℝ)

-- Define the length (l) and width (m) of the rectangle JKLM
variable (l m : ℝ)

-- Conditions given in the problem
variable (area_overlap : ℝ)
variable (area_condition1 : area_overlap = 0.25 * s * s)
variable (area_condition2 : area_overlap = 0.40 * l * m)

theorem ratio_jl_jm (h1 : area_overlap = 0.25 * s * s) (h2 : area_overlap = 0.40 * l * m) : l / m = 2 / 5 :=
by
  sorry

end ratio_jl_jm_l764_764807


namespace drop_in_water_level_l764_764716

theorem drop_in_water_level (rise_level : ℝ) (drop_level : ℝ) 
  (h : rise_level = 1) : drop_level = -2 :=
by
  sorry

end drop_in_water_level_l764_764716


namespace count_two_digit_numbers_l764_764330

theorem count_two_digit_numbers : (99 - 10 + 1) = 90 := by
  sorry

end count_two_digit_numbers_l764_764330


namespace compute_u_l764_764705

noncomputable def a : ℝ × ℝ × ℝ := ⟨2, 3, -1⟩
noncomputable def b : ℝ × ℝ × ℝ := ⟨1, -1, 2⟩
noncomputable def targetVector : ℝ × ℝ × ℝ := ⟨5, -2, 3⟩

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.2 * v.3 - u.3 * v.2.2,
   u.3 * v.1 - u.1 * v.3,
   u.1 * v.2.2 - u.2.2 * v.1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2.2 * v.2.2 + u.3 * v.3

def find_scalar_u (a b target : ℝ × ℝ × ℝ) : ℝ :=
  let ab_cross := cross_product a b
  in (dot_product ab_cross target) / (dot_product ab_cross ab_cross)

theorem compute_u : find_scalar_u a b targetVector = 16 / 59 :=
by
  sorry

end compute_u_l764_764705


namespace complement_union_in_U_l764_764722

def U := {1, 2, 3, 4}
def M := {1, 2}
def N := {2, 3}

theorem complement_union_in_U : (U \ (M ∪ N)) = {4} :=
by
  sorry

end complement_union_in_U_l764_764722


namespace sum_of_distinct_primes_even_l764_764724

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def even (n : ℕ) : Prop := n % 2 = 0

def is_sum_even (a b : ℕ) : Prop := even (a + b)

def num_combinations (n k : ℕ) : ℕ :=
  n.choose k

def num_even_sum_pairs : ℕ :=
  (first_seven_primes.filter even).length * 
  (first_seven_primes.filter (fun p => ¬even p)).length

def total_combinations : ℕ := num_combinations 7 2

def probability_even_sum : ℚ :=
  (total_combinations - num_even_sum_pairs) / total_combinations

theorem sum_of_distinct_primes_even :
  probability_even_sum = 5 / 7 := 
  by
  sorry

end sum_of_distinct_primes_even_l764_764724


namespace lines_through_M_integer_length_chords_l764_764161

def point_M : ℝ × ℝ := (2, 2)
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 25

theorem lines_through_M_integer_length_chords :
  ∃ l : ℝ → Prop, ∃ M : ℝ × ℝ, ∃ n : ℕ,
    M = point_M ∧
    (∀ p : ℝ × ℝ, (∃ q : ℝ × ℝ, p ≠ q ∧ l p ∧ l q ∧ circle_eq (fst p) (snd p) ∧ circle_eq (fst q) (snd q)) → n = 8) :=
sorry

end lines_through_M_integer_length_chords_l764_764161


namespace domain_of_function_l764_764445

theorem domain_of_function :
  (∀ x : ℝ, 2 + x ≥ 0 ∧ 3 - x ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry

end domain_of_function_l764_764445


namespace arithmetic_square_root_of_9_l764_764824

theorem arithmetic_square_root_of_9 : ∃ y : ℕ, y^2 = 9 ∧ y = 3 :=
by
  sorry

end arithmetic_square_root_of_9_l764_764824


namespace mod7_sum198_l764_764115

-- Define the sum of numbers from 1 to 198
def sum198 : ℕ := ∑ k in Finset.range (198 + 1), k

-- Statement of the problem to be proven: sum modulo 7 is 3
theorem mod7_sum198 : sum198 % 7 = 3 :=
sorry

end mod7_sum198_l764_764115


namespace find_x_l764_764235

noncomputable def x (x : ℝ) : Prop :=
  (⌈x⌉₊ * x = 210)

theorem find_x : ∃ x : ℝ, x = 14 ∧ x (14) :=
  by
    sorry

end find_x_l764_764235


namespace circles_intersection_points_l764_764104

theorem circles_intersection_points (
  A1 A2 B1 B2 C1 C2 : Point,
  circle1 : Circle,
  circle2 : Circle,
  circle3 : Circle,
  h1 : A1 ∈ circle1,
  h2 : A2 ∈ circle1,
  h3 : B1 ∈ circle2,
  h4 : B2 ∈ circle2,
  h5 : C1 ∈ circle3,
  h6 : C2 ∈ circle3,
  hA : LineThrough(A1, A2) ∩ circle2 ≠ ∅,
  hB : LineThrough(B1, B2) ∩ circle3 ≠ ∅,
  hC : LineThrough(C1, C2) ∩ circle1 ≠ ∅
) : 
  A1.dist(B2) * B1.dist(C2) * C1.dist(A2) = A2.dist(B1) * B2.dist(C1) * C2.dist(A1) := 
sorry

end circles_intersection_points_l764_764104


namespace min_f_range_a_l764_764694

-- Define the function f
def f (x : ℝ) : ℝ := x * Real.log x

-- Minimum value of f(x)
theorem min_f : ∃ x, f x = -1 / Real.exp 1 :=
by
  sorry

-- Range of the real number a
theorem range_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → f x ≥ a * x - 1) ↔ a ∈ Set.Iic 1 :=
by
  sorry

end min_f_range_a_l764_764694


namespace largest_angle_l764_764070

noncomputable def altitudes := (10, 24, 15)

theorem largest_angle (h1: (10, 24, 15) = altitudes) : 
  ∃ A B C : ℝ, ∃ d1 d2 d3 : ℝ,
  (d1, d2, d3) = altitudes ∧ 
  (10: ℝ) = d1 ∧ 
  (24: ℝ) = d2 ∧ 
  (15: ℝ) = d3 ∧ 
  ∃ largest_angle : ℝ, largest_angle = 100 :=
sorry

end largest_angle_l764_764070


namespace area_square_field_l764_764517

-- Define the side length of the square
def side_length : ℕ := 12

-- Define the area of the square with the given side length
def area_of_square (side : ℕ) : ℕ := side * side

-- The theorem to state and prove
theorem area_square_field : area_of_square side_length = 144 :=
by
  sorry

end area_square_field_l764_764517


namespace g_three_value_l764_764079

-- Function g that satisfies the given condition
def g (x : ℝ) := sorry

theorem g_three_value : g(3) = -((27:ℝ) + 3 * 3^(1/3)) / 8 :=
by
  sorry

end g_three_value_l764_764079


namespace required_total_money_l764_764150

def bundle_count := 100
def number_of_bundles := 10
def bill_5_value := 5
def bill_10_value := 10
def bill_20_value := 20

-- Sum up the total money required to fill the machine
theorem required_total_money : 
  (bundle_count * bill_5_value * number_of_bundles) + 
  (bundle_count * bill_10_value * number_of_bundles) + 
  (bundle_count * bill_20_value * number_of_bundles) = 35000 := 
by 
  sorry

end required_total_money_l764_764150


namespace sqrt_square_eq_orig_l764_764125

theorem sqrt_square_eq_orig : (sqrt 25)^2 = 25 :=
by
  sorry

end sqrt_square_eq_orig_l764_764125


namespace max_tank_volume_l764_764600

-- Defining the conditions given in the problem.
def side_length : ℝ := 120

-- The function for volume given side length x.
def volume (x : ℝ) : ℝ := - (1 / 2) * x^3 + 60 * x^2

-- The conditions for x.
def valid_x (x : ℝ) : Prop := 0 < x ∧ x < side_length

-- The statement to be proved.
theorem max_tank_volume : ∃ x, valid_x x ∧ 
  ∀ y, valid_x y → volume y ≤ volume x ∧ 
  volume x = 128000 :=
sorry

end max_tank_volume_l764_764600


namespace least_possible_third_side_l764_764353

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) : ∃ c, c = 24 ∧ a^2 - c^2 = 527  :=
by
  use (√527)
  sorry

end least_possible_third_side_l764_764353


namespace smallest_b_of_factored_quadratic_l764_764244

theorem smallest_b_of_factored_quadratic (r s : ℕ) (h1 : r * s = 1620) : (r + s) = 84 :=
sorry

end smallest_b_of_factored_quadratic_l764_764244


namespace sum_of_all_g_10_values_l764_764775

noncomputable def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
noncomputable def g (x : ℝ) : ℝ := 3 * x + 1

theorem sum_of_all_g_10_values : 
  let x_vals := {x : ℝ | f x = 10},
      g_10_vals := x_vals.image (λ x, g x) in
  g_10_vals.sum = 29 :=
by sorry

end sum_of_all_g_10_values_l764_764775


namespace ratio_areas_l764_764215

-- Definitions and conditions
def regular_octagon (P : Type) [metric_space P] := 
  ∃ octagon : list P, octagon.length = 8 ∧
  ∀ (i j : ℕ), (0 ≤ i ∧ i < 8) → (0 ≤ j ∧ j < 8) → 
  dist (list.nth_le octagon i sorry) (list.nth_le octagon (i+1) sorry) = dist (list.nth_le octagon j sorry) (list.nth_le octagon (j+1) sorry) ∧ 
  ∠ (list.nth_le octagon i sorry) (list.nth_le octagon (i+1) sorry) (list.nth_le octagon (i+2) sorry) = 135

def midpoint {P : Type} [metric_space P] (a b : P) := 
  ∃ m : P, dist a m = dist b m ∧ dist m b = dist a b

-- Given entities
variable (P : Type) [metric_space P]
variable (A B C E : P)
variable (octagon : list P) (M : P)

-- Conditions
hypothesis regular_octagon : regular_octagon P
hypothesis vertices : A = list.nth_le octagon 0 sorry ∧ B = list.nth_le octagon 1 sorry ∧ C = list.nth_le octagon 2 sorry ∧ E = list.nth_le octagon 4 sorry
hypothesis m_midpoint : midpoint B C = M

-- Question to prove
theorem ratio_areas : 
  let area_triangle (a b c : P) := sorry in -- We skip the exact area computation for the sake of the statement
  area_triangle A B M / area_triangle A C E = 1 / 6 :=
sorry

end ratio_areas_l764_764215


namespace probability_of_consecutive_draws_l764_764149

-- Assume chips and their counts are represented as variables for clarity
def red_chips : ℕ := 4
def green_chips : ℕ := 3
def blue_chips : ℕ := 5
def total_chips : ℕ := red_chips + green_chips + blue_chips

-- factorial calculations
def fact (n : ℕ) : ℕ := Nat.factorial n
def favorable_outcomes : ℕ := fact red_chips * fact green_chips * fact blue_chips * fact 3
def total_outcomes : ℕ := fact total_chips

theorem probability_of_consecutive_draws : 
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4620 := 
by
  sorry

end probability_of_consecutive_draws_l764_764149


namespace quadratic_function_expression_rational_function_expression_l764_764144

-- Problem 1:
theorem quadratic_function_expression (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 3 * x) ∧ (f 0 = 1) → (∀ x, f x = (3 / 2) * x^2 - (3 / 2) * x + 1) :=
by
  sorry

-- Problem 2:
theorem rational_function_expression (f : ℝ → ℝ) : 
  (∀ x, x ≠ 0 → 3 * f (1 / x) + f x = x) → 
  (∀ x, x ≠ 0 → f x = 3 / (8 * x) - x / 8) :=
by
  sorry

end quadratic_function_expression_rational_function_expression_l764_764144


namespace options_validation_l764_764127

variables (a b : Vect) (A B AC : ℝ)
variable {α : ℝ}
variables {m : ℝ}

noncomputable def sufficient_not_necessary_condition (a b : Vect) : Prop := 
  a = 1/2 * b

lemma vector_normalization_property :
  sufficient_not_necessary_condition a b →
  ∀ (a b : Vect), (a ≠ 0 ∧ b ≠ 0 → (a / |a| = b / |b| → a = |a| / |b| * b)) :=
sorry

lemma angle_point_property (α : ℝ) (m : ℝ) :
  sin α = -2 / sqrt 13 ∧ point_on_terminal_side α (3, -m) → m = 2 :=
sorry

lemma angle_sine_inequality (A B : ℝ) (a b : ℝ) :
  (A < B) ↔ (sin A < sin B) :=
sorry

lemma unique_triangle (AB B AC : ℝ) :
  AB = 2 * sqrt 2 → B = π/4 → AC = 3 →
  ∃! (ABC : Triangle), sides ABC = (AB, AC, _ ∧ ∠B = π/4) :=
sorry

theorem options_validation :
  (vector_normalization_property a b ∧ 
  ¬angle_point_property α m ∧ 
  angle_sine_inequality A B a b ∧ 
  unique_triangle AB B AC) :=
sorry

end options_validation_l764_764127


namespace rank_32_boxers_in_15_days_l764_764476

theorem rank_32_boxers_in_15_days :
  ∀ (boxers : List ℕ) (days : ℕ),
  (∀ x ∈ boxers, 0 ≤ x ∧ x < 32) ∧
  (days >= 15) ∧
  (∀ (i : ℕ), i ∈ boxers → ∃ n : ℕ, n = i) →
  ∃ ranking : List ℕ, ranking.length = boxers.length ∧ (∀ (i j : ℕ), i < j → ranking[i] < ranking[j]) :=
sorry

end rank_32_boxers_in_15_days_l764_764476


namespace quadratic_transformation_concept_l764_764505

theorem quadratic_transformation_concept :
  ∀ x : ℝ, (x-3)^2 - 4*(x-3) = 0 ↔ (x = 3 ∨ x = 7) :=
by
  intro x
  sorry

end quadratic_transformation_concept_l764_764505


namespace replaced_weight_l764_764438

theorem replaced_weight (avg_increase : ℝ) (num_girls : ℕ) (new_girl_weight : ℝ) :
  avg_increase = 5 → num_girls = 10 → new_girl_weight = 100 → ∃ (W : ℝ), W = 50 :=
by
  intros h1 h2 h3
  use 50
  sorry

end replaced_weight_l764_764438


namespace max_plus_min_of_f_l764_764030

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + Real.sin x) / (x ^ 2 + 1)

theorem max_plus_min_of_f : 
  let M := Real.sup (set.image f set.univ) in
  let N := Real.inf (set.image f set.univ) in
  M + N = 2 := by
  sorry

end max_plus_min_of_f_l764_764030


namespace median_positive_l764_764521

theorem median_positive (t : ℝ) :
  let a := t^3 - 100 * t
  let b := 2^t - 16
  let c := sin t - 1/2
  let med := (if a <= b then if b <= c then b else if a <= c then c else a else if a <= c then a else if b <= c then c else b)
  (10 < t ∨ (2 * Real.pi + Real.pi / 6 < t ∧ t < 2 * Real.pi + 5 * Real.pi / 6) ∨ (-2 * Real.pi + Real.pi / 6 < t ∧ t < -2 * Real.pi + 5 * Real.pi / 6)) ↔ med > 0 :=
by
  sorry

end median_positive_l764_764521


namespace total_files_deleted_l764_764969

theorem total_files_deleted 
  (initial_files : ℕ) (initial_apps : ℕ)
  (deleted_files1 : ℕ) (deleted_apps1 : ℕ)
  (added_files1 : ℕ) (added_apps1 : ℕ)
  (deleted_files2 : ℕ) (deleted_apps2 : ℕ)
  (added_files2 : ℕ) (added_apps2 : ℕ)
  (final_files : ℕ) (final_apps : ℕ)
  (h_initial_files : initial_files = 24)
  (h_initial_apps : initial_apps = 13)
  (h_deleted_files1 : deleted_files1 = 5)
  (h_deleted_apps1 : deleted_apps1 = 3)
  (h_added_files1 : added_files1 = 7)
  (h_added_apps1 : added_apps1 = 4)
  (h_deleted_files2 : deleted_files2 = 10)
  (h_deleted_apps2 : deleted_apps2 = 4)
  (h_added_files2 : added_files2 = 5)
  (h_added_apps2 : added_apps2 = 7)
  (h_final_files : final_files = 21)
  (h_final_apps : final_apps = 17) :
  (deleted_files1 + deleted_files2 = 15) := 
by
  sorry

end total_files_deleted_l764_764969


namespace sufficient_not_necessary_l764_764401

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2) ↔ (x + y ≥ 4) :=
by sorry

end sufficient_not_necessary_l764_764401


namespace parametric_graph_intersection_count_l764_764913

theorem parametric_graph_intersection_count : 
  (graph_intersections x y (1: ℝ) (30: ℝ)) = 9 :=
by
  let x (t: ℝ) := Real.cos t + t / 3
  let y (t: ℝ) := Real.sin t
  sorry -- Proof steps are omitted as instructed.

end parametric_graph_intersection_count_l764_764913


namespace part1_part2_l764_764271

-- Problem setup variables
variables {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}
variable {q : ℝ}
variable {a1 : ℝ}

-- Problem conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a 1 * q ^ n
axiom sum_first_four_terms : S 4 = 15 / 8
axiom arith_seq : a 1 + a 2 = 6 * a 3

-- Derived quantities
def b_n (n : ℕ) : ℝ := a n * (Real.log 2 (a n) - 1)
def sum_b_first_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

-- Problems to prove
theorem part1 : ∃ q : ℝ, 0 < q ∧ q ≠ 1 ∧ (∀ n, (S n - 2) / (S (n - 1) - 2) = q) :=
sorry

theorem part2 (n : ℕ) : sum_b_first_n n = (n + 2) * (1 / 2) ^ (n - 1) - 4 :=
sorry

end part1_part2_l764_764271


namespace shoe_size_combination_l764_764752

theorem shoe_size_combination (J A : ℕ) (hJ : J = 7) (hA : A = 2 * J) : J + A = 21 := by
  sorry

end shoe_size_combination_l764_764752


namespace reduction_when_fifth_runner_twice_as_fast_l764_764249

theorem reduction_when_fifth_runner_twice_as_fast (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h_T1 : (T1 / 2 + T2 + T3 + T4 + T5) = 0.95 * T)
  (h_T2 : (T1 + T2 / 2 + T3 + T4 + T5) = 0.90 * T)
  (h_T3 : (T1 + T2 + T3 / 2 + T4 + T5) = 0.88 * T)
  (h_T4 : (T1 + T2 + T3 + T4 / 2 + T5) = 0.85 * T)
  : (T1 + T2 + T3 + T4 + T5 / 2) = 0.92 * T := 
sorry

end reduction_when_fifth_runner_twice_as_fast_l764_764249


namespace limit_a_n_sub_b_n_l764_764012

noncomputable def f : ℝ → ℝ := sorry
-- Define f as an increasing differentiable function
def f_increasing : ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ := sorry
def f_differentiable : Differentiable ℝ f := sorry
def f_lim_at_infty : Tendsto f atTop (𝓝 ⊤) := sorry
def f'_bounded : ∃ M, ∀ x, ∥deriv f x∥ ≤ M := sorry

noncomputable def F (x : ℝ) : ℝ := ∫ t in 0..x, f t

/-- Sequence a_n defined recursively: a₀ = 1, a_(n+1) = a_n + 1/f(a_n) -/
noncomputable def a_seq : ℕ → ℝ
| 0 => 1
| (n+1) => a_seq n + 1 / f (a_seq n)

noncomputable def F_inv (n : ℝ) : ℝ := sorry -- Inverse of F, should be increasing, unbounded 
noncomputable def b_seq (n : ℕ) : ℝ := F_inv n

theorem limit_a_n_sub_b_n : Tendsto (λ n, (a_seq n - b_seq n)) atTop (𝓝 0) := sorry

end limit_a_n_sub_b_n_l764_764012


namespace binomial_alternating_sum_l764_764585

theorem binomial_alternating_sum:
  (∑ k in Finset.range (51), (-1:ℤ)^k * (Nat.choose 50 k)) = 0 :=
  sorry

end binomial_alternating_sum_l764_764585


namespace prob_event_A_correct_prob_event_B_correct_l764_764695

open Finset

noncomputable def prob_event_A : ℚ := 
  let P := {1, 2, 3, 4}
  let Q := {2, 4, 6, 8}
  let total_p := P.card * Q.card
  let event_A_vals := {(a, b) | a ∈ P ∧ b ∈ Q ∧ b = 2 * a}
  (event_A_vals.card : ℚ) / total_p

theorem prob_event_A_correct : prob_event_A = 1 / 4 := sorry

noncomputable def prob_event_B : ℚ := 
  let P := {1, 2, 3, 4}
  let Q := {2, 4, 6, 8}
  let total_p := P.card * Q.card
  let event_B_vals := {(a, b) | a ∈ P ∧ b ∈ Q ∧ b^2 > 4 * a}
  (event_B_vals.card : ℚ) / total_p

theorem prob_event_B_correct : prob_event_B = 11 / 16 := sorry

end prob_event_A_correct_prob_event_B_correct_l764_764695


namespace range_of_a_decreasing_function_l764_764289

theorem range_of_a_decreasing_function (a : ℝ) :
  (∀ x < 1, ∀ y < x, (3 * a - 1) * x + 4 * a ≥ (3 * a - 1) * y + 4 * a) ∧ 
  (∀ x ≥ 1, ∀ y > x, -a * x ≤ -a * y) ∧
  (∀ x < 1, ∀ y ≥ 1, (3 * a - 1) * x + 4 * a ≥ -a * y)  →
  (1 / 8 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
sorry

end range_of_a_decreasing_function_l764_764289


namespace simplify_fraction_sequence_l764_764061

theorem simplify_fraction_sequence :
  (∏ k in finset.range 1000, (3 * (k + 3) + 3) / (3 * (k + 3)))
  = 1001 :=
by
suffices : (∏ k in finset.range 1000, (3 * (k + 3) + 3) / (3 * (k + 3))) = 3003 / 3,
{ rwa [div_eq_mul_inv, mul_inv_cancel (3 : ℤ), inv_one, mul_one] },
sorry

end simplify_fraction_sequence_l764_764061


namespace find_radius_r3_l764_764033

theorem find_radius_r3 (r1 r2 r3 : ℝ) (h1 : r1 = 18) (h2 : r2 = 8) : r3 = 12 :=
by
  have h : r3 = Real.sqrt (r1 * r2) := sorry
  rw [h1, h2] at h
  norm_num at h
  exact h

end find_radius_r3_l764_764033


namespace sqrt_product_of_divisors_of_36_l764_764643

theorem sqrt_product_of_divisors_of_36 : 
  (∃ n : ℕ, prime_factors 36 = [2, 2, 3, 3] ∧ n = sqrt (∏ d in (divisors 36), d)) → n = 6^(4.5) :=
by {
  sorry -- proof not required
}

end sqrt_product_of_divisors_of_36_l764_764643


namespace total_shopping_cost_l764_764582

def tuna_cost_per_pack := 2
def tuna_num_packs := 5
def water_cost_per_bottle := 1.5
def water_num_bottles := 4
def different_goods_cost := 40

theorem total_shopping_cost:
  (tuna_num_packs * tuna_cost_per_pack) + 
  (water_num_bottles * water_cost_per_bottle) + 
  different_goods_cost = 56 :=
by sorry

end total_shopping_cost_l764_764582


namespace train_passing_time_l764_764925

-- Definitions based on the given conditions
def train_length : ℝ := 280
def train_speed_kmph : ℝ := 72
def conversion_factor : ℝ := 5 / 18

-- Definition of the converted speed
def train_speed_mps : ℝ := train_speed_kmph * conversion_factor

-- Statement of the problem rewritten in Lean 4
theorem train_passing_time : train_length / train_speed_mps = 14 := by
  sorry

end train_passing_time_l764_764925


namespace tiffany_lost_lives_l764_764852

theorem tiffany_lost_lives :
  ∃ L : ℕ, 43 - L + 27 = 56 ∧ L = 14 :=
begin
  use 14,
  split,
  { norm_num, },
  { norm_num, }
end

end tiffany_lost_lives_l764_764852


namespace problem1_problem2_l764_764956

-- First problem
theorem problem1 : (π - 2)^0 - (1 / 2)^(-2) + 3^2 = 6 := 
by 
  sorry

-- Second problem
theorem problem2 (x : ℝ) : (-2 * x^2)^2 + x^3 * x - x^5 / x = 4 * x^4 := 
by 
  sorry

end problem1_problem2_l764_764956


namespace find_OA_l764_764749

theorem find_OA 
    (O A B M: Type) 
    [RightTriangle ∠(A O B)]
    (h: Perpendicular (height O A B))
    (M_intersect: M extends h)
    (distance_M: distance_from_side M 2)
    (distance_B: distance_from_side B 1) :
  distance (O A) = sqrt(2) := by
  sorry

end find_OA_l764_764749


namespace log_inequality_l764_764712

theorem log_inequality (a b c : ℝ) (ha : a > b) (hb : b > 0) (hc : c > 1) : log c a > log c b :=
  sorry

end log_inequality_l764_764712


namespace henrys_friend_money_l764_764708

theorem henrys_friend_money (h1 h2 : ℕ) (T : ℕ) (f : ℕ) : h1 = 5 → h2 = 2 → T = 20 → h1 + h2 + f = T → f = 13 :=
by
  intros h1_eq h2_eq T_eq total_eq
  rw [h1_eq, h2_eq, T_eq] at total_eq
  sorry

end henrys_friend_money_l764_764708


namespace common_tangents_concentric_circles_l764_764964

open EuclideanGeometry

noncomputable def num_common_tangents_concentric (r : ℝ) : ℕ :=
  let C₁ := circle (0, 0) r
  let C₂ := circle (0, 0) (2 * r)
  -- Prove that the number of common tangents is 0
  0

theorem common_tangents_concentric_circles (r : ℝ) (h : r > 0) :
  num_common_tangents_concentric r = 0 := sorry

end common_tangents_concentric_circles_l764_764964


namespace difference_in_profit_percentage_l764_764189

-- Definitions from conditions
def selling_price1 : ℝ := 350
def selling_price2 : ℝ := 340
def cost_price : ℝ := 200

-- Definition to calculate profit percentage
def profit_percentage (selling_price : ℝ) (cost_price : ℝ) : ℝ := 
  ((selling_price - cost_price) / cost_price) * 100

-- Theorem to prove the difference in profit percentages is 5%
theorem difference_in_profit_percentage : 
  profit_percentage selling_price1 cost_price - profit_percentage selling_price2 cost_price = 5 :=
sorry

end difference_in_profit_percentage_l764_764189


namespace simplify_product_l764_764058

noncomputable def product (n : ℕ) : ℚ :=
  ∏ k in Finset.range n, (3 * k + 6) / (3 * k)

theorem simplify_product : product 997 = 1001 := 
sorry

end simplify_product_l764_764058


namespace solve_abs_equation_l764_764427

theorem solve_abs_equation (x : ℝ) :
  |2 * x - 1| + |x - 2| = |x + 1| ↔ 1 / 2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end solve_abs_equation_l764_764427


namespace magnitude_AD_l764_764323

open Real

noncomputable def m : ℝ × ℝ := sorry
noncomputable def n : ℝ × ℝ := sorry

-- Given conditions
axiom m_magnitude : |m| = sqrt 3
axiom n_magnitude : |n| = 2
axiom angle_mn : ∀ θ, cos θ = cos (π / 6)

-- Definition of vectors AB and AC
noncomputable def AB : ℝ × ℝ := 2 • m + n
noncomputable def AC : ℝ × ℝ := 2 • m - 6 • n

-- Midpoint D is defined, thus AD is defined in terms of AB and AC.
noncomputable def AD : ℝ × ℝ := 0.5 • (AB + AC)

-- The theorem we want to prove
theorem magnitude_AD : |AD| = sqrt 7 := by
  sorry

end magnitude_AD_l764_764323


namespace eventually_constant_and_lim_value_l764_764783

variable {r s : ℕ}

def sequence (r s : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then r
  else if n = 1 then s
  else
    let an := sequence r s (n - 1)
    let anm1 := sequence r s (n - 2)
    Nat.gcd an anm1

theorem eventually_constant_and_lim_value
  (h1 : r > 0) (h2 : s > 0)
  (h3 : Nat.odd r) (h4 : Nat.odd s) :
  ∃ N g, (∀ n ≥ N, sequence r s n = g) ∧ g = Nat.gcd r s := by
  sorry

end eventually_constant_and_lim_value_l764_764783


namespace worker_time_proof_l764_764416

theorem worker_time_proof (x : ℝ) (h1 : x > 2) (h2 : (100 / (x - 2) - 100 / x) = 5 / 2) : 
  (x = 10) ∧ (x - 2 = 8) :=
by
  sorry

end worker_time_proof_l764_764416


namespace female_wage_per_day_l764_764898

variables (F : ℝ)

-- given conditions
def male_workers := 20
def female_workers := 15
def child_workers := 5

def male_wage := 35
def child_wage := 8

def average_wage_per_day := 26

def total_workers := male_workers + female_workers + child_workers
def total_wage := total_workers * average_wage_per_day

def total_male_wage := male_workers * male_wage
def total_child_wage := child_workers * child_wage

def total_loss := total_wage - (total_male_wage + total_child_wage)

-- Proof statement
theorem female_wage_per_day : F = 20 :=
by
  -- introduce the facts we know
  let F := 20
  have h_total_wage : total_wage = 1040 := by sorry
  have h_total_male_wage : total_male_wage = 700 := by sorry
  have h_total_child_wage : total_child_wage = 40 := by sorry
  have h_total_female_wage : total_loss = total_wage - (total_male_wage + total_child_wage) := by sorry
  have h_female_wage : F = total_loss / female_workers := by sorry
  -- expression for F
  show F = 20 from h_female_wage

end female_wage_per_day_l764_764898


namespace find_constants_l764_764622

theorem find_constants :
  ∃ (P Q R : ℚ), 
  (∀ x : ℚ, x ≠ 4 → x ≠ 2 →
    (3 * x^2 - 2 * x) / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) 
  ∧ P = 10 ∧ Q = -7 ∧ R = -4 :=
by {
  let P := 10,
  let Q := -7,
  let R := -4,
  use [P, Q, R],
  split,
  { intros x hx4 hx2,
    field_simp [hx4, hx2],
    ring_nf,
    have e : (10 * x * x + -7 * x * x + -4 * (x * - 4) = 3 * x * x + -2 * x),
    { ring_exp,
      norm_num,
      ring_nf,
    },
    assumption, },
  simp,
  split;
  refl,
}

end find_constants_l764_764622


namespace rectangular_field_length_l764_764170

theorem rectangular_field_length (w : ℝ) (h₁ : w * (w + 10) = 171) : w + 10 = 19 := 
by
  sorry

end rectangular_field_length_l764_764170


namespace propositions_correct_l764_764300

theorem propositions_correct :
  (∀ (m l : Set Point) (α β : Set Point) (A : Point),
    (m ⊆ α) → (A ∈ l ∩ α) → (A ∉ m) → ¬Coplanar l m) ∧
  (∀ (m l n : Set Point) (α : Set Point),
    (Skew m l) → (Parallel l α) → (Parallel m α) → (Perpendicular n l) → (Perpendicular n m) → Perpendicular n α) ∧
  ¬ (∀ (l m : Set Point) (α β : Set Point),
    (Parallel l α) → (Parallel m β) → (Parallel α β) → (Parallel l m)) ∧
  (∀ (l m : Set Point) (α β : Set Point) (A : Point),
    (l ⊆ α) → (m ⊆ α) → (A ∈ l ∩ m) → (Parallel l β) → (Parallel m β) → (Parallel α β)) :=
sorry

end propositions_correct_l764_764300


namespace polygon_sides_l764_764471

theorem polygon_sides (n : ℕ) : (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end polygon_sides_l764_764471


namespace smallest_number_of_rectangles_needed_l764_764119

theorem smallest_number_of_rectangles_needed :
  ∃ n : ℕ, ∀ r : ℕ, c : ℕ, s : ℕ, 
    (r = 3) → (c = 4) → (s = 12) → 
    (s * s = n * r * c) → (n = 12) :=
by
  sorry

end smallest_number_of_rectangles_needed_l764_764119


namespace correct_option_l764_764872

-- Definitions from the conditions of the problem

def second_quadrant_angle (θ : ℝ) : Prop :=
  π/2 < θ ∧ θ < π

def terminal_side (θ : ℝ) (α : ℝ) : Prop :=
  ∃ k : ℤ, θ = α + 2 * k * π

-- restating options in formal terms
def option_A : Prop :=
  ∀ θ, second_quadrant_angle θ → θ > π/2 ∧ θ < π

def option_B : Prop :=
  ∀ θ α, θ = α → terminal_side θ α

def option_C : Prop :=
  ∀ θ α, terminal_side θ α → θ = α

def option_D : Prop :=
  ∀ θ α, θ ≠ α → ¬ terminal_side θ α

-- The main theorem restating which option is correct
theorem correct_option : option_B ∧ ¬ option_A ∧ ¬ option_C ∧ ¬ option_D :=
by
  sorry

end correct_option_l764_764872


namespace sum_of_repeating_decimals_l764_764770

def repeating_decimal_sum := 
  let T := {x : ℝ | ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999}
  ∑ x in T, x = 908.208208208208

theorem sum_of_repeating_decimals :
  repeating_decimal_sum := sorry

end sum_of_repeating_decimals_l764_764770


namespace polynomial_P0_values_l764_764217

theorem polynomial_P0_values (P : ℝ[X]) :
  (∀ x y : ℝ, |y^2 - P.eval x| ≤ 2 * |x| ↔ |x^2 - P.eval y| ≤ 2 * |y|) →
  (P.eval 0 ∈ Set.Iic 0 ∪ {1}) :=
begin
  sorry
end

end polynomial_P0_values_l764_764217


namespace percentage_non_science_majors_l764_764731

-- Definitions of given conditions
def class_total := 100
def percentage_men := 0.40 * class_total
def percentage_women := class_total - percentage_men
def percentage_women_science_majors := 0.30 * percentage_women
def percentage_men_science_majors := 0.5500000000000001 * percentage_men
def percentage_science_majors := percentage_women_science_majors + percentage_men_science_majors

-- Problem statement to prove
theorem percentage_non_science_majors : 
  class_total - percentage_science_majors = 0.60 * class_total :=
by
  -- This construction will capture the idea to be proven, adding 'sorry' to skip actual proof steps
  sorry

end percentage_non_science_majors_l764_764731


namespace minnows_left_over_l764_764006

theorem minnows_left_over (total_minnows : ℕ) (minnows_per_prize : ℕ) (total_players : ℕ) (winning_percentage : ℝ) :
  total_minnows = 600 →
  minnows_per_prize = 3 →
  total_players = 800 →
  winning_percentage = 0.15 →
  (total_minnows - (nat.floor ((winning_percentage * total_players : ℝ)) * minnows_per_prize) = 240) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end minnows_left_over_l764_764006


namespace two_digit_numbers_remainders_l764_764562

theorem two_digit_numbers_remainders :
  let numbers := {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ A % 4 = 3 ∧ A % 3 = 2} in
  numbers = {11, 23, 35, 47, 59, 71, 83, 95} :=
by
  sorry

end two_digit_numbers_remainders_l764_764562


namespace part1_part2_l764_764017

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
if h : n > 0 then (-1 : ℝ)^n * a n - 1 / (2^n : ℝ)
else 0

theorem part1 (a : ℕ → ℝ) : S 3 a = (-1 : ℝ)^3 * a 3 - 1 / 2^3 → a 3 = - 1 / 16 :=
sorry

theorem part2 (a : ℕ → ℝ) : (∀ n : ℕ, n > 0 → S n a = (-1 : ℝ)^n * a n - 1 / 2^n) →
  (Σ (k : ℕ) in finset.range 100, S (k + 1) a) = 1 / 3 * (1 / 2^100 - 1) :=
sorry

end part1_part2_l764_764017


namespace ab_difference_l764_764719

theorem ab_difference (a b : ℝ) 
  (h1 : 10 = a * 3 + b)
  (h2 : 22 = a * 7 + b) : 
  a - b = 2 := 
  sorry

end ab_difference_l764_764719


namespace fraction_remains_unchanged_l764_764723

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) :=
by {
  sorry
}

end fraction_remains_unchanged_l764_764723


namespace sum_elements_T_l764_764766

noncomputable def T := {x : ℝ | ∃ a b c d : ℕ, 0 ≤ a ∧ a < 10 ∧ 
                                   0 ≤ b ∧ b < 10 ∧ 
                                   0 ≤ c ∧ c < 10 ∧ 
                                   0 ≤ d ∧ d < 10 ∧ 
                                   a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 
                                   d ≠ b ∧ d ≠ c ∧ x = a * 10^3 + b * 10^2 + c * 10 + d }

theorem sum_elements_T : ∑ x in T, x = 28028 / 111 :=
by
    sorry

end sum_elements_T_l764_764766


namespace diplomats_number_l764_764134

theorem diplomats_number (J R : ℕ) (D : ℝ) 
  (hm1 : J = 20)
  (hm2 : D - ↑R = 32)
  (hm3 : D - (J ∪ R) = 0.2 * D)
  (hm4 : J ∩ R = 0.1 * D) : 
  D = 120 :=
begin
  sorry
end

end diplomats_number_l764_764134


namespace investor_difference_l764_764574

def investment_A : ℝ := 300
def investment_B : ℝ := 200
def rate_A : ℝ := 0.30
def rate_B : ℝ := 0.50

theorem investor_difference :
  ((investment_A * (1 + rate_A)) - (investment_B * (1 + rate_B))) = 90 := 
by
  sorry

end investor_difference_l764_764574


namespace remaining_content_end_of_second_day_l764_764133

-- This leans ensures we have to use rational numbers
variable (initial_content remaining_content remaining_content_second_day : ℚ)

-- Conditions specified in the problem
def condition1 (initial_content remaining_content : ℚ) : Prop :=
  remaining_content = initial_content * (1 - 2/3)

def condition2 (remaining_content remaining_content_second_day : ℚ) : Prop :=
  remaining_content_second_day = remaining_content * (1 - 1/4)

-- Theorem to be proven
theorem remaining_content_end_of_second_day (initial_content remaining_content remaining_content_second_day : ℚ) 
  (h1 : condition1 initial_content remaining_content) 
  (h2 : condition2 remaining_content remaining_content_second_day) :
  remaining_content_second_day = initial_content * 1/4 :=
begin
  -- Proof to be provided
  sorry
end

end remaining_content_end_of_second_day_l764_764133


namespace rejuvenating_apples_l764_764004

-- Define the statements made by Baba Yaga, Koschei, and Leshy
def baba_yaga_statement1 : Prop := "Koschei has the apples."
def baba_yaga_statement2 : Prop := "If Leshy had them, he would give them to me."

def koschei_statement1 : Prop := "Baba Yaga hid the apples."
def koschei_statement2 : Prop := "Baba Yaga has the apples."
def koschei_statement3 : Prop := "Koschei does not have the apples."

def leshy_statement1 : Prop := "Leshy does not have the apples."
def leshy_statement2 : Prop := "Baba Yaga does not have the apples."

-- Define their truth values considering they always lie
def baba_yaga_lies : Prop := ¬(baba_yaga_statement1 ∧ baba_yaga_statement2)
def koschei_lies : Prop := ¬(koschei_statement1 ∧ koschei_statement2 ∧ koschei_statement3)
def leshy_lies : Prop := ¬(leshy_statement1 ∧ leshy_statement2)

-- At least one of them has the apples
def at_least_one_has_apples : Prop := (baba_yaga_statement2 → ¬leshy_statement1) ∨ ¬koschei_statement3 ∨ ¬leshy_statement1

-- Proof to be written here
theorem rejuvenating_apples :
  (leshy_statement1 = false) ∧
  (baba_yaga_statement2 = false) ∧
  (¬leshy_statement1) ∧
  at_least_one_has_apples :=
by
  -- Insert proof steps here
  sorry

end rejuvenating_apples_l764_764004


namespace ship_distance_graph_pattern_l764_764919

noncomputable def ship_route_graph {X : Type} (r : ℝ) (A B C D Y : X) : X → ℝ :=
  sorry

theorem ship_distance_graph_pattern (r : ℝ) (A B C D Y : X) :
  (∀ t ∈ [0, 1/4], ship_route_graph r A B C D Y (circle_route t A B) = r) ∧
  (∀ t ∈ [0, 1], ship_route_graph r A B C D Y (straight_route t B C Y) = graph_valley_peak t) ∧
  (∀ t ∈ [0, 1], ship_route_graph r A B C D Y (straight_route t C D) = graph_peak_reduction t) →
  ∃ correct_graph_pattern, correct_graph_pattern = "valley followed by a peak and another reduction" :=
  sorry

end ship_distance_graph_pattern_l764_764919


namespace length_major_axis_l764_764315

/-- 
Proof problem:
The length of the major axis of the ellipse given by:
\begin{cases}
x = 3 \cos \phi \\
y = 4 \sin \phi 
\end{cases}
is equal to 8.
-/
theorem length_major_axis (φ : ℝ) : 
  let x := 3 * Real.cos φ in
  let y := 4 * Real.sin φ in
  (2 * max x y = 8) := 
  sorry

end length_major_axis_l764_764315


namespace investment_ratio_proof_l764_764931

noncomputable def investment_ratio {A_invest B_invest C_invest : ℝ} (profit total_profit : ℝ) (A_times_B : ℝ) : ℝ :=
  C_invest / (A_times_B * B_invest + B_invest + C_invest)

theorem investment_ratio_proof (A_invest B_invest C_invest : ℝ)
  (profit total_profit : ℝ) (A_times_B : ℝ) 
  (h_profit : total_profit = 55000)
  (h_C_share : profit = 15000.000000000002)
  (h_A_times_B : A_times_B = 3)
  (h_ratio_eq : A_times_B * B_invest + B_invest + C_invest = 11 * B_invest / 3) :
  (A_invest / C_invest = 2) :=
by
  sorry

end investment_ratio_proof_l764_764931


namespace counterexample_to_proposition_l764_764109

theorem counterexample_to_proposition (a b : ℤ) (h : a > b) : ∃ (a b : ℤ), 
  a = -2 ∧ b = -3 ∧ a > b ∧ ¬ (a^2 > a * b) :=
by {
  use [-2, -3],
  split, { refl },
  split, { refl },
  split, { linarith },
  simp,
  linarith,
}

end counterexample_to_proposition_l764_764109


namespace area_of_45_45_90_triangle_l764_764454

theorem area_of_45_45_90_triangle (h : ℝ) (a : ℝ) :
  h = 10 * sqrt 2 → a = 45 → (∃ leg : ℝ, leg = h / sqrt 2 ∧ (1 / 2) * leg * leg = 50) :=
by
  intros
  refine ⟨_, _, _⟩
  sorry

end area_of_45_45_90_triangle_l764_764454


namespace intersections_even_l764_764881

open Function

-- Define polygonal line as a list of points
structure PolygonalLine where
  points: List (ℝ × ℝ)
  closed: points.head? = points.last?

-- Define the problem of intersection
def number_of_intersections_is_even (p1 p2: PolygonalLine) : Prop :=
  (PolygonalLine.points p1 ∩ PolygonalLine.points p2).length % 2 = 0

theorem intersections_even (p1 p2: PolygonalLine) 
  (h1: closed (PolygonalLine.points p1)) 
  (h2: closed (PolygonalLine.points p2)) 
  (general_position: ¬(∃ p: ℝ × ℝ, length ((PolygonalLine.points p1 ∩ PolygonalLine.points p2).filter (λ x, x = p)) > 1)) :
  number_of_intersections_is_even p1 p2 :=
by
  sorry

end intersections_even_l764_764881


namespace projectile_height_35_l764_764077

def height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 30 * t

theorem projectile_height_35 (t : ℝ) :
  height t = 35 → t = 10 / 7 :=
by
  sorry

end projectile_height_35_l764_764077


namespace Tim_running_hours_per_week_l764_764108

noncomputable def running_time_per_week : ℝ :=
  let MWF_morning : ℝ := (1 * 60 + 20 - 10) / 60 -- minutes to hours
  let MWF_evening : ℝ := (45 - 10) / 60 -- minutes to hours
  let TS_morning : ℝ := (1 * 60 + 5 - 10) / 60 -- minutes to hours
  let TS_evening : ℝ := (50 - 10) / 60 -- minutes to hours
  let MWF_total : ℝ := (MWF_morning + MWF_evening) * 3
  let TS_total : ℝ := (TS_morning + TS_evening) * 2
  MWF_total + TS_total

theorem Tim_running_hours_per_week : running_time_per_week = 8.42 := by
  -- Add the detailed proof here
  sorry

end Tim_running_hours_per_week_l764_764108


namespace divisibility_by_1897_l764_764425

theorem divisibility_by_1897 (n : ℕ) : 1897 ∣ (2903 ^ n - 803 ^ n - 464 ^ n + 261 ^ n) :=
sorry

end divisibility_by_1897_l764_764425


namespace median_moons_of_planet_x_l764_764910

noncomputable def moons_per_planet : List ℕ := [0, 0, 1, 2, 2, 4, 5, 15, 16, 23]

def sorted_moons (l : List ℕ) : List ℕ := List.qsort (≤) l

def median_of_list (l : List ℕ) : ℕ :=
  if h : l.length % 2 = 0 then
    let mid := l.length / 2
    (l.get (mid - 1) + l.get mid) / 2
  else 
    l.get (l.length / 2)

theorem median_moons_of_planet_x :
  median_of_list (sorted_moons moons_per_planet) = 3 :=
by
  sorry

end median_moons_of_planet_x_l764_764910


namespace fred_bought_two_tickets_l764_764037

-- Definitions of constants
def ticket_cost : ℝ := 5.92
def movie_borrow_cost : ℝ := 6.79
def amount_paid : ℝ := 20
def change_received : ℝ := 1.37

-- Definition of conditions in Lean
def total_amount_spent : ℝ := amount_paid - change_received
def total_amount_spent_on_tickets : ℝ := total_amount_spent - movie_borrow_cost
def number_of_tickets_purchased : ℝ := total_amount_spent_on_tickets / ticket_cost

-- The theorem to prove that Fred bought 2 tickets given the conditions
theorem fred_bought_two_tickets : number_of_tickets_purchased = 2 := by
  sorry

end fred_bought_two_tickets_l764_764037


namespace lighter_shopping_bag_weight_l764_764854

theorem lighter_shopping_bag_weight :
  ∀ (G : ℕ), (G + 7 = 10) → (G = 3) := by
  intros G h
  sorry

end lighter_shopping_bag_weight_l764_764854


namespace largest_n_with_last_nonzero_digit_one_l764_764140

-- Definitions for conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_nonzero_digit (n : ℕ) : ℕ :=
  let digits := (factorial n).digits 10
  digits.reverse.find (λ d, d ≠ 0)

-- The theorem to prove
theorem largest_n_with_last_nonzero_digit_one : ∀ n : ℕ, last_nonzero_digit n = 1 → n ≤ 1 :=
by
  sorry

end largest_n_with_last_nonzero_digit_one_l764_764140


namespace volume_of_solid_eq_sphere_l764_764089

theorem volume_of_solid_eq_sphere (x y z : ℝ) :
  let u := (x, y, z) in 
  u.1 * u.1 + u.2 * u.2 + u.3 * u.3 = 6 * u.1 - 28 * u.2 + 12 * u.3 →
  (4 / 3) * π * (241 ^ (3 / 2)) = 
  volume_of_solid_formed_by_vectors u :=
sorry

end volume_of_solid_eq_sphere_l764_764089


namespace molecular_weight_l764_764500

/--
  Prove that the molecular weight of a compound having 6 C (including 2 C-13 isotopes),
  12 H, 3 O, and 2 N is 162.116 amu given the atomic weights for the most common isotopes.
-/
theorem molecular_weight
  (C_12_weight : Real := 12.00)
  (C_13_weight : Real := 13.00)
  (H_weight : Real := 1.008)
  (O_weight : Real := 16.00)
  (N_weight : Real := 14.01) :
  let total_weight := 4 * C_12_weight + 2 * C_13_weight + 12 * H_weight + 3 * O_weight + 2 * N_weight
  in total_weight = 162.116 := by
  sorry

end molecular_weight_l764_764500


namespace hexagon_vertex_labels_impossible_l764_764612

noncomputable theory

open_locale big_operators

def operation_one (a b : ℤ) : (ℤ × ℤ) :=
  (a + 2, b - 3)

def operation_two (a : ℤ) : ℤ :=
  6 * a

theorem hexagon_vertex_labels_impossible :
  let A1 A2 A3 A4 A5 A6 : ℤ := 0
  in ¬(∃ ops : list (ℤ × ℤ → ℤ × ℤ ∨ ℤ → ℤ),
          let final_state := ops.foldl (λ (state : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ),
                λ (op : ℤ × ℤ → ℤ × ℤ ∨ ℤ → ℤ),
                match op with
                | (sum.inl f) := (λ (S : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) k, 
                      match k with 
                      | 0 := f S.fst S.snd 
                      ...
                      end)
                | (sum.inr g) := (λ (S : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) k,
                      match k with 
                      | 0 := (g S.fst)
                      ...
                      end) end) 
                (A1, A2, A3, A4, A5, A6)
          in final_state = (1, 2, 3, 4, 5, 6)) :=
  sorry

end hexagon_vertex_labels_impossible_l764_764612


namespace inequality_l764_764788

noncomputable def X_k (a : ℕ → ℝ) (k : ℕ) : ℝ := ∑ i in finset.range (2^k+1).filter (λ i, i > 0), a i

noncomputable def Y_k (a : ℕ → ℝ) (k : ℕ) : ℝ := ∑ i in finset.range (2^k+1).filter (λ i, i > 0), ⌊2^k / i⌋ * a i

theorem inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n) (n : ℕ) (h_pos : 0 < n) :
  X_k a n ≤ Y_k a n - ∑ i in finset.range n, Y_k a i ∧ Y_k a n - ∑ i in finset.range n, Y_k a i ≤ ∑ i in finset.range (n+1), X_k a i :=
sorry

end inequality_l764_764788


namespace sum_of_T_l764_764761

def is_repeating_decimal (x : ℝ) :=
  ∃ (a b c d : ℕ), (∀ i j, {i, j} ⊆ {a, b, c, d} → i ≠ j) ∧
  x = (a * 1000 + b * 100 + c * 10 + d) / 9999

def T : set ℝ := { x | is_repeating_decimal x }

theorem sum_of_T : ∑ x in T, x = 280 := sorry

end sum_of_T_l764_764761


namespace range_of_f_l764_764840

-- Define the quadratic form function
def quadratic_fn (x : ℝ) : ℝ := x^2 - 6 * x + 17

-- Define the log base 1/2 function
def log_base_half (u : ℝ) : ℝ := real.log u / real.log (1 / 2)

-- Define the composed function
def f (x : ℝ) : ℝ := log_base_half (quadratic_fn x)

-- State the main theorem
theorem range_of_f : (∀ y, ∃ x, y = log_base_half (quadratic_fn x)) ↔ (y ∈ Iic (-3)) := by
  sorry

end range_of_f_l764_764840


namespace fourth_stack_taller_l764_764850

def stacks (stack1 stack2 stack3 stack4 stack5 : ℕ) : Prop :=
  stack1 = 7 ∧ 
  stack2 = stack1 + 3 ∧ 
  stack3 = stack2 - 6 ∧ 
  stack5 = 2 * stack2 ∧ 
  7 + stack2 + stack3 + stack5 + stack4 = 55

theorem fourth_stack_taller (stack1 stack2 stack3 stack4 stack5 : ℕ) 
  (h : stacks stack1 stack2 stack3 stack4 stack5) :
  stack4 - stack3 = 10 :=
by
  cases h with h1 h_cond
  sorry

end fourth_stack_taller_l764_764850


namespace negation_of_prop_l764_764116

open Classical

variable (a b : ℝ)

theorem negation_of_prop :
  (¬ (a > b) ∨ 2a > 2b) = (a ≤ b → 2a ≤ 2b) :=
by
  sorry

end negation_of_prop_l764_764116


namespace probability_is_one_over_145_l764_764858

-- Define the domain and properties
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even (n : ℕ) : Prop :=
  n % 2 = 0

-- Total number of ways to pick 2 distinct numbers from 1 to 30
def total_ways_to_pick_two_distinct : ℕ :=
  (30 * 29) / 2

-- Calculate prime numbers between 1 and 30
def primes_from_1_to_30 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Filter valid pairs where both numbers are prime and at least one of them is 2
def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 7), (2, 11), (2, 13), (2, 17), (2, 19), (2, 23), (2, 29)]

def count_valid_pairs (l : List (ℕ × ℕ)) : ℕ :=
  l.length

-- Probability calculation
def probability_prime_and_even : ℚ :=
  count_valid_pairs (valid_pairs primes_from_1_to_30) / total_ways_to_pick_two_distinct

-- Prove that the probability is 1/145
theorem probability_is_one_over_145 : probability_prime_and_even = 1 / 145 :=
by
  sorry

end probability_is_one_over_145_l764_764858


namespace sum_of_squares_eq_two_l764_764265

variable {p q r s : ℝ}
variable (B : Matrix (Fin 2) (Fin 2) ℝ)

-- Given conditions
def B_mat_eq : B = ![![p, q], ![r, s]] := sorry
def B_transpose_inv : Bᵀ = B⁻¹ := sorry
def p_equals_s : p = s := sorry

-- Proof goal
theorem sum_of_squares_eq_two (hpq : B_mat_eq) (htranspose_inv : B_transpose_inv) (hpeq : p_equals_s) :
  p^2 + q^2 + r^2 + s^2 = 2 := sorry

end sum_of_squares_eq_two_l764_764265


namespace product_of_slopes_l764_764392

variable {a b x₀ y₀ x₁ y₁ : ℝ}

-- Given Conditions
def on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Slopes of the lines P P₁ and P P₂
def slope (x₀ y₀ x₁ y₁ : ℝ) : ℝ :=
  (y₀ - y₁) / (x₀ - x₁)

def slope' (x₀ y₀ x₁ y₁ : ℝ) : ℝ :=
  (y₀ + y₁) / (x₀ + x₁)

theorem product_of_slopes (h₁ : on_ellipse x₀ y₀ a b)
  (h₂ : on_ellipse x₁ y₁ a b) (h₃ : a > b) (h₄ : b > 0) :
  slope x₀ y₀ x₁ y₁ * slope' x₀ y₀ x₁ y₁ = -b^2 / a^2 :=
by
  sorry

end product_of_slopes_l764_764392


namespace first_statement_second_statement_difference_between_statements_l764_764230

variable (A B C : Prop)

-- First statement: (A ∨ B) → C
theorem first_statement : (A ∨ B) → C :=
sorry

-- Second statement: (A ∧ B) → C
theorem second_statement : (A ∧ B) → C :=
sorry

-- Proof that shows the difference between the two statements
theorem difference_between_statements :
  ((A ∨ B) → C) ↔ ¬((A ∧ B) → C) :=
sorry

end first_statement_second_statement_difference_between_statements_l764_764230


namespace tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l764_764957

theorem tan_45_add_reciprocal_half_add_abs_neg_two_eq_five :
  (Real.tan (Real.pi / 4) + (1 / 2)⁻¹ + |(-2 : ℝ)|) = 5 :=
by
  -- Assuming the conditions provided in part a)
  have h1 : Real.tan (Real.pi / 4) = 1 := by sorry
  have h2 : (1 / 2 : ℝ)⁻¹ = 2 := by sorry
  have h3 : |(-2 : ℝ)| = 2 := by sorry

  -- Proof of the problem using the conditions
  rw [h1, h2, h3]
  norm_num

end tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l764_764957


namespace lim_l_div_n2_eq_2_div_7_l764_764066
/-- Prove that lim n→∞ l(n) / n^2 = 2 / 7 given the above conditions. -/
theorem lim_l_div_n2_eq_2_div_7 :
  (∀ (n : ℕ) (k : ℕ) (hk1 : 1 ≤ k) (hk2 : k ≤ n), 
  ∃ (vertices : finset (ℕ × ℕ)), 
    (∀ (r : ℕ) (hr1 : 0 < r) (hr2 : r ≤ k), 
     ∃ (c : ℕ) (hc1 : 0 < c) (hc2 : c ≤ k), (r, c) ∈ vertices) ∧ 
    card vertices = l n) →
  (has_limit (λ n : ℕ, l(n) / (n : ℝ)^2) (2 / 7)) :=
sorry

end lim_l_div_n2_eq_2_div_7_l764_764066


namespace find_trajectory_equation_l764_764245

noncomputable def trajectory_equation (a c : ℝ) (hac : c > a) : (ℝ × ℝ) → Prop :=
  λ P, let b := real.sqrt (c^2 - a^2) in P.1^2 / a^2 - P.2^2 / b^2 = 1

theorem find_trajectory_equation (a c : ℝ) (hac : c > a) :
  ∃ P : ℝ × ℝ, trajectory_equation a c hac P :=
sorry

end find_trajectory_equation_l764_764245


namespace least_number_to_add_l764_764124

theorem least_number_to_add (n : ℕ) (h : n = 17 * 23 * 29) : 
  ∃ k, k + 1024 ≡ 0 [MOD n] ∧ 
       (∀ m, (m + 1024) ≡ 0 [MOD n] → k ≤ m) ∧ 
       k = 10315 :=
by 
  sorry

end least_number_to_add_l764_764124


namespace total_days_needed_l764_764944

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 10
def work_rate_B : ℚ := 1 / 12
def work_rate_C : ℚ := 1 / 15

-- Define the work done by A in 2 days
def work_by_A_in_2_days : ℚ := 2 * work_rate_A

-- Define the combined work rate of B and C
def combined_work_rate_BC : ℚ := work_rate_B + work_rate_C

-- Define the work done by B and C in 4 days
def work_by_BC_in_4_days : ℚ := 4 * combined_work_rate_BC

-- Define the total work done by all in the given days
def total_work_done : ℚ := work_by_A_in_2_days + work_by_BC_in_4_days

-- Define the remaining work left to be done
def remaining_work : ℚ := 1 - total_work_done

-- Define the days required for C to complete the remaining work
def days_by_C_to_complete_remaining_work : ℚ := remaining_work / work_rate_C

-- Define the final total days required
def total_days : ℚ := 2 + 4 + days_by_C_to_complete_remaining_work

-- Proof that the total number of days required is equal to 9
theorem total_days_needed : total_days = 9 := 
by {
  unfold work_rate_A work_rate_B work_rate_C,
  unfold work_by_A_in_2_days combined_work_rate_BC work_by_BC_in_4_days,
  unfold total_work_done remaining_work days_by_C_to_complete_remaining_work,
  unfold total_days,
  sorry
}

end total_days_needed_l764_764944


namespace arithmetic_sequence_sum_l764_764740

def a (n : ℕ) : ℝ := 75 + (n - 1000) * 0.25

theorem arithmetic_sequence_sum :
  99 * 100 * (∑ n in finset.range (2020 - 1580), 1 / (a (1580 + n) * a (1581 + n))) = 60 :=
by {
  sorry
}

end arithmetic_sequence_sum_l764_764740


namespace intersect_is_one_l764_764702

def SetA : Set ℝ := {x | 0 < x ∧ x < 2}

def SetB : Set ℝ := {0, 1, 2, 3}

theorem intersect_is_one : SetA ∩ SetB = {1} :=
by
  sorry

end intersect_is_one_l764_764702


namespace parallelogram_area_proof_l764_764836

noncomputable def parallelogram_area (BC CD : ℝ) (h1 h2 : ℝ) (perimeter : ℝ) : ℝ :=
  BC * h1

theorem parallelogram_area_proof (BC CD h1 h2 perimeter : ℝ) (h1_eq : h1 = 14)
  (h2_eq : h2 = 16) (perimeter_eq : perimeter = 75)
  (ratio_eq : BC * h1 = CD * h2) (perimeter_condition : 2 * (BC + CD) = perimeter) :
  parallelogram_area BC CD h1 h2 perimeter = 280 := by
  rw [h1_eq, h2_eq] at ratio_eq perimeter_condition
  rw parallelogram_area
  sorry

end parallelogram_area_proof_l764_764836


namespace min_x_4y_is_minimum_l764_764263

noncomputable def min_value_x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 / x) + (1 / (2 * y)) = 2) : ℝ :=
  x + 4 * y

theorem min_x_4y_is_minimum : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 2) ∧ (x + 4 * y = (3 / 2) + Real.sqrt 2) :=
sorry

end min_x_4y_is_minimum_l764_764263


namespace quadruple_count_l764_764275

theorem quadruple_count (n : ℕ) (h : 0 < n) : 
  (Finset.card { p : ℕ × ℕ × ℕ × ℕ | 
    let (a, b, c, d) := p in 0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ n }) = Nat.choose (n+4) 4 :=
  sorry

end quadruple_count_l764_764275


namespace eccentricity_of_ellipse_l764_764314

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def parabola_eq (y x : ℝ) : Prop :=
  y ^ 2 = 4 * x

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1

noncomputable def intersect_point (T : ℝ × ℝ) (y : ℝ) : Prop :=
  T = (1, y) ∧ y ^ 2 = 4

noncomputable def perpendicular_condition (T : ℝ × ℝ) : Prop :=
  T.snd = 2

noncomputable def semi_focal_distance (a b : ℝ) : ℝ :=
  real.sqrt (a^2 - b^2)

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ e : ℝ, ellipse 1 2 a b ∧ 
           intersect_point (1, 2) 2 ∧ 
           perpendicular_condition (1, 2) ∧ 
           semi_focal_distance a b = 1 ∧ 
           eccentricity 1 a = e ∧ 
           e = real.sqrt 2 - 1 :=
sorry

end eccentricity_of_ellipse_l764_764314


namespace max_min_abs_l764_764675

-- Definitions of the conditions
variable {z : ℂ} (h1 : |z - 2| = 1)

-- Definitions of the results
noncomputable def max_value := Complex.abs (z + 2 + 5 * Complex.I)
noncomputable def min_value := Complex.abs (z + 2 + 5 * Complex.I)

-- The Lean 4 statement
theorem max_min_abs :
  ∀ z : ℂ, (|z - 2| = 1) → (Complex.abs (z + 2 + 5 * Complex.I) = sqrt 41 + 1 ∨ Complex.abs (z + 2 + 5 * Complex.I) = sqrt 41 - 1) :=
by
  sorry

end max_min_abs_l764_764675


namespace find_perpendicular_tangent_line_l764_764628

noncomputable def line_eq (a b c x y : ℝ) : Prop :=
a * x + b * y + c = 0

def perp_line (a b c d e f : ℝ) (x y : ℝ) : Prop :=
b * x - a * y = 0  -- Perpendicular condition

def tangent_line (f : ℝ → ℝ) (a b c x : ℝ) : Prop :=
∃ t, f t = a * t + b * (f t) + c ∧ (deriv f t) = -a / b  -- Tangency condition with derivative

theorem find_perpendicular_tangent_line :
  let f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 5 in
  ∃ a b c d e f: ℝ, perp_line 2 (-6) 1 a b c ∧ tangent_line f a b c ∧ line_eq 3 1 6 (x : ℝ) (f x) :=
sorry

end find_perpendicular_tangent_line_l764_764628


namespace gas_pressure_inversely_proportional_l764_764946

theorem gas_pressure_inversely_proportional (p v p' v' k : ℝ) (h1 : p * v = k) (h2 : 3 * 6 = k) (h3 : v' = 4.5) : 
  p' = 4 :=
by
  -- Definitions
  have h4 : p = 6 := sorry
  have h5 : v = 3 := sorry
  -- Apply the given conditions
  have h6 : k = 18 := by rw [←h2]
  -- Use the inverse proportionality
  have h7 : p' * v' = k := sorry
  -- Solve for p'
  have h8 : p' = k / v' := sorry
  -- Conclude p'
  rw [h3, h6] at h8
  exact h8.symm.trans (by norm_num [h3])
  sorry

end gas_pressure_inversely_proportional_l764_764946


namespace Anne_cleaning_time_l764_764584

theorem Anne_cleaning_time (B A C : ℚ) 
  (h1 : B + A + C = 1 / 6) 
  (h2 : B + 2 * A + 3 * C = 1 / 2)
  (h3 : B + A = 1 / 4)
  (h4 : B + C = 1 / 3) : 
  A = 1 / 6 := 
sorry

end Anne_cleaning_time_l764_764584


namespace equal_profit_for_Robi_and_Rudy_l764_764049

theorem equal_profit_for_Robi_and_Rudy
  (robi_contrib : ℕ)
  (rudy_extra_contrib : ℕ)
  (profit_percent : ℚ)
  (share_profit_equally : Prop)
  (total_profit: ℚ)
  (each_share: ℕ) :
  robi_contrib = 4000 →
  rudy_extra_contrib = (1/4) * robi_contrib →
  profit_percent = 0.20 →
  share_profit_equally →
  total_profit = profit_percent * (robi_contrib + robi_contrib + rudy_extra_contrib) →
  each_share = (total_profit / 2) →
  each_share = 900 :=
by {
  sorry
}

end equal_profit_for_Robi_and_Rudy_l764_764049


namespace find_largest_number_among_three_l764_764469

noncomputable def A (B : ℝ) := 2 * B - 43
noncomputable def C (A : ℝ) := 0.5 * A + 5

-- The main statement to be proven
theorem find_largest_number_among_three : 
  ∃ (A B C : ℝ), 
  A + B + C = 50 ∧ 
  A = 2 * B - 43 ∧ 
  C = 0.5 * A + 5 ∧ 
  max A (max B C) = 27.375 :=
by
  sorry

end find_largest_number_among_three_l764_764469


namespace eccentricity_ellipse_related_curves_l764_764865

noncomputable def eccentricity_of_ellipse (F1 F2 P : ℝ × ℝ) (c a1 a2 : ℝ) :=
  let e1 := c / a1 in
  let e2 := c / a2 in
  ∠ (F1.1, P.1, F2.1) = 60 ∧
  F1.2 = P.2 ∨ F2.2 = P.2 ∧
    -- conditions for being related curves and reciprocal eccentricities
    m + n = 2 * a1 ∧
    m - n = 2 * a2 ∧
    c * c = a1 * a1 + a2 * a2 ∧
    e1 * e2 = 1

-- The theorem statement to prove
theorem eccentricity_ellipse_related_curves (F1 F2 P : ℝ × ℝ) (c a1 a2 : ℝ) :
  eccentricity_of_ellipse F1 F2 P c a1 a2 → 
  let e1 := c / 3 * a2 in 
  e1 = sqrt 3 / 3 :=
sorry

end eccentricity_ellipse_related_curves_l764_764865


namespace find_CD_l764_764619

noncomputable def C : ℝ := 32 / 9
noncomputable def D : ℝ := 4 / 9

theorem find_CD :
  (∀ x, x ≠ 6 ∧ x ≠ -3 → (4 * x + 8) / (x^2 - 3 * x - 18) = 
       C / (x - 6) + D / (x + 3)) →
  C = 32 / 9 ∧ D = 4 / 9 :=
by sorry

end find_CD_l764_764619


namespace sum_of_diagonals_l764_764016

noncomputable def PQRST_diagonals_sum : ℕ := 90

theorem sum_of_diagonals (PQRST : Type) [InscribedInCircle PQRST] (PQ RS : ℕ) (QR ST : ℕ) (PT : ℕ)
  (h1 : PQ = 4) (h2 : RS = 4) (h3 : QR = 11) (h4 : ST = 11) (h5 : PT = 15) :
  PQRST_diagonals_sum = 90 :=
by sorry

end sum_of_diagonals_l764_764016


namespace probability_face_cards_of_different_suits_l764_764793

-- Conditions definitions
def isFaceCard (card : ℕ) : Prop := card ∈ {11, 12, 13}  -- Assuming J=11, Q=12, K=13
def differentSuits (cards : List ℕ) : Prop := cards.Nodup  -- No duplicates for different suits

-- The main proof problem statement
theorem probability_face_cards_of_different_suits :
  let totalCards := 52
  let totalFaceCards := 12
  let P_firstFaceCard := (totalFaceCards : ℚ) / totalCards
  let P_secondFaceCardDiffSuit := 8 / (totalCards - 1)
  let P_thirdFaceCardDiffSuit := 6 / (totalCards - 2)
  (P_firstFaceCard * P_secondFaceCardDiffSuit * P_thirdFaceCardDiffSuit = 4 / 915) :=
by
  -- This is the way to establish the conditions and prove the result.
  sorry

end probability_face_cards_of_different_suits_l764_764793


namespace remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l764_764644

-- Part (a): Remainder of (1989 * 1990 * 1991 + 1992^2) when divided by 7 is 0.
theorem remainder_of_product_and_square_is_zero_mod_7 :
  (1989 * 1990 * 1991 + 1992^2) % 7 = 0 :=
sorry

-- Part (b): Remainder of 9^100 when divided by 8 is 1.
theorem remainder_of_9_pow_100_mod_8 :
  9^100 % 8 = 1 :=
sorry

end remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l764_764644


namespace configuration_exists_l764_764750

/-- There exists a configuration of 10 lines such that for any set of 8 chosen lines, there is at least
one intersection point not on any of them, and any set of 9 chosen lines covers all intersection points. -/
theorem configuration_exists : ∃ (lines : Fin 10 → Set (ℝ × ℝ)),
  (∀ (s : Finset (Fin 10)), s.card = 8 → ∃ p : ℝ × ℝ, ∀ i ∈ s, p ∉ lines i) ∧
  (∀ (s : Finset (Fin 10)), s.card = 9 → ∀ p : ℝ × ℝ, (∃ i, p ∈ lines i) :=
begin
  sorry
end

end configuration_exists_l764_764750


namespace quadratic_solutions_l764_764842

/-- Solutions to the quadratic equation x^2 - 2x - 8 = 0 are x = 4 and x = -2. -/
theorem quadratic_solutions : 
  ∀ x : ℝ, x^2 - 2 * x - 8 = 0 ↔ (x = 4 ∨ x = -2) :=
by 
  intro x
  simp only [sub_eq_add_neg, add_eq_zero_iff_eq_neg, add_assoc, mul_neg_eq_neg_mul_symm, pow_two, mul_comm]
  calc
    (x + 2) * (x - 4) = 0
    ↔ x = 4 ∨ x = -2 
: sorry

end quadratic_solutions_l764_764842


namespace sum_of_squares_not_prime_l764_764042

theorem sum_of_squares_not_prime {n : ℤ} (h : ∃ a b x y : ℤ, a ≠ x ∧ b ≠ y ∧ n = a^2 + b^2 ∧ n = x^2 + y^2) : ¬ nat.prime (int.nat_abs n) := 
sorry

end sum_of_squares_not_prime_l764_764042


namespace abs_diff_inequality_solution_set_l764_764344

-- Definition of the absolute value difference
def abs_diff := λ (x : ℝ), abs (x + 2) - abs (x + 3)

-- Prove the range for m such that the inequality abs_diff x > m has solutions
theorem abs_diff_inequality_solution_set :
  (∃ x : ℝ, abs_diff x > m) ↔ (m < -1) :=
begin
  sorry
end

end abs_diff_inequality_solution_set_l764_764344


namespace numbers_square_and_cube_root_l764_764834

theorem numbers_square_and_cube_root (x : ℝ) : (x^2 = x ∧ x^3 = x) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by
  sorry

end numbers_square_and_cube_root_l764_764834


namespace max_min_value_f_l764_764649

theorem max_min_value_f (x m : ℝ) : ∃ m : ℝ, (∀ x : ℝ, x^2 - 2*m*x + 8*m + 4 ≥ -m^2 + 8*m + 4) ∧ (∀ n : ℝ, -n^2 + 8*n + 4 ≤ 20) :=
  sorry

end max_min_value_f_l764_764649


namespace perpendicular_tangent_line_l764_764624

theorem perpendicular_tangent_line (a b : ℝ) : 
  let line_slope := 1 / 3,
      perp_slope := -3,
      curve := λ x, x^3 + 3 * x^2 - 5,
      derivative_curve := λ x, 3 * x^2 + 6 * x,
      a_eqn := a = -1,
      b_eqn := b = -3
  in
  (2 * a - 6 * b + 1 = 0) ∧ (a, b) = P ∧ (derivative_curve a = -3) → 
  3 * x + y + 6 = 0 :=
by
  sorry

end perpendicular_tangent_line_l764_764624


namespace sum_of_arithmetic_sequence_6_7_8_l764_764689

theorem sum_of_arithmetic_sequence_6_7_8 {a : ℕ → ℝ} (a1 : ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_13 : ∑ i in Finset.range 13, a (i + 1) = 39) : 
  a 6 + a 7 + a 8 = 39 :=
sorry

end sum_of_arithmetic_sequence_6_7_8_l764_764689


namespace midpoint_coordinates_l764_764384

theorem midpoint_coordinates (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 2) (hy1 : y1 = 10) (hx2 : x2 = 6) (hy2 : y2 = 2) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx = 4 ∧ my = 6 :=
by
  sorry

end midpoint_coordinates_l764_764384


namespace even_adjacent_red_squares_each_row_l764_764591

variable (n : ℕ)

/-- A chessboard of size (2n+1) x (2n+1) -/
structure ChessBoard :=
  (squares : Fin (2 * n + 1) × Fin (2 * n + 1) → Bool)

def isRed (b : ChessBoard n) (i j : Fin (2 * n + 1)) : Prop :=
  b.squares (i, j) = true

def pairwiseAdjacentRedSquares (b : ChessBoard n) (row : Fin (2 * n + 1)) : ℕ :=
  let rowSquares := List.map (λ j => isRed b row j) (List.finRange (2 * n + 1))
  List.length (List.filter id (List.zipWith (&&) rowSquares (List.tail rowSquares)))

theorem even_adjacent_red_squares_each_row (b : ChessBoard n) (L : List (Fin (2 * n + 1) × Fin (2 * n + 1))):
  (is_closed_non_intersecting_path L ∧ traverses_all_vertices L) →
  (∀ row : Fin (2 * n + 1), pairwiseAdjacentRedSquares b row % 2 = 0) :=
sorry

end even_adjacent_red_squares_each_row_l764_764591


namespace solve_cubic_eq_l764_764813

theorem solve_cubic_eq (x : ℝ) (h1 : (x + 1)^3 = x^3) (h2 : 0 ≤ x) (h3 : x < 1) : x = 0 :=
by
  sorry

end solve_cubic_eq_l764_764813


namespace sasha_studies_more_avg_l764_764080

noncomputable def average_daily_difference (diffs : List Int) : Float :=
  (List.sum diffs).toFloat / diffs.length.toFloat

theorem sasha_studies_more_avg :
  average_daily_difference [10, -10, 20, 30, -20] = 6 := 
by
  sorry

end sasha_studies_more_avg_l764_764080


namespace question_1_question_2_l764_764031

noncomputable def a_seq (n : ℕ) : ℕ := 2^(n-1)
def b_n (n : ℕ) : ℝ := (a_seq (n + 1) : ℝ) / ((a_seq (n + 1) - 1) * (a_seq (n + 2) - 1))

-- Definition of S_n and a_n sequence
theorem question_1 (n : ℕ) : a_seq n = 2^(n-1) :=
sorry

-- Definition of b_n based on a_n and proving the inequality
theorem question_2 (n : ℕ) : (2/3 : ℝ) ≤ ∑ k in Finset.range n, b_n (k + 1) ∧ ∑ k in Finset.range n, b_n (k + 1) < 1 :=
sorry

end question_1_question_2_l764_764031


namespace minimum_value_h_l764_764449

noncomputable theory

-- Define the conditions
variables {f : ℝ → ℝ}
variable (f_monotonic : ∀ x y : ℝ, (0 < x ∧ 0 < y) → (x < y ↔ f x < f y))
variable (f_eq : ∀ x, (0 < x) → f (f x - Real.log x) = Real.exp 1 + 1)

-- Define the function h
def h (x : ℝ) : ℝ := x * f x - Real.exp 1 * x

-- Statement of the proof problem
theorem minimum_value_h :
  ∃ x ∈ Set.Ioi 0, ∀ y ∈ Set.Ioi 0, h y ≥ (-1 / Real.exp 1) := sorry

end minimum_value_h_l764_764449


namespace real_solutions_count_l764_764596

theorem real_solutions_count :
  ∃ (x : ℝ), 9 * x^2 - 27 * (⌊x⌋ : ℝ) + 22 = 0 → 4 :=
begin
  sorry
end

end real_solutions_count_l764_764596


namespace infinite_set_divisibles_by_2_finite_set_positive_ints_less_than_1_billion_l764_764972

-- Problem 1: Set of all numbers divisible by 2
def divisibles_by_2 := {x : ℤ | ∃ n : ℤ, x = 2 * n}

theorem infinite_set_divisibles_by_2 : set.infinite divisibles_by_2 :=
sorry

-- Problem 2: Set of positive integers less than 1 billion
def positive_ints_less_than_1_billion := {x : ℕ | x > 0 ∧ x < 1000000000}

theorem finite_set_positive_ints_less_than_1_billion : set.finite positive_ints_less_than_1_billion :=
sorry

end infinite_set_divisibles_by_2_finite_set_positive_ints_less_than_1_billion_l764_764972


namespace probability_equal_dice_l764_764528

noncomputable def prob_equal_one_two_digit (n : Nat) (p : ℚ) := 
  (Finset.card (Finset.range 10) : ℚ) / n

noncomputable def prob_equal_two_digit (n : Nat) (p : ℚ) := 
  (Finset.card (Finset.Icc 10 15) : ℚ) / n

theorem probability_equal_dice (n : Nat) (k : Nat) : 
  let p1 := prob_equal_one_two_digit 15,
      p2 := prob_equal_two_digit 15,
      bincomb := nat.choose 6 3,
      prob_part := (p1 ^ 3) * (p2 ^ 3)
  in (bincomb * prob_part = (4320 : ℚ) / 15625) :=
by sorry

end probability_equal_dice_l764_764528


namespace total_canoes_built_by_may_l764_764583

-- Define the sequence of canoes built each month
def b : ℕ → ℕ
| 0     := 5
| (n+1) := 3 * (b n)

-- Define the sum of canoes built from January to May
def total_canoes_built (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, b i

-- Theorem stating the total number of canoes built by the end of May
theorem total_canoes_built_by_may : total_canoes_built 5 = 605 := by
  sorry

end total_canoes_built_by_may_l764_764583


namespace total_apples_eaten_l764_764199

-- Define the variables based on the conditions
variable (tuesday_apples : ℕ)
variable (wednesday_apples : ℕ)
variable (thursday_apples : ℕ)
variable (total_apples : ℕ)

-- Define the conditions
def cond1 : Prop := tuesday_apples = 4
def cond2 : Prop := wednesday_apples = 2 * tuesday_apples
def cond3 : Prop := thursday_apples = tuesday_apples / 2

-- Define the total apples
def total : Prop := total_apples = tuesday_apples + wednesday_apples + thursday_apples

-- Prove the equivalence
theorem total_apples_eaten : 
  cond1 → cond2 → cond3 → total_apples = 14 :=
by 
  sorry

end total_apples_eaten_l764_764199


namespace num_divisors_8n3_eq_280_l764_764777

/-- Let n be an odd integer with exactly 12 positive divisors. -/
def odd_integer_with_12_divisors (n : ℕ) : Prop :=
  nat.odd n ∧ (nat.num_divisors n = 12)

/-- The number of positive divisors of 8 * n^3 is 280. -/
theorem num_divisors_8n3_eq_280 (n : ℕ) (h : odd_integer_with_12_divisors n) : 
  nat.num_divisors (8 * n ^ 3) = 280 := 
sorry

end num_divisors_8n3_eq_280_l764_764777


namespace point_in_fourth_quadrant_l764_764292

theorem point_in_fourth_quadrant (m : ℝ) (h1 : m + 2 > 0) (h2 : m < 0) : -2 < m ∧ m < 0 := by
  sorry

end point_in_fourth_quadrant_l764_764292


namespace simplify_and_evaluate_l764_764810

theorem simplify_and_evaluate (x : ℝ) (hx : x = real.sqrt 7) :
  ((2 / (x - 3) - 1 / (x + 3)) / ((x ^ 2 + 9 * x) / (x ^ 2 - 9))) = real.sqrt 7 / 7 :=
by
  sorry

end simplify_and_evaluate_l764_764810


namespace probability_multiple_3_prime_l764_764107

/-- Prove that the probability of selecting a ticket with a number 
    that is both a multiple of 3 and a prime number, given that 
    tickets are numbered from 1 to 50 and selected randomly, is 1/50. -/
theorem probability_multiple_3_prime (n : ℕ) (H : 1 ≤ n ∧ n ≤ 50) :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  let multiples_of_3_and_prime := [3] in
  let total_tickets := 50 in
  (multiples_of_3_and_prime.length : ℚ) / total_tickets = 1 / 50 :=
by
  sorry

end probability_multiple_3_prime_l764_764107


namespace sqrt_meaningful_iff_ge_neg3_l764_764717

theorem sqrt_meaningful_iff_ge_neg3 (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 3)) ↔ x ≥ -3 :=
by
  sorry

end sqrt_meaningful_iff_ge_neg3_l764_764717


namespace find_price_of_second_variety_l764_764069

def price_of_second_variety
  (q1 q3 mixture_worth : ℝ)
  (ratio_q1_q2_q3 : ℕ × ℕ × ℕ)
  (total_value_per_kg : ℝ) :
  ℝ :=
  let total_weight := (ratio_q1_q2_q3.1 + ratio_q1_q2_q3.2 + ratio_q1_q2_q3.3 : ℝ) in
  let total_value := total_weight * total_value_per_kg in
  (total_value - q1 - (q3 * ratio_q1_q2_q3.3))/(ratio_q1_q2_q3.2 : ℝ)

theorem find_price_of_second_variety :
  price_of_second_variety 126 175.5 153 (1, 1, 2) 153 = 135 := 
by
  sorry

end find_price_of_second_variety_l764_764069


namespace max_value_φ_difference_l764_764389

open Nat

def φ (n : ℕ) : ℕ := (finset.range n).filter (nat.coprime n).card

theorem max_value_φ_difference : ∃ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ φ(n^2 + 2 * n) - φ(n^2) = 72 := 
sorry

end max_value_φ_difference_l764_764389


namespace total_apples_eaten_l764_764195

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end total_apples_eaten_l764_764195


namespace total_cars_at_end_of_play_l764_764010

def carsInFront : ℕ := 100
def carsInBack : ℕ := 2 * carsInFront
def additionalCars : ℕ := 300

theorem total_cars_at_end_of_play : carsInFront + carsInBack + additionalCars = 600 := by
  sorry

end total_cars_at_end_of_play_l764_764010


namespace lawrence_county_population_l764_764228

variable (att5_8 att9_12 att13_15 att16_18 : ℝ)
variable (total_attending5_8 total_attending9_12 total_attending13_15 total_attending16_18 : ℕ)

def total_kids (total_attending : ℕ) (attendance_rate : ℝ) : ℕ :=
  (total_attending : ℝ) / attendance_rate |> floor

def total_kids_lc (total_attending5_8 total_attending9_12 total_attending13_15 total_attending16_18 : ℕ)
 (att5_8 att9_12 att13_15 att16_18 : ℝ) : ℕ :=
  total_kids total_attending5_8 att5_8 +
  total_kids total_attending9_12 att9_12 +
  total_kids total_attending13_15 att13_15 +
  total_kids total_attending16_18 att16_18

theorem lawrence_county_population :
  att5_8 = 0.45 → att9_12 = 0.55 → att13_15 = 0.60 → att16_18 = 0.35 →
  total_attending5_8 = 181320 → total_attending9_12 = 245575 →
  total_attending13_15 = 180224 → total_attending16_18 = 135471 →
  total_kids total_attending5_8 att5_8 = 402933 ∧ 
  total_kids total_attending9_12 att9_12 = 446500 ∧
  total_kids total_attending13_15 att13_15 = 300373 ∧
  total_kids total_attending16_18 att16_18 = 387060 ∧
  total_kids_lc total_attending5_8 total_attending9_12 total_attending13_15
    total_attending16_18 att5_8 att9_12 att13_15 att16_18 = 1536866 :=
by
  intros
  sorry

end lawrence_county_population_l764_764228


namespace stadium_visibility_time_l764_764429

theorem stadium_visibility_time:
  (∃ t : ℕ, 
    let steven_speed := 4,
    let linda_speed := 2,
    let path_distance := 300,
    let stadium_diameter := 200,
    let motion_distance := 300,
    (7 * t = 225) ∧ 
    (t = 225) ∧
    ((numerator_denominator_sum t) = 226)
  ) :=
sorry

def numerator_denominator_sum (t : ℕ) : ℕ :=
  let numerator := 225 in
  let denominator := 1 in
  numerator + denominator

end stadium_visibility_time_l764_764429


namespace area_of_closed_figure_l764_764238

noncomputable def integral_of_difference (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, (f x - g x)

theorem area_of_closed_figure :
  integral_of_difference sqrt (λ x, x^2) 0 1 = 1 / 3 :=
sorry

end area_of_closed_figure_l764_764238


namespace tangent_line_length_l764_764780

def Circle (center : Real × Real) (radius : Real) :=
  ∀ (P : Real × Real), dist P center = radius

def LineSegment (P Q : Real × Real) (length : Real) :=
  dist P Q = length

theorem tangent_line_length
  (C1 : Circle (12, 0) 5)
  (C2 : Circle (-18, 0) 8) :
  ∃ R S : Real × Real, Tangent R (12, 0) ∧ Tangent S (-18, 0) ∧ LineSegment R S 30 :=
sorry

end tangent_line_length_l764_764780


namespace pyramid_inscribed_sphere_tangency_and_vertices_circle_l764_764939

theorem pyramid_inscribed_sphere_tangency_and_vertices_circle
  (n : ℕ)
  (pyramid : Pyramid n)
  (inscribed_sphere : Sphere)
  (tangent_points_coincide : ∀ (face : Face pyramid), 
     tangent_face_sphere face inscribed_sphere = H)
  (vertices_lie_on_circle : ∀ (v : Vertex face), 
     distance v H = radius)
  : ∀ (face : Face pyramid),
      tangent_points_coincide face inscribed_sphere H ∧
      vertices_lie_on_circle H :=
  sorry

end pyramid_inscribed_sphere_tangency_and_vertices_circle_l764_764939


namespace distance_AO_min_distance_BM_l764_764366

open Real

-- Definition of rectangular distance
def rectangular_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

-- Point A and O
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)

-- Point B
def B : ℝ × ℝ := (1, 0)

-- Line "x - y + 2 = 0"
def on_line (M : ℝ × ℝ) : Prop :=
  M.1 - M.2 + 2 = 0

-- Proof statement 1: distance from A to O is 4
theorem distance_AO : rectangular_distance A O = 4 := 
sorry

-- Proof statement 2: minimum distance from B to any point on the line is 3
theorem min_distance_BM (M : ℝ × ℝ) (h : on_line M) : rectangular_distance B M = 3 := 
sorry

end distance_AO_min_distance_BM_l764_764366


namespace division_points_form_parallelogram_centers_are_collinear_l764_764443

variables (A B C D A1 B1 C1 D1 A0 B0 C0 D0 : Type*) [affine_space ℝ A]
variables (P Q R S P1 Q1 R1 S1 P0 Q0 R0 S0 : Type*) [affine_space ℝ P]

-- Assuming the points lie in an affine space and defining the segments dividing the points in equal ratios.
hypothesis h1 : ∀ (t : ℝ), t ∈ (0, 1) → 
  (affine_combination ℝ A A1 t = A0 ∧
   affine_combination ℝ B B1 t = B0 ∧
   affine_combination ℝ C C1 t = C0 ∧
   affine_combination ℝ D D1 t = D0)

-- Defining the parallelograms.
def is_parallelogram (A B C D : Type*) := 
  (∀ (t : ℝ), t ∈ (0, 1) → affine_combination ℝ A B t = affine_combination ℝ D C t)

-- Proving that A0, B0, C0, D0 form a parallelogram.
theorem division_points_form_parallelogram :
  is_parallelogram A B C D → 
  is_parallelogram A1 B1 C1 D1 →
  is_parallelogram A0 B0 C0 D0 :=
sorry

-- Defining the center of a parallelogram.
def center_of_parallelogram (A B C D : Type*) : Type* := 
  affine_combination ℝ (affine_combination ℝ A C 0.5) (affine_combination ℝ B D 0.5) 0.5

-- Proving the centers are collinear.
theorem centers_are_collinear :
  is_parallelogram A B C D → 
  is_parallelogram A1 B1 C1 D1 →
  collinear ([center_of_parallelogram A B C D, center_of_parallelogram A1 B1 C1 D1, center_of_parallelogram A0 B0 C0 D0]: list (Type*)) :=
sorry

end division_points_form_parallelogram_centers_are_collinear_l764_764443


namespace maximize_profit_correct_l764_764534

noncomputable def maximize_profit : ℝ × ℝ :=
  let initial_selling_price : ℝ := 50
  let purchase_price : ℝ := 40
  let initial_sales_volume : ℝ := 500
  let sales_volume_decrease_rate : ℝ := 10
  let x := 20
  let optimal_selling_price := initial_selling_price + x
  let maximum_profit := -10 * x^2 + 400 * x + 5000
  (optimal_selling_price, maximum_profit)

theorem maximize_profit_correct :
  maximize_profit = (70, 9000) :=
  sorry

end maximize_profit_correct_l764_764534


namespace value_of_q_l764_764703

theorem value_of_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 :=
by
  sorry

end value_of_q_l764_764703


namespace max_value_of_x_l764_764784

theorem max_value_of_x :
  ∀ (x y : ℕ) (a b : ℕ) (α β : ℝ),
    x < y →
    a^2 + b^2 = 5 →
    α + β = 1 →
    ∃ a' α' b' β', ⌊real.log10 x⌋ = a' ∧ frac (real.log10 x) = α' ∧
                      ⌊real.log10 y⌋ = b' ∧ frac (real.log10 y) = β' ∧
                      x ≤ 80 :=
by
  sorry

end max_value_of_x_l764_764784


namespace g_symmetric_l764_764941

noncomputable def g (x : ℝ) : ℝ := |⌊2 * x⌋| - |⌊2 - 2 * x⌋|

theorem g_symmetric : ∀ x : ℝ, g x = g (1 - x) := by
  sorry

end g_symmetric_l764_764941


namespace sum_of_selected_terms_l764_764686

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function from natural numbers to rational numbers

noncomputable def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_of_selected_terms (h₁ : sum_first_n_terms a 13 = 39) : a 6 + a 7 + a 8 = 13 :=
sorry

end sum_of_selected_terms_l764_764686


namespace E1_E2_complementary_l764_764106

-- Define the universal set for a fair die with six faces
def universalSet : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define each event as a set based on the problem conditions
def E1 : Set ℕ := {1, 3, 5}
def E2 : Set ℕ := {2, 4, 6}
def E3 : Set ℕ := {4, 5, 6}
def E4 : Set ℕ := {1, 2}

-- Define complementary events
def areComplementary (A B : Set ℕ) : Prop :=
  (A ∪ B = universalSet) ∧ (A ∩ B = ∅)

-- State the theorem that events E1 and E2 are complementary
theorem E1_E2_complementary : areComplementary E1 E2 :=
sorry

end E1_E2_complementary_l764_764106


namespace live_flowers_l764_764529

theorem live_flowers (total withered : ℕ) (h₁ : total = 13) (h₂ : withered = 7) : total - withered = 6 := by
  rw [h₁, h₂]
  exact rfl

end live_flowers_l764_764529


namespace fifth_runner_twice_as_fast_reduction_l764_764247

/-- Given the individual times of the five runners and the percentage reductions, 
if the fifth runner had run twice as fast, the total time reduction is 8%. -/
theorem fifth_runner_twice_as_fast_reduction (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h2 : T1 = 0.10 * T)
  (h3 : T2 = 0.20 * T)
  (h4 : T3 = 0.24 * T)
  (h5 : T4 = 0.30 * T)
  (h6 : T5 = 0.16 * T) :
  let T' := T1 + T2 + T3 + T4 + T5 / 2 in
  T - T' = 0.08 * T :=
by
  sorry

end fifth_runner_twice_as_fast_reduction_l764_764247


namespace notebooks_difference_l764_764406

theorem notebooks_difference 
  (cost_mika : ℝ) (cost_leo : ℝ) (notebook_price : ℝ)
  (h_cost_mika : cost_mika = 2.40)
  (h_cost_leo : cost_leo = 3.20)
  (h_notebook_price : notebook_price > 0.10)
  (h_mika : ∃ m : ℕ, cost_mika = m * notebook_price)
  (h_leo : ∃ l : ℕ, cost_leo = l * notebook_price)
  : ∃ n : ℕ, (l - m = 4) :=
by
  sorry

end notebooks_difference_l764_764406


namespace correct_quadrilateral_statement_l764_764128

theorem correct_quadrilateral_statement :
  (∀ (Q : Type) [Quadrilateral Q], 
    (∀ (p1 p2 p3 p4 : Point Q), 
      is_rectangle Q p1 p2 p3 p4 → 
      (diagonal_length Q p1 p3 = diagonal_length Q p2 p4 ∧ bisect Q p1 p3 p2 p4)) ∧
    (∃ (Q' : Type) [Quadrilateral Q'], 
      (∃ (p1' p2' p3' p4' : Point Q'), 
        equal_diagonals Q' p1' p2' p3' p4' ∧ 
        ¬ is_rectangle Q' p1' p2' p3' p4')) ∧
    (∃ (Q'' : Type) [Quadrilateral Q''], 
      (∃ (p1'' p2'' p3'' p4'' : Point Q''), 
        perpendicular_diagonals Q'' p1'' p2'' p3'' p4'' ∧ 
        ¬ is_square Q'' p1'' p2'' p3'' p4'')) ∧
    (∀ (P : Type) [Parallelogram P], 
      (∀ (p1 p2 p3 p4 : Point P), 
        is_parallelogram P p1 p2 p3 p4 → bisect Q p1 p3 p2 p4))) →
  (statement A : Quadrilateral Q → ∀ (p1 p2 p3 p4 : Point Q) → equal_diagonals Q p1 p3 p2 p4 → is_rectangle Q p1 p2 p3 p4) = false ∧
  (statement B : Quadrilateral Q → ∀ (p1 p2 p3 p4 : Point Q) → perpendicular_diagonals Q p1 p3 p2 p4 → is_square Q p1 p2 p3 p4) = false ∧
  (statement C : Parallelogram P → ∀ (p1 p2 p3 p4 : Point P) → bisect P p1 p3 p2 p4 = true) = correct ∧
  (statement D : Rectangle R → ∀ (p1 p2 p3 p4 : Point R) → diagonal_length R p1 p3 = diagonal_length R p2 p4 ∧ bisect R p1 p3 p2 p4) = correct
  := by sorry

end correct_quadrilateral_statement_l764_764128


namespace exists_triangle_with_given_median_and_altitude_l764_764226

-- Define the triangle and required properties
structure Triangle (α : Type) [LinearOrder α] :=
(A B C : α) (AB AC BC : ℝ)
(median : ℝ) (altitude : ℝ)
(median_eq_AB : median = AB)
(altitude_eq_AC : altitude = AC)

-- Existence statement
theorem exists_triangle_with_given_median_and_altitude (α : Type) [LinearOrder α] :
  ∃ (T : Triangle α), T.altitude_eq_AC ∧ T.median_eq_AB := by
  admit -- To be proven

end exists_triangle_with_given_median_and_altitude_l764_764226


namespace meaningful_sqrt_iff_l764_764086

theorem meaningful_sqrt_iff (x : ℝ) : (∃ y : ℝ, (y = real.sqrt (x - 2))) ↔ (x ≥ 2) :=
by
    sorry

end meaningful_sqrt_iff_l764_764086


namespace red_candies_remain_percentage_l764_764131

noncomputable def percent_red_candies_remain (N : ℝ) : ℝ :=
let total_initial_candies : ℝ := 5 * N
let green_candies_eat : ℝ := N
let remaining_after_green : ℝ := total_initial_candies - green_candies_eat

let half_orange_candies_eat : ℝ := N / 2
let remaining_after_half_orange : ℝ := remaining_after_green - half_orange_candies_eat

let half_all_remaining_candies_eat : ℝ := (N / 2) + (N / 4) + (N / 2) + (N / 2)
let remaining_after_half_all : ℝ := remaining_after_half_orange - half_all_remaining_candies_eat

let final_remaining_candies : ℝ := 0.32 * total_initial_candies
let candies_to_eat_finally : ℝ := remaining_after_half_all - final_remaining_candies
let each_color_final_eat : ℝ := candies_to_eat_finally / 2

let remaining_red_candies : ℝ := (N / 2) - each_color_final_eat

(remaining_red_candies / N) * 100

theorem red_candies_remain_percentage (N : ℝ) : percent_red_candies_remain N = 42.5 := by
  -- Proof skipped
  sorry

end red_candies_remain_percentage_l764_764131


namespace target_percentage_is_correct_l764_764547

-- Define the initial conditions
def initial_volume : ℝ := 125
def initial_water_percentage : ℝ := 0.20
def added_water_volume : ℝ := 8.333333333333334

-- Calculate the initial amount of water
def initial_water_volume : ℝ := initial_volume * initial_water_percentage

-- Calculate the new volume of the mixture
def new_total_volume : ℝ := initial_volume + added_water_volume

-- Calculate the new amount of water
def new_water_volume : ℝ := initial_water_volume + added_water_volume

-- Define the target percentage of water
def target_water_percentage : ℝ := (new_water_volume / new_total_volume) * 100

theorem target_percentage_is_correct : target_water_percentage = 25 := by
  sorry  -- proof is not required, only the statement

end target_percentage_is_correct_l764_764547


namespace negation_of_existential_square_inequality_l764_764459

theorem negation_of_existential_square_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_square_inequality_l764_764459


namespace circle_and_ellipse_focus_l764_764657

theorem circle_and_ellipse_focus (n m : ℝ) (hC : ∀ (x y : ℝ), x^2 + (y + 1)^2 = n)
    (hFocus : ∃ (f₁ f₂ : ℝ), f₁ = (0, -1) ∧ f₂ = (0, 1) ∧ ell_focus n m f₂) :
    n / m = 8 :=
by
  sorry

end circle_and_ellipse_focus_l764_764657


namespace calculate_brick_quantity_l764_764709

noncomputable def brick_quantity (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ) : ℝ :=
  let brick_volume := brick_length * brick_width * brick_height
  let wall_volume := wall_length * wall_height * wall_width
  wall_volume / brick_volume

theorem calculate_brick_quantity :
  brick_quantity 20 10 8 1000 800 2450 = 1225000 := 
by 
  -- Volume calculations are shown but proof is omitted
  sorry

end calculate_brick_quantity_l764_764709


namespace inner_rectangle_length_l764_764175

theorem inner_rectangle_length :
  ∃ x : ℕ, (∀ (a1 a2 a3 : ℕ), a1 = x ∧ a2 = 2 * x + 6 ∧ a3 = 2 * x + 14 
  → a2 - a1 = a3 - a2 ∧ x = 2) :=
begin
  sorry
end

end inner_rectangle_length_l764_764175


namespace probability_of_point_in_sphere_l764_764550

noncomputable def probability_inside_sphere : ℝ :=
  let cube_volume := 4 ^ 3 in
  let sphere_volume := (4 / 3) * Real.pi * (2 ^ 3) in
  sphere_volume / cube_volume

theorem probability_of_point_in_sphere :
  ∀ (x y z : ℝ), 
    (-2 ≤ x ∧ x ≤ 2) ∧ 
    (-2 ≤ y ∧ y ≤ 2) ∧ 
    (-2 ≤ z ∧ z ≤ 2) → 
    (probability_inside_sphere = (Real.pi / 6)) := by
  sorry

end probability_of_point_in_sphere_l764_764550


namespace part_a_smallest_N_part_b_smallest_15_l764_764497

-- Define property (P) for a number
def has_property_P (n : ℕ) : Prop :=
  ∃ p k, prime p ∧ k ≥ 3 ∧ p ^ k ∣ n

-- Part (a): Smallest N such that any set of N consecutive numbers contains a number with property (P)
theorem part_a_smallest_N :
  ∀ N : ℕ, (∀ n, ∃ m ∈ finset.range(N + 1).map (λ x, x + n), has_property_P m) ↔ N = 16 :=
sorry

-- Part (b): Smallest set of 15 consecutive numbers whose sum of multiples has property (P)
theorem part_b_smallest_15 :
  let seq := list.range' 1 15 in
  (∀ n ∈ seq, ¬ has_property_P n) ∧ has_property_P (5 * seq.sum) :=
sorry

end part_a_smallest_N_part_b_smallest_15_l764_764497


namespace min_hours_to_pass_message_ge_55_l764_764549

theorem min_hours_to_pass_message_ge_55 : 
  ∃ (n: ℕ), (∀ m: ℕ, m < n → 2^(m+1) - 2 ≤ 55) ∧ 2^(n+1) - 2 > 55 :=
by sorry

end min_hours_to_pass_message_ge_55_l764_764549


namespace max_divisible_integers_l764_764934

theorem max_divisible_integers (a n : ℕ) :
  let integers := finset.range (2 * n);
  let divisors := finset.range (n + 1) \ finset.range (1);
  ∃ m : ℕ, 
  (m ≤ n + ⌊(n : ℚ) / 2⌋) ∧ 
  (finset.card (finset.filter (λ x, ∃ d ∈ divisors, (x + a) % d = 0) integers) = m) :=
sorry

end max_divisible_integers_l764_764934


namespace possible_triangle_sides_l764_764499

theorem possible_triangle_sides :
  {x : ℤ | 3 < x ∧ x < 13}.finite.toFinset.card = 9 := 
sorry

end possible_triangle_sides_l764_764499


namespace stocking_stuffers_total_l764_764327

-- Defining the number of items per category
def candy_canes := 4
def beanie_babies := 2
def books := 1
def small_toys := 3
def gift_cards := 1

-- Total number of stocking stuffers per child
def items_per_child := candy_canes + beanie_babies + books + small_toys + gift_cards

-- Number of children
def number_of_children := 3

-- Total number of stocking stuffers for all children
def total_stocking_stuffers := items_per_child * number_of_children

-- Statement to be proved
theorem stocking_stuffers_total : total_stocking_stuffers = 33 := by
  sorry

end stocking_stuffers_total_l764_764327


namespace flash_rate_l764_764160

theorem flash_rate (flashes : ℕ) (time_in_seconds : ℕ) (h1 : flashes = 120) (h2 : time_in_seconds = 900) :
  time_in_seconds / flashes = 7.5 :=
by sorry

end flash_rate_l764_764160


namespace tom_pays_correct_amount_l764_764485

-- Definitions based on conditions
def cost_of_apples := 8 * 70     -- => 560
def cost_of_mangoes := 9 * 90    -- => 810
def cost_of_grapes := 5 * 150    -- => 750
def total_amount_before_discount := cost_of_apples + cost_of_mangoes + cost_of_grapes -- => 2120
def discount_rate := 0.10
def discount_amount := discount_rate * total_amount_before_discount -- => 212
def total_amount_after_discount := total_amount_before_discount - discount_amount -- => 1908
def tax_rate := 0.05
def sales_tax_amount := tax_rate * total_amount_after_discount -- => 95.4
def final_amount_tom_pays := total_amount_after_discount + sales_tax_amount -- => 2003.4

-- The theorem to prove
theorem tom_pays_correct_amount :
  final_amount_tom_pays = 2003.4 :=
by
  sorry

end tom_pays_correct_amount_l764_764485


namespace ellipse_properties_l764_764280

-- Define the conditions
def ellipse_foci : ℝ × ℝ × ℝ × ℝ := (-2 * sqrt 6, 0, 2 * sqrt 6, 0)
def point_A : ℝ × ℝ := (2 * sqrt 6, 2)
def line_length_AB : ℝ := 6

-- Define the standard equation of the ellipse
def standard_equation_ellipse (a b : ℝ) : Prop :=
  a = 6 ∧
  b = sqrt (a^2 - (2 * sqrt 6)^2) ∧
  (a > b ∧ b > 0) ∧
  (2 * a =
    sqrt ((2 * sqrt 6 + 2 * sqrt 6)^2 + 2^2) +
    sqrt ((2 * sqrt 6 - 2 * sqrt 6)^2 + 2^2))

-- Define the maximum area of triangle AOB
def max_area_triangle_AOB (area : ℝ) : Prop := 
  ∃ k m : ℝ, 
  (|line_length_AB| = 6 → 
  (area = 1 / 2 * 6 * abs m / sqrt (k^2 + 1)) ∧
  (abs m = sqrt (3 (3 * k^2 + 1) (k^2 + 3) / (k^2 + 1))))

-- Combining the results with the given conditions
theorem ellipse_properties :
  (∃ a b : ℝ, standard_equation_ellipse a b) ∧ 
  max_area_triangle_AOB 9 :=
by sorry

end ellipse_properties_l764_764280


namespace denis_fourth_board_points_l764_764604

theorem denis_fourth_board_points :
  ∃ x : ℕ, (30 + 38 = 2 * x) ∧ x = 34 :=
by
  use 34
  split
  · norm_num
  · norm_num

end denis_fourth_board_points_l764_764604


namespace scientific_notation_of_neg_0_000008691_l764_764231

theorem scientific_notation_of_neg_0_000008691:
  -0.000008691 = -8.691 * 10^(-6) :=
sorry

end scientific_notation_of_neg_0_000008691_l764_764231


namespace min_value_is_zero_l764_764023

noncomputable def min_expression_value (x : ℝ) (h : x > 0) : ℝ :=
  let u := x + 1/x in
  2 * (u^2 - 2 * u)

theorem min_value_is_zero (x : ℝ) (hx : x > 0) (hxy : x = x):
  min_expression_value x hx = 0 :=
by
  have h: x = 1 := sorry
  exact h

end min_value_is_zero_l764_764023


namespace speed_of_water_l764_764169

theorem speed_of_water (v : ℝ) :
  (∀ (distance time : ℝ), distance = 16 ∧ time = 8 → distance = (4 - v) * time) → 
  v = 2 :=
by
  intro h
  have h1 : 16 = (4 - v) * 8 := h 16 8 (by simp)
  sorry

end speed_of_water_l764_764169


namespace sum_segments_property_l764_764041

variables {Point Segment : Type} [Add Point] [Add Segment]

/-- A function to check if two segments are parallel -/
def are_parallel (s1 s2 : Segment) : Prop :=
  sorry

/-- A function to find the length of a segment -/
def segment_length (s : Segment) : ℝ :=
  sorry

/-- The sum of two segments resulting in another geometric figure -/
def sum_segments (s1 s2 : Segment) : Point :=
  sorry

theorem sum_segments_property (Φ₁ Φ₂ : Segment) :
  (¬ are_parallel Φ₁ Φ₂ → ∃ p1 p2 p3 p4 : Point, sum_segments Φ₁ Φ₂ = parallelogram p1 p2 p3 p4)
  ∧ (are_parallel Φ₁ Φ₂ → ∃ (s : Segment), sum_segments Φ₁ Φ₂ = s ∧ segment_length s = segment_length Φ₁ + segment_length Φ₂ ∧ are_parallel s Φ₁ ∧ are_parallel s Φ₂) :=
sorry

end sum_segments_property_l764_764041


namespace compute_58_sq_pattern_l764_764359

theorem compute_58_sq_pattern : (58 * 58 = 56 * 60 + 4) :=
by
  sorry

end compute_58_sq_pattern_l764_764359


namespace avg_temp_in_october_l764_764466

theorem avg_temp_in_october (a A : ℝ)
  (h1 : 28 = a + A)
  (h2 : 18 = a - A)
  (x := 10)
  (temperature : ℝ := a + A * Real.cos (π / 6 * (x - 6))) :
  temperature = 20.5 :=
by
  sorry

end avg_temp_in_october_l764_764466


namespace reducedRatesFraction_l764_764879

variable (total_hours_per_week : ℕ := 168)
variable (reduced_rate_hours_weekdays : ℕ := 12 * 5)
variable (reduced_rate_hours_weekends : ℕ := 24 * 2)

theorem reducedRatesFraction
  (h1 : total_hours_per_week = 7 * 24)
  (h2 : reduced_rate_hours_weekdays = 12 * 5)
  (h3 : reduced_rate_hours_weekends = 24 * 2) :
  (reduced_rate_hours_weekdays + reduced_rate_hours_weekends) / total_hours_per_week = 9 / 14 := 
  sorry

end reducedRatesFraction_l764_764879


namespace total_cost_is_correct_l764_764385

-- Define the conditions
def piano_cost : ℝ := 500
def lesson_cost_per_lesson : ℝ := 40
def number_of_lessons : ℝ := 20
def discount_rate : ℝ := 0.25

-- Define the total cost of lessons before discount
def total_lesson_cost_before_discount : ℝ := lesson_cost_per_lesson * number_of_lessons

-- Define the discount amount
def discount_amount : ℝ := discount_rate * total_lesson_cost_before_discount

-- Define the total cost of lessons after discount
def total_lesson_cost_after_discount : ℝ := total_lesson_cost_before_discount - discount_amount

-- Define the total cost of everything
def total_cost : ℝ := piano_cost + total_lesson_cost_after_discount

-- The statement to be proven
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end total_cost_is_correct_l764_764385


namespace equal_segments_l764_764787

structure Triangle (α : Type*) [Field α] :=
(A B C : α × α)
(is_scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A)
(is_acute : 
  ∀ P ∈ {A, B, C}, ∠P ≠ 90)

noncomputable def midpoint {α : Type*} [Field α] (P Q : α × α) : α × α :=
((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def altitude {α : Type*} [Field α] (A B C : α × α) : α × α :=
sorry -- This should be the function to find the foot of the altitude from A to BC

noncomputable def angle {α : Type*} [Field α] (P Q R : α × α) : α :=
sorry -- This function calculates the angle ∠PQR

theorem equal_segments {α : Type*} [Field α]
  (ABC : Triangle α)
  (A1 := altitude ABC.A ABC.B ABC.C)
  (C1 := altitude ABC.C ABC.A ABC.B)
  (K := midpoint ABC.A ABC.B)
  (L := midpoint ABC.B ABC.C)
  (M := midpoint ABC.C ABC.A)
  (h : angle C1 M A1 = angle ABC.B ABC.C ABC.A)
  : dist C1 K = dist A1 L :=
sorry

end equal_segments_l764_764787


namespace find_x_l764_764234

theorem find_x (x : ℝ) (hx_pos : x > 0) (hx_ceil_eq : ⌈x⌉ = 15) : x = 14 :=
by
  -- Define the condition
  have h_eq : ⌈x⌉ * x = 210 := sorry
  -- Prove that the only solution is x = 14
  sorry

end find_x_l764_764234


namespace contradiction_example_l764_764805

theorem contradiction_example (a b : ℝ) (h : a^2 + |b| = 0) : a = 0 ∧ b = 0 :=
by
  apply classical.by_contradiction
  intro h1
  cases h1 with ha hb
  sorry

end contradiction_example_l764_764805


namespace max_magnitude_of_c_l764_764286

variables {a b c : EuclideanSpace ℝ 3}

theorem max_magnitude_of_c (unit_a : ‖a‖ = 1) 
  (unit_b : ‖b‖ = 1) (orthogonal_ab : dot_product a b = 0)
  (c_property : ‖c - (a + b)‖ = ‖a - b‖) : 
  ‖c‖ ≤ 2 * real.sqrt 2 :=
begin
  sorry
end

end max_magnitude_of_c_l764_764286


namespace find_h_l764_764250

theorem find_h (a h : ℝ) (hs : 1 = 1) (area_small_square: ℝ) (h_area: area_small_square = 1) 
  (small_square_side: ℝ) (h_small_side: small_square_side = sqrt area_small_square)
  (total_length_eqn: 1 + a + 3 = a + h) : h = 4 := 
by
  sorry

end find_h_l764_764250


namespace graph_does_not_pass_first_quadrant_l764_764261

variables {a b x : ℝ}

theorem graph_does_not_pass_first_quadrant 
  (h₁ : 0 < a ∧ a < 1) 
  (h₂ : b < -1) : 
  ¬ ∃ x : ℝ, 0 < x ∧ 0 < a^x + b :=
sorry

end graph_does_not_pass_first_quadrant_l764_764261


namespace black_pens_count_l764_764478

variable (T B : ℕ)
variable (h1 : (3/10:ℚ) * T = 12)
variable (h2 : (1/5:ℚ) * T = B)

theorem black_pens_count (h1 : (3/10:ℚ) * T = 12) (h2 : (1/5:ℚ) * T = B) : B = 8 := by
  sorry

end black_pens_count_l764_764478


namespace quadratic_expression_factorization_l764_764078

theorem quadratic_expression_factorization :
  ∃ c d : ℕ, (c > d) ∧ (x^2 - 18*x + 72 = (x - c) * (x - d)) ∧ (4*d - c = 12) := 
by
  sorry

end quadratic_expression_factorization_l764_764078


namespace triangle_construction_possible_l764_764601

-- Definitions for given conditions
variables {R a : ℝ} {k : ℝ} (hR : R > 0) (ha : a > 0) (hk : k > 0)

theorem triangle_construction_possible :
  ∃ (b c : ℝ), (b / c = k) ∧ (b > 0 ∧ c > 0) ∧
  ∃ (θ θ' : ℝ), (θ ≠ θ') ∧ (2 * R * Real.sin θ = a) ∧ (2 * R * Real.sin θ' = a) ∧
  length_of_sides_in_ratio a b c R :=
sorry

-- An auxiliary definition to describe the sides of the triangle.
def length_of_sides_in_ratio (a b c R : ℝ) : Prop :=
  (b / c = k) ∧ (2 * R * Real.sin (Real.arcsin (a / (2 * R))) = a)

end triangle_construction_possible_l764_764601


namespace median_of_S_is_11_l764_764396

def is_valid_S (S : Set ℕ) : Prop :=
  S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧
  S.card = 7 ∧
  ∀ a b, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)

theorem median_of_S_is_11 (S : Set ℕ) (h : is_valid_S S) : median (S.toList) = 11 := 
sorry

end median_of_S_is_11_l764_764396


namespace total_molecular_weight_is_1317_12_l764_764977

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

def molecular_weight_Al2S3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_S)
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + (1 * atomic_weight_O)
def molecular_weight_CO2 : ℝ := (1 * atomic_weight_C) + (2 * atomic_weight_O)

def total_weight_7_Al2S3 : ℝ := 7 * molecular_weight_Al2S3
def total_weight_5_H2O : ℝ := 5 * molecular_weight_H2O
def total_weight_4_CO2 : ℝ := 4 * molecular_weight_CO2

def total_molecular_weight : ℝ := total_weight_7_Al2S3 + total_weight_5_H2O + total_weight_4_CO2

theorem total_molecular_weight_is_1317_12 : total_molecular_weight = 1317.12 := by
  sorry

end total_molecular_weight_is_1317_12_l764_764977


namespace new_cylinder_volume_l764_764465

theorem new_cylinder_volume (r h : ℝ) (π_ne_zero : 0 < π) (original_volume : π * r^2 * h = 10) : 
  π * (3 * r)^2 * (2 * h) = 180 :=
by
  sorry

end new_cylinder_volume_l764_764465


namespace skew_lines_distance_proof_l764_764660

open Real EuclideanGeometry

noncomputable def skew_lines_distance : ℝ :=
  let A := (6, 0, 0 : ℝ×ℝ×ℝ)
  let C1 := (0, 6, 6 : ℝ×ℝ×ℝ)
  let S := (3, 0, 0 : ℝ×ℝ×ℝ)
  let P := (5, 6, 5 : ℝ×ℝ×ℝ)
  let vec_AC1 := (-6, 6, 6 : ℝ×ℝ×ℝ)
  let vec_SP := (2, 6, 5 : ℝ×ℝ×ℝ)
  let n := (1, -7, 8 : ℝ×ℝ×ℝ)
  let SA := (3, 0, 0 : ℝ×ℝ×ℝ)
  let magnitude_n := Real.sqrt (1^2 + (-7)^2 + 8^2)
  let dot_product := λ (v1 v2 : ℝ×ℝ×ℝ), v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
  abs (dot_product SA n) / magnitude_n

theorem skew_lines_distance_proof : skew_lines_distance = (sqrt 114 / 38 : ℝ) := by
  sorry

end skew_lines_distance_proof_l764_764660


namespace intersect_on_XY_l764_764838

open EuclideanGeometry

theorem intersect_on_XY {X Y Z A1 B1 A2 B2 : Point} (hXYZ: collinear X Y Z) (hXY_ne_YZ: dist X Y ≠ dist Y Z)
  (hA1_on_k1: circle_with_diameter X Y A1) (hB1_on_k1: circle_with_diameter X Y B1)
  (hA2_on_k2: circle_with_diameter Y Z A2) (hB2_on_k2: circle_with_diameter Y Z B2)
  (angleA1Y_A2_90: angle A1 Y A2 = 90)
  (angleB1Y_B2_90: angle B1 Y B2 = 90):
  ∃ P ∈ line_through X Y, intersect (line_through A1 A2) (line_through B1 B2) = {P} :=
sorry

end intersect_on_XY_l764_764838


namespace number_of_subsets_of_union_set_is_8_l764_764284

def setA : Set ℝ := {x | x^2 = x}
def setB : Set ℝ := {x | x^3 = x}
def unionSet := setA ∪ setB

theorem number_of_subsets_of_union_set_is_8 : (set.univ).countSubsets unionSet = 8 := by
  sorry

end number_of_subsets_of_union_set_is_8_l764_764284


namespace one_fourth_of_8_8_is_11_over_5_l764_764992

theorem one_fourth_of_8_8_is_11_over_5 :
  ∀ (a b : ℚ), a = 8.8 → b = 1/4 → b * a = 11/5 :=
by
  assume a b : ℚ,
  assume ha : a = 8.8,
  assume hb : b = 1/4,
  sorry

end one_fourth_of_8_8_is_11_over_5_l764_764992


namespace ellipse_ratio_sum_l764_764218

theorem ellipse_ratio_sum : 
  let ellipse (x y : ℝ) := 3 * x^2 + 2 * x * y + 4 * y^2 - 13 * x - 26 * y + 52 = 0
  in 
  (let k_max := (the maximum value of k such that ∃ x y, y = k * x ∧ ellipse x y),
   k_min := (the minimum value of k such that ∃ x y, y = k * x ∧ ellipse x y)
   in k_max + k_min = 65 / 39) := sorry

end ellipse_ratio_sum_l764_764218


namespace right_triangle_conditions_l764_764935

-- Definitions
def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

-- Conditions
def cond1 (A B C : ℝ) : Prop := A + B = C
def cond2 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def cond3 (A B C : ℝ) : Prop := A = B ∧ B = 2 * C
def cond4 (A B C : ℝ) : Prop := A = 2 * B ∧ B = 3 * C

-- Problem statement
theorem right_triangle_conditions (A B C : ℝ) :
  (cond1 A B C → is_right_triangle A B C) ∧
  (cond2 A B C → is_right_triangle A B C) ∧
  ¬(cond3 A B C → is_right_triangle A B C) ∧
  ¬(cond4 A B C → is_right_triangle A B C) :=
by
  sorry

end right_triangle_conditions_l764_764935


namespace not_divisible_by_8_l764_764378

theorem not_divisible_by_8 : ¬ (456294604884 % 8 = 0) := 
by
  have h : 456294604884 % 1000 = 884 := sorry -- This step reflects the conclusion that the last three digits are 884.
  have h_div : ¬ (884 % 8 = 0) := sorry -- This reflects that 884 is not divisible by 8.
  sorry

end not_divisible_by_8_l764_764378


namespace eccentricity_is_2_plus_sqrt_3_l764_764831

open Real

noncomputable def eccentricity_of_hyperbola : ℝ :=
  let a := 1  -- a needs to be defined properly
  let b := 1  -- b needs to be defined properly
  let c := sqrt (a^2 + b^2)
  let e := c / a
  2 + sqrt 3  -- e is calculated final answer

theorem eccentricity_is_2_plus_sqrt_3 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ c : ℝ, c = sqrt(a^2 + b^2) ∧
  (∃ e : ℝ, e = c / a ∧ e = 2 + sqrt 3)) :=
sorry

end eccentricity_is_2_plus_sqrt_3_l764_764831


namespace handshakes_at_convention_l764_764099

theorem handshakes_at_convention :
  ∃ (total_handshakes : ℕ), total_handshakes = 128 ∧
    ∀ (rep_per_company : ℕ) (num_companies : ℕ) (exclusion_count_A : ℕ) (other_company_exclusion_count : ℕ),
    rep_per_company = 4 →
    num_companies = 5 →
    exclusion_count_A = 4 →
    other_company_exclusion_count = 4 →
    let total_attendees := rep_per_company * num_companies in
    let total_possible_handshakes := total_attendees * (total_attendees - 1) / 2 in
    let handshakes_by_company_A := rep_per_company * (total_attendees - exclusion_count_A) in
    let handshakes_by_other_companies := (total_attendees - rep_per_company) * (total_attendees - exclusion_count_A - rep_per_company) in
    total_handshakes = (handshakes_by_company_A + handshakes_by_other_companies) / 2 :=
begin
  use 128,
  intros,
  sorry
end

end handshakes_at_convention_l764_764099


namespace binomial_prob_lemma_l764_764276

open ProbabilityTheory

noncomputable def X : ℕ → Probability ℕ := binomial 4 (1 / 2)

-- Statement to be proved
theorem binomial_prob_lemma : P(X < 3) = 11 / 16 := sorry

end binomial_prob_lemma_l764_764276


namespace number_of_ways_to_fill_circles_l764_764618

theorem number_of_ways_to_fill_circles :
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ P Q : ℕ, 
    (∀ a b c d e f g h i ∈ numbers,
      a != b → a != c → a != d → a != e → a != f → a != g → a != h → a != i →
      b != c → b != d → b != e → b != f → b != g → b != h → b != i →
      c != d → c != e → c != f → c != g → c != h → c != i →
      d != e → d != f → d != g → d != h → d != i →
      e != f → e != g → e != h → e != i →
      f != g → f != h → f != i →
      g != h → g != i →
      h != i →
      a + b + c + d = P ∧
      d + e + f + g = P ∧
      g + h + i + a = P ∧
      (a^2 + b^2 + c^2 + d^2) % 3 = Q ∧
      (d^2 + e^2 + f^2 + g^2) % 3 = Q ∧
      (g^2 + h^2 + i^2 + a^2) % 3 = Q ∧
      (a + d + g) % 3 = 0 ∧
      (a^2 + d^2 + g^2) % 3 = 0) →
  sorry

end number_of_ways_to_fill_circles_l764_764618


namespace probability_all_students_get_their_own_lunch_l764_764559

def n : ℕ := 4

theorem probability_all_students_get_their_own_lunch :
  let total_possibilities := 4!,
      favorable_outcomes := 1 in
    (favorable_outcomes : ℚ) / total_possibilities = 1 / 24 :=
by {
  sorry
}

end probability_all_students_get_their_own_lunch_l764_764559


namespace slope_of_tangent_line_l764_764299

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x - a * Real.exp (-x))

theorem slope_of_tangent_line (a : ℝ) (h_even : ∀ x, f a (-x) = f a x) :
  a = 1 → ∀ x, x = 1 → (deriv (f a) x = 2 * Real.exp 1) := by
  intros ha h1
  refine Eq.trans (deriv f a x) _
  -- Proof will be inserted here
  sorry

end slope_of_tangent_line_l764_764299


namespace pizza_toppings_combination_l764_764809

theorem pizza_toppings_combination : nat.choose 7 3 = 35 := 
by 
  sorry

end pizza_toppings_combination_l764_764809


namespace tangent_line_at_origin_l764_764341

def f (a : ℕ) (x : ℝ) : ℝ := (x^2 - a*x + a + 1) * Real.exp x

theorem tangent_line_at_origin (a : ℕ) (h : 4 < a ∧ a < 16 / 3) :
  let f' := λ x, (x^2 + (2 - a)*x + 1) * Real.exp x in
  let f_0 := f a 0 in
  let f'_0 := f' 0 in
  a = 5 →
  f_0 = 6 ∧ f'_0 = 1 ∧ ∀ x y : ℝ, (y - f_0 = f'_0 * x) ↔ (x - y + 6 = 0) :=
  by
    sorry

end tangent_line_at_origin_l764_764341


namespace arithmetic_square_root_of_nine_l764_764822

theorem arithmetic_square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l764_764822


namespace sequence_geom_sum_bn_l764_764274

-- Definition and condition for the first part, proving that {S_n - 2} is a geometric sequence
theorem sequence_geom (S : ℕ → ℚ) (a : ℕ → ℚ) (q : ℚ) (hq : q > 0) 
  (hS4 : S 4 = 15/8)
  (h_arith : a 1 + a 2 = 6 * a 3)
  (h_geom : ∀ n, S (n) = (a 1 * (1 - q^n)) / (1 - q)) :
  ∃ q1 > 0, ∀ n > 0, S n - 2 = (-q1)^n :=
by sorry

-- Definition and condition for the second part, finding the sum of the first n terms of {b_n}
theorem sum_bn (a b : ℕ → ℚ) (q : ℚ) (hq : q > 0) 
  (ha : ∀ n, a n = (1/2)^(n-1))
  (hb : ∀ n, b n = a n * (Real.log (2 : ℚ) (a n) - 1)) :
  ∀ n, ∑ i in finset.range n, b (i + 1) = (n + 2) * (1/2)^(n-1) - 4 :=
by sorry

end sequence_geom_sum_bn_l764_764274


namespace exists_angle_geq_2_arcsin_sqrt6_div3_l764_764098

theorem exists_angle_geq_2_arcsin_sqrt6_div3 
  (n : ℕ) (R : ℝ) (points : Fin n -> EuclideanSpace ℝ 3)
  (h₀ : 4 ≤ n)
  (h₁ : ¬ ∃ v : EuclideanSpace ℝ 3, ∀ i : Fin n, inner v (points i) > 0) :
  ∃ i j : Fin n, i ≠ j ∧ ∠ (points i) 0 (points j) ≥ 2 * Real.arcsin (Real.sqrt 6 / 3) := by
  sorry

end exists_angle_geq_2_arcsin_sqrt6_div3_l764_764098


namespace S_symmetric_l764_764817

variables {S : set ℕ}  (H1 : S ⊆ {0, 1, 2, ..., n})
variables (H2 : ∀ p, S.tiles p)

noncomputable def zero_in_S : ∃ 0 ∈ S 
   sorry

theorem S_symmetric :  ∃ (n: ℕ), S ~ -S :=
begin
  sorry
end

end S_symmetric_l764_764817


namespace right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l764_764589

theorem right_triangle_arithmetic_progression_is_345 (a b c : ℕ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ d, b = a + d ∧ c = a + 2 * d)
  : (a, b, c) = (3, 4, 5) :=
by
  sorry

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

noncomputable def sqrt_golden_ratio_div_2 := Real.sqrt ((1 + Real.sqrt 5) / 2)

theorem right_triangle_geometric_progression 
  (a b c : ℝ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ r, b = a * r ∧ c = a * r * r)
  : (a, b, c) = (1, sqrt_golden_ratio_div_2, golden_ratio) :=
by
  sorry

end right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l764_764589


namespace triangle_area_and_coordinates_l764_764487

noncomputable def positive_diff_of_coordinates (A B C R S : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (xr, yr) := R
  let (xs, ys) := S
  if xr = xs then abs (xr - (10 - (x3 - xr)))
  else 0 -- Should never be this case if conditions are properly followed

theorem triangle_area_and_coordinates
  (A B C R S : ℝ × ℝ)
  (h_A : A = (0, 10))
  (h_B : B = (4, 0))
  (h_C : C = (10, 0))
  (h_vertical : R.fst = S.fst)
  (h_intersect_AC : R.snd = -(R.fst - 10))
  (h_intersect_BC : S.snd = 0 ∧ S.fst = 10 - (C.fst - R.fst))
  (h_area : 1/2 * ((R.fst - C.fst) * (R.snd - C.snd)) = 15) :
  positive_diff_of_coordinates A B C R S = 2 * Real.sqrt 30 - 10 := sorry

end triangle_area_and_coordinates_l764_764487


namespace length_of_train_l764_764183

-- Define the given conditions
def train_speed_kmh : ℝ := 45
def time_to_cross_bridge_s : ℝ := 30
def bridge_length_m : ℝ := 255

-- Calculate the speed in meters per second
def train_speed_mps : ℝ := train_speed_kmh * 1000 / 3600

-- Calculate the distance the train travels
def distance_travelled_m : ℝ := train_speed_mps * time_to_cross_bridge_s

-- Define the total distance as the sum of train length and bridge length
def train_length_m : ℝ := distance_travelled_m - bridge_length_m

-- Theorem stating the length of the train
theorem length_of_train : train_length_m = 120 := by
  sorry

end length_of_train_l764_764183


namespace min_value_of_reciprocal_sum_l764_764251

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ab : a + b = 1) : 
    ∃ c : ℝ, c = 3 + 2 * real.sqrt 2 ∧ ∀ x y : ℝ, (0 < x) → (0 < y) → (x + y = 1) → (1 / x + 2 / y) ≥ c := sorry

end min_value_of_reciprocal_sum_l764_764251


namespace maximal_sparse_subset_size_l764_764753

-- Definition of set A
def A : Set (Fin 8 → ℕ) :=
  { v | ∀ i : Fin 8, 1 ≤ v i ∧ v i ≤ i + 2 }

-- Definition of a sparse subset
def is_sparse (X : Set (Fin 8 → ℕ)) : Prop :=
  ∀ (a b ∈ X), a ≠ b → (Finset.univ.filter (λ i, a i ≠ b i)).card ≥ 3

-- Find the maximal possible number of elements in a sparse subset of A
theorem maximal_sparse_subset_size : ∃ (X : Set (Fin 8 → ℕ)), X ⊆ A ∧ is_sparse X ∧ Finset.card (X.toFinset) = 7! :=
  sorry

end maximal_sparse_subset_size_l764_764753


namespace one_fourth_of_8_point8_simplified_l764_764996

noncomputable def one_fourth_of (x : ℚ) : ℚ := x / 4

def convert_to_fraction (x : ℚ) : ℚ := 
  let num := 22
  let denom := 10
  num / denom

def simplify_fraction (num denom : ℚ) (gcd : ℚ) : ℚ := 
  (num / gcd) / (denom / gcd)

theorem one_fourth_of_8_point8_simplified : one_fourth_of 8.8 = (11 / 5) := 
by
  have h : one_fourth_of 8.8 = 2.2 := by sorry
  have h_frac : 2.2 = (22 / 10) := by sorry
  have h_simplified : (22 / 10) = (11 / 5) := by sorry
  rw [h, h_frac, h_simplified]
  exact rfl

end one_fourth_of_8_point8_simplified_l764_764996


namespace cot_225_eq_one_l764_764232

theorem cot_225_eq_one : Real.cot (225 * Real.pi / 180) = 1 :=
by 
  have h_tan_225 : Real.tan (225 * Real.pi / 180) = Real.tan (45 * Real.pi / 180),
  { rw [Real.tan_periodic, show 225 * Real.pi / 180 = Real.pi + 45 * Real.pi / 180 by norm_num],
    exact Real.tan_pi_add (45 * Real.pi / 180) },
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by norm_num,
  rw [h_tan_225, h_tan_45],
  exact Real.cot_one_simp

end cot_225_eq_one_l764_764232


namespace trigonometric_identity_l764_764676

-- Define the conditions and the target statement
theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l764_764676


namespace distance_is_correct_l764_764076

noncomputable def distance_from_point_to_hyperbola_asymptote : ℝ := by
  let a := 3  -- a = sqrt(9)
  let b := 4  -- b = sqrt(16)
  let x₀ := 0
  let y₀ := 3
  let A := 3
  let B := 4
  let C := 0
  let numerator := abs (A * x₀ + B * y₀ + C)
  let denominator := Real.sqrt (A ^ 2 + B ^ 2)
  exact numerator / denominator

theorem distance_is_correct : distance_from_point_to_hyperbola_asymptote = 12 / 5 := by
  sorry

end distance_is_correct_l764_764076


namespace graph_of_y1_no_third_quadrant_l764_764829

theorem graph_of_y1_no_third_quadrant (b : ℝ) (x : ℝ) :
  (∀ x: ℝ, y1 = -x + b) ∧
  (∀ x: ℝ, y2 = -x) ∧
  (y2 = y1 - 2) →
  b = 2 ∧ ∀ x : ℝ, (b - x > 0) :=
begin
  sorry
end

end graph_of_y1_no_third_quadrant_l764_764829


namespace probability_all_three_dice_20_l764_764876

theorem probability_all_three_dice_20 :
  let dice := [20, 19, ⟨_, sorry⟩, ⟨_, sorry⟩, ⟨_, sorry⟩] in
  let remaining_dice := filter (fun x => x ≠ 20) dice.drop(2) in
  dice.length = 5 →
  dice[0] = 20 →
  dice[1] = 19 →
  (3 ≤ count (fun x => x = 20) dice) →
  (count (fun x => x = 20) (dice.drop 2)) ≥ 2 →
  (count (fun x => x = 20) (dice.drop 2)) = 3 →
  (remaining_dice.length = 1) →
  probability (count (fun x => x = 20) (dice.drop 2) = 3) = 1 / 58 :=
by
  sorry

end probability_all_three_dice_20_l764_764876


namespace g_not_monotonically_decreasing_l764_764855

def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x) - sqrt 2 * cos (2 * x) + 1

def f1 (x : ℝ) : ℝ := f (x + π / 4)

def g (x : ℝ) : ℝ := f1 (x) - 1

theorem g_not_monotonically_decreasing :
  ¬ (∀ x ∈ set.Icc (π / 12) (5 * π / 8), ∀ y ∈ set.Icc (π / 12) (5 * π / 8), x < y → g x > g y) :=
sorry

end g_not_monotonically_decreasing_l764_764855


namespace tetrahedron_cross_section_area_l764_764237

theorem tetrahedron_cross_section_area (a : ℝ) : 
  let DM := a * Real.sqrt (2 / 3),
      AL := (a * Real.sqrt 3) / 2 in
  (1/2 * AL * DM = a^2 * Real.sqrt 2 / 4) :=
by
  let DM := a * Real.sqrt (2 / 3)
  let AL := (a * Real.sqrt 3) / 2
  have h1 : 1/2 * AL * DM = (1/2 * ((a * Real.sqrt 3) / 2) * (a * Real.sqrt (2 / 3))),
  sorry

end tetrahedron_cross_section_area_l764_764237


namespace right_angled_triangle_area_l764_764068

/-- 
  Suppose a right-angled triangle is inscribed in a circle of radius 100.
  Let α and β be its acute angles. 
  If tan α = 4 * tan β, then the area of the triangle is 8000.
-/
theorem right_angled_triangle_area
  (r : ℝ)
  (α β : ℝ)
  (h1 : r = 100)
  (h2 : tan α = 4 * tan β)
  (h3 : α + β = π / 2) : 
  let hypotenuse := 2 * r,
      a         := hypotenuse * tan α / sqrt (1 + tan α ^ 2),
      b         := hypotenuse * (1 / (1 + tan α ^ 2)) in
  (1 / 2) * a * b = 8000 :=
by
  sorry -- proof omitted

end right_angled_triangle_area_l764_764068


namespace relationship_among_a_b_c_l764_764020

def a : ℝ := Real.log 3 / Real.log 7
def b : ℝ := -Real.log 7 / Real.log 3
def c : ℝ := 3 ^ 0.7

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l764_764020


namespace sequence_a_converges_to_golden_ratio_sequence_b_converges_to_neg_recip_golden_ratio_l764_764495

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 1

noncomputable def f' (x : ℝ) : ℝ := 2 * x - 1

noncomputable def newton_iter (x : ℝ) : ℝ := (x^2 + 1) / (2 * x - 1)

def sequence_a (n : ℕ) : ℝ := nat.rec_on n 1 (λ n x_n, newton_iter x_n)

def sequence_b (n : ℕ) : ℝ := nat.rec_on n 0 (λ n x_n, newton_iter x_n)

def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

def neg_recip_golden_ratio : ℝ := -(2 / (1 + Real.sqrt 5))

theorem sequence_a_converges_to_golden_ratio :
  filter.tendsto sequence_a filter.at_top (nhds golden_ratio) :=
sorry

theorem sequence_b_converges_to_neg_recip_golden_ratio :
  filter.tendsto sequence_b filter.at_top (nhds neg_recip_golden_ratio) :=
sorry

end sequence_a_converges_to_golden_ratio_sequence_b_converges_to_neg_recip_golden_ratio_l764_764495


namespace diagonals_in_polygon_with_one_vertex_not_connecting_l764_764329

theorem diagonals_in_polygon_with_one_vertex_not_connecting (n : ℕ) (hn : n = 19) :
  let total_diagonals := n * (n - 3) / 2
  total_diagonals - (n - 3) = 136 :=
by
  have h_total_diagonals : total_diagonals = 152 :=
    by
      rw [hn]
      calc
        19 * (19 - 3) / 2 = 19 * 16 / 2  := by norm_num
        ... = 152 := by norm_num
  rw [hn] at *
  rw [h_total_diagonals]
  calc
    152 - (19 - 3) = 152 - 16 := by norm_num
    ... = 136 := by norm_num

end diagonals_in_polygon_with_one_vertex_not_connecting_l764_764329


namespace black_lambs_count_l764_764616

def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193
def brown_lambs : ℕ := 527

theorem black_lambs_count :
  total_lambs - white_lambs - brown_lambs = 5328 :=
by
  -- Proof omitted
  sorry

end black_lambs_count_l764_764616


namespace find_dot_product_l764_764580

open Function

-- Define the square with its vertices A, B, C, D
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : (∥B - A∥ = side_length ∧ ∥C - B∥ = side_length ∧ ∥D - C∥ = side_length ∧ ∥A - D∥ = side_length))

-- Define midpoint of DC
def midpoint (D C : ℝ × ℝ) : ℝ × ℝ :=
  ((D.1 + C.1) / 2, (D.2 + C.2) / 2)

-- Coordinates of vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (3, 3)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := midpoint D C

-- Intersection point F of lines AE and BD
def intersection (A E B D : ℝ × ℝ) : ℝ × ℝ :=
  let slope_ae := (E.2 - A.2) / (E.1 - A.1)
  let intercept_ae := A.2 - slope_ae * A.1
  let slope_bd := (B.2 - D.2) / (B.1 - D.1)
  let intercept_bd := D.2 - slope_bd * D.1
  let x := (intercept_bd - intercept_ae) / (slope_ae - slope_bd)
  let y := slope_ae * x + intercept_ae
  (x, y)

def F : ℝ × ℝ := intersection A E B D

-- Vector operations
def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Vectors FD and DE
def FD := vector_sub D F
def DE := vector_sub E D

-- The proof statement
theorem find_dot_product :
  dot_product FD DE = -3 / 2 :=
by {
  sorry
}

end find_dot_product_l764_764580


namespace largest_interval_invertible_l764_764252

-- Define the quadratic function g(x)
def g (x : ℝ) : ℝ := 3*x^2 - 9*x - 4

-- Statement that the largest interval making g invertible including x = 3 is [3/2, ∞)
theorem largest_interval_invertible (x : ℝ) (H : x = 3) : 
  ∃ (a : ℝ), a = 1.5 ∧ ∀ y : ℝ, y ∈ set.Ici a →
  function.injective g :=
sorry

end largest_interval_invertible_l764_764252


namespace increasing_interval_g_l764_764306

def f (x : ℝ) : ℝ := cos (4 * x - π / 3) + 2 * cos (2 * x) ^ 2

def g (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 1

theorem increasing_interval_g :
  ∃ (a b : ℝ), (a = -π / 4 ∧ b = π / 4) ∧ (∀ x : ℝ, a < x ∧ x < b → g (x - a) < g (x + a)) :=
sorry

end increasing_interval_g_l764_764306


namespace greatest_and_next_to_greatest_l764_764966

def exponentiation (a b : ℝ) : ℝ := a ^ b

noncomputable def max_val := max (
  exponentiation 4 (1/4)
  (max (
    exponentiation 5 (1/5)
    (max (
      exponentiation 16 (1/16)
      exponentiation 25 (1/25)
    ))
  ))
)

noncomputable def second_max_val (lst : List ℝ) : ℝ :=
  let sorted_lst := lst.qsort (· < ·)
  sorted_lst.get! (sorted_lst.length - 2)

theorem greatest_and_next_to_greatest :
  (max_val = exponentiation 4 (1/4)) ∧
  (second_max_val [exponentiation 4 (1/4), exponentiation 5 (1/5), exponentiation 16 (1/16), exponentiation 25 (1/25)] = exponentiation 5 (1/5)) :=
by
  sorry

end greatest_and_next_to_greatest_l764_764966


namespace min_third_side_length_l764_764349

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end min_third_side_length_l764_764349


namespace lewis_speed_is_90_l764_764970

noncomputable def david_speed : ℝ := 50 -- mph
noncomputable def distance_chennai_hyderabad : ℝ := 350 -- miles
noncomputable def distance_meeting_point : ℝ := 250 -- miles

theorem lewis_speed_is_90 :
  ∃ L : ℝ, 
    (∀ t : ℝ, david_speed * t = distance_meeting_point) →
    (∀ t : ℝ, L * t = (distance_chennai_hyderabad + (distance_meeting_point - distance_chennai_hyderabad))) →
    L = 90 :=
by
  sorry

end lewis_speed_is_90_l764_764970


namespace stable_polynomials_K_l764_764781

def is_stable (f : ℕ → ℕ) : Prop :=
  ∀ (x : ℕ), ¬(∃ d, d.digitt sb.name1 = 7)

def polynomials_with_non_negative_integer_coeff (f : ℕ → ℕ) : Prop :=
  ∀ x, f(x) = ∑ a_i * x ^ i where a_i ∈ ℕ

def satisfies_condition (f : ℕ → ℕ) (K : set ℕ) : Prop :=
  ∀ x ∈ K, f x ∈ K

def specific_form (f : ℕ → ℕ) (K : set ℕ) : Prop :=
  (∃ k ∈ K, f = λ x, k) ∨
  (∃ m ∈ ℕ, f = λ x, 10 ^ m * x) ∨
  (∃ m ∈ ℕ, ∃ k ∈ K, k < 10 ^ m ∧ f = λ x, 10 ^ m * x + k)

theorem stable_polynomials_K (K : set ℕ) :
  (∀ k : ℕ, k ∈ K ↔ ¬(∃ d, (0 : ℕ) .digitt.eqsb k = 7 )) →
  (∀ f : ℕ → ℕ, satisfies_condition f K →
    specific_form f K) :=
begin
  sorry,
end

end stable_polynomials_K_l764_764781


namespace perpendicular_tangent_line_exists_and_correct_l764_764631

theorem perpendicular_tangent_line_exists_and_correct :
  ∃ L : ℝ → ℝ → Prop,
    (∀ x y, L x y ↔ 3 * x + y + 6 = 0) ∧
    (∀ x y, 2 * x - 6 * y + 1 = 0 → 3 * x + y + 6 ≠ 0) ∧
    (∃ a b : ℝ, 
       b = a^3 + 3*a^2 - 5 ∧ 
       (a, b) ∈ { p : ℝ × ℝ | ∃ f' : ℝ → ℝ, f' a = 3 * a^2 + 6 * a ∧ f' a * 3 + 1 = 0 } ∧
       L a b)
:= 
sorry

end perpendicular_tangent_line_exists_and_correct_l764_764631


namespace negation_of_universal_statement_l764_764460

variable (f : ℝ → ℝ)

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, f(x) > 0) ↔ ∃ x : ℝ, f(x) ≤ 0 :=
by
  sorry

end negation_of_universal_statement_l764_764460


namespace improper_integral_vp_value_l764_764586

noncomputable def improper_integral := ∫ x in (1/(exp(1)))..(exp(1)), (1 / (x * (Real.log x)))

theorem improper_integral_vp_value : improper_integral = 0 := by
  sorry

end improper_integral_vp_value_l764_764586


namespace graph_quadrant_exclusion_l764_764258

theorem graph_quadrant_exclusion (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ∀ x : ℝ, ¬ ((a^x + b > 0) ∧ (x > 0)) :=
by
  sorry

end graph_quadrant_exclusion_l764_764258


namespace value_of_f_l764_764267

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom f_has_derivative : ∀ x, deriv f x = f' x
axiom f_equation : ∀ x, f x = 3 * x^2 + 2 * x * (f' 1)

-- Proof goal
theorem value_of_f'_at_3 : f' 3 = 6 := by
  sorry

end value_of_f_l764_764267


namespace smallest_prime_factor_in_C_is_76_l764_764424

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then Classical.choose h else n

def C : set ℕ := {67, 71, 73, 76, 79}

theorem smallest_prime_factor_in_C_is_76 : 
  ∃ x ∈ C, smallest_prime_factor x = 2 ∧ ∀ y ∈ C, smallest_prime_factor y ≥ 2 :=
begin
  use 76,
  split,
  { exact set.mem_insert 76 (set.insert 73 (set.insert 71 (set.insert 67 {79}))) },
  split,
  {
    change smallest_prime_factor 76 = 2,
    unfold smallest_prime_factor,
    have h : ∃ p, is_prime p ∧ p ∣ 76,
    { use 2,
      split,
      { unfold is_prime,
        split,
        { norm_num },
        { intros m h,
          norm_num at h,
          exact or.inl rfl } },
      norm_num },
    exact Classical.choose_spec h,
  },
  { -- Proof that for every y in C, smallest_prime_factor y ≥ 2
    intros y hy,
    change smallest_prime_factor y ≥ 2,
    unfold smallest_prime_factor,
    by_cases h : ∃ p, is_prime p ∧ p ∣ y,
    { exact (Classical.choose_spec h).1.ge },
    { simp only [if_neg h],
      exact dec_trivial },
  }
end

end smallest_prime_factor_in_C_is_76_l764_764424


namespace house_floors_eq_3_l764_764791

noncomputable def total_number_of_floors
  (earnings_per_window : ℕ)
  (subtracted_amount : ℕ)
  (days_to_finish : ℕ)
  (total_paid : ℕ) : ℕ :=
  let F := (total_paid + subtracted_amount * (days_to_finish / 3)) / earnings_per_window
  in F

theorem house_floors_eq_3 :
  total_number_of_floors 6 1 6 16 = 3 :=
by
  unfold total_number_of_floors
  simp
  sorry

end house_floors_eq_3_l764_764791


namespace solution_set_correct_l764_764222

open Real

def differentiable (f : ℝ → ℝ) : Prop := ∀ x, ∃ f' : ℝ, has_deriv_at f f' x

noncomputable def problem_statement (f : ℝ → ℝ) (hf_diff : differentiable f) (hf_ineq : ∀ x, f x > f' x)
  (hf_zero : f 0 = 1) : Set ℝ := {x : ℝ | (f x / exp x) < 1}

theorem solution_set_correct {f : ℝ → ℝ} (hf_diff : differentiable f) (hf_ineq : ∀ x, f x > f' x)
  (hf_zero : f 0 = 1) : problem_statement f hf_diff hf_ineq hf_zero = Ioi 0 :=
begin
  sorry
end

end solution_set_correct_l764_764222


namespace relationship_between_M_and_P_l764_764404

variable {U : Type*} [nonempty U]
variable {M N P : Set U}

-- Conditions
axiom hU : (U ≠ ∅)
axiom h1 : (M = Set.compl N)
axiom h2 : (N = Set.compl P)

theorem relationship_between_M_and_P : M = P := 
  sorry

end relationship_between_M_and_P_l764_764404


namespace harper_brother_rubber_bands_l764_764328

theorem harper_brother_rubber_bands :
  ∀ (H B : ℕ), H = 15 → (H + B = 24) → (H - B = 6) :=
by
  intros H B H_eq HB_eq
  rw H_eq at HB_eq
  have B_val : B = 24 - 15 := by linarith
  rw B_val
  linarith

end harper_brother_rubber_bands_l764_764328


namespace percentage_increase_second_half_l764_764794

def total_distance : ℝ := 640
def first_half_distance : ℝ := total_distance / 2
def average_speed_first_half : ℝ := 80
def time_first_half : ℝ := first_half_distance / average_speed_first_half
def average_speed_total : ℝ := 40
def total_time : ℝ := total_distance / average_speed_total
def time_second_half : ℝ := total_time - time_first_half
def percentage_increase := (time_second_half - time_first_half) / time_first_half * 100

theorem percentage_increase_second_half : 
  percentage_increase = 200 :=
by
  sorry

end percentage_increase_second_half_l764_764794


namespace set_median_is_9_l764_764518

theorem set_median_is_9 
  (S : Set ℕ) 
  (S = {8, 46, 53, 127}) 
  (S' := ({8, 46, 53, 127, 6, 7, 9} : Set ℕ)) : 
  sorted (S' : List ℕ) ∧ (S'.toList.nth 3 = some 9) :=
by 
  sorry

end set_median_is_9_l764_764518


namespace inequality_solution_set_l764_764091

theorem inequality_solution_set (x : ℝ) : 9 > -3 * x → x > -3 :=
by
  intro h
  sorry

end inequality_solution_set_l764_764091


namespace part_a_part_b_l764_764019

-- Condition: ξᵢ are i.i.d. Bernoulli random variables
variable {n m : ℕ}
variable {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1)
noncomputable def ξ : ℕ → ℝ → bool := λ i p, Bernoulli(p)

-- Sum of random variables Sn
def S (n : ℕ) : ℝ → ℕ := λ p, finset.sum finset.range n (λ i, if ξ i p then 1 else 0)

-- Part (a)
theorem part_a (xi : ℕ → ℝ → bool) (Sn : ℕ → ℝ → ℕ) (k : ℕ) :
  ∀ (x₁ x₂ ... xₙ : ℕ),
  P (ξ 1 = x₁, ξ 2 = x₂, ..., ξ n = xₙ | S n p = k) = indicator k [x₁+x₂+...+xₙ] / choose n k := sorry

-- Part (b)
theorem part_b (xi : ℕ → ℝ → bool) (S : ℕ → ℝ → ℕ) (n m k x : ℕ) :
  P (S n = x | S (n + m) p = k) = (choose n x * choose m (k - x)) / choose (n + m) k := sorry

end part_a_part_b_l764_764019


namespace remainder_9_pow_2023_div_50_l764_764503

theorem remainder_9_pow_2023_div_50 : (9 ^ 2023) % 50 = 41 := by
  sorry

end remainder_9_pow_2023_div_50_l764_764503


namespace max_k_inequality_l764_764242

open Real

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 :=
by
  sorry

end max_k_inequality_l764_764242


namespace pentagon_perimeter_sum_l764_764594

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter (pts : List (ℝ × ℝ)) : ℝ :=
  (pts.zip (pts.tail ++ [pts.head])).sumBy (λ p, distance p.1 p.2)

theorem pentagon_perimeter_sum :
  let pts := [(1,1), (3,1), (4,3), (2,4), (1,3)] in
  let d := 4
  let e := 2
  let f := 1
  let m := 5
  let n := 2
  perimeter pts = d + e * real.sqrt m + f * real.sqrt n ∧
  d + e + f = 7 :=
by
  sorry

end pentagon_perimeter_sum_l764_764594


namespace prime_indices_pair_l764_764186

open Nat

/-- Define p_n as the nth prime number, following the problem's notation -/
def nth_prime (n : ℕ) : ℕ :=
  if h : n > 0 then Nat.minFac (Nat.factorization (n + 1)).keys.head else 2

theorem prime_indices_pair :
  ∀ a b : ℕ, a - b ≥ 2 → prime (nth_prime a) → prime (nth_prime b) →
  ∃ x, x = (4, 2) ∧ (x.1 - x.2) ≥ 2 ∧ 
  (nth_prime x.1 - nth_prime x.2) ∣ 2 * (x.1 - x.2) :=
by
  sorry


end prime_indices_pair_l764_764186


namespace line_equation_pairs_l764_764641

theorem line_equation_pairs (a b : ℝ) :
  (∃! a b : ℝ, (4 / (3 * b)) = (a / 4) ∧ (a / 4) = (b / 15) ∧ (4x + a * y + b = 0) ∧ (3 * b * x + 4 * y + 15 = 0)) :=
sorry

end line_equation_pairs_l764_764641


namespace fill_time_with_both_pipes_open_l764_764513

-- Define the rates based on the problem conditions
def rate_pipe_A : ℝ := 1 / 20
def rate_pipe_B : ℝ := 4 * rate_pipe_A

-- Define the combined rate and the time to fill the tank
def combined_rate : ℝ := rate_pipe_A + rate_pipe_B
def time_to_fill (r_combined : ℝ) : ℝ := 1 / r_combined

-- Theorem stating the time to fill the tank when both pipes are open
theorem fill_time_with_both_pipes_open : time_to_fill combined_rate = 4 := by
  sorry  -- Proof to be filled in later, but Structurally valid.

end fill_time_with_both_pipes_open_l764_764513


namespace largest_angle_of_convex_hexagon_is_122_5_l764_764537

-- defining the problem statements
def convex_hexagon (x : ℝ) : Prop :=
  ∀ (i : ℕ), i ∈ {1, 2, 3, 4, 5, 6} → 0 < (x - 3) + (i : ℝ) ∧ (x - 3) + (i : ℝ) < 180

def sum_of_interior_angles (x : ℝ) : Prop :=
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720

-- proving the largest angle measure is 122.5 degrees
theorem largest_angle_of_convex_hexagon_is_122_5 (x : ℝ) (h1 : convex_hexagon x) (h2 : sum_of_interior_angles x) : 
  (x + 2 = 122.5) :=
sorry

end largest_angle_of_convex_hexagon_is_122_5_l764_764537


namespace sin_cos_ratio_l764_764785

theorem sin_cos_ratio (x y : ℝ) (h1 : sin x / sin y = 2) (h2 : cos x / cos (y + π / 4) = 3) :
  (sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y)) = 6 * Real.sqrt 2 + 11 / 29 :=
by
  -- Placeholder for the actual proof
  sorry

end sin_cos_ratio_l764_764785


namespace re_z1_lt_1_product_z1_eq_z4039_l764_764691
noncomputable def z_sequence : ℕ → ℂ
| 0       := arbitrary ℂ  -- arbitrary initial value, typically assumed given
| (n + 1) := z_sequence n ^ 2 + 1

axiom z_2021_eq_1 : z_sequence 2021 = 1

theorem re_z1_lt_1 : ∀ z_sequence, z_sequence 2021 = 1 → z_sequence (n + 1) = z_sequence n ^ 2 + 1 → re(z_sequence 1) < 1 :=
by
  sorry

theorem product_z1_eq_z4039 (P : ℂ) : ∀ z_sequence, z_sequence 2021 = 1 → z_sequence (n + 1) = z_sequence n ^ 2 + 1 → P = z_sequence 4039 :=
by
  sorry

end re_z1_lt_1_product_z1_eq_z4039_l764_764691


namespace first_team_speed_l764_764111

theorem first_team_speed:
  ∃ v: ℝ, 
  (∀ (t: ℝ), t = 2.5 → 
  (∀ s: ℝ, s = 125 → 
  (v + 30) * t = s) ∧ v = 20) := 
  sorry

end first_team_speed_l764_764111


namespace suit_price_after_discount_l764_764516

-- Definitions based on given conditions 
def original_price : ℝ := 200
def price_increase : ℝ := 0.30 * original_price
def new_price : ℝ := original_price + price_increase
def discount : ℝ := 0.30 * new_price
def final_price : ℝ := new_price - discount

-- The theorem
theorem suit_price_after_discount :
  final_price = 182 :=
by
  -- Here we would provide the proof if required
  sorry

end suit_price_after_discount_l764_764516


namespace find_alpha_l764_764367

noncomputable def parametric_eq_line (α t : Real) : Real × Real :=
  (1 + t * Real.cos α, t * Real.sin α)

def cartesian_eq_curve (x y : Real) : Prop :=
  y^2 = 4 * x

def intersection_condition (α t₁ t₂ : Real) : Prop :=
  Real.sin α ≠ 0 ∧ 
  (1 + t₁ * Real.cos α, t₁ * Real.sin α) = (1 + t₂ * Real.cos α, t₂ * Real.sin α) ∧ 
  Real.sqrt ((t₁ + t₂)^2 - 4 * (-4 / (Real.sin α)^2)) = 8

theorem find_alpha (α : Real) (t₁ t₂ : Real) 
  (h1: 0 < α) (h2: α < π) (h3: intersection_condition α t₁ t₂) : 
  α = π/4 ∨ α = 3*π/4 :=
by 
  sorry

end find_alpha_l764_764367


namespace calculate_star_difference_l764_764652

def star (a b : ℕ) : ℕ := a^2 + 2 * a * b + b^2

theorem calculate_star_difference : (star 3 5) - (star 2 4) = 28 := by
  sorry

end calculate_star_difference_l764_764652


namespace sum_of_roots_abs_equation_l764_764608

theorem sum_of_roots_abs_equation : 
  (∑ n in {n : ℝ | |3 * n - 4| = n + 2}.to_finset, n) = 7 / 2 :=
sorry

end sum_of_roots_abs_equation_l764_764608


namespace solve_equation_l764_764814

theorem solve_equation :
  (∃ x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3) → (x = -1) :=
by
  sorry

end solve_equation_l764_764814


namespace triathlon_cycling_time_and_speed_l764_764928

theorem triathlon_cycling_time_and_speed (v1 v2 v3 : ℝ) :
  -- Speeds in practice
  let t_swim_practice := 1/16 in
  let t_run_practice := 1/49 in
  let t_cycle_practice := 1/49 in
  let d_practice := 1 * t_swim_practice + 25 * t_cycle_practice + 4 * t_run_practice in
  
  -- Match the practice and competition speed
  (1 / v1 = t_swim_practice) →
  (25 / v2 = t_cycle_practice) →
  (4 / v3 = t_run_practice) →
  
  -- Condition given in competition
  (1 / v1 + 25 / v2 + 4 / v3 = 5/4) →

  -- Prove the time spent cycling and the speed
  v2 = 35 ∧ (25 / 35 = 5/7) :=
by
  intros
  sorry

end triathlon_cycling_time_and_speed_l764_764928


namespace probability_is_4_over_15_l764_764911

-- Define the set of consecutive natural numbers from 1 to 60
def consecutive_set : Finset ℕ := Finset.range 61

-- Define the factorial of 5
def five_factorial : ℕ := Nat.factorial 5

-- Define that 5! is 120
def five_factorial_is_120 : five_factorial = 120 := by
  rfl

-- Predicate to check if a number is a factor of 120
def is_factor_of_120 (n : ℕ) : Prop := n ∣ five_factorial

-- Compute the size of the set of factors of 120 within the set {1, 2, 3, ..., 60}
def factors_in_set : Finset ℕ := (consecutive_set.filter is_factor_of_120)

-- Define the probability that a randomly chosen number from the set {1, 2, 3, ..., 60} is a factor of 120
def probability_factor_120 : ℚ :=
  (factors_in_set.card : ℚ) / (consecutive_set.card : ℚ)

-- Statement of the problem
theorem probability_is_4_over_15 : probability_factor_120 = 4 / 15 := by
  sorry

end probability_is_4_over_15_l764_764911


namespace toothpick_removal_l764_764812

noncomputable def removalStrategy : ℕ :=
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4

  -- minimum toothpicks to remove to achieve the goal
  15

theorem toothpick_removal :
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4
  removalStrategy = 15 := by
  sorry

end toothpick_removal_l764_764812


namespace extra_bananas_each_child_gets_l764_764411

-- Define the total number of students and the number of absent students
def total_students : ℕ := 260
def absent_students : ℕ := 130

-- Define the total number of bananas
variable (B : ℕ)

-- The proof statement
theorem extra_bananas_each_child_gets :
  ∀ B : ℕ, (B / (total_students - absent_students)) = (B / total_students) + (B / total_students) :=
by
  intro B
  sorry

end extra_bananas_each_child_gets_l764_764411


namespace coeff_x6_in_expansion_l764_764239

-- Define the binomial coefficient as a function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression (x - 2)^10 using binomial expansion
noncomputable def binomial_expansion (x : ℝ) (n : ℕ) (a b : ℝ) : ℝ → ℕ → ℝ :=
  λ x k, binom n k * x^(n - k) * b^k

-- State the theorem to find the coefficient of x^6 in the expansion of (x-2)^10.
theorem coeff_x6_in_expansion :
  ∀ (x : ℝ), (∃ k : ℕ, 10 - k = 6 ∧ binomial_expansion x 10 x (-2) x^6 = 16 * binom 10 4) :=
by
  sorry

end coeff_x6_in_expansion_l764_764239


namespace solve_for_a_l764_764597

noncomputable def y (a b x : ℝ) : ℝ := a * Real.sec (b * x)

theorem solve_for_a (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_max : ∀ x : ℝ, 0 ≤ x → y a b x ≤ 4) :
  a = 4 :=
sorry

end solve_for_a_l764_764597


namespace larger_root_change_exceeds_1000_l764_764825

theorem larger_root_change_exceeds_1000 (p q: ℝ) (a: ℝ) (h_p: p = 10^9) (h_q: q = 0) (h_a: a = 5 * 10^(-4)) :
  let original_equation := (x: ℝ) => x^2 + p * x + q = 0
  let perturbed_equation := (x: ℝ) => x^2 + (p + a) * x + q = 0
  let original_root := p
  let larger_perturbed_root := (p + a) + sqrt(2 * p * a)
  larger_perturbed_root > original_root + 1000 :=
sorry

end larger_root_change_exceeds_1000_l764_764825


namespace general_term_sum_reciprocal_l764_764684

open_locale big_operators

variables {a b : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition given: S_n = 2 * a_n - 1 for all n ∈ ℕ*
axiom sum_condition (n : ℕ) (hn : 0 < n) : S n = 2 * a n - 1

-- To prove: a_n = 2^(n-1)
theorem general_term (n : ℕ) (hn : 0 < n) : a n = 2^(n-1) :=
sorry

-- To prove: T_n = n / (n + 1)
theorem sum_reciprocal (n : ℕ) (hn : 0 < n) : 
  let b (k : ℕ) := real.log 2 (a (k+1)),
      T (m : ℕ) := ∑ k in range m, 1 / (b k * b (k + 1)) 
  in T n = n / (n + 1) :=
sorry

end general_term_sum_reciprocal_l764_764684


namespace increasing_on_interval_solution_set_l764_764308

noncomputable def f (x : ℝ) : ℝ := x / (x ^ 2 + 1)

/- Problem 1 -/
theorem increasing_on_interval : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2 :=
by
  sorry

/- Problem 2 -/
theorem solution_set : ∀ x : ℝ, f (2 * x - 1) + f x < 0 ↔ 0 < x ∧ x < 1 / 3 :=
by
  sorry

end increasing_on_interval_solution_set_l764_764308


namespace sum_c_seq_formula_l764_764682

noncomputable def a_seq (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_seq (n : ℕ) : ℕ := 2 ^ n
noncomputable def c_seq (n : ℕ) : ℕ := a_seq n * b_seq n

noncomputable def sum_c_seq (n : ℕ) : ℕ :=
  (finset.range n).sum (λ i, c_seq (i + 1))

theorem sum_c_seq_formula (n : ℕ) : sum_c_seq n = 6 + (2 * n - 3) * 2^(n + 1) :=
  sorry

end sum_c_seq_formula_l764_764682


namespace _l764_764680

noncomputable theorem normal_distribution_properties 
  (xi : ℝ → ℝ) (μ : ℝ) (σ : ℝ) (h1 : xi ∼ normal_dist μ 7) 
  (h2 : P(xi < 2) = P(xi > 4)) : μ = 3 ∧ σ = 7 :=
by
  sorry

end _l764_764680


namespace meal_combination_count_l764_764203

-- Define basic properties and conditions
constant Student : Type
constant options : Type
constant numStudents : ℕ := 5
constant numOptions : ℕ := 4
constant students : Fin numStudents → Student
constant foodOptions : Fin numOptions → options
constant restrictedOption : options -- This represents the steamed bun
constant allowedOptionsForA : options → Prop -- This restricts Student A's choices

-- Specific conditions
axiom H1 : ∀ s : Student, ∃ o : options, o ≠ restrictedOption ∧ (s = students 0 → allowedOptionsForA o)
axiom H2 : ¬ allowedOptionsForA (foodOptions 0) -- Student A (students 0) cannot eat rice
axiom H3 : foodOptions 1 = restrictedOption -- The steamed bun has limited availability
axiom H4 : (∀ i, 0 < i → ∃ s : Student, ∃ o : options, students i ≠ s ∧ restrictedOption ≠ o) -- Non first student picks have choices
axiom H5 : ∀ o : options, ∃ s : Student, ∃ i : Fin numStudents, students i = s ∧ foodOptions i = o -- Each option is picked by at least one student

noncomputable def mealCombinationPlans : ℕ :=
sorry -- This is where the computation of the number of different meal combination plans would be implemented, but is omitted for this example

theorem meal_combination_count : mealCombinationPlans = 132 :=
sorry -- The proof would show that the number of meal combination plans equals 132, based on provided conditions

end meal_combination_count_l764_764203


namespace minimum_groups_of_players_l764_764901

theorem minimum_groups_of_players (total_players : ℕ) (max_size : ℕ) (h_size_cond : max_size ≥ 12)
  (h_player_cond : total_players = 30) : 
  ∃ (n : ℕ) (sizes : fin n → ℕ), 
  (
    (∀ i, sizes i ≤ max_size) ∧ 
    (∑ i, sizes i = total_players) ∧ 
    (∃ i j, sizes i = 2 * sizes j) ∧ 
    (n = 3)
  ) :=
by
  sorry

end minimum_groups_of_players_l764_764901


namespace hyperbola_eccentricity_proof_l764_764681

noncomputable def hyperbola_eccentricity (a : ℝ) (c : ℝ) (b_squared : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity_proof (a c : ℝ) (h1 : 8 = b_squared) (h2 : c = 3) 
    (h3 : c^2 = a^2 + b_squared) : hyperbola_eccentricity a c b_squared = 3 :=
by 
  have b_squared_equals_8 : b_squared = 8 := by
    exact eq.symm h1
  
  have c_squared_value : c^2 = 9 := by
    rw [h2]
    exact calc
      3^2 = 9 : rfl

  have a_squared_plus_8_equals_9 : a^2 + 8 = 9 := by
    rw [←c_squared_value]
    exact h3

  have a_squared_value : a^2 = 1 := by
    linarith

  have a_value : abs a = 1 := by
    exact real.sqrt_eq_iff_eq_sq.2 ⟨rfl, rfl⟩

  have e_value : hyperbola_eccentricity a c b_squared = 3 := by
    rw [hyperbola_eccentricity, h2, ←a_value]
    norm_num

  exact e_value

end hyperbola_eccentricity_proof_l764_764681


namespace area_triangle_eq_area_quadrilateral_l764_764726

/-- Given a right triangle ABC with ∠A = 90° and AB = AC,
points P on AB and Q on AC such that PQ = BP + CQ,
and the circumcircle of ΔAPQ intersects side BC at points E and F,
prove that the area of ΔAEF is equal to the area of quadrilateral PEFC. -/
theorem area_triangle_eq_area_quadrilateral {A B C P Q E F : Point}
  (hA_right : ∠A = 90)
  (hAB_eq_AC : AB = AC)
  (hP_on_AB : P ∈ AB)
  (hQ_on_AC : Q ∈ AC)
  (hPQ_eq_BP_plus_CQ : PQ = BP + CQ)
  (hcircumcircle : circle (ΔAPQ) ∩ BC = {E, F}) :
  area (ΔAEF) = area (quadrilateral PEFC) :=
sorry

end area_triangle_eq_area_quadrilateral_l764_764726


namespace min_marked_cells_15x15_l764_764869

-- Define the board size and the conditions
def board_size : ℕ := 15

def min_marked_cells (board_size : ℕ) : ℕ := 
  let n := (board_size - 1) / 2 in
  4 * n

theorem min_marked_cells_15x15 : min_marked_cells 15 = 28 :=
by
  have n := (15 - 1) / 2
  calc
  min_marked_cells 15
      = 4 * n : rfl
  ... = 4 * 7 : by norm_num
  ... = 28 : rfl

end min_marked_cells_15x15_l764_764869


namespace vector_AB_to_vector_BA_l764_764441

theorem vector_AB_to_vector_BA (z : ℂ) (hz : z = -3 + 2 * Complex.I) : -z = 3 - 2 * Complex.I :=
by
  rw [hz]
  sorry

end vector_AB_to_vector_BA_l764_764441


namespace pits_less_than_22222_l764_764544

def isDescendingTriple (a b c : ℕ) : Prop := a > b ∧ b > c
def isAscendingPair (d e : ℕ) : Prop := d < e

def isPit (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 = n ∧
  isDescendingTriple d1 d2 d3 ∧ isAscendingPair d4 d5

def countPitsUnder (maxNum : ℕ) : ℕ :=
  (List.range maxNum).countp (λ n => isPit n)

theorem pits_less_than_22222 : countPitsUnder 22222 = 36 := by
  sorry

end pits_less_than_22222_l764_764544


namespace map_distance_ratio_l764_764293

theorem map_distance_ratio (actual_distance_km : ℝ) (map_distance_cm : ℝ) (h_actual_distance : actual_distance_km = 5) (h_map_distance : map_distance_cm = 2) :
  map_distance_cm / (actual_distance_km * 100000) = 1 / 250000 :=
by
  -- Given the actual distance in kilometers and map distance in centimeters, prove the scale ratio
  -- skip the proof
  sorry

end map_distance_ratio_l764_764293


namespace ratio_of_boys_in_class_l764_764415

noncomputable def boy_to_total_ratio (p_boy p_girl : ℚ) : ℚ :=
p_boy / (p_boy + p_girl)

theorem ratio_of_boys_in_class (p_boy p_girl total_students : ℚ)
    (h1 : p_boy = (3/4) * p_girl)
    (h2 : p_boy + p_girl = 1)
    (h3 : total_students = 1) :
    boy_to_total_ratio p_boy p_girl = 3/7 :=
by
  sorry

end ratio_of_boys_in_class_l764_764415


namespace Ursula_change_l764_764860

theorem Ursula_change : 
  let cost_hot_dogs := 5 * 1.50
  let cost_salads := 3 * 2.50
  let total_cost := cost_hot_dogs + cost_salads
  let amount_ursula_had := 2 * 10
  let change := amount_ursula_had - total_cost
  change = 5 := 
by
  let cost_hot_dogs := 5 * 1.50
  let cost_salads := 3 * 2.50
  let total_cost := cost_hot_dogs + cost_salads
  let amount_ursula_had := 2 * 10
  let change := amount_ursula_had - total_cost
  have h1 : cost_hot_dogs = 7.50 := sorry
  have h2 : cost_salads = 7.50 := sorry
  have h3 : total_cost = 15.00 := sorry
  have h4 : amount_ursula_had = 20.00 := sorry
  have h5 : change = 5.00 := sorry
  exact h5

end Ursula_change_l764_764860


namespace sum_of_six_is_22_l764_764470

-- Definitions of the numbers as constants
def num1 := 1
def num2 := 2
def num3 := 3
def num4 := 4
def num5 := 5
def num6 := 7

-- The list of the six numbers
def numbers := [num1, num2, num3, num4, num5, num6]

-- Prove the sum of the numbers is 22
theorem sum_of_six_is_22 : list.sum numbers = 22 := 
by {
  reduce,
  -- sorry to skip the remaining proof
  sorry
}

end sum_of_six_is_22_l764_764470


namespace average_remaining_checks_l764_764561

-- Definitions for the conditions
variables (x y z : ℕ)
variables (total_travelers_checks total_worth amount_spent remaining_checks : ℕ)
noncomputable theory

-- Conditions
axiom total_checks : x + y = 30
axiom total_worth_check : 50 * x + z * y = 1800
axiom checks_spent : 24 * 50 = 1200

-- Theorem statement
theorem average_remaining_checks : 
  let remaining_worth := 50 * (x - 24) + z * y,
      remaining_total_checks := (x - 24) + y,
      remaining_worth = 600,
      remaining_total_checks = 6
  in remaining_checks = 600 / 6 := 
sorry

end average_remaining_checks_l764_764561


namespace polynomial_at_one_l764_764664

def f (x : ℝ) : ℝ := x^4 - 7*x^3 - 9*x^2 + 11*x + 7

theorem polynomial_at_one :
  f 1 = 3 := 
by
  sorry

end polynomial_at_one_l764_764664


namespace total_cost_is_correct_l764_764386

-- Define the conditions
def piano_cost : ℝ := 500
def lesson_cost_per_lesson : ℝ := 40
def number_of_lessons : ℝ := 20
def discount_rate : ℝ := 0.25

-- Define the total cost of lessons before discount
def total_lesson_cost_before_discount : ℝ := lesson_cost_per_lesson * number_of_lessons

-- Define the discount amount
def discount_amount : ℝ := discount_rate * total_lesson_cost_before_discount

-- Define the total cost of lessons after discount
def total_lesson_cost_after_discount : ℝ := total_lesson_cost_before_discount - discount_amount

-- Define the total cost of everything
def total_cost : ℝ := piano_cost + total_lesson_cost_after_discount

-- The statement to be proven
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end total_cost_is_correct_l764_764386


namespace tan_double_angle_l764_764291

noncomputable def α : ℝ := sorry

theorem tan_double_angle 
  (h1 : α ∈ Ioo (3 * Real.pi / 2) (2 * Real.pi))
  (h2 : Real.sin (Real.pi / 2 + α) = 1 / 3) : 
  Real.tan (Real.pi + 2 * α) = 4 * Real.sqrt 2 / 7 := 
sorry

end tan_double_angle_l764_764291


namespace john_read_books_in_15_hours_l764_764007

theorem john_read_books_in_15_hours (hreads_faster_ratio : ℝ) (brother_time : ℝ) (john_read_time : ℝ) : john_read_time = brother_time / hreads_faster_ratio → 3 * john_read_time = 15 :=
by
  intros H
  sorry

end john_read_books_in_15_hours_l764_764007


namespace system_non_zero_solution_condition_l764_764501

theorem system_non_zero_solution_condition (a b c : ℝ) :
  (∃ (x y z : ℝ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  (2 * a * b * c + a * b + b * c + c * a - 1 = 0) :=
sorry

end system_non_zero_solution_condition_l764_764501


namespace general_formula_for_a_n_l764_764685

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 3 else 4 * 3^(n-2)

theorem general_formula_for_a_n {S : ℕ → ℝ} {a : ℕ → ℝ}
  (h1 : a 1 = 3)
  (h2 : ∀ n, S n = (1/2) * a (n + 1) + 1)
  (h3 : ∀ n, S n = ∑ i in finset.range (n + 1), a i) :
  ∀ n, a n = if n = 1 then 3 else 4 * 3^(n-2) :=
by sorry

end general_formula_for_a_n_l764_764685


namespace find_parallel_side_length_l764_764637

-- Define the conditions
def length_b := 28 -- the length of the other parallel side
def height := 21 -- the distance between the parallel sides
def area := 504 -- the area of the trapezium
def target_length := 20 -- the target length of the unknown side

-- State the theorem to prove
theorem find_parallel_side_length (a : ℕ) :
  (1 / 2 : ℝ) * (a + length_b : ℕ) * height = area →
  a = target_length :=
by
  assume h : (1 / 2 : ℝ) * (a + length_b : ℕ) * height = area
  sorry

end find_parallel_side_length_l764_764637


namespace third_row_valid_l764_764617

/- Define the properties of the 5x5 grid -/
def is_valid_sudoku (grid : Array (Array Nat)) : Prop :=
  -- Ensure the grid is 5x5
  grid.size = 5 ∧ grid.all (λ row => row.size = 5) ∧
  -- Each row contains numbers 1 to 5 without repetition
  grid.all (λ row => row.perm (Array.range 1 6)) ∧
  -- Each column contains numbers 1 to 5 without repetition
  (Array.range 0 5).all (λ i => 
    (Array.range 0 5).map (λ j => grid[j]![i]).perm (Array.range 1 6))

/- Define the third row -/
def third_row (grid : Array (Array Nat)) : Array Nat :=
  grid[2]

/- Theorem statement -/
theorem third_row_valid (grid : Array (Array Nat)) (h : is_valid_sudoku grid) :
  third_row grid = #[1, 2, 5, 4, 3] :=
sorry

end third_row_valid_l764_764617


namespace maximum_value_l764_764679

theorem maximum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  ∃ (k : ℝ), (k = 1 / 18) ∧ (∀ (u v : ℝ), (0 < u) → (0 < v) → (u + 2 * v = 1) → 
  (u * v / (u + 8 * v) ≤ k)) :=
begin
  sorry
end

end maximum_value_l764_764679


namespace digital_watch_max_digit_sum_l764_764155

theorem digital_watch_max_digit_sum : 
  let hours_digit_sum := max (∑ d in (12).digits, d),
      minutes_digit_sum := max (∑ d in (59).digits, d),
      seconds_digit_sum := max (∑ d in (59).digits, d)
  in hours_digit_sum + minutes_digit_sum + seconds_digit_sum = 37 :=
by {
  -- conditions
  let H_range := (12 : ℕ × ℕ) = (9, 3),
  let M_range := fit (59 : ℕ),
  let S_range := fit (59 : ℕ),
  -- proof logic
  sorry
}

end digital_watch_max_digit_sum_l764_764155


namespace find_truck_tank_radius_l764_764546

-- Definition of the stationary tank with given dimensions
def stationary_tank_radius : ℝ := 100
def stationary_tank_height : ℝ := 25

-- Definition of the oil truck's tank height
def truck_tank_height : ℝ := 12

-- Oil level drop in the stationary tank
def oil_level_drop : ℝ := 0.03

-- Volume of the cylinder formula
def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r ^ 2 * h

-- Volume of oil pumped out from the stationary tank
def volume_pumped : ℝ := cylinder_volume stationary_tank_radius oil_level_drop

-- Volume of the oil truck's tank (should be equal to the volume pumped)
def truck_tank_volume (r : ℝ) : ℝ := cylinder_volume r truck_tank_height

theorem find_truck_tank_radius : ∃ r : ℝ, truck_tank_volume r = volume_pumped ∧ r = 5 :=
by
  sorry

end find_truck_tank_radius_l764_764546


namespace one_fourth_of_8_point8_simplified_l764_764995

noncomputable def one_fourth_of (x : ℚ) : ℚ := x / 4

def convert_to_fraction (x : ℚ) : ℚ := 
  let num := 22
  let denom := 10
  num / denom

def simplify_fraction (num denom : ℚ) (gcd : ℚ) : ℚ := 
  (num / gcd) / (denom / gcd)

theorem one_fourth_of_8_point8_simplified : one_fourth_of 8.8 = (11 / 5) := 
by
  have h : one_fourth_of 8.8 = 2.2 := by sorry
  have h_frac : 2.2 = (22 / 10) := by sorry
  have h_simplified : (22 / 10) = (11 / 5) := by sorry
  rw [h, h_frac, h_simplified]
  exact rfl

end one_fourth_of_8_point8_simplified_l764_764995


namespace surface_area_of_cube_with_same_volume_l764_764171

noncomputable def volume_of_prism (length width height : ℝ) : ℝ :=
  length * width * height

noncomputable def edge_length_of_cube (volume : ℝ) : ℝ :=
  real.cbrt volume

noncomputable def surface_area_of_cube (s : ℝ) : ℝ :=
  6 * s^2

theorem surface_area_of_cube_with_same_volume :
  let vol_prism := volume_of_prism 12 3 24 in
  let edge_length := edge_length_of_cube vol_prism in
  surface_area_of_cube edge_length = 545.02 :=
begin
  sorry
end

end surface_area_of_cube_with_same_volume_l764_764171


namespace choir_average_age_l764_764072

theorem choir_average_age :
  let num_females := 10
  let avg_age_females := 32
  let num_males := 18
  let avg_age_males := 35
  let num_people := num_females + num_males
  let sum_ages_females := avg_age_females * num_females
  let sum_ages_males := avg_age_males * num_males
  let total_sum_ages := sum_ages_females + sum_ages_males
  let avg_age := (total_sum_ages : ℚ) / num_people
  avg_age = 33.92857 := by
  sorry

end choir_average_age_l764_764072


namespace initial_bowls_count_l764_764856

variable (B S L Br : ℕ)
variable (Payment : ℤ)

axiom (initial_bowls : L = 12)
axiom (broken_bowls : Br = 15)
axiom (total_payment : Payment = 1825)
axiom (payment_formula : Payment = 100 + 3 * S - 4 * (L + Br))

theorem initial_bowls_count : B = S + L + Br ∧ S = 611 → B = 638 :=
by
  intros h
  sorry

end initial_bowls_count_l764_764856


namespace area_of_circle_through_two_points_l764_764117

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def area_of_circle (radius : ℝ) : ℝ :=
  real.pi * radius^2

theorem area_of_circle_through_two_points 
(P Q : ℝ × ℝ) 
(hP : P = (-3, 4)) 
(hQ : Q = (9, -3)) : 
area_of_circle (distance P Q) = 193 * real.pi :=
by
  sorry

end area_of_circle_through_two_points_l764_764117


namespace sum_of_T_elements_l764_764758

-- Define T to represent the set of numbers 0.abcd with a, b, c, d being distinct digits.
def is_valid_abcd (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def T : Set ℝ := { x | ∃ (a b c d : ℕ), is_valid_abcd a b c d ∧ x = ((1000 * a + 100 * b + 10 * c + d : ℝ) / 9999) }

-- The main theorem statement.
theorem sum_of_T_elements : ∑ x in T, x = 2520 := by
  sorry

end sum_of_T_elements_l764_764758


namespace simplify_product_l764_764056

noncomputable def product (n : ℕ) : ℚ :=
  ∏ k in Finset.range n, (3 * k + 6) / (3 * k)

theorem simplify_product : product 997 = 1001 := 
sorry

end simplify_product_l764_764056


namespace solve_for_x_l764_764707

theorem solve_for_x (x : ℝ) (h : 6^(x + 2) = 1) : x = -2 :=
sorry

end solve_for_x_l764_764707


namespace x_add_y_eq_neg_one_l764_764655

theorem x_add_y_eq_neg_one (x y : ℝ) (h : |x + 3| + (y - 2)^2 = 0) : x + y = -1 :=
by sorry

end x_add_y_eq_neg_one_l764_764655


namespace xyz_sum_eq_40_l764_764890

theorem xyz_sum_eq_40
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + x * z + x^2 = 91) :
  x * y + y * z + z * x = 40 :=
sorry

end xyz_sum_eq_40_l764_764890


namespace area_pentagon_l764_764177

noncomputable def area_of_square (s : ℝ) : ℝ :=
  s^2

noncomputable def side_length_of_square_given_area (a : ℝ) : ℝ :=
  real.sqrt a

noncomputable def side_length_of_pentagon_given_perimeter (s : ℝ) : ℝ :=
  4 * s / 5

noncomputable def area_of_regular_pentagon (p : ℝ) : ℝ :=
  (5 * p^2 * real.tan (real.pi / 5)) / 4

theorem area_pentagon (a : ℝ) (s_perimeter : ℝ) (s_area : ℝ) (p_area : ℝ) : a = 16 →
  s_area = area_of_square s_perimeter →
  s_perimeter = side_length_of_square_given_area a →
  s_perimeter * 4 / 5 = side_length_of_pentagon_given_perimeter s_perimeter →
  p_area = area_of_regular_pentagon (side_length_of_pentagon_given_perimeter s_perimeter) →
  p_area ≈ 17.563 :=
by
  sorry

end area_pentagon_l764_764177


namespace total_price_of_jacket_l764_764906

def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.10

theorem total_price_of_jacket :
    let discount := discount_rate * original_price in
    let sale_price := original_price - discount in
    let tax := tax_rate * sale_price in
    let total_price := sale_price + tax in
    total_price = 92.4 :=
by
    sorry

end total_price_of_jacket_l764_764906


namespace part1_part2_l764_764272

-- Problem setup variables
variables {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}
variable {q : ℝ}
variable {a1 : ℝ}

-- Problem conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a 1 * q ^ n
axiom sum_first_four_terms : S 4 = 15 / 8
axiom arith_seq : a 1 + a 2 = 6 * a 3

-- Derived quantities
def b_n (n : ℕ) : ℝ := a n * (Real.log 2 (a n) - 1)
def sum_b_first_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

-- Problems to prove
theorem part1 : ∃ q : ℝ, 0 < q ∧ q ≠ 1 ∧ (∀ n, (S n - 2) / (S (n - 1) - 2) = q) :=
sorry

theorem part2 (n : ℕ) : sum_b_first_n n = (n + 2) * (1 / 2) ^ (n - 1) - 4 :=
sorry

end part1_part2_l764_764272


namespace find_number_of_small_branches_each_branch_grows_l764_764980

theorem find_number_of_small_branches_each_branch_grows :
  ∃ x : ℕ, 1 + x + x^2 = 43 ∧ x = 6 :=
by {
  sorry
}

end find_number_of_small_branches_each_branch_grows_l764_764980


namespace largest_prime_factor_problem_l764_764509

def largest_prime_factor (n : ℕ) : ℕ :=
  -- This function calculates the largest prime factor of n
  sorry

theorem largest_prime_factor_problem :
  largest_prime_factor 57 = 19 ∧
  largest_prime_factor 133 = 19 ∧
  ∀ n, n = 63 ∨ n = 85 ∨ n = 143 → largest_prime_factor n < 19 :=
by
  sorry

end largest_prime_factor_problem_l764_764509


namespace fifth_runner_twice_as_fast_reduction_l764_764246

/-- Given the individual times of the five runners and the percentage reductions, 
if the fifth runner had run twice as fast, the total time reduction is 8%. -/
theorem fifth_runner_twice_as_fast_reduction (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h2 : T1 = 0.10 * T)
  (h3 : T2 = 0.20 * T)
  (h4 : T3 = 0.24 * T)
  (h5 : T4 = 0.30 * T)
  (h6 : T5 = 0.16 * T) :
  let T' := T1 + T2 + T3 + T4 + T5 / 2 in
  T - T' = 0.08 * T :=
by
  sorry

end fifth_runner_twice_as_fast_reduction_l764_764246


namespace ceil_floor_expression_l764_764985

theorem ceil_floor_expression :
  let expr1 := 15 / 8 * (-35 / 4)
  let term1 := Int.ceil expr1
  let floor_inner := Int.floor (-35 / 4)
  let expr2 := 15 / 8 * floor_inner
  let term2 := Int.floor expr2
  term1 - term2 = 1 := 
begin
  sorry
end

end ceil_floor_expression_l764_764985


namespace unique_solution_l764_764988

theorem unique_solution (a b : ℤ) (h : a > b ∧ b > 0) (hab : a * b - a - b = 1) : a = 3 ∧ b = 2 := by
  sorry

end unique_solution_l764_764988


namespace area_of_region_l764_764207

open Real

theorem area_of_region : 
  let equation := λ x y : ℝ, x^2 + y^2 + 6 * x - 8 * y + 9 = 0
  ∃ (r : ℝ), ∀ (x y : ℝ), (equation x y) → (π * r^2 = 16 * π) := 
sorry

end area_of_region_l764_764207


namespace probability_X_eq_4_l764_764531

-- Define the setup for the problem
def balls : Finset ℕ := {1, 2, 3, 4, 5}
def draw_size : ℕ := 3

-- Define a function to calculate combinations
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define X (highest number drawn)
def X (s : Finset ℕ) : ℕ := s.sup id

-- Define the probability calculation
def P_X_eq_4 : ℚ :=
  let total_outcomes : ℚ := C 5 3
  let favorable_outcomes : ℚ := C 3 2
  favorable_outcomes / total_outcomes

-- The proof statement
theorem probability_X_eq_4 : P_X_eq_4 = 0.3 := by
  sorry

end probability_X_eq_4_l764_764531


namespace cost_of_fencing_l764_764514

-- Definitions of ratio and area conditions
def sides_ratio (length width : ℕ) : Prop := length / width = 3 / 2
def area (length width : ℕ) : Prop := length * width = 3750

-- Define the cost per meter in paise
def cost_per_meter : ℕ := 70

-- Convert paise to rupees
def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

-- The main statement we want to prove
theorem cost_of_fencing (length width perimeter : ℕ)
  (H1 : sides_ratio length width)
  (H2 : area length width)
  (H3 : perimeter = 2 * length + 2 * width) :
  paise_to_rupees (perimeter * cost_per_meter) = 175 := by
  sorry

end cost_of_fencing_l764_764514


namespace smallest_N_conditions_l764_764164

theorem smallest_N_conditions:
  ∃N : ℕ, N % 9 = 8 ∧
           N % 8 = 7 ∧
           N % 7 = 6 ∧
           N % 6 = 5 ∧
           N % 5 = 4 ∧
           N % 4 = 3 ∧
           N % 3 = 2 ∧
           N % 2 = 1 ∧
           N = 2519 :=
sorry

end smallest_N_conditions_l764_764164


namespace vanessa_shirt_price_l764_764492

theorem vanessa_shirt_price
  (price_dress : ℕ)
  (qty_dresses : ℕ)
  (total_amount : ℕ)
  (qty_shirts : ℕ) :
  price_dress = 7 →
  qty_dresses = 7 →
  total_amount = 69 →
  qty_shirts = 4 →
  let price_shirt := (total_amount - price_dress * qty_dresses) / qty_shirts in
  price_shirt = 5 :=
by
  intros h1 h2 h3 h4
  let price_shirt := (total_amount - price_dress * qty_dresses) / qty_shirts
  show price_shirt = 5 from sorry

end vanessa_shirt_price_l764_764492


namespace probability_bus_there_when_carla_arrives_l764_764151

open Set

def bus_arrival_time := Icc 0 60 -- interval from 0 to 60 minutes

noncomputable def carla_arrival_time := Icc 0 60 -- interval from 0 to 60 minutes

lemma bus_waits_for_15_minutes (y : ℝ) (hy : y ∈ bus_arrival_time) : 
  ∀ x ∈ carla_arrival_time, y ≤ x ∧ x ≤ y + 15 → x - y ≤ 15 :=
by simp only [mem_Icc, sub_nonneg, and_imp, true_imp_iff]; norm_num; exact sorry

theorem probability_bus_there_when_carla_arrives :
  ∃ (P : ℝ), 0 ≤ P ∧ P ≤ 1 ∧ P = 7 / 32 :=
begin
  use 7 / 32,
  split, {norm_num}, split, {norm_num},
  sorry
end

end probability_bus_there_when_carla_arrives_l764_764151


namespace problem_l764_764735

noncomputable def BFr({p q r : ℕ} (hp : r > 0) (hr_nat : ∀ n : ℕ, ¬ (n * n) ∣ r)) := 
  ∃ (BF : ℝ), BF = p + q * Real.sqrt r

theorem problem 
  (ABCD : Type)
  [affine_space Euclidean_space affine.simplex.orthonormal.canonical ℝ (fin 2)]
  (A B C D O : Euclidean_space)
  [HasAdd Euclidean_space]
  [HasSub Euclidean_space]
  [HasSmul ℝ Euclidean_space]
  (hSquare : (dist A B) = 1000 ∧ (dist B C) = 1000 ∧ (dist C D) = 1000 ∧ (dist D A) = 1000)
  (hCenter : dist O A = dist O C ∧ dist O B = dist O D)
  (E F : Euclidean_space) 
  (hAE_BF : dist A E < dist B F ∧ E ∈ segment ℝ A B ∧ F ∈ segment ℝ A B ∧ E ≠ F ∧ dist E F = 500)
  (hAngle : angle E O F = real.pi / 3) :
  ∃ (p q r : ℕ), BFr {p q r} ∧ r ≤ 3and (p + q + r = 378) :=
begin 
  sorry 
end

end problem_l764_764735


namespace y_intercept_tangent_line_l764_764103

/-- Three circles have radii 3, 2, and 1 respectively. The first circle has center at (3,0), 
the second at (7,0), and the third at (11,0). A line is tangent to all three circles 
at points in the first quadrant. Prove the y-intercept of this line is 36.
-/
theorem y_intercept_tangent_line
  (r1 r2 r3 : ℝ) (h1 : r1 = 3) (h2 : r2 = 2) (h3 : r3 = 1)
  (c1 c2 c3 : ℝ × ℝ) (hc1 : c1 = (3, 0)) (hc2 : c2 = (7, 0)) (hc3 : c3 = (11, 0)) :
  ∃ y_intercept : ℝ, y_intercept = 36 :=
sorry

end y_intercept_tangent_line_l764_764103


namespace alice_bike_speed_correct_l764_764566

def swim_distance : ℝ := 0.5
def swim_speed : ℝ := 3
def run_distance : ℝ := 5
def run_speed : ℝ := 10
def bike_distance : ℝ := 20
def total_time : ℝ := 2

-- Define the time taken for swimming and running
def swim_time : ℝ := swim_distance / swim_speed
def run_time : ℝ := run_distance / run_speed
def other_activities_time : ℝ := swim_time + run_time

-- Define the time available for the bike ride
def bike_time : ℝ := total_time - other_activities_time

-- Define the required average speed for the bike ride
def required_bike_speed : ℝ := bike_distance / bike_time

-- Assertion to prove
theorem alice_bike_speed_correct : required_bike_speed = 15 := by
  sorry

end alice_bike_speed_correct_l764_764566


namespace quadratic_solution_intervals_l764_764520

theorem quadratic_solution_intervals (a : ℝ) :
  (6 - 3 * a > 0) ∧ (a > 0) ∧ (3 * a^2 + a - 2 ≥ 0) →
  a ∈ Ioo 0 (2/3) ∪ Ioo (2/3) (5/3) ∪ Ioo (5/3) 2 :=
by sorry

end quadratic_solution_intervals_l764_764520


namespace infinite_chain_on_parabola_l764_764830

-- Given definitions
def circle_eq (n : ℕ) := λ x y : ℝ, x^2 + y^2 = (n : ℝ)^2
def line_eq (m : ℤ) := λ (y : ℝ), y = (m : ℝ)
def intersection (x y : ℝ) (n m : ℤ) := 
  circle_eq n x y ∧ line_eq m y 

-- Prove that all points of the chain lie on the same parabola
theorem infinite_chain_on_parabola
  (x y : ℝ) (m n : ℤ) (h : intersection x y n m)
  (h_diff : n - m = 6) :
  y = x^2 / 12 - 3 := 
sorry

end infinite_chain_on_parabola_l764_764830


namespace round_periodic_decimal_to_nearest_thousandth_l764_764423

noncomputable def periodic_to_rational_approximation (x : ℝ) (n : ℕ) : ℝ :=
  (floor ((10^n) * x) / (10^n))

def periodic_decimal : ℝ := 67 + (36 / 99)

theorem round_periodic_decimal_to_nearest_thousandth :
  periodic_to_rational_approximation periodic_decimal 3 = 67.364 :=
by
  sorry

end round_periodic_decimal_to_nearest_thousandth_l764_764423


namespace minimum_value_expr_eq_neg6680_25_l764_764638

noncomputable def expr (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200

theorem minimum_value_expr_eq_neg6680_25 : ∃ x : ℝ, (∀ y : ℝ, expr y ≥ expr x) ∧ expr x = -6680.25 :=
sorry

end minimum_value_expr_eq_neg6680_25_l764_764638


namespace total_apples_eaten_l764_764194

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end total_apples_eaten_l764_764194


namespace smaller_angle_at_3_15_l764_764114

namespace ClockAngle

def degrees_per_hour := 30
def degrees_per_minute := 6
def hour_hand_position_at_3 := 90
def additional_hour_hand_position := degrees_per_hour * (15 / 60)
def total_hour_hand_position := hour_hand_position_at_3 + additional_hour_hand_position
def minute_hand_position_at_15 := degrees_per_minute * 15
def angle_difference := abs (total_hour_hand_position - minute_hand_position_at_15)

theorem smaller_angle_at_3_15 :
  angle_difference = 7.5 :=
by
  sorry

end ClockAngle

end smaller_angle_at_3_15_l764_764114


namespace problem_statement_l764_764295

variables {R : Type*} [linear_ordered_field R]
variables (f : R → R) (a b : R)

-- Assume f is monotonically increasing
def monotonically_increasing := ∀ x y, x ≤ y → f x ≤ f y

-- Assume f is symmetric with respect to (a, b)
def symmetric_wrt_point := ∀ x, f (a - x) + f (a + x) = 2 * b

-- The proposition to prove: if f(x₁) + f(x₂) > 2b, then x₁ + x₂ > 2a
theorem problem_statement (h_mono : monotonically_increasing f)
  (h_symm : symmetric_wrt_point f a b) :
  ∀ x₁ x₂, f x₁ + f x₂ > 2 * b → x₁ + x₂ > 2 * a :=
begin
  sorry
end

end problem_statement_l764_764295


namespace sine_difference_decreases_l764_764372

noncomputable theory

open Real

def one_minute_deg : ℝ := 1 / 60

theorem sine_difference_decreases (α : ℝ) (hα : 0 ≤ α ∧ α ≤ 90) :
  (sin (α + one_minute_deg) - sin α) > 
  (sin (α + one_minute_deg + one_minute_deg) - sin (α + one_minute_deg)) := 
sorry

end sine_difference_decreases_l764_764372


namespace sum_of_polynomials_at_1_l764_764388

noncomputable def a (y : ℤ) : ℤ := sorry
noncomputable def b (y : ℤ) : ℤ := sorry

theorem sum_of_polynomials_at_1 :
  (∀ y : ℤ, y^8 - 50 * y^4 + 25 = a y * b y) →
  (∀ y : ℤ, (monic (a y)) ∧ (monic (b y))) →
  (∃ p : ℤ, (p = a 1 + b 1) ∧ p = 52) := by
  sorry

end sum_of_polynomials_at_1_l764_764388


namespace symmetric_point_xoz_plane_l764_764374

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_xoz (M : Point3D) : Point3D :=
  ⟨M.x, -M.y, M.z⟩

theorem symmetric_point_xoz_plane :
  let M := Point3D.mk 5 1 (-2)
  symmetric_xoz M = Point3D.mk 5 (-1) (-2) :=
by
  sorry

end symmetric_point_xoz_plane_l764_764374


namespace part1_part2_f_extremum_and_decreasing_l764_764028

noncomputable def f (a x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

def f_deriv (a x : ℝ) : ℝ := (-3 * x^2 + (6 - a) * x + a) / Real.exp x

theorem part1 (h : f_deriv a 0 = 0) : a = 0 :=
suffices : (6 - a) * 0 + a = 0,
  by simp at h; simp [h] at this; exact this,
by simp [f_deriv, Real.exp_zero, mul_comm, mul_zero, add_zero, mul_one, zero_mul] at h; exact h

theorem part2 (h : ∀ x, 3 ≤ x → f_deriv a x ≤ 0) : a ≥ -9 / 2 :=
let u := λ x, (-3 * x^2 + 6 * x) / (x - 1) in
huffices : ∀ x, 3 ≤ x → a ≥ u x,
  from this 3 (by norm_num),
by intro x hx; exact le_of_not_gt (λ hax, (not_le_of_gt (lt_of_not_ge hax)) (h x hx))

# The full proof problem combining part1 and part2:
theorem f_extremum_and_decreasing (a : ℝ)
  (h_ext : f_deriv a 0 = 0) 
  (h_decr : ∀ x, 3 ≤ x → f_deriv a x ≤ 0) :
  a = 0 ∧ a ≥ -9 / 2 :=
begin
  split,
  { apply part1, assumption },
  { apply part2, assumption }
end

end part1_part2_f_extremum_and_decreasing_l764_764028


namespace graph_does_not_pass_first_quadrant_l764_764260

variables {a b x : ℝ}

theorem graph_does_not_pass_first_quadrant 
  (h₁ : 0 < a ∧ a < 1) 
  (h₂ : b < -1) : 
  ¬ ∃ x : ℝ, 0 < x ∧ 0 < a^x + b :=
sorry

end graph_does_not_pass_first_quadrant_l764_764260


namespace polynomial_root_relation_l764_764398

noncomputable theory
variables {p q r : ℂ}

theorem polynomial_root_relation :
  (∃ (p q r : ℂ), p + q + r = 8 ∧ pq + pr + qr = 10 ∧ pqr = 3) →
  ((p / (qr + 2)) + (q / (pr + 2)) + (r / (pq + 2)) = 41 / 10) :=
begin
  sorry,
end

end polynomial_root_relation_l764_764398


namespace log_comparison_l764_764200

theorem log_comparison (a b c : ℝ) (ha : a = log 3 6) (hb : b = log 5 10) (hc : c = log 7 14) : a > b ∧ b > c := by
  sorry

end log_comparison_l764_764200


namespace minimum_value_of_complex_expression_l764_764658

noncomputable def complex_number_min_value (z : ℂ) (hz : |z| = 1) : ℝ :=
  |z^2 - 2*z + 3|

theorem minimum_value_of_complex_expression :
  ∀ z : ℂ, |z| = 1 → (complex_number_min_value z (by sorry)) = (2 * Real.sqrt 6) / 3 := sorry

end minimum_value_of_complex_expression_l764_764658


namespace solution_set_eq_2m_add_2_gt_zero_l764_764357

theorem solution_set_eq_2m_add_2_gt_zero {m : ℝ} (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) : m = -1 :=
sorry

end solution_set_eq_2m_add_2_gt_zero_l764_764357


namespace truncated_quadrangular_pyramid_faces_l764_764502

-- Define what it means to be a truncated quadrangular pyramid
structure TruncatedQuadrangularPyramid :=
  (base_square : bool)
  (top_square : bool)
  (lateral_faces : ℕ)

-- State the condition in Lean
def is_truncated_quadrangular_pyramid (tqp : TruncatedQuadrangularPyramid) : Prop :=
  tqp.base_square = true ∧
  tqp.top_square = true ∧
  tqp.lateral_faces = 4

-- Define the theorem to prove the number of faces
theorem truncated_quadrangular_pyramid_faces
  (tqp : TruncatedQuadrangularPyramid) :
  is_truncated_quadrangular_pyramid(tqp) → tqp.lateral_faces + 2 = 6 :=
by
  intro h
  sorry

end truncated_quadrangular_pyramid_faces_l764_764502


namespace area_correct_l764_764013

-- Defining the piecewise function g
def g (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 3 then 2 * x else
if 3 < x ∧ x ≤ 6 then 3 * x - 3 else
0

-- Defining the bounds and the calculated area A
def area_under_curve : ℝ :=
let A1 := (1/2) * 3 * 6,
    A2 := (1/2) * (6 + 15) * 3 in
A1 + A2

-- Statement to prove the calculated area equals 40.5
theorem area_correct : area_under_curve = 40.5 :=
by
  sorry

end area_correct_l764_764013


namespace jellybeans_left_in_jar_l764_764475

def original_jellybeans : ℕ := 250
def class_size : ℕ := 24
def sick_children : ℕ := 2
def sick_jellybeans_each : ℕ := 7
def first_group_size : ℕ := 12
def first_group_jellybeans_each : ℕ := 5
def second_group_size : ℕ := 10
def second_group_jellybeans_each : ℕ := 4

theorem jellybeans_left_in_jar : 
  original_jellybeans - ((first_group_size * first_group_jellybeans_each) + 
  (second_group_size * second_group_jellybeans_each)) = 150 := by
  sorry

end jellybeans_left_in_jar_l764_764475


namespace solve_log_equation_l764_764062

theorem solve_log_equation :
  ∀ x : ℝ, log 10 (x^2 - 10*x + 16) = 1 → (x = 5 + sqrt 19 ∨ x = 5 - sqrt 19) :=
by
  intro x
  intro h
  sorry

end solve_log_equation_l764_764062


namespace binomial_sum_square_l764_764803

open Nat 

theorem binomial_sum_square (n : ℕ) : 
  (∑ k in Finset.range (n + 1), (nat.factorial (2 * n) / ((nat.factorial k)^2 * (nat.factorial (n - k))^2))) = (nat.choose (2 * n) n)^2 := 
sorry

end binomial_sum_square_l764_764803


namespace square_ratio_l764_764558

theorem square_ratio (x y : ℝ) (hx : x = 60 / 17) (hy : y = 780 / 169) : 
  x / y = 169 / 220 :=
by
  sorry

end square_ratio_l764_764558


namespace not_possible_to_move_minus_one_l764_764522

def vertices := Fin 12 → ℤ

def initial_state (v : vertices) : Prop :=
  v 0 = -1 ∧ (∀ i, 0 < i → v i = 1)

def change_signs (v : vertices) (k : ℕ) (start : Fin 12) : vertices :=
  λ i, if start ≤ i ∧ i < start + k then -v i else v i

theorem not_possible_to_move_minus_one (k : ℕ) (h : k = 3 ∨ k = 4 ∨ k = 6) (v : vertices) :
  initial_state v →
  ¬ ∃ start : Fin 12, initial_state (change_signs v k start) :=
by
  sorry

end not_possible_to_move_minus_one_l764_764522


namespace sum_of_integers_greater_than_2_and_less_than_15_l764_764122

-- Define the set of integers greater than 2 and less than 15
def integersInRange : List ℕ := [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define the sum of these integers
def sumIntegersInRange : ℕ := integersInRange.sum

-- The main theorem to prove the sum
theorem sum_of_integers_greater_than_2_and_less_than_15 : sumIntegersInRange = 102 := by
  -- The proof part is omitted as per instructions
  sorry

end sum_of_integers_greater_than_2_and_less_than_15_l764_764122


namespace average_bull_weight_l764_764732

def ratioA : ℚ := 7 / 28  -- Ratio of cows to total cattle in section A
def ratioB : ℚ := 5 / 20  -- Ratio of cows to total cattle in section B
def ratioC : ℚ := 3 / 12  -- Ratio of cows to total cattle in section C

def total_cattle : ℕ := 1220  -- Total cattle on the farm
def total_bull_weight : ℚ := 200000  -- Total weight of bulls in kg

theorem average_bull_weight :
  ratioA = 7 / 28 ∧
  ratioB = 5 / 20 ∧
  ratioC = 3 / 12 ∧
  total_cattle = 1220 ∧
  total_bull_weight = 200000 →
  ∃ avg_weight : ℚ, avg_weight = 218.579 :=
sorry

end average_bull_weight_l764_764732


namespace minimum_value_g_l764_764290

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - x - 2

def g (x : ℝ) : ℝ := (x + a)^2 - (x + a) - 2 + x

theorem minimum_value_g (a : ℝ) :
  (if 1 ≤ a then g a (-1) = a^2 - 3 * a - 1 else
   if -3 < a ∧ a < 1 then g a (-a) = -a - 2 else
   if a ≤ -3 then g a 3 = a^2 + 5 * a + 7 else false) :=
by
  sorry

end minimum_value_g_l764_764290


namespace drink_cans_ratio_l764_764540

theorem drink_cans_ratio 
  (Maaza_vol : ℕ) 
  (Pepsi_vol : ℕ) 
  (Sprite_vol : ℕ) 
  (num_cans : ℕ) 
  (V : ℕ) 
  (Maaza_cans : ℕ) 
  (Pepsi_cans : ℕ) 
  (Sprite_cans : ℕ) 
  (Total_vol : ℕ)
  (h1 : Maaza_vol = 10) 
  (h2 : Pepsi_vol = 144) 
  (h3 : Sprite_vol = 368) 
  (h4 : num_cans = 261) 
  (h5 : Total_vol = Maaza_vol + Pepsi_vol + Sprite_vol) 
  (h6 : V = Total_vol / num_cans) 
  (h7 : V = 2) 
  (h8 : Maaza_cans = Maaza_vol / V) 
  (h9 : Pepsi_cans = Pepsi_vol / V) 
  (h10 : Sprite_cans = Sprite_vol / V) 
  (h11 : Maaza_cans + Pepsi_cans + Sprite_cans = num_cans) 
  : Maaza_cans : Pepsi_cans : Sprite_cans = 5 : 72 : 184 :=
begin
  sorry,
end

end drink_cans_ratio_l764_764540


namespace fraction_decomposition_l764_764421

theorem fraction_decomposition 
  (a b : ℕ) (n : ℕ) 
  (h : ∃ (c : ℕ → ℕ), (∀ i j, i ≠ j → c i ≠ c j) ∧ (∑ i in finset.range n, ((c i)⁻¹ : ℚ) = a / b)) :
  ∃ (m : ℕ) (d : ℕ → ℕ), m > n ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ (∑ i in finset.range m, ((d i)⁻¹ : ℚ) = a / b) := 
sorry

end fraction_decomposition_l764_764421


namespace shift_right_transformation_l764_764451

theorem shift_right_transformation :
  ∀ x : ℝ, (x^2 - 2 * x + 1) = (x - 1)^2 :=
by
  intro x
  calc
    x^2 - 2 * x + 1 = (x^2 - 2 * x + 1) : by rfl
                 ... = (x - 1)^2         : by sorry

end shift_right_transformation_l764_764451


namespace money_distribution_l764_764185

theorem money_distribution (A B C : ℝ) (h1 : A + B + C = 1000) (h2 : B + C = 600) (h3 : C = 300) : A + C = 700 := by
  sorry

end money_distribution_l764_764185


namespace difference_in_profit_percentage_l764_764188

-- Definitions from conditions
def selling_price1 : ℝ := 350
def selling_price2 : ℝ := 340
def cost_price : ℝ := 200

-- Definition to calculate profit percentage
def profit_percentage (selling_price : ℝ) (cost_price : ℝ) : ℝ := 
  ((selling_price - cost_price) / cost_price) * 100

-- Theorem to prove the difference in profit percentages is 5%
theorem difference_in_profit_percentage : 
  profit_percentage selling_price1 cost_price - profit_percentage selling_price2 cost_price = 5 :=
sorry

end difference_in_profit_percentage_l764_764188


namespace smallest_N_exists_l764_764166

theorem smallest_N_exists : ∃ N : ℕ, 
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  N = 503 :=
by {
  sorry
}

end smallest_N_exists_l764_764166


namespace identically_dominoted_squares_l764_764725

-- Definitions for conditions
def is8x8grid (g : Array (Array Bool)) : Prop :=
  g.size = 8 ∧ (∀ row, row ∈ g -> Array.size row = 8)

def blue_cells_count (g : Array (Array Bool)) : Nat :=
  g.foldl (λ acc row => acc + row.foldl (λ acc' cell => if cell then acc' + 1 else acc') 0) 0

-- Statement of theorem
theorem identically_dominoted_squares (g1 g2 : Array (Array Bool)) (n : Nat) :
  is8x8grid g1 ∧ is8x8grid g2 ∧ blue_cells_count g1 = n ∧ blue_cells_count g2 = n ->
  ∃ dominoes1 dominoes2 : List (List (Nat × Nat)), 
    -- Each list represents a single domino using coordinates (row, col) pairs
    (∀ domino, domino ∈ dominoes1 ∧ domino ∈ dominoes2 -> length domino = 2) ∧
    -- Ensure each domino consists of two adjacent (by edge) cells
    (∀ (r1 c1 r2 c2 : Nat), (r1, c1) ∈ domino1 ∧ (r2, c2) ∈ domino1 -> 
      ((r1 = r2 ∧ c1 = c2 + 1) ∨ (r1 = r2 ∧ c1 + 1 = c2) ∨ (r1 = r2 + 1 ∧ c1 = c2) ∨ (r1 + 1 = r2 ∧ c1 = c2))) ∧
    -- Reconstruct the grid using these dominoes should yield identical blue patterns
    (∀ (r c : Nat), (g1[r][c] = true) ↔ (g2[r][c] = true)) :=
sorry

end identically_dominoted_squares_l764_764725


namespace length_of_wire_l764_764145

theorem length_of_wire (V : ℝ) (d : ℝ) (hV : V = 2.2) (hd : d = 0.50) : 
  ∃ L : ℝ, L ≈ 112.09 :=
by
  let V_cm³ := V * 1000
  have hV_cm³ : V_cm³ = 2200 := by
    rw [hV, mul_assoc, mul_comm 1000]
    norm_num
  let r := d / 2
  have hr : r = 0.25 := by
    rw hd
    norm_num
  let A := Real.pi * r^2
  have hA : A = 0.19634954084936207 := by
    rw [hr, sq]
    norm_num
    apply Real.pi_def
  let L := V_cm³ / A
  have hL : L ≈ 11208.837208791208 := by
    rw [hV_cm³, hA]
    apply Real.div_approx
  exact ⟨L / 100, by
    rw hL
    norm_num⟩

end length_of_wire_l764_764145


namespace min_value_reciprocal_sum_l764_764773

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  3 ≤ (1 / a) + (1 / b) + (1 / c) :=
by sorry

end min_value_reciprocal_sum_l764_764773


namespace polynomials_no_solutions_l764_764024

noncomputable def P : ℝ → ℝ := sorry
noncomputable def Q : ℝ → ℝ := sorry

theorem polynomials_no_solutions (h1 : ∀ x : ℝ, P(Q(x)) = Q(P(x))) 
                                  (h2 : ¬ ∃ x : ℝ, P(x) = Q(x)) :
  ¬ ∃ x : ℝ, P(P(x)) = Q(Q(x)) :=
sorry

end polynomials_no_solutions_l764_764024


namespace octahedron_faces_connected_and_flat_l764_764593

-- Definitions to be used in the proof problem
structure Cube :=
  (faces : Set (Set ℝ))
  (edges : Set (Set ℝ))

structure Octahedron :=
  (faces : Set (Set ℝ))
  (edges : Set (Set ℝ))

def midpoint (a b : ℝ) : ℝ := (a + b) / 2

-- Cube and Octahedron models with their respective properties
axiom cube_model : Cube
axiom octahedron_model : Octahedron

-- Given the correspondence conditions
axiom edges_correspondence :
  ∀ e ∈ cube_model.edges, ∃ o ∈ octahedron_model.edges, midpoint (Set.toList e !! 0) (Set.toList e !! 1) ∈ o

-- Conditions in the problem
axiom cube_edges_cut : ∃ cut_edges : Finset (Set ℝ), cut_edges.card = 7 ∧
  ∀ e ∈ cut_edges, e ∈ cube_model.edges

axiom cube_unfold_connected : ∃ uncut_edges : Finset (Set ℝ), uncut_edges.card = cube_model.edges.card - 7 ∧
  ∀ e ∈ uncut_edges, e ∉ cube_edges_cut ∧ connected_faces e cube_model

-- Define connected_faces function (this needs to be correctly wired)
-- axiom connected_faces (e : Set ℝ) (c : Cube) : Prop

-- The equivalent proof problem
theorem octahedron_faces_connected_and_flat :
  (∃ o_edges : Finset (Set ℝ), ∀ o ∈ o_edges, o ∈ (octahedron_model.edges \ edges_correspondence.toFinset)) →
  (∀ f ∈ octahedron_model.faces, connected_faces f octahedron_model ∧ can_be_laid_flat f) :=
by
  sorry

end octahedron_faces_connected_and_flat_l764_764593


namespace no_integer_solutions_for_x2_minus_4y2_eq_2011_l764_764462

theorem no_integer_solutions_for_x2_minus_4y2_eq_2011 :
  ∀ (x y : ℤ), x^2 - 4 * y^2 ≠ 2011 := by
sorry

end no_integer_solutions_for_x2_minus_4y2_eq_2011_l764_764462


namespace area_ADC_l764_764000

-- Definitions for the conditions
def triangle (A B C : Type) [add_group A] : Prop :=
∃ x : A, x > 0

variable {AB AC BC DC : ℝ}

theorem area_ADC (h1 : AB = 72)
                 (h2 : BC = 54)
                 (h3 : AC = 90)
                 (h4 : angle_ABC : ∠ABC = 90)
                 (h5 : angle_bisector AD BD DC) :
  area (triangle ADC) = 1080 :=
by
  sorry

end area_ADC_l764_764000


namespace one_eighth_of_two_power_36_equals_two_power_x_l764_764336

theorem one_eighth_of_two_power_36_equals_two_power_x (x : ℕ) :
  (1 / 8) * (2 : ℝ) ^ 36 = (2 : ℝ) ^ x → x = 33 :=
by
  intro h
  sorry

end one_eighth_of_two_power_36_equals_two_power_x_l764_764336


namespace color_of_199th_marble_l764_764560

def marble_color (pos : ℕ) : String :=
  let cycle := ["gray", "gray", "gray", "gray", "white", "white", "white", "black", "black", "blue"]
  cycle.get!? ((pos % 10) + (if pos % 10 == 0 then -1 else 0))

theorem color_of_199th_marble : marble_color 199 = "black" :=
sorry

end color_of_199th_marble_l764_764560


namespace find_nat_numbers_l764_764620

theorem find_nat_numbers (a b : ℕ) (c : ℕ) (h : ∀ n : ℕ, a^n + b^n = c^(n+1)) : a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_nat_numbers_l764_764620


namespace square_window_side_length_l764_764227

-- Define the width and height of each pane.
def width_of_pane := x
def height_of_pane := 3 * x

-- Define the total width and height of the window.
def total_width := 4 * x + 15
def total_height := 6 * x + 9

-- Define the proof problem.
theorem square_window_side_length :
  4 * x + 15 = 6 * x + 9 → 
  4 * 3 + 15 = 27 := by
  sorry

end square_window_side_length_l764_764227


namespace minimum_room_size_l764_764896

-- Definitions based on conditions identified in step a)
def bookshelf_width : ℝ := 6
def bookshelf_height : ℝ := 8

-- Definition of the diagonal using the Pythagorean theorem
def bookshelf_diagonal : ℝ := Real.sqrt (bookshelf_width^2 + bookshelf_height^2)

-- Hypothesis that the side length S must be at least as large as the diagonal
def min_side_length (S : ℝ) : Prop := S ≥ bookshelf_diagonal

-- Statement to be proved
theorem minimum_room_size : ∃ S : ℝ, min_side_length S ∧ S = 10 :=
by
  use 10
  unfold min_side_length bookshelf_diagonal bookshelf_width bookshelf_height
  sorry

end minimum_room_size_l764_764896


namespace inverse_example_l764_764844

noncomputable def f : ℕ → ℕ
| 1       := 3
| 2       := 5
| 3       := 1
| 4       := 2
| 5       := 4
| _       := 0  -- Include for totality, though not used for given inputs

theorem inverse_example :
  (f⁻¹(f⁻¹(f⁻¹(5)))) = 5 :=
by {
  have h1 : ∀ n, f n = 5 → n = 2, from 
    λ n h, by { cases n; finish [f] };
  have h2 : f 4 = 2, from rfl;
  have h3 : f 5 = 4, from rfl;
  sorry
}

end inverse_example_l764_764844


namespace abs_expression_value_l764_764399

theorem abs_expression_value :
  let x := -2023 in
  (| (| | x | - x | - | x |) | - x) = 4046 :=
by
  sorry

end abs_expression_value_l764_764399


namespace complement_union_of_sets_l764_764391

variable {U M N : Set ℕ}

theorem complement_union_of_sets (h₁ : M ⊆ N) (h₂ : N ⊆ U) :
  (U \ M) ∪ (U \ N) = U \ M :=
by
  sorry

end complement_union_of_sets_l764_764391


namespace largest_n_satisfying_inequality_l764_764868

theorem largest_n_satisfying_inequality : ∃ n : ℕ, (n < 16) ∧ (n^200 < 3^500) ∧ (∀ m : ℕ, (m > n) → ¬ (m^200 < 3^500)) :=
by
  use 15
  split
  . exact nat.lt_succ_self 15
  . split
  . norm_num
  . intros m hm hfalse
  exact not_le_of_gt hfalse (nat.le_of_lt_succ (nat.lt_of_succ_lt hm))

end largest_n_satisfying_inequality_l764_764868


namespace probability_A1_four_shots_probability_A2B2_four_shots_l764_764488

-- Define the probabilities of hitting the target
def prob_A := 2 / 3
def prob_B := 3 / 4

-- Given functions for binomial coefficients and power calculations
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k
noncomputable def power (base: ℚ) (exp: ℕ) : ℚ := base ^ exp

-- Define the events and calculate their probabilities
def event_A1 := 1 - power prob_A 4
def event_A2 := binom 4 2 * power prob_A 2 * power (1 - prob_A) 2
def event_B2 := binom 4 3 * power prob_B 3 * power (1 - prob_B) 1

-- Final probabilities
def prob_A1 := event_A1
def prob_A2B2 := event_A2 * event_B2

-- The proof statement
theorem probability_A1_four_shots : prob_A1 = 65 / 81 := 
by sorry

theorem probability_A2B2_four_shots : prob_A2B2 = 1 / 8 := 
by sorry

end probability_A1_four_shots_probability_A2B2_four_shots_l764_764488


namespace fraction_of_area_l764_764801

def area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem fraction_of_area :
  let A := (2, 1)
  let B := (7, 10)
  let C := (14, 2)
  let X := (5, 2)
  let Y := (8, 6)
  let Z := (11, 1)
  let area_ABC := area 2 1 7 10 14 2
  let area_XYZ := area 5 2 8 6 11 1
  (area_XYZ / area_ABC) = (27 / 103) :=
by {
  sorry -- Proof goes here
}

end fraction_of_area_l764_764801


namespace determine_k_range_l764_764343

noncomputable def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y : ℝ, a < x ∧ x < b ∧ a < y ∧ y < b ∧ (f x ≤ f y ∧ f y ≤ f x ∨ f x ≥ f y ∧ f y ≥ f x)

def f (x : ℝ) := 2 * x^2 - real.log x

theorem determine_k_range (k : ℝ) :
    1 ≤ k ∧ k < 3 / 2 ↔ is_not_monotonic f (k-1) (k+1) :=
sorry

end determine_k_range_l764_764343


namespace tens_digit_19_pow_1987_l764_764225

theorem tens_digit_19_pow_1987 : (19 ^ 1987) % 100 / 10 = 3 := 
sorry

end tens_digit_19_pow_1987_l764_764225


namespace unique_zero_point_in_interval_l764_764342

theorem unique_zero_point_in_interval (a : ℝ) :
  (∃! x ∈ Ioo (-1 : ℝ) (1 : ℝ), 3 * x^2 + 2 * x - a = 0) ↔ (a = -1/3 ∨ (1 < a ∧ a < 5)) :=
by
  sorry

end unique_zero_point_in_interval_l764_764342


namespace find_c_l764_764026

theorem find_c (a b c M : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : M ≠ 1) :
    (M ^ ((1 : ℝ) / a + (1 / (a * b)) + (3 / (a * b * c))) = M ^ (14 / 24)) → c = 6 :=
by
  sorry

end find_c_l764_764026


namespace divides_if_and_only_if_even_l764_764044

noncomputable def g (x : ℂ) (n : ℕ) : polynomial ℂ :=
  ∑ i in (finset.range (n + 1)).to_list, polynomial.C (x ^ (2 * i))

noncomputable def f (x : ℂ) (m : ℕ) : polynomial ℂ :=
  ∑ i in (finset.range (m + 1)).to_list, polynomial.C (x ^ (4 * i))

theorem divides_if_and_only_if_even (x : ℂ) (n m : ℕ) :
  g x n ∣ f x m ↔ even n := sorry

end divides_if_and_only_if_even_l764_764044


namespace probability_multiple_of_four_l764_764948

theorem probability_multiple_of_four : 
  let multiples_of_4_count := 25 in
  let total_count := 100 in
  let prob_not_multiple_4 := (total_count - multiples_of_4_count : ℝ) / total_count in
  let prob_not_multiple_4_twice := prob_not_multiple_4 * prob_not_multiple_4 in
  let prob_at_least_one_multiple_4 := 1 - prob_not_multiple_4_twice in
  prob_at_least_one_multiple_4 = (7 / 16 : ℝ) :=
by
  let multiples_of_4_count := 25
  let total_count := 100
  let prob_not_multiple_4 := (total_count - multiples_of_4_count : ℝ) / total_count
  let prob_not_multiple_4_twice := prob_not_multiple_4 * prob_not_multiple_4
  let prob_at_least_one_multiple_4 := 1 - prob_not_multiple_4_twice
  show prob_at_least_one_multiple_4 = (7 / 16 : ℝ)
  sorry

end probability_multiple_of_four_l764_764948


namespace value_of_a_l764_764692

def f (a x : ℝ) : ℝ :=
if x < 1 then 2^x + 1 else x^2 + a * x

theorem value_of_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 := by
  sorry

end value_of_a_l764_764692


namespace min_of_A_div_B_l764_764334

theorem min_of_A_div_B {x A B : ℝ} (hA : x^2 + 1/x^2 = A) (hB : 3 * (x - 1/x) = B) (hA_pos : 0 < A) (hB_pos : 0 < B) :
  (∃ B : ℝ, B = 3 * real.sqrt 2 ∧ A = (B^2 / 9 + 2)) → min_val (A / B) = (2 * real.sqrt 2 / 3) := 
sorry

end min_of_A_div_B_l764_764334


namespace eval_definite_integral_sin_l764_764229

noncomputable def definite_integral_eval : ℝ :=
  ∫ x in 0..π, Real.sin x 

theorem eval_definite_integral_sin : definite_integral_eval = 2 :=
by
  sorry

end eval_definite_integral_sin_l764_764229


namespace one_fourth_of_8_point_8_is_fraction_l764_764997

theorem one_fourth_of_8_point_8_is_fraction:
  (1 / 4) * 8.8 = 11 / 5 :=
by sorry

end one_fourth_of_8_point_8_is_fraction_l764_764997


namespace find_p_plus_q_l764_764917

theorem find_p_plus_q
  (p q : ℕ) (h : ℝ)
  (h_frac : h = p / q)
  (perimeter_condition : real.sqrt (h^2 / 4 + 56.25) + real.sqrt (156.25 + h^2 / 4) + real.sqrt (100 + h^2 / 4) = 45)
  (rel_prime : nat.coprime p q) :
  p + q = 13 :=
sorry

end find_p_plus_q_l764_764917


namespace arc_length_l764_764436

/-- Given a circle with a radius of 5 cm and a sector area of 11.25 cm², 
prove that the length of the arc forming the sector is 4.5 cm. --/
theorem arc_length (r : ℝ) (A : ℝ) (θ : ℝ) (arc_length : ℝ) 
  (h_r : r = 5) 
  (h_A : A = 11.25) 
  (h_area_formula : A = (θ / (2 * π)) * π * r ^ 2) 
  (h_arc_length_formula : arc_length = r * θ) :
  arc_length = 4.5 :=
sorry

end arc_length_l764_764436


namespace simplify_f_values_of_cos_tan_l764_764303

def f (x : ℝ) : ℝ := (cos (x - (π / 2))) / (sin ((7 * π / 2) + x)) * (cos (π - x))

theorem simplify_f (x : ℝ) : f x = sin x :=
by sorry

theorem values_of_cos_tan (α : ℝ) (h : f α = -5 / 13) :
  cos α = 12 / 13 ∨ cos α = -12 / 13 ∧
  (tan α = 5 / 12 ∨ tan α = -5 / 12) :=
by sorry

end simplify_f_values_of_cos_tan_l764_764303


namespace complex_solution_l764_764240

theorem complex_solution (w : ℂ) (h : 3 * w + 4 * conj w = 8 - 10 * complex.I) : 
  w = 8 / 7 + 10 * complex.I :=
sorry

end complex_solution_l764_764240


namespace common_point_of_four_circles_l764_764804

-- Define points A, B, C, D on a plane.
variables {P : Type} [metric_space P]

def midpoint (A B : P) : P := sorry

-- Definitions of midpoints for all segments
def M_AB (A B : P) : P := midpoint A B
def M_AC (A C : P) : P := midpoint A C
def M_AD (A D : P) : P := midpoint A D
def M_BA (B A : P) : P := midpoint B A
def M_BC (B C : P) : P := midpoint B C
def M_BD (B D : P) : P := midpoint B D
def M_CA (C A : P) : P := midpoint C A
def M_CB (C B : P) : P := midpoint C B
def M_CD (C D : P) : P := midpoint C D
def M_DA (D A : P) : P := midpoint D A
def M_DB (D B : P) : P := midpoint D B
def M_DC (D C : P) : P := midpoint D C

-- Define the circles that pass through the midpoints
def Γ_A (A B C : P) : Type := { P | P = M_AB A B ∨ P = M_AC A C ∨ P = M_AD A D }
def Γ_B (A B C : P) : Type := { P | P = M_BA B A ∨ P = M_BC B C ∨ P = M_BD B D }
def Γ_C (A B C : P) : Type := { P | P = M_CA C A ∨ P = M_CB C B ∨ P = M_CD C D }
def Γ_D (A B C : P) : Type := { P | P = M_DA D A ∨ P = M_DB D B ∨ P = M_DC D C }

noncomputable def circles_have_common_point (A B C D : P) : Prop :=
  ∃ P : P, P ∈ Γ_A A B C ∧ P ∈ Γ_B A B C ∧ P ∈ Γ_C A B C ∧ P ∈ Γ_D A B C

theorem common_point_of_four_circles (A B C D : P) :
  circles_have_common_point A B C D :=
sorry

end common_point_of_four_circles_l764_764804


namespace domain_of_function_l764_764446

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 1 / Real.sqrt (2 - x^2)

theorem domain_of_function : 
  {x : ℝ | x > -1 ∧ x < Real.sqrt 2} = {x : ℝ | x ∈ Set.Ioo (-1) (Real.sqrt 2)} :=
by
  sorry

end domain_of_function_l764_764446


namespace number_of_true_propositions_is_three_l764_764570

-- Definitions based on the conditions provided
variables (α β r : Plane) (l m : Line)

-- Propositions to evaluate
def prop1 := (α ⊥ r) ∧ (β ⊥ r) → (α ∥ β)
def prop2 := (α ∥ r) ∧ (β ∥ r) → (α ∥ β)
def prop3 := (α ⊥ l) ∧ (β ⊥ l) → (α ∥ β)
def prop4 := (skew l m) ∧ (l ∥ α) ∧ (m ∥ α) ∧ (l ∥ β) ∧ (m ∥ β) → (α ∥ β)

-- The theorem statement
theorem number_of_true_propositions_is_three :
  (prop1 → false) ∧ prop2 ∧ prop3 ∧ prop4 :=
sorry

end number_of_true_propositions_is_three_l764_764570


namespace acute_triangle_OAB_l764_764371

theorem acute_triangle_OAB (a : ℝ) :
  let OA := (0, a, 1) in
  let OB := (-1, 2, 2) in
  let AB := (-1, 2 - a, 1) in
  (0 * -1 + a * 2 + 1 * 2 > 0) ∧ 
  (-1 * 0 + (2 - a) * -a + 1 * -1 > 0) ∧ 
  (1 * 1 + (a - 2) * -2 - 1 * 2 > 0) ↔ 
  1 + real.sqrt 2 < a ∧ a < 7 / 2 :=
sorry

end acute_triangle_OAB_l764_764371


namespace taishan_maiden_tea_prices_l764_764900

theorem taishan_maiden_tea_prices (x y : ℝ) 
  (h1 : 30 * x + 20 * y = 6000)
  (h2 : 24 * x + 18 * y = 5100) :
  x = 100 ∧ y = 150 :=
by
  sorry

end taishan_maiden_tea_prices_l764_764900


namespace original_price_l764_764915

theorem original_price (P S : ℝ) (h1 : S = 1.25 * P) (h2 : S - P = 625) : P = 2500 := by
  sorry

end original_price_l764_764915


namespace valid_integer_pairs_l764_764699

theorem valid_integer_pairs :
  { (x, y) : ℤ × ℤ |
    (∃ α β : ℝ, α^2 + β^2 < 4 ∧ α + β = (-x : ℝ) ∧ α * β = y ∧ x^2 - 4 * y ≥ 0) } =
  {(-2,1), (-1,-1), (-1,0), (0, -1), (0,0), (1,0), (1,-1), (2,1)} :=
sorry

end valid_integer_pairs_l764_764699


namespace minimum_value_of_S_l764_764967

open Real

noncomputable def polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_S 
(h0 : polynomial a b c 0 = 0) 
(h1 : polynomial a b c 2 = 2) 
: ∃ S, S = ∫ x in 0..2, abs (2 * a * x + b) ∧ S = 2 :=
sorry

end minimum_value_of_S_l764_764967


namespace find_fraction_l764_764950

theorem find_fraction : 
  ∀ (x : ℚ), (120 - x * 125 = 45) → x = 3 / 5 :=
by
  intro x
  intro h
  sorry

end find_fraction_l764_764950


namespace perpendicular_tangent_line_exists_and_correct_l764_764632

theorem perpendicular_tangent_line_exists_and_correct :
  ∃ L : ℝ → ℝ → Prop,
    (∀ x y, L x y ↔ 3 * x + y + 6 = 0) ∧
    (∀ x y, 2 * x - 6 * y + 1 = 0 → 3 * x + y + 6 ≠ 0) ∧
    (∃ a b : ℝ, 
       b = a^3 + 3*a^2 - 5 ∧ 
       (a, b) ∈ { p : ℝ × ℝ | ∃ f' : ℝ → ℝ, f' a = 3 * a^2 + 6 * a ∧ f' a * 3 + 1 = 0 } ∧
       L a b)
:= 
sorry

end perpendicular_tangent_line_exists_and_correct_l764_764632


namespace triangle_max_area_l764_764376

open Real

theorem triangle_max_area (a b c : ℝ) (B : ℝ) (hB : B = π / 4) (hAC : b = 4) :
  let S := (1 / 2) * a * c * sin B in
  S ≤ 4 + 4 * sqrt 2 :=
sorry

end triangle_max_area_l764_764376


namespace marissas_sunflower_height_in_meters_l764_764792

-- Define the conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Define the given data
def sister_height_feet : ℝ := 4.15
def additional_height_cm : ℝ := 37
def height_difference_inches : ℝ := 63

-- Calculate the height of Marissa's sunflower in meters
theorem marissas_sunflower_height_in_meters :
  let sister_height_inches := sister_height_feet * inches_per_foot
  let sister_height_cm := sister_height_inches * cm_per_inch
  let total_sister_height_cm := sister_height_cm + additional_height_cm
  let height_difference_cm := height_difference_inches * cm_per_inch
  let marissas_sunflower_height_cm := total_sister_height_cm + height_difference_cm
  let marissas_sunflower_height_m := marissas_sunflower_height_cm / cm_per_meter
  marissas_sunflower_height_m = 3.23512 :=
by
  sorry

end marissas_sunflower_height_in_meters_l764_764792


namespace problem_statement_l764_764572

-- Define the given functions
def f_A (x : ℝ) : ℝ := 1 / x
def f_B (x : ℝ) : ℝ := -x + 1 / x
def f_C (x : ℝ) : ℝ := -x * abs x
def f_D (x : ℝ) : ℝ := if x > 0 then -x + 1 else -x - 1

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define what it means for a function to be decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

-- Main statement to prove that among the given functions, only f_C is both odd and decreasing
theorem problem_statement :
  (is_odd f_C ∧ is_decreasing f_C) ∧
  ¬(is_odd f_A ∧ is_decreasing f_A) ∧
  ¬(is_odd f_B ∧ is_decreasing f_B) ∧
  ¬(is_odd f_D ∧ is_decreasing f_D) :=
by
  sorry

end problem_statement_l764_764572


namespace relationship_among_abc_l764_764673

noncomputable def a := (0.5:ℝ)^0.4
noncomputable def b := Real.log 0.3 / Real.log 0.4
noncomputable def c := Real.log 0.4 / Real.log 8

theorem relationship_among_abc : c < a ∧ a < b := sorry

end relationship_among_abc_l764_764673


namespace exists_N_gt_2017_pow_2017_l764_764400

theorem exists_N_gt_2017_pow_2017 :
  ∃ N : ℕ, 1 ≤ N ∧ (∑ i in Finset.range N, a i) > 2017 ^ 2017 :=
sorry

namespace sequence

def a : ℕ → ℝ
| 0     := 1
| (n+1) := 1 / (Finset.range (n+1)).sum (λ i, a i ^ 2)

end sequence

end exists_N_gt_2017_pow_2017_l764_764400


namespace perimeter_of_second_square_l764_764835

-- Definitions and conditions
def perimeter_first_square : ℝ := 24
def perimeter_sum_area_square : ℝ := 40

-- Calculate the required perimeter
theorem perimeter_of_second_square : ∃ perimeter_second_square : ℝ, perimeter_second_square = 32 :=
by
  let s₁ := perimeter_first_square / 4
  have area₁ := s₁ ^ 2

  let s₃ := perimeter_sum_area_square / 4
  have area₃ := s₃ ^ 2

  have area₂ := area₃ - area₁
  let s₂ := real.sqrt area₂
  let p₂ := 4 * s₂

  use p₂
  sorry

end perimeter_of_second_square_l764_764835


namespace one_eighth_of_power_l764_764338

theorem one_eighth_of_power (x : ℕ) (h : (1 / 8) * (2 ^ 36) = 2 ^ x) : x = 33 :=
by 
  -- Proof steps are not needed, so we leave it as sorry.
  sorry

end one_eighth_of_power_l764_764338


namespace solve_inverse_function_l764_764064

-- Define the given functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 1
def g (x : ℝ) : ℝ := x^4 - x^3 + 4*x^2 + 8*x + 8
def h (x : ℝ) : ℝ := x + 1

-- State the mathematical equivalent proof problem
theorem solve_inverse_function (x : ℝ) :
  f ⁻¹' {g x} = {y | h y = x + 1} ↔
  (x = (3 + Real.sqrt 5) / 2) ∨ (x = (3 - Real.sqrt 5) / 2) :=
sorry -- Proof is omitted

end solve_inverse_function_l764_764064


namespace intervals_of_increase_range_of_m_l764_764307

-- Definition of f(x)
def f (x : ℝ) (a b : ℝ) := (1/3) * x^3 + a * x + b

-- Conditions stated in the problem
variables (a b : ℝ)
axiom a_eq : a = -4
axiom b_eq : b = 4

-- Function for derivative of f(x)
def f' (x : ℝ) := x^2 + a

-- Additional given conditions
axiom local_min_cond : f 2 a b = -4 / 3

-- Lean proof statements
theorem intervals_of_increase : {x : ℝ | 0 < f' x} = {x : ℝ | x < -2 ∨ 2 < x} :=
by 
  sorry

theorem range_of_m (m : ℝ) : (∀ x ∈ set.Icc (-4 : ℝ) 3, f x (-4) 4 ≤ m^2 + m + 10 / 3) ↔ (m ≤ -2 ∨ 1 ≤ m) :=
by 
  sorry

end intervals_of_increase_range_of_m_l764_764307


namespace problem_1_simplification_l764_764588

theorem problem_1_simplification (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) : 
  (x - 2) / (x ^ 2) / (1 - 2 / x) = 1 / x := 
  sorry

end problem_1_simplification_l764_764588


namespace total_weight_of_verna_sherry_jake_l764_764494

-- Definitions based on given conditions
def haley_weight : ℝ := 103
def verna_weight : ℝ := haley_weight + 17
def sherry_weight : ℝ := 2 * verna_weight
def combined_haley_verna_weight : ℝ := haley_weight + verna_weight
def jake_weight : ℝ := (3 / 5) * combined_haley_verna_weight

-- Total combined weight of Verna, Sherry, and Jake
def total_weight : ℝ := verna_weight + sherry_weight + jake_weight

-- Theorem to prove the equivalent proof problem
theorem total_weight_of_verna_sherry_jake : total_weight = 493.8 :=
by
  -- Insert proof here
  sorry

end total_weight_of_verna_sherry_jake_l764_764494


namespace quadrilateral_inscribed_circumscribed_l764_764137

theorem quadrilateral_inscribed_circumscribed 
  (r R d : ℝ) --Given variables with their types
  (K O : Type) (radius_K : K → ℝ) (radius_O : O → ℝ) (dist : (K × O) → ℝ)  -- Defining circles properties
  (K_inside_O : ∀ p : K × O, radius_K p.fst < radius_O p.snd) 
  (dist_centers : ∀ p : K × O, dist p = d) -- Distance between the centers
  : 
  (1 / (R + d)^2) + (1 / (R - d)^2) = (1 / r^2) := 
by 
  sorry

end quadrilateral_inscribed_circumscribed_l764_764137


namespace inequality_solution_l764_764065

theorem inequality_solution (x : ℝ) :
  ( (x^2 + 3*x + 3) > 0 ) → ( ((x^2 + 3*x + 3)^(5*x^3 - 3*x^2)) ≤ ((x^2 + 3*x + 3)^(3*x^3 + 5*x)) )
  ↔ ( x ∈ (Set.Iic (-2) ∪ ({-1} : Set ℝ) ∪ Set.Icc 0 (5/2)) ) :=
by
  sorry

end inequality_solution_l764_764065


namespace mighty_L_teams_l764_764368

theorem mighty_L_teams (n : ℕ) 
  (H : ∃ (n : ℕ), (n * (n - 1)) / 2 = 28) : n = 8 :=
by {
  have H_eq : (n * (n - 1)) / 2 = 28 := H.some_spec,
  sorry,
}

end mighty_L_teams_l764_764368


namespace sin_product_l764_764962

theorem sin_product :
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (36 * Real.pi / 180)) *
  (Real.sin (72 * Real.pi / 180)) *
  (Real.sin (84 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end sin_product_l764_764962


namespace sum_of_T_elements_l764_764759

-- Define T to represent the set of numbers 0.abcd with a, b, c, d being distinct digits.
def is_valid_abcd (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def T : Set ℝ := { x | ∃ (a b c d : ℕ), is_valid_abcd a b c d ∧ x = ((1000 * a + 100 * b + 10 * c + d : ℝ) / 9999) }

-- The main theorem statement.
theorem sum_of_T_elements : ∑ x in T, x = 2520 := by
  sorry

end sum_of_T_elements_l764_764759


namespace max_value_quadratic_function_l764_764458

theorem max_value_quadratic_function : 
  ∃ x : ℝ, ∀ y : ℝ, (y = -2 * x^2 + 4 * x - 6) ∧ (∀ z, y ≥ -2 * z^2 + 4 * z - 6) :=
begin
  sorry
end

end max_value_quadratic_function_l764_764458


namespace probability_of_entirely_black_grid_l764_764148

open Classical

noncomputable def probability_grid_black : ℚ :=
  let pairs := 8 in
  let prob_pair_black := (1 : ℚ) / 4 in
  prob_pair_black ^ pairs

theorem probability_of_entirely_black_grid :
  probability_grid_black = 1 / 65536 :=
by
  sorry

end probability_of_entirely_black_grid_l764_764148


namespace domain_of_g_l764_764241

def g (x : ℝ) : ℝ := sqrt (x + 3) + log2 (6 - x)

theorem domain_of_g :
  {x : ℝ | -3 ≤ x ∧ x < 6} = {x : ℝ | ∃ y : ℝ, g(y) = g(x)} :=
by
  sorry

end domain_of_g_l764_764241


namespace min_value_expression_l764_764393

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ z, a ^ 3 + b ^ 3 + 1 / (a + b) ^ 3 = z ∧ z = (3 √ 4)/2 :=
sorry

end min_value_expression_l764_764393


namespace phantom_needs_more_money_l764_764417

def amount_phantom_has : ℤ := 50
def cost_black : ℤ := 11
def count_black : ℕ := 2
def cost_red : ℤ := 15
def count_red : ℕ := 3
def cost_yellow : ℤ := 13
def count_yellow : ℕ := 2

def total_cost : ℤ := cost_black * count_black + cost_red * count_red + cost_yellow * count_yellow
def additional_amount_needed : ℤ := total_cost - amount_phantom_has

theorem phantom_needs_more_money : additional_amount_needed = 43 := by
  sorry

end phantom_needs_more_money_l764_764417


namespace correct_X_Y_Z_l764_764373

def nucleotide_types (A_types C_types T_types : ℕ) : ℕ :=
  A_types + C_types + T_types

def lowest_stability_period := "interphase"

def separation_period := "late meiosis I or late meiosis II"

theorem correct_X_Y_Z :
  nucleotide_types 2 2 1 = 3 ∧ 
  lowest_stability_period = "interphase" ∧ 
  separation_period = "late meiosis I or late meiosis II" :=
by
  sorry

end correct_X_Y_Z_l764_764373


namespace arithmetic_square_root_of_nine_l764_764821

theorem arithmetic_square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l764_764821


namespace sales_volume_formula_selling_price_for_profit_8000_maximum_profit_l764_764156

section fruit_store

variables (x y : ℝ)
parameters 
  (purchase_price : ℝ := 30)
  (base_price : ℝ := 40)
  (base_sales : ℝ := 500)
  (decrease_rate : ℝ := 10)
  (desired_profit : ℝ := 8000)

-- Condition 1: Sales volume formula
def sales_volume (x : ℝ) : ℝ := -10 * x + 900

-- Condition 2: Selling price for a specific profit
def profit_eq_8000 (x : ℝ) : Prop := (x - purchase_price) * sales_volume x = desired_profit

-- Condition 3: Maximum profit
def max_profit : ℝ := 9000
def max_profit_price : ℝ := 60

theorem sales_volume_formula : sales_volume x = -10 * x + 900 := rfl

theorem selling_price_for_profit_8000 : ∀ x, profit_eq_8000 x → x = 50 := sorry

theorem maximum_profit : ∀ x, x = max_profit_price ↔ ( ∀ y, (-10 * y + 900) * (y - purchase_price) ≤ max_profit) := sorry

end fruit_store

end sales_volume_formula_selling_price_for_profit_8000_maximum_profit_l764_764156


namespace gcd_special_case_l764_764453

theorem gcd_special_case (m n : ℕ) (h : Nat.gcd m n = 1) :
  Nat.gcd (m + 2000 * n) (n + 2000 * m) = 2000^2 - 1 :=
sorry

end gcd_special_case_l764_764453


namespace median_room_number_of_present_mathletes_l764_764945

theorem median_room_number_of_present_mathletes : 
  let available_rooms := (List.range 25).map (λ x, x + 1)
  let occupied_rooms := List.erase (List.erase (List.erase available_rooms 10) 11) 25
  let sorted_occupied_rooms := List.sort (λ a b => a < b) occupied_rooms
  let n := List.length sorted_occupied_rooms
  let median_value := (sorted_occupied_rooms.get (n / 2 - 1) + sorted_occupied_rooms.get (n / 2)) / 2
  n = 22 ∧ median_value = 12.5 :=
by
  sorry

end median_room_number_of_present_mathletes_l764_764945


namespace oranges_picked_l764_764410

theorem oranges_picked (total_oranges second_tree third_tree : ℕ) 
    (h1 : total_oranges = 260) 
    (h2 : second_tree = 60) 
    (h3 : third_tree = 120) : 
    total_oranges - (second_tree + third_tree) = 80 := by 
  sorry

end oranges_picked_l764_764410


namespace sum_of_c_n_l764_764296

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1 else 3 ^ (n - 1)

noncomputable def S (n : ℕ) : ℕ :=
n^2 + 2*n + 1

noncomputable def b (n : ℕ) : ℕ :=
if n = 1 then 4 else 2*n + 1

noncomputable def c (n : ℕ) : ℕ :=
if n = 1 then 4 else (2*n + 1) * 3^(n - 1)

noncomputable def T (n : ℕ) : ℕ :=
n * 3^n + 1

theorem sum_of_c_n (n : ℕ) : T n = ∑ k in finset.range n, c (k + 1) :=
sorry

end sum_of_c_n_l764_764296


namespace simplify_product_l764_764053

theorem simplify_product : 
  (∀ (n: ℕ), n ≥ 1 →
    let term := (∏ (k: ℕ) in (finset.range n).filter (λ k, k > 0), ((3 * k + 6) / (3 * k))) in
    term = (3003 / 3)) → 
  ∑ (k : ℕ) in finset.range 1001, k = 1001 :=
sorry

end simplify_product_l764_764053


namespace find_term_2016_l764_764667

variable {n : ℕ}

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ a 2 = 2 ∧ 
  ∀ n ≥ 2, (S (n + 1) = 3 * S n - 2 * S (n - 1) - a (n - 1) + 2)

theorem find_term_2016 (a : ℕ → ℤ) (S : ℕ → ℤ) :
  sequence a → a 2016 = 2016^2 - 2 :=
sorry

end find_term_2016_l764_764667


namespace man_walking_rate_l764_764908

theorem man_walking_rate (x : ℝ) 
  (woman_rate : ℝ := 15)
  (woman_time_after_passing : ℝ := 2 / 60)
  (man_time_to_catch_up : ℝ := 4 / 60)
  (distance_woman : ℝ := woman_rate * woman_time_after_passing)
  (distance_man : ℝ := x * man_time_to_catch_up)
  (h : distance_man = distance_woman) :
  x = 7.5 :=
sorry

end man_walking_rate_l764_764908


namespace tiles_needed_to_cover_floor_l764_764904

theorem tiles_needed_to_cover_floor :
    let floor_length := 10 in
    let floor_width := 15 in
    let tile_length := (3 / 12 : ℝ) in
    let tile_width := (9 / 12 : ℝ) in
    let floor_area := floor_length * floor_width in
    let tile_area := tile_length * tile_width in
    floor_area / tile_area = 800 := by
  let floor_length := 10
  let floor_width := 15
  let tile_length := (3 / 12 : ℝ)
  let tile_width := (9 / 12 : ℝ)
  let floor_area := floor_length * floor_width
  let tile_area := tile_length * tile_width
  show floor_area / tile_area = 800
  by
    sorry

end tiles_needed_to_cover_floor_l764_764904


namespace probability_computation_l764_764553

noncomputable def probability_inside_sphere : ℝ :=
  let volume_of_cube : ℝ := 64
  let volume_of_sphere : ℝ := (4/3) * Real.pi * (2^3)
  volume_of_sphere / volume_of_cube

theorem probability_computation :
  probability_inside_sphere = Real.pi / 6 :=
by
  sorry

end probability_computation_l764_764553


namespace sin_cos_difference_identity_l764_764609

theorem sin_cos_difference_identity :
  sin (77 * real.pi / 180) * cos (47 * real.pi / 180) - cos (77 * real.pi / 180) * sin (47 * real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_cos_difference_identity_l764_764609


namespace area_of_IJKL_is_144_l764_764428

-- Define the setup of the squares and their characteristics
variables (WXYZ : Type) [geometry WXYZ] (IJKL : Type) [geometry IJKL]

-- Side length of square WXYZ
constant side_length_WXYZ : ℝ
axiom side_length_WXYZ_is_10 : side_length_WXYZ = 10

-- Distance WI
constant WI : ℝ
axiom WI_is_2 : WI = 2

-- Inner square properties
constant area_IJKL : ℝ

-- Proof statement that the area of IJKL is 144
theorem area_of_IJKL_is_144 (h1 : IJKL ⊆ WXYZ)
  (h2 : side_length_WXYZ = 10)
  (h3 : WI = 2) : area_IJKL = 144 := 
sorry

end area_of_IJKL_is_144_l764_764428


namespace option_C_is_correct_l764_764126

-- Define the conditions as propositions
def condition_A := |-2| = 2
def condition_B := (-1)^2 = 1
def condition_C := -7 + 3 = -4
def condition_D := 6 / (-2) = -3

-- The statement that option C is correct
theorem option_C_is_correct : condition_C := by
  sorry

end option_C_is_correct_l764_764126


namespace work_rate_l764_764530

/-- 
A alone can finish a work in some days which B alone can finish in 15 days. 
If they work together and finish it, then out of a total wages of Rs. 3400, 
A will get Rs. 2040. Prove that A alone can finish the work in 22.5 days. 
-/
theorem work_rate (A : ℚ) (B_rate : ℚ) 
  (total_wages : ℚ) (A_wages : ℚ) 
  (total_rate : ℚ) 
  (hB : B_rate = 1 / 15) 
  (hWages : total_wages = 3400 ∧ A_wages = 2040) 
  (hTotal : total_rate = 1 / A + B_rate)
  (hWorkTogether : 
    (A_wages / (total_wages - A_wages) = 51 / 34) ↔ 
    (A / (A + 15) = 51 / 85)) : 
  A = 22.5 := 
sorry

end work_rate_l764_764530


namespace investor_more_money_in_A_l764_764577

noncomputable def investment_difference 
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ) :
  ℝ :=
investment_A * (1 + yield_A) - investment_B * (1 + yield_B)

theorem investor_more_money_in_A
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ)
  (hA : investment_A = 300)
  (hB : investment_B = 200)
  (hYA : yield_A = 0.3)
  (hYB : yield_B = 0.5)
  :
  investment_difference investment_A investment_B yield_A yield_B = 90 := 
by
  sorry

end investor_more_money_in_A_l764_764577


namespace plastic_skulls_number_l764_764968

-- Define the conditions
def num_broomsticks : ℕ := 4
def num_spiderwebs : ℕ := 12
def num_pumpkins := 2 * num_spiderwebs
def num_cauldron : ℕ := 1
def budget_left_to_buy : ℕ := 20
def num_left_to_put_up : ℕ := 10
def total_decorations : ℕ := 83

-- The number of plastic skulls calculation as a function
def num_other_decorations : ℕ :=
  num_broomsticks + num_spiderwebs + num_pumpkins + num_cauldron + budget_left_to_buy + num_left_to_put_up

def num_plastic_skulls := total_decorations - num_other_decorations

-- The theorem to be proved
theorem plastic_skulls_number : num_plastic_skulls = 12 := by
  sorry

end plastic_skulls_number_l764_764968


namespace problem1_problem2_l764_764304

-- Lean 4 statement for part (1)
theorem problem1 {a : ℝ} : 
  (∀ x ∈ Icc (-1 : ℝ) 0, f x = (1 / 4^x) + (a / 2^x) + 1) →
  ((∀ x ∈ Icc (-1 : ℝ) 0, f x ≥ -7) ∧ ∃ x ∈ Icc (-1 : ℝ) 0, f x = -7) →
  a = -6 := by
  sorry

-- Lean 4 statement for part (2)
theorem problem2 {a : ℝ} :
  (∀ x ∈ Ici (0 :ℝ), f x = (1 / 4^x) + (a / 2^x) + 1) →
  (∀ x ∈ Ici (0 : ℝ), |f x| ≤ 3) →
  -5 ≤ a ∧ a ≤ 1 := by
  sorry

end problem1_problem2_l764_764304


namespace find_perpendicular_tangent_line_l764_764629

noncomputable def line_eq (a b c x y : ℝ) : Prop :=
a * x + b * y + c = 0

def perp_line (a b c d e f : ℝ) (x y : ℝ) : Prop :=
b * x - a * y = 0  -- Perpendicular condition

def tangent_line (f : ℝ → ℝ) (a b c x : ℝ) : Prop :=
∃ t, f t = a * t + b * (f t) + c ∧ (deriv f t) = -a / b  -- Tangency condition with derivative

theorem find_perpendicular_tangent_line :
  let f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 5 in
  ∃ a b c d e f: ℝ, perp_line 2 (-6) 1 a b c ∧ tangent_line f a b c ∧ line_eq 3 1 6 (x : ℝ) (f x) :=
sorry

end find_perpendicular_tangent_line_l764_764629


namespace probability_multiple_of_105_l764_764358

open Set

def S : Set ℕ := {5, 15, 21, 35, 45, 49, 63}

theorem probability_multiple_of_105 : 
  (let pairs := {x | ∃ (a ∈ S) (b ∈ S), a ≠ b ∧ x = (a, b)}.toFinset 
   in (pairs.filter (λ x => 105 ∣ (x.1 * x.2))).card) / ({x | ∃ (a ∈ S) (b ∈ S), a ≠ b}.toFinset.card : ℚ) 
     = 4 / 7 :=
by
  sorry

end probability_multiple_of_105_l764_764358


namespace library_books_count_l764_764556

def students_per_day : List ℕ := [4, 5, 6, 9]
def books_per_student : ℕ := 5
def total_books_given (students : List ℕ) (books_per_student : ℕ) : ℕ :=
  students.foldl (λ acc n => acc + n * books_per_student) 0

theorem library_books_count :
  total_books_given students_per_day books_per_student = 120 :=
by
  sorry

end library_books_count_l764_764556


namespace problem_statement_l764_764264

theorem problem_statement (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) = a (i + 1) ^ a (i + 2)): 
  a 0 = a 1 :=
sorry

end problem_statement_l764_764264


namespace simplify_fraction_sequence_l764_764059

theorem simplify_fraction_sequence :
  (∏ k in finset.range 1000, (3 * (k + 3) + 3) / (3 * (k + 3)))
  = 1001 :=
by
suffices : (∏ k in finset.range 1000, (3 * (k + 3) + 3) / (3 * (k + 3))) = 3003 / 3,
{ rwa [div_eq_mul_inv, mul_inv_cancel (3 : ℤ), inv_one, mul_one] },
sorry

end simplify_fraction_sequence_l764_764059


namespace calculate_value_l764_764123

theorem calculate_value (x y d : ℕ) (hx : x = 2024) (hy : y = 1935) (hd : d = 225) : 
  (x - y)^2 / d = 35 := by
  sorry

end calculate_value_l764_764123


namespace problem1_statement_problem2_statement_l764_764032

-- Defining the sets A and B
def set_A (x : ℝ) := 2*x^2 - 7*x + 3 ≤ 0
def set_B (x a : ℝ) := x + a < 0

-- Problem 1: Intersection of A and B when a = -2
def question1 (x : ℝ) : Prop := set_A x ∧ set_B x (-2)

-- Problem 2: Range of a for A ∩ B = A
def question2 (a : ℝ) : Prop := ∀ x, set_A x → set_B x a

theorem problem1_statement :
  ∀ x, question1 x ↔ x >= 1/2 ∧ x < 2 :=
by sorry

theorem problem2_statement :
  ∀ a, (∀ x, set_A x → set_B x a) ↔ a < -3 :=
by sorry

end problem1_statement_problem2_statement_l764_764032


namespace train_speed_l764_764182

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 280) (h_time : time = 20) :
  (length / time) * 3.6 = 50.4 :=
by
  -- Definitions
  rw [h_length, h_time]
  -- Simplify and calculate
  have h1 : 280 / 20 = 14 := by norm_num
  have h2 : 14 * 3.6 = 50.4 := by norm_num
  -- Combine results
  rw [h1, h2]
  -- Conclude
  rfl

end train_speed_l764_764182


namespace sin_neg_330_eq_half_l764_764213

theorem sin_neg_330_eq_half : (Real.sin (-330 * Real.pi / 180)) = 1 / 2 :=
by
  have h1 : -330 + 360 = 30 := by sorry
  have h2 : Real.sin (-330 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) := by
    rw [Real.sin_angle_add, Real.sin_neg, Real.sin_2pi, h1]
  rw [h2] 
  exact Real.sin_30

end sin_neg_330_eq_half_l764_764213


namespace verify_solution_1_verify_solution_2_verify_solution_3_l764_764493

-- 1. Verify y = Cx is a solution to xy' - y = 0
theorem verify_solution_1 (C : ℝ) : 
  ∀ (x : ℝ), x * (deriv (λ x, C * x)) - (C * x) = 0 :=
  by sorry

-- 2. Verify y = C₁x + C₂x² is a solution to y'' - (2/x)y' + (2y/x²) = 0
theorem verify_solution_2 (C₁ C₂ : ℝ) : 
  ∀ (x : ℝ), deriv (deriv (λ x, C₁ * x + C₂ * x^2)) - (2 / x) * deriv (λ x, C₁ * x + C₂ * x^2) + (2 / x^2) * (C₁ * x + C₂ * x^2) = 0 :=
  by sorry

-- 3. Verify v = C₁/r + C₂ is a solution to v'' + (2/r)v' = 0
theorem verify_solution_3 (C₁ C₂ : ℝ) : 
  ∀ (r : ℝ), deriv (deriv (λ r, C₁ / r + C₂)) + (2 / r) * deriv (λ r, C₁ / r + C₂) = 0 :=
  by sorry

end verify_solution_1_verify_solution_2_verify_solution_3_l764_764493


namespace max_non_dominating_partitions_20_l764_764914

def partition (n : ℕ) := { l : List ℕ // l.sum = n ∧ (∀ x ∈ l, x > 0) }

def dominates (a b : List ℕ) : Prop :=
  ∀ i : ℕ, (∑ j in List.finRange (min a.length (i + 1)), a[j]) ≥ 
           (∑ j in List.finRange (min b.length (i + 1)), b[j])

theorem max_non_dominating_partitions_20 : 
  ∃ (partitions : Finset (partition 20)), partitions.card = 20 ∧ 
  ∀ (a b ∈ partitions), ¬ dominates a.val b.val ∧ ¬ dominates b.val a.val := 
sorry

end max_non_dominating_partitions_20_l764_764914


namespace ellipse_proof_fixed_point_proof_l764_764279

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (c = a * (Real.sqrt 2) / 2) ∧ ∀ x y : ℝ,
  (y = x + 1) → (y ^ 2 = 4 * x) → (x ^ 2 / 2 + y ^ 2 = 1)

noncomputable def fixed_point : Prop :=
  ∃ (T : ℝ × ℝ), T = (0, 1) ∧ ∀ A B : ℝ × ℝ, ∃ l : ℝ → ℝ,
  l(0) = -1/3 ∧ (∃ x y : ℝ, x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) →
  ∀ x : ℝ, (x ^ 2 + (x + 1/3) ^ 2 = (4/3) ^ 2) ∨ (x ^ 2 + x ^ 2 = 1) →
  (A.1 * B.1 + A.2 * B.2 = 0) ∧ (T ∈ (Circle (A, B).diameter))

theorem ellipse_proof : ellipse_equation := sorry

theorem fixed_point_proof : fixed_point := sorry

end ellipse_proof_fixed_point_proof_l764_764279


namespace solve_pair_N_n_l764_764986

def is_solution_pair (N n : ℕ) : Prop :=
  N ^ 2 = 1 + n * (N + n)

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem solve_pair_N_n (N n : ℕ) (i : ℕ) :
  is_solution_pair N n ↔ N = fibonacci (i + 1) ∧ n = fibonacci i := sorry

end solve_pair_N_n_l764_764986


namespace matrix_min_sum_l764_764778

theorem matrix_min_sum (p q r s : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h : (pmatrix.of [[p, q], [r, s]] ⬝ pmatrix.of [[p, q], [r, s]]) = pmatrix.of [[10, 0], [0, 10]]) : 
  |p| + |q| + |r| + |s| = 8 :=
by
  sorry

end matrix_min_sum_l764_764778


namespace bodies_distance_apart_l764_764003

def distance_fallen (t : ℝ) : ℝ := 4.9 * t^2

theorem bodies_distance_apart (t : ℝ) (h₁ : 220.5 = distance_fallen t - distance_fallen (t - 5)) : t = 7 :=
by {
  sorry
}

end bodies_distance_apart_l764_764003


namespace water_hyacinth_indicates_connection_l764_764365

-- Definitions based on the conditions
def universally_interconnected : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (c : Type), (a ≠ c) ∧ (b ≠ c)

def connections_diverse : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (f : a → b), ∀ (x y : a), x ≠ y → f x ≠ f y

def connections_created : Prop :=
  ∃ (a b : Type), a ≠ b ∧ (∀ (f : a → b), False)

def connections_humanized : Prop :=
  ∀ (a b : Type), a ≠ b → (∃ c : Type, a = c) ∧ (∃ d : Type, b = d)

-- Problem statement
theorem water_hyacinth_indicates_connection : 
  universally_interconnected ∧ connections_diverse :=
by
  sorry

end water_hyacinth_indicates_connection_l764_764365


namespace orthogonal_trajectories_angle_at_origin_l764_764642

theorem orthogonal_trajectories_angle_at_origin (x y : ℝ) (a : ℝ) :
  ((x + 2 * y) ^ 2 = a * (x + y)) →
  (∃ φ : ℝ, φ = π / 4) :=
by
  sorry

end orthogonal_trajectories_angle_at_origin_l764_764642


namespace train_speed_including_stoppages_l764_764615

variable (speed_excluding_stoppages : ℝ) (stoppage_minutes_per_hour : ℝ)

def stoppage_hours_per_hour := stoppage_minutes_per_hour / 60
def time_moving_per_hour := 1 - stoppage_hours_per_hour
def distance_moving_per_hour := speed_excluding_stoppages * time_moving_per_hour
def speed_including_stoppages := distance_moving_per_hour / 1

theorem train_speed_including_stoppages 
  (h1 : speed_excluding_stoppages = 45) 
  (h2 : stoppage_minutes_per_hour = 14.67) :
  speed_including_stoppages = 34 := 
sorry

end train_speed_including_stoppages_l764_764615


namespace line_parallel_polar_axis_eq_l764_764745

theorem line_parallel_polar_axis_eq (ρ θ : ℝ) :
  (∃ (ρ θ : ℝ), (ρ = 1 ∧ θ = π / 2)) →
  (∀ (ρ θ : ℝ), (ρ * sin θ = 1)) :=
by
  sorry

end line_parallel_polar_axis_eq_l764_764745


namespace area_ratio_is_five_to_one_l764_764138

-- Definition of a regular hexagon and intersection of diagonals
variables (ABCDEF : Type) [hexagon : IsRegularHexagon ABCDEF]
variables (F C B D G : Point) -- Points on the hexagon
variables (FC BD : Line)
variable  (G : Point)

-- Conditions as Lean definitions
def hexagon_conditions (h1 : F ∈ ABCDEF) (h2 : C ∈ ABCDEF) (h3 : B ∈ ABCDEF) (h4 : D ∈ ABCDEF) 
  (h5 : diagonal FC F C) (h6 : diagonal BD B D) 
  (h7 : FC ∩ BD = G) : Prop := 
  IsIntersectionAt G FC BD

-- The proof statement
theorem area_ratio_is_five_to_one (h1 : F ∈ ABCDEF) (h2 : C ∈ ABCDEF) (h3 : B ∈ ABCDEF) (h4 : D ∈ ABCDEF) 
  (h5 : diagonal FC F C) (h6 : diagonal BD B D) 
  (h7 : FC ∩ BD = G) :
  ratio_of_areas (quadrilateral_area ABCDEF F E D G) (triangle_area ABCDEF B C G) = 5 := 
sorry

end area_ratio_is_five_to_one_l764_764138


namespace simplify_and_evaluate_expression_l764_764811

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = 1) (h₂ : b = -2) :
  (2 * a + b)^2 - 3 * a * (2 * a - b) = -12 :=
by
  rw [h₁, h₂]
  -- Now the expression to prove transforms to:
  -- (2 * 1 + (-2))^2 - 3 * 1 * (2 * 1 - (-2)) = -12
  -- Subsequent proof steps would follow simplification directly.
  sorry

end simplify_and_evaluate_expression_l764_764811


namespace lizzy_loan_amount_l764_764034

noncomputable def interest_rate : ℝ := 0.20
noncomputable def initial_amount : ℝ := 30
noncomputable def final_amount : ℝ := 33

theorem lizzy_loan_amount (X : ℝ) (h : initial_amount + (1 + interest_rate) * X = final_amount) : X = 2.5 := 
by
  sorry

end lizzy_loan_amount_l764_764034


namespace angle_GIH_gt_90_l764_764096

-- Definitions of the points G, I, and H based on the provided conditions in a)
noncomputable def centroid (A B C : Point) : Point := (A + B + C) / 3
noncomputable def orthocenter (A B C : Point) : Point := A + B + C  -- Simplified definition for illustrative purposes.
noncomputable def incenter (A B C : Point) (a b c : ℝ) : Point := (a * A + b * B + c * C) / (a + b + c)

-- Prove the main theorem
theorem angle_GIH_gt_90 
  (A B C : Point)
  (a b c : ℝ) 
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : C ≠ A)
  :
  angle (centroid A B C) (incenter A B C a b c) (orthocenter A B C) > 90 := by 
  sorry

end angle_GIH_gt_90_l764_764096


namespace cost_of_hard_lenses_l764_764579

theorem cost_of_hard_lenses (x H : ℕ) (h1 : x + (x + 5) = 11)
    (h2 : 150 * (x + 5) + H * x = 1455) : H = 85 := by
  sorry

end cost_of_hard_lenses_l764_764579


namespace ad_over_ab_l764_764736

theorem ad_over_ab (A B C D E : Type) [Triangle A B C] 
  (angle_A_eq_45 : angle A = 45)
  (angle_B_eq_60 : angle B = 60)
  (angle_ADE_eq_30 : ∠ ADE = 30)
  (area_divide : area (Triangle ADE) = area (Triangle ABC) / 2)
  (D_on_AB : D ∈ segment AB)
  (E_eq_C : E = C) : 
  ratio AD AB = (sqrt 6 + sqrt 2) / 4 :=
sorry

end ad_over_ab_l764_764736


namespace smallest_positive_integer_form_l764_764121

theorem smallest_positive_integer_form (m n : ℤ) :
  ∃ k : ℤ, k > 0 ∧ (∃ m n : ℤ, k = 2016 * m + 40404 * n) ∧ 
  ∀ j : ℤ, (j > 0 ∧ (∃ a b : ℤ, j = 2016 * a + 40404 * b)) → k ≤ j :=
begin
  sorry
end

end smallest_positive_integer_form_l764_764121


namespace angle_PSR_eq_24_l764_764746

/-- In triangle PQR, if ∠PRQ = ∠PQR, ∠QPR = 24°, and line segment QS bisects ∠PQR,
    then the measure of ∠PSR is 24°. -/
theorem angle_PSR_eq_24 
  (P Q R S : Type)
  (h1 : ∠ PRQ = ∠ PQR)
  (h2 : ∠ QPR = 24)
  (h3 : bisects (QS) ∠ PQR) :
  ∠ PSR = 24 :=
sorry

end angle_PSR_eq_24_l764_764746


namespace orthocenter_distance_eq_l764_764468

variables {c φ : ℝ} (ABC : Type) [triangle ABC]

-- Let AB be the length of side AB in triangle ABC
def side_length_AB (t : triangle ABC) : ℝ := c

-- Point M is on side AB such that ∠CMA = φ
def angle_CMA_eq_phi (t : triangle ABC) : Prop := ∃ M : ABC, ∠(C.1, M.1, A.1) = φ

-- Define distance between orthocenters of triangles AMC and BMC
def distance_between_orthocenters (t1 t2 : triangle ABC) : ℝ := 
  c * |Real.cot φ|

theorem orthocenter_distance_eq (t : triangle ABC) : 
  distance_between_orthocenters t t = c * |Real.cot φ| :=
sorry

end orthocenter_distance_eq_l764_764468


namespace fish_tank_problem_l764_764479

theorem fish_tank_problem
  (x : ℕ)
  (h1 : x / 3)
  (h2 : (x / 3) / 2)
  (h3 : (x / 3) / 2 = 5) : x = 30 :=
by
  sorry

end fish_tank_problem_l764_764479


namespace series_sum_l764_764394

def c_d_condition (c d : ℝ) : Prop := (c / d + c / d^2 + c / d^3 + ... = 7)

def series_value (c d : ℝ) : ℝ :=
  c / (2 * c + d) + c / (2 * c + d)^2 + c / (2 * c + d)^3 + ...

theorem series_sum (c d : ℝ) (h : c_d_condition c d) : series_value c d = 7 / 15 :=
sorry

end series_sum_l764_764394


namespace no_figure_with_two_axes_of_symmetry_without_central_symmetry_l764_764979

theorem no_figure_with_two_axes_of_symmetry_without_central_symmetry
  (F : Type) [finite_dimensional (euclidean_space ℝ F)]
  (l1 l2 : F → F)
  (h_symm_l1 : ∀ x, l1 (l1 x) = x)
  (h_symm_l2 : ∀ x, l2 (l2 x) = x)
  (h_axes : ∀ x, (l2 ∘ l1) x = (l1 ∘ l2) x)
  (h_two_axes : ∀ x, (l1 x = x ∧ l2 x = x) → x = 0)
  (hx_non_zero : ∃ x, x ≠ 0) :
  ∃ x, x ≠ 0 ∧ (l1 l2 x = x ∧ l2 l1 x = x) → ∃ c, ∀ x, (x + c = 0) :=
by
  sorry

end no_figure_with_two_axes_of_symmetry_without_central_symmetry_l764_764979


namespace simplify_powers_of_ten_l764_764118

theorem simplify_powers_of_ten :
  (10^0.4) * (10^0.5) * (10^0.2) * (10^(-0.6)) * (10^0.5) = 10 := 
by
  sorry

end simplify_powers_of_ten_l764_764118


namespace part_a_lineup_possible_part_b_lineup_not_necessary_l764_764162

-- Part (a)
theorem part_a_lineup_possible (n : ℕ) (h_n : n = 1000000) 
  (friend : fin n → fin n → Prop) (h_symm : ∀ {a b : fin n}, friend a b → friend b a) 
  (h_deg : ∀ (a : fin n), (finset.filter (friend a) (finset.univ)).card ≤ 2) 
  : ∃ lineup : fin n → fin n, ∀ {a b : fin n}, friend a b → abs (lineup a - lineup b) ≤ 2017 :=
sorry

-- Part (b)
theorem part_b_lineup_not_necessary (n : ℕ) (h_n : n = 1000000) 
  (friend : fin n → fin n → Prop) (h_symm : ∀ {a b : fin n}, friend a b → friend b a) 
  (h_deg : ∀ (a : fin n), (finset.filter (friend a) (finset.univ)).card ≤ 3) 
  : ¬ ∀ lineup : fin n → fin n, ∀ {a b : fin n}, friend a b → abs (lineup a - lineup b) ≤ 2017 
  :=
sorry

end part_a_lineup_possible_part_b_lineup_not_necessary_l764_764162


namespace find_x_l764_764236

noncomputable def x (x : ℝ) : Prop :=
  (⌈x⌉₊ * x = 210)

theorem find_x : ∃ x : ℝ, x = 14 ∧ x (14) :=
  by
    sorry

end find_x_l764_764236


namespace total_seats_l764_764361

-- Define the conditions
variable {S : ℝ} -- Total number of seats in the hall
variable {vacantSeats : ℝ} (h_vacant : vacantSeats = 240) -- Number of vacant seats
variable {filledPercentage : ℝ} (h_filled : filledPercentage = 0.60) -- Percentage of seats filled

-- Total seats in the hall
theorem total_seats (h : 0.40 * S = 240) : S = 600 :=
sorry

end total_seats_l764_764361


namespace trailing_zeroes_1000_factorial_l764_764083

/-- A function that defines the number of trailing zeroes in n!. -/
def count_trailing_zeroes (n : ℕ) : ℕ :=
  let count_factors (n k : ℕ) : ℕ := n / k in
  count_factors n 5 + count_factors n 25 + count_factors n 125 + count_factors n 625

theorem trailing_zeroes_1000_factorial : count_trailing_zeroes 1000 = 249 := by
  sorry

end trailing_zeroes_1000_factorial_l764_764083


namespace value_of_m_l764_764718

theorem value_of_m (x m : ℝ) (h : x ≠ 3) (H : (x / (x - 3) = 2 - m / (3 - x))) : m = 3 :=
sorry

end value_of_m_l764_764718


namespace neither_sufficient_nor_necessary_l764_764704

-- For given real numbers x and y
-- Prove the statement "at least one of x and y is greater than 1" is not necessary and not sufficient for x^2 + y^2 > 2.
noncomputable def at_least_one_gt_one (x y : ℝ) : Prop := (x > 1) ∨ (y > 1)
def sum_of_squares_gt_two (x y : ℝ) : Prop := x^2 + y^2 > 2

theorem neither_sufficient_nor_necessary (x y : ℝ) :
  ¬(at_least_one_gt_one x y → sum_of_squares_gt_two x y) ∧ ¬(sum_of_squares_gt_two x y → at_least_one_gt_one x y) :=
by
  sorry

end neither_sufficient_nor_necessary_l764_764704


namespace perpendicular_CO_BG_l764_764739

open EuclideanGeometry

variables {A B C E F H G O : Point}
variable [AcuteTriangle A B C]
variable (Γ₁ : Circle (Segment B A).circum)
variable (Γ₂ : Circle (Segment C A).circum)

-- Define the conditions
axiom intersect_ac (h1 : Γ₁.intersects AC) : Point = E
axiom intersect_ab (h2 : Γ₂.intersects AB) : Point = F
axiom intersect_be_cf (h3 : Line BE ∩ Line CF = Point) : Point = H
axiom intersect_ah_ef (h4 : Line AH ∩ Line EF = Point) : Point = G
axiom circumcenter_AEF (h5 : Circumcenter ∠A ∠E ∠F) : Point = O

theorem perpendicular_CO_BG :
  Perpendicular (Line Segment O C) (Line Segment B G) :=
sorry

end perpendicular_CO_BG_l764_764739


namespace least_possible_third_side_l764_764352

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) : ∃ c, c = 24 ∧ a^2 - c^2 = 527  :=
by
  use (√527)
  sorry

end least_possible_third_side_l764_764352


namespace heptagon_sum_of_squares_l764_764045

theorem heptagon_sum_of_squares (α : Real) :
  let vertices := λ k : ℕ, (Real.cos (α + (2 * Real.pi * k / 7)), Real.sin (α + (2 * Real.pi * k / 7)))
  let distances := λ k : ℕ, vertices k.2 * vertices k.2
  ∑ k in Finset.range 7, distances k = 7 / 2 := 
sorry

end heptagon_sum_of_squares_l764_764045


namespace minimum_value_of_f_l764_764698

open Real

def f (x : ℝ) : ℝ := x^2 - x + sqrt (2 * x^4 - 6 * x^2 + 8 * x + 16)

theorem minimum_value_of_f :
  ∃ x : ℝ, f(x) = 4 :=
sorry

end minimum_value_of_f_l764_764698


namespace geometric_sequence_problem_l764_764662

variable {a : ℕ → ℝ}

-- Condition: geometric sequence with common ratio q
def is_geometric_sequence (q : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = a n * q

-- Condition: Given fraction equality
axiom given_condition (q : ℝ) (h_geom : is_geometric_sequence q) : 
  (a 4 - a 0) / (a 2 - a 0) = 3

-- Proof statement we need to prove:
theorem geometric_sequence_problem (q : ℝ) (h_geom : is_geometric_sequence q) 
  (h_cond : given_condition q h_geom) : 
  (a 9 - a 1) / (a 5 + a 1) = 3 := 
sorry

end geometric_sequence_problem_l764_764662


namespace total_wet_surface_area_l764_764536

-- Necessary definitions based on conditions
def length : ℝ := 6
def width : ℝ := 4
def water_level : ℝ := 1.25

-- Defining the areas
def bottom_area : ℝ := length * width
def side_area (height : ℝ) (side_length : ℝ) : ℝ := height * side_length

-- Proof statement
theorem total_wet_surface_area :
  bottom_area + 2 * side_area water_level length + 2 * side_area water_level width = 49 := 
sorry

end total_wet_surface_area_l764_764536


namespace average_age_at_marriage_l764_764204

theorem average_age_at_marriage
  (A : ℕ)
  (combined_age_at_marriage : husband_age + wife_age = 2 * A)
  (combined_age_after_5_years : (A + 5) + (A + 5) + 1 = 57) :
  A = 23 := 
sorry

end average_age_at_marriage_l764_764204


namespace hyperbola_vertex_distance_to_asymptote_l764_764826

theorem hyperbola_vertex_distance_to_asymptote :
  let h : ℝ → ℝ := λ x => (x^2 / 4) - 1 in
  let vertex : (ℝ × ℝ) := (2, 0) in
  let asymptote : ℝ → ℝ := λ x => x / 2 in
  let distance : ℝ := (|vertex.1 + 2 * vertex.2|) / real.sqrt (1^2 + 2^2) in
  distance = (2 * real.sqrt 5) / 5 := by
  sorry

end hyperbola_vertex_distance_to_asymptote_l764_764826


namespace eval_expression_l764_764982

theorem eval_expression :
  2^0 + 9^5 / 9^3 = 82 :=
by
  have h1 : 2^0 = 1 := by sorry
  have h2 : 9^5 / 9^3 = 9^(5-3) := by sorry
  have h3 : 9^(5-3) = 9^2 := by sorry
  have h4 : 9^2 = 81 := by sorry
  sorry

end eval_expression_l764_764982


namespace relation_between_incircle_radius_perimeter_area_l764_764535

theorem relation_between_incircle_radius_perimeter_area (r p S : ℝ) (h : S = (1 / 2) * r * p) : S = (1 / 2) * r * p :=
by {
  sorry
}

end relation_between_incircle_radius_perimeter_area_l764_764535


namespace black_squares_in_10th_row_l764_764893

/-- Defines the number of total squares T in the nth row based on the given conditions. -/
def total_squares (n : ℕ) : ℕ :=
match n with
| 0 => 0
| 1 => 1
| n+1 => total_squares n + 2^n

/-- Theorem that the number of black squares in the 10th row is 511. -/
theorem black_squares_in_10th_row : 
  let T10 := total_squares 10
  in (T10 - 1) / 2 = 511 := 
by {
  let T10 := total_squares 10;
  sorry
}

end black_squares_in_10th_row_l764_764893


namespace max_value_abs_diff_l764_764450

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 - x + 1

theorem max_value_abs_diff :
  (x0 : ℝ) (m : ℝ) 
  (hx0_max : ∀ x, f x ≤ f x0) 
  (hx0_ne : m ≠ x0) 
  (hf_eq : f x0 = f m) :
  |m - x0| = 2 * sqrt 3 := 
begin
  sorry
end

end max_value_abs_diff_l764_764450


namespace find_log2_cos_x_l764_764288

-- Given conditions
variables (x a : ℝ)
def b := 2
noncomputable def sin_x := sin x
noncomputable def cos_x := cos x
noncomputable def log2_tan_x := log2 (tan x)

-- Ensure that the conditions hold
axiom sin_pos : sin_x > 0
axiom cos_pos : cos_x > 0
axiom log2_tan_x_eq_a : log2_tan_x = a

-- The target statement to be proved
theorem find_log2_cos_x 
    (sin_pos : sin x > 0)
    (cos_pos : cos x > 0)
    (h : log2 (tan x) = a) : log2 (cos x) = - (1 / 2) * log2 (2 ^ (2 * a) + 1) :=
sorry

end find_log2_cos_x_l764_764288


namespace phantom_needs_more_money_l764_764420

variable (given_money black_ink_price red_ink_price yellow_ink_price total_black_inks total_red_inks total_yellow_inks : ℕ)

def total_cost (total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price : ℕ) : ℕ :=
  total_black_inks * black_ink_price + total_red_inks * red_ink_price + total_yellow_inks * yellow_ink_price

theorem phantom_needs_more_money
  (h_given : given_money = 50)
  (h_black : black_ink_price = 11)
  (h_red : red_ink_price = 15)
  (h_yellow : yellow_ink_price = 13)
  (h_total_black : total_black_inks = 2)
  (h_total_red : total_red_inks = 3)
  (h_total_yellow : total_yellow_inks = 2) :
  given_money < total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price →
  total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price - given_money = 43 := by
  sorry

end phantom_needs_more_money_l764_764420


namespace parabolas_symmetric_about_y_axis_l764_764440

theorem parabolas_symmetric_about_y_axis :
  let f₁ := λ x : ℝ, 2 * x^2
  let f₂ := λ x : ℝ, -2 * x^2
  let f₃ := λ x : ℝ, x^2
  symmetric_about_y_axis (f₁ : ℝ → ℝ) ∧ symmetric_about_y_axis (f₂ : ℝ → ℝ) ∧ symmetric_about_y_axis (f₃ : ℝ → ℝ) :=
begin
  sorry
end

end parabolas_symmetric_about_y_axis_l764_764440


namespace total_charge_for_trip_l764_764382

-- Conditions
def initial_fee : ℝ := 2.25
def additional_charge_per_increment : ℝ := 0.15
def increment_distance : ℝ := 2 / 5
def trip_distance : ℝ := 3.6

-- Assertion to prove
theorem total_charge_for_trip : 
  (initial_fee + (trip_distance / increment_distance) * additional_charge_per_increment) = 3.60 := by 
  sorry

end total_charge_for_trip_l764_764382


namespace sequence_geometric_progression_sequence_sum_l764_764431

noncomputable def sequence_an : ℕ → ℕ
| 0 => 0 -- since we typically start from 0 in programming, but a_1 = 1 in the math problem
| n + 1 => 2 ^ (n + 1) - 1

def sum_sequence (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, sequence_an (i + 1))

theorem sequence_geometric_progression :
  ∀ n : ℕ, sequence_an (n + 1) = 2 * sequence_an n + 1 := 
by
  sorry

theorem sequence_sum (n : ℕ) :
  sum_sequence n = 2^(n + 1) - 2 - n :=
by
  sorry

end sequence_geometric_progression_sequence_sum_l764_764431


namespace intersect_parallelepiped_l764_764669

-- Defining the given points
variables {A A' B C D B' C' D' : Type}
variables (M N P : Type)

-- Setting up the conditions from the problem
def M_on_AA' : Prop := M ∈ [A, A']
def N_on_BC : Prop := N ∈ [B, C]
def P_on_C'D' : Prop := P ∈ [C', D']

-- Claiming the vertices of the hexagon
def vertices_of_hexagon (K R T : Type) : Prop :=
  hexagon_vertices = [M, K, P, R, N, T]

-- The formal proof statement in Lean
theorem intersect_parallelepiped (K R T : Type) (hexagon_vertices : List Type)
  (hM : M_on_AA' M) (hN : N_on_BC N) (hP : P_on_C'D' P) :
  vertices_of_hexagon M N P K R T :=
sorry

end intersect_parallelepiped_l764_764669


namespace eccentricity_of_hyperbola_l764_764311

-- Define the parameters of the hyperbola
variables (a c e : ℝ)
variable (h : a > 0)

-- Given conditions
def hyperbola := (x : ℝ) (y : ℝ) → (x^2 / a^2) - (y^2 / 9) = 1
def foci_distance := 2 * c = 10

-- Define eccentricity
def eccentricity := e = c / a

-- Main theorem stating the eccentricity of the hyperbola
theorem eccentricity_of_hyperbola (h1 : 2 * c = 10) (h2 : a^2 = c^2 - 9) : e = 5 / 4 :=
by
  have h3 : c = 5, by linarith
  have h4 : a^2 = 16, by rw [h3, pow_two, sub_eq_iff_eq_add, add_assoc]; simp
  have h5 : a = 4, by linarith
  have h6 : e = 5 / 4, by simp [eccentricity, h3, h5, div]
  exact h6

end eccentricity_of_hyperbola_l764_764311


namespace solve_proof_l764_764877

noncomputable def solve_t (t : ℝ) : Prop :=
  5.41 * tan t = (sin t * sin t + sin (2 * t) - 1) / (cos t * cos t - sin (2 * t) + 1)

noncomputable def solve_z (z : ℝ) : Prop :=
  5.421 + sin z + cos z + sin (2 * z) + cos (2 * z) = 0

noncomputable def proof (t : ℝ) : Prop :=
  solve_t t → ∃ (k : ℤ), 
  t = (↑k * Real.pi) + Real.pi / 4 ∨ 
  t = Real.arctan ((1 + Real.sqrt 5) / 2) + (↑k * Real.pi) ∨ 
  t = Real.arctan ((1 - Real.sqrt 5) / 2) + (↑k * Real.pi)

theorem solve_proof (t : ℝ) (z : ℝ) (h1 : solve_t t) (h2 : solve_z z) : proof t :=
  sorry

end solve_proof_l764_764877


namespace original_price_l764_764912

theorem original_price (P : ℝ) (h₁ : P - 0.30 * P = 0.70 * P) (h₂ : P - 0.20 * P = 0.80 * P) (h₃ : 0.70 * P + 0.80 * P = 50) :
  P = 100 / 3 :=
by
  -- Proof skipped
  sorry

end original_price_l764_764912


namespace train_length_l764_764926

/-- The length of a train given its speed and the time it takes to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_result : ℝ) : 
  speed_kmph = 144 → 
  time_sec = 2.7497800175985923 → 
  length_result = 110.9912007039437 →
  let speed_mps := speed_kmph * (5 / 18) in
  let length_train := speed_mps * time_sec in
  length_train = length_result :=
by
  intros h_speed h_time h_result
  simp [h_speed, h_time, h_result]
  sorry

end train_length_l764_764926


namespace B_and_C_have_together_l764_764563

theorem B_and_C_have_together
  (A B C : ℕ)
  (h1 : A + B + C = 700)
  (h2 : A + C = 300)
  (h3 : C = 200) :
  B + C = 600 := by
  sorry

end B_and_C_have_together_l764_764563


namespace unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l764_764480

-- Definitions of points and lines
structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (a : α) (b : α) -- Represented as ax + by = 0

-- Given conditions
variables {α : Type*} [Field α]
variables (P Q : Point α)
variables (L1 L2 : Line α) -- L1 and L2 are perpendicular

-- Proof problem statement
theorem unique_ellipse_through_points_with_perpendicular_axes (P Q : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
(P ≠ Q) → 
∃! (E : Set (Point α)), -- E represents the ellipse as a set of points
(∀ (p : Point α), p ∈ E → (p = P ∨ p = Q)) ∧ -- E passes through P and Q
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

theorem infinite_ellipses_when_points_coincide (P : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
∃ (E : Set (Point α)), -- E represents an ellipse
(∀ (p : Point α), p ∈ E → p = P) ∧ -- E passes through P
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

end unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l764_764480


namespace composite_product_ratio_l764_764978

def composites_gt_10 : list ℕ := [12, 14, 15, 16, 18]
def composites_gt_20 : list ℕ := [21, 22, 24, 25, 26]

def product (l : list ℕ) : ℕ := l.foldl (*) 1

theorem composite_product_ratio :
  (product composites_gt_10 : ℚ) / (product composites_gt_20 : ℚ) = 72 / 715 :=
by
  sorry

end composite_product_ratio_l764_764978


namespace correct_option_c_l764_764507

theorem correct_option_c (b : ℝ) : 3 * b^3 * 2 * b^2 = 6 * b^5 := by
  calc
    3 * b^3 * 2 * b^2 = (3 * 2) * (b^3 * b^2) : by ring
                   ... = 6 * b^(3 + 2)     : by rw [mul_pow]
                   ... = 6 * b^5           : by ring

end correct_option_c_l764_764507


namespace reciprocal_inequality_l764_764332

variable (a b : ℝ)

theorem reciprocal_inequality (ha : a < 0) (hb : b > 0) : (1 / a) < (1 / b) := sorry

end reciprocal_inequality_l764_764332


namespace count_right_triangles_with_conditions_l764_764973

theorem count_right_triangles_with_conditions :
  ∃ n : ℕ, n = 10 ∧
    (∀ (a b : ℕ),
      (a ^ 2 + b ^ 2 = (b + 2) ^ 2) →
      (b < 100) →
      (∃ k : ℕ, a = 2 * k ∧ k ^ 2 = b + 1) →
      n = 10) :=
by
  -- The proof goes here
  sorry

end count_right_triangles_with_conditions_l764_764973


namespace phantom_needs_more_money_l764_764419

variable (given_money black_ink_price red_ink_price yellow_ink_price total_black_inks total_red_inks total_yellow_inks : ℕ)

def total_cost (total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price : ℕ) : ℕ :=
  total_black_inks * black_ink_price + total_red_inks * red_ink_price + total_yellow_inks * yellow_ink_price

theorem phantom_needs_more_money
  (h_given : given_money = 50)
  (h_black : black_ink_price = 11)
  (h_red : red_ink_price = 15)
  (h_yellow : yellow_ink_price = 13)
  (h_total_black : total_black_inks = 2)
  (h_total_red : total_red_inks = 3)
  (h_total_yellow : total_yellow_inks = 2) :
  given_money < total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price →
  total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price - given_money = 43 := by
  sorry

end phantom_needs_more_money_l764_764419


namespace problem_1_problem_2_l764_764193

noncomputable def equation_of_ellipse : Prop :=
  ∀ (x y : ℝ), (ellipse_center_origin x y) → 
    ((left_focus_coincides_with_parabola x y) ∧ 
    (line_l_intersects_ellipse x y) ∧
    (line_l_intersects_parabola x y) ∧ 
    (line_perpendicular_x_axis x y)) → 
    (x^2 / 2 + y^2 = 1)

noncomputable def equation_of_circle : Prop :=
  ∀ (x y : ℝ), (circle_passing_through_O_F1_tangent_to_latus_rectum x y) → 
    ((point_O x y) ∧ 
    (point_F1 x y) ∧ 
    (tangent_to_left_latus_rectum x y)) → 
    ((x + 1/2)^2 + (y - sqrt 2)^2 = 9/4)

noncomputable def max_dot_product : ℝ :=
  ∃ (A B F2 : ℝ × ℝ), 
    (line_l_intersects_ellipse (A.fst) (A.snd)) ∧ 
    (line_l_intersects_ellipse (B.fst) (B.snd)) ∧ 
    (F2.snd = -sqrt 2 ∨ F2.snd = sqrt 2) ∧ 
    (F2.fst = 1) ∧ 
    (dot_product F2 A B = 7/2)

axiom ellipse_center_origin : ∀ (x y : ℝ), Prop
axiom left_focus_coincides_with_parabola : ∀ (x y : ℝ), Prop
axiom line_l_intersects_ellipse : ∀ (x y : ℝ), Prop
axiom line_l_intersects_parabola : ∀ (x y : ℝ), Prop
axiom line_perpendicular_x_axis : ∀ (x y : ℝ), Prop
axiom circle_passing_through_O_F1_tangent_to_latus_rectum : ∀ (x y : ℝ), Prop
axiom point_O : ∀ (x y : ℝ), Prop
axiom point_F1 : ∀ (x y : ℝ), Prop
axiom tangent_to_left_latus_rectum : ∀ (x y : ℝ), Prop
axiom dot_product : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ

-- Statements of the problems
theorem problem_1 : equation_of_ellipse := by sorry
theorem problem_2 : equation_of_circle := by sorry
noncomputable def problem_3 := max_dot_product 

end problem_1_problem_2_l764_764193


namespace number_that_solves_equation_l764_764085

theorem number_that_solves_equation :
  let x := (1/3 - 1/2)^(-1)
  in x = -6 :=
by
  sorry

end number_that_solves_equation_l764_764085


namespace xiao_ming_correct_answers_l764_764129

theorem xiao_ming_correct_answers :
  let prob1 := (-2 - 2) = 0
  let prob2 := (-2 - (-2)) = -4
  let prob3 := (-3 + 5 - 6) = -4
  (if prob1 then 1 else 0) + (if prob2 then 1 else 0) + (if prob3 then 1 else 0) = 1 :=
by
  sorry

end xiao_ming_correct_answers_l764_764129


namespace number_of_pits_less_than_22222_l764_764543

def is_pit (n : ℕ) : Prop :=
  let digits := n.digits 10;
  digits.length = 5 ∧
  digits.take 3 = (digits.take 3).sort (· > ·) ∧
  digits.drop 2 = (digits.drop 2).sort (· < ·)

def is_less_than_22222 (n : ℕ) : Prop := n < 22222

theorem number_of_pits_less_than_22222 : { n // is_pit n ∧ is_less_than_22222 n }.card = 36 :=
by 
  sorry

end number_of_pits_less_than_22222_l764_764543


namespace nominal_rate_of_interest_correct_l764_764827

noncomputable def nominal_rate_of_interest (EAR : ℝ) (n : ℕ) : ℝ :=
  let i := by 
    sorry
  i

theorem nominal_rate_of_interest_correct :
  nominal_rate_of_interest 0.0609 2 = 0.0598 :=
by 
  sorry

end nominal_rate_of_interest_correct_l764_764827


namespace g_88_value_l764_764223

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n m : ℕ) (h : n < m) : g n < g m
axiom g_multiplicative (m n : ℕ) : g (m * n) = g m * g n
axiom g_exponential_condition (m n : ℕ) (h : m ≠ n ∧ m ^ n = n ^ m) : g m = n ∨ g n = m

theorem g_88_value : g 88 = 7744 :=
sorry

end g_88_value_l764_764223


namespace function_properties_l764_764224

-- Conditions: Define the function f(x) = |cos(2x)|
def f (x : ℝ) : ℝ := abs (cos (2 * x))

-- Problem: Prove that f(x) is even and has a period of π/2
theorem function_properties :
  (∀ x, f (-x) = f x) ∧ (∀ x, f (x + π / 2) = f x) := by
  sorry

end function_properties_l764_764224


namespace correct_statements_l764_764262

variables (p q : Prop)

lemma problem_condition_p : 1 ∈ {1, 2} := by simp

lemma problem_condition_q : ¬ ({1} ∈ {1, 2}) := 
begin
  simp,
  intro h,
  cases h,
  simp at h,
  contradiction,
end

theorem correct_statements :
  (p = (1 ∈ {1, 2}) ∧ q = (\{1\} ∈ {1, 2})) →
  (¬ (p ∧ q) ∧ (p ∨ q) ∧ ¬ ¬ p) :=
by sorry

end correct_statements_l764_764262


namespace min_value_of_expression_l764_764648

noncomputable def find_min_value : ℝ := -4

theorem min_value_of_expression (a : ℝ) (x1 x2 x3 : ℝ) (ha_pos : 0 < a)
  (h1 : x1 + x2 + x3 = a )
  (h2 : x1 * x2 + x2 * x3 + x3 * x1 = a)
  (h3 : x1 * x2 * x3 = a) : find_min_value = x1^3 + x2^3 + x3^3 - 3 * x1 * x2 * x3 :=
begin
  sorry
end

end min_value_of_expression_l764_764648


namespace sufficient_but_not_necessary_l764_764713

variable (a : ℝ)

theorem sufficient_but_not_necessary : (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 → False) :=
by
  sorry

end sufficient_but_not_necessary_l764_764713


namespace hall_length_l764_764905

theorem hall_length 
    (breadth_hall_m : ℕ) (stone_length_dm stone_breadth_dm : ℕ) (num_stones : ℕ)
    (hall_breadth := breadth_hall_m * 10)  -- convert breadth to dm
    (stone_area := stone_length_dm * stone_breadth_dm)
    (total_stone_area := num_stones * stone_area)
    (hall_area := hall_breadth * L) : hall_breadth = 150 → stone_length_dm = 3 → stone_breadth_dm = 5 → num_stones = 3600 → L = 360 →
    breadth_hall_m = 15 → 
    (hall_length_m : ℕ) (L := hall_length_m * 10)  -- convert length back to meters
    : hall_length_m = 36 :=
sorry

end hall_length_l764_764905


namespace area_of_sector_is_10_l764_764515

noncomputable def radius : ℝ := 5
noncomputable def arc_length : ℝ := 4

theorem area_of_sector_is_10 :
  (arc_length / (2 * real.pi * radius)) * (real.pi * radius ^ 2) = 10 := by
  sorry

end area_of_sector_is_10_l764_764515


namespace triangle_segment_length_l764_764841

theorem triangle_segment_length (a b c : ℝ) (h : c = 51 ∧ (a = 24 ∧ b = 45)) :
  ∃ x y : ℝ, (a^2 = x^2 + y^2) ∧ (b^2 = (c-x)^2 + y^2) ∧ (c - x = 40) :=
by {
  sorry,
}

end triangle_segment_length_l764_764841


namespace cuberoot_sqrt_inverse_zero_power_l764_764209

theorem cuberoot_sqrt_inverse_zero_power :
  (3:ℝ) - 2 - 3 + 1 = -1 := by
  have h1: (∛(27:ℝ)) = 3 := by sorry 
  have h2: (sqrt(4:ℝ)) = 2 := by sorry
  have h3: ((1/3:ℝ)^(-1:ℝ)) = 3 := by sorry
  have h4: ((-2020:ℝ)^(0:ℝ)) = 1 := by sorry
  rw [h1, h2, h3, h4]
  norm_num

end cuberoot_sqrt_inverse_zero_power_l764_764209


namespace area_of_square_l764_764039

-- Definitions used in Lean 4 statement coming directly from the conditions in a)
def is_square (A B C D : Point) : Prop := 
  A.x = D.x ∧ A.y = B.y ∧
  B.x = C.x ∧ C.y = D.y ∧
  (B.x - A.x) = (B.y - D.y)

def aligned_segment (P Q : Point) (d : ℝ) : Prop := 
  P.y = Q.y ∧ |Q.x - P.x| = d ∨ P.x = Q.x ∧ |Q.y - P.y| = d

-- The theorem to prove
theorem area_of_square (A B C D E F : Point) (x : ℝ) (h_square : is_square A B C D)
  (h1 : E ∈ segment A D) (h2 : F ∈ segment B C)
  (h_lengths : aligned_segment B E 40 ∧ aligned_segment E F 40 ∧ aligned_segment F D 40) :
  x^2 = 2880 :=
by
  sorry

end area_of_square_l764_764039


namespace balls_in_boxes_l764_764710

theorem balls_in_boxes : 
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  total_ways - exclude_one_empty = 537 := 
by
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  have h : total_ways - exclude_one_empty = 537 := sorry
  exact h

end balls_in_boxes_l764_764710


namespace find_interval_of_m_l764_764603

noncomputable def sequence (n : ℕ) : ℝ :=
nat.rec_on n 7 (λ n x_n, (x_n^2 + 3 * x_n + 2) / (x_n + 4))

def limit_value := 3 + (1 / 2^15 : ℝ)

def least_m (P : ℕ → Prop) : ℕ := nat.find (exists_nat_of_not_forall_not P)

theorem find_interval_of_m :
  let m := least_m (λ n, sequence n ≤ limit_value) in
  19 ≤ m ∧ m ≤ 54 :=
by
  sorry

end find_interval_of_m_l764_764603


namespace stable_teams_min_max_l764_764434

-- Define the main theorem that states the minimum and maximum number of stable teams
theorem stable_teams_min_max 
   (teams : Fin 10 → Fin 10 → Fin 10)  -- function defining the ranking in two tournaments
   (stable : (Fin 10 → Fin 10 → Fin 10 → Prop)) : 
   (∃ t1 t2, (stable teams t1) = 9) ∧ (∃ t1 t2, (stable teams t2) = 0) := 
sorry

end stable_teams_min_max_l764_764434


namespace circles_intersect_l764_764606

noncomputable def circle_equation (a b r : ℝ) (x y : ℝ) := (x - a)^2 + (y - b)^2 = r^2

theorem circles_intersect :
  (∀ x y : ℝ, circle_equation 1 0 2 x y ↔ x^2 + y^2 - 2 * x - 3 = 0) →
  (∀ x y : ℝ, circle_equation 2 (-1) 1 x y ↔ x^2 + y^2 - 4 * x + 2 * y + 4 = 0) →
  ∃ x y : ℝ, (circle_equation 1 0 2 x y) ∧ (circle_equation 2 (-1) 1 x y) :=
begin
  intros h1 h2,
  sorry -- Proof is omitted as requested.
end

end circles_intersect_l764_764606


namespace equal_tuesdays_fridays_l764_764909

/-!
Given a month of 30 days, prove that there are four possible starting days of the week
such that the number of Tuesdays and Fridays are equal.
-/

def days_in_month := 30
def days_in_week := 7

/-- A proof that for a 30-day month, the number of possible starting days of the week
    that result in an equal number of Tuesdays and Fridays is 4. -/
theorem equal_tuesdays_fridays : 
  ∃ (count : ℕ), count = 4 ∧
  count = (set.to_finset {x ∈ finset.range days_in_week | 
    let extra_days := (days_in_month % days_in_week),
    let num_tuesdays := (days_in_month // days_in_week) + ite (x <= 1) 1 0,
    let num_fridays := (days_in_month // days_in_week) + ite (x = 0 ∨ x = 3) 1 0,
    num_tuesdays = num_fridays }).card :=
by
  sorry

end equal_tuesdays_fridays_l764_764909


namespace proof_of_total_cost_l764_764789

noncomputable theory

-- Definitions
def cost_mango_A := 10
def cost_rice_B := 8
def kg_flour_A := 21
def conversion_rate := 2
def discount := 0.10

-- Price calculations
def cost_mango_4_A := 4 * cost_mango_A
def cost_rice_3_A := (3 * cost_rice_B) / conversion_rate
def cost_flour_5_A := 5 * kg_flour_A

-- Total cost calculation before discount
def total_cost_A := cost_mango_4_A + cost_rice_3_A + cost_flour_5_A

-- Total cost after discount
def total_cost_after_discount := total_cost_A - (discount * total_cost_A)

-- Target value
def expected_total_cost_A := 141.30

-- The theorem to prove
theorem proof_of_total_cost:
  total_cost_after_discount = expected_total_cost_A := sorry

end proof_of_total_cost_l764_764789


namespace bx_lt_ab_l764_764001

theorem bx_lt_ab (A B C X : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space X]
  (triangle_ABC : triangle A B C) (X_on_AC : lies_on_segment X A C) (angle_ACB_obtuse : angle C > 90) :
  distance B X < distance A B := 
sorry

end bx_lt_ab_l764_764001


namespace cindy_envelope_problem_l764_764211

theorem cindy_envelope_problem :
  ∀ (Cindy_envelopes : ℕ) (friends: ℕ) (envelopes_per_friend : ℕ),
    Cindy_envelopes = 74 →
    friends = 11 →
    envelopes_per_friend = 6 →
    Cindy_envelopes - friends * envelopes_per_friend = 8 :=
by
  intros Cindy_envelopes friends envelopes_per_friend h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end cindy_envelope_problem_l764_764211


namespace simplify_product_l764_764057

noncomputable def product (n : ℕ) : ℚ :=
  ∏ k in Finset.range n, (3 * k + 6) / (3 * k)

theorem simplify_product : product 997 = 1001 := 
sorry

end simplify_product_l764_764057


namespace investment_sum_l764_764885

theorem investment_sum (P : ℝ) 
  (SI1 : P * (4 / 100) * 7 = SI1) 
  (SI2 : P * (4.5 / 100) * 7 = SI2) 
  (difference : SI2 - SI1 = 31.50) :
  P = 900 :=
by
  sorry

end investment_sum_l764_764885


namespace problem1_l764_764891

theorem problem1 :
  0.027^(-1/3) + (Real.sqrt 8)^(4/3) - 3^(-1) + (Real.sqrt 2 - 1)^0 = 8 := by
  sorry

end problem1_l764_764891


namespace angle_between_plane_and_base_l764_764158

variable (α k : ℝ)
variable (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
variable (h_ratio : ∀ A D S : ℝ, AD / DS = k)

theorem angle_between_plane_and_base (α k : ℝ) 
  (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (h_ratio : ∀ A D S : ℝ, AD / DS = k) 
  : ∃ γ : ℝ, γ = Real.arctan (k / (k + 3) * Real.tan α) :=
by
  sorry

end angle_between_plane_and_base_l764_764158


namespace least_amount_to_add_l764_764883

theorem least_amount_to_add (current_amount : ℕ) (n : ℕ) (divisor : ℕ) [NeZero divisor]
  (current_amount_eq : current_amount = 449774) (n_eq : n = 1) (divisor_eq : divisor = 6) :
  ∃ k : ℕ, (current_amount + k) % divisor = 0 ∧ k = n := by
  sorry

end least_amount_to_add_l764_764883


namespace calculate_polygon_perimeter_l764_764208

theorem calculate_polygon_perimeter (n s : ℕ) (h_n : n = 10) (h_s : s = 3) : 
    n * s = 30 :=
by
  rw [h_n, h_s]
  norm_num

end calculate_polygon_perimeter_l764_764208


namespace angle_measure_l764_764940

theorem angle_measure (x : ℝ) 
  (h : x = 2 * (90 - x) - 60) : 
  x = 40 := 
  sorry

end angle_measure_l764_764940


namespace difference_in_profit_percentage_l764_764190

-- Constants
def selling_price1 : ℝ := 350
def selling_price2 : ℝ := 340
def cost_price : ℝ := 200

-- Profit calculations
def profit1 : ℝ := selling_price1 - cost_price
def profit2 : ℝ := selling_price2 - cost_price

-- Profit percentage calculations
def profit_percentage1 : ℝ := (profit1 / cost_price) * 100
def profit_percentage2 : ℝ := (profit2 / cost_price) * 100

-- Statement of the problem: Difference in profit percentage
theorem difference_in_profit_percentage : profit_percentage1 - profit_percentage2 = 5 := by
  sorry

end difference_in_profit_percentage_l764_764190


namespace total_spent_is_correct_l764_764864

def cost_orange_juice : ℕ := 70
def cost_apple_juice : ℕ := 60
def total_bottles : ℕ := 70
def bottles_orange_juice : ℕ := 42

-- Assuming 100 cents in a dollar
def total_amount_spent_dollars : ℕ := 46.20

theorem total_spent_is_correct:
    (bottles_orange_juice * cost_orange_juice +
     (total_bottles - bottles_orange_juice) * cost_apple_juice) / 100 = total_amount_spent_dollars :=
by
    sorry

end total_spent_is_correct_l764_764864


namespace volume_of_polyhedron_l764_764665

theorem volume_of_polyhedron
  (a b c h : ℝ) :
  (Volume := 1/6 * b * h * (2 * a + c)) :=
sorry

end volume_of_polyhedron_l764_764665


namespace inscribed_triangle_perimeter_l764_764402

theorem inscribed_triangle_perimeter 
  (O P R S A B : Type) 
  [circle_center : center O] 
  (tangents : tangents P R S A B)
  (vertices_diff : vertices_on_diff_sides A B P R S):
  ∀ (A' B' : Type), inscribed_triangle A' B' A B O → 
  perimeter A' B' A B O ≥ segment_length P R :=
sorry

end inscribed_triangle_perimeter_l764_764402


namespace range_of_a_unique_positive_zero_l764_764309

theorem range_of_a_unique_positive_zero (a : ℝ) :
  (∃! x : ℝ, (λ x, a * x^3 + 3 * x^2 + 1) x = 0 ∧ x > 0) ↔ a < 0 :=
by
  sorry

end range_of_a_unique_positive_zero_l764_764309


namespace ratio_BD_BO_l764_764040

noncomputable theory
open_locale classical

-- Definitions of the given conditions
universe u
variables {V : Type u} [normed_group V] [normed_space ℝ V] [inner_product_space ℝ V]

variables (A B C O D : V)
variables (r : ℝ)
variables (h_circle : ∀ (P : V), has_norm.norm (P - O) = r)
variables (triangle_equilateral : dist A B = dist B C ∧ dist B C = dist C A)
variables (tangent_BA : ∀ (P : V), ∃ t : V, is_tangent_line_at P t O A)
variables (tangent_BC : ∀ (P : V), ∃ t : V, is_tangent_line_at P t O C)
variables (circle_intersects_BO : D ∈ range (λ t, O + t • (B - O)))

-- The statement we need to prove
theorem ratio_BD_BO : dist B D / dist B O = 1 / 2 :=
by {
  sorry -- The proof is omitted
}

end ratio_BD_BO_l764_764040


namespace distance_from_point_to_line_le_two_cm_theorem_l764_764270

noncomputable def distance_from_point_to_line_le_two_cm
  (P : Type) (l : set P) [metric_space P] (Q : P) : Prop :=
  ∀ {d : ℝ}, P ∉ l → dist P Q = 2 → ∃ x ∈ l, dist P x ≤ 2

-- Statement of the theorem
theorem distance_from_point_to_line_le_two_cm_theorem 
  (P : Type) (l : set P) [metric_space P] (Q : P)
  (h1 : P ∉ l) (h2 : dist P Q = 2) : 
  distance_from_point_to_line_le_two_cm P l Q :=
begin
  sorry
end

end distance_from_point_to_line_le_two_cm_theorem_l764_764270


namespace problem_solution_l764_764634

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1)^2

theorem problem_solution :
  (∀ x : ℝ, (0 < x ∧ x ≤ 5) → x ≤ f x ∧ f x ≤ 2 * |x - 1| + 1) →
  (f 1 = 4 * (1 / 4) + 1) →
  (∃ (t m : ℝ), m > 1 ∧ 
               (∀ x : ℝ, (1 ≤ x ∧ x ≤ m) → f t ≤ (1 / 4) * (x + t + 1)^2)) →
  (1 / 4 = 1 / 4) ∧ (m = 2) :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l764_764634


namespace log2_ratio_is_integer_probability_l764_764110

open Finset

theorem log2_ratio_is_integer_probability:
  (∃ a b ∈ {x : ℕ | ∃ k, x = 2^k ∧ k ≤ 15}.toFinset, 
    a ≠ b ∧ ∃ z : ℕ, ∃ x y : ℕ, a = 2^x ∧ b = 2^y ∧ y = x * z)
  → (7 / 30) :=
by
  -- Definitions of the problem conditions and correct answer.
  sorry -- The detailed proof would be placed here, but it is not required for now.

end log2_ratio_is_integer_probability_l764_764110


namespace fixed_point_log_func_l764_764452

noncomputable def log_func (a : ℝ) (x : ℝ) : ℝ :=
  if a > 0 ∧ a ≠ 1 then real.log (x - 1) / real.log a + 2 else 0

theorem fixed_point_log_func (a : ℝ) (h : 0 < a ∧ a ≠ 1) : log_func a 2 = 2 :=
by
  unfold log_func
  rw [if_pos h]
  simp [real.log_one]
  exact h
  sorry

end fixed_point_log_func_l764_764452


namespace simplify_fraction_sequence_l764_764060

theorem simplify_fraction_sequence :
  (∏ k in finset.range 1000, (3 * (k + 3) + 3) / (3 * (k + 3)))
  = 1001 :=
by
suffices : (∏ k in finset.range 1000, (3 * (k + 3) + 3) / (3 * (k + 3))) = 3003 / 3,
{ rwa [div_eq_mul_inv, mul_inv_cancel (3 : ℤ), inv_one, mul_one] },
sorry

end simplify_fraction_sequence_l764_764060


namespace number_of_incorrect_statements_l764_764461

-- Define the propositions and their relations.
def prop1_contrapositive (b a c: ℝ) : Prop := (b^2 - 4 * a * c <= 0) → (ax^2 + bx + c ≠ 0)
def cond2 : Prop := ¬(x = 2) → (x^2 - 3x + 2 = 0)
def cond3_negation_correct: Prop := (xy ≠ 0) → ((x ≠ 0) ∧ (y ≠ 0))
def cond4_p_negation (p : ℝ → Prop) : Prop := (∀ x, ¬(x^2 + x + 1 < 0)) → (¬p)
def cond5_evaluation (p q : Prop) : Prop := (¬p → p) → ((¬q → q) → (¬p ∧ q) ∧ (p ∨ ¬q))

-- The main theorem to state that there are two incorrect statements
theorem number_of_incorrect_statements : 
  ∀ (b a c x y : ℝ) (p q : Prop), 
    prop1_contrapositive b a c ∧ cond2 ∧ cond3_negation_correct ∧ (¬cond4_p_negation p) ∧ (¬cond5_evaluation p q) → 
    ∃ (n = 2), n = 2 := 
 sorry

end number_of_incorrect_statements_l764_764461


namespace weight_of_stationary_object_l764_764922

noncomputable def weight_of_object : ℝ :=
  let F1 := (1 : ℝ)
  let F2 := (2 : ℝ)
  let F3 := (3 : ℝ)
  let dot_product (F1 F2 F3 : ℝ) (angle : ℝ) : ℝ := F1 * F2 * Real.cos angle
  let angle := Real.pi / 3 -- 60 degrees in radians
  let F1_dot_F2 := dot_product F1 F2 angle
  let F2_dot_F3 := dot_product F2 F3 angle
  let F3_dot_F1 := dot_product F3 F1 angle
  Real.sqrt (F1^2 + F2^2 + F3^2 + 2 * (F1_dot_F2 + F2_dot_F3 + F3_dot_F1))

theorem weight_of_stationary_object : weight_of_object = 5 := by
  sorry

end weight_of_stationary_object_l764_764922


namespace binom_sum_alternating_l764_764331

variable {n : ℕ}
variable (a : ℕ → ℕ)

-- Conditions
axiom C23_eq : (nat.choose 23 (3*n + 1) = nat.choose 23 (n + 6))
axiom binom_expansion : (3 - x)^n = ∑ i in finset.range (n+1), (a i) * x^i

-- Theorem Statement
theorem binom_sum_alternating (h : n > 0) : 
  (∑ i in finset.range (n+1), (-1)^i * a i) = 256 :=
sorry

end binom_sum_alternating_l764_764331


namespace graph_quadrant_exclusion_l764_764259

theorem graph_quadrant_exclusion (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ∀ x : ℝ, ¬ ((a^x + b > 0) ∧ (x > 0)) :=
by
  sorry

end graph_quadrant_exclusion_l764_764259


namespace blue_paint_quantity_l764_764863

-- Conditions
def paint_ratio (r b y w : ℕ) : Prop := r = 2 * w / 4 ∧ b = 3 * w / 4 ∧ y = 1 * w / 4 ∧ w = 4 * (r + b + y + w) / 10

-- Given
def quart_white_paint : ℕ := 16

-- Prove that Victor should use 12 quarts of blue paint
theorem blue_paint_quantity (r b y w : ℕ) (h : paint_ratio r b y w) (hw : w = quart_white_paint) : 
  b = 12 := by
  sorry

end blue_paint_quantity_l764_764863


namespace find_theta_for_pure_imaginary_l764_764690

-- Definition of the problem conditions and the equivalent Lean 4 statement
def complex_is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def theta_values_for_pure_imaginary (θ : ℝ) : Prop :=
  θ = (3 * Real.pi) / 4 ∨ θ = (7 * Real.pi) / 4

theorem find_theta_for_pure_imaginary (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  let z := (Complex.cos θ + Complex.I) * (2 * Complex.sin θ - Complex.I) in
  complex_is_pure_imaginary z → theta_values_for_pure_imaginary θ :=
by
  sorry

end find_theta_for_pure_imaginary_l764_764690


namespace find_PQR_l764_764623

noncomputable def PQR : ℚ → ℚ → ℚ → Prop :=
λ P Q R, ∀ x, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
(x^2 + 2) / ((x - 1) * (x - 4) * (x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6)

theorem find_PQR :
  ∃ P Q R : ℚ, PQR (1 / 5) (-3) (19 / 5) (P Q R) :=
begin
  use [1 / 5, -3, 19 / 5],
  unfold PQR,
  intros x hx,
  simp [hx],
  field_simp [hx.1, hx.2.1, hx.2.2],
  sorry
end

end find_PQR_l764_764623


namespace cos_inequality_solution_set_l764_764092

theorem cos_inequality_solution_set (x : ℝ) : 
  (∃ k : ℤ, 2 * k * Real.pi + (2 * Real.pi / 3) ≤ x ∧ x ≤ 2 * k * Real.pi + (4 * Real.pi / 3)) ↔ 
  (cos x + 1 / 2 ≤ 0) :=
by
  sorry

end cos_inequality_solution_set_l764_764092


namespace vicky_divide_to_one_l764_764067

theorem vicky_divide_to_one : 
  ∀ n : ℕ, n = 256 → (nat.iter_div2_to_1 n).length = 8 :=
sorry

end vicky_divide_to_one_l764_764067


namespace correct_statements_l764_764036

theorem correct_statements :
  (20 / 100 * 40 = 8) ∧
  (2^3 = 8) ∧
  (7 - 3 * 2 ≠ 8) ∧
  (3^2 - 1^2 = 8) ∧
  (2 * (6 - 4)^2 = 8) :=
by
  sorry

end correct_statements_l764_764036


namespace one_fourth_of_8_point_8_is_fraction_l764_764999

theorem one_fourth_of_8_point_8_is_fraction:
  (1 / 4) * 8.8 = 11 / 5 :=
by sorry

end one_fourth_of_8_point_8_is_fraction_l764_764999


namespace total_wheels_in_garage_l764_764477

theorem total_wheels_in_garage :
  let bicycles := 5
  let bicycle_wheels := 2
  let cars := 12
  let car_wheels := 4
  let tricycles := 3
  let tricycle_wheels := 3
  let single_axle_trailers := 2
  let single_axle_trailer_wheels := 2
  let double_axle_trailers := 2
  let double_axle_trailer_wheels := 4
  let eighteen_wheeler := 1
  let eighteen_wheeler_wheels := 18
  bicycles * bicycle_wheels + cars * car_wheels +
  tricycles * tricycle_wheels + single_axle_trailers * single_axle_trailer_wheels +
  double_axle_trailers * double_axle_trailer_wheels + eighteen_wheeler * eighteen_wheeler_wheels = 97 := 
by
  let bicycles := 5
  let bicycle_wheels := 2
  let cars := 12
  let car_wheels := 4
  let tricycles := 3
  let tricycle_wheels := 3
  let single_axle_trailers := 2
  let single_axle_trailer_wheels := 2
  let double_axle_trailers := 2
  let double_axle_trailer_wheels := 4
  let eighteen_wheeler := 1
  let eighteen_wheeler_wheels := 18
  have h : bicycles * bicycle_wheels + cars * car_wheels +
            tricycles * tricycle_wheels + single_axle_trailers * single_axle_trailer_wheels +
            double_axle_trailers * double_axle_trailer_wheels + eighteen_wheeler * eighteen_wheeler_wheels = 97 := by
    calc 5 * 2 + 12 * 4 + 3 * 3 + 2 * 2 + 2 * 4 + 1 * 18 = 10 + 48 + 9 + 4 + 8 + 18 : by rfl
                                            ... = 97 : by rfl
  exact h

end total_wheels_in_garage_l764_764477
