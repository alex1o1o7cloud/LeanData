import Mathlib

namespace greatest_integer_gcd_30_is_125_l789_789526

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l789_789526


namespace vaccine_cost_reduction_l789_789089

theorem vaccine_cost_reduction (x : ℝ) :
  let cost_two_years_ago := 5000
  ∧ let annual_rate := x
  ∧ let cost_last_year := cost_two_years_ago * (1 - annual_rate)
  ∧ let cost_this_year := cost_last_year * (1 - annual_rate)
  in (cost_last_year - cost_this_year = 5000 * x - 5000 * x^2) :=
by
  sorry

end vaccine_cost_reduction_l789_789089


namespace estimate_diff_and_prod_l789_789917

variable {x y : ℝ}
variable (hx : x > y) (hy : y > 0)

theorem estimate_diff_and_prod :
  (1.1*x) - (y - 2) = (x - y) + 0.1 * x + 2 ∧ (1.1 * x) * (y - 2) = 1.1 * (x * y) - 2.2 * x :=
by 
  sorry -- Proof details go here

end estimate_diff_and_prod_l789_789917


namespace towels_per_person_l789_789580

-- Define the conditions
def num_rooms : ℕ := 10
def people_per_room : ℕ := 3
def total_towels : ℕ := 60

-- Define the total number of people
def total_people : ℕ := num_rooms * people_per_room

-- Define the proposition to prove
theorem towels_per_person : total_towels / total_people = 2 :=
by sorry

end towels_per_person_l789_789580


namespace largest_integer_less_than_100_with_remainder_4_l789_789204

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789204


namespace power_of_two_plus_one_div_by_power_of_three_l789_789830

theorem power_of_two_plus_one_div_by_power_of_three (n : ℕ) : 3^(n + 1) ∣ (2^(3^n) + 1) :=
sorry

end power_of_two_plus_one_div_by_power_of_three_l789_789830


namespace find_interest_rate_l789_789151

noncomputable def interest_rate_solution : ℝ :=
  let P := 800
  let A := 1760
  let t := 4
  let n := 1
  (A / P) ^ (1 / (n * t)) - 1

theorem find_interest_rate : interest_rate_solution = 0.1892 := 
by
  sorry

end find_interest_rate_l789_789151


namespace symmetry_x_y_axis_symmetry_line_y_neg1_l789_789861

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := { x := 1, y := 2 }

-- Condition for symmetry with respect to x-axis
def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Condition for symmetry with respect to the line y = -1
def symmetric_line_y_neg1 (p : Point) : Point :=
  { x := p.x, y := 2 * 1 - p.y - 1 }

-- Theorem statements
theorem symmetry_x_y_axis : symmetric_x P = { x := 1, y := -2 } := sorry
theorem symmetry_line_y_neg1 : symmetric_line_y_neg1 { x := 1, y := -2 } = { x := 1, y := 3 } := sorry

end symmetry_x_y_axis_symmetry_line_y_neg1_l789_789861


namespace max_val_f_l789_789327

def f (x : ℝ) : ℝ := 2 * real.exp 1 * (real.exp 1) * real.ln x - x / real.exp 1

theorem max_val_f :
  ∃ x : ℝ, f x = 2 * real.ln 2 := sorry

end max_val_f_l789_789327


namespace general_term_and_sum_of_b_n_l789_789399

theorem general_term_and_sum_of_b_n :
  (∀ n : ℕ, a_n = 2^(n-1))
  ∧ (∀ n : ℕ, T_n = (3 * n * (n + 1) / 2) * ln 2)
  :=
begin
  let a : ℕ → ℝ := λ n, 2^(n-1),
  let b : ℕ → ℝ := λ n, ln (a (3 * n + 1)),
  let T : ℕ → ℝ := λ n, (3 * n * (n + 1) / 2) * ln 2,

  have h_geom_seq : ∀ n : ℕ, a (n + 1) = 2 * a n,
  { sorry },

  have h_increasing : ∀ n : ℕ, a n < a (n + 1),
  { sorry },

  have a1_a3_sum_5 : a 1 + a 3 = 5,
  { sorry },

  have seq_arithmetic : a 1 + 3, 3 * a 2, a 3 + 4 form arithmetic_sequence,
  { sorry },

  -- part 1: Prove that a_n = 2^(n-1)
  have h_gen_term_proved : ∀ n : ℕ, a n = 2^(n-1),
  { sorry },

  -- part 2: Prove that T_n = (3 * n * (n + 1) / 2) * ln 2
  have h_sum_proved : ∀ n : ℕ, T n = (3 * n * (n + 1) / 2) * ln 2,
  { sorry },

  exact ⟨h_gen_term_proved, h_sum_proved⟩,
end

end general_term_and_sum_of_b_n_l789_789399


namespace greatest_integer_with_gcd_l789_789520

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l789_789520


namespace tan_beta_value_l789_789281

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = -3 / 4) (h2 : Real.tan (α + β) = 1) : Real.tan β = 7 :=
sorry

end tan_beta_value_l789_789281


namespace perpendicular_lines_l789_789469

theorem perpendicular_lines (m : ℝ) :
  (m+2)*(m-1) + m*(m-4) = 0 ↔ m = 2 ∨ m = -1/2 :=
by 
  sorry

end perpendicular_lines_l789_789469


namespace right_focus_hyperbola_l789_789199

def hyperbola_eqn (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

theorem right_focus_hyperbola :
    ∃ c : ℝ, c = 5 ∧ (hyperbola_eqn c 0) :=
by
  use 5
  split
  sorry -- This will be replaced with the actual proof

end right_focus_hyperbola_l789_789199


namespace chairs_in_sixth_row_l789_789915

theorem chairs_in_sixth_row : 
  let number_of_chairs (n : ℕ) := 14 + 9 * (n - 1) 
  in number_of_chairs 6 = 59 :=
by
  sorry

end chairs_in_sixth_row_l789_789915


namespace train_speed_l789_789086

theorem train_speed (v : ℝ) 
  (h1 : 50 * 2.5 + v * 2.5 = 285) : v = 64 := 
by
  -- h1 unfolds conditions into the mathematical equation
  -- here we would have the proof steps, adding a "sorry" to skip proof steps.
  sorry

end train_speed_l789_789086


namespace polar_coords_of_M_l789_789369

theorem polar_coords_of_M :
  (M : ℝ × ℝ) = (-√3, -1) →
  (polar_coords : ℝ × ℝ) = (2, 7 * Real.pi / 6) :=
by
  sorry

end polar_coords_of_M_l789_789369


namespace period_of_tan_x_div_3_l789_789107

theorem period_of_tan_x_div_3 : ∃ T > 0, ∀ x, tan (x / 3) = tan ((x + T) / 3) :=
by
  use 3 * Real.pi
  intros x
  rew_rw (3 * Real.pi)
  rw [Real.tan_periodic]
  sorry

end period_of_tan_x_div_3_l789_789107


namespace area_fraction_l789_789471

noncomputable theory
open_locale classical

-- Define the octagon and midpoints structure
structure RegularOctagon (α : Type) [OrderedField α] :=
  (vertices : Fin 8 → EuclideanSpace α (Fin 2))
  (regular : ∀ i j : Fin 8, ∥vertices i - vertices j∥ = ∥vertices 0 - vertices 1∥)

-- Define the midpoints formation of the octagon
def Midpoints (α : Type) [OrderedField α] (O : RegularOctagon α) : Fin 8 → EuclideanSpace α (Fin 2) :=
  λ i, (O.vertices i + O.vertices ((i + 1) % 8)) / 2

-- The areas of octagons (larger and smaller)
def area (α : Type) [OrderedField α] (O : RegularOctagon α) : α := sorry
def area_smaller(α : Type) [OrderedField α] (O : RegularOctagon α) (midpoints : Fin 8 → EuclideanSpace α (Fin 2)) : α := sorry

-- The final theorem
theorem area_fraction (α : Type) [OrderedField α] (O : RegularOctagon α) (M : Fin 8 → EuclideanSpace α (Fin 2) := Midpoints α O) :
  area_smaller α O M = area α O / 4 :=
sorry

end area_fraction_l789_789471


namespace problem1_problem2_l789_789567

-- Problem 1: Prove that if f(x+1) = x^2 - 2x, then f(x) = x^2 - 4x + 3.
theorem problem1 (f : ℝ → ℝ) (h : ∀ x, f(x+1) = x^2 - 2x) : ∀ x, f(x) = x^2 - 4x + 3 := 
sorry

-- Problem 2: Prove that the maximum value of the function f(x) = 1 / (1 - x * (1 - x)) is 4/3.
theorem problem2 : ∃ x : ℝ, (∀ y : ℝ, 1 / (1 - y * (1 - y)) ≤ 4/3)  ∧ 1 / (1 - x * (1 - x)) = 4/3 :=
sorry

end problem1_problem2_l789_789567


namespace utility_bill_amount_l789_789011

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l789_789011


namespace problem_l789_789283

def f (x : ℚ) : ℚ := x^(-2) + (x^(-2) / (1 + x^(-2)))

theorem problem (x := 3 : ℚ) : f (f x) = 71468700 / 3055721 :=
  by
    let y := (19 / 90 : ℚ)
    have hy : f x = y := by sorry
    have hx : f y = 71468700 / 3055721 := by sorry
    exact hx

end problem_l789_789283


namespace trajectory_C1_max_area_MPQ_l789_789001

section math_proof

variables {x y t : ℝ}

-- Ellipse C1
def C1 (x y : ℝ) : Prop := (x^2 / 5) + (y^2 / 4) = 1

-- Parabola C2
def C2 (x : ℝ) : ℝ := x^2

noncomputable def max_area_triangle_MPQ (t : ℝ) : ℝ :=
  (Real.sqrt 5 / 10) * Real.sqrt (104 - (t^2 - 10)^2)

-- Condition: Slopes product constraint
axiom slopes_product_condition (x y : ℝ) (h : C1 x y) : (y / (x + Real.sqrt 5)) * (y / (x - Real.sqrt 5)) = -4/5

-- Proof: Equation of the trajectory C1
theorem trajectory_C1 (x y : ℝ) (hx : x ≠ -Real.sqrt 5) (hx' : x ≠ Real.sqrt 5) :
  slopes_product_condition x y (C1 x y) ↔ C1 x y := 
by sorry

-- Proof: Maximum area of triangle MPQ
theorem max_area_MPQ (M : ℝ × ℝ) (M_prop : M = (0, 1/5)) 
    (N : ℝ × ℝ) (N_prop : ∃ t : ℝ, N = (t, C2 t)) : 
    ∃ t : ℝ, max_area_triangle_MPQ t = Real.sqrt 130 / 5 :=
by sorry

end math_proof

end trajectory_C1_max_area_MPQ_l789_789001


namespace small_pump_fill_time_l789_789592

noncomputable def small_pump_time (large_pump_time combined_time : ℝ) : ℝ :=
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate

theorem small_pump_fill_time :
  small_pump_time (1 / 3) 0.2857142857142857 = 2 :=
by
  sorry

end small_pump_fill_time_l789_789592


namespace shirt_original_price_l789_789591

theorem shirt_original_price {P : ℝ} :
  (P * 0.80045740423098913 * 0.8745 = 105) → P = 150 :=
by sorry

end shirt_original_price_l789_789591


namespace solution_set_of_inequality_l789_789068

theorem solution_set_of_inequality : {x : ℝ | x^2 - 2 * x ≤ 0} = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l789_789068


namespace arithmetic_sequence_tenth_term_l789_789071

theorem arithmetic_sequence_tenth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 6 * d = 13) :
  a + 9 * d = 19 := 
sorry

end arithmetic_sequence_tenth_term_l789_789071


namespace total_flowers_l789_789896

theorem total_flowers (R T L : ℕ) 
  (hR : R = 58)
  (hT : R = T + 15)
  (hL : R = L - 25) :
  R + T + L = 184 :=
by 
  sorry

end total_flowers_l789_789896


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789244

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789244


namespace perpendicular_tangents_intersection_l789_789292

open Point Circle Line Segment 

theorem perpendicular_tangents_intersection
    {A B C D P Q : Point}
    (hAOnCircle: A ∈ Circle C D)
    (hBOnCircle: B ∈ Circle C D)
    (hBOnCircle: B ∈ Circle C D)
    (hCDNotDiameter: ¬ Segment C D = Diameter C D)
    (hABDiameter: Segment A B = Diameter A B)
    (hP: TangentIntersection C D)
    (hQ: ← Intersection (Line A C) (Line B D)) :
    Perpendicular (Line P Q) (Line A B) := by
    sorry

end perpendicular_tangents_intersection_l789_789292


namespace bankers_gain_correct_l789_789454

def PW : ℝ := 600
def R : ℝ := 0.10
def n : ℕ := 2

def A : ℝ := PW * (1 + R)^n
def BG : ℝ := A - PW

theorem bankers_gain_correct : BG = 126 :=
by
  sorry

end bankers_gain_correct_l789_789454


namespace swap_numbers_l789_789043

theorem swap_numbers (a b t : ℕ) (h₁ : a = 8) (h₂ : b = 9) :
    let t := b in
    let b := a in
    let a := t in
    a = 9 ∧ b = 8 :=
by
  sorry

end swap_numbers_l789_789043


namespace hyperbola_eccentricity_l789_789308

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (A B : ℝ × ℝ) (F : ℝ × ℝ)
variable (hA : A = (a, 0))
variable (hF : F = (c, 0))
variable (hB : B = (c, b ^ 2 / a))
variable (h_slope : (b ^ 2 / a - 0) / (c - a) = 3)
variable (h_hyperbola : b ^ 2 = c ^ 2 - a ^ 2)
def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity : eccentricity a c = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789308


namespace combined_pumps_fill_time_l789_789556

theorem combined_pumps_fill_time (small_pump_time large_pump_time : ℝ) (h1 : small_pump_time = 4) (h2 : large_pump_time = 1/2) : 
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  (1 / combined_rate) = 4 / 9 :=
by
  -- Definitions of rates
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  
  -- Using placeholder for the proof.
  sorry

end combined_pumps_fill_time_l789_789556


namespace equation_of_circle_center_0_4_passing_through_3_0_l789_789867

noncomputable def circle_radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem equation_of_circle_center_0_4_passing_through_3_0 :
  ∃ (r : ℝ), (r = circle_radius 0 4 3 0) ∧ (r = 5) ∧ ((x y : ℝ) → ((x - 0) ^ 2 + (y - 4) ^ 2 = r ^ 2) ↔ (x ^ 2 + (y - 4) ^ 2 = 25)) :=
by
  sorry

end equation_of_circle_center_0_4_passing_through_3_0_l789_789867


namespace probability_of_three_even_numbers_l789_789616

theorem probability_of_three_even_numbers (n : ℕ) (k : ℕ) (p_even : ℚ) (p_odd : ℚ) (comb : ℕ → ℕ → ℕ) 
    (h_n : n = 5) (h_k : k = 3) (h_p_even : p_even = 1/2) (h_p_odd : p_odd = 1/2) 
    (h_comb : comb 5 3 = 10) :
    comb n k * (p_even ^ k) * (p_odd ^ (n - k)) = 5 / 16 :=
by sorry

end probability_of_three_even_numbers_l789_789616


namespace largest_int_less_than_100_by_7_l789_789228

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789228


namespace initial_set_must_be_equal_l789_789289

def all_numbers_equal (n : ℕ) (a : Fin n → ℤ) : Prop :=
  ∀ i j : Fin n, a i = a j

theorem initial_set_must_be_equal
    (n : ℕ)
    (hodd : n % 2 = 1)
    (a : Fin n → ℤ)
    (hprocess : ∀ i : Fin n, (a i + a ((i + 1) % n)) % 2 = 0) :
  all_numbers_equal n a :=
by
  sorry

end initial_set_must_be_equal_l789_789289


namespace nat_sum_factors_l789_789029

theorem nat_sum_factors (n : ℕ) (a : ℕ) (h : a ≤ n!) : 
  ∃ (l : List ℕ), l.length ≤ n ∧ (∀ x ∈ l, x ∣ n!) ∧ (∀ i j, i ≠ j → (i < l.length ∧ j < l.length → l.nth i ≠ l.nth j)) ∧ l.sum = a :=
sorry

end nat_sum_factors_l789_789029


namespace mutually_exclusive_non_contradictory_events_l789_789278

def event_space {α : Type} (bag : set α) (draw : ℕ) :=
  {outcome : multiset α | outcome.card = draw ∧ outcome ⊆ bag}

def events_mutually_exclusive {α : Type} (E1 E2: set (multiset α)) :=
  ∀ e : multiset α, e ∈ E1 → ¬ e ∈ E2

def not_contradictory_event {α : Type} (E1 E2: set (multiset α)) :=
  ∃ e : multiset α, e ∈ E1 ∨ e ∈ E2

theorem mutually_exclusive_non_contradictory_events :
  let bag := {'R', 'R', 'W', 'W'}
  let E1 := {e | e.card = 2 ∧ e.count 'W' = 1 ∧ e.count 'R' = 1}
  let E2 := {e | e.card = 2 ∧ e.count 'W' = 2 ∧ e.count 'R' = 0}
  ∀ (draw : ℕ),  draw = 2 →
  events_mutually_exclusive (event_space bag draw) E1 E2 ∧ 
  not_contradictory_event (event_space bag draw) E1 E2 :=
by
  intros,
  sorry

end mutually_exclusive_non_contradictory_events_l789_789278


namespace find_k_for_coplanarity_l789_789807

open_locale real_inner_product_space
open real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]
variables {O A B C D : V} (k : ℝ)

def coplanar (A B C D : V) : Prop :=
  ∃ a b c : ℝ, a • (A - O) + b • (B - O) + c • (C - O) = (D - O)

theorem find_k_for_coplanarity (h : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = 0) :
  k = -7 :=
sorry

end find_k_for_coplanarity_l789_789807


namespace solution1_solution2_l789_789435

section
variable {α : Type} [Fintype α] (s : Finset α)

-- Problem 1: The number of ways to select 2 males and 2 females from 4 males and 5 females.
def problem1 : Prop :=
  (s.filter (λ x, x ∈ finset.range 4)).card = 2 ∧
  (s.filter (λ x, x ∈ finset.range 5)).card = 2 →
  s.card = 60

-- Problem 2: The number of ways to select at least 1 male and 1 female,
-- and male student A and female student B cannot be selected together, is 99.
def problem2 (A B : α) : Prop :=
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 3 ∧ (s.filter (λ x, x ∈ finset.range 4)).card = k ∧ (s.filter (λ x, x ∈ finset.range 5)).card = (4 - k)) ∧
  ¬(A ∈ s ∧ B ∈ s) →
  s.card = 99

end

-- Assertions to the assumptions and results.
theorem solution1 : problem1 := sorry
theorem solution2 {α : Type} [Fintype α] (A B : α) : problem2 A B := sorry

end solution1_solution2_l789_789435


namespace problem_relationship_between_lines_l789_789726

-- Given three lines a, b, and c, where a is parallel to b and a intersects c,
-- prove that b and c are either skew or intersecting.

def relationship_between_b_and_c (a b c : Set Point) : Prop :=
  (a ∥ b) ∧ (a ∩ c ≠ ∅) → ((b ∩ c ≠ ∅) ∨ ((∃ p : Point, p ∉ (PLANE_G1 b ∪ PLANE_G2 b) ∧ p ∈ c)))

theorem problem_relationship_between_lines
  {a b c : Set Point}
  (H1 : a ∥ b)
  (H2 : a ∩ c ≠ ∅)
: (b ∩ c ≠ ∅) ∨ ((∃ p : Point, p ∉ (PLANE_G1 b ∪ PLANE_G2 b) ∧ p ∈ c)) :=
begin
  sorry
end

end problem_relationship_between_lines_l789_789726


namespace chemistry_class_size_l789_789167

theorem chemistry_class_size (total_students : ℕ) (students_both : ℕ) (students_total : ℕ → ℕ → ℕ → Prop) 
(students_ratio : ℕ → ℕ ℕ → Prop) :
  let b := total_students - students_both - c in
  let c := 2*b + students_both - students_both in
  students_total 52 :=
  52 = b + 2 * (b + 8) + 8 :=
  c := students_both *(students_total 2b) :=
-- Proof of number of students in chemistry class
    sorry

end chemistry_class_size_l789_789167


namespace pythagorean_theorem_l789_789752

theorem pythagorean_theorem (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (AC BC AB : ℝ) (hABC : (c : ℝ) -> ℝ):
  (angle A B C = 90) -> (AC^2 + BC^2 = AB^2) :=
by
  sorry

end pythagorean_theorem_l789_789752


namespace range_of_a_l789_789718

noncomputable def f (x : ℝ) := -Real.exp x - x
noncomputable def g (a x : ℝ) := a * x + Real.cos x

theorem range_of_a :
  (∀ x : ℝ, ∃ y : ℝ, (g a y - g a y) / (y - y) * ((f x - f x) / (x - x)) = -1) →
  (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l789_789718


namespace plant_distribution_l789_789499

theorem plant_distribution:
  ∀ (plants : List String) (garden_centers : List String),
    plants = ["Daisy", "Rose", "Bud", "Leaf", "Stem"] →
    garden_centers.length = 4 →
    (∀ (gc : String) (parent : String) (offspring : String),
      gc ∈ garden_centers →
      (parent, offspring) ∈ [("Daisy", "Bud"), ("Daisy", "Leaf"), ("Daisy", "Stem"), ("Rose", "Bud"), ("Rose", "Leaf"), ("Rose", "Stem")] →
      ¬ (gc ∈ plants ∧ parent ∈ plants ∧ offspring ∈ plants)) →
    -- To prove
    num_ways_to_distribute plants garden_centers = 132 :=
begin
  sorry
end

end plant_distribution_l789_789499


namespace distance_between_parametric_points_l789_789170

theorem distance_between_parametric_points:
  (let x := λ t : ℝ, 2 + 3 * t,
       y := λ t : ℝ, 2 + t,
       p1 := (x 0, y 0),
       p2 := (x 1, y 1) in
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)) = real.sqrt 10 :=
by {
  let x := λ t : ℝ, 2 + 3 * t,
  let y := λ t : ℝ, 2 + t,
  let p1 := (x 0, y 0),
  let p2 := (x 1, y 1),
  have h1 : p1 = (2, 2), by { simp [x, y], },
  have h2 : p2 = (5, 3), by { simp [x, y], },
  rw [h1, h2], simp,
  exact eq.symm (real.sqrt_eq_iff_sq_eq.2 (by norm_num)),
}

end distance_between_parametric_points_l789_789170


namespace black_white_ratio_l789_789643

theorem black_white_ratio :
  let original_black := 18
  let original_white := 39
  let replaced_black := original_black + 13
  let inner_border_black := (9^2 - 7^2)
  let outer_border_white := (11^2 - 9^2)
  let total_black := replaced_black + inner_border_black
  let total_white := original_white + outer_border_white
  let ratio_black_white := total_black / total_white
  ratio_black_white = 63 / 79 :=
sorry

end black_white_ratio_l789_789643


namespace inequalities_hold_l789_789694

theorem inequalities_hold (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (ab_le : a * b ≤ 1) ∧
  (a2_b2_ge : a^2 + b^2 ≥ 2) ∧
  (sqrt_a_sqrt_b_le : Real.sqrt a + Real.sqrt b ≤ 2) ∧
  ¬(frac1a_frac2b_ge : 1/a + 2/b ≥ 3) := sorry

end inequalities_hold_l789_789694


namespace greatest_integer_gcd_30_is_125_l789_789529

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l789_789529


namespace boat_transport_in_2_days_l789_789637

theorem boat_transport_in_2_days (trips_per_day : ℕ) (people_per_trip : ℕ) (days : ℕ) 
    (h1 : trips_per_day = 4) (h2 : people_per_trip = 12) (h3 : days = 2) : 
    trips_per_day * people_per_trip * days = 96 :=
by 
  rw [h1, h2, h3]
  norm_num

end boat_transport_in_2_days_l789_789637


namespace find_dividend_l789_789098

noncomputable def divisor := (-14 : ℚ) / 3
noncomputable def quotient := (-286 : ℚ) / 5
noncomputable def remainder := (19 : ℚ) / 9
noncomputable def dividend := 269 + (2 / 45 : ℚ)

theorem find_dividend :
  dividend = (divisor * quotient) + remainder := by
  sorry

end find_dividend_l789_789098


namespace problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l789_789719

theorem problem_a_eq_2 (x : ℝ) : (12 * x^2 - 2 * x > 4) ↔ (x < -1 / 2 ∨ x > 2 / 3) := sorry

theorem problem_a_real_pos (a x : ℝ) (h : a > 0) : (12 * x^2 - a * x > a^2) ↔ (x < -a / 4 ∨ x > a / 3) := sorry

theorem problem_a_real_zero (x : ℝ) : (12 * x^2 > 0) ↔ (x ≠ 0) := sorry

theorem problem_a_real_neg (a x : ℝ) (h : a < 0) : (12 * x^2 - a * x > a^2) ↔ (x < a / 3 ∨ x > -a / 4) := sorry

end problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l789_789719


namespace trajectory_of_M_is_ellipse_l789_789693

def circle_eq (x y : ℝ) : Prop := ((x + 3)^2 + y^2 = 100)

def point_B (x y : ℝ) : Prop := (x = 3 ∧ y = 0)

def point_on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, P = (x, y) ∧ circle_eq x y

def perpendicular_bisector_intersects_CQ_at_M (B P M : ℝ × ℝ) : Prop :=
  (B.fst = 3 ∧ B.snd = 0) ∧
  point_on_circle P ∧
  ∃ r : ℝ, (P.fst + B.fst) / 2 = M.fst ∧ r = (M.snd - P.snd) / (M.fst - P.fst) ∧ 
  r = -(P.fst - B.fst) / (P.snd - B.snd)

theorem trajectory_of_M_is_ellipse (M : ℝ × ℝ) 
  (hC : ∀ x y, circle_eq x y)
  (hB : point_B 3 0)
  (hP : ∃ P : ℝ × ℝ, point_on_circle P)
  (hM : ∃ B P : ℝ × ℝ, perpendicular_bisector_intersects_CQ_at_M B P M) 
: (M.fst^2 / 25 + M.snd^2 / 16 = 1) := 
sorry

end trajectory_of_M_is_ellipse_l789_789693


namespace ducks_at_NP_l789_789444

-- Define the variables
variable (M_LM : ℕ) (M_LM_eq : M_LM = 100)
variable (M_NP : ℕ) (P_NP : ℕ)

-- Define the conditions
def cond1 (M_NP : ℕ) : Prop := M_NP = 2 * M_LM + 6
def cond2 (P_NP : ℕ) : Prop := P_NP = 4 * M_LM

-- Define the total number of ducks at North Pond
def total_ducks_NP (M_NP P_NP : ℕ) : ℕ := M_NP + P_NP

-- The theorem to prove
theorem ducks_at_NP : cond1 M_NP → cond2 P_NP → total_ducks_NP M_NP P_NP = 606 :=
by
  intros cond1 cond2
  rw [cond1, cond2, M_LM_eq]
  simp
  sorry

end ducks_at_NP_l789_789444


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l789_789533

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l789_789533


namespace combined_weight_of_barney_and_five_dinosaurs_l789_789614

theorem combined_weight_of_barney_and_five_dinosaurs:
  let w := 800
  let combined_weight_five_regular := 5 * w
  let barney_weight := combined_weight_five_regular + 1500
  let combined_weight := barney_weight + combined_weight_five_regular
  in combined_weight = 9500 := by
  sorry

end combined_weight_of_barney_and_five_dinosaurs_l789_789614


namespace combined_weight_l789_789607

-- Definition of conditions
def regular_dinosaur_weight := 800
def five_regular_dinosaurs_weight := 5 * regular_dinosaur_weight
def barney_weight := five_regular_dinosaurs_weight + 1500

-- Statement to prove
theorem combined_weight (h1: five_regular_dinosaurs_weight = 5 * regular_dinosaur_weight)
                        (h2: barney_weight = five_regular_dinosaurs_weight + 1500) : 
        (barney_weight + five_regular_dinosaurs_weight = 9500) :=
by
    sorry

end combined_weight_l789_789607


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l789_789536

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l789_789536


namespace lambda_range_l789_789685
noncomputable def a_seq : ℕ → ℝ
| 0     := 1
| (n+1) := a_seq n / (a_seq n + 2)

noncomputable def b_seq (λ : ℝ) : ℕ → ℝ
| 0     := - λ
| (n+1) := (n - 2 * λ) * (1 / (a_seq n) + 1)

theorem lambda_range (λ : ℝ) (h_increasing: ∀ n : ℕ, b_seq λ (n+1) > b_seq λ n) : λ < 2 / 3 :=
sorry

end lambda_range_l789_789685


namespace man_speed_in_still_water_l789_789942

theorem man_speed_in_still_water (c_speed : ℝ) (distance_m : ℝ) (time_sec : ℝ) (downstream_distance_km : ℝ) (downstream_time_hr : ℝ) :
    c_speed = 3 →
    distance_m = 15 →
    time_sec = 2.9997600191984644 →
    downstream_distance_km = distance_m / 1000 →
    downstream_time_hr = time_sec / 3600 →
    (downstream_distance_km / downstream_time_hr) - c_speed = 15 :=
by
  intros hc hd ht hdownstream_distance hdownstream_time 
  sorry

end man_speed_in_still_water_l789_789942


namespace greatest_int_with_gcd_five_l789_789542

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l789_789542


namespace ratio_of_AC_to_BD_l789_789027

theorem ratio_of_AC_to_BD (A B C D : ℝ) (AB BC AD AC BD : ℝ) 
  (h1 : AB = 2) (h2 : BC = 5) (h3 : AD = 14) (h4 : AC = AB + BC) (h5 : BD = AD - AB) :
  AC / BD = 7 / 12 := by
  sorry

end ratio_of_AC_to_BD_l789_789027


namespace yi_wins_probability_l789_789791

open Finset

theorem yi_wins_probability :
  let pJia := 2/3,
      pYi := 1/3 in
  ( ∑ k in range 3, nat.choose 5 k * (pYi ^ k) * (pJia ^ (5 - k)) )
  + ( ∑ k in range 1, nat.choose 5 (k + 3) * (pYi ^ (k + 3)) * (pJia ^ (5 - (k + 3))) )
  = 17/81 :=
by
  sorry

end yi_wins_probability_l789_789791


namespace violet_ticket_cost_l789_789093

theorem violet_ticket_cost :
  (2 * 35 + 5 * 20 = 170) ∧
  (((35 - 17.50) + 35 + 5 * 20) = 152.50) ∧
  ((152.50 - 150) = 2.50) :=
by
  sorry

end violet_ticket_cost_l789_789093


namespace nina_unfair_coin_flips_l789_789427

open Classical

def coin_flip (n m : ℕ) (pH pT : ℚ) : ℚ :=
  (nat.choose n m) * (pH^m) * (pT^(n - m))

theorem nina_unfair_coin_flips :
  coin_flip 10 3 (1/3) (2/3) = 512 / 1969 := 
by 
  sorry

end nina_unfair_coin_flips_l789_789427


namespace derivative_at_pi_over_2_l789_789325

def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem derivative_at_pi_over_2 : 
  deriv f (Real.pi / 2) = 0 :=
by
  sorry

end derivative_at_pi_over_2_l789_789325


namespace soccer_game_goals_l789_789190

theorem soccer_game_goals (A1_first_half A2_first_half B1_first_half B2_first_half : ℕ) 
  (h1 : A1_first_half = 8)
  (h2 : B1_first_half = A1_first_half / 2)
  (h3 : B2_first_half = A1_first_half)
  (h4 : A2_first_half = B2_first_half - 2) : 
  A1_first_half + A2_first_half + B1_first_half + B2_first_half = 26 :=
by
  -- The proof is not needed, so we use sorry to skip it.
  sorry

end soccer_game_goals_l789_789190


namespace fourth_vertex_is_4i_l789_789372

open Complex

noncomputable def fourth_vertex : ℂ :=
let A := (3 + 3 * I : ℂ) in
let B := (2 - 2 * I : ℂ) in
let C := (-1 - I : ℂ) in
let M := ((A.re + C.re) / 2 + ((A.im + C.im) / 2) * I : ℂ) in
let D := (2 * M.re - B.re + (2 * M.im - B.im) * I : ℂ) in
D

theorem fourth_vertex_is_4i :
  fourth_vertex = 4 * I := by
  sorry

end fourth_vertex_is_4i_l789_789372


namespace apple_ratio_l789_789755

theorem apple_ratio (R G Y : ℕ) 
  (hR : R = 16)
  (hG : G = 5 * R / 2)
  (hY : Y = G + 18) : 
  (R / 2) : (G / 2) : (Y / 2) = 8 : 20 : 29 :=
by
  sorry

end apple_ratio_l789_789755


namespace find_y_of_series_eq_92_l789_789644

theorem find_y_of_series_eq_92 (y : ℝ) (h : (∑' n, (2 + 5 * n) * y^n) = 92) (converge : abs y < 1) : y = 18 / 23 :=
sorry

end find_y_of_series_eq_92_l789_789644


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789240

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789240


namespace b_finishes_remaining_work_in_5_days_l789_789930

theorem b_finishes_remaining_work_in_5_days :
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  days_b_to_finish = 5 :=
by
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  show days_b_to_finish = 5
  sorry

end b_finishes_remaining_work_in_5_days_l789_789930


namespace grassy_plot_width_l789_789947

theorem grassy_plot_width (L : ℝ) (P : ℝ) (C : ℝ) (cost_per_sqm : ℝ) (W : ℝ) : 
  L = 110 →
  P = 2.5 →
  C = 510 →
  cost_per_sqm = 0.6 →
  (115 * (W + 5) - 110 * W = C / cost_per_sqm) →
  W = 55 :=
by
  intros hL hP hC hcost_per_sqm harea
  sorry

end grassy_plot_width_l789_789947


namespace gcd_lcm_product_eq_abc_l789_789339

theorem gcd_lcm_product_eq_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  let D := Nat.gcd (Nat.gcd a b) c
  let m := Nat.lcm (Nat.lcm a b) c
  D * m = a * b * c :=
by
  sorry

end gcd_lcm_product_eq_abc_l789_789339


namespace percentage_of_childrens_books_l789_789941

/-- Conditions: 
- There are 160 books in total.
- 104 of them are for adults.
Prove that the percentage of books intended for children is 35%. --/
theorem percentage_of_childrens_books (total_books : ℕ) (adult_books : ℕ) 
  (h_total : total_books = 160) (h_adult : adult_books = 104) :
  (160 - 104) / 160 * 100 = 35 := 
by {
  sorry -- Proof skipped
}

end percentage_of_childrens_books_l789_789941


namespace ratio_B_C_l789_789939

variable (A B C : ℕ)
variable (h1 : A = B + 2)
variable (h2 : A + B + C = 37)
variable (h3 : B = 14)

theorem ratio_B_C : B / C = 2 := by
  sorry

end ratio_B_C_l789_789939


namespace problem_solution_l789_789353

/-- The given conditions. -/
variable (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 2 / b = 1)

/-- The expressions corresponding to parts B and D. -/
def part_B : Prop := 2 / (a - 1) + 1 / (b - 2) ≥ 2
def part_D : Prop := 2 * a + b ≥ 8

/-- The main theorem. -/
theorem problem_solution : part_B a b ∧ part_D a b := 
by
  sorry

end problem_solution_l789_789353


namespace largest_integer_remainder_condition_l789_789262

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789262


namespace greatest_integer_with_gcd_l789_789519

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l789_789519


namespace cross_bridge_time_l789_789582

def distance_in_meters (km : ℕ) : ℕ := km * 1000
def time_in_minutes (hr : ℕ) : ℕ := hr * 60
def speed_in_meters_per_minute (km_per_hr : ℕ) : ℕ := distance_in_meters km_per_hr / time_in_minutes 1

theorem cross_bridge_time (distance_km_hr : ℕ) (bridge_length_m : ℕ) : 
  distance_km_hr = 6 → 
  bridge_length_m = 1500 →
  distance_in_meters distance_km_hr / time_in_minutes 1 / 60 = 15 :=
by
  intros h1 h2
  rw [h1, h2]
  rw [distance_in_meters, time_in_minutes]
  sorry

end cross_bridge_time_l789_789582


namespace hyperbola_eccentricity_proof_l789_789299

noncomputable def hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : ℝ :=
2

theorem hyperbola_eccentricity_proof 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : hyperbola_eccentricity a b h1 h2 C F A B h3 = 2 :=
sorry

end hyperbola_eccentricity_proof_l789_789299


namespace geologists_probability_l789_789371

theorem geologists_probability
  (n roads : ℕ) (speed_per_hour : ℕ) 
  (angle_between_neighbors : ℕ)
  (distance_limit : ℝ) : 
  n = 6 ∧ speed_per_hour = 4 ∧ angle_between_neighbors = 60 ∧ distance_limit = 6 → 
  prob_distance_at_least_6_km = 0.5 :=
by
  sorry

noncomputable def prob_distance_at_least_6_km : ℝ := 0.5  -- Placeholder definition

end geologists_probability_l789_789371


namespace ordinary_line_eq_rectangular_circle_eq_min_distance_PQ_l789_789834

-- Definitions based on conditions
def param_line (t : ℝ) : ℝ × ℝ := (- sqrt 2 / 2 * t, sqrt 2 / 2 * t)
def polar_circle (θ : ℝ) : ℝ := 2 * (Real.cos θ)

-- Translations of the problems into Lean theorems

-- Theorem for the ordinary equation of the line l
theorem ordinary_line_eq : ∀ t : ℝ, (param_line t).fst - (param_line t).snd + 1 = 0 :=
sorry

-- Theorem for the rectangular coordinate equation of the circle C
theorem rectangular_circle_eq : ∀ θ : ℝ,
  (polar_circle θ) ^ 2 = 2 * (polar_circle θ) * (Real.cos θ) ↔ 
  ∃ x y : ℝ, x^2 + y^2 - 2*x = 0 :=
sorry

-- Theorem for the minimum distance |PQ|
theorem min_distance_PQ : 
  let P := param_line
  let Q := fun θ : ℝ => (polar_circle θ * Real.cos θ, polar_circle θ * Real.sin θ)
  ∀ θ : ℝ, ∀ t : ℝ, 
  let d := (Real.sqrt 2) - 1
  Real.sqrt ((P t).fst - (Q θ).fst)^2 + ((P t).snd - (Q θ).snd)^2 = d :=
sorry


end ordinary_line_eq_rectangular_circle_eq_min_distance_PQ_l789_789834


namespace university_committee_l789_789871

open Finset

theorem university_committee (male_count female_count department_count total_committee_count physics_count rcb_count : ℕ)
  (h_male : male_count = 3) (h_female : female_count = 3) (h_department : department_count = 3)
  (h_total : total_committee_count = 7) (h_physics : physics_count = 3) 
  (h_rcb : rcb_count = 4)
  (h_distributed : rcb_count % 2 = 0) :
  (3.choose 2 * 3.choose 1 * 3.choose 1 * 3.choose 1 * 3.choose 1 * 3.choose 1 + 3.choose 3 * 3.choose 1 * 3.choose 1 = 738) :=
by
  rw [h_male, h_female, h_department, h_total, h_physics, h_rcb, h_distributed]
  iterate 2 { rw binomial_eq_choose }
  sorry

end university_committee_l789_789871


namespace no_obtuse_triangle_probability_l789_789667

-- Define the problem in Lean 4

noncomputable def probability_no_obtuse_triangles (points : Fin 4 → ℝ) : ℝ :=
  -- Assuming the points are uniformly distributed and simplifying the problem calculation as suggested
  (3 / 8) ^ 6

-- Statement of the problem
theorem no_obtuse_triangle_probability : ∃ (points : Fin 4 → ℝ), probability_no_obtuse_triangles points = (3 / 8) ^ 6 := 
by 
  sorry -- Proof is omitted

end no_obtuse_triangle_probability_l789_789667


namespace infinite_sum_result_l789_789804

open Real

theorem infinite_sum_result (x : ℝ) (h : x > 1) :
  ∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n)) = 1 / (x - 1) :=
sorry

end infinite_sum_result_l789_789804


namespace sum_of_solutions_l789_789811

noncomputable def f : ℝ → ℝ :=
λ x, if x < -3 then 3 * x + 5 else -x^2 + x + 2

theorem sum_of_solutions :
  (∑ x in { x : ℝ | f x = -1 }.to_finset, x) = 3 :=
by
  sorry

end sum_of_solutions_l789_789811


namespace distance_between_consecutive_trees_l789_789570

theorem distance_between_consecutive_trees 
  (yard_length : ℕ) (num_trees : ℕ) (tree_at_each_end : yard_length > 0 ∧ num_trees ≥ 2) 
  (equal_distances : ∀ k, k < num_trees - 1 → (yard_length / (num_trees - 1) : ℝ) = 12) :
  yard_length = 360 → num_trees = 31 → (yard_length / (num_trees - 1) : ℝ) = 12 := 
by
  sorry

end distance_between_consecutive_trees_l789_789570


namespace dave_pays_more_l789_789149

-- Define the conditions
def total_pizza_cost := 15
def dave_slices := 8
def dave_cost_per_slice := 1.25
def dave_total_cost := dave_slices * dave_cost_per_slice

def doug_slices := 12 - dave_slices
def doug_cost_per_slice := (total_pizza_cost - dave_total_cost) / doug_slices
def doug_total_cost := doug_slices * doug_cost_per_slice

-- Lean 4 statement expressing the question with this setup
theorem dave_pays_more : (dave_total_cost - doug_total_cost) = 6 := by
  -- Abbreviate calculations and verification steps.
  -- This would involve showing LHS == RHS
  sorry

end dave_pays_more_l789_789149


namespace infinite_initial_values_l789_789561

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^2

noncomputable def sequence (x0 : ℝ) (n : ℕ) : ℝ :=
  Nat.recOn n x0 (λ n x_n, f x_n)

theorem infinite_initial_values (x0 : ℝ) :
  ∃ (a : ℕ → ℝ), (∀ m, a m < 2) ∧ (∀ m, a m = 2 - real.sqrt (4 - a (m - 1))) ∧
  ∃ (n : ℕ), sequence x0 n ∈ ({a m | m : ℕ} ∪ {3}) :=
sorry

end infinite_initial_values_l789_789561


namespace new_arithmetic_mean_is_74_74_l789_789703

noncomputable def original_mean := 75
noncomputable def total_numbers := 60
noncomputable def removed_numbers := [70, 80, 90]
noncomputable def original_sum := original_mean * total_numbers
noncomputable def removed_sum := removed_numbers.sum
noncomputable def new_sum := original_sum - removed_sum
noncomputable def remaining_numbers := total_numbers - removed_numbers.length
noncomputable def new_mean := new_sum / remaining_numbers

theorem new_arithmetic_mean_is_74_74 : new_mean = 74.74 := by
  sorry

end new_arithmetic_mean_is_74_74_l789_789703


namespace sum_binom_eq_l789_789028

theorem sum_binom_eq : 
  (∑ k in Finset.range 996, (-1 : ℚ)^k / (1991 - k) * Nat.choose (1991 - k) k) = 1 / 1991 := 
sorry

end sum_binom_eq_l789_789028


namespace math_problem_l789_789750

theorem math_problem (x : ℝ) 
(h₁ : (sqrt x - 5) / 7 = 7) : 
  ((x - 14)^2) / 10 = 842240.4 :=
by
  sorry

end math_problem_l789_789750


namespace train_speed_is_60_kmph_l789_789595

noncomputable def train_length : ℝ := 110
noncomputable def time_to_pass_man : ℝ := 5.999520038396929
noncomputable def man_speed_kmph : ℝ := 6

theorem train_speed_is_60_kmph :
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / time_to_pass_man
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph = 60 :=
by
  sorry

end train_speed_is_60_kmph_l789_789595


namespace largest_integer_less_than_100_with_remainder_4_l789_789201

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789201


namespace monotonicity_of_f_extreme_points_inequality_l789_789702

theorem monotonicity_of_f (k : ℝ) (f : ℝ → ℝ) (h : ∀ x > 1, f(x-1) = 2 * ln(x-1) - k / x + k)
  (hk : -1 ≤ k ∧ k ≤ 0) : 
  ∀ x > 0, monotone f := 
sorry

theorem extreme_points_inequality (k : ℝ) (f : ℝ → ℝ) (x x₁ x₂ : ℝ)
  (h : ∀ x > 1, f(x-1) = 2 * ln(x-1) - k / x + k)
  (h_ex1 : x₁ ≠ x₂)
  (h_ex2 : ∀ x > 1, f(x₁) = 0 ∧ f(x₂) = 0) :
  x * (f(x₁) + f(x₂)) ≥ (x + 1) * (f(x) + 2 - 2 * x) := 
sorry

end monotonicity_of_f_extreme_points_inequality_l789_789702


namespace hyperbola_eccentricity_l789_789305

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (h_hyp : a^2 - y^2 = b^2)
variable (hF : (c, 0) ∈ hyperbola(a,b))
variable (hA : (a, 0) ∈ hyperbola(a,b))
variable (hB : (c, b^2 / a) ∈ hyperbola(a,b))
variable (h_slope : (b^2 / a) / (c - a) = 3)

theorem hyperbola_eccentricity (ha : 0 < a) (hb : 0 < b) (hF : (c, 0))
    (hA : (a, 0)) (hB : (c, b^2 / a))
    (h_slope : (b^2 / a) / (c - a) = 3) : (eccentricity(c, a) = 2) := by
  sorry

end hyperbola_eccentricity_l789_789305


namespace expected_potato_yield_l789_789426

-- Conditions expressed as definitions
def step_length := 1.5 -- each step in feet
def yield_per_sq_foot := 0.75 -- pounds per square foot

def section1_length_steps := 10 -- first section length in steps
def section1_width_steps := 25 -- first section width in steps
def section2_side_steps := 10 -- second section is a square with side length in steps

-- Conversion to feet
def section1_length_feet := section1_length_steps * step_length
def section1_width_feet := section1_width_steps * step_length
def section2_side_feet := section2_side_steps * step_length

-- Calculation of areas
def section1_area := section1_length_feet * section1_width_feet
def section2_area := section2_side_feet * section2_side_feet

-- Total area
def total_area := section1_area + section2_area

-- Calculate expected yield
def expected_yield := total_area * yield_per_sq_foot

-- The proof statement
theorem expected_potato_yield :
  expected_yield = 590.625 := by
  sorry

end expected_potato_yield_l789_789426


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789223

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789223


namespace find_f_1789_l789_789806

def f : ℕ → ℕ := sorry

axiom f_1 : f 1 = 5
axiom f_f_n : ∀ n, f (f n) = 4 * n + 9
axiom f_2n : ∀ n, f (2 * n) = (2 * n) + 1 + 3

theorem find_f_1789 : f 1789 = 3581 :=
by
  sorry

end find_f_1789_l789_789806


namespace math_proof_problem_l789_789168

noncomputable def mixed_number_eval : ℚ :=
  65 * ((4 + 1 / 3) + (3 + 1 / 2)) / ((2 + 1 / 5) - (1 + 2 / 3))

theorem math_proof_problem :
  mixed_number_eval = 954 + 33 / 48 := 
sorry

end math_proof_problem_l789_789168


namespace min_value_of_sum_of_squares_l789_789781

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x - 2 * y - 3 * z = 4) : 
  (x^2 + y^2 + z^2) ≥ 8 / 7 :=
sorry

end min_value_of_sum_of_squares_l789_789781


namespace value_of_a_l789_789354

theorem value_of_a (a : ℝ) (h : (coeff (expand (5 : ℕ) (λ k : ℕ, a ^ (5 - k) * (-1)^k) 3) = 80) : a = 2 :=
sorry

end value_of_a_l789_789354


namespace shortest_distance_on_ellipse_l789_789004

theorem shortest_distance_on_ellipse {foci_dist major_axis : ℝ} (h1 : foci_dist = 24) (h2 : major_axis = 40) :
  ∃ d, d = 80 :=
by
  have ellipse_property : ∀ P : ℝ × ℝ, dist P (foci_dist / 2, 0) + dist P (-foci_dist / 2, 0) = major_axis, 
  sorry
  use 2 * major_axis
  sorry

end shortest_distance_on_ellipse_l789_789004


namespace sequence_general_term_l789_789051

theorem sequence_general_term (n : ℕ) : 
  (a : ℕ → ℕ), 
  (∀ n, 
    (a 1 = 1) ∧ 
    (a 2 = 3) ∧ 
    (a 3 = 7) ∧ 
    (a 4 = 15) 
    → a n = 2^n - 1) :=
by {
  sorry
}

end sequence_general_term_l789_789051


namespace train_cross_time_l789_789143

/-- Definitions for the ensuring problem --/
def length_of_train : ℝ := 360 -- in meters
def speed_of_train_kmh : ℝ := 216 -- in km/h

/-- Conversion factor from km/h to m/s --/
def kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

/-- Speed of the train in meters per second --/
def speed_of_train_mps : ℝ :=
  kmh_to_mps speed_of_train_kmh

/-- The time it takes for the train to cross the man in seconds --/
theorem train_cross_time
  (L : ℝ := length_of_train)
  (S_kmh : ℝ := speed_of_train_kmh)
  (S_mps : ℝ := kmh_to_mps S_kmh)
  (t : ℝ := L / S_mps) :
  t = 6 :=
sorry

end train_cross_time_l789_789143


namespace shortest_distance_to_parabola_l789_789273

open Real EuclideanGeometry

def parabola (y : ℝ) : ℝ := (y ^ 2) / 4

def point := (4 : ℝ, 10 : ℝ)
def shortest_distance := (41 : ℝ) ^ (1/2 : ℝ)

theorem shortest_distance_to_parabola : 
  ∃ Q : ℝ × ℝ, Q = (parabola 6, 6) ∧ dist point Q = shortest_distance :=
by
  sorry

end shortest_distance_to_parabola_l789_789273


namespace P_barycentric_coordinates_l789_789779

variables {A B C E F P : Type} [AddCommGroup P] [Module ℝ P]
variables (a b c e f : P)

def barycentric_coordinates (A B C P : P) (lambda mu nu : ℝ) : Prop :=
  P = lambda • A + mu • B + nu • C

def AE_over_EC_ratio (A E C : P) (k : ℝ) : Prop :=
  E = (1 / (1 + k)) • A + (k / (1 + k)) • C

def AF_over_FB_ratio (A F B : P) (k : ℝ) : Prop :=
  F = (1 / (1 + k)) • A + (k / (1 + k)) • B

theorem P_barycentric_coordinates 
  (hE : AE_over_EC_ratio a e c (3 / 2)) 
  (hF : AF_over_FB_ratio a f b (2 / 3))
  (hP : barycentric_coordinates e b f (45 / 69) (4 / 69) (20 / 69)) :
  barycentric_coordinates a b c (18 / 69) (4 / 69) (27 / 69) := 
  sorry

end P_barycentric_coordinates_l789_789779


namespace shauna_fifth_test_score_l789_789034

theorem shauna_fifth_test_score :
  ∀ (a1 a2 a3 a4: ℕ), a1 = 76 → a2 = 94 → a3 = 87 → a4 = 92 →
  (∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / 5 = 85 ∧ a5 = 76) :=
by
  sorry

end shauna_fifth_test_score_l789_789034


namespace evaluate_expression_l789_789192

noncomputable def x : ℚ := 4 / 7
noncomputable def y : ℚ := 6 / 8

theorem evaluate_expression : (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end evaluate_expression_l789_789192


namespace problem_l789_789802

def f (x : ℤ) := 3 * x + 2

theorem problem : f (f (f 3)) = 107 := by
  sorry

end problem_l789_789802


namespace measure_angle_XYZ_l789_789373

-- Define the conditions
variable {X Y Z : Type}
variable (XZ n m : X)

-- Conditions: XZ ⊥ n, m || n, ∠ZXY = 105°
def is_perpendicular (a b : X) : Prop := sorry
def is_parallel (a b : X) : Prop := sorry
def angle (a b c : X) : ℝ := sorry

axiom XZ_perpendicular_n : is_perpendicular XZ n
axiom m_parallel_n : is_parallel m n
axiom angle_ZXY : angle Z X Y = 105

-- The statement to prove: ∠XYZ = 15°
theorem measure_angle_XYZ : angle X Y Z = 15 :=
by sorry

end measure_angle_XYZ_l789_789373


namespace central_angle_of_sector_l789_789590

theorem central_angle_of_sector (R θ l : ℝ) (h1 : 2 * R + l = π * R) : θ = π - 2 := 
by
  sorry

end central_angle_of_sector_l789_789590


namespace smallest_y_for_square_l789_789634

theorem smallest_y_for_square (y M : ℕ) (h1 : 2310 * y = M^2) (h2 : 2310 = 2 * 3 * 5 * 7 * 11) : y = 2310 :=
by sorry

end smallest_y_for_square_l789_789634


namespace math_problem_l789_789865

def non_decreasing (f : ℝ → ℝ) (D : set ℝ) : Prop :=
∀ ⦃x1 x2 : ℝ⦄, x1 ∈ D → x2 ∈ D → x1 < x2 → f x1 ≤ f x2

def problem_conditions (f : ℝ → ℝ) : Prop :=
  (non_decreasing f (set.Icc 0 1)) ∧
  (f 0 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x / 3) = (1 / 2 : ℝ) * f x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (1 - x) = 1 - f x)

theorem math_problem (f : ℝ → ℝ) :
  problem_conditions f →
  f (5 / 12) + f (1 / 8) = (3 / 4 : ℝ) :=
sorry

end math_problem_l789_789865


namespace electricity_consumption_scientific_notation_l789_789795

def electricity_consumption (x : Float) : String := 
  let scientific_notation := "3.64 × 10^4"
  scientific_notation

theorem electricity_consumption_scientific_notation :
  electricity_consumption 36400 = "3.64 × 10^4" :=
by 
  sorry

end electricity_consumption_scientific_notation_l789_789795


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789222

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789222


namespace sum_of_interior_angles_of_regular_polygon_with_exterior_72_l789_789743

-- Define the exterior angle and the regular polygon conditions
def exterior_angle (θ : ℝ) := θ = 72
def regular_polygon (θ : ℝ) := exterior_angle θ ∧ ∀ n : ℕ, n = 360 / θ

-- Translate the problem statement into a theorem to prove
theorem sum_of_interior_angles_of_regular_polygon_with_exterior_72 :
  regular_polygon 72 → sum_of_interior_angles 5 = 540 :=
begin
  sorry,
end

-- Function to calculate the sum of interior angles of a polygon.
-- This might be a built-in Lean function, using a placeholder here.
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Directly from the conditions given
example : exterior_angle 72 := by sorry

end sum_of_interior_angles_of_regular_polygon_with_exterior_72_l789_789743


namespace probability_exactly_two_approve_l789_789943

theorem probability_exactly_two_approve (P_A : ℚ) (P_A_value : P_A = 0.6):
  (∃ P : ℚ, P = 0.3456 ∧ P = (choose 4 2) * (P_A^2 * (1 - P_A)^2)) :=
by
  use 0.3456
  simp [P_A_value, choose]
  sorry

end probability_exactly_two_approve_l789_789943


namespace combined_areas_ratio_is_three_over_eight_l789_789813

def ratio_of_combined_areas (r: ℝ) (r1: ℝ) (r2: ℝ) : ℝ :=
  (1/2 * Real.pi * r1^2 + 1/2 * Real.pi * r2^2) / (Real.pi * r^2)

theorem combined_areas_ratio_is_three_over_eight (r : ℝ) (r1 r2 : ℝ)
  (h1 : r1 = r / Real.sqrt 2) (h2 : r2 = r / 2) :
  ratio_of_combined_areas r r1 r2 = 3 / 8 :=
by sorry

end combined_areas_ratio_is_three_over_eight_l789_789813


namespace temperature_value_l789_789746

theorem temperature_value (k : ℝ) (t : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 221) : t = 105 :=
by
  sorry

end temperature_value_l789_789746


namespace sum_g_inv_l789_789178

def g (x : ℝ) : ℝ :=
if x ≤ 3 then 2 * x + 1 else x^2

noncomputable def g_inv (y : ℝ) : ℝ :=
if y ≤ 7 then (y - 1) / 2 else real.sqrt y

theorem sum_g_inv :
  ∑ x in finset.range 29, g_inv (x - 3 : ℝ) = 46.5 :=
by
  sorry

end sum_g_inv_l789_789178


namespace smallest_product_in_num_set_l789_789655

-- Define the set
def num_set : Set ℤ := { -10, -4, -2, 0, 6 }

-- Define a function to find the smallest product of any two distinct elements from the set
def smallest_product (s : Set ℤ) : ℤ :=
  Finset.fold min (0 : ℤ) (Finset.image (Prod.uncurry (*)) ((Finset.powerset s.to_finset).filter (λ t, t.card = 2)))

-- State the theorem to prove
theorem smallest_product_in_num_set : smallest_product num_set = -60 :=
  sorry

end smallest_product_in_num_set_l789_789655


namespace polar_eq_c1_rect_eq_c2_intersection_product_l789_789772

noncomputable def curve_c1_eq : ℝ × ℝ → Prop :=
λ p, let (x, y) := p in x^2 + y^2 - 2 * y = 0

noncomputable def curve_c2_eq_polar : ℝ × ℝ → Prop :=
λ p, let (ρ, θ) := p in ρ * (sin θ)^2 = 4 * cos θ

def ray_eq : ℝ → ℝ → Prop :=
λ x y, y = (3 / 4) * x ∧ x ≥ 0

-- Prove the polar coordinate equation of C1 is ρ = 2 * sin θ
theorem polar_eq_c1 : ∀ (θ : ℝ), curve_c1_eq (ρ * cos θ, ρ * sin θ) → ρ = 2 * sin θ :=
sorry

-- Prove the rectangular coordinate equation of C2 is y^2 = 4 * x
theorem rect_eq_c2 : ∀ (x y : ℝ), curve_c2_eq_polar (ρ, θ) → y^2 = 4 * x :=
sorry

-- Prove |OA| * |OB| = 6
theorem intersection_product :
  ∀ (ρ1 ρ2 θ α : ℝ),
    (sin θ = 4 / 5 ∧ cos θ = 3 / 5) →
    curve_c1_eq (ρ1 * cos θ, ρ1 * sin θ) →
    curve_c2_eq_polar (ρ2, θ) →
    ray_eq (ρ1 * cos θ) (ρ1 * sin θ) →
    ray_eq (ρ2 * cos θ) (ρ2 * sin θ) →
    ρ1 * ρ2 = 6 :=
sorry

end polar_eq_c1_rect_eq_c2_intersection_product_l789_789772


namespace largest_n_l789_789264

def canBeFactored (A B : ℤ) : Bool :=
  A * B = 54

theorem largest_n (n : ℤ) (h : ∃ (A B : ℤ), canBeFactored A B ∧ 3 * B + A = n) :
  n = 163 :=
by
  sorry

end largest_n_l789_789264


namespace min_tangent_length_l789_789692

def point : Type := ℝ × ℝ

def is_symmetric (C : point) (L : ℝ → ℝ → ℝ) := L C.1 C.2 = 0

def circle (c : point) (r : ℝ) : set point :=
  { p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 }

noncomputable def tangent_length (P : point) (C : point) (r : ℝ) : ℝ :=
  real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2 - r^2)

theorem min_tangent_length :
  ∀ (a : ℝ) (b : ℝ), b = -a - 1 → 
  tangent_length (a, b) (1, 2) 1 = real.sqrt 7 := by {
  intros,
  sorry -- Proof is omitted as per instructions
}

end min_tangent_length_l789_789692


namespace round_to_nearest_thousand_scientific_notation_rounding_l789_789032

theorem round_to_nearest_thousand (n : ℤ) (h : n = 130542) :
  (round_to_nearest_thousand n : ℤ) = 131000 := 
sorry

theorem scientific_notation_rounding (n : ℤ) (rounded_val : ℤ)
  (hr : round_to_nearest_thousand n = rounded_val) (hval : rounded_val = 131000) :
  rounded_val = 1.31 * 10^5 :=
sorry

end round_to_nearest_thousand_scientific_notation_rounding_l789_789032


namespace breadth_of_rectangular_plot_l789_789129

theorem breadth_of_rectangular_plot (b l A : ℝ) (h1 : l = 3 * b) (h2 : A = 588) (h3 : A = l * b) : b = 14 :=
by
  -- We start our proof here
  sorry

end breadth_of_rectangular_plot_l789_789129


namespace Tanya_days_to_complete_work_l789_789127

theorem Tanya_days_to_complete_work (sakshi_days : ℕ) (efficiency_increase : ℚ) (tanya_days : ℚ) :
  sakshi_days = 15 → efficiency_increase = 25/100 → tanya_days = 12 :=
by
  intro h_sakshi h_efficiency,
  -- Use the conditions to establish the result
  -- sorry

end Tanya_days_to_complete_work_l789_789127


namespace all_girls_select_same_color_probability_l789_789573

theorem all_girls_select_same_color_probability :
  let white_marbles := 10
  let black_marbles := 10
  let red_marbles := 10
  let girls := 15
  ∀ (total_marbles : ℕ), total_marbles = white_marbles + black_marbles + red_marbles →
  (white_marbles < girls ∧ black_marbles < girls ∧ red_marbles < girls) →
  0 = 0 :=
by
  intros
  sorry

end all_girls_select_same_color_probability_l789_789573


namespace kitchen_clock_real_time_l789_789432

theorem kitchen_clock_real_time :
  ∀ (kitchen_start phone_start real_start : ℝ) (kitchen_end phone_end : ℝ),
    kitchen_start = 8 →
    phone_start = 8 →
    phone_end = 9 →
    kitchen_end = 9 + 10/60 →
    let gain_rate := (kitchen_end - kitchen_start) / (phone_end - phone_start) in
    let duration_real_time := 12 / gain_rate in
    8 + duration_real_time = 18 + 17/60 :=
begin
  intro kitchen_start phone_start real_start kitchen_end phone_end,
  assume h1 h2 h3 h4,
  let gain_rate := (kitchen_end - kitchen_start) / (phone_end - phone_start),
  let duration_real_time := 12 / gain_rate,
  exact calc
  8 + duration_real_time = 8 + 72 / 7 : by sorry 
                      ... = 18 + 17/60 : by sorry
end

end kitchen_clock_real_time_l789_789432


namespace th_eq_sh_div_ch_ch_squared_minus_sh_squared_eq_one_sh_double_angle_identity_derivative_of_sh_tangent_line_at_zero_l789_789050

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2
noncomputable def th (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

theorem th_eq_sh_div_ch (x : ℝ) : th x = sh x / ch x := by
  sorry

theorem ch_squared_minus_sh_squared_eq_one (x : ℝ) : (ch x) ^ 2 - (sh x) ^ 2 = 1 := by
  sorry

theorem sh_double_angle_identity (x : ℝ) : sh (2 * x) = 2 * sh x * ch x := by
  sorry

theorem derivative_of_sh (x : ℝ) : deriv sh x = ch x := by
  sorry

theorem tangent_line_at_zero : ∀ (x : ℝ), (sh'(0) * x) = x := by
  have h1 : sh'(0) = 1 := by
    sorry
  intro x
  rw [h1, mul_one]
  refl

end th_eq_sh_div_ch_ch_squared_minus_sh_squared_eq_one_sh_double_angle_identity_derivative_of_sh_tangent_line_at_zero_l789_789050


namespace polygon_sides_l789_789559

theorem polygon_sides (n : ℕ) (f : ℕ) (h1 : f = n * (n - 3) / 2) (h2 : 2 * n = f) : n = 7 :=
  by
  sorry

end polygon_sides_l789_789559


namespace extraneous_root_equation_l789_789446

-- Given the condition that extraneous root x = 1
theorem extraneous_root_equation (m : ℝ) : 
  (∀ (x : ℝ), x = 1 → (m = x - 3)) → (m = -2) :=
by
  intro h
  have h1 := h 1  (by rfl)
  exact h1

end extraneous_root_equation_l789_789446


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789243

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789243


namespace max_area_OAB_l789_789686

def sequence (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, 2 * a(n + 2) + a(n) = 3 * a(n + 1)

def a : ℕ → ℝ
| 1 := 1
| 2 := 5
| n := 9 - 2^(4 - n)

-- Function representing the area of triangle OAB
def area_OAB (n : ℕ) : ℝ :=
  n * 2^(3 - n)

-- Theorem statement
theorem max_area_OAB (a_seq : ∀ n : ℕ, sequence a) : ∃ n : ℕ, area_OAB n = 4 :=
sorry

end max_area_OAB_l789_789686


namespace OC_in_terms_of_s_and_c_l789_789577

variables {r k θ : ℝ} (s c OC : ℝ)

-- Conditions from part a)
def circle_has_radius_r (O A : ℝ) (r : ℝ) : Prop :=
  (dist O A = r)

def angle_AOB (A O B : Point) : Prop := 
  (angle A O B = k * θ)

def point_C_on_OA (O A C : Point) : Prop :=
  (lies_on_segment O A C)

def BC_bisects_angle_ABO (B C O : Point) : Prop :=
  (bisects_angle B C O)

def s_def : Prop :=
  s = sin(k * θ)

def c_def : Prop :=
  c = cos(k * θ)

-- The proof problem
theorem OC_in_terms_of_s_and_c (O A B C : Point)
  (h1 : circle_has_radius_r O A r)
  (h2 : angle_AOB A O B)
  (h3 : point_C_on_OA O A C)
  (h4 : BC_bisects_angle_ABO B C O)
  (h5 : s_def)
  (h6 : c_def) :
  OC = r / (1 + s) :=
begin
  sorry
end

end OC_in_terms_of_s_and_c_l789_789577


namespace number_of_quarters_l789_789119
-- Definitions of the coin values
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

-- Number of each type of coin used in the proof
variable (pennies nickels dimes quarters half_dollars : ℕ)

-- Conditions from step (a)
axiom one_penny : pennies > 0
axiom one_nickel : nickels > 0
axiom one_dime : dimes > 0
axiom one_quarter : quarters > 0
axiom one_half_dollar : half_dollars > 0
axiom total_coins : pennies + nickels + dimes + quarters + half_dollars = 11
axiom total_value : pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value = 163

-- The conclusion we want to prove
theorem number_of_quarters : quarters = 1 := 
sorry

end number_of_quarters_l789_789119


namespace smallest_integer_value_of_m_l789_789662

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem smallest_integer_value_of_m :
  ∀ m : ℤ, (x^2 + 4 * x - m = 0) ∧ has_two_distinct_real_roots 1 4 (-m : ℝ) → m ≥ -3 :=
by
  intro m h
  sorry

end smallest_integer_value_of_m_l789_789662


namespace isosceles_triangle_l789_789386

noncomputable def triangle_is_isosceles (A B C a b c : ℝ) (h_triangle : a = 2 * b * Real.cos C) : Prop :=
  ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)

theorem isosceles_triangle
  (A B C a b c : ℝ)
  (h_sides : a = 2 * b * Real.cos C)
  (h_triangle : ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)) :
  B = C :=
sorry

end isosceles_triangle_l789_789386


namespace infinite_triangular_square_numbers_l789_789839

theorem infinite_triangular_square_numbers :
  ∃ (S : ℕ → ℕ) (h : ∀ n, (S n) * (S n + 1) / 2 = (S n) ^ 2),
  function.injective S :=
sorry

end infinite_triangular_square_numbers_l789_789839


namespace num_factors_147456_l789_789988

theorem num_factors_147456 : 
  ∃ (n : ℕ), 147456 = 2^14 * 3^2 ∧ 
             (∀ (p : ℕ), p.prime → ∃! (e : ℕ), 147456 % p^e = 0 ∧ 147456 % p^(e+1) ≠ 0) ∧
             (n = ((14 + 1) * (2 + 1))) :=
begin
  use 45,
  split,
  { norm_num, },
  split,
  { intros p hp,
    use (if p = 2 then 14 else if p = 3 then 2 else 0),
    split,
    { by_cases h2 : p = 2,
      { rw h2, norm_num },
      { by_cases h3 : p = 3,
        { rw h3, norm_num },
        { exfalso, 
          -- Remember, no proof steps are required; just assume conditions are correct
          sorry }}},
    { intros q hq,
      split_ifs,
      { rw h, exact nat.succ_pos' },
      { rw h_1, exact nat.succ_pos' },
      { exact h.symm.trans (congr_fun (by simp) q).symm }}},
  sorry
end

end num_factors_147456_l789_789988


namespace eccentricity_of_ellipse_l789_789878

theorem eccentricity_of_ellipse
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∀ x y : ℝ, y = -sqrt 3 * x → (x^2 / a^2 + y^2 / b^2 = 1 → ∃ A B : ℝ × ℝ, (A = (x, y) ∨ B = (x, y)) ∧ ∃ C : ℝ × ℝ, √3*B = AB))
  (h4 : ∃ F : ℝ × ℝ, (dist_focus : F) = (foci (ellipse (a b))) → circle (diameter AB) ∋ F) :
  ecc (ellipse (a b)) = sqrt 3 - 1 :=
by sorry

end eccentricity_of_ellipse_l789_789878


namespace xy_product_condition_l789_789741

theorem xy_product_condition (x y : ℝ) (h : |x - 2| + real.sqrt(y + 3) = 0) : x * y = -6 :=
sorry

end xy_product_condition_l789_789741


namespace product_of_five_integers_negative_probability_l789_789657

def int_set : set ℤ := {-3, -1, -6, 2, 5, 8}

theorem product_of_five_integers_negative_probability :
  (∃ (chosen : finset ℤ), chosen.card = 5 ∧ chosen ⊆ int_set) →
  (∃ (p : ℚ), p = 3 / 5) :=
by
  sorry

end product_of_five_integers_negative_probability_l789_789657


namespace tetrahedron_properties_l789_789049

theorem tetrahedron_properties :
  (∀ T : Type, (is_equilateral T → (is_tetrahedron T → (radius_circumsphere T ≠ (sqrt 3) / 3 ∧ volume T ≠ sqrt 3 ∧ surface_area T ≠ sqrt 6 + sqrt 3 + 1 ∧ surface_area_circumsphere T ≠ (16 * π) / 3)))) :=
by
  sorry

end tetrahedron_properties_l789_789049


namespace problem_solution_l789_789993

def local_value (d : ℕ) (place_value : ℕ) : ℕ := d * place_value
def difference (local_val face_val : ℕ) : ℕ := local_val - face_val

def first_8_local := local_value 8 100_000_000
def second_8_local := local_value 8 10
def the_7_local := local_value 7 10_000_000

def first_8_difference := difference first_8_local 8
def second_8_difference := difference second_8_local 8
def the_7_difference := difference the_7_local 7

def sum_of_differences := first_8_difference + second_8_difference
def result := sum_of_differences * the_7_difference

theorem problem_solution : result = 55_999_994_048_000_192 := by
  sorry

end problem_solution_l789_789993


namespace common_root_for_equations_common_root_value_l789_789624

theorem common_root_for_equations (a b c d : ℝ) (h1 : a + d = 2017) (h2 : b + c = 2017) :
  (a ≠ c) → (b ≠ d) → (x : ℝ) ∃ a b c d, (x - a) * (x - b) = (x - c) * (x - d) := 
begin
  sorry
end

theorem common_root_value :
  ∀ (a b c d : ℝ), (a + d = 2017) → (b + c = 2017) → x = 2017 / 2 :=
begin
  sorry
end

end common_root_for_equations_common_root_value_l789_789624


namespace number_of_valid_r_l789_789421

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = if a n ≤ n then a n + n else a n - n

noncomputable def num_valid_r (n : ℕ) :=
  (n - 2019) / 2

theorem number_of_valid_r :
  ∀ (a : ℕ → ℤ), sequence a → num_valid_r (3 ^ 2017) = (3 ^ 2017 - 2019) / 2 :=
by
  intros
  sorry

end number_of_valid_r_l789_789421


namespace positive_cyclic_sum_l789_789437

theorem positive_cyclic_sum (n : ℕ) (f : Fin n → ℝ) (h : 0 < (Finset.univ : Finset (Fin n)).sum f) :
  ∃ i : Fin n, 0 < (Finset.range n).sum (λ j, f ⟨(i + j) % n, sorry⟩) :=
by
  sorry

end positive_cyclic_sum_l789_789437


namespace min_tan_A_add_2_tan_C_l789_789312

theorem min_tan_A_add_2_tan_C
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : A < π / 2)
  (h3 : B < π / 2)
  (h4 : C < π / 2)
  (h5 : ∃ (R : ℝ), area_seq (R : ℝ) (\(1 / 2) R^2 * (sin (2 * B))\) (\(1 / 2) R^2 * (sin (2 * A))\) (sin (2 * B) = sin (2 * (A+C)))
  : (∃ (x : ℝ), x = 2 * sqrt 6 := sorry

end min_tan_A_add_2_tan_C_l789_789312


namespace mathematicians_graph_theorem_l789_789767

noncomputable def Graph.c (G : Graph) (n : ℕ) : ℕ := -- sorry: placeholder for the actual definition

theorem mathematicians_graph_theorem (G : Graph)
  (H1 : ∀ (cycle : list (Vertex G)), cycle.length ≥ 4 → ∃ (a b : Vertex G), a ≠ b ∧ a ∈ cycle ∧ b ∈ cycle ∧ (a, b) ∈ edge_set G)
  (H2 : connected G) :
  (Graph.c G 1) - (Graph.c G 2) + (Graph.c G 3) - (Graph.c G 4) + ... = 1 := sorry

end mathematicians_graph_theorem_l789_789767


namespace number_of_solutions_sin_abs_eq_abs_cos_l789_789883

theorem number_of_solutions_sin_abs_eq_abs_cos (n : ℕ) :
  (∀ x : ℝ, x ∈ [-10 * Real.pi, 10 * Real.pi] → sin (|x|) = |cos x|) → n = 20 :=
sorry

end number_of_solutions_sin_abs_eq_abs_cos_l789_789883


namespace parabola_properties_l789_789375

noncomputable def parabola : (ℝ → ℝ) := λ x, x^2 - 4 * x - 5

theorem parabola_properties (a b : ℝ) 
  (h_parabola : ∀ x, parabola x = x^2 - 4 * x - 5 ) 
  (h_vertex : ∃ M : ℝ × ℝ, M = (2, -9) )
  (PointP : ℝ → ℝ)
  (PointQ : ℝ → ℝ)
  (P_in_axis_symmetry : ∀ m, PointP = (2, m) )
  (Q_on_x_axis : ∃ n, ∀ x, PointQ x = (n, 0) )
  (PQ_PC_orthogonal : ∀ P, ∀ Q, P =  (2, _), ∃ n, n = (P, Q))
  (P_moves_on_MN : ∀l m, ( -9 ≤ m ∧ m ≤ 0 )) :
  ( ∃ eq : (ℝ → ℝ), eq = parabola) ∧
  ( ∃ Mco : (ℝ × ℝ), Mco = (2, -9) ) ∧
  ( ∃ range_n : ( ℝ × ℝ), range_n = (-9 / 8, 20) ) :=
by {
  use parabola,
  use (2, -9),
  use (-9 / 8, 20),
  sorry
}

end parabola_properties_l789_789375


namespace find_omega_find_minimum_l789_789710

noncomputable def f (ω x : ℝ) : ℝ :=
  sin (π - ω * x) * cos (ω * x) + cos (ω * x)^2

theorem find_omega (ω : ℝ) (hω : ω > 0) (hf : is_periodic (f ω) π) : ω = 1 :=
sorry

noncomputable def g (x : ℝ) : ℝ :=
  f 1 (2 * x)

theorem find_minimum (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 16) : g x = 1 :=
sorry

end find_omega_find_minimum_l789_789710


namespace find_phi_l789_789715

theorem find_phi 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + Real.pi / 6))
  (h2 : 0 < phi ∧ phi < Real.pi / 2)
  (h3 : ∀ x, y x = f (x - phi) ∧ y x = y (-x)) :
  phi = Real.pi / 3 :=
by
  sorry

end find_phi_l789_789715


namespace polynomial_divisors_l789_789797

noncomputable def exists_n_k (P : ℝ[X]) : Prop :=
  ∃ (n k : ℕ), (Int.log10 (k : ℝ)).ceil = n ∧ (k.factors.length > P.eval n)

theorem polynomial_divisors (P : ℝ[X]) : exists_n_k P :=
  sorry

end polynomial_divisors_l789_789797


namespace cashier_overestimation_l789_789931

def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

def nickels_counted_as_dimes := 15
def quarters_counted_as_half_dollars := 10

noncomputable def overestimation_due_to_nickels_as_dimes : Nat := 
  (dime_value - nickel_value) * nickels_counted_as_dimes

noncomputable def overestimation_due_to_quarters_as_half_dollars : Nat := 
  (half_dollar_value - quarter_value) * quarters_counted_as_half_dollars

noncomputable def total_overestimation : Nat := 
  overestimation_due_to_nickels_as_dimes + overestimation_due_to_quarters_as_half_dollars

theorem cashier_overestimation : total_overestimation = 325 := by
  sorry

end cashier_overestimation_l789_789931


namespace triangle_XYZ_properties_l789_789040

theorem triangle_XYZ_properties
  (X Y Z : Type) [inner_product_space ℝ X] [finite_dimensional ℝ X]
  (XYZ_triangle : triangle X Y Z)
  (right_angle_Y : angle Y = 90)
  (cos_X : cos (angle X) = 3 / 5)
  (YZ_len : dist Y Z = 15) :
  (dist X Y = 12) ∧ (1 / 2 * dist X Z * dist Y Z = 67.5) :=
by
  sorry

end triangle_XYZ_properties_l789_789040


namespace bus_probability_l789_789166

theorem bus_probability (buses : Finset ℕ) (wanted_buses : Finset ℕ) (h_buses: buses = {1, 3, 5, 8}) (h_wanted: wanted_buses = {1, 3}) :
  (wanted_buses.card / buses.card : ℚ) = 1 / 2 :=
by
  have h_card_buses : buses.card = 4 := by rw [h_buses]; simp
  have h_card_wanted : wanted_buses.card = 2 := by rw [h_wanted]; simp
  rw [h_card_wanted, h_card_buses]
  norm_num
  sorry

end bus_probability_l789_789166


namespace number_properties_l789_789583

theorem number_properties : 
    ∃ (N : ℕ), 
    35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 :=
by 
  sorry

end number_properties_l789_789583


namespace sum_inequality_l789_789837

theorem sum_inequality (t : Fin 5 → ℝ) :
  (∑ j in Finset.range 5, (1 - t j) * Real.exp (∑ k in Finset.range (j + 1), t k)) ≤ Real.exp (Real.exp (Real.exp (Real.exp 1))) := 
sorry

end sum_inequality_l789_789837


namespace danny_reach_stevens_house_l789_789627

theorem danny_reach_stevens_house (t : ℝ) :
  (∀ t : ℝ, (∀ t : ℝ, t = 25 ↔ (∀ t1 t2 : ℝ, (t = t1 / 2 ∧ t1 = 2 * t ∧ t - t / 2 = 12.5) → t = 25))) :=
by 
  intro t,
  split,
  { intro h, assumption },
  { intro h, sorry }

end danny_reach_stevens_house_l789_789627


namespace b_alone_work_days_l789_789919

theorem b_alone_work_days 
  (W : ℝ)
  (R_A R_B : ℝ)
  (h1 : R_A + R_B = W / 10)
  (h2 : R_A = W / 20) :
  R_B = W / 20 :=
by 
  calc 
    R_B = W / 10 - W / 20 : by linarith 
    ... = (2 * W) / 20 - (W) / 20 : by congr; ring 
    ... = W / 20 : by ring

end b_alone_work_days_l789_789919


namespace utility_bills_total_correct_l789_789019

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l789_789019


namespace inequality_lemma_l789_789810

theorem inequality_lemma (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z)
    (h2 : (1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1)) :
    (1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1) := 
by
  sorry

end inequality_lemma_l789_789810


namespace sam_final_amount_l789_789033

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def initial_amount : ℕ := (initial_dimes * dime_value) + (initial_quarters * quarter_value) + (initial_nickels * nickel_value)
def dad_amount : ℕ := (dad_dimes * dime_value) + (dad_quarters * quarter_value)
def mom_amount : ℕ := (mom_nickels * nickel_value) + (mom_dimes * dime_value)

def final_amount : ℕ := initial_amount + dad_amount - mom_amount

theorem sam_final_amount : final_amount = 325 := by
  sorry

end sam_final_amount_l789_789033


namespace line_passes_through_fixed_point_l789_789823

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (x = -1 ∧ y = 3) ∧ (m * x - y + 3 + m = 0) :=
by
  intro m
  use (-1, 3)
  simp
  sorry

end line_passes_through_fixed_point_l789_789823


namespace find_largest_integer_l789_789248

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789248


namespace combined_weight_l789_789609

-- Definition of conditions
def regular_dinosaur_weight := 800
def five_regular_dinosaurs_weight := 5 * regular_dinosaur_weight
def barney_weight := five_regular_dinosaurs_weight + 1500

-- Statement to prove
theorem combined_weight (h1: five_regular_dinosaurs_weight = 5 * regular_dinosaur_weight)
                        (h2: barney_weight = five_regular_dinosaurs_weight + 1500) : 
        (barney_weight + five_regular_dinosaurs_weight = 9500) :=
by
    sorry

end combined_weight_l789_789609


namespace elective_courses_count_l789_789009

theorem elective_courses_count :
  let courses := 4 in
  let max_courses_per_year := 3 in
  let years := 3 in
  let ways_to_group :=
    (Nat.choose courses 2) + (Nat.choose courses 2 / 2) + (Nat.choose courses 3) in
  let ways_to_arrange := Nat.factorial years in
  (ways_to_group * ways_to_arrange) = 78 := by
  let courses := 4
  let max_courses_per_year := 3
  let years := 3
  let ways_to_group := (Nat.choose courses 2) + (Nat.choose courses 2 / 2) + (Nat.choose courses 3)
  let ways_to_arrange := Nat.factorial years
  show (ways_to_group * ways_to_arrange) = 78
  sorry

end elective_courses_count_l789_789009


namespace average_people_per_hour_l789_789770

theorem average_people_per_hour :
  let people := 3000
  let days := 4
  let hours := 24 * days
  let average := people / (hours: ℝ)
  let rounded_average := Real.floor (average + 0.5)
  rounded_average = 31 :=
by
  let people := 3000
  let days := 4
  let hours := 24 * days
  let average := people / (hours: ℝ)
  let rounded_average := Real.floor (average + 0.5)
  show rounded_average = 31
  -- Proof is skipped
  sorry

end average_people_per_hour_l789_789770


namespace unique_solution_for_cubed_root_function_l789_789445

theorem unique_solution_for_cubed_root_function (y : ℝ) : 
  (∃! y, ∛(30 * y + ∛(30 * y + 25)) = 15) → y = 335 / 3 :=
by
  sorry

end unique_solution_for_cubed_root_function_l789_789445


namespace product_xyz_42_l789_789696

theorem product_xyz_42 (x y z : ℝ) 
  (h1 : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (h2 : x + y + z = 12) : x * y * z = 42 :=
by
  sorry

end product_xyz_42_l789_789696


namespace find_intersection_sums_l789_789052

noncomputable def y1 := x_1^3 - 4 * x_1^2 + 3 * x_1 + 2
noncomputable def y2 := x_2^3 - 4 * x_2^2 + 3 * x_2 + 2
noncomputable def y3 := x_3^3 - 4 * x_3^2 + 3 * x_3 + 2

def intersects_graphs (x y : ℝ) : Prop :=
  y = x^3 - 4 * x^2 + 3 * x + 2 ∧ 2 * x + 4 * y = 8

theorem find_intersection_sums :
  ∃ (x_1 x_2 x_3 y_1 y_2 y_3: ℝ), intersects_graphs x_1 y1 ∧
                                 intersects_graphs x_2 y2 ∧
                                 intersects_graphs x_3 y3 ∧
                                 y1 = 2 - x_1 / 2 ∧
                                 y2 = 2 - x_2 / 2 ∧
                                 y3 = 2 - x_3 / 2 ∧
                                 (x_1 + x_2 + x_3 = 4) ∧
                                 (y1 + y2 + y3 = 4) :=
sorry

end find_intersection_sums_l789_789052


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789242

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789242


namespace tim_sarah_age_ratio_l789_789658

theorem tim_sarah_age_ratio :
  ∀ (x : ℕ), ∃ (t s : ℕ),
    t = 23 ∧ s = 11 ∧
    (23 + x) * 2 = (11 + x) * 3 → x = 13 :=
by
  sorry

end tim_sarah_age_ratio_l789_789658


namespace zongzi_profit_l789_789855

def initial_cost : ℕ := 10
def initial_price : ℕ := 16
def initial_bags_sold : ℕ := 200
def additional_sales_per_yuan (x : ℕ) : ℕ := 80 * x
def profit_per_bag (x : ℕ) : ℕ := initial_price - x - initial_cost
def number_of_bags_sold (x : ℕ) : ℕ := initial_bags_sold + additional_sales_per_yuan x
def total_profit (profit_per_bag : ℕ) (number_of_bags_sold : ℕ) : ℕ := profit_per_bag * number_of_bags_sold

theorem zongzi_profit (x : ℕ) : 
  total_profit (profit_per_bag x) (number_of_bags_sold x) = 1440 := 
sorry

end zongzi_profit_l789_789855


namespace assoc_mul_l789_789911

-- Conditions from the problem
variables (x y z : Type) [Mul x] [Mul y] [Mul z]

theorem assoc_mul (a b c : x) : (a * b) * c = a * (b * c) := by sorry

end assoc_mul_l789_789911


namespace tangent_line_B_E_l789_789796

theorem tangent_line_B_E
  (A B C D E : Point)
  (Gamma : Circle)
  (h_isosceles : dist A B = dist A C)
  (h_circle_tangent : tangent_to_line_at Gamma A C C)
  (h_point_on_circle : on_circle D Gamma)
  (h_circumcircle_tangent : tangent_to_circle_at (Circumcircle A B D) Gamma D)
  (h_segment_intersect : intersects_at (Line A D) Gamma E D) :
  tangent_to_line_at Gamma B E E :=
sorry

end tangent_line_B_E_l789_789796


namespace a6_equals_8_l789_789380

-- Defining Sn as given in the condition
def S (n : ℕ) : ℤ :=
  if n = 0 then 0
  else n^2 - 3*n

-- Defining a_n in terms of the differences stated in the solution
def a (n : ℕ) : ℤ := S n - S (n-1)

-- The problem statement to prove
theorem a6_equals_8 : a 6 = 8 :=
by
  sorry

end a6_equals_8_l789_789380


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789245

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789245


namespace katya_sequences_l789_789393

def is_valid_digit_replacement (original : ℕ) (left_neigh : Option ℕ) (right_neigh : Option ℕ) : ℕ :=
  (if left_neigh.isSome ∧ left_neigh.get < original then 1 else 0) +
  (if right_neigh.isSome ∧ right_neigh.get < original then 1 else 0)

def valid_transformation_sequence (seq : List ℕ) : Bool :=
  seq.length = 10 ∧
  ∀ i, i < 10 -> 
    let left_neigh := if i > 0 then some (seq.get ⟨i - 1, by linarith⟩) else none;
    let right_neigh := if i < 9 then some (seq.get ⟨i + 1, by linarith⟩) else none;
    seq.get ⟨i, by linarith⟩ = is_valid_digit_replacement(i, left_neigh, right_neigh)

theorem katya_sequences :
  valid_transformation_sequence [1, 1, 0, 1, 1, 1, 1, 1, 1, 1] ∧
  ¬valid_transformation_sequence [1, 2, 0, 1, 2, 0, 1, 0, 2, 0] ∧
  valid_transformation_sequence [1, 0, 2, 1, 0, 2, 1, 0, 2, 0] ∧
  valid_transformation_sequence [0, 1, 1, 2, 1, 0, 2, 0, 1, 1] := 
by {
  sorry
}

end katya_sequences_l789_789393


namespace correct_proposition_statement_l789_789913

theorem correct_proposition_statement :
  let p := ∀ x : ℝ, log 2 x > 0 in
  (¬ (∃ x₀ : ℝ, log 2 x₀ ≤ 0) ↔ ∀ x : ℝ, log 2 x > 0) :=
by
  sorry

end correct_proposition_statement_l789_789913


namespace elective_ways_l789_789007

theorem elective_ways (students_courses : ℕ → ℕ) (n_courses_per_year : ℕ) : 
  (∀ n, students_courses n ≤ 3) ∧ 
  (∑ i in Finset.Icc 1 3, students_courses i = 4) → 
  78 = 78 :=
by
  sorry

end elective_ways_l789_789007


namespace vector_subtraction_correct_dot_product_correct_l789_789729

variables (a b : ℝ × ℝ)

def vector_a := (1, 2)
def vector_b := (3, 1)

-- Proof that 2a - b = (-1, 3)
theorem vector_subtraction_correct : 2 • vector_a - vector_b = (-1, 3) := 
  sorry

-- Proof that a • b = 5
theorem dot_product_correct : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 5 := 
  sorry

end vector_subtraction_correct_dot_product_correct_l789_789729


namespace emily_sold_toys_l789_789640

theorem emily_sold_toys (initial_toys : ℕ) (remaining_toys : ℕ) (sold_toys : ℕ) 
  (h_initial : initial_toys = 7) 
  (h_remaining : remaining_toys = 4) 
  (h_sold : sold_toys = initial_toys - remaining_toys) :
  sold_toys = 3 :=
by sorry

end emily_sold_toys_l789_789640


namespace sum_inverse_roots_l789_789400

theorem sum_inverse_roots (a : Fin 2018 → ℝ) :
  (∀ x : ℝ, (∃ n : Fin 2018, x = a n) ↔ x^2018 + 2 * x^2017 + ... + 3 * x^2 + 4 * x = 2000) →
  ∑ i, 1 / (1 - a i) = 4030 / 1999 := by
  intros h
  sorry

end sum_inverse_roots_l789_789400


namespace range_m_l789_789690

noncomputable def circle_c (x y : ℝ) : Prop := (x - 4) ^ 2 + (y - 3) ^ 2 = 4

def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

theorem range_m (m : ℝ) (P : ℝ × ℝ) :
  circle_c P.1 P.2 ∧ m > 0 ∧ (∃ (a b : ℝ), P = (a, b) ∧ (a + m) * (a - m) + b ^ 2 = 0) → m ∈ Set.Icc 3 7 :=
sorry

end range_m_l789_789690


namespace part_a_l789_789843

def isCyclicTriplet (A B C : Player) [Arena] : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  wins A B ∧ wins B C ∧ wins C A

def canPartition (players : List Player) [Arena] : Prop :=
  ∃ (room1 room2 : List Player), 
    (∀ {A B C : Player}, A ∈ room1 ∧ B ∈ room1 ∧ C ∈ room1 → ¬ isCyclicTriplet A B C) ∧ 
    (∀ {A B C : Player}, A ∈ room2 ∧ B ∈ room2 ∧ C ∈ room2 → ¬ isCyclicTriplet A B C)

theorem part_a (players : List Player) [Arena] (h_len : players.length = 6) :
  canPartition players :=
sorry

end part_a_l789_789843


namespace utility_bill_amount_l789_789013

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l789_789013


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l789_789535

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l789_789535


namespace apple_ratios_l789_789496

theorem apple_ratios (R G Y : ℕ) (h1 : G = R + 12) (h2 : Y = G + 18) (h3 : R = 16) :
  R : G : Y = 16 : 28 : 46 := by
  sorry

end apple_ratios_l789_789496


namespace find_point_P_l789_789890

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def isEquidistant (p1 p2 : Point3D) (q : Point3D) : Prop :=
  (q.x - p1.x)^2 + (q.y - p1.y)^2 + (q.z - p1.z)^2 = (q.x - p2.x)^2 + (q.y - p2.y)^2 + (q.z - p2.z)^2

theorem find_point_P (P : Point3D) :
  (∀ (Q : Point3D), isEquidistant ⟨2, 3, -4⟩ P Q → (8 * Q.x - 6 * Q.y + 18 * Q.z = 70)) →
  P = ⟨6, 0, 5⟩ :=
by 
  sorry

end find_point_P_l789_789890


namespace number_of_factors_l789_789344

theorem number_of_factors : 
  let n := 4^5 * 5^2 * 6^3 in
  ∃ (count : ℕ), count = 168 ∧ ∀ f : ℕ, f ∣ n → (f = 2^a * 3^b * 5^c ∧ 0 ≤ a ∧ a ≤ 13 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 2) :=
begin
  sorry
end

end number_of_factors_l789_789344


namespace CK_parallel_AB_l789_789053

variables {A B C A₁ B₁ C₁ I K : Point}
variable {ABC : Triangle A B C}

-- Conditions given
def incircle_touches_bc_ca_ab_at_A₁_B₁_C₁ (incircle : Circle) : Prop :=
  incircle.touches BC A₁ ∧ incircle.touches CA B₁ ∧ incircle.touches AB C₁

def is_incenter_of_triangle (I : Point) (ABC : Triangle A B C) : Prop :=
  I = incenter ABC

def line_is_perpendicular_to_median_from_vertex_C (l : Line) (ABC : Triangle A B C) : Prop :=
  ∃ M : Point, M = midpoint B C ∧ l = Line.perpendicular_through_point I M

def point_K_is_intersection_of_perpendicular_and_A₁B₁ (I : Point) (l : Line) (K : Point) (A₁ B₁ : Point) : Prop :=
  ∃ p : Line, p = Line.through I K ∧ Line.intersection p (Line.through A₁ B₁) K

-- Proof statement
theorem CK_parallel_AB (incircle : Circle) (l : Line)
  (h1 : incircle_touches_bc_ca_ab_at_A₁_B₁_C₁ incircle)
  (h2 : is_incenter_of_triangle I ABC)
  (h3 : line_is_perpendicular_to_median_from_vertex_C l ABC)
  (h4 : point_K_is_intersection_of_perpendicular_and_A₁B₁ I l K A₁ B₁) :
  parallel (Line.through C K) (Line.through A B) :=
sorry

end CK_parallel_AB_l789_789053


namespace passing_percentage_is_30_l789_789960

-- Define variables and conditions as given in the problem statement
def student_marks : ℝ := 80
def fail_by_marks : ℝ := 10
def max_marks : ℝ := 300

-- Calculate the marks required to pass
def passing_marks : ℝ := student_marks + fail_by_marks

-- Calculate the percentage of marks required to pass
def passing_percentage : ℝ := (passing_marks / max_marks) * 100

-- State the theorem: The percentage of marks needed to pass the test is 30%
theorem passing_percentage_is_30 : passing_percentage = 30 := 
by 
  sorry

end passing_percentage_is_30_l789_789960


namespace LCM_GCD_even_nonnegative_l789_789439

theorem LCM_GCD_even_nonnegative (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  : ∃ (n : ℕ), (n = Nat.lcm a b + Nat.gcd a b - a - b) ∧ (n % 2 = 0) ∧ (0 ≤ n) := 
sorry

end LCM_GCD_even_nonnegative_l789_789439


namespace integral_sin_squared_diverge_l789_789783

theorem integral_sin_squared_diverge :
  ¬convergent (∫ x in (Set.Ioi 1), (sin x) ^ 2 / x) :=
sorry

end integral_sin_squared_diverge_l789_789783


namespace max_m_f_has_P_k_l789_789704

noncomputable def f : ℝ → ℝ 
| x := if x ≤ 1/4 then -4*x + 1 
       else if x < 3/4 then 4*x - 1 
       else -4*x + 5

def P (m : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x0 ∈ Set.Icc 0 (1 - m), f x0 = f (x0 + m)

-- Problem 1
theorem max_m (h : ContinuousOn f (Set.Icc 0 1)) : ∃ x, ∀ m, P m f ↔ m ≤ 1/2 := 
sorry

-- Problem 2
theorem f_has_P_k (h : f 0 = f 1) (k : ℕ) (hk : 2 ≤ k) : P (1/k) f := 
sorry

end max_m_f_has_P_k_l789_789704


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789211

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789211


namespace option_two_not_binomial_l789_789138

section
  variables {n : ℕ} {p : ℝ}

  -- Conditions for the distributions
  def computer_virus_distribution (X : ℕ → Prop) : Prop :=
    ∃ (n : ℕ), ∀ (X : ℕ), X = binomial n 0.65

  def first_hit_distribution (X : ℕ → Prop) : Prop :=
    ∀ (X : ℕ), X = geometric p

  def target_hits_distribution (X : ℕ → Prop) : Prop :=
    ∃ (n : ℕ), ∀ (X : ℕ), X = binomial n p

  def refueling_cars_distribution (X : ℕ → Prop) : Prop :=
    ∃ (k : ℕ), k = 50 ∧ ∀ (X : ℕ), X = binomial 50 0.6

  theorem option_two_not_binomial :
    (∃ (X : ℕ → Prop), computer_virus_distribution X) ∧
    (∃ (X : ℕ → Prop), first_hit_distribution X) ∧ 
    (∃ (X : ℕ → Prop), target_hits_distribution X) ∧ 
    (∃ (X : ℕ → Prop), refueling_cars_distribution X) →
    ∀ X, first_hit_distribution X → ¬ (∃ (X : ℕ → Prop), binomial X)
  :=
  sorry
end

end option_two_not_binomial_l789_789138


namespace radius_of_inscribed_circle_l789_789936

/-- Given a circular sector with a central angle of 120 degrees
    and an original circle with radius R, the radius of the inscribed circle is 
    sqrt(3) * R * (2 - sqrt(3)) -/
theorem radius_of_inscribed_circle (R : ℝ) : 
  let r := (√3 * R * (2 - √3)) 
  in r = sqrt(3) * R * (2 - sqrt(3)) :=
by sorry

end radius_of_inscribed_circle_l789_789936


namespace symmetry_about_2024_pi_l789_789717

def f (x : ℝ) : ℝ := |Real.tan x|

theorem symmetry_about_2024_pi :
  ∀ x : ℝ, f(2024 * Real.pi - x) = f(2024 * Real.pi + x) :=
by sorry

end symmetry_about_2024_pi_l789_789717


namespace greatest_integer_gcd_30_is_125_l789_789528

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l789_789528


namespace greatest_int_with_gcd_five_l789_789538

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l789_789538


namespace expression_undefined_l789_789664

theorem expression_undefined (a : ℝ) : a = 3 ∨ a = -3 ↔ a^2 - 9 = 0 :=
by
  sorry

end expression_undefined_l789_789664


namespace right_triangle_angle_bisector_radius_condition_l789_789985

theorem right_triangle_angle_bisector_radius_condition
  (f varrho : ℝ) : f > sqrt (8 * varrho) :=
sorry

end right_triangle_angle_bisector_radius_condition_l789_789985


namespace positive_integer_solution_inequality_l789_789060

theorem positive_integer_solution_inequality (x : ℕ) (h : 2 * (x + 1) ≥ 5 * x - 3) : x = 1 :=
by {
  sorry
}

end positive_integer_solution_inequality_l789_789060


namespace minimum_value_PF_PQ_l789_789678

noncomputable def hyperbola_eq (h a b : ℝ) : Prop :=
  h = (λ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ a > 0 ∧ b > 0)

noncomputable def parabola_eq : ℝ → ℝ → Prop :=
  λ x y : ℝ, y^2 = 8 * x 

noncomputable def asymptote_distance (f : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (x, y) := f in x = 2 ∧ y = 0 ∧ (2 * b / real.sqrt (b^2 + a^2) = 1)

noncomputable def min_PF_PQ (h : ℝ → ℝ → Prop) (f : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ :=
  let F := (λ x y : ℝ, (x, y) = f) in 2 * real.sqrt 3 + real.sqrt ((F.1 - Q.1)^2 + (F.2 - 3^2))

theorem minimum_value_PF_PQ (a b : ℝ) 
  (C1 : ∀ x y, hyperbola_eq (λ x y, (x, y)) a b)
  (H : ∀ x y, parabola_eq x y → parabola_eq 2 0)
  (D : ∀ a b, asymptote_distance (2, 0) a b) : 
  (min_PF_PQ (λ x y, hyperbola_eq (x, y) a b) (2, 0) (1, 3) = 2 * real.sqrt 3 + 3 * real.sqrt 2) :=
sorry

end minimum_value_PF_PQ_l789_789678


namespace chosen_numbers_sum_l789_789080

theorem chosen_numbers_sum
  (A B : Finset ℕ)
  (hA : A.card = 25)
  (hB : B.card = 25)
  (hA_sub : ∀ a ∈ A, 1 ≤ a ∧ a ≤ 50)
  (hB_sub : ∀ b ∈ B, 51 ≤ b ∧ b ≤ 100)
  (h_diff : ∀ a ∈ A, ∀ b ∈ B, a ≠ b - 50) :
  (∑ x in A, x) + (∑ y in B, y) = 2525 := 
sorry

end chosen_numbers_sum_l789_789080


namespace largest_integer_less_than_100_with_remainder_4_l789_789205

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789205


namespace line_intersects_circle_but_not_through_center_l789_789886

noncomputable def distance_from_point_to_line (point : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x, y) := point in
  (abs (a * x + b * y + c)) / (sqrt (a^2 + b^2))

theorem line_intersects_circle_but_not_through_center :
  let center := (2, 1)
  let r := 5
  let d := distance_from_point_to_line center 3 4 10
  0 < d ∧ d < r :=
by
  let center := (2, 1)
  let r := 5
  let d := distance_from_point_to_line center 3 4 10
  have h1 : d = 4 := sorry
  have h2 : 0 < 4 := by norm_num
  have h3 : 4 < 5 := by norm_num
  exact ⟨h2, h3⟩

end line_intersects_circle_but_not_through_center_l789_789886


namespace smallest_positive_integer_99_l789_789274

theorem smallest_positive_integer_99 :
  ∃ (M : ℕ), 
    (M > 0) ∧
    (
      ((M % 4 = 0 ∧ (M + 1) % 9 = 0 ∧ (M + 2) % 25 = 0) ∨
      ((M + 1) % 4 = 0 ∧ (M + 2) % 9 = 0 ∧ M % 25 = 0) ∨
      ((M + 2) % 4 = 0 ∧ M % 9 = 0 ∧ (M + 1) % 25 = 0)
    ) ∧
    M = 99 :=
by 
  -- Proof starts here
  sorry

end smallest_positive_integer_99_l789_789274


namespace parabola_focus_coords_l789_789860

theorem parabola_focus_coords :
  ∀ (x y : ℝ), y^2 = -4 * x → (x, y) = (-1, 0) :=
by
  intros x y h
  sorry

end parabola_focus_coords_l789_789860


namespace abs_difference_x_y_l789_789415

variable (x y : ℝ)

-- Define the floor and fractional parts of x and y
def floor_x := Real.floor x
def frac_y := x - Real.floor x
def frac_x := y - Real.floor y
def floor_y := Real.floor y

-- Conditions
axiom h1 : floor_x + frac_y = 3.5
axiom h2 : frac_x + floor_y = 6.2

-- Theorem proving the absolute difference between x and y
theorem abs_difference_x_y : |x - y| = 3.3 :=
by
  sorry

end abs_difference_x_y_l789_789415


namespace partitions_distinct_parts_eq_odd_parts_l789_789441

def num_partitions_into_distinct_parts (n : ℕ) : ℕ := sorry
def num_partitions_into_odd_parts (n : ℕ) : ℕ := sorry

theorem partitions_distinct_parts_eq_odd_parts (n : ℕ) :
  num_partitions_into_distinct_parts n = num_partitions_into_odd_parts n :=
  sorry

end partitions_distinct_parts_eq_odd_parts_l789_789441


namespace solution_problem_l789_789314

noncomputable def problem :=
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 →
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c)

theorem solution_problem : problem :=
  sorry

end solution_problem_l789_789314


namespace sum_mod_17_l789_789171

theorem sum_mod_17 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 17 = 2 :=
by
  sorry

end sum_mod_17_l789_789171


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789236

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789236


namespace max_n_convex_ngon_l789_789665

theorem max_n_convex_ngon (n : ℕ) (polygon : Type) [convex polygon] (side : ℕ) (diagonals : polygon → polygon → ℕ) 
  (h_side : side = 1) (h_diagonals : ∀ (A B : polygon), A ≠ B → ∃ k : ℕ, diagonals A B = k) : n ≤ 5 := 
sorry

end max_n_convex_ngon_l789_789665


namespace general_formula_a_S_n_no_arithmetic_sequence_in_b_l789_789684

def sequence_a (a : ℕ → ℚ) :=
  (a 1 = 1 / 4) ∧ (∀ n : ℕ, n > 0 → 3 * a (n + 1) - 2 * a n = 1)

def sequence_b (b : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n

theorem general_formula_a_S_n (a : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence_a a →
  (∀ n : ℕ, n > 0 → a n = 1 - (3 / 4) * (2 / 3)^(n - 1)) →
  (∀ n : ℕ, n > 0 → S n = (2 / 3)^(n - 2) + n - 9 / 4) →
  True := sorry

theorem no_arithmetic_sequence_in_b (b : ℕ → ℚ) (a : ℕ → ℚ) :
  sequence_b b a →
  (∀ n : ℕ, n > 0 → b n = (1 / 4) * (2 / 3)^(n - 1)) →
  (∀ r s t : ℕ, r < s ∧ s < t → ¬ (b s - b r = b t - b s)) :=
  sorry

end general_formula_a_S_n_no_arithmetic_sequence_in_b_l789_789684


namespace equilateral_triangles_is_a_set_l789_789965

-- Defining the conditions as predicates
def is_definite (S : Type) [Set S] : Prop :=
  ∀ x y : S, x = y ∨ x ≠ y  -- A basic condition for definiteness, unorderedness, and distinctness

def large_numbers : Set ℕ := {n : ℕ | n > 10^100}
def smart_people : Set string := {s : string | s = "intelligent"}
def difficult_problems : Set string := {s : string | s = "difficult"}
def equilateral_triangles : Set (ℝ × ℝ × ℝ) := {t : ℝ × ℝ × ℝ | t.1 = t.2 ∧ t.2 = t.3}

-- The task is to prove that equilateral_triangles can form a set under the given conditions
theorem equilateral_triangles_is_a_set : is_definite (ℝ × ℝ × ℝ) := 
  sorry -- Proof is omitted

end equilateral_triangles_is_a_set_l789_789965


namespace Fm_plus_Fn_positive_l789_789334

theorem Fm_plus_Fn_positive (a m n : ℝ)
  (hf_even : ∀ x : ℝ, f x = f (-x))
  (hm_pos : m > 0)
  (hn_neg : n < 0)
  (hmn_pos : m + n > 0)
  (ha_pos : a > 0)
  (hf_def : ∀ x : ℝ, f x = a * x ^ 2 + b * x + 1)
  (F_def : ∀ x : ℝ, F x = if x > 0 then f x else -f x) :
  F m + F n > 0 := sorry

end Fm_plus_Fn_positive_l789_789334


namespace sum_of_first_10_terms_arithmetic_sequence_l789_789070

theorem sum_of_first_10_terms_arithmetic_sequence :
  let a : ℕ → ℤ := λ n, 1 + (n - 1) * (-4)
  let S : ℕ → ℤ := λ n, n * (1 + a n) / 2
  S 10 = -170 :=
by
  let a : ℕ → ℤ := λ n, 1 + (n - 1) * (-4)
  let S : ℕ → ℤ := λ n, n * (1 + a n) / 2
  specialize S 10
  sorry

end sum_of_first_10_terms_arithmetic_sequence_l789_789070


namespace sum_of_possible_values_of_g1_l789_789412

noncomputable def g : ℝ → ℝ := sorry -- Assume g is a non-constant quadratic polynomial that we need to define

theorem sum_of_possible_values_of_g1 :
  (∀ x : ℝ, x ≠ 0 → g(x - 1) + g(x) + g(x + 1) = g(x)^2 / (2023 * x)) →
  g(1) = 3 :=
by
  sorry

end sum_of_possible_values_of_g1_l789_789412


namespace even_function_a_value_monotonicity_on_neg_infinity_l789_789318

noncomputable def f (x a : ℝ) : ℝ := ((x + 1) * (x + a)) / (x^2)

-- (1) Proving f(x) is even implies a = -1
theorem even_function_a_value (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = -1 :=
by
  sorry

-- (2) Proving monotonicity on (-∞, 0) for f(x) with a = -1
theorem monotonicity_on_neg_infinity (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ < 0) :
  (f x₁ (-1) > f x₂ (-1)) :=
by
  sorry

end even_function_a_value_monotonicity_on_neg_infinity_l789_789318


namespace max_cardinality_subset_l789_789799

theorem max_cardinality_subset (X : Set ℕ) (hX : X = { n | 1 ≤ n ∧ n ≤ 100 }) :
  ∃ A ⊆ X, (∀ (x y : ℕ), x ∈ A → y ∈ A → x < y → y ≠ 3 * x) ∧ A.card = 76 :=
begin
  sorry
end

end max_cardinality_subset_l789_789799


namespace function_properties_l789_789418

noncomputable def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

theorem function_properties 
  (b c : ℝ) :
  ((c = 0 → (∀ x : ℝ, f (-x) b 0 = -f x b 0)) ∧
   (b = 0 → (∀ x₁ x₂ : ℝ, (x₁ ≤ x₂ → f x₁ 0 c ≤ f x₂ 0 c))) ∧
   (∃ (c : ℝ), ∀ (x : ℝ), f (x + c) b c = f (x - c) b c) ∧
   (¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0))) := 
by
  sorry

end function_properties_l789_789418


namespace number_of_pipes_l789_789990

theorem number_of_pipes (L : ℝ) : 
  let r_small := 1
  let r_large := 3
  let len_small := L
  let len_large := 2 * L
  let volume_large := π * r_large^2 * len_large
  let volume_small := π * r_small^2 * len_small
  volume_large = 18 * volume_small :=
by
  sorry

end number_of_pipes_l789_789990


namespace lambda_even_of_not_in_H_no_5_order_H_table_l789_789762

/-- Definition of an n-order H table. -/
def is_H_table (n : ℕ) (a : ℕ → ℕ → ℤ) : Prop :=
  (∀ i j, a i j ∈ {-1, 0, 1}) ∧
  let row_sums := λ i, ∑ j in finset.range n, a i j in
  let col_sums := λ j, ∑ i in finset.range n, a i j in
  list.nodup (list.of_fn row_sums) ∧ list.nodup (list.of_fn col_sums) ∧ -- All distinct sums
  list.disjoint (list.of_fn row_sums) (list.of_fn col_sums)

/- (I) Example of a 2-order H table -/
def example_2_order_H_table : matrix (fin 2) (fin 2) ℤ :=
  λ i j, match i, j with
  | 0, 0 => 1
  | 0, 1 => 1
  | 1, 0 => 0
  | 1, 1 => -1
  | _, _ => sorry -- This case should never happen

example : is_H_table 2 (λ i j, example_2_order_H_table i j) := by
  unfold is_H_table example_2_order_H_table
  split
  · intros i j
    cases i; cases j; norm_num
  · let row_sums := λ i, finset.sum (finset.range 2) (λ j, example_2_order_H_table i j)
    let col_sums := λ j, finset.sum (finset.range 2) (λ i, example_2_order_H_table i j)
    split
    · unfold row_sums list.of_fn finset.sum example_2_order_H_table at *
      norm_num
      simp
    · unfold col_sums list.of_fn finset.sum example_2_order_H_table at *
      norm_num
      simp

/- (II) For any n-order H table, if λ ∉ H, then λ is even -/
theorem lambda_even_of_not_in_H (n : ℕ) (a : ℕ → ℕ → ℤ) 
  [is_H_table n a] (λ : ℤ) (h : λ ∉ {i | (∑ j in finset.range n, a i j) ∪ {j | (∑ i in finset.range n, a i j)}}) : λ % 2 = 0 :=
sorry

/- (III) There does not exist a 5-order H table -/
theorem no_5_order_H_table : ¬ ∃ a : ℕ → ℕ → ℤ, is_H_table 5 a :=
sorry

end lambda_even_of_not_in_H_no_5_order_H_table_l789_789762


namespace shaded_region_area_l789_789979

noncomputable
def area_shaded_region 
  (radius : ℝ) 
  (OA_dist : ℝ) 
  (midpoint_O : ℝ)
  (tangent_OC_OD : Prop)
  (common_tangent_EF: Prop)
  (O_midpoint_AB : midpoint_O = sqrt(2) * radius)
  (seg_length_AB : seg_length_AB = 2 * OA_dist)
  : ℝ :=
    let radius_square := radius * radius
    let rectangle_area := radius * (2 * midpoint_O)
    let triangle_area := radius_square / 2
    let sector_area := (π * radius_square) / 4
    rectangle_area - triangle_area - sector_area

theorem shaded_region_area
  (radius : ℝ := 3)
  (midpoint_O : ℝ := 3*sqrt(2))
  (tangent_OC_OD : Prop := true)
  (common_tangent_EF: Prop := true)
  (O_midpoint_AB : midpoint_O = 3*sqrt(2))
  (seg_length_AB : seg_length_AB = 6*sqrt(2))
  : area_shaded_region 3 3*sqrt(2) (3*sqrt(2)) tangent_OC_OD common_tangent_EF = 18*sqrt(2) - 9 - (9*π)/4 :=
by 
  sorry

end shaded_region_area_l789_789979


namespace number_of_invertibles_mod_15_l789_789632

theorem number_of_invertibles_mod_15 : 
  ∃ (count : ℕ), count = 8 ∧ (set_of (λ a : ℕ, a < 15 ∧ Nat.gcd a 15 = 1)).card = count :=
by {
  sorry
}

end number_of_invertibles_mod_15_l789_789632


namespace value_of_a_b_squared_l789_789282

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a - b = Real.sqrt 2
axiom h2 : a * b = 4

theorem value_of_a_b_squared : (a + b)^2 = 18 := by
   sorry

end value_of_a_b_squared_l789_789282


namespace part_1_part_2_part_3_l789_789416

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2
noncomputable def g (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := a * Real.log x
noncomputable def F (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x * g x a h
noncomputable def G (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x - g x a h + (a - 1) * x 

theorem part_1 (a : ℝ) (h : 0 < a) :
  ∃(x : ℝ), x = -(a / (4 * Real.exp 1)) :=
sorry

theorem part_2 (a : ℝ) (h1 : 0 < a) : 
  (∃ x1 x2, (1/e) < x1 ∧ x1 < e ∧ (1/e) < x2 ∧ x2 < e ∧ G x1 a h1 = 0 ∧ G x2 a h1 = 0) 
    ↔ (a > (2 * Real.exp 1 - 1) / (2 * (Real.exp 1)^2 + 2 * Real.exp 1) ∧ a < 1/2) :=
sorry

theorem part_3 : 
  ∀ {x : ℝ}, 0 < x → Real.log x + (3 / (4 * x^2)) - (1 / Real.exp x) > 0 :=
sorry

end part_1_part_2_part_3_l789_789416


namespace heights_inequality_l789_789410

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h : a ≤ b ∧ b ≤ c) : 
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) := 
sorry

end heights_inequality_l789_789410


namespace find_y_dot_l789_789681

def regression_line (x : ℝ) : ℝ := 1.5 * x + 45

def x_values : List ℝ := [1, 5, 7, 13, 19]

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem find_y_dot :
  let x_bar := mean x_values in
  let y_dot := regression_line x_bar in
  y_dot = 58.5 :=
by
  sorry

end find_y_dot_l789_789681


namespace no_function_satisfies_condition_l789_789636

theorem no_function_satisfies_condition :
  ¬ ∃ (f: ℕ → ℕ), ∀ (n: ℕ), f (f n) = n + 2017 :=
by
  -- Proof details are omitted
  sorry

end no_function_satisfies_condition_l789_789636


namespace cube_root_simplification_l789_789193

theorem cube_root_simplification : 
  (Real.cbrt (6 / 20.25)) = (2 / 3) := 
by
  sorry

end cube_root_simplification_l789_789193


namespace part_I_part_II_part_III_l789_789711

open Real

-- Define the function f(x) = e^x - a * x
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x

-- Statement for the first question
theorem part_I 
  (a : ℝ) 
  (h_tangent : ∃ x₀ : ℝ, x₀ = 0 ∧ ∃ y₀ : ℝ, y₀ = 0 ∧ (tangent_line_on_f : ∀ x : ℝ, x = 1)) : 
  a = 2 := sorry

-- Statement for the second question
theorem part_II 
  (a : ℝ) 
  (h_no_zeros : ∀ x : ℝ, -1 < x → (f x a) ≠ 0) :
  -1 / exp 1 ≤ a ∧ a < exp 1 := sorry

-- Statement for the third question
theorem part_III
  (a : ℝ)
  (h_a_eq_1 : a = 1)
  (x : ℝ) :
  f x 1 ≥ (1 + x) / (f x 1 + x) := sorry

end part_I_part_II_part_III_l789_789711


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789235

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789235


namespace original_garden_area_l789_789928

theorem original_garden_area :
  ∃ (x : ℕ), (x + 2) * (x + 3) = 182 ∧ x * x = 121 :=
begin
  sorry
end

end original_garden_area_l789_789928


namespace a_81_eq_640_l789_789683

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 0 then 0 -- auxiliary value since sequence begins from n=1
else if n = 1 then 1
else (2 * n - 1) ^ 2 - (2 * n - 3) ^ 2

theorem a_81_eq_640 : sequence_a 81 = 640 :=
by
  sorry

end a_81_eq_640_l789_789683


namespace exists_mk_for_any_n_l789_789422

open Set Nat

def P : Set ℕ := {1, 2, 3, 4, 5}

def f (m k : ℕ) : ℕ :=
  ∑ i in P, (⌊m * Real.sqrt ((k + 1) / (i + 1))⌋ : ℕ)

theorem exists_mk_for_any_n (n : ℕ) (n_pos : n > 0) :
  ∃ (m k : ℕ), k ∈ P ∧ m > 0 ∧ f m k = n :=
sorry

end exists_mk_for_any_n_l789_789422


namespace tan_difference_l789_789280

theorem tan_difference (α : ℝ) (h1 : Real.cos α = -3/5) (h2 : α ∈ Ioc (π / 2) π) :
  Real.tan (π / 4 - α) = -7 := by
  sorry

end tan_difference_l789_789280


namespace mean_books_read_l789_789938

theorem mean_books_read :
  let readers1 := 4
  let books1 := 3
  let readers2 := 5
  let books2 := 5
  let readers3 := 2
  let books3 := 7
  let readers4 := 1
  let books4 := 10
  let total_readers := readers1 + readers2 + readers3 + readers4
  let total_books := (readers1 * books1) + (readers2 * books2) + (readers3 * books3) + (readers4 * books4)
  let mean_books := total_books / total_readers
  mean_books = 5.0833 :=
by
  sorry

end mean_books_read_l789_789938


namespace solve_for_m_l789_789467

namespace ProofProblem

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem solve_for_m (m : ℝ) : 3 * f 3 m = g 3 m → m = 0 := by
  sorry

end ProofProblem

end solve_for_m_l789_789467


namespace find_a_of_real_sum_l789_789291

noncomputable def z1 (a : ℝ) : ℂ := complex.mk (3 / (a + 5)) (10 - a ^ 2)
noncomputable def z2 (a : ℝ) : ℂ := complex.mk (2 / (1 - a)) (2 * a - 5)

theorem find_a_of_real_sum (a : ℝ) (h : complex.conj (z1 a) + z2 a).im = 0 : a = 3 :=
sorry

end find_a_of_real_sum_l789_789291


namespace distance_from_O_to_plane_ABC_correct_l789_789725

noncomputable def distance_from_O_to_plane_ABC
    (A B C S O : ℝ × ℝ × ℝ)
    (h1 : ∃ x y, A = (x, y, 0) ∧ B = (x + 2, y, 0) ∧ C = (x, y + 2, 0) ∧ x^2 + y^2 = 2)
    (h2 : dist S A = 2 ∧ dist S B = 2 ∧ dist S C = 2)
    (h3 : dist A B = 2)
    (h4 : dist S O = 2 ∧ dist A O = 2 ∧ dist B O = 2 ∧ dist C O = 2) : ℝ :=
  sqrt 3 / 3

theorem distance_from_O_to_plane_ABC_correct
    (A B C S O : ℝ × ℝ × ℝ)
    (h1 : ∃ x y, A = (x, y, 0) ∧ B = (x + 2, y, 0) ∧ C = (x, y + 2, 0) ∧ x^2 + y^2 = 2)
    (h2 : dist S A = 2 ∧ dist S B = 2 ∧ dist S C = 2)
    (h3 : dist A B = 2)
    (h4 : dist S O = 2 ∧ dist A O = 2 ∧ dist B O = 2 ∧ dist C O = 2) :
  distance_from_O_to_plane_ABC A B C S O h1 h2 h3 h4 = sqrt 3 / 3 :=
begin
  sorry
end

end distance_from_O_to_plane_ABC_correct_l789_789725


namespace imaginary_part_of_z_is_2_l789_789707

noncomputable def z : ℂ := (3 * Complex.I + 1) / (1 - Complex.I)

theorem imaginary_part_of_z_is_2 : z.im = 2 := 
by 
  -- proof goes here
  sorry

end imaginary_part_of_z_is_2_l789_789707


namespace sqrt_expression_evaluation_l789_789999

theorem sqrt_expression_evaluation :
  ∃ (a b c : ℤ), (sqrt (72 + 24 * sqrt 6) = (a : ℝ) + (b : ℝ) * sqrt c) ∧ 
                 (c.factors.map(sqrt).filter (λ x, x > 1) = []) ∧ 
                 (a + b + c = 15) := 
by {
  sorry
}

end sqrt_expression_evaluation_l789_789999


namespace barney_and_regular_dinosaurs_combined_weight_l789_789610

theorem barney_and_regular_dinosaurs_combined_weight :
  (∀ (barney five_dinos_weight) (reg_dino_weight : ℕ),
    (barney = five_dinos_weight + 1500) →
    (five_dinos_weight = 5 * reg_dino_weight) →
    (reg_dino_weight = 800) →
    barney + five_dinos_weight = 9500) :=
by 
  intros barney five_dinos_weight reg_dino_weight h1 h2 h3
  have h4 : five_dinos_weight = 5 * 800 := by rw [h3, mul_comm]
  rw h4 at h2
  have h5 : five_dinos_weight = 4000 := by linarith
  rw h5 at h1
  have h6 : barney = 4000 + 1500 := h1
  have h7 : barney = 5500 := by linarith
  rw [h5, h7]
  linarith

end barney_and_regular_dinosaurs_combined_weight_l789_789610


namespace sum_first_nine_terms_l789_789763

variable {a : ℕ → ℝ}
variable {a1 : ℝ} (h1 : a 1 = 3) {a9 : ℝ} (h9 : a 9 = 11)
noncomputable def S : ℕ → ℝ
| n => ∑ i in Finset.range (n + 1), a i

theorem sum_first_nine_terms : S 9 = 63 := by
  -- Sorry placeholder indicating that the proof is omitted.
  sorry

end sum_first_nine_terms_l789_789763


namespace problem_statement_l789_789853

theorem problem_statement (x : ℂ) (h₁ : x ^ 2023 - 3 * x + 2 = 0) (h₂ : x ≠ 1) :
  x ^ 2022 + x ^ 2021 + ⋯ + x + 1 = 3 := 
sorry

end problem_statement_l789_789853


namespace unit_price_large_notebook_l789_789554

theorem unit_price_large_notebook :
  ∃ (x : ℕ), (4 * x + 6 * (x - 3) = 62) ∧ (x = 8) :=
by
  use 8
  simp
  sorry

end unit_price_large_notebook_l789_789554


namespace correct_statements_l789_789463

-- Define the floor function and properties of the floor function
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f(x) = floor(x)
def f (x : ℝ) : ℤ := floor x

-- Define the properties given in the problem
def property1 (x : ℝ) : Prop := x ∈ set.Icc 1 2 → f x ∈ {0, 1, 2}
def property2 (x : ℝ) : Prop := f (x + 1) = f x + 1
def property3 (x1 x2 : ℝ) : Prop := f (x1 + x2) = f x1 + f x2
def g (x : ℝ) : ℝ := x - f x
def property4 (x : ℝ) : Prop := ∃ p : ℝ, p != 0 ∧ ∀ k : ℕ, g (x + k * p) = g x

-- The theorem stating which statements are correct and which are not
theorem correct_statements : ¬ property1 ∧ property2 ∧ ¬ property3 ∧ property4 :=
by
  -- Omitted proof
  sorry

end correct_statements_l789_789463


namespace dishonest_dealer_profit_l789_789920

theorem dishonest_dealer_profit (cost_weight actual_weight : ℝ) (kg_in_g : ℝ) 
  (h1 : cost_weight = 1000) (h2 : actual_weight = 920) (h3 : kg_in_g = 1000) :
  ((cost_weight - actual_weight) / actual_weight) * 100 = 8.7 := by
  sorry

end dishonest_dealer_profit_l789_789920


namespace largest_integer_less_than_100_with_remainder_4_l789_789206

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789206


namespace area_square_EFGH_l789_789448

noncomputable section
open_locale classical

variables {O : Type*} [normed_group O] [normed_space ℝ O]
variables (C : metric_space O) (square_ABCD square_EFGH : set O) (center : O)
variables (a x y: ℝ)

-- Defining the conditions
def is_square (s : set O) : Prop := sorry
def inscribed_in (s : set O) (o : O) : Prop := sorry
def touches (s : set O) (o : set O) : Prop := sorry

axiom square_ABCD_is_square : is_square square_ABCD
axiom square_EFGH_is_square : is_square square_EFGH
axiom ABCD_inscribed_in_circle : inscribed_in square_ABCD center
axiom area_square_ABCD : set.volume square_ABCD = 4
axiom vertex_C_on_ABCD : C ∈ square_ABCD
axiom vertex_EF_on_C : E ∈ square_EFGH ∧ F ∈ square_EFGH
axiom vertices_GH_on_circle : touches square_EFGH (metric.closed_ball center (x * y))

-- Proving that the area of square EFGH is 4
theorem area_square_EFGH : set.volume square_EFGH = 4 :=
sorry

end area_square_EFGH_l789_789448


namespace number_of_invertibles_mod_15_l789_789633

theorem number_of_invertibles_mod_15 : 
  ∃ (count : ℕ), count = 8 ∧ (set_of (λ a : ℕ, a < 15 ∧ Nat.gcd a 15 = 1)).card = count :=
by {
  sorry
}

end number_of_invertibles_mod_15_l789_789633


namespace geometric_sequence_problem_l789_789760

noncomputable def a (n : ℕ) : ℝ := sorry -- Assume a given geometric sequence

-- Conditions
def cond1 : ℝ := a 1 + a 3 + a 5
def cond2 : ℝ := a 5 + a 7 + a 9

theorem geometric_sequence_problem 
  (h1 : cond1 = 7) 
  (h2 : cond2 = 28) :
  a 9 + a 11 + a 13 = 112 := 
sorry

end geometric_sequence_problem_l789_789760


namespace room_dimension_l789_789460

theorem room_dimension (x : ℕ) 
  (h1 : cost_per_square_foot = 5) 
  (h2 : door_area = 6 * 3) 
  (h3 : window_area = 4 * 3) 
  (h4 : total_cost = 4530) : 
  x = 25 :=
by
  let room_height := 12
  let room_length := 15
  let total_wall_area := 2 * (x * room_height) + 2 * (room_length * room_height)
  let total_door_window_area := 6 * 3 + 3 * (4 * 3)
  let area_to_whitewash := total_wall_area - total_door_window_area
  have total_cost_eq := area_to_whitewash * cost_per_square_foot
  have : total_cost_eq = total_cost, from sorry
  have : area_to_whitewash = (24 * x + 306), from sorry
  have : total_cost_eq = 5 * (24 * x + 306), from sorry
  have : 5 * (24 * x + 306) = 4530, from sorry
  have : 24 * x + 306 = 906, from sorry
  have : 24 * x = 600, from sorry
  have : x = 25, from sorry
  exact this

end room_dimension_l789_789460


namespace product_of_roots_product_of_x_for_undefined_l789_789908

theorem product_of_roots (a b c : ℝ) (h : a ≠ 0) (h_eq : a*x^2 + b*x + c = 0 ∧ ∃ x, a*x^2 + b*x + c = 0) : 
  (∏ x in Finset.filter (roots (λ x, a * x^2 + b * x + c)), x) = c / a := by
  sorry

theorem product_of_x_for_undefined :
  (∏ x in Finset.filter (λ x : ℝ, x^2 + 4 * x - 5 = 0), x) = -5 := by
  apply product_of_roots 1 4 (-5)
  { norm_num }
  { existsi [1, -5]
    ring
  }
  sorry

end product_of_roots_product_of_x_for_undefined_l789_789908


namespace class_gpa_l789_789128

theorem class_gpa (n : ℕ) (h1 : (n / 3) * 60 + (2 * (n / 3)) * 66 = total_gpa) :
  total_gpa / n = 64 :=
by
  sorry

end class_gpa_l789_789128


namespace root_in_interval_l789_789889

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 1) - 1 / (3 * x + 2)

theorem root_in_interval :
  ∃ x : ℝ, x ∈ set.Ioo (1 / 4) (1 / 2) ∧ f x = 0 :=
begin
  have f_14 : f (1 / 4) = Real.log 1.5 - 4 / 11,
        -- Assume the reference value as given
        rw [Real.log_eq_approx 1.5 0.41],
        norm_num,
  have f_12 : f (1 / 2) = Real.log 2 - 2 / 7,
        -- Assume the reference value as given
        rw [Real.log_eq_approx 2 0.69],
        norm_num,
  
  have h_f14_neg : f (1 / 4) < 0,
    exact calc
      Real.log 1.5 - 4 / 11
        ... < 0 : by linarith,

  have h_f12_pos : f (1 / 2) > 0,
    exact calc
      Real.log 2 - 2 / 7
        ... > 0 : by linarith,

  exact ⟨x, _, _⟩, sorry -- We need to show the existence of a root within the interval using IVT or similar theorem
end

end root_in_interval_l789_789889


namespace largest_integer_less_than_100_with_remainder_4_l789_789203

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789203


namespace largest_integer_remainder_condition_l789_789261

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789261


namespace probability_f_ge1_l789_789713

noncomputable def f (x: ℝ) : ℝ := 3*x^2 - x - 1

def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def valid_intervals : Set ℝ := { x | -1 ≤ x ∧ x ≤ -2/3 } ∪ { x | 1 ≤ x ∧ x ≤ 2 }

def interval_length (a b : ℝ) : ℝ := b - a

theorem probability_f_ge1 : 
  (interval_length (-2/3) (-1) + interval_length 1 2) / interval_length (-1) 2 = 4 / 9 := 
by
  sorry

end probability_f_ge1_l789_789713


namespace simplified_evaluated_expression_l789_789842

noncomputable def a : ℚ := 1 / 3
noncomputable def b : ℚ := 1 / 2
noncomputable def c : ℚ := 1

def expression (a b c : ℚ) : ℚ := a^2 + 2 * b - c

theorem simplified_evaluated_expression :
  expression a b c = 1 / 9 :=
by
  sorry

end simplified_evaluated_expression_l789_789842


namespace vector_addition_result_l789_789653

-- Define the vectors
def vec1 : Matrix (Fin 2) (Fin 1) ℤ := ![![5], ![-9]]
def vec2 : Matrix (Fin 2) (Fin 1) ℤ := ![[-8], ![14]]
def result : Matrix (Fin 2) (Fin 1) ℤ := ![[-3], ![5]]

-- The theorem
theorem vector_addition_result : vec1 + vec2 = result := by
  sorry

end vector_addition_result_l789_789653


namespace range_of_k_for_angle_APB_sixty_degrees_l789_789691

theorem range_of_k_for_angle_APB_sixty_degrees
  (k : Real)
  (P : Real × Real)
  -- Conditions
  (circle_O : P ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 1})
  (line_l : P ∈ {p : ℝ × ℝ | p.1 + p.2 + k = 0})
  (angle_APB : ∃ A B : Real × Real,
    is_tangent O A ∧ is_tangent O B ∧ ∠APB = 60 * (ℝ.pi / 180)) :
  -- To Prove
  -2 * Real.sqrt 2 ≤ k ∧ k ≤ 2 * Real.sqrt 2 :=
by
  sorry

end range_of_k_for_angle_APB_sixty_degrees_l789_789691


namespace sum_first_100_terms_l789_789065

noncomputable def a_seq : ℕ → ℕ
| 0     := 0  -- Dummy to handle zeroth element, won't be used in sum.
| 1     := 1
| (n+1) := a_seq n + n + 1

theorem sum_first_100_terms :
  (∑ n in Finset.range 100, (1 : ℝ) / a_seq (n+1)) = 200 / 101 :=
by
  sorry

end sum_first_100_terms_l789_789065


namespace tom_took_out_beads_l789_789500

-- Definitions of the conditions
def green_beads : Nat := 1
def brown_beads : Nat := 2
def red_beads : Nat := 3
def beads_left_in_container : Nat := 4

-- Total initial beads
def total_beads : Nat := green_beads + brown_beads + red_beads

-- The Lean problem statement to prove
theorem tom_took_out_beads : (total_beads - beads_left_in_container) = 2 :=
by
  sorry

end tom_took_out_beads_l789_789500


namespace real_root_exists_l789_789293

theorem real_root_exists (a b c : ℝ) :
  (∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b - c) * x + (c - a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c - a) * x + (a - b) = 0) :=
by {
  sorry
}

end real_root_exists_l789_789293


namespace sophie_hours_needed_l789_789849

-- Sophie needs 206 hours to finish the analysis of all bones.
theorem sophie_hours_needed (num_bones : ℕ) (time_per_bone : ℕ) (total_hours : ℕ) (h1 : num_bones = 206) (h2 : time_per_bone = 1) : 
  total_hours = num_bones * time_per_bone :=
by
  rw [h1, h2]
  norm_num
  sorry

end sophie_hours_needed_l789_789849


namespace correct_options_BD_l789_789159

-- Definition of conditions
variables (a b c d : ℝ)
variables (hA : a < b ∧ b < 0)          -- condition for option A
variables (hB : c > d ∧ a > b)          -- condition for option B
variables (hC : b < a ∧ a < 0 ∧ c < 0)  -- condition for option C
variables (hD : a > 0 ∧ b > c ∧ c > 0)  -- condition for option D

-- Prove that B and D are the correct options
theorem correct_options_BD : (a - d > b - c) ∧ ((c + a) / (b + a) > c / b) :=
by {
  have hB1 : a - d > b - c, from sorry,          -- proof for option B
  have hD1 : (c + a) / (b + a) > c / b, from sorry, -- proof for option D
  exact ⟨hB1, hD1⟩
}

end correct_options_BD_l789_789159


namespace sum_of_series_equals_one_half_l789_789181

theorem sum_of_series_equals_one_half : 
  (∑' k : ℕ, (1 / ((2 * k + 1) * (2 * k + 3)))) = 1 / 2 :=
sorry

end sum_of_series_equals_one_half_l789_789181


namespace find_k_perpendicular_find_d_parallel_l789_789378

variables (a b c : (ℝ × ℝ)) (k : ℝ) (d : (ℝ × ℝ))

-- Given conditions
def vec_a := (3, 2)
def vec_b := (-1, 2)
def vec_c := (4, 1)

-- Question 1: Prove k
theorem find_k_perpendicular (h : (vec_a.1 + k * vec_c.1, vec_a.2 + k * vec_c.2) ∙ (2 * vec_b.1 - vec_a.1, 2 * vec_b.2 - vec_a.2) = 0) :
  k = -11 / 18 := sorry

-- Question 2: Prove d
theorem find_d_parallel (h1 : ∃ x : ℝ, d = (4 * x, x)) (h2 : real.sqrt ((d.1)^2 + (d.2)^2) = real.sqrt 34) :
  d = (4 * real.sqrt 2, real.sqrt 2) ∨ d = (-4 * real.sqrt 2, -real.sqrt 2) := sorry

end find_k_perpendicular_find_d_parallel_l789_789378


namespace equilateral_triangle_ratio_l789_789967

theorem equilateral_triangle_ratio (α : ℝ) :
  ∀ (ABC A1B1C1 : Type) [equilateral_triangle ABC] [equilateral_triangle A1B1C1]
    (A B C A1 B1 C1 : ABC)
    (h1 : A1 ∈ BC) (h2 : B1 ∈ AC) (h3 : C1 ∈ AB)
    (h4 : ∠ (A1, B1, C) = α),
  AB / A1B1 = 2 * sin (30 + α) := by
  sorry

end equilateral_triangle_ratio_l789_789967


namespace daniel_total_spent_l789_789626

/-
Daniel buys various items with given prices, receives a 10% coupon discount,
a store credit of $1.50, a 5% student discount, and faces a 6.5% sales tax.
Prove that the total amount he spends is $8.23.
-/
def total_spent (prices : List ℝ) (coupon_discount store_credit student_discount sales_tax : ℝ) : ℝ :=
  let initial_total := prices.sum
  let after_coupon := initial_total * (1 - coupon_discount)
  let after_student := after_coupon * (1 - student_discount)
  let after_store_credit := after_student - store_credit
  let final_total := after_store_credit * (1 + sales_tax)
  final_total

theorem daniel_total_spent :
  total_spent 
    [0.85, 0.50, 1.25, 3.75, 2.99, 1.45] -- prices of items
    0.10 -- 10% coupon discount
    1.50 -- $1.50 store credit
    0.05 -- 5% student discount
    0.065 -- 6.5% sales tax
  = 8.23 :=
by
  sorry

end daniel_total_spent_l789_789626


namespace parabola_focus_directrix_distance_l789_789462

theorem parabola_focus_directrix_distance :
  ∃ p : ℝ, (p = 1) ∧ (∃ (x y : ℝ), x^2 = 2 * y ∧ distance (0, p - 2) (0, -p) = 1) :=
by
  sorry

end parabola_focus_directrix_distance_l789_789462


namespace katya_sequences_l789_789392

def is_valid_digit_replacement (original : ℕ) (left_neigh : Option ℕ) (right_neigh : Option ℕ) : ℕ :=
  (if left_neigh.isSome ∧ left_neigh.get < original then 1 else 0) +
  (if right_neigh.isSome ∧ right_neigh.get < original then 1 else 0)

def valid_transformation_sequence (seq : List ℕ) : Bool :=
  seq.length = 10 ∧
  ∀ i, i < 10 -> 
    let left_neigh := if i > 0 then some (seq.get ⟨i - 1, by linarith⟩) else none;
    let right_neigh := if i < 9 then some (seq.get ⟨i + 1, by linarith⟩) else none;
    seq.get ⟨i, by linarith⟩ = is_valid_digit_replacement(i, left_neigh, right_neigh)

theorem katya_sequences :
  valid_transformation_sequence [1, 1, 0, 1, 1, 1, 1, 1, 1, 1] ∧
  ¬valid_transformation_sequence [1, 2, 0, 1, 2, 0, 1, 0, 2, 0] ∧
  valid_transformation_sequence [1, 0, 2, 1, 0, 2, 1, 0, 2, 0] ∧
  valid_transformation_sequence [0, 1, 1, 2, 1, 0, 2, 0, 1, 1] := 
by {
  sorry
}

end katya_sequences_l789_789392


namespace lateral_surface_area_of_cylinder_l789_789031

theorem lateral_surface_area_of_cylinder (V : ℝ) (hV : V = 27 * Real.pi) : 
  ∃ (S : ℝ), S = 18 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l789_789031


namespace greatest_integer_with_gcd_l789_789524

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l789_789524


namespace hyperbola_eccentricity_l789_789306

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (A B : ℝ × ℝ) (F : ℝ × ℝ)
variable (hA : A = (a, 0))
variable (hF : F = (c, 0))
variable (hB : B = (c, b ^ 2 / a))
variable (h_slope : (b ^ 2 / a - 0) / (c - a) = 3)
variable (h_hyperbola : b ^ 2 = c ^ 2 - a ^ 2)
def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity : eccentricity a c = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789306


namespace complex_fraction_simplify_in_first_quadrant_top_condition_l789_789140

-- Definitions and assumptions utilized in the Lean statement should reflect the conditions identified in step a).
variable (m : ℝ)

def z (m : ℝ) : ℂ := (m+2) + (m^2 - m - 2) * Complex.I

theorem complex_fraction_simplify :
  (Complex.mk (-3) 1) / (Complex.mk 2 (-4)) = (Complex.mk (-1/2) (-1/2)) := by
  sorry

theorem in_first_quadrant_top_condition (m : ℝ) (h1 : m + 2 > 0) (h2 : m^2 - m - 2 > 0) : 
  m ∈ (Set.Ioo 2 ∞) := by
  sorry

end complex_fraction_simplify_in_first_quadrant_top_condition_l789_789140


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789518

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789518


namespace perpendicular_plane_line_not_perpendicular_plane_line_necessary_but_not_sufficient_condition_l789_789279

variable (α β : Type) [Plane α] [Plane β] (m : Line α)

theorem perpendicular_plane_line :
  (α ⟂ β) → (m ∈ α) ∧ (m ⟂ β) := by sorry

theorem not_perpendicular_plane_line :
  ¬((m ∈ α) ∧ (m ⟂ β) → (α ⟂ β)) := by sorry

theorem necessary_but_not_sufficient_condition :
  (α ⟂ β) ↔ ((m ∈ α) ∧ (m ⟂ β)) := by
  exact ⟨perpendicular_plane_line α β m, by 
    intro h,
    have h₁ : (m ∈ α) ∧ (m ⟂ β) := ⟨h.left, h.right⟩,
    contradiction_weapon.not_perpendicular_plane_line α β m h₁ ⟩

end perpendicular_plane_line_not_perpendicular_plane_line_necessary_but_not_sufficient_condition_l789_789279


namespace cos_sin_equation_l789_789918

def cos (x : ℝ) : ℝ := sorry  -- assuming cosine function
def sin (x : ℝ) : ℝ := sorry  -- assuming sine function

noncomputable def solution_set_x1 : ℤ → ℝ := λ k, (real.pi / 2) * (2 * k + 1)
noncomputable def solution_set_x2 : ℤ → ℝ := λ n, (real.pi / 5) * n

theorem cos_sin_equation {x : ℝ} (h : cos x * cos (2 * x) * sin (3 * x) = 0.25 * sin (2 * x)) :
  (∃ k : ℤ, x = solution_set_x1 k) ∨ (∃ n : ℤ, x = solution_set_x2 n) :=
sorry

end cos_sin_equation_l789_789918


namespace solve_equation_1_solve_equation_2_l789_789656

theorem solve_equation_1 (x : ℝ) (h : x^3 - 3 = ⅜) : x = 3 / 2 :=
sorry

theorem solve_equation_2 (x : ℝ) (h : (x - 1)^2 = 25) : x = 6 ∨ x = -4 :=
sorry

end solve_equation_1_solve_equation_2_l789_789656


namespace inequality_of_f_g_l789_789406

theorem inequality_of_f_g (f g : ℝ → ℝ) (h_diff_f : differentiable ℝ f) (h_diff_g : differentiable ℝ g)
  (h_pos_f : ∀ x, 0 < f x) (h_pos_g : ∀ x, 0 < g x)
  (h_inequality : ∀ x, f'' x * g x - f x * g'' x < 0)
  (a b x : ℝ) (h_interval : b < x ∧ x < a) : 
  f x * g a > f a * g x := 
sorry

end inequality_of_f_g_l789_789406


namespace area_of_regular_octadecagon_l789_789950

-- Definition of a regular octadecagon inscribed in a circle with radius r
def regular_octadecagon_inscribed (r : ℝ) : Prop :=
  ∀ (r : ℝ), r > 0 → 
  let θ := 20 * (Real.pi / 180), -- convert degrees to radians
      area_triangle := 0.5 * r^2 * Real.sin θ in
  18 * area_triangle = 3.078 * r^2

-- Theorem statement
theorem area_of_regular_octadecagon {r : ℝ} (h : regular_octadecagon_inscribed r) : 
  18 * (0.5 * r^2 * Real.sin (20 * (Real.pi / 180))) = 3.078 * r^2 :=
sorry

end area_of_regular_octadecagon_l789_789950


namespace number_of_questions_in_test_l789_789491

theorem number_of_questions_in_test (x : ℕ) (h1 : 20 < 0.70 * x) (h2 : 0.60 * x < 20)
  (h3 : x % 4 = 0)
: x = 32 :=
sorry

end number_of_questions_in_test_l789_789491


namespace questions_in_test_l789_789493

theorem questions_in_test
  (x : ℕ)
  (h_sections : x % 4 = 0)
  (h_correct : 20 < 0.70 * x)
  (h_correct2 : 20 > 0.60 * x) :
  x = 32 := 
by
  sorry

end questions_in_test_l789_789493


namespace problem_1_problem_2_l789_789701

open Real

noncomputable def lg (x : ℝ) : ℝ := logb 10 x

def a_seq (n : ℕ) : ℝ :=
  match n with
  | 0 => 2
  | (nat.succ n) => a_seq n ^ 2 + 2 * a_seq n

def b_seq (n : ℕ) : ℝ :=
  1 / a_seq n + 1 / (a_seq n + 2)

def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_seq (i + 1)

theorem problem_1 (n : ℕ) : lg (1 + a_seq (n + 1)) = 2 * lg (1 + a_seq n) := sorry

theorem problem_2 (n : ℕ) : S_n n = 1 - 2 / (3 ^ (2 ^ n) - 1) := sorry

end problem_1_problem_2_l789_789701


namespace triangle_square_ratio_l789_789951

theorem triangle_square_ratio (s_t s_s : ℝ) (h : 3 * s_t = 4 * s_s) : s_t / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l789_789951


namespace number_of_questions_in_test_l789_789490

theorem number_of_questions_in_test (x : ℕ) (h1 : 20 < 0.70 * x) (h2 : 0.60 * x < 20)
  (h3 : x % 4 = 0)
: x = 32 :=
sorry

end number_of_questions_in_test_l789_789490


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789241

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789241


namespace problem_I_problem_II_l789_789568

-- Statement for Problem I
theorem problem_I (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) : a^4 + b^4 ≥ 2 := 
sorry

-- Statement for Problem II
theorem problem_II (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^3 + b^3 + c^3 + (1/a + 1/b + 1/c)^3 ≥ 18 ∧
  (∀ x : ℝ, x > 0 → 
    ((a = x ∧ b = x ∧ c = x) ↔ x = real.cbrt 3)) := 
sorry

end problem_I_problem_II_l789_789568


namespace triangle_is_isosceles_l789_789384

-- Given condition as an assumption in Lean
def sides_opposite_to_angles (a b c A B C : ℝ) (triangle : Prop) :=
  a = 2 * b * real.cos C

-- Conclusion that needs to be proved
theorem triangle_is_isosceles
  {a b c A B C : ℝ}
  (h1 : sides_opposite_to_angles a b c A B C (triangle a b c A B C)) :
  (∃ t : triangle, is_isosceles t) :=
sorry

end triangle_is_isosceles_l789_789384


namespace part1_part2_l789_789000

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (m n : ℝ) : f (m + n) = f m * f n
axiom positive_property (x : ℝ) (h : x > 0) : 0 < f x ∧ f x < 1

theorem part1 (x : ℝ) : f 0 = 1 ∧ (x < 0 → f x > 1) := by
  sorry

theorem part2 (x : ℝ) : 
  f (2 * x^2 - 4 * x - 1) < 1 ∧ f (x - 1) < 1 → x < -1/2 ∨ x > 2 := by
  sorry

end part1_part2_l789_789000


namespace arccos_zero_is_pi_div_two_l789_789623

def arccos_eq_pi_div_two : Prop :=
  arccos 0 = π / 2

theorem arccos_zero_is_pi_div_two : arccos_eq_pi_div_two :=
by
  sorry

end arccos_zero_is_pi_div_two_l789_789623


namespace total_goals_during_match_l789_789187

theorem total_goals_during_match (
  A1_points_first_half : ℕ := 8,
  B_points_first_half : ℕ := A1_points_first_half / 2,
  B_points_second_half : ℕ := A1_points_first_half,
  A2_points_second_half : ℕ := B_points_second_half - 2
) : (A1_points_first_half + A2_points_second_half + B_points_first_half + B_points_second_half = 26) := by
  sorry

end total_goals_during_match_l789_789187


namespace series_sum_l789_789135

noncomputable def sum_series (n : ℕ) : ℝ :=
  ∑ k in Finset.range n.succ, (k : ℝ) / (Nat.factorial (k + 1) : ℝ)

theorem series_sum (n : ℕ) :
  sum_series n = 1 - 1 / (Nat.factorial (n + 1) : ℝ) := by
  sorry

end series_sum_l789_789135


namespace total_cost_stationery_l789_789154

theorem total_cost_stationery
    (boxes : ℕ)
    (pencils_per_box : ℕ)
    (pen_cost : ℕ)
    (pencil_cost : ℕ)
    (extra_pens : ℕ) :
    boxes = 15 →
    pencils_per_box = 80 →
    pen_cost = 5 →
    pencil_cost = 4 →
    extra_pens = 300 →
    let total_pencils := boxes * pencils_per_box in
    let total_pencil_cost := total_pencils * pencil_cost in
    let total_pens := 2 * total_pencils + extra_pens in
    let total_pen_cost := total_pens * pen_cost in
    total_pencil_cost + total_pen_cost = 18300 := by
  sorry

end total_cost_stationery_l789_789154


namespace correct_option_C_l789_789912

theorem correct_option_C (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := 
sorry

end correct_option_C_l789_789912


namespace one_div_m_plus_one_div_n_l789_789875

theorem one_div_m_plus_one_div_n
  {m n : ℕ} 
  (h1 : Nat.gcd m n = 5) 
  (h2 : Nat.lcm m n = 210)
  (h3 : m + n = 75) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 14 :=
by
  sorry

end one_div_m_plus_one_div_n_l789_789875


namespace math_problem_l789_789361

noncomputable def mean_and_median_difference : ℝ :=
  let total_students := 20
  let scores := [60, 60, 60, 75, 75, 75, 75, 85, 85, 85, 85, 85, 90, 90, 90, 90, 90, 100, 100, 100]
  let mean := (60 * 3 + 75 * 4 + 85 * 5 + 90 * 5 + 100 * 3) / (total_students : ℝ)
  let median := (scores.nth 9).getD 0 + (scores.nth 10).getD 0) / 2
  median - mean

theorem math_problem : mean_and_median_difference = 2.25 := by
  sorry

end math_problem_l789_789361


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789214

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789214


namespace service_center_location_l789_789892

def serviceCenterMilepost (x3 x10 : ℕ) (r : ℚ) : ℚ :=
  x3 + r * (x10 - x3)

theorem service_center_location :
  (serviceCenterMilepost 50 170 (2/3) : ℚ) = 130 :=
by
  -- placeholder for the actual proof
  sorry

end service_center_location_l789_789892


namespace problem1_problem2_l789_789816

-- Define vectors a, b, c and the interval for x
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.cos x)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (1, 1)

-- Define the conditions for x in the interval [0, π]
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi

-- Problem 1: Prove that x = 5π/6 given (a + b) ∥ c
theorem problem1 (x : ℝ) (h : interval x) (h_parallel : ∃ k : ℝ, a x + b = k • c) : x = 5 * Real.pi / 6 :=
sorry

-- Problem 2: Prove that sin(x + π/6) = √15 / 4 given a • b = 1 / 2
theorem problem2 (x : ℝ) (h : interval x) (h_dot : (a x).1 * b.1 + (a x).2 * b.2 = 1 / 2) : 
  Real.sin (x + Real.pi / 6) = Real.sqrt 15 / 4 :=
sorry

end problem1_problem2_l789_789816


namespace figure_50_squares_eq_7651_l789_789196

def number_of_squares (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

theorem figure_50_squares_eq_7651 :
  number_of_squares 50 = 7651 := by
  simp [number_of_squares]
  sorry

end figure_50_squares_eq_7651_l789_789196


namespace total_bottles_remaining_is_14090_l789_789921

-- Define the constants
def total_small_bottles : ℕ := 5000
def total_big_bottles : ℕ := 12000
def small_bottles_sold_percentage : ℕ := 15
def big_bottles_sold_percentage : ℕ := 18

-- Define the remaining bottles
def calc_remaining_bottles (total_bottles sold_percentage : ℕ) : ℕ :=
  total_bottles - (sold_percentage * total_bottles / 100)

-- Define the remaining small and big bottles
def remaining_small_bottles : ℕ := calc_remaining_bottles total_small_bottles small_bottles_sold_percentage
def remaining_big_bottles : ℕ := calc_remaining_bottles total_big_bottles big_bottles_sold_percentage

-- Define the total remaining bottles
def total_remaining_bottles : ℕ := remaining_small_bottles + remaining_big_bottles

-- State the theorem
theorem total_bottles_remaining_is_14090 : total_remaining_bottles = 14090 := by
  sorry

end total_bottles_remaining_is_14090_l789_789921


namespace d_minus_c_eq_3_l789_789585

variables (c d : ℝ)

-- Definitions based on conditions:
def point_Q : ℝ × ℝ := (c, d)
def rotation_point : ℝ × ℝ := (2, 3)
def reflected_image : ℝ × ℝ := (7, -4)

-- Function to reflect a point about the line y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, p.fst)

-- Function to rotate a point by 90 degrees counterclockwise around another point
def rotate_90_ccw (p center : ℝ × ℝ) : ℝ × ℝ :=
  let (h, k) := center in
  let (x, y) := p in
  (h + (y - k), k - (x - h))

-- Define the original point by reversing the transformations
def original_point : ℝ × ℝ :=
  let reflected := reflect (7, -4) in
  rotate_90_ccw reflected rotation_point

-- Proof statement
theorem d_minus_c_eq_3 (h : original_point = (6, 9)) : d - c = 3 := 
  sorry

end d_minus_c_eq_3_l789_789585


namespace find_x_find_y_l789_789902

variables (x y : ℝ)

def condition1 : Prop := 0.003 = (x / 100) * 0.09
def condition2 : Prop := 0.008 = (y / 100) * 0.15

theorem find_x (h1 : condition1) : x = 3.33 :=
by
  sorry

theorem find_y (h2 : condition2) : y = 5.33 :=
by
  sorry

end find_x_find_y_l789_789902


namespace solve_for_x_l789_789852

def diamond (a b : ℝ) : ℝ := 4 * a + 2 * b

theorem solve_for_x : ∃ x : ℝ, diamond 3 (diamond x 7) = 5 ∧ x = -35 / 8 := 
by {
  use -35 / 8,
  sorry,
}

end solve_for_x_l789_789852


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789220

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789220


namespace number_above_210_is_165_l789_789112

def triangular_number (k : ℕ) : ℕ := k * (k + 1) / 2
def tetrahedral_number (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6
def row_start (k : ℕ) : ℕ := tetrahedral_number (k - 1) + 1

theorem number_above_210_is_165 :
  ∀ k, triangular_number k = 210 →
  ∃ n, n = 165 → 
  ∀ m, row_start (k - 1) ≤ m ∧ m < row_start k →
  m = 210 →
  n = m - triangular_number (k - 1) :=
  sorry

end number_above_210_is_165_l789_789112


namespace students_who_like_both_apple_pie_and_chocolate_cake_l789_789757

def total_students := 50
def students_who_like_apple_pie := 22
def students_who_like_chocolate_cake := 20
def students_who_like_neither := 10
def students_who_like_only_cookies := 5

theorem students_who_like_both_apple_pie_and_chocolate_cake :
  (students_who_like_apple_pie + students_who_like_chocolate_cake - (total_students - students_who_like_neither - students_who_like_only_cookies)) = 7 := 
by
  sorry

end students_who_like_both_apple_pie_and_chocolate_cake_l789_789757


namespace box_depth_l789_789433

theorem box_depth (rate : ℝ) (time : ℝ) (length : ℝ) (width : ℝ) (Volume : ℝ) (Depth : ℝ) : 
  rate = 4 → 
  time = 21 → 
  length = 7 → 
  width = 6 → 
  Volume = rate * time → 
  Depth = Volume / (length * width) → 
  Depth = 2 :=
by
  intros h_rate h_time h_length h_width h_volume h_depth
  rw [h_rate, h_time, h_length, h_width] at h_volume
  rw [h_volume, h_length, h_width] at h_depth
  exact h_depth

end box_depth_l789_789433


namespace angle_ne_iff_cos2angle_ne_l789_789751

theorem angle_ne_iff_cos2angle_ne (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A ≠ B) ↔ (Real.cos (2 * A) ≠ Real.cos (2 * B)) :=
sorry

end angle_ne_iff_cos2angle_ne_l789_789751


namespace greatest_integer_gcd_30_is_125_l789_789527

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l789_789527


namespace remaining_funds_correct_l789_789628

def david_initial_funds : ℝ := 1800
def emma_initial_funds : ℝ := 2400
def john_initial_funds : ℝ := 1200

def david_spent_percentage : ℝ := 0.60
def emma_spent_percentage : ℝ := 0.75
def john_spent_percentage : ℝ := 0.50

def david_remaining_funds : ℝ := david_initial_funds * (1 - david_spent_percentage)
def emma_spent : ℝ := emma_initial_funds * emma_spent_percentage
def emma_remaining_funds : ℝ := emma_spent - 800
def john_remaining_funds : ℝ := john_initial_funds * (1 - john_spent_percentage)

theorem remaining_funds_correct :
  david_remaining_funds = 720 ∧
  emma_remaining_funds = 1400 ∧
  john_remaining_funds = 600 :=
by
  sorry

end remaining_funds_correct_l789_789628


namespace good_sequence_length_bound_l789_789508

variable {n : ℕ}

def is_constant {α β : Type*} (f : α → β) : Prop :=
  ∃ b : β, ∀ a : α, f a = b

def good_seq (F : set (ℕ → ℕ)) (seq: list (ℕ → ℕ)) : Prop :=
  ∀ f ∈ seq, f ∈ F ∧ is_constant (seq.foldr (∘) id)

theorem good_sequence_length_bound (F : set (ℕ → ℕ)) (n : ℕ) :
  (∃ seq : list (ℕ → ℕ), good_seq F seq) → (∃ seq : list (ℕ → ℕ), good_seq F seq ∧ seq.length ≤ n^3) :=
by
  sorry

end good_sequence_length_bound_l789_789508


namespace ζn_converges_in_prob_to_c_ξn_ζn_pair_converges_in_distr_expected_val_converges_ξn_mult_ζn_converges_in_distr_l789_789563

open MeasureTheory ProbabilityTheory TopologicalSpace

variables {α : Type*} {β : Type*} {γ : Type*} 
          {F : Type*} {P : Type*} {G : Type*}

noncomputable theory

-- Assuming the distributions and convergence as given conditions.
variables (ζn : ℕ → α) (c : α)
variables (ξn : ℕ → β) (ξ : β)

-- Condition: ζn converges in distribution to c.
axiom ζn_converges_to_c : ∀ (ε : ℝ), ∃ (N : ℕ), ∀ (n ≥ N), 
  |ζn n - c| ≤ ε

-- Condition: ξn converges in distribution to ξ.
axiom ξn_converges_to_ξ: ∀ (ε : ℝ), ∃ (N : ℕ), ∀ (n ≥ N), 
  |ξn n - ξ| ≤ ε

-- Statement: ζn converges in probability to c.
theorem ζn_converges_in_prob_to_c : ℕ → ℝ

-- Statement: ξn, ζn pair converges in distribution to (ξ, c).
theorem ξn_ζn_pair_converges_in_distr : ∀ (f : α × β → ℝ) (hf : continuous f ∧ ∃ C, ∀ x, ∥f x∥ ≤ C), 
  lintegral (ξn × ζn) f = lintegral (ξ × c) f

-- Statement: E f(ξn, ζn) converges to E f(ξ, c) for any continuous bounded function f.
theorem expected_val_converges : ∀ (f : α × β → ℝ) (hf : continuous f ∧ ∃ C, ∀ x, ∥f x∥ ≤ C), 
  (∫ x in measure_space α, f (ξn x, ζn x)) = (∫ x in measure_space α, f (ξ x, c))

-- Statement: ξn · ζn converges in distribution to c · ξ.
theorem ξn_mult_ζn_converges_in_distr : 
  (ξn × ζn) = (ξ × c)

sorry -- Proofs are omitted as per instructions.

end ζn_converges_in_prob_to_c_ξn_ζn_pair_converges_in_distr_expected_val_converges_ξn_mult_ζn_converges_in_distr_l789_789563


namespace number_of_three_digit_numbers_without_repeated_digits_l789_789906

theorem number_of_three_digit_numbers_without_repeated_digits : 
  (finset.range 10).choose 3 * 3!.val = 648 :=
by
  -- Digits: 0 to 9 => Finset.range 10 has 10 elements
  -- Choose 3 digits from 10 (0 to 9): 10 choose 3 (ₙCₖ with n=10, k=3)
  -- Arrange those 3 digits in 3 spots (permutations): 3!
  -- Note that the 0 cannot be the leading digit
  -- Hence, we consider 9 options for the first digit, then 9 options for the second digit (after
  -- choosing one fewer, which is excluded as 0 cannot start), then 8 options left for the third
  sorry

end number_of_three_digit_numbers_without_repeated_digits_l789_789906


namespace equation_of_line_l_l789_789877

theorem equation_of_line_l (P : ℝ × ℝ) (hP : P = (1, -1)) (θ₁ θ₂ : ℕ) (hθ₁ : θ₁ = 45) (hθ₂ : θ₂ = θ₁ * 2) (hθ₂_90 : θ₂ = 90) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = l (P.fst)) := 
sorry

end equation_of_line_l_l789_789877


namespace velocity_volleyball_league_members_l789_789010

theorem velocity_volleyball_league_members (total_cost : ℕ) (socks_cost t_shirt_cost cost_per_member members : ℕ)
  (h_socks_cost : socks_cost = 6)
  (h_t_shirt_cost : t_shirt_cost = socks_cost + 7)
  (h_cost_per_member : cost_per_member = 2 * (socks_cost + t_shirt_cost))
  (h_total_cost : total_cost = 3510)
  (h_total_cost_eq : total_cost = cost_per_member * members) :
  members = 92 :=
by
  sorry

end velocity_volleyball_league_members_l789_789010


namespace simplify_expression_l789_789840

-- Definitions and conditions
def sin_cos_sum (x : ℝ) := 1 + Real.sin (x + Real.pi / 6) - Real.cos (x + Real.pi / 6)
def sin_cos_diff (x : ℝ) := 1 + Real.sin (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

-- Statement of the proof problem
theorem simplify_expression (x : ℝ) : 
  sin_cos_sum x / sin_cos_diff x = Real.tan (x / 2 + Real.pi / 12) :=
by
  sorry

end simplify_expression_l789_789840


namespace scientific_notation_of_876000_l789_789952

def scientific_notation (a : Float) (n : Int) : Float := a * (10 ^ n)

theorem scientific_notation_of_876000 : ∃ (a : Float) (n : Int), (1 ≤ |a| ∧ |a| < 10) ∧ scientific_notation a n = 876000 :=
by {
  use [8.76, 5],
  split,
  {
    split,
    norm_num,
    norm_num,
  },
  {
    norm_num,
    sorry
  }
}

end scientific_notation_of_876000_l789_789952


namespace largest_int_less_than_100_by_7_l789_789224

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789224


namespace ratio_of_three_numbers_l789_789900

theorem ratio_of_three_numbers (A B C : ℕ) 
  (h1 : C = 70)
  (h2 : C - A = 40)
  (h3 : B - A = C - B) : 
  (A:int) / 10 = 3 ∧ (B:int) / 10 = 5 ∧ (C:int) / 10 = 7 := 
by
  sorry

end ratio_of_three_numbers_l789_789900


namespace find_largest_integer_l789_789253

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789253


namespace angle_B_degree_measure_l789_789288

variables {a b c : ℝ} (A B C : ℝ)

theorem angle_B_degree_measure 
  (h1 : 3 * a * cos C = 2 * c * cos A)
  (h2 : tan A = 1 / 3) :
  B = 135 := 
sorry

end angle_B_degree_measure_l789_789288


namespace validCardSelections_l789_789074

def numberOfValidSelections : ℕ :=
  let totalCards := 12
  let redCards := 4
  let otherColors := 8 -- 4 yellow + 4 blue
  let totalSelections := Nat.choose totalCards 3
  let nonRedSelections := Nat.choose otherColors 3
  let oneRedSelections := Nat.choose redCards 1 * Nat.choose otherColors 2
  let sameColorSelections := 3 * Nat.choose 4 3 -- 3 colors, 4 cards each, selecting 3
  (nonRedSelections + oneRedSelections)

theorem validCardSelections : numberOfValidSelections = 160 := by
  sorry

end validCardSelections_l789_789074


namespace max_min_revenue_difference_l789_789822

noncomputable def giraffe_statue_grams := 120
noncomputable def elephant_statue_grams := 240
noncomputable def rhinoceros_statue_grams := 180

noncomputable def giraffe_statue_price := 150
noncomputable def elephant_statue_price := 350
noncomputable def rhinoceros_statue_price := 250

noncomputable def discount_rate := 0.9
noncomputable def jade_total := 1920

noncomputable def statutes_count (statue_grams : ℕ) : ℕ :=
    jade_total / statue_grams

noncomputable def discounted_price (price : ℕ) : ℕ :=
    price * discount_rate

noncomputable def total_revenue (count : ℕ) (price : ℕ) : ℕ :=
    if count > 3 then count * discounted_price(price) else count * price

theorem max_min_revenue_difference
    (giraffes statues_count giraffe_statue_grams)
    (elephants statues_count elephant_statue_grams)
    (rhinoceroses statues_count rhinoceros_statue_grams)
    (giraffe_revenue total_revenue giraffes giraffe_statue_price)
    (elephant_revenue total_revenue elephants elephant_statue_price)
    (rhinoceros_revenue total_revenue rhinoceroses rhinoceros_statue_price) :
    max giraffe_revenue (max elephant_revenue rhinoceros_revenue) - min giraffe_revenue (min elephant_revenue rhinoceros_revenue) = 270 := 
sorry -- no proof

end max_min_revenue_difference_l789_789822


namespace ordering_of_powers_l789_789511

theorem ordering_of_powers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by sorry

end ordering_of_powers_l789_789511


namespace second_derivative_yxx_l789_789922

-- Define the parametric functions x(t) and y(t)
def x (t : ℝ) : ℝ := sin t - t * cos t
def y (t : ℝ) : ℝ := cos t + t * sin t

-- The main theorem: the second derivative y'' with respect to x
theorem second_derivative_yxx (t : ℝ) (ht : t ≠ 0) (ht_cos : cos t ≠ 0) (ht_sin : sin t ≠ 0) : 
  (y xx' t) = -1 / (t * (sin t) ^ 3) := 
sorry

end second_derivative_yxx_l789_789922


namespace triangle_equilateral_if_median_equals_altitude_l789_789774

theorem triangle_equilateral_if_median_equals_altitude
  (A B C D H : Point)
  (h_angle_A : angle A B C = 60)
  (h_median : is_median B D)
  (h_altitude : is_altitude C H)
  (h_equal : dist B D = dist C H) :
  is_equilateral A B C := 
sorry

end triangle_equilateral_if_median_equals_altitude_l789_789774


namespace ellipse_tangent_x_y_axes_has_focus_l789_789966

noncomputable def calculate_d : ℚ :=
  let F1 := (3 : ℚ, 5 : ℚ)
  let F2 := (d : ℚ, 5 : ℚ)
  let C := ((d + F1.1) / 2, 5 : ℚ)
  let T := (C.1, 0 : ℚ)
  d

theorem ellipse_tangent_x_y_axes_has_focus :
  let F1 := (3 : ℚ, 5 : ℚ)
  let F2 := (d : ℚ, 5 : ℚ)
  let C := ((d + F1.1) / 2, 5 : ℚ)
  let T := (C.1, 0 : ℚ)
  2 * real.sqrt(((d - 3) / 2)^2 + (25 : ℚ)) = d + 3 →
  d = 4 / 3 :=
by
  intros
  sorry

end ellipse_tangent_x_y_axes_has_focus_l789_789966


namespace shaded_area_of_hexagon_with_semicircles_l789_789362

theorem shaded_area_of_hexagon_with_semicircles :
  let s := 3
  let r := 3 / 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let semicircle_area := 3 * (1/2 * Real.pi * r^2)
  let shaded_area := hexagon_area - semicircle_area
  shaded_area = 13.5 * Real.sqrt 3 - 27 * Real.pi / 8 :=
by
  sorry

end shaded_area_of_hexagon_with_semicircles_l789_789362


namespace find_triples_l789_789184

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≤ y) (hyz : y ≤ z) 
  (h_eq : x * y + y * z + z * x - x * y * z = 2) : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 2 ∧ y = 3 ∧ z = 4) := 
by 
  sorry

end find_triples_l789_789184


namespace problem_statement_l789_789397

-- Definitions based on given conditions
variables (s : ℝ) 
-- Let ABC be an equilateral triangle (we obtain the necessary properties from it being equilateral)
def is_equilateral_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Defining the triangle DEF
variables {D E F : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F]
def triangle_DEF (D E F : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
  is_equilateral_triangle D E F

-- Extended sides beyond D, E, F
variables (D' E' F' : Type) [MetricSpace D'] [MetricSpace E'] [MetricSpace F']
def extended_sides (D E F D' E' F': Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace D'] [MetricSpace E'] [MetricSpace F'] 
:= (dist D E' = 3 * dist D E) ∧ (dist E F' = 5 * dist E F) ∧ (dist F D' = 4 * dist F D)

-- Final ratio of areas
theorem problem_statement 
  (h1 : is_equilateral_triangle D E F) 
  (h2 : ∀ (s : ℝ), extended_sides D E F D' E' F') : 
  (area_of_triangle D' E' F') / (area_of_triangle D E F) = 16 :=
begin
  sorry
end

end problem_statement_l789_789397


namespace quadrilateral_area_l789_789831

-- Define the conditions and question in Lean 4
def AB := 5
def BC := 4
def CD := 5
def DA := 6
def angle_property_bicentric (AB CD BC DA : ℝ) : Prop := AB + CD = BC + DA
def Ptolemy_theorem (x : ℝ) (AB CD BC DA : ℝ) : Prop := x^2 = AB * CD + BC * DA
def area_quadrilateral (AB BC CD DA : ℝ) : ℝ := 10 * sqrt(6)

-- State that given these conditions, the area of quadrilateral ABCD is 10 * sqrt(6)
theorem quadrilateral_area : 
    angle_property_bicentric AB CD BC DA ∧ Ptolemy_theorem 7 AB CD BC DA 
    → area_quadrilateral AB BC CD DA = 10 * sqrt(6) :=
by
  sorry

end quadrilateral_area_l789_789831


namespace product_of_seventh_and_eighth_games_is_35_l789_789358

-- Definitions based on the conditions
def game_points := [10, 5, 8, 6, 11, 4]

-- Define a condition function to check if the game scores yield integer averages and whether the points scored are valid
def valid_game_scores (pts7 pts8 : Nat) :=
  pts7 < 15 ∧ pts8 < 15 ∧ 
  (44 + pts7) % 7 = 0 ∧
  (44 + pts7 + pts8) % 8 = 0

-- Statement proving the correct answer
theorem product_of_seventh_and_eighth_games_is_35 :
  ∃ pts7 pts8, valid_game_scores pts7 pts8 ∧ (pts7 * pts8 = 35) :=
by
  exists 5, 7
  simp [valid_game_scores]
  sorry

end product_of_seventh_and_eighth_games_is_35_l789_789358


namespace dictionary_prices_and_max_A_l789_789145

-- Definitions for the problem
def price_A := 70
def price_B := 50

-- Conditions from the problem
def condition1 := (price_A + 2 * price_B = 170)
def condition2 := (2 * price_A + 3 * price_B = 290)

-- The proof problem statement
theorem dictionary_prices_and_max_A (h1 : price_A + 2 * price_B = 170) (h2 : 2 * price_A + 3 * price_B = 290) :
  price_A = 70 ∧ price_B = 50 ∧ (∀ (x y : ℕ), x + y = 30 → 70 * x + 50 * y ≤ 1600 → x ≤ 5) :=
by
  sorry

end dictionary_prices_and_max_A_l789_789145


namespace counting_valid_numbers_l789_789734

def is_valid_number (n : ℕ) : Prop := 
  n < 500 ∧ n = 8 * (n.digits 10).sum

theorem counting_valid_numbers : 
  (finset.filter is_valid_number (finset.range 500)).card = 3 := 
sorry

end counting_valid_numbers_l789_789734


namespace max_perimeter_lattice_polygon_l789_789360

theorem max_perimeter_lattice_polygon (m n : ℕ) : 
  ∃ k, k = (m + 1) * (n + 1) + ((-1) ^ ((m + 1) * (n + 1)) - 1) / 2 := sorry

end max_perimeter_lattice_polygon_l789_789360


namespace color_identifiable_set_size_l789_789761

-- Definitions of the conditions
variables (n t : ℕ) (ht : t > 0) (hn : n > 0)

-- Lean statement of the problem to be proved.
theorem color_identifiable_set_size : 
  ∃ g : ℕ, ∀ (n t : ℕ) (ht : t > 0) (hn : n > 0), g = Int.ceil (n / t) :=
sorry

end color_identifiable_set_size_l789_789761


namespace improper_integral_convergence_l789_789387

open Set Filter Topology

variable (a b : ℝ) (α : ℝ)

theorem improper_integral_convergence (h₀ : 0 < α) (h₁ : a < b) :
  (tendsto (λ ε : ℝ, ∫ x in Ioc (a + ε) b, (x - a)⁻¹ ^ α)
            (𝓝[>] 0) (𝓝 (if α < 1 then (b - a) ^ (1 - α) / (1 - α) else ∞))) :=
sorry

end improper_integral_convergence_l789_789387


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789219

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789219


namespace greatest_integer_gcd_l789_789545

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l789_789545


namespace vector_problem_l789_789801

open Real EuclideanGeometry

theorem vector_problem {OM ON : EuclideanSpace ℝ (Fin 3)} (UOM : ‖OM‖ = 1) (UON : ‖ON‖ = 1)
  (angle_MN : angle OM ON = π / 3) (x y : ℝ)
  (OP : EuclideanSpace ℝ (Fin 3) := x • OM + y • ON)
  (rTri : is_right_triangle (OP - OM) (ON - OM) OP OM ON) :
  x - y = 1 := 
sorry

end vector_problem_l789_789801


namespace sum_of_z_values_sum_of_all_z_l789_789404

def f (x : ℝ) : ℝ := x^2 + x + 1

theorem sum_of_z_values (z : ℝ) (h : f (3 * z) = 7): z = 2/9 ∨ z = -1/3 :=
by
  -- Proof skipped
  sorry

theorem sum_of_all_z : ∑ z in {z | f (3 * z) = 7}, z = -1/9 :=
by
  -- Proof skipped
  sorry

end sum_of_z_values_sum_of_all_z_l789_789404


namespace ratio_of_costs_l789_789978

theorem ratio_of_costs (R N : ℝ) (hR : 3 * R = 0.25 * (3 * R + 3 * N)) : N / R = 3 := 
sorry

end ratio_of_costs_l789_789978


namespace area_ratio_of_squares_l789_789058

theorem area_ratio_of_squares (s L : ℝ) 
  (H : 4 * L = 4 * 4 * s) : (L^2) = 16 * (s^2) :=
by
  -- assuming the utilization of the given condition
  sorry

end area_ratio_of_squares_l789_789058


namespace select_2_boys_2_girls_select_4_students_at_least_1_boy_l789_789638

-- Definitions for the problem conditions
def boys : ℕ := 3
def girls : ℕ := 5
def total_students : ℕ := boys + girls

-- Problems to prove
theorem select_2_boys_2_girls : (comb boys 2) * (comb girls 2) = 30 := by
  sorry

theorem select_4_students_at_least_1_boy : (comb total_students 4) - (comb girls 4) = 65 := by
  sorry

end select_2_boys_2_girls_select_4_students_at_least_1_boy_l789_789638


namespace circle_center_radius_find_center_find_radius_l789_789647

theorem circle_center_radius (x y : ℝ) :
  (x^2 + 8*x + y^2 - 4*y = 16) → ((x + 4)^2 + (y - 2)^2 = 36) :=
by
  assume h
  sorry

theorem find_center (x y : ℝ) :
  (x^2 + 8*x + y^2 - 4*y = 16) → (∃ h k : ℝ, (h, k) = (-4, 2)) :=
by
  assume h
  existsi (-4 : ℝ)
  existsi 2
  sorry

theorem find_radius (x y : ℝ) :
  (x^2 + 8*x + y^2 - 4*y = 16) → (∃ r : ℝ, r = 6) :=
by
  assume h
  existsi (6 : ℝ)
  sorry

end circle_center_radius_find_center_find_radius_l789_789647


namespace remainder_mod_105_l789_789041

theorem remainder_mod_105 (x : ℤ) 
  (h1 : 3 + x ≡ 4 [ZMOD 27])
  (h2 : 5 + x ≡ 9 [ZMOD 125])
  (h3 : 7 + x ≡ 25 [ZMOD 343]) :
  x % 105 = 4 :=
  sorry

end remainder_mod_105_l789_789041


namespace trapezium_first_parallel_side_length_l789_789646

-- Define a trapezium with given conditions and prove the length of the first parallel side
theorem trapezium_first_parallel_side_length 
  (a b h A : ℝ)
  (h1 : b = 18)
  (h2 : h = 12)
  (h3 : A = 228)
  : a = 20 :=
begin
  -- The proof would go here, but it's omitted as per instructions.
  sorry
end

end trapezium_first_parallel_side_length_l789_789646


namespace best_value_l789_789593

variables {cS qS cM qL cL : ℝ}
variables (medium_cost : cM = 1.4 * cS) (medium_quantity : qM = 0.7 * qL)
variables (large_quantity : qL = 1.5 * qS) (large_cost : cL = 1.2 * cM)

theorem best_value :
  let small_value := cS / qS
  let medium_value := cM / (0.7 * qL)
  let large_value := cL / qL
  small_value < large_value ∧ large_value < medium_value :=
sorry

end best_value_l789_789593


namespace decimal_representation_prime_has_zeros_l789_789679

theorem decimal_representation_prime_has_zeros (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, 10^2002 ∣ p^n * 10^k :=
sorry

end decimal_representation_prime_has_zeros_l789_789679


namespace football_championship_min_games_l789_789759

theorem football_championship_min_games (n : ℕ) (h : n = 20) :
  ∃ m : ℕ, (∀ s : Finset ℕ, s.card = 3 → ∃ i j, i ≠ j ∧ (i ∈ s) ∧ (j ∈ s) ∧ (game_played i j)) ∧ m ≥ 90 := by
  sorry

def game_played (i j : ℕ) : Prop := sorry

end football_championship_min_games_l789_789759


namespace solve_system_l789_789847

theorem solve_system :
  ∃ (x y : ℤ), 2 * x + y = 4 ∧ x + 2 * y = -1 ∧ x = 3 ∧ y = -2 :=
by
  use [3, -2]
  simp
  ring
  sorry

end solve_system_l789_789847


namespace square_ish_pairs_lt_100_l789_789096

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_square_ish (a b : ℕ) : Prop :=
  a > b ∧ is_square (a + b) ∧ is_square (a - b)

def number_of_square_ish_pairs (n : ℕ) : ℕ :=
  Nat.card {p : ℕ × ℕ // is_square_ish p.1 p.2 ∧ (p.1 + p.2 < n)}

theorem square_ish_pairs_lt_100 : number_of_square_ish_pairs 100 = 16 := by
  sorry

end square_ish_pairs_lt_100_l789_789096


namespace range_of_a_l789_789714

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (1/2 : ℝ) 1, deriv (λ x, real.logb 2 (x^2 - 2 * a * x + 3)) x < 0) ↔ 1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l789_789714


namespace hyperbola_sine_identity_l789_789379

noncomputable def eccentricity (a b c : ℝ) (m : ℝ) : ℝ := c / (2 * real.sqrt m)

noncomputable def hyperbola_sine_relation (e : ℝ) (sin_A sin_B sin_C : ℝ) : Prop :=
  e * |sin_A - sin_B| = sin_C

theorem hyperbola_sine_identity
  (A B C : ℝ) (m n : ℝ) (h_m_pos : m > 0) (h_n_neg : n < 0)
  (sin_A sin_B sin_C : ℝ) (c : ℝ) (e : ℝ)
  (h_e : e = eccentricity A B C m)
  (h_hyperbola : hyperbola_sine_relation e sin_A sin_B sin_C) :
  e * |sin_A - sin_B| = sin_C :=
sorry

end hyperbola_sine_identity_l789_789379


namespace alcohol_exceeds_target_l789_789601

noncomputable def initial_mixture : ℕ := 18
noncomputable def initial_alcohol : ℕ := 4.5
noncomputable def initial_glycerin : ℕ := 5.4
noncomputable def initial_water : ℕ := 8.1

noncomputable def additional_alcohol : ℕ := 3
noncomputable def additional_water : ℕ := 2

noncomputable def total_volume : ℕ := initial_mixture + additional_alcohol + additional_water
noncomputable def total_alcohol : ℕ := initial_alcohol + additional_alcohol
noncomputable def percentage_alcohol : ℚ := (total_alcohol : ℚ) / (total_volume : ℚ) * 100

theorem alcohol_exceeds_target (h1 : percentage_alcohol = 32.61) : percentage_alcohol > 22 :=  
by
  simp
  sorry

end alcohol_exceeds_target_l789_789601


namespace hyperbola_eccentricity_l789_789294

variable {a b c : ℝ} (h_a : a > 0) (h_b : b > 0)
variable (C : Set (ℝ × ℝ)) (h_C : ∀ (x y : ℝ), (x, y) ∈ C ↔ (ℝ × ℝ) := {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
variable (A : ℝ × ℝ := (a, 0))
variable (F : ℝ × ℝ := (c, 0))
variable (B : ℝ × ℝ := (c, b^2 / a))
variable (h_slope : (b^2 / a - 0) / (c - a) = 3)

theorem hyperbola_eccentricity (h_b_square : b^2 = c^2 - a^2) : c / a = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789294


namespace mowing_lawn_time_l789_789425

theorem mowing_lawn_time (mary_time tom_time tom_solo_work : ℝ) 
  (mary_rate tom_rate : ℝ)
  (combined_rate remaining_lawn total_time : ℝ) :
  mary_time = 3 → 
  tom_time = 6 → 
  tom_solo_work = 3 → 
  mary_rate = 1 / mary_time → 
  tom_rate = 1 / tom_time → 
  combined_rate = mary_rate + tom_rate →
  remaining_lawn = 1 - (tom_solo_work * tom_rate) →
  total_time = tom_solo_work + (remaining_lawn / combined_rate) →
  total_time = 4 :=
by sorry

end mowing_lawn_time_l789_789425


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789213

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789213


namespace num_lines_tangent_to_circles_l789_789042

/-- 
Given two points A and B on a plane, 
and given that the distance between A and B is 33 cm, 
every line that is at a distance of 7 cm from point A 
and 26 cm from point B, 
there are exactly 3 such lines. 
-/
theorem num_lines_tangent_to_circles (A B : Type) [Plane A] 
  (d_AB : dist A B = 33)
  (distance_to_A : ℝ := 7)
  (distance_to_B : ℝ := 26) :
  ∃! ℓ : Line, dist ℓ A = distance_to_A ∧ dist ℓ B = distance_to_B :=
sorry

end num_lines_tangent_to_circles_l789_789042


namespace range_of_m_l789_789677

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^2 + 4 * x + 5)
  (h2 : ∀ x : ℝ, f (-2 + x) = f (-2 - x))
  (h3 : ∀ x : ℝ, m ≤ x ∧ x ≤ 0 → 1 ≤ f x ∧ f x ≤ 5)
  : -4 ≤ m ∧ m ≤ -2 :=
  sorry

end range_of_m_l789_789677


namespace sin_inequality_acute_angle_l789_789828

theorem sin_inequality_acute_angle (α : ℝ) (h : 0 < α ∧ α < Real.pi / 2) :
  sin (2 * α) + 2 / sin (2 * α) ≥ 3 :=
  sorry

end sin_inequality_acute_angle_l789_789828


namespace barney_and_regular_dinosaurs_combined_weight_l789_789611

theorem barney_and_regular_dinosaurs_combined_weight :
  (∀ (barney five_dinos_weight) (reg_dino_weight : ℕ),
    (barney = five_dinos_weight + 1500) →
    (five_dinos_weight = 5 * reg_dino_weight) →
    (reg_dino_weight = 800) →
    barney + five_dinos_weight = 9500) :=
by 
  intros barney five_dinos_weight reg_dino_weight h1 h2 h3
  have h4 : five_dinos_weight = 5 * 800 := by rw [h3, mul_comm]
  rw h4 at h2
  have h5 : five_dinos_weight = 4000 := by linarith
  rw h5 at h1
  have h6 : barney = 4000 + 1500 := h1
  have h7 : barney = 5500 := by linarith
  rw [h5, h7]
  linarith

end barney_and_regular_dinosaurs_combined_weight_l789_789611


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789216

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789216


namespace simplify_trig_expr_l789_789443

theorem simplify_trig_expr (theta : ℝ) :
  (tan (2 * π - theta) * sin (-2 * π - theta) * cos (6 * π - theta)) / (cos (theta - π) * sin (5 * π + theta)) = tan theta :=
sorry

end simplify_trig_expr_l789_789443


namespace initial_candies_l789_789164

-- Define the conditions as variables
variable (packs_given_to_sister : Nat)
variable (pieces_per_pack : Nat)
variable (packs_left : Nat)

-- Define the statement to prove
theorem initial_candies (packs_given_to_sister = 1) (pieces_per_pack = 20) (packs_left = 2) : 
  (packs_given_to_sister * pieces_per_pack + packs_left * pieces_per_pack) = 60 := 
sorry

end initial_candies_l789_789164


namespace combined_weight_of_barney_and_five_dinosaurs_l789_789613

theorem combined_weight_of_barney_and_five_dinosaurs:
  let w := 800
  let combined_weight_five_regular := 5 * w
  let barney_weight := combined_weight_five_regular + 1500
  let combined_weight := barney_weight + combined_weight_five_regular
  in combined_weight = 9500 := by
  sorry

end combined_weight_of_barney_and_five_dinosaurs_l789_789613


namespace value_of_b_l789_789356

theorem value_of_b (b : ℝ) : 
  (∃ x : ℝ, y = x^2 - b * x + 8 ∧ x < 0) → b = -4 * real.sqrt 2 :=
sorry

end value_of_b_l789_789356


namespace largest_integer_remainder_condition_l789_789260

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789260


namespace find_a8_l789_789411

-- Define the sequence
def a : ℕ → ℕ
| 0 := sorry  -- initial value a_1
| 1 := sorry  -- initial value a_2
| n + 2 := a n + a (n + 1)

-- Provide initial values
axiom a1_pos (h1: a 0 > 0) : a1_pos
axiom a2_pos (h2: a 1 > 0) : a2_pos

-- Provide for a_7 condition
axiom a7_eq (h7: a 6 = 240) : a7_eq

-- Theorem stating the desired proof
theorem find_a8 (h1: a 0 > 0) (h2: a 1 > 0) (h7: a 6 = 240) : a 7 = 386 :=
sorry

end find_a8_l789_789411


namespace countSumPairs_correct_l789_789764

def countSumPairs (n : ℕ) : ℕ :=
  n / 2

theorem countSumPairs_correct (n : ℕ) : countSumPairs n = n / 2 := by
  sorry

end countSumPairs_correct_l789_789764


namespace sqrt_two_irrational_triangle_s_lt_2a_l789_789136

-- Statement 1: Prove that √2 is irrational
theorem sqrt_two_irrational : ¬ ∃ (m n: ℕ), Nat.gcd m n = 1 ∧ (√2: ℝ) = m / n :=
by
  sorry

-- Statement 2: Prove s < 2a under given conditions
theorem triangle_s_lt_2a (a b c s: ℝ) (h₁: a + b > c) (h₂: s = (a + b + c) / 2) (h₃: s^2 = 2 * a * b) : s < 2 * a :=
by
  sorry

end sqrt_two_irrational_triangle_s_lt_2a_l789_789136


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789232

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789232


namespace sum_of_z_values_sum_of_all_z_l789_789403

def f (x : ℝ) : ℝ := x^2 + x + 1

theorem sum_of_z_values (z : ℝ) (h : f (3 * z) = 7): z = 2/9 ∨ z = -1/3 :=
by
  -- Proof skipped
  sorry

theorem sum_of_all_z : ∑ z in {z | f (3 * z) = 7}, z = -1/9 :=
by
  -- Proof skipped
  sorry

end sum_of_z_values_sum_of_all_z_l789_789403


namespace painting_time_l789_789835

noncomputable def work_rate (t : ℕ) : ℚ := 1 / t

theorem painting_time (shawn_time karen_time alex_time total_work_rate : ℚ)
  (h_shawn : shawn_time = 18)
  (h_karen : karen_time = 12)
  (h_alex : alex_time = 15) :
  total_work_rate = 1 / (shawn_time + karen_time + alex_time) :=
by
  sorry

end painting_time_l789_789835


namespace smaller_octagon_area_ratio_l789_789479

theorem smaller_octagon_area_ratio 
  (ABCDEFGH : Type) 
  [regular_octagon ABCDEFGH]
  (midpoints : ∀ (A B : ABCDEFGH), midpoint A B → smaller_octagon)
  : ∃ smaller_octagon,
    (area(smaller_octagon) / area(ABCDEFGH)) = 1/4 :=
sorry

end smaller_octagon_area_ratio_l789_789479


namespace numIrrationalNumbers_l789_789964

def isRational (x : ℝ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q

def isIrrational (x : ℝ) : Prop := ¬ isRational x

theorem numIrrationalNumbers : 
    let nums := [3.1415, 0.2060060006, 0, (0.2).repeat, -Real.pi, Real.cbrt 5, 22 / 7, Real.sqrt 7, Real.sqrt 64] in
    (nums.filter isIrrational).length = 3 :=
sorry

end numIrrationalNumbers_l789_789964


namespace planes_parallel_if_any_line_parallel_l789_789825

axiom Plane : Type
axiom Line : Type
axiom contains : Plane → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_lines : Line → Plane → Prop

theorem planes_parallel_if_any_line_parallel (α β : Plane)
  (h₁ : ∀ l, contains α l → parallel_lines l β) :
  parallel α β :=
sorry

end planes_parallel_if_any_line_parallel_l789_789825


namespace trisha_spent_on_eggs_l789_789504

def totalSpent (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ) : ℕ :=
  initialAmount - (meat + chicken + veggies + dogFood + amountLeft)

theorem trisha_spent_on_eggs :
  ∀ (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ),
    meat = 17 →
    chicken = 22 →
    veggies = 43 →
    dogFood = 45 →
    amountLeft = 35 →
    initialAmount = 167 →
    totalSpent meat chicken veggies eggs dogFood amountLeft initialAmount = 5 :=
by
  intros meat chicken veggies eggs dogFood amountLeft initialAmount
  sorry

end trisha_spent_on_eggs_l789_789504


namespace function_period_roots_l789_789814

theorem function_period_roots
  (f : ℝ → ℝ)
  (h1 : ∀ x, f(x) = f(4-x))
  (h2 : ∀ x, f(7-x) = f(7+x))
  (h3 : f(1) = 0)
  (h4 : f(3) = 0):
  (∃ T > 0, ∀ x, f(x) = f(x + T) ∧ T = 10) ∧ (∀ x ∈ (-2005 : ℤ).fintype, x ∈ finset.Icc (-2005) 2005 → f x = 0) :=
sorry

end function_period_roots_l789_789814


namespace set_list_method_l789_789194

theorem set_list_method : 
  {x : ℝ | x^2 - 2 * x + 1 = 0} = {1} :=
sorry

end set_list_method_l789_789194


namespace grasshopper_ways_eq_fib_l789_789968

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

-- Define the number of ways the grasshopper can reach the nth cell
def ways_to_reach : ℕ → ℕ
| 1 := 1
| 2 := 1
| (n + 3) := ways_to_reach (n + 2) + ways_to_reach (n + 1)

-- Proposition that the number of ways to reach the nth cell is the (n-1)th Fibonacci number
theorem grasshopper_ways_eq_fib (n : ℕ) : ways_to_reach n = fib (n - 1) :=
  sorry

end grasshopper_ways_eq_fib_l789_789968


namespace greatest_int_with_gcd_five_l789_789540

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l789_789540


namespace hyperbola_eccentricity_l789_789309

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (A B : ℝ × ℝ) (F : ℝ × ℝ)
variable (hA : A = (a, 0))
variable (hF : F = (c, 0))
variable (hB : B = (c, b ^ 2 / a))
variable (h_slope : (b ^ 2 / a - 0) / (c - a) = 3)
variable (h_hyperbola : b ^ 2 = c ^ 2 - a ^ 2)
def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity : eccentricity a c = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789309


namespace log_expression_evaluation_l789_789982

theorem log_expression_evaluation : 
  log 3 (427 / 3) + log 10 25 + 2 * log 10 2 + Real.exp (Real.log 2) = 15 / 4 :=
by 
  sorry

end log_expression_evaluation_l789_789982


namespace correct_graph_for_race_l789_789594

-- Define the conditions for the race.
def tortoise_constant_speed (d t : ℝ) := 
  ∃ k : ℝ, k > 0 ∧ d = k * t

def hare_behavior (d t t_nap t_end d_nap : ℝ) :=
  ∃ k1 k2 : ℝ, k1 > 0 ∧ k2 > 0 ∧ t_nap > 0 ∧ t_end > t_nap ∧
  (d = k1 * t ∨ (t_nap < t ∧ t < t_end ∧ d = d_nap) ∨ (t_end ≥ t ∧ d = d_nap + k2 * (t - t_end)))

-- Define the competition outcome.
def tortoise_wins (d_tortoise d_hare : ℝ) :=
  d_tortoise > d_hare

-- Proof that the graph which describes the race is Option (B).
theorem correct_graph_for_race :
  ∃ d_t d_h t t_nap t_end d_nap, 
    tortoise_constant_speed d_t t ∧ hare_behavior d_h t t_nap t_end d_nap ∧ tortoise_wins d_t d_h → "Option B" = "correct" :=
sorry -- Proof omitted.

end correct_graph_for_race_l789_789594


namespace sum_E_eq_1000_2n_minus_1_2_999n_l789_789409

open Finset

noncomputable def S (n : ℕ) : Finset (Fin (n+1) → Finset (Fin 1000)) := univ

def E {n : ℕ} (a : Fin (n+1) → Finset (Fin 1000)) : ℕ := (Finset.univ.bUnion a).card

theorem sum_E_eq_1000_2n_minus_1_2_999n (n : ℕ) :
  ∑ a in S n, E a = 1000 * (2^n - 1) * 2^(999 * n) := sorry

end sum_E_eq_1000_2n_minus_1_2_999n_l789_789409


namespace vector_CB_correct_l789_789347

-- Define the vectors AB and AC
def AB : ℝ × ℝ := (2, 3)
def AC : ℝ × ℝ := (-1, 2)

-- Define the vector CB as the difference of AB and AC
def CB (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Prove that CB = (3, 1) given AB and AC
theorem vector_CB_correct : CB AB AC = (3, 1) :=
by
  sorry

end vector_CB_correct_l789_789347


namespace kayak_rental_cost_is_18_l789_789095

def cost_per_kayak (canoe_cost kayak_rentals: ℕ) (total_revenue: ℕ) (canoe_kayak_diff: ℕ): ℕ :=
  let canoes := (canoe_rentals: ℕ) = (3/2: ℕ) * kayak_rentals
  let total_cost := (canoes * canoe_cost) + (kayak_rentals * kayak_rentals_price)
  if total_cost == total_revenue ∧ kayak_rentals + 5 == canoes then kayak_rentals_price = 18 else sorry

theorem kayak_rental_cost_is_18:
  cost_per_kayak 15 10 405 5 = 18 :=
begin
  have h1: 3/2 * 10 = 15, by norm_num,
  have h2: 15 * 15 + 10 * 18 = 405, by norm_num,
  exact h2,
sorry 
end

end kayak_rental_cost_is_18_l789_789095


namespace four_digit_number_count_l789_789733

theorem four_digit_number_count :
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  (card {n : ℕ | (1000 ≤ n ∧ n < 10000) ∧ 
                 (∃ d1 d2 d3 d4 : ℕ, n = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 
                 d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ 
                 d1 % 3 = 0 ∧ 
                 d2 % 2 = 1 ∧ 
                 d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4))} 
  = 504 := sorry

end four_digit_number_count_l789_789733


namespace hyperbola_eccentricity_l789_789302

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (h_hyp : a^2 - y^2 = b^2)
variable (hF : (c, 0) ∈ hyperbola(a,b))
variable (hA : (a, 0) ∈ hyperbola(a,b))
variable (hB : (c, b^2 / a) ∈ hyperbola(a,b))
variable (h_slope : (b^2 / a) / (c - a) = 3)

theorem hyperbola_eccentricity (ha : 0 < a) (hb : 0 < b) (hF : (c, 0))
    (hA : (a, 0)) (hB : (c, b^2 / a))
    (h_slope : (b^2 / a) / (c - a) = 3) : (eccentricity(c, a) = 2) := by
  sorry

end hyperbola_eccentricity_l789_789302


namespace rectangle_area_k_l789_789888

noncomputable def rectangle_k (d : ℝ) (length width : ℝ) (ratio : length / width = 5 / 2) 
  (diagonal : d = Real.sqrt (length^2 + width^2)) : ℝ := 
  (length * width / d^2)

theorem rectangle_area_k {d length width : ℝ} 
  (h_ratio : length / width = 5 / 2)
  (h_diagonal : d = Real.sqrt (length^2 + width^2)) :
  rectangle_k d length width h_ratio h_diagonal = 10 / 29 :=
by
  sorry

end rectangle_area_k_l789_789888


namespace projection_eq_l789_789652

-- Vector a
def a : ℝ × ℝ := (3, -3)
-- Vector b
def b : ℝ × ℝ := (5, 1)

-- Definition of projection
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := (u.1 * v.1 + u.2 * v.2)
  let norm_squared := (v.1 * v.1 + v.2 * v.2)
  (dot_product / norm_squared * v.1, dot_product / norm_squared * v.2)

theorem projection_eq :
  proj a b = (30/13, 6/13) :=
  sorry

end projection_eq_l789_789652


namespace sum_of_powers_eq_neg_one_l789_789319

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  log ((3^x + 1) / logBase (1/3)) + (1/2) * a * b * x

noncomputable def g (x : ℝ) (a b : ℝ) : ℝ :=
  2^x + (a + b) / 2^x

theorem sum_of_powers_eq_neg_one
  (a b : ℝ)
  (h1 : ∀ x, f x a b = f (-x) a b)
  (h2 : ∀ x, g x a b = -g (-x) a b)
  (h_ab1 : a + b = -1)
  (h_ab2 : a * b = 1) :
  (∑ k in Finset.range 2008, a^k + b^k) = -1 :=
sorry

end sum_of_powers_eq_neg_one_l789_789319


namespace evaluate_nested_radical_l789_789641

theorem evaluate_nested_radical : ∃ x : ℝ, x = sqrt (16 + x) ∧ x = (1 + sqrt 65) / 2 :=
by
  use (1 + sqrt 65) / 2
  split
  { sorry }
  { sorry }

end evaluate_nested_radical_l789_789641


namespace cleaning_time_is_one_hour_l789_789786

def trees_rows : ℕ := 4
def trees_cols : ℕ := 5
def minutes_per_tree : ℕ := 6
noncomputable def help : ℝ := 1 / 2

noncomputable def total_trees : ℕ := trees_rows * trees_cols
noncomputable def total_minutes_without_help : ℕ := total_trees * minutes_per_tree
noncomputable def total_hours_without_help : ℝ := total_minutes_without_help / 60
noncomputable def total_hours_with_help : ℝ := total_hours_without_help * help

theorem cleaning_time_is_one_hour : total_hours_with_help = 1 :=
by {
  have h1 : total_trees = 20 := rfl,
  have h2 : total_minutes_without_help = 120 := rfl,
  have h3 : total_hours_without_help = 2 := by norm_num,
  have h4 : total_hours_with_help = 1 := by norm_num,
  exact h4,
}

end cleaning_time_is_one_hour_l789_789786


namespace factorial_power_sum_l789_789629

def legendre_formula (n p: ℕ) : ℕ :=
  if p < 2 then 0 else ((list.finRange (int.log p n).toNat).map (λ k, n / p^k)).sum

noncomputable def highest_power_10_in_20 : ℕ :=
  min (legendre_formula 20 2) (legendre_formula 20 5)

noncomputable def highest_power_3_in_20 : ℕ := 
  legendre_formula 20 3

theorem factorial_power_sum : highest_power_10_in_20 + highest_power_3_in_20 = 12 :=
  sorry

end factorial_power_sum_l789_789629


namespace g_value_l789_789873

theorem g_value (g : ℝ → ℝ)
  (h0 : g 0 = 0)
  (h_mono : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (h_symm : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (h_prop : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 5) = 1 / 2 :=
sorry

end g_value_l789_789873


namespace find_largest_integer_l789_789250

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789250


namespace f_neither_even_nor_odd_l789_789134

def f (x : ℝ) : ℝ := 3^(x^2 - 3*x + 1) - |x|

theorem f_neither_even_nor_odd : (∀ x : ℝ, f x ≠ f (-x)) ∧ (∀ x : ℝ, f x ≠ -f (-x)) :=
by
  intros x
  sorry

end f_neither_even_nor_odd_l789_789134


namespace paths_order_correct_l789_789976

theorem paths_order_correct :
  let paths_through (p : Point) := number_of_paths_through p in
  sorted (λ x y, paths_through x > paths_through y) [A, F, C, G, E, D, B] :=
by
  -- Using the conditions from the problem statement
  have hA : paths_through A = 462 := sorry
  have hB : paths_through B = 1 := sorry
  have hC : paths_through C = 350 := sorry
  have hD : paths_through D = 105 := sorry
  have hE : paths_through E = 224 := sorry
  have hF : paths_through F = 400 := sorry
  have hG : paths_through G = 225 := sorry
  -- Proving the sorted order
  show 
    [paths_through A, paths_through F, paths_through C, paths_through G, paths_through E, paths_through D, paths_through B] =
    [462, 400, 350, 225, 224, 105, 1] from 
  -- fill in with proper mathematical deduction
  sorry

end paths_order_correct_l789_789976


namespace find_m_l789_789925

noncomputable def factor_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, (x + 3) * (x + a) = x^2 - m * x - 15

theorem find_m (m : ℝ) : (factor_condition m) → m = 2 :=
by
  intro h
  sorry

end find_m_l789_789925


namespace k_combinations_k_combinations_repetition_ten_combinations_l789_789650

-- Problem 1
theorem k_combinations (n k : ℕ) : (@finset.powerset_len n k (finset.range n)).card = nat.choose n k := 
sorry

-- Problem 2
theorem k_combinations_repetition (n k : ℕ) : 
 (@finset.card {s : multiset (fin n) // s.card = k } (multiset.card s) = nat.choose (k + n - 1) (n - 1)) :=
  sorry

-- Problem 3
theorem ten_combinations : @finset.card ({a : ℕ × ℕ × ℕ // a.0 + a.1 + a.2 = 10 ∧ a.0 ≤ 3 ∧ a.1 ≤ 4 ∧ a.2 ≤ 5}) 6 :=
 sorry

end k_combinations_k_combinations_repetition_ten_combinations_l789_789650


namespace largest_integer_remainder_condition_l789_789263

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789263


namespace soccer_game_goals_l789_789189

theorem soccer_game_goals (A1_first_half A2_first_half B1_first_half B2_first_half : ℕ) 
  (h1 : A1_first_half = 8)
  (h2 : B1_first_half = A1_first_half / 2)
  (h3 : B2_first_half = A1_first_half)
  (h4 : A2_first_half = B2_first_half - 2) : 
  A1_first_half + A2_first_half + B1_first_half + B2_first_half = 26 :=
by
  -- The proof is not needed, so we use sorry to skip it.
  sorry

end soccer_game_goals_l789_789189


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l789_789534

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l789_789534


namespace age_sum_in_5_years_l789_789820

variable (MikeAge MomAge : ℕ)
variable (h1 : MikeAge = MomAge - 30)
variable (h2 : MikeAge + MomAge = 70)

theorem age_sum_in_5_years (h1 : MikeAge = MomAge - 30) (h2 : MikeAge + MomAge = 70) :
  (MikeAge + 5) + (MomAge + 5) = 80 := by
  sorry

end age_sum_in_5_years_l789_789820


namespace parabolas_intersect_eq_l789_789505

noncomputable def parabolas_intersect_probability :
  (fin 6 → ℤ) → ℚ :=
  λ choices,
  let a := choices 0,
      b := choices 1,
      c := choices 2,
      d := choices 3 in
  -- Indicator function for intersection condition
  if (a + c ≠ 0) ∨ (d = b) then 211 / 216 else 0.

-- Proposition statement
theorem parabolas_intersect_eq :
  ∀ choices: fin 6 → ℤ, 
  choices ∈ ({i | 0 ≤ i ∧ i ≤ 5}^4 : set (fin 6 → ℤ)) →
  parabolas_intersect_probability choices = 211 / 216 :=
begin
  assume choices h_choices,
  sorry -- actual proof omitted
end

end parabolas_intersect_eq_l789_789505


namespace exists_special_member_l789_789688

theorem exists_special_member :
  ∃ (country : ℕ → ℤ) (member_number : ℤ),
    (1 ≤ member_number ∧ member_number ≤ 1978) ∧
    (∀ n, 1 ≤ n ∧ n ≤ 1978 → country n ∈ {A: ℕ | 1 ≤ A ∧ A ≤ 6}) ∧ 
    (country member_number ≠ country 0 → member_number ∉ {a + b | a b: ℤ, 1 ≤ a ∧ a ≤ 1978 ∧ 1 ≤ b ∧ b ≤ 1978 ∧ country a = country b}) ∧ 
    (country member_number ≠ country 0 → member_number ∉ {2 * a | a: ℤ, 1 ≤ a ∧ a ≤ 1978 ∧ country a = country member_number}) sorry

end exists_special_member_l789_789688


namespace part1_part2_l789_789708

noncomputable def z (a : ℝ) : ℂ := (1 - complex.I) * a^2 - 3 * a + 2 + complex.I

theorem part1 (a : ℝ) (h_imaginary : z a = (1 - a^2) * complex.I) : z a = -3 * complex.I := by
  sorry

theorem part2 (a : ℝ) (h_third_quadrant : (z a).re < 0 ∧ (z a).im < 0) : 1 < a ∧ a < 2 := by
  sorry

end part1_part2_l789_789708


namespace circumcircle_area_l789_789578


open Real

theorem circumcircle_area (A B C: Point) (r: ℝ) (AB AC BC: Segment)
  (hAB: length AB = 4) (hAC: length AC = 4) (hBC: length BC = 3):
  area (Circle.mk (midpoint A B) r) = 4 * π :=
by
  sorry

end circumcircle_area_l789_789578


namespace exists_n_consecutive_non_prime_powers_l789_789836

theorem exists_n_consecutive_non_prime_powers (n : ℕ) (h : n > 0) : 
  ∃ (x : ℕ), ∀ (i : ℕ), (i < n → ¬ (∃ (p : ℕ) (a : ℕ), prime p ∧ x + i = p^a ∧ a ≥ 1)) := 
by
  sorry

end exists_n_consecutive_non_prime_powers_l789_789836


namespace disk_division_max_areas_l789_789935

theorem disk_division_max_areas (n : ℕ) (h : n > 0) :
  max_non_overlapping_areas 2 * n 2 = 3 * n + 4 :=
sorry

end disk_division_max_areas_l789_789935


namespace cleaning_time_is_one_hour_l789_789785

def trees_rows : ℕ := 4
def trees_cols : ℕ := 5
def minutes_per_tree : ℕ := 6
noncomputable def help : ℝ := 1 / 2

noncomputable def total_trees : ℕ := trees_rows * trees_cols
noncomputable def total_minutes_without_help : ℕ := total_trees * minutes_per_tree
noncomputable def total_hours_without_help : ℝ := total_minutes_without_help / 60
noncomputable def total_hours_with_help : ℝ := total_hours_without_help * help

theorem cleaning_time_is_one_hour : total_hours_with_help = 1 :=
by {
  have h1 : total_trees = 20 := rfl,
  have h2 : total_minutes_without_help = 120 := rfl,
  have h3 : total_hours_without_help = 2 := by norm_num,
  have h4 : total_hours_with_help = 1 := by norm_num,
  exact h4,
}

end cleaning_time_is_one_hour_l789_789785


namespace cassy_initial_jars_l789_789173

theorem cassy_initial_jars (boxes1 jars1 boxes2 jars2 leftover: ℕ) (h1: boxes1 = 10) (h2: jars1 = 12) (h3: boxes2 = 30) (h4: jars2 = 10) (h5: leftover = 80) : 
  boxes1 * jars1 + boxes2 * jars2 + leftover = 500 := 
by 
  sorry

end cassy_initial_jars_l789_789173


namespace unique_polynomial_P_l789_789265

/-- Prove the unique nonconstant polynomial P(x) which satisfies P(P(x)) = (x^2 + x + 1) P(x)
is P(x) = x^2 + x. -/
theorem unique_polynomial_P (P : Polynomial ℝ) (h : ¬(Polynomial.degree P = 0)) (h_eq : P.comp(P) = (Polynomial.X^2 + Polynomial.X + 1) * P) :
  P = Polynomial.X^2 + Polynomial.X := 
sorry

end unique_polynomial_P_l789_789265


namespace regular_tetrahedron_edges_l789_789732

theorem regular_tetrahedron_edges : ∀ (T : Type) [IsRegularTetrahedron T], edges T = 6 :=
by
  sorry

end regular_tetrahedron_edges_l789_789732


namespace find_radius_of_circle_l789_789125

theorem find_radius_of_circle :
  ∃ (r : ℤ), circle ( -2, -3 ) (r : ℤ)  ∧ r ∈ s :=
by
-- Define the center of the circle
let center := (-2, -3 : ℤ × ℤ),

-- Define the point inside the circle
let point_inside := (-2, 2 : ℤ × ℤ),

-- Define the point outside the circle
let point_outside := (5, -3 : ℤ × ℤ),

-- Calculate the distance from the center to the point inside (should be less than r)
let distance_inside := (point_inside.1 - center.1) ^ 2 + (point_inside.2 - center.2) ^ 2,

-- Calculate the distance from the center to the point outside (should be more than r)
let distance_outside := (point_outside.1 - center.1) ^ 2 + (point_outside.2 - center.2) ^ 2,

-- Given distance_inside = 25 (5^2) and distance_outside = 49 (7^2), 
-- the radius r must satisfy 5 < r < 7 and r is an integer
r = 6,
exists.intro (r : ℤ)

sorry

end find_radius_of_circle_l789_789125


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789515

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789515


namespace part_a_part_b_l789_789395

def g (n : ℕ) : ℕ := (n.digits 10).prod

theorem part_a : ∀ n : ℕ, g n ≤ n :=
by
  -- Proof omitted
  sorry

theorem part_b : {n : ℕ | n^2 - 12*n + 36 = g n} = {4, 9} :=
by
  -- Proof omitted
  sorry

end part_a_part_b_l789_789395


namespace trolls_count_20_l789_789997

noncomputable def num_creatures := 60
noncomputable def trolls_lie := true
noncomputable def elves_truth := true
noncomputable def elves_mistake := 2
noncomputable def creatures_claim := true

theorem trolls_count_20 :
  ∃ t : ℕ, t = 20 ∧ 
    (t <= num_creatures ∧
     elves_mistake = 2 ∧
     ∀ c, c < num_creatures → creatures_claim) := 
begin
  existsi 20,
  split,
  { refl },
  split,
  { exact nat.le_refl 20 },
  split,
  { exact rfl },
  { intros, exact true.intro }
end

end trolls_count_20_l789_789997


namespace b_arithmetic_sum_a_seq_l789_789722

-- Define the sequence a_n
def a_seq : ℕ → ℕ
| 1 := 5
| (n + 1) := 2 * a_seq n + 2^(n + 1) - 1

-- Define b_n = (a_n - 1) / 2^n
def b_seq (n : ℕ) : ℕ :=
  (a_seq n - 1) / 2^n

-- (1) Prove that b_seq is an arithmetic sequence
theorem b_arithmetic : ∀ n : ℕ, b_seq (n + 1) = b_seq n + 1 :=
by sorry

-- Define the sum of the first n terms of a_seq
def S_n (n : ℕ) : ℕ :=
  ∑ i in finset.range n, a_seq (i + 1)

-- (2) Prove that S_n = n * (2^(n+1) + 1)
theorem sum_a_seq : ∀ n : ℕ, S_n n = n * (2^(n + 1) + 1) :=
by sorry

end b_arithmetic_sum_a_seq_l789_789722


namespace polynomial_division_remainder_l789_789635

noncomputable theory

open Polynomial

theorem polynomial_division_remainder :
  let P := (X^6 + X^5 + 2*X^3 - X^2 + 3)
  let D := ((X + 2) * (X - 1))
  let R := (-X + 5)
  ∃ Q : Polynomial ℚ, P = D * Q + R :=
by
  sorry

end polynomial_division_remainder_l789_789635


namespace elisa_total_paint_area_l789_789428

def area_rect (length : ℝ) (width : ℝ) : ℝ :=
  length * width

def area_triangle (base : ℝ) (height : ℝ) : ℝ :=
  1/2 * base * height

theorem elisa_total_paint_area :
  let monday_area := area_rect 8 6 in
  let tuesday_first_area := area_rect 12 4 in
  let tuesday_second_area := area_rect 6 6 in
  let wednesday_area := area_triangle 10 4 in
  let total_area := monday_area + tuesday_first_area + tuesday_second_area + wednesday_area in
  total_area = 152 :=
by
  let monday_area := area_rect 8 6
  let tuesday_first_area := area_rect 12 4
  let tuesday_second_area := area_rect 6 6
  let wednesday_area := area_triangle 10 4
  let total_area := monday_area + tuesday_first_area + tuesday_second_area + wednesday_area
  show total_area = 152
  sorry

end elisa_total_paint_area_l789_789428


namespace largest_integer_remainder_condition_l789_789256

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789256


namespace oliver_mom_gave_32_dollars_l789_789022

theorem oliver_mom_gave_32_dollars :
  let initial := 33 in
  let spent := 4 in
  let final := 61 in
  let mom_gave := final - (initial - spent) in
  mom_gave = 32 :=
by
  let initial := 33
  let spent := 4
  let final := 61
  let mom_gave := final - (initial - spent)
  show mom_gave = 32
  sorry

end oliver_mom_gave_32_dollars_l789_789022


namespace find_largest_integer_l789_789255

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789255


namespace specific_value_is_19_l789_789910

def specific_value (x : ℕ) : ℕ := x + 3 + 5

theorem specific_value_is_19 (x : ℕ) (h : x = 11) : specific_value x = 19 :=
by
  rw [h]
  dsimp [specific_value]
  norm_num

end specific_value_is_19_l789_789910


namespace locus_of_vertex_C_fixed_base_and_median_l789_789157

theorem locus_of_vertex_C_fixed_base_and_median 
  (A : Point) (B C : Point) 
  (AB_length : dist A B = 6)
  (median_length : ∃ D : Point, dist A D = 4 ∧ ∠(A, D, B) = 60) :
  ∃ r : ℝ, r = 7 ∧ ∀ C, dist A C = r :=
sorry

end locus_of_vertex_C_fixed_base_and_median_l789_789157


namespace population_increase_l789_789669

theorem population_increase (i j : ℝ) : 
  ∀ (m : ℝ), m * (1 + i / 100) * (1 + j / 100) = m * (1 + (i + j + i * j / 100) / 100) := 
by
  intro m
  sorry

end population_increase_l789_789669


namespace polynomial_real_coeff_of_conditions_l789_789809

noncomputable theory

open Complex Polynomial

def is_real_polynomial (f : Polynomial ℂ) : Prop :=
  ∀ (a : ℂ), is_real a → f.is_root a → is_real (f a)

theorem polynomial_real_coeff_of_conditions (f : Polynomial ℂ) (n : ℕ) :
  (degree f = 2 * n + 1) →
  (∀ {z : ℂ}, Im z ≠ 0 → z ≠ 0 → f.is_root (conj z) → f.is_root z) →
  (∃ (roots : Fin (n + 1) → ℂ), (∀ i j, i ≠ j → abs (roots i) ≠ abs (roots j)) ∧ (∀ i, f.is_root (roots i))) →
  ∀ k, (f.coeff k = conj(f.coeff k)) :=
begin
  intros h_deg h_cond h_imag_root,
  sorry
end

end polynomial_real_coeff_of_conditions_l789_789809


namespace minimum_value_xsq_plus_ysq_l789_789695

theorem minimum_value_xsq_plus_ysq (x y : ℝ) (h : x^2 + 2 * (sqrt 3) * x * y - y^2 = 1) : x^2 + y^2 ≥ 1 / 2 :=
sorry

end minimum_value_xsq_plus_ysq_l789_789695


namespace first_nonzero_digit_fraction_one_over_197_l789_789512

theorem first_nonzero_digit_fraction_one_over_197 : 
  ∃ d : ℤ, d ≠ 0 ∧ (1 / 197 : ℚ) * 10 ^ (find (λ n, exists_digit (10^n * (1 / 197) % 10)) = d) = 5 :=
sorry

end first_nonzero_digit_fraction_one_over_197_l789_789512


namespace period_of_tan_x_div_3_l789_789109

theorem period_of_tan_x_div_3 : ∃ T > 0, ∀ x, tan (x / 3) = tan ((x + T) / 3) :=
by
  use 3 * Real.pi
  intros x
  rew_rw (3 * Real.pi)
  rw [Real.tan_periodic]
  sorry

end period_of_tan_x_div_3_l789_789109


namespace slices_leftover_l789_789957

def total_slices (small_pizzas large_pizzas : ℕ) : ℕ :=
  (3 * 4) + (2 * 8)

def slices_eaten_by_people (george bob susie bill fred mark : ℕ) : ℕ :=
  george + bob + susie + bill + fred + mark

theorem slices_leftover :
  total_slices 3 2 - slices_eaten_by_people 3 4 2 3 3 3 = 10 :=
by sorry

end slices_leftover_l789_789957


namespace polygon_covered_by_circle_l789_789035

-- Definition of a polygon's perimeter being 1 unit.
def isUnitPerimeter (poly: Type) [CommutativeAddGroup poly] : Prop :=
  True -- Assuming necessary conditions for defining a unit perimeter polygon.

-- Statement that there exists a point O such that the circle of radius (1/4) covers the polygon.
theorem polygon_covered_by_circle (poly: Type) [CommRing poly] (h : isUnitPerimeter poly) :
  ∃ O : poly, ∀ P : poly, ∃ r : ℝ, (r = 1/4) ∧ (dist O P ≤ r) ∧ (dist P O ≤ 1/4) :=
sorry

end polygon_covered_by_circle_l789_789035


namespace sides_form_consecutive_integers_l789_789775

theorem sides_form_consecutive_integers (n : ℕ) (hn : n = 33) :
  ∃ (perm : Fin n → ℕ), 
    (∀ i : Fin n, 1 ≤ perm i ∧ perm i ≤ n) ∧ 
    (∀ i j : Fin n, perm i = perm j → i = j) ∧ 
    (∀ i : Fin n, 
      let j := if i.1 = n - 1 then ⟨0, Nat.zero_lt_succ _⟩ else ⟨i.1 + 1, Nat.lt_succ_self _⟩ in
      ∃ k : ℕ, 
        (∀ x, ⟨x, Nat.lt_succ_self x⟩ ∈ Finset.Ico 1 34 → 
          Finset.sum (Finset.range n (λ i, perm i)) (λ x, perm x + perm (if x = n-1 then 0 else x+1)) = k) ∧ 
        ∀ (a : ℕ), (a ∈ Finset.Ico 18 50) → (∃ b : ℕ, b ∈ Finset.Ico 1 34 ∧ 
          (perm b + perm (if b = n - 1 then 0 else b + 1)) = a)) :=
sorry

end sides_form_consecutive_integers_l789_789775


namespace rectangle_difference_l789_789864

theorem rectangle_difference (L B D : ℝ)
  (h1 : L - B = D)
  (h2 : 2 * (L + B) = 186)
  (h3 : L * B = 2030) :
  D = 23 :=
by
  sorry

end rectangle_difference_l789_789864


namespace value_of_each_gift_card_l789_789821

theorem value_of_each_gift_card (students total_thank_you_cards with_gift_cards total_value : ℕ) 
  (h1 : students = 50)
  (h2 : total_thank_you_cards = 30 * students / 100)
  (h3 : with_gift_cards = total_thank_you_cards / 3)
  (h4 : total_value = 50) :
  total_value / with_gift_cards = 10 := by
  sorry

end value_of_each_gift_card_l789_789821


namespace difference_of_parallel_lines_l789_789857

variable (A B C D E F G : Type) (length : Type) [LinearOrder length] [Real length]
variable (AB AC BC DE FG : length)
variable (h1 : IsoscelesTriangle A B C AB AC BC)
variable (base_length : BC = 20)
variable (h2 : Parallel DE BC)
variable (h3 : Parallel FG BC)
variable (h4 : DivideEqualAreaTriangle DE FG (Triangle ABC))

theorem difference_of_parallel_lines :
  FG - DE = 20 * (Real.sqrt 6 - Real.sqrt 3) / 3 :=
by
  sorry

end difference_of_parallel_lines_l789_789857


namespace greatest_integer_gcd_l789_789547

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l789_789547


namespace rectangle_sum_l789_789945

-- Define the vertices of the rectangle
def A := (1, 1)
def B := (1, 5)
def C := (6, 5)
def D := (6, 1)

-- Define the length of the sides of the rectangle
def length : ℕ := abs (6 - 1) -- Length of horizontal side
def width : ℕ := abs (5 - 1)  -- Length of vertical side

-- Define the perimeter and area of the rectangle
def perimeter : ℕ := 2 * (length + width)
def area : ℕ := length * width

-- Main statement: the sum of the perimeter and the area
theorem rectangle_sum :
  perimeter + area = 38 := by
  -- Proof goes here
  sorry

end rectangle_sum_l789_789945


namespace smallest_positive_m_l789_789909

theorem smallest_positive_m (m : ℕ) :
  (∃ (r s : ℤ), 18 * r * s = 252 ∧ m = 18 * (r + s) ∧ r ≠ s) ∧ m > 0 →
  m = 162 := 
sorry

end smallest_positive_m_l789_789909


namespace calculate_two_neg_one_l789_789172

open Real

-- Define the condition of negative exponents
def neg_exponent (a : ℝ) (n : ℤ) : Prop :=
∀ (a : ℝ) (n : ℤ), a ≠ 0 → a^(-n) = 1 / a^n

-- The problem to be proved
theorem calculate_two_neg_one : 2^(-1 : ℤ) = 1 / 2 :=
by
  sorry

end calculate_two_neg_one_l789_789172


namespace find_angle_B_max_a2_c2_and_angles_l789_789357

-- Define angles and sides in triangle ABC
variables (A B C a b c : ℝ)
-- Summarize the conditions of the triangle
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
          (h4 : A + B + C = π) -- Sum of angles in a triangle is π
          (h5 : a / sin A = b / sin B) -- Law of sines part 1
          (h6 : b / sin B = c / sin C) -- Law of sines part 2

-- Special conditions given in the problem
variables (h7 : sin (A - B) + sin C = sqrt 2 * sin A) -- Given trigonometric relation

-- Firstly prove the value of angle B
theorem find_angle_B : B = π / 4 :=
by sorry

-- Then, if b = 2, find the maximum value of a^2 + c^2 and the corresponding angles A and C
theorem max_a2_c2_and_angles 
  (h8 : b = 2) : (∃ A C, A = 3 * π / 8 ∧ C = 3 * π / 8 ∧ a^2 + c^2 = 8 + 4 * sqrt 2) :=
by sorry

end find_angle_B_max_a2_c2_and_angles_l789_789357


namespace find_value_l789_789700

variable (a b : ℝ)

def quadratic_equation_roots : Prop :=
  a^2 - 4 * a - 1 = 0 ∧ b^2 - 4 * b - 1 = 0

def sum_of_roots : Prop :=
  a + b = 4

def product_of_roots : Prop :=
  a * b = -1

theorem find_value (ha : quadratic_equation_roots a b) (hs : sum_of_roots a b) (hp : product_of_roots a b) :
  2 * a^2 + 3 / b + 5 * b = 22 :=
sorry

end find_value_l789_789700


namespace parameter_range_l789_789331

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (1 - a) * x + 3 else log x - 2 * a

theorem parameter_range (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f(a, x) = y) ↔ -4 ≤ a ∧ a < 1 :=
by
  sorry

end parameter_range_l789_789331


namespace irrational_number_l789_789963

theorem irrational_number (a b c d : ℝ) (h₁ : 0.55555 = a) (h₂ : a ∈ ℚ)
  (h₃ : b = π / 2) (h₄ : c = 22 / 3) (h₅ : c ∈ ℚ) (h₆ : d = 3.121121121112) (h₇ : d ∈ ℚ) :
  irrational b :=
by sorry

end irrational_number_l789_789963


namespace lateral_surface_area_of_cone_l789_789316

theorem lateral_surface_area_of_cone :
  let r := 3 in
  let h := 4 in
  let slant_height := Float.sqrt (r^2 + h^2) in
  π * r * slant_height = 15 * π :=
by
  -- Proof will go here, but we use sorry to skip the proof
  sorry

end lateral_surface_area_of_cone_l789_789316


namespace greatest_integer_gcd_l789_789543

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l789_789543


namespace complex_modulus_square_l789_789414

open Complex

theorem complex_modulus_square (z : ℂ) (h : z^2 + abs z ^ 2 = 7 + 6 * I) : abs z ^ 2 = 85 / 14 :=
sorry

end complex_modulus_square_l789_789414


namespace basic_cable_cost_l789_789389

variable (B M S : ℝ)

def CostOfMovieChannels (B : ℝ) : ℝ := B + 12
def CostOfSportsChannels (M : ℝ) : ℝ := M - 3

theorem basic_cable_cost :
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  B + M + S = 36 → B = 5 :=
by
  intro h
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  sorry

end basic_cable_cost_l789_789389


namespace smallest_n_Sn_pos_l789_789699

theorem smallest_n_Sn_pos {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : ∀ n, (n ≠ 5 → S n > S 5))
  (h3 : |a 5| > |a 6|) :
  ∃ n : ℕ, S n > 0 ∧ ∀ m < n, S m ≤ 0 :=
by 
  -- Actual proof steps would go here.
  sorry

end smallest_n_Sn_pos_l789_789699


namespace largest_int_less_than_100_by_7_l789_789227

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789227


namespace days_to_learn_all_vowels_l789_789277

-- Defining the number of vowels
def number_of_vowels : Nat := 5

-- Defining the days Charles takes to learn one alphabet
def days_per_vowel : Nat := 7

-- Prove that Charles needs 35 days to learn all the vowels
theorem days_to_learn_all_vowels : number_of_vowels * days_per_vowel = 35 := by
  sorry

end days_to_learn_all_vowels_l789_789277


namespace phase_shift_of_cosine_transformation_l789_789268

theorem phase_shift_of_cosine_transformation :
  ∀ (A B C : ℝ), 
  (∀ x : ℝ, y = A * cos (B * x + C)) →
  A = 3 → B = 3 → C = -π / 4 →
  (∃ φ : ℝ, φ = -C / B ∧ φ = π / 12) :=
by
  intros A B C h y_eq_cos A_eq B_eq C_eq
  sorry

end phase_shift_of_cosine_transformation_l789_789268


namespace force_equilibrium_magnitude_l789_789944

noncomputable def vector_magnitude {α} [normed_field α] (v : EuclideanSpace α) : α := ∥v∥

noncomputable def force_3_magnitude (F1 F2 : EuclideanSpace ℝ) (angle : ℝ) : ℝ :=
vector_magnitude (-(F1 + F2))

theorem force_equilibrium_magnitude (F1 F2 : EuclideanSpace ℝ) (H_angle : angle F1 F2 = (2 * Real.pi) / 3)
  (H_F1_mag : vector_magnitude F1 = 6) (H_F2_mag : vector_magnitude F2 = 6) :
  force_3_magnitude F1 F2 (2 * Real.pi / 3) = 6 :=
by
  -- proofs to be filled in here
  sorry

end force_equilibrium_magnitude_l789_789944


namespace smaller_octagon_area_ratio_l789_789477

theorem smaller_octagon_area_ratio 
  (ABCDEFGH : Type) 
  [regular_octagon ABCDEFGH]
  (midpoints : ∀ (A B : ABCDEFGH), midpoint A B → smaller_octagon)
  : ∃ smaller_octagon,
    (area(smaller_octagon) / area(ABCDEFGH)) = 1/4 :=
sorry

end smaller_octagon_area_ratio_l789_789477


namespace sin_z_over_z_infinite_product_eq_l789_789838

noncomputable def sin_z_over_z_infinite_product (z : ℂ) : ℂ :=
∏ n in finset.range (n + 1), (1 + 2 * complex.cos (2 * z / 3 ^ n) / 3)

theorem sin_z_over_z_infinite_product_eq (z : ℂ) : 
  (complex.sin z / z) = (sin_z_over_z_infinite_product z) :=
sorry

end sin_z_over_z_infinite_product_eq_l789_789838


namespace tile_arrangement_solution_l789_789374

-- Define the dimensions of the ceiling
def Ceiling := {length : ℕ, width : ℕ}

-- Define a function to count the number of possible tile arrangements
noncomputable def tile_arrangements (ceiling : Ceiling) (tiles : ℕ) (tile_size : Ceiling) (beam_position : ℕ) : ℕ :=
  if ceiling = {length := 6, width := 4} ∧ tiles = 12 ∧ tile_size = {length := 1, width := 2} ∧ beam_position = 2 then
    180
  else
    0

-- Define the main theorem to be proven
theorem tile_arrangement_solution : tile_arrangements {length := 6, width := 4} 12 {length := 1, width := 2} 2 = 180 :=
  sorry

end tile_arrangement_solution_l789_789374


namespace exists_disc_with_few_intersections_l789_789394

variable {n : ℕ}
variable (D : Fin n → Set (ℝ × ℝ))
variable (max_discs : ℕ)
variable (condition : ∀ p : ℝ × ℝ, ∑ i, if p ∈ D i then 1 else 0 ≤ max_discs)

theorem exists_disc_with_few_intersections (H : max_discs = 2003) :
  ∃ k : Fin n, ∑ i, if ¬ Disjoint (D k) (D i) then 1 else 0 ≤ 7 * max_discs - 1 :=
by
  sorry

end exists_disc_with_few_intersections_l789_789394


namespace four_numbers_equal_differences_in_subset_l789_789723

theorem four_numbers_equal_differences_in_subset 
  (S : Set ℕ) (T : Set ℕ)
  (hS : S = { n | 1 ≤ n ∧ n ≤ 20 }) 
  (hT : T ⊆ S)
  (hcard_T : T.card = 10) :
  ∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ a ≠ b ∧ c ≠ d ∧ a - b = c - d :=
by
  sorry

end four_numbers_equal_differences_in_subset_l789_789723


namespace simplified_expression_l789_789036

-- Non-computable context since we are dealing with square roots and division
noncomputable def expr (x : ℝ) : ℝ := ((x / (x - 1)) - 1) / ((x^2 + 2 * x + 1) / (x^2 - 1))

theorem simplified_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : expr x = Real.sqrt 2 / 2 := by
  sorry

end simplified_expression_l789_789036


namespace greatest_integer_gcd_30_is_125_l789_789530

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l789_789530


namespace percentage_decrease_is_20_l789_789482

-- Define the original and new prices in Rs.
def original_price : ℕ := 775
def new_price : ℕ := 620

-- Define the decrease in price
def decrease_in_price : ℕ := original_price - new_price

-- Define the formula to calculate the percentage decrease
def percentage_decrease (orig_price new_price : ℕ) : ℕ :=
  (decrease_in_price * 100) / orig_price

-- Prove that the percentage decrease is 20%
theorem percentage_decrease_is_20 :
  percentage_decrease original_price new_price = 20 :=
by
  sorry

end percentage_decrease_is_20_l789_789482


namespace find_ff_1over16_l789_789709

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 4 else 3 ^ x

theorem find_ff_1over16 : f (f (1 / 16)) = 1 / 9 := by
  sorry

end find_ff_1over16_l789_789709


namespace smallest_possible_recording_l789_789932

theorem smallest_possible_recording :
  ∃ (A B C : ℤ), 
      (0 ≤ A ∧ A ≤ 10) ∧ 
      (0 ≤ B ∧ B ≤ 10) ∧ 
      (0 ≤ C ∧ C ≤ 10) ∧ 
      (A + B + C = 12) ∧ 
      (A + B + C) % 5 = 0 ∧ 
      A = 0 :=
by
  sorry

end smallest_possible_recording_l789_789932


namespace calculate_x_one_minus_f_l789_789413

noncomputable def x := (2 + Real.sqrt 3) ^ 500
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem calculate_x_one_minus_f : x * (1 - f) = 1 := by
  sorry

end calculate_x_one_minus_f_l789_789413


namespace smaller_number_is_eleven_l789_789025

theorem smaller_number_is_eleven (x : ℕ) (h₁ : 3 * x + 11 + x = 55) : x = 11 :=
begin
  sorry
end

end smaller_number_is_eleven_l789_789025


namespace circle_equation_l789_789869

theorem circle_equation (x y : ℝ) :
  let center := (0, 4)
  let point_on_circle := (3, 0)
  (x - center.1)^2 + (y - center.2)^2 = 25 :=
by
  sorry

end circle_equation_l789_789869


namespace problem1_problem2_l789_789317

variable {f : ℝ → ℝ} {g : ℝ → ℝ}
variable {A B : set ℝ}
variable {a b k : ℝ}

-- Condition for f
def domain_f :=
  ∀ x, x ∈ A ↔ x^2 + a * x + b > 0

-- Condition for g
def domain_g :=
  ∀ x, x ∈ B ↔ k * x^2 + 4 * x + k + 3 ≥ 0

-- First question equivalent in Lean
theorem problem1 (hB : B = set.univ) : k ∈ set.Ici 1 := sorry

-- Second question equivalent in Lean
theorem problem2 (h_inter : B ∩ (set.univ \ A) = B) (h_union : B ∪ (set.univ \ A) = {x | -2 ≤ x ∧ x ≤ 3}) :
  a = -1 ∧ b = -6 ∧ k ∈ set.Icc (-4) (-3 / 2) := sorry

end problem1_problem2_l789_789317


namespace assign_roles_l789_789150

def num_men : ℕ := 6
def num_women : ℕ := 7
def male_roles : ℕ := 2
def female_roles : ℕ := 2
def either_gender_roles : ℕ := 3

theorem assign_roles :
  let male_ways := num_men * (num_men - 1),
      female_ways := num_women * (num_women - 1),
      num_actors_remain := (num_men + num_women) - male_roles - female_roles,
      either_gender_ways := num_actors_remain * (num_actors_remain - 1) * (num_actors_remain - 2)
  in male_ways * female_ways * either_gender_ways = 635040 := by
  sorry

end assign_roles_l789_789150


namespace porter_monthly_earnings_l789_789827

-- Definitions
def regular_daily_rate : ℝ := 8
def days_per_week : ℕ := 5
def overtime_rate : ℝ := 1.5
def tax_deduction_rate : ℝ := 0.10
def insurance_deduction_rate : ℝ := 0.05
def weeks_per_month : ℕ := 4

-- Intermediate Calculations
def regular_weekly_earnings := regular_daily_rate * days_per_week
def extra_day_rate := regular_daily_rate * overtime_rate
def total_weekly_earnings := regular_weekly_earnings + extra_day_rate
def total_monthly_earnings_before_deductions := total_weekly_earnings * weeks_per_month

-- Deductions
def tax_deduction := total_monthly_earnings_before_deductions * tax_deduction_rate
def insurance_deduction := total_monthly_earnings_before_deductions * insurance_deduction_rate
def total_deductions := tax_deduction + insurance_deduction
def total_monthly_earnings_after_deductions := total_monthly_earnings_before_deductions - total_deductions

-- Theorem Statement
theorem porter_monthly_earnings : total_monthly_earnings_after_deductions = 176.80 := by
  sorry

end porter_monthly_earnings_l789_789827


namespace probability_correct_guess_l789_789431

def tens_digit_is_odd (n : ℕ) : Prop :=
  let d := n / 10 in d = 7 ∨ d = 9

def units_digit_is_prime (n : ℕ) : Prop :=
  let u := n % 10 in u = 2 ∨ u = 3 ∨ u = 5 ∨ u = 7

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n > 65 ∧ tens_digit_is_odd n ∧ units_digit_is_prime n

def count_valid_numbers : ℕ :=
  finset.card (finset.filter valid_number (finset.range 100))

theorem probability_correct_guess : count_valid_numbers = 8 →
  1 / count_valid_numbers = 1 / 8 :=
sorry

end probability_correct_guess_l789_789431


namespace determine_abs_d_l789_789850

noncomputable def z : ℂ := 3 + complex.i
def polynomial_eq_zero (a b c d : ℤ) : Prop := 
  a * z^6 + b * z^5 + c * z^4 + d * z^3 + c * z^2 + b * z + a = 0

theorem determine_abs_d (a b c d : ℤ) 
  (h_gcd : Int.gcd (Int.gcd a b) (Int.gcd c d) = 1) 
  (h_eq : polynomial_eq_zero a b c d) : 
  abs d = 540 :=
sorry

end determine_abs_d_l789_789850


namespace additional_days_to_finish_project_l789_789571

theorem additional_days_to_finish_project
  (total_hours : ℤ)
  (initial_workers : ℤ)
  (initial_days : ℤ)
  (initial_hours_per_day : ℤ)
  (new_workers : ℤ)
  (new_hours_per_day : ℤ)
  (increased_hours_per_day : ℤ)
  (first_phase_days : ℤ)
  (remaining_hours : ℤ)
  (daily_hours_after_new_workers : ℤ)
  (expected_days : ℤ) :
  initial_workers * initial_hours_per_day * initial_days = total_hours →
  initial_workers * initial_hours_per_day * first_phase_days = remaining_hours →
  initial_days - first_phase_days ≥ 0 →
  remaining_hours = total_hours - (initial_workers * initial_hours_per_day * first_phase_days) →
  daily_hours_after_new_workers = (initial_workers * increased_hours_per_day) + (new_workers * new_hours_per_day) →
  (remaining_hours / daily_hours_after_new_workers).ceil = expected_days →
  expected_days = 3 :=
by
  sorry

end additional_days_to_finish_project_l789_789571


namespace problem1a_problem1b_problem2_problem3_l789_789333

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem problem1a : f 2 + f (1/2) = 1 :=
sorry

theorem problem1b : f 3 + f (1/3) = 1 :=
sorry

theorem problem2 : ∀ x ≠ 0, f x + f (1/x) = 1 :=
sorry

theorem problem3 : (finset.range 2017).sum (λ i, f (i + 2) + f (1 / (i + 2))) = 2017 :=
sorry

end problem1a_problem1b_problem2_problem3_l789_789333


namespace cost_reduction_l789_789091

-- Define the conditions
def cost_two_years_ago : ℝ := 5000
def annual_decrease_rate : ℝ := x

-- Define the costs
def cost_last_year (cost_two_years_ago : ℝ) (annual_decrease_rate : ℝ) : ℝ :=
  cost_two_years_ago * (1 - annual_decrease_rate)

def cost_this_year (cost_last_year : ℝ) (annual_decrease_rate : ℝ) : ℝ :=
  cost_last_year * (1 - annual_decrease_rate)

-- Lean 4 statement of the problem
theorem cost_reduction (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1):
  let cost_two_years_ago := 5000 in
  let cost_last_year := cost_two_years_ago * (1 - x) in
  let cost_this_year := cost_last_year * (1 - x) in
  cost_last_year - cost_this_year = 5000x - 5000x^2 :=
by sorry

end cost_reduction_l789_789091


namespace length_CI_l789_789078

variables {AB AC BC M I E : ℝ}
variables (AIME_cyclic : cyclic quadrilateral A I M E)
variables (AI_eq_2AE : 2 * AE = AI)
variables (EMI_area : triangle_area E M I = 5)
variables (isosceles_right_triangle_ABC : isosceles_right_triangle AB AC 5)

theorem length_CI :
  ∃ a b c : ℕ, CI = \frac{5-10\sqrt{2}}{1} :=
sorry

end length_CI_l789_789078


namespace altitude_of_triangle_l789_789045

open EuclideanGeometry

variables {A B C L D E F : Point}

-- Definition of the triangle and angle bisector
variables (hABC : Triangle A B C)
variables (h_bisector : AngleBisector A C B L)

-- Definitions of the perpendicular points D and E
variables (hD : Perpendicular C D A L)
variables (hE : Perpendicular L E A B)

-- Definition of the intersection point F
variables (h_intersection : Intersects C B D E F)

-- The main theorem
theorem altitude_of_triangle
  (hABC : Triangle A B C)
  (h_bisector : AngleBisector A C B L)
  (hD : Perpendicular C D A L)
  (hE : Perpendicular L E A B)
  (h_intersection : Intersects C B D E F) :
  IsAltitude A F B C :=
sorry

end altitude_of_triangle_l789_789045


namespace period_of_tan_scaled_l789_789103

theorem period_of_tan_scaled (a : ℝ) (h : a ≠ 0) : 
  (∃ l : ℝ, l > 0 ∧ ∀ x : ℝ, tan(x / a) = tan((x + l) / a)) ↔ 
  a = 1/3 → (∃ l : ℝ, l = 3 * π) := 
sorry

end period_of_tan_scaled_l789_789103


namespace largest_integer_remainder_condition_l789_789258

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789258


namespace first_tap_fill_time_l789_789671

/-- Geetha has two taps with the following properties:
1. The second tap alone can fill the sink in 214 seconds.
2. Both taps together can fill the sink in approximately 105.99 seconds.
Prove that the first tap alone can fill the sink in approximately 209.94 seconds.
-/
theorem first_tap_fill_time 
  (tap2_fill_time : ℝ) (tap2_fill_time_value : tap2_fill_time = 214)
  (combined_fill_time : ℝ) (combined_fill_time_value : combined_fill_time ≈ 105.99) :
  ∃ tap1_fill_time : ℝ, tap1_fill_time ≈ 209.94 :=
sorry

end first_tap_fill_time_l789_789671


namespace hyperbola_eccentricity_l789_789297

variable {a b c : ℝ} (h_a : a > 0) (h_b : b > 0)
variable (C : Set (ℝ × ℝ)) (h_C : ∀ (x y : ℝ), (x, y) ∈ C ↔ (ℝ × ℝ) := {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
variable (A : ℝ × ℝ := (a, 0))
variable (F : ℝ × ℝ := (c, 0))
variable (B : ℝ × ℝ := (c, b^2 / a))
variable (h_slope : (b^2 / a - 0) / (c - a) = 3)

theorem hyperbola_eccentricity (h_b_square : b^2 = c^2 - a^2) : c / a = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789297


namespace intersection_A_B_l789_789417

-- Define the sets A and B based on the given conditions.
def A : Set ℝ := {x | ∃ y, y = Real.ln (1 - x)}
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Statement: Prove that A ∩ B = {x | 0 ≤ x ∧ x < 1}.
theorem intersection_A_B :
  A ∩ B = {x | 0 ≤ x ∧ x < 1} :=
sorry

end intersection_A_B_l789_789417


namespace prism_inequalities_l789_789680

variables {x y z : ℝ} (hxy : x < y) (hyz : y < z)

def p := 4 * (x + y + z)
def s := 2 * (x * y + y * z + z * x)
def d := Real.sqrt (x^2 + y^2 + z^2)

theorem prism_inequalities (hx_lt : x < y) (hy_gt : y < z) :
  x < (1 / 3) * (1 / 4 * p - Real.sqrt (d^2 - (1 / 2) * s)) ∧
  z > (1 / 3) * (1 / 4 * p + Real.sqrt (d^2 - (1 / 2) * s)) :=
  sorry

end prism_inequalities_l789_789680


namespace conditional_probability_rain_l789_789961

variables {Day : Type} (Weihai_rain Zibo_rain : Day → Prop)

axiom proportion_rain_Weihai : ∀ d, Prop → (P(Weihai_rain d) = 0.2)
axiom proportion_rain_Zibo : ∀ d, Prop → (P(Zibo_rain d) = 0.15)
axiom proportion_rain_both : ∀ d, Prop → (P(Weihai_rain d ∧ Zibo_rain d) = 0.06)

theorem conditional_probability_rain (d : Day) :
  P(Zibo_rain d | Weihai_rain d) = 0.3 :=
sorry

end conditional_probability_rain_l789_789961


namespace circumference_given_area_l789_789124

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def circumference_of_circle (r : ℝ) : ℝ := 2 * Real.pi * r

theorem circumference_given_area :
  (∃ r : ℝ, area_of_circle r = 616) →
  circumference_of_circle 14 = 2 * Real.pi * 14 :=
by
  sorry

end circumference_given_area_l789_789124


namespace min_rain_fourth_day_l789_789898

def rain_overflow_problem : Prop :=
    let holding_capacity := 6 * 12 -- in inches
    let drainage_per_day := 3 -- in inches
    let rainfall_day1 := 10 -- in inches
    let rainfall_day2 := 2 * rainfall_day1 -- 20 inches
    let rainfall_day3 := 1.5 * rainfall_day2 -- 30 inches
    let total_rain_three_days := rainfall_day1 + rainfall_day2 + rainfall_day3 -- 60 inches
    let total_drainage_three_days := 3 * drainage_per_day -- 9 inches
    let remaining_capacity := holding_capacity - (total_rain_three_days - total_drainage_three_days) -- 21 inches
    (remaining_capacity = 21)

theorem min_rain_fourth_day : rain_overflow_problem := sorry

end min_rain_fourth_day_l789_789898


namespace problem_statement_l789_789674

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end problem_statement_l789_789674


namespace length_of_second_sheet_l789_789458

theorem length_of_second_sheet 
  (A : ℝ) (L : ℝ) (Area1 : A = 198) 
  (Area2 : A = 22 * L + 100) : 
  L = 98 / 22 := 
by
  rw [Area1, Area2]
  ring
  linarith 

#eval length_of_second_sheet 198 (98 / 22) rfl (by linarith)

end length_of_second_sheet_l789_789458


namespace greatest_integer_gcd_l789_789544

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l789_789544


namespace unique_solution_for_a_eq_1_l789_789996

theorem unique_solution_for_a_eq_1 :
  ∃! x ∈ set.Ico 0 Real.pi, sin (2 * x) * sin (4 * x) - sin x * sin (3 * x) = 1 :=
sorry

end unique_solution_for_a_eq_1_l789_789996


namespace utility_bills_total_correct_l789_789018

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l789_789018


namespace original_avg_age_is_fifty_l789_789557

-- Definitions based on conditions
variable (N : ℕ) -- original number of students
variable (A : ℕ) -- original average age
variable (new_students : ℕ) -- number of new students
variable (new_avg_age : ℕ) -- average age of new students
variable (decreased_avg_age : ℕ) -- new average age after new students join

-- Conditions given in the problem
def original_avg_age_condition : Prop := A = 50
def new_students_condition : Prop := new_students = 12
def avg_age_new_students_condition : Prop := new_avg_age = 32
def decreased_avg_age_condition : Prop := decreased_avg_age = 46

-- Final Mathematical Equivalent Proof Problem
theorem original_avg_age_is_fifty
  (h1 : original_avg_age_condition A)
  (h2 : new_students_condition new_students)
  (h3 : avg_age_new_students_condition new_avg_age)
  (h4 : decreased_avg_age_condition decreased_avg_age) :
  A = 50 :=
by sorry

end original_avg_age_is_fifty_l789_789557


namespace rectangle_perimeter_l789_789832

noncomputable def perimeter_rectangle (x y : ℝ) : ℝ := 2 * (x + y)

theorem rectangle_perimeter
  (x y a b : ℝ)
  (H1 : x * y = 2006)
  (H2 : x + y = 2 * a)
  (H3 : x^2 + y^2 = 4 * (a^2 - b^2))
  (b_val : b = Real.sqrt 1003)
  (a_val : a = 2 * Real.sqrt 1003) :
  perimeter_rectangle x y = 8 * Real.sqrt 1003 := by
  sorry

end rectangle_perimeter_l789_789832


namespace tony_additional_degrees_l789_789077

-- Definitions for the conditions
def total_years : ℕ := 14
def science_degree_years : ℕ := 4
def physics_degree_years : ℕ := 2
def additional_degree_years : ℤ := total_years - (science_degree_years + physics_degree_years)
def each_additional_degree_years : ℕ := 4
def additional_degrees : ℤ := additional_degree_years / each_additional_degree_years

-- Theorem stating the problem and the answer
theorem tony_additional_degrees : additional_degrees = 2 :=
 by
     sorry

end tony_additional_degrees_l789_789077


namespace factor_polynomial_l789_789195

theorem factor_polynomial (x : ℝ) :
  3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) :=
by
  sorry

end factor_polynomial_l789_789195


namespace largest_int_less_than_100_by_7_l789_789226

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789226


namespace purely_imaginary_complex_number_l789_789355

theorem purely_imaginary_complex_number (m : ℝ) (z : ℂ) (h : z = (m-1) * complex.I + m^2 - 1) (hz : z.im = 0) : m = -1 :=
by
  sorry

end purely_imaginary_complex_number_l789_789355


namespace sphere_surface_area_l789_789893

theorem sphere_surface_area (a : ℝ) (l R : ℝ)
  (h₁ : 6 * l^2 = a)
  (h₂ : l * Real.sqrt 3 = 2 * R) :
  4 * Real.pi * R^2 = (Real.pi / 2) * a :=
sorry

end sphere_surface_area_l789_789893


namespace jack_cleaning_time_is_one_hour_l789_789788

def jackGrove : ℕ × ℕ := (4, 5)
def timeToCleanEachTree : ℕ := 6
def timeReductionFactor : ℕ := 2
def totalCleaningTimeWithHelpMin : ℕ :=
  (jackGrove.fst * jackGrove.snd) * (timeToCleanEachTree / timeReductionFactor)
def totalCleaningTimeWithHelpHours : ℕ :=
  totalCleaningTimeWithHelpMin / 60

theorem jack_cleaning_time_is_one_hour :
  totalCleaningTimeWithHelpHours = 1 := by
  sorry

end jack_cleaning_time_is_one_hour_l789_789788


namespace ratio_of_prices_l789_789555

noncomputable def initial_budget : ℝ := 60
noncomputable def remaining_money : ℝ := 6
noncomputable def price_new_frame : ℝ := initial_budget + 0.2 * initial_budget
noncomputable def price_smaller_frame : ℝ := initial_budget - remaining_money
noncomputable def ratio : ℝ := price_smaller_frame / price_new_frame

theorem ratio_of_prices : ratio = 3 / 4 := 
by 
  have budget_eq : initial_budget = 60 := rfl
  have remaining_eq : remaining_money = 6 := rfl
  have new_frame_eq : price_new_frame = 72 :=
    by 
      rw [initial_budget]
      norm_num
  have smaller_frame_eq : price_smaller_frame = 54 :=
    by 
      rw [initial_budget, remaining_money]
      norm_num
  have ratio_eq : ratio = 54 / 72 := 
    by 
      rw [price_smaller_frame, price_new_frame]
      norm_num
  rw [ratio_eq]
  norm_num
  sorry

end ratio_of_prices_l789_789555


namespace locus_of_Y_l789_789156

open EuclideanGeometry

axiom trapezoid (A B C D : Point) : 
  (Line A D) ∥ (Line B C)

axiom perpendicular_line (ell : Line) (A B C D : Point) : 
  is_perpendicular ell (Line A D) ∧ is_perpendicular ell (Line B C)

axiom point_moving_along_line (X : Point) (ell : Line) : 
  moves_along X ell

axiom perpendicular_from_points (A B C D X Y : Point) :
  is_perpendicular (Line A X) (Line B X) ∧ is_perpendicular (Line D X) (Line C X)

theorem locus_of_Y (A B C D X Y ell : Point) :
  trapezoid A B C D →
  perpendicular_line ell A B C D →
  point_moving_along_line X ell →
  perpendicular_from_points A B C D X Y →
  locus Y = symmetric_line (locus X) (bisector A D) :=
by
  sorry

end locus_of_Y_l789_789156


namespace reservoir_water_level_at_6_pm_l789_789023

/-
  Initial conditions:
  - initial_water_level: Water level at 8 a.m.
  - increase_rate: Rate of increase in water level from 8 a.m. to 12 p.m.
  - decrease_rate: Rate of decrease in water level from 12 p.m. to 6 p.m.
  - start_increase_time: Starting time of increase (in hours from 8 a.m.)
  - end_increase_time: Ending time of increase (in hours from 8 a.m.)
  - start_decrease_time: Starting time of decrease (in hours from 12 p.m.)
  - end_decrease_time: Ending time of decrease (in hours from 12 p.m.)
-/
def initial_water_level : ℝ := 45
def increase_rate : ℝ := 0.6
def decrease_rate : ℝ := 0.3
def start_increase_time : ℝ := 0 -- 8 a.m. in hours from 8 a.m.
def end_increase_time : ℝ := 4 -- 12 p.m. in hours from 8 a.m.
def start_decrease_time : ℝ := 0 -- 12 p.m. in hours from 12 p.m.
def end_decrease_time : ℝ := 6 -- 6 p.m. in hours from 12 p.m.

theorem reservoir_water_level_at_6_pm :
  initial_water_level
  + (end_increase_time - start_increase_time) * increase_rate
  - (end_decrease_time - start_decrease_time) * decrease_rate
  = 45.6 :=
by
  sorry

end reservoir_water_level_at_6_pm_l789_789023


namespace percentageDecreaseSecondMonthWas20_l789_789887

-- Define the initial price
def initialPrice : ℝ := 1000

-- Define the first month's decrease percentage
def firstMonthDecreasePercentage : ℝ := 10

-- Define the first month's decrease amount
def firstMonthDecreaseAmount : ℝ := initialPrice * (firstMonthDecreasePercentage / 100)

-- Define the price after the first month
def priceAfterFirstMonth : ℝ := initialPrice - firstMonthDecreaseAmount

-- Define the price after the second month
def priceAfterSecondMonth : ℝ := 720

-- Define the percentage decrease in the second month
def percentageDecreaseSecondMonth (initial: ℝ) (final: ℝ) : ℝ :=
  ((initial - final) / initial) * 100

-- Theorem to prove that the percentage decrease in the second month was 20%
theorem percentageDecreaseSecondMonthWas20:
  percentageDecreaseSecondMonth priceAfterFirstMonth priceAfterSecondMonth = 20 := by
  sorry

end percentageDecreaseSecondMonthWas20_l789_789887


namespace orthogonality_find_k_l789_789336

-- Definitions for the given parabola and line
def parabola (x y : ℝ) := y^2 = -x
def line (x y k : ℝ) := y = k * (x + 1)

-- Points of intersection A and B between the parabola and line
def points_of_intersection (x1 x2 y1 y2 k : ℝ) :=
  parabola x1 y1 ∧ line x1 y1 k ∧
  parabola x2 y2 ∧ line x2 y2 k

-- Theorem to prove OA ⊥ OB
theorem orthogonality (x1 x2 y1 y2 : ℝ) (h : x1 * x2 = 1 ∧ y1 * y2 = -1):
  (x1 * x2 + y1 * y2 = 0) :=
by sorry

-- Theorem to find the value of k given the area condition
theorem find_k (k : ℝ) (h : ∃ x1 x2 y1 y2, points_of_intersection x1 x2 y1 y2 k) 
  (area_condition : ℝ) (h_area : area_condition = sqrt 10):
  k = 1/6 ∨ k = -1/6 :=
by sorry

end orthogonality_find_k_l789_789336


namespace no_dead_end_path_l789_789481

theorem no_dead_end_path (R : ℝ) :
  ∀ (path : List (ℝ × ℝ × ℝ × ℝ)), 
  (∀ square, (square ∈ chessboard_pattern_2R) → 
    ∃ qtr_circle ∈ path, qtr_circle.contained_in(square) ∧ qtr_circle.r = R) →
  (∀ track, (track ∈ path.cons) → smooth_transition(path.head, path.tail)) →
  (start = path.head ∧ end = path.last ∧ smooth_transition(start, end)) →
  false := 

sorry

end no_dead_end_path_l789_789481


namespace kevin_correct_guesses_l789_789366

noncomputable def card_deck : Type := sorry -- Define a card deck type
noncomputable def suits : Type := sorry -- Define a suits type

axiom standard_deck : card_deck -- A standard 52 card deck
axiom suits_contain_13_cards_each : ∀ (s : suits), ∃ (c : card_deck), sorry -- Each suit contains 13 cards.

noncomputable def kevin : Type := sorry -- Kevin's guessing type

axiom guess_strategy : kevin → card_deck → suits -- Kevin's strategy

theorem kevin_correct_guesses (k : kevin) (d : card_deck) (g : kevin → card_deck → suits)
    (h1 : ∀ (s : suits), ∃ (c : d), sorry)
    (h2 : ∀ t : card_deck, g k d = t → ∃ s : suits, t ∈ s ∧ max_remaining s (d \ t)) :
  ∃ g_correct : nat, g_correct ≥ 13 :=
sorry

end kevin_correct_guesses_l789_789366


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789218

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789218


namespace barney_and_regular_dinosaurs_combined_weight_l789_789612

theorem barney_and_regular_dinosaurs_combined_weight :
  (∀ (barney five_dinos_weight) (reg_dino_weight : ℕ),
    (barney = five_dinos_weight + 1500) →
    (five_dinos_weight = 5 * reg_dino_weight) →
    (reg_dino_weight = 800) →
    barney + five_dinos_weight = 9500) :=
by 
  intros barney five_dinos_weight reg_dino_weight h1 h2 h3
  have h4 : five_dinos_weight = 5 * 800 := by rw [h3, mul_comm]
  rw h4 at h2
  have h5 : five_dinos_weight = 4000 := by linarith
  rw h5 at h1
  have h6 : barney = 4000 + 1500 := h1
  have h7 : barney = 5500 := by linarith
  rw [h5, h7]
  linarith

end barney_and_regular_dinosaurs_combined_weight_l789_789612


namespace first_player_wins_533_first_player_wins_1000_l789_789082

-- Define the rules and the winning condition for the game.
def loses_if_sum_is (target_sum : ℕ) (moves : List ℕ) : Bool :=
  ∃ n, n + 1 ≤ moves.length ∧ moves.take (n + 1) = moves.drop n ∧ (moves.take (n + 1)).sum = target_sum

-- Define a function that verifies if the first player wins for a given target sum
def first_player_wins_for_sum (target_sum : ℕ) : Prop :=
  ∀ (moves : List ℕ), (forall i, (moves.nth i).value = 1 ∨ (moves.nth i).value = 2) → 
  ¬ loses_if_sum_is target_sum moves

-- The theorem for the case where the sum target is 533
theorem first_player_wins_533 : first_player_wins_for_sum 533 := 
  by sorry

-- The theorem for the case where the sum target is 1000
theorem first_player_wins_1000 : first_player_wins_for_sum 1000 := 
  by sorry

end first_player_wins_533_first_player_wins_1000_l789_789082


namespace sum_of_integers_greater_than_1_and_less_than_10_l789_789111

theorem sum_of_integers_greater_than_1_and_less_than_10 : ∑ i in finset.Icc 2 9, i = 44 :=
by
  sorry

end sum_of_integers_greater_than_1_and_less_than_10_l789_789111


namespace construct_triangle_with_given_properties_l789_789179

variables (A O B C : Type) [EuclideanGeometry A] [EuclideanGeometry O] [EuclideanGeometry B] [EuclideanGeometry C]

def triangle_construction_exists (b c f_a AO : ℝ) : Prop :=
  ∃ (A B C : EuclideanGeometry), 
    (b - c = d) ∧
    (segment AO) ∧
    (angle_bisector A B C = f_a) ∧
    (inscribed_circle_center A B C = O) ∧
    (u < f_a < 2 * u) ∧
    (d ≤ (2 * u * v) / (u - v))

theorem construct_triangle_with_given_properties
(b c f_a AO : ℝ) (d u v : ℝ) (h1 : u < f_a < 2 * u) (h2 : d ≤ (2 * u * v) / (u - v)) :
  ∃ (A O B C : EuclideanGeometry), 
    triangle_construction_exists b c f_a AO :=
sorry

end construct_triangle_with_given_properties_l789_789179


namespace ellipse_ratio_df_ab_l789_789323

theorem ellipse_ratio_df_ab :
  ∃ (F : ℝ × ℝ) (A B D : ℝ × ℝ) (C : set (ℝ × ℝ)),
  (C = {p | (p.1^2 / 25) + (p.2^2 / 9) = 1}) ∧
  (∃ x1 x2 y1 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧
  line_through F A ∧ line_through F B ∧
  let M := midpoint A B in let x0 := M.1 in let y0 := M.2 in
  let l := line_through F (x1, y1) in 
  let l' := perp_bisector A B in 
  l' ∈ major_axis D ∧ 
  (DF = D.1 - F.1) ∧ (AB = dist A B) ∧
  (D.1 = (16 * x0 / 25)) ∧ (|DF / AB| = 2 / 5))

end ellipse_ratio_df_ab_l789_789323


namespace valid_sequences_l789_789391

-- Define the transformation function for a ten-digit number
noncomputable def transform (n : ℕ) : ℕ := sorry

-- Given sequences
def seq1 := 1101111111
def seq2 := 1201201020
def seq3 := 1021021020
def seq4 := 0112102011

-- The proof problem statement
theorem valid_sequences :
  (transform 1101111111 = seq1) ∧
  (transform 1021021020 = seq3) ∧
  (transform 0112102011 = seq4) :=
sorry

end valid_sequences_l789_789391


namespace period_of_tan_x_over_3_l789_789105

theorem period_of_tan_x_over_3 : ∃ T > 0, ∀ x, tan (x / 3) = tan ((x + T) / 3) :=
by
  use 3 * Real.pi
  sorry

end period_of_tan_x_over_3_l789_789105


namespace utility_bills_total_l789_789014

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l789_789014


namespace A_lt_B_l789_789346

variable (x y : ℝ)

def A (x y : ℝ) : ℝ := - y^2 + 4 * x - 3
def B (x y : ℝ) : ℝ := x^2 + 2 * x + 2 * y

theorem A_lt_B (x y : ℝ) : A x y < B x y := 
by
  sorry

end A_lt_B_l789_789346


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l789_789531

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l789_789531


namespace OHaraTriple_example_l789_789502

def OHaraTriple (a b x : ℕ) : Prop :=
  (Nat.sqrt a + Nat.sqrt b = x)

theorem OHaraTriple_example : OHaraTriple 49 64 15 :=
by
  sorry

end OHaraTriple_example_l789_789502


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789208

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789208


namespace area_ratio_of_midpoints_of_regular_octagon_l789_789476

-- Define the regular octagon and its properties

section
variables (A B C D E F G H A' B' C' D' E' F' G' H' : Type) [regOct_X : RegularOctagon A B C D E F G H]
[RegularOctagonMidpoints : RegularOctagonMidpoints A B C D E F G H A' B' C' D' E' F' G' H']

-- Statement: The area ratio between the smaller and the larger octagon is 1/4
theorem area_ratio_of_midpoints_of_regular_octagon
  (h_midpoints : MidpointsOfSidesOfRegularOctagon A B C D E F G H A' B' C' D' E' F' G' H') :
  AreaOfOctagon A' B' C' D' E' F' G' H' = (1/4 : ℚ) * AreaOfOctagon A B C D E F G H :=
  sorry
end

end area_ratio_of_midpoints_of_regular_octagon_l789_789476


namespace parallelogram_isosceles_angles_l789_789094

def angle_sum_isosceles_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = b ∨ b = c ∨ a = c)

theorem parallelogram_isosceles_angles :
  ∀ (A B C D P : Type) (AB BC CD DA BD : ℝ)
    (angle_DAB angle_BCD angle_ABC angle_CDA angle_ABP angle_BAP angle_PBD angle_BDP angle_CBD angle_BCD : ℝ),
  AB ≠ BC →
  angle_DAB = 72 →
  angle_BCD = 72 →
  angle_ABC = 108 →
  angle_CDA = 108 →
  angle_sum_isosceles_triangle angle_ABP angle_BAP 108 →
  angle_sum_isosceles_triangle 72 72 angle_BDP →
  angle_sum_isosceles_triangle 108 36 36 →
  ∃! (ABP BPD BCD : Type),
   (angle_ABP = 36 ∧ angle_BAP = 36 ∧ angle_PBA = 108) ∧
   (angle_PBD = 72 ∧ angle_PDB = 72 ∧ angle_BPD = 36) ∧
   (angle_CBD = 108 ∧ angle_BCD = 36 ∧ angle_BDC = 36) :=
sorry

end parallelogram_isosceles_angles_l789_789094


namespace area_ratio_of_midpoints_of_regular_octagon_l789_789475

-- Define the regular octagon and its properties

section
variables (A B C D E F G H A' B' C' D' E' F' G' H' : Type) [regOct_X : RegularOctagon A B C D E F G H]
[RegularOctagonMidpoints : RegularOctagonMidpoints A B C D E F G H A' B' C' D' E' F' G' H']

-- Statement: The area ratio between the smaller and the larger octagon is 1/4
theorem area_ratio_of_midpoints_of_regular_octagon
  (h_midpoints : MidpointsOfSidesOfRegularOctagon A B C D E F G H A' B' C' D' E' F' G' H') :
  AreaOfOctagon A' B' C' D' E' F' G' H' = (1/4 : ℚ) * AreaOfOctagon A B C D E F G H :=
  sorry
end

end area_ratio_of_midpoints_of_regular_octagon_l789_789475


namespace fraction_of_original_price_l789_789602

-- Define the given conditions.
variables (CP P SP F : ℝ)

-- Condition 1: The original price P and gain of 35%
def original_price := P = 1.35 * CP

-- Condition 2: Selling price SP with 10% loss
def selling_price := SP = 0.90 * CP

-- Definition of fraction F at which the article is sold
def fraction_sale := SP = F * P

-- Theorem statement: Prove F = 2/3
theorem fraction_of_original_price
  (h1 : original_price)
  (h2 : selling_price)
  (h3 : fraction_sale) : F = 2 / 3 :=
by {
  sorry
}

end fraction_of_original_price_l789_789602


namespace phase_shift_of_cosine_transformation_l789_789269

theorem phase_shift_of_cosine_transformation :
  ∀ (A B C : ℝ), 
  (∀ x : ℝ, y = A * cos (B * x + C)) →
  A = 3 → B = 3 → C = -π / 4 →
  (∃ φ : ℝ, φ = -C / B ∧ φ = π / 12) :=
by
  intros A B C h y_eq_cos A_eq B_eq C_eq
  sorry

end phase_shift_of_cosine_transformation_l789_789269


namespace intersection_is_correct_l789_789337

def setA := {x : ℝ | 3 * x - x^2 > 0}
def setB := {x : ℝ | x ≤ 1}

theorem intersection_is_correct : 
  setA ∩ setB = {x | 0 < x ∧ x ≤ 1} :=
sorry

end intersection_is_correct_l789_789337


namespace speed_of_other_train_l789_789084

theorem speed_of_other_train
  (v : ℝ) -- speed of the second train
  (t : ℝ := 2.5) -- time in hours
  (distance : ℝ := 285) -- total distance
  (speed_first_train : ℝ := 50) -- speed of the first train
  (h : speed_first_train * t + v * t = distance) :
  v = 64 :=
by
  -- The proof will be assumed
  sorry

end speed_of_other_train_l789_789084


namespace largest_integer_less_than_100_with_remainder_4_l789_789207

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789207


namespace people_who_came_to_game_l789_789879

def total_seats : Nat := 92
def people_with_banners : Nat := 38
def empty_seats : Nat := 45

theorem people_who_came_to_game : (total_seats - empty_seats = 47) :=
by 
  sorry

end people_who_came_to_game_l789_789879


namespace find_cost_price_l789_789126

variables (SP CP : ℝ)
variables (discount profit : ℝ)
variable (h1 : SP = 24000)
variable (h2 : discount = 0.10)
variable (h3 : profit = 0.08)

theorem find_cost_price 
  (h1 : SP = 24000)
  (h2 : discount = 0.10)
  (h3 : profit = 0.08)
  (h4 : SP * (1 - discount) = CP * (1 + profit)) :
  CP = 20000 := 
by
  sorry

end find_cost_price_l789_789126


namespace lloyd_regular_hourly_rate_l789_789819

-- Define the conditions as hypotheses.
variables (R : ℝ) -- Lloyd's regular hourly rate
variables (h1 : ∀ x, Lloyd hours = 7.5)
variables (h2 : ∀ x, excess_hours > 7.5 -> payment per hour = 2.5 * R)
variables (h3 : Lloyd hours on given day = 10.5)
variables (h4 : earnings on given day = 67.5)

-- State the proof problem.
theorem lloyd_regular_hourly_rate : R = 4.5 :=
by
  sorry

end lloyd_regular_hourly_rate_l789_789819


namespace greatest_integer_with_gcd_l789_789522

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l789_789522


namespace solve_for_C_l789_789983

-- Given constants and assumptions
def SumOfDigitsFirst (A B : ℕ) := 8 + 4 + A + 5 + 3 + B + 2 + 1
def SumOfDigitsSecond (A B C : ℕ) := 5 + 2 + 7 + A + B + 6 + 0 + C

theorem solve_for_C (A B C : ℕ) 
  (h1 : (SumOfDigitsFirst A B % 9) = 0)
  (h2 : (SumOfDigitsSecond A B C % 9) = 0) 
  : C = 3 :=
sorry

end solve_for_C_l789_789983


namespace sides_form_consecutive_integers_l789_789776

theorem sides_form_consecutive_integers (n : ℕ) (hn : n = 33) :
  ∃ (perm : Fin n → ℕ), 
    (∀ i : Fin n, 1 ≤ perm i ∧ perm i ≤ n) ∧ 
    (∀ i j : Fin n, perm i = perm j → i = j) ∧ 
    (∀ i : Fin n, 
      let j := if i.1 = n - 1 then ⟨0, Nat.zero_lt_succ _⟩ else ⟨i.1 + 1, Nat.lt_succ_self _⟩ in
      ∃ k : ℕ, 
        (∀ x, ⟨x, Nat.lt_succ_self x⟩ ∈ Finset.Ico 1 34 → 
          Finset.sum (Finset.range n (λ i, perm i)) (λ x, perm x + perm (if x = n-1 then 0 else x+1)) = k) ∧ 
        ∀ (a : ℕ), (a ∈ Finset.Ico 18 50) → (∃ b : ℕ, b ∈ Finset.Ico 1 34 ∧ 
          (perm b + perm (if b = n - 1 then 0 else b + 1)) = a)) :=
sorry

end sides_form_consecutive_integers_l789_789776


namespace pacific_ocean_area_rounded_l789_789054

def pacific_ocean_area : ℕ := 19996800

def ten_thousand : ℕ := 10000

noncomputable def pacific_ocean_area_in_ten_thousands (area : ℕ) : ℕ :=
  (area / ten_thousand + if (area % ten_thousand) >= (ten_thousand / 2) then 1 else 0)

theorem pacific_ocean_area_rounded :
  pacific_ocean_area_in_ten_thousands pacific_ocean_area = 2000 :=
by
  sorry

end pacific_ocean_area_rounded_l789_789054


namespace minimize_surface_area_l789_789424

noncomputable theory

def volume := 32 -- volume in m^3
def height := 2 -- height in m
def surface_area (x : ℝ) : ℝ := 4 * x + (64 / x) + 32 -- surface area as a function of x

def min_paper_usage (x : ℝ) : Prop :=
  surface_area x = 64 

theorem minimize_surface_area (x : ℝ) (hx_pos : 0 < x) (hvol : volume = 2 * x * (16 / x)) :
  min_paper_usage 4 :=
sorry -- Proof goes here

end minimize_surface_area_l789_789424


namespace binom_13_8_l789_789310

theorem binom_13_8 :
  (nat.choose 14 7 = 3432) →
  (nat.choose 14 8 = 3003) →
  (nat.choose 12 7 = 792) →
  (nat.choose 13 8 = 1287) :=
begin
  intros h1 h2 h3,
  have h4: nat.choose 12 6 = 924,
    -- calculations skipped
    sorry,
  have h5: nat.choose 13 7 = nat.choose 12 7 + nat.choose 12 6,
    from nat.choose_succ_succ 12 7,
  rw [h3, h4] at h5,
  have h6: nat.choose 13 7 = 1716,
    from h5,
  have h7: nat.choose 13 8 = nat.choose 14 8 - nat.choose 13 7,
    from nat.choose_succ_succ 13 8,
  rw [h2, h6] at h7,
  exact h7,
  sorry
end

end binom_13_8_l789_789310


namespace largest_integer_less_than_100_with_remainder_4_l789_789202

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789202


namespace probability_approximation_l789_789507

theorem probability_approximation (m n : ℝ) (h1 : m = random_simulation_estimate) (h2 : n = actual_probability) : m ≈ n := by
  sorry

end probability_approximation_l789_789507


namespace range_of_values_for_a_l789_789566

def is_valid_triangle (a : ℝ) : Prop :=
  a > 1 ∧ a^2 + a^2 + 2*a + 1 - (a^2 + 4*a + 4) ≥ -a*(a + 1)

theorem range_of_values_for_a (a : ℝ) [IsValidTriangle a] : 
  3 / 2 ≤ a ∧ a < 3 :=
sorry

end range_of_values_for_a_l789_789566


namespace largest_last_digit_is_nine_l789_789872

-- Definition of the string of 2500 digits
def is_valid_sequence (s : List Nat) : Prop :=
  s.length = 2500 ∧
  s.head = some 2 ∧
  (∀ i : ℕ, i < 2499 → (23 ∣ (10 * s[i] + s[i + 1]) ∨ 29 ∣ (10 * s[i] + s[i + 1])))

-- The main theorem stating the largest last digit is 9
theorem largest_last_digit_is_nine (s : List Nat) (h_valid : is_valid_sequence s) : 
  s.last = some 9 :=
sorry

end largest_last_digit_is_nine_l789_789872


namespace expr_equals_expected_l789_789975

-- Define the expressions
def expr1 : ℝ := (Real.sqrt 48 - Real.sqrt 27) / Real.sqrt 3 + Real.sqrt 6 * (2 * Real.sqrt (1 / 3))
def expected : ℝ := 1 + 2 * Real.sqrt 2

-- Statement that the given expression equals the expected value
theorem expr_equals_expected : expr1 = expected :=
by
  sorry

end expr_equals_expected_l789_789975


namespace root_difference_eqn_l789_789648

theorem root_difference_eqn (q : ℝ) :
  let a := 1
      b := -q
      c := (q^2 - 4*q + 3) / 4
  in (-b + (b^2 - 4*a*c).sqrt) / (2*a) - (-b - (b^2 - 4*a*c).sqrt) / (2*a) = 2 * (q - 3/4).sqrt := 
by 
  sorry

end root_difference_eqn_l789_789648


namespace correct_statements_are_1_and_3_l789_789117

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ := (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem correct_statements_are_1_and_3
  (X : ℕ → ℚ) (Y : ℚ → ℚ) (σ : ℚ) 
  (h1 : X = λ n, binomial_probability 6 3 (1/2))
  (h2 : Y = λ n, 1 / 2 * (1 + math.erf ((n - 2) / (σ * sqrt 2))))
  (h2_condition : ∀ n > 0, P(Y < 4) = 0.9)
  (h3_A : ℚ)
  (h3_B : ℚ)
  (h3_AB : ℚ)
  (h3 : P(A|B) = h3_AB / h3_B = 2/9)
  (h4 : E(3*X + 1) = 3*E(X) + 1 ∧ Var(3*X + 1) ≠ 9*Var(X) + 1): 
  true :=
by
  sorry

end correct_statements_are_1_and_3_l789_789117


namespace simplified_expression_l789_789551

theorem simplified_expression (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 0) (h3 : y ≠ 0) : 
  (x - y)^{-2} * (x^{-1} - y^{-1}) = -1 / (x * y * (x - y)) :=
by
  sorry

end simplified_expression_l789_789551


namespace maximize_S2_plus_T2_l789_789771

open Real

def max_S2_plus_T2 (a h1 h2 : ℝ) : ℝ :=
  (a^2 / 4) * (h1^2 + h2^2)

theorem maximize_S2_plus_T2 
  (AB CD AD DC CB : ℝ)
  (h1 h2 : ℝ)
  (S T : ℝ)
  (hS : S = (1 / 2) * AB * h1)
  (hT : T = (1 / 2) * CD * h2)
  (hAD_EQ_1 : AD = 1)
  (hDC_EQ_1 : DC = 1)
  (hCB_EQ_1 : CB = 1)
  (hAB_EQ_CD : AB = CD) :
  max_S2_plus_T2 AB h1 h2 = 1 / 2 :=
sorry

end maximize_S2_plus_T2_l789_789771


namespace smallest_number_of_cookies_l789_789575

theorem smallest_number_of_cookies
  (n : ℕ) 
  (hn : 4 * n - 4 = (n^2) / 2) : n = 7 → n^2 = 49 := 
by
  sorry

end smallest_number_of_cookies_l789_789575


namespace analytic_expression_and_decreasing_interval_area_of_circumscribed_circle_l789_789342

variables (x θ : ℝ)

-- Definition of vectors a and b
def a : ℝ × ℝ := (sqrt 2 * sin (2 * x), sqrt 2 * cos (2 * x))
def b : ℝ × ℝ := (cos θ, sin θ)

-- Definition of f(x)
def f (x : ℝ) : ℝ := (sqrt 2 * sin (2 * x)) * cos θ + (sqrt 2 * cos (2 * x)) * sin θ

-- Question (I)
theorem analytic_expression_and_decreasing_interval (hx : f x = sqrt 2 * sin (2 * x + θ)) (h_symm : 2 * (π / 6) + θ = (k : ℤ) * π + π / 2) (θ_bound : |θ| < π / 2) :
  f x = sqrt 2 * sin (2 * x + π / 6) ∧
  ∀ k : ℤ, ∃ I : set ℝ, I = set.Ioc (k * π + π / 6) (k * π + 2 * π / 3) ∧ 
    ∀ x ∈ I, ∃ y ∈ I, f(x) > f(y) := sorry

-- Definitions related to triangle ABC
variables (A b c R : ℝ)

-- Given conditions
def triangle_condition : Prop := b = 5 ∧ c = 2 * sqrt 3 ∧ f A = sqrt 2

theorem area_of_circumscribed_circle (hA : A = π / 6) (ha : a^2 = b^2 + c^2 - 2 * b * c * cos A) :
  a = sqrt (25 + 12 - 2 * 5 * 2 * sqrt 3 * (cos (π / 6))) ∧ 
  R = sqrt 7 ∧
  S = 7 * π := sorry

end analytic_expression_and_decreasing_interval_area_of_circumscribed_circle_l789_789342


namespace Borya_Grisha_not_same_color_pants_l789_789163

theorem Borya_Grisha_not_same_color_pants:
  (∀ pants: {gray | brown | raspberry},
    ∃ winners: {Anton | Borya | Vova | Grisha | Dima},
      first_pants: winners.first_place = gray ∧
      second_pants: winners.second_place = brown ∧
      third_pants: winners.third_place = raspberry) →
  (∀ p: {gray | brown | raspberry}, (Anton wears p → Anton ≠ first_place) ∧
                                   (Dima wears p → Dima ≠ second_place) ∧
                                   (Vova wears p → Vova ≠ third_place)) →
  ¬ (Borya.wears p ∧ Grisha.wears p) :=
begin
  sorry
end

end Borya_Grisha_not_same_color_pants_l789_789163


namespace compounded_interest_is_615_l789_789737

-- Define the conditions as constants and assumptions
variables (P : ℝ) (y : ℝ) (T : ℝ)
variables (SI : ℝ) (CI : ℝ)

-- Set the known conditions
def principal := (6000.0 : ℝ)
def time := (2.0 : ℝ)
def simple_interest := (600.0 : ℝ)

-- Given the rate (y) derived from simple interest formula
def rate := (100.0 * simple_interest) / (principal * time)

-- Formula for compound interest for two years
def compound_interest := principal * (1 + (rate / 100.0)) ^ time - principal

-- Statement to prove
theorem compounded_interest_is_615 :
  compound_interest = 615.0 :=
by
  -- Skip the proof steps for now
  sorry

end compounded_interest_is_615_l789_789737


namespace complex_multiplication_l789_789169

theorem complex_multiplication : ∀ (i : ℂ), i^2 = -1 → i * (2 + 3 * i) = (-3 : ℂ) + 2 * i :=
by
  intros i hi
  sorry

end complex_multiplication_l789_789169


namespace sum_of_values_of_z_sum_of_roots_l789_789401

def f : ℚ → ℚ := λ x, x^2 + x + 1

theorem sum_of_values_of_z (z : ℚ) :
  (f (3 * z) = 7) → z = -1/3 ∨ z = 2/9 :=
begin
  sorry
end

theorem sum_of_roots :
  (∑ z in ({-1/3, 2/9} : finset ℚ), z) = -1/9 :=
begin
  sorry
end

end sum_of_values_of_z_sum_of_roots_l789_789401


namespace students_count_l789_789790

theorem students_count (S : ℕ) (num_adults : ℕ) (cost_student cost_adult total_cost : ℕ)
  (h1 : num_adults = 4)
  (h2 : cost_student = 5)
  (h3 : cost_adult = 6)
  (h4 : total_cost = 199) :
  5 * S + 4 * 6 = 199 → S = 35 := by
  sorry

end students_count_l789_789790


namespace greatest_int_with_gcd_five_l789_789537

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l789_789537


namespace water_filter_capacity_l789_789468

theorem water_filter_capacity (x : ℝ) (h : 0.30 * x = 36) : x = 120 :=
sorry

end water_filter_capacity_l789_789468


namespace convex_polygon_diagonals_25_convex_polygon_triangles_25_l789_789929

-- Define a convex polygon with 25 sides
def convex_polygon_sides : ℕ := 25

-- Define the number of diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Define the number of triangles that can be formed by choosing any three vertices from n vertices
def number_of_triangles (n : ℕ) : ℕ := n.choose 3

-- Theorem to prove the number of diagonals is 275 for a convex polygon with 25 sides
theorem convex_polygon_diagonals_25 : number_of_diagonals convex_polygon_sides = 275 :=
by sorry

-- Theorem to prove the number of triangles is 2300 for a convex polygon with 25 sides
theorem convex_polygon_triangles_25 : number_of_triangles convex_polygon_sides = 2300 :=
by sorry

end convex_polygon_diagonals_25_convex_polygon_triangles_25_l789_789929


namespace f_g_minus_g_f_eq_five_halves_l789_789405

def f (x : ℝ) : ℝ := 5 * x - 3
def g (x : ℝ) : ℝ := x / 2 + 1

theorem f_g_minus_g_f_eq_five_halves (x : ℝ) : f(g(x)) - g(f(x)) = 5 / 2 := by
  calc
    f(g(x)) - g(f(x)) = sorry

end f_g_minus_g_f_eq_five_halves_l789_789405


namespace find_largest_integer_l789_789251

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789251


namespace rabbit_walked_distance_l789_789153

theorem rabbit_walked_distance
  (side_length : ℕ)
  (h : side_length = 13) :
  let perimeter := 4 * side_length in
  perimeter = 52 :=
by
  -- Use the given condition to initialize the side_length to 13.
  have h1 : side_length = 13 := h
  -- Define perimeter according to the given side_length.
  let perimeter := 4 * side_length
  -- Show that the calculated perimeter is equal to 52.
  have h2 : perimeter = 4 * 13 := by rw [h1]
  have h3 : 4 * 13 = 52 := by decide
  rw [←h2, h3]
  sorry

end rabbit_walked_distance_l789_789153


namespace greatest_integer_gcd_l789_789548

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l789_789548


namespace sequence_formula_l789_789381

theorem sequence_formula (a : ℕ → ℕ) (c : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = a n + c * n) 
(h₃ : a 1 ≠ a 2) (h₄ : a 2 * a 2 = a 1 * a 3) : c = 2 ∧ ∀ n, a n = n^2 - n + 2 :=
by
  sorry

end sequence_formula_l789_789381


namespace triangle_perimeter_ellipse_l789_789706

theorem triangle_perimeter_ellipse 
  (a b : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 → 2 * sqrt 3) 
  (is_focus : (a = 0) ∨ (a = focus_side)) :
  ∃ B C : (ℝ × ℝ), 
  (B.1^2 / 4 + B.2^2 / 3 = 1) ∧ (C.1^2 / 4 + C.2^2 / 3 = 1) ∧
  (A = (c, 0)) ∧  (focus_side ∈ segment (B, C)) → 
  perimeter ⟨A, B, C⟩ = 8 := 
sorry

end triangle_perimeter_ellipse_l789_789706


namespace problem1_eq_problem2_eq_l789_789565

-- Problem 1 in Lean 4
theorem problem1_eq : 
  sqrt (8 : ℝ) + real.cbrt (8 : ℝ) - |sqrt (3 : ℝ) - 2| + sqrt (1 : ℝ) = 2 * sqrt 2 + 5 - sqrt 3 :=
by 
  sorry

-- Problem 2 in Lean 4
theorem problem2_eq (x y : ℝ) (h₁ : (x - 1) / 2 - (y + 2) / 4 = 1) 
  (h₂ : -x + y = -3) : x = 5 ∧ y = 2 :=
by 
  sorry

end problem1_eq_problem2_eq_l789_789565


namespace circles_externally_tangent_l789_789885

-- Defining the first circle equation
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Defining the second circle equation
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*y + 5 = 0

-- The main theorem that we want to prove
theorem circles_externally_tangent :
  ∀ (x y : ℝ), 
  (circle1 x y) ∧ (circle2 x y) → 
  externally_tangent circle1 circle2 :=
begin
  sorry -- Proof goes here
end

end circles_externally_tangent_l789_789885


namespace smallest_possible_N_l789_789597

open Nat

def has_common_digit (a b : ℕ) : Prop :=
  ∃ d : ℕ, (digit_in d a) ∧ (digit_in d b)

def digit_in (d n : ℕ) : Prop :=
  ∃ k : ℕ, k < 10 ∧ n % 10^(k + 1) / 10^k = d

noncomputable def smallest_N : ℕ :=
  29

theorem smallest_possible_N (N : ℕ) (hN1 : 2 ≤ N) : 
  (∀ n1 n2 : ℕ, 1 ≤ n1 ∧ n1 ≤ N ∧ 1 ≤ n2 ∧ n2 ≤ N ∧ n1 ≠ n2 → has_common_digit n1 n2) →
  N ≥ smallest_N := 
sorry

end smallest_possible_N_l789_789597


namespace expected_moves_to_erase_numbers_l789_789668

theorem expected_moves_to_erase_numbers :
  ∑ n in Finset.range 20, (1 : ℚ) / ⌊20 / (n + 1)⌋₊ = 131 / 10 :=
by sorry

end expected_moves_to_erase_numbers_l789_789668


namespace younger_brother_height_is_109_l789_789495

variable (younger_brother_height minkyung_height father_height : ℕ)

-- Conditions:
axiom (h1 : younger_brother_height = minkyung_height - 28)
axiom (h2 : father_height = minkyung_height + 35)
axiom (h3 : father_height = 172)

-- Statement to prove:
theorem younger_brother_height_is_109 : younger_brother_height = 109 :=
by
  sorry

end younger_brother_height_is_109_l789_789495


namespace correct_statement_l789_789914

theorem correct_statement :
  ∀ (n : ℕ), n ≥ 3 →
  (StatementA : (∀ (data : List ℝ), 
    var data > 0 → 
    mean data ≈ 0 → 
    ¬ (var data < 1 → data_fluctuation data > 1))) →
  (StatementB : ¬ (∀ (p : ℝ × ℝ × ℝ), 
    let eq_tri := equilateral_triangle p in 
    central_symmetry eq_tri)) →
  (StatementC : ¬ (∀ (q : ℝ × ℝ × ℝ × ℝ), 
    let quadrilateral := quadrilateral_with_equal_diagonals q in 
    rectangle quadrilateral)) →
  (StatementD : ∀ (poly : regular_polygon n), 
    sum_of_exterior_angles poly = 360) :=
by
  intros n hn StatementA StatementB StatementC poly
  sorry

end correct_statement_l789_789914


namespace number_of_coprimes_to_15_l789_789630

open Nat

theorem number_of_coprimes_to_15 : (Finset.filter (λ a, gcd a 15 = 1) (Finset.range 15)).card = 8 := by
  sorry

end number_of_coprimes_to_15_l789_789630


namespace Haley_first_album_pictures_l789_789730

theorem Haley_first_album_pictures :
  ∀ (total_pictures : ℕ) (albums : ℕ) (pictures_per_album : ℕ),
  total_pictures = 65 →
  albums = 6 →
  pictures_per_album = 8 →
  ∃ (first_album_pictures : ℕ),
  first_album_pictures = total_pictures - albums * pictures_per_album ∧ 
  first_album_pictures = 17 :=
by
  intros total_pictures albums pictures_per_album
  assume h1 h2 h3
  use total_pictures - albums * pictures_per_album
  split
  { rw [h1, h2, h3], refl }
  sorry

end Haley_first_album_pictures_l789_789730


namespace sin_30_plus_cos_45_sin2_60_plus_cos2_60_minus_tan_45_l789_789569

theorem sin_30_plus_cos_45 :
  Real.sin (π / 6) + Real.cos (π / √2) = (1 + Real.sqrt 2) / 2 :=
sorry

theorem sin2_60_plus_cos2_60_minus_tan_45 :
  Real.sin (π / 3)^2 + Real.cos (π / 3)^2 - Real.tan (π / 4) = 0 :=
sorry

end sin_30_plus_cos_45_sin2_60_plus_cos2_60_minus_tan_45_l789_789569


namespace pet_store_monday_dogs_l789_789148

/-- Problem Statement:
A pet store had 2 dogs. On Sunday they got 5 more dogs, and on Monday they got some more.
The pet store now has 10 dogs. How many dogs did the pet store get on Monday?
-/

theorem pet_store_monday_dogs 
  (initial_dogs : ℕ := 2)
  (sunday_dogs : ℕ := 5)
  (total_dogs : ℕ := 10) :
  ∃ monday_dogs : ℕ, monday_dogs = 3 := 
begin
  use (total_dogs - (initial_dogs + sunday_dogs)),
  exact nat.sub_eq_iff_eq_add.mpr rfl,
end

end pet_store_monday_dogs_l789_789148


namespace complex_traces_ellipse_l789_789859

theorem complex_traces_ellipse (z : ℂ) (h : ∥z∥ = 3) : ∃ a b : ℝ, 
  (z + z⁻¹ + 2) = a + b * complex.I ∧ 
  ((a - 2)^2 / (7.29) + b^2 / (7.29) = 1) := 
sorry

end complex_traces_ellipse_l789_789859


namespace intersection_of_A_and_B_l789_789697

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := {x | x^2 - 2 * x < 0}
def setB : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem to prove the intersection A ∩ B
theorem intersection_of_A_and_B : ((setA ∩ setB) = {x : ℝ | 0 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_A_and_B_l789_789697


namespace interval_length_sum_l789_789182
-- Importing necessary library

-- Defining the conditions
variables {a b : ℝ}
variables (h : a > b)

-- Stating the theorem
theorem interval_length_sum (a b : ℝ) (h : a > b) :
  let intervals_sum := (λ x : ℝ, (1 / (x - a) + 1 / (x - b))) in
  ∃ x₁ x₂ : ℝ, intervals_sum x₁ + intervals_sum x₂ = 2 :=
sorry

end interval_length_sum_l789_789182


namespace cars_in_garage_l789_789497

theorem cars_in_garage (bicycles : ℕ) (wheels : ℕ) (bicycle_wheels : bicycles = 9) (total_wheels : wheels = 82) : 
    ∃ c : ℕ, 4 * c + 2 * bicycles = wheels ∧ c = 16 :=
begin
  sorry
end

end cars_in_garage_l789_789497


namespace candy_cools_at_7_degrees_per_minute_l789_789162

def candy_cooling_rate
  (initial_temp final_heating_temp final_cooling_temp heating_rate total_time: ℝ)
  (h1 : initial_temp = 60)
  (h2 : final_heating_temp = 240)
  (h3 : final_cooling_temp = 170)
  (h4 : heating_rate = 5)
  (h5 : total_time = 46) : ℝ :=
  (final_heating_temp - final_cooling_temp) / ((total_time - (final_heating_temp - initial_temp) / heating_rate))

theorem candy_cools_at_7_degrees_per_minute 
  (initial_temp final_heating_temp final_cooling_temp heating_rate total_time cooling_rate: ℝ)
  (h1 : initial_temp = 60)
  (h2 : final_heating_temp = 240)
  (h3 : final_cooling_temp = 170)
  (h4 : heating_rate = 5)
  (h5 : total_time = 46)
  (h6 : cooling_rate = candy_cooling_rate initial_temp final_heating_temp final_cooling_temp heating_rate total_time h1 h2 h3 h4 h5) :
  cooling_rate = 7 := sorry

end candy_cools_at_7_degrees_per_minute_l789_789162


namespace weight_of_NH4I_H2O_l789_789621

noncomputable def total_weight (moles_NH4I : ℕ) (molar_mass_NH4I : ℝ) 
                             (moles_H2O : ℕ) (molar_mass_H2O : ℝ) : ℝ :=
  (moles_NH4I * molar_mass_NH4I) + (moles_H2O * molar_mass_H2O)

theorem weight_of_NH4I_H2O :
  total_weight 15 144.95 7 18.02 = 2300.39 :=
by
  sorry

end weight_of_NH4I_H2O_l789_789621


namespace expenditure_ratio_l789_789064

def ratio_of_incomes (I1 I2 : ℕ) : Prop := I1 / I2 = 5 / 4
def savings (I E : ℕ) : ℕ := I - E
def ratio_of_expenditures (E1 E2 : ℕ) : Prop := E1 / E2 = 3 / 2

theorem expenditure_ratio (I1 I2 E1 E2 : ℕ) 
  (I1_income : I1 = 5500)
  (income_ratio : ratio_of_incomes I1 I2)
  (savings_equal : savings I1 E1 = 2200 ∧ savings I2 E2 = 2200)
  : ratio_of_expenditures E1 E2 :=
by 
  sorry

end expenditure_ratio_l789_789064


namespace find_a_plus_b_l789_789330

noncomputable def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1 / 3
noncomputable def f_prime (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a^2

theorem find_a_plus_b 
  (a b : ℝ)
  (h_deriv : f_prime a b 1 = 0)
  (h_extreme : f a b 1 = 0) :
  a + b = -7 / 9 := 
sorry

end find_a_plus_b_l789_789330


namespace setC_is_right_triangle_l789_789116

-- Definitions based on the problem conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def setA : Prop := is_right_triangle 1 2 2
def setB : Prop := is_right_triangle 1 1 (Real.sqrt 3)
def setC : Prop := is_right_triangle 3 4 5
def setD : Prop := is_right_triangle 4 5 6

-- Prove that only set C satisfies the Pythagorean theorem
theorem setC_is_right_triangle :
  ¬setA ∧ ¬setB ∧ setC ∧ ¬setD :=
by
  sorry

end setC_is_right_triangle_l789_789116


namespace matrix_multiplication_correct_l789_789175

open Matrix

def A : Matrix (Fin 4) (Fin 3) ℤ :=
  ![
  ![2, 0, -1],
  ![1, 3, -2],
  ![0, 5, 4],
  ![-2, 2, 3]
  ]

def B : Matrix (Fin 3) (Fin 4) ℤ :=
  ![
  ![3, -1, 0, 4],
  ![2, 0, -3, -1],
  ![1, 0, 0, 2]
  ]

def C : Matrix (Fin 4) (Fin 4) ℤ :=
  ![
  ![5, -2, 0, 6],
  ![7, -1, -9, -3],
  ![14, 0, -15, 3],
  ![1, 2, -6, -4]
  ]

theorem matrix_multiplication_correct : A ⬝ B = C :=
by
  sorry

end matrix_multiplication_correct_l789_789175


namespace math_problem_l789_789619

theorem math_problem :
  (Real.pi - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 :=
by
  sorry

end math_problem_l789_789619


namespace estimated_probability_is_correct_l789_789891

-- Define the conditions
def num_throws : list ℕ := [40, 120, 320, 480, 720, 800, 920, 1000]
def frequencies : list ℕ := [20, 50, 146, 219, 328, 366, 421, 463]
def probabilities : list ℝ := [0.500, 0.417, 0.456, 0.456, 0.456, 0.458, 0.458, 0.463]

-- Define the proof statement
theorem estimated_probability_is_correct : 
  (∀ n ∈ num_throws, ∃ f ∈ frequencies, ∃ p ∈ probabilities, p = (f.to_real / n.to_real)) → (∃ approx_prob, approx_prob = 0.46) :=
sorry

end estimated_probability_is_correct_l789_789891


namespace Z_is_normal_dist_XZ_dist_YZ_dist_X_plus_Z_cov_X_Z_l789_789808

-- Definitions of the given conditions
def X : Distribution := Normal 0 1
def Y : Distribution := uniform [{-1, 1}]
def Z (X Y : Distribution) := X * Y

-- Proof that Z follows a normal distribution
theorem Z_is_normal : Law (Z X Y) = Normal 0 1 := sorry

-- Distribution of the vector (X, Z)
theorem dist_XZ (x z : ℝ) : P(X ≤ x ∧ Z ≤ z) = 
  (Φ (min x z) / 2) + ((Φ x - Φ (-z)) / 2) * (if x ≥ -z then 1 else 0) := sorry

-- Distribution of the vector (Y, Z)
theorem dist_YZ : ∀ y ∈ {-1, 1}, Law (Z X (Y = y)) = Normal 0 1 := sorry

-- Distribution of the random variable X + Z
theorem dist_X_plus_Z (x : ℝ) : P(X + Z ≤ x) = (Φ (x / 2) + if x ≥ 0 then 1 else 0) / 2 := sorry

-- Proving that X and Z are uncorrelated
theorem cov_X_Z : cov X Z = 0 := sorry

end Z_is_normal_dist_XZ_dist_YZ_dist_X_plus_Z_cov_X_Z_l789_789808


namespace find_general_formula_sum_b_lt_one_l789_789705

open Nat

-- Define the given conditions
def sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) := ∀ n, S (n + 1) + S n = (n + 1) ^ 2

def initial_term (a : ℕ → ℕ) := a 1 = 1

-- Define the sequence \{a_{n}\}
noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| n + 2 => 2 * (n + 1) + 1 - a (n + 1)

-- State the main results to be proven
theorem find_general_formula (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : sequence_sum S a)
  (h2 : initial_term a) : 
  ∀ n, a n = n := sorry

-- Define the \( b_{n} \) sequence
def b (a S : ℕ → ℕ) (n : ℕ) : ℕ := 
a (n + 1) / (S n * S (n + 1))

-- State the inequality to be proven
theorem sum_b_lt_one (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (b : ℕ → ℕ) 
  (h1 : sequence_sum S a) 
  (h2 : initial_term a) 
  (h3 : ∀ n, b n = a (n + 1) / (S n * S (n + 1))) : 
  ∀ n, ∑ i in range (n + 1), b i < 1 := sorry

end find_general_formula_sum_b_lt_one_l789_789705


namespace celine_cycled_miles_approx_l789_789625

theorem celine_cycled_miles_approx:
  ∀ (revolutions_per_reset : ℕ) (resets : ℕ) (final_revolutions : ℕ) (revolutions_per_mile : ℕ),
    revolutions_per_reset = 90000 → resets = 37 → final_revolutions = 25000 → revolutions_per_mile = 1500 → 
    (37 * 90000 + 25000) / 1500 ≈ 2237 :=
by
  intros _ _ _ _
  assume h1 h2 h3 h4
  have h5 : 37 * 90000 + 25000 = 3355000, by sorry
  have h6 : 3355000 / 1500 ≈ 2237, by sorry
  exact h6

end celine_cycled_miles_approx_l789_789625


namespace integer_pairs_satisfy_equation_l789_789645

theorem integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), a ≥ 1 ∧ b ≥ 1 → a ^ (b ^ 2) = b ^ a →
  (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) :=
by
  intros a b h1 h2
  -- skip proof
  sorry

end integer_pairs_satisfy_equation_l789_789645


namespace g_value_at_2_l789_789465

theorem g_value_at_2 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2 - 2) : g 2 = 11 / 28 :=
sorry

end g_value_at_2_l789_789465


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789238

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789238


namespace largest_possible_integer_in_list_l789_789147

theorem largest_possible_integer_in_list :
  ∃ (a b c d e : ℕ), 
  (a = 6) ∧ 
  (b = 6) ∧ 
  (c = 7) ∧ 
  (∀ x, x ≠ a ∨ x ≠ b ∨ x ≠ c → x ≠ 6) ∧ 
  (d > 7) ∧ 
  (12 = (a + b + c + d + e) / 5) ∧ 
  (max a (max b (max c (max d e))) = 33) := by
  sorry

end largest_possible_integer_in_list_l789_789147


namespace min_g_value_l789_789287

variable {α : Type _} [NormedField α] [NormedSpace α α] [EuclideanSpace α]

-- Given conditions
variables (A B C D : EuclideanSpace α) 
variables (AD BC AC BD AB CD : α)
variables (g : EuclideanSpace α → α)
variables (p q : ℕ)

-- Defining conditions from the problem
def conditions : Prop := 
  (AD = 26) ∧ (BC = 26) ∧ 
  (AC = 42) ∧ (BD = 42) ∧ 
  (AB = 50) ∧ (CD = 50) ∧ 
  (∀ Y : EuclideanSpace α, g Y = (dist A Y + dist B Y + dist C Y + dist D Y)) ∧ 
  (g = λ Y, dist A Y + dist B Y + dist C Y + dist D Y) ∧
  (∃ p q : ℕ, nat.gcd p q = 1 ∧ ¬∃ k, k^2 ∣ q ∧ g (minimizing_point : EuclideanSpace α) = p * real.sqrt q)

-- The theorem we want to prove
theorem min_g_value : conditions A B C D AD BC AC BD AB CD g →
  ∃ p q : ℕ, g (minimizing_point : EuclideanSpace α) = 4 * real.sqrt 650 ∧ (4 + 650 = 654) :=
sorry

end min_g_value_l789_789287


namespace solution_set_of_inequality_l789_789067

theorem solution_set_of_inequality (x : ℝ) : x * (2 - x) > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l789_789067


namespace term_250_of_non_square_sequence_l789_789509

/-- The sequence of positive integers, omitting all the perfect squares. -/
def non_square_sequence : Nat → Nat
| 0 => 1
| n + 1 => if is_square (non_square_sequence n + 1) then non_square_sequence n + 2 else non_square_sequence n + 1

/-- A number is a perfect square if there exists some integer whose square is equal to that number. -/
def is_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

/-- The 250th term of the increasing sequence of positive integers, omitting all the perfect squares, is 265. -/
theorem term_250_of_non_square_sequence : non_square_sequence 249 = 265 :=
sorry

end term_250_of_non_square_sequence_l789_789509


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789517

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789517


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789221

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789221


namespace sum_of_f_p_squared_l789_789661

def f (p : ℕ) : ℕ :=
  Nat.find (λ a, ¬ ∃ n, p ∣ n^2 - a)

def first_100k_odd_primes_sum (N : ℕ) : Prop :=
  N = (Finset.range 100000).sum (λ i, (f (Nat.prime i)^2))

theorem sum_of_f_p_squared : ∃ N, first_100k_odd_primes_sum N := sorry

end sum_of_f_p_squared_l789_789661


namespace utility_bill_amount_l789_789012

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l789_789012


namespace find_m_l789_789817

variables (a b c : ℝ × ℝ) (m : ℝ)

def vector_a := (-2, 3)
def vector_b := (3, 1)
def vector_c := (-7, m)

theorem find_m (h : (vector_a + (3 : ℝ) • vector_b) = (7, 6))
  (h_parallel : (7, 6) = k • vector_c) :
  m = -6 :=
sorry

end find_m_l789_789817


namespace parabola_focus_distance_l789_789048

theorem parabola_focus_distance :
  ∀ (P : Point) (x : ℝ) (F : Point),
    P.2 = 4 →
    P.1 = x →
    P.2 = 1/4 * P.1^2 →
    F = (0, 1) →
    distance F P = 5 :=
by
  sorry

end parabola_focus_distance_l789_789048


namespace sin_theta_of_perpendicular_lines_l789_789321

theorem sin_theta_of_perpendicular_lines (θ : ℝ) :
  (∃ l : ℝ, Line.is_perpendicular (Line.mk l θ) (Line.mk (-1 / 2) (3 / 2))) →
  Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by
  intro h
  sorry

end sin_theta_of_perpendicular_lines_l789_789321


namespace triangle_is_isosceles_l789_789383

-- Given condition as an assumption in Lean
def sides_opposite_to_angles (a b c A B C : ℝ) (triangle : Prop) :=
  a = 2 * b * real.cos C

-- Conclusion that needs to be proved
theorem triangle_is_isosceles
  {a b c A B C : ℝ}
  (h1 : sides_opposite_to_angles a b c A B C (triangle a b c A B C)) :
  (∃ t : triangle, is_isosceles t) :=
sorry

end triangle_is_isosceles_l789_789383


namespace train_speed_l789_789087

theorem train_speed (v : ℝ) 
  (h1 : 50 * 2.5 + v * 2.5 = 285) : v = 64 := 
by
  -- h1 unfolds conditions into the mathematical equation
  -- here we would have the proof steps, adding a "sorry" to skip proof steps.
  sorry

end train_speed_l789_789087


namespace radius_ratio_of_circumscribed_truncated_cone_l789_789894

theorem radius_ratio_of_circumscribed_truncated_cone 
  (R r ρ : ℝ) 
  (h : ℝ) 
  (Vcs Vg : ℝ) 
  (h_eq : h = 2 * ρ)
  (Vcs_eq : Vcs = (π / 3) * h * (R^2 + r^2 + R * r))
  (Vg_eq : Vg = (4 * π * (ρ^3)) / 3)
  (Vcs_Vg_eq : Vcs = 2 * Vg) :
  (R / r) = (3 + Real.sqrt 5) / 2 := 
sorry

end radius_ratio_of_circumscribed_truncated_cone_l789_789894


namespace digits_difference_l789_789047

-- Definitions based on conditions
variables (X Y : ℕ)

-- Condition: The difference between the original number and the interchanged number is 27
def difference_condition : Prop :=
  (10 * X + Y) - (10 * Y + X) = 27

-- Problem to prove: The difference between the two digits is 3
theorem digits_difference (h : difference_condition X Y) : X - Y = 3 :=
by sorry

end digits_difference_l789_789047


namespace avg_of_multiples_of_10_from_10_to_300_l789_789510

-- Define the sequence parameters
def first_term := 10
def common_difference := 10
def last_term := 300

-- Define the number of terms in the sequence
def number_of_terms : Nat := ((last_term - first_term) / common_difference) + 1

-- Define the sum of the sequence
def sum_of_sequence := (number_of_terms * (first_term + last_term)) / 2

-- Define the average of the sequence
def average_of_sequence := sum_of_sequence / number_of_terms

theorem avg_of_multiples_of_10_from_10_to_300 : average_of_sequence = 155 :=
by
  rw [first_term, common_difference, last_term, number_of_terms, sum_of_sequence, average_of_sequence]
  simp
  sorry

end avg_of_multiples_of_10_from_10_to_300_l789_789510


namespace regular_triangular_pyramid_lateral_surface_area_l789_789131

open Real

noncomputable def lateral_surface_area (a : ℝ) : ℝ :=
  (a^2 * sqrt 39) / 4

theorem regular_triangular_pyramid_lateral_surface_area
  (a : ℝ) 
  (h : ∀ e, angle_between_edge_and_base e = 60) :
  lateral_surface_area a = (a^2 * sqrt 39) / 4 :=
sorry

end regular_triangular_pyramid_lateral_surface_area_l789_789131


namespace rolling_circle_trace_eq_envelope_l789_789934

-- Definitions for the geometrical setup
variable {a : ℝ} (C : ℝ → ℝ → Prop)

-- The main statement to prove
theorem rolling_circle_trace_eq_envelope (hC : ∀ t : ℝ, C (a * t) a) :
  ∃ P : ℝ × ℝ → Prop, ∀ t : ℝ, C (a/2 * t + a/2 * Real.sin t) (a/2 + a/2 * Real.cos t) :=
by
  sorry

end rolling_circle_trace_eq_envelope_l789_789934


namespace radius_of_circle_of_roots_l789_789158

theorem radius_of_circle_of_roots (z : ℂ)
  (h : (z + 2)^6 = 64 * z^6) :
  ∃ r : ℝ, r = 4 / 3 ∧ ∀ z, (z + 2)^6 = 64 * z^6 →
  abs (z + 2) = (4 / 3 : ℝ) * abs z :=
by
  sorry

end radius_of_circle_of_roots_l789_789158


namespace normal_price_of_article_l789_789907

theorem normal_price_of_article (P : ℝ) (h : 0.90 * 0.80 * P = 36) : P = 50 :=
by {
  sorry
}

end normal_price_of_article_l789_789907


namespace find_f_half_l789_789672

theorem find_f_half (f : ℝ → ℝ) (h : ∀ x, f (2 * x / (x + 1)) = x^2 - 1) : f (1 / 2) = -8 / 9 :=
by
  sorry

end find_f_half_l789_789672


namespace negate_proposition_l789_789720

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 + x + 1 > 0)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 + x + 1 ≤ 0) := by
  sorry

end negate_proposition_l789_789720


namespace number_of_triples_l789_789992

def sign (a : ℝ) : ℝ :=
if a > 0 then 1
else if a = 0 then 0
else -1

theorem number_of_triples : 
  (∃ x y z : ℝ, x = 3027 - 3028 * sign (y + z + 1) ∧ y = 3027 - 3028 * sign (x + z + 1) ∧ z = 3027 - 3028 * sign (x + y + 1)) = 3 :=
sorry

end number_of_triples_l789_789992


namespace exponential_function_decreasing_when_0_lt_a_lt_1_l789_789062

def exponential_function_incorrect_major_premise (a : ℝ) (h : 0 < a) : Prop :=
  ¬ (∀ a > 0, ∀ x : ℝ, a^x > a^(x + 1))

theorem exponential_function_decreasing_when_0_lt_a_lt_1 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  exponential_function_incorrect_major_premise a h1 :=
by
  sorry

end exponential_function_decreasing_when_0_lt_a_lt_1_l789_789062


namespace find_integer_n_l789_789651

theorem find_integer_n : ∃ (n : ℤ), (-90 ≤ n ∧ n ≤ 90) ∧ (Real.sin (n * Real.pi / 180) = Real.cos (456 * Real.pi / 180)) ∧ n = -6 := 
by
  sorry

end find_integer_n_l789_789651


namespace elective_ways_l789_789006

theorem elective_ways (students_courses : ℕ → ℕ) (n_courses_per_year : ℕ) : 
  (∀ n, students_courses n ≤ 3) ∧ 
  (∑ i in Finset.Icc 1 3, students_courses i = 4) → 
  78 = 78 :=
by
  sorry

end elective_ways_l789_789006


namespace area_fraction_l789_789473

noncomputable theory
open_locale classical

-- Define the octagon and midpoints structure
structure RegularOctagon (α : Type) [OrderedField α] :=
  (vertices : Fin 8 → EuclideanSpace α (Fin 2))
  (regular : ∀ i j : Fin 8, ∥vertices i - vertices j∥ = ∥vertices 0 - vertices 1∥)

-- Define the midpoints formation of the octagon
def Midpoints (α : Type) [OrderedField α] (O : RegularOctagon α) : Fin 8 → EuclideanSpace α (Fin 2) :=
  λ i, (O.vertices i + O.vertices ((i + 1) % 8)) / 2

-- The areas of octagons (larger and smaller)
def area (α : Type) [OrderedField α] (O : RegularOctagon α) : α := sorry
def area_smaller(α : Type) [OrderedField α] (O : RegularOctagon α) (midpoints : Fin 8 → EuclideanSpace α (Fin 2)) : α := sorry

-- The final theorem
theorem area_fraction (α : Type) [OrderedField α] (O : RegularOctagon α) (M : Fin 8 → EuclideanSpace α (Fin 2) := Midpoints α O) :
  area_smaller α O M = area α O / 4 :=
sorry

end area_fraction_l789_789473


namespace period_of_tan_scaled_l789_789102

theorem period_of_tan_scaled (a : ℝ) (h : a ≠ 0) : 
  (∃ l : ℝ, l > 0 ∧ ∀ x : ℝ, tan(x / a) = tan((x + l) / a)) ↔ 
  a = 1/3 → (∃ l : ℝ, l = 3 * π) := 
sorry

end period_of_tan_scaled_l789_789102


namespace hyperbola_eccentricity_l789_789296

variable {a b c : ℝ} (h_a : a > 0) (h_b : b > 0)
variable (C : Set (ℝ × ℝ)) (h_C : ∀ (x y : ℝ), (x, y) ∈ C ↔ (ℝ × ℝ) := {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
variable (A : ℝ × ℝ := (a, 0))
variable (F : ℝ × ℝ := (c, 0))
variable (B : ℝ × ℝ := (c, b^2 / a))
variable (h_slope : (b^2 / a - 0) / (c - a) = 3)

theorem hyperbola_eccentricity (h_b_square : b^2 = c^2 - a^2) : c / a = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789296


namespace period_of_tan_x_div_3_l789_789108

theorem period_of_tan_x_div_3 : ∃ T > 0, ∀ x, tan (x / 3) = tan ((x + T) / 3) :=
by
  use 3 * Real.pi
  intros x
  rew_rw (3 * Real.pi)
  rw [Real.tan_periodic]
  sorry

end period_of_tan_x_div_3_l789_789108


namespace find_geometric_sequence_pairs_l789_789066

theorem find_geometric_sequence_pairs (b s : ℕ) (hb : 0 < b) (hs : 0 < s) 
    (hlog_sum : (Finset.range 15).sum (λ n, Real.logb 4 ((b : ℝ) * s^n)) = 2015) :
    ∃ n : ℕ, n = 39 :=
by
  sorry

end find_geometric_sequence_pairs_l789_789066


namespace Alice_min_speed_l789_789461

theorem Alice_min_speed
  (distance : Real := 120)
  (bob_speed : Real := 40)
  (alice_delay : Real := 0.5)
  (alice_min_speed : Real := distance / (distance / bob_speed - alice_delay)) :
  alice_min_speed = 48 := 
by
  sorry

end Alice_min_speed_l789_789461


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789247

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789247


namespace stone_solution_l789_789132

noncomputable def stone_problem : Prop :=
  ∃ y : ℕ, (∃ x z : ℕ, x + y + z = 100 ∧ x + 10 * y + 50 * z = 500) ∧
    ∀ y1 y2 : ℕ, (∃ x1 z1 : ℕ, x1 + y1 + z1 = 100 ∧ x1 + 10 * y1 + 50 * z1 = 500) ∧
                (∃ x2 z2 : ℕ, x2 + y2 + z2 = 100 ∧ x2 + 10 * y2 + 50 * z2 = 500) →
                y1 = y2

theorem stone_solution : stone_problem :=
sorry

end stone_solution_l789_789132


namespace sqrt_meaningful_condition_l789_789903

theorem sqrt_meaningful_condition (a : ℝ) : 2 - a ≥ 0 → a ≤ 2 := by
  sorry

end sqrt_meaningful_condition_l789_789903


namespace smallest_number_three_squares_three_ways_l789_789076

theorem smallest_number_three_squares_three_ways :
  ∃ n : ℕ, n = 110 ∧ 
    (∃ a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℕ, 
      n = a1^2 + b1^2 + c1^2 ∧ n = a2^2 + b2^2 + c2^2 ∧ n = a3^2 + b3^2 + c3^2 ∧ 
      {⟨a1, b1, c1⟩, ⟨a2, b2, c2⟩, ⟨a3, b3, c3⟩}.nodup) ∧
    (∀ m : ℕ, (∃ a b c x y z w u v : ℕ, 
      m = a^2 + b^2 + c^2 ∧ m = x^2 + y^2 + z^2 ∧ m = w^2 + u^2 + v^2 ∧ 
      {⟨a, b, c⟩, ⟨x, y, z⟩, ⟨w, u, v⟩}.nodup) → m ≥ 110) := 
begin
  sorry
end

end smallest_number_three_squares_three_ways_l789_789076


namespace solution_to_fractional_equation_l789_789487

theorem solution_to_fractional_equation :
  ∃ x : ℝ, 4 / (x - 1) = 3 / x ∧ x = -3 :=
by
  use -3
  split
  sorry

end solution_to_fractional_equation_l789_789487


namespace number_of_zeros_f_l789_789057

noncomputable def f (x : ℝ) : ℝ := 2 * x * |Real.log 0.5 x| - 1

theorem number_of_zeros_f : (set_of (λ x, f x = 0)).finite ∧ (set_of (λ x, f x = 0)).to_finset.card = 2 :=
sorry

end number_of_zeros_f_l789_789057


namespace solve_inequality_system_l789_789846

theorem solve_inequality_system (x : ℝ) (h1 : 2 * x + 1 < 5) (h2 : 2 - x ≤ 1) : 1 ≤ x ∧ x < 2 :=
by
  sorry

end solve_inequality_system_l789_789846


namespace distance_PQ_is_4_l789_789765

noncomputable theory

open Real

-- Definitions of circle C and lines l₁, l₂
def circle_parametric := ∀ θ : ℝ, (x y : ℝ) × (x = 2 + 2 * cos θ ∧ y = 2 * sin θ)

def line_cartesian_l1 := ∀ x y : ℝ, x + 1 = 0

def line_polar_l2 := ∀ (ρ : ℝ), θ = π / 3

-- Proof problem statement
theorem distance_PQ_is_4 :
  (∀ θ, ∃ (x y : ℝ), x = 2 + 2 * cos θ ∧ y = 2 * sin θ) → -- Circle C parametric equation
  (∀ x, x + 1 = 0) →                                    -- Line l1 in Cartesian coordinates
  (θ = π / 3 ∧ ∃ ρ : ℝ, true) →                            -- Line l2 in polar coordinates
  (|2 - (-2)| = 4) := sorry

end distance_PQ_is_4_l789_789765


namespace isosceles_trapezoid_solid_of_revolution_l789_789604

noncomputable def surface_area_of_solid_of_revolution (AD BC : ℝ) (angle_DAB : ℝ) : ℝ := 
  4 * π * Real.sqrt 3

noncomputable def volume_of_solid_of_revolution (AD BC : ℝ) (angle_DAB : ℝ) : ℝ := 
  2 * π

theorem isosceles_trapezoid_solid_of_revolution (AD BC : ℝ) (angle_DAB : ℝ)
  (hAD : AD = 2) (hBC : BC = 3) (hAngle : angle_DAB = 60) :
  surface_area_of_solid_of_revolution AD BC angle_DAB = 4 * π * Real.sqrt 3 ∧
  volume_of_solid_of_revolution AD BC angle_DAB = 2 * π := 
by
  sorry

end isosceles_trapezoid_solid_of_revolution_l789_789604


namespace tangent_lines_to_circle_l789_789498

-- Conditions
def regions_not_enclosed := 68
def num_lines := 30 - 4

-- Theorem statement
theorem tangent_lines_to_circle (h: regions_not_enclosed = 68) : num_lines = 26 :=
by {
  sorry
}

end tangent_lines_to_circle_l789_789498


namespace gunther_typing_l789_789639

theorem gunther_typing :
  ∀ (t1 t2 t3 : ℕ) (s1 s2 s3 : ℕ),
  (t1 = 120) → (s1 = 160) →
  (t2 = 180) → (s2 = 200) →
  (t3 = 240) → (s3 = 140) →
  (t1 / 3 * s1 + t2 / 3 * s2 + t3 / 3 * s3 = 29600) :=
by
  intros t1 t2 t3 s1 s2 s3
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end gunther_typing_l789_789639


namespace expression_in_multiply_form_l789_789622

def a : ℕ := 3 ^ 1005
def b : ℕ := 7 ^ 1006
def m : ℕ := 114337548

theorem expression_in_multiply_form : 
  (a + b)^2 - (a - b)^2 = m * 10 ^ 1006 :=
by
  sorry

end expression_in_multiply_form_l789_789622


namespace positive_integer_arithmetic_D_l789_789659

def D (n : ℕ) : set ℕ := { k | 1 ≤ k ∧ k ≤ n ∧ Nat.coprime k n }

theorem positive_integer_arithmetic_D (n : ℕ) : 
  (∃ (a d : ℕ), ∀ k ∈ D n, ∃ m : ℕ, k = a + m * d) ↔ (Nat.Prime n ∧ n ≥ 5) ∨ (∃ t : ℕ, n = 2^t ∧ t ≥ 3) :=
sorry

end positive_integer_arithmetic_D_l789_789659


namespace buses_required_l789_789953

theorem buses_required (students : ℕ) (bus_capacity : ℕ) (h_students : students = 325) (h_bus_capacity : bus_capacity = 45) : 
∃ n : ℕ, n = 8 ∧ bus_capacity * n ≥ students :=
by
  sorry

end buses_required_l789_789953


namespace largest_integer_less_than_100_with_remainder_4_l789_789200

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l789_789200


namespace find_c_l789_789061

theorem find_c (c : ℝ) 
  (h : (⟨9, c⟩ : ℝ × ℝ) = (11/13 : ℝ) • ⟨-3, 2⟩) : 
  c = 19 :=
sorry

end find_c_l789_789061


namespace solve_system_l789_789848

theorem solve_system :
  ∃ (x y : ℤ), 2 * x + y = 4 ∧ x + 2 * y = -1 ∧ x = 3 ∧ y = -2 :=
by
  use [3, -2]
  simp
  ring
  sorry

end solve_system_l789_789848


namespace pencil_perpendicular_to_floor_l789_789826

-- Define the given conditions
variable {Floor : Type} [plane : EuclideanSpace ℝ (Fin 2)] (P : Floor)

-- Define the vertical pencil as being represented perpendicular to the floor
def vertical_pencil (p : point P) : prop := 
  ∀ (fc : line P), contains p fc → ∀ (l : line P), contains p l → is_perpendicular fc l

-- Statement we want to prove: Any line on the floor through the contact point is perpendicular to the line representing the pencil.
theorem pencil_perpendicular_to_floor (p : point P) :
  vertical_pencil P p → ∀ l : line P, contains p l → is_perpendicular (vertical_pencil P p) l := 
  sorry

end pencil_perpendicular_to_floor_l789_789826


namespace AE_perpendicular_CD_l789_789904

variables {C1 C2 : Type*}
variables {A B P Q M N C D E : Type*}

-- Situations and conditions
-- Two circles C1, C2 intersecting at A and B
-- Points P and Q on C1 and C2 respectively such that |AP| = |AQ|
-- Segment PQ intersects C1 and C2 at M and N respectively
-- C is the center of the arc BP of C1 not containing A
-- D is the center of the arc BQ of C2 not containing A
-- E is the intersection of CM and DN

theorem AE_perpendicular_CD
  (intersect_C1_C2 : intersects (C1, C2) A B)
  (P_on_C1 : on_circle C1 P)
  (Q_on_C2 : on_circle C2 Q)
  (AP_EQ_AQ : distance A P = distance A Q)
  (PQ_intersects_C1_at_M : intersects_segment (P, Q) C1 M)
  (PQ_intersects_C2_at_N : intersects_segment (P, Q) C2 N)
  (C_center_of_arc_BP_not_containing_A : center_of_arc_not_containing (C1, B, P) A C)
  (D_center_of_arc_BQ_not_containing_A : center_of_arc_not_containing (C2, B, Q) A D)
  (E_intersection : intersection_point (line_through C M) (line_through D N) E) :
  perpendicular (line_through A E) (line_through C D) :=
begin
  sorry,
end

end AE_perpendicular_CD_l789_789904


namespace woody_savings_l789_789553

-- Definitions from conditions
def console_cost : Int := 282
def weekly_allowance : Int := 24
def saving_weeks : Int := 10

-- Theorem to prove that the amount Woody already has is $42
theorem woody_savings :
  (console_cost - (weekly_allowance * saving_weeks)) = 42 := 
by
  sorry

end woody_savings_l789_789553


namespace calc_f_x_plus_2_minus_f_x_l789_789739

variable (x : ℝ)

noncomputable def f : ℝ → ℝ := λ x, 9 ^ x

theorem calc_f_x_plus_2_minus_f_x : f (x + 2) - f x = 80 * f x :=
by
  sorry

end calc_f_x_plus_2_minus_f_x_l789_789739


namespace hexagon_largest_angle_l789_789081

theorem hexagon_largest_angle (x : ℝ) 
  (h_angles_sum : 80 + 100 + x + x + x + (2 * x + 20) = 720) : 
  (2 * x + 20) = 228 :=
by 
  sorry

end hexagon_largest_angle_l789_789081


namespace initial_weight_l789_789075

noncomputable def initial_average_weight (A : ℝ) : Prop :=
  let total_weight_initial := 20 * A
  let total_weight_new := total_weight_initial + 210
  let new_average_weight := 181.42857142857142
  total_weight_new / 21 = new_average_weight

theorem initial_weight:
  ∃ A : ℝ, initial_average_weight A ∧ A = 180 :=
by
  sorry

end initial_weight_l789_789075


namespace period_of_tan_x_over_3_l789_789104

theorem period_of_tan_x_over_3 : ∃ T > 0, ∀ x, tan (x / 3) = tan ((x + T) / 3) :=
by
  use 3 * Real.pi
  sorry

end period_of_tan_x_over_3_l789_789104


namespace impossible_cube_placement_l789_789784

open Function

def cube_vertices (n : Nat) : Prop := ∃ (a b c d e f g h : Nat), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h ∧
  (∀ (x y : Nat), (1 ≤ x ∧ x ≤ 245) ∧ (1 ≤ y ∧ y ≤ 245) ∧ x ≠ y → ¬ (13 ∣ x) ∧ ¬ (13 ∣ y)) ∧
  (∀ (x y : Nat), (x, y) ∈ {(a, b), (a, c), (a, d), (d, e), (b, e), (c, f), (b, h), (c, g), (d, h), (e, f), (f, g), (g, h)} → gcd x y > 1) ∧
  (∀ (x y : Nat), (x ≠ y) ∧ (x, y) ∉ {(a, b), (a, c), (a, d), (d, e), (b, e), (c, f), (b, h), (c, g), (d, h), (e, f), (f, g), (g, h)} → gcd x y = 1)

theorem impossible_cube_placement : ¬ cube_vertices 8 :=
by
  sorry

end impossible_cube_placement_l789_789784


namespace max_integer_k_l789_789335

-- Definitions of the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2

-- Definition of the inequality condition
theorem max_integer_k (k : ℝ) : 
  (∀ x : ℝ, x > 2 → k * (x - 2) < x * f x + 2 * g' x + 3) ↔
  k ≤ 5 :=
sorry

end max_integer_k_l789_789335


namespace min_value_of_fraction_l789_789882

def f (x : ℝ) : ℝ := (3 * x^2 + 6 * x + 5) / (1/2 * x^2 + x + 1)

theorem min_value_of_fraction : ∃ y, y = 4 ∧ ∀ x, f x ≥ y := 
by
  sorry

end min_value_of_fraction_l789_789882


namespace greatest_integer_with_gcd_l789_789521

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l789_789521


namespace bricks_needed_l789_789731

noncomputable def volume (length width height : ℝ) : ℝ := length * width * height

theorem bricks_needed (length_wall height_wall width_wall length_brick height_brick width_brick : ℝ) :
  length_wall = 850 → height_wall = 600 → width_wall = 22.5 →
  length_brick = 25 → height_brick = 11.25 → width_brick = 6 →
  volume length_wall height_wall width_wall / volume length_brick height_brick width_brick = 6800 :=
by {
  intros,
  have volume_wall := volume length_wall height_wall width_wall,
  have volume_wall_value : volume_wall = 850 * 600 * 22.5, {
    simp [volume, *]
  },
  have volume_brick := volume length_brick height_brick width_brick,
  have volume_brick_value : volume_brick = 25 * 11.25 * 6, {
    simp [volume, *]
  },
  have bricks_needed_value : volume length_wall height_wall width_wall / volume length_brick height_brick width_brick = 11_475_000 / 1_687.5, {
    simp [volume_wall_value, volume_brick_value],
  },
  norm_num at bricks_needed_value,
  exact bricks_needed_value,
}

end bricks_needed_l789_789731


namespace parking_spots_first_level_l789_789584

theorem parking_spots_first_level (x : ℕ) 
    (h1 : ∃ x, x + (x + 7) + (x + 13) + 14 = 46) : x = 4 :=
by
  sorry

end parking_spots_first_level_l789_789584


namespace original_price_is_approx_41_15_l789_789971

noncomputable def original_price_given_conditions : ℝ :=
  let num_people := 5
  let amount_paid_per_person := 8
  let discount_rate := 0.10
  let tax_rate := 0.08
  let final_total_paid := (num_people:ℝ) * amount_paid_per_person
  let final_price := 0.90 * (1 + tax_rate)
  final_total_paid / final_price

theorem original_price_is_approx_41_15 :
  original_price_given_conditions ≈ 41.15 := 
by
  sorry

end original_price_is_approx_41_15_l789_789971


namespace part1_part2_l789_789341

variable (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1)

noncomputable def a_seq : ℕ → ℝ
| 1     := a
| (n+1) := a_seq n * b_seq n

noncomputable def b_seq : ℕ → ℝ
| 1     := 1 - a
| (n+1) := b_seq n / (1 - (a_seq n)^2)

theorem part1 (n : ℕ) (hn: n > 0) : a_seq a n + b_seq a n = 1 := 
by sorry

theorem part2 (n : ℕ) (hn: n > 0) : a_seq a n = a / (1 + (n-1) * a) := 
by sorry

end part1_part2_l789_789341


namespace log_diff_example_l789_789998

theorem log_diff_example : log 5 625 - log 5 (1/25) = 6 := 
by sorry

end log_diff_example_l789_789998


namespace correct_payment_l789_789972

/-
Bruce purchased 7 kg of grapes at the rate of $70 per kg,
9 kg of mangoes at the rate of $55 per kg,
5 kg of apples at the rate of $40 per kg,
and 3 kg of oranges at the rate of $30 per kg.
The shopkeeper offered a 10% discount on the total amount.
Additionally, there is a 5% sales tax applied.
-/

def grapes_kg : ℕ := 7
def grapes_rate : ℕ := 70
def mangoes_kg : ℕ := 9
def mangoes_rate : ℕ := 55
def apples_kg : ℕ := 5
def apples_rate : ℕ := 40
def oranges_kg : ℕ := 3
def oranges_rate : ℕ := 30
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

/-
Question: How much amount did Bruce pay to the shopkeeper after considering the discount and sales tax?
Answer: $1204.88
-/
def total_amount_paid : ℝ := 1204.88

theorem correct_payment :
  let total_cost_before_discount := (grapes_kg * grapes_rate) + (mangoes_kg * mangoes_rate) + (apples_kg * apples_rate) + (oranges_kg * oranges_rate)
  let discount_amount := discount_rate * total_cost_before_discount
  let amount_after_discount := total_cost_before_discount - discount_amount
  let tax_amount := tax_rate * amount_after_discount
  total_amount_paid = Float.ceil_dec (amount_after_discount + tax_amount * 100) / 100 := 
begin
  sorry
end

end correct_payment_l789_789972


namespace sqrt_eq_solutions_l789_789198

theorem sqrt_eq_solutions (x : ℝ) : 
  (Real.sqrt ((2 + Real.sqrt 5) ^ x) + Real.sqrt ((2 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end sqrt_eq_solutions_l789_789198


namespace probability_twice_as_large_is_one_third_l789_789977

open ProbabilityTheory

noncomputable def prob_twice_as_large : ℝ :=
  let P := (volume (({p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1000 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3000} ∩ {p : ℝ × ℝ | p.2 ≥ 2 * p.1})) / volume {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1000 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3000}) in
  P.toReal

theorem probability_twice_as_large_is_one_third :
  prob_twice_as_large = 1 / 3 :=
by
  sorry

end probability_twice_as_large_is_one_third_l789_789977


namespace area_ratio_l789_789382

open_locale classical
noncomputable theory

variables {Point : Type*} [metric_space Point] [measurable_space Point] [normed_space ℝ Point]
variables (A B C P Q R : Point)

-- Assume points form a triangle and segment the perimeter
variables (h_triangle : triangle A B C)
variables (h_segment : segment_perimeter_eq_thirds A B C P Q R)
variables (h_on_side : P ∈ [A, B] ∧ Q ∈ [A, B])

-- Define the area function
def area (a b c : Point) : ℝ := sorry

-- Define the statement to be proven
theorem area_ratio (h_triangle : triangle A B C) (h_segment : segment_perimeter_eq_thirds A B C P Q R)
  (h_on_side : P ∈ [A, B] ∧ Q ∈ [A, B]) : 
  (area P Q R) / (area A B C) > 2 / 9 :=
sorry

end area_ratio_l789_789382


namespace hyperbola_eccentricity_proof_l789_789298

noncomputable def hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : ℝ :=
2

theorem hyperbola_eccentricity_proof 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : hyperbola_eccentricity a b h1 h2 C F A B h3 = 2 :=
sorry

end hyperbola_eccentricity_proof_l789_789298


namespace hyperbola_eccentricity_l789_789307

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (A B : ℝ × ℝ) (F : ℝ × ℝ)
variable (hA : A = (a, 0))
variable (hF : F = (c, 0))
variable (hB : B = (c, b ^ 2 / a))
variable (h_slope : (b ^ 2 / a - 0) / (c - a) = 3)
variable (h_hyperbola : b ^ 2 = c ^ 2 - a ^ 2)
def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity : eccentricity a c = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789307


namespace correct_option_B_l789_789916

theorem correct_option_B :
  let expr_A := (-7) - (+4) - (-5),
  let expr_B := - (+7) - (-4) - (+5),
  let expr_C := - (+7) + (+4) - (-5),
  let expr_D := (-7) - (+4) + (-5)
  expr_B = -7 + 4 - 5 :=
by
  -- skipping proof for brevity
  sorry

end correct_option_B_l789_789916


namespace distinct_integer_values_count_l789_789266

noncomputable def f (x : ℝ) : ℤ :=
  Int.floor x + Int.floor (2 * x) + Int.floor (3 * x) + 
  Int.floor (x / 2) + Int.floor (x / 3)

theorem distinct_integer_values_count :
  (finset.image f (finset.range 101)).card = 401 := 
sorry

end distinct_integer_values_count_l789_789266


namespace system_has_102_solutions_l789_789736

theorem system_has_102_solutions :
  (∃ (x y : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ 
  (3 * x + 2 * y) * (3 / x + 1 / y) = 2 ∧ 
  x^2 + y^2 ≤ 2012) → 
  (∃ n : ℤ, n = 102) := 
begin
  -- sorry is used to skip the proof
  sorry
end

end system_has_102_solutions_l789_789736


namespace find_largest_integer_l789_789254

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789254


namespace average_age_team_l789_789858

theorem average_age_team 
  (captain_age : ℕ := 32)
  (wicket_keeper_age : ℕ := 37)
  (total_members : ℕ := 15)
  (remaining_members : ℕ := total_members - 2)
  (average_age_diff : ℕ := 2)
  (A : ℚ := (43 : ℚ) / (2 : ℚ)) :
  ((total_members : ℚ) * A = (captain_age : ℚ) + (wicket_keeper_age : ℚ) + (remaining_members : ℚ) * (A - average_age_diff)) :=
by
  exact_mod_cast eq.refl 21.5

end average_age_team_l789_789858


namespace area_reflection_eq_l789_789587

-- Define the structure of a regular polygon
structure RegularPolygon (n : ℕ) :=
(vertices : Fin n → Point)

-- Define a point in a 2D space
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of the midpoint of a segment
def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

-- Definition of the reflection of a point across another point
def reflect (p m : Point) : Point :=
{ x := 2 * m.x - p.x, y := 2 * m.y - p.y }

-- Definition of the area of a polygon using shoelace formula
noncomputable def area (poly : RegularPolygon n) : ℝ :=
0.5 * | ∑ i in Finset.range n, 
(poly.vertices i).x * (poly.vertices (i + 1) % n).y - 
(poly.vertices i).y * (poly.vertices (i + 1) % n).x |

theorem area_reflection_eq (n : ℕ) (h : n = 2009) 
(poly : RegularPolygon n) (chosen_points : Fin n → Point)
(reflected_points : Fin n → Point 
 := λ i, reflect (chosen_points i) (midpoint (poly.vertices i) (poly.vertices (i + 1) % n))) :
area ⟨reflected_points⟩ = area ⟨chosen_points⟩ :=
sorry

end area_reflection_eq_l789_789587


namespace day_of_100th_day_of_2005_l789_789854

-- Define the days of the week
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Define a function to add days to a given weekday
def add_days (d: Weekday) (n: ℕ) : Weekday :=
  match d with
  | Sunday => [Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday].get? (n % 7) |>.getD Sunday
  | Monday => [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday].get? (n % 7) |>.getD Monday
  | Tuesday => [Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, Monday].get? (n % 7) |>.getD Tuesday
  | Wednesday => [Wednesday, Thursday, Friday, Saturday, Sunday, Monday, Tuesday].get? (n % 7) |>.getD Wednesday
  | Thursday => [Thursday, Friday, Saturday, Sunday, Monday, Tuesday, Wednesday].get? (n % 7) |>.getD Thursday
  | Friday => [Friday, Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday].get? (n % 7) |>.getD Friday
  | Saturday => [Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday].get? (n % 7) |>.getD Saturday

-- State the theorem
theorem day_of_100th_day_of_2005 :
  add_days Tuesday 55 = Monday :=
by sorry

end day_of_100th_day_of_2005_l789_789854


namespace base7_to_base10_conversion_l789_789851

theorem base7_to_base10_conversion (x y : ℕ) (h1 : 563 = 5 * 7^2 + 6 * 7^1 + 3 * 7^0) (h2 : x = 2) (h3 : y = 9) :
  (x + y) / 9 = 11 / 9 :=
by
  rw [h2, h3]
  norm_num
  sorry

end base7_to_base10_conversion_l789_789851


namespace correct_operation_l789_789115

theorem correct_operation (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
by
  sorry

end correct_operation_l789_789115


namespace rightmost_nonzero_digit_of_a_k_is_9_l789_789560

def a_n (n : ℕ) := (n + 9)! / (n - 1)!

theorem rightmost_nonzero_digit_of_a_k_is_9 :
  ∃ k : ℕ, (∀ m : ℕ, m < k → (a_n m % 10 ≠ 1 ∧ a_n m % 10 ≠ 3 ∧ a_n m % 10 ≠ 5 ∧ a_n m % 10 ≠ 7 ∧ a_n m % 10 ≠ 9)) ∧ (a_n k % 10 = 9) := sorry

end rightmost_nonzero_digit_of_a_k_is_9_l789_789560


namespace find_box_depth_l789_789946

-- Definitions and conditions
noncomputable def length : ℝ := 1.6
noncomputable def width : ℝ := 1.0
noncomputable def edge : ℝ := 0.2
noncomputable def number_of_blocks : ℝ := 120

-- The goal is to find the depth of the box
theorem find_box_depth (d : ℝ) :
  length * width * d = number_of_blocks * (edge ^ 3) →
  d = 0.6 := 
sorry

end find_box_depth_l789_789946


namespace maximal_roads_l789_789133

open Finset

-- Define the main proof problem, encapsulating the conditions and conclusion
theorem maximal_roads (N : ℕ) (hN : N ≥ 1) :
  ∃ (d : ℕ), d = Nat.choose N 3 ∧
  (∀ (f : Fin N.succ → Set (Fin N.succ)),
    (∀ (i : Fin N.succ), f i ⊆ (Fin N.succ).erase i) →
    (∀ i j, i ≠ j → (f i ∩ f j).card ≤ 1) →
    (∀ S : Finset (Fin N.succ), S.card < N → 
      ∃ i ∈ S, ∀ j ∈ S, i ≠ j → f i ∪ f j ≠ univ) →
    (∑ i, (f i).card = d)) :=
begin
  sorry
end

end maximal_roads_l789_789133


namespace lara_additional_miles_needed_l789_789794

theorem lara_additional_miles_needed :
  ∀ (d1 d2 d_total t1 speed1 speed2 avg_speed : ℝ),
    d1 = 20 →
    speed1 = 25 →
    speed2 = 40 →
    avg_speed = 35 →
    t1 = d1 / speed1 →
    d_total = d1 + d2 →
    avg_speed = (d_total) / (t1 + d2 / speed2) →
    d2 = 64 :=
by sorry

end lara_additional_miles_needed_l789_789794


namespace eval_five_over_two_l789_789466

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 2 else Real.log (x - 1) / Real.log 2

theorem eval_five_over_two : f (5 / 2) = -1 := by
  sorry

end eval_five_over_two_l789_789466


namespace inequality_has_exactly_one_solution_l789_789987

-- Definitions based on the conditions
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 3 * a

-- The main theorem that encodes the proof problem
theorem inequality_has_exactly_one_solution (a : ℝ) : 
  (∃! x : ℝ, |f x a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end inequality_has_exactly_one_solution_l789_789987


namespace equation_of_trisection_line_l789_789423

/-- Let P be the point (1, 2) and let A and B be the points (2, 3) and (-3, 0), respectively. 
    One of the lines through point P and a trisection point of the line segment joining A and B has 
    the equation 3x + 7y = 17. -/
theorem equation_of_trisection_line :
  let P : ℝ × ℝ := (1, 2)
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (-3, 0)
  -- Definition of the trisection points
  let T1 : ℝ × ℝ := ((2 + (-3 - 2) / 3) / 1, (3 + (0 - 3) / 3) / 1) -- First trisection point
  let T2 : ℝ × ℝ := ((2 + 2 * (-3 - 2) / 3) / 1, (3 + 2 * (0 - 3) / 3) / 1) -- Second trisection point
  -- Equation of the line through P and T2 is 3x + 7y = 17
  3 * (P.1 + P.2) + 7 * (P.2 + T2.2) = 17 :=
sorry

end equation_of_trisection_line_l789_789423


namespace train_bridge_length_l789_789876

theorem train_bridge_length :
  ∀ (train_length : ℕ) (train_speed_kmph : ℕ) (crossing_time_sec : ℕ)
  (bridge_length : ℕ),
  train_length = 150 →
  train_speed_kmph = 45 →
  crossing_time_sec = 30 →
  bridge_length = (train_speed_kmph * 1000 / 3600) * crossing_time_sec - train_length →
  bridge_length = 225 :=
by
  intros train_length train_speed_kmph crossing_time_sec bridge_length
  assume h1 h2 h3 h4
  sorry

end train_bridge_length_l789_789876


namespace intersection_solution_l789_789286

noncomputable def polar_to_cartesian := sorry

def curve_C1 (ρ θ : ℝ) : Prop := ρ * (sin θ)^2 - 4 * cos θ = 0

def curve_C2 (φ : ℝ) : ℝ × ℝ :=
  (-1 + 2 * cos φ, 2 * sin φ)

def point_P := (1/2 : ℝ, 0)

def line_l (t : ℝ) : ℝ × ℝ :=
  (1/2 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t)

theorem intersection_solution :
  (∀ ρ θ, curve_C1 ρ θ → (ρ^2 = 4 * (ρ * cos θ))) ∧
  (∀ φ, (let (x, y) := curve_C2 φ in (x + 1)^2 + y^2 = 4)) ∧
  (∃ (t1 t2 : ℝ),
    let (Mx, My) := line_l t1,
    let (Nx, Ny) := line_l t2 in
    Mx^2 + My^2 = 4 * Mx ∧ Nx^2 + Ny^2 = 4 * Nx ∧ 
    t1 + t2 = 4 * sqrt 2 ∧ t1 * t2 = -4 ∧
    (1/abs t1 + 1/abs t2 = sqrt 3)) :=
  sorry

end intersection_solution_l789_789286


namespace base_eq_9_l789_789994

theorem base_eq_9 : ∀ (b : ℕ), b ≥ 2 → (1 * b^2 + 7 * b + 2) + (1 * b^2 + 4 * b + 5) = 3 * b^2 + 2 * b + 7 ↔ b = 9 :=
by
  intro b h
  sorry

end base_eq_9_l789_789994


namespace function_monotonic_a_range_l789_789464

theorem function_monotonic_a_range:
  (∀ x1 x2, 2 ≤ x1 → x1 ≤ x2 → x2 ≤ 3 → (f x1 ≤ f x2 ∨ f x1 ≥ f x2)) → 
  (a ≤ 2 ∨ a ≥ 3) :=
by
  sorry

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * a * x + 3

variable (x : ℝ)
variable (a : ℝ)

end function_monotonic_a_range_l789_789464


namespace problem_conditions_l789_789284

variable {R : Type*} [LinearOrderedField R] 

def f (x : R) : R 

theorem problem_conditions (∀ x y : R, f(x + y) = f(x) + f(y) - 1)
  (∀ x : R, x > 0 → f(x) > 1)
  (f(1) = 3) :
proof_parts:

-- (1) Prove that f(0) = 1 and f(x) is monotonically increasing.
(1 : ∃ (f : R → R), f(0) = 1 ∧ (∀ x1 x2 : R, x1 < x2 → f(x1) < f(x2))) ∧

-- (2) Given ∀ x ∈ R, f(ax^2) + f(2x) < 6 always holds, prove the range of a is (-∞, -1/2).
(2 : ∀ (a : R), (∀ x : R, f(a * x ^ 2) + f(2 * x) < 6) → a < -1/2) := sorry

end problem_conditions_l789_789284


namespace largest_integer_lt_100_with_rem_4_div_7_l789_789217

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l789_789217


namespace smallest_form_a_l789_789654

theorem smallest_form_a : ∃ k n : ℕ, abs (11^k - 5^n) = 4 := by
  sorry

end smallest_form_a_l789_789654


namespace find_n_for_roots_form_l789_789588

theorem find_n_for_roots_form (a b c : ℤ) (h_eq : 3 * a^2 - 7 * a + 2 = 0) (m p: ℤ) (h_coprime : m.gcd n = 1) : 
  ∃ n : ℤ, ((x = m + sqrt n / p) ∨ (x = m - sqrt n / p)) ∧ n = 25 :=
by 
  sorry

end find_n_for_roots_form_l789_789588


namespace tan_half_angle_product_l789_789747

theorem tan_half_angle_product (A C : ℝ) 
    (h : 5 * (cos A + cos C) + 4 * (cos A * cos C + 1) = 0) :
    tan (A / 2) * tan (C / 2) = 3 :=
by
  sorry

end tan_half_angle_product_l789_789747


namespace greatest_integer_gcd_l789_789546

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l789_789546


namespace total_goals_during_match_l789_789188

theorem total_goals_during_match (
  A1_points_first_half : ℕ := 8,
  B_points_first_half : ℕ := A1_points_first_half / 2,
  B_points_second_half : ℕ := A1_points_first_half,
  A2_points_second_half : ℕ := B_points_second_half - 2
) : (A1_points_first_half + A2_points_second_half + B_points_first_half + B_points_second_half = 26) := by
  sorry

end total_goals_during_match_l789_789188


namespace quadrilateral_inscribed_in_conic_l789_789160

variable (Γ : Ellipse) (F : Point) (X Y Z W : Point)

-- Conditions
hypothesis (h1 : F ∈ Foci Γ)
hypothesis (h2 : Line_through F X ∧ Line_through F Y)
hypothesis (h3 : Line_through F Z ∧ Line_through F W)
hypothesis (h4 : Perpendicular (Line_through F X) (Line_through F Y))
hypothesis (h5 : Perpendicular (Line_through F Z) (Line_through F W))
hypothesis (h6 : Tangent_at Γ X ∧ Tangent_at Γ Y ∧ Tangent_at Γ Z ∧ Tangent_at Γ W)

-- Goal
theorem quadrilateral_inscribed_in_conic :
  ∃ conic, Inscribed quadrilateral conic to (Quadrilateral (Tangent_at Γ X) (Tangent_at Γ Y) (Tangent_at Γ Z) (Tangent_at Γ W)) ∧ Focus conic = F := sorry

end quadrilateral_inscribed_in_conic_l789_789160


namespace train_speed_equivalent_l789_789506

def length_train1 : ℝ := 180
def length_train2 : ℝ := 160
def speed_train1 : ℝ := 60 
def crossing_time_sec : ℝ := 12.239020878329734

noncomputable def speed_train2 (length1 length2 speed1 time : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hr := time / 3600
  let relative_speed := total_length_km / time_hr
  relative_speed - speed1

theorem train_speed_equivalent :
  speed_train2 length_train1 length_train2 speed_train1 crossing_time_sec = 40 :=
by
  simp [length_train1, length_train2, speed_train1, crossing_time_sec, speed_train2]
  sorry

end train_speed_equivalent_l789_789506


namespace find_integer_solutions_l789_789897

theorem find_integer_solutions (n : ℕ) (h1 : ∃ b : ℤ, 8 * n - 7 = b^2) (h2 : ∃ a : ℤ, 18 * n - 35 = a^2) : 
  n = 2 ∨ n = 22 := 
sorry

end find_integer_solutions_l789_789897


namespace part_a_part_b_l789_789141

-- Part (a)
theorem part_a (n k : ℕ) (C : ℕ → ℕ) (S : ℕ → ℕ → ℕ) 
  (hS : S k n = (C k 1) * S (2 * k - 1) n + (C k 3) * S (2 * k - 3) n + 
  (C k 5) * S (2 * k - 5) n + ... + (if k % 2 = 1 then C k k * S k n else C k (k - 1) * S (k + 1) n)) :
  S k n = (n^k * (n + 1)^k) / 2 :=
by
  sorry

-- Part (b)
theorem part_b (n k : ℕ) (S : ℕ → ℕ → ℕ) 
  (hpoly : S (2 * k - 1) n = polynomial.degree k (n * (n + 1) / 2)) :
  S (2 * k - 1) n = polynomial.degree k (n * (n + 1) / 2) :=
by
  sorry

end part_a_part_b_l789_789141


namespace simplify_sqrt_l789_789037

theorem simplify_sqrt (x : ℝ) (h : x = (Real.sqrt 3) + 1) : Real.sqrt (x^2) = Real.sqrt 3 + 1 :=
by
  -- This will serve as the placeholder for the proof.
  sorry

end simplify_sqrt_l789_789037


namespace seventeenth_permutation_1235_l789_789059

theorem seventeenth_permutation_1235 : 
  let digits := {1, 2, 3, 5}
  let n := 17
  ∃! x : ℕ, 
  ∃ d1 d2 d3 d4 : ℕ, 
  x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 
  {d1, d2, d3, d4} = digits ∧ 
  (list.permutations [1, 2, 3, 5]).nth (n - 1) = some [d1, d2, d3, d4] :=
begin
  sorry
end

end seventeenth_permutation_1235_l789_789059


namespace Petya_wins_l789_789562

theorem Petya_wins
  (n : ℕ)
  (h1 : n > 0)
  (h2 : ∀ d, d > n^2 → ∃ m, (m < n ∧ Nat.Prime m ∧ d - m ≤ n^2) ∨ (m % n = 0 ∧ d - m ≤ n^2) ∨ (m = 1 ∧ d - 1 ≤ n^2)) :
  ∀ d, d > n^2 → ¬(Vasya_winning_strategy n d) → Petya_winning d := by
  sorry

end Petya_wins_l789_789562


namespace find_positive_integers_l789_789197

theorem find_positive_integers (a b c : ℕ) (ha : a ≥ b) (hb : b ≥ c) :
  (∃ n₁ : ℕ, a^2 + 3 * b = n₁^2) ∧ 
  (∃ n₂ : ℕ, b^2 + 3 * c = n₂^2) ∧ 
  (∃ n₃ : ℕ, c^2 + 3 * a = n₃^2) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 37 ∧ b = 25 ∧ c = 17) :=
by
  sorry

end find_positive_integers_l789_789197


namespace range_of_m_l789_789728

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + 2 ≥ m

def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → -(7 - 3*m)^x > -(7 - 3*m)^y

theorem range_of_m (m : ℝ) :
  (proposition_p m ∧ ¬ proposition_q m) ∨ (¬ proposition_p m ∧ proposition_q m) ↔ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l789_789728


namespace evaluate_f_neg_pi_over_eight_l789_789712

-- Define the properties and conditions
axiom ω_gt_zero (ω : ℝ) : ω > 0
axiom phi_range (φ : ℝ) : 0 < φ ∧ φ < π
axiom is_even_function (f : ℝ → ℝ) : ∀ x, f (x) = f (-x)
axiom symmetry_distance (ω : ℝ) : 2 * π / ω = π

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (2 * x))

-- State the main proof problem
theorem evaluate_f_neg_pi_over_eight :
  f (-π / 8) = sqrt 2 :=
begin
  sorry,
end

end evaluate_f_neg_pi_over_eight_l789_789712


namespace find_m_l789_789420

variable {n : ℕ}
variable (a : ℕ → ℝ)
variable (T : ℕ → ℝ)

theorem find_m (m : ℕ) (h1 : a (m-1) * a (m+1) - 2 * a m = 0)
  (hT : T (2*m-1) = 128) 
  (hT_def : ∀ n, T n = ∏ i in finset.range n, a (i+1)) : 
  m = 4 :=
by
  sorry

end find_m_l789_789420


namespace smallest_angle_l789_789056

theorem smallest_angle (k : ℝ) (h1 : 4 * k + 5 * k + 7 * k = 180) : 4 * k = 45 :=
by sorry

end smallest_angle_l789_789056


namespace find_lower_percentage_l789_789933

theorem find_lower_percentage (P : ℝ) : 
  (12000 * 0.15 * 2 - 720 = 12000 * (P / 100) * 2) → P = 12 := by
  sorry

end find_lower_percentage_l789_789933


namespace select_3_products_select_exactly_1_defective_select_at_least_1_defective_l789_789754

noncomputable def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

namespace ProductInspection

def total_products : Nat := 100
def qualified_products : Nat := 98
def defective_products : Nat := 2

-- Proof Problem 1
theorem select_3_products (h : combination total_products 3 = 161700) : True := by
  trivial

-- Proof Problem 2
theorem select_exactly_1_defective (h : combination defective_products 1 * combination qualified_products 2 = 9506) : True := by
  trivial

-- Proof Problem 3
theorem select_at_least_1_defective (h : combination total_products 3 - combination qualified_products 3 = 9604) : True := by
  trivial

end ProductInspection

end select_3_products_select_exactly_1_defective_select_at_least_1_defective_l789_789754


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789514

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789514


namespace number_of_zeros_after_1_l789_789642

theorem number_of_zeros_after_1 (a b c : ℕ) (h1 : 5000 = 5 * 10^3) (h2 : (a * b)^c = a^c * b^c) (h3 : ∀ n : ℕ, 10^n contributes n zeros) :
  (5000^50) has 150 zeros following the digit 1 in its decimal expansion :=
sorry

end number_of_zeros_after_1_l789_789642


namespace tangent_line_at_point_l789_789649

noncomputable def curve (x : ℝ) : ℝ := Real.exp x + x

theorem tangent_line_at_point :
  (∃ k b : ℝ, (∀ x : ℝ, curve x = k * x + b) ∧ k = 2 ∧ b = 1) :=
by
  sorry

end tangent_line_at_point_l789_789649


namespace can_sides_be_consecutive_ints_l789_789778

theorem can_sides_be_consecutive_ints :
  ∃ (sequence : Fin 33 → Fin 34), 
    (∀ i, sequence i ∈ Finset.range 34) ∧ 
    (∀ s, ∃ k, k ∈ Finset.range 34 ∧ 
    (s = sequence s.fst + sequence s.snd ∧ 
    (s ∈ Finset.range 34))) :=
sorry

end can_sides_be_consecutive_ints_l789_789778


namespace product_of_integers_prime_at_most_one_prime_l789_789749

open Nat

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem product_of_integers_prime_at_most_one_prime (a b p : ℤ) (hp : is_prime (Int.natAbs p)) (hprod : a * b = p) :
  (is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b)) ∨ (¬is_prime (Int.natAbs a) ∧ is_prime (Int.natAbs b)) ∨ ¬is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b) :=
sorry

end product_of_integers_prime_at_most_one_prime_l789_789749


namespace probability_of_odd_factors_lt_8_l789_789550

def is_factor (n d : ℕ) : Prop := d ∣ n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def probability (A B : ℕ) : ℚ := A / B

theorem probability_of_odd_factors_lt_8 (n : ℕ) (h_factor : n = 90) : 
  probability (↑((finset.filter (λ x, is_factor n x ∧ x < 8 ∧ is_odd x) (finset.range (n+1))).card)) 
              (↑((finset.filter (λ x, is_factor n x) (finset.range (n+1))).card)) = 1/4 := by
  sorry

end probability_of_odd_factors_lt_8_l789_789550


namespace number_of_coprimes_to_15_l789_789631

open Nat

theorem number_of_coprimes_to_15 : (Finset.filter (λ a, gcd a 15 = 1) (Finset.range 15)).card = 8 := by
  sorry

end number_of_coprimes_to_15_l789_789631


namespace min_product_l789_789100

theorem min_product (S : Set ℤ) (hS : S = {-9, -7, -5, 0, 4, 6, 8}) :
  ∃ a b c : ℤ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = -336 :=
by
  use [-7, 6, 8]
  simp [hS]
  sorry

end min_product_l789_789100


namespace inequality_solution_l789_789097

theorem inequality_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 2) :
  ∀ y : ℝ, y > 0 → 4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y :=
by
  intro y hy
  sorry

end inequality_solution_l789_789097


namespace sqrt_nine_eq_three_l789_789924

theorem sqrt_nine_eq_three : Real.sqrt 9 = 3 :=
by
  sorry

end sqrt_nine_eq_three_l789_789924


namespace problem_proof_l789_789328

noncomputable def ω := 1

def f (x : ℝ) := 2 * sin (2 * x + π / 6)

def x_intervals (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (f x₁ < f x₂) ∧ (f x₂ > f x₃) ∧ (f x₃ < f x₄) ∧
  (x₄ - x₃ = π / 3) ∧ (x₂ - x₁ = π / 3) ∧ (x₃ - x₂ = π / 2)

theorem problem_proof (x₁ x₂ x₃ x₄ : ℝ) (k : ℤ) :
  x_intervals x₁ x₂ x₃ x₄ →
  x₁ = -π / 6 + k * π ∧ x₄ = k * π + π :=
sorry

end problem_proof_l789_789328


namespace abc_product_l789_789845

/-- Given a b c + a b + b c + a c + a + b + c = 164 -/
theorem abc_product :
  ∃ (a b c : ℕ), a * b * c + a * b + b * c + a * c + a + b + c = 164 ∧ a * b * c = 80 :=
by
  sorry

end abc_product_l789_789845


namespace fred_tenth_l789_789753

def position (racer : Type) : racer → ℕ

variables {Racer : Type} [decRacer : DecidableEq Racer]

variable [inhabited Racer]

/-- Eliza finished 2 places behind Bob. -/
def eliza_behind_bob (position : Racer → ℕ) (E B : Racer) : Prop :=
  position E = position B + 2

/-- Bob finished 9 places ahead of Dan. -/
def bob_ahead_dan (position : Racer → ℕ) (B D : Racer) : Prop :=
  position B = position D - 9

/-- Carla finished 3 places behind Dan. -/
def carla_behind_dan (position : Racer → ℕ) (C D : Racer) : Prop :=
  position C = position D + 3

/-- Alice finished 2 places ahead of Fred. -/
def alice_ahead_fred (position : Racer → ℕ) (A F : Racer) : Prop :=
  position A = position F - 2

/-- Fred finished 3 places behind Eliza. -/
def fred_behind_eliza (position : Racer → ℕ) (F E : Racer) : Prop :=
  position F = position E + 3

/-- Bob finished in 5th place. -/
def bob_fifth (position : Racer → ℕ) (B : Racer) : Prop :=
  position B = 5

theorem fred_tenth {Racer : Type} [decRacer : DecidableEq Racer] [inhabited Racer]
  (position : Racer → ℕ)
  (B E D C A F : Racer)
  (h1 : eliza_behind_bob position E B)
  (h2 : bob_ahead_dan position B D)
  (h3 : carla_behind_dan position C D)
  (h4 : alice_ahead_fred position A F)
  (h5 : fred_behind_eliza position F E)
  (h6 : bob_fifth position B) :
  position F = 10 :=
sorry

end fred_tenth_l789_789753


namespace quadratic_real_solution_l789_789663

theorem quadratic_real_solution (m : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h_quad : ∃ z : ℝ, z^2 + (i * z) + m = 0) : m = 0 :=
sorry

end quadratic_real_solution_l789_789663


namespace order_of_numbers_l789_789884

theorem order_of_numbers (a b c : ℝ) (h1 : a = 6^0.7) (h2 : b = 0.7^6) (h3 : c = log 0.7 6) :
  c < b ∧ b < a :=
by
  sorry

end order_of_numbers_l789_789884


namespace jack_cleaning_time_is_one_hour_l789_789787

def jackGrove : ℕ × ℕ := (4, 5)
def timeToCleanEachTree : ℕ := 6
def timeReductionFactor : ℕ := 2
def totalCleaningTimeWithHelpMin : ℕ :=
  (jackGrove.fst * jackGrove.snd) * (timeToCleanEachTree / timeReductionFactor)
def totalCleaningTimeWithHelpHours : ℕ :=
  totalCleaningTimeWithHelpMin / 60

theorem jack_cleaning_time_is_one_hour :
  totalCleaningTimeWithHelpHours = 1 := by
  sorry

end jack_cleaning_time_is_one_hour_l789_789787


namespace apples_to_eat_raw_l789_789002

noncomputable def total_apples : ℕ := 200
noncomputable def wormy_percentage : ℝ := 0.15
noncomputable def bruises_difference : ℕ := 9

def wormy_apples : ℕ := (wormy_percentage * total_apples).to_nat
def moldy_apples : ℕ := 2 * wormy_apples
def bruised_apples : ℕ := wormy_apples + bruises_difference

def total_unsuitable_apples : ℕ := wormy_apples + moldy_apples + bruised_apples
def apples_left_to_eat_raw : ℕ := total_apples - total_unsuitable_apples

theorem apples_to_eat_raw : apples_left_to_eat_raw = 71 := by
  sorry

end apples_to_eat_raw_l789_789002


namespace mean_proportion_of_3_and_4_l789_789880

theorem mean_proportion_of_3_and_4 : ∃ x : ℝ, 3 / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end mean_proportion_of_3_and_4_l789_789880


namespace milburg_children_count_l789_789072

theorem milburg_children_count (total_population adults : ℕ) 
  (h1 : total_population = 5256) 
  (h2 : adults = 2269) : 
  total_population - adults = 2987 := 
by
  rw [h1, h2]
  norm_num
  exact rfl

end milburg_children_count_l789_789072


namespace exists_right_angle_triangle_l789_789598

noncomputable theory

open Function

-- Define the color type
inductive Color
| Red
| Blue
| Green

-- Assume a coloring function for integer coordinates on the plane
def color (p : ℤ × ℤ) : Color := sorry

-- Given points with their colors
axiom point_0_0_is_red : color (0, 0) = Color.Red
axiom point_0_1_is_blue : color (0, 1) = Color.Blue
axiom each_color_used : ∀ (c : Color), ∃ (p : ℤ × ℤ), color p = c

theorem exists_right_angle_triangle : ∃ (p1 p2 p3 : ℤ × ℤ), 
  (color p1 ≠ color p2) ∧ (color p2 ≠ color p3) ∧ (color p1 ≠ color p3) ∧ 
  ((p1.1 = p2.1 ∧ p2.2 = p3.2) ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1)) := 
sorry

end exists_right_angle_triangle_l789_789598


namespace regular_hexagon_to_rhombus_l789_789949

theorem regular_hexagon_to_rhombus :
  ∀ (hex : ℕ → ℕ) (H : regular_hexagon hex), 
  ∃ (parts : list (ℕ → ℕ)), 
    (hexagon_split_into_three_parts hex parts) ∧ 
    (rearrange_parts_into_rhombus parts) := 
sorry

end regular_hexagon_to_rhombus_l789_789949


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789237

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789237


namespace volume_pyramid_PABCD_is_384_l789_789833

noncomputable def volume_of_pyramid : ℝ :=
  let AB := 12
  let BC := 6
  let PA := Real.sqrt (20^2 - 12^2)
  let base_area := AB * BC
  (1 / 3) * base_area * PA

theorem volume_pyramid_PABCD_is_384 :
  volume_of_pyramid = 384 := 
by
  sorry

end volume_pyramid_PABCD_is_384_l789_789833


namespace find_beta_ratio_l789_789984

noncomputable def triangle_area : ℝ :=
  let s := (15 + 20 + 13) / 2 in
  real.sqrt (s * (s - 15) * (s - 20) * (s - 13))

noncomputable def beta (omega : ℝ) : ℝ :=
  (18 * real.sqrt 22) / 50

theorem find_beta_ratio : 
  let alpha := 20 * beta 0 in
  beta (10) = 9 * real.sqrt 22 / 50 :=
by
  sorry

end find_beta_ratio_l789_789984


namespace questions_in_test_l789_789492

theorem questions_in_test
  (x : ℕ)
  (h_sections : x % 4 = 0)
  (h_correct : 20 < 0.70 * x)
  (h_correct2 : 20 > 0.60 * x) :
  x = 32 := 
by
  sorry

end questions_in_test_l789_789492


namespace dave_earnings_l789_789180

def total_games : Nat := 10
def non_working_games : Nat := 2
def price_per_game : Nat := 4
def working_games : Nat := total_games - non_working_games
def money_earned : Nat := working_games * price_per_game

theorem dave_earnings : money_earned = 32 := by
  sorry

end dave_earnings_l789_789180


namespace tomato_plant_percentage_l789_789026

noncomputable def total_plants : ℤ := 25 + 30 + 45 + 35 + 50
noncomputable def tomato_plants_garden1 : ℚ := 0.2 * 25
noncomputable def tomato_plants_garden2 : ℚ := 0.25 * 30
noncomputable def tomato_plants_garden3 : ℚ := 0.1 * 45
noncomputable def tomato_plants_garden4 : ℚ := (5/7) * 35
noncomputable def tomato_plants_garden5 : ℚ := 0.4 * 50
noncomputable def total_tomato_plants : ℚ := tomato_plants_garden1 + tomato_plants_garden2 + tomato_plants_garden3 + tomato_plants_garden4 + tomato_plants_garden5
noncomputable def percentage_tomato_plants : ℚ := (total_tomato_plants / total_plants) * 100

theorem tomato_plant_percentage : percentage_tomato_plants ≈ 33.51 := 
by 
  sorry

end tomato_plant_percentage_l789_789026


namespace part1_part2_l789_789177

noncomputable theory
open Classical

namespace proof_problem

def f (x a : ℝ) := abs (x - a) + 3 * x

theorem part1 (a : ℝ) (h : a = 1) : 
  {x : ℝ | f x 1 ≥ 3 * x + 2} = {x | x ≥ 3} ∪ {x | x ≤ -1} :=
by
  exact sorry

theorem part2 (a : ℝ) (ha : 0 < a) (h : ∀ x : ℝ, f x a ≤ 0 → x ≤ -1) : 
  a = 2 :=
by
  exact sorry

end proof_problem

end part1_part2_l789_789177


namespace apples_kilos_first_scenario_l789_789742

noncomputable def cost_per_kilo_oranges : ℝ := 29
noncomputable def cost_per_kilo_apples : ℝ := 29
noncomputable def cost_first_scenario : ℝ := 419
noncomputable def cost_second_scenario : ℝ := 488
noncomputable def kilos_oranges_first_scenario : ℝ := 6
noncomputable def kilos_oranges_second_scenario : ℝ := 5
noncomputable def kilos_apples_second_scenario : ℝ := 7

theorem apples_kilos_first_scenario
  (O A : ℝ) 
  (cost1 cost2 : ℝ) 
  (k_oranges1 k_oranges2 k_apples2 : ℝ) 
  (hO : O = 29) (hA : A = 29) 
  (hCost1 : k_oranges1 * O + x * A = cost1) 
  (hCost2 : k_oranges2 * O + k_apples2 * A = cost2) 
  : x = 8 :=
by
  have hO : O = 29 := sorry
  have hA : A = 29 := sorry
  have h1 : k_oranges1 * O + x * A = cost1 := sorry
  have h2 : k_oranges2 * O + k_apples2 * A = cost2 := sorry
  sorry

end apples_kilos_first_scenario_l789_789742


namespace period_of_tan_scaled_l789_789101

theorem period_of_tan_scaled (a : ℝ) (h : a ≠ 0) : 
  (∃ l : ℝ, l > 0 ∧ ∀ x : ℝ, tan(x / a) = tan((x + l) / a)) ↔ 
  a = 1/3 → (∃ l : ℝ, l = 3 * π) := 
sorry

end period_of_tan_scaled_l789_789101


namespace s_l789_789139

def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def total_money_spent : ℝ := 15.0

theorem s'mores_per_scout :
  (total_money_spent / cost_per_chocolate_bar * sections_per_chocolate_bar) / scouts = 2 :=
by
  sorry

end s_l789_789139


namespace fraction_of_x_by_110_l789_789572

theorem fraction_of_x_by_110 (x : ℝ) (f : ℝ) (h1 : 0.6 * x = f * x + 110) (h2 : x = 412.5) : f = 1 / 3 :=
by 
  sorry

end fraction_of_x_by_110_l789_789572


namespace problem_statement_l789_789329

noncomputable def f (x : ℝ) (A : ℝ) (ϕ : ℝ) : ℝ := A * Real.cos (2 * x + ϕ)

theorem problem_statement {A ϕ : ℝ} (hA : A > 0) (hϕ : |ϕ| < π / 2)
  (h1 : f (-π / 4) A ϕ = 2 * Real.sqrt 2)
  (h2 : f 0 A ϕ = 2 * Real.sqrt 6)
  (h3 : f (π / 12) A ϕ = 2 * Real.sqrt 2)
  (h4 : f (π / 4) A ϕ = -2 * Real.sqrt 2)
  (h5 : f (π / 3) A ϕ = -2 * Real.sqrt 6) :
  ϕ = π / 6 ∧ f (5 * π / 12) A ϕ = -4 * Real.sqrt 2 := 
sorry

end problem_statement_l789_789329


namespace inscribed_square_side_length_l789_789367

noncomputable def side_length (AB EF : ℝ) := 24 * real.sqrt 3 - 22

theorem inscribed_square_side_length
    (h1 : ∀ (A B C D E F : ℝ), (is_equiangular_hexagon A B C D E F))
    (h2 : ∀ (M N P Q : ℝ), (is_square M N P Q))
    (h3 : 50 ∈ AB)
    (h4 : 45 * (real.sqrt 3 - 2) ∈ EF) :
    side_length AB EF = 24 * real.sqrt 3 - 22 :=
by sorry

end inscribed_square_side_length_l789_789367


namespace power_equiv_l789_789503

theorem power_equiv (x_0 : ℝ) (h : x_0 ^ 11 + x_0 ^ 7 + x_0 ^ 3 = 1) : x_0 ^ 4 + x_0 ^ 3 - 1 = x_0 ^ 15 :=
by
  -- the proof goes here
  sorry

end power_equiv_l789_789503


namespace hyperbola_eccentricity_l789_789303

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (h_hyp : a^2 - y^2 = b^2)
variable (hF : (c, 0) ∈ hyperbola(a,b))
variable (hA : (a, 0) ∈ hyperbola(a,b))
variable (hB : (c, b^2 / a) ∈ hyperbola(a,b))
variable (h_slope : (b^2 / a) / (c - a) = 3)

theorem hyperbola_eccentricity (ha : 0 < a) (hb : 0 < b) (hF : (c, 0))
    (hA : (a, 0)) (hB : (c, b^2 / a))
    (h_slope : (b^2 / a) / (c - a) = 3) : (eccentricity(c, a) = 2) := by
  sorry

end hyperbola_eccentricity_l789_789303


namespace intersection_correct_l789_789724

def M : Set Int := {-1, 1, 3, 5}
def N : Set Int := {-3, 1, 5}

theorem intersection_correct : M ∩ N = {1, 5} := 
by 
    sorry

end intersection_correct_l789_789724


namespace maximize_income_at_22_l789_789581

-- Define the conditions
def bed_price (x : ℕ) : Prop := 6 ≤ x ∧ x ≤ 38
def daily_expense : ℕ := 575

-- Define the function y expressed in terms of x based on given conditions
def net_income (x : ℕ) : ℤ :=
  if x ≤ 10 then 100 * x - daily_expense
  else -3 * x^2 + 130 * x - daily_expense

-- Define the maximizing condition: net income must be maximized at bed price 22
theorem maximize_income_at_22 :
  ∃ (x : ℕ), bed_price x ∧ net_income x = 833 :=
begin
  use 22,
  split,
  { exact ⟨by norm_num, by norm_num⟩, },
  { norm_num, sorry },
end

end maximize_income_at_22_l789_789581


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789239

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789239


namespace population_reaches_6000_around_2075_l789_789191
open Real

noncomputable def year_population_6000 : ℕ :=
  let initial_year := 2000
  let initial_population := 200
  let growth_period := 25
  let target_population := 6000
  let growth_factor := 3

  let n := floor (log (target_population / initial_population) / log growth_factor)

  initial_year + (n * growth_period)

theorem population_reaches_6000_around_2075 :
  let y := year_population_6000
  (y = 2075) :=
by 
  sorry

end population_reaches_6000_around_2075_l789_789191


namespace blue_rectangles_form_1x2_l789_789142

-- Definitions of the problem conditions
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (area : ℝ := length * width)

def isCheckerboardPattern (rects : list Rectangle) : Prop := sorry -- defining checkerboard pattern

def isEqualArea (rects1 rects2 : list Rectangle) : Prop :=
  (rects1.map (λ r => r.area)).sum = (rects2.map (λ r => r.area)).sum


-- The problem statement
theorem blue_rectangles_form_1x2 :
  ∃ rects : list Rectangle,
    -- Condition 1: 2 × 2 square divided into several smaller rectangles
    (∃ n m : ℕ, n * m = rects.length) ∧ 

    -- Condition 2: rectangles are painted in yellow and blue in a checkerboard pattern
    isCheckerboardPattern rects ∧ 

    -- Condition 3: total area of blue rectangles equals the total area of yellow rectangles
    (let (blueRects, yellowRects) := rects.partition (λ r => r.color = color.blue) in
    isEqualArea blueRects yellowRects) →

    -- Conclusion: it is possible to arrange the blue rectangles into a 1 × 2 rectangle
    ∃ blueRects : list Rectangle, 
    (blueRects.map (λ r => r.area)).sum = 2 ∧
    existsArrangementInto1x2Rect blueRects := sorry

end blue_rectangles_form_1x2_l789_789142


namespace min_students_in_class_l789_789758

theorem min_students_in_class (n : ℤ) (g : ℤ) : (0.25 < g.to_float / n.to_float) ∧ (g.to_float / n.to_float < 0.30) → n ≥ 7 :=
by
  sorry

end min_students_in_class_l789_789758


namespace solve_printer_problem_l789_789118

noncomputable def printer_problem : Prop :=
  let rate_A := 10
  let rate_B := rate_A + 8
  let rate_C := rate_B - 4
  let combined_rate := rate_A + rate_B + rate_C
  let total_minutes := 20
  let total_pages := combined_rate * total_minutes
  total_pages = 840

theorem solve_printer_problem : printer_problem :=
by
  sorry

end solve_printer_problem_l789_789118


namespace galaxish_8_letter_words_mod_1000_l789_789365

def a_n : ℕ → ℕ
def b_n : ℕ → ℕ
def c_n : ℕ → ℕ

def initial_conditions : Prop :=
  a_n 3 = 8 ∧ b_n 3 = 0 ∧ c_n 3 = 0

def recurrence_relations : Prop :=
  (∀ n, a_n (n + 1) = 4 * (a_n n + c_n n)) ∧ 
  (∀ n, b_n (n + 1) = a_n n) ∧ 
  (∀ n, c_n (n + 1) = 2 * b_n n)

theorem galaxish_8_letter_words_mod_1000 :
  initial_conditions →
  recurrence_relations →
  ((a_n 8 + b_n 8 + c_n 8) % 1000) = 56 :=
by
  sorry

end galaxish_8_letter_words_mod_1000_l789_789365


namespace find_largest_integer_l789_789252

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789252


namespace probability_no_coinciding_sides_l789_789948

theorem probability_no_coinciding_sides :
  let total_triangles := Nat.choose 10 3
  let unfavorable_outcomes := 60 + 10
  let favorable_outcomes := total_triangles - unfavorable_outcomes
  favorable_outcomes / total_triangles = 5 / 12 := by
  sorry

end probability_no_coinciding_sides_l789_789948


namespace shaded_region_perimeter_area_l789_789376

-- Define the necessary elements based on the problem statement
variables {A C E B : Type} [metric_space A] [metric_space C] [metric_space E] [metric_space B]
variables (dAE : dist A E = 4) (dBE : dist B E = 2) (arc_P1 : Metric.Ball A 4) (arc_P2 : Metric.Ball C 2)

-- State the theorem including the proof goals for perimeter and area
theorem shaded_region_perimeter_area :
  let P := 4 + ((3 * 4 * 2) / 4) + ((3 * 2 * 2) / 4),
  let A := ((3 * 4^2) / 4) + ((3 * 2^2) / 4) - (2 * 4) in
  P = 13 ∧ A = 7 := by
  sorry

end shaded_region_perimeter_area_l789_789376


namespace analytical_expression_of_function_l789_789332

theorem analytical_expression_of_function
  (A ω φ B : Real)
  (f : Real → Real)
  (A_pos : A > 0)
  (ω_pos : ω > 0)
  (phi_bound : |φ| < π / 2)
  (max_val : Π x, f x ≤ 3)
  (min_val : Π x, 1 ≤ f x)
  (period : ∀ x, f x = f (x + π / (2 * ω)))
  (symm_axis : ∀ k ∈ ℤ, f (π / 3) = f (π / 3 + k * (π / ω))) :
  f = λ x, sin (4 * x + π / 6) + 2 := 
sorry

end analytical_expression_of_function_l789_789332


namespace students_like_all_three_l789_789670

variables (N : ℕ) (r : ℚ) (j : ℚ) (o : ℕ) (n : ℕ)

-- Number of students in the class
def num_students := N = 40

-- Fraction of students who like Rock
def fraction_rock := r = 1/4

-- Fraction of students who like Jazz
def fraction_jazz := j = 1/5

-- Number of students who like other genres
def num_other_genres := o = 8

-- Number of students who do not like any of the three genres
def num_no_genres := n = 6

---- Proof theorem
theorem students_like_all_three
  (h1 : num_students N)
  (h2 : fraction_rock r)
  (h3 : fraction_jazz j)
  (h4 : num_other_genres o)
  (h5 : num_no_genres n) :
  ∃ z : ℕ, z = 2 := 
sorry

end students_like_all_three_l789_789670


namespace negation_of_universal_proposition_l789_789744

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_universal_proposition_l789_789744


namespace f_maximum_at_sqrt2_div_2_f_monotonically_increasing_f_less_than_exp_minus_x2_minus_2_l789_789716

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2

theorem f_maximum_at_sqrt2_div_2 : 
  is_max (0, ∞) f (Real.sqrt 2 / 2) := 
sorry

theorem f_monotonically_increasing :
  ∀ {x : ℝ}, (1 / 2 < x) ∧ (x < Real.sqrt 2 / 2) → monotone_on f (1 / 2, Real.sqrt 2 / 2) :=
sorry

theorem f_less_than_exp_minus_x2_minus_2 :
  ∀ x > 0, f(x) < Real.exp x - x^2 - 2 :=
sorry

end f_maximum_at_sqrt2_div_2_f_monotonically_increasing_f_less_than_exp_minus_x2_minus_2_l789_789716


namespace a_n_is_integer_and_positive_a_n_a_n1_minus_1_is_perfect_square_l789_789484

-- Definitions based on conditions from part a)
noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := (7 * a n + real.sqrt (45 * (a n)^2 - 36)) / 2

-- Theorem 1: For any n ∈ ℕ, a_n is a positive integer
theorem a_n_is_integer_and_positive (n : ℕ) : ∃ k : ℕ, a n = k ∧ 0 < k :=
by sorry

-- Theorem 2: For any n ∈ ℕ, a_n * a_{n+1} - 1 is a perfect square
theorem a_n_a_n1_minus_1_is_perfect_square (n : ℕ) : ∃ m : ℕ, a n * a (n + 1) - 1 = m ^ 2 :=
by sorry

end a_n_is_integer_and_positive_a_n_a_n1_minus_1_is_perfect_square_l789_789484


namespace bubble_bath_amount_l789_789388

noncomputable def total_bubble_bath_needed 
  (couple_rooms : ℕ) (single_rooms : ℕ) (people_per_couple_room : ℕ) (people_per_single_room : ℕ) (ml_per_bath : ℕ) : ℕ :=
  couple_rooms * people_per_couple_room * ml_per_bath + single_rooms * people_per_single_room * ml_per_bath

theorem bubble_bath_amount :
  total_bubble_bath_needed 13 14 2 1 10 = 400 := by 
  sorry

end bubble_bath_amount_l789_789388


namespace probability_odd_sum_of_six_selected_primes_l789_789038

open Finset

def firstTwelvePrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

theorem probability_odd_sum_of_six_selected_primes :
  let selections := (firstTwelvePrimes.choose 6)
  let odd_sum (s : Finset ℕ) := (s.sum id) % 2 = 1
  let total_ways := selections.card
  let odd_ways := (selections.filter odd_sum).card
  total_ways > 0 -> (odd_ways / total_ways : ℚ) = 1 / 2 :=
by 
  sorry

end probability_odd_sum_of_six_selected_primes_l789_789038


namespace correct_region_l789_789564

-- Define the condition for x > 1
def condition_x_gt_1 (x : ℝ) (y : ℝ) : Prop :=
  x > 1 → y^2 > x

-- Define the condition for 0 < x < 1
def condition_0_lt_x_lt_1 (x : ℝ) (y : ℝ) : Prop :=
  0 < x ∧ x < 1 → 0 < y^2 ∧ y^2 < x

-- Formal statement to check the correct region
theorem correct_region (x y : ℝ) : 
  (condition_x_gt_1 x y ∨ condition_0_lt_x_lt_1 x y) →
  y^2 > x ∨ (0 < y^2 ∧ y^2 < x) :=
sorry

end correct_region_l789_789564


namespace largest_integer_remainder_condition_l789_789259

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789259


namespace min_value_expression_l789_789805

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 48) :
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2 ≥ 144 :=
sorry

end min_value_expression_l789_789805


namespace fraction_not_simplifiable_l789_789438

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_not_simplifiable_l789_789438


namespace can_sides_be_consecutive_ints_l789_789777

theorem can_sides_be_consecutive_ints :
  ∃ (sequence : Fin 33 → Fin 34), 
    (∀ i, sequence i ∈ Finset.range 34) ∧ 
    (∀ s, ∃ k, k ∈ Finset.range 34 ∧ 
    (s = sequence s.fst + sequence s.snd ∧ 
    (s ∈ Finset.range 34))) :=
sorry

end can_sides_be_consecutive_ints_l789_789777


namespace solution_set_of_inequality_l789_789275

theorem solution_set_of_inequality (x : ℝ) : (2 * x + 3) * (4 - x) > 0 ↔ -3 / 2 < x ∧ x < 4 :=
by
  sorry

end solution_set_of_inequality_l789_789275


namespace dihedral_angle_cosine_distance_point_plane_l789_789165

-- Definitions for the rhombus, points, angles, and distances
structure Rhombus where
  A B C D : ℝ^3
  side_length : ℝ
  angle_BAD : ℝ

structure RightPrism where
  base1 base2 : Triangle
  height1 height2 : ℝ

structure Triangle where
  A B C : ℝ^3

def angleAPB (A P B : ℝ^3) : Prop := (∠APB = 90)

-- Propositions to prove
noncomputable def cosine_dihedral_angle (r : Rhombus) (p : RightPrism) : ℝ :=
  (cos (dihedral_angle P BD Q))

noncomputable def distance_PQBD (P : ℝ^3) (QBD : Triangle) : ℝ :=
  (distance P (plane QBD))

-- Theorem to verify the cosine of dihedral angle == 1/3
theorem dihedral_angle_cosine {A B C D P Q : ℝ^3} (r : Rhombus) (p : RightPrism)
  (h1 : r.side_length = 1) (h2 : r.angle_BAD = 60) (h3 : angleAPB A P B) :
  cosine_dihedral_angle r p = 1/3 :=
sorry

-- Theorem to verify the distance from point P to plane QBD == sqrt(2)/3
theorem distance_point_plane {A B C D P Q : ℝ^3} (r : Rhombus) (p : RightPrism)
  (h1 : r.side_length = 1) (h2 : r.angle_BAD = 60) (h3 : angleAPB A P B) :
  distance_PQBD P (Triangle.mk Q B D) = sqrt(2)/3 :=
sorry

end dihedral_angle_cosine_distance_point_plane_l789_789165


namespace rice_mixture_ratio_l789_789780

theorem rice_mixture_ratio (x y : ℝ) (h1 : 7 * x + 8.75 * y = 7.50 * (x + y)) : x / y = 2.5 :=
by
  sorry

end rice_mixture_ratio_l789_789780


namespace slices_leftover_l789_789958

def total_slices (small_pizzas large_pizzas : ℕ) : ℕ :=
  (3 * 4) + (2 * 8)

def slices_eaten_by_people (george bob susie bill fred mark : ℕ) : ℕ :=
  george + bob + susie + bill + fred + mark

theorem slices_leftover :
  total_slices 3 2 - slices_eaten_by_people 3 4 2 3 3 3 = 10 :=
by sorry

end slices_leftover_l789_789958


namespace keith_spent_on_cards_l789_789793

theorem keith_spent_on_cards :
  let digimon_card_cost := 4.45
  let num_digimon_packs := 4
  let baseball_card_cost := 6.06
  let total_spent := num_digimon_packs * digimon_card_cost + baseball_card_cost
  total_spent = 23.86 :=
by
  sorry

end keith_spent_on_cards_l789_789793


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789513

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789513


namespace equation_has_one_solution_l789_789185

theorem equation_has_one_solution : ∀ x : ℝ, x - 6 / (x - 2) = 4 - 6 / (x - 2) ↔ x = 4 :=
by {
  -- proof goes here
  sorry
}

end equation_has_one_solution_l789_789185


namespace determine_m_l789_789995

theorem determine_m {a b c d : ℝ} (a_ne_zero : a ≠ 0) (3a_ne_2b : 3 * a - 2 * b ≠ 0) :
  (∃ m : ℝ, (∀ x : ℝ, (x^2 - 2 * b * x + d) * (m + 2) = (3 * a * x - 4 * c) * (m - 2) 
  → (x ≠ 0 → -x satisfies this equation)) → m = 4 * b / (3 * a - 2 * b)) :=
sorry

end determine_m_l789_789995


namespace bankers_gain_l789_789451

-- Definitions of given conditions
def present_worth : ℝ := 600
def rate_of_interest : ℝ := 0.10
def time_period : ℕ := 2

-- Statement of the problem to be proved: The banker's gain is 126
theorem bankers_gain 
  (PW : ℝ := present_worth) 
  (r : ℝ := rate_of_interest) 
  (n : ℕ := time_period) :
  let A := PW * (1 + r) ^ n in 
  let BG := A - PW in 
  BG = 126 := 
by 
  sorry

end bankers_gain_l789_789451


namespace area_fraction_l789_789472

noncomputable theory
open_locale classical

-- Define the octagon and midpoints structure
structure RegularOctagon (α : Type) [OrderedField α] :=
  (vertices : Fin 8 → EuclideanSpace α (Fin 2))
  (regular : ∀ i j : Fin 8, ∥vertices i - vertices j∥ = ∥vertices 0 - vertices 1∥)

-- Define the midpoints formation of the octagon
def Midpoints (α : Type) [OrderedField α] (O : RegularOctagon α) : Fin 8 → EuclideanSpace α (Fin 2) :=
  λ i, (O.vertices i + O.vertices ((i + 1) % 8)) / 2

-- The areas of octagons (larger and smaller)
def area (α : Type) [OrderedField α] (O : RegularOctagon α) : α := sorry
def area_smaller(α : Type) [OrderedField α] (O : RegularOctagon α) (midpoints : Fin 8 → EuclideanSpace α (Fin 2)) : α := sorry

-- The final theorem
theorem area_fraction (α : Type) [OrderedField α] (O : RegularOctagon α) (M : Fin 8 → EuclideanSpace α (Fin 2) := Midpoints α O) :
  area_smaller α O M = area α O / 4 :=
sorry

end area_fraction_l789_789472


namespace hyperbola_eccentricity_proof_l789_789300

noncomputable def hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : ℝ :=
2

theorem hyperbola_eccentricity_proof 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : hyperbola_eccentricity a b h1 h2 C F A B h3 = 2 :=
sorry

end hyperbola_eccentricity_proof_l789_789300


namespace largest_int_less_than_100_by_7_l789_789231

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789231


namespace eggs_per_box_l789_789969

theorem eggs_per_box (hens : ℕ) (eggs_per_hen_per_day : ℕ) (days_in_week : ℕ) (boxes_per_week : ℕ) (total_eggs : ℕ) (eggs_per_box : ℕ) :
  hens = 270 →
  eggs_per_hen_per_day = 1 →
  days_in_week = 7 →
  boxes_per_week = 315 →
  total_eggs = hens * eggs_per_hen_per_day * days_in_week →
  eggs_per_box = total_eggs / boxes_per_week →
  eggs_per_box = 6 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw [h1, h2, h3, h4] at *,
  have h7 : total_eggs = 270 * 1 * 7, from h5,
  rw [mul_one, mul_comm] at h7,
  replace h7 : total_eggs = 1890 := h7,
  rw [h7] at h6,
  have h8 : 1890 / 315 = 6, from by norm_num,
  rw h8 at h6,
  exact h6
}

end eggs_per_box_l789_789969


namespace sequence_contains_perfect_square_l789_789183

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then
    nat.find_greatest prime (λ p, p ≤ n ∧ p ∣ n)
  else
    n

noncomputable def seq (a1 : ℕ) (h : a1 > 1) : ℕ → ℕ
| 1     := a1
| (n+1) := seq n + largest_prime_factor (seq n)

theorem sequence_contains_perfect_square (a1 : ℕ) (h : a1 > 1) :
  ∃ n, ∃ k, seq a1 h n = k * k :=
sorry

end sequence_contains_perfect_square_l789_789183


namespace route_numbers_351_l789_789046

theorem route_numbers_351 (displayed : ℕ) (up_to_two_segments_broken : Prop) : 
  displayed = 351 → 
  (up_to_two_segments_broken → 
    {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991} = 
    {n : ℕ | ∃ route : ℕ, route = 351 ∧ up_to_two_segments_broken}) :=
by 
  intros h1 h2
  apply set.ext
  intro n
  split
  -- Direction 1: n in the left set implies n in the right set
  { intro hn
    cases hn
    -- Prove for each element in the set
    exact ⟨351, rfl, h2⟩
    -- Repeat this for each number in the set or use automation if known
  }
  -- Direction 2: n in the right set implies n in the left set
  { rintro ⟨route, hr1, hr2⟩
    rw hr1 at hr2
    -- Prove that each possible route number is in the set on the right
    exact hn
  }
  sorry

end route_numbers_351_l789_789046


namespace max_value_of_perfect_sequence_l789_789954

def isPerfectSequence (c : ℕ → ℕ) : Prop := ∀ n m : ℕ, 1 ≤ m ∧ m ≤ (Finset.range (n + 1)).sum (fun k => c k) → 
  ∃ (a : ℕ → ℕ), m = (Finset.range (n + 1)).sum (fun k => c k / a k)

theorem max_value_of_perfect_sequence (n : ℕ) : 
  ∃ c : ℕ → ℕ, isPerfectSequence c ∧
    (∀ i, i ≤ n → c i ≤ if i = 1 then 2 else 4 * 3^(i - 2)) ∧
    c n = if n = 1 then 2 else 4 * 3^(n - 2) :=
by
  sorry

end max_value_of_perfect_sequence_l789_789954


namespace find_point_coordinates_l789_789824

theorem find_point_coordinates (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so x < 0
  (h2 : P.2 > 0) -- Point P is in the second quadrant, so y > 0
  (h3 : abs P.2 = 4) -- distance from P to x-axis is 4
  (h4 : abs P.1 = 5) -- distance from P to y-axis is 5
  : P = (-5, 4) :=
by {
  -- point P is in the second quadrant, so x < 0 and y > 0
  -- |y| = 4 -> y = 4 
  -- |x| = 5 -> x = -5
  sorry
}

end find_point_coordinates_l789_789824


namespace number_divisible_by_5_l789_789044

theorem number_divisible_by_5 (A B C : ℕ) :
  (∃ (k1 k2 k3 k4 k5 k6 : ℕ), 3*10^6 + 10^5 + 7*10^4 + A*10^3 + B*10^2 + 4*10 + C = k1 ∧ 5 * k1 = 0 ∧
                          5 * k2 + 10 = 5 * k2 ∧ 5 * k3 + 5 = 5 * k3 ∧ 
                          5 * k4 + 3 = 5 * k4 ∧ 5 * k5 + 1 = 5 * k5 ∧ 
                          5 * k6 + 7 = 5 * k6) → C = 5 :=
by
  sorry

end number_divisible_by_5_l789_789044


namespace find_number_l789_789113

-- Define the problem condition
def condition (x : ℕ) : Prop := 4 * x + 7 * x = 55

-- Define the statement to prove
theorem find_number : ∃ x : ℕ, condition x ∧ x = 5 :=
by
  existsi 5
  simp [condition]
  exact congr_arg (+ 0) rfl

end find_number_l789_789113


namespace cost_plan1_cost_plan2_plan1_more_effective_for_x_eq_30_l789_789937

variable (x : ℕ)

def suit_price := 400
def tie_price := 80
def num_suits := 20
def x_cond := x > 20

def plan1_cost (x : ℕ) : ℕ := suit_price * num_suits + (x - num_suits) * tie_price
def plan2_cost (x : ℕ) : ℕ := 0.9 * suit_price * num_suits + 0.9 * tie_price * x

theorem cost_plan1 (hx : x_cond) : plan1_cost x = 80 * x + 6400 := by
  sorry

theorem cost_plan2 (hx : x_cond) : plan2_cost x = 72 * x + 7200 := by
  sorry

theorem plan1_more_effective_for_x_eq_30 : (plan1_cost 30 < plan2_cost 30) := by
  sorry

end cost_plan1_cost_plan2_plan1_more_effective_for_x_eq_30_l789_789937


namespace problem_statement_l789_789815

open Set Real

theorem problem_statement :
  let I := univ : Set ℝ
  let A := {y | ∃ x, y = log 2 x ∧ x > 2}
  let B := {y | y ≥ 1}
  A ⊆ B :=
by
  sorry

end problem_statement_l789_789815


namespace sum_of_squares_of_segments_eq_seven_l789_789055

def Circle := { center : Point, radius : ℝ }
def Point := (ℝ × ℝ)
def Line := (Point × Point)

variables (center : Point) (radius : ℝ)

def cross_intersects_at := (cross_center : Point) (d : ℝ) := 
  dist center cross_center = d

noncomputable def sum_of_squares_of_segments (circle : Circle) (cross_center : Point) : ℝ := sorry

theorem sum_of_squares_of_segments_eq_seven :
  ∀ (center : Point) (cross_center : Point),
    radius = 1 ∧ cross_intersects_at cross_center 0.5 →
    sum_of_squares_of_segments ⟨center, radius⟩ cross_center = 7 := sorry

end sum_of_squares_of_segments_eq_seven_l789_789055


namespace hyperbola_eccentricity_l789_789304

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
variable (h_hyp : a^2 - y^2 = b^2)
variable (hF : (c, 0) ∈ hyperbola(a,b))
variable (hA : (a, 0) ∈ hyperbola(a,b))
variable (hB : (c, b^2 / a) ∈ hyperbola(a,b))
variable (h_slope : (b^2 / a) / (c - a) = 3)

theorem hyperbola_eccentricity (ha : 0 < a) (hb : 0 < b) (hF : (c, 0))
    (hA : (a, 0)) (hB : (c, b^2 / a))
    (h_slope : (b^2 / a) / (c - a) = 3) : (eccentricity(c, a) = 2) := by
  sorry

end hyperbola_eccentricity_l789_789304


namespace combined_weight_l789_789608

-- Definition of conditions
def regular_dinosaur_weight := 800
def five_regular_dinosaurs_weight := 5 * regular_dinosaur_weight
def barney_weight := five_regular_dinosaurs_weight + 1500

-- Statement to prove
theorem combined_weight (h1: five_regular_dinosaurs_weight = 5 * regular_dinosaur_weight)
                        (h2: barney_weight = five_regular_dinosaurs_weight + 1500) : 
        (barney_weight + five_regular_dinosaurs_weight = 9500) :=
by
    sorry

end combined_weight_l789_789608


namespace fourth_and_fifth_suppliers_cars_equal_l789_789599

-- Define the conditions
def total_cars : ℕ := 5650000
def cars_supplier_1 : ℕ := 1000000
def cars_supplier_2 : ℕ := cars_supplier_1 + 500000
def cars_supplier_3 : ℕ := cars_supplier_1 + cars_supplier_2
def cars_distributed_first_three : ℕ := cars_supplier_1 + cars_supplier_2 + cars_supplier_3
def cars_remaining : ℕ := total_cars - cars_distributed_first_three

-- Theorem stating the question and answer
theorem fourth_and_fifth_suppliers_cars_equal 
  : (cars_remaining / 2) = 325000 := by
  sorry

end fourth_and_fifth_suppliers_cars_equal_l789_789599


namespace average_after_discard_l789_789558

theorem average_after_discard (avg1 : ℕ) (sum_discarded : ℕ) (n : ℕ) (discard_count : ℕ) (new_avg : ℚ) :
  avg1 = 56 →
  sum_discarded = 45 + 55 →
  n = 50 →
  discard_count = 2 →
  new_avg = (avg1 * n - sum_discarded) / (n - discard_count) →
  new_avg = 56.25 :=
by 
  intros h1 h2 h3 h4 h5
  rw [h5]
  linarith
  sorry

end average_after_discard_l789_789558


namespace volume_intersection_l789_789083

-- Define the conditions
def edge_length := 12
def base_length := edge_length
def height := Real.sqrt 108
def intersection_height := height / 2
def base_area := (base_length / 2) ^ 2

-- Define the volume calculation of a single pyramid given base area and height
def volume_pyramid (area : ℝ) (h : ℝ) := (1 / 3) * area * h

-- Calculate the total volume of the intersection
def total_volume_intersection := 2 * volume_pyramid base_area (intersection_height)

-- Proving the volume of the intersection
theorem volume_intersection : total_volume_intersection = 72 := by
  sorry

end volume_intersection_l789_789083


namespace math_problem_l789_789789

theorem math_problem :
  let result := 83 - 29
  let final_sum := result + 58
  let rounded := if final_sum % 10 < 5 then final_sum - final_sum % 10 else final_sum + (10 - final_sum % 10)
  rounded = 110 := by
  sorry

end math_problem_l789_789789


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789215

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789215


namespace probability_of_two_red_balls_l789_789574

-- Define the total number of balls, number of red balls, and number of white balls
def total_balls := 6
def red_balls := 4
def white_balls := 2
def drawn_balls := 2

-- Define the combination formula
def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to choose 2 red balls from 4
def ways_to_choose_red := choose 4 2

-- The number of ways to choose any 2 balls from the total of 6
def ways_to_choose_any := choose 6 2

-- The corresponding probability
def probability := ways_to_choose_red / ways_to_choose_any

-- The theorem we want to prove
theorem probability_of_two_red_balls :
  probability = 2 / 5 :=
by
  sorry

end probability_of_two_red_balls_l789_789574


namespace area_ratio_of_midpoints_of_regular_octagon_l789_789474

-- Define the regular octagon and its properties

section
variables (A B C D E F G H A' B' C' D' E' F' G' H' : Type) [regOct_X : RegularOctagon A B C D E F G H]
[RegularOctagonMidpoints : RegularOctagonMidpoints A B C D E F G H A' B' C' D' E' F' G' H']

-- Statement: The area ratio between the smaller and the larger octagon is 1/4
theorem area_ratio_of_midpoints_of_regular_octagon
  (h_midpoints : MidpointsOfSidesOfRegularOctagon A B C D E F G H A' B' C' D' E' F' G' H') :
  AreaOfOctagon A' B' C' D' E' F' G' H' = (1/4 : ℚ) * AreaOfOctagon A B C D E F G H :=
  sorry
end

end area_ratio_of_midpoints_of_regular_octagon_l789_789474


namespace math_problem_l789_789348

theorem math_problem (a b c d x : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |x| = 2) :
  x^4 + c * d * x^2 - a - b = 20 :=
sorry

end math_problem_l789_789348


namespace find_first_number_l789_789856

theorem find_first_number (y x : ℤ) (h1 : (y + 76 + x) / 3 = 5) (h2 : x = -63) : y = 2 :=
by
  -- To be filled in with the proof steps
  sorry

end find_first_number_l789_789856


namespace number_of_possible_integer_multisets_l789_789457

noncomputable def polynomial_p (a : Fin 7 → ℤ) (x : ℤ) : ℤ :=
a 6 * x^6 + a 5 * x^5 + a 4 * x^4 + a 3 * x^3 + a 2 * x^2 + a 1 * x + a 0

noncomputable def polynomial_q (a : Fin 7 → ℤ) (x : ℤ) : ℤ :=
a 0 * x^6 + a 1 * x^5 + a 2 * x^4 + a 3 * x^3 + a 4 * x^2 + a 5 * x + a 6

theorem number_of_possible_integer_multisets 
{a : Fin 7 → ℤ} 
(hp : ∀ s : ℤ, polynomial_p a s = 0 → s = 1 ∨ s = -1)
(hq : ∀ s : ℤ, polynomial_q a s = 0 → s = 1 ∨ s = -1):
  (∃ S : Multiset ℤ, S.card = 6 ∧ ∀ si : ℤ, si ∈ S → (si = 1 ∨ si = -1)) → 
  ∃ M : Finset (Multiset ℤ), M.card = 7 := 
sorry

end number_of_possible_integer_multisets_l789_789457


namespace simplified_evaluated_expression_l789_789841

noncomputable def a : ℚ := 1 / 3
noncomputable def b : ℚ := 1 / 2
noncomputable def c : ℚ := 1

def expression (a b c : ℚ) : ℚ := a^2 + 2 * b - c

theorem simplified_evaluated_expression :
  expression a b c = 1 / 9 :=
by
  sorry

end simplified_evaluated_expression_l789_789841


namespace bird_migration_difference_correct_l789_789552

def bird_migration_difference : ℕ := 54

/--
There are 250 bird families consisting of 3 different bird species, each with varying migration patterns.

Species A: 100 bird families; 35% fly to Africa, 65% fly to Asia
Species B: 120 bird families; 50% fly to Africa, 50% fly to Asia
Species C: 30 bird families; 10% fly to Africa, 90% fly to Asia

Prove that the difference in the number of bird families migrating to Asia and Africa is 54.
-/
theorem bird_migration_difference_correct (A_Africa_percent : ℕ := 35) (A_Asia_percent : ℕ := 65)
  (B_Africa_percent : ℕ := 50) (B_Asia_percent : ℕ := 50)
  (C_Africa_percent : ℕ := 10) (C_Asia_percent : ℕ := 90)
  (A_count : ℕ := 100) (B_count : ℕ := 120) (C_count : ℕ := 30) :
    bird_migration_difference = 
      (A_count * A_Asia_percent / 100 + B_count * B_Asia_percent / 100 + C_count * C_Asia_percent / 100) - 
      (A_count * A_Africa_percent / 100 + B_count * B_Africa_percent / 100 + C_count * C_Africa_percent / 100) :=
by sorry

end bird_migration_difference_correct_l789_789552


namespace line_slope_l789_789155

theorem line_slope (x1 y1 x2 y2 : ℝ) (h1 : x1 = 0) (h2 : y1 = 100) (h3 : x2 = 50) (h4 : y2 = 300) :
  (y2 - y1) / (x2 - x1) = 4 :=
by sorry

end line_slope_l789_789155


namespace find_pc_l789_789368

def convex_quadrilateral (A B C D P : Type) : Prop :=
  ∃ (CD AC BD AB AD : ℝ),
  (CD = 72) ∧
  (AB = 35) ∧
  (AP = 15) ∧
  (CD ⊥ AC) ∧
  (AB ⊥ BD) ∧
  (P ∈ line_through B ∧ is_perpendicular line_through B AD ∧ P ∈ AC) ∧
  (∃ (PC : ℝ), PC = 72.5)

theorem find_pc {A B C D P : Type} :
  convex_quadrilateral A B C D P → PC = 72.5 :=
by
  sorry

end find_pc_l789_789368


namespace smallest_integer_solution_l789_789324

-- Define the function f
def f (x : ℝ) : ℝ := 2^x + 2 * x - 6

-- Define the root of the function as x0
def x0 : ℝ := sorry -- assume a value for x0 to be used later

-- The theorem we want to state
theorem smallest_integer_solution (h1 : 1 < x0) (h2 : x0 < 2) : 
  let k := 6 in k - 4 > x0 :=
by
  -- Insert proof here
  sorry

end smallest_integer_solution_l789_789324


namespace nobita_dorayakis_correct_doraemon_dorayakis_correct_l789_789186

-- Defining the conditions and the hypothesis
def total_rounds : ℕ := 20
def nobita_received : ℕ := 30

-- Nobita's strategy per 10 rounds
def rounds_scissors_per_10 : ℕ := 1
def rounds_rock_per_10 : ℕ := 9
def doraemon_move : string := "rock"

-- Definition of rounds
def rounds_scissors : ℕ :=
  (total_rounds / 10) * rounds_scissors_per_10
def rounds_rock : ℕ :=
  total_rounds - rounds_scissors

-- Definitions of distribution rules
def win_reward : ℕ := 2
def lose_reward : ℕ := 0
def tie_reward : ℕ := 1

-- Outcomes for Nobita and Doraemon
def nobita_losses : ℕ := rounds_scissors
def nobita_ties : ℕ := rounds_rock
def nobita_dorayakis : ℕ :=
  nobita_ties * tie_reward + nobita_losses * lose_reward

def doraemon_wins : ℕ := rounds_scissors
def doraemon_ties : ℕ := rounds_rock
def doraemon_dorayakis : ℕ :=
  doraemon_wins * win_reward + doraemon_ties * tie_reward

-- Theorem stating that Nobita's dorayakis count matches given condition
theorem nobita_dorayakis_correct :
  nobita_dorayakis = nobita_received := by
  sorry

-- Theorem stating the amount received by Doraemon
theorem doraemon_dorayakis_correct :
  doraemon_dorayakis = 10 := by
  sorry

end nobita_dorayakis_correct_doraemon_dorayakis_correct_l789_789186


namespace circle_equation_l789_789868

theorem circle_equation (x y : ℝ) :
  let center := (0, 4)
  let point_on_circle := (3, 0)
  (x - center.1)^2 + (y - center.2)^2 = 25 :=
by
  sorry

end circle_equation_l789_789868


namespace isosceles_triangle_l789_789385

noncomputable def triangle_is_isosceles (A B C a b c : ℝ) (h_triangle : a = 2 * b * Real.cos C) : Prop :=
  ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)

theorem isosceles_triangle
  (A B C a b c : ℝ)
  (h_sides : a = 2 * b * Real.cos C)
  (h_triangle : ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)) :
  B = C :=
sorry

end isosceles_triangle_l789_789385


namespace problem_part_I_problem_part_II_l789_789687

/-- Part (I): Given sequence {a_n} forming a triangular array where the first number of each row 
     forms an arithmetic sequence {b_n}, S_n represents the sum of the first n terms of {b_n}, 
     b_1 = a_1 = 1, S_5 = 15, each row from the third row forms a geometric sequence with a 
     common ratio r > 0, and a_9 = 16, prove a_{50} = 160. -/
theorem problem_part_I (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) 
  (r : ℝ) (q : ℝ) 
  (h_arithmetic : ∀ n, b (n + 1) - b n = 1)
  (h_S_sum : ∀ n, S n = n * (n + 1) / 2)
  (h_b1 : b 1 = 1) 
  (h_S5 : S 5 = 15)
  (h_r_pos : r > 0)
  (h_a_geometric : ∀ i j, a (i + j * (j + 1) / 2) = b i * q ^ (j - 1))
  (h_a9 : a 9 = 16) :
  a 50 = 160 := 
sorry

/-- Part (II): Let T_n = 1 / S_{n+1} + 1 / S_{n+2} + ... + 1 / S_{2n}. When m ∈ [-1, 1], 
    and t^3 - 2mt - 8/3 > T_n always holds for any n ∈ ℕ*, prove the range of t is 
    (-∞, -3) ∪ (3, ∞). -/
theorem problem_part_II (S : ℕ → ℝ) (T : ℕ → ℝ) (m t : ℝ) 
  (h_S_formula : ∀ n, S n = n * (n + 1) / 2)
  (h_T_formula : ∀ n, T n = ∑ i in finset.range n, 1 / (S (n + i)))
  (h_ineq : ∀ n, t^3 - 2 * m * t - 8 / 3 > T n)
  (h_m_range : -1 ≤ m ∧ m ≤ 1) :
  t < -3 ∨ t > 3 := 
sorry

end problem_part_I_problem_part_II_l789_789687


namespace largest_integer_remainder_condition_l789_789257

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l789_789257


namespace standard_eqn_of_ellipse_l789_789290

def is_min_axis_endpoint (C : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
C x y ∧ ∃ b, ∀ k, C k b → b > 0

def is_focus (C : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
C x y ∧ ∃ a b c, a > b ∧ b > 0 ∧ c = real.sqrt (a^2 - b^2)

def circle (x y : ℝ) : Prop :=
x^2 + y^2 = 4

def ellipse (a b : ℝ) : ℝ → ℝ → Prop :=
λ x y, (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_eqn_of_ellipse (a b : ℝ) :
(∃ b c, b = 2 ∧ c = 2 ∧ a^2 = b^2 + c^2 ∧ ellipse a b = ellipse 2 (real.sqrt 8)) →
ellipse a b = ellipse (real.sqrt 8) 2 :=
by
  assume h,
  sorry

end standard_eqn_of_ellipse_l789_789290


namespace probability_two_positive_roots_probability_no_real_roots_l789_789721

-- Definitions and conditions
def quadratic_eq (a b x : ℝ) : ℝ := x^2 - 2 * (a - 2) * x - b^2 + 16

-- Defining the ranges for a and b based on dice rolls
def valid_a (a : ℝ) : Prop := a ∈ {2, 3, 4, 5, 6}
def valid_b (b : ℝ) : Prop := b ∈ {1, 2, 3, 4, 5, 6}

-- Probability of the quadratic equation having two positive roots
theorem probability_two_positive_roots : ∀ (a b : ℝ),
  valid_a a → valid_b b →
  ( (a > 2) ∧ (-4 < b ∧ b < 4) ∧ ((a - 2)^2 + b^2 ≥ 16) ) → 
  36 * ∃! (a, b) ∈ {(6, 1), (6, 2), (6, 3), (5, 3)} := sorry

-- Probability of the quadratic equation having no real roots
theorem probability_no_real_roots : ∀ (a b : ℝ),
  2 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 4 →
  ((a - 2)^2 + b^2 < 16) → 
  16 * (4 / (1/4 * π * 4^2)) = π / 4 := sorry

end probability_two_positive_roots_probability_no_real_roots_l789_789721


namespace probability_two_green_balls_picked_l789_789144

theorem probability_two_green_balls_picked :
  let total_balls := 12 in
  let green_balls := 5 in
  let yellow_balls := 3 in
  let blue_balls := 4 in
  let chosen_balls := 4 in
  let combinations (n k : ℕ) := Nat.choose n k in
  let total_ways := combinations total_balls chosen_balls in
  let ways_to_pick_two_green := combinations green_balls 2 in
  let ways_to_pick_two_remaining := combinations (total_balls - green_balls) (chosen_balls - 2) in
  let successful_outcomes := ways_to_pick_two_green * ways_to_pick_two_remaining in
  let probability := successful_outcomes / total_ways in
  probability = (42 : ℚ) / 99 :=  
by
  sorry

end probability_two_green_balls_picked_l789_789144


namespace area_of_shaded_region_l789_789905

theorem area_of_shaded_region (r₁ r₂ : ℝ) (r₁_pos : 0 < r₁) (r₂_pos : 0 < r₂) (htangent : r₁ = 4 ∧ r₂ = 5) : 
  let R := r₁ + r₂ in
  let area_large_circle := π * R^2 in
  let area_small_circle₁ := π * r₁^2 in
  let area_small_circle₂ := π * r₂^2 in
  area_large_circle - (area_small_circle₁ + area_small_circle₂) = 40 * π := 
by {
  -- Radius of the larger circle
  have R_eq : R = 9, from calc
    R = r₁ + r₂ : by rw [htangent]; linarith
    ... = 9 : by norm_num,

  -- Compute the area of the large circle
  have area_large_circle_eq : area_large_circle = 81 * π, from calc
    area_large_circle = π * R ^ 2 : rfl
    ... = π * 9 ^ 2 : by rw [R_eq]
    ... = 81 * π : by norm_num,

  -- Compute the area of the first smaller circle
  have area_small_circle₁_eq : area_small_circle₁ = 16 * π, from calc
    area_small_circle₁ = π * r₁ ^ 2 : rfl
    ... = π * 4 ^ 2 : by { rw [htangent], norm_num }
    ... = 16 * π : by norm_num,

  -- Compute the area of the second smaller circle
  have area_small_circle₂_eq : area_small_circle₂ = 25 * π, from calc
    area_small_circle₂ = π * r₂ ^ 2 : rfl
    ... = π * 5 ^ 2 : by { rw [htangent], norm_num }
    ... = 25 * π : by norm_num,

  -- Prove the final area of the shaded region
  calc
    area_large_circle - (area_small_circle₁ + area_small_circle₂)
      = (81 * π) - (16 * π + 25 * π) : by rw [area_large_circle_eq, area_small_circle₁_eq, area_small_circle₂_eq]
    ... = 81 * π - 41 * π : by norm_num
    ... = 40 * π : by norm_num,
}

end area_of_shaded_region_l789_789905


namespace largest_int_less_than_100_by_7_l789_789229

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789229


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l789_789532

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l789_789532


namespace view_angle_sq_l789_789782

theorem view_angle_sq (A B C D L : Type) 
  [square A B C D] 
  [isosceles_triangle A B L]
  (L_angle_AB : angle L A B = 15) 
  (L_angle_BA : angle L B A = 15) : 
  angle L view C D = 110 :=
sorry

end view_angle_sq_l789_789782


namespace utility_bills_total_correct_l789_789017

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l789_789017


namespace cube_face_min_sum_l789_789021

open Set

theorem cube_face_min_sum (S : Finset ℕ)
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8})
  (h_faces_sum : ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → 
                    (a + b + c >= 10) ∨ 
                    (a + b + d >= 10) ∨ 
                    (a + c + d >= 10) ∨ 
                    (b + c + d >= 10)) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 16 :=
sorry

end cube_face_min_sum_l789_789021


namespace greatest_int_with_gcd_five_l789_789539

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l789_789539


namespace hyperbola_eccentricity_l789_789295

variable {a b c : ℝ} (h_a : a > 0) (h_b : b > 0)
variable (C : Set (ℝ × ℝ)) (h_C : ∀ (x y : ℝ), (x, y) ∈ C ↔ (ℝ × ℝ) := {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
variable (A : ℝ × ℝ := (a, 0))
variable (F : ℝ × ℝ := (c, 0))
variable (B : ℝ × ℝ := (c, b^2 / a))
variable (h_slope : (b^2 / a - 0) / (c - a) = 3)

theorem hyperbola_eccentricity (h_b_square : b^2 = c^2 - a^2) : c / a = 2 :=
by
  sorry

end hyperbola_eccentricity_l789_789295


namespace largest_multiple_of_11_neg_greater_minus_210_l789_789099

theorem largest_multiple_of_11_neg_greater_minus_210 :
  ∃ (x : ℤ), x % 11 = 0 ∧ -x < -210 ∧ ∀ y, y % 11 = 0 ∧ -y < -210 → y ≤ x :=
sorry

end largest_multiple_of_11_neg_greater_minus_210_l789_789099


namespace probability_of_obtaining_face_F_l789_789870

noncomputable def biased_die_probability (F : ℕ) : ℚ := 
  let p := (1/12 : ℚ) in
  let x := (1/16 : ℚ) in
  p + x 

theorem probability_of_obtaining_face_F :
  let p := (1/12 : ℚ) in
  let x := (1/16 : ℚ) in
  biased_die_probability F = (7/48 : ℚ) := 
by
  sorry

end probability_of_obtaining_face_F_l789_789870


namespace chickens_increased_l789_789899

-- Definitions and conditions
def initial_chickens := 45
def chickens_bought_day1 := 18
def chickens_bought_day2 := 12
def total_chickens_bought := chickens_bought_day1 + chickens_bought_day2

-- Proof statement
theorem chickens_increased :
  total_chickens_bought = 30 :=
by
  sorry

end chickens_increased_l789_789899


namespace hyperbola_eccentricity_proof_l789_789301

noncomputable def hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : ℝ :=
2

theorem hyperbola_eccentricity_proof 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0))
  (A : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (real.sqrt (a^2 + b^2), b^2 / a))
  (h3: (B.snd - A.snd) / (B.fst - A.fst) = 3)
  : hyperbola_eccentricity a b h1 h2 C F A B h3 = 2 :=
sorry

end hyperbola_eccentricity_proof_l789_789301


namespace problem_part1_problem_part2_l789_789698

-- Definitions and conditions
def a_n : ℕ → ℕ
| 0       := 0
| (n + 1) := 2^(n + 2)  -- Given geometric sequence

def S : ℕ → ℕ := λ n, (Finset.range n).sum (λ i, a_n i)

theorem problem_part1 :
  a_n 1 = 8 ∧ (S 3 + 3 * a_n 4 = S 5) → 
  ∀ n, a_n n = 2^(n + 2) :=
by sorry

-- Additional definitions for b_n, c_n, P_n and Q_n
def b_n (n : ℕ) : ℕ := 2 * n + 5

def c_n (n : ℕ) : ℕ := 1 / (b_n n * b_n (n + 1))

def P_n (n : ℕ) : ℕ := n^2 + 6 * n

def Q_n (n : ℕ) : ℕ := n / (14 * n + 49)

theorem problem_part2 :
  ( ∀ n, b_n n = log 2 (a_n n * a_n (n + 1)) ∧
    ∀ n, c_n n = 1 / (b_n n * b_n (n + 1)) ) →
  ∀ n, P_n n = n^2 + 6 * n ∧ Q_n n = n / (14 * n + 49) :=
by sorry

end problem_part1_problem_part2_l789_789698


namespace solve_equation_l789_789447

theorem solve_equation (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end solve_equation_l789_789447


namespace nancy_hourly_wage_l789_789020

-- Define the costs and financial aid conditions
def tuition : ℝ := 22000
def housing : ℝ := 6000
def meal_plan : ℝ := 2500
def textbooks : ℝ := 800
def parents_contribution : ℝ := tuition / 2
def merit_scholarship : ℝ := 3000
def need_based_scholarship : ℝ := 1500
def student_loan : ℝ := 2 * merit_scholarship
def total_work_hours : ℝ := 200

-- Define the total cost
def total_cost : ℝ := tuition + housing + meal_plan + textbooks

-- Define the total financial support
def total_financial_support : ℝ :=
  parents_contribution + merit_scholarship + need_based_scholarship + student_loan

-- Define the remaining expenses
def remaining_expenses : ℝ := total_cost - total_financial_support

-- Define the hourly wage needed
def hourly_wage_needed : ℝ := remaining_expenses / total_work_hours

-- The proof problem statement
theorem nancy_hourly_wage : hourly_wage_needed = 49 := by
  sorry

end nancy_hourly_wage_l789_789020


namespace arithmetic_sequence_fifth_term_l789_789370

theorem arithmetic_sequence_fifth_term :
  ∀ (a₁ d n : ℕ), a₁ = 3 → d = 4 → n = 5 → a₁ + (n - 1) * d = 19 :=
by
  intros a₁ d n ha₁ hd hn
  sorry

end arithmetic_sequence_fifth_term_l789_789370


namespace theo_eats_cookies_x_times_a_day_l789_789073

variable (x : ℕ)

variables (cookies_per_time day_in_month total_cookies_in_3_months_times_day : ℕ)
variable (days_in_month month_in_3_months : ℕ)

-- Conditions
def cookie_condition_1 : Prop := cookies_per_time = 13
def cookie_condition_2 : Prop := days_in_month = 20
def cookie_condition_3 : Prop := month_in_3_months = 3
def cookie_condition_4 : Prop := total_cookies_in_3_months_times_day = 2340

-- Translation of condition (the number of times he eats cookies per day)
def number_of_times_per_day_theo_eats_cookies := 
  total_cookies_in_3_months_times_day / (days_in_month * month_in_3_months * cookies_per_time)

-- Goal: Theo eats cookies x times a day
theorem theo_eats_cookies_x_times_a_day (h1 : cookie_condition_1)
    (h2 : cookie_condition_2)
    (h3 : cookie_condition_3)
    (h4 : cookie_condition_4) : number_of_times_per_day_theo_eats_cookies = 3 := 
by 
  unfold cookie_condition_1 at h1
  unfold cookie_condition_2 at h2
  unfold cookie_condition_3 at h3
  unfold cookie_condition_4 at h4
  unfold number_of_times_per_day_theo_eats_cookies
  sorry

end theo_eats_cookies_x_times_a_day_l789_789073


namespace smallest_abs_diff_l789_789408

theorem smallest_abs_diff (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - 10 * x + 3 * y = 670) : 
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y - 10 * x + 3 * y = 670 ∧ |x - y| = 16 :=
by 
  have h := x * y - 10 * x + 3 * y = 670
  sorry

end smallest_abs_diff_l789_789408


namespace magnitude_of_sum_l789_789812

-- Defining the real numbers x and y
variable (x y : ℝ)

-- Defining the vectors a, b, and c
def a : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (2, y)
def c : ℝ × ℝ := (-2, 2)

-- Conditions given in the problem
axiom a_perp_c : a x y.1 * c.1 + a x y.2 * c.2 = 0
axiom b_par_c : ∃ k : ℝ, b x y = k • c

-- Prove that the magnitude of a + b is sqrt(10)
theorem magnitude_of_sum :
  let a_plus_b := (a x y).1 + (b x y).1, (a x y).2 + (b x y).2 in
  (a_perp_c x y) -> (b_par_c x y) -> ‖a_plus_b‖ = Real.sqrt 10 :=
by
  sorry

end magnitude_of_sum_l789_789812


namespace fly_travel_distance_proof_l789_789579

def fly_travel_distance (radius : ℝ) (third_leg : ℝ) : ℝ :=
  let diameter := 2 * radius
  let other_leg := Real.sqrt (diameter ^ 2 - third_leg ^ 2)
  diameter + third_leg + other_leg

theorem fly_travel_distance_proof :
  fly_travel_distance 75 70 = 352.6 :=
by
  simp [fly_travel_distance]
  rfl

end fly_travel_distance_proof_l789_789579


namespace trisha_weeks_worked_l789_789079

-- Definitions based on conditions
def hourly_pay : ℝ := 15
def hours_per_week : ℝ := 40
def tax_withholding_rate : ℝ := 0.20
def annual_take_home_pay : ℝ := 24960
def weekly_pay_before_taxes := hourly_pay * hours_per_week
def weekly_withholding := tax_withholding_rate * weekly_pay_before_taxes
def weekly_take_home_pay := weekly_pay_before_taxes - weekly_withholding
def weeks_worked := annual_take_home_pay / weekly_take_home_pay

-- Proof statement to show weeks_worked equals 52
theorem trisha_weeks_worked : weeks_worked = 52 := by
  sorry

end trisha_weeks_worked_l789_789079


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789212

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789212


namespace segment_AB_length_area_triangle_ABF1_l789_789320

-- Given conditions
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def line (x y : ℝ) : Prop := y = x - 1
def focus1 := (-1, 0 : ℝ)

theorem segment_AB_length :
  ∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ line A.1 A.2 ∧ ellipse B.1 B.2 ∧ line B.1 B.2 ∧ 
  dist A B = (4 * real.sqrt 2) / 3 := sorry

theorem area_triangle_ABF1 :
  ∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ line A.1 A.2 ∧ ellipse B.1 B.2 ∧ line B.1 B.2 ∧ 
  area_triangle A B focus1 = 4 / 3 := sorry

end segment_AB_length_area_triangle_ABF1_l789_789320


namespace y_intercept_of_line_b_is_minus_8_l789_789818

/-- Define a line in slope-intercept form y = mx + c --/
structure Line :=
  (m : ℝ)   -- slope
  (c : ℝ)   -- y-intercept

/-- Define a point in 2D Cartesian coordinate system --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define conditions for the problem --/
def line_b_parallel_to (l: Line) (p: Point) : Prop :=
  l.m = 2 ∧ p.x = 3 ∧ p.y = -2

/-- Define the target statement to prove --/
theorem y_intercept_of_line_b_is_minus_8 :
  ∀ (b: Line) (p: Point), line_b_parallel_to b p → b.c = -8 := by
  -- proof goes here
  sorry

end y_intercept_of_line_b_is_minus_8_l789_789818


namespace coeff_x2_l789_789456

-- Define the polynomial functions
def f (x : ℝ) := x^2 + x + 1
def g (x : ℝ) := (1 - x)^4

-- Statement claiming the coefficient of x^2 in the expansion 
-- of the product f(x) * g(x) equals 3
theorem coeff_x2 : coeff (x^2) ((f * g)) = 3 :=
by
  sorry

end coeff_x2_l789_789456


namespace probability_drawing_k_white_balls_l789_789161

def probability_k_white_balls (n k : Nat) : ℚ :=
  (Nat.choose n k)^2 / Nat.choose (2 * n) n

theorem probability_drawing_k_white_balls (n k : ℕ) :
  n white balls in urn →
  n black balls in urn →
  probability_k_white_balls n k = (Nat.choose n k)^2 / (Nat.choose (2 * n) n) :=
by sorry

end probability_drawing_k_white_balls_l789_789161


namespace round_table_seating_l789_789606

-- Define the conditions
variable {n : ℕ}
variable (people : Finset ℕ) (knows : ℕ → ℕ → Prop)

-- Explicit conditions
def conditions (n : ℕ) (people : Finset ℕ) (knows : ℕ → ℕ → Prop) :=
  people.card = 2 * n ∧ ∀ p ∈ people, (Finset.filter (knows p) people).card ≥ n

-- Question to prove
theorem round_table_seating (n : ℕ) (people : Finset ℕ) (knows : ℕ → ℕ → Prop) :
  conditions n people knows →
  ∃ (a b c d : ℕ),
    a ∈ people ∧ b ∈ people ∧ c ∈ people ∧ d ∈ people ∧
    knows a b ∧ knows b c ∧ knows c d ∧ knows d a :=
by {
  intro h,
  sorry
}

end round_table_seating_l789_789606


namespace ellipse_standard_equation_l789_789488

theorem ellipse_standard_equation :
  ∀ (a b c : ℝ), a = 9 → c = 6 → b = Real.sqrt (a^2 - c^2) →
  (b ≠ 0 ∧ a ≠ 0 → (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)) :=
by
  sorry

end ellipse_standard_equation_l789_789488


namespace find_smaller_number_l789_789489

-- Define the conditions
def condition1 (x y : ℤ) : Prop := x + y = 30
def condition2 (x y : ℤ) : Prop := x - y = 10

-- Define the theorem to prove the smaller number is 10
theorem find_smaller_number (x y : ℤ) (h1 : condition1 x y) (h2 : condition2 x y) : y = 10 := 
sorry

end find_smaller_number_l789_789489


namespace arithmetic_sequence_a2_value_l789_789315

theorem arithmetic_sequence_a2_value 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) = a n + 3)
  (h2 : S n = n * (a 1 + a n) / 2)
  (hS13 : S 13 = 156) :
  a 2 = -3 := 
    sorry

end arithmetic_sequence_a2_value_l789_789315


namespace inequality_proof_l789_789396

variables {x y z : ℝ}

theorem inequality_proof 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x) 
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) : 
  4 * x + y ≥ 4 * z :=
sorry

end inequality_proof_l789_789396


namespace test_total_questions_l789_789120

theorem test_total_questions (total_points : ℕ) (num_5_point_questions : ℕ) (points_per_5_point_question : ℕ) (points_per_10_point_question : ℕ) : 
  total_points = 200 → 
  num_5_point_questions = 20 → 
  points_per_5_point_question = 5 → 
  points_per_10_point_question = 10 → 
  (total_points = (num_5_point_questions * points_per_5_point_question) + 
    ((total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) * points_per_10_point_question) →
  (num_5_point_questions + (total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end test_total_questions_l789_789120


namespace math_problem_l789_789351

noncomputable def condition1 (a b : ℤ) : Prop :=
  |2 + a| + |b - 3| = 0

noncomputable def condition2 (c d : ℝ) : Prop :=
  1 / c = -d

noncomputable def condition3 (e : ℤ) : Prop :=
  e = -5

theorem math_problem (a b e : ℤ) (c d : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 c d) 
  (h3 : condition3 e) : 
  -a^b + 1 / c - e + d = 13 :=
by
  sorry

end math_problem_l789_789351


namespace magnitude_a_plus_2b_l789_789343

noncomputable theory

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 3)
variable (hab : a ⬝ (a + b) = -1)

theorem magnitude_a_plus_2b : ‖a + 2 • b‖ = 2 * Real.sqrt 5 :=
by
  sorry

end magnitude_a_plus_2b_l789_789343


namespace sum_of_values_of_z_sum_of_roots_l789_789402

def f : ℚ → ℚ := λ x, x^2 + x + 1

theorem sum_of_values_of_z (z : ℚ) :
  (f (3 * z) = 7) → z = -1/3 ∨ z = 2/9 :=
begin
  sorry
end

theorem sum_of_roots :
  (∑ z in ({-1/3, 2/9} : finset ℚ), z) = -1/9 :=
begin
  sorry
end

end sum_of_values_of_z_sum_of_roots_l789_789402


namespace count_terminating_fractions_l789_789660

-- Defining the range for n
def in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 210

-- Defining the condition for the decimal representation of n/210 to terminate
def terminates (n : ℕ) : Prop := ∃ k, n = 21 * k

-- The theorem statement
theorem count_terminating_fractions : 
  (finset.filter terminates (finset.range_succ 210)).card = 10 :=
by
  sorry

end count_terminating_fractions_l789_789660


namespace convex_17gon_not_14_triangles_min_17gon_triangles_l789_789863

theorem convex_17gon_not_14_triangles :
  ¬(∃ (T : ℕ), T = 14 ∧ ∑ (x in finset.range 17), (180 : ℝ) = 2700) :=
by sorry

theorem min_17gon_triangles (n : ℕ) (hk : n ≥ 6) :
  ∃ (T : ℕ), T ≥ 6 ∧ 3 * T ≥ 17 :=
by sorry

end convex_17gon_not_14_triangles_min_17gon_triangles_l789_789863


namespace solve_log_equation_l789_789844

theorem solve_log_equation (x : ℝ) (h1 : (4 * x + 12) / (6 * x - 4) > 0) (h2 : (6 * x - 4) / (x - 3) > 0) :
  log 3 ((4 * x + 12) / (6 * x - 4)) + log 3 ((6 * x - 4) / (x - 3)) = 2 
  → x = 39 / 5 :=
by sorry

end solve_log_equation_l789_789844


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789516

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l789_789516


namespace pizza_slices_leftover_l789_789955

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def small_pizzas_purchased := 3
def large_pizzas_purchased := 2

def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3

def total_slices := small_pizzas_purchased * slices_per_small_pizza + large_pizzas_purchased * slices_per_large_pizza
def total_eaten_slices := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

def slices_leftover := total_slices - total_eaten_slices

theorem pizza_slices_leftover : slices_leftover = 10 := by
  sorry

end pizza_slices_leftover_l789_789955


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789209

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789209


namespace value_of_x_is_two_l789_789494

theorem value_of_x_is_two (x : ℝ) (h : x + x^3 = 10) : x = 2 :=
sorry

end value_of_x_is_two_l789_789494


namespace average_weight_proof_l789_789450

noncomputable def average_weight_of_all_boys 
  (average_weight_group1 : ℕ → ℝ) 
  (n1 : ℕ) 
  (average_weight_group2 : ℕ → ℝ) 
  (n2 : ℕ) := 
  (n1 * average_weight_group1 n1 + n2 * average_weight_group2 n2) / (n1 + n2)

theorem average_weight_proof :
  average_weight_of_all_boys (λ _, 50.25) 16 (λ _, 45.15) 8 = 48.55 :=
by 
  sorry

end average_weight_proof_l789_789450


namespace probability_is_1_l789_789146

-- Define the side length of the cube and the properties of the spheres
def cube_side_length (s : ℝ) : ℝ := s

def circumscribed_sphere_radius (s : ℝ) : ℝ :=
  (s * Real.sqrt 3) / 2

def inscribed_sphere_radius (s : ℝ) : ℝ :=
  s / 2

def small_sphere_radius (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 - 1) / 2

-- Definition of volume of a sphere
def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- Probability calculation function
def probability_P_lies_in_smaller_spheres (s : ℝ) : ℝ :=
  let R := circumscribed_sphere_radius s in
  let R_small := small_sphere_radius s in
  let volume_ratio := (R_small / R)^3 in
  let single_sphere_probability := volume_ratio in
  let total_probability := 7 * single_sphere_probability in
  if total_probability > 1 then 1 else total_probability

-- Theorem to prove
theorem probability_is_1 (s : ℝ) : 
  probability_P_lies_in_smaller_spheres s = 1 :=
  sorry

end probability_is_1_l789_789146


namespace Moe_has_least_amount_of_money_l789_789617

variables {B C F J M Z : ℕ}

theorem Moe_has_least_amount_of_money
  (h1 : Z > F) (h2 : F > B) (h3 : Z > C) (h4 : B > M) (h5 : C > M) (h6 : Z > J) (h7 : J > M) :
  ∀ x, x ≠ M → x > M :=
by {
  sorry
}

end Moe_has_least_amount_of_money_l789_789617


namespace one_line_perpendicular_to_plane_infinitely_many_lines_parallel_to_plane_infinitely_many_planes_perpendicular_to_plane_one_plane_parallel_to_plane_l789_789901

-- Definitions
variable {P : Type} [HasAffineGeometry P]
variable (point : P) (plane : Set P)

-- Conditions
variable (point_not_on_plane : point ∉ plane)

-- Statements
theorem one_line_perpendicular_to_plane : ∃! l : Line P, (point ∈ l) ∧ (l ⊥ plane) :=
sorry

theorem infinitely_many_lines_parallel_to_plane : Infinite {l : Line P // point ∈ l ∧ l ∥ plane} :=
sorry

theorem infinitely_many_planes_perpendicular_to_plane : Infinite {π : Set P // plane ⊥ π ∧ point ∈ π} :=
sorry

theorem one_plane_parallel_to_plane : ∃! π : Set P, (plane ∥ π) ∧ (point ∈ π) :=
sorry

end one_line_perpendicular_to_plane_infinitely_many_lines_parallel_to_plane_infinitely_many_planes_perpendicular_to_plane_one_plane_parallel_to_plane_l789_789901


namespace find_a_b_l789_789748

-- Define the conditions
variables (a b x y x₁ y₁ x₂ y₂ : ℝ)
variables (A B : (ℝ × ℝ))
variables (M : (ℝ × ℝ))
variables (O : (ℝ × ℝ))

-- Condition 1: The equation of the ellipse
def ellipse : Prop := a*x^2 + b*y^2 = 1

-- Condition 2: A and B are points of intersection of ellipse with x + y = 1
def intersects : Prop := ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂ ∧ x₁ + y₁ = 1 ∧ x₂ + y₂ = 1

-- Condition 3: M is the midpoint of A and B
def midpoint (A B M : (ℝ × ℝ)) : Prop := M = (((A.1 + B.1) / 2), ((A.2 + B.2) / 2))

-- Condition 4: The slope of OM is 2
def slopeOM (O M : (ℝ × ℝ)) : Prop := (M.2 - O.2) / (M.1 - O.1) = 2

-- Condition 5: OA is perpendicular to OB
def perpendicular (O A B : (ℝ × ℝ)) [A ≠ B] : Prop := (A.2 / A.1) * (B.2 / B.1) = -1

-- The theorem we need to prove
theorem find_a_b (h₁ : ellipse a b x₁ y₁) (h₂ : ellipse a b x₂ y₂) (h₃ : intersects a b x₁ y₁ x₂ y₂) 
(h₄ : midpoint (x₁, y₁) (x₂, y₂) M) (h₅ : slopeOM (0, 0) M) (h₆ : perpendicular (0, 0) (x₁, y₁) (x₂, y₂)) :
  a = 4 / 3 ∧ b = 2 / 3 := sorry

end find_a_b_l789_789748


namespace vector_subtraction_scalar_mul_l789_789618

theorem vector_subtraction_scalar_mul :
  let v₁ := (3, -8) 
  let scalar := -5 
  let v₂ := (4, 6)
  v₁.1 - scalar * v₂.1 = 23 ∧ v₁.2 - scalar * v₂.2 = 22 := by
    sorry

end vector_subtraction_scalar_mul_l789_789618


namespace folded_isosceles_right_triangle_angle_l789_789689

-- Given an isosceles right triangle
structure IsoscelesRightTriangle (α β γ : ℝ) : Prop :=
(hypotenuse: α^2 + α^2 = γ^2)
(right_angle: α + α = β)

-- Folding along the height to the hypotenuse
def fold_along_height (α β γ : ℝ) [IsoscelesRightTriangle α β γ] : Prop := 
  ∃ θ : ℝ, θ = 60 ∧ (after_folding : ∀ a b : ℝ, α = a ∧ β = b)

-- The angle between the two legs after folding is 60 degrees.
theorem folded_isosceles_right_triangle_angle (α β γ : ℝ) 
  [IsoscelesRightTriangle α β γ] 
  : ∃ θ : ℝ, fold_along_height α β γ ∧ θ = 60 := 
sorry

end folded_isosceles_right_triangle_angle_l789_789689


namespace intersection_M_N_l789_789745

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ t : ℝ, x = 2^(-t) }

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Theorem stating the intersection of M and N
theorem intersection_M_N :
  (M ∩ N) = {y : ℝ | 0 < y ∧ y ≤ 1} :=
by
  sorry

end intersection_M_N_l789_789745


namespace correct_propositions_l789_789600

-- Definitions
def collinear (A B C : Point) : Prop :=
  ∃ (line : Line), A ∈ line ∧ B ∈ line ∧ C ∈ line

def coplanar (A B C D : Point) : Prop :=
  ∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane

-- Theorem statement
theorem correct_propositions (A B C D : Point) :
  (∀ (A B C : Point), collinear A B C → coplanar A B C D) ∧
  (¬ coplanar A B C D → ∀ (A B C : Point), ¬ collinear A B C) :=
by
  sorry

end correct_propositions_l789_789600


namespace general_solution_of_ODE_l789_789440

theorem general_solution_of_ODE (C1 C2 : ℝ) :
  let y := λ x : ℝ, C1 * Real.exp (3 * x) + C2 * Real.exp x + (1/3) * x + (1/9)
  let y' := λ x : ℝ, 3 * C1 * Real.exp (3 * x) + C2 * Real.exp x + (1/3)
  let y'' := λ x : ℝ, 9 * C1 * Real.exp (3 * x) + C2 * Real.exp x
  ∀ x : ℝ, y'' x - 4 * y' x + 3 * y x = x - 1 :=
by
  sorry

end general_solution_of_ODE_l789_789440


namespace number_of_triangles_from_intersections_of_chords_l789_789436

theorem number_of_triangles_from_intersections_of_chords
  (points_on_circle : Finset Point)
  (h_points : points_on_circle.card = 7)
  (no_three_chords_intersect : ∀ c1 c2 c3 : Chord, intersection c1 c2 ≠ intersection c2 c3) :
  (points_on_circle.card.choose 2).choose 3 = 6545 := by
  sorry

end number_of_triangles_from_intersections_of_chords_l789_789436


namespace milk_carton_volume_l789_789881

def width : ℝ := 9
def length : ℝ := 4
def height : ℝ := 7
def volume : ℝ := length * width * height

theorem milk_carton_volume : volume = 252 := 
by 
  unfold volume length width height
  -- This is skipped by sorry
  -- Exact manual computation here but skipping the proof
  sorry

end milk_carton_volume_l789_789881


namespace rachel_earnings_l789_789030

theorem rachel_earnings (x : ℝ) (h_lunch : x / 4) (h_dvd : x / 2) (h_remaining : x / 4 = 50) : x = 200 := 
by 
  sorry

end rachel_earnings_l789_789030


namespace continuous_density_exists_l789_789137

variable {X : Type} [MeasurableSpace X] [ProbabilityMeasure X]

noncomputable def phi (z : ℂ) : ℂ := Sorry -- Assume this is given: E[e^{-zX}] for Re(z) = 1
axiom integrable_phi : ∫ (t : ℝ), ∥phi (1 - I * t)∥ < ∞

theorem continuous_density_exists (EX : ∫ x, Real.exp (-x) ∂(X) < ∞) :
  ∃ f : ℝ → ℝ, (∀ x, f x = (1 / (2 * π * I)) * ∫ (z : ℂ) in (subtype (fun z => Real.re z = 1)), Real.exp (x * z) * phi z) :=
sorry

end continuous_density_exists_l789_789137


namespace cos_sum_tan_identity_l789_789313

theorem cos_sum_tan_identity :
  ∃ (p q : ℕ), (p + q = 17) ∧
               (\sum k in Finset.range 40, Real.cos (4 * (k+1) * Real.pi / 180) = 
                Real.tan (p * Real.pi / (q * 180))) ∧ 
               (p.gcd q = 1) ∧ (p * 1 < 180 * q) := by
  sorry

end cos_sum_tan_identity_l789_789313


namespace greatest_integer_gcd_30_is_125_l789_789525

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l789_789525


namespace no_quadratic_term_in_expression_l789_789766

noncomputable def find_k (k : ℝ) (x y : ℝ) : Prop :=
  let expr := (-3 * k * x * y + 3 * y) + (9 * x * y - 8 * x + 1)
  let simplified_expr := (-3 * k + 9) * x * y + 3 * y - 8 * x + 1
  (∀ x y : ℝ, simplified_expr) = (3 * y - 8 * x + 1)

theorem no_quadratic_term_in_expression :
  ∀ k x y : ℝ, find_k k x y → k = 3 :=
by
  sorry

end no_quadratic_term_in_expression_l789_789766


namespace galya_overtakes_sasha_l789_789434

variable {L : ℝ} -- Length of the track
variable (Sasha_uphill_speed : ℝ := 8)
variable (Sasha_downhill_speed : ℝ := 24)
variable (Galya_uphill_speed : ℝ := 16)
variable (Galya_downhill_speed : ℝ := 18)

noncomputable def average_speed (uphill_speed: ℝ) (downhill_speed: ℝ) : ℝ :=
  1 / ((1 / (4 * uphill_speed)) + (3 / (4 * downhill_speed)))

noncomputable def time_for_one_lap (L: ℝ) (speed: ℝ) : ℝ :=
  L / speed

theorem galya_overtakes_sasha 
  (L_pos : 0 < L) :
  let v_Sasha := average_speed Sasha_uphill_speed Sasha_downhill_speed
  let v_Galya := average_speed Galya_uphill_speed Galya_downhill_speed
  let t_Sasha := time_for_one_lap L v_Sasha
  let t_Galya := time_for_one_lap L v_Galya
  (L * 11 / v_Galya) < (L * 10 / v_Sasha) :=
by
  sorry

end galya_overtakes_sasha_l789_789434


namespace eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l789_789926

-- Proof 1: Line through intersection and parallel
theorem eqn_of_line_through_intersection_parallel :
  ∃ k : ℝ, (9 : ℝ) * (x: ℝ) + (18: ℝ) * (y: ℝ) - 4 = 0 ∧
           (∀ x y : ℝ, (2 * x + 3 * y - 5 = 0) → (7 * x + 15 * y + 1 = 0) → (x + 2 * y + k = 0)) :=
sorry

-- Proof 2: Line perpendicular and specific distance from origin
theorem eqn_of_line_perpendicular_distance :
  ∃ k : ℝ, (∃ m : ℝ, (k = 30 ∨ k = -30) ∧ (4 * (x: ℝ) - 3 * (y: ℝ) + m = 0 ∧ (∃ d : ℝ, d = 6 ∧ (|m| / (4 ^ 2 + (-3) ^ 2).sqrt) = d))) :=
sorry

end eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l789_789926


namespace susan_strawberries_l789_789449

theorem susan_strawberries (totalStrawberries : ℕ) (h : totalStrawberries = 75) : 
  let strawberriesInBasket := (totalStrawberries / 5) * 4 in
  strawberriesInBasket = 60 :=
by 
  have fact1 : totalStrawberries = 75 := h
  have fact2 : strawberriesInBasket = (totalStrawberries / 5) * 4 := rfl
  sorry

end susan_strawberries_l789_789449


namespace men_with_tv_at_least_15_l789_789501

theorem men_with_tv_at_least_15
  (total_men : ℕ) (M : ℕ) (R : ℕ) (A : ℕ) (TMRA : ℕ)
  (total_men_eq : total_men = 100)
  (M_eq : M = 85)
  (R_eq : R = 85)
  (A_eq : A = 70)
  (TMRA_eq : TMRA = 15) :
  15 ≤ TMRA :=
by
  rw [TMRA_eq]
  apply le_refl
  -- We use le_refl because TMRA = 15, so 15 ≤ 15

end men_with_tv_at_least_15_l789_789501


namespace profit_percentage_of_revenues_l789_789756

theorem profit_percentage_of_revenues (R P : ℝ)
  (H1 : R > 0)
  (H2 : P > 0)
  (H3 : P * 0.98 = R * 0.098) :
  (P / R) * 100 = 10 := by
  sorry

end profit_percentage_of_revenues_l789_789756


namespace function_equivalence_l789_789923

-- Definitions for our problem
def is_holey (f : ℝ → ℝ) : Prop :=
  ∃ I : set ℝ, ∀ x : ℝ, f x ∉ I

def is_presentable (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ) (fs : fin n → (ℝ → ℝ)),
    (∀ i, fs i = (λ x, k_i * x + b) ∨ fs i = (λ x, x⁻¹) ∨ fs i = (λ x, x^2)) ∧
    (f = (fs 0 ∘ fs 1 ∘ ... ∘ fs (n-1)))

-- Condition that the polynomials do not have common roots
def no_common_roots (a b c d : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + a * x + b = 0 ∧ x^2 + c * x + d = 0) → False

-- Main statement
theorem function_equivalence (a b c d : ℝ) (f : ℝ → ℝ) (h_1 : f = (λ x, (x^2 + a * x + b) / (x^2 + c * x + d))) :
  no_common_roots a b c d →
  (is_holey f ↔ is_presentable f) :=
begin
  sorry
end

end function_equivalence_l789_789923


namespace squared_distances_l789_789798

open Real

-- Given conditions
variables {a b c : ℝ^3} -- Points A, B, and C
variable (λ : ℝ) -- Scalar for P on line BC
def p : ℝ^3 := λ • b + (1 - λ) • c -- Point P on line BC
def g : ℝ^3 := (2 • a + b + c) / 4 -- Centroid G with given weights

-- Define squared distances
def PA2 : ℝ := ((p λ) - a).norm_sq
def PB2 : ℝ := ((p λ) - b).norm_sq
def PC2 : ℝ := ((p λ) - c).norm_sq
def GA2 : ℝ := (g - a).norm_sq
def GB2 : ℝ := (g - b).norm_sq
def GC2 : ℝ := (g - c).norm_sq
def PG2 : ℝ := ((p λ) - g).norm_sq

-- The theorem to prove
theorem squared_distances (λ : ℝ) :
  PA2 λ + PB2 λ + PC2 λ = 4 * PG2 λ + GA2 + GB2 + GC2 :=
by sorry

end squared_distances_l789_789798


namespace largest_multiple_of_36_with_even_and_distinct_digits_l789_789989

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def distinct_digits (digits : List ℕ) : Prop :=
  digits.nodup

def sum_digits (digits : List ℕ) : ℕ :=
  digits.sum

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def divisible_by_36 (n : ℕ) : Prop :=
  divisible_by_4 n ∧ divisible_by_9 n

def largest_distinct_even_digits_multiple_of_36 (n : ℕ) : Prop :=
  ∀ m : ℕ,
  (∀ d ∈ m.digits 10, is_even_digit d) ∧ 
  (distinct_digits m.digits 10) ∧
  divisible_by_36 m →
  m ≤ n

theorem largest_multiple_of_36_with_even_and_distinct_digits :
  largest_distinct_even_digits_multiple_of_36 8640 :=
by
  sorry

end largest_multiple_of_36_with_even_and_distinct_digits_l789_789989


namespace bankers_gain_correct_l789_789453

def PW : ℝ := 600
def R : ℝ := 0.10
def n : ℕ := 2

def A : ℝ := PW * (1 + R)^n
def BG : ℝ := A - PW

theorem bankers_gain_correct : BG = 126 :=
by
  sorry

end bankers_gain_correct_l789_789453


namespace collinearity_of_points_l789_789727

theorem collinearity_of_points (a : ℝ) : 
  (1 : ℝ, -1 : ℝ) ≠ (a, 3) ∧ (1 : ℝ, -1 : ℝ) ≠ (4, 5) ∧ (a, 3) ≠ (4, 5) → 
  ((3 + 1) / (a - 1) = (5 + 1) / (4 - 1)) → 
  a = 3 :=
by
  intro h1 h2
  sorry

end collinearity_of_points_l789_789727


namespace number_of_solutions_l789_789991

theorem number_of_solutions (x y : ℤ) (h : 2^(2 * x) - 3^(2 * y) = 63) : 
  (∃! (x, y) : ℤ × ℤ, 2^(2 * x) - 3^(2 * y) = 63) :=
sorry

end number_of_solutions_l789_789991


namespace find_sin_value_l789_789740

noncomputable def sin_expr (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ sin (3 * π / 8 - x) = 1 / 3

theorem find_sin_value (x : ℝ) (h : sin_expr x) : 
  sin (π / 8 + x) = 2 * real.sqrt 2 / 3 :=
begin
  sorry,
end

end find_sin_value_l789_789740


namespace truncated_pyramid_volume_l789_789589

theorem truncated_pyramid_volume (a : ℝ) (h : a > 0) :
  let angle : ℝ := π / 4,
  let MO := (a * sqrt 3) / 6,
  let area_ABC := (1 / 2) * a * (a * sqrt 3) / 2,
  let area_ECF := (1 / 4) * area_ABC,
  let TK := (3 / 4) * MO,
  let volume_truncated := (1 / 3) * area_ECF * TK,
  volume_truncated = (a^3) / 128 :=
by
  let angle : ℝ := π / 4
  let MO := (a * sqrt 3) / 6
  let area_ABC := (1 / 2) * a * (a * sqrt 3) / 2
  let area_ECF := (1 / 4) * area_ABC
  let TK := (3 / 4) * MO
  let volume_truncated := (1 / 3) * area_ECF * TK
  sorry

end truncated_pyramid_volume_l789_789589


namespace limit_of_a_n_is_two_fifth_l789_789874

def a_n (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 100 then (1 / 3) ^ n
  else (2 * n + 1) / (5 * n - 1)

theorem limit_of_a_n_is_two_fifth : 
  filter.tendsto a_n filter.at_top (nhds (2/5)) :=
sorry

end limit_of_a_n_is_two_fifth_l789_789874


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789234

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789234


namespace king_pages_and_ducats_l789_789024

theorem king_pages_and_ducats (n : ℕ) (d : ℕ) 
  (h_cond : ∃ (page : ℕ), page > 0 ∧ page ≤ n ∧ (n * 2 - page * 2) + 2 = 32) 
  : (n = 16 ∧ d = 992) ∨ (n = 8 ∧ d = 240) :=
by sorry

end king_pages_and_ducats_l789_789024


namespace speed_of_man_in_still_water_l789_789122

variable {v_m v_s : ℝ}

def downstream_time := 3
def downstream_distance := 54
def downstream_speed := v_m + v_s

def upstream_time := 3
def upstream_distance := 18
def upstream_speed := v_m - v_s

theorem speed_of_man_in_still_water :
  (downstream_speed * downstream_time = downstream_distance) →
  (upstream_speed * upstream_time = upstream_distance) →
  v_m = 12 :=
by
  sorry

end speed_of_man_in_still_water_l789_789122


namespace count_perfect_squares_16000_l789_789430

noncomputable def count_perfect_squares : ℕ :=
  let max_n := (Real.floor (Real.sqrt 16000)).natAbs
  let odd_squares := (List.range (max_n + 1)).filter (λ n, n % 2 = 1)
  let can_be_difference_of_squares := odd_squares.filter (λ n, 
    let b := (n - 1) / 2
    (b % 2 = 1 ∧ (b + 1)^2 - b^2 = 2 * b.succ * n)
  )
  can_be_difference_of_squares.length

theorem count_perfect_squares_16000 : 
  count_perfect_squares = 62 := 
  by
  sorry

end count_perfect_squares_16000_l789_789430


namespace sum_of_num_and_denom_of_repeating_decimal_l789_789114

theorem sum_of_num_and_denom_of_repeating_decimal :
  let x := 0.567567567567... (actual infinite decimal representation)
  let fraction := 21 / 37
  in fraction + 37 + 21 = 58 := 
sorry

end sum_of_num_and_denom_of_repeating_decimal_l789_789114


namespace circuit_current_proof_l789_789576

theorem circuit_current_proof
    (E : ℝ) (r : ℝ) (n : ℕ) (R_A : ℝ) (R_B : ℝ)
    (hE : E = 1.8) (hr : r = 0.2) (hn : n = 6) 
    (hR_A : R_A = 6) (hR_B : R_B = 3) :
  ∃ (I_total i1 i2 : ℝ), 
    (I_total ≈ 3.38) ∧ 
    (i1 ≈ 1.13) ∧ 
    (i2 ≈ 2.25) := sorry

end circuit_current_proof_l789_789576


namespace find_base_l789_789092

theorem find_base (b : ℕ) (h : ∑ i in finset.range b, i = 3 * b + 4) : b = 8 :=
sorry

end find_base_l789_789092


namespace vector_expression_simplify_l789_789442

variables (a b : Type) [AddCommGroup a] [AddCommGroup b]

theorem vector_expression_simplify 
  (v1 v2 : a) (v3 v4 : b) :
  (1/2 : ℝ) • (2 • v1 + 8 • v2) - (4 • v1 - 2 • v2) = 6 • v2 - 3 • v1 :=
by sorry

end vector_expression_simplify_l789_789442


namespace cost_reduction_l789_789090

-- Define the conditions
def cost_two_years_ago : ℝ := 5000
def annual_decrease_rate : ℝ := x

-- Define the costs
def cost_last_year (cost_two_years_ago : ℝ) (annual_decrease_rate : ℝ) : ℝ :=
  cost_two_years_ago * (1 - annual_decrease_rate)

def cost_this_year (cost_last_year : ℝ) (annual_decrease_rate : ℝ) : ℝ :=
  cost_last_year * (1 - annual_decrease_rate)

-- Lean 4 statement of the problem
theorem cost_reduction (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1):
  let cost_two_years_ago := 5000 in
  let cost_last_year := cost_two_years_ago * (1 - x) in
  let cost_this_year := cost_last_year * (1 - x) in
  cost_last_year - cost_this_year = 5000x - 5000x^2 :=
by sorry

end cost_reduction_l789_789090


namespace positive_s_value_l789_789986

def E (a b c : ℕ) : ℕ := a * b^c

theorem positive_s_value (s : ℝ) (h1 : E s s 4 = 1296) : s = 6^(0.8) :=
by
  unfold E at h1
  sorry

end positive_s_value_l789_789986


namespace sum_of_products_not_equal_l789_789829

theorem sum_of_products_not_equal :
  (∑ k in Finset.range 98, (k + 1) * (k + 2) * (k + 3)) ≠ 19891988 :=
by
  sorry

end sum_of_products_not_equal_l789_789829


namespace find_AX_l789_789769

theorem find_AX (AB AC BC : ℝ) (CX_bisects_ACB : Prop) (h1 : AB = 50) (h2 : AC = 28) (h3 : BC = 56) : AX = 50 / 3 :=
by
  -- Proof can be added here
  sorry

end find_AX_l789_789769


namespace greatest_int_with_gcd_five_l789_789541

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l789_789541


namespace compute_values_and_sum_l789_789455

noncomputable def f : ℕ → ℝ → ℝ → ℝ
| 0     x y := 0
| (n+1) x y := abs (x + abs (y + f n x y))

def p (n : ℕ) : ℝ := 
  ∫ x in -2..2, ∫ y in -2..2, (if (f n x y < 1) then 1 else 0) / 16

theorem compute_values_and_sum :
  (∃ a b c d : ℕ,
    (tendsto (λ n, p (2 * n + 1)) at_top (𝓝 ((π^2 + a) / b)) ∧
     tendsto (λ n, p (2 * n)) at_top (𝓝 ((π^2 + c) / d)) ∧
     1000*a + 100*b + 10*c + d = 42564)) :=
by
  -- The proof goes here
  sorry

end compute_values_and_sum_l789_789455


namespace sum_m_n_for_jar_candies_l789_789940

theorem sum_m_n_for_jar_candies :
  let total_prob_same_comb : ℚ :=
    (55 / 1615) + (70 / 4845) + (3696 / 14535)
  let simplified_prob := (256 / 909)
  (total_prob_same_comb = simplified_prob) ∧ (256 + 909 = 1165) :=
by
  sorry

end sum_m_n_for_jar_candies_l789_789940


namespace largest_integer_less_than_100_leaving_remainder_4_l789_789233

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l789_789233


namespace ratio_approximately_4099_l789_789130

def prod_inc (k : Nat) (j : Nat) : Nat :=
  (List.range j).map (λ i => k + i).prod

def a := prod_inc 2020 4 -- equivalent to 2020 @ 4
def b := prod_inc 2120 4 -- equivalent to 2120 @ 4

noncomputable def e : Float := (a.toFloat / b.toFloat)

theorem ratio_approximately_4099 : abs (e - 0.4099) < 0.0001 :=
  by
    sorry

end ratio_approximately_4099_l789_789130


namespace product_of_roots_of_equation_l789_789483

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, x = 5 ∨ x = 1/5 → ∏ (x : ℝ) in ({5, 1/5}: finset ℝ), x = 1

theorem product_of_roots_of_equation : problem_statement := 
by 
  sorry

end product_of_roots_of_equation_l789_789483


namespace compute_expression_l789_789174

theorem compute_expression :
    (3 + 5)^2 + (3^2 + 5^2 + 3 * 5) = 113 := 
by sorry

end compute_expression_l789_789174


namespace angle_A_area_triangle_l789_789311

theorem angle_A (a b c : ℝ) (h : (b - c)^2 = a^2 - bc) : 
  ∠A = π / 3 := 
sorry

theorem area_triangle (b : ℝ) (h₁: sin C = 2 * sin B) (h₂ : a = 3) :
  area a b c = 3 * sqrt 3 / 2 := 
sorry

end angle_A_area_triangle_l789_789311


namespace valid_sequences_l789_789390

-- Define the transformation function for a ten-digit number
noncomputable def transform (n : ℕ) : ℕ := sorry

-- Given sequences
def seq1 := 1101111111
def seq2 := 1201201020
def seq3 := 1021021020
def seq4 := 0112102011

-- The proof problem statement
theorem valid_sequences :
  (transform 1101111111 = seq1) ∧
  (transform 1021021020 = seq3) ∧
  (transform 0112102011 = seq4) :=
sorry

end valid_sequences_l789_789390


namespace speed_of_other_train_l789_789085

theorem speed_of_other_train
  (v : ℝ) -- speed of the second train
  (t : ℝ := 2.5) -- time in hours
  (distance : ℝ := 285) -- total distance
  (speed_first_train : ℝ := 50) -- speed of the first train
  (h : speed_first_train * t + v * t = distance) :
  v = 64 :=
by
  -- The proof will be assumed
  sorry

end speed_of_other_train_l789_789085


namespace cloud_height_l789_789620

/--
Given:
- α : ℝ (elevation angle from the top of a tower)
- β : ℝ (depression angle seen in the lake)
- m : ℝ (height of the tower)
Prove:
- The height of the cloud hovering above the observer (h - m) is given by
 2 * m * cos β * sin α / sin (β - α)
-/
theorem cloud_height (α β m : ℝ) :
  (∃ h : ℝ, h - m = 2 * m * Real.cos β * Real.sin α / Real.sin (β - α)) :=
by
  sorry

end cloud_height_l789_789620


namespace length_of_bridge_is_correct_l789_789121

-- Definitions for conditions
def speed_kmph : ℝ := 9 -- speed in km/hr
def time_min : ℝ := 15 -- time in minutes

-- Conversion from km/hr to km/min
def speed_kmpmin : ℝ := speed_kmph / 60

-- distance = speed * time
def distance_km : ℝ := speed_kmpmin * time_min

-- Target statement
theorem length_of_bridge_is_correct : distance_km = 2.25 := by
  sorry

end length_of_bridge_is_correct_l789_789121


namespace is_not_innovative_54_l789_789352

def is_innovative (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < b ∧ b < a ∧ n = a^2 - b^2

theorem is_not_innovative_54 : ¬ is_innovative 54 :=
sorry

end is_not_innovative_54_l789_789352


namespace trapezoid_area_l789_789429

variables {A B C D K M : Type}
variables {AB CD LM : ℝ}
variables {AD BC : ℝ} -- non-parallel sides length
variables {S : ℝ} -- area of trapezoid
variables (isMidpointK : isMidpoint K A D) (isMidpointM : isMidpoint M B C)
variables (isPerpendicular : isPerpendicular M L AB)

-- Define the lengths of the parallel sides and the perpendicular
variables (lengthAB : length AB = ab_val) (lengthCD : length CD = cd_val)
variables (lengthLM : length LM = hm)

theorem trapezoid_area :
  S = AB * LM :=
by sorry

end trapezoid_area_l789_789429


namespace utility_bills_total_l789_789016

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l789_789016


namespace complement_A_in_U_l789_789338

def U : Set ℝ := {x | -3 < x ∧ x < 3}
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

theorem complement_A_in_U : (U \ A) = (λ x, -3 < x ∧ x <= -2) ∪ (λ x, 1 < x ∧ x < 3) := by
  sorry

end complement_A_in_U_l789_789338


namespace phase_shift_of_cosine_l789_789271

theorem phase_shift_of_cosine (a b c : ℝ) (h : c = -π / 4 ∧ b = 3) :
  (-c / b) = π / 12 :=
by
  sorry

end phase_shift_of_cosine_l789_789271


namespace union_of_P_and_Q_l789_789398

theorem union_of_P_and_Q :
  let P := {1, 3, 6}
  let Q := {1, 2, 4, 6}
  P ∪ Q = {1, 2, 3, 4, 6} :=
by {
  let P := {1, 3, 6}
  let Q := {1, 2, 4, 6}
  have h : P ∪ Q = {1, 2, 3, 4, 6},
  sorry
  exact h
}

end union_of_P_and_Q_l789_789398


namespace periodic_function_rational_periodicity_periodic_function_irrational_periodicity_l789_789407

noncomputable def is_periodic_function (f : ℝ → ℝ) (T : ℝ) := ∀ x, f(x + T) = f(x)

theorem periodic_function_rational_periodicity
  (f : ℝ → ℝ) (T : ℝ) (hT : is_periodic_function f T) (h1 : is_periodic_function f 1)
  (hT_bound : 0 < T ∧ T < 1) (hT_rat : ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (T = n / m)) :
  ∃ p : ℕ, 0 < p ∧ is_periodic_function f (1 / p) :=
by
  sorry

theorem periodic_function_irrational_periodicity
  (f : ℝ → ℝ) (T : ℝ) (hT : is_periodic_function f T) (h1 : is_periodic_function f 1)
  (hT_bound : 0 < T ∧ T < 1) (hT_irr : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (T = n / m)) :
  ∃ (a : ℕ → ℝ), (∀ n, a n > 0 ∧ a n < 1 ∧ is_periodic_function f (a n) ∧ (∀ k, a (k + 1) < a k)) :=
by
  sorry

end periodic_function_rational_periodicity_periodic_function_irrational_periodicity_l789_789407


namespace pizza_slices_leftover_l789_789956

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def small_pizzas_purchased := 3
def large_pizzas_purchased := 2

def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3

def total_slices := small_pizzas_purchased * slices_per_small_pizza + large_pizzas_purchased * slices_per_large_pizza
def total_eaten_slices := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

def slices_leftover := total_slices - total_eaten_slices

theorem pizza_slices_leftover : slices_leftover = 10 := by
  sorry

end pizza_slices_leftover_l789_789956


namespace period_of_tan_x_over_3_l789_789106

theorem period_of_tan_x_over_3 : ∃ T > 0, ∀ x, tan (x / 3) = tan ((x + T) / 3) :=
by
  use 3 * Real.pi
  sorry

end period_of_tan_x_over_3_l789_789106


namespace max_of_function_l789_789470

noncomputable def max_value (f : ℝ → ℝ) : ℝ := sorry

theorem max_of_function : max_value (λ x : ℝ, -(x + 1)^2 + 5) = 5 :=
sorry

end max_of_function_l789_789470


namespace area_square_ratio_l789_789110

theorem area_square_ratio (r : ℝ) (h1 : r > 0)
  (s1 : ℝ) (hs1 : s1^2 = r^2)
  (s2 : ℝ) (hs2 : s2^2 = (4/5) * r^2) : 
  (s1^2 / s2^2) = (5 / 4) :=
by 
  sorry

end area_square_ratio_l789_789110


namespace rational_number_theorem_l789_789349

theorem rational_number_theorem (x y : ℚ) 
  (h1 : |(x + 2017 : ℚ)| + (y - 2017) ^ 2 = 0) : 
  (x / y) ^ 2017 = -1 := 
by
  sorry

end rational_number_theorem_l789_789349


namespace sum_of_fractions_l789_789973

theorem sum_of_fractions : (∑ k in Finset.range 16, (k + 1) / 7) = 17.142857 :=
by
  have h : (∑ k in Finset.range 16, (k + 1) / 7) = (∑ k in Finset.range 16, (k + 1)) / 7,
  { sorry },
  have h_sum : (∑ k in Finset.range 16, (k + 1)) = 120,
  { sorry },
  sorry

end sum_of_fractions_l789_789973


namespace sequence_general_formula_and_inequality_l789_789322

theorem sequence_general_formula_and_inequality (a : ℕ → ℕ) (d : ℕ)
  (h1 : ∀ n : ℕ, ∃ c : ℕ, ∃ k : ℕ, a n - 1 = 2^c * k ∧ k < 2^c)
  (h2 : a 1 = 3)
  (h3 : a 3 = 9) :
  (∀ n : ℕ, a n = 2^n + 1) ∧ 
  (∀ n : ℕ, 1 ≤ n → (∑ i in Finset.range (n+1).succ, 1 / (a (i+1) - a i).toReal) < 1) := 
sorry

end sequence_general_formula_and_inequality_l789_789322


namespace find_largest_integer_l789_789249

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l789_789249


namespace derivative_at_one_third_l789_789419

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

theorem derivative_at_one_third : (deriv f (1 / 3) = -3) := by
  sorry

end derivative_at_one_third_l789_789419


namespace utility_bills_total_l789_789015

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l789_789015


namespace alpha_beta_quadratic_l789_789800

noncomputable def complex_alpha_beta (omega : ℂ) (alpha : ℂ) (beta : ℂ) :=
  (omega^9 = 1) ∧ (omega ≠ 1) ∧
  (alpha = omega + omega^3 + omega^6) ∧
  (beta = omega^2 + omega^4 + omega^8) ∧
  (∀ a b : ℝ, (x: ℂ), (x^2 + a * x + b = 0) → a = 1 ∧ b = 1)

theorem alpha_beta_quadratic :
  ∃ omega alpha beta : ℂ, complex_alpha_beta omega alpha beta :=
by {
  sorry
}

end alpha_beta_quadratic_l789_789800


namespace inequality_solution_l789_789350

theorem inequality_solution (x : ℝ) : 
  x^2 - 9 * x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := 
by
  sorry

end inequality_solution_l789_789350


namespace average_book_width_is_3_point_9375_l789_789792

def book_widths : List ℚ := [3, 4, 3/4, 1.5, 7, 2, 5.25, 8]
def number_of_books : ℚ := 8
def total_width : ℚ := List.sum book_widths
def average_width : ℚ := total_width / number_of_books

theorem average_book_width_is_3_point_9375 :
  average_width = 3.9375 := by
  sorry

end average_book_width_is_3_point_9375_l789_789792


namespace rectangle_perimeter_of_divided_square_l789_789069

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

end rectangle_perimeter_of_divided_square_l789_789069


namespace find_certain_number_l789_789480

theorem find_certain_number :
  ∃ n : ℝ, 0.8 = n + 0.675 ∧ n = 0.125 :=
by {
  existsi (0.125 : ℝ),
  split,
  {
    norm_num,
  },
  {
    refl,
  },
}

end find_certain_number_l789_789480


namespace determine_a_plus_h_l789_789176

/-
Consider a hyperbola for which the equations of the asymptotes are 
  y = 3x + 2 
and 
  y = -3x + 8. 

It is given that this hyperbola passes through the point (1, 6). 
The standard form for the equation of the hyperbola is:
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1,
where a, b, h, and k are constants, and a, b > 0.
-/
def hyperbola_condition (a b h k : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  ∀ x y, ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1

theorem determine_a_plus_h :
  ∃ a b h k : ℝ, 
    a > 0 ∧ b > 0 ∧ (∀ y x, y = 3 * x + 2 ∨ y = -3 * x + 8) ∧
    hyperbola_condition a b h k a_pos b_pos ∧
    (1, 6) ∈ set_of (λ xy, hyperbola_condition a b h k a_pos b_pos) ∧
    a + h = 2 := sorry

end determine_a_plus_h_l789_789176


namespace valid_sequences_of_length_23_l789_789735

def validSequences : ℕ → ℕ
| 6     := 1
| 7     := 1
| 8     := 2
| 9     := 2
| 10    := 3
| n     := validSequences (n - 4) + validSequences (n - 5) + 2 * validSequences (n - 6) + validSequences (n - 7)

theorem valid_sequences_of_length_23 : validSequences 23 = 160 :=
by sorry

end valid_sequences_of_length_23_l789_789735


namespace fraction_power_equiv_l789_789981

theorem fraction_power_equiv : (75000^4) / (25000^4) = 81 := by
  sorry

end fraction_power_equiv_l789_789981


namespace vertex_of_quadratic_function_l789_789459

theorem vertex_of_quadratic_function :
  ∃ vertex : ℝ × ℝ, 
  (∀ x : ℝ, let y := -x^2 + 6 * x + 3 in y) → vertex = (3, 12) :=
sorry

end vertex_of_quadratic_function_l789_789459


namespace range_of_m_l789_789673

theorem range_of_m (m : ℝ) :
  (∃ (m : ℝ), (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + 1 = 0 ∧ x2^2 + m * x2 + 1 = 0) ∧ 
  (∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≤ 0)) ↔ (m ≤ 1 ∨ m ≥ 3 ∨ m < -2) :=
by
  sorry

end range_of_m_l789_789673


namespace largest_int_less_than_100_by_7_l789_789230

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789230


namespace semi_circle_perimeter_is_correct_l789_789063

-- Define the given conditions
def radius : ℝ := 35

-- Perimeter of a semicircle with the given radius
def semicircle_perimeter (r : ℝ) := (Real.pi * r) + (2 * r)

-- Prove that the perimeter of the semicircle is 179.9 cm
theorem semi_circle_perimeter_is_correct : semicircle_perimeter radius = 179.9 := 
by
  sorry

end semi_circle_perimeter_is_correct_l789_789063


namespace sin_intersections_l789_789773

theorem sin_intersections : 
  ∃! n, n = 2 ∧ ∀ x ∈ set.Ico 0 (2 * Real.pi), 
    (sin (x + Real.pi / 3) = 1 / 2) → x = Real.pi / 2 ∨ x = 11 * Real.pi / 6 := 
begin
  sorry
end

end sin_intersections_l789_789773


namespace vaccine_cost_reduction_l789_789088

theorem vaccine_cost_reduction (x : ℝ) :
  let cost_two_years_ago := 5000
  ∧ let annual_rate := x
  ∧ let cost_last_year := cost_two_years_ago * (1 - annual_rate)
  ∧ let cost_this_year := cost_last_year * (1 - annual_rate)
  in (cost_last_year - cost_this_year = 5000 * x - 5000 * x^2) :=
by
  sorry

end vaccine_cost_reduction_l789_789088


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789210

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l789_789210


namespace hexagon_diagonal_squares_l789_789603

theorem hexagon_diagonal_squares
  (n : ℕ) (len width1 width2 : ℕ)
  (hexagon : hexagon_shape)
  (squares : finset (fin n)) :
  hexagon.num_squares = 78 →
  hexagon.overall_length = 12 →
  hexagon.width1 = 8 →
  hexagon.width2 = 6 →
  at_least_12_squares (hexagon.diagonal squares) :=
begin
  sorry
end

end hexagon_diagonal_squares_l789_789603


namespace total_books_equals_45_l789_789970

-- Define the number of books bought in each category
def adventure_books : ℝ := 13.0
def mystery_books : ℝ := 17.0
def crime_books : ℝ := 15.0

-- Total number of books bought
def total_books := adventure_books + mystery_books + crime_books

-- The theorem we need to prove
theorem total_books_equals_45 : total_books = 45.0 := by
  -- placeholder for the proof
  sorry

end total_books_equals_45_l789_789970


namespace count_non_prime_repeating_decimals_l789_789276

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_repeating_decimal (n : ℕ) : Prop :=
  let d := n + 1
  ∀ k, (d ≠ 1 ∧ ∀ p : ℕ, prime p → p ∣ d → p ≠ 2 ∧ p ≠ 5)

theorem count_non_prime_repeating_decimals :
  let N := 200
  (σ (λ n, 1) (finset.filter (λ n : ℕ, (1 ≤ n ∧ n ≤ N) ∧ ¬ is_prime n ∧ is_repeating_decimal n) (finset.range (N + 1)))) = 137 :=
sorry

end count_non_prime_repeating_decimals_l789_789276


namespace reciprocal_of_sum_l789_789272

theorem reciprocal_of_sum (h1 : 1/4 : ℚ = 1/4) (h2 : 1/6 : ℚ = 1/6) :
  (1 / ((1/4) + (1/6))) = 12 / 5 :=
by
  sorry

end reciprocal_of_sum_l789_789272


namespace largest_int_less_than_100_by_7_l789_789225

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l789_789225


namespace find_common_ratio_l789_789285

theorem find_common_ratio (a_1 q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS1 : S 1 = a_1)
  (hS2 : S 2 = a_1 * (1 + q))
  (hS3 : S 3 = a_1 * (1 + q + q^2))
  (ha2 : a 2 = a_1 * q)
  (ha3 : a 3 = a_1 * q^2)
  (hcond : 2 * (S 1 + 2 * a 2) = S 3 + a 3 + S 2 + a 2) :
  q = -1/2 :=
by
  sorry

end find_common_ratio_l789_789285


namespace statement_1_statement_4_l789_789803

variable {α β : Type}
variable {m n : α} -- Lines m and n
variable {α β : β} -- Planes α and β

-- Conditions for statements
variable (line_perp : Π {α β : Type}, α → α → Prop)
variable (line_in : Π {α β : Type}, α → β → Prop)
variable (line_parallel: Π {α β : Type}, α → β → Prop)
variable (plane_perp : Π {α β : Type}, β → β → Prop)
variable (plane_intersect : Π {α β γ : Type}, β → β → Set α)
variable (distinct : Π {α : Type}, α → α → Prop)

-- Correct statements
theorem statement_1 :
  distinct m n → distinct α β → line_perp m n → line_perp m α → ¬ line_in n α → line_parallel n α := sorry

theorem statement_4 :
  distinct m n → distinct α β → line_perp m n → line_perp m α → line_perp n β → plane_perp α β := sorry

end statement_1_statement_4_l789_789803


namespace surface_area_of_new_solid_l789_789959

noncomputable def original_prism_surface_area : ℕ := 2 * (4 * 2 + 2 * 2 + 2 * 4)

def new_solid_surface_area : ℕ :=
  let removed_cube_faces := 3 * (1 * 1)
  original_prism_surface_area

theorem surface_area_of_new_solid : new_solid_surface_area = 40 :=
by
  -- We're reworking the surface area according to the condition that the total area remains the same.
  have original_area : original_prism_surface_area = 40,
  { sorry },
  show new_solid_surface_area = 40,
  { rw original_area,
    simp,
    sorry }

end surface_area_of_new_solid_l789_789959


namespace youtube_more_than_tiktok_l789_789005

-- Definitions for followers in different social media platforms
def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500
def total_followers : ℕ := 3840

-- Number of followers on Twitter is half the sum of followers on Instagram and Facebook
def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2

-- Number of followers on TikTok is 3 times the followers on Twitter
def tiktok_followers : ℕ := 3 * twitter_followers

-- Calculate the number of followers on all social media except YouTube
def other_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers

-- Number of followers on YouTube
def youtube_followers : ℕ := total_followers - other_followers

-- Prove the number of followers on YouTube is greater than TikTok by a certain amount
theorem youtube_more_than_tiktok : youtube_followers - tiktok_followers = 510 := by
  -- Sorry is a placeholder for the proof
  sorry

end youtube_more_than_tiktok_l789_789005


namespace coordinates_of_point_l789_789377

variables {P : Type} [MetricSpace P] [AddGroup P] [TopologicalSpace P] [OrderTopology P]
variables (Q R : P) (d1 d2 : ℝ)

def fourth_quadrant (Q : P) [has_order_lt P] [Neg P] : Prop :=
∀ (x y : ℝ), 0 < x ∧ y < 0 → Q = (x, y)

def distance_x (Q : P) (d : ℝ) [dist P] : Prop :=
dist Q (x_F Q) = d

def distance_y (Q : P) (d : ℝ) [dist P] : Prop :=
dist Q (y_F Q) = d

theorem coordinates_of_point (Q : P) [has_order_lt P] [Neg P] [dist P] :
  fourth_quadrant Q ∧ distance_x Q 5 ∧ distance_y Q 3 → Q = (5, -3) := sorry

end coordinates_of_point_l789_789377


namespace smaller_octagon_area_ratio_l789_789478

theorem smaller_octagon_area_ratio 
  (ABCDEFGH : Type) 
  [regular_octagon ABCDEFGH]
  (midpoints : ∀ (A B : ABCDEFGH), midpoint A B → smaller_octagon)
  : ∃ smaller_octagon,
    (area(smaller_octagon) / area(ABCDEFGH)) = 1/4 :=
sorry

end smaller_octagon_area_ratio_l789_789478


namespace smallest_n_not_prime_l789_789666

theorem smallest_n_not_prime : ∃ n, n = 4 ∧ ∀ m : ℕ, m < 4 → Prime (2 * m + 1) ∧ ¬ Prime (2 * 4 + 1) :=
by
  sorry

end smallest_n_not_prime_l789_789666


namespace equation_of_circle_center_0_4_passing_through_3_0_l789_789866

noncomputable def circle_radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem equation_of_circle_center_0_4_passing_through_3_0 :
  ∃ (r : ℝ), (r = circle_radius 0 4 3 0) ∧ (r = 5) ∧ ((x y : ℝ) → ((x - 0) ^ 2 + (y - 4) ^ 2 = r ^ 2) ↔ (x ^ 2 + (y - 4) ^ 2 = 25)) :=
by
  sorry

end equation_of_circle_center_0_4_passing_through_3_0_l789_789866


namespace right_angled_triangle_sides_and_angles_l789_789485

theorem right_angled_triangle_sides_and_angles
  (a d : ℝ)
  (h_area : 2 * 486 = 6 * d^2)   -- Area condition rewritten
  (h_ap : a = 4 * d)             -- Arithmetic progression condition
  (sides : set ℝ := {a - d, a, a + d}) :
  sides = {27, 36, 45} ∧
  ∃ α : ℝ, sin α = 4 / 5 ∧ α = real.arcsin (4 / 5) :=
by
  sorry 

end right_angled_triangle_sides_and_angles_l789_789485


namespace bankers_gain_l789_789452

-- Definitions of given conditions
def present_worth : ℝ := 600
def rate_of_interest : ℝ := 0.10
def time_period : ℕ := 2

-- Statement of the problem to be proved: The banker's gain is 126
theorem bankers_gain 
  (PW : ℝ := present_worth) 
  (r : ℝ := rate_of_interest) 
  (n : ℕ := time_period) :
  let A := PW * (1 + r) ^ n in 
  let BG := A - PW in 
  BG = 126 := 
by 
  sorry

end bankers_gain_l789_789452


namespace constant_sequence_sufficient_arithmetic_sequence_constant_sequence_not_necessary_arithmetic_sequence_l789_789738

noncomputable def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in range (n + 1), a i

def is_arithmetic_sequence (S : ℕ → ℕ) : Prop :=
  ∀ n, S (n + 1) - S n = S 1 - S 0

def is_constant_sequence (a : ℕ → ℕ) : Prop :=
  ∃ c, ∀ n, a n = c

theorem constant_sequence_sufficient_arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = Sn a n) :
  is_constant_sequence a → is_arithmetic_sequence S :=
sorry

theorem constant_sequence_not_necessary_arithmetic_sequence 
  (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : ∀ n, S n = Sn a n) :
  ¬ (is_arithmetic_sequence S → is_constant_sequence a) :=
sorry

end constant_sequence_sufficient_arithmetic_sequence_constant_sequence_not_necessary_arithmetic_sequence_l789_789738


namespace limit_of_exp_over_sin_l789_789980

theorem limit_of_exp_over_sin (α β : ℝ) : 
  tendsto (λ x : ℝ, (exp (α * x) - exp (β * x)) / (sin (α * x) - sin (β * x))) (𝓝 0) (𝓝 1) :=
begin
  sorry
end

end limit_of_exp_over_sin_l789_789980


namespace al_and_barb_coinciding_rest_days_l789_789596

-- Definitions for Al's and Barb's schedules
def al_rest_days : ℕ → bool
| n := n % 6 = 4 ∨ n % 6 = 5

def barb_rest_days : ℕ → bool
| n := n % 7 = 5 ∨ n % 7 = 6

-- The theorem to prove the number of coinciding rest-days in the first 1200 days is 56
theorem al_and_barb_coinciding_rest_days :
  (finset.range 1200).filter (λ n, al_rest_days n ∧ barb_rest_days n).card = 56 :=
by sorry

end al_and_barb_coinciding_rest_days_l789_789596


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789246

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l789_789246


namespace expr_eval_l789_789974

noncomputable def expr_value : ℕ :=
  (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6)

theorem expr_eval : expr_value = 18 := by
  sorry

end expr_eval_l789_789974


namespace sum_of_edge_lengths_of_truncated_octahedron_prism_l789_789152

-- Define the vertices, edge length, and the assumption of the prism being a truncated octahedron
def prism_vertices : ℕ := 24
def edge_length : ℕ := 5
def truncated_octahedron_edges : ℕ := 36

-- The Lean statement to prove the sum of edge lengths
theorem sum_of_edge_lengths_of_truncated_octahedron_prism :
  prism_vertices = 24 ∧ edge_length = 5 ∧ truncated_octahedron_edges = 36 →
  truncated_octahedron_edges * edge_length = 180 :=
by
  sorry

end sum_of_edge_lengths_of_truncated_octahedron_prism_l789_789152


namespace pens_bought_is_17_l789_789862

def number_of_pens_bought (C S : ℝ) (bought_pens : ℝ) : Prop :=
  (bought_pens * C = 12 * S) ∧ (0.4 = (S - C) / C)

theorem pens_bought_is_17 (C S : ℝ) (bought_pens : ℝ) 
  (h1 : bought_pens * C = 12 * S)
  (h2 : 0.4 = (S - C) / C) :
  bought_pens = 17 :=
sorry

end pens_bought_is_17_l789_789862


namespace value_in_parentheses_l789_789768

theorem value_in_parentheses (x : ℝ) (h : x / Real.sqrt 18 = Real.sqrt 2) : x = 6 :=
sorry

end value_in_parentheses_l789_789768


namespace elective_courses_count_l789_789008

theorem elective_courses_count :
  let courses := 4 in
  let max_courses_per_year := 3 in
  let years := 3 in
  let ways_to_group :=
    (Nat.choose courses 2) + (Nat.choose courses 2 / 2) + (Nat.choose courses 3) in
  let ways_to_arrange := Nat.factorial years in
  (ways_to_group * ways_to_arrange) = 78 := by
  let courses := 4
  let max_courses_per_year := 3
  let years := 3
  let ways_to_group := (Nat.choose courses 2) + (Nat.choose courses 2 / 2) + (Nat.choose courses 3)
  let ways_to_arrange := Nat.factorial years
  show (ways_to_group * ways_to_arrange) = 78
  sorry

end elective_courses_count_l789_789008


namespace moles_of_water_used_l789_789267

-- Define the balanced chemical equation's molar ratios
def balanced_reaction (Li3N_moles : ℕ) (H2O_moles : ℕ) (LiOH_moles : ℕ) (NH3_moles : ℕ) : Prop :=
  Li3N_moles = 1 ∧ H2O_moles = 3 ∧ LiOH_moles = 3 ∧ NH3_moles = 1

-- Given 1 mole of lithium nitride and 3 moles of lithium hydroxide produced, 
-- prove that 3 moles of water were used.
theorem moles_of_water_used (Li3N_moles : ℕ) (LiOH_moles : ℕ) (H2O_moles : ℕ) :
  Li3N_moles = 1 → LiOH_moles = 3 → H2O_moles = 3 :=
by
  intros h1 h2
  sorry

end moles_of_water_used_l789_789267


namespace distance_A_B_l789_789486

theorem distance_A_B 
  (perimeter_small_square : ℝ)
  (area_large_square : ℝ)
  (h1 : perimeter_small_square = 8)
  (h2 : area_large_square = 64) :
  let side_small_square := perimeter_small_square / 4
  let side_large_square := Real.sqrt area_large_square
  let horizontal_distance := side_small_square + side_large_square
  let vertical_distance := side_large_square - side_small_square
  let distance_AB := Real.sqrt (horizontal_distance^2 + vertical_distance^2)
  distance_AB = 11.7 :=
  by sorry

end distance_A_B_l789_789486


namespace least_integer_value_l789_789549

theorem least_integer_value (x : ℤ) :
  (∀ x, (|x^2 + 3*x + 10| ≤ 25 - x)) → (-5 ≤ x ∧ x = -5) :=
by
  sorry

end least_integer_value_l789_789549


namespace domain_of_g_l789_789039

theorem domain_of_g {f : ℝ → ℝ} (h : ∀ x ∈ Icc (-6 : ℝ) 9, ContinuousAt f x) :
  (∀ x ∈ Icc (-3 : ℝ) 2, ContinuousAt (λ x, f (-3 * x)) x) :=
by
  -- Proof is omitted
  sorry

end domain_of_g_l789_789039


namespace smallest_positive_period_of_f_intervals_where_f_is_monotonically_decreasing_max_min_values_of_f_in_interval_l789_789326

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - cos x ^ 2 + sin x ^ 2

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x ∈ ℝ, f (x + T) = f x ∧ T = π :=
sorry

theorem intervals_where_f_is_monotonically_decreasing : 
  ∀ k : ℤ, ∀ x ∈ ℝ, x ∈ Set.Icc (k * π + 3 * π / 8) (k * π + 7 * π / 8) → (∀ y ∈ Set.Icc x (k * π + 7 * π / 8), f y < f x ∨ f y = f x) :=
sorry

theorem max_min_values_of_f_in_interval : 
  ∃ x ∈ Set.Icc 0 (π / 2), ∃ y ∈ Set.Icc 0 (π / 2), f x = √2 ∧ f y = -1 :=
sorry

end smallest_positive_period_of_f_intervals_where_f_is_monotonically_decreasing_max_min_values_of_f_in_interval_l789_789326


namespace cube_cross_section_area_range_l789_789676

theorem cube_cross_section_area_range
  (edge_length : ℝ)
  (BD1_line_exists_in_cube : ∃ (B D1 : ℝ), (BD1 = B) + (BD1 = D1) = edge_length)
  (plane_passing_through_BD1 : ∀ (α : ℝ), α ≠ 0) :
  ∃ (S_min S_max : ℝ), S_min = sqrt 6 / 2 ∧ S_max = sqrt 2 :=
begin
  sorry
end

end cube_cross_section_area_range_l789_789676


namespace phase_shift_of_cosine_l789_789270

theorem phase_shift_of_cosine (a b c : ℝ) (h : c = -π / 4 ∧ b = 3) :
  (-c / b) = π / 12 :=
by
  sorry

end phase_shift_of_cosine_l789_789270


namespace parallel_tangents_l789_789340

theorem parallel_tangents (x₀ : ℝ) (h₁ : ∀ x, deriv (λ x, x^2 - 1) x = 2 * x)
  (h₂ : ∀ x, deriv (λ x, 1 - x^3) x = -3 * x^2) (h_parallel : 2 * x₀ = -3 * x₀^2) :
  x₀ = 0 ∨ x₀ = -2 / 3 :=
by
  sorry

end parallel_tangents_l789_789340


namespace complex_symmetry_about_real_axis_l789_789675

noncomputable def complex_conjugate (z : ℂ) : ℂ := conj z

theorem complex_symmetry_about_real_axis (z : ℂ) : 
  (∃ a b : ℝ, z = a + b * I) ∧ (complex_conjugate z = a - b * I) → 
  symmetric_about_real_axis z complex_conjugate(z) :=
by
  sorry
  
def symmetric_about_real_axis (z1 z2 : ℂ) : Prop := 
  ∃ a b : ℝ, z1 = a + b * I ∧ z2 = a - b * I

end complex_symmetry_about_real_axis_l789_789675


namespace intersections_of_line_with_shapes_l789_789605

def lattice (p q : ℝ) : Prop := 
  ∃ m n : ℤ, p = m ∧ q = n

def circle_centered_at (x y r : ℝ) : Prop := 
  ∃ k l : ℤ, x = k ∧ y = l ∧ r = 1 / 8

def square_centered_at (x y s : ℝ) : Prop := 
  ∃ k l : ℤ, x = k ∧ y = l ∧ s = 1 / 4

def line_diagonal (x y : ℝ) : Prop := 
  ∃ t : ℝ, x = 401 * t ∧ y = 229 * t
  
theorem intersections_of_line_with_shapes :
  ∀ (m n : ℕ), 
  (∃ p q : ℝ, lattice p q → square_centered_at p q 1/4 ∧ line_diagonal (401 : ℝ) (229 : ℝ)) 
  → 
  (∃ x y : ℝ, lattice x y → circle_centered_at x y 1/8 ∧ line_diagonal (401 : ℝ) (229 : ℝ))
  → 
  m + n = 404 :=
by
  sorry

end intersections_of_line_with_shapes_l789_789605


namespace positive_numbers_with_cube_root_lt_10_l789_789345

def cube_root_lt_10 (n : ℕ) : Prop :=
  (↑n : ℝ)^(1 / 3 : ℝ) < 10

theorem positive_numbers_with_cube_root_lt_10 : 
  ∃ (count : ℕ), (count = 999) ∧ ∀ n : ℕ, (1 ≤ n ∧ n ≤ 999) → cube_root_lt_10 n :=
by
  sorry

end positive_numbers_with_cube_root_lt_10_l789_789345


namespace right_triangle_acute_angle_l789_789363

theorem right_triangle_acute_angle (a b c : ℝ) 
  (h1 : c^2 = 2 * a * b) (h2 : c^2 = a^2 + b^2) : 
  ∃ θ : ℝ, θ = 45 ∧ acute_angle θ a b :=
by
  sorry

end right_triangle_acute_angle_l789_789363


namespace six_rays_pairwise_angle_not_exceed_90_l789_789123

theorem six_rays_pairwise_angle_not_exceed_90 :
  ∀ (rays : Fin 6 → ℝ^3),
  (∀ (i j : Fin 6), i ≠ j → angle (rays i) (rays j) ≥ 90) →
  ∃ (i j : Fin 6), i ≠ j ∧ angle (rays i) (rays j) ≤ 90 :=
begin
  sorry,
end

end six_rays_pairwise_angle_not_exceed_90_l789_789123


namespace combined_weight_of_barney_and_five_dinosaurs_l789_789615

theorem combined_weight_of_barney_and_five_dinosaurs:
  let w := 800
  let combined_weight_five_regular := 5 * w
  let barney_weight := combined_weight_five_regular + 1500
  let combined_weight := barney_weight + combined_weight_five_regular
  in combined_weight = 9500 := by
  sorry

end combined_weight_of_barney_and_five_dinosaurs_l789_789615


namespace right_triangle_hypotenuse_l789_789364

theorem right_triangle_hypotenuse (x : ℝ) (h : x^2 = 3^2 + 5^2) : x = Real.sqrt 34 :=
by sorry

end right_triangle_hypotenuse_l789_789364


namespace greatest_integer_with_gcd_l789_789523

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l789_789523


namespace find_other_number_l789_789003

def integers_three_and_four_sum (a b : ℤ) : Prop :=
  3 * a + 4 * b = 131

def one_of_the_numbers_is (x : ℤ) : Prop :=
  x = 17

theorem find_other_number (a b : ℤ) (h1 : integers_three_and_four_sum a b) (h2 : one_of_the_numbers_is a ∨ one_of_the_numbers_is b) :
  (a = 21 ∨ b = 21) :=
sorry

end find_other_number_l789_789003


namespace d_approx_l789_789586
noncomputable def find_d (d : ℝ) : Prop :=
  let prob := (π * d^2) in
  prob = 1 / 3

theorem d_approx (d : ℝ) (h : find_d d) : abs (d - 0.3) < 0.1 :=
  by
    sorry

end d_approx_l789_789586


namespace many_vertices_one_edge_l789_789359

-- Definitions based on the problem's conditions
def initial_graph (G : Type) [Graph G] : Prop :=
  ∃ (vertices : Finset G) (h_card : vertices.card = 2002) (h_connected : connected G)
    (h_vertex_removal : ∀ (v : G), connected (G.remove_vertex v) ≠ erased G),
    true

def cyclic_route_removal (G : Type) [Graph G] (cyclic_route : List G) : G → Type :=
  λ new_vertex, (G.add_vertex new_vertex).connects_to_vertices cyclic_route
    ∧ (G.remove_edges cyclic_route).connected

-- The final theorem statement ensures that enough vertices have exactly one edge
theorem many_vertices_one_edge (G : Type) [Graph G] (h_initial : initial_graph G) 
  (h_transform : ∀ (cyclic_route : List G) (new_vertex : G), cyclic_route_removal G cyclic_route new_vertex) :
  ∃ (vertices : Finset G), (vertices.filter (λ v, v.degree = 1)).card ≥ 2002 :=
begin
  sorry
end

end many_vertices_one_edge_l789_789359


namespace length_of_PQ_in_right_triangle_l789_789682

theorem length_of_PQ_in_right_triangle
  (P Q R : Type)
  [inhabited P] [inhabited Q] [inhabited R]
  (PR QR PQ : ℝ) 
  (h1 : PR = 10) 
  (h2 : ∠ PQR = 45) 
  (h3 : ∠ PRQ = 90) :
  PQ = 10 := sorry

end length_of_PQ_in_right_triangle_l789_789682


namespace complement_union_A_B_l789_789927

open Set

variable {U : Type*} [Preorder U] [BoundedOrder U]

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  compl (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_A_B_l789_789927


namespace blue_balls_unchanged_l789_789895

def blue_ball_invariance (initial_blue_balls : ℕ) (initial_red_balls : ℕ) (added_red_balls : ℕ) : Prop :=
  initial_blue_balls = 3 → initial_red_balls = 5 → added_red_balls = 2 → initial_blue_balls = 3

theorem blue_balls_unchanged : blue_ball_invariance 3 5 2 :=
by
  assume h1 h2 h3
  exact h1

end blue_balls_unchanged_l789_789895


namespace skater_problem_correct_l789_789962

noncomputable def initial_velocity (d : Float) (t : Float) : Float :=
  (2 * d) / t

noncomputable def coefficient_of_friction (v0 : Float) (t : Float) (g : Float) : Float :=
  - v0 / (t * g)

theorem skater_problem_correct : 
  ∀ (d t g : Float), 
  d = 250 → t = 45 → g = 9.81 →
  initial_velocity d t = 500 / 45 ∧ 
  coefficient_of_friction (initial_velocity d t) t g ≈ 0.025 :=
by
  intros d t g hd ht hg
  have v0_def : initial_velocity d t = 500 / 45 :=
    by sorry
  have rho_def : coefficient_of_friction (initial_velocity d t) t g ≈ 0.025 :=
    by sorry
  exact And.intro v0_def rho_def
  
end skater_problem_correct_l789_789962
