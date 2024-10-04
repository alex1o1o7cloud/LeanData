import Mathlib

namespace books_in_school_libraries_l124_124082

theorem books_in_school_libraries 
  (total_books : ℕ) 
  (public_library_books : ℕ) 
  (school_library_books : ℕ) 
  (h1 : total_books = 7092)
  (h2 : public_library_books = 1986)
  : school_library_books = total_books - public_library_books 
  := by
    have h3 : total_books - public_library_books = 5106 := by sorry
    exact h3

end books_in_school_libraries_l124_124082


namespace total_stickers_l124_124769

theorem total_stickers (r s t : ℕ) (h1 : r = 30) (h2 : s = 3 * r) (h3 : t = s + 20) : r + s + t = 230 :=
by sorry

end total_stickers_l124_124769


namespace euclidean_algorithm_gcd_l124_124370

theorem euclidean_algorithm_gcd (a b : ℕ) (h : a > b) :
  ∃ d, (let gcd := Nat.gcd a b in gcd = d ∧ Nat.gcd a b = gcd ∧ Nat.gcd b (a % b) = gcd ∧ d = (a % b)) using euclidean algorithm
sorry

end euclidean_algorithm_gcd_l124_124370


namespace positional_relationship_between_a_and_c_l124_124656

-- Defining the types for lines and skew.
def Line : Type := sorry
def skew (l1 l2 : Line) : Prop := sorry

-- Defining the conditions.
variables {a b c : Line}
axiom skew_ab : skew a b
axiom skew_bc : skew b c

-- The proof statement.
theorem positional_relationship_between_a_and_c : (skew a c ∨ (∃ p, intersect_line a c p) ∨ parallel a c) :=
sorry

end positional_relationship_between_a_and_c_l124_124656


namespace josh_marbles_l124_124333

theorem josh_marbles (initial_marbles lost_marbles remaining_marbles : ℤ) 
  (h1 : initial_marbles = 19) 
  (h2 : lost_marbles = 11) 
  (h3 : remaining_marbles = initial_marbles - lost_marbles) : 
  remaining_marbles = 8 := 
by
  sorry

end josh_marbles_l124_124333


namespace exists_convex_polygon_diagonals_l124_124862

theorem exists_convex_polygon_diagonals :
  ∃ n : ℕ, n * (n - 3) / 2 = 54 :=
by
  sorry

end exists_convex_polygon_diagonals_l124_124862


namespace opposite_of_83_is_84_l124_124932

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l124_124932


namespace kyle_money_left_l124_124707

-- Define variables and conditions
variables (d k : ℕ)
variables (has_kyle : k = 3 * d - 12) (has_dave : d = 46)

-- State the theorem to prove 
theorem kyle_money_left (d k : ℕ) (has_kyle : k = 3 * d - 12) (has_dave : d = 46) :
  k - k / 3 = 84 :=
by
  -- Sorry to complete the proof block
  sorry

end kyle_money_left_l124_124707


namespace coefficient_of_x_squared_l124_124395

theorem coefficient_of_x_squared :
  ∀ (x : ℝ), 
    let expansion := (1 - (1 / x)) * (1 + x)^4
    in (∀ x, (expansion).coeff_of (2) = 2) :=
sorry

end coefficient_of_x_squared_l124_124395


namespace diametrically_opposite_to_83_l124_124907

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l124_124907


namespace probability_all_successful_pairs_expected_successful_pairs_l124_124833

-- Lean statement for part (a)
theorem probability_all_successful_pairs (n : ℕ) :
  let prob_all_successful := (2 ^ n * n.factorial) / (2 * n).factorial
  in prob_all_successful = (1 / (2 ^ n)) * (n.factorial / (2 * n).factorial)
:= by sorry

-- Lean statement for part (b)
theorem expected_successful_pairs (n : ℕ) :
  let expected_value := n / (2 * n - 1)
  in expected_value > 0.5
:= by sorry

end probability_all_successful_pairs_expected_successful_pairs_l124_124833


namespace problem_statement_l124_124278

theorem problem_statement (a b c : ℝ) 
  (h1 : a - 2 * b + c = 0) 
  (h2 : a + 2 * b + c < 0) : b < 0 ∧ b^2 - a * c ≥ 0 :=
by
  sorry

end problem_statement_l124_124278


namespace guard_maximum_demand_l124_124942

-- Definitions based on conditions from part a)
-- A type for coins
def Coins := ℕ

-- Definitions based on question and conditions
def maximum_guard_demand (x : Coins) : Prop :=
  ∀ x, (x ≤ 199 ∧ x - 100 < 100) ∨ (x > 199 → 100 ≤ x - 100)

-- Statement of the problem in Lean 4
theorem guard_maximum_demand : ∃ x : Coins, maximum_guard_demand x ∧ x = 199 :=
by
  sorry

end guard_maximum_demand_l124_124942


namespace ben_owes_rachel_l124_124767

theorem ben_owes_rachel :
  let dollars_per_lawn := (13 : ℚ) / 3
  let lawns_mowed := (8 : ℚ) / 5
  let total_owed := (104 : ℚ) / 15
  dollars_per_lawn * lawns_mowed = total_owed := 
by 
  sorry

end ben_owes_rachel_l124_124767


namespace multiples_of_2_between_l124_124645

theorem multiples_of_2_between : ∀ (n : ℕ), (n = 1002) → (finset.card (finset.filter (λ x, x % 2 = 0) (finset.Ico 102 n)) = 450) :=
by
  intros n hn
  rw hn
  sorry

end multiples_of_2_between_l124_124645


namespace find_original_intensity_l124_124780

variable (I : ℝ)  -- Define intensity of the original red paint (in percentage).

-- Conditions:
variable (fractionReplaced : ℝ) (newIntensity : ℝ) (replacingIntensity : ℝ)
  (fractionReplaced_eq : fractionReplaced = 0.8)
  (newIntensity_eq : newIntensity = 30)
  (replacingIntensity_eq : replacingIntensity = 25)

-- Theorem statement:
theorem find_original_intensity :
  (1 - fractionReplaced) * I + fractionReplaced * replacingIntensity = newIntensity → I = 50 :=
sorry

end find_original_intensity_l124_124780


namespace point_outside_circle_l124_124661

theorem point_outside_circle (D E F x0 y0 : ℝ) (h : (x0 + D / 2)^2 + (y0 + E / 2)^2 > (D^2 + E^2 - 4 * F) / 4) :
  x0^2 + y0^2 + D * x0 + E * y0 + F > 0 :=
sorry

end point_outside_circle_l124_124661


namespace train_length_l124_124162

/-- Proof problem: 
  Given the speed of a train is 52 km/hr and it crosses a 280-meter long platform in 18 seconds,
  prove that the length of the train is 259.92 meters.
-/
theorem train_length (speed_kmh : ℕ) (platform_length : ℕ) (time_sec : ℕ) (speed_mps : ℝ) 
  (distance_covered : ℝ) (train_length : ℝ) :
  speed_kmh = 52 → platform_length = 280 → time_sec = 18 → 
  speed_mps = (speed_kmh * 1000) / 3600 → distance_covered = speed_mps * time_sec →
  train_length = distance_covered - platform_length →
  train_length = 259.92 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_length_l124_124162


namespace probability_joint_l124_124415

variables (Ω : Type) [ProbabilitySpace Ω]

def eventA : Event Ω := {ω | passesTest ω}
def eventB : Event Ω := {ω | passesExam ω}

theorem probability_joint :
  (probability eventA = 0.8) →
  (probability (eventB | eventA) = 0.9) →
  probability (eventA ∩ eventB) = 0.72 :=
begin
  sorry
end

end probability_joint_l124_124415


namespace expand_expression_l124_124208

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := 
by 
  sorry

end expand_expression_l124_124208


namespace max_omega_l124_124401

def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

def g (ω x : ℝ) : ℝ := Real.sin (ω * x - ω * Real.pi / 4 + Real.pi / 4)

def is_monotonic_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ (x y : ℝ), x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem max_omega (ω : ℝ) :
  (0 < ω) → (is_monotonic_on (g ω) (Set.Ioo (Real.pi / 4) (5 * Real.pi / 4))) → 
  (ω ≤ 1 / 4) := 
by
  sorry

end max_omega_l124_124401


namespace max_min_f_on_interval_l124_124067

def f (x : ℝ) : ℝ := x^2 - 6 * x + 8

theorem max_min_f_on_interval :
  (∀ x ∈ set.Icc (2 : ℝ) 6, f x ≤ 8) ∧
  (f 2 ≤ 8) ∧
  (f 6 = 8) ∧
  (∀ x ∈ set.Icc (2 : ℝ) 6, f x ≥ -1) ∧
  (f 3 = -1) :=
by
  sorry

end max_min_f_on_interval_l124_124067


namespace alice_age_multiple_sum_l124_124516

theorem alice_age_multiple_sum (B : ℕ) (C : ℕ := 3) (A : ℕ := B + 2) (next_multiple_age : ℕ := A + (3 - (A % 3))) :
  B % C = 0 ∧ A = B + 2 ∧ C = 3 → 
  (next_multiple_age % 3 = 0 ∧
   (next_multiple_age / 10) + (next_multiple_age % 10) = 6) := 
by
  intros h
  sorry

end alice_age_multiple_sum_l124_124516


namespace brick_width_l124_124140

/-- Let dimensions of the wall be 700 cm (length), 600 cm (height), and 22.5 cm (thickness).
    Let dimensions of each brick be 25 cm (length), W cm (width), and 6 cm (height).
    Given that 5600 bricks are required to build the wall, prove that the width of each brick is 11.25 cm. -/
theorem brick_width (W : ℝ)
  (h_wall_dimensions : 700 = 700) (h_wall_height : 600 = 600) (h_wall_thickness : 22.5 = 22.5)
  (h_brick_length : 25 = 25) (h_brick_height : 6 = 6) (h_num_bricks : 5600 = 5600)
  (h_wall_volume : 700 * 600 * 22.5 = 9450000)
  (h_brick_volume : 25 * W * 6 = 9450000 / 5600) :
  W = 11.25 :=
sorry

end brick_width_l124_124140


namespace find_point_B_l124_124618

-- Definition of Point
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of conditions
def A : Point := ⟨1, 2⟩
def d : ℝ := 3
def AB_parallel_x (A B : Point) : Prop := A.y = B.y

theorem find_point_B (B : Point) (h_parallel : AB_parallel_x A B) (h_dist : abs (B.x - A.x) = d) :
  (B = ⟨4, 2⟩) ∨ (B = ⟨-2, 2⟩) :=
by
  sorry

end find_point_B_l124_124618


namespace q_factor_change_l124_124124

variable {w f z v : ℝ}

def q (w f z v : ℝ) : ℝ := (5 * w) / (4 * v * f * (z ^ 2))

theorem q_factor_change (w_new f_new z_new : ℝ) (h_w: w_new = 4 * w) (h_f: f_new = 2 * f) (h_z: z_new = 3 * z) :
  q w_new f_new z_new v = (2/9) * q w f z v :=
by 
  sorry

end q_factor_change_l124_124124


namespace smallest_w_for_factors_l124_124123

theorem smallest_w_for_factors (w : ℕ) (h_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (13^2 ∣ 936 * w) ↔ w = 156 := 
sorry

end smallest_w_for_factors_l124_124123


namespace number_of_elements_in_set_l124_124813

theorem number_of_elements_in_set 
  (S : ℝ) (n : ℝ) 
  (h_avg : S / n = 6.8) 
  (a : ℝ) (h_a : a = 6) 
  (h_new_avg : (S + 2 * a) / n = 9.2) : 
  n = 5 := 
  sorry

end number_of_elements_in_set_l124_124813


namespace perpendicular_and_area_l124_124273

structure Point where
  x : ℝ
  y : ℝ

def parabola (p : Point) : Prop := p.y^2 = -p.x

def line (k : ℝ) (p : Point) : Prop := p.y = k * (p.x + 1)

def orthogonal (a b : Point) : Prop := a.x * b.x + a.y * b.y = 0

def area_triangle (a b : Point) : ℝ :=
  0.5 * |a.y - b.y|

theorem perpendicular_and_area (k : ℝ) (A B : Point)
  (h1 : parabola A) (h2 : parabola B) (h3 : line k A) (h4 : line k B) :
  orthogonal A B ∧ (if (area_triangle A B = sqrt 10) then k = 1 / 6 ∨ k = -1 / 6 else True) := by
  sorry

end perpendicular_and_area_l124_124273


namespace opposite_point_83_is_84_l124_124896

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l124_124896


namespace gcd_triang_num_gcd_triang_num_max_l124_124225

open Nat

theorem gcd_triang_num (n : ℕ) : n > 0 → gcd (3 * (n * (n + 1) / 2) + n) (n + 3) ≤ 12 :=
by sorry

theorem gcd_triang_num_max (n : ℕ) : ∃ k, n = 6*k - 3 ∧ gcd (3 * (n * (n + 1) / 2) + n) (n + 3) = 12 :=
by sorry

end gcd_triang_num_gcd_triang_num_max_l124_124225


namespace opposite_point_83_is_84_l124_124900

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l124_124900


namespace radius_of_cookie_l124_124056

theorem radius_of_cookie (x y : ℝ) :
  x^2 + y^2 + 2 * x - 4 * y - 7 = 0 →
  ∃ r : ℝ, r = 2 * real.sqrt 3 :=
sorry

end radius_of_cookie_l124_124056


namespace angle_ABC_l124_124075

theorem angle_ABC {A B C O : Type}
  (h_center : O = center_of_circumscribed_circle A B C)
  (h_angle_BOC : ∠ B O C = 130)
  (h_angle_AOB : ∠ A O B = 150) :
  ∠ A B C = 40 := by
  sorry

end angle_ABC_l124_124075


namespace parabola_intersection_probability_l124_124438

-- Definitions
def parabola1 (a b x : ℝ) : ℝ := x^2 + a * x + b
def parabola2 (c d x : ℝ) : ℝ := x^2 + c * x + d + 2

-- Main Theorem
theorem parabola_intersection_probability :
  (∀ (a b c d : ℕ), 
  a ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  b ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  c ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  d ∈ {1, 2, 3, 4, 5, 6, 7, 8}) → 
  (prob (parabola1 a b = parabola2 c d) = 57 / 64) :=
sorry

end parabola_intersection_probability_l124_124438


namespace number_of_subsets_of_M_l124_124802

def M : Set (ℕ × ℕ) :=
  {p | let x := p.1 in let y := p.2 in (x > 0 ∧ y > 0 ∧ (Real.log x / Real.log 4 + Real.log y / Real.log 4) ≤ 1)}

open Finset

theorem number_of_subsets_of_M : card (powerset (M.toFinset)).val = 256 := 
  sorry

end number_of_subsets_of_M_l124_124802


namespace integer_rational_ratio_l124_124461

open Real

theorem integer_rational_ratio (a b : ℤ) (h : (a : ℝ) + sqrt b = sqrt (15 + sqrt 216)) : (a : ℚ) / b = 1 / 2 := 
by 
  -- Omitted proof 
  sorry

end integer_rational_ratio_l124_124461


namespace at_least_half_sectors_occupied_l124_124142

theorem at_least_half_sectors_occupied (n : ℕ) (chips : Finset (Fin n.succ)) 
(h_chips_count: chips.card = n + 1) :
  ∃ (steps : ℕ), ∀ (t : ℕ), t ≥ steps → (∃ sector_occupied : Finset (Fin n), sector_occupied.card ≥ n / 2) :=
sorry

end at_least_half_sectors_occupied_l124_124142


namespace smallest_possible_value_of_beta_plus_delta_l124_124146

noncomputable def g (z : ℂ) (β δ : ℂ) : ℂ := (3 + 2*complex.I) * z^2 + β * z + δ

theorem smallest_possible_value_of_beta_plus_delta (β δ : ℂ)
  (h1 : (g 1 β δ).im = 0)
  (h2 : (g (-complex.I) β δ).im = 0) :
  |β| + |δ| = 2 * real.sqrt 2 :=
sorry

end smallest_possible_value_of_beta_plus_delta_l124_124146


namespace workers_together_time_l124_124465

theorem workers_together_time (A_time B_time : ℝ) (hA : A_time = 8) (hB : B_time = 10) :
  let rateA := 1 / A_time
  let rateB := 1 / B_time
  let combined_rate := rateA + rateB
  combined_rate * (40 / 9) = 1 :=
by 
  sorry

end workers_together_time_l124_124465


namespace prob_all_successful_pairs_l124_124831

theorem prob_all_successful_pairs (n : ℕ) : 
  let num_pairs := (2 * n)!
  let num_successful_pairs := 2^n * n!
  prob_all_successful_pairs := num_successful_pairs / num_pairs 
  prob_all_successful_pairs = (2^n * n!) / (2 * n)! := 
sorry

end prob_all_successful_pairs_l124_124831


namespace board_coloring_l124_124682

theorem board_coloring (n : ℕ) (h : n = 2018) :
  ∃ (coloring : matrix (fin n) (fin n) (fin 2)), 
  (∀ (i j : fin n), (coloring i j = coloring i j).distinct ∧ 
                    (coloring j i = coloring j i).distinct)  ∧ 
  (∃ (ways : ℕ), ways = 2 * (n!)^2) := 
begin
  -- Sorry is a placeholder for the proof
  sorry,
end

end board_coloring_l124_124682


namespace subset_condition_l124_124609

theorem subset_condition (m : ℝ) (A : Set ℝ) (B : Set ℝ) :
  A = {1, 3} ∧ B = {1, 2, m} ∧ A ⊆ B → m = 3 :=
by
  sorry

end subset_condition_l124_124609


namespace diametrically_opposite_number_l124_124923

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l124_124923


namespace units_digit_of_G_500_is_3_l124_124190

def G_n : ℕ → ℕ
| 0 := 2 + 1
| (n+1) := 2^(3^(n+1)) + 1 -- the given sequence

theorem units_digit_of_G_500_is_3 : (G_n 500) % 10 = 3 :=
by
  -- Sorry indicates this proof needs to be completed
  sorry

end units_digit_of_G_500_is_3_l124_124190


namespace OP_parallel_BD_l124_124015

theorem OP_parallel_BD {k : Type} [metric_space k]
  (O A B C D P : k)
  (hO : metric.has_center k O)
  (hab_diameter : AB = 2 * radius k)
  (horder : ∀ {X : k}, X ∈ {A, B, C, D} → True)
  (hcod_circum : circumcircle {C, O, D} ∩ AC = {P}) :
  is_parallel (line_segment O P) (line_segment B D) :=
sorry

end OP_parallel_BD_l124_124015


namespace distance_is_correct_l124_124411

noncomputable def distance_between_intersections : ℝ :=
  let parabola := { p : ℝ × ℝ | p.2 ^ 2 = 12 * p.1 }
  let circle := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 - 6 * p.2 = 0 }
  let intersections := { p : ℝ × ℝ | p ∈ parabola ∧ p ∈ circle }
  if h : intersections.to_finset = {[0, 0], [3, 6]} then
    dist (0,0) (3,6)
  else
    0

theorem distance_is_correct : distance_between_intersections = 3 * real.sqrt 5 := 
  sorry

end distance_is_correct_l124_124411


namespace faster_speed_l124_124358

theorem faster_speed (distance initial_speed saved_time : ℝ) 
  (h_distance : distance = 1200) 
  (h_initial_speed : initial_speed = 50) 
  (h_saved_time : saved_time = 4) : 
  ∃ v : ℝ, v = 60 :=
by
  -- Definitions corresponding to the problem conditions
  let time_at_initial_speed := distance / initial_speed
  let time_at_faster_speed := time_at_initial_speed - saved_time
  have h_time_at_initial_speed : time_at_initial_speed = 24, 
  sorry
  
  have h_time_at_faster_speed : time_at_faster_speed = 20, 
  sorry
  
  let v := distance / time_at_faster_speed
  use v
  have h_v : v = 60, 
  sorry

end faster_speed_l124_124358


namespace maximum_value_of_expression_l124_124730

noncomputable def calc_value (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression :
  ∃ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧ x ≥ y ∧ y ≥ z ∧
  calc_value x y z = 2916 / 729 :=
by
  sorry

end maximum_value_of_expression_l124_124730


namespace necessary_but_not_sufficient_l124_124342

def M (x : ℝ) : Prop := x > 0
def N (x : ℝ) : Prop := x > 1

theorem necessary_but_not_sufficient (a : ℝ) 
  (h1 : N a → M a) 
  (h2 : M a → ¬N a) : 
  (∀ a, N a → M a) ∧ (¬ (∀ a, M a → N a)) :=
by
  split
  · -- proof of (∀ a, N a → M a)
    intro a h
    exact h1 h
  · -- proof of ¬ (∀ a, M a → N a)
    intro hcontra
    have h := hcontra a (h2 a)
    exact h (h2 a)

end necessary_but_not_sufficient_l124_124342


namespace phi_calculation_l124_124445

def phi (m n : ℝ) : ℝ :=
if m < n then 2 * real.sqrt m + real.sqrt n else 2 * real.sqrt m - real.sqrt n

theorem phi_calculation : phi 3 2 - phi 8 12 = -5 * real.sqrt 2 :=
by
  sorry

end phi_calculation_l124_124445


namespace remainder_division_Q_l124_124735

noncomputable def Q_rest : Polynomial ℝ := -(Polynomial.X : Polynomial ℝ) + 125

theorem remainder_division_Q (Q : Polynomial ℝ) :
  Q.eval 20 = 105 ∧ Q.eval 105 = 20 →
  ∃ R : Polynomial ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 105) * R + Q_rest :=
by sorry

end remainder_division_Q_l124_124735


namespace sequence_general_term_l124_124261

theorem sequence_general_term (a : ℕ → ℚ)
  (h1 : a 1 = 3 / 2)
  (h2 : a 2 = 1)
  (h3 : a 3 = 5 / 8)
  (h4 : a 4 = 3 / 8) :
  a = λ n, (n^2 - 11*n + 34) / 16 := sorry

end sequence_general_term_l124_124261


namespace incorrect_statement_l124_124313

-- Definitions of conditions based on the given problem
def condA : Prop := ∃ (model real_model : Type) (f : model → real_model), ∃ (x y : model), f x ≠ f y
def condB : Prop := ∀ (R2 : ℝ), ∀ (fit : ℝ → Prop), (fit R2 ∧ R2 > 0) → (R2₁ R2₂ : ℝ), (R2₁ > R2₂) → fit R2₁
def condC : Prop := ∀ (s₁ s₂ : ℝ), (s₁ < s₂) ↔ (s₁ ≠ s₂) ∧ (s₁ < s₂ ∨ s₁ > s₂)
def condD : Prop := ∀ (R2₁ R2₂ : ℝ), (R2₁ > R2₂) → (s₁ s₂ : ℝ), (s₁ < s₂) → ¬(R2₁ > R2₂)

-- The problem: proving the incorrect statement
theorem incorrect_statement : condA ∧ condB ∧ condC ∧ condD → "D" := sorry

end incorrect_statement_l124_124313


namespace cube_sum_l124_124014

theorem cube_sum (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 :=
by
  sorry

end cube_sum_l124_124014


namespace total_distance_in_12_hours_l124_124081

-- Definition of the speed function
def speed (t : ℝ) : ℝ := 35 + t^2

-- Statement of the theorem
theorem total_distance_in_12_hours : ∫ (t : ℝ) in 0..12, speed t = 996 := by
  sorry

end total_distance_in_12_hours_l124_124081


namespace grape_candies_count_l124_124526

theorem grape_candies_count
    (x : ℕ) -- number of cherry candies
    (total_cost : ℝ)
    (cost_per_candy : ℝ) 
    (total_candies : ℕ) 
    (grape_candies : ℕ) 
    (apple_candies : ℕ)
    (h_cost : total_cost = 200)
    (h_price : cost_per_candy = 2.50)
    (h_total_candies : total_candies = floor (total_cost / cost_per_candy))
    (h_grape_candies : grape_candies = 3 * x)
    (h_apple_candies : apple_candies = 6 * grape_candies)
    (h_sum_candies : x + grape_candies + apple_candies = total_candies)
    : grape_candies = 24 :=
by
  sorry

end grape_candies_count_l124_124526


namespace valentino_farm_total_birds_l124_124750

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end valentino_farm_total_birds_l124_124750


namespace fourth_term_of_sequence_l124_124986

theorem fourth_term_of_sequence :
  let a1 := 2^(1/2)
  let a2 := 2^(1/4)
  let a3 := 2^(1/8)
  let r := 2^(-1/4)
  let a4 := a3 * r
  a4 = 2^(-1/8) :=
by
  -- Definitions of terms
  let a1 := 2^(1/2)
  let a2 := 2^(1/4)
  let a3 := 2^(1/8)
  let r := 2^(-1/4)
  let a4 := a3 * r
  -- Proof of the fourth term
  show a4 = 2^(-1/8) from sorry

end fourth_term_of_sequence_l124_124986


namespace problem1_problem2_l124_124779

theorem problem1 : (-2:ℚ)^2 - 2021^0 + ((-1/2):ℚ)^(-2) = 7 := by
  sorry

theorem problem2 : ((3 * 10^2:ℚ)^2) * ((-2 * 10^3:ℚ)^3) = -7.2 * 10^(14) := by
  sorry

end problem1_problem2_l124_124779


namespace foci_of_ellipse_l124_124552

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 169) = 1

-- Define the major and minor axes lengths
def a := sqrt 169
def b := sqrt 25

-- Calculate the distance to the foci
def c := sqrt (a * a - b * b)

-- Prove that the foci are at (0, c) and (0, -c)
theorem foci_of_ellipse :
  (c = 12) →
  (foci := [(0, c), (0, -c)]) →
  (foci = [(0, 12), (0, -12)]) :=
by
  sorry

end foci_of_ellipse_l124_124552


namespace youngest_member_age_l124_124391

def avg_age_family (n : ℕ) (sum_ages : ℕ) : ℝ := (sum_ages : ℝ) / (n : ℝ)
def present_age_youngest_member : ℕ := 5

theorem youngest_member_age (family_avg_age : ℕ) (member_count : ℕ) (remaining_members_avg_age : ℕ) : 
  present_age_youngest_member = 5 :=
by
  have h1 : member_count = 7 := rfl
  have h2 : family_avg_age = 29 := rfl
  have h3 : remaining_members_avg_age = 28 := rfl
  have sum_family_ages : ℕ := 7 * 29
  have sum_remaining_members_age : ℕ := 6 * 28
  have sum_remaining_members_now : ℕ := sum_remaining_members_age + 6 * present_age_youngest_member
  have total_sum_ages := sum_family_ages
  have equation : total_sum_ages = sum_remaining_members_now + present_age_youngest_member := (
by
  simp [sum_family_ages, sum_remaining_members_now]
  sorry
  )
  sorry

end youngest_member_age_l124_124391


namespace find_a_l124_124010

-- define the necessary mathematical objects and properties

noncomputable def f (x : ℝ) : ℝ := 3 + (Real.log x / Real.log 3)

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → (x ≤ y → f x ≤ f y)

-- define the function g
def g (x : ℝ) : ℝ :=
  f x - 2* (1 / (x * Real.log 3)) - 3

theorem find_a (a : ℕ) (x0 : ℝ) (h_monotonic : is_monotonic f)
  (h_cond : ∀ x : ℝ, 0 < x → f (f x - Real.log x / Real.log 3) = 4)
  (h_sol : g x0 = 0) (h_interval : x0 ∈ Set.Ioo a (a + 1)) :
  a = 2 :=
sorry

end find_a_l124_124010


namespace carlos_books_in_june_l124_124978

def books_in_july : ℕ := 28
def books_in_august : ℕ := 30
def goal_books : ℕ := 100

theorem carlos_books_in_june :
  let books_in_july_august := books_in_july + books_in_august
  let books_needed_june := goal_books - books_in_july_august
  books_needed_june = 42 := 
by
  sorry

end carlos_books_in_june_l124_124978


namespace problem_l124_124346

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x - 1) else -f (-x)

theorem problem (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_def : ∀ x : ℝ, x ≥ 0 → f x = x * (x - 1)) :
  f (-2) = -2 :=
by
  sorry

end problem_l124_124346


namespace cistern_width_l124_124938

theorem cistern_width :
  ∃ (w : ℝ), 
    (let l := 12 in
     let d := 1.25 in
     let total_wet_surface_area := 88 in
     total_wet_surface_area = (l * w) + 2 * (d * l) + 2 * (d * w)) ∧
    w = 4 :=
begin
  sorry
end

end cistern_width_l124_124938


namespace weight_of_new_person_l124_124394

theorem weight_of_new_person (avg_increase : ℝ) (original_weight : ℝ) (num_persons : ℕ) (new_weight : ℝ) :
  avg_increase = 1.5 → original_weight = 65 → num_persons = 5 → 
  new_weight = original_weight + avg_increase * num_persons → 
  new_weight = 72.5 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end weight_of_new_person_l124_124394


namespace power_sum_is_two_l124_124102

theorem power_sum_is_two :
  (3 ^ (-3) ^ 0 + (3 ^ 0) ^ 4) = 2 := by
    sorry

end power_sum_is_two_l124_124102


namespace carpet_area_rounded_l124_124155

noncomputable def carpet_area_in_square_meters (length_ft : ℝ) (width_ft : ℝ) (conversion_factor : ℝ) : ℝ := 
  (length_ft * width_ft) * conversion_factor

theorem carpet_area_rounded (h1 : length_ft = 15) (h2 : width_ft = 8) (h3 : conversion_factor = 0.0929) : 
  Real.floor (carpet_area_in_square_meters length_ft width_ft conversion_factor + 0.5) = 11 := 
by 
  sorry

end carpet_area_rounded_l124_124155


namespace max_value_of_expression_l124_124728

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h_sum : x + y + z = 3) 
  (h_order : x ≥ y ∧ y ≥ z) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 :=
sorry

end max_value_of_expression_l124_124728


namespace increasing_sequence_range_of_a_l124_124574

theorem increasing_sequence_range_of_a (a : ℝ) (a_n : ℕ → ℝ) (h : ∀ n : ℕ, a_n n = a * n ^ 2 + n) (increasing : ∀ n : ℕ, a_n (n + 1) > a_n n) : 0 ≤ a :=
by
  sorry

end increasing_sequence_range_of_a_l124_124574


namespace opposite_of_83_is_84_l124_124937

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l124_124937


namespace product_of_primes_sum_ten_l124_124423

theorem product_of_primes_sum_ten :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ Prime p1 ∧ Prime p2 ∧ p1 + p2 = 10 ∧ p1 * p2 = 21 := 
by
  sorry

end product_of_primes_sum_ten_l124_124423


namespace platform_length_400_l124_124480
-- Import the necessary library

-- Define the given conditions
variables (train_length : ℕ) (time_to_cross_platform : ℕ) (time_to_cross_signal : ℕ)

-- Set the given specific values
def train_length := 500
def time_to_cross_platform := 45
def time_to_cross_signal := 25

-- The main statement to prove
theorem platform_length_400 
  (train_length_eq : train_length = 500)
  (time_to_cross_platform_eq : time_to_cross_platform = 45)
  (time_to_cross_signal_eq : time_to_cross_signal = 25) :
  ∃ (platform_length : ℕ), platform_length = 400 :=
by
  -- Here the proof will go, but it's skipped using sorry.
  sorry

end platform_length_400_l124_124480


namespace trigonometry_identity_l124_124621

theorem trigonometry_identity (α : ℝ) (P : ℝ × ℝ) (h : P = (4, -3)) :
  let x := P.1
  let y := P.2
  let r := Real.sqrt (x^2 + y^2)
  x = 4 →
  y = -3 →
  r = 5 →
  Real.tan α = y / x := by
  intros x y r hx hy hr
  rw [hx, hy]
  simp [Real.tan, div_eq_mul_inv, mul_comm]
  sorry

end trigonometry_identity_l124_124621


namespace similar_pentagon_area_l124_124359

theorem similar_pentagon_area
  (K1 K2 : ℝ) (L1 L2 : ℝ)
  (h_similar : true)  -- simplifying the similarity condition as true for the purpose of this example
  (h_K1 : K1 = 18)
  (h_K2 : K2 = 24)
  (h_L1 : L1 = 8.4375) :
  L2 = 15 :=
by
  sorry

end similar_pentagon_area_l124_124359


namespace problem_l124_124211

noncomputable def t : ℝ := Real.cbrt 4

theorem problem (t : ℝ) : 4 * log 3 t = log 3 (4 * t) → t = Real.cbrt 4 :=
by
  intro h
  have htlog4 := eq_of_mul_log_four_eq_log_four_times h
  have ht : t^3 = 4
  from htlog4
  exact eq_cbrt_of_pow3_eq_four ht
end

end problem_l124_124211


namespace total_spokes_is_60_l124_124522

def num_spokes_front : ℕ := 20
def num_spokes_back : ℕ := 2 * num_spokes_front
def total_spokes : ℕ := num_spokes_front + num_spokes_back

theorem total_spokes_is_60 : total_spokes = 60 :=
by
  sorry

end total_spokes_is_60_l124_124522


namespace reduced_price_per_dozen_bananas_l124_124120

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price_one_banana (P : ℝ) := 0.60 * P
noncomputable def number_bananas_original (P : ℝ) := 40 / P
noncomputable def number_bananas_reduced (P : ℝ) := 40 / (0.60 * P)
noncomputable def difference_bananas (P : ℝ) := (number_bananas_reduced P) - (number_bananas_original P)

theorem reduced_price_per_dozen_bananas 
  (P : ℝ) 
  (h1 : difference_bananas P = 67) 
  (h2 : P = 16 / 40.2) :
  12 * reduced_price_one_banana P = 2.856 :=
sorry

end reduced_price_per_dozen_bananas_l124_124120


namespace probability_no_adjacent_birches_l124_124496

theorem probability_no_adjacent_birches :
  let total_arrangements := Nat.choose 14 6,
      favorable_arrangements := Nat.choose 7 6,
      constrained_arrangements := Nat.choose 12 6,
      probability := favorable_arrangements / constrained_arrangements in
  probability == (1 / 132) :=
begin
  sorry
end

end probability_no_adjacent_birches_l124_124496


namespace prob_all_successful_pairs_l124_124828

theorem prob_all_successful_pairs (n : ℕ) : 
  let num_pairs := (2 * n)!
  let num_successful_pairs := 2^n * n!
  prob_all_successful_pairs := num_successful_pairs / num_pairs 
  prob_all_successful_pairs = (2^n * n!) / (2 * n)! := 
sorry

end prob_all_successful_pairs_l124_124828


namespace find_prime_power_solutions_l124_124551

theorem find_prime_power_solutions (p n m : ℕ) (hp : Nat.Prime p) (hn : n > 0) (hm : m > 0) 
  (h : p^n + 144 = m^2) :
  (p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 27) :=
by sorry

end find_prime_power_solutions_l124_124551


namespace square_transformation_l124_124245

theorem square_transformation :
  let O := (0, 0)
  let A := (1, 0)
  let B := (1, 1)
  let C := (0, 1)
  let transformation (x y : ℝ) := (x^2 - y^2, 2 * x * y)
  let O' := transformation 0 0
  let A' := transformation 1 0
  let B' := transformation 1 1
  let C' := transformation 0 1
  O' = (0, 0) ∧ A' = (1, 0) ∧ B' = (0, 2) ∧ C' = (-1, 0) :=
by {
  dsimp [O, A, B, C, transformation],
  split; refl,
  split; refl,
  split; refl,
  split; refl
}

end square_transformation_l124_124245


namespace lucas_scores_l124_124022

theorem lucas_scores :
  let scores := [92, 75, 71, 94, 93]
  ∃ (scores : list ℕ), 
  scores.length = 5 ∧ 
  (list.sum scores) / 5 = 85 ∧ 
  ∀ x ∈ scores, x < 95 ∧ 
  (finset.univ : finset ℕ).card = scores.nodup.card ∧ 
  scores.sort_eq [94, 93, 92, 75, 71] := 
by
  sorry

end lucas_scores_l124_124022


namespace ellipse_nec_but_not_suff_l124_124128

-- Definitions and conditions
def isEllipse (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c

/-- Given that the sum of the distances from a moving point P in the plane to two fixed points is constant,
the condition is necessary but not sufficient for the trajectory of the moving point P being an ellipse. -/
theorem ellipse_nec_but_not_suff (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (c : ℝ) :
  (∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) →
  (c > dist F1 F2 → ¬ isEllipse P F1 F2) ∧ (isEllipse P F1 F2 → ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) :=
by
  sorry

end ellipse_nec_but_not_suff_l124_124128


namespace find_x_l124_124587

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l124_124587


namespace maximize_economic_benefit_l124_124057

theorem maximize_economic_benefit:
  ∃ x : ℕ, x = 80 ∧ 
    (320 - x ≥ 240) ∧ 
    (y = (320 - x) * (20 + 0.2 * x) - 6 * x) ∧ 
    (y = 9160) :=
by {
  use 80,
  split,
  { refl },
  split,
  { linarith },
  split,
  { simp only [*, mul_add, mul_comm (0.2 : ℝ)], linarith },
  sorry
}

end maximize_economic_benefit_l124_124057


namespace josie_money_left_l124_124335

noncomputable def milk_cost : ℝ := 3.80
noncomputable def milk_discount : ℝ := 0.85
noncomputable def bread_cost : ℝ := 4.25
noncomputable def detergent_cost : ℝ := 11.50
noncomputable def detergent_coupon : ℝ := 2.00
noncomputable def banana_cost_per_pound : ℝ := 0.95
noncomputable def banana_weight : ℝ := 3
noncomputable def eggs_cost : ℝ := 2.80
noncomputable def chicken_cost : ℝ := 8.45
noncomputable def chicken_discount : ℝ := 0.80
noncomputable def apples_cost : ℝ := 6.30
noncomputable def loyalty_card_discount : ℝ := 0.10
noncomputable def sales_tax : ℝ := 0.08
noncomputable def initial_money : ℝ := 75.00

theorem josie_money_left :
  let milk_final_cost := milk_cost * milk_discount,
      bread_final_cost := bread_cost + (bread_cost * 0.50),
      detergent_final_cost := detergent_cost - detergent_coupon,
      banana_final_cost := banana_cost_per_pound * banana_weight,
      chicken_final_cost := chicken_cost * chicken_discount,
      total_cost := milk_final_cost + bread_final_cost + detergent_final_cost + banana_final_cost + eggs_cost + chicken_final_cost + apples_cost,
      total_cost_after_loyalty := total_cost * (1 - loyalty_card_discount),
      total_cost_after_tax := total_cost_after_loyalty * (1 + sales_tax),
      final_money_left := initial_money - total_cost_after_tax
  in final_money_left = 38.25 := sorry

end josie_money_left_l124_124335


namespace proof_math_problem_l124_124103

-- Definitions based on the conditions
def pow_zero (x : ℝ) : x^0 = 1 := by
  rw [pow_zero]
  exact one

def three_pow_neg_three := (3 : ℝ)^(-3)
def three_pow_zero := (3 : ℝ)^0

-- The final statement to be proven
theorem proof_math_problem :
  (three_pow_neg_three)^0 + (three_pow_zero)^4 = 2 :=
by
  simp [three_pow_neg_three, three_pow_zero, pow_zero]
  sorry

end proof_math_problem_l124_124103


namespace storage_space_calc_correct_l124_124965

noncomputable def storage_space_available 
    (second_floor_total : ℕ)
    (box_space : ℕ)
    (one_quarter_second_floor : ℕ)
    (first_floor_ratio : ℕ) 
    (second_floor_ratio : ℕ) : ℕ :=
  if (box_space = one_quarter_second_floor ∧
      first_floor_ratio = 2 ∧
      second_floor_ratio = 1) then
    let total_building_space := second_floor_total + (first_floor_ratio * second_floor_total)
    in total_building_space - box_space
  else 0

theorem storage_space_calc_correct :
  storage_space_available 20000 5000 5000 2 1 = 55000 :=
by sorry

end storage_space_calc_correct_l124_124965


namespace prob_neg2_leq_X_leq_2_l124_124597

noncomputable def normal_distribution (μ σ : ℝ) : ProbabilityMassFunction ℝ := sorry

variable {X : ℝ}
variable {σ : ℝ}
variable {a : ℝ}

-- Condition: X ~ N(0, σ^2)
axiom axiom_normal : X ∼ normal_distribution 0 σ^2
-- Condition: P(x > 2) = a and 0 < a < 1
axiom axiom_prob_gt_2 : 0 < a ∧ a < 1 ∧ P(X > 2) = a

theorem prob_neg2_leq_X_leq_2 : P(-2 ≤ X ∧ X ≤ 2) = 1 - 2 * a :=
by 
  sorry

end prob_neg2_leq_X_leq_2_l124_124597


namespace number_of_correct_conclusions_l124_124262

theorem number_of_correct_conclusions :
  let triangle_abc : Type := {AC : ℝ // AC = 2} 
  ∧ ∀ (B : ℝ) (C : ℝ), B = π / 2 ∧ C = π / 6 
  ∧ ∃ (D : ℝ), D ∈ interval 0 2 ∧ abs (D - 2) = 1 
  ∧ (∀ (E : ℝ), E ∈ interval 0 D → E ∈ interval 0 (D - 1/sqrt 3)) 
  ∧ (∃ (x y : ℝ), linear_relation x y 0.85 85.71 ∧ height_increment_weight 1 0.85)
  ∧ (∀ (corr: ℝ), corr ∈ interval (-1) 1 → abs corr = 1 ↔ strong_correlation corr)
  ∧ (probability_one_odd_one_even : ℝ) = 1 / 2 →
  number_of_correct_conclusions = 3 := 
begin
  sorry
end

end number_of_correct_conclusions_l124_124262


namespace find_m_l124_124284

variables {a1 a2 b1 b2 c1 c2 : ℝ} {m : ℝ}
def vectorA := (3, -2 * m)
def vectorB := (m - 1, 2)
def vectorC := (-2, 1)
def vectorAC := (5, -2 * m - 1)

theorem find_m (h : (5 * (m - 1) + (-2 * m - 1) * 2) = 0) : 
  m = 7 := 
  sorry

end find_m_l124_124284


namespace vanaspati_percentage_l124_124314

theorem vanaspati_percentage (Q : ℝ) (PureGhee Vanaspati : ℝ) (AddedPureGhee : ℝ) (FinalStrengthVanaspati : ℝ):
  Q = 10 → 
  PureGhee = 0.60 * Q →
  Vanaspati = 0.40 * Q →
  AddedPureGhee = 10 →
  FinalStrengthVanaspati = 0.20 →
  (Vanaspati / Q * 100) = 40 :=
by
  intros hQ hPure hVanaspati hAdded hFinalStrength
  have h0 : 0.40 * Q = 2 := by
    rw [←hVanaspati, hQ] 
    norm_num
  have h1 : Q = 10 := by
    linarith
  have h2 : Vanaspati = 0.40 * 10 := by
    rw [hQ, hVanaspati]
    norm_num
  have h3 : Vanaspati = 4 := by
    rw h2
    norm_num
  have h4 : Vanaspati / Q * 100 = 4 / 10 * 100 := by
    rw [hVanaspati, hQ]
  have h5 : 4 / 10 * 100 = 40 := by
    norm_num
  exact h5

end vanaspati_percentage_l124_124314


namespace total_cost_of_purchases_l124_124953

theorem total_cost_of_purchases :
  let almonds_kg := 1.5
  let walnuts_kg := 1
  let cashews_kg := 0.5
  let raisins_kg := 1
  let apricots_kg := 1.5
  let almonds_price_per_kg := 12
  let walnuts_price_per_kg := 10
  let cashews_price_per_kg := 20
  let raisins_price_per_kg := 8
  let apricots_price_per_kg := 6
  let almonds_cost := almonds_kg * almonds_price_per_kg
  let walnuts_cost := walnuts_kg * walnuts_price_per_kg
  let cashews_cost := cashews_kg * cashews_price_per_kg
  let raisins_cost := raisins_kg * raisins_price_per_kg
  let apricots_cost := apricots_kg * apricots_price_per_kg
  let total_cost := almonds_cost + walnuts_cost + cashews_cost + raisins_cost + apricots_cost
  total_cost = 55 :=
by
  let almonds_kg := 1.5
  let walnuts_kg := 1
  let cashews_kg := 0.5
  let raisins_kg := 1
  let apricots_kg := 1.5
  let almonds_price_per_kg := 12
  let walnuts_price_per_kg := 10
  let cashews_price_per_kg := 20
  let raisins_price_per_kg := 8
  let apricots_price_per_kg := 6
  let almonds_cost := almonds_kg * almonds_price_per_kg
  let walnuts_cost := walnuts_kg * walnuts_price_per_kg
  let cashews_cost := cashews_kg * cashews_price_per_kg
  let raisins_cost := raisins_kg * raisins_price_per_kg
  let apricots_cost := apricots_kg * apricots_price_per_kg
  let total_cost := almonds_cost + walnuts_cost + cashews_cost + raisins_cost + apricots_cost
  -- Proof goes here.
  sorry

end total_cost_of_purchases_l124_124953


namespace diametrically_opposite_number_is_84_l124_124893

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l124_124893


namespace new_average_l124_124957

def original_sum : ℝ := 82 * 60
def new_sum : ℝ := original_sum - (95 + 97)
def new_avg : ℝ := new_sum / 58

theorem new_average (h : original_sum = 4920 ∧ new_sum = 4728) : new_avg = 81.52 := by
  sorry

end new_average_l124_124957


namespace square_side_length_correct_l124_124074

noncomputable def side_length_square_equals_area_circle (s : ℝ) (π : ℝ) : Prop :=
  let r := s / 2 in
  let area_circle := π * r^2 in
  let perimeter_square := 4 * s in
  perimeter_square = area_circle → s = 16 / π

axiom pi_value : ∃ π: ℝ, π = real.pi

theorem square_side_length_correct : ∀ (s π : ℝ), side_length_square_equals_area_circle s π :=
begin
  intros s π,
  assume h,
  rw side_length_square_equals_area_circle at h,
  rw ←pi_value at h,
  sorry
end

end square_side_length_correct_l124_124074


namespace dante_coconuts_left_l124_124033

variable (Paolo : ℕ) (Dante : ℕ)

theorem dante_coconuts_left :
  Paolo = 14 →
  Dante = 3 * Paolo →
  Dante - 10 = 32 :=
by
  intros hPaolo hDante
  rw [hPaolo, hDante]
  sorry

end dante_coconuts_left_l124_124033


namespace positive_solution_count_l124_124407

-- Define the predicate for positive integer solutions
def positive_integer_solution (x y : ℕ) : Prop :=
  2 * x + 3 * y = 763 ∧ x > 0 ∧ y > 0

-- Define the total number of solutions
noncomputable def number_of_solutions : ℕ :=
  {t : ℤ | 
    let x := 380 + 3 * t in 
    let y := 1 - 2 * t in 
    2 * x + 3 * y = 763 ∧ x > 0 ∧ y > 0}.to_finset.card

-- The theorem statement
theorem positive_solution_count : number_of_solutions = 127 :=
sorry

end positive_solution_count_l124_124407


namespace closest_approach_time_and_distance_l124_124080

-- Definitions based on the problem conditions
def equilateral_triangle (a b c : ℝ) : Prop := a = b ∧ b = c

-- Given initial conditions
def side_length : ℝ := 60 -- side length in mm
def speed1 : ℝ := 4 -- speed of bug 1 in mm/s
def speed2 : ℝ := 3 -- speed of bug 2 in mm/s

-- Proof of the minimum time and distance where bugs are closest to each other
theorem closest_approach_time_and_distance :
  (∀ (a b c : ℝ), equilateral_triangle a b c → a = side_length) →
  ∃ t d : ℝ, t = 300 / 37 ∧ d = sqrt (43200 / 37) :=
by 
  intros h_triangle
  sorry

end closest_approach_time_and_distance_l124_124080


namespace isothermal_work_correct_l124_124145

noncomputable def isothermal_compression_work (H R p₀ h: ℝ) : ℝ := 
  let V₀ := π * R^2 * H
  let c := p₀ * V₀
  c * Real.log (H / (H - h))

theorem isothermal_work_correct :
  ∀ (H R p₀ h: ℝ),
  H = 1.5 →
  R = 0.4 →
  p₀ = 10330 →
  h = 1.2 →
  isothermal_compression_work H R p₀ h ≈ 12533.3 :=
begin
  intros H R p₀ h H_eq R_eq p₀_eq h_eq,
  rw [H_eq, R_eq, p₀_eq, h_eq],
  sorry
end

end isothermal_work_correct_l124_124145


namespace smallest_n_such_that_Qn_lt_1_div_2024_l124_124384

/-- Let Q(n) be the probability that Victoria stops upon drawing exactly n marbles.
    Prove that the smallest n such that Q(n) < 1/2024 is 26.
    - There are 2024 boxes in a line.
    - Each box contains exactly one blue marble.
    - The k-th box contains 2k green marbles.
    - Victoria draws marbles sequentially. She stops when she draws a blue marble.
    - Q(n) is given by a specific probability calculation.
-/
theorem smallest_n_such_that_Qn_lt_1_div_2024 :
  ∃ (n : ℕ), 1 ≤ n ∧ 2 * n - 1 ≤ 4047 ∧
    (let Q_n := (finset.range (n - 1)).prod (λ k, (2 * (k + 1) : ℚ) / (2 * (k + 1) + 1))
             * (1 / (2 * n + 1 : ℚ))
     in Q_n < 1 / 2024) :=
sorry

end smallest_n_such_that_Qn_lt_1_div_2024_l124_124384


namespace total_people_on_boats_l124_124477

-- Definitions based on the given conditions
def boats : Nat := 5
def people_per_boat : Nat := 3

-- Theorem statement to prove the total number of people on boats in the lake
theorem total_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l124_124477


namespace sum_of_six_consecutive_integers_l124_124537

theorem sum_of_six_consecutive_integers (m : ℤ) : 
  (m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) = 6 * m + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l124_124537


namespace find_b_value_l124_124317

/-- Given a line segment from point (0, b) to (8, 0) with a slope of -3/2, 
    prove that the value of b is 12. -/
theorem find_b_value (b : ℝ) : (8 - 0) ≠ 0 ∧ ((0 - b) / (8 - 0) = -3/2) → b = 12 := 
by
  intro h
  sorry

end find_b_value_l124_124317


namespace Mia_studied_fraction_l124_124025

-- Define the conditions
def total_minutes_per_day := 1440
def time_spent_watching_TV := total_minutes_per_day * 1 / 5
def time_spent_studying := 288
def remaining_time := total_minutes_per_day - time_spent_watching_TV
def fraction_studying := time_spent_studying / remaining_time

-- State the proof goal
theorem Mia_studied_fraction : fraction_studying = 1 / 4 := by
  sorry

end Mia_studied_fraction_l124_124025


namespace arithmetic_sequence_ratio_l124_124011

variable {α : Type}
variable [LinearOrderedField α]

def a1 (a_1 : α) : Prop := a_1 ≠ 0 
def a2_eq_3a1 (a_1 a_2 : α) : Prop := a_2 = 3 * a_1 

noncomputable def common_difference (a_1 a_2 : α) : α :=
  a_2 - a_1

noncomputable def S (n : ℕ) (a_1 d : α) : α :=
  n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio
  (a_1 a_2 : α)
  (h₀ : a1 a_1)
  (h₁ : a2_eq_3a1 a_1 a_2) :
  (S 10 a_1 (common_difference a_1 a_2)) / (S 5 a_1 (common_difference a_1 a_2)) = 4 := 
by
  sorry

end arithmetic_sequence_ratio_l124_124011


namespace difference_of_modified_sums_l124_124852

theorem difference_of_modified_sums :
  let evens := (List.range (3001)).map (λ n => 2 * n + 3) in
  let odds := (List.range (3001)).map (λ n => 2 * n + 1 - 3) in
  (evens.sum - odds.sum) = 26007 :=
by
  let evens := (List.range (3001)).map (λ n => 2 * n + 3)
  let odds := (List.range (3001)).map (λ n => 2 * n + 1 - 3)
  sorry

end difference_of_modified_sums_l124_124852


namespace algebraic_expression_value_l124_124561

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 1 + Real.sqrt 2) (h2 : b = Real.sqrt 3) : 
  a^2 + b^2 - 2 * a + 1 = 5 := 
by
  sorry

end algebraic_expression_value_l124_124561


namespace solution_set_of_inequality_l124_124218

theorem solution_set_of_inequality :
  {x : ℝ | 1 < |1 - x| ∧ |1 - x| ≤ 2} = (set.Ico (-1 : ℝ) 0) ∪ (set.Ioc 2 3) :=
by {
  sorry
}

end solution_set_of_inequality_l124_124218


namespace sum_of_powers_inequality_l124_124097

theorem sum_of_powers_inequality (a : ℕ → ℕ) (n : ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j) :
  (∑ k in finset.range n, (a k)^7 + (a k)^5) ≥ 2 * (∑ k in finset.range n, (a k)^3)^2 := by
  sorry

end sum_of_powers_inequality_l124_124097


namespace quadratic_root_k_l124_124546

theorem quadratic_root_k (k : ℝ) :
  (∀ x : ℂ, x^2 * 8 + x * 4 + k = 0 ↔ x = (-4 + complex.I * (real.sqrt 380)) / 16 ∨ 
           x = (-4 - complex.I * (real.sqrt 380)) / 16) →
  k = 12.375 :=
by
  intro h
  sorry

end quadratic_root_k_l124_124546


namespace scientific_notation_of_170000_l124_124389

-- Define the concept of scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (x = a * 10^n)

-- The main statement to prove
theorem scientific_notation_of_170000 : is_scientific_notation 1.7 5 170000 :=
by sorry

end scientific_notation_of_170000_l124_124389


namespace trigonometric_product_l124_124186

theorem trigonometric_product :
  (1 - Real.sin (Real.pi / 12)) * 
  (1 - Real.sin (5 * Real.pi / 12)) * 
  (1 - Real.sin (7 * Real.pi / 12)) * 
  (1 - Real.sin (11 * Real.pi / 12)) = 1 / 4 :=
by sorry

end trigonometric_product_l124_124186


namespace CarlosBooksInJune_l124_124975

def CarlosBooksInJuly := 28
def CarlosBooksInAugust := 30
def CarlosTotalBooksGoal := 100

theorem CarlosBooksInJune :
  ∃ x : ℕ, x = CarlosTotalBooksGoal - (CarlosBooksInJuly + CarlosBooksInAugust) :=
begin
  use 42,
  dsimp [CarlosTotalBooksGoal, CarlosBooksInJuly, CarlosBooksInAugust],
  norm_num,
  sorry
end

end CarlosBooksInJune_l124_124975


namespace decimal_equivalence_l124_124062

theorem decimal_equivalence : 4 + 3 / 10 + 9 / 1000 = 4.309 := 
by
  sorry

end decimal_equivalence_l124_124062


namespace find_k_and_q_l124_124599

noncomputable def a_n (k : ℕ) (n : ℕ) : ℕ :=
if k = 1 then 6 * n - 3 else if k = 2 then 5 * n - 4 else 0

def S_n (k : ℕ) (n : ℕ) : ℕ :=
n * (a_n k 1 + a_n k n) / 2

def T_3 (q : ℝ) : ℝ :=
1 + q + q^2

theorem find_k_and_q : 
    ∃ k ∈ ℕ, 
    (a_k k = k^2 + 2) ∧ (a_2k k = (k+2)^2) ∧
    (∀ n, if k = 1 then a_n k n = 6 * n - 3 else if k = 2 then a_n k n = 5 * n - 4 else True) ∧
    (a_1 k > 1) ∧ 
    (∃ m ∈ ℕ, m > 0 ∧ m = 1 ∨ m = 2) ∧ 
    ((S_n k 2 / S_n k m) = T_3 (q := (sqrt 13 - 1) / 2)) := 
begin
  sorry
end

end find_k_and_q_l124_124599


namespace maria_sandwich_count_l124_124356

open Nat

noncomputable def numberOfSandwiches (meat_choices cheese_choices topping_choices : Nat) :=
  (choose meat_choices 2) * (choose cheese_choices 2) * (choose topping_choices 2)

theorem maria_sandwich_count : numberOfSandwiches 12 11 8 = 101640 := by
  sorry

end maria_sandwich_count_l124_124356


namespace divisibility_by_120_l124_124773

theorem divisibility_by_120 (n : ℕ) : 120 ∣ (n^7 - n^3) :=
sorry

end divisibility_by_120_l124_124773


namespace positive_solution_x_l124_124275

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 8 - x - 4 * y) (h2 : y * z = 12 - 3 * y - 6 * z) (h3 : x * z = 40 - 5 * x - 2 * z) (hy : y = 3) (hz : z = -1) : x = 6 :=
by
  sorry

end positive_solution_x_l124_124275


namespace hyperbola_eccentricity_l124_124272

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 -> ∃ (F1 F2 A B : ℝ × ℝ), 
    ∀ (M : ℝ × ℝ), M ∈ 
    { M : ℝ × ℝ | ∃ (k : ℝ), (M.2 = (b / a) * M.1 \/ M.2 = - (b / a) * M.1) /\ (M.1^2 + M.2^2 = c^2) } ∧ 
    angle_between_points M A B = π / 6):
  (c = sqrt(7) * a / sqrt(3)) -> ( c / a = sqrt(21) / 3 ) :=
by
  sorry

end hyperbola_eccentricity_l124_124272


namespace max_value_theorem_l124_124250

noncomputable def max_value_expr (a b c: ℝ) (λ : ℝ) (x1 x2 x3 : ℝ) : ℝ :=
  (2 * a ^ 3 + 27 * c - 9 * a * b) / λ^3

theorem max_value_theorem :
  ∀ (a b c λ x1 x2 x3 : ℝ),
    0 < λ →
    x2 = x1 + λ →
    x3 > (x1 + x2) / 2 →
    x1 + x2 + x3 = -a →
    x1 * x2 + x2 * x3 + x3 * x1 = b →
    x1 * x2 * x3 = -c →
    max_value_expr a b c λ x1 x2 x3 = 3 * sqrt 3 / 2 :=
by
  intro a b c λ x1 x2 x3 hλ hx2 hx3 ha hb hc
  sorry

end max_value_theorem_l124_124250


namespace opposite_of_83_is_84_l124_124930

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l124_124930


namespace contrapositive_cos_identity_l124_124865

theorem contrapositive_cos_identity (x y : ℝ) : 
  (x = y → cos x = cos y) → (cos x ≠ cos y → x ≠ y) :=
by
  intro h h1 h2
  apply h1
  exact h h2

#check contrapositive_cos_identity

end contrapositive_cos_identity_l124_124865


namespace eval_floor_ceil_sum_l124_124207

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem eval_floor_ceil_sum : floor (-3.67) + ceil 34.7 = 31 := by
  sorry

end eval_floor_ceil_sum_l124_124207


namespace faithful_line_existence_l124_124149

-- Define a faithful line for a triangle
def is_faithful_line (d : Line) (ABC : Triangle) : Prop :=
  ∃ (H : Point), H ∈ orthocenter(ABC) ∧ d ⟷ H

-- Define the main theorem
theorem faithful_line_existence (ABC DEF : Triangle) (H_ABC H_DEF : Point) 
    (H_ABC_orth : H_ABC ∈ orthocenter(ABC)) 
    (H_DEF_orth : H_DEF ∈ orthocenter(DEF))
    (acute_ABC : ∀ (a b c : Angle), a ∈ angles(ABC) → b ∈ angles(ABC) → c ∈ angles(ABC) → acute a ∧ acute b ∧ acute c)
    (acute_DEF : ∀ (d e f : Angle), d ∈ angles(DEF) → e ∈ angles(DEF) → f ∈ angles(DEF) → acute d ∧ acute e ∧ acute f) 
    (same_plane : ∀ (p q r s t u : Point), p ∈ vertices(ABC) → q ∈ vertices(ABC) → r ∈ vertices(ABC) → s ∈ vertices(DEF) → t ∈ vertices(DEF) → u ∈ vertices(DEF) → coplanar p q r s t u):
  (H_ABC ≠ H_DEF → ∃! d : Line, is_faithful_line(d, ABC) ∧ is_faithful_line(d, DEF)) ∨ 
  (H_ABC = H_DEF → ∃ d : Line, is_faithful_line(d, ABC) ∧ is_faithful_line(d, DEF) ∧ ∀ d' : Line, d' ≠ d → is_faithful_line(d', ABC) ∧ is_faithful_line(d', DEF)) :=
begin
  sorry,
end

end faithful_line_existence_l124_124149


namespace question_1_part_1_question_1_part_2_question_2_l124_124268

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * (sin x) * (cos x) + (cos x) ^ 2 + 1

theorem question_1_part_1 : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∃ T > 0, T = π :=
sorry

theorem question_1_part_2 : 
  (∀ x k, f (x) = sin (2 * x + π / 6) + 3 / 2 → 
          ((π / 6 + k * π) ≤ x ∧ x ≤ (2 * π / 3 + k * π) ↔ k ∈ ℤ)) :=
sorry

variables {A B C : ℝ} {a b c : ℝ}

theorem question_2 
  (h1 : f C = 2)
  (h2 : a + b = 4)
  (h3 : (1 / 2) * a * b * (sin C) = sqrt 3 / 3)
  : (∃ R, R = (a * b * c) / (4 * (sqrt 3 / 4)) ∧ R = 2) :=
sorry

end question_1_part_1_question_1_part_2_question_2_l124_124268


namespace number_of_candies_l124_124471

-- Define variables for the conditions
def tickets_whack_a_mole := 2
def tickets_skee_ball := 13
def ticket_cost_per_candy := 3

-- Define the total number of tickets won
def total_tickets := tickets_whack_a_mole + tickets_skee_ball

-- Theorem to prove the number of candies that can be bought
theorem number_of_candies (h1 : tickets_whack_a_mole = 2) (h2 : tickets_skee_ball = 13) (h3 : ticket_cost_per_candy = 3) :
  total_tickets / ticket_cost_per_candy = 5 :=
by
  rw [total_tickets, h1, h2, h3]
  exact rfl

-- Since the proof is omitted, sorry is used

end number_of_candies_l124_124471


namespace cost_of_six_dvds_l124_124093

variable (cost_per_two_dvds : ℕ)
variable (cost_per_one_dvd : ℕ)

axiom cost_of_two_dvds (h1 : cost_per_two_dvds = 36) 
    : cost_per_one_dvd = cost_per_two_dvds / 2

theorem cost_of_six_dvds (h1 : cost_per_two_dvds = 36) 
    (h2 : cost_per_one_dvd = cost_per_two_dvds / 2)
    : 6 * cost_per_one_dvd = 108 := 
by 
    have cost_one : cost_per_one_dvd = 18 := by 
        rw [h2, h1]
        exact (36 / 2)
    rw [cost_one]
    exact (6 * 18)

end cost_of_six_dvds_l124_124093


namespace find_x_l124_124593

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l124_124593


namespace exponential_function_explicit_formula_l124_124626

noncomputable def f (x : ℝ) := a^x

theorem exponential_function_explicit_formula 
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f 2 = 4) : 
  ∀ x, f x = 2^x :=
by
  sorry

end exponential_function_explicit_formula_l124_124626


namespace angle_AEB_is_39_l124_124310

variables {A B C D E : Type} [IsRegularPolygon A B C D 20]

theorem angle_AEB_is_39 (h1 : dist A E = dist D E)
                        (h2 : ∠ B E C = 2 * ∠ C E D) : 
  ∠ A E B = 39 :=
by
  sorry

end angle_AEB_is_39_l124_124310


namespace split_bill_equally_l124_124223

theorem split_bill_equally :
  let hamburger_cost := 3
  let hamburger_count := 5
  let fries_cost := 1.20
  let fries_count := 4
  let soda_cost := 0.50
  let soda_count := 5
  let spaghetti_cost := 2.70
  let friend_count := 5
  let total_cost := (hamburger_cost * hamburger_count) + (fries_cost * fries_count) + (soda_cost * soda_count) + spaghetti_cost
  in total_cost / friend_count = 5 := 
by
  sorry

end split_bill_equally_l124_124223


namespace ironed_clothing_count_l124_124374

theorem ironed_clothing_count : 
  (4 * 2 + 5 * 3) + (3 * 3 + 4 * 2) + (2 * 1 + 3 * 1) = 45 := by
  sorry

end ironed_clothing_count_l124_124374


namespace sequence_properties_l124_124620

/-- The arithmetic sequence with first term 2 and sum of first three terms equal to 12 --/
def arithmetic_sequence (n : ℕ) : ℕ :=
  2 + (n - 1) * 2

/-- Sum of the first n terms of the arithmetic sequence --/
def sum_of_sequence (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sequence_properties :
  (arithmetic_sequence 1 = 2) ∧
  (arithmetic_sequence 1 + arithmetic_sequence 2 + arithmetic_sequence 3 = 12) ∧
  (∀ n, arithmetic_sequence n = 2 * n) ∧
  (∀ n, sum_of_sequence n = n * (n + 1)) ∧
  (Σ n in Finset.range 11, 1 / sum_of_sequence n = 10/11) :=
by
  sorry

end sequence_properties_l124_124620


namespace percentage_saved_is_approx_11_l124_124159

theorem percentage_saved_is_approx_11 :
  ∃ original_price, original_price = 36 + 4.50 ∧ (4.50 / original_price) * 100 ≈ 11 :=
by
  sorry

end percentage_saved_is_approx_11_l124_124159


namespace planes_perpendicular_l124_124475

variables {b c : Type} [Line b] [Line c]
variables {α β : Type} [Plane α] [Plane β]

theorem planes_perpendicular (h1 : c ∥ α) (h2 : c ⊥ β) : α ⊥ β :=
sorry

end planes_perpendicular_l124_124475


namespace complex_multiplication_result_l124_124253

theorem complex_multiplication_result (i : ℂ) (p q : ℤ) (h : i^2 = -1) :
  let z := (1 + i) * (1 - i)
  in (z = p + q * i) → p + q = 2 :=
by sorry

end complex_multiplication_result_l124_124253


namespace projections_concyclic_l124_124732

noncomputable def concyclicity (A B C D A' C' B' D' : Point) :=
  let A' = orthogonal_projection A B D
  let C' = orthogonal_projection C B D
  let B' = orthogonal_projection B A C
  let D' = orthogonal_projection D A C
  affine_span_circumscribes : ∀ (G : affine_subspace ℝ V), is_regular_polygon G 4 → ∀ (w : ℝ), w > 0 → affine_line.basic (A', B', C', D',G,w)

-- The main theorem
theorem projections_concyclic (A B C D A' C' B' D' : Point) :
  (are_cyclic_points A B C D) → 
  (A' = orthogonal_projection A B D) → 
  (C' = orthogonal_projection C B D) → 
  (B' = orthogonal_projection B A C) → 
  (D' = orthogonal_projection D A C) → 
  are_cyclic_points A' B' C' D' :=
begin
  sorry
end

end projections_concyclic_l124_124732


namespace congruent_circumcircles_of_triangles_ABD_and_MNC_l124_124367

-- Defining the setup as described in the conditions
variables {A B C D M N : Point}
variable [circumcenter_triangle_ABC : Circumcenter ABC D]

-- Assume D is the circumcenter of triangle ABC.
@[instance]
def circumcenter_triangle_ABC : Prop := (∀ A B C D : Point, circumcenter ABC D)

-- The circle passing through points A, B, and D intersects AC and BC at M and N respectively
def circle_through_ABD_intersects_AC_at_M_and_BC_at_N : Prop :=
  circle_through A B D ∧ intersects AC M ∧ intersects BC N

-- The theorem statement to prove the congruence of circumcircles
theorem congruent_circumcircles_of_triangles_ABD_and_MNC :
  circumcenter_triangle_ABC A B C D →
  circle_through_ABD_intersects_AC_at_M_and_BC_at_N A B D M N →
  circumcircle A B D = circumcircle M N C :=
by 
  intros h1 h2
  sorry

end congruent_circumcircles_of_triangles_ABD_and_MNC_l124_124367


namespace smaller_angle_at_3_40_l124_124113

theorem smaller_angle_at_3_40 : 
  ∀ (minute_angle hour_angle : ℕ) (minute_position hour_position : ℕ), 
  minute_angle = 240 → 
  hour_angle = 110 → 
  |minute_angle - hour_angle| = 130 :=
by
  intros minute_angle hour_angle minute_position hour_position hm hh
  have h1 : minute_angle = 240 := hm
  have h2 : hour_angle = 110 := hh
  rw [h1, h2]
  norm_num
  sorry

end smaller_angle_at_3_40_l124_124113


namespace mary_flour_l124_124024

-- Define the constants given in the problem
def total_flour : ℕ := 11
def total_sugar : ℕ := 7
def extra_flour_needed : ℕ := 2

-- Define the number of cups of flour Mary has already put in
def cups_of_flour_already_put_in (F : ℕ) : Prop :=
F + (extra_flour_needed + total_sugar) = total_flour

-- Prove that Mary has already put in 2 cups of flour
theorem mary_flour : ∃ F, cups_of_flour_already_put_in F ∧ F = 2 :=
by
  existsi 2
  split
  . unfold cups_of_flour_already_put_in
    rw [Nat.add_assoc, Nat.add_comm extra_flour_needed, ← Nat.add_assoc]
    exact Nat.add_left_cancel_iff.mpr (rfl)
  . rfl

end mary_flour_l124_124024


namespace max_real_roots_l124_124175

noncomputable definition polynomial_max_distinct_real_roots (P : Polynomial ℝ) : Prop :=
∀ x y ∈ P.roots, x ≠ y → x * y ∈ P.roots

theorem max_real_roots {P : Polynomial ℝ} (h : polynomial_max_distinct_real_roots P) : 
  P.roots.to_finset.card ≤ 4 :=
sorry

end max_real_roots_l124_124175


namespace tan_periodic_solution_l124_124987

theorem tan_periodic_solution 
  (a b : ℝ)
  (h_period : ∀ x, real.tan(2 * x) = real.tan(x) * 2)
  (h_point : a * real.tan(b * (π / 4)) = 3)
  (h_period_condition : b = 2) : 
  a * b = 6 :=
by
  sorry

end tan_periodic_solution_l124_124987


namespace hundreds_digit_of_factorial_difference_l124_124854

theorem hundreds_digit_of_factorial_difference : 
  (20! % 1000 = 0) → (25! % 1000 = 0) → (((25! - 20!) / 100) % 10) = 0 :=
by
  intros h1 h2
  sorry

end hundreds_digit_of_factorial_difference_l124_124854


namespace successful_pairs_probability_expected_successful_pairs_l124_124838

-- Part (a)
theorem successful_pairs_probability {n : ℕ} : 
  (2 * n)!! = ite (n = 0) 1 ((2 * n) * (2 * n - 2) * (2 * n - 4) * ... * 2) →
  (fact (2 * n) / (2 ^ n * fact n)) = (2 ^ n * fact n / fact (2 * n)) :=
sorry

-- Part (b)
theorem expected_successful_pairs {n : ℕ} :
  (∑ k in range n, (if (successful_pair k) then 1 else 0)) / n > 0.5 :=
sorry

end successful_pairs_probability_expected_successful_pairs_l124_124838


namespace triangle_angle_60_l124_124434

theorem triangle_angle_60 (A B C A1 C1 I : Type) (hA1 : A1 = bisector A B C)
  (hC1 : C1 = bisector C A B) (hI : I = intersection A1 C1)
  (h_angle_AIC1 : angle A I C1 = 60) :
  (angle A B C = 60) ∨ (angle B A C = 60) ∨ (angle B C A = 60) :=
by
  sorry

end triangle_angle_60_l124_124434


namespace lemon_juice_calculation_l124_124702

noncomputable def lemon_juice_per_lemon (table_per_dozen : ℕ) (dozens : ℕ) (lemons : ℕ) : ℕ :=
  (table_per_dozen * dozens) / lemons

theorem lemon_juice_calculation :
  lemon_juice_per_lemon 12 3 9 = 4 :=
by
  -- proof would be here
  sorry

end lemon_juice_calculation_l124_124702


namespace competition_score_l124_124671

theorem competition_score
    (x : ℕ)
    (h1 : 20 ≥ x)
    (h2 : 5 * x - (20 - x) = 70) :
    x = 15 :=
sorry

end competition_score_l124_124671


namespace exists_circle_with_n_grid_points_l124_124378

theorem exists_circle_with_n_grid_points (n : ℕ) (hn : 0 < n) :
  ∃ (c : ℝ × ℝ) (r : ℝ), (∃ m : ℕ, (1 ≤ m) ∧ m = n) ∧
    (set.count {p : ℤ × ℤ | real.sqrt ((p.1 - c.1) ^ 2 + (p.2 - c.2) ^ 2) ≤ r} = n) :=
sorry

end exists_circle_with_n_grid_points_l124_124378


namespace limit_exists_and_uniform_l124_124000

variables {X : Type} [metric_space X]
variable {f : X → X}
hypothesis is_isometry : ∀ x y : X, dist (f x) (f y) = dist x y

theorem limit_exists_and_uniform :
  (∃ L : ℝ, ∀ x : X, tendsto (λ n, dist x (f^[n] x) / n) at_top (𝓝 L)) ∧
  (∃ L : ℝ, ∀ x y : X, tendsto (λ n, dist x (f^[n] x) / n) at_top (𝓝 L) ∧ tendsto (λ n, dist y (f^[n] y) / n) at_top (𝓝 L)) :=
by {
  sorry
}

end limit_exists_and_uniform_l124_124000


namespace scientific_notation_integer_l124_124947

theorem scientific_notation_integer (x : ℝ) (h1 : x > 10) :
  ∃ (A : ℝ) (N : ℤ), (1 ≤ A ∧ A < 10) ∧ x = A * 10^N :=
by
  sorry

end scientific_notation_integer_l124_124947


namespace gcd_of_powers_l124_124187

theorem gcd_of_powers (a b c : ℕ) (h1 : a = 2^105 - 1) (h2 : b = 2^115 - 1) (h3 : c = 1023) :
  Nat.gcd a b = c :=
by sorry

end gcd_of_powers_l124_124187


namespace evaluate_expression_l124_124043

theorem evaluate_expression (x y : ℤ) (h1 : x = 1) (h2 : y = -2) :
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y) * (2 * x - 3 * y)) / (-2 * y) = -16 :=
by
  subst h1
  subst h2
  norm_num
  sorry

end evaluate_expression_l124_124043


namespace part_I_part_II_l124_124575

   -- Definition of the function f
   def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 1 - a * log x

   -- Conditions and proof for Part I
   theorem part_I (a : ℝ) : 
     (∀ x : ℝ, f x a = 0 → x = 1) →
     (a ≤ 0 ∨ a = 2) := by
     sorry

   -- Definition of the inequality for Part II
   def g (x : ℝ) : ℝ := exp (x - 1) + x^2 - x - 1

   -- Conditions and proof for Part II
   theorem part_II (a : ℝ) : 
     (∀ x ≥ 1, f x a ≤ g x) →
     (0 ≤ a) := by
     sorry
   
end part_I_part_II_l124_124575


namespace solve_equation_l124_124777

theorem solve_equation : 
  ∀ x : ℝ, ((x = 1) ↔ (x^2 + 2 * x + 3) / (x^2 - 1) = x + 3) := 
by
  intro x
  split
  · intro h
    rw [h]
    calc
      (1 : ℝ)^2 + 2 * 1 + 3 = 1 + 2 * 1 + 3 := by norm_num
      ...(1 + 2 + 3) / (1^2 - 1) = (1 + 3) : by norm_num
  · intro H
    sorry

end solve_equation_l124_124777


namespace find_original_price_l124_124045

noncomputable def original_price (P : ℝ) : Prop :=
  0.90 * P = 9

theorem find_original_price : ∃ (P : ℝ), original_price P ∧ P = 10 :=
by
  use 10
  split
  . -- Validation of the condition
    unfold original_price
    exact rfl
  . -- P is indeed 10
    exact rfl

end find_original_price_l124_124045


namespace find_distance_CD_l124_124409

theorem find_distance_CD :
  let C := (0, 0)
  let D := (6, 6 * Real.sqrt 2)
  let distance := Real.sqrt ((6 - 0)^2 + (6 * Real.sqrt 2 - 0)^2)
  C = (0, 0) ∧ D = (6, 6 * Real.sqrt 2) ∧ distance = 6 * Real.sqrt 3 := 
by
  have hC : C = (0, 0) := rfl
  have hD : D = (6, 6 * Real.sqrt 2) := rfl
  have hDist : distance = Real.sqrt (6^2 + (6 * Real.sqrt 2)^2) := 
    by rw [<- sub_zero 6, <- sub_zero (6 * Real.sqrt 2), Real.dist_eq]
  rw [distance, hDist]
  have hDistVal : distance = 6 * Real.sqrt 3 := 
    by norm_num [Real.sqrt, Real.pow_succ, Real.pow_mul]
  exact ⟨hC, hD, hDistVal⟩

end find_distance_CD_l124_124409


namespace opposite_number_l124_124914

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l124_124914


namespace number_of_ways_subsets_l124_124429

def set_T : Set ℕ := {1, 2, 3, 4, 5, 6}
def number_of_ways (A B C : Set ℕ) : ℕ :=
  if A ∪ B ∪ C = set_T ∧ (A ∩ B ∩ C).card = 3 then 1 else 0

theorem number_of_ways_subsets : ∑ (A B C : Set ℕ), number_of_ways A B C = 6860 := 
  by
    sorry

end number_of_ways_subsets_l124_124429


namespace combined_average_age_l124_124044

def num_people_room_X : ℕ := 8
def avg_age_room_X : ℕ := 30
def num_people_room_Y : ℕ := 6
def avg_age_room_Y : ℕ := 45

def total_age_room_X : ℕ := num_people_room_X * avg_age_room_X
def total_age_room_Y : ℕ := num_people_room_Y * avg_age_room_Y
def combined_total_age : ℕ := total_age_room_X + total_age_room_Y
def total_num_people : ℕ := num_people_room_X + num_people_room_Y

theorem combined_average_age : combined_total_age / total_num_people = 36.5 := by
  sorry

end combined_average_age_l124_124044


namespace problem_1_problem_2_problem_3_l124_124200

-- Non-computational definitions and conditions for the problem
noncomputable def deck := {card // card.1 ∈ {1, 2, ..., 10} ∧ card.2 ∈ {"Hearts", "Spades", "Diamonds", "Clubs"}}

def drawing_heart (card : deck) : Prop := card.2 = "Hearts"
def drawing_spade (card : deck) : Prop := card.2 = "Spades"
def drawing_red_card (card : deck) : Prop := card.2 ∈ {"Hearts", "Diamonds"}
def drawing_black_card (card : deck) : Prop := card.2 ∈ {"Spades", "Clubs"}
def number_multiple_of_5 (card : deck) : Prop := card.1 % 5 = 0
def number_greater_than_9 (card : deck) : Prop := card.1 > 9

-- Theorems to be proven
theorem problem_1 : 
  (∀ card : deck, drawing_heart card → ¬drawing_spade card) ∧
  ¬(∀ card : deck, drawing_heart card ∨ drawing_spade card) :=
sorry

theorem problem_2 :
  (∀ card : deck, drawing_red_card card → ¬drawing_black_card card) ∧
  (∀ card : deck, drawing_red_card card ∨ drawing_black_card card) :=
sorry 

theorem problem_3 :
  ¬(∀ card : deck, number_multiple_of_5 card → ¬number_greater_than_9 card) ∧
  ¬(∀ card : deck, number_multiple_of_5 card ∨ number_greater_than_9 card) := 
sorry

end problem_1_problem_2_problem_3_l124_124200


namespace num_people_on_boats_l124_124478

-- Definitions based on the conditions
def boats := 5
def people_per_boat := 3

-- Theorem stating the problem to be solved
theorem num_people_on_boats : boats * people_per_boat = 15 :=
by sorry

end num_people_on_boats_l124_124478


namespace probability_winning_ticket_l124_124760

theorem probability_winning_ticket :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    (1 ≤ x1 ∧ x1 ≤ 49) ∧ 
    (1 ≤ x2 ∧ x2 ≤ 49) ∧ 
    (1 ≤ x3 ∧ x3 ≤ 49) ∧ 
    (1 ≤ x4 ∧ x4 ≤ 49) ∧ 
    (1 ≤ x5 ∧ x5 ≤ 49) ∧ 
    (1 ≤ x6 ∧ x6 ≤ 49) ∧ 
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧ 
     x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧ 
     x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧ 
     x4 ≠ x5 ∧ x4 ≠ x6 ∧ 
     x5 ≠ x6) ∧
    (log 10 x1 + log 10 x2 + log 10 x3 + log 10 x4 + log 10 x5 + log 10 x6 ∈ ℤ) ∧
    (x1 + x2 + x3 + x4 + x5 + x6) % 2 = 0) →
  (∃ (w1 w2 w3 w4 w5 w6 : ℕ),
    (1 ≤ w1 ∧ w1 ≤ 49) ∧ 
    (1 ≤ w2 ∧ w2 ≤ 49) ∧ 
    (1 ≤ w3 ∧ w3 ≤ 49) ∧ 
    (1 ≤ w4 ∧ w4 ≤ 49) ∧ 
    (1 ≤ w5 ∧ w5 ≤ 49) ∧ 
    (1 ≤ w6 ∧ w6 ≤ 49) ∧ 
    (w1 ≠ w2 ∧ w1 ≠ w3 ∧ w1 ≠ w4 ∧ w1 ≠ w5 ∧ w1 ≠ w6 ∧ 
     w2 ≠ w3 ∧ w2 ≠ w4 ∧ w2 ≠ w5 ∧ w2 ≠ w6 ∧ 
     w3 ≠ w4 ∧ w3 ≠ w5 ∧ w3 ≠ w6 ∧ 
     w4 ≠ w5 ∧ w4 ≠ w6 ∧ 
     w5 ≠ w6) ∧
    (log 10 w1 + log 10 w2 + log 10 w3 + log 10 w4 + log 10 w5 + log 10 w6 ∈ ℤ) ∧
    (w1 + w2 + w3 + w4 + w5 + w6) % 2 = 0 ∧
    set_of ({x1, x2, x3, x4, x5, x6} = {w1, w2, w3, w4, w5, w6})) →
  1/3 :=
sorry

end probability_winning_ticket_l124_124760


namespace Sarah_collected_40_today_l124_124118

noncomputable def Sarah_yesterday : ℕ := 50
noncomputable def Lara_yesterday : ℕ := Sarah_yesterday + 30
noncomputable def Lara_today : ℕ := 70
noncomputable def Total_yesterday : ℕ := Sarah_yesterday + Lara_yesterday
noncomputable def Total_today : ℕ := Total_yesterday - 20
noncomputable def Sarah_today : ℕ := Total_today - Lara_today

theorem Sarah_collected_40_today : Sarah_today = 40 := 
by
  sorry

end Sarah_collected_40_today_l124_124118


namespace opposite_of_83_is_84_l124_124934

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l124_124934


namespace f_at_8_l124_124266

noncomputable def f : ℝ → ℝ
| x => if x < 0 then x^3 - 1
       else if -1 ≤ x && x ≤ 1 then 
         if 0 < x then by sorry else -f(-x)
       else if x > 1 / 2 then fun y => y

theorem f_at_8 : f(8) = 2 := 
sorry

end f_at_8_l124_124266


namespace mr_valentino_birds_l124_124755

theorem mr_valentino_birds : 
  ∀ (chickens ducks turkeys : ℕ), 
  chickens = 200 → 
  ducks = 2 * chickens → 
  turkeys = 3 * ducks → 
  chickens + ducks + turkeys = 1800 :=
by
  intros chickens ducks turkeys
  assume h1 : chickens = 200
  assume h2 : ducks = 2 * chickens
  assume h3 : turkeys = 3 * ducks
  sorry

end mr_valentino_birds_l124_124755


namespace logan_money_left_l124_124021

-- Defining the given conditions
def income : ℕ := 65000
def rent_expense : ℕ := 20000
def groceries_expense : ℕ := 5000
def gas_expense : ℕ := 8000
def additional_income_needed : ℕ := 10000

-- Calculating total expenses
def total_expense : ℕ := rent_expense + groceries_expense + gas_expense

-- Desired income
def desired_income : ℕ := income + additional_income_needed

-- The theorem to prove
theorem logan_money_left : (desired_income - total_expense) = 42000 :=
by
  -- A placeholder for the proof
  sorry

end logan_money_left_l124_124021


namespace evaluate_expression_l124_124109

-- any_nonzero_num_pow_zero condition
lemma any_nonzero_num_pow_zero (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  -- exponentiation rule for nonzero numbers to the power of zero
  sorry

-- num_to_zero_power condition
lemma num_to_zero_power : 3^0 = 1 := by
  -- exponentiation rule for numbers to the zero power
  exact (Nat.cast_pow 3 0).trans (nat.pow_zero 3).symm

theorem evaluate_expression : (3^(-3))^0 + (3^0)^4 = 2 := by
  have h1 : (3^(-3))^0 = 1 := any_nonzero_num_pow_zero (3^(-3)) (by linarith [pow_ne_zero (-3) (ne_of_gt (by norm_num : 3 > 0))]),
  have h2 : (3^0)^4 = 1 := by
    rw num_to_zero_power,
    norm_num,
  linarith

end evaluate_expression_l124_124109


namespace axis_of_symmetry_axis_is_3_l124_124019

variable {α : Type*} [AddGroup α]

/-- Proof of axis of symmetry -/
theorem axis_of_symmetry (f : α → ℝ) (h : ∀ x, f x = f (6 - x)) : ∀ x, f (x + 3) = f (3 - x) :=
by
  intro x
  have h1 : f (x + 3) = f (6 - (x + 3)) := h (x + 3)
  rw [sub_add_eq_sub_sub, add_sub_cancel' 6 3] at h1
  exact h1
#align axis_of_symmetry

-- Theorem statement for the axis of symmetry being x = 3
theorem axis_is_3 {f : ℝ → ℝ} (h : ∀ x, f x = f (6 - x)) : x = 3 := 
sorry

end axis_of_symmetry_axis_is_3_l124_124019


namespace find_x_l124_124589

theorem find_x (n : ℕ) (hn : Odd n) (hx : ∃ p1 p2 p3 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 11 ∈ {p1, p2, p3} ∧ (6^n + 1) = p1 * p2 * p3) :
  6^n + 1 = 7777 :=
by
  sorry

end find_x_l124_124589


namespace cary_earnings_l124_124181

variable (shoe_cost : ℕ) (saved_amount : ℕ)
variable (lawns_per_weekend : ℕ) (weeks_needed : ℕ)
variable (total_cost_needed : ℕ) (total_lawns : ℕ) (earn_per_lawn : ℕ)
variable (h1 : shoe_cost = 120)
variable (h2 : saved_amount = 30)
variable (h3 : lawns_per_weekend = 3)
variable (h4 : weeks_needed = 6)
variable (h5 : total_cost_needed = shoe_cost - saved_amount)
variable (h6 : total_lawns = lawns_per_weekend * weeks_needed)
variable (h7 : earn_per_lawn = total_cost_needed / total_lawns)

theorem cary_earnings :
  earn_per_lawn = 5 :=
by 
  sorry

end cary_earnings_l124_124181


namespace complex_conjugate_location_l124_124354

theorem complex_conjugate_location :
  ∀ z : ℂ, (1 - complex.i) * z = complex.abs (1 + real.sqrt 3 * complex.i) → 
  (complex.re (conj z) > 0 ∧ complex.im (conj z) < 0) :=
by
  -- Reasoning omitted, only statement provided
  sorry

end complex_conjugate_location_l124_124354


namespace probability_of_selecting_balls_l124_124884

theorem probability_of_selecting_balls :
  let total_balls := 20
  let white_balls := 9
  let red_balls := 5
  let black_balls := 6
  let selected_balls := 10 in
  (3 ≤ white_balls ∧ white_balls ≤ 7) ∧
  (2 ≤ red_balls ∧ red_balls ≤ 5) ∧
  (1 ≤ black_balls ∧ black_balls ≤ 3) →
  (white_balls + red_balls + black_balls = selected_balls) →
  selected_balls = 10 →
  9 + 5 + 6 = 20 →
  let favorable_ways := 14 in
  let total_ways := 184756 in
  let probability := (favorable_ways : ℚ) / total_ways in
  probability = 7 / 92378 :=
by
  sorry

end probability_of_selecting_balls_l124_124884


namespace diametrically_opposite_number_l124_124922

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l124_124922


namespace gravel_weight_is_correct_l124_124138

def weight_of_gravel (total_weight : ℝ) (fraction_sand : ℝ) (fraction_water : ℝ) : ℝ :=
  total_weight - (fraction_sand * total_weight + fraction_water * total_weight)

theorem gravel_weight_is_correct :
  weight_of_gravel 23.999999999999996 (1 / 3) (1 / 4) = 10 :=
by
  sorry

end gravel_weight_is_correct_l124_124138


namespace compare_sqrt_l124_124983

noncomputable def a : ℝ := 3 * Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 15

theorem compare_sqrt : a > b :=
by
  sorry

end compare_sqrt_l124_124983


namespace tangent_line_to_circle_range_mn_l124_124018

theorem tangent_line_to_circle_range_mn (m n : ℝ) 
  (h1 : (m + 1) * (m + 1) + (n + 1) * (n + 1) = 4) :
  (m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end tangent_line_to_circle_range_mn_l124_124018


namespace female_managers_count_l124_124669

-- Definitions for the given conditions
def total_employees (E : ℕ) : Prop := true
def female_employees (F : ℕ) : Prop := F = 500
def fraction_managers (x : ℝ) : Prop := x = 2 / 5
def total_managers (E : ℕ) (x : ℝ) : ℕ := (x * E).natAbs
def male_employees (E : ℕ) (F : ℕ) : ℕ := E - F
def male_managers (E F : ℕ) (x : ℝ) : ℕ := (x * (E - F)).natAbs
def female_managers (E F : ℕ) (x : ℝ) : ℕ := total_managers E x - male_managers E F x

-- The theorem to prove the number of female managers is 200
theorem female_managers_count (E F : ℕ) (x : ℝ) (hF : female_employees F) (hx : fraction_managers x) (hE : total_employees E) : 
  female_managers E F x = 200 :=
  sorry

end female_managers_count_l124_124669


namespace book_distribution_l124_124547

theorem book_distribution (x : ℕ) (books : ℕ) :
  (books = 3 * x + 8) ∧ (books < 5 * x - 5 + 2) → (x = 6 ∧ books = 26) :=
by
  sorry

end book_distribution_l124_124547


namespace carlos_books_in_june_l124_124973

theorem carlos_books_in_june
  (books_july : ℕ)
  (books_august : ℕ)
  (total_books_needed : ℕ)
  (books_june : ℕ) : 
  books_july = 28 →
  books_august = 30 →
  total_books_needed = 100 →
  books_june = total_books_needed - (books_july + books_august) →
  books_june = 42 :=
by intros books_july books_august total_books_needed books_june h1 h2 h3 h4
   sorry

end carlos_books_in_june_l124_124973


namespace verify_labels_in_three_weighings_l124_124849

/--
For 13 balls with weights from 1 to 13 kg, each ball having a unique label indicating its weight, 
and a two-pan balance scale that compares the total weight of any two sets of balls, 
we can ensure in 3 weighings that all labels are correctly attached.
-/
theorem verify_labels_in_three_weighings (balls : Fin 13 → Fin 13 → ℕ) :
  (∀ permutation, (∑ i in {0,1,2,3,4,5,6,7}.toFinset, balls i permutation(i)) =
                  (∑ i in {10,11,12}.toFinset, balls i permutation(i))) →
  (∑ i in {0,1,2,8}.toFinset, balls i permutation(i)) =
  (∑ i in {6,7}.toFinset, balls i permutation(i)) →
  (∑ i in {0,3,6,8,10}.toFinset, balls i permutation(i)) =
  (∑ i in {2,5,9,12}.toFinset, balls i permutation(i)) →
  ∀ i, balls i permutation(i) = i + 1 :=
begin
  sorry
end

end verify_labels_in_three_weighings_l124_124849


namespace value_of_x_plus_y_l124_124716

-- Definitions from the conditions
def greatest_int (z : ℝ) : ℤ := Int.floor z

axiom x : ℝ
axiom y : ℝ
axiom h1 : y = 4 * ↑(greatest_int x) + 5
axiom h2 : y = 5 * ↑(greatest_int (x - 3)) + 2 * x + 7
axiom h3 : x > 3
axiom h4 : ¬(x ∈ Int)

-- The theorem to be proven
theorem value_of_x_plus_y : x + y = 32.5 :=
by
  -- Proof goes here
  sorry

end value_of_x_plus_y_l124_124716


namespace shortest_path_length_l124_124527

-- Define the unit square and vertices
structure UnitSquare where
  A B C D : Point
  unit_length : dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1

-- Definition of the right isosceles triangle with hypotenuse BD
structure IsoscelesRightTriangle where
  B D : Point
  right_angle_vertex : Point
  right_angle_at_right_angle_vertex : angle B right_angle_vertex D = pi / 2
  isosceles_legs : dist B right_angle_vertex = dist right_angle_vertex D

-- Define the conditions of the problem
variables (sq : UnitSquare) (iso_triangle : IsoscelesRightTriangle) 
  (h1 : sq.B = iso_triangle.B)
  (h2 : sq.D = iso_triangle.D)

-- Define the proof goal to show the shortest path length
theorem shortest_path_length : 
  shortest_path sq.A sq.C iso_triangle = 2 :=
sorry

end shortest_path_length_l124_124527


namespace find_k_from_polynomial_l124_124063

theorem find_k_from_polynomial :
  ∃ (k : ℝ),
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₂ * x₃ * x₄ = -1984 ∧
    x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄ = k ∧
    x₁ + x₂ + x₃ + x₄ = 18 ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32))
  → k = 86 :=
by
  sorry

end find_k_from_polynomial_l124_124063


namespace valentino_farm_birds_total_l124_124752

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end valentino_farm_birds_total_l124_124752


namespace octahedron_paths_top_to_bottom_l124_124157

-- Definitions corresponding to the conditions of the problem
inductive OctahedronFace
| top
| bottom
| face1
| face2
| face3
| face4
| face5
| face6

open OctahedronFace

-- A move is only valid if it is downwards i.e., top -> middle faces -> bottom
def valid_moves : OctahedronFace → OctahedronFace → Prop
| top, face1 => true
| top, face2 => true
| top, face3 => true
| top, face4 => true
| face1, face5 => true
| face1, face6 => true
| face2, face5 => true
| face2, face6 => true
| face3, face5 => true
| face3, face6 => true
| face4, face5 => true
| face4, face6 => true
| face5, bottom => true
| face6, bottom => true
| _, _ => false

-- Main statement to prove
theorem octahedron_paths_top_to_bottom : 
  ∃ n : ℕ, n = 8 ∧ 
  ∀ (start := top) (end := bottom), 
  (∑ final in {face1, face2, face3, face4}, 
    (∑ middle in {face5, face6}, 
      if valid_moves final middle ∧ valid_moves middle end then 1 else 0)) = n := 
  sorry

end octahedron_paths_top_to_bottom_l124_124157


namespace sum_f_g_eq_30_l124_124664

variable {R : Type} [Field R]

def f : R → R := sorry
def g : R → R := sorry

theorem sum_f_g_eq_30
  (H1 : ∀ x : R, x ∈ (Set.Univ : Set R)) -- The domain of functions f(x) and g(x) is R.
  (H2 : ∀ x : R, f(x) / g(x) = g(x+2) / f(x-2)) -- Given condition
  (H3 : f 2022 / g 2024 = 2) -- Given specific value
  : ∑ k in Finset.range 24, f (2 * k) / g (2 * k + 2) = 30 :=
sorry

end sum_f_g_eq_30_l124_124664


namespace stickers_total_l124_124770

theorem stickers_total (ryan_has : Ryan_stickers = 30)
  (steven_has : Steven_stickers = 3 * Ryan_stickers)
  (terry_has : Terry_stickers = Steven_stickers + 20) :
  Ryan_stickers + Steven_stickers + Terry_stickers = 230 :=
sorry

end stickers_total_l124_124770


namespace cannot_inscribe_rectangle_l124_124692

-- Definition of the side ratios of the outer and inner rectangles
def outer_ratio : ℚ := 9 / 16
def inner_ratio : ℚ := 4 / 7

theorem cannot_inscribe_rectangle :
  ¬ (∃ (R_outer R_inner : Type) (outer_rect inner_rect : R_outer → Prop) 
     (A B C D A1 B1 C1 D1 : R_outer),
     ∀ (x : R_outer → ℝ), outer_rect x → inner_rect x ∧
       inner_rect A1 ∧ inner_rect B1 ∧ inner_rect C1 ∧ inner_rect D1 ∧
       A1 ∈ [A, D] ∧ B1 ∈ [A, B] ∧ C1 ∈ [B, C] ∧ D1 ∈ [C, D] ∧ 
       ∃ (r_outer r_inner : ℚ), r_outer = outer_ratio ∧ r_inner = inner_ratio ∧
         (A1, B1, C1, D1 have a side ratio r_inner to the outer rectangle with ratio r_outer)) :=
by {
  sorry
}

end cannot_inscribe_rectangle_l124_124692


namespace probability_two_white_balls_is_4_over_15_l124_124484

-- Define the conditions of the problem
def total_balls : ℕ := 15
def white_balls_initial : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 2 -- Note: Even though not explicitly required, it's part of the context

-- Calculate the probability of drawing two white balls without replacement
noncomputable def probability_two_white_balls : ℚ :=
  (white_balls_initial / total_balls) * ((white_balls_initial - 1) / (total_balls - 1))

-- The theorem to prove
theorem probability_two_white_balls_is_4_over_15 :
  probability_two_white_balls = 4 / 15 := by
  sorry

end probability_two_white_balls_is_4_over_15_l124_124484


namespace vector_angle_condition_l124_124283

variable {R : Type*} [LinearOrderedField R]

def is_nonzero_vector (c : R × R) : Prop := c ≠ (0, 0)

def makes_equal_angles (c a b : R × R) : Prop :=
  (c.1 * a.1 + c.2 * a.2) * (real.sqrt ((a.1)^2 + (a.2)^2)) * 
  (real.sqrt ((c.1)^2 + (c.2)^2)) =
  (c.1 * b.1 + c.2 * b.2) * (real.sqrt ((b.1)^2 + (b.2)^2)) * 
  (real.sqrt ((c.1)^2 + (c.2)^2))

theorem vector_angle_condition (c : ℝ × ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (4, 2)) :
  is_nonzero_vector c → makes_equal_angles c a b → c.1 = c.2 :=
by
  sorry

end vector_angle_condition_l124_124283


namespace knights_in_exchange_l124_124364

noncomputable def count_knights (total_islanders : ℕ) (odd_statements : ℕ) (even_statements : ℕ) : ℕ :=
if total_islanders % 2 = 0 ∧ odd_statements = total_islanders ∧ even_statements = total_islanders then
    total_islanders / 2
else
    0

theorem knights_in_exchange : count_knights 30 30 30 = 15 :=
by
    -- proof part will go here but is not required.
    sorry

end knights_in_exchange_l124_124364


namespace problem_statement_l124_124002

noncomputable def P (x : ℕ) : ℕ := x^2016 + 2*x^2015 + ... + 2017
noncomputable def Q (x : ℕ) : ℕ := 1399*x^1398 + ... + 2*x + 1

theorem problem_statement :
  ∃ (a b : ℕ → ℕ), 
    (∀ i, b (i + 1) > b i) ∧
    (∀ i, a (i + 1) > 0) ∧
    (∀ i, gcd (a i) (a (i + 1)) = 1) ∧
    (∀ i, 
      (even i → ¬(P (b i) ∣ a i) ∧ Q (b i) ∣ a i) ∧
      (odd i → P (b i) ∣ a i ∧ ¬(Q (b i) ∣ a i))) :=
by
  sorry

end problem_statement_l124_124002


namespace kyle_money_l124_124704

theorem kyle_money (dave_money : ℕ) (kyle_initial : ℕ) (kyle_remaining : ℕ)
  (h1 : dave_money = 46)
  (h2 : kyle_initial = 3 * dave_money - 12)
  (h3 : kyle_remaining = kyle_initial - kyle_initial / 3) :
  kyle_remaining = 84 :=
by
  -- Define Dave's money and provide the assumption
  let dave_money := 46
  have h1 : dave_money = 46 := rfl

  -- Define Kyle's initial money based on Dave's money
  let kyle_initial := 3 * dave_money - 12
  have h2 : kyle_initial = 3 * dave_money - 12 := rfl

  -- Define Kyle's remaining money after spending one third on snowboarding
  let kyle_remaining := kyle_initial - kyle_initial / 3
  have h3 : kyle_remaining = kyle_initial - kyle_initial / 3 := rfl

  -- Now we prove that Kyle's remaining money is 84
  sorry -- Proof steps omitted

end kyle_money_l124_124704


namespace projected_increase_is_25_l124_124023

variable (R P : ℝ) -- variables for last year's revenue and projected increase in percentage

-- Conditions
axiom h1 : ∀ (R : ℝ), R > 0
axiom h2 : ∀ (P : ℝ), P/100 ≥ 0
axiom h3 : ∀ (R : ℝ), 0.75 * R = 0.60 * (R + (P/100) * R)

-- Goal
theorem projected_increase_is_25 (R : ℝ) : P = 25 :=
by {
    -- import the required axioms and provide the necessary proof
    apply sorry
}

end projected_increase_is_25_l124_124023


namespace sum_of_roots_quadratic_eq_m_plus_n_l124_124538

theorem sum_of_roots_quadratic_eq_m_plus_n :
  let a := 5
  let b := -11
  let c := 2
  let sum_roots := (11 + 9) / 10 + (11 - 9) / 10
  let m := 121
  let n := 5
  sum_roots = 2.2 ∧ (Int.sqrt m = 11 ∧ n = 5) → m + n = 126 :=
by
  sorry

end sum_of_roots_quadratic_eq_m_plus_n_l124_124538


namespace number_of_nines_in_n_cubed_l124_124351

noncomputable def n : ℕ := (10^2007 - 1)

theorem number_of_nines_in_n_cubed : 
    number_of_digit_in_decimal_representation (n ^ 3) 9 = 4015 := by
    sorry

end number_of_nines_in_n_cubed_l124_124351


namespace time_for_tom_to_complete_l124_124871

-- Define the rates at which Avery and Tom work respectively
def avery_rate := (1 : ℝ) / 4
def tom_rate := (1 : ℝ) / 2

-- Define the combined rate when they work together
def combined_rate := avery_rate + tom_rate

-- Define the amount of the wall completed when they work together for 1 hour
def completed_in_one_hour := combined_rate * 1

-- Define the remaining part of the wall
def remaining_wall := 1 - completed_in_one_hour

-- Prove the time it will take Tom to complete the remaining wall
theorem time_for_tom_to_complete :
  (remaining_wall / tom_rate) = 1 / 2 :=
by 
  sorry

end time_for_tom_to_complete_l124_124871


namespace average_speed_correct_l124_124462

variable (t1 t2 : ℝ) -- time components in hours
variable (v1 v2 : ℝ) -- speed components in km/h

-- conditions
def time1 := 20 / 60 -- 20 minutes converted to hours
def time2 := 40 / 60 -- 40 minutes converted to hours
def speed1 := 60 -- speed in km/h for the first segment
def speed2 := 90 -- speed in km/h for the second segment

-- total distance traveled
def distance1 := speed1 * time1
def distance2 := speed2 * time2
def total_distance := distance1 + distance2

-- total time taken
def total_time := time1 + time2

-- average speed
def average_speed := total_distance / total_time

-- proof statement
theorem average_speed_correct : average_speed = 80 := by
  sorry

end average_speed_correct_l124_124462


namespace product_sum_identity_l124_124766

theorem product_sum_identity
  (k n : ℕ) :
  (∑ i in Finset.range n, ∏ j in Finset.range k, (i + 1 + j)) = 
  (∏ i in Finset.range (k + 1), (n + k - i)) / (k + 1) := 
sorry

end product_sum_identity_l124_124766


namespace sphere_radius_eq_three_l124_124667

theorem sphere_radius_eq_three (r : ℝ) (h : 4 / 3 * π * r ^ 3 = 4 * π * r ^ 2) : r = 3 :=
by
  sorry

end sphere_radius_eq_three_l124_124667


namespace triangle_inequality_contr_l124_124761

theorem triangle_inequality_contr (A B C : Type) (a b : ℝ) 
  (sides_opposite : A -> B -> C -> Type) 
  (h: a < b) : ∃ (tri : A), (∀ (tri : B), angle_less B A → length A < length B) → false := 
begin
  assume h_geq_ab: a ≥ b,
  sorry
end

end triangle_inequality_contr_l124_124761


namespace probability_13_knowers_probability_14_knowers_expected_knowers_l124_124875

noncomputable def scientist_count : ℕ := 18
noncomputable def initial_knowers : ℕ := 10

def pairs (n : ℕ) : ℕ := n / 2

-- Part (a)
theorem probability_13_knowers : 
  ∀ (scientists : ℕ) (initial_knowers : ℕ), 
  scientists = 18 → initial_knowers = 10 →
  probability (λ (x : ℕ), x = 13) = 0 := sorry

-- Part (b)
theorem probability_14_knowers : 
  ∀ (scientists : ℕ) (initial_knowers : ℕ),
  scientists = 18 → initial_knowers = 10 →
  probability (λ (x : ℕ), x = 14) ≈ 0.461 := sorry

-- Part (c)
theorem expected_knowers : 
  ∀ (scientists : ℕ) (initial_knowers : ℕ),
  scientists = 18 → initial_knowers = 10 →
  expectation (λ (x: ℕ), x) ≈ 14.705 := sorry

end probability_13_knowers_probability_14_knowers_expected_knowers_l124_124875


namespace problem_statement_l124_124236

def f (x : ℝ) : ℝ := | x + 2 | - | 2 * x - 1 |

def M x : Prop := -1 / 3 < x ∧ x < 3

theorem problem_statement (x y : ℝ) (hx : M x) (hy : M y) : | x + y + x * y | < 15 := 
sorry

end problem_statement_l124_124236


namespace midpoint_locus_l124_124280

theorem midpoint_locus (a b : Line) (A B : Point) (P : Point) :
  skew_lines a b ∧ angle a b = 60 ∧ orthogonal_segment a b EF ∧ length EF = 2 ∧
  length (line_segment A B) = 4 ∧ A ∈ a ∧ B ∈ b →
  locus_eq P (line_segment A B) (λ P, (P.x^2 / 9) + P.y^2 = 1) :=
by {
  sorry
}

end midpoint_locus_l124_124280


namespace discriminant_square_eq_l124_124302

variable {a b c x : ℝ}

-- Condition: a ≠ 0
axiom h_a : a ≠ 0

-- Condition: x is a root of the quadratic equation ax^2 + bx + c = 0
axiom h_root : a * x^2 + b * x + c = 0

theorem discriminant_square_eq (h_a : a ≠ 0) (h_root : a * x^2 + b * x + c = 0) :
  (2 * a * x + b)^2 = b^2 - 4 * a * c :=
by 
  sorry

end discriminant_square_eq_l124_124302


namespace well_depth_l124_124950

theorem well_depth :
  (∃ t₁ t₂ : ℝ, t₁ + t₂ = 9.5 ∧ 20 * t₁ ^ 2 = d ∧ t₂ = d / 1000 ∧ d = 1332.25) :=
by
  sorry

end well_depth_l124_124950


namespace S_geq_P_imp_S_l124_124352

variables {n : ℕ} {x : ℕ → ℝ}

def S (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  (1 / (2 * n) * finset.sum (finset.range (2 * n)) (λ i, (x i + 2)^n))^(1 / n)

def S' (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  (1 / (2 * n) * finset.sum (finset.range (2 * n)) (λ i, (x i + 1)^n))^(1 / n)

def P (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  (finset.prod (finset.range (2 * n)) (λ i, x i))^(1 / n)

theorem S_geq_P_imp_S'_geq_three_fourths_P
  (n_pos : 0 < n)
  (x_pos : ∀ i, 0 < x i)
  (S_geq_P : S n x ≥ P n x) :
  S' n x ≥ (3 / 4) * P n x :=
sorry

end S_geq_P_imp_S_l124_124352


namespace circle_equation_proof_l124_124420

theorem circle_equation_proof (a b r : ℝ) :
  (∃ (a b r : ℝ), (3 - a) ^ 2 + (1 - b) ^ 2 = r^2 ∧ (1 - a) ^ 2 + (1 - b) ^ 2 = r^2 ∧ (0 - a) ^ 2 + (2 - b) ^ 2 = (2 + r) ^ 2) →
  (∃ (a b r : ℝ), a = 2 ∧ b = 0 ∧ r = real.sqrt 2) :=
sorry

end circle_equation_proof_l124_124420


namespace inequality_solution_l124_124805

theorem inequality_solution : { x : ℝ | (x - 1) / (x + 3) < 0 } = { x : ℝ | -3 < x ∧ x < 1 } :=
sorry

end inequality_solution_l124_124805


namespace divides_trans_l124_124733

theorem divides_trans (m n : ℤ) (h : n ∣ m * (n + 1)) : n ∣ m :=
by
  sorry

end divides_trans_l124_124733


namespace number_of_correct_props_is_four_l124_124605

theorem number_of_correct_props_is_four :
  (2 ≤ 3) ∧
  (∀ m : ℝ, m ≥ 0 → ∃ x : ℝ, x^2 + x - m = 0) ∧
  (∀ x y : ℝ, x^2 = y^2 → abs x = abs y) ∧
  (∀ a b c : ℝ, (a > b ↔ a + c > b + c)) →
  4 :=
by {
  sorry
}

end number_of_correct_props_is_four_l124_124605


namespace proof_sqrt_expr_sum_eq_l124_124996

noncomputable def sqrt_expr_sum_eq : Prop :=
  sqrt (25 - 10 * sqrt 6) + sqrt (25 + 10 * sqrt 6) = sqrt 60

theorem proof_sqrt_expr_sum_eq : sqrt_expr_sum_eq := 
  sorry

end proof_sqrt_expr_sum_eq_l124_124996


namespace inequality_solution_set_l124_124804

theorem inequality_solution_set (x : ℝ) : (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 :=
by sorry

end inequality_solution_set_l124_124804


namespace fraction_of_remaining_prize_money_each_winner_receives_l124_124133

-- Definitions based on conditions
def total_prize_money : ℕ := 2400
def first_winner_fraction : ℚ := 1 / 3
def each_following_winner_prize : ℕ := 160

-- Calculate the first winner's prize
def first_winner_prize : ℚ := first_winner_fraction * total_prize_money

-- Calculate the remaining prize money after the first winner
def remaining_prize_money : ℚ := total_prize_money - first_winner_prize

-- Calculate the fraction of the remaining prize money that each of the next ten winners will receive
def following_winner_fraction : ℚ := each_following_winner_prize / remaining_prize_money

-- Theorem statement
theorem fraction_of_remaining_prize_money_each_winner_receives :
  following_winner_fraction = 1 / 10 :=
sorry

end fraction_of_remaining_prize_money_each_winner_receives_l124_124133


namespace mean_days_of_exercise_l124_124308

theorem mean_days_of_exercise :
  let n := [2, 4, 2, 5, 4, 7, 3, 2] in
  let d := [0, 1, 2, 3, 4, 5, 6, 7] in
  let total_students := n.sum in
  let total_days := (List.zipWith (*) n d).sum in
  let mean_days := (total_days : ℚ) / total_students in
  Float.round (mean_days.toReal * 100) / 100 = 3.66 := 
sorry

end mean_days_of_exercise_l124_124308


namespace divisible_by_7_tail_cutting_method_l124_124995

theorem divisible_by_7_tail_cutting_method (A : ℕ) : 
  (∃ k, ∃ m : ℕ, A = 10^k * m) ↔ 
  (∃ n, ∃ B : ℕ, A = 7 * n ∧ (exists_tail_cutting_sequence B → 7 ∣ B)) :=
by sorry

end divisible_by_7_tail_cutting_method_l124_124995


namespace opposite_point_83_is_84_l124_124897

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l124_124897


namespace concurrency_circumcircles_of_trapezoid_l124_124710

variables {A B C D E F X : Type} [affine_torsor A X] {AB CD AC BD BE CF : line_segment A}
          (abc_trap : ∀ {P Q : A}, P ∈ AB ∧ Q ∈ CD → ∥P - Q∥ = ∥C - D∥)
          (e_on_ac : E ∈ AC)
          (f_on_bd : F ∈ BD)
          (be_parallel_cf : ∀ {P Q : A}, P ∈ BE ∧ Q ∈ CF → ∥P - Q∥ = ∥B - E∥)

theorem concurrency_circumcircles_of_trapezoid :
  concurrent (circumcircle (triangle A B F)) (circumcircle (triangle B E D)) AC :=
sorry

end concurrency_circumcircles_of_trapezoid_l124_124710


namespace evaluate_expression_l124_124108

-- any_nonzero_num_pow_zero condition
lemma any_nonzero_num_pow_zero (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  -- exponentiation rule for nonzero numbers to the power of zero
  sorry

-- num_to_zero_power condition
lemma num_to_zero_power : 3^0 = 1 := by
  -- exponentiation rule for numbers to the zero power
  exact (Nat.cast_pow 3 0).trans (nat.pow_zero 3).symm

theorem evaluate_expression : (3^(-3))^0 + (3^0)^4 = 2 := by
  have h1 : (3^(-3))^0 = 1 := any_nonzero_num_pow_zero (3^(-3)) (by linarith [pow_ne_zero (-3) (ne_of_gt (by norm_num : 3 > 0))]),
  have h2 : (3^0)^4 = 1 := by
    rw num_to_zero_power,
    norm_num,
  linarith

end evaluate_expression_l124_124108


namespace average_difference_l124_124059

theorem average_difference :
  let a1 := 20
  let a2 := 40
  let a3 := 60
  let b1 := 10
  let b2 := 70
  let b3 := 13
  (a1 + a2 + a3) / 3 - (b1 + b2 + b3) / 3 = 9 := by
sorry

end average_difference_l124_124059


namespace overall_gain_percent_is_0_93_l124_124663

open Real

noncomputable def cost_price_60 (cp_per_article : ℝ) : ℝ := 60 * cp_per_article
noncomputable def cost_price_120 (cp_per_article : ℝ) : ℝ := 120 * cp_per_article
noncomputable def selling_price_100 (cp_per_article : ℝ) : ℝ := 100 * (120 * cp_per_article / 100)
noncomputable def tax_first_60 (cp_per_article : ℝ) : ℝ := 0.10 * cost_price_60(cp_per_article)
noncomputable def tax_remaining_60 (cp_per_article : ℝ) : ℝ := 0.05 * cost_price_60(cp_per_article)
noncomputable def total_tax (cp_per_article : ℝ) : ℝ := tax_first_60(cp_per_article) + tax_remaining_60(cp_per_article)
noncomputable def total_cost_with_tax (cp_per_article : ℝ) : ℝ := cost_price_120(cp_per_article) + total_tax(cp_per_article)
noncomputable def discount_first_50 (sp_per_article : ℝ) : ℝ := 0.15 * 50 * sp_per_article
noncomputable def discount_remaining_50 (sp_per_article : ℝ) : ℝ := 0.08 * 50 * sp_per_article
noncomputable def total_discount (sp_per_article : ℝ) : ℝ := discount_first_50(sp_per_article) + discount_remaining_50(sp_per_article)
noncomputable def total_selling_price_after_discount (sp_per_article : ℝ) : ℝ := (120 * sp_per_article) - total_discount(sp_per_article)
noncomputable def gain_or_loss (cp_per_article : ℝ) (sp_per_article : ℝ) : ℝ := total_selling_price_after_discount(sp_per_article) - total_cost_with_tax(cp_per_article)
noncomputable def gain_or_loss_percent (cp_per_article : ℝ) (sp_per_article : ℝ) : ℝ := (gain_or_loss(cp_per_article, sp_per_article) / total_cost_with_tax(cp_per_article)) * 100

theorem overall_gain_percent_is_0_93 :
  ∀ cp_per_article sp_per_article,
    cp_per_article = 1 →
    sp_per_article = 1.2 →
    gain_or_loss_percent(cp_per_article, sp_per_article) > 0.929 ∧
    gain_or_loss_percent(cp_per_article, sp_per_article) < 0.931 :=
by {
  intros,
  sorry
}

end overall_gain_percent_is_0_93_l124_124663


namespace countUphillIntegersDivisibleBy15_l124_124969

-- Define the concept of an uphill integer
def isUphillInteger (n : ℕ) : Prop :=
  let digits := Integer.digits 10 n in
  ∀ i (h : i < digits.length - 1), digits.nthLe i h < digits.nthLe (i + 1) (by linarith)

-- Define the concept of divisibility by 15
def isDivisibleBy15 (n : ℕ) : Prop :=
  (n % 15 = 0)

-- Problem statement: The number of uphill integers within the digits 1 to 5 that are divisible by 15
theorem countUphillIntegersDivisibleBy15 :
  (finset.filter (λ n, isUphillInteger n ∧ isDivisibleBy15 n) (finset.range 54321)).card = 6 :=
  sorry

end countUphillIntegersDivisibleBy15_l124_124969


namespace solve_equation_16x_eq_256_l124_124296

theorem solve_equation_16x_eq_256 (x : ℝ) (h : 16^x = 256) : x = 2 :=
sorry

end solve_equation_16x_eq_256_l124_124296


namespace power_sum_is_two_l124_124101

theorem power_sum_is_two :
  (3 ^ (-3) ^ 0 + (3 ^ 0) ^ 4) = 2 := by
    sorry

end power_sum_is_two_l124_124101


namespace quadratic_equation_root_condition_l124_124241

theorem quadratic_equation_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, (a - 1) * x1^2 - 4 * x1 - 1 = 0 ∧ (a - 1) * x2^2 - 4 * x2 - 1 = 0) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end quadratic_equation_root_condition_l124_124241


namespace find_x_l124_124591

theorem find_x (n : ℕ) (hn : Odd n) (hx : ∃ p1 p2 p3 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 11 ∈ {p1, p2, p3} ∧ (6^n + 1) = p1 * p2 * p3) :
  6^n + 1 = 7777 :=
by
  sorry

end find_x_l124_124591


namespace number_of_throwers_l124_124746

theorem number_of_throwers (total_players throwers right_handed : ℕ) 
  (h1 : total_players = 64)
  (h2 : right_handed = 55) 
  (h3 : ∀ T N, T + N = total_players → 
  T + (2/3 : ℚ) * N = right_handed) : 
  throwers = 37 := 
sorry

end number_of_throwers_l124_124746


namespace triangle_inequality_l124_124246

open Real

variables (a b c m γ : ℝ)
variables [Triangle a b c] [Height c m] [Angle c γ]

theorem triangle_inequality:
  a + b + m ≤ (2 + cos (γ / 2)) / (2 * sin (γ / 2)) * c :=
sorry

end triangle_inequality_l124_124246


namespace closed_broken_line_covered_by_circle_l124_124369

theorem closed_broken_line_covered_by_circle :
  ∀ (broken_line : Set ℝ) (circle : Set ℝ), 
    IsClosed broken_line ∧ (∫ x in interval 0 1, length broken_line x) = 1 ∧
    (∃ r : ℝ, 0 < r ∧ r = 1 / 4 ∧ ∃ (center : ℝ), circle = metric.ball center r) →
    broken_line ⊆ circle :=
by
  sorry

end closed_broken_line_covered_by_circle_l124_124369


namespace used_crayons_l124_124375

open Nat

theorem used_crayons (N B T U : ℕ) (h1 : N = 2) (h2 : B = 8) (h3 : T = 14) (h4 : T = N + U + B) : U = 4 :=
by
  -- Proceed with the proof here
  sorry

end used_crayons_l124_124375


namespace find_a2015_l124_124306

def sequence (a : ℕ → ℝ) :=
  (a 1 = 2/3) ∧ (∀ n: ℕ, a (n + 1) - a n = real.sqrt (2/3 * (a (n + 1) + a n)))

theorem find_a2015 (a : ℕ → ℝ) (h : sequence a) : a 2015 = 1354080 := 
sorry

end find_a2015_l124_124306


namespace ratio_of_sum_and_difference_l124_124418

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (x + y) / (x - y) = x / y) : x / y = 1 + Real.sqrt 2 :=
sorry

end ratio_of_sum_and_difference_l124_124418


namespace find_principal_amount_l124_124122

theorem find_principal_amount 
  (A : ℝ) (r : ℝ) (t : ℝ) 
  (hA : A = 1120) 
  (hr : r = 0.07) 
  (ht : t = 2.4) :
  let P := A / (1 + r * t) in 
  P ≈ 958.90 :=
by
  let P := A / (1 + r * t)
  have h: P = 958.9041 := sorry -- Detailed calculation skipped
  sorry -- Proof of approximation

end find_principal_amount_l124_124122


namespace watch_cost_price_l124_124163

open Real

theorem watch_cost_price (CP SP1 SP2 : ℝ)
    (h1 : SP1 = CP * 0.85)
    (h2 : SP2 = CP * 1.10)
    (h3 : SP2 = SP1 + 450) : CP = 1800 :=
by
  sorry

end watch_cost_price_l124_124163


namespace value_of_x_squared_plus_one_l124_124297

-- Given condition
def condition (x : ℝ) : Prop := 2^(2 * x) + 4 = 12 * 2^x

-- Theorem statement
theorem value_of_x_squared_plus_one (x : ℝ) (h : condition x) : x^2 + 1 = (Real.log 6 2 + 4 * Real.log (ℝ.sqrt 2) 2)^2 + 1 :=
sorry

end value_of_x_squared_plus_one_l124_124297


namespace archer_probability_l124_124521

noncomputable def prob_hit : ℝ := 0.9
noncomputable def prob_miss : ℝ := 1 - prob_hit

theorem archer_probability :
  (prob_miss * prob_hit * prob_hit * prob_hit) = 0.0729 :=
by
  have h_prob_miss : prob_miss = 0.1 := by norm_num
  have h_prob_seq : prob_miss * prob_hit * prob_hit * prob_hit 
                    = 0.1 * 0.9 * 0.9 * 0.9 := by rw [h_prob_miss]
  rw [h_prob_seq]
  norm_num
  sorry

end archer_probability_l124_124521


namespace find_legs_of_triangle_l124_124503

theorem find_legs_of_triangle (a b : ℝ) (h : a / b = 3 / 4) (h_sum : a^2 + b^2 = 70^2) : 
  (a = 42) ∧ (b = 56) :=
sorry

end find_legs_of_triangle_l124_124503


namespace find_x_l124_124584

theorem find_x :
  ∃ n : ℕ, odd n ∧ (6^n + 1).natPrimeFactors.length = 3 ∧ 11 ∈ (6^n + 1).natPrimeFactors ∧ 6^n + 1 = 7777 :=
by sorry

end find_x_l124_124584


namespace agatha_money_left_l124_124515

theorem agatha_money_left (initial amount_spent_on_frame amount_spent_on_wheel remaining : ℤ) 
  (h1 : initial = 60)
  (h2 : amount_spent_on_frame = 15)
  (h3 : amount_spent_on_wheel = 25)
  (h4 : remaining = initial - (amount_spent_on_frame + amount_spent_on_wheel)) :
  remaining = 20 :=
by
  rw [h1, h2, h3]
  exact h4

end agatha_money_left_l124_124515


namespace probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124843

-- Definition of conditions
def num_pairs_socks : ℕ := n

def successful_pair (socks : List (ℕ × ℕ)) (k : ℕ) : Prop :=
  (socks[n - k]).fst = (socks[n - k]).snd

-- Part (a): Prove the probability that all pairs are successful is \(\frac{2^n n!}{(2n)!}\)
theorem probability_all_successful_pairs (n : ℕ) :
  ∃ p : ℚ, p = (2^n * nat.factorial n) / nat.factorial (2 * n) :=
sorry

-- Part (b): Prove the expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half (n : ℕ) :
  (n / (2 * n - 1)) > 1 / 2 :=
sorry

end probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124843


namespace block_addition_l124_124514

variable (initial_blocks : ℝ) (final_blocks : ℝ) (added_blocks : ℝ)
variable (h_initial : initial_blocks = 35.0)
variable (h_final : final_blocks = 100.0)

theorem block_addition :
  added_blocks = final_blocks - initial_blocks → added_blocks = 65.0 :=
by
  intro h
  calc
    added_blocks = final_blocks - initial_blocks := h
    ... = 100.0 - 35.0 := by rw [h_final, h_initial]
    ... = 65.0 := by norm_num

end block_addition_l124_124514


namespace sheets_required_to_print_numbers_l124_124292

/-- We are to calculate the number of sheets required to print all numbers from 1 to 1,000,000.
    Each sheet can hold 30 lines, with each line containing 60 characters. Additionally, there are
    spaces between numbers equivalent to two characters. -/
theorem sheets_required_to_print_numbers :
  let
    digits_count (n : Nat) : Nat := if n < 10 then 1
                                    else if n < 100 then 2
                                    else if n < 1000 then 3
                                    else if n < 10000 then 4
                                    else if n < 100000 then 5
                                    else if n < 1000000 then 6
                                    else 7
    total_digits : Nat := (∑ n in Finset.range (1000000 + 1), digits_count n)
    spaces_count : Nat := 999999  * 2 
    total_characters : Nat := total_digits + spaces_count
    characters_per_sheet : Nat := 30 * 60
  in
    (total_characters + characters_per_sheet - 1) / characters_per_sheet = 4383 :=
by
  sorry

end sheets_required_to_print_numbers_l124_124292


namespace compare_numbers_l124_124182

-- Definitions to match the conditions
def a : ℝ := -6.5
def b : ℝ := -6 - 3 / 5

-- Proof statement
theorem compare_numbers : a > b := 
sorry

end compare_numbers_l124_124182


namespace number_of_four_digit_numbers_l124_124495

theorem number_of_four_digit_numbers :
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ (count : ℕ), count = 1080 ∧ 
    ∀ (num : ℕ), (num ∈ digits) ∧
    (num.digits.length = 4) ∧ 
    (num.digits ∩ {2, 4, 6, 8} ≤ 1) -> num ∈ digits := sorry

end number_of_four_digit_numbers_l124_124495


namespace evaluate_expression_l124_124110

-- any_nonzero_num_pow_zero condition
lemma any_nonzero_num_pow_zero (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  -- exponentiation rule for nonzero numbers to the power of zero
  sorry

-- num_to_zero_power condition
lemma num_to_zero_power : 3^0 = 1 := by
  -- exponentiation rule for numbers to the zero power
  exact (Nat.cast_pow 3 0).trans (nat.pow_zero 3).symm

theorem evaluate_expression : (3^(-3))^0 + (3^0)^4 = 2 := by
  have h1 : (3^(-3))^0 = 1 := any_nonzero_num_pow_zero (3^(-3)) (by linarith [pow_ne_zero (-3) (ne_of_gt (by norm_num : 3 > 0))]),
  have h2 : (3^0)^4 = 1 := by
    rw num_to_zero_power,
    norm_num,
  linarith

end evaluate_expression_l124_124110


namespace bound_translations_l124_124524

open Real
open Set

noncomputable theory

variables {d : ℕ} {n : ℕ} {C : Set (EuclideanSpace ℝ d)} (v : Fin n → EuclideanSpace ℝ d)
variables (C_is_convex : Convex ℝ C)
variables (C_i : Fin n → Set (EuclideanSpace ℝ d)) (hCi : ∀ i, C_i i = C + {v i})
variables (hCi_inter_C : ∀ i, (C_i i ∩ C).Nonempty)
variables (hCi_disjoint : ∀ i j, i ≠ j → (C_i i ∩ C_i j) = ∅)

theorem bound_translations : n ≤ 3 ^ d - 1 := 
sorry

end bound_translations_l124_124524


namespace find_y_intercept_of_line_l124_124562

open Lean Meta

-- Conditions: 
-- The line passes through points (3, 2), (1, k), and (-4, 1)
def point1 := (3 : ℝ, 2 : ℝ)
def point2 (k : ℝ) := (1 : ℝ, k)
def point3 := (-4 : ℝ, 1 : ℝ)

-- Assume k is found by checking the collinearity condition
def k := 12 / 7

-- y-intercept calculation based on the collinearity and solving process
def y_intercept := 11 / 7

-- The question and the proof goal:
theorem find_y_intercept_of_line : 
  (collinear point1 (point2 k) point3) → 
  (y_intercept_of_line_passing_through point1 (point2 k) point3 = 11 / 7) := by
sorry

end find_y_intercept_of_line_l124_124562


namespace solve_equation_l124_124994

def op (a b : ℝ) : ℝ :=
  if a ≥ b then b^a else b^2

theorem solve_equation : {x : ℝ // 3 * x = 27} ∈ [{x | x = 3 ∨ x = 3 * real.sqrt 3 }] :=
by
  sorry

end solve_equation_l124_124994


namespace composite_surface_area_l124_124156

theorem composite_surface_area (p q r : ℕ) (h_prime_p : nat.prime p) (h_prime_q : nat.prime q) (h_prime_r : nat.prime r) 
  (h_volume : p * q * r = 1001) (cube_side_length_is_r : r = 13) :
  let initial_surface_area := 2 * (p * q + p * r + q * r),
      cube_surface_area := 6 * r^2,
      shared_face_area := r^2,
      total_surface_area := initial_surface_area + cube_surface_area - shared_face_area
  in total_surface_area = 1467 :=
by
  sorry

end composite_surface_area_l124_124156


namespace seq_150th_term_is_150_l124_124193

noncomputable def nth_term_of_sequence (n : ℕ) : ℕ :=
  let seq := list.range (2 ^ n) in
  seq.nth (n - 1).get_or_else 0

theorem seq_150th_term_is_150 :
  nth_term_of_sequence 150 = 150 :=
sorry

end seq_150th_term_is_150_l124_124193


namespace average_run_per_day_l124_124172

theorem average_run_per_day (n6 n7 n8 : ℕ) 
  (h1 : 3 * n7 = n6) 
  (h2 : 3 * n8 = n7) 
  (h3 : n6 * 20 + n7 * 18 + n8 * 16 = 250 * n8) : 
  (n6 * 20 + n7 * 18 + n8 * 16) / (n6 + n7 + n8) = 250 / 13 :=
by sorry

end average_run_per_day_l124_124172


namespace smallest_k_positive_cos_square_eq_one_l124_124558

theorem smallest_k_positive_cos_square_eq_one :
  ∃ k : ℕ, 0 < k ∧ cos ((k^2 + 49) * (π / 180)) = 1 ∧ k = 49 :=
by
  sorry

end smallest_k_positive_cos_square_eq_one_l124_124558


namespace even_squares_sum_even_squares_sum_formula_final_even_squares_sum_odd_squares_sum_final_odd_squares_sum_l124_124126

theorem even_squares_sum (n : ℕ) : 
  2^2 + 4^2 + ∑ k in finset.range n, (2 * (k + 1))^2 = 
  4 * ∑ k in finset.range (n+1), k^2 := by
  sorry

theorem even_squares_sum_formula (n : ℕ) : 
  2^2 + 4^2 + ∑ k in finset.range n, (2 * (k + 1))^2 = 
  4 * (n * (n + 1) * (2 * n + 1) / 6) := by
  sorry

theorem final_even_squares_sum (n : ℕ) : 
  2^2 + 4^2 + ∑ k in finset.range n, (2 * (k + 1))^2 = 
  2 * n * (n + 1) * (2 * n + 1) / 3 := by
  sorry

theorem odd_squares_sum (n : ℕ) : 
  1 + 3^2 + ∑ k in finset.range (2 * n + 1), if k % 2 = 1 then k^2 else 0 =
  (2 * n + 1) * (n + 1) * (2 * n + 3) / 3 := by
  sorry

theorem final_odd_squares_sum (n : ℕ) : 
  1 + 3^2 + ∑ k in finset.range (2 * n + 1), if k % 2 = 1 then k^2 else 0 =
  (2 * n + 1) * (n + 1) * (2 * n + 3) / 3 := by
  sorry

end even_squares_sum_even_squares_sum_formula_final_even_squares_sum_odd_squares_sum_final_odd_squares_sum_l124_124126


namespace value_of_x_l124_124390

theorem value_of_x (x : ℝ)
  (h1 : ∃ s_1 : ℝ, s_1^2 = x^2 + 12x + 36)
  (h2 : ∃ s_2 : ℝ, s_2^2 = 4x^2 - 12x + 9)
  (h3 : ∃ s_1 s_2 : ℝ, 4 * s_1 + 4 * s_2 = 64) :
  x = 13 / 3 :=
by {
  sorry
}

end value_of_x_l124_124390


namespace last_person_is_knight_l124_124703

def KnightLiarsGame1 (n : ℕ) : Prop :=
  let m := 10
  let p := 13
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

def KnightLiarsGame2 (n : ℕ) : Prop :=
  let m := 12
  let p := 9
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

theorem last_person_is_knight :
  ∃ n, KnightLiarsGame1 n ∧ KnightLiarsGame2 n :=
by 
  sorry

end last_person_is_knight_l124_124703


namespace Sam_weight_l124_124440

theorem Sam_weight :
  ∃ (sam_weight : ℕ), (∀ (tyler_weight : ℕ), (∀ (peter_weight : ℕ), peter_weight = 65 → tyler_weight = 2 * peter_weight → tyler_weight = sam_weight + 25 → sam_weight = 105)) :=
by {
    sorry
}

end Sam_weight_l124_124440


namespace probability_of_two_white_balls_correct_l124_124482

noncomputable def probability_of_two_white_balls : ℚ :=
  let total_balls := 15
  let white_balls := 8
  let first_draw_white := (white_balls : ℚ) / total_balls
  let second_draw_white := (white_balls - 1 : ℚ) / (total_balls - 1)
  first_draw_white * second_draw_white

theorem probability_of_two_white_balls_correct :
  probability_of_two_white_balls = 4 / 15 :=
by
  sorry

end probability_of_two_white_balls_correct_l124_124482


namespace find_S12_l124_124714

theorem find_S12 (S : ℕ → ℕ) (h1 : S 3 = 6) (h2 : S 9 = 15) : S 12 = 18 :=
by
  sorry

end find_S12_l124_124714


namespace max_profit_jars_max_tax_value_l124_124678

-- Part a: Prove the optimal number of jars for maximum profit
theorem max_profit_jars (Q : ℝ) 
  (h : ∀ Q, Q >= 0 → (310 - 3 * Q) * Q - 10 * Q ≤ (310 - 3 * 50) * 50 - 10 * 50):
  Q = 50 :=
sorry

-- Part b: Prove the optimal tax for maximum tax revenue
theorem max_tax_value (t : ℝ) 
  (h : ∀ t, t >= 0 → ((300 * t - t^2) / 6) ≤ ((300 * 150 - 150^2) / 6)):
  t = 150 :=
sorry

end max_profit_jars_max_tax_value_l124_124678


namespace cucumber_weight_evaporation_l124_124870

theorem cucumber_weight_evaporation :
  ∀ {W : ℝ},
    (∀ {initial_weight : ℝ}, initial_weight = 100 → 
     ∀ {initial_water_perc : ℝ}, initial_water_perc = 0.99 →
     ∀ {final_water_perc : ℝ}, final_water_perc = 0.95 →
     0.05 * W = initial_weight * (1 - initial_water_perc)) →
    W = 20 :=
by
  intros W h
  specialize h 100 rfl 0.99 rfl 0.95 rfl
  linarith

end cucumber_weight_evaporation_l124_124870


namespace original_value_of_changed_number_l124_124392

theorem original_value_of_changed_number :
  ∀ (numbers : Fin 5 → ℝ), 
  (∀ i, (numbers i) ≠ 90) → -- original numbers, none of them is 90 initially
  (∑ i, numbers i) / 5 = 70 → 
  ∃ i (new_numbers : Fin 5 → ℝ), 
  (new_numbers i = 90 ∧ 
  (∀ j, j ≠ i → new_numbers j = numbers j) ∧ 
  (∑ j, new_numbers j) / 5 = 80) → 
  (numbers (i) = 40) :=
sorry

end original_value_of_changed_number_l124_124392


namespace mean_score_remaining_students_l124_124672

theorem mean_score_remaining_students (k : ℕ) (h1 : k > 10) (h2 : (10 * 15 + (k - 10) * (8 - 8) / k) = 8) : 
  (10 * 15 + (8k - 150)) / (k - 10) = 8 := sorry

end mean_score_remaining_students_l124_124672


namespace opposite_number_l124_124916

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l124_124916


namespace tan_alpha_cases_l124_124652

theorem tan_alpha_cases (α : ℝ) (h : 2 * sin (2 * α) = 1 - cos (2 * α)) : tan α = 0 ∨ tan α = 2 :=
by
  sorry

end tan_alpha_cases_l124_124652


namespace min_value_expression_l124_124038

theorem min_value_expression (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 3) : 
  let A := (a^3 + b^3) / (8 * a * b + 9 - c^2) + 
           (b^3 + c^3) / (8 * b * c + 9 - a^2) + 
           (c^3 + a^3) / (8 * c * a + 9 - b^2) 
  in A = 3 / 8 :=
sorry

end min_value_expression_l124_124038


namespace find_rate_of_interest_l124_124173

/-- At what rate percent on simple interest will Rs. 25,000 amount to Rs. 34,500 in 5 years? 
    Given Principal (P) = Rs. 25,000, Amount (A) = Rs. 34,500, Time (T) = 5 years. 
    We need to find the Rate (R). -/
def principal : ℝ := 25000
def amount : ℝ := 34500
def time : ℝ := 5

theorem find_rate_of_interest (P A T : ℝ) : 
  P = principal → 
  A = amount → 
  T = time → 
  ∃ R : ℝ, R = 7.6 :=
by
  intros hP hA hT
  -- proof goes here
  sorry

end find_rate_of_interest_l124_124173


namespace proposition_logic_l124_124634

theorem proposition_logic {a b : ℝ} (h : a > b > 0) 
  (f : ℝ → ℝ) 
  (hef : ∀ x, f (-(x - 1)) = f (x - 1)) : 
  ¬(a > b > 0 ∧ ∀ x, f (-(x - 1)) = f (x - 1)) → (a > b > 0 ∨ ¬(∀ x, f (-(x - 1)) = f (x - 1))) :=
by
  sorry

end proposition_logic_l124_124634


namespace mass_percentage_H3O4Cl3_correct_l124_124214

variables {H_molar_mass O_molar_mass Cl_molar_mass : ℝ}
variables {H_num_atoms O_num_atoms Cl_num_atoms : ℕ}

-- Definitions for atomic masses and number of atoms
def molar_mass_H := 1.01 -- g/mol
def molar_mass_O := 16.00 -- g/mol
def molar_mass_Cl := 35.45 -- g/mol

def num_atoms_H := 3
def num_atoms_O := 4
def num_atoms_Cl := 3

-- Compound's total molar mass
def molar_mass_H3O4Cl3 := (num_atoms_H * molar_mass_H) + (num_atoms_O * molar_mass_O) + (num_atoms_Cl * molar_mass_Cl)

-- Definitions for computed mass percentages
def mass_percentage_H := (num_atoms_H * molar_mass_H / molar_mass_H3O4Cl3) * 100
def mass_percentage_O := (num_atoms_O * molar_mass_O / molar_mass_H3O4Cl3) * 100
def mass_percentage_Cl := (num_atoms_Cl * molar_mass_Cl / molar_mass_H3O4Cl3) * 100

-- Lean 4 statement for proof 
theorem mass_percentage_H3O4Cl3_correct :
  mass_percentage_H = 1.75 ∧
  mass_percentage_O = 36.92 ∧
  mass_percentage_Cl = 61.33 :=
by
  sorry

end mass_percentage_H3O4Cl3_correct_l124_124214


namespace greatest_divisible_by_five_base8_3digit_l124_124453

open Nat

theorem greatest_divisible_by_five_base8_3digit :
    ∃ n : ℕ, n.toDigits 8 = [7, 7, 6] ∧ (n ≤ 7 * 8^2 + 7 * 8^1 + 7) ∧ (5 ∣ n) :=
by
  sorry

end greatest_divisible_by_five_base8_3digit_l124_124453


namespace domain_symmetry_parity_symmetry_determination_odd_function_at_origin_l124_124397

/-- This definition ensures that f(x) is even. -/
def is_even_function {α : Type} [AddGroup α] (f : α → α) : Prop :=
∀ x, f (-x) = f x

/-- This definition ensures that f(x) is odd. -/
def is_odd_function {α : Type} [AddGroup α] (f : α → α) : Prop :=
∀ x, f (-x) = -f x

/-- Prove that the domain of a function with parity symmetry (either even or odd) is symmetric about the origin. -/
theorem domain_symmetry {α : Type} [AddGroup α] [DecidableEq α] [Inhabited α]
  (f : α → α) (h1 : is_even_function f ∨ is_odd_function f) :
  ∀ x, x ∈ (λ y, true) ↔ -x ∈ (λ y, true) :=
sorry

/-- Prove the difference between determining whether a function has parity symmetry 
(even or odd) and determining that it does not. -/
theorem parity_symmetry_determination {α : Type} [AddGroup α] [DecidableEq α] [Inhabited α]
  (f : α → α) :
  (is_even_function f ∨ is_odd_function f) ↔ ¬ (∃ x, f (-x) ≠ f x ∧ f (-x) ≠ -f x) :=
sorry

/-- Prove that if an odd function is defined at the origin, then it must be zero at the origin. -/
theorem odd_function_at_origin {α : Type} [AddGroup α] [Inhabited α]
  (f : α → α) (h : is_odd_function f) : f 0 = 0 :=
sorry

end domain_symmetry_parity_symmetry_determination_odd_function_at_origin_l124_124397


namespace work_together_days_l124_124148

theorem work_together_days (A B : ℝ) (h1 : A = 1/2 * B) (h2 : B = 1/48) :
  1 / (A + B) = 32 :=
by
  sorry

end work_together_days_l124_124148


namespace break_even_point_l124_124744

noncomputable def cost (handles : ℕ) : ℝ := 7640 + 0.60 * handles
noncomputable def revenue (handles : ℕ) : ℝ := 4.60 * handles

theorem break_even_point : ∃ (handles : ℕ), cost handles = revenue handles ∧ handles = 1910 := 
by
  use 1910
  show cost 1910 = revenue 1910
  show 1910 = 1910
  sorry

end break_even_point_l124_124744


namespace sum_of_squares_l124_124808

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 :=
  sorry

end sum_of_squares_l124_124808


namespace trig_product_computation_l124_124183

theorem trig_product_computation :
  (1 - sin (Real.pi / 12)) * (1 - sin (5 * Real.pi / 12)) *
  (1 - sin (7 * Real.pi / 12)) * (1 - sin (11 * Real.pi / 12)) 
  = 1 / 16 :=
by 
  sorry

end trig_product_computation_l124_124183


namespace solve_for_x_l124_124049

theorem solve_for_x:
  ∀ (x : ℝ), (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -(48 / 23) :=
by
  sorry

end solve_for_x_l124_124049


namespace mutually_exclusive_not_contradictory_l124_124232

-- Definitions for the problem
def red_ball : Type := {b : Type | true} 
def black_ball : Type := {b : Type | true}
def bag : Type := {b : Type | b = red_ball ∨ b = black_ball}

-- Number of each ball
def num_red_balls : Nat := 2
def num_black_balls : Nat := 3

def draw_two_balls (balls : list bag) : Prop := balls.length = 2

-- Events
def at_least_one_black_ball (balls : list bag) : Prop := balls.filter (λ b => b = black_ball).length ≥ 1
def both_are_black_balls (balls : list bag) : Prop := balls.filter (λ b => b = black_ball).length = 2
def at_least_one_red_ball (balls : list bag) : Prop := balls.filter (λ b => b = red_ball).length ≥ 1
def both_are_red_balls (balls : list bag) : Prop := balls.filter (λ b => b = red_ball).length = 2
def exactly_one_black_ball (balls : list bag) : Prop := balls.filter (λ b => b = black_ball).length = 1
def exactly_two_black_balls (balls : list bag) : Prop := balls.filter (λ b => b = black_ball).length = 2

-- Mutually exclusive but not contradictory events
theorem mutually_exclusive_not_contradictory :
  ∀ (balls : list bag), draw_two_balls balls →
  (¬ (exactly_one_black_ball balls ∧ exactly_two_black_balls balls)) ∧
  (∃ e1 e2, ¬ (exactly_one_black_ball balls ∧ exactly_two_black_balls balls) 
                        ∧ (e1 ∨ e2)) :=
sorry

end mutually_exclusive_not_contradictory_l124_124232


namespace find_angle_degree_l124_124060

theorem find_angle_degree (x : ℝ) (h : 90 - x = (1 / 3) * (180 - x) + 20) : x = 75 := by
    sorry

end find_angle_degree_l124_124060


namespace sum_of_possible_chocolates_l124_124859

theorem sum_of_possible_chocolates : 
  let possible_N := {N | (N % 6 = 5) ∧ (N % 8 = 7) ∧ (N < 100)} in
  ∑ N in possible_N, N = 236 :=
by
  let possible_N := {N | (N % 6 = 5) ∧ (N % 8 = 7) ∧ (N < 100)}
  have h : ∑ N in possible_N, N = 236 := sorry
  exact h

end sum_of_possible_chocolates_l124_124859


namespace min_value_of_A_l124_124041

noncomputable def A (a b c : ℝ) : ℝ :=
  (a^3 + b^3) / (8 * a * b + 9 - c^2) +
  (b^3 + c^3) / (8 * b * c + 9 - a^2) +
  (c^3 + a^3) / (8 * c * a + 9 - b^2)

theorem min_value_of_A (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  A a b c = 3 / 8 :=
sorry

end min_value_of_A_l124_124041


namespace arrangement_count_correct_l124_124644

-- Definitions for conditions
def valid_first_section (arrangement : List Char) : Prop :=
  ∀ (c : Char), c ∈ arrangement.take 4 → c = 'B' ∨ c = 'C' ∨ c = 'D'

def valid_middle_section (arrangement : List Char) : Prop :=
  ∀ (c : Char), c ∈ arrangement.drop 4.take 4 → c = 'A' ∨ c = 'C' ∨ c = 'D'

def valid_last_section (arrangement : List Char) : Prop :=
  ∀ (c : Char), c ∈ arrangement.drop 8 → c = 'A' ∨ c = 'B'

def is_valid_arrangement (arrangement : List Char) : Prop :=
  valid_first_section arrangement ∧ valid_middle_section arrangement ∧ valid_last_section arrangement

def count_valid_arrangements : ℕ :=
  -- Expression calculating the total valid arrangements

-- Main statement to be proven
theorem arrangement_count_correct : count_valid_arrangements = 192 := by
  sorry

end arrangement_count_correct_l124_124644


namespace james_bought_100_cattle_l124_124696

noncomputable def number_of_cattle (purchase_price : ℝ) (feeding_ratio : ℝ) (weight_per_cattle : ℝ) (price_per_pound : ℝ) (profit : ℝ) : ℝ :=
  let feeding_cost := purchase_price * feeding_ratio
  let total_feeding_cost := purchase_price + feeding_cost
  let total_cost := purchase_price + total_feeding_cost
  let selling_price_per_cattle := weight_per_cattle * price_per_pound
  let total_revenue := total_cost + profit
  total_revenue / selling_price_per_cattle

theorem james_bought_100_cattle :
  number_of_cattle 40000 0.20 1000 2 112000 = 100 :=
by {
  sorry
}

end james_bought_100_cattle_l124_124696


namespace intersection_count_zero_l124_124649

noncomputable def circle_center_and_radius (r_eq : ℝ → ℝ) : (ℝ × ℝ) × ℝ :=
  if h : r_eq = (λ θ, 3 * Real.cos θ) then ((3/2, 0), 3/2)
  else if h : r_eq = (λ θ, 5 * Real.sin θ) then ((0, 5/2), 5/2)
  else ((0,0), 0)

theorem intersection_count_zero (r1_eq r2_eq : ℝ → ℝ) (h1 : r1_eq = (λ θ, 3 * Real.cos θ)) (h2 : r2_eq = (λ θ, 5 * Real.sin θ)) :
  let c1 := (circle_center_and_radius r1_eq).fst,
      r1 := (circle_center_and_radius r1_eq).snd,
      c2 := (circle_center_and_radius r2_eq).fst,
      r2 := (circle_center_and_radius r2_eq).snd in
  Real.dist c1 c2 > r1 + r2 :=
begin
  have h_c1 : c1 = (3/2, 0), by {dsimp only [circle_center_and_radius], rw h1, refl},
  have h_r1 : r1 = 3/2, by {dsimp only [circle_center_and_radius], rw h1, refl},
  have h_c2 : c2 = (0, 5/2), by {dsimp only [circle_center_and_radius], rw h2, refl},
  have h_r2 : r2 = 5/2, by {dsimp only [circle_center_and_radius], rw h2, refl},

  rw [h_c1, h_c2, h_r1, h_r2],
  dsimp only [Real.dist],
  rw [Real.dist_left, Real.sqrt_sq_eq_abs, (abs_of_nonneg (by norm_num : 9/4 + 25/4 ≥ 0))],
  norm_num,
  linarith
end

end intersection_count_zero_l124_124649


namespace sum_of_roots_equals_18_l124_124738

-- Define the conditions
variable (f : ℝ → ℝ)
variable (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x))
variable (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0))

-- The theorem statement
theorem sum_of_roots_equals_18 (f : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x)) 
  (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0)) :
  ∀ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0) → xs.sum id = 18 :=
by
  sorry

end sum_of_roots_equals_18_l124_124738


namespace smallest_c_for_exponential_inequality_l124_124856

theorem smallest_c_for_exponential_inequality : ∃ c : ℤ, 27^c > 3^24 ∧ ∀ n : ℤ, 27^n > 3^24 → c ≤ n := sorry

end smallest_c_for_exponential_inequality_l124_124856


namespace line_BB_l124_124686

-- Definitions of geometric objects and properties
variables {Ω : Type*} [EuclideanGeometry Ω]
variables (A B C A₁ C₁ A' C' B' O : Ω)

-- Conditions of the problem
axiom h1 : acute_triangle A B C
axiom h2 : altitude A A₁ B C
axiom h3 : altitude C C₁ A B
axiom h4 : is_circumcircle Ω A B C
axiom h5 : intersects_line A₁ C₁ Ω A' C'
axiom h6 : tangent_at Ω A' intersects B'
axiom h7 : tangent_at Ω C' intersects B'

-- Conclusion to be proved
theorem line_BB'_passes_center :
  on_line B B' O ∧ center_of_circumcircle Ω A B C O :=
sorry

end line_BB_l124_124686


namespace ellipse_equation_and_constant_sum_l124_124399

def is_ellipse (C : Type) (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), a > b ∧ b > 0 ∧ x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1

theorem ellipse_equation_and_constant_sum :
  ∀ (a b m x0 y0 : ℝ),
    let e := (3 : ℝ) / 5,
    let P := (m, 0),
    let PA := (x0, y0),
    let PB := (-x0, -y0),
    let l := λ x, (4 : ℝ) / 5 * x,
    let inner_prod := λ (u v : ℝ × ℝ), u.1 * v.1 + u.2 * v.2,
    is_ellipse C a b →
    inner_prod (PA.1 - m, PA.2) (PB.1 - m, PB.2) = -41 / 2 →
    a ^ 2 = 25 ∧ b ^ 2 = 16 ∧
    (∀ m P A B,
      inner_prod (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = -41 / 2 →
      let PA2 := (A.1 - P.1) ^ 2 + A.2 ^ 2,
      let PB2 := (B.1 - P.1) ^ 2 + B.2 ^ 2,
      PA2 + PB2 = 41) :=
begin
  sorry
end

end ellipse_equation_and_constant_sum_l124_124399


namespace min_value_expression_l124_124039

theorem min_value_expression (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 3) : 
  let A := (a^3 + b^3) / (8 * a * b + 9 - c^2) + 
           (b^3 + c^3) / (8 * b * c + 9 - a^2) + 
           (c^3 + a^3) / (8 * c * a + 9 - b^2) 
  in A = 3 / 8 :=
sorry

end min_value_expression_l124_124039


namespace largest_n_exists_l124_124554

theorem largest_n_exists :
  ∃ (n : ℕ), (∀ (x : ℕ → ℝ), (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → (1 + x i * x j)^2 ≤ 0.99 * (1 + x i^2) * (1 + x j^2))) ∧ n = 31 :=
sorry

end largest_n_exists_l124_124554


namespace percent_within_one_std_dev_l124_124487

variable {α : Type*} [MeasureTheory.ProbabilitySpace α]

def symmetric_about_mean (f : α → ℝ) (m : ℝ) : Prop :=
∀ x, f m - x = f m + x

def within_one_std_dev (f : α → ℝ) (m d : ℝ) : Prop :=
  (∫ x in -d..d, f (m + x)) / (∫ f) = 0.68

theorem percent_within_one_std_dev 
  (f : α → ℝ) (m d : ℝ) 
  (h_sym : symmetric_about_mean f m) 
  (h_less_m_plus_d : ∫ x in -∞..d, f (m + x) = 0.84) : 
  within_one_std_dev f m d :=
sorry

end percent_within_one_std_dev_l124_124487


namespace find_x_l124_124588

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l124_124588


namespace hikers_meet_2_5_miles_closer_to_A_l124_124816

noncomputable def hikers_meet_closer_to_A : Prop :=
  ∃ (t : ℕ), 
    let distance_A := 5 * t in
    let speed_B (k : ℕ) := 4 + 0.25 * k in
    let distance_B := finset.sum (finset.range t) speed_B in
    let total_distance := distance_A + distance_B in
    total_distance = 100 ∧ 
    distance_B - distance_A = 2.5

theorem hikers_meet_2_5_miles_closer_to_A : hikers_meet_closer_to_A :=
by {
  sorry
}

end hikers_meet_2_5_miles_closer_to_A_l124_124816


namespace sqrt_eq_l124_124233

noncomputable def sqrt_22500 := 150

theorem sqrt_eq (h : sqrt_22500 = 150) : Real.sqrt 0.0225 = 0.15 :=
sorry

end sqrt_eq_l124_124233


namespace trapezoid_JKLM_perimeter_l124_124431

-- Definitions for the vertices of the trapezoid
def J := (-2: ℝ, -4: ℝ)
def K := (-2: ℝ, 1: ℝ)
def L := (6: ℝ, 7: ℝ)
def M := (6: ℝ, -4: ℝ)

-- Euclidean distance between two points
def euclidean_dist (P Q: ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Definitions for the lengths of the sides
def JK := euclidean_dist J K
def KL := euclidean_dist K L
def LM := euclidean_dist L M
def MJ := euclidean_dist M J

-- Definition for the perimeter
def perimeter := JK + KL + LM + MJ

-- Theorem statement for the perimeter of trapezoid JKLM
theorem trapezoid_JKLM_perimeter : 
  perimeter = 34 :=
  sorry

end trapezoid_JKLM_perimeter_l124_124431


namespace total_spent_is_correct_l124_124205

def meal_prices : List ℕ := [12, 15, 10, 18, 20]
def ice_cream_prices : List ℕ := [2, 3, 3, 4, 4]
def tip_percentage : ℝ := 0.15
def tax_percentage : ℝ := 0.08

def total_meal_cost (prices : List ℕ) : ℝ :=
  prices.sum

def total_ice_cream_cost (prices : List ℕ) : ℝ :=
  prices.sum

def calculate_tip (total_meal_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  total_meal_cost * tip_percentage

def calculate_tax (total_meal_cost : ℝ) (tax_percentage : ℝ) : ℝ :=
  total_meal_cost * tax_percentage

def total_amount_spent (meal_prices : List ℕ) (ice_cream_prices : List ℕ) (tip_percentage : ℝ) (tax_percentage : ℝ) : ℝ :=
  let total_meal := total_meal_cost meal_prices
  let total_ice_cream := total_ice_cream_cost ice_cream_prices
  let tip := calculate_tip total_meal tip_percentage
  let tax := calculate_tax total_meal tax_percentage
  total_meal + total_ice_cream + tip + tax

theorem total_spent_is_correct :
  total_amount_spent meal_prices ice_cream_prices tip_percentage tax_percentage = 108.25 := 
by
  sorry

end total_spent_is_correct_l124_124205


namespace dante_coconuts_l124_124035

theorem dante_coconuts (P : ℕ) (D : ℕ) (S : ℕ) (hP : P = 14) (hD : D = 3 * P) (hS : S = 10) :
  (D - S) = 32 :=
by
  sorry

end dante_coconuts_l124_124035


namespace quadratic_eq1_solution_quadratic_eq2_solution_l124_124050

-- Define the first problem and its conditions
theorem quadratic_eq1_solution :
  ∀ x : ℝ, 4 * x^2 + x - (1 / 2) = 0 ↔ (x = -1 / 2 ∨ x = 1 / 4) :=
by
  -- The proof is omitted
  sorry

-- Define the second problem and its conditions
theorem quadratic_eq2_solution :
  ∀ y : ℝ, (y - 2) * (y + 3) = 6 ↔ (y = -4 ∨ y = 3) :=
by
  -- The proof is omitted
  sorry

end quadratic_eq1_solution_quadratic_eq2_solution_l124_124050


namespace y_alone_time_l124_124125

-- Definitions based on conditions
def work_rate (hours : ℕ) : ℚ := 1 / hours

axiom x_rate : work_rate 8 = 1 / 8
axiom yz_rate : work_rate 6 = 1 / 6
axiom xz_rate : work_rate 4 = 1 / 4

-- Theorem stating what we need to prove
theorem y_alone_time : 
  let x_rate := work_rate 8 in
  let yz_rate := work_rate 6 in
  let xz_rate := work_rate 4 in
  let z_rate := xz_rate - x_rate in
  let y_rate := yz_rate - z_rate in
  1 / y_rate = 24 :=
by
  sorry

end y_alone_time_l124_124125


namespace monotonic_intervals_sum_y_1_to_y_2018_l124_124271

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * (cos x + sqrt 3 * sin x) - 1

-- Define the sequence x_n
def x_sequence (n : ℕ) : ℝ :=
  if n = 0 then π / 6
  else x_sequence (n - 1) + π / 2

-- Define y_n as f evaluated at each point in the sequence x_n
def y_sequence (n : ℕ) : ℝ :=
  f (x_sequence n)

theorem monotonic_intervals (k : ℤ) : ∃ a b : ℝ, a = -(π / 3) + k * π ∧ b = π / 6 + k * π ∧
  ∀ x : ℝ, a ≤ x ∧ x ≤ b → (has_deriv_at f x).to_fun > 0 :=
sorry

theorem sum_y_1_to_y_2018 : ∑ i in finset.range 2018, y_sequence (i + 1) = 0 :=
sorry

end monotonic_intervals_sum_y_1_to_y_2018_l124_124271


namespace transformed_function_correct_l124_124090

def f (x : ℝ) : ℝ := sin (2 * x)

theorem transformed_function_correct :
  ∀ x : ℝ, (f (x + π / 3)) = sin (4 * x + (2 * π / 3)) :=
by
  sorry

end transformed_function_correct_l124_124090


namespace geom_progression_contra_l124_124970

theorem geom_progression_contra (q : ℝ) (p n : ℕ) (hp : p > 0) (hn : n > 0) :
  (11 = 10 * q^p) → (12 = 10 * q^n) → False :=
by
  -- proof steps should follow here
  sorry

end geom_progression_contra_l124_124970


namespace fenced_area_with_cutouts_l124_124400

theorem fenced_area_with_cutouts :
  let rectangle_area := 20 * 16 in
  let square_cutout_area := 4 * 4 in
  let triangular_cutout_area := 1 / 2 * 3 * 3 in
  rectangle_area - square_cutout_area - triangular_cutout_area = 299.5 :=
by
  sorry

end fenced_area_with_cutouts_l124_124400


namespace probability_all_successful_pairs_expected_successful_pairs_l124_124832

-- Lean statement for part (a)
theorem probability_all_successful_pairs (n : ℕ) :
  let prob_all_successful := (2 ^ n * n.factorial) / (2 * n).factorial
  in prob_all_successful = (1 / (2 ^ n)) * (n.factorial / (2 * n).factorial)
:= by sorry

-- Lean statement for part (b)
theorem expected_successful_pairs (n : ℕ) :
  let expected_value := n / (2 * n - 1)
  in expected_value > 0.5
:= by sorry

end probability_all_successful_pairs_expected_successful_pairs_l124_124832


namespace find_x_l124_124594

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l124_124594


namespace length_of_first_train_l124_124510

theorem length_of_first_train
    (speed_first_train : ℝ)
    (speed_second_train : ℝ)
    (crossing_time : ℝ)
    (length_second_train : ℝ)
    (h1 : speed_first_train = 120)
    (h2 : speed_second_train = 80)
    (h3 : crossing_time = 9)
    (h4 : length_second_train = 240.04) : 
    (260 : ℝ) :=
by
  have relative_speed := speed_first_train + speed_second_train
  have relative_speed_mps := (relative_speed * 1000) / 3600
  have combined_length := relative_speed_mps * crossing_time
  have length_first_train := combined_length - length_second_train
  exact length_first_train

#eval length_of_first_train 120 80 9 240.04

end length_of_first_train_l124_124510


namespace number_of_multiples_of_6_and_9_is_5_l124_124291

def is_multiple (n k : ℕ) : Prop := ∃ m, n = k * m

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem number_of_multiples_of_6_and_9_is_5 :
  { n : ℕ | is_two_digit n ∧ is_multiple n (lcm 6 9) }.to_finset.card = 5 :=
by
  sorry

end number_of_multiples_of_6_and_9_is_5_l124_124291


namespace find_number_of_girls_l124_124683

variable (G : ℕ)

-- Given conditions
def avg_weight_girls (total_weight_girls : ℕ) : Prop := total_weight_girls = 45 * G
def avg_weight_boys (total_weight_boys : ℕ) : Prop := total_weight_boys = 275
def avg_weight_students (total_weight_students : ℕ) : Prop := total_weight_students = 500

-- Proposition to prove
theorem find_number_of_girls 
  (total_weight_girls : ℕ) 
  (total_weight_boys : ℕ) 
  (total_weight_students : ℕ) 
  (h1 : avg_weight_girls G total_weight_girls)
  (h2 : avg_weight_boys total_weight_boys)
  (h3 : avg_weight_students total_weight_students) : 
  G = 5 :=
by sorry

end find_number_of_girls_l124_124683


namespace initial_masses_l124_124673

def area_of_base : ℝ := 15
def density_water : ℝ := 1
def density_ice : ℝ := 0.92
def change_in_water_level : ℝ := 5
def final_height_of_water : ℝ := 115

theorem initial_masses (m_ice m_water : ℝ) :
  m_ice = 675 ∧ m_water = 1050 :=
by
  -- Calculate the change in volume of water
  let delta_v := area_of_base * change_in_water_level

  -- Relate this volume change to the volume difference between ice and water
  let lhs := m_ice / density_ice - m_ice / density_water
  let eq1 := delta_v

  -- Solve for the mass of ice
  have h_ice : m_ice = 675 := 
  sorry

  -- Determine the final volume of water
  let final_volume_of_water := final_height_of_water * area_of_base

  -- Determine the initial mass of water
  let mass_of_water_total := density_water * final_volume_of_water
  let initial_mass_of_water :=
    mass_of_water_total - m_ice

  have h_water : m_water = 1050 := 
  sorry

  exact ⟨h_ice, h_water⟩

end initial_masses_l124_124673


namespace exists_five_non_zero_divisible_numbers_l124_124442

def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def non_zero_numbers := {n : ℕ | n ≠ 0}

theorem exists_five_non_zero_divisible_numbers : 
  ∃ (a b c d e : ℕ), 
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (d ≠ 0) ∧ (e ≠ 0) ∧ 
  (a ∈ digits) ∧ (b ∈ digits) ∧ (c ∈ digits) ∧ (d ∈ digits) ∧ (e ∈ digits) ∧ 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ 
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ 
  (c ≠ d) ∧ (c ≠ e) ∧ 
  (d ≠ e) ∧ 
  (b % a = 0) ∧ (c % b = 0) ∧ (d % c = 0) ∧ (e % d = 0) := 
sorry

end exists_five_non_zero_divisible_numbers_l124_124442


namespace sum_of_complex_exponentials_l124_124967

noncomputable theory
open Complex

theorem sum_of_complex_exponentials : 
  let ω := exp (Complex.pi * Complex.I / 11)
  in ω + ω^3 + ω^5 + ω^7 + ω^9 + ω^11 + ω^13 + ω^15 + ω^17 + ω^19 + ω^21 = 0 :=
by
  let ω := exp (Complex.pi * Complex.I / 11)
  have ω_pow_22 : ω^22 = 1, from by 
    rw [←Complex.exp_nat_mul, mul_div_cancel' (2 * Complex.pi) (by norm_num), Complex.exp_two_pi_I],
  sorry

end sum_of_complex_exponentials_l124_124967


namespace smaller_circle_with_integer_points_l124_124578

theorem smaller_circle_with_integer_points (R : ℝ) (hR : 0 < R) :
  ∃ (R' : ℝ) (hR' : R' < R) (center : ℝ × ℝ), 
    (∀ (x y : ℤ), (x^2 + y^2 ≤ R^2) → 
      (∃ (x' y' : ℤ), ((x'^2 + y'^2 ≤ R'^2) ∧ (x'^2 + y'^2 ≤ (x^2 + y^2)))) :=
by
  sorry

end smaller_circle_with_integer_points_l124_124578


namespace eccentricity_is_sqrt2_div2_l124_124600

noncomputable def eccentricity_square_ellipse (a b c : ℝ) : ℝ :=
  c / (Real.sqrt (b ^ 2 + c ^ 2))

theorem eccentricity_is_sqrt2_div2 (a b c : ℝ) (h : b = c) : 
  eccentricity_square_ellipse a b c = Real.sqrt 2 / 2 :=
by
  -- The proof will show that the eccentricity calculation is correct given the conditions.
  sorry

end eccentricity_is_sqrt2_div2_l124_124600


namespace problem_statement_l124_124238

variable {α : Type*} [OrderedAddCommGroup α] [OrderedCommMonoid α]
  [LinearOrderedField α] [LinearOrderedAddCommGroup α] [OrderedRing α]

-- Variables and Definitions
variable (n : ℕ) (a : Fin n → α)
-- Assumptions
def a_positive (i : Fin n) : Prop := 0 < a i
def sum_a_eq_one : Prop := ∑ i, a i = 1
def a_n_plus_1_eq_a1 : α := a ⟨0, by linarith⟩

-- Main Theorem Statement
theorem problem_statement
  (hp : ∀ i, a_positive n a i)
  (hs : sum_a_eq_one n a)
  (hn : a_n_plus_1_eq_a1 n a = a ⟨0, by linarith⟩) :
  (∑ i : Fin n, a i * a ⟨i.1 + 1 % n, by linarith⟩) * (∑ i : Fin n, a i / (a ⟨i.1 + 1 % n, by linarith⟩ ^ 2 + a ⟨i.1 + 1 % n, by linarith⟩)) ≥ n / (n + 1) := 
by {
  sorry
}

end problem_statement_l124_124238


namespace number_of_terms_in_arithmetic_sequence_l124_124541

theorem number_of_terms_in_arithmetic_sequence :
  let seq := [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57]
  in seq.length = 12 :=
by
  -- Use the conditions provided to establish the theorem
  have h : seq = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57] := rfl
  have len_seq : seq.length = 12 := rfl
  exact len_seq

end number_of_terms_in_arithmetic_sequence_l124_124541


namespace cutoff_score_505_l124_124958

/-- 
 Given:
  - The scores of the second mock exam follow a normal distribution ξ ~ N(480, 100^2),
  - The admission rate is 40%, and
  - Φ(0.25) = 0.6.

 Prove that the cut-off score for admission to first-tier universities is 505.
-/
theorem cutoff_score_505
  (h_normal_dist : ∀ x : ℝ, (1 / (100 * Mathlib.Real.sqrt (2 * Mathlib.Real.pi))) * (Mathlib.Real.exp (- ((x - 480)^2 / (2 * 100^2)))) = Mathlib.Normal.pdf 480 100 x)
  (h_admission_rate : ∀ {p: ℝ}, 0 < p ∧ p < 1 → if p = 0.4 then true else false)
  (h_Phi : Mathlib.Probability.StandardNormal.cdf 0.25 = 0.6) :
  ∃ cutoff_score : ℝ, cutoff_score = 505 :=
sorry

end cutoff_score_505_l124_124958


namespace largest_n_for_factorable_poly_l124_124196

theorem largest_n_for_factorable_poly :
  ∃ n : ℤ, (∀ A B : ℤ, (3 * B + A = n) ∧ (A * B = 72) → (A = 1 ∧ B = 72 ∧ n = 217)) ∧
           (∀ A B : ℤ, A * B = 72 → 3 * B + A ≤ 217) :=
by
  sorry

end largest_n_for_factorable_poly_l124_124196


namespace count_divisibles_3_4_5_l124_124648

theorem count_divisibles_3_4_5 : 
  let l := list.filter (λ n, n > 0 ∧ n < 300 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) (list.range 300) 
  in list.length l = 5 := 
by
  sorry

end count_divisibles_3_4_5_l124_124648


namespace scientific_notation_135000_l124_124473

theorem scientific_notation_135000 :
  135000 = 1.35 * 10^5 := sorry

end scientific_notation_135000_l124_124473


namespace geometric_sequence_second_term_l124_124497

theorem geometric_sequence_second_term (a r : ℕ) (h1 : a = 5) (h2 : a * r^4 = 1280) : a * r = 20 :=
by
  sorry

end geometric_sequence_second_term_l124_124497


namespace opposite_point_83_is_84_l124_124902

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l124_124902


namespace distance_between_vertices_l124_124012

def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  let y := a * x^2 + b * x + c
  (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_vertices :
  let C := vertex 1 6 13
  let D := vertex 1 (-4) 5
  distance C D = Real.sqrt 34 := by
  sorry

end distance_between_vertices_l124_124012


namespace probability_of_A_and_B_l124_124668

-- Define the events and their probabilities.
def event_A : Prop := "the first ball drawn is white"
def event_B : Prop := "the second ball drawn is white"
def bag : Type := {balls : List (Bool × String) // balls.length = 8}

-- Assumptions about the bag.
def initial_bag_condition : bag := {
  balls := [(true, "white")] ++ [(false, "red")] -- ball's color
  sorry -- fill out the remaining balls to meet the constraint of total 8 balls.
}

-- Define the probability workspace.
noncomputable def P (A : Prop) : Real := sorry 
noncomputable def P_condition (A B : Prop) : Real := sorry

-- We need to show the final probability calculation.
theorem probability_of_A_and_B :
  let P_A := P event_A 
  let P_B_given_A := P_condition event_A event_B 
  let P_A_and_B := P_A * P_B_given_A
  P_A_and_B = 5 / 14 :=
sorry

end probability_of_A_and_B_l124_124668


namespace range_of_k_l124_124270

theorem range_of_k (k : ℝ) :
  (∀ y : ℝ, y ∈ ℝ → (∃ x : ℝ, y = Real.logb 2 (x^2 - 2*k*x + k))) ↔ k ≤ 0 ∨ k ≥ 1 :=
sorry

end range_of_k_l124_124270


namespace probability_of_Bernie_l124_124414

theorem probability_of_Bernie (P_Carol P_Both P_Bernie : ℚ) 
  (h1 : P_Carol = 4 / 5) 
  (h2 : P_Both = 0.48) 
  : P_Carol * P_Bernie = P_Both → P_Bernie = 3 / 5 :=
by 
  intros h3
  have : P_Bernie = 0.6 := by
    rw [h1, h2] at h3
    field_simp at h3
    exact h3
  rw this
  norm_num
  sorry

end probability_of_Bernie_l124_124414


namespace kyle_money_l124_124705

theorem kyle_money (dave_money : ℕ) (kyle_initial : ℕ) (kyle_remaining : ℕ)
  (h1 : dave_money = 46)
  (h2 : kyle_initial = 3 * dave_money - 12)
  (h3 : kyle_remaining = kyle_initial - kyle_initial / 3) :
  kyle_remaining = 84 :=
by
  -- Define Dave's money and provide the assumption
  let dave_money := 46
  have h1 : dave_money = 46 := rfl

  -- Define Kyle's initial money based on Dave's money
  let kyle_initial := 3 * dave_money - 12
  have h2 : kyle_initial = 3 * dave_money - 12 := rfl

  -- Define Kyle's remaining money after spending one third on snowboarding
  let kyle_remaining := kyle_initial - kyle_initial / 3
  have h3 : kyle_remaining = kyle_initial - kyle_initial / 3 := rfl

  -- Now we prove that Kyle's remaining money is 84
  sorry -- Proof steps omitted

end kyle_money_l124_124705


namespace prob_not_D_l124_124464

variable {Ω : Type} [ProbabilitySpace Ω]

def P (A : Set Ω) : ℝ := sorry  -- Placeholder for probability measure

variable (D C : Set Ω)

-- Conditions
axiom h1 : P D = 0.60
axiom h2 : P (D ∩ Cᶜ) = 0.20

-- Goal
theorem prob_not_D : P Dᶜ = 0.40 :=
by
  have h3 : P D + P Dᶜ = 1 := axioms.prob_compl
  rw [h1] at h3
  linarith

end prob_not_D_l124_124464


namespace opposite_number_l124_124915

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l124_124915


namespace problem_b_l124_124267

noncomputable def f (x : ℝ) : ℝ :=
  -2 * cos (2 * x + (Real.pi / 3)) * sin (2 * x) - (Real.sqrt 3 / 2)

theorem problem_b : monotone_on f (Set.Icc (Real.pi / 6) (Real.pi / 4)) :=
sorry

end problem_b_l124_124267


namespace find_numbers_l124_124422

theorem find_numbers (x y z : ℕ) :
  x + y + z = 35 → 
  2 * y = x + z + 1 → 
  y^2 = (x + 3) * z → 
  (x = 15 ∧ y = 12 ∧ z = 8) ∨ (x = 5 ∧ y = 12 ∧ z = 18) :=
by
  sorry

end find_numbers_l124_124422


namespace diametrically_opposite_number_is_84_l124_124891

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l124_124891


namespace employee_payment_correct_l124_124506

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the percentage markup for retail price
def markup_percentage : ℝ := 0.20

-- Define the retail_price based on wholesale cost and markup percentage
def retail_price : ℝ := wholesale_cost + (markup_percentage * wholesale_cost)

-- Define the employee discount percentage
def discount_percentage : ℝ := 0.20

-- Define the discount amount based on retail price and discount percentage
def discount_amount : ℝ := retail_price * discount_percentage

-- Define the final price the employee pays after applying the discount
def employee_price : ℝ := retail_price - discount_amount

-- State the theorem to prove
theorem employee_payment_correct :
  employee_price = 192 :=
  by
    sorry

end employee_payment_correct_l124_124506


namespace largest_semicircle_properties_l124_124548

def largest_semicircle_perimeter (length : ℝ) (width : ℝ) : ℝ :=
  let diameter := length
  let radius := diameter / 2
  (3.14 * radius) + diameter

def largest_semicircle_area (length : ℝ) (width : ℝ) : ℝ :=
  let radius := length / 2
  (3.14 * radius^2) / 2

theorem largest_semicircle_properties :
  largest_semicircle_perimeter 10 8 = 25.7 ∧
  largest_semicircle_area 10 8 = 39.25 :=
by
  sorry


end largest_semicircle_properties_l124_124548


namespace bianca_winning_strategy_for_initial_pile_sizes_l124_124955

def is_losing_position (a b : ℕ) : Prop :=
  ∃ k : ℤ, k ≠ 0 ∧ (a + 1) = (b + 1) * 2^k

theorem bianca_winning_strategy_for_initial_pile_sizes :
  ∀ (a b : ℕ), (a + b = 100) → 
  (a = 50 ∧ b = 50) ∨ 
  (a = 67 ∧ b = 33) ∨ 
  (a = 33 ∧ b = 67) ∨ 
  (a = 95 ∧ b = 5) ∨ 
  (a = 5 ∧ b = 95) → 
  ¬ is_losing_position a b :=
by
  intros a b h_sum h_piles
  sorry

end bianca_winning_strategy_for_initial_pile_sizes_l124_124955


namespace probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124822

-- Define the problem's parameters and conditions
def n : ℕ  -- number of pairs of socks

-- Define the successful pair probability calculation
theorem probability_all_pairs_successful (n : ℕ) :
  (∑ s in Finset.range (n + 1), s.factorial * 2 ^ s) / ((2 * n).factorial) = 2^n * n.factorial := sorry

-- Define the expected value calculation
theorem expected_successful_pairs_greater_than_half (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n, (1 : ℝ) / (2 * n - 1) : ℝ) / n > 0.5 := sorry

end probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124822


namespace two_A_geq_B_l124_124191

universe u

def M (n : ℕ) : Finset ℕ := Finset.range (n + 1)

def A (n : ℕ) (colors : Finset ℕ → Fin ℕ) : Finset (ℕ × ℕ × ℕ) :=
  { x | ∃ (x y z : ℕ), x ∈ M n ∧ y ∈ M n ∧ z ∈ M n ∧
  (x + y + z) % n = 0 ∧ colors {x, y, z} = 1 ∧ colors {x, y, z} = 2 ∧ colors {x, y, z} = 3 }

def B (n : ℕ) (colors : ℕ → Fin ℕ) : Finset (ℕ × ℕ × ℕ) :=
  { x | ∃ (x y z : ℕ), x ∈ M n ∧ y ∈ M n ∧ z ∈ M n ∧
  (x + y + z) % n = 0 ∧ colors x ≠ colors y ∧ colors y ≠ colors z ∧ colors z ≠ colors x }

theorem two_A_geq_B (n : ℕ) (colors : ℕ → Fin 3)
  : 2 * A n colors.card ≥ B n colors.card := 
by
  sorry

end two_A_geq_B_l124_124191


namespace three_digit_palindrome_probability_divisible_by_11_l124_124508

theorem three_digit_palindrome_probability_divisible_by_11 :
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 101 * a + 10 * b % 11 = 0) →
  (∃ p : ℚ, p = 1 / 5) :=
by
  sorry

end three_digit_palindrome_probability_divisible_by_11_l124_124508


namespace misha_is_older_l124_124054

-- Definitions for the conditions
def tanya_age_19_months_ago : ℕ := 16
def months_ago_for_tanya : ℕ := 19
def misha_age_in_16_months : ℕ := 19
def months_ahead_for_misha : ℕ := 16

-- Convert months to years and residual months
def months_to_years_months (m : ℕ) : ℕ × ℕ := (m / 12, m % 12)

-- Computation for Tanya's current age
def tanya_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ago_for_tanya
  (tanya_age_19_months_ago + years, months)

-- Computation for Misha's current age
def misha_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ahead_for_misha
  (misha_age_in_16_months - years, months)

-- Proof statement
theorem misha_is_older : misha_age_now > tanya_age_now := by
  sorry

end misha_is_older_l124_124054


namespace tabitha_item_cost_l124_124385

theorem tabitha_item_cost :
  ∀ (start_money gave_mom invest fraction_remain spend item_count remain_money item_cost : ℝ),
    start_money = 25 →
    gave_mom = 8 →
    invest = (start_money - gave_mom) / 2 →
    fraction_remain = start_money - gave_mom - invest →
    spend = fraction_remain - remain_money →
    item_count = 5 →
    remain_money = 6 →
    item_cost = spend / item_count →
    item_cost = 0.5 :=
by
  intros
  sorry

end tabitha_item_cost_l124_124385


namespace subset_sum_divisible_l124_124341

theorem subset_sum_divisible (n : ℕ) (h1 : 4 ≤ n) (a : fin n → ℕ)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j)
  (h3 : ∀ i, 0 < a i ∧ a i < 2 * n) :
  ∃ s : finset (fin n), (s.sum a) % (2 * n) = 0 :=
sorry

end subset_sum_divisible_l124_124341


namespace total_legs_camden_dogs_l124_124531

variable (c r j : ℕ) -- c: Camden's dogs, r: Rico's dogs, j: Justin's dogs

theorem total_legs_camden_dogs :
  (r = j + 10) ∧ (j = 14) ∧ (c = (3 * r) / 4) → 4 * c = 72 :=
by
  sorry

end total_legs_camden_dogs_l124_124531


namespace exponent_problem_l124_124298

theorem exponent_problem (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  intro h
  sorry

end exponent_problem_l124_124298


namespace trig_identity_l124_124265

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem trig_identity (x : ℝ) (h : f x = 2 * f' x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 := by
  sorry

end trig_identity_l124_124265


namespace cardinality_square_is_continuum_l124_124765

theorem cardinality_square_is_continuum :
  #((set.Icc 0 1) ×ˢ (set.Icc 0 1)) = cardinal.continuum :=
begin
  sorry
end

end cardinality_square_is_continuum_l124_124765


namespace john_made_6_dozens_of_cookies_l124_124331

-- Defining the conditions
def selling_price_per_cookie : ℝ := 1.5
def cost_per_cookie : ℝ := 0.25
def profit_per_charity : ℝ := 45
def num_charities : ℕ := 2

-- Calculating the correct answer
def total_profit : ℝ := profit_per_charity * num_charities
def profit_per_cookie : ℝ := selling_price_per_cookie - cost_per_cookie
def total_number_of_cookies : ℝ := total_profit / profit_per_cookie
def number_of_dozen_of_cookies : ℝ := total_number_of_cookies / 12

-- Proof statement
theorem john_made_6_dozens_of_cookies : number_of_dozen_of_cookies = 6 := by
  sorry

end john_made_6_dozens_of_cookies_l124_124331


namespace largest_multiple_of_9_less_than_75_is_72_l124_124855

theorem largest_multiple_of_9_less_than_75_is_72 : 
  ∃ n : ℕ, 9 * n < 75 ∧ ∀ m : ℕ, 9 * m < 75 → 9 * m ≤ 9 * n :=
sorry

end largest_multiple_of_9_less_than_75_is_72_l124_124855


namespace four_squares_cover_larger_square_l124_124379

structure Square :=
  (side : ℝ) (h_positive : side > 0)

theorem four_squares_cover_larger_square (large small : Square) 
  (h_side_relation: large.side = 2 * small.side) : 
  large.side^2 = 4 * small.side^2 :=
by
  sorry

end four_squares_cover_larger_square_l124_124379


namespace sum_binom_coeff_zero_sum_binom_coeff_n_fact_l124_124368

theorem sum_binom_coeff_zero (n m : ℕ) (h : m < n) :
  ∑ k in Finset.range (n + 1), (-1) ^ k * k ^ m * Nat.choose n k = 0 :=
sorry

theorem sum_binom_coeff_n_fact (n : ℕ) :
  ∑ k in Finset.range (n + 1), (-1) ^ k * k ^ n * Nat.choose n k = (-1) ^ n * Nat.factorial n :=
sorry

end sum_binom_coeff_zero_sum_binom_coeff_n_fact_l124_124368


namespace Emily_distance_from_home_to_school_l124_124092
-- Import the necessary library

-- Define the variables and conditions
def Troy_distance_to_school := 75
def Emily_extra_distance_in_5_days := 230
def Emily_distance_to_school := 98

-- Define the theorem
theorem Emily_distance_from_home_to_school:
  let d_T := Troy_distance_to_school in
  let extra_dist := Emily_extra_distance_in_5_days in
  d_T = 75 → 
  extra_dist = 230 → 
  let d_E := 75 + extra_dist / 10 in
  d_E = Emily_distance_to_school :=
by
  intros
  sorry

end Emily_distance_from_home_to_school_l124_124092


namespace smallest_d_l124_124782

theorem smallest_d (c d : ℕ) (h1 : c - d = 8)
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : d = 4 := by
  sorry

end smallest_d_l124_124782


namespace initial_population_is_3162_l124_124880

noncomputable def initial_population (P : ℕ) : Prop :=
  let after_bombardment := 0.95 * (P : ℝ)
  let after_fear := 0.85 * after_bombardment
  after_fear = 2553

theorem initial_population_is_3162 : initial_population 3162 :=
  by
    -- By our condition setup, we need to prove:
    -- let after_bombardment := 0.95 * 3162
    -- let after_fear := 0.85 * after_bombardment
    -- after_fear = 2553

    -- This can be directly stated and verified through concrete calculations as in the problem steps.
    sorry

end initial_population_is_3162_l124_124880


namespace at_least_one_ge_one_l124_124762

theorem at_least_one_ge_one (x y : ℝ) (h : x + y ≥ 2) : x ≥ 1 ∨ y ≥ 1 :=
sorry

end at_least_one_ge_one_l124_124762


namespace num_people_on_boats_l124_124479

-- Definitions based on the conditions
def boats := 5
def people_per_boat := 3

-- Theorem stating the problem to be solved
theorem num_people_on_boats : boats * people_per_boat = 15 :=
by sorry

end num_people_on_boats_l124_124479


namespace length_of_AC_l124_124785

def bisector_bac_intersects_bc_at (A B C L : Point) : Prop := sorry
def external_bisector_acb_intersects_ba_at (A B C K : Point) : Prop := sorry
def length_AK_eq_perimeter_acl (A C L K : Point) : Prop := sorry
def length_LB (L B : Point) (length : ℝ) : Prop := sorry
def angle_ABC_eq_36 (A B C : Point) : Prop := sorry

noncomputable def Point : Type := sorry

theorem length_of_AC (A B C L K : Point) :
    bisector_bac_intersects_bc_at A B C L →
    external_bisector_acb_intersects_ba_at A B C K →
    length_AK_eq_perimeter_acl A C L K →
    length_LB L B 1 →
    angle_ABC_eq_36 A B C →
    distance A C = 1 :=
by 
  sorry

end length_of_AC_l124_124785


namespace length_of_ladder_l124_124944

theorem length_of_ladder (a b : ℝ) (ha : a = 20) (hb : b = 15) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 25 := by
  sorry

end length_of_ladder_l124_124944


namespace opposite_of_83_is_84_l124_124933

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l124_124933


namespace angle_ABC_in_rhombus_l124_124037

theorem angle_ABC_in_rhombus :
  ∀ (A B C D E F : Point),
    rhombus A B C D →
    (E ∈ segment B C) →
    (F ∈ segment C D) →
    (AB = AE) →
    (AB = AF) →
    (AE = AF) →
    (AE = EF) →
    0 < FC →
    0 < DF →
    0 < BE →
    0 < EC →
    ∠ABC = 80 :=
by
  intros A B C D E F h_rhombus h_E_on_BC h_F_on_CD h_AB_AE h_AB_AF h_AE_AF h_AE_EF h_pos_FC h_pos_DF h_pos_BE h_pos_EC
  sorry

end angle_ABC_in_rhombus_l124_124037


namespace sec_cos_identity_l124_124535

theorem sec_cos_identity :
  sec (Real.pi / 9) + 3 * cos (Real.pi / 9) = 4 - 3 * sin (Real.pi / 9) ^ 2 :=
by sorry

end sec_cos_identity_l124_124535


namespace perimeter_F_is_18_l124_124402

-- Define the dimensions of the rectangles.
def vertical_rectangle : ℤ × ℤ := (3, 5)
def horizontal_rectangle : ℤ × ℤ := (1, 5)

-- Define the perimeter calculation for a single rectangle.
def perimeter (width_height : ℤ × ℤ) : ℤ :=
  2 * width_height.1 + 2 * width_height.2

-- The overlapping width and height.
def overlap_width : ℤ := 5
def overlap_height : ℤ := 1

-- Perimeter of the letter F.
def perimeter_F : ℤ :=
  perimeter vertical_rectangle + perimeter horizontal_rectangle - 2 * overlap_width

-- Statement to prove.
theorem perimeter_F_is_18 : perimeter_F = 18 := by sorry

end perimeter_F_is_18_l124_124402


namespace fruit_fly_chromosome_single_set_l124_124517

theorem fruit_fly_chromosome_single_set :
  (∀ cell : Type, (cell = 'Zygote' → cell.chromosome_sets = 2) →
                   (cell = 'SomaticCell' → cell.chromosome_sets = 2) →
                   (cell = 'Spermatogonium' → cell.chromosome_sets = 2) →
                   (cell = 'Sperm' → cell.chromosome_sets = 1)) →
  ∃ cell : Type, cell = 'Sperm' ∧ cell.chromosome_sets = 1 :=
by
  sorry

end fruit_fly_chromosome_single_set_l124_124517


namespace difference_between_largest_and_third_smallest_eq_216_l124_124851

-- Given conditions
def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5
def is_odd_digit (d : ℕ) : Prop := d % 2 = 1
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Possible combinations of digits forming a valid number condition
def valid_number (n : ℕ) : Prop :=
  ∃ d1 d2 d3 : ℕ, is_valid_digit d1 ∧ is_valid_digit d2 ∧ is_valid_digit d3 ∧
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
  n = d1 * 100 + d2 * 10 + d3 ∧ is_three_digit_number(n)

-- The largest number formed by digits 1, 3, 5
def largest_number : ℕ := 531

-- The third smallest number formed by digits 1, 3, 5
def third_smallest_number : ℕ := 315

-- The proof problem
theorem difference_between_largest_and_third_smallest_eq_216 :
  valid_number largest_number ∧ valid_number third_smallest_number →
  largest_number - third_smallest_number = 216 :=
by
  -- Proof goes here
  sorry

end difference_between_largest_and_third_smallest_eq_216_l124_124851


namespace diametrically_opposite_number_l124_124921

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l124_124921


namespace first_carpenter_alone_days_l124_124137

theorem first_carpenter_alone_days :
  (∃ (x : ℕ), 
    let first_carpenter_rate := 1 / (x + 4)
    let second_carpenter_rate := 1 / 5
    2 * (first_carpenter_rate + second_carpenter_rate) = 4 * first_carpenter_rate) ↔ x = 1 :=
by {
  let first_carpenter_rate := 1 / (x + 4)
  let second_carpenter_rate := 1 / 5
  have h : 2 * (first_carpenter_rate + second_carpenter_rate) = 4 * first_carpenter_rate,
  -- proof goes here
  sorry
}

end first_carpenter_alone_days_l124_124137


namespace circumcircle_nine_point_circle_incircle_coaxial_l124_124602

-- Define the triangle and its points
variables {A B C : Point} (Δ : Triangle A B C)
-- Define midpoints L, M, N of the triangle sides
variables (L M N : Point)
-- Define intersections P, Q, R of the incircle with the triangle sides
variables (P Q R : Point)
-- Define the circumcircle and incircle
variables (circumcircle : Circle Δ) (incircle : Circle Δ)

-- Conditions encoding the mathematical relations
def midpoints_property :=
  midpoint L B C ∧ midpoint M C A ∧ midpoint N A B

def incircle_intersections_property :=
  IncircleIntersections incircle Δ P Q R

-- Theorem statement: the circumcircle of Δ, the nine-point circle, and incircle are coaxial
theorem circumcircle_nine_point_circle_incircle_coaxial
  (h_midpoints : midpoints_property Δ L M N)
  (h_incircle_intersections : incircle_intersections_property Δ P Q R) :
  coaxial circumcircle (nine_point_circle Δ) incircle := by
    sorry

end circumcircle_nine_point_circle_incircle_coaxial_l124_124602


namespace fraction_spent_on_raw_material_l124_124494

variable (C : ℝ)
variable (x : ℝ)

theorem fraction_spent_on_raw_material :
  C - x * C - (1/10) * (C * (1 - x)) = 0.675 * C → x = 1/4 :=
by
  sorry

end fraction_spent_on_raw_material_l124_124494


namespace probability_two_students_same_event_l124_124428

theorem probability_two_students_same_event :
  let S := {1, 2, 3}    -- Represent students
  let E := {1, 2, 3}    -- Represent events High Jump, Long Jump, Shot Put
  let total_ways := 6 + 18  -- Distinct ways to choose events + ways for two students choosing same event
  let favorable_ways := 18  -- Ways for exactly two students choosing the same event
  (favorable_ways / total_ways : ℚ) = 3 / 4 :=
by
  sorry

end probability_two_students_same_event_l124_124428


namespace abs_expression_eq_l124_124655

theorem abs_expression_eq (a : ℝ) (h : a < -2) : |2 - |1 - a|| = -(1 + a) :=
by {
  -- The rewriting of must follow the logical steps but only needs to state the final equality
  sorry
}

end abs_expression_eq_l124_124655


namespace opposite_of_83_is_84_l124_124926

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l124_124926


namespace smaller_angle_at_3_40_l124_124114

theorem smaller_angle_at_3_40 : 
  ∀ (minute_angle hour_angle : ℕ) (minute_position hour_position : ℕ), 
  minute_angle = 240 → 
  hour_angle = 110 → 
  |minute_angle - hour_angle| = 130 :=
by
  intros minute_angle hour_angle minute_position hour_position hm hh
  have h1 : minute_angle = 240 := hm
  have h2 : hour_angle = 110 := hh
  rw [h1, h2]
  norm_num
  sorry

end smaller_angle_at_3_40_l124_124114


namespace intersection_of_lines_l124_124553

theorem intersection_of_lines : ∃ (x y : ℝ), (9 * x - 4 * y = 30) ∧ (7 * x + y = 11) ∧ (x = 2) ∧ (y = -3) := 
by
  sorry

end intersection_of_lines_l124_124553


namespace probability_of_purple_marble_l124_124885

theorem probability_of_purple_marble 
  (P_blue : ℝ) 
  (P_green : ℝ) 
  (P_purple : ℝ) 
  (h1 : P_blue = 0.25) 
  (h2 : P_green = 0.55) 
  (h3 : P_blue + P_green + P_purple = 1) 
  : P_purple = 0.20 := 
by 
  sorry

end probability_of_purple_marble_l124_124885


namespace intersection_M_N_l124_124274

def N : set ℤ := {x | 1/2 < 2^(x+1) ∧ 2^(x+1) < 4}
def M : set ℤ := {-1, 1}

theorem intersection_M_N :
  M ∩ N = {-1} :=
sorry

end intersection_M_N_l124_124274


namespace sec2_product_sum_l124_124866

theorem sec2_product_sum (p q : ℕ) (h₁ : 1 < p) (h₂ : 1 < q) :
  (∏ k in (finset.range 22).map (λ i, 4 * (i + 1)), sec^2 (k : ℝ) * π / 180) = p ^ q 
  → p + q = 46 :=
sorry

end sec2_product_sum_l124_124866


namespace problem_l124_124052

theorem problem (a b c d e : ℤ) 
  (h1 : a - b + c - e = 7)
  (h2 : b - c + d + e = 8)
  (h3 : c - d + a - e = 4)
  (h4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
  sorry

end problem_l124_124052


namespace count_numbers_with_square_factors_l124_124289

open Finset

theorem count_numbers_with_square_factors : 
  let numbers := range 101 \ {0}
  ∃ count, count = 50 ∧
    count = (filter (λ n, ∃ k, k^2 ∣ n ∧ k > 1) numbers).card :=
by
  sorry

end count_numbers_with_square_factors_l124_124289


namespace slower_train_pass_time_l124_124820

/--
Two trains, each 500 m long, are running in opposite directions on parallel tracks.
Their speeds are 45 km/hr and 30 km/hr respectively.
Prove that it takes approximately 24.01 seconds for the slower train to pass the driver of the faster one.
-/
theorem slower_train_pass_time :
  let train_length := 500 -- meters
  let speed_train1_kmph := 45 -- km/hr
  let speed_train2_kmph := 30 -- km/hr
  let speed_train1_mps := (speed_train1_kmph * 1000) / 3600 -- converting from km/hr to m/s
  let speed_train2_mps := (speed_train2_kmph * 1000) / 3600 -- converting from km/hr to m/s
  let relative_speed := speed_train1_mps + speed_train2_mps -- relative speed in m/s
  let time := train_length / relative_speed -- time in seconds
  time ≈ 24.01 := sorry

end slower_train_pass_time_l124_124820


namespace super_knight_cannot_visit_every_square_exactly_once_and_return_l124_124027

-- Define the dimensions of the chessboard and the knight's move
def board_width : ℕ := 12
def board_height : ℕ := 12
def knight_move : (ℤ × ℤ) := (3, 4)

-- The main theorem to prove
theorem super_knight_cannot_visit_every_square_exactly_once_and_return :
  ¬(∃ (path : list (ℤ × ℤ)),
    -- path starts at (r, c) and ends at (r, c)
    path.head = (0, 0) ∧ 
    path.last = (0, 0) ∧
    -- path covers every square exactly once
    (∀ p : ℤ × ℤ, (0 ≤ p.1 ∧ p.1 < board_height) ∧ (0 ≤ p.2 ∧ p.2 < board_width) → p ∈ path) ∧
    -- consecutive squares in path form valid knight's moves
    (∀ i < path.length - 1, let (r1, c1) := path.nth_le i sorry in
                            let (r2, c2) := path.nth_le (i + 1) sorry in
                            (abs (r2 - r1) = knight_move.1 ∧ abs (c2 - c1) = knight_move.2) ∨
                            (abs (r2 - r1) = knight_move.2 ∧ abs (c2 - c1) = knight_move.1))) :=
sorry

end super_knight_cannot_visit_every_square_exactly_once_and_return_l124_124027


namespace opposite_number_l124_124911

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l124_124911


namespace wheel_circumferences_satisfy_conditions_l124_124874

def C_f : ℝ := 24
def C_r : ℝ := 18

theorem wheel_circumferences_satisfy_conditions:
  360 / C_f = 360 / C_r + 4 ∧ 360 / (C_f - 3) = 360 / (C_r - 3) + 6 :=
by 
  have h1: 360 / C_f = 360 / C_r + 4 := sorry
  have h2: 360 / (C_f - 3) = 360 / (C_r - 3) + 6 := sorry
  exact ⟨h1, h2⟩

end wheel_circumferences_satisfy_conditions_l124_124874


namespace quadratic_range_l124_124636

open Real

theorem quadratic_range (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) : 
  a > -2 ∧ a ≠ 0 :=
by 
  sorry

end quadratic_range_l124_124636


namespace circle_sector_area_l124_124451

theorem circle_sector_area :
  let center := (4 : ℝ, 0 : ℝ)
  let radius := 10 : ℝ
  let line_slope := -2 : ℝ
  let line_intercept := 6 : ℝ
  (∃ θ : ℝ, θ = real.arctan 2 ∧
    let sector_angle := (real.pi / 2) - θ in
    let proportion := sector_angle / (2 * real.pi) in
    proportion * (real.pi * radius ^ 2) = 2.34 * real.pi) :=
by
  trivial

end circle_sector_area_l124_124451


namespace minimum_value_on_line_l124_124758

theorem minimum_value_on_line : ∃ (x y : ℝ), (x + y = 4) ∧ (∀ x' y', (x' + y' = 4) → (x^2 + y^2 ≤ x'^2 + y'^2)) ∧ (x^2 + y^2 = 8) :=
sorry

end minimum_value_on_line_l124_124758


namespace polynomials_equal_l124_124867

open Complex Polynomial

theorem polynomials_equal {p q : Polynomial ℂ} :
  (∀ z : ℂ, p.is_root z ↔ q.is_root z) →
  (∀ z : ℂ, (p + 1).is_root z ↔ (q + 1).is_root z) →
  p = q :=
by
  sorry

end polynomials_equal_l124_124867


namespace melanie_trout_catch_l124_124742

theorem melanie_trout_catch (T M : ℕ) 
  (h1 : T = 2 * M) 
  (h2 : T = 16) : 
  M = 8 :=
by
  sorry

end melanie_trout_catch_l124_124742


namespace find_distance_CD_l124_124410

theorem find_distance_CD :
  let C := (0, 0)
  let D := (6, 6 * Real.sqrt 2)
  let distance := Real.sqrt ((6 - 0)^2 + (6 * Real.sqrt 2 - 0)^2)
  C = (0, 0) ∧ D = (6, 6 * Real.sqrt 2) ∧ distance = 6 * Real.sqrt 3 := 
by
  have hC : C = (0, 0) := rfl
  have hD : D = (6, 6 * Real.sqrt 2) := rfl
  have hDist : distance = Real.sqrt (6^2 + (6 * Real.sqrt 2)^2) := 
    by rw [<- sub_zero 6, <- sub_zero (6 * Real.sqrt 2), Real.dist_eq]
  rw [distance, hDist]
  have hDistVal : distance = 6 * Real.sqrt 3 := 
    by norm_num [Real.sqrt, Real.pow_succ, Real.pow_mul]
  exact ⟨hC, hD, hDistVal⟩

end find_distance_CD_l124_124410


namespace find_x_l124_124590

theorem find_x (n : ℕ) (hn : Odd n) (hx : ∃ p1 p2 p3 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 11 ∈ {p1, p2, p3} ∧ (6^n + 1) = p1 * p2 * p3) :
  6^n + 1 = 7777 :=
by
  sorry

end find_x_l124_124590


namespace probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124825

-- Define the problem's parameters and conditions
def n : ℕ  -- number of pairs of socks

-- Define the successful pair probability calculation
theorem probability_all_pairs_successful (n : ℕ) :
  (∑ s in Finset.range (n + 1), s.factorial * 2 ^ s) / ((2 * n).factorial) = 2^n * n.factorial := sorry

-- Define the expected value calculation
theorem expected_successful_pairs_greater_than_half (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n, (1 : ℝ) / (2 * n - 1) : ℝ) / n > 0.5 := sorry

end probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124825


namespace at_least_one_good_part_l124_124083

theorem at_least_one_good_part 
  (total_parts : ℕ) 
  (good_parts : ℕ) 
  (defective_parts : ℕ) 
  (choose : ℕ → ℕ → ℕ) 
  (total_ways : ℕ) 
  (defective_ways : ℕ) 
  (result : ℕ) : 
  total_parts = 20 → 
  good_parts = 16 → 
  defective_parts = 4 → 
  choose 20 3 = total_ways → 
  choose 4 3 = defective_ways → 
  total_ways - defective_ways = result → 
  result = 1136 :=
by 
  intros;
  sorry

end at_least_one_good_part_l124_124083


namespace positive_degree_le_2n_l124_124569

-- Definition of positive degree of an algebraic integer
def positive_degree (α : ℤ) (k : ℕ) : Prop :=
  ∃ (A : matrix (fin k) (fin k) ℕ), ∃ v : (fin k) → ℤ,
    (A.mul_vec v = v) ∧ (v ≠ 0) ∧ (α = v 0)

theorem positive_degree_le_2n (n : ℕ) (α : ℤ) 
  (h1 : algebraic_on_special_field α n) : positive_degree α (2 * n) :=
sorry

end positive_degree_le_2n_l124_124569


namespace six_valid_digits_l124_124230

theorem six_valid_digits :
  {n : ℕ | n ≤ 9 ∧ n ≠ 0 ∧ 26 * n % n = 0}.to_finset.card = 6 := 
sorry

end six_valid_digits_l124_124230


namespace strawberries_left_l124_124165

theorem strawberries_left (picked: ℕ) (eaten: ℕ) (initial_count: picked = 35) (eaten_count: eaten = 2) :
  picked - eaten = 33 :=
by
  sorry

end strawberries_left_l124_124165


namespace negation_of_universal_prop_l124_124796

-- Define the conditions
variable (f : ℝ → ℝ)

-- Theorem statement
theorem negation_of_universal_prop : 
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by
  sorry

end negation_of_universal_prop_l124_124796


namespace smaller_angle_3_40_l124_124111

theorem smaller_angle_3_40 :
  let angle_per_hour := 30
  let minute_angle := 240
  let hour_angle := 3 * angle_per_hour + (2/3 : ℚ) * angle_per_hour
  let abs_diff := (240 - hour_angle).natAbs
  let smaller_angle := if abs_diff <= 180 then abs_diff else 360 - abs_diff
  smaller_angle = 130 :=
by
  sorry

end smaller_angle_3_40_l124_124111


namespace jerry_zinc_intake_l124_124328

-- Define the conditions provided in the problem
def large_antacid_weight := 2 -- in grams
def large_antacid_zinc_percent := 0.05 -- 5%

def small_antacid_weight := 1 -- in grams
def small_antacid_zinc_percent := 0.15 -- 15%

def large_antacids_count := 2
def small_antacids_count := 3

def total_zinc_mg :=
  (large_antacid_weight * large_antacid_zinc_percent * large_antacids_count +
   small_antacid_weight * small_antacid_zinc_percent * small_antacids_count) * 1000 -- converting grams to milligrams

-- The theorem to be proven
theorem jerry_zinc_intake : total_zinc_mg = 650 :=
by 
  -- translating the setup directly
  sorry -- proof will be filled here

end jerry_zinc_intake_l124_124328


namespace diametrically_opposite_number_l124_124919

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l124_124919


namespace successful_pairs_probability_expected_successful_pairs_l124_124841

-- Part (a)
theorem successful_pairs_probability {n : ℕ} : 
  (2 * n)!! = ite (n = 0) 1 ((2 * n) * (2 * n - 2) * (2 * n - 4) * ... * 2) →
  (fact (2 * n) / (2 ^ n * fact n)) = (2 ^ n * fact n / fact (2 * n)) :=
sorry

-- Part (b)
theorem expected_successful_pairs {n : ℕ} :
  (∑ k in range n, (if (successful_pair k) then 1 else 0)) / n > 0.5 :=
sorry

end successful_pairs_probability_expected_successful_pairs_l124_124841


namespace find_x_l124_124592

theorem find_x (n : ℕ) (hn : Odd n) (hx : ∃ p1 p2 p3 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 11 ∈ {p1, p2, p3} ∧ (6^n + 1) = p1 * p2 * p3) :
  6^n + 1 = 7777 :=
by
  sorry

end find_x_l124_124592


namespace sum_of_R_eq_22000_base3_l124_124007

namespace Problem

-- Definition of the set R
def R : Set ℕ := 
  { n | ∃ (a b : ℕ), a ∈ {0, 1, 2} ∧ b ∈ {0, 1, 2} ∧ n = 2 * 3^2 + a * 3^1 + b }

-- The specific problem to prove
theorem sum_of_R_eq_22000_base3 : 
  let sum_R := ∑ n in R, n
  sum_R = 22000₃ := 
sorry

end Problem

end sum_of_R_eq_22000_base3_l124_124007


namespace probability_of_longer_piece_l124_124486

noncomputable def probability_longer_piece_condition : ℝ :=
  let total_length : ℝ := 2
  let condition1 (C : ℝ) : Prop := (2 - C) ≥ 3 * C
  let condition2 (C : ℝ) : Prop := (2 - C) < 1.9
  let valid_range : set ℝ := { C | 0.1 < C ∧ C ≤ 0.5 }
  let length_of_valid_interval := 0.5 - 0.1
  let total_probability := length_of_valid_interval / total_length
  total_probability


theorem probability_of_longer_piece :
  probability_longer_piece_condition = 0.2 :=
by
  sorry

end probability_of_longer_piece_l124_124486


namespace lawn_chair_price_decrease_l124_124881

theorem lawn_chair_price_decrease (original_price sale_price difference: ℝ) 
  (h_original : original_price = 77.95)
  (h_sale : sale_price = 59.95)
  (h_difference : difference = original_price - sale_price) :
  (difference / original_price) * 100 ≈ 23.08 :=
by
  sorry

end lawn_chair_price_decrease_l124_124881


namespace max_value_of_expression_l124_124727

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h_sum : x + y + z = 3) 
  (h_order : x ≥ y ∧ y ≥ z) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 :=
sorry

end max_value_of_expression_l124_124727


namespace valerie_laptop_purchase_l124_124443

/-- Valerie wants to buy a new laptop priced at $800. She receives $100 dollars from her parents,
$60 dollars from her uncle, and $40 dollars from her siblings for her graduation.
She also makes $20 dollars each week from tutoring. How many weeks must she save 
her tutoring income, along with her graduation money, to buy the laptop? -/
theorem valerie_laptop_purchase :
  let price_of_laptop : ℕ := 800
  let graduation_money : ℕ := 100 + 60 + 40
  let weekly_tutoring_income : ℕ := 20
  let remaining_amount_needed : ℕ := price_of_laptop - graduation_money
  let weeks_needed := remaining_amount_needed / weekly_tutoring_income
  weeks_needed = 30 :=
by
  sorry

end valerie_laptop_purchase_l124_124443


namespace multiplication_identity_l124_124743

theorem multiplication_identity (x y : ℝ) : 
  (2*x^3 - 5*y^2) * (4*x^6 + 10*x^3*y^2 + 25*y^4) = 8*x^9 - 125*y^6 := 
by
  sorry

end multiplication_identity_l124_124743


namespace gift_wrapping_combinations_l124_124886

theorem gift_wrapping_combinations :
  (10 * 5 * 6 * 2 = 600) :=
by
  sorry

end gift_wrapping_combinations_l124_124886


namespace initial_investment_correct_l124_124171

-- Definitions of the parameters and conditions in the problem
def A : ℝ := 661.5
def r : ℝ := 0.10
def n : ℕ := 2
def t : ℕ := 1
def compounding_factor : ℝ := 1 + r / n
def compounding_periods : ℕ := n * t

-- The compound interest formula
def compound_interest (P : ℝ) : ℝ :=
  P * (compounding_factor ^ compounding_periods)

-- The target amount we schedule to prove
def target_amount : ℝ := 600

-- The proof statement
theorem initial_investment_correct :
  (compound_interest target_amount) = A := by
  sorry

end initial_investment_correct_l124_124171


namespace carol_first_round_points_l124_124980

theorem carol_first_round_points (P : ℤ) (h1 : P + 6 - 16 = 7) : P = 17 :=
by
  sorry

end carol_first_round_points_l124_124980


namespace johns_trip_distance_solution_l124_124701

def johns_trip_distance_problem (d t : ℝ) : Prop :=
  let t₁ : ℝ := 0.8
  let v₁ : ℝ := 50
  let v₂ : ℝ := 70
  let late : ℝ := 1.5
  let early : ℝ := 0.25
  (d = v₁ * (t + late)) →
  ((d - 40) = v₂ * (t - t₁ - early))

theorem johns_trip_distance_solution : ∃ d : ℝ, johns_trip_distance_problem d 5.425 :=
  ⟨346.25, by
    unfold johns_trip_distance_problem
    intros h₁ h₂
    let h₃ := calc
      40 + 70 * (5.425 - (0.8 + 0.25)) = 70 * 4.375
      ... = 306.25
      50 * (5.425 + 1.5) = 346.25
    sorry ⟩

end johns_trip_distance_solution_l124_124701


namespace intervals_of_monotonicity_and_critical_points_of_f_range_of_a_for_f_le_g_range_of_m_for_three_distinct_real_roots_l124_124606

theorem intervals_of_monotonicity_and_critical_points_of_f :
  ∀ x : ℝ, (0 < x ∧ x < 1 / Real.exp 1 → deriv (λ x, x * Real.log x) x < 0) ∧
           (x > 1 / Real.exp 1 → deriv (λ x, x * Real.log x) x > 0) := sorry

theorem range_of_a_for_f_le_g :
  ∀ (a : ℝ), (1 ≤ a) → ∀ x > 0, x * Real.log x ≤ a * x^2 - x := sorry

theorem range_of_m_for_three_distinct_real_roots :
  ∃ (m : ℝ), (7 / 8 < m) ∧ (m < 15 / 8 - 3 / 4 * Real.log 3) ∧
              ∀ (a : ℝ), a = 1/8 →
              ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
              (3 * (λ x, x * Real.log x) x / (4 * x) + m + (1/8 * x^2 - x) = 0) := sorry

end intervals_of_monotonicity_and_critical_points_of_f_range_of_a_for_f_le_g_range_of_m_for_three_distinct_real_roots_l124_124606


namespace cost_of_eraser_l124_124500

theorem cost_of_eraser 
  (s n c : ℕ)
  (h1 : s > 18)
  (h2 : n > 2)
  (h3 : c > n)
  (h4 : s * c * n = 3978) : 
  c = 17 :=
sorry

end cost_of_eraser_l124_124500


namespace maximum_value_of_expression_l124_124729

noncomputable def calc_value (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression :
  ∃ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧ x ≥ y ∧ y ≥ z ∧
  calc_value x y z = 2916 / 729 :=
by
  sorry

end maximum_value_of_expression_l124_124729


namespace taxi_ride_cost_l124_124887

-- Define the initial conditions
def initial_charge_peak      := 5.00
def additional_charge_peak   := 0.60
def initial_distance         := 0.25 -- 1/4 mile
def total_distance           := 12.4
def passengers               := 3
def luggage                  := 2
def additional_passenger_fee := 1.00
def luggage_fee              := 2.00

-- Calculate remaining distance
def remaining_distance := total_distance - initial_distance

-- Calculate number of additional 1/4 miles, rounded up
def num_additional_quarters := Int.ceil (remaining_distance / initial_distance)

-- Calculate additional distance charge
def distance_charge := num_additional_quarters * additional_charge_peak

-- Calculate passenger charge (only additional passengers)
def passenger_charge := (passengers - 1) * additional_passenger_fee

-- Calculate luggage charge
def luggage_charge := luggage * luggage_fee

-- Calculate total cost
def total_cost := initial_charge_peak + distance_charge + passenger_charge + luggage_charge

-- The theorem to be proved
theorem taxi_ride_cost : total_cost = 39.80 := by
  -- placeholder for proof
  sorry

end taxi_ride_cost_l124_124887


namespace inverse_propositions_l124_124879

-- Given conditions
lemma right_angles_equal : ∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90) :=
sorry

lemma equal_angles_right : ∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2) :=
sorry

-- Theorem to be proven
theorem inverse_propositions :
  (∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90)) ↔
  (∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2)) :=
sorry

end inverse_propositions_l124_124879


namespace expected_value_X_l124_124619

variable (X : Type) [ProbabilitySpace X]

-- Assume that X is a normally distributed random variable
axiom normal_dist_X : ∃ μ σ, X ∼ Normal μ σ

-- Given condition: The probability of X falling in (-3, -1) is equal to the probability of X falling in (3, 5)
axiom prob_condition : ∀ (μ σ : ℝ), X ∼ Normal μ σ →
  (probability (λ x, -3 < x ∧ x < -1) = probability (λ x, 3 < x ∧ x < 5))

-- The expected value of X
theorem expected_value_X : ∀ (μ σ : ℝ), X ∼ Normal μ σ → μ = 1 := 
by
  intros μ σ h
  sorry

end expected_value_X_l124_124619


namespace spherical_to_rectangular_correct_l124_124989

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 10 (3 * Real.pi / 4) (Real.pi / 4) = (-5, 5, 5 * Real.sqrt 2) := 
  sorry

end spherical_to_rectangular_correct_l124_124989


namespace triangle_shape_iff_l124_124307

theorem triangle_shape_iff (A B C a b c : ℝ) (h1 : sin C + sin (B - A) = sin (2 * A)) : 
  (is_isosceles_triangle A B C ∨ is_right_angle_triangle A B C) := sorry

-- Definitions needed
def is_isosceles_triangle (A B C : ℝ) : Prop := 
  (A = B) ∨ (B = C) ∨ (C = A)

def is_right_angle_triangle (A B C : ℝ) : Prop := 
  (A = π / 2) ∨ (B = π / 2) ∨ (C = π / 2)

end triangle_shape_iff_l124_124307


namespace right_isosceles_triangle_area_perimeter_l124_124684

theorem right_isosceles_triangle_area_perimeter (ABC : Type)
  (A B C : ABC)
  (h_right : ∀ (α β γ : ABC), α = A → β = B → γ = C → (∠ ABC = 90 ∧ ∠ B = ∠ C))
  (h_BC : BC = 10 * sqrt 2) :
  ∃ (area : ℝ) (perimeter : ℝ),
    area = 50 ∧ perimeter = 20 + 10 * sqrt 2 :=
by {
  sorry
}

end right_isosceles_triangle_area_perimeter_l124_124684


namespace carlos_books_in_june_l124_124972

theorem carlos_books_in_june
  (books_july : ℕ)
  (books_august : ℕ)
  (total_books_needed : ℕ)
  (books_june : ℕ) : 
  books_july = 28 →
  books_august = 30 →
  total_books_needed = 100 →
  books_june = total_books_needed - (books_july + books_august) →
  books_june = 42 :=
by intros books_july books_august total_books_needed books_june h1 h2 h3 h4
   sorry

end carlos_books_in_june_l124_124972


namespace three_digit_numbers_divisible_by_3_count_l124_124287

theorem three_digit_numbers_divisible_by_3_count:
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∀ d ∈ (Int.digits 10 n), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ n % 3 = 0} in
  S.card = 300 :=
sorry

end three_digit_numbers_divisible_by_3_count_l124_124287


namespace triangle_midpoints_parallelogram_l124_124028

theorem triangle_midpoints_parallelogram (A B C D E F : ℝ^3)
  (k α : ℝ)
  (h1 : ∥D - A∥ / ∥B - D∥ = k)
  (h2 : ∥E - B∥ / ∥C - E∥ = k)
  (h3 : ∥F - C∥ / ∥A - F∥ = k)
  (h4 : angle D A B = α)
  (h5 : angle E B C = α)
  (h6 : angle F C A = α)
  (h7 : geometric_property_is_parallelogram (midpoint A C) (midpoint D C) (midpoint B C) (midpoint E F)) :
  let P := midpoint A C,
      Q := midpoint D C,
      R := midpoint B C,
      S := midpoint E F in
  is_parallelogram P Q R S ∧
  angle P Q R = α ∧
  ∥Q - P∥ / ∥R - Q∥ = k :=
sorry

end triangle_midpoints_parallelogram_l124_124028


namespace number_of_impossible_d_values_l124_124799

noncomputable def rectangle_exceeds_triangle (d : ℕ) : Prop :=
  let w : ℕ := 3 * d - 1950 in
  d > 650

theorem number_of_impossible_d_values : 
  ∃ n, n = 650 ∧ ∀ d : ℕ, rectangle_exceeds_triangle d → d > n := 
by
  sorry

end number_of_impossible_d_values_l124_124799


namespace exams_in_fourth_year_l124_124160

noncomputable def student_exam_counts 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 ∧ 
  a_5 = 3 * a_1 ∧ 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5

theorem exams_in_fourth_year 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) (h : student_exam_counts a_1 a_2 a_3 a_4 a_5) : 
  a_4 = 8 :=
sorry

end exams_in_fourth_year_l124_124160


namespace math_problem_james_ladder_l124_124698

def convertFeetToInches(feet : ℕ) : ℕ :=
  feet * 12

def totalRungSpace(rungLength inchesApart : ℕ) : ℕ :=
  rungLength + inchesApart

def totalRungsRequired(totalHeight rungSpace : ℕ) : ℕ :=
  totalHeight / rungSpace

def totalWoodRequiredInInches(rungsRequired rungLength : ℕ) : ℕ :=
  rungsRequired * rungLength

def convertInchesToFeet(inches : ℕ) : ℕ :=
  inches / 12

def woodRequiredForRungs(feetToClimb rungLength inchesApart : ℕ) : ℕ :=
   convertInchesToFeet
    (totalWoodRequiredInInches
      (totalRungsRequired
        (convertFeetToInches feetToClimb) 
        (totalRungSpace rungLength inchesApart))
      rungLength)

theorem math_problem_james_ladder : 
  woodRequiredForRungs 50 18 6 = 37.5 :=
sorry

end math_problem_james_ladder_l124_124698


namespace valentino_farm_birds_total_l124_124753

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end valentino_farm_birds_total_l124_124753


namespace greatest_integer_x_l124_124853

theorem greatest_integer_x (x : ℤ) : 
  (∃ k : ℤ, (x - 4) = k ∧ x^2 - 3 * x + 4 = k * (x - 4) + 8) →
  x ≤ 12 :=
by
  sorry

end greatest_integer_x_l124_124853


namespace opposite_of_83_is_84_l124_124936

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l124_124936


namespace mean_greater_than_median_by_17_6_l124_124088

def weights : List ℝ := [6, 6, 8, 10, 98]
def median_weight : ℝ := 8
def mean_weight : ℝ := 25.6

theorem mean_greater_than_median_by_17_6 :
  (List.sum weights / weights.length) = mean_weight ∧
  List.median weights = median_weight ∧
  (mean_weight - median_weight) = 17.6 :=
by
  sorry

end mean_greater_than_median_by_17_6_l124_124088


namespace mr_valentino_birds_l124_124757

theorem mr_valentino_birds : 
  ∀ (chickens ducks turkeys : ℕ), 
  chickens = 200 → 
  ducks = 2 * chickens → 
  turkeys = 3 * ducks → 
  chickens + ducks + turkeys = 1800 :=
by
  intros chickens ducks turkeys
  assume h1 : chickens = 200
  assume h2 : ducks = 2 * chickens
  assume h3 : turkeys = 3 * ducks
  sorry

end mr_valentino_birds_l124_124757


namespace isosceles_triangle_angles_l124_124676

theorem isosceles_triangle_angles (A B C : ℝ) (h_iso: (A = B) ∨ (B = C) ∨ (A = C)) (angle_A : A = 50) :
  (B = 50) ∨ (B = 65) ∨ (B = 80) :=
by
  sorry

end isosceles_triangle_angles_l124_124676


namespace largest_possible_median_l124_124055

theorem largest_possible_median :
  let given_numbers := [5, 9, 3, 6, 10, 8],
      additional_numbers := [11, 12, 13, 14],
      full_list := (given_numbers ++ additional_numbers).qsort (λ a b => a < b)
  in (full_list.nth! 4 + full_list.nth! 5) / 2 = 9.5 :=
by
  sorry

end largest_possible_median_l124_124055


namespace diametrically_opposite_number_is_84_l124_124895

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l124_124895


namespace find_percent_l124_124662

theorem find_percent (x y z : ℝ) (h1 : z * (x - y) = 0.15 * (x + y)) (h2 : y = 0.25 * x) : 
  z = 0.25 := 
sorry

end find_percent_l124_124662


namespace right_triangle_altitude_divides_hypotenuse_l124_124877

theorem right_triangle_altitude_divides_hypotenuse (x : ℝ) :
  let BC := 6 * x; let AC := 5 * x; let AB := 122
  in 61 * x^2 = AB^2 → ∃ AD BD : ℝ, AD = 50 ∧ BD = 72 :=
by
  assume h : 61 * x^2 = 122^2
  -- Let BC = 6x and AC = 5x
  let BC := 6 * x
  let AC := 5 * x
  let AB := 122
  -- x^2 calculation
  have x_sq : x^2 = 244 := by
    sorry
  -- AD calculation
  let AD := AC^2 / AB
  have AD_val : AD = 50 := by
    sorry
  -- BD calculation
  let BD := BC^2 / AB
  have BD_val : BD = 72 := by
    sorry
  -- Conclusion
  use [AD, BD],
  exact ⟨AD_val, BD_val⟩

end right_triangle_altitude_divides_hypotenuse_l124_124877


namespace max_a_plus_b_plus_c_l124_124009

theorem max_a_plus_b_plus_c (a b c : ℤ)
  (hA : A = (1 / 7 : ℚ) • ![
    [-5, a],
    [b, c]])  
  (hA2 : A^2 = (1 : ℚ) • 1) : 
  a + b + c = 30 :=
by sorry

end max_a_plus_b_plus_c_l124_124009


namespace original_number_l124_124948

theorem original_number (x : ℝ) (h1 : 1.5 * x = 135) : x = 90 :=
by
  sorry

end original_number_l124_124948


namespace ant_moves_probability_l124_124170

theorem ant_moves_probability :
  let m := 73
  let n := 48
  m + n = 121 := by
  sorry

end ant_moves_probability_l124_124170


namespace math_problem_statement_l124_124791

noncomputable def integral_result (x : ℝ) : ℝ :=
  (∫ t in -1 .. x, t)

theorem math_problem_statement :
  ∀ n : ℝ, (4:ℝ) ^ n = 16 → integral_result 2 = 3 / 2 :=
by
  intros n hn
  -- Provide the proof here
  sorry

end math_problem_statement_l124_124791


namespace trigonometric_product_l124_124185

theorem trigonometric_product :
  (1 - Real.sin (Real.pi / 12)) * 
  (1 - Real.sin (5 * Real.pi / 12)) * 
  (1 - Real.sin (7 * Real.pi / 12)) * 
  (1 - Real.sin (11 * Real.pi / 12)) = 1 / 4 :=
by sorry

end trigonometric_product_l124_124185


namespace matrix_determinant_example_l124_124709

open Matrix

theorem matrix_determinant_example :
  let A := ![![2, 4], ![1, 3]] in
  det (3 • (A * A - 2 • A)) = -144 :=
by 
  let A := ![![2, 4], ![1, 3]];
  sorry

end matrix_determinant_example_l124_124709


namespace sharp_triple_72_l124_124992

-- Definition of the transformation function
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem sharp_triple_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end sharp_triple_72_l124_124992


namespace smallest_positive_a_l124_124784

variable (f : ℝ → ℝ)

def periodic_function (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ (x : ℝ), f(x - period) = f(x)

theorem smallest_positive_a (h : periodic_function f 30) :
  ∃ (a : ℝ), a > 0 ∧
  (∀ (x : ℝ), f ((x - a) / 3) = f (x / 3)) ∧
  (∀ (b : ℝ), b > 0 ∧ (∀ (x : ℝ), f ((x - b) / 3) = f (x / 3)) → b ≥ a) := 
sorry

end smallest_positive_a_l124_124784


namespace find_least_n_geq_100_l124_124712

noncomputable def is_isosceles_right_triangle (C₀ C₁ D : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := C₀
  let (x₁, y₁) := C₁
  let (xd, yd) := D
  ((xd - x₀) ^ 2 + (yd - y₀) ^ 2 = (xd - x₁) ^ 2 + (yd - y₁) ^ 2) ∧
  (yd - y₀) ^ 2 = (xd - x₀) * (xd - x₁)

theorem find_least_n_geq_100 :
  let C₀ := (0 : ℝ, 0 : ℝ)
  ∃ (C : ℕ → ℝ × ℝ) (D : ℕ → ℝ × ℝ),
    (∀ n : ℕ, C n).1 ≠ C₀.1 ∧
    (∀ n : ℕ, D n).2 = ((D n).1) ^ 2 ∧
    (∀ n : ℕ, is_isosceles_right_triangle (C (n - 1)) (C n) (D n)) ∧
    (dist C₀ (C 283) ≥ 100) := sorry

end find_least_n_geq_100_l124_124712


namespace pool_fill_time_with_P_and_Q_l124_124999

noncomputable def pool_fill_time_estimation (p q r : ℝ) (h1 : p + q + r = 1/2) 
  (h2 : p + r = 1/3) (h3 : q + r = 1/4) : ℝ :=
1 / (p + q)

theorem pool_fill_time_with_P_and_Q (p q r : ℝ) (h1 : p + q + r = 1/2) 
  (h2 : p + r = 1/3) (h3 : q + r = 1/4) : pool_fill_time_estimation p q r h1 h2 h3 = 2.4 :=
by 
  let q := 1/6
  let p := 1/4
  show pool_fill_time_estimation p q r h1 h2 h3 = 2.4
  sorry

end pool_fill_time_with_P_and_Q_l124_124999


namespace exam_max_marks_l124_124540

theorem exam_max_marks (M : ℝ) (h1: 0.30 * M = 66) : M = 220 :=
by
  sorry

end exam_max_marks_l124_124540


namespace opposite_of_83_is_84_l124_124928

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l124_124928


namespace symmetric_about_origin_l124_124570

theorem symmetric_about_origin (x y : ℝ) :
  (∀ (x y : ℝ), (x*y - x^2 = 1) → ((-x)*(-y) - (-x)^2 = 1)) :=
by
  intros x y h
  sorry

end symmetric_about_origin_l124_124570


namespace segment_length_cd_l124_124036

theorem segment_length_cd
  (AB : ℝ)
  (M : ℝ)
  (N : ℝ)
  (P : ℝ)
  (C : ℝ)
  (D : ℝ)
  (h₁ : AB = 60)
  (h₂ : N = M / 2)
  (h₃ : P = (AB - M) / 2)
  (h₄ : C = N / 2)
  (h₅ : D = P / 2) :
  |C - D| = 15 :=
by
  sorry

end segment_length_cd_l124_124036


namespace sequence_term_formula_l124_124745

theorem sequence_term_formula (n : ℕ) (h : 0 < n) :
  let a_n := λ n, (1 / n : ℚ) - (1 / (n + 2) : ℚ) in
  (λ a, (a 1 = 1 - 1/3) ∧ (a 2 = 1/2 - 1/4) ∧ (a 3 = 1/3 - 1/5) ∧ (a 4 = 1/4 - 1/6)) a_n → 
  a_n n = (1 / n) - (1 / (n + 2)) :=
sorry

end sequence_term_formula_l124_124745


namespace original_price_correct_percentage_growth_rate_l124_124362

-- Definitions and conditions
def original_price := 45
def sale_discount := 15
def price_after_discount := original_price - sale_discount

def initial_cost_before_event := 90
def final_cost_during_event := 120
def ratio_of_chickens := 2

def initial_buyers := 50
def increase_percentage := 20
def total_sales := 5460
def time_slots := 2  -- 1 hour = 2 slots of 30 minutes each

-- The problem: Prove the original price and growth rate
theorem original_price_correct (x : ℕ) : (120 / (x - 15) = 2 * (90 / x) → x = original_price) :=
by
  sorry

theorem percentage_growth_rate (m : ℕ) :
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = total_sales / (original_price - sale_discount) →
  m = increase_percentage) :=
by
  sorry

end original_price_correct_percentage_growth_rate_l124_124362


namespace spherical_to_rectangular_conversion_l124_124991

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 10 (3 * Real.pi / 4) (Real.pi / 4) = (-5, 5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l124_124991


namespace arithmetic_sequence_properties_l124_124739

variable {a : ℕ → ℕ} -- Represents the arithmetic sequence

-- Defining the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℕ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

-- Problem Statement in Lean
theorem arithmetic_sequence_properties
    (h1 : (a 2012 - 1)^3 + 2014 * a 2012 = 0)
    (h2 : (a 3 - 1)^3 + 2014 * a 3 = 4028) :
    S 2014 = 2014 ∧ a 2012 < a 3 := sorry

end arithmetic_sequence_properties_l124_124739


namespace leisure_time_is_5_hours_l124_124355

structure MadelineActivities where
  class_hrs : ℕ
  homework_hrs_per_day : ℕ
  extrac_hrs_per_session : ℕ
  extrac_days : ℕ
  tutoring_hrs_per_session : ℕ
  tutoring_days : ℕ
  work_mon : ℕ
  work_tue : ℕ
  work_thu : ℕ
  work_sat : ℕ
  sleep_weekday_per_day : ℕ
  sleep_weekend_per_day : ℕ
  exercise_per_day : ℕ
  commute_per_day : ℕ
  errands_per_week : ℕ

open Nat

noncomputable def total_hrs_spent (ma : MadelineActivities) : ℕ :=
  ma.class_hrs + 
  (ma.homework_hrs_per_day * 7) +
  (ma.extrac_hrs_per_session * ma.extrac_days) +
  (ma.tutoring_hrs_per_session * ma.tutoring_days) +
  (ma.work_mon + ma.work_tue + ma.work_thu + ma.work_sat) +
  ((ma.sleep_weekday_per_day * 5) + (ma.sleep_weekend_per_day * 2)) +
  (ma.exercise_per_day * 7) +
  (ma.commute_per_day * 7) +
  ma.errands_per_week

def total_hours_in_week : ℕ := 168

def leisure_hours (ma : MadelineActivities) : ℕ := 
  total_hours_in_week - total_hrs_spent(ma)

theorem leisure_time_is_5_hours (ma : MadelineActivities) (h_class : ma.class_hrs = 18)
  (h_homework : ma.homework_hrs_per_day = 4) (h_extrac : ma.extrac_hrs_per_session = 3)
  (h_extrac_days : ma.extrac_days = 3) (h_tutoring : ma.tutoring_hrs_per_session = 1)
  (h_tutoring_days : ma.tutoring_days = 2) (h_work_mon : ma.work_mon = 5) (h_work_tue : ma.work_tue = 4)
  (h_work_thu : ma.work_thu = 4) (h_work_sat : ma.work_sat = 7) (h_sleep_weekday : ma.sleep_weekday_per_day = 8)
  (h_sleep_weekend : ma.sleep_weekend_per_day = 10) (h_exercise : ma.exercise_per_day = 1) 
  (h_commute : ma.commute_per_day = 2) (h_errands : ma.errands_per_week = 5): 
  leisure_hours ma = 5 :=
by
  -- Insert proof here
  sorry

end leisure_time_is_5_hours_l124_124355


namespace diametrically_opposite_to_83_l124_124909

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l124_124909


namespace probability_all_successful_pairs_expected_successful_pairs_l124_124834

-- Lean statement for part (a)
theorem probability_all_successful_pairs (n : ℕ) :
  let prob_all_successful := (2 ^ n * n.factorial) / (2 * n).factorial
  in prob_all_successful = (1 / (2 ^ n)) * (n.factorial / (2 * n).factorial)
:= by sorry

-- Lean statement for part (b)
theorem expected_successful_pairs (n : ℕ) :
  let expected_value := n / (2 * n - 1)
  in expected_value > 0.5
:= by sorry

end probability_all_successful_pairs_expected_successful_pairs_l124_124834


namespace Vinnie_exceeded_word_limit_l124_124848

theorem Vinnie_exceeded_word_limit :
  let words_limit := 1000
  let words_saturday := 450
  let words_sunday := 650
  let total_words := words_saturday + words_sunday
  total_words - words_limit = 100 :=
by
  sorry

end Vinnie_exceeded_word_limit_l124_124848


namespace sequence_geometric_bounds_for_T_l124_124598

open Nat

/-- Definition of terms and sequences as per problem conditions -/
def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n-1)

def S (n : ℕ) : ℕ := ∑ i in range (n+1), a i

def c (n : ℕ) : ℕ :=
  n / a n

def T (n : ℕ) : ℕ := ∑ i in range (n+1), c i

theorem sequence_geometric :
  (∀ n ≥ 2, S n - 2 * S (n - 1) = 1) →
  (∀ n, a (n + 1) = 2 * a n) :=
by sorry

theorem bounds_for_T (n : ℕ) (m M : ℕ) :
  (∀ n, T n < 4) →
  (∀ n, T n ≥ 1) →
  (∃ M ≥ 4, ∀ n, T n < M) ∧ (∀ n, T n ≥ 1) :=
by sorry

end sequence_geometric_bounds_for_T_l124_124598


namespace maxxy_yz_l124_124658

noncomputable def max_value_ratio (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (xy + yz) / (x^2 + y^2 + z^2)

theorem maxxy_yz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  max_value_ratio x y z hx hy hz ≤ sqrt 2 / 2 :=
sorry

end maxxy_yz_l124_124658


namespace exists_x_in_interval_l124_124734

theorem exists_x_in_interval
  (x : Fin 100 → ℝ)
  (h₀ : ∀ k : Fin 100, 0 ≤ x k ∧ x k ≤ 1) :
  ∃ y ∈ Icc (0:ℝ) 1, (∑ k, |y - x k|) = 50 :=
by
  sorry

end exists_x_in_interval_l124_124734


namespace evaluate_expression_l124_124107

-- any_nonzero_num_pow_zero condition
lemma any_nonzero_num_pow_zero (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  -- exponentiation rule for nonzero numbers to the power of zero
  sorry

-- num_to_zero_power condition
lemma num_to_zero_power : 3^0 = 1 := by
  -- exponentiation rule for numbers to the zero power
  exact (Nat.cast_pow 3 0).trans (nat.pow_zero 3).symm

theorem evaluate_expression : (3^(-3))^0 + (3^0)^4 = 2 := by
  have h1 : (3^(-3))^0 = 1 := any_nonzero_num_pow_zero (3^(-3)) (by linarith [pow_ne_zero (-3) (ne_of_gt (by norm_num : 3 > 0))]),
  have h2 : (3^0)^4 = 1 := by
    rw num_to_zero_power,
    norm_num,
  linarith

end evaluate_expression_l124_124107


namespace expected_value_12_sided_die_l124_124452

theorem expected_value_12_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let n := 12 in
  let probability_each := 1 / n in
  (probability_each * ∑ x in outcomes, x) = 6.5 :=
by
  sorry

end expected_value_12_sided_die_l124_124452


namespace probability_two_white_balls_is_4_over_15_l124_124483

-- Define the conditions of the problem
def total_balls : ℕ := 15
def white_balls_initial : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 2 -- Note: Even though not explicitly required, it's part of the context

-- Calculate the probability of drawing two white balls without replacement
noncomputable def probability_two_white_balls : ℚ :=
  (white_balls_initial / total_balls) * ((white_balls_initial - 1) / (total_balls - 1))

-- The theorem to prove
theorem probability_two_white_balls_is_4_over_15 :
  probability_two_white_balls = 4 / 15 := by
  sorry

end probability_two_white_balls_is_4_over_15_l124_124483


namespace dante_coconuts_left_l124_124032

variable (Paolo : ℕ) (Dante : ℕ)

theorem dante_coconuts_left :
  Paolo = 14 →
  Dante = 3 * Paolo →
  Dante - 10 = 32 :=
by
  intros hPaolo hDante
  rw [hPaolo, hDante]
  sorry

end dante_coconuts_left_l124_124032


namespace ratio_of_circle_areas_l124_124687

variable (S L A : ℝ)

theorem ratio_of_circle_areas 
  (h1 : A = (3 / 5) * S)
  (h2 : A = (6 / 25) * L)
  : S / L = 2 / 5 :=
by
  sorry

end ratio_of_circle_areas_l124_124687


namespace storage_space_remaining_l124_124962

def total_space_remaining (first_floor second_floor: ℕ) (boxes: ℕ) : ℕ :=
  first_floor + second_floor - boxes

theorem storage_space_remaining :
  ∀ (first_floor second_floor boxes: ℕ),
  (first_floor = 2 * second_floor) →
  (boxes = 5000) →
  (boxes = second_floor / 4) →
  total_space_remaining first_floor second_floor boxes = 55000 :=
by
  intros first_floor second_floor boxes h1 h2 h3
  sorry

end storage_space_remaining_l124_124962


namespace correct_formula_l124_124276

theorem correct_formula : ∀ (x y : ℕ), (x = 1 ∧ y = 2 ∨ x = 2 ∧ y = 6 ∨ x = 3 ∧ y = 12 ∨ x = 4 ∧ y = 20 ∨ x = 5 ∧ y = 30) → y = x^2 + x := by
  intro x y h
  cases h with
  | inl hxy =>
    cases hxy with hx hy
    rw [hx, hy]
    norm_num
  | inr h => 
    cases h with
    | inl hxy =>
      cases hxy with hx hy
      rw [hx, hy]
      norm_num
    | inr h => 
      cases h with
      | inl hxy =>
        cases hxy with hx hy
        rw [hx, hy]
        norm_num
      | inr h => 
        cases h with
        | inl hxy =>
          cases hxy with hx hy
          rw [hx, hy]
          norm_num
        | inr hxy =>
          cases hxy with hx hy
          rw [hx, hy]
          norm_num

end correct_formula_l124_124276


namespace successful_pairs_probability_expected_successful_pairs_l124_124840

-- Part (a)
theorem successful_pairs_probability {n : ℕ} : 
  (2 * n)!! = ite (n = 0) 1 ((2 * n) * (2 * n - 2) * (2 * n - 4) * ... * 2) →
  (fact (2 * n) / (2 ^ n * fact n)) = (2 ^ n * fact n / fact (2 * n)) :=
sorry

-- Part (b)
theorem expected_successful_pairs {n : ℕ} :
  (∑ k in range n, (if (successful_pair k) then 1 else 0)) / n > 0.5 :=
sorry

end successful_pairs_probability_expected_successful_pairs_l124_124840


namespace boys_bound_l124_124469

open Nat

noncomputable def num_students := 1650
noncomputable def num_rows := 22
noncomputable def num_cols := 75
noncomputable def max_pairs_same_sex := 11

-- Assume we have a function that gives the number of boys.
axiom number_of_boys : ℕ
axiom col_pairs_property : ∀ (c1 c2 : ℕ), ∀ (r : ℕ), c1 ≠ c2 → r ≤ num_rows → 
  (number_of_boys ≤ max_pairs_same_sex)

theorem boys_bound : number_of_boys ≤ 920 :=
sorry

end boys_bound_l124_124469


namespace ellipse_properties_l124_124247

open Real

noncomputable def ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_properties :
  ∀ (M N : ℝ × ℝ), 
  ∃ a b x y : ℝ, ∀ e : ℝ, (a > b ∧ b > 0) ∧ (M ≠ N) ∧ (e = (sqrt 2) / 2) ∧ (ellipse a b x y)
  ∧ (∃ (x1 x2 y1 y2 : ℝ), 
  (x1 + x2 = 4 / 3) ∧ 
  (x1 * x2 = -2 / 3) ∧
  (y1 = x1 - 1) ∧
  (y2 = x2 - 1) ∧
  y = x - 1 ∧
  (distance M N = (4 * sqrt 5) / 3)) :=
by
  sorry

end ellipse_properties_l124_124247


namespace DE_less_than_EF_l124_124324

theorem DE_less_than_EF
  (A B C D E F : Type)
  [triangle A B C]
  (h_obtuse_ABC : angle A B C > π / 2)
  (h_altitude_A_D : is_altitude A D B C)
  (h_angle_bisector_A_E : is_angle_bisector A E B C)
  (h_median_A_F : is_median A F B C) :
  distance D E < distance E F := 
sorry

end DE_less_than_EF_l124_124324


namespace practice_match_allocations_count_l124_124174

theorem practice_match_allocations_count :
  ∃ (x : Fin 7 → ℕ), (∑ i, x i) = 270 ∧
                      (∀ i : Fin 4, x i % 7 = 0) ∧
                      (∀ i : Fin (7 - 4), x (i + 4) % 13 = 0) ∧
                      fintype.card {x // 
                        (∑ i, x i = 270) ∧ 
                        (∀ (i : Fin 4), x i % 7 = 0) ∧ 
                        (∀ (i : Fin (7 - 4)), x (i + 4) % 13 = 0)
                      } = 27352 :=
sorry

end practice_match_allocations_count_l124_124174


namespace seating_arrangements_count_l124_124815

/--
Twelve chairs are evenly spaced around a round table and numbered clockwise from 1 through 12.
Six married couples are to sit in the chairs with men and women alternating, 
and no one is to sit either next to or across from his/her spouse.
Prove that the number of seating arrangements is 1152.
-/
theorem seating_arrangements_count :
  let N := 12 -- Number of chairs
  let couples := 6 -- Number of couples
  let alternating (s : Fin N → ℕ) := ∀ i, s i % 2 ≠ s (i+1) % 2 -- Men and women alternate
  let no_adjacent (s : Fin N → ℕ) := ∀ i, s i ≠ (s (i+1) + N/2) % N -- No one sits next to or across from spouse
  let seating_valid (s : Fin N → ℕ) := alternating s ∧ no_adjacent s
  let total_valid_seatings := 
    (Finset.univ.filter seating_valid).card
  total_valid_seatings = 1152 := sorry

end seating_arrangements_count_l124_124815


namespace red_numbers_le_totient_l124_124444

def is_red_number (red_numbers : finset ℕ) (n : ℕ) : Prop :=
∀ a b c ∈ red_numbers, a * (b - c) % n = 0 → b = c

theorem red_numbers_le_totient (n : ℕ) (red_numbers : finset ℕ) (h : is_red_number red_numbers n) :
  red_numbers.card ≤ nat.totient n :=
sorry

end red_numbers_le_totient_l124_124444


namespace calculate_expression_l124_124179

theorem calculate_expression :
  -((1: ℝ)^2022) + ((1 / 3: ℝ)^(-2)) + |(Real.sqrt 3 - 2: ℝ)| = 10 - Real.sqrt 3 :=
  sorry

end calculate_expression_l124_124179


namespace sequences_stabilize_l124_124244

noncomputable def sequence_S_n (S₁ : List ℕ) (n : ℕ) (f : List ℕ → List ℕ) : List ℕ :=
  match n with
  | 0 => S₁
  | k+1 => f (sequence_S_n S₁ k f)

def condition (S₁ : List ℕ) : Prop :=
  ∀ (i : ℕ), i < S₁.length → S₁.get i ≤ i

def transition (S : List ℕ) : List ℕ :=
  S.map_with_index (λ i ai, S.take i |>.count (≠ ai))

theorem sequences_stabilize (S₁ : List ℕ) (h : condition S₁) :
  sequence_S_n S₁ S₁.length transition = sequence_S_n S₁ (S₁.length + 1) transition :=
sorry

end sequences_stabilize_l124_124244


namespace coffee_cups_per_week_l124_124493

theorem coffee_cups_per_week 
  (cups_per_hour_weekday : ℕ)
  (total_weekend_cups : ℕ)
  (hours_per_day : ℕ)
  (days_in_week : ℕ)
  (weekdays : ℕ)
  : cups_per_hour_weekday * hours_per_day * weekdays + total_weekend_cups = 370 :=
by
  -- Assume values based on given conditions
  have h1 : cups_per_hour_weekday = 10 := sorry
  have h2 : total_weekend_cups = 120 := sorry
  have h3 : hours_per_day = 5 := sorry
  have h4 : weekdays = 5 := sorry
  have h5 : days_in_week = 7 := sorry
  
  -- Calculate total weekday production
  have weekday_production : ℕ := cups_per_hour_weekday * hours_per_day * weekdays
  
  -- Substitute and calculate total weekly production
  have total_production : ℕ := weekday_production + total_weekend_cups
  
  -- The total production should be equal to 370
  show total_production = 370 from by
    rw [weekday_production, h1, h3, h4]
    exact sorry

  sorry

end coffee_cups_per_week_l124_124493


namespace total_receipts_for_the_day_l124_124798

-- Definitions based on conditions
def total_tickets_sold : ℕ := 522
def adult_tickets_sold : ℕ := 130
def cost_per_adult_ticket : ℕ := 15
def cost_per_child_ticket : ℕ := 8

-- Computation values 
def money_from_adult_tickets : ℕ := adult_tickets_sold * cost_per_adult_ticket
def child_tickets_sold : ℕ := total_tickets_sold - adult_tickets_sold
def money_from_child_tickets : ℕ := child_tickets_sold * cost_per_child_ticket
def total_receipts : ℕ := money_from_adult_tickets + money_from_child_tickets

-- Proof statement
theorem total_receipts_for_the_day : total_receipts = 5086 := 
by 
  -- automatic calculation of the exact value 
  simp [total_receipts, money_from_adult_tickets, money_from_child_tickets, adult_tickets_sold, cost_per_adult_ticket, child_tickets_sold, total_tickets_sold, cost_per_child_ticket]
  -- stub for correct value
  -- the exact mathematical correctness to be filled to make sure lean calculate correctly.
  sorry

end total_receipts_for_the_day_l124_124798


namespace problem_equivalent_l124_124617

variable (f : ℝ → ℝ)

theorem problem_equivalent (h₁ : ∀ x, deriv f x = deriv (deriv f) x)
                            (h₂ : ∀ x, deriv (deriv f) x < f x) : 
                            f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := sorry

end problem_equivalent_l124_124617


namespace find_x_l124_124586

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l124_124586


namespace diametrically_opposite_number_is_84_l124_124890

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l124_124890


namespace valid_x_count_correct_l124_124571

open Real

noncomputable def count_valid_x : ℕ :=
  let valid_x_values := { x : ℝ | 0 ≤ x ∧ ∃ k : ℤ, 0 ≤ k ∧ k ≤ 13 ∧ sqrt (169 - (x ^ (1 / 4))) = k }
  valid_x_values.to_finset.card

theorem valid_x_count_correct : count_valid_x = 14 := sorry

end valid_x_count_correct_l124_124571


namespace forecast_is_the_word_l124_124759

-- Conditions as definitions
def means_predict_announce_in_advance (word : String) : Prop :=
  word = "forecast"

-- Theorem statement that proves the word fulfilling the condition is "forecast"
theorem forecast_is_the_word :
  means_predict_announce_in_advance "forecast" :=
begin
  unfold means_predict_announce_in_advance,
  refl,
end

end forecast_is_the_word_l124_124759


namespace trig_frac_value_l124_124622

-- Define the main condition.
def alpha_condition : Prop :=
  ∃ α : ℝ, tan α = 2

-- The target statement to prove
theorem trig_frac_value (α : ℝ) (h : alpha_condition) : 
  (sin α + cos α) / (sin α - cos α) = 3 := 
by
  sorry

end trig_frac_value_l124_124622


namespace solve_for_x_l124_124821

theorem solve_for_x (a r s x : ℝ) (h1 : s > r) (h2 : r * (x + a) = s * (x - a)) :
  x = a * (s + r) / (s - r) :=
sorry

end solve_for_x_l124_124821


namespace fruit_salad_weight_l124_124046

theorem fruit_salad_weight (melon berries : ℝ) (h_melon : melon = 0.25) (h_berries : berries = 0.38) : melon + berries = 0.63 :=
by
  sorry

end fruit_salad_weight_l124_124046


namespace price_of_33_kg_is_663_l124_124523

-- Definitions and hypotheses:
variables (l q : ℝ)
variable (P_33 : ℝ)

-- Conditions
axiom cost_36_kg : 30 * l + 6 * q = 726
axiom cost_first_10kg : 10 * l = 200

-- The price of 33 kilograms of apples is defined based on the given conditions
noncomputable def price_33_kg : ℝ := 30 * l + 3 * q

-- The theorem stating that the calculated price equals the given correct answer
theorem price_of_33_kg_is_663 : price_33_kg l q = 663 :=
by
  have l_value : l = 20 := by
    have h := cost_first_10kg l q
    rw [mul_comm 10 l] at h
    exact eq_div_of_mul_eq 10 zero_lt_ten h
  have q_value : q = 21 := by
    have h := cost_36_kg l q
    rw [l_value, mul_comm 30 20] at h
    have step1 := tsub_eq_of_eq_add (add_right_cancel' h)
    have step2 : (726 - 600) = 6 * q := step1
    exact eq_div_of_mul_eq zero_lt_six step2
  have calc_price : price_33_kg l q = 30 * l + 3 * q := rfl
  rw [l_value, q_value] at calc_price
  simp at calc_price
  exact calc_price

end price_of_33_kg_is_663_l124_124523


namespace triangle_area_l124_124450

theorem triangle_area (base height : ℝ) (h_base : base = 25) (h_height : height = 60) : 
  (base * height) / 2 = 750 := 
by 
  rw [h_base, h_height]
  norm_num
  sorry

end triangle_area_l124_124450


namespace not_equiv_2_pi_six_and_11_pi_six_l124_124864

def polar_equiv (r θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * ↑k * Real.pi

theorem not_equiv_2_pi_six_and_11_pi_six :
  ¬ polar_equiv 2 (Real.pi / 6) (11 * Real.pi / 6) := 
sorry

end not_equiv_2_pi_six_and_11_pi_six_l124_124864


namespace point_in_second_quadrant_l124_124316

noncomputable def complex_point_quadrant : Prop :=
  let z := (1 : ℂ) / ((1 + complex.I) ^ 2 + 1)
  ∈ ((z.re < 0) ∧ (z.im > 0))

theorem point_in_second_quadrant : complex_point_quadrant :=
sorry

end point_in_second_quadrant_l124_124316


namespace factorial_zeros_in_base_16_l124_124068

theorem factorial_zeros_in_base_16 : 
  let div_count := (Nat.factorial 15).factorization 2
  4 * 2 ≤ div_count ∧ div_count < 4 * 3 :=
by
  sorry

end factorial_zeros_in_base_16_l124_124068


namespace time_to_reach_madison_l124_124960

-- Definitions based on the conditions
def map_distance : ℝ := 5 -- inches
def average_speed : ℝ := 60 -- miles per hour
def map_scale : ℝ := 0.016666666666666666 -- inches per mile

-- The time taken by Pete to arrive in Madison
noncomputable def time_to_madison := map_distance / map_scale / average_speed

-- The theorem to prove
theorem time_to_reach_madison : time_to_madison = 5 := 
by
  sorry

end time_to_reach_madison_l124_124960


namespace graduates_distribution_l124_124201

theorem graduates_distribution (n : ℕ) (k : ℕ)
    (h_n : n = 5) (h_k : k = 3)
    (h_dist : ∀ e : Fin k, ∃ g : Finset (Fin n), g.card ≥ 1) :
    ∃ d : ℕ, d = 150 :=
by
  have h_distribution := 150
  use h_distribution
  sorry

end graduates_distribution_l124_124201


namespace emily_toys_l124_124206

theorem emily_toys (initial_toys sold_toys: Nat) (h₀ : initial_toys = 7) (h₁ : sold_toys = 3) : initial_toys - sold_toys = 4 := by
  sorry

end emily_toys_l124_124206


namespace perimeter_non_shaded_region_l124_124058

def shaded_area : ℤ := 78
def large_rect_area : ℤ := 80
def small_rect_area : ℤ := 8
def total_area : ℤ := large_rect_area + small_rect_area
def non_shaded_area : ℤ := total_area - shaded_area
def non_shaded_width : ℤ := 2
def non_shaded_length : ℤ := non_shaded_area / non_shaded_width
def non_shaded_perimeter : ℤ := 2 * (non_shaded_length + non_shaded_width)

theorem perimeter_non_shaded_region : non_shaded_perimeter = 14 := 
by
  exact rfl

end perimeter_non_shaded_region_l124_124058


namespace problem_B_l124_124576

-- Definitions and Hypotheses
variables {Line Plane : Type}
variables (m n : Line) (α β γ : Plane)
variable (perp : Line → Plane → Prop) -- perpendicularity between a line and a plane
variable (parallel : Line → Line → Prop) -- parallelism between two lines
variable (plane_parallel : Plane → Plane → Prop) -- parallelism between two planes
variable (plane_perp : Plane → Plane → Prop) -- perpendicularity between two planes

-- Hypothesis for proof problem based on conditions in the problem statement
hypothesis h1 : perp m α
hypothesis h2 : parallel n α

-- The theorem to prove
theorem problem_B : perp m n :=
by sorry

end problem_B_l124_124576


namespace range_of_a_l124_124633

open Real

-- The quadratic expression
def quadratic (a x : ℝ) : ℝ := a*x^2 + 2*x + a

-- The condition of the problem
def quadratic_nonnegative_for_all (a : ℝ) := ∀ x : ℝ, quadratic a x ≥ 0

-- The theorem to be proven
theorem range_of_a (a : ℝ) (h : quadratic_nonnegative_for_all a) : a ≥ 1 :=
sorry

end range_of_a_l124_124633


namespace order_divides_exp_l124_124763

theorem order_divides_exp {x p m d : ℕ} (hp : Nat.Prime p) (hx : Nat.Coprime x p) 
  (h_ord : Nat.OrderOf x p = d) (h_exp : x ^ m ≡ 1 [MOD p]) : d ∣ m := 
by sorry

end order_divides_exp_l124_124763


namespace locus_of_midpoints_is_circle_l124_124435

open EuclideanGeometry

-- Definitions for the given problem
variables {Ω₁ Ω₂ : Circle} {A B X Y M : Point}

-- Assuming two circles intersect at points A and B
axiom circles_intersect_at_two_points (Ω₁ Ω₂ : Circle) (A B : Point) :
  A ∈ Ω₁ ∧ A ∈ Ω₂ ∧ B ∈ Ω₁ ∧ B ∈ Ω₂

-- Assuming an arbitrary line through B intersects the circles at points X and Y
axiom line_through_B_intersects_circles (Ω₁ Ω₂ : Circle) (B X Y : Point) :
  B ∈ line (X, Y) ∧ X ∈ Ω₁ ∧ Y ∈ Ω₂

-- Define the midpoint M of segment XY
noncomputable def midpoint (X Y : Point) : Point := 
  (X + Y) / 2

-- Prove that the locus of midpoints M is a circle with diameter AB
theorem locus_of_midpoints_is_circle (Ω₁ Ω₂ : Circle) (A B : Point) :
  ∃ O R, ∀ X Y, 
    (B ∈ line (X, Y) ∧ X ∈ Ω₁ ∧ Y ∈ Ω₂) →
    midpoint X Y ∈ Circle (center := O, radius := R) :=
sorry

end locus_of_midpoints_is_circle_l124_124435


namespace max_profit_jars_max_tax_value_l124_124677

-- Part a: Prove the optimal number of jars for maximum profit
theorem max_profit_jars (Q : ℝ) 
  (h : ∀ Q, Q >= 0 → (310 - 3 * Q) * Q - 10 * Q ≤ (310 - 3 * 50) * 50 - 10 * 50):
  Q = 50 :=
sorry

-- Part b: Prove the optimal tax for maximum tax revenue
theorem max_tax_value (t : ℝ) 
  (h : ∀ t, t >= 0 → ((300 * t - t^2) / 6) ≤ ((300 * 150 - 150^2) / 6)):
  t = 150 :=
sorry

end max_profit_jars_max_tax_value_l124_124677


namespace diametrically_opposite_number_is_84_l124_124889

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l124_124889


namespace simplify_and_evaluate_expr_l124_124048

theorem simplify_and_evaluate_expr (a : ℝ) (h : a = Real.sqrt 2 + 2) : 
  ( (1 / (a - 2) - 2 / (a^2 - 4)) / (a^2 - 2 * a) / (a^2 - 4) )  = Real.sqrt 2 / 2 := 
by
  rw [h, pow_two (Real.sqrt 2 + 2), Real.sqrt_mul_self,
      sub_self, mul_zero, mul_zero, div_zero]
  sorry

end simplify_and_evaluate_expr_l124_124048


namespace problem_l124_124209

noncomputable def t : ℝ := Real.cbrt 4

theorem problem (t : ℝ) : 4 * log 3 t = log 3 (4 * t) → t = Real.cbrt 4 :=
by
  intro h
  have htlog4 := eq_of_mul_log_four_eq_log_four_times h
  have ht : t^3 = 4
  from htlog4
  exact eq_cbrt_of_pow3_eq_four ht
end

end problem_l124_124209


namespace no_matching_formula_l124_124277

def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

def formula_a (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 2 * x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 - x + 4
def formula_d (x : ℕ) : ℕ := 3 * x^3 + 2 * x^2 + x + 1

theorem no_matching_formula :
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_a pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_b pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_c pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_d pair.fst) :=
by
  sorry

end no_matching_formula_l124_124277


namespace distance_beta_alpha_l124_124166

def alpha : ℂ := 0
def omega : ℂ := 0 + 3900 * complex.I
def beta : ℂ := 1170 + 1560 * complex.I

theorem distance_beta_alpha : complex.abs (beta - alpha) = 1950 := 
sorry

end distance_beta_alpha_l124_124166


namespace periodic_even_condition_l124_124349

noncomputable def f : ℝ → ℝ := sorry

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x k : ℝ, f(x) = f(x + k * p)

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x) = f(-x)

theorem periodic_even_condition :
  (periodic f 2) ∧ (even_function f) ∧ (∀ x, 2 ≤ x ∧ x ≤ 3 → f(x) = x) →
  (∀ x, -2 ≤ x ∧ x ≤ 0 → f(x) = 3 - abs(x + 1)) :=
by
  sorry

end periodic_even_condition_l124_124349


namespace camden_dogs_total_legs_l124_124534

theorem camden_dogs_total_legs :
  ∀ (Justin_dogs : ℕ), Justin_dogs = 14 →
  let Rico_dogs := Justin_dogs + 10 in
  let Camden_dogs := (3 * Rico_dogs) / 4 in
  let total_legs := Camden_dogs * 4 in
  total_legs = 72 :=
by
  intros Justin_dogs hJustin_dogs
  let Rico_dogs := Justin_dogs + 10
  let Camden_dogs := (3 * Rico_dogs) / 4
  let total_legs := Camden_dogs * 4
  have hJustin_dogs_14 : Justin_dogs = 14 := hJustin_dogs
  rw [hJustin_dogs_14] at *
  have hRico_dogs : Rico_dogs = 24 := by norm_num
  rw [hRico_dogs] at *
  have hCamden_dogs : Camden_dogs = 18 := by norm_num
  rw [hCamden_dogs] at *
  have hTotal_legs : total_legs = 72 := by norm_num
  exact hTotal_legs

end camden_dogs_total_legs_l124_124534


namespace max_intersections_l124_124741

theorem max_intersections (n : ℕ) (h_valid_streets : n = 12) (h_no_parallel : ∀ i j : ℕ, i ≠ j → ¬(parallel i j)) (h_no_three_meet : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(meet_at_single_point i j k)) : 
  let intersections := (n - 1) * n / 2 in 
  intersections = 66 := 
by 
  sorry

end max_intersections_l124_124741


namespace problem_correct_answer_l124_124235

noncomputable def f (x : ℝ) : ℝ := sin^2 (x + π / 4)

def a : ℝ := f (log 5)
def b : ℝ := f (-log 5)

theorem problem_correct_answer : a + b = 1 :=
by
  sorry

end problem_correct_answer_l124_124235


namespace distinct_ways_to_place_digits_l124_124659

theorem distinct_ways_to_place_digits : 
  let boxes := finset.finrange 5
  let digits := {1, 2, 3, 4}
  ∃ (f : boxes → option (fin 5)), 
    (∀ b, b ∈ digits ∪ {∅} ∧ 
    ∀ d₁ d₂, d₁ ≠ d₂ → f d₁ ≠ f d₂) → (boxes.card.factorial = 120) := sorry

end distinct_ways_to_place_digits_l124_124659


namespace trigonometric_identity_l124_124085

theorem trigonometric_identity (c d : ℝ) (h₁ : c = 1/2) (h₂ : d = 0) :
  ∀ θ : ℝ, cos θ * cos θ = c * cos (2 * θ) + d * cos θ :=
by
  intros θ
  rw [h₁, h₂]
  sorry

end trigonometric_identity_l124_124085


namespace find_b_when_remainder_is_constant_l124_124560

variable (b : ℝ)

def polynomial := 8 * X^3 - b * X^2 + 2 * X + 5
def divisor := X^2 - 2 * X + 2

theorem find_b_when_remainder_is_constant
  (H : ∀ (r : Polynomial ℝ), polynomial = (X^2 - 2 * X + 2) * r + (37 - 2 * b)) :
  b = 25 :=
by {
  sorry
}

end find_b_when_remainder_is_constant_l124_124560


namespace arcsin_arccos_inequality_l124_124051

theorem arcsin_arccos_inequality (x : ℝ) (h : x ∈ Icc (-1 : ℝ) (1 : ℝ)) :
  arcsin ((5 / (2 * π)) * arccos x) > arccos ((10 / (3 * π)) * arcsin x) ↔
  x ∈ (Icc (Real.cos (2 * π / 5)) (Real.cos (8 * π / 25)) ∪ Icc (Real.cos (8 * π / 25)) (Real.cos (π / 5))) :=
sorry

end arcsin_arccos_inequality_l124_124051


namespace polar_to_cartesian_coordinates_l124_124077

theorem polar_to_cartesian_coordinates : ∀ (ρ θ : ℝ), ρ = 1 → θ = π → (ρ * real.cos θ, ρ * real.sin θ) = (-1, 0) :=
by
  intros ρ θ hρ hθ
  simp [hρ, hθ]
  split
    -- Proof steps omitted here
    sorry


end polar_to_cartesian_coordinates_l124_124077


namespace count_divisible_by_n_l124_124227

theorem count_divisible_by_n :
  {n : ℕ | n ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : finset ℕ) ∧ (26 * n) % n = 0}.card = 6 :=
by
  sorry

end count_divisible_by_n_l124_124227


namespace log_inequality_l124_124353

   variable {a b : ℝ}

   theorem log_inequality (h1 : a > b) (h2 : b > 1) (h3 : a > 1) : log 2 a > log 2 b ∧ log 2 b > 0 ↔ a > b ∧ b > 1 :=
   by
     sorry
   
end log_inequality_l124_124353


namespace camden_dogs_total_legs_l124_124533

theorem camden_dogs_total_legs :
  ∀ (Justin_dogs : ℕ), Justin_dogs = 14 →
  let Rico_dogs := Justin_dogs + 10 in
  let Camden_dogs := (3 * Rico_dogs) / 4 in
  let total_legs := Camden_dogs * 4 in
  total_legs = 72 :=
by
  intros Justin_dogs hJustin_dogs
  let Rico_dogs := Justin_dogs + 10
  let Camden_dogs := (3 * Rico_dogs) / 4
  let total_legs := Camden_dogs * 4
  have hJustin_dogs_14 : Justin_dogs = 14 := hJustin_dogs
  rw [hJustin_dogs_14] at *
  have hRico_dogs : Rico_dogs = 24 := by norm_num
  rw [hRico_dogs] at *
  have hCamden_dogs : Camden_dogs = 18 := by norm_num
  rw [hCamden_dogs] at *
  have hTotal_legs : total_legs = 72 := by norm_num
  exact hTotal_legs

end camden_dogs_total_legs_l124_124533


namespace part_a_part_b_l124_124468

variables {A B C D K P Q R : Type*} [Circle A B C D]

-- Definitions for given points and conditions
variables (A B C D K P Q R : Point) 
variables (h_cyclic : CyclicQuadrilateral A B C D)
variables (h_tangent_point_B : TangentAt B K A C)
variables (h_tangent_point_D : TangentAt D K A C)
variables (h_parallel : Parallel (Line.through K B) (Line.through P Q R))

-- The proof for part (a)
theorem part_a : segment_length A B * segment_length C D = segment_length B C * segment_length A D := sorry

-- The proof for part (b)
theorem part_b : segment_length P Q = segment_length Q R := sorry

end part_a_part_b_l124_124468


namespace sequence_bound_l124_124339

theorem sequence_bound (n : ℕ) (a : ℝ) (a_seq : ℕ → ℝ) 
  (h1 : a_seq 1 = a) 
  (h2 : a_seq n = a) 
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k < n - 1 → a_seq (k + 1) ≤ (a_seq k + a_seq (k + 2)) / 2) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a_seq k ≤ a := 
by
  sorry

end sequence_bound_l124_124339


namespace smallest_n_14_l124_124545

def A (n : ℕ) : ℕ := 
  -- definition of A as concatenated decimal result of 2^{10}, 2^{20}, ..., 2^{10n}
  sorry

theorem smallest_n_14 : ∃ n : ℕ, (n ≥ 3 ∧ A n % 2^170 = 2^(10 * n) % 2^170 ∧ n = 14) :=
begin
  use 14,
  split,
  { exact nat.le_succ 13 },
  split,
  { -- here should be the proof for A 14 ≡ 2^{10*14} (mod 2^{170})
    sorry },
  { reflexivity }
end

end smallest_n_14_l124_124545


namespace problem_statement_l124_124458

def line_tangent_circle_perpendicular (O P : Point) (L : Line) : Prop :=
  is_center_circle O ∧ is_tangent_point P L ∧ is_tangent_line O P L → 
  is_perpendicular (line_through O P) L

def plane_tangent_sphere_perpendicular (O P : Point3D) (Π : Plane) : Prop :=
  is_center_sphere O ∧ is_tangent_point3D P Π ∧ is_tangent_plane O P Π → 
  is_perpendicular3D (line_through3D O P) Π

def uses_analogical_reasoning : Prop :=
  (∀ (O P : Point) (L : Line), line_tangent_circle_perpendicular O P L) →
  (∀ (O P : Point3D) (Π : Plane), plane_tangent_sphere_perpendicular O P Π) →
  analogical_reasoning (line_tangent_circle_perpendicular) (plane_tangent_sphere_perpendicular)

theorem problem_statement : uses_analogical_reasoning :=
by
  sorry

end problem_statement_l124_124458


namespace parameterize_line_l124_124794

theorem parameterize_line (r k : ℝ) 
  (H1 : ∀ t : ℝ, (3 * (r + 4 * t) - 11 = k * t + 1)) : 
  r = 4 ∧ k = 12 :=
by {
  have r_eq : (3 * r - 11 = 1),
  { specialize H1 0,
    simp at H1,
    exact H1, },
  have k_eq : (k + 1 = 13),
  { specialize H1 1,
    simp at H1,
    exact H1, },
  split,
  { linarith,},
  { linarith, },
}

end parameterize_line_l124_124794


namespace simplest_quadratic_radical_l124_124519

theorem simplest_quadratic_radical (a : ℝ) :
  let A := sqrt (0.1 * a),
      B := sqrt ((1 : ℝ) / (2 * a)),
      C := sqrt (a^3),
      D := sqrt (a^2 + 1)
  in D = sqrt (a^2 + 1) := by
  sorry

end simplest_quadratic_radical_l124_124519


namespace polar_eq_C1_tangent_line_eq_P_longest_dist_Q_l124_124685

section math_proof_problem

variable (θ : ℝ) -- θ is the parameter

def C1_param_x := 1 + cos θ
def C1_param_y := sin θ

-- Definition of point P
def P : ℝ × ℝ := (2, 0)

-- Definition of curve C2 in polar coordinates
def C2_eq (ρ θ : ℝ) : Prop := ρ * cos θ + ρ * sin θ + 3 = 0

-- Goal 1: The polar equation of curve C1 is ρ = 2 * cos θ
theorem polar_eq_C1 (ρ : ℝ) : (ρ = sqrt ((C1_param_x θ - 1)^2 + C1_param_y θ^2)) → (ρ = 2 * cos θ) := 
sorry

-- Goal 2: The polar equation of the tangent line at P is ρ * cos θ = 2
theorem tangent_line_eq_P (ρ : ℝ) (θ : ℝ) : (cond : P.2 = 0 ∧ P.1 = 2) → (ρ * cos θ = 2) :=
sorry

-- Goal 3: The point Q(a, b) on C1 with the longest distance to C2 is ( (2 + sqrt 2)/2, sqrt 2/2 )
def Q := ((2 + sqrt 2) / 2, sqrt 2 / 2)

-- Solve for Q on C1
theorem longest_dist_Q (a b : ℝ) : 
  (C1_param_x θ = a ∧ C1_param_y θ = b) ∧ ∀ (x y : ℝ), (C1_param_x θ = x ∧ C1_param_y θ = y) → 
  dist (x, y) C2_eq ≤ dist (a, b) C2_eq := 
sorry

end math_proof_problem

end polar_eq_C1_tangent_line_eq_P_longest_dist_Q_l124_124685


namespace smaller_angle_3_40_l124_124112

theorem smaller_angle_3_40 :
  let angle_per_hour := 30
  let minute_angle := 240
  let hour_angle := 3 * angle_per_hour + (2/3 : ℚ) * angle_per_hour
  let abs_diff := (240 - hour_angle).natAbs
  let smaller_angle := if abs_diff <= 180 then abs_diff else 360 - abs_diff
  smaller_angle = 130 :=
by
  sorry

end smaller_angle_3_40_l124_124112


namespace diametrically_opposite_to_83_l124_124906

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l124_124906


namespace sarah_score_l124_124377

-- Given conditions
variable (s g : ℕ) -- Sarah's score and Greg's score
variable (h1 : s = g + 60) -- Sarah's score is 60 points more than Greg's
variable (h2 : (s + g) / 2 = 130) -- The average of their two scores is 130

-- Proof statement
theorem sarah_score : s = 160 :=
by
  sorry

end sarah_score_l124_124377


namespace find_k_inequality_G_l124_124624

-- Given the function f(x) = k * x * log x (k ≠ 0) has a minimum value of -1/e.
def f (x : ℝ) (k : ℝ) : ℝ := k * x * Real.log x

-- (1) Find the value of the real number k
theorem find_k (k : ℝ) : (∃ x : ℝ, f x k = -1 / Real.exp 1) → k = 1 :=
by
  sorry

-- (2) Let real numbers a, b satisfy 0 < a < b.
variable (a b : ℝ)
hypothesis h : 0 < a ∧ a < b

-- (2) (i) Calculate integral
def G (a b : ℝ) : ℝ :=
  ∫ x in a..b, |Real.log x - Real.log ((a + b) / 2)|

-- (2) (ii) Prove the inequality
theorem inequality_G (a b : ℝ) (h : 0 < a ∧ a < b) :
  (1 / (b - a)) * G a b < Real.log 2 :=
by
  sorry

end find_k_inequality_G_l124_124624


namespace problem_statement_l124_124072

def odot (a b : ℝ) : ℝ := (a^3) / (b^2)

theorem problem_statement :
  ((odot (odot 2 4) 6) - (odot 2 (odot 4 6))) = -2591 / 288 :=
by
  sorry

end problem_statement_l124_124072


namespace maximize_profit_maximize_tax_revenue_l124_124679

open Real

-- Conditions
def inverse_demand_function (P Q : ℝ) : Prop :=
  P = 310 - 3 * Q

def production_cost_per_jar : ℝ := 10

-- Part (a): Prove that Q = 50 maximizes profit
theorem maximize_profit : (Q : ℝ) (P : ℝ) (∏ : ℝ -> ℝ) (C : ℝ -> ℝ) 
  (h_inv_demand : inverse_demand_function P Q)
  (h_revenue : (R : ℝ -> ℝ), R Q = P * Q)
  (h_cost : (C : ℝ -> ℝ), C Q = production_cost_per_jar * Q)
  (h_profit : (∏ : ℝ -> ℝ), ∏ Q = R Q - C Q)
  (h_derive : ∀ Q, deriv ∏ Q = 300 - 6 * Q) :
  Q = 50 := sorry

-- Part (b): Prove that t = 150 maximizes tax revenue for the government
theorem maximize_tax_revenue : (Q t : ℝ) (T : ℝ -> ℝ)
  (h_inv_demand : inverse_demand_function (310 - 3 * Q) Q)
  (h_tax_revenue : (T : ℝ -> ℝ), T t = Q * t)
  (h_tax_revenue_simplified : (T : ℝ -> ℝ), T t = (300 * t - t ^ 2) / 6)
  (h_derive_tax_revenue : ∀ t, deriv T t = 50 - t / 3) :
  t = 150 := sorry

end maximize_profit_maximize_tax_revenue_l124_124679


namespace opposite_of_83_is_84_l124_124925

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l124_124925


namespace min_cost_ratio_l124_124192

noncomputable def V : ℝ := sorry
noncomputable def p_iron : ℝ := sorry
noncomputable def p_aluminum := 3 * p_iron

def h (r : ℝ) : ℝ := V / (π * r^2)
def cost (r : ℝ) : ℝ :=
  p_iron * ( 2 * V / r + π * r^2 ) + p_aluminum * π * r^2

def optimal_radius : ℝ := real.cbrt (V / (4 * π))

theorem min_cost_ratio :
  ∀ r, h r ≠ 0 →
  cost r = cost optimal_radius →
  (optimal_radius / (h optimal_radius)) = 1 / 4 :=
by
  intros r hr_cost_eq
  sorry

end min_cost_ratio_l124_124192


namespace count_divisible_by_n_l124_124228

theorem count_divisible_by_n :
  {n : ℕ | n ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : finset ℕ) ∧ (26 * n) % n = 0}.card = 6 :=
by
  sorry

end count_divisible_by_n_l124_124228


namespace min_value_of_quadratic_l124_124627

-- Define the given quadratic function
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the assertion that the minimum value of the quadratic function is 29/3
theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 29/3 ∧ ∀ y : ℝ, quadratic y ≥ 29/3 :=
by
  sorry

end min_value_of_quadratic_l124_124627


namespace initial_books_l124_124151

-- Define the variables and conditions
def B : ℕ := 75
def loaned_books : ℕ := 60
def returned_books : ℕ := (70 * loaned_books) / 100
def not_returned_books : ℕ := loaned_books - returned_books
def end_of_month_books : ℕ := 57

-- State the theorem
theorem initial_books (h1 : returned_books = 42)
                      (h2 : end_of_month_books = 57)
                      (h3 : loaned_books = 60) :
  B = end_of_month_books + not_returned_books :=
by sorry

end initial_books_l124_124151


namespace sum_of_angles_of_fifth_roots_l124_124219

theorem sum_of_angles_of_fifth_roots (z : ℂ) (a : ℕ) (θ : ℝ) (r : ℝ) (θ1 θ2 θ3 θ4 θ5 : ℝ) :
  z ^ 5 = -32 * complex.I →
  (∀ k, ∃ r_k : ℝ, r_k > 0 ∧ 0 ≤ θ_k ∧ θ_k < 360 ∧ 
    (z = r_k * complex.exp (θ_k * complex.I))) →
  r = complex.abs z →
  complex.of_real r = 2 →
  a = 2 ^ 5 →
  θ = 270 →
  θ1 + θ2 + θ3 + θ4 + θ5 = 990 := sorry

end sum_of_angles_of_fifth_roots_l124_124219


namespace determinant_non_zero_expansion_l124_124774

theorem determinant_non_zero_expansion (n : ℕ) : n > 0 →
  let A := λ i j, if i = j then 0 else 1 in
  det A = n! * (∑ k in Finset.range (n + 1), (-1)^k / (k! : ℚ)) :=
begin
  sorry,
end

end determinant_non_zero_expansion_l124_124774


namespace wheat_distribution_l124_124029

def mill1_rate := 19 / 3 -- quintals per hour
def mill2_rate := 32 / 5 -- quintals per hour
def mill3_rate := 5     -- quintals per hour

def total_wheat := 1330 -- total wheat in quintals

theorem wheat_distribution :
    ∃ (x1 x2 x3 : ℚ), 
    x1 = 475 ∧ x2 = 480 ∧ x3 = 375 ∧ 
    x1 / mill1_rate = x2 / mill2_rate ∧ x2 / mill2_rate = x3 / mill3_rate ∧ 
    x1 + x2 + x3 = total_wheat :=
by {
  sorry
}

end wheat_distribution_l124_124029


namespace cubic_sum_l124_124188

variable {q : ℝ → ℝ}

noncomputable def cubic_polynomial_through_points := 
  ∀ x ∈ ({0, 1, 2, ... , 17} : set ℝ), q x + q (17 - x) = 22

theorem cubic_sum :
  (q 0) + (q 1) + (q 2) + ... + (q 17) = 198 :=
by
  sorry

end cubic_sum_l124_124188


namespace maximize_monthly_profit_l124_124688

variables (x y w : ℝ)

-- given conditions
def initial_conditions : Prop :=
  x = 80 ∧ y = 100 ∧ ∀ (d : ℝ), y = 100 + 5 * (80 - (x - d))

-- functional relationship between y and x
def relationship_y_x (x y : ℝ) : Prop :=
  y = -5 * x + 500

-- profit function
def profit (x : ℝ) : ℝ :=
  (x - 40) * (-5 * x + 500)

-- maximum profit
def maximum_profit (x : ℝ) : ℝ :=
  -5 * (x - 70) ^ 2 + 4500

-- Lean 4 statement
theorem maximize_monthly_profit : 
  initial_conditions x y →
  (relationship_y_x x y) → 
  ∃ d w, d = 80 - 70 ∧ w = 4500 :=
begin
  sorry
end

end maximize_monthly_profit_l124_124688


namespace problem_statement_l124_124237

theorem problem_statement (a b c : ℝ) (h1 : a ∈ Set.Ioi 0) (h2 : b ∈ Set.Ioi 0) (h3 : c ∈ Set.Ioi 0) (h4 : a^2 + b^2 + c^2 = 3) : 
  1 / (2 - a) + 1 / (2 - b) + 1 / (2 - c) ≥ 3 := 
sorry

end problem_statement_l124_124237


namespace six_valid_digits_l124_124229

theorem six_valid_digits :
  {n : ℕ | n ≤ 9 ∧ n ≠ 0 ∧ 26 * n % n = 0}.to_finset.card = 6 := 
sorry

end six_valid_digits_l124_124229


namespace find_x_l124_124595

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l124_124595


namespace vasya_has_greater_number_l124_124847

-- Define the two multiplications
def vasya_number : ℕ := (4 * (27 * 9))
def petya_number : ℕ := (55 * 3)

-- Theorem statement: Vasya's result is greater than Petya's result
theorem vasya_has_greater_number : vasya_number > petya_number :=
by {
  have h_vasya : vasya_number = 972 := by rfl,
  have h_petya : petya_number = 165 := by rfl,
  rw [h_vasya, h_petya],
  exact nat.lt_trans (165 : ℕ).lt_succ_self 972
}

end vasya_has_greater_number_l124_124847


namespace prop_range_a_l124_124608

theorem prop_range_a (a : ℝ) 
  (p : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → x^2 ≥ a)
  (q : ∃ (x : ℝ), x^2 + 2 * a * x + (2 - a) = 0)
  : a = 1 ∨ a ≤ -2 :=
sorry

end prop_range_a_l124_124608


namespace pyramid_volume_l124_124809

theorem pyramid_volume {P A B C : Type} (PA PB PC : ℝ)
  (h_perpendicular : PA ⊥ PB ∧ PB ⊥ PC ∧ PC ⊥ PA)
  (h_PA : PA = 2) (h_PB : PB = 3) (h_PC : PC = 4) :
  volume P A B C = 4 :=
sorry

end pyramid_volume_l124_124809


namespace multiply_powers_same_base_l124_124863

theorem multiply_powers_same_base (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end multiply_powers_same_base_l124_124863


namespace alice_min_score_for_geometry_class_l124_124814

theorem alice_min_score_for_geometry_class :
  ∀ (x5 : ℕ), 
  (84 + 88 + 82 + 79 + x5) / 5 ≥ 85 → x5 ≥ 92 :=
begin
  sorry
end

end alice_min_score_for_geometry_class_l124_124814


namespace ratio_AN_BK_ratio_NK_AB_l124_124143

-- Definitions of given conditions
variables (A N C B K : Point)
variables (triangle_ACN : Triangle A C N)
variables (circle_through_A_N : Circle A N)
variables (AC_intersection_B : AC ∩ circle_through_A_N = {B})
variables (CN_intersection_K : CN ∩ circle_through_A_N = {K})
variables (area_ratio_BCK_to_ACN : area (Triangle B C K) / area triangle_ACN = 1/4)
variables (area_ratio_BCN_to_ACK : area (Triangle B C N) / area (Triangle A C K) = 9/16)

-- Part (a): Prove ratio AN:BK = 2:1
theorem ratio_AN_BK : length (Segment A N) / length (Segment B K) = 2 :=
by sorry

-- Part (b): Prove ratio NK:AB = 2:5
theorem ratio_NK_AB : length (Segment N K) / length (Segment A B) = 2/5 :=
by sorry

end ratio_AN_BK_ratio_NK_AB_l124_124143


namespace exam_questions_upper_bound_l124_124312

theorem exam_questions_upper_bound (Q: Type) (S: Type) [Fintype Q] [Fintype S] :
  (∀ q : Q, Fintype.card { s : S // answers s q } = 4) →
  (∀ (q1 q2 : Q) (h : q1 ≠ q2), ∃! s : S, answers s q1 ∧ answers s q2) →
  (∀ s : S, ∃ q : Q, ¬ answers s q) →
  Fintype.card Q ≤ 13 :=
by
  sorry

end exam_questions_upper_bound_l124_124312


namespace geometric_seq_a3_l124_124318

theorem geometric_seq_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 6 = a 3 * r^3)
  (h2 : a 9 = a 3 * r^6)
  (h3 : a 6 = 6)
  (h4 : a 9 = 9) : 
  a 3 = 4 := 
sorry

end geometric_seq_a3_l124_124318


namespace defective_bulb_probability_l124_124135

theorem defective_bulb_probability (total_bulbs : ℕ) (pass_rate : ℚ) (h1 : total_bulbs = 24) (h2 : pass_rate = 0.875) :
  1 - pass_rate = 0.125 :=
by
  rw h2
  norm_num

end defective_bulb_probability_l124_124135


namespace ellipse_problem_l124_124248

noncomputable def ellipse_focus_eq : Prop :=
    ∃ (a b : ℝ) (h₁ : a > b) (h₂ : b > 0),
    let C := ∀ x y : ℝ, x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1 in
    let F := (sqrt 3, 0) in
    let A := (0, -2) in
    let distFocal := sqrt (a^2 + 4) = 2 * sqrt 2 in
    let equationC := (λ x y, x ^ 2 / 4 + y ^ 2 = 1) in
    let line_eq1 := (λ x y, 2 * y - sqrt 7 * x + 4 = 0) in
    let line_eq2 := (λ x y, 2 * y + sqrt 7 * x + 4 = 0) in
    ∀ (x y : ℝ), (C x y = So equationC x y) ∧ (∀ k: ℝ, (line_eq1 x y) ∨ (line_eq2 x y))

-- To assert the proof 
theorem ellipse_problem 
    (a b : ℝ) (h_ab : a > b) (h_b0 : b > 0)
    (F : ℝ × ℝ := (sqrt 3, 0))
    (A : ℝ × ℝ := (0, -2))
    (distFocal : sqrt (a^2 + 4) = 2 * sqrt 2)
    : ellipse_focus_eq :=
    sorry

end ellipse_problem_l124_124248


namespace new_ratio_is_three_half_l124_124801

theorem new_ratio_is_three_half (F J : ℕ) (h1 : F * 4 = J * 5) (h2 : J = 120) :
  ((F + 30) : ℚ) / J = 3 / 2 :=
by
  sorry

end new_ratio_is_three_half_l124_124801


namespace ab_value_is_3360_l124_124781

noncomputable def find_ab (a b : ℤ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧
  (∃ r s : ℤ, 
    (x : ℤ) → 
      (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s)) ∧ 
      (2 * r + s = -a) ∧ 
      (r^2 + 2 * r * s = b) ∧ 
      (r^2 * s = -16 * a))

theorem ab_value_is_3360 (a b : ℤ) (h : find_ab a b) : |a * b| = 3360 :=
sorry

end ab_value_is_3360_l124_124781


namespace geometric_sequence_sufficient_and_necessary_l124_124718

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_sufficient_and_necessary (a : ℕ → ℝ) (h1 : a 0 > 0) :
  (a 0 < a 1) ↔ (is_geometric_sequence a ∧ is_increasing_sequence a) :=
sorry

end geometric_sequence_sufficient_and_necessary_l124_124718


namespace seq_form_correct_l124_124065

theorem seq_form_correct : ∀ n : ℕ, 
  (λ n, (-1)^n * (4 * n - 3)) n = if n % 2 = 0 then 5 * (n // 2) + 1 else - (5 * (n // 2) + 1) :=
by
  sorry

end seq_form_correct_l124_124065


namespace polygon_perimeter_is_50_l124_124319

-- Defining the conditions
def is_right_angle (angle: ℝ) : Prop := angle = 90
def all_segments_one_foot (segments: List ℝ) : Prop := ∀ s ∈ segments, s = 1

-- Dimension of the outer rectangle
def width := 15
def height := 10
def outer_area := width * height

-- Defining the shape conditions
def complex_shape_area := 108
def staircase_and_extra_tier_subtracted_area := outer_area - complex_shape_area

-- Perimeter Function
def perimeter (w h: ℝ) : ℝ := 2 * (w + h)

-- The statement to prove
theorem polygon_perimeter_is_50 :
  -- Given conditions
  is_right_angle 90 →
  all_segments_one_foot (List.replicate 12 1) →
  outer_area = 150 →
  complex_shape_area = 108 →
  staircase_and_extra_tier_subtracted_area = 42 →
  -- Prove total perimeter
  perimeter width height = 50 :=
by 
  sorry

end polygon_perimeter_is_50_l124_124319


namespace g_difference_l124_124564

def pi (n : ℕ) : ℕ := ∏ i in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), i

def g (n : ℕ) : ℕ := pi n / n

theorem g_difference :
  g 180 - g 90 = 180^8 - 90^5 := 
sorry

end g_difference_l124_124564


namespace plane_through_line_at_distance_l124_124502

variables (A : ℝ × ℝ × ℝ) (ℓ : set (ℝ × ℝ × ℝ)) (d : ℝ)

/-- There exists a plane passing through line ℓ and at distance d from point A -/
theorem plane_through_line_at_distance (h_line : ∃ x y, (x, y, 0) ∈ ℓ) 
                                      (h_dist : d > 0):
  ∃ (P : ℝ × ℝ × ℝ → Prop), 
    (∀ p, p ∈ ℓ → P p) ∧ 
    ∃ (n : ℝ × ℝ × ℝ) (c : ℝ), 
        (∀ p, P p ↔ n.1 * p.1 + n.2 * p.2 + n.3 * p.3 = c) ∧
        abs ((n.1 * A.1 + n.2 * A.2 + n.3 * A.3 - c) / sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2)) = d :=
begin
  sorry
end

end plane_through_line_at_distance_l124_124502


namespace equation_of_line_through_P_l124_124150

theorem equation_of_line_through_P (P : (ℝ × ℝ)) (A B : (ℝ × ℝ))
  (hP : P = (1, 3))
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hA : A.2 = 0)
  (hB : B.1 = 0) :
  ∃ c : ℝ, 3 * c + 1 = 3 ∧ (3 * A.1 / c + A.2 / 6 = 1) ∧ (3 * B.1 / c + B.2 / 6 = 1) := sorry

end equation_of_line_through_P_l124_124150


namespace sasha_work_fraction_l124_124366

theorem sasha_work_fraction :
  let sasha_first := 1 / 3
  let sasha_second := 1 / 5
  let sasha_third := 1 / 15
  let total_sasha_contribution := sasha_first + sasha_second + sasha_third
  let fraction_per_car := total_sasha_contribution / 3
  fraction_per_car = 1 / 5 :=
by
  sorry

end sasha_work_fraction_l124_124366


namespace factorial_fraction_simplification_l124_124457

theorem factorial_fraction_simplification :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 := by
  sorry

end factorial_fraction_simplification_l124_124457


namespace value_of_a_l124_124320

theorem value_of_a {a : ℝ} (h₀ : 0 < a) (h₁ : ∀ (ρ θ : ℝ), ρ * (sin θ)^2 = a * cos θ) 
    (h₂ : ∀ t : ℝ, ∃ x y : ℝ, x = -2 + (sqrt 2)/2 * t ∧ y = -4 + (sqrt 2)/2 * t)
    (h₃ : ∀ t₁ t₂ : ℝ, t₁ = t₂ ∨ t₁ * t₂ = ((t₁ - t₂)^2) ∨ t₁ + t₂ = sqrt 2 * (a + 8)) :
  a = 2 :=
by
  sorry

end value_of_a_l124_124320


namespace smallest_n_for_square_and_fifth_power_l124_124455

theorem smallest_n_for_square_and_fifth_power : ∃ n : ℕ, 0 < n ∧ 
  is_perfect_square (2 * n) ∧ is_perfect_fifth_power (5 * n) ∧ n = 5000 :=
begin
  sorry
end

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

def is_perfect_fifth_power (m : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = m

end smallest_n_for_square_and_fifth_power_l124_124455


namespace proof_math_problem_l124_124106

-- Definitions based on the conditions
def pow_zero (x : ℝ) : x^0 = 1 := by
  rw [pow_zero]
  exact one

def three_pow_neg_three := (3 : ℝ)^(-3)
def three_pow_zero := (3 : ℝ)^0

-- The final statement to be proven
theorem proof_math_problem :
  (three_pow_neg_three)^0 + (three_pow_zero)^4 = 2 :=
by
  simp [three_pow_neg_three, three_pow_zero, pow_zero]
  sorry

end proof_math_problem_l124_124106


namespace scientific_notation_of_170000_l124_124388

-- Define the concept of scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (x = a * 10^n)

-- The main statement to prove
theorem scientific_notation_of_170000 : is_scientific_notation 1.7 5 170000 :=
by sorry

end scientific_notation_of_170000_l124_124388


namespace problem_statement_l124_124256

theorem problem_statement (a : ℕ → ℝ)
  (h_recur : ∀ n, n ≥ 1 → a (n + 1) = a (n - 1) / (1 + n * a (n - 1) * a n))
  (h_initial_0 : a 0 = 1)
  (h_initial_1 : a 1 = 1) :
  1 / (a 190 * a 200) = 19901 :=
by
  sorry

end problem_statement_l124_124256


namespace Sara_has_7_pears_l124_124772

variable (initial_pears : ℕ) (given_pears : ℕ)

theorem Sara_has_7_pears (h1 : initial_pears = 35) (h2 : given_pears = 28) :
  initial_pears - given_pears = 7 :=
by
  rw [h1, h2]
  rfl

end Sara_has_7_pears_l124_124772


namespace total_stickers_l124_124768

theorem total_stickers (r s t : ℕ) (h1 : r = 30) (h2 : s = 3 * r) (h3 : t = s + 20) : r + s + t = 230 :=
by sorry

end total_stickers_l124_124768


namespace periodic_function_property_l124_124722

theorem periodic_function_property
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_period : ∀ x, f (x + 2) = f x)
  (h_def1 : ∀ x, -1 ≤ x ∧ x < 0 → f x = a * x + 1)
  (h_def2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (b * x + 2) / (x + 1))
  (h_eq : f (1 / 2) = f (3 / 2)) :
  3 * a + 2 * b = -8 := by
  sorry

end periodic_function_property_l124_124722


namespace no_integer_solutions_for_2891_l124_124665

theorem no_integer_solutions_for_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) :=
sorry

end no_integer_solutions_for_2891_l124_124665


namespace solve_for_x_l124_124997

theorem solve_for_x : ∃ x : ℝ, 18^3 = (27^2 / 9) * 3^(9 * x) ∧ x = 2 / 9 :=
by
  sorry

end solve_for_x_l124_124997


namespace exists_positive_ℓ_l124_124340

theorem exists_positive_ℓ (k : ℕ) (h_prime: 0 < k) :
  ∃ ℓ : ℕ, 0 < ℓ ∧ 
  (∀ m n : ℕ, m > 0 → n > 0 → Nat.gcd m ℓ = 1 → Nat.gcd n ℓ = 1 →  m ^ m % ℓ = n ^ n % ℓ → m % k = n % k) :=
sorry

end exists_positive_ℓ_l124_124340


namespace topology_on_X_l124_124653

-- Define the universal set X
def X : Set ℕ := {1, 2, 3}

-- Sequences of candidate sets v
def v1 : Set (Set ℕ) := {∅, {1}, {3}, {1, 2, 3}}
def v2 : Set (Set ℕ) := {∅, {2}, {3}, {2, 3}, {1, 2, 3}}
def v3 : Set (Set ℕ) := {∅, {1}, {1, 2}, {1, 3}}
def v4 : Set (Set ℕ) := {∅, {1, 3}, {2, 3}, {3}, {1, 2, 3}}

-- Define the conditions that determine a topology
def isTopology (X : Set ℕ) (v : Set (Set ℕ)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧ 
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋃₀ s ∈ v) ∧
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋂₀ s ∈ v)

-- The statement we want to prove
theorem topology_on_X : 
  isTopology X v2 ∧ isTopology X v4 :=
by
  sorry

end topology_on_X_l124_124653


namespace problem_statement_l124_124064

noncomputable def f (x : ℝ) : ℝ := 2^(2*x) - 2^(x+1) + 2

def domain_f (M : set ℝ) : Prop :=
  ∀ x, f x ∈ set.Icc 1 2 ↔ x ∈ M

theorem problem_statement (M : set ℝ) (hM : domain_f M) :
  M ⊂ set.Iic 1 ∧ (1 ∈ M) ∧ (0 ∈ M) :=
by {
  sorry
}

end problem_statement_l124_124064


namespace edge_length_of_cube_l124_124304

noncomputable def cost_per_quart : ℝ := 3.20
noncomputable def coverage_per_quart : ℕ := 120
noncomputable def total_cost : ℝ := 16
noncomputable def total_coverage : ℕ := 600 -- From 5 quarts * 120 square feet per quart
noncomputable def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)

theorem edge_length_of_cube :
  (∃ edge_length : ℝ, surface_area edge_length = total_coverage) → 
  ∃ edge_length : ℝ, edge_length = 10 :=
by
  sorry

end edge_length_of_cube_l124_124304


namespace two_digit_even_integers_count_l124_124603

theorem two_digit_even_integers_count :
  ∃ (digits : List ℕ), 
    digits = [1, 3, 4, 6] ∧
    (∀ n ∈ digits, 0 < n ∧ n < 10) ∧ 
    ∀ m n ∈ digits, m ≠ n → (odd m ∨ odd n) ∧ 
    (∃ count, count = 6) := 
by
  sorry

end two_digit_even_integers_count_l124_124603


namespace feet_of_wood_required_l124_124699

def rung_length_in_inches : ℤ := 18
def spacing_between_rungs_in_inches : ℤ := 6
def height_to_climb_in_feet : ℤ := 50

def feet_per_rung := rung_length_in_inches / 12
def rungs_per_foot := 12 / spacing_between_rungs_in_inches
def total_rungs := height_to_climb_in_feet * rungs_per_foot
def total_feet_of_wood := total_rungs * feet_per_rung

theorem feet_of_wood_required :
  total_feet_of_wood = 150 :=
by
  sorry

end feet_of_wood_required_l124_124699


namespace carlos_books_in_june_l124_124979

def books_in_july : ℕ := 28
def books_in_august : ℕ := 30
def goal_books : ℕ := 100

theorem carlos_books_in_june :
  let books_in_july_august := books_in_july + books_in_august
  let books_needed_june := goal_books - books_in_july_august
  books_needed_june = 42 := 
by
  sorry

end carlos_books_in_june_l124_124979


namespace slower_train_pass_time_l124_124819

/--
Two trains, each 500 m long, are running in opposite directions on parallel tracks.
Their speeds are 45 km/hr and 30 km/hr respectively.
Prove that it takes approximately 24.01 seconds for the slower train to pass the driver of the faster one.
-/
theorem slower_train_pass_time :
  let train_length := 500 -- meters
  let speed_train1_kmph := 45 -- km/hr
  let speed_train2_kmph := 30 -- km/hr
  let speed_train1_mps := (speed_train1_kmph * 1000) / 3600 -- converting from km/hr to m/s
  let speed_train2_mps := (speed_train2_kmph * 1000) / 3600 -- converting from km/hr to m/s
  let relative_speed := speed_train1_mps + speed_train2_mps -- relative speed in m/s
  let time := train_length / relative_speed -- time in seconds
  time ≈ 24.01 := sorry

end slower_train_pass_time_l124_124819


namespace division_correct_l124_124119

theorem division_correct (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 :=
by
  sorry

end division_correct_l124_124119


namespace _l124_124981

noncomputable def charlesPictures : Prop :=
  ∀ (bought : ℕ) (drew_today : ℕ) (drew_yesterday_after_work : ℕ) (left : ℕ),
    (bought = 20) →
    (drew_today = 6) →
    (drew_yesterday_after_work = 6) →
    (left = 2) →
    (bought - left - drew_today - drew_yesterday_after_work = 6)

-- We can use this statement "charlesPictures" to represent the theorem to be proved in Lean 4.

end _l124_124981


namespace total_people_in_room_l124_124311

noncomputable def total_people (P : ℕ) : Prop :=
  (2 / 5 * P : ℝ).denom = 1 ∧ (1 / 2 * P : ℝ).denom = 1 ∧ 
  (1 / 2 * P = 32 : ℝ)

theorem total_people_in_room (P : ℕ) (h : total_people P) : P = 64 :=
begin
  sorry,
end

end total_people_in_room_l124_124311


namespace num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l124_124286

theorem num_of_tenths_in_1_9 : (1.9 / 0.1) = 19 :=
by sorry

theorem num_of_hundredths_in_0_8 : (0.8 / 0.01) = 80 :=
by sorry

end num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l124_124286


namespace solve_for_x_l124_124776

theorem solve_for_x (x : ℝ) : 27^x * 27^x * 27^x = 243^3 → x = 5 / 3 :=
by
  intro h
  sorry

end solve_for_x_l124_124776


namespace minimum_spend_on_boxes_l124_124873

noncomputable def box_length : ℕ := 20
noncomputable def box_width : ℕ := 20
noncomputable def box_height : ℕ := 12
noncomputable def cost_per_box : ℝ := 0.40
noncomputable def total_volume : ℕ := 2400000

theorem minimum_spend_on_boxes : 
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 200 :=
by
  sorry

end minimum_spend_on_boxes_l124_124873


namespace opposite_of_83_is_84_l124_124927

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l124_124927


namespace problem_statement_l124_124017

noncomputable def floor (t : ℝ) : ℤ := ⌊t⌋
noncomputable def T (t : ℝ) : ℝ := |t - floor t|

def region_S (t : ℝ) : set (ℝ × ℝ) :=
{p : ℝ × ℝ | (p.1 - T t)^2 / 4 + p.2^2 ≤ (T t)^2}

def statement_A (t : ℝ) : Prop := (0, 0) ∉ region_S t
def statement_B (t : ℝ) : Prop := 0 ≤ (real.pi * T t * 2) * T t / 2 ∧ (real.pi * T t * 2) * T t / 2 ≤ (real.pi / 2)
def statement_C (t : ℝ) : Prop := ∀ {x y}, x ≥ 0 → y ≥ 0 → (x, y) ∈ region_S t
def statement_D (t : ℝ) : Prop := ∃ c ∈ region_S t, c.1 = c.2

theorem problem_statement (t : ℝ) : ¬ statement_A t ∧ ¬ statement_B t ∧ ¬ statement_C t ∧ ¬ statement_D t :=
by sorry

end problem_statement_l124_124017


namespace find_a_plus_b_l124_124300

noncomputable def a_b_real_and_frac_condition (a b : ℝ) : Prop :=
  (a / (1 - (complex.I)) + b / (1 - 2 * (complex.I)) = (1 + 3 * (complex.I)) / 4)

theorem find_a_plus_b (a b : ℝ) (h : a_b_real_and_frac_condition a b) : a + b = 2 :=
  sorry

end find_a_plus_b_l124_124300


namespace smallest_d_l124_124783

theorem smallest_d (c d : ℕ) (h1 : c - d = 8)
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : d = 4 := by
  sorry

end smallest_d_l124_124783


namespace compute_complex_sum_l124_124984

noncomputable def ω : ℂ := complex.exp (complex.I * real.pi / 6)

theorem compute_complex_sum :
  (complex.exp (complex.I * real.pi / 6) + complex.exp (2 * complex.I * real.pi / 6) + 
  complex.exp (3 * complex.I * real.pi / 6) + complex.exp (4 * complex.I * real.pi / 6) + 
  complex.exp (5 * complex.I * real.pi / 6) + complex.exp (6 * complex.I * real.pi / 6) + 
  complex.exp (7 * complex.I * real.pi / 6) + complex.exp (8 * complex.I * real.pi / 6) + 
  complex.exp (9 * complex.I * real.pi / 6) + complex.exp (10 * complex.I * real.pi / 6) + 
  complex.exp (11 * complex.I * real.pi / 6) + complex.exp (12 * complex.I * real.pi / 6) + 5 = 5) :=
sorry

end compute_complex_sum_l124_124984


namespace successful_pairs_probability_expected_successful_pairs_l124_124837

-- Part (a)
theorem successful_pairs_probability {n : ℕ} : 
  (2 * n)!! = ite (n = 0) 1 ((2 * n) * (2 * n - 2) * (2 * n - 4) * ... * 2) →
  (fact (2 * n) / (2 ^ n * fact n)) = (2 ^ n * fact n / fact (2 * n)) :=
sorry

-- Part (b)
theorem expected_successful_pairs {n : ℕ} :
  (∑ k in range n, (if (successful_pair k) then 1 else 0)) / n > 0.5 :=
sorry

end successful_pairs_probability_expected_successful_pairs_l124_124837


namespace pyramid_surface_area_l124_124717

noncomputable def surface_area (D A B C : Point) : Real :=
  let s := 37 / 2
  let area_13_24_24 := sqrt (s * (s - 13) * (s - 24) * (s - 24)) / 2
  let faces_area := 4 * area_13_24_24
  faces_area

theorem pyramid_surface_area
  {D A B C : Point}
  (h₁ : length (A, B) = 13 ∨ length (A, B) = 24 ∨ length (A, B) = 37)
  (h₂ : length (A, C) = 13 ∨ length (A, C) = 24 ∨ length (A, C) = 37)
  (h₃ : length (B, C) = 13 ∨ length (B, C) = 24 ∨ length (B, C) = 37)
  (h₄ : length (A, D) = 13 ∨ length (A, D) = 24 ∨ length (A, D) = 37)
  (h₅ : length (B, D) = 13 ∨ length (B, D) = 24 ∨ length (B, D) = 37)
  (h₆ : length (C, D) = 13 ∨ length (C, D) = 24 ∨ length (C, D) = 37)
  (h₇ : ∀ T : Triangle, T ⊂ DABC → ¬is_equilateral T) :
  surface_area D A B C = 600.6 :=
by
  sorry

end pyramid_surface_area_l124_124717


namespace average_of_real_solutions_l124_124154

-- The definition of the quadratic equation having two real solutions
def has_two_real_solutions (b : ℝ) : Prop :=
  let discriminant := (-6 * b)^2 - 4 * 3 * 2 * b in
  discriminant ≥ 0

-- The statement we want to prove
theorem average_of_real_solutions (b : ℝ) (h : has_two_real_solutions b) :
  let roots := (3 : ℝ) * x^2 - (6 * b) * x + (2 * b) = 0 in
  let sum_of_roots := ((-(-6 * b)) / 3) in
  let average := sum_of_roots / 2 in
  average = b :=
sorry

end average_of_real_solutions_l124_124154


namespace opposite_of_83_is_84_l124_124935

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l124_124935


namespace find_intersection_point_l124_124731

theorem find_intersection_point :
  let A := (2 : ℝ, 4 : ℝ)
  let B := (-9/4 : ℝ, 81/16 : ℝ)
  let y := fun x : ℝ => x^2
  let normal_slope := -1 / (2 * 2)
  let normal_line := fun x : ℝ => (-1/4) * x + 9/2
  A.2 = y A.1 →
  normal_line A.1 = A.2 →
  normal_line B.1 = y B.1 →
  ∃ B' : ℝ × ℝ, B' = B :=
by
  intros
  exists B
  sorry

end find_intersection_point_l124_124731


namespace arithmetic_sequence_S9_l124_124612

open Nat

noncomputable def arithmetic_sequence (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_S9
  (a1 d : ℤ)
  (h : arithmetic_sequence 3 a1 d + arithmetic_sequence 4 a1 d + arithmetic_sequence 8 a1 d = 9) :
  let a5 := arithmetic_sequence 5 a1 d in
  let S9 := 9 * a5 in
  S9 = 27 :=
by
  simp [arithmetic_sequence] at h
  sorry

end arithmetic_sequence_S9_l124_124612


namespace leftmost_rectangle_is_B_l124_124998

def isLeftmostRectangle (wA wB wC wD wE : ℕ) : Prop := 
  wB < wD ∧ wB < wE

theorem leftmost_rectangle_is_B :
  let wA := 5
  let wB := 2
  let wC := 4
  let wD := 9
  let wE := 10
  let xA := 2
  let xB := 1
  let xC := 7
  let xD := 6
  let xE := 4
  let yA := 8
  let yB := 6
  let yC := 3
  let yD := 5
  let yE := 7
  let zA := 10
  let zB := 9
  let zC := 0
  let zD := 11
  let zE := 2
  isLeftmostRectangle wA wB wC wD wE :=
by
  simp only
  sorry

end leftmost_rectangle_is_B_l124_124998


namespace cookies_difference_l124_124031

theorem cookies_difference (cookies_ate : ℕ) (cookies_given : ℕ) (cookies_start : ℕ) : 
  cookies_start = 41 →
  cookies_ate = 18 →
  cookies_given = 9 →
  cookies_ate - cookies_given = 9 :=
by
  intros h1 h2 h3
  rw [h2, h3]
  norm_num

end cookies_difference_l124_124031


namespace diametrically_opposite_number_is_84_l124_124892

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l124_124892


namespace triangle_angle_proof_l124_124543

theorem triangle_angle_proof (a b c : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let C := real.arccos (0)
  let A := real.arccos (4/5)
  let B := real.arccos (3/5)
  is_angle A (36.869) ∧ is_angle B (53.130) ∧ is_angle C (90) := 
sorry

end triangle_angle_proof_l124_124543


namespace circle_radius_l124_124416

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y + 1 = 0) : 
    ∃ r : ℝ, r = 2 ∧ (x - 2)^2 + (y - 1)^2 = r^2 :=
by
  sorry

end circle_radius_l124_124416


namespace prob_all_successful_pairs_l124_124830

theorem prob_all_successful_pairs (n : ℕ) : 
  let num_pairs := (2 * n)!
  let num_successful_pairs := 2^n * n!
  prob_all_successful_pairs := num_successful_pairs / num_pairs 
  prob_all_successful_pairs = (2^n * n!) / (2 * n)! := 
sorry

end prob_all_successful_pairs_l124_124830


namespace geometric_sequence_term_three_eq_one_l124_124615

variable {a : ℕ → ℝ}

-- Condition: All terms are positive
def all_positive (a : ℕ → ℝ) := ∀ n, 0 < a n

-- Condition: Geometric sequence property
def is_geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = a 1 * a n / a 0

-- Condition: The product of the first n terms
def product_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in range (n+1), a i

-- Given T5 = 1
def T_5_eq_one (a : ℕ → ℝ) := product_of_first_n_terms a 5 = 1

theorem geometric_sequence_term_three_eq_one (a : ℕ → ℝ) (h1 : all_positive a) (h2 : is_geometric_sequence a) (h3 : T_5_eq_one a) : a 3 = 1 := 
by 
  sorry

end geometric_sequence_term_three_eq_one_l124_124615


namespace mr_valentino_birds_l124_124756

theorem mr_valentino_birds : 
  ∀ (chickens ducks turkeys : ℕ), 
  chickens = 200 → 
  ducks = 2 * chickens → 
  turkeys = 3 * ducks → 
  chickens + ducks + turkeys = 1800 :=
by
  intros chickens ducks turkeys
  assume h1 : chickens = 200
  assume h2 : ducks = 2 * chickens
  assume h3 : turkeys = 3 * ducks
  sorry

end mr_valentino_birds_l124_124756


namespace sum_proper_divisors_560_l124_124456

theorem sum_proper_divisors_560 : 
  let n := 560
  ∃ (sum_divisors : ℕ), 
    (∑ d in (Finset.filter (λ d, d < n) (Finset.divisors n)), d) = 928 := 
by
  let n := 560
  exists 928
  sorry

end sum_proper_divisors_560_l124_124456


namespace parametric_inclination_l124_124542

noncomputable def angle_of_inclination (x y : ℝ) : ℝ := 50

theorem parametric_inclination (t : ℝ) (x y : ℝ) :
  x = t * Real.sin 40 → y = -1 + t * Real.cos 40 → angle_of_inclination x y = 50 :=
by
  intros hx hy
  -- This is where the proof would go, but we skip it.
  sorry

end parametric_inclination_l124_124542


namespace hyperbola_satisfies_eccentricity_condition_l124_124945

noncomputable def hyperbola_eccentricity_ge_sqrt5_plus_1_div_2 (a b : ℝ) (e : ℝ) : Prop :=
  e = c / a ∧ c = sqrt (a^2 + b^2) → e ≥ (sqrt 5 + 1) / 2

theorem hyperbola_satisfies_eccentricity_condition
  (a b c t : ℝ) (e : ℝ)
  (h1 : a > 0) (h2 : b > 0)
  (h_hyperbola_eq : c = sqrt (a^2 + b^2))
  (h_AC_dot_BC_eq_0 : t^2 = b^4 / a^2 - c^2) :
  hyperbola_eccentricity_ge_sqrt5_plus_1_div_2 a b e :=
by
  intro h_ec
  sorry

end hyperbola_satisfies_eccentricity_condition_l124_124945


namespace cos_pi_over_12_minus_sin_pi_over_12_eq_sqrt2_over_2_l124_124474

theorem cos_pi_over_12_minus_sin_pi_over_12_eq_sqrt2_over_2 : 
  cos (Real.pi / 12) - sin (Real.pi / 12) = Real.sqrt 2 / 2 := 
by
  sorry

end cos_pi_over_12_minus_sin_pi_over_12_eq_sqrt2_over_2_l124_124474


namespace nonnegative_int_ternary_count_l124_124288

theorem nonnegative_int_ternary_count :
  (∃ b : Fin 10 → ℤ, ∀ i : Fin 10, b i ∈ {-1, 0, 1} ∧ (0 ≤ ∑ i, b i * 3^i ∧ (∑ i, b i * 3^i) ≤ 29524)) → 
  card { n : ℕ | ∃ b : Fin 10 → ℤ, n = ∑ i, b i * 3^i ∧ ∀ i : Fin 10, b i ∈ {-1, 0, 1} ∧ 0 ≤ n } = 29525 := 
by
  sorry

end nonnegative_int_ternary_count_l124_124288


namespace A_investment_amount_l124_124164

theorem A_investment_amount
  (B_investment : ℝ) (C_investment : ℝ) 
  (total_profit : ℝ) (A_profit_share : ℝ)
  (h1 : B_investment = 4200)
  (h2 : C_investment = 10500)
  (h3 : total_profit = 14200)
  (h4 : A_profit_share = 4260) :
  ∃ (A_investment : ℝ), 
    A_profit_share / total_profit = A_investment / (A_investment + B_investment + C_investment) ∧ 
    A_investment = 6600 :=
by {
  sorry  -- Proof not required per instructions
}

end A_investment_amount_l124_124164


namespace find_z_plus_1_over_y_l124_124053

theorem find_z_plus_1_over_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 20) : 
  z + 1 / y = 29 / 139 := 
by 
  sorry

end find_z_plus_1_over_y_l124_124053


namespace probability_C_after_B_given_A_before_B_l124_124878

noncomputable theory

/-- 
Given 2018 people standing in a line, each permutation equally likely, 
and that person A stands before person B, the probability that person C stands after person B 
is 1/3.
-/
theorem probability_C_after_B_given_A_before_B : 
  ∀ (n : ℕ), (n ≥ 3) → (perm : Finset (Fin n)), 
  (h : (perm.card = n)),
  (∃ A B C : Fin n, A ≠ B ∧ B ≠ C ∧ A ≠ C),
  (hAB : A < B),
  (P : ℚ),
  P = 1 / 3 :=
sorry

end probability_C_after_B_given_A_before_B_l124_124878


namespace additional_area_grazed_l124_124869

theorem additional_area_grazed (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 25) : 
  π * r₂ ^ 2 - π * r₁ ^ 2 = 481 * π :=
by
  have A1 := π * (12:ℝ) ^ 2,
  rw [h₁] at A1,
  have A2 := π * (25:ℝ) ^ 2,
  rw [h₂] at A2,
  calc
  π * r₂ ^ 2 - π * r₁ ^ 2 = 625 * π - 144 * π : by rwa [← A2, ← A1]
  ... = (625 - 144) * π : by rw [sub_mul]
  ... = 481 * π : by norm_num

end additional_area_grazed_l124_124869


namespace cone_base_radius_l124_124259

/--
Given a cone with the following properties:
1. The surface area of the cone is \(3\pi\).
2. The lateral surface of the cone unfolds into a semicircle (which implies the slant height is twice the radius of the base).
Prove that the radius of the base of the cone is \(1\).
-/
theorem cone_base_radius 
  (S : ℝ)
  (r l : ℝ)
  (h1 : S = 3 * Real.pi)
  (h2 : l = 2 * r)
  : r = 1 := 
  sorry

end cone_base_radius_l124_124259


namespace correct_value_l124_124141

theorem correct_value (x : ℝ) (h : x / 3.6 = 2.5) : (x * 3.6) / 2 = 16.2 :=
by {
  -- Proof would go here
  sorry
}

end correct_value_l124_124141


namespace carnival_activity_order_l124_124807

theorem carnival_activity_order :
  let dodgeball := 3 / 8
  let magic_show := 9 / 24
  let petting_zoo := 1 / 3
  let face_painting := 5 / 12
  let ordered_activities := ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"]
  (face_painting > dodgeball) ∧ (dodgeball = magic_show) ∧ (magic_show > petting_zoo) →
  ordered_activities = ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"] :=
by {
  sorry
}

end carnival_activity_order_l124_124807


namespace power_sum_is_two_l124_124100

theorem power_sum_is_two :
  (3 ^ (-3) ^ 0 + (3 ^ 0) ^ 4) = 2 := by
    sorry

end power_sum_is_two_l124_124100


namespace dante_coconuts_l124_124034

theorem dante_coconuts (P : ℕ) (D : ℕ) (S : ℕ) (hP : P = 14) (hD : D = 3 * P) (hS : S = 10) :
  (D - S) = 32 :=
by
  sorry

end dante_coconuts_l124_124034


namespace probability_of_green_l124_124539

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

end probability_of_green_l124_124539


namespace diametrically_opposite_number_l124_124920

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l124_124920


namespace solve_fractional_equation_l124_124778

theorem solve_fractional_equation (x : ℝ) (h : (3 * x + 6) / (x ^ 2 + 5 * x - 6) = (3 - x) / (x - 1)) (hx : x ≠ 1) : x = -4 := 
sorry

end solve_fractional_equation_l124_124778


namespace total_weekly_cups_brewed_l124_124491

-- Define the given conditions
def weekday_cups_per_hour : ℕ := 10
def weekend_total_cups : ℕ := 120
def shop_open_hours_per_day : ℕ := 5
def weekdays_in_week : ℕ := 5

-- Prove the total number of coffee cups brewed in one week
theorem total_weekly_cups_brewed : 
  (weekday_cups_per_hour * shop_open_hours_per_day * weekdays_in_week) 
  + weekend_total_cups = 370 := 
by
  sorry

end total_weekly_cups_brewed_l124_124491


namespace CE2_DE2_sum_l124_124008

-- Definition of the circle and its properties
structure Circle where
  O : Point
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 5 * Real.sqrt 2}
def B : Point := {x := 0, y := -5 * Real.sqrt 2}
def E : Point := {x := 0, y := 2 * Real.sqrt 5}

-- Angle AEC
def angle_AEC : ℝ := Real.pi / 4  -- 45 degrees in radians

-- Geometry context
def circle : Circle := { O := {x := 0, y := 0}, radius :=  5 * Real.sqrt 2}

-- Definition of chord CD and its properties
structure Chord where
  C : Point
  D : Point
  intersects_at_E : Prop

def chord_CD : Chord := {
  C := {x := 0, y := 2 * Real.sqrt 5},
  D := {x := 0, y := -2 * Real.sqrt 5},
  intersects_at_E := true
}

-- The main theorem
theorem CE2_DE2_sum (circle : Circle) (A B E : Point) (chord_CD : Chord)
  (h1 : dist circle.O A = circle.radius)
  (h2 : dist circle.O B = circle.radius)
  (h3 : dist B E = 2 * Real.sqrt 5)
  (h4 : ∠ AEC = angle_AEC) :
  (dist chord_CD.C E) ^ 2 + (dist chord_CD.D E) ^ 2 = 100 :=
sorry

end CE2_DE2_sum_l124_124008


namespace female_managers_count_l124_124670

-- Definitions for the given conditions
def total_employees (E : ℕ) : Prop := true
def female_employees (F : ℕ) : Prop := F = 500
def fraction_managers (x : ℝ) : Prop := x = 2 / 5
def total_managers (E : ℕ) (x : ℝ) : ℕ := (x * E).natAbs
def male_employees (E : ℕ) (F : ℕ) : ℕ := E - F
def male_managers (E F : ℕ) (x : ℝ) : ℕ := (x * (E - F)).natAbs
def female_managers (E F : ℕ) (x : ℝ) : ℕ := total_managers E x - male_managers E F x

-- The theorem to prove the number of female managers is 200
theorem female_managers_count (E F : ℕ) (x : ℝ) (hF : female_employees F) (hx : fraction_managers x) (hE : total_employees E) : 
  female_managers E F x = 200 :=
  sorry

end female_managers_count_l124_124670


namespace all_points_covered_by_circle_l124_124577

theorem all_points_covered_by_circle {n : ℕ} (points : Fin n → ℝ × ℝ)
  (h : ∀ (a b c : Fin n), ∃ (center : ℝ × ℝ) (radius : ℝ), radius = 1 ∧ 
  (∥ center - points a ∥ ≤ radius) ∧
  (∥ center - points b ∥ ≤ radius) ∧
  (∥ center - points c ∥ ≤ radius)) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), radius = 1 ∧
  ∀ (i : Fin n), ∥ center - points i ∥ ≤ radius :=
by
  sorry

end all_points_covered_by_circle_l124_124577


namespace successful_pairs_probability_expected_successful_pairs_l124_124839

-- Part (a)
theorem successful_pairs_probability {n : ℕ} : 
  (2 * n)!! = ite (n = 0) 1 ((2 * n) * (2 * n - 2) * (2 * n - 4) * ... * 2) →
  (fact (2 * n) / (2 ^ n * fact n)) = (2 ^ n * fact n / fact (2 * n)) :=
sorry

-- Part (b)
theorem expected_successful_pairs {n : ℕ} :
  (∑ k in range n, (if (successful_pair k) then 1 else 0)) / n > 0.5 :=
sorry

end successful_pairs_probability_expected_successful_pairs_l124_124839


namespace induction_inequality_l124_124095

theorem induction_inequality (n : Nat) (h1 : 1 < n) : 1 + ∑ k in Finset.range (2 * n - 1), (1 / (k + 1 : ℚ)) < n := 
by
  sorry

# Base case verification
# For n = 2, the left side should equal 1 + 1/2 + 1/3

example : 1 + 1 / 2 + 1 / 3 = 1 + 1 / 2 + 1 / 3 := 
by
  rfl

end induction_inequality_l124_124095


namespace exact_time_now_l124_124695

theorem exact_time_now (t : ℝ) (h₀ : 0 ≤ t ∧ t ≤ 60)
    (h₁ : |6 * (t + 5) - (90 + 0.5 * (t - 4))| = 178) : t ≈ 42.91 :=
by
  sorry

end exact_time_now_l124_124695


namespace find_number_l124_124529

theorem find_number (x : ℕ) (h : (537 - x) / (463 + x) = 1 / 9) : x = 437 :=
sorry

end find_number_l124_124529


namespace sum_of_first_9_terms_zero_l124_124224

variable (a_n : ℕ → ℝ) (d a₁ : ℝ)
def arithmetic_seq := ∀ n, a_n n = a₁ + (n - 1) * d

def condition (a_n : ℕ → ℝ) := (a_n 2 + a_n 9 = a_n 6)

theorem sum_of_first_9_terms_zero 
  (h_arith : arithmetic_seq a_n d a₁) 
  (h_cond : condition a_n) : 
  (9 * a₁ + (9 * 8 / 2) * d) = 0 :=
by
  sorry

end sum_of_first_9_terms_zero_l124_124224


namespace total_weekly_cups_brewed_l124_124490

-- Define the given conditions
def weekday_cups_per_hour : ℕ := 10
def weekend_total_cups : ℕ := 120
def shop_open_hours_per_day : ℕ := 5
def weekdays_in_week : ℕ := 5

-- Prove the total number of coffee cups brewed in one week
theorem total_weekly_cups_brewed : 
  (weekday_cups_per_hour * shop_open_hours_per_day * weekdays_in_week) 
  + weekend_total_cups = 370 := 
by
  sorry

end total_weekly_cups_brewed_l124_124490


namespace inequality_holds_l124_124711

theorem inequality_holds (a b : ℕ) (ha : a > 1) (hb : b > 2) : a ^ b + 1 ≥ b * (a + 1) :=
sorry

end inequality_holds_l124_124711


namespace tangent_line_equation_l124_124790

-- Define the given function
def f (x : ℝ) : ℝ := 2 - x * Real.exp x

-- Define the point where the tangent is evaluated
def p : ℝ × ℝ := (0, 2)

-- State the proof problem
theorem tangent_line_equation : ∃ (m b : ℝ), (f p.fst - p.snd) = 0 ∧ m = f' p.fst ∧ 
    (∀ x y : ℝ, y = m * x + b ↔ x + y - b = 0) := sorry

end tangent_line_equation_l124_124790


namespace oranges_to_friend_is_two_l124_124485

-- Definitions based on the conditions.

def initial_oranges : ℕ := 12

def oranges_to_brother (n : ℕ) : ℕ := n / 3

def remainder_after_brother (n : ℕ) : ℕ := n - oranges_to_brother n

def oranges_to_friend (n : ℕ) : ℕ := remainder_after_brother n / 4

-- Theorem stating the problem to be proven.
theorem oranges_to_friend_is_two : oranges_to_friend initial_oranges = 2 :=
sorry

end oranges_to_friend_is_two_l124_124485


namespace dice_probability_l124_124231

theorem dice_probability :
  let six_sided_die := { n : ℕ | 1 ≤ n ∧ n ≤ 6 }
  let outcomes := { (d1, d2, d3, d4) : six_sided_die × six_sided_die × six_sided_die × six_sided_die | 
                   d1 ∈ six_sided_die ∧ d2 ∈ six_sided_die ∧ d3 ∈ six_sided_die ∧ d4 ∈ six_sided_die }
  let favorable_outcomes := { (d1, d2, d3, d4) ∈ outcomes | d1 = d2 + d3 + d4 ∨ d2 = d1 + d3 + d4 ∨ d3 = d1 + d2 + d4 ∨ d4 = d1 + d2 + d3 }
  in (↑(favorable_outcomes.size) / ↑(outcomes.size)) = (10 / 27) :=
sorry

end dice_probability_l124_124231


namespace prob_all_successful_pairs_l124_124827

theorem prob_all_successful_pairs (n : ℕ) : 
  let num_pairs := (2 * n)!
  let num_successful_pairs := 2^n * n!
  prob_all_successful_pairs := num_successful_pairs / num_pairs 
  prob_all_successful_pairs = (2^n * n!) / (2 * n)! := 
sorry

end prob_all_successful_pairs_l124_124827


namespace tangent_circumcircle_A_l124_124127

variables {A B C X Y P Q A' O : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]
[InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]

-- Triangle ABC
variable (triangleABC : A × B × C)

-- Points X and Y on BC
variable (XY_on_BC : X ∈ Segment (B, C) ∧ Y ∈ Segment (B, C) ∧ Between B X Y)

-- Condition: 2XY = BC
variable (double_XY_eq_BC : 2 * (distance X Y) = distance B C)

-- AA' is the diameter of the circumcircle of ΔAXY
noncomputable def circ_A_X_Y : Circle ℝ := sorry
variable (AA'_is_diameter : diameter (circ_A_X_Y A X Y) = distance A A')

-- Perpendicular from B to BC intersects AX at P
variable (perpendicular_BC_AX_P : is_perpendicular (BC B C) (AXX A X) P)

-- Perpendicular from C to BC intersects AY at Q
variable (perpendicular_BC_AY_Q : is_perpendicular (BC B C) (AYY A Y) Q)

-- O is the circumcenter of ΔAPQ
variable (O_is_circumcenter : is_circumcenter O (AP A P) (PQ P Q))

-- To prove: tangent to the circumcircle of ΔAXY at A' passes through circumcenter of ΔAPQ
theorem tangent_circumcircle_A' (circ_A_X_Y : Circle ℝ) :
  tangent_to_circle_at (circ_A_X_Y A X Y) A' → passes_through O :=
begin
  sorry
end

end tangent_circumcircle_A_l124_124127


namespace train_pass_jogger_in_36_sec_l124_124498

noncomputable def time_to_pass_jogger (speed_jogger speed_train : ℝ) (lead_jogger len_train : ℝ) : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := lead_jogger + len_train
  total_distance / relative_speed

theorem train_pass_jogger_in_36_sec :
  time_to_pass_jogger 9 45 240 120 = 36 := by
  sorry

end train_pass_jogger_in_36_sec_l124_124498


namespace polynomial_qx_l124_124382

theorem polynomial_qx :
  ∀ (q : ℚ[X]),
    q + (2 * X^6 + 5 * X^4 + 10 * X^2) = (9 * X^4 + 30 * X^3 + 50 * X^2 + 4) →
    q = -2 * X^6 + 4 * X^4 + 30 * X^3 + 40 * X^2 + 4 :=
  by sorry

end polynomial_qx_l124_124382


namespace ellipse_equation_and_area_triangle_l124_124788

theorem ellipse_equation_and_area_triangle :
  (∀ (x y a b : ℝ), a > b > 0 → 
    (frac x^2 a^2 + frac y^2 b^2 = 1) → 
    (c : ℝ) → (e = c / a = sqrt 2 / 2) → (d = sqrt 2) →
    (distance(x, y, F1, F2) = sqrt 2) → 
    (eq : frac x^2 2 + y^2 = 1)) ∧

  (∀ (P M N : Point)(Q : ℝ × 0) (t : ℝ), P ∈ C1 → 
    (l passes through Q) → 
    (l intersects C2 at M and N) →
    (find the range of area of △PMN)) → 
    (area_range = [sqrt 6 - 2, +∞)) := 
sorry

end ellipse_equation_and_area_triangle_l124_124788


namespace carlos_books_in_june_l124_124971

theorem carlos_books_in_june
  (books_july : ℕ)
  (books_august : ℕ)
  (total_books_needed : ℕ)
  (books_june : ℕ) : 
  books_july = 28 →
  books_august = 30 →
  total_books_needed = 100 →
  books_june = total_books_needed - (books_july + books_august) →
  books_june = 42 :=
by intros books_july books_august total_books_needed books_june h1 h2 h3 h4
   sorry

end carlos_books_in_june_l124_124971


namespace range_of_f_l124_124417

noncomputable def f (x : ℝ) : ℝ := (1 + sqrt 3 * sin (2 * x) + cos (2 * x)) / (1 + sin x + sqrt 3 * cos x)

theorem range_of_f : 
  (∀ x : ℝ, 
    let y : ℝ := (1 + sqrt 3 * sin (2 * x) + cos (2 * x)) / (1 + sin x + sqrt 3 * cos x) in
    -1 ≤ sin (x + π / 3) ∧ sin (x + π / 3) ≤ 1 ∧ 2 * sin (x + π / 3) + 1 ≠ 0 ∧
    (y = (2 * sin (x + π / 3) - 1))
  ) →
  (∀ y : ℝ, y ∈ ([-3, -2) ∪ (-2, 1])) :=
sorry

end range_of_f_l124_124417


namespace abs_neg_five_l124_124449

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l124_124449


namespace percentage_change_volume_cuboid_l124_124448

theorem percentage_change_volume_cuboid
  (L W H : ℝ)
  (hL : L = 100)
  (hW : W = 100)
  (hH : H = 100)
  (L_new : ℝ := L * 1.3)
  (W_new : ℝ := W * 0.7)
  (H_new : ℝ := H * 1.2) :
  let V_original := L * W * H,
      V_new := L_new * W_new * H_new,
      percentage_change := ((V_new - V_original) / V_original) * 100 in
  percentage_change = 9.2 := by
  sorry

end percentage_change_volume_cuboid_l124_124448


namespace distance_points_PQ_l124_124639

theorem distance_points_PQ :
  let P := (-1 : ℝ, 2 : ℝ, -3 : ℝ)
  let Q := (3 : ℝ, -2 : ℝ, -1 : ℝ)
  let dist3d (P Q : ℝ × ℝ × ℝ) := 
    real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (Q.3 - P.3)^2)
  (dist3d P Q) = 6 := by
  sorry

end distance_points_PQ_l124_124639


namespace exists_balanced_set_l124_124447

def is_balanced (S : set (point)) : Prop :=
∀ (A B : point), A ≠ B → ∃ C : point, C ∈ S ∧ dist A C = dist B C

theorem exists_balanced_set (n : ℕ) (hn : n ≥ 3) :
  ∃ S : set (point), S.finite ∧ S.card = n ∧ is_balanced S :=
sorry

end exists_balanced_set_l124_124447


namespace find_m_equals_3_l124_124604

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (P A B C : V) (m : ℝ)

-- Defining the vectors
def vector_PA : V := A - P
def vector_PB : V := B - P
def vector_PC : V := C - P
def vector_AB : V := B - A
def vector_AC : V := C - A
def vector_AP : V := P - A

-- Given conditions
axiom condition1 : vector_PA + vector_PB + vector_PC = 0
axiom condition2 : vector_AB + vector_AC = m * vector_AP

-- To prove
theorem find_m_equals_3 : m = 3 :=
by sorry

end find_m_equals_3_l124_124604


namespace probability_six_heads_before_tail_l124_124347

theorem probability_six_heads_before_tail :
  let q := 1 / 64 in q = 1/64 :=
by
  let q := 1 / 64
  sorry

end probability_six_heads_before_tail_l124_124347


namespace number_divisible_by_12_not_20_l124_124520
-- Import the necessary library

-- Define the problem in Lean 4
theorem number_divisible_by_12_not_20 (n : ℕ) (h : n ≤ 2017) :
  (∑ i in finset.Icc 1 2017, if i % 12 = 0 ∧ i % 20 ≠ 0 then 1 else 0) = 135 := 
by
  sorry

end number_divisible_by_12_not_20_l124_124520


namespace not_weighted_voting_l124_124939

def acceptableWayToDecide (method : (Fin 6 → Bool) → Bool) : Prop :=
  (∀ (votes : Fin 6 → Bool) (i : Fin 6) (h : votes i = false), 
     method (votes ⟨i.1, by simp [Fin.is_lt]⟩ true) → method votes) ∧
  (∀ (votes : Fin 6 → Bool), 
     method votes = !method (λ j, !votes j))

def specificMethod (votes : Fin 6 → Bool) : Bool :=
  let agree := {i | votes i = true}
  let disagree := {i | votes i = false}
  if agree.card > 3 then true
  else if disagree.card > 3 then false
  else if agree.card = 3 then
    agree = {0, 1, 2} ∨ agree = {0, 2, 3} ∨ agree = {0, 3, 4} ∨ agree = {0, 4, 5} ∨ agree = {0, 1, 5} ∨ 
    agree = {2, 4, 5} ∨ agree = {2, 3, 5} ∨ agree = {1, 3, 4} ∨ agree = {1, 3, 5} ∨ agree = {1, 2, 4}
  else false

theorem not_weighted_voting (method : (Fin 6 → Bool) → Bool)
  (h1 : acceptableWayToDecide method)
  (h2 : method = specificMethod) :
  ¬ ∃ (weights : Fin 6 → ℕ), (∀ (votes : Fin 6 → Bool), method votes = (∑ i in Finset.univ.filter (λ j, votes j), weights i) > (∑ i in Finset.univ.filter (λ j, !votes j), weights i)) :=
sorry

end not_weighted_voting_l124_124939


namespace min_distance_is_pi_div_2_l124_124403

noncomputable def min_intersection_distance : ℝ :=
  let f := λ x : ℝ, tan (2 * x - π / 3)
  let g := λ a : ℝ, -a
  let intersection_points := {x | ∃ a : ℝ, f x = g a}
  (π / 2)

theorem min_distance_is_pi_div_2 :
  min_intersection_distance = π / 2 :=
by
  sorry

end min_distance_is_pi_div_2_l124_124403


namespace valentino_farm_total_birds_l124_124749

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end valentino_farm_total_birds_l124_124749


namespace certain_event_l124_124861

theorem certain_event (bag : Fin 5 → String) (h : ∀ i, bag i = "red") :
  (∃ i, bag i = "red") :=
by
  existsi 0
  rw [h 0]
  sorry

end certain_event_l124_124861


namespace sum_reciprocal_chords_constant_l124_124426

def parabola (y : ℝ) (x : ℝ) := y = x^2
def point (x y : ℝ) := (x, y)
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def reciprocal_distance_sum (A B C : ℝ × ℝ) := 
  let AC := distance A C
  let BC := distance B C
  1 / AC + 1 / BC

theorem sum_reciprocal_chords_constant 
  (A B C : ℝ × ℝ) (h_parabola : ∀ x, parabola (A.snd) (A.fst) ∧ parabola (B.snd) (B.fst))
  (h_C : C = point 0 (1/9)) (h_on_chord : ∃ λ : ℝ, A.snd = λ * A.fst + 1 / 9 ∧ B.snd = λ * B.fst + 1 / 9) :
  reciprocal_distance_sum A B C = 9 := 
sorry

end sum_reciprocal_chords_constant_l124_124426


namespace split_bill_equally_l124_124222

theorem split_bill_equally :
  let hamburger_cost := 3
  let hamburger_count := 5
  let fries_cost := 1.20
  let fries_count := 4
  let soda_cost := 0.50
  let soda_count := 5
  let spaghetti_cost := 2.70
  let friend_count := 5
  let total_cost := (hamburger_cost * hamburger_count) + (fries_cost * fries_count) + (soda_cost * soda_count) + spaghetti_cost
  in total_cost / friend_count = 5 := 
by
  sorry

end split_bill_equally_l124_124222


namespace reporters_cover_local_politics_l124_124488

-- Definitions of percentages and total reporters
def total_reporters : ℕ := 100
def politics_coverage_percent : ℕ := 20 -- Derived from 100 - 80
def politics_reporters : ℕ := (politics_coverage_percent * total_reporters) / 100
def not_local_politics_percent : ℕ := 40
def local_politics_percent : ℕ := 60 -- Derived from 100 - 40
def local_politics_reporters : ℕ := (local_politics_percent * politics_reporters) / 100

theorem reporters_cover_local_politics :
  (local_politics_reporters * 100) / total_reporters = 12 :=
by
  exact sorry

end reporters_cover_local_politics_l124_124488


namespace tan_A_of_right_triangle_l124_124321

theorem tan_A_of_right_triangle (A B C : Type) [Real.Angle] 
  (triangle : Triangle A B C) (hC : ∠C = π / 2) (hBC : BC = 2) (hAC : AC = 4) : 
  tan (∠A) = 1 / 2 := 
  sorry

end tan_A_of_right_triangle_l124_124321


namespace probability_all_successful_pairs_expected_successful_pairs_l124_124835

-- Lean statement for part (a)
theorem probability_all_successful_pairs (n : ℕ) :
  let prob_all_successful := (2 ^ n * n.factorial) / (2 * n).factorial
  in prob_all_successful = (1 / (2 ^ n)) * (n.factorial / (2 * n).factorial)
:= by sorry

-- Lean statement for part (b)
theorem expected_successful_pairs (n : ℕ) :
  let expected_value := n / (2 * n - 1)
  in expected_value > 0.5
:= by sorry

end probability_all_successful_pairs_expected_successful_pairs_l124_124835


namespace collinear_ABC_l124_124660

-- Definitions of circle intersections and concyclic properties
variables (O O₁ O₂ O₃ A B C : Point)
variables (circle₁ : Circle O₁) (circle₂ : Circle O₂) (circle₃ : Circle O₃)

-- Hypotheses as per the conditions
hypothesis (h1 : circle₁ ∩ circle₂ = {A, O})
hypothesis (h2 : circle₂ ∩ circle₃ = {B, O})
hypothesis (h3 : circle₁ ∩ circle₃ = {C, O})
hypothesis (h4 : Concyclic O O₁ O₂ O₃)

-- Proof goal
theorem collinear_ABC : Collinear A B C := by
  sorry

end collinear_ABC_l124_124660


namespace simple_interest_l124_124952

theorem simple_interest (P R T : ℝ) (hP : P = 8965) (hR : R = 9) (hT : T = 5) : 
    (P * R * T) / 100 = 806.85 := 
by 
  sorry

end simple_interest_l124_124952


namespace spherical_to_rectangular_correct_l124_124988

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 10 (3 * Real.pi / 4) (Real.pi / 4) = (-5, 5, 5 * Real.sqrt 2) := 
  sorry

end spherical_to_rectangular_correct_l124_124988


namespace concurrency_of_T1T2_AB_H1H2_l124_124001

variables {A B C H1 H2 H3 T1 T2 S1 S2 : Type}

/-- Given triangle ABC with altitudes AH1, BH2.
    Let S1 be the point where the tangent to the circumcircle of ABC at A meets BC.
    Let S2 be the point where the tangent to the circumcircle of ABC at B meets AC.
    Let T1 be the midpoint of AS1 and T2 be the midpoint of BS2.
    Then the lines T1T2, AB, and H1H2 concur. -/
theorem concurrency_of_T1T2_AB_H1H2
  (triangle_ABC : Type)
  (circumcircle_tangents : (tri : triangle_ABC) → (A B C : tri) → (S1 S2 : tri) → Prop)
  (midpoints : (tri : triangle_ABC) → (A B C : tri) → (T1 T2 : tri) → Prop)
  (altitudes : (tri : triangle_ABC) → (A B C : tri) → (H1 H2 : tri) → Prop)
  (h_altitudes : ∀ tri A B C, altitudes tri A B C H1 H2)
  (h_tangents : ∀ tri A B C, circumcircle_tangents tri A B C S1 S2)
  (h_midpoints : ∀ tri A B C, midpoints tri A B C T1 T2)
  : ∃ P, collinear [T1, T2, P] ∧ collinear [A, B, P] ∧ collinear [H1, H2, P] := by
sorry

end concurrency_of_T1T2_AB_H1H2_l124_124001


namespace boxcar_capacity_ratio_l124_124376

theorem boxcar_capacity_ratio :
  ∀ (total_capacity : ℕ)
    (num_red num_blue num_black : ℕ)
    (black_capacity blue_capacity : ℕ)
    (red_capacity : ℕ),
    num_red = 3 →
    num_blue = 4 →
    num_black = 7 →
    black_capacity = 4000 →
    blue_capacity = 2 * black_capacity →
    total_capacity = 132000 →
    total_capacity = num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity →
    (red_capacity / blue_capacity = 3) :=
by
  intros total_capacity num_red num_blue num_black black_capacity blue_capacity red_capacity
         h_num_red h_num_blue h_num_black h_black_capacity h_blue_capacity h_total_capacity h_combined_capacity
  sorry

end boxcar_capacity_ratio_l124_124376


namespace avg_of_first_200_terms_l124_124189

theorem avg_of_first_200_terms (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a (n + 1) = (-1)^(n + 2) * (n + 1)^2) :
  (∑ i in finRange 200, a (i + 1) : ℤ) / 200 = 200 := 
sorry

end avg_of_first_200_terms_l124_124189


namespace prove_side_c_prove_sin_B_prove_area_circumcircle_l124_124691

-- Define the given conditions
def triangle_ABC (a b A : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3

-- Prove that side 'c' is equal to 3
theorem prove_side_c (h : triangle_ABC a b A) : c = 3 := by
  sorry

-- Prove that sin B is equal to \frac{\sqrt{21}}{7}
theorem prove_sin_B (h : triangle_ABC a b A) : Real.sin B = Real.sqrt 21 / 7 := by
  sorry

-- Prove that the area of the circumcircle is \frac{7\pi}{3}
theorem prove_area_circumcircle (h : triangle_ABC a b A) (R : ℝ) : 
  let circumcircle_area := Real.pi * R^2
  circumcircle_area = 7 * Real.pi / 3 := by
  sorry

end prove_side_c_prove_sin_B_prove_area_circumcircle_l124_124691


namespace maximize_S_l124_124797

-- Define the sets and the problem statement
def a1 := 2
def a2 := 3
def a3 := 4
def b1 := 5
def b2 := 6
def b3 := 7
def c1 := 8
def c2 := 9
def c3 := 10

def maxS (a1 a2 a3 b1 b2 b3 c1 c2 c3: ℕ) : ℕ :=
  a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3

theorem maximize_S : 
  maxS 8 9 10 5 6 7 2 3 4 = 954 :=
by sorry

end maximize_S_l124_124797


namespace probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124844

-- Definition of conditions
def num_pairs_socks : ℕ := n

def successful_pair (socks : List (ℕ × ℕ)) (k : ℕ) : Prop :=
  (socks[n - k]).fst = (socks[n - k]).snd

-- Part (a): Prove the probability that all pairs are successful is \(\frac{2^n n!}{(2n)!}\)
theorem probability_all_successful_pairs (n : ℕ) :
  ∃ p : ℚ, p = (2^n * nat.factorial n) / nat.factorial (2 * n) :=
sorry

-- Part (b): Prove the expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half (n : ℕ) :
  (n / (2 * n - 1)) > 1 / 2 :=
sorry

end probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124844


namespace two_pow_gt_n_square_plus_one_l124_124373

theorem two_pow_gt_n_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := 
by {
  sorry
}

end two_pow_gt_n_square_plus_one_l124_124373


namespace trees_in_one_row_l124_124326

variable (total_trees_cleaned : ℕ)
variable (trees_per_row : ℕ)

theorem trees_in_one_row (h1 : total_trees_cleaned = 20) (h2 : trees_per_row = 5) :
  (total_trees_cleaned / trees_per_row) = 4 :=
by
  sorry

end trees_in_one_row_l124_124326


namespace golden_ellipse_eccentricity_l124_124616

theorem golden_ellipse_eccentricity (a c : ℝ) (h₁ : 0 < a) (h₂ : 0 < c) (h₃ : eccentricity := (sqrt 5 + 1) / 2) 
: ∃ e : ℝ, 1 = e^2 + e ∧ e = (sqrt 5 - 1) / 2 := 
  sorry

end golden_ellipse_eccentricity_l124_124616


namespace average_temperature_second_to_fifth_days_l124_124393

variable (T1 T2 T3 T4 T5 : ℝ)

theorem average_temperature_second_to_fifth_days 
  (h1 : (T1 + T2 + T3 + T4) / 4 = 58)
  (h2 : T1 / T5 = 7 / 8)
  (h3 : T5 = 32) :
  (T2 + T3 + T4 + T5) / 4 = 59 :=
by
  sorry

end average_temperature_second_to_fifth_days_l124_124393


namespace perimeter_triangle_ab1m_distance_a1_to_plane_ab1m_l124_124239

variable (a : ℝ)

structure Cube :=
(edge_length : ℝ)
(A A1 B B1 C C1 D D1 : Prod3 ℝ)
(M : Prod3 ℝ)
(ratio : ℝ)
(A1M_ratio_MD1 : A1M_ratio_MD1 = 1 / (1 + 2))

noncomputable def perimeter_of_triangle 
  (A B1 M : Prod3 ℝ): ℝ := 
  dist_point_to_point A M + dist_point_to_point B1 M + dist_point_to_point A B1 

noncomputable def distance_from_point_to_plane 
  (P : Prod3 ℝ) (plane : Plane): ℝ := 
  sorry -- Formula for distance from a point to a plane

theorem perimeter_triangle_ab1m (cube : Cube a) :
  perimeter_of_triangle cube.A cube.B1 cube.M = 
    a * (Real.sqrt 5 + Real.sqrt 14 + 3 * Real.sqrt 3) / 3 := 
sorry

theorem distance_a1_to_plane_ab1m (cube : Cube a) : 
  distance_from_point_to_plane cube.A1 (Plane.mk cube.A cube.B1 cube.M) = 
    3 * a * Real.sqrt 22 / 22 := 
sorry

end perimeter_triangle_ab1m_distance_a1_to_plane_ab1m_l124_124239


namespace solve_problem_l124_124719

noncomputable theory

-- Condition 1: a_1 = 1
def a1 : ℕ → ℕ
| 1 := 1
| _ := 0 -- this will be overridden in what we prove, placeholder for Lean completeness

-- Condition 2: S_(n+1) = S_n + 1
def S (n : ℕ) : ℕ :=
∑ i in Finset.range n, a1 i.succ

-- Condition 3 & 4: Recurrence relation for a_n and sum T_n
def a (n : ℕ) : ℕ := 2 ^ (n - 1)
def b (n : ℕ) : ℕ := (4 * a n) / ((a (n + 1) - 1) * (a (n + 2) - 1))
def T (n : ℕ) : ℕ := ∑ i in Finset.range n, b i.succ

-- Condition 5: m^2 - 4/3 * m < T_n
def m_check (m : ℝ) (n : ℕ) : Prop := m ^ 2 - (4 / 3) * m < T n

-- Define the problem to solve
theorem solve_problem : 
(a 1 = 1) ∧
(∀ n : ℕ, T n = 2 * (1 - 1 / (2 ^ n - 1))) ∧
(∀ m : ℝ, m_check m 1 -> -2 / 3 < m ∧ m < 2) :=
by {
  sorry
}

end solve_problem_l124_124719


namespace eval_expression_l124_124177

theorem eval_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 :=
by sorry

end eval_expression_l124_124177


namespace potency_range_l124_124512

theorem potency_range (x : ℝ) (hx1 : 0 <= x) (hx2 : x <= 1) :
  (25 < (40 * 0.15 + 50 * x) / (40 + 50) * 100) ∧ ((40 * 0.15 + 50 * x) / (40 + 50) * 100 < 30) ↔ 
  (33 < x * 100) ∧ (x * 100 < 42) :=
begin
  sorry
end

end potency_range_l124_124512


namespace sacks_of_oranges_harvested_per_day_l124_124087

theorem sacks_of_oranges_harvested_per_day (discards_per_day total_days remaining_sacks : ℕ) 
    (h_discards : discards_per_day = 71) 
    (h_days : total_days = 51) 
    (h_remaining : remaining_sacks = 153) : 
    let x := (remaining_sacks + (discards_per_day * total_days)) / total_days in
    x = 74 := 
by
  sorry

end sacks_of_oranges_harvested_per_day_l124_124087


namespace geometric_sequence_term_l124_124789

theorem geometric_sequence_term :
  ∃ {r : ℝ} (a8 a11 : ℝ), a8 = 8 ∧ a11 = 64 ∧ a11 = a8 * r^3 ∧ (a8 * r^7) * r^4 = 1024 :=
by
  sorry

end geometric_sequence_term_l124_124789


namespace cot_tan_rewrite_l124_124566

theorem cot_tan_rewrite (x : ℝ) (h : (sin (x / 4) * cos x) ≠ 0) :
  (cot (x / 4) + tan x) = (sin (5 * x / 4) / (sin (x / 4) * cos x)) :=
sorry

end cot_tan_rewrite_l124_124566


namespace angle_in_third_quadrant_l124_124654

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  180 - α > -90 - k * 360 ∧ 180 - α < -k * 360 := 
by sorry

end angle_in_third_quadrant_l124_124654


namespace find_square_l124_124299

theorem find_square (y : ℝ) (h : (y + 5)^(1/3) = 3) : (y + 5)^2 = 729 := 
sorry

end find_square_l124_124299


namespace part_a_part_b_part_c_l124_124876

-- Define points and triangles
variable {Point : Type} [Add Point] [HasSmul ℝ Point]

structure Triangle (P : Type) := (A B C : P)

-- Definitions of triangles and point O
variable (ABC DEF : Triangle Point) (O : Point)

-- Points X and Y are inside triangles ABC and DEF respectively
variable (X ∈ {t : Triangle Point | t = ABC})
variable (Y ∈ {t : Triangle Point | t = DEF})

-- Z is obtained as completion of parallelogram OXYZ
def Z (X Y : Point) : Point := X + Y

-- Prove the statements
theorem part_a (X ∈ ABC) (Y ∈ DEF) : ∃ φ : Set Point, ∀ z : Point, z ∈ φ ↔ ∃ (X' X'' : Point), X' ∈ ABC ∧ X'' ∈ DEF ∧ Z X' X'' = z :=
sorry

theorem part_b (X ∈ ABC) (Y ∈ DEF) : ∃ φ : Set Point, ∀ z : Point, z ∈ φ ↔ ∃ (X' X'' : Point), X' ∈ ABC ∧ X'' ∈ DEF ∧ Z X' X'' = z ∧ set.finite {z ∈ φ}, 6 :=
sorry

theorem part_c (X ∈ ABC) (Y ∈ DEF) : ∃ φ : Set Point, ∀ z : Point, z ∈ φ ↔ ∃ (X' X'' : Point), X' ∈ ABC ∧ X'' ∈ DEF ∧ Z X' X'' = z ∧ perimeter φ = perimeter ABC + perimeter DEF :=
sorry

end part_a_part_b_part_c_l124_124876


namespace find_value_of_f_l124_124723

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ π then sin x
else if -π < x ∧ x < 0 then cos x
else 0 -- Placeholder for the periodic definition

theorem find_value_of_f :
  f (-13 * π / 4) = sqrt 2 / 2 :=
by 
  sorry

end find_value_of_f_l124_124723


namespace tetrahedrons_in_triangular_prism_l124_124293

theorem tetrahedrons_in_triangular_prism : 
  let V := 6  -- Number of vertices
  let choose := Nat.choose  -- Binomial Coefficient
  let same_face_count := 3  -- Combinations where all vertices are on the same face
  choose V 4 - same_face_count = 12 :=
by
  simp [Nat.choose]
  sorry

end tetrahedrons_in_triangular_prism_l124_124293


namespace vector_triangle_inequality_l124_124226

variable {E : Type*} [inner_product_space ℝ E] (a b c : E) (k : ℝ)

theorem vector_triangle_inequality (a b : E) : 
  ‖a + b‖ ≤ ‖a‖ + ‖b‖ :=
begin
  sorry
end

end vector_triangle_inequality_l124_124226


namespace option_B_option_C_option_D_l124_124003

variable (Ω : Type) [ProbabilitySpace Ω]
variable (A B : Event Ω)
variable h₁ : P(A) = 2/5
variable h₂ : P(¬B | A) = 3/4
variable h₃ : P(B | ¬A) = 1/3

theorem option_B : P(¬A ∩ B) = 1/5 := sorry
theorem option_C : P(B) = 3/10 := sorry
theorem option_D : P(A | B) = 1/3 := sorry

end option_B_option_C_option_D_l124_124003


namespace tetrahedron_projection_max_area_l124_124433

noncomputable def maxProjectionArea (s : ℝ) (θ : ℝ) : ℝ :=
  let S := (sqrt 3 / 4) * s^2
  S

theorem tetrahedron_projection_max_area :
  ∀ (s : ℝ) (θ : ℝ), 
  s = 3 ∧ θ = π / 6 → 
  maxProjectionArea s θ = (9 * sqrt 3) / 4 :=
by 
  intros s θ h
  cases h
  simp only [maxProjectionArea, h_left, h_right]
  have : (sqrt 3) / 4 * (3:ℝ)^2 = (9 * sqrt 3) / 4 := by
    norm_num
    ring
  exact this
  sorry

end tetrahedron_projection_max_area_l124_124433


namespace maximal_suitable_pairs_l124_124811

namespace MaximalSuitablePairs

-- Given n girls and n boys
variables (n : ℕ) (girls boys : Fin n → Type) -- Type for girls and boys indexed by Fin n

-- Definition of a pair being suitable
def suitable_pairs (G : Fin n → Type) (B : Fin n → Type) (suitable : G → B → Prop) : Prop :=
  ∀ i j, suitable (girls i) (boys j) → j ≤ i

-- Statement that there is exactly one way to pair each girl with a distinct boy (unique matching)
def unique_matching (G : Fin n → Type) (B : Fin n → Type) (suitable : G → B → Prop) : Prop :=
  ∃! f : Fin n → Fin n, ∀ i, suitable (girls i) (boys (f i))

-- The main theorem to prove the maximal number of suitable pairs
theorem maximal_suitable_pairs (G : Fin n → Type) (B : Fin n → Type) (suitable : G → B → Prop)
  (uniqueMatch : unique_matching G B suitable) : 
  ∃ m, (m = n * (n + 1) / 2) ∧ ∀ i j, suitable (girls i) (boys j) → j ≤ i ∧ ∑ m in finset.range n, (m + 1) = n * (n + 1) / 2 :=
sorry

end MaximalSuitablePairs

end maximal_suitable_pairs_l124_124811


namespace max_inscribed_circle_radius_equilateral_l124_124167

variables {a b c : ℝ}
def p : ℝ := (a + b + c) / 2
def S : ℝ := sqrt (p * (p - a) * (p - b) * (p - c))
def r : ℝ := S / p

theorem max_inscribed_circle_radius_equilateral :
  (∀ a b c, r ≤ p / sqrt 27) ∧ (r = p / sqrt 27 ↔ a = b ∧ b = c) := 
by
  sorry

end max_inscribed_circle_radius_equilateral_l124_124167


namespace chess_tournament_players_l124_124467

noncomputable def total_players_in_tournament (n : ℕ) : Prop :=
  (∑ i in finset.range n, i) / 2 = 90 + (n-10)*(n-11)

/--
In a chess tournament where each player plays exactly one game with every other player,
the winner receives 1 point, the loser receives 0 points, and a draw gives each player 0.5 points,
the total number of players participating in the tournament is 25 if 
  1) Each player's total score is exactly half of the points they earned from games against the 10 lowest-scoring players.
  2) Half of the points for the 10 lowest-scoring players came from games among themselves.
-/
theorem chess_tournament_players (n : ℕ) (h : total_players_in_tournament n) : n = 25 := 
sorry

end chess_tournament_players_l124_124467


namespace QR_eq_b_l124_124076

theorem QR_eq_b (a b c : ℝ) 
  (hP : b = c * Real.cosh (a / c))
  (hQ : c = c * Real.cosh (0)) :
  ∃ R : ℝ × ℝ, (R.2 = 0) ∧ (let QR := Real.sqrt ((R.1 - 0)^2 + (R.2 - c)^2) in QR = b) := by
  sorry

end QR_eq_b_l124_124076


namespace sqrt_95_floor_l124_124549

theorem sqrt_95_floor : (Real.sqrt 95).floor = 9 := by
  have h1 : 81 < 95 := by linarith
  have h2 : 95 < 100 := by linarith
  have h3 : 9^2 = 81 := by norm_num
  have h4 : 10^2 = 100 := by norm_num
  have h5 : (Real.sqrt 81) < (Real.sqrt 95) := Real.sqrt_lt Real.zero_lt_one h1
  have h6 : (Real.sqrt 95) < (Real.sqrt 100) := Real.sqrt_lt Real.zero_lt_one h2
  have h7 : (Real.sqrt 81) = 9 := Real.sqrt_eq rfl.ge
  have h8 : (Real.sqrt 100) = 10 := Real.sqrt_eq rfl.ge
  exact sorry

end sqrt_95_floor_l124_124549


namespace greatest_of_consecutive_integers_sum_18_l124_124454

theorem greatest_of_consecutive_integers_sum_18 
  (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := 
sorry

end greatest_of_consecutive_integers_sum_18_l124_124454


namespace train_length_l124_124509

def speed_kmph := 72   -- Speed in kilometers per hour
def time_sec := 14     -- Time in seconds

/-- Function to convert speed from km/hr to m/s -/
def convert_speed (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

/-- Function to calculate distance given speed and time -/
def calculate_distance (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

theorem train_length :
  calculate_distance (convert_speed speed_kmph) time_sec = 280 :=
by
  sorry

end train_length_l124_124509


namespace find_b_find_area_l124_124690

def triangle_sides {a b c : ℝ} : Prop :=
  a^2 - c^2 = 2 * b ∧ sin A * cos C = 3 * cos A * sin C

theorem find_b (a b c : ℝ) (h : triangle_sides) : b = 4 :=
sorry

theorem find_area (a b c : ℝ) (h : triangle_sides) (ha : a = 6) : 
  ∃ area : ℝ, area = 6 * sqrt 3 :=
sorry

end find_b_find_area_l124_124690


namespace find_1234th_digit_to_right_l124_124725

def concatenated_digit_1234th : ℕ :=
  let segmentA_digit_count := 9 in  -- 9 one-digit numbers
  let segmentB_digit_count := 90 * 2 in  -- 90 two-digit numbers
  let segmentC_digit_count := (499 - 100 + 1) * 3 in  -- Three-digit numbers from 100 to 499

  let remaining_digits_after_AB := 1234 - segmentA_digit_count - segmentB_digit_count in
  let (full_numbers_in_segmentC, remainder) := Nat.divMod remaining_digits_after_AB 3 in

  let target_number := 100 + full_numbers_in_segmentC in
  if remainder = 1 then target_number / 100
  else if remainder = 2 then (target_number / 10) % 10
  else target_number % 10

theorem find_1234th_digit_to_right : concatenated_digit_1234th = 4 :=
by
  have h1 : concatenated_digit_1234th = 
    let segmentA_digit_count := 9 in
    let segmentB_digit_count := 90 * 2 in
    let segmentC_digit_count := (499 - 100 + 1) * 3 in

    let remaining_digits_after_AB := 1234 - segmentA_digit_count - segmentB_digit_count in
    let (full_numbers_in_segmentC, remainder) := Nat.divMod remaining_digits_after_AB 3 in

    let target_number := 100 + full_numbers_in_segmentC in
    if remainder = 1 then target_number / 100
    else if remainder = 2 then (target_number / 10) % 10
    else target_number % 10
  calc
    concatenated_digit_1234th = 4
    : sorry

end find_1234th_digit_to_right_l124_124725


namespace range_of_a_l124_124635

theorem range_of_a (a : ℝ) :
  ¬(∃ x₀ : ℝ, a * x₀ ^ 2 - a * x₀ - 2 > 0) ↔ a ∈ Icc (-8) 0 := 
sorry

end range_of_a_l124_124635


namespace gcd_triangular_number_l124_124565

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem gcd_triangular_number (n : ℕ) (h : n > 2) :
  ∃ k, n = 12 * k + 2 → gcd (6 * triangular_number n) (n - 2) = 12 :=
  sorry

end gcd_triangular_number_l124_124565


namespace opposite_number_l124_124912

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l124_124912


namespace stickers_total_l124_124771

theorem stickers_total (ryan_has : Ryan_stickers = 30)
  (steven_has : Steven_stickers = 3 * Ryan_stickers)
  (terry_has : Terry_stickers = Steven_stickers + 20) :
  Ryan_stickers + Steven_stickers + Terry_stickers = 230 :=
sorry

end stickers_total_l124_124771


namespace opposite_number_l124_124913

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l124_124913


namespace eval_expression_l124_124176

theorem eval_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 :=
by sorry

end eval_expression_l124_124176


namespace diametrically_opposite_number_is_84_l124_124894

-- Define the circular arrangement problem
def equal_arcs (n : ℕ) (k : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ i, i < n → ∃ j, j < n ∧ j ≠ i ∧ k j = k i + n / 2 ∨ k j = k i - n / 2

-- Define the specific number condition
def exactly_one_to_100 (k : ℕ) : Prop := k = 83

theorem diametrically_opposite_number_is_84 (n : ℕ) (k : ℕ → ℕ) (m : ℕ) :
  n = 100 →
  equal_arcs n k m →
  exactly_one_to_100 (k 83) →
  k 83 + 1 = 84 :=
by {
  intros,
  sorry
}

end diametrically_opposite_number_is_84_l124_124894


namespace anne_bob_total_difference_l124_124079

-- Define specific values as constants
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.08

-- Define the calculations according to Anne's method
def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)

-- Define the calculations according to Bob's method
def bob_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

-- State the theorem that the difference between Anne's and Bob's totals is zero
theorem anne_bob_total_difference : anne_total - bob_total = 0 :=
by sorry  -- Proof not required

end anne_bob_total_difference_l124_124079


namespace scalene_triangle_spheres_rel_l124_124427

theorem scalene_triangle_spheres_rel (a b c R r ρ : ℝ) (h1 : scalene_triangle a b c) 
  (h2 : spheres_touch_at_vertices a b c) 
  (h3 : spheres_touch_each_other a b c)
  (h4 : ρ > r) :
  ∃ s1 s2, touches_two_spheres_and_plane s1 a b c ∧ touches_two_spheres_and_plane s2 a b c
  ∧ (1 / r - 1 / ρ = 2 * Real.sqrt 3 / R) := 
sorry

end scalene_triangle_spheres_rel_l124_124427


namespace baseball_card_devaluation_l124_124134

variable (x : ℝ) -- Note: x will represent the yearly percent decrease in decimal form (e.g., x = 0.10 for 10%)

theorem baseball_card_devaluation :
  (1 - x) * (1 - x) = 0.81 → x = 0.10 :=
by
  sorry

end baseball_card_devaluation_l124_124134


namespace sum_elements_l124_124343

noncomputable def S_n (n : ℕ) : ℝ :=
  let M_n := {f : Fin n → ℕ // ∀ i : Fin (n - 1), f i = 0 ∨ f i = 1 ∧ f ⟨n-1, sorry⟩ = 1}
  let T_n : ℕ := 2^(n-1)
  let sum_elements_in_M_n : ℝ :=
    T_n / (10^n) * (2^(n-1) - 1) + 2^(n-2) * (10^-n)
  2^(n-2) * (sum_elements_in_M_n)

-- Proof is skipped using sorry
theorem sum_elements (n : ℕ) : S_n n = 2^(n-2) * (0.111111111111111111111111111111 + 10^-n) := sorry

end sum_elements_l124_124343


namespace different_selection_methods_l124_124139

theorem different_selection_methods (students: Finset ℕ) (h_card: students.card = 5) : 
  ∃ methods, methods = 60 ∧ methods = (students.choose 3).card * (3!).do {
  sorry
}

end different_selection_methods_l124_124139


namespace pairs_bought_after_donation_l124_124365

-- Definitions from conditions
def initial_pairs : ℕ := 80
def donation_percentage : ℕ := 30
def post_donation_pairs : ℕ := 62

-- The theorem to be proven
theorem pairs_bought_after_donation : (initial_pairs - (donation_percentage * initial_pairs / 100) + 6 = post_donation_pairs) :=
by
  sorry

end pairs_bought_after_donation_l124_124365


namespace problem_statement_l124_124204

noncomputable def monochromatic_triangle_in_K6 : Prop :=
  let vertices := {0, 1, 2, 3, 4, 5} in
  ∃ (edges : vertices × vertices → Bool), 
    (∀ e, e ∈ vertices × vertices → Bool = red ∨ Bool = blue) ∧
    (∀ (A B C : vertices), A ≠ B ∧ B ≠ C ∧ C ≠ A →
        ((edges (A, B) = red ∧ edges (B, C) = red ∧ edges (C, A) = red) ∨
         (edges (A, B) = blue ∧ edges (B, C) = blue ∧ edges (C, A) = blue)))

theorem problem_statement :
  (probability monochromatic_triangle_in_K6 vertices × vertices → Bool = 1/2) = 255 / 256 := sorry

end problem_statement_l124_124204


namespace johns_chore_homework_time_l124_124550

-- Definitions based on problem conditions
def cartoons_time : ℕ := 150  -- John's cartoon watching time in minutes
def chores_homework_per_10 : ℕ := 13  -- 13 minutes combined chores and homework per 10 minutes of cartoons
def cartoon_period : ℕ := 10  -- Per 10 minutes period

-- Theorem statement
theorem johns_chore_homework_time :
  cartoons_time / cartoon_period * chores_homework_per_10 = 195 :=
by sorry

end johns_chore_homework_time_l124_124550


namespace find_x_l124_124582

theorem find_x :
  ∃ n : ℕ, odd n ∧ (6^n + 1).natPrimeFactors.length = 3 ∧ 11 ∈ (6^n + 1).natPrimeFactors ∧ 6^n + 1 = 7777 :=
by sorry

end find_x_l124_124582


namespace Ursula_hours_per_day_l124_124094

theorem Ursula_hours_per_day (hourly_wage : ℝ) (days_per_month : ℕ) (annual_salary : ℝ) (months_per_year : ℕ) :
  hourly_wage = 8.5 →
  days_per_month = 20 →
  annual_salary = 16320 →
  months_per_year = 12 →
  (annual_salary / months_per_year / days_per_month / hourly_wage) = 8 :=
by
  intros
  sorry

end Ursula_hours_per_day_l124_124094


namespace proof_math_problem_l124_124104

-- Definitions based on the conditions
def pow_zero (x : ℝ) : x^0 = 1 := by
  rw [pow_zero]
  exact one

def three_pow_neg_three := (3 : ℝ)^(-3)
def three_pow_zero := (3 : ℝ)^0

-- The final statement to be proven
theorem proof_math_problem :
  (three_pow_neg_three)^0 + (three_pow_zero)^4 = 2 :=
by
  simp [three_pow_neg_three, three_pow_zero, pow_zero]
  sorry

end proof_math_problem_l124_124104


namespace probability_all_successful_pairs_expected_successful_pairs_l124_124836

-- Lean statement for part (a)
theorem probability_all_successful_pairs (n : ℕ) :
  let prob_all_successful := (2 ^ n * n.factorial) / (2 * n).factorial
  in prob_all_successful = (1 / (2 ^ n)) * (n.factorial / (2 * n).factorial)
:= by sorry

-- Lean statement for part (b)
theorem expected_successful_pairs (n : ℕ) :
  let expected_value := n / (2 * n - 1)
  in expected_value > 0.5
:= by sorry

end probability_all_successful_pairs_expected_successful_pairs_l124_124836


namespace total_legs_camden_dogs_l124_124532

variable (c r j : ℕ) -- c: Camden's dogs, r: Rico's dogs, j: Justin's dogs

theorem total_legs_camden_dogs :
  (r = j + 10) ∧ (j = 14) ∧ (c = (3 * r) / 4) → 4 * c = 72 :=
by
  sorry

end total_legs_camden_dogs_l124_124532


namespace square_perimeter_l124_124507

theorem square_perimeter (s : ℕ) (h : s = 13) : 4 * s = 52 :=
by
  rw [h]
  norm_num

end square_perimeter_l124_124507


namespace value_of_a_l124_124066

theorem value_of_a {a : ℝ} 
  (h : ∀ x y : ℝ, ax - 2*y + 2 = 0 ↔ x + (a-3)*y + 1 = 0) : 
  a = 1 := 
by 
  sorry

end value_of_a_l124_124066


namespace valid_x_count_correct_l124_124572

open Real

noncomputable def count_valid_x : ℕ :=
  let valid_x_values := { x : ℝ | 0 ≤ x ∧ ∃ k : ℤ, 0 ≤ k ∧ k ≤ 13 ∧ sqrt (169 - (x ^ (1 / 4))) = k }
  valid_x_values.to_finset.card

theorem valid_x_count_correct : count_valid_x = 14 := sorry

end valid_x_count_correct_l124_124572


namespace perimeter_of_JKLM_is_32_l124_124872

structure Point where
  x : Int
  y : Int

def J : Point := ⟨-2, -3⟩
def K : Point := ⟨-2, 1⟩
def L : Point := ⟨6, 7⟩
def M : Point := ⟨6, -3⟩

def distance (p1 p2 : Point) : Real :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def length_JK : Real := distance J K
def length_KL : Real := distance K L
def length_LM : Real := distance L M
def length_MJ : Real := distance M J

def perimeter_JKLM : Real := length_JK + length_KL + length_LM + length_MJ

theorem perimeter_of_JKLM_is_32 : perimeter_JKLM = 32 := by
  sorry

end perimeter_of_JKLM_is_32_l124_124872


namespace bob_weight_is_120_l124_124329

-- Definitions
def jim_weight := j
def bob_weight := b
def total_weight := 200
def weight_difference := b - j
def one_third_b := b / 3

-- Lean 4 statement
theorem bob_weight_is_120 (j b : ℝ) (h1 : j + b = total_weight) (h2 : weight_difference = one_third_b) : b = 120 := 
by sorry

end bob_weight_is_120_l124_124329


namespace opposite_of_83_is_84_l124_124931

-- Define the problem context
def circle_with_points (n : ℕ) := { p : ℕ // p < n }

-- Define the even distribution condition
def even_distribution (n k : ℕ) (points : circle_with_points n → ℕ) :=
  ∀ k < n, let diameter := set_of (λ p, points p < k) in
           cardinal.mk diameter % 2 = 0

-- Define the diametrically opposite point function
def diametrically_opposite (n k : ℕ) (points : circle_with_points n → ℕ) : ℕ :=
  (points ⟨k + (n / 2), sorry⟩) -- We consider n is always even, else modulus might be required

-- Problem statement to prove
theorem opposite_of_83_is_84 :
  ∀ (points : circle_with_points 100 → ℕ)
    (hne : even_distribution 100 83 points),
    diametrically_opposite 100 83 points = 84 := 
sorry

end opposite_of_83_is_84_l124_124931


namespace faster_speed_l124_124153

theorem faster_speed (Speed1 : ℝ) (ExtraDistance : ℝ) (ActualDistance : ℝ) (v : ℝ) : 
  Speed1 = 10 ∧ ExtraDistance = 31 ∧ ActualDistance = 20.67 ∧ 
  (ActualDistance / Speed1 = (ActualDistance + ExtraDistance) / v) → 
  v = 25 :=
by
  sorry

end faster_speed_l124_124153


namespace probability_of_one_unit_apart_l124_124387

theorem probability_of_one_unit_apart : 
  let points := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  let total_pairs := (points.card.choose 2 : ℕ) in
  let favorable_pairs := 14 in
  (favorable_pairs : ℚ) / (total_pairs : ℚ) = 14 / 45 :=
by
  sorry

end probability_of_one_unit_apart_l124_124387


namespace harmonic_density_zero_l124_124982

open Nat

/-
  Let S be a subset of ℤ where the sum of the chosen terms from the harmonic series is finite.
  d(n) is defined as the density function: d(n) = (1 / n) * #(S ∩ [1, n])
  
  We need to prove that the limit of d(n) as n approaches infinity is 0.
-/

def density (S : Set ℕ) (n : ℕ) : ℝ := (1 / n) * (Set.Finite.count (S ∩ (Set.Icc 1 n)))

theorem harmonic_density_zero (S : Set ℕ) (h : ∑' k in S, (1 / k : ℝ) < ∞) :
  tendsto (λ n, density S n) at_top (𝓝 0) := 
sorry

end harmonic_density_zero_l124_124982


namespace paths_from_A_to_C_l124_124563

theorem paths_from_A_to_C :
  let paths_A_B: ℕ := 2,
      paths_B_D: ℕ := 2,
      paths_D_C: ℕ := 2,
      direct_paths_A_D: ℕ := 1,
      direct_paths_A_C: ℕ := 1,
      paths_through_B_D := paths_A_B * paths_B_D * paths_D_C,
      paths_through_D := direct_paths_A_D * paths_D_C,
      total_paths := paths_through_B_D + paths_through_D + direct_paths_A_C
  in total_paths = 11 := by
  sorry

end paths_from_A_to_C_l124_124563


namespace valid_numbers_formula_l124_124568

def is_prime (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def no_two_consecutive_digits_equal (n : ℕ) (num : ℕ) : Prop :=
  ∀ i, i < n - 1 → (num / 10^i) % 10 ≠ (num / 10^(i+1)) % 10

def valid_n_digit_number (n : ℕ) (num : ℕ) : Prop :=
  (num / 10^(n-1) > 0) ∧ 
  no_two_consecutive_digits_equal n num ∧ 
  is_prime (num % 10)

def count_valid_numbers (n : ℕ) : ℕ :=
  Nat.card {num : ℕ | valid_n_digit_number n num}

theorem valid_numbers_formula (n : ℕ) : 
  count_valid_numbers n = (2 * (9^n + (-1)^(n+1))) / 5 := 
sorry

end valid_numbers_formula_l124_124568


namespace vershoks_per_arshin_l124_124290

theorem vershoks_per_arshin (plank_length_arshins : ℝ) (plank_width_vershoks : ℝ) 
    (room_side_length_arshins : ℝ) (total_planks : ℕ) (n : ℝ)
    (h1 : plank_length_arshins = 6) (h2 : plank_width_vershoks = 6)
    (h3 : room_side_length_arshins = 12) (h4 : total_planks = 64) 
    (h5 : (total_planks : ℝ) * (plank_length_arshins * (plank_width_vershoks / n)) = room_side_length_arshins^2) :
    n = 16 :=
by {
  sorry
}

end vershoks_per_arshin_l124_124290


namespace tan_ratio_l124_124736

theorem tan_ratio (p q : ℝ) 
  (h1: Real.sin (p + q) = 5 / 8)
  (h2: Real.sin (p - q) = 3 / 8) : Real.tan p / Real.tan q = 4 := 
by
  sorry

end tan_ratio_l124_124736


namespace S_infinite_max_value_f_l124_124013

def S : set ℕ := 
  {p : ℕ | nat.prime p ∧ ∃ r : ℕ, (∀ k, (k ≠ 0 → 10^k % p = 1 ↔ k = r * 3))}

definition r (p : ℕ) (h : p ∈ S) : ℕ :=
  classical.some h.2

definition a (p : ℕ) (h : p ∈ S) (k : ℕ) : ℕ :=
  classical.some (nat.digits_def (p, h, k)).1 -- Function to represent the k-th digit in the decimal expansion of 1/p

def f (k p : ℕ) (h : p ∈ S) : ℕ :=
  a p h k + a p h (k + r p h) + a p h (k + 2 * r p h)

theorem S_infinite : ∀ (p ∈ S), infinite S := 
sorry

theorem max_value_f : ∀ (k ≥ 1) (p ∈ S), f k p h ≤ 19 := 
sorry

end S_infinite_max_value_f_l124_124013


namespace min_norm_l124_124641

theorem min_norm (t : ℝ) : ∃ t₀, (|((2 : ℝ) • (1, t) + (t, (-6 : ℝ)))| = 2 * Real.sqrt 5) :=
  sorry

end min_norm_l124_124641


namespace line_A2B2_passes_through_C1_l124_124322

open EuclideanGeometry

theorem line_A2B2_passes_through_C1
  (A B C : Point)
  (a b c : ℝ)
  (M_a : Point)
  (M_b : Point)
  (A_2 B_2 C_1 : Point)
  (hBC_midpoint : midpoint A B = M_a)
  (hAC_midpoint : midpoint B C = M_b)
  (hA2_distance : dist M_a A_2 = a / 2)
  (hB2_distance : dist M_b B_2 = b / 2)
  (hC1_orth : is_foot_of_altitude C A B C_1)
  (hA2_perpendicular : perpendicular M_a A_2 (line_through A B))
  (hB2_perpendicular : perpendicular M_b B_2 (line_through A C)) :
  collinear {A_2, B_2, C_1} :=
by
  sorry

end line_A2B2_passes_through_C1_l124_124322


namespace volume_of_spheres_l124_124436

noncomputable def sphere_volume (a : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3

theorem volume_of_spheres (a : ℝ) : 
  ∃ r : ℝ, r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
  sphere_volume a = (4 / 3) * Real.pi * r^3 := 
sorry

end volume_of_spheres_l124_124436


namespace desalination_reservoir_shortage_l124_124121

theorem desalination_reservoir_shortage
    (total_capacity : ℝ)
    (current_level : ℝ)
    (current_is_twice_normal : current_level = 2 * (current_level / 2))
    (current_is_60_percent : current_level = 0.60 * total_capacity) :
    (total_capacity - (current_level / 2)) = 7 :=
by
  have current_level_eq_6 : current_level = 6 := sorry -- from the problem statement
  have total_capacity_eq_10 : total_capacity = 6 / 0.60 := sorry -- calculated from current_is_60_percent
  
  rw [current_level_eq_6, total_capacity_eq_10]
  norm_num
  sorry

end desalination_reservoir_shortage_l124_124121


namespace sec_tan_sum_l124_124559

theorem sec_tan_sum :
  let θ1 : ℝ := 150 * Real.pi / 180    -- Convert 150 degrees to radians
  let θ2 : ℝ := 225 * Real.pi / 180    -- Convert 225 degrees to radians
  sec θ1 + tan θ2 = (3 - 2 * Real.sqrt 3) / 3 :=
by {
  -- Definition of secant
  have h1: sec θ1 = 1 / cos θ1 := sorry,
  -- Value of cos 150 degrees (= -√3 / 2)
  have h2: cos θ1 = -Real.sqrt 3 / 2 := sorry,
  -- Calculation of sec 150 degrees from h1 and h2
  have h3: sec θ1 = -2 / Real.sqrt 3 := sorry,
  -- Rationalization of -2 / √3
  have h4: sec θ1 = -2 * Real.sqrt 3 / 3 := sorry,
  -- Definition of tangent
  have h5: tan θ2 = tan (Real.pi + Real.pi / 4) := sorry,
  -- Tangent of addition of angles equal to tangent of acute angle
  have h6: tan (Real.pi + Real.pi / 4) = tan (Real.pi / 4) := sorry,
  -- Tangent of 45 degrees is 1
  have h7: tan (Real.pi / 4) = 1 := sorry,
  -- Tan 225 degrees
  have h8: tan θ2 = 1 := sorry,
  -- Sum sec 150 + tan 225
  have h9: sec θ1 + tan θ2 = -2 * Real.sqrt 3 / 3 + 1 := sorry,
  -- Simplify to final answer
  show (sec θ1 + tan θ2) = (3 - 2 * Real.sqrt 3) / 3 := sorry,
}

end sec_tan_sum_l124_124559


namespace cookie_cost_proof_l124_124567

def cost_per_cookie (total_spent : ℕ) (days : ℕ) (cookies_per_day : ℕ) : ℕ :=
  total_spent / (days * cookies_per_day)

theorem cookie_cost_proof : cost_per_cookie 1395 31 3 = 15 := by
  sorry

end cookie_cost_proof_l124_124567


namespace limit_T_l124_124601

noncomputable def S (n : ℕ) : ℝ :=
if n = 1 then 1 else (1/4) * S (n-1)

noncomputable def T (n : ℕ) : ℝ :=
∑ i in finset.range n, S (i + 1)

theorem limit_T :
  tendsto (λ n, T n) at_top (𝓝 (4/3)) :=
sorry

end limit_T_l124_124601


namespace min_value_of_expr_l124_124279

noncomputable def min_value_expr (a b : ℝ) : ℝ := a^2 + b^2 - 6a - 4b + 13

theorem min_value_of_expr : ∃ a b : ℝ, 
  (a + 3*b - 5 = 0) ∧
  min_value_expr a b = 8 / 5 := by
  sorry

end min_value_of_expr_l124_124279


namespace coefficient_x2_in_expansion_sum_of_coefficients_in_expansion_l124_124786

theorem coefficient_x2_in_expansion :
  (polynomial.expand (1 - 2 * polynomial.X)^6).coeff 2 = 60 :=
sorry

theorem sum_of_coefficients_in_expansion :
  (polynomial.eval 1 (1 - 2 * polynomial.X)^6) = 1 :=
sorry

end coefficient_x2_in_expansion_sum_of_coefficients_in_expansion_l124_124786


namespace valentino_farm_birds_total_l124_124754

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end valentino_farm_birds_total_l124_124754


namespace slower_train_passes_faster_driver_in_24_seconds_l124_124818

theorem slower_train_passes_faster_driver_in_24_seconds
  (length_train : ℕ)
  (speed_1 : ℕ)
  (speed_2 : ℕ)
  (length_train = 500)
  (speed_1 = 45)  -- speed in km/hr
  (speed_2 = 30)  -- speed in km/hr :
  ∃ t : ℕ, t = 24 := sorry

end slower_train_passes_faster_driver_in_24_seconds_l124_124818


namespace angle_between_vectors_l124_124640

noncomputable def vec_a : ℝ × ℝ := (-2 * Real.sqrt 3, 2)
noncomputable def vec_b : ℝ × ℝ := (1, - Real.sqrt 3)

-- Define magnitudes
noncomputable def mag_a : ℝ := Real.sqrt ((-2 * Real.sqrt 3) ^ 2 + 2^2)
noncomputable def mag_b : ℝ := Real.sqrt (1^2 + (- Real.sqrt 3) ^ 2)

-- Define the dot product
noncomputable def dot_product : ℝ := (-2 * Real.sqrt 3) * 1 + 2 * (- Real.sqrt 3)

-- Define cosine of the angle theta
-- We use mag_a and mag_b defined above
noncomputable def cos_theta : ℝ := dot_product / (mag_a * mag_b)

-- Define the angle theta, within the range [0, π]
noncomputable def theta : ℝ := Real.arccos cos_theta

-- The expected result is θ = 5π / 6
theorem angle_between_vectors : theta = (5 * Real.pi) / 6 :=
by
  sorry

end angle_between_vectors_l124_124640


namespace ribbons_left_l124_124360

theorem ribbons_left {initial_ribbons morning_giveaway afternoon_giveaway ribbons_left : ℕ} 
    (h1 : initial_ribbons = 38) 
    (h2 : morning_giveaway = 14) 
    (h3 : afternoon_giveaway = 16) 
    (h4 : ribbons_left = initial_ribbons - (morning_giveaway + afternoon_giveaway)) : 
  ribbons_left = 8 := 
by 
  sorry

end ribbons_left_l124_124360


namespace storage_space_remaining_l124_124963

def total_space_remaining (first_floor second_floor: ℕ) (boxes: ℕ) : ℕ :=
  first_floor + second_floor - boxes

theorem storage_space_remaining :
  ∀ (first_floor second_floor boxes: ℕ),
  (first_floor = 2 * second_floor) →
  (boxes = 5000) →
  (boxes = second_floor / 4) →
  total_space_remaining first_floor second_floor boxes = 55000 :=
by
  intros first_floor second_floor boxes h1 h2 h3
  sorry

end storage_space_remaining_l124_124963


namespace original_cost_of_meal_l124_124330

theorem original_cost_of_meal (sales_tax tip : ℝ) (total_payment : ℝ) 
  (h_tax : sales_tax = 0.095) (h_tip : tip = 0.18) (h_total : total_payment = 45.95) :
  let x := total_payment / (1 + h_tax + h_tip) in x = 36.04 :=
by
  let x := total_payment / (1 + sales_tax + tip)
  calc
    x = 45.95 / 1.275 : by conv => {to_lhs, rw [h_tax, h_tip]}
    ... ≈ 36.04 : by norm_num⟩
  sorry

end original_cost_of_meal_l124_124330


namespace largest_r_satisfying_condition_l124_124047

theorem largest_r_satisfying_condition :
  ∃ M : ℕ, ∀ (a : ℕ → ℕ) (r : ℝ) (h : ∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))),
  (∀ n : ℕ, n ≥ M → a (n + 2) = a n) → r = 2 := 
by
  sorry

end largest_r_satisfying_condition_l124_124047


namespace diametrically_opposite_number_l124_124917

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l124_124917


namespace area_of_triangle_l124_124212

noncomputable def vector_u : ℝ × ℝ × ℝ := (2, 1, 0)
noncomputable def vector_v : ℝ × ℝ × ℝ := (5, 3, 2)
noncomputable def vector_w : ℝ × ℝ × ℝ := (11, 7, 4)

def vector_sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)

theorem area_of_triangle :
  let u := vector_u
      v := vector_v
      w := vector_w
      a := vector_sub v u
      b := vector_sub w u
      cross_prod := cross_product a b
      area := 0.5 * magnitude cross_prod
  in area = Real.sqrt 22 :=
by {
  sorry
}

end area_of_triangle_l124_124212


namespace diametrically_opposite_to_83_l124_124905

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l124_124905


namespace circle_arrangement_area_division_l124_124536

theorem circle_arrangement_area_division :
  ∃ (a b c : ℤ), (∀ n, n > 0 → Int.gcd a (Int.gcd b c) = 1) ∧
    ∃ (line : ℚ × ℚ -> Prop),
    (∀ slope x1 y1 x2 y2,
      slope = 3 ∧ line (x1, y1) ∧ line (x2, y2) → slope = (y2 - y1) / (x2 - x1)) ∧
    (∃ (region_bound : ℚ -> ℚ -> Prop),
      (∀ x y,
        region_bound x y ↔
        ((x - 1) ^ 2 + (y - 1) ^ 2 ≤ 1 ^ 2 ∨ 
         (x - 1) ^ 2 + (y - 3) ^ 2 ≤ 1 ^ 2 ∨
         (x - 1) ^ 2 + (y - 5) ^ 2 ≤ 1 ^ 2 ∨
         (x - 3) ^ 2 + (y - 1) ^ 2 ≤ 1 ^ 2 ∨
         (x - 3) ^ 2 + (y - 3) ^ 2 ≤ 1 ^ 2 ∨
         (x - 3) ^ 2 + (y - 5) ^ 2 ≤ 1 ^ 2 ∨
         (x - 5) ^ 2 + (y - 1) ^ 2 ≤ 1 ^ 2 ∨
         (x - 5) ^ 2 + (y - 3) ^ 2 ≤ 1 ^ 2 ∨
         (x - 5) ^ 2 + (y - 5) ^ 2 ≤ 1 ^ 2))) ∧
      (∃ (l : ℚ → ℚ),
        (∀ a b c, l = fun x => (3 : ℚ) * x - (4 : ℚ) ∧
         let a : ℚ := 3 in
         let b : ℚ := -1 in
         let c : ℚ := -4 in
         a^2 + b^2 + c^2 = 26)) := sorry

end circle_arrangement_area_division_l124_124536


namespace minimum_perimeter_rectangle_divided_into_squares_l124_124504

theorem minimum_perimeter_rectangle_divided_into_squares (a b : ℕ) (h_condition : 5 * a + 2 * b = 20 * a - 3 * b) : 
    2 * (2 * a + 2 * b + (3 * a + 2 * b)) = 52 := 
by 
  have h : 5 * b = 15 * a := by rw [← h_condition]; ring
  have ha : a = 1 := sorry -- Assume the smallest integer value for a
  have hb : b = 3 * a := by rw [h, ha]; ring
  rw [ha, hb]; ring
  sorry

end minimum_perimeter_rectangle_divided_into_squares_l124_124504


namespace probability_of_two_white_balls_correct_l124_124481

noncomputable def probability_of_two_white_balls : ℚ :=
  let total_balls := 15
  let white_balls := 8
  let first_draw_white := (white_balls : ℚ) / total_balls
  let second_draw_white := (white_balls - 1 : ℚ) / (total_balls - 1)
  first_draw_white * second_draw_white

theorem probability_of_two_white_balls_correct :
  probability_of_two_white_balls = 4 / 15 :=
by
  sorry

end probability_of_two_white_balls_correct_l124_124481


namespace truck_driver_pay_per_mile_l124_124511

def cost_per_gallon : ℝ := 2
def miles_per_gallon : ℝ := 10
def speed : ℝ := 30
def hours_driven : ℝ := 10
def total_earnings : ℝ := 90

def total_distance (speed : ℝ) (hours : ℝ) : ℝ := speed * hours
def gallons_needed (distance : ℝ) (miles_per_gallon : ℝ) : ℝ := distance / miles_per_gallon
def total_gas_cost (gallons : ℝ) (cost_per_gallon : ℝ) : ℝ := gallons * cost_per_gallon
def profit (earnings : ℝ) (cost : ℝ) : ℝ := earnings - cost
def pay_per_mile (profit : ℝ) (distance : ℝ) : ℝ := profit / distance

theorem truck_driver_pay_per_mile :
  let distance := total_distance speed hours_driven,
      gallons := gallons_needed distance miles_per_gallon,
      gas_cost := total_gas_cost gallons cost_per_gallon,
      net_profit := profit total_earnings gas_cost,
      pay_mile := pay_per_mile net_profit distance in
  pay_mile = 0.10 :=
by
  sorry

end truck_driver_pay_per_mile_l124_124511


namespace modulus_of_complex_division_l124_124404

theorem modulus_of_complex_division :
  complex.abs (1 / (1 + complex.I * real.sqrt 3)) = 1 / 4 :=
  sorry

end modulus_of_complex_division_l124_124404


namespace trigonometric_identity_l124_124380

theorem trigonometric_identity (α β γ : ℝ) :
  cos (α - β) * cos (β - γ) - sin (α - β) * sin (β - γ) = cos (α - γ) :=
sorry

end trigonometric_identity_l124_124380


namespace jerry_zinc_intake_l124_124327

-- Define the conditions provided in the problem
def large_antacid_weight := 2 -- in grams
def large_antacid_zinc_percent := 0.05 -- 5%

def small_antacid_weight := 1 -- in grams
def small_antacid_zinc_percent := 0.15 -- 15%

def large_antacids_count := 2
def small_antacids_count := 3

def total_zinc_mg :=
  (large_antacid_weight * large_antacid_zinc_percent * large_antacids_count +
   small_antacid_weight * small_antacid_zinc_percent * small_antacids_count) * 1000 -- converting grams to milligrams

-- The theorem to be proven
theorem jerry_zinc_intake : total_zinc_mg = 650 :=
by 
  -- translating the setup directly
  sorry -- proof will be filled here

end jerry_zinc_intake_l124_124327


namespace slower_train_passes_faster_driver_in_24_seconds_l124_124817

theorem slower_train_passes_faster_driver_in_24_seconds
  (length_train : ℕ)
  (speed_1 : ℕ)
  (speed_2 : ℕ)
  (length_train = 500)
  (speed_1 = 45)  -- speed in km/hr
  (speed_2 = 30)  -- speed in km/hr :
  ∃ t : ℕ, t = 24 := sorry

end slower_train_passes_faster_driver_in_24_seconds_l124_124817


namespace police_catches_thief_in_4_steps_l124_124787

-- Define the initial positions and the graph structure
def initial_position_P : String := "A"
def initial_position_T : String := "B"

-- Define the move as one step along the lines
def valid_move (current next : String) : Prop :=
  (current, next) ∈ [("A", "C"), ("B", "E"), ("C", "D"), ("E", "F"), ("D", "A")]

-- Prove that the minimum number of steps required for the police to catch the thief is 4
theorem police_catches_thief_in_4_steps 
  (initialP : String) (initialT : String)
  (h_initialP : initialP = initial_position_P)
  (h_initialT : initialT = initial_position_T) :
  ∃ steps, steps ≤ 4 ∧ (police_catches_thief steps initialP initialT) :=
  sorry

-- Additional definitions and theorems needed for the proof would be defined here:
-- - police_catches_thief
--   Ensuring capturing of the logic that validates if the police catches the thief in given steps starting from given positions.

end police_catches_thief_in_4_steps_l124_124787


namespace increase_average_by_4_l124_124940

-- Definitions and conditions
def innings_10_average : ℕ := 18

def total_runs_10_innings : ℕ := 180

def runs_next_inning : ℕ := 62

def total_runs_11_innings : ℕ := total_runs_10_innings + runs_next_inning

theorem increase_average_by_4 :
  let x := 4 in
  (innings_10_average + x) * 11 = total_runs_11_innings :=
by
  sorry

end increase_average_by_4_l124_124940


namespace original_price_of_cycle_l124_124144

/-
  A cycle is sold for Rs. 1100, resulting in a gain of 22.22222222222222%. 
  Prove that the original price of the cycle was approximately Rs. 900.
-/

theorem original_price_of_cycle (S : ℝ) (G : ℝ) (P : ℝ) (approx_P : ℝ) 
    (hS : S = 1100) 
    (hG : G = 22.22222222222222 / 100) 
    (approx_P_eq : approx_P = 900) :
    ((S - P) / P) = G → P ≈ approx_P := 
sorry

end original_price_of_cycle_l124_124144


namespace probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124826

-- Define the problem's parameters and conditions
def n : ℕ  -- number of pairs of socks

-- Define the successful pair probability calculation
theorem probability_all_pairs_successful (n : ℕ) :
  (∑ s in Finset.range (n + 1), s.factorial * 2 ^ s) / ((2 * n).factorial) = 2^n * n.factorial := sorry

-- Define the expected value calculation
theorem expected_successful_pairs_greater_than_half (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n, (1 : ℝ) / (2 * n - 1) : ℝ) / n > 0.5 := sorry

end probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124826


namespace sqrt_div_l124_124180

theorem sqrt_div (a b : ℝ) (h1 : a = 28) (h2 : b = 7) :
  Real.sqrt a / Real.sqrt b = 2 := 
by 
  sorry

end sqrt_div_l124_124180


namespace triangle_area_is_24_l124_124260

-- conditions
def square_area1 : ℝ := 36
def square_area2 : ℝ := 64
def square_area3 : ℝ := 100

-- definitions
def side1 := Real.sqrt square_area1
def side2 := Real.sqrt square_area2
def side3 := Real.sqrt square_area3

-- theorem
theorem triangle_area_is_24 :
  side1 = 6 ∧ side2 = 8 ∧ side3 = 10 →
  (side1^2 + side2^2 = side3^2) →
  (1/2 * side1 * side2 = 24) := 
by {
  intros,
  sorry
}

end triangle_area_is_24_l124_124260


namespace area_hexagon_ABCDEF_l124_124004

-- Define the regular hexagon ABCDEF
variables {A B C D E F J K L M N O : Type}

-- Assume the conditions given in the problem
variables [hexagon: regular_hexagon A B C D E F]
variables [midpoints: midpoints J K L A B C D E F]
variables [division_pts: division_points M N O B C D E F A]

-- Given condition about the area of triangle JKL
axiom area_JKL : area J K L = 100

-- Statement of the theorem to be proved
theorem area_hexagon_ABCDEF : area A B C D E F = 300 :=
by
  sorry

end area_hexagon_ABCDEF_l124_124004


namespace proof_math_problem_l124_124105

-- Definitions based on the conditions
def pow_zero (x : ℝ) : x^0 = 1 := by
  rw [pow_zero]
  exact one

def three_pow_neg_three := (3 : ℝ)^(-3)
def three_pow_zero := (3 : ℝ)^0

-- The final statement to be proven
theorem proof_math_problem :
  (three_pow_neg_three)^0 + (three_pow_zero)^4 = 2 :=
by
  simp [three_pow_neg_three, three_pow_zero, pow_zero]
  sorry

end proof_math_problem_l124_124105


namespace longest_boat_length_l124_124026

variable (saved money : ℕ) (license_fee docking_multiplier boat_cost : ℕ)

theorem longest_boat_length (h1 : saved = 20000) 
                           (h2 : license_fee = 500) 
                           (h3 : docking_multiplier = 3)
                           (h4 : boat_cost = 1500) : 
                           (saved - license_fee - docking_multiplier * license_fee) / boat_cost = 12 := 
by 
  sorry

end longest_boat_length_l124_124026


namespace resulting_solid_faces_l124_124130

-- Define a cube structure with a given number of faces
structure Cube where
  faces : Nat

-- Define the problem conditions and prove the total faces of the resulting solid
def original_cube := Cube.mk 6

def new_faces_per_cube := 5

def total_new_faces := original_cube.faces * new_faces_per_cube

def total_faces_of_resulting_solid := total_new_faces + original_cube.faces

theorem resulting_solid_faces : total_faces_of_resulting_solid = 36 := by
  sorry

end resulting_solid_faces_l124_124130


namespace ellipse_eq_proof_l124_124315

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_eq_proof :
  ∃ (a b : ℝ), b = 1 ∧ a = sqrt 2 ∧ (ellipse_equation a b) :=
by
  use [sqrt 2, 1]
  split
  · exact rfl
  split
  · exact rfl
  · intros x y
    sorry

end ellipse_eq_proof_l124_124315


namespace count_three_digit_numbers_using_1_and_2_l124_124071

theorem count_three_digit_numbers_using_1_and_2 : 
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 6 :=
by
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 6
  sorry

end count_three_digit_numbers_using_1_and_2_l124_124071


namespace estimate_students_less_than_2_hours_probability_one_male_one_female_l124_124089

-- Definitions from the conditions
def total_students_surveyed : ℕ := 40
def total_grade_ninth_students : ℕ := 400
def freq_0_1 : ℕ := 8
def freq_1_2 : ℕ := 20
def freq_2_3 : ℕ := 7
def freq_3_4 : ℕ := 5
def male_students_at_least_3_hours : ℕ := 2
def female_students_at_least_3_hours : ℕ := 3

-- Question 1 proof statement
theorem estimate_students_less_than_2_hours :
  total_grade_ninth_students * (freq_0_1 + freq_1_2) / total_students_surveyed = 280 :=
by sorry

-- Question 2 proof statement
theorem probability_one_male_one_female :
  (male_students_at_least_3_hours * female_students_at_least_3_hours) / (Nat.choose 5 2) = (3 / 5) :=
by sorry

end estimate_students_less_than_2_hours_probability_one_male_one_female_l124_124089


namespace flour_amount_indeterminable_l124_124357

variable (flour_required : ℕ)
variable (sugar_required : ℕ := 11)
variable (sugar_added : ℕ := 10)
variable (flour_added : ℕ := 12)
variable (sugar_to_add : ℕ := 1)

theorem flour_amount_indeterminable :
  ¬ ∃ (flour_required : ℕ), flour_additional = flour_required - flour_added :=
by
  sorry

end flour_amount_indeterminable_l124_124357


namespace minor_arc_length_l124_124713

theorem minor_arc_length (D E F : Point) (c : Circle) (r : ℝ) (h₁ : r = 24)
  (h₂ : is_on_circle D c) (h₃ : is_on_circle E c) (h₄ : is_on_circle F c) (h₅ : ∠ DEF = 60) :
  arc_length D F c = 8 * π :=
by sorry

end minor_arc_length_l124_124713


namespace minimum_PM_PN_is_7_l124_124610

noncomputable def minimum_PM_PN : ℝ :=
  let ellipse (P : ℝ × ℝ) := ((P.1^2) / 25 + (P.2^2) / 16 = 1)
  let circle1 (M : ℝ × ℝ) := (((M.1 + 3) ^ 2) + (M.2 ^ 2) = 1)
  let circle2 (N : ℝ × ℝ) := (((N.1 - 3) ^ 2) + (N.2 ^ 2) = 4)
  Inf ((λ (P M N : ℝ × ℝ), |P.1 - M.1| + |P.2 - M.2| + |P.1 - N.1| + |P.2 - N.2|)
       '' { (P, M, N) | ellipse P ∧ circle1 M ∧ circle2 N})

theorem minimum_PM_PN_is_7 :
  minimum_PM_PN = 7 := sorry

end minimum_PM_PN_is_7_l124_124610


namespace opposite_point_83_is_84_l124_124899

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l124_124899


namespace trapezoid_reassembly_area_conservation_l124_124131

theorem trapezoid_reassembly_area_conservation
  {height length new_width : ℝ}
  (h1 : height = 9)
  (h2 : length = 16)
  (h3 : new_width = y)  -- each base of the trapezoid measures y.
  (div_trapezoids : ∀ (a b c : ℝ), 3 * a = height → a = 9 / 3)
  (area_conserved : length * height = (3 / 2) * (3 * (length + new_width)))
  : new_width = 16 :=
by
  -- The proof is skipped
  sorry

end trapezoid_reassembly_area_conservation_l124_124131


namespace fourth_smallest_part_value_l124_124202

noncomputable def sum_ratios : ℝ := 3.5 + 6.5 + 8.25 + 5.6 + 9.1 + 7.8

noncomputable def part (r : ℝ) : ℝ := (r * 1200) / sum_ratios

noncomputable def fourth_smallest_part : ℝ := 
  let parts := [part 3.5, part 6.5, part 8.25, part 5.6, part 9.1, part 7.8]
  (parts.sorted_nth 3).get_or_else 0

theorem fourth_smallest_part_value : fourth_smallest_part ≈ 229.02 := by
  sorry

end fourth_smallest_part_value_l124_124202


namespace toys_produced_each_day_l124_124941

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) 
  (h1 : weekly_production = 5500) (h2 : days_worked = 4) : 
  (weekly_production / days_worked = 1375) :=
sorry

end toys_produced_each_day_l124_124941


namespace disk_tangent_position_l124_124489

theorem disk_tangent_position (R_clock R_disk : ℝ) (h_clock_nonneg: 0 ≤ R_clock) (h_disk_nonneg: 0 ≤ R_disk) (h_relation: R_disk = R_clock / 2) :
  let C_clock := 2 * Real.pi * R_clock,
      C_disk := 2 * Real.pi * R_disk,
      rotation_ratio := C_clock / C_disk in
  (360 * rotation_ratio / 2) % 360 = 180 →
  nat.mod (12 - int.natAbs ((360 * rotation_ratio / 2) % 360) / 30) 12 = 6 :=
by
  sorry

end disk_tangent_position_l124_124489


namespace monotonicity_range_of_a_l124_124737

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - (1 + a) * x^2 + 4 * a * x + 24 * a

theorem monotonicity (a : ℝ) (h₀ : 1 < a) : 
  (∀ x, x < 2 → f (x, a) ≤ f (2, a)) ∧ 
  (∀ x, 2 < x ∧ x < 2*a → f (2, a) ≤ f (x, a)) ∧ 
  (∀ x, 2*a < x → f (2*a, a) ≤ f (x, a)) :=
sorry

theorem range_of_a (a : ℝ) :
  (1 < a ∧ (∀ x, x ≥ 0 → f (x, a) > 0)) → (1 < a ∧ a < 6) :=
sorry

end monotonicity_range_of_a_l124_124737


namespace opposite_number_l124_124910

theorem opposite_number (n : ℕ) (hn : n = 100) (N : Fin n) (hN : N = 83) :
    ∃ M : Fin n, M = 84 :=
by
  sorry
 
end opposite_number_l124_124910


namespace value_of_x_l124_124303

theorem value_of_x (a x y : ℝ) 
  (h1 : a^(x - y) = 343) 
  (h2 : a^(x + y) = 16807) : x = 4 :=
by
  sorry

end value_of_x_l124_124303


namespace greatest_L_l124_124446

def is_sweet (a : Fin 2023 → ℕ) : Prop :=
  (∑ i, a i) = 2023 ∧ (∑ i, (a i) / (2^((i : ℕ) + 1) : ℝ)) ≤ 1

theorem greatest_L (a : Fin 2023 → ℕ) (h : is_sweet a) : 
  (∑ i, (a i) * (i : ℕ + 1)) ≥ 22228 :=
sorry

end greatest_L_l124_124446


namespace minimum_x_plus_y_l124_124254

theorem minimum_x_plus_y (x y : ℝ) (h1 : x ∈ set.Icc 2 4) (h2 : (-1 + 5 - 1/x + y) / 4 = 3) : x + y = 21 / 2 :=
begin
  sorry
end

end minimum_x_plus_y_l124_124254


namespace find_b_c_angle_between_m_n_l124_124607

section vector_problems

-- Definitions of the vectors and conditions
def a : (ℝ × ℝ) := (3, 4)
def b : (ℝ × ℝ) := (9, 12)
def c : (ℝ × ℝ) := (4, -3)

-- Problem (1) assertion
theorem find_b_c : b = (9, 12) ∧ c = (4, -3) :=
  sorry

-- Definitions of m and n
def m : (ℝ × ℝ) := (2 * 3 - 9, 2 * 4 - 12)  -- == (-3, -4)
def n : (ℝ × ℝ) := (3 + 4, 4 - 3)  -- == (7, 1)

-- Problem (2) assertion
theorem angle_between_m_n : real.arccos ((m.1 * n.1 + m.2 * n.2) / (real.sqrt (m.1^2 + m.2^2) * real.sqrt (n.1^2 + n.2^2))) = 3 * real.pi / 4 :=
  sorry

end vector_problems

end find_b_c_angle_between_m_n_l124_124607


namespace p_expression_l124_124249

theorem p_expression (m n p : ℤ) (r1 r2 : ℝ) 
  (h1 : r1 + r2 = m) 
  (h2 : r1 * r2 = n) 
  (h3 : r1^2 + r2^2 = p) : 
  p = m^2 - 2 * n := by
  sorry

end p_expression_l124_124249


namespace ratio_of_octagon_areas_l124_124949

-- Define the properties of the circle and octagons
def radius (r : ℝ) : Prop := r > 0
def inscribed_octagon_area (r : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * (2 * r * Real.sin (Real.pi / 8)) ^ 2
def circumscribed_octagon_area (r : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * (r * Real.sec (Real.pi / 8)) ^ 2

-- The theorem stating the ratio of the areas
theorem ratio_of_octagon_areas (r : ℝ) (hr : radius r) :
  circumscribed_octagon_area r / inscribed_octagon_area r = 1 / 2 :=
sorry

end ratio_of_octagon_areas_l124_124949


namespace sqrt_pow_mul_l124_124985

theorem sqrt_pow_mul (a b : ℝ) : (a = 3) → (b = 5) → (Real.sqrt (a^2 * b^6) = 375) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end sqrt_pow_mul_l124_124985


namespace perpendicular_vectors_k_value_l124_124282

theorem perpendicular_vectors_k_value (k : ℝ) (a b: ℝ × ℝ)
  (h_a : a = (-1, 3)) (h_b : b = (1, k)) (h_perp : (a.1 * b.1 + a.2 * b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end perpendicular_vectors_k_value_l124_124282


namespace unique_solution_condition_l124_124381

variable (c d x : ℝ)

-- Define the equation
def equation : Prop := 4 * x - 7 + c = d * x + 3

-- Lean theorem for the proof problem
theorem unique_solution_condition :
  (∃! x, equation c d x) ↔ d ≠ 4 :=
sorry

end unique_solution_condition_l124_124381


namespace diametrically_opposite_to_83_l124_124908

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l124_124908


namespace no_convex_polygon_with_special_properties_l124_124693

noncomputable def exists_convex_polygon_with_special_properties : Prop :=
  ∃ (P : Polygon), isConvex P ∧ (∀ (side : Side P), ∃ (diag : Diagonal P), length side = length diag) ∧
                   (∀ (diag : Diagonal P), ∃ (side : Side P), length diag = length side)

theorem no_convex_polygon_with_special_properties : ¬ exists_convex_polygon_with_special_properties := 
by
  sorry

end no_convex_polygon_with_special_properties_l124_124693


namespace solution_interval_log_eq_l124_124803

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) + x - 3

theorem solution_interval_log_eq (h_mono : ∀ x y, (0 < x ∧ x < y) → f x < f y)
  (h_f2 : f 2 = 0)
  (h_f3 : f 3 > 0) :
  ∃ x, (2 ≤ x ∧ x < 3 ∧ f x = 0) :=
by
  sorry

end solution_interval_log_eq_l124_124803


namespace collinear_points_l124_124437

noncomputable theory
open_locale classical

variables (a b : Line)
variables (A1 A2 B1 B2 P M1 M2 : Point)
variables (a1 b1 A2P B2P : Line)

-- Conditions
def lines_intersect (a b : Line) (P Q : Point) : Prop :=
  P ≠ Q ∧ P ∈ a ∧ Q ∈ a ∧ P ∈ b ∧ Q ∈ b

def is_midpoint (M A B : Point) : Prop :=
  dist M A = dist M B ∧ M ∈ open_segment A B

def are_parallel (l1 l2 : Line) : Prop :=
  ∃ u v : Point, u ≠ v ∧ u ∈ l1 ∧ v ∈ l1 ∧ u ∈ l2 ∧ v ∈ l2 ∧ is_parallel l1 l2

def on_line (P : Point) (l : Line) : Prop :=
  P ∈ l

-- Assumptions
axiom lines_a_b_intersect : lines_intersect a b A1 A2 ∧ lines_intersect a b B1 B2
axiom P_on_A1B1 : on_line P (line_through A1 B1)
axiom M1_is_midpoint_A1B1 : is_midpoint M1 A1 B1
axiom a1_parallel_a : are_parallel a1 a
axiom b1_parallel_b : are_parallel b1 b

-- Problem Statement
theorem collinear_points : 
  ∃ X Y : Point, 
    (on_line X A2P ∧ on_line X a1 ∧ on_line Y B2P ∧ on_line Y b1 ∧
    is_midpoint M2 A2 B2 ∧ collinear {X, Y, M2}) :=
sorry

end collinear_points_l124_124437


namespace right_triangle_cos_q_l124_124383

theorem right_triangle_cos_q (P Q R : Type) [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q] [InnerProductSpace ℝ R]
  (hPQR_th : ∠ P Q R = 90)
  (hcosQ : cos (∠ P Q R) = 3 / 5)
  (hQR : dist Q R = 3) :
  dist P R = 5 :=
sorry

end right_triangle_cos_q_l124_124383


namespace real_root_of_sqrt_eq_l124_124556

theorem real_root_of_sqrt_eq : ∃ (x : ℝ), sqrt x + sqrt (x + 8) = 8 ∧ x = 49 / 4 :=
by
  existsi (49 / 4 : ℝ)
  split
  sorry

end real_root_of_sqrt_eq_l124_124556


namespace diametrically_opposite_to_83_l124_124903

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l124_124903


namespace equation_of_parabola_equation_of_line_l_through_F_l124_124240

-- Conditions
def parabola_C (p : ℝ) := ∀ (x y : ℝ), y^2 = 2 * p * x
def focus_F (F : ℝ × ℝ) := F = (1, 0)
def line_at_y (y_intersect : ℝ → ℝ) := y_intersect 4
def points_P_Q (P Q : ℝ × ℝ) :=  P = (0, 4) ∧ (∃ x0, Q = (x0, 4))
def dist_condition (abs_dist : ℝ → ℝ → Prop) := ∀ (Q F P : ℝ × ℝ), abs_dist |QF| = (5 / 4) * |PQ|

-- Proof Problems
theorem equation_of_parabola : ∃ (p : ℝ), p = 2 ∧ parabola_C 2 := sorry

theorem equation_of_line_l_through_F (F : ℝ × ℝ) (l : ℝ × ℝ → Prop) : 
  focus_F F →
  (∀ (A B : ℝ × ℝ), l (A.1, A.2) → l (B.1, B.2)) → 
  (∃ (eq_l : ℝ × ℝ → Prop), (eq_l (x, y) ↔ x - y - 1 = 0 ∨ x + y - 1 = 0)) :=
sorry

end equation_of_parabola_equation_of_line_l_through_F_l124_124240


namespace each_friend_pays_l124_124220

def hamburgers_cost : ℝ := 5 * 3
def fries_cost : ℝ := 4 * 1.20
def soda_cost : ℝ := 5 * 0.50
def spaghetti_cost : ℝ := 1 * 2.70
def total_cost : ℝ := hamburgers_cost + fries_cost + soda_cost + spaghetti_cost
def num_friends : ℝ := 5

theorem each_friend_pays :
  total_cost / num_friends = 5 :=
by
  sorry

end each_friend_pays_l124_124220


namespace find_k_solution_l124_124630

theorem find_k_solution 
  (k : ℝ)
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) : 
  k = 2 :=
sorry

end find_k_solution_l124_124630


namespace new_polynomial_roots_l124_124721

noncomputable def transformed_roots_polynomial : Polynomial ℂ :=
  Polynomial.ofRoots [(4 - a) / a, (4 - b) / b, (4 - c) / c, (4 - d) / d]

def original_polynomial : Polynomial ℂ :=
  Polynomial.C (-2) + Polynomial.X * (Polynomial.C (-7) + Polynomial.X * (Polynomial.C 14 + Polynomial.X * (Polynomial.C (-7) + Polynomial.X^4)))

def roots_of_original_polynomial (a b c d : ℂ) : Prop :=
  original_polynomial.isRoot a ∧ original_polynomial.isRoot b ∧ original_polynomial.isRoot c ∧ original_polynomial.isRoot d

theorem new_polynomial_roots (a b c d : ℂ) (h : roots_of_original_polynomial a b c d) :
  transformed_roots_polynomial = Polynomial.C 2 + Polynomial.X * (Polynomial.C 28 + Polynomial.X * (-Polynomial.C 224 + Polynomial.X * (Polynomial.C 448 - Polynomial.X^4))) :=
sorry

end new_polynomial_roots_l124_124721


namespace number_of_licenses_l124_124501

-- We define the conditions for the problem
def number_of_letters : ℕ := 3  -- B, C, or D
def number_of_digits : ℕ := 4   -- Four digits following the letter
def choices_per_digit : ℕ := 10 -- Each digit can range from 0 to 9

-- We define the total number of licenses that can be generated
def total_licenses : ℕ := number_of_letters * (choices_per_digit ^ number_of_digits)

-- We now state the theorem to be proved
theorem number_of_licenses : total_licenses = 30000 :=
by
  sorry

end number_of_licenses_l124_124501


namespace tan_20_tan_70_eq_one_l124_124860

theorem tan_20_tan_70_eq_one
    (h1: Real.sin 50 = Real.cos 40)
    (h2: Real.cos 40 ≠ Real.sin 40)
    (h3: Real.tan 70 = Real.cot 20)
    (h4: Real.tan 20 * Real.cot 20 = 1)
    (h5: 30 < 35)
    (h6: Real.cos 30 > Real.cos 35)
    (h7: Real.sin 30^2 + Real.cos 30^2 = 1)
    (h8: Real.sin 30 ≠ Real.cos 30) :
    Real.tan 20 * Real.tan 70 = 1 :=
by
  sorry

end tan_20_tan_70_eq_one_l124_124860


namespace nine_digit_palindromes_count_l124_124096

theorem nine_digit_palindromes_count :
  ∃ n : ℕ, (n = 2500) ∧ 
  (∀ d1 d2 d3 d4 d5 : ℕ,
     (d1 ∈ {1, 7, 8, 9}) ∧
     (d2 ∈ {0, 1, 7, 8, 9}) ∧
     (d3 ∈ {0, 1, 7, 8, 9}) ∧
     (d4 ∈ {0, 1, 7, 8, 9}) ∧
     (d5 ∈ {0, 1, 7, 8, 9}) → 
     (∃ d6 d7 d8 d9 : ℕ,
        d6 = d4 ∧ d7 = d3 ∧ d8 = d2 ∧ d9 = d1 ∧
        palindrome (d1::d2::d3::d4::d5::d6::d7::d8::d9::[]))) :=
by {
  dsuffices h : 4 * 5^4 = 2500, from ⟨2500, h, sorry⟩,
  norm_num,
  sorry
}

end nine_digit_palindromes_count_l124_124096


namespace domain_of_f_exists_m_with_minimum_value_l124_124263

-- Define the domain condition
def domain_condition {t : ℝ} : Prop :=
  2 - t > 0 ∧ t - 1 ≥ 0

-- Define the function f(t)
def f (t : ℝ) : ℝ := Real.log 2 (2 - t) + Real.sqrt (t - 1)

-- Define the function g(x)
def g (x m : ℝ) : ℝ := x^2 + 2 * m * x - m^2

-- Prove that the domain of f is [1, 2)
theorem domain_of_f : ∀ t, domain_condition t ↔ (1 ≤ t ∧ t < 2) := by
  sorry

-- Prove there exists an m such that g(x) has a minimum value of 2 on the domain [1, 2)
theorem exists_m_with_minimum_value : ∃ m, ∀ x, 1 ≤ x ∧ x < 2 → g x m ≥ 2 ∧ g 1 m = 2 := by
  use 1
  intros x hx
  split
  -- prove g(x) ≥ 2 within the domain [1, 2)
  sorry
  -- prove g(1, 1) = 2
  sorry

end domain_of_f_exists_m_with_minimum_value_l124_124263


namespace seating_arrangements_possible_l124_124573

theorem seating_arrangements_possible (siblings : Fin 8) : 
  siblings ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
  (∀ i ∈ {1, 2, 3, 4}, (siblings[i] ≠ siblings[i+4])) → 
  ∀ (row_first : List (Fin 4)) (row_second : List (Fin 4)), 
  (∀ i, row_first[i] ≠ row_second[i]) ∧
  (∀ i, (siblings.row_first[i] ≠ siblings.row_second[i])) →
  List.length row_first = 4 ∧ List.length row_second = 4 →
  List.length (options mates row_first row_second) = 3456
  := 
sorry

end seating_arrangements_possible_l124_124573


namespace f_242_value_l124_124419

noncomputable def f : ℕ → ℝ := sorry

axiom f_eq (a b n : ℕ) (h : a + b = 3^n) : f a + f b = 2 * n^2

axiom f_hyp_1 : f(1) = 1
axiom f_hyp_2 : f(2) = 1

lemma f_26 : f 26 = 2 * 3^2 - f 2 := by sorry

lemma f_242 : f 242 = 2 * 5^2 - f 26 := by sorry

theorem f_242_value : f 242 = 33 :=
by
  have hyp_f_2 := f_hyp_2
  have hyp_f_1 := f_hyp_1
  have f_2_eq := f 2 + f 1 = 2
  have f2_val : f 2 = 1 := hyp_f_2
  have f26_val : f 26 = 2 * 3^2 - f2_val := by sorry
  have f242_val : f 242 = 2 * 5^2 - f26_val := by sorry
  sorry

end f_242_value_l124_124419


namespace find_a_l124_124637

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2a-1, a^2+1}

theorem find_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 ∨ a = 0 := sorry

end find_a_l124_124637


namespace circle_center_radius_l124_124213

theorem circle_center_radius (x y : ℝ) :
    x^2 + y^2 - 2*x - 3 = 0 → ∃ h k r : ℝ, (h = 1) ∧ (k = 0) ∧ (r = 2) ∧ ((x - h)^2 + (y - k)^2 = r^2) :=
by
  intro h_eq
  use 1
  use 0
  use 2
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  sorry

end circle_center_radius_l124_124213


namespace five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l124_124424

-- Prove that the number of five-digit numbers is 27216
theorem five_digit_numbers_count : ∃ n, n = 9 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the number of five-digit numbers greater than or equal to 30000 is 21168
theorem five_digit_numbers_ge_30000 : 
  ∃ n, n = 7 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the rank of 50124 among five-digit numbers with distinct digits in descending order is 15119
theorem rank_of_50124 : 
  ∃ n, n = (Nat.factorial 5) - 1 := by
  sorry

end five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l124_124424


namespace amy_school_year_hours_per_week_l124_124169

-- Define the conditions
def summer_hours_per_week : ℝ := 45
def summer_weeks : ℝ := 8
def summer_earnings : ℝ := 3600

def school_year_weeks : ℝ := 24
def target_earnings : ℝ := 3600

-- Total hours worked in summer
def total_summer_hours := summer_hours_per_week * summer_weeks

-- Hourly wage calculated from summer earnings
def hourly_wage := summer_earnings / total_summer_hours

-- Total hours needed during the school year to make target earnings
def total_school_year_hours := target_earnings / hourly_wage

-- Hours per week needed during the school year
def school_year_hours_per_week := total_school_year_hours / school_year_weeks

theorem amy_school_year_hours_per_week : school_year_hours_per_week = 15 := 
  sorry

end amy_school_year_hours_per_week_l124_124169


namespace focal_length_of_hyperbola_l124_124629

-- Definition and conditions from the problem
def hyperbola_has_perpendicular_asymptote (a : ℝ) :=
  (∃ c, (∀ x y, x ≠ y → c * (x + y + 1) = 0) ∧ (∃ y z, y ≠ z → c * y = ±(x / a))) ∧
  (∃ x y, x ≠ y → (x + y + 1) = 0)

-- Focal length calculation
def focal_length (a : ℝ) (b : ℝ) :=
  2 * Real.sqrt (a^2 + b^2)

theorem focal_length_of_hyperbola : hyperbola_has_perpendicular_asymptote 1 → focal_length 1 1 = 2 * Real.sqrt 2 := sorry

end focal_length_of_hyperbola_l124_124629


namespace initial_card_distribution_l124_124954

variables {A B C D : ℕ}

theorem initial_card_distribution 
  (total_cards : A + B + C + D = 32)
  (alfred_final : ∀ c, c = A → ((c / 2) + (c / 2)) + B + C + D = 8)
  (bruno_final : ∀ c, c = B → ((c / 2) + (c / 2)) + A + C + D = 8)
  (christof_final : ∀ c, c = C → ((c / 2) + (c / 2)) + A + B + D = 8)
  : A = 7 ∧ B = 7 ∧ C = 10 ∧ D = 8 :=
by sorry

end initial_card_distribution_l124_124954


namespace edge_labeling_large_count_l124_124505

-- Define a regular dodecahedron
def dodecahedron : SimpleGraph ℕ := sorry

-- Each face must have exactly 2 edges labeled as 1, and the remaining edges as 0
def valid_edge_labeling (labels : Finset (ℕ × ℕ)) : Prop :=
  ∀ f ∈ faces, (∑ e in f.edges, labels.count (1)) = 2

-- Main statement, asserting the number of valid labelings is very large
theorem edge_labeling_large_count : 
  ∃ N : ℕ, N ≥ 10^12 ∧ 
  (∃ labels : Finset (ℕ × ℕ), valid_edge_labeling labels) :=
sorry

end edge_labeling_large_count_l124_124505


namespace find_t_from_integral_l124_124295

theorem find_t_from_integral :
  (∫ x in (1 : ℝ)..t, (-1 / x + 2 * x)) = (3 - Real.log 2) → t = 2 :=
by
  sorry

end find_t_from_integral_l124_124295


namespace rice_grain_difference_l124_124747

theorem rice_grain_difference :
  (3^8) - (3^1 + 3^2 + 3^3 + 3^4 + 3^5) = 6198 :=
by
  sorry

end rice_grain_difference_l124_124747


namespace jerry_pool_depth_l124_124203

theorem jerry_pool_depth :
  ∀ (total_gallons : ℝ) (gallons_drinking_cooking : ℝ) (gallons_per_shower : ℝ)
    (number_of_showers : ℝ) (pool_length : ℝ) (pool_width : ℝ)
    (gallons_per_cubic_foot : ℝ),
    total_gallons = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    number_of_showers = 15 →
    pool_length = 10 →
    pool_width = 10 →
    gallons_per_cubic_foot = 1 →
    (total_gallons - (gallons_drinking_cooking + gallons_per_shower * number_of_showers)) / 
    (pool_length * pool_width) = 6 := 
by
  intros total_gallons gallons_drinking_cooking gallons_per_shower number_of_showers pool_length pool_width gallons_per_cubic_foot
  intros total_gallons_eq drinking_cooking_eq shower_eq showers_eq length_eq width_eq cubic_foot_eq
  sorry

end jerry_pool_depth_l124_124203


namespace parabola_hyperbola_tangent_l124_124073

theorem parabola_hyperbola_tangent (m : ℝ) :
  (∃ x y : ℝ, (y = x^2 + 4) ∧ (y^2 - 4*m*x^2 = 4))
  → (m = 2 + √3 ∨ m = 2 - √3) := by
  sorry

end parabola_hyperbola_tangent_l124_124073


namespace tourist_attraction_visitors_l124_124888

variable (m n : ℕ)

def visitors_day_1 := m
def visitors_day_2 := m + (n + 1000)
def total_visitors := visitors_day_1 + visitors_day_2

theorem tourist_attraction_visitors :
  total_visitors m n = 2 * m + n + 1000 :=
by sorry

end tourist_attraction_visitors_l124_124888


namespace segment_length_reflection_l124_124091

theorem segment_length_reflection (F : ℝ × ℝ) (F' : ℝ × ℝ)
  (hF : F = (-4, -2)) (hF' : F' = (4, -2)) :
  dist F F' = 8 :=
by
  sorry

end segment_length_reflection_l124_124091


namespace smallest_angle_between_radii_l124_124098

theorem smallest_angle_between_radii (n : ℕ) (k : ℕ) (angle_step : ℕ) (angle_smallest : ℕ) 
(h_n : n = 40) 
(h_k : k = 23) 
(h_angle_step : angle_step = k) 
(h_angle_smallest : angle_smallest = 23) : 
angle_smallest = 23 :=
sorry

end smallest_angle_between_radii_l124_124098


namespace acute_triangle_sine_cosine_inequality_l124_124294

theorem acute_triangle_sine_cosine_inequality
  (α β γ : ℝ)
  (hαβγ_sum : α + β + γ = 180)
  (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (hα_acute : α < 90) (hβ_acute : β < 90) (hγ_acute : γ < 90) :
  sin α + sin β > cos α + cos β + cos γ :=
sorry

end acute_triangle_sine_cosine_inequality_l124_124294


namespace percent_capacity_per_cup_l124_124439

theorem percent_capacity_per_cup 
  (C : ℝ) (hC : C > 0) 
  (juice_fraction : ℝ) (h_fraction : juice_fraction = 2 / 3)
  (num_cups : ℕ) (h_cups : num_cups = 6) :
  let juice_amount := juice_fraction * C in
  let cup_capacity := juice_amount / num_cups in
  let percent_per_cup := (cup_capacity / C) * 100 in
  percent_per_cup ≈ 11.11 :=
by
  -- The proof will be completed here
  sorry

end percent_capacity_per_cup_l124_124439


namespace power_sum_is_two_l124_124099

theorem power_sum_is_two :
  (3 ^ (-3) ^ 0 + (3 ^ 0) ^ 4) = 2 := by
    sorry

end power_sum_is_two_l124_124099


namespace total_profit_l124_124882

theorem total_profit (investment_B : ℝ) (period_B : ℝ) (profit_B : ℝ) (investment_A : ℝ) (period_A : ℝ) (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = 6000)
  (h4 : profit_B / (profit_A * 6 + profit_B) = profit_B) : total_profit = 7 * 6000 :=
by 
  sorry

#print axioms total_profit

end total_profit_l124_124882


namespace shortest_distance_is_sqrt_5_l124_124006
noncomputable def min_distance_between_lines : ℝ :=
let a := (4, 2, 1)
let b := (1, 1, 5)
let u := (3, -3, 2)
let v := (2, 4, -2)
let distance_square (t s : ℝ) :=
  let P := (3 * t + a.1, -3 * t + a.2, 2 * t + a.3)
  let Q := (2 * s + b.1, 4 * s + b.2, -2 * s + b.3)
  ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)
let min_distance_square :=
  let s := λ t, -t - 1
  let pq2 := λ t, distance_square t (s t)
  minimize pq2
have pq2_min : pq2 (-3 / 26) = 26 := by sorry
sqrt pq2_min = 5

theorem shortest_distance_is_sqrt_5 :
  min_distance_between_lines = sqrt 5 :=
by sorry

end shortest_distance_is_sqrt_5_l124_124006


namespace count_sets_l124_124070

open Set

def M : Set (Set ℕ) := {M | {1, 2} ⊂ M ∧ M ⊂ {1, 2, 3, 4, 5}}

theorem count_sets : (M.count) = 6 := sorry

end count_sets_l124_124070


namespace avg_customers_per_table_l124_124513

theorem avg_customers_per_table 
    (tables : ℝ) 
    (women : ℝ) 
    (men : ℝ)
    (h_tables : tables = 9.0)
    (h_women : women = 7.0)
    (h_men : men = 3.0) : 
    (women + men) / tables ≈ 1.11 := 
by 
    sorry

end avg_customers_per_table_l124_124513


namespace trig_product_computation_l124_124184

theorem trig_product_computation :
  (1 - sin (Real.pi / 12)) * (1 - sin (5 * Real.pi / 12)) *
  (1 - sin (7 * Real.pi / 12)) * (1 - sin (11 * Real.pi / 12)) 
  = 1 / 16 :=
by 
  sorry

end trig_product_computation_l124_124184


namespace find_x_l124_124596

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l124_124596


namespace area_of_sector_l124_124243

-- Condition: central angle
def α : Real := 7 / (2 * Real.pi)

-- Condition: arc length
def l : Real := 7

-- Variable: radius, derived from arc length and central angle
def r : Real := l / α

-- Variable: area of the sector
def S : Real := (1 / 2) * l * r

-- We state the theorem to prove that the area is 7π
theorem area_of_sector : S = 7 * Real.pi := by
  sorry  -- The proof is omitted as per instructions

end area_of_sector_l124_124243


namespace junior_toys_l124_124336

theorem junior_toys (x : ℕ) (h1 : 16 = 16) (h2 : 16 * 3 = 48) 
  (h3 : x + 2 * x + 4 * x + x = 8 * x) (h4 : 8 * x = 48) : x = 6 :=
  by
  rw ←h2 at h4
  exact (Nat.eq_of_mul_eq_mul_left _ h4).mp rfl
  -- sorry could also be used here to skip the proof if required by your own instruction.

end junior_toys_l124_124336


namespace rational_root_even_coefficient_l124_124764

theorem rational_root_even_coefficient 
    (a b c : ℤ) (ha : a ≠ 0) (hx : ∃ x : ℚ, a * x^2 + b * x + c = 0) :
    even a ∨ even b ∨ even c :=
by sorry

end rational_root_even_coefficient_l124_124764


namespace prob_all_successful_pairs_l124_124829

theorem prob_all_successful_pairs (n : ℕ) : 
  let num_pairs := (2 * n)!
  let num_successful_pairs := 2^n * n!
  prob_all_successful_pairs := num_successful_pairs / num_pairs 
  prob_all_successful_pairs = (2^n * n!) / (2 * n)! := 
sorry

end prob_all_successful_pairs_l124_124829


namespace evaluate_composition_l124_124345

def f(x : ℝ) : ℝ := 3 * x + 2
def g(x : ℝ) : ℝ := 2 * x - 1

theorem evaluate_composition : f(g(g(3))) = 29 := by
  sorry

end evaluate_composition_l124_124345


namespace total_amount_l124_124334

theorem total_amount (B J D : ℝ) (hB : B = 12.000000000000002) 
  (hJ1 : J = 2 * B) (hJ2 : J = (3 / 4) * D) :
  B + J + D = 68.00000000000001 :=
by
  have hJ : J = 2 * 12.000000000000002 := by rw [hB]; refl
  have hJ_val : J = 24.000000000000004 := by rwa mul_comm
  have hD : D = J / (3 / 4) := by rw [hJ2]
  have hD_val : D = 24.000000000000004 * (4 / 3) := by rw [hJ_val]; norm_num
  sorry

end total_amount_l124_124334


namespace midpoints_form_square_l124_124325

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoints_form_square (A B C D K M X Y : ℝ × ℝ)
  (h_square_ABCD : is_square A B C D)
  (h_square_KMXY : is_square_inside K M X Y A B C D) :
  let 
    A_K := midpoint A K,
    B_M := midpoint B M,
    C_X := midpoint C X,
    D_Y := midpoint D Y
  in is_square A_K B_M C_X D_Y :=
sorry

end midpoints_form_square_l124_124325


namespace final_number_is_even_l124_124499

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem final_number_is_even :
  ∀ l : list ℕ, 
    (∀ k, k ∈ l → 1 ≤ k ∧ k ≤ 2011) → 
    (l.sum % 2 = 0) → 
    (∃ m, m ∈ l ∧ l.length = 1 ∧ is_even m) :=
by sorry

end final_number_is_even_l124_124499


namespace right_triangle_set_l124_124956

theorem right_triangle_set :
  ∃ (a b c : ℝ), (a = 1 ∧ b = 1 ∧ c = Real.sqrt 2) ∧ a^2 + b^2 = c^2 := by
  use 1, 1, Real.sqrt 2
  simp
  left
  trivial

end right_triangle_set_l124_124956


namespace xy_sum_l124_124281

-- Definitions for the conditions in Lean
variables {x y : ℝ}
def a : (ℝ × ℝ × ℝ) := (2, 4, x)
def b : (ℝ × ℝ × ℝ) := (2, y, 2)
def magnitude_a := real.sqrt (2^2 + 4^2 + x^2) = 6
def perpendicular := (2 * 2 + 4 * y + x * 2) = 0

-- Theorem stating the proof goal
theorem xy_sum (magnitude_a : magnitude_a) (perpendicular : perpendicular) : x + y = 1 ∨ x + y = -3 :=
by {
  sorry
}

end xy_sum_l124_124281


namespace no_finite_set_with_properties_l124_124372

theorem no_finite_set_with_properties (N : ℕ) (hN : N > 3) : 
  ∀ (S : finset ℝ^2), |S| > 2 * N →
  (∀ A ⊆ S, |A| = N → ∃ B ⊆ S, |B| = N - 1 ∧ (∑ x in (A ∪ B).to_finset, x) = 0) →
  (∀ A ⊆ S, |A| = N → ∃ B ⊆ S, |B| = N ∧ (∑ x in (A ∪ B).to_finset, x) = 0) →
  false := 
λ S hS h1 h2, sorry

end no_finite_set_with_properties_l124_124372


namespace distance_missouri_to_new_york_by_car_l124_124398

variable (d_flight d_car : ℚ)

theorem distance_missouri_to_new_york_by_car :
  d_car = 1.4 * d_flight → 
  d_car = 1400 → 
  (d_car / 2 = 700) :=
by
  intros h1 h2
  sorry

end distance_missouri_to_new_york_by_car_l124_124398


namespace valentino_farm_total_birds_l124_124751

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end valentino_farm_total_birds_l124_124751


namespace range_of_k_l124_124305

theorem range_of_k :
  ∃ k : ℝ, is_property (λ k : ℝ, ∀ x : ℝ, (k^2 - 1) * x^2 - (k + 1) * x + 1 > 0) (λ k : ℝ, k > -1 ∧ k < 5 / 3) :=
sorry

end range_of_k_l124_124305


namespace max_m_satisfying_inequality_l124_124793

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ (0, 1] then x * (x - 1)
else 2 * f (x - 1)

theorem max_m_satisfying_inequality :
  ∃ m : ℝ, (∀ x : ℝ, x ≤ (7 / 3) → f x ≥ - (8 / 9)) ∧
           (∀ y : ℝ, y > (7 / 3) → ∃ z : ℝ, z ≤ y ∧ f z < - (8 / 9)) :=
sorry

end max_m_satisfying_inequality_l124_124793


namespace monotonic_increasing_interval_l124_124405

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x ^ 2 - 2 * x - 8)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, (4 ≤ x → monotone_on f (Ici 4)) :=
by
  sorry

end monotonic_increasing_interval_l124_124405


namespace positive_integers_exist_poly_satisfying_conditions_l124_124195

theorem positive_integers_exist_poly_satisfying_conditions (n : ℕ) (h_pos : 0 < n)
  (h_exists_ks : ∃ (k : Fin n → ℤ), Function.Injective k)
  (h_exists_P : ∃ P : Polynomial ℤ, 
    (degree P ≤ n ∧ ∀ i : Fin n, P.eval (k i) = n) ∧ ∃ z : ℤ, P.eval z = 0) :
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 :=
sorry

end positive_integers_exist_poly_satisfying_conditions_l124_124195


namespace option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l124_124117

theorem option_a_correct (a : ℝ) : 2 * a^2 - 3 * a^2 = - a^2 :=
by
  sorry

theorem option_b_incorrect : (-3)^2 ≠ 6 :=
by
  sorry

theorem option_c_incorrect (a : ℝ) : 6 * a^3 + 4 * a^4 ≠ 10 * a^7 :=
by
  sorry

theorem option_d_incorrect (a b : ℝ) : 3 * a^2 * b - 3 * b^2 * a ≠ 0 :=
by
  sorry

end option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l124_124117


namespace problem1_problem2_l124_124020

-- Define the set A
def A := {x | x ∈ [1, 2, 3, 4, 5]}

-- Define the set B
def B := {x | (x - 1) * (x - 2) = 0}

-- Define the set C
def C (a : ℝ) := {a, a^2 + 1}

-- Step 1: Prove A ∩ (complement of B in U) = {3, 4, 5} and A ∪ B = {1, 2, 3, 4, 5}
theorem problem1 : 
  A ∩ (A \ B) = {3, 4, 5} ∧ A ∪ B = {1, 2, 3, 4, 5} := sorry

-- Step 2: Prove the value of a given B ⊆ C and C ⊆ B
theorem problem2 (a : ℝ) (h1 : B ⊆ C a) (h2 : C a ⊆ B) : a = 1 :=
by
  sorry

end problem1_problem2_l124_124020


namespace polynomial_expansion_sum_constants_l124_124651

theorem polynomial_expansion_sum_constants :
  ∃ (A B C D : ℤ), ((x - 3) * (4 * x ^ 2 + 2 * x - 7) = A * x ^ 3 + B * x ^ 2 + C * x + D) → A + B + C + D = 2 := 
by
  sorry

end polynomial_expansion_sum_constants_l124_124651


namespace value_of_expression_l124_124657

variables (u v w : ℝ)

theorem value_of_expression (h1 : u = 3 * v) (h2 : w = 5 * u) : 2 * v + u + w = 20 * v :=
by sorry

end value_of_expression_l124_124657


namespace roller_coaster_rides_l124_124084

theorem roller_coaster_rides (total_people cars_per_ride people_per_car : ℕ) (people_waiting : total_people = 84) (cars : cars_per_ride = 7) (seats_per_car : people_per_car = 2) : 
  total_people / (cars_per_ride * people_per_car) = 6 := 
by 
  rw [people_waiting, cars, seats_per_car]
  norm_num
  sorry

end roller_coaster_rides_l124_124084


namespace total_accidents_l124_124708

-- Define the given vehicle counts for the highways
def total_vehicles_A : ℕ := 4 * 10^9
def total_vehicles_B : ℕ := 2 * 10^9
def total_vehicles_C : ℕ := 1 * 10^9

-- Define the accident ratios per highway
def accident_ratio_A : ℕ := 80
def accident_ratio_B : ℕ := 120
def accident_ratio_C : ℕ := 65

-- Define the number of vehicles in millions
def million := 10^6

-- Define the accident calculations per highway
def accidents_A : ℕ := (total_vehicles_A / (100 * million)) * accident_ratio_A
def accidents_B : ℕ := (total_vehicles_B / (200 * million)) * accident_ratio_B
def accidents_C : ℕ := (total_vehicles_C / (50 * million)) * accident_ratio_C

-- Prove the total number of accidents across all highways
theorem total_accidents : accidents_A + accidents_B + accidents_C = 5700 := by
  have : accidents_A = 3200 := by sorry
  have : accidents_B = 1200 := by sorry
  have : accidents_C = 1300 := by sorry
  sorry

end total_accidents_l124_124708


namespace count_zero_vectors_l124_124544

variable {V : Type} [AddCommGroup V]

variables (A B C D M O : V)

def vector_expressions_1 := (A - B) + (B - C) + (C - A) = 0
def vector_expressions_2 := (A - B) + (M - B) + (B - O) + (O - M) ≠ 0
def vector_expressions_3 := (A - B) - (A - C) + (B - D) - (C - D) = 0
def vector_expressions_4 := (O - A) + (O - C) + (B - O) + (C - O) ≠ 0

theorem count_zero_vectors :
  (vector_expressions_1 A B C) ∧
  (vector_expressions_2 A B M O) ∧
  (vector_expressions_3 A B C D) ∧
  (vector_expressions_4 O A C B) →
  (2 = 2) :=
sorry

end count_zero_vectors_l124_124544


namespace probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124846

-- Definition of conditions
def num_pairs_socks : ℕ := n

def successful_pair (socks : List (ℕ × ℕ)) (k : ℕ) : Prop :=
  (socks[n - k]).fst = (socks[n - k]).snd

-- Part (a): Prove the probability that all pairs are successful is \(\frac{2^n n!}{(2n)!}\)
theorem probability_all_successful_pairs (n : ℕ) :
  ∃ p : ℚ, p = (2^n * nat.factorial n) / nat.factorial (2 * n) :=
sorry

-- Part (b): Prove the expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half (n : ℕ) :
  (n / (2 * n - 1)) > 1 / 2 :=
sorry

end probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124846


namespace compute_area_isosceles_trapezoid_l124_124337

noncomputable theory

open Real

-- Conditions 
-- Isosceles trapezoid definition
def is_isosceles_trapezoid (ABCD : Quadrilateral) : Prop :=
  ABCD.parallel AD BC ∧ ABCD.adjacent_size AD = ABCD.adjacent_size BC

-- Assumptions based on problem conditions
axiom PI_8 (ABC : Triangle) (I : Point) (P : Point) : (distance P I) = 8
axiom IJ_25 (I J : Point) : (distance I J) = 25
axiom JQ_15 (ABD : Triangle) (J : Point) (Q : Point) : (distance Q J) = 15
axiom AD_parallel_BC (ABCD : Quadrilateral) : ABCD.parallel AD BC

-- Problem statement
theorem compute_area_isosceles_trapezoid (ABCD : Quadrilateral)
  (PI_8 : (distance P I) = 8)
  (IJ_25 : (distance I J) = 25)
  (JQ_15 : (distance Q J) = 15)
  (AD_parallel_BC : ABCD.parallel AD BC) :
  greatest_integer_area_leq (ABCD.area) <= 1728 :=
sorry

end compute_area_isosceles_trapezoid_l124_124337


namespace solve_for_x_l124_124775

theorem solve_for_x (x : ℝ) : 27^x * 27^x * 27^x = 243^3 → x = 5 / 3 :=
by
  intro h
  sorry

end solve_for_x_l124_124775


namespace diameter_perpendicular_bisects_l124_124168

theorem diameter_perpendicular_bisects (C : Type*) [metric_space C] {O : C} {r : ℝ} 
  (circ : metric.sphere O r)
  (D : ℝ) (chord : set C) (M : C) (diameter : set C) 
  (hD : diameter = {x : C | dist O x = D} ∩ circ)
  (h_chord : ∀ x ∈ chord, ∃ y ∈ chord, dist x y = D ∧ dist y M = dist y M) :
  (∀ x ∈ chord ∩ diameter, dist x M = dist O x / 2) :=
begin
  sorry
end

end diameter_perpendicular_bisects_l124_124168


namespace guard_maximum_demand_l124_124943

-- Definitions based on conditions from part a)
-- A type for coins
def Coins := ℕ

-- Definitions based on question and conditions
def maximum_guard_demand (x : Coins) : Prop :=
  ∀ x, (x ≤ 199 ∧ x - 100 < 100) ∨ (x > 199 → 100 ≤ x - 100)

-- Statement of the problem in Lean 4
theorem guard_maximum_demand : ∃ x : Coins, maximum_guard_demand x ∧ x = 199 :=
by
  sorry

end guard_maximum_demand_l124_124943


namespace bianca_deleted_text_files_l124_124460

theorem bianca_deleted_text_files (pictures songs total : ℕ) (h₁ : pictures = 2) (h₂ : songs = 8) (h₃ : total = 17) :
  total - (pictures + songs) = 7 :=
by {
  sorry
}

end bianca_deleted_text_files_l124_124460


namespace find_n_l124_124030

theorem find_n :
  ∃ n : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = n ^ 5 ∧ 
  (∀ m : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = m ^ 5 → m = 144) :=
by
  sorry

end find_n_l124_124030


namespace spherical_to_rectangular_conversion_l124_124990

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 10 (3 * Real.pi / 4) (Real.pi / 4) = (-5, 5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l124_124990


namespace coffee_cups_per_week_l124_124492

theorem coffee_cups_per_week 
  (cups_per_hour_weekday : ℕ)
  (total_weekend_cups : ℕ)
  (hours_per_day : ℕ)
  (days_in_week : ℕ)
  (weekdays : ℕ)
  : cups_per_hour_weekday * hours_per_day * weekdays + total_weekend_cups = 370 :=
by
  -- Assume values based on given conditions
  have h1 : cups_per_hour_weekday = 10 := sorry
  have h2 : total_weekend_cups = 120 := sorry
  have h3 : hours_per_day = 5 := sorry
  have h4 : weekdays = 5 := sorry
  have h5 : days_in_week = 7 := sorry
  
  -- Calculate total weekday production
  have weekday_production : ℕ := cups_per_hour_weekday * hours_per_day * weekdays
  
  -- Substitute and calculate total weekly production
  have total_production : ℕ := weekday_production + total_weekend_cups
  
  -- The total production should be equal to 370
  show total_production = 370 from by
    rw [weekday_production, h1, h3, h4]
    exact sorry

  sorry

end coffee_cups_per_week_l124_124492


namespace probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124845

-- Definition of conditions
def num_pairs_socks : ℕ := n

def successful_pair (socks : List (ℕ × ℕ)) (k : ℕ) : Prop :=
  (socks[n - k]).fst = (socks[n - k]).snd

-- Part (a): Prove the probability that all pairs are successful is \(\frac{2^n n!}{(2n)!}\)
theorem probability_all_successful_pairs (n : ℕ) :
  ∃ p : ℚ, p = (2^n * nat.factorial n) / nat.factorial (2 * n) :=
sorry

-- Part (b): Prove the expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half (n : ℕ) :
  (n / (2 * n - 1)) > 1 / 2 :=
sorry

end probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124845


namespace room_dimension_l124_124061

theorem room_dimension
  (x : ℕ)
  (cost_per_sqft : ℕ := 4)
  (dimension_1 : ℕ := 15)
  (dimension_2 : ℕ := 12)
  (door_width : ℕ := 6)
  (door_height : ℕ := 3)
  (num_windows : ℕ := 3)
  (window_width : ℕ := 4)
  (window_height : ℕ := 3)
  (total_cost : ℕ := 3624) :
  (2 * (x * dimension_1) + 2 * (x * dimension_2) - (door_width * door_height + num_windows * (window_width * window_height))) * cost_per_sqft = total_cost →
  x = 18 :=
by
  sorry

end room_dimension_l124_124061


namespace diametrically_opposite_to_83_l124_124904

variable (n : ℕ) (k : ℕ)
variable (h1 : n = 100)
variable (h2 : 1 ≤ k ∧ k ≤ n)
variable (h_dist : ∀ k, k < 83 → (number_of_points_less_than_k_on_left + number_of_points_less_than_k_on_right = k - 1) ∧ number_of_points_less_than_k_on_left = number_of_points_less_than_k_on_right)

-- define what's "diametrically opposite" means
def opposite_number (n k : ℕ) : ℕ := 
  if (k + n/2) ≤ n then k + n/2 else (k + n/2) - n

-- the exact statement to demonstrate the problem above
theorem diametrically_opposite_to_83 (h1 : n = 100) (h2 : 1 ≤ k ∧ k ≤ n) (h3 : k = 83) : 
  opposite_number n k = 84 :=
by
  have h_opposite : opposite_number n 83 = 84 := sorry
  exact h_opposite


end diametrically_opposite_to_83_l124_124904


namespace min_tan4_sec4_l124_124215

theorem min_tan4_sec4 (x : ℝ) : (tan x ^ 4 + sec x ^ 4) ≥ 1 := by
  have h1 : sec x ^ 2 = 1 + tan x ^ 2 := by sorry
  sorry

end min_tan4_sec4_l124_124215


namespace last_round_total_matches_l124_124961

variable (R1 R2 L T : ℕ)

-- Definitions based on conditions
def first_two_rounds_matches_won := (R1 = 6) ∧ (R2 = 6) -- 6 matches each in the first two rounds
def total_matches_won := T = 14 -- Brendan won 14 matches in total
def last_round_matches_won := L / 2 -- Brendan won half of the matches in the last round
def total_matches_formula := R1 + R2 + (L / 2) = T -- Total match calculation

theorem last_round_total_matches (R1 R2 L T : ℕ) :
  first_two_rounds_matches_won R1 R2 ∧ total_matches_won T ∧ last_round_matches_won L →
  total_matches_formula R1 R2 L T →
  L = 4 :=
by
  intros _ _
  sorry

end last_round_total_matches_l124_124961


namespace lunch_cost_before_tip_l124_124430

theorem lunch_cost_before_tip (tip_rate : ℝ) (total_spent : ℝ) (C : ℝ) : 
  tip_rate = 0.20 ∧ total_spent = 72.96 ∧ C + tip_rate * C = total_spent → C = 60.80 :=
by
  intro h
  sorry

end lunch_cost_before_tip_l124_124430


namespace parallel_lines_condition_l124_124406

-- We define the conditions as Lean definitions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0
def parallel_condition (a : ℝ) : Prop := (a ≠ 0) ∧ (a ≠ 1) ∧ (a ≠ -1) ∧ (a * (a^2 - 1) ≠ 6)

-- Mathematically equivalent Lean 4 statement
theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y → line2 a x y → (line1 a x y ↔ line2 a x y)) ↔ (a = -1) :=
by 
  -- The full proof would be written here
  sorry

end parallel_lines_condition_l124_124406


namespace triangle_PQR_properties_l124_124323

-- We'll define the setup and the conditions, then state the main theorem.

def midpoint (P R M : Point) : Prop :=
  dist P M = dist M R ∧ dist P R = 2 * dist P M

def isosceles (P Q M : Point) (d : ℝ) : Prop :=
  dist P Q = d ∧ dist P M = d

def distance (P Q : Point) (d : ℝ) : Prop :=
  dist P Q = d

theorem triangle_PQR_properties
  (P Q R M : Point)
  (d : ℝ)
  (h1 : midpoint P R M)
  (h2 : line_segment RQ)
  (h3 : on_line_segment Q PM)
  (h4 : isosceles P Q M d)
  (h5 : distance P R 4):
  dist P Q ^ 2 = 4 :=
sorry

end triangle_PQR_properties_l124_124323


namespace sum_of_numbers_l124_124666

theorem sum_of_numbers (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 :=
by
  sorry

end sum_of_numbers_l124_124666


namespace slices_with_both_ingredients_l124_124132

theorem slices_with_both_ingredients : 
    ∀ (total_slices pepperoni_slices mushroom_slices vegetarian_slices : ℕ), 
    total_slices = 24 → 
    pepperoni_slices = 12 → 
    mushroom_slices = 14 → 
    vegetarian_slices = 4 → 
    pepperoni_slices - (pepperoni_slices - (total_slices - vegetarian_slices - mushroom_slices)) = 6 :=
by
  intros total_slices pepperoni_slices mushroom_slices vegetarian_slices
  intros h_total h_pepperoni h_mushroom h_vegetarian
  rw [h_total, h_pepperoni, h_mushroom, h_vegetarian]
  sorry

end slices_with_both_ingredients_l124_124132


namespace cone_base_circumference_l124_124158

theorem cone_base_circumference (r : ℝ) (h_r : r = 6) (sector_deg : ℝ) (h_sector_deg : sector_deg = 180) :
  ∃ (C : ℝ), C = 6 * Real.pi :=
by
  have h1 : 2 * Real.pi * r = 12 * Real.pi, from sorry
  have h2 : sector_deg / 360 * 12 * Real.pi = 6 * Real.pi, from sorry
  use 6 * Real.pi
  exact sorry

end cone_base_circumference_l124_124158


namespace maximize_profit_maximize_tax_revenue_l124_124680

open Real

-- Conditions
def inverse_demand_function (P Q : ℝ) : Prop :=
  P = 310 - 3 * Q

def production_cost_per_jar : ℝ := 10

-- Part (a): Prove that Q = 50 maximizes profit
theorem maximize_profit : (Q : ℝ) (P : ℝ) (∏ : ℝ -> ℝ) (C : ℝ -> ℝ) 
  (h_inv_demand : inverse_demand_function P Q)
  (h_revenue : (R : ℝ -> ℝ), R Q = P * Q)
  (h_cost : (C : ℝ -> ℝ), C Q = production_cost_per_jar * Q)
  (h_profit : (∏ : ℝ -> ℝ), ∏ Q = R Q - C Q)
  (h_derive : ∀ Q, deriv ∏ Q = 300 - 6 * Q) :
  Q = 50 := sorry

-- Part (b): Prove that t = 150 maximizes tax revenue for the government
theorem maximize_tax_revenue : (Q t : ℝ) (T : ℝ -> ℝ)
  (h_inv_demand : inverse_demand_function (310 - 3 * Q) Q)
  (h_tax_revenue : (T : ℝ -> ℝ), T t = Q * t)
  (h_tax_revenue_simplified : (T : ℝ -> ℝ), T t = (300 * t - t ^ 2) / 6)
  (h_derive_tax_revenue : ∀ t, deriv T t = 50 - t / 3) :
  t = 150 := sorry

end maximize_profit_maximize_tax_revenue_l124_124680


namespace number_of_true_statements_is_zero_l124_124518

-- Define the lines
variables {a b c : Type}

-- Define the propositions as functions that return Prop
def proposition_1 (a b c : Type) [IsParallel a b] [¬IsCoplanar a c] : Prop :=
  ¬IsCoplanar b c

def proposition_2 (a b c : Type) [IsCoplanar a b] [¬IsCoplanar b c] : Prop :=
  ¬IsCoplanar a c

def proposition_3 (a b c : Type) [¬IsCoplanar a b] [IsCoplanar a c] : Prop :=
  ¬IsCoplanar b c

def proposition_4 (a b c : Type) [¬IsCoplanar a b] [DoesNotIntersect b c] : Prop :=
  DoesNotIntersect a c

-- Check that all propositions are false under any given conditions
theorem number_of_true_statements_is_zero : 
  ∀ (a b c : Type), 
  ¬proposition_1 a b c → ¬proposition_2 a b c → ¬proposition_3 a b c → ¬proposition_4 a b c → 
  (0 = 0) :=
by
  intros
  exact rfl

end number_of_true_statements_is_zero_l124_124518


namespace today_is_wednesday_l124_124858

def days_of_week : Type := {d : Fin 7 // 0 ≤ d.val ∧ d.val ≤ 6}

def day_after (d : days_of_week) : days_of_week :=
  ⟨(d.val + 1) % 7, sorry⟩

def day_after_tomorrow (d : days_of_week) : days_of_week :=
  day_after (day_after d)

def day_yesterday (d : days_of_week) : days_of_week :=
  ⟨(d.val + 6) % 7, sorry⟩

def day_after_tomorrow_becomes_yesterday (today : days_of_week) : Prop :=
  day_yesterday (day_after_tomorrow today) = day_yesterday day_yesterday

def distance_from_sunday (d : days_of_week) : Nat :=
  (d.val + 6) % 7

def condition_met (today : days_of_week) : Prop :=
  distance_from_sunday today = distance_from_sunday (day_yesterday (day_after today))

theorem today_is_wednesday : ∃ (today : days_of_week), day_after_tomorrow_becomes_yesterday today ∧ condition_met today ∧ today.val = 3 :=
begin
  sorry
end

end today_is_wednesday_l124_124858


namespace son_complete_work_alone_in_6_55_days_l124_124946

noncomputable def man_work_rate : ℝ := 1 / 6  -- Usual daily work rate of the man

def variable_work_rate_day1 : ℝ := 1.2 * man_work_rate  -- Work rate on Day 1
def variable_work_rate_day2 : ℝ := variable_work_rate_day1 * (1 - 0.1)  -- Work rate on Day 2
def variable_work_rate_day3 : ℝ := variable_work_rate_day2 * (1 - 0.1)  -- Work rate on Day 3

-- Total work done by the man in 3 days
def total_man_work_3_days : ℝ := variable_work_rate_day1 + variable_work_rate_day2 + variable_work_rate_day3

def total_work : ℝ := 1  -- Total work is 1 (or 100% of the work)

def son_work_rate (total_work : ℝ) (total_man_work_3_days : ℝ) : ℝ :=
  (total_work - total_man_work_3_days) / 3  -- Son's daily work rate

def days_to_complete_work_alone (son_work_rate : ℝ) : ℝ :=
  total_work / son_work_rate  -- Time for the son to complete the work alone

theorem son_complete_work_alone_in_6_55_days :
  days_to_complete_work_alone (son_work_rate total_work total_man_work_3_days) ≈ 6.55 :=
sorry

end son_complete_work_alone_in_6_55_days_l124_124946


namespace max_elements_non_divisible_l124_124116

theorem max_elements_non_divisible : ∃ (S : set ℕ), S ⊆ {n : ℕ | 1 ≤ n ∧ n ≤ 100} ∧ (∀ a b ∈ S, a ≠ b → ¬ (a ∣ b ∨ b ∣ a)) ∧ S.card = 50 :=
by {
  sorry
}

end max_elements_non_divisible_l124_124116


namespace angle_DCE_eq_45_l124_124350

-- Define the problem in Lean 4
noncomputable def semicircle := sorry -- place holder type definition
axiom diameter (h : semicircle) : Point × Point
variable (A B : Point)
axiom (A, B) = diameter h
axiom P : Point
axiom P_in_AB : P ∈ segment A B
axiom C : Point
axiom normal_intersect := segment A P ⟹ C ∈ h
axiom circle_1 : Circle := sorry -- first inscribed circle
axiom circle_2 : Circle := sorry -- second inscribed circle
axiom D E : Point
axiom touches_D := D ∈ segment A P ∧ tangent_to circle_1 A P D
axiom touches_E := E ∈ segment P B ∧ tangent_to circle_2 P B E
axiom D_between_A_P : A < D < P -- This assumes a suitable order relation on points on AB.

-- Statement of the theorem
theorem angle_DCE_eq_45 (h : semicircle) (A B P C D E : Point) 
  (P_in_AB : P ∈ segment A B) (normal_intersect : segment A P ⟹ (C ∈ h)) 
  (D_between_A_P : A < D < P) 
  : ∠DCE = 45 := 
sorry

end angle_DCE_eq_45_l124_124350


namespace find_number_l124_124966

theorem find_number (x : Real) (h1 : (2 / 5) * 300 = 120) (h2 : 120 - (3 / 5) * x = 45) : x = 125 :=
by
  sorry

end find_number_l124_124966


namespace four_x_plus_t_odd_l124_124463

theorem four_x_plus_t_odd (x t : ℤ) (hx : 2 * x - t = 11) : ¬(∃ n : ℤ, 4 * x + t = 2 * n) :=
by
  -- Since we need to prove the statement, we start a proof block
  sorry -- skipping the actual proof part for this statement

end four_x_plus_t_odd_l124_124463


namespace area_of_triangle_XYZ_is_correct_l124_124432

-- Definitions related to the problem conditions
def square_area : ℝ := 1 

def ratio_KX_XL : ℝ := 3 / 2
def ratio_KY_YN : ℝ := 4 / 1
def ratio_NZ_ZM : ℝ := 2 / 3

-- Points on the square
def X : ℝ × ℝ := (3 / 5, 1)
def Y : ℝ × ℝ := (0, 1 / 5)
def Z : ℝ × ℝ := (2 / 5, 0)

-- Function to compute the area using the Shoelace Theorem
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(fst A * snd B + fst B * snd C + fst C * snd A) - (snd A * fst B + snd B * fst C + snd C * fst A)|

-- Statement to prove
theorem area_of_triangle_XYZ_is_correct :
  triangle_area X Y Z = 11 / 50 := by
  -- Definitions and conditions already given
  sorry

end area_of_triangle_XYZ_is_correct_l124_124432


namespace max_distance_PQ_l124_124257

noncomputable def point := ℝ × ℝ

def line_l1 (m n : ℝ) : set point := {p | m * p.1 - n * p.2 - 5 * m + n = 0}
def line_l2 (m n : ℝ) : set point := {p | n * p.1 + m * p.2 - 5 * m - n = 0}

def intersection_point (m n : ℝ) : point :=
if h : m ^ 2 + n ^ 2 ≠ 0 then
  classical.some (exists_point_of_lines m n h)
else (0, 0) -- default value when m^2 + n^2 = 0, may not happen per condition

def circle_C : set point := {p | (p.1 + 1) ^ 2 + p.2 ^ 2 = 1}

def |PQ| (P Q : point) := real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem max_distance_PQ (m n : ℝ) (h : m^2 + n^2 ≠ 0) (Q : point) (Q_on_circle : Q ∈ circle_C) :
  let P := intersection_point m n in
  (|PQ| P Q) ≤ 6 + 2 * real.sqrt 2 := sorry

-- Helper lemma to find a point on the given two lines
lemma exists_point_of_lines (m n : ℝ) (h : m^2 + n^2 ≠ 0) : ∃ p : point, p ∈ line_l1 m n ∧ p ∈ line_l2 m n :=
sorry

end max_distance_PQ_l124_124257


namespace min_n_constant_term_exists_l124_124724

theorem min_n_constant_term_exists (n : ℕ) (h : 0 < n) :
  (∃ r : ℕ, (2 * n = 3 * r) ∧ n > 0) ↔ n = 3 :=
by
  sorry

end min_n_constant_term_exists_l124_124724


namespace problem_l124_124210

noncomputable def t : ℝ := Real.cbrt 4

theorem problem (t : ℝ) : 4 * log 3 t = log 3 (4 * t) → t = Real.cbrt 4 :=
by
  intro h
  have htlog4 := eq_of_mul_log_four_eq_log_four_times h
  have ht : t^3 = 4
  from htlog4
  exact eq_cbrt_of_pow3_eq_four ht
end

end problem_l124_124210


namespace smallest_n_satisfies_f_l124_124217

def f (n : ℕ) : ℕ :=
  (n / 2)
  + (n / 23)
  - 2 * (n / 46)

theorem smallest_n_satisfies_f :
  (∃ (n : ℕ), 0 < n ∧ f(n) = 2323) ∧
  (∀ m : ℕ, 0 < m ∧ f(m) = 2323 → 4644 ≤ m) :=
by
  sorry

end smallest_n_satisfies_f_l124_124217


namespace exist_xy_cosine_inequality_l124_124042

theorem exist_xy_cosine_inequality (a b c d : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2) 
  (hc : 0 < c ∧ c < π / 2) (hd : 0 < d ∧ d < π / 2) : 
  ∃ x y ∈ {a, b, c, d}, 8 * Real.cos x * Real.cos y * Real.cos (x - y) + 1 > 4 * (Real.cos x ^ 2 + Real.cos y ^ 2) :=
by
  sorry

end exist_xy_cosine_inequality_l124_124042


namespace price_reduction_l124_124951

theorem price_reduction :
  ∀ (P : ℝ), P > 0 →
  let first_day_price := P * (1 - 0.09) in
  let second_day_price := first_day_price * (1 - 0.10) in
  second_day_price = P * 0.819 :=
begin
  assume (P : ℝ) (P_pos : P > 0),
  let first_day_price := P * (1 - 0.09),
  let second_day_price := first_day_price * (1 - 0.10),
  calc
    second_day_price = P * 0.91 * 0.90   : by rw [←mul_assoc, ←(show 0.91 = 1 - 0.09, by norm_num), ←(show 0.90 = 1 - 0.10, by norm_num)]
                  ... = P * 0.819       : by norm_num,
end

end price_reduction_l124_124951


namespace books_ratio_l124_124812

theorem books_ratio (c e : ℕ) (h_ratio : c / e = 2 / 5) (h_sampled : c = 10) : e = 25 :=
by
  sorry

end books_ratio_l124_124812


namespace opposite_of_83_is_84_l124_124924

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l124_124924


namespace age_ratio_l124_124361

noncomputable def ratio_of_ages (Nick_age : ℕ) (age_difference : ℕ) (years_ahead : ℕ) (Brother_age_future : ℕ) : ℚ :=
  let Sister_age := Nick_age + age_difference
  let Combined_age := Nick_age + Sister_age
  let Brother_current_age := Brother_age_future - years_ahead
  (Brother_current_age : ℚ) / (Combined_age : ℚ)

theorem age_ratio (Nick_age age_difference years_ahead Brother_age_future : ℕ) (h1 : Nick_age = 13) (h2 : age_difference = 6) (h3 : years_ahead = 5) (h4 : Brother_age_future = 21) :
  ratio_of_ages Nick_age age_difference years_ahead Brother_age_future = (1 : ℚ) / 2 :=
by
  rw [h1, h2, h3, h4]
  unfold ratio_of_ages
  norm_num
  sorry

end age_ratio_l124_124361


namespace distance_is_correct_l124_124412

noncomputable def distance_between_intersections : ℝ :=
  let parabola := { p : ℝ × ℝ | p.2 ^ 2 = 12 * p.1 }
  let circle := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 - 6 * p.2 = 0 }
  let intersections := { p : ℝ × ℝ | p ∈ parabola ∧ p ∈ circle }
  if h : intersections.to_finset = {[0, 0], [3, 6]} then
    dist (0,0) (3,6)
  else
    0

theorem distance_is_correct : distance_between_intersections = 3 * real.sqrt 5 := 
  sorry

end distance_is_correct_l124_124412


namespace diametrically_opposite_number_l124_124918

theorem diametrically_opposite_number (n k : ℕ) (h1 : n = 100) (h2 : k = 83) 
  (h3 : ∀ k ∈ finset.range n, 
        let m := (k - 1) / 2 in 
        let points_lt_k := finset.range k in 
        points_lt_k.card = 2 * m) : 
  True → 
  (k + 1 = 84) :=
by
  sorry

end diametrically_opposite_number_l124_124918


namespace CarlosBooksInJune_l124_124976

def CarlosBooksInJuly := 28
def CarlosBooksInAugust := 30
def CarlosTotalBooksGoal := 100

theorem CarlosBooksInJune :
  ∃ x : ℕ, x = CarlosTotalBooksGoal - (CarlosBooksInJuly + CarlosBooksInAugust) :=
begin
  use 42,
  dsimp [CarlosTotalBooksGoal, CarlosBooksInJuly, CarlosBooksInAugust],
  norm_num,
  sorry
end

end CarlosBooksInJune_l124_124976


namespace mode_of_data_l124_124675

noncomputable def mode (s : Multiset ℕ) : ℕ :=
s.filter (λ n, s.count n = s.count (s.mode)).head

theorem mode_of_data : mode {1, 2, 3, 4, 3, 5} = 3 := 
by
  -- Here normally we would prove the statement step by step
  sorry

end mode_of_data_l124_124675


namespace feet_of_wood_required_l124_124700

def rung_length_in_inches : ℤ := 18
def spacing_between_rungs_in_inches : ℤ := 6
def height_to_climb_in_feet : ℤ := 50

def feet_per_rung := rung_length_in_inches / 12
def rungs_per_foot := 12 / spacing_between_rungs_in_inches
def total_rungs := height_to_climb_in_feet * rungs_per_foot
def total_feet_of_wood := total_rungs * feet_per_rung

theorem feet_of_wood_required :
  total_feet_of_wood = 150 :=
by
  sorry

end feet_of_wood_required_l124_124700


namespace complement_intersection_l124_124638

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def A := {1, 2, 3}
def B := {2, 3, 4, 5}

theorem complement_intersection :
  U \ (A ∩ B) = {1, 4, 5, 6, 7, 8} :=
by {
  sorry
}

end complement_intersection_l124_124638


namespace problem1_problem2_problem3_problem4_l124_124530

-- Problem 1
theorem problem1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) : m^4 * (m^2)^3 / m^8 = m^2 := 
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (-2 * x - 1) * (2 * x - 1) = 1 - 4 * x^2 := 
by
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : (-3 * x + 2)^2 = 9 * x^2 - 12 * x + 4 := 
by
  sorry

end problem1_problem2_problem3_problem4_l124_124530


namespace probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124842

-- Definition of conditions
def num_pairs_socks : ℕ := n

def successful_pair (socks : List (ℕ × ℕ)) (k : ℕ) : Prop :=
  (socks[n - k]).fst = (socks[n - k]).snd

-- Part (a): Prove the probability that all pairs are successful is \(\frac{2^n n!}{(2n)!}\)
theorem probability_all_successful_pairs (n : ℕ) :
  ∃ p : ℚ, p = (2^n * nat.factorial n) / nat.factorial (2 * n) :=
sorry

-- Part (b): Prove the expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half (n : ℕ) :
  (n / (2 * n - 1)) > 1 / 2 :=
sorry

end probability_all_successful_pairs_expected_successful_pairs_gt_half_l124_124842


namespace car_speed_l124_124868

theorem car_speed (v : ℝ) (h : (1 / v) = (1 / 100 + 2 / 3600)) : v = 3600 / 38 := 
by
  sorry

end car_speed_l124_124868


namespace probability_X_equals_5_l124_124883

noncomputable def bag_probability_event : ℚ :=
  let red_ball_probability := 1 / 3 in
  let white_ball_probability := 2 / 3 in
  let number_of_ways := Nat.choose 4 2 in
  number_of_ways * (red_ball_probability ^ 2) * (white_ball_probability ^ 2) * red_ball_probability

theorem probability_X_equals_5 :
  bag_probability_event = 8 / 81 := by
  sorry

end probability_X_equals_5_l124_124883


namespace lateral_surface_area_pyramid_l124_124413

theorem lateral_surface_area_pyramid (α R : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < π) :
    let S := 4 * R^2 * (Real.cot (α / 2)) * (Real.tan (π / 4 + α / 2))
    S = 4 * R^2 * (Real.cot (α / 2)) * (Real.tan (π / 4 + α / 2)) :=
by
    sorry

end lateral_surface_area_pyramid_l124_124413


namespace min_a_l124_124234

def line1 (b a : ℝ) : ℝ × ℝ → Prop :=
λ (x y : ℝ), (b^2 + 1) * x + a * y + 2 = 0

def line2 (b : ℝ) : ℝ × ℝ → Prop :=
λ (x y : ℝ), x - (b - 1) * y - 1 = 0

def perpendicular (m1 m2 : ℝ) : Prop :=
m1 * m2 = -1

noncomputable def slope1 (b a : ℝ) : ℝ :=
- (b^2 + 1) / a

noncomputable def slope2 (b : ℝ) : ℝ :=
b - 1

theorem min_a (b : ℝ) (hb : 1 < b) : 
  perpendicular (slope1 b a) (slope2 b) → a = 2 * Real.sqrt 2 + 2 :=
sorry

end min_a_l124_124234


namespace kyle_money_left_l124_124706

-- Define variables and conditions
variables (d k : ℕ)
variables (has_kyle : k = 3 * d - 12) (has_dave : d = 46)

-- State the theorem to prove 
theorem kyle_money_left (d k : ℕ) (has_kyle : k = 3 * d - 12) (has_dave : d = 46) :
  k - k / 3 = 84 :=
by
  -- Sorry to complete the proof block
  sorry

end kyle_money_left_l124_124706


namespace fraction_identity_l124_124408

variable (x y z : ℝ)

theorem fraction_identity (h : (x / (y + z)) + (y / (z + x)) + (z / (x + y)) = 1) :
  (x^2 / (y + z)) + (y^2 / (z + x)) + (z^2 / (x + y)) = 0 :=
  sorry

end fraction_identity_l124_124408


namespace correct_description_of_a_l124_124625

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)

theorem correct_description_of_a {a b : ℝ} (h : ∃ (x1 x2 : ℝ), x1 ∈ set.Icc a b ∧ x2 ∈ set.Icc a b ∧ x1 < x2 ∧ f x1 ≥ f x2) : a < 1 := 
sorry

end correct_description_of_a_l124_124625


namespace part_a_part_b_l124_124800

variable (R : ℝ) -- Declare R as a real number representing the radius

theorem part_a (AP BP CP DP : ℝ) (AP_squared : AP ^ 2) (BP_squared : BP ^ 2) (CP_squared : CP ^ 2) (DP_squared : DP ^ 2) 
(h : AP_squared + BP_squared + CP_squared + DP_squared = 4 * R ^ 2) :
    AP^2 + BP^2 + CP^2 + DP^2 = 4 * R^2 :=
by sorry

theorem part_b (AB BC CD DA : ℝ) 
(h1 : AB ^ 2 + CD ^ 2 = 4 * R ^ 2) 
(h2 : BC ^ 2 + DA ^ 2 = 4 * R ^ 2) :
    AB^2 + BC^2 + CD^2 + DA^2 = 8 * R^2 :=
by sorry

end part_a_part_b_l124_124800


namespace unique_points_M_and_N_l124_124348

theorem unique_points_M_and_N (ABC : Triangle) (h_scalene : ABC.Scalene) (I : Point) (h_incenter : is_incenter I ABC) :
  ∃! (M N : Point), (M ∈ BC ∧ N ∈ CA) ∧ (∠A I M = 90 ∧ ∠B I N = 90) :=
sorry

end unique_points_M_and_N_l124_124348


namespace unique_polynomial_function_exists_l124_124647

theorem unique_polynomial_function_exists :
  ∃! (f : polynomial ℂ), (degree f ≥ 1) ∧ (∀ x, f(x^2) = (f(x))^3) ∧ (∀ x, f(f(x)) = f(x)) :=
sorry

end unique_polynomial_function_exists_l124_124647


namespace sum_of_positive_factors_of_120_l124_124115

theorem sum_of_positive_factors_of_120 : 
  let n := 2^3 * 3 * 5 in
  let sum_of_factors := (1 + 2 + 2^2 + 2^3) * (1 + 3) * (1 + 5) in
  sum_of_factors = 360 :=
by
  let n := 2^3 * 3 * 5
  let sum_of_factors := (1 + 2 + 2^2 + 2^3) * (1 + 3) * (1 + 5)
  have h1 : n = 120 := rfl
  have h2 : sum_of_factors = 360 := rfl
  rw [h1, h2]
  exact rfl

end sum_of_positive_factors_of_120_l124_124115


namespace ratio_books_to_pens_l124_124078

-- Define the given ratios and known constants.
def ratio_pencils : ℕ := 14
def ratio_pens : ℕ := 4
def ratio_books : ℕ := 3
def actual_pencils : ℕ := 140

-- Assume the actual number of pens can be calculated from ratio.
def actual_pens : ℕ := (actual_pencils / ratio_pencils) * ratio_pens

-- Prove that the ratio of exercise books to pens is as expected.
theorem ratio_books_to_pens (h1 : actual_pencils = 140) 
                            (h2 : actual_pens = 40) : 
  ((actual_pencils / ratio_pencils) * ratio_books) / actual_pens = 3 / 4 :=
by
  -- The following proof steps are omitted as per instruction
  sorry

end ratio_books_to_pens_l124_124078


namespace inequality_g_tan_ui_l124_124252

variable {t : ℝ} {u₁ u₂ u₃ : ℝ}

noncomputable def f (x t : ℝ) := (2 * x - t) / (x ^ 2 + 1)

def g (t : ℝ) := ((2 * t^2 + 5) / (2 * (t^2 + 1)) : ℝ)

theorem inequality_g_tan_ui 
  (hα : 4 * α^2 - 4 * t * α - 1 = 0)
  (hβ : 4 * β^2 - 4 * t * β - 1 = 0)
  (h_distinct : α ≠ β)
  (h_sum : α + β = t)
  (h_product : α * β = -1 / 4)
  (h_sum_sin : sin u₁ + sin u₂ + sin u₃ = 1)
  (h_cos_u₁_pos : cos u₁ > 0)
  (h_cos_u₂_pos : cos u₂ > 0)
  (h_cos_u₃_pos : cos u₃ > 0)
  (h_ui_interval : ∀ i ∈ [u₁, u₂, u₃], 0 < i ∧ i < π / 2) :
  (1 / g (tan u₁) + 1 / g (tan u₂) + 1 / g (tan u₃)) < (3 / 4) * sqrt 6 := by 
    sorry

end inequality_g_tan_ui_l124_124252


namespace ratio_CL_AB_l124_124242

theorem ratio_CL_AB (ABCDE : set Point) (K L : Point)
  (h_pentagon : regular_pentagon ABCDE)
  (h_K_on_AE : K ∈ segment AE)
  (h_L_on_CD : L ∈ segment CD)
  (h_angles : ∠LAE + ∠KCD = 108)
  (h_ratio_AK_KE : AK / KE = 3 / 7) :
  CL / AB = 0.7 :=
sorry

end ratio_CL_AB_l124_124242


namespace ball_drawing_probability_l124_124136

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def arrangements (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem ball_drawing_probability :
  let successful_patterns := 6 in
  let total_arrangements := arrangements 9 (arrangements 3 3) in
  total_arrangements = 1680 →
  successful_patterns / total_arrangements = 1 / 280 := by
  intros successful_patterns total_arrangements h_total
  sorry

end ball_drawing_probability_l124_124136


namespace problem_solution_l124_124285

theorem problem_solution (x : ℝ) (h : x^2 - 8*x - 3 = 0) : (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 :=
by sorry

end problem_solution_l124_124285


namespace range_and_period_f_perimeter_triangle_ABC_l124_124642

noncomputable def f (x : ℝ) : ℝ :=
  let m := (3 / 2, -Real.sin x)
  let n := (1, Real.sin x + Real.sqrt 3 * Real.cos x)
  m.1 * n.1 + m.2 * n.2

-- Proving the range and period of f(x)
theorem range_and_period_f :
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 2) ∧ (∀ x : ℝ, f (x + π) = f x) :=
  sorry

-- Given conditions for triangle ABC
variables {A B C : ℝ} {a b c : ℝ}
variable hA : f A = 0
variable ha : a = Real.sqrt 3
variable hbc : b * c = 2

-- Proving the perimeter of triangle ABC
theorem perimeter_triangle_ABC :
  a + b + c = 3 + Real.sqrt 3 :=
  sorry

end range_and_period_f_perimeter_triangle_ABC_l124_124642


namespace range_of_f_l124_124194

noncomputable def f (x : ℝ) : ℝ := arctan x + arctan ((x - 2) / (x + 2))

theorem range_of_f : ∃ y ∈ {-real.pi / 4, arctan 2}, ∀ x : ℝ, f x = y :=
by
  sorry

end range_of_f_l124_124194


namespace least_possible_value_expression_l124_124689

theorem least_possible_value_expression 
  (a b c : ℕ) 
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : b ≠ c)
  (h4 : a ∈ {2, 3, 5})
  (h5 : b ∈ {2, 3, 5})
  (h6 : c ∈ {2, 3, 5}) : 
  (a + b) / c / 2 = (1 / 2) :=
by
  -- proof steps to be filled in
  sorry

end least_possible_value_expression_l124_124689


namespace opposite_point_83_is_84_l124_124901

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l124_124901


namespace polynomial_expansion_sum_constants_l124_124650

theorem polynomial_expansion_sum_constants :
  ∃ (A B C D : ℤ), ((x - 3) * (4 * x ^ 2 + 2 * x - 7) = A * x ^ 3 + B * x ^ 2 + C * x + D) → A + B + C + D = 2 := 
by
  sorry

end polynomial_expansion_sum_constants_l124_124650


namespace unique_real_root_l124_124371

theorem unique_real_root (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π / 2) :
  ∃! (x : ℝ), x^3 + x^2 * Real.cos α + x * Real.sin α + 1 = 0 :=
begin
  sorry
end

end unique_real_root_l124_124371


namespace phi_sufficient_but_not_necessary_l124_124472

theorem phi_sufficient_but_not_necessary (ϕ : ℝ) 
  (h : ϕ = π / 2) : 
  ∀ x, sin (x + ϕ) = cos x ∧ (∀ k : ℤ, ϕ ≠ k * π + π / 2) :=
by
  sorry

end phi_sufficient_but_not_necessary_l124_124472


namespace hyperbola_equation_l124_124258

-- Define the conditions
def center := (0, 0)
def focus1 := (-Real.sqrt 5, 0)
def pointP := (Real.sqrt 5, 4)
def midpoint_P_F1 := (0, 2)

-- Define the final equation we need to prove
theorem hyperbola_equation :
  (∃ a b : ℝ, a^2 = 3 ∧ b^2 = 2 ∧ ∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1) ↔ (x^2 - (y^2 / 4) = 1)) := sorry

end hyperbola_equation_l124_124258


namespace max_PA2_PB2_l124_124632

noncomputable def parametric_curve_C1 (t : ℝ) : ℝ × ℝ :=
  (2 - t, real.sqrt 3 * t)

def polar_coordinate (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let θ := real.arctan2 y x
  (r, θ)

def polar_curve_C2 (θ : ℝ) : ℝ :=
  6 / real.sqrt (9 + 3 * real.sin θ^2)

def point_A := parametric_curve_C1 (-1)
def point_B := (-fst point_A, -snd point_A)
def polar_A := polar_coordinate (fst point_A) (snd point_A)
def polar_B := polar_coordinate (fst point_B) (snd point_B)

theorem max_PA2_PB2 :
  let P (θ : ℝ) := (polar_curve_C2 θ * real.cos θ, polar_curve_C2 θ * real.sin θ)
  ∃ θ, (fst (P θ) - fst point_A)^2 + (snd (P θ) - snd point_A)^2 + 
       (fst (P θ) - fst point_B)^2 + (snd (P θ) - snd point_B)^2 ≤ 32 := by
  sorry

end max_PA2_PB2_l124_124632


namespace boys_and_girls_original_total_l124_124425

theorem boys_and_girls_original_total (b g : ℕ) 
(h1 : b = 3 * g) 
(h2 : b - 4 = 5 * (g - 4)) : 
b + g = 32 := 
sorry

end boys_and_girls_original_total_l124_124425


namespace johns_leisure_travel_miles_per_week_l124_124332

-- Define the given conditions
def mpg : Nat := 30
def work_round_trip_miles : Nat := 20 * 2  -- 20 miles to work + 20 miles back home
def work_days_per_week : Nat := 5
def weekly_fuel_usage_gallons : Nat := 8

-- Define the property to prove
theorem johns_leisure_travel_miles_per_week :
  let work_miles_per_week := work_round_trip_miles * work_days_per_week
  let total_possible_miles := weekly_fuel_usage_gallons * mpg
  let leisure_miles := total_possible_miles - work_miles_per_week
  leisure_miles = 40 :=
by
  sorry

end johns_leisure_travel_miles_per_week_l124_124332


namespace solve_problem_l124_124720

noncomputable def problem_statement : Prop :=
  ∃ a b : ℝ, 
  (a ≠ b) ∧ 
  ((x - 3) * (3 * x + 7) = x^2 - 16 * x + 55) → 
  (a + 2) * (b + 2) = -54

theorem solve_problem : problem_statement :=
  sorry

end solve_problem_l124_124720


namespace find_values_and_tangent_line_l124_124628

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 + 3 * Real.log x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := -b * x
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x - g b x

theorem find_values_and_tangent_line (a b : ℝ) 
  (f_der_sqrt2 : (∀ (a : ℝ), (λ x, a * x + 3 * (x ^ (-1))) (Real.sqrt 2 / 2) = 0))
  (f_der_one_g_neg_one : f a 1 + 3 = -(g b (-1)) - 2) :
  a = -6 ∧ b = -1 ∧ (∀ (x : ℝ), (h a b x = -3 * x ^ 2 + 3 * Real.log x - x) →
  ((∃ k : ℝ, k = -4) → (∀ y, y = -4 * x)) ) :=
begin
  -- Proof steps would go here
  sorry
end

end find_values_and_tangent_line_l124_124628


namespace chessboard_black_squares_with_odd_numbers_l124_124806

theorem chessboard_black_squares_with_odd_numbers :
  (numbered_chessboard : array (fin 64) (fin 64)) →
  (initial_black : fin 1 → Prop) →
  (numbered_chessboard 0 = 1) →
  (numbered_chessboard (k+1) = numbered_chessboard k + 1) →
  (parity (numbered_chessboard 0) = 1 % 2) →
  (∀ i : fin 8, (i % 2 = 0) → 
  (∀ j : fin 8, (j % 2 = 0 →
  is_black (numbered_chessboard (8*i + j))
  ∧ ((numbered_chessboard (8*i + j)) % 2 = 1))) →
  (∀ i : fin 8, (i % 2 = 1) → 
  (∀ j : fin 8, (j % 2 = 1 →
  is_black (numbered_chessboard (8*i + j))
  ∧ ((numbered_chessboard (8*i + j)) % 2 = 1))) →  
  ∃ n : ℕ, n = 16 := sorry

end chessboard_black_squares_with_odd_numbers_l124_124806


namespace storage_space_calc_correct_l124_124964

noncomputable def storage_space_available 
    (second_floor_total : ℕ)
    (box_space : ℕ)
    (one_quarter_second_floor : ℕ)
    (first_floor_ratio : ℕ) 
    (second_floor_ratio : ℕ) : ℕ :=
  if (box_space = one_quarter_second_floor ∧
      first_floor_ratio = 2 ∧
      second_floor_ratio = 1) then
    let total_building_space := second_floor_total + (first_floor_ratio * second_floor_total)
    in total_building_space - box_space
  else 0

theorem storage_space_calc_correct :
  storage_space_available 20000 5000 5000 2 1 = 55000 :=
by sorry

end storage_space_calc_correct_l124_124964


namespace total_red_stripes_correct_l124_124748

-- Define the total number of stripes on a flag
def total_stripes : ℕ := 13

-- Define the red stripes on one flag based on conditions
def red_stripes_on_one_flag : ℕ :=
  let remaining_stripes := total_stripes - 1
  let half_remaining_red := remaining_stripes / 2
  1 + half_remaining_red

-- Define the number of flags
def number_of_flags : ℕ := 10

-- Calculate the total red stripes on all flags
def total_red_stripes : ℕ := red_stripes_on_one_flag * number_of_flags

-- The statement to prove
theorem total_red_stripes_correct :
  total_red_stripes = 70 :=
by
  -- Calculate the expected red stripes on one flag
  have red_on_one_flag := 7
  -- Assume number of flags
  have flags := 10
  -- Check the computed total red stripes
  show 7 * 10 = 70 from rfl

end total_red_stripes_correct_l124_124748


namespace product_of_x_values_product_of_solutions_l124_124555

theorem product_of_x_values (x : ℝ) : (|4 * x| + 3 = 35) → (x = 8 ∨ x = -8) :=
by sorry

example : (|4 * 8| + 3 = 35) ∧ (|4 * (-8)| + 3 = 35) :=
by simp

theorem product_of_solutions : (|4 * 8| + 3 = 35) → (|4 * -8| + 3 = 35) → (8 * -8 = -64) :=
by sorry

end product_of_x_values_product_of_solutions_l124_124555


namespace find_x_l124_124583

theorem find_x :
  ∃ n : ℕ, odd n ∧ (6^n + 1).natPrimeFactors.length = 3 ∧ 11 ∈ (6^n + 1).natPrimeFactors ∧ 6^n + 1 = 7777 :=
by sorry

end find_x_l124_124583


namespace sum_is_perfect_square_l124_124198

def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n), (i+1) * 2^i

theorem sum_is_perfect_square (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, S n = k * k) ↔ n = 1 ∨ n = 4 := 
sorry

end sum_is_perfect_square_l124_124198


namespace first_digit_after_decimal_l124_124421

/-- Given the conditions in the problem, prove that the first digit after the decimal point of the sum S is 6. -/
theorem first_digit_after_decimal :
  let S := ∑' n, ((∏ k in finset.range (n+1), (3*k - 1)) / (3^n * nat.factorial n))
  in (floor (10 * S) % 10) = 6 :=
by
  let S := ∑' n, ((∏ k in finset.range (n+1), (3*k - 1)) / (3^n * nat.factorial n))
  have : S = 2 / 3 := sorry
  have : floor (10 * (2 / 3)) % 10 = 6 := by norm_num
  exact this

end first_digit_after_decimal_l124_124421


namespace min_stamps_to_value_l124_124528

theorem min_stamps_to_value (c f g : ℕ) : 
  3 * c + 4 * f + 5 * g = 50 → c + f + g = 10 ↔ 2 ∃ f = 8 ∧ g = 2 ∧ c = 2 :=
sorry

end min_stamps_to_value_l124_124528


namespace car_Z_probability_l124_124674

theorem car_Z_probability :
  let P_X := 1/6
  let P_Y := 1/10
  let P_XYZ := 0.39166666666666666
  ∃ P_Z : ℝ, P_X + P_Y + P_Z = P_XYZ ∧ P_Z = 0.125 :=
by
  sorry

end car_Z_probability_l124_124674


namespace opposite_of_83_is_84_l124_124929

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l124_124929


namespace tamika_greater_prob_l124_124386

-- Conditions: Define the possible products for Tamika and Carlos.
def tamika_products : set ℕ := {7 * 8, 7 * 11, 7 * 13, 8 * 11, 8 * 13, 11 * 13}
def carlos_products : set ℕ := {2 * 4, 2 * 9, 4 * 9}

-- Proof that the probability that Tamika's result is greater than Carlos' result is 1.
theorem tamika_greater_prob : 
  (∀ t ∈ tamika_products, ∀ c ∈ carlos_products, t > c) → 
  (1 : ℚ) = (1 : ℚ) := 
by
  intros,
  sorry

end tamika_greater_prob_l124_124386


namespace freight_train_speed_l124_124152

-- Define the conditions
def distance : ℕ := 460 -- Distance between A and B in km
def time : ℕ := 2 -- Time after which the trains meet in hours
def passenger_speed : ℕ := 120 -- Speed of the passenger train in km/h

-- Formalize the problem statement and expected result
theorem freight_train_speed :
  let total_speed := distance / time in
  let freight_speed := total_speed - passenger_speed in
  freight_speed = 110 :=
by
  -- Proof outline:
  -- 1. total_speed = distance / time = 460 / 2 = 230 km/h
  -- 2. freight_speed = total_speed - passenger_speed = 230 - 120 = 110 km/h
  sorry

end freight_train_speed_l124_124152


namespace find_f_neg1_l124_124613

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f : ℝ → ℝ
| x => if 0 < x then x^2 + 2 else if x = 0 then 2 else -(x^2 + 2)

axiom odd_f : is_odd_function f

theorem find_f_neg1 : f (-1) = -3 := by
  sorry

end find_f_neg1_l124_124613


namespace number_of_zeros_of_f_l124_124197

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem number_of_zeros_of_f :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l124_124197


namespace problem_1_l124_124178

theorem problem_1 : -9 + 5 - (-12) + (-3) = 5 :=
by {
  -- Proof goes here
  sorry
}

end problem_1_l124_124178


namespace number_of_cows_l124_124525

theorem number_of_cows (H : ℕ) (C : ℕ) (h1 : H = 6) (h2 : C / H = 7 / 2) : C = 21 :=
by
  sorry

end number_of_cows_l124_124525


namespace each_friend_pays_l124_124221

def hamburgers_cost : ℝ := 5 * 3
def fries_cost : ℝ := 4 * 1.20
def soda_cost : ℝ := 5 * 0.50
def spaghetti_cost : ℝ := 1 * 2.70
def total_cost : ℝ := hamburgers_cost + fries_cost + soda_cost + spaghetti_cost
def num_friends : ℝ := 5

theorem each_friend_pays :
  total_cost / num_friends = 5 :=
by
  sorry

end each_friend_pays_l124_124221


namespace increasing_on_iff_decreasing_on_periodic_even_l124_124614

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f x = f (x + p)
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

theorem increasing_on_iff_decreasing_on_periodic_even :
  (is_even f ∧ is_periodic f 2 ∧ is_increasing_on f 0 1) ↔ is_decreasing_on f 3 4 := 
by
  sorry

end increasing_on_iff_decreasing_on_periodic_even_l124_124614


namespace complement_intersection_l124_124251

open Set

theorem complement_intersection :
  (\compl ({x : ℝ | ∃ y : ℝ, y = Real.log(x-1)}) ∩ {x : ℝ | -1 < x ∧ x < 2}) = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  sorry

end complement_intersection_l124_124251


namespace no_prime_sum_forty_seven_l124_124681

theorem no_prime_sum_forty_seven : ¬ ∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 47 :=
by
  sorry

end no_prime_sum_forty_seven_l124_124681


namespace average_score_cannot_be_85_l124_124161

/-- Given the scores 85, 67, m, 80, and 93 in five exams where m > 0, and the median is 80,
prove that the average score cannot be 85. -/
theorem average_score_cannot_be_85 
  (m : ℝ) (h1 : m > 0) (h2 : (85, 67, m, 80, 93).med = 80) : 
  (85 + 67 + m + 80 + 93) / 5 ≠ 85 :=
sorry

end average_score_cannot_be_85_l124_124161


namespace find_x_l124_124585

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l124_124585


namespace smallest_large_chips_l124_124810

theorem smallest_large_chips :
  ∃ l : ℕ, ∃ s : ℕ, ∃ n : ℕ, n^2 ≤ 72 ∧ s + l = 72 ∧ s = l - n^2 ∧ ∀ k : ℕ, k < 38 → 
    (∀ m : ℕ, m^2 ≤ 72 → s + k = 72 ∧ s = k - m^2 → k := 38 := λ s l n hs hl hs_eq,  sorry

end smallest_large_chips_l124_124810


namespace last_triangle_perimeter_l124_124715

theorem last_triangle_perimeter :
  let T1 : ℕ × ℕ × ℕ := (1001, 1002, 1003)
  let rec next_triangle (T : ℕ × ℕ × ℕ) (n : ℕ) : ℕ × ℕ × ℕ :=
    match n with
    | 0 => T
    | n + 1 => let (a, b, c) := T
               (a / 2, b / 2, c / 2)
  let perimeter (T : ℕ × ℕ × ℕ) : ℚ :=
    let (a, b, c) := T
    (a + b + c : ℚ)
  in perimeter (next_triangle T1 9) = 1503 / 256 :=
begin
  sorry
end

end last_triangle_perimeter_l124_124715


namespace cos_2sum_zero_l124_124726

noncomputable def cos_sum_eq_one (x y z : ℝ) : Prop := 
  real.cos x + real.cos y + real.cos z = 1

noncomputable def sin_sum_eq_one (x y z : ℝ) : Prop := 
  real.sin x + real.sin y + real.sin z = 1

theorem cos_2sum_zero (x y z : ℝ) 
  (h_cos_sum: cos_sum_eq_one x y z) 
  (h_sin_sum: sin_sum_eq_one x y z) :
  real.cos (2*x) + real.cos (2*y) + real.cos (2*z) = 0 :=
by
  sorry

end cos_2sum_zero_l124_124726


namespace poplars_and_lindens_l124_124069

theorem poplars_and_lindens (x y : ℕ) (h1 : 100 ≤ x ∧ x < 1000) (h2 : 10 ≤ y ∧ y < 100) 
  (h3 : x + y = 144) (h4 : let x' := 100 * (x % 10) + 10 * (x / 10 % 10) + (x / 100),
                          let y' := 10 * (y % 10) + (y / 10) in
                          x' + y' = 603) : 
  x = 105 ∧ y = 39 :=
by
  sorry

end poplars_and_lindens_l124_124069


namespace solution_set_f_x_greater_f_2x_minus_4_l124_124264

def f (x : ℝ) : ℝ := log x + 2^x + real.sqrt x - 1

theorem solution_set_f_x_greater_f_2x_minus_4 : 
  { x : ℝ | 0 < x ∧ 2 < x ∧ f x > f (2*x - 4) } = { x : ℝ | 2 < x ∧ x < 4 } :=
by
  sorry

end solution_set_f_x_greater_f_2x_minus_4_l124_124264


namespace sum_remainder_l124_124466

theorem sum_remainder : ∃ S : ℕ, 
  let S := (2015 : ℝ) * ∑ k in finset.range 401, (1 / (5 * k - 2) - 1 / (5 * k + 3)) in
  (S / 5).floor.mod 5 = 4 := sorry

end sum_remainder_l124_124466


namespace correct_ordering_of_fractions_l124_124850

theorem correct_ordering_of_fractions :
  let a := (6 : ℚ) / 17
  let b := (8 : ℚ) / 25
  let c := (10 : ℚ) / 31
  let d := (1 : ℚ) / 3
  b < d ∧ d < c ∧ c < a :=
by
  sorry

end correct_ordering_of_fractions_l124_124850


namespace PascalTheorem_l124_124344

variable {A B C D E F P Q R : Type}

-- Conditions about geometric about points on the circle and intersection properties
axiom GeoCirc (Γ : Type) (A B C D E F : Γ) : PointOnCircle A Γ → PointOnCircle B Γ → PointOnCircle C Γ → PointOnCircle D Γ → PointOnCircle E Γ → PointOnCircle F Γ

-- Points P, Q, R as intersections
axiom IntersectionP (Γ : Type) (P : Γ) (A B D E : Point) : IntersectionPoint P (Line A B) (Line D E)
axiom IntersectionQ (Γ : Type) (Q : Γ) (B C E F : Point) : IntersectionPoint Q (Line B C) (Line E F)
axiom IntersectionR (Γ : Type) (R : Γ) (C D F A : Point) : IntersectionPoint R (Line C D) (Line F A)

-- Collinearity
theorem PascalTheorem (Γ : Type) (A B C D E F P Q R : Point) 
  [GeoCirc Γ A B C D E F]
  [IntersectionP Γ P A B D E]
  [IntersectionQ Γ Q B C E F]
  [IntersectionR Γ R C D F A] : Collinear P Q R := sorry

end PascalTheorem_l124_124344


namespace probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124823

-- Define the problem's parameters and conditions
def n : ℕ  -- number of pairs of socks

-- Define the successful pair probability calculation
theorem probability_all_pairs_successful (n : ℕ) :
  (∑ s in Finset.range (n + 1), s.factorial * 2 ^ s) / ((2 * n).factorial) = 2^n * n.factorial := sorry

-- Define the expected value calculation
theorem expected_successful_pairs_greater_than_half (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n, (1 : ℝ) / (2 * n - 1) : ℝ) / n > 0.5 := sorry

end probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124823


namespace smallest_m_satisfying_condition_l124_124557

theorem smallest_m_satisfying_condition :
  ∃ m : ℕ, m > 0 ∧ (∀ p : ℕ, Nat.Prime p → p > 3 → 105 ∣ 9^(p^2) - 29^p + m) ∧
          m = 95 :=
begin
  sorry,
end

end smallest_m_satisfying_condition_l124_124557


namespace optimal_discount_savings_l124_124396

def cookbooks_cover_price := 50 -- dollars
def discount_dollars := 10 -- dollars
def discount_percent := 0.25

def apply_discounts (price : ℕ) : ℕ :=
  let first_order := (price - discount_dollars)
  let second_order := (price * Float.ofNat(1 - discount_percent))
  (first_order * Float.ofNat(1 - discount_percent), (second_order - discount_dollars):ℕ)

theorem optimal_discount_savings :
  let price := cookbooks_cover_price
  let (final_first, final_second) := apply_discounts price
  (final_first - final_second) = 250
:=
  sorry

end optimal_discount_savings_l124_124396


namespace min_value_of_A_l124_124040

noncomputable def A (a b c : ℝ) : ℝ :=
  (a^3 + b^3) / (8 * a * b + 9 - c^2) +
  (b^3 + c^3) / (8 * b * c + 9 - a^2) +
  (c^3 + a^3) / (8 * c * a + 9 - b^2)

theorem min_value_of_A (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  A a b c = 3 / 8 :=
sorry

end min_value_of_A_l124_124040


namespace find_m_l124_124301

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : ∃ s : ℝ, (s = (m + 1 - 4) / (2 - m)) ∧ s = Real.sqrt 5) :
  m = (10 - Real.sqrt 5) / 4 :=
by
  sorry

end find_m_l124_124301


namespace modulus_of_squared_complex_l124_124623

open Complex

theorem modulus_of_squared_complex :
  let z : ℂ := (3 + Complex.i)^2
  |z| = 10 :=
by
  let z : ℂ := (3 + Complex.i)^2
  have h : |z| = 10
  sorry

end modulus_of_squared_complex_l124_124623


namespace tickets_used_to_buy_toys_l124_124959

-- Definitions for the conditions
def initial_tickets : ℕ := 13
def leftover_tickets : ℕ := 7

-- The theorem we want to prove
theorem tickets_used_to_buy_toys : initial_tickets - leftover_tickets = 6 :=
by
  sorry

end tickets_used_to_buy_toys_l124_124959


namespace find_B_l124_124147

theorem find_B (A C B : ℕ) (hA : A = 520) (hC : C = A + 204) (hCB : C = B + 179) : B = 545 :=
by
  sorry

end find_B_l124_124147


namespace min_val_proof_l124_124016

noncomputable def minimum_value (x y z: ℝ) := 9 / x + 4 / y + 1 / z

theorem min_val_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * y + 3 * z = 12) :
  minimum_value x y z ≥ 49 / 12 :=
by {
  sorry
}

end min_val_proof_l124_124016


namespace count_multiples_3_or_4_between_1_and_601_l124_124646

theorem count_multiples_3_or_4_between_1_and_601 : 
  let N := Nat.floor (601 / 3)
  let M := Nat.floor (601 / 4)
  let L := Nat.floor (601 / 12)
  N + M - L = 300 :=
by 
  let N := Nat.floor (601 / 3)
  let M := Nat.floor (601 / 4)
  let L := Nat.floor (601 / 12)
  have N_cal : N = 200 := by sorry
  have M_cal : M = 150 := by sorry
  have L_cal : L = 50 := by sorry
  show N + M - L = 300 from by rw [N_cal, M_cal, L_cal]; sorry

end count_multiples_3_or_4_between_1_and_601_l124_124646


namespace range_of_b_l124_124631

theorem range_of_b {b : ℝ} (h_b_ne_zero : b ≠ 0) :
  (∃ x : ℝ, (0 ≤ x ∧ x ≤ 3) ∧ (2 * x + b = 3)) ↔ -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 :=
by
  sorry

end range_of_b_l124_124631


namespace f_2013_eq_neg_3_l124_124993

noncomputable def f : ℤ → ℝ
| x <= 0 := log (2, 8 - x)
| x > 0 := f (x - 1) - f (x - 2)

theorem f_2013_eq_neg_3 : f 2013 = -3 := 
sorry

end f_2013_eq_neg_3_l124_124993


namespace correct_calculation_l124_124459

theorem correct_calculation :
  let A := (sqrt 3 + sqrt 4 = sqrt 7) in
  let B := (3 * sqrt 5 - sqrt 5 = 3) in
  let C := (sqrt 2 * sqrt 5 = 10) in
  let D := (sqrt 18 / sqrt 2 = 3) in
  ¬A ∧ ¬B ∧ ¬C ∧ D :=
by
  let A := (sqrt 3 + sqrt 4 = sqrt 7)
  let B := (3 * sqrt 5 - sqrt 5 = 3)
  let C := (sqrt 2 * sqrt 5 = 10)
  let D := (sqrt 18 / sqrt 2 = 3)
  have hA : ¬A := sorry
  have hB : ¬B := sorry
  have hC : ¬C := sorry
  have hD : D := sorry
  exact ⟨hA, hB, hC, hD⟩

end correct_calculation_l124_124459


namespace binom_19_9_l124_124611

variable (n k : ℕ)

theorem binom_19_9
  (h₁ : nat.choose 17 7 = 19448)
  (h₂ : nat.choose 17 8 = 24310)
  (h₃ : nat.choose 17 9 = 24310) :
  nat.choose 19 9 = 92378 := by
  sorry

end binom_19_9_l124_124611


namespace factorization_25x2_minus_155x_minus_150_l124_124792

theorem factorization_25x2_minus_155x_minus_150 :
  ∃ (a b : ℤ), (a + b) * 5 = -155 ∧ a * b = -150 ∧ a + 2 * b = 27 :=
by
  sorry

end factorization_25x2_minus_155x_minus_150_l124_124792


namespace cos_beta_given_conditions_l124_124255

theorem cos_beta_given_conditions 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h_cos_α : cos α = 5 / 13) 
  (h_cos_αβ : cos (α + β) = - 4 / 5) :
  cos β = 16 / 65 := 
by
  -- Proof goes here
  sorry

end cos_beta_given_conditions_l124_124255


namespace candies_distribution_l124_124740

theorem candies_distribution : (nat.partitions_with_length 10 5).length = 30 := 
sorry

end candies_distribution_l124_124740


namespace S8_correct_l124_124579

noncomputable theory

variables {a_n : ℕ → ℝ} -- Define the geometric sequence
variable (q : ℝ)        -- Define the common ratio
variable S : ℕ → ℝ      -- Define the summation function for the first n terms

-- Conditions of the problem
axiom h_q : q = 2
axiom h_S4 : S 4 = 1
axiom geo_seq : ∀ n, S n = a_n 0 * (1 - q^(n)) / (1 - q)

-- Question and correct answer
theorem S8_correct : S 8 = 17 :=
by sorry

end S8_correct_l124_124579


namespace probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124824

-- Define the problem's parameters and conditions
def n : ℕ  -- number of pairs of socks

-- Define the successful pair probability calculation
theorem probability_all_pairs_successful (n : ℕ) :
  (∑ s in Finset.range (n + 1), s.factorial * 2 ^ s) / ((2 * n).factorial) = 2^n * n.factorial := sorry

-- Define the expected value calculation
theorem expected_successful_pairs_greater_than_half (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n, (1 : ℝ) / (2 * n - 1) : ℝ) / n > 0.5 := sorry

end probability_all_pairs_successful_expected_successful_pairs_greater_than_half_l124_124824


namespace total_people_on_boats_l124_124476

-- Definitions based on the given conditions
def boats : Nat := 5
def people_per_boat : Nat := 3

-- Theorem statement to prove the total number of people on boats in the lake
theorem total_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l124_124476


namespace divides_power_sum_l124_124338

theorem divides_power_sum (a b c : ℤ) (h : a + b + c ∣ a^2 + b^2 + c^2) : ∀ k : ℕ, a + b + c ∣ a^(2^k) + b^(2^k) + c^(2^k) :=
by
  intro k
  induction k with
  | zero =>
    sorry -- Base case proof
  | succ k ih =>
    sorry -- Inductive step proof

end divides_power_sum_l124_124338


namespace real_part_of_difference_l124_124968

-- Defining the complex numbers
def c1 : ℂ := -5 - 3 * complex.i
def c2 : ℂ := 2 + 6 * complex.i

-- Proving that the real part of (c1 - c2) is -7
theorem real_part_of_difference : complex.re (c1 - c2) = -7 := 
by sorry

end real_part_of_difference_l124_124968


namespace exists_set_P_l124_124470

theorem exists_set_P (S : Finset (EuclideanSpace ℝ (fin 2))) (hS : S.card = n)
  (h_collinear: ∀ (A B C : EuclideanSpace ℝ (fin 2)), A ∈ S → B ∈ S → C ∈ S → collinear ℝ ({A, B, C} : Set (EuclideanSpace ℝ (fin 2))) → false) :
  ∃ P : Finset (EuclideanSpace ℝ (fin 2)), P.card = 2 * n - 5 ∧ ∀ (A B C : EuclideanSpace ℝ (fin 2)), A ∈ S → B ∈ S → C ∈ S → ∃ D ∈ P, D ∈ interior (convex_hull ℝ ({A, B, C} : Set (EuclideanSpace ℝ (fin 2)))) :=
by
  sorry

end exists_set_P_l124_124470


namespace find_x_l124_124581

theorem find_x :
  ∃ n : ℕ, odd n ∧ (6^n + 1).natPrimeFactors.length = 3 ∧ 11 ∈ (6^n + 1).natPrimeFactors ∧ 6^n + 1 = 7777 :=
by sorry

end find_x_l124_124581


namespace carlos_books_in_june_l124_124977

def books_in_july : ℕ := 28
def books_in_august : ℕ := 30
def goal_books : ℕ := 100

theorem carlos_books_in_june :
  let books_in_july_august := books_in_july + books_in_august
  let books_needed_june := goal_books - books_in_july_august
  books_needed_june = 42 := 
by
  sorry

end carlos_books_in_june_l124_124977


namespace CarlosBooksInJune_l124_124974

def CarlosBooksInJuly := 28
def CarlosBooksInAugust := 30
def CarlosTotalBooksGoal := 100

theorem CarlosBooksInJune :
  ∃ x : ℕ, x = CarlosTotalBooksGoal - (CarlosBooksInJuly + CarlosBooksInAugust) :=
begin
  use 42,
  dsimp [CarlosTotalBooksGoal, CarlosBooksInJuly, CarlosBooksInAugust],
  norm_num,
  sorry
end

end CarlosBooksInJune_l124_124974


namespace sine_sum_identity_example_l124_124199

theorem sine_sum_identity_example :
  sin (15 * Real.pi / 180) * cos (75 * Real.pi / 180) + cos (15 * Real.pi / 180) * sin (75 * Real.pi / 180) = 1 := 
by 
  sorry

end sine_sum_identity_example_l124_124199


namespace common_remainder_l124_124857

theorem common_remainder (n : ℕ) : n = 1391 → (n % 7 = 5 ∧ n % 9 = 5 ∧ n % 11 = 5) :=
by
  assume h : n = 1391,
  sorry

end common_remainder_l124_124857


namespace knights_in_exchange_l124_124363

noncomputable def count_knights (total_islanders : ℕ) (odd_statements : ℕ) (even_statements : ℕ) : ℕ :=
if total_islanders % 2 = 0 ∧ odd_statements = total_islanders ∧ even_statements = total_islanders then
    total_islanders / 2
else
    0

theorem knights_in_exchange : count_knights 30 30 30 = 15 :=
by
    -- proof part will go here but is not required.
    sorry

end knights_in_exchange_l124_124363


namespace median_of_set_is_88_5_l124_124795

theorem median_of_set_is_88_5 :
  let numbers := [92, 90, 85, 88, 89, 87] in
  let median := (88 + 89) / 2 in
  median = 88.5 := by
sorry

end median_of_set_is_88_5_l124_124795


namespace concyclic_points_l124_124643

-- The points A, B, C, D, E, O, P, Q, R, H
variables (A B C D E O P Q R H : Type)

-- The conditions
variables [is_triangle ABC] -- \( O \) is the circumcenter of \( \triangle ABC \)
variables [on_line_segment AC D] -- \( D \) is on \( AC \)
variables [on_line_segment AB E] -- \( E \) is on \( AB \)
variables [midpoint_of_line_segment DE P] -- \( P \) is the midpoint of \( DE \)
variables [midpoint_of_line_segment BD Q] -- \( Q \) is the midpoint of \( BD \)
variables [midpoint_of_line_segment CE R] -- \( R \) is the midpoint of \( CE \)
variables [perpendicular_from O_to_line_segment DE H] -- \( OH \perp DE \) with \( H \) as the foot of the perpendicular

-- The theorem to prove
theorem concyclic_points : cyclic_quadrilateral P Q R H :=
sorry

end concyclic_points_l124_124643


namespace calculation_of_expression_l124_124129

theorem calculation_of_expression :
  16^(1/2:ℝ) + (1/81:ℝ)^(-0.25) - (-1/2:ℝ)^0 = 6 := by
  sorry

end calculation_of_expression_l124_124129


namespace necessary_but_not_sufficient_l124_124580

-- Definitions of lines, planes and parallelism
variables {Point : Type} [affine_space Point] (a b : set Point) (α β : set Point)

-- Conditions as hypotheses
variables (ha : a ⊆ α) (hb : b ⊆ α) (hpa : is_parallel a β) (hpb : is_parallel b β)

-- Target statement
theorem necessary_but_not_sufficient :
  (a ⊆ α) → (b ⊆ α) → (is_parallel a β) → (is_parallel b β) → (necessary_condition (is_parallel α β) ∧ ¬sufficient_condition (is_parallel α β)) :=
begin
  assume ha hb hpa hpb,
  sorry
end

end necessary_but_not_sufficient_l124_124580


namespace range_of_h_l124_124216

def h (t : ℝ) : ℝ := (t^2 + 2*t) / (t^2 + 2*t + 3)

theorem range_of_h : set.range h = set.Icc (-1/2 : ℝ) 1 :=
sorry

end range_of_h_l124_124216


namespace math_problem_james_ladder_l124_124697

def convertFeetToInches(feet : ℕ) : ℕ :=
  feet * 12

def totalRungSpace(rungLength inchesApart : ℕ) : ℕ :=
  rungLength + inchesApart

def totalRungsRequired(totalHeight rungSpace : ℕ) : ℕ :=
  totalHeight / rungSpace

def totalWoodRequiredInInches(rungsRequired rungLength : ℕ) : ℕ :=
  rungsRequired * rungLength

def convertInchesToFeet(inches : ℕ) : ℕ :=
  inches / 12

def woodRequiredForRungs(feetToClimb rungLength inchesApart : ℕ) : ℕ :=
   convertInchesToFeet
    (totalWoodRequiredInInches
      (totalRungsRequired
        (convertFeetToInches feetToClimb) 
        (totalRungSpace rungLength inchesApart))
      rungLength)

theorem math_problem_james_ladder : 
  woodRequiredForRungs 50 18 6 = 37.5 :=
sorry

end math_problem_james_ladder_l124_124697


namespace optimal_addition_amount_l124_124441

theorem optimal_addition_amount (a b g : ℝ) (h₁ : a = 628) (h₂ : b = 774) (h₃ : g = 718) : 
    b + a - g = 684 :=
by
  sorry

end optimal_addition_amount_l124_124441


namespace statement_A_statement_C_l124_124309

variables {Student : Type} (score : Student → ℝ) (grade : Student → ℝ)
def B : ℝ := 3.0
def C : ℝ := 2.0

-- Condition 1
axiom condition1 : ∀ s : Student, score s ≥ 90 → grade s ≥ B

-- Condition 2
axiom condition2 : ∀ s : Student, score s < 70 → grade s ≤ C

-- Statements to prove
theorem statement_A : (∀ s : Student, grade s < B → score s < 90) :=
by 
  intro s
  intro h
  have contrapositive_condition1 := λ (s : Student) (h : grade s < B), not_le.mp (λ h', not_le.mp (condition1 s h')) h
  exact contrapositive_condition1 s h

theorem statement_C : (∀ s : Student, grade s > C → score s ≥ 70) :=
by
  intro s
  intro h
  have contrapositive_condition2 := λ (s : Student) (h : grade s > C), not_lt.mp (λ h', not_le.mp (λ h'', ne.symm (lt_asymm h) $ lt_of_le_of_ne (condition2 s h'') $ not_le.mp h'))
  exact contrapositive_condition2 s h


end statement_A_statement_C_l124_124309


namespace trapezoid_reflection_angle_sum_eq_90_l124_124086

-- Define the given conditions and prove the statement
theorem trapezoid_reflection_angle_sum_eq_90
  {A B C D P Q M : Type*}
  [trapezoid A B C D]
  (h_parallel : parallel A B C D)
  (h_diagonals_perpendicular : perpendicular_diagonals A C B D P)
  (h_Q_reflection : reflection_point_over_line P A B Q)
  (h_M_midpoint_CD : midpoint M C D) :
  angle A M B + angle C Q D = 90 :=
sorry

end trapezoid_reflection_angle_sum_eq_90_l124_124086


namespace brothers_savings_l124_124694

theorem brothers_savings :
  ∀ (isabelle_ticket brothers_ticket isabelle_saved work_weeks pay_per_week : ℕ),
  isabelle_ticket = 20 →
  brothers_ticket = 10 →
  isabelle_saved = 5 →
  work_weeks = 10 →
  pay_per_week = 3 →
  let total_ticket_cost := isabelle_ticket + 2 * brothers_ticket,
      isabelle_needs := total_ticket_cost - isabelle_saved,
      isabelle_earns := work_weeks * pay_per_week in
  (isabelle_needs - isabelle_earns = 5) :=
by
  intros isabelle_ticket brothers_ticket isabelle_saved work_weeks pay_per_week
  intros h1 h2 h3 h4 h5
  let total_ticket_cost := isabelle_ticket + 2 * brothers_ticket
  let isabelle_needs := total_ticket_cost - isabelle_saved
  let isabelle_earns := work_weeks * pay_per_week
  show isabelle_needs - isabelle_earns = 5 from sorry

end brothers_savings_l124_124694


namespace opposite_point_83_is_84_l124_124898

theorem opposite_point_83_is_84 :
  ∀ (n : ℕ) (k : ℕ) (points : Fin n) 
  (H : n = 100) 
  (H1 : k = 83) 
  (H2 : ∀ k, 1 ≤ k ∧ k ≤ 100 ∧ (∑ i in range k, i < k ↔ ∑ i in range 100, i = k)), 
  ∃ m, m = 84 ∧ diametrically_opposite k m := 
begin
  sorry
end

end opposite_point_83_is_84_l124_124898


namespace extreme_points_inequality_l124_124269

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) + 1 / 2 * x ^ 2 - x

theorem extreme_points_inequality 
  (a : ℝ)
  (ha : 0 < a ∧ a < 1)
  (alpha beta : ℝ)
  (h_eq_alpha : alpha = -Real.sqrt (1 - a))
  (h_eq_beta : beta = Real.sqrt (1 - a))
  (h_order : alpha < beta) :
  (f a beta / alpha) < (1 / 2) :=
sorry

end extreme_points_inequality_l124_124269


namespace area_of_triangle_PF1F2_l124_124005

noncomputable def ellipse := {P : ℝ × ℝ // (4 * P.1^2) / 49 + (P.2^2) / 6 = 1}

noncomputable def area_triangle (P F1 F2 : ℝ × ℝ) :=
  1 / 2 * abs ((F1.1 - P.1) * (F2.2 - P.2) - (F1.2 - P.2) * (F2.1 - P.1))

theorem area_of_triangle_PF1F2 :
  ∀ (F1 F2 : ℝ × ℝ) (P : ellipse), 
    (dist P.1 F1 = 4) →
    (dist P.1 F2 = 3) →
    (dist F1 F2 = 5) →
    area_triangle P.1 F1 F2 = 6 :=
by sorry

end area_of_triangle_PF1F2_l124_124005
