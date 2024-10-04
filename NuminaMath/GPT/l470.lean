import Mathlib

namespace min_x1_x2_squared_l470_470831

theorem min_x1_x2_squared (x1 x2 m : ℝ) (hm : (m + 3)^2 ≥ 0) 
  (h_sum : x1 + x2 = -(m + 1)) 
  (h_prod : x1 * x2 = 2 * m - 2) : 
  (x1^2 + x2^2 = (m - 1)^2 + 4) ∧ ∃ m, m = 1 → x1^2 + x2^2 = 4 :=
by {
  sorry
}

end min_x1_x2_squared_l470_470831


namespace journey_length_is_225_l470_470832

noncomputable def lengthOfJourney (L T : ℝ) : Prop :=
  L = 60 * T ∧ L = 50 * (T + 3 / 4)

theorem journey_length_is_225 : ∃ (L T : ℝ), lengthOfJourney L T ∧ L = 225 :=
by {
  use 225, -- suggested length L
  use 3.75, -- suggested time T
  unfold lengthOfJourney,
  split,
  { calc
    225 = 60 * 3.75 : by norm_num },
  { calc
    225 = 50 * (3.75 + 3 / 4) : by norm_num }
}

end journey_length_is_225_l470_470832


namespace total_travel_cost_l470_470762

noncomputable def travel_cost (CA AB: ℝ) (bus_cost_per_km airplane_cost_per_km airplane_booking_fee: ℝ) : ℝ :=
  let BC := real.sqrt (AB^2 - CA^2)
  let fare_cost (distance: ℝ) := min (distance * bus_cost_per_km) (airplane_booking_fee + distance * airplane_cost_per_km)
  (fare_cost AB + fare_cost BC + fare_cost CA)

theorem total_travel_cost :
  travel_cost 4000 4500 0.20 0.12 120 = 1627.39 :=
by
  sorry

end total_travel_cost_l470_470762


namespace part_a_part_b_l470_470051

theorem part_a (x y : ℝ) (h : x ≥ y^2) :
  √(x + 2 * y * √(x - y^2)) + √(x - 2 * y * √(x - y^2)) = max (2 * |y|) (2 * √(x - y^2)) :=
sorry

theorem part_b (x y z : ℝ) (h : x * y + y * z + z * x = 1) :
  (2 * x * y * z) / √((1 + x^2) * (1 + y^2) * (1 + z^2)) = (2 * x * y * z) / |x + y + z - x * y * z| :=
sorry

end part_a_part_b_l470_470051


namespace fn_expression_l470_470166

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
if n = 0 then 0 else (λ f1, Nat.recOn (n - 1) f1 (λ k fk, f1 fk)) (λ y, y / Real.sqrt(1 + y^2)) x

theorem fn_expression (n : ℕ) (x : ℝ) (h : n > 0) (hx : x > 0) :
  f n x = x / Real.sqrt(1 + (n : ℝ) * x^2) :=
by sorry

end fn_expression_l470_470166


namespace find_AD_l470_470617

-- Define the conditions for the problem
variables (A B C D X : Point)
variable (is_midpoint_AC : midpoint X A C)
variable (is_parallel_CD_BX : parallel CD BX)
variable (length_BX : length BX = 3)
variable (length_BC : length BC = 7)
variable (length_CD : length CD = 6)
variable (is_convex_ABCD : convex ABCD)

-- Define the main theorem that states the problem
theorem find_AD :
  length (segment A D) = 14 :=
sorry

end find_AD_l470_470617


namespace children_exceed_bridge_limit_l470_470079

theorem children_exceed_bridge_limit :
  ∀ (Kelly_weight : ℕ) (Megan_weight : ℕ) (Mike_weight : ℕ),
  Kelly_weight = 34 ∧
  Kelly_weight = (85 * Megan_weight) / 100 ∧
  Mike_weight = Megan_weight + 5 →
  Kelly_weight + Megan_weight + Mike_weight - 100 = 19 :=
by sorry

end children_exceed_bridge_limit_l470_470079


namespace func_domain_condition_min_value_condition_l470_470493

-- Problem 1
theorem func_domain_condition (a : ℝ) :
  (∀ x : ℝ, {|x + 1| + |x - 2| - a} ≥ 0) ↔ a ≤ 3 :=
sorry

-- Problem 2
theorem min_value_condition (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) :
  x^2 + y^2 + z^2 ≥ 1 / 14 :=
sorry

end func_domain_condition_min_value_condition_l470_470493


namespace finite_subsets_of_R_condition_l470_470587

noncomputable def M_sets : set (set ℝ) :=
  {{-Real.sqrt 5 / 3, Real.sqrt 5 / 3},
   {(1 - Real.sqrt 17) / 6, (1 + Real.sqrt 17) / 6},
   {(-1 - Real.sqrt 17) / 6, (-1 + Real.sqrt 17) / 6}}

theorem finite_subsets_of_R_condition :
  ∀ (M : set ℝ), (finite M ∧ 2 ≤ M.card ∧ ∀ a b ∈ M, a ≠ b → (a^3 - 4 / 9 * b) ∈ M) ↔ M ∈ M_sets :=
by 
  sorry

end finite_subsets_of_R_condition_l470_470587


namespace age_of_youngest_child_l470_470009

theorem age_of_youngest_child
  (x : ℕ)
  (sum_of_ages : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) :
  x = 4 :=
sorry

end age_of_youngest_child_l470_470009


namespace inequality_holds_for_all_real_numbers_l470_470373

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470373


namespace parallel_vectors_x_value_l470_470992

def vectors_are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, vectors_are_parallel (-1, 4) (x, 2) → x = -1 / 2 := 
by 
  sorry

end parallel_vectors_x_value_l470_470992


namespace true_proposition_l470_470628

def p := ∀ (a : ℝ) (x : ℝ), ¬(a^(x + 1) = 1) ∨ x ≠ 0
def q := ∀ (f : ℝ → ℝ), (∀ x, f(x) = f(-x)) → (∀ x, f(x + 1) = f(-(x - 1))) 

theorem true_proposition : p ∨ ¬q :=
by
  sorry

end true_proposition_l470_470628


namespace find_arith_seq_params_l470_470826

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- The conditions given in the problem
theorem find_arith_seq_params :
  ∃ a d : ℤ, 
  (arithmetic_sequence a d 8) = 5 * (arithmetic_sequence a d 1) ∧
  (arithmetic_sequence a d 12) = 2 * (arithmetic_sequence a d 5) + 5 ∧
  a = 3 ∧
  d = 4 :=
by
  sorry

end find_arith_seq_params_l470_470826


namespace determinant_scaled_matrix_l470_470954

theorem determinant_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 7) :
  |3 * a 3 * b|
  |3 * c 3 * d| = 63 := 
begin
  sorry
end

end determinant_scaled_matrix_l470_470954


namespace rotated_square_distance_l470_470443

theorem rotated_square_distance
  (side_length : ℝ)
  (rot_angle : ℝ)
  (h_side_length : side_length = 4)
  (h_rot_angle : rot_angle = 30) :
  let diagonal := side_length * real.sqrt 2 in
  let height := (diagonal / 2) * real.sin (rot_angle * real.pi / 180) in
  height = real.sqrt 2 := by
  sorry

end rotated_square_distance_l470_470443


namespace prove_minimum_disks_needed_l470_470116

def file_storage_proof_problem : Prop :=
  ∃ (total_files disk_capacity : ℕ) (file_sizes : list ℝ), 
    total_files = 45 ∧
    disk_capacity = 1.44 ∧
    file_sizes =
      [1.0, 1.0, 1.0, 1.0, 1.0, 
       0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 
       0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 
       0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
       0.3, 0.3, 0.3, 0.3, 0.3] ∧ 
      (∀ (disk := ℕ) (disk_index : list ℝ), ∃ (i : ℕ), 
        (sum disk_index ≤ i * disk_capacity) ∧ 
        list.length disk_index = i ∧ 
        disk = i) ∧
    ∃ (total_disks_needed : ℕ), total_disks_needed = 16

theorem prove_minimum_disks_needed : file_storage_proof_problem := by
  sorry

end prove_minimum_disks_needed_l470_470116


namespace slope_of_line_dividing_rectangle_l470_470438

-- Define the coordinates of the vertices of the rectangle
def vertices : List (ℝ × ℝ) := [(1, 0), (5, 0), (1, 2), (5, 2)]

-- Define the point representing the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the function to calculate the center of the rectangle
def center_of_rectangle (v1 v2 v3 v4 : ℝ × ℝ) : ℝ × ℝ :=
  let x_coords := [v1.1, v2.1, v3.1, v4.1]
  let y_coords := [v1.2, v2.2, v3.2, v4.2]
  ((x_coords.sum / x_coords.length), (y_coords.sum / y_coords.length))

-- Calculate the center of the given rectangle
def rectangle_center : ℝ × ℝ :=
  center_of_rectangle (1, 0) (5, 0) (1, 2) (5, 2)

-- Define the function to calculate the slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the theorem we want to prove
theorem slope_of_line_dividing_rectangle :
  slope origin rectangle_center = 1 / 3 := by
  sorry

end slope_of_line_dividing_rectangle_l470_470438


namespace coefficient_x4_expansion_l470_470815

theorem coefficient_x4_expansion :
  let f := fun (x : ℚ) => (1 - 2 * x^2) ^ 5 in
  (∃ c : ℚ, c * x^4 ∈ f.expand ℚ [x]) ∧
  c = 40 :=
by
  sorry

end coefficient_x4_expansion_l470_470815


namespace ellipse_eccentricity_l470_470642

noncomputable def eccentricity_of_ellipse (a c : ℝ) : ℝ :=
  c / a

theorem ellipse_eccentricity (F1 A : ℝ) (v : ℝ) (a c : ℝ)
  (h1 : 4 * a = 10 * (a - c))
  (h2 : F1 = 0 ∧ A = 0 ∧ v ≠ 0) :
  eccentricity_of_ellipse a c = 3 / 5 := by
sorry

end ellipse_eccentricity_l470_470642


namespace sum_nonperfect_square_l470_470102

theorem sum_nonperfect_square (p : ℕ) :
  let k : ℕ := 6 * p - 1 in
  ¬ ∃ (n : ℕ), n ^ 2 = ∑ i in Finset.range (k + 1), i * (i + 1) :=
by {
  let k := 6 * p - 1,
  sorry
}

end sum_nonperfect_square_l470_470102


namespace smallest_number_of_stamps_l470_470915

theorem smallest_number_of_stamps (m : ℕ) (W : ℕ → Prop) (number_of_proper_divisors : ∀ m, 7) : m = 36 := 
by
  sorry

end smallest_number_of_stamps_l470_470915


namespace geometric_shape_is_line_l470_470918

theorem geometric_shape_is_line (θ : ℝ) (h : θ = π / 4) : 
  ∃ (L : Set (ℝ × ℝ)), (∀ p ∈ L, ∃ r : ℝ, r * cos θ = p.1 ∧ r * sin θ = p.2) ∧ 
  ∀ p ∈ L, p.2 = p.1 := 
sorry

end geometric_shape_is_line_l470_470918


namespace total_crayons_correct_l470_470812

def wanda_crayons : ℕ := 62
def dina_crayons : ℕ := 28
def jacob_crayons : ℕ := dina_crayons - 2
def emma_crayons : ℕ := (2 * wanda_crayons) - 3
def xavier_crayons : ℕ := (Int.natAbs((jacob_crayons + dina_crayons) / 2)) ^ 3 - 7
def total_crayons_without_hannah : ℕ := wanda_crayons + dina_crayons + jacob_crayons + emma_crayons + xavier_crayons
def hannah_crayons : ℕ := total_crayons_without_hannah / 5 -- Equivalent to 20%
def total_crayons_with_hannah : ℕ := total_crayons_without_hannah + hannah_crayons

theorem total_crayons_correct : total_crayons_with_hannah = 23895 := by
  sorry

end total_crayons_correct_l470_470812


namespace simplify_expression_l470_470419

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  (x^2 - 4x + 3) / (x^2 - 6x + 8) / ( (x^2 - 6x + 5) / (x^2 - 8x + 15)) = 1 / ((x - 2) * (x - 4)) :=
by
  sorry

end simplify_expression_l470_470419


namespace normal_line_eq_l470_470829

theorem normal_line_eq (x : ℝ) (y : ℝ) : 
  let f := λ x, x^2 + 8 * real.sqrt x - 32 in
  let f' := λ x, 2 * x + 4 / (real.sqrt x) in
  (x = 4) → (y = f x) → (y = - x / 10 + 2 / 5) :=
sorry

end normal_line_eq_l470_470829


namespace terminal_side_quadrant_l470_470011

theorem terminal_side_quadrant (α : ℝ) (h : α = 2) : 
  90 < α * (180 / Real.pi) ∧ α * (180 / Real.pi) < 180 := 
by
  sorry

end terminal_side_quadrant_l470_470011


namespace find_a1_l470_470169

noncomputable def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a (n+1) + a n = 4*n

theorem find_a1 (a : ℕ → ℕ) (h : is_arithmetic_sequence a) : a 1 = 1 := by
  sorry

end find_a1_l470_470169


namespace find_value_of_k_l470_470719

noncomputable def value_of_k {m n : ℝ} (p q k : ℝ) : Prop :=
  (m + 2 = (m + p) / 2) ∧
  (n + k = (n + q) / 2) ∧
  (p = m + 4) ∧
  (q = n + 2k) ∧
  (m - 5n + 1 = 0) ∧
  (p - 5q + 1 = 0)

theorem find_value_of_k {m n p q k : ℝ} (h_curve : m^2 - 5*m*n + 2*n^2 + 7*m - 6*n + 3 = 0)
  (h_curve' : p^2 - 5*p*q + 2*q^2 + 7*p - 6*q + 3 = 0)
  (h_midpoint : m + 2 = (m + p) / 2)
  (h_midpoint' : n + k = (n + q) / 2)
  (h_line1 : m - 5*n + 1 = 0)
  (h_line2 : p - 5*q + 1 = 0) :
  k = 2 / 5 :=
sorry

end find_value_of_k_l470_470719


namespace tan_Z_right_triangle_l470_470709

theorem tan_Z_right_triangle 
  (X Y Z : Type) 
  (XY XZ YZ : ℝ) 
  (hXY : XY = 30) 
  (hYZ : YZ = 34) 
  (hXZ : XZ = real.sqrt (YZ^2 - XY^2)) 
  (hXZ_val : XZ = 16) 
  : real.tan Z = 15 / 8 := 
by 
  sorry

end tan_Z_right_triangle_l470_470709


namespace rectangle_area_l470_470073

-- Define the rectangular properties
variables {w l d x : ℝ}
def width (w : ℝ) : ℝ := w
def length (w : ℝ) : ℝ := 3 * w
def diagonal (w : ℝ) : ℝ := x

theorem rectangle_area (w x : ℝ) (hw : w ^ 2 + (3 * w) ^ 2 = x ^ 2) : w * 3 * w = 3 / 10 * x ^ 2 :=
by 
  sorry

end rectangle_area_l470_470073


namespace bill_after_late_charges_l470_470444

theorem bill_after_late_charges (initial_amount : ℝ) (first_late_charge : ℝ) (second_late_charge : ℝ) :
  initial_amount = 500 → first_late_charge = 0.02 → second_late_charge = 0.03 → 
  let amount_after_first_charge := initial_amount * (1 + first_late_charge) in
  let final_amount := amount_after_first_charge * (1 + second_late_charge) in
  final_amount = 525.30 :=
by
  intros
  let amount_after_first_charge := initial_amount * (1 + first_late_charge)
  let final_amount := amount_after_first_charge * (1 + second_late_charge)
  sorry

end bill_after_late_charges_l470_470444


namespace coordinates_of_point_l470_470638

theorem coordinates_of_point (x : ℝ) (P : ℝ × ℝ) (h : P = (1 - x, 2 * x + 1)) (y_axis : P.1 = 0) : P = (0, 3) :=
by
  sorry

end coordinates_of_point_l470_470638


namespace find_n_values_l470_470070

def numbers : Set ℝ := {4, 7, 8, 12}

theorem find_n_values : 
  (∀ n : ℝ, ∃ median: ℝ, 
    let new_numbers := {4, 7, 8, 12, n}
    let mean := (31 + n) / 5
    (median = 7 ∧ n < 7 ∧ mean = median) ∨
    (median = n ∧ 7 < n ∧ n < 8 ∧ mean = median) ∨
    (median = 8 ∧ n > 8 ∧ mean = median)
  ) → 
  ∃(count : ℕ), count = 2 := 
by 
  sorry

end find_n_values_l470_470070


namespace bridge_weight_excess_l470_470077

theorem bridge_weight_excess :
  ∀ (Kelly_weight Megan_weight Mike_weight : ℕ),
  Kelly_weight = 34 →
  Kelly_weight = 85 * Megan_weight / 100 →
  Mike_weight = Megan_weight + 5 →
  (Kelly_weight + Megan_weight + Mike_weight) - 100 = 19 :=
by
  intros Kelly_weight Megan_weight Mike_weight
  intros h1 h2 h3
  sorry

end bridge_weight_excess_l470_470077


namespace all_propositions_true_l470_470600

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

def is_stable_point (f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x

def fixed_points (f : ℝ → ℝ) : set ℝ := {x | is_fixed_point f x}

def stable_points (f : ℝ → ℝ) : set ℝ := {x | is_stable_point f x}

theorem all_propositions_true (f : ℝ → ℝ) :
  (fixed_points f ⊆ stable_points f) ∧
  (∃ (infinitely_many_stable_points : true), true) ∧
  (∀ x_0, is_stable_point f x_0 → (monotone f → is_fixed_point f x_0)) :=
by
  split
  · -- Proof of Prop_1: M ⊆ N
    sorry
  split
  · -- Proof of Prop_2: f(x) can have infinitely many stable points
    use true.intro
    sorry
  · -- Proof of Prop_3: If f(x) is monotonically increasing and x_0 is a stable point, then x_0 is a fixed point
    intros x_0 h_stable h_mono
    sorry

end all_propositions_true_l470_470600


namespace Mitch_weekly_net_earnings_l470_470304

variable (work_hours_mon_wed : ℕ) (hourly_rate_mon_wed : ℕ)
variable (work_hours_thu_fri : ℕ) (hourly_rate_thu_fri : ℕ)
variable (work_hours_weekend : ℕ) (hourly_rate_sat : ℕ) (hourly_rate_sun : ℕ)
variable (weekly_expenses : ℕ)

theorem Mitch_weekly_net_earnings :
  work_hours_mon_wed = 5 → hourly_rate_mon_wed = 3 →
  work_hours_thu_fri = 6 → hourly_rate_thu_fri = 4 →
  work_hours_weekend = 3 → hourly_rate_sat = 6 → hourly_rate_sun = 8 →
  weekly_expenses = 25 →
  let total_earning := (3 * (work_hours_mon_wed * hourly_rate_mon_wed)) +
                       (2 * (work_hours_thu_fri * hourly_rate_thu_fri)) +
                       (work_hours_weekend * hourly_rate_sat) +
                       (work_hours_weekend * hourly_rate_sun) in
  total_earning - weekly_expenses = 110 :=
by
  intros
  sorry

end Mitch_weekly_net_earnings_l470_470304


namespace count_positive_integers_in_interval_l470_470947

theorem count_positive_integers_in_interval : 
  (finset.filter 
    (λ x : ℤ, 150 ≤ x^2 ∧ x^2 ≤ 300 ∧ 0 < x) 
    (finset.Icc 1 18)).card = 5 := 
by {
  -- proof
  sorry
}

end count_positive_integers_in_interval_l470_470947


namespace wall_cost_equal_l470_470068

theorem wall_cost_equal (A B C : ℝ) (d_1 d_2 : ℝ) (h1 : A = B) (h2 : B = C) : d_1 = d_2 :=
by
  -- sorry is used to skip the proof
  sorry

end wall_cost_equal_l470_470068


namespace triangle_crease_length_l470_470880

theorem triangle_crease_length (A B : ℝ) (h_triangle : A = 6 ∧ B = 10 ∧ C = 8 )
  (fold : Point A falls on point B) :
  crease length = 15/4 :=
sorry

end triangle_crease_length_l470_470880


namespace P_roots_implies_Q_square_roots_l470_470736

noncomputable def P (x : ℝ) : ℝ := x^3 - 2 * x + 1

noncomputable def Q (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x - 1

theorem P_roots_implies_Q_square_roots (r : ℝ) (h : P r = 0) : Q (r^2) = 0 := sorry

end P_roots_implies_Q_square_roots_l470_470736


namespace prime_looking_numbers_count_l470_470905

theorem prime_looking_numbers_count:
  let prime_looking n := ¬ nat.prime n ∧ n ≠ 1 ∧ (2 ∣ n ∨ 3 ∣ n ∨ 7 ∣ n) = false in
  let num_primes_lt_1200 := 191 in
  let num_div2 := 599 in
  let num_div3 := 399 in
  let num_div7 := 171 in
  let num_div2_3 := 199 in
  let num_div2_7 := 85 in
  let num_div3_7 := 57 in
  let num_div2_3_7 := 28 in
  let inclusive_exclusive_count := num_div2 + num_div3 + num_div7 - num_div2_3 - num_div2_7 - num_div3_7 + num_div2_3_7 in
  let total_integers_lt_1200 := 1199 in
  let composite_count := total_integers_lt_1200 - num_primes_lt_1200 - 1 in
  prime_looking_numbers_count : 
    (composite_count - inclusive_exclusive_count) = 154 :=
  by sorry

end prime_looking_numbers_count_l470_470905


namespace distinct_triplets_count_l470_470050

theorem distinct_triplets_count :
  ∃ (count : ℕ), count = 440 ∧ ∀ n, n ≤ 600 →
  (⟦ n / 2 ⟧ ≠ ⟦ (n+1) / 2 ⟧ ∨ ⟦ n / 3 ⟧ ≠ ⟦ (n+1) / 3 ⟧ ∨ ⟦ n / 5 ⟧ ≠ ⟦ (n+1) / 5 ⟧) :=
begin
  sorry
end

end distinct_triplets_count_l470_470050


namespace final_people_not_on_ride_l470_470782

namespace FerrisWheel

-- Define initial conditions
def total_capacity := 168
def initial_people := 92
def leave_percentage := 10 / 100

-- Function to calculate people leaving after each rotation
def leave_after_each_rotation (people_remaining: ℕ) : ℕ :=
  Nat.round (leave_percentage * people_remaining)

-- Function to calculate people remaining after each rotation
def remaining_after_each_rotation (people_remaining: ℕ) : ℕ :=
  people_remaining - leave_after_each_rotation people_remaining - 56

-- Constants after each rotation
def after_first_rotation : ℕ := remaining_after_each_rotation initial_people
def after_second_rotation : ℕ := remaining_after_each_rotation after_first_rotation
def after_third_rotation : ℕ := remaining_after_each_rotation after_second_rotation

-- Theorem to prove the final number of people who won't get on the ride at all
theorem final_people_not_on_ride: after_third_rotation = 0 := by
  sorry

end FerrisWheel

end final_people_not_on_ride_l470_470782


namespace min_value_of_f_l470_470424

def f (x : ℝ) : ℝ :=
  (Real.cos (2 * x)) + 2 * (Real.sin x)

theorem min_value_of_f : (∃ x : ℝ, f x = -3) :=
by sorry

end min_value_of_f_l470_470424


namespace inequality_holds_for_all_reals_l470_470348

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470348


namespace geometric_sequence_a_n_l470_470631

theorem geometric_sequence_a_n (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = 4^(n - 1))
  (h2 : ∀ n, b n = 3*n - 1)
  (h3 : a 1 = 1)
  (h4 : a 5 = 256)
  (h5 : b 1 = 2)
  (h6 : 5 * S 5 = 2 * S 8)
  (hS : ∀ n, S n = n * (2 + (n - 1) * 3) / 2)
  :
  (∀ n, a n = 4^(n - 1)) ∧ 
  (∀ n, b n = 3*n - 1) ∧ 
  (∀ n, T n = ∑ i in Finset.range n, a i.succ * b i.succ)
:=
by
  sorry

end geometric_sequence_a_n_l470_470631


namespace chemical_b_percentage_solution_x_l470_470768

noncomputable theory

variables (B : ℝ) (x y : Type) 

-- Let x be a solution with 10% chemical a and B% chemical b
structure SolutionX :=
  (a_percentage b_percentage : ℝ)
  (h_x : a_percentage = 0.10)

-- Let y be a solution with 20% chemical a and 80% chemical b
structure SolutionY :=
  (a_percentage b_percentage : ℝ)
  (h_y : a_percentage = 0.20)
  (h_y_b : b_percentage = 0.80)

-- Mixture conditions
structure Mixture :=
  (a_percentage b_percentage : ℝ)
  (h_mix_a : a_percentage = 0.12)
  (h_mix_x_percentage : ℝ)
  (h_mix_x : h_mix_x_percentage = 0.80)
  (h_mix_y_percentage : ℝ)
  (h_mix_y : h_mix_y_percentage = 0.20)

-- Prove the percentage of chemical b in SolutionX is 0.90
theorem chemical_b_percentage_solution_x (Sx : SolutionX) (Sy : SolutionY) (M : Mixture) :
  Sx.b_percentage = 0.90 :=
sorry

end chemical_b_percentage_solution_x_l470_470768


namespace jack_walking_rate_l470_470042

variables (distance : ℝ) (time_hours : ℝ)
#check distance  -- ℝ (real number)
#check time_hours  -- ℝ (real number)

-- Define the conditions
def jack_distance : Prop := distance = 9
def jack_time : Prop := time_hours = 1 + 15 / 60

-- Define the statement to prove
theorem jack_walking_rate (h1 : jack_distance distance) (h2 : jack_time time_hours) :
  (distance / time_hours) = 7.2 :=
sorry

end jack_walking_rate_l470_470042


namespace cross_product_zero_of_self_subtract_l470_470207

def vec3 := ℝ × ℝ × ℝ

def cross_product (u v : vec3) : vec3 :=
  (u.2.2 * v.3 - u.3 * v.2.2, 
   u.3 * v.1 - u.1 * v.3,
   u.1 * v.2.2 - u.2.2 * v.1)

theorem cross_product_zero_of_self_subtract (u z : vec3)
  (H : cross_product u z = (-3, 7, 1)) :
  cross_product (u.1 - z.1, u.2.2 - z.2.2, u.3 - z.3)
                (u.1 - z.1, u.2.2 - z.2.2, u.3 - z.3) = (0, 0, 0) :=
by
  sorry

end cross_product_zero_of_self_subtract_l470_470207


namespace no_such_set_exists_l470_470149

theorem no_such_set_exists :
  ¬ ∃ (A : Finset ℕ), A.card = 11 ∧
  (∀ (s : Finset ℕ), s ⊆ A → s.card = 6 → ¬ 6 ∣ s.sum id) :=
sorry

end no_such_set_exists_l470_470149


namespace common_external_tangents_parallel_l470_470317

open EuclideanGeometry

/- Define the centers and radius of the circles -/
variables (O1 O2 : Point)
variable (R : ℝ)
variable (hR : R > 0)

theorem common_external_tangents_parallel
  (circle1 : Circle O1 R) 
  (circle2 : Circle O2 R) 
  (L1 L2 : Line) 
  (h1 : is_tangent circle1 L1 ∧ is_tangent circle2 L1) 
  (h2 : is_tangent circle1 L2 ∧ is_tangent circle2 L2) :
  are_parallel L1 (line_through O1 O2) ∧ are_parallel L2 (line_through O1 O2) :=
sorry

end common_external_tangents_parallel_l470_470317


namespace eval_f_function_l470_470745

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem eval_f_function : f (f (f (-1))) = Real.pi + 1 :=
  sorry

end eval_f_function_l470_470745


namespace question1_question2_question3_l470_470959

def conditions_a := (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f 3 = 2)

def f (a x : ℝ) := a*x - a + 1

theorem question1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 = 2) : a = 3 :=
sorry

def g (x : ℝ) := 3 * (x + 3) - 2

def h (x : ℝ) := (x - 7) / 3

theorem question2 : h (g x) = (x - 7) / 3 :=
sorry

lemma inequality_vld (h (x : ℝ) := (x - 7) / 3) (x : ℝ) :=
(h(x) + 2) * (h(x) + 2) ≤ h(x * x) + m + 2

theorem question3 (m : ℝ) : (1 ≤ x ∧ x ≤ 9) → inequality_vld h x → m ≥ 5 :=
sorry

end question1_question2_question3_l470_470959


namespace monotonicity_f_f_gt_lower_bound_l470_470664

-- Definition of the function
def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

-- Statement 1: Monotonicity discussion
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, (f a)' x < 0) ∧ 
  (a > 0 → ∀ x : ℝ,
    (f a)' x < 0 ∧ x < Real.log (1 / a) ∨
    (f a)' x > 0 ∧ x > Real.log (1 / a)) :=
sorry

-- Statement 2: Proof for f(x) > 2 ln a + 3/2 for a > 0
theorem f_gt_lower_bound (a x : ℝ) (ha : 0 < a) :
  f a x > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_f_f_gt_lower_bound_l470_470664


namespace tiles_per_row_l470_470390

noncomputable def area_square_room : ℕ := 400
noncomputable def side_length_feet : ℕ := nat.sqrt area_square_room
noncomputable def side_length_inches : ℕ := side_length_feet * 12
noncomputable def tile_size : ℕ := 8

theorem tiles_per_row : (side_length_inches / tile_size) = 30 := 
by
  have h1: side_length_feet = 20 := by sorry
  have h2: side_length_inches = 240 := by sorry
  have h3: side_length_inches / tile_size = 30 := by sorry
  exact h3

end tiles_per_row_l470_470390


namespace concyclic_min_fP_l470_470618

-- We define the necessary entities for our problem
variables (A B C D P E : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D] [EuclideanSpace P] [EuclideanSpace E]
-- Define the function f(P)
def f (P : EuclideanSpace P) (A B C D : EuclideanSpace A) (BC CA AB : ℝ) : ℝ :=
  PA P * BC + PD P * CA + PC P * AB

-- Problem statement part 1: Prove that P, A, B, C are concyclic when f(P) attains its minimum.
theorem concyclic (P A B C D : EuclideanSpace) (BC CA AB : ℝ) (h : ∀ P, f P A B C D = PA P * BC + PD P * CA + PC P * AB) :
  let f_min := arg_min (f P A B C D BC CA AB) in
  are_concyclic P A B C :=
sorry

-- Problem statement part 2: Find the minimum of f(P).
theorem min_fP (P A B C D E : EuclideanSpace) (O : Circle) 
(AB AE AC BC EC : ℝ)
(h1 : AE = (sqrt 3)/2 * AB)
(h2 : BC = (sqrt 3 - 1) * EC)
(h3 : angle ECA = 2 * angle ECB)
(h4 : tangent D A O) (h5 : tangent D C O) (h6 : AC = sqrt 2) :
  ∃ m, f P A B C D BC CA AB = sqrt 10 :=
sorry

end concyclic_min_fP_l470_470618


namespace solve_inequality_l470_470774

theorem solve_inequality (α x : ℝ) : 
  (α x^2 + (α - 2) * x - 2 ≥ 0) ↔ 
  (α = 0 ∧ x ≤ -1) ∨ 
  (α > 0 ∧ (x ≥ 2 / α ∨ x ≤ -1)) ∨ 
  (-2 < α ∧ α < 0 ∧ 2 / α ≤ x ∧ x ≤ -1) ∨ 
  (α = -2 ∧ x = -1) ∨ 
  (α < -2 ∧ -1 ≤ x ∧ x ≤ 2 / α) :=
by
  sorry

end solve_inequality_l470_470774


namespace range_of_m_l470_470196

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x else 1 - x

theorem range_of_m :
  {m : ℝ | f m > 1} = {m : ℝ | m < 0} ∪ {m : ℝ | m > Real.exp 1} :=
by
  sorry

end range_of_m_l470_470196


namespace inequality_inequality_l470_470363

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470363


namespace tiles_in_each_row_l470_470394

theorem tiles_in_each_row (area : ℝ) (tile_side : ℝ) (feet_to_inches : ℝ) (number_of_tiles : ℕ) :
  area = 400 ∧ tile_side = 8 ∧ feet_to_inches = 12 → number_of_tiles = 30 :=
by
  intros h
  cases h with h_area h_rest
  cases h_rest with h_tile_side h_feet_to_inches
  sorry

end tiles_in_each_row_l470_470394


namespace arithmetic_geometric_sequence_l470_470964

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a n = 2 + (n - 1) * d) → 
  d ≠ 0 → 
  (2, 2 + d, 2 + 4 * d) ∈ set_of (λ t : ℝ × ℝ × ℝ, t.snd ^ 2 = t.fst * t.snd.snd) → 
  d = 4 :=
sorry

end arithmetic_geometric_sequence_l470_470964


namespace no_four_consecutive_powers_l470_470922

/-- 
  There do not exist four consecutive natural numbers 
  such that each of them is a power (greater than 1) of another natural number.
-/
theorem no_four_consecutive_powers : 
  ¬ ∃ (n : ℕ), (∀ (i : ℕ), i < 4 → ∃ (a k : ℕ), k > 1 ∧ n + i = a^k) := sorry

end no_four_consecutive_powers_l470_470922


namespace number_of_selection_methods_l470_470797

theorem number_of_selection_methods 
    (n : ℕ)
    (k : ℕ) 
    (h_n : n = 3) 
    (h_k : k = 5) : 
    (n ^ k) = 243 := 
by 
    rw [h_n, h_k]
    exact Nat.pow_succ 3 4 9

end number_of_selection_methods_l470_470797


namespace parallel_lines_condition_suff_not_nec_l470_470608

theorem parallel_lines_condition_suff_not_nec 
  (a : ℝ) : (a = -2) → 
  (∀ x y : ℝ, ax + 2 * y - 1 = 0) → 
  (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) → 
  (∀ x1 y1 x2 y2 : ℝ, ((a = -2) → (2 * y1 - 2 * x1 = 1) → (y2 - x2 = -4) → (x1 = x2 → y1 = y2))) ∧ 
  (∃ b : ℝ, ¬ (b = -2) ∧ ((2 * y1 - b * x1 = 1) → (x2 - (b + 1) * y2 = -4) → ¬(x1 = x2 → y1 = y2)))
   :=
by
  sorry

end parallel_lines_condition_suff_not_nec_l470_470608


namespace point_in_first_quadrant_l470_470427

def i : ℂ := complex.I

theorem point_in_first_quadrant :
  let z := 1 / (1 - i) in
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end point_in_first_quadrant_l470_470427


namespace trajectory_of_vertex_C_l470_470168

theorem trajectory_of_vertex_C :
  ∀ (C : ℝ × ℝ), let A := (-4 : ℝ, 0 : ℝ) 
                    B := (4 : ℝ, 0 : ℝ) 
                    a := real.sqrt ((C.fst + 4)^2 + C.snd^2) 
                    b := real.sqrt ((C.fst - 4)^2 + C.snd^2) in
  (2 * real.sin a + real.sin (real.atan (C.snd / (C.fst + 4))) = 2 * real.sin b) →
  (C.fst^2 / 4 - C.snd^2 / 12 = 1 ∧ C.fst > 2) := 
sorry

end trajectory_of_vertex_C_l470_470168


namespace determine_sum_of_digits_l470_470689

theorem determine_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10)
  (h : ∃ a b c d : ℕ, 
       a = 30 + x ∧ b = 10 * y + 4 ∧
       c = (a * (b % 10)) % 100 ∧ 
       d = (a * (b % 10)) / 100 ∧ 
       10 * d + c = 156) :
  x + y = 13 :=
by
  sorry

end determine_sum_of_digits_l470_470689


namespace probability_no_adjacent_birch_is_correct_l470_470868

noncomputable def count_maple : ℕ := 4
noncomputable def count_oak : ℕ := 5
noncomputable def count_birch : ℕ := 6

def slots (n m k : ℕ) : ℕ := n + m + 1

def ways_to_place_birch (slots k : ℕ) : ℕ := Nat.choose slots k

def total_arrangements (n m k : ℕ) : ℕ := 
  Nat.choose (n + m + k) k * Nat.choose (n + m) n

def probability_no_adjacent_birch (n m k : ℕ) : ℚ :=
  ways_to_place_birch (slots n m k) k / (total_arrangements n m k : ℚ)

theorem probability_no_adjacent_birch_is_correct :
  probability_no_adjacent_birch count_maple count_oak count_birch = 1 / 3003 :=
by
  unfold count_maple count_oak count_birch slots ways_to_place_birch total_arrangements probability_no_adjacent_birch
  sorry

end probability_no_adjacent_birch_is_correct_l470_470868


namespace cost_per_pack_l470_470434

variable (total_amount : ℕ) (number_of_packs : ℕ)

theorem cost_per_pack (h1 : total_amount = 132) (h2 : number_of_packs = 11) : 
  total_amount / number_of_packs = 12 := by
  sorry

end cost_per_pack_l470_470434


namespace scaled_determinant_l470_470952

variable {a b c d : ℝ}

theorem scaled_determinant (h : ∣↑⟨a, b, c, d⟩∣ = 7) : ∣↑⟨3 * a, 3 * b, 3 * c, 3 * d⟩∣ = 63 := by
  sorry

end scaled_determinant_l470_470952


namespace chuck_play_area_l470_470906

-- Definitions of the given conditions
def shed_length : ℝ := 4
def shed_width : ℝ := 3
def leash_length : ℝ := 5

-- Main theorem to prove
theorem chuck_play_area : 
  let area : ℝ := (3/4) * Real.pi * leash_length^2 +
                  (1/4) * Real.pi * 1^2 +
                  (1/4) * Real.pi * 2^2 
  in area = 20 * Real.pi :=
by
  sorry

end chuck_play_area_l470_470906


namespace percentage_decrease_l470_470863

theorem percentage_decrease (purchase_price selling_price decrease gross_profit : ℝ)
  (h_purchase : purchase_price = 81)
  (h_markup : selling_price = purchase_price + 0.25 * selling_price)
  (h_gross_profit : gross_profit = 5.40)
  (h_decrease : decrease = 108 - 102.60) :
  (decrease / 108) * 100 = 5 :=
by sorry

end percentage_decrease_l470_470863


namespace distance_downstream_is_96_l470_470498

-- Definitions and conditions given in the problem:
def V_b : ℝ := 20  -- Speed of the boat in still water in km/h
def time_downstream : ℝ := 3  -- Time travelled downstream in hours
def distance_upstream : ℝ := 88  -- Distance travelled upstream in km
def time_upstream : ℝ := 11  -- Time travelled upstream in hours

-- Intermediate calculations to find V_up, V_s, and V_down
def V_up : ℝ := distance_upstream / time_upstream  -- Effective speed upstream
def V_s : ℝ := V_b - V_up  -- Speed of the stream
def V_down : ℝ := V_b + V_s  -- Effective speed downstream

-- Distance covered downstream
def distance_downstream : ℝ := V_down * time_downstream

-- The theorem to prove
theorem distance_downstream_is_96 : distance_downstream = 96 := 
by
  sorry

end distance_downstream_is_96_l470_470498


namespace smallest_m_l470_470822

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 12 * p * p - m * p + 432 = 0) (h_sum : p + q = m / 12) (h_prod : p * q = 36) :
  m = 144 :=
by
  sorry

end smallest_m_l470_470822


namespace radius_of_tangent_circle_l470_470502

theorem radius_of_tangent_circle (k : ℝ) (h : k < -6) :
  ∃ (r : ℝ), (r = 6*Real.sqrt(2) + 6) ∧
      ∀ (x y : ℝ), (x = 0 ∧ y = k) →
      (∃ t : ℝ, y = x + t * Real.sqrt(2) ∨ y = -x + t * Real.sqrt(2) ∨ y = -6) →
      Real.sqrt((x - 0)^2 + (y - k)^2) = r := 
sorry

end radius_of_tangent_circle_l470_470502


namespace minimum_selection_integers_l470_470998

theorem minimum_selection_integers (s : Finset ℕ) (h : s ⊆ Finset.range 21 ∧ s.card = 11) :
  ∃ a b ∈ s, a - b = 2 ∨ b - a = 2 :=
by
  sorry

end minimum_selection_integers_l470_470998


namespace ones_digit_7_pow_35_l470_470455

theorem ones_digit_7_pow_35 : (7^35) % 10 = 3 := 
by
  sorry

end ones_digit_7_pow_35_l470_470455


namespace shortest_chord_length_l470_470432

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 2 = 0

noncomputable def center : ℝ × ℝ := ⟨1, -1⟩

noncomputable def radius : ℝ := 2

noncomputable def point : ℝ × ℝ := ⟨0, 0⟩

theorem shortest_chord_length :
  let d := real.sqrt ((center.1 - point.1) ^ 2 + (center.2 - point.2) ^ 2) in
  let chord_len := 2 * real.sqrt (radius ^ 2 - d ^ 2) in
  chord_len = 2 * real.sqrt 2 :=
by
  sorry

end shortest_chord_length_l470_470432


namespace area_of_square_l470_470309

theorem area_of_square (s : ℝ) (A B C D E F : Point)
  (h_square : is_square A B C D s)
  (hE_on_AD : E ∈ line_segment A D)
  (hF_on_BC : F ∈ line_segment B C)
  (hBE : dist B E = 30)
  (hEF : dist E F = 30)
  (hFD : dist F D = 30) :
  s = 22.5 → s^2 = 506.25 :=
by
  sorry

end area_of_square_l470_470309


namespace sum_of_perimeters_eq_two_l470_470528

theorem sum_of_perimeters_eq_two
  (side length square : ℝ) (distance parallel_lines : ℝ)
  (intersection_points : (ℝ × ℝ) → Prop)
  (P1 P2 : (ℝ × ℝ))
  (triangle1 triangle2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) :
  side length square = 1 →
  distance parallel_lines = 1 →
  intersection_points P1 →
  intersection_points P2 →
  sum_of_perimeters triangle1 triangle2 = 2 :=
by
  sorry

end sum_of_perimeters_eq_two_l470_470528


namespace evaluate_series_l470_470578

theorem evaluate_series :
  (∑ n in Finset.range 1008, (1 / (2 * n + 1) - 1 / (2 * n + 1)) / (1 / (2 * n + 1) * 1 / (2 * n) * 1 / (2 * n + 1))) = 2034144 :=
by
  sorry

end evaluate_series_l470_470578


namespace domain_of_f_l470_470572

def f (x : ℝ) : ℝ := real.cbrt (2 * x - 3) + real.sqrt (9 - x)

theorem domain_of_f :
  { x : ℝ | -∞ < x ∧ x ≤ 9 } = { x : ℝ | ∃ y, f y = f x } :=
by
  sorry

end domain_of_f_l470_470572


namespace angle_of_inclination_l470_470118

theorem angle_of_inclination : 
  let line_eq := ∀ x y : ℝ, x - real.sqrt 3 * y + 6 = 0 
  ∃ θ : ℝ, θ = real.arctan (real.sqrt 3 / 3) ∧ θ = 30 :=
by
  let line_eq := ∀ x y : ℝ, x - real.sqrt 3 * y + 6 = 0
  have slope_def : slope = real.sqrt 3 / 3 := sorry
  have theta_def : θ = real.arctan slope := sorry
  use θ
  split
  · have θ_is_arctan : θ = real.arctan (real.sqrt 3 / 3) := sorry
  · have θ_is_30_deg : θ = 30 := sorry

end angle_of_inclination_l470_470118


namespace sequence_a_four_l470_470199

-- Definitions given in the problem
def a : ℕ → ℚ
| 0 := -2
| (n+1) := 2 + (2 * a n) / (1 - a n)

-- Statement of the problem to prove a4 = -2/5
theorem sequence_a_four : a 3 = -2/5 :=
sorry

end sequence_a_four_l470_470199


namespace max_value_cauchy_schwarz_l470_470180

noncomputable def max_value_sqrt_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) : ℝ :=
  sqrt (a + 1) + sqrt (b + 1) + sqrt (c + 1)

theorem max_value_cauchy_schwarz :
  ∃ v, (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a + b + c = 1 → sqrt (a + 1) + sqrt (b + 1) + sqrt (c + 1) ≤ v)
    ∧ (∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 ∧ sqrt (a + 1) + sqrt (b + 1) + sqrt (c + 1) = v) :=
  sorry

end max_value_cauchy_schwarz_l470_470180


namespace complement_A_in_U_l470_470200

open Set

-- Definitions for sets
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- The proof goal: prove that the complement of A in U is {4}
theorem complement_A_in_U : (U \ A) = {4} := by
  sorry

end complement_A_in_U_l470_470200


namespace num_possible_D_values_is_six_l470_470715

noncomputable def numberOfPossibleDValues : Nat :=
  if ∃ (A B C D : Nat), A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
    (A + B + 1 = D) ∧ (C + D = D + 1) ∧ (C = 1) then 
    6 
  else 
    0

theorem num_possible_D_values_is_six : numberOfPossibleDValues = 6 := 
by 
  sorry

end num_possible_D_values_is_six_l470_470715


namespace linen_tablecloth_cost_l470_470306

def num_tables : ℕ := 20
def cost_per_place_setting : ℕ := 10
def num_place_settings_per_table : ℕ := 4
def cost_per_rose : ℕ := 5
def num_roses_per_centerpiece : ℕ := 10
def cost_per_lily : ℕ := 4
def num_lilies_per_centerpiece : ℕ := 15
def total_decoration_cost : ℕ := 3500

theorem linen_tablecloth_cost :
  (total_decoration_cost - (num_tables * num_place_settings_per_table * cost_per_place_setting + num_tables * (num_roses_per_centerpiece * cost_per_rose + num_lilies_per_centerpiece * cost_per_lily))) / num_tables = 25 :=
  sorry

end linen_tablecloth_cost_l470_470306


namespace set_intersection_l470_470749

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2, 5}
noncomputable def B : Set ℕ := {x ∈ U | (3 / (2 - x) + 1 ≤ 0)}
noncomputable def C_U_B : Set ℕ := U \ B

theorem set_intersection : A ∩ C_U_B = {1, 2} :=
by {
  sorry
}

end set_intersection_l470_470749


namespace total_candies_in_store_l470_470859

-- Define the quantities of chocolates in each box
def box_chocolates_1 := 200
def box_chocolates_2 := 320
def box_chocolates_3 := 500
def box_chocolates_4 := 500
def box_chocolates_5 := 768
def box_chocolates_6 := 768

-- Define the quantities of candies in each tub
def tub_candies_1 := 1380
def tub_candies_2 := 1150
def tub_candies_3 := 1150
def tub_candies_4 := 1720

-- Sum of all chocolates and candies
def total_chocolates := box_chocolates_1 + box_chocolates_2 + box_chocolates_3 + box_chocolates_4 + box_chocolates_5 + box_chocolates_6
def total_candies := tub_candies_1 + tub_candies_2 + tub_candies_3 + tub_candies_4
def total_store_candies := total_chocolates + total_candies

theorem total_candies_in_store : total_store_candies = 8456 := by
  sorry

end total_candies_in_store_l470_470859


namespace length_BC_fraction_AD_l470_470311

theorem length_BC_fraction_AD 
  (A B C D E : Point)
  [LiesOn B A D]
  [LiesOn C A D]
  [LiesOn E A D]
  (h1 : length_AB B D A = 3 * length_BD B D)
  (h2 : length_AC C D A = 7 * length_CD C D)
  (h3 : length_DE D E = 2 * length_CE C E)
  : length_BC B C / length_AD A D = 1 / 8 := 
by
  sorry

end length_BC_fraction_AD_l470_470311


namespace exponent_sum_l470_470927

theorem exponent_sum (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^123 + i^223 + i^323 = -3 * i :=
by
  sorry

end exponent_sum_l470_470927


namespace simplify_fraction_expression_l470_470479

theorem simplify_fraction_expression (d : ℤ) :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 :=
by
  -- skip the proof by adding sorry
  sorry

end simplify_fraction_expression_l470_470479


namespace grid_sum_C_D_l470_470561

theorem grid_sum_C_D :
  ∃ (C D : ℕ) (grid : matrix (fin 4) (fin 4) ℕ),
    (∀ i, ∃! a, grid i a = 1 ∧ grid i a = 2 ∧ grid i a = 3 ∧ grid i a = 4) ∧
    (∀ j, ∃! b, grid b j = 1 ∧ grid b j = 2 ∧ grid b j = 3 ∧ grid b j = 4) ∧
    grid 0 0 = 1 ∧ grid 0 3 = 4 ∧ grid 1 1 = 2 ∧ grid 2 3 = 3 ∧ grid 3 2 = D ∧
    C = grid 1 3 ∧ 
    C + D = 6 :=
sorry

end grid_sum_C_D_l470_470561


namespace inequality_inequality_l470_470360

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470360


namespace inequality_proof_l470_470351

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470351


namespace original_number_is_perfect_square_l470_470024

variable (n : ℕ)

theorem original_number_is_perfect_square
  (h1 : n = 1296)
  (h2 : ∃ m : ℕ, (n + 148) = m^2) : ∃ k : ℕ, n = k^2 :=
by
  sorry

end original_number_is_perfect_square_l470_470024


namespace problem_1_solution_problem_2_solution_l470_470669

def f (x : ℝ) : ℝ := abs (x + 1)
def g (x : ℝ) (a : ℝ) : ℝ := 2 * abs x + a

def problem_1_condition := a = -1
def problem_2_condition (x0 : ℝ) := f x0 ≥ (g x0 a) / 2

theorem problem_1_solution : problem_1_condition →
  {x : ℝ | f x ≤ g x (-1)} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} :=
sorry

theorem problem_2_solution :
  (∃ x0 : ℝ, problem_2_condition x0) → a ≤ 2 :=
sorry

end problem_1_solution_problem_2_solution_l470_470669


namespace decimal_digits_l470_470683

theorem decimal_digits (a b c : ℕ) (h1 : a = 2^7) (h2 : b = 5^5) (h3 : c = 7^3) :
  (how_many_digits_right_of_decimal a b c = 5) :=
sorry

def how_many_digits_right_of_decimal (a b c : ℕ) : ℕ :=
  let result := (a : ℝ) / ((b * c) : ℝ)
  count_digits_right_of_decimal result

def count_digits_right_of_decimal (x : ℝ) : ℕ :=
  let s := to_string x
  let decimal_part := s.split (λ c, c = '.').tail.head
  decimal_part.length

end decimal_digits_l470_470683


namespace value_of_c_l470_470234

theorem value_of_c (b c : ℝ) (h1 : (x : ℝ) → (x + 4) * (x + b) = x^2 + c * x + 12) : c = 7 :=
by
  have h2 : 4 * b = 12 := by sorry
  have h3 : b = 3 := by sorry
  have h4 : c = b + 4 := by sorry
  rw [h3] at h4
  rw [h4]
  exact by norm_num

end value_of_c_l470_470234


namespace solve_for_x_l470_470914

def tensor (a b : ℝ) : ℝ := 1/a + 1/b

theorem solve_for_x (x : ℝ) (h₀ : x ≠ 0) (h₁ : x + 1 ≠ 0) (h₂ : tensor (x + 1) x = 2) : 
  x = Real.sqrt 2 / 2 ∨ x = -Real.sqrt 2 / 2 := by
  sorry

end solve_for_x_l470_470914


namespace part1_part2_l470_470612

-- Part (1): Proving the range of x when a = 1
theorem part1 (x : ℝ) : (x^2 - 6 * 1 * x + 8 < 0) ∧ (x^2 - 4 * x + 3 ≤ 0) ↔ 2 < x ∧ x ≤ 3 := 
by sorry

-- Part (2): Proving the range of a when p is a sufficient but not necessary condition for q
theorem part2 (a : ℝ) : (∀ x : ℝ, (x^2 - 6 * a * x + 8 * a^2 < 0 → x^2 - 4 * x + 3 ≤ 0) 
  ∧ (∃ x : ℝ, x^2 - 4 * x + 3 ≤ 0 ∧ x^2 - 6 * a * x + 8 * a^2 ≥ 0)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end part1_part2_l470_470612


namespace probability_of_exactly_four_treasures_with_no_only_traps_l470_470071

noncomputable def probability_exactly_four_treasures_no_only_traps : ℚ :=
  let p_treasure : ℚ := 1 / 5
  let p_traps_no_treasure : ℚ := 1 / 10
  let p_both : ℚ := 1 / 10
  let p_neither : ℚ := 3 / 5
  let p_treasure_combined : ℚ := p_treasure + p_both
  let num_islands := 8
  let num_treasure_islands := 4
  let p_treasure_and_notraps := binomial num_islands num_treasure_islands * p_treasure_combined^num_treasure_islands * p_neither^(num_islands - num_treasure_islands)
  p_treasure_and_notraps

theorem probability_of_exactly_four_treasures_with_no_only_traps : 
  probability_exactly_four_treasures_no_only_traps = 91854 / 1250000 :=
sorry

end probability_of_exactly_four_treasures_with_no_only_traps_l470_470071


namespace triangle_area_l470_470701

theorem triangle_area (a b C : ℝ) (a_eq : a = 2) (b_eq : b = 3) (C_eq : C = π / 6) : 
  let S := (1 / 2) * a * b * real.sin C in 
  S = 3 / 2 := 
by 
  sorry

end triangle_area_l470_470701


namespace charge_y1_charge_y2_cost_effective_range_call_duration_difference_l470_470712

def y1 (x : ℕ) : ℝ :=
  if x ≤ 600 then 30 else 0.1 * x - 30

def y2 (x : ℕ) : ℝ :=
  if x ≤ 1200 then 50 else 0.1 * x - 70

theorem charge_y1 (x : ℕ) :
  (x ≤ 600 → y1 x = 30) ∧ (x > 600 → y1 x = 0.1 * x - 30) :=
by sorry

theorem charge_y2 (x : ℕ) :
  (x ≤ 1200 → y2 x = 50) ∧ (x > 1200 → y2 x = 0.1 * x - 70) :=
by sorry

theorem cost_effective_range (x : ℕ) :
  (0 ≤ x) ∧ (x < 800) → y1 x < y2 x :=
by sorry

noncomputable def call_time_xiaoming : ℕ := 1300
noncomputable def call_time_xiaohua : ℕ := 900

theorem call_duration_difference :
  call_time_xiaoming = call_time_xiaohua + 400 :=
by sorry

end charge_y1_charge_y2_cost_effective_range_call_duration_difference_l470_470712


namespace candy_per_person_division_l470_470014

noncomputable def total_candy_bars := 5.0
noncomputable def total_people := 3.0
noncomputable def expected_candy_bars_per_person := 1.67

theorem candy_per_person_division :
  (total_candy_bars / total_people) ≈ expected_candy_bars_per_person :=
by
  sorry

end candy_per_person_division_l470_470014


namespace events_d_mutually_exclusive_not_opposite_l470_470870

def are_mutually_exclusive {Ω : Type} (A B : set Ω) : Prop :=
  ∀ ω, ω ∈ A → ω ∉ B

def are_not_opposite {Ω : Type} (A B : set Ω) : Prop :=
  A ∪ B ≠ set.univ

def event_exactly_one_girl (Ω : Type) [finite Ω] (students : set Ω) : set Ω :=
  {s | s.card = 1 ∧ ∃ x ∈ s, x ∈ students}

def event_exactly_two_girls (Ω : Type) [finite Ω] (students : set Ω) : set Ω :=
  {s | s.card = 2 ∧ ∀ x ∈ s, x ∈ students}

theorem events_d_mutually_exclusive_not_opposite
    (Ω : Type) [finite Ω] (students : set Ω) :
  are_mutually_exclusive (event_exactly_one_girl Ω students)
                          (event_exactly_two_girls Ω students) ∧
  are_not_opposite (event_exactly_one_girl Ω students)
                   (event_exactly_two_girls Ω students) :=
sorry

end events_d_mutually_exclusive_not_opposite_l470_470870


namespace isosceles_triangle_angle_in_circle_l470_470543

theorem isosceles_triangle_angle_in_circle :
  ∀ (A B C D : Point) (k : ℝ),
    (isosceles_triangle A B C ∧ inscribed_in_circle A B C ∧ tangents_intersect_at D B C ∧ 
      ∠ ABC = ∠ ACB ∧ ∠ ABC = 3 * ∠ D ∧ ∠ BAC = k * π)
    → k = 5 / 11 :=
by
  sorry

end isosceles_triangle_angle_in_circle_l470_470543


namespace PM_perpendicular_y_axis_range_area_triangle_PAB_l470_470548

noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem PM_perpendicular_y_axis
  (p : ℝ) (hp : 0 < p) (P : ℝ × ℝ) (hP : P.1 < 0)
  (A B : ℝ × ℝ) (hA : A.2 ^ 2 = 2 * p * A.1) (hB : B.2 ^ 2 = 2 * p * B.1)
  (PA_mid : ℝ × ℝ := midpoint P A)
  (PB_mid : ℝ × ℝ := midpoint P B)
  (hPA_mid : PA_mid.2 ^ 2 = 2 * p * PA_mid.1)
  (hPB_mid : PB_mid.2 ^ 2 = 2 * p * PB_mid.1)
  (M : ℝ × ℝ := midpoint A B) :
  M.1 = P.1 :=
sorry

theorem range_area_triangle_PAB
  (p : ℝ) (hp : 0 < p) (P : ℝ × ℝ) (hP : P.1 < 0 ∧ P.1 ^ 2 + P.2 ^ 2 / (2 * p) = 1)
  (A B : ℝ × ℝ) (hA : A.2 ^ 2 = 2 * p * A.1) (hB : B.2 ^ 2 = 2 * p * B.1) :
  2.577 ∈ {(1 / 2) * abs (P.1 * (A.2 - B.2) + A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2))} :=
sorry

end PM_perpendicular_y_axis_range_area_triangle_PAB_l470_470548


namespace integral_evaluation_l470_470853

noncomputable def integral_answer : ℝ :=
  ∫ x in 1..2, ((Real.exp x) + 1 / x)

theorem integral_evaluation :
  integral_answer = Real.exp 2 - Real.exp 1 + Real.log 2 :=
by
  sorry

end integral_evaluation_l470_470853


namespace work_completion_in_16_days_l470_470038

theorem work_completion_in_16_days (A B : ℕ) :
  (1 / A + 1 / B = 1 / 40) → (10 * (1 / A + 1 / B) = 1 / 4) →
  (12 * 1 / A = 3 / 4) → A = 16 :=
by
  intros h1 h2 h3
  -- Proof is omitted by "sorry".
  sorry

end work_completion_in_16_days_l470_470038


namespace inequality_proof_l470_470356

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470356


namespace find_number_l470_470574

theorem find_number (x : ℝ) (h : x / 0.05 = 900) : x = 45 :=
by sorry

end find_number_l470_470574


namespace sum_of_dice_not_19_l470_470537

theorem sum_of_dice_not_19 (a b c d : ℕ) (h1 : a ∈ {1, 2, 3, 4, 5, 6}) (h2 : b ∈ {1, 2, 3, 4, 5, 6}) (h3 : c ∈ {1, 2, 3, 4, 5, 6}) (h4 : d ∈ {1, 2, 3, 4, 5, 6}) (h_product : a * b * c * d = 216) :
  a + b + c + d ≠ 19 :=
sorry

end sum_of_dice_not_19_l470_470537


namespace tan_half_alpha_third_quadrant_sine_cos_expression_l470_470856

-- Problem (1): Proof for tan(α/2) = -5 given the conditions
theorem tan_half_alpha_third_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.sin α = -5/13) :
  Real.tan (α / 2) = -5 := by
  sorry

-- Problem (2): Proof for sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5 given the condition
theorem sine_cos_expression (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 := by
  sorry

end tan_half_alpha_third_quadrant_sine_cos_expression_l470_470856


namespace correct_proposition_l470_470474

-- Define the propositions as Lean 4 statements.
def PropA (a : ℝ) : Prop := a^4 + a^2 = a^6
def PropB (a : ℝ) : Prop := (-2 * a^2)^3 = -6 * a^8
def PropC (a : ℝ) : Prop := 6 * a - a = 5
def PropD (a : ℝ) : Prop := a^2 * a^3 = a^5

-- The main theorem statement that only PropD is true.
theorem correct_proposition (a : ℝ) : ¬ PropA a ∧ ¬ PropB a ∧ ¬ PropC a ∧ PropD a :=
by
  sorry

end correct_proposition_l470_470474


namespace S7_is_28_l470_470963

variables {a_n : ℕ → ℤ} -- Sequence definition
variables {S_n : ℕ → ℤ} -- Sum of the first n terms

-- Define an arithmetic sequence condition
def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Given conditions
axiom sum_condition : a_n 2 + a_n 4 + a_n 6 = 12
axiom sum_formula (n : ℕ) : S_n n = n * (a_n 1 + a_n n) / 2
axiom arith_seq : is_arithmetic_sequence a_n

-- The statement to be proven
theorem S7_is_28 : S_n 7 = 28 :=
sorry

end S7_is_28_l470_470963


namespace asha_gift_amount_l470_470549

-- Define the conditions
def borrowed_from_brother := 20
def borrowed_from_father := 40
def borrowed_from_mother := 30
def savings := 100
def remaining_money := 65
def fraction_remaining := (1:ℝ) / 4

-- Define the total money she has as all borrowed money plus savings
def initial_money := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + savings

-- Define the proportion of money she spent
def spent_fraction := (3:ℝ) / 4

-- The total money after she received the gift
axiom total_money_with_gift : ℝ
axiom granny_gift : ℝ

theorem asha_gift_amount :
  total_money_with_gift = initial_money + granny_gift →
  remaining_money = fraction_remaining * total_money_with_gift →
  granny_gift = 70 :=
begin
  intros h1 h2,
  sorry
end

end asha_gift_amount_l470_470549


namespace coplanar_points_count_l470_470081

-- Define a structure to represent the tetrahedron
structure Tetrahedron :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)
  (AB_midpoint : Point)
  (AC_midpoint : Point)
  (AD_midpoint : Point)
  (BC_midpoint : Point)
  (BD_midpoint : Point)
  (CD_midpoint : Point)

-- The theorem we need to prove
theorem coplanar_points_count (T : Tetrahedron) : 
  ∃ pts : Finset Point, pts.card = 3 ∧ (pts ∪ {T.A}).card = 4 ∧
  pts.subset ({T.B, T.C, T.D, T.AB_midpoint, T.AC_midpoint, T.AD_midpoint, T.BC_midpoint, T.BD_midpoint, T.CD_midpoint} : Finset Point) ∧
  ((∃ a b c : Point, pts = {a, b, c}) → (coplanar ({T.A, a, b, c} : Finset Point))) ∧
  (pts.card = 3) ≠ 33 := sorry

end coplanar_points_count_l470_470081


namespace derivative_f_eq_l470_470135

noncomputable def f (x : ℝ) : ℝ :=
  (7^x * (3 * Real.sin (3 * x) + Real.cos (3 * x) * Real.log 7)) / (9 + Real.log 7 ^ 2)

theorem derivative_f_eq :
  ∀ x : ℝ, deriv f x = 7^x * Real.cos (3 * x) :=
by
  intro x
  sorry

end derivative_f_eq_l470_470135


namespace order_of_magnitudes_l470_470965

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h_odd : ∀ x, f(-x) = -f(x))
variable (h_deriv : ∀ x ≠ 0, f'(x) + f(x) / x > 0)

noncomputable def a : ℝ := (1/3) * f (1/3)
noncomputable def b : ℝ := -3 * f (-3)
noncomputable def c : ℝ := (Real.log (1/3)) * f (Real.log (1/3))

theorem order_of_magnitudes : a f < c f < b f := 
by sorry

end order_of_magnitudes_l470_470965


namespace remainder_of_number_divided_by_39_l470_470861

theorem remainder_of_number_divided_by_39 
  (N : ℤ) 
  (k m : ℤ) 
  (h₁ : N % 195 = 79) 
  (h₂ : N % 273 = 109) : 
  N % 39 = 1 :=
by 
  sorry

end remainder_of_number_divided_by_39_l470_470861


namespace count_satisfying_integers_correct_l470_470684

noncomputable def count_satisfying_integers : ℕ :=
  (Finset.filter (λ (n : ℤ), (n - 3) * (n + 5) * (n + 9) < 0)
    (Finset.Icc (-13 : ℤ) 13)).card

theorem count_satisfying_integers_correct : count_satisfying_integers = 11 :=
by sorry

end count_satisfying_integers_correct_l470_470684


namespace max_value_of_3x_plus_4y_on_curve_C_l470_470674

theorem max_value_of_3x_plus_4y_on_curve_C :
  ∀ (x y : ℝ),
  (∃ (ρ θ : ℝ), ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (P : ℝ × ℝ) →
  (P = (x, y)) →
  3 * x + 4 * y ≤ Real.sqrt 145 ∧ ∃ (α : ℝ), 0 ≤ α ∧ α < 2 * Real.pi ∧ 3 * x + 4 * y = Real.sqrt 145 := 
by
  intros x y h_exists P hP
  sorry

end max_value_of_3x_plus_4y_on_curve_C_l470_470674


namespace math_problem_l470_470053

theorem math_problem : (3.14 - Real.pi)^0 - 2^(-1) = 1 / 2 := by
  sorry

end math_problem_l470_470053


namespace percentage_not_even_l470_470490

variable (S : Finset ℕ)
variable (T : ℕ) -- Total number of elements in set S
variable (E : ℕ) -- Total number of even numbers in set S

/-- 36% of the numbers in S are even multiples of 3 -/
def even_multiples_of_3 (S : Finset ℕ) : ℕ := 36 * T / 100

/-- 40% of the even numbers in S are not multiples of 3 -/
def even_non_multiples_of_3 (E : ℕ) : ℕ := 40 * E / 100

/-- Calculate the percentage of numbers in S that are not even integers -/
theorem percentage_not_even (h1 : even_multiples_of_3 S = 36 * T / 100)
  (h2 : even_non_multiples_of_3 E = 40 * E / 100) : 
  ((T - E) / T : ℚ) = 0.4 := by
  sorry

end percentage_not_even_l470_470490


namespace inequality_proof_l470_470352

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470352


namespace smallest_value_complex_expression_min_l470_470731

noncomputable def smallest_value_complex_expression (a b c d : ℤ) (ζ : ℂ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
(h2 : ζ^4 = 1) (h3 : ζ ≠ 1) : ℝ := 
|a + b * ζ + c * ζ^2 + d * ζ^3|

theorem smallest_value_complex_expression_min : ∀ (a b c d : ℤ) (ζ : ℂ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (ζ^4 = 1) →
  (ζ ≠ 1) →
  smallest_value_complex_expression a b c d ζ = 2 := 
by sorry

end smallest_value_complex_expression_min_l470_470731


namespace max_edges_intersected_by_plane_l470_470072

/-
A plane has no vertex of a regular dodecahedron on it.
Prove that the plane intersects at most 10 edges of the dodecahedron.
-/

theorem max_edges_intersected_by_plane (dodecahedron : Type)
  [regular_dodecahedron dodecahedron]
  (plane : Type)
  [does_not_pass_through_any_vertex plane dodecahedron] :
  exists n ≤ 10, plane_intersects_n_edges_of_dodecahedron plane dodecahedron n :=
sorry

end max_edges_intersected_by_plane_l470_470072


namespace integer_a_satisfies_equation_l470_470916

theorem integer_a_satisfies_equation (a b c : ℤ) :
  (∃ b c : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) → 
    a = 2 :=
by
  intro h_eq
  -- Proof goes here
  sorry

end integer_a_satisfies_equation_l470_470916


namespace average_probable_weight_l470_470842

-- Define the conditions
def Arun_opinion (w : ℝ) : Prop := 64 < w ∧ w < 72
def Brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def Mother_opinion (w : ℝ) : Prop := w ≤ 67

-- The proof problem statement
theorem average_probable_weight :
  ∃ (w : ℝ), Arun_opinion w ∧ Brother_opinion w ∧ Mother_opinion w →
  (64 + 67) / 2 = 65.5 :=
by
  sorry

end average_probable_weight_l470_470842


namespace abs_x_sub_y_eq_4_l470_470517

noncomputable theory
open Real

-- Given conditions
variables (x y : ℝ)
axiom h1 : (x + y + 10 + 11 + 9) / 5 = 10
axiom h2 : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2

-- The proof problem
theorem abs_x_sub_y_eq_4 : |x - y| = 4 :=
by sorry

end abs_x_sub_y_eq_4_l470_470517


namespace negation_of_p_l470_470746

variable (p : Prop) (n : ℕ)

def proposition_p := ∃ n : ℕ, n^2 > 2^n

theorem negation_of_p : ¬ proposition_p ↔ ∀ n : ℕ, n^2 <= 2^n :=
by
  sorry

end negation_of_p_l470_470746


namespace complement_union_l470_470284

open Set

variable (U M N : Set ℕ)

def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem complement_union (hU : U = {0, 1, 2, 3, 4, 5, 6})
                          (hM : M = {1, 3, 5})
                          (hN : N = {2, 4, 6}) :
  (complement_U U M) ∪ (complement_U U N) = {0, 1, 2, 3, 4, 5, 6} :=
by 
  sorry

end complement_union_l470_470284


namespace RahulPlayedMatchesSolver_l470_470043

noncomputable def RahulPlayedMatches (current_average new_average runs_in_today current_matches : ℕ) : ℕ :=
  let total_runs_before := current_average * current_matches
  let total_runs_after := total_runs_before + runs_in_today
  let total_matches_after := current_matches + 1
  total_runs_after / new_average

theorem RahulPlayedMatchesSolver:
  RahulPlayedMatches 52 54 78 12 = 12 :=
by
  sorry

end RahulPlayedMatchesSolver_l470_470043


namespace minimum_perimeter_dough_l470_470482

theorem minimum_perimeter_dough 
  (width_dough : ℝ) 
  (side_length_mold : ℝ) 
  (remainder_width : ℝ)
  (total_cookies : ℕ) 
  (width_dough = 34) 
  (side_length_mold = 4) 
  (remainder_width = 2) 
  (total_cookies = 24) 
  : (2 * width_dough + 2 * (total_cookies / (width_dough - remainder_width) * side_length_mold)) = 92 := by
  sorry

end minimum_perimeter_dough_l470_470482


namespace fifth_term_of_sequence_is_minus_six_l470_470525

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n, a (n + 2) = a (n + 1) - a n

theorem fifth_term_of_sequence_is_minus_six (a : ℕ → ℤ) (h : sequence a) : a 5 = -6 :=
by
  sorry

end fifth_term_of_sequence_is_minus_six_l470_470525


namespace parallel_lines_l470_470729

-- Definitions of the geometric objects and concepts required for the problem
section

variables {Γ Γ' : Type} [MetricSpace Γ] [MetricSpace Γ']
variables {P Q A : Γ} {P' A' : Γ'}

-- Assuming the following conditions:
-- 1. P and Q are intersection points of circles Γ and Γ'
-- 2. A is a point on circle Γ
-- 3. The tangent to Γ at P meets Γ' at P'
-- 4. The line (AQ) intersects Γ' at A'

-- Using collinear to represent line segments and tangent condition
def is_tangent (Γ : Type) [MetricSpace Γ] (P : Γ) : Prop := sorry

def intersects (Γ : Type) [MetricSpace Γ] (A B C : Γ) : Prop := sorry

-- The definitions for the specific points and circles given the conditions
noncomputable def circles_intersect (Γ Γ' : Type) [MetricSpace Γ] [MetricSpace Γ'] 
  (P Q : Γ) (A : Γ) : Prop := sorry

noncomputable def tangent_at_point (Γ : Type) [MetricSpace Γ] (P : Γ) (P' : Γ') : Prop := sorry

noncomputable def line_intersects (Γ : Type) [MetricSpace Γ] (A Q : Γ) (A' : Γ') : Prop := sorry

theorem parallel_lines (Γ Γ' : Type) [MetricSpace Γ] [MetricSpace Γ'] 
  (P Q A : Γ) (P' A' : Γ') 
  (h1 : circles_intersect Γ Γ' P Q A) 
  (h2 : tangent_at_point Γ P P')
  (h3 : line_intersects Γ A Q A') :
  ∥(A, P)∥ = ∥(A', P')∥ :=
sorry

end

end parallel_lines_l470_470729


namespace range_G_l470_470120

noncomputable def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

theorem range_G : Set.range G = Set.Icc (-8 : ℝ) 8 := sorry

end range_G_l470_470120


namespace inequality_solution_range_l470_470919

theorem inequality_solution_range (x : ℝ) : (x^2 + 3*x - 10 < 0) ↔ (-5 < x ∧ x < 2) :=
by
  sorry

end inequality_solution_range_l470_470919


namespace inequality_holds_for_all_real_numbers_l470_470372

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470372


namespace range_of_a_l470_470179

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 7 * x1^2 - (a + 13) * x1 + a^2 - a - 2 = 0 ∧
                 7 * x2^2 - (a + 13) * x2 + a^2 - a - 2 = 0 ∧
                 0 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < 2) →
  (-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4) :=
by
  intro h
  sorry

end range_of_a_l470_470179


namespace radius_any_positive_real_l470_470241

theorem radius_any_positive_real (r : ℝ) (h₁ : r > 0) 
    (h₂ : r * (2 * Real.pi * r) = 2 * Real.pi * r^2) : True :=
by
  sorry

end radius_any_positive_real_l470_470241


namespace inequality_xyz_l470_470333

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470333


namespace cat_weight_is_correct_l470_470996

def puppy_weight : ℝ := 7.5
def additional_weight : ℝ := 5
def cat_weight : ℝ := puppy_weight + additional_weight

theorem cat_weight_is_correct : cat_weight = 12.5 :=
by
  have h1 : puppy_weight = 7.5 := rfl
  have h2 : additional_weight = 5 := rfl
  have h3 : cat_weight = puppy_weight + additional_weight := rfl
  have h4 : cat_weight = 7.5 + 5 := by rw [←h1, ←h2, ←h3]
  show cat_weight = 12.5, by rw h4; norm_num

end cat_weight_is_correct_l470_470996


namespace walnut_tree_total_l470_470796

theorem walnut_tree_total (initial_trees : ℕ) (planted_trees : ℕ) (total_trees : ℕ) 
  (h1 : initial_trees = 4) (h2 : planted_trees = 6) : total_trees = 10 :=
by
  -- Using initial conditions and the arithmetic operation to verify the result
  have ht : initial_trees + planted_trees = total_trees := by sorry
  -- Apply the conditions h1 and h2 to prove total_trees is 10
  rw [h1, h2] at ht
  exact ht

end walnut_tree_total_l470_470796


namespace prove_inequality_l470_470627

theorem prove_inequality
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0)
  (h₄ : a ≤ b)
  (h₅ : b ≤ c)
  (h₆ : c ≤ d)
  (h₇ : a + b + c + d ≥ 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 :=
by
  sorry

end prove_inequality_l470_470627


namespace salt_percentage_in_first_solution_l470_470889

theorem salt_percentage_in_first_solution
    (S : ℝ)
    (h1 : ∀ w : ℝ, w ≥ 0 → ∃ q : ℝ, q = w)  -- One fourth of the first solution was replaced by the second solution
    (h2 : ∀ w1 w2 w3 : ℝ,
            w1 + w2 = w3 →
            (w1 / w3 * S + w2 / w3 * 25 = 16)) :  -- Resulting solution was 16 percent salt by weight
  S = 13 :=   -- Correct answer
sorry

end salt_percentage_in_first_solution_l470_470889


namespace orthocenter_lies_on_altitude_l470_470374

def point := (ℝ × ℝ)
def Triangle (A B C : point) := true

variables {A B C T E D M O: point}

-- Assume the triangle ABC is a right-angled triangle at C
def right_angle (p1 p2 p3: point) : Prop := true

-- The angles at A and B are alpha and beta respectively
def angle_alpha := ∠ A C B = 90
def angle_beta := ∠ B C A = 90
def alpha_beta_relation := ∠ B C A = 90 - ∠ A B C

-- The points T, E, D are touchpoints of the incircle on sides AB, AC, and BC respectively
def incircle_touchpoints (T E D : point) : Prop := true

-- O is the center of the incircle
def incircle_center (O : point) := true

-- Points C, D, E, O form a square
def square (C D E O : point) := true

-- M is the orthocenter of the triangle TDE
def orthocenter (T E D M: point) := true

-- To prove: M lies on the altitude from C to AB
def lies_on_altitude (C M A B: point) := true

theorem orthocenter_lies_on_altitude :
  right_angle A B C ∧
  angle_alpha ∧
  angle_beta ∧
  alpha_beta_relation ∧
  incircle_touchpoints T E D ∧
  incircle_center O ∧
  square C D E O ∧
  orthocenter T E D M → 
  lies_on_altitude C M A B :=
sorry

end orthocenter_lies_on_altitude_l470_470374


namespace repeating_decimal_to_fraction_l470_470130

theorem repeating_decimal_to_fraction : 
  (let x := 7.2 + (34 : ℚ) / 99,
    100 * x = 734 + (34 : ℚ) / 100 ) → 
    x = (36357 : ℚ) / 4950 := 
  by
    intros h,
    sorry

end repeating_decimal_to_fraction_l470_470130


namespace ratio_b_to_c_l470_470833

variable (a b c k : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = k * c
def condition3 : Prop := a + b + c = 32
def condition4 : Prop := b = 12

-- Question: Prove that ratio of b to c is 2:1
theorem ratio_b_to_c
  (h1 : condition1 a b)
  (h2 : condition2 b k c)
  (h3 : condition3 a b c)
  (h4 : condition4 b) :
  b = 2 * c := 
sorry

end ratio_b_to_c_l470_470833


namespace state_a_selection_percentage_l470_470707

-- Definitions based on the conditions
variables {P : ℕ} -- percentage of candidates selected in State A

theorem state_a_selection_percentage 
  (candidates : ℕ) 
  (state_b_percentage : ℕ) 
  (extra_selected_in_b : ℕ) 
  (total_selected_in_b : ℕ) 
  (total_selected_in_a : ℕ)
  (appeared_in_each_state : ℕ) 
  (H1 : appeared_in_each_state = 8200)
  (H2 : state_b_percentage = 7)
  (H3 : extra_selected_in_b = 82)
  (H4 : total_selected_in_b = (state_b_percentage * appeared_in_each_state) / 100)
  (H5 : total_selected_in_a = total_selected_in_b - extra_selected_in_b)
  (H6 : total_selected_in_a = (P * appeared_in_each_state) / 100)
  : P = 6 :=
by {
  sorry
}

end state_a_selection_percentage_l470_470707


namespace phillip_spent_on_oranges_l470_470310

theorem phillip_spent_on_oranges 
  (M : ℕ) (A : ℕ) (C : ℕ) (L : ℕ) (O : ℕ)
  (hM : M = 95) (hA : A = 25) (hC : C = 6) (hL : L = 50)
  (h_total_spending : O + A + C = M - L) : 
  O = 14 := 
sorry

end phillip_spent_on_oranges_l470_470310


namespace classify_numbers_l470_470908

def given_numbers : List ℚ := [-3, -1, 0, 20, 1/4, 17/100, -8 - 1/2, -(-2), 22/7]

def set_of_positive_integers := {x | x ∈ given_numbers ∧ x > 0 ∧ ∃ n : ℕ, n ≠ 0 ∧ x = n}

def set_of_fractions := {x | x ∈ given_numbers ∧ ∃ p q : ℤ, q ≠ 0 ∧ x = p / q}

def set_of_non_positive_rationals := {x | x ∈ given_numbers ∧ (x < 0 ∨ x = 0)}

theorem classify_numbers :
  set.set_of set_of_positive_integers = {20, 2} ∧
  set.set_of set_of_fractions = {1/4, 17/100, -17/2, 22/7} ∧
  set.set_of set_of_non_positive_rationals = {-3, -1, 0, -17/2} :=
by sorry

end classify_numbers_l470_470908


namespace min_eq_implies_dist_eq_zero_dist_eq_zero_implies_inter_nonempty_l470_470943

section
variable {ι : Type*} [LinearOrder ι]

def dist (A B : Finset ι) : ι := Finset.min' (Finset.image (λ p : ι × ι, |p.1 - p.2|) (A.product B)) sorry

theorem min_eq_implies_dist_eq_zero {A B : Finset ι} (h : A.nonempty ∧ B.nonempty) (h₁ : A.min' (Finset.nonempty_of_ne_empty h.1) = B.min' (Finset.nonempty_of_ne_empty h.2)) : 
  dist A B = 0 := sorry

theorem dist_eq_zero_implies_inter_nonempty {A B : Finset ι} (h : A.nonempty ∧ B.nonempty) (h₁ : dist A B = 0) : 
  A ∩ B ≠ ∅ := sorry
end

end min_eq_implies_dist_eq_zero_dist_eq_zero_implies_inter_nonempty_l470_470943


namespace pipe_a_fills_tank_in_42_hours_l470_470040

-- Define the rates as real numbers
variables (A B C : ℝ)

-- Conditions extracted from the problem statement
def condition1 := A + B + C = 1 / 6
def condition2 := C = 2 * B
def condition3 := B = 2 * A

-- Main theorem: prove that 1/A = 42, which means pipe A alone takes 42 hours
theorem pipe_a_fills_tank_in_42_hours (h1 : condition1) (h2 : condition2) (h3 : condition3) : 1 / A = 42 := 
  by sorry

end pipe_a_fills_tank_in_42_hours_l470_470040


namespace toms_total_score_l470_470807

def points_per_enemy : ℕ := 10
def enemies_killed : ℕ := 175

def base_score (enemies : ℕ) : ℝ := enemies * points_per_enemy

def bonus_percentage (enemies : ℕ) : ℝ :=
  if 100 ≤ enemies ∧ enemies < 150 then 0.50
  else if 150 ≤ enemies ∧ enemies < 200 then 0.75
  else if enemies ≥ 200 then 1.00
  else 0.0

def total_score (enemies : ℕ) : ℝ :=
  let base := base_score enemies
  let bonus := base * bonus_percentage enemies
  base + bonus

theorem toms_total_score :
  total_score enemies_killed = 3063 :=
by
  -- The proof will show the computed total score
  -- matches the expected value
  sorry

end toms_total_score_l470_470807


namespace greatest_real_part_in_candidates_l470_470307

def z_candidates : List ℂ :=
  [(-3 : ℂ), (-2 + (1 : ℂ) * complex.I), (-1 + (2 : ℂ) * complex.I), (-1 + (3 : ℂ) * complex.I), (0 + (3 : ℂ) * complex.I)]

def real_part_of_z4 (z : ℂ) : ℝ :=
  (z ^ 4).re

theorem greatest_real_part_in_candidates :
  ∀ z ∈ z_candidates, real_part_of_z4 z ≤ real_part_of_z4 (-3) :=
by
  sorry

end greatest_real_part_in_candidates_l470_470307


namespace zuminglish_words_mod_2000_l470_470249

noncomputable def zuminglish_sequences_modulo : ℕ :=
let a : ℕ → ℕ := λ n, if n = 2 then 4 else 2 * (a (n - 1) + c (n - 1)),
    b : ℕ → ℕ := λ n, if n = 2 then 2 else a (n - 1),
    c : ℕ → ℕ := λ n, if n = 2 then 2 else 2 * b (n - 1),
    n := 12 in
(a n + b n + c n) % 2000

theorem zuminglish_words_mod_2000 : zuminglish_sequences_modulo = 192 := by
  sorry

end zuminglish_words_mod_2000_l470_470249


namespace part_I_part_II_l470_470656

noncomputable def f_I (x : ℝ) : ℝ := abs (3*x - 1) + abs (x + 3)

theorem part_I :
  ∀ x : ℝ, f_I x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 :=
by sorry

noncomputable def f_II (x b c : ℝ) : ℝ := abs (x - b) + abs (x + c)

theorem part_II :
  ∀ b c : ℝ, b > 0 → c > 0 → b + c = 1 → 
  (∀ x : ℝ, f_II x b c ≥ 1) → (1 / b + 1 / c = 4) :=
by sorry

end part_I_part_II_l470_470656


namespace parallel_if_perpendicular_to_same_plane_l470_470201

variables (m n l : Line) (α β γ : Plane)

-- Definitions for distinct lines and planes
axiom distinct_lines (m n l : Line) : m ≠ n ∧ n ≠ l ∧ m ≠ l
axiom distinct_planes (α β γ : Plane) : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Two lines are perpendicular to the same plane
axiom perp_to_plane (m : Line) (α : Plane) : m ⊥ α
axiom perp_to_plane (n : Line) (α : Plane) : n ⊥ α

-- The main theorem statement
theorem parallel_if_perpendicular_to_same_plane (h1 : m ⊥ α) (h2 : n ⊥ α) : m ∥ n := sorry

end parallel_if_perpendicular_to_same_plane_l470_470201


namespace frankie_carla_ratio_l470_470602

theorem frankie_carla_ratio (total_games : ℕ) (carla_wins : ℕ) (frankie_wins : ℕ)
    (h_total : total_games = 30) (h_carla : carla_wins = 20) 
    (h_frankie : frankie_wins = total_games - carla_wins) :
    (frankie_wins : ℚ) / (carla_wins : ℚ) = 1 / 2 :=
by
  rw [h_total, h_carla, h_frankie]
  norm_num
  sorry

end frankie_carla_ratio_l470_470602


namespace max_real_factors_l470_470584

-- Given the polynomial x^{10} - 1
def poly := polynomial C

-- Define the polynomial x^{10} - 1
noncomputable def poly10_minus_1 : poly := X^10 - 1

-- Define the maximum number of non-constant factors over the reals
def max_factors (p : polynomial C) : ℕ := 3

/-- The maximum number of non-constant factors of x^10 - 1 with real coefficients is 3 -/
theorem max_real_factors (p : poly) (h : p = X^10 - 1) : max_factors p = 3 :=
sorry

end max_real_factors_l470_470584


namespace inequality_xyz_l470_470326

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470326


namespace objective_function_range_l470_470185

theorem objective_function_range:
  (∃ x y : ℝ, x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) ∧
  (∀ x y : ℝ, (x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) →
  (3*x + y ≥ (19:ℝ) / 9 ∧ 3*x + y ≤ 6)) :=
sorry

-- We have defined the conditions, the objective function, and the assertion in Lean 4.

end objective_function_range_l470_470185


namespace inequality_inequality_l470_470364

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470364


namespace ryan_total_commuting_time_l470_470802

def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def bus_commutes : ℕ := 3
def total_bus_time : ℕ := bus_time * bus_commutes
def friend_time : ℕ := biking_time - (2 * biking_time / 3)
def total_commuting_time : ℕ := biking_time + total_bus_time + friend_time

theorem ryan_total_commuting_time :
  total_commuting_time = 160 :=
by
  sorry

end ryan_total_commuting_time_l470_470802


namespace triangle_angle_contradiction_l470_470471

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (sum_angles : A + B + C = 180) : 
  (¬ (A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60)) = (A > 60 ∧ B > 60 ∧ C > 60) :=
by sorry

end triangle_angle_contradiction_l470_470471


namespace bridge_weight_excess_l470_470076

theorem bridge_weight_excess :
  ∀ (Kelly_weight Megan_weight Mike_weight : ℕ),
  Kelly_weight = 34 →
  Kelly_weight = 85 * Megan_weight / 100 →
  Mike_weight = Megan_weight + 5 →
  (Kelly_weight + Megan_weight + Mike_weight) - 100 = 19 :=
by
  intros Kelly_weight Megan_weight Mike_weight
  intros h1 h2 h3
  sorry

end bridge_weight_excess_l470_470076


namespace find_f_inequality_solution_set_l470_470288

-- Define the given conditions
variables {ℝ : Type*} [linear_ordered_field ℝ] [has_deriv ℝ]
variables (f : ℝ → ℝ)
variables (f' : ℝ → ℝ) [∀ x, differentiable_at ℝ f x]

-- Assuming f' = f - 2 and f(0) = 2020
axiom hf' : ∀ x, has_deriv_at f (f x - 2) x
axiom hf0 : f 0 = 2020

-- Theorem to prove f(x) = 2 + 2018e^x
theorem find_f :
  ∀ x, f x = 2 + 2018 * Real.exp x :=
sorry

-- Theorem to prove the solution set of the inequality
theorem inequality_solution_set :
  { x : ℝ | f x + 4034 > 2 * (f x - 2) } = set.Iio (Real.log 2) :=
sorry

end find_f_inequality_solution_set_l470_470288


namespace sequence_integers_and_explicit_formula_l470_470526

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 5 ∧ 
  (∀ n ≥ 3, a n = (a (n-1))^2 + 4 / a (n-2))

theorem sequence_integers_and_explicit_formula (a : ℕ → ℝ) : 
  sequence a → 
  (∀ n, a n ∈ ℤ) ∧ 
  (∀ n, a n = ((2 + Real.sqrt 2)/4) * (3 - 2 * Real.sqrt 2)^n + ((2 - Real.sqrt 2)/4) * (3 + 2 * Real.sqrt 2)^n) :=
by
  sorry

end sequence_integers_and_explicit_formula_l470_470526


namespace least_integer_nk_l470_470961

noncomputable def min_nk (k : ℕ) : ℕ :=
  (5 * k + 1) / 2

theorem least_integer_nk (k : ℕ) (S : Fin 5 → Finset ℕ) :
  (∀ j : Fin 5, (S j).card = k) →
  (∀ i : Fin 4, (S i ∩ S (i + 1)).card = 0) →
  (S 4 ∩ S 0).card = 0 →
  (∃ nk, (∃ (U : Finset ℕ), (∀ j : Fin 5, S j ⊆ U) ∧ U.card = nk) ∧ nk = min_nk k) :=
by
  sorry

end least_integer_nk_l470_470961


namespace count_integers_between_powers_l470_470215

noncomputable def power (a : ℝ) (b : ℝ) : ℝ := a^b

theorem count_integers_between_powers:
  let a := 10
  let b1 := 0.1
  let b2 := 0.4
  have exp1 : Float := (a + b1)
  have exp2 : Float := (a + b2)
  have n1 : ℤ := exp1^3.ceil
  have n2 : ℤ := exp2^3.floor
  n2 - n1 + 1 = 94 := 
begin
  sorry
end

end count_integers_between_powers_l470_470215


namespace distribute_students_l470_470122

theorem distribute_students :
  ∃ (distribution : Finset (Finset (Fin 5) × Finset (Fin 5) × Finset (Fin 5))),
    (∀ (s : Finset (Finset (Fin 5) × Finset (Fin 5) × Finset (Fin 5))),
      (s ∈ distribution →
        ∃ (a b c : Finset (Fin 5)),
          a ∪ b ∪ c = {0, 1, 2, 3, 4} ∧
          1 ≤ a.card ∧ a.card ≤ 2 ∧
          1 ≤ b.card ∧ b.card ≤ 2 ∧
          1 ≤ c.card ∧ c.card ≤ 2 ∧
          ¬ (0 ∈ a) ∧ -- Student A (denoted by 0) not in dormitory a
          s = (a, b, c))) ∧
    distribution.card = 60 :=
sorry

end distribute_students_l470_470122


namespace integer_roots_of_quadratic_eq_l470_470134

theorem integer_roots_of_quadratic_eq (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 + x2 = a ∧ x1 * x2 = 9 * a) ↔
  a = 100 ∨ a = -64 ∨ a = 48 ∨ a = -12 ∨ a = 36 ∨ a = 0 :=
by sorry

end integer_roots_of_quadratic_eq_l470_470134


namespace probability_gt_2_of_dice_roll_l470_470841

theorem probability_gt_2_of_dice_roll : 
  (∃ outcomes : finset ℕ, 
   outcomes = {1, 2, 3, 4, 5, 6} ∧ 
   (∃ favorable : finset ℕ, 
    favorable = {3, 4, 5, 6} ∧ 
    (favorable.card / outcomes.card : ℚ) = (2 / 3 : ℚ))) := 
sorry

end probability_gt_2_of_dice_roll_l470_470841


namespace range_of_x_range_of_a_l470_470614

-- Part (1): 
theorem range_of_x (x : ℝ) : 
  (a = 1) → (x^2 - 6 * a * x + 8 * a^2 < 0) → (x^2 - 4 * x + 3 ≤ 0) → (2 < x ∧ x ≤ 3) := sorry

-- Part (2):
theorem range_of_a (a : ℝ) : 
  (a ≠ 0) → (∀ x, (x^2 - 4 * x + 3 ≤ 0) → (x^2 - 6 * a * x + 8 * a^2 < 0)) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 4) := sorry

end range_of_x_range_of_a_l470_470614


namespace evan_chen_ineq_l470_470596

noncomputable def S (X : Finset ℕ) : ℕ := X.sum id
noncomputable def P (X : Finset ℕ) : ℕ := X.prod id

theorem evan_chen_ineq (A B : Finset ℕ)
  (h1 : A.card = B.card)
  (h2 : P A = P B)
  (h3 : S A ≠ S B)
  (h4 : ∀ n ∈ (A ∪ B), ∀ p : ℕ, p.Prime → p ∣ n → p^36 ∣ n ∧ ¬ p^37 ∣ n) :
  | (S A : ℤ) - (S B : ℤ) | > 1.9 * 10^6 := sorry

end evan_chen_ineq_l470_470596


namespace length_of_platform_is_180_l470_470531

-- Define the train passing a platform and a man with given speeds and times
def train_pass_platform (speed : ℝ) (time_man time_platform : ℝ) (length_train length_platform : ℝ) :=
  time_man = length_train / speed ∧ 
  time_platform = (length_train + length_platform) / speed

-- Given conditions
noncomputable def train_length_platform :=
  ∃ length_platform,
    train_pass_platform 15 20 32 300 length_platform ∧
    length_platform = 180

-- The main theorem we want to prove
theorem length_of_platform_is_180 : train_length_platform :=
sorry

end length_of_platform_is_180_l470_470531


namespace total_registration_methods_l470_470913

theorem total_registration_methods (n : ℕ) (h : n = 5) : (2 ^ n) = 32 :=
by
  sorry

end total_registration_methods_l470_470913


namespace total_area_of_pyramid_faces_l470_470824

-- Define the conditions of the problem
def base_edge : ℝ := 6
def lateral_edge : ℝ := 5
def altitude_triangle_face : ℝ := 4
def area_one_triangle_face : ℝ := 1/2 * base_edge * altitude_triangle_face
def total_area : ℝ := 4 * area_one_triangle_face

-- Prove that the total area of the four triangular faces is 48 square units
theorem total_area_of_pyramid_faces : total_area = 48 := by
  sorry

end total_area_of_pyramid_faces_l470_470824


namespace remainder_55_pow_55_plus_10_mod_8_l470_470005

theorem remainder_55_pow_55_plus_10_mod_8 : (55 ^ 55 + 10) % 8 = 1 :=
by
  sorry

end remainder_55_pow_55_plus_10_mod_8_l470_470005


namespace age_sum_is_27_l470_470488

noncomputable def a : ℕ := 12
noncomputable def b : ℕ := 10
noncomputable def c : ℕ := 5

theorem age_sum_is_27
  (h1: a = b + 2)
  (h2: b = 2 * c)
  (h3: b = 10) :
  a + b + c = 27 :=
  sorry

end age_sum_is_27_l470_470488


namespace unknown_number_is_7_l470_470520

theorem unknown_number_is_7 (x : ℤ) (hx : x > 0)
  (h : (1 / 4 : ℚ) * (10 * x + 7 - x ^ 2) - x = 0) : x = 7 :=
  sorry

end unknown_number_is_7_l470_470520


namespace medians_concurrent_at_centroid_midline_through_centroid_l470_470838

structure Tetrahedron (V : Type*) [euclidean_space V] :=
  (A B C D : V)

def midpoint {V : Type*} [euclidean_space V] (P Q : V) : V :=
  (1/2) • (P + Q)

def median {V : Type*} [euclidean_space V] (T : Tetrahedron V) (vertex : V) : V :=
  match T with
  | ⟨A, B, C, D⟩ =>
    if vertex = A then
      midpoint B (midpoint C D)
    else if vertex = B then
      midpoint A (midpoint C D)
    else if vertex = C then
      midpoint A (midpoint B D)
    else
      midpoint A (midpoint B C)

def centroid {V : Type*} [euclidean_space V] (T : Tetrahedron V) : V :=
  let A := T.A in
  let B := T.B in
  let C := T.C in
  let D := T.D in
  (1/4) • (A + B + C + D)

theorem medians_concurrent_at_centroid {V : Type*} [euclidean_space V] (T : Tetrahedron V) :
  ∃ K : V, (∀ vertex, median T vertex = K) ∧ K = centroid T :=
by
  sorry

theorem midline_through_centroid {V : Type*} [euclidean_space V] (T : Tetrahedron V) :
  ∀ (P Q : V), (P ∈ [T.A, T.B, T.C, T.D] ∧ Q ∈ [T.A, T.B, T.C, T.D]) → 
  (midpoint P Q = centroid T) →
  ∃ K : V, K = centroid T :=
by
  sorry

end medians_concurrent_at_centroid_midline_through_centroid_l470_470838


namespace find_R_l470_470165

variable {R : ℝ → ℝ}
variable {P : ℝ → ℝ}

theorem find_R (h1 : ∀ t : ℝ, 
   7 * (sin t) ^ 31 + 8 * (sin t) ^ 13 - 5 * (sin t) ^ 5 * (cos t) ^ 4 - 10 * (sin t) ^ 7 + 5 * (sin t) ^ 5 - 2 = 
   P(sin t) * ((sin t) ^ 4 - (1 + (sin t)) * ((cos t) ^ 2 - 2)) + R(sin t))
   (h2 : ∀ t : ℝ, sin t = x ∧ cos t = √(1 - x^2)) : 
   R(x) = 13 * x ^ 3 + 5 * x ^ 2 + 12 * x + 3 := 
begin
  sorry
end

end find_R_l470_470165


namespace area_ratio_l470_470267

noncomputable def triangle_side_lengths : ℝ := {
  XY := 30,
  YZ := 45,
  XZ := 51 }

noncomputable def points_on_sides : ℝ := {
  XP := 18,
  XQ := 15 }

theorem area_ratio (XY YZ XZ XP XQ : ℝ) (h1 : XY = 30) (h2 : YZ = 45) (h3 : XZ = 51) 
  (h4 : XP = 18) (h5 : XQ = 15) :
  ((area_of_triangle XPQ) / (area_of_quadrilateral PQZY)) = 459 / 625 :=
by
  sorry

end area_ratio_l470_470267


namespace medical_staff_gender_profession_l470_470513

theorem medical_staff_gender_profession
  (a b c d : ℕ)
  (total_staff : a + b + c + d = 17)
  (doctors_ge_nurses : a + b ≥ c + d)
  (female_nurses_gt_male_doctors : d > a)
  (male_doctors_gt_female_doctors : a > b)
  (male_nurses_at_least_two : c ≥ 2) :
  b = 4 ∧ a = 5 ∧ c = 2 ∧ d = 6 ∧ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →
  "Female doctor" := sorry

end medical_staff_gender_profession_l470_470513


namespace find_smallest_n_l470_470737

noncomputable def smallest_n (m : ℕ) : ℕ :=
  let n := 6 in -- since n is a proven solution from the problem
  let r := (m - n^3) / (3 * n^2 + 3 * n * (m - n^3) + (m - n^3) ^ 2) in
  if m = (n + r)^3 ∧ r < 1 / 100 then n else 0 -- only return n if conditions are met

theorem find_smallest_n :
  ∃ n : ℕ, ∀ (m : ℕ) (r : ℝ), 
    (m = (n + r)^3) ∧ (r < 1 / 100) ∧ (0 < n) → 
    (n = 6) :=
by
  sorry -- Proof is omitted

end find_smallest_n_l470_470737


namespace pipe_b_rate_l470_470448

variables (B : ℕ) -- rate at which Pipe B fills the tank in liters per minute

theorem pipe_b_rate :
  let tank_capacity := 5000 in
  let pipe_a_rate := 200 in
  let pipe_c_rate := 25 in
  let pipe_a_time := 1 in
  let pipe_b_time := 2 in
  let pipe_c_time := 2 in
  let cycle_time := pipe_a_time + pipe_b_time + pipe_c_time in
  let total_time := 100 in
  let cycles := total_time / cycle_time in
  let net_per_cycle := pipe_a_rate * pipe_a_time + B * pipe_b_time - pipe_c_rate * pipe_c_time in
  cycles * net_per_cycle = tank_capacity 
  → B = 50 :=
begin
  intros,
  sorry
end

end pipe_b_rate_l470_470448


namespace polynomial_quotient_l470_470458

theorem polynomial_quotient : 
  (12 * x^3 + 20 * x^2 - 7 * x + 4) / (3 * x + 4) = 4 * x^2 + (4/3) * x - 37/9 :=
by
  sorry

end polynomial_quotient_l470_470458


namespace adjusted_area_difference_l470_470902

noncomputable def largest_circle_area (d : ℝ) : ℝ :=
  let r := d / 2
  r^2 * Real.pi

noncomputable def middle_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

noncomputable def smaller_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

theorem adjusted_area_difference (d_large r_middle r_small : ℝ) 
  (h_large : d_large = 30) (h_middle : r_middle = 10) (h_small : r_small = 5) :
  largest_circle_area d_large - middle_circle_area r_middle - smaller_circle_area r_small = 100 * Real.pi :=
by
  sorry

end adjusted_area_difference_l470_470902


namespace inequality_xyz_l470_470327

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470327


namespace calculate_series_sum_l470_470904

theorem calculate_series_sum : 
  (∑ n in Finset.range (25), (-1 : ℤ)^(n - 12)) + 
  (∑ n in Finset.range (25), 2 * (-2 : ℤ)^(n - 12)) = 8193 := 
by
sqrt
 
end calculate_series_sum_l470_470904


namespace tiles_in_each_row_l470_470392

theorem tiles_in_each_row (area : ℝ) (tile_side : ℝ) (feet_to_inches : ℝ) (number_of_tiles : ℕ) :
  area = 400 ∧ tile_side = 8 ∧ feet_to_inches = 12 → number_of_tiles = 30 :=
by
  intros h
  cases h with h_area h_rest
  cases h_rest with h_tile_side h_feet_to_inches
  sorry

end tiles_in_each_row_l470_470392


namespace doughnuts_served_initially_l470_470123

def initial_doughnuts_served (staff_count : Nat) (doughnuts_per_staff : Nat) (doughnuts_left : Nat) : Nat :=
  staff_count * doughnuts_per_staff + doughnuts_left

theorem doughnuts_served_initially :
  ∀ (staff_count doughnuts_per_staff doughnuts_left : Nat), staff_count = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  initial_doughnuts_served staff_count doughnuts_per_staff doughnuts_left = 50 :=
by
  intros staff_count doughnuts_per_staff doughnuts_left hstaff hdonuts hleft
  rw [hstaff, hdonuts, hleft]
  rfl

#check doughnuts_served_initially

end doughnuts_served_initially_l470_470123


namespace rice_price_per_kg_l470_470219

theorem rice_price_per_kg (price1 price2 : ℝ) (amount1 amount2 : ℝ) (total_cost total_weight : ℝ) (P : ℝ)
  (h1 : price1 = 6.60)
  (h2 : amount1 = 49)
  (h3 : price2 = 9.60)
  (h4 : amount2 = 56)
  (h5 : total_cost = price1 * amount1 + price2 * amount2)
  (h6 : total_weight = amount1 + amount2)
  (h7 : P = total_cost / total_weight) :
  P = 8.20 := 
by sorry

end rice_price_per_kg_l470_470219


namespace fill_tank_time_l470_470761

-- Define the rates at which the pipes fill or empty the tank
def rateA : ℚ := 1 / 16
def rateB : ℚ := - (1 / 24)  -- Since pipe B empties the tank, it's negative.

-- Define the time after which pipe B is closed
def timeBClosed : ℚ := 21

-- Define the initial combined rate of both pipes
def combinedRate : ℚ := rateA + rateB

-- Define the proportion of the tank filled in the initial 21 minutes
def filledIn21Minutes : ℚ := combinedRate * timeBClosed

-- Define the remaining tank to be filled after pipe B is closed
def remainingTank : ℚ := 1 - filledIn21Minutes

-- Define the additional time required to fill the remaining part of the tank with only pipe A
def additionalTime : ℚ := remainingTank / rateA

-- Total time is the sum of the initial time and additional time
def totalTime : ℚ := timeBClosed + additionalTime

theorem fill_tank_time : totalTime = 30 :=
by
  -- Proof omitted
  sorry

end fill_tank_time_l470_470761


namespace unique_integer_for_P5_l470_470882

-- Define the polynomial P with integer coefficients
variable (P : ℤ → ℤ)

-- The conditions given in the problem
variable (x1 x2 x3 : ℤ)
variable (Hx1 : P x1 = 1)
variable (Hx2 : P x2 = 2)
variable (Hx3 : P x3 = 3)

-- The main theorem to prove
theorem unique_integer_for_P5 {P : ℤ → ℤ} {x1 x2 x3 : ℤ}
(Hx1 : P x1 = 1) (Hx2 : P x2 = 2) (Hx3 : P x3 = 3) :
  ∃!(x : ℤ), P x = 5 := sorry

end unique_integer_for_P5_l470_470882


namespace path_length_traversed_by_S_l470_470380

-- Define the basic geometrical structures and their properties
noncomputable def square_side_length : ℝ := 6
noncomputable def triangle_side_length : ℝ := 3
def angle_of_rotation : ℝ := 120 * real.pi / 180

-- Prove the length of the path traversed by vertex S
theorem path_length_traversed_by_S 
  (square_side_length: ℝ) (triangle_side_length: ℝ) (angle_of_rotation: ℝ) :
  let arc_length := triangle_side_length * angle_of_rotation in 
  let total_path_length := 4 * 3 * arc_length in 
  total_path_length = 24 * real.pi :=
by
  let arc_length := triangle_side_length * angle_of_rotation
  let total_path_length := 4 * 3 * arc_length
  have h1 : arc_length = 2 * real.pi := sorry
  have h2 : total_path_length = 24 * real.pi := sorry
  exact h2

end path_length_traversed_by_S_l470_470380


namespace circle_fits_in_rectangle_l470_470496

-- Definitions for the problem
def side_length_square := 1
def diameter_circle := 1
def width_rectangle := 20
def height_rectangle := 25
def num_squares := 120

-- Main theorem statement
theorem circle_fits_in_rectangle (side_length_square diameter_circle width_rectangle height_rectangle : ℝ) 
(num_squares : ℕ) (h_pos: side_length_square > 0) (d_pos: diameter_circle > 0) 
(h_width_pos: width_rectangle > 0) (h_height_pos: height_rectangle > 0) :
  ∃ (O : ℝ × ℝ), 
    (O.1 - diameter_circle / 2 > 0) ∧ 
    (O.1 + diameter_circle / 2 < width_rectangle) ∧
    (O.2 - diameter_circle / 2 > 0) ∧
    (O.2 + diameter_circle / 2 < height_rectangle) ∧
    ∀ (s : ℕ), s < num_squares →
      let square : (ℝ × ℝ) := (4 * s : ℝ, 4 * s : ℝ) in -- simplified arbitrary placement
      let dist := ((O.1 - square.1) ^ 2 + (O.2 - square.2) ^ 2) ^ (1 / 2) in
      dist > 1 / 2 :=
sorry

end circle_fits_in_rectangle_l470_470496


namespace largest_constant_for_interesting_sequence_l470_470558

def is_interesting_sequence (z : ℕ → ℂ) : Prop :=
  |z 1| = 1 ∧ ∀ n : ℕ, 0 < n → 4 * (z (n + 1))^2 + 2 * (z n) * (z (n + 1)) + (z n)^2 = 0

theorem largest_constant_for_interesting_sequence (C : ℝ) :
  (∀ (z : ℕ → ℂ) (m : ℕ), is_interesting_sequence z → 0 < m →
    |∑ i in finset.range m, z (i + 1)| ≥ C) ↔ C = real.sqrt 3 / 3 :=
sorry

end largest_constant_for_interesting_sequence_l470_470558


namespace jane_usable_sheets_l470_470721

theorem jane_usable_sheets :
  let brown_A4_total := 28
  let brown_A4_less_than_70 := 3
  let yellow_A4_total := 18
  let yellow_A4_less_than_70 := 5
  let yellow_A3_total := 9
  let yellow_A3_less_than_70 := 2
  brown_A4_total - brown_A4_less_than_70 +
  yellow_A4_total - yellow_A4_less_than_70 +
  yellow_A3_total - yellow_A3_less_than_70 = 45 :=
by
  rw [←Nat.sub_add_eq_add_sub],
  rw [add_comm],
  sorry

end jane_usable_sheets_l470_470721


namespace increasing_sequence_gcd_property_l470_470892

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → a m < a n

noncomputable def gcd_property (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, gcd (a m) (a n) = a (gcd m n)

theorem increasing_sequence_gcd_property 
  (a : ℕ → ℕ) (h_seq: sequence a) (h_gcd: gcd_property a) 
  (k : ℕ) (h_k: ∃ r s : ℕ, r < k ∧ k < s ∧ a k ^ 2 = a r * a s) :
  ∃ r s : ℕ, r < k ∧ k < s ∧ a k ^ 2 = a r * a s →
  (r ∣ k) ∧ (k ∣ s) :=
by
  sorry

end increasing_sequence_gcd_property_l470_470892


namespace yolk_count_proof_l470_470873

-- Define the conditions of the problem
def eggs_in_carton : ℕ := 12
def double_yolk_eggs : ℕ := 5
def single_yolk_eggs : ℕ := eggs_in_carton - double_yolk_eggs
def yolks_in_double_yolk_eggs : ℕ := double_yolk_eggs * 2
def yolks_in_single_yolk_eggs : ℕ := single_yolk_eggs
def total_yolks : ℕ := yolks_in_single_yolk_eggs + yolks_in_double_yolk_eggs

-- Stating the theorem to prove the total number of yolks is 17
theorem yolk_count_proof : total_yolks = 17 := 
by
  sorry

end yolk_count_proof_l470_470873


namespace ryan_weekly_commuting_time_l470_470804

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end ryan_weekly_commuting_time_l470_470804


namespace inequality_holds_for_real_numbers_l470_470322

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470322


namespace tiles_in_each_row_l470_470395

theorem tiles_in_each_row (area : ℝ) (tile_side : ℝ) (feet_to_inches : ℝ) (number_of_tiles : ℕ) :
  area = 400 ∧ tile_side = 8 ∧ feet_to_inches = 12 → number_of_tiles = 30 :=
by
  intros h
  cases h with h_area h_rest
  cases h_rest with h_tile_side h_feet_to_inches
  sorry

end tiles_in_each_row_l470_470395


namespace half_angle_quadrant_l470_470178

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_quadrant_l470_470178


namespace decimal_to_fraction_l470_470035

theorem decimal_to_fraction (a b : ℚ) (h : a = 3.56) (h1 : b = 56/100) (h2 : 56.gcd 100 = 4) :
  a = 89/25 := by
  sorry

end decimal_to_fraction_l470_470035


namespace slope_of_line_eq_slope_of_line_l470_470144

theorem slope_of_line_eq (x y : ℝ) (h : 4 * x + 6 * y = 24) : (6 * y = -4 * x + 24) → (y = - (2 : ℝ) / 3 * x + 4) :=
by
  intro h1
  sorry

theorem slope_of_line (x y m : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = - (2 : ℝ) / 3 * x + 4) : m = - (2 : ℝ) / 3 :=
by
  sorry

end slope_of_line_eq_slope_of_line_l470_470144


namespace gcd_36745_59858_l470_470590

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 :=
sorry

end gcd_36745_59858_l470_470590


namespace cos_sixty_deg_l470_470023

theorem cos_sixty_deg : Real.cos (Float.pi / 3) = 1 / 2 :=
by
  sorry

end cos_sixty_deg_l470_470023


namespace inequality_holds_for_all_reals_l470_470343

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470343


namespace tangentReflectionSolution_l470_470741

noncomputable def tangentReflectionProblem (A B : Circle) (m : Line) 
  (A_side m_side B_side : A.side = same ∧ B.side = same) : ℕ :=
  let B' := B.reflected_over(m) in 
  A.tangents_to(B') + B'.tangents_to(A)

theorem tangentReflectionSolution (A B : Circle) (m : Line) 
  (hA : A.side m = same_side) (hB : B.side m = same_side) : tangentReflectionProblem A B m hA hB = 4 := 
by 
  sorry

end tangentReflectionSolution_l470_470741


namespace part1_extreme_value_at_2_part2_increasing_function_l470_470194

noncomputable def f (a x : ℝ) := a * x - a / x - 2 * Real.log x

theorem part1_extreme_value_at_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y : ℝ, f a x ≥ f a y) → a = 4 / 5 ∧ f a 1/2 = 2 * Real.log 2 - 6 / 5 := by
  sorry

theorem part2_increasing_function (a : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (f a) x ≥ 0) → a ≥ 1 := by
  sorry

end part1_extreme_value_at_2_part2_increasing_function_l470_470194


namespace white_marbles_count_l470_470497

theorem white_marbles_count (total_marbles blue_marbles red_marbles : ℕ) (probability_red_or_white : ℚ)
    (h_total : total_marbles = 60)
    (h_blue : blue_marbles = 5)
    (h_red : red_marbles = 9)
    (h_probability : probability_red_or_white = 0.9166666666666666) :
    ∃ W : ℕ, W = total_marbles - blue_marbles - red_marbles ∧ probability_red_or_white = (red_marbles + W)/(total_marbles) ∧ W = 46 :=
by
  sorry

end white_marbles_count_l470_470497


namespace range_of_m_for_nonempty_solution_set_l470_470696

theorem range_of_m_for_nonempty_solution_set :
  {m : ℝ | ∃ x : ℝ, m * x^2 - m * x + 1 < 0} = {m : ℝ | m < 0} ∪ {m : ℝ | m > 4} :=
by sorry

end range_of_m_for_nonempty_solution_set_l470_470696


namespace angles_not_both_acute_l470_470621

noncomputable theory

-- Definitions of points and triangle intersections
variables {P : Type*} [euclidean_geometry P]
variables {A B C S A₁ B₁ C₁ : P}

-- Assume S is an internal point of triangle ABC
def is_internal (A B C S : P) : Prop := 
  euclidean_geometry.collinear A B C ∧ S ≠ A ∧ S ≠ B ∧ S ≠ C

-- Definitions of intersections
def intersection_with_side (A B C P : P) : Prop := 
  euclidean_geometry.collinear A B P ∧ euclidean_geometry.collinear S C P

-- Conditions based on the problem statement
axiom conditions 
  (hA : intersection_with_side A B C A₁)
  (hB : intersection_with_side B C A B₁)
  (hC : intersection_with_side C A B C₁)
  (hS : is_internal A B C S) : Prop 

-- The core theorem we need to prove
theorem angles_not_both_acute
  (h : conditions hA hB hC hS) : 
  quadrilateral_system.exists_non_acute_angles A B C S A₁ B₁ C₁ :=
sorry

end angles_not_both_acute_l470_470621


namespace part1_part2_l470_470671

theorem part1 (m : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y - 4 = 0 ∧ y = x + m) → -3 - 3 * Real.sqrt 2 < m ∧ m < -3 + 3 * Real.sqrt 2) :=
sorry

theorem part2 (m x1 x2 y1 y2 : ℝ) (h1 : x1 + x2 = -(m + 1)) (h2 : x1 * x2 = (m^2 + 4 * m - 4) / 2) 
(h3 : (x - x1) * (x - x2) + (x1 + m) * (x2 + m) = 0) : 
  m = -4 ∨ m = 1 →
  (∀ x y : ℝ, y = x + m ↔ x - y - 4 = 0 ∨ x - y + 1 = 0) :=
sorry

end part1_part2_l470_470671


namespace diagonals_in_seven_sided_polygon_l470_470210

open Nat

def binomial (n : ℕ) (k : ℕ) : ℕ := (n.choose k)

theorem diagonals_in_seven_sided_polygon : 
    binomial 7 2 - 7 = 14 :=
by
  -- We use the binomial coefficient to calculate the number of ways to connect any two vertices.
  let total_connections := binomial 7 2
  -- Given a seven-sided polygon, the number of sides is 7.
  let sides_of_polygon := 7
  -- The number of diagonals is total connections minus the sides.
  have total_connections_val : total_connections = 21 :=
    by
      simp [binomial, choose, factorial]
      norm_num
  have sides_val : sides_of_polygon = 7 := rfl
  have diagonals := total_connections - sides_of_polygon
  have diagonals_val : diagonals = 14 :=
    by
      simp [total_connections_val, sides_val]
      norm_num
  exact diagonals_val

end diagonals_in_seven_sided_polygon_l470_470210


namespace three_point_five_six_as_fraction_l470_470031

theorem three_point_five_six_as_fraction : (356 / 100 : ℝ) = (89 / 25 : ℝ) :=
begin
  sorry
end

end three_point_five_six_as_fraction_l470_470031


namespace max_slope_of_circle_tangent_l470_470695

theorem max_slope_of_circle_tangent (x y : ℝ) (h : (x-2)^2 + y^2 = 1) : 
  ∃ k : ℝ, k = y / x ∧ y / x ≤ √3 / 3 :=
sorry

end max_slope_of_circle_tangent_l470_470695


namespace power_series_expansion_and_convergence_l470_470142

noncomputable def power_series_expansion_ln (x : ℝ) : ℝ :=
  ∑ n in Nat.range 0 ∞, (-1)^n * (2*n)! / (4^n * (n!)^2) * x^(2*n+1) / (2*n+1)

theorem power_series_expansion_and_convergence (x : ℝ) :
  (|x| ≤ 1) →
  (ln (x + sqrt (1 + x^2)) = power_series_expansion_ln x) ∧
  (|x| ≤ 1)
:= by
  sorry

end power_series_expansion_and_convergence_l470_470142


namespace determine_a_l470_470634

noncomputable def f (x : ℝ) (a : ℝ) (g : ℝ → ℝ) : ℝ := 2 * a^x * g x

theorem determine_a (g : ℝ → ℝ) (a : ℝ) (h1 : ∀ x, f x a g = 2 * a^x * g x)
  (h2 : ∀ x, g x ≠ 0) (h3 : ∀ x, f x a g * (deriv g x) < (deriv (f x a g) x) * g x)
  (h4 : 2 * a + 2 / a = 5) (ha_gt_zero : 0 < a) (ha_ne_one : a ≠ 1) : a = 2 :=
sorry

end determine_a_l470_470634


namespace minimum_value_of_a2b_l470_470690

noncomputable def minimum_value (a b : ℝ) := a + 2 * b

theorem minimum_value_of_a2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / (2 * a + b) + 1 / (b + 1) = 1) :
  minimum_value a b = (2 * Real.sqrt 3 + 1) / 2 :=
sorry

end minimum_value_of_a2b_l470_470690


namespace inequality_holds_for_real_numbers_l470_470323

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470323


namespace no_real_solution_matrix_l470_470297

theorem no_real_solution_matrix (x : ℝ) :
  let a := 3 * x,
      b := 2,
      c := 4,
      d := x,
      matrix_value := a * d - b * c,
      equation_rhs := 2 * x^2 - 3 * x - 4
  in matrix_value ≠ equation_rhs :=
by
  let a := 3 * x,
      b := 2,
      c := 4,
      d := x,
      matrix_value := a * d - b * c,
      equation_rhs := 2 * x^2 - 3 * x - 4
  show matrix_value ≠ equation_rhs
  sorry

end no_real_solution_matrix_l470_470297


namespace number_of_tiles_per_row_l470_470410

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l470_470410


namespace parabola_tangency_point_l470_470935

-- Definitions of the parabola equations
def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 20
def parabola2 (y : ℝ) : ℝ := y^2 + 36 * y + 380

-- The proof statement
theorem parabola_tangency_point : 
  ∃ (x y : ℝ), 
    parabola1 x = y ∧ parabola2 y = x ∧ x = -9 / 2 ∧ y = -35 / 2 :=
by
  sorry

end parabola_tangency_point_l470_470935


namespace man_avg_speed_l470_470877

noncomputable def average_speed_of_travel 
  (speed_up : ℝ) (speed_down : ℝ) (altitude_m : ℝ) : ℝ :=
  let distance_km := 2 * altitude_m / 1000 in
  let time_up := distance_km / speed_up in
  let time_down := distance_km / speed_down in
  distance_km / (time_up + time_down)

theorem man_avg_speed 
  (speed_up : ℝ) (speed_down : ℝ) (altitude_m : ℝ) :
  speed_up = 15 → speed_down = 28 → altitude_m = 230 →
  average_speed_of_travel speed_up speed_down altitude_m ≈ 19.54 := sorry

end man_avg_speed_l470_470877


namespace problem_l470_470111

noncomputable def g (a b c d h x : ℝ) : ℝ := 
  (a * (x + h) + b) / (c * (x + h) + d)

variable {a b c d h : ℝ}

axiom hnonzero : h ≠ 0
axiom anzero : a ≠ 0
axiom bnzero : b ≠ 0
axiom cnzero : c ≠ 0
axiom dnzero : d ≠ 0

theorem problem (h: h ≠ 0) (a: a ≠ 0) (b: b ≠ 0) (c: c ≠ 0) (d: d ≠ 0) : 
  (∀ x : ℝ, g a b c d h (g a b c d h x) = x) → a + d - 2 * c * h = 0 :=
sorry

end problem_l470_470111


namespace square_area_l470_470021

def point := (ℝ × ℝ)

def P : point := (1, 1)
def Q : point := (-4, 2)
def R : point := (-3, 7)
def S : point := (2, 6)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def is_square (P Q R S : point) : Prop :=
  distance P Q = distance Q R ∧
  distance Q R = distance R S ∧
  distance R S = distance S P ∧
  distance S P = distance P Q

theorem square_area : is_square P Q R S → (distance P Q)^2 = 26 :=
by
  sorry

end square_area_l470_470021


namespace find_initial_volume_of_mixture_l470_470878

def initial_volume_of_mixture 
(V : ℝ) 
(initial_percentage_water : ℝ) 
(final_percentage_water : ℝ) 
(additional_water : ℝ) 
(h1 : initial_percentage_water = 0.20) 
(h2 : final_percentage_water = 0.25) 
(h3 : additional_water = 8.333333333333334) : Prop :=
let initial_water := initial_percentage_water * V in
let new_total_volume := V + additional_water in
let new_water_volume := initial_water + additional_water in
final_percentage_water * new_total_volume = new_water_volume

theorem find_initial_volume_of_mixture 
(V : ℝ) 
(h1 : 0.20 = 0.20) 
(h2 : 0.25 = 0.25) 
(h3 : 8.333333333333334 = 8.333333333333334) : 
initial_volume_of_mixture 125 0.20 0.25 8.333333333333334 h1 h2 h3 :=
sorry

end find_initial_volume_of_mixture_l470_470878


namespace num_of_log_diffs_is_22_l470_470603

open Set

-- Defining the set of numbers and the difference of logs
def num_set := {1, 2, 3, 4, 5, 6}

-- Function to calculate the difference of logs
def log_diff (a b : ℝ) : ℝ := Real.log a - Real.log b

-- Defining a predicate that ensures that a and b are distinct elements of num_set
def distinct_elements (a b : ℝ) : Prop :=
a ∈ num_set ∧ b ∈ num_set ∧ a ≠ b

-- The theorem statement
theorem num_of_log_diffs_is_22 : 
  (∃ S : Set ℝ, (S = { log_diff a b | a b : ℝ, distinct_elements a b } ∧ S.card = 22)) :=
sorry

end num_of_log_diffs_is_22_l470_470603


namespace percentage_difference_between_maximum_and_minimum_changes_is_40_l470_470095

-- Definitions of initial and final survey conditions
def initialYesPercentage : ℝ := 0.40
def initialNoPercentage : ℝ := 0.60
def finalYesPercentage : ℝ := 0.80
def finalNoPercentage : ℝ := 0.20
def absenteePercentage : ℝ := 0.10

-- Main theorem stating the problem
theorem percentage_difference_between_maximum_and_minimum_changes_is_40 :
  let attendeesPercentage := 1 - absenteePercentage
  let adjustedFinalYesPercentage := finalYesPercentage / attendeesPercentage
  let minChange := adjustedFinalYesPercentage - initialYesPercentage
  let maxChange := initialYesPercentage + minChange
  maxChange - minChange = 0.40 :=
by
  -- Proof is omitted
  sorry

end percentage_difference_between_maximum_and_minimum_changes_is_40_l470_470095


namespace sum_of_slopes_range_l470_470162

theorem sum_of_slopes_range (p b : ℝ) (hpb : 2 * p > b) (hp : p > 0) 
  (K1 K2 : ℝ) (A B : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1) (hB : B.2^2 = 2 * p * B.1)
  (hl1 : A.2 = A.1 + b) (hl2 : B.2 = B.1 + b) 
  (hA_pos : A.2 > 0) (hB_pos : B.2 > 0) :
  4 < K1 + K2 :=
sorry

end sum_of_slopes_range_l470_470162


namespace george_total_payment_in_dollars_l470_470156
noncomputable def total_cost_in_dollars : ℝ := 
  let sandwich_cost : ℝ := 4
  let juice_cost : ℝ := 2 * sandwich_cost * 0.9
  let coffee_cost : ℝ := sandwich_cost / 2
  let milk_cost : ℝ := 0.75 * (sandwich_cost + juice_cost)
  let milk_cost_dollars : ℝ := milk_cost * 1.2
  let chocolate_bar_cost_pounds : ℝ := 3
  let chocolate_bar_cost_dollars : ℝ := chocolate_bar_cost_pounds * 1.25
  let total_euros_in_items : ℝ := 2 * sandwich_cost + juice_cost + coffee_cost
  let total_euros_to_dollars : ℝ := total_euros_in_items * 1.2
  total_euros_to_dollars + milk_cost_dollars + chocolate_bar_cost_dollars

theorem george_total_payment_in_dollars : total_cost_in_dollars = 38.07 := by
  sorry

end george_total_payment_in_dollars_l470_470156


namespace work_time_for_two_people_l470_470272

theorem work_time_for_two_people (individual_time : ℕ) (num_people : ℕ) (combined_time : ℕ) 
    (h1 : individual_time = 10)
    (h2 : num_people = 2)
    (h3 : ∀ t n, combined_time * n = t) : 
  combined_time = 5 :=
by
  -- Assumptions
  have h4 : 10 = individual_time, from h1
  have h5 : 2 = num_people, from h2
  
  -- Proof (skipped)
  sorry

end work_time_for_two_people_l470_470272


namespace simplify_fraction_expression_l470_470478

variable (d : ℝ)

theorem simplify_fraction_expression :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := 
by
  sorry

end simplify_fraction_expression_l470_470478


namespace general_formula_sum_formula_and_min_value_l470_470257

-- Define the arithmetic sequence {a_n} with given conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  (a 3 = 2) ∧ (a 2 + a 6 = 8) ∧ (∃ d : ℤ, ∀ n : ℕ, a n = a 1 + n * d)

-- Prove the general formula for a_n given the conditions
theorem general_formula (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n : ℕ, a n = 2 * n - 4 :=
sorry

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) :=
  (∑ i in Finset.range (n + 1), a i)

-- Prove the formula for S_n and its minimum value
theorem sum_formula_and_min_value (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n : ℕ,
    sum_of_first_n_terms a n = n * (n - 3) ∧
    (sum_of_first_n_terms a 1 = -2 ∧ sum_of_first_n_terms a 2 = -2) :=
sorry

end general_formula_sum_formula_and_min_value_l470_470257


namespace log_eq_three_has_two_integer_solutions_l470_470937

theorem log_eq_three_has_two_integer_solutions :
  {a : ℝ | log 10 (a ^ 2 - 10 * a) = 3} = {37, -27} :=
sorry

end log_eq_three_has_two_integer_solutions_l470_470937


namespace midpoint_incircle_center_l470_470263

theorem midpoint_incircle_center (A B C P Q M : Point) (AB_eq_AC : AB = AC)
  (h_isosceles : IsoscelesTriangle A B C AB_eq_AC)
  (h_circ_tangent : CircleTangentInternalToCircumcircleOfABC A B C P Q)
  (h_tangent_to_AB_AC : TangentToABAndAC A B C P Q) :
  Midpoint P Q M ∧ IncenterOfTriangle A B C M :=
sorry

end midpoint_incircle_center_l470_470263


namespace delta_comparison_eps_based_on_gamma_l470_470265

-- Definitions for the problem
variable {α β γ δ ε : ℝ}
variable {A B C : Type}
variable (s f m : Type)

-- Conditions from problem
variable (triangle_ABC : α ≠ β)
variable (median_s_from_C : s)
variable (angle_bisector_f : f)
variable (altitude_m : m)
variable (angle_between_f_m : δ = sorry)
variable (angle_between_f_s : ε = sorry)
variable (angle_at_vertex_C : γ = sorry)

-- Main statement to prove
theorem delta_comparison_eps_based_on_gamma (h1 : α ≠ β) (h2 : δ = sorry) (h3 : ε = sorry) (h4 : γ = sorry) :
  if γ < 90 then δ < ε else if γ = 90 then δ = ε else δ > ε :=
sorry

end delta_comparison_eps_based_on_gamma_l470_470265


namespace sum_of_squares_of_exponents_of_992_l470_470929

theorem sum_of_squares_of_exponents_of_992 :
  ∑ i in {9, 8, 7, 6, 5}, i^2 = 255 :=
by {
  sorry
}

end sum_of_squares_of_exponents_of_992_l470_470929


namespace technicians_count_l470_470778

def avg_salary_all := 9500
def avg_salary_technicians := 12000
def avg_salary_rest := 6000
def total_workers := 12

theorem technicians_count : 
  ∃ (T R : ℕ), 
  (T + R = total_workers) ∧ 
  ((T * avg_salary_technicians + R * avg_salary_rest) / total_workers = avg_salary_all) ∧ 
  (T = 7) :=
by sorry

end technicians_count_l470_470778


namespace circle_radius_tangent_lines_l470_470064

theorem circle_radius_tangent_lines (k : ℝ) (hk : k > 6) :
  ∃ r : ℝ, ∀ x y : ℝ, (x - 0)^2 + (y - k)^2 = r^2 ∧ 
           (y = x ∨ y = -x ∨ y = 6) → r = 6 * real.sqrt 2 + 6 := 
sorry

end circle_radius_tangent_lines_l470_470064


namespace G_is_even_l470_470233

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_even (a : ℝ) (F : ℝ → ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1)
  (hF : ∀ x : ℝ, F (-x) = - F x) : 
  ∀ x : ℝ, G F a (-x) = G F a x :=
by 
  sorry

end G_is_even_l470_470233


namespace average_of_quantities_l470_470412

theorem average_of_quantities (a1 a2 a3 a4 a5 : ℝ) :
  ((a1 + a2 + a3) / 3 = 4) →
  ((a4 + a5) / 2 = 21.5) →
  ((a1 + a2 + a3 + a4 + a5) / 5 = 11) :=
by
  intros h3 h2
  sorry

end average_of_quantities_l470_470412


namespace monotonicity_case1_monotonicity_case2_lower_bound_fx_l470_470662

noncomputable def f (x a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 {x a : ℝ} (h : a ≤ 0) : 
  ∀ x, (differentiable ℝ (λ x, f x a) ∧ deriv (λ x, f x a) x ≤ -1) :=
sorry

theorem monotonicity_case2 {x a : ℝ} (h : 0 < a) : 
  ∀ x, (x < Real.log (1 / a) → (f x a) < (f (Real.log (1 / a)) a)) ∧ (Real.log (1 / a) < x → (f (Real.log (1 / a)) a) < f x a) :=
sorry

theorem lower_bound_fx {x a : ℝ} (h : 0 < a) : f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_case1_monotonicity_case2_lower_bound_fx_l470_470662


namespace boat_capacity_problem_l470_470800

variables (L S : ℕ)

theorem boat_capacity_problem
  (h1 : L + 4 * S = 46)
  (h2 : 2 * L + 3 * S = 57) :
  3 * L + 6 * S = 96 :=
sorry

end boat_capacity_problem_l470_470800


namespace scientific_notation_proof_l470_470581

-- Let's start by providing the necessary constants and definitions.
def scientific_notation (a : Float) (b : Int) : Prop :=
  a = 3.04 ∧ b = -10

theorem scientific_notation_proof :
  scientific_notation 3.04 (-10) :=
begin
  sorry
end

end scientific_notation_proof_l470_470581


namespace max_quadratic_function_l470_470000

def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

theorem max_quadratic_function : ∃ x, quadratic_function x = 7 ∧ ∀ x', quadratic_function x' ≤ 7 :=
by
  sorry

end max_quadratic_function_l470_470000


namespace exterior_angle_C_of_triangle_ABC_l470_470890

theorem exterior_angle_C_of_triangle_ABC 
  (square : Type)
  (inscribed_circle : circle)
  (A B C : point)
  (AB : side square)
  (BC : line)
  (mid_point : point)
  (square_property : is_square square inscribed_circle)
  (connect_midpoint : is_connected_to_midpoint B C A)
  (angle_ABC_eq_90 : ∠ ABC = 90) :
  ∠ exterior_C = 90 :=
sorry

end exterior_angle_C_of_triangle_ABC_l470_470890


namespace unique_solution_pairs_l470_470946

theorem unique_solution_pairs :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 = 4 * c) ∧ (c^2 = 4 * b) :=
sorry

end unique_solution_pairs_l470_470946


namespace number_of_tiles_per_row_l470_470409

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l470_470409


namespace min_sum_moduli_diff_l470_470269

theorem min_sum_moduli_diff (n : ℕ) (hn : 1 < n) (a : Fin n → ℕ) (hperm : ∀ i, a i ∈ Finset.range n.succ):
  ∃ a : Fin n → ℕ, 
  (∀ i, |a i - a (i + 1) % n| = 1) ∧ 
  (|a 0 - a (n - 1)| = n - 1) 
  → (∑ i in Finset.range n, |a i - a ((i + 1) % n)|) = 2 * n - 2 :=
sorry

end min_sum_moduli_diff_l470_470269


namespace simplify_expression_part1_simplify_expression_part2_l470_470147

-- Part (1)
theorem simplify_expression_part1 : 
  2 * 7^(2/3) + 2 * (Real.exp 1 - 1)^0 + 1 / (Real.sqrt 5 + 2) - 16^(1/4) + (3 - Real.pi)^4^(1/4) = 4 + Real.pi + Real.sqrt 5 :=
sorry

-- Part (2)
theorem simplify_expression_part2 :
  ((Real.logBase 2 3) + (Real.logBase 5 3)) * ((Real.logBase 3 5) + (Real.logBase 9 5)) * (Real.log 2) = 3 / 2 :=
sorry

end simplify_expression_part1_simplify_expression_part2_l470_470147


namespace isosceles_right_triangle_hypotenuse_length_l470_470592

theorem isosceles_right_triangle_hypotenuse_length (A B C : ℝ) (h1 : (A = 0) ∧ (B = 0) ∧ (C = 1)) (h2 : AC = 5) (h3 : BC = 5) : 
  AB = 5 * Real.sqrt 2 := 
sorry

end isosceles_right_triangle_hypotenuse_length_l470_470592


namespace tiles_per_row_l470_470389

noncomputable def area_square_room : ℕ := 400
noncomputable def side_length_feet : ℕ := nat.sqrt area_square_room
noncomputable def side_length_inches : ℕ := side_length_feet * 12
noncomputable def tile_size : ℕ := 8

theorem tiles_per_row : (side_length_inches / tile_size) = 30 := 
by
  have h1: side_length_feet = 20 := by sorry
  have h2: side_length_inches = 240 := by sorry
  have h3: side_length_inches / tile_size = 30 := by sorry
  exact h3

end tiles_per_row_l470_470389


namespace ratio_of_number_halving_l470_470442

theorem ratio_of_number_halving (x y : ℕ) (h1 : y = x / 2) (h2 : y = 9) : x / y = 2 :=
by
  sorry

end ratio_of_number_halving_l470_470442


namespace charlotte_age_l470_470552

theorem charlotte_age : 
  ∀ (B C E : ℝ), 
    (B = 4 * C) → 
    (E = C + 5) → 
    (B = E) → 
    C = 5 / 3 :=
by
  intros B C E h1 h2 h3
  /- start of the proof -/
  sorry

end charlotte_age_l470_470552


namespace train_cross_post_time_l470_470082

theorem train_cross_post_time (length_of_train : ℕ) 
  (platform_length : ℕ) (time_to_cross_platform : ℕ) 
  (total_length: length_of_train + platform_length = 250)
  (speed_of_train : length_of_train + platform_length / time_to_cross_platform = 10): 
  (time_to_cross_post : (length_of_train / speed_of_train) = 15) :=
begin
  sorry,
end

end train_cross_post_time_l470_470082


namespace verify_second_derivative_geometric_sequence_l470_470717

noncomputable def geometric_sequence := (a : ℕ → ℝ) (r : ℝ) (a0 : a 1 = 2) (a7 : a 8 = 4) 
  (geo : ∀ n, a n = 2 * r^(n - 1)) : Prop := sorry

noncomputable def f (a : ℕ → ℝ) (x : ℝ) := 
  x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

theorem verify_second_derivative_geometric_sequence
  (f : ℝ → ℝ)
  [differentiable ℝ f]
  (a : ℕ → ℝ)
  (r : ℝ)
  (a0 : a 1 = 2)
  (a7 : a 8 = 4)
  (geo : ∀ n, a n = 2 * r^(n - 1)) :
  ((deriv^[2] f) 0) = 2 ^ 12 :=
begin
  sorry,
end

end verify_second_derivative_geometric_sequence_l470_470717


namespace integral_sum_l470_470579

open Real

theorem integral_sum (h1 : ∫ x in 0..2, sqrt (4 - x^2) = π)
                     (h2 : ∫ x in 0..2, x = 2) :
    (∫ x in 0..2, (sqrt (4 - x^2) + x)) = π + 2 :=
by {
  sorry
}

end integral_sum_l470_470579


namespace find_c_l470_470133

theorem find_c (a b c : ℤ) (h1 : c ≥ 0) (h2 : ¬∃ m : ℤ, 2 * a * b = m^2)
  (h3 : ∀ n : ℕ, n > 0 → (a^n + (2 : ℤ)^n) ∣ (b^n + c)) :
  c = 0 ∨ c = 1 :=
by
  sorry

end find_c_l470_470133


namespace vector_pointing_to_line_l470_470112

theorem vector_pointing_to_line (t k a b : ℝ) (h1 : a = 5 * t + 3) (h2 : b = t + 3) (h3 : a = 3 * k) (h4 : b = k) : 
  ∃ a b : ℝ, a = 18 ∧ b = 6 :=
by
  use 18
  use 6
  sorry

end vector_pointing_to_line_l470_470112


namespace proj_magnitude_cos_theta_l470_470285

variables {V : Type*} [inner_product_space ℝ V]
variables (v w : V)
variable (theta : ℝ)

-- Conditions in Lean
def dot_product_condition : Prop := inner_product_space.inner v w = 4
def norm_w_condition : Prop := ‖w‖ = 8

-- Questions to prove
def proj_magnitude_statement (h₁ : dot_product_condition v w) (h₂ : norm_w_condition w) : Prop :=
‖(inner_product_space.proj w v)‖ = 4

def cos_theta_expression (h₁ : dot_product_condition v w) (h₂ : norm_w_condition w) : Prop :=
Real.cos theta = (inner_product_space.inner v w) / (inner_product_space.norm v * 8)

-- Lean 4 theorem statements
theorem proj_magnitude (h₁ : dot_product_condition v w) (h₂ : norm_w_condition w) : 
  proj_magnitude_statement v w h₁ h₂ :=
by
  sorry

theorem cos_theta (h₁ : dot_product_condition v w) (h₂ : norm_w_condition w) :
  cos_theta_expression v w θ h₁ h₂ :=
by
  sorry

end proj_magnitude_cos_theta_l470_470285


namespace hyperbola_eccentricity_rhombus_l470_470784

theorem hyperbola_eccentricity_rhombus
  (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let e := (1 : ℝ) + real.sqrt 3 in
  let C := set_of (λ p : ℝ × ℝ, (p.2 ^ 2) / (a ^ 2) - (p.1 ^ 2) / (b ^ 2) = 1) in
  let F := (0, real.sqrt (a^2 + b^2)) in
  ∀ A B : ℝ × ℝ, A ∈ C → B ∈ C → (0, 0) = F ∧ (∃ x_A y_A x_B y_B : ℝ,
      A = (x_A, y_A) ∧ B = (x_B, y_B) ∧ (x_A - 0)^2 + (y_A - (real.sqrt (a^2 + b^2)))^2 = (x_B - 0)^2 + (y_B - (real.sqrt (a^2 + b^2)))^2) →
  (1 + real.sqrt 3) = e := sorry

end hyperbola_eccentricity_rhombus_l470_470784


namespace rational_root_theorem_example_l470_470518

theorem rational_root_theorem_example
  (b3 b2 b1 : ℤ) :
  let p := 7 * X^4 + C b3 * X^3 + C b2 * X^2 + C b1 * X + 18
  in polynomial.rat_root_set p = { 1, -1, 2, -2, 3, -3, 6, -6, 9, -9, 18, -18,
    1/7, -1/7, 2/7, -2/7, 3/7, -3/7, 6/7, -6/7, 9/7, -9/7, 18/7, -18/7 }.card
    := by
  sorry

end rational_root_theorem_example_l470_470518


namespace number_of_ordered_pairs_l470_470685

-- Introduce the necessary definitions and theorems
theorem number_of_ordered_pairs (M N : ℕ) (hM : M > 0) (hN : N > 0) : 
  (M ∣ 128) ∧ (∃ k : ℕ, M * k = 128) ∧ (N = 128 / M) → 
  (∃ ! (p : ℕ × ℕ), p.fst > 0 ∧ p.snd > 0 ∧ p.fst * p.snd = 128) ∧ p.snd = 128 / p.fst :=
begin
  sorry
end

end number_of_ordered_pairs_l470_470685


namespace complex_power_difference_l470_470227

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 40 - (1 - i) ^ 40 = 0 := by 
  sorry

end complex_power_difference_l470_470227


namespace find_x_l470_470585

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 135) (h2 : x > 0) : x = 11.25 :=
by
  sorry

end find_x_l470_470585


namespace total_cost_l470_470303

theorem total_cost
  (cost_berries   : ℝ := 11.08)
  (cost_apples    : ℝ := 14.33)
  (cost_peaches   : ℝ := 9.31)
  (cost_grapes    : ℝ := 7.50)
  (cost_bananas   : ℝ := 5.25)
  (cost_pineapples: ℝ := 4.62)
  (total_cost     : ℝ := cost_berries + cost_apples + cost_peaches + cost_grapes + cost_bananas + cost_pineapples) :
  total_cost = 52.09 :=
by
  sorry

end total_cost_l470_470303


namespace students_enrolled_in_only_english_l470_470706

theorem students_enrolled_in_only_english (total_students both_english_german total_german : ℕ) (h1 : total_students = 40) (h2 : both_english_german = 12) (h3 : total_german = 22) (h4 : ∀ s, s < 40) :
  (total_students - (total_german - both_english_german) - both_english_german) = 18 := 
by {
  sorry
}

end students_enrolled_in_only_english_l470_470706


namespace tens_digit_of_factorial_difference_l470_470465

-- Definitions for conditions
def is_divisible_by_10000 (n: ℕ) : Prop := n % 10000 = 0

-- The problem as a Lean theorem 
theorem tens_digit_of_factorial_difference :
  is_divisible_by_10000 (25! - 20!) → (25! - 20!) % 100 = 0 :=
by
  intro h
  sorry

end tens_digit_of_factorial_difference_l470_470465


namespace a_perp_b_min_value_k_t_l470_470681

variables (θ t k : ℝ)
variables (a b x y : ℝ × ℝ)

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (Real.cos (-θ), Real.sin (-θ))
def b : ℝ × ℝ := (Real.cos (π / 2 - θ), Real.sin (π / 2 - θ))

-- Prove a is perpendicular to b
theorem a_perp_b : a θ t k ⬝ b θ t k = 0 :=
sorry

-- Definitions for the vectors x and y
def x : ℝ × ℝ := a θ t k + (t^2 + 3) • b θ t k
def y : ℝ × ℝ := -k • a θ t k + t • b θ t k

-- Given x is perpendicular to y, finding minimum value of (k + t^2) / t
theorem min_value_k_t (hx_perp_hy : x θ t k ⬝ y θ t k = 0) : ∃ t, (k + t^2) / t = 11 / 4 :=
sorry

end a_perp_b_min_value_k_t_l470_470681


namespace f_inequality_l470_470659

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a > 2 * Real.log a + 3 / 2 :=
sorry

end f_inequality_l470_470659


namespace absolute_value_of_h_l470_470563

theorem absolute_value_of_h (h : ℝ) (roots : Set ℝ) 
  (h_quad_eq : ∀ x ∈ roots, x^2 + 2*h*x - 8 = 0)
  (h_sum_of_squares : ∑ x in roots, x^2 = 20) 
  : |h| = 1 :=
sorry

end absolute_value_of_h_l470_470563


namespace probability_of_two_positive_roots_in_interval_l470_470261

noncomputable def probability_two_positive_roots : ℝ :=
  let interval := set.Icc (-1 : ℝ) 5 in
  let favorable1 := set.Ioo (3 / 4 : ℝ) 1 in
  let favorable2 := set.Ici (3 : ℝ) in
  let favorable := (favorable1 ∪ favorable2) ∩ interval in
  (set.finite_measure (interval ∩ favorable1).measure / set.finite_measure interval.measure) +
  (set.finite_measure (interval ∩ favorable2).measure / set.finite_measure interval.measure)

theorem probability_of_two_positive_roots_in_interval : probability_two_positive_roots = 3 / 8 :=
sorry

end probability_of_two_positive_roots_in_interval_l470_470261


namespace water_evaporation_each_day_l470_470511

theorem water_evaporation_each_day :
  let initial_water := 10 -- ounces
  let evaporation_percentage := 0.06 -- 6%
  let period := 20 -- days
  let total_evaporation := initial_water * evaporation_percentage -- ounces
  let evaporation_per_day := total_evaporation / period -- ounces per day
  evaporation_per_day = 0.03 :=
by
  let initial_water := 10 
  let evaporation_percentage := 0.06 
  let period := 20 
  let total_evaporation := initial_water * evaporation_percentage 
  let evaporation_per_day := total_evaporation / period 
  have h1 : total_evaporation = 0.6, by sorry
  have h2 : evaporation_per_day = total_evaporation / period, by sorry
  have h3 : evaporation_per_day = 0.03, by sorry
  exact h3

end water_evaporation_each_day_l470_470511


namespace find_m_l470_470206

open Real

noncomputable def vector_a : ℝ × ℝ := sorry
noncomputable def vector_b : ℝ × ℝ := sorry

theorem find_m 
  (h1 : ‖vector_a‖ = 3):
  (h2 : ‖vector_b‖ = 2):
  (h3 : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 3):
  (h4 : (vector_a.1 - m * vector_b.1) * vector_a.1 + (vector_a.2 - m * vector_b.2) * vector_a.2 = 0):
  m = 3 :=
sorry

end find_m_l470_470206


namespace bus_rental_costs_correct_cost_effective_rental_plan_l470_470221

noncomputable def bus_rental_costs : ℕ → ℕ → ℕ := sorry

noncomputable def rental_plan_cost : ℕ → ℕ → ℕ := sorry

theorem bus_rental_costs_correct :
  let x := 400 in let y := 300 in
  (bus_rental_costs x y = (x, y)) :=
by sorry

theorem cost_effective_rental_plan :
  let total_cost := 2300 in
  let teachers := 6 in let students := 234 in
  let bus_capacity_large := 45 in let bus_capacity_small := 30 in
  let large_bus_cost := 400 in let small_bus_cost := 300 in
  ∃ (large small : ℕ),
  (large ≥ 4 ∧ large ≤ 5) ∧ (small = 6 - large) ∧ (teachers + students ≤ large * bus_capacity_large + small * bus_capacity_small) ∧ (rental_plan_cost large small ≤ total_cost) :=
by sorry

end bus_rental_costs_correct_cost_effective_rental_plan_l470_470221


namespace digits_of_2_pow_1000_l470_470211

theorem digits_of_2_pow_1000 : 
  let lg2 := 0.30102999566 in
  let lg := Real.log10 in
  ∀ (x : ℕ), x = 2^1000 → (Nat.floor (lg (x : ℝ)) + 1) = 302 :=
by
  intro x hx
  have h1 : (x : ℝ) = 2^1000 := by exact_mod_cast hx
  sorry

end digits_of_2_pow_1000_l470_470211


namespace five_chosen_product_l470_470686

theorem five_chosen_product (s : Finset ℕ) (h1 : s = {1, 2, 3, 4, 5, 6, 7}) 
  (h2 : s.card = 7) (h3 : ∃ s' : Finset ℕ, s'.card = 5 ∧ 
    (∀ s'' ∈ s'.powerset.filter (λ t, t.card = 2), 
      let p := s.product id / s''.product id
      in (∃ t ∈ s.powerset.filter (λ t, t.card = 2), t.product id = s''.product id ∧ 
        (s.sum id - t.sum id) % 2 = 0 ↔ (s'.sum id - s''.sum id) % 2 = 1))
  )
: ∃ (p : ℕ), p = 420 := 
by
  sorry

end five_chosen_product_l470_470686


namespace triangle_coordinates_sum_l470_470808

noncomputable def coordinates_of_triangle_A (p q : ℚ) : Prop :=
  let B := (12, 19)
  let C := (23, 20)
  let area := ((B.1 * C.2 + C.1 * q + p * B.2) - (B.2 * C.1 + C.2 * p + q * B.1)) / 2 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let median_slope := (q - M.2) / (p - M.1)
  area = 60 ∧ median_slope = 3 

theorem triangle_coordinates_sum (p q : ℚ) 
(h : coordinates_of_triangle_A p q) : p + q = 52 := 
sorry

end triangle_coordinates_sum_l470_470808


namespace cubic_binomial_values_l470_470183

theorem cubic_binomial_values (m n : ℕ) :
  (3 * (x : ℝ) ^ n - (m - 1) * x + 1 = 3 * x ^ 3 - 0 * x + 1) →
  (n = 3 ∧ m = 1) :=
  begin
    sorry
  end

end cubic_binomial_values_l470_470183


namespace juice_distribution_l470_470810

theorem juice_distribution (C : ℝ) (hC : C > 0) :
  let total_juice := (2 / 3) * C in
  let juice_per_cup := total_juice / 6 in
  let percentage_per_cup := (juice_per_cup / C) * 100 in
  percentage_per_cup = 100 / 9 := 
by
  sorry

end juice_distribution_l470_470810


namespace probability_positive_difference_ge_three_l470_470019

/-- Given a set of numbers {1, 2, ..., 10}, the probability of selecting two numbers with
    a positive difference of 3 or greater is 28/45. -/
theorem probability_positive_difference_ge_three :
  (∑ n in finset.range 10, ∑ m in finset.range n, if abs (n - m) >= 3 then 1 else 0) / (nat.choose 10 2) = 28 / 45 := 
sorry

end probability_positive_difference_ge_three_l470_470019


namespace total_savings_percentage_l470_470505

theorem total_savings_percentage :
  let coat_price := 100
  let hat_price := 50
  let shoes_price := 75
  let coat_discount := 0.30
  let hat_discount := 0.40
  let shoes_discount := 0.25
  let original_total := coat_price + hat_price + shoes_price
  let coat_savings := coat_price * coat_discount
  let hat_savings := hat_price * hat_discount
  let shoes_savings := shoes_price * shoes_discount
  let total_savings := coat_savings + hat_savings + shoes_savings
  let savings_percentage := (total_savings / original_total) * 100
  savings_percentage = 30.556 :=
by
  sorry

end total_savings_percentage_l470_470505


namespace g_invertible_on_interval_and_largest_l470_470100

def g (x : ℝ) : ℝ := 3*x^2 + 6*x - 2

theorem g_invertible_on_interval_and_largest (x : ℝ) :
  ∃ (I : set ℝ), I = set.Iic (-1) ∧ 
                 ∀ x ∈ I, ∃ y, g y = x ∧
                 ∀ I' : set ℝ, I ⊆ I' → g.invertible_on I → 
                 I = I' :=
by sorry

end g_invertible_on_interval_and_largest_l470_470100


namespace maximize_revenue_l470_470755

-- Define the problem conditions
def is_valid (x y : ℕ) : Prop :=
  x + y ≤ 60 ∧ 6 * x + 30 * y ≤ 600

-- Define the objective function
def revenue (x y : ℕ) : ℚ :=
  2.5 * x + 7.5 * y

-- State the theorem with the given conditions
theorem maximize_revenue : 
  (∃ x y : ℕ, is_valid x y ∧ ∀ a b : ℕ, is_valid a b → revenue x y >= revenue a b) ∧
  ∃ x y, is_valid x y ∧ revenue x y = revenue 50 10 := 
sorry

end maximize_revenue_l470_470755


namespace length_of_faster_train_proof_l470_470848

-- Definitions based on the given conditions
def faster_train_speed_kmh := 72 -- in km/h
def slower_train_speed_kmh := 36 -- in km/h
def time_to_cross_seconds := 18 -- in seconds

-- Conversion factor from km/h to m/s
def kmh_to_ms := 5 / 18

-- Define the relative speed in m/s
def relative_speed_ms := (faster_train_speed_kmh - slower_train_speed_kmh) * kmh_to_ms

-- Length of the faster train in meters
def length_of_faster_train := relative_speed_ms * time_to_cross_seconds

-- The theorem statement for the Lean prover
theorem length_of_faster_train_proof : length_of_faster_train = 180 := by
  sorry

end length_of_faster_train_proof_l470_470848


namespace top_leftmost_tile_is_tile_B_l470_470925

-- Define the labels for each tile
inductive Tile
| A | B | C | D | E

open Tile

-- Define the sides labeled with integers, for each tile
def p : Tile → ℕ
| A => 5
| B => 2
| C => 4
| D => 10
| E => 11

def q : Tile → ℕ
| A => 2
| B => 1
| C => 9
| D => 6
| E => 3

def r : Tile → ℕ
| A => 8
| B => 4
| C => 6
| D => 5
| E => 7

def s : Tile → ℕ
| A => 11
| B => 7
| C => 3
| D => 9
| E => 0

-- Statement of the problem
theorem top_leftmost_tile_is_tile_B : ∀ (t : Tile), (p t) = 2 → t = B :=
begin
  -- This is where the proof would go. Sorry is used to skip the proof part.
  sorry
end

end top_leftmost_tile_is_tile_B_l470_470925


namespace measure_angle_A_measure_angle_B_l470_470700

-- Definitions and conditions from the original problem
variables (A B C : ℝ) (a b c : ℝ)

-- First part: Prove measure of angle A
theorem measure_angle_A (h1 : b^2 + c^2 - a^2 = bc) (h2 : 0 < A ∧ A < π) :
  A = π / 3 :=
by 
  sorry

-- Second part: Prove measure of angle B
theorem measure_angle_B (h3 : sin A ^ 2 + sin B ^ 2 = sin C ^ 2) (h4 : A = π / 3) :
  B = π / 6 :=
by 
  sorry

end measure_angle_A_measure_angle_B_l470_470700


namespace imaginary_part_of_z_l470_470616

theorem imaginary_part_of_z (z : ℂ) (h : (z * (1 + complex.I) * complex.I^3) / (1 - complex.I) = 1 - complex.I) :
  z.im = -1 := sorry

end imaginary_part_of_z_l470_470616


namespace permutation_cardinality_relation_l470_470152

open Set

noncomputable def A (n : ℕ) (f : Fin n → Fin n) : Set (Fin n) := { i | i > f i }

noncomputable def B (n : ℕ) (f : Fin n → Fin n) : Set (Fin n × Fin n) := 
  { (i, j) | i < j ∧ (j ≤ f j ∧ f j < f i) ∨ (f j < f i ∧ i < j) }

noncomputable def C (n : ℕ) (f : Fin n → Fin n) : Set (Fin n × Fin n) := 
  { (i, j) | i < j ∧ (j ≤ f i ∧ f i < f j) ∨ (f i < f j ∧ i < j) }

noncomputable def D (n : ℕ) (f : Fin n → Fin n) : Set (Fin n × Fin n) := 
  { (i, j) | i < j ∧ f i > f j }

theorem permutation_cardinality_relation (n : ℕ) (f : Fin n → Fin n) (h : Function.Bijective f) :
  |A n f| + 2 * |B n f| + |C n f| = |D n f| := by
  sorry

end permutation_cardinality_relation_l470_470152


namespace division_of_neg6_by_3_l470_470553

theorem division_of_neg6_by_3 : (-6 : ℤ) / 3 = -2 := 
by
  sorry

end division_of_neg6_by_3_l470_470553


namespace count_distinct_special_sums_l470_470559

noncomputable def special_fractions : Finset ℚ :=
Finset.filter (λ q, ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 18 ∧ q = a / b) (Finset.image (λ p : ℕ × ℕ, p.1 / p.2) { p : ℕ × ℕ | p.1 + p.2 = 18 ∧ 0 < p.1 ∧ 0 < p.2}.to_finset)

noncomputable def special_sums : Finset ℤ :=
Finset.image (λ p : ℚ × ℚ, (p.1 + p.2).num) (special_fractions.product special_fractions)

theorem count_distinct_special_sums : special_sums.card = 15 :=
sorry

end count_distinct_special_sums_l470_470559


namespace percentage_increase_of_x_over_y_is_20_percent_l470_470245

-- Define the conditions
variables {x y z: ℝ}

-- Given
def condition1 : y = 0.4 * z := sorry
def condition2 : x = 0.48 * z := sorry

-- To prove
theorem percentage_increase_of_x_over_y_is_20_percent
  (h1 : condition1)
  (h2 : condition2) :
  (x - y) / y * 100 = 20 :=
by sorry

end percentage_increase_of_x_over_y_is_20_percent_l470_470245


namespace time_to_cross_platform_l470_470067

-- Define the conditions
def speed_kmph : ℝ := 72
def length_train : ℝ := 270.0416
def length_platform : ℝ := 250
def conversion_factor : ℝ := 1000 / 3600
def speed_mps := speed_kmph * conversion_factor
def total_distance := length_train + length_platform

-- Statement to be proved
theorem time_to_cross_platform : 
  total_distance / speed_mps = 26.00208 := 
begin
  sorry
end

end time_to_cross_platform_l470_470067


namespace volume_of_tetrahedron_abcd_l470_470529

theorem volume_of_tetrahedron_abcd :
  ∀ (A B C D : ℝ³), 
  dist A B = 6 → dist A C = 4 → dist A D = 5 → 
  dist B C = 5 → dist B D = 3 → dist C D = 7 → 
  ∀ (orthogonal : (B - A) ⬝ (C - A) = 0 ∧ (C - A) ⬝ (D - A) = 0 ∧ (D - A) ⬝ (B - A) = 0), 
  volume_of_tetrahedron A B C D = 20 :=
by 
  intros A B C D hAB hAC hAD hBC hBD hCD orthogonal
  sorry

end volume_of_tetrahedron_abcd_l470_470529


namespace cube_painted_faces_l470_470066

noncomputable def painted_faces_count (side_length painted_cubes_edge middle_cubes_edge : ℕ) : ℕ :=
  let total_corners := 8
  let total_edges := 12
  total_corners + total_edges * middle_cubes_edge

theorem cube_painted_faces :
  ∀ side_length : ℕ, side_length = 4 →
  ∀ painted_cubes_edge middle_cubes_edge total_cubes : ℕ,
  total_cubes = side_length * side_length * side_length →
  painted_cubes_edge = 3 →
  middle_cubes_edge = 2 →
  painted_faces_count side_length painted_cubes_edge middle_cubes_edge = 32 := sorry

end cube_painted_faces_l470_470066


namespace coins_distribution_l470_470013

theorem coins_distribution :
  ∃ (x y z : ℕ), x + y + z = 1000 ∧ x + 2 * y + 5 * z = 2000 ∧ Nat.Prime x ∧ x = 3 ∧ y = 996 ∧ z = 1 :=
by
  sorry

end coins_distribution_l470_470013


namespace isogonal_conjugates_concurrent_l470_470163

variables {A B C : Type} [point_space A B C]
variables {α β γ a b c : ℝ}

def isogonal_conjugate_barycentric (α β γ a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a^2 / α, b^2 / β, c^2 / γ)

theorem isogonal_conjugates_concurrent {α β γ : ℝ}
  (h_concurrent : are_concurrent A B C)
  : intersection_point A B C = isogonal_conjugate_barycentric α β γ a b c :=
sorry

end isogonal_conjugates_concurrent_l470_470163


namespace inequality_holds_for_all_reals_l470_470342

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470342


namespace tiles_per_row_proof_l470_470396

def area_square_room : ℝ := 400 -- in square feet
def tile_size : ℝ := 8 / 12 -- each tile is 8 inches, converted to feet (8/12 feet)

noncomputable def number_of_tiles_per_row : ℕ :=
  let side_length := Math.sqrt area_square_room in
  let side_length_in_inch := side_length * 12 in
  Nat.floor (side_length_in_inch / tile_size)

theorem tiles_per_row_proof :
  number_of_tiles_per_row = 30 := by
  sorry

end tiles_per_row_proof_l470_470396


namespace temperature_difference_l470_470308

theorem temperature_difference (high low : ℝ) (h_high : high = 5) (h_low : low = -3) :
  high - low = 8 :=
by {
  -- Proof goes here
  sorry
}

end temperature_difference_l470_470308


namespace number_of_tiles_per_row_in_square_room_l470_470401

theorem number_of_tiles_per_row_in_square_room 
  (area_of_floor: ℝ)
  (tile_side_length: ℝ)
  (conversion_rate: ℝ)
  (h1: area_of_floor = 400) 
  (h2: tile_side_length = 8) 
  (h3: conversion_rate = 12) 
: (sqrt area_of_floor * conversion_rate / tile_side_length) = 30 := 
by
  sorry

end number_of_tiles_per_row_in_square_room_l470_470401


namespace min_distance_l470_470969

theorem min_distance (x y : ℝ) (h : x + y = 3) : ∃ d : ℝ, d = sqrt ((x - 2)^2 + (y + 1)^2) ∧ d = sqrt 2 :=
by
  sorry

end min_distance_l470_470969


namespace correct_statement_about_Digital_Earth_l470_470827

-- Definitions of the statements
def statement_A : Prop :=
  "Digital Earth is a reflection of the real Earth through digital means" = "Correct statement about Digital Earth"

def statement_B : Prop :=
  "Digital Earth is an extension of GIS technology" = "Correct statement about Digital Earth"

def statement_C : Prop :=
  "Digital Earth can only achieve global information sharing through the internet" = "Correct statement about Digital Earth"

def statement_D : Prop :=
  "The core idea of Digital Earth is to use digital means to uniformly address Earth's issues" = "Correct statement about Digital Earth"

-- Theorem that needs to be proved 
theorem correct_statement_about_Digital_Earth : statement_C :=
by 
  sorry

end correct_statement_about_Digital_Earth_l470_470827


namespace directrix_of_parabola_l470_470932

theorem directrix_of_parabola (y x : ℝ) : 
  (∃ a h k : ℝ, y = a * (x - h)^2 + k ∧ a = 1/8 ∧ h = 4 ∧ k = 0) → 
  y = -1/2 :=
by
  intro h
  sorry

end directrix_of_parabola_l470_470932


namespace total_profit_eq_9600_l470_470484

variable (P : ℝ) (A_capital B_capital A_received : ℝ)

axiom h_A_capital : A_capital = 3500
axiom h_B_capital : B_capital = 1500
axiom h_A_received : A_received = 7008

theorem total_profit_eq_9600 :
  let total_capital := A_capital + B_capital,
      A_share := A_capital / total_capital,
      B_share := B_capital / total_capital,
      remaining_profit := 0.90 * P,
      A_management_fee := 0.10 * P,
      A_total := A_management_fee + A_share * remaining_profit in
  A_total = A_received → P = 9600 :=
by
  intros h;
  sorry

end total_profit_eq_9600_l470_470484


namespace parallelogram_with_angle_condition_is_rectangle_l470_470087

-- Assume ABCD is a parallelogram
variables (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

def is_parallelogram (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] := sorry

noncomputable def is_rectangle (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] := sorry

theorem parallelogram_with_angle_condition_is_rectangle (A B C D : Type) 
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
  (h_parallelogram: is_parallelogram A B C D)
  (h_angle_condition : ∠B + ∠D = 180) : is_rectangle A B C D := 
sorry

end parallelogram_with_angle_condition_is_rectangle_l470_470087


namespace find_ellipse_l470_470647

-- Definitions based on the conditions
def ellipse (a b : ℝ) : Set (ℝ × ℝ) := { p | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1 }

def is_focus (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop := 
  sqrt (F1.1 ^ 2 + F1.2 ^ 2) + sqrt (F2.1 ^ 2 + F2.2 ^ 2) = 2 * a

def perpendicular_through_focus (F : ℝ × ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  F.1 = p1.1 ∧ F.1 = p2.1

def equilateral_triangle (A B F : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B F = side_length ∧ dist F A = side_length

-- Final theorem statement
theorem find_ellipse (a b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h1 : is_focus F1 F2 (sqrt 3))
  (h2 : perpendicular_through_focus F2 A B)
  (h3 : equilateral_triangle A B F1 ((4 * sqrt 3) / 3))
  (h4 : dist F1 F2 = 2) :
  a = sqrt 3 ∧ b = sqrt 2 ∧ (∀ p, p ∈ ellipse a b ↔ p.1 ^ 2 / 3 + p.2 ^ 2 / 2 = 1) :=
sorry

end find_ellipse_l470_470647


namespace cell_diameter_scientific_notation_l470_470086

theorem cell_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * 10 ^ n ∧ a = 2.5 ∧ n = -6 :=
by
  use 2.5, -6
  split
  · norm_num
  · split; norm_num

end cell_diameter_scientific_notation_l470_470086


namespace monochromatic_triangle_probability_l470_470125

theorem monochromatic_triangle_probability :
  let hexagon_edges := 6 + 9 in
  let edge_colorings := 3^hexagon_edges in
  let triangles_in_hexagon := 20 in
  let non_monochromatic_prob := (2/9) in
  let probability := 1 - (non_monochromatic_prob^triangles_in_hexagon) in
  probability = 1 := by 
  sorry

end monochromatic_triangle_probability_l470_470125


namespace coordinates_of_B_l470_470255

-- Define the initial coordinates of point A
def A : ℝ × ℝ := (1, -2)

-- Define the transformation to get point B from A
def B : ℝ × ℝ := (A.1 - 2, A.2 + 3)

theorem coordinates_of_B : B = (-1, 1) :=
by
  sorry

end coordinates_of_B_l470_470255


namespace find_equation_of_parallel_line_l470_470137

-- defining the point P (1, 2)
def P : ℝ × ℝ := (1, 2)

-- defining the line that passes through the point P and is parallel to the given line
def parallel_line_through_P (c : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x - y + c = 0

-- defining the given line equation
def given_line : ℝ → ℝ → Prop :=
  λ x y, x - y + 2 = 0

-- statement of the theorem
theorem find_equation_of_parallel_line : 
  (∃ c, parallel_line_through_P c P.1 P.2 ∧ (∀ x y, given_line x y → (parallel_line_through_P c x y))) →
  parallel_line_through_P 1 :=
sorry

end find_equation_of_parallel_line_l470_470137


namespace marie_erasers_l470_470301

-- Define the initial conditions
def initial_erasers : ℝ := 95.0
def additional_erasers : ℝ := 42.0

-- Define the target final erasers count
def final_erasers : ℝ := 137.0

-- The theorem we need to prove
theorem marie_erasers :
  initial_erasers + additional_erasers = final_erasers := by
  sorry

end marie_erasers_l470_470301


namespace area_of_PQRS_l470_470312

noncomputable def length_EF := 6
noncomputable def width_EF := 4

noncomputable def area_PQRS := (length_EF + 6 * Real.sqrt 3) * (width_EF + 4 * Real.sqrt 3)

theorem area_of_PQRS :
  area_PQRS = 60 + 48 * Real.sqrt 3 := by
  sorry

end area_of_PQRS_l470_470312


namespace find_f_l470_470586

def f (m : ℝ) : ℝ := (-1 + (sqrt (4 * m - 3))) / 2

theorem find_f (t : ℝ) (ht : t ≥ 0) : f (t^2 + t + 1) = t :=
by
  -- omit proof
  sorry

end find_f_l470_470586


namespace no_real_x_satisfies_log_eq_l470_470235

noncomputable def log_equation (x : ℝ) : Prop :=
  log (x + 5) + log (2 * x - 2) = log (2 * x ^ 2 + x - 10)

theorem no_real_x_satisfies_log_eq :
  ¬ ∃ (x : ℝ), log_equation x := 
sorry

end no_real_x_satisfies_log_eq_l470_470235


namespace intersection_area_example_l470_470268

-- Definitions and Hypotheses
variable (Square : Type) [MeasureSpace Square]
variable {polygon : Square → Set Set}
variable (side_length : ℝ) (polygons : Fin 7 → Set Square)
variable (area : MeasureTheory.Measure Square)
variable (h1 : side_length = 2)
variable (h2 : ∀ i, area (polygons i) ≥ 1)

-- Theorem statement
theorem intersection_area_example : 
  ∃ i j : Fin 7, i ≠ j ∧ area (polygons i ∩ polygons j) ≥ 1 / 7 :=
sorry

end intersection_area_example_l470_470268


namespace trains_cross_time_l470_470811

def time_to_cross_trains (length_train : ℝ) (speed_faster : ℝ) (speed_ratio : ℝ) : ℝ :=
  let speed_slower := speed_faster / speed_ratio
  let relative_speed := speed_faster + speed_slower
  let total_distance := 2 * length_train
  total_distance / relative_speed

theorem trains_cross_time :
  ∀ (length_train speed_faster speed_ratio : ℝ),
    length_train = 100 → 
    speed_faster = 40 → 
    speed_ratio = 2 →
    time_to_cross_trains length_train speed_faster speed_ratio = 10 / 3 :=
by
  intros length_train speed_faster speed_ratio h1 h2 h3
  unfold time_to_cross_trains
  rw [h1, h2, h3]
  norm_num
  sorry

end trains_cross_time_l470_470811


namespace inequality_proof_l470_470337

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470337


namespace two_people_work_time_l470_470275

theorem two_people_work_time (time_one_person : ℕ)
    (num_people : ℕ)
    (h1 : time_one_person = 10)
    (h2 : num_people = 2)
    (equally_skilled : true) : 
    let time_two_people := time_one_person / num_people in
    time_two_people = 5 := by 
    -- According to the given conditions
    sorry

end two_people_work_time_l470_470275


namespace inequality_holds_for_all_real_numbers_l470_470371

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470371


namespace each_group_has_145_bananas_l470_470799

theorem each_group_has_145_bananas (total_bananas : ℕ) (groups_bananas : ℕ) : 
  total_bananas = 290 ∧ groups_bananas = 2 → total_bananas / groups_bananas = 145 := 
by 
  sorry

end each_group_has_145_bananas_l470_470799


namespace company_max_revenue_l470_470506

structure Conditions where
  max_total_time : ℕ -- maximum total time in minutes
  max_total_cost : ℕ -- maximum total cost in yuan
  rate_A : ℕ -- rate per minute for TV A in yuan
  rate_B : ℕ -- rate per minute for TV B in yuan
  revenue_A : ℕ -- revenue per minute for TV A in million yuan
  revenue_B : ℕ -- revenue per minute for TV B in million yuan

def company_conditions : Conditions :=
  { max_total_time := 300,
    max_total_cost := 90000,
    rate_A := 500,
    rate_B := 200,
    revenue_A := 3, -- as 0.3 million yuan converted to 3 tenths (integer representation)
    revenue_B := 2  -- as 0.2 million yuan converted to 2 tenths (integer representation)
  }

def advertising_strategy
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : Prop :=
  time_A + time_B ≤ conditions.max_total_time ∧
  time_A * conditions.rate_A + time_B * conditions.rate_B ≤ conditions.max_total_cost

def revenue
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : ℕ :=
  time_A * conditions.revenue_A + time_B * conditions.revenue_B

theorem company_max_revenue (time_A time_B : ℕ)
  (h : advertising_strategy company_conditions time_A time_B) :
  revenue company_conditions time_A time_B = 70 := 
  by
  have h1 : time_A = 100 := sorry
  have h2 : time_B = 200 := sorry
  sorry

end company_max_revenue_l470_470506


namespace find_b_l470_470781

open_locale real

-- Define the two points
def P1 := (-4 : ℝ, 6 : ℝ)
def P2 := (3 : ℝ, -3 : ℝ)

-- Define the direction vector based on the given points
def direction_vector :=
  (P2.1 - P1.1, P2.2 - P1.2)

-- Define b as the x-coordinate of the direction vector scaled so that y is 1
def b := -7 / 9

-- Define the target direction vector with the b component
def target_direction_vector :=
  (b, 1 : ℝ)

-- State the problem as a theorem
theorem find_b : direction_vector = (7, -9) ∧ target_direction_vector = (-7 / 9, 1) :=
by
  sorry

end find_b_l470_470781


namespace tangent_line_eq_range_of_m_l470_470187

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

theorem tangent_line_eq :
  let y := 7 in
  let slope := 12 in
  ∀ x: ℝ, 12 * x - y - 17 = 0 :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ (-3 < m) ∧ (m < -2) :=
by sorry

end tangent_line_eq_range_of_m_l470_470187


namespace number_of_As_correct_l470_470437

theorem number_of_As_correct : 
  ∃ A : ℕ, 
  let normal_recess := 20 in
  let total_recess := 47 in
  let minutes_per_A := 2 in
  let minutes_per_B := 1 in
  let minutes_per_C := 0 in
  let minutes_per_D := 1 in
  let Bs := 12 in
  let Cs := 14 in
  let Ds := 5 in
  (total_recess - normal_recess = A * minutes_per_A + Bs * minutes_per_B - Ds * minutes_per_D) ∧ A = 10 :=
sorry

end number_of_As_correct_l470_470437


namespace measure_angle_ACB_l470_470286

theorem measure_angle_ACB (A B C D E F : Point) (H : Point) (K : ℝ) 
(hAD : is_altitude A D B C) 
(hBE : is_altitude B E A C) 
(hCF : is_altitude C F A B) 
(h_eq : 5 • (vector A D) + 3 • (vector B E) + 8 • (vector C F) = 0) : 
  angle A C B = 120 :=
sorry

end measure_angle_ACB_l470_470286


namespace inequality_proof_l470_470355

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470355


namespace primes_solution_l470_470930

theorem primes_solution (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m ≥ 2) (hn : n ≥ 2) :
    p^n = q^m + 1 ∨ p^n = q^m - 1 → (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by
  sorry

end primes_solution_l470_470930


namespace determinant_scaled_matrix_l470_470955

theorem determinant_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 7) :
  |3 * a 3 * b|
  |3 * c 3 * d| = 63 := 
begin
  sorry
end

end determinant_scaled_matrix_l470_470955


namespace find_values_l470_470640

theorem find_values (a b : ℝ) 
  (h1 : Real.cbrt (7 * a + 1) = 1 / 2)
  (h2 : Real.sqrt (8 * a + b - 2) = 2 ∨ Real.sqrt (8 * a + b - 2) = -2) :
  a = -1/8 ∧ b = 7 ∧ (Real.sqrt (-8 * a + 3 * b + 3) = 5 ∨ Real.sqrt (-8 * a + 3 * b + 3) = -5) :=
by
  sorry

end find_values_l470_470640


namespace find_angle_between_vectors_l470_470971

noncomputable def angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : ℝ :=
  60

theorem find_angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : angle_between_vectors a b a_nonzero b_nonzero perp1 perp2 = 60 :=
  by 
  sorry

end find_angle_between_vectors_l470_470971


namespace correct_statements_l470_470028

theorem correct_statements : 
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end correct_statements_l470_470028


namespace security_deposit_correct_l470_470276

-- Definitions (Conditions)
def daily_rate : ℝ := 125
def pet_fee_per_dog : ℝ := 100
def number_of_dogs : ℕ := 2
def tourism_tax_rate : ℝ := 0.10
def service_fee_rate : ℝ := 0.20
def activity_cost_per_person : ℝ := 45
def number_of_activities_per_person : ℕ := 3
def number_of_people : ℕ := 2
def security_deposit_rate : ℝ := 0.50
def usd_to_euro_conversion_rate : ℝ := 0.83

-- Function to calculate total cost
def total_cost_in_euros : ℝ :=
  let rental_cost := daily_rate * 14
  let pet_cost := pet_fee_per_dog * number_of_dogs
  let tourism_tax := tourism_tax_rate * rental_cost
  let service_fee := service_fee_rate * rental_cost
  let cabin_total := rental_cost + pet_cost + tourism_tax + service_fee
  let activities_total := number_of_activities_per_person * activity_cost_per_person * number_of_people
  let total_cost := cabin_total + activities_total
  let security_deposit_usd := security_deposit_rate * total_cost
  security_deposit_usd * usd_to_euro_conversion_rate

-- Theorem to prove
theorem security_deposit_correct :
  total_cost_in_euros = 1139.18 := 
sorry

end security_deposit_correct_l470_470276


namespace extreme_value_when_m_one_monotonic_intervals_range_of_t_l470_470651

noncomputable def f (x m : ℝ) : ℝ := x^2 + (2 * m - 1) * x - m * log x

theorem extreme_value_when_m_one :
  ∃ x, 0 < x ∧ f x 1 = (3 / 4) - log 2 :=
sorry

theorem monotonic_intervals (m : ℝ) :
  (0 ≤ m ∧ (∀ x, 0 < x ∧ x < (1 / 2) → deriv (f x m) < 0) ∧ (∀ x, x > (1 / 2) → deriv (f x m) > 0)) ∨
  (- (1 / 2) < m ∧ m < 0 ∧ (∀ x, 0 < x ∧ x < -m → deriv (f x m) > 0) ∧ (∀ x, -m < x ∧ x < (1 / 2) → deriv (f x m) < 0) ∧ 
      (∀ x, x > (1 / 2) → deriv (f x m) > 0)) ∨
  (m = - (1 / 2) ∧ (∀ x, 0 < x → deriv (f x m) = 2 * (x - (1 / 2)) ^ 2 / x ∧ deriv (f x m) ≥ 0)) ∨
  (m < - (1 / 2) ∧ (∀ x, 0 < x ∧ x < (1 / 2) → deriv (f x m) > 0 ∧ ∀ x, (1 / 2) < x ∧ x < -m → deriv (f x m) < 0) ∧ (∀ x, x > -m → deriv (f x m) > 0)) :=
sorry

theorem range_of_t (t : ℝ) :
  (∀ m, 2 < m ∧ m < 3 → ∀ x, 1 ≤ x ∧ x ≤ 3 → m * t - f x m < 1) ↔ t ≤ 7 / 3 :=
sorry

end extreme_value_when_m_one_monotonic_intervals_range_of_t_l470_470651


namespace tiles_per_row_proof_l470_470399

def area_square_room : ℝ := 400 -- in square feet
def tile_size : ℝ := 8 / 12 -- each tile is 8 inches, converted to feet (8/12 feet)

noncomputable def number_of_tiles_per_row : ℕ :=
  let side_length := Math.sqrt area_square_room in
  let side_length_in_inch := side_length * 12 in
  Nat.floor (side_length_in_inch / tile_size)

theorem tiles_per_row_proof :
  number_of_tiles_per_row = 30 := by
  sorry

end tiles_per_row_proof_l470_470399


namespace union_of_sets_l470_470678

noncomputable def M (a : ℝ) : Set ℝ := {3, Real.logBase 2 a}
def N (a b : ℝ) : Set ℝ := {a, b}
def cond (a b : ℝ) : Prop := (M a) ∩ (N a b) = {0}

theorem union_of_sets (a b : ℝ) (h : cond a b) : (M a) ∪ (N a b) = {0, 1, 3} :=
by
  sorry

end union_of_sets_l470_470678


namespace find_x_l470_470452

theorem find_x 
  (AB AC BC : ℝ) 
  (x : ℝ)
  (hO : π * (AB / 2)^2 = 12 + 2 * x)
  (hP : π * (AC / 2)^2 = 24 + x)
  (hQ : π * (BC / 2)^2 = 108 - x)
  : AC^2 + BC^2 = AB^2 → x = 60 :=
by {
   sorry
}

end find_x_l470_470452


namespace geometric_progression_solution_l470_470121

theorem geometric_progression_solution (x : ℝ) :
  (2 * x + 10) ^ 2 = x * (5 * x + 10) → x = 15 + 5 * Real.sqrt 5 :=
by
  intro h
  sorry

end geometric_progression_solution_l470_470121


namespace cyclic_quadrilateral_tangent_sum_l470_470862

theorem cyclic_quadrilateral_tangent_sum (A B C D : Point) (O : Point) (r : ℝ)
    (h_circ : IsCyclic {A, B, C, D})
    (h_center : O ∈ Segment A B)
    (hr : ∀ (P ∈ {A, D, B, C}), Dist P O = r)
    (h_tangent_AD : ∀ (P ∈ Segment A D), Dist P O = r)
    (h_tangent_CD : ∀ (P ∈ Segment C D), Dist P O = r)
    (h_tangent_BC : ∀ (P ∈ Segment B C), Dist P O = r) :
  (length (Segment A D) + length (Segment B C)) = length (Segment A B) :=
begin
  sorry
end

end cyclic_quadrilateral_tangent_sum_l470_470862


namespace unique_sequence_length_l470_470533

theorem unique_sequence_length :
  ∃ (b : ℕ → ℕ) (m : ℕ), (strict_mono b) ∧ 
  (∀ i,  i < m → b i ∈ {0, 1, ..., b m}) ∧
  (∀ i j, i < j → b i < b j) ∧
  (∑ i in finset.range m, 2 ^ (b i) = (2 ^ 49 + 1) / (2 ^ 7 + 1)) ∧ 
  m = 8 :=
sorry

end unique_sequence_length_l470_470533


namespace anastasia_pairs_sum_l470_470725

theorem anastasia_pairs_sum (m n : ℕ) (h_pos : m > 1) : 
  ∃ (pairs : list (ℕ × ℕ)), 
  (∀ x ∈ pairs, (x.fst + x.snd = 2m)) ∧
  (∀ (selection : list ℕ), (∀ x ∈ selection, ∃ y ∈ pairs, (y.fst = x ∨ y.snd = x)) → 
  (list.sum selection ≠ n)) :=
sorry

end anastasia_pairs_sum_l470_470725


namespace problem1_problem2_l470_470492

variables (a b : ℝ^3)
variables (h₁ : ∥a∥ = 3) (h₂ : ∥b∥ = 3)
variables (h₃ : real.angle a b = real.pi / 3) (h₄ : disjoint a b)

theorem problem1 :
  ∥a + b∥ = 3 ∧ ∥2 * a - b∥ = 3 * real.sqrt 7 := 
sorry

variables (nonzero_a : a ≠ 0) (nonzero_b : b ≠ 0)
variables (perp1 : ⟪a + 3 * b, 7 * a - 5 * b⟫ = 0)
variables (perp2 : ⟪a - 4 * b, 7 * a - 2 * b⟫ = 0)

theorem problem2 : real.angle a b = real.pi / 3 :=
sorry

end problem1_problem2_l470_470492


namespace tank_empties_in_4320_minutes_l470_470489

-- Define the initial conditions
def tankVolumeCubicFeet: ℝ := 30
def inletPipeRateCubicInchesPerMin: ℝ := 5
def outletPipe1RateCubicInchesPerMin: ℝ := 9
def outletPipe2RateCubicInchesPerMin: ℝ := 8
def feetToInches: ℝ := 12

-- Conversion from cubic feet to cubic inches
def tankVolumeCubicInches: ℝ := tankVolumeCubicFeet * feetToInches^3

-- Net rate of emptying in cubic inches per minute
def netRateOfEmptying: ℝ := (outletPipe1RateCubicInchesPerMin + outletPipe2RateCubicInchesPerMin) - inletPipeRateCubicInchesPerMin

-- Time to empty the tank
noncomputable def timeToEmptyTank: ℝ := tankVolumeCubicInches / netRateOfEmptying

-- The theorem to prove
theorem tank_empties_in_4320_minutes :
  timeToEmptyTank = 4320 := by
  sorry

end tank_empties_in_4320_minutes_l470_470489


namespace age_sum_l470_470486

variable (b : ℕ)
variable (a : ℕ := b + 2)
variable (c : ℕ := b / 2)

theorem age_sum : b = 10 → a + b + c = 27 :=
by
  intros h
  rw [h]
  sorry

end age_sum_l470_470486


namespace incorrect_proposition_B_l470_470733

variable {m n : Line} {α β : Plane}

/-- The proposition B states that:
If m is parallel to α, n is perpendicular to β, and α is perpendicular to β, then m is parallel to n,
which we will prove is incorrect given these conditions. -/
theorem incorrect_proposition_B
  (h1 : m ∥ α)
  (h2 : n ⊥ β)
  (h3 : α ⊥ β) :
  ¬ (m ∥ n) := sorry

end incorrect_proposition_B_l470_470733


namespace corrected_mean_l470_470785

theorem corrected_mean (mean_initial : ℝ) (num_obs : ℕ) (obs_incorrect : ℝ) (obs_correct : ℝ) :
  mean_initial = 36 → num_obs = 50 → obs_incorrect = 23 → obs_correct = 30 →
  (mean_initial * ↑num_obs + (obs_correct - obs_incorrect)) / ↑num_obs = 36.14 :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_l470_470785


namespace infimum_expression_l470_470131

open Real

/-- Given \( n \ge 1 \), \( \forall j \in \{1, 2, \ldots, n\}, a_j > 0 \), and 
\( \sum_{j=1}^n a_j < \pi \), this proof problem states that the 
infimum of the given expression is -\pi. -/
theorem infimum_expression (n : ℕ) (a : Fin n → ℝ) (h1 : 1 ≤ n) 
  (h2 : ∀ j, 0 < a j) (h3 : (∑ j in Finset.range n, a j) < π) :
  infi (λ (a : Fin n → ℝ), ∑ j in Finset.range n, (a j) * cos (∑ k in Finset.range (j + 1), a k)) = -π :=
sorry

end infimum_expression_l470_470131


namespace angle_division_quadrant_l470_470176

variable (k : ℤ)
variable (α : ℝ)
variable (h : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi)

theorem angle_division_quadrant 
  (hα_sec_quadrant : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi) : 
  (∃ m : ℤ, (m = 0 ∧ Real.pi / 4 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
             (m = 1 ∧ Real.pi * (2 * m + 1) / 4 < α / 2 ∧ α / 2 < Real.pi * (2 * m + 1) / 2)) :=
sorry

end angle_division_quadrant_l470_470176


namespace minimum_selection_integers_l470_470999

theorem minimum_selection_integers (s : Finset ℕ) (h : s ⊆ Finset.range 21 ∧ s.card = 11) :
  ∃ a b ∈ s, a - b = 2 ∨ b - a = 2 :=
by
  sorry

end minimum_selection_integers_l470_470999


namespace max_distinct_sum_squares_3000_l470_470818

   theorem max_distinct_sum_squares_3000 : 
     ∃ (n : ℕ) (k : Fin n → ℕ), (∀ i j, i ≠ j → k i ≠ k j) ∧ (∑ i, (k i)^2 = 3000) ∧ n = 20 :=
   sorry
   
end max_distinct_sum_squares_3000_l470_470818


namespace find_constants_and_calculate_result_l470_470025

theorem find_constants_and_calculate_result :
  ∃ (a b : ℤ), 
    (∀ (x : ℤ), (x + a) * (x + 6) = x^2 + 8 * x + 12) ∧ 
    (∀ (x : ℤ), (x - a) * (x + b) = x^2 + x - 6) ∧ 
    (∀ (x : ℤ), (x + a) * (x + b) = x^2 + 5 * x + 6) :=
by
  sorry

end find_constants_and_calculate_result_l470_470025


namespace simplify_expression_l470_470054

theorem simplify_expression : 
    2 * Real.sqrt 12 + 3 * Real.sqrt (4 / 3) - Real.sqrt (16 / 3) - (2 / 3) * Real.sqrt 48 = 2 * Real.sqrt 3 :=
by
  sorry

end simplify_expression_l470_470054


namespace center_is_path_endpoint_of_odd_grid_l470_470295

noncomputable def is_path_endpoint {n : ℕ} (grid : Fin n → Fin n → ℕ) (color : ℕ) (path : List (Fin n × Fin n)) : Prop :=
  path.head = (n / 2, n / 2) ∨ path.last = (n / 2, n / 2)

theorem center_is_path_endpoint_of_odd_grid {n : ℕ} (hn : n % 2 = 1) (colored_grid : Fin n → Fin n → ℕ) (is_path : ∀ color ∈ {0, 1}, 
List (Fin n × Fin n)) :
  ∃ color ∈ {0, 1}, is_path_endpoint colored_grid color (is_path color (by sorry)) :=
sorry

end center_is_path_endpoint_of_odd_grid_l470_470295


namespace fibonacci_divisibility_sequence_l470_470764

-- Define the Fibonacci sequence
def fib: ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- Statement that the Fibonacci sequence is a divisibility sequence
theorem fibonacci_divisibility_sequence (m n : ℕ) (h : m ∣ n) : fib m ∣ fib n := sorry

end fibonacci_divisibility_sequence_l470_470764


namespace tiles_per_row_l470_470387

noncomputable def area_square_room : ℕ := 400
noncomputable def side_length_feet : ℕ := nat.sqrt area_square_room
noncomputable def side_length_inches : ℕ := side_length_feet * 12
noncomputable def tile_size : ℕ := 8

theorem tiles_per_row : (side_length_inches / tile_size) = 30 := 
by
  have h1: side_length_feet = 20 := by sorry
  have h2: side_length_inches = 240 := by sorry
  have h3: side_length_inches / tile_size = 30 := by sorry
  exact h3

end tiles_per_row_l470_470387


namespace sum_of_two_digit_integers_whose_squares_end_in_49_l470_470823

/-- The sum of all two-digit positive integers whose squares end with the digits 49 is 130. -/
theorem sum_of_two_digit_integers_whose_squares_end_in_49 :
  let valid_numbers := {n : ℕ // n ≥ 10 ∧ n < 100 ∧ (n^2 % 100) = 49} in
  ∑ n in valid_numbers, n = 130 :=
by
  sorry

end sum_of_two_digit_integers_whose_squares_end_in_49_l470_470823


namespace Eliana_spent_63_dollars_l470_470576

-- Define the conditions
def cost_per_refill : ℕ := 21
def number_of_refills : ℕ := 3

-- Define the total amount Eliana spent on fuel
def total_spent : ℕ := number_of_refills * cost_per_refill

-- State the theorem to be proven
theorem Eliana_spent_63_dollars : total_spent = 63 := 
by simp [total_spent, number_of_refills, cost_per_refill]; sorry

end Eliana_spent_63_dollars_l470_470576


namespace average_speed_l470_470722

theorem average_speed (d1 d2 s1 s2 : ℝ) (h1 : d1 = 360) (h2 : s1 = 60) (h3 : d2 = 120) (h4 : s2 = 40) :
  (d1 + d2) / ((d1 / s1) + (d2 / s2)) ≈ 53.33 :=
by
  sorry

end average_speed_l470_470722


namespace percentage_of_flowers_cut_l470_470756

theorem percentage_of_flowers_cut :
  (50 * 400) - 8000 = 20_000 - 8_000 →
  (20_000 - 8_000) / 20_000 * 100 = 60 := by
  sorry

end percentage_of_flowers_cut_l470_470756


namespace g_f_neg3_eq_7_l470_470287

def f (x : ℝ) : ℝ := 2 * x ^ 2 - 5

variable (g : ℝ → ℝ)
variable (h_gf3 : g (f 3) = 7)

theorem g_f_neg3_eq_7 : g (f (-3)) = 7 :=
by {
  have h_f3 : f 3 = 13 := by simp [f],
  have h_f_neg3 : f (-3) = 13 := by simp [f],
  rw [h_f_neg3, h_gf3, h_f3],
  sorry,
}

end g_f_neg3_eq_7_l470_470287


namespace probability_diff_faces_l470_470508

def total_lines : ℕ := 15

def total_pairs : ℕ := Nat.choose total_lines 2

def different_faces_pairs : ℕ := 36

def expected_probability : ℚ := 12 / 35

theorem probability_diff_faces : 
  (different_faces_pairs : ℚ) / total_pairs = expected_probability := by
  sorry

end probability_diff_faces_l470_470508


namespace smallest_four_digit_integer_l470_470821

theorem smallest_four_digit_integer (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : ∀ d ∈ [1, 5, 6], n % d = 0)
  (h3 : ∀ d1 d2, d1 ≠ d2 → d1 ∈ [1, 5, 6] → d2 ∈ [1, 5, 6] → d1 ≠ d2) :
  n = 1560 :=
by
  sorry

end smallest_four_digit_integer_l470_470821


namespace intersection_sum_is_14_l470_470184

def h : ℝ → ℝ := sorry
def j : ℝ → ℝ := sorry

axiom intersect_points :
  h 1 = 1 ∧ h 3 = 4 ∧ h 5 = 9 ∧ h 7 = 12 ∧
  j 1 = 1 ∧ j 3 = 4 ∧ j 5 = 9 ∧ j 7 = 12

theorem intersection_sum_is_14 :
  ∃ (x : ℝ), h(3 * x + 2) = 3 * j(x + 1) ∧ (3 * x + 2) + (3 * j(x + 1)) = 14 :=
by
  sorry

end intersection_sum_is_14_l470_470184


namespace age_ratio_l470_470835

theorem age_ratio 
  (a b c : ℕ)
  (h1 : a = b + 2)
  (h2 : a + b + c = 32)
  (h3 : b = 12) :
  b = 2 * c :=
by
  sorry

end age_ratio_l470_470835


namespace sum_of_possible_x_values_l470_470744

noncomputable
def f : ℝ → ℝ := λ x,
  if x < 3 then 12 * x + 21
  else 3 * x - 21

theorem sum_of_possible_x_values (x₁ x₂ : ℝ) (h₁ : f x₁ = -3) (h₂ : f x₂ = -3) (hx₁ : x₁ < 3) (hx₂ : x₂ ≥ 3) :
  x₁ + x₂ = 4 :=
sorry

end sum_of_possible_x_values_l470_470744


namespace number_of_tiles_per_row_in_square_room_l470_470405

theorem number_of_tiles_per_row_in_square_room 
  (area_of_floor: ℝ)
  (tile_side_length: ℝ)
  (conversion_rate: ℝ)
  (h1: area_of_floor = 400) 
  (h2: tile_side_length = 8) 
  (h3: conversion_rate = 12) 
: (sqrt area_of_floor * conversion_rate / tile_side_length) = 30 := 
by
  sorry

end number_of_tiles_per_row_in_square_room_l470_470405


namespace triangle_inequality_solution_l470_470945

theorem triangle_inequality_solution :
  let S := {x : ℤ | 4 < x ∧ x < 64} in
  S.card = 59 :=
by
  sorry

end triangle_inequality_solution_l470_470945


namespace sum_of_positive_differences_T_l470_470728

def T : Finset ℤ := {3^0, 3^1, 3^2, 3^3, 3^4, 3^5, 3^6}

def sum_of_positive_differences (s : Finset ℤ) : ℤ :=
  ∑ x in s, ∑ y in s, if x > y then x - y else 0

theorem sum_of_positive_differences_T : sum_of_positive_differences T = 5472 := by
  sorry

end sum_of_positive_differences_T_l470_470728


namespace ryan_weekly_commuting_time_l470_470803

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end ryan_weekly_commuting_time_l470_470803


namespace find_lambda_l470_470264

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C D E : V}
variables (λ : ℝ)

-- Conditions
variable (h1 : B - C = 2 • (C - D)) -- Equivalent to \overrightarrow{BC} = 2\overrightarrow{CD}
variable (h2 : E = (A + D) / 2) -- Point E is the midpoint of line segment AD
variable (h3 : E - A = λ • (B - A) + (3/4) • (C - A))

-- Proof statement
theorem find_lambda :
  λ = -1 / 4 :=
sorry 

end find_lambda_l470_470264


namespace maria_scored_33_points_l470_470704

-- Defining constants and parameters
def num_shots := 40
def equal_distribution : ℕ := num_shots / 3 -- each type of shot

-- Given success rates
def success_rate_three_point : ℚ := 0.25
def success_rate_two_point : ℚ := 0.50
def success_rate_free_throw : ℚ := 0.80

-- Defining the points per successful shot
def points_per_successful_three_point_shot : ℕ := 3
def points_per_successful_two_point_shot : ℕ := 2
def points_per_successful_free_throw_shot : ℕ := 1

-- Calculating total points scored
def total_points_scored :=
  (success_rate_three_point * points_per_successful_three_point_shot * equal_distribution) +
  (success_rate_two_point * points_per_successful_two_point_shot * equal_distribution) +
  (success_rate_free_throw * points_per_successful_free_throw_shot * equal_distribution)

theorem maria_scored_33_points :
  total_points_scored = 33 := 
sorry

end maria_scored_33_points_l470_470704


namespace scientific_notation_of_153000_l470_470876

theorem scientific_notation_of_153000 :
  ∃ (a : ℝ) (n : ℤ), 153000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.53 ∧ n = 5 := 
by
  sorry

end scientific_notation_of_153000_l470_470876


namespace circle_standard_form_through_points_and_line_l470_470594

theorem circle_standard_form_through_points_and_line :
  ∃ (D E F : ℝ),
  (∀ (x y : ℝ), (x^2 + y^2 + D*x + E*y + F = 0 ↔ 
    (x = 0 ∧ y = 4) ∨ (x = 4 ∧ y = 6)))
  ∧ (-D / 2 + E - 2 = 0)
  → (∀ (x y : ℝ),
      (x^2 + y^2 + D*x + E*y + F = 0 ↔ (x - 4)^2 + (y - 1)^2 = 25)) :=
begin
  sorry
end

end circle_standard_form_through_points_and_line_l470_470594


namespace june_earnings_l470_470724

def count_per_petals (num_clovers : ℕ) (percentage : ℚ) : ℕ :=
  (percentage * num_clovers).natAbs

def earnings_per_type (num_clovers : ℕ) (value : ℕ) : ℕ :=
  num_clovers * value

theorem june_earnings :
  let n := 500 in
  let p3 := 0.62 in let p2 := 0.25 in let p4 := 0.10 in let p5 := 0.02 in let p6 := 0.01 in
  let v3 := 1 in let v2 := 2 in let v4 := 5 in let v5 := 10 in let v6 := 20 in
  let c3 := count_per_petals n p3 in
  let c2 := count_per_petals n p2 in
  let c4 := count_per_petals n p4 in
  let c5 := count_per_petals n p5 in
  let c6 := count_per_petals n p6 in
  let e3 := earnings_per_type c3 v3 in
  let e2 := earnings_per_type c2 v2 in
  let e4 := earnings_per_type c4 v4 in
  let e5 := earnings_per_type c5 v5 in
  let e6 := earnings_per_type c6 v6 in
  e3 + e2 + e4 + e5 + e6 = 1010 :=
by
  sorry

end june_earnings_l470_470724


namespace minimum_b_l470_470675

noncomputable def a : ℕ → ℕ
| 0       := 0   -- Not in the domain {1, 2, ...}, but needed for technical reasons
| (n + 1) := 2^n + a n

noncomputable def b (n : ℕ) : ℤ := a n - 15 * n + 1

theorem minimum_b : ∃ (n : ℕ), b n = -44 := by
  existsi 4
  sorry

end minimum_b_l470_470675


namespace milk_water_ratio_vessel2_l470_470020

variables (V x y : ℝ)
-- Condition: Volumes are in ratio 3:5.
def volume_ratio (v1 v2 : ℝ) : Prop := v1 / v2 = 3 / 5

-- Condition: Ratio of milk to water in the first vessel is 1:2.
def ratio_milk_water_vessel1 (milk water : ℝ) : Prop := milk / (milk + water) = 1 / 3

-- When combined, the ratio of milk to water is 1:1.
def combined_ratio (milk1 milk2 water1 water2 : ℝ) : Prop := (milk1 + milk2) / (water1 + water2) = 1

theorem milk_water_ratio_vessel2 
    (V : ℝ)
    (milk1 water1 milk2 water2 : ℝ)
    (h_ratio1 : ratio_milk_water_vessel1 milk1 water1)
    (h_volume1 : volume_ratio 3V 5V)
    (h_combined : combined_ratio milk1 milk2 water1 water2) :
    milk2 / water2 = 3 / 2 :=
begin
  sorry 
end

end milk_water_ratio_vessel2_l470_470020


namespace angle_B_pi_div_4_max_dot_product_l470_470246

-- Conditions for Part (I)
variable (A B C a b c : ℝ)
variable (triangle_condition : c^2 = a^2 + b^2 - a * b)
variable (tan_condition : tan A - tan B = (sqrt 3 / 2) * (1 + tan A * tan B))

-- Prove angle B is π/4 given the conditions
theorem angle_B_pi_div_4 (H : triangle_condition) (H2 : tan_condition) :
  B = π / 4 :=
sorry

-- Conditions for Part (II)
variable (vec_m : ℝ × ℝ)
variable (vec_n : ℝ × ℝ)
variable (angle_condition : 0 < A ∧ A < 2 * π / 3)
variable (m_definition : vec_m = (sin A, 1))
variable (n_definition : vec_n = (3, cos (2 * A)))

-- Prove maximum value of dot product is 17 / 8 given the conditions
theorem max_dot_product (H3 : triangle_condition) (H4 : angle_condition)
  (H5 : m_definition) (H6 : n_definition) :
  vec_m.1 * vec_n.1 + vec_m.2 * vec_n.2 ≤ 17 / 8 :=
sorry

end angle_B_pi_div_4_max_dot_product_l470_470246


namespace solve_s_l470_470567

theorem solve_s (s : ℝ) (h_pos : 0 < s) (h_eq : s^3 = 256) : s = 4 :=
sorry

end solve_s_l470_470567


namespace integral_solution_l470_470854

noncomputable def integral_problem : Real :=
  ∫ x in 1..2, Real.exp x + x⁻¹

theorem integral_solution : integral_problem = Real.exp 2 - Real.exp 1 + Real.log 2 :=
by sorry

end integral_solution_l470_470854


namespace no_integer_sets_for_all_k_l470_470575

theorem no_integer_sets_for_all_k 
(real_exist : ∃ (m n : ℕ) (hm : m ≠ n) (a : fin m → ℕ) (b : fin n → ℕ), 
  ∀ k : ℕ, 0 < k ∧ (∃ h : ((a 0)^k + (a 1)^k + ... + (a (m-1))^k) - ((b 0)^k + (b 1)^k + ... + (b (n-1))^k)) ∈ (k : ℤ)) : false :=
begin
    sorry
end

end no_integer_sets_for_all_k_l470_470575


namespace parabola_equation_fixed_point_distance_l470_470672

-- Given problem
def given_point_on_parabola {p y₀ : ℝ} (hp : p > 0) : Prop :=
  y₀^2 = 4 * p

def area_triangle (p y₀ : ℝ) (hp : p > 0) : Prop :=
  (1 / 2) * (p / 2) * |y₀| = 4

-- Part 1: Prove the equation of the parabola
theorem parabola_equation (hp : (4: ℝ) > 0) : 
  (∃ (p : ℝ), y₀^2 = 4 * p ∧ (1 / 2) * (p /2) * |y₀| = 4) →
  y^2 = 8 * x :=
by sorry

-- Part 2: Prove the existence of the fixed point Q and the constant distance |NQ|
theorem fixed_point_distance {A B : ℝ × ℝ} (y₁ y₂ : ℝ) (hA : A ≠ (0, 0)) (hB : B ≠ 0) :
  (angle A B = 90) → 
  (line ON AB) → 
  (∃ Q : ℝ × ℝ, Q = (4, 0) ∧ ∀ N : ℝ × ℝ, N ≠ O ∧ N lies on circle ((x - 4)^2 + y^2 = 16) → 
  dist N Q = 4) :=
by sorry

end parabola_equation_fixed_point_distance_l470_470672


namespace prop_A_prop_C_not_prop_B_not_prop_D_l470_470942

def min_element (S : Finset ℕ) : ℕ :=
  S.min' (by simp [Finset.nonempty])

def distance (A B : Finset ℕ) : ℕ :=
  (Finset.image (λ (ab : ℕ × ℕ), (ab.fst - ab.snd).natAbs) (A.product B)).min' (by simp [Finset.nonempty])

theorem prop_A (A B : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (h : min_element A = min_element B) : distance A B = 0 := sorry

theorem prop_C (A B : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (h : distance A B = 0) : (A ∩ B).nonempty := sorry

theorem not_prop_B (A B : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (h : min_element A > min_element B) : distance A B ≤ 0 := sorry

theorem not_prop_D : ∃ (A B C : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (hC : C.nonempty), distance A B + distance B C < distance A C := sorry

end prop_A_prop_C_not_prop_B_not_prop_D_l470_470942


namespace part_a_part_b_l470_470494

-- Part (a)
theorem part_a (n : ℕ) (h₁ : ¬(sqrt (n : ℝ) ∈ ℚ)) 
  (a b : ℕ) (r : ℝ) (h₂ : 0 < r)
  (hr1 : r^a + sqrt n ∈ ℚ) 
  (hr2 : r^b + sqrt n ∈ ℚ) : 
  a = b := 
sorry

-- Part (b)
theorem part_b (n : ℕ) (h₁ : ¬(sqrt (n : ℝ) ∈ ℚ)) 
  (a b : ℕ) (r : ℝ) 
  (hr1 : r^a + sqrt n ∈ ℚ) 
  (hr2 : r^b + sqrt n ∈ ℚ) : 
  a = b ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) := 
sorry

end part_a_part_b_l470_470494


namespace distance_to_valley_l470_470500

theorem distance_to_valley (car_speed_kph : ℕ) (time_seconds : ℕ) (sound_speed_mps : ℕ) 
  (car_speed_mps : ℕ) (distance_by_car : ℕ) (distance_by_sound : ℕ) 
  (total_distance_equation : 2 * x + distance_by_car = distance_by_sound) : x = 640 :=
by
  have car_speed_kph := 72
  have time_seconds := 4
  have sound_speed_mps := 340
  have car_speed_mps := car_speed_kph * 1000 / 3600
  have distance_by_car := time_seconds * car_speed_mps
  have distance_by_sound := time_seconds * sound_speed_mps
  have total_distance_equation := (2 * x + distance_by_car = distance_by_sound)
  exact sorry

end distance_to_valley_l470_470500


namespace train_length_l470_470084

theorem train_length (v_train : ℝ) (v_man : ℝ) (t : ℝ) (length_train : ℝ)
  (h1 : v_train = 55) (h2 : v_man = 7) (h3 : t = 10.45077684107852) :
  length_train = 180 :=
by
  sorry

end train_length_l470_470084


namespace periodic_even_function_value_l470_470732

-- Conditions
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

noncomputable def f : ℝ → ℝ
| x => if x ∈ Icc (0 : ℝ) 1 then x + 1 else f (mod (2 : ℝ) x)

-- Lean proof statement
theorem periodic_even_function_value :
  periodic_function f ∧ even_function f ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x + 1) → f (3/2) = 3/2 := by
  intros h
  sorry

end periodic_even_function_value_l470_470732


namespace inequality_holds_for_all_reals_l470_470349

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470349


namespace find_product_xy_l470_470962

theorem find_product_xy (x y : ℝ) 
  (h1 : (9 + 10 + 11 + x + y) / 5 = 10)
  (h2 : ((9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2) / 5 = 4) :
  x * y = 191 :=
sorry

end find_product_xy_l470_470962


namespace correct_assignment_l470_470541

structure Person :=
(name : String)

def A := Person.mk "A"
def B := Person.mk "B"
def C := Person.mk "C"

inductive Profession
| Teacher
| Journalist
| Doctor

open Profession

variables (profession : Person → Profession)
variables (age : Person → ℕ)

theorem correct_assignment :
  (age C > age (profession.elim C Doctor)) ∧
  (age A ≠ age (profession.elim A Journalist)) ∧
  (age (profession.elim Journalist Journalist) < age B) →
  profession A = Doctor ∧ profession B = Teacher ∧ profession C = Journalist :=
by
  sorry

end correct_assignment_l470_470541


namespace correct_number_of_propositions_l470_470539

/-
Given conditions:
1. Two planes parallel to the same line are parallel.
2. Two planes parallel to the same plane are parallel.
3. Two lines perpendicular to the same line are parallel.
4. Two lines perpendicular to the same plane are parallel.

Prove that the number of correct propositions among these four is exactly 2.
-/

def proposition_1 : Prop := ∀ (P Q: Plane) (L: Line), (P ∥ L ∧ Q ∥ L) → (P ∥ Q)
def proposition_2 : Prop := ∀ (P Q R: Plane), (P ∥ R ∧ Q ∥ R) → (P ∥ Q)
def proposition_3 : Prop := ∀ (L1 L2: Line) (L3: Line), (L1 ⊥ L3 ∧ L2 ⊥ L3) → (L1 ∥ L2)
def proposition_4 : Prop := ∀ (L1 L2: Line) (P: Plane), (L1 ⊥ P ∧ L2 ⊥ P) → (L1 ∥ L2)

theorem correct_number_of_propositions : (num_propositions_correct [proposition_1, proposition_2, proposition_3, proposition_4] = 2) :=
sorry

end correct_number_of_propositions_l470_470539


namespace hypotenuse_length_l470_470469

-- Definitions and conditions
def right_triangle (a b : ℝ) : Prop := a > 0 ∧ b > 0

def cone_volume (radius height : ℝ) : ℝ := (1 / 3) * Real.pi * radius^2 * height

-- Given conditions
def condition1 (a b : ℝ) (V_a V_b : ℝ) : Prop :=
  V_a = 2 * V_b

def condition2 (V_a V_b : ℝ) : Prop :=
  V_a + V_b = 4480 * Real.pi

def hypotenuse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

-- The statement to prove
theorem hypotenuse_length (a b : ℝ) :
  right_triangle a b →
  ∀ V_a V_b : ℝ,
    condition1 a b V_a V_b →
    condition2 V_a V_b →
    hypotenuse a b ≈ 22.89 :=
begin
  intros h triangle_volume V_a V_b cond1 cond2,
  sorry
end

end hypotenuse_length_l470_470469


namespace find_m_l470_470680

-- Define vectors as pairs of real numbers
def a : ℝ × ℝ := (-2, 3)
def b (m : ℝ) : ℝ × ℝ := (3, m)

-- Define the dot product for 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The theorem to prove
theorem find_m (m : ℝ) : dot_product a (b m) = 0 → m = 2 :=
begin
  -- start proof
  sorry -- proof to be written
end

end find_m_l470_470680


namespace real_root_of_equation_l470_470857

theorem real_root_of_equation (x : ℝ) (hx : sqrt x + sqrt (x + 4) = 12) : x = 1225 / 36 :=
by
  sorry

end real_root_of_equation_l470_470857


namespace range_of_m_increment_function_l470_470747

noncomputable def f (x : ℝ) : ℝ := x^2

theorem range_of_m_increment_function :
  ∀ (m : ℝ), (∀ (x : ℝ), x ∈ set.Ici (-1) → (x + m) ∈ set.Ici (-1) ∧ f (x + m) ≥ f x) ↔ m ≥ 2 := by
  sorry

end range_of_m_increment_function_l470_470747


namespace cows_total_l470_470016

theorem cows_total (M F : ℕ) 
  (h1 : F = 2 * M) 
  (h2 : F / 2 = M / 2 + 50) : 
  M + F = 300 :=
by
  sorry

end cows_total_l470_470016


namespace repair_cost_percentage_l470_470765

-- Definitions for given conditions
def cost_per_apple : ℝ := 1.25
def total_apples_sold : ℕ := 20
def bike_cost : ℝ := 80
def money_remaining_fraction : ℝ := 1/5

-- Calculation of total earnings
def total_earnings : ℝ := total_apples_sold * cost_per_apple

-- Calculation of repair cost
def repair_cost : ℝ := (4/5) * total_earnings

-- Theorem statement
theorem repair_cost_percentage : (repair_cost / bike_cost) * 100 = 25 := by
  sorry

end repair_cost_percentage_l470_470765


namespace lines_connecting_tangents_l470_470883

-- Define the quadrilateral and the inscribed circle with points of tangency
variables {A B C D P Q R S : Point}

-- Define the condition of tangency equalities at each vertex
def tangent_props (A B C D P Q R S : Point) : Prop :=
  dist A P = dist A S ∧ dist B P = dist B Q ∧
  dist C Q = dist C R ∧ dist D S = dist D R

-- Main theorem statement
theorem lines_connecting_tangents (A B C D P Q R S : Point)
  (h_quad: quad_inscribed_circle A B C D P Q R S)
  (h_tangent: tangent_props A B C D P Q R S) :
  (∀ P Q R S, (P ≠ Q ∧ P ≠ S ∧ P ≠ R ∧ Q ≠ S ∧ Q ≠ R ∧ R ≠ S) → 
  (P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D) ∧
  (Q ≠ A ∧ Q ≠ B ∧ Q ≠ C ∧ Q ≠ D) ∧
  (R ≠ A ∧ R ≠ B ∧ R ≠ C ∧ R ≠ D)): 
  ∃ X : Point, (X ∈ extension A C) ∨ (X ∈ extension P Q ∧ X ∈ extension R S) -> 
  (parallel P Q R S) :=
sorry

end lines_connecting_tangents_l470_470883


namespace children_got_on_at_stop_D_l470_470757

/--
Initially, there were 64 children on the bus.
At stop A, 8 children got off the bus, and 12 children got on the bus.
At stop B, 4 children got off the bus, and 6 children got on the bus.
At stop C, 14 children got off the bus, and 22 children got on the bus.
At stop D, 10 children got off the bus, and an unknown number of children got on the bus.
When the bus reached the school, there were 78 children on the bus.
Prove that 10 children got on the bus at stop D.
-/
theorem children_got_on_at_stop_D : 
  ∃ (x : ℕ), 
    (x = 10 ∧ 
     let initial := 64 in
     let after_A := initial - 8 + 12 in
     let after_B := after_A - 4 + 6 in
     let after_C := after_B - 14 + 22 in
     let after_D_before_school := after_C - 10 + x in
     after_D_before_school = 78) :=
sorry

end children_got_on_at_stop_D_l470_470757


namespace broccoli_area_l470_470867

/--
A farmer grows broccoli in a square-shaped farm. This year, he produced 2601 broccoli,
which is 101 more than last year. The shape of the area used for growing the broccoli 
has remained square in both years. Assuming each broccoli takes up an equal amount of 
area, prove that each broccoli takes up 1 square unit of area.
-/
theorem broccoli_area (x y : ℕ) 
  (h1 : y^2 = x^2 + 101) 
  (h2 : y^2 = 2601) : 
  1 = 1 := 
sorry

end broccoli_area_l470_470867


namespace cylinder_lateral_surface_area_l470_470643

-- Definitions for conditions
def radius : ℝ := 2
def height : ℝ := 2

-- Lateral surface area formula for a cylinder
def lateral_surface_area (r h : ℝ) : ℝ := 2 * real.pi * r * h

-- Statement to prove
theorem cylinder_lateral_surface_area : lateral_surface_area radius height = 8 * real.pi := by
  -- Leaving the proof as sorry since it's not required
  sorry

end cylinder_lateral_surface_area_l470_470643


namespace geometric_series_sum_l470_470909

theorem geometric_series_sum :
  let a := 1
  let r := 3
  let n := 9
  (∑ i in finset.range n, a * r ^ i) = 9841 := by
  sorry

end geometric_series_sum_l470_470909


namespace dihedral_angle_B_FB1_E_distance_D_to_B1EF_exists_M_on_DD1_l470_470898

variables {a : ℝ} -- edge length of the cube
variables (A B C D A1 B1 C1 D1 E F : ℝ × ℝ × ℝ)
-- Conditions of cube vertices and midpoints E, F
variables (CubeVertices : 
    (A = (0, 0, 0)) ∧
    (B = (a, 0, 0)) ∧
    (C = (a, a, 0)) ∧
    (D = (0, a, 0)) ∧
    (A1 = (0, 0, a)) ∧
    (B1 = (a, 0, a)) ∧
    (C1 = (a, a, a)) ∧
    (D1 = (0, a, a)) ∧
    (E = ((a / 2), 0, 0)) ∧
    (F = (a, (a / 2), 0)))

-- The Lean statement for each required proof
theorem dihedral_angle_B_FB1_E : CubeVertices a A B C D A1 B1 C1 D1 E F → 
  (∃ theta, theta = real.arctan (real.sqrt 5 / 2)) := sorry

theorem distance_D_to_B1EF : CubeVertices a A B C D A1 B1 C1 D1 E F →
  (∃ d, d = a) := sorry

theorem exists_M_on_DD1 : CubeVertices a A B C D A1 B1 C1 D1 E F →
  (∃ (M : ℝ × ℝ × ℝ), (M = (0, a, x)) ∧ x = (a / 2) ∧ 
  vector.orthogonal (M - B) (plane_span {B1, E, F})) := sorry

end dihedral_angle_B_FB1_E_distance_D_to_B1EF_exists_M_on_DD1_l470_470898


namespace find_angle_SRQ_l470_470259

noncomputable def angle_SRQ (angle_RSQ : ℝ) (angle_RQS : ℝ) : ℝ := 180 - angle_RSQ - angle_RQS

theorem find_angle_SRQ (l k : Type) [affine_space ℝ l] [affine_space ℝ k] 
  (h_parallel : l ∥ k) (angle_RSQ : ℝ) (h_angle_RSQ : angle_RSQ = 130) (angle_RQS : ℝ) (h_angle_RQS : angle_RQS = 90) :
  angle_SRQ angle_RSQ angle_RQS = 40 :=
by
  simp [angle_SRQ, h_angle_RSQ, h_angle_RQS]
  norm_num
  sorry

end find_angle_SRQ_l470_470259


namespace number_of_tiles_per_row_l470_470408

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l470_470408


namespace largest_lambda_existence_and_uniqueness_l470_470591

theorem largest_lambda_existence_and_uniqueness :
  ∃ (lambda : ℝ), (lambda = 2) ∧ 
  (∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 
  (a^2 + b^2 + c^2 + d^2 + a * b^2 ≥ a * b + lambda * b * c + c * d)) ∧
  (∀ (lambda' : ℝ), (∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 
  (a^2 + b^2 + c^2 + d^2 + a * b^2 ≥ a * b + lambda' * b * c + c * d)) 
  → lambda' ≤ lambda) :=
begin
  use 2,
  split,
  { refl },
  { split,
    { intros a b c d ha hb hc hd,
      sorry },
    { intros lambda' h,
      sorry } }
end

end largest_lambda_existence_and_uniqueness_l470_470591


namespace area_of_circleB_l470_470907

-- Definitions based on conditions
def Circle (radius : ℝ) :=
  ∃ center : ℝ × ℝ, true

def area (c : Circle) : ℝ :=
  match c with
  | ⟨radius⟩ => π * radius^2

-- Given conditions
def circleA : Circle 3 := ⟨(0, 0), trivial⟩
def circleB : Circle 6 := ⟨(6, 0), trivial⟩

-- The required proof statement
theorem area_of_circleB : area circleB = 36 * π := by
  sorry

end area_of_circleB_l470_470907


namespace simplify_fraction_expression_l470_470480

theorem simplify_fraction_expression (d : ℤ) :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 :=
by
  -- skip the proof by adding sorry
  sorry

end simplify_fraction_expression_l470_470480


namespace sequence_a_n_is_n_l470_470748

-- Definitions and statements based on the conditions
def sequence_cond (a : ℕ → ℕ) (n : ℕ) : ℕ := 
1 / 2 * (a n) ^ 2 + n / 2

theorem sequence_a_n_is_n :
  ∀ (a : ℕ → ℕ), (∀ n, n > 0 → ∃ (S_n : ℕ), S_n = sequence_cond a n) → 
  (∀ n, n > 0 → a n = n) :=
by
  sorry

end sequence_a_n_is_n_l470_470748


namespace exists_polynomial_q_l470_470278

theorem exists_polynomial_q (n : ℕ) (ε : ℂ) (p : polynomial ℤ)
  (h1 : ε^n = 1) (h2 : (p.eval ε).im = 0) :
  ∃ q : polynomial ℤ, (p.eval ε).re = q.eval (2 * (Real.cos (2 * Real.pi / n))) :=
by
  sorry

end exists_polynomial_q_l470_470278


namespace simplify_expression_l470_470454

variable (x : ℝ)

theorem simplify_expression :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x) ^ 3) = (25 / 8) * x^2 :=
by
  sorry

end simplify_expression_l470_470454


namespace greatest_of_four_consecutive_sum_is_102_l470_470845

theorem greatest_of_four_consecutive_sum_is_102 :
  ∃ n : ℤ, let a := n in let b := n+1 in let c := n+2 in let d := n+3 in 
    a + b + c + d = 102 ∧ d = 27 :=
sorry

end greatest_of_four_consecutive_sum_is_102_l470_470845


namespace age_difference_l470_470846

variable (A B C : ℕ)

theorem age_difference : A + B = B + C + 11 → A - C = 11 := by
  sorry

end age_difference_l470_470846


namespace proof_a_plus_b_l470_470609

theorem proof_a_plus_b (a b c : ℝ) (h1 : a - b = 4) (h2 : ab + c^2 + 4 = 0) : a + b = 0 := 
begin
  sorry
end

end proof_a_plus_b_l470_470609


namespace tigger_climb_ratio_l470_470476

-- Define the times taken by Tigger to climb up and climb down
variables (T t : ℝ)

-- Define the conditions given in the problem
-- Condition 1: Winnie climbs up twice as slowly as Tigger
def winnie_climb_up_time : ℝ := 2 * T

-- Condition 2: Winnie descends three times faster than Tigger
def winnie_descend_time : ℝ := t / 3

-- Condition 3: They started and finished at the same time
def equal_total_time (T t : ℝ) : Prop :=
    T + t = 2 * T + t / 3

-- Formalizing the problem statement
theorem tigger_climb_ratio (T t : ℝ)
  (h1 : winnie_climb_up_time T = 2 * T)
  (h2 : winnie_descend_time t = t / 3)
  (h3 : equal_total_time T t) : (T / t = 1.5) :=
sorry

end tigger_climb_ratio_l470_470476


namespace equations_of_motion_velocity_of_M_l470_470106

noncomputable def crank_radius : ℝ := 90
noncomputable def angular_velocity : ℝ := 10
noncomputable def AM_AB_ratio : ℝ := 2 / 3
noncomputable def AM : ℝ := AM_AB_ratio * crank_radius

def x_M (t : ℝ) : ℝ := crank_radius * Real.cos (angular_velocity * t) - AM * Real.sin (angular_velocity * t)
def y_M (t : ℝ) : ℝ := crank_radius * Real.sin (angular_velocity * t) - AM * Real.cos (angular_velocity * t)

def v_x (t : ℝ) : ℝ := -9 * angular_velocity * crank_radius * Real.sin (angular_velocity * t) - 6 * angular_velocity * crank_radius * Real.cos (angular_velocity * t)
def v_y (t : ℝ) : ℝ := 9 * angular_velocity * crank_radius * Real.cos (angular_velocity * t) + 6 * angular_velocity * crank_radius * Real.sin (angular_velocity * t)

def v_M (t : ℝ) : ℝ := Real.sqrt ((v_x t) ^ 2 + (v_y t) ^ 2)

theorem equations_of_motion :
  ∀ t : ℝ, 
    x_M t = crank_radius * Real.cos (angular_velocity * t) - AM * Real.sin (angular_velocity * t) ∧
    y_M t = crank_radius * Real.sin (angular_velocity * t) - AM * Real.cos (angular_velocity * t) := 
by sorry

theorem velocity_of_M :
  ∀ t : ℝ,
    v_x t = -9 * angular_velocity * crank_radius * Real.sin (angular_velocity * t) - 6 * angular_velocity * crank_radius * Real.cos (angular_velocity * t) ∧
    v_y t = 9 * angular_velocity * crank_radius * Real.cos (angular_velocity * t) + 6 * angular_velocity * crank_radius * Real.sin (angular_velocity * t) ∧
    v_M t = Real.sqrt ((v_x t) ^ 2 + (v_y t) ^ 2) :=
by sorry

end equations_of_motion_velocity_of_M_l470_470106


namespace sum_of_real_roots_l470_470939

theorem sum_of_real_roots (P : Polynomial ℝ) (h : P = Polynomial.C (-1) + Polynomial.X ^ 4 - 4 * Polynomial.X ^ 3) :
  (∑ x in P.roots.toFinset, x) = 4 :=
by sorry

end sum_of_real_roots_l470_470939


namespace general_formula_arithmetic_geometric_sum_T_n_l470_470622

open_locale big_operators

variable {ℕ : Type}

def arithmetic_sequence (d : ℕ) (a₁ : ℕ) : ℕ → ℕ
| n := a₁ + (n - 1) * d

def geometric_sequence (q : ℕ) (b₁ : ℕ) : ℕ → ℕ
| n := b₁ * q^(n - 1)

def c_n (a_n b_n : ℕ → ℕ) (n : ℕ) : ℕ :=
2^(a_n n) * (nat.log (b_n n) / nat.log 2).nat_abs

def T_n (c_n : ℕ → ℕ) (n : ℕ) : ℕ :=
∑ i in finset.range n, c_n (i + 1)

theorem general_formula_arithmetic_geometric (a_n b_n : ℕ → ℕ) (d q a₁ b₁ n : ℕ)
  (ha₁ : a_n 1 = a₁) (hb₁ : b_n 1 = b₁) (hd : d = 1) (hq : q = 2) 
  (ha1b1 : a₁ * b₁ = 1) 
  (h2adotb : a₁ * b₁ + (a₁ + d) * b₁ * q = 5) :
  (∀ n, a_n n = n) ∧ (∀ n, b_n n = 2^(n-1)) :=
sorry

theorem sum_T_n (a_n b_n : ℕ → ℕ) (n : ℕ)
  (ha_n : ∀ n, a_n n = n) (hb_n : ∀ n, b_n n = 2^(n-1)) :
  T_n (c_n a_n b_n) n = 4 + (n - 2) * 2^(n + 1) :=
sorry

end general_formula_arithmetic_geometric_sum_T_n_l470_470622


namespace find_d_values_l470_470588

open Set

theorem find_d_values :
  ∀ {f : ℝ → ℝ}, ContinuousOn f (Icc 0 1) → (f 0 = f 1) →
  ∃ (d : ℝ), d ∈ Ioo 0 1 ∧ (∀ x₀, x₀ ∈ Icc 0 (1 - d) → (f x₀ = f (x₀ + d))) ↔
  ∃ k : ℕ, d = 1 / k :=
by
  sorry

end find_d_values_l470_470588


namespace inequality_proof_l470_470334

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470334


namespace geometric_sequence_log_sum_l470_470645

noncomputable def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n n > 0

theorem geometric_sequence_log_sum (a_n : ℕ → ℝ) 
    (h_seq : geometric_sequence a_n)
    (h_cond : a_n 5 * a_n 6 + a_n 4 * a_n 7 = 18) :
  (Real.log 3 (a_n 1) + Real.log 3 (a_n 2) + Real.log 3 (a_n 3) + Real.log 3 (a_n 4) 
  + Real.log 3 (a_n 5) + Real.log 3 (a_n 6) + Real.log 3 (a_n 7) + Real.log 3 (a_n 8) 
  + Real.log 3 (a_n 9) + Real.log 3 (a_n 10)) = 10 := 
sorry

end geometric_sequence_log_sum_l470_470645


namespace inequality_proof_l470_470339

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470339


namespace ellipse_line_intersection_l470_470170

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)

def right_focus (c : ℝ) : Prop :=
  ∃ a b : ℝ, c = real.sqrt (a^2 - b^2)

def intersection_points (k b : ℝ) : ℝ :=
  let eq := λ x y, (x^2 + 4 * (k * x + b)^2 - 4 = 0) in
  ∃ x y : ℝ, eq x y

theorem ellipse_line_intersection
  (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b)
  (h₃ : 2 * a = 4) (h₄ : right_focus (real.sqrt 3))
  (F : ℝ × ℝ) (hF : F = (real.sqrt 3, 0))
  (A B : ℝ × ℝ) (intersect_A : A.1 = 2) (intersect_B : B.1 = -2)
  (dot_product : (F.1 - 2) * (B.1 - F.1) + F.2 * B.2 = 0) :
  (ellipse_eq a b) ∧ (∀ k b, b^2 = 1 + 4 * k^2 → intersection_points k b = 1) :=
begin
  sorry
end

end ellipse_line_intersection_l470_470170


namespace unique_root_sum_sum_of_distinct_p_l470_470046

theorem unique_root_sum (p : ℝ) :
  (∃ x : ℝ, (x^2 - 2*p*x + p^2 + p - 20 = 0) ∧ (∀ y : ℝ, y^2 - 2*p*y + p^2 + p - 20 = 0 → y = x)) → p = 20 :=
begin
  sorry
end

theorem sum_of_distinct_p : ℝ :=
begin
  exact 20
end

end unique_root_sum_sum_of_distinct_p_l470_470046


namespace problem1_intervals_of_monotonicity_problem2_range_of_a_l470_470189

section problem1
  variable (x : ℝ) (f : ℝ → ℝ) [derivable f]

  -- Define the function
  def f (x : ℝ) : ℝ := x / Real.log x
  -- Define the derivative of the function
  def f' (x : ℝ) : ℝ := (Real.log x - 1) / (Real.log x)^2
  
  -- Problem 1 statement
  theorem problem1_intervals_of_monotonicity :
    (∀ x, x > Real.exp 1 → 0 < f' x) ∧
    (∀ x, (1 < x ∧ x < Real.exp 1) ∨ (0 < x ∧ x < 1) → f' x < 0) :=
  sorry
end problem1

section problem2
  variable (a x : ℝ) (f : ℝ → ℝ)

  -- Define the function
  def f (x : ℝ) := (x - a) / Real.log x
  
  -- Problem 2 statement
  theorem problem2_range_of_a (h : ∀ x, 1 < x → f x > Real.sqrt x) : a ≤ 1 :=
  sorry
end problem2

end problem1_intervals_of_monotonicity_problem2_range_of_a_l470_470189


namespace sequence_properties_l470_470524

-- Given the sequence {a_n} with positive terms and its partial sum equation:
-- (1) 2S_n^2 - (3n^2 + 3n - 2)S_n - 3(n^2 + n) = 0, where n is a natural number
-- (2) b_n = a_n / 3^(n+1)

open Nat

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Define the conditions given in the problem
def positive_terms (a : ℕ → ℝ) := ∀ n, 0 < a n
def partial_sum_eqn (S : ℕ → ℝ) (n : ℕ) := 
  2 * (S n) ^ 2 - (3 * n ^ 2 + 3 * n - 2) * (S n) - 3 * (n ^ 2 + n) = 0
def bn_def (a : ℕ → ℝ) (b : ℕ → ℝ) := ∀ n, b n = a n / 3 ^ (n + 1)

-- Proof problem:
theorem sequence_properties 
  {a : ℕ → ℝ} {S : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ} 
  (h_pos : positive_terms a)
  (h_sum_eqn : ∀ n, n > 0 → partial_sum_eqn S n)
  (h_bn : bn_def a b) :
  (a 1 = 3) ∧
  (∀ n, n > 0 → a n = 3 * n) ∧
  (∀ n, n > 0 → T n = (∑ i in Finset.range n, b i) → T n = 3 / 4 - (2 * n + 3) / (4 * 3 ^ n)) :=
by 
  sorry

end sequence_properties_l470_470524


namespace number_of_oxygen_atoms_in_compound_l470_470933

-- Definitions

def molecularWeightCompound : Float := 122.0
def atomicWeightAl : Float := 26.98
def atomicWeightP : Float := 30.97
def atomicWeightO : Float := 16.00

-- Theorem statement
theorem number_of_oxygen_atoms_in_compound :
  let totalWeightAlP := atomicWeightAl + atomicWeightP
  let weightOfO := molecularWeightCompound - totalWeightAlP
  let numOxygenAtoms := weightOfO / atomicWeightO
  Int.round numOxygenAtoms = 4 := 
by
  sorry

end number_of_oxygen_atoms_in_compound_l470_470933


namespace ellipse_properties_l470_470639

noncomputable def ellipse_center_at_origin (a b c : ℝ) : Prop :=
  let e := c / a in
  a^2 - b^2 = c^2 ∧ a = 2 * b 

theorem ellipse_properties :
  ∀ (a b c : ℝ),
  ellipse_center_at_origin a b c →
  c = 2 →
  (c / a = Real.sqrt 3 / 2 ∧ 
  (∀ x y : ℝ, (y^2 / a^2 + x^2 / b^2 = 1) = (y^2 / (16/3) + x^2 / (4/3) = 1))) :=
by 
  sorry

end ellipse_properties_l470_470639


namespace statementA_correct_statementB_incorrect_statementC_correct_statementD_incorrect_l470_470828

theorem statementA_correct (x : ℝ) (h : x > 1) : 
  let y := 2 * x + 4 / (x - 1) - 1 in
  y ≥ 4 * Real.sqrt 2 + 1 ∧ (x = Real.sqrt 2 + 1 → y = 4 * Real.sqrt 2 + 1) :=
sorry

theorem statementB_incorrect (x : ℝ) :
  ¬∃ x, let y := (x^2 + 1) / x in y = 2 :=
sorry

theorem statementC_correct (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 3 * x * y) :
  let z := 2 * x + y in
  z ≥ 3 ∧ (x = 1 ∧ y = 1 → z = 3) :=
sorry

theorem statementD_incorrect (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 3 * x * y) :
  ¬∃ z, let z := 2 * x + y in z = 3 :=
sorry

end statementA_correct_statementB_incorrect_statementC_correct_statementD_incorrect_l470_470828


namespace decimal_to_fraction_l470_470033

theorem decimal_to_fraction :
  (3.56 : ℚ) = 89 / 25 := 
sorry

end decimal_to_fraction_l470_470033


namespace prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l470_470901

noncomputable def P_A := 4 / 5
noncomputable def P_B := 3 / 4

def independent (P_X P_Y : ℚ) := P_X * P_Y

theorem prob_both_shoot_in_one_round : independent P_A P_B = 3 / 5 := by
  sorry

noncomputable def P_A_1 := 2 * (4 / 5) * (1 / 5)
noncomputable def P_A_2 := (4 / 5) * (4 / 5)
noncomputable def P_B_1 := 2 * (3 / 4) * (1 / 4)
noncomputable def P_B_2 := (3 / 4) * (3 / 4)

def event_A (P_A_1 P_A_2 P_B_1 P_B_2 : ℚ) := (P_A_1 * P_B_2) + (P_A_2 * P_B_1)

theorem prob_specified_shots_in_two_rounds : event_A P_A_1 P_A_2 P_B_1 P_B_2 = 3 / 10 := by
  sorry

end prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l470_470901


namespace right_trapezoid_similarity_points_l470_470547

variable (P A B C D : Type)
variables (AP PB AB AD BC : ℝ)
variables (tPAD tPBC : Type)

theorem right_trapezoid_similarity_points (ABCD : right_trapezoid ABCD) 
  (h_AB : AB = 7) (h_AD : AD = 2) (h_BC : BC = 3) 
  (h_similarity1 : similar_triangles tPAD tPBC)
  (h_on_line : point_on_line P AB) : 
  ∃ (x : ℝ), count_points P AB (PAD P A D) (PBC P B C) = 3 :=
sorry

end right_trapezoid_similarity_points_l470_470547


namespace sum_of_odd_divisors_360_l470_470461

theorem sum_of_odd_divisors_360 : 
  let n := 360 in
  let prime_factors := 2^3 * 3^2 * 5^1 in
  let odd_divisors_sum :=
    let a := ∑ k in (finset.range 3), 3^k in  -- Sum for 3^0, 3^1, 3^2 
    let b := ∑ k in (finset.range 2), 5^k in  -- Sum for 5^0, 5^1
    a * b
  in
  odd_divisors_sum = 78 :=
by
  let a := ∑ k in (finset.range 3), 3^k
  let b := ∑ k in (finset.range 2), 5^k
  let odd_divisors_sum := a * b
  have ha : a = 13 := by sorry   -- Sum of 3^0 + 3^1 + 3^2
  have hb : b = 6 := by sorry    -- Sum of 5^0 + 5^1
  have hsum : odd_divisors_sum = 13 * 6 := by sorry
  show odd_divisors_sum = 78, from by rw [hsum, ha, hb]; exact rfl

end sum_of_odd_divisors_360_l470_470461


namespace pyramid_game_minimal_k_l470_470858

-- Definition of the pyramid game conditions
def pyramid_game_conditions (n : ℕ) :=
  ∀ (E : Finset (Fin n × Fin n)) (k : ℕ),
    (∀ (u v : Fin n), (u, v) ∈ E → u < v) ∧      -- edges defined by pairs of vertices
    (∀ (u v : Fin n) (α β : ℕ), (u, v) ∈ E → u ≠ v → α ≠ β → colored u = α → colored v ≠ β)  -- coloring constraint
    ∧ (∀ (v : Fin n), v = 2016 ∨ ∃ p, p > 2016 ∧ (v, p) ∈ E.-- all vertices either 2016-vertex or have connections to apex vertex

-- Definition of the minimum k value for which B can guarantee coloring
def minimal_k (n : ℕ) := ∃ k, (pyramid_game_conditions n) → k = 2016

-- The theorem to state the problem
theorem pyramid_game_minimal_k : minimal_k 4032 :=
begin
  -- proof omitted since it is stated that proof is not required
  sorry
end

end pyramid_game_minimal_k_l470_470858


namespace area_of_circle_below_line_l470_470814

theorem area_of_circle_below_line (x y : ℝ) :
  (x - 3)^2 + (y - 5)^2 = 9 →
  y ≤ 8 →
  ∃ (A : ℝ), A = 9 * Real.pi :=
sorry

end area_of_circle_below_line_l470_470814


namespace no_real_quadruples_l470_470934

open Matrix

theorem no_real_quadruples (a b c d : ℝ) (h_inv : (matrix.of ![![a, b], ![c, d]].inverse = ![![2 / a, 1 / b], ![1 / c, 2 / d]] : Matrix (Fin 2) (Fin 2) ℝ)) : False :=
by
  sorry

end no_real_quadruples_l470_470934


namespace angle_division_quadrant_l470_470175

variable (k : ℤ)
variable (α : ℝ)
variable (h : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi)

theorem angle_division_quadrant 
  (hα_sec_quadrant : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi) : 
  (∃ m : ℤ, (m = 0 ∧ Real.pi / 4 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
             (m = 1 ∧ Real.pi * (2 * m + 1) / 4 < α / 2 ∧ α / 2 < Real.pi * (2 * m + 1) / 2)) :=
sorry

end angle_division_quadrant_l470_470175


namespace cost_per_pound_of_mixture_l470_470065

theorem cost_per_pound_of_mixture (w1 w2 : ℤ) (p1 p2 : ℝ)
  (h_w1 : w1 = 20) (h_w2 : w2 = 80) (h_p1 : p1 = 10) (h_p2 : p2 = 5) :
  (w1 * p1 + w2 * p2) / (w1 + w2) = 6 :=
by
  -- Conditions given in the problem
  have h_total_weight : w1 + w2 = 100 := by
    -- Calculation details and skips
    rw [h_w1, h_w2]
    norm_num

  have h_total_cost : w1 * p1 + w2 * p2 = 600 := by
    -- Calculation details and skips
    rw [h_w1, h_w2, h_p1, h_p2]
    norm_num

  rw [h_total_weight, h_total_cost]
  norm_num

end cost_per_pound_of_mixture_l470_470065


namespace root_in_interval_l470_470958

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 4

theorem root_in_interval : ∃ x ∈ set.Ioo 1 2, f x = 0 := 
by {
  sorry
}

end root_in_interval_l470_470958


namespace set_A_is_listed_l470_470582

noncomputable def A : Set ℤ := { x | 2 / (x + 1) ∈ ℤ }

theorem set_A_is_listed :
  A = {-3, -2, 0, 1} :=
by
  sorry

end set_A_is_listed_l470_470582


namespace extreme_point_inequality_l470_470667

open Real

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x - 1 - a * log x

theorem extreme_point_inequality (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < 1) (h4 : -1/2 < a) (h5 : a < 0)
  (h6 : (2 * x1^2 - 2 * x1 - a = 0)) (h7 : (2 * x2^2 - 2 * x2 - a = 0)) :
  (f x1 a) / x2 > -7/2 - log 2 :=
sorry

end extreme_point_inequality_l470_470667


namespace proof_of_expression_l470_470691

theorem proof_of_expression (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 :=
by {
  sorry
}

end proof_of_expression_l470_470691


namespace simplify_fraction_expression_l470_470477

variable (d : ℝ)

theorem simplify_fraction_expression :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := 
by
  sorry

end simplify_fraction_expression_l470_470477


namespace inequality_holds_for_real_numbers_l470_470321

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470321


namespace N_scaling_l470_470140

open Real Matrix

axiom N : Matrix (Fin 2) (Fin 2) ℝ

theorem N_scaling (N : Matrix (Fin 2) (Fin 2) ℝ) (h : ∀ w : Fin 2 → ℝ, mulVec N w = 7 • w) :
  N = (λ i j, if i = j then 7 else 0) := 
sorry

end N_scaling_l470_470140


namespace problem_solution_l470_470240

theorem problem_solution :
  (∀ (p q : ℚ), 
    (∀ (x : ℚ), (x + 3 * p) * (x^2 - x + (1 / 3) * q) = x^3 + (3 * p - 1) * x^2 + ((1 / 3) * q - 3 * p) * x + p * q) →
    (3 * p - 1 = 0) →
    ((1 / 3) * q - 3 * p = 0) →
    p = 1 / 3 ∧ q = 3)
  ∧ ((1 / 3) ^ 2020 * 3 ^ 2021 = 3) :=
by
  sorry

end problem_solution_l470_470240


namespace count_integers_in_sqrt_seq_l470_470951

theorem count_integers_in_sqrt_seq (n : ℕ) : 
  (λ i : ℕ, (∃ k : ℤ, (1024 : ℚ)^(1 / (i : ℚ)) = k)) = 2 := 
sorry

end count_integers_in_sqrt_seq_l470_470951


namespace wu_yang_competition_winners_l470_470150

theorem wu_yang_competition_winners :
  ∃ (order : list char), 
    (A_pred (order.nth 0 = some 'C') ∧ (order.nth 1 = some 'B')) ∨ 
    (B_pred (order.nth 2 = some 'A') ∧ (order.nth 3 = some 'D')) ∨ 
    (C_pred (order.nth 3 = some 'E') ∧ (order.nth 4 = some 'D')) ∨ 
    (D_pred (order.nth 2 = some 'B') ∧ (order.nth 4 = some 'C')) ∨ 
    (E_pred (order.nth 0 = some 'A') ∧ (order.nth 3 = some 'E'))
    :=
    order = ['D', 'B', 'A', 'E', 'C'] := 
sorry

end wu_yang_competition_winners_l470_470150


namespace slopes_and_angles_l470_470780

theorem slopes_and_angles (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : θ₁ = 3 * θ₂)
  (h2 : m = 5 * n)
  (h3 : m = Real.tan θ₁)
  (h4 : n = Real.tan θ₂)
  (h5 : m ≠ 0) :
  m * n = 5 / 7 :=
by {
  sorry
}

end slopes_and_angles_l470_470780


namespace shortest_chord_length_l470_470431

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 2 = 0

noncomputable def center : ℝ × ℝ := ⟨1, -1⟩

noncomputable def radius : ℝ := 2

noncomputable def point : ℝ × ℝ := ⟨0, 0⟩

theorem shortest_chord_length :
  let d := real.sqrt ((center.1 - point.1) ^ 2 + (center.2 - point.2) ^ 2) in
  let chord_len := 2 * real.sqrt (radius ^ 2 - d ^ 2) in
  chord_len = 2 * real.sqrt 2 :=
by
  sorry

end shortest_chord_length_l470_470431


namespace sum_odd_divisors_360_is_78_l470_470463

/-- Define what it means to be an odd divisor of 360 --/
def is_odd_divisor (n : ℕ) : Prop :=
  n ∣ 360 ∧ n % 2 = 1

/-- Define the sum of all odd divisors of 360 --/
def sum_odd_divisors_360 : ℕ :=
  ∑ d in (Finset.filter is_odd_divisor (Finset.range 361)), d

/-- Prove that the sum of all odd divisors of 360 is 78 --/
theorem sum_odd_divisors_360_is_78 : sum_odd_divisors_360 = 78 := by
  sorry

end sum_odd_divisors_360_is_78_l470_470463


namespace quadratic_eq_is_E1_l470_470090

-- Defining conditions
def E1 : Prop := ∃ (x : ℝ), x^2 - 5 * x = 0
def E2 : Prop := ∃ (x : ℝ), x + 1 = 0
def E3 : Prop := ∃ (x y : ℝ), y - 2 * x = 0
def E4 : Prop := ∃ (x : ℝ), 2 * x^3 - 2 = 0

-- Statement to prove E1 is quadratic
theorem quadratic_eq_is_E1 : 
  (E1 ∧ (¬ E2 ∧ ¬ E3 ∧ ¬ E4)) :=
by
  split
  -- Provide trivial proof parts
  sorry

end quadratic_eq_is_E1_l470_470090


namespace emily_spent_on_flowers_l470_470897

theorem emily_spent_on_flowers :
  let cost_per_flower := 3 in
  let num_roses := 2 in
  let num_daisies := 2 in
  let total_flowers := num_roses + num_daisies in
  let pre_discount_cost := total_flowers * cost_per_flower in
  let discount := if total_flowers > 3 then 0.20 * pre_discount_cost else 0 in
  let final_cost := pre_discount_cost - discount in
  final_cost = 9.60 := 
by
  sorry

end emily_spent_on_flowers_l470_470897


namespace intersection_points_of_line_l470_470779

theorem intersection_points_of_line (x y : ℝ) :
  ((y = 2 * x - 1) → (y = 0 → x = 0.5)) ∧
  ((y = 2 * x - 1) → (x = 0 → y = -1)) :=
by sorry

end intersection_points_of_line_l470_470779


namespace inequality_holds_for_all_reals_l470_470345

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470345


namespace grade_10_sample_l470_470004

theorem grade_10_sample (x : ℕ) (hx : x > 0) : 
    let total_students := 4 * x + 4 * x + 5 * x in
    let sample_size := 65 in
    let grade_10_fraction := (4 * x : ℚ) / (total_students : ℚ) in
    (grade_10_fraction * sample_size).natAbs = 20 :=
by
  -- Definitions
  let total_students := 4 * x + 4 * x + 5 * x
  let sample_size := 65
  let grade_10_fraction := (4 * x : ℚ) / (total_students : ℚ)
  -- Calculation
  have h1 : total_students = 13 * x := by ring
  have h2 : grade_10_fraction = 4 / 13 := by field_simp [h1, hx]
  have h3 : (grade_10_fraction * sample_size).natAbs = 20
    := by norm_num; exact (nat.abs_of_nat 20).symm
  exact h3

end grade_10_sample_l470_470004


namespace range_of_m_l470_470727

noncomputable def M (m : ℝ) := {x : ℝ | x + m ≥ 0}
noncomputable def N := {x : ℝ | x^2 - 2*x - 8 < 0}
def U := set.univ : set ℝ
def C_U_M (m : ℝ) := {x : ℝ | x < -m}

theorem range_of_m (m : ℝ) : (∃ x, x ∈ (C_U_M m ∩ N)) → m ≤ 2 :=
by
  sorry

end range_of_m_l470_470727


namespace solve_trigonometric_equation_l470_470772

theorem solve_trigonometric_equation :
  ∃ n : ℤ, ∀ x : ℝ, (sin x ^ 3 + 6 * cos x ^ 3 + (1 / sqrt 2) * sin (2 * x) * sin (x + π / 4) = 0 ↔ x = -arctan 2 + n * π) :=
sorry

end solve_trigonometric_equation_l470_470772


namespace shift_graph_right_l470_470197

def g (x : ℝ) : ℝ := Math.sin (2 * x)
def f (x : ℝ) : ℝ := Math.sin (2 * x - π / 4)

theorem shift_graph_right (x : ℝ) : g (x - π / 8) = f x :=
sorry

end shift_graph_right_l470_470197


namespace inequality_proof_l470_470341

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470341


namespace inequality_proof_l470_470338

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470338


namespace sum_first_2017_terms_l470_470435

-- Given sequence definition
def a : ℕ → ℕ
| 0       => 0 -- a_0 (dummy term for 1-based index convenience)
| 1       => 1
| (n + 2) => 3 * 2^(n) - a (n + 1)

-- Sum of the first n terms of the sequence {a_n}
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + a (n + 1)

-- Theorem to prove
theorem sum_first_2017_terms : S 2017 = 2^2017 - 1 :=
sorry

end sum_first_2017_terms_l470_470435


namespace area_of_triangle_KEM_fraction_l470_470758

-- Defining the geometrical conditions:
variable {α : Type} [LinearOrderedField α]
variables (A B C K E M : EuclideanGeometry.Point α)
variables (AB AC BC : α)
variable (x : α)

def is_isosceles_triangle (AB AC : α) : Prop :=
  AB = AC

def on_line_segment (P A C : EuclideanGeometry.Point α) : Prop := 
  ∃ t : α, 0 ≤ t ∧ t ≤ 1 ∧ P = EuclideanGeometry.affine_combination A C t

def parallel (l₁ l₂ : EuclideanGeometry.Line α) : Prop :=
  ∃ k : α, k ≠ 0 ∧ l₁.direction = k • l₂.direction

-- Instantiate the geometric conditions to our problem
def given_conditions (A B C K E M : EuclideanGeometry.Point α)
  (AB AC BC : α) (x : α) :=
  is_isosceles_triangle AB AC ∧
  on_line_segment E A C ∧
  on_line_segment K A B ∧
  on_line_segment M B C ∧
  parallel (EuclideanGeometry.line_through K E) (EuclideanGeometry.line_through B C) ∧
  parallel (EuclideanGeometry.line_through E M) (EuclideanGeometry.line_through A B) ∧
  BM = 2 * x ∧ EM = 3 * x

-- Statement of the proof problem
theorem area_of_triangle_KEM_fraction (A B C K E M : EuclideanGeometry.Point α)
  (AB AC BC : α) (x : α) 
  (h : given_conditions A B C K E M AB AC BC x) :
  EuclideanGeometry.area (triangle A B C) > 0 → 
  EuclideanGeometry.area (triangle K E M) = 6 / 25 * EuclideanGeometry.area (triangle A B C) :=
sorry

end area_of_triangle_KEM_fraction_l470_470758


namespace liouville_theorem_l470_470314

theorem liouville_theorem (p : ℕ) (m : ℕ) : prime p ∧ p > 5 ∧ m > 0 → (factorial (p - 1)) + 1 ≠ p^m :=
by
  sorry

end liouville_theorem_l470_470314


namespace range_of_f_when_lambda_is_three_halves_value_of_lambda_when_min_f_is_one_l470_470190

-- Definition of the function f(x)
def f (x : ℝ) (λ : ℝ) : ℝ := (1 / (4 ^ x)) - (λ / (2 ^ (x - 1))) + 3

-- Condition: interval for x
def interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2

-- Proof Statements

-- Part 1: Prove the range of f(x) when λ = 3/2
theorem range_of_f_when_lambda_is_three_halves :
  (∀ x, interval x → (3/4) ≤ f x (3/2) ∧ f x (3/2) ≤ 37/16) :=
sorry

-- Part 2: Prove the value of λ when the minimum value of f(x) is 1
theorem value_of_lambda_when_min_f_is_one :
  (∀ x, interval x → (∃ λ, (f x λ ≥ 1) ∧ (∀ y, interval y → f y λ ≥ f x λ) → λ = sqrt 2)) :=
sorry

end range_of_f_when_lambda_is_three_halves_value_of_lambda_when_min_f_is_one_l470_470190


namespace shortest_chord_length_l470_470430

-- Definitions for the problem
def circle_eqn (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - 2 = 0
  
def point_P := (0 : ℝ, 0 : ℝ)
def center_C := (1 : ℝ, -1 : ℝ)
def radius := 2

-- Definition of the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The theorem to prove: The shortest chord length cut by the line passing through point P on the circle is 2√2.
theorem shortest_chord_length : ∀ (x y : ℝ),
  circle_eqn x y → distance point_P center_C < radius →
  ∃ d : ℝ, d = 2 * real.sqrt 2 :=
sorry

end shortest_chord_length_l470_470430


namespace f_analytical_expression_f_increasing_solve_inequality_l470_470420

section MathProofs

variables {a b t x : ℝ}

/-- Definition of f(x) given the conditions -/
def f (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

/-- Conditions of the problem -/
axiom odd_function : ∀ x, f (-x) = -f (x)
axiom f_half : f (1 / 2) = 2 / 5
axiom f_zero : f 0 = 0

/-- Prove the analytical expression of f(x) is f(x) = x / (1 + x^2) -/
theorem f_analytical_expression : f x = x / (1 + x^2) :=
sorry

/-- Prove that f(x) is an increasing function on (-1, 1) -/
theorem f_increasing : ∀ x1 x2 ∈ set.Ioo (-1 : ℝ) 1, x1 < x2 → f x1 < f x2 :=
sorry

/-- Solve the inequality f(t-1) + f(t) < 0 for 0 < t < 1/2 -/
theorem solve_inequality (h : 0 < t ∧ t < 1/2) : f (t - 1) + f t < 0 :=
sorry

end MathProofs

end f_analytical_expression_f_increasing_solve_inequality_l470_470420


namespace inequality_holds_for_all_real_numbers_l470_470367

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470367


namespace length_of_KN_is_124_l470_470248

-- Declare the setup for ∆XYZ
variables (X Y Z N K : Type)
variables (point : X → Y → Z → Prop)
variables (midpoint : N → Y → Z → Prop)
variables (foot_of_altitude : K → Z → X → Y → Prop)

-- Declare the properties of the triangle
axiom XYZ_sides : point X Y Z ∧ XY = 15 ∧ YZ = 17 ∧ ZX = 20
axiom N_midpoint_YZ : midpoint N Y Z
axiom K_foot_of_altitude_ZY : foot_of_altitude K Z X Y

-- Define the proof for the length of KN being 12.4
theorem length_of_KN_is_124 :
  ∃ (KN : ℝ), KN = 12.4 :=
begin
  use 12.4,
  sorry
end

end length_of_KN_is_124_l470_470248


namespace largest_prime_factor_1296_l470_470816

theorem largest_prime_factor_1296 : ∃ p : ℕ, nat.prime p ∧ p.factorization 1296 = 1 ∧ ∀ q : ℕ, nat.prime q ∧ q.factorization 1296 = 1 → q ≤ p :=
by
  sorry

end largest_prime_factor_1296_l470_470816


namespace net_income_difference_l470_470383

theorem net_income_difference
    (terry_daily_income : ℝ := 24) (terry_daily_hours : ℝ := 6) (terry_days : ℕ := 7)
    (jordan_daily_income : ℝ := 30) (jordan_daily_hours : ℝ := 8) (jordan_days : ℕ := 6)
    (standard_week_hours : ℝ := 40) (overtime_rate_multiplier : ℝ := 1.5)
    (terry_tax_rate : ℝ := 0.12) (jordan_tax_rate : ℝ := 0.15) :
    jordan_daily_income * jordan_days - jordan_daily_income * jordan_days * jordan_tax_rate 
      + jordan_daily_income * jordan_days * jordan_daily_hours * (overtime_rate_multiplier - 1) * jordan_tax_rate
    - (terry_daily_income * terry_days - terry_daily_income * terry_days * terry_tax_rate 
      + terry_daily_income * terry_days * terry_daily_hours * (overtime_rate_multiplier - 1) * terry_tax_rate) 
      = 32.85 := 
sorry

end net_income_difference_l470_470383


namespace notecard_area_new_dimension_l470_470750

theorem notecard_area_new_dimension :
  ∀ (length : ℕ) (width : ℕ) (shortened : ℕ),
    length = 7 →
    width = 5 →
    shortened = 2 →
    (width - shortened) * length = 21 →
    (length - shortened) * (width - shortened + shortened) = 25 :=
by
  intros length width shortened h_length h_width h_shortened h_area
  sorry

end notecard_area_new_dimension_l470_470750


namespace find_f_10_l470_470657

def f : ℕ → ℝ
| 1       := 1
| (n + 1) := (f n) / (1 + (f n))

theorem find_f_10 : f 10 = 1 / 10 := 
by 
  sorry

end find_f_10_l470_470657


namespace smoothie_combinations_l470_470888

theorem smoothie_combinations (flavors : ℕ) (toppings : ℕ) (h_flavors : flavors = 5) (h_toppings : toppings = 8) :
  (flavors * choose toppings 3) = 280 :=
by
  rw [h_flavors, h_toppings]
  exact calculate_combinations

def calculate_combinations : 5 * choose 8 3 = 280 := sorry

end smoothie_combinations_l470_470888


namespace unrelated_statement_l470_470895

-- Definitions
def timely_snow_promises_harvest : Prop := true -- assumes it has a related factor
def upper_beam_not_straight_lower_beam_crooked : Prop := true -- assumes it has a related factor
def smoking_harmful_to_health : Prop := true -- assumes it has a related factor
def magpies_signify_joy_crows_signify_mourning : Prop := false -- does not have an inevitable relationship

-- Theorem
theorem unrelated_statement :
  ¬magpies_signify_joy_crows_signify_mourning :=
by 
  -- proof to be provided
  sorry

end unrelated_statement_l470_470895


namespace sum_odd_divisors_360_is_78_l470_470464

/-- Define what it means to be an odd divisor of 360 --/
def is_odd_divisor (n : ℕ) : Prop :=
  n ∣ 360 ∧ n % 2 = 1

/-- Define the sum of all odd divisors of 360 --/
def sum_odd_divisors_360 : ℕ :=
  ∑ d in (Finset.filter is_odd_divisor (Finset.range 361)), d

/-- Prove that the sum of all odd divisors of 360 is 78 --/
theorem sum_odd_divisors_360_is_78 : sum_odd_divisors_360 = 78 := by
  sorry

end sum_odd_divisors_360_is_78_l470_470464


namespace solution_l470_470105

noncomputable def smallest_positive_angle : ℝ :=
  Classical.choose (exists_unique_x (3 / 2))

theorem solution :
  12 * (real.sin smallest_positive_angle)^3 * (real.cos smallest_positive_angle)^2
  - 12 * (real.sin smallest_positive_angle)^2 * (real.cos smallest_positive_angle)^3 = 3 / 2
  ∧ smallest_positive_angle > 0
  ∧ smallest_positive_angle < 90
  ∧ smallest_positive_angle = 7.5 :=
sorry

end solution_l470_470105


namespace number_of_people_l470_470766

theorem number_of_people 
  (g b : ℕ)
  (h1 : g = 7 + 12)
  (h2 : 0.75 * b = 12) : 
  g + b = 35 := 
by
  sorry

end number_of_people_l470_470766


namespace functions_strictly_decreasing_on_positive_reals_l470_470649

open Function

def f1 (x : ℝ) : ℝ := 1 / x
def f2 (x : ℝ) : ℝ := -abs x
def f3 (x : ℝ) : ℝ := -2 * x - 1
def f4 (x : ℝ) : ℝ := (x - 1)^2

theorem functions_strictly_decreasing_on_positive_reals :
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f1 x1 > f1 x2) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f2 x1 > f2 x2) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f3 x1 > f3 x2) ∧
  ¬ (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f4 x1 > f4 x2) :=
by sorry

end functions_strictly_decreasing_on_positive_reals_l470_470649


namespace total_yellow_leaves_l470_470099

noncomputable def calculate_yellow_leaves (total : ℕ) (percent_brown : ℕ) (percent_green : ℕ) : ℕ :=
  let brown_leaves := (total * percent_brown + 50) / 100
  let green_leaves := (total * percent_green + 50) / 100
  total - (brown_leaves + green_leaves)

theorem total_yellow_leaves :
  let t_yellow := calculate_yellow_leaves 15 25 40
  let f_yellow := calculate_yellow_leaves 22 30 20
  let s_yellow := calculate_yellow_leaves 30 15 50
  t_yellow + f_yellow + s_yellow = 26 :=
by
  sorry

end total_yellow_leaves_l470_470099


namespace yolks_in_carton_l470_470875

/-- A local farm is famous for having lots of double yolks in their eggs. One carton of 12 eggs had five eggs with double yolks. Prove that the total number of yolks in the whole carton is equal to 17. -/
theorem yolks_in_carton (total_eggs : ℕ) (double_yolk_eggs : ℕ) (single_yolk_per_egg : ℕ) (double_yolk_per_egg : ℕ) 
    (total_eggs = 12) (double_yolk_eggs = 5) (single_yolk_per_egg = 1) (double_yolk_per_egg = 2) : 
    (double_yolk_eggs * double_yolk_per_egg + (total_eggs - double_yolk_eggs) * single_yolk_per_egg) = 17 := 
by
    sorry

end yolks_in_carton_l470_470875


namespace find_unit_vector_in_same_direction_l470_470173

noncomputable def point := (ℝ × ℝ)

noncomputable def vector (A B : point) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  if h : magnitude(v) ≠ 0 then
    (v.1 / magnitude(v), v.2 / magnitude(v))
  else
    (0, 0)  -- By convention when magnitude is zero

theorem find_unit_vector_in_same_direction :
  unit_vector (vector (1, 3) (4, -1)) = (3/5, -4/5) :=
by
  sorry

end find_unit_vector_in_same_direction_l470_470173


namespace train_speed_km_per_hr_l470_470894

theorem train_speed_km_per_hr:
  ∀ (length_train length_bridge time_secs : ℕ), 
  length_train = 110 →
  length_bridge = 265 →
  time_secs = 30 →
  let total_distance := length_train + length_bridge in
  let speed_m_per_s := total_distance / time_secs in
  let speed_km_per_hr := speed_m_per_s * 3600 / 1000 in
  speed_km_per_hr = 45 :=
by
  intros length_train length_bridge time_secs h_train h_bridge h_time
  rw [h_train, h_bridge, h_time]
  let total_distance := 110 + 265
  let speed_m_per_s := total_distance / 30
  have h_speed_m_per_s: speed_m_per_s = 12.5 := by sorry
  let speed_km_per_hr := speed_m_per_s * 3600 / 1000
  have h_conversion: speed_km_per_hr = 45 := by sorry
  exact h_conversion

end train_speed_km_per_hr_l470_470894


namespace winning_votes_l470_470710

variable (V : ℕ) -- total number of votes in the election
variable (W : ℕ) -- total number of votes for the winning candidate
variable (M : ℕ) -- margin of victory

-- We know the winning candidate received 58% of the votes
axiom percent_win : W = 58 * V / 100

-- And the margin of victory is 1,200 votes
axiom margin : M = 1200

theorem winning_votes : W = 58 * V / 100 :=
by simp [percent_win, margin]; sorry

end winning_votes_l470_470710


namespace maxValue_computation_l470_470291

open Complex

noncomputable def maxValue (z : ℂ) (hz : abs z = Real.sqrt 3) : ℝ :=
  abs ((z - 2)^2 * (z + 2))

theorem maxValue_computation : ∀ (z : ℂ), abs z = Real.sqrt 3 → maxValue z (abs_eq _ _) = Real.sqrt 637 :=
by
  intro z hz
  sorry

end maxValue_computation_l470_470291


namespace cost_of_paving_floor_l470_470006

-- Define the constants given in the problem
def length1 : ℝ := 5.5
def width1 : ℝ := 3.75
def length2 : ℝ := 4
def width2 : ℝ := 3
def cost_per_sq_meter : ℝ := 800

-- Define the areas of the two rectangles
def area1 : ℝ := length1 * width1
def area2 : ℝ := length2 * width2

-- Define the total area of the floor
def total_area : ℝ := area1 + area2

-- Define the total cost of paving the floor
def total_cost : ℝ := total_area * cost_per_sq_meter

-- The statement to prove: the total cost equals 26100 Rs
theorem cost_of_paving_floor : total_cost = 26100 := by
  -- Proof skipped
  sorry

end cost_of_paving_floor_l470_470006


namespace max_magnitude_γ_eq_13_half_l470_470646

noncomputable def maximum_magnitude_γ (α β γ : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖α‖ = 1)
  (h2 : ‖β‖ = 1)
  (h3 : α ⬝ β = 0)
  (h4 : (5 • α - 2 • γ) ⬝ (12 • β - 2 • γ) = 0) : ℝ :=
  if h : γ ≠ 0 then
    dist (0 : EuclideanSpace ℝ (Fin 2)) γ + (13/4)
  else
    0

-- Theorem statement
theorem max_magnitude_γ_eq_13_half (α β γ : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖α‖ = 1)
  (h2 : ‖β‖ = 1)
  (h3 : α ⬝ β = 0)
  (h4 : (5 • α - 2 • γ) ⬝ (12 • β - 2 • γ) = 0) 
  : maximum_magnitude_γ α β γ h1 h2 h3 h4 = 13 / 2 :=
sorry

end max_magnitude_γ_eq_13_half_l470_470646


namespace c_20_equals_3_pow_4181_l470_470569

def c : ℕ → ℕ
| 0       := 0  -- To handle the 0-indexing naturally
| 1       := 3
| 2       := 3^2
| (n + 3) := c (n + 2) * c (n + 1)

theorem c_20_equals_3_pow_4181 : c 20 = 3^4181 := by
  sorry

end c_20_equals_3_pow_4181_l470_470569


namespace alpha_in_fourth_quadrant_l470_470968

def point_in_third_quadrant (α : ℝ) : Prop :=
  (Real.tan α < 0) ∧ (Real.sin α < 0)

theorem alpha_in_fourth_quadrant (α : ℝ) (h : point_in_third_quadrant α) : 
  α ∈ Set.Ioc (3 * Real.pi / 2) (2 * Real.pi) :=
by sorry

end alpha_in_fourth_quadrant_l470_470968


namespace P_B_and_C_P_A_XOR_C_P_A_and_B_and_C_l470_470298

variables (A B C : Prop)
variables (P_A P_B P_C P_A_and_B P_A_or_B : ℝ)
variables (independent : Prop)
variables (h1 : P_A = 0.4) 
variables (h2 : P_A_and_B = 0.25) 
variables (h3 : P_A_or_B = 0.6) 
variables (h4 : P_C = 0.55)
variables (h5 : ¬ independent)

noncomputable def P_B : ℝ := P_A_or_B - P_A + P_A_and_B

theorem P_B_and_C (h : P_B = 0.45) : P_B * P_C = 0.2475 := by
  sorry

theorem P_A_XOR_C (h : P_A * P_C = 0.22) : P_A + P_C - 2 * (P_A * P_C) = 0.51 := by
  sorry

theorem P_A_and_B_and_C : P_A_and_B * P_C = 0.1375 := by
  sorry

end P_B_and_C_P_A_XOR_C_P_A_and_B_and_C_l470_470298


namespace compare_abc_l470_470632

noncomputable def a : ℝ := 2 ^ (-2)
noncomputable def b : ℝ := 3 ^ (1/2)
noncomputable def c : ℝ := Real.log 5 / Real.log 2 -- Using natural logarithm for base change

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l470_470632


namespace point_equidistant_l470_470735

def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def P : ℝ × ℝ × ℝ := (5, -3, 4)

theorem point_equidistant : dist A P = dist B P ∧ dist B P = dist C P ∧ dist C P = dist D P :=
by
  sorry

end point_equidistant_l470_470735


namespace parabola_vertex_on_x_axis_l470_470698

def parabola_vertex_y_coord (a b c : ℝ) : ℝ :=
  (b ^ 2 - 4 * a * c) / (4 * a)

theorem parabola_vertex_on_x_axis (c : ℝ) :
  parabola_vertex_y_coord 1 (-4) c = 0 → c = 4 :=
by
  sorry

end parabola_vertex_on_x_axis_l470_470698


namespace seven_pow_350_mod_43_l470_470057

theorem seven_pow_350_mod_43 :
  (7^350) % 43 = 6 :=
by
  -- Using the provided conditions
  have h1 : 7 % 43 = 7, from rfl,
  have h2 : (7^2) % 43 = 6, by norm_num1,
  have h3 : (7^3) % 43 = 42 % 43 := by norm_num1,
  have h4 : (7^4) % 43 = 36, by norm_num1,
  -- Proceed with the proof using these
  sorry

end seven_pow_350_mod_43_l470_470057


namespace sqrt_x_div_sqrt_y_as_fraction_l470_470928

theorem sqrt_x_div_sqrt_y_as_fraction 
  (x y : ℝ)
  (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = 54 * x / 115 * y * ((1/5)^2 + (1/7)^2 + (1/8)^2)) : 
  (Real.sqrt x) / (Real.sqrt y) = 49 / 29 :=
by
  sorry

end sqrt_x_div_sqrt_y_as_fraction_l470_470928


namespace triangle_sides_consecutive_and_area_84_l470_470593

noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_sides_consecutive_and_area_84 :
  ∃ (x : ℝ), 
    herons_formula (x-1) x (x+1) = 84 ∧
    [x-1, x, x+1] = [13, 14, 15] :=
sorry

end triangle_sides_consecutive_and_area_84_l470_470593


namespace time_b_started_walking_l470_470080

/-- A's speed is 7 kmph, B's speed is 7.555555555555555 kmph, and B overtakes A after 1.8 hours. -/
theorem time_b_started_walking (t : ℝ) (A_speed : ℝ) (B_speed : ℝ) (overtake_time : ℝ)
    (hA : A_speed = 7) (hB : B_speed = 7.555555555555555) (hOvertake : overtake_time = 1.8) 
    (distance_A : ℝ) (distance_B : ℝ)
    (hDistanceA : distance_A = (t + overtake_time) * A_speed)
    (hDistanceB : distance_B = B_speed * overtake_time) :
  t = 8.57 / 60 := by
  sorry

end time_b_started_walking_l470_470080


namespace number_of_ways_to_fill_l470_470532

-- Definitions and conditions
def triangular_array (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the triangular array structure
  sorry 

def sum_based (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the sum-based condition
  sorry 

def valid_filling (x : Fin 13 → ℕ) :=
  (∀ i, x i = 0 ∨ x i = 1) ∧
  (x 0 + x 12) % 5 = 0

theorem number_of_ways_to_fill (x : Fin 13 → ℕ) :
  triangular_array 13 1 → sum_based 13 1 →
  valid_filling x → 
  (∃ (count : ℕ), count = 4096) :=
sorry

end number_of_ways_to_fill_l470_470532


namespace sin_2x_value_l470_470652

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

theorem sin_2x_value (x : ℝ) (h1 : f x = 5 / 3) (h2 : -Real.pi / 6 < x) (h3 : x < Real.pi / 6) :
  Real.sin (2 * x) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6 := 
sorry

end sin_2x_value_l470_470652


namespace cost_price_approx_l470_470896

noncomputable def cost_price (M : ℝ) : ℝ := (M * 0.91) / 1.25

theorem cost_price_approx (M : ℝ) (hM : M ≈ 65.25) : cost_price M ≈ 47.50 :=
by
  sorry

end cost_price_approx_l470_470896


namespace chessboard_tiling_possible_l470_470565

-- Define the concepts related to the chessboard and tiling problem
def cell := (ℕ × ℕ)  -- A cell is defined by its row and column coordinates
def chessboard := fin 8 × fin 8  -- An 8x8 chessboard with Finite indices

-- Definition for a tile consisting of 3 cells
def tile := (cell × cell × cell)

-- Predicate to check if a tile fits within the bounds of the 8x8 chessboard
def tile_within_bounds (t : tile) : Prop :=
  ∀ (c : cell), c = t.1 ∨ c = t.2 ∨ c = t.3 → c.1 < 8 ∧ c.2 < 8

-- Definition that one cell is removed from the chessboard
def chessboard_with_one_removed (r c : fin 8) : set cell :=
  { p | p.1 < 8 ∧ p.2 < 8 ∧ ¬(p.1 = r ∧ p.2 = c) }

-- Predicate to validate whether the remaining cells can be fully covered by 1x3 tiles
def can_be_tiled (board : set cell) : Prop :=
  ∃ (t : finset tile), (∀ (t ∈ t), tile_within_bounds t) ∧ 
  ∀ (c : cell), board c ↔ ∃ (t ∈ t), c = t.1 ∨ c = t.2 ∨ c = t.3

-- Main theorem statement
theorem chessboard_tiling_possible (r c : fin 8) : 
  ∃ (tile_plan : finset tile), can_be_tiled (chessboard_with_one_removed r c) :=
sorry  -- Proof is not required

end chessboard_tiling_possible_l470_470565


namespace baron_minchausen_lied_l470_470096

open FiniteGraph

def baron_minchausen_graph (G : Type) [Graph G] : Prop :=
  (Connected G) ∧
  (∀ cycle, is_cycle G cycle → even (length cycle)) ∧
  (∃ v w, v ≠ w ∧ degree v ≠ degree w ∧ (∀ u, (u ≠ v ∧ u ≠ w) → degree u = degree v))

theorem baron_minchausen_lied (G : Type) [Graph G] :
  baron_minchausen_graph G → false :=
begin
  intros h,
  sorry,
end

end baron_minchausen_lied_l470_470096


namespace distinct_triplets_count_l470_470049

theorem distinct_triplets_count :
  ∃ (count : ℕ), count = 440 ∧ ∀ n, n ≤ 600 →
  (⟦ n / 2 ⟧ ≠ ⟦ (n+1) / 2 ⟧ ∨ ⟦ n / 3 ⟧ ≠ ⟦ (n+1) / 3 ⟧ ∨ ⟦ n / 5 ⟧ ≠ ⟦ (n+1) / 5 ⟧) :=
begin
  sorry
end

end distinct_triplets_count_l470_470049


namespace upper_side_length_trapezoid_l470_470470

theorem upper_side_length_trapezoid
  (L U : ℝ) 
  (h : ℝ := 8) 
  (A : ℝ := 72) 
  (cond1 : U = L - 6)
  (cond2 : 1/2 * (L + U) * h = A) :
  U = 6 := 
by 
  sorry

end upper_side_length_trapezoid_l470_470470


namespace count_integers_between_powers_l470_470214

noncomputable def power (a : ℝ) (b : ℝ) : ℝ := a^b

theorem count_integers_between_powers:
  let a := 10
  let b1 := 0.1
  let b2 := 0.4
  have exp1 : Float := (a + b1)
  have exp2 : Float := (a + b2)
  have n1 : ℤ := exp1^3.ceil
  have n2 : ℤ := exp2^3.floor
  n2 - n1 + 1 = 94 := 
begin
  sorry
end

end count_integers_between_powers_l470_470214


namespace inequality_holds_for_all_reals_l470_470344

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470344


namespace negation_of_universal_proposition_l470_470786

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by
  sorry

end negation_of_universal_proposition_l470_470786


namespace area_of_sector_l470_470237

theorem area_of_sector (L θ : ℝ) (hL : L = 4) (hθ : θ = 2) : 
  (1 / 2) * ((L / θ) ^ 2) * θ = 4 := by
  sorry

end area_of_sector_l470_470237


namespace count_integers_between_cubes_l470_470216

noncomputable def a := (10.1)^3
noncomputable def b := (10.4)^3

theorem count_integers_between_cubes : 
  ∃ (count : ℕ), count = 94 ∧ (1030.031 < a) ∧ (a < b) ∧ (b < 1124.864) := 
  sorry

end count_integers_between_cubes_l470_470216


namespace limit_of_function_square_l470_470226

theorem limit_of_function_square (f : ℝ → ℝ) (h : ∀ x, f x = x^2) :
  (∃ l, Tendsto (fun Δx => (f (-1 + Δx) - f (-1)) / Δx) (𝓝 0) (𝓝 l) ∧ l = -2) :=
by
  -- Proof would be placed here
  sorry

end limit_of_function_square_l470_470226


namespace distance_DE_is_6_75_l470_470445

theorem distance_DE_is_6_75 :
  ∃ (A B C D E P : Point ℝ),
    B = Point.mk 0 0 ∧ 
    C = Point.mk 15 0 ∧
    (distance B A = 12) ∧ 
    (distance B C = 15) ∧
    (distance A C = 18) ∧
    P ∈ segment A C ∧
    (distance P C = 12) ∧
    D ∈ line_through B P ∧
    E ∈ line_through B P ∧
    is_trapezoid A B C D ∧
    is_trapezoid A B C E →
    distance D E = 6.75 :=
begin
    sorry -- proof to be completed
end

end distance_DE_is_6_75_l470_470445


namespace two_people_work_time_l470_470274

theorem two_people_work_time (time_one_person : ℕ)
    (num_people : ℕ)
    (h1 : time_one_person = 10)
    (h2 : num_people = 2)
    (equally_skilled : true) : 
    let time_two_people := time_one_person / num_people in
    time_two_people = 5 := by 
    -- According to the given conditions
    sorry

end two_people_work_time_l470_470274


namespace average_annual_reduction_10_percent_l470_470510

theorem average_annual_reduction_10_percent :
  ∀ x : ℝ, (1 - x) ^ 2 = 1 - 0.19 → x = 0.1 :=
by
  intros x h
  -- Proof to be filled in
  sorry

end average_annual_reduction_10_percent_l470_470510


namespace radar_placement_and_coverage_area_l470_470805

theorem radar_placement_and_coverage_area (r : ℝ) (w : ℝ) (n : ℕ) (h_radars : n = 5) (h_radius : r = 13) (h_width : w = 10) :
  let max_dist := 12 / Real.sin (Real.pi / 5)
  let area_ring := (240 * Real.pi) / Real.tan (Real.pi / 5)
  max_dist = 12 / Real.sin (Real.pi / 5) ∧ area_ring = (240 * Real.pi) / Real.tan (Real.pi / 5) :=
by
  sorry

end radar_placement_and_coverage_area_l470_470805


namespace angle_in_second_quadrant_l470_470236

theorem angle_in_second_quadrant (α : ℝ) (h : α = 2) : (Real.pi / 2 < α ∧ α < Real.pi) :=
by 
  rw [h]
  exact ⟨by norm_num, by norm_num⟩

end angle_in_second_quadrant_l470_470236


namespace sum_of_valid_four_digit_numbers_l470_470938

-- Define the set of digits to be used
def valid_digits := {1, 2, 3, 6, 7, 8}

-- Define a function that lists all valid four-digit numbers
def valid_four_digit_numbers (digits : Set Nat) : List Nat :=
  (digits.to_list.product (digits.to_list.product (digits.to_list.product digits.to_list))).map
    (fun ((a, b), (c, d)) => a * 1000 + b * 100 + c * 10 + d)

-- Prove the sum of all valid four-digit numbers is 6479352
theorem sum_of_valid_four_digit_numbers :
  valid_four_digit_numbers valid_digits.sum = 6479352 :=
by
  sorry

end sum_of_valid_four_digit_numbers_l470_470938


namespace pentagon_coloring_valid_l470_470258

-- Define the colors
inductive Color
| Red
| Blue

-- Define the vertices as a type
inductive Vertex
| A | B | C | D | E

open Vertex Color

-- Define an edge as a pair of vertices
def Edge := Vertex × Vertex

-- Define the coloring function
def color : Edge → Color := sorry

-- Define the pentagon
def pentagon_edges : List Edge :=
  [(A, B), (B, C), (C, D), (D, E), (E, A), (A, C), (A, D), (A, E), (B, D), (B, E), (C, E)]

-- Define the condition for a valid triangle coloring
def valid_triangle_coloring (e1 e2 e3 : Edge) : Prop :=
  (color e1 = Red ∧ (color e2 = Blue ∨ color e3 = Blue)) ∨
  (color e2 = Red ∧ (color e1 = Blue ∨ color e3 = Blue)) ∨
  (color e3 = Red ∧ (color e1 = Blue ∨ color e2 = Blue))

-- Define the condition for all triangles formed by the vertices of the pentagon
def all_triangles_valid : Prop :=
  ∀ v1 v2 v3 : Vertex,
    v1 ≠ v2 → v2 ≠ v3 → v1 ≠ v3 →
    valid_triangle_coloring (v1, v2) (v2, v3) (v1, v3)

-- Statement: Prove that there are 12 valid ways to color the pentagon
theorem pentagon_coloring_valid : (∃ (coloring : Edge → Color), all_triangles_valid) :=
  sorry

end pentagon_coloring_valid_l470_470258


namespace interest_calculation_years_l470_470007

theorem interest_calculation_years
  (principal : ℤ) (rate : ℝ) (difference : ℤ) (n : ℤ)
  (h_principal : principal = 2400)
  (h_rate : rate = 0.04)
  (h_difference : difference = 1920)
  (h_equation : (principal : ℝ) * rate * n = principal - difference) :
  n = 5 := 
sorry

end interest_calculation_years_l470_470007


namespace final_points_l470_470577

-- Definitions of the points in each round
def first_round_points : Int := 16
def second_round_points : Int := 33
def last_round_points : Int := -48

-- The theorem to prove Emily's final points
theorem final_points :
  first_round_points + second_round_points + last_round_points = 1 :=
by
  sorry

end final_points_l470_470577


namespace tangent_line_at_2_is_20x_sub_y_sub_19_monotonicity_of_f_l470_470161

-- Statement for Part 1
theorem tangent_line_at_2_is_20x_sub_y_sub_19 {f : ℝ → ℝ} (a : ℝ) (h : a = 1) :
  f = (λ (x : ℝ), x^3 + a*x^2 + 4*x + 1) →
  f' 2 = 20 ∧
  f 2 = 21 →
  (∀ y : ℝ, y = 20*(2 : ℝ) - f 2) ∨ y = 20 → 
  20 * (2 : ℝ) - f 2 - 19 = 0 := sorry

-- Statement for Part 2
theorem monotonicity_of_f (a : ℝ):
  (f : ℝ → ℝ) →
  f = (λ (x : ℝ), x^3 + a*x^2 + 4*x + 1) →
  (∀ x : ℝ, x ∈ Ioi 0 → 
    (if -2*real.sqrt 3 ≤ a ∧ a < 0 
    then deriv f x > 0
    else if x ≤ root1 ∨ root2 ≤ x then deriv f x > 0 ∧ root1 < x ∧ x < root2 ∧ deriv f x < 0)) := sorry


end tangent_line_at_2_is_20x_sub_y_sub_19_monotonicity_of_f_l470_470161


namespace prove_asymptote_l470_470670

noncomputable def hyperbola := 
  { x : ℝ | ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x = λ t, ((t / a)^2 - (t / b)^2) = 19 ∧ eccentricity = sqrt(5) / 2 }

def is_asymptote (a b : ℝ) (y x : ℝ) :=
  y = b / a * x ∨ y = -(b / a) * x

theorem prove_asymptote (a b : ℝ) : 
  (∃ x y : ℝ, is_asymptote a b y x) → eccentricity a b = sqrt 5 / 2 → b / a = 1 / 2 :=
by
  sorry

end prove_asymptote_l470_470670


namespace volume_of_prism_length_of_CT_l470_470783

def radius : ℝ := (3 * Real.sqrt 5) / 2
def height : ℝ := 12
def base_area : ℝ := (Real.sqrt 3 / 4) * (3 * Real.sqrt 15) ^ 2
def volume : ℝ := base_area * height

theorem volume_of_prism (height : ℝ) (radius : ℝ) : volume = 405 * Real.sqrt 3 := by
  sorry

theorem length_of_CT (radius : ℝ) : ∃ T F : ℝ, (FT_parallel_BC : Bool) →
  (fa1c1_touches_sphere : Bool) → (ABT_touches_sphere : Bool) → 
  (CT = 9 ∨ CT = 3) := by
  sorry

end volume_of_prism_length_of_CT_l470_470783


namespace smallest_num_with_given_divisors_l470_470460
noncomputable def hasDivisors (n : ℕ) : Prop :=
  (nat.filter (λ d, d % 2 = 1) (nat.divisors n)).length = 8 ∧
  (nat.filter (λ d, d % 2 = 0) (nat.divisors n)).length = 10

theorem smallest_num_with_given_divisors : 
  ∃ (n : ℕ), hasDivisors n ∧ (∀ m : ℕ, hasDivisors m → n ≤ m) := 
begin
  use 53760,
  split,
  {
    -- condition for 8 odd divisors
    sorry,
  },
  {
    split,
    {
      -- condition for 10 even divisors
      sorry,
    },
    {
      -- minimality condition
      sorry,
    }
  }
end

end smallest_num_with_given_divisors_l470_470460


namespace shared_birthday_probability_l470_470457

theorem shared_birthday_probability :
  ( ∃ P, P = 1 - (finset.prod (finset.range 30) (λ i, (365 - i) / 365)) ) :=
by sorry

end shared_birthday_probability_l470_470457


namespace inequality_holds_for_real_numbers_l470_470320

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470320


namespace ratio_of_rooms_l470_470566

theorem ratio_of_rooms (rooms_danielle : ℕ) (rooms_grant : ℕ) (ratio_grant_heidi : ℚ)
  (h1 : rooms_danielle = 6)
  (h2 : rooms_grant = 2)
  (h3 : ratio_grant_heidi = 1/9) :
  (18 : ℚ) / rooms_danielle = 3 :=
by
  sorry

end ratio_of_rooms_l470_470566


namespace nearest_integer_x4_l470_470820

noncomputable def x : Real := 3 + Real.sqrt 2

theorem nearest_integer_x4 : Int.nearest (x^4) = 386 := by
  sorry

end nearest_integer_x4_l470_470820


namespace removal_of_4_achieves_average_6_3_l470_470467

theorem removal_of_4_achieves_average_6_3 : 
  let nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
  let num_to_remove := 4
  let remaining_numbers := nums.erase num_to_remove
  (remaining_numbers.sum / remaining_numbers.length : Float) = 6.3 :=
by
  sorry

end removal_of_4_achieves_average_6_3_l470_470467


namespace range_of_a_l470_470192

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ :=
  (m * x + n) / (x ^ 2 + 1)

example (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1) : 
  m = 2 ∧ n = 0 :=
sorry

theorem range_of_a (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1)
  (h_m : m = 2) (h_n : n = 0) {a : ℝ} : f (a-1) m n + f (a^2-1) m n < 0 ↔ 0 ≤ a ∧ a < 1 :=
sorry

end range_of_a_l470_470192


namespace pentagon_diagonal_sum_correct_final_answer_is_516_l470_470281

noncomputable def pentagon_diagonal_sum : ℝ :=
  let AB := 5
  let CD := 5
  let BC := 8
  let DE := 8
  let AE := 12
  let z := 15
  let x := (z^2 - 64) / 5
  let y := (z^2 - 25) / 8
  3 * z + x + y

theorem pentagon_diagonal_sum_correct :
  let AB := 5
  let CD := 5
  let BC := 8
  let DE := 8
  let AE := 12
  let z := 15
  let x := (z^2 - 64) / 5
  let y := (z^2 - 25) / 8
  (3 * z + x + y) = 102.2 :=
by 
  let AB := 5
  let CD := 5
  let BC := 8
  let DE := 8
  let AE := 12
  let z := 15
  let x := (z^2 - 64) / 5
  let y := (z^2 - 25) / 8
  show (3 * z + x + y) = 102.2
  sorry

theorem final_answer_is_516 : 511 + 5 = 516 := 
by 
  show 511 + 5 = 516
  sorry

end pentagon_diagonal_sum_correct_final_answer_is_516_l470_470281


namespace elsa_data_usage_l470_470127

theorem elsa_data_usage (D : ℝ) 
  (h_condition : D - 300 - (2/5) * (D - 300) = 120) : D = 500 := 
sorry

end elsa_data_usage_l470_470127


namespace monotonic_intervals_of_f_range_of_m_l470_470654

-- Definition of the function f(x) and g(x)
def f (x : ℝ) : ℝ := exp x / x
def g (x : ℝ) (m : ℝ) : ℝ := m * x

-- Statement of the monotonic intervals of f(x)
theorem monotonic_intervals_of_f :
  (∀ x : ℝ, x ∈ Ioo (-∞ : ℝ) 1 → deriv f x < 0) ∧ (∀ x : ℝ, x ∈ Ioi 1 → deriv f x > 0) :=
sorry

-- Statement of the range of m for the inequality f(x) + g(x) > 0 for all x in (0, +∞)
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ∈ Ioi 0 → f x + g x m > 0) ↔ m > - (exp 2 / 4) :=
sorry

end monotonic_intervals_of_f_range_of_m_l470_470654


namespace net_population_increase_l470_470840

-- Definitions for initial conditions
def birth_rate_per_two_seconds : ℕ := 8
def death_rate_per_two_seconds : ℕ := 6
def seconds_per_two_seconds : ℕ := 2
def seconds_per_day : ℕ := 24 * 60 * 60

-- Main statement
theorem net_population_increase (birth_rate_per_two_seconds death_rate_per_two_seconds seconds_per_two_seconds seconds_per_day : ℕ) : 
  let net_increase_per_two_seconds := birth_rate_per_two_seconds - death_rate_per_two_seconds in
  let net_increase_per_second := net_increase_per_two_seconds / seconds_per_two_seconds in
  net_increase_per_second * seconds_per_day = 86400 :=
by 
  sorry

end net_population_increase_l470_470840


namespace normal_distribution_probability_l470_470198

noncomputable def X : Type := sorry

-- Given conditions
def X_normal_dist : Prop :=
  ∀ (X : real), (X ∼ N(1,4))

def reference_data_1 : Prop :=
  P (1 - 2 < X ≤ 1 + 2) = 0.6826

def reference_data_2 : Prop :=
  P (1 - 2*2 < X ≤ 1 + 2*2) = 0.9544

-- Proof statement
theorem normal_distribution_probability :
  X_normal_dist →
  reference_data_1 →
  reference_data_2 →
  P(-3 < X < 1) = 0.4772 := by
  sorry

end normal_distribution_probability_l470_470198


namespace hyperbola_eccentricity_range_l470_470516

noncomputable def range_of_eccentricity : Set ℝ :=
  {e : ℝ | e > 2}

theorem hyperbola_eccentricity_range
  (F A B M : ℝ → ℝ)
  (hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (real_axis : ∀ (x y : ℝ), y = 0)
  (perpendicular : ∀ (x y : ℝ), x = F x)
  (line_intersect : ∀ (x y : ℝ), hyperbola x y ∧ perpendicular x y)
  (left_vertex_inside_circle : ∃ (k : ℝ), M k ∈ set.prod (set.Ioo (A k) (B k)) (set.Ioo (A k) (B k))) :
  ∀ e : ℝ, e ∈ range_of_eccentricity :=
sorry

end hyperbola_eccentricity_range_l470_470516


namespace total_boat_licenses_l470_470891

/-- A state modifies its boat license requirements to include any one of the letters A, M, or S
followed by any six digits. How many different boat licenses can now be issued? -/
theorem total_boat_licenses : 
  let letters := 3
  let digits := 10
  letters * digits^6 = 3000000 := by
  sorry

end total_boat_licenses_l470_470891


namespace routes_in_grid_l470_470107

open Finset

theorem routes_in_grid (rows cols : ℕ) (h_rows : rows = 3) (h_cols : cols = 2) :
  (univ.filter (fun s => s.card = 2)).card = 10 :=
by
  sorry

end routes_in_grid_l470_470107


namespace debate_students_handshake_l470_470017

theorem debate_students_handshake 
    (S1 S2 S3 : ℕ)
    (h1 : S1 = 2 * S2)
    (h2 : S2 = S3 + 40)
    (h3 : S3 = 200) :
    S1 + S2 + S3 = 920 :=
by
  sorry

end debate_students_handshake_l470_470017


namespace quadratic_roots_imaginary_l470_470113

theorem quadratic_roots_imaginary :
  ∃ (a b c : ℝ), a = 1 ∧ b = -2 * Real.sqrt 5 ∧ c = 7 ∧
  (∀ (Δ : ℝ), Δ = b^2 - 4 * a * c → Δ < 0 → ∃ (x1 x2 : ℂ), x1 = Real.sqrt 5 + Complex.i * Real.sqrt 2 ∧ x2 = Real.sqrt 5 - Complex.i * Real.sqrt 2) :=
by
  let a := 1
  let b := -2 * Real.sqrt 5
  let c := 7
  use [a, b, c]
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  intros Δ hΔ
  use Δ
  split 
  { exact hΔ }
  intros hΔ_lt
  have h := hΔ_lt
  -- Proof omitted
  sorry

end quadratic_roots_imaginary_l470_470113


namespace tiles_per_row_proof_l470_470397

def area_square_room : ℝ := 400 -- in square feet
def tile_size : ℝ := 8 / 12 -- each tile is 8 inches, converted to feet (8/12 feet)

noncomputable def number_of_tiles_per_row : ℕ :=
  let side_length := Math.sqrt area_square_room in
  let side_length_in_inch := side_length * 12 in
  Nat.floor (side_length_in_inch / tile_size)

theorem tiles_per_row_proof :
  number_of_tiles_per_row = 30 := by
  sorry

end tiles_per_row_proof_l470_470397


namespace nearest_integer_x4_l470_470819

noncomputable def x : Real := 3 + Real.sqrt 2

theorem nearest_integer_x4 : Int.nearest (x^4) = 386 := by
  sorry

end nearest_integer_x4_l470_470819


namespace infinitely_many_not_sum_of_three_fourth_powers_l470_470378

theorem infinitely_many_not_sum_of_three_fourth_powers : ∀ n : ℕ, n > 0 → n ≡ 5 [MOD 16] → ¬(∃ a b c : ℤ, n = a^4 + b^4 + c^4) :=
by sorry

end infinitely_many_not_sum_of_three_fourth_powers_l470_470378


namespace max_distinct_planes_l470_470742

-- Define the two distinct planes α and β.
variables (α β : Type) [plane α] [plane β]

-- Define the sets of points on each plane.
constant points_on_plane_α : finset α
constant points_on_plane_β : finset β

-- Assume five points on plane α and seven points on plane β.
axiom five_points_on_plane_α : points_on_plane_α.card = 5
axiom seven_points_on_plane_β : points_on_plane_β.card = 7

-- Define the problem statement: proving the maximum number of distinct planes is 177.
theorem max_distinct_planes (hαβ : α ≠ β) : 
  max_planes points_on_plane_α points_on_plane_β = 177 :=
sorry

end max_distinct_planes_l470_470742


namespace petya_vasya_game_l470_470760

theorem petya_vasya_game :
  ∃ (k : ℕ), k = 84 ∧ ∀ (marks : Finset (ℕ × ℕ)), marks.card = k →
    ∀ (rect : Finset (ℕ × ℕ)), (rect.card = 6 ∧
    ∀ (rot : Bool), ∃ (c : ℕ × ℕ) (d : Bool), 
      (rot = d ∧ rect = (if d then (Finset.image (λ p : ℕ × ℕ, (p.snd, p.fst)) rect)
                        else (Finset.image (λ p : ℕ × ℕ, p) rect))) →
    (∃! r ∈ Finset.powerset (Finset.univ (Fin (13 * 13))), rect ⊆ r ∧ marks ∩ r = marks ∩ rect)) :=
by
  sorry

end petya_vasya_game_l470_470760


namespace vector_addition_l470_470956

variables (α : ℝ) (AB BC : ℝ × ℝ)

noncomputable def condition1 : Prop := 
  (sin α / (sin α + cos α) = 1 / 2)

noncomputable def AB_def : Prop :=
  AB = (tan α, 1)

noncomputable def BC_def : Prop := 
  BC = (tan α, 2)

noncomputable def AC : ℝ × ℝ := 
  (AB.1 + BC.1, AB.2 + BC.2)

theorem vector_addition (h1 : condition1 α) (h2 : AB_def α AB) (h3 : BC_def α BC):
  AC α AB BC = (2, 3) :=
by
  sorry

end vector_addition_l470_470956


namespace sum_of_center_coordinates_l470_470788

theorem sum_of_center_coordinates (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7) (h2 : y1 = -6) (h3 : x2 = -5) (h4 : y2 = 4) :
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
  -- Definitions and setup
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  sorry

end sum_of_center_coordinates_l470_470788


namespace homework_problems_l470_470097

theorem homework_problems (p t : ℕ)
  (h_rate : p > 15)
  (h_eq : p * t = (2 * p - 6) * (t - 3)) :
  p = 18 ∧ t = 7 ∧ p * t = 126 :=
by
  have h1 : (p - 6) * (t - 6) = 18, by sorry
  have hp : p = 18, by sorry
  have ht : t = 7, by sorry
  have h_prod : p * t = 126, by sorry
  exact ⟨hp, ht, h_prod⟩

end homework_problems_l470_470097


namespace fg_of_2_l470_470668

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

-- Prove the specific property
theorem fg_of_2 : f (g 2) = -19 := by
  -- Placeholder for the proof
  sorry

end fg_of_2_l470_470668


namespace smallest_angle_in_isosceles_trapezoid_l470_470711

theorem smallest_angle_in_isosceles_trapezoid :
  ∀ (a d : ℝ), 
  a < a + d ∧ a + d < a + 2d ∧ a + 2d < 140 ∧ (a + (a + d) + (a + 2d) + 140 = 360) → 
  a = 40 :=
by 
  intro a d,
  intro h,
  sorry

end smallest_angle_in_isosceles_trapezoid_l470_470711


namespace inequality_proof_l470_470625

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) (h_sum : a + b + c + d ≥ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := 
by 
  sorry

end inequality_proof_l470_470625


namespace perimeter_PSTU_equals_30_l470_470247

noncomputable def Point := (ℝ, ℝ)

structure Triangle :=
  (P Q R : Point)
  (PQ PR : ℝ)
  (QR : ℝ)
  (hPQ_PR : PQ = PR)
  (hPQ_value : PQ = 15)
  (hQR_value : QR = 14)

structure Parallelogram :=
  (P S T U : Point)
  (hST_parallel_PR : ∀ (k : ℝ), (S.1 - T.1) = k * (P.1 - R.1) ∧ (S.2 - T.2) = k * (P.2 - R.2))
  (hTU_parallel_PQ : ∀ (k : ℝ), (T.1 - U.1) = k * (P.1 - Q.1) ∧ (T.2 - U.2) = k * (P.2 - Q.2))

theorem perimeter_PSTU_equals_30 (P Q R S T U : Point)
  (hPQ_PR : PQ = PR)
  (hPQ_value : PQ = 15)
  (hQR_value : QR = 14)
  (hST_parallel_PR : ∀ (k : ℝ), (S.1 - T.1) = k * (P.1 - R.1) ∧ (S.2 - T.2) = k * (P.2 - R.2))
  (hTU_parallel_PQ : ∀ (k : ℝ), (T.1 - U.1) = k * (P.1 - Q.1) ∧ (T.2 - U.2) = k * (P.2 - Q.2)) :
  (2 * hPQ_value + 2 * QR) = 30 := by
  sorry

end perimeter_PSTU_equals_30_l470_470247


namespace minimum_value_a10_correct_l470_470730

noncomputable def minimum_a10 (A : Set ℕ) : ℕ :=
  let σ := λ S : Set ℕ, S.toFinite.sum id
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 : ℕ,
    a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧
    a7 < a8 ∧ a8 < a9 ∧ a9 < a10 ∧ a10 < a11 ∧
    A = {a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11} ∧
    (∀ n : ℕ, n ≤ 1500 → ∃ S ⊆ A, σ S = n) ∧
    a10 = 248

theorem minimum_value_a10_correct {A : Set ℕ} : minimum_a10 A := 
by sorry

end minimum_value_a10_correct_l470_470730


namespace three_mul_distrib_not_maybe_equal_l470_470886

variable (a b : ℝ)

#check ∀ a b : ℝ, 3 * (a + b) = 3 * a + b

theorem three_mul_distrib (a b : ℝ) : 3 * (a + b) = 3 * a + 3 * b :=
by ring

theorem not_maybe_equal (a b : ℝ) : (3 * (a + b) = 3 * a + b) ↔ (b = 0) :=
by
  split
  . intro h
    have : 3 * (a + b) = 3 * a + 3 * b := by apply three_mul_distrib
    rw [h] at this
    linarith
  . intro h
    rw [h]
    ring

end three_mul_distrib_not_maybe_equal_l470_470886


namespace monotonicity_of_f_when_a_is_2_range_of_a_l470_470655

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x + 1) + 0.5 * a * x^2 - x

-- For part (I)
theorem monotonicity_of_f_when_a_is_2 :
  ∀ x : ℝ, 
  if x ∈ Ioo (-1:ℝ) (-0.5:ℝ) ∪ Ioo (0:ℝ) (⊤:ℝ) then 
    f 2 x > f 2 (x - 1)
  else if x ∈ Ioo (-0.5:ℝ) (0:ℝ) then 
    f 2 x < f 2 (x - 1)
  else 
    true :=
by 
  sorry

-- For part (II)
theorem range_of_a (x : ℝ) :
  (∀ x > 0, f a x ≥ f 1 x) ↔ (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end monotonicity_of_f_when_a_is_2_range_of_a_l470_470655


namespace find_a_l470_470589

def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋ 

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ P > 0, ∀ x, f (x + P) = f x

theorem find_a (f : ℝ → ℝ) (h : ∀ x, f x = fractional_part (a * x + Real.sin x)) :
  is_periodic f → ∃ r : ℚ, a = r / Real.pi :=
sorry

end find_a_l470_470589


namespace committee_count_l470_470864

-- Definitions based on conditions
def num_males := 15
def num_females := 10

-- Define the binomial coefficient
def binomial (n k : ℕ) := Nat.choose n k

-- Define the total number of committees
def num_committees_with_at_least_two_females : ℕ :=
  binomial num_females 2 * binomial num_males 3 +
  binomial num_females 3 * binomial num_males 2 +
  binomial num_females 4 * binomial num_males 1 +
  binomial num_females 5 * binomial num_males 0

theorem committee_count : num_committees_with_at_least_two_females = 36477 :=
by {
  sorry
}

end committee_count_l470_470864


namespace math_problem_l470_470315

noncomputable def proof_problem (r R α β γ : ℝ) := 
  α + β + γ = Real.pi ∧ 
  r = 4 * R * Real.sin(α / 2) * Real.sin(β / 2) * Real.sin(γ / 2) → 
  r / R ≤ 2 * Real.sin(α / 2) * (1 - Real.sin(α / 2))

theorem math_problem (r R α β γ : ℝ) (h1 : α + β + γ = Real.pi) 
(h2 : r = 4 * R * Real.sin(α / 2) * Real.sin(β / 2) * Real.sin(γ / 2)) : 
  r / R ≤ 2 * Real.sin(α / 2) * (1 - Real.sin(α / 2)) := 
by 
  sorry

end math_problem_l470_470315


namespace arithmetic_sequence_proof_l470_470256

variable {a : ℕ → ℝ}

-- Condition of the problem
def sequence_condition : Prop := a 13 + a 5 = 32

-- Target equation to prove
def target_proof : Prop := a 9 = 16

theorem arithmetic_sequence_proof (h : sequence_condition) : target_proof := by
  sorry

end arithmetic_sequence_proof_l470_470256


namespace min_positive_numbers_l470_470439

theorem min_positive_numbers (n : ℕ) (numbers : ℕ → ℤ) 
  (h_length : n = 103) 
  (h_consecutive : ∀ i : ℕ, i < n → (∃ (p1 p2 : ℕ), p1 < 5 ∧ p2 < 5 ∧ p1 ≠ p2 ∧ numbers (i + p1) > 0 ∧ numbers (i + p2) > 0)) :
  ∃ (min_positive : ℕ), min_positive = 42 :=
by
  sorry

end min_positive_numbers_l470_470439


namespace maximum_triangle_area_l470_470982

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h_eccentricity : (1 : ℝ) / 2 = 1 / a) : Prop :=
  (a = 2) ∧ (b = real.sqrt 3) ∧ (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)

theorem maximum_triangle_area 
  (F G A B : ℝ × ℝ) 
  (h_F : F = (1, 0)) 
  (h_G : G = (-1, 0))
  (h_AB_on_line : ∃ m : ℝ, ∀ y : ℝ, A = (m * y + 1, y) ∧ B = (m * y + 1, -y))
  : ∃ area : ℝ, area ≤ 3 := sorry


end maximum_triangle_area_l470_470982


namespace vector_dot_product_l470_470957

variable (AB AC : ℝ × ℝ)
variable t : ℝ
variable (h1 : AB = (2, 3))
variable (h2 : AC = (3, t))
variable (h3 : ∥AC - AB∥ = 1)

theorem vector_dot_product : AB ⋅ (AC - AB) = 2 :=
by
  sorry

end vector_dot_product_l470_470957


namespace eggs_per_hen_l470_470926

theorem eggs_per_hen (total_eggs : Float) (num_hens : Float) (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) : 
  total_eggs / num_hens = 10.821428571428571 :=
by 
  sorry

end eggs_per_hen_l470_470926


namespace invariant_final_number_l470_470441

theorem invariant_final_number (n : ℕ) (a : Fin n → ℝ) (h : 2 ≤ n) :
  ∀ steps : List (Fin n × Fin n),
  ∃ last_number : ℝ, last_number = (∑ i, (a i)⁻¹)⁻¹ :=
by sorry

end invariant_final_number_l470_470441


namespace perpendicular_condition_l470_470158

theorem perpendicular_condition (α β : Plane) (l : Line) (h1 : α ≠ β) (h2 : l ∈ α) :
  (α ⊥ β → l ⊥ β) ∧ (l ⊥ β → α ⊥ β) = false :=
sorry

end perpendicular_condition_l470_470158


namespace division_of_neg6_by_3_l470_470554

theorem division_of_neg6_by_3 : (-6 : ℤ) / 3 = -2 := 
by
  sorry

end division_of_neg6_by_3_l470_470554


namespace smallest_period_for_given_point_l470_470726

def f (a b x : ℝ) : ℝ := a * Real.cos (2 * x) + b * Real.sin (2 * x)

theorem smallest_period_for_given_point :
  ∃ T > 0, ∀ x : ℝ, f 1 (Real.sqrt 3) (x + T) = f 1 (Real.sqrt 3) x ∧
  ∀ T' > 0, (∀ x : ℝ, f 1 (Real.sqrt 3) (x + T') = f 1 (Real.sqrt 3) x) → T ≤ T' :=
sorry

end smallest_period_for_given_point_l470_470726


namespace patrol_final_position_total_electricity_consumed_l470_470062

section
variables (mileage : List ℝ)
#eval mileage

theorem patrol_final_position {l : List ℝ} (h: l = [-6, -2, 8, -3, 6, -4, 6, 3]) :
  let total_distance := l.sum
  in total_distance = 8 := by
  simp [h]; norm_num; sorry

theorem total_electricity_consumed {l : List ℝ} (h: l = [-6, -2, 8, -3, 6, -4, 6, 3]) :
  let abs_distance := l.map (λ x, abs x)
      total_distance := abs_distance.sum
      consumption_rate := 0.15
  in consumption_rate * total_distance = 5.7 := by
  simp [h]; norm_num; sorry
end

end patrol_final_position_total_electricity_consumed_l470_470062


namespace car_travel_distance_l470_470860

-- Define the conditions
def speed : ℝ := 23
def time : ℝ := 3

-- Define the formula for distance
def distance_traveled (s : ℝ) (t : ℝ) : ℝ := s * t

-- State the theorem to prove the distance the car traveled
theorem car_travel_distance : distance_traveled speed time = 69 :=
by
  -- The proof would normally go here, but we're skipping it as per the instructions
  sorry

end car_travel_distance_l470_470860


namespace correct_scatter_plot_axes_l470_470825

-- Definitions
def forecast_variable : Type := sorry
def explanatory_variable : Type := sorry

-- Condition: The forecast variable is analogous to the function value.
def forecast_is_function_value {f : forecast_variable -> explanatory_variable} : Prop := sorry

-- Condition: The explanatory variable is analogous to the independent variable.
def explanatory_is_independent_variable {f : explanatory_variable -> forecast_variable} : Prop := sorry

-- Theorem to be proved
theorem correct_scatter_plot_axes :
  forecast_is_function_value → explanatory_is_independent_variable →
  (∀ (x_axis_var : explanatory_variable) (y_axis_var : forecast_variable),
    y_axis_var = forecast_variable ∧ x_axis_var = explanatory_variable) :=
by
  intros
  sorry

end correct_scatter_plot_axes_l470_470825


namespace shortest_chord_length_l470_470429

-- Definitions for the problem
def circle_eqn (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - 2 = 0
  
def point_P := (0 : ℝ, 0 : ℝ)
def center_C := (1 : ℝ, -1 : ℝ)
def radius := 2

-- Definition of the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The theorem to prove: The shortest chord length cut by the line passing through point P on the circle is 2√2.
theorem shortest_chord_length : ∀ (x y : ℝ),
  circle_eqn x y → distance point_P center_C < radius →
  ∃ d : ℝ, d = 2 * real.sqrt 2 :=
sorry

end shortest_chord_length_l470_470429


namespace factorial_expression_simplification_l470_470104

theorem factorial_expression_simplification : (6! + 7!) / 5! = 48 :=
by
  sorry

end factorial_expression_simplification_l470_470104


namespace original_population_l470_470530

-- Define the conditions
def population_increase (n : ℕ) : ℕ := n + 1200
def population_decrease (p : ℕ) : ℕ := (89 * p) / 100
def final_population (n : ℕ) : ℕ := population_decrease (population_increase n)

-- Claim that needs to be proven
theorem original_population (n : ℕ) (H : final_population n = n - 32) : n = 10000 :=
by
  sorry

end original_population_l470_470530


namespace min_operations_to_flip_grid_l470_470713

theorem min_operations_to_flip_grid (n : ℕ) (h : n = 10) :
  ∃ (ops : ℕ), ops = 100 ∧ 
    (∀ (grid : matrix (fin n) (fin n) bool),
      (∀ i j, grid i j = ff) →
      (grid = λ i j, ff) →
      ops ≤ 100) := 
begin
  sorry,
end

end min_operations_to_flip_grid_l470_470713


namespace initial_stock_is_700_l470_470723

-- Define the initial condition where John sold a certain number of books each day from Monday to Friday.
def books_sold_monday : ℕ := 50
def books_sold_tuesday : ℕ := 82
def books_sold_wednesday : ℕ := 60
def books_sold_thursday : ℕ := 48
def books_sold_friday : ℕ := 40

-- Define the total number of books sold.
def total_books_sold : ℕ := books_sold_monday + books_sold_tuesday + books_sold_wednesday + books_sold_thursday + books_sold_friday

-- Define the percentage of books that were not sold.
def percent_not_sold : ℝ := 0.60

-- Define the proof problem that asserts the initial stock was 700.
theorem initial_stock_is_700 (H : total_books_sold = 280) (H2 : total_books_sold = (0.40 * 700) ∧ 0.40 * 700 = 280) : initial_stock = 700 :=
by 
    -- The Lean theorem proposes that given the total books sold and the not sold percentage, the initial stock was 700.
    sorry

end initial_stock_is_700_l470_470723


namespace luke_remaining_pieces_l470_470751

def pieces_left_after_n_days (initial_pieces : ℕ) (day1_percent : ℕ) (day2_percent : ℕ) (day3_percent : ℕ) (day4_percent : ℕ) (day5_percent : ℕ) : ℕ :=
  let day1_completed := initial_pieces * day1_percent / 100 in
  let after_day1 := initial_pieces - day1_completed in
  let day2_completed := after_day1 * day2_percent / 100 in
  let after_day2 := after_day1 - day2_completed in
  let day3_completed := after_day2 * day3_percent / 100 in
  let after_day3 := after_day2 - day3_completed in
  let day4_completed := after_day3 * day4_percent / 100 in
  let after_day4 := after_day3 - day4_completed in
  let day5_completed := after_day4 * day5_percent / 100 in
  after_day4 - day5_completed

theorem luke_remaining_pieces :
  pieces_left_after_n_days 2000 10 25 30 40 35 = 369 := by
  sorry

end luke_remaining_pieces_l470_470751


namespace range_of_t_l470_470984

theorem range_of_t (t : ℝ) : 
  (∀ x ∈ set.Ioo 1 3, 3 * x ^ 2 - 2 * t * x + 3 ≤ 0) → t ≥ 5 :=
sorry

end range_of_t_l470_470984


namespace ordered_pairs_count_l470_470153

theorem ordered_pairs_count : 
  (∃ (a b : ℝ), 
  (∃ (x y : ℤ), a * x + b * y = 1 ∧ x^2 + y^2 = 65)) → 
  32 := 
sorry

end ordered_pairs_count_l470_470153


namespace triangle_number_replacement_l470_470570

/-!
# Triangle Number Replacement Problem

Given there are nine smaller triangles, some labeled with numbers from {1, 2, 3, 4, 5, 6} 
and some with letters A, B, C, D, E, F, and the numbers in white triangles are the 
sums of numbers in adjacent gray triangles, we need to prove that:

(A, B, C, D, E, F) = (1, 3, 2, 5, 6, 4)
-/

theorem triangle_number_replacement :
  ∃ (A B C D E F : ℕ), 
    A + B + C + D + E + F = 1 + 2 + 3 + 4 + 5 + 6 ∧
    B + D + E = 14 ∧
    C + E + F = 12 ∧
    A ∈ {1, 2, 3, 4, 5, 6} ∧ 
    B ∈ {1, 2, 3, 4, 5, 6} ∧ 
    C ∈ {1, 2, 3, 4, 5, 6} ∧ 
    D ∈ {1, 2, 3, 4, 5, 6} ∧ 
    E ∈ {1, 2, 3, 4, 5, 6} ∧ 
    F ∈ {1, 2, 3, 4, 5, 6} ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ 
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 :=
by {
    sorry
}

end triangle_number_replacement_l470_470570


namespace sum_of_squares_pattern_l470_470648

theorem sum_of_squares_pattern (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sum_of_squares_pattern_l470_470648


namespace inequality_xyz_l470_470331

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470331


namespace probability_no_adjacent_rational_terms_l470_470260

theorem probability_no_adjacent_rational_terms :
  let x := ℝ
  let binomial_expansion := (√x + 1 / (2 * x^(1/6))) ^ 8
  (coeff_first := (√x) ^ 8) 
  (coeff_second := (8 * (√x)^7 * (1 / (2 * x^(1/6)))) = 4 * x^(20/6 - 1/3) )
  (coeff_third := ((8 * 7 / 2) * (√x)^6 * (1 / (2 * x^(1/6)))^2) = 14 / 8 * x^(4 - 2/3)) 
  (is_arithmetic_sequence := 2 * (4 * x^(20/6 - 1/3)) = (√x) ^ 8 + ((8 * 7 / 2) * (√x)^6 * (1 / (2 * x^(1/6)))^2)) 
  (n_value := 8) 
  (general_term_r := (1 / 2)^r * (8 choose r) * x^(4 - 2r / 3))
  (rational_terms := [T_{0+1}, T_{3+1}, T_{6+1}]) 
  (irrational_terms := 6)
  (total_permutations := 9!)
  (valid_permutations := (7 choose 3) * (6 choose 6))
  (desired_probability := valid_permutations / total_permutations) :
  desired_probability = 5 / 12 :=
by sorry

end probability_no_adjacent_rational_terms_l470_470260


namespace paul_buys_18_6_pounds_of_corn_l470_470912

-- Definitions of conditions
def corn_cost := 99 -- cents per pound
def beans_cost := 51 -- cents per pound
def total_weight := 22 -- total pounds of corn and beans
def total_cost := 2013 -- cost in cents

-- Proof problem setup
theorem paul_buys_18_6_pounds_of_corn :
  ∃ (c : ℚ), 0 ≤ c ∧ (c + (total_weight - c) = total_weight) ∧ ((beans_cost * (total_weight - c) + corn_cost * c = total_cost)) ∧ c ≈ 18.6 :=
by {
  sorry
}

end paul_buys_18_6_pounds_of_corn_l470_470912


namespace largest_angle_in_triangle_l470_470703

theorem largest_angle_in_triangle (A B C : ℝ) 
  (a b c : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin C = Real.sqrt 2 * Real.sin B)
  : B = 90 :=
by
  sorry

end largest_angle_in_triangle_l470_470703


namespace max_log_value_min_reciprocal_sum_l470_470974

-- Definitions for conditions
variables {x y : ℝ}
variables (h₁ : x > 0) (h₂ : y > 0)
variable (h₃ : 2 * x + 5 * y = 20)

-- Statements for the proof problems

-- Prove that under the conditions, the maximum value of u = log x + log y is 1
theorem max_log_value : ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + 5 * y = 20 → (log x + log y) ≤ 1 :=
by
  intros x y h₁ h₂ h₃
  sorry

-- Prove that under the conditions, the minimum value of 1/x + 1/y is (7 + 2 * sqrt 10) / 20
theorem min_reciprocal_sum : ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + 5 * y = 20 → (1/x + 1/y) ≥ (7 + 2 * sqrt 10) / 20 :=
by
  intros x y h₁ h₂ h₃
  sorry

end max_log_value_min_reciprocal_sum_l470_470974


namespace hyperbola_eccentricity_l470_470693

-- Definitions of conditions
variables {a b c : ℝ}
variables (h : a > 0) (h' : b > 0)
variables (hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
variables (parab : ∀ y : ℝ, y^2 = 4 * b * y)
variables (ratio_cond : (b + c) / (c - b) = 5 / 3)

-- Proof statement
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 4 * Real.sqrt 15 / 15 :=
by
  have hyp_foci_distance : ∃ c : ℝ, c^2 = a^2 + b^2 := sorry
  have e := (4 * Real.sqrt 15) / 15
  use e
  sorry

end hyperbola_eccentricity_l470_470693


namespace defined_interval_l470_470949

noncomputable def is_defined (x : ℝ) : Prop :=
  log (5 + x) / sqrt (2 - x)

theorem defined_interval (x : ℝ) : 
  (∀ (f : ℝ → ℝ), f = is_defined → f x ≠ NaN ↔ -5 < x ∧ x < 2) := 
by
  -- here would go the actual proof steps
  sorry

end defined_interval_l470_470949


namespace miquel_point_projections_collinear_l470_470171

-- Define the collinearity of projections of a point on four lines
theorem miquel_point_projections_collinear 
  (l1 l2 l3 l4 : Line)
  (P : Point) -- Miquel Point
  (h : is_miquel_point P l1 l2 l3 l4)  -- P is the Miquel point of the four lines
  : collinear (project P l1) (project P l2) (project P l3) (project P l4) :=
sorry

end miquel_point_projections_collinear_l470_470171


namespace trig_identity_l470_470767

theorem trig_identity (x y : ℝ) :
  (sin x)^2 + (sin (x + y))^2 + 2 * sin x * sin y * sin (x + y) = 
  2 - (cos x)^2 - (cos (x + y))^2 :=
by
  -- We needs the proof here
  sorry

end trig_identity_l470_470767


namespace expected_stones_approx_l470_470481

open BigOperators

noncomputable def expected_num_stones (width jump : ℕ) : ℝ :=
  let n := width / jump
  100 * Real.log n

theorem expected_stones_approx 
  (width jump : ℕ)
  (h_width: width = 400)
  (h_jump: jump = 4) : 
  expected_num_stones width jump ≈ 712.811 :=
by
  rw [h_width, h_jump]
  sorry

end expected_stones_approx_l470_470481


namespace sn_values_l470_470740

noncomputable def s (x1 x2 x3 : ℂ) (n : ℕ) : ℂ :=
  x1^n + x2^n + x3^n

theorem sn_values (p q x1 x2 x3 : ℂ) (h_root1 : x1^3 + p * x1 + q = 0)
                    (h_root2 : x2^3 + p * x2 + q = 0)
                    (h_root3 : x3^3 + p * x3 + q = 0) :
  s x1 x2 x3 2 = -3 * q ∧
  s x1 x2 x3 3 = 3 * q^2 ∧
  s x1 x2 x3 4 = 2 * p^2 ∧
  s x1 x2 x3 5 = 5 * p * q ∧
  s x1 x2 x3 6 = -2 * p^3 + 3 * q^2 ∧
  s x1 x2 x3 7 = -7 * p^2 * q ∧
  s x1 x2 x3 8 = 2 * p^4 - 8 * p * q^2 ∧
  s x1 x2 x3 9 = 9 * p^3 * q - 3 * q^3 ∧
  s x1 x2 x3 10 = -2 * p^5 + 15 * p^2 * q^2 :=
by {
  sorry
}

end sn_values_l470_470740


namespace circle_tangent_problem_l470_470417

noncomputable def circle_tangent_center := {a : ℝ // a = 1 ∨ a = -3/2}

noncomputable def circle_equation (a : ℝ) : ℝ × ℝ → ℝ :=
  match a with
  | 1 => λ p, (p.1 - 1) ^ 2 + (p.2 - 2) ^ 2 - 5
  | -3/2 => λ p, (p.1 - (-3 / 2)) ^ 2 + (p.2 - (-3)) ^ 2 - 5

theorem circle_tangent_problem :
  ∃ a : ℝ, (a = 1 ∨ a = (-3/2)) ∧ (circle_equation a (1,2) = 0) :=
sorry

end circle_tangent_problem_l470_470417


namespace min_value_abs_a_add_tb_colinear_solve_t_l470_470682

section
variables (a b c : ℝ × ℝ) (t : ℝ)

def vec_min_value (a b : ℝ × ℝ) : ℝ :=
Real.sqrt (5 * (t - 4 / 5)^2 + 49 / 5)

theorem min_value_abs_a_add_tb (ha : a = (-3, 2)) (hb : b = (2, 1)) : 
 ∃ (t : ℝ), t = 4 / 5 ∧ vec_min_value a b = 7 * Real.sqrt 5 / 5 := 
sorry

def colinear (a b c : ℝ × ℝ) : Prop :=
(-3 - (2 * t)) * (-1) - (2 - t) * 3 = 0 

theorem colinear_solve_t (ha : a = (-3, 2)) (hb : b = (2, 1)) (hc : c = (3, -1)) :
  colinear a b c → t = 3 / 5 :=
sorry

end

end min_value_abs_a_add_tb_colinear_solve_t_l470_470682


namespace larger_number_l470_470416

theorem larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end larger_number_l470_470416


namespace tyler_age_l470_470449

theorem tyler_age (T C : ℕ) (h1 : T = 3 * C + 1) (h2 : T + C = 21) : T = 16 :=
by
  sorry

end tyler_age_l470_470449


namespace scaled_determinant_l470_470953

variable {a b c d : ℝ}

theorem scaled_determinant (h : ∣↑⟨a, b, c, d⟩∣ = 7) : ∣↑⟨3 * a, 3 * b, 3 * c, 3 * d⟩∣ = 63 := by
  sorry

end scaled_determinant_l470_470953


namespace count_valid_n_l470_470599

noncomputable def integer_count (f : ℤ → ℝ) (pred : ℤ → Prop) : ℕ :=
  (finset.filter pred (finset.range 100)).card  -- Assume range limit for simplicity

def valid_integer_value (n : ℤ) : Prop :=
  ∃ k : ℕ, 2400 * (3 : ℝ)^n * (5 : ℝ)^(-n) = k

theorem count_valid_n : integer_count (λ n, 2400 * (3 / 5)^n) valid_integer_value = 2 :=
by
  sorry

end count_valid_n_l470_470599


namespace range_of_a_l470_470983

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + 1/2 * x^2

theorem range_of_a (a : ℝ)
  (H : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a (x1 + a) - f a (x2 + a)) / (x1 - x2) ≥ 3) :
  a ≥ 9 / 4 :=
sorry

end range_of_a_l470_470983


namespace max_leap_years_l470_470900

theorem max_leap_years (k n : ℕ) (hk : k = 5) (hn : n = 125) :
  let cycles := n / k in
  cycles = 25 :=
by
  -- Proof steps would go here
  sorry

end max_leap_years_l470_470900


namespace yogurt_combination_count_l470_470534

theorem yogurt_combination_count :
  let num_flavors := 6 in
  let num_toppings := 8 in
  let choose_k (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) in
  num_flavors * choose_k num_toppings 3 = 336 :=
by
  sorry

end yogurt_combination_count_l470_470534


namespace arithmetic_sequence_difference_l470_470101

theorem arithmetic_sequence_difference :
  (∑ k in finset.range 100, (1901 + k)) - (∑ k in finset.range 100, (100 + k)) = 180100 :=
by
  sorry

end arithmetic_sequence_difference_l470_470101


namespace inequality_xyz_l470_470330

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470330


namespace x_14_and_inverse_x_14_l470_470976

theorem x_14_and_inverse_x_14 (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + x⁻¹^14 = -1 :=
by
  sorry

end x_14_and_inverse_x_14_l470_470976


namespace tiles_per_row_proof_l470_470398

def area_square_room : ℝ := 400 -- in square feet
def tile_size : ℝ := 8 / 12 -- each tile is 8 inches, converted to feet (8/12 feet)

noncomputable def number_of_tiles_per_row : ℕ :=
  let side_length := Math.sqrt area_square_room in
  let side_length_in_inch := side_length * 12 in
  Nat.floor (side_length_in_inch / tile_size)

theorem tiles_per_row_proof :
  number_of_tiles_per_row = 30 := by
  sorry

end tiles_per_row_proof_l470_470398


namespace log_inequality_sqrt_inequality_l470_470375

-- Proof problem for part (1)
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 :=
sorry

-- Proof problem for part (2)
theorem sqrt_inequality :
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end log_inequality_sqrt_inequality_l470_470375


namespace min_airplane_speed_l470_470155

variables (d : ℝ) (α β : ℝ)
-- Conditions: d is the distance between points A and B, α and β are the observed angles (in radians)
-- The angles α and β are given to be acute.
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)

noncomputable def min_speed : ℝ :=
  2 * d / (Real.cot (α / 2) + Real.cot (β / 2))

theorem min_airplane_speed (d : ℝ) (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  min_speed d α β = 2 * d / (Real.cot (α / 2) + Real.cot (β / 2)) :=
by
  unfold min_speed
  sorry

end min_airplane_speed_l470_470155


namespace inequality_inequality_l470_470362

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470362


namespace sequence_first_number_l470_470527

theorem sequence_first_number :
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℤ,
  (a8 = 27) ∧
  (a9 = 44) ∧
  (a10 = 71) ∧
  (∀ n ≥ 3, ∃ (a_n : ℤ),
    a_n = (if n = 3 then a3
    else if n = 4 then a4
    else if n = 5 then a5
    else if n = 6 then a6
    else if n = 7 then a7
    else if n = 8 then a8
    else if n = 9 then a9
    else a10)) ∧
  (∀ n ≥ 3, a_n = a_{n-1} + a_{n-2} - a1) →
  a1 = -2 :=
by
  sorry

end sequence_first_number_l470_470527


namespace trig_identity_evaluation_l470_470605

theorem trig_identity_evaluation (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : -sin α = 2 * cos α) :
    2 * sin α ^ 2 - sin α * cos α + cos α ^ 2 = 11 / 5 :=
by
  sorry

end trig_identity_evaluation_l470_470605


namespace cheaper_to_buy_more_books_l470_470776

def C (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 20 then 15 * n
  else if 21 ≤ n ∧ n ≤ 40 then 13 * n
  else if 41 ≤ n ∧ n ≤ 60 then 11 * n
  else 9 * n

theorem cheaper_to_buy_more_books : (finset.filter (λ n => C (n+1) < C n) (finset.range 61)).card = 8 := sorry

end cheaper_to_buy_more_books_l470_470776


namespace BM_CN_intersect_on_circumcircle_l470_470313

open EuclideanGeometry

variables {A B C P Q M N : Point}

-- Acute-angled triangle ABC
axiom acute_triangle : is_acute_triangle A B C

-- Points P and Q on side BC such that ∠PAB = ∠BCA and ∠CAQ = ∠ABC
axiom angle_PAB_eq_BCA : angle_eq (angle P A B) (angle B C A)
axiom angle_CAQ_eq_ABC : angle_eq (angle C A Q) (angle A B C)

-- Points M and N on lines AP and AQ, with P and Q as midpoints of AM and AN respectively
axiom midpoint_P_AM : midpoint P A M
axiom midpoint_Q_AN : midpoint Q A N

-- Prove that lines BM and CN intersect on the circumcircle of triangle ABC
theorem BM_CN_intersect_on_circumcircle :
  ∃ (R : Point), lies_on_circumcircle R A B C ∧ collinear B M R ∧ collinear C N R :=
sorry

end BM_CN_intersect_on_circumcircle_l470_470313


namespace least_six_digit_divisible_by_198_l470_470817

/-- The least 6-digit natural number that is divisible by 198 is 100188. -/
theorem least_six_digit_divisible_by_198 : 
  ∃ n : ℕ, n ≥ 100000 ∧ n % 198 = 0 ∧ n = 100188 :=
by
  use 100188
  sorry

end least_six_digit_divisible_by_198_l470_470817


namespace well_diameter_l470_470503

noncomputable def volume_of_cylinder (r h : ℝ) := real.pi * r^2 * h

theorem well_diameter : 
  ∀ (r : ℝ), volume_of_cylinder r 24 = 301.59289474462014 → 2 * r = 4 :=
by
  intro r
  intro h
  rw volume_of_cylinder at h
  sorry

end well_diameter_l470_470503


namespace u_v_existence_l470_470967

-- Define the required problem statement as a theorem.
theorem u_v_existence 
  (u_0 u_1 u_2 u_3 u_4 : ℝ) :
  ∃ (v_0 v_1 v_2 v_3 v_4 : ℝ), 
  u_0 - v_0 ∈ (mathlib.set.Univ : Set ℝ) ∧ 
  u_1 - v_1 ∈ (mathlib.set.Univ : Set ℝ) ∧ 
  u_2 - v_2 ∈ (mathlib.set.Univ : Set ℝ) ∧ 
  u_3 - v_3 ∈ (mathlib.set.Univ : Set ℝ) ∧ 
  u_4 - v_4 ∈ (mathlib.set.Univ : Set ℝ) ∧ 
  (∑ (i : Fin 4) in finSet.unorderedPairs {0,1,2,3}, (v_i.val - v_j.val)^2) < 4 := 
begin
  sorry -- proof goes here
end

end u_v_existence_l470_470967


namespace intersection_a_four_range_of_a_l470_470991

variable {x a : ℝ}

-- Problem 1: Intersection of A and B for a = 4
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a^2 + 2}

theorem intersection_a_four : A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} := 
by  sorry

-- Problem 2: Range of a given condition
theorem range_of_a (a : ℝ) (h1 : a > -3/2) (h2 : ∀ x ∈ A a, x ∈ B a) : 1 ≤ a ∧ a ≤ 3 := 
by  sorry

end intersection_a_four_range_of_a_l470_470991


namespace decimal_to_fraction_l470_470037

theorem decimal_to_fraction (a b : ℚ) (h : a = 3.56) (h1 : b = 56/100) (h2 : 56.gcd 100 = 4) :
  a = 89/25 := by
  sorry

end decimal_to_fraction_l470_470037


namespace acquaintanceship_identity_l470_470052

theorem acquaintanceship_identity (n k ℓ m : ℕ) 
  (h1 : ∀ (G : Type) [graph G] (v : G), degree v = k) 
  (h2 : ∀ (G : Type) [graph G] (u v : G), edge u v → common_neighbors u v = ℓ) 
  (h3 : ∀ (G : Type) [graph G] (u v : G), ¬ edge u v → common_neighbors u v = m)
  : m * (n - k - 1) = k * (k - ℓ - 1) := sorry

end acquaintanceship_identity_l470_470052


namespace quartic_polynomial_root_l470_470132

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x - 2

theorem quartic_polynomial_root :
  Q (Real.sqrt (Real.sqrt 3) + 1) = 0 :=
by
  sorry

end quartic_polynomial_root_l470_470132


namespace two_hours_charge_l470_470501

def charge_condition_1 (F A : ℕ) : Prop :=
  F = A + 35

def charge_condition_2 (F A : ℕ) : Prop :=
  F + 4 * A = 350

theorem two_hours_charge (F A : ℕ) (h1 : charge_condition_1 F A) (h2 : charge_condition_2 F A) : 
  F + A = 161 := 
sorry

end two_hours_charge_l470_470501


namespace even_n_of_system_l470_470849

theorem even_n_of_system (n : ℕ) (x : ℕ → ℕ)
  (h1 : 2 * x 1 - x 2 = 1)
  (h2 : ∀ k : ℕ, 2 ≤ k ∧ k ≤ n - 1 → - x (k - 1) + 2 * x k - x (k + 1) = 1)
  (hn : - x (n - 1) + 2 * x n = 1) :
  Even n := 
sorry

end even_n_of_system_l470_470849


namespace area_enclosed_l470_470557

-- Define the parametric equations 
def x (t : ℝ) := 9 * cos t
def y (t : ℝ) := 4 * sin t

-- Define the boundary condition for the height of the rectangle
def Y := 2

-- Define the range for t within one period where intersections occur
def t1 := π / 6
def t2 := 5 * π / 6

-- Statement: The area enclosed by the parametric curve and the line is 12π - 18√3
theorem area_enclosed : 
  ∫ t in (t1)..(t2), (y t) * (deriv x t) dx = 12 * π - 18 * sqrt 3 :=
sorry

end area_enclosed_l470_470557


namespace quadrilateral_diagonals_l470_470620

-- Define the points of the quadrilateral
variables {A B C D P Q R S : ℝ × ℝ}

-- Define the midpoints condition
def is_midpoint (M : ℝ × ℝ) (X Y : ℝ × ℝ) := M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- Define the lengths squared condition
def dist_sq (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Main theorem to prove
theorem quadrilateral_diagonals (hP : is_midpoint P A B) (hQ : is_midpoint Q B C)
  (hR : is_midpoint R C D) (hS : is_midpoint S D A) :
  dist_sq A C + dist_sq B D = 2 * (dist_sq P R + dist_sq Q S) :=
by
  sorry

end quadrilateral_diagonals_l470_470620


namespace basic_quantities_l470_470089

-- Define the geometric sequence and the sum of the first n terms
def geom_seq (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n-1)

def sum_geom_seq (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a1
  else a1 * (1 - q^n) / (1 - q)

-- Main theorem statement
theorem basic_quantities (S1 S2 a2 S3 a1 q an : ℝ) (n : ℕ) (h1 : n > 1):
  (geom_seq a1 q 1 = a1) →
  (geom_seq a1 q 2 = a2) →
  (sum_geom_seq a1 q 1 = S1) →
  (sum_geom_seq a1 q 2 = S2) →
  (sum_geom_seq a1 q n = an) →
  {(S1, S2), (q, an)} = {(true, true)} :=
sorry

end basic_quantities_l470_470089


namespace sum_combinatorial_identity_l470_470382

theorem sum_combinatorial_identity (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) (hkn : k ≤ n) :
  (∑ r in Finset.range (m + 1), k * (Nat.choose m r) * (Nat.choose n k) / ((r + k) * (Nat.choose (m + n) (r + k)))) = 1 :=
sorry

end sum_combinatorial_identity_l470_470382


namespace sum_interior_diagonals_l470_470884

variables (a b c : ℝ)

-- Conditions
def surface_area_condition : Prop := 2 * (a * b + b * c + c * a) = 142
def edge_length_condition : Prop := a + b + c = 15

-- Theorem to prove the sum of the lengths of all interior diagonals
theorem sum_interior_diagonals (h1: surface_area_condition a b c) (h2: edge_length_condition a b c) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 83 := 
begin 
  sorry
end

end sum_interior_diagonals_l470_470884


namespace infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l470_470495

-- Problem 1: Infinitely many primes congruent to 3 modulo 4
theorem infinite_primes_congruent_3_mod_4 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 4 = 3) → ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ ps :=
by
  sorry

-- Problem 2: Infinitely many primes congruent to 5 modulo 6
theorem infinite_primes_congruent_5_mod_6 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 6 = 5) → ∃ q, Nat.Prime q ∧ q % 6 = 5 ∧ q ∉ ps :=
by
  sorry

end infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l470_470495


namespace max_area_triangle_proof_l470_470972

noncomputable def max_area_triangle (a b c : ℝ) (A B C : ℝ) (angle_opposite : ℝ → ℝ) : ℝ :=
if h : a = 2 ∧ (2 + b) * (real.sin A - real.sin B) = (c - b) * real.sin C then
√3 else 0

theorem max_area_triangle_proof (a b c A B C : ℝ) 
  (h₁ : a = 2) 
  (h₂ : (2 + b) * (real.sin A - real.sin B) = (c - b) * real.sin C) :
  max_area_triangle a b c A B C = √3 := 
by 
  simp [max_area_triangle, h₁, h₂]


end max_area_triangle_proof_l470_470972


namespace no_integer_n_gte_1_where_9_divides_7n_plus_n3_l470_470546

theorem no_integer_n_gte_1_where_9_divides_7n_plus_n3 :
  ∀ n : ℕ, 1 ≤ n → ¬ (7^n + n^3) % 9 = 0 := 
by
  intros n hn
  sorry

end no_integer_n_gte_1_where_9_divides_7n_plus_n3_l470_470546


namespace max_value_of_f_l470_470270

variable (n : ℕ)

-- Define the quadratic function with coefficients a, b, and c.
noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
axiom f_n : ∃ a b c, f n a b c = 6
axiom f_n1 : ∃ a b c, f (n + 1) a b c = 14
axiom f_n2 : ∃ a b c, f (n + 2) a b c = 14

-- The main goal is to prove the maximum value of f(x) is 15.
theorem max_value_of_f : ∃ a b c, (∀ x : ℝ, f x a b c ≤ 15) :=
by
  sorry

end max_value_of_f_l470_470270


namespace tiles_in_each_row_l470_470391

theorem tiles_in_each_row (area : ℝ) (tile_side : ℝ) (feet_to_inches : ℝ) (number_of_tiles : ℕ) :
  area = 400 ∧ tile_side = 8 ∧ feet_to_inches = 12 → number_of_tiles = 30 :=
by
  intros h
  cases h with h_area h_rest
  cases h_rest with h_tile_side h_feet_to_inches
  sorry

end tiles_in_each_row_l470_470391


namespace probability_factor_less_than_ten_l470_470456

def prime_factors (n : ℕ) : Prop := 
  n = 2^1 * 3^2 * 5^1

def is_factor (m n : ℕ) : Prop :=
  n % m = 0

def factors_less_than (n bound : ℕ) : ℕ :=
  (List.filter (λ x, x < bound) (List.range (n+1))).count (λ x, is_factor x n)

theorem probability_factor_less_than_ten :
  prime_factors 90 →
  (factors_less_than 90 10).val = 6 →
  (6 / 12 : ℚ) = 1 / 2 :=
by
  sorry

end probability_factor_less_than_ten_l470_470456


namespace total_first_class_passengers_l470_470881

-- Define the problem parameters and conditions
def total_passengers : ℕ := 300
def women_percentage : ℚ := 50 / 100
def men_percentage : ℚ := 50 / 100
def women_first_class_percentage : ℚ := 20 / 100
def men_first_class_percentage : ℚ := 15 / 100

-- Define the number of women, men, and passengers in first class
def number_of_women := total_passengers * women_percentage
def number_of_men := total_passengers * men_percentage
def women_first_class := number_of_women * women_first_class_percentage
def men_first_class := (number_of_men * men_first_class_percentage).round

-- Define the total number of first class passengers
def first_class_passengers := women_first_class + men_first_class

-- The theorem to prove the total number of first class passengers
theorem total_first_class_passengers : first_class_passengers = 53 := by
  sorry  -- Proof would be provided here


end total_first_class_passengers_l470_470881


namespace fields_medal_statistics_l470_470384

def data_set : list ℕ := [31, 32, 33, 35, 35, 39]

def mode (l: list ℕ) : ℕ := 
  l.foldr 
    (λ x (acc : ℕ × ℕ), 
      if (l.count x > acc.2) then (x, l.count x) else acc) 
    (0, 0) 
  .1 

def median (l : list ℕ) : ℕ := 
  if (l.length % 2 = 1) then 
    l.nth_le (l.length / 2) (by sorry) 
  else 
    (l.nth_le (l.length / 2 - 1) (by sorry) + l.nth_le (l.length / 2) (by sorry)) / 2

theorem fields_medal_statistics : mode data_set = 35 ∧ median data_set = 34 :=
by sorry

end fields_medal_statistics_l470_470384


namespace math_problem_l470_470515

theorem math_problem
  (N O : ℝ)
  (h₁ : 96 / 100 = |(O - 5 * N) / (5 * N)|)
  (h₂ : 5 * N ≠ 0) :
  O = 0.2 * N :=
by
  sorry

end math_problem_l470_470515


namespace sum_of_edges_l470_470115

theorem sum_of_edges (n : ℕ) (total_length large_edge small_edge : ℤ) : 
  n = 27 → 
  total_length = 828 → -- convert to millimeters
  large_edge = total_length / 12 → 
  small_edge = large_edge / 3 → 
  (large_edge + small_edge) / 10 = 92 :=
by
  intros
  sorry

end sum_of_edges_l470_470115


namespace sum_200_to_299_l470_470222

variable (a : ℕ)

-- Condition: Sum of the first 100 natural numbers is equal to a
def sum_100 := (100 * 101) / 2

-- Main Theorem: Sum from 200 to 299 in terms of a
theorem sum_200_to_299 (h : sum_100 = a) : (299 * 300 / 2 - 199 * 200 / 2) = 19900 + a := by
  sorry

end sum_200_to_299_l470_470222


namespace inequality_holds_for_real_numbers_l470_470319

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470319


namespace urn_possible_contents_l470_470093

def initial_white := 150
def initial_black := 50

inductive Operation
| Rule1 : Operation -- 4W -> 2B
| Rule2 : Operation -- 3W + 1B -> 2W + 1B
| Rule3 : Operation -- 2W + 2B -> 1W + 2B
| Rule4 : Operation -- 3B + 1W -> 1B + 1W
| Rule5 : Operation -- 4B -> 3B

def perform_operation (op : Operation) (w b : ℕ) : ℕ × ℕ :=
  match op with
  | Operation.Rule1 => (w - 4, b + 2)
  | Operation.Rule2 => (w - 1, b)
  | Operation.Rule3 => (w - 1, b + 1)
  | Operation.Rule4 => (w, b - 2)
  | Operation.Rule5 => (w, b - 1)

def even_total (w b : ℕ) : Prop :=
  (w + b) % 2 = 0

theorem urn_possible_contents : ∃ (ops : list Operation), 
  let ⟨w', b'⟩ := ops.foldl (λ ⟨w, b⟩ op, perform_operation op w b) (initial_white, initial_black) in
  even_total w' b' ∧ ((w' = 78 ∧ b' = 72) ∨ (w' = 126 ∧ b' = 24)) :=
sorry

end urn_possible_contents_l470_470093


namespace minimal_n_l470_470294

open Set

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 280}

-- Define a function to check if a set contains five pairwise relatively prime numbers
def has_five_pairwise_rel_prime (subset : Set ℕ) : Prop :=
  ∃ a b c d e, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ e ∈ subset ∧
  Nat.coprime a b ∧ Nat.coprime a c ∧ Nat.coprime a d ∧ Nat.coprime a e ∧
  Nat.coprime b c ∧ Nat.coprime b d ∧ Nat.coprime b e ∧
  Nat.coprime c d ∧ Nat.coprime c e ∧ Nat.coprime d e
  
theorem minimal_n (n : ℕ) : n = 217 ↔ ∀ (subset : Set ℕ), subset ⊆ S ∧ subset.card = n → has_five_pairwise_rel_prime subset :=
sorry

end minimal_n_l470_470294


namespace value_of_abc_l470_470238

noncomputable def f (x a b c : ℝ) := |(1 - x^2) * (x^2 + a * x + b)| - c

theorem value_of_abc :
  (∀ x : ℝ, f (x + 4) 8 15 9 = f (-x) 8 15 9) ∧
  (∃ x : ℝ, f x 8 15 9 = 0) ∧
  (∃ x : ℝ, f (-(x-4)) 8 15 9 = 0) ∧
  (∀ c : ℝ, c ≠ 0) →
  8 + 15 + 9 = 32 :=
by sorry

end value_of_abc_l470_470238


namespace bus_stop_time_l470_470129

open Real

/-- 
  Excluding stoppages, the speed of a bus is 75 kmph, 
  and including stoppages, it is 45 kmph. Prove 
  that the bus stops for 24 minutes per hour.
-/
theorem bus_stop_time (v_excluding : ℝ) (v_including : ℝ) (h1 : v_excluding = 75) (h2 : v_including = 45) :
  ∃ t : ℝ, t = 24 :=
by
  -- define the difference in speed due to stoppages
  have diff_speed : ℝ := v_excluding - v_including
  rw [h1, h2] at diff_speed
  rw [show 75 - 45 = 30 by norm_num] at diff_speed

  -- convert the speed from kmph to km per minute
  have speed_per_minute : ℝ := v_excluding / 60
  rw h1 at speed_per_minute
  rw [show 75 / 60 = 1.25 by norm_num] at speed_per_minute

  -- calculate the stop time
  have stop_time : ℝ := diff_speed / speed_per_minute
  rw [show 30 / 1.25 = 24 by norm_num, norm_num1] at stop_time
  use stop_time
  exact rfl


end bus_stop_time_l470_470129


namespace find_point_with_instantaneous_rate_of_change_l470_470186

theorem find_point_with_instantaneous_rate_of_change :
  ∃ (x0 y0 : ℝ), (f(x0) = 2*x0^2 + 1 ∧ (f'(x0) = 4 * x0 ∧ 4 * x0 = -8) ∧ y0 = f(-2) ∧ y0 = 9) :=
by 
  sorry

end find_point_with_instantaneous_rate_of_change_l470_470186


namespace correct_derivative_of_log2_l470_470027

theorem correct_derivative_of_log2 (x : ℝ) (hx : x > 0) : 
  (\big[(log x) / (log 2)])' = 1 / (x * log 2) := by
sory

end correct_derivative_of_log2_l470_470027


namespace inequality_range_a_l470_470239

theorem inequality_range_a (a : ℝ) :
  (∀ x : ℝ, a * (sin x)^2 + cos x ≥ a^2 - 1) → a = 0 :=
by
  sorry

end inequality_range_a_l470_470239


namespace tangerine_cost_l470_470923

theorem tangerine_cost (M : ℕ) (h : M = 8000) : M / 2 = 4000 :=
by
  have H1 : M / 2 = 4000,
  calc
    M / 2 = 8000 / 2 : by rw [h]
    ... = 4000       : by norm_num
  exact H1

example : tangerine_cost 8000 (rfl) = 4000 :=
by sorry

end tangerine_cost_l470_470923


namespace angles_of_rhombus_l470_470296

variables (ABCD : Type) [metric_space ABCD] [add_torsor ABCD] 
variables (A B C D P Q : ABCD)
variables (angle : ABCD → ABCD → ABCD → ℝ) (side_length : ℝ)
variables (equilateral_triangle : ABCD × ABCD × ABCD → Prop)
variables (is_rhombus : ABCD × ABCD × ABCD × ABCD → Prop)

-- Given conditions
axiom rhombus_ABCD : is_rhombus (A, B, C, D)
axiom angle_BAD_gt_angle_ABC : angle A B D > angle B A C
axiom points_on_AB_AD : P ∈ line_segment B A ∧ Q ∈ line_segment D A
axiom triangle_PCQ_equilateral : equilateral_triangle (P, C, Q)
axiom sides_equal : ∀ (X Y : ABCD), side_length = dist X Y

-- Question
theorem angles_of_rhombus :
  ∀ (α β : ℝ), 
  α + β = 180 ∧ 
  angle A B C = α ∧ angle A D C = α ∧ 
  angle B A D = β ∧ angle B C D = β → 
  (α = 80 ∧ β = 100) ∧
  angle A B C = 80 ∧ angle A D C = 80 ∧ 
  angle B A D = 100 ∧ angle B C D = 100 :=
sorry

end angles_of_rhombus_l470_470296


namespace distinct_intersection_values_l470_470440

theorem distinct_intersection_values (L : Fin 5 → Line) (m : ℕ) :
  (∃ (mvalues : Finset ℕ), 
     (∀ L, 
      mvalues = 
        { x | x = (0 ∨ x = 1 ∨ x = 4 ∨ x = 6 ∨ x = 7 ∨ x = 9 ∨ x = 10) }) ∧ 
      (mvalues.card = 9)).
sorry

end distinct_intersection_values_l470_470440


namespace find_b_l470_470436

theorem find_b (c : ℝ) : (∀ x : ℝ, x = -1 → x = - (b / (2 : ℝ))) → b = 2 :=
begin
  intros h,
  sorry
end

end find_b_l470_470436


namespace smallest_int_between_53_and_104_l470_470791

def smallest_positive_int (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ (m % 3 = 1) ∧ (m % 5 = 1) ∧ (m % 7 = 1) ∧ m = n

def within_range (n : ℕ) (a b : ℕ) : Prop :=
  a < n ∧ n ≤ b

theorem smallest_int_between_53_and_104 :
  ∃ n, smallest_positive_int n ∧ within_range n 53 104 :=
begin
  sorry
end

end smallest_int_between_53_and_104_l470_470791


namespace proj_v_w_l470_470151

noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
  let w_dot_w := dot_product w w
  let v_dot_w := dot_product v w
  let scalar := v_dot_w / w_dot_w
  (scalar * w.1, scalar * w.2)

theorem proj_v_w :
  let v := (4, -3)
  let w := (12, 5)
  proj v w = (396 / 169, 165 / 169) :=
by
  sorry

end proj_v_w_l470_470151


namespace ursula_annual_salary_l470_470450

def hourly_wage : ℝ := 8.50
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

noncomputable def daily_earnings : ℝ := hourly_wage * hours_per_day
noncomputable def monthly_earnings : ℝ := daily_earnings * days_per_month
noncomputable def annual_salary : ℝ := monthly_earnings * months_per_year

theorem ursula_annual_salary : annual_salary = 16320 := 
  by sorry

end ursula_annual_salary_l470_470450


namespace problem1_condition_problem2_problem3_l470_470604

variables {x : ℝ} {n : ℕ} [fact (0 < n)] 
noncomputable def C (m k : ℕ) : ℚ := nat.choose m k

def a (k : ℕ) : ℚ :=
  ∑ i in finset.range (k+1), C (2 * n) i * (1 / (4 : ℚ)) ^ i

theorem problem1_condition 
  (hn : (1 + x / 4) ^ (2 * n) = ∑ i in finset.range (2 * n + 1), (a i : ℝ) * x ^ i)
  (h_sum : ∑ i in finset.range (2 * n + 1), a i = 625 / 256) :
  a 3 = 1 / 16 :=
sorry

theorem problem2
  (hn : (1 + x / 4)^ (2 * n) = ∑ i in finset.range (2 * n + 1), (a i : ℝ) * x^i) :
  ∀ (n : ℕ) [fact (0 < n)], a n < 1/real.sqrt (2*n + 1) :=
sorry

theorem problem3
  (hn : (1 + x / 4)^ (2 * n) = ∑ i in finset.range (2 * n + 1), (a i : ℝ) * x^i) :
  ∃ k : ℕ, 0 ≤ k ∧ k ≤ 2 * n ∧ ∀ m : ℕ, 0 ≤ m ∧ m ≤ 2 * n → a k >= a m ∧ 
  ( ∀ j : ℕ, 0 ≤ j ∧ j ≤ 2 * n → a j = a k → j = k ∨ j = k - 1) :=
sorry

end problem1_condition_problem2_problem3_l470_470604


namespace ratio_b_to_c_l470_470834

variable (a b c k : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = k * c
def condition3 : Prop := a + b + c = 32
def condition4 : Prop := b = 12

-- Question: Prove that ratio of b to c is 2:1
theorem ratio_b_to_c
  (h1 : condition1 a b)
  (h2 : condition2 b k c)
  (h3 : condition3 a b c)
  (h4 : condition4 b) :
  b = 2 * c := 
sorry

end ratio_b_to_c_l470_470834


namespace part1_minimum_a_part2_range_a_l470_470995

theorem part1_minimum_a (x : ℝ) (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_parallel : (ln x, 1 - a * ln x) = (λ x, (x, f x)) x)
  (h_monotonic : ∀ x > 1, f' x ≤ 0) :
  a ≥ 1 / 4 :=
sorry

theorem part2_range_a (x₁ x₂ : ℝ) (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_parallel : (ln x, 1 - a * ln x) = (λ x, (x, f x)) x)
  (h_exists : ∃ x₁ x₂ ∈ Icc (real.exp 1) (real.exp 2), f x₁ ≤ f'(x₂) + a) :
  a ∈ Icc (1 / 4) ⊤ :=
sorry

end part1_minimum_a_part2_range_a_l470_470995


namespace annual_payment_amount_total_interest_paid_l470_470512

variable (c p n : ℝ)

def k : ℝ := 1 + p / 100

theorem annual_payment_amount (a : ℝ) (hn : n ≠ 0) (hp : 1 + p / 100 > 1) :
  a = c * k p * (k p - 1) / (k p ^ n - 1) :=
by
  sorry

theorem total_interest_paid (interest : ℝ) (hn : n ≠ 0) (hp : 1 + p / 100 > 1) :
  interest = (n * (k p) ^ (n + 1) - (n + 1) * (k p) ^ n + 1) / (k p ^ n - 1) :=
by
  sorry

end annual_payment_amount_total_interest_paid_l470_470512


namespace solve_trig_equation_l470_470769

theorem solve_trig_equation (n : ℤ) :
  ∃ x, 
    (sin x ^ 3 + 6 * cos x ^ 3 + (1 / Real.sqrt 2) * sin (2 * x) * sin (x + Real.pi / 4) = 0) ∧ 
    (x = -Real.arctan 2 + Real.pi * n) :=
sorry

end solve_trig_equation_l470_470769


namespace maria_initial_gum_count_l470_470300

variable {x : ℕ}

theorem maria_initial_gum_count
  (Tommy_gave : 16)
  (Luis_gave : 20)
  (total_now : (x + 16 + 20) = 61) :
  x = 25 := by
  sorry

end maria_initial_gum_count_l470_470300


namespace function_two_common_points_with_xaxis_l470_470195

theorem function_two_common_points_with_xaxis (c : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x + c = 0 → x = -1 ∨ x = 1) → (c = -2 ∨ c = 2) :=
by
  sorry

end function_two_common_points_with_xaxis_l470_470195


namespace total_gold_coins_l470_470544

/--
An old man distributed all the gold coins he had to his two sons into 
two different numbers such that the difference between the squares 
of the two numbers is 49 times the difference between the two numbers. 
Prove that the total number of gold coins the old man had is 49.
-/
theorem total_gold_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 :=
sorry

end total_gold_coins_l470_470544


namespace angle_CDB_is_45_l470_470885

-- Define the basic setup
noncomputable def Point := ℕ -- Using Nat instead of specific coordinates for points simplicity

-- Let T be a right-angled isosceles triangle
structure Triangle :=
(A B C : Point)
(∠A : ℕ)
(∠B : ℕ)
(∠C : ℕ)

-- Let R be a rectangle
structure Rectangle :=
(A B C D : Point)
(∠A : ℕ)
(∠B : ℕ)
(∠C : ℕ)
(∠D : ℕ)

-- Define the right-angled isosceles triangle with the property
def right_angled_isosceles_triangle (T : Triangle) : Prop :=
T.∠A = 45 ∧ T.∠B = 90 ∧ T.∠C = 45

-- Define the rectangle
def rectangle (R : Rectangle) : Prop :=
R.∠A = 90 ∧ R.∠B = 90 ∧ R.∠C = 90 ∧ R.∠D = 90

-- Define sharing side condition
def shares_side (T : Triangle) (R : Rectangle) : Prop :=
T.B = R.B ∧ T.C = R.C

-- Prove that m∠CDB = 45°
theorem angle_CDB_is_45 (T : Triangle) (R : Rectangle) (h1 : right_angled_isosceles_triangle T) (h2 : rectangle R) (h3 : shares_side T R) : 
  T.∠C = 45 :=
by sorry

end angle_CDB_is_45_l470_470885


namespace special_op_equality_l470_470230

def special_op (x y : ℕ) : ℕ := x * y - x - 2 * y

theorem special_op_equality : (special_op 7 4) - (special_op 4 7) = 3 := by
  sorry

end special_op_equality_l470_470230


namespace new_supervisor_salary_l470_470044

namespace FactorySalaries

variables (W S2 : ℝ)

def old_supervisor_salary : ℝ := 870
def old_average_salary : ℝ := 430
def new_average_salary : ℝ := 440

theorem new_supervisor_salary :
  (W + old_supervisor_salary) / 9 = old_average_salary →
  (W + S2) / 9 = new_average_salary →
  S2 = 960 :=
by
  intros h1 h2
  -- Proof steps would go here
  sorry

end FactorySalaries

end new_supervisor_salary_l470_470044


namespace cylinder_cube_volume_ratio_l470_470509

theorem cylinder_cube_volume_ratio (s : ℝ) (h : s > 0) :
  let r := s / 2,
  let V_cylinder := π * r^2 * s,
  let V_cube := s^3
  in V_cylinder / V_cube = π / 4 :=
by
  let r := s / 2
  let V_cylinder := π * r^2 * s
  let V_cube := s^3
  sorry

end cylinder_cube_volume_ratio_l470_470509


namespace snow_volume_correct_l470_470299

noncomputable def volume_snow_to_shovel 
  (total_length : ℝ) 
  (width : ℝ) 
  (depth : ℝ) 
  (no_shovel_length : ℝ) : ℝ :=
  let effective_length := total_length - no_shovel_length in
  effective_length * width * depth

theorem snow_volume_correct :
  volume_snow_to_shovel 30 2.5 0.75 5 = 46.875 :=
by
  sorry

end snow_volume_correct_l470_470299


namespace ladder_geometric_sequence_solution_l470_470074

-- A sequence {aₙ} is a 3rd-order ladder geometric sequence given by a_{n+3}^2 = a_n * a_{n+6} for any positive integer n
def ladder_geometric_3rd_order (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 3) ^ 2 = a n * a (n + 6)

-- Initial conditions
def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 4 = 2

-- Main theorem to be proven in Lean 4
theorem ladder_geometric_sequence_solution :
  ∃ a : ℕ → ℝ, ladder_geometric_3rd_order a ∧ initial_conditions a ∧ a 10 = 8 :=
by
  sorry

end ladder_geometric_sequence_solution_l470_470074


namespace inequality_holds_for_all_real_numbers_l470_470370

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470370


namespace pencils_more_than_pens_l470_470428

def ratio_of_pens_to_pencils (pens pencils : ℕ) : Prop := pens * 6 = pencils * 5

def number_of_pencils : ℕ := 24
def number_of_pens : ℕ := 20

theorem pencils_more_than_pens :
  ratio_of_pens_to_pencils number_of_pens number_of_pencils → (number_of_pencils - number_of_pens) = 4 :=
begin
  intro ratio,
  unfold ratio_of_pens_to_pencils at ratio,
  rw [← nat.mul_right_inj 5, nat.mul_add_eq_mul_add_5_6_right] at ratio,
  rw nat.mul_comm at ratio,
  have eq_pens_20 : number_of_pens = 20,
  { sorry },
  rw eq_pens_20,
  refl,
end

end pencils_more_than_pens_l470_470428


namespace correct_statements_l470_470110

noncomputable def f (x : ℝ) : ℝ := x * Real.arcsin x

theorem correct_statements : 
Proposition :=
begin
  -- Statement ②: f(x) is an even function
  have h_even : ∀ x, f (-x) = f x,
  { intro x,
    unfold f,
    rw [neg_mul_eq_neg_mul, Real.arcsin_neg],
    rw [neg_mul_eq_neg_mul, neg_neg] },

  -- Statement ④: The maximum value of f(x) is π/2 and the minimum value is 0
  have h_max_min : (∀ x, x ∈ (-1, 1) → f x ≤ f 1) ∧ (∀ x, f x ≥ 0),
  { split,
    { intros x hx,
      rw [Icc, le_max_iff, le_min_iff] at hx,
      exact Real.arcsin_le },
    { intros x,
      rw [unfold f, like others] }

  exact (h_even, h_max_min),
end

end correct_statements_l470_470110


namespace monotonicity_f_f_gt_lower_bound_l470_470665

-- Definition of the function
def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

-- Statement 1: Monotonicity discussion
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, (f a)' x < 0) ∧ 
  (a > 0 → ∀ x : ℝ,
    (f a)' x < 0 ∧ x < Real.log (1 / a) ∨
    (f a)' x > 0 ∧ x > Real.log (1 / a)) :=
sorry

-- Statement 2: Proof for f(x) > 2 ln a + 3/2 for a > 0
theorem f_gt_lower_bound (a x : ℝ) (ha : 0 < a) :
  f a x > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_f_f_gt_lower_bound_l470_470665


namespace dot_product_eq_eight_l470_470994

def vec_a : ℝ × ℝ := (0, 4)
def vec_b : ℝ × ℝ := (2, 2)

theorem dot_product_eq_eight : (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2) = 8 := by
  sorry

end dot_product_eq_eight_l470_470994


namespace max_min_difference_l470_470573

variable (m : ℕ)
variable (q : ℕ → set ℝ^3)
variable (R : set ℝ^3)

def T (R : set ℝ^3) : set ℝ^3 := 
  -- Union of the faces of the cube R
  sorry

def Q (m : ℕ) (q : ℕ → set ℝ^3) : set ℝ^3 := 
  -- Union of m distinct planes q_1, q_2, ..., q_m
  ⋃ i in finset.range m, q i

def intersects (Q : set ℝ^3) (T : set ℝ^3) : set (set ℝ^3) :=
  -- The set of closed figures formed when Q intersects T
  sorry

theorem max_min_difference : 
  (max_m : ℕ, min_m : ℕ, 
  (∀ m, m = max_m ∧ m = min_m → Q m q ∩ T R ≠ ∅) → 
  max_m - min_m = 9) :=
sorry

end max_min_difference_l470_470573


namespace equal_distribution_l470_470752

theorem equal_distribution (num_friends : ℕ) (total_cakes : ℕ) (cakes_per_friend : ℕ) (h1 : num_friends = 4) (h2 : total_cakes = 8) : cakes_per_friend = nat.div total_cakes num_friends :=
by
  rw [h1, h2]
  norm_num
  exact rfl

end equal_distribution_l470_470752


namespace integer_values_within_bounds_l470_470997

theorem integer_values_within_bounds : 
  ({x : ℤ | abs x < 4 * Real.pi}).card = 25 := 
begin
  sorry,
end

end integer_values_within_bounds_l470_470997


namespace spiral_stripe_length_l470_470060

noncomputable def length_of_stripe (height circumference : ℝ) : ℝ :=
  real.sqrt (height^2 + (2 * circumference)^2)

theorem spiral_stripe_length :
  length_of_stripe 8 18 = real.sqrt 1360 :=
by
  sorry

end spiral_stripe_length_l470_470060


namespace number_of_tiles_per_row_in_square_room_l470_470403

theorem number_of_tiles_per_row_in_square_room 
  (area_of_floor: ℝ)
  (tile_side_length: ℝ)
  (conversion_rate: ℝ)
  (h1: area_of_floor = 400) 
  (h2: tile_side_length = 8) 
  (h3: conversion_rate = 12) 
: (sqrt area_of_floor * conversion_rate / tile_side_length) = 30 := 
by
  sorry

end number_of_tiles_per_row_in_square_room_l470_470403


namespace distinct_triplets_count_l470_470048

open Nat

theorem distinct_triplets_count : 
  let count_distinct := (finset.range 601).count (λ n => 
    (floor (n / 2) ≠ floor ((n + 1) / 2)) ∨
    (floor (n / 3) ≠ floor ((n + 1) / 3)) ∨
    (floor (n / 5) ≠ floor ((n + 1) / 5))
  ) 
  in count_distinct = 440 
:= by
  sorry

end distinct_triplets_count_l470_470048


namespace train_length_approximately_l470_470085

-- Given conditions
def bridge_length : ℝ := 200
def train_speed_kmph : ℝ := 36
def time_to_cross_bridge_seconds : ℝ := 29.997600191984642

-- Conversion from kmph to m/s
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

-- Converting to solve for the length of the train
def train_length : ℝ := (train_speed_mps * time_to_cross_bridge_seconds) - bridge_length

-- Proof statement
theorem train_length_approximately :
  abs (train_length - 99.97600191984642) < 0.0001 :=
by
  sorry

end train_length_approximately_l470_470085


namespace find_y_given_conditions_l470_470224

theorem find_y_given_conditions : 
  ∀ (x y : ℝ), (1.5 * x = 0.75 * y) ∧ (x = 24) → (y = 48) :=
by
  intros x y h
  cases h with h1 h2
  rw h2 at h1
  sorry

end find_y_given_conditions_l470_470224


namespace minimum_teachers_l470_470523

theorem minimum_teachers 
(hM : ℕ) (hP : ℕ) (hC : ℕ) (max_subjects : ℕ)
  (hM_pos : hM = 4)
  (hP_pos : hP = 3)
  (hC_pos : hC = 3)
  (max_subjects_cond : max_subjects = 2) : 
  ∃ min_teachers : ℕ, min_teachers = 6 := 
by
  use 6
  sorry

end minimum_teachers_l470_470523


namespace children_exceed_bridge_limit_l470_470078

theorem children_exceed_bridge_limit :
  ∀ (Kelly_weight : ℕ) (Megan_weight : ℕ) (Mike_weight : ℕ),
  Kelly_weight = 34 ∧
  Kelly_weight = (85 * Megan_weight) / 100 ∧
  Mike_weight = Megan_weight + 5 →
  Kelly_weight + Megan_weight + Mike_weight - 100 = 19 :=
by sorry

end children_exceed_bridge_limit_l470_470078


namespace problem_solution_l470_470677

-- Define the sets and the conditions given in the problem
def setA : Set ℝ := 
  {y | ∃ (x : ℝ), (x ∈ Set.Icc (3 / 4) 2) ∧ (y = x^2 - (3 / 2) * x + 1)}

def setB (m : ℝ) : Set ℝ := 
  {x | x + m^2 ≥ 1}

-- The proof statement contains two parts
theorem problem_solution (m : ℝ) :
  -- Part (I) - Prove the set A
  setA = Set.Icc (7 / 16) 2
  ∧
  -- Part (II) - Prove the range for m
  (∀ x, x ∈ setA → x ∈ setB m) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
  sorry

end problem_solution_l470_470677


namespace work_time_for_two_people_l470_470273

theorem work_time_for_two_people (individual_time : ℕ) (num_people : ℕ) (combined_time : ℕ) 
    (h1 : individual_time = 10)
    (h2 : num_people = 2)
    (h3 : ∀ t n, combined_time * n = t) : 
  combined_time = 5 :=
by
  -- Assumptions
  have h4 : 10 = individual_time, from h1
  have h5 : 2 = num_people, from h2
  
  -- Proof (skipped)
  sorry

end work_time_for_two_people_l470_470273


namespace sum_powers_mod_p_eq_zero_l470_470734

theorem sum_powers_mod_p_eq_zero (p : ℕ) (hp : Nat.Prime p) (i : ℕ) (hi : 1 ≤ i ∧ i ≤ p - 1) :
  (∑ k in Finset.range p \ {0}, k^i) % p = 0 := 
sorry

end sum_powers_mod_p_eq_zero_l470_470734


namespace division_of_neg_six_by_three_l470_470555

theorem division_of_neg_six_by_three : (-6) / 3 = -2 := by
  sorry

end division_of_neg_six_by_three_l470_470555


namespace curve_is_line_l470_470119

def curve_theta (theta : ℝ) : Prop :=
  theta = Real.pi / 4

theorem curve_is_line : curve_theta θ → (curve_type = "line") :=
by
  intros h
  cases h
  -- This is where the proof would go, but we'll use a placeholder for now.
  -- The essence of the proof will show that all points making an angle of π/4 with the x-axis lie on a line.
  exact sorry

end curve_is_line_l470_470119


namespace integral_evaluation_l470_470852

noncomputable def integral_answer : ℝ :=
  ∫ x in 1..2, ((Real.exp x) + 1 / x)

theorem integral_evaluation :
  integral_answer = Real.exp 2 - Real.exp 1 + Real.log 2 :=
by
  sorry

end integral_evaluation_l470_470852


namespace distance_shifted_l470_470108

theorem distance_shifted (A B C : Type) [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C]
  (s : ℝ) (y : ℝ) (area_sq : ℝ) (equal_areas : ℝ) :
  (s^2 = 18) ∧ (y = 2 * Real.sqrt 3) →
  sqrt(2 * y^2) = 2 * sqrt 6 :=
by
  sorry

end distance_shifted_l470_470108


namespace inequality_holds_for_real_numbers_l470_470318

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470318


namespace count_4_digit_numbers_gt_1000_l470_470209

def digits : List ℕ := [2, 0, 2, 3]

def is_valid_4_digit_number (n : ℕ) : Prop :=
  let ds := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  n >= 1000 ∧ 
  n < 10000 ∧ 
  ds.perm digits

theorem count_4_digit_numbers_gt_1000 :
  (List.filter is_valid_4_digit_number (List.range 1000 10000)).length = 3 :=
by
  sorry

end count_4_digit_numbers_gt_1000_l470_470209


namespace midpoint_sum_of_coords_l470_470562

theorem midpoint_sum_of_coords (x1 y1 x2 y2 : ℝ) (hx1 : x1 = -4) (hy1 : y1 = 7) (hx2 : x2 = 12) (hy2 : y2 = -5) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 5 :=
by
  rw [hx1, hy1, hx2, hy2]
  norm_num
  sorry

end midpoint_sum_of_coords_l470_470562


namespace fair_coin_heads_probability_l470_470775

theorem fair_coin_heads_probability
  (fair_coin : ∀ n : ℕ, (∀ (heads tails : ℕ), heads + tails = n → (heads / n = 1 / 2) ∧ (tails / n = 1 / 2)))
  (n : ℕ)
  (heads : ℕ)
  (tails : ℕ)
  (h1 : n = 20)
  (h2 : heads = 8)
  (h3 : tails = 12)
  (h4 : heads + tails = n)
  : heads / n = 1 / 2 :=
by
  sorry

end fair_coin_heads_probability_l470_470775


namespace jake_marks_difference_l470_470720

theorem jake_marks_difference : 
  ∀ (x : ℕ), 
  (80 + x + 65 + 65 = 300) → 
  (x = 90) → 
  (x - 80 = 10) :=
by
  intros x h₁ h₂
  rw ← h₂
  ring
  sorry

end jake_marks_difference_l470_470720


namespace find_sale_month4_l470_470869

-- Define sales for each month
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month5 : ℕ := 4500
def sale_month6 : ℕ := 1200
def avg_sale_per_month : ℕ := 5600

-- Define the total number of months
def num_months : ℕ := 6

-- Define the expression for total sales required
def total_sales_required : ℕ := avg_sale_per_month * num_months

-- Define the expression for total known sales
def total_known_sales : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6

-- State and prove the theorem:
theorem find_sale_month4 : sale_month1 = 5400 → sale_month2 = 9000 → sale_month3 = 6300 → 
                            sale_month5 = 4500 → sale_month6 = 1200 → avg_sale_per_month = 5600 →
                            num_months = 6 → (total_sales_required - total_known_sales = 8200) := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end find_sale_month4_l470_470869


namespace num_odd_sum_pairs_l470_470091

open Finset

theorem num_odd_sum_pairs : (card ((filter (λ p : ℕ × ℕ, ((p.1 + p.2) % 2 = 1)) ((range 10).product (range 10)))) = 25) :=
sorry

end num_odd_sum_pairs_l470_470091


namespace number_of_tiles_per_row_in_square_room_l470_470404

theorem number_of_tiles_per_row_in_square_room 
  (area_of_floor: ℝ)
  (tile_side_length: ℝ)
  (conversion_rate: ℝ)
  (h1: area_of_floor = 400) 
  (h2: tile_side_length = 8) 
  (h3: conversion_rate = 12) 
: (sqrt area_of_floor * conversion_rate / tile_side_length) = 30 := 
by
  sorry

end number_of_tiles_per_row_in_square_room_l470_470404


namespace number_of_tiles_per_row_l470_470406

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l470_470406


namespace sum_xyz_l470_470229

theorem sum_xyz (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 := 
by
  sorry

end sum_xyz_l470_470229


namespace correct_proposition_l470_470475

theorem correct_proposition :
  (∀ x : ℝ, exp x > 0) ∨ -- Option A: incorrect (negation is ∃ x, exp x ≤ 0)
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 + 2 * x ≥ (a : ℝ) * x → min (x^2 + 2 * x) (a * x)) ∨ -- Option B: incorrect
  (∀ x y : ℝ, x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1) ∨ -- Option C: correct
  (∀ a : ℝ, (a = -1 → (λ x, a * x^2 + 2 * x - 1) = 0) → -- Option D: incorrect (inverse is not true)
     (∀ b : ℝ, (λ x, b * x^2 + 2 * x - 1) = 0 → a = b)) := 
begin
 sorry
end

end correct_proposition_l470_470475


namespace speaker_is_male_nurse_l470_470708

noncomputable theory

open Classical

def total_members := 13

def number_of_nurses (num_nurses : ℕ) : Prop :=
  ∃ num_doctors: ℕ, 
    num_nurses ≥ num_doctors ∧
    ∃ male_doctors female_nurses male_nurses female_doctors: ℕ, 
      male_doctors > female_nurses ∧
      female_nurses > male_nurses ∧
      female_doctors ≥ 1 ∧ 
      num_nurses = male_nurses + female_nurses ∧
      num_doctors = male_doctors + female_doctors ∧
      male_doctors + female_nurses + male_nurses + female_doctors + 1 = total_members

theorem speaker_is_male_nurse :
  ∃ num_nurses : ℕ,
  number_of_nurses num_nurses ∧
  true ∧
  false ∧
  false :=
sorry

end speaker_is_male_nurse_l470_470708


namespace inequality_proof_l470_470353

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470353


namespace compare_s₁_s₂_l470_470282

variable {A B C G : Point}
variable (GA GB GC AB BC CA : ℝ)
variable [MetricSpace Point]

/-- G is the centroid of triangle ABC -/
axiom centroid_of_triangle : is_centroid G A B C

/-- s₁ is the sum of distances from G to vertices A, B, and C -/
def s₁ : ℝ := GA + GB + GC

/-- s₂ is the perimeter of triangle ABC -/
def s₂ : ℝ := AB + BC + CA

/-- The main theorem to prove: s₁ < s₂ / 3 -/
theorem compare_s₁_s₂
  (hG_s₁ : s₁ = GA + GB + GC)
  (hG_s₂ : s₂ = AB + BC + CA)
  (hG_centroid : is_centroid G A B C) :
  s₁ < s₂ / 3 :=
by
  -- Proof to be provided, hence using sorry.
  sorry

end compare_s₁_s₂_l470_470282


namespace yolk_count_proof_l470_470872

-- Define the conditions of the problem
def eggs_in_carton : ℕ := 12
def double_yolk_eggs : ℕ := 5
def single_yolk_eggs : ℕ := eggs_in_carton - double_yolk_eggs
def yolks_in_double_yolk_eggs : ℕ := double_yolk_eggs * 2
def yolks_in_single_yolk_eggs : ℕ := single_yolk_eggs
def total_yolks : ℕ := yolks_in_single_yolk_eggs + yolks_in_double_yolk_eggs

-- Stating the theorem to prove the total number of yolks is 17
theorem yolk_count_proof : total_yolks = 17 := 
by
  sorry

end yolk_count_proof_l470_470872


namespace count_integers_between_cubes_l470_470218

noncomputable def a := (10.1)^3
noncomputable def b := (10.4)^3

theorem count_integers_between_cubes : 
  ∃ (count : ℕ), count = 94 ∧ (1030.031 < a) ∧ (a < b) ∧ (b < 1124.864) := 
  sorry

end count_integers_between_cubes_l470_470218


namespace triangle_area_l470_470109

open Real

/-- Let A, B, C, D be points that form a square with side length 8 inches, such that 
    B is directly east of A, D is directly north of C, and P is a point inside the square 
    where  PA = PB, PC is perpendicular to FD, and PB is perpendicular to AC.
    Then the area of triangle APB is 32/5 square inches. --/
theorem triangle_area (A B C D P F : Point) (s : ℝ) (h_s : s = 8)
  (hA : A = (0, 0)) (hB : B = (s, 0)) (hC : C = (s, s)) (hD : D = (0, s)) (hF : F = (s/2, s))
  (hPA_PB : ∥P - A∥ = ∥P - B∥) (hPC_FD_perp : ∠(P - C) (F - D) = π / 2)
  (hPB_AC_perp : ∠(P - B) (A - C) = π / 2)
  : area (triangle A P B) = 32 / 5 :=
sorry

end triangle_area_l470_470109


namespace f_inequality_l470_470658

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a > 2 * Real.log a + 3 / 2 :=
sorry

end f_inequality_l470_470658


namespace largest_angle_of_tangent_and_secant_l470_470564

noncomputable def largest_angle (O A B C : Point) (circle : Circle) (line : Line) : Prop :=
  circle.passes_through A ∧
  circle.passes_through B ∧
  circle.tangent line ∧
  O ≠ A ∧
  O ≠ B ∧
  C = point_of_tangency circle line ∧
  (∀ P : Point, P ≠ C → ∃ M : Point, M ≠ C ∧ M ≠ point_of_tangency circle line ∧
    (∠ACB > ∠AMB))

-- The proof itself is omitted, as per instructions.
theorem largest_angle_of_tangent_and_secant (O A B C : Point) (line : Line) (circle : Circle) (h : largest_angle O A B C circle line) :
  ∀ M : Point, M ≠ C → ∃ Q : Point, ∠ACB > ∠AMB
:= by
  sorry

end largest_angle_of_tangent_and_secant_l470_470564


namespace wedge_volume_l470_470865

/--
A cylindrical log has a diameter of 16 inches. A wedge is cut from the log by making two planar cuts.
The first cut is perpendicular to the axis of the cylinder, and the plane of the second cut forms a
60 degree angle with the plane of the first cut. The intersection of these two planes touches the log
at exactly one point. Determine the volume of the wedge, expressed as mπ, where m is a positive integer.
-/
theorem wedge_volume (d radius : ℝ) (h : ℝ) (angle : ℝ) (m : ℤ) 
  (h_diameter : d = 16) 
  (h_radius : radius = d / 2)
  (h_height : h = d)
  (h_angle : angle = 60)
  (h_intersection : ∃ P : ℝ^3, True) :
  ∃ (m : ℤ), m > 0 ∧ Volume_of_wedge d angle = m * π := 
begin
  sorry
end

end wedge_volume_l470_470865


namespace determine_hyperbola_equation_l470_470960

noncomputable def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2) - (y^2 / b^2) = 1

def is_perpendicular_to (a b : ℝ) (m : ℝ) : Prop := 
  b / a = m

def distance_from_focus_to_line (a b c : ℝ) (d : ℝ) : Prop := 
  (c / 2) = d

theorem determine_hyperbola_equation : 
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧ 
  is_perpendicular_to a b (real.sqrt 3) ∧ 
  distance_from_focus_to_line a b (real.sqrt (a^2 + b^2)) 1 ∧
  hyperbola_equation a b 1 (real.sqrt 3) :=
sorry

end determine_hyperbola_equation_l470_470960


namespace Taimour_can_paint_alone_in_18_hours_l470_470843

variable (T : ℝ)

def Jamshid_time (T : ℝ) := T / 2
def together_time := 6
def combined_work_rate (T : ℝ) := 1/T + 2/T
def required_work_rate := 1 / together_time

theorem Taimour_can_paint_alone_in_18_hours
  (h1 : combined_work_rate T = required_work_rate) :
  T = 18 := by
  sorry

end Taimour_can_paint_alone_in_18_hours_l470_470843


namespace find_k_l470_470993

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (k : ℝ)

def vec_a : EuclideanSpace ℝ (Fin 3) := ![-1, 0, 1]
def vec_b : EuclideanSpace ℝ (Fin 3) := ![1, 2, 3]

theorem find_k (h : (k • vec_a - vec_b) ⬝ vec_b = 0) : k = 7 :=
sorry

end find_k_l470_470993


namespace inequality_holds_for_positive_x_l470_470763

theorem inequality_holds_for_positive_x (x : ℝ) (h : 0 < x) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 :=
sorry

end inequality_holds_for_positive_x_l470_470763


namespace product_of_solutions_eq_zero_l470_470143

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (3 * x + 5) / (6 * x + 5) = (5 * x + 4) / (9 * x + 4) → (x = 0 ∨ x = 8 / 3)) →
  0 * (8 / 3) = 0 :=
by
  intro h
  sorry

end product_of_solutions_eq_zero_l470_470143


namespace number_conversion_l470_470851

theorem number_conversion (a b c d : ℕ) : 
  4090000 = 409 * 10000 ∧ (a = 800000) ∧ (b = 5000) ∧ (c = 20) ∧ (d = 4) → 
  (a + b + c + d = 805024) :=
by
  sorry

end number_conversion_l470_470851


namespace bottles_recycled_l470_470601

theorem bottles_recycled (start_bottles : ℕ) (recycle_ratio : ℕ) (answer : ℕ)
  (h_start : start_bottles = 256) (h_recycle : recycle_ratio = 4) : answer = 85 :=
sorry

end bottles_recycled_l470_470601


namespace train_length_approx_250_l470_470083

-- Necessary definitions for the problem conditions
def speed_kmh : ℝ := 55 -- speed in km/hr
def platform_length : ℝ := 300 -- platform length in meters
def crossing_time : ℝ := 35.99712023038157 -- time in seconds

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Speed in m/s
def speed_mps : ℝ := speed_kmh * conversion_factor

-- Total distance covered during the crossing
def total_distance : ℝ := speed_mps * crossing_time

-- Length of the train
def train_length : ℝ := total_distance - platform_length

-- Main theorem to prove the length of the train is approximately 250 meters
theorem train_length_approx_250 : abs (train_length - 250) < 1 :=
by
  sorry -- Proof will be provided here

end train_length_approx_250_l470_470083


namespace integers_count_l470_470212

noncomputable section

def num_integers_satisfying_conditions (n : ℕ) : ℕ :=
  if h : n ≥ 2 then
    (n - 1)
  else
    0

theorem integers_count (n : ℕ) (h : n ≥ 2) :
  ∀ z1 z2 ... zn : ℂ, (|z1| = 1) ∧ (|z2| = 1) ∧ ... ∧ (|zn| = 1) → 
  (∀ k, k ∣ n → (∃ subset_k : finset ℕ, subset_k.card = k ∧ (∀ i j, (i < j → i ∈ subset_k → j ∈ subset_k → (z i + z j = 0) → ...))) →
  ∃ equal_spaced_on_unit_circle, 
  num_integers_satisfying_conditions n = n - 1 :=
by
  intros
  sorry

end integers_count_l470_470212


namespace grover_profit_l470_470208

theorem grover_profit (original_price : ℝ) (discount_rate : ℝ) (num_boxes : ℕ) 
                      (boxes_contents : Fin num_boxes → ℕ) (sell_price_per_mask : ℝ) :
  original_price = 8 →
  discount_rate = 0.20 →
  num_boxes = 3 →
  (λ i, boxes_contents i) = [25, 30, 35] →
  sell_price_per_mask = 0.60 →
  let discount_amount_per_box := original_price * discount_rate in
  let discounted_price_per_box := original_price - discount_amount_per_box in
  let total_cost := num_boxes * discounted_price_per_box in
  let total_masks := (Finset.univ : Finset (Fin num_boxes)).sum (λ i, boxes_contents i) in
  let total_revenue := total_masks * sell_price_per_mask in
  let total_profit := total_revenue - total_cost in
  total_profit = 34.80 :=
by
  intros
  sorry

end grover_profit_l470_470208


namespace min_possible_value_of_P_over_Q_l470_470231

-- Define conditions
variables {x P Q : ℝ}
variable (hP : x^2 + 1/x^2 = P)
variable (hQ : x^3 - 1/x^3 = Q)
variable (hP_pos : P > 0)
variable (hQ_pos : Q > 0)

-- Theorem statement
theorem min_possible_value_of_P_over_Q :
  ∃ r : ℝ, r = (x - 1/x) ∧ (P = r^2 + 2) ∧ (Q = r^3 + 3r) ∧ ∀ r > 0, (P / Q = 1 / real.sqrt 3) := sorry

end min_possible_value_of_P_over_Q_l470_470231


namespace find_t_l470_470936

noncomputable def magnitude_z1 (t : ℝ) : ℝ := Real.sqrt (t^2 + 48)

noncomputable def magnitude_z2 : ℝ := Real.sqrt (49 + 4)

theorem find_t (t : ℝ) :
  (magnitude_z1 t) * magnitude_z2 = 17 * Real.sqrt 13 ∧ t > 2 → t ≈ 4.78 :=
sorry

end find_t_l470_470936


namespace domain_of_composite_function_l470_470979

theorem domain_of_composite_function :
  (∀ x : ℝ, -1 < x ∧ x < 0 → f x) →
  (∀ y : ℝ, -1 < y ∧ y < - (1 / 2) → f (2 * y + 1)) :=
by
  intro h1
  sorry

end domain_of_composite_function_l470_470979


namespace trig_identity_second_quadrant_l470_470630

theorem trig_identity_second_quadrant (α : ℝ) 
  (h1 : α > π / 2) 
  (h2 : α < π) 
  (h3 : sin α > 0) 
  (h4 : cos α < 0) : 
  (2 * sin α / real.sqrt (1 - cos α^2) + real.sqrt (1 - sin α^2) / cos α) = 1 :=
by
  sorry

end trig_identity_second_quadrant_l470_470630


namespace number_of_tiles_per_row_l470_470407

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l470_470407


namespace number_of_alpha_satisfying_conditions_l470_470606

theorem number_of_alpha_satisfying_conditions :
  let S := {-2, -1, (1 : ℚ)/2, 1, 2, 3}
  (∃! α ∈ S, ∃ n m : ℝ, (0 < n) ∧ (0 < m) ∧ (n ≠ m) ∧ (x : ℝ) → 
    (x > 0 → (x ^ α > n) ∧ odd (λ x, x ^ α) ∧ 
    (∀ x y : ℝ, (0 < x) → (x < y) → (x^α < y^α)))) := 2 := 
sorry

end number_of_alpha_satisfying_conditions_l470_470606


namespace stratified_sampling_l470_470063

variable (total_students : ℕ) (first_year_students : ℕ) (male_second_year_students : ℕ) 
(variable (prob_female_second_year : ℚ) (sample_size : ℕ))

theorem stratified_sampling :
  total_students = 1000 →
  first_year_students = 380 →
  male_second_year_students = 180 →
  prob_female_second_year = 0.19 →
  sample_size = 100 →
  let female_second_year_students := prob_female_second_year * total_students in
  let total_second_year_students := female_second_year_students + male_second_year_students in
  let third_year_students := total_students - (first_year_students + total_second_year_students) in
  let num_third_year_drawn := (third_year_students / total_students) * sample_size in
  num_third_year_drawn = 25 :=
by
  sorry

end stratified_sampling_l470_470063


namespace cricket_team_captain_age_l470_470414

theorem cricket_team_captain_age
    (C W : ℕ)
    (h1 : W = C + 3)
    (h2 : (23 * 11) = (22 * 9) + C + W)
    : C = 26 :=
by
    sorry

end cricket_team_captain_age_l470_470414


namespace units_digit_n_squared_plus_two_pow_n_l470_470289

theorem units_digit_n_squared_plus_two_pow_n
  (n : ℕ)
  (h : n = 2018^2 + 2^2018) : 
  (n^2 + 2^n) % 10 = 5 := by
  sorry

end units_digit_n_squared_plus_two_pow_n_l470_470289


namespace division_of_neg_six_by_three_l470_470556

theorem division_of_neg_six_by_three : (-6) / 3 = -2 := by
  sorry

end division_of_neg_six_by_three_l470_470556


namespace inequality_proof_l470_470357

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470357


namespace find_a3_l470_470986

noncomputable def geometricSequenceSum (a q : ℚ) (n : ℕ) : ℚ :=
  (a * (q ^ n - 1)) / (q - 1)

noncomputable def a3 (a1 q : ℚ) : ℚ :=
  a1 * q ^ 2

theorem find_a3 (a1 q a3 : ℚ) (S3 S4 : ℚ) (h1 : q = 3) (h2 : S3 = geometricSequenceSum a1 q 3)
  (h3 : S4 = geometricSequenceSum a1 q 4) (h4 : S3 + S4 = 53 / 3) :
  a3 = 3 :=
by
  have eq_a1 : a1 = 1 / 3 := sorry
  have eq_a3 : a3 = a1 * q ^ 2 := sorry
  rw [eq_a3, eq_a1, h1]
  norm_num
  exact sorry

end find_a3_l470_470986


namespace negation_proposition_l470_470787

theorem negation_proposition (a b : ℝ) :
  (ab = 0 → a = 0 ∨ b = 0) ↔ (ab = 0 → ¬ (a = 0 ∨ b = 0)) :=
sorry

end negation_proposition_l470_470787


namespace original_salary_l470_470844

variable (x : ℝ)

def raisedSalaryByTenPercent (x : ℝ) : ℝ := 1.10 * x
def reducedSalaryByFivePercent (x : ℝ) : ℝ := 1.045 * x

theorem original_salary (final_salary : ℝ) (h1 : reducedSalaryByFivePercent x = final_salary) :
  x = 5000 :=
by
  have h2 : reducedSalaryByFivePercent 5000 = final_salary := by sorry
  rw [h2] at h1
  exact sorry

end original_salary_l470_470844


namespace trig_matrix_determinant_l470_470128

noncomputable def trig_matrix (θ φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [Real.sin θ * Real.cos φ, Real.sin θ * Real.sin φ, Real.cos θ],
    [Real.cos φ, -Real.sin φ, 0],
    [Real.cos θ * Real.cos φ, Real.cos θ * Real.sin φ, Real.sin θ]
  ]

theorem trig_matrix_determinant (θ φ : ℝ) : (trig_matrix θ φ).det = Real.cos θ :=
by
  sorry

end trig_matrix_determinant_l470_470128


namespace partial_fraction_decomposition_l470_470426

noncomputable def partial_fraction_product (A B C : ℤ) : ℤ :=
  A * B * C

theorem partial_fraction_decomposition:
  ∃ A B C : ℤ, 
  (∀ x : ℤ, (x^2 - 19 = A * (x + 2) * (x - 3) 
                    + B * (x - 1) * (x - 3) 
                    + C * (x - 1) * (x + 2) )) 
  → partial_fraction_product A B C = 3 :=
by
  sorry

end partial_fraction_decomposition_l470_470426


namespace distinct_triplets_count_l470_470047

open Nat

theorem distinct_triplets_count : 
  let count_distinct := (finset.range 601).count (λ n => 
    (floor (n / 2) ≠ floor ((n + 1) / 2)) ∨
    (floor (n / 3) ≠ floor ((n + 1) / 3)) ∨
    (floor (n / 5) ≠ floor ((n + 1) / 5))
  ) 
  in count_distinct = 440 
:= by
  sorry

end distinct_triplets_count_l470_470047


namespace tiles_in_each_row_l470_470393

theorem tiles_in_each_row (area : ℝ) (tile_side : ℝ) (feet_to_inches : ℝ) (number_of_tiles : ℕ) :
  area = 400 ∧ tile_side = 8 ∧ feet_to_inches = 12 → number_of_tiles = 30 :=
by
  intros h
  cases h with h_area h_rest
  cases h_rest with h_tile_side h_feet_to_inches
  sorry

end tiles_in_each_row_l470_470393


namespace cube_volume_derivative_equals_half_surface_area_l470_470167

-- Define the volume of a cube function
def V_cube (x : ℝ) : ℝ := x^3

-- Define the surface area of a cube function
def S_cube (x : ℝ) : ℝ := 6 * x^2

-- Formulate the theorem stating the desired conclusion
theorem cube_volume_derivative_equals_half_surface_area (x : ℝ) :
  deriv (V_cube x) = (1 / 2) * S_cube x :=
sorry

end cube_volume_derivative_equals_half_surface_area_l470_470167


namespace exists_lim_integral_nonincreasing_lim_x_fx_zero_l470_470277

open Set Filter

noncomputable def nonincreasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ y → f y ≤ f x

variable {f : ℝ → ℝ}
variable h1 : nonincreasing_function f
variable h2 : ∀ x ≥ 0, ∫ t in 0..x, f t < 1

-- Part (a)
theorem exists_lim_integral_nonincreasing :
  ∃ (L : ℝ), tendsto (λ x, ∫ t in (0:ℝ)..x, f t) at_top (nhds L) :=
sorry

-- Part (b)
theorem lim_x_fx_zero :
  tendsto (λ x, x * f x) at_top (nhds 0) :=
sorry

end exists_lim_integral_nonincreasing_lim_x_fx_zero_l470_470277


namespace polygon_sides_l470_470535

theorem polygon_sides (n : ℕ) (h : n ≥ 3) (sum_angles : (n - 2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end polygon_sides_l470_470535


namespace tiles_per_row_proof_l470_470400

def area_square_room : ℝ := 400 -- in square feet
def tile_size : ℝ := 8 / 12 -- each tile is 8 inches, converted to feet (8/12 feet)

noncomputable def number_of_tiles_per_row : ℕ :=
  let side_length := Math.sqrt area_square_room in
  let side_length_in_inch := side_length * 12 in
  Nat.floor (side_length_in_inch / tile_size)

theorem tiles_per_row_proof :
  number_of_tiles_per_row = 30 := by
  sorry

end tiles_per_row_proof_l470_470400


namespace diamond_initial_value_l470_470003

variables {r : ℕ} {d : Fin r → ℝ} {c : ℝ}

theorem diamond_initial_value
  (h1: 100 * (∑ i, (d i)^2) + 3 * c = 5000000)
  (h2: 25 * (∑ i, (d i)^2) + 3/2 * c = 2000000) :
  100 * (∑ i, (d i)^2) = 2000000 := by
    sorry

end diamond_initial_value_l470_470003


namespace range_a_correct_l470_470160

noncomputable def range_a : set ℝ :=
  {a : ℝ | a ≥ -2 ∧
          (let A := {x : ℝ | -2 ≤ x ∧ x ≤ a} in
           let B := {y : ℝ | ∃ x ∈ A, y = 2 * x + 3} in
           let C := {t : ℝ | ∃ x ∈ A, t = x^2} in
           C ⊆ B) }

theorem range_a_correct :
  range_a = {a | a ∈ set.Icc (1/2 : ℝ) 2 ∪ set.Ioo 2 3} :=
sorry

end range_a_correct_l470_470160


namespace find_smaller_number_l470_470794

-- Define the conditions as hypotheses and the goal as a proposition
theorem find_smaller_number (x y : ℕ) (h1 : x + y = 77) (h2 : x = 42 ∨ y = 42) (h3 : 5 * x = 6 * y) : x = 35 :=
sorry

end find_smaller_number_l470_470794


namespace points_in_triangle_l470_470453

theorem points_in_triangle {n : ℕ} (hn : n ≥ 3) 
  (points : Fin n → ℝ × ℝ)
  (hArea : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    let (xi, yi) := points i in
    let (xj, yj) := points j in
    let (xk, yk) := points k in
    (1 / 2) * abs (xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)) ≤ 1) :
  ∃ (T : ℝ × ℝ) (U : ℝ × ℝ) (V : ℝ × ℝ),
    let area := (1 / 2) * abs (
      (fst T) * (snd U - snd V) + (fst U) * (snd V - snd T) + (fst V) * (snd T - snd U))
    in
    (∀ (i : Fin n),
      let (x, y) := points i in
      ∃ (a b c : ℝ), a + b + c = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
      x = a * fst T + b * fst U + c * fst V ∧
      y = a * snd T + b * snd U + c * snd V) ∧
    area ≤ 4 :=
sorry

end points_in_triangle_l470_470453


namespace correct_answer_l470_470538

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def has_zero_point (f : ℝ → ℝ) : Prop := ∃ x : ℝ, f x = 0

def candidate_A := (λ x : ℝ, x^2 + 1)
def candidate_B := (λ x : ℝ, abs (log x))
def candidate_C := (λ x : ℝ, cos x)
def candidate_D := (λ x : ℝ, exp x - 1)

theorem correct_answer : is_even candidate_C ∧ has_zero_point candidate_C :=
by
  sorry

end correct_answer_l470_470538


namespace fire_brigade_allocation_l470_470504

theorem fire_brigade_allocation (sites fire_brigades : ℕ) (h_sites : sites = 3) (h_fire_brigades : fire_brigades = 4) :
  ∃ (allocations : ℕ), (∀ (site : ℕ), 1 ≤ site ∧ site ≤ sites → ∃ (brigades : ℕ), brigades ≥ 1) ∧ allocations = 36 := 
by {
  sorry
}

end fire_brigade_allocation_l470_470504


namespace geometric_sequence_problem_l470_470644

/-
Given:
- Sequence {a_n} is geometric with first term a_1 and common ratio q.
- S_n is the sum of the first n terms of the sequence.
- T_n = ∑_{k=1}^{n} S_k.
- Condition 1: a_2 * a_3 = 2 * a_1
- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5 / 4.
Prove:
- T_4 = 98
-/
noncomputable def geometric_seq_sum (a₁ q : ℝ) (n : ℕ) : ℝ := 
  a₁ * (1 - q^n) / (1 - q)

noncomputable def accumulated_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), geometric_seq_sum a₁ q k

theorem geometric_sequence_problem (a₁ q : ℝ)
  (h1 : (a₁ * q) * (a₁ * q^2) = 2 * a₁)
  (h2 : (a₁ * q^3 + 2 * (a₁ * q^6)) / 2 = 5 / 4) :
  accumulated_sum a₁ q 4 = 98 :=
    sorry

end geometric_sequence_problem_l470_470644


namespace triangle_with_different_colors_exists_l470_470262

theorem triangle_with_different_colors_exists (n : ℕ) (points : set point) 
    (color : point → color_type) (conn : point → set point) :
  (∃ (p_color : point → color_type), 
   (∀ p : point, p_color p = white ∨ p_color p = blue ∨ p_color p = black) ∧
   ∀ p : point, (conn p).card = n + 1 ∧
     (∀ q ∈ conn p, p_color q ≠ p_color p)) →
  ∃ (a b c : point),
    color a ≠ color b ∧ color b ≠ color c ∧ color c ≠ color a ∧
    a ∈ conn b ∧ b ∈ conn c ∧ c ∈ conn a :=
sorry

end triangle_with_different_colors_exists_l470_470262


namespace find_area_of_triangle_BMN_l470_470636

noncomputable def area_of_triangle_BMN (A B : ℝ) :=
  let circle_A := ∀ (x y : ℝ), (x^2 + y^2 - 2 * x = 0)
  let circle_B := ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0)
  let area := 3 / 2
  ∀ (M N : ℝ×ℝ), 
  (circle_A M.1 M.2) ∧ (circle_B M.1 M.2) → 
  (circle_A N.1 N.2) ∧ (circle_B N.1 N.2) → 
  |det (vec_M - vec_B) (vec_N - vec_B)| = area

theorem find_area_of_triangle_BMN : 
  ∀ (M N : (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x^2 + y^2 - 2 * x = 0) → (x = M.1) -> (y = M.2)) ∧ 
    (∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0) → (x = M.1) -> (y = M.2)) ->
    (∀ (x y : ℝ), (x^2 + y^2 - 2 * x = 0) → (x = N.1) -> (y = N.2)) ∧ 
    (∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0) → (x = N.1) -> (y = N.2)) → 
    let B := (-1, 2) in 
    let BM := (M.1 + 1, M.2 - 2) in 
    let BN := (N.1 + 1, N.2 - 2) in 
    |det BM BN| = 3 / 2 :=
by
 sorry

end find_area_of_triangle_BMN_l470_470636


namespace log_roots_equivalence_l470_470148

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 5 / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 5

theorem log_roots_equivalence :
  (x : ℝ) → (x = a ∨ x = b ∨ x = c) ↔ (x^3 - (a + b + c)*x^2 + (a*b + b*c + c*a)*x - a*b*c = 0) := by
  sorry

end log_roots_equivalence_l470_470148


namespace inequality_proof_l470_470354

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470354


namespace directrix_of_parabola_l470_470136

theorem directrix_of_parabola :
  ∀ (x : ℝ), (∃ k : ℝ, y = (x^2 - 8 * x + 16) / 8 → k = -2) :=
by
  sorry

end directrix_of_parabola_l470_470136


namespace sum_of_P_digits_l470_470283

def P (n : ℕ) : ℕ :=
  (n.digits 10).filter (· ≠ 0).prod

theorem sum_of_P_digits :
  (Finset.range 1000).sum P = 97335 := by
  sorry

end sum_of_P_digits_l470_470283


namespace original_sugar_amount_l470_470302

theorem original_sugar_amount (f : ℕ) (s t r : ℕ) (h1 : f = 5) (h2 : r = 10) (h3 : t = 14) (h4 : f * 2 = r):
  s = t / 2 := sorry

end original_sugar_amount_l470_470302


namespace inequality_inequality_l470_470365

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470365


namespace amoeba_count_after_ten_days_l470_470887

theorem amoeba_count_after_ten_days : 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  (initial_amoebas * splits_per_day ^ days) = 59049 := 
by 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  show (initial_amoebas * splits_per_day ^ days) = 59049
  sorry

end amoeba_count_after_ten_days_l470_470887


namespace value_of_x_l470_470739

theorem value_of_x :
  let x := (Real.cos (36 * Real.pi / 180)) - (Real.cos (72 * Real.pi / 180)) in
  x = 1 / 2 :=
by
  sorry

end value_of_x_l470_470739


namespace calculate_expression_l470_470903

theorem calculate_expression : 2 - (-3)^(3 - (-3)) = -727 :=
by
  -- proof goes here
  sorry

end calculate_expression_l470_470903


namespace sphere_surface_area_of_inscribed_in_cube_l470_470010

theorem sphere_surface_area_of_inscribed_in_cube (a : ℝ) (h : a = 4) : 
  let r := a / 2 in
  4 * π * r ^ 2 = 16 * π :=
by
  sorry

end sphere_surface_area_of_inscribed_in_cube_l470_470010


namespace rachel_total_time_spent_l470_470376

theorem rachel_total_time_spent 
    (writing_time_per_page : ℕ → ℕ := λ pages, pages * 1800)
    (researching_time : ℕ := 2700)
    (outline_time : ℕ := 15 * 60) 
    (brainstorming_time : ℕ := 1200)
    (total_pages_written : ℕ := 6)
    (break_time_per_page : ℕ → ℕ := λ pages, pages * 600)
    (editing_time : ℕ := 4500)
    (proofreading_time : ℕ := 1800) :
    (researching_time + outline_time + brainstorming_time + writing_time_per_page total_pages_written +
    break_time_per_page total_pages_written + editing_time + proofreading_time) / 3600 = 7.0833 := 
by
  sorry

end rachel_total_time_spent_l470_470376


namespace inequality_proof_l470_470336

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470336


namespace f_decreasing_exists_x0_l470_470193

noncomputable def f (x : ℝ) : ℝ := x / (real.exp x - 1)

theorem f_decreasing (x : ℝ) (hx : x > 0) : 
  ∀ x1 x2 : ℝ, x1 > x2 → f x1 ≤ f x2 :=
sorry

theorem exists_x0 (a : ℝ) (ha : a > 2) : 
  ∃ x0 > 0, f x0 < a / (real.exp x0 + 1) :=
sorry

end f_decreasing_exists_x0_l470_470193


namespace correct_unit_prices_correct_minimum_cost_l470_470813

noncomputable def unit_prices (a b : ℕ) : Prop :=
  a + 2 * b = 110 ∧ 2 * a + 3 * b = 190

noncomputable def minimum_cost (a b : ℕ) (x : ℕ) (total : ℕ) : Prop :=
  let w := (a - 5) * x + b * (total - x) in
  a = 50 ∧ b = 30 ∧ total = 100 ∧ 
  (∀ x, 3 * x ≥ total - x) ∧
  w = 3375

theorem correct_unit_prices : ∃ a b, unit_prices a b :=
begin
  use 50,
  use 30,
  sorry,
end

theorem correct_minimum_cost : ∃ a b x total, minimum_cost a b x total :=
begin
  use 50,
  use 30,
  use 25,
  use 100,
  sorry,
end

end correct_unit_prices_correct_minimum_cost_l470_470813


namespace c_share_l470_470837

theorem c_share (x : ℕ) (a b c d : ℕ) 
  (h1: a = 5 * x)
  (h2: b = 3 * x)
  (h3: c = 2 * x)
  (h4: d = 3 * x)
  (h5: a = b + 1000): 
  c = 1000 := 
by 
  sorry

end c_share_l470_470837


namespace inequality_xyz_l470_470332

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470332


namespace problem_statement_l470_470114

namespace CoinFlipping

/-- 
Define the probability that Alice and Bob both get the same number of heads
when flipping three coins where two are fair and one is biased with a probability
of 3/5 for heads. We aim to calculate p + q where p/q is this probability and 
output the final result - p + q should equal 263.
-/
def same_heads_probability_sum : ℕ :=
  let p := 63
  let q := 200
  p + q

theorem problem_statement : same_heads_probability_sum = 263 :=
  by
  -- proof to be filled in
  sorry

end CoinFlipping

end problem_statement_l470_470114


namespace angle_between_vectors_l470_470159

open Real EuclideanGeometry

variables (a b : EuclideanSpace ℝ (fin 2))

-- definitions based on the problem conditions
def vector_a_magnitude (a : EuclideanSpace ℝ (fin 2)) : Prop :=
  ‖a‖ = 1

def vector_b_magnitude (b : EuclideanSpace ℝ (fin 2)) : Prop :=
  ‖b‖ = sqrt 2

def vectors_perpendicular (a b : EuclideanSpace ℝ (fin 2)) : Prop :=
  a ⟂ (a - b)

-- theorem statement to be proven
theorem angle_between_vectors (ha : vector_a_magnitude a) (hb : vector_b_magnitude b) 
  (hperp : vectors_perpendicular a b) : angle a b = π / 4 :=
sorry

end angle_between_vectors_l470_470159


namespace number_of_valid_pairs_l470_470522

theorem number_of_valid_pairs
  (a b : ℕ)
  (h_ba : b > a)
  (h_area : b ≥ 4 ∧ a ≥ 4)
  (h_equation : a * b = 3 * (a - 4) * (b - 4)) :
  (∃ (s : finset (ℕ × ℕ)), s = finset.filter (λ p, b > a) 
  (finset.product (finset.range (b + 1)) (finset.range (a + 1))) ∧
  s.card = 4) := by
  sorry

end number_of_valid_pairs_l470_470522


namespace claudia_charge_per_class_l470_470560

/-- Claudia offers art classes and earns 300 dollars over the weekend,
    with 20 kids attending Saturday's class and 10 kids attending Sunday's class.
    We need to prove that Claudia charges 10 dollars per class. -/
theorem claudia_charge_per_class : 
  ∃ x : ℝ, (20 * x + 10 * x = 300) ∧ x = 10 :=
by
  use 10
  split
  . norm_num
  . rfl

end claudia_charge_per_class_l470_470560


namespace max_colored_cells_in_100x100_grid_l470_470250

theorem max_colored_cells_in_100x100_grid :
  ∃ (N : ℕ), (N = 198) ∧ ∀ (grid : list (list bool)) (H : grid.length = 100) (H' : ∀ row, row ∈ grid → row.length = 100) (H_unique: ∀ i j k, ((grid i j = tt) ∧ (grid i k = tt) → j = k) ∧ ((grid i j = tt) ∧ (grid k j = tt) → i = k)), 
  (count_colored_cells grid) ≤ N := sorry

noncomputable def count_colored_cells (grid : list (list bool)) :=
  grid.foldl (λ acc row, acc + (row.foldl (λ acc' cell, if cell then acc' + 1 else acc') 0)) 0

end max_colored_cells_in_100x100_grid_l470_470250


namespace color_change_probability_is_correct_l470_470893

-- Given definitions
def cycle_time : ℕ := 45 + 5 + 10 + 40

def favorable_time : ℕ := 5 + 5 + 5

def probability_color_change : ℚ := favorable_time / cycle_time

-- Theorem statement to prove the probability
theorem color_change_probability_is_correct :
  probability_color_change = 0.15 := 
sorry

end color_change_probability_is_correct_l470_470893


namespace find_a_l470_470940

def f (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem find_a : ∃ a : ℝ, (a > -1) ∧ (a < 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ 2 → f x ≤ f a) ∧ f a = 15 / 4 :=
by
  exists -1 / 2
  sorry

end find_a_l470_470940


namespace inequality_holds_for_real_numbers_l470_470325

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470325


namespace count_valid_m_l470_470377

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_m (m : ℕ) : Prop :=
  m > 1 ∧ is_divisor m 480 ∧ (480 / m) > 1

theorem count_valid_m : (∃ m, valid_m m) → Nat.card {m // valid_m m} = 22 :=
by sorry

end count_valid_m_l470_470377


namespace tangent_slope_at_1_l470_470008

noncomputable def f : ℝ → ℝ :=
  λ x, x^3 - x + 3

theorem tangent_slope_at_1 :
  deriv f 1 = 2 :=
by 
  sorry

end tangent_slope_at_1_l470_470008


namespace hyperbola_asymptotes_l470_470987

theorem hyperbola_asymptotes 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (M_x M_y : ℝ) 
  (M_on_hyperbola : M_y^2 / a^2 - M_x^2 / b^2 = 1) 
  (MF_3DF : ∀ D_x D_y, 
    (D_x^2 + (D_y - (c / 3))^2 = (a^2 / 9)) ∧
    ((M_x, M_y) = (3 * D_x, -3 * D_y))) : 
  (b = 2 * a) → (∀ y x, y = ± (1 / 2) * x) :=
begin
  sorry
end

end hyperbola_asymptotes_l470_470987


namespace sqrt_sum_eq_seven_l470_470688

theorem sqrt_sum_eq_seven (x : ℝ) (h : sqrt (5 + x) + sqrt (20 - x) = 7) : (5 + x) * (20 - x) = 144 :=
sorry

end sqrt_sum_eq_seven_l470_470688


namespace vision_condition_proved_l470_470806

-- Given conditions for the problem
def vision_survey (n : ℕ) (freq_first4 : ℕ → ℕ) (freq_last6 : ℕ → ℕ) 
                   (max_group : ℕ) (freq_vision : ℝ) : Prop :=
  n = 200 ∧ 
  (∀ i : ℕ, i < 4 → ∃ r : ℕ, ∀ j : ℕ, j < 3 → freq_first4 j * r = freq_first4 (j + 1)) ∧
  (∀ i : ℕ, 4 ≤ i → i < 10 → ∃ d : ℕ, ∀ j : ℕ, j < 5 → freq_last6 j + d = freq_last6 (j + 1)) ∧
  max_group = 54 ∧ 
  freq_vision = 0.78

-- Statement to prove
theorem vision_condition_proved : ∃ (a b : ℕ × ℝ), 
  let a := (54 : ℕ) in
  let b := (0.78 : ℝ) in
  vision_survey 200 (λ i, if i < 4 then 54 / (2 ^ i) else 0) (λ i, if 4 ≤ i ∧ i < 10 then 64 + (i - 4) * 4 else 0) 54 0.78 :=
by {
  existsi (54, 0.78),
  exact sorry,
}

end vision_condition_proved_l470_470806


namespace percentage_no_job_diploma_l470_470254

def percentage_with_university_diploma {total_population : ℕ} (has_diploma : ℕ) : ℕ :=
  (has_diploma / total_population) * 100

variables {total_population : ℕ} (p_no_diploma_and_job : ℕ) (p_with_job : ℕ) (p_diploma : ℕ)

axiom percentage_no_diploma_job :
  p_no_diploma_and_job = 10

axiom percentage_with_job :
  p_with_job = 40

axiom percentage_diploma :
  p_diploma = 39

theorem percentage_no_job_diploma :
  ∃ p : ℕ, p = (9 / 60) * 100 := sorry

end percentage_no_job_diploma_l470_470254


namespace shanghai_world_expo_ticket_sales_l470_470056

theorem shanghai_world_expo_ticket_sales :
  ∃ (x y : ℕ), x + y = 1200 ∧ 200 * x + 120 * y = 216000 ∧ x = 900 ∧ y = 300 :=
by {
  use 900,
  use 300,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { refl },
  { refl }
}

end shanghai_world_expo_ticket_sales_l470_470056


namespace trigonometric_identity_l470_470921

theorem trigonometric_identity : 
  (Real.cos (15 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) - Real.cos (75 * Real.pi / 180) * Real.sin (105 * Real.pi / 180))
  = -1 / 2 :=
by
  sorry

end trigonometric_identity_l470_470921


namespace quadratic_inequality_solution_set_l470_470146

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - 3 * x - 2 ≥ 0} = (set.Iic (-1/2) ∪ set.Ici 2) :=
by
  sorry

end quadratic_inequality_solution_set_l470_470146


namespace max_edges_intersected_by_plane_l470_470001

noncomputable def max_intersected_edges (E : ℕ) : ℕ :=
  if E = 99 then 66 else 0

theorem max_edges_intersected_by_plane :
  ∀ (E : ℕ), E = 99 → max_intersected_edges E = 66 :=
by
  intros E hE
  simp [max_intersected_edges, hE]
  exact sorry

end max_edges_intersected_by_plane_l470_470001


namespace greatest_x_solution_l470_470138

theorem greatest_x_solution : ∀ x : ℝ, 
    (x ≠ 6) ∧ (x ≠ -4) ∧ ((x^2 - 3*x - 18)/(x-6) = 2 / (x+4)) → x ≤ -2 :=
by
  sorry

end greatest_x_solution_l470_470138


namespace moving_circle_fixed_point_l470_470182

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

def tangent_line (c : ℝ × ℝ) (r : ℝ) : Prop :=
  abs (c.1 + 1) = r

theorem moving_circle_fixed_point :
  ∀ (c : ℝ × ℝ) (r : ℝ),
    parabola c →
    tangent_line c r →
    (1, 0) ∈ {p : ℝ × ℝ | dist c p = r} :=
by
  intro c r hc ht
  sorry

end moving_circle_fixed_point_l470_470182


namespace inequality_holds_for_all_real_numbers_l470_470369

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470369


namespace tetrahedron_volume_l470_470536

noncomputable def volume_tetrahedron (s : ℝ) : ℝ :=
  s^3 * Real.sqrt 2 / 12

theorem tetrahedron_volume (s : ℝ) : 
  (∀ Δ : Set (Set ℝ), (Δ.card = 4) ∧ 
  (∀ face ∈ Δ, ∃ a b c : ℝ, face = {a, b, c} ∧ a^2 + b^2 = c^2) ∧ 
  (∃ edge_set ∈ Δ, edge_set.card = 3 ∧ (∀ e ∈ edge_set, ∃ a : ℝ, a = s))) →
  volume_tetrahedron s = s^3 * Real.sqrt 2 / 12 := 
sorry

end tetrahedron_volume_l470_470536


namespace prove_inequality_l470_470626

theorem prove_inequality
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0)
  (h₄ : a ≤ b)
  (h₅ : b ≤ c)
  (h₆ : c ≤ d)
  (h₇ : a + b + c + d ≥ 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 :=
by
  sorry

end prove_inequality_l470_470626


namespace range_of_a_l470_470629

-- Define sets A and B
def A : set ℝ := {x : ℝ | x < 3}
def B (a : ℝ) : set ℝ := {x : ℝ | a < x}

-- State the proof problem
theorem range_of_a (a : ℝ) (h : A ∪ B a = set.univ) : a < 3 :=
sorry

end range_of_a_l470_470629


namespace problem_a_problem_b_l470_470910

variables {A B C D E P Q T : Type*}

-- Note: Actual types of A, B, C, D, E, P, Q, T would be points in a geometric context
-- Here, we treat them abstractly for illustration purposes.

-- Given conditions as hypotheses
hypothesis h1 : on_circle A B C D E
hypothesis h2 : dist A B = dist B C
hypothesis h3 : dist C D = dist D E
hypothesis h4 : is_intersection_point P (line_through A D) (line_through B E)
hypothesis h5 : is_intersection_point Q (line_through A C) (line_through B D)
hypothesis h6 : is_intersection_point T (line_through B D) (line_through C E)

-- Prove statements
theorem problem_a : concyclic A B P Q :=
sorry

theorem problem_b : is_isosceles_triangle P Q T :=
sorry

end problem_a_problem_b_l470_470910


namespace sum_first_75_odd_numbers_l470_470924

theorem sum_first_75_odd_numbers : (75^2) = 5625 :=
by
  sorry

end sum_first_75_odd_numbers_l470_470924


namespace train_cross_bridge_time_l470_470839

-- Length of the train in meters
def train_length : ℕ := 165

-- Length of the bridge in meters
def bridge_length : ℕ := 660

-- Speed of the train in kmph
def train_speed_kmph : ℕ := 54

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℚ := 5 / 18

-- Total distance to be traveled by the train to cross the bridge
def total_distance : ℕ := train_length + bridge_length

-- Speed of the train in meters per second (m/s)
def train_speed_mps : ℚ := train_speed_kmph * kmph_to_mps

-- Time taken for the train to cross the bridge (in seconds)
def time_to_cross_bridge : ℚ := total_distance / train_speed_mps

-- Prove that the time taken for the train to cross the bridge is 55 seconds
theorem train_cross_bridge_time : time_to_cross_bridge = 55 := by
  -- Proof goes here
  sorry

end train_cross_bridge_time_l470_470839


namespace z_value_l470_470687

theorem z_value (x y z : ℝ) (h : 1 / x + 1 / y = 2 / z) : z = (x * y) / 2 :=
by
  sorry

end z_value_l470_470687


namespace problem_equivalence_l470_470232

variable (j k l m : ℝ)

-- Define the conditions given in the problem
def cond1 : Prop := 1.25 * j = 0.25 * k
def cond2 : Prop := 1.5 * k = 0.5 * l
def cond3 : Prop := 0.2 * m = 7 * j

-- Define the statement to be proven
def statement : Prop := 1.75 * l = 0.75 * m

-- Prove the statement using Lean's theorem prover
theorem problem_equivalence (h1 : cond1) (h2 : cond2) (h3 : cond3) : statement :=
by
  sorry

end problem_equivalence_l470_470232


namespace age_sum_l470_470485

variable (b : ℕ)
variable (a : ℕ := b + 2)
variable (c : ℕ := b / 2)

theorem age_sum : b = 10 → a + b + c = 27 :=
by
  intros h
  rw [h]
  sorry

end age_sum_l470_470485


namespace polynomial_distinct_positive_roots_l470_470271

theorem polynomial_distinct_positive_roots (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^3 + a * x^2 + b * x - 1) 
(hroots : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0) : 
  P (-1) < -8 := 
by
  sorry

end polynomial_distinct_positive_roots_l470_470271


namespace tiles_per_row_l470_470388

noncomputable def area_square_room : ℕ := 400
noncomputable def side_length_feet : ℕ := nat.sqrt area_square_room
noncomputable def side_length_inches : ℕ := side_length_feet * 12
noncomputable def tile_size : ℕ := 8

theorem tiles_per_row : (side_length_inches / tile_size) = 30 := 
by
  have h1: side_length_feet = 20 := by sorry
  have h2: side_length_inches = 240 := by sorry
  have h3: side_length_inches / tile_size = 30 := by sorry
  exact h3

end tiles_per_row_l470_470388


namespace min_eq_implies_dist_eq_zero_dist_eq_zero_implies_inter_nonempty_l470_470944

section
variable {ι : Type*} [LinearOrder ι]

def dist (A B : Finset ι) : ι := Finset.min' (Finset.image (λ p : ι × ι, |p.1 - p.2|) (A.product B)) sorry

theorem min_eq_implies_dist_eq_zero {A B : Finset ι} (h : A.nonempty ∧ B.nonempty) (h₁ : A.min' (Finset.nonempty_of_ne_empty h.1) = B.min' (Finset.nonempty_of_ne_empty h.2)) : 
  dist A B = 0 := sorry

theorem dist_eq_zero_implies_inter_nonempty {A B : Finset ι} (h : A.nonempty ∧ B.nonempty) (h₁ : dist A B = 0) : 
  A ∩ B ≠ ∅ := sorry
end

end min_eq_implies_dist_eq_zero_dist_eq_zero_implies_inter_nonempty_l470_470944


namespace three_point_five_six_as_fraction_l470_470029

theorem three_point_five_six_as_fraction : (356 / 100 : ℝ) = (89 / 25 : ℝ) :=
begin
  sorry
end

end three_point_five_six_as_fraction_l470_470029


namespace count_integers_between_powers_l470_470213

noncomputable def power (a : ℝ) (b : ℝ) : ℝ := a^b

theorem count_integers_between_powers:
  let a := 10
  let b1 := 0.1
  let b2 := 0.4
  have exp1 : Float := (a + b1)
  have exp2 : Float := (a + b2)
  have n1 : ℤ := exp1^3.ceil
  have n2 : ℤ := exp2^3.floor
  n2 - n1 + 1 = 94 := 
begin
  sorry
end

end count_integers_between_powers_l470_470213


namespace triangle_ABC_B_eq_triangle_ABC_area_l470_470266

open Real

def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * b * sin C

theorem triangle_ABC_B_eq 
  (a b c A B C : ℝ) 
  (h₁ : c = 6) 
  (h₂ : sin A - sin C = sin (A - B)) :
  B = π / 3 :=
sorry

theorem triangle_ABC_area 
  (a b c A B C : ℝ) 
  (h₁ : c = 6) 
  (h₂ : sin A - sin C = sin (A - B)) 
  (h₃ : B = π / 3) 
  (h₄ : b = 2 * sqrt 7) :
  area_of_triangle a b c A B C = 3 * sqrt 3 ∨ area_of_triangle a b c A B C = 6 * sqrt 3 :=
sorry

end triangle_ABC_B_eq_triangle_ABC_area_l470_470266


namespace government_profit_l470_470421

theorem government_profit (num_people : ℕ) (percentage : ℝ) (stimulus_per_person : ℝ) (return_multiplier : ℝ) 
  (h_num_people : num_people = 1000)
  (h_percentage : percentage = 0.2)
  (h_stimulus_per_person : stimulus_per_person = 2000)
  (h_return_multiplier : return_multiplier = 5) :
  5 * (0.2 * 1000 * 2000) - (0.2 * 1000 * 2000) = 1600000 :=
by {
  rw [h_num_people, h_percentage, h_stimulus_per_person, h_return_multiplier],
  norm_num,
  exact rfl,
}

end government_profit_l470_470421


namespace units_produced_today_l470_470948

theorem units_produced_today (n : ℕ) (X : ℕ) 
  (h1 : n = 9) 
  (h2 : (360 + X) / (n + 1) = 45) 
  (h3 : 40 * n = 360) : 
  X = 90 := 
sorry

end units_produced_today_l470_470948


namespace mean_values_are_two_l470_470777

noncomputable def verify_means (a b : ℝ) : Prop :=
  (a + b) / 2 = 2 ∧ 2 / ((1 / a) + (1 / b)) = 2

theorem mean_values_are_two (a b : ℝ) (h : verify_means a b) : a = 2 ∧ b = 2 :=
  sorry

end mean_values_are_two_l470_470777


namespace percentage_of_seniors_is_90_l470_470754

-- Definitions of the given conditions
def total_students : ℕ := 120
def students_in_statistics : ℕ := total_students / 2
def seniors_in_statistics : ℕ := 54

-- Statement to prove
theorem percentage_of_seniors_is_90 : 
  ( seniors_in_statistics / students_in_statistics : ℚ ) * 100 = 90 := 
by
  sorry  -- Proof will be provided here.

end percentage_of_seniors_is_90_l470_470754


namespace chord_length_cut_by_line_from_curve_l470_470154

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + 4 * t, -1 - 3 * t)

noncomputable def curve_polar (theta : ℝ) : ℝ :=
  sqrt 2 * cos (theta + π / 4)

theorem chord_length_cut_by_line_from_curve :
  ∃ t theta x y,
    (x = 1 + 4 * t) ∧ (y = -1 - 3 * t) ∧
    ((x - 1/2)^2 + (y + 1/2)^2 = 1 / 2) ∧
    sqrt 2 * cos (theta + π / 4) = sqrt (1 / 2) ∧
    2 * sqrt ((1 / 2)^2 - (1 / 10)^2) = 7 / 5 :=
sorry

end chord_length_cut_by_line_from_curve_l470_470154


namespace hyperbola_eccentricity_l470_470619

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_intersect : ∃ x : ℝ, (x^2 - (b/a)*x + 1 = 0) ∧ (∃! y : ℝ, y = x^2 + 1 ∧ y = (b/a)*x)) :
  let e := Real.sqrt (1 + (b/a)^2) in
  e = Real.sqrt 5 :=
sorry

end hyperbola_eccentricity_l470_470619


namespace smallest_number_meets_conditions_l470_470145

-- Define the predicate for our problem:
def satisfies_property (N : ℕ) : Prop :=
  N > 9 ∧ N % 7 ≠ 0 ∧
  ∀ d, d ∈ Nat.digits 10 N → ∀ m, m = Nat.digits 10 N ∨ m ∉ Nat.digits 10 N -> 
  let new_number := Nat.of_digits 10 (Nat.digits 10 N).replace (λ x, if x = d then 7 else x) in
  new_number % 7 = 0

-- Define the exact number we found:
def N_candidate := 13264513

-- Theorem statement:
theorem smallest_number_meets_conditions : satisfies_property N_candidate := 
by 
  sorry

end smallest_number_meets_conditions_l470_470145


namespace age_sum_is_27_l470_470487

noncomputable def a : ℕ := 12
noncomputable def b : ℕ := 10
noncomputable def c : ℕ := 5

theorem age_sum_is_27
  (h1: a = b + 2)
  (h2: b = 2 * c)
  (h3: b = 10) :
  a + b + c = 27 :=
  sorry

end age_sum_is_27_l470_470487


namespace exists_edge_contraction_preserves_three_connected_l470_470714

variable {V : Type} [Fintype V]

-- 3-connected graph definition
def is_three_connected (G : SimpleGraph V) : Prop :=
  ∀ (S : Set V), S.card < 3 → (G.vertexSet \ S).connected

-- Definition of K4
def is_K4 (G : SimpleGraph V) : Prop :=
  ∀ v, ∃ subset : Finset V, subset.card = 4 ∧ G.inducedGraph subset = G.completeGraph

-- The theorem statement
theorem exists_edge_contraction_preserves_three_connected (G : SimpleGraph V) :
  is_three_connected G ∧ ¬ is_K4 G → ∃ e : sym2 V, is_three_connected (G.deleteEdges {e}) :=
by
  sorry

end exists_edge_contraction_preserves_three_connected_l470_470714


namespace integral_solution_l470_470855

noncomputable def integral_problem : Real :=
  ∫ x in 1..2, Real.exp x + x⁻¹

theorem integral_solution : integral_problem = Real.exp 2 - Real.exp 1 + Real.log 2 :=
by sorry

end integral_solution_l470_470855


namespace tangent_line_eq_range_of_a_l470_470650

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x + 1

-- Define the tangent line problem statement
theorem tangent_line_eq (a : ℝ) :
  (∃ (m : ℝ) (b : ℝ), (∀ x, f a x = m * (x - 1) + b) ∧ m = a + 1) := 
sorry

-- Define the inequality problem statement
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 0) ↔ (a ≤ -1) :=
sorry

end tangent_line_eq_range_of_a_l470_470650


namespace perfect_square_divisors_count_l470_470220

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def product_of_factorials : Nat := factorial 1 * factorial 2 * factorial 3 * factorial 4 * factorial 5 *
                                   factorial 6 * factorial 7 * factorial 8 * factorial 9 * factorial 10

def count_perfect_square_divisors (n : Nat) : Nat := sorry -- This would involve the correct function implementation.

theorem perfect_square_divisors_count :
  count_perfect_square_divisors product_of_factorials = 2160 :=
sorry

end perfect_square_divisors_count_l470_470220


namespace solve_trigonometric_equation_l470_470771

theorem solve_trigonometric_equation :
  ∃ n : ℤ, ∀ x : ℝ, (sin x ^ 3 + 6 * cos x ^ 3 + (1 / sqrt 2) * sin (2 * x) * sin (x + π / 4) = 0 ↔ x = -arctan 2 + n * π) :=
sorry

end solve_trigonometric_equation_l470_470771


namespace min_value_of_f_l470_470653

-- Define the function f(x) as given in the conditions
def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 - Real.sqrt 3 * Real.sin (2 * x)

-- State the problem as a theorem to be proved
theorem min_value_of_f : ∃ x : ℝ, f x = -1 := sorry

end min_value_of_f_l470_470653


namespace f_inequality_l470_470660

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a > 2 * Real.log a + 3 / 2 :=
sorry

end f_inequality_l470_470660


namespace solve_inequality_l470_470773

/-- 
Given the inequality 
(3x + 4 - 2sqrt(2x^2 + 7x + 3))(abs(x^2 - 4x + 2) - abs(x - 2)) ≤ 0,
prove that the solution set is 
x ∈ (-∞, -3] ∪ [0, 1] ∪ {2} ∪ [3, 4]
-/
theorem solve_inequality :
  {x : ℝ | (3 * x + 4 - 2 * sqrt (2 * x^2 + 7 * x + 3)) * 
            (abs (x^2 - 4 * x + 2) - abs (x - 2)) ≤ 0} = 
  {x : ℝ | x ∈ (-∞, -3] ∪ [0, 1] ∪ {2} ∪ [3, 4]} := 
sorry

end solve_inequality_l470_470773


namespace carson_gold_stars_yesterday_l470_470103

def goldStarsEarnedYesterday (total: ℕ) (earnedToday: ℕ) : ℕ :=
  total - earnedToday

theorem carson_gold_stars_yesterday :
  goldStarsEarnedYesterday 15 9 = 6 :=
by 
  sorry

end carson_gold_stars_yesterday_l470_470103


namespace sin_alpha_value_l470_470607

theorem sin_alpha_value (α : ℝ) (h1 : Real.tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_value_l470_470607


namespace grandma_red_bacon_bits_l470_470551

theorem grandma_red_bacon_bits:
  ∀ (mushrooms cherryTomatoes pickles baconBits redBaconBits : ℕ),
    mushrooms = 3 →
    cherryTomatoes = 2 * mushrooms →
    pickles = 4 * cherryTomatoes →
    baconBits = 4 * pickles →
    redBaconBits = 1 / 3 * baconBits →
    redBaconBits = 32 := 
by
  intros mushrooms cherryTomatoes pickles baconBits redBaconBits
  intros h1 h2 h3 h4 h5
  sorry

end grandma_red_bacon_bits_l470_470551


namespace find_line_l2_eqn_l470_470172

/-- Given line l1: 2 * x - y + 1 = 0, and line l2 passes through the point (1,1)
with its angle of inclination being twice that of line l1's angle of inclination,
then the equation of line l2 is 4 * x + 3 * y - 7 = 0. -/
theorem find_line_l2_eqn :
  ∃ l2 : ℝ × ℝ × ℝ, 
    let (a, b, c) := (2, -1, 1) in 
    let l1 := (a * x + b * y + c = 0) in 
    l2 = (4 * x + 3 * y - 7 = 0) ∧ 
    (x, y) ∈ l2 := sorry

end find_line_l2_eqn_l470_470172


namespace three_digit_odd_number_is_803_l470_470243

theorem three_digit_odd_number_is_803 :
  ∃ (a b c : ℕ), 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ c % 2 = 1 ∧
  100 * a + 10 * b + c = 803 ∧ (100 * a + 10 * b + c) / 11 = a^2 + b^2 + c^2 :=
by {
  sorry
}

end three_digit_odd_number_is_803_l470_470243


namespace inequality_sum_l470_470637

variables {a b c : ℝ}

theorem inequality_sum (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end inequality_sum_l470_470637


namespace perpendicular_lines_slope_parallel_lines_slope_l470_470989

def line_1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, x + a * y - a = 0
def line_2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x - (2 * a - 3) * y + a - 2 = 0

theorem perpendicular_lines_slope (a : ℝ) : 
  (a ≠ 0) → 
  let m1 := - (1 : ℝ) / a in
  let m2 := (a : ℝ) / (2 * a - 3) in
  m1 * m2 = -1 → 
  a = 2 := 
sorry

theorem parallel_lines_slope (a : ℝ) :
  let m1 := - (1 : ℝ) / a in
  let m2 := (a : ℝ) / (2 * a - 3) in
  m1 = m2 → 
  a = -3 := 
sorry

end perpendicular_lines_slope_parallel_lines_slope_l470_470989


namespace neither_sufficient_nor_necessary_l470_470793

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬(-1 < x ∧ x < 2 → |x - 2| < 1) ∧ ¬(|x - 2| < 1 → -1 < x ∧ x < 2) :=
by
  sorry

end neither_sufficient_nor_necessary_l470_470793


namespace inequality_of_sums_l470_470316

theorem inequality_of_sums (x y α : ℝ) (h : sqrt (1 + x) + sqrt (1 + y) = 2 * sqrt (1 + α)) : 
  x + y ≥ 2 * α :=
sorry

end inequality_of_sums_l470_470316


namespace extra_minutes_l470_470451

def man_usual_speed (S : ℝ) := S
def distance (D : ℝ) := D
def usual_time (T : ℝ) := T = 8
def slower_speed (S : ℝ) := (1 / 4) * S
def new_time (T' : ℝ) (S : ℝ) (T : ℝ) := T' = 4 * T
def extra_time (T' : ℝ) (T : ℝ) := T' - T = 24

theorem extra_minutes {S D T T' : ℝ} (h1 : T = 8) (h2 : T' = 4 * T) : T' - T = 24 :=
sorry

end extra_minutes_l470_470451


namespace monotonicity_case1_monotonicity_case2_lower_bound_fx_l470_470661

noncomputable def f (x a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 {x a : ℝ} (h : a ≤ 0) : 
  ∀ x, (differentiable ℝ (λ x, f x a) ∧ deriv (λ x, f x a) x ≤ -1) :=
sorry

theorem monotonicity_case2 {x a : ℝ} (h : 0 < a) : 
  ∀ x, (x < Real.log (1 / a) → (f x a) < (f (Real.log (1 / a)) a)) ∧ (Real.log (1 / a) < x → (f (Real.log (1 / a)) a) < f x a) :=
sorry

theorem lower_bound_fx {x a : ℝ} (h : 0 < a) : f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_case1_monotonicity_case2_lower_bound_fx_l470_470661


namespace inequality_positive_integers_l470_470597

theorem inequality_positive_integers (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 :=
sorry

end inequality_positive_integers_l470_470597


namespace sum_abc_equals_265_l470_470290

noncomputable def x : ℝ := Real.sqrt((Real.sqrt(73) / 2) + (5 / 2))

theorem sum_abc_equals_265
  (a b c : ℕ)
  (h₁ : x ^ 80 = 3 * x ^ 78 + 18 * x ^ 74 + 15 * x ^ 72 - x ^ 40 + a * x ^ 36 + b * x ^ 34 + c * x ^ 30) :
  a + b + c = 265 := by
  sorry

end sum_abc_equals_265_l470_470290


namespace log_expression_simplify_l470_470379

variable (x y : ℝ)

theorem log_expression_simplify (hx : 0 < x) (hx' : x ≠ 1) (hy : 0 < y) (hy' : y ≠ 1) :
  (Real.log x^2 / Real.log y^4) * 
  (Real.log y^3 / Real.log x^3) * 
  (Real.log x^4 / Real.log y^5) * 
  (Real.log y^5 / Real.log x^2) * 
  (Real.log x^3 / Real.log y^3) = (1 / 3) * Real.log x / Real.log y := 
sorry

end log_expression_simplify_l470_470379


namespace area_of_bounded_region_l470_470931

theorem area_of_bounded_region :
  let circle1 := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 2)^2 = 4}
  let circle2 := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 4)^2 = 16}
  let line1 := {p : ℝ × ℝ | p.2 = p.1 / real.sqrt 3}
  let line2 := {p : ℝ × ℝ | p.1 = 0}
  let bounded_region := circle1 ∩ circle2 ∩ (line1 ∪ line2)
  let area := 4 * real.pi + 6 * real.sqrt 3
  set.bounded bounded_region → sorry ->
  ∃ (area_calculated: ℝ), area_calculated = area := sorry

end area_of_bounded_region_l470_470931


namespace members_not_playing_either_l470_470251

theorem members_not_playing_either 
    (total_members : ℕ) 
    (badminton_players : ℕ) 
    (tennis_players : ℕ) 
    (both_sports : ℕ)
    (h1 : total_members = 30)
    (h2 : badminton_players = 18)
    (h3 : tennis_players = 19) 
    (h4 : both_sports = 9) : 
    total_members - ((badminton_players + tennis_players) - both_sports) = 2 := 
by 
    rw [h1, h2, h3, h4]
    norm_num
    sorry

end members_not_playing_either_l470_470251


namespace find_S40_l470_470126

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem find_S40 (a r : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = geometric_sequence_sum a r n)
  (h2 : S 10 = 10)
  (h3 : S 30 = 70) :
  S 40 = 150 ∨ S 40 = 110 := 
sorry

end find_S40_l470_470126


namespace canoe_speed_downstream_l470_470499

theorem canoe_speed_downstream (V_upstream V_s V_c V_downstream : ℝ) 
    (h1 : V_upstream = 6) 
    (h2 : V_s = 2) 
    (h3 : V_upstream = V_c - V_s) 
    (h4 : V_downstream = V_c + V_s) : 
  V_downstream = 10 := 
by 
  sorry

end canoe_speed_downstream_l470_470499


namespace sum_of_a_k_l470_470980

noncomputable def a : ℕ → ℝ := sorry

lemma geometric_sequence (a_n : ℕ → ℝ) (increasing : ∀ {i j : ℕ}, i < j → a_n i < a_n j) 
  (h1 : a_n 5 ^ 2 = a_n 10) 
  (h2 : ∀ n : ℕ, n > 0 → 2 * (a_n n + a_n (n + 2)) = 5 * a_n (n + 1)) 
  : ∀ n : ℕ, n > 0 → a_n n = 2 ^ n := 
sorry

noncomputable def c (a_n : ℕ → ℝ) (n : ℕ) : ℝ := 1 - (-1)^n * a_n n

theorem sum_of_a_k (a_n : ℕ → ℝ) (c_n : ℕ → ℝ) 
  (h3 : ∀ n : ℕ, n > 0 → a_n n = 2 ^ n) 
  (h4 : ∀ n : ℕ, n > 0 → c_n n = 1 - (-1)^n * a_n n) 
  : (∑ k in finset.filter (λ k, c_n k ≥ 2014) (finset.range 101), a_n k) = (2^101 - 2048) / 3 := 
sorry

end sum_of_a_k_l470_470980


namespace circle_equation_l470_470418

theorem circle_equation (a : ℝ) (r : ℝ) :
  ((y = 2 * x) ∧ (tangent_to_x_axis (circle (a, 2 * a) r)) ∧ (chord_length_interception (circle (a, 2 * a) r) (line x - y = 0) = sqrt 14)) →
  ((x - 1)^2 + (y - 2)^2 = 4 ∨ (x + 1)^2 + (y + 2)^2 = 4) :=
sorry

end circle_equation_l470_470418


namespace problem_solution_l470_470699

-- Define the triangle with given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variable (△ : A + B + C = π)
variable (hA : A = π / 4)
variable (hb : b = (sqrt 2 / 2) * a)

-- Definitions for B and area
def find_B (A B : ℝ) (b a : ℝ) : Prop :=
  B = π / 6

def find_area (a b : ℝ) (C : ℝ) : ℝ :=
  (1 / 2) * a * b * sin C

-- Problem statement
theorem problem_solution (a : ℝ) (hA : A = π / 4) (hb : b = (sqrt 2 / 2) * a) 
  (ha : a = sqrt 2) 
  (h1 : find_B A B b a)
  (h2 : find_area a b (C : ℝ) = (sqrt 3 + 1) / 4) : Prop :=
  h1 ∧ (h2 = (sqrt 3 + 1) / 4) :=
by
  sorry

end problem_solution_l470_470699


namespace half_angle_quadrant_l470_470177

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_quadrant_l470_470177


namespace num_elements_A_inter_B_l470_470676

def A : Set (ℝ × ℝ) := {(x, y) | y = x ^ 2}
def B : Set (ℝ × ℝ) := {(x, y) | y = 1 - | x |}

theorem num_elements_A_inter_B : 
  (A ∩ B).card = 2 :=
sorry

end num_elements_A_inter_B_l470_470676


namespace strongest_team_wins_both_games_l470_470491

theorem strongest_team_wins_both_games :
  ∃ (team : ℕ), team = 2018 ∧
    (∀ (teams : fin 2018 → ℕ), 
      (∀ i, teams i ≠ teams ((i + 1) % 2018)) ∧
      (∀ i, teams i ≠ teams ((i + 2) % 2018)) ∧
      (∀ i < 1009, 
        let match1_winner := max (teams (2*i)) (teams (2*i + 1)) in
        let match2_winner := max (teams (2*i + 1)) (teams (2*i + 2)) in
        (match1_winner = 2018 ∨ match2_winner = 2018)) ∧
      ∃ k, (∀ i ≠ k, 
        let match1_winner := max (teams (2*i)) (teams (2*i + 1)) in
        let match2_winner := max (teams (2*i + 1)) (teams (2*i + 2)) in
        (match1_winner ≠ 2018 ∨ match2_winner ≠ 2018))
    ) sorry

end strongest_team_wins_both_games_l470_470491


namespace volume_and_surface_area_of_inscribed_sphere_l470_470181

theorem volume_and_surface_area_of_inscribed_sphere (edge_length : ℝ) (h_edge : edge_length = 10) :
    let r := edge_length / 2
    let V := (4 / 3) * π * r^3
    let A := 4 * π * r^2
    V = (500 / 3) * π ∧ A = 100 * π := 
by
  sorry

end volume_and_surface_area_of_inscribed_sphere_l470_470181


namespace count_integers_between_cubes_l470_470217

noncomputable def a := (10.1)^3
noncomputable def b := (10.4)^3

theorem count_integers_between_cubes : 
  ∃ (count : ℕ), count = 94 ∧ (1030.031 < a) ∧ (a < b) ∧ (b < 1124.864) := 
  sorry

end count_integers_between_cubes_l470_470217


namespace compare_A_B_l470_470635

variable {x y : ℝ}
variable (h1 : x > 0) (h2 : y > 0)

noncomputable def A := (x + y) / (1 + x + y)
noncomputable def B := (x / (1 + x)) + (y / (1 + y))

theorem compare_A_B (h1 : x > 0) (h2 : y > 0) : A < B := 
by
  sorry

end compare_A_B_l470_470635


namespace circle_equation_tangent_to_x_axis_l470_470792

/--
The standard equation of a circle with center (-5, 4) and tangent to the x-axis
is given by (x + 5)² + (y - 4)² = 16.
-/
theorem circle_equation_tangent_to_x_axis :
  ∀ x y : ℝ, (x + 5) ^ 2 + (y - 4) ^ 2 = 16 ↔
    (x, y) ∈ {p : ℝ × ℝ | (p.1 + 5) ^ 2 + (p.2 - 4) ^ 2 = 16} :=
by 
  sorry

end circle_equation_tangent_to_x_axis_l470_470792


namespace findCircleEquation_minimumTriangleArea_l470_470615

-- Definitions based on the given conditions
def isTangentToYAxis (C : ℝ × ℝ) (r : ℝ) : Prop := C.1 = r
def isInFirstQuadrant (C : ℝ × ℝ) : Prop := C.1 > 0 ∧ C.2 > 0
def chordLength (C : ℝ × ℝ) (r : ℝ) : Prop := (C.2 - r)^2 + (r * r / (2)) = 3 * 2

-- Theorem statements based on solution steps
theorem findCircleEquation :
  ∃ C : ℝ × ℝ, ∃ r : ℝ,
    isTangentToYAxis C r ∧
    isInFirstQuadrant C ∧
    chordLength C r ∧
    (∀ x y, (x - C.1) ^ 2 + (y - C.2) ^ 2 = r^2 ↔ (x - 2) ^ 2 + (y - 1) ^ 2 = 4) :=
sorry

def isOnLine (P : ℝ × ℝ) : Prop := 3 * P.1 + 4 * P.2 + 5 = 0
def distanceFromCenterToLine (C : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ :=
  abs (3 * C.1 + 4 * C.2 + 5) / sqrt (3 ^ 2 + 4 ^ 2)
def minDistance (C : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := 3 -- from solution step calculation

theorem minimumTriangleArea :
  ∃ P : ℝ × ℝ, ∃ B : ℝ × ℝ, ∃ C : ℝ × ℝ, ∃ r : ℝ,
    isTangentToYAxis C r ∧
    isInFirstQuadrant C ∧
    chordLength C r ∧
    isOnLine P ∧
    distanceFromCenterToLine C isOnLine = 3 ∧
    sqrt (3^2 - 2^2) * 2 / 2 = sqrt 5 :=
sorry

end findCircleEquation_minimumTriangleArea_l470_470615


namespace helena_returns_first_l470_470041

theorem helena_returns_first
  (V_G V_H V_C : ℝ)
  (h_VG : 0 < V_G)
  (h_VH : 0 < V_H)
  (h_VC : 0 < V_C) :
  let t_G := 5 * (V_G + V_C) / (V_G - V_C),
      t_H := 5 * (V_H - V_C) / (V_H + V_C) in
  t_H < t_G :=
by
  simp [t_G, t_H]
  sorry

end helena_returns_first_l470_470041


namespace quadratic_radical_simplified_l470_470242

theorem quadratic_radical_simplified (a : ℕ) : 
  (∃ (b : ℕ), a = 3 * b^2) -> a = 3 := 
by
  sorry

end quadratic_radical_simplified_l470_470242


namespace sphere_radius_l470_470521

theorem sphere_radius (x y z r : ℝ) (h1 : 2 * x * y + 2 * y * z + 2 * z * x = 384)
  (h2 : x + y + z = 28) (h3 : (2 * r)^2 = x^2 + y^2 + z^2) : r = 10 := sorry

end sphere_radius_l470_470521


namespace inequality_holds_for_all_reals_l470_470347

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470347


namespace solution_for_factorial_equation_l470_470117

theorem solution_for_factorial_equation:
  { (n, k) : ℕ × ℕ | 0 < n ∧ 0 < k ∧ n! + n = n^k } = {(2,2), (3,2), (5,3)} :=
by
  sorry

end solution_for_factorial_equation_l470_470117


namespace num_1000_pointed_quadratic_stars_l470_470568

def gcd (a b : ℕ) : ℕ := nat.gcd a b

def euler_totient (n : ℕ) : ℕ := nat.totient n

noncomputable def count_non_similar_stars (n : ℕ) : ℕ :=
  let candidates := ∑ m in finset.range n, if (gcd m n = 1 ∧ m ^ 2 % n ≠ m) then 1 else 0 in
  candidates / 2

theorem num_1000_pointed_quadratic_stars :
  count_non_similar_stars 1000 = 200 :=
by
  sorry

end num_1000_pointed_quadratic_stars_l470_470568


namespace inequality_holds_for_real_numbers_l470_470324

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l470_470324


namespace average_age_of_team_l470_470413

noncomputable def average_age_team (total_players : ℕ) (captain_age : ℕ) (wicket_keeper_age : ℕ) (remaining_players_avg_age : ℕ → ℕ) : ℕ :=
let A := sorry in
A

theorem average_age_of_team (A : ℕ) :
  ∃ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_extra : ℕ) (remaining_players_size : ℕ),
  team_size = 11 ∧
  captain_age = 27 ∧
  wicket_keeper_extra = 3 ∧
  remaining_players_size = team_size - 2 ∧
  remaining_players_avg_age = A - 1 ∧
  11 * A = (captain_age + (captain_age + wicket_keeper_extra)) + 9 * (remaining_players_avg_age : ℕ → ℕ)
  by
    have team_size := 11
    have captain_age := 27
    have wicket_keeper_extra := 3
    have remaining_players_size := team_size - 2
    have remaining_players_avg_age := A - 1
    exists team_size, captain_age, wicket_keeper_extra, remaining_players_size
    split
    sorry

end average_age_of_team_l470_470413


namespace inequality_inequality_l470_470358

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470358


namespace age_ratio_l470_470836

theorem age_ratio 
  (a b c : ℕ)
  (h1 : a = b + 2)
  (h2 : a + b + c = 32)
  (h3 : b = 12) :
  b = 2 * c :=
by
  sorry

end age_ratio_l470_470836


namespace prove_a4_l470_470292

variable {a: ℕ → ℕ}

-- Define sequence a_n such that a_1 = 3 and a_{n+1} = 3 * a_n
def a_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | n + 1 => 3 * a_sequence n

-- Define S_n as the sum of the first n terms of the sequence a_n
def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a_sequence i

-- Given condition
axiom condition (n : ℕ) : 2 * S n = 3 * a_sequence n - 3

-- Prove that a_4 = 81
theorem prove_a4 : a_sequence 4 = 81 :=
  sorry

end prove_a4_l470_470292


namespace decimal_to_fraction_l470_470032

theorem decimal_to_fraction :
  (3.56 : ℚ) = 89 / 25 := 
sorry

end decimal_to_fraction_l470_470032


namespace inequality_proof_l470_470340

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470340


namespace upper_quartile_is_90_l470_470061

/-- Define the list of scores -/
def scores : List ℕ := [70, 85, 90, 75, 95]

/-- Function to calculate the upper quartile -/
def upper_quartile (scores : List ℕ) : ℕ :=
  let sorted_scores := scores.qsort (λ a b => a < b)
  sorted_scores.get! (sorted_scores.length * 3 / 4)

/-- Prove that the upper quartile of the given scores is 90 -/
theorem upper_quartile_is_90 : upper_quartile scores = 90 :=
by
  sorry

end upper_quartile_is_90_l470_470061


namespace sum_of_numbers_l470_470022

theorem sum_of_numbers : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 :=
by
  sorry

end sum_of_numbers_l470_470022


namespace volume_common_part_of_reflected_tetrahedrons_l470_470012

/-- Given two regular tetrahedrons ABCD and A'B'C'D', where A'B'C'D' is the reflection of ABCD through
the center O, and the volume of ABCD is 1, prove that the volume of the common part of these two tetrahedrons is 1/2. -/
theorem volume_common_part_of_reflected_tetrahedrons (O : Point) (A B C D A' B' C' D' : Point)
    (h1 : center O A B C D) (h2 : reflected_through_center O A B C D A' B' C' D')
    (vol_ABCD : volume (tetrahedron A B C D) = 1) :
  volume (common_part (tetrahedron A B C D) (tetrahedron A' B' C' D')) = 1/2 := 
sorry

end volume_common_part_of_reflected_tetrahedrons_l470_470012


namespace value_after_trebling_l470_470514

theorem value_after_trebling (n : ℕ) (h : n = 10) : (3 * (2 * n + 8)) = 84 := by
  rw [h]
  norm_num
  sorry

end value_after_trebling_l470_470514


namespace bicycle_selling_prices_l470_470871

def purchase_price1 : ℝ := 1800
def loss_percentage1 : ℝ := 25
def purchase_price2 : ℝ := 2700
def loss_percentage2 : ℝ := 15
def purchase_price3 : ℝ := 2200
def loss_percentage3 : ℝ := 20

def loss_amount (price : ℝ) (percentage : ℝ) : ℝ :=
  (percentage / 100) * price

def selling_price (price : ℝ) (percentage : ℝ) : ℝ :=
  price - loss_amount price percentage

theorem bicycle_selling_prices :
  selling_price purchase_price1 loss_percentage1 = 1350 ∧
  selling_price purchase_price2 loss_percentage2 = 2295 ∧
  selling_price purchase_price3 loss_percentage3 = 1760 :=
by
  sorry

end bicycle_selling_prices_l470_470871


namespace chord_length_of_circle_intersected_by_line_l470_470415

open Real

-- Definitions for the conditions given in the problem
def line_eqn (x y : ℝ) : Prop := x - y - 1 = 0
def circle_eqn (x y : ℝ) : Prop := x^2 - 4 * x + y^2 = 4

-- The proof statement (problem) in Lean 4
theorem chord_length_of_circle_intersected_by_line :
  ∀ (x y : ℝ), circle_eqn x y → line_eqn x y → ∃ L : ℝ, L = sqrt 17 := by
  sorry

end chord_length_of_circle_intersected_by_line_l470_470415


namespace length_XQ_eq_diameter_l470_470899

open EuclideanGeometry

variables {O1 O2 A B Y Z X Q : Point}
variable Ω : Circle

-- Given: Two circles with centers O1 and O2 intersect at points A and B.
-- Line passing through A intersects the circles at points Y and Z.
-- Tangents to the circles at Y and Z intersect at point X.
-- Triangle O1 O2 B is circumcised by circle Ω.
-- XB intersects Ω at another point Q.
-- Prove: Length of XQ equals the diameter of Ω.
theorem length_XQ_eq_diameter {r : ℝ} (h1 : ∀ Ω, Circle Ω → circumcircle Ω O1 O2 B)
    (h2 : ∀ O₁, IsCircle Ω O₁) 
    (h3 : XB ∩ Ω = { B, Q }) : 
    distance X Q = 2 * radius Ω := 
sorry

end length_XQ_eq_diameter_l470_470899


namespace boat_speed_in_still_water_l470_470433

theorem boat_speed_in_still_water (rate_of_current distance_downstream : ℝ) 
  (time_downstream : ℝ) (H1 : rate_of_current = 4) 
  (H2 : distance_downstream = 9.6) (H3 : time_downstream = 24 / 60) :
  ∃ x : ℝ, x + rate_of_current = 20 :=
by
  -- Given data
  have hx : 0.4 * (x + 4) = 9.6, from sorry
  -- Solving for x
  have h_eq : 0.4 * x + 1.6 = 9.6, by sorry
  have h_sub : 0.4 * x = 8, by sorry
  have h_div : x = 20, by sorry
  exact ⟨x, h_div⟩

end boat_speed_in_still_water_l470_470433


namespace perfect_square_difference_of_solutions_l470_470280

theorem perfect_square_difference_of_solutions
  (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℤ, k^2 = x - y := 
sorry

end perfect_square_difference_of_solutions_l470_470280


namespace first_reduction_percentage_l470_470789

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.6 = P * 0.45 → x = 25 :=
by
  sorry

end first_reduction_percentage_l470_470789


namespace sequence_all_integers_l470_470790

theorem sequence_all_integers (a : ℕ → ℚ) (n : ℕ) (h1: a 1 = 1) (h2: a 2 = 143)
  (h3: ∀ n ≥ 2, a (n + 1) = 5 * (∑ i in finset.range n, a (i + 1)) / n) : 
  ∀ n, a n ∈ ℤ :=
by sorry

end sequence_all_integers_l470_470790


namespace proof_statement_l470_470540

def degrees_to_radians (d : ℝ) : ℝ := d * real.pi / 180

def in_second_quadrant (θ : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * real.pi + real.pi / 2 < θ ∧ θ < 2 * k * real.pi + real.pi

def normalize_angle (θ : ℝ) : ℝ :=
  let τ := 2 * real.pi in
  ((θ / τ - θ.floor).abs : ℝ) * τ

def problem_statement : Prop :=
  (in_second_quadrant (normalize_angle (degrees_to_radians 160))) ∧
  (in_second_quadrant (normalize_angle (degrees_to_radians 480))) ∧
  (in_second_quadrant (normalize_angle (degrees_to_radians (-960)))) ∧
  ¬ (in_second_quadrant (normalize_angle (degrees_to_radians 1530)))

theorem proof_statement : problem_statement := sorry

end proof_statement_l470_470540


namespace range_of_distance_l470_470673

-- Define the angle theta and the trigonometric functions involved
variables (θ : Real)

-- Define the point and line based on the given condition
def point : Real × Real := (Real.sin θ, Real.cos θ)

noncomputable def line (x y : Real) : Real := x * Real.cos θ + y * Real.sin θ + 1

-- Define the distance d given the distance formula
noncomputable def distance : Real := abs ((Real.sin θ * Real.cos θ + Real.cos θ * Real.sin θ + 1) / Math.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2))

-- Simplify the expression for the distance
noncomputable def simplified_distance : Real := abs (Real.sin (2 * θ) + 1)

theorem range_of_distance : ∀ (θ : Real), 0 ≤ abs (Real.sin (2 * θ) + 1) ∧ abs (Real.sin (2 * θ) + 1) ≤ 2 := 
sorry

end range_of_distance_l470_470673


namespace combined_degrees_l470_470381

theorem combined_degrees (S J W : ℕ) (h1 : S = 150) (h2 : J = S - 5) (h3 : W = S - 3) : S + J + W = 442 :=
by
  sorry

end combined_degrees_l470_470381


namespace eq_axis_symm_for_g_l470_470580

def f (x : ℝ) := 3 * Real.sin (4 * x + Real.pi / 6)

def g (x : ℝ) := 3 * Real.sin x

theorem eq_axis_symm_for_g : 
  ∃ (k : ℤ), x = Real.pi / 3 + k * Real.pi :=
sorry

end eq_axis_symm_for_g_l470_470580


namespace f_has_exactly_one_zero_l470_470002

def f(x : ℝ) : ℝ := Real.exp x + x - 2

theorem f_has_exactly_one_zero : ∃! x : ℝ, f(x) = 0 := sorry

end f_has_exactly_one_zero_l470_470002


namespace prop_A_prop_C_not_prop_B_not_prop_D_l470_470941

def min_element (S : Finset ℕ) : ℕ :=
  S.min' (by simp [Finset.nonempty])

def distance (A B : Finset ℕ) : ℕ :=
  (Finset.image (λ (ab : ℕ × ℕ), (ab.fst - ab.snd).natAbs) (A.product B)).min' (by simp [Finset.nonempty])

theorem prop_A (A B : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (h : min_element A = min_element B) : distance A B = 0 := sorry

theorem prop_C (A B : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (h : distance A B = 0) : (A ∩ B).nonempty := sorry

theorem not_prop_B (A B : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (h : min_element A > min_element B) : distance A B ≤ 0 := sorry

theorem not_prop_D : ∃ (A B C : Finset ℕ) (hA : A.nonempty) (hB : B.nonempty) (hC : C.nonempty), distance A B + distance B C < distance A C := sorry

end prop_A_prop_C_not_prop_B_not_prop_D_l470_470941


namespace transcendental_iff_not_algebraic_skew_iff_non_parallel_non_intersecting_l470_470157

section

-- Definition and condition for transcendental numbers
def is_algebraic (x : ℝ) : Prop :=
  ∃ (p : polynomial ℚ), p ≠ 0 ∧ p.eval x = 0

def is_transcendental (x : ℝ) : Prop :=
  ¬ is_algebraic x

-- We are proving that a transcendental number is not algebraic (negatively defined)
theorem transcendental_iff_not_algebraic (x : ℝ) : is_transcendental x ↔ ¬ is_algebraic x := 
by sorry

-- Definition and conditions for skew lines
structure Line3D :=
  (point1 : ℝ × ℝ × ℝ)
  (point2 : ℝ × ℝ × ℝ)

def lines_intersect (l1 l2 : Line3D) : Prop :=
  -- This is a placeholder definition, normally you need to check if there exists a common point on both lines
  false -- stub, replace with actual geometric condition

def lines_parallel (l1 l2 : Line3D) : Prop :=
  -- This is a placeholder definition, normally you need to check if the direction vectors are parallel
  false -- stub, replace with actual geometric condition

def skew_lines (l1 l2 : Line3D) : Prop :=
  ¬ lines_intersect l1 l2 ∧ ¬ lines_parallel l1 l2

theorem skew_iff_non_parallel_non_intersecting (l1 l2 : Line3D) : skew_lines l1 l2 ↔ (¬ lines_intersect l1 l2 ∧ ¬ lines_parallel l1 l2) :=
by sorry

end

end transcendental_iff_not_algebraic_skew_iff_non_parallel_non_intersecting_l470_470157


namespace series_evaluation_l470_470598

-- Define T(n) as the ceiling of the square root of n
def T (n : ℕ) : ℕ := Nat.ceil (Real.sqrt n)

-- State the theorem, given the conditions, to prove the equivalence
theorem series_evaluation : 
  (∑' n : ℕ, (2^(T n) + 2^(-(T n))) / 3^n) = ∑' k : ℕ, (2^(2*k + 1 - 2*k) - 2^(2*k + 1 - 4*k)) := 
sorry

end series_evaluation_l470_470598


namespace ratio_of_volume_to_surface_area_l470_470075

-- Define conditions
def unit_cube_volume : ℕ := 1
def unit_cube_exposed_faces_at_ends : ℕ := 5
def unit_cube_exposed_faces_in_middle : ℕ := 4
def num_end_cubes : ℕ := 2
def num_middle_cubes : ℕ := 6
def total_cubes : ℕ := 8

-- Define volume
def volume_of_shape : ℕ := total_cubes * unit_cube_volume

-- Define surface area
def surface_area_of_shape : ℕ := 
  num_end_cubes * unit_cube_exposed_faces_at_ends + 
  num_middle_cubes * unit_cube_exposed_faces_in_middle

-- Define ratio
def volume_to_surface_area_ratio : ℚ := 
  (volume_of_shape : ℚ) / (surface_area_of_shape : ℚ)

-- Proposition stating the ratio
theorem ratio_of_volume_to_surface_area :
  volume_to_surface_area_ratio = 4 / 17 := by
  sorry

end ratio_of_volume_to_surface_area_l470_470075


namespace monotonicity_f_f_gt_lower_bound_l470_470666

-- Definition of the function
def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

-- Statement 1: Monotonicity discussion
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, (f a)' x < 0) ∧ 
  (a > 0 → ∀ x : ℝ,
    (f a)' x < 0 ∧ x < Real.log (1 / a) ∨
    (f a)' x > 0 ∧ x > Real.log (1 / a)) :=
sorry

-- Statement 2: Proof for f(x) > 2 ln a + 3/2 for a > 0
theorem f_gt_lower_bound (a x : ℝ) (ha : 0 < a) :
  f a x > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_f_f_gt_lower_bound_l470_470666


namespace decimal_to_fraction_l470_470036

theorem decimal_to_fraction (a b : ℚ) (h : a = 3.56) (h1 : b = 56/100) (h2 : 56.gcd 100 = 4) :
  a = 89/25 := by
  sorry

end decimal_to_fraction_l470_470036


namespace maximum_questions_missed_7_l470_470550

def maximum_number_of_questions_missed (total_questions : ℕ) (success_percentage : ℚ) : ℕ := 
  let allowed_mistakes := (1 - success_percentage / 100) * total_questions
  in ⌊allowed_mistakes⌋ -- This is the floor function to round down

theorem maximum_questions_missed_7 :
  maximum_number_of_questions_missed 50 85 = 7 :=
by 
  -- Compute the allowed mistakes
  let allowed_mistakes := (1 - 85 / 100) * 50
  -- Compute the floor of 7.5
  have h1 : allowed_mistakes = 7.5 := rfl
  have h2 : ⌊7.5⌋ = 7 := rfl
  -- Conclude the theorem
  show maximum_number_of_questions_missed 50 85 = 7 from h2

end maximum_questions_missed_7_l470_470550


namespace chef_made_accidentally_l470_470866

-- Definitions based on problem conditions
def fraction_chocolate_initial := 0.40
def fraction_chocolate_target := 0.50
def fraction_raspberry_initial := 0.60
def fraction_raspberry_target := 0.50
def amount_to_remove := 2.5
def amount_of_chocolate_added := 2.5

-- Proof statement
theorem chef_made_accidentally (x : ℝ) :
  0.40 * x - (0.40 * 2.5) + 2.5 = 0.50 * (x - 2.5 + 2.5) → 
  x = 12.5 := by try_simp

end chef_made_accidentally_l470_470866


namespace range_of_exp_decay_l470_470633

noncomputable def f (k x : ℝ) := Real.exp (-k * x)

theorem range_of_exp_decay (k : ℝ) (h : k > 0) :
  set.range (f k) = set.Ioo 0 1 ∪ set.Icc 0 1 :=
by
  sorry

end range_of_exp_decay_l470_470633


namespace robot_capacities_and_min_robots_l470_470507

-- Define material handling capacities for robots A and B
def handle_capacity_A (x : ℕ) : ℕ := x + 30
def handle_capacity_B (x : ℕ) : ℕ := x

-- Define conditions
def condition1 : Prop := ∀ x, handle_capacity_A x = handle_capacity_B x + 30
def condition2 : Prop := ∀ x, 1000 / handle_capacity_A x = 800 / handle_capacity_B x
def condition3 (a b : ℕ) : Prop := a + b = 20
def condition4 (a b : ℕ) : Prop := 150 * a + 120 * b ≥ 2800

-- Prove the capacities and number of robots needed
theorem robot_capacities_and_min_robots :
  ∃ (x : ℕ) (a : ℕ), handle_capacity_B x = 120 ∧ handle_capacity_A x = 150 ∧ a ≥ 14 ∧ condition1 x ∧ condition2 x ∧ condition3 a (20 - a) ∧ condition4 a (20 - a) :=
by
  sorry

end robot_capacities_and_min_robots_l470_470507


namespace average_speed_l470_470039

theorem average_speed (d1 d2 : ℝ) (s1 s2 : ℝ) (H1 : d1 = 7) (H2 : s1 = 10) (H3 : d2 = 10) (H4 : s2 = 7) :
  let t1 := d1 / s1,
      t2 := d2 / s2,
      total_distance := d1 + d2,
      total_time := t1 + t2
  in abs ((total_distance / total_time) - 7.98) < 0.01 :=
by
  sorry

end average_speed_l470_470039


namespace garbage_decomposition_time_l470_470978

noncomputable def decomposition_time (a b : ℝ) (t : ℕ) : ℝ :=
  a * b ^ t

theorem garbage_decomposition_time :
  ∃ t : ℕ, (
    ∃ a b : ℝ, 
      0 < a ∧ 0 < b ∧ 
      a * b^6 = 0.05 ∧ 
      a * b^12 = 0.1 ∧ 
      abs ((Real.log 2) - 0.3 * (10:ℝ).log10 / (2:ℝ).log10) < 0.1
  ) ∧ 
  decomposition_time (1 / 40) (2^(1/6:ℝ)) t = 1 ∧ t ≈ 32 :=
sorry

end garbage_decomposition_time_l470_470978


namespace r_not_divide_m_add_1_l470_470743

def α (r s : ℕ) (r_gt_s : r > s) : ℚ := r / s

def N_α (r s : ℕ) (r_gt_s : r > s) : set ℤ :=
  {m | ∃ n : ℕ, m = int.floor (n * (α r s r_gt_s))}

theorem r_not_divide_m_add_1 (r s : ℕ) (r_pos : r > 0) (s_pos : s > 0) (r_gt_s : r > s) (gcd_rs : r.gcd s = 1) :
  ∀ m ∈ N_α r s r_gt_s, ¬ r ∣ (m + 1) := 
sorry

end r_not_divide_m_add_1_l470_470743


namespace find_expression_for_f_intervals_of_increase_and_symmetry_center_range_of_m_l470_470191

noncomputable theory

variables {A ω h ϕ : ℝ} (x : ℝ)
variables (k : ℤ)
variables (f : ℝ → ℝ := λ x, A * sin(ω * x + ϕ) + h)

-- Conditions
axiom A_pos : A > 0
axiom ω_pos : ω > 0
axiom abs_ϕ_lt_pi : |ϕ| < π
axiom max_value_at_pi_over_12 : f (π / 12) = 6
axiom min_value_at_7pi_over_12 : f (7 * π / 12) = 0

theorem find_expression_for_f :
  f = λ x, 3 * sin (2 * x + π / 3) + 3 := 
sorry

theorem intervals_of_increase_and_symmetry_center :
  (∀ k, ∀ x ∈ Icc (-5 * π / 12 + k * π) (π / 12 + k * π), f x = 3 * sin(2 * x + π / 3) + 3) ∧
  (∀ k, (k * π / 2 - π / 6, 3) ∈ {p : ℝ × ℝ | ∃ x, f x = p.2}) :=
sorry

theorem range_of_m :
  (∀ m, mf (π / 6) - 1 = 0 → m ∈ Icc (1 / 6) (2 / 9)) :=
sorry

end find_expression_for_f_intervals_of_increase_and_symmetry_center_range_of_m_l470_470191


namespace initial_pens_l470_470830

theorem initial_pens (P : ℤ) (INIT : 2 * (P + 22) - 19 = 39) : P = 7 :=
by
  sorry

end initial_pens_l470_470830


namespace number_of_women_l470_470795

variable (W : ℕ) (x : ℝ)

-- Conditions
def daily_wage_men_and_women (W : ℕ) (x : ℝ) : Prop :=
  24 * 350 + W * x = 11600

def half_men_and_37_women (W : ℕ) (x : ℝ) : Prop :=
  12 * 350 + 37 * x = 24 * 350 + W * x

def daily_wage_man := (350 : ℝ)

-- Proposition to prove
theorem number_of_women (W : ℕ) (x : ℝ) (h1 : daily_wage_men_and_women W x)
  (h2 : half_men_and_37_women W x) : W = 16 := 
  by
  sorry

end number_of_women_l470_470795


namespace sum_of_nonnegative_numbers_eq_10_l470_470697

theorem sum_of_nonnegative_numbers_eq_10 (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 48)
  (h2 : ab + bc + ca = 26)
  (h3 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) : a + b + c = 10 := 
by
  sorry

end sum_of_nonnegative_numbers_eq_10_l470_470697


namespace volume_tetrahedron_ABCD_l470_470977

-- Given conditions in the Lean 4 statement
variables {A B C D O : Point}
variable {r : Real}
variable (volume : Real)

-- Geometry setup: Tetrahedron inscribed in a sphere with specific sides and properties
variables (tetrahedron_inscribed : inscribed_in_sphere A B C D O)
variable (ad_is_diameter : diameter O A D)
variable (triangle_ABC_eq : equilateral_triangle A B C 1)
variable (triangle_BCD_eq : equilateral_triangle B C D 1)

-- The final statement to be proven
theorem volume_tetrahedron_ABCD : 
  volume_of_tetrahedron A B C D = √2 / 12 :=
sorry

end volume_tetrahedron_ABCD_l470_470977


namespace problem1_problem2_problem3_l470_470975

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions
axiom norm_a : ‖a‖ = 1
axiom norm_b : ‖b‖ = Real.sqrt 2

-- 1. Prove |a + b| = sqrt(3 + sqrt(2)) given the angle between a and b is 60 degrees.
theorem problem1
  (angle_ab : real.angle (inner_product_space.inner_product a b) = Real.pi / 3) :
  ‖a + b‖ = Real.sqrt (3 + Real.sqrt 2) :=
sorry

-- 2. Prove the angle between a and b is 45 degrees, given a - b is perpendicular to a.
theorem problem2
  (perpendicular_condition : inner_product_space.inner_product (a - b) a = 0) :
  real.angle (inner_product_space.inner_product_a b) = Real.pi / 4 :=
sorry

-- 3. Prove a ⋅ b = sqrt(2) or a ⋅ b = -sqrt(2) given a is parallel to b.
theorem problem3
  (parallel_condition : ∃ k : ℝ, a = k • b) :
  (inner_product_space.inner_product a b = Real.sqrt 2) ∨
  (inner_product_space.inner_product a b = -Real.sqrt 2) :=
sorry

end problem1_problem2_problem3_l470_470975


namespace right_triangle_incircle_l470_470718

noncomputable def incircle_relations (A B C O D E F : Point) (M N : Point) : Prop :=
  ∀ (AD BD : ℝ), 
  is_right_triangle A B C ∧
  touches_incircle O D AB ∧
  touches_incircle O E BC ∧
  touches_incircle O F CA ∧
  perpendicular_from D AC M ∧
  perpendicular_from D BC N ∧
  rectangle_area (C M D N) = 8 →
  (AD = dist A D) →
  (BD = dist B D) →
  1 / AD + 1 / BD = 1 / 2

theorem right_triangle_incircle (A B C O D E F : Point) (M N : Point) :
  incircle_relations A B C O D E F M N :=
begin
  sorry
end

end right_triangle_incircle_l470_470718


namespace find_inverse_sum_l470_470202

theorem find_inverse_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 :=
sorry

end find_inverse_sum_l470_470202


namespace factor_polynomial_l470_470583

theorem factor_polynomial (x : ℤ) :
  36 * x ^ 6 - 189 * x ^ 12 + 81 * x ^ 9 = 9 * x ^ 6 * (4 + 9 * x ^ 3 - 21 * x ^ 6) := 
sorry

end factor_polynomial_l470_470583


namespace inequality_proof_l470_470350

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l470_470350


namespace first_three_decimal_digits_l470_470917

theorem first_three_decimal_digits (a : ℝ) 
  (ha : a = (10^2003 + 1)^(11/8)) :
  ∃ (d1 d2 d3 : ℕ), d1 = 3 ∧ d2 = 7 ∧ d3 = 5 ∧
  ( ∀ ε > 0, (real.fract a - 0.375).abs < ε ) :=
begin
  sorry
end

end first_three_decimal_digits_l470_470917


namespace c_payment_l470_470059

theorem c_payment 
  (A_rate : ℝ) (B_rate : ℝ) (days : ℝ) (total_payment : ℝ) (C_fraction : ℝ) 
  (hA : A_rate = 1 / 6) 
  (hB : B_rate = 1 / 8) 
  (hdays : days = 3) 
  (hpayment : total_payment = 3200)
  (hC_fraction : C_fraction = 1 / 8) :
  total_payment * C_fraction = 400 :=
by {
  -- The proof would go here
  sorry
}

end c_payment_l470_470059


namespace total_games_played_l470_470847

theorem total_games_played (n k: ℕ) (h_n: n = 15) (h_k: k = 2) :
  nat.choose n k = 105 :=
by {
  rw [h_n, h_k],
  norm_num,
  sorry
}

end total_games_played_l470_470847


namespace part1_part2_l470_470611

-- Part (1): Proving the range of x when a = 1
theorem part1 (x : ℝ) : (x^2 - 6 * 1 * x + 8 < 0) ∧ (x^2 - 4 * x + 3 ≤ 0) ↔ 2 < x ∧ x ≤ 3 := 
by sorry

-- Part (2): Proving the range of a when p is a sufficient but not necessary condition for q
theorem part2 (a : ℝ) : (∀ x : ℝ, (x^2 - 6 * a * x + 8 * a^2 < 0 → x^2 - 4 * x + 3 ≤ 0) 
  ∧ (∃ x : ℝ, x^2 - 4 * x + 3 ≤ 0 ∧ x^2 - 6 * a * x + 8 * a^2 ≥ 0)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end part1_part2_l470_470611


namespace inequality_inequality_l470_470361

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470361


namespace inequality_inequality_l470_470359

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l470_470359


namespace percentage_error_in_side_l470_470092

section
variable (x e : ℝ) (h : (x + e)^2 = 1.21 * x^2)

def percentage_error_in_area := 21

theorem percentage_error_in_side : (2 * e ≈ 0.21 * x) → (e / x * 100 = 10.5) :=
by
  intro h1
  sorry

end

end percentage_error_in_side_l470_470092


namespace inverse_square_relationship_l470_470045

theorem inverse_square_relationship (k : ℝ) (y : ℝ) (h1 : ∀ x y, x = k / y^2)
  (h2 : ∃ y, 1 = k / y^2) (h3 : 0.5625 = k / 4^2) :
  ∃ y, 1 = 9 / y^2 ∧ y = 3 :=
by
  sorry

end inverse_square_relationship_l470_470045


namespace ryan_total_commuting_time_l470_470801

def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def bus_commutes : ℕ := 3
def total_bus_time : ℕ := bus_time * bus_commutes
def friend_time : ℕ := biking_time - (2 * biking_time / 3)
def total_commuting_time : ℕ := biking_time + total_bus_time + friend_time

theorem ryan_total_commuting_time :
  total_commuting_time = 160 :=
by
  sorry

end ryan_total_commuting_time_l470_470801


namespace hexagon_ball_strike_range_l470_470519

def is_theta_within_range (θ : ℝ) : Prop :=
  arctan (3 * sqrt 3 / 10) < θ ∧ θ < arctan (3 * sqrt 3 / 8)

theorem hexagon_ball_strike_range (θ : ℝ) :
  ∀ (hexagon : Hexagon) (P : Point) (Q : Point),
  is_midpoint P (AB hexagon) ∧ 
  is_on_side Q (BC hexagon) ∧ 
  angle_between (line_segment P Q) = θ →
  is_theta_within_range θ :=
sorry

end hexagon_ball_strike_range_l470_470519


namespace determine_m_l470_470205

open Set Real

theorem determine_m (m : ℝ) : (∀ x, x ∈ { x | x ≥ 3 } ∪ { x | x < m }) ∧ (∀ x, x ∉ { x | x ≥ 3 } ∩ { x | x < m }) → m = 3 :=
by
  intros h
  sorry

end determine_m_l470_470205


namespace minimal_abs_diff_l470_470228

theorem minimal_abs_diff (x y : ℕ) (h_cond : x * y - 8 * x + 7 * y = 775) : |x - y| = 703 :=
by
  sorry

end minimal_abs_diff_l470_470228


namespace inequality_holds_for_all_real_numbers_l470_470366

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470366


namespace binomial_coefficient_x3_expansion_l470_470571

theorem binomial_coefficient_x3_expansion :
  let x := @X ℚ _ 
  let term := (Polynomial.X - (Polynomial.C (1 : ℚ) * Polynomial.X ^ -2)) ^ 6
  (term.coeff 3) = -6 := by
  sorry

end binomial_coefficient_x3_expansion_l470_470571


namespace sum_of_acute_angles_is_pi_l470_470970

theorem sum_of_acute_angles_is_pi 
  {α β γ : ℝ} 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : 0 < γ ∧ γ < π/2) 
  (h4 : cos α + cos β + cos γ = 1 + 4 * sin (α/2) * sin (β/2) * sin (γ/2)) : 
  α + β + γ = π :=
sorry

end sum_of_acute_angles_is_pi_l470_470970


namespace tiles_per_row_l470_470386

noncomputable def area_square_room : ℕ := 400
noncomputable def side_length_feet : ℕ := nat.sqrt area_square_room
noncomputable def side_length_inches : ℕ := side_length_feet * 12
noncomputable def tile_size : ℕ := 8

theorem tiles_per_row : (side_length_inches / tile_size) = 30 := 
by
  have h1: side_length_feet = 20 := by sorry
  have h2: side_length_inches = 240 := by sorry
  have h3: side_length_inches / tile_size = 30 := by sorry
  exact h3

end tiles_per_row_l470_470386


namespace problem_solution_l470_470973

open Nat

theorem problem_solution (n : ℕ) (h1 : n > 0)
  (h2 : 2 * binom n 1 = (1 / 5) * 2^2 * binom n 2) :
  n = 6 ∧ ∑ i in range (n + 1), (2 : ℕ)^i = 64 := by
  sorry

end problem_solution_l470_470973


namespace inequalities_hold_l470_470026

variable {a b c x y z : ℝ}

theorem inequalities_hold 
  (h1 : x ≤ a)
  (h2 : y ≤ b)
  (h3 : z ≤ c) :
  x * y + y * z + z * x ≤ a * b + b * c + c * a ∧
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  x * y * z ≤ a * b * c :=
sorry

end inequalities_hold_l470_470026


namespace remainder_7_10_20_2_20_5_mod_9_l470_470920

theorem remainder_7_10_20_2_20_5_mod_9 :
  (7 * 10^20 + 2^20 + 5) % 9 = 7 :=
begin
  have h1 : 10 % 9 = 1, by norm_num,
  have h2 : 10^20 % 9 = 1 % 9, by { rw ←h1,  exact nat.pow_mod 10 20 9},
  have h3 : 2^6 % 9 = 1, by norm_num,
  have h4 : 2^20 % 9 = 4 % 9, by { rw ←nat.mod_repeat 2 6 20,  norm_num [h3]},
  rw [mul_mod 7 (10^20) 9, h2, mul_one, add_mod 7 4 5, add_mod (7 + 4) 5, ←add_assoc, ←add_eq_add_iff_eq],
  norm_num
end

end remainder_7_10_20_2_20_5_mod_9_l470_470920


namespace earnings_correct_l470_470545

-- Define the initial number of roses, the number of roses left, and the price per rose.
def initial_roses : ℕ := 13
def roses_left : ℕ := 4
def price_per_rose : ℕ := 4

-- Calculate the number of roses sold.
def roses_sold : ℕ := initial_roses - roses_left

-- Calculate the total earnings.
def earnings : ℕ := roses_sold * price_per_rose

-- Prove that the earnings are 36 dollars.
theorem earnings_correct : earnings = 36 := by
  sorry

end earnings_correct_l470_470545


namespace boarding_students_total_l470_470798

def total_students := 50
def male_students := 33
def female_students := total_students - male_students
def female_youth_league := 7
def female_boarding_students := 9
def non_boarding_youth_league := 15
def male_boarding_youth_league := 6
def male_non_youth_league_non_boarding := 8
def female_non_youth_league_non_boarding := 3

theorem boarding_students_total :
  ∀ (total_students male_students female_students female_youth_league female_boarding_students non_boarding_youth_league male_boarding_youth_league male_non_youth_league_non_boarding female_non_youth_league_non_boarding : ℕ),
  total_students = 50 → male_students = 33 → female_students = total_students - male_students →
  female_youth_league = 7 → female_boarding_students = 9 →
  non_boarding_youth_league = 15 → male_boarding_youth_league = 6 →
  male_non_youth_league_non_boarding = 8 → female_non_youth_league_non_boarding = 3 →
  (let male_boarding_non_youth_league := male_students - male_boarding_youth_league - male_non_youth_league_non_boarding in
   male_boarding_non_youth_league + female_boarding_students = 28) :=
by {
  intros,
  sorry
}

end boarding_students_total_l470_470798


namespace containers_filled_with_tea_l470_470446

-- We assume the conditions provided in the problem
def gallons := 20
def pints_in_gallon := 8
def total_pints := gallons * pints_in_gallon

def pint_consumed := 7
def containers_consumed := 3.5
def pints_per_container := pint_consumed / containers_consumed

-- Main theorem: Prove the number of containers filled with tea is 80
theorem containers_filled_with_tea : total_pints / pints_per_container = 80 :=
sorry  -- proof goes here

end containers_filled_with_tea_l470_470446


namespace factor_square_x_y_l470_470738

variable (x y : ℕ)
variable (p R : ℕ → ℕ → ℕ)

-- Conditions
axiom symmetric : ∀ x y, p(x, y) = p(y, x)
axiom factor_x_y : ∀ x y, ∃ Q, p(x, y) = (x-y) * Q

-- Proof Statement
theorem factor_square_x_y : ∀ x y, ∃ R, p(x, y) = (x-y) * (x-y) * R := 
sorry

end factor_square_x_y_l470_470738


namespace Brian_traveled_60_miles_l470_470850

theorem Brian_traveled_60_miles (mpg gallons : ℕ) (hmpg : mpg = 20) (hgallons : gallons = 3) :
    mpg * gallons = 60 := by
  sorry

end Brian_traveled_60_miles_l470_470850


namespace Patrick_hours_less_than_twice_Greg_l470_470759

def J := 18
def G := J - 6
def total_hours := 50
def P : ℕ := sorry -- To be defined, we need to establish the proof later with the condition J + G + P = 50
def X : ℕ := sorry -- To be defined, we need to establish the proof later with the condition P = 2 * G - X

theorem Patrick_hours_less_than_twice_Greg : X = 4 := by
  -- Placeholder definitions for P and X based on the given conditions
  let P := total_hours - (J + G)
  let X := 2 * G - P
  sorry -- Proof details to be filled in

end Patrick_hours_less_than_twice_Greg_l470_470759


namespace no_xy_term_when_k_eq_one_third_l470_470468

theorem no_xy_term_when_k_eq_one_third :
  ∃ k : ℚ, (∀ (x y : ℚ), x^2 - 3 * k * x * y + 3 * y^2 + x * y - 8 ≠ (-3 * k + 1) * x * y) ∧ k = 1 / 3 :=
by
  let k := (1 : ℚ) / 3
  use k
  split
  {
    intros x y
    sorry
  }
  {
    refl
  }

end no_xy_term_when_k_eq_one_third_l470_470468


namespace num_subsets_to_one_l470_470279

variable (n : ℕ) 

-- Definition of function f
def f (S : Finset ℕ) : Finset ℕ := 
  Finset.filter (λ k, k ≤ n ∧ odd (Finset.card (S.filter (λ d, d ∣ k)))) (Finset.range (n+1))

-- Theorem statement, no proof required
theorem num_subsets_to_one (hn : 0 < n) : 
  (∃ k : ℕ, 2 ^ k = 2 ^ Nat.floor (Float.log2 (Float.log2 n).toReal.toFloat)) :=
sorry

end num_subsets_to_one_l470_470279


namespace cos_double_angle_nonpositive_l470_470950

theorem cos_double_angle_nonpositive (α β : ℝ) (φ : ℝ) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := 
sorry

end cos_double_angle_nonpositive_l470_470950


namespace smallest_number_l470_470459

-- Define conditions
def ends_with_28 (n : ℕ) : Prop :=
  n % 100 = 28

def divisible_by_28 (n : ℕ) : Prop :=
  n % 28 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def desired_sum (n : ℕ) : Prop :=
  sum_of_digits n = 28

-- Proof problem statement
theorem smallest_number (n : ℕ) :
  ends_with_28 n ∧ divisible_by_28 n ∧ desired_sum n → n = 18928 :=
begin
  sorry
end

end smallest_number_l470_470459


namespace monotonicity_case1_monotonicity_case2_lower_bound_fx_l470_470663

noncomputable def f (x a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 {x a : ℝ} (h : a ≤ 0) : 
  ∀ x, (differentiable ℝ (λ x, f x a) ∧ deriv (λ x, f x a) x ≤ -1) :=
sorry

theorem monotonicity_case2 {x a : ℝ} (h : 0 < a) : 
  ∀ x, (x < Real.log (1 / a) → (f x a) < (f (Real.log (1 / a)) a)) ∧ (Real.log (1 / a) < x → (f (Real.log (1 / a)) a) < f x a) :=
sorry

theorem lower_bound_fx {x a : ℝ} (h : 0 < a) : f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_case1_monotonicity_case2_lower_bound_fx_l470_470663


namespace decimal_to_fraction_l470_470034

theorem decimal_to_fraction :
  (3.56 : ℚ) = 89 / 25 := 
sorry

end decimal_to_fraction_l470_470034


namespace least_number_added_1054_l470_470466

theorem least_number_added_1054 (x d: ℕ) (h_cond: 1054 + x = 1058) (h_div: d = 2) : 1058 % d = 0 :=
by
  sorry

end least_number_added_1054_l470_470466


namespace parabola_equation_l470_470623

theorem parabola_equation
  (p : ℝ) (h_pos : p > 0)
  (A B : ℝ × ℝ)
  (hC1A : A ∈ SetOf fun pt => pt.1^2 + (pt.2-2)^2 = 4)
  (hC1B : B ∈ SetOf fun pt => pt.1^2 + (pt.2-2)^2 = 4)
  (hC2A : A.2^2 = 2 * p * A.1)
  (hC2B : B.2^2 = 2 * p * B.1)
  (h_dist : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 5 / 5) :
  (2 * p = 32 / 5) :=
by
  sorry

end parabola_equation_l470_470623


namespace labourer_income_l470_470411

noncomputable def monthly_income : ℤ := 75

theorem labourer_income:
  ∃ (I D : ℤ),
  (80 * 6 = 480) ∧
  (I * 6 - D + (I * 4) = 480 + 240 + D + 30) →
  I = monthly_income :=
by
  sorry

end labourer_income_l470_470411


namespace range_of_a_l470_470188

-- Defining the piecewise function f(x) as given in the problem statement.
def f (a : ℝ) : ℝ -> ℝ :=
  λ x, if x < 0 then x^2 + (4 * a - 3) * x + 3 * a
       else if 0 ≤ x ∧ x < π / 2 then -sin x
       else 0  -- f(x) is not defined outside these ranges, so we define it as 0 for other x.

-- Stating the theorem to find the range of 'a' such that f(x) is monotonically decreasing.
theorem range_of_a (a : ℝ) : 0 ≤ a ∧ a ≤ 3 / 4 :=
begin
  sorry
end

end range_of_a_l470_470188


namespace walking_direction_l470_470244

theorem walking_direction (h_south_pos: south = +1) (h_south_dist: dist_south = 48) (h_north_dist: dist_north = 32): dist_north_sign = - (dist_north) :=
by
  -- Conditions Provided:
  -- h_south_pos: south = +1 means walking south is denoted as positive
  -- h_south_dist: distance traveled south is 48m
  -- h_north_dist: distance traveled north is 32m
  
  -- Therefore, upon negating the condition for the north direction:
  -- dist_north_sign: distance traveled north should be -32m
  sorry

end walking_direction_l470_470244


namespace base9_perfect_square_l470_470692

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : ∃ k : ℕ, (729 * a + 81 * b + 36 + d) = k * k) :
    d = 0 ∨ d = 1 ∨ d = 4 ∨ d = 7 :=
sorry

end base9_perfect_square_l470_470692


namespace problem1_problem2_l470_470990

-- Definition of parametric equations and identity
def parametric_eq1 (φ : ℝ) : ℝ := 2 * cos φ
def parametric_eq2 (φ : ℝ) : ℝ := 2 * sin φ
def trig_identity (φ : ℝ) : Prop := (sin φ) ^ 2 + (cos φ) ^ 2 = 1

-- Standard Cartesian equation of circle
def cartesian_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

-- Polar equations
def polar_circle_eq (ρ θ : ℝ) : Prop := (ρ * cos θ - 2) ^ 2 + ρ ^ 2 * (sin θ) ^ 2 = 4
def polar_line_eq (ρ θ : ℝ) : Prop := ρ * cos θ = 4

-- Proof problems
theorem problem1 (φ x y : ℝ) (h1 : parametric_eq1 φ = x - 2) (h2 : parametric_eq2 φ = y) (h3 : trig_identity φ) :
  cartesian_eq x y := 
by sorry

theorem problem2 (x y ρ θ : ℝ) (h4 : cartesian_eq x y) :
  polar_circle_eq ρ θ ∧ polar_line_eq ρ θ :=
by sorry

end problem1_problem2_l470_470990


namespace inequality_proof_l470_470335

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l470_470335


namespace perfect_square_condition_l470_470595

theorem perfect_square_condition (n : ℤ) : 
    ∃ k : ℤ, n^2 + 6*n + 1 = k^2 ↔ n = 0 ∨ n = -6 := by
  sorry

end perfect_square_condition_l470_470595


namespace base_length_of_isosceles_triangle_l470_470018

theorem base_length_of_isosceles_triangle (r p R : ℝ) (h₁ : r < p) (h₂ : r > 0) (h₃ : p > 0) (h₄ : R > r) (h₅ : R > p) :
  let O₁O₂ := r + p
  let OO₁ := R - r
  let OO₂ := R - p
  let angle_O₁OO₂ := ∠(O₁, O, O₂)
  in 2 * (Mathlib.Real.pi) / 3 < angle_O₁OO₂ →
  O₁O₂ = R - r :=
sorry

end base_length_of_isosceles_triangle_l470_470018


namespace cosine_of_acute_angle_l470_470069

noncomputable def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let norm_v1 := Real.sqrt (v1.1^2 + v1.2^2)
  let norm_v2 := Real.sqrt (v2.1^2 + v2.2^2)
  dot_product / (norm_v1 * norm_v2)

theorem cosine_of_acute_angle :
  cos_angle (4, 3) (2, 5) = 23 / (5 * Real.sqrt 29) :=
by
  sorry

end cosine_of_acute_angle_l470_470069


namespace polygons_vertices_do_not_necessarily_coincide_l470_470203

-- Definitions for the problem
variables {α : Type*} [has_lt α] [has_le α]
variables (F F' : set (α × α))

def equal_polygons (F F' : set (α × α)) : Prop :=
  ∀ (x y : α × α), (x ∈ F ∧ y ∈ F') → (x = y)


-- We need to prove that the assertion "all vertices of these polygons coincide" is false.
theorem polygons_vertices_do_not_necessarily_coincide
  (F F' : set (α × α))
  (H1 : equal_polygons F F')
  (H2 : ∀ v : α × α, v ∈ F → v ∈ F') :
  ¬(∀ v : α × α, v ∈ F → ∃ w ∈ F', v = w) :=
sorry

end polygons_vertices_do_not_necessarily_coincide_l470_470203


namespace weeks_worked_l470_470015

theorem weeks_worked (days_per_week hours_per_day : ℕ)
  (regular_rate overtime_rate total_earnings total_hours : ℚ) :
  days_per_week = 6 →
  hours_per_day = 10 →
  regular_rate = 2.10 →
  overtime_rate = 4.20 →
  total_earnings = 525 →
  total_hours = 245 →
  ∃ W : ℕ, (W * days_per_week * hours_per_day * regular_rate
    + (total_hours - W * days_per_week * hours_per_day) * overtime_rate) = total_earnings
    ∧ (W * days_per_week * hours_per_day + (total_hours - W * days_per_week * hours_per_day)) = total_hours
  → W = 4 :=
begin
  intros h_days h_hours h_reg_rate h_ovrt_rate h_tot_earnings h_tot_hours,
  have h1 : regular_rate * (days_per_week * hours_per_day) * W
          + overtime_rate * (total_hours - regular_rate * (days_per_week * hours_per_day) * W) = total_earnings, 
  { sorry },
  have h2 : (days_per_week * hours_per_day * W
          + (total_hours - days_per_week * hours_per_day * W)) = total_hours, 
  { sorry },
  use 4,
  sorry
end

end weeks_worked_l470_470015


namespace angle_ABC_is_60_l470_470252

-- Given: A, B, C, D, E form an irregular pentagon
-- \(\angle ABC = 2 \cdot \(\angle DBE\).
-- The sum of the internal angles of a pentagon is \(540^\circ\).

def irregular_pentagon (A B C D E : Type) :=
  ∃ (α β : ℝ),
    ∠ABC = 2 * ∠DBE ∧
    (∠ABC + ∠ABE + ∠BCD + ∠CDE + ∠DEA = 540) ∧
    (∠ABE = 180 - 2 * α) ∧
    (∠BCD = 180 - 2 * β) ∧
    (∠DBE = 180 - α - β)

theorem angle_ABC_is_60 
  {A B C D E : Type}
  (h : irregular_pentagon A B C D E) :
  ∠ABC = 60 :=
begin
  sorry
end

end angle_ABC_is_60_l470_470252


namespace number_of_hardcover_copies_sold_l470_470542

constant total_sales_paper_cover : ℝ := 32000 * 0.20
constant author_earnings_paper_cover : ℝ := 0.06 * total_sales_paper_cover
constant total_earnings : ℝ := 1104
constant author_earnings_hardcover : ℝ := total_earnings - author_earnings_paper_cover
constant total_sales_hardcover : ℝ := author_earnings_hardcover / 0.12
constant price_per_hardcover : ℝ := 0.40
noncomputable def hardcover_copies_sold : ℝ := total_sales_hardcover / price_per_hardcover

theorem number_of_hardcover_copies_sold : hardcover_copies_sold = 15000 :=
by
  sorry

end number_of_hardcover_copies_sold_l470_470542


namespace exponential_function_increasing_l470_470174

theorem exponential_function_increasing {m n : ℝ} (h : 2^m > 2^n) : m > n :=
by sorry

end exponential_function_increasing_l470_470174


namespace eccentricity_of_ellipse_l470_470641

theorem eccentricity_of_ellipse {a b c e : ℝ} 
  (h1 : b^2 = 3) 
  (h2 : c = 1 / 4)
  (h3 : a^2 = b^2 + c^2)
  (h4 : a = 7 / 4) 
  : e = c / a → e = 1 / 7 :=
by 
  intros
  sorry

end eccentricity_of_ellipse_l470_470641


namespace largest_three_digit_divisible_by_digits_and_11_l470_470139

theorem largest_three_digit_divisible_by_digits_and_11 (n : ℕ) : 
  (∃ (h : n = 924), 100 ≤ n ∧ n < 1000 ∧
  (∀ d ∈ [9, 2, 4], d ≠ 0 ∧ n % d = 0) ∧
  n % 11 = 0) :=
by
  use 924
  repeat {intro h},
  sorry

end largest_three_digit_divisible_by_digits_and_11_l470_470139


namespace projection_of_5a_minus_3b_onto_a_l470_470679

variable (m : ℝ)
def veca := (2 * m - 1, 2)
def vecb := (-2, 3 * m - 2)
def ab_sum := (veca.1 + vecb.1, veca.2 + vecb.2)
def ab_diff := (veca.1 - vecb.1, veca.2 - vecb.2)

theorem projection_of_5a_minus_3b_onto_a :
  (ab_sum.1 ^ 2 + ab_sum.2 ^ 2 = ab_diff.1 ^ 2 + ab_diff.2 ^ 2) →
  (m = 1) →
  let u := (5 * veca.1 - 3 * vecb.1, 5 * veca.2 - 3 * vecb.2) in
  let v := veca in
  ((u.1 * v.1 + u.2 * v.2) / Math.sqrt (v.1 ^ 2 + v.2 ^ 2)) = 5 * Math.sqrt 5 :=
by
  sorry

end projection_of_5a_minus_3b_onto_a_l470_470679


namespace maximum_value_of_f_l470_470141

noncomputable def f (x : ℝ) : ℝ := real.sqrt x - 2 * x

theorem maximum_value_of_f : ∀ (x : ℝ), x ≥ 0 → f x ≤ 1 / 8 ∧ (f x = 1 / 8 → x = 1 / 16) := 
by 
  sorry

end maximum_value_of_f_l470_470141


namespace range_of_x_range_of_a_l470_470613

-- Part (1): 
theorem range_of_x (x : ℝ) : 
  (a = 1) → (x^2 - 6 * a * x + 8 * a^2 < 0) → (x^2 - 4 * x + 3 ≤ 0) → (2 < x ∧ x ≤ 3) := sorry

-- Part (2):
theorem range_of_a (a : ℝ) : 
  (a ≠ 0) → (∀ x, (x^2 - 4 * x + 3 ≤ 0) → (x^2 - 6 * a * x + 8 * a^2 < 0)) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 4) := sorry

end range_of_x_range_of_a_l470_470613


namespace inequality_holds_for_all_real_numbers_l470_470368

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l470_470368


namespace midpoint_of_complex_numbers_l470_470716

theorem midpoint_of_complex_numbers :
  let A := (1 - 1*I) / (1 + 1)
  let B := (1 + 1*I) / (1 + 1)
  (A + B) / 2 = 1 / 2 := by
sorry

end midpoint_of_complex_numbers_l470_470716


namespace boat_speed_in_still_water_l470_470058

/-- Prove the speed of the boat in still water given the conditions -/
theorem boat_speed_in_still_water (V_s : ℝ) (T : ℝ) (D : ℝ) (V_b : ℝ) :
  V_s = 4 ∧ T = 4 ∧ D = 112 ∧ (D / T = V_b + V_s) → V_b = 24 := sorry

end boat_speed_in_still_water_l470_470058


namespace abs_complex_expression_l470_470055

theorem abs_complex_expression : |2 * (complex.I) * (1 - 2 * (complex.I))| = 2 * Real.sqrt 5 := 
by
  sorry

end abs_complex_expression_l470_470055


namespace find_x0_l470_470985

noncomputable def f (x : ℝ) : ℝ := x * (2016 + Real.log x)

theorem find_x0 :
  ∃ x0 : ℝ, (f' x0 = 2017) ↔ (x0 = 1) :=
sorry

end find_x0_l470_470985


namespace relationship_of_y_values_l470_470694

theorem relationship_of_y_values (m : ℝ) (y1 y2 y3 : ℝ) :
  (∀ x y, (x = -2 ∧ y = y1 ∨ x = -1 ∧ y = y2 ∨ x = 1 ∧ y = y3) → (y = (m^2 + 1) / x)) →
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_of_y_values_l470_470694


namespace three_point_five_six_as_fraction_l470_470030

theorem three_point_five_six_as_fraction : (356 / 100 : ℝ) = (89 / 25 : ℝ) :=
begin
  sorry
end

end three_point_five_six_as_fraction_l470_470030


namespace total_oranges_and_weight_l470_470124

theorem total_oranges_and_weight 
  (oranges_per_child : ℕ) (num_children : ℕ) (average_weight_per_orange : ℝ)
  (h1 : oranges_per_child = 3)
  (h2 : num_children = 4)
  (h3 : average_weight_per_orange = 0.3) :
  oranges_per_child * num_children = 12 ∧ (oranges_per_child * num_children : ℝ) * average_weight_per_orange = 3.6 :=
by
  sorry

end total_oranges_and_weight_l470_470124


namespace sum_of_odd_divisors_360_l470_470462

theorem sum_of_odd_divisors_360 : 
  let n := 360 in
  let prime_factors := 2^3 * 3^2 * 5^1 in
  let odd_divisors_sum :=
    let a := ∑ k in (finset.range 3), 3^k in  -- Sum for 3^0, 3^1, 3^2 
    let b := ∑ k in (finset.range 2), 5^k in  -- Sum for 5^0, 5^1
    a * b
  in
  odd_divisors_sum = 78 :=
by
  let a := ∑ k in (finset.range 3), 3^k
  let b := ∑ k in (finset.range 2), 5^k
  let odd_divisors_sum := a * b
  have ha : a = 13 := by sorry   -- Sum of 3^0 + 3^1 + 3^2
  have hb : b = 6 := by sorry    -- Sum of 5^0 + 5^1
  have hsum : odd_divisors_sum = 13 * 6 := by sorry
  show odd_divisors_sum = 78, from by rw [hsum, ha, hb]; exact rfl

end sum_of_odd_divisors_360_l470_470462


namespace inequality_proof_l470_470624

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) (h_sum : a + b + c + d ≥ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := 
by 
  sorry

end inequality_proof_l470_470624


namespace intersecting_line_relation_l470_470164

variables {A B C D A1 B1 C1 D1 H F X : ℝ} -- Coordinates assumed to be in ℝ for simplicity
variable (l : ℝ → ℝ)

-- Definitions and conditions based on the problem statement
def base_square : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
  dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1

def center_of_symmetry : Prop :=
  H = (A + B + C + D) / 4

def midpoint_F : Prop :=
  F = (A + A1) / 2

def passes_through_center (l : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, l t = H

def intersects_BC1 (l : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, l t = B ∨ l t = C1

def intersects_FB1 (l : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, l t = F ∨ l t = B1

-- The final proof statement
theorem intersecting_line_relation :
  base_square ∧ center_of_symmetry ∧ midpoint_F ∧ 
  passes_through_center l ∧ intersects_BC1 l ∧ intersects_FB1 l →
  dist X F = 2 * dist B1 F :=
begin
  sorry
end

end intersecting_line_relation_l470_470164


namespace brad_trips_to_fill_barrel_l470_470098

noncomputable def volume_hemisphere_bucket (r : ℝ) : ℝ :=
  (2 / 3) * π * r^3

noncomputable def volume_cylindrical_barrel (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem brad_trips_to_fill_barrel : 
  ∀ (r1 r2 h : ℝ), r1 = 8 → r2 = 8 → h = 24 →
  let v_bucket := volume_hemisphere_bucket r1 in
  let v_barrel := volume_cylindrical_barrel r2 h in
  (v_barrel / v_bucket).ceil = 5 :=
by
  intros r1 r2 h r1_def r2_def h_def
  let v_bucket := volume_hemisphere_bucket r1
  let v_barrel := volume_cylindrical_barrel r2 h
  sorry

end brad_trips_to_fill_barrel_l470_470098


namespace rectangle_area_l470_470385

theorem rectangle_area (x : ℝ) (h1 : x > 0) (h2 : x * 4 = 28) : x = 7 :=
sorry

end rectangle_area_l470_470385


namespace number_of_tiles_per_row_in_square_room_l470_470402

theorem number_of_tiles_per_row_in_square_room 
  (area_of_floor: ℝ)
  (tile_side_length: ℝ)
  (conversion_rate: ℝ)
  (h1: area_of_floor = 400) 
  (h2: tile_side_length = 8) 
  (h3: conversion_rate = 12) 
: (sqrt area_of_floor * conversion_rate / tile_side_length) = 30 := 
by
  sorry

end number_of_tiles_per_row_in_square_room_l470_470402


namespace problem_solution_l470_470253

noncomputable def triangle_ABC (a b : ℝ) (cosC : ℝ) :=
  ∃ c A : ℝ, a = 7 ∧ b = 3 ∧ cosC = 11 / 14 ∧
  c = 5 ∧ A = 120 ∧ 
  ∃ sin_val : ℝ, sin_val = sin (2 * acos (11/14) - π / 6) ∧ sin_val = 71 / 98

theorem problem_solution : triangle_ABC 7 3 (11 / 14) :=
sorry

end problem_solution_l470_470253


namespace fraction_relation_l470_470225

theorem fraction_relation (a b : ℝ) (h : a / b = 2 / 3) : (a - b) / b = -1 / 3 :=
by
  sorry

end fraction_relation_l470_470225


namespace possible_distances_between_andrey_and_gleb_l470_470422

theorem possible_distances_between_andrey_and_gleb (A B V G : Point) 
  (d_AB : ℝ) (d_VG : ℝ) (d_BV : ℝ) (d_AG : ℝ)
  (h1 : d_AB = 600) 
  (h2 : d_VG = 600) 
  (h3 : d_AG = 3 * d_BV) : 
  d_AG = 900 ∨ d_AG = 1800 :=
by {
  sorry
}

end possible_distances_between_andrey_and_gleb_l470_470422


namespace range_of_m_plus_n_l470_470610

theorem range_of_m_plus_n (m n : ℝ)
  (tangent_condition : (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 = 1)) :
  m + n ∈ (Set.Iic (2 - 2*Real.sqrt 2) ∪ Set.Ici (2 + 2*Real.sqrt 2)) :=
sorry

end range_of_m_plus_n_l470_470610


namespace geometric_sequence_general_formula_sum_first_n_terms_min_positive_integer_m_l470_470204

noncomputable def a : ℕ → ℕ
| 1 := 1
| (n + 2) := 2 * a (n + 1) + 1

def c (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

def T (n : ℕ) : ℚ := (∑ i in range (n + 1), c i)

theorem geometric_sequence {a_n : ℕ → ℕ} (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, (a n + 1) = 2 * 2 ^ (n - 1) := sorry

theorem general_formula {a_n : ℕ → ℕ} :
  ∀ n, a n = 2 ^ n - 1 := sorry

theorem sum_first_n_terms {c_n : ℕ → ℚ} :
  ∀ n, T n = (n : ℚ) / (6 * n + 9) := sorry

theorem min_positive_integer_m (m : ℕ) :
  ∀ (T : ℕ → ℚ) (a_n : ℕ → ℕ), (∀ n, T n > 1 / a m) → m = 5 := sorry

end geometric_sequence_general_formula_sum_first_n_terms_min_positive_integer_m_l470_470204


namespace sugar_initial_weight_l470_470425

theorem sugar_initial_weight (packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) (used_percentage : ℝ)
  (h1 : packs = 30)
  (h2 : pack_weight = 350)
  (h3 : leftover = 50)
  (h4 : used_percentage = 0.60) : 
  (packs * pack_weight + leftover) = 10550 :=
by 
  sorry

end sugar_initial_weight_l470_470425


namespace min_distance_l470_470988

axiom tangent_line_circle
  (a b : ℝ)
  (tangent : (∀ x y : ℝ, (x / a + y / b = 1) ∧ (x^2 + y^2 = 1) → False)) :
  (1 / a^2 + 1 / b^2 = 1) → (real.sqrt (a^2 + b^2) ≥ 2)

theorem min_distance (a b : ℝ)
  (tangent_condition : ∀ x y : ℝ, (x / a + y / b = 1) ∧ (x^2 + y^2 = 1) → False)
  (condition : 1 / a^2 + 1 / b^2 = 1) : 
  real.sqrt (a^2 + b^2) = 2 := 
by
  exact eq.symm (le_antisymm (tangent_line_circle a b tangent_condition condition) (show 2 ≤ real.sqrt (a^2 + b^2), from sorry))

end min_distance_l470_470988


namespace yolks_in_carton_l470_470874

/-- A local farm is famous for having lots of double yolks in their eggs. One carton of 12 eggs had five eggs with double yolks. Prove that the total number of yolks in the whole carton is equal to 17. -/
theorem yolks_in_carton (total_eggs : ℕ) (double_yolk_eggs : ℕ) (single_yolk_per_egg : ℕ) (double_yolk_per_egg : ℕ) 
    (total_eggs = 12) (double_yolk_eggs = 5) (single_yolk_per_egg = 1) (double_yolk_per_egg = 2) : 
    (double_yolk_eggs * double_yolk_per_egg + (total_eggs - double_yolk_eggs) * single_yolk_per_egg) = 17 := 
by
    sorry

end yolks_in_carton_l470_470874


namespace inequality_holds_for_all_reals_l470_470346

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l470_470346


namespace inequality_xyz_l470_470329

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470329


namespace repeating_decimal_as_fraction_l470_470911

def repeating_decimal := 567 / 999

theorem repeating_decimal_as_fraction : repeating_decimal = 21 / 37 := by
  sorry

end repeating_decimal_as_fraction_l470_470911


namespace lines_through_origin_l470_470088

-- Define that a, b, c are in geometric progression
def geo_prog (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the property of the line passing through the common point (0, 0)
def passes_through_origin (a b c : ℝ) : Prop :=
  ∀ x y, (a * x + b * y = c) → (x = 0 ∧ y = 0)

theorem lines_through_origin (a b c : ℝ) (h : geo_prog a b c) : passes_through_origin a b c :=
by
  sorry

end lines_through_origin_l470_470088


namespace fractions_order_l470_470473

theorem fractions_order :
  let frac1 := (21 : ℚ) / (17 : ℚ)
  let frac2 := (23 : ℚ) / (19 : ℚ)
  let frac3 := (25 : ℚ) / (21 : ℚ)
  frac3 < frac2 ∧ frac2 < frac1 :=
by sorry

end fractions_order_l470_470473


namespace modulus_z_l470_470981

noncomputable def z : ℂ := (1 + 2 * complex.I) / (3 - complex.I)

theorem modulus_z : complex.abs z = real.sqrt 2 / 2 :=
by sorry

end modulus_z_l470_470981


namespace game_cannot_end_if_n_ge_1994_game_must_end_if_n_lt_1994_l470_470094

-- Define the conditions
def number_of_girls : ℕ := 1994

def initial_cards (n : ℕ) : Prop := n ≥ 0

def game_invariant (n : ℕ) : Prop :=
  ∀ girls : Fin number_of_girls → ℕ,  -- girls is a function from 0 to 1993 to the number of cards they hold
    (∀ i, girls i ≥ 0) ∧
    (∀ i, ∃ j k, i ≠ j ∧ i ≠ k ∧ girls i ≥ 2 → girls j := girls j + 1 ∧ girls k := girls k + 1 ∧ girls i := girls i - 2)
    
-- Statements for part (a)
theorem game_cannot_end_if_n_ge_1994 (n : ℕ) (h : n ≥ number_of_girls) :
  ¬ ∀ girls : Fin number_of_girls → ℕ,  -- assumes at the end each girl holds at most 1 card
      (∀ i, girls i ≤ 1) := sorry

-- Statements for part (b)
theorem game_must_end_if_n_lt_1994 (n : ℕ) (h : n < number_of_girls) :
  ∃ girls : Fin number_of_girls → ℕ,  -- assumes at the end each girl holds at most 1 card
    (∀ i, girls i ≤ 1) := sorry

end game_cannot_end_if_n_ge_1994_game_must_end_if_n_lt_1994_l470_470094


namespace compute_AX_l470_470293

-- Define the conditions
variables {ω : EuclideanSpace ℝ (Fin 2)} {A C D B E X : EuclideanSpace ℝ (Fin 2)}
variables (radius : ℝ) (CD : ℝ) (DB_perp_AC : ℝ) (D_midpoint_DB : ℝ) (tangent_intersects : ℝ)

-- Radius of the circle
-- Circle ω has radius 1 and AC is its diameter
def circle_radius (ω : EuclideanSpace ℝ (Fin 2)) (A C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A C = 2 * radius

-- CD is a specific length
def point_D_on_AC (C D : EuclideanSpace ℝ (Fin 2)) : Prop := 
  dist C D = CD

-- DB is perpendicular to AC
def db_perpendicular_ac (D B A C : EuclideanSpace ℝ (Fin 2)) : Prop := 
  DB_perp_AC = ∠ B D A = π/2 ∧ ∠ B D C = π/2

-- E is the midpoint of DB
def e_midpoint_db (D B E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist D E = dist B E ∧ D_midpoint_DB = (D + B) / 2

-- Tangent at B intersects CE at X
def tangent_intersects_ce (ω : EuclideanSpace ℝ (Fin 2)) (B C E X : EuclideanSpace ℝ (Fin 2)) : Prop :=
  tangent_intersects = ∀ (tangent : Line ℝ B) (CE_line : Line ℝ C E), (tangent.is_tangent ω B) ∧ (CE_line = Line.mk C E) → tangent ∩ CE_line = {X}

-- Prove AX = 3
theorem compute_AX 
  (h1 : circle_radius ω A C)
  (h2 : point_D_on_AC C D)
  (h3 : db_perpendicular_ac D B A C)
  (h4 : e_midpoint_db D B E)
  (h5 : tangent_intersects_ce ω B C E X) : 
  dist A X = 3 := 
sorry

end compute_AX_l470_470293


namespace exist_v_i_l470_470966

/-- Given five real numbers u_0, u_1, u_2, u_3, u_4,
    there exist real numbers v_0, v_1, v_2, v_3, v_4 
    that satisfy the given conditions.
-/
theorem exist_v_i (u : Fin 5 → ℝ) : 
  ∃ (v : Fin 5 → ℝ), 
    (∀ i : Fin 5, ∃ n : ℕ, u i - v i = n) ∧ 
    (∑ i j in Finset.range 5, if i < j then (v i - v j)^2 else 0) < 4 := 
  by
  sorry

end exist_v_i_l470_470966


namespace width_of_road_is_l470_470483

-- Define the problem conditions
def cond1 (r R : ℝ) : Prop := 2 * real.pi * r + 2 * real.pi * R = 88
def cond2 (r R : ℝ) : Prop := r = (1 / 3) * R

-- Define the width of the road
def width_of_road (r R : ℝ) : ℝ := R - r

-- The theorem to prove
theorem width_of_road_is (r R : ℝ) (h1 : cond1 r R) (h2 : cond2 r R) :
  width_of_road r R = 22 / real.pi :=
by sorry

end width_of_road_is_l470_470483


namespace problem_to_prove_l470_470702

-- Let’s define the setting and conditions for the problem
variable (O D C F H A B G E : Type*) [Triangle O D C]
variable (AB AC BD FH : LineSegment)
variable [intersects : Intersects AB FH G]
variable [intersects2 : Intersects AC BD E]
variable [G E H F : Point]
variable (GE EH d : ℝ) (hGE : GE = 1) (hEH : EH = c) (hParallel : FH ∥ OC) (hEF : d = EF)

-- Here's the proposition we aim to prove
theorem problem_to_prove : d = 2 := by
  sorry

end problem_to_prove_l470_470702


namespace intersection_area_of_equilateral_triangles_in_circle_l470_470809

noncomputable def intersection_area_of_equilateral_triangles (R : ℝ) : ℝ :=
  if R > 0 then (√3 * R^2) / 2 else 0

theorem intersection_area_of_equilateral_triangles_in_circle (R : ℝ) (hR : R > 0) :
  intersection_area_of_equilateral_triangles R = (√3 * R^2) / 2 :=
sorry

end intersection_area_of_equilateral_triangles_in_circle_l470_470809


namespace mindy_messages_total_l470_470753

theorem mindy_messages_total (P : ℕ) (h1 : 83 = 9 * P - 7) : 83 + P = 93 :=
  by
    sorry

end mindy_messages_total_l470_470753


namespace solve_trig_equation_l470_470770

theorem solve_trig_equation (n : ℤ) :
  ∃ x, 
    (sin x ^ 3 + 6 * cos x ^ 3 + (1 / Real.sqrt 2) * sin (2 * x) * sin (x + Real.pi / 4) = 0) ∧ 
    (x = -Real.arctan 2 + Real.pi * n) :=
sorry

end solve_trig_equation_l470_470770


namespace compare_fractions_l470_470472

theorem compare_fractions:
  let x := 1234567 in
  let y := 7654321 in
  x / y < (x + 1) / (y + 1) :=
by
  sorry

end compare_fractions_l470_470472


namespace surcharge_X_is_2_17_percent_l470_470447

def priceX : ℝ := 575
def priceY : ℝ := 530
def surchargeY : ℝ := 0.03
def totalSaved : ℝ := 41.60

theorem surcharge_X_is_2_17_percent :
  let surchargeX := (2.17 / 100)
  let totalCostX := priceX + (priceX * surchargeX)
  let totalCostY := priceY + (priceY * surchargeY)
  (totalCostX - totalCostY = totalSaved) →
  surchargeX * 100 = 2.17 :=
by
  sorry

end surcharge_X_is_2_17_percent_l470_470447


namespace republican_support_for_A_l470_470705

theorem republican_support_for_A
  (V : ℝ) -- total number of voters
  (D : ℝ) (R : ℝ) -- number of Democrats and Republicans
  (d_support : ℝ) (r_support : ℝ) -- percent of Democrats and Republicans supporting A
  (total_support : ℝ) -- percent of total support for A
  (h1 : D = 0.60 * V)
  (h2 : R = 0.40 * V)
  (h3 : d_support = 0.75)
  (h4 : total_support = 0.57) :
  r_support = 0.30 :=
by
  have h_d_votes : D * d_support = 0.45 * V, from calc
    D * d_support = (0.60 * V) * 0.75 : by rw [h1, h3]
               ... = 0.45 * V : by norm_num,
  have h_A_votes : D * d_support + R * r_support = total_support * V, from calc
    D * d_support + R * r_support = 0.45 * V + R * r_support : by rw h_d_votes
                          ... = 0.57 * V : by rw h4,
  have h_r_A_votes : R * r_support = 0.12 * V, from calc
    R * r_support = 0.57 * V - 0.45 * V : by rw <- h_A_votes
               ... = 0.12 * V : by norm_num,
  have h_final : r_support = 0.30, from calc
    r_support = (0.12 * V) / (0.40 * V) : by rw h_r_A_votes; ring
           ... = 0.30 : by norm_num,
  exact h_final

end republican_support_for_A_l470_470705


namespace count_measures_of_angle_A_l470_470423

theorem count_measures_of_angle_A :
  ∃ n : ℕ, n = 17 ∧
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A + B = 180 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (∀ (A' B' : ℕ), A' > 0 ∧ B' > 0 ∧ A' + B' = 180 ∧ (∀ k : ℕ, k ≥ 1 ∧ A' = k * B') → n = 17) :=
sorry

end count_measures_of_angle_A_l470_470423


namespace find_y_given_conditions_l470_470223

theorem find_y_given_conditions : 
  ∀ (x y : ℝ), (1.5 * x = 0.75 * y) ∧ (x = 24) → (y = 48) :=
by
  intros x y h
  cases h with h1 h2
  rw h2 at h1
  sorry

end find_y_given_conditions_l470_470223


namespace inequality_xyz_l470_470328

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l470_470328


namespace chosen_number_l470_470879

theorem chosen_number (x : ℕ) (h : (x / 12) - 240 = 8) : x = 2976 :=
sorry

end chosen_number_l470_470879


namespace mrs_thompson_no_contact_days_l470_470305

open Nat

theorem mrs_thompson_no_contact_days :
  let days_in_2016 := 366
  let contact_every_2_days := (days_in_2016 / 2)
  let contact_every_6_days := (days_in_2016 / 6)
  let contact_every_4_days := (days_in_2016 / 4)
  let lcm_2_6 := lcm 2 6
  let lcm_2_4 := lcm 2 4
  let lcm_6_4 := lcm 6 4
  let lcm_2_6_4 := lcm (lcm 2 6) 4
  let contact_2_6_days := (days_in_2016 / lcm_2_6)
  let contact_2_4_days := (days_in_2016 / lcm_2_4)
  let contact_6_4_days := (days_in_2016 / lcm_6_4)
  let contact_2_6_4_days := (days_in_2016 / lcm_2_6_4)
  let total_contact_days := contact_every_2_days + contact_every_6_days + contact_every_4_days - contact_2_6_days - contact_2_4_days - contact_6_4_days + contact_2_6_4_days
  let no_contact_days := days_in_2016 - total_contact_days
  no_contact_days = 183 := by
  let days_in_2016 := 366
  let contact_every_2_days := Nat.floor (days_in_2016 / 2)
  let contact_every_6_days := Nat.floor (days_in_2016 / 6)
  let contact_every_4_days := Nat.floor (days_in_2016 / 4)
  let lcm_2_6 := lcm 2 6
  let lcm_2_4 := lcm 2 4
  let lcm_6_4 := lcm 6 4
  let lcm_2_6_4 := lcm (lcm 2 6) 4
  let contact_2_6_days := Nat.floor (days_in_2016 / lcm_2_6)
  let contact_2_4_days := Nat.floor (days_in_2016 / lcm_2_4)
  let contact_6_4_days := Nat.floor (days_in_2016 / lcm_6_4)
  let contact_2_6_4_days := Nat.floor (days_in_2016 / lcm_2_6_4)
  let total_contact_days := contact_every_2_days + contact_every_6_days + contact_every_4_days - contact_2_6_days - contact_2_4_days - contact_6_4_days + contact_2_6_4_days
  let no_contact_days := days_in_2016 - total_contact_days
  exact no_contact_days = 183

end mrs_thompson_no_contact_days_l470_470305
