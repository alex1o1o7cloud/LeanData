import Mathlib

namespace height_increase_each_year_l237_237327

/-- Define the annual growth of the tree height as a constant h -/
def annual_growth (height : ℝ) (years : ℝ) : ℝ :=
  height + years * h

/-- Define the initial height of the tree -/
def initial_height : ℝ := 4

/-- Prove the yearly increase in height (h) -/
theorem height_increase_each_year (h : ℝ) :
  let height4 := initial_height + 4 * h
  let height6 := initial_height + 6 * h
  height6 = height4 + (1 / 3) * height4 → h = 2 / 3 :=
by
  sorry

end height_increase_each_year_l237_237327


namespace evaluate_expression_l237_237830

theorem evaluate_expression : (real.sqrt 16) ^ 8 = 256 := 
by 
  let step1 : real.sqrt 16 = 16 ^ (1 / 4) := sorry
  have step2 : (16 ^ (1 / 4)) ^ 8 = 16 ^ ((1 / 4) * 8) := sorry
  have step3 : 16 ^ ((1 / 4) * 8) = 16 ^ 2 := sorry
  have step4 : 16 ^ 2 = 256 := sorry
  sorry

end evaluate_expression_l237_237830


namespace trees_left_unwatered_l237_237350

theorem trees_left_unwatered :
  let total_trees := 29
  let boys_trees := [2, 3, 1, 3, 2, 4, 3, 2, 5]
  (total_trees - boys_trees.sum) = 4 :=
by
  let total_trees := 29
  let boys_trees := [2, 3, 1, 3, 2, 4, 3, 2, 5]
  have h1: boys_trees.sum = 25 := sorry -- sum calculation
  show (total_trees - boys_trees.sum) = 4
  calc
    total_trees - boys_trees.sum
        = 29 - 25 : by rw h1
    ... = 4      : by norm_num

end trees_left_unwatered_l237_237350


namespace magnitude_of_z_l237_237951

noncomputable def z : ℂ := 1 + 2 * complex.i + complex.i ^ 3

theorem magnitude_of_z : complex.abs z = real.sqrt 2 := 
by
  -- this 'sorry' is a placeholder for the actual proof
  sorry

end magnitude_of_z_l237_237951


namespace greatest_integer_third_side_l237_237665

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l237_237665


namespace Desargues_Theorem_l237_237260

variables (O A1 A2 B1 B2 C1 C2 A B C : Type)
variables [line a O A1 A2] [line b O B1 B2] [line c O C1 C2]
variables [intersection A B1 C1 B2 C2] [intersection B C1 A1 C2 A2] [intersection C A1 B1 A2 B2]

theorem Desargues_Theorem (h1 : lies_on A1 a) (h2 : lies_on A2 a)
                          (h3 : lies_on B1 b) (h4 : lies_on B2 b)
                          (h5 : lies_on C1 c) (h6 : lies_on C2 c)
                          (h7 : intersects A (B1, C1) (B2, C2))
                          (h8 : intersects B (C1, A1) (C2, A2))
                          (h9 : intersects C (A1, B1) (A2, B2)):
                          collinear A B C :=
sorry

end Desargues_Theorem_l237_237260


namespace disk_difference_l237_237338

/-- Given the following conditions:
    1. Every disk is either blue, yellow, green, or red.
    2. The ratio of blue disks to yellow disks to green disks to red disks is 3 : 7 : 8 : 4.
    3. The total number of disks in the bag is 176.
    Prove that the number of green disks minus the number of blue disks is 40.
-/
theorem disk_difference (b y g r : ℕ) (h_ratio : b * 7 = y * 3 ∧ b * 8 = g * 3 ∧ b * 4 = r * 3) (h_total : b + y + g + r = 176) : g - b = 40 :=
by
  sorry

end disk_difference_l237_237338


namespace candy_remaining_l237_237545

def initial_candy : ℝ := 1012.5
def talitha_took : ℝ := 283.7
def solomon_took : ℝ := 398.2
def maya_took : ℝ := 197.6

theorem candy_remaining : initial_candy - (talitha_took + solomon_took + maya_took) = 133 := 
by
  sorry

end candy_remaining_l237_237545


namespace sequence_bn_general_formula_l237_237114

theorem sequence_bn_general_formula {p : ℝ} (h : p > 0) (n : ℕ) (n_pos : 0 < n) :
  ∃ a : ℕ → ℝ, a 1 = 2 * p ∧ 
                (∀ n : ℕ, a (n + 1) = 0.5 * (a n + p ^ 2 / a n)) ∧ 
                (∀ b : ℕ → ℝ, b n = (a n + p) / (a n - p) → 
                b n = 3 ^ (2 ^ (n - 1))) :=
begin
  sorry
end

end sequence_bn_general_formula_l237_237114


namespace inequality_sum_powers_l237_237599

theorem inequality_sum_powers (n k : ℕ) :
  ∑ i in Finset.range (n + 1), i ^ k ≤ (n ^ (2 * k) - (n - 1) ^ k) / (n ^ k - (n - 1) ^ k) :=
by sorry

end inequality_sum_powers_l237_237599


namespace find_n_l237_237844

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n % 9 = 4897 % 9 ∧ n = 1 :=
by
  use 1
  sorry

end find_n_l237_237844


namespace point_equidistant_from_neg2_and_4_l237_237588

theorem point_equidistant_from_neg2_and_4 : ∃ x : ℝ, (abs (x + 2) = abs (x - 4)) ∧ x = 1 :=
by
  let x := 1
  have h1 : abs (x + 2) = abs (x - 4)
  {
    -- Mathematical calculations not needed, just the conclusions
    sorry
  }
  existsi x
  split
  { exact h1 }
  { refl }

end point_equidistant_from_neg2_and_4_l237_237588


namespace thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l237_237049

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_number : triangular_number 30 = 465 := 
by
  sorry

theorem sum_of_thirtieth_and_twentyninth_triangular_numbers : triangular_number 30 + triangular_number 29 = 900 := 
by
  sorry

end thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l237_237049


namespace marble_count_l237_237926

def marbleSums (n m k : ℕ) : ℕ :=
  if k = n + m then 1 else 0

def totalSumsFromBag (k : ℕ) : ℕ :=
  (Finset.univ.filter (λ nm : ℕ × ℕ, marbleSums nm.1 nm.2 k = 1)).card

theorem marble_count (n m : ℕ) :
  (Finset.range (8 + 1)).sum (λ k => 2 * totalSumsFromBag k) = 26 :=
by
  sorry

end marble_count_l237_237926


namespace heather_bicycling_time_l237_237931

theorem heather_bicycling_time (distance speed : ℝ) (h_distance : distance = 40) (h_speed : speed = 8) : (distance / speed) = 5 := 
by
  rw [h_distance, h_speed]
  norm_num

end heather_bicycling_time_l237_237931


namespace ratio_of_parallelogram_perpendiculars_l237_237522

variables {A B C D M E F : Type*}
variables [parallelogram A B C D] [point M A C] [perpendicular E M B] [perpendicular F M D]

theorem ratio_of_parallelogram_perpendiculars (ABCD : Type*) (M : Type*) (E : Type*) (F : Type*) 
  [parallelogram ABCD] [point M A C] [perpendicular E M B] [perpendicular F M D] : 
  ∀ (A B C D : point) (M E F : point),
    parallelogram ABCD → 
    M ∈ line A C →
    perpendicular E M (line A B) →
    perpendicular F M (line A D) → 
    (ME / MF) = (AD / AB) :=
by sorry

end ratio_of_parallelogram_perpendiculars_l237_237522


namespace eq_has_no_solution_ACE_l237_237039

theorem eq_has_no_solution_ACE :
  (∀ x : ℝ, ¬((x-3)^2 + 1 = 0)) ∧ 
  (∀ x : ℝ, ¬((√(5 - x)) + 3 = 0)) ∧ 
  (∀ x : ℝ, ¬(|2x+2| + 5 = 0)) :=
by
  sorry

end eq_has_no_solution_ACE_l237_237039


namespace equilateral_triangle_circumcircle_ratio_l237_237205

theorem equilateral_triangle_circumcircle_ratio (a x : ℝ) :
  let R := d / (Real.sqrt 3),
      r := x / (Real.sqrt 3),
      t2 := Real.pi * R^2,
      t1 := Real.pi * r^2,
      d  := Real.sqrt (a^2 - a * x + x^2)
  in
    a > 0 ->
    (let ratio := t2 / t1 in ratio = 3) -> x = a / 2 := 
by
  intros a x hpos hratio
  sorry

end equilateral_triangle_circumcircle_ratio_l237_237205


namespace integer_segments_on_hypotenuse_l237_237231

theorem integer_segments_on_hypotenuse (A B C : Type) [Euclidean_space A B C]
  (right_triangle : right_triangle A B C)
  (AB : segment A B) (BC : segment B C)
  (AB_length : length AB = 18) (BC_length : length BC = 24) :
  num_integer_segments_from_B_to_hypotenuse B A C = 10 :=
sorry

end integer_segments_on_hypotenuse_l237_237231


namespace smallest_n_containing_375_consecutively_l237_237616

theorem smallest_n_containing_375_consecutively :
  ∃ (m n : ℕ), m < n ∧ Nat.gcd m n = 1 ∧ (n = 8) ∧ (∀ (d : ℕ), d < 1000 →
  ∃ (k : ℕ), k * d % n = m ∧ (d / 100) % 10 = 3 ∧ (d / 10) % 10 = 7 ∧ d % 10 = 5) :=
sorry

end smallest_n_containing_375_consecutively_l237_237616


namespace collinearity_P_E_F_l237_237398

open_locale classical
noncomputable theory

variables {A B C D E F P Q : Point}
variables (circle : Circle) (inscribed : cyclic_quadrilateral A B C D circle)
variables (PQ_int : intersect_ext_AB_DC A B D C = P) (QR_int : intersect_ext_AD_BC A D B C = Q)
variables (E_tangent : tangent_point Q circle = E) (F_tangent : tangent_point Q circle = F)

theorem collinearity_P_E_F : collinear P E F :=
sorry

end collinearity_P_E_F_l237_237398


namespace total_marbles_l237_237515

-- Definitions based on given conditions
def ratio_white := 2
def ratio_purple := 3
def ratio_red := 5
def ratio_blue := 4
def ratio_green := 6
def blue_marbles := 24

-- Definition of sum of ratio parts
def sum_of_ratio_parts := ratio_white + ratio_purple + ratio_red + ratio_blue + ratio_green

-- Definition of ratio of blue marbles to total
def ratio_blue_to_total := ratio_blue / sum_of_ratio_parts

-- Proof goal: total number of marbles
theorem total_marbles : blue_marbles / ratio_blue_to_total = 120 := by
  sorry

end total_marbles_l237_237515


namespace evaluate_expression_l237_237836

-- Definition of variables a, b, c as given in conditions
def a : ℕ := 7
def b : ℕ := 11
def c : ℕ := 13

-- The theorem to prove the given expression equals 31
theorem evaluate_expression : 
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 31 :=
by
  sorry

end evaluate_expression_l237_237836


namespace isosceles_right_triangle_count_l237_237126

theorem isosceles_right_triangle_count :
  ∃! (a : ℝ), (2 * a + a * real.sqrt 2) = a^2 :=
sorry

end isosceles_right_triangle_count_l237_237126


namespace perpendicular_lines_l237_237040

-- given conditions
variable {α : Type} [MetricSpace α]

variable (A1 A2 A3 A4 A5 A6 A7 : α)
variable (C : Set α) -- Represents the circle

-- Conditions on the points
variable (h1 : A1 ∈ C)
variable (h2 : A2 ∈ C)
variable (h3 : A3 ∈ C)
variable (h4 : A4 ∈ C)
variable (h5 : A5 ∈ C)
variable (h12 : dist A1 A2 = dist A2 A3)
variable (h23 : dist A2 A3 = dist A3 A4)
variable (h34 : dist A3 A4 = dist A4 A5)
variable (h6 : A6 ∈ C ∧ dist A2 A6 = 2 * radius_of_circle C)
variable (h7 : ∃ x, x ∈ line_through A1 A5 ∧ x ∈ line_through A3 A6 ∧ A7 = x)

-- Goal
theorem perpendicular_lines : ∃ o, is_center_of_circle C o →
perpendicular (line_through A1 A6) (line_through A4 A7) := 
sorry

end perpendicular_lines_l237_237040


namespace count_valid_integers_between_2030_and_2500_with_odd_distinct_increasing_digits_l237_237913

theorem count_valid_integers_between_2030_and_2500_with_odd_distinct_increasing_digits :
  ∃ n, n = 10 ∧ 
    ∀ x, 2030 ≤ x ∧ x ≤ 2500 ∧ 
          (∀ d, d ∈ digits x → odd d) ∧ 
          (∀ (i j : ℕ), i < j → i < size x ∧ j < size x → digits x[i] < digits x[j]) ∧ 
          (∀ (i j : ℕ), i ≠ j → digits x[i] ≠ digits x[j]) → 
          x ∈ valid_integers :=
  sorry

end count_valid_integers_between_2030_and_2500_with_odd_distinct_increasing_digits_l237_237913


namespace solve_for_x_l237_237038

-- Define the given condition using logs
def condition (x : ℝ) : Prop :=
  log 5 (x - 1) + log 5 (x^2 - 1) + log (1/5) (x - 1) = 3

-- The statement to prove
theorem solve_for_x : ∃ (x : ℝ), x > 0 ∧ condition x ∧ x = 3 * real.sqrt 14 := by
  sorry

end solve_for_x_l237_237038


namespace limit_of_expected_value_l237_237731

-- Define the conditions and problem
def n_balls_n_boxes (n : ℕ) : set (ℕ × ℕ):= { p | let (b, boxes) := p in 
  (boxes = n) ∧ (b ≤ n)}

def expected_value_b⁴ (n : ℕ) [uniformly_random (n_balls_n_boxes n)] : ℝ :=
  ∑ k in finset.range n.succ, (k ^ 4 : ℝ) * 
  (nat.choose n k * (1 / n) ^ k * (1 - 1 / n) ^ (n - k)).to_real

def limit_en : ℝ := sorry

theorem limit_of_expected_value : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |expected_value_b⁴ n - 15| < ε := 
by
  sorry

end limit_of_expected_value_l237_237731


namespace minimum_bail_rate_l237_237042

theorem minimum_bail_rate
  (distance : ℝ) (leak_rate : ℝ) (rain_rate : ℝ) (sink_threshold : ℝ) (rowing_speed : ℝ) (time_in_minutes : ℝ) (bail_rate : ℝ) : 
  (distance = 2) → 
  (leak_rate = 15) → 
  (rain_rate = 5) →
  (sink_threshold = 60) →
  (rowing_speed = 3) →
  (time_in_minutes = (2 / 3) * 60) →
  (bail_rate = sink_threshold / (time_in_minutes) - (rain_rate + leak_rate)) →
  bail_rate ≥ 18.5 :=
by
  intros h_distance h_leak_rate h_rain_rate h_sink_threshold h_rowing_speed h_time_in_minutes h_bail_rate
  sorry

end minimum_bail_rate_l237_237042


namespace find_num_unbounded_sequences_l237_237422

def g1 (n : ℕ) : ℕ :=
  if n = 1 then 1
  else let factors := n.factors
       factors.foldl (λ acc p, acc * (p + 2)^(factors.count p - 1)) 1

def g (m n : ℕ) : ℕ :=
  if m = 1 then g1 n else g1 (g (m - 1) n)

def is_unbounded_sequence (N : ℕ) : Prop :=
  ∀ M, ∃ m, g m N > M

theorem find_num_unbounded_sequences : ∃ n, n = 27 ∧ 
  (∃ count, (∀ k, 1 ≤ k ∧ k ≤ 300 → (is_unbounded_sequence k ↔ k ∈ count)) ∧ count.card = 27) := 
sorry

end find_num_unbounded_sequences_l237_237422


namespace number_of_paths_l237_237912

-- Define points involved
inductive Point
| A | B | C | D | E | F | G

-- Define segments in the figure
def segments : set (Point × Point) :=
  { (Point.A, Point.C), (Point.A, Point.G), (Point.A, Point.D),
    (Point.C, Point.B), (Point.C, Point.F), (Point.C, Point.D),
    (Point.D, Point.C), (Point.D, Point.F), (Point.D, Point.E),
    (Point.E, Point.F), (Point.F, Point.B),
    (Point.G, Point.C), (Point.G, Point.D), (Point.G, Point.F), (Point.G, Point.B) }

-- Condition: Paths must not revisit any point
def no_revisit (path : list Point) : Prop :=
  list.nodup path

-- Condition: Valid path in the figure
def valid_path (path : list Point) : Prop :=
  ∀ (p1 p2 : Point), (p1, p2) ∈ segments ∨ (p2, p1) ∈ segments → p1 ∈ path → p2 ∈ path

-- Final goal: Prove there are exactly 21 valid paths from A to B without revisiting points
theorem number_of_paths : 
  ∃ (paths : list (list Point)), 
  (∀ (path : list Point), path ∈ paths → list.head path = some Point.A ∧ list.reverse path = Point.B :: nil ∧ no_revisit path ∧ valid_path path)
  ∧ list.length paths = 21 := sorry

end number_of_paths_l237_237912


namespace john_sleep_hours_l237_237554

/-- 
  To achieve an average score of 85 on the first and second exams combined,
  given that the amount of sleep he gets before an exam and his score are inversely related,
  and having gotten 6 hours of sleep and scored 80 on the first exam,
  John needs to sleep approximately 5.3 hours the night before his second exam.
-/
theorem john_sleep_hours (h s1: ℝ) (inversely_related : ∀ x y, x * y = s1) 
  (s1_value : s1 = 80) (h1 : ℝ) (h1_value: h1 = 6) (s2 : ℝ) (s2_value: s2 = (2 * 85) - s1) 
  (goal : h = 480 / s2) :
  h ≈ 5.3 := 
sorry

end john_sleep_hours_l237_237554


namespace hash_value_l237_237060

variable (x y : ℝ)

def op_hash (x y : ℝ) : ℝ :=
if y = 0 then x
else if x = 0 then y
else (x.hash (y - 1) + 2 * y + 1) - 1

theorem hash_value :
  (∀ x y : ℝ, 0 \# y = y) →
  (∀ x y : ℝ, x \# y = y \# x) →
  (∀ x y : ℝ, (x + 1) \# y = (x \# y) + 2 * y + 1) →
  7 \# 3 = 52 :=
by
  intros h1 h2 h3
  sorry

end hash_value_l237_237060


namespace max_traffic_flow_at_40_range_for_flow_exceeding_10_l237_237779

/-- Conditions for the traffic flow problem --/
def traffic_flow (v : ℝ) := 920 * v / (v^2 + 3 * v + 1600)

/-- Correctness Statement --/
theorem max_traffic_flow_at_40 :
  ∀ v > 0, traffic_flow v ≤ 920 / 83 ∧ traffic_flow 40 = 920 / 83 := 
     sorry

theorem range_for_flow_exceeding_10 :
  ∀ v > 0, traffic_flow v > 10 ↔ 25 < v ∧ v < 64 :=
     sorry

end max_traffic_flow_at_40_range_for_flow_exceeding_10_l237_237779


namespace problems_left_proof_l237_237068

noncomputable def problems_left (total_problems completed_in_20mins: ℕ) : ℕ :=
  let additional_problems := 2 * completed_in_20mins
  let total_completed := completed_in_20mins + additional_problems
  total_problems - total_completed

theorem problems_left_proof (total_problems completed_in_20mins : ℕ) :
  total_problems = 75 → completed_in_20mins = 10 → problems_left total_problems completed_in_20mins = 45 :=
by
  intros h1 h2
  rw [h1, h2]
  simp [problems_left]
  sorry

end problems_left_proof_l237_237068


namespace pens_to_sell_to_make_profit_l237_237352

theorem pens_to_sell_to_make_profit (initial_pens : ℕ) (purchase_price selling_price profit : ℝ) :
  initial_pens = 2000 →
  purchase_price = 0.15 →
  selling_price = 0.30 →
  profit = 150 →
  (initial_pens * selling_price - initial_pens * purchase_price = profit) →
  initial_pens * profit / (selling_price - purchase_price) = 1500 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pens_to_sell_to_make_profit_l237_237352


namespace angle_A_is_60_degrees_l237_237543

theorem angle_A_is_60_degrees
  (a b c : ℝ) (A : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : 0 < A) (h3 : A < 180) : 
  A = 60 := 
  sorry

end angle_A_is_60_degrees_l237_237543


namespace find_c_max_t_l237_237956

theorem find_c (A B C : ℝ) (a b c : ℝ) (h_a : a = 3) (h_b : b = sqrt 13)
  (h_arith : 2 * B = A + C) (h_triangle : A + B + C = π) : c = 4 := sorry

theorem max_t (A B C : ℝ) (a b c t : ℝ) (h_arith : 2 * B = A + C) (h_triangle : A + B + C = π) :
  (t = sin A * sin C) → t ≤ 3 / 4 := sorry

end find_c_max_t_l237_237956


namespace angle_CAB_l237_237516

theorem angle_CAB (circle : Type) (O A B C D : circle) (R : ℝ) 
  (h1 : distance O A = R) (h2 : distance O B = R) (h3 : distance O C = R) 
  (h4 : distance O D = R) (h5 : distance C (line[O, D]) = R / 2) :
  angle C A B = 75 :=
sorry

end angle_CAB_l237_237516


namespace number_of_roots_ge_5_l237_237031

namespace RootProblem

noncomputable def f : ℝ → ℝ := sorry
variable {T : ℝ}
hypothesis odd_f : ∀ x : ℝ, f (-x) = -f x
hypothesis periodic_f : ∀ x : ℝ, f (x + T) = f x
hypothesis T_positive : T > 0

theorem number_of_roots_ge_5 : 
  (∃ n : ℕ, (n = {x : ℝ | f x = 0 ∧ -T ≤ x ∧ x ≤ T}.toFinset.card) ∧ n ≥ 5) 
:= sorry

end RootProblem

end number_of_roots_ge_5_l237_237031


namespace evaluate_ff_neg99_l237_237482

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 10 else 1 - x

theorem evaluate_ff_neg99 : f (f (-99)) = 2 :=
by
  sorry

end evaluate_ff_neg99_l237_237482


namespace minimum_distance_exists_l237_237198

noncomputable def radius : ℝ := Real.sqrt 5

def P (t : ℝ) : ℝ × ℝ := (t, 6 - 2 * t)

def Q (θ : ℝ) : ℝ × ℝ := (1 + Real.sqrt 5 * Real.cos θ, -2 + Real.sqrt 5 * Real.sin θ)

def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem minimum_distance_exists : 
  ∃ t θ, distance (P t) (Q θ) = Real.sqrt 5 / 5 :=
sorry

end minimum_distance_exists_l237_237198


namespace in_range_p_1_to_100_l237_237570

def p (m n : ℤ) : ℤ :=
  2 * m^2 - 6 * m * n + 5 * n^2

-- Predicate that asserts k is in the range of p
def in_range_p (k : ℤ) : Prop :=
  ∃ m n : ℤ, p m n = k

-- Lean statement for the theorem
theorem in_range_p_1_to_100 :
  {k : ℕ | 1 ≤ k ∧ k ≤ 100 ∧ in_range_p k} = 
  {1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100} :=
  by
    sorry

end in_range_p_1_to_100_l237_237570


namespace inscribed_square_area_diagram2_l237_237727

noncomputable def side_length_of_square_in_diagram1 : ℝ := real.sqrt 441
noncomputable def side_length_of_square_in_diagram2 : ℝ := 14 * real.sqrt 2

theorem inscribed_square_area_diagram2 (a : ℝ) (h1 : ∀ (Δ : Type) [is_triangle Δ], isosceles Δ)
  (h2 : a = 2 * side_length_of_square_in_diagram1) :
  (side_length_of_square_in_diagram2)^2 = 392 :=
by
  sorry

end inscribed_square_area_diagram2_l237_237727


namespace sally_baseball_cards_l237_237233

theorem sally_baseball_cards (initial_cards torn_cards purchased_cards : ℕ) 
    (h_initial : initial_cards = 39)
    (h_torn : torn_cards = 9)
    (h_purchased : purchased_cards = 24) :
    initial_cards - torn_cards - purchased_cards = 6 := by
  sorry

end sally_baseball_cards_l237_237233


namespace ratio_of_x_and_y_l237_237937

theorem ratio_of_x_and_y (x y : ℤ) (h : (3 * x - 2 * y) * 4 = 3 * (2 * x + y)) : (x : ℚ) / y = 11 / 6 :=
  sorry

end ratio_of_x_and_y_l237_237937


namespace fourteen_card_shuffle_l237_237018

def shuffle (deck : List ℕ) : List ℕ :=
  deck.reverse

def shuffles_to_original (n : ℕ) (s : ℕ) : Prop :=
  n ≠ 0 ∧ List.foldr (λ _ l, shuffle l) (List.range n) (List.range s) = List.range n

theorem fourteen_card_shuffle : shuffles_to_original 14 14 :=
by 
  -- Details of proof are omitted as per instructions
  sorry

end fourteen_card_shuffle_l237_237018


namespace max_det_M_eq_zero_l237_237847

noncomputable def maximum_determinant_value : ℝ :=
  let M (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
    Matrix.of ![
      ![1, 1, 1],
      ![1 + Real.tan θ, 1, 1],
      ![1, 1, 1 + Real.cos θ]]
  in 0

theorem max_det_M_eq_zero : ∀ θ : ℝ, Matrix.det (M θ) ≤ maximum_determinant_value :=
by sorry -- Proof to be provided

end max_det_M_eq_zero_l237_237847


namespace smallest_nonprime_int_gt_1_with_no_prime_lt_20_l237_237197

def smallest_nonprime_condition : Prop :=
  ∃ n: ℕ, n > 1 ∧ (∀ p: ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529

theorem smallest_nonprime_int_gt_1_with_no_prime_lt_20 : smallest_nonprime_condition :=
sorry

end smallest_nonprime_int_gt_1_with_no_prime_lt_20_l237_237197


namespace dust_particles_after_walking_l237_237234

/-- Samuel cleared nine-tenths of the dust particles. -/
def dust_cleared_fraction : ℝ := 9 / 10

/-- Samuel's shoes left 223 dust particles behind. -/
def dust_left_by_shoes : ℤ := 223

/-- There were 1080 dust particles before Samuel swept. -/
def initial_dust_count : ℤ := 1080

theorem dust_particles_after_walking :
  let dust_cleared := initial_dust_count * (1 - dust_cleared_fraction)
  let remaining_dust := initial_dust_count - dust_cleared.toInt
  remaining_dust + dust_left_by_shoes = 331 :=
by
  sorry

end dust_particles_after_walking_l237_237234


namespace probability_odd_multiple_of_5_l237_237226

theorem probability_odd_multiple_of_5 (cards : Finset ℕ) (h1 : cards = Finset.range 101) :
  let total_cards := (Finset.range 100).card
  let odd_multiple_of_5 := Finset.filter (λ n, (n % 5 = 0) ∧ (n % 2 = 1)) (Finset.range 101)
  let success_cards := odd_multiple_of_5.card
  (success_cards : ℚ / total_cards) = 1 / 10 :=
by
  sorry

end probability_odd_multiple_of_5_l237_237226


namespace hyperbola_equation_l237_237111

-- Define the basic structure of hyperbola and the given conditions
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def correct_hyperbola (x y : ℝ) : Prop := 
  hyperbola x y 1 (sqrt 3) ∨ hyperbola x y (sqrt 3) 1

-- The target theorem to prove
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : c = 2) (h5 : a^2 + b^2 = c^2) (h6 : ∃ θ : ℝ, θ = π / 3 ∧ (b / a = √3 ∨ b / a = 1 / √3)) :
  ∀ x y : ℝ, 
  (hyperbola x y a b ↔ correct_hyperbola x y) :=
by
  intros x y
  sorry

end hyperbola_equation_l237_237111


namespace radius_is_8_l237_237356

noncomputable def find_radius (x y : ℝ) (h1 : x + y = 80 * real.pi) : ℝ :=
  let r := classical.some (show ∃ r : ℝ, x = real.pi * r^2 ∧ y = 2 * real.pi * r, by sorry) in r

theorem radius_is_8 (x y : ℝ) (h1 : x + y = 80 * real.pi) : find_radius x y h1 = 8 :=
begin
  -- Proof details are omitted as they are not required.
  sorry
end

end radius_is_8_l237_237356


namespace phones_left_is_7500_l237_237371

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end phones_left_is_7500_l237_237371


namespace distinct_prime_factors_120_l237_237496

theorem distinct_prime_factors_120 : 
  (Set.toFinset {p : ℕ | p.Prime ∧ p ∣ 120}).card = 3 := 
by 
  -- proof omitted 
  sorry

end distinct_prime_factors_120_l237_237496


namespace lattice_points_count_in_square_l237_237723

def is_lattice_point (x y : ℤ) : Prop :=
  True  -- Here for clear definition that it is a lattice point, always true for integers x and y.

def is_inside_square (x y : ℤ) : Prop :=
  abs x < 2 ∧ abs y < 2

theorem lattice_points_count_in_square : 
  (Finset.card {p : ℤ × ℤ | is_inside_square p.1 p.2}.toFinset) = 6 :=
by 
  -- Here we need to provide the proof
  sorry

end lattice_points_count_in_square_l237_237723


namespace triangle_third_side_l237_237685

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l237_237685


namespace judy_pays_41_35_l237_237997

def carrot_cost (qty : ℕ) (price : ℝ) : ℝ := qty * price
def milk_cost (qty : ℕ) (price : ℝ) : ℝ := qty * price
def pineapple_cost (qty : ℕ) (price : ℝ) (discount : ℝ) : ℝ := qty * price * discount
def flour_cost (price : ℝ) : ℝ := price
def cookies_cost (price : ℝ) : ℝ := price

def total_cost (carrot_cost : ℝ) (milk_cost : ℝ) (pineapple_cost : ℝ) (flour_cost : ℝ) (cookies_cost : ℝ) : ℝ :=
  carrot_cost + milk_cost + pineapple_cost + flour_cost + cookies_cost

def apply_discount (total_cost : ℝ) (threshold : ℝ) (discount_rate : ℝ) : ℝ :=
  if total_cost > threshold then total_cost * discount_rate else total_cost

def apply_coupon (total_cost : ℝ) (threshold : ℝ) (coupon_value : ℝ) : ℝ :=
  if total_cost > threshold then total_cost - coupon_value else total_cost

theorem judy_pays_41_35
  (carrot_qty : ℕ := 7)
  (carrot_price : ℝ := 2)
  (milk_qty : ℕ := 4)
  (milk_price : ℝ := 3)
  (pineapple_qty : ℕ := 3)
  (pineapple_price : ℝ := 5)
  (discount : ℝ := 0.5)
  (flour_price : ℝ := 8)
  (cookies_price : ℝ := 10)
  (total_threshold : ℝ := 40)
  (discount_rate : ℝ := 0.90)
  (coupon_threshold : ℝ := 25)
  (coupon_value : ℝ := 5):
  apply_coupon (apply_discount (total_cost (carrot_cost carrot_qty carrot_price)
                                            (milk_cost milk_qty milk_price)
                                            (pineapple_cost pineapple_qty pineapple_price discount)
                                            (flour_cost flour_price)
                                            (cookies_cost cookies_price))
                               total_threshold discount_rate) coupon_threshold coupon_value = 41.35 :=
by
  sorry

end judy_pays_41_35_l237_237997


namespace sample_size_l237_237746

theorem sample_size {n : ℕ} (h_ratio : 2+3+4 = 9)
  (h_units_A : ∃ a : ℕ, a = 16)
  (h_stratified_sampling : ∃ B C : ℕ, B = 24 ∧ C = 32)
  : n = 16 + 24 + 32 := by
  sorry

end sample_size_l237_237746


namespace sum_of_squares_l237_237273

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : a * b + b * c + c * a = 131) : 
  a^2 + b^2 + c^2 = 138 := 
sorry

end sum_of_squares_l237_237273


namespace product_of_fractions_l237_237406

-- Define the fractions as ratios.
def fraction1 : ℚ := 2 / 5
def fraction2 : ℚ := 7 / 10

-- State the theorem that proves the product of the fractions is equal to the simplified result.
theorem product_of_fractions : fraction1 * fraction2 = 7 / 25 :=
by
  -- Skip the proof.
  sorry

end product_of_fractions_l237_237406


namespace sum_bases_l237_237528

noncomputable def R1 : ℝ := 12  -- Derived from the verification in the solution
noncomputable def R2 : ℝ := 9   -- Derived from the verification in the solution

theorem sum_bases :
  let F1_R1 := (4 * R1 + 7) / (R1^2 - 1)
  let F2_R1 := (7 * R1 + 4) / (R1^2 - 1)
  let F1_R2 := (1 * R2 + 6) / (R2^2 - 1)
  let F2_R2 := (6 * R2 + 1) / (R2^2 - 1)
  F1_R1 = F1_R2 → F2_R1 = F2_R2 → R1 + R2 = 21 :=
by
  intros
  symmetry
  sorry

end sum_bases_l237_237528


namespace triangle_third_side_l237_237683

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l237_237683


namespace joel_picked_185_non_hot_peppers_l237_237176

def total_peppers_week :=
  [ (7, 10, 13)  -- Sunday: (hot, sweet, mild)
  , (12, 8, 10)  -- Monday
  , (14, 19, 7)  -- Tuesday
  , (12, 5, 23)  -- Wednesday
  , (5, 20, 5)   -- Thursday
  , (18, 15, 12) -- Friday
  , (12, 8, 30)  -- Saturday
  ]

noncomputable def non_hot_peppers_week (peppers : List (Nat × Nat × Nat)) : Nat :=
  peppers.foldr (λ (hm : Nat × Nat × Nat) acc, acc + (hm.2.1 + hm.2.2)) 0

theorem joel_picked_185_non_hot_peppers :
  non_hot_peppers_week total_peppers_week = 185 :=
by
  sorry

end joel_picked_185_non_hot_peppers_l237_237176


namespace triangle_FYH_area_l237_237293

noncomputable def trapezoid_area (EF GH h : ℝ) : ℝ :=
  1 / 2 * h * (EF + GH)

noncomputable def triangle_area (base height : ℝ) : ℝ :=
  1 / 2 * base * height

theorem triangle_FYH_area :
  ∀ (EF GH : ℝ) (trapezoid_area_val triangle_FYH_area_val : ℝ), 
  EF = 24 → GH = 36 → trapezoid_area_val = 360 → 
  trapezoid_area EF GH 12 = 360 → triangle_FYH_area_val = 86.4 :=
by
  intros EF GH trapezoid_area_val triangle_FYH_area_val hEF hGH htarea hheight
  rw [hEF, hGH, htarea, hheight]
  sorry

end triangle_FYH_area_l237_237293


namespace tetrahedron_angle_sum_invariant_l237_237222

theorem tetrahedron_angle_sum_invariant (A B X Y : Point) (S : Sphere)
(h1 : Tetrahedron A X B Y)
(h2 : CircumscribedAbout S (Tetrahedron A X B Y))
: ∀ (X' Y' : Point),
  (A = A) → (B = B) →
  ∠AX'B + ∠X'BY' + ∠BY'A + ∠Y'AX' = 
  ∠AXB + ∠XBY + ∠BYA + ∠YAX :=
begin
  sorry,
end

end tetrahedron_angle_sum_invariant_l237_237222


namespace range_of_a_l237_237900

noncomputable def log_base (a x : ℝ) : ℝ := log x / log a

theorem range_of_a {a x : ℝ} (h1 : x > 1) (h2: x < 100) (h3: a > 0) (h4: a ≠ 1) :
  (log_base a x - (log x)^2 < 4) -> a ∈ (Set.Ioo 0 1) ∪ {b | b > Real.exp (1/4)} :=
sorry

end range_of_a_l237_237900


namespace investmentAmounts_l237_237591

variable (totalInvestment : ℝ) (bonds stocks mutualFunds : ℝ)

-- Given conditions
def conditions := 
  totalInvestment = 210000 ∧
  stocks = 2 * bonds ∧
  mutualFunds = 4 * stocks ∧
  bonds + stocks + mutualFunds = totalInvestment

-- Prove the investments
theorem investmentAmounts (h : conditions totalInvestment bonds stocks mutualFunds) :
  bonds = 19090.91 ∧ stocks = 38181.82 ∧ mutualFunds = 152727.27 :=
sorry

end investmentAmounts_l237_237591


namespace maximum_fly_path_length_in_box_l237_237813

theorem maximum_fly_path_length_in_box
  (length width height : ℝ)
  (h_length : length = 1)
  (h_width : width = 1)
  (h_height : height = 2) :
  ∃ l, l = (Real.sqrt 6 + 2 * Real.sqrt 5 + Real.sqrt 2 + 1) :=
by
  sorry

end maximum_fly_path_length_in_box_l237_237813


namespace inequality_C_false_l237_237855

theorem inequality_C_false (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : (1 / a) ^ (1 / b) ≤ 1 := 
sorry

end inequality_C_false_l237_237855


namespace students_play_long_tennis_l237_237517

theorem students_play_long_tennis (total_students : ℕ) (play_football : ℕ) (play_both : ℕ) 
(play_neither : ℕ) (play_long_tennis : ℕ) :
  total_students = 36 → play_football = 26 → play_both = 17 → play_neither = 7 → 
  play_long_tennis = 20 := by
  intro h_total h_football h_both h_neither
  have h_at_least_one := total_students - play_neither
  have h_both_and_football := play_football - play_both
  have h_long_tennis := h_at_least_one - h_both_and_football - play_both
  rw [h_total, h_football, h_both, h_neither] at *
  have h:= h_long_tennis
  linarith

end students_play_long_tennis_l237_237517


namespace find_a_l237_237418

def f (x : ℝ) : ℝ :=
  if x <= 0 then -x else x^2

theorem find_a (a : ℝ) (h : f a = 2) : a = Real.sqrt 2 ∨ a = -2 :=
sorry

end find_a_l237_237418


namespace compound_interest_rate_l237_237403

theorem compound_interest_rate
  (A P : ℝ) (t n : ℝ)
  (HA : A = 1348.32)
  (HP : P = 1200)
  (Ht : t = 2)
  (Hn : n = 1) :
  ∃ r : ℝ, 0 ≤ r ∧ ((A / P) ^ (1 / (n * t)) - 1) = r ∧ r = 0.06 := 
sorry

end compound_interest_rate_l237_237403


namespace determine_value_of_m_l237_237470

theorem determine_value_of_m (m : ℝ) (x1 x2 : ℝ)
  (h_real_roots : ∃ (a b c : ℝ), a = 1 ∧ b = -(2 * m - 1) ∧ c = m^2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)
  (h_condition : (x1 + 1) * (x2 + 1) = 3) : m = -3 := 
begin
  sorry
end

end determine_value_of_m_l237_237470


namespace fruit_basket_ratio_l237_237362

theorem fruit_basket_ratio :
  ∃ x : ℕ, x > 2 ∧
  (let cost_bananas := 4 * 1,
       cost_apples := 3 * 2,
       cost_strawberries := (24 / 12) * 4,
       cost_avocados := 2 * 3,
       total_cost_other_fruits := cost_bananas + cost_apples + cost_strawberries + cost_avocados,
       total_cost := 28,
       cost_portion_grapes := total_cost - total_cost_other_fruits,
       cost_one_portion_grapes := 2,
       portion_grapes_value := cost_portion_grapes / cost_one_portion_grapes
   in portion_grapes_value = 2) :=
begin
  sorry
end

end fruit_basket_ratio_l237_237362


namespace find_x_l237_237118

-- Define vector types and dot product
def vector2 := (ℕ × ℕ)

def dot_product (v1 v2 : vector2) : ℕ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the vectors
def a : vector2 := (2, -1)
def b (x : ℕ) : vector2 := (3, x)

-- Define the theorem
theorem find_x (x : ℕ) (h : dot_product a (b x) = 3) : x = 3 :=
by
  sorry

end find_x_l237_237118


namespace cube_side_length_l237_237240

theorem cube_side_length (n : ℕ) (h : n^3 - (n-2)^3 = 98) : n = 5 :=
by sorry

end cube_side_length_l237_237240


namespace circle_through_AKL_tangent_BC_l237_237558

variables (A B C D E F M L K : Point)
variables (h1 : is_tangent D (incircle A B C) BC)
variables (h2 : is_tangent E (incircle A B C) AC)
variables (h3 : is_tangent F (incircle A B C) AB)
variables (h4 : M = midpoint E F)
variables (h5 : L ∈ circle_through D M F ∩ segment A B)
variables (h6 : K ∈ circle_through D M E ∩ segment A C)

theorem circle_through_AKL_tangent_BC
  (h1 : is_tangent D (incircle A B C) BC)
  (h2 : is_tangent E (incircle A B C) AC)
  (h3 : is_tangent F (incircle A B C) AB)
  (h4 : M = midpoint E F)
  (h5 : L ∈ circle_through D M F ∩ segment A B)
  (h6 : K ∈ circle_through D M E ∩ segment A C) :
  is_tangent (circle_through A K L) BC :=
sorry

end circle_through_AKL_tangent_BC_l237_237558


namespace complex_modulus_l237_237455

open Complex -- Use the complex number definitions

-- Define the given problem
def givenComplexNumber : ℂ := (1 + 2 * Complex.I) / (2 - Complex.I)

-- State the proof problem
theorem complex_modulus : Complex.abs givenComplexNumber = 1 :=
by
  sorry -- The proof steps go here

end complex_modulus_l237_237455


namespace cos_diff_of_symmetric_sines_l237_237532

theorem cos_diff_of_symmetric_sines (a β : Real) (h1 : Real.sin a = 1 / 3) 
  (h2 : Real.sin β = 1 / 3) (h3 : Real.cos a = -Real.cos β) : 
  Real.cos (a - β) = -7 / 9 := by
  sorry

end cos_diff_of_symmetric_sines_l237_237532


namespace solve_symmetric_cosine_phi_l237_237889

noncomputable def symmetric_cosine_phi : Prop :=
  ∃ (φ : ℝ), (φ ∈ set.Icc 0 real.pi) ∧ (∀ (x : ℝ), 3 * real.cos (x + φ) - 1 = 3 * real.cos (2 * real.pi / 3 - x + φ) - 1) ∧ φ = 2 * real.pi / 3

theorem solve_symmetric_cosine_phi : symmetric_cosine_phi :=
  sorry

end solve_symmetric_cosine_phi_l237_237889


namespace giuseppe_can_cut_rectangles_l237_237860

theorem giuseppe_can_cut_rectangles : 
  let board_length := 22
  let board_width := 15
  let rectangle_length := 3
  let rectangle_width := 5
  (board_length * board_width) / (rectangle_length * rectangle_width) = 22 :=
by
  sorry

end giuseppe_can_cut_rectangles_l237_237860


namespace max_n_and_a_l237_237474

noncomputable def ak (k : ℕ) (hk : k < 4) : ℕ := 
match k with 
| 1 => 40
| 2 => 30
| 3 => 24
| 4 => 20
| _ => 0

theorem max_n_and_a {n : ℕ} (hn : n = 5) : 
  (∀ k ∈ {1, 2, 3, 4}, ak k (by decide) = 120 / (k + 2)) ∧
  ∃ a₅ : ℕ, a₅ = 120 - (ak 1 (by decide) + ak 2 (by decide) + ak 3 (by decide) + ak 4 (by decide)) :=
begin
  split,
  { intros k hk,
    fin_cases k;
    simp [ak, ← nat.cast_add, nat.cast_one, nat.cast_bit0, nat.cast_bit1]; norm_num },
  { use 6,
    simp [ak, ← nat.cast_add, nat.cast_one, nat.cast_bit0, nat.cast_bit1],
    norm_num }
end

end max_n_and_a_l237_237474


namespace correct_options_l237_237077

variable (f : ℝ → ℝ)

axiom functional_eq : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y
axiom half_value : f (1 / 2) = 0

theorem correct_options :
  (f 0 ≠ -1) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ y : ℝ, f ((1 / 2) + y) + f ((1 / 2) - y) = 0) ∧
  (∑ k in Finset.range 2022, f (k + 1) = 0) :=
by
  sorry

end correct_options_l237_237077


namespace perfect_square_conditions_l237_237659

theorem perfect_square_conditions (x y k : ℝ) :
  (∃ a : ℝ, x^2 + k * x * y + 81 * y^2 = a^2) ↔ (k = 18 ∨ k = -18) :=
sorry

end perfect_square_conditions_l237_237659


namespace total_hours_eq_52_l237_237390

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l237_237390


namespace collinear_points_iff_l237_237796

variables {A B C P : Type*}
variables {AP AB AC BP PC BC : ℝ}
variables {α β : ℝ}

theorem collinear_points_iff :
  (0 < α + β ∧ α + β < 180) ∧
  (AP = 0 ∨ AB = 0 ∨ AC = 0 ∨ BP = 0 ∨ PC = 0 ∨ BC = 0 ∨
   (sin (α + β) / AP = sin α / AC + sin β / AB ∧
   AB^2 * PC + AC^2 * BP = AP^2 * BC + BP * PC * BC)) ↔
  collinear B P C :=
sorry

end collinear_points_iff_l237_237796


namespace smallest_k_for_news_sharing_l237_237728

def friends_learn_news (k : ℕ) (days : ℕ) (total_friends : ℕ) (available_days : ℕ) : Prop :=
  ∀ (P : fin total_friends → set (fin days)),
    (∀ i, (P i).card = available_days) →
    ∃ S : finset (fin days), S.card = k ∧ 
    ∀ i j, ∃ d ∈ S, d ∈ P i ∧ d ∈ P j

theorem smallest_k_for_news_sharing :
  friends_learn_news 5 10 45 8 :=
sorry

end smallest_k_for_news_sharing_l237_237728


namespace general_term_a_n_range_of_a_l237_237081

section
  variable (a : ℕ → ℕ)
  variable (S : ℕ → ℕ)

  -- Conditions for the sequence {a_n}
  axiom a1 : a 1 = 1
  axiom a_rec : ∀ n : ℕ, a (n + 1) = 2 * Nat.sqrt (S n) + 1
  axiom Sn : ∀ n : ℕ, S (n + 1) = (Nat.sqrt (S n) + 1) ^ 2

  -- General term formula
  theorem general_term_a_n (n : ℕ) (hn : 1 ≤ n) : a n = 2 * n - 1 := 
  by 
    sorry

  variable (b : ℕ → ℝ)
  variable (T : ℕ → ℝ)

  -- Sequence {b_n} and sum {T_n}
  axiom b_def : ∀ n : ℕ, b n = 4 * n ^ 2 / (a n * a (n + 1))
  axiom T_def : ∀ n : ℕ, T n = ∑ i in (Finset.range n), b i

  -- Inequality condition and range of a
  theorem range_of_a (a_val : ℝ) : (∀ n : ℕ, n > 0 → T n - n * a_val < 0) → a_val > 4 / 3 :=
  by 
    sorry
end

end general_term_a_n_range_of_a_l237_237081


namespace find_constant_x_geom_prog_l237_237050

theorem find_constant_x_geom_prog (x : ℝ) :
  (30 + x) ^ 2 = (10 + x) * (90 + x) → x = 0 :=
by
  -- Proof omitted
  sorry

end find_constant_x_geom_prog_l237_237050


namespace problem_solution_l237_237571

-- Define a function f(n) under the given conditions
def x_k (k : ℕ) : ℕ := k^2

def f (n : ℕ) : ℚ :=
  (∑ k in Finset.range (n + 1), k^2) / n

theorem problem_solution (n : ℕ) (h : 0 < n) :
  f n = (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

end problem_solution_l237_237571


namespace Gilda_marble_percentage_l237_237069

theorem Gilda_marble_percentage (
    (M : ℝ) (hM : M > 0) :
    
    let M1 := 0.70 * M,
    let M2 := M1 - 0.20 * M1,
    let M3 := M2 - 0.15 * M2,
    let M4 := M3 - 0.10 * M3,
    
    M4 / M * 100 = 43 :=
sorry

end Gilda_marble_percentage_l237_237069


namespace tea_sales_l237_237255

theorem tea_sales (L T : ℕ) (h1 : L = 32) (h2 : L = 4 * T + 8) : T = 6 :=
by
  sorry

end tea_sales_l237_237255


namespace sam_gave_plums_l237_237584

variable (initial_plums : ℝ) (total_plums : ℝ) (plums_given : ℝ)

theorem sam_gave_plums (h1 : initial_plums = 7.0) (h2 : total_plums = 10.0) (h3 : total_plums = initial_plums + plums_given) :
  plums_given = 3 := 
by
  sorry

end sam_gave_plums_l237_237584


namespace magnitude_of_complex_l237_237945

theorem magnitude_of_complex :
  ∀ (z : ℂ), (z = 1 + 2 * complex.i + complex.i ^ 3) → complex.abs z = real.sqrt 2 :=
by
  intros z h
  sorry

end magnitude_of_complex_l237_237945


namespace pipes_empty_8_minutes_l237_237651

noncomputable def empty_rate_A := (1/4) / 12
noncomputable def empty_rate_B := (1/3) / 20
noncomputable def empty_rate_C := (1/5) / 30

def combined_rate := empty_rate_A + empty_rate_B + empty_rate_C

theorem pipes_empty_8_minutes :
  combined_rate * 8 = 53 / 150 :=
by
  -- Proof goes here
  sorry

end pipes_empty_8_minutes_l237_237651


namespace prime_impossibility_l237_237858

noncomputable def transformed_sequence (a b c d : ℤ) : ℕ → ℤ × ℤ × ℤ × ℤ
| 0     := (a, b, c, d)
| (n+1) :=
  let (a', b', c', d') := transformed_sequence a b c d n in
  (a' - b', b' - c', c' - d', d' - a')

theorem prime_impossibility (a0 b0 c0 d0 : ℤ) :
  let (a, b, c, d) := transformed_sequence a0 b0 c0 d0 1997 in
  a + b + c + d = 0 →
  ¬ (∀ q1 q2 q3 : ℤ, q1 = |b * c - a * d| → q2 = |a * c - b * d| → q3 = |a * b - c * d| → Prime q1 ∧ Prime q2 ∧ Prime q3) :=
by
  intros a b c d h H
  sorry

end prime_impossibility_l237_237858


namespace sum_remainder_product_remainder_l237_237204

open Nat

-- Define the modulus conditions
variables (x y z : ℕ)
def condition1 : Prop := x % 15 = 11
def condition2 : Prop := y % 15 = 13
def condition3 : Prop := z % 15 = 14

-- Proof statement for the sum remainder
theorem sum_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x + y + z) % 15 = 8 :=
by
  sorry

-- Proof statement for the product remainder
theorem product_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x * y * z) % 15 = 2 :=
by
  sorry

end sum_remainder_product_remainder_l237_237204


namespace heptagon_angle_sum_l237_237749

-- Given: Heptagon A1, A2, A3, A4, A5, A6, A7 is inscribed in a circle
-- The center of the circle lies inside the heptagon

theorem heptagon_angle_sum (A1 A2 A3 A4 A5 A6 A7 : Point)
  (O : Point) (h1 : ∀i ∈ {A1, A2, A3, A4, A5, A6, A7}, dist i O = dist A1 O)
  (h2 : O ∈ heptagon_interior A1 A2 A3 A4 A5 A6 A7) :
  ∠A1 + ∠A3 + ∠A5 < 450 :=
sorry

end heptagon_angle_sum_l237_237749


namespace student_score_statistics_l237_237000

open ProbabilityTheory

-- Define the parameters given in the problem as constants
def n : ℕ := 25
def p : ℝ := 0.6
def q : ℝ := 1 - p
def points_per_correct_answer : ℝ := 4

-- Define the random variable η following a binomial distribution B(n, p)
noncomputable def η : ℕ → ℝ := λ k, binomial_prob n p k

-- Define ξ as the total score
noncomputable def ξ : ℝ := 4 * η

-- Define the expected value and variance of ξ
theorem student_score_statistics :
  E ξ = 60 ∧ var ξ = 96 :=
by
  sorry

end student_score_statistics_l237_237000


namespace mike_washed_cars_l237_237548

theorem mike_washed_cars 
    (total_work_time : ℕ := 4 * 60) 
    (wash_time : ℕ := 10)
    (oil_change_time : ℕ := 15) 
    (tire_change_time : ℕ := 30) 
    (num_oil_changes : ℕ := 6) 
    (num_tire_changes : ℕ := 2) 
    (remaining_time : ℕ := total_work_time - (num_oil_changes * oil_change_time + num_tire_changes * tire_change_time))
    (num_cars_washed : ℕ := remaining_time / wash_time) :
    num_cars_washed = 9 := by
  sorry

end mike_washed_cars_l237_237548


namespace common_difference_arithmetic_sequence_l237_237201

variables (a b d : ℤ)

theorem common_difference_arithmetic_sequence (a_1 a_2 a_4 a_6 : ℤ)
  (h1 : a_1 * a_2 = 35)
  (h2 : 2 * a_4 - a_6 = 7)
  (ha_2 : a_2 = a_1 + d)
  (ha_4 : a_4 = a_1 + 3 * d)
  (ha_6 : a_6 = a_1 + 5 * d) :
  d = 2 :=
by
  sorry

end common_difference_arithmetic_sequence_l237_237201


namespace matrix_multiplication_correct_l237_237811

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [2, 0, -1],
      [0, 3, -2],
      [-2, 3, 2]]

def matrix_B : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [1, -1, 0],
      [2, 0, -4],
      [3, 0, 0]]

def matrix_C : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [-1, -2, 0],
      [0, 0, -12],
      [10, 2, -12]]

theorem matrix_multiplication_correct : matrix_A ⬝ matrix_B = matrix_C :=
  by 
    -- Matrix multiplication setup
    sorry

end matrix_multiplication_correct_l237_237811


namespace problem_proof_l237_237195

-- Assume definitions for lines and planes, and their relationships like parallel and perpendicular exist.

variables (m n : Line) (α β : Plane)

-- Define conditions
def line_is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_is_parallel_to_line (l1 l2 : Line) : Prop := sorry
def planes_are_perpendicular (p1 p2 : Plane) : Prop := sorry

-- Problem statement
theorem problem_proof :
  (line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n α) → 
  (line_is_parallel_to_line m n) ∧
  ((line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n β) ∧ (line_is_perpendicular_to_plane m n) → 
  (planes_are_perpendicular α β)) := 
sorry

end problem_proof_l237_237195


namespace jim_catches_up_to_cara_l237_237725

noncomputable def time_to_catch_up (jim_speed: ℝ) (cara_speed: ℝ) (initial_time: ℝ) (stretch_time: ℝ) : ℝ :=
  let initial_distance_jim := jim_speed * initial_time
  let initial_distance_cara := cara_speed * initial_time
  let added_distance_cara := cara_speed * stretch_time
  let distance_gap := added_distance_cara
  let relative_speed := jim_speed - cara_speed
  distance_gap / relative_speed

theorem jim_catches_up_to_cara :
  time_to_catch_up 6 5 (30/60) (18/60) * 60 = 90 :=
by
  sorry

end jim_catches_up_to_cara_l237_237725


namespace test_of_independence_l237_237984

-- Define the events A and B
variables {Ω : Type} {P : ProbabilisticSpace Ω} {A B : Event P}

-- Define mutually independent events
def mutually_independent (A B : Event P) : Prop :=
  P(A ∩ B) = P(A) * P(B)

-- State the hypothesis of the test of independence
theorem test_of_independence :
  (test_of_independence_hypothesis : ∀ A B : Event P, mutually_independent A B) :=
sorry

end test_of_independence_l237_237984


namespace rain_at_least_one_day_probability_l237_237278

-- Definitions based on given conditions
def P_rain_Friday : ℝ := 0.30
def P_rain_Monday : ℝ := 0.20

-- Events probabilities based on independence
def P_no_rain_Friday := 1 - P_rain_Friday
def P_no_rain_Monday := 1 - P_rain_Monday
def P_no_rain_both := P_no_rain_Friday * P_no_rain_Monday

-- The probability of raining at least one day
def P_rain_at_least_one_day := 1 - P_no_rain_both

-- Expected probability
def expected_probability : ℝ := 0.44

theorem rain_at_least_one_day_probability : 
  P_rain_at_least_one_day = expected_probability := by
  sorry

end rain_at_least_one_day_probability_l237_237278


namespace find_polynomial_q_l237_237054

def polynomial_q (q : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, q(x^3) - q(x^3 - 3) = (q(x))^2 + 18

theorem find_polynomial_q : 
  ∃ q : ℝ → ℝ, polynomial_q q ∧ q = (λ x, 9 * x^3 - 9) :=
begin
  sorry
end

end find_polynomial_q_l237_237054


namespace greatest_third_side_l237_237671

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l237_237671


namespace find_integers_l237_237272

theorem find_integers (x y : ℤ) 
  (h1 : x * y + (x + y) = 95) 
  (h2 : x * y - (x + y) = 59) : 
  (x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11) :=
by
  sorry

end find_integers_l237_237272


namespace larger_angle_at_3_48_is_186_l237_237706

-- Define the time in hours and minutes
def time := (3, 48)

-- Define the movement of the minute hand (degrees per minute)
def minute_hand_movement := 6

-- Define the movement of the hour hand (degrees per minute)
def hour_hand_movement := 0.5

-- Calculate the position of the minute hand at the given time
def minute_position := time.2 * minute_hand_movement

-- Calculate the position of the hour hand at the given time
def hour_position := (time.1 * 60 + time.2) * hour_hand_movement

-- Define the direct angle between the hands
def direct_angle := abs (minute_position - hour_position)

-- Define the larger angle
def larger_angle := 360 - direct_angle

-- Proof statement
theorem larger_angle_at_3_48_is_186 : larger_angle = 186 :=
by
  sorry

end larger_angle_at_3_48_is_186_l237_237706


namespace trignometric_identity_proof_l237_237927

theorem trignometric_identity_proof
  (α β : ℝ)
  (h : (cos α)^6 / (cos (2 * β))^2 + (sin α)^6 / (sin (2 * β))^2 = 1) :
  (sin (2 * β))^6 / (sin α)^2 + (cos (2 * β))^6 / (cos α)^2 = 1 :=
begin
  sorry
end

end trignometric_identity_proof_l237_237927


namespace average_marks_of_failed_candidates_l237_237614

theorem average_marks_of_failed_candidates
  (total_candidates : ℕ)
  (avg_marks_all : ℝ)
  (passed_candidates : ℕ)
  (avg_marks_passed : ℝ)
  (failed_candidates : ℕ := total_candidates - passed_candidates)
  (total_marks_all : ℝ := avg_marks_all * total_candidates)
  (total_marks_passed : ℝ := avg_marks_passed * passed_candidates)
  (total_marks_failed : ℝ := total_marks_all - total_marks_passed) :
  passed_candidates = 100 ∧ total_candidates = 120 ∧ avg_marks_all = 35 ∧ avg_marks_passed = 39 →
  total_marks_failed / failed_candidates = 15 := 
by
  intro h,
  cases h with h0 h1,
  cases h1 with h2 h3,
  cases h3 with h4 h5,
  rw [h0, h2, h4, h5],
  norm_num,
  sorry

end average_marks_of_failed_candidates_l237_237614


namespace lesson_duration_tuesday_l237_237383

theorem lesson_duration_tuesday
  (monday_lessons : ℕ)
  (monday_duration : ℕ)
  (tuesday_lessons : ℕ)
  (wednesday_multiplier : ℕ)
  (total_time : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (H1 : monday_lessons = 6)
  (H2 : monday_duration = 30)
  (H3 : tuesday_lessons = 3)
  (H4 : wednesday_multiplier = 2)
  (H5 : total_time = 12)
  (H6 : monday_hours = monday_lessons * monday_duration / 60)
  (H7 : tuesday_hours = tuesday_lessons * T)
  (H8 : wednesday_hours = wednesday_multiplier * tuesday_hours)
  (H9 : monday_hours + tuesday_hours + wednesday_hours = total_time) :
  T = 1 := by
  sorry

end lesson_duration_tuesday_l237_237383


namespace find_f_of_16_l237_237471

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem find_f_of_16 : (∃ a : ℝ, f 2 a = Real.sqrt 2) → f 16 (1/2) = 4 :=
by
  intro h
  sorry

end find_f_of_16_l237_237471


namespace intersection_M_N_l237_237115

-- Define the sets M and N.
def setM : Set ℝ := {x : ℝ | log 3 x < 2}
def setN : Set ℝ := {x : ℝ | (3 * x + 4) * (x - 3) > 0}

-- State the theorem to find the intersection of M and N.
theorem intersection_M_N : setM ∩ setN = {x : ℝ | 3 < x ∧ x < 9} :=
sorry

end intersection_M_N_l237_237115


namespace partition_of_sum_l237_237271

-- Define the conditions
def is_positive_integer (n : ℕ) : Prop := n > 0
def is_bounded_integer (n : ℕ) : Prop := n ≤ 10
def can_be_partitioned (S : ℕ) (integers : List ℕ) : Prop :=
  ∃ (A B : List ℕ), 
    A.sum ≤ 70 ∧ 
    B.sum ≤ 70 ∧ 
    A ++ B = integers

-- Define the theorem statement
theorem partition_of_sum (S : ℕ) (integers : List ℕ)
  (h1 : ∀ x ∈ integers, is_positive_integer x ∧ is_bounded_integer x)
  (h2 : List.sum integers = S) :
  S ≤ 133 ↔ can_be_partitioned S integers :=
sorry

end partition_of_sum_l237_237271


namespace find_third_root_l237_237414

theorem find_third_root (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3 * b) * 1^2 + (b - 4 * a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (6 - a) = 0)
  : ∃ c : ℚ, c = 7 / 13 :=
sorry

end find_third_root_l237_237414


namespace greatest_possible_third_side_l237_237679

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l237_237679


namespace laura_weekly_mileage_l237_237181

-- Define the core conditions

-- Distance to school per round trip (house <-> school)
def school_trip_distance : ℕ := 20

-- Number of trips to school per week
def school_trips_per_week : ℕ := 7

-- Distance to supermarket: 10 miles farther than school
def extra_distance_to_supermarket : ℕ := 10
def supermarket_trip_distance : ℕ := school_trip_distance + 2 * extra_distance_to_supermarket

-- Number of trips to supermarket per week
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly distance
def total_distance_per_week : ℕ := 
  (school_trips_per_week * school_trip_distance) +
  (supermarket_trips_per_week * supermarket_trip_distance)

-- Theorem to prove the total distance Laura drives per week
theorem laura_weekly_mileage :
  total_distance_per_week = 220 := by
  sorry

end laura_weekly_mileage_l237_237181


namespace increasing_interval_of_y_l237_237824

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x

theorem increasing_interval_of_y :
  ∃ (a b : ℝ), 0 < a ∧ a < e ∧ (∀ x : ℝ, a < x ∧ x < e → y x < y (x + ε)) :=
sorry

end increasing_interval_of_y_l237_237824


namespace problem1_problem2_l237_237481

noncomputable def f (x a : ℝ) : ℝ := x^3 + a * x^2 - 2 * x + 1

theorem problem1 (a : ℝ) (h : deriv (λ x, f x a) 1 = 0) : a = -1 / 2 :=
by
  -- Proof omitted
  sorry

theorem problem2 :
  let a := -1 / 2 in
  let fa := f (·) a in
  (deriv fa > 0) ∣ (-∞, -(2 / 3)) ∪ (1, ∞) ∧ 
  (deriv fa < 0) ∣ (-(2 / 3), 1) ∧
  (fa (-(2 / 3)) = 49 / 27) ∧
  (fa 1 = -1 / 2) :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l237_237481


namespace sqrt_difference_calc_l237_237801

theorem sqrt_difference_calc: 
  sqrt 27 - sqrt (1 / 3) = (8 * sqrt 3) / 3 := 
by 
  sorry

end sqrt_difference_calc_l237_237801


namespace rock_paper_scissors_sixth_game_end_probability_l237_237014

theorem rock_paper_scissors_sixth_game_end_probability :
  let tie_prob := (1 : ℝ) / 3
  let win_loss_prob := (2 : ℝ) / 3
  in (tie_prob ^ 5 * win_loss_prob) = (2 / 729 : ℝ) :=
by
  let tie_prob := (1 : ℝ) / 3
  let win_loss_prob := (2 : ℝ) / 3
  have calc (tie_prob ^ 5 * win_loss_prob) = ((1 / 3 : ℝ) ^ 5 * (2 / 3 : ℝ)) : by rfl
  sorry

end rock_paper_scissors_sixth_game_end_probability_l237_237014


namespace maria_profit_l237_237582

theorem maria_profit (buy4_price : ℝ) (sell3_price : ℝ) (profit_goal : ℝ) (cost_per_disk : buy4_price / 4 = 1.25)
(sell_per_disk: sell3_price / 3 = 1.67)
: profit_goal = 100 → buy4_price = 5 → sell3_price = 5 → 240 := 
by
  sorry

end maria_profit_l237_237582


namespace transform_to_quadratic_l237_237622

-- Define the function that needs to be transformed
def original_function (x : ℝ) : ℝ :=
  (2^x - 2)^2 + (2^(-x) + 2)^2

-- Define the quadratic form
def quadratic_form (t m : ℝ) : ℝ :=
  t^2 - 4 * t + m

-- Define phi function as per the correct answer identified
def phi (x : ℝ) : ℝ :=
  2^x - 2^(-x)

-- State the theorem to be proven
theorem transform_to_quadratic (x : ℝ) :
  ∃ (m : ℝ), original_function x = quadratic_form (phi x) m :=
sorry

end transform_to_quadratic_l237_237622


namespace greatest_possible_third_side_l237_237678

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l237_237678


namespace rainfall_wednesday_correct_l237_237172

def monday_rainfall : ℝ := 0.9
def tuesday_rainfall : ℝ := monday_rainfall - 0.7
def wednesday_rainfall : ℝ := 2 * (monday_rainfall + tuesday_rainfall)

theorem rainfall_wednesday_correct : wednesday_rainfall = 2.2 := by
sorry

end rainfall_wednesday_correct_l237_237172


namespace greatest_integer_third_side_l237_237666

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l237_237666


namespace sum_sequence_formula_l237_237807
open_locale big_operators

noncomputable def sum_sequence (a : ℝ) (n : ℕ) : ℝ :=
  (∑ k in finset.range(n+1), a^k - k)

theorem sum_sequence_formula (a : ℝ) (n : ℕ) (h₀ : a ≠ 0) : 
  sum_sequence a n = 
  if h : a = 1 then (n - n^2) / 2 
  else (a * (1 - a^n) / (1 - a)) - (n * (n + 1) / 2) :=
sorry

end sum_sequence_formula_l237_237807


namespace geometric_sequence_sum_l237_237444

-- Defining the geometric sequence related properties and conditions
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * r) → 
  S 3 = a 0 + a 1 + a 2 →
  S 6 = a 3 + a 4 + a 5 →
  S 12 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 →
  S 3 = 3 →
  S 6 = 6 →
  S 12 = 45 :=
by
  sorry

end geometric_sequence_sum_l237_237444


namespace length_of_diagonal_l237_237341

-- Given the area of the square is 4802 m^2, prove that the length of the diagonal is approximately 98 meters
theorem length_of_diagonal (area : ℝ) (side diagonal : ℝ) (h : area = 4802) :
  side^2 = area → diagonal = real.sqrt (2 * side^2) → diagonal ≈ 98 := by
  sorry

end length_of_diagonal_l237_237341


namespace elise_money_after_expenses_l237_237044

theorem elise_money_after_expenses (initial saved spent_comic_book spent_puzzle : ℕ) :
  initial = 8 → saved = 13 → spent_comic_book = 2 → spent_puzzle = 18 → (initial + saved - spent_comic_book - spent_puzzle) = 1 := by
  intros h_initial h_saved h_spent_comic_book h_spent_puzzle
  rw [h_initial, h_saved, h_spent_comic_book, h_spent_puzzle]
  exact Nat.sub_eq_of_eq_add 21 20 (by norm_num)

end elise_money_after_expenses_l237_237044


namespace area_bounded_by_lines_and_parabola_l237_237815

noncomputable def calculateArea (a : ℝ) : ℝ :=
  let f (x : ℝ) := 3 * (Real.sqrt x)
  2 * (∫ x in 0..(10), f x)

theorem area_bounded_by_lines_and_parabola : calculateArea 5 = 40 * Real.sqrt 10 := 
by sorry

end area_bounded_by_lines_and_parabola_l237_237815


namespace find_science_books_l237_237784

theorem find_science_books
  (S : ℕ)
  (h1 : 2 * 3 + 3 * 2 + 3 * S = 30) :
  S = 6 :=
by
  sorry

end find_science_books_l237_237784


namespace number_of_differences_is_one_l237_237917

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def can_be_written_as_difference_of_two_primes (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 - p2

def set_of_numbers : set ℕ := { n | ∃ k : ℕ, n = 7 + 10 * k }

def count_numbers_that_are_differences_of_two_primes : ℕ :=
  (set_of_numbers.filter can_be_written_as_difference_of_two_primes).card

theorem number_of_differences_is_one :
  count_numbers_that_are_differences_of_two_primes = 1 :=
sorry

end number_of_differences_is_one_l237_237917


namespace q_at_two_l237_237405

-- Define the function q(x)
noncomputable def q : ℝ → ℝ := λ x, 
  if x = 2 then 5 else sorry -- as we only need to show q(2) here

-- Theorem
theorem q_at_two : q 2 = 5 :=
by {
  -- From the graph, the point (2, 5) is given
  have h_point : (q 2, 2) = (5, 2),
  from sorry, -- This would be evident from the graphical data
  -- Extract the value corresponding to x = 2
  exact sorry,
}

end q_at_two_l237_237405


namespace number_of_classes_l237_237267

theorem number_of_classes (total_basketballs classes_basketballs : ℕ) (h1 : total_basketballs = 54) (h2 : classes_basketballs = 7) : total_basketballs / classes_basketballs = 7 := by
  sorry

end number_of_classes_l237_237267


namespace intersection_is_correct_l237_237462

noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def B : Set ℝ := {x | x^2 - 5 * x - 6 < 0}

theorem intersection_is_correct : A ∩ B = {x | -1 < x ∧ x < 2} := 
by { sorry }

end intersection_is_correct_l237_237462


namespace possible_not_coplanar_edges_l237_237521

-- Define the structure of a parallelepiped 
structure Parallelepiped :=
  (edges : Finset (Fin 12))

-- Define the arbitrary line within the plane of one of its faces
structure FaceLine (P : Parallelepiped) :=
  (line : Finset (Fin 4))

-- Define the proof problem statement
theorem possible_not_coplanar_edges (P : Parallelepiped) (line : FaceLine P) : 
  ∃ n ∈ ({4, 6, 7, 8} : Finset Nat), ∃ (E : Finset (Fin 12)), (E.card = n ∧ ∀ e ∈ E, ¬coplanar_with_line line.line e) :=
by
  sorry

end possible_not_coplanar_edges_l237_237521


namespace log_inequality_l237_237090

theorem log_inequality (a b c : ℝ) (h1 : b^2 - a * c < 0) :
  ∀ x y : ℝ, a * (Real.log x)^2 + 2 * b * (Real.log x) * (Real.log y) + c * (Real.log y)^2 = 1 
  → a * 1^2 + 2 * b * 1 * (-1) + c * (-1)^2 = 1 → 
  -1 / Real.sqrt (a * c - b^2) ≤ Real.log (x * y) ∧ Real.log (x * y) ≤ 1 / Real.sqrt (a * c - b^2) :=
by
  sorry

end log_inequality_l237_237090


namespace process_never_stops_l237_237771

theorem process_never_stops (N : ℕ) (h_digits : N.to_digits = list.repeat 1 900)
  (transformation_rule : ∀ N, N > 100 → ∃ A B : ℕ, B < 100 
     ∧ (N = 100 * A + B ∧ N' = 2 * A + 8 * B) → (N' > 100)) :
  ¬ ∃ k : ℕ, iterate (λ x, 2 * (x / 100) + 8 * (x % 100)) k N < 100 :=
by
  sorry

end process_never_stops_l237_237771


namespace length_segment_CD_twice_median_AM_l237_237518

theorem length_segment_CD_twice_median_AM
  (A B C D E M : Point)
  (h1 : AE = AD)
  (h2 : AB = AC)
  (h3 : ∠CAD = ∠AEB + ∠ABE)
  (hM : M is the midpoint of BE) :
  length(CD) = 2 * length(AM) :=
sorry

end length_segment_CD_twice_median_AM_l237_237518


namespace plane_equation_l237_237754

theorem plane_equation :
  ∃ (A B C D : ℤ), (A = 3 ∧ B = -7 ∧ C = -3 ∧ D = 40) ∧
  (∀ (s t : ℝ), ∃ (x y z : ℝ),
    x = 2 + 2*s - 3*t ∧ y = 4 + s ∧ z = 6 - 3*s + t) ∧
  (A > 0) ∧
  Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 :=
by
  use 3, -7, -3, 40
  split
  . exact ⟨rfl, rfl, rfl, rfl⟩
  split
  . intros s t
    use 2 + 2*s - 3*t, 4 + s, 6 - 3*s + t
    exact ⟨rfl, rfl, rfl⟩
  split
  . exact by simp
  . exact by sorry

end plane_equation_l237_237754


namespace brick_wall_l237_237742

theorem brick_wall (x : ℕ) 
  (h1 : x / 9 * 9 = x)
  (h2 : x / 10 * 10 = x)
  (h3 : 5 * (x / 9 + x / 10 - 10) = x) :
  x = 900 := 
sorry

end brick_wall_l237_237742


namespace calculate_postage_l237_237751

def postage_rate (weight : ℝ) : ℝ :=
  if weight <= 1 then 0.25
  else
    let surcharge := if weight > 3 then 0.10 else 0
    let additional_weight := weight - 1
    let additional_charges := (additional_weight.ceil : ℝ) * 0.18
    0.25 + additional_charges + surcharge

theorem calculate_postage :
  postage_rate 5.5 = 1.25 := by
  sorry

end calculate_postage_l237_237751


namespace coloring_possible_l237_237546

-- Definitions of the conditions
def is_vertical_line (n : ℤ) (p : ℤ × ℤ) : Prop := p.1 = n
def is_horizontal_line (m : ℤ) (p : ℤ × ℤ) : Prop := p.2 = m
def is_diagonal_line (p : ℤ × ℤ → Prop) : Prop := ∃ c : ℤ, p = (λ q, q.1 - q.2 = c) ∨ p = (λ q, q.1 + q.2 = c)

def color (p : ℤ × ℤ) : Prop :=
  | p.2 >= p.1^2 ∨ p.1 >= p.2^2

-- Problem statement
theorem coloring_possible :
  (∀ n : ℤ, ∃ fin_set_white_ver : set (ℤ × ℤ), (∀ p : ℤ × ℤ, is_vertical_line n p → p ∉ fin_set_white_ver → ¬ color p)) ∧
  (∀ m : ℤ, ∃ fin_set_white_hor : set (ℤ × ℤ), (∀ p : ℤ × ℤ, is_horizontal_line m p → p ∉ fin_set_white_hor → ¬ color p)) ∧
  (∀ diag : (ℤ × ℤ → Prop), is_diagonal_line diag → ∃ fin_set_black_diag : set (ℤ × ℤ), (∀ p : ℤ × ℤ, diag p → p ∈ fin_set_black_diag → color p)) :=
sorry

end coloring_possible_l237_237546


namespace sugar_more_than_salt_l237_237213

variables (flour_used flour_recipe sugar_recipe salt_recipe : ℕ)

-- Given conditions
def conditions (flour_used flour_recipe sugar_recipe salt_recipe : ℕ) :=
  flour_recipe = 6 ∧
  sugar_recipe = 8 ∧
  salt_recipe = 7 ∧
  flour_used = 5

-- Goal to prove
theorem sugar_more_than_salt (flour_used flour_recipe sugar_recipe salt_recipe : ℕ) :
  conditions flour_used flour_recipe sugar_recipe salt_recipe → (sugar_recipe - salt_recipe = 1) :=
by
  intro h
  cases' h with h1 rest
  cases' rest with h2 rest
  cases' rest with h3 h4
  rw [h2, h3]
  sorry

end sugar_more_than_salt_l237_237213


namespace even_product_probability_l237_237296

theorem even_product_probability :
  let spinner1 := {2, 5, 7, 11}
  let spinner2 := {3, 4, 6, 8, 10}
  let total_outcomes := 4 * 5
  let odd_spinner1 := {5, 7, 11}
  let odd_spinner2 := {3}
  let odd_product_outcomes := odd_spinner1.card * odd_spinner2.card
  let even_product_outcomes := total_outcomes - odd_product_outcomes
  let probability_even := even_product_outcomes / total_outcomes
  probability_even = 17/20 :=
by
  sorry

end even_product_probability_l237_237296


namespace annie_extracurricular_hours_l237_237396

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end annie_extracurricular_hours_l237_237396


namespace quadratic_function_range_l237_237936

theorem quadratic_function_range (f : ℝ → ℝ) (a : ℝ)
  (h_quad : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_sym : ∀ x, f (2 + x) = f (2 - x))
  (h_cond : f a ≤ f 0 ∧ f 0 < f 1) :
  a ≤ 0 ∨ a ≥ 4 :=
sorry

end quadratic_function_range_l237_237936


namespace sum_of_divisors_of_45_l237_237313

theorem sum_of_divisors_of_45 : 
  (∑ d in (finset.filter (λ x : ℕ, 45 % x = 0) (finset.range (45 + 1))), d) = 78 :=
by sorry

end sum_of_divisors_of_45_l237_237313


namespace perfect_squares_divisible_by_9_count_l237_237923

-- Define the range of numbers from 1 to 20
def numbers : List ℕ := List.range 20

-- Define a predicate to check divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define the list of numbers that are multiples of 3 in the given range
def multiples_of_3 : List ℕ := numbers.filter divisible_by_3

-- Define the list of perfect squares of these multiples of 3
def perfect_squares_of_multiples_of_3 : List ℕ := multiples_of_3.map (λ n => n * n)

-- Define the predicate to check if a number is divisible by 9
def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

-- Define the list of perfect squares that are divisible by 9
def perfect_squares_divisible_by_9 : List ℕ := perfect_squares_of_multiples_of_3.filter divisible_by_9

-- Prove that the length of this list is 6
theorem perfect_squares_divisible_by_9_count : perfect_squares_divisible_by_9.length = 6 := by
  sorry

end perfect_squares_divisible_by_9_count_l237_237923


namespace minimum_reciprocal_sum_l237_237488

theorem minimum_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : ∃ x y, (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧ (2 * a * x - b * y + 2 = 0))
  (h2 : ∃ d, d = 0 ∧ 2 * sqrt (4 - d^2) = 4) :
  ∃ a b, (a + b = 1) ∧ (a > 0) ∧ (b > 0) ∧ (1/a + 1/b = 4) :=
by
  sorry

end minimum_reciprocal_sum_l237_237488


namespace segments_of_diameter_l237_237962

theorem segments_of_diameter 
  (O C D A B K : ℝ) 
  (radius : ℝ) 
  (CH : ℝ) 
  (ON CN : ℝ)
  (H1 : CD = 2 * radius)
  (H2 : AB = 2 * radius)
  (H3 : radius = 6)
  (H4 : CD ⊥ AB)
  (H5 : CH = 10)
  (H6 : K ∈ AB) :
  let a := abs CN in
  let AN := radius - a in
  let NB := radius + a in
    (AN, NB) = (6 - real.sqrt 11, 6 + real.sqrt 11) :=
by {
    sorry
}

end segments_of_diameter_l237_237962


namespace dining_bill_before_tip_l237_237748

theorem dining_bill_before_tip : 
  ∃ B : ℝ, B = 139 ∧ 
  let total_paid := 7 * 21.842857142857145 in
  total_paid = 1.10 * B :=
sorry

end dining_bill_before_tip_l237_237748


namespace max_area_triangle_ABC_l237_237161

-- Definitions of the given conditions
def PA := 3
def PB := 4
def PC := 5
def BC := 6

-- The problem statement to prove
theorem max_area_triangle_ABC : ∀ (h : ℝ), h = PA → 1/2 * BC * h = 9 :=
by
  intros h h_eq
  rw [h_eq, PA, BC]
  linarith

end max_area_triangle_ABC_l237_237161


namespace monotonicity_of_f_minimum_value_of_g_l237_237109

-- Define the function f(x) and g(x)
def f (x m : ℝ) : ℝ := x^2 - 2*x + m*log x
def g (x : ℝ) : ℝ := (x - 3/4) * exp x

-- The first goal is to determine the monotonicity of f(x) based on m
theorem monotonicity_of_f (m : ℝ) :
  ∃ I1 I2 I3 : Set ℝ, 
    (∀ x ∈ I1, MonotoneStrict.Increasing (f x m)) ∧ 
    (∀ x ∈ I2, MonotoneStrict.Decreasing (f x m)) ∧ 
    (∀ x ∈ I3, MonotoneStrict.Increasing (f x m)) := 
sorry

-- The second goal is to find the minimum value of g(x1 - x2) given the conditions
theorem minimum_value_of_g (m x1 x2 : ℝ) (h1 : 0 < m) (h2 : m < 1/2) (h3 :  x1 < x2) (h4 : f' x1 = 0) (h5 : f' x2 = 0) :
  g (x1 - x2) = -exp (-1/4) :=
sorry

end monotonicity_of_f_minimum_value_of_g_l237_237109


namespace complex_number_solution_l237_237507

theorem complex_number_solution (z : ℂ) (h : 2 * z + conj z = 3 - 2 * complex.I) : z = 1 - 2 * complex.I :=
by sorry

end complex_number_solution_l237_237507


namespace remy_pieces_of_furniture_l237_237224

def is_correct_charge (total_paid : ℕ) (overcharged_amount : ℕ) : ℕ :=
  total_paid - overcharged_amount

def number_of_pieces (correct_charge : ℕ) (cost_per_piece : ℕ) : ℕ :=
  correct_charge / cost_per_piece

theorem remy_pieces_of_furniture :
  ∀ (total_paid overcharged_amount cost_per_piece correct_charge : ℕ),
  total_paid = 20700 → 
  overcharged_amount = 600 → 
  cost_per_piece = 134 →
  correct_charge = is_correct_charge total_paid overcharged_amount →
  number_of_pieces correct_charge cost_per_piece = 150 :=
by
  intros total_paid overcharged_amount cost_per_piece correct_charge h1 h2 h3 h4
  rw [← h1, ← h2, ← h3, ← h4]
  simp [is_correct_charge, number_of_pieces]
  sorry

end remy_pieces_of_furniture_l237_237224


namespace angle_correspondences_l237_237185

variables {A B C D M N X : Type*}
variables [rhombus A B C D] [equilateral_triangle A B D] [equilateral_triangle B C D]
variables (M_on_BC : M ∈ line (B, C)) (N_on_CD : N ∈ line (C, D))
variables (intersection_X : X = intersection (diagonal A C) (diagonal B D))
variables (angle_MAN_30 : ∠(M, A, N) = 30)

theorem angle_correspondences
  (h1 : rhombus A B C D)
  (h2 : equilateral_triangle A B D)
  (h3 : equilateral_triangle B C D)
  (h4 : M ∈ line (B, C))
  (h5 : N ∈ line (C, D))
  (h6 : X = intersection (diagonal A C) (diagonal B D))
  (h7 : ∠(M, A, N) = 30) :
  ∠(X, M, N) = ∠(D, A, M) ∧ ∠(X, N, M) = ∠(B, A, N) :=
sorry

end angle_correspondences_l237_237185


namespace element_set_M_a1_b2_periodic_sequence_a_neg_b_neg_min_elements_set_M_a_nonneg_b_nonneg_l237_237459

noncomputable def a_sequence (a b : ℝ) : ℕ → ℝ
| 0       := a
| 1       := b
| (n + 2) := |a_sequence (n + 1)| - a_sequence n

def set_M (a b : ℝ) : set ℝ :=
  {x | x ∈ {a_sequence a b n | n : ℕ}}

theorem element_set_M_a1_b2 : set_M 1 2 = {1, 2, -1, 0} :=
sorry

theorem periodic_sequence_a_neg_b_neg (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  ∃ p : ℕ, p ≥ 1 ∧ ∀ n : ℕ, a_sequence a b (n + p) = a_sequence a b n :=
sorry

theorem min_elements_set_M_a_nonneg_b_nonneg (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b ≠ 0) :
  ∃ n : ℕ, n ≥ 4 ∧ ∀ S : set ℝ, S ⊆ set_M a b → S.finite → S.card = n :=
sorry

end element_set_M_a1_b2_periodic_sequence_a_neg_b_neg_min_elements_set_M_a_nonneg_b_nonneg_l237_237459


namespace discount_savings_difference_l237_237002

def cover_price : ℝ := 30
def discount_amount : ℝ := 5
def discount_percentage : ℝ := 0.25

theorem discount_savings_difference :
  let price_after_discount := cover_price - discount_amount
  let price_after_percentage_first := cover_price * (1 - discount_percentage)
  let new_price_after_percentage := price_after_discount * (1 - discount_percentage)
  let new_price_after_discount := price_after_percentage_first - discount_amount
  (new_price_after_percentage - new_price_after_discount) * 100 = 125 :=
by
  sorry

end discount_savings_difference_l237_237002


namespace sum_divisors_45_l237_237316

theorem sum_divisors_45 : ∑ d in (45 : ℕ).divisors, d = 78 :=
by
  sorry

end sum_divisors_45_l237_237316


namespace eliminate_denominators_eq_l237_237242

theorem eliminate_denominators_eq :
  ∀ (x : ℝ), 1 - (x + 3) / 6 = x / 2 → 6 - x - 3 = 3 * x :=
by
  intro x
  intro h
  -- Place proof steps here.
  sorry

end eliminate_denominators_eq_l237_237242


namespace power_sum_geq_pow_product_l237_237929

theorem power_sum_geq_pow_product (α β : ℝ) (n : ℕ) (a : Fin n → ℝ) (h_α : 0 < α) (h_β : 0 < β) (h_a : ∀ i, 0 < a i) :
  (∑ i, (a i)^(α + β)) ≥ (∑ i in Finset.range n, (a i)^(α) * (a ((i + 1) % n))^(β) :=
by
  sorry

end power_sum_geq_pow_product_l237_237929


namespace part1_max_traffic_flow_part2_traffic_flow_exceeds_10_l237_237781

def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

theorem part1_max_traffic_flow :
  ∃ (v : ℝ), v > 0 ∧ traffic_flow v = 920 / 83 :=
sorry

theorem part2_traffic_flow_exceeds_10 :
  ∃ (v : ℝ), 25 < v ∧ v < 64 ∧ traffic_flow v > 10 :=
sorry

end part1_max_traffic_flow_part2_traffic_flow_exceeds_10_l237_237781


namespace cookies_per_child_is_22_l237_237856

def total_cookies (num_packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  num_packages * cookies_per_package

def total_children (num_friends : ℕ) : ℕ :=
  num_friends + 1

def cookies_per_child (total_cookies : ℕ) (total_children : ℕ) : ℕ :=
  total_cookies / total_children

theorem cookies_per_child_is_22 :
  total_cookies 5 36 / total_children 7 = 22 := 
by
  sorry

end cookies_per_child_is_22_l237_237856


namespace sin_inequality_solution_set_l237_237160

theorem sin_inequality_solution_set :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * π ∧ sin x < -sqrt 3 / 2} = {x : ℝ | 4 * π / 3 < x ∧ x < 5 * π / 3} :=
by sorry

end sin_inequality_solution_set_l237_237160


namespace incircle_tangency_l237_237530

theorem incircle_tangency 
  (A B C D : Point)
  (hAC : Line A C)
  (hBD : Line B D)
  (h_tangent_ABC_ADC : tangent_to_each_other (incircle (Triangle A B C)) (incircle (Triangle A D C))) :
  tangent_to_each_other (incircle (Triangle B A D)) (incircle (Triangle B C D)) :=
sorry

end incircle_tangency_l237_237530


namespace max_area_region_T_l237_237523

theorem max_area_region_T (r1 r2 r3 r4 : ℝ) (radii : {r // r = [2, 4, 6, 8]})
  (tangent_point : ℝ) (line_ell : ℝ → Prop) (B : ℝ) (h_tangent : ∀ r ∈ radii, line_ell B) :
  ∃ (max_area : ℝ), max_area = 80 * Real.pi :=
by
  use 80 * Real.pi
  sorry

end max_area_region_T_l237_237523


namespace rectangle_RS_minimum_l237_237560

theorem rectangle_RS_minimum (ABCD : Type) [rect : rectangle ABCD] 
  (A B C D M R S : point ABCD)
  (hAB : distance A B = 24)
  (hBC : distance B C = 10)
  (hM_on_BC : on_line_segment B C M)
  (hR_perp_AC : perpendicular R (line_segment A C))
  (hS_perp_AD : perpendicular S (line_segment A D)) :
  ∃ (minimum_RS : ℝ), minimum_RS = 9.10 :=
sorry

end rectangle_RS_minimum_l237_237560


namespace f_triple_application_l237_237573

def f (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 6 else x - 15

theorem f_triple_application : f (f (f 20)) = 4 := sorry

end f_triple_application_l237_237573


namespace total_missing_keys_l237_237995

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end total_missing_keys_l237_237995


namespace angelina_journey_equation_l237_237793

theorem angelina_journey_equation (t : ℝ) :
    4 = t + 15/60 + (4 - 15/60 - t) →
    60 * t + 90 * (15/4 - t) = 255 :=
    by
    sorry

end angelina_journey_equation_l237_237793


namespace abs_diff_eq_five_l237_237563

theorem abs_diff_eq_five (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 :=
by
  sorry

end abs_diff_eq_five_l237_237563


namespace total_legs_of_all_animals_l237_237212

def num_kangaroos : ℕ := 23
def num_goats : ℕ := 3 * num_kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_legs_of_all_animals : num_kangaroos * legs_of_kangaroo + num_goats * legs_of_goat = 322 :=
by 
  sorry

end total_legs_of_all_animals_l237_237212


namespace hands_in_class_not_including_peters_l237_237653

def total_students : ℕ := 11
def hands_per_student : ℕ := 2
def peter_hands : ℕ := 2

theorem hands_in_class_not_including_peters :  (total_students * hands_per_student) - peter_hands = 20 :=
by
  sorry

end hands_in_class_not_including_peters_l237_237653


namespace distinct_prime_factors_N_l237_237095

theorem distinct_prime_factors_N (N : ℕ) (h : log 2 (log 3 (log 5 (log 7 N))) = 11) : 
  (Nat.factors N).toFinset.card = 1 := 
sorry

end distinct_prime_factors_N_l237_237095


namespace base9_subtraction_multiple_of_seven_l237_237245

theorem base9_subtraction_multiple_of_seven (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 9) 
(h2 : (3 * 9^6 + 1 * 9^5 + 5 * 9^4 + 4 * 9^3 + 6 * 9^2 + 7 * 9^1 + 2 * 9^0) - b % 7 = 0) : b = 0 :=
sorry

end base9_subtraction_multiple_of_seven_l237_237245


namespace dihedral_angle_range_of_prism_l237_237147

theorem dihedral_angle_range_of_prism (n : ℕ) (hn : n ≥ 3) : 
  dihedral_angle_range n = (n-2) * Real.pi / n, Real.pi :=
sorry

end dihedral_angle_range_of_prism_l237_237147


namespace greatest_possible_third_side_l237_237681

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l237_237681


namespace number_of_8_digit_integers_l237_237122

theorem number_of_8_digit_integers : 
  ∃ n, n = 90000000 ∧ 
    (∀ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
     d1 ≠ 0 → 0 ≤ d1 ∧ d1 ≤ 9 ∧ 
     0 ≤ d2 ∧ d2 ≤ 9 ∧ 
     0 ≤ d3 ∧ d3 ≤ 9 ∧ 
     0 ≤ d4 ∧ d4 ≤ 9 ∧ 
     0 ≤ d5 ∧ d5 ≤ 9 ∧ 
     0 ≤ d6 ∧ d6 ≤ 9 ∧ 
     0 ≤ d7 ∧ d7 ≤ 9 ∧ 
     0 ≤ d8 ∧ d8 ≤ 9 →
     ∀ count, count = (if d1 ≠ 0 then 9 * 10^7 else 0)) :=
sorry

end number_of_8_digit_integers_l237_237122


namespace part1_part2_l237_237867

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

theorem part1 (h : (derivative (f a) 0) = -1) : a = 2 ∧ ∃ x_min : ℝ, x_min = Real.log 2 ∧ is_minimum (f 2) x_min (2 - Real.log 4) := by
  sorry

theorem part2 (x : ℝ) (h : x > 0) : x^2 < Real.exp x := by
  sorry

end part1_part2_l237_237867


namespace number_of_true_statements_l237_237062

theorem number_of_true_statements (a b c : ℝ) :
  ((∀ a b c, (a = b ↔ a * c = b * c)) = false) →
  ((∀ a, irrational (a + 5) ↔ irrational a) = true) →
  ((∀ a b, (a > b → a^2 > b^2)) = false) →
  ((∀ a, (a < 5 → a < 3)) = true) →
  (count_true_statements = 2) :=
by
  -- We assert four separate statements about real numbers.
  have h1 : (∀ a b c, (a = b ↔ a * c = b * c)) = false,
  -- We assert that
  have h2 : (∀ a, irrational (a + 5) ↔ irrational a) = true,
  -- We assert the statement
  have h3 : (∀ a b, (a > b → a^2 > b^2)) = false,
  -- We assert the final statement 
  have h4 : (∀ a, (a < 5 → a < 3)) = true,
  -- Number of true statements is exactly 2.
  unfold count_true_statements,
  sorry

end number_of_true_statements_l237_237062


namespace min_omega_l237_237897

noncomputable theory
open Real

-- Conditions
def f (ω φ x : ℝ) := sin (ω * x + φ)

def condition_1 (ω φ : ℝ) := ∃ k : ℤ, ω * (π / 2) + φ = k * π

def condition_2 (ω φ : ℝ) := f ω φ (π / 4) = 1 / 2

-- Main theorem
theorem min_omega (ω φ : ℝ) (h1 : condition_1 ω φ) (h2 : condition_2 ω φ) (h : ω > 0) : ω = 2 / 3 :=
sorry

end min_omega_l237_237897


namespace slope_y_intercept_sum_half_area_line_l237_237294

constant VertexX : (ℝ × ℝ)
constant VertexY : (ℝ × ℝ)
constant VertexZ : (ℝ × ℝ)

axiom VertexX_def : VertexX = (-1, 7)
axiom VertexY_def : VertexY = (3, -1)
axiom VertexZ_def : VertexZ = (9, -1)

theorem slope_y_intercept_sum_half_area_line : 
  let m := (VertexX.1 + VertexZ.1) / 2
  let n := (VertexX.2 + VertexZ.2) / 2
  let line_slope := (n - VertexY.2) / (m - VertexY.1) in
  let intercept := VertexY.2 - line_slope * VertexY.1 in
  line_slope + intercept = -9 :=
begin
  sorry
end

end slope_y_intercept_sum_half_area_line_l237_237294


namespace percentage_less_than_l237_237939

theorem percentage_less_than (x y : ℝ) (h : y = 1.80 * x) : (x / y) * 100 = 100 - 44.44 :=
by
  sorry

end percentage_less_than_l237_237939


namespace range_of_a_div_b_l237_237863

variable (a b : ℝ)
variable (h1 : a > b) (h2 : b > 0)

def e1 := Real.sqrt (a^2 - b^2) / a
def e2 := Real.sqrt (a^2 - b^2) / b
def t := a / b

theorem range_of_a_div_b (h3 : e1 * e2 < 1) : Real.sqrt 2 < t ∧ t < (1 + Real.sqrt 5) / 2 :=
by
  sorry

end range_of_a_div_b_l237_237863


namespace graph_of_f_not_necessarily_straight_l237_237064

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 2 - x else 1/x

theorem graph_of_f_not_necessarily_straight :
  (∀ x : ℝ, rational x → rational (f x)) ∧ 
  (∀ x : ℝ, irrational x → irrational (f x)) ∧ 
  (∀ x : ℝ, differentiableAt ℝ f x) ∧ 
  (continuous f) →
  ¬ ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
begin
  sorry
end

end graph_of_f_not_necessarily_straight_l237_237064


namespace f_bound_l237_237094

noncomputable def f : ℕ+ → ℝ := sorry

axiom f_1 : f 1 = 3 / 2
axiom f_ineq (x y : ℕ+) : f (x + y) ≥ (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_bound (x : ℕ+) : f x ≥ 1 / 4 * x * (x + 1) * (2 * x + 1) := sorry

end f_bound_l237_237094


namespace cone_cylinder_volume_ratio_l237_237744

theorem cone_cylinder_volume_ratio (r h_cylinder : ℝ) (h_pos : r > 0) (h_cylinder_pos : h_cylinder > 0) :
  let h_cone := (1/3) * h_cylinder in
  let V_cylinder := π * r^2 * h_cylinder in
  let V_cone := (1/3) * π * r^2 * h_cone in
  V_cone / V_cylinder = 1 / 9 :=
by
  let h_cone := (1/3) * h_cylinder
  let V_cylinder := π * r^2 * h_cylinder
  let V_cone := (1/3) * π * r^2 * h_cone
  have h_cone_eq : h_cone = (1/3) * h_cylinder := rfl
  have V_cone_def : V_cone = (1/9) * π * r^2 * h_cylinder := by 
    rw [h_cone_eq]
    ring
  have V_ratio := V_cone / V_cylinder
  rw [←V_cone_def]
  rw [div_mul_cancel]
  norm_num
  field_simp
  ring
  sorry

end cone_cylinder_volume_ratio_l237_237744


namespace value_of_x_sq_plus_y_sq_is_5_l237_237073

-- Define the conditions
def conditions (x : ℝ) (y : ℝ) (h1 : 0 < y) : Prop :=
  let A := {x^2 + x + 1, -x, -x - 1}
  let B := {-y, -y/2, y + 1}
  A = B

-- The theorem to prove
theorem value_of_x_sq_plus_y_sq_is_5 (x y : ℝ) (h1 : 0 < y) (h2 : conditions x y h1) : x^2 + y^2 = 5 :=
sorry

end value_of_x_sq_plus_y_sq_is_5_l237_237073


namespace annual_interest_rate_is_5_03_percent_l237_237305

/-- 
Let P (principal) = 15000 (Rs. 15000)
Let A (amount after t years) = 21500 (P + compound interest)
Let t (time in years) = 7
Let n (number of times interest is compounded per year) = 1
Prove that the annual interest rate r is approximately 0.0503 (or 5.03%).
-/
theorem annual_interest_rate_is_5_03_percent
  (P : ℝ)
  (A : ℝ)
  (t : ℕ)
  (n : ℕ)
  (r : ℝ) :
  P = 15000 → 
  A = 21500 → 
  t = 7 → 
  n = 1 →
  (A = P * (1 + r / n)^(n * t)) →
  r ≈ 0.0503 :=
by 
  intros 
  sorry

end annual_interest_rate_is_5_03_percent_l237_237305


namespace sqrt_difference_calc_l237_237802

theorem sqrt_difference_calc: 
  sqrt 27 - sqrt (1 / 3) = (8 * sqrt 3) / 3 := 
by 
  sorry

end sqrt_difference_calc_l237_237802


namespace misfortune_seconds_in_a_day_l237_237628

theorem misfortune_seconds_in_a_day (minutes_in_day : ℕ) (seconds_in_hour : ℕ) (b : ℕ)
  (h_minutes_in_day : minutes_in_day = 77)
  (h_seconds_in_hour : seconds_in_hour = 91)
  (h_b : b = 7)
  (h_a : 77 = 11 * b)
  (h_c : 91 = b * 13) : 
  minutes_in_day * 11 * 13 = 1001 :=
by
  rw [h_minutes_in_day, h_seconds_in_hour, h_b, h_a, h_c]
  exact rfl

end misfortune_seconds_in_a_day_l237_237628


namespace gymnastics_team_l237_237017

def number_of_rows (n m k : ℕ) : Prop :=
  n = k * (2 * m + k - 1) / 2

def members_in_first_row (n m k : ℕ) : Prop :=
  number_of_rows n m k ∧ 16 < k

theorem gymnastics_team : ∃ m k : ℕ, members_in_first_row 1000 m k ∧ k = 25 ∧ m = 28 :=
by
  sorry

end gymnastics_team_l237_237017


namespace sequence_positive_integers_l237_237342

-- Definitions for the sequence
noncomputable def a_seq : ℕ → ℝ
| 0     := c
| (n+1) := c * a_seq n + real.sqrt ((c^2 - 1) * (a_seq n^2 - 1))

-- The theorem statement
theorem sequence_positive_integers (c : ℕ) (h : c > 0) :
  ∀ n : ℕ, 0 < a_seq n ∧ ∃ k : ℕ, a_seq n = k := 
begin
  sorry
end

end sequence_positive_integers_l237_237342


namespace fraction_value_l237_237324

theorem fraction_value : (5 * 7 : ℝ) / 10 = 3.5 := by
  sorry

end fraction_value_l237_237324


namespace arrangement_same_side_of_C_l237_237652

-- Define the number of arrangements of 6 people
noncomputable def arrangements (n : ℕ) : ℕ :=
  nat.factorial n

-- Define the number of arrangements where A and B are on the same side of C
theorem arrangement_same_side_of_C :
  arrangements 6 * 2 / 3 = 480 := by
  sorry

end arrangement_same_side_of_C_l237_237652


namespace vulgar_fraction_denom_is_50_l237_237745

-- Define the decimal number and corresponding vulgar fraction properties
def vulgar_fraction_equivalent (d: ℚ) (n numerator: ℚ) (denom: ℚ) : Prop :=
  d = n / denom ∧ numerator = 16

-- State the theorem for the given problem
theorem vulgar_fraction_denom_is_50 :
  ∃ denom, vulgar_fraction_equivalent 0.32 16 denom :=
begin
  use 50,
  split,
  { norm_num, },
  { refl, }
end

end vulgar_fraction_denom_is_50_l237_237745


namespace swimmers_meetings_in_10_minutes_l237_237700

theorem swimmers_meetings_in_10_minutes :
  let pool_length := 100
  let swimmer1_speed := 4
  let swimmer2_speed := 3
  let total_time := 600
  ∃ (meetings : ℕ), meetings = 12 := 
begin
  sorry,
end

end swimmers_meetings_in_10_minutes_l237_237700


namespace largest_possible_number_of_cookies_without_ingredients_l237_237175

theorem largest_possible_number_of_cookies_without_ingredients :
  let total_cookies := 60
  let peanuts_cookies := total_cookies / 3
  let chocolate_cookies := total_cookies / 4
  let almond_cookies := total_cookies / 5
  let raisin_cookies := total_cookies / 9
  let total_ingredient_cookies := peanuts_cookies + chocolate_cookies + almond_cookies + raisin_cookies
  let cookies_with_no_ingredients := total_cookies - total_ingredient_cookies
  cookies_with_no_ingredients = 6 := by
  let total_cookies := 60
  let peanuts_cookies := total_cookies / 3
  let chocolate_cookies := total_cookies / 4
  lt almond_cookies := total_cookies / 5
  let raisin_cookies := (total_cookies + 8) / 9
  let total_ingredient_cookies := peanuts_cookies + chocolate_cookies + almond_cookies + raisin_cookies
  let cookies_with_no_ingredients := total_cookies - total_ingredient_cookies
  calc
    cookies_with_no_ingredients 
      = 60 - (20 + 15 + 12 + 7) : by norm_num
   ... = 6 : by norm_num


end largest_possible_number_of_cookies_without_ingredients_l237_237175


namespace rhombus_side_length_l237_237761

theorem rhombus_side_length (d1 d2 : ℝ) (h_ratio : d1 / d2 = 1 / 2) (h_d1 : d1 = 4) : 
  (let s := d2 / 2) in 2 * real.sqrt (s^2 + (d1 / 2)^2) = 2 * real.sqrt 5 :=
by
  sorry

end rhombus_side_length_l237_237761


namespace isosceles_trapezoid_minimal_x_squared_l237_237189

theorem isosceles_trapezoid_minimal_x_squared :
  ∃ (x : ℝ), 
    (∃ (AB CD : ℝ), AB = 100 ∧ CD = 25) ∧ 
    (∃ (AD BC : ℝ), AD = x ∧ BC = x) ∧
    (∃ (M : ℝ), M ∈ set.Icc 0 100 ∧
      metric.sphere M (AD / 2) ∪ 
      metric.sphere M (BC / 2) ⊆
      segment ℝ (line[AD]) ∧
      metric.sphere M (AD / 2) ∪ 
      metric.sphere M (BC / 2) ⊆
      segment ℝ (line[BC])) → 
  x^2 = 1875 :=
by
  sorry

end isosceles_trapezoid_minimal_x_squared_l237_237189


namespace kite_to_rectangle_l237_237559

-- Definitions of the key geometric terms and properties
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

def circle_circumscribing_triangle (T : Triangle) : Point := sorry

structure Kite :=
(A : Point)
(B : Point)
(C : Point)
(D : Point)
(hAB_AD : A ≠ B ∧ ‖A - B‖ = ‖A - D‖)
(hBC_CD : B ≠ C ∧ ‖B - C‖ = ‖C - D‖)
(intersects_at_E : ∃ E : Point, line_through A C ∩ line_through B D = {E})

noncomputable def kite_diagonal_intersection_center (K : Kite) : Point := sorry

noncomputable def circumcenter_of_triangle (T : Triangle) : Point :=
  circle_circumscribing_triangle T

noncomputable def circumscribed_rectangle (K : Kite) : Prop :=
  let E := kite_diagonal_intersection_center K,
      P := circumcenter_of_triangle ⟨K.A, K.B, E⟩,
      Q := circumcenter_of_triangle ⟨K.B, K.C, E⟩,
      R := circumcenter_of_triangle ⟨K.C, K.D, E⟩,
      S := circumcenter_of_triangle ⟨K.D, K.A, E⟩ in
  is_rectangle ⟨P, Q, R, S⟩

theorem kite_to_rectangle (K : Kite) : circumscribed_rectangle K :=
sorry

end kite_to_rectangle_l237_237559


namespace sum_divisors_45_l237_237315

theorem sum_divisors_45 : ∑ d in (45 : ℕ).divisors, d = 78 :=
by
  sorry

end sum_divisors_45_l237_237315


namespace kenny_mow_lawns_l237_237556

theorem kenny_mow_lawns :
  ∃ L : ℕ, 
    (15 * L = 225 + 300) ∧ 
    ∀ Vnum Bnum : ℕ, 
      (Vnum = 5 ∧ Bnum = 60 → 
      (45 * Vnum + 5 * Bnum = 525)) :=
begin
  use 35,
  split,
  { norm_num, },
  { intros _ _ h,
    cases h with hVnum hBnum,
    rw [show 45 * 5 = 225, by norm_num, show 5 * 60 = 300, by norm_num, hVnum, hBnum],
    norm_num, },
end

end kenny_mow_lawns_l237_237556


namespace antenna_tower_height_l237_237300

theorem antenna_tower_height :
  ∃ x : ℝ, (∀ α β γ : ℝ, α + β + γ = π / 2 → tan α = x / 100 → tan β = x / 200 
    → tan γ = x / 300 → x = 100) :=
begin
  use 100,
  intros α β γ h_sum h_tan_α h_tan_β h_tan_γ,
  -- Proof skipped
  sorry
end

end antenna_tower_height_l237_237300


namespace cube_volume_l237_237644

theorem cube_volume (s : ℝ) (h : 12 * s = 96) : s^3 = 512 :=
by
  sorry

end cube_volume_l237_237644


namespace lisa_rem_quiz_max_lower_than_a_l237_237401

noncomputable def lisa_goal_met (total_quizzes completed_scores goal_percentage : ℕ) (completed_a_scores : ℕ) :
  ℕ :=
  let remaining_quizzes := total_quizzes - (completed_scores)
  let required_total_a_scores := goal_percentage * total_quizzes / 100
  let remaining_a_needed := required_total_a_scores - completed_a_scores
  remaining_quizzes - remaining_a_needed

theorem lisa_rem_quiz_max_lower_than_a (total_quizzes completed_scores goal_percentage : ℕ) (completed_a_scores : ℕ) :
  total_quizzes = 50 ∧ completed_scores = 30 ∧ goal_percentage = 80 ∧ completed_a_scores = 22 →
  lisa_goal_met total_quizzes completed_scores goal_percentage completed_a_scores = 2 :=
by intros h; 
   cases h with h_total rest;
   cases rest with h_completed_scores rest;
   cases rest with h_goal_percentage h_completed_a_scores;
   rw [h_total, h_completed_scores, h_goal_percentage, h_completed_a_scores];
   dsimp [lisa_goal_met];
   sorry

end lisa_rem_quiz_max_lower_than_a_l237_237401


namespace find_general_term_l237_237540

-- Define the sequence and the sum of the first n terms condition
def Sn (n : ℕ) (a : ℕ → ℕ) := (n + 2) / 3 * a n

theorem find_general_term (a : ℕ → ℕ) :
  (∀ n, Sn n a = (n + 2) / 3 * a n) →
  (∀ n, a 1 = 1) →
  (∀ n ≥ 2, (n - 1) * a n = (n + 1) * a (n - 1)) →
  (∀ n, a n = n * (n + 1) / 2) :=
begin
  intros h_sn h_a1 h_rec,
  sorry
end

end find_general_term_l237_237540


namespace sqrt_eq_sqrt_modulus_squares_l237_237417

theorem sqrt_eq_sqrt_modulus_squares (a b : ℂ) : 
  (Real.sqrt ((a ^ 2) + (b ^ 2)).re = Real.sqrt (abs a ^ 2 + abs b ^ 2)) :=
sorry

end sqrt_eq_sqrt_modulus_squares_l237_237417


namespace strictly_increasing_interval_l237_237826

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem strictly_increasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = Real.exp 1) ∧ ∀ x : ℝ, a < x ∧ x < b → f' x > 0 :=
by
  let f' := λ x, (1 - Real.log x) / (x^2)
  sorry

end strictly_increasing_interval_l237_237826


namespace magnitude_of_z_l237_237949

noncomputable def z : ℂ := 1 + 2 * complex.i + complex.i ^ 3

theorem magnitude_of_z : complex.abs z = real.sqrt 2 := 
by
  -- this 'sorry' is a placeholder for the actual proof
  sorry

end magnitude_of_z_l237_237949


namespace eight_chair_problem_solution_l237_237043

def eight_chair_arrangement (n : ℕ) : Prop :=
  ∃ (perm : Finset ℕ), perm.card = 8 ∧ (∀ (i : Fin n), 
  let seats := Finset.univ.filter (λ i, i < 8) in 
  let adj (a b : ℕ) : Prop := abs a - b = 1 ∨ abs b - a = 1 in 
  i ∉ seats ∧ ∀ j ∈ seats, ¬adj i j)

theorem eight_chair_problem_solution : ∃ (num_ways : ℕ), (num_ways = 32 ∨ num_ways = 34 ∨ num_ways = 36 ∨ num_ways = 38 ∨ num_ways = 40) ∧ eight_chair_arrangement num_ways :=
sorry

end eight_chair_problem_solution_l237_237043


namespace time_after_delay_l237_237615

theorem time_after_delay (current_time : ℕ) (initial_delay : ℕ) (additional_minutes : ℕ) : ℕ :=
  let total_minutes := initial_delay + additional_minutes,
      total_hours := total_minutes / 60,
      remaining_minutes := total_minutes % 60,
      hours_in_day := total_hours % 24,
      final_time := (current_time + hours_in_day) % 24 * 60 + remaining_minutes 
  in final_time

example : time_after_delay (3 * 60) 30 2500 = 1290 := by sorry

end time_after_delay_l237_237615


namespace average_speed_comparison_l237_237410

-- Defining the average speeds given the conditions

def average_speed_car_A (D u v w : ℝ) : ℝ :=
  D / (D / (4 * u) + D / (2 * v) + D / (4 * w))

def average_speed_car_B (t u v w : ℝ) : ℝ :=
  (u * t / 3 + v * t / 3 + w * t / 3) / t

-- The average speeds according to the formulas obtained in the solution
noncomputable def x (u v w : ℝ) : ℝ := (4 * u * v * w) / (u * (v + w) + 2 * v * w)
noncomputable def y (u v w : ℝ) : ℝ := (u + v + w) / 3

-- Statement that they cannot be compared without specific values
theorem average_speed_comparison (u v w : ℝ) :
  ∀ x y, x = (4 * u * v * w) / (u * (v + w) + 2 * v * w) → y = (u + v + w) / 3 → 
  (u * (v + w) + 2 * v * w ≠ 0 ∧ u + v + w ≠ 0 ∧ (u ≠ 0 ∨ v ≠ 0 ∨ w ≠ 0)) → 
  (x = y ↔ false) := 
  by intros x y hx hy hconds 
     sorry 

end average_speed_comparison_l237_237410


namespace find_S_2018_l237_237531

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)

axiom arithmetic_sequence : ∀ n : ℕ, a_n (n + 1) = a_n n + d
axiom initial_term : a_n 1 = -2018
axiom sum_formula : ∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1) / 2) * d
axiom given_condition : S_n 2016 / 2016 - S_n 10 / 10 = 2006

theorem find_S_2018 : S_n 2018 = -2018 :=
by
  sorry

end find_S_2018_l237_237531


namespace math_problem_l237_237469

noncomputable def circle_eqn : Prop := ∀ (x y : ℝ), x^2 + (y - 1)^2 = 9

noncomputable def chord_length : Prop :=
  let line_eqn := λ x y : ℝ, 12 * x - 5 * y - 8 = 0
  in distance_from_center_to_line (0, 1) line_eqn = 1 ∧
     2 * real.sqrt(3^2 - 1) = 4 * real.sqrt 2

noncomputable def sum_of_inverses : Prop :=
  let y1 y2 : ℝ := sorry
  in (1 / y1) + (1 / y2) = -1 / 4

noncomputable def line_l_eqn : Prop :=
  let Q := (1, 2)
  let A := (sorry, sorry)
  let B := (sorry, sorry)
  in dist Q A = real.sqrt 22 ∧
     dist Q B = real.sqrt 22 ∧
     equation_of_line_through_origin_with_slope_1 A B = true

theorem math_problem : chord_length ∧ sum_of_inverses ∧ line_l_eqn :=
by
  constructor; sorry

end math_problem_l237_237469


namespace triangle_division_l237_237388

theorem triangle_division (n : ℕ) (h : n = 1536) : 
  (n^2) % 3 = 0 :=
by
  rw h
  norm_num
  sorry

end triangle_division_l237_237388


namespace bird_families_difference_l237_237332

theorem bird_families_difference {initial_families flown_away : ℕ} (h1 : initial_families = 87) (h2 : flown_away = 7) :
  (initial_families - flown_away) - flown_away = 73 := by
sorry

end bird_families_difference_l237_237332


namespace necessary_but_not_sufficient_l237_237732

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x - y > -1) → (x^3 + x > x^2 * y + y) → 
  ∃ z : ℝ, z - y > -1 ∧ ¬ (z^3 + z > z^2 * y + y) :=
sorry

end necessary_but_not_sufficient_l237_237732


namespace find_pairs_l237_237053

noncomputable def x (a b : ℝ) : ℝ := b^2 - (a - 1)/2
noncomputable def y (a b : ℝ) : ℝ := a^2 + (b + 1)/2
def valid_pair (a b : ℝ) : Prop := max (x a b) (y a b) ≤ 7 / 16

theorem find_pairs : valid_pair (1/4) (-1/4) :=
  sorry

end find_pairs_l237_237053


namespace fraction_value_l237_237323

theorem fraction_value : (5 * 7 : ℝ) / 10 = 3.5 := by
  sorry

end fraction_value_l237_237323


namespace find_sum_of_reciprocals_l237_237070

variable (a b : ℝ)

theorem find_sum_of_reciprocals (h₁ : 2 ^ a = 10) (h₂ : 5 ^ b = 10) : 
  (1 / a + 1 / b) = 1 := 
  sorry

end find_sum_of_reciprocals_l237_237070


namespace part1_part2_l237_237343

-- Part 1
theorem part1 (x : ℝ) (h : 25 * x ^ 2 - 36 = 0) : x = 6 / 5 ∨ x = - (6 / 5) :=
by
  sorry

-- Part 2
theorem part2 (x a : ℝ) (hx : x + 2 = real.sqrt a) (hy : 3 * x - 10 = real.sqrt a) :
  x = 2 ∧ a = 16 :=
by
  sorry

end part1_part2_l237_237343


namespace sum_of_divisors_of_45_l237_237321

theorem sum_of_divisors_of_45 :
  let divisors := [1, 3, 9, 5, 15, 45] in
  list.sum divisors = 78 :=
by
  -- this is where the proof would go
  sorry


end sum_of_divisors_of_45_l237_237321


namespace problem1_problem2_l237_237902

def A (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < a + 2 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Problem (I)
theorem problem1 (a : ℝ) (h_a : a = 1) : (A a) ∪ B = { x | -1 < x ∧ x < 3 } :=
by unfold A; rw h_a; ext; simp; sorry

-- Problem (II)
theorem problem2 (a : ℝ) (h : (A a) ∩ B = ∅) : a ≤ -3 ∨ a ≥ 3 :=
by unfold A at h; simp at h; sorry

end problem1_problem2_l237_237902


namespace agrey_fish_count_difference_l237_237998

theorem agrey_fish_count_difference :
  ∀ (leo A: ℕ), leo = 40 → leo + A = 100 → A - leo = 20 :=
by
  intros leo A
  assume h1 : leo = 40
  assume h2 : leo + A = 100
  sorry

end agrey_fish_count_difference_l237_237998


namespace find_a_l237_237128

noncomputable def f (x a : ℝ) : ℝ := x^2 + real.log x - a * x

-- Assuming derivative calculation is correct
noncomputable def f_prime (x a : ℝ) : ℝ := 2 * x + 1 / x - a

-- The theorem to prove
theorem find_a (a : ℝ) :
  (f_prime 1 a = 2) ↔ (a = 1) :=
by
  simp [f_prime, f]
  sorry

end find_a_l237_237128


namespace tree_height_when_boy_is_36_l237_237001

namespace TreeGrowth

-- Definitions of initial heights and growth rates.
def initial_tree_height : ℕ := 16
def initial_boy_height : ℕ := 24
def target_boy_height : ℕ := 36
def growth_rate_tree : ℕ := 2  -- Tree grows twice as fast as the boy.

-- Definition that calculates the tree's height when the boy is 36 inches tall.
def final_tree_height (initial_tree_height initial_boy_height target_boy_height growth_rate_tree : ℕ) : ℕ :=
  let boy_growth := target_boy_height - initial_boy_height in
  let tree_growth := boy_growth * growth_rate_tree in
  initial_tree_height + tree_growth

-- Statement to prove the final tree height is 40 inches given the conditions.
theorem tree_height_when_boy_is_36 :
  final_tree_height initial_tree_height initial_boy_height target_boy_height growth_rate_tree = 40 := by
  sorry

end TreeGrowth

end tree_height_when_boy_is_36_l237_237001


namespace box_volume_l237_237791

theorem box_volume (x y : ℝ) (hx : 0 < x ∧ x < 6) (hy : 0 < y ∧ y < 8) :
  (16 - 2 * x) * (12 - 2 * y) * y = 192 * y - 32 * y^2 - 24 * x * y + 4 * x * y^2 :=
by
  sorry

end box_volume_l237_237791


namespace ellipse_cartesian_point_A_cartesian_coordinates_area_of_triangle_APQ_l237_237726

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1/2 + (Real.sqrt 2)/2 * t, 1/2 - (Real.sqrt 2)/2 * t)

noncomputable def ellipse_parametric (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sin α)

def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem ellipse_cartesian (x y : ℝ) (h : ∃ α, x = 2 * Real.cos α ∧ y = Real.sin α) :
  x^2 / 4 + y^2 = 1 := sorry

theorem point_A_cartesian_coordinates :
  polar_to_cartesian 2 (Real.pi / 3) = (1 : ℝ, Real.sqrt 3) := sorry

theorem area_of_triangle_APQ :
  let A := (1, Real.sqrt 3)
  let P := (0, 1)
  let Q := (8/5, -3/5)
  (1/2) * Real.abs (A.1 * (P.2 - Q.2) + P.1 * (Q.2 - A.2) + Q.1 * (A.2 - P.2)) = (4 * Real.sqrt 3) / 5 := sorry

end ellipse_cartesian_point_A_cartesian_coordinates_area_of_triangle_APQ_l237_237726


namespace count_two_digit_numbers_satisfying_conditions_l237_237035

def is_valid_pair (a b : ℕ) : Prop :=
  (a + b = 10) ∧ (a + b) % 3 = 0

def count_valid_pairs : ℕ :=
  (List.filter (λ (p : ℕ × ℕ), is_valid_pair p.1 p.2) [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]).length

theorem count_two_digit_numbers_satisfying_conditions : count_valid_pairs = 3 :=
by
  -- Proof will be added later
  sorry

end count_two_digit_numbers_satisfying_conditions_l237_237035


namespace marble_count_l237_237365

-- Definitions from conditions
variable (M P : ℕ)
def condition1 : Prop := M = 26 * P
def condition2 : Prop := M = 28 * (P - 1)

-- Theorem to be proved
theorem marble_count (h1 : condition1 M P) (h2 : condition2 M P) : M = 364 := 
by
  sorry

end marble_count_l237_237365


namespace odd_power_preserves_order_l237_237882

theorem odd_power_preserves_order {n : ℤ} (h1 : n > 0) (h2 : n % 2 = 1) :
  ∀ (a b : ℝ), a > b → a^n > b^n :=
by
  sorry

end odd_power_preserves_order_l237_237882


namespace solution_set_l237_237326

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set (x : ℝ) (h₁ : ∀ x : ℝ, f' x < 0) (h₂ : f 3 = 0) :
  (x - 1) * f (x + 1) > 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end solution_set_l237_237326


namespace blue_red_ratio_is_eleven_over_fourteen_l237_237057

-- Define the radii of the five concentric circles
def radii := [2, 4, 6, 8, 10]

-- Function to calculate the area of a circle
def circle_area (r : ℝ) : ℝ := real.pi * r^2

-- Define the areas of the five concentric circles
def areas := radii.map circle_area

-- Define the regions (rings) between the circles
def blue_areas := [areas.head!, areas.nth! 2 - areas.nth! 1, areas.nth! 4 - areas.nth! 3]
def red_areas := [areas.nth! 1 - areas.head!, areas.nth! 3 - areas.nth! 2]

-- Sum the areas for blue and red regions
def total_blue_area := (4 * real.pi) + (12 * real.pi) + (28 * real.pi)
def total_red_area := (20 * real.pi) + (36 * real.pi)

-- Define the ratio of blue area to red area
def blue_to_red_ratio := total_blue_area / total_red_area

-- Proof statement
theorem blue_red_ratio_is_eleven_over_fourteen : blue_to_red_ratio = 11 / 14 := by
  sorry

end blue_red_ratio_is_eleven_over_fourteen_l237_237057


namespace missing_keys_total_l237_237991

-- Definitions for the problem conditions

def num_consonants : ℕ := 21
def num_vowels : ℕ := 5
def missing_consonants_fraction : ℚ := 1 / 7
def missing_vowels : ℕ := 2

-- Statement to prove the total number of missing keys

theorem missing_keys_total :
  let missing_consonants := num_consonants * missing_consonants_fraction in
  let total_missing_keys := missing_consonants + missing_vowels in
  total_missing_keys = 5 :=
by {
  -- Placeholder proof
  sorry
}

end missing_keys_total_l237_237991


namespace triangle_side_length_b_l237_237166

variable (A : ℝ) (b : ℝ)

theorem triangle_side_length_b (hC : ∠C = 4 * ∠A) (ha : a = 20) (hc : c = 40) :
  b = 20 * (16 * (9 * sqrt 3 / 16) - 20 * (3 * sqrt 3 / 4) + 5 * sqrt 3) :=
sorry

end triangle_side_length_b_l237_237166


namespace point_on_xaxis_y_coord_zero_l237_237473

theorem point_on_xaxis_y_coord_zero (m : ℝ) (h : (3, m).snd = 0) : m = 0 :=
by 
  -- proof goes here
  sorry

end point_on_xaxis_y_coord_zero_l237_237473


namespace eve_age_l237_237778

variable (E : ℕ)

theorem eve_age (h1 : ∀ (a : ℕ), a = 9 → (E + 1) = 3 * (9 - 4)) : E = 14 := 
by
  have h2 : 9 - 4 = 5 := by norm_num
  have h3 : 3 * 5 = 15 := by norm_num
  have h4 : (E + 1) = 15 := h1 9 rfl
  linarith

end eve_age_l237_237778


namespace exists_periodic_decimal_without_forbidden_sequence_l237_237579

-- Definitions based on the conditions
def infinite_decimal_fraction := ℕ → ℕ  -- A function from natural numbers to digits (0-9)

def forbidden_sequence (n : ℕ) := Fin n → ℕ  -- A finite sequence with length n

-- Main theorem statement
theorem exists_periodic_decimal_without_forbidden_sequence (f : Set (Σ n, forbidden_sequence n)) :
  (∃ (a : infinite_decimal_fraction), ∀ s ∈ f, ∀ n, (¬ s ∈ (set.range (λ i, (λ x, a (x + i)) '' set.univ)))) →
  ∃ (a : infinite_decimal_fraction), (∃ N, ∀ i j, i ≤ N → j ≤ N → (λ x, a (x+i)) = (λ x, a (x+j))) ∧ 
  ∀ s ∈ f, ∀ n, (¬ s ∈ (set.range (λ i, (λ x, a (x + i)) '' set.univ))) :=
begin
  sorry
end

end exists_periodic_decimal_without_forbidden_sequence_l237_237579


namespace complex_pow_sum_l237_237504

theorem complex_pow_sum (z : ℂ) (h : z + z⁻¹ = Real.sqrt 2) : z^2012 + (z^(-2012)) = -2 := by
  sorry

end complex_pow_sum_l237_237504


namespace determine_c16_l237_237079

noncomputable def polynomial_h (z : ℂ) (c : ℕ → ℕ) : ℂ :=
  ∏ i in range 1 17, (1 - z^(i))^(c i)

theorem determine_c16 (c : ℕ → ℕ) (h_eq : polynomial_h z c ≡ 1 - 4 * z [MOD z^17])
  : c 16 = 8 := sorry

end determine_c16_l237_237079


namespace least_x_for_divisibility_l237_237304

theorem least_x_for_divisibility :
  ∃ x : ℕ, (1894 * x) % 3 = 0 ∧ ∀ y : ℕ, (1894 * y) % 3 = 0 → x ≤ y :=
begin
  use 2,
  split,
  { norm_num, },
  { intros y hy,
    sorry
  }
end

end least_x_for_divisibility_l237_237304


namespace math_proof_problem_l237_237188

open EuclideanGeometry

variables {α : Type*} [LinearOrder α] [LinearOrderₓ α] [ConditionallyCompleteLinearOrder α]
  [Archimedean α] [FloorOrder α] [Preorder α] 

def is_cyclic {α : Type*} [OrderedRing α] (A B S T U : Point α) : Prop :=
  collinear A B T U ∧ collinear A B S U

def parallel_lines {α : Type*} [LinearOrder α] (S T B C : Point α) : Prop :=
  ∃ S' B' C', collinear S T S' ∧ collinear B C B' ∧ collinear S' B' ∧ collinear T C'

noncomputable def proofPoints : Prop :=
∃ (A B C U T S : Point ℝ),
  (AC > AB) ∧ 
  is_circumcenter A B C U ∧ 
  is_tangent A T ∧ 
  is_tangent B T ∧
  is_perpendicular_bisector S B C ∧
  is_cyclic A B S T U ∧
  parallel_lines S T B C

theorem math_proof_problem : proofPoints :=
sorry

end math_proof_problem_l237_237188


namespace binomial_expansion_identity_l237_237928

theorem binomial_expansion_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) (h : (2 * x + real.sqrt 3) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) :
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 :=
by
  sorry

end binomial_expansion_identity_l237_237928


namespace card_pairs_divisible_by_5_l237_237587

def cards := list.range 1 101 -- Define the 100 cards sequence

def a_n (n : ℕ) := 3 * n -- Define the sequence function

theorem card_pairs_divisible_by_5 :
  (∑ i in finset.choose_unordered 100 2, 
    if (a_n i.1 + a_n i.2) % 5 = 0 then 1 else 0) = 990 := 
by sorry -- Proof to be provided

end card_pairs_divisible_by_5_l237_237587


namespace total_adjusted_claims_is_174_max_claims_below_limit_l237_237809

-- Definitions based on conditions
def jan_claims := 25
def john_claims := (1.3 * jan_claims).to_int <-- Rounding down, since the problem does this
def missy_claims := (john_claims + 20).to_int
def average_claims := ((jan_claims + john_claims + missy_claims) / 3).to_int
def nancy_claims := (0.9 * average_claims).to_int
def peter_claims := ((2 / 3) * (nancy_claims + jan_claims)).to_int
def missy_adjusted_claims := (missy_claims - 5).to_int
def total_adjusted_claims := jan_claims + john_claims + missy_adjusted_claims + nancy_claims + peter_claims
def max_claim_limit := 430

-- Propositions to be proved
theorem total_adjusted_claims_is_174 : total_adjusted_claims = 174 := by
  sorry

theorem max_claims_below_limit : total_adjusted_claims ≤ max_claim_limit := by
  sorry

end total_adjusted_claims_is_174_max_claims_below_limit_l237_237809


namespace angle_between_vectors_is_60_degrees_l237_237117

variable (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition1 : ‖a‖ = 2 := sorry
def condition2 : ‖b‖ = 1 := sorry
def condition3 : (a - b) ⬝ b = 0 := sorry

-- Prove the angle between vectors a and b is 60 degrees
theorem angle_between_vectors_is_60_degrees :
  real.angle a b = real.pi / 3 :=
by
  rw [real.angle, real.pi_div_three]
  sorry

end angle_between_vectors_is_60_degrees_l237_237117


namespace similar_triangle_area_l237_237790

noncomputable def area_of_larger_similar_triangle (a b c : ℝ) (ratio : ℝ) (short_side_larger : ℝ) : ℝ :=
  let s := (short_side_larger + b * ratio + c * ratio) / 2 in
  Real.sqrt (s * (s - short_side_larger) * (s - b * ratio) * (s - c * ratio))

theorem similar_triangle_area :
  let a := 13
  let b := 13
  let c := 10
  let short_side_larger := 25
  let ratio := short_side_larger / c
  area_of_larger_similar_triangle a b c ratio short_side_larger = 375 :=
by
  sorry

end similar_triangle_area_l237_237790


namespace greatest_integer_floor_l237_237705

noncomputable def power_of_4 (x : ℕ) : ℕ := 4^x
noncomputable def power_of_3 (y : ℕ) : ℕ := 3^y

open Nat

theorem greatest_integer_floor :
  floor ((power_of_4 50 + power_of_3 50) / (power_of_4 47 + power_of_3 47)) = 64 := 
begin
  sorry
end

end greatest_integer_floor_l237_237705


namespace volume_of_regular_tetrahedron_length_6_l237_237252

noncomputable def regular_tetrahedron_volume (s : ℝ) : ℝ :=
  (sqrt 2 / 12) * s^3

theorem volume_of_regular_tetrahedron_length_6 :
  regular_tetrahedron_volume 6 = 12 * sqrt 2 :=
by
  sorry

end volume_of_regular_tetrahedron_length_6_l237_237252


namespace product_of_digits_in_base8_of_7098_l237_237707

def base8_representation_digits (n : ℕ) : list ℕ :=
  -- function that generates the list of digits in base 8 representation
  sorry

def product_of_list (lst : list ℕ) : ℕ :=
  -- function that computes the product of elements in a list
  lst.foldl (*) 1

theorem product_of_digits_in_base8_of_7098 :
  product_of_list (base8_representation_digits 7098) = 210 :=
by
  -- Apply the conversion and product calculation, specific implementation is omitted.
  sorry

end product_of_digits_in_base8_of_7098_l237_237707


namespace functional_expression_y_x_maximize_profit_price_reduction_and_profit_l237_237257

-- Define the conditions
variable (C_selling C_cost : ℝ := 80) (C_costComponent : ℝ := 30) (initialSales : ℝ := 600) 
variable (dec_price : ℝ := 2) (inc_sales : ℝ := 30)
variable (decrease x : ℝ)

-- Define and prove part 1: Functional expression between y and x
theorem functional_expression_y_x : (decrease : ℝ) → (15 * decrease + initialSales : ℝ) = (inc_sales / dec_price * decrease + initialSales) :=
by sorry

-- Define the function for weekly profit
def weekly_profit (x : ℝ) : ℝ := 
  let selling_price := C_selling - x
  let cost_price := C_costComponent
  let sales_volume := 15 * x + initialSales
  (selling_price - cost_price) * sales_volume

-- Prove the condition for maximizing weekly sales profit
theorem maximize_profit_price_reduction_and_profit : 
  (∀ x : ℤ, x % 2 = 0 → weekly_profit x ≤ 30360) ∧
  weekly_profit 4 = 30360 ∧ 
  weekly_profit 6 = 30360 :=
by sorry

end functional_expression_y_x_maximize_profit_price_reduction_and_profit_l237_237257


namespace decimal_to_binary_l237_237029

-- Prove that the binary representation of 53 is 110101 using "divide by k and take the remainder" method.
theorem decimal_to_binary (n : ℕ) (h : n = 53) : binary_repr n = "110101" := by
  sorry

end decimal_to_binary_l237_237029


namespace total_area_inside_circles_but_outside_triangle_l237_237381

def area_inside_circles_but_outside_triangle (a b c r : ℝ) : ℝ :=
  let triangle_area := Real.sqrt (let s := (a + b + c) / 2 
                                  in s * (s - a) * (s - b) * (s - c))
  let circle_area := 3 * (Real.pi * r^2)
  let sector_area := 1.5 * (Real.pi * r^2)
  circle_area - sector_area

theorem total_area_inside_circles_but_outside_triangle :
  area_inside_circles_but_outside_triangle 16 18 21 6 = 54 * Real.pi :=
by
  sorry

end total_area_inside_circles_but_outside_triangle_l237_237381


namespace cut_and_reassemble_circle_to_hexagon_l237_237547

-- Define a regular hexagon inscribed in a circle
structure RegularHexagon (r : ℝ) :=
(vertices : Fin 6 → ℝ × ℝ)
(center : ℝ × ℝ)
(radius_eq : ∀ v, dist center (vertices v) = r)
(edges_eq : ∀ v₁ v₂, dist (vertices v₁) (vertices (v₁ + 1 % 6)) = dist (vertices v₂) (vertices (v₂ + 1 % 6)))
(angles_eq : ∀ v₁ v₂, angle (vertices v₁ - center) (vertices (v₁ + 1 % 6) - center) = angle (vertices v₂ - center) (vertices (v₂ + 1 % 6) - center))

-- Define a circular segment
structure CircularSegment (O : ℝ × ℝ) (r : ℝ) :=
(arc_mid : ℝ × ℝ)
(radius_eq : dist O arc_mid = r)

-- Define the problem statement
theorem cut_and_reassemble_circle_to_hexagon (r : ℝ) :
  ∃ parts : Fin 3 → CircularSegment (0, 0) r, 
    (∀ i, (0, 0) ∈ parts i) ∧ 
    (∃ hexagon : RegularHexagon r, True) :=
by
  sorry

end cut_and_reassemble_circle_to_hexagon_l237_237547


namespace Georgie_paths_l237_237519

theorem Georgie_paths (n : ℕ) (h : n = 8) : ∃ p : ℕ, p = 8 * 7 * 6 ∧ p = 336 :=
  by
  exists 8 * 7 * 6
  split
  { exact rfl }
  { norm_num }
  sorry

end Georgie_paths_l237_237519


namespace factor_problem_l237_237501

theorem factor_problem (x y m : ℝ) (h : (1 - 2 * x + y) ∣ (4 * x * y - 4 * x^2 - y^2 - m)) :
  m = -1 :=
by
  sorry

end factor_problem_l237_237501


namespace cylinder_inscribed_in_sphere_volume_diff_l237_237375

theorem cylinder_inscribed_in_sphere_volume_diff :
  let r_sphere := 7
  let r_cylinder := 4
  let h := real.sqrt (132:ℚ)
  let V_sphere := (4 / 3) * real.pi * (r_sphere ^ 3)
  let V_cylinder := real.pi * (r_cylinder ^ 2) * h
  (V_sphere - V_cylinder) = ((1372:ℚ) / 3 - 16 * real.sqrt (132:ℚ)) * real.pi :=
by
  -- Definitions
  let r_sphere := 7
  let r_cylinder := 4
  let h := real.sqrt (132:ℚ)
  let V_sphere := (4 / 3) * real.pi * (r_sphere ^ 3)
  let V_cylinder := real.pi * (r_cylinder ^ 2) * h
  -- Assertion
  calc 
    V_sphere - V_cylinder
        = ((1372:ℚ) / 3 * real.pi) - (16 * real.pi * real.sqrt (132:ℚ))
        = ((1372:ℚ) / 3 - 16 * real.sqrt (132:ℚ)) * real.pi : sorry

end cylinder_inscribed_in_sphere_volume_diff_l237_237375


namespace neighbor_oranges_weight_l237_237607

theorem neighbor_oranges_weight (N : ℕ) : 
  let first_week := 10 + N in
  let second_week := 2 * 10 in
  let third_week := 2 * 10 in
  let total := 75 in
  first_week + second_week + third_week = total → N = 25 :=
by
  intros
  let first_week := 10 + N
  let second_week := 2 * 10
  let third_week := 2 * 10
  let total := 75
  have h_total : first_week + second_week + third_week = total := sorry
  have h_N : N = 25 := sorry
  exact h_N

end neighbor_oranges_weight_l237_237607


namespace third_side_triangle_max_l237_237691

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l237_237691


namespace correct_calculation_is_D_l237_237328

theorem correct_calculation_is_D 
  (a b x : ℝ) :
  ¬ (5 * a + 2 * b = 7 * a * b) ∧
  ¬ (x ^ 2 - 3 * x ^ 2 = -2) ∧
  ¬ (7 * a - b + (7 * a + b) = 0) ∧
  (4 * a - (-7 * a) = 11 * a) :=
by 
  sorry

end correct_calculation_is_D_l237_237328


namespace area_BEIH_l237_237334

def Point := (ℝ × ℝ)

def B : Point := (0, 0)
def A : Point := (0, 3)
def D : Point := (3, 3)
def C : Point := (3, 0)
def E : Point := (0, 1.5)
def F : Point := (1.5, 0)

def line_eq (p1 p2 : Point) : ℝ → ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  let m := (y2 - y1) / (x2 - x1)
  fun x => m * x + (y1 - m * x1)

def I : Point :=
  let af := line_eq A F
  let de := line_eq D E
  let x := (3 - 1.5) / (0.5 - (-2))
  (x, af x)

def H : Point := 
  let af := line_eq A F
  let bd : ℝ → ℝ := fun x => x
  let x := 3 / 2
  (x, af x)

def area (p1 p2 p3 p4 : Point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let (x4, y4) := p4
  0.5 * abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

theorem area_BEIH : 
  area B E I H = 27 / 200 := 
  sorry

end area_BEIH_l237_237334


namespace square_area_from_conditions_l237_237629

theorem square_area_from_conditions :
  ∀ (r s l b : ℝ), 
  l = r / 4 →
  r = s →
  l * b = 35 →
  b = 5 →
  s^2 = 784 := 
by 
  intros r s l b h1 h2 h3 h4
  sorry

end square_area_from_conditions_l237_237629


namespace bus_driver_total_compensation_l237_237738

def regular_rate := 14 -- dollars per hour
def regular_hours := 40 -- hours
def total_hours_worked := 57.88 -- hours

def regular_pay (rate : ℝ) (hours : ℝ) : ℝ := rate * hours
def overtime_rate (rate : ℝ) : ℝ := rate + (0.75 * rate)
def overtime_hours (total_hours : ℝ) (regular_hours : ℝ) : ℝ := total_hours - regular_hours
def overtime_pay (o_rate : ℝ) (o_hours : ℝ) : ℝ := o_rate * o_hours

noncomputable def total_compensation (rate : ℝ) (regular_hours : ℝ) (total_hours : ℝ) : ℝ :=
  regular_pay rate regular_hours + overtime_pay (overtime_rate rate) (overtime_hours total_hours regular_hours)

theorem bus_driver_total_compensation :
  total_compensation regular_rate regular_hours total_hours_worked ≈ 998.06 :=
by
  sorry

end bus_driver_total_compensation_l237_237738


namespace next_work_together_instructors_l237_237144

theorem next_work_together_instructors :
  let Alex := 9
  let Morgan := 5
  let Jamie := 8
  let Pat := 10
  let Kim := 12
  Nat.lcm Alex (Nat.lcm Morgan (Nat.lcm Jamie (Nat.lcm Pat Kim))) = 360 := 
by 
  let Alex := 9
  let Morgan := 5
  let Jamie := 8
  let Pat := 10
  let Kim := 12
  calc
    Nat.lcm Alex (Nat.lcm Morgan (Nat.lcm Jamie (Nat.lcm Pat Kim)))
        = Nat.lcm 9 (Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 12)))  := by rfl
    ... = 360  := by sorry

end next_work_together_instructors_l237_237144


namespace total_hours_before_midterms_l237_237392

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l237_237392


namespace masks_purchased_in_first_batch_l237_237660

theorem masks_purchased_in_first_batch
    (cost_first_batch cost_second_batch : ℝ)
    (quantity_ratio : ℝ)
    (unit_price_difference : ℝ)
    (h1 : cost_first_batch = 1600)
    (h2 : cost_second_batch = 6000)
    (h3 : quantity_ratio = 3)
    (h4 : unit_price_difference = 2) :
    ∃ x : ℝ, (cost_first_batch / x) + unit_price_difference = (cost_second_batch / (quantity_ratio * x)) ∧ x = 200 :=
by {
    sorry
}

end masks_purchased_in_first_batch_l237_237660


namespace find_a3_plus_a5_l237_237982

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (m : ℝ)
variable (n : ℕ)

-- Conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q a₁, a = λ n, a₁ * q^n

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (finset.range n).sum a

axiom sum_condition : ∀ n, S (n + 1) = m * 2^(n + 1) - 5
axiom fourth_term : a 4 = 40

-- Question
theorem find_a3_plus_a5 (h1 : geometric_sequence a) (h2 : sum_first_n_terms a S) (h3 : sum_condition) (h4 : fourth_term) :
  a 3 + a 5 = 100 :=
sorry

end find_a3_plus_a5_l237_237982


namespace count_diff_of_two_primes_l237_237920

theorem count_diff_of_two_primes (s : Set ℕ) (p : ℕ → Prop) (n : ℕ) :
  (∀ i, i ∈ s ↔ ∃ n : ℕ, i = 10 * n + 7) →
  (∀ k, k ∈ s → ∃ a b : ℕ, nat.prime a → nat.prime b → a - b = k) →
  #{i ∈ s | ∃ a b : ℕ, nat.prime a ∧ nat.prime b ∧ a - b = i} = 2 :=
by
  -- sorry as the proof would be filled in here
  sorry

end count_diff_of_two_primes_l237_237920


namespace problem_statement_l237_237355

noncomputable def f (x k : ℝ) : ℝ :=
  (1/5) * (x - k + 4500 / x)

noncomputable def fuel_consumption_100km (x k : ℝ) : ℝ :=
  100 / x * f x k

theorem problem_statement (x k : ℝ)
  (hx1 : 60 ≤ x) (hx2 : x ≤ 120)
  (hk1 : 60 ≤ k) (hk2 : k ≤ 100)
  (H : f 120 k = 11.5) :

  (∀ x, 60 ≤ x ∧ x ≤ 100 → f x k ≤ 9 ∧ 
  (if 75 ≤ k ∧ k ≤ 100 then fuel_consumption_100km (9000 / k) k = 20 - k^2 / 900
   else fuel_consumption_100km 120 k = 105 / 4 - k / 6)) :=
  sorry

end problem_statement_l237_237355


namespace vector_proof_l237_237492

noncomputable def vector_angle (a b : ℝ) : ℝ := 
  let θ : ℝ := (real.acos (-1 / 2))
  θ

noncomputable def vector_magnitude (a : ℝ) (b : ℝ) : ℝ :=
  real.sqrt (9 * a^2 + b^2 + 6 * a * b * (-1 / 2))

noncomputable def dot_product (a b : ℝ) : ℝ := 
  (1 / 2 * a^2) + (1 / 2 * b^2) + (5 / 4 * a * b)

theorem vector_proof :
  (|a| = 2) → (|b| = 4) → ((a - b) • b = -20) → 
  vector_angle a b = (2 * real.pi / 3) ∧ 
  vector_magnitude 3 a b = 2 * real.sqrt 7 ∧
  dot_product (1 / 2 * a + b) (a + 1 / 2 * b) = 10 :=
by
  intros ha hb hdot
  sorry

end vector_proof_l237_237492


namespace magnitude_of_z_l237_237940

def i : ℂ := complex.I

theorem magnitude_of_z :
  let z := (1 : ℂ) + 2 * i + i^3 in
  complex.abs z = real.sqrt 2 :=
by
  let z := (1 : ℂ) + 2 * i + i^3
  sorry

end magnitude_of_z_l237_237940


namespace wendy_points_calc_l237_237302

/-- 
Wendy earned 5 points for each bag of cans she recycled, 
and 10 points for each bundle of newspapers she recycled. 
If she had 11 bags of cans, recycled 9 of them, and recycled 3 bundles of newspapers, 
how many points would she have earned in total?
-/
theorem wendy_points_calc (cans_points : ℕ) (newspapers_points : ℕ) (cans_recycled : ℕ) (newspapers_recycled : ℕ) : 
  cans_points = 5 ∧ newspapers_points = 10 ∧ cans_recycled = 9 ∧ newspapers_recycled = 3 → 
  cans_recycled * cans_points + newspapers_recycled * newspapers_points = 75 :=
by
  intros h
  cases h with hcans hnewspapers
  cases hnewspapers with hcans_recycled hnewspapers_recycled
  cases hcans_recycled with hcans_points hcans_recycled
  rw [hcans_points, hcans, hnewspapers_recycled, hnewspapers_recycled]
  norm_num
  sorry

end wendy_points_calc_l237_237302


namespace gravitational_force_inversely_proportional_square_distance_l237_237626

theorem gravitational_force_inversely_proportional_square_distance :
  ∀ (f₁ : ℝ) (d₁ d₂ : ℝ), f₁ = 480 ∧ d₁ = 5000 ∧ d₂ = 300000 →
  ∃ (f₂ : ℝ), f₂ = 1 / 75 :=
by
  intros f₁ d₁ d₂ h,
  obtain ⟨hf₁, hd₁, hd₂⟩ := h,
  have h₁ : f₁ * d₁^2 = 12000000000,
  { rw [hf₁, hd₁],
    norm_num },
  use 1 / 75,
  sorry

end gravitational_force_inversely_proportional_square_distance_l237_237626


namespace total_hours_eq_52_l237_237389

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l237_237389


namespace number_of_differences_is_one_l237_237918

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def can_be_written_as_difference_of_two_primes (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 - p2

def set_of_numbers : set ℕ := { n | ∃ k : ℕ, n = 7 + 10 * k }

def count_numbers_that_are_differences_of_two_primes : ℕ :=
  (set_of_numbers.filter can_be_written_as_difference_of_two_primes).card

theorem number_of_differences_is_one :
  count_numbers_that_are_differences_of_two_primes = 1 :=
sorry

end number_of_differences_is_one_l237_237918


namespace graph_two_lines_l237_237036

theorem graph_two_lines (x y : ℝ) : 3 * x^2 - 36 * y^2 - 18 * x + 27 = 0 ↔ (∃ (a : ℝ), x = 3 + 2 * (sqrt 3) * y) ∨ (∃ (b : ℝ), x = 3 - 2 * (sqrt 3) * y) :=
sorry

end graph_two_lines_l237_237036


namespace general_term_formula_l237_237891

variable (a S : ℕ → ℚ)

-- Condition 1: The sum of the first n terms of the sequence {a_n} is S_n
def sum_first_n_terms (n : ℕ) : ℚ := S n

-- Condition 2: a_n = 3S_n - 2
def a_n (n : ℕ) : Prop := a n = 3 * S n - 2

theorem general_term_formula (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ k, k ≥ 2 → a (k) = - (1/2) * a (k - 1) ) : 
  a n = (-1/2)^(n-1) :=
sorry

end general_term_formula_l237_237891


namespace smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l237_237544

theorem smallest_number_after_operations_n_111 :
  ∀ (n : ℕ), n = 111 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 111 →
       (f l) = 0)) :=
by 
  sorry

theorem smallest_number_after_operations_n_110 :
  ∀ (n : ℕ), n = 110 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 110 →
       (f l) = 1)) :=
by 
  sorry

end smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l237_237544


namespace sum_of_divisors_of_45_l237_237312

theorem sum_of_divisors_of_45 : 
  (∑ d in (finset.filter (λ x : ℕ, 45 % x = 0) (finset.range (45 + 1))), d) = 78 :=
by sorry

end sum_of_divisors_of_45_l237_237312


namespace sum_of_divisors_of_45_l237_237306

theorem sum_of_divisors_of_45 : (∑ d in (Finset.filter (λ x, 45 % x = 0) (Finset.range 46)), d) = 78 :=
by
  -- We'll need to provide the proof here
  sorry

end sum_of_divisors_of_45_l237_237306


namespace quadratic_tangent_sum_l237_237817

theorem quadratic_tangent_sum (b x₁ x₂ x₃ : ℝ) :
  (-7 * x + -15 ≤ 2 * x + -2 ∧ 2 * x + -2 ≤ 5 * x + 10) ∨ 
  (-7 * x + -15 ≤ 5 * x + 10 ∧ 5 * x + 10 ≤ 2 * x + -2) → 
  (q(x) - (-7 * x - 15) = b * (x - x₁)^2 ∧ q(x) - (2 * x - 2) = b * (x - x₂)^2 ∧ q(x) - (5 * x + 10) = b * (x - x₃)^2):
  x₁ + x₂ + x₃ = -17/3 :=
  sorry

end quadratic_tangent_sum_l237_237817


namespace number_of_extreme_value_points_l237_237263

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem number_of_extreme_value_points : ∃! c : ℝ, c > 0 ∧ (deriv f c = 0) :=
by
  sorry

end number_of_extreme_value_points_l237_237263


namespace number_of_differences_is_one_l237_237919

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def can_be_written_as_difference_of_two_primes (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 - p2

def set_of_numbers : set ℕ := { n | ∃ k : ℕ, n = 7 + 10 * k }

def count_numbers_that_are_differences_of_two_primes : ℕ :=
  (set_of_numbers.filter can_be_written_as_difference_of_two_primes).card

theorem number_of_differences_is_one :
  count_numbers_that_are_differences_of_two_primes = 1 :=
sorry

end number_of_differences_is_one_l237_237919


namespace non_intersecting_chords_l237_237112

theorem non_intersecting_chords (n : ℕ) (h : n > 0) :
  ∃ (C : set (ℕ × ℕ)), C.card = 2 * n ∧ 
  (∀ (x y : ℕ × ℕ), x ∈ C → y ∈ C → disjoint (set Ioo x) (set Ioo y)) ∧ 
  (∀ (x : ℕ × ℕ), x ∈ C → | x.fst - x.snd | ≤ 3 * n - 1) :=
sorry

end non_intersecting_chords_l237_237112


namespace parameter_values_for_roots_l237_237065

theorem parameter_values_for_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 5 * x2 ∧ a * x1^2 - (2 * a + 5) * x1 + 10 = 0 ∧ a * x2^2 - (2 * a + 5) * x2 + 10 = 0)
  ↔ (a = 5 / 3 ∨ a = 5) := 
sorry

end parameter_values_for_roots_l237_237065


namespace keys_missing_l237_237988

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end keys_missing_l237_237988


namespace problem_l237_237130

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≥ f y
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def max_value_in (f : ℝ → ℝ) (a b : ℝ) (v : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → f x ≤ v ∧ (∃ z, a ≤ z ∧ z ≤ b ∧ f z = v)

theorem problem
  (h_even : even_function f)
  (h_decreasing : decreasing_on f (-5) (-2))
  (h_max : max_value_in f (-5) (-2) 7) :
  increasing_on f 2 5 ∧ max_value_in f 2 5 7 :=
by
  sorry

end problem_l237_237130


namespace Alan_eggs_count_l237_237783

theorem Alan_eggs_count (Price_per_egg Chickens_bought Price_per_chicken Total_spent : ℕ)
  (h1 : Price_per_egg = 2) (h2 : Chickens_bought = 6) (h3 : Price_per_chicken = 8) (h4 : Total_spent = 88) :
  ∃ E : ℕ, 2 * E + Chickens_bought * Price_per_chicken = Total_spent ∧ E = 20 :=
by
  sorry

end Alan_eggs_count_l237_237783


namespace collinear_vectors_l237_237071

theorem collinear_vectors (m n : ℝ) 
  (h_collinear : ∃ (λ : ℝ), (1, -2, m) = (λ * n, λ * 4, λ * 6)) :
  m - 2 * n = 1 :=
by
  cases h_collinear with λ hλ
  sorry

end collinear_vectors_l237_237071


namespace system_solution_l237_237424

theorem system_solution (x y z : ℝ) 
  (h1 : 2 * x - 3 * y + z = 8) 
  (h2 : 4 * x - 6 * y + 2 * z = 16) 
  (h3 : x + y - z = 1) : 
  x = 11 / 3 ∧ y = 1 ∧ z = 11 / 3 :=
by
  sorry

end system_solution_l237_237424


namespace find_k_value_l237_237877

theorem find_k_value
  (A : ℝ × ℝ)
  (k m : ℝ)
  (l_intersects_O_at_B_and_C : ∃ B C : ℝ × ℝ, (B ≠ C) ∧ (B.1^2 + B.2^2 = 1) ∧ (C.1^2 + C.2^2 = 1) ∧ (B.2 = k * B.1 - m) ∧ (C.2 = k * C.1 - m))
  (S1 S2 : ℝ)
  (h_area_ABC : S1)
  (h_area_OBC : S2)
  (angle_BAC_eq_60 : ∀ B C : ℝ × ℝ, (60 : ℝ) = real.angle ((B.1 - 0, B.2 - 1), (C.1 - 0, C.2 - 1)))
  (area_relation : S1 = 2 * S2) :
  k = real.sqrt 3 ∨ k = -real.sqrt 3 := 
sorry

end find_k_value_l237_237877


namespace expression_is_integer_l237_237228

theorem expression_is_integer (n : ℤ) : (∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k) := 
sorry

end expression_is_integer_l237_237228


namespace stratified_sampling_third_grade_l237_237140

theorem stratified_sampling_third_grade (total_students : ℕ) (first_grade_students : ℕ)
  (second_grade_students : ℕ) (third_grade_students : ℕ) (sample_size : ℕ)
  (h_total : total_students = 270000) (h_first : first_grade_students = 99000)
  (h_second : second_grade_students = 90000) (h_third : third_grade_students = 81000)
  (h_sample : sample_size = 3000) :
  third_grade_students * (sample_size / total_students) = 900 := 
by {
  sorry
}

end stratified_sampling_third_grade_l237_237140


namespace distinct_arrangements_l237_237426

theorem distinct_arrangements : 
  ∃ (n : ℕ), (∃ (students dormitories : set ℕ), students.card = 5 ∧ dormitories.card = 2 ∧ ∀ d ∈ dormitories, d.card ≥ 2) ∧ n = 20 :=
by
  let students := {1, 2, 3, 4, 5}
  let dormitories := {A, B}
  have h_students_card : students.card = 5 := by sorry
  have h_dormitories_card : dormitories.card = 2 := by sorry
  have h_each_dormitory : ∀ d ∈ dormitories, d.card ≥ 2 := by sorry
  existsi 20
  exact ⟨⟨students, dormitories, h_students_card, h_dormitories_card, h_each_dormitory⟩, rfl⟩

end distinct_arrangements_l237_237426


namespace distance_from_F_to_B_l237_237979

-- Conditions
variables (A B C F : Type) [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace F]

-- Triangle ABC is right-angled at A
variables (AB AC BC : ℕ)
variable (h1 : AB = 160)
variable (h2 : AC = 120)
variable (h3 : BC ^ 2 = AB ^ 2 + AC ^ 2)
-- Jogging distances
variable (FB FC : ℕ)
variable (h4 : FB + B = FC + C)
variable (Jack_Jill_Same_Distance : AB + FB = AC + FC)

-- Prove
theorem distance_from_F_to_B : 
  FB = 80 :=
by sorry

end distance_from_F_to_B_l237_237979


namespace proof_problem_l237_237821

-- Define the function f(x) = -x - x^3
def f (x : ℝ) : ℝ := -x - x^3

-- Define the main theorem according to the conditions and the required proofs.
theorem proof_problem (x1 x2 : ℝ) (h : x1 + x2 ≤ 0) :
  (f x1) * (f (-x1)) ≤ 0 ∧ (f x1 + f x2) ≥ (f (-x1) + f (-x2)) :=
by
  sorry

end proof_problem_l237_237821


namespace grasshopper_can_jump_exactly_7_3_meters_l237_237747

-- Define the regular jump distance
def jump_distance : ℝ := 0.5

-- Define the total required distance
def required_distance : ℝ := 7.3

-- Define the triangular jump pattern that adds to the remaining distance
noncomputable def can_jump_to (d : ℝ) : Prop :=
  ∃ (n : ℕ), (n * jump_distance) ≤ d ∧ (d - n * jump_distance) ∈ {0.3}

-- Theorem statement
theorem grasshopper_can_jump_exactly_7_3_meters :
  can_jump_to required_distance :=
by {
  sorry
}

end grasshopper_can_jump_exactly_7_3_meters_l237_237747


namespace student_score_max_marks_l237_237337

theorem student_score_max_marks (M : ℝ)
  (pass_threshold : ℝ := 0.60 * M)
  (student_marks : ℝ := 80)
  (fail_by : ℝ := 40)
  (required_passing_score : ℝ := student_marks + fail_by) :
  pass_threshold = required_passing_score → M = 200 := 
by
  sorry

end student_score_max_marks_l237_237337


namespace n_mod_5_division_of_grid_l237_237446

theorem n_mod_5_division_of_grid (n : ℕ) :
  (∃ m : ℕ, n^2 = 4 + 5 * m) ↔ n % 5 = 2 :=
by
  sorry

end n_mod_5_division_of_grid_l237_237446


namespace reduced_price_per_kg_l237_237759

noncomputable def original_price (P : Real) : Real := P

theorem reduced_price_per_kg (P X : Real) (h1 : 0 = P - (20/100) * P)
  (h2 : 684 = X * P)
  (h3 : 684 = (X + 4) * (P - (0.2 * P))) :
  (684 / (X + 4)) = 34.2 := 
by
  -- Reduced price R can be substituted as 0.8P or P - 0.2P
  let R := 0.8 * P
  have h_reduced_price : R = 0.8 * P, by sorry
  -- The quantity of oil after the price reduction
  let new_quantity := X + 4
  have h_quantity : new_quantity = X + 4, by sorry
  -- Equation equality conditions
  have hc1 : 684 = X * P, by sorry
  have hc2 : 684 = (X + 4) * 0.8 * P, by sorry
  -- Therefore, reduced price per kg of oil
  exact Real.div_eq_of_eq_mul _ _ _ sorry

end reduced_price_per_kg_l237_237759


namespace emails_in_morning_and_evening_l237_237550

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 4
def evening_emails : ℕ := 8

theorem emails_in_morning_and_evening : morning_emails + evening_emails = 11 :=
by
  sorry

end emails_in_morning_and_evening_l237_237550


namespace align_vertex_with_protractor_center_l237_237718

theorem align_vertex_with_protractor_center
    (angle : Type)
    (protractor : Type)
    (vertex : Type)
    (center_point : vertex = protractor.center) :
  "When measuring an angle, the vertex of the angle should align with the protractor's center point." :=
begin
  sorry
end

end align_vertex_with_protractor_center_l237_237718


namespace middle_circle_radius_l237_237853

-- Define the conditions as variables and statements
variables {L : Type*} [line L]
variables {circle : Type*} [circle_tangent_to_line : circle → L → Prop]
variables (C1 C2 C3 C4 C5 : circle)
variables (r1 r2 r3 r4 r5 : ℝ)

-- Specific conditions given in the problem
-- Radii of the largest (r1) and smallest (r5) circles
def largest_radius := 20
def smallest_radius := 10

-- The middle circle to prove
def middle_radius := 10 * real.sqrt 2

-- Proof problem statement
theorem middle_circle_radius :
    circle_tangent_to_line C1 L ∧
    circle_tangent_to_line C2 L ∧
    circle_tangent_to_line C3 L ∧
    circle_tangent_to_line C4 L ∧
    circle_tangent_to_line C5 L ∧
    (∀ i j, (i ≠ j → tangent (circle i) (circle j))) ∧
    r1 = largest_radius ∧
    r5 = smallest_radius  ∧
    (∃ r, r^2 = r1 * r5) →
    r3 = middle_radius :=
begin
  sorry,
end

end middle_circle_radius_l237_237853


namespace six_digit_number_l237_237500

def is_permutation (n m : ℕ) : Prop :=
  let digits := n.digits 10
  let mdigits := m.digits 10
  digits.length = mdigits.length ∧ Multiset.ofList digits = Multiset.ofList mdigits

theorem six_digit_number (N : ℕ) (h : N = 142857) :
  (2 * N).digits 10 = (N.digits 10).perm ∧
  (3 * N).digits 10 = (N.digits 10).perm ∧
  (4 * N).digits 10 = (N.digits 10).perm ∧
  (5 * N).digits 10 = (N.digits 10).perm ∧
  (6 * N).digits 10 = (N.digits 10).perm :=
by
  introv h
  rw [h, mul_assoc]
  sorry

end six_digit_number_l237_237500


namespace count_numbers_with_three_transitions_l237_237034

def has_three_transitions (n : ℕ) : Prop :=
  let binary_digits := (nat.digits 2 n).reverse in
  (list.zip_with (≠) binary_digits (list.tail binary_digits)).count id = 3

theorem count_numbers_with_three_transitions : 
  (finset.range 51).filter has_three_transitions).card = 4 := by
  sorry

end count_numbers_with_three_transitions_l237_237034


namespace exists_integer_midpoint_l237_237236

theorem exists_integer_midpoint (P : Fin 5 → (ℤ × ℤ)) :
  ∃ (i j : Fin 5), i ≠ j ∧ ∃ Q: ℤ × ℤ, Q ≠ P i ∧ Q ≠ P j ∧
  (∃ (k: Fin 2), (Q.1, Q.2) = ((P i).1 + (P j).1) / 2, ((P i).2 + (P j).2) / 2) :=
by
  sorry

end exists_integer_midpoint_l237_237236


namespace find_english_marks_l237_237420

-- Define David's marks in different subjects and the average marks
def marks (mathematics physics chemistry biology english : ℕ) : Prop :=
  mathematics = 65 ∧
  physics = 82 ∧
  chemistry = 67 ∧
  biology = 85 ∧
  ((mathematics + physics + chemistry + biology + english) / 5) = 75

-- Theorem to find David's marks in English
theorem find_english_marks : ∃ e : ℕ, marks 65 82 67 85 e ∧ e = 76 :=
begin
  -- Proof area to be filled
  sorry
end

end find_english_marks_l237_237420


namespace shortest_path_room_sum_l237_237154

theorem shortest_path_room_sum {rooms : Type*} (reachable : rooms → rooms → Prop) :
  ∃ (p : list rooms), 
    p.head = some 1 ∧ 
    p.last = some 16 ∧ 
    (∀ i, i ∈ p → reachable (p.nth i) (p.nth (i + 1))) ∧ 
    list.length p = 13 ∧ 
    list.sum p = 114 :=
sorry

end shortest_path_room_sum_l237_237154


namespace tangent_line_at_M_l237_237477

noncomputable def isOnCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def M : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem tangent_line_at_M (hM : isOnCircle (M.1) (M.2)) : (∀ x y, M.1 = x ∨ M.2 = y → x + y = Real.sqrt 2) :=
by
  sorry

end tangent_line_at_M_l237_237477


namespace evaluate_imaginary_unit_powers_l237_237832

theorem evaluate_imaginary_unit_powers :
  let i := Complex.I in
  (i ^ 23 + i ^ 45 = 0) :=
by
  sorry

end evaluate_imaginary_unit_powers_l237_237832


namespace valid_q_range_l237_237434

noncomputable def polynomial_has_nonneg_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ (x^4 + q*x^3 + x^2 + q*x + 4 = 0)

theorem valid_q_range (q : ℝ) : polynomial_has_nonneg_root q → q ≤ -2 * Real.sqrt 2 := 
sorry

end valid_q_range_l237_237434


namespace real_m_of_complex_num_l237_237102

theorem real_m_of_complex_num (m : ℝ) : 
  let z : ℂ := (1 + m * complex.I) / (1 + complex.I) in z.im = 0 → m = 1 :=
by
  intro z_im_eq_zero
  sorry

end real_m_of_complex_num_l237_237102


namespace height_of_water_l237_237612

theorem height_of_water (volume : ℝ) (bottom_area : ℝ) (h_volume : volume = 2000) (h_bottom_area : bottom_area = 50) : (volume / bottom_area) = 40 :=
by {
  rw [h_volume, h_bottom_area],
  norm_num,
  sorry
}

end height_of_water_l237_237612


namespace inequality_sum_powers_l237_237598

theorem inequality_sum_powers (n k : ℕ) :
  ∑ i in Finset.range (n + 1), i ^ k ≤ (n ^ (2 * k) - (n - 1) ^ k) / (n ^ k - (n - 1) ^ k) :=
by sorry

end inequality_sum_powers_l237_237598


namespace value_of_sum_is_eleven_l237_237248

-- Define the context and conditions

theorem value_of_sum_is_eleven (x y z w : ℤ) 
  (h1 : x - y + z = 7)
  (h2 : y - z + w = 8)
  (h3 : z - w + x = 4)
  (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
begin
  sorry
end

end value_of_sum_is_eleven_l237_237248


namespace sum_of_divisors_of_45_l237_237310

theorem sum_of_divisors_of_45 : 
  (∑ d in (finset.filter (λ x : ℕ, 45 % x = 0) (finset.range (45 + 1))), d) = 78 :=
by sorry

end sum_of_divisors_of_45_l237_237310


namespace g_at_pi_over_3_eq_one_l237_237866

def f (ω φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + φ)
def g (ω φ : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (ω * x + φ) + 1

theorem g_at_pi_over_3_eq_one (ω φ : ℝ) (hf_sym : ∀ x : ℝ, 3 * Real.sin (ω * (π / 3 + x) + φ) = 3 * Real.sin (ω * (π / 3 - x) + φ)) :
  g ω φ (π / 3) = 1 := by
  sorry

end g_at_pi_over_3_eq_one_l237_237866


namespace triangle_side_range_l237_237525

variable {A B C : ℝ} {a b c S : ℝ}

theorem triangle_side_range (h1 : S = 2) (h2 : a ∈ ℕ ∧ b ∈ ℕ ∧ c ∈ ℕ) (h3 : a * Real.cos B = b * (1 + Real.cos A)) (h4 : ∠A + ∠B + ∠C = Real.pi) :
  8 * Real.sqrt 2 - 8 < (c + a - b) * (c + b - a) ∧ (c + a - b) * (c + b - a) < 8 :=
by
  sorry

end triangle_side_range_l237_237525


namespace true_option_l237_237089

variables (x x₀ : ℝ)

def p : Prop := ∀ x < 1, log (1/3 : ℝ) x < 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 ≥ 2^x₀

theorem true_option : p ∨ q :=
by {
  -- Proposition q is true because there exists such an x₀ that x₀ = 2 satisfies x₀^2 ≥ 2^x₀.
  have hq: q, {
    use 2,
    norm_num,
  },
  -- Since we have q, we can conclude p ∨ q.
  exact or.inr hq,
}

end true_option_l237_237089


namespace jaylen_dog_food_consumption_l237_237173

theorem jaylen_dog_food_consumption :
  ∀ (morning evening daily_consumption total_food : ℕ)
  (days : ℕ),
  (morning = evening) →
  (total_food = 32) →
  (days = 16) →
  (daily_consumption = total_food / days) →
  (morning + evening = daily_consumption) →
  morning = 1 := by
  intros morning evening daily_consumption total_food days h_eq h_total h_days h_daily h_sum
  sorry

end jaylen_dog_food_consumption_l237_237173


namespace painted_cubes_multiple_unpainted_cubes_l237_237299

/-- Prove that there exist values of \(n\) such that the number of painted cubes 
    is a multiple of the number of unpainted cubes in a cube with side length \(n + 2\). -/
theorem painted_cubes_multiple_unpainted_cubes (n : ℕ) (hn : n ∈ {1, 2, 4, 8}) :
  n^3 ∣ (n + 2)^3 - n^3 :=
by {
  sorry
}

end painted_cubes_multiple_unpainted_cubes_l237_237299


namespace intersection_A_B_subset_A_C_l237_237904

section
variables {x m : ℝ}

noncomputable def A := {x | x^2 - 5 * x - 6 < 0}
noncomputable def B := {x | 6 * x^2 - 5 * x + 1 ≥ 0}
noncomputable def C (m : ℝ) := {x | (x - m) * (x - (m + 9)) < 0}

-- (1) Prove that A ∩ B = {x | -1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6}
theorem intersection_A_B : 
  (∀ x, x ∈ A ∩ B ↔ (-1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6)) := sorry

-- (2) Prove that if A ⊆ C, then −3 ≤ m ≤ −1
theorem subset_A_C (m : ℝ) : 
  (A ⊆ C m) → (-3 ≤ m ∧ m ≤ -1) := sorry

end

end intersection_A_B_subset_A_C_l237_237904


namespace decrypt_message_consistent_l237_237218

def digit_value : Type := ℤ

structure Decipher :=
  (M S T I K : digit_value)
  (distinct : ∀ a b c d e f g h i j, set.Union {a, b, c, d, e, f, g, h, i, j} = {S, T, I, K, M} → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j)
  (no_leading_zero : ∀ (a b : digit_value), a ≠ 0)

theorem decrypt_message_consistent (d : Decipher) (M S T I K : digit_value) : 
  M = 1 ∧ S = 5 ∧ T = 7 ∧ I = 8 ∧ K = 6 → 
  (S * 10000 + T * 1000 + I * 100 + K * 10 + S) + 
  (S * 10000 + T * 1000 + I * 100 + K * 10 + S) = 
  (M * 100000 + S * 10000 + T * 1000 + I * 100 + K * 10 + S) :=
begin
  sorry
end

end decrypt_message_consistent_l237_237218


namespace no_pentagon_on_cube_edges_l237_237041

-- Define a regular pentagon.
structure RegularPentagon :=
(vertices : Fin 5 → EuclideanSpace ℝ (Fin 3))
(side_length : ℝ)
(is_regular : ∀ i : Fin 5, ∃ j : Fin 5, j ≠ i ∧ dist (vertices i) (vertices j) = side_length)

-- Define a cube and its edges.
structure Cube :=
(vertices : Fin 8 → EuclideanSpace ℝ (Fin 3))
(is_cubical : ∀ (i j : Fin 8), ∃ a b c : ℝ, vertices j - vertices i = a * (EuclideanSpace.basis 0) + b * (EuclideanSpace.basis 1) + c * (EuclideanSpace.basis 2) ∧ (a^2 + b^2 + c^2 = 1 ∨ a^2 + b^2 + c^2 = 0.5))

-- Define the problem statement.
theorem no_pentagon_on_cube_edges (P : RegularPentagon) (C : Cube) :
  ¬ (∃ (f : Fin 5 → Fin 12), ∀ i: Fin 5, ∃ (e : Fin 2), ∃ (a b : ℝ), C.vertices (Fin.toFin8 f i).fst + a * (EuclideanSpace.basis e.fst) = P.vertices i) :=
sorry

end no_pentagon_on_cube_edges_l237_237041


namespace sum_of_prime_factors_of_six_pow_six_minus_six_pow_two_l237_237055

theorem sum_of_prime_factors_of_six_pow_six_minus_six_pow_two : 
  let number := 6^6 - 6^2 in
  let factors := [2, 3, 5, 7, 37] in
  ∑ (i : ℕ) in factors.to_finset, i = 54 := by
  sorry

end sum_of_prime_factors_of_six_pow_six_minus_six_pow_two_l237_237055


namespace andrew_vacation_days_l237_237792

theorem andrew_vacation_days (days_worked last_year vacation_per_10 worked_days in_march in_september : ℕ)
  (h1 : vacation_per_10 = 10)
  (h2 : days_worked_last_year = 300)
  (h3 : worked_days = days_worked_last_year / vacation_per_10)
  (h4 : in_march = 5)
  (h5 : in_september = 2 * in_march)
  (h6 : days_taken = in_march + in_september)
  (h7 : vacation_days_remaining = worked_days - days_taken) :
  vacation_days_remaining = 15 :=
by
  sorry

end andrew_vacation_days_l237_237792


namespace problem_l237_237537

-- Condition that defines s and t
def s : ℤ := 4
def t : ℤ := 3

theorem problem (s t : ℤ) (h_s : s = 4) (h_t : t = 3) : s - 2 * t = -2 := by
  sorry

end problem_l237_237537


namespace height_of_formed_cylinder_l237_237663

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4/3) * π * (r^3)

noncomputable def total_volume_of_spheres (n : ℕ) (r : ℝ) : ℝ :=
  n * volume_of_sphere r

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * (r^2) * h

noncomputable def height_of_cylinder (total_volume : ℝ) (r : ℝ) : ℝ :=
  total_volume / (π * (r^2))

theorem height_of_formed_cylinder :
  height_of_cylinder (total_volume_of_spheres 12 2) 3 = 128 / 9 :=
by
  sorry

end height_of_formed_cylinder_l237_237663


namespace number_of_distinct_triangles_l237_237818

open Int Finset

theorem number_of_distinct_triangles : 
  let xs := (range (2010 / 31 + 1)).filter fun x => 31 * x < 2011
  let ps := xs.to_finset.powerset \ (range (2010 / 31 + 1)).powerset.filter (fun s => s.card != 2)
  let triangles := ps.filter (fun s => x_1 - x_2 % 2 = 0)
  triangles.card = 1024 :=
by sorry

end number_of_distinct_triangles_l237_237818


namespace triangle_QS_l237_237244

theorem triangle_QS (RS QR QS: ℝ) : RS = 13 → QR = 5 → sqrt (RS^2 - QR^2) = 12 :=
by
  intros h1 h2
  rw [h1, h2]
  calc
  sqrt (13^2 - 5^2) = sqrt 144 : by norm_num
  ...                = 12       : by norm_num

end triangle_QS_l237_237244


namespace find_f_2012_l237_237575

noncomputable def f : ℝ → ℝ := sorry

lemma functional_eq (x : ℝ) : f(x + 2) + f(x) = x :=
sorry

lemma interval_eq (x : ℝ) (h : -2 < x ∧ x ≤ 0) : f(x) = x^3 :=
sorry

theorem find_f_2012 : f(2012) = 1006 :=
sorry

end find_f_2012_l237_237575


namespace club_members_tennis_l237_237524

theorem club_members_tennis :
  ∀ (total_members badminton_players no_sport both_sports : ℕ)
    (h_total : total_members = 30)
    (h_badminton : badminton_players = 16)
    (h_no_sport : no_sport = 2)
    (h_both : both_sports = 7),
  let total_playing_sport := total_members - no_sport,
      only_badminton := badminton_players - both_sports,
      only_tennis := total_playing_sport - only_badminton - both_sports in
  only_tennis + both_sports = 19 :=
by intros; simp; sorry

end club_members_tennis_l237_237524


namespace positional_relationship_between_MA_and_BD_l237_237502

-- Definitions and assumptions
variables {A B C D M : Type} 
variable [h_rhombus : Rhombus A B C D]
variable [perpendicular : isPerpendicular (line A M) (plane A B C D M)]

-- Problem Statement
theorem positional_relationship_between_MA_and_BD :
  skew (line A M) (line B D) ∧ ¬perpendicular (line A M) (line B D) :=
sorry

end positional_relationship_between_MA_and_BD_l237_237502


namespace alpha_identity_l237_237097

theorem alpha_identity (α : ℝ) (hα : α ≠ 0) (h_tan : Real.tan α = -α) : 
    (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := 
by
  sorry

end alpha_identity_l237_237097


namespace distance_from_point_P_to_plane_l237_237892

noncomputable def distance_point_to_plane
  (P A : (ℝ × ℝ × ℝ))
  (n : (ℝ × ℝ × ℝ)) : ℝ :=
  abs (n.1 * (A.1 - P.1) + n.2 * (A.2 - P.2) + n.3 * (A.3 - P.3)) / (real.sqrt (n.1^2 + n.2^2 + n.3^2))

theorem distance_from_point_P_to_plane :
  let P := (4, 3, 2)
  let A := (2, 3, 1)
  let n := (1, 0, -1)
  distance_point_to_plane P A n = real.sqrt 2 / 2 := by
  sorry

end distance_from_point_P_to_plane_l237_237892


namespace ratio_of_areas_ratio_of_perimeters_l237_237268

-- Define side lengths
def side_length_A : ℕ := 48
def side_length_B : ℕ := 60

-- Define the area of squares
def area_square (side_length : ℕ) : ℕ := side_length * side_length

-- Define the perimeter of squares
def perimeter_square (side_length : ℕ) : ℕ := 4 * side_length

-- Theorem for the ratio of areas
theorem ratio_of_areas : (area_square side_length_A) / (area_square side_length_B) = 16 / 25 :=
by
  sorry

-- Theorem for the ratio of perimeters
theorem ratio_of_perimeters : (perimeter_square side_length_A) / (perimeter_square side_length_B) = 4 / 5 :=
by
  sorry

end ratio_of_areas_ratio_of_perimeters_l237_237268


namespace find_initial_terms_general_formula_l237_237148

noncomputable def sequence (n : ℕ) : ℝ := if n = 0 then 0 else real.sqrt n - real.sqrt (n - 1)

theorem find_initial_terms :
  (sequence 1 = 1) ∧ 
  (sequence 2 = real.sqrt 2 - 1) ∧ 
  (sequence 3 = real.sqrt 3 - real.sqrt 2) :=
by {
  -- Proof to be provided
  sorry
}

theorem general_formula (n : ℕ) (h : n > 0) :
  sequence n = real.sqrt n - real.sqrt (n - 1) :=
by {
  -- Proof to be provided
  sorry
}

end find_initial_terms_general_formula_l237_237148


namespace no_such_n_exists_l237_237046

theorem no_such_n_exists : ∀ (n : ℕ), n ≥ 1 → ¬ Prime (n^n - 4 * n + 3) :=
by
  intro n hn
  sorry

end no_such_n_exists_l237_237046


namespace effect_on_revenue_decrease_l237_237340

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q

def new_price (P : ℝ) : ℝ := P * 1.40

def new_quantity (Q : ℝ) : ℝ := Q * 0.65

def new_revenue (P Q : ℝ) : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue_decrease :
  new_revenue P Q = original_revenue P Q * 0.91 →
  new_revenue P Q - original_revenue P Q = original_revenue P Q * -0.09 :=
by
  sorry

end effect_on_revenue_decrease_l237_237340


namespace greatest_integer_third_side_l237_237667

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l237_237667


namespace find_f_prime_1_l237_237509

noncomputable def f (f_prime_1 : ℝ) (x : ℝ) := (1 / 3) * x^3 - f_prime_1 * x^2 - x

def derivative (f_prime_1 : ℝ) (x : ℝ) := x^2 - 2 * f_prime_1 * x - 1

theorem find_f_prime_1 : 
  ∃ f_prime_1 : ℝ, derivative f_prime_1 1 = f_prime_1 := 
begin
  use 0,
  unfold derivative,
  simp,
end

end find_f_prime_1_l237_237509


namespace unique_valid_triplets_l237_237850

-- Define the condition that a tuple (a, b, c) satisfies the required conditions
def is_valid_triplet (a b c : ℕ) : Prop :=
  a - b - 8 ∈ Prime ∧ b - c - 8 ∈ Prime ∧ a ∈ Prime ∧ b ∈ Prime ∧ c ∈ Prime

-- Define the valid triplets as (23, 13, 2) and (23, 13, 3)
def valid_triplet_1 : Prop := (23, 13, 2) = (23, 13, 2)
def valid_triplet_2 : Prop := (23, 13, 3) = (23, 13, 3)

-- The theorem to be proved: these are the only triplets satisfying the conditions
theorem unique_valid_triplets (a b c : ℕ) : is_valid_triplet a b c ↔ (valid_triplet_1 ∨ valid_triplet_2) :=
by
  sorry

end unique_valid_triplets_l237_237850


namespace range_of_a_l237_237480

noncomputable def f(x : Real) : Real := Real.exp(x) + (Real.exp(x))⁻¹

theorem range_of_a (a : Real) : 
  (∃ x0 : Real, 1 ≤ x0 ∧ f x0 < a * (-x0^3 + 3 * x0)) → 
  a > 0.5 * (Real.exp(1) + (Real.exp(1))⁻¹) :=
by
  intro h
  sorry

end range_of_a_l237_237480


namespace multinomial_theorem_l237_237047

theorem multinomial_theorem {α : Type*} [CommRing α] (x : ℕ → α) (n m : ℕ)
  (h : 2 ≤ m) :
  (∑ i in Finset.range m, x i)^n = 
  ∑ (k : Fin m → ℕ) (H : ∑ i, k i = n),
    (n.factorial / Finset.univ.prod (λ i, (k i).factorial)) *
    ∏ i, x i ^ k i :=
sorry

end multinomial_theorem_l237_237047


namespace evan_amount_l237_237854

def adrian : ℤ := sorry
def brenda : ℤ := sorry
def charlie : ℤ := sorry
def dana : ℤ := sorry
def evan : ℤ := sorry

def amounts_sum : Prop := adrian + brenda + charlie + dana + evan = 72
def abs_diff_1 : Prop := abs (adrian - brenda) = 21
def abs_diff_2 : Prop := abs (brenda - charlie) = 8
def abs_diff_3 : Prop := abs (charlie - dana) = 6
def abs_diff_4 : Prop := abs (dana - evan) = 5
def abs_diff_5 : Prop := abs (evan - adrian) = 14

theorem evan_amount
  (h_sum : amounts_sum)
  (h_diff1 : abs_diff_1)
  (h_diff2 : abs_diff_2)
  (h_diff3 : abs_diff_3)
  (h_diff4 : abs_diff_4)
  (h_diff5 : abs_diff_5) :
  evan = 21 := sorry

end evan_amount_l237_237854


namespace weight_left_after_two_deliveries_l237_237776

-- Definitions and conditions
def initial_load : ℝ := 50000
def first_store_percentage : ℝ := 0.1
def second_store_percentage : ℝ := 0.2

-- The statement to be proven
theorem weight_left_after_two_deliveries :
  let weight_after_first_store := initial_load * (1 - first_store_percentage)
  let weight_after_second_store := weight_after_first_store * (1 - second_store_percentage)
  weight_after_second_store = 36000 :=
by sorry  -- Proof omitted

end weight_left_after_two_deliveries_l237_237776


namespace number_subtracted_l237_237768

theorem number_subtracted (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 :=
by
  sorry

end number_subtracted_l237_237768


namespace abs_eq_15_sol_diff_l237_237436

noncomputable def positive_difference_between_solutions (a b : ℝ) : ℝ :=
if a > b then a - b else b - a

theorem abs_eq_15_sol_diff : 
  ∀ (x : ℝ), |x - 3| = 15 -> positive_difference_between_solutions 18 (-12) = 30 := 
by 
  intros x h,
  rw [positive_difference_between_solutions],
  norm_num,
  sorry

end abs_eq_15_sol_diff_l237_237436


namespace sum_divisors_45_l237_237314

theorem sum_divisors_45 : ∑ d in (45 : ℕ).divisors, d = 78 :=
by
  sorry

end sum_divisors_45_l237_237314


namespace efficiency_ratio_l237_237735

theorem efficiency_ratio (A B : ℝ) (h1 : A + B = 1 / 26) (h2 : B = 1 / 39) : A / B = 1 / 2 := 
by
  sorry

end efficiency_ratio_l237_237735


namespace find_pairs_l237_237431

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ a b, (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3))
  ↔ (∃ a b, a > 0 ∧ b > 0 ∧ (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0) := by
  sorry

end find_pairs_l237_237431


namespace probability_palindromic_division_by_7_is_1_over_20_l237_237005

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString 
  s = s.reverse

def is_5_digit_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s.length = 5 ∧ s = s.reverse

theorem probability_palindromic_division_by_7_is_1_over_20 :
  let palindromes := { n : ℕ | is_5_digit_palindrome n }
  let div_7_palindromes := { n : ℕ | is_5_digit_palindrome n ∧ is_palindrome (n / 7) }
  ∃ k, ∃ total_palindromes, k = 45 ∧ total_palindromes = 900 ∧ (set.card div_7_palindromes).toRat / (set.card palindromes).toRat = 1 / 20 :=
sorry

end probability_palindromic_division_by_7_is_1_over_20_l237_237005


namespace prove_value_l237_237472

def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem prove_value (α : ℝ) (h : 2^α = 16) : power_function α (Real.sqrt 3) = 9 := by
  sorry

end prove_value_l237_237472


namespace score_after_7_hours_l237_237959

theorem score_after_7_hours (score : ℕ) (time : ℕ) : 
  (score / time = 90 / 5) → time = 7 → score = 126 :=
by
  sorry

end score_after_7_hours_l237_237959


namespace equal_area_division_l237_237217

theorem equal_area_division (c : ℝ) :
  let total_area := 9
  let target_area := total_area / 2
  let line_eq : ℝ → ℝ := λ x, (3 / (3 - c)) * (x - c)
  let triangle_area := (1 / 2) * (3 - c) * 3
  in triangle_area = target_area → c = 0 :=
by
  let total_area := 9
  let target_area := total_area / 2
  let line_eq : ℝ → ℝ := λ x, (3 / (3 - c)) * (x - c)
  let triangle_area := (1 / 2) * (3 - c) * 3
  assume h : triangle_area = target_area
  have h1: triangle_area = 4.5, from h,
  have h2: (3 * (3 - c)) / 2 = 4.5, by exact h1,
  have h3: 3 * (3 - c) = 9, by linarith,
  have h4: 3 - c = 3, by linarith,
  have h5: c = 0, by linarith,
  exact h5
  sorry

end equal_area_division_l237_237217


namespace annie_extracurricular_hours_l237_237395

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end annie_extracurricular_hours_l237_237395


namespace no_integral_solutions_l237_237125

theorem no_integral_solutions : ∀ (x : ℤ), x^5 - 31 * x + 2015 ≠ 0 :=
by
  sorry

end no_integral_solutions_l237_237125


namespace unique_outfits_count_l237_237251

theorem unique_outfits_count (s t b : ℕ) (hs : s = 8) (ht : t = 6) (hb : b = 4) : s * t * b = 192 := by
  rw [hs, ht, hb]
  norm_num
  sorry

end unique_outfits_count_l237_237251


namespace smallest_zarks_l237_237534

theorem smallest_zarks (n : ℕ) : (n^2 > 15 * n) → (n ≥ 16) := sorry

end smallest_zarks_l237_237534


namespace phones_left_l237_237373

theorem phones_left (last_year_production : ℕ) 
                    (this_year_production : ℕ) 
                    (sold_phones : ℕ) 
                    (left_phones : ℕ) 
                    (h1 : last_year_production = 5000) 
                    (h2 : this_year_production = 2 * last_year_production) 
                    (h3 : sold_phones = this_year_production / 4) 
                    (h4 : left_phones = this_year_production - sold_phones) : 
                    left_phones = 7500 :=
by
  rw [h1, h2]
  simp only
  rw [h3, h4]
  norm_num
  sorry

end phones_left_l237_237373


namespace angle_B_leq_60_l237_237986

-- Given three side lengths of triangle ABC that form a geometric progression
variable {a b c : ℝ} -- sides of the triangle
variable (B : ℝ) -- Angle B in the triangle
variable (r : ℝ) -- positive real number such that a = br and c = b/r
variable [fact (triangle a b c)] -- Given it's a triangle, embedding triangle inequality
variable (h1 : a = b * r) -- condition for geometric progression
variable (h2 : c = b / r) -- condition for geometric progression

theorem angle_B_leq_60 (h3 : 2 * cos B = r^2 + 1/r^2 - 1) : B ≤ 60 :=
sorry

end angle_B_leq_60_l237_237986


namespace length_of_AC_l237_237533

theorem length_of_AC 
  (A B C D E : Point)
  (h1: distance A B = 15)
  (h2: distance D C = 24)
  (h3: distance A D = 9) :
  distance A C = 29.5 :=
  sorry

end length_of_AC_l237_237533


namespace triangle_altitudes_l237_237190

namespace TriangleAngles

variables {α : Type*} [LinearOrderedSemiring α] 

structure Triangle (α : Type*) :=
  (A B C : α)

def ortho_center (ABC : Triangle α) : α := sorry
def altitude_foot (P : α) (ABC : Triangle α) : α := sorry

theorem triangle_altitudes (ABC : Triangle α) (H := ortho_center ABC) 
  (H_A := altitude_foot ABC.A ABC) (H_B := altitude_foot ABC.B ABC) (H_C := altitude_foot ABC.C ABC) 
  (angle_A angle_B angle_C : α) 
  (angle_A_def : angle_A = 180 - angle_B - angle_C)
  (angle_A_acute : angle_A < 90) (angle_B_acute : angle_B < 90) (angle_C_acute : angle_C < 90) :
  
  (angle_at H_A H_B H_C = 180 - 2 * angle_B) ∧
  (angle_at H_B H_C H_A = 180 - 2 * angle_C) ∧
  (angle_at H_C H_A H_B = 180 - 2 * angle_A) ∧

  (angle_at H_B H H_C = angle_A) ∧
  (angle_at H_A H H_C = angle_B) ∧
  (angle_at H_A H H_B = angle_C) ∧
  (angle_at A H_B H_C = angle_C) ∧
  (angle_at A H_C H_B = angle_B) ∧
  (angle_at B H_A H_C = angle_C) ∧
  (angle_at B H_C H_A = angle_A) ∧
  (angle_at C H_A H_B = angle_B) ∧
  (angle_at C H_B H_A = angle_A) := sorry

end TriangleAngles

end triangle_altitudes_l237_237190


namespace range_of_a_l237_237132

def f (a x : ℝ) := (1 / 2) * cos (2 * x) + 3 * a * (sin x - cos x) + (4 * a - 1) * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -π/2 ≤ x ∧ x ≤ 0 → 0 ≤ (-sin (2 * x) + 3 * a * (cos x + sin x) + 4 * a - 1)) →
  1 ≤ a :=
sorry

end range_of_a_l237_237132


namespace triangle_third_side_l237_237686

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l237_237686


namespace trapezoid_area_l237_237164

theorem trapezoid_area
  (AD BC AC BD : ℝ)
  (h1 : AD = 24)
  (h2 : BC = 8)
  (h3 : AC = 13)
  (h4 : BD = 5 * Real.sqrt 17) : 
  ∃ (area : ℝ), area = 80 :=
by
  let area := (1 / 2) * (AD + BC) * 5
  existsi area
  sorry

end trapezoid_area_l237_237164


namespace sandys_average_price_l237_237605

noncomputable def average_price_per_book (priceA : ℝ) (discountA : ℝ) (booksA : ℕ) (priceB : ℝ) (discountB : ℝ) (booksB : ℕ) (conversion_rate : ℝ) : ℝ :=
  let costA := priceA / (1 - discountA)
  let priceB_in_usd := priceB / conversion_rate
  let costB := priceB_in_usd / (1 - discountB)
  let total_cost := costA + costB
  let total_books := booksA + booksB
  total_cost / total_books

theorem sandys_average_price :
  average_price_per_book 1380 0.15 65 900 0.10 55 0.85 = 23.33 :=
by
  sorry

end sandys_average_price_l237_237605


namespace parallel_lines_through_point_l237_237634

theorem parallel_lines_through_point (l : Line) (P : Point) : 
  (number_of_parallel_lines_through_point l P = 0 ∨ number_of_parallel_lines_through_point l P = 1) :=
sorry

end parallel_lines_through_point_l237_237634


namespace problem1_problem2_l237_237407

-- Problem 1
theorem problem1 (α : ℝ) (h1 : Real.tan α = 2) :
  Real.sin α * (Real.sin α + Real.cos α) = 6 / 5 :=
sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h2 : 5^a = 10) (h3 : 4^b = 10) :
  (2 / a) + (1 / b) = 2 :=
sorry

end problem1_problem2_l237_237407


namespace harold_wrapping_cost_l237_237911

noncomputable def wrapping_paper_rolls_cost (cost_per_roll : ℕ) (rolls_needed : ℕ) : ℕ := cost_per_roll * rolls_needed

noncomputable def total_paper_rolls (shirt_boxes : ℕ) (shirt_boxes_per_roll : ℕ) (xl_boxes : ℕ) (xl_boxes_per_roll : ℕ) : ℕ :=
  (shirt_boxes / shirt_boxes_per_roll) + (xl_boxes / xl_boxes_per_roll)

theorem harold_wrapping_cost 
  (cost_per_roll : ℕ) (shirt_boxes : ℕ) (shirt_boxes_per_roll : ℕ) (xl_boxes : ℕ) (xl_boxes_per_roll : ℕ) :
  shirt_boxes = 20 → shirt_boxes_per_roll = 5 → xl_boxes = 12 → xl_boxes_per_roll = 3 → cost_per_roll = 4 → 
  wrapping_paper_rolls_cost cost_per_roll (total_paper_rolls shirt_boxes shirt_boxes_per_roll xl_boxes xl_boxes_per_roll) = 32 :=
by
  intros hshirt_boxes hshirt_boxes_per_roll hxl_boxes hxl_boxes_per_roll hcost_per_roll
  simp [wrapping_paper_rolls_cost, total_paper_rolls, hshirt_boxes, hshirt_boxes_per_roll, hxl_boxes, hxl_boxes_per_roll, hcost_per_roll]
  rfl

end harold_wrapping_cost_l237_237911


namespace sum_last_two_digits_l237_237322

theorem sum_last_two_digits (h1 : 9 ^ 23 ≡ a [MOD 100]) (h2 : 11 ^ 23 ≡ b [MOD 100]) :
  (a + b) % 100 = 60 := 
  sorry

end sum_last_two_digits_l237_237322


namespace probability_even_product_l237_237010

-- Assuming definitions for convenience
noncomputable def tetrahedron_faces := {1, 2, 3, 4}

-- Two independent throws modeled as list of pairs of numbers
def outcomes (faces : set ℕ) : list (ℕ × ℕ) :=
  (faces ×ˢ faces).toList

-- Function to check if a product of a pair is even
def is_even_product (x : ℕ × ℕ) : Prop :=
  (x.1 * x.2) % 2 = 0

-- The main statement
theorem probability_even_product (faces : set ℕ) (h : faces = tetrahedron_faces) :
  (finset.filter is_even_product (finset.product (finset.of_set faces) (finset.of_set faces))).card.toR / (finset.card (finset.product (finset.of_set faces) (finset.of_set faces))).toR = 3 / 4 := by
  sorry

end probability_even_product_l237_237010


namespace total_number_of_legs_l237_237210

def kangaroos : ℕ := 23
def goats : ℕ := 3 * kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_number_of_legs : 
  (kangaroos * legs_of_kangaroo + goats * legs_of_goat) = 322 := by
  sorry

end total_number_of_legs_l237_237210


namespace greatest_third_side_l237_237696

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l237_237696


namespace quadratic_complete_square_l237_237637
#import necessary libraries

theorem quadratic_complete_square:
  ∃ b c: ℝ, (∀ x: ℝ, (x ^ 2 - 40 * x + 121) = (x + b) ^ 2 + c) ∧ b + c = -299 :=
begin
  sorry
end

end quadratic_complete_square_l237_237637


namespace greatest_third_side_l237_237674

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l237_237674


namespace find_variance_of_sample_l237_237868

variable (a b : ℝ)
variable  (xs : List ℝ)

# The given sample
def sample := [a, 3, 5, 7]

# Mean of the sample
def mean (l : List ℝ) := l.sum / l.length

# Polynomial condition
def poly_condition (p q : ℝ) : Prop :=
  p*q = 4 ∧ p + q = 5

# Variance of the sample
def variance (l : List ℝ) [Nonempty l] : ℝ := 
  let avg := mean l
  (l.map (λ x => (x - avg) ^ 2)).sum / (l.length - 1)

theorem find_variance_of_sample
  (h1 : mean sample = b)
  (h2 : poly_condition a b)
  : variance sample = 5 := 
by 
  sorry

end find_variance_of_sample_l237_237868


namespace new_job_hourly_wage_l237_237207

def current_job_weekly_earnings : ℝ := 8 * 10
def new_job_hours_per_week : ℝ := 4
def new_job_bonus : ℝ := 35
def new_job_expected_additional_wage : ℝ := 15

theorem new_job_hourly_wage (W : ℝ) 
  (h_current_job : current_job_weekly_earnings = 80)
  (h_new_job : new_job_hours_per_week * W + new_job_bonus = current_job_weekly_earnings + new_job_expected_additional_wage) : 
  W = 15 :=
by 
  sorry

end new_job_hourly_wage_l237_237207


namespace magnitude_of_z_l237_237948

noncomputable def z : ℂ := 1 + 2 * complex.i + complex.i ^ 3

theorem magnitude_of_z : complex.abs z = real.sqrt 2 := 
by
  -- this 'sorry' is a placeholder for the actual proof
  sorry

end magnitude_of_z_l237_237948


namespace lisa_rem_quiz_max_lower_than_a_l237_237402

noncomputable def lisa_goal_met (total_quizzes completed_scores goal_percentage : ℕ) (completed_a_scores : ℕ) :
  ℕ :=
  let remaining_quizzes := total_quizzes - (completed_scores)
  let required_total_a_scores := goal_percentage * total_quizzes / 100
  let remaining_a_needed := required_total_a_scores - completed_a_scores
  remaining_quizzes - remaining_a_needed

theorem lisa_rem_quiz_max_lower_than_a (total_quizzes completed_scores goal_percentage : ℕ) (completed_a_scores : ℕ) :
  total_quizzes = 50 ∧ completed_scores = 30 ∧ goal_percentage = 80 ∧ completed_a_scores = 22 →
  lisa_goal_met total_quizzes completed_scores goal_percentage completed_a_scores = 2 :=
by intros h; 
   cases h with h_total rest;
   cases rest with h_completed_scores rest;
   cases rest with h_goal_percentage h_completed_a_scores;
   rw [h_total, h_completed_scores, h_goal_percentage, h_completed_a_scores];
   dsimp [lisa_goal_met];
   sorry

end lisa_rem_quiz_max_lower_than_a_l237_237402


namespace minimum_distance_AB_l237_237261

noncomputable def distance_AB_min_value (a : ℝ) : ℝ :=
let x1 := (2 * exp a + a - 3) / 3,
    x2 := exp a in
| x2 - x1 |

theorem minimum_distance_AB (a : ℝ) : distance_AB_min_value a = 4 / 3 :=
sorry

end minimum_distance_AB_l237_237261


namespace pow_mod_l237_237710

theorem pow_mod (a n m : ℕ) (h : a % m = 1) : (a ^ n) % m = 1 := by
  sorry

example : (11 ^ 2023) % 5 = 1 := by
  apply pow_mod 11 2023 5
  norm_num
  have : 11 % 5 = 1 := by norm_num
  exact this

end pow_mod_l237_237710


namespace minimum_spent_on_orange_juice_l237_237419

theorem minimum_spent_on_orange_juice
  (price_per_bottle : ℝ)
  (price_per_pack : ℝ)
  (bottles_per_pack : ℕ)
  (total_bottles_needed : ℕ)
  (pack_cost : ℝ)
  (individual_cost : ℝ) :
  price_per_bottle = 2.80 →
  price_per_pack = 15.00 →
  bottles_per_pack = 6 →
  total_bottles_needed = 22 →
  let packs_needed := total_bottles_needed / bottles_per_pack in
  let remaining_bottles := total_bottles_needed % bottles_per_pack in
  let cost_from_packs := packs_needed * price_per_pack in
  let cost_from_individual := remaining_bottles * price_per_bottle in
  let total_cost := cost_from_packs + cost_from_individual in
  total_cost = 56.20 :=
sorry

end minimum_spent_on_orange_juice_l237_237419


namespace four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l237_237048

noncomputable def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

noncomputable def sum_of_digits_is_11 (n : ℕ) : Prop := 
  let d1 := n / 1000
  let r1 := n % 1000
  let d2 := r1 / 100
  let r2 := r1 % 100
  let d3 := r2 / 10
  let d4 := r2 % 10
  d1 + d2 + d3 + d4 = 11

theorem four_digit_numbers_divisible_by_11_with_sum_of_digits_11
  (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : is_divisible_by_11 n)
  (h3 : sum_of_digits_is_11 n) : 
  n = 2090 ∨ n = 3080 ∨ n = 4070 ∨ n = 5060 ∨ n = 6050 ∨ n = 7040 ∨ n = 8030 ∨ n = 9020 :=
sorry

end four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l237_237048


namespace selling_price_is_320_l237_237378

noncomputable def sales_volume (x : ℝ) : ℝ := 8000 / x

def cost_price : ℝ := 180

def desired_profit : ℝ := 3500

def selling_price_for_desired_profit (x : ℝ) : Prop :=
  (x - cost_price) * sales_volume x = desired_profit

/-- The selling price of the small electrical appliance to achieve a daily sales profit 
    of $3500 dollars is $320 dollars. -/
theorem selling_price_is_320 : selling_price_for_desired_profit 320 :=
by
  -- We skip the proof as per instructions
  sorry

end selling_price_is_320_l237_237378


namespace train_speed_l237_237008

theorem train_speed (len_train len_bridge time : ℝ) (h_len_train : len_train = 120)
  (h_len_bridge : len_bridge = 150) (h_time : time = 26.997840172786177) :
  let total_distance := len_train + len_bridge
  let speed_m_s := total_distance / time
  let speed_km_h := speed_m_s * 3.6
  speed_km_h = 36 :=
by
  -- Proof goes here
  sorry

end train_speed_l237_237008


namespace concurrency_and_collinearity_of_altitudes_in_triangle_PQR_l237_237557

-- Definitions of points and triangle
variables (A B C P Q R L M N : Type)

noncomputable def incircle_contact_points (triangle : Type) : Type :=
  sorry -- Define the points P, Q, R as the incircle contact points with sides

noncomputable def feet_of_altitudes (triangle : Type) : Type :=
  sorry -- Define the points L, M, N as the feet of the altitudes

def concurrent_lines (lines : List Type) : Type :=
  sorry -- True if the given lines are concurrent

def lies_on_euler_line (point : Type) (triangle : Type) : Prop :=
  sorry -- True if the point lies on the Euler line of the given triangle

theorem concurrency_and_collinearity_of_altitudes_in_triangle_PQR
    (A B C P Q R L M N : Type)
    [incircle_contact_points (triangle A B C)] 
    [feet_of_altitudes (triangle P Q R)] :
  (concurrent_lines [line A N, line B L, line C M]) ∧
  (lies_on_euler_line (concurrency_point [line A N, line B L, line C M]) (triangle P Q R)) :=
  by sorry

end concurrency_and_collinearity_of_altitudes_in_triangle_PQR_l237_237557


namespace lion_to_leopard_ratio_l237_237146

variable (L P E : ℕ)

axiom lion_count : L = 200
axiom total_population : L + P + E = 450
axiom elephants_relation : E = (1 / 2 : ℚ) * (L + P)

theorem lion_to_leopard_ratio : L / P = 2 :=
by
  sorry

end lion_to_leopard_ratio_l237_237146


namespace alex_mean_score_l237_237859

theorem alex_mean_score 
  (scores : List ℕ)
  (n : ℕ)
  (h_scores : scores = [86, 88, 90, 92, 95, 97, 99])
  (h_len : scores.length = 7)
  (jordan_mean : ℕ)
  (h_jordan_mean : jordan_mean = 92)
  (jordan_scores_count : ℕ)
  (h_jordan_scores_count : jordan_scores_count = 4) :
  let
    total_sum := scores.sum,
    jordan_total := jordan_mean * jordan_scores_count,
    alex_total := total_sum - jordan_total,
    alex_scores_count := scores.length - jordan_scores_count
  in
    alex_total / alex_scores_count = 93 :=
by
  sorry

end alex_mean_score_l237_237859


namespace min_moves_equal_stacks_l237_237281

theorem min_moves_equal_stacks :
  let a1 := 9
  let a2 := 7
  let a3 := 5
  let a4 := 10
  let total_coins := a1 + a2 + a3 + a4
  let N := 11
  ∃ k, (4 * k = total_coins + 3 * N) 
  : N = 11 :=
by
  let a1 := 9
  let a2 := 7
  let a3 := 5
  let a4 := 10
  let total_coins := a1 + a2 + a3 + a4
  have : total_coins = 31 := by norm_num
  have : 31 + 3 * 11 = 4 * ((31 + 3 * 11) / 4) := by norm_num
  existsi (31 + 3 * 11) / 4
  sorry

end min_moves_equal_stacks_l237_237281


namespace coefficient_of_x3y3_in_expansion_equals_40_l237_237822

noncomputable theory

def coefficient_x3y3_in_expansion : ℤ :=
  let T_r (r : ℕ) := binomial 5 r * 2^(5-r) * (-1)^r in
  let term1 := T_r 3 * 1 - 40 in
  let term2 := T_r 2 * 1 * 80 in
  term1 + term2

theorem coefficient_of_x3y3_in_expansion_equals_40 :
  coefficient_x3y3_in_expansion = 40 := by
  sorry

end coefficient_of_x3y3_in_expansion_equals_40_l237_237822


namespace range_of_a_l237_237486

-- Given definition of the function
def f (x a : ℝ) := abs (x - a)

-- Statement of the problem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < -1 → x₂ < -1 → f x₁ a ≤ f x₂ a) → a ≥ -1 :=
by
  sorry

end range_of_a_l237_237486


namespace total_legs_of_all_animals_l237_237211

def num_kangaroos : ℕ := 23
def num_goats : ℕ := 3 * num_kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_legs_of_all_animals : num_kangaroos * legs_of_kangaroo + num_goats * legs_of_goat = 322 :=
by 
  sorry

end total_legs_of_all_animals_l237_237211


namespace problem_example_l237_237494

def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_digits_different (n : ℕ) : Prop :=
  let digits := (n.to_digits 10).nub;
  digits.length = (n.to_digits 10).length

def non_zero_leading_digit (n : ℕ) : Prop :=
  let digits := n.to_digits 10;
  digits.rev.head ≠ 0

def multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def largest_digit_is_8 (n : ℕ) : Prop :=
  let digits := (n.to_digits 10).nub;
  8 ∈ digits ∧ ∀ d ∈ digits, d ≤ 8

theorem problem_example :
  {n : ℕ | is_4_digit n ∧ all_digits_different n ∧ non_zero_leading_digit n ∧ multiple_of_5 n ∧ largest_digit_is_8 n}.card = 408 :=
sorry

end problem_example_l237_237494


namespace greatest_root_of_polynomial_l237_237052

theorem greatest_root_of_polynomial :
  ∃ (x : ℝ), (f : ℝ → ℝ) (hx : f x = 0) (hmax: ∀ y, f y = 0 → y ≤ x), 
    f x = 16 * x^4 - 8 * x^3 + 9 * x^2 - 3 * x + 1 ∧ x = 0.5 := by
  sorry

end greatest_root_of_polynomial_l237_237052


namespace distinct_prime_factors_of_120_l237_237498

theorem distinct_prime_factors_of_120 : ∃ p1 p2 p3 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ (120 = p1 * p2 * p3 * (p1 + p2 + p3)) := 
sorry

end distinct_prime_factors_of_120_l237_237498


namespace contradiction_assumption_l237_237719

-- Proposition P: "Among a, b, c, d, at least one is negative"
def P (a b c d : ℝ) : Prop :=
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0

-- Correct assumption when using contradiction: all are non-negative
def notP (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof problem statement: assuming notP leads to contradiction to prove P
theorem contradiction_assumption (a b c d : ℝ) (h : ¬ P a b c d) : notP a b c d :=
by
  sorry

end contradiction_assumption_l237_237719


namespace winning_candidate_percentage_l237_237286

theorem winning_candidate_percentage
  (votes1 votes2 votes3 : ℕ)
  (h1 : votes1 = 3000)
  (h2 : votes2 = 5000)
  (h3 : votes3 = 20000) :
  ((votes3 : ℝ) / (votes1 + votes2 + votes3) * 100) = 71.43 := by
  sorry

end winning_candidate_percentage_l237_237286


namespace determine_a_l237_237903

-- Define the sets A and B
def A : Set ℝ := { -1, 0, 2 }
def B (a : ℝ) : Set ℝ := { 2^a }

-- State the main theorem
theorem determine_a (a : ℝ) (h : B a ⊆ A) : a = 1 :=
by
  sorry

end determine_a_l237_237903


namespace triangular_prism_properties_l237_237757

theorem triangular_prism_properties :
  ∀ (prism : Type) (h : is_triangular_prism prism), 
    num_faces prism = 5 ∧ num_edges prism = 9 ∧ num_lateral_edges prism = 3 ∧ num_vertices prism = 6 :=
by
  sorry

end triangular_prism_properties_l237_237757


namespace largest_prime_factor_of_9871_l237_237846

theorem largest_prime_factor_of_9871 : ∃ p, prime p ∧ p ∈ {71, 3, 43} ∧ ∀ q ∈ {71, 3, 43}, q ≤ p :=
by 
  have h : 9871 = 71 * 129 := sorry
  have h1 : 129 = 3 * 43 := sorry
  have h2 : prime 71 := sorry
  have h3 : prime 3 := sorry
  have h4 : prime 43 := sorry
  existsi (71 : ℕ)
  split
  sorry
  split
  sorry
  intro q hq
  cases hq
  sorry
  cases hq
  sorry
  sorry

end largest_prime_factor_of_9871_l237_237846


namespace find_magnitudes_product_l237_237415

def ellipse_eq (x y : ℝ) := (x^2 / 16) + (y^2 / 12) = 1

def dot_product_condition (PF1 PF2 : ℝ) := PF1 * PF2 * real.cos θ = 9

def magnitudes_product (PF1 PF2 : ℝ) := PF1 * PF2 = 15

theorem find_magnitudes_product (x y PF1 PF2 : ℝ) (θ : ℝ) 
(h1 : ellipse_eq x y) 
(h2 : dot_product_condition PF1 PF2) 
(h3 : PF1 + PF2 = 8) 
(h4 : |PF1 - PF2| = 4) : 
magnitudes_product PF1 PF2 :=
begin
    sorry
end

end find_magnitudes_product_l237_237415


namespace repeating_decimal_sum_is_one_l237_237428

noncomputable def repeating_decimal_sum : ℝ :=
  let x := (1/3 : ℝ)
  let y := (2/3 : ℝ)
  x + y

theorem repeating_decimal_sum_is_one : repeating_decimal_sum = 1 := by
  sorry

end repeating_decimal_sum_is_one_l237_237428


namespace total_hours_eq_52_l237_237391

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l237_237391


namespace digit_after_decimal_l237_237702

theorem digit_after_decimal (n : ℕ) (hn : n = 35)
  (h1 : (1 : ℝ) / 9 = 0.111111111111111111111111111111111... )
  (h2 : (1 : ℝ) / 5 = 0.222222222222222222222222222222222... ) :
  digit_of_decimal (decimal_sum (1 / 9 : ℝ) (1 / 5 : ℝ)) 35 = 3 :=
sorry

end digit_after_decimal_l237_237702


namespace total_balls_l237_237810

theorem total_balls (blue red green yellow purple orange black white : ℕ) 
  (h1 : blue = 8)
  (h2 : red = 5)
  (h3 : green = 3 * (2 * blue - 1))
  (h4 : yellow = Nat.floor (2 * Real.sqrt (red * blue)))
  (h5 : purple = 4 * (blue + green))
  (h6 : orange = 7)
  (h7 : black + white = blue + red + green + yellow + purple + orange)
  (h8 : blue + red + green + yellow + purple + orange + black + white = 3 * (red + green + yellow + purple) + orange / 2)
  : blue + red + green + yellow + purple + orange + black + white = 829 :=
by
  sorry

end total_balls_l237_237810


namespace concurrency_of_lines_l237_237795

theorem concurrency_of_lines
  (A B C A1 B1 C1 : Type)
  (h_triangle : is_triangle (A, B, C))
  (h_similar_isosceles_AC1B : is_similar_isosceles (A, C1, B))
  (h_similar_isosceles_BA1C : is_similar_isosceles (B, A1, C))
  (h_similar_isosceles_CB1A : is_similar_isosceles (C, B1, A))
  (h_angles : ∀ α β γ, ∠ A B1 C = α ∧
                       ∠ A B C1 = α ∧
                       ∠ A1 B C = α ∧
                       ∠ B A1 C = β ∧
                       ∠ B A C1 = β ∧
                       ∠ B1 A C = β) :
  concurrency (AA1 : line) (BB1 : line) (CC1 : line) :=
begin
  sorry
end

end concurrency_of_lines_l237_237795


namespace number_of_primes_diffs_in_set_l237_237914

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def candidates : List ℕ := List.range' 7 10 4

def num_of_numbers_can_be_written_as_diff_of_two_primes : ℕ :=
  candidates.countp (λ x => is_prime (x + 2))

theorem number_of_primes_diffs_in_set : num_of_numbers_can_be_written_as_diff_of_two_primes = 2 := 
sorry

end number_of_primes_diffs_in_set_l237_237914


namespace sum_inequality_l237_237601

theorem sum_inequality (n k : ℕ) :
  1^k + 2^k + ... + n^k ≤ (n^2k - (n-1)^k) / (n^k - (n-1)^k) :=
sorry

end sum_inequality_l237_237601


namespace part1_max_traffic_flow_part2_traffic_flow_exceeds_10_l237_237782

def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

theorem part1_max_traffic_flow :
  ∃ (v : ℝ), v > 0 ∧ traffic_flow v = 920 / 83 :=
sorry

theorem part2_traffic_flow_exceeds_10 :
  ∃ (v : ℝ), 25 < v ∧ v < 64 ∧ traffic_flow v > 10 :=
sorry

end part1_max_traffic_flow_part2_traffic_flow_exceeds_10_l237_237782


namespace sum_reciprocals_less_than_80_l237_237186

-- The set M of all positive integers that do not contain the digit 9
def M : Set ℕ := {x | ∀ d ∈ (Nat.digits 10 x), d ≠ 9}

-- Property stating the necessary condition about x_j's being distinct members of M
theorem sum_reciprocals_less_than_80 (n : ℕ) (x : Fin n → ℕ)
  (h_distinct : Function.Injective x) (h_in_M : ∀ i, x i ∈ M) :
  ∑ i, (1 / (x i : ℝ)) < 80 := 
sorry  -- proof is not required

end sum_reciprocals_less_than_80_l237_237186


namespace monotonicity_intervals_minimum_a_l237_237478

open Real

noncomputable def f (x a : ℝ) : ℝ := ln x + a / x - x + 1 - a

theorem monotonicity_intervals (a : ℝ) :
  (∀ x ∈ Ioo 0 (1 + sqrt (1 - 4 * a)) / 2, f x a ≤ f x a) ∧
  (∀ x ∈ Ioo (1 + sqrt (1 - 4 * a)) / 2 ∞, f x a ≥ f x a) ∨
  (∀ x ∈ Ioo (1 - sqrt (1 - 4 * a)) / 2 (1 + sqrt (1 - 4 * a)) / 2, f x a ≤ f x a) ∧
  ((∀ x ∈ Ioo 0 (1 - sqrt (1 - 4 * a)) / 2 ∨ ∀ x ∈ Ioo (1 + sqrt (1 - 4 * a)) / 2 ∞), f x a ≥ f x a) ∨
  (a ≥ 1 / 4 → ∀ x > 0, f x a ≥ f x a) :=
sorry

theorem minimum_a (h : ∃ x > 1, f x a + x < (1 - x) / x) : a = 5 :=
sorry

end monotonicity_intervals_minimum_a_l237_237478


namespace shirts_production_l237_237232

-- Definitions
def constant_rate (r : ℕ) : Prop := ∀ n : ℕ, 8 * n * r = 160 * n

theorem shirts_production (r : ℕ) (h : constant_rate r) : 16 * r = 32 :=
by sorry

end shirts_production_l237_237232


namespace total_factories_checked_l237_237287

theorem total_factories_checked
  (first_group : ℕ) (second_group : ℕ) (unchecked : ℕ)
  (h1 : first_group = 69)
  (h2 : second_group = 52)
  (h3 : unchecked = 48) : 
  first_group + second_group + unchecked = 169 :=
by
  rw [h1, h2, h3]
  rfl

end total_factories_checked_l237_237287


namespace minimum_value_of_f_at_zero_inequality_f_geq_term_l237_237893

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem minimum_value_of_f_at_zero (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≥ f a x ∧ f a x = 0) → a = 2 :=
by
  sorry

theorem inequality_f_geq_term (x : ℝ) (hx : x > 1) : 
  f 2 x ≥ 1 / x - Real.exp (1 - x) :=
by
  sorry

end minimum_value_of_f_at_zero_inequality_f_geq_term_l237_237893


namespace time_per_room_l237_237753

theorem time_per_room (R P T: ℕ) (h: ℕ) (h₁ : R = 11) (h₂ : P = 2) (h₃ : T = 63) (h₄ : h = T / (R - P)) : h = 7 :=
by
  sorry

end time_per_room_l237_237753


namespace calculate_omega_varphi_l237_237091

theorem calculate_omega_varphi : 
  ∃ (ω φ : ℝ), (ω > 0) 
    ∧ (cos (ω * (-π / 6) + φ) = 0) 
    ∧ (cos (ω * (5 * π / 6) + φ) = 0) 
    ∧ (∀ x1 x2 : ℝ, x1 < x2 ∧ x1 > (-π / 6) ∧ x2 < (5 * π / 6) → x2 - x1 = π → x2 = x1 + π)
    ∧ ω * φ = (5 * π / 3) := 
by 
  sorry

end calculate_omega_varphi_l237_237091


namespace distance_origin_to_line_l237_237621

def distance_from_origin_to_line (a b c : ℝ) : ℝ :=
  |a * 0 + b * 0 + c| / (Real.sqrt (a^2 + b^2))

theorem distance_origin_to_line : 
  distance_from_origin_to_line 1 2 (-5) = Real.sqrt 5 := by 
  sorry

end distance_origin_to_line_l237_237621


namespace sum_of_divisors_of_45_l237_237320

theorem sum_of_divisors_of_45 :
  let divisors := [1, 3, 9, 5, 15, 45] in
  list.sum divisors = 78 :=
by
  -- this is where the proof would go
  sorry


end sum_of_divisors_of_45_l237_237320


namespace parabola_trajectory_l237_237750

-- Defining the problem conditions
def F : Point := ⟨0, -4⟩
def line : Line := ⟨0, 3⟩ -- line is y = 3

-- Define the concept of distance from a point to another point
def distance_to_point (P Q : Point) : ℝ :=
sorry -- implementation of distance formula

-- Define the concept of distance from a point to a line
def distance_to_line (P : Point) (L : Line) : ℝ :=
sorry -- implementation of point to line distance formula

-- Condition is that distance to point F is 1 more than its distance to the line y - 3 = 0
def condition (P : Point) : Prop :=
distance_to_point P F = distance_to_line P line + 1

-- Statement to prove the trajectory equation
theorem parabola_trajectory : ∀ (P : Point), condition P → (P.x^2 = -16 * P.y) :=
by
  sorry

end parabola_trajectory_l237_237750


namespace quadratic_root_range_quadratic_product_of_roots_l237_237635

-- Problem (1): Prove the range of m.
theorem quadratic_root_range (m : ℝ) :
  (∀ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 → x1 ≠ x2) ↔ m < 1 := 
sorry

-- Problem (2): Prove the existence of m such that x1 * x2 = 0.
theorem quadratic_product_of_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 ∧ x1 * x2 = 0) ↔ m = -1 := 
sorry

end quadratic_root_range_quadratic_product_of_roots_l237_237635


namespace rhombus_perimeter_l237_237618

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 14) (h_d2 : d2 = 48) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side = 100 :=
by
  sorry

end rhombus_perimeter_l237_237618


namespace monotonic_increasing_interval_l237_237632

def f (x : ℝ) : ℝ := log (1/2 : ℝ) (-x^2 + 4*x - 3)

theorem monotonic_increasing_interval :
  monotonic_increasing_on f (set.Ico 2 3) :=
begin
  sorry
end

end monotonic_increasing_interval_l237_237632


namespace not_p_and_q_equiv_not_p_or_not_q_l237_237134

variable (p q : Prop)

theorem not_p_and_q_equiv_not_p_or_not_q (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end not_p_and_q_equiv_not_p_or_not_q_l237_237134


namespace keys_missing_l237_237990

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end keys_missing_l237_237990


namespace magnitude_of_z_l237_237953

-- Define the complex number z and the condition
def z : ℂ := 1 + 2 * Complex.i + Complex.i ^ 3

-- The main theorem stating the magnitude of z
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l237_237953


namespace distinct_prime_factors_of_120_l237_237499

theorem distinct_prime_factors_of_120 : ∃ p1 p2 p3 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ (120 = p1 * p2 * p3 * (p1 + p2 + p3)) := 
sorry

end distinct_prime_factors_of_120_l237_237499


namespace complement_A_is_01_l237_237136

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A given the conditions
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}

-- State the theorem: complement of A is the interval [0, 1)
theorem complement_A_is_01 : Set.compl A = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end complement_A_is_01_l237_237136


namespace center_of_symmetry_monotonic_increasing_l237_237243

def f (x : ℝ) := 2 * cos x ^ 2 + 2 * (real.sqrt 3) * sin x * cos x - 1

theorem center_of_symmetry :
  f (-π / 12) = 0 :=
sorry

theorem monotonic_increasing :
  ∀ x y : ℝ, -π / 6 < x → x < y → y < π / 6 → f x < f y :=
sorry

end center_of_symmetry_monotonic_increasing_l237_237243


namespace tens_digit_6_pow_45_l237_237056

theorem tens_digit_6_pow_45 : (6 ^ 45 % 100) / 10 = 0 := 
by 
  sorry

end tens_digit_6_pow_45_l237_237056


namespace trapezoid_area_l237_237985

theorem trapezoid_area
  (A B C D : ℝ)
  (BC AD AC : ℝ)
  (radius circle_center : ℝ)
  (h : ℝ)
  (angleBAD angleADC : ℝ)
  (tangency : Bool) :
  BC = 13 → 
  angleBAD = 2 * angleADC →
  radius = 5 →
  tangency = true →
  1/2 * (BC + AD) * h = 157.5 :=
by
  sorry

end trapezoid_area_l237_237985


namespace annie_extracurricular_hours_l237_237397

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end annie_extracurricular_hours_l237_237397


namespace part1_part2_l237_237869

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     := 1
| (n+1) := sorry    -- placeholder for a_{n+1} = c - 1 / a_n

-- Define b_n for c = 5/2 and b_n = 1 / (a_n - 2)
def b (n : ℕ) : ℝ := 1 / (a n - 2)

-- Part (I): Prove the general formula for the sequence {b_n}
theorem part1 (n : ℕ) : b n = -1 / 3 * (4 ^ (n-1) + 2) :=
sorry

-- Parameters
variable {c : ℝ}
-- Define the sequence a_n for any c
def a_general : ℕ → ℝ
| 0     := 1
| (n+1) := c - 1 / (a_general n)

-- Part (II): Prove the range for c such that a_n < a_{n+1} < 3
theorem part2 (hc : 2 < c ∧ c ≤ 10 / 3) (n : ℕ) : 
  a_general n < a_general (n + 1) ∧ a_general (n + 1) < 3 :=
sorry

end part1_part2_l237_237869


namespace sum_of_digits_8_pow_2003_l237_237716

noncomputable def units_digit (n : ℕ) : ℕ :=
n % 10

noncomputable def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

noncomputable def sum_of_tens_and_units_digits (n : ℕ) : ℕ :=
units_digit n + tens_digit n

theorem sum_of_digits_8_pow_2003 :
  sum_of_tens_and_units_digits (8 ^ 2003) = 2 :=
by
  sorry

end sum_of_digits_8_pow_2003_l237_237716


namespace magnitude_of_complex_l237_237947

theorem magnitude_of_complex :
  ∀ (z : ℂ), (z = 1 + 2 * complex.i + complex.i ^ 3) → complex.abs z = real.sqrt 2 :=
by
  intros z h
  sorry

end magnitude_of_complex_l237_237947


namespace integer_solution_m_l237_237131

def fractional_equation (m x : ℤ) : Prop :=
  (mx - 1) / (x - 2) + 1 / (2 - x) = 2

theorem integer_solution_m (m : ℤ) :
  (∃ x : ℤ, fractional_equation m x) ↔ (m = 4 ∨ m = 3 ∨ m = 0) :=
by
  sorry

end integer_solution_m_l237_237131


namespace wire_length_l237_237184

def f (a : ℝ) : ℝ := (|(|a| - 2)| - 1|)

def S : set (ℝ × ℝ) := {p | f p.1 + f p.2 = 1}

theorem wire_length (hS: ∀ (p : ℝ × ℝ), p ∈ S ↔ f p.1 + f p.2 = 1) : 
  ∃ (a b : ℕ), a = 8 ∧ b = 2 ∧ is_irrational (real.sqrt (b : ℝ)) ∧ (length_of_wire S = a * real.sqrt (b)) ∧ (a + b = 10) :=
begin
  sorry
end

end wire_length_l237_237184


namespace taxi_fare_distance_condition_l237_237275

theorem taxi_fare_distance_condition (x : ℝ) (h1 : 7 + (max (x - 3) 0) * 2.4 = 19) : x ≤ 8 := 
by
  sorry

end taxi_fare_distance_condition_l237_237275


namespace number_zero_points_eq_three_l237_237827

noncomputable def f (x : ℝ) : ℝ := 2^(x - 1) - x^2

theorem number_zero_points_eq_three : ∃ x1 x2 x3 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (∀ y : ℝ, f y = 0 → (y = x1 ∨ y = x2 ∨ y = x3)) :=
sorry

end number_zero_points_eq_three_l237_237827


namespace triangle_median_question_l237_237215

open Real

noncomputable theory

-- Define the main entities and conditions as hypothesis
variables {D E F P Q G : Type} [Point]

-- Definitions of medians and their lengths
variable (h1 : Median D P ∧ Median E Q)
variable (h2 : DP = 15)
variable (h3 : EQ = 20)
variable (h4 : Perpendicular DP EQ)

-- Proving the length of ∆ DEF
theorem triangle_median_question
  (h1 : Median D P ∧ Median E Q)
  (h2 : DP = 15)
  (h3 : EQ = 20)
  (h4 : Perpendicular DP EQ) :
  DF = 20 * (√13) / 3 := sorry

end triangle_median_question_l237_237215


namespace simplify_expression_l237_237833

theorem simplify_expression : (- (1 / 343 : ℝ)) ^ (-3 / 5) = -343 := by sorry

end simplify_expression_l237_237833


namespace problem_part1_problem_part2_l237_237974

-- Definitions based on conditions
variables {A B C D E F M N : Type*} -- Points in our geometric configuration
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E]
          [inhabited F] [inhabited M] [inhabited N]
variables [mathlib]
(AB CD AC BD DF : Line) -- Lines forming sides and diagonals
[incircle ABCD : Circle] -- Quadrilateral is inscribed in a circle

-- Definition of conditions
def condition1 : Prop := cyclic_quad ABCD
def condition2 : Prop := intersects AC BD E
def condition3 : Prop := is_perpendicular AC BD
def condition4 : Prop := same_length AB AC ∧ same_length AC BD
def condition5 : Prop := perpendicular_from DF D BD
def condition6 : Prop := angle_bisector BFD AD BD M N

-- Lean statement for questions
theorem problem_part1 : condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ condition6 →
  angle BAD = 3 * angle DAC :=
begin
  sorry
end

theorem problem_part2 : condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ condition6 →
  MN = MD →
  length BF = length CD + length DF :=
begin
  sorry
end

end problem_part1_problem_part2_l237_237974


namespace ratio_proof_l237_237258

-- Define the probabilities of events A and B
def P_A : ℝ := 0.15789473684210525
def P_B : ℝ := P_A / 2

-- Events A and B are independent
axiom H1 : ∀ (A B : Prop) (PA PB : ℝ), (PA * PB = P_A * P_B)

-- The probability of A is greater than 0
axiom H2 : P_A > 0

-- The probability of at least one of A or B occurs
def P_A_or_B : ℝ := P_A + P_B - P_A * P_B

-- The probability of both A and B occurs is P_A * P_B
def P_A_and_B : ℝ := P_A * P_B

-- The ratio of the probability of at least one of A or B to the probability of both A and B
def ratio : ℝ := P_A_or_B / P_A_and_B

theorem ratio_proof : ratio = 18 := 
by
  sorry

end ratio_proof_l237_237258


namespace cost_per_ball_correct_l237_237933

-- Define the values given in the conditions
def total_amount_paid : ℝ := 4.62
def number_of_balls : ℝ := 3.0

-- Define the expected cost per ball according to the problem statement
def expected_cost_per_ball : ℝ := 1.54

-- Statement to prove that the cost per ball is as expected
theorem cost_per_ball_correct : (total_amount_paid / number_of_balls) = expected_cost_per_ball := 
sorry

end cost_per_ball_correct_l237_237933


namespace laura_weekly_mileage_l237_237180

-- Define the core conditions

-- Distance to school per round trip (house <-> school)
def school_trip_distance : ℕ := 20

-- Number of trips to school per week
def school_trips_per_week : ℕ := 7

-- Distance to supermarket: 10 miles farther than school
def extra_distance_to_supermarket : ℕ := 10
def supermarket_trip_distance : ℕ := school_trip_distance + 2 * extra_distance_to_supermarket

-- Number of trips to supermarket per week
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly distance
def total_distance_per_week : ℕ := 
  (school_trips_per_week * school_trip_distance) +
  (supermarket_trips_per_week * supermarket_trip_distance)

-- Theorem to prove the total distance Laura drives per week
theorem laura_weekly_mileage :
  total_distance_per_week = 220 := by
  sorry

end laura_weekly_mileage_l237_237180


namespace triangle_third_side_l237_237682

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l237_237682


namespace wall_length_is_800_l237_237495

variables (x : ℝ)
-- Conditions
def brick_volume := 80 * 11.25 * 6 -- volume of one brick in cm^3
def wall_height := 600 -- height of the wall in cm
def wall_width := 22.5 -- width of the wall in cm
def wall_volume := x * wall_width * wall_height
def total_brick_volume := 2000 * brick_volume

-- Assertion
theorem wall_length_is_800 :
  wall_volume x = total_brick_volume → x = 800 :=
by
  sorry

end wall_length_is_800_l237_237495


namespace parabola_FV_sum_l237_237028

noncomputable def problem_statement : Prop :=
  ∃ (F V B : ℝ × ℝ), 
  -- Conditions
  (∥B - F∥ = 25) ∧ (∥B - V∥ = 24) ∧
  -- Question with Answer
  (let FV := ∥F - V∥ in FV = 50 / 3)

theorem parabola_FV_sum : problem_statement := 
by 
  sorry

end parabola_FV_sum_l237_237028


namespace max_integer_solutions_self_centered_poly_l237_237755

noncomputable def self_centered_polynomial (p : ℤ[X]) : Prop :=
  polynomial.coeff p (nat_degree p) ∈ ℤ ∧ polynomial.eval 500 p = 500

theorem max_integer_solutions_self_centered_poly (p : ℤ[X]) (k : ℤ) :
  self_centered_polynomial p → polynomial.eval k p = k^2 → k = 495 ∨ k = 505 ∨ k = 490 ∨ k = 510 ∨ k = 485 ∨ k = 515 :=
sorry

end max_integer_solutions_self_centered_poly_l237_237755


namespace original_number_is_400_l237_237004

theorem original_number_is_400
  (N : ℝ)
  (h1 : N + 0.20 * N = 480)
  (x : ℝ)
  (h2 : 408 * x^2 = 5 * x^3 + 24 * x - 50) :
  N = 400 :=
begin
  sorry
end

end original_number_is_400_l237_237004


namespace ashwin_polygon_area_l237_237797

-- Define the problem conditions
def starts_at_origin : ℕ × ℕ := (0, 0)

def odd_step_move_right (n : ℕ) : Prop :=
  odd n → (n % 2 = 1)

def even_step_move_up (n m : ℕ) : Prop :=
  even n → (m = Nat.log2 n)

def total_steps : ℕ := 2^2017 - 1
def final_x_coordinate : ℕ := 2^2016
def final_y_coordinate : ℕ := 2^2017 - 2018

-- Statement of the proof problem
theorem ashwin_polygon_area :
  area_of_polygon_bounded_by_ashwin_path starts_at_origin total_steps odd_step_move_right even_step_move_up final_x_coordinate final_y_coordinate =
  2^2015 * (2^2017 - 2018) :=
by
  sorry

end ashwin_polygon_area_l237_237797


namespace product_formula_l237_237803

theorem product_formula (n : ℕ) (h : 2 ≤ n) :
    ∏ k in Finset.range(n+1) \ Finset.range(2), (k^3 - 1) / (k^3 + 1) = 
    (2 / 3) * (1 + 1 / (n * (n + 1))) := 
sorry

end product_formula_l237_237803


namespace smallest_positive_debt_resolvable_l237_237655

theorem smallest_positive_debt_resolvable :
  ∃ p g : ℤ, 280 * p + 200 * g = 40 ∧
  ∀ k : ℤ, k > 0 → (∃ p g : ℤ, 280 * p + 200 * g = k) → 40 ≤ k :=
by
  sorry

end smallest_positive_debt_resolvable_l237_237655


namespace intersecting_lines_angle_difference_l237_237137

-- Define the conditions
def angle_y : ℝ := 40
def straight_angle_sum : ℝ := 180

-- Define the variables representing the angles
variable (x y : ℝ)

-- Define the proof problem
theorem intersecting_lines_angle_difference : 
  ∀ x y : ℝ, 
  y = angle_y → 
  (∃ (a b : ℝ), a + b = straight_angle_sum ∧ a = y ∧ b = x) → 
  x - y = 100 :=
by
  intros x y hy h
  sorry

end intersecting_lines_angle_difference_l237_237137


namespace Laura_weekly_driving_distance_l237_237183

theorem Laura_weekly_driving_distance :
  ∀ (house_to_school : ℕ) (extra_to_supermarket : ℕ) (school_days_per_week : ℕ) (supermarket_trips_per_week : ℕ),
    house_to_school = 20 →
    extra_to_supermarket = 10 →
    school_days_per_week = 5 →
    supermarket_trips_per_week = 2 →
    (school_days_per_week * house_to_school + supermarket_trips_per_week * ((house_to_school / 2) + extra_to_supermarket) * 2) = 180 :=
by
  intros house_to_school extra_to_supermarket school_days_per_week supermarket_trips_per_week
  assume house_to_school_eq : house_to_school = 20
  assume extra_to_supermarket_eq : extra_to_supermarket = 10
  assume school_days_per_week_eq : school_days_per_week = 5
  assume supermarket_trips_per_week_eq : supermarket_trips_per_week = 2
  rw [house_to_school_eq, extra_to_supermarket_eq, school_days_per_week_eq, supermarket_trips_per_week_eq]
  sorry

end Laura_weekly_driving_distance_l237_237183


namespace area_triangle_DGF_l237_237377

-- Define the vertices of the square and the right triangle
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, 12⟩
def C : Point := ⟨10, 12⟩
def D : Point := ⟨10, 0⟩
def E : Point := ⟨22, 0⟩
def F : Point := ⟨10, 18⟩

-- Define the intersection point G
def G : Point := ⟨6, 0⟩

-- Define a function to calculate the area of the triangle
def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * Real.abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)))

-- Declare the theorem to be proved
theorem area_triangle_DGF : triangle_area D G F = 90 := by
  sorry

end area_triangle_DGF_l237_237377


namespace problem_l237_237110

def f (x : ℝ) : ℝ := 
  if x < 1 then 1 + Real.log2 (2 - x)
  else Real.exp2 (x - 1)

theorem problem (h₁ : f(-2) = 1 + Real.log2 4) (h₂ : Real.log2 4 = 2) (h₃ : f(Real.log2 3) = Real.exp2 (Real.log2 3 - 1)) (h₄ : Real.exp2 (Real.log2 3 - 1) = 3 / 2):
  f(-2) + f(Real.log2 3) = 9 / 2 := by
sorry

end problem_l237_237110


namespace investment_time_q_proof_l237_237656

noncomputable def investment_time_q (P Q R : ℕ) (Profit_p Profit_q Profit_r : ℕ) 
  (Time_p Time_r : ℕ) (ratio_investments : P:Q:R = 7:5:8)
  (ratio_profits : Profit_p:Profit_q:Profit_r = 14:10:24) 
  (p_time_invested : Time_p = 10) (r_time_invested : Time_r = 15) : ℕ := 
  (50 : ℕ)

theorem investment_time_q_proof (P Q R : ℕ) (Profit_p Profit_q Profit_r : ℕ) 
  (Time_p Time_r : ℕ) (ratio_investments : P:Q:R = 7:5:8)
  (ratio_profits : Profit_p:Profit_q:Profit_r = 14:10:24) 
  (p_time_invested : Time_p = 10) (r_time_invested : Time_r = 15) :
  investment_time_q P Q R Profit_p Profit_q Profit_r Time_p Time_r ratio_investments 
  ratio_profits p_time_invested r_time_invested = 50 :=
sorry

end investment_time_q_proof_l237_237656


namespace number_of_primes_diffs_in_set_l237_237916

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def candidates : List ℕ := List.range' 7 10 4

def num_of_numbers_can_be_written_as_diff_of_two_primes : ℕ :=
  candidates.countp (λ x => is_prime (x + 2))

theorem number_of_primes_diffs_in_set : num_of_numbers_can_be_written_as_diff_of_two_primes = 2 := 
sorry

end number_of_primes_diffs_in_set_l237_237916


namespace trigonometric_identity_proof_l237_237099

noncomputable def α : Real := sorry

theorem trigonometric_identity_proof :
  let sinα := - (Real.sqrt 3) / 2,
      cosα := -1 / 2,
      tanα := Real.sqrt 3 in
  (sinα = Real.sin α) ∧ 
  (cosα = Real.cos α) ∧
  (tanα = Real.tan α) ∧
  ((Real.sin (α - Real.pi) + Real.cos (α + Real.pi / 2)) / (Real.tan (Real.pi + α)) = 1) ∧
  (Real.tan (α + Real.pi / 4) < 0) := 
by 
  simp [Real.sin, Real.cos, Real.tan]
  sorry

end trigonometric_identity_proof_l237_237099


namespace probability_of_event_l237_237096

noncomputable def probability_gteq (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : Prop :=
  7 * x - 3 ≥ 0

theorem probability_of_event : 
  (∀ x, probability_gteq x → measure_theory.measure_space.probability_space.measure (set.Icc x 1) (measure_theory.measure_space.borel ℝ) = 4 / 7) :=
sorry

end probability_of_event_l237_237096


namespace third_side_triangle_max_l237_237693

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l237_237693


namespace range_of_a_l237_237113

-- Given conditions and target to prove the range of a
theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  -real.sqrt 6 / 3 ≤ a ∧ a ≤ real.sqrt 6 / 3 :=
sorry

end range_of_a_l237_237113


namespace intersection_product_eq_two_l237_237901

theorem intersection_product_eq_two : 
  ∀ (t : ℝ),
  let x := -1 + (sqrt 2 / 2) * t,
      y := (sqrt 2 / 2) * t,
      curve := (r : ℝ) (θ : ℝ) => r ^ 2 * cos θ ^ 2 = r * sin θ,
      line := (ρ : ℝ) (θ : ℝ) => sqrt 2 * ρ * cos (θ + (π / 4)) = -1,
      M := (-1 : ℝ, 0 : ℝ),
      f := (t : ℝ) => ((sqrt 2 / 2) * t)^2 - 3 * sqrt 2 * t + 2 in
  (x, y) ∈ curve →
  ∃ t1 t2 : ℝ, f t1 = 0 ∧ f t2 = 0 ∧ |t1 * t2| = 2 :=
by
  intro t x y curve line M f h
  sorry

end intersection_product_eq_two_l237_237901


namespace luke_remaining_amount_l237_237206

def octal_to_decimal (n : Nat) : Nat :=
  List.foldr (λ (x : Nat) (acc : Nat), x + 8 * acc) 0 (Nat.digits 8 n)

def total_expenses : Nat :=
  1200 + 600

def remaining_amount (savings : Nat) : Nat :=
  savings - total_expenses

theorem luke_remaining_amount : remaining_amount (octal_to_decimal 5377) = 1015 :=
  by sorry

end luke_remaining_amount_l237_237206


namespace determine_a20_l237_237756

noncomputable def f (z : ℂ) (a : ℕ → ℕ) : ℂ :=
  (1 - z)^(a 1) * (1 - z^2)^(a 2) * (1 - z^3)^(a 3) * ... * (1 - z^20)^(a 20)

def f_truncated (z : ℂ) (a : ℕ → ℕ) : ℂ :=
  f z a - ∑ k in finset.range (21), z^k * coeff (z^k) (f z a)

theorem determine_a20 (a : ℕ → ℕ) : 
  (∀ z : ℂ, f_truncated z a = 1 - 3*z) →
  a 20 = 2^17 - 3^5 := 
by
  -- We skip the proof
  sorry

end determine_a20_l237_237756


namespace sin_x_in_terms_of_a_b_l237_237932

theorem sin_x_in_terms_of_a_b (a b x : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < x) (h4 : x < π / 2)
  (h5 : Real.cot x = (a ^ 2 - b ^ 2) / (2 * a * b)) : 
  Real.sin x = (2 * a * b) / (a ^ 2 + b ^ 2) :=
sorry

end sin_x_in_terms_of_a_b_l237_237932


namespace count_8_digit_numbers_l237_237123

theorem count_8_digit_numbers : 
  (∃ n : ℕ, n = 90_000_000) ↔ 
  (∃ L : Fin 9 → Fin 10, ∀ i : Fin 9, L i ≠ 0) :=
begin
  sorry
end

end count_8_digit_numbers_l237_237123


namespace roger_received_money_l237_237603

variable (initial_amount spent current_amount received_from_mom : ℝ)
variable (h_initial : initial_amount = 45)
variable (h_spent : spent = 20)
variable (h_current : current_amount = 71)
variable (h_remaining : initial_amount - spent = 25)

theorem roger_received_money :
  current_amount - (initial_amount - spent) = 46 :=
by
  rw [h_initial, h_spent, h_current, h_remaining]
  sorry

end roger_received_money_l237_237603


namespace problem_statement_l237_237975

theorem problem_statement (EF GH : ℝ) (EH FG : ℕ) (E G : ℝ) 
  (h1 : ∠E = ∠G) 
  (h2 : EF = 200) 
  (h3 : GH = 200)
  (h4 : EH = FG)
  (h5 : EF + GH + EH + FG = 760) 
  (h6 : EH + FG ≠ 200) :
  ⌊1000 * cos E⌋ = 1000 := 
by
  sorry

end problem_statement_l237_237975


namespace solution_set_x_f_x_neg_l237_237564

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f (-x) = -f(x)) -- f is odd
variable (h2 : ∀ x > 0, f x > f 0) -- f is increasing in (0, +∞)
variable (h3 : f (-3) = 0) -- f(-3) = 0

theorem solution_set_x_f_x_neg : 
  ∀ x, x * f x < 0 ↔ (0 < x ∧ x < 3) ∨ (-3 < x ∧ x < 0) := 
by
  sorry

end solution_set_x_f_x_neg_l237_237564


namespace sum_of_divisors_of_45_l237_237309

theorem sum_of_divisors_of_45 : (∑ d in (Finset.filter (λ x, 45 % x = 0) (Finset.range 46)), d) = 78 :=
by
  -- We'll need to provide the proof here
  sorry

end sum_of_divisors_of_45_l237_237309


namespace parallelogram_area_l237_237704

theorem parallelogram_area :
  ∀ (b h : ℝ), b = 20 → h = 4 → b * h = 80 := by
  intros b h hb hh
  rw [hb, hh]
  norm_num
  done
  sorry

end parallelogram_area_l237_237704


namespace average_bracelets_per_day_l237_237662

theorem average_bracelets_per_day
  (cost_of_bike : ℕ)
  (price_per_bracelet : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (h1 : cost_of_bike = 112)
  (h2 : price_per_bracelet = 1)
  (h3 : weeks = 2)
  (h4 : days_per_week = 7) :
  (cost_of_bike / price_per_bracelet) / (weeks * days_per_week) = 8 :=
by
  sorry

end average_bracelets_per_day_l237_237662


namespace cd_squared_eq_am_mul_bn_l237_237631

-- Given points A, B, C, M, N, D and line l such that:
-- 1. Line l is tangent to the circle with diameter AB at point C.
-- 2. M and N are projections of A and B onto the line l, respectively.
-- 3. D is the projection of point C onto AB.

variables {A B C M N D : Point}
variables {l : Line}
variables {circle : Circle}

-- Assume the necessary conditions as Lean hypotheses
axiom tangent_at_C : l.is_tangent_circle circle C
axiom circle_diameter_AB : circle.diameter A B
axiom projection_A_on_l : M = l.projection A
axiom projection_B_on_l : N = l.projection B
axiom projection_C_on_AB : D = Line (A, B).projection C

-- State the theorem
theorem cd_squared_eq_am_mul_bn : (distance C D) ^ 2 = (distance A M) * (distance B N) :=  
sorry

end cd_squared_eq_am_mul_bn_l237_237631


namespace second_player_wins_l237_237074

theorem second_player_wins (board : matrix (fin 4) (fin 2017) ℕ) :
  (∀ i j, board i j = 0) →
  (∀ turn : ℕ, ∀ i j : fin 4, ∀ x y : fin 4, x ≠ y → 
    (board i j = 1 ∧ ∃ a b, board x a = board y b ∧ board x a = 1 → board y b = 1)) →
  (∀ turn : ℕ, ∀ i j : fin 4, (turn % 2 = 1 → ∃ k l, board k l = 1 ∧ i ≠ k ∧ j ≠ l) →
    ∃ k l, board k l = 1 ∧ (turn % 2 = 0 → ∃ k l, board k l = 1 ∧ i ≠ k ∧ j ≠ l)) →
  (∃ turn : ℕ, ∀ i j : fin 4, (turn % 2 = 1 → ∃ k l, board k l = 1 ∧ i ≠ k ∧ j ≠ l) →
    ∃ i j, board i j = 1) →
  ∀ first_move : ℕ, ∃ second_move : ℕ, second_move = first_move + 1 :=
sorry

end second_player_wins_l237_237074


namespace perimeter_square_C_l237_237839

theorem perimeter_square_C {A B C : Type} (perim_A : ℕ) (perim_B : ℕ) (perim_C : ℕ) 
    (side_A side_B side_C : ℕ) 
    (hA : perim_A = 16) 
    (hB : perim_B = 32) 
    (hsideA : 4 * side_A = perim_A) 
    (hsideB : 4 * side_B = perim_B) 
    (hC : perim_C = 4 * (side_B - side_A)) : 
    perim_C = 64 := 
by 
  rw [hsideA, hA] at hsideA
  rw [hsideB, hB] at hsideB 
  rw [hsideA, hsideB, hC]
  sorry

end perimeter_square_C_l237_237839


namespace question_1_question_2_l237_237872

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define the set A
def A : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

-- Define the set B depending on parameter a
def B (a : ℝ) : Set ℝ := { x | (1 / 4) < (2 ^ (x - a)) ∧ (2 ^ (x - a)) ≤ 8 }

-- Question 1: Prove the intersection of the complement of A in U and B when a = 0
theorem question_1 : (U \ A) ∩ (B 0) = Icc (-2:ℝ) 0 :=
by
  sorry

-- Question 2: Prove the range of a if A ∪ B = B
theorem question_2 : (A ∪ (B a)) = B a → -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end question_1_question_2_l237_237872


namespace min_value_of_expression_l237_237572

theorem min_value_of_expression (y : Fin 50 → ℝ) 
  (h_pos : ∀ i, 0 < y i) 
  (h_sum_eq_one : ∑ i, (y i)^2 = 1) :
  ∑ i, (y i)^3 / (1 - (y i)^2) ≥ 27 / 4 :=
sorry

end min_value_of_expression_l237_237572


namespace range_of_x_l237_237202

def largestIntNotGreaterThan (x : ℝ) : ℤ := ⌊x⌋₊

theorem range_of_x (x : ℝ) (h : 0 < largestIntNotGreaterThan (2 * x + 2) ∧ largestIntNotGreaterThan (2 * x + 2) < 3) :
    -1 / 2 ≤ x ∧ x < 1 / 2 := 
sorry

end range_of_x_l237_237202


namespace price_of_green_tractor_l237_237292

theorem price_of_green_tractor
  (red_tractors_sold : ℕ) 
  (green_tractors_sold : ℕ) 
  (price_red_tractor : ℝ) 
  (total_salary : ℝ) 
  (commission_red : ℝ) 
  (commission_green : ℝ) :
  red_tractors_sold = 2 → 
  green_tractors_sold = 3 → 
  price_red_tractor = 20000 → 
  total_salary = 7000 →
  commission_red = 0.10 → 
  commission_green = 0.20 → 
  let commission_from_red := red_tractors_sold * commission_red * price_red_tractor in
  let remaining_salary := total_salary - commission_from_red in
  let commission_per_green := remaining_salary / green_tractors_sold in
  let price_green_tractor := commission_per_green / commission_green in
  price_green_tractor = 5000 := by
  intros hsold_red hsold_green hprice_red hsalary hcommission_red hcommission_green
  let commission_from_red := red_tractors_sold * commission_red * price_red_tractor
  let remaining_salary := total_salary - commission_from_red
  let commission_per_green := remaining_salary / green_tractors_sold
  let price_green_tractor := commission_per_green / commission_green
  sorry

end price_of_green_tractor_l237_237292


namespace solution_l237_237905

namespace Proof

open Set

def proof_problem : Prop :=
  let U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5, 6}
  A ∩ (U \ B) = {1, 2}

theorem solution : proof_problem := by
  -- The pre-defined proof_problem must be shown here
  -- Proof: sorry
  sorry

end Proof

end solution_l237_237905


namespace linear_system_solution_l237_237116

theorem linear_system_solution (a b : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := 
by
  sorry

end linear_system_solution_l237_237116


namespace find_PQ_length_l237_237969

variable (P Q R M N : Type)
variable [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variable [MetricSpace M] [MetricSpace N]

variable (dist : P → Q → ℝ)
variable {right_triangle : Prop}
variable (isRightTriangle : right_triangle)
variable (isMedianPM : P → M → Prop)
variable (isMedianQN : Q → N → Prop)

theorem find_PQ_length
  (hPQ : right_triangle P Q R)
  (hPM_median : isMedianPM P M)
  (hQN_median : isMedianQN Q N)
  (PM_length : dist P M = 8)
  (QN_length : dist Q N = 4 * Real.sqrt 5) :
  dist P Q = 16 :=
by
  sorry

end find_PQ_length_l237_237969


namespace line_perpendicular_two_planes_parallel_l237_237331

theorem line_perpendicular_two_planes_parallel (l : ℝ) (α β : ℝ) : 
  l ⊥ α → l ⊥ β → α ∥ β :=
by
  sorry

end line_perpendicular_two_planes_parallel_l237_237331


namespace relationship_l237_237589

def M : set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}
def N : set (ℝ × ℝ) := 
  {p | real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * real.sqrt 2}
def P : set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

theorem relationship : M ⊆ N ∧ N ⊆ P ∧ (∃ x ∈ N, x ∉ M) ∧ (∃ y ∈ P, y ∉ N) :=
by
  sorry

end relationship_l237_237589


namespace tangent_line_at_point_l237_237092

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 
  Real.log (-x) + 3 * x 
else 
  Real.log x - 3 * x

def tangent_line_equation (x0 y0 slope : ℝ) : (ℝ → ℝ) :=
  λ x, slope * (x - x0) + y0

theorem tangent_line_at_point :
  f 1 = -3 ∧ f(-1) = f 1 ∧ (∃ l : ℝ, ∀ x > 0, has_deriv_at f (1/x - 3) x) 
  → tangent_line_equation 1 (-3) (-2) = λ x, -2 * x + 1 :=
begin
  sorry
end

end tangent_line_at_point_l237_237092


namespace area_of_triangle_AEB_is_correct_l237_237153

noncomputable def area_triangle_AEB : ℚ :=
by
  -- Definitions of given conditions
  let AB := 5
  let BC := 3
  let DF := 1
  let GC := 2

  -- Conditions of the problem
  have h1 : AB = 5 := rfl
  have h2 : BC = 3 := rfl
  have h3 : DF = 1 := rfl
  have h4 : GC = 2 := rfl

  -- The goal to prove
  exact 25 / 2

-- Statement in Lean 4 with the conditions and the correct answer
theorem area_of_triangle_AEB_is_correct :
  area_triangle_AEB = 25 / 2 := sorry -- The proof is omitted for this example

end area_of_triangle_AEB_is_correct_l237_237153


namespace toys_per_week_production_l237_237361

-- Define the necessary conditions
def days_per_week : Nat := 4
def toys_per_day : Nat := 1500

-- Define the theorem to prove the total number of toys produced per week
theorem toys_per_week_production : 
  ∀ (days_per_week toys_per_day : Nat), 
    (days_per_week = 4) →
    (toys_per_day = 1500) →
    (days_per_week * toys_per_day = 6000) := 
by
  intros
  sorry

end toys_per_week_production_l237_237361


namespace trigonometric_identity_l237_237733

theorem trigonometric_identity :
  sin (14 * real.pi / 180) * cos (46 * real.pi / 180) + sin (46 * real.pi / 180) * cos (14 * real.pi / 180) = real.sqrt 3 / 2 :=
sorry

end trigonometric_identity_l237_237733


namespace sum_of_divisors_of_45_l237_237319

theorem sum_of_divisors_of_45 :
  let divisors := [1, 3, 9, 5, 15, 45] in
  list.sum divisors = 78 :=
by
  -- this is where the proof would go
  sorry


end sum_of_divisors_of_45_l237_237319


namespace shauna_lowest_possible_score_l237_237235

theorem shauna_lowest_possible_score :
  ∀ (score1 score2 score3 max_points desired_avg num_tests total_needed_score last_two_total needed_score_on_one),
    score1 = 86 →
    score2 = 102 →
    score3 = 97 →
    max_points = 120 →
    desired_avg = 90 →
    num_tests = 5 →
    total_needed_score = desired_avg * num_tests →
    total_needed_score = 450 →
    last_two_total = total_needed_score - (score1 + score2 + score3) →
    last_two_total = 165 →
    (∃ score4 score5, score4 ≤ max_points ∧ score5 ≤ max_points ∧ score4 + score5 = last_two_total) →
    (∃ score4 score5, (score4 = max_points ∧ score5 = 45) ∨ (score5 = max_points ∧ score4 = 45)) :=
by
  intros score1 score2 score3 max_points desired_avg num_tests total_needed_score last_two_total needed_score_on_one h_score1 h_score2 h_score3 h_max_points h_desired_avg h_num_tests h_total_needed_score h_tns_equals h_last_two_total h_exists_last_two_total
  sorry

end shauna_lowest_possible_score_l237_237235


namespace angle_sum_proof_l237_237980

theorem angle_sum_proof 
  (A B C x y : ℝ)
  (hA : A = 34)
  (hB : B = 80)
  (hC : C = 38)
  (hx_y_sum : x + y = 2 * C) :
  x + y = 76 :=
by
  rw [hC] at hx_y_sum
  exact hx_y_sum.symm

end angle_sum_proof_l237_237980


namespace decrypt_message_consistent_l237_237219

def digit_value : Type := ℤ

structure Decipher :=
  (M S T I K : digit_value)
  (distinct : ∀ a b c d e f g h i j, set.Union {a, b, c, d, e, f, g, h, i, j} = {S, T, I, K, M} → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j)
  (no_leading_zero : ∀ (a b : digit_value), a ≠ 0)

theorem decrypt_message_consistent (d : Decipher) (M S T I K : digit_value) : 
  M = 1 ∧ S = 5 ∧ T = 7 ∧ I = 8 ∧ K = 6 → 
  (S * 10000 + T * 1000 + I * 100 + K * 10 + S) + 
  (S * 10000 + T * 1000 + I * 100 + K * 10 + S) = 
  (M * 100000 + S * 10000 + T * 1000 + I * 100 + K * 10 + S) :=
begin
  sorry
end

end decrypt_message_consistent_l237_237219


namespace sum_floor_log_eq_18064_l237_237025

noncomputable def sum_floor_log : ℕ :=
  ∑ N in Finset.range (2048 + 1), Int.floor (Real.log N / Real.log 3)

theorem sum_floor_log_eq_18064 : sum_floor_log = 18064 :=
by sorry

end sum_floor_log_eq_18064_l237_237025


namespace cube_volume_l237_237647

theorem cube_volume (s : ℝ) (hs : 12 * s = 96) : s^3 = 512 := by
  have s_eq : s = 8 := by
    linarith
  rw s_eq
  norm_num

end cube_volume_l237_237647


namespace equation_of_line_AC_equation_of_circumcircle_l237_237886

-- Define the statements for the line AC and the circumcircle

def point (x y : ℝ) : Prop := True

def line (a b c : ℝ) : Prop := True -- represents ax + by + c = 0

def symmetric_about (p₁ p₂ l_x l_y l_c : ℝ) : Prop :=
  -- Placeholder definition for symmetry about a line
  p₂ - p₁ = 2 * (l_x * p₁ + l_y * p₂ + l_c) * (1 / (l_x^2 + l_y^2))

theorem equation_of_line_AC :
  ∃ (C_x C_y : ℝ), point C_x C_y ∧
  symmetric_about 1 1 C_x C_y 1 (-1) 2 ∧
  -- line AC should have the form ax + by + c = 0
  line (1:ℝ) (1:ℝ) (-2: ℝ) := 
sorry

theorem equation_of_circumcircle :
  ∃ (d e f : ℝ),
  (x^2 + y^2 - (3/2) * x + (11/2) * y - 17 = 0) := 
sorry


end equation_of_line_AC_equation_of_circumcircle_l237_237886


namespace joe_time_to_school_is_3_25_l237_237553

-- Define the constants and conditions as given in the problem
def time_to_school (d : ℝ) (rw tr : ℝ) (tw tr_time : ℝ) : Prop :=
  (tr = 4 * rw) ∧
  (tw = 3) ∧
  (rw * tw = 3 * (d / 4)) ∧
  (rw = d / 4) ∧
  (tr_time = (d / 4) / (4 * rw))

-- Define the theorem to be proven
theorem joe_time_to_school_is_3_25 :
  ∃ d rw tr tw tr_time,
  time_to_school d rw tr tw tr_time →
  (tw + tr_time = 3.25) :=
begin
  sorry
end

end joe_time_to_school_is_3_25_l237_237553


namespace count_solid_circles_among_first_2006_l237_237769

-- Definition of the sequence sum for location calculation
def sequence_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2 - 1

-- Main theorem
theorem count_solid_circles_among_first_2006 : 
  ∃ n : ℕ, sequence_sum (n - 1) < 2006 ∧ 2006 ≤ sequence_sum n ∧ n = 62 :=
by {
  sorry
}

end count_solid_circles_among_first_2006_l237_237769


namespace greatest_third_side_l237_237672

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l237_237672


namespace sum_divisors_45_l237_237317

theorem sum_divisors_45 : ∑ d in (45 : ℕ).divisors, d = 78 :=
by
  sorry

end sum_divisors_45_l237_237317


namespace minimum_expression_value_l237_237865

theorem minimum_expression_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := 
by
  sorry

end minimum_expression_value_l237_237865


namespace total_missing_keys_l237_237994

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end total_missing_keys_l237_237994


namespace curve_symmetric_about_y_axis_l237_237884

theorem curve_symmetric_about_y_axis :
  ∀ (M : ℝ × ℝ), (∃ (C : set (ℝ × ℝ)), (∀ (M ∈ C), (dist M (M.1, 0) = dist M (0, 4)))) →
  ∃ (C : set (ℝ × ℝ)), (symmetric := λ (p : ℝ × ℝ), (p.1, -p.2) ∈ C ↔ p ∈ C) :=
by
  sorry

end curve_symmetric_about_y_axis_l237_237884


namespace greatest_third_side_l237_237694

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l237_237694


namespace primeDates2008to2012_l237_237812

-- Definitions for the specific months and the leap year conditions
def isLeapYear (year : ℕ) : Prop := (year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0))

def primeDaysInMonth (month : ℕ) (isLeap : Bool) : ℕ :=
  match month, isLeap with
  | 2, true  => 10 -- February in a leap year
  | 2, false =>  9 -- February in a non-leap year
  | 3, _     => 11 -- March
  | 5, _     => 11 -- May
  | 7, _     => 11 -- July
  | 11, _    => 10 -- November
  | _, _     =>  0 -- Other months are not considered

def primeDatesInYear (year : ℕ) : ℕ :=
  let isLeap := isLeapYear year
  (primeDaysInMonth 2 isLeap) +
  (primeDaysInMonth 3 isLeap) +
  (primeDaysInMonth 5 isLeap) +
  (primeDaysInMonth 7 isLeap) +
  (primeDaysInMonth 11 isLeap)

-- Main theorem statement
theorem primeDates2008to2012 : primeDatesInYear 2008 + primeDatesInYear 2009 + primeDatesInYear 2010 + primeDatesInYear 2011 + primeDatesInYear 2012 = 262 := sorry

end primeDates2008to2012_l237_237812


namespace find_a_l237_237490

-- Definitions of points and collinearity condition  
structure Point :=
  (x : ℝ) (y : ℝ)

def A (a : ℝ) : Point := ⟨a, 2⟩
def B : Point := ⟨5, 1⟩
def C (a : ℝ) : Point := ⟨-4, 2 * a⟩

def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem find_a (a : ℝ) : collinear (A a) B (C a) → (a = 2 ∨ a = 7 / 2) :=
begin
  assume h,
  sorry,
end

end find_a_l237_237490


namespace digit_is_4_l237_237253

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem digit_is_4 (d : ℕ) (hd0 : is_even d) (hd1 : is_divisible_by_3 (14 + d)) : d = 4 :=
  sorry

end digit_is_4_l237_237253


namespace true_proposition_l237_237106

theorem true_proposition (l : Line) (α : Plane) : 
  ∃ (m : Line), m ∈ α ∧ m ⊥ l :=
sorry

end true_proposition_l237_237106


namespace triangle_third_side_l237_237684

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l237_237684


namespace integer_solutions_count_l237_237633

theorem integer_solutions_count : 
  ((set_of (λ x : ℤ, (x^2 - x - 1) ^ (x + 2) = 1)).finite ∧ 
  (finset.card (finset.filter (λ x : ℤ, (x^2 - x - 1) ^ (x + 2) = 1) (finset.Icc -100 100))) = 4) :=
by
  sorry

end integer_solutions_count_l237_237633


namespace cryptarithm_correct_l237_237221

-- Definitions for letters as digits
variables {S T I K M A D R U Z : ℕ}

-- Conditions for the problem
def cryptarithm_conditions : Prop :=
  (S ≠ T) ∧
  (S ≠ I) ∧
  (S ≠ K) ∧
  (S ≠ M) ∧
  (S ≠ A) ∧
  (S ≠ D) ∧
  (S ≠ R) ∧
  (S ≠ U) ∧
  (S ≠ Z) ∧
  (T ≠ I) ∧
  (T ≠ K) ∧
  (T ≠ M) ∧
  (T ≠ A) ∧
  (T ≠ D) ∧
  (T ≠ R) ∧
  (T ≠ U) ∧
  (T ≠ Z) ∧
  (I ≠ K) ∧
  (I ≠ M) ∧
  (I ≠ A) ∧
  (I ≠ D) ∧
  (I ≠ R) ∧
  (I ≠ U) ∧
  (I ≠ Z) ∧
  (K ≠ M) ∧
  (K ≠ A) ∧
  (K ≠ D) ∧
  (K ≠ R) ∧
  (K ≠ U) ∧
  (K ≠ Z) ∧
  (M ≠ A) ∧
  (M ≠ D) ∧
  (M ≠ R) ∧
  (M ≠ U) ∧
  (M ≠ Z) ∧
  (A ≠ D) ∧
  (A ≠ R) ∧
  (A ≠ U) ∧
  (A ≠ Z) ∧
  (D ≠ R) ∧
  (D ≠ U) ∧
  (D ≠ Z) ∧
  (R ≠ U) ∧
  (R ≠ Z) ∧
  (U ≠ Z) ∧
  5 ≤ S ∧  (* contribution of leading digits being valid *)
  S + S + x = 10 + M ∧
  T + T + x = 10*y + A ∧
  I + I = 10*z + 3 ∧
  K + K = 2 + S ∧
  2*S + z = 10 +1

-- Prove the numerical equivalent using cryptarithm function
theorem cryptarithm_correct (h : cryptarithm_conditions) : 
  ∃ (S T I K M A D R U Z : ℕ), cryptarithm_conditions ∧
  let s := "STIKS" in
  let sum := "MASTIKS" in
  let result := "DRIUZIS" in

end cryptarithm_correct_l237_237221


namespace inequality_integral_l237_237200

theorem inequality_integral {f g : ℝ → ℝ} (a : ℝ) 
    (hf1 : differentiable ℝ f) (hg1 : differentiable ℝ g)
    (hf_cont : continuous f) (hg_cont : continuous g)
    (h1 : f 0 = 0)
    (h2 : ∀ x ∈ set.Icc 0 1, 0 ≤ f' x)
    (h3 : ∀ x ∈ set.Icc 0 1, 0 ≤ g' x)
    (ha : a ∈ set.Icc 0 1) :
    ∫ x in 0 .. a, g x * f' x + ∫ x in 0 .. 1, f x * g' x ≥ f a * g 1 :=
sorry

end inequality_integral_l237_237200


namespace find_a_l237_237103

noncomputable def complex_z (a : ℝ) : ℂ :=
  (a : ℂ) / (2 + (1 : ℂ) * Complex.I) + (2 + (1 : ℂ) * Complex.I) / 5

theorem find_a : ∃ (a : ℝ), a = 3 :=
  ∃ a : ℝ, (complex_z a).re + (complex_z a).im = 1 → a = 3

end find_a_l237_237103


namespace problem_1_problem_2_l237_237162

theorem problem_1 (ρ θ : ℝ) :
    (ρ^2 - 2*ρ*(Real.cos θ) - 1 = 0) ↔ (ρ * (ρ - 2 * (Real.cos θ)) = 1) := sorry

theorem problem_2 (x y m A B t1 t2 : ℝ) (h1 : y = x - Real.sqrt 3)
  (PA PB : ℝ) (hPA : PA = Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2))
  (hPB : PB = Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2))
  (h2 : |PA * PB| = 2) :
  x - y - Real.sqrt 3 = 0 := sorry

end problem_1_problem_2_l237_237162


namespace factor_polynomial_l237_237838

theorem factor_polynomial (y : ℝ) :
  y^8 - 4 * y^6 + 6 * y^4 - 4 * y^2 + 1 = ((y - 1) * (y + 1))^4 :=
sorry

end factor_polynomial_l237_237838


namespace mark_total_votes_l237_237527

-- Definitions for the problem conditions
def first_area_registered_voters : ℕ := 100000
def first_area_undecided_percentage : ℕ := 5
def first_area_mark_votes_percentage : ℕ := 70

def remaining_area_increase_percentage : ℕ := 20
def remaining_area_undecided_percentage : ℕ := 7
def multiplier_for_remaining_area_votes : ℕ := 2

-- The Lean statement
theorem mark_total_votes : 
  let first_area_undecided_voters := first_area_registered_voters * first_area_undecided_percentage / 100
  let first_area_votes_cast := first_area_registered_voters - first_area_undecided_voters
  let first_area_mark_votes := first_area_votes_cast * first_area_mark_votes_percentage / 100

  let remaining_area_registered_voters := first_area_registered_voters * (1 + remaining_area_increase_percentage / 100)
  let remaining_area_undecided_voters := remaining_area_registered_voters * remaining_area_undecided_percentage / 100
  let remaining_area_votes_cast := remaining_area_registered_voters - remaining_area_undecided_voters
  let remaining_area_mark_votes := first_area_mark_votes * multiplier_for_remaining_area_votes

  let total_mark_votes := first_area_mark_votes + remaining_area_mark_votes
  total_mark_votes = 199500 := 
by
  -- We skipped the proof (it's not required as per instructions)
  sorry

end mark_total_votes_l237_237527


namespace range_of_m_l237_237449

theorem range_of_m (x : ℝ) (m : ℝ) (h : sqrt 3 * sin x - cos x = 4 - m) : 2 ≤ m ∧ m ≤ 6 :=
by
  sorry

end range_of_m_l237_237449


namespace profit_function_relationship_max_profit_and_reduction_l237_237357

/- Define initial conditions -/
def cost_per_kg : ℝ := 30
def initial_price_per_kg : ℝ := 48
def initial_sales_volume : ℝ := 500
def price_reduction_effect_per_kg (x : ℝ) : ℝ := 50 * x

/- Define the profit function -/
def profit_function (x : ℝ) : ℝ :=
  let new_price := initial_price_per_kg - x
  let new_sales_volume := initial_sales_volume + price_reduction_effect_per_kg x
  let profit_per_kg := new_price - cost_per_kg
  profit_per_kg * new_sales_volume

/- Prove the functional relationship -/
theorem profit_function_relationship (x : ℝ) : profit_function x = -50 * x^2 + 400 * x + 9000 :=
  by
  have h1 : profit_function x = (18 - x) * (500 + 50 * x), from sorry,
  have h2 : (18 - x) * (500 + 50 * x) = -50 * x^2 + 400 * x + 9000, from sorry,
  exact h1.trans h2

/- Prove the maximized profit and the corresponding reduction -/
theorem max_profit_and_reduction :
  let x := 4
  in profit_function x = 9800 :=
  by
  have h1 : profit_function 4 = -50 * 4^2 + 400 * 4 + 9000, from sorry,
  have h2 : -50 * 4^2 + 400 * 4 + 9000 = 9800, from sorry,
  exact h1.trans h2

end profit_function_relationship_max_profit_and_reduction_l237_237357


namespace light_ray_reflection_reverse_direction_l237_237630

theorem light_ray_reflection_reverse_direction (n : ℕ) (α : ℝ) (initial_angle : ℝ) :
  (α = 90 / n) → (∃ k : ℕ, k > 0 ∧ reflects_opposite k α initial_angle) :=
by
  sorry

end light_ray_reflection_reverse_direction_l237_237630


namespace polynomial_has_factor_x_plus_one_l237_237374

theorem polynomial_has_factor_x_plus_one : 
  ∃ (P : Polynomial ℝ), (P = (fun x => x^2 - 1) ∧ (P = (fun x => (x + 1) * (x - 1))) := 
by 
  sorry

end polynomial_has_factor_x_plus_one_l237_237374


namespace find_f_comp_l237_237483

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 3 else 2 ^ x

theorem find_f_comp (x : ℝ) (h : x = 1 / 3) : 
  f (f (f x)) = log (1 / 2) / log 3 :=
by 
  -- The detailed proof would be inserted here, but we use sorry to skip the proof.
  sorry

end find_f_comp_l237_237483


namespace prime_numbers_with_unique_digits_in_base_l237_237432

theorem prime_numbers_with_unique_digits_in_base (p : ℕ) (b : ℕ) :
  prime p ∧ (∀ k < b, ∃! d, p.digit b k = d) →
  p = 2 ∨ p = 5 ∨ p = 7 ∨ p = 11 ∨ p = 19 :=
by
  sorry

end prime_numbers_with_unique_digits_in_base_l237_237432


namespace triangle_properties_l237_237957

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosA cosB tan2A S : ℝ)

-- Initial conditions
def cos_A := cosA = sqrt 6 / 3
def cos_B := cosB = 2 * sqrt 2 / 3
def side_c := c = 2 * sqrt 2

-- First part, tan 2A
def tan_2A := tan2A = 2 * sqrt 2

-- Second part, the area of triangle ABC
def area_ABC := S = 2 * sqrt 2 / 3

-- The final proof problem
theorem triangle_properties (cos_A_condition : cos_A)
                            (cos_B_condition : cos_B)
                            (side_c_condition : side_c) :
  (tan_2A ∧ area_ABC) := by
  sorry

end triangle_properties_l237_237957


namespace circle_probability_l237_237964

noncomputable def problem_statement : Prop :=
  let outer_radius := 3
  let inner_radius := 1
  let pivotal_radius := 2
  let outer_area := Real.pi * outer_radius ^ 2
  let inner_area := Real.pi * pivotal_radius ^ 2
  let probability := inner_area / outer_area
  probability = 4 / 9

theorem circle_probability : problem_statement := sorry

end circle_probability_l237_237964


namespace cube_coverable_with_fewer_than_six_rhombuses_l237_237170

noncomputable def cover_unit_cube_with_fewer_than_six_rhombuses : Prop :=
  ∃ (R : Type) [Fintype R] [rhombus : R → Prop], Fintype.card R < 6 ∧
    (∃ (cover : R → set (ℝ × ℝ)), 
      (∀ r, rhombus r → is_rhombus (cover r)) ∧
      (⋃ r, cover r) = surface_of_cube ∧
      ∀ i j, i ≠ j → disjoint (cover i) (cover j))

axiom is_rhombus {S : set (ℝ × ℝ)} : Prop -- Definition/axiom for identifying sets that are rhombuses
axiom surface_of_cube : set (ℝ × ℝ) -- Definition/axiom for the surface of the cube

theorem cube_coverable_with_fewer_than_six_rhombuses :
  cover_unit_cube_with_fewer_than_six_rhombuses :=
sorry

end cube_coverable_with_fewer_than_six_rhombuses_l237_237170


namespace sum_of_three_fourth_powers_not_end_in_2019_l237_237408

theorem sum_of_three_fourth_powers_not_end_in_2019
  (a b c : ℤ) :
  let fourth_power_units := {0, 1, 5, 6}
  in ((a^4 % 10) ∈ fourth_power_units) ∧ 
     ((b^4 % 10) ∈ fourth_power_units) ∧ 
     ((c^4 % 10) ∈ fourth_power_units) →
  ((a^4 + b^4 + c^4) % 10) ≠ 9 :=
by
  sorry

end sum_of_three_fourth_powers_not_end_in_2019_l237_237408


namespace marcie_and_martin_in_picture_l237_237208

noncomputable def marcie_prob_in_picture : ℚ :=
  let marcie_lap_time := 100
  let martin_lap_time := 75
  let start_time := 720
  let end_time := 780
  let picture_duration := 60
  let marcie_position_720 := (720 % marcie_lap_time) / marcie_lap_time
  let marcie_in_pic_start := 0
  let marcie_in_pic_end := 20 + 33 + 1/3
  let martin_position_720 := (720 % martin_lap_time) / martin_lap_time
  let martin_in_pic_start := 20
  let martin_in_pic_end := 45 + 25
  let overlap_start := max marcie_in_pic_start martin_in_pic_start
  let overlap_end := min marcie_in_pic_end martin_in_pic_end
  let overlap_duration := overlap_end - overlap_start
  overlap_duration / picture_duration

theorem marcie_and_martin_in_picture :
  marcie_prob_in_picture = 111 / 200 :=
by
  sorry

end marcie_and_martin_in_picture_l237_237208


namespace count_diff_of_two_primes_l237_237921

theorem count_diff_of_two_primes (s : Set ℕ) (p : ℕ → Prop) (n : ℕ) :
  (∀ i, i ∈ s ↔ ∃ n : ℕ, i = 10 * n + 7) →
  (∀ k, k ∈ s → ∃ a b : ℕ, nat.prime a → nat.prime b → a - b = k) →
  #{i ∈ s | ∃ a b : ℕ, nat.prime a ∧ nat.prime b ∧ a - b = i} = 2 :=
by
  -- sorry as the proof would be filled in here
  sorry

end count_diff_of_two_primes_l237_237921


namespace shaded_region_area_l237_237159

noncomputable def radius_larger := 10
def radius_smaller := radius_larger / 3
def area_larger := Real.pi * radius_larger^2
def area_smaller := Real.pi * radius_smaller^2
def area_total_smaller := 2 * area_smaller
def area_shaded := area_larger - area_total_smaller

theorem shaded_region_area :
  area_shaded = (700 / 9) * Real.pi := by
  sorry

end shaded_region_area_l237_237159


namespace calculate_expression_l237_237084

theorem calculate_expression (a b c : ℝ) (A B C : ℝ)
  (hA : A ≠ 0 ∧ A ≠ π)
  (hB : B ≠ 0 ∧ B ≠ π)
  (hC : C ≠ 0 ∧ C ≠ π)
  (ha : a = 2)
  (hb : b = 3)
  (hc : c = 4)
  (cos_rule_A : c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos A)
  (cos_rule_B : b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B)
  (cos_rule_C : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos C) :
  2 * b * c * Real.cos A + 2 * c * a * Real.cos B + 2 * a * b * Real.cos C = 29 :=
by
  sorry

end calculate_expression_l237_237084


namespace tangency_condition_for_parabola_and_line_l237_237852

theorem tangency_condition_for_parabola_and_line (k : ℚ) :
  (∀ x y : ℚ, (6 * x - 4 * y + k = 0) ↔ (y^2 = 16 * x)) ↔ (k = 32 / 3) :=
  sorry

end tangency_condition_for_parabola_and_line_l237_237852


namespace solve_x_fx_neg_l237_237503

-- Define an odd function f which is increasing on (0, +∞) and f(-3) = 0
noncomputable def odd_increasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, 0 < x ∧ x < y → f x < f y)

theorem solve_x_fx_neg (f : ℝ → ℝ) (h_odd : odd_increasing_function f) (h_f_minus_3 : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = set.Ioc (-3) 0 ∪ set.Ioo 0 3 :=
by
  sorry

end solve_x_fx_neg_l237_237503


namespace fewest_cookies_by_ben_l237_237067

noncomputable def cookie_problem : Prop :=
  let ana_area := 4 * Real.pi
  let ben_area := 9
  let carol_area := Real.sqrt (5 * (5 + 2 * Real.sqrt 5))
  let dave_area := 3.375 * Real.sqrt 3
  let dough := ana_area * 10
  let ana_cookies := dough / ana_area
  let ben_cookies := dough / ben_area
  let carol_cookies := dough / carol_area
  let dave_cookies := dough / dave_area
  ben_cookies < ana_cookies ∧ ben_cookies < carol_cookies ∧ ben_cookies < dave_cookies

theorem fewest_cookies_by_ben : cookie_problem := by
  sorry

end fewest_cookies_by_ben_l237_237067


namespace trig_identity_l237_237344

theorem trig_identity : 
  sin (40 * Real.pi / 180) * sin (10 * Real.pi / 180) + cos (40 * Real.pi / 180) * sin (80 * Real.pi / 180) = sqrt 3 / 2 :=
sorry

end trig_identity_l237_237344


namespace train_pass_time_l237_237724

-- Define the given conditions
def train_length : ℕ := 20  -- length of the train in meters
def train_speed_kmph : ℕ := 36  -- speed of the train in kmph
def kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600  -- conversion function

-- Convert speed from kmph to m/s
def train_speed_mps : ℕ := kmph_to_mps train_speed_kmph

-- Statement to prove
theorem train_pass_time (h1 : train_length = 20) (h2 : train_speed_kmph = 36) : 
  (train_length / train_speed_mps) = 2 :=
by
  -- condition that speed is converted correctly
  have h3: train_speed_mps = 10 := by sorry
  
  -- using the length and converted speed to get time
  rw [←h3] 
  calc
  20 / 10 = 2 : by norm_num

end train_pass_time_l237_237724


namespace tractor_trailer_weight_after_deliveries_l237_237774

def initial_weight := 50000
def first_store_unload_percent := 0.10
def second_store_unload_percent := 0.20

theorem tractor_trailer_weight_after_deliveries: 
  let weight_after_first_store := initial_weight - (first_store_unload_percent * initial_weight)
  let weight_after_second_store := weight_after_first_store - (second_store_unload_percent * weight_after_first_store)
  weight_after_second_store = 36000 :=
by
  sorry

end tractor_trailer_weight_after_deliveries_l237_237774


namespace find_k_hexadecimal_to_decimal_l237_237935

theorem find_k_hexadecimal_to_decimal (k : ℕ) (hk : k > 0) :
  (1 * 6^3 + k * 6^1 + 5 = 239) → k = 3 :=
by {
  intro h,
  -- Proof omitted
  sorry
}

end find_k_hexadecimal_to_decimal_l237_237935


namespace min_area_of_triangle_l237_237463

noncomputable def hyperbola : set (ℝ × ℝ) :=
  {p | p.1^2 - (p.2^2) / 8 = 1}

def right_focus : ℝ × ℝ :=
  (3, 0)

def point_A : ℝ × ℝ :=
  (0, 6 * Real.sqrt 6)

def area_triangle (A F P : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 * F.2 + F.1 * P.2 + P.1 * A.2) - (A.2 * F.1 + F.2 * P.1 + P.2 * A.1))

theorem min_area_of_triangle (P : ℝ × ℝ) (hP : P ∈ hyperbola ∧ P.1 < 0) :
    area_triangle point_A right_focus P = 6 + 9 * Real.sqrt 6 :=
  sorry

end min_area_of_triangle_l237_237463


namespace missing_keys_total_l237_237993

-- Definitions for the problem conditions

def num_consonants : ℕ := 21
def num_vowels : ℕ := 5
def missing_consonants_fraction : ℚ := 1 / 7
def missing_vowels : ℕ := 2

-- Statement to prove the total number of missing keys

theorem missing_keys_total :
  let missing_consonants := num_consonants * missing_consonants_fraction in
  let total_missing_keys := missing_consonants + missing_vowels in
  total_missing_keys = 5 :=
by {
  -- Placeholder proof
  sorry
}

end missing_keys_total_l237_237993


namespace chord_radius_l237_237963

def rad (deg : ℝ) : ℝ :=
  deg * Real.pi / 180

theorem chord_radius (r : ℝ) :
  Exists (λ r, 
    let a := 20
    let b := 26
    let theta := rad (36 + 38 / 60) -- 36° 38'
    let d := a^2 + b^2 - 2 * a * b * Real.cos(theta)
    let chord_length := Real.sqrt d
    r = 13.02
    2 * r * Real.sin(theta / 2) = chord_length
  ) :=
  sorry

end chord_radius_l237_237963


namespace expression_simplified_l237_237246

theorem expression_simplified (d : ℤ) (h : d ≠ 0) :
  let a := 24
  let b := 61
  let c := 96
  a + b + c = 181 ∧ 
  (15 * d ^ 2 + 7 * d + 15 + (3 * d + 9) ^ 2 = a * d ^ 2 + b * d + c) := by
{
  sorry
}

end expression_simplified_l237_237246


namespace females_in_band_not_orchestra_l237_237602

/-- The band at Pythagoras High School has 120 female members. -/
def females_in_band : ℕ := 120

/-- The orchestra at Pythagoras High School has 70 female members. -/
def females_in_orchestra : ℕ := 70

/-- There are 45 females who are members of both the band and the orchestra. -/
def females_in_both : ℕ := 45

/-- The combined total number of students involved in either the band or orchestra or both is 250. -/
def total_students : ℕ := 250

/-- The number of females in the band who are NOT in the orchestra. -/
def females_in_band_only : ℕ := females_in_band - females_in_both

theorem females_in_band_not_orchestra : females_in_band_only = 75 := by
  sorry

end females_in_band_not_orchestra_l237_237602


namespace bears_per_shelf_l237_237379

def bears_initial : ℕ := 6

def shipment : ℕ := 18

def shelves : ℕ := 4

theorem bears_per_shelf : (bears_initial + shipment) / shelves = 6 := by
  sorry

end bears_per_shelf_l237_237379


namespace find_omega_l237_237348

def f (ω x : ℝ) : ℝ := sin (ω * x) + sqrt 3 * cos (ω * x)

theorem find_omega (ω α β : ℝ) (h1 : f ω α = -2) (h2 : f ω β = 0) (h3 : abs (α - β) = 3 * π / 4) (ω_pos : 0 < ω) :
  ω = 2 / 3 :=
sorry

end find_omega_l237_237348


namespace cross_shaped_pyramid_ways_l237_237610

-- Definition of the cross-shaped figure
def is_cross_shaped (squares : set (ℤ × ℤ)) : Prop :=
  (0, 0) ∈ squares ∧ (1, 0) ∈ squares ∧ (-1, 0) ∈ squares ∧
  (0, 1) ∈ squares ∧ (0, -1) ∈ squares ∧ |squares| = 5

-- Definition of a valid square placement
def is_valid_placement (squares : set (ℤ × ℤ)) (extra_square : (ℤ × ℤ)) : Prop :=
  is_cross_shaped squares ∧ extra_square ∉ squares ∧
  (extra_square = (2, 0) ∨ extra_square = (-2, 0) ∨
   extra_square = (0, 2) ∨ extra_square = (0, -2))

-- Statement of the problem
theorem cross_shaped_pyramid_ways :
  ∀ (squares : set (ℤ × ℤ)),
  is_cross_shaped squares →
  ∃ (extra_squares : set (ℤ × ℤ)), card extra_squares = 4 ∧
  ∀ (es : ℤ × ℤ), es ∈ extra_squares ↔ is_valid_placement squares es :=
sorry

end cross_shaped_pyramid_ways_l237_237610


namespace min_operations_to_reach_goal_l237_237399

-- Define the initial and final configuration of the letters
structure Configuration where
  A : Char := 'A'
  B : Char := 'B'
  C : Char := 'C'
  D : Char := 'D'
  E : Char := 'E'
  F : Char := 'F'
  G : Char := 'G'

-- Define a valid rotation operation
inductive Rotation
| rotate_ABC : Rotation
| rotate_ABD : Rotation
| rotate_DEF : Rotation
| rotate_EFC : Rotation

-- Function representing a single rotation
def applyRotation : Configuration -> Rotation -> Configuration
| config, Rotation.rotate_ABC => 
  { A := config.C, B := config.A, C := config.B, D := config.D, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_ABD => 
  { A := config.B, B := config.D, D := config.A, C := config.C, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_DEF => 
  { D := config.E, E := config.F, F := config.D, A := config.A, B := config.B, C := config.C, G := config.G }
| config, Rotation.rotate_EFC => 
  { E := config.F, F := config.C, C := config.E, A := config.A, B := config.B, D := config.D, G := config.G }

-- Define the goal configuration
def goalConfiguration : Configuration := 
  { A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G' }

-- Function to apply multiple rotations
def applyRotations (config : Configuration) (rotations : List Rotation) : Configuration :=
  rotations.foldl applyRotation config

-- Main theorem statement 
theorem min_operations_to_reach_goal : 
  ∃ rotations : List Rotation, rotations.length = 3 ∧ applyRotations {A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G'} rotations = goalConfiguration :=
sorry

end min_operations_to_reach_goal_l237_237399


namespace total_paths_A_to_C_via_B_l237_237120

-- Define the conditions
def steps_from_A_to_B : Nat := 6
def steps_from_B_to_C : Nat := 6
def right_moves_A_to_B : Nat := 4
def down_moves_A_to_B : Nat := 2
def right_moves_B_to_C : Nat := 3
def down_moves_B_to_C : Nat := 3

-- Define binomial coefficient function
def binom (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of paths for each segment
def paths_A_to_B : Nat := binom steps_from_A_to_B down_moves_A_to_B
def paths_B_to_C : Nat := binom steps_from_B_to_C down_moves_B_to_C

-- Theorem stating the total number of distinct paths
theorem total_paths_A_to_C_via_B : paths_A_to_B * paths_B_to_C = 300 :=
by
  sorry

end total_paths_A_to_C_via_B_l237_237120


namespace tan_BAD_l237_237973

-- Definition of equilateral triangle with side length 6 and midpoint D of BC
structure Triangle (α : Type) where
  A B C D : α 
  eq_side : dist A B = 6 ∧ dist B C = 6 ∧ dist C A = 6
  midpoint_D : D = midpoint B C

noncomputable def tan_angle_BAD (α : Type) [metric_space α] [inner_product_space ℝ α] (T : Triangle α) : ℝ :=
  sorry

theorem tan_BAD {α : Type} [metric_space α] [inner_product_space ℝ α] (T : Triangle α) :
  tan_angle_BAD α T = 1 / real.sqrt 3 :=
  sorry

end tan_BAD_l237_237973


namespace strictly_increasing_interval_l237_237825

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem strictly_increasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = Real.exp 1) ∧ ∀ x : ℝ, a < x ∧ x < b → f' x > 0 :=
by
  let f' := λ x, (1 - Real.log x) / (x^2)
  sorry

end strictly_increasing_interval_l237_237825


namespace longest_segment_in_ABCD_l237_237003

noncomputable def quadrilateral_longest_segment : Prop :=
  ∀ (A B C D : Type) 
  [is_point A] [is_point B] [is_point C] [is_point D]
  (angle_ABD angle_ADB angle_CBD angle_BDC : ℝ),
  angle_ABD = 30 ∧ angle_ADB = 70 ∧ angle_CBD = 60 ∧ angle_BDC = 80 →
  BD > AB ∧ BD > AD ∧ BD > BC ∧ BD > CD

-- Below statement needs to be proved
theorem longest_segment_in_ABCD : quadrilateral_longest_segment :=
  by sorry

end longest_segment_in_ABCD_l237_237003


namespace locus_of_G_as_E_moves_l237_237163

noncomputable section

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

variables {a b : ℝ} (A B C D E F G : Point)

-- Condition statements
def right_triangle (A B C : Point) : Prop := -- Definition of right triangle
  (B.y = 0 ∧ C.y = 0 ∧ C.x = 0) ∧ (A.x = 0)

axiom midpoint_D : D = midpoint B C
axiom point_E_on_AD : ∃ (λ ∈ ℝ), E = { x := λ * D.x, y := (1 - λ) * A.y }
axiom intersection_CE_AB : ∃ (F : Point), -- F is intersection of CE and AB
  (∃ λ : ℝ, E = { x := λ * D.x, y := (1 - λ) * A.y }) ∧ F = { x := ... , y := ... } -- Derived from equations
axiom perpendicular_F_to_BC_G : ∃ (G : Point), -- G is intersection of perpendicular from F to BC and BE
  ...

-- The theorem statement
theorem locus_of_G_as_E_moves :
  right_triangle A B C →
  midpoint_D →
  (∃ λ ∈ ℝ, E = { x := λ * (b / 2), y := (1 - λ) * a }) →
  intersection_CE_AB →
  perpendicular_F_to_BC_G →
  ∃ (x0 y0 : ℝ), G = (x0, y0) ∧ y0 = (a / b^2) * (x0 - b)^2 :=
sorry

end locus_of_G_as_E_moves_l237_237163


namespace find_average_of_xyz_l237_237578

variable (x y z k : ℝ)

def system_of_equations : Prop :=
  (2 * x + y - z = 26) ∧
  (x + 2 * y + z = 10) ∧
  (x - y + z = k)

theorem find_average_of_xyz (h : system_of_equations x y z k) : 
  (x + y + z) / 3 = (36 + k) / 6 :=
by sorry

end find_average_of_xyz_l237_237578


namespace ellipse_standard_equation_triangle_area_l237_237873

-- Conditions
def isCircle (x y : ℝ) := x^2 + y^2 - 2 * sqrt 2 * x = 0
def isPoint (P : ℝ × ℝ) := P = (sqrt 2, 1)
def lineEquation (k x : ℝ) := k * x + 1
def ellipseEquation (x y : ℝ) := (x^2) / 4 + (y^2) / 2 = 1

-- Prove the standard equation of the ellipse
theorem ellipse_standard_equation
  (h1 : ∃ c : ℝ × ℝ, c = (sqrt 2, 0) ∧ isCircle c.1 c.2)
  (h2 : isPoint (sqrt 2, 1)) :
  (ellipseEquation (sqrt 2) 1) := 
sorry

-- Prove the area of triangle AOB
theorem triangle_area
  (h1 : ellipseEquation (sqrt 2) 1)
  (h2 : ∀ (k : ℝ) (x : ℝ), lineEquation k x = k * x + 1)
  (h3 : (∃ k : ℝ, k^2 = 1 / 14)) :
  (area_triangle : ℝ := 3 * sqrt 14 / 8) := 
sorry

end ellipse_standard_equation_triangle_area_l237_237873


namespace evaluate_expression_l237_237834

theorem evaluate_expression : 
  let a := 7
  let b := 11
  let c := 13
  in 
  (
    (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) /
    (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b))
  ) = a + b + c :=
by
  sorry

end evaluate_expression_l237_237834


namespace study_diff_avg_corrected_l237_237007

theorem study_diff_avg_corrected :
  let diffs := [12, -8, 25, 17, -15, 20, -11]
  let corrected_diffs := [2, -8, 25, 17, -15, 20, -11] ∨ [12, -8, 25, 7, -15, 20, -11] ∨ [12, -8, 25, 17, -15, 10, -11] 
  (list.sum corrected_diffs) / 7 = 4.29 :=
by
  sorry

end study_diff_avg_corrected_l237_237007


namespace quadratic_factored_b_l237_237638

theorem quadratic_factored_b (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q) ∧ m * p = 15 ∧ n * q = 30 ∧ m * q + n * p = b) ↔ b = 43 :=
by {
  sorry
}

end quadratic_factored_b_l237_237638


namespace verify_solution_l237_237104

noncomputable def y (x : ℝ) : ℝ := x * real.sqrt (1 - x^2) + x
noncomputable def y' (x : ℝ) : ℝ := real.sqrt (1 - x^2) - (x^2 / real.sqrt (1 - x^2)) + 1

theorem verify_solution : 
  ∀ (x : ℝ), ((x - x^3) * y' x + (2 * x^2 - 1) * y x - x^3 = 0) := 
by {
  intro x,
  sorry
}

end verify_solution_l237_237104


namespace find_length_of_EH_l237_237167

noncomputable def proof_problem : Prop :=
  ∀ (A B C D E F G H J : Type *)
  [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  [inhabited E] [inhabited F] [inhabited G] [inhabited H]
  [inhabited J],
  let AB := 12,
  let DE := 7,
  let GH := 4,
  exists EH : ℝ, EH = 6 ∧
  let parallel := (DE / AB) = (GH / AB) in
  (D : AC) ∧ (E G : BC) ∧ (H : AC) ∧ (AG_extends_bisects_angle_BHE).

theorem find_length_of_EH : proof_problem := by
  intros,
  sorry

end find_length_of_EH_l237_237167


namespace sin_equal_three_l237_237087

variables (a b c : ℝ)
variables (A B C : ℝ) -- Angles in triangle

noncomputable def ellipse := ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity := c / a = 1/3
def foci_A := (-c, 0)
def foci_B := (c, 0)
def point_on_ellipse (x y : ℝ) := (x, y) ≠ (a, 0) ∧ (x, y) ≠ (-a, 0)
def distances (AC BC AB : ℝ) := AC + BC = 2 * a ∧ AB = 2 * c

theorem sin_equal_three (h : ∃ (AC BC AB : ℝ), ellipse a b ∧ eccentricity ∧ distances AC BC AB) :
  (sin A + sin B) / sin C = 3 :=
sorry

end sin_equal_three_l237_237087


namespace min_concerts_l237_237730

theorem min_concerts (n : ℕ) (h1 : n = 8) :
  ∃ m : ℕ, (∀ (S : ℕ),
    (∀ i j : fin n, i ≠ j →
      ∃ m : ℕ, (∀ k l : fin n, k ≠ l → occurrences_of_pair k l m = S)) →
    S = 3) → m = 14 :=
by
  sorry

end min_concerts_l237_237730


namespace part1_part2_l237_237119

open_locale real

def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x)
def b (x : ℝ) : ℝ × ℝ := (cos x, sin x)

-- (1)
theorem part1 (x : ℝ) (h : x ∈ Icc (π/2) π) (h_ab : euclidean_dist (a x) (b x) = 2) : 
x = 2 * π / 3 := sorry

-- (2)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem part2 (x : ℝ) (h : x ∈ Icc (π/2) π) : 
f x ∈ Icc (-1/2) 1 := sorry

end part1_part2_l237_237119


namespace greatest_third_side_l237_237699

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l237_237699


namespace Tim_apartment_complexes_l237_237657

theorem Tim_apartment_complexes 
  (keys_per_lock : ℕ) 
  (total_keys : ℕ) 
  (apartments_per_complex : ℕ) 
  (h1 : keys_per_lock = 3) 
  (h2 : total_keys = 72) 
  (h3 : apartments_per_complex = 12) : 
  total_keys / keys_per_lock / apartments_per_complex = 2 := 
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end Tim_apartment_complexes_l237_237657


namespace a4_is_15_l237_237460

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 1

theorem a4_is_15 (a : ℕ → ℕ) (h : sequence a) : a 4 = 15 :=
by
  sorry

end a4_is_15_l237_237460


namespace problem_statement_l237_237565

def f(x : ℝ) : ℝ := (4 * x^2 + 6 * x + 9) / (x^2 - x + 4)
def g(x : ℝ) : ℝ := x^2 - 2 * x + 1

theorem problem_statement : f(g(2)) + g(f(2)) = 119 / 12 := by
  sorry

end problem_statement_l237_237565


namespace triangle_divisible_into_2019_bicentric_quadrilaterals_l237_237595

def is_bicentric (q : Quadrilateral) : Prop :=
  q.inscribed ∧ q.circumscribed

def can_divide_into_bicentric_quadrilaterals (triangle : Triangle) (n : ℕ) : Prop :=
  ∃ (quadrilaterals : List Quadrilateral), quadrilaterals.length = n ∧ 
    (∀ q ∈ quadrilaterals, is_bicentric q) ∧
    (triangle.area = quadrilaterals.sum (λ q, q.area))

theorem triangle_divisible_into_2019_bicentric_quadrilaterals (triangle : Triangle) : 
  can_divide_into_bicentric_quadrilaterals triangle 2019 :=
sorry

end triangle_divisible_into_2019_bicentric_quadrilaterals_l237_237595


namespace a_4_eq_5_div_3_l237_237539

/-- Sequence definition -/
def a : ℕ → ℚ
| 0       := 1
| (n + 1) := 1 + 1 / a n

theorem a_4_eq_5_div_3 : a 3 = 5 / 3 := 
by sorry

end a_4_eq_5_div_3_l237_237539


namespace percentage_discount_four_friends_l237_237349

theorem percentage_discount_four_friends 
  (num_friends : ℕ)
  (original_price : ℝ)
  (total_spent : ℝ)
  (item_per_friend : ℕ)
  (total_items : ℕ)
  (each_spent : ℝ)
  (discount_percentage : ℝ):
  num_friends = 4 →
  original_price = 20 →
  total_spent = 40 →
  item_per_friend = 1 →
  total_items = num_friends * item_per_friend →
  each_spent = total_spent / num_friends →
  discount_percentage = ((original_price - each_spent) / original_price) * 100 →
  discount_percentage = 50 :=
by
  sorry

end percentage_discount_four_friends_l237_237349


namespace unique_pairs_pos_int_satisfy_eq_l237_237842

theorem unique_pairs_pos_int_satisfy_eq (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) := 
by
  sorry

end unique_pairs_pos_int_satisfy_eq_l237_237842


namespace magnitude_of_z_l237_237955

-- Define the complex number z and the condition
def z : ℂ := 1 + 2 * Complex.i + Complex.i ^ 3

-- The main theorem stating the magnitude of z
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l237_237955


namespace sequence_contains_even_number_l237_237789

theorem sequence_contains_even_number (x : ℕ → ℕ) 
  (h_infinite : ∀ n, ∃ m, n ≤ m)
  (h_next_term : ∀ n, ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ x (n+1) = x n + d) : 
  ∃ n, even (x n) :=
sorry

end sequence_contains_even_number_l237_237789


namespace students_facing_teacher_l237_237429

theorem students_facing_teacher : 
  let N := 50
  let multiples_of_3 := Nat.floorDiv N 3 
  let multiples_of_7 := Nat.floorDiv N 7 
  let multiples_of_21 := Nat.floorDiv N 21 
  let turned_around_once := (multiples_of_3 - multiples_of_21) + (multiples_of_7 - multiples_of_21)
  let facing_teacher := N - turned_around_once
  facing_teacher = 31 := 
by 
  sorry

end students_facing_teacher_l237_237429


namespace sum_of_divisors_of_45_l237_237308

theorem sum_of_divisors_of_45 : (∑ d in (Finset.filter (λ x, 45 % x = 0) (Finset.range 46)), d) = 78 :=
by
  -- We'll need to provide the proof here
  sorry

end sum_of_divisors_of_45_l237_237308


namespace inscribed_regular_polygon_sides_l237_237535

theorem inscribed_regular_polygon_sides (n : ℕ) (h_central_angle : 360 / n = 72) : n = 5 :=
by
  sorry

end inscribed_regular_polygon_sides_l237_237535


namespace boat_distance_against_stream_in_one_hour_l237_237976

-- Define the conditions
def speed_in_still_water : ℝ := 4 -- speed of the boat in still water (km/hr)
def downstream_distance_in_one_hour : ℝ := 6 -- distance traveled along the stream in one hour (km)

-- Define the function to compute the speed of the stream
def speed_of_stream (downstream_distance : ℝ) (boat_speed_still_water : ℝ) : ℝ :=
  downstream_distance - boat_speed_still_water

-- Define the effective speed against the stream
def effective_speed_against_stream (boat_speed_still_water : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed_still_water - stream_speed

-- Prove that the boat travels 2 km against the stream in one hour given the conditions
theorem boat_distance_against_stream_in_one_hour :
  effective_speed_against_stream speed_in_still_water (speed_of_stream downstream_distance_in_one_hour speed_in_still_water) * 1 = 2 := 
by
  sorry

end boat_distance_against_stream_in_one_hour_l237_237976


namespace equation_of_lines_l237_237843

theorem equation_of_lines :
  (∃ l : ℝ → ℝ → Prop, 
  ((∀ x y : ℝ, l x y ↔ y = (1/3) * (x + 4) ∨ y = -(1/3) * (x + 4) ∨ x = 5 ∨ (y = (3/4) * x - 25/4)) ∧ 
  (l -4 0) ∧ 
  (l 5 10))) 
  := sorry

end equation_of_lines_l237_237843


namespace probability_of_successful_meeting_l237_237736

noncomputable def successful_meeting_probability : ℝ :=
  let volume_hypercube := 16.0
  let volume_pyramid := (1.0/3.0) * 2.0^3 * 2.0
  let volume_reduced_base := volume_pyramid / 4.0
  let successful_meeting_volume := volume_reduced_base
  successful_meeting_volume / volume_hypercube

theorem probability_of_successful_meeting : successful_meeting_probability = 1 / 12 :=
  sorry

end probability_of_successful_meeting_l237_237736


namespace find_d_l237_237938

-- Conditions
variables (c d : ℝ)
axiom ratio_cond : c / d = 4
axiom eq_cond : c = 20 - 6 * d

theorem find_d : d = 2 :=
by
  sorry

end find_d_l237_237938


namespace smallest_positive_integer_n_l237_237347

theorem smallest_positive_integer_n (n : ℕ) :
  (∃ n1 n2 n3 : ℕ, 5 * n = n1 ^ 5 ∧ 6 * n = n2 ^ 6 ∧ 7 * n = n3 ^ 7) →
  n = 2^5 * 3^5 * 5^4 * 7^6 :=
by
  sorry

end smallest_positive_integer_n_l237_237347


namespace determinant_property_l237_237880

variable {R : Type} [CommRing R]
variable (x y z w : R)

theorem determinant_property 
  (h : x * w - y * z = 7) :
  (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by sorry

end determinant_property_l237_237880


namespace F_is_even_G_is_odd_g_value_l237_237033

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x + f (-x)) / 2
def G (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x - f (-x)) / 2

theorem F_is_even (f : ℝ → ℝ) : is_even (F f) :=
by sorry

theorem G_is_odd (f : ℝ → ℝ) : is_odd (G f) :=
by sorry

def f (x : ℝ) : ℝ := Real.log (Real.exp x + 1)
def g (x : ℝ) : ℝ := (f x - f (-x)) / 2

theorem g_value : ∀ x, g x = x / 2 :=
by sorry

end F_is_even_G_is_odd_g_value_l237_237033


namespace range_of_k_l237_237541

variable (x k : ℝ)

def operation (x y : ℝ) : ℝ :=
  x * (1 - y)

axiom exists_distinct_x1_x2 (k : ℝ) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation 1 (2 * k - 3 - k * x) = 1 + real.sqrt (4 - x^2)

theorem range_of_k :
  ∃ k : ℝ, (5/12) < k ∧ k ≤ (3/4) :=
sorry

end range_of_k_l237_237541


namespace simplify_expr_l237_237238

theorem simplify_expr :
  (4 + 2 * Complex.i) / (4 - 2 * Complex.i) + (4 - 6 * Complex.i) / (4 + 6 * Complex.i) =
  (14 / 65) + (-8 / 65) * Complex.i :=
by
  sorry

end simplify_expr_l237_237238


namespace probability_multiple_of_3_l237_237022

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

theorem probability_multiple_of_3 :
  (∃ n, 10000 ≤ n ∧ n < 100000 ∧ is_multiple_of_3 n) → 
  (prob_of (λ n, 10000 ≤ n ∧ n < 100000) (is_multiple_of_3) = 1/3) :=
sorry

end probability_multiple_of_3_l237_237022


namespace tree_height_is_6_point_4_l237_237380

noncomputable def treeHeight
    (AC : ℝ) (AD : ℝ) (DE : ℝ)
    (h_similar : ∀ ABC DEC, ABC ∼ DEC) : ℝ :=
by
  have DC := AC - AD
  have h_AB := (DE / DC) * AC
  exact h_AB

theorem tree_height_is_6_point_4 :
    treeHeight 4 3 1.6 (λ ABC DEC, sorry) = 6.4 :=
sorry

end tree_height_is_6_point_4_l237_237380


namespace prob_student_C_first_l237_237967

variables (students : Set String) [DecidableEq String] 

def not_first (student : String) :=
  ¬(student = "first")

def not_last (student : String) :=
  ¬(student = "last")

-- Assume there are five students: A, B, C, D, E
variables (students : List String) (A B C D E : String)
#eval students = ["A", "B", "C", "D", "E"]
#eval A = "A"
#eval B = "B"
#eval C = "C"
#eval D = "D"
#eval E = "E"

-- Theorem statement
theorem prob_student_C_first :
  (not_first A) →
  (not_first B) →
  (not_last A) →
  (not_last B) →
  (students.contains C) →
  (student ∈ students → student ≠ "first" → student ≠ "last") →
  (students.countP (λ s, s = "first") = 1) →
  (students.countP (λ s, s ≠ "first" ∧ s ≠ "last") = 3) →
  (students.countP (λ s, s ≠ "first" ∧ s = "C") = 1) →
  probability C first = 1 / 3 :=
begin
  sorry
end

end prob_student_C_first_l237_237967


namespace probability_of_odd_sum_gt_10_is_5_over_27_l237_237174

noncomputable def probability_odd_sum_gt_10 : ℚ :=
  let P : Finset ℕ := {1, 4, 3}
  let Q : Finset ℕ := {2, 4, 6}
  let R : Finset ℕ := {1, 3, 5}
  let S : Finset ℕ := {2, 3, 8}
  let is_odd (n : ℕ) := n % 2 = 1
  let is_sum_gt_10 (x y z w : ℕ) := x + y + z + w > 10
  let successful_outcomes := { 
    (x, y, z, w) ∈ P.product (Q.product (R.product S)) |
      is_odd (x + y + z + w) && is_sum_gt_10 x y z w 
  }
  let total_outcomes := P.card * Q.card * R.card * S.card
  (successful_outcomes.card : ℚ) / total_outcomes

theorem probability_of_odd_sum_gt_10_is_5_over_27 :
  probability_odd_sum_gt_10 = 5 / 27 := 
sorry

end probability_of_odd_sum_gt_10_is_5_over_27_l237_237174


namespace arithmetic_sequence_sum_l237_237978

noncomputable def a_n (n : ℕ) : ℕ :=
  n

noncomputable def b_n (n : ℕ) : ℕ :=
  2 ^ (a_n n)

noncomputable def S_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (b_n (i + 1))

theorem arithmetic_sequence_sum (n : ℕ) :
  S_n n = 2 * (2 ^ n - 1) :=
by
  sorry

end arithmetic_sequence_sum_l237_237978


namespace greatest_possible_third_side_l237_237676

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l237_237676


namespace equation_solution_is_log3_7_l237_237642

noncomputable def solve_equation : ℝ :=
  if h : (∃ x : ℝ, 9^x - 6 * 3^x - 7 = 0) then
    Classical.choose h
  else
    0

theorem equation_solution_is_log3_7 :
  solve_equation = Real.log 7 / Real.log 3 :=
by
  sorry

end equation_solution_is_log3_7_l237_237642


namespace maxwell_walking_speed_l237_237583

open Real

theorem maxwell_walking_speed (v : ℝ) : 
  (∀ (v : ℝ), (4 * v + 6 * 3 = 34)) → v = 4 :=
by
  intros
  have h1 : 4 * v + 18 = 34 := by sorry
  have h2 : 4 * v = 16 := by sorry
  have h3 : v = 4 := by sorry
  exact h3

end maxwell_walking_speed_l237_237583


namespace multiplicative_inverse_of_151_mod_257_l237_237027

theorem multiplicative_inverse_of_151_mod_257 :
  ∃ b : ℤ, 0 ≤ b ∧ b ≤ 256 ∧ (151 * b ≡ 1 [MOD 257]) ∧ b = 153 :=
by
  use 153
  split
  norm_num
  split
  norm_num
  split
  norm_num
  exact (by norm_num : 151 * 153 ≡ 1 [MOD 257])
  exact rfl

end multiplicative_inverse_of_151_mod_257_l237_237027


namespace part_a_part_b_l237_237650

-- Definitions for maximum factor increases
def f (n : ℕ) (a : ℕ) : ℚ := sorry
def t (n : ℕ) (a : ℕ) : ℚ := sorry

-- Part (a): Prove the factor increase for exactly 1 blue cube in 100 boxes
theorem part_a : f 100 1 = 2^100 / 100 := sorry

-- Part (b): Prove the factor increase for some integer \( k \) blue cubes in 100 boxes, \( 1 < k \leq 100 \)
theorem part_b (k : ℕ) (hk : 1 < k ∧ k ≤ 100) : t 100 k = 2^100 / (2^100 - k - 1) := sorry

end part_a_part_b_l237_237650


namespace student_thought_six_is_seven_l237_237799

theorem student_thought_six_is_seven
  (n : ℕ → ℕ)
  (h1 : (n 1 + n 3) / 2 = 2)
  (h2 : (n 2 + n 4) / 2 = 3)
  (h3 : (n 3 + n 5) / 2 = 4)
  (h4 : (n 4 + n 6) / 2 = 5)
  (h5 : (n 5 + n 7) / 2 = 6)
  (h6 : (n 6 + n 8) / 2 = 7)
  (h7 : (n 7 + n 9) / 2 = 8)
  (h8 : (n 8 + n 10) / 2 = 9)
  (h9 : (n 9 + n 1) / 2 = 10)
  (h10 : (n 10 + n 2) / 2 = 1) : 
  n 6 = 7 := 
  sorry

end student_thought_six_is_seven_l237_237799


namespace sum_k1_k2_k3_l237_237506

theorem sum_k1_k2_k3 :
  ∀ (k1 k2 k3 t1 t2 t3 : ℝ),
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  t1 = (5 / 9) * (k1 - 32) →
  t2 = (5 / 9) * (k2 - 32) →
  t3 = (5 / 9) * (k3 - 32) →
  k1 + k2 + k3 = 510 :=
by
  intros k1 k2 k3 t1 t2 t3 ht1 ht2 ht3 ht1k1 ht2k2 ht3k3
  sorry

end sum_k1_k2_k3_l237_237506


namespace slope_constant_on_ellipse_l237_237086

noncomputable def ellipse : Prop :=
  ∃ (E F : ℝ × ℝ), 
    let A : ℝ × ℝ := (1, 3/2) in
    let f1 : ℝ × ℝ := (-1, 0) in
    let f2 : ℝ × ℝ := (1, 0) in
    -- Condition 1: Equation of ellipse
    (E.1^2 / 4 + E.2^2 / 3 = 1) ∧
    (F.1^2 / 4 + F.2^2 / 3 = 1) ∧
    -- Condition 2 & 3 enforced
    ((E = A ∨ F = A) ∧
    E ≠ F) ∧
    -- Slope conditions
    ((E.2 - A.2) / (E.1 - A.1) = -(1 / ((F.2 - A.2) / (F.1 - A.1))) ∧
    -- Conclusion: Constant slope of EF
    (E ≠ F → (F.2 - E.2) / (F.1 - E.1) = 1 / 2))

-- Theorem to be proved
theorem slope_constant_on_ellipse (h : ellipse) : true :=
sorry

end slope_constant_on_ellipse_l237_237086


namespace find_c_l237_237887

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem find_c (a b : ℝ) (h_range : (∀ x: ℝ, 0 ≤ f x a b)) (m : ℝ)
  (h_sol_set : ∀ x, (m < x ∧ x < m + 6) ↔ f x a b < 9) :
  ∃ c, c = 9 :=
by
  use 9
  sorry

end find_c_l237_237887


namespace sum_of_solutions_l237_237715

theorem sum_of_solutions : 
  let f := λ x : ℝ => (4 * x + 7) * (3 * x - 10) in
  let solutions := {x | f x = 0} in
  (∃ s1 s2 : ℝ, s1 < s2 ∧ s1 ∈ solutions ∧ s2 ∈ solutions ∧ 
  s1 + s2 = 19 / 12) :=
begin
  let f := λ x : ℝ => (4 * x + 7) * (3 * x - 10),
  let solutions := {x | f x = 0},
  have H1 : (4 * (-7 / 4) + 7) * (3 * (-7 / 4) - 10) = 0, by sorry,
  have H2 : (4 * (10 / 3) + 7) * (3 * (10 / 3) - 10) = 0, by sorry,
  use [-7 / 4, 10 / 3],
  split,
  { norm_num },
  split,
  { exact H1 },
  split,
  { exact H2 },
  norm_num
end

end sum_of_solutions_l237_237715


namespace expected_profit_calculation_l237_237971

theorem expected_profit_calculation:
  let odd1 := 1.28
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let initial_bet := 5.00
  let total_payout := initial_bet * (odd1 * odd2 * odd3 * odd4)
  let expected_profit := total_payout - initial_bet
  expected_profit = 212.822 := by
  sorry

end expected_profit_calculation_l237_237971


namespace exponents_sum_correct_l237_237714

def sum_of_exponents (a b c : ℕ) : ℕ :=
  let radicand := 40 * a^6 * b^8 * c^{14}
  let cube_root := (8 * a^6 * b^6 * c^{12}) * 5 * b^2 * c^2
  let outside_radical := 2 * a^2 * b^2 * c^4
  2 + 2 + 4

theorem exponents_sum_correct (a b c : ℕ) : sum_of_exponents a b c = 8 := by
  sorry

end exponents_sum_correct_l237_237714


namespace nth_inequality_l237_237223

theorem nth_inequality (n : ℕ) : 
  (∑ i in Finset.range n, Real.sqrt (i + 1) * Real.sqrt (i + 2)) < (n * (n + 2)) / 2 := 
sorry

end nth_inequality_l237_237223


namespace radius_of_sphere_A_l237_237609

def sphere_surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

theorem radius_of_sphere_A (r_A r_B : ℝ) (h1 : r_B = 10)
  (h2 : sphere_surface_area r_A / sphere_surface_area r_B = 16) : 
  r_A = 40 := 
by
  sorry

end radius_of_sphere_A_l237_237609


namespace problem_proof_l237_237196

-- Assume definitions for lines and planes, and their relationships like parallel and perpendicular exist.

variables (m n : Line) (α β : Plane)

-- Define conditions
def line_is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_is_parallel_to_line (l1 l2 : Line) : Prop := sorry
def planes_are_perpendicular (p1 p2 : Plane) : Prop := sorry

-- Problem statement
theorem problem_proof :
  (line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n α) → 
  (line_is_parallel_to_line m n) ∧
  ((line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n β) ∧ (line_is_perpendicular_to_plane m n) → 
  (planes_are_perpendicular α β)) := 
sorry

end problem_proof_l237_237196


namespace gameEndsIn27Rounds_l237_237520

-- Define the initial conditions and rules of the game
structure GameState where
  A : ℕ
  B : ℕ
  C : ℕ
  rounds : ℕ

def initialState : GameState := {A := 10, B := 9, C := 8, rounds := 0}

-- Function to update the game state according to the rules
def updateState (s : GameState) : GameState :=
  let discard := 2
  let gain := 1
  let (A, B, C) := (s.A, s.B, s.C)
  if A >= B ∧ A >= C then {A := A - (gain * 2 + discard), B := B + gain, C := C + gain, rounds := s.rounds + 1}
  else if B >= A ∧ B >= C then {A := A + gain, B := B - (gain * 2 + discard), C := C + gain, rounds := s.rounds + 1}
  else {A := A + gain, B := B + gain, C := C - (gain * 2 + discard), rounds := s.rounds + 1}

-- Function to check if the game has ended
def gameEnded (s : GameState) : Prop :=
  s.A = 0 ∨ s.B = 0 ∨ s.C = 0

-- Function to simulate the game
def simulateGame (s : GameState) : GameState :=
  if gameEnded s then s else simulateGame (updateState s)

-- Main theorem statement
theorem gameEndsIn27Rounds : simulateGame initialState.rounds = 27 :=
  by
  sorry

end gameEndsIn27Rounds_l237_237520


namespace repair_cost_l237_237739

theorem repair_cost
  (R : ℝ) -- R is the cost to repair the used shoes
  (new_shoes_cost : ℝ := 30) -- New shoes cost $30.00
  (new_shoes_lifetime : ℝ := 2) -- New shoes last for 2 years
  (percentage_increase : ℝ := 42.857142857142854) 
  (h1 : new_shoes_cost / new_shoes_lifetime = R + (percentage_increase / 100) * R) :
  R = 10.50 :=
by
  sorry

end repair_cost_l237_237739


namespace fraction_even_odd_phonenumbers_l237_237798

-- Define a predicate for valid phone numbers
def isValidPhoneNumber (n : Nat) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ (n / 1000000 ≠ 0) ∧ (n / 1000000 ≠ 1)

-- Calculate the total number of valid phone numbers
def totalValidPhoneNumbers : Nat :=
  4 * 10^6

-- Calculate the number of valid phone numbers that begin with an even digit and end with an odd digit
def validEvenOddPhoneNumbers : Nat :=
  4 * (10^5) * 5

-- Determine the fraction of such phone numbers (valid ones and valid even-odd ones)
theorem fraction_even_odd_phonenumbers : 
  (validEvenOddPhoneNumbers) / (totalValidPhoneNumbers) = 1 / 2 :=
by {
  sorry
}

end fraction_even_odd_phonenumbers_l237_237798


namespace abs_diff_arith_seq_l237_237303

theorem abs_diff_arith_seq :
  let d := 11
  let a := -10
  let a_n (n : ℕ) := a + (n - 1) * d
  let a_2010 := a_n 2010
  let a_2025 := a_n 2025
  abs (a_2025 - a_2010) = 165 :=
by
  let d := 11
  let a := -10
  let a_n := λ (n : ℕ), a + (n - 1) * d
  let a_2010 := a_n 2010
  let a_2025 := a_n 2025
  show abs (a_2025 - a_2010) = 165
  sorry

end abs_diff_arith_seq_l237_237303


namespace minimize_distance_l237_237461

noncomputable def minimize_sum_distances (P Q R : ℝ × ℝ) := 
  let PR := dist P R
  let RQ := dist R Q
  PR + RQ

theorem minimize_distance 
  (P Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (hP : P = (-2 : ℝ, -3 : ℝ))
  (hQ : Q = (5 : ℝ, 3 : ℝ))
  (hR : R = (2 : ℝ, m)) :
  m = (3/7 : ℝ) :=
sorry

end minimize_distance_l237_237461


namespace function_symmetry_implies_even_l237_237133

theorem function_symmetry_implies_even (f : ℝ → ℝ) (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ x y, f x = y ↔ -f (-x) = -y) : ∀ x, f x = f (-x) :=
by
  sorry

end function_symmetry_implies_even_l237_237133


namespace find_BC_length_l237_237225

-- Define the points and the associated geometric properties
variables {A B C D E M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace M]

-- The problem constraints
variables (rectangle : ∀ (A B C D : Type), is_rectangle A B C D)
variables (on_side : ∀ (A D E : Type), point_on_side E A D)
variables (on_segment : ∀ (E C M : Type), point_on_segment M E C)
variables (AB_BM : ∀ (A B M : Type), distance A B = distance B M)
variables (AE_EM : ∀ (A E M : Type), distance A E = distance E M)
variables (ED_value : ∀ (E D : Type), distance E D = 16)
variables (CD_value : ∀ (C D : Type), distance C D = 12)

-- The conclusion
theorem find_BC_length : ∀ (B C : Type), is_rectangle A B C D ∧ point_on_side E A D ∧ point_on_segment M E C ∧ 
  (distance A B = distance B M) ∧ (distance A E = distance E M) ∧ (distance E D = 16) ∧ (distance C D = 12) →
  distance B C = 20 :=
by
  sorry

end find_BC_length_l237_237225


namespace count_boys_correct_l237_237443

def total_vans : ℕ := 5
def students_per_van : ℕ := 28
def number_of_girls : ℕ := 80

theorem count_boys_correct : 
  (total_vans * students_per_van) - number_of_girls = 60 := 
by
  sorry

end count_boys_correct_l237_237443


namespace compound_interest_years_l237_237265

noncomputable def log {A : Type*} [log α] (a : α) : α → α := sorry

theorem compound_interest_years (PV FV : ℝ) (r : ℝ) (n : ℝ) : 
  PV = 156.25 → FV = 169 → r = 0.04 → 
  n = (log ((FV) / (PV)) / log (1 + r)) → 
  n = 2 := by 
    intros hPV hFV hr hn
    rw [hPV, hFV, hr] at hn
    sorry

end compound_interest_years_l237_237265


namespace patches_overlap_min_area_l237_237139

theorem patches_overlap_min_area {X : Type} [measurable_space X] {μ : measure_theory.measure X} 
  (hX : μ set.univ = 1) (patches : fin 5 → set X) 
  (h_patches_area : ∀ i, μ (patches i) ≥ 1/2) :
  ∃ (i j : fin 5), i ≠ j ∧ μ (patches i ∩ patches j) ≥ 1/5 :=
begin
  sorry
end

end patches_overlap_min_area_l237_237139


namespace problem1_l237_237345

theorem problem1 (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) := by 
  sorry

end problem1_l237_237345


namespace E_expansion_Δ_expansion_l237_237279

noncomputable def E : (ℕ → ℝ) → (ℕ → ℝ) :=
  fun f x => f (x + 1)

noncomputable def Δ : (ℕ → ℝ) → (ℕ → ℝ) :=
  fun f x => f (x + 1) - f x

-- Conditions
lemma Δ_constant (C : ℝ) : Δ (fun _ => C) = (fun _ => 0) :=
sorry

lemma Δ_linear (f g : ℕ → ℝ) (λ μ : ℝ) : Δ (fun x => λ * f x + μ * g x) = (fun x => λ * Δ f x + μ * Δ g x) :=
sorry

lemma Δ_product (f g : ℕ → ℝ) : Δ (fun x => f x * g x) = 
  (fun x => f x * Δ g x + g (x + 1) * Δ f x) :=
sorry

theorem E_expansion (f : ℕ → ℝ) (n : ℕ) : 
  (E^[n] f) = (fun x => ∑ k in finset.range n.succ, (nat.choose n k) * (Δ^[k] f x)) :=
sorry

theorem Δ_expansion (f : ℕ → ℝ) (n : ℕ) :
  (Δ^[n] f) = (fun x => ∑ k in finset.range n.succ, (-1)^(n - k) * (nat.choose n k) * (E^[k] f) x) :=
sorry

end E_expansion_Δ_expansion_l237_237279


namespace square_area_below_parabola_l237_237765

theorem square_area_below_parabola :
  ∀ (s : ℝ), 
  (∀ (x : ℝ), x ∈ {x : ℝ | x^2 - 6*x + 5 ≥ 0}) → 
  s = 3 / 2 →
  (2*s)^2 = 9 :=
by 
  intros s h x_eq.
  sorry

end square_area_below_parabola_l237_237765


namespace number_of_primes_diffs_in_set_l237_237915

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def candidates : List ℕ := List.range' 7 10 4

def num_of_numbers_can_be_written_as_diff_of_two_primes : ℕ :=
  candidates.countp (λ x => is_prime (x + 2))

theorem number_of_primes_diffs_in_set : num_of_numbers_can_be_written_as_diff_of_two_primes = 2 := 
sorry

end number_of_primes_diffs_in_set_l237_237915


namespace complement_of_A_l237_237512

variables (U : Set ℝ) (A : Set ℝ)
def universal_set : Prop := U = Set.univ
def range_of_function : Prop := A = {x : ℝ | 0 ≤ x}

theorem complement_of_A (hU : universal_set U) (hA : range_of_function A) : 
  U \ A = {x : ℝ | x < 0} :=
by 
  sorry

end complement_of_A_l237_237512


namespace evaluate_expression_l237_237835

theorem evaluate_expression : 
  let a := 7
  let b := 11
  let c := 13
  in 
  (
    (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) /
    (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b))
  ) = a + b + c :=
by
  sorry

end evaluate_expression_l237_237835


namespace B_cycling_speed_l237_237011

-- Define constants
def speed_A : ℝ := 10  -- A's walking speed in kmph
def distance_catchup : ℝ := 120  -- Distance where B catches up with A in km
def time_diff : ℝ := 6  -- Time difference between when A starts and B starts in hours

-- Calculate B's cycling speed
theorem B_cycling_speed : ∃ (speed_B : ℝ), speed_B = 20 :=
by
  -- A's travel time until catch-up
  let time_A := distance_catchup / speed_A
  -- B's cycling time
  let time_B := time_A - time_diff
  -- B's cycling speed
  let speed_B := distance_catchup / time_B
  -- Prove B's cycling speed is 20 kmph
  use speed_B
  have h1 : time_A = 12 := by norm_num
  have h2 : time_B = 6 := by norm_num
  have h3 : speed_B = 20 := by norm_num
  exact h3

end B_cycling_speed_l237_237011


namespace supercomputer_process_never_stops_l237_237773

theorem supercomputer_process_never_stops (n : ℕ) (hn : n = 10^899 + 10^898 + ... + 10 + 1) :
  ∀ m, m = n → let process (k : ℕ) := 2 * (k / 100) + 8 * (k % 100) in
  ∀ k', k' ≠ 0 → m = process k' → k' ≥ 100 :=
begin
  sorry -- proof is not required as per the instructions
end

end supercomputer_process_never_stops_l237_237773


namespace median_adjusted_scores_higher_l237_237143

namespace ScoreAdjustment

variable (n : ℕ) (x : Fin n → ℝ) (y : Fin n → ℝ) (a b : ℝ)

-- Conditions
def total_students : ℕ := 40
def avg_score (x : Fin total_students → ℝ) : ℝ := 70
def highest_score (x : Fin total_students → ℝ) : ℝ := 100
def lowest_score (x : Fin total_students → ℝ) : ℝ := 50
def adjustment_formula (a b : ℝ) := ∀ i, (a > 0) ∧ (y i = a * x i + b)
def highest_adjusted_score (y : Fin total_students → ℝ) : ℝ := 100
def lowest_adjusted_score (y : Fin total_students → ℝ) : ℝ := 60

-- Theorem
theorem median_adjusted_scores_higher
  (h_count : n = total_students)
  (h_avg : avg_score x = 70)
  (h_max : ∀ i, x i ≤ 100 ∧ ∃ j, x j = 100)
  (h_min : ∀ i, x i ≥ 50 ∧ ∃ j, x j = 50)
  (h_adj : adjustment_formula a b)
  (h_max_adj : ∀ i, y i ≤ 100 ∧ ∃ j, y j = 100)
  (h_min_adj : ∀ i, y i ≥ 60 ∧ ∃ j, y j = 60)
  : median y > median x := sorry

end ScoreAdjustment

end median_adjusted_scores_higher_l237_237143


namespace greatest_third_side_l237_237675

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l237_237675


namespace evaluate_expression_l237_237717

theorem evaluate_expression :
  2 ^ (0 ^ (1 ^ 9)) + ((2 ^ 0) ^ 1) ^ 9 = 2 := 
sorry

end evaluate_expression_l237_237717


namespace ab_greater_than_a_plus_b_l237_237593

variable {a b : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (h : a - b = a / b)

theorem ab_greater_than_a_plus_b : a * b > a + b :=
sorry

end ab_greater_than_a_plus_b_l237_237593


namespace area_of_transformed_region_is_correct_l237_237561

-- Define the transformation matrix
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 2], ![4, -1]]

-- Area of the initial region T
def area_T : ℝ := 15

-- Area of the transformed region T'
def area_T' : ℝ :=
  let determinant := abs (det transformation_matrix)
  determinant * area_T

-- The proof statement, verifying the area of T' is 165
theorem area_of_transformed_region_is_correct : area_T' = 165 :=
by
  -- The proof will go here
  sorry

end area_of_transformed_region_is_correct_l237_237561


namespace problem_B_problem_D_l237_237194

/-
  Given distinct lines m, n and distinct planes α, β,
  we want to prove the following two statements:
  
  1. If m is perpendicular to α and n is perpendicular to α, then m is parallel to n.
  2. If m is perpendicular to α, n is perpendicular to β, and m is perpendicular to n, then α is perpendicular to β.
-/

variables {m n : Type} -- Types representing distinct lines
variables {α β : Type} -- Types representing distinct planes

-- Hypotheses for the statements
variable [linear_order m n α β] -- Assume we have a linear ordering for the geometric entities

-- Define helper functions for parallelism and perpendicularity
def is_parallel (x y : Type) : Prop := sorry
def is_perpendicular (x y : Type) : Prop := sorry

-- Statement for problem B
theorem problem_B (h1 : is_perpendicular m α) (h2 : is_perpendicular n α) : is_parallel m n :=
  sorry

-- Statement for problem D
theorem problem_D (h1 : is_perpendicular m α) (h2 : is_perpendicular m n) (h3 : is_perpendicular n β) : is_perpendicular α β :=
  sorry

end problem_B_problem_D_l237_237194


namespace problem1_problem2_l237_237734

-- Problem 1: Set intersection with complement
def A : set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : set ℝ := {x | 2 < x ∧ x < 10}
def complement_A : set ℝ := {x | x < 3 ∨ x ≥ 7}
def expected_set : set ℝ := {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}

theorem problem1 : (complement_A ∩ B) = expected_set := by
  sorry

-- Problem 2: Logarithmic conditions
def log_set (m : ℝ) : set ℝ := {x | x = Real.logb 2 m}
def B_subset : set ℝ := {1, 2}

theorem problem2 (m : ℝ) (hB : log_set m ⊆ B_subset) : m = 2 ∨ m = 4 := by
  sorry

end problem1_problem2_l237_237734


namespace number_of_people_in_each_van_l237_237363

/-- A group of science students went on a field trip. They took 2 vans and 3 buses. 
There were 20 people in each bus and a total of 76 people went on the field trip. 
Prove that there were 8 people in each van. -/
theorem number_of_people_in_each_van : 
  ∀ (V : ℕ),
    (number_of_vans = 2) → 
    (number_of_buses = 3) → 
    (people_per_bus = 20) → 
    (total_people = 76) → 
    (total_people = 2 * V + number_of_buses * people_per_bus) → 
    V = 8 :=
by
  intros V _ _ _ _ h
  have h1 : 76 = 2 * V + 3 * 20 := h
  have h2 : 76 = 2 * V + 60 := by rw [h1]
  have h3 : 2 * V = 16 := by linarith
  have h4 : V = 8 := by linarith [h3]
  exact h4

end number_of_people_in_each_van_l237_237363


namespace count_8_digit_numbers_l237_237124

theorem count_8_digit_numbers : 
  (∃ n : ℕ, n = 90_000_000) ↔ 
  (∃ L : Fin 9 → Fin 10, ∀ i : Fin 9, L i ≠ 0) :=
begin
  sorry
end

end count_8_digit_numbers_l237_237124


namespace count_valid_c_l237_237063

theorem count_valid_c : 
  set.counts { c | c ∈ set.Icc 0 500 ∧ (∃ x, 5 * x.floor + 3 * x.ceiling = c) } = 126 := sorry

end count_valid_c_l237_237063


namespace cube_volume_l237_237646

theorem cube_volume (s : ℝ) (hs : 12 * s = 96) : s^3 = 512 := by
  have s_eq : s = 8 := by
    linarith
  rw s_eq
  norm_num

end cube_volume_l237_237646


namespace chord_through_point_has_fixed_sum_l237_237284

theorem chord_through_point_has_fixed_sum (d : ℝ) (h1 : 0 < d) (h2 : d < 1) :
  ∀ (A B : ℝ × ℝ), A ≠ B ∧ A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ ∃ m : ℝ, A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d →
  let AC := (A.1^2 + (A.2 - d)^2).sqrt,
      BC := (B.1^2 + (B.2 - d)^2).sqrt in
  (1 / AC + 1 / BC) = 2 / ((1 - d^2).sqrt) :=
by sorry

end chord_through_point_has_fixed_sum_l237_237284


namespace cafeteria_B_turnover_higher_in_May_l237_237958

noncomputable def initial_turnover (X a r : ℝ) : Prop :=
  ∃ (X a r : ℝ),
    (X + 8 * a = X * (1 + r) ^ 8) ∧
    ((X + 4 * a) < (X * (1 + r) ^ 4))

theorem cafeteria_B_turnover_higher_in_May (X a r : ℝ) :
    (X + 8 * a = X * (1 + r) ^ 8) → (X + 4 * a < X * (1 + r) ^ 4) :=
  sorry

end cafeteria_B_turnover_higher_in_May_l237_237958


namespace triangle_side_range_l237_237088

open Real

noncomputable def is_valid_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B > C ∧ A + C > B ∧ B + C > A

theorem triangle_side_range (A B C : ℝ) (hABC : is_valid_triangle A B C) (h_angle : A = 60) (h_AB : B = 6) (h_two_triangles : ∃ ∆ : Type, is_valid_triangle A B C) :
    (3 * sqrt 3 < C) ∧ (C < 6) :=
by
  sorry

end triangle_side_range_l237_237088


namespace cryptarithm_correct_l237_237220

-- Definitions for letters as digits
variables {S T I K M A D R U Z : ℕ}

-- Conditions for the problem
def cryptarithm_conditions : Prop :=
  (S ≠ T) ∧
  (S ≠ I) ∧
  (S ≠ K) ∧
  (S ≠ M) ∧
  (S ≠ A) ∧
  (S ≠ D) ∧
  (S ≠ R) ∧
  (S ≠ U) ∧
  (S ≠ Z) ∧
  (T ≠ I) ∧
  (T ≠ K) ∧
  (T ≠ M) ∧
  (T ≠ A) ∧
  (T ≠ D) ∧
  (T ≠ R) ∧
  (T ≠ U) ∧
  (T ≠ Z) ∧
  (I ≠ K) ∧
  (I ≠ M) ∧
  (I ≠ A) ∧
  (I ≠ D) ∧
  (I ≠ R) ∧
  (I ≠ U) ∧
  (I ≠ Z) ∧
  (K ≠ M) ∧
  (K ≠ A) ∧
  (K ≠ D) ∧
  (K ≠ R) ∧
  (K ≠ U) ∧
  (K ≠ Z) ∧
  (M ≠ A) ∧
  (M ≠ D) ∧
  (M ≠ R) ∧
  (M ≠ U) ∧
  (M ≠ Z) ∧
  (A ≠ D) ∧
  (A ≠ R) ∧
  (A ≠ U) ∧
  (A ≠ Z) ∧
  (D ≠ R) ∧
  (D ≠ U) ∧
  (D ≠ Z) ∧
  (R ≠ U) ∧
  (R ≠ Z) ∧
  (U ≠ Z) ∧
  5 ≤ S ∧  (* contribution of leading digits being valid *)
  S + S + x = 10 + M ∧
  T + T + x = 10*y + A ∧
  I + I = 10*z + 3 ∧
  K + K = 2 + S ∧
  2*S + z = 10 +1

-- Prove the numerical equivalent using cryptarithm function
theorem cryptarithm_correct (h : cryptarithm_conditions) : 
  ∃ (S T I K M A D R U Z : ℕ), cryptarithm_conditions ∧
  let s := "STIKS" in
  let sum := "MASTIKS" in
  let result := "DRIUZIS" in

end cryptarithm_correct_l237_237220


namespace number_of_ways_to_choose_one_person_l237_237411

-- Define the number of people who know each method
def num_people_method1 := 5
def num_people_method2 := 4

-- The total number of people
def total_people := num_people_method1 + num_people_method2

-- The theorem to prove
theorem number_of_ways_to_choose_one_person (h1 : num_people_method1 = 5) (h2 : num_people_method2 = 4) :
  total_people = 9 :=
by
  rw [total_people, h1, h2]
  exact rfl


end number_of_ways_to_choose_one_person_l237_237411


namespace num_customers_served_today_l237_237023

theorem num_customers_served_today (x C : ℕ) (avg_old avg_new : ℕ) 
    (hx : x = 1) (havg_old : avg_old = 65) (havg_new : avg_new = 90) :
    (65 * x + C = 90 * (x + 1)) → (C = 115) :=
by
  intros h
  rw [hx, havg_old, havg_new] at h
  norm_num at h
  exact h

end num_customers_served_today_l237_237023


namespace range_of_a_l237_237484

theorem range_of_a (a : ℝ) (h : ∃ x0 ∈ set.Ioo (-1 : ℝ) 1, 2 * a * x0 - a + 3 = 0) : 
  a ∈ set.Ioo (1 : ℝ) (⊤) ∪ set.Ioo (⊥) (-3 : ℝ) :=
sorry

end range_of_a_l237_237484


namespace arithmetic_sequence_sum_l237_237816

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m: ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Sum of the first n terms of a sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Specific statement we want to prove
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a)
  (h_S9 : sum_seq a 9 = 72) :
  a 2 + a 4 + a 9 = 24 :=
sorry

end arithmetic_sequence_sum_l237_237816


namespace pow_mod_l237_237709

theorem pow_mod (a n m : ℕ) (h : a % m = 1) : (a ^ n) % m = 1 := by
  sorry

example : (11 ^ 2023) % 5 = 1 := by
  apply pow_mod 11 2023 5
  norm_num
  have : 11 % 5 = 1 := by norm_num
  exact this

end pow_mod_l237_237709


namespace value_of_expression_l237_237851

theorem value_of_expression : (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = (27 / 89) :=
by
  sorry

end value_of_expression_l237_237851


namespace problem_statement_l237_237890

-- Definitions for the given conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- The main statement that needs to be proved
theorem problem_statement (f : ℝ → ℝ) (h_odd : odd_function f) (h_monotone : monotone_decreasing f) : f (-1) > f 3 :=
by 
  sorry

end problem_statement_l237_237890


namespace magnitude_of_z_l237_237942

def i : ℂ := complex.I

theorem magnitude_of_z :
  let z := (1 : ℂ) + 2 * i + i^3 in
  complex.abs z = real.sqrt 2 :=
by
  let z := (1 : ℂ) + 2 * i + i^3
  sorry

end magnitude_of_z_l237_237942


namespace weight_left_after_two_deliveries_l237_237777

-- Definitions and conditions
def initial_load : ℝ := 50000
def first_store_percentage : ℝ := 0.1
def second_store_percentage : ℝ := 0.2

-- The statement to be proven
theorem weight_left_after_two_deliveries :
  let weight_after_first_store := initial_load * (1 - first_store_percentage)
  let weight_after_second_store := weight_after_first_store * (1 - second_store_percentage)
  weight_after_second_store = 36000 :=
by sorry  -- Proof omitted

end weight_left_after_two_deliveries_l237_237777


namespace part1_part2_l237_237333

-- Definitions and conditions from the problem
def cost_price : ℝ := 10
def sales_quantity (x : ℝ) : ℝ := -10 * x + 400
def profit (x y : ℝ) : ℝ := y * (x - cost_price)

-- Part 1 
theorem part1 : ∃ x : ℝ, 15 < x ∧ x ≤ 40 ∧ profit x (sales_quantity x) = 1250 := sorry

-- Part 2
theorem part2 : 
  let W (x : ℝ) := -10 * x^2 + 500 * x - 4000 in
  28 ≤ x → x ≤ 35 → W 28 ≤ W x ∧ W x ≤ W 35 := sorry

end part1_part2_l237_237333


namespace trapezoid_circumscribed_radius_l237_237019

theorem trapezoid_circumscribed_radius (r α : ℝ) (h : 0 < r ∧ 0 < α ∧ α < π / 2) :
  let R := r * (Real.sqrt (1 + Real.sin α ^ 2)) / (Real.sin α ^ 2)
  in R = r * (Real.sqrt (1 + Real.sin α ^ 2)) / (Real.sin α ^ 2) :=
by
  sorry

end trapezoid_circumscribed_radius_l237_237019


namespace third_side_triangle_max_l237_237688

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l237_237688


namespace sum_of_percentages_is_correct_l237_237654

def students_entered_first_intersection : ℕ := 12
def students_entered_second_intersection : ℕ := 18
def students_entered_third_intersection : ℕ := 15

def percentage_first_intersection : ℝ := 30 / 100
def percentage_second_intersection : ℝ := 50 / 100
def percentage_third_intersection : ℝ := 20 / 100

def calculate_percentage (percent : ℝ) (students : ℕ) : ℝ :=
  percent * students

def result : ℝ :=
  (calculate_percentage percentage_first_intersection students_entered_first_intersection) +
  (calculate_percentage percentage_second_intersection students_entered_second_intersection) +
  (calculate_percentage percentage_third_intersection students_entered_third_intersection)

theorem sum_of_percentages_is_correct : result = 15.6 :=
by
  sorry

end sum_of_percentages_is_correct_l237_237654


namespace number_of_correct_statements_is_two_l237_237262

-- Define the five statements as predicates.
def statement_1 : Prop := ∀ (a b : ℝ), a ≠ b → ∃ (l : ℝ), l = (a - b)

def statement_2 : Prop := ∀ (a b : ℝ), a ≠ b → ∃! (l : ℝ), l = (a / b)

def statement_3 : Prop := ∀ (a b : ℝ), a ≠ b → ∃! (l : ℝ), l = (a + b)

def statement_4 : Prop := ∀ (a b c : ℝ), a ≠ b ∧ b ≠ c → (a // c) = (a // b)

def statement_5 : Prop := ∀ (a b c : ℝ), a ≠ b ∧ c ≠ b → a > c

-- Prove that only two statements among the given conditions are true.
theorem number_of_correct_statements_is_two :
  (statement_1 = false) ∧
  (statement_2 = true) ∧
  (statement_3 = true) ∧
  (statement_4 = true) ∧
  (statement_5 = true) →
  2 = 2 :=
by sorry

end number_of_correct_statements_is_two_l237_237262


namespace projection_correct_l237_237930

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

-- Define c such that a + c = 0
def c : ℝ × ℝ := (-2, -3)

-- Projection of c in the direction of b
def projection (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_v := Real.sqrt (v.1 * v.1 + v.2 * v.2)
  dot_product / magnitude_v

-- Statement to prove
theorem projection_correct :
  projection c b = -Real.sqrt 65 / 5 :=
by
  sorry

end projection_correct_l237_237930


namespace winning_ticket_probability_l237_237594

theorem winning_ticket_probability (eligible_numbers : List ℕ) (length_eligible_numbers : eligible_numbers.length = 12)
(pick_6 : Π(t : List ℕ), List ℕ) (valid_ticket : List ℕ → Bool) (probability : ℚ) : 
(probability = (1 : ℚ) / (4 : ℚ)) :=
  sorry

end winning_ticket_probability_l237_237594


namespace range_of_m_l237_237861

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  (A ⊆ B) ↔ (m ≤ (11:ℝ)/3) :=
by
  sorry

end range_of_m_l237_237861


namespace quadrilateral_is_rhombus_l237_237617

variable {A B C D O : Point}
variable (AB AO BO OC OD : ℝ)
variable (quad_perimeters_eq : AO + OB + AB = BO + OC + BC ∧
                              BO + OC + BC = CO + OD + CD ∧
                              CO + OD + CD = DO + OA + AD ∧
                              DO + OA + AD = AO + OB + AB)
variable (ACbisection : ∀ {X : Point}, X = midpoint A C → X = O)
variable (BDbisection : ∀ {Y : Point}, Y = midpoint B D → Y = O)

-- Prove the quadrilateral is a rhombus
theorem quadrilateral_is_rhombus
    (quad_perimeters_eq : AO + OB + AB = BO + OC + BC ∧
                          BO + OC + BC = CO + OD + CD ∧
                          CO + OD + CD = DO + OA + AD ∧
                          DO + OA + AD = AO + OB + AB)
    (ACbisection : ∀ {X : Point}, X = midpoint A C → X = O)
    (BDbisection : ∀ {Y : Point}, Y = midpoint B D → Y = O) :
    is_rhombus A B C D :=
sorry

end quadrilateral_is_rhombus_l237_237617


namespace zero_in_interval_l237_237107

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_in_interval : ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The correct statement only
  sorry

end zero_in_interval_l237_237107


namespace mica_should_have_28_26_euros_l237_237585

namespace GroceryShopping

def pasta_cost : ℝ := 3 * 1.70
def ground_beef_cost : ℝ := 0.5 * 8.20
def pasta_sauce_base_cost : ℝ := 3 * 2.30
def pasta_sauce_discount : ℝ := pasta_sauce_base_cost * 0.10
def pasta_sauce_discounted_cost : ℝ := pasta_sauce_base_cost - pasta_sauce_discount
def quesadillas_cost : ℝ := 11.50

def total_cost_before_vat : ℝ :=
  pasta_cost + ground_beef_cost + pasta_sauce_discounted_cost + quesadillas_cost

def vat : ℝ := total_cost_before_vat * 0.05

def total_cost_including_vat : ℝ := total_cost_before_vat + vat

theorem mica_should_have_28_26_euros :
  total_cost_including_vat = 28.26 := by
  -- This is the statement without the proof. 
  sorry

end GroceryShopping

end mica_should_have_28_26_euros_l237_237585


namespace problem_equivalent_proof_l237_237468

noncomputable def general_term (n r : ℕ) : ℚ :=
  (-1/2)^r * (Nat.choose n r : ℚ) * x ^ (n - 2 * r) / 3

noncomputable def evaluate_at_r (r : ℕ) (n : ℕ) : Prop :=
  n = 10 ∧ (general_term 10 2) = 45 / 4

theorem problem_equivalent_proof (n : ℕ) (x : ℚ) :
  (is_constant (general_term n 5) → n = 10) 
  ∧ evaluate_at_r 2 10 :=
begin
  sorry
end

end problem_equivalent_proof_l237_237468


namespace quilt_shaded_fraction_l237_237270

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_squares := 8
  let fully_shaded := 4
  let half_shaded := 4
  let shaded_area := fully_shaded + half_shaded * 1 / 2
  shaded_area / total_squares = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l237_237270


namespace sin_ineq_l237_237059

open Real

theorem sin_ineq (n : ℕ) (h : n > 0) : sin (π / (4 * n)) ≥ (sqrt 2) / (2 * n) :=
sorry

end sin_ineq_l237_237059


namespace sequence_formula_l237_237870

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 1) (h_recurrence : ∀ n : ℕ, 2 * n * a n + 1 = (n + 1) * a n) :
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
sorry

end sequence_formula_l237_237870


namespace greatest_third_side_l237_237670

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l237_237670


namespace intersection_point_of_lines_l237_237435

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), (2 * y = 3 * x - 6) ∧ (x + 5 * y = 10) ∧ (x = 50 / 17) ∧ (y = 24 / 17) :=
by
  sorry

end intersection_point_of_lines_l237_237435


namespace increasing_interval_of_y_l237_237823

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x

theorem increasing_interval_of_y :
  ∃ (a b : ℝ), 0 < a ∧ a < e ∧ (∀ x : ℝ, a < x ∧ x < e → y x < y (x + ε)) :=
sorry

end increasing_interval_of_y_l237_237823


namespace max_notebooks_l237_237291

-- Define the conditions using Lean definitions
variables (x : ℕ) (y : ℕ)

def total_items := x + y = 10
def notebook_cost := ∀ x, 12 * x
def pencil_case_cost := ∀ y, 7 * y
def total_cost_condition := (12 * x + 7 * y) ≤ 100

-- The goal is to prove that the maximum number of notebooks x is 6
theorem max_notebooks (h1 : total_items x y) (h2 : total_cost_condition x y) : x ≤ 6 :=
sorry

end max_notebooks_l237_237291


namespace product_of_integers_abs_less_than_4_l237_237636

theorem product_of_integers_abs_less_than_4 (S : Set ℤ) (h : S = {-3, -2, -1, 0, 1, 2, 3}) : 
  S.prod id = 0 := by
  rw [h]
  sorry

end product_of_integers_abs_less_than_4_l237_237636


namespace base_of_number_with_61_digits_l237_237752

theorem base_of_number_with_61_digits :
  ∃ b : ℤ,  b ≥ 2 ∧ (floor (200 * (Real.log10 b))) + 1 = 61 :=
sorry

end base_of_number_with_61_digits_l237_237752


namespace greatest_integer_third_side_l237_237664

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l237_237664


namespace max_traffic_flow_at_40_range_for_flow_exceeding_10_l237_237780

/-- Conditions for the traffic flow problem --/
def traffic_flow (v : ℝ) := 920 * v / (v^2 + 3 * v + 1600)

/-- Correctness Statement --/
theorem max_traffic_flow_at_40 :
  ∀ v > 0, traffic_flow v ≤ 920 / 83 ∧ traffic_flow 40 = 920 / 83 := 
     sorry

theorem range_for_flow_exceeding_10 :
  ∀ v > 0, traffic_flow v > 10 ↔ 25 < v ∧ v < 64 :=
     sorry

end max_traffic_flow_at_40_range_for_flow_exceeding_10_l237_237780


namespace simplified_expression_value_l237_237237

noncomputable def expression (a b : ℝ) : ℝ :=
  3 * a ^ 2 - b ^ 2 - (a ^ 2 - 6 * a) - 2 * (-b ^ 2 + 3 * a)

theorem simplified_expression_value :
  expression (-1/2) 3 = 19 / 2 :=
by
  sorry

end simplified_expression_value_l237_237237


namespace find_constants_solve_inequality_l237_237888

-- Given function f
def f (x : ℝ) (a b : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

-- f is odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Problem 1: Prove values of a and b
theorem find_constants (h₁ : is_odd (f a b)) : a = 2 ∧ b = 1 :=
sorry

-- Problem 2: Solve the inequality
theorem solve_inequality (h₁ : is_odd (f 2 1)) (h₂ : ∀ x y : ℝ, x < y → f x > f y) 
(t : ℝ) : f (t^2 - 2*t) (2 : ℝ) (1 : ℝ) + f (2*t^2 - 1) (2 : ℝ) (1 : ℝ) < 0 ↔ t > 1 ∨ t < -1/3 :=
sorry

end find_constants_solve_inequality_l237_237888


namespace decreasing_geometric_sequence_l237_237577

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q ^ n

theorem decreasing_geometric_sequence (a₁ q : ℝ) (aₙ : ℕ → ℝ) (hₙ : ∀ n, aₙ n = geometric_sequence a₁ q n) 
  (h_condition : 0 < q ∧ q < 1) : ¬(0 < q ∧ q < 1 ↔ ∀ n, aₙ n > aₙ (n + 1)) :=
sorry

end decreasing_geometric_sequence_l237_237577


namespace sum_of_squares_of_roots_l237_237266

theorem sum_of_squares_of_roots :
  let a := 3
  let b := 4
  let c := -9
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  let sum_of_squares := (sum_of_roots^2) - 2 * product_of_roots
  sum_of_squares = 70 / 9 :=
by
  let x1 := sum_of_roots
  let x2 := product_of_roots
  have h1 : x1 = -b / a := rfl
  have h2 : x2 = c / a := rfl
  let sum_of_squares := (x1^2) - 2 * x2
  have h3 : sum_of_squares = (x1^2) - 2 * x2 := rfl
  sorry

end sum_of_squares_of_roots_l237_237266


namespace find_fourth_number_in_sequence_l237_237149

-- Define the conditions of the sequence
def first_number : ℤ := 1370
def second_number : ℤ := 1310
def third_number : ℤ := 1070
def fifth_number : ℤ := -6430

-- Define the differences
def difference1 : ℤ := second_number - first_number
def difference2 : ℤ := third_number - second_number

-- Define the ratio of differences
def ratio : ℤ := 4
def next_difference : ℤ := difference2 * ratio

-- Define the fourth number
def fourth_number : ℤ := third_number - (-next_difference)

-- Theorem stating the proof problem
theorem find_fourth_number_in_sequence : fourth_number = 2030 :=
by sorry

end find_fourth_number_in_sequence_l237_237149


namespace find_f_neg2_l237_237574

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Define the conditions and statement to be proved
theorem find_f_neg2 (a b : ℝ) (h1 : f a b 2 = 9) : f a b (-2) = 13 :=
by
  -- Conditions lead to the conclusion to be proved
  sorry

end find_f_neg2_l237_237574


namespace gravitational_force_at_384000km_l237_237627

theorem gravitational_force_at_384000km
  (d1 d2 : ℝ)
  (f1 f2 : ℝ)
  (k : ℝ)
  (h1 : d1 = 6400)
  (h2 : d2 = 384000)
  (h3 : f1 = 800)
  (h4 : f1 * d1^2 = k)
  (h5 : f2 * d2^2 = k) :
  f2 = 2 / 9 :=
by
  sorry

end gravitational_force_at_384000km_l237_237627


namespace correct_identity_l237_237015

def sin_add (α β : ℝ) : ℝ := Real.sin (α + β)
def cos_add (α β : ℝ) : ℝ := Real.cos (α + β)
def sin_sub (α β : ℝ) : ℝ := Real.sin (α - β)
def cos_sub (α β : ℝ) : ℝ := Real.cos (α - β)

-- Defining the fundamental trigonometric identities
lemma sin_add_identity (α β : ℝ) : Real.sin (α + β) = Real.sin α * Real.cos β + Real.cos α * Real.sin β :=
by sorry

lemma cos_add_identity (α β : ℝ) : Real.cos (α + β) = Real.cos α * Real.cos β - Real.sin α * Real.sin β :=
by sorry

lemma sin_sub_identity (α β : ℝ) : Real.sin (α - β) = Real.sin α * Real.cos β - Real.cos α * Real.sin β :=
by sorry

lemma cos_sub_identity (α β : ℝ) : Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β :=
by sorry

-- The main theorem to prove the correct identity among the options
theorem correct_identity : sin_add α β = Real.sin α * Real.cos β + Real.cos α * Real.sin β :=
by exact sin_add_identity α β

end correct_identity_l237_237015


namespace spell_AMCB_paths_equals_24_l237_237416

def central_A_reachable_M : Nat := 4
def M_reachable_C : Nat := 2
def C_reachable_B : Nat := 3

theorem spell_AMCB_paths_equals_24 :
  central_A_reachable_M * M_reachable_C * C_reachable_B = 24 := by
  sorry

end spell_AMCB_paths_equals_24_l237_237416


namespace sum_cubes_is_zero_l237_237806

theorem sum_cubes_is_zero :
  (∑ i in Finset.range 50, (i + 1)^3) + (∑ i in Finset.range 50, (- (i + 1))^3) = 0 :=
by 
  sorry

end sum_cubes_is_zero_l237_237806


namespace units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l237_237440

-- Define the cycle of the units digits of powers of 7
def units_digit_of_7_power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1  -- 7^4, 7^8, ...
  | 1 => 7  -- 7^1, 7^5, ...
  | 2 => 9  -- 7^2, 7^6, ...
  | 3 => 3  -- 7^3, 7^7, ...
  | _ => 0  -- unreachable

-- The main theorem to prove
theorem units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared : 
  units_digit_of_7_power (3 ^ (5 ^ 2)) = 3 :=
by
  sorry

end units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l237_237440


namespace probability_of_drawing_red_ball_probability_of_drawing_two_black_balls_l237_237171

-- Define the conditions
def total_balls := 5
def red_balls := 2
def black_balls := 3

-- Problem 1: Probability of drawing a red ball
theorem probability_of_drawing_red_ball : 
  (red_balls: ℚ) / (total_balls: ℚ) = 2 / 5 :=
by sorry

-- Problem 2: Probability of drawing two black balls without replacement
theorem probability_of_drawing_two_black_balls :
  (↑(nat.choose black_balls 2) / ↑(nat.choose total_balls 2) : ℚ) = 3 / 10 :=
by sorry

end probability_of_drawing_red_ball_probability_of_drawing_two_black_balls_l237_237171


namespace find_quadratic_function_l237_237346

-- Conditions as definitions in Lean 4
def is_quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c

def has_equal_roots (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c ∧ (b^2 - 4 * a * c = 0)

def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, deriv f x = f' x

-- The problem statement
theorem find_quadratic_function (f : ℝ → ℝ) :
  is_quadratic_function f ∧ has_equal_roots f ∧ derivative f (λ x, 2 * x + 2) →
  f = λ x, x^2 + 2 * x + 1 :=
by
  sorry

end find_quadratic_function_l237_237346


namespace fraction_of_milk_in_cup1_l237_237385

noncomputable def coffeeMilkFraction : ℚ :=
let cup1 := (6, 0) in  -- (6 ounces coffee, 0 ounces milk)
let cup2 := (0, 6) in  -- (0 ounces coffee, 6 ounces milk)
let cup1_after_pour1 := (cup1.1 - 2, cup1.2) in  -- Cup 1 after pour 1
let cup2_after_pour1 := (2, 6) in  -- Cup 2 after pour 1
let cup1_after_pour2 := (cup1_after_pour1.1 + 0.5, cup1_after_pour1.2 + 1.5) in  -- Cup 1 after pour 2
1 * (cup1_after_pour2.2 / (cup1_after_pour2.1 + cup1_after_pour2.2))  -- Fraction of milk in cup 1

theorem fraction_of_milk_in_cup1 : coffeeMilkFraction = 1 / 4 := 
by 
  unfold coffeeMilkFraction
  sorry

end fraction_of_milk_in_cup1_l237_237385


namespace min_birthdays_on_wednesday_l237_237026

theorem min_birthdays_on_wednesday 
  (total_employees : ℕ)
  (excluded_march_birthdays : ℕ)
  (employees_without_march : total_employees - excluded_march_birthdays = 50)
  (days_in_week : ℕ)
  (days_distribution : ∀ d, d ≠ 3 → ∃ (x : ℕ), ∀ y, y = 7 * x)
  (wednesday_more_than_rest : ∀ x, (wednesday_birthdays : ℕ), wednesday_birthdays > x → wednesday_birthdays = x + 1)
  : ∃ (wednesday_birthdays : ℕ), wednesday_birthdays = 8 := 
sorry

end min_birthdays_on_wednesday_l237_237026


namespace total_applicants_is_40_l237_237590

def total_applicants (PS GPA_high Not_PS_GPA_low both : ℕ) : ℕ :=
  let PS_or_GPA_high := PS + GPA_high - both 
  PS_or_GPA_high + Not_PS_GPA_low

theorem total_applicants_is_40 :
  total_applicants 15 20 10 5 = 40 :=
by
  sorry

end total_applicants_is_40_l237_237590


namespace arithmetic_sequence_s11_l237_237157

-- Define the arithmetic sequence and sum function
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d
def sum_first_n (a d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a + (n - 1) * d)

-- State the main theorem to be proven
theorem arithmetic_sequence_s11 (a d : ℝ)
  (h1 : arithmetic_sequence a d 1 + arithmetic_sequence a d 6 + arithmetic_sequence a d 9 = 54)
  (S11 : sum_first_n a d 11 = 297) : S11 = 297 :=
by sorry

end arithmetic_sequence_s11_l237_237157


namespace equivalent_problem_l237_237465

noncomputable def question (f : ℝ → ℝ) (x₀ : ℝ) := 
  limit (fun (Δx : ℝ) => (f (x₀ + 2 * Δx) - f x₀) / (3 * Δx)) (𝓝 0)

theorem equivalent_problem (f : ℝ → ℝ) (x₀ : ℝ) 
  (h_deriv : deriv f x₀ = 3) : 
  question f x₀ = 2 :=
by
  sorry

end equivalent_problem_l237_237465


namespace Bruce_initial_eggs_l237_237800

variable (B : ℕ)

theorem Bruce_initial_eggs (h : B - 70 = 5) : B = 75 := by
  sorry

end Bruce_initial_eggs_l237_237800


namespace midpoint_C_is_either_l237_237620

def A : ℝ := -7
def dist_AB : ℝ := 5

theorem midpoint_C_is_either (C : ℝ) (h : C = (A + (A + dist_AB / 2)) / 2 ∨ C = (A + (A - dist_AB / 2)) / 2) : 
  C = -9 / 2 ∨ C = -19 / 2 := 
sorry

end midpoint_C_is_either_l237_237620


namespace pass_through_midpoint_l237_237568

noncomputable def centroid (A B C : Point) : Point :=
sorry  -- Define centroid mathematically

-- Definition of the given geometric configuration
variables {A B C G E F X Y P : Point}

-- Given Conditions
def conditions : Prop :=
  is_triangle A B C ∧
  centroid A B C = G ∧
  on_line E B C ∧ on_line F B C ∧
  distance B E = distance E F ∧ distance E F = distance F C ∧
  on_line X A B ∧ on_line Y A C ∧ 
  ¬ collinear X Y G ∧
  parallel (line_through E (XG)) (line_through E P) ∧
  parallel (line_through F (YG)) (line_through F P) ∧
  P ≠ G

-- Proof goal: GP passes through the midpoint of XY
theorem pass_through_midpoint :
  conditions → passes_through_midpoint (line_through G P) X Y :=
sorry  -- Proof omitted

end pass_through_midpoint_l237_237568


namespace cookies_remaining_in_jar_l237_237581

-- Definition of the conditions
variable (initial_cookies : Nat)

def cookies_taken_by_Lou_Senior := 3 + 1
def cookies_taken_by_Louie_Junior := 7
def total_cookies_taken := cookies_taken_by_Lou_Senior + cookies_taken_by_Louie_Junior

-- Debra's assumption and the proof goal
theorem cookies_remaining_in_jar (half_cookies_removed : total_cookies_taken = initial_cookies / 2) : 
  initial_cookies - total_cookies_taken = 11 := by
  sorry

end cookies_remaining_in_jar_l237_237581


namespace tractor_trailer_weight_after_deliveries_l237_237775

def initial_weight := 50000
def first_store_unload_percent := 0.10
def second_store_unload_percent := 0.20

theorem tractor_trailer_weight_after_deliveries: 
  let weight_after_first_store := initial_weight - (first_store_unload_percent * initial_weight)
  let weight_after_second_store := weight_after_first_store - (second_store_unload_percent * weight_after_first_store)
  weight_after_second_store = 36000 :=
by
  sorry

end tractor_trailer_weight_after_deliveries_l237_237775


namespace probability_diff_ranks_floor_l237_237721

theorem probability_diff_ranks_floor:
  let a := (1248 : ℚ) / 1326 in
  ⌊1000 * a⌋ = 941 :=
by
  have h : a = (208 : ℚ) / 221 := by sorry
  have h1 : 1000 * a = 1000 * (208 / 221) := by rw [h]
  have h2 : 1000 * (208 / 221) ≈ 941.176 := by sorry
  have hf : ⌊1000 * (208 / 221)⌋ = 941 := by sorry
  exact hf

end probability_diff_ranks_floor_l237_237721


namespace convex_polygon_with_tiles_l237_237722

variable (n : ℕ)

def canFormConvexPolygon (n : ℕ) : Prop :=
  3 ≤ n ∧ n ≤ 12

theorem convex_polygon_with_tiles (n : ℕ) 
  (square_internal_angle : ℕ := 90) 
  (equilateral_triangle_internal_angle : ℕ := 60)
  (external_angle_step : ℕ := 30)
  (total_external_angle : ℕ := 360) :
  canFormConvexPolygon n :=
by 
  sorry

end convex_polygon_with_tiles_l237_237722


namespace points_of_intersection_altitudes_lie_on_line_l237_237871

theorem points_of_intersection_altitudes_lie_on_line
  (A B C D E F : Point)
  (h1: Line AB D)
  (h2: Line BC E)
  (h3: Line CA F)
  (h4: AltitudesIntersectionsOnLine (triangleABC: Triangle A B C)
                                    (triangleBDE: Triangle B E D)
                                    (triangleDAF: Triangle D A F)
                                    (triangleCEF: Triangle C E F)
                                    (LineOfIntersections: Line))
  (h5: LinePerpendicular LineOfIntersections GaussLineOfABC) :
  True := sorry

end points_of_intersection_altitudes_lie_on_line_l237_237871


namespace third_side_triangle_max_l237_237692

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l237_237692


namespace supercomputer_process_never_stops_l237_237772

theorem supercomputer_process_never_stops (n : ℕ) (hn : n = 10^899 + 10^898 + ... + 10 + 1) :
  ∀ m, m = n → let process (k : ℕ) := 2 * (k / 100) + 8 * (k % 100) in
  ∀ k', k' ≠ 0 → m = process k' → k' ≥ 100 :=
begin
  sorry -- proof is not required as per the instructions
end

end supercomputer_process_never_stops_l237_237772


namespace triangle_third_side_l237_237687

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l237_237687


namespace probability_of_drawing_odd_card_l237_237282

theorem probability_of_drawing_odd_card :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 9}
  let odd_cards := {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ n % 2 = 1}
  ∃ (total_cards odd_cards_count : ℕ), 
    total_cards = Finset.card (Finset.filter (λ n, n ∈ cards) (Finset.range 10)) ∧
    odd_cards_count = Finset.card (Finset.filter (λ n, n ∈ odd_cards) (Finset.range 10)) ∧
    (odd_cards_count : ℚ) / (total_cards : ℚ) = 5 / 9 :=
by
  sorry

end probability_of_drawing_odd_card_l237_237282


namespace correct_statements_l237_237105

theorem correct_statements :
  (∀ mode median : ℕ, (∀ left_area right_area : ℕ, left_area ≠ right_area)) ∧
  (∀ std_dev : ℕ, std_dev > 0 → (∀ fluctuation : ℕ, std_dev < fluctuation → true)) ∧
  (∀ regression_analysis : ℕ, ¬(∀ correlated_events : bool, regression_analysis = correlated_events)) ∧
  (∀ forecast_var explanatory_var random_error : ℤ, forecast_var = explanatory_var + random_error) ∧
  (∀ R2 : ℝ, (R2 > 0 → (∀ sum_squared_residuals : ℝ, sum_squared_residuals > 0 → (R2 * R2) < sum_squared_residuals)))
  ↔ [2, 4, 5] := 
  sorry

end correct_statements_l237_237105


namespace solve_for_x_l237_237241

noncomputable def solve_ratio_eqn (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -4) : Prop :=
  (x + 5) / (x - 3) = (x - 2) / (x + 4)

theorem solve_for_x (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -4) :
  solve_ratio_eqn x h1 h2 → x = -1 :=
begin
  sorry
end

end solve_for_x_l237_237241


namespace product_mod_17_eq_zero_l237_237713

theorem product_mod_17_eq_zero :
    (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end product_mod_17_eq_zero_l237_237713


namespace circle_center_radius_l237_237100

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 4 → (x - h)^2 + (y - k)^2 = r^2) ∧
    h = -1 ∧ k = 1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l237_237100


namespace min_value_of_expression_l237_237451

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (6 * real.sqrt(a * b) + 3 / a + 3 / b) ≥ 12 := 
sorry

end min_value_of_expression_l237_237451


namespace cosine_double_angle_l237_237456

theorem cosine_double_angle 
  (theta : ℝ) 
  (h₁ : Real.sin theta = 3/5) 
  (h₂ : θ ∈ set.Ioo π/2 π) :
  Real.cos (2*theta) = 7/25 :=
sorry

end cosine_double_angle_l237_237456


namespace determine_b_eq_l237_237423

theorem determine_b_eq (b : ℝ) : (∃! (x : ℝ), |x^2 + 3 * b * x + 4 * b| ≤ 3) ↔ b = 4 / 3 ∨ b = 1 := 
by sorry

end determine_b_eq_l237_237423


namespace cookie_cost_per_day_l237_237061

theorem cookie_cost_per_day
    (days_in_April : ℕ)
    (cookies_per_day : ℕ)
    (total_spent : ℕ)
    (total_cookies : ℕ := days_in_April * cookies_per_day)
    (cost_per_cookie : ℕ := total_spent / total_cookies) :
  days_in_April = 30 ∧ cookies_per_day = 3 ∧ total_spent = 1620 → cost_per_cookie = 18 :=
by
  sorry

end cookie_cost_per_day_l237_237061


namespace find_m_plus_b_l237_237442

theorem find_m_plus_b (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (-3:ℝ, 8:ℝ)) (h2 : (x2, y2) = (0:ℝ, -1:ℝ)) :
  let m := (y2 - y1) / (x2 - x1),
      b := y2 - m * x2
  in m + b = -4 :=
by {
  cases h1,
  cases h2,
  let m : ℝ := (-1 - 8) / (0 - (-3)),
  let b : ℝ := (-1 - m * 0),
  have h_m : m = -3 := by sorry, -- Proof of calculated m would go here
  have h_b : b = -1 := by sorry, -- Proof of calculated b would go here
  calc m + b = -3 + -1 : by rw [h_m, h_b]
          ... = -4 : by norm_num
}

end find_m_plus_b_l237_237442


namespace tip_percentage_l237_237354

theorem tip_percentage (T : ℝ) 
  (total_cost meal_cost sales_tax : ℝ)
  (h1 : meal_cost = 61.48)
  (h2 : sales_tax = 0.07 * meal_cost)
  (h3 : total_cost = meal_cost + sales_tax + T * meal_cost)
  (h4 : total_cost ≤ 75) :
  T ≤ 0.1499 :=
by
  -- main proof goes here
  sorry

end tip_percentage_l237_237354


namespace value_of_sum_is_eleven_l237_237247

-- Define the context and conditions

theorem value_of_sum_is_eleven (x y z w : ℤ) 
  (h1 : x - y + z = 7)
  (h2 : y - z + w = 8)
  (h3 : z - w + x = 4)
  (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
begin
  sorry
end

end value_of_sum_is_eleven_l237_237247


namespace solution_set_sin_cos_eq_one_l237_237849

open Real TrigonometricFunction

theorem solution_set_sin_cos_eq_one :
  {x : ℝ | ∃ k : ℤ, x = k * π + π / 4 ∨ x = k * π + π / 2} =
  {x : ℝ | sin (x / 2) - cos (x / 2) = 1} :=
by
  -- proof is omitted
  sorry


end solution_set_sin_cos_eq_one_l237_237849


namespace rooms_per_floor_l237_237737

-- Definitions for each of the conditions
def numberOfFloors : ℕ := 4
def hoursPerRoom : ℕ := 6
def hourlyRate : ℕ := 15
def totalEarnings : ℕ := 3600

-- Statement of the problem
theorem rooms_per_floor : 
  (totalEarnings / hourlyRate) / hoursPerRoom / numberOfFloors = 10 := 
  sorry

end rooms_per_floor_l237_237737


namespace solve_inequality_l237_237848

theorem solve_inequality (x : ℝ) : abs ((3 - x) / 4) < 1 ↔ 2 < x ∧ x < 7 :=
by {
  sorry
}

end solve_inequality_l237_237848


namespace frogs_per_fish_per_day_l237_237829

theorem frogs_per_fish_per_day
  (f g n F : ℕ)
  (h1 : f = 30)
  (h2 : g = 15)
  (h3 : n = 9)
  (h4 : F = 32400) :
  F / f / (n * g) = 8 := by
  sorry

end frogs_per_fish_per_day_l237_237829


namespace largest_nonrepresentable_sum_135_to_144_l237_237845

theorem largest_nonrepresentable_sum_135_to_144 : ∃ n : ℕ, n = 2024 ∧ (¬ ∃ a b c d e f g h i j k : ℕ, n = 135 * a + 136 * b + 137 * c + 138 * d + 139 * e + 140 * f + 141 * g + 142 * h + 143 * i + 144 * j) :=
by { 
  let n := 2024,
  use n,
  split,
  { refl },
  { sorry }
}

end largest_nonrepresentable_sum_135_to_144_l237_237845


namespace caroline_lassis_l237_237409

def ratio_lassis_per_mango (lassis : ℕ) (mangoes : ℕ) : ℕ :=
  lassis / mangoes

theorem caroline_lassis (lassis_made : ℕ) (mangoes_used : ℕ) (mangoes_given : ℕ) (ratio : ℕ) :
  lasis_made = 15 → mangoes_used = 3 → ratio_lassis_per_mango lasis_made mangoes_used = ratio → 
  mangoes_given = 18 → lasis_made * mangoes_given / mangoes_used = 90 :=
by
  intros h1 h2 r h3
  rw [h1, h2]
  simp only [ratio_lassis_per_mango, nat.div_self h2.ne_zero, nat.div_mul_cancel r]
  assumption

#lint -- checking the code formatting and validity

end caroline_lassis_l237_237409


namespace cube_volume_l237_237645

theorem cube_volume (s : ℝ) (h : 12 * s = 96) : s^3 = 512 :=
by
  sorry

end cube_volume_l237_237645


namespace Z_in_fourth_quadrant_l237_237158

def Z : ℂ := (2 : ℂ) / (3 - I) + I ^ (2015 : ℤ)

theorem Z_in_fourth_quadrant (Z : ℂ) : Z = (2 : ℂ) / (3 - I) + I ^ (2015 : ℤ) → 0 < Z.re ∧ Z.im < 0 :=
by
  intro hZ
  sorry

end Z_in_fourth_quadrant_l237_237158


namespace Laura_weekly_driving_distance_l237_237182

theorem Laura_weekly_driving_distance :
  ∀ (house_to_school : ℕ) (extra_to_supermarket : ℕ) (school_days_per_week : ℕ) (supermarket_trips_per_week : ℕ),
    house_to_school = 20 →
    extra_to_supermarket = 10 →
    school_days_per_week = 5 →
    supermarket_trips_per_week = 2 →
    (school_days_per_week * house_to_school + supermarket_trips_per_week * ((house_to_school / 2) + extra_to_supermarket) * 2) = 180 :=
by
  intros house_to_school extra_to_supermarket school_days_per_week supermarket_trips_per_week
  assume house_to_school_eq : house_to_school = 20
  assume extra_to_supermarket_eq : extra_to_supermarket = 10
  assume school_days_per_week_eq : school_days_per_week = 5
  assume supermarket_trips_per_week_eq : supermarket_trips_per_week = 2
  rw [house_to_school_eq, extra_to_supermarket_eq, school_days_per_week_eq, supermarket_trips_per_week_eq]
  sorry

end Laura_weekly_driving_distance_l237_237182


namespace sqrt_6_between_2_and_3_l237_237427

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by
  sorry

end sqrt_6_between_2_and_3_l237_237427


namespace aubrey_total_yield_l237_237404

structure GardenConditions where
  total_rows : Nat
  pattern : List (String × Nat)
  tomato_yield : Nat → Nat
  cucumber_yield_A : Nat
  cucumber_yield_B : Nat
  bell_pepper_yield : Nat

def aubrey_garden_conditions : GardenConditions := {
  total_rows := 20,
  pattern := [("tomatoes", 1), ("cucumbers", 2), ("bell_peppers", 1)],
  tomato_yield := λ i => if i = 0 ∨ i = 7 then 6 else 4,
  cucumber_yield_A := 4,
  cucumber_yield_B := 5,
  bell_pepper_yield := 2
}

theorem aubrey_total_yield
  (cond: GardenConditions)
  (htom : cond.pattern.filter (λ p => p.fst = "tomatoes").sum (λ p => p.snd) * 8 *
           2 * 6 + cond.pattern.filter (λ p => p.fst = "tomatoes").sum (λ p => p.snd) * 8 *
           6 * 4 = 180)
  (hcuc : cond.pattern.filter (λ p => p.fst = "cucumbers").sum (λ p => p.snd) * 6 *
           (3 * cond.cucumber_yield_A + 3 * cond.cucumber_yield_B) = 270)
  (hpep : cond.pattern.filter (λ p => p.fst = "bell_peppers").sum (λ p => p.snd) * 12 *
           cond.bell_pepper_yield = 120) :
  True := 
    sorry

end aubrey_total_yield_l237_237404


namespace surface_area_of_circumscribed_sphere_l237_237760

theorem surface_area_of_circumscribed_sphere (a : ℝ) (h1 : a = 2) : 
  4 * Real.pi * ((Real.sqrt 6 / 2) ^ 2) = 6 * Real.pi := 
by
  -- defining the edge length of the tetrahedron.
  let edge_length := a
  -- defining the length of the cube derived from the tetrahedron.
  have h2 : Real.sqrt 2 * edge_length / 2 = Real.sqrt 6, from sorry,
  -- expressing the diameter of the circumscribed sphere.
  let diameter := Real.sqrt 6 / 2
  -- calculating the surface area of the sphere
  let surface_area := 4 * Real.pi * (diameter ^ 2)
  -- assert and verify the surface area is equal to 6pi
  exact sorry

end surface_area_of_circumscribed_sphere_l237_237760


namespace evaluate_expression_l237_237837

-- Definition of variables a, b, c as given in conditions
def a : ℕ := 7
def b : ℕ := 11
def c : ℕ := 13

-- The theorem to prove the given expression equals 31
theorem evaluate_expression : 
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 31 :=
by
  sorry

end evaluate_expression_l237_237837


namespace mikes_original_speed_l237_237586

variable (x : ℕ) -- x is the original typing speed of Mike

-- Condition: After the accident, Mike's typing speed is 20 words per minute less
def currentSpeed : ℕ := x - 20

-- Condition: It takes Mike 18 minutes to type 810 words at his reduced speed
def typingTimeCondition : Prop := 18 * currentSpeed x = 810

-- Proof goal: Prove that Mike's original typing speed is 65 words per minute
theorem mikes_original_speed (h : typingTimeCondition x) : x = 65 := 
sorry

end mikes_original_speed_l237_237586


namespace smallest_benches_l237_237351

theorem smallest_benches (N : ℕ) (h1 : ∃ n, 8 * n = 40 ∧ 10 * n = 40) : N = 20 :=
sorry

end smallest_benches_l237_237351


namespace solution_set_f_inequality_l237_237899

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 3 then 9 else -x^2 + 6 * x

theorem solution_set_f_inequality : 
  { x : ℝ | f (x^2 - 2 * x) < f (3 * x - 4) } = {x : ℝ | 1 < x ∧ x < 3} :=
by 
  sorry

end solution_set_f_inequality_l237_237899


namespace find_x_l237_237576

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2 * x + 1 else 2^x

theorem find_x (h : f (f x) = 2) : x = -1 :=
sorry

end find_x_l237_237576


namespace standard_eq_circle_C_equation_line_AB_l237_237075

-- Define the center of circle C and the line l
def center_C : ℝ × ℝ := (2, 1)
def line_l (x y : ℝ) : Prop := x = 3

-- Define the standard equation of circle C
def eq_circle_C (x y : ℝ) : Prop :=
  (x - center_C.1)^2 + (y - center_C.2)^2 = 1

-- Equation of circle O
def eq_circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the condition that circle C intersects with circle O at points A and B
def intersects (x y : ℝ) : Prop :=
  eq_circle_C x y ∧ eq_circle_O x y

-- Define the equation of line AB in general form
def eq_line_AB (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Prove the standard equation of circle C is (x-2)^2 + (y-1)^2 = 1
theorem standard_eq_circle_C:
  eq_circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

-- Prove that the equation of line AB is 2x + y - 4 = 0, given the intersection points A and B
theorem equation_line_AB (x y : ℝ) (h : intersects x y) :
  eq_line_AB x y :=
sorry

end standard_eq_circle_C_equation_line_AB_l237_237075


namespace inequality_one_solution_inequality_two_solution_l237_237608

-- The statement for the first inequality
theorem inequality_one_solution (x : ℝ) :
  |1 - ((2 * x - 1) / 3)| ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5 := sorry

-- The statement for the second inequality
theorem inequality_two_solution (x : ℝ) :
  (2 - x) * (x + 3) < 2 - x ↔ x < -2 ∨ x > 2 := sorry

end inequality_one_solution_inequality_two_solution_l237_237608


namespace add_sub_inverse_sub_add_inverse_mul_div_inverse_div_mul_inverse_exp_root_inverse_exp_log_inverse_unique_inverse_add_mul_unique_inverse_exp_l237_237703

-- Define operations and their properties
variable {α : Type*}

-- Assume commutativity for operations where it is required
axiom commutative {f : α → α → α} (hf : ∀ a b : α, f a b = f b a)

-- Define the operation and its inverse for addition and multiplication
def add (a b : ℕ) : ℕ := a + b
def sub (a b : ℕ) : ℕ := a - b

def mul (a b : ℕ) : ℕ := a * b
def div (a b : ℕ) : ℕ := a / b

-- Define the operation and its two inverse for exponentiation
def exp (a b : ℕ) : ℕ := a ^ b
def root (a b : ℕ) : ℕ := Nat.root a b  -- assuming a root operation
def log (a b : ℕ) : ℕ := Int.toNat (log a b) -- assuming log is handled properly

-- Proof statements (no proof, only statements)

theorem add_sub_inverse (a b : ℕ) : add a b - b = a :=
sorry

theorem sub_add_inverse (c b : ℕ) : c - b + b = c :=
sorry

theorem mul_div_inverse (a b : ℕ) : mul a b / b = a :=
sorry

theorem div_mul_inverse (c b : ℕ) : c / b * b = c :=
sorry

theorem exp_root_inverse (a b : ℕ) : root (exp a b) b = a :=
sorry

theorem exp_log_inverse (c b : ℕ) : log (exp b c) b = c :=
sorry

theorem unique_inverse_add_mul (a b c : ℕ) :
  (∃ (inverse : ℕ → ℕ → ℕ), (inverse b a = c ∧ inverse c b = a)) →
  (∀ (inverse' : ℕ → ℕ → ℕ), inverse' b c = inverse b c) :=
sorry

theorem unique_inverse_exp (a b c : ℕ) :
  (∃ (inverse1 inverse2 : ℕ → ℕ → ℕ),
     (inverse1 (exp a b) = a ∧ inverse2 (exp a b) = b)) →
  (∃ unique1 unique2, unique1 (exp b c) = inverse1 ∧ unique2 (exp b c) = inverse2) :=
sorry

end add_sub_inverse_sub_add_inverse_mul_div_inverse_div_mul_inverse_exp_root_inverse_exp_log_inverse_unique_inverse_add_mul_unique_inverse_exp_l237_237703


namespace projection_ratio_l237_237562

variables (V : Type) [InnerProductSpace ℝ V] (v w : V)
noncomputable def p : V := (inner v w / inner w w) • w
noncomputable def r : V := (inner w p / inner p p) • p

-- Condition given in the problem
axiom hp : ‖p v w‖ / ‖v‖ = 3 / 4

-- Goal: Prove the ratio of ‖r‖ to ‖w‖ is 3/4
theorem projection_ratio (hp : ‖p v w‖ / ‖v‖ = 3 / 4)
  : ‖r v w‖ / ‖w‖ = 3 / 4 := sorry

end projection_ratio_l237_237562


namespace third_side_triangle_max_l237_237690

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l237_237690


namespace keys_missing_l237_237989

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end keys_missing_l237_237989


namespace matrix_calculation_l237_237805

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[4, -3], [2, 5]]

def matrix2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[-2, 1], [0, 3]]

def scalar : ℤ := 3

theorem matrix_calculation :
  matrix1 + (scalar • matrix2) = ![[-2, 0], [2, 14]] :=
by
  sorry

end matrix_calculation_l237_237805


namespace greatest_integer_third_side_l237_237669

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l237_237669


namespace value_of_P0_l237_237569

noncomputable def P (x : ℝ) : ℝ := sorry

theorem value_of_P0 :
  ∃ (a b c : ℝ),
    (∀ x, x^3 + 3*x^2 + 5*x + 7 = 0 → x = a ∨ x = b ∨ x = c) ∧
    P(a) = b + c ∧
    P(b) = a + c ∧
    P(c) = a + b ∧
    P(a + b + c) = -16 ∧
    P(0) = 11 :=
sorry

end value_of_P0_l237_237569


namespace arithmetic_mean_increased_by_30_l237_237127

/-- Given a set of 5 numbers, if each number is increased by 30,
    prove that the arithmetic mean is increased by 30. -/
theorem arithmetic_mean_increased_by_30
  (b1 b2 b3 b4 b5 : ℝ) :
  let T := b1 + b2 + b3 + b4 + b5 in
  (T + 150) / 5 = (T / 5) + 30 :=
by
  sorry

end arithmetic_mean_increased_by_30_l237_237127


namespace calculate_purple_pants_l237_237611

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def plaid_shirts : ℕ := 3
def non_plaid_non_purple_items : ℕ := 21

theorem calculate_purple_pants : total_pants - (non_plaid_non_purple_items - (total_shirts - plaid_shirts)) = 5 :=
by 
  sorry

end calculate_purple_pants_l237_237611


namespace sum_converges_to_one_l237_237412

noncomputable def series_sum (n: ℕ) : ℝ :=
  if n ≥ 2 then (6 * n^3 - 2 * n^2 - 2 * n + 1) / (n^6 - 2 * n^5 + 2 * n^4 - n^3 + n^2 - 2 * n)
  else 0

theorem sum_converges_to_one : 
  (∑' n, series_sum n) = 1 := by
  sorry

end sum_converges_to_one_l237_237412


namespace solution_is_thirteen_over_nine_l237_237828

noncomputable def check_solution (x : ℝ) : Prop :=
  (3 * x^2 / (x - 2) - (3 * x + 9) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0) ∧
  (x^3 ≠ 3 * x + 1)

theorem solution_is_thirteen_over_nine :
  check_solution (13 / 9) :=
by
  sorry

end solution_is_thirteen_over_nine_l237_237828


namespace speed_in_m_per_s_eq_l237_237359

theorem speed_in_m_per_s_eq : (1 : ℝ) / 3.6 = (0.27777 : ℝ) :=
by sorry

end speed_in_m_per_s_eq_l237_237359


namespace sum_squares_second_15_eq_8215_l237_237274

theorem sum_squares_second_15_eq_8215 :
  let sum_squares := λ n : ℕ, n * (n + 1) * (2 * n + 1) / 6 in
  let sum_first_15 := 1240 in
  sum_squares 30 - sum_first_15 = 8215 :=
by
  let sum_squares := λ n : ℕ, n * (n + 1) * (2 * n + 1) / 6
  let sum_first_15 := 1240
  sorry

end sum_squares_second_15_eq_8215_l237_237274


namespace greatest_third_side_l237_237673

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l237_237673


namespace find_r_l237_237840

theorem find_r (r : ℝ) (h : 5 * (r - 9) = 6 * (3 - 3 * r) + 6) : r = 3 :=
by
  sorry

end find_r_l237_237840


namespace total_revenue_l237_237289

theorem total_revenue (C A : ℕ) (P_C P_A total_tickets adult_tickets revenue : ℕ)
  (hCC : C = 6) -- Children's ticket price
  (hAC : A = 9) -- Adult's ticket price
  (hTT : total_tickets = 225) -- Total tickets sold
  (hAT : adult_tickets = 175) -- Adult tickets sold
  (hTR : revenue = 1875) -- Total revenue
  : revenue = adult_tickets * A + (total_tickets - adult_tickets) * C := sorry

end total_revenue_l237_237289


namespace sum_inequality_l237_237600

theorem sum_inequality (n k : ℕ) :
  1^k + 2^k + ... + n^k ≤ (n^2k - (n-1)^k) / (n^k - (n-1)^k) :=
sorry

end sum_inequality_l237_237600


namespace net_change_and_total_fees_l237_237613

-- Define the daily changes and the loading/unloading fee
def daily_changes : List ℤ := [25, -31, -16, 33, -36, -20]
def fee_per_ton : ℕ := 5

-- Statement of the proof problem
theorem net_change_and_total_fees :
  (daily_changes.sum = -45) ∧ 
  (fee_per_ton * daily_changes.map Int.natAbs |>.sum = 805) := by
  sorry

end net_change_and_total_fees_l237_237613


namespace problem_statement_l237_237808

theorem problem_statement :
  (-2) ^ 0 - (1 / 2) ^ (-1) = -1 := sorry

end problem_statement_l237_237808


namespace bowling_competition_award_sequences_l237_237981

theorem bowling_competition_award_sequences : 
  let players := ['A, 'B, 'C, 'D, 'E],
      matches := [
        (players[4], players[3]), -- E vs D
        (match_winner(players[4], players[3]), players[2]), -- Winner(E_vs_D) vs C
        (match_winner(match_winner(players[4], players[3]), players[2]), players[1]), -- Winner(Winner(E_vs_D)_vs_C) vs B
        (match_winner(match_winner(match_winner(players[4], players[3]), players[2]), players[1]), players[0]) -- Winner(Winner(Winner(E_vs_D)_vs_C)_vs_B) vs A
      ]:
      (2 ^ matches.length) = 16 := by
  sorry

end bowling_competition_award_sequences_l237_237981


namespace tim_minus_tom_l237_237640

def sales_tax_rate : ℝ := 0.07
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def city_tax_rate : ℝ := 0.05

noncomputable def tim_total : ℝ :=
  let price_with_tax := original_price * (1 + sales_tax_rate)
  price_with_tax * (1 - discount_rate)

noncomputable def tom_total : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_sales_tax := discounted_price * (1 + sales_tax_rate)
  price_with_sales_tax * (1 + city_tax_rate)

theorem tim_minus_tom : tim_total - tom_total = -4.82 := 
by sorry

end tim_minus_tom_l237_237640


namespace original_wage_l237_237012

theorem original_wage (W : ℝ) 
  (h1: 1.40 * W = 28) : 
  W = 20 :=
sorry

end original_wage_l237_237012


namespace sin_45_plus_sqrt2_over_2_eq_sqrt2_l237_237648

theorem sin_45_plus_sqrt2_over_2_eq_sqrt2 :
  sin (45 * Real.pi / 180) + (Real.sqrt 2 / 2) = Real.sqrt 2 := by
  sorry

end sin_45_plus_sqrt2_over_2_eq_sqrt2_l237_237648


namespace triangle_area_ratio_l237_237203

-- Define parabola and focus
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def focus : (ℝ × ℝ) := (2, 0)

-- Define the line passing through the focus and intersecting the parabola
def line_through_focus (f : ℝ × ℝ) (a b : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  l (f.1) = f.2 ∧ parabola a.1 a.2 ∧ parabola b.1 b.2 ∧   -- line passes through the focus and intersects parabola at a and b
  l a.1 = a.2 ∧ l b.1 = b.2 ∧ 
  |a.1 - f.1| + |a.2 - f.2| = 3 ∧ -- condition |AF| = 3
  (f = (2, 0))

-- The proof problem
theorem triangle_area_ratio (a b : ℝ × ℝ) (l : ℝ → ℝ) 
  (h_line : line_through_focus focus a b l) :
  ∃ r, r = (1 / 2) := 
sorry

end triangle_area_ratio_l237_237203


namespace total_books_count_l237_237701

theorem total_books_count (total_cost : ℕ) (math_book_cost : ℕ) (history_book_cost : ℕ) 
    (math_books_count : ℕ) (history_books_count : ℕ) (total_books : ℕ) :
    total_cost = 390 ∧ math_book_cost = 4 ∧ history_book_cost = 5 ∧ 
    math_books_count = 10 ∧ total_books = math_books_count + history_books_count ∧ 
    total_cost = (math_book_cost * math_books_count) + (history_book_cost * history_books_count) →
    total_books = 80 := by
  sorry

end total_books_count_l237_237701


namespace division_theorem_l237_237437

noncomputable def p (x : ℝ) : ℝ := x^4 + 3*x^3 - 17*x^2 + 8*x - 12
noncomputable def d (x : ℝ) : ℝ := x - 3
noncomputable def q (x : ℝ) : ℝ := x^3 + 6*x^2 + x + 11
noncomputable def r : ℝ := 21

theorem division_theorem : p = (d * q) + (λ x, r) :=
by
  sorry

end division_theorem_l237_237437


namespace num_valid_A_values_l237_237857

theorem num_valid_A_values: 
  (∃ A, A ∈ {1, 3, 5, 7, 9} ∧ 63 % A = 0) ∧ 
  (A2 % 4 = 0 ∧ to_digit2 A) → 
  finset.card {A ∈ {1, 3, 7, 9} | 63 % A = 0} = 4 :=
by
  sorry

end num_valid_A_values_l237_237857


namespace greatest_third_side_l237_237698

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l237_237698


namespace max_score_top_three_teams_l237_237970

theorem max_score_top_three_teams : 
  ∀ (teams : Finset String) (points : String → ℕ), 
    teams.card = 6 →
    (∀ team, team ∈ teams → (points team = 0 ∨ points team = 1 ∨ points team = 3)) →
    ∃ top_teams : Finset String, top_teams.card = 3 ∧ 
    (∀ team, team ∈ top_teams → points team = 24) := 
by sorry

end max_score_top_three_teams_l237_237970


namespace linear_system_solution_l237_237489

theorem linear_system_solution (x y m : ℝ) (h1 : x + 2 * y = m) (h2 : 2 * x - 3 * y = 4) (h3 : x + y = 7) : 
  m = 9 :=
sorry

end linear_system_solution_l237_237489


namespace magnitude_of_complex_l237_237946

theorem magnitude_of_complex :
  ∀ (z : ℂ), (z = 1 + 2 * complex.i + complex.i ^ 3) → complex.abs z = real.sqrt 2 :=
by
  intros z h
  sorry

end magnitude_of_complex_l237_237946


namespace greatest_third_side_l237_237695

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l237_237695


namespace sum_of_vars_l237_237249

variables (x y z w : ℤ)

theorem sum_of_vars (h1 : x - y + z = 7)
                    (h2 : y - z + w = 8)
                    (h3 : z - w + x = 4)
                    (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
by
  sorry

end sum_of_vars_l237_237249


namespace lake_90_percent_algae_free_l237_237254

def algae_double_coverage (days_before_full : ℕ) : ℝ :=
  2 ^ (-days_before_full)

theorem lake_90_percent_algae_free (day : ℕ) 
  (h : algae_double_coverage (20 - day) = 0.1) :
  day = 17 :=
by
  sorry

end lake_90_percent_algae_free_l237_237254


namespace count_diff_of_two_primes_l237_237922

theorem count_diff_of_two_primes (s : Set ℕ) (p : ℕ → Prop) (n : ℕ) :
  (∀ i, i ∈ s ↔ ∃ n : ℕ, i = 10 * n + 7) →
  (∀ k, k ∈ s → ∃ a b : ℕ, nat.prime a → nat.prime b → a - b = k) →
  #{i ∈ s | ∃ a b : ℕ, nat.prime a ∧ nat.prime b ∧ a - b = i} = 2 :=
by
  -- sorry as the proof would be filled in here
  sorry

end count_diff_of_two_primes_l237_237922


namespace right_triangle_a_value_l237_237977

theorem right_triangle_a_value (a b c : ℝ)
  (h1 : ∠C = 90°)
  (h2 : a = 2 * b)
  (h3 : c^2 = 125) :
  a = 10 :=
by
  sorry

end right_triangle_a_value_l237_237977


namespace tyrone_pennies_l237_237297

-- Define the conditions
def one_dollar_bills : ℕ := 2
def five_dollar_bill : ℕ := 1
def quarters : ℕ := 13
def dimes : ℕ := 20
def nickels : ℕ := 8
def total_money : ℝ := 13

-- Define the values of bills and coins in dollars
def one_dollar_value : ℝ := 1
def five_dollar_value : ℝ := 5
def quarter_value : ℝ := 0.25
def dime_value : ℝ := 0.10
def nickel_value : ℝ := 0.05
def penny_value : ℝ := 0.01

-- Summarize the problem
theorem tyrone_pennies : 
  let total_without_pennies := (one_dollar_bills * one_dollar_value) + 
                               (five_dollar_bill * five_dollar_value) + 
                               (quarters * quarter_value) + 
                               (dimes * dime_value) + 
                               (nickels * nickel_value) in
  let pennies := (total_money - total_without_pennies) / penny_value in
  pennies = 35 := sorry

end tyrone_pennies_l237_237297


namespace point_in_third_quadrant_l237_237093

noncomputable def Z : ℂ := -1 + (1 - complex.i) ^ 2

-- Defining the point corresponding to the complex number Z
noncomputable def pointZ : ℂ := Z

-- Define a predicate indicating the membership of the third quadrant
def isThirdQuadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem point_in_third_quadrant : isThirdQuadrant (pointZ) :=
by
  have hZ : Z = -2 * complex.i := sorry
  show isThirdQuadrant (pointZ)
  rw hZ
  unfold isThirdQuadrant
  split
  { linarith } -- Real part of -2i is 0 which should be < 0
  { linarith } -- Imaginary part of -2i is -2 which should be < 0

end point_in_third_quadrant_l237_237093


namespace simplify_trig_expression_l237_237239

theorem simplify_trig_expression :
  cos (15 * Real.pi / 180) * cos (45 * Real.pi / 180) - cos (75 * Real.pi / 180) * sin (45 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end simplify_trig_expression_l237_237239


namespace sequence_an_l237_237082

noncomputable def a : ℕ → ℕ
| 1       => 2
| (n + 1) => a n + 2

noncomputable def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + a (n + 1)

theorem sequence_an (n : ℕ) (h : 0 < n) : a n = 2 * n :=
by
  sorry

end sequence_an_l237_237082


namespace remainder_of_3_pow_100_add_4_mod_5_equals_0_l237_237708

theorem remainder_of_3_pow_100_add_4_mod_5_equals_0 :
  (3^100 + 4) % 5 = 0 :=
by
  -- Given conditions
  have h1 : 3^1 % 5 = 3 := rfl
  have h2 : 3^2 % 5 = 4 := by norm_num
  have h3 : 3^3 % 5 = 2 := by norm_num
  have h4 : 3^4 % 5 = 1 := by norm_num
  -- Prove the result
  sorry

end remainder_of_3_pow_100_add_4_mod_5_equals_0_l237_237708


namespace log_expression_l237_237452

variable (a : ℝ) (log3 : ℝ → ℝ)
axiom h_a : a = log3 2
axiom log3_8_eq : log3 8 = 3 * log3 2
axiom log3_6_eq : log3 6 = log3 2 + 1

theorem log_expression (log_def : log3 8 - 2 * log3 6 = a - 2) :
  log3 8 - 2 * log3 6 = a - 2 := by
  sorry

end log_expression_l237_237452


namespace rhombus_area_eq_54_l237_237758

theorem rhombus_area_eq_54
  (a b : ℝ) (eq_long_side : a = 4 * Real.sqrt 3) (eq_short_side : b = 3 * Real.sqrt 3)
  (rhombus_diagonal1 : ℝ := 9 * Real.sqrt 3) (rhombus_diagonal2 : ℝ := 4 * Real.sqrt 3) :
  (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2 = 54 := by
  sorry

end rhombus_area_eq_54_l237_237758


namespace counterfeit_bag_l237_237280

noncomputable def identify_counterfeit_bag
  (P : ℕ)  -- Measured actual total weight
  (n : ℕ := 10)
  (true_weight : ℕ := 10) 
  (false_weight : ℕ := 11)
  (bag_sum : ℕ := (n * (n + 1)) / 2) := 
  P - (true_weight * bag_sum)

theorem counterfeit_bag
  (P : ℕ)  -- Measured actual total weight
  (n : ℕ := 10)
  (true_weight : ℕ := 10) 
  (false_weight : ℕ := 11)
  (bag_sum : ℕ := (n * (n + 1)) / 2):
  identify_counterfeit_bag P n true_weight false_weight bag_sum = (P - (true_weight * bag_sum)) := 
by
  refl

end counterfeit_bag_l237_237280


namespace triangle_AD_length_l237_237987

theorem triangle_AD_length (a b : ℝ) (A B C D : Type*) 
  (h1 : Triangle A B C) (h2 : Point D on line_segment B C)
  (angle_BAC_eq : ∠A B C = 2 * ∠A C B)
  (BD_eq : segment_length B D = 1/2 * segment_length D C)
  (BC_eq : segment_length B C = b) 
  (AC_eq : segment_length A C = a)
  (AD_plus_DC_eq : segment_length A D + segment_length D C = segment_length A B) :
  segment_length A D = a - 2/3 * b :=
sorry

end triangle_AD_length_l237_237987


namespace triangle_AC_value_l237_237168

theorem triangle_AC_value (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (angle_A : ∀(a b c : Float), ∠A = 90)
  (BC_eq : ∀(b c : Float), BC = 25)
  (tanC_sinC_relation : ∀(t c : Float), tan C = 3 * sin C) :
  AC = 2500 / 3 := sorry

end triangle_AC_value_l237_237168


namespace average_visitors_per_day_l237_237366

theorem average_visitors_per_day 
  (avg_visitors_sunday : ℕ)
  (avg_visitors_other : ℕ)
  (days_in_month : ℕ)
  (starts_on_sunday : Bool)
  (num_public_holidays : ℕ)
  (holiday_multiplier : ℕ)
  (num_special_events : ℕ)
  (special_event_multiplier : ℕ) :
  avg_visitors_sunday = 510 →
  avg_visitors_other = 240 →
  days_in_month = 30 →
  starts_on_sunday = true →
  num_public_holidays = 2 →
  holiday_multiplier = 2 →
  num_special_events = 1 →
  special_event_multiplier = 3 →
  (4 * avg_visitors_sunday +
   num_public_holidays * (holiday_multiplier * avg_visitors_other) +
   num_special_events * (special_event_multiplier * avg_visitors_other) +
   (days_in_month - 4 - num_public_holidays - num_special_events) * avg_visitors_other)
  / days_in_month = 308 :=
by
  intros,
  sorry

end average_visitors_per_day_l237_237366


namespace class_8_1_total_score_l237_237740

noncomputable def total_score (spirit neatness standard_of_movements : ℝ) 
(weights_spirit weights_neatness weights_standard : ℝ) : ℝ :=
  (spirit * weights_spirit + neatness * weights_neatness + standard_of_movements * weights_standard) / 
  (weights_spirit + weights_neatness + weights_standard)

theorem class_8_1_total_score :
  total_score 8 9 10 2 3 5 = 9.3 :=
by
  sorry

end class_8_1_total_score_l237_237740


namespace find_smaller_number_l237_237288

noncomputable def min (a b : ℕ) : ℕ := if a ≤ b then a else b

theorem find_smaller_number (x y : ℕ) (h1 : 3 * x - y = 20) (h2 : x + y = 48) : min x y = 17 := by
  sorry

end find_smaller_number_l237_237288


namespace students_above_90_l237_237907

theorem students_above_90 (total_students : ℕ) (above_90_chinese : ℕ) (above_90_math : ℕ)
  (all_above_90_at_least_one_subject : total_students = 50 ∧ above_90_chinese = 33 ∧ above_90_math = 38 ∧ 
    ∀ (n : ℕ), n < total_students → (n < above_90_chinese ∨ n < above_90_math)) :
  (above_90_chinese + above_90_math - total_students) = 21 :=
by
  sorry

end students_above_90_l237_237907


namespace solve_for_x_l237_237875

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x (x : ℝ) : 2 * f x - 10 = f (x - 2) ↔ x = 3 :=
by
  sorry

end solve_for_x_l237_237875


namespace strip_dimensions_l237_237819

theorem strip_dimensions (a b c : ℕ) 
    (H1 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
    a = 1 ∧ b + c = 22 := 
begin
    sorry,
end

end strip_dimensions_l237_237819


namespace orthocenter_circumcenter_perpendicular_to_CC_l237_237078

variables {A B C A1 B1 C' O H : Type} [has_coords A B C A1 B1 C' O H]
          [triangle ABC] [non_isosceles_acute_triangle ABC]
          [altitude A1 from A to BC] [altitude B1 from B to AC]
          [midline_parallel_to AB A1B1 intersects_midline_at C' AB]
          [circumcenter O ABC] [orthocenter H ABC] [circumcenter_of ABC O]
          [orthocenter_of ABC H] [line_through O H]

theorem orthocenter_circumcenter_perpendicular_to_CC' :
  ∀ (ABC : triangle) (A B C A1 B1 C' O H : Point),
  non_isosceles_acute_triangle ABC 
  → altitude A1 from A to BC 
  → altitude B1 from B to AC 
  → midline_parallel_to AB A1B1 intersects_midline_at C'
  → circumcenter O ABC 
  → orthocenter H ABC 
  → line_through O H 
  → is_perpendicular (line_through O H) (line_segments (C, C')) :=
sorry

end orthocenter_circumcenter_perpendicular_to_CC_l237_237078


namespace polynomial_remainder_example_l237_237438

noncomputable theory

open Polynomial

theorem polynomial_remainder_example :
  (X^4 + 2 * X^3) % (X^2 + 7 * X + 2) = 33 * X^2 + 10 * X :=
by sorry

end polynomial_remainder_example_l237_237438


namespace least_m_for_no_real_roots_l237_237037

theorem least_m_for_no_real_roots : ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧ m = 4 := 
sorry

end least_m_for_no_real_roots_l237_237037


namespace range_of_m_l237_237487

/-- Given the function y = x^2 - 2x + 3, prove that the interval [1, 2] for m ensures
    the specified extrema on [0, m]. -/
theorem range_of_m {m : ℝ} (h : 1 ≤ m ∧ m ≤ 2) : 
  ∃ f : ℝ → ℝ, (∀ x, f x = x^2 - 2*x + 3) ∧ 
                (∀ x ∈ set.Icc (0:ℝ) m, 2 ≤ f x ∧ f x ≤ 3) := 
by 
  sorry

end range_of_m_l237_237487


namespace josh_age_when_married_l237_237555

theorem josh_age_when_married (J : ℕ) (H1 : ∀ J : ℕ, 5 * J = (J + 30) + 58) : J = 22 :=
by
  have h1 : 5 * J = J + 88 := H1 J
  have h2 : 5 * J - J = 88 := by linarith
  have h3 : 4 * J = 88 := by linarith
  have h4 : J = 88 / 4 := by linarith
  exact h4

end josh_age_when_married_l237_237555


namespace parabola_through_P_l237_237643

-- Define the point P
def P : ℝ × ℝ := (4, -2)

-- Define a condition function for equations y^2 = a*x
def satisfies_y_eq_ax (a : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ y^2 = a * x

-- Define a condition function for equations x^2 = b*y
def satisfies_x_eq_by (b : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ x^2 = b * y

-- Lean's theorem statement
theorem parabola_through_P : satisfies_y_eq_ax 1 ∨ satisfies_x_eq_by (-8) :=
sorry

end parabola_through_P_l237_237643


namespace sum_of_all_possible_quantities_of_stickers_l237_237604

noncomputable def sum_possible_S : ℤ :=
  let possible_S := {S : ℤ | (∃ a : ℤ, S = 6 * a + 5) ∧ (∃ b : ℤ, S = 8 * b + 2) ∧ S < 50}
  in possible_S.sum

theorem sum_of_all_possible_quantities_of_stickers (S : ℤ) :
  (∃ a : ℤ, S = 6 * a + 5) ∧ (∃ b : ℤ, S = 8 * b + 2) ∧ S < 50 →
  ∑ S in {S | (∃ a : ℤ, S = 6 * a + 5) ∧ (∃ b : ℤ, S = 8 * b + 2) ∧ S < 50}, S = 68 :=
begin
  sorry
end

end sum_of_all_possible_quantities_of_stickers_l237_237604


namespace symmetric_origin_l237_237256

def symmetric_point (p : (Int × Int)) : (Int × Int) :=
  (-p.1, -p.2)

theorem symmetric_origin : symmetric_point (-2, 5) = (2, -5) :=
by
  -- proof goes here
  -- we use sorry to indicate the place where the solution would go
  sorry

end symmetric_origin_l237_237256


namespace evaluate_expression_l237_237831

theorem evaluate_expression : (real.sqrt 16) ^ 8 = 256 := 
by 
  let step1 : real.sqrt 16 = 16 ^ (1 / 4) := sorry
  have step2 : (16 ^ (1 / 4)) ^ 8 = 16 ^ ((1 / 4) * 8) := sorry
  have step3 : 16 ^ ((1 / 4) * 8) = 16 ^ 2 := sorry
  have step4 : 16 ^ 2 = 256 := sorry
  sorry

end evaluate_expression_l237_237831


namespace gcd_sum_equality_l237_237447

theorem gcd_sum_equality (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℤ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
by
  sorry

end gcd_sum_equality_l237_237447


namespace find_f_4_l237_237623

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_4 (h : ∀ (x : ℝ), f(x) + 3 * f(1 - x) = 6 * x^2 - 1) : f 4 = 35 / 4 := sorry

end find_f_4_l237_237623


namespace Rebecca_income_approximation_l237_237230

-- Define constants used in the problem
def Jimmy_income : ℝ := 10
def Rebecca_increase : ℝ := 7.78
def combined_percentage : ℝ := 0.55

-- State the theorem with the conditions and the equation to solve for Rebecca's income
theorem Rebecca_income_approximation (R : ℝ) (h1 : R + Rebecca_increase = combined_percentage * ((R + Rebecca_increase) + Jimmy_income)) :
  R ≈ 4.44 :=
by sorry

end Rebecca_income_approximation_l237_237230


namespace number_of_digits_in_x20_l237_237276

theorem number_of_digits_in_x20 (x : ℝ) (hx1 : 10^(7/4) ≤ x) (hx2 : x < 10^2) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 :=
by
  -- Proof goes here
  sorry

end number_of_digits_in_x20_l237_237276


namespace find_angle_C_find_perimeter_l237_237169

variables {A B C : ℝ} {a b c : ℝ}
variable (h1 : 2 * cos C * (a * cos B + b * cos A) = c)
variable (h2 : c = 3 * sqrt 2)
variable (h3 : 1 / 2 * a * b * sin C = 3 * sqrt 3 / 2)

theorem find_angle_C : C = π / 3 :=
sorry

theorem find_perimeter (hC : C = π / 3) : a + b + c = 6 + 3 * sqrt 2 :=
sorry

end find_angle_C_find_perimeter_l237_237169


namespace profit_percent_calculation_l237_237368

theorem profit_percent_calculation:
  let marked_price_per_pen := 1.0 in
  let cost_price := 46.0 in
  let num_pens := 54 in
  let discount := 0.01 in
  let selling_price_per_pen := marked_price_per_pen * (1.0 - discount) in
  let total_selling_price := num_pens * selling_price_per_pen in
  let profit := total_selling_price - cost_price in
  let profit_percent := (profit / cost_price) * 100.0 in
  profit_percent = 16.22 :=
by
  -- Proof skipped
  sorry

end profit_percent_calculation_l237_237368


namespace eigenvalue_and_square_l237_237879

noncomputable def A (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, a], ![-1, 4]]

noncomputable def alpha : Vector ℝ :=
  ![2, 1]

theorem eigenvalue_and_square (a : ℝ) (λ : ℝ) (h1 : A a ⬝ alpha = λ • alpha) : 
  a = 2 ∧ λ = 2 ∧ A 2 ⬝ A 2 = ![![(-1:ℝ), 10], ![-5, 14]] :=
by
  sorry

end eigenvalue_and_square_l237_237879


namespace problem1_problem2_l237_237072

theorem problem1 (a b x y : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y) : 
  (a^2 / x + b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a * y = b * x → (a^2 / x + b^2 / y) = ((a + b)^2 / (x + y))) :=
sorry

theorem problem2 (x : ℝ) (h : 0 < x ∧ x < 1 / 2) :
  (∀ x, 0 < x ∧ x < 1 / 2 → ((2 / x + 9 / (1 - 2 * x)) ≥ 25)) ∧ (2 * (1 - 2 * (1 / 5)) = 9 * (1 / 5) → (2 / (1 / 5) + 9 / (1 - 2 * (1 / 5)) = 25)) :=
sorry

end problem1_problem2_l237_237072


namespace star_polygon_edges_congruent_l237_237526

theorem star_polygon_edges_congruent
  (n : ℕ)
  (α β : ℝ)
  (h1 : ∀ i j : ℕ, i ≠ j → (n = 133))
  (h2 : α = (5 / 14) * β)
  (h3 : n * (α + β) = 360) :
n = 133 :=
by sorry

end star_polygon_edges_congruent_l237_237526


namespace smallest_prime_not_in_form_2a_minus_3b_l237_237439

theorem smallest_prime_not_in_form_2a_minus_3b : ∃ p, prime p ∧ p ≥ 41 ∧ (∀ a b : ℕ, p ≠ |2^a - 3^b|) ∧ (∀ n : ℕ, prime n ∧ ∀ a b : ℕ, n ≠ |2^a - 3^b| → n ≥ 41) :=
sorry

end smallest_prime_not_in_form_2a_minus_3b_l237_237439


namespace process_never_stops_l237_237770

theorem process_never_stops (N : ℕ) (h_digits : N.to_digits = list.repeat 1 900)
  (transformation_rule : ∀ N, N > 100 → ∃ A B : ℕ, B < 100 
     ∧ (N = 100 * A + B ∧ N' = 2 * A + 8 * B) → (N' > 100)) :
  ¬ ∃ k : ℕ, iterate (λ x, 2 * (x / 100) + 8 * (x % 100)) k N < 100 :=
by
  sorry

end process_never_stops_l237_237770


namespace log_2_3_gt_log_3_4_ln_3_plus_4_over_ln_3_gt_2_ln_2_plus_2_over_ln_2_l237_237330

theorem log_2_3_gt_log_3_4 : log 2 3 > log 3 4 := sorry

theorem ln_3_plus_4_over_ln_3_gt_2_ln_2_plus_2_over_ln_2 : 
  log (Real.exp 1) 3 + 4 / log (Real.exp 1) 3 > 2 * log (Real.exp 1) 2 + 2 / log (Real.exp 1) 2 := sorry

end log_2_3_gt_log_3_4_ln_3_plus_4_over_ln_3_gt_2_ln_2_plus_2_over_ln_2_l237_237330


namespace possible_medians_of_S_count_l237_237421

theorem possible_medians_of_S_count:
  ∀ S : Set ℤ, (S.card = 9 ∧ {5, 7, 11, 13, 17, 19} ⊆ S) →
    ∃ medians : Finset ℤ, medians.card = 5 ∧ ∀ m ∈ medians, median S m :=
by
  sorry

end possible_medians_of_S_count_l237_237421


namespace find_t_l237_237820

noncomputable def f : ℝ → ℝ := sorry

axiom monotone_decreasing (x k : ℝ) (hk : 0 < k) : f(x + k) < f(x)

axiom points_condition (M : ℝ × ℝ) (N : ℝ × ℝ) : (M = (-6, 2)) ∧ (N = (2, -6)) ∧ (f(-6) = 2) ∧ (f(2) = -6)

axiom solution_set_condition (t : ℝ) : ∀ x, (|f(x - t) + 2| < 4) ↔ (x ∈ Ioo (-4) 4)

theorem find_t : ∃ t : ℝ, t = 2 :=
by
  use 2
  sorry

end find_t_l237_237820


namespace magnitude_of_z_l237_237954

-- Define the complex number z and the condition
def z : ℂ := 1 + 2 * Complex.i + Complex.i ^ 3

-- The main theorem stating the magnitude of z
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l237_237954


namespace sin_75_is_sqrt_6_add_sqrt_2_div_4_l237_237441

noncomputable def sin_75_angle (a : Real) (b : Real) : Real :=
  Real.sin (75 * Real.pi / 180)

theorem sin_75_is_sqrt_6_add_sqrt_2_div_4 :
  sin_75_angle π (π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_is_sqrt_6_add_sqrt_2_div_4_l237_237441


namespace dot_product_value_l237_237098

variables (a b : EuclideanSpace ℝ (Fin 3)) (angle_ab : Real.Angle)
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 5) (hab_angle : angle_ab = Real.Angle.ofRad (2 * Real.pi / 3))

theorem dot_product_value :
  (2 • a - b) ⬝ a = 13 :=
by
  sorry

end dot_product_value_l237_237098


namespace increasing_sequence_l237_237080

variable {a : ℕ → ℝ}

theorem increasing_sequence (hpos : ∀ n, a n > 0) (hrec : ∀ n, a (n + 1) = 2 * a n) : ∀ n, a (n + 1) > a n :=
by
  intro n
  calc
    a (n + 1) = 2 * a n : hrec n
          ... > a n     : by
              have h : 2 > 1 := by linarith
              have han : 0 < a n := hpos n
              linarith

end increasing_sequence_l237_237080


namespace find_b_l237_237129

theorem find_b (a b : ℝ) (f : ℝ → ℝ) (df : ℝ → ℝ) (x₀ : ℝ)
  (h₁ : ∀ x, f x = a * x + Real.log x)
  (h₂ : ∀ x, f x = 2 * x + b)
  (h₃ : x₀ = 1)
  (h₄ : f x₀ = a) :
  b = -1 := 
by
  sorry

end find_b_l237_237129


namespace mary_weekly_earnings_correct_l237_237214

noncomputable def total_earnings_weekly : ℝ := 691.25

def earnings_calculation (hours_worked : ℝ) (weekend_shifts : ℝ) : ℝ :=
  let P_regular := 10
  let P_overtime := 1.3 * P_regular
  let P_additional := 1.6 * P_regular
  let bonus := 75 * weekend_shifts
  let earnings_regular := if hours_worked <= 40 then hours_worked * P_regular else 40 * P_regular
  let earnings_overtime := if hours_worked > 40 ∧ hours_worked <= 60 then (hours_worked - 40) * P_overtime else if hours_worked > 60 then 20 * P_overtime else 0
  let earnings_additional := if hours_worked > 60 then (hours_worked - 60) * P_additional else 0
  let earnings_before_deductions := earnings_regular + earnings_overtime + earnings_additional + bonus
  let deductions := 
    let base_tax := 0.15
    let insurance := 50
    let additional_tax_401_600 := 0.1 * (min 600 earnings_before_deductions - 400)
    let additional_tax_601 := 0.25 * (earnings_before_deductions - 600)
    if earnings_before_deductions <= 400 then 
      base_tax * earnings_before_deductions + insurance 
    else if earnings_before_deductions <= 600 then 
      base_tax * 400 + insurance + additional_tax_401_600
    else 
      base_tax * 400 + insurance + additional_tax_401_600 + additional_tax_601
  earnings_before_deductions - deductions

theorem mary_weekly_earnings_correct :
  earnings_calculation 70 1 = total_earnings_weekly :=
by
  sorry

end mary_weekly_earnings_correct_l237_237214


namespace triangle_angle_and_side_l237_237138

theorem triangle_angle_and_side (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
  (h2 : a + b = 6)
  (h3 : 1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3)
  : C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := by
  -- proof omitted
  sorry

end triangle_angle_and_side_l237_237138


namespace range_of_x_l237_237894

noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sin x

theorem range_of_x (x : ℝ) (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) 2) :
  f (m * x - 3) + f x < 0 → -3 < x ∧ x < 1 :=
sorry

end range_of_x_l237_237894


namespace perpendicular_lines_l237_237972

noncomputable def plane : Type := sorry
noncomputable def line : Type := sorry

axiom non_coinciding_lines (a b : line) : a ≠ b
axiom distinct_planes (α β : plane) : α ≠ β
axiom parallel_planes (α β : plane) : α ∥ β
axiom perpendicular_line_plane (a : line) (α : plane) : a ⊥ α
axiom line_in_plane (b : line) (β : plane) : b ⊆ β

theorem perpendicular_lines (a b : line) (α β : plane) 
  (h_non_coinciding : non_coinciding_lines a b)
  (h_distinct_planes : distinct_planes α β)
  (h_parallel_planes : parallel_planes α β)
  (h_perpendicular_line_plane : perpendicular_line_plane a α)
  (h_line_in_plane : line_in_plane b β) : a ⊥ b :=
sorry

end perpendicular_lines_l237_237972


namespace checkerboard_sum_l237_237413

def f (i j : Nat) : Nat := 15 * (i - 1) + j
def g (i j : Nat) : Nat := 14 * (j - 1) + i

theorem checkerboard_sum :
  let matches := [(i, j) | i <- List.range 14, j <- List.range 15, f i j = g i j]
  let sum := matches.map (λ (i, j) => f i j).sum
  sum = 210 :=
by sorry

end checkerboard_sum_l237_237413


namespace subject_combinations_l237_237960

theorem subject_combinations :
  let compulsory_subjects := {Chinese, Mathematics, ForeignLanguage}
  let optional_subjects1 := {Physics, History}
  let optional_subjects2 := {Chemistry, Biology, Politics, Geography}

  ∃ total_combinations : ℕ,
  ( (combinatorics.choose 2 1) * (combinatorics.choose 4 2) +
    (combinatorics.choose 2 2) * (combinatorics.choose 4 1) )
  = total_combinations ∧ total_combinations = 16 :=
by
  sorry

end subject_combinations_l237_237960


namespace number_of_arrangements_l237_237325

/-- There are 6 distinct people: A, B, C, D, E, and F.
    We need to find the number of ways to arrange them
    in a line such that E and F are next to each other. -/
theorem number_of_arrangements (A B C D E F : Type) :
  ∃ n : ℕ, n = 240 :=
by
  let distinct_people := [A, B, C, D, E, F]
  let EF_next_to_each_other := [([A, B, C, D] ++ [E, F])]
  let number_of_ways := factorial 5 * 2
  exact ⟨number_of_ways, rfl⟩

end number_of_arrangements_l237_237325


namespace range_of_a_l237_237259

def has_extreme_values_on_R (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ f a = f b

noncomputable def cubic_function (a : ℝ) : ℝ → ℝ := 
  λ x, x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) :
  has_extreme_values_on_R (cubic_function a) ↔ (a < -3 ∨ a > 6) :=
sorry

end range_of_a_l237_237259


namespace sequence_general_term_l237_237624

theorem sequence_general_term (n : ℕ) :
  let a_n := (-1)^(n+1) * (2*n - 1) in
  (a_n ∈ {x : ℤ | ∃ m : ℕ, (m = 0 → x = -1) ∧
    (∀ k, m = k + 1 → (x = 3 * k - 1) ∨ (x = 3 * k + 1))}) :=
sorry

end sequence_general_term_l237_237624


namespace pow_mod_eq_l237_237711

theorem pow_mod_eq :
  11 ^ 2023 % 5 = 1 :=
by
  sorry

end pow_mod_eq_l237_237711


namespace problem_inequality_l237_237187

theorem problem_inequality
  (n : ℕ) (h_n : n > 2) (h_even : n % 2 = 0)
  (a : ℕ → ℝ) (h_increasing : ∀ k, 1 ≤ k → k < n → a k < a (k + 1))
  (h_diff_le_one : ∀ k, 1 ≤ k → k < n → a (k + 1) - a k ≤ 1) :
  let A := { (i, j) | 1 ≤ i ∧ i < j ∧ j ≤ n ∧ (j - i) % 2 = 0 }
  let B := { (i, j) | 1 ≤ i ∧ i < j ∧ j ≤ n ∧ (j - i) % 2 = 1 } in
  (∏ (p : ℕ × ℕ) in A, a p.2 - a p.1) > (∏ (p : ℕ × ℕ) in B, a p.2 - a p.1) := sorry

end problem_inequality_l237_237187


namespace probability_five_correct_is_zero_l237_237283

-- Let's define the problem in Lean.
theorem probability_five_correct_is_zero :
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F' }
  let people := { 'P1', 'P2', 'P3', 'P4', 'P5', 'P6' }
  let permutations := letters.prod_map.people -- all possible mappings
  let correct_distribution := 1 / (people.card)! -- probability of one correct distribution
  let exactly_five_correct (mapping : people → letters) := 
    (∃ person, ∀ p i ≠ person, mapping p = letters p) ∧ 
    ∃ wrong_person, mapping wrong_person ≠ letters wrong_person
  in ∀ mapping ∈ permutations, exactly_five_correct mapping → 0 :=
by
  sorry

end probability_five_correct_is_zero_l237_237283


namespace sum_of_vars_l237_237250

variables (x y z w : ℤ)

theorem sum_of_vars (h1 : x - y + z = 7)
                    (h2 : y - z + w = 8)
                    (h3 : z - w + x = 4)
                    (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
by
  sorry

end sum_of_vars_l237_237250


namespace sum_of_m_and_n_eq_zero_l237_237295

variables {m n : ℝ}

def line_equation (x y : ℝ) := x + y = 0

def circle_centers_on_line (Mx My Nx Ny : ℝ) := 
  line_equation  (Mx) (My) ∧ 
  line_equation (Nx) (Ny)

theorem sum_of_m_and_n_eq_zero (M : ℝ × ℝ) (N : ℝ × ℝ):
  (line_equation M.1 M.2) →
  (line_equation N.1 N.2) →
  M.2 = 1 →
  N.1 = -1 →
  ∃ (m n : ℝ), M = (m,1) ∧ N = (-1,n) ∧ (m + n = 0) :=
begin
  intros hM hN hM1 hN1,
  sorry
end

end sum_of_m_and_n_eq_zero_l237_237295


namespace pow_mod_eq_l237_237712

theorem pow_mod_eq :
  11 ^ 2023 % 5 = 1 :=
by
  sorry

end pow_mod_eq_l237_237712


namespace polygon_division_l237_237864

/-- Given 500 points inside a convex 1000-sided polygon, along with the polygon's vertices (a total of 1500 points), none of which are collinear, the polygon is divided into triangles with these 1500 points as the vertices of the triangles. Prove that the number of triangles formed is 1998. -/
theorem polygon_division (P : Finset (Fin 1500)) (h₁ : P.card = 1500) (h₂ : ∀ p1 p2 p3 ∈ P, ¬ collinear p1 p2 p3)
  (h₃ : ∑ h ∈ (P.filter (λ x, x ≤ 999)), 1 = 1000)
  (h₄ : ∑ h ∈ (P.filter (λ x, x ≥ 1000)), 1 = 500) :
  ∃ n, n = 1998 ∧ ∑ (t : { t // t ∈ triangles P }), 1 = n :=
by {
  sorry
}

end polygon_division_l237_237864


namespace percentage_of_female_students_l237_237145

noncomputable theory
open_locale classical

variables {n : ℕ} -- Let n be the total number of students
variables {m f : ℕ} -- m is the number of male students, f is the number of female students

-- Assume conditions
def prob_conditions (n m f : ℕ) :=
  m = 48 * n / 100 ∧ 
  f = 52 * n / 100 ∧
  0.5 * m + 0.8 * f = 0.6 * n

theorem percentage_of_female_students (n m f : ℕ) (h : prob_conditions n m f) : 
  f * 100 / n = 52 :=
begin
  sorry, 
end

end percentage_of_female_students_l237_237145


namespace jims_buicks_l237_237552

theorem jims_buicks : 
  ∀ F B C : ℕ, F + B + C = 301 ∧ B = 4 * F ∧ F = 2 * C + 3 → B = 220 := 
by
  intros F B C h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  sorry

end jims_buicks_l237_237552


namespace figures_can_be_drawn_l237_237785

structure Figure :=
  (degrees : List ℕ) -- List of degrees of the vertices in the graph associated with the figure.

-- Define a predicate to check if a figure can be drawn without lifting the pencil and without retracing
def canBeDrawnWithoutLifting (fig : Figure) : Prop :=
  let odd_degree_vertices := fig.degrees.filter (λ d => d % 2 = 1)
  odd_degree_vertices.length = 0 ∨ odd_degree_vertices.length = 2

-- Define the figures A, B, C, D with their degrees (examples, these should match the problem's context)
def figureA : Figure := { degrees := [2, 2, 2, 2] }
def figureB : Figure := { degrees := [2, 2, 2, 2, 4] }
def figureC : Figure := { degrees := [3, 3, 3, 3] }
def figureD : Figure := { degrees := [4, 4, 2, 2] }

-- State the theorem that figures A, B, and D can be drawn without lifting the pencil
theorem figures_can_be_drawn :
  canBeDrawnWithoutLifting figureA ∧ canBeDrawnWithoutLifting figureB ∧ canBeDrawnWithoutLifting figureD :=
  by sorry -- Proof to be completed

end figures_can_be_drawn_l237_237785


namespace parallel_lines_not_coincident_l237_237491

theorem parallel_lines_not_coincident (x y : ℝ) (m : ℝ) :
  (∀ y, x + (1 + m) * y = 2 - m ∧ ∀ y, m * x + 2 * y + 8 = 0) → (m =1) := 
sorry

end parallel_lines_not_coincident_l237_237491


namespace min_toddlers_l237_237364

theorem min_toddlers (n : ℕ) (teeth : ℕ → ℕ)
  (H1 : ∑ i in finset.range n, teeth i = 90)
  (H2 : ∀ i j, i ≠ j → teeth i + teeth j ≤ 9) :
  n ≥ 23 :=
sorry

end min_toddlers_l237_237364


namespace total_playing_time_scenarios_l237_237529

theorem total_playing_time_scenarios :
  (∑ (x y : ℕ) in (finset.Icc (0 : ℕ) 33).product (finset.Icc (0 : ℕ) 20),
    (if 7 * x + 13 * y = 270 then (nat.choose (x + 3) 3 * nat.choose (y + 2) 2) else 0)) = 42244 :=
by sorry

end total_playing_time_scenarios_l237_237529


namespace slope_of_tangent_at_minus_1_l237_237508

theorem slope_of_tangent_at_minus_1
  (c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (x - 2) * (x^2 + c))
  (h_extremum : deriv f 1 = 0) :
  deriv f (-1) = 8 :=
by
  sorry

end slope_of_tangent_at_minus_1_l237_237508


namespace workers_for_type_A_total_processing_cost_l237_237013

/-- Given 50 workers in a workshop where each worker processes 30 parts of type A or 20 parts of type B per day. 
A car set requires 7 parts of type A and 2 parts of type B each day. Prove that 35 workers are needed to 
process type A parts to meet the production requirements. -/
theorem workers_for_type_A (x : ℕ) :
  ∃ x, (30 * x / 7) = (20 * (50 - x) / 2) ∧ x = 35 :=
by
  existsi 35
  sorry

/-- Given 50 workers in total, with 35 workers working on type A parts and 15 on type B parts, 
where the cost of processing one part of type A is $10 and one part of type B is $12. 
Prove that the total processing cost for these 50 workers in one day is 14100 yuan. -/
theorem total_processing_cost (total_cost : ℕ) :
  let cost_A := 10
      cost_B := 12
      num_workers_A := 35
      num_workers_B := 15
      parts_per_worker_A := 30
      parts_per_worker_B := 20
      total_cost := (num_workers_A * parts_per_worker_A * cost_A) + (num_workers_B * parts_per_worker_B * cost_B)
  in total_cost = 14100 :=
by
  let cost_A := 10
  let cost_B := 12
  let num_workers_A := 35
  let num_workers_B := 15
  let parts_per_worker_A := 30
  let parts_per_worker_B := 20
  let total_cost := (num_workers_A * parts_per_worker_A * cost_A) + (num_workers_B * parts_per_worker_B * cost_B)
  sorry

end workers_for_type_A_total_processing_cost_l237_237013


namespace int_coordinate_differences_l237_237458

def polygon := set (ℝ × ℝ)
def area (M : polygon) := sorry  -- Placeholder definition for area

theorem int_coordinate_differences (M : polygon) (n : ℕ) (h : area M > n) :
  ∃ (P : fin (n+1) → (ℝ × ℝ)),
    ∀ (i j : fin (n+1)), 
      ∃ (k l : ℤ), 
        P i = (k, l) ∧ 
        P j = (k, l - 1) ∨ 
        P j = (k - 1, l) ∨
        P j = (k - 1, l - 1) := 
sorry

end int_coordinate_differences_l237_237458


namespace magnitude_of_z_l237_237952

-- Define the complex number z and the condition
def z : ℂ := 1 + 2 * Complex.i + Complex.i ^ 3

-- The main theorem stating the magnitude of z
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l237_237952


namespace percentage_deposit_l237_237788

theorem percentage_deposit (deposited : ℝ) (initial_amount : ℝ) (amount_deposited : ℝ) (P : ℝ) 
  (h1 : deposited = 750) 
  (h2 : initial_amount = 50000)
  (h3 : amount_deposited = 0.20 * (P / 100) * (0.25 * initial_amount))
  (h4 : amount_deposited = deposited) : 
  P = 30 := 
sorry

end percentage_deposit_l237_237788


namespace remainder_division_39_l237_237358

theorem remainder_division_39 (N : ℕ) (k m R1 : ℕ) (hN1 : N = 39 * k + R1) (hN2 : N % 13 = 5) (hR1_lt_39 : R1 < 39) :
  R1 = 5 :=
by sorry

end remainder_division_39_l237_237358


namespace problem_statement_l237_237450

theorem problem_statement 
  (a b : ℝ) 
  (h1 : 3^a = 2) 
  (h2 : 5^b = 3) : 
  (a + 1/a > b + 1/b) ∧ (a + a^b < b + b^a) :=
  sorry

end problem_statement_l237_237450


namespace sum_f_a_n_8_vals_l237_237464

def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + (8/3) * x + 1

def a (n : ℕ) : ℝ := 2 * n - 7

theorem sum_f_a_n_8_vals :
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) + f (a 8) = 8 :=
by
  sorry

end sum_f_a_n_8_vals_l237_237464


namespace sum_first_9_terms_arithmetic_seq_l237_237085

variable {a : ℕ → ℝ} {l1 : ℝ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) - a n = d

def line_l1_passes_through_point_5_3 (l1 : ℝ → ℝ) : Prop :=
  l1 5 = 3

theorem sum_first_9_terms_arithmetic_seq (h_arith : is_arithmetic_sequence a)
  (h_line : ∀ n, a n = l1 n) (h_l1 : line_l1_passes_through_point_5_3 l1) :
  (finset.range 9).sum (λ n, a (n + 1)) = 27 :=
sorry

end sum_first_9_terms_arithmetic_seq_l237_237085


namespace phones_left_is_7500_l237_237370

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end phones_left_is_7500_l237_237370


namespace points_on_circle_l237_237445

theorem points_on_circle (t : ℝ) : 
  ( (2 - 3 * t^2) / (2 + t^2) )^2 + ( 3 * t / (2 + t^2) )^2 = 1 := 
by 
  sorry

end points_on_circle_l237_237445


namespace chords_are_diameters_l237_237141

theorem chords_are_diameters (C : Circle) (chords : Finset (Chord C)) :
(∀ c1 ∈ chords, ∃ c2 ∈ chords, midpoint c1 ∈ c2) →
(∀ c ∈ chords, is_diameter c) :=
begin
  sorry
end

end chords_are_diameters_l237_237141


namespace pieces_given_l237_237661

def pieces_initially := 38
def pieces_now := 54

theorem pieces_given : pieces_now - pieces_initially = 16 := by
  sorry

end pieces_given_l237_237661


namespace rational_expression_is_rational_l237_237596

theorem rational_expression_is_rational (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ r : ℚ, 
    r = Real.sqrt ((1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2)) :=
sorry

end rational_expression_is_rational_l237_237596


namespace total_cookies_is_58_l237_237794

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

end total_cookies_is_58_l237_237794


namespace total_missing_keys_l237_237996

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end total_missing_keys_l237_237996


namespace binomial_coefficient_bounds_l237_237729

theorem binomial_coefficient_bounds :
  let binom : ℕ → ℕ → ℕ := λ n k, Nat.choose n k
  in (2^100 / (10 * Real.sqrt 2) < binom 100 50 ∧ binom 100 50 < 210) := 
by {
  sorry
}

end binomial_coefficient_bounds_l237_237729


namespace areas_of_shared_parts_l237_237536

-- Define the areas of the non-overlapping parts
def area_non_overlap_1 : ℝ := 68
def area_non_overlap_2 : ℝ := 110
def area_non_overlap_3 : ℝ := 87

-- Define the total area of each circle
def total_area : ℝ := area_non_overlap_2 + area_non_overlap_3 - area_non_overlap_1

-- Define the areas of the shared parts A and B
def area_shared_A : ℝ := total_area - area_non_overlap_2
def area_shared_B : ℝ := total_area - area_non_overlap_3

-- Prove the areas of the shared parts
theorem areas_of_shared_parts :
  area_shared_A = 19 ∧ area_shared_B = 42 :=
by
  sorry

end areas_of_shared_parts_l237_237536


namespace sum_of_number_and_square_l237_237511

theorem sum_of_number_and_square (n : ℕ) (h : n = 8) : n + n^2 = 72 :=
by {
  rw [h],
  norm_num,
  sorry
}

end sum_of_number_and_square_l237_237511


namespace find_number_is_9_l237_237934

noncomputable def number (y : ℕ) : ℕ := 3^(12 / y)

theorem find_number_is_9 (y : ℕ) (h_y : y = 6) (h_eq : (number y)^y = 3^12) : number y = 9 :=
by
  sorry

end find_number_is_9_l237_237934


namespace magnitude_of_z_l237_237950

noncomputable def z : ℂ := 1 + 2 * complex.i + complex.i ^ 3

theorem magnitude_of_z : complex.abs z = real.sqrt 2 := 
by
  -- this 'sorry' is a placeholder for the actual proof
  sorry

end magnitude_of_z_l237_237950


namespace total_extracurricular_hours_l237_237179

def soccer_days := 3
def soccer_hours_per_day := 2
def band_days := 2
def band_hours_per_day := 1.5

theorem total_extracurricular_hours:
  (soccer_days * soccer_hours_per_day) + (band_days * band_hours_per_day) = 9 := 
by
  sorry

end total_extracurricular_hours_l237_237179


namespace even_numbers_probability_different_digits_l237_237787

theorem even_numbers_probability_different_digits :
  let even_numbers := list.range' 10 89 |>.filter (λ n, n % 2 = 0),
      numbers_with_same_digits := [22, 44, 66, 88],
      total_even_numbers := even_numbers.length,
      count_same_digits := numbers_with_same_digits.length,
      probability_different_digits := 1 - (count_same_digits / total_even_numbers : ℚ) in
  probability_different_digits = 41 / 45 := by
  sorry

end even_numbers_probability_different_digits_l237_237787


namespace conclusion_may_not_be_correct_l237_237763

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {n_ge_3 : n ≥ 3}
variable {x_avg x'_avg s_squared s'_squared m m' t t' : ℝ}

-- Hypotheses about the original data
hypothesis (average_def : x_avg = (∑ i, x i) / n)
hypothesis (variance_def : s_squared = (∑ i, (x i - x_avg)^2) / n)
hypothesis (range_def : m = (finset.max (finset.image (λ i, x i) finset.univ) - finset.min (finset.image (λ i, x i) finset.univ)))
hypothesis (median_def : t = if even n then ((x (n/2 - 1) + x (n/2)) / 2) else x (n/2))

-- Data after removing min and max
variable {x' : Fin (n - 2) → ℝ}

hypothesis (x'_def : ∀ i, x' i = x (i+1)) -- assuming x is sorted and non-equal elements are guaranteed
hypothesis (average'_def : x'_avg = (∑ i, x' i) / (n - 2))
hypothesis (variance'_def : s'_squared = (∑ i, (x' i - x'_avg)^2) / (n - 2))
hypothesis (range'_def : m' = (finset.max (finset.image (λ i, x' i) finset.univ) - finset.min (finset.image (λ i, x' i) finset.univ)))
hypothesis (median'_def : t' = if even (n - 2) then ((x' ((n-2)/2 - 1) + x' ((n-2)/2)) / 2) else x' ((n-2)/2))

theorem conclusion_may_not_be_correct : x_avg ≠ x'_avg :=
by
  sorry

end conclusion_may_not_be_correct_l237_237763


namespace time_to_pass_bridge_approx_l237_237009

-- Define the length of the train, speed of the train, and length of the bridge
def train_length : ℝ := 850 -- in meters
def train_speed_km_per_hr : ℝ := 95 -- in kilometers per hour
def bridge_length : ℝ := 325 -- in meters

-- Define the conversion factors
def meters_per_kilometer : ℝ := 1000
def seconds_per_hour : ℝ := 3600

-- Calculate the total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- Convert speed from km/h to m/s
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (meters_per_kilometer / seconds_per_hour)

-- Calculate the time to pass the bridge
def time_to_pass_bridge : ℝ := total_distance / train_speed_m_per_s

-- State the theorem that the time to pass the bridge is approximately 44.52 seconds
theorem time_to_pass_bridge_approx : abs (time_to_pass_bridge - 44.52) < 0.01 := by sorry

end time_to_pass_bridge_approx_l237_237009


namespace volume_ratio_l237_237741

noncomputable def ratio_of_volumes (α : Real) : Real :=
  (2 * (cos α)^2) / ((cos (2 * α)) * (tan α))

theorem volume_ratio (R : Real) (h : Real) (r : Real) (α : Real)
  (h_eq : h = R * tan α)
  (r_eq : r = R * (sqrt (cos (2 * α))) / (cos α))
  (V_cone : Real := (1 / 3) * π * r^2 * h)
  (V_hemisphere : Real := (2 / 3) * π * R^3) :
  V_hemisphere / V_cone = ratio_of_volumes α :=
sorry

end volume_ratio_l237_237741


namespace greatest_possible_third_side_l237_237680

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l237_237680


namespace find_line_through_and_perpendicular_l237_237051

def point (x y : ℝ) := (x, y)

def passes_through (P : ℝ × ℝ) (a b c : ℝ) :=
  a * P.1 + b * P.2 + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) :=
  a1 * a2 + b1 * b2 = 0

theorem find_line_through_and_perpendicular :
  ∃ c : ℝ, passes_through (1, -1) 1 1 c ∧ is_perpendicular 1 (-1) 1 1 → 
  c = 0 :=
by
  sorry

end find_line_through_and_perpendicular_l237_237051


namespace lemonade_cups_count_l237_237961

theorem lemonade_cups_count :
  ∃ x y : ℕ, x + y = 400 ∧ x + 2 * y = 546 ∧ x = 254 :=
by
  sorry

end lemonade_cups_count_l237_237961


namespace find_length_PO_l237_237999

-- Definitions for Lean 4 statement reflecting the conditions
def semicircle (O : Point) (A B C D : Point) (O_center : O = midpoint A B) (on_semicircle : C ∈ semicircle O A B ∧ D ∈ semicircle O A B) : Prop :=
  semicircle_axioms O A B C D O_center on_semicircle

def convex_quadrilateral (A B C D : Point) : Prop :=
  convex_quadrilateral_axioms A B C D

def intersect_diagonals (A B C D Q : Point) : Prop :=
  Q = intersection (line_through A C) (line_through B D)

def tangent_lines (C D P : Point) (O : Point) : Prop :=
  tangent O C P ∧ tangent O D P

def angle_measure (A Q B : Point) (alpha : ℝ) : Prop :=
  angle A Q B = 2 * alpha

def diameter_length (AB : ℝ) : Prop :=
  AB = 2

noncomputable def length_PO (P O : Point) (length : ℝ) : Prop :=
  length = distance P O

-- Combining conditions into a theorem to be proved
theorem find_length_PO
  (O A B C D P Q : Point)
  (O_center : O = midpoint A B)
  (on_semicircle : C ∈ semicircle O A B ∧ D ∈ semicircle O A B)
  (convex_quad : convex_quadrilateral A B C D)
  (diagonals_intersect : intersect_diagonals A B C D Q)
  (tangents_intersect : tangent_lines C D P O)
  (angle_condition : angle_measure A Q B alpha)
  (diameter_condition : diameter_length 2) :
  length_PO P O (2 * sqrt(3) / 3) :=
sorry

end find_length_PO_l237_237999


namespace max_a_inequality_l237_237510

theorem max_a_inequality :
  ∃ (a : ℕ), (∀ (n : ℕ), n > 0 → (∑ i in finseq.range (2*n + 1), 1 / ((n + 1) + i : ℝ)) > a / 24) ∧ a = 25 :=
by
  use 25
  intros n hn
  have h1 : ∑ i in finset.range (2*n + 1), 1 / ((n + 1) + i : ℝ) > 25 / 24 := sorry
  exact h1

end max_a_inequality_l237_237510


namespace max_value_f_inequality_l237_237108

theorem max_value_f_inequality :
  ∃ a, (∀ x ∈ set.Icc 3 5, (2 * x) / (x - 1) ≥ a) ∧ a = 5 / 2 := sorry

end max_value_f_inequality_l237_237108


namespace solve_equation_l237_237433

-- Define the given equation
def equation (x : ℝ) : Prop := (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = -3

-- State the theorem indicating the solutions to the equation
theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  equation x ↔ x = -2 ∨ x = 3 / 2 :=
sorry

end solve_equation_l237_237433


namespace derivative_at_pi_div_3_l237_237076

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 2) * Real.sin x - Real.cos x

theorem derivative_at_pi_div_3 :
  deriv f (π / 3) = (1 / 2) * (1 + Real.sqrt 2 + Real.sqrt 3) :=
by
  sorry

end derivative_at_pi_div_3_l237_237076


namespace H_is_orthocenter_l237_237269

open EuclideanGeometry

variables (A B C D H O : Point)

-- Our assumptions
axiom tetrahedron (ABCD : Tetrahedron A B C D)
axiom insphere (s : Sphere) (inscribed : IsInscribed s ABCD)
axiom touching_face_at_H : TouchingAt s (⟨A, B, C⟩ : Plane) H
axiom another_sphere (t : Sphere) (touches_face_at_O : TouchingAt t (⟨A, B, C⟩ : Plane) O)
axiom circumcenter_O : IsCircumcenter O (Triangle A B C)

-- The proof goal
theorem H_is_orthocenter :
  IsOrthocenter H (Triangle A B C) := sorry

end H_is_orthocenter_l237_237269


namespace max_tension_of_pendulum_l237_237764

theorem max_tension_of_pendulum 
  (m g L θ₀ : ℝ) 
  (h₀ : θ₀ < π / 2) 
  (T₀ : ℝ) 
  (no_air_resistance : true) 
  (no_friction : true) : 
  ∃ T_max, T_max = m * g * (3 - 2 * Real.cos θ₀) := 
by 
  sorry

end max_tension_of_pendulum_l237_237764


namespace longest_side_similar_triangle_l237_237641

-- Define the sides of the original triangle
def original_sides : list ℝ := [8, 10, 12]

-- Define the perimeter of the similar triangle
def similar_triangle_perimeter : ℝ := 150

-- Define the scaling factor to find the longest side in the similar triangle
theorem longest_side_similar_triangle : (original_sides.max' (by decide)) * (similar_triangle_perimeter / original_sides.sum) = 60 := by
  sorry

end longest_side_similar_triangle_l237_237641


namespace t_bounds_f_bounds_l237_237454

noncomputable def t (x : ℝ) : ℝ := 3^x

noncomputable def f (x : ℝ) : ℝ := 9^x - 2 * 3^x + 4

theorem t_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (1/3 ≤ t x ∧ t x ≤ 9) :=
sorry

theorem f_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (3 ≤ f x ∧ f x ≤ 67) :=
sorry

end t_bounds_f_bounds_l237_237454


namespace rectangle_graph_points_l237_237178

/-- 
  The solution set of pairs (w, l) for which the product equals 18 
  and w, l are positive integers is {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)}.
-/
theorem rectangle_graph_points :
  { p : ℕ × ℕ // p.1 * p.2 = 18 } =
  { ⟨1, 18⟩, ⟨2, 9⟩, ⟨3, 6⟩, ⟨6, 3⟩, ⟨9, 2⟩, ⟨18, 1⟩ } :=
sorry

end rectangle_graph_points_l237_237178


namespace determine_n_l237_237874

theorem determine_n (n : ℕ) (h1 : n > 2020) (h2 : ∃ m : ℤ, (n - 2020) = m^2 * (2120 - n)) : 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := 
sorry

end determine_n_l237_237874


namespace find_value_l237_237881

noncomputable section

variables {a b c x y z : ℝ}

def conditions (a b c x y z : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
  a^2 + b^2 + c^2 = 49 ∧
  x^2 + y^2 + z^2 = 16 ∧
  ax + by + cz = 28

theorem find_value (h : conditions a b c x y z) : (a + b + c) / (x + y + z) = 7 / 4 :=
sorry

end find_value_l237_237881


namespace math_problem_l237_237430

-- Define positive integers a and n
def positive_integers := {n : ℕ // n > 0}

-- Condition for a and n: a > 2
def condition (a n : positive_integers) : Prop := (a.val > 2)

-- Resulting solution given the conditions
def valid_pair : Prop := ∃ l : positive_integers, (a, n) = (2 ^ l.val - 1, 2)

-- Resulting proof statement
theorem math_problem (a n : positive_integers) (h : condition a n) : valid_pair := 
sorry

end math_problem_l237_237430


namespace count_good_numbers_from_1_to_100_l237_237058

-- Define what it means for a number to be a "good number".
def is_good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a + b + a * b

-- Main theorem stating the number of good numbers between 1 and 100
theorem count_good_numbers_from_1_to_100 : 
  (finset.filter is_good_number (finset.range 101)).card = 74 :=
by
  sorry

end count_good_numbers_from_1_to_100_l237_237058


namespace inequality_bound_l237_237335

theorem inequality_bound 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : a ≤ 1)
  (hb : 0 ≤ b) (hc : b ≤ 1)
  (hc : 0 ≤ c) (hd : c ≤ 1)
  (hd : 0 ≤ d) (ha2 : d ≤ 1) : 
  ab * (a - b) + bc * (b - c) + cd * (c - d) + da * (d - a) ≤ 8/27 := 
by
  sorry

end inequality_bound_l237_237335


namespace tetrahedron_volume_of_cube_l237_237743

theorem tetrahedron_volume_of_cube (s : ℝ) (h₁ : s = 8) (h₂ : ∀ a b : ℕ, s^3 = 512 ∧ (s * s) / 2 = 32 ∧ ((s * s) / 2 * s / 3) = 85.33 ∧ 4 * ((s * s) / 2 * s / 3) = 341.33) :
  let volume := s^3 - 4 * ((s * s) / 2 * s / 3)
  volume = 170.67 :=
by 
  intros
  have h₃ : s^3 = 512 := by linarith [h₁]
  have ht : ((s * s) / 2 * s / 3) = 85.33 := by linarith [h₂ 1 2]
  have h₄ : 4 * ((s * s) / 2 * s / 3) = 341.33 := by linarith [h₂ 3 4]
  have volume := s^3 - 4 * ((s * s) / 2 * s / 3)
  show volume = 170.67
  sorry

end tetrahedron_volume_of_cube_l237_237743


namespace regression_lines_meet_at_avg_l237_237290

-- Let's define the conditions.
variable (n1 n2 : ℕ) -- Number of experiments
variable (l1 l2 : ℝ → ℝ) -- Regression lines
variable (s t : ℝ) -- Average values

-- Hypotheses (conditions)
axiom h1 : n1 = 10 -- Student A conducted 10 experiments
axiom h2 : n2 = 15 -- Student B conducted 15 experiments
axiom h_avg_x : ∀ i ∈ {1, 2}, avg (dataset_i i x) = s -- Average x-value in both experiments
axiom h_avg_y : ∀ i ∈ {1, 2}, avg (dataset_i i y) = t -- Average y-value in both experiments
axiom h_reg_line : ∀ i ∈ {1, 2}, regression_line (dataset_i i) = l_i -- Regression line through dataset

-- The theorem to prove.
theorem regression_lines_meet_at_avg (s t : ℝ) (n1 n2 : ℕ) (l1 l2 : ℝ → ℝ) 
  (h1 : n1 = 10) (h2 : n2 = 15) 
  (h_avg_x : ∀ i ∈ {1, 2}, avg (dataset_i i x) = s)
  (h_avg_y : ∀ i ∈ {1, 2}, avg (dataset_i i y) = t)
  (h_reg_line : ∀ i ∈ {1, 2}, regression_line (dataset_i i) = l_i) : 
  l1 s = t ∧ l2 s = t :=
sorry

end regression_lines_meet_at_avg_l237_237290


namespace emily_dog_count_l237_237045

theorem emily_dog_count (dogs : ℕ) 
  (food_per_day_per_dog : ℕ := 250) 
  (vacation_days : ℕ := 14)
  (total_food_kg : ℕ := 14)
  (kg_to_grams : ℕ := 1000) 
  (total_food_grams : ℕ := total_food_kg * kg_to_grams)
  (food_needed_per_dog : ℕ := food_per_day_per_dog * vacation_days) 
  (total_food_needed : ℕ := dogs * food_needed_per_dog) 
  (h : total_food_needed = total_food_grams) : 
  dogs = 4 := 
sorry

end emily_dog_count_l237_237045


namespace calculate_conditional_probability_l237_237152

-- Definition of the problem conditions
def total_scenarios : ℕ := 16
def neither_chooses_B : ℕ := 9 
def scenarios_with_at_least_one_B : ℕ := total_scenarios - neither_chooses_B
def different_attractions (A B : ℕ) (chosen_A_B : Fin 4) (chosen_B_A : Fin 4) : Prop :=
  chosen_A_B ≠ chosen_B_A
def at_least_one_chooses_B (A B : Fin 4) (chosen_A_B : Fin 4) (chosen_B_A : Fin 4) : Prop :=
  chosen_A_B = 1 ∨ chosen_B_A = 1

-- Proof statement
theorem calculate_conditional_probability :
  ∀ (A B : Fin 4), 
  let total := total_scenarios,
  let neither_B := neither_chooses_B,
  let M := scenarios_with_at_least_one_B,
  let MN := 6,
  M = total - neither_B →
  P(N|M) = MN / M :=
  sorry

end calculate_conditional_probability_l237_237152


namespace harold_wrapping_paper_cost_l237_237909

theorem harold_wrapping_paper_cost :
  let rolls_for_shirt_boxes := 20 / 5
  let rolls_for_xl_boxes := 12 / 3
  let total_rolls := rolls_for_shirt_boxes + rolls_for_xl_boxes
  let cost_per_roll := 4  -- dollars
  (total_rolls * cost_per_roll) = 32 := by
  sorry

end harold_wrapping_paper_cost_l237_237909


namespace circumcircle_reflected_lines_tangent_l237_237199

open Geometry

/-- Let ABC be an acute-angled triangle, Γ the circumcircle of the triangle,
and ℓ a tangent line to Γ. Denote by ℓ_a, ℓ_b, ℓ_c the lines obtained
by reflecting ℓ over the lines BC, CA, and AB, respectively. Prove that
the circumcircle of the triangle determined by the lines ℓ_a, ℓ_b, ℓ_c
is tangent to the circle Γ. -/
theorem circumcircle_reflected_lines_tangent
    (ABC : Triangle)
    (h_acute : IsAcuteTriangle ABC)
    (Γ : Circle)
    (h_Γ : Circumcircle ABC Γ)
    (ℓ : Line)
    (h_tangent : IsTangentTo ℓ Γ)
    (ℓ_a ℓ_b ℓ_c : Line)
    (h_ℓa : Reflection ℓ (ABC.side BC) ℓ_a)
    (h_ℓb : Reflection ℓ (ABC.side CA) ℓ_b)
    (h_ℓc : Reflection ℓ (ABC.side AB) ℓ_c) :
  ∃ (Γ' : Circle), Circumcircle (triangle_from_lines ℓ_a ℓ_b ℓ_c) Γ' 
  ∧ IsTangentTo Γ' Γ := 
  sorry

end circumcircle_reflected_lines_tangent_l237_237199


namespace find_side_c_l237_237514

-- Define the given parameters and the cosine function for the specific angle
noncomputable def a := 2
noncomputable def b := 3
noncomputable def C := real.pi / 3  -- 60 degrees in radians

-- Law of Cosines states that:
-- c^2 = a^2 + b^2 - 2ab * cos(C)
-- We need to prove: c = sqrt(7) given the above conditions

theorem find_side_c (a b : ℝ) (C : ℝ) (h_a : a = 2) (h_b : b = 3) (h_C : C = real.pi / 3) : c = real.sqrt 7 :=
by
  sorry

end find_side_c_l237_237514


namespace ratio_of_chris_to_amy_l237_237016

-- Definitions based on the conditions in the problem
def combined_age (Amy_age Jeremy_age Chris_age : ℕ) : Prop :=
  Amy_age + Jeremy_age + Chris_age = 132

def amy_is_one_third_jeremy (Amy_age Jeremy_age : ℕ) : Prop :=
  Amy_age = Jeremy_age / 3

def jeremy_age : ℕ := 66

-- The main theorem we need to prove
theorem ratio_of_chris_to_amy (Amy_age Chris_age : ℕ) (h1 : combined_age Amy_age jeremy_age Chris_age)
  (h2 : amy_is_one_third_jeremy Amy_age jeremy_age) : Chris_age / Amy_age = 2 :=
sorry

end ratio_of_chris_to_amy_l237_237016


namespace smallest_number_l237_237369

theorem smallest_number (N : ℤ) : (∃ (k : ℤ), N = 24 * k + 34) ∧ ∀ n, (∃ (k : ℤ), n = 24 * k + 10) -> n ≥ 34 := sorry

end smallest_number_l237_237369


namespace rakesh_cash_in_hand_l237_237229

def salary : ℝ := 4000
def fixed_deposit_percentage : ℝ := 0.15
def groceries_percentage : ℝ := 0.30

theorem rakesh_cash_in_hand : 
  let fixed_deposit := fixed_deposit_percentage * salary,
      remaining_after_deposit := salary - fixed_deposit,
      groceries_expense := groceries_percentage * remaining_after_deposit,
      cash_in_hand := remaining_after_deposit - groceries_expense
  in cash_in_hand = 2380 := 
by 
  sorry

end rakesh_cash_in_hand_l237_237229


namespace determine_x_l237_237425

theorem determine_x :
  (32^20 + 32^20 + 32^20 = 2^(Real.log 3 / Real.log 2 + 100)) :=
by
  sorry

end determine_x_l237_237425


namespace complex_modulus_proof_l237_237101

noncomputable def complex_modulus_problem (z : ℂ) : Prop :=
  (1 + 2 * complex.I) * z = 1 - complex.I → complex.abs z = real.sqrt 10 / 5

-- Statement to be proved
theorem complex_modulus_proof (z : ℂ) : complex_modulus_problem z :=
by
  -- Condition provided by the problem
  intro h,
  -- Proof placeholder
  sorry

end complex_modulus_proof_l237_237101


namespace average_cost_is_22_cents_l237_237387

-- Total number of pens
def num_pens : ℕ := 150

-- Cost of pens in dollars
def cost_pens : ℝ := 24.75

-- Cost of shipping in dollars
def shipping_cost : ℝ := 8.25

-- Total cost in dollars
def total_cost : ℝ := cost_pens + shipping_cost

-- Conversion of dollars to cents
def total_cost_in_cents : ℕ := (total_cost * 100).to_nat

-- Calculate the average cost per pen in cents
def average_cost_per_pen : ℕ := total_cost_in_cents / num_pens

theorem average_cost_is_22_cents : average_cost_per_pen = 22 := by
  -- Placeholder for the proof
  sorry

end average_cost_is_22_cents_l237_237387


namespace find_k_solution_l237_237841

noncomputable def vec1 : ℝ × ℝ := (3, -4)
noncomputable def vec2 : ℝ × ℝ := (5, 8)
noncomputable def target_norm : ℝ := 3 * Real.sqrt 10

theorem find_k_solution : ∃ k : ℝ, 0 ≤ k ∧ ‖(k * vec1.1 - vec2.1, k * vec1.2 - vec2.2)‖ = target_norm ∧ k = 0.0288 :=
by
  sorry

end find_k_solution_l237_237841


namespace area_of_bounded_curve_is_64_pi_l237_237024

noncomputable def bounded_curve_area : Real :=
  let curve_eq (x y : ℝ) : Prop := (2 * x + 3 * y + 5) ^ 2 + (x + 2 * y - 3) ^ 2 = 64
  let S : Real := 64 * Real.pi
  S

theorem area_of_bounded_curve_is_64_pi : bounded_curve_area = 64 * Real.pi := 
by
  sorry

end area_of_bounded_curve_is_64_pi_l237_237024


namespace bernoulli_inequality_l237_237466

theorem bernoulli_inequality (n : ℕ) (h : 1 ≤ n) (x : ℝ) (h1 : x > -1) : (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l237_237466


namespace find_a_l237_237885

open Real

theorem find_a (a : ℝ) (h : ∀ x y : ℝ, (x - a)^2 + y^2 = 4 → x - y = 2 → (√ ((x - a)^2 + y^2)) = 2√2) : a = 0 ∨ a = 4 :=
sorry

end find_a_l237_237885


namespace solve_a_value_l237_237475

def curve (a : ℝ) (x : ℝ) : ℝ := (a * x ^ 2) / (x + 1)

def tangent_at_point_has_slope (f : ℝ → ℝ) (x₀ : ℝ) (m : ℝ) : Prop :=
  ∃ f' : ℝ → ℝ, (∀ y : ℝ, derivative f y = f' y) ∧ f' x₀ = m

theorem solve_a_value :
  (a : ℝ) (h : tangent_at_point_has_slope (curve a) 1 1) :
  a = 4 / 3 :=
sorry

end solve_a_value_l237_237475


namespace arithmetic_sequence_sum_l237_237192

def arithmetic_sequence (a d : ℕ → ℝ ) (n : ℕ) := a 1 + (n - 1) * d 1

def sum_arithmetic_sequence (a d : ℕ → ℝ ) (n : ℕ) := (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum :
  ∀ (a d n : ℕ), (a 1 + a 6 + a 11 = 18) → sum_arithmetic_sequence a d 11 = 66 :=
by
  sorry

end arithmetic_sequence_sum_l237_237192


namespace range_of_y_C_l237_237878

theorem range_of_y_C (y1 y : ℝ) :
  (∃ (xB : ℝ), xB = y1^2 - 4 ∧ (A : ℝ × ℝ), A = (0, 2) ∧
   (∃ (xC : ℝ), xC = y^2 - 4 ∧
    ∃ (kAB kBC : ℝ), kAB = (y1 - 2) / (y1^2 - 4) ∧
    kBC = - (y1 + 2) ∧
    kBC = (y - y1) / ((y^2 - 4) - (y1^2 - 4)) ∧
    kBC = 1 / (y + y1) ∧
    (y + y1 = - 1 / (y1 + 2))
   )
  ) →
  y ≤ 0 ∨ y ≥ 4 :=
by sorry

end range_of_y_C_l237_237878


namespace distinct_prime_factors_120_l237_237497

theorem distinct_prime_factors_120 : 
  (Set.toFinset {p : ℕ | p.Prime ∧ p ∣ 120}).card = 3 := 
by 
  -- proof omitted 
  sorry

end distinct_prime_factors_120_l237_237497


namespace range_of_a_l237_237896

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x > 1) → f x = x) ∧ (∀ x : ℝ, (x ≤ 1) → f x = a * x^2 + 2 * x) ∧ 
  (∃ R : Set ℝ, ∀ y ∈ R, ∃ x : ℝ, f x = y) → (a ∈ Icc (-1 : ℝ) 0) :=
by
  sorry

end range_of_a_l237_237896


namespace problem_I_problem_II_l237_237906

variables (a b : ℝ^3) -- vectors in 3D space
variables (angle_ab : ℝ) -- angle between vectors a and b
variables (norm_a norm_b : ℝ) -- norms of vectors a and b

-- Given conditions
def conditions : Prop :=
  angle_ab = 120 ∧
  norm_a = 1 ∧
  norm_b = 3
  
-- Problem statements
theorem problem_I (h : conditions a b angle_ab norm_a norm_b) :
  ‖(5:ℝ) • a - b‖ = 7 :=
sorry

theorem problem_II (h : conditions a b angle_ab norm_a norm_b) :
  let c := (5:ℝ) • a - b in
  (c ⬝ a) / (‖c‖ * ‖a‖) = 13 / 14 :=
sorry

end problem_I_problem_II_l237_237906


namespace horner_multiplications_additions_count_l237_237298

noncomputable def polynomial := λ x : ℤ, 5 * x^5 + 4 * x^4 + 3 * x^3 - 2 * x^2 - x - 1

theorem horner_multiplications_additions_count :
  let x := -4 in
  let f := polynomial x in
  (count_multiplications f x, count_additions f x) = (5, 5) :=
sorry

end horner_multiplications_additions_count_l237_237298


namespace greatest_third_side_l237_237697

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l237_237697


namespace value_range_abs_function_l237_237649

theorem value_range_abs_function : 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 9 → 1 ≤ (abs (x - 3) + 1) ∧ (abs (x - 3) + 1) ≤ 7 :=
by
  intro x hx
  sorry

end value_range_abs_function_l237_237649


namespace distance_between_adjacent_parallel_lines_l237_237066

noncomputable def distance_between_lines (r d : ℝ) : ℝ :=
  (49 * r^2 - 49 * 600.25 - (49 / 4) * d^2) / (1 - 49 / 4)

theorem distance_between_adjacent_parallel_lines :
  ∃ d : ℝ, ∀ (r : ℝ), 
    (r^2 = 506.25 + (1 / 4) * d^2 ∧ r^2 = 600.25 + (49 / 4) * d^2) →
    d = 2.8 :=
sorry

end distance_between_adjacent_parallel_lines_l237_237066


namespace initial_investment_proof_l237_237767

-- Definitions for the conditions
def initial_investment_A : ℝ := sorry
def contribution_B : ℝ := 15750
def profit_ratio_A : ℝ := 2
def profit_ratio_B : ℝ := 3
def time_A : ℝ := 12
def time_B : ℝ := 4

-- Lean statement to prove
theorem initial_investment_proof : initial_investment_A * time_A * profit_ratio_B = contribution_B * time_B * profit_ratio_A → initial_investment_A = 1750 :=
by
  sorry

end initial_investment_proof_l237_237767


namespace total_hours_before_midterms_l237_237394

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l237_237394


namespace div_binomial_expansion_l237_237597

theorem div_binomial_expansion
  (a n b : Nat)
  (hb : a^n ∣ b) :
  a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end div_binomial_expansion_l237_237597


namespace magnitude_of_complex_l237_237944

theorem magnitude_of_complex :
  ∀ (z : ℂ), (z = 1 + 2 * complex.i + complex.i ^ 3) → complex.abs z = real.sqrt 2 :=
by
  intros z h
  sorry

end magnitude_of_complex_l237_237944


namespace adults_first_station_l237_237285

-- Conditions as definitions
def children_fewer_than_adults (A C : ℕ) := C = A - 17
def people_got_on (adults_got_on children_got_on : ℕ) := adults_got_on = 57 ∧ children_got_on = 18
def people_got_off (num_people : ℕ) := num_people = 44
def total_people (total : ℕ) := total = 502

-- Main statement to prove
theorem adults_first_station (A C adults_got_on children_got_on total_people num_people : ℕ) 
  (h1 : children_fewer_than_adults A C)
  (h2 : people_got_on adults_got_on children_got_on)
  (h3 : people_got_off num_people)
  (h4 : total_people 502) :
  (let total := (A + C) + (adults_got_on + children_got_on) - num_people in total = 502) →
  A = 244 :=
by
  sorry

end adults_first_station_l237_237285


namespace magnitude_of_z_l237_237941

def i : ℂ := complex.I

theorem magnitude_of_z :
  let z := (1 : ℂ) + 2 * i + i^3 in
  complex.abs z = real.sqrt 2 :=
by
  let z := (1 : ℂ) + 2 * i + i^3
  sorry

end magnitude_of_z_l237_237941


namespace sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l237_237606

theorem sqrt_99_eq_9801 : 99^2 = 9801 := by
  sorry

theorem expr_2000_1999_2001_eq_1 : 2000^2 - 1999 * 2001 = 1 := by
  sorry

end sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l237_237606


namespace missing_keys_total_l237_237992

-- Definitions for the problem conditions

def num_consonants : ℕ := 21
def num_vowels : ℕ := 5
def missing_consonants_fraction : ℚ := 1 / 7
def missing_vowels : ℕ := 2

-- Statement to prove the total number of missing keys

theorem missing_keys_total :
  let missing_consonants := num_consonants * missing_consonants_fraction in
  let total_missing_keys := missing_consonants + missing_vowels in
  total_missing_keys = 5 :=
by {
  -- Placeholder proof
  sorry
}

end missing_keys_total_l237_237992


namespace pyramid_property_l237_237382

-- Define the areas of the faces of the right-angled triangular pyramid.
variables (S_ABC S_ACD S_ADB S_BCD : ℝ)

-- Define the condition that the areas correspond to a right-angled triangular pyramid.
def right_angled_triangular_pyramid (S_ABC S_ACD S_ADB S_BCD : ℝ) : Prop :=
  S_BCD^2 = S_ABC^2 + S_ACD^2 + S_ADB^2

-- State the theorem to be proven.
theorem pyramid_property : right_angled_triangular_pyramid S_ABC S_ACD S_ADB S_BCD :=
sorry

end pyramid_property_l237_237382


namespace range_of_s2_minus_c2_l237_237814

variable (x y z : ℝ)

def r := Real.sqrt (x^2 + y^2 + z^2)
def s := y / r
def c := x / r

theorem range_of_s2_minus_c2 : -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 := by
  sorry

end range_of_s2_minus_c2_l237_237814


namespace phones_left_l237_237372

theorem phones_left (last_year_production : ℕ) 
                    (this_year_production : ℕ) 
                    (sold_phones : ℕ) 
                    (left_phones : ℕ) 
                    (h1 : last_year_production = 5000) 
                    (h2 : this_year_production = 2 * last_year_production) 
                    (h3 : sold_phones = this_year_production / 4) 
                    (h4 : left_phones = this_year_production - sold_phones) : 
                    left_phones = 7500 :=
by
  rw [h1, h2]
  simp only
  rw [h3, h4]
  norm_num
  sorry

end phones_left_l237_237372


namespace trigonometric_identity_l237_237862

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 3) : 1 / (Real.sin x ^ 2 - 2 * Real.cos x ^ 2) = 10 / 7 :=
by
  sorry

end trigonometric_identity_l237_237862


namespace number_of_valid_pairs_l237_237924

theorem number_of_valid_pairs : 
  {n : ℕ // ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = n ∧ ∀ (a b : ℕ), (a, b) ∈ pairs → 
    (a + b ≤ 150) ∧ (1 / (a : ℝ) + b = 17) ∧ 0 < a ∧ 0 < b} = 8 :=
sorry

end number_of_valid_pairs_l237_237924


namespace relationship_among_B_b_β_l237_237505

/-- Definitions for point, line, plane and their relationships -/
variable (Point : Type) [Nonempty Point]
variable (Line : Type) [Nonempty Line]
variable (Plane : Type) [Nonempty Plane]

/-- Relationship predicates -/
variable (on_line : Point → Line → Prop)
variable (in_plane : Line → Plane → Prop)

/-- Variables representing entities -/
variable (B : Point) (b : Line) (β : Plane)

/-- Given relationships -/
axiom B_on_b : on_line B b
axiom b_in_β : in_plane b β

/-- The relationship among B, b, and β -/
theorem relationship_among_B_b_β : B ∈ b ∧ b ⊆ β := sorry

end relationship_among_B_b_β_l237_237505


namespace correct_inequality_l237_237386

theorem correct_inequality :
  (1 / 2)^(2 / 3) < (1 / 2)^(1 / 3) ∧ (1 / 2)^(1 / 3) < 1 :=
by sorry

end correct_inequality_l237_237386


namespace brownies_left_is_zero_l237_237658

-- Definitions of the conditions
def total_brownies : ℝ := 24
def tina_lunch : ℝ := 1.5 * 5
def tina_dinner : ℝ := 0.5 * 5
def tina_total : ℝ := tina_lunch + tina_dinner
def husband_total : ℝ := 0.75 * 5
def guests_total : ℝ := 2.5 * 2
def daughter_total : ℝ := 2 * 3

-- Formulate the proof statement
theorem brownies_left_is_zero :
    total_brownies - (tina_total + husband_total + guests_total + daughter_total) = 0 := by
  sorry

end brownies_left_is_zero_l237_237658


namespace explicit_formula_for_a_seq_l237_237376

noncomputable def a_seq : ℕ → ℝ
| 1       := 1
| (n + 1) := (1 + 4 * a_seq n + (1 + 24 * a_seq n).sqrt) / 16

theorem explicit_formula_for_a_seq (n : ℕ) : 
  a_seq n = (1 / 3) * (1 + 1 / (2 ^ (n - 1))) * (1 + 1 / (2 ^ n)) := sorry

end explicit_formula_for_a_seq_l237_237376


namespace lambda_value_l237_237542

-- Conditions in the form of definitions and Lean assumptions
variable {A B C D : Type}
variable [add_comm_group A]
variable [module ℝ A]
variable [add_comm_group B]
variable [module ℝ B]
variable [add_comm_group C]
variable [module ℝ C]
variable [add_comm_group D]
variable [module ℝ D]

variable (AD DB CA CB CD : A)
variable [is_linear_map ℝ AD]
variable [is_linear_map ℝ DB]
variable [is_linear_map ℝ CA]
variable [is_linear_map ℝ CB]
variable [is_linear_map ℝ CD]

-- Point D lies on side AB and divides it in the ratio 2:1.
axiom AD_eq_2_DB : ∀ {a b : A}, AD = 2 • DB

-- Given vector relationship in the problem.
axiom CD_eq_1_3_CA_plus_lambda_CB : ∀ {a b : A}, CD = (1 / 3 : ℝ) • CA + λ • CB

-- Prove that λ = 2/3 given the conditions
theorem lambda_value : λ = 2 / 3 := by
  sorry

end lambda_value_l237_237542


namespace greatest_possible_third_side_l237_237677

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l237_237677


namespace sum_of_divisors_of_45_l237_237307

theorem sum_of_divisors_of_45 : (∑ d in (Finset.filter (λ x, 45 % x = 0) (Finset.range 46)), d) = 78 :=
by
  -- We'll need to provide the proof here
  sorry

end sum_of_divisors_of_45_l237_237307


namespace michael_eggs_count_l237_237216

def initial_crates : List ℕ := [24, 28, 32, 36, 40, 44]
def wednesday_given : List ℕ := [28, 32, 40]
def thursday_purchases : List ℕ := [50, 45, 55, 60]
def friday_sold : List ℕ := [60, 55]

theorem michael_eggs_count :
  let total_tuesday := initial_crates.sum
  let total_given_wednesday := wednesday_given.sum
  let remaining_wednesday := total_tuesday - total_given_wednesday
  let total_thursday := thursday_purchases.sum
  let total_after_thursday := remaining_wednesday + total_thursday
  let total_sold_friday := friday_sold.sum
  total_after_thursday - total_sold_friday = 199 :=
by
  sorry

end michael_eggs_count_l237_237216


namespace quadratic_roots_are_real_and_equal_l237_237619

theorem quadratic_roots_are_real_and_equal :
  let a := 1
  let b := 2 * Real.sqrt 3
  let c := 3
  let delta := b * b - 4 * a * c
  delta = 0 →
  ∃ x : ℝ, (a * x * x + b * x + c = 0) ∧ (a * x * x + b * x + c = 0) :=
by
  let a := 1
  let b := 2 * Real.sqrt 3
  let c := 3
  let delta := b * b - 4 * a * c
  assume h : delta = 0
  -- proof steps would go here
  let x := -Real.sqrt 3
  use x
  split
  repeat { field_simp, ring, exact h }
  sorry -- placeholder for proving x is a double root

end quadratic_roots_are_real_and_equal_l237_237619


namespace jerry_liters_of_mustard_oil_l237_237551

-- Definitions
def cost_per_liter_mustard_oil : ℕ := 13
def cost_per_pound_penne_pasta : ℕ := 4
def cost_per_pound_pasta_sauce : ℕ := 5
def total_money_jerry_had : ℕ := 50
def money_left_with_jerry : ℕ := 7
def pounds_of_penne_pasta : ℕ := 3
def pounds_of_pasta_sauce : ℕ := 1

-- Our goal is to calculate how many liters of mustard oil Jerry bought
theorem jerry_liters_of_mustard_oil : ℕ :=
  let cost_of_penne_pasta := pounds_of_penne_pasta * cost_per_pound_penne_pasta
  let cost_of_pasta_sauce := pounds_of_pasta_sauce * cost_per_pound_pasta_sauce
  let total_spent := total_money_jerry_had - money_left_with_jerry
  let spent_on_pasta_and_sauce := cost_of_penne_pasta + cost_of_pasta_sauce
  let spent_on_mustard_oil := total_spent - spent_on_pasta_and_sauce
  spent_on_mustard_oil / cost_per_liter_mustard_oil

example : jerry_liters_of_mustard_oil = 2 := by
  unfold jerry_liters_of_mustard_oil
  simp
  sorry

end jerry_liters_of_mustard_oil_l237_237551


namespace sequence_evaluation_l237_237762

noncomputable def a : ℕ → ℤ → ℤ
| 0, x => 1
| 1, x => x^2 + x + 1
| (n + 2), x => (x^n + 1) * a (n + 1) x - a n x 

theorem sequence_evaluation : a 2010 1 = 4021 := by
  sorry

end sequence_evaluation_l237_237762


namespace circle_properties_l237_237476

theorem circle_properties :
  ∀ (x y : ℝ), x^2 + y^2 - 2 * x + 4 * y + 3 = 0 → (∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ a = 1 ∧ b = -2 ∧ r = sqrt 2) :=
by
  intros x y h
  sorry

end circle_properties_l237_237476


namespace printing_completion_time_l237_237006

theorem printing_completion_time :
  ∀ (start_time : ℕ) (quarter_time : ℕ) (completion_hour : ℕ),
  start_time = 9 ∧ quarter_time = 3 ∧ completion_hour = 9 →
  4 * quarter_time + start_time = completion_hour :=
by
  intros start_time quarter_time completion_hour h,
  rcases h with ⟨h1, h2, h3⟩,
  rw [h1, h2, h3],
  sorry

end printing_completion_time_l237_237006


namespace jellybean_probability_l237_237353

-- Define the main problem statement
theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 6
  let blue_jellybeans := 3
  let green_jellybeans := 6
  let picks := 4
  let successful_outcomes := Nat.choose red_jellybeans 2 * Nat.choose green_jellybeans 2
  let total_outcomes := Nat.choose total_jellybeans picks
  successful_outcomes.to_rat / total_outcomes.to_rat = 5 / 9 :=
by
  sorry

end jellybean_probability_l237_237353


namespace probability_X_eq_4_l237_237966

open Nat

def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def P_X_eq_4 : ℚ :=
  (combination 7 4 * combination 8 6 : ℚ) / combination 15 10

theorem probability_X_eq_4 : P_X_eq_4 = 1 / 30 :=
by
  -- Proof is required but omitted
  sorry

end probability_X_eq_4_l237_237966


namespace total_number_of_legs_l237_237209

def kangaroos : ℕ := 23
def goats : ℕ := 3 * kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_number_of_legs : 
  (kangaroos * legs_of_kangaroo + goats * legs_of_goat) = 322 := by
  sorry

end total_number_of_legs_l237_237209


namespace minimal_withdrawals_proof_l237_237457

-- Defining the conditions
def red_marbles : ℕ := 200
def blue_marbles : ℕ := 300
def green_marbles : ℕ := 400

def max_red_withdrawal_per_time : ℕ := 1
def max_blue_withdrawal_per_time : ℕ := 2
def max_total_withdrawal_per_time : ℕ := 5

-- The target minimal number of withdrawals
def minimal_withdrawals : ℕ := 200

-- Lean statement of the proof problem
theorem minimal_withdrawals_proof :
  ∃ (w : ℕ), w = minimal_withdrawals ∧ 
    (∀ n, n ≤ w →
      (n = 200 ∧ 
       (∀ r b g, r ≤ max_red_withdrawal_per_time ∧ b ≤ max_blue_withdrawal_per_time ∧ (r + b + g) ≤ max_total_withdrawal_per_time))) :=
sorry

end minimal_withdrawals_proof_l237_237457


namespace ED_perpendicular_BL_l237_237165

variables {A B C M L D E : Type} [Field A] [OrderedField B] [Geometry C]

-- Given conditions:
-- Triangle ABC with AB > BC
def triangle_ABC (A B C: C) : Prop := 
  ∃ (AB BC: B), 
    AB > BC ∧ ∃ (BM BL: Line A), 
       (BM: Line A) ∧ (BL: Line A) ∧ 
       -- BM is a median and BL is an angle bisector
       median BM A B C ∧ angle_bisector BL A B C ∧
       -- Line through M parallel to AB intersects BL at D
       (∃ (M D: Point A), parallel (line_through M A B) (line_through A B) ∧ intersect_at (line_through M A B) BL D) ∧
       -- Line through L parallel to BC intersects BM at E
       (∃ (L E: Point A), parallel (line_through L B C) (line_through B C) ∧ intersect_at (line_through L B C) BM E)

-- Prove ED is perpendicular to BL
theorem ED_perpendicular_BL {A B C M L D E : C} (h : triangle_ABC A B C M L D E) : 
  perpendicular (line_through E D) (line_through B L) :=
sorry

end ED_perpendicular_BL_l237_237165


namespace calc_minimum_travel_time_l237_237301

noncomputable theory

open Real

-- Define the given conditions
def v1 : ℝ := 30 -- Speed on the road in km/h
def v2 : ℝ := 15 -- Speed on the field in km/h
def a : ℝ := 12  -- Distance along the road in km
def b : ℝ := 3   -- Distance perpendicular from the road to point B in km

-- Define a function for the total travel time t(x)
def total_travel_time (x : ℝ) : ℝ := 
  ((a - x) / v1) + (sqrt (b^2 + x^2) / v2)

-- The main theorem statement: the optimal x value that minimizes the travel time
theorem calc_minimum_travel_time : ∃ x : ℝ, x = sqrt 3 ∧ ∀ y : ℝ, (total_travel_time y) ≥ (total_travel_time x) :=
sorry

end calc_minimum_travel_time_l237_237301


namespace percentage_non_indian_attendees_l237_237965

def total_non_indian_attendees_percentage
  (total_attendees : ℕ)
  (non_indian_men : ℝ)
  (non_indian_women : ℝ)
  (non_indian_children : ℝ)
  (total_non_indian : ℝ) : ℝ :=
  (total_non_indian / total_attendees.to_real) * 100

theorem percentage_non_indian_attendees
  (men : ℕ) (men_participants : ℕ) (men_volunteers : ℕ) (men_participants_indian_pct : ℝ) (men_volunteers_indian_pct : ℝ)
  (women : ℕ) (women_participants : ℕ) (women_volunteers : ℕ) (women_participants_indian_pct : ℝ) (women_volunteers_indian_pct : ℝ)
  (children : ℕ) (children_indian_pct : ℝ)
  (scientists : ℕ) (male_scientists : ℕ) (female_scientists : ℕ) (male_scientists_indian_pct : ℝ) (female_scientists_indian_pct : ℝ)
  (gov_officials : ℕ) (male_gov_officials : ℕ) (female_gov_officials : ℕ) (male_gov_officials_indian_pct : ℝ) (female_gov_officials_indian_pct : ℝ)
  (total_attendees := men + women + children + scientists + gov_officials)
  (non_indian_men := (men_participants * (1 - men_participants_indian_pct)) + (men_volunteers * (1 - men_volunteers_indian_pct)) + (male_scientists * (1 - male_scientists_indian_pct)) + (male_gov_officials * (1 - male_gov_officials_indian_pct)))
  (non_indian_women := (women_participants * (1 - women_participants_indian_pct)) + (women_volunteers * (1 - women_volunteers_indian_pct)) + (female_scientists * (1 - female_scientists_indian_pct)) + (female_gov_officials * (1 - female_gov_officials_indian_pct)))
  (non_indian_children := children * (1 - children_indian_pct))
  (total_non_indian := non_indian_men + non_indian_women + non_indian_children) :
  total_non_indian_attendees_percentage total_attendees non_indian_men non_indian_women non_indian_children total_non_indian = 72.61 :=
by 
  sorry

end percentage_non_indian_attendees_l237_237965


namespace number_of_8_digit_integers_l237_237121

theorem number_of_8_digit_integers : 
  ∃ n, n = 90000000 ∧ 
    (∀ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
     d1 ≠ 0 → 0 ≤ d1 ∧ d1 ≤ 9 ∧ 
     0 ≤ d2 ∧ d2 ≤ 9 ∧ 
     0 ≤ d3 ∧ d3 ≤ 9 ∧ 
     0 ≤ d4 ∧ d4 ≤ 9 ∧ 
     0 ≤ d5 ∧ d5 ≤ 9 ∧ 
     0 ≤ d6 ∧ d6 ≤ 9 ∧ 
     0 ≤ d7 ∧ d7 ≤ 9 ∧ 
     0 ≤ d8 ∧ d8 ≤ 9 →
     ∀ count, count = (if d1 ≠ 0 then 9 * 10^7 else 0)) :=
sorry

end number_of_8_digit_integers_l237_237121


namespace bag_numbers_correct_l237_237400

-- Define the conditions
def total_bags : ℕ := 500
def valid_bag_number (n : ℕ) : Prop := n < total_bags
def starting_position : ℕ × ℕ := (8, 4)

-- Random number table excerpt given in the problem (from rows 7 to 9)
def random_table : list (list ℕ) :=
  [ [84, 42, 17, 53, 31, 57, 24, 55, 6, 88, 77, 4, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76],
    [63, 1, 64, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 85, 12, 86, 73, 58, 7, 44, 39, 52, 38, 79],
    [33, 21, 12, 34, 29, 78, 64, 56, 7, 82, 52, 42, 7, 44, 38, 15, 51, 0, 13, 42, 99, 66, 2, 79, 54] ]

-- Function to read numbers from the given starting position and select valid numbers
noncomputable def first_5_valid_bag_numbers : list ℕ :=
let nums := random_table.join.drop ((starting_position.1 - 7) * 25 + (starting_position.2 - 1)) in
nums.filter valid_bag_number |>.take 5

-- Prove that the first 5 valid bag numbers are as specified
theorem bag_numbers_correct :
  first_5_valid_bag_numbers = [164, 199, 185, 128, 395] :=
sorry

end bag_numbers_correct_l237_237400


namespace distance_between_A_and_B_l237_237156

open Real

def C1_param_eq (θ : ℝ) : ℝ × ℝ := (1 + cos θ, sin θ)
def C2_polar_eq (θ : ℝ) : ℝ := sin θ + cos θ
def C3_polar_eq (θ : ℝ) : Prop := θ = π / 6

noncomputable def polar_coordinates_of_A : ℝ := 2 * cos (π / 6)
noncomputable def polar_coordinates_of_B : ℝ := sin (π / 6) + cos (π / 6)

theorem distance_between_A_and_B :
  abs (polar_coordinates_of_A - polar_coordinates_of_B) = (sqrt 3 - 1) / 2 :=
sorry

end distance_between_A_and_B_l237_237156


namespace mean_of_remaining_students_l237_237142

theorem mean_of_remaining_students (k : ℕ) (h_k : k > 15) 
  (avg_all : ∀ (scores : List ℝ), (∑ x in scores, x) / k = 10)
  (avg_15 : ∀ (scores_15 : List ℝ), scores_15.length = 15 → 
    (∑ x in scores_15, x) / 15 = 12) : 
  ∀ (scores_remaining : List ℝ), scores_remaining.length = (k - 15) →
  (∑ x in scores_remaining, x) / (k - 15) = (10 * k - 180) / (k - 15) := by
  sorry

end mean_of_remaining_students_l237_237142


namespace intersection_M_N_l237_237191

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := 
sorry

end intersection_M_N_l237_237191


namespace find_exp_l237_237453

noncomputable def a : ℝ := sorry
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom a_m_eq_six : a ^ m = 6
axiom a_n_eq_six : a ^ n = 6

theorem find_exp : a ^ (2 * m - n) = 6 :=
by
  sorry

end find_exp_l237_237453


namespace remainder_sum_mod_11_l237_237804

theorem remainder_sum_mod_11 :
  (72501 + 72502 + 72503 + 72504 + 72505 + 72506 + 72507 + 72508 + 72509 + 72510) % 11 = 5 :=
by
  sorry

end remainder_sum_mod_11_l237_237804


namespace problem_solution_l237_237227

-- Define a function to sum the digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  d1 + d2 + d3 + d4

-- Define the problem numbers.
def nums : List ℕ := [4272, 4281, 4290, 4311, 4320]

-- Check if the sum of digits is divisible by 9.
def divisible_by_9 (n : ℕ) : Prop :=
  sum_digits n % 9 = 0

-- Main theorem asserting the result.
theorem problem_solution :
  ∃ n ∈ nums, ¬divisible_by_9 n ∧ (n % 100 / 10) * (n % 10) = 14 := by
  sorry

end problem_solution_l237_237227


namespace closest_fraction_to_one_l237_237329

theorem closest_fraction_to_one : 
  let diff (x : ℚ) := abs (1 - x) in
  ∀ (A B C D E : ℚ), 
    A = 7 / 8 ∧ B = 8 / 7 ∧ C = 9 / 10 ∧ D = 10 / 11 ∧ E = 11 / 10 →
    diff D < diff A ∧ diff D < diff B ∧ diff D < diff C ∧ diff D < diff E :=
by
  intros diff A B C D E h
  have hA : A = 7 / 8 := h.left
  have hB : B = 8 / 7 := h.right.left
  have hC : C = 9 / 10 := h.right.right.left
  have hD : D = 10 / 11 := h.right.right.right.left
  have hE : E = 11 / 10 := h.right.right.right.right
  sorry

end closest_fraction_to_one_l237_237329


namespace determine_meaning_of_bal_l237_237513

/-- A native will respond either with "bal" or "da", where "da" means "yes".
Given these responses, we aim to prove that it is possible to determine the meaning of "bal" by
asking the native a single question. -/
theorem determine_meaning_of_bal (response : String) : 
  response = "bal" → determine_meaning_of_bal response :=
sorry

end determine_meaning_of_bal_l237_237513


namespace posititve_integers_not_divisible_by_5_or_7_l237_237264

theorem posititve_integers_not_divisible_by_5_or_7 :
  let S := {n : ℕ | n < 1000 ∧ (n % 5 ≠ 0 ∧ n % 7 ≠ 0)} in 
  S.card = 686 :=
by
  sorry

end posititve_integers_not_divisible_by_5_or_7_l237_237264


namespace regular_pentagon_condition_l237_237968

section Pentagon

variables {A B C D E : Type}

-- Define what it means for a pentagon to be regular
def is_regular_pentagon (angles : List ℝ) : Prop :=
  angles = [72, 72, 72, 72, 72, 72, 72]

-- Define the seven angles that need to be noted
def noted_angles (α : ℝ) : List ℝ :=
  [α, α, α, α, α, α, α]

-- Main theorem statement
theorem regular_pentagon_condition (α : ℝ) (h : noted_angles α = [72, 72, 72, 72, 72, 72, 72]) :
  is_regular_pentagon (noted_angles α) :=
by
  rw [←h]
  exact is_regular_pentagon _

end Pentagon

end regular_pentagon_condition_l237_237968


namespace meet_time_at_10_30_l237_237384

noncomputable def alicia_distance (t : ℝ) : ℝ := 15 * t
noncomputable def david_distance (t : ℝ) : ℝ := 18 * (t - 0.5)

theorem meet_time_at_10_30 :
  Exists (λ t : ℝ, alicia_distance t + david_distance t = 84) ∧ (7 * 60 + 45) + ((93 / 33) * 60) = 10 * 60 + 30 :=
by
  sorry

end meet_time_at_10_30_l237_237384


namespace length_OC_l237_237538

open Real

theorem length_OC (A B : EuclideanSpace ℝ fin 3)
                 (O : EuclideanSpace ℝ fin 3)
                 (C : EuclideanSpace ℝ fin 3) :
                 A = ![1, 2, 3] → B = ![1, 0, 1] → O = ![0, 0, 0] →
                 C = midpoint ℝ (ofEuclidean ℝ A) (ofEuclidean ℝ B) →
                 dist O C = sqrt 6 :=
by
  intros hA hB hO hC
  rw [hA, hB, hO, hC]
  simp [dist, sqrt]
  sorry

end length_OC_l237_237538


namespace number_of_cheaper_values_l237_237493

def C (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 60 then 13 * n
  else if 61 ≤ n then 12 * n
  else 0

theorem number_of_cheaper_values : 
  finset.card ({ n ∈ finset.range 61 | C n > C (n + 1) }) = 5 :=
by
  sorry

end number_of_cheaper_values_l237_237493


namespace sum_of_divisors_of_45_l237_237311

theorem sum_of_divisors_of_45 : 
  (∑ d in (finset.filter (λ x : ℕ, 45 % x = 0) (finset.range (45 + 1))), d) = 78 :=
by sorry

end sum_of_divisors_of_45_l237_237311


namespace f_lg_equality_l237_237479

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_lg_equality : f (Real.log 2) + f (Real.log (1 / 2)) = 2 := sorry

end f_lg_equality_l237_237479


namespace greatest_integer_third_side_l237_237668

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l237_237668


namespace problem_B_problem_D_l237_237193

/-
  Given distinct lines m, n and distinct planes α, β,
  we want to prove the following two statements:
  
  1. If m is perpendicular to α and n is perpendicular to α, then m is parallel to n.
  2. If m is perpendicular to α, n is perpendicular to β, and m is perpendicular to n, then α is perpendicular to β.
-/

variables {m n : Type} -- Types representing distinct lines
variables {α β : Type} -- Types representing distinct planes

-- Hypotheses for the statements
variable [linear_order m n α β] -- Assume we have a linear ordering for the geometric entities

-- Define helper functions for parallelism and perpendicularity
def is_parallel (x y : Type) : Prop := sorry
def is_perpendicular (x y : Type) : Prop := sorry

-- Statement for problem B
theorem problem_B (h1 : is_perpendicular m α) (h2 : is_perpendicular n α) : is_parallel m n :=
  sorry

-- Statement for problem D
theorem problem_D (h1 : is_perpendicular m α) (h2 : is_perpendicular m n) (h3 : is_perpendicular n β) : is_perpendicular α β :=
  sorry

end problem_B_problem_D_l237_237193


namespace probability_of_draw_l237_237592

noncomputable def P_aw : ℝ := 0.4
noncomputable def P_anl : ℝ := 0.9
noncomputable def P_draw : ℝ := P_anl - P_aw

theorem probability_of_draw :
  P_draw = 0.5 :=
by
  rw [P_draw, P_anl, P_aw]
  norm_num
  sorry

end probability_of_draw_l237_237592


namespace find_eccentricity_find_range_of_k_l237_237367

-- Definitions and conditions for Part 1
variables {a b x₀ y₀ x₁ y₁ : Real}
axiom ellipse_eq: (a > b) ∧ (b > 0) ∧ ((x₀^2 / a^2) + (y₀^2 / b^2) = 1)
axiom line_slope_eq: ((y₀ - y₁) / (x₀ - x₁)) * ((y₀ + y₁) / (x₀ + x₁)) = -1 / 4

-- Proof goal for Part 1
theorem find_eccentricity : sqrt(1 - (b^2 / a^2)) = sqrt(3) / 2 :=
by
  sorry

-- Definitions and conditions for Part 2
variables {k : Real}
axiom ellipse_mod_eq: ((x^2) / (4 * b^2)) + ((y^2) / b^2) = 1
axiom line_eq : y = k * (x - sqrt(3) * b)
axiom eccentricity_eq : sqrt(1 - (b^2 / (2 * b)^2)) = sqrt(3) / 2

-- Proof goal for Part 2
theorem find_range_of_k : abs(k) < sqrt(47) / 47 :=
by
  sorry

end find_eccentricity_find_range_of_k_l237_237367


namespace sum_of_consecutive_integers_and_cubic_equality_l237_237580

theorem sum_of_consecutive_integers_and_cubic_equality :
  let a := 19^2 + 1,
      l := 20^2,
      sum_left := (a + l) * (l - a + 1) / 2,
      sum_right := 19^3 + 20^3
  in sum_left = sum_right :=
by
  sorry

end sum_of_consecutive_integers_and_cubic_equality_l237_237580


namespace magnitude_of_z_l237_237943

def i : ℂ := complex.I

theorem magnitude_of_z :
  let z := (1 : ℂ) + 2 * i + i^3 in
  complex.abs z = real.sqrt 2 :=
by
  let z := (1 : ℂ) + 2 * i + i^3
  sorry

end magnitude_of_z_l237_237943


namespace johns_salary_percentage_increase_l237_237177

theorem johns_salary_percentage_increase (initial_salary final_salary : ℕ) (h1 : initial_salary = 50) (h2 : final_salary = 90) :
  ((final_salary - initial_salary : ℕ) / initial_salary : ℚ) * 100 = 80 := by
  sorry

end johns_salary_percentage_increase_l237_237177


namespace range_of_b_equation_of_circle_l237_237155

section
variable (b : ℝ)

def quadratic_function := λ x : ℝ, x^2 + 2*x + b
def discriminant := λ b : ℝ, 4 - 4*b

-- Condition: b ≠ 0 and b < 1
axiom b_nonzero : b ≠ 0
axiom b_range : b < 1

-- Statement 1: The range of values for b
theorem range_of_b : b < 1 ∧ b ≠ 0 :=
by
  -- Proof is omitted
  sorry

-- Statement 2: Equation of the circle C
theorem equation_of_circle : x^2 + y^2 + 2*x - (b + 1)*y + b = 0 :=
by
  -- Proof is omitted
  sorry
end

end range_of_b_equation_of_circle_l237_237155


namespace range_of_a_l237_237898

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ (m n p : ℝ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ f m a = 2024 ∧ f n a = 2024 ∧ f p a = 2024) ↔
  2022 < a ∧ a < 2026 :=
sorry

end range_of_a_l237_237898


namespace spherical_to_rectangular_coordinates_l237_237030

theorem spherical_to_rectangular_coordinates : 
  ∀ (rho theta phi : ℝ), 
    (rho = 6) → 
    (theta = 7 * Real.pi / 4) → 
    (phi = Real.pi / 3) → 
    let x := rho * Real.sin phi * Real.cos theta,
        y := rho * Real.sin phi * Real.sin theta,
        z := rho * Real.cos phi in
      (x = 3 * Real.sqrt 6) ∧ 
      (y = -3 * Real.sqrt 6) ∧ 
      (z = 3) := 
by
  intros rho theta phi h_rho h_theta h_phi
  let x := rho * Real.sin phi * Real.cos theta
  let y := rho * Real.sin phi * Real.sin theta
  let z := rho * Real.cos phi
  sorry

end spherical_to_rectangular_coordinates_l237_237030


namespace harold_wrapping_cost_l237_237910

noncomputable def wrapping_paper_rolls_cost (cost_per_roll : ℕ) (rolls_needed : ℕ) : ℕ := cost_per_roll * rolls_needed

noncomputable def total_paper_rolls (shirt_boxes : ℕ) (shirt_boxes_per_roll : ℕ) (xl_boxes : ℕ) (xl_boxes_per_roll : ℕ) : ℕ :=
  (shirt_boxes / shirt_boxes_per_roll) + (xl_boxes / xl_boxes_per_roll)

theorem harold_wrapping_cost 
  (cost_per_roll : ℕ) (shirt_boxes : ℕ) (shirt_boxes_per_roll : ℕ) (xl_boxes : ℕ) (xl_boxes_per_roll : ℕ) :
  shirt_boxes = 20 → shirt_boxes_per_roll = 5 → xl_boxes = 12 → xl_boxes_per_roll = 3 → cost_per_roll = 4 → 
  wrapping_paper_rolls_cost cost_per_roll (total_paper_rolls shirt_boxes shirt_boxes_per_roll xl_boxes xl_boxes_per_roll) = 32 :=
by
  intros hshirt_boxes hshirt_boxes_per_roll hxl_boxes hxl_boxes_per_roll hcost_per_roll
  simp [wrapping_paper_rolls_cost, total_paper_rolls, hshirt_boxes, hshirt_boxes_per_roll, hxl_boxes, hxl_boxes_per_roll, hcost_per_roll]
  rfl

end harold_wrapping_cost_l237_237910


namespace edge_length_of_divided_cube_l237_237277

theorem edge_length_of_divided_cube (volume_original_cube : ℕ) (num_divisions : ℕ) (volume_of_one_smaller_cube : ℕ) (edge_length : ℕ) :
  volume_original_cube = 1000 →
  num_divisions = 8 →
  volume_of_one_smaller_cube = volume_original_cube / num_divisions →
  volume_of_one_smaller_cube = edge_length ^ 3 →
  edge_length = 5 :=
by
  sorry

end edge_length_of_divided_cube_l237_237277


namespace total_hours_before_midterms_l237_237393

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l237_237393


namespace harold_wrapping_paper_cost_l237_237908

theorem harold_wrapping_paper_cost :
  let rolls_for_shirt_boxes := 20 / 5
  let rolls_for_xl_boxes := 12 / 3
  let total_rolls := rolls_for_shirt_boxes + rolls_for_xl_boxes
  let cost_per_roll := 4  -- dollars
  (total_rolls * cost_per_roll) = 32 := by
  sorry

end harold_wrapping_paper_cost_l237_237908


namespace locus_of_points_l237_237083

theorem locus_of_points (s : ℝ) (x y : ℝ) :
  (x, y ∈ set_of (
    λ (P : ℝ × ℝ), 
      let PA := (P.1 - 0)^2 + (P.2 - 0)^2 in
      let PB := (P.1 - 0)^2 + (P.2 - s)^2 in
      let PC := (P.1 - s)^2 + (P.2 - 0)^2 in
      let PD := (P.1 - s)^2 + (P.2 - s)^2 in
        PA + PB + PC + PD = 4 * s^2
    )) ↔ ((x - s/2)^2 + (y - s/2)^2 = (s^2 / 2)) :=
by sorry

end locus_of_points_l237_237083


namespace triangle_B_eq_2A_range_of_a_l237_237876

theorem triangle_B_eq_2A (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = c) : B = 2 * A := 
sorry

theorem range_of_a (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = 2) (h6 : 0 < (π - A - B)) (h7 : (π - A - B) < π/2) : 1 < a ∧ a < 2 := 
sorry

end triangle_B_eq_2A_range_of_a_l237_237876


namespace tangency_point_ratios_equal_l237_237639

variables (A B C D O O' M N P Q : Type)
variables [InCircle A B C D O]
variables [Circumscribe A B C D O' M N P Q]

theorem tangency_point_ratios_equal :
  ∀ (A B C D M N : Type), 
  InCircle A B C D O → 
  Circumscribe A B C D O' M N P Q → 
  (AM / MB = DN / NC) :=
by 
  -- Add the necessary conditions and proofs step-by-step
  sorry

end tangency_point_ratios_equal_l237_237639


namespace sum_f_1_to_2015_l237_237032

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -(x + 2)^2 else
  if -1 ≤ x ∧ x < 3 then x else
  f (x - 6)

theorem sum_f_1_to_2015 :
  (∑ n in Finset.range 2015, f (n + 1)) = 336 :=
sorry

end sum_f_1_to_2015_l237_237032


namespace total_members_in_club_l237_237150

-- Define the conditions as hypotheses
variables (B T BT N : ℕ)
variables (hB : B = 16) (hT : T = 19) (hBT : BT = 7) (hN : N = 2)

-- Lean statement that encapsulates the proof problem
theorem total_members_in_club : (B + T) - BT + N = 30 :=
by
  -- assumptions
  rw [hB, hT, hBT, hN]
  -- computation
  sorry

end total_members_in_club_l237_237150


namespace april_plant_arrangement_l237_237021

theorem april_plant_arrangement :
    let nBasil := 5
    let nTomato := 4
    let nPairs := nTomato / 2
    let nUnits := nBasil + nPairs
    let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
    totalWays = 20160 := by
{
  let nBasil := 5
  let nTomato := 4
  let nPairs := nTomato / 2
  let nUnits := nBasil + nPairs
  let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
  sorry
}

end april_plant_arrangement_l237_237021


namespace cos_double_plus_cos_l237_237467

theorem cos_double_plus_cos (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 3) :
  Real.cos (2 * α) + Real.cos α = -4 / 9 :=
by
  sorry

end cos_double_plus_cos_l237_237467


namespace sacred_k_words_n10_k4_l237_237786

/- Definitions for the problem -/
def sacred_k_words_count (n k : ℕ) (hk : k < n / 2) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * (Nat.factorial k / k)

theorem sacred_k_words_n10_k4 : sacred_k_words_count 10 4 (by norm_num : 4 < 10 / 2) = 600 := by
  sorry

end sacred_k_words_n10_k4_l237_237786


namespace max_statements_true_l237_237567

theorem max_statements_true : ∃ x : ℝ, 
  (0 < x^2 ∧ x^2 < 1 ∨ x^2 > 1) ∧ 
  (-1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1) ∧ 
  (0 < (x - x^3) ∧ (x - x^3) < 1) :=
  sorry

end max_statements_true_l237_237567


namespace contestant_A_wins_by_100_meters_l237_237339

-- Definitions of the conditions
def race_length : ℕ := 500
def speed_ratio_A_to_B : ℚ := 3 / 4
def head_start_A : ℕ := 200

-- The proof statement
theorem contestant_A_wins_by_100_meters :
  let v_A := 1 in -- Normalize v_A to 1 for simplicity
  let v_B := (4 / 3) * v_A in
  let t_A := (race_length - head_start_A) / v_A in
  let t_B := (race_length : ℚ) / v_B in
  let Δt := t_B - t_A in
  let d_B := v_B * Δt in
  d_B = 100 := sorry

end contestant_A_wins_by_100_meters_l237_237339


namespace sum_is_correct_l237_237151

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a n = a 1 + (n - 1) * (a 2 - a 1)

variable (a : ℕ → ℝ)
variable h_arith : arithmetic_seq a
variable h_cond : ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n-1) - a (n+1) = 0

theorem sum_is_correct :
  ∑ i in Finset.range 2009, a i = 4018 :=
sorry

end sum_is_correct_l237_237151


namespace part_1_l237_237485

noncomputable def f (x : ℝ) : ℝ := Real.log x
namespace question

variables {a b : ℝ} (h1 : b > a) (h2 : a > 1) (h3 : b > 1)

theorem part_1 : 
    let m := 1
    let n := 0
    let A := f((a + b) / 2)
    let B := (f(a) + f(b)) / 2 
    let C := (b * f(b) - a * f(a)) / (b - a) - 1
    A > C ∧ C > B := sorry

end question

end part_1_l237_237485


namespace decreasing_on_interval_problem_statement_l237_237720

noncomputable def f1 (x : ℝ) : ℝ := -abs (x - 1)
noncomputable def f2 (x : ℝ) : ℝ := x^2 - 2 * x + 4
noncomputable def f3 (x : ℝ) : ℝ := Real.log (x + 2)
noncomputable def f4 (x : ℝ) : ℝ := (1 / 2) ^ x

theorem decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y < f x

theorem problem_statement : decreasing_on_interval f4 0 ∞ :=
  sorry

end decreasing_on_interval_problem_statement_l237_237720


namespace sum_of_divisors_of_45_l237_237318

theorem sum_of_divisors_of_45 :
  let divisors := [1, 3, 9, 5, 15, 45] in
  list.sum divisors = 78 :=
by
  -- this is where the proof would go
  sorry


end sum_of_divisors_of_45_l237_237318


namespace num_whole_numbers_between_cubic_roots_l237_237925

theorem num_whole_numbers_between_cubic_roots :
  let a := real.cbrt 15
  let b := real.cbrt 500
  2 < a ∧ a < 3 ∧ 7 < b ∧ b < 8 →
  ∃ (n : ℕ), (3 ≤ n ∧ n ≤ 7) ∧ n = 5 := 
by
  intros h
  have h_interval : 2 < real.cbrt 15 ∧ real.cbrt 15 < 3 ∧ 7 < real.cbrt 500 ∧ real.cbrt 500 < 8 := h
  sorry

end num_whole_numbers_between_cubic_roots_l237_237925


namespace math_problem_l237_237566

noncomputable def f : ℝ → ℝ := sorry

theorem math_problem :
  (∀ x y z : ℝ, f (x^2 + y * f(z)) = x^2 * f(x) + z^2 * f(y)) →
  let vals := {f2 | f2 = 0 ∨ f2 = 2} in
  let n := vals.to_finset.card in
  let s := vals.sum id in
  n * s = 4 :=
begin
  intros h,
  sorry
end

end math_problem_l237_237566


namespace part_I_part_II_l237_237983

noncomputable def sequence (a : ℝ) : (ℕ → ℝ)
| 0     := a
| (n+1) := 2 * sequence n ^ 2 / (4 * sequence n - 1)

noncomputable def S_n (a : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, sequence a i

theorem part_I (a : ℝ) :
  (∀ n : ℕ, n > 0 → sequence a n > 1/2) ↔ (a > 1/4 ∧ a ≠ 1/2) := sorry

theorem part_II (n : ℕ) (hn : n > 0) :
  S_n 1 n < n^2/4 + 1 := sorry

end part_I_part_II_l237_237983


namespace tower_remainder_l237_237360

noncomputable def num_towers : ℕ → ℕ 
| 1     := 1
| 2     := 2
| (m+1) := 4 * num_towers m

def towers_mod (n : ℕ) : ℕ :=
  num_towers n % 1000

theorem tower_remainder :
  towers_mod 10 = 72 :=
by
sorry

end tower_remainder_l237_237360


namespace proof_f_f_neg1_l237_237895

def f (x : ℝ) : ℝ := if x ≤ 0 then 1 - 2^x else x^(1/2)

theorem proof_f_f_neg1 :
  f (f (-1)) = (sqrt 2) / 2 :=
by
  sorry

end proof_f_f_neg1_l237_237895


namespace zengshan_suanfa_tongzong_l237_237020

-- Definitions
variables (x y : ℝ)
variables (h1 : x = y + 5) (h2 : (1 / 2) * x = y - 5)

-- Theorem
theorem zengshan_suanfa_tongzong :
  x = y + 5 ∧ (1 / 2) * x = y - 5 :=
by
  -- Starting with the given hypotheses
  exact ⟨h1, h2⟩

end zengshan_suanfa_tongzong_l237_237020


namespace cos_trig_identity_solution_l237_237336

theorem cos_trig_identity_solution (x : ℝ) :
  (cos (10 * x) + 2 * (cos (4 * x))^2 + 6 * (cos (3 * x)) * (cos x) = cos x + 8 * (cos x) * (cos (3 * x))^3) →
  ∃ k : ℤ, x = 2 * π * k :=
by
  sorry

end cos_trig_identity_solution_l237_237336


namespace minimum_point_transformed_graph_l237_237625

def original_graph (x : ℝ) : ℝ :=
  |x| - 3

def transformed_graph (x : ℝ) : ℝ :=
  original_graph (x + 2) - 3

theorem minimum_point_transformed_graph :
  ∃ (x y : ℝ), transformed_graph x = y ∧ y ≤ transformed_graph z ∀ z : ℝ ∧ (x, y) = (-2, -6) := sorry

end minimum_point_transformed_graph_l237_237625


namespace student_B_incorrect_l237_237448

-- Define the quadratic function and the non-zero condition on 'a'
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x - 6

-- Conditions stated by the students
def student_A_condition (a b : ℝ) : Prop := -b / (2 * a) = 1
def student_B_condition (a b : ℝ) : Prop := quadratic a b 3 = -6
def student_C_condition (a b : ℝ) : Prop := (4 * a * (-6) - b^2) / (4 * a) = -8
def student_D_condition (a b : ℝ) : Prop := quadratic a b 3 = 0

-- The proof problem: Student B's conclusion is incorrect
theorem student_B_incorrect : 
  ∀ (a b : ℝ), 
  a ≠ 0 → 
  student_A_condition a b ∧ 
  student_C_condition a b ∧ 
  student_D_condition a b → 
  ¬ student_B_condition a b :=
by 
  -- problem converted to Lean problem format 
  -- based on the conditions provided
  sorry

end student_B_incorrect_l237_237448


namespace tan_theta_eq_2_l237_237883

theorem tan_theta_eq_2 {θ : ℝ} (h : Real.tan θ = 2) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (Real.cos θ ^ 2 + Real.sin θ * Real.cos θ) = 8 / 3 :=
by
  sorry

end tan_theta_eq_2_l237_237883


namespace third_side_triangle_max_l237_237689

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l237_237689


namespace find_m_l237_237135

theorem find_m (m : ℚ) (h : (9 : ℚ) * x^2 + 5 * x + m = 0) 
  (roots : ∀ x, x = (-5 + complex.I * real.sqrt 391) / 18 ∨ x = (-5 - complex.I * real.sqrt 391) / 18) : 
  m = 104 / 9 :=
by
  sorry

end find_m_l237_237135


namespace ratio_of_speeds_l237_237549

-- Definitions for conditions:
def distance : ℝ := 41
def time_jack : ℝ := 4.5
def time_jill : ℝ := 4.1

-- Definitions for speeds:
def speed_jack : ℝ := distance / time_jack
def speed_jill : ℝ := distance / time_jill

-- Definition for the ratio of speeds:
def speed_ratio : ℝ * ℝ := speed_jack / speed_jill

-- Theorem stating the ratio of speeds equals the computed ratio 82:90
theorem ratio_of_speeds : 82 / 90 = speed_jack / speed_jill :=
by
  -- Proof steps here
  -- This is just a placeholder
  sorry

end ratio_of_speeds_l237_237549


namespace total_area_of_removed_triangles_l237_237766

theorem total_area_of_removed_triangles (side_length : ℝ) (half_leg_length : ℝ) :
  side_length = 16 →
  half_leg_length = side_length / 4 →
  4 * (1 / 2) * half_leg_length^2 = 32 :=
by
  intro h_side_length h_half_leg_length
  simp [h_side_length, h_half_leg_length]
  sorry

end total_area_of_removed_triangles_l237_237766
