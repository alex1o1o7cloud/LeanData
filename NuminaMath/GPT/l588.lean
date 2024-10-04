import Mathlib

namespace area_swept_by_curve_l588_588157

noncomputable def polar_eq_curve_C (θ : ℝ) : ℝ := 1 + Real.cos θ
def point_A : ℝ × ℝ := (2, 0)

theorem area_swept_by_curve (θ : ℝ) :
  ∫ x in 0..2*Real.pi, 1/2 * (polar_eq_curve_C x - 2)^2 = Real.pi :=
by
  sorry

end area_swept_by_curve_l588_588157


namespace problem1_l588_588137

theorem problem1 (P : ℝ → ℝ)
  (h_deg : ∃ n, n ≤ 4 ∧ ∀ x, monic (P x))
  (h1 : P 1 = 1)
  (h2 : P 2 = 4)
  (h3 : P 3 = 9)
  (h4 : P 4 = 16) :
  ∀ x, P x = x ^ 2 := 
sorry

end problem1_l588_588137


namespace log_y_minus_x_plus_1_gt_0_l588_588848

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588848


namespace r_5_eq_5_r_7_eq_10_r_k_general_l588_588740

def S_k (k : ℕ) : set (ℕ × ℕ) := 
  {p | p.1 ≤ k ∧ p.2 ≤ k ∧ p.1 > 0 ∧ p.2 > 0}

def indistinguishable (k : ℕ) (p q : ℕ × ℕ) : Prop := 
  (p.1 - q.1) % k = 0 ∨ (p.1 - q.1) % k = 1 ∨ (p.1 - q.1) % k = k - 1 ∧
  (p.2 - q.2) % k = 0 ∨ (p.2 - q.2) % k = 1 ∨ (p.2 - q.2) % k = k - 1

def distinguishable_set (k : ℕ) (A : set (ℕ × ℕ)) : Prop :=
  ∀ p q ∈ A, p ≠ q → ¬ indistinguishable k p q 

def r_k (k : ℕ) : ℕ :=
  nat.floor (k / 2) * nat.floor (k / 2)

theorem r_5_eq_5 : r_k 5 = 5 := sorry

theorem r_7_eq_10 : r_k 7 = 10 := sorry

theorem r_k_general (k : ℕ) : r_k k = nat.floor (k / 2) * nat.floor (k / 2) := sorry

end r_5_eq_5_r_7_eq_10_r_k_general_l588_588740


namespace hyperbola_eccentricity_l588_588554

open Real

variables {a b c e : ℝ} {x y : ℝ}

def hyperbola (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def line (x y : ℝ) : Prop := y = (3 / 2) * x

def intersection_projection (a b : ℝ) : ℝ := b^2 / a

def focus_distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

def eccentricity (a b : ℝ) : ℝ := (sqrt (a^2 + b^2)) / a

-- Prove the eccentricity is 2 given the line and hyperbola properties.
theorem hyperbola_eccentricity (a b c e : ℝ) (ha : a > 0) (hb : b > 0) (hy : line x y) 
  (h_intersection_proj : intersection_projection a b = 3 * focus_distance a b / 2) : 
  eccentricity a b = 2 := by
  sorry

end hyperbola_eccentricity_l588_588554


namespace T_2023_eq_6064_l588_588358

def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then 1
  else Real.log (sequence_a (n - 1)) + 2

def greatest_int (x : ℝ) : ℤ := Int.floor x

def T_n (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, greatest_int (sequence_a (i + 1))

theorem T_2023_eq_6064 : T_n 2023 = 6064 :=
  sorry

end T_2023_eq_6064_l588_588358


namespace min_marked_midpoints_l588_588494

-- Define the problem statement in Lean
theorem min_marked_midpoints (N : ℕ) (hN : N ≥ 2) : 
  ∃ (S : Finset (ℝ × ℝ)), (S.card = N) → ∀ (A B : ℝ × ℝ) (hA : A ∈ S) (hB : B ∈ S) (A ≠ B), 
  (∃ (M : Finset (ℝ × ℝ)), M.card = 2 * N - 3 ∧ 
  (∀ (C ∈ S), ∃ (m ∈ M), m = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) ∨ m = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))) :=
by sorry

end min_marked_midpoints_l588_588494


namespace area_of_shaded_region_l588_588190

-- Define points F, G, H, I, J with their coordinates
def F := (0, 0)
def G := (4, 0)
def H := (16, 0)
def I := (16, 12)
def J := (4, 3)

-- Define the similarity condition
def similar_triangles_JFG_IHG : Prop :=
  (triangle.similar F G J) (triangle.similar H G I)

-- The lengths of the segments based on problem conditions
def length_HG := 12
def length_JG := 3
def length_IG := 9

-- Area calculation of triangle IJG
def area_IJG := (1/2 * length_IG * length_JG).toReal

-- Final proof statement
theorem area_of_shaded_region :
  similar_triangles_JFG_IHG →
  length_HG = 12 →
  length_JG = length_HG/4 →
  length_IG = length_HG - length_JG →
  real.floor (area_IJG + 0.5) = 14 :=
by
  intros h_sim h_HG h_JG h_IG
  sorry

end area_of_shaded_region_l588_588190


namespace shaded_area_is_54_l588_588201

-- Define the coordinates of points O, A, B, C, D, E
structure Point where
  x : ℝ
  y : ℝ

-- Given points
def O := Point.mk 0 0
def A := Point.mk 4 0
def B := Point.mk 16 0
def C := Point.mk 16 12
def D := Point.mk 4 12
def E := Point.mk 4 3

-- Define the function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2) ^ (1/2)

-- Define similarity of triangles and calculate side lengths involved
def triangles_similarity (OA OB CB EA : ℝ) : Prop :=
  OA / OB = EA / CB

-- Define the condition
def condition : Prop := 
  triangles_similarity (distance O A) (distance O B) 12 (distance E A) ∧
  distance E A = 3 ∧
  distance D E = 9

-- Define the calculation of area of triangle given base and height
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- State that the area of triangle CDE is 54 cm²
def area_shaded_region : Prop :=
  triangle_area 9 12 = 54

-- Main theorem statement
theorem shaded_area_is_54 : condition → area_shaded_region := by
  sorry

end shaded_area_is_54_l588_588201


namespace distance_from_origin_to_neg12_16_l588_588425

theorem distance_from_origin_to_neg12_16 : 
  (Math.sqrt ((-12 : ℝ)^2 + (16 : ℝ)^2) = 20) :=
by
  sorry

end distance_from_origin_to_neg12_16_l588_588425


namespace cara_neighbors_l588_588297

def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem cara_neighbors : number_of_pairs 7 = 21 :=
by
  sorry

end cara_neighbors_l588_588297


namespace probability_four_same_color_l588_588259

theorem probability_four_same_color (green white blue red : ℕ) (total_balls : ℕ) (choose : ℕ → ℕ → ℕ) :
  green = 5 →
  white = 8 →
  blue = 4 →
  red = 3 →
  total_balls = green + white + blue + red →
  choose total_balls 4 = 4845 →
  choose green 4 = 5 →
  choose white 4 = 70 →
  choose blue 4 = 1 →
  choose red 4 = 0 →
  (5 + 70 + 1) / 4845 = 76 / 4845 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

-- Definitions required to make the statement complete.
def choose (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

end probability_four_same_color_l588_588259


namespace convex_polygons_of_four_or_more_sides_l588_588180

theorem convex_polygons_of_four_or_more_sides (n : ℕ) (h : n = 12) : 
  let total_subsets := 2^n,
      subsets_lt_4 := Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3 in
  total_subsets - subsets_lt_4 = 3797 :=
sorry

end convex_polygons_of_four_or_more_sides_l588_588180


namespace eq_x_add_q_l588_588910

theorem eq_x_add_q (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x > 5) : x + q = 5 + 2*q :=
by {
  sorry
}

end eq_x_add_q_l588_588910


namespace triangles_with_positive_area_l588_588397

-- Define the set of points in the coordinate grid
def points := { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4 }

-- Number of ways to choose 3 points from the grid
def total_triples := Nat.choose 16 3

-- Number of collinear triples
def collinear_triples := 32 + 8 + 4

-- Number of triangles with positive area
theorem triangles_with_positive_area :
  (total_triples - collinear_triples) = 516 :=
by
  -- Definitions for total_triples and collinear_triples.
  -- Proof steps would go here.
  sorry

end triangles_with_positive_area_l588_588397


namespace complex_quad_area_l588_588167

noncomputable def is_int (a : ℂ) : Prop :=
  isInteger a.re ∧ isInteger a.im

noncomputable def quad_area (z1 z2 z3 z4 : ℂ) : ℝ := 
  -- Use the Shoelace formula or any other method here
  sorry

theorem complex_quad_area : 
  ∃ (z1 z2 z3 z4 : ℂ), 
    (z1 * (conj z1)^3 + (conj z1) * (z1)^3 = 180) ∧
    (z2 * (conj z2)^3 + (conj z2) * (z2)^3 = 180) ∧ 
    (z3 * (conj z3)^3 + (conj z3) * (z3)^3 = 180) ∧ 
    (z4 * (conj z4)^3 + (conj z4) * (z4)^3 = 180) ∧ 
    is_int z1 ∧ is_int z2 ∧ is_int z3 ∧ is_int z4 ∧ 
    quad_area z1 z2 z3 z4 = 36 :=
begin
  sorry
end

end complex_quad_area_l588_588167


namespace grass_field_width_l588_588639

theorem grass_field_width (w : ℝ) (length_field : ℝ) (path_width : ℝ) (area_path : ℝ) :
  length_field = 85 → path_width = 2.5 → area_path = 1450 →
  (90 * (w + path_width * 2) - length_field * w = area_path) → w = 200 :=
by
  intros h_length_field h_path_width h_area_path h_eq
  sorry

end grass_field_width_l588_588639


namespace telephone_pole_height_l588_588283

-- Define the conditions
variables (h : ℝ) (AC : ℝ := 4) (AD : ℝ := 3) (DE : ℝ := 1.6)
-- Define the distance Leah walks towards the cable
def DC : ℝ := AC - AD
-- Define the similarity ratio
noncomputable def AB := (DE / DC) * AC

-- Statement of the problem
theorem telephone_pole_height : AB = 6.4 :=
by
  unfold AB DC
  have h1 : (4 * 1.6) = 6.4,
  { norm_num, },
  exact h1

end telephone_pole_height_l588_588283


namespace solve_equation_l588_588518

theorem solve_equation :
  ∃ (x y z : ℕ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 4 ∧ 
  (2^x + 3^y + 7 = nat.factorial z) ∧
  ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
by {
  sorry
}

end solve_equation_l588_588518


namespace distinct_sums_modulo_l588_588472

open Nat

theorem distinct_sums_modulo (p k l : ℕ) (hp : Nat.Prime p)
  (hx : ∀ (i j : ℕ), i ≠ j → (x i : ℤ) % p ≠ (x j : ℤ) % p)
  (hy : ∀ (i j : ℕ), i ≠ j → (y i : ℤ) % p ≠ (y j : ℤ) % p) :
  ∃ (s : Finset ℤ), (∀ i j, 1 ≤ i → i ≤ k → 1 ≤ j → j ≤ l → ((x i + y j) % p) ∈ s)
                    ∧ s.card ≥ min p (k + l - 1) :=
sorry

end distinct_sums_modulo_l588_588472


namespace simplify_and_evaluate_correct_l588_588128

noncomputable def simplify_and_evaluate (x y : ℚ) : ℚ :=
  3 * (x^2 - 2 * x * y) - (3 * x^2 - 2 * y + 2 * (x * y + y))

theorem simplify_and_evaluate_correct : 
  simplify_and_evaluate (-1 / 2 : ℚ) (-3 : ℚ) = -12 := by
  sorry

end simplify_and_evaluate_correct_l588_588128


namespace leopards_arrangement_1440_l588_588483

noncomputable def arrangement_count (n : Nat) (h_pos : ∀ l, l = n -> False) : Nat :=
  (factorial (n - 2)) * 2

theorem leopards_arrangement_1440 : arrangement_count 8 (by simp [Nat.zero_ne_zero]) = 1440 :=
by
  unfold arrangement_count
  simp
  norm_num
  sorry

end leopards_arrangement_1440_l588_588483


namespace ROI_difference_l588_588686

-- Definitions based on the conditions
def Emma_investment : ℝ := 300
def Briana_investment : ℝ := 500
def Emma_yield : ℝ := 0.15
def Briana_yield : ℝ := 0.10
def years : ℕ := 2

-- The goal is to prove that the difference between their 2-year ROI is $10
theorem ROI_difference :
  let Emma_ROI := Emma_investment * Emma_yield * years
  let Briana_ROI := Briana_investment * Briana_yield * years
  (Briana_ROI - Emma_ROI) = 10 :=
by
  sorry

end ROI_difference_l588_588686


namespace no_super_plus_good_exists_at_most_one_super_plus_good_l588_588301

def is_super_plus_good (board : ℕ → ℕ → ℕ) (n : ℕ) (i j : ℕ) : Prop :=
  (∀ k, k < n → board i k ≤ board i j) ∧ 
  (∀ k, k < n → board k j ≥ board i j)

def arrangement (n : ℕ) := { board : ℕ → ℕ → ℕ // ∀ i j, i < n → j < n → 1 ≤ board i j ∧ board i j ≤ n * n }

-- Prove that in some arrangements, there is no super-plus-good number.
theorem no_super_plus_good_exists (n : ℕ) (h₁ : n = 8) :
  ∃ (b : arrangement n), ∀ i j, ¬ is_super_plus_good b.val n i j := sorry

-- Prove that in every arrangement, there is at most one super-plus-good number.
theorem at_most_one_super_plus_good (n : ℕ) (h : n = 8) :
  ∀ (b : arrangement n), ∃! i j, is_super_plus_good b.val n i j := sorry

end no_super_plus_good_exists_at_most_one_super_plus_good_l588_588301


namespace solve_equation_nat_numbers_l588_588515

theorem solve_equation_nat_numbers :
  ∃ (x y z : ℕ), (2 ^ x + 3 ^ y + 7 = z!) ∧ ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
sorry

end solve_equation_nat_numbers_l588_588515


namespace smallest_positive_four_digit_integer_equivalent_to_2_mod_5_l588_588223

theorem smallest_positive_four_digit_integer_equivalent_to_2_mod_5 :
  ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 2 ∧ n = 1002 :=
begin
  sorry,
end

end smallest_positive_four_digit_integer_equivalent_to_2_mod_5_l588_588223


namespace inequality_implies_log_pos_l588_588828

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588828


namespace sum_of_three_consecutive_integers_product_990_l588_588551

theorem sum_of_three_consecutive_integers_product_990 
  (a b c : ℕ) 
  (h1 : b = a + 1)
  (h2 : c = b + 1)
  (h3 : a * b * c = 990) :
  a + b + c = 30 :=
sorry

end sum_of_three_consecutive_integers_product_990_l588_588551


namespace points_per_correct_answer_l588_588043

theorem points_per_correct_answer (x : ℕ) : 
  let total_questions := 30
  let points_deducted_per_incorrect := 5
  let total_score := 325
  let correct_answers := 19
  let incorrect_answers := total_questions - correct_answers
  let points_lost_from_incorrect := incorrect_answers * points_deducted_per_incorrect
  let score_from_correct := correct_answers * x
  (score_from_correct - points_lost_from_incorrect = total_score) → x = 20 :=
by {
  sorry
}

end points_per_correct_answer_l588_588043


namespace lock_settings_count_l588_588282

noncomputable def different_lock_settings : ℕ :=
  ∏ i in finset.range(10), if i < 4 then (10 - i) else 1

theorem lock_settings_count : different_lock_settings = 5040 := by
  sorry

end lock_settings_count_l588_588282


namespace count_numbers_without_1_or_2_l588_588398

/-- The number of whole numbers between 1 and 2000 that do not contain the digits 1 or 2 is 511. -/
theorem count_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 511 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2000 →
      ¬ (∃ d : ℕ, (k.digits 10).contains d ∧ (d = 1 ∨ d = 2)) → n = 511) :=
sorry

end count_numbers_without_1_or_2_l588_588398


namespace correspondence_mappings_l588_588315

open Set

def is_mapping (A B : Set α) (f : α → β) : Prop :=
  ∀ (x : α), x ∈ A → ∃! (y : β), y = f x ∧ y ∈ B

/-- Problem statement -/
theorem correspondence_mappings :
  let A1 := {1, 2, 3, 4}
  let B1 := {3, 4, 5, 6, 7, 8, 9}
  let f1 := λ x, 2 * x + 1

  let A2 := {0, 1, 2}
  let B2 := {-1, 0, 1, 2}
  let f2 := λ x, 2 * x - 1

  let A3 := {x | x ∈ ℕ ∧ x ≠ 0}
  let B3 := {0, 1}
  let f3 := λ x, x % 2

  let A4 := ℝ
  let f4 := λ x, if x = 0 then 0 else sqrt x
  let B4 := ℝ

  (is_mapping A1 B1 f1) ∧ ¬(is_mapping A2 B2 f2) ∧ (is_mapping A3 B3 f3) ∧ ¬(is_mapping A4 B4 f4) := 
begin
  sorry
end

end correspondence_mappings_l588_588315


namespace length_of_OC_l588_588938

-- Define the quadratic function y = x^2 + ax + b
def quadratic (a b : ℝ) (x : ℝ) := x^2 + a * x + b

-- Condition 1: The function y = x^2 + ax + b
-- Condition 2: Line AB is perpendicular to line y = x (slope is -1)
-- Condition 3: Line AB is parallel to the line y = -x
-- Condition 4: Line AB passes through the point B(0, b)
-- Condition 5: x = b is one of the roots of the quadratic equation x^2 + ax + b = 0

theorem length_of_OC (a b : ℝ) (hB : 0 ^ 2 + a * 0 + b = b) (hAB_parallel : ∀ x, quadratic a b x = -x + b) 
    (hVieta1 : b * 1 = b) 
    (hVieta2 : b + 1 = -a): 
    length OC = 1 := 
by 
  sorry

end length_of_OC_l588_588938


namespace sum_sequence_conditions_l588_588771

variable {S : ℕ → ℝ}
variable {a : ℕ → ℝ}

noncomputable def common_difference (d : ℝ) : Prop :=
  ∀ n : ℕ, a(n+1) = a(n) + d

theorem sum_sequence_conditions (h1 : S 5 < S 6) (h2 : S 6 = S 7 ∧ S 7 > S 8) :
  (S 6 = S 7) ∧ (a 7 = 0) ∧ (∃ d : ℝ, common_difference d ∧ d < 0) ∧ ¬ (S 9 > S 5) :=
by
  sorry

end sum_sequence_conditions_l588_588771


namespace find_third_vertex_l588_588183

def vertex1 : ℝ × ℝ := (7, 5)
def vertex2 : ℝ × ℝ := (0, 0)
def area_of_triangle : ℝ := 35
def y_difference : ℝ := 5
def base : ℝ := (2 * area_of_triangle) / y_difference
def third_vertex := (-(base), 0)

theorem find_third_vertex :
  third_vertex = (-14, 0) :=
by
  unfold third_vertex base y_difference area_of_triangle vertex1 vertex2
  calc
    third_vertex = (-(2 * 35 / 5), 0) : by simp [area_of_triangle, y_difference]
             ... = (-14, 0)         : by norm_num

end find_third_vertex_l588_588183


namespace area_of_shaded_region_l588_588192

-- Define points F, G, H, I, J with their coordinates
def F := (0, 0)
def G := (4, 0)
def H := (16, 0)
def I := (16, 12)
def J := (4, 3)

-- Define the similarity condition
def similar_triangles_JFG_IHG : Prop :=
  (triangle.similar F G J) (triangle.similar H G I)

-- The lengths of the segments based on problem conditions
def length_HG := 12
def length_JG := 3
def length_IG := 9

-- Area calculation of triangle IJG
def area_IJG := (1/2 * length_IG * length_JG).toReal

-- Final proof statement
theorem area_of_shaded_region :
  similar_triangles_JFG_IHG →
  length_HG = 12 →
  length_JG = length_HG/4 →
  length_IG = length_HG - length_JG →
  real.floor (area_IJG + 0.5) = 14 :=
by
  intros h_sim h_HG h_JG h_IG
  sorry

end area_of_shaded_region_l588_588192


namespace james_total_riding_time_including_rest_stop_l588_588065

theorem james_total_riding_time_including_rest_stop :
  let distance1 := 40 -- miles
  let speed1 := 16 -- miles per hour
  let distance2 := 40 -- miles
  let speed2 := 20 -- miles per hour
  let rest_stop := 20 -- minutes
  let rest_stop_in_hours := rest_stop / 60 -- convert to hours
  let time1 := distance1 / speed1 -- time for the first part
  let time2 := distance2 / speed2 -- time for the second part
  let total_time := time1 + rest_stop_in_hours + time2 -- total time including rest
  total_time = 4.83 :=
by
  sorry

end james_total_riding_time_including_rest_stop_l588_588065


namespace polygon_center_zero_vector_sum_x_l588_588963

variables {A : Type} [add_comm_group A]
variables (n : ℕ)

def centroid (p : fin n → A) : A := (finset.univ.sum p) / n

variables {A1 A2 A3 : Type} (O : A)
variables {X : A}

-- Assuming group operations and additive identities as necessary
theorem polygon_center_zero (A_i : fin n → A) (regular_polygon : ∀ i, A_i i = A_i ((i + 1) % n)) :
  ∑ i, A_i i = 0 :=
by sorry

-- The vector sum from an arbitrary point
theorem vector_sum_x (A_i : fin n → A) (regular_polygon : ∀ i, A_i i = A_i ((i + 1) % n)) : 
  ∑ i, (X + A_i i) = n * X :=
by sorry

end polygon_center_zero_vector_sum_x_l588_588963


namespace length_of_angle_bisector_PT_approx_l588_588057

-- Define the triangle and its properties
variables {P Q R T : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace T]
noncomputable theory

-- Define the sides and the cosine of the angle at P
def PQ : ℝ := 5
def PR : ℝ := 8
def cos_P : ℝ := 1 / 5

-- Define the length of the angle bisector PT
def length_PT : ℝ := 5.05

-- State the theorem
theorem length_of_angle_bisector_PT_approx : 
  ∃ (P Q R T : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace T], 
  PQ = 5 → PR = 8 → cos_P = 1 / 5 → dist P T = 5.05 :=
sorry

end length_of_angle_bisector_PT_approx_l588_588057


namespace fill_pool_in_time_l588_588644

noncomputable def combined_fill_time (W X Y : ℝ) : ℝ :=
  let rate_total := (19.5 / 36) * 0.9
  in 1 / rate_total

theorem fill_pool_in_time :
  ∀ (W X Y : ℝ), 
    W + X = 1/3 →
    W + Y = 1/6 →
    X + Y = 2/9 →
    combined_fill_time W X Y = 2.05 :=
by
  intros W X Y h1 h2 h3
  sorry

end fill_pool_in_time_l588_588644


namespace imaginary_part_conjugate_z_l588_588773

noncomputable def z : ℂ := (1 + 2 * complex.I) / (1 + complex.I)

theorem imaginary_part_conjugate_z :
  complex.im (conj z) = -1 / 2 := 
sorry

end imaginary_part_conjugate_z_l588_588773


namespace mean_equal_l588_588542

theorem mean_equal (y : ℚ) :
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := 
by
  sorry

end mean_equal_l588_588542


namespace count_lines_through_points_l588_588103

-- Define the conditions
def points_on_plane : list (ℤ × ℤ) :=
  [(x, y) | x in [0, 1, 2], y in list.range 27]

-- Definition for the line through exactly three points
def line_through_three_points (a b c : ℤ) : bool :=
  (2 * b = a + c)

#eval (let e := (list.range 14).map (λ n, 2 * n) in
       let o := (list.range 13).map (λ n, 2 * n + 1) in
       2 * (e.length * e.length + o.length * o.length))

def num_lines_through_three_points : ℕ :=
  (let even_vals := list.range' 0 14).map (λ n, 2 * n) in 
  let odd_vals := list.range' 0 13).map (λ n, 2 * n + 1) in 
  2 * (even_vals.length * even_vals.length + odd_vals.length * odd_vals.length)

theorem count_lines_through_points : num_lines_through_three_points = 365 :=
sorry

end count_lines_through_points_l588_588103


namespace product_three_consecutive_integers_divisible_by_six_l588_588075

theorem product_three_consecutive_integers_divisible_by_six
  (n : ℕ) (h_pos : 0 < n) : ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k :=
by sorry

end product_three_consecutive_integers_divisible_by_six_l588_588075


namespace cardinality_B_l588_588567

-- Definitions based on the conditions
variables {U A B : Finset α} 

-- Condition 1: |U| = 193
def cardinality_U : U.card = 193 := sorry

-- Condition 2: 59 items are not members of either set A or B
def not_in_A_or_B : U.filter (λ x, x ∉ A ∧ x ∉ B).card = 59 := sorry

-- Condition 3: |A ∩ B| = 23
def cardinality_A_inter_B : (A ∩ B).card = 23 := sorry

-- Condition 4: |A| = 116
def cardinality_A : A.card = 116 := sorry

-- The proof that |B| = 41 given the above conditions
theorem cardinality_B : B.card = 41 :=
by sorry

end cardinality_B_l588_588567


namespace sum_first_53_odd_numbers_l588_588601

-- Definitions based on the given conditions
def first_odd_number := 1

def nth_odd_number (n : ℕ) : ℕ :=
  1 + (n - 1) * 2

def sum_n_odd_numbers (n : ℕ) : ℕ :=
  (n * n)

-- Theorem statement to prove
theorem sum_first_53_odd_numbers : sum_n_odd_numbers 53 = 2809 := 
by
  sorry

end sum_first_53_odd_numbers_l588_588601


namespace number_of_apples_remaining_l588_588009

def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples_before_giving_away : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℕ := total_apples_before_giving_away / 5
def apples_remaining : ℕ := total_apples_before_giving_away - apples_given_to_son

theorem number_of_apples_remaining : apples_remaining = 12 :=
by
  sorry

end number_of_apples_remaining_l588_588009


namespace function_count_l588_588956

theorem function_count (n k : ℕ) (X : Finset ℕ) (f : ℕ → ℕ) (iterate : ℕ → (ℕ → ℕ)) :
  X = {1, 2, ..., n} ∧ k ∈ X ∧ (∀ x ∈ X, ∃ j ≥ 0, iterate j f x ≤ k) → 
  (Finset.card {f : X → X | ∀ x ∈ X, ∃ j ≥ 0, iterate j f x ≤ k} = k * n^(n-1)) :=
by
  sorry

end function_count_l588_588956


namespace max_value_problem_l588_588336

theorem max_value_problem :
  ∃ (x y : ℝ), 
  (∀ (a b : ℝ), (a, b) = (x, y) → 
  max (λ z : ℝ × ℝ, (λ (x y : ℝ), (2 * x + 3 * y + 4) / sqrt (2 * x^2 + 3 * y^2 + 5)) = (z.1, z.2)) = sqrt 28) :=
sorry

end max_value_problem_l588_588336


namespace simplify_expression_l588_588021

theorem simplify_expression (x y : ℝ) (P Q : ℝ) (hP : P = 2 * x + 3 * y) (hQ : Q = 3 * x + 2 * y) :
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (24 * x ^ 2 + 52 * x * y + 24 * y ^ 2) / (5 * x * y - 5 * y ^ 2) :=
by
  sorry

end simplify_expression_l588_588021


namespace num_of_even_distinct_digits_l588_588801

noncomputable def count_even_distinct (n : ℕ) : ℕ :=
if h : 1000 ≤ n ∧ n ≤ 9999 then
  let digits := (n / 1000) % 10 :: (n / 100) % 10 :: (n / 10) % 10 :: n % 10 :: [] in
  let is_distinct := digits.nodup in
  if is_distinct ∧ n % 2 = 0 then 1 else 0
else 0

theorem num_of_even_distinct_digits : 
  (Finset.range 9000).sum count_even_distinct = 2240 := 
by sorry

end num_of_even_distinct_digits_l588_588801


namespace probability_of_winning_pair_is_correct_l588_588931

noncomputable def probability_of_winning_pair : ℚ :=
  let total_cards := 10
  let red_cards := 5
  let blue_cards := 5
  let total_ways := Nat.choose total_cards 2 -- Combination C(10,2)
  let same_color_ways := Nat.choose red_cards 2 + Nat.choose blue_cards 2 -- Combination C(5,2) for each color
  let consecutive_pairs_per_color := 4
  let consecutive_ways := 2 * consecutive_pairs_per_color -- Two colors
  let favorable_ways := same_color_ways + consecutive_ways
  favorable_ways / total_ways

theorem probability_of_winning_pair_is_correct : 
  probability_of_winning_pair = 28 / 45 := sorry

end probability_of_winning_pair_is_correct_l588_588931


namespace functional_equation_l588_588341

theorem functional_equation 
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
  (h2 : f 0 ≠ 0) :
  f 2009 = 1 :=
sorry

end functional_equation_l588_588341


namespace main_theorem_l588_588957

variable {M : Type} (Ω : Set (Set M))

-- Given conditions
def condition_i : Prop :=
  ∀ A B ∈ Ω, (A ∩ B).card ≥ 2 → A = B

def condition_ii : Prop :=
  ∃ A B C ∈ Ω, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (A ∩ B ∩ C).card = 1

def condition_iii : Prop :=
  ∀ A ∈ Ω, ∀ a ∈ (M \ A : Set M), ∃! B ∈ Ω, a ∈ B ∧ A ∩ B = ∅

-- Main statement to prove
theorem main_theorem (h_i : condition_i Ω) (h_ii : condition_ii Ω) (h_iii : condition_iii Ω) :
  ∃ (p s : ℕ), (∀ a ∈ M, (Finset.filter (λ A : Set M, a ∈ A) Ω).card = p) ∧
  (∀ A ∈ Ω, Finset.card A = s) ∧
  s + 1 ≥ p :=
sorry

end main_theorem_l588_588957


namespace candy_total_l588_588728

theorem candy_total (pieces_per_bag : ℕ) (number_of_bags : ℕ) : pieces_per_bag = 11 → number_of_bags = 2 → pieces_per_bag * number_of_bags = 22 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end candy_total_l588_588728


namespace segment_length_eq_ten_l588_588592

theorem segment_length_eq_ten :
  (abs (8 - (-2)) = 10) :=
by
  -- Given conditions
  have h1 : 8 = real.cbrt 27 + 5 := sorry
  have h2 : -2 = real.cbrt 27 - 5 := sorry
  
  -- Using the conditions to prove the length
  sorry

end segment_length_eq_ten_l588_588592


namespace total_colored_hangers_l588_588033

theorem total_colored_hangers (pink_hangers green_hangers : ℕ) (h1 : pink_hangers = 7) (h2 : green_hangers = 4)
  (blue_hangers yellow_hangers : ℕ) (h3 : blue_hangers = green_hangers - 1) (h4 : yellow_hangers = blue_hangers - 1) :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 :=
by
  sorry

end total_colored_hangers_l588_588033


namespace rearrangements_count_is_4_l588_588804

def isValidRearrangement (s : List Char) : Prop :=
  (∀ i, i < s.length - 1 → (s.get i, s.get (i+1)) ≠ ('w', 'x') ∧ (s.get i, s.get (i+1)) ≠ ('x', 'y') ∧ (s.get i, s.get (i+1)) ≠ ('y', 'z')) ∧
  (('w', s.filter (· = 'w')).length = 2)

def validRearrangementsCount : ℕ :=
  (["w", "w", "x", "y", "z"].permutations.filter isValidRearrangement).length

theorem rearrangements_count_is_4 : validRearrangementsCount = 4 := by
  sorry

end rearrangements_count_is_4_l588_588804


namespace number_of_zeros_f_max_ab_if_f_leq_l588_588783

section problem_1

def f (x : ℝ) := Real.log(-x + 1) + Real.exp(x - 1)
theorem number_of_zeros_f : ∃! c : ℝ, f c = 0 :=
sorry

end problem_1

section problem_2

noncomputable def g (a b x : ℝ) := Real.log(a * x + b) + Real.exp(x - 1)
theorem max_ab_if_f_leq : ∀ a b : ℝ, (∀ x : ℝ, g a b x ≤ Real.exp(x - 1) + x + 1) → a * b ≤ (1/2) * Real.exp 3 :=
sorry

end problem_2

end number_of_zeros_f_max_ab_if_f_leq_l588_588783


namespace discount_is_10_percent_l588_588636

def cost_price : ℝ := 100
def markup_percentage : ℝ := 0.20
def profit_percentage : ℝ := 0.08

def marked_price (CP : ℝ) (markup : ℝ) : ℝ := CP * (1 + markup)
def selling_price (CP : ℝ) (profit : ℝ) : ℝ := CP * (1 + profit)

-- Now defining the discount percentage
def discount_percentage (MP SP : ℝ) : ℝ := (MP - SP) / MP * 100

theorem discount_is_10_percent :
  discount_percentage (marked_price cost_price markup_percentage) (selling_price cost_price profit_percentage) = 10 :=
by
  sorry

end discount_is_10_percent_l588_588636


namespace martin_waste_time_l588_588485

theorem martin_waste_time : 
  let waiting_traffic := 2
  let trying_off_freeway := 4 * waiting_traffic
  let detours := 3 * 30 / 60
  let meal := 45 / 60
  let delays := (20 + 40) / 60
  waiting_traffic + trying_off_freeway + detours + meal + delays = 13.25 := 
by
  sorry

end martin_waste_time_l588_588485


namespace triangle_angle_BDE_l588_588943

theorem triangle_angle_BDE :
  ∀ (A B C D E : Type) [Triangle A B C]
    (h1 : AB = AC)
    (h2 : Point D on segment AC)
    (h3 : Bisects (BD) (angle ABC))
    (h4 : Point E on segment BC)
    (h5 : Bisects (DE) (angle BDC))
    (h6 : BD = BC), 
  measure (angle BDE) = 36 :=
by
  sorry

end triangle_angle_BDE_l588_588943


namespace tangent_lines_l588_588775

noncomputable def circle : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 = 5

def point_Q : (ℝ × ℝ) := (3, 1)
def point_P : (ℝ × ℝ) := (2, 1)

def tangent_equation (x y k : ℝ) : Prop :=
  y - 1 = k * (x - 3)

def tangent_equations (x y : ℝ) : Prop :=
  (x + 2 * y - 5 = 0) ∨ (2 * x - y - 5 = 0) ∨ (2 * x + y - 5 = 0)

theorem tangent_lines :
  ∀ x y : ℝ, circle x y → (tangent_equation x y 2 ∨ tangent_equation x y (-1/2) ∨ tangent_equations x y) → tangent_equations x y ∧ tangent_equation x y (-2)
  :=
by
  sorry

end tangent_lines_l588_588775


namespace wooden_toys_count_l588_588488

theorem wooden_toys_count :
  ∃ T : ℤ, 
    10 * 40 + 20 * T - (10 * 36 + 17 * T) = 64 ∧ T = 8 :=
by
  use 8
  sorry

end wooden_toys_count_l588_588488


namespace proof_problem_l588_588250

def feasible_plans
  (total_corners : ℕ)
  (total_science_books total_humanities_books : ℕ) :
  list (ℕ × ℕ) :=
[(18, 12), (19, 11), (20, 10)]

noncomputable def lowest_cost_plan
  (medium_cost small_cost : ℕ)
  (plans : list (ℕ × ℕ)) :
  ℕ :=
let costs := plans.map (λ p, p.1 * medium_cost + p.2 * small_cost) in
costs.minimum.get_or_else 0

theorem proof_problem :
  lowest_cost_plan
    860 570 (feasible_plans 30 1900 1620) = 22320 :=
by sorry

end proof_problem_l588_588250


namespace minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l588_588711

theorem minimal_solution (x : ℝ) (h : x * |x| = 3 * x + 2) : -2 ≤ x :=
begin
  sorry
end

theorem x_eq_neg_two_is_solution : ( -2 : ℝ ) * |-2| = 3 * -2 + 2 :=
begin
  norm_num,
end

/-- The smallest value of x satisfying x|x| = 3x + 2 is -2 -/
theorem smallest_solution : ∃ x : ℝ, x * |x| = 3 * x + 2 ∧ ∀ y : ℝ, y * |y| = 3 * y + 2 → y ≥ x :=
begin
  use -2,
  split,
  { norm_num },
  { intro y,
    sorry }
end

end minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l588_588711


namespace number_of_neutrons_eq_l588_588939

variable (A n x : ℕ)

/-- The number of neutrons N in the nucleus of an atom R, given that:
  1. A is the atomic mass number of R.
  2. The ion RO3^(n-) contains x outer electrons. -/
theorem number_of_neutrons_eq (N : ℕ) (h : A - N + 24 + n = x) : N = A + n + 24 - x :=
by sorry

end number_of_neutrons_eq_l588_588939


namespace num_and_sum_of_f50_l588_588464

noncomputable def f : ℕ → ℕ := sorry 

axiom f_condition : ∀ a b : ℕ, 2 * f (a^2 + b^2) = (f a)^2 + (f b)^2

theorem num_and_sum_of_f50 : 
  let values := {f 50 | ∃ (a b : ℕ), 2 * f (a^2 + b^2) = (f a)^2 + (f b)^2} in
  let n := values.card in
  let s := values.sum in
  n = 3 ∧ s = 401 ∧ n * s = 1203 :=
by {
  sorry
}

end num_and_sum_of_f50_l588_588464


namespace p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l588_588753

def p (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 1) + (y^2) / (m - 4) = 1
def q (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 2) + (y^2) / (4 - m) = 1

theorem p_hyperbola_implies_m_range (m : ℝ) (x y : ℝ) :
  p m x y → 1 < m ∧ m < 4 :=
sorry

theorem p_necessary_not_sufficient_for_q (m : ℝ) (x y : ℝ) :
  (1 < m ∧ m < 4) ∧ p m x y →
  (q m x y → (2 < m ∧ m < 3) ∨ (3 < m ∧ m < 4)) :=
sorry

end p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l588_588753


namespace find_value_ratio_l588_588235

noncomputable def roots_of_quadratic := {a b : ℝ // a ≠ b ∧ is_root (λ x : ℝ, x^2 + 8*x + 4) a ∧ is_root (λ x : ℝ, x^2 + 8*x + 4) b}

theorem find_value_ratio {a b : ℝ} (h : a ≠ b) (ha : is_root (λ x: ℝ, x^2 + 8*x + 4) a) (hb : is_root (λ x : ℝ, x^2 + 8*x + 4) b) : 
  (a / b + b / a) = 14 :=
sorry

end find_value_ratio_l588_588235


namespace slope_of_parallel_lines_l588_588004

theorem slope_of_parallel_lines (P1 P2 : ℝ × ℝ)
  (h1 : P1 = (1, 0)) 
  (h2 : P2 = (0, 5))
  (h3 : ∃ m : ℝ, (P1, P2) ⊆ {p : ℝ × ℝ | p.2 = m * p.1}) 
  (h4 : ∀ l1 l2 : ℝ × ℝ → ℝ, 
    (l1 = λ p, 0 - p.1) ∧ (l2 = λ p, 5 - p.1) ∧ 
    ∀ p : ℝ × ℝ, distance p l1 = 5 → distance p l2 = 5) :
  ∃ k : ℝ, k = 0 ∨ k = 5/12 :=
sorry

end slope_of_parallel_lines_l588_588004


namespace range_of_x_l588_588557

variable (x : ℝ)

-- Conditions used in the problem
def sqrt_condition : Prop := x + 2 ≥ 0
def non_zero_condition : Prop := x + 1 ≠ 0

-- The statement to be proven
theorem range_of_x : sqrt_condition x ∧ non_zero_condition x ↔ (x ≥ -2 ∧ x ≠ -1) :=
by
  sorry

end range_of_x_l588_588557


namespace x_sq_plus_inv_sq_l588_588403

theorem x_sq_plus_inv_sq (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
  sorry

end x_sq_plus_inv_sq_l588_588403


namespace total_blocks_fell_l588_588445

-- Definitions based on the conditions
def first_stack_height := 7
def second_stack_height := first_stack_height + 5
def third_stack_height := second_stack_height + 7

def first_stack_fallen_blocks := first_stack_height  -- All blocks fell down
def second_stack_fallen_blocks := second_stack_height - 2  -- 2 blocks left standing
def third_stack_fallen_blocks := third_stack_height - 3  -- 3 blocks left standing

-- Total fallen blocks
def total_fallen_blocks := first_stack_fallen_blocks + second_stack_fallen_blocks + third_stack_fallen_blocks

-- Theorem to prove the total number of fallen blocks
theorem total_blocks_fell : total_fallen_blocks = 33 :=
by
  -- Proof omitted, statement given as required
  sorry

end total_blocks_fell_l588_588445


namespace LN_eq_MN_l588_588346

noncomputable def triangle (A B C : Point) : Triangle := sorry
noncomputable def median (A B C : Point) : Line := sorry
noncomputable def perpendicular_from_point (P : Point) (L : Line) : Line := sorry
noncomputable def intersection (L1 L2 : Line) : Point := sorry
noncomputable def distance (P1 P2 : Point) : ℝ := sorry

variable {A B C P L M N : Point}

theorem LN_eq_MN : 
  ∀ (ABC : Triangle) (P : Point),
  let L := intersection (perpendicular_from_point P (line A B)) (line A B),
  let M := intersection (perpendicular_from_point P (line A C)) (line A C),
  let E := midpoint A (midpoint B C),
  let N := intersection (perpendicular_from_point P (line A E)) (line A D) in
  distance L N = distance M N :=
sorry

end LN_eq_MN_l588_588346


namespace new_manufacturing_cost_proof_l588_588676

-- Definitions based on conditions
variables (P C : ℝ) -- P: selling price, C: new manufacturing cost

-- Initial condition: manufacturing cost was $70 and initial profit was 30% of selling price
def initial_manufacturing_cost : ℝ := 70
def initial_profit : ℝ := 0.30 * P

-- Selling price equation before cost decrease
def selling_price_initial : Prop := P = initial_manufacturing_cost + initial_profit

-- New condition: profit is 50% of selling price
def new_profit : ℝ := 0.50 * P

-- Selling price equation after cost decrease
def selling_price_new : Prop := P = C + new_profit

-- Resulting equation to prove
theorem new_manufacturing_cost_proof (h1: selling_price_initial) (h2: selling_price_new) : C = 50 :=
by {
  sorry
}

end new_manufacturing_cost_proof_l588_588676


namespace segment_length_eq_ten_l588_588590

theorem segment_length_eq_ten :
  (abs (8 - (-2)) = 10) :=
by
  -- Given conditions
  have h1 : 8 = real.cbrt 27 + 5 := sorry
  have h2 : -2 = real.cbrt 27 - 5 := sorry
  
  -- Using the conditions to prove the length
  sorry

end segment_length_eq_ten_l588_588590


namespace john_ate_10_chips_l588_588949

variable (c p : ℕ)

/-- Given the total calories from potato chips and the calories increment of cheezits,
prove the number of potato chips John ate. -/
theorem john_ate_10_chips (h₀ : p * c = 60)
  (h₁ : ∃ c_cheezit, (c_cheezit = (4 / 3 : ℝ) * c))
  (h₂ : ∀ c_cheezit, p * c + 6 * c_cheezit = 108) :
  p = 10 :=
by {
  sorry
}

end john_ate_10_chips_l588_588949


namespace segment_length_abs_eq_five_l588_588582

theorem segment_length_abs_eq_five : 
  (length : ℝ) (∀ x : ℝ, abs (x - (27 : ℝ)^(1 : ℝ) / (3 : ℝ)) = 5 → x = 8 ∨ x = -2) 
  → length = 10 := 
begin
  sorry
end

end segment_length_abs_eq_five_l588_588582


namespace surface_area_of_rectangular_solid_is_334_l588_588682

theorem surface_area_of_rectangular_solid_is_334
  (l w h : ℕ)
  (h_l_prime : Prime l)
  (h_w_prime : Prime w)
  (h_h_prime : Prime h)
  (volume_eq_385 : l * w * h = 385) : 
  2 * (l * w + l * h + w * h) = 334 := 
sorry

end surface_area_of_rectangular_solid_is_334_l588_588682


namespace alpha_gamma_relationship_l588_588946

theorem alpha_gamma_relationship (α β γ : ℝ) (h1 : β = 10^(1 / (1 - Real.log10 α)))
  (h2 : γ = 10^(1 / (1 - Real.log10 β))) : 
  α = 10^(1 / (1 - Real.log10 γ)) :=
by
  sorry

end alpha_gamma_relationship_l588_588946


namespace total_colored_hangers_l588_588032

theorem total_colored_hangers (pink_hangers green_hangers : ℕ) (h1 : pink_hangers = 7) (h2 : green_hangers = 4)
  (blue_hangers yellow_hangers : ℕ) (h3 : blue_hangers = green_hangers - 1) (h4 : yellow_hangers = blue_hangers - 1) :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 :=
by
  sorry

end total_colored_hangers_l588_588032


namespace right_triangle_hypotenuse_l588_588426

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 6) (k : b = 8) (pt : a^2 + b^2 = c^2) : c = 10 := by
  sorry

end right_triangle_hypotenuse_l588_588426


namespace clara_total_points_l588_588036

-- Define the constants
def percentage_three_point_shots : ℝ := 0.25
def points_per_successful_three_point_shot : ℝ := 3
def percentage_two_point_shots : ℝ := 0.40
def points_per_successful_two_point_shot : ℝ := 2
def total_attempts : ℕ := 40

-- Define the function to calculate the total score
def total_score (x y : ℕ) : ℝ :=
  (percentage_three_point_shots * points_per_successful_three_point_shot) * x +
  (percentage_two_point_shots * points_per_successful_two_point_shot) * y

-- The proof statement
theorem clara_total_points (x y : ℕ) (h : x + y = total_attempts) : 
  total_score x y = 32 :=
by
  -- This is a placeholder for the actual proof
  sorry

end clara_total_points_l588_588036


namespace ball_travel_distance_l588_588257

noncomputable def total_distance_traveled (initial_height : ℕ) (bounce_fraction : ℚ) (hits : ℕ) : ℚ :=
  let descent := (0 : ℚ).add (initial_height) + (hits.nat_pred).sum (λ n, initial_height * (bounce_fraction ^ n))
  let ascent := (hits).sum (λ n, initial_height * (bounce_fraction ^ n))
  descent + ascent

theorem ball_travel_distance :
  total_distance_traveled 20 (2/3) 4 ≈ 80 :=
by
  sorry

end ball_travel_distance_l588_588257


namespace log_base2_1500_bounds_sum_l588_588325

theorem log_base2_1500_bounds_sum : 
  ∃ a b : ℤ, a < Real.logb 2 1500 ∧ Real.logb 2 1500 < b ∧ a + b = 21 :=
by 
  have h₁ : Real.logb 2 1024 = 10 := by sorry,
  have h₂ : Real.logb 2 2048 = 11 := by sorry,
  have h₃ : 1024 < 1500 ∧ 1500 < 2048 := by sorry,
  use [10, 11],
  split,
  { sorry },
  split,
  { sorry },
  { sorry }

end log_base2_1500_bounds_sum_l588_588325


namespace intersection_M_N_l588_588074

open Set

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l588_588074


namespace line_bisects_circle_l588_588247

theorem line_bisects_circle (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, l x y ↔ x - y = 0) → 
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 = 1) → 
  ∀ x y : ℝ, (x - y = 0) ∨ (x + y = 0) → l x y ∧ C x y → l x y = (x - y = 0) := by
  sorry

end line_bisects_circle_l588_588247


namespace base8_subtraction_l588_588703

-- Define the base 8 notation for the given numbers
def b8_256 := 256
def b8_167 := 167
def b8_145 := 145

-- Define the sum of 256_8 and 167_8 in base 8
def sum_b8 := 435

-- Define the result of subtracting 145_8 from the sum in base 8
def result_b8 := 370

-- Prove that the result of the entire operation is 370_8
theorem base8_subtraction : sum_b8 - b8_145 = result_b8 := by
  sorry

end base8_subtraction_l588_588703


namespace third_week_cut_correct_l588_588280

-- Definitions of conditions
def original_weight : ℝ := 180
def first_week_percent_cut : ℝ := 0.28
def second_week_percent_cut : ℝ := 0.18
def final_weight : ℝ := 85.0176

-- Definitions used in the proof
def weight_after_first_week : ℝ := original_weight * (1 - first_week_percent_cut)
def weight_after_second_week : ℝ := weight_after_first_week * (1 - second_week_percent_cut)

def third_week_percent_cut : ℝ :=
  100 * (1 - final_weight / weight_after_second_week)

-- Lean 4 statement to prove the third week percentage cut
theorem third_week_cut_correct : third_week_percent_cut = 20.01 := by
  sorry

end third_week_cut_correct_l588_588280


namespace segment_length_abs_cubed_root_l588_588596

theorem segment_length_abs_cubed_root (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  ∃ a b : ℝ, a = 3 + 5 ∧ b = 3 - 5 ∧ (b - a).abs = 10 :=
by {
  have h1 : real.cbrt 27 = 3 := by norm_num,
  rw h1 at h,
  have h2 : |x - 3| = 5 := h,
  use [8, -2],
  split,
  { refl },
  { split,
    { refl },
    { norm_num } }
}

end segment_length_abs_cubed_root_l588_588596


namespace proof_part1_A_inter_B_proof_part1_complement_B_union_A_proof_part2_C_subset_A_range_a_l588_588388

section
variable (x a : ℝ)

noncomputable def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
noncomputable def B : Set ℝ := {x | log 2 x > 1}
noncomputable def C (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 1}

theorem proof_part1_A_inter_B :
  A x ∧ B x ↔ 2 < x ∧ x ≤ 5 := sorry

theorem proof_part1_complement_B_union_A :
  (x ≤ 2 ∨ A x) ↔ x ≤ 5 := sorry

theorem proof_part2_C_subset_A_range_a :
  (∀ x, (C a x → A x)) ↔ (1 ≤ a ∨ a > 2) := sorry

end

end proof_part1_A_inter_B_proof_part1_complement_B_union_A_proof_part2_C_subset_A_range_a_l588_588388


namespace smallest_period_and_monotonic_intervals_minimum_alpha_for_g_odd_l588_588782

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sin x + sqrt 3 * cos x) - sqrt 3 / 2

theorem smallest_period_and_monotonic_intervals :
  (∀ x : ℝ, f (x + π) = f x) ∧ 
  (∀ k : ℤ, ∀ x : ℝ, k * π - (5 * π / 12) ≤ x ∧ x ≤ k * π + (π / 12) → 
    f' x > 0) :=
sorry

theorem minimum_alpha_for_g_odd (α : ℝ) 
  (hα : α > 0)
  (hodd : ∀ x : ℝ, f (x + α) = -f (x - α)) :
  α = π / 3 :=
sorry

end smallest_period_and_monotonic_intervals_minimum_alpha_for_g_odd_l588_588782


namespace value_of_a_l588_588766

noncomputable def f (a : ℝ) (x : ℝ) := (x-1)*(x^2 - 3*x + a)

-- Define the condition that 1 is not a critical point
def not_critical (a : ℝ) : Prop := f a 1 ≠ 0

theorem value_of_a (a : ℝ) (h : not_critical a) : a = 2 := 
sorry

end value_of_a_l588_588766


namespace find_x_l588_588718

theorem find_x (x : ℝ) (y : ℝ) : (∀ y, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 := by
  intros h
  -- At this point, you would include the necessary proof steps, but for now we skip it.
  sorry

end find_x_l588_588718


namespace jason_cooks_dinner_in_73_minutes_l588_588948

-- Definitions for the given problem conditions
def T_initial : ℕ := 41
def T_final : ℕ := 212
def temp_increase_per_minute : ℕ := 3
def pasta_cook_time : ℕ := 12
def mix_time_factor : ℕ := 1 / 3

-- Proof statement
theorem jason_cooks_dinner_in_73_minutes :
  let temp_diff := T_final - T_initial,
      boil_time := temp_diff / temp_increase_per_minute,
      pasta_cook_time := 12,
      mix_time := pasta_cook_time / 3,
      total_time := boil_time + pasta_cook_time + mix_time
  in total_time = 73 := 
by {
  sorry
}

end jason_cooks_dinner_in_73_minutes_l588_588948


namespace initial_bacteria_count_l588_588529

theorem initial_bacteria_count 
  (double_every_30_seconds : ∀ n : ℕ, n * 2^(240 / 30) = 262144) : 
  ∃ n : ℕ, n = 1024 :=
by
  -- Define the initial number of bacteria.
  let n := 262144 / (2^8)
  -- Assert that the initial number is 1024.
  use n
  -- To skip the proof.
  sorry

end initial_bacteria_count_l588_588529


namespace pigeon_problem_l588_588623

theorem pigeon_problem (x y : ℕ) :
  (1 / 6 : ℝ) * (x + y) = y - 1 ∧ x - 1 = y + 1 → x = 4 ∧ y = 2 :=
by
  sorry

end pigeon_problem_l588_588623


namespace parabola_min_area_l588_588378

-- Definition of the parabola C with vertex at the origin and focus on the positive y-axis
-- (Conditions 1 and 2)
def parabola_eq (x y : ℝ) : Prop := x^2 = 6 * y

-- Line l defined by mx + y - 3/2 = 0 (Condition 3)
def line_eq (m x y : ℝ) : Prop := m * x + y - 3 / 2 = 0

-- Formal statement combining all conditions to prove the equivalent Lean statement
theorem parabola_min_area :
  (∀ x y : ℝ, parabola_eq x y ↔ x^2 = 6 * y) ∧
  (∀ m x y : ℝ, line_eq m x y ↔ m * x + y - 3 / 2 = 0) →
  (parabola_eq 0 0) ∧ (∃ y > 0, parabola_eq 0 y ∧ line_eq 0 0 (y/2) ∧ y = 3 / 2) ∧
  ∀ A B P : ℝ, line_eq 0 A B ∧ line_eq 0 B P ∧ A^2 + B^2 > 0 → 
  ∃ min_S : ℝ, min_S = 9 :=
by
  sorry

end parabola_min_area_l588_588378


namespace possible_values_l588_588463

theorem possible_values (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  ∃ S : Set ℝ, S = {x : ℝ | 4 ≤ x} ∧ (1 / a + 1 / b) ∈ S :=
by
  sorry

end possible_values_l588_588463


namespace smallest_solution_l588_588708

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end smallest_solution_l588_588708


namespace third_group_people_count_l588_588296

theorem third_group_people_count : 
  ∃ x, (x * 2 + 6) + (6 * 1) = 20 → 
        6 + x = 8 := 
  begin
    sorry
  end

end third_group_people_count_l588_588296


namespace sum_of_n_values_l588_588226

theorem sum_of_n_values : ∃ n1 n2 : ℚ, (abs (3 * n1 - 4) = 5) ∧ (abs (3 * n2 - 4) = 5) ∧ n1 + n2 = 8 / 3 :=
by
  sorry

end sum_of_n_values_l588_588226


namespace tan_product_identity_l588_588012

theorem tan_product_identity (m : ℕ) :
  ((1 + tan (1 : ℝ)) * (1 + tan (2 : ℝ)) * (1 + tan (3 : ℝ)) * (1 + tan (4 : ℝ)) *
   (1 + tan (5 : ℝ)) * (1 + tan (6 : ℝ)) * (1 + tan (7 : ℝ)) * (1 + tan (8 : ℝ)) *
   (1 + tan (9 : ℝ)) * (1 + tan (10 : ℝ)) * (1 + tan (11 : ℝ)) * (1 + tan (12 : ℝ)) *
   (1 + tan (13 : ℝ)) * (1 + tan (14 : ℝ)) * (1 + tan (15 : ℝ)) * (1 + tan (16 : ℝ)) *
   (1 + tan (17 : ℝ)) * (1 + tan (18 : ℝ)) * (1 + tan (19 : ℝ)) * (1 + tan (20 : ℝ)) *
   (1 + tan (21 : ℝ)) * (1 + tan (22 : ℝ)) * (1 + tan (23 : ℝ)) * (1 + tan (24 : ℝ)) *
   (1 + tan (25 : ℝ)) * (1 + tan (26 : ℝ)) * (1 + tan (27 : ℝ)) * (1 + tan (28 : ℝ)) *
   (1 + tan (29 : ℝ)) * (1 + tan (30 : ℝ))) = 2^16 := 
sorry

end tan_product_identity_l588_588012


namespace general_term_sequence_l588_588386

theorem general_term_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, 1 ≤ n → (finset.range n).sum (λ k, (k + 1) * a (k + 1)) = n + 1) :
  (∀ n : ℕ, n = 1 → a n = 2) ∧ (∀ n : ℕ, 2 ≤ n → a n = 1 / n) :=
by
  sorry

end general_term_sequence_l588_588386


namespace parallel_lines_minimum_distance_l588_588436

theorem parallel_lines_minimum_distance :
  ∀ (m n : ℝ) (k : ℝ), 
  k = 2 ∧ ∀ (L1 L2 : ℝ → ℝ), -- we define L1 and L2 as functions
  (L1 = λ y => 2 * y + 3) ∧ (L2 = λ y => k * y - 1) ∧ 
  ((L1 n = m) ∧ (L2 (n + k) = m + 2)) → 
  dist (m, n) (m + 2, n + 2) = 2 * Real.sqrt 2 := 
sorry

end parallel_lines_minimum_distance_l588_588436


namespace negate_exactly_one_even_l588_588489

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even (a b c : ℕ) :
  ¬ ((is_even a ∧ is_odd b ∧ is_odd c) ∨ (is_odd a ∧ is_even b ∧ is_odd c) ∨ (is_odd a ∧ is_odd b ∧ is_even c)) ↔ 
  ((is_odd a ∧ is_odd b ∧ is_odd c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c)) :=
sorry

end negate_exactly_one_even_l588_588489


namespace overtaking_time_is_280_seconds_l588_588182

noncomputable def speed_in_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * (1000 / 3600)

def length_train_a : ℝ := 400
def speed_train_a : ℝ := speed_in_mps 63
def length_train_b : ℝ := 300
def speed_train_b : ℝ := speed_in_mps 54
def speed_man : ℝ := speed_in_mps 3

def relative_speed_train_a_b : ℝ := speed_train_a - speed_train_b
def relative_speed_train_a_man : ℝ := speed_train_a - speed_man

def distance_train_a_b : ℝ := length_train_a + length_train_b
def distance_train_a_man : ℝ := length_train_a

def time_to_overtake_train_b : ℝ :=
  distance_train_a_b / relative_speed_train_a_b

def time_to_overtake_man : ℝ :=
  distance_train_a_man / relative_speed_train_a_man

def time_to_completely_overtake : ℝ :=
  max time_to_overtake_train_b time_to_overtake_man

theorem overtaking_time_is_280_seconds : time_to_completely_overtake = 280 := 
  sorry

end overtaking_time_is_280_seconds_l588_588182


namespace super_ball_distance_traveled_l588_588256

theorem super_ball_distance_traveled :
    let d0 := 20 in
    let ratio := (2 : ℚ) / 3 in
    let d1 := d0 * ratio in
    let d2 := d1 * ratio in
    let d3 := d2 * ratio in
    let d4 := d3 * ratio in
    let total_distance := d0 + d1 + d1 + d2 + d2 + d3 + d3 + d4 in
    total_distance.to_real ≈ 80 :=
by
  -- Definitions of each distance step
  let d0 := 20 : ℚ
  let ratio := (2 : ℚ) / 3
  let d1 := d0 * ratio
  let d2 := d1 * ratio
  let d3 := d2 * ratio
  let d4 := d3 * ratio
  
  -- Calculate total distance
  let total_distance := d0 + d1 + d1 + d2 + d2 + d3 + d3 + d4

  -- Show total distance is approximately 80
  sorry

end super_ball_distance_traveled_l588_588256


namespace distance_between_students_l588_588416

theorem distance_between_students (speed1 speed2 time : ℝ) (h_speed1 : speed1 = 6) (h_speed2 : speed2 = 9) (h_time : time = 4) :
  let distance1 := speed1 * time,
      distance2 := speed2 * time,
      total_distance := distance1 + distance2
  in total_distance = 60 :=
by
  sorry

end distance_between_students_l588_588416


namespace number_of_solutions_l588_588700

open Function

theorem number_of_solutions (c : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f(x + f(y)) = x + y + c) ↔ (c = 0 ∧ ∃! f : ℝ → ℝ, ∀ x : ℝ, f(x) = x) :=
by
  sorry

end number_of_solutions_l588_588700


namespace triangle_area_parabola_hyperbola_l588_588333

theorem triangle_area_parabola_hyperbola :
  let axis_parabola := (λ x, x = -3)
  let asymptote1_hyperbola := (λ (x y : ℝ), y = (√3 / 3) * x)
  let asymptote2_hyperbola := (λ (x y : ℝ), y = -(√3 / 3) * x)
  ∃ (area : ℝ), 
    (axis_parabola (-3)) ∧ 
    (asymptote1_hyperbola (3*√3) (√3)) ∧ 
    (asymptote2_hyperbola (-3*√3) (√3)) ∧ 
    area = 3 * √3 := 
begin
  sorry,
end

end triangle_area_parabola_hyperbola_l588_588333


namespace num_of_lines_through_3_points_eq_365_l588_588098

-- Define the ranges and properties of our marked points
def points : List (ℕ × ℕ) :=
  [(x, y) | x <- [0, 1, 2], y <- [0, 1, 2, ..., 26]]

-- Helper function for even check
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Helper function for odd check
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- Main theorem to be proven
theorem num_of_lines_through_3_points_eq_365 :
  ∃! l, ∃ a b c : ℕ,
  0 ≤ a ∧ a ≤ 26 ∧ 0 ≤ b ∧ b ≤ 26 ∧ 0 ≤ c ∧ c ≤ 26 ∧
  (is_even a ∧ is_even c ∨ is_odd a ∧ is_odd c) ∧
  b = (a + c) / 2 ∧
  l = 365 := sorry

end num_of_lines_through_3_points_eq_365_l588_588098


namespace investment_ratio_l588_588159

theorem investment_ratio (P Q : ℝ) (h1 : (P * 5) / (Q * 9) = 7 / 9) : P / Q = 7 / 5 :=
by sorry

end investment_ratio_l588_588159


namespace spinning_class_frequency_l588_588066

/--
We define the conditions given in the problem:
- duration of each class in hours,
- calorie burn rate per minute,
- total calories burned per week.
We then state that the number of classes James attends per week is equal to 3.
-/
def class_duration_hours : ℝ := 1.5
def calories_per_minute : ℝ := 7
def total_calories_per_week : ℝ := 1890

theorem spinning_class_frequency :
  total_calories_per_week / (class_duration_hours * 60 * calories_per_minute) = 3 :=
by
  sorry

end spinning_class_frequency_l588_588066


namespace sum_of_consecutive_integers_product_l588_588552

noncomputable def consecutive_integers_sum (n m k : ℤ) : ℤ :=
  n + m + k

theorem sum_of_consecutive_integers_product (n m k : ℤ)
  (h1 : n = m - 1)
  (h2 : k = m + 1)
  (h3 : n * m * k = 990) :
  consecutive_integers_sum n m k = 30 :=
by
  sorry

end sum_of_consecutive_integers_product_l588_588552


namespace candy_bar_cost_correct_l588_588308

-- Definitions based on conditions
def candy_bar_cost := 3
def chocolate_cost := candy_bar_cost + 5
def total_cost := chocolate_cost + candy_bar_cost

-- Assertion to be proved
theorem candy_bar_cost_correct :
  total_cost = 11 → candy_bar_cost = 3 :=
by
  intro h
  simp [total_cost, chocolate_cost, candy_bar_cost] at h
  sorry

end candy_bar_cost_correct_l588_588308


namespace chords_intersect_probability_l588_588568

theorem chords_intersect_probability
  (n : ℕ) (h : n = 1996)
  (A B C D : Fin n.succ) :
  random_selection (A, B, C, D) →
  probability (chords_intersect (A, B, C, D)) = 1 / 4 := by
s257572 : n = 1996 := h
s152708 : random_selection (A, B, C, D) := sorry
s211478 : probability (chords_intersect (A, B, C, D)) = 1 / 4 := sorry

  sorry

end chords_intersect_probability_l588_588568


namespace computeSumFormula_l588_588969

noncomputable def computeSum (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x ^ (3^n) - x ^ -(3^n))

theorem computeSumFormula (x : ℝ) (h : 1 < x) : computeSum x h = 1 / (x - 1) :=
  sorry

end computeSumFormula_l588_588969


namespace main_proof_l588_588432

noncomputable def ellipse_eq : Prop :=
  ∃ (a b c : ℝ) (h1 : a > b) (h2 : b > 0),
  ellipse_eq = (M : ℝ → ℝ → Prop) (foci_eq : ℝ → ℝ → Prop) 
  (vertices_eq : list (ℝ × ℝ)) (F1_eq : ℝ × ℝ) (F2_eq : ℝ × ℝ) (B1_eq : ℝ × ℝ) (B2_eq : ℝ × ℝ),
  M (x, y) ↔ (x, y) ∈ ellipse_eq (∑ (x^2/a^2) + (y^2/b^2) = 1 (a := sqrt(b^2 + c^2) b := b c := c))
  ∧ foci_eq (-c, 0) (c, 0) (h3 : c^2 = a^2 - b^2)
  ∧ vertices_eq = [(0, b), (0, -b)]
  ∧ F1_eq = (-c, 0)
  ∧ F2_eq = (c, 0) 
  ∧ (1, sqrt(2) / 2) ∈ M.

noncomputable def max_area_triangle : Prop :=
  let l : ℝ → ℝ := fun slope => (fun x => slope * (x + 2)) in
  let F2 : (ℝ × ℝ) := (1, 0) in
  ∃ k : ℝ, line_intersection F2 (1 / 2) (2 - k^2 < 2), 
  triangle_area F2 ≤ 3 * sqrt(2) / 4

theorem main_proof : ellipse_eq ∧ max_area_triangle :=
  by
  sorry

end main_proof_l588_588432


namespace log_pos_if_exp_diff_l588_588874

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588874


namespace probability_multiple_of_3_l588_588173

theorem probability_multiple_of_3:
  let tickets := Finset.range 20 |>.map (λ x => x + 1)
  let multiples_of_3 := tickets.filter (λ x => x % 3 = 0)
  let probability := (multiples_of_3.card : ℚ) / tickets.card
  probability = 3 / 10 := by
sorry

end probability_multiple_of_3_l588_588173


namespace length_of_segment_eq_ten_l588_588588

theorem length_of_segment_eq_ten (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  let y1 := real.cbrt 27 + 5,
      y2 := real.cbrt 27 - 5
  in abs (y1 - y2) = 10 := 
by
  sorry

end length_of_segment_eq_ten_l588_588588


namespace no_values_t_for_expression_l588_588313

theorem no_values_t_for_expression (t : ℂ) : ¬ (sqrt(49 - t^2) + 7 = 0) := 
sorry

end no_values_t_for_expression_l588_588313


namespace sin_sq_sub_sin_double_l588_588757

open Real

theorem sin_sq_sub_sin_double (alpha : ℝ) (h : tan alpha = 1 / 2) : sin alpha ^ 2 - sin (2 * alpha) = -3 / 5 := 
by 
  sorry

end sin_sq_sub_sin_double_l588_588757


namespace general_term_correct_sum_absolute_values_first_30_l588_588433

variable (a : ℕ → ℤ)
variable (a1 : a 1 = -60)
variable (a17 : a 17 = -12)

def general_term (a : ℕ → ℤ) : ℕ → ℤ :=
  λ n => a 1 + (n - 1) * ((a 17 - a 1) / 16)

theorem general_term_correct :
  (general_term a) = λ n, 3 * n - 63 :=
by sorry

theorem sum_absolute_values_first_30 :
  (Finset.sum (Finset.range 30) (λ n => abs (general_term a (n + 1)))) = 765 :=
by sorry

end general_term_correct_sum_absolute_values_first_30_l588_588433


namespace no_rational_roots_of_odd_l588_588120

theorem no_rational_roots_of_odd (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) : ¬ ∃ x : ℚ, x^2 + 2 * m * x + 2 * n = 0 :=
sorry

end no_rational_roots_of_odd_l588_588120


namespace coefficient_of_middle_term_l588_588937

theorem coefficient_of_middle_term (n : ℕ) 
  (h: 2^(n-1) = 1024) : 
  binomial (n-1) 5 = 462 ∧ binomial (n-1) 6 = 462 := by
  have h1 : n = 11 := sorry
  sorry

end coefficient_of_middle_term_l588_588937


namespace total_children_in_class_l588_588421

theorem total_children_in_class {T S N B C : ℕ} 
  (hT : T = 19) 
  (hS : S = 21) 
  (hN : N = 10) 
  (hB : B = 12) 
  (hC : C = (T + S - B) + N) : 
  C = 38 := 
by
  rw [hT, hS, hN, hB] at hC
  rw [←hC]
  norm_num
  sorry

end total_children_in_class_l588_588421


namespace log_y_minus_x_plus_1_pos_l588_588864

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588864


namespace area_of_shaded_region_l588_588206

axiom OA : ℝ := 4
axiom OB : ℝ := 16
axiom OC : ℝ := 12
axiom similarity (EA CB : ℝ) : EA / CB = OA / OB

theorem area_of_shaded_region (DE DC : ℝ) (h_DE : DE = OC - EA)
    (h_DC : DC = 12) (h_EA_CB : EA = 3) :
    (1 / 2) * DE * DC = 54 := by
  sorry

end area_of_shaded_region_l588_588206


namespace magnitude_relationship_l588_588402

noncomputable def a := 0.3^4
noncomputable def b := 4^0.3
noncomputable def c := Real.logBase 0.3 4

theorem magnitude_relationship : b > a ∧ a > c := by
  sorry

end magnitude_relationship_l588_588402


namespace Josie_shopping_time_l588_588107

theorem Josie_shopping_time
  (wait_cart : ℕ := 3)
  (wait_employee : ℕ := 13)
  (wait_stocker : ℕ := 14)
  (wait_line : ℕ := 18)
  (total_shopping_trip : ℕ := 90) :
  total_shopping_trip - (wait_cart + wait_employee + wait_stocker + wait_line) = 42 :=
by
  -- Convert the total shopping trip time from hours to minutes
  have trip_minutes : total_shopping_trip = 90 := rfl
  -- Sum the waiting times
  have waiting_total : wait_cart + wait_employee + wait_stocker + wait_line = 48 := rfl
  -- Subtract the waiting time from the total trip time
  have shopping_time : total_shopping_trip - (wait_cart + wait_employee + wait_stocker + wait_line) = 90 - 48 := rfl
  -- Hence, Josie spent 42 minutes shopping
  rw [trip_minutes,waiting_total,shopping_time]
  exact rfl
  sorry

end Josie_shopping_time_l588_588107


namespace triangle_RZY_area_l588_588047

/-- Given a square XYZ with area 324, with point P on side XW, 
points Q and R are the midpoints of XP and ZP respectively, 
and quadrilateral WPRQ has area 54, prove that the area of triangle RZY is 45. --/
theorem triangle_RZY_area (area_XYZW : ℝ) (area_WPRQ : ℝ)
  (P_on_XW : P ∈ line_segment X W) 
  (Q_mid_XP : midpoint Q X P) 
  (R_mid_ZP : midpoint R Z P) 
  (square_XYZW : is_square X Y Z W) 
  (area_XYZW_equals_324 : area_XYZW = 324)
  (area_WPRQ_equals_54 :area_WPRQ = 54) :
  area (triangle R Z Y) = 45 := 
sorry

end triangle_RZY_area_l588_588047


namespace sequence_product_perfect_square_l588_588641

theorem sequence_product_perfect_square (seq : Fin 20 → ℕ) (h_distinct : Function.Injective seq)
  (h_first : seq 0 = 42)
  (h_perfect_square_prod : ∀ i : Fin 19, ∃ k : ℕ, (seq i) * (seq ⟨i + 1, by decide⟩) = k ^ 2) :
  ∃ i : Fin 20, seq i > 16000 := 
sorry

end sequence_product_perfect_square_l588_588641


namespace cubics_inequality_l588_588351

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 :=
sorry

end cubics_inequality_l588_588351


namespace area_of_shaded_region_l588_588208

axiom OA : ℝ := 4
axiom OB : ℝ := 16
axiom OC : ℝ := 12
axiom similarity (EA CB : ℝ) : EA / CB = OA / OB

theorem area_of_shaded_region (DE DC : ℝ) (h_DE : DE = OC - EA)
    (h_DC : DC = 12) (h_EA_CB : EA = 3) :
    (1 / 2) * DE * DC = 54 := by
  sorry

end area_of_shaded_region_l588_588208


namespace sum_of_extreme_values_l588_588162

theorem sum_of_extreme_values : 
  let f := λ x : ℝ, 2 * x^3 - 3 * x^2 - 12 * x + 5 in
  let a : ℝ := 0 in
  let b : ℝ := 3 in
  (∀ x ∈ [a, b], exists (ymax ymin : ℝ), 
    (∀ y ∈ [a, b], f y ≤ ymax) ∧ 
    (∀ y ∈ [a, b], f y ≥ ymin) ∧
    ymax + ymin = -10) :=
sorry

end sum_of_extreme_values_l588_588162


namespace total_votes_l588_588045

theorem total_votes (V : ℝ) 
  (winning_votes : V * 0.70)
  (losing_votes : V * 0.30)
  (majority : winning_votes - losing_votes = 192) : 
  V = 480 := 
by
  sorry

end total_votes_l588_588045


namespace part_a_equally_likely_part_b_more_even_l588_588992

-- Conditions for Part (a)
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
-- The multiplication button is broken
def multiplication_broken : Prop := True
-- Initial display is 0
def initial_display : ℕ := 0
-- The event that the result is an odd number
def result_odd (n : ℕ) : Prop := n % 2 = 1

-- Part (a) Theorem
theorem part_a_equally_likely : 
  ( ∀ seq : List ℕ, (seq.length > 0) → 
    (multiplication_broken) → 
    ( ∃ p : ℝ, 0 <= p ∧ p <= 1 ∧ result_odd (seq.foldl (λ acc x => acc + x) initial_display))
  ) = (1 / 2) := 
sorry

-- Conditions for Part (b)
def multiplication_fixed : Prop := True
-- Probabilities and recurrence relation
noncomputable def p_n (n : ℕ) [fact (n ≥ 1)] : ℝ := 
if n = 1 then (1 / 2) else (1 / 4) + (1 / 4) * (p_n (n - 1))

-- Part (b) Theorem
theorem part_b_more_even : 
  ( ∀ n, 1 <= n → p_n n < (1 / 2)) := 
sorry

end part_a_equally_likely_part_b_more_even_l588_588992


namespace factorial_sum_mod_21_remainder_l588_588412

theorem factorial_sum_mod_21_remainder :
  let S := (Finset.range 51).sum (λ n, n.fact)
  S % 21 = 12 :=
by
  sorry

end factorial_sum_mod_21_remainder_l588_588412


namespace minimize_sum_of_reciprocals_l588_588434

theorem minimize_sum_of_reciprocals (a b : ℕ) (h : 4 * a + b = 6) : 
  a = 1 ∧ b = 2 ∨ a = 2 ∧ b = 1 :=
by
  sorry

end minimize_sum_of_reciprocals_l588_588434


namespace bowling_ball_weight_l588_588138

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = 20 := by
  sorry

end bowling_ball_weight_l588_588138


namespace exist_apollonius_intersection_l588_588306

variables {Point Circle : Type}
variables (O1 O2 O3 : Point) (r1 r2 r3 : ℝ)
variables (d O1P O2P O3P : ℝ)

def distance (P Q : Point) : ℝ := d -- Assume we have a function for distance

def on_apollonius_circle 
  (O1 O2 : Point) (r1 r2 : ℝ) (P : Point) : Prop := 
  distance O1 P / distance O2 P = r1 / r2

theorem exist_apollonius_intersection
  (O1 O2 O3 : Point) (r1 r2 r3 : ℝ):
  ∃ (R : Point), 
  on_apollonius_circle O1 O2 r1 r2 R ∧
  on_apollonius_circle O2 O3 r2 r3 R ∧
  on_apollonius_circle O1 O3 r1 r3 R := 
sorry

end exist_apollonius_intersection_l588_588306


namespace find_lambda_l588_588962

-- Define the hyperbola, points, and conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1
def asymptote1 (x y : ℝ) : Prop := y = 2 * x
def asymptote2 (x y : ℝ) : Prop := y = -2 * x
def distance_to_asymptote (x y : ℝ) : Prop := abs (2 * x - y) / sqrt 5 = 2

def point_P (x y : ℝ) : Prop := hyperbola x y ∧ distance_to_asymptote x y
def point_Q (x y : ℝ) : Prop := asymptote2 x y

def vec (a b c d : ℝ) := ((c - a), (d - b))

def collinear (a b c d e f : ℝ) (λ : ℝ) : Prop := 
vec a b c d = λ • vec d e a b

-- Prove that λ = 4 given the conditions
theorem find_lambda :
  ∃ (x_P y_P x_Q y_Q λ: ℝ), 
  point_P x_P y_P ∧ 
  point_Q x_Q y_Q ∧
  collinear (sqrt 5) 0 x_P y_P x_Q y_Q λ ∧ λ = 4 := 
sorry

end find_lambda_l588_588962


namespace log_of_y_sub_x_plus_one_positive_l588_588897

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588897


namespace area_of_shaded_region_l588_588207

axiom OA : ℝ := 4
axiom OB : ℝ := 16
axiom OC : ℝ := 12
axiom similarity (EA CB : ℝ) : EA / CB = OA / OB

theorem area_of_shaded_region (DE DC : ℝ) (h_DE : DE = OC - EA)
    (h_DC : DC = 12) (h_EA_CB : EA = 3) :
    (1 / 2) * DE * DC = 54 := by
  sorry

end area_of_shaded_region_l588_588207


namespace degrees_of_remainder_is_correct_l588_588228

noncomputable def degrees_of_remainder (P D : Polynomial ℤ) : Finset ℕ :=
  if D.degree = 3 then {0, 1, 2} else ∅

theorem degrees_of_remainder_is_correct
(P : Polynomial ℤ) :
  degrees_of_remainder P (Polynomial.C 3 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X^2 + Polynomial.C 2 * Polynomial.X - Polynomial.C 4) = {0, 1, 2} :=
by
  -- Proof omitted
  sorry

end degrees_of_remainder_is_correct_l588_588228


namespace exists_equilateral_triangle_points_l588_588071

variable {n : ℕ}
variable {B : Set (Fin n → ℤ)}
variable [DecidableEq (Fin n → ℤ)]

def is_equilateral_triangle (a b c : Fin n → ℤ) : Prop :=
  dist a b = dist b c ∧ dist b c = dist c a ∧ dist a b = dist c a

theorem exists_equilateral_triangle_points
  (hB_card : B.card > 2^(n+1) / n)
  (hB_coords : ∀ p ∈ B, ∃ (a : (Fin n → ℤ)), ∀ i, p i = 1 ∨ p i = -1)
  (h_n_ge_3 : 3 ≤ n) :
  ∃ a b c ∈ B, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ is_equilateral_triangle a b c :=
sorry

end exists_equilateral_triangle_points_l588_588071


namespace max_blue_points_l588_588754

/-- 
Given several unit circles on a plane, each center is colored blue. 
We mark some points on the circumferences of the circles in red, 
such that exactly 2 red points are placed on each circle's circumference. 
The total number of colored points (both blue and red) is 25. 
-/
theorem max_blue_points (H : ∀ (n : ℕ), 2 * (n * (n - 1) / 2) ≤ 25) : 
  ∃ m : ℕ, m + 2 * (m * (m - 1) / 2) = 25 ∧ m = 20 :=
begin
  sorry
end

end max_blue_points_l588_588754


namespace area_of_CDE_is_54_l588_588214

-- Define points O, A, B, C, D, and E using coordinates.
def point := (ℝ × ℝ)
def O : point := (0, 0)
def A : point := (4, 0)
def B : point := (16, 0)
def C : point := (16, 12)
def D : point := (4, 12)
def E : point := (4, 3)  -- Midpoint derived from problem's similarity conditions

-- Define the line segment lengths based on the points.
def length (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Define DE and DC
noncomputable def DE : ℝ := 9
noncomputable def DC : ℝ := 12

-- Area of triangle CDE
noncomputable def area_CDE : ℝ := (1 / 2) * DE * DC

-- Theorem statement
theorem area_of_CDE_is_54 : area_CDE = 54 := by
  sorry

end area_of_CDE_is_54_l588_588214


namespace smallest_solution_of_abs_eq_l588_588707

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end smallest_solution_of_abs_eq_l588_588707


namespace xyz_equal_six_l588_588974

noncomputable def complex_numbers : Type := Complex

theorem xyz_equal_six 
  (a b c x y z : complex_numbers)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (cond1 : a = (b + c) / (x - 3))
  (cond2 : b = (a + c) / (y - 3))
  (cond3 : c = (a + b) / (z - 3))
  (cond4 : x * y + x * z + y * z = 10)
  (cond5 : x + y + z = 6) : 
  x * y * z = 6 :=
by
  sorry

end xyz_equal_six_l588_588974


namespace total_bill_correct_l588_588667

section
variable (colored_cost white_cost : ℕ) (total_copies colored_copies white_copies : ℕ)

-- Definitions based on the conditions
def cost_per_colored_copy := 10
def cost_per_white_copy := 5
def total_num_copies := 400
def num_colored_copies := 50
def num_white_copies := total_num_copies - num_colored_copies

-- Definitions of costs
def cost_colored_copies := num_colored_copies * cost_per_colored_copy
def cost_white_copies := num_white_copies * cost_per_white_copy
def total_bill := cost_colored_copies + cost_white_copies

-- Theorem to state the total bill
theorem total_bill_correct : 
  total_bill = 2250 := 
begin 
  -- Ensure that we align with the right units, as Lean deals with cents, not dollars.
  let total_bill_in_dollars := total_bill / 100,
  have : total_bill_in_dollars = 22.50 := by sorry,
  
  -- Ensure final total bill correctness
  simp [total_bill_in_dollars] at *,
  have : total_bill = 2250 := by sorry,
  assumption,
  sorry,
end

end total_bill_correct_l588_588667


namespace integer_solutions_l588_588331

theorem integer_solutions :
  ∀ (m n : ℤ), (m^3 - n^3 = 2 * m * n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2)) :=
by
  intros m n
  sorry

end integer_solutions_l588_588331


namespace area_of_CDE_is_54_l588_588216

-- Define points O, A, B, C, D, and E using coordinates.
def point := (ℝ × ℝ)
def O : point := (0, 0)
def A : point := (4, 0)
def B : point := (16, 0)
def C : point := (16, 12)
def D : point := (4, 12)
def E : point := (4, 3)  -- Midpoint derived from problem's similarity conditions

-- Define the line segment lengths based on the points.
def length (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Define DE and DC
noncomputable def DE : ℝ := 9
noncomputable def DC : ℝ := 12

-- Area of triangle CDE
noncomputable def area_CDE : ℝ := (1 / 2) * DE * DC

-- Theorem statement
theorem area_of_CDE_is_54 : area_CDE = 54 := by
  sorry

end area_of_CDE_is_54_l588_588216


namespace log_of_y_sub_x_plus_one_positive_l588_588900

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588900


namespace average_fish_per_person_l588_588912

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l588_588912


namespace problem_solution_l588_588911

theorem problem_solution (x y : ℤ) (h1 : 3^x * 4^y = 531441) (h2 : x - y = 12) : x = 12 := 
by
  sorry

end problem_solution_l588_588911


namespace super_ball_distance_traveled_l588_588255

theorem super_ball_distance_traveled :
    let d0 := 20 in
    let ratio := (2 : ℚ) / 3 in
    let d1 := d0 * ratio in
    let d2 := d1 * ratio in
    let d3 := d2 * ratio in
    let d4 := d3 * ratio in
    let total_distance := d0 + d1 + d1 + d2 + d2 + d3 + d3 + d4 in
    total_distance.to_real ≈ 80 :=
by
  -- Definitions of each distance step
  let d0 := 20 : ℚ
  let ratio := (2 : ℚ) / 3
  let d1 := d0 * ratio
  let d2 := d1 * ratio
  let d3 := d2 * ratio
  let d4 := d3 * ratio
  
  -- Calculate total distance
  let total_distance := d0 + d1 + d1 + d2 + d2 + d3 + d3 + d4

  -- Show total distance is approximately 80
  sorry

end super_ball_distance_traveled_l588_588255


namespace inequality_implies_log_pos_l588_588820

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588820


namespace problem_1_problem_2_l588_588680

-- First problem: Find possible values of x for the complete union set M
theorem problem_1 (M : Set ℕ) (hM : M = {1, x, 3, 4, 5, 6}) :
  ∃ x ∈ {7, 9, 11}, ∀ (A B C : Set ℕ),
    A ∪ B ∪ C = M ∧ A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧ 
    (∀ k, a_k + b_k = c_k) ∧ ∀ i j, i < j → c_i < c_j → 
    True := 
sorry

-- Second problem: Find the set C with the smallest product elements in a complete union set M
theorem problem_2 (M : Set ℕ) (hM : M = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) :
  ∃ C : Set ℕ, C = {6, 10, 11, 12} ∧ 
    ∀ C' : Set ℕ,
    C' = {c | c ∈ M ∧ ∀ i j, i < j → c_i < c_j} ∧ 
    (∏ x in C', x) ≥ (∏ x in C, x) :=
sorry

end problem_1_problem_2_l588_588680


namespace triangle_PQR_area_l588_588218

structure Point :=
(x : ℤ)
(y : ℤ)

def triangle_area (P Q R : Point) : ℚ := 
  1/2 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

theorem triangle_PQR_area :
  let P := Point.mk 2 3 in
  let Q := Point.mk (-4) 3 in
  let R := Point.mk (-1) (-2) in
  triangle_area P Q R = 15 :=
by
  -- sorry to skip the proof
  sorry

end triangle_PQR_area_l588_588218


namespace randy_mango_trees_l588_588506

theorem randy_mango_trees (M C : ℕ) 
  (h1 : C = M / 2 - 5) 
  (h2 : M + C = 85) : 
  M = 60 := 
sorry

end randy_mango_trees_l588_588506


namespace length_of_segment_eq_ten_l588_588589

theorem length_of_segment_eq_ten (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  let y1 := real.cbrt 27 + 5,
      y2 := real.cbrt 27 - 5
  in abs (y1 - y2) = 10 := 
by
  sorry

end length_of_segment_eq_ten_l588_588589


namespace union_of_M_and_N_l588_588978

noncomputable def M : Set ℝ := { x | x^2 - 5 * x + 6 < 0 }
noncomputable def N : Set ℝ := { x | 3^x ≥ 27 }

theorem union_of_M_and_N : M ∪ N = { x : ℝ | x > 2 } := by 
  sorry

end union_of_M_and_N_l588_588978


namespace area_of_shaded_region_l588_588193

-- Define points F, G, H, I, J with their coordinates
def F := (0, 0)
def G := (4, 0)
def H := (16, 0)
def I := (16, 12)
def J := (4, 3)

-- Define the similarity condition
def similar_triangles_JFG_IHG : Prop :=
  (triangle.similar F G J) (triangle.similar H G I)

-- The lengths of the segments based on problem conditions
def length_HG := 12
def length_JG := 3
def length_IG := 9

-- Area calculation of triangle IJG
def area_IJG := (1/2 * length_IG * length_JG).toReal

-- Final proof statement
theorem area_of_shaded_region :
  similar_triangles_JFG_IHG →
  length_HG = 12 →
  length_JG = length_HG/4 →
  length_IG = length_HG - length_JG →
  real.floor (area_IJG + 0.5) = 14 :=
by
  intros h_sim h_HG h_JG h_IG
  sorry

end area_of_shaded_region_l588_588193


namespace log_y_minus_x_plus_1_pos_l588_588865

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588865


namespace sequence_sum_formula_l588_588024

open Nat

def sequence_term (n : ℕ) : ℕ := 2^n + 2*n - 1

def sequence_sum (n : ℕ) : ℕ := ∑ k in range (n + 1), sequence_term k

theorem sequence_sum_formula (n : ℕ) : sequence_sum n = 2^(n+1) + n^2 - 2 := by
  sorry

end sequence_sum_formula_l588_588024


namespace min_value_of_f_l588_588153

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

-- Statement of the problem
theorem min_value_of_f :
  ∃ x ∈ Icc (-1 : ℝ) 1, (∀ y ∈ Icc (-1 : ℝ) 1, f y ≥ f x) ∧ f x = -2 :=
sorry

end min_value_of_f_l588_588153


namespace product_of_roots_l588_588460

theorem product_of_roots :
  let Q: Polynomial ℚ := Polynomial.Cubic 1 0 (-15) (-50)
  ∃ r1 r2 r3 : ℚ, (Q.eval r1 = 0 ∧ Q.eval r2 = 0 ∧ Q.eval r3 = 0) ∧ r1 * r2 * r3 = -50 :=
by
  let r := (Real.cbrt 5) + (Real.cbrt 25)
  have root_of_Q : is_root (Q : Polynomial ℚ) (r : ℚ) := sorry
  have product_of_roots_by_vieta : Q.roots.product = -50 := sorry
  exact ⟨r, sorry, sorry, product_of_roots_by_vieta⟩

end product_of_roots_l588_588460


namespace value_of_X_l588_588015

def M := 2007 / 3
def N := M / 3
def X := M - N

theorem value_of_X : X = 446 := by
  sorry

end value_of_X_l588_588015


namespace cookies_on_ninth_plate_l588_588164

-- Define the geometric sequence
def cookies_on_plate (n : ℕ) : ℕ :=
  2 * 2^(n - 1)

-- State the theorem
theorem cookies_on_ninth_plate : cookies_on_plate 9 = 512 :=
by
  sorry

end cookies_on_ninth_plate_l588_588164


namespace segment_length_abs_cubed_root_l588_588595

theorem segment_length_abs_cubed_root (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  ∃ a b : ℝ, a = 3 + 5 ∧ b = 3 - 5 ∧ (b - a).abs = 10 :=
by {
  have h1 : real.cbrt 27 = 3 := by norm_num,
  rw h1 at h,
  have h2 : |x - 3| = 5 := h,
  use [8, -2],
  split,
  { refl },
  { split,
    { refl },
    { norm_num } }
}

end segment_length_abs_cubed_root_l588_588595


namespace smallest_integer_min_value_l588_588616

theorem smallest_integer_min_value :
  ∃ (A B C D : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
  B ≠ C ∧ B ≠ D ∧ 
  C ≠ D ∧ 
  (A + B + C + D) = 288 ∧ 
  D = 90 ∧ 
  (A = 21) := 
sorry

end smallest_integer_min_value_l588_588616


namespace smallest_digit_divisible_by_9_l588_588704

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (0 ≤ d ∧ d < 10) ∧ (∃ k : ℕ, 26 + d = 9 * k) ∧ d = 1 :=
by
  sorry

end smallest_digit_divisible_by_9_l588_588704


namespace number_of_lines_through_three_points_l588_588096

theorem number_of_lines_through_three_points : 
  (∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 26) →
  (∃ (count : ℕ), count = 365) :=
begin
  sorry
end

end number_of_lines_through_three_points_l588_588096


namespace find_ellipse_eq_find_max_area_line_eq_l588_588749

-- Given conditions for the ellipse
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1
def a_gt_one (a : ℝ) : Prop := a > 1

-- Coordinates of specific points
def A : ℝ × ℝ := (0, 1)
def point_M : ℝ × ℝ := (-3, 1)
def point_F (c : ℝ) : ℝ × ℝ := (-c, 0)
def point_Tangent : ℝ × ℝ := (0, -1/2)

-- Condition that line AF is tangent to the circle M at point_Tangent
def is_tangent
  (A : ℝ × ℝ) 
  (F : ℝ × ℝ) 
  (center : ℝ × ℝ) 
  (radius : ℝ) 
  (T : ℝ × ℝ) 
  : Prop :=
  let x_c := T.1 - center.1
  let y_c := T.2 - center.2
  let dist := Real.sqrt (x_c^2 + y_c^2)
  dist = radius

-- Theorem 1: Finding the equation of the ellipse
theorem find_ellipse_eq (a : ℝ) (h : a_gt_one a) (c : ℝ) (hc : c = Real.sqrt (a^2 - 1)) :
  (ellipse a = ellipse 3) :=
sorry

-- Theorem 2: Finding the equation of the line when area is maximized
theorem find_max_area_line_eq 
  (a : ℝ) (h : a_gt_one a) (c : ℝ) (hc : c = Real.sqrt (a^2 - 1)) : 
  ∃ k : ℝ, line_eq (0, -1/2) k = (λ x, -1/2 : ℝ → ℝ) :=
sorry

end find_ellipse_eq_find_max_area_line_eq_l588_588749


namespace find_a_l588_588920

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 4 → x^2 + (y - a)^2 = 1 → False) →
  |a| = 1 :=
begin
  intro h,
  unfold abs,
  split_ifs,
  sorry,
end

end find_a_l588_588920


namespace problem_l588_588817

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588817


namespace inequality_ln_positive_l588_588859

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588859


namespace quadratic_inequality_no_real_roots_l588_588923

theorem quadratic_inequality_no_real_roots (a b c : ℝ) (h : a ≠ 0) (h_Δ : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) :=
sorry

end quadratic_inequality_no_real_roots_l588_588923


namespace largest_power_of_2_that_divides_n_l588_588677

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_2_that_divides_n :
  ∃ k : ℕ, 2^k ∣ n ∧ ¬ (2^(k+1) ∣ n) ∧ k = 5 := sorry

end largest_power_of_2_that_divides_n_l588_588677


namespace find_x_for_equation_l588_588254

theorem find_x_for_equation :
  ∃ x : ℝ, 9 - 3 / (1 / x) + 3 = 3 ↔ x = 3 := 
by 
  sorry

end find_x_for_equation_l588_588254


namespace inequality_ln_positive_l588_588861

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588861


namespace count_lines_through_points_l588_588101

-- Define the conditions
def points_on_plane : list (ℤ × ℤ) :=
  [(x, y) | x in [0, 1, 2], y in list.range 27]

-- Definition for the line through exactly three points
def line_through_three_points (a b c : ℤ) : bool :=
  (2 * b = a + c)

#eval (let e := (list.range 14).map (λ n, 2 * n) in
       let o := (list.range 13).map (λ n, 2 * n + 1) in
       2 * (e.length * e.length + o.length * o.length))

def num_lines_through_three_points : ℕ :=
  (let even_vals := list.range' 0 14).map (λ n, 2 * n) in 
  let odd_vals := list.range' 0 13).map (λ n, 2 * n + 1) in 
  2 * (even_vals.length * even_vals.length + odd_vals.length * odd_vals.length)

theorem count_lines_through_points : num_lines_through_three_points = 365 :=
sorry

end count_lines_through_points_l588_588101


namespace leg_length_of_right_triangle_l588_588725

theorem leg_length_of_right_triangle (s : ℝ) 
  (hypotenuse_square_sum : ∑ i in finset.range 4, 1 = s^2) : 
  ∃ (x : ℝ), x = sqrt (1 / 2) :=
by sorry

end leg_length_of_right_triangle_l588_588725


namespace log_y_minus_x_plus_1_gt_0_l588_588846

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588846


namespace three_letter_sets_count_eq_343_l588_588800

theorem three_letter_sets_count_eq_343 :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  count_three_letter_sets(letters) = 343 :=
by
  sorry

def count_three_letter_sets (letters : List Char) : Nat :=
  letters.length ^ 3

end three_letter_sets_count_eq_343_l588_588800


namespace range_of_AB_l588_588790

def f (x : ℝ) := Real.sin x + Real.cos x
def g (x : ℝ) := 2 * Real.cos x

theorem range_of_AB : ∀ t : ℝ, 0 ≤ |(f t) - (g t)| ∧ |(f t) - (g t)| ≤ sqrt 2 :=
by
  intro t
  have h1 : |(f t) - (g t)| = |Real.sin t - Real.cos t| := by
    simp [f, g]
    ring
  rw h1
  have h2 : |Real.sin t - Real.cos t| = |sqrt 2 * Real.sin (t - π/4)| := by
    sorry
  rw h2
  exact abs_le_of_le_sqrt_two (Real.sin (t - π/4))

end range_of_AB_l588_588790


namespace problem_1_problem_2_problem_3_l588_588784

-- Problem (1)
theorem problem_1 (a : ℝ) (f : ℝ → ℝ) (x : ℝ) (hx : a = 1) (hf : f = λ x, a * x - Real.log x) : 
  ∃ x₀ : ℝ, f x₀ = 1 ∧ ∀ y : ℝ, f y ≥ f x₀ := 
sorry

-- Problem (2)
theorem problem_2 (a : ℝ) (f : ℝ → ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ f x = 1) →
  (f = λ x, a * x - Real.log x) → 
  0 ≤ a ∧ a ≤ 1 :=
sorry

-- Problem (3)
theorem problem_3 (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x : ℝ, 1 ≤ x → f x ≥ f (1 / x)) → 
  (f = λ x, a * x - Real.log x) → 
  1 ≤ a :=
sorry

end problem_1_problem_2_problem_3_l588_588784


namespace problem_l588_588808

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588808


namespace arrangements_containing_a_correct_l588_588510

noncomputable def countArrangementsContainingA : Nat :=
let letters := ['s', 'h', 'a', 'd', 'o', 'w']
let choose := Nat.choose 5 3
let permutations := Nat.factorial 4
in choose * permutations

theorem arrangements_containing_a_correct : countArrangementsContainingA = 240 := by
  sorry

end arrangements_containing_a_correct_l588_588510


namespace election_votes_total_l588_588042

theorem election_votes_total 
  (winner_votes : ℕ) (opponent1_votes opponent2_votes opponent3_votes : ℕ)
  (excess1 excess2 excess3 : ℕ)
  (h1 : winner_votes = opponent1_votes + excess1)
  (h2 : winner_votes = opponent2_votes + excess2)
  (h3 : winner_votes = opponent3_votes + excess3)
  (votes_winner : winner_votes = 195)
  (votes_opponent1 : opponent1_votes = 142)
  (votes_opponent2 : opponent2_votes = 116)
  (votes_opponent3 : opponent3_votes = 90)
  (he1 : excess1 = 53)
  (he2 : excess2 = 79)
  (he3 : excess3 = 105) :
  winner_votes + opponent1_votes + opponent2_votes + opponent3_votes = 543 :=
by sorry

end election_votes_total_l588_588042


namespace minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l588_588713

theorem minimal_solution (x : ℝ) (h : x * |x| = 3 * x + 2) : -2 ≤ x :=
begin
  sorry
end

theorem x_eq_neg_two_is_solution : ( -2 : ℝ ) * |-2| = 3 * -2 + 2 :=
begin
  norm_num,
end

/-- The smallest value of x satisfying x|x| = 3x + 2 is -2 -/
theorem smallest_solution : ∃ x : ℝ, x * |x| = 3 * x + 2 ∧ ∀ y : ℝ, y * |y| = 3 * y + 2 → y ≥ x :=
begin
  use -2,
  split,
  { norm_num },
  { intro y,
    sorry }
end

end minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l588_588713


namespace number_of_lines_through_three_points_l588_588097

theorem number_of_lines_through_three_points : 
  (∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 26) →
  (∃ (count : ℕ), count = 365) :=
begin
  sorry
end

end number_of_lines_through_three_points_l588_588097


namespace number_of_solution_pairs_l588_588337

theorem number_of_solution_pairs : 
  (∃ n : ℕ, ∀ (x y : ℕ), 4 * x + 7 * y = 588 → 0 < x → 0 < y ↔ x = 7 * k ∧ y = 84 - 4 * k ∧ k < 21) :=
begin
  sorry
end

end number_of_solution_pairs_l588_588337


namespace mod_remainder_7_10_20_3_20_l588_588702

theorem mod_remainder_7_10_20_3_20 : (7 * 10^20 + 3^20) % 9 = 7 := sorry

end mod_remainder_7_10_20_3_20_l588_588702


namespace cos_value_l588_588732

theorem cos_value (α : ℝ) (h : sin (α - π / 4) = 1 / 3) : cos (α + 5 * π / 4) = 1 / 3 :=
  sorry -- Proof is omitted as requested

end cos_value_l588_588732


namespace sequence_general_term_l588_588055

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → (∀ a: ℕ → ℝ,  a 1 = 4 ∧ (∀ n: ℕ, n > 0 → a (n + 1) = (3 * a n + 2) / (a n + 4))
  → a n = (2 ^ (n - 1) + 5 ^ (n - 1)) / (5 ^ (n - 1) - 2 ^ (n - 1))) :=
by
  sorry

end sequence_general_term_l588_588055


namespace minimum_square_side_length_l588_588482

theorem minimum_square_side_length (s : ℝ) (h1 : s^2 ≥ 625) (h2 : ∃ (t : ℝ), t = s / 2) : s = 25 :=
by
  sorry

end minimum_square_side_length_l588_588482


namespace parametric_eq_to_ordinary_l588_588546

theorem parametric_eq_to_ordinary (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
    let x := abs (Real.sin (θ / 2) + Real.cos (θ / 2))
    let y := 1 + Real.sin θ
    x ^ 2 = y := by sorry

end parametric_eq_to_ordinary_l588_588546


namespace pipes_fill_tank_in_1_5_hours_l588_588499

theorem pipes_fill_tank_in_1_5_hours :
  (1 / 3 + 1 / 9 + 1 / 18 + 1 / 6) = (2 / 3) →
  (1 / (2 / 3)) = (3 / 2) :=
by sorry

end pipes_fill_tank_in_1_5_hours_l588_588499


namespace polynomial_form_l588_588692

theorem polynomial_form (P : ℂ[X]) (h_nonconstant : P.degree > 0)
  (h_roots_abs_one : ∀ z : ℂ, P.is_root z → abs z = 1)
  (h_shifted_roots_abs_one : ∀ z : ℂ, (P - 1).is_root z → abs z = 1) :
  ∃ (λ μ : ℂ) (n : ℕ), P = λ * X^n - C μ ∧ abs λ = abs μ ∧ μ.re = -1/2 :=
sorry

end polynomial_form_l588_588692


namespace min_weight_of_automobile_l588_588270

theorem min_weight_of_automobile (ferry_weight_tons: ℝ) (auto_max_weight: ℝ) 
  (max_autos: ℝ) (ferry_weight_pounds: ℝ) (min_auto_weight: ℝ) : 
  ferry_weight_tons = 50 → 
  auto_max_weight = 3200 → 
  max_autos = 62.5 → 
  ferry_weight_pounds = ferry_weight_tons * 2000 → 
  min_auto_weight = ferry_weight_pounds / max_autos → 
  min_auto_weight = 1600 :=
by
  intros
  sorry

end min_weight_of_automobile_l588_588270


namespace ternary_digits_of_57_l588_588307

def decimal_to_ternary_digits (n : ℕ) : ℕ :=
  let rec count_digits (n : ℕ) (base : ℕ) (digits : ℕ) : ℕ :=
    if n < base then digits else count_digits (n / base) base (digits + 1)
  count_digits n 3 1

theorem ternary_digits_of_57 : decimal_to_ternary_digits 57 = 4 :=
  by
  sorry

end ternary_digits_of_57_l588_588307


namespace quadrilateral_area_ABCDEF_l588_588113

theorem quadrilateral_area_ABCDEF :
  ∀ (A B C D E : Type)
  (AC CD AE : ℝ) 
  (angle_ABC angle_ACD : ℝ),
  angle_ABC = 90 ∧
  angle_ACD = 90 ∧
  AC = 20 ∧
  CD = 30 ∧
  AE = 5 →
  ∃ S : ℝ, S = 360 :=
by
  sorry

end quadrilateral_area_ABCDEF_l588_588113


namespace distinct_pairwise_distances_conditional_l588_588061

-- Define the types for color and points
inductive Color
| Blue
| Red
| Green

def plane := ℝ × ℝ

-- Define the main theorem
theorem distinct_pairwise_distances_conditional : 
  ∃ (points : Fin 30 → plane)
    (color : Fin 30 → Color), 
    (∀ i j, i ≠ j → dist (points i) (points j) ≠ dist (points j) (points i)) ∧
    (∀ (i : Fin 10), 
       ((second_closest_point (color i) Blue points) = Red) ∧ 
       ((second_closest_point (color (i + 10)) Red points) = Green) ∧ 
       ((second_closest_point (color (i + 20)) Green points) = Blue)) := sorry

-- Function to find the second closest point
-- Definition would be required in the full proof
noncomputable def second_closest_point 

  (col : Color)
  (c : Color)
  (points : Fin 30 → plane) :
  Color :=
  sorry

end distinct_pairwise_distances_conditional_l588_588061


namespace javier_dogs_l588_588442

theorem javier_dogs (humans : ℕ) (javier_legs : ℕ) (total_legs : ℕ) : humans = 5 ∧ javier_legs = humans * 2 ∧ total_legs = 22 → (total_legs - javier_legs) / 4 = 3 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h2a h2b
  rw [h2a, h1] at h2b
  sorry

end javier_dogs_l588_588442


namespace relationship_abc_l588_588752

noncomputable def f (x : ℝ) : ℝ := sorry

theorem relationship_abc
  (odd_f : ∀ x, f(-x) = -f(x)) 
  (monotone_decreasing_on_negative : ∀ ⦃x y : ℝ⦄, x < y → y < 0 → f(y) < f(x))
  (f_minus_one : f(-1) = 0)
  (a := f (-real.log 8 / real.log 3))
  (b := f (-2))
  (c := f (real.exp (2 * real.log (real.exp (2/3)) / 3)))
  : c < a ∧ a < b := 
sorry

end relationship_abc_l588_588752


namespace area_of_CDE_is_54_l588_588215

-- Define points O, A, B, C, D, and E using coordinates.
def point := (ℝ × ℝ)
def O : point := (0, 0)
def A : point := (4, 0)
def B : point := (16, 0)
def C : point := (16, 12)
def D : point := (4, 12)
def E : point := (4, 3)  -- Midpoint derived from problem's similarity conditions

-- Define the line segment lengths based on the points.
def length (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Define DE and DC
noncomputable def DE : ℝ := 9
noncomputable def DC : ℝ := 12

-- Area of triangle CDE
noncomputable def area_CDE : ℝ := (1 / 2) * DE * DC

-- Theorem statement
theorem area_of_CDE_is_54 : area_CDE = 54 := by
  sorry

end area_of_CDE_is_54_l588_588215


namespace sum_complex_multiple_of_5_l588_588919

theorem sum_complex_multiple_of_5 (n : ℕ) (h : n % 5 = 0) : 
  (∑ k in Finset.range (n+1), (↑(k+1) : ℂ) * complex.I ^ k) = (↑n + 3 - (2 * n / 5) * complex.I) :=
by
  sorry

end sum_complex_multiple_of_5_l588_588919


namespace contrapositive_l588_588534

theorem contrapositive (m : ℝ) :
  (∀ m > 0, ∃ x : ℝ, x^2 + x - m = 0) ↔ (∀ m ≤ 0, ∀ x : ℝ, x^2 + x - m ≠ 0) := by
  sorry

end contrapositive_l588_588534


namespace camel_taller_than_hare_l588_588934

theorem camel_taller_than_hare (hare_height_in_inches : ℝ) (camel_height_in_feet : ℝ) 
  (hare_to_feet_conversion : ℝ) (camel_height_in_feet_div_hare_height_in_feet : ℝ) 
  (hare_height_converted : hare_height_in_inches / hare_to_feet_conversion = hare_height_in_feet)
  (camel_taller : camel_height_in_feet / hare_height_in_feet = camel_height_in_feet_div_hare_height_in_feet) :
  camel_height_in_feet_div_hare_height_in_feet ≈ 24 :=
begin
  sorry
end

end camel_taller_than_hare_l588_588934


namespace segment_length_abs_cubed_root_l588_588597

theorem segment_length_abs_cubed_root (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  ∃ a b : ℝ, a = 3 + 5 ∧ b = 3 - 5 ∧ (b - a).abs = 10 :=
by {
  have h1 : real.cbrt 27 = 3 := by norm_num,
  rw h1 at h,
  have h2 : |x - 3| = 5 := h,
  use [8, -2],
  split,
  { refl },
  { split,
    { refl },
    { norm_num } }
}

end segment_length_abs_cubed_root_l588_588597


namespace dollars_sum_l588_588177

theorem dollars_sum : 
  (5 / 8 : ℝ) + (2 / 5) = 1.025 :=
by
  sorry

end dollars_sum_l588_588177


namespace buddy_met_boy_students_l588_588184

theorem buddy_met_boy_students (total_students : ℕ) (girl_students : ℕ) (boy_students : ℕ) (h1 : total_students = 123) (h2 : girl_students = 57) : boy_students = 66 :=
by
  sorry

end buddy_met_boy_students_l588_588184


namespace imaginary_unit_power_division_l588_588761

noncomputable def i : ℂ := complex.I

theorem imaginary_unit_power_division :
  i^3 + (1 / i) = -2 * i :=
by sorry

end imaginary_unit_power_division_l588_588761


namespace solve_for_x_l588_588017

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 2 * x = 0) (h₁ : x ≠ 0) : x = 2 :=
sorry

end solve_for_x_l588_588017


namespace smallest_n_circle_l588_588998

theorem smallest_n_circle (n : ℕ) 
    (h1 : ∀ i j : ℕ, i < j → j - i = 3 ∨ j - i = 4 ∨ j - i = 5) :
    n = 7 :=
sorry

end smallest_n_circle_l588_588998


namespace geometry_problem_l588_588356

theorem geometry_problem
  (F : ℝ × ℝ) (H : ℝ × ℝ)
  (A B C : ℝ × ℝ)
  (a b r : ℝ)
  (ha : A = (4, 4))
  (line_AB : β × ℝ := y = 4 / 5 * (x - 1))
  (circle_ABC : β × ℝ × ℝ = (x-a)^2 + (y-b)^2 = r^2) :
  a^2 = r^2 + 1 :=
  sorry

end geometry_problem_l588_588356


namespace exists_n_with_ratio_999_l588_588340

def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def isPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

noncomputable def D2 (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, d ∣ n ∧ isPerfectSquare d).card
noncomputable def D3 (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, d ∣ n ∧ isPerfectCube d).card

theorem exists_n_with_ratio_999 : ∃ n : ℕ, D2 n = 999 * D3 n := by
  sorry

end exists_n_with_ratio_999_l588_588340


namespace range_s_l588_588678

def s (x : ℝ) : ℝ := 1 / (1 - x)^3

theorem range_s : set.range s = set.Ioi 0 := by
  sorry

end range_s_l588_588678


namespace num_of_lines_through_3_points_eq_365_l588_588099

-- Define the ranges and properties of our marked points
def points : List (ℕ × ℕ) :=
  [(x, y) | x <- [0, 1, 2], y <- [0, 1, 2, ..., 26]]

-- Helper function for even check
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Helper function for odd check
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- Main theorem to be proven
theorem num_of_lines_through_3_points_eq_365 :
  ∃! l, ∃ a b c : ℕ,
  0 ≤ a ∧ a ≤ 26 ∧ 0 ≤ b ∧ b ≤ 26 ∧ 0 ≤ c ∧ c ≤ 26 ∧
  (is_even a ∧ is_even c ∨ is_odd a ∧ is_odd c) ∧
  b = (a + c) / 2 ∧
  l = 365 := sorry

end num_of_lines_through_3_points_eq_365_l588_588099


namespace perpendicular_vectors_t_value_l588_588795

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem perpendicular_vectors_t_value :
  ∀ (t : ℝ),
  let a := (t, 1) in
  let b := (1, 2) in
  dot_product a b = 0 → t = -2 :=
sorry

end perpendicular_vectors_t_value_l588_588795


namespace no_solution_in_specific_quadrants_l588_588023

theorem no_solution_in_specific_quadrants (θ : ℝ) :
  (1 + sin θ * real.sqrt (sin θ ^ 2) + cos θ * real.sqrt (cos θ ^ 2) = 0) →
  ¬ (0 < θ ∧ θ < π/2 ∨ π/2 < θ ∧ θ < π ∨ 3*π/2 < θ ∧ θ < 2*π) :=
by
  sorry

end no_solution_in_specific_quadrants_l588_588023


namespace find_angle_for_triangle_l588_588437

noncomputable def triangle_angle (A B : ℝ) : ℝ := π - (A + B)

lemma tan_sum (A B : ℝ) (hA : tan A = 1/3) (hB : tan B = -2) :
  ∃ C, C = triangle_angle A B ∧ tan C = 1 ∧ 0 < C ∧ C < π :=
by
  use π/4
  split
  { exact (π - (A + B)) },
  split
  { -- calculation to show that tan(π - (A + B)) = 1
    sorry },
  { split
    { norm_num, linarith },
    { norm_num, linarith } }

-- Now the actual theorem we want to prove
theorem find_angle_for_triangle (A B : ℝ) (hA : tan A = 1/3) (hB : tan B = -2) :
  ∃ C, 0 < C ∧ C < π ∧ tan C = 1 :=
by
  obtain ⟨C, hC⟩ := tan_sum A B hA hB
  use C
  exact hC

end find_angle_for_triangle_l588_588437


namespace log_y_minus_x_plus_1_pos_l588_588869

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588869


namespace find_ellipse_eq_find_triangle_area_l588_588361

-- Definitions based on the conditions of the problem
structure Ellipse :=
(a b : ℝ)
(h_a_pos : a > 0)
(h_b_pos : b > 0)
(h_a_gt_b : a > b)

structure Point :=
(x y : ℝ)

def focus_distance (F1 F2 : Point) : ℝ :=
((F2.x - F1.x)^2 + (F2.y - F1.y)^2).sqrt

def ellipse_focal_points (e : Ellipse) : Point × Point :=
({ x := -((e.a^2 - e.b^2).sqrt), y := 0 }, { x := (e.a^2 - e.b^2).sqrt, y := 0 })

def ellipse_eq (e : Ellipse) (P : Point) : Prop :=
P.x^2 / e.a^2 + P.y^2 / e.b^2 = 1

-- Problem statement Part 1
theorem find_ellipse_eq (e : Ellipse) (F1 F2 : Point) (AB : Point × Point)
  (h_focus_dist : focus_distance F1 F2 = 2)
  (h_triangle_perimeter : (focus_distance F1 AB.1 + focus_distance F2 AB.1 + focus_distance F2 AB.2 = 4 * Real.sqrt 3))
  : ellipse_eq { a := Real.sqrt 3, b := Real.sqrt 2, ..e } (AB.1) :=
sorry

-- Problem statement Part 2
theorem find_triangle_area (F1 : Point) (F2 : Point) (AB : Point × Point)
  (h_slope : 2)
  (h_focus_dist : focus_distance F1 F2 = 2)
  (h_triangle_perimeter : (focus_distance F1 AB.1 + focus_distance F2 AB.1 + focus_distance F2 AB.2 = 4 * Real.sqrt 3)) :
  let d := 4 * Real.sqrt 15 / 7
  ∃ (A B : Point), 1/2 * focus_distance AB.1 AB.2 * focus_distance F2 { x := focus_distance F2.1 (AB.1.1 + 1), y := 2 * (focus_distance F2.1 + 1) } = d :=
sorry

end find_ellipse_eq_find_triangle_area_l588_588361


namespace find_ellipse_l588_588750

open Real

def ellipse (a b : ℝ) := ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1)

def passes_through (x y a b : ℝ) := (x = 0) ∧ (y = 4)

def eccentricity (c a : ℝ) := c = a * (3/5)

noncomputable def equation_of_ellipse (a b : ℝ) := aa = 5 ∧ b = 4

theorem find_ellipse :
  ∃ (a b : ℝ),
  a > b ∧ b > 0 ∧
  ellipse a b 0 4 ∧
  eccentricity (sqrt (a^2 - b^2)) a ∧
  equation_of_ellipse a b = (a = 5 ∧ b = 4) :=
by 
  sorry

end find_ellipse_l588_588750


namespace least_positive_integer_special_property_l588_588335

/-- 
  Prove that 9990 is the least positive integer whose digits sum to a multiple of 27 
  and the number itself is not a multiple of 27.
-/
theorem least_positive_integer_special_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (Nat.digits 10 n).sum % 27 = 0 ∧ 
  n % 27 ≠ 0 ∧ 
  ∀ m : ℕ, (m > 0 ∧ (Nat.digits 10 m).sum % 27 = 0 ∧ m % 27 ≠ 0 → n ≤ m) := 
by
  sorry

end least_positive_integer_special_property_l588_588335


namespace max_min_PA_and_radius_inscribed_circle_triangle_l588_588364

noncomputable def circle_center : ℝ × ℝ := (-3, m)
noncomputable def point_A : ℝ × ℝ := (2, 0)
noncomputable def radius : ℝ := real.sqrt 13
noncomputable def distance_from_line_to_center : ℝ := abs ((-12 + 3 * m + 1) / 5)
noncomputable def m_value : ℝ := 2
noncomputable def PA_max : ℝ := real.sqrt 29 + real.sqrt 13
noncomputable def PA_min : ℝ := real.sqrt 29 - real.sqrt 13
noncomputable def triangle_vertices : list (ℝ × ℝ) := [(0, 4), (0, 0), (-6, 0)]
noncomputable def inscribed_circle_radius : ℝ := 5 - real.sqrt 13

theorem max_min_PA_and_radius_inscribed_circle_triangle : 
  ∃ m < 3, 
    distance_from_line_to_center = 1 ∧ 
    m_value = 2 ∧ 
    PA_max = abs ((2 - (-3)) ^ 2 + (0 - 2) ^ 2 + 13)^(1/2) + real.sqrt 13 ∧
    PA_min = abs ((2 - (-3)) ^ 2 + (0 - 2) ^ 2 + 13)^(1/2) - real.sqrt 13 ∧
    (∃ r > 0, r = 5 - real.sqrt 13) :=
sorry

end max_min_PA_and_radius_inscribed_circle_triangle_l588_588364


namespace total_time_is_10_l588_588645

-- Definitions based on conditions
def total_distance : ℕ := 224
def first_half_distance : ℕ := total_distance / 2
def second_half_distance : ℕ := total_distance / 2
def speed_first_half : ℕ := 21
def speed_second_half : ℕ := 24

-- Definition of time taken for each half of the journey
def time_first_half : ℚ := first_half_distance / speed_first_half
def time_second_half : ℚ := second_half_distance / speed_second_half

-- Total time is the sum of time taken for each half
def total_time : ℚ := time_first_half + time_second_half

-- Theorem stating the total time taken for the journey
theorem total_time_is_10 : total_time = 10 := by
  sorry

end total_time_is_10_l588_588645


namespace log_pos_if_exp_diff_l588_588883

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588883


namespace log_pos_given_ineq_l588_588890

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588890


namespace find_N_l588_588698

open Matrix

theorem find_N (u : ℝ^3) : 
  let cross_product := @cross_product ℝ _ ⟨3, -1, 4⟩ u
  let N := ![
      ![0, -4, -1],
      ![4, 0, -3],
      ![1, 3, 0]
  ]
  N ⬝ u = cross_product :=
by {
  sorry
}

end find_N_l588_588698


namespace log_of_y_sub_x_plus_one_positive_l588_588903

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588903


namespace book_arrangement_count_l588_588399

-- Define the conditions
def total_books : ℕ := 6
def identical_books : ℕ := 3
def different_books : ℕ := total_books - identical_books

-- Prove the number of arrangements
theorem book_arrangement_count : (total_books.factorial / identical_books.factorial) = 120 := by
  sorry

end book_arrangement_count_l588_588399


namespace problem_l588_588815

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588815


namespace log_of_y_sub_x_plus_one_positive_l588_588898

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588898


namespace log_eval_l588_588326

-- Define the cube root of 6
def cube_root_6 : ℝ := Real.cbrt 6

-- Define the term 648 * cube_root_6
def term : ℝ := 648 * cube_root_6

-- The logarithm base cube_root_6 of the term should equal 11.5
theorem log_eval : Real.logb cube_root_6 term = 11.5 := by sorry

end log_eval_l588_588326


namespace group_contains_1991st_odd_l588_588395

theorem group_contains_1991st_odd (n : ℕ) : 
    ∀ k : ℕ, k ∈ {1, 3, 5, ...} ∧ ((2*n - 1) odd numbers per group) → 
    k = 1991 → n = 32 :=
by
  sorry

end group_contains_1991st_odd_l588_588395


namespace problem_l588_588810

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588810


namespace min_distance_P_to_CC1_l588_588651

-- Define the cube and its properties
def Cube (length : ℝ) :=
  {A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ // 
    (A = (0, 0, 0)) ∧ (B = (0, length, 0)) ∧ (C = (length, length, 0)) ∧ (D = (length, 0, 0)) ∧
    (A1 = (0, 0, length)) ∧ (B1 = (0, length, length)) ∧ (C1 = (length, length, length)) ∧ (D1 = (length, 0, length))}

-- Define the midpoint function
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the point P movement
def P_moves_along (D1 E : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (D1.1 * (1 - t) + E.1 * t, D1.2 * (1 - t) + E.2 * t, D1.3 * (1 - t) + E.3 * t)

-- Define the distance from P to a line (CC1) given certain conditions
theorem min_distance_P_to_CC1 {t : ℝ} (cube : Cube 2) :
  let E := midpoint cube.val.C cube.val.B in
  let P := P_moves_along cube.val.D1 E t in
  distance P (cube.val.C, cube.val.C1) = 2 * sqrt 5 / 5 := sorry

end min_distance_P_to_CC1_l588_588651


namespace dina_dolls_l588_588319

theorem dina_dolls (Ivy_collectors: ℕ) (h1: Ivy_collectors = 20) (h2: ∀ y : ℕ, 2 * y / 3 = Ivy_collectors) :
  ∃ x : ℕ, 2 * x = 60 :=
  sorry

end dina_dolls_l588_588319


namespace Maria_selling_price_l588_588984

-- Define the constants based on the given conditions
def brush_cost : ℕ := 20
def canvas_cost : ℕ := 3 * brush_cost
def paint_cost_per_liter : ℕ := 8
def paint_needed : ℕ := 5
def earnings : ℕ := 80

-- Calculate the total cost and the selling price
def total_cost : ℕ := brush_cost + canvas_cost + (paint_cost_per_liter * paint_needed)
def selling_price : ℕ := total_cost + earnings

-- Proof statement
theorem Maria_selling_price : selling_price = 200 := by
  sorry

end Maria_selling_price_l588_588984


namespace sum_of_f_l588_588363

-- Given conditions as Lean definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g(x) = g(-x)
def is_even_when_shifted (f : ℝ → ℝ) : Prop := is_even_function (λ x, f(x + 2))

-- Main theorem statement
theorem sum_of_f (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd_function f) 
  (h_even_shift : is_even_when_shifted f) 
  (h_value : f 1 = a) :
  (∑ i in (finset.range 1010).map (λ i, 1 + 2 * i), f i) = 2 * a :=
sorry

end sum_of_f_l588_588363


namespace minimum_obtuse_triangles_in_triangulation_of_2003gon_l588_588672

-- Defining the concepts of a polygon, a circle, and a triangulation
def polygon (n : ℕ) := { v : ℕ // 3 ≤ n }

def inscribed_circle (n : ℕ) (P : polygon n) := 
  ∃ C : ℝ × ℝ, ∀ v ∈ P, ∃ r : ℝ, (circle_equation C r v)

def triangulation (n : ℕ) (P : polygon n) := 
  ∀ t, t ∈ (triangularization P) → (obtuse t) ∨ (acute t ∨ right t)

-- Main statement we want to prove
theorem minimum_obtuse_triangles_in_triangulation_of_2003gon : 
  ∀ (P : polygon 2003), inscribed_circle 2003 P → ∃ k, k = 1999 ∧
  (∀ t ∈ (triangularization P), obtuse t → k = 1999) :=
by
  sorry

end minimum_obtuse_triangles_in_triangulation_of_2003gon_l588_588672


namespace percentage_increase_l588_588269

def initialProductivity := 120
def totalArea := 1440
def daysInitialProductivity := 2
def daysAheadOfSchedule := 2

theorem percentage_increase :
  let originalDays := totalArea / initialProductivity
  let daysWithIncrease := originalDays - daysAheadOfSchedule
  let daysWithNewProductivity := daysWithIncrease - daysInitialProductivity
  let remainingArea := totalArea - (daysInitialProductivity * initialProductivity)
  let newProductivity := remainingArea / daysWithNewProductivity
  let increase := ((newProductivity - initialProductivity) / initialProductivity) * 100
  increase = 25 :=
by
  sorry

end percentage_increase_l588_588269


namespace smallest_obtuse_triangles_l588_588671

def obtuseTrianglesInTriangulation (n : Nat) : Nat :=
  if n < 3 then 0 else (n - 2) - 2

theorem smallest_obtuse_triangles (n : Nat) (h : n = 2003) :
  obtuseTrianglesInTriangulation n = 1999 := by
  sorry

end smallest_obtuse_triangles_l588_588671


namespace cosine_AB_AC_l588_588695

-- Definitions of the given points
def A : ℝ × ℝ × ℝ := (0, -3, 6)
def B : ℝ × ℝ × ℝ := (-12, -3, -3)
def C : ℝ × ℝ × ℝ := (-9, -3, -6)

-- Definition of cosine between vectors AB and AC
noncomputable def vector_cosine (v w : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let magnitude_v := real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let magnitude_w := real.sqrt (w.1 ^ 2 + w.2 ^ 2 + w.3 ^ 2)
  dot_product / (magnitude_v * magnitude_w)

-- Vectors AB and AC
def AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Proof statement
theorem cosine_AB_AC : vector_cosine AB AC = 0.96 := sorry

end cosine_AB_AC_l588_588695


namespace cost_of_sneakers_l588_588450

theorem cost_of_sneakers (saved money per_action_figure final_money cost : ℤ) 
  (h1 : saved = 15) 
  (h2 : money = 10) 
  (h3 : per_action_figure = 10) 
  (h4 : final_money = 25) 
  (h5 : money * per_action_figure + saved - cost = final_money) 
  : cost = 90 := 
sorry

end cost_of_sneakers_l588_588450


namespace sally_balance_fraction_l588_588117

variable (G : ℝ) (x : ℝ)
-- spending limit on gold card is G
-- spending limit on platinum card is 2G
-- Balance on platinum card is G/2
-- After transfer, 0.5833333333333334 portion of platinum card remains unspent

theorem sally_balance_fraction
  (h1 : (5/12) * 2 * G = G / 2 + x * G) : x = 1 / 3 :=
by
  sorry

end sally_balance_fraction_l588_588117


namespace number_of_men_in_first_group_l588_588134

-- Define the conditions and the proof problem
theorem number_of_men_in_first_group (M : ℕ) 
  (h1 : ∀ t : ℝ, 22 * t = M) 
  (h2 : ∀ t' : ℝ, 18 * 17.11111111111111 = t') :
  M = 14 := 
by
  sorry

end number_of_men_in_first_group_l588_588134


namespace semi_circle_perimeter_approx_l588_588555

noncomputable def perimeter_of_semi_circle (r : ℝ) : ℝ :=
  real.pi * r + 2 * r

theorem semi_circle_perimeter_approx (radius : ℝ) (h : radius = 6.83) : 
  abs (perimeter_of_semi_circle radius - 35.12) < 0.01 :=
by
  sorry

end semi_circle_perimeter_approx_l588_588555


namespace area_ratio_PQRS_WXYZ_l588_588135

-- Definitions for vertices of the square WXYZ
def point (α : Type) := prod α α

-- Definitions and conditions based on the problem statement
def square_WXYZ (s : ℝ) : set (point ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

def point_P (s : ℝ) : point ℝ := (3 * s / 4, 0)

def area (s : ℝ) : ℝ := s * s

-- Theorem statement of the proof problem
theorem area_ratio_PQRS_WXYZ (s : ℝ) (h₁: s > 0) (P at (3 * s / 4, 0)) :
  point P ∈ square_WXYZ s ->
  (area_PQRS s) / (area s) = 1 / 8 :=
sorry

end area_ratio_PQRS_WXYZ_l588_588135


namespace max_value_of_function_l588_588541

theorem max_value_of_function : 
  ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → (3 * x - 4 * x^3) ≤ 1 :=
by
  intro x hx0 hx1
  -- proof goes here
  sorry

end max_value_of_function_l588_588541


namespace percentage_quarters_is_correct_l588_588233

def dime_value : ℕ := 40 * 10
def quarter_value : ℕ := 30 * 25
def halfdollar_value : ℕ := 10 * 50
def total_value : ℕ := dime_value + quarter_value + halfdollar_value
def percent_quarters : ℚ := (quarter_value.to_rat / total_value.to_rat) * 100

theorem percentage_quarters_is_correct :
  percent_quarters = 45.45 := 
sorry

end percentage_quarters_is_correct_l588_588233


namespace variance_of_scores_l588_588281

-- Define the list of scores
def scores : List ℕ := [110, 114, 121, 119, 126]

-- Define the formula for variance calculation
def variance (l : List ℕ) : ℚ :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  (l.map (λ x => ((x : ℚ) - mean) ^ 2)).sum / n

-- The main theorem to be proved
theorem variance_of_scores :
  variance scores = 30.8 := 
  by
    sorry

end variance_of_scores_l588_588281


namespace smallest_integer_in_set_l588_588933

def avg_seven_consecutive_integers (n : ℤ) : ℤ :=
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7

theorem smallest_integer_in_set : ∃ (n : ℤ), n = 0 ∧ (n + 6 < 3 * avg_seven_consecutive_integers n) :=
by
  sorry

end smallest_integer_in_set_l588_588933


namespace percentage_markup_l588_588547

theorem percentage_markup :
  let SP := 5400
  let CP := 4090.9090909090905
  (SP - CP) / CP * 100 ≈ 32 := 
by
  sorry

end percentage_markup_l588_588547


namespace inequality_ln_pos_l588_588840

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588840


namespace total_stamp_arrangements_l588_588317

-- Define the available stamps quantities
def stamps : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

-- Define the problem to prove
theorem total_stamp_arrangements :
  ∃ (n : ℕ), 
  (∀ (arrangement : List (ℕ × ℕ)), 
    arrangement.sum (fun (a : ℕ × ℕ) => a.1 * a.2) = 15 → unique_arrangements arrangement stamps)
  = n :=
sorry

end total_stamp_arrangements_l588_588317


namespace smallest_solution_of_abs_eq_l588_588706

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end smallest_solution_of_abs_eq_l588_588706


namespace permutations_of_BANANA_l588_588662

open Nat

-- Define the number of letters and their repetitions
def wordLength : Nat := 6
def countA : Nat := 3
def countN : Nat := 2

-- Define the factorial function
def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem permutations_of_BANANA :
  (factorial wordLength) / ((factorial countA) * (factorial countN)) = 60 := by
  sorry

end permutations_of_BANANA_l588_588662


namespace find_a_l588_588080

theorem find_a (x y z a : ℝ) (h1 : ∃ k : ℝ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) 
              (h2 : x + y + z = 70) 
              (h3 : y = 15 * a - 5) : 
  a = 5 / 3 := 
by sorry

end find_a_l588_588080


namespace log_y_minus_x_plus_1_pos_l588_588867

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588867


namespace range_of_a_l588_588158

open Set

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) := ∀ x : ℝ, x ∈ (Icc 1 2) → x^2 ≥ a

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ (Ioo 1 2 ∪ Iic (-2)) :=
by sorry

end range_of_a_l588_588158


namespace problem_l588_588811

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588811


namespace dina_dolls_count_l588_588320

-- Define the conditions
variable (Ivy_dolls : ℕ)
variable (Collectors_Ivy_dolls : ℕ := 20)
variable (Dina_dolls : ℕ)

-- Condition: Ivy has 2/3 of her dolls as collectors editions
def collectors_edition_condition : Prop := (2 / 3 : ℝ) * Ivy_dolls = Collectors_Ivy_dolls

-- Condition: Dina has twice as many dolls as Ivy
def dina_ivy_dolls_relationship : Prop := Dina_dolls = 2 * Ivy_dolls

-- Theorem: Prove that Dina has 60 dolls
theorem dina_dolls_count : collectors_edition_condition Ivy_dolls ∧ dina_ivy_dolls_relationship Ivy_dolls Dina_dolls → Dina_dolls = 60 := by
  sorry

end dina_dolls_count_l588_588320


namespace log_pos_given_ineq_l588_588891

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588891


namespace centroid_quadrilateral_area_l588_588136

open EuclideanGeometry

-- Definitions for the square and the point Q inside the square
def Square (W X Y Z : Point) (s : ℝ) : Prop :=
  (dist W X = s) ∧ (dist X Y = s) ∧ (dist Y Z = s) ∧ (dist Z W = s) ∧
  (angle W X Y = π/2) ∧ (angle X Y Z = π/2) ∧ (angle Y Z W = π/2) ∧ (angle Z W X = π/2)

def PointInSquare (Q W X Y Z : Point) (wq xq : ℝ) : Prop :=
  dist W Q = wq ∧ dist X Q = xq

-- The theorem statement
theorem centroid_quadrilateral_area
  (W X Y Z Q : Point)
  (s a b : ℝ)
  (hs : s = 40)
  (ha : a = 15)
  (hb : b = 32)
  (hSquare : Square W X Y Z s)
  (hPointInSquare : PointInSquare Q W X Y Z a b) :
  1/2 * ((2 * s / 3) * (2 * s / 3)) = 3200 / 9 :=
by
  sorry

end centroid_quadrilateral_area_l588_588136


namespace smallest_AC_l588_588300

theorem smallest_AC (A B C D : ℝ) (AB AC : ℝ) (h₁ : AB = AC) (AC_int : AC ∈ ℤ) (CD_int : (C - D) ∈ ℤ) (h₂ : D = (A + C) / 2) 
 (BD : ℝ) (h₃ : BD^2 = 85) (h₄ : ∃ k : ℝ, k * (B - D) = 0) :
  AC = 13 := sorry

end smallest_AC_l588_588300


namespace unique_perpendicular_plane_infinite_parallel_planes_l588_588803

variables {l : Type*} {P : Type*} [Line l] [Point P]

-- Assumption that there exists a line l and a point P outside the line l.
variable (hP_outside_l : ¬ ∃ point ∈ l, point = P)

-- Theorem: There is exactly one plane perpendicular to the given line through an external point.
theorem unique_perpendicular_plane : ∃! plane, ∀ (point ∈ plane), plane ⊥ l ∧ point ≠ P :=
sorry

-- Theorem: There are infinitely many planes parallel to the given line through an external point.
theorem infinite_parallel_planes : ∃ plane, ∀ (point ∈ plane), plane ∥ l ∧ point ≠ P :=
sorry

end unique_perpendicular_plane_infinite_parallel_planes_l588_588803


namespace brianna_books_gift_l588_588656

theorem brianna_books_gift (books_per_month : ℕ) (months_per_year : ℕ) (books_bought : ℕ) 
  (borrow_difference : ℕ) (books_reread : ℕ) (total_books_needed : ℕ) : 
  (books_per_month * months_per_year = total_books_needed) →
  ((books_per_month * months_per_year) - books_reread - 
  (books_bought + (books_bought - borrow_difference)) = 
  books_given) →
  books_given = 6 := 
by
  intro h1 h2
  sorry

end brianna_books_gift_l588_588656


namespace log_pos_if_exp_diff_l588_588875

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588875


namespace bananas_distribution_l588_588309

noncomputable def total_bananas : ℝ := 550.5
noncomputable def lydia_bananas : ℝ := 80.25
noncomputable def dawn_bananas : ℝ := lydia_bananas + 93
noncomputable def emily_bananas : ℝ := 198
noncomputable def donna_bananas : ℝ := emily_bananas / 2

theorem bananas_distribution :
  dawn_bananas = 173.25 ∧
  lydia_bananas = 80.25 ∧
  donna_bananas = 99 ∧
  emily_bananas = 198 ∧
  dawn_bananas + lydia_bananas + donna_bananas + emily_bananas = total_bananas :=
by
  sorry

end bananas_distribution_l588_588309


namespace tangent_length_and_line_DE_secant_line_AB_l588_588741

-- Given definitions for the circle C and point M
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0
def pointM := (4 : ℝ, -8 : ℝ)

-- Problem (1): Prove the length of the tangent and the equation of line DE
theorem tangent_length_and_line_DE :
  (∃ l_tangent : ℝ, l_tangent = 3 * Real.sqrt 5) ∧ 
  (∃ eq_line_DE : ℝ → ℝ → Prop, eq_line_DE = λ x y, 2 * x - 7 * y - 19 = 0) :=
by {
  sorry,
}

-- Problem (2): Prove the equation of the line AB given |AB|=4
theorem secant_line_AB :
  (∃ eq_line_AB : ℝ → ℝ → Prop, 
    (eq_line_AB = λ x y, 45 * x + 28 * y + 44 = 0) ∨ 
    (eq_line_AB = λ x, x = 4)) :=
by {
  sorry,
}

end tangent_length_and_line_DE_secant_line_AB_l588_588741


namespace temperature_difference_l588_588149

variable (highest_temp : ℤ)
variable (lowest_temp : ℤ)

theorem temperature_difference : 
  highest_temp = 2 ∧ lowest_temp = -8 → (highest_temp - lowest_temp = 10) := by
  sorry

end temperature_difference_l588_588149


namespace total_earnings_correct_l588_588229

-- Definitions of wealth received by each friend
def Zachary_earnings := 40 * 5
def Jason_earnings := Zachary_earnings + 0.30 * Zachary_earnings
def Ryan_earnings := Jason_earnings + 50
def Emily_earnings := Ryan_earnings - 0.20 * Ryan_earnings
def Lily_earnings := Emily_earnings + 70

-- Total earnings of all five friends
def total_earnings := Zachary_earnings + Jason_earnings + Ryan_earnings + Emily_earnings + Lily_earnings

-- The main theorem to prove
theorem total_earnings_correct : total_earnings = 1336 :=
by
  sorry

end total_earnings_correct_l588_588229


namespace arithmetic_sequence_twenty_fourth_term_l588_588566

-- Given definitions (conditions)
def third_term (a d : ℚ) : ℚ := a + 2 * d
def tenth_term (a d : ℚ) : ℚ := a + 9 * d
def twenty_fourth_term (a d : ℚ) : ℚ := a + 23 * d

-- The main theorem to be proved
theorem arithmetic_sequence_twenty_fourth_term 
  (a d : ℚ) 
  (h1 : third_term a d = 7) 
  (h2 : tenth_term a d = 27) :
  twenty_fourth_term a d = 67 := by
  sorry

end arithmetic_sequence_twenty_fourth_term_l588_588566


namespace min_rods_of_rook_is_n_l588_588751

def n_ge_2 (n : ℕ) : Prop := n ≥ 2

def is_rook {n : ℕ} (A : set (ℕ × ℕ)) : Prop :=
  A.card = n ∧ (∀ a b ∈ A, a ≠ b → a.1 ≠ b.1 ∧ a.2 ≠ b.2)

def is_rod {m : ℕ} (R : set (ℕ × ℕ)) : Prop :=
  (∃ k, k ∈ ℕ ∧ (R = { (r, c) | r = m ∨ c = m }))

def m_of (A : set (ℕ × ℕ)) (m : ℕ) : Prop :=
  ∃ partition : set (set (ℕ × ℕ)),
  (∀ R ∈ partition, is_rod R) ∧
  (∀ x : ℕ × ℕ, x ∈ A ↔ ∃ R ∈ partition, x ∈ R) ∧
  partition.card = m

theorem min_rods_of_rook_is_n (n : ℕ) (A : set (ℕ × ℕ)) :
  n_ge_2 n → is_rook A → (∀ m, m_of A m → m ≥ n) := sorry

end min_rods_of_rook_is_n_l588_588751


namespace inequality_ln_pos_l588_588832

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588832


namespace unique_zero_exists_bounds_for_zero_inequality_for_zero_l588_588354

noncomputable def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ := e ^ x - x - a

theorem unique_zero_exists {a : ℝ} (ha : 1 < a ∧ a ≤ 2) :
  ∃! (x : ℝ), (0 < x) ∧ f a x = 0 := sorry

theorem bounds_for_zero {a : ℝ} (ha : 1 < a ∧ a ≤ 2) (x₀ : ℝ)
  (hx₀ : (0 < x₀) ∧ f a x₀ = 0) :
  Real.sqrt (a - 1) ≤ x₀ ∧ x₀ ≤ Real.sqrt (2 * (a - 1)) := sorry

theorem inequality_for_zero {a : ℝ} (ha : 1 < a ∧ a ≤ 2) (x₀ : ℝ)
  (hx₀ : (0 < x₀) ∧ f a x₀ = 0) :
  x₀ * f a (e ^ x₀) ≥ (e - 1) * (a - 1) * a := sorry

end unique_zero_exists_bounds_for_zero_inequality_for_zero_l588_588354


namespace angle_of_inclination_l588_588573

theorem angle_of_inclination 
  (α : ℝ) 
  (h_tan : Real.tan α = -Real.sqrt 3)
  (h_range : 0 ≤ α ∧ α < 180) : α = 120 :=
by
  sorry

end angle_of_inclination_l588_588573


namespace factorize_expression_l588_588690

-- Define the expression E
def E (x y z : ℝ) : ℝ := x^2 + x*y - x*z - y*z

-- State the theorem to prove \(E = (x + y)(x - z)\)
theorem factorize_expression (x y z : ℝ) : 
  E x y z = (x + y) * (x - z) := 
sorry

end factorize_expression_l588_588690


namespace inequality_ln_positive_l588_588855

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588855


namespace cone_volume_and_surface_area_l588_588161

-- Define the given conditions
def slant_height : ℝ := 15
def height : ℝ := 9

-- Define the radius of the base using the Pythagorean theorem
def radius : ℝ := real.sqrt (slant_height ^ 2 - height ^ 2)

-- Define the formulas for the volume and surface area
def volume (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h
def surface_area (r l : ℝ) : ℝ := real.pi * r^2 + real.pi * r * l

-- Prove the volume is 432π cubic centimeters and surface area is 324π square centimeters
theorem cone_volume_and_surface_area :
  volume radius height = 432 * real.pi ∧
  surface_area radius slant_height = 324 * real.pi :=
by
  -- We use sorry to skip the proof
  sorry

end cone_volume_and_surface_area_l588_588161


namespace theta_in_second_quadrant_l588_588756

theorem theta_in_second_quadrant (θ : ℝ) (h1 : Real.sin θ > 0) (h2 : Real.cos θ < 0) : 
    -- the condition of lying in the second quadrant can be defined by the 
    -- angle being between 90° and 180° (π/2 and π in radians)
    θ ∈ set.Ioo (Float.pi / 2) Float.pi :=
by
  sorry

end theta_in_second_quadrant_l588_588756


namespace Adam_total_shopping_cost_l588_588289

theorem Adam_total_shopping_cost :
  let sandwiches := 3
  let sandwich_cost := 3
  let water_cost := 2
  (sandwiches * sandwich_cost + water_cost) = 11 := 
by
  sorry

end Adam_total_shopping_cost_l588_588289


namespace log_49_48_in_terms_of_a_and_b_l588_588368

-- Define the constants and hypotheses
variable (a b : ℝ)
variable (h1 : a = Real.logb 7 3)
variable (h2 : b = Real.logb 7 4)

-- Define the statement to be proved
theorem log_49_48_in_terms_of_a_and_b (a b : ℝ) (h1 : a = Real.logb 7 3) (h2 : b = Real.logb 7 4) :
  Real.logb 49 48 = (a + 2 * b) / 2 :=
by
  sorry

end log_49_48_in_terms_of_a_and_b_l588_588368


namespace parallel_lines_condition_l588_588619

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 ↔ x + ay - 1 = 0) ↔ (a = 1) :=
sorry

end parallel_lines_condition_l588_588619


namespace diametrically_opposite_P_S_l588_588989

structure Point where
  name : String

inductive Face
| top
| bottom
| left
| right
| front
| back

structure Cube where
  point_on_face : Face → Option Point

def net : Cube :=
{
  point_on_face := 
    λ face, match face with
      | Face.top => some ⟨"P"⟩
      | Face.bottom => some ⟨"S"⟩
      | Face.left => some ⟨"Q"⟩
      | Face.right => some ⟨"R"⟩
      | Face.front => some ⟨"T"⟩
      | Face.back => none
}

def diametrically_opposite (c : Cube) (pt1 pt2 : Point) : Prop :=
  ∃ face1 face2, c.point_on_face face1 = some pt1 ∧ c.point_on_face face2 = some pt2 ∧ face1 ≠ face2

theorem diametrically_opposite_P_S : diametrically_opposite net ⟨"P"⟩ ⟨"S"⟩ :=
by
  sorry

end diametrically_opposite_P_S_l588_588989


namespace find_starting_time_l588_588151

noncomputable def light_glows_every := 13 -- seconds
noncomputable def max_glows := 382.2307692307692 -- times
noncomputable def end_time_seconds := 3 * 3600 + 20 * 60 + 47 -- 3:20:47 AM in seconds
noncomputable def start_time_seconds := 1 * 3600 + 58 * 60 + 1 -- 1:58:01 AM in seconds

theorem find_starting_time :
  let total_glows := real.floor max_glows
      total_time := total_glows * light_glows_every
  in end_time_seconds - total_time = start_time_seconds := by
  sorry

end find_starting_time_l588_588151


namespace find_shaded_area_l588_588675

noncomputable def area_of_shaded_region (r_A r_B : ℝ) (h_small_circle : r_A = 4)
  (h_large_circle : r_B = 2 * r_A) : ℝ :=
  let area_large_circle := π * r_B^2
  let area_small_circle := π * r_A^2
  area_large_circle - area_small_circle

theorem find_shaded_area : area_of_shaded_region 4 8 (by rfl) (by norm_num) = 48 * π :=
by
  dsimp [area_of_shaded_region]
  norm_num
  ring

end find_shaded_area_l588_588675


namespace intersection_shape_is_circle_l588_588022

def plane_intersects_circle_shape (p : Plane) (cyl : Cylinder) (sph : Sphere) : Prop :=
  ∀ (int_shape : Set Point), 
    (intersection_plane_sphere p sph = int_shape) ∧ 
    (intersection_plane_cylinder p cyl = int_shape) → 
    int_shape = circle p

theorem intersection_shape_is_circle (p : Plane) (cyl : Cylinder) (sph : Sphere)
  (h_inter_sph : ∃ (int_shape : Set Point), intersection_plane_sphere p sph = int_shape)
  (h_inter_cyl : ∃ (int_shape : Set Point), intersection_plane_cylinder p cyl = int_shape)
  (h_shapes_same : ∀ (int_shape : Set Point), 
    (intersection_plane_sphere p sph = int_shape) → 
    (intersection_plane_cylinder p cyl = int_shape) → 
    intersection_plane_sphere p sph = intersection_plane_cylinder p cyl) :
  plane_intersects_circle_shape p cyl sph :=
sorry

end intersection_shape_is_circle_l588_588022


namespace calculate_taxes_l588_588026

theorem calculate_taxes (gross_pay net_pay : ℕ) (h_gross_pay : gross_pay = 450) (h_net_pay : net_pay = 315) : gross_pay - net_pay = 135 :=
by
  rw [h_gross_pay, h_net_pay]
  norm_num
  sorry

end calculate_taxes_l588_588026


namespace length_LM_correct_l588_588427

-- Let XYZ be a right triangle with specified side lengths
variables (X Y Z M L : Point)
variable (hXYZ : ∠ (XYZ) = 90°)
variable (XY XZ : ℝ)
variable (XY_len : XY = 5)
variable (XZ_len : XZ = 12)
variable (perpendicular : ∃ L M, is_perpendicular L M Y XZ)

-- Define LM as the length of the perpendicular from Y to the hypotenuse XZ
noncomputable def length_LM (L M : Point) : ℝ :=
  distance L M

-- Define the condition to prove
theorem length_LM_correct :
  ∀ (L M : Point),
    is_perpendicular L M Y XZ → 
    length_LM L M = (5 * sqrt 119) / 12 := by
  sorry

end length_LM_correct_l588_588427


namespace finite_convergence_and_final_position_l588_588652

theorem finite_convergence_and_final_position :
  (∀ (0 <= i ∧ i <= 2022), ∃ p_i : ℕ), (∀ j k, (j < k) → (|p_j - p_k| ≥ 2) → (p_j + 1) = (p_k - 1)) → 
  ∃ N : ℕ, (∀ n > N, ∀ i (0 ≤ i ∧ i ≤ 2022), p_i = 1011) :=
sorry

end finite_convergence_and_final_position_l588_588652


namespace range_of_a_l588_588400

def necessary_but_not_sufficient_condition {a : ℝ} (x : ℝ) := x > 2 → x > a

theorem range_of_a : {a : ℝ | ∀ x, necessary_but_not_sufficient_condition x a} = {a : ℝ | a < 2} :=
sorry

end range_of_a_l588_588400


namespace price_of_other_calculator_l588_588260

theorem price_of_other_calculator :
  ∀ (total_calculators : ℕ) (total_sales : ℕ) (kind1_calculators : ℕ) (price1 : ℕ),
  total_calculators = 85 →
  total_sales = 3875 →
  kind1_calculators = 35 →
  price1 = 15 →
  let kind1_total_sales := kind1_calculators * price1 in
  let kind2_calculators := total_calculators - kind1_calculators in
  let kind2_total_sales := total_sales - kind1_total_sales in
  let price2 := kind2_total_sales / kind2_calculators in
  price2 = 67 :=
by
  intros total_calculators total_sales kind1_calculators price1
  intros h1 h2 h3 h4
  let kind1_total_sales := kind1_calculators * price1
  let kind2_calculators := total_calculators - kind1_calculators
  let kind2_total_sales := total_sales - kind1_total_sales
  let price2 := kind2_total_sales / kind2_calculators
  have h5 : kind1_total_sales = 35 * 15 := by rw [h3, h4]
  have h6 : kind1_total_sales = 525 := by norm_num
  have h7 : kind2_total_sales = 3875 - 525 := by rw [h2, h6]
  have h8 : kind2_total_sales = 3350 := by norm_num
  have h9 : kind2_calculators = 85 - 35 := by rw [h1, h3]
  have h10 : kind2_calculators = 50 := by norm_num
  have h11 : price2 = 3350 / 50 := by rw [h8, h10]
  have h12 : price2 = 67 := by norm_num
  exact h12

end price_of_other_calculator_l588_588260


namespace spending_on_clothes_transport_per_month_l588_588240

noncomputable def monthly_spending_on_clothes_transport (S : ℝ) : ℝ :=
  0.2 * S

theorem spending_on_clothes_transport_per_month :
  ∃ (S : ℝ), (monthly_spending_on_clothes_transport S = 1584) ∧
             (12 * S - (12 * 0.6 * S + 12 * monthly_spending_on_clothes_transport S) = 19008) :=
by
  sorry

end spending_on_clothes_transport_per_month_l588_588240


namespace balls_distribution_l588_588505

theorem balls_distribution : 
  ∃ (n : ℕ), 
    (∀ (b1 b2 : ℕ), ∀ (h : b1 + b2 = 4), b1 ≥ 1 ∧ b2 ≥ 2 → n = 10) :=
sorry

end balls_distribution_l588_588505


namespace distinct_m_values_l588_588466

theorem distinct_m_values : ∀ (m : ℤ), 
  (∃ x1 x2 : ℤ, (x1 * x2 = 36) ∧ (x1 + x2 = m)) ↔ 
  (finite { m : ℤ | ∃ x1 x2 : ℤ, (x1 * x2 = 36) ∧ (x1 + x2 = m) } ∧
   ∃ n, n = 10 := 
by {
  sorry

end distinct_m_values_l588_588466


namespace PQ_perpendicular_to_KX_l588_588496

def midpoint (A B : Point) : Point := 
  sorry -- Assume the midpoint function is defined

def circumcenter (A B C : Point) : Point := 
  sorry -- Assume the circumcenter function is defined

theorem PQ_perpendicular_to_KX {A B C D K L M N P Q X : Point}
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_equilateral_triangle_outside A B K)
  (h3 : is_equilateral_triangle_outside B C L)
  (h4 : is_equilateral_triangle_outside C D M)
  (h5 : is_equilateral_triangle_outside D A N)
  (hP : P = midpoint B L)
  (hQ : Q = midpoint A N)
  (hX : X = circumcenter C M D) :
  is_perpendicular P Q K X :=
sorry

end PQ_perpendicular_to_KX_l588_588496


namespace axis_of_symmetry_for_g_l588_588778

theorem axis_of_symmetry_for_g :
  let f := λ x : ℝ, sin x - cos x
  let g := λ x : ℝ, sqrt 2 * sin ((1/2) * x - 5 * π / 12)
  ∃ k : ℤ, x = 2 * k * π + 11 * π / 6 :=
sorry

end axis_of_symmetry_for_g_l588_588778


namespace find_a_l588_588408

theorem find_a (x : ℝ) (h_a_pos : ∃ a : ℝ, a > 0 ∧ ((2 * x + 6 = real.sqrt a ∨ 2 * x + 6 = -real.sqrt a) ∧ (x - 18 = real.sqrt a ∨ x - 18 = -real.sqrt a))) : 
  x = 4 ∧ ∃ a: ℝ, a = 196 :=
by
  classical
  sorry

end find_a_l588_588408


namespace find_q_conditions_for_negative_roots_l588_588332

def polynomial_has_two_distinct_negative_real_roots (q : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ 
                 (x1^4 + 2*q*x1^3 + 2*x1^2 + 2*q*x1 + 4 = 0) ∧ 
                 (x2^4 + 2*q*x2^3 + 2*x2^2 + 2*q*x2 + 4 = 0)

theorem find_q_conditions_for_negative_roots (q : ℝ) :
  polynomial_has_two_distinct_negative_real_roots q ↔
  q ∈ Ioi (3 * Real.sqrt 2 / 4) :=
sorry

end find_q_conditions_for_negative_roots_l588_588332


namespace area_of_CDE_is_54_l588_588213

-- Define points O, A, B, C, D, and E using coordinates.
def point := (ℝ × ℝ)
def O : point := (0, 0)
def A : point := (4, 0)
def B : point := (16, 0)
def C : point := (16, 12)
def D : point := (4, 12)
def E : point := (4, 3)  -- Midpoint derived from problem's similarity conditions

-- Define the line segment lengths based on the points.
def length (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Define DE and DC
noncomputable def DE : ℝ := 9
noncomputable def DC : ℝ := 12

-- Area of triangle CDE
noncomputable def area_CDE : ℝ := (1 / 2) * DE * DC

-- Theorem statement
theorem area_of_CDE_is_54 : area_CDE = 54 := by
  sorry

end area_of_CDE_is_54_l588_588213


namespace log_y_minus_x_plus_1_pos_l588_588868

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588868


namespace managers_participation_l588_588293

theorem managers_participation (teams : ℕ) (people_per_team : ℕ) (employees : ℕ) (total_people : teams * people_per_team = 6) (num_employees : employees = 3) :
  teams * people_per_team - employees = 3 :=
by
  sorry

end managers_participation_l588_588293


namespace cost_price_per_metre_l588_588279

theorem cost_price_per_metre (total_selling_price : ℕ) (total_metres : ℕ) (loss_per_metre : ℕ)
  (h1 : total_selling_price = 9000)
  (h2 : total_metres = 300)
  (h3 : loss_per_metre = 6) :
  (total_selling_price + (loss_per_metre * total_metres)) / total_metres = 36 :=
by
  sorry

end cost_price_per_metre_l588_588279


namespace sum_of_three_consecutive_integers_product_990_l588_588550

theorem sum_of_three_consecutive_integers_product_990 
  (a b c : ℕ) 
  (h1 : b = a + 1)
  (h2 : c = b + 1)
  (h3 : a * b * c = 990) :
  a + b + c = 30 :=
sorry

end sum_of_three_consecutive_integers_product_990_l588_588550


namespace area_of_shaded_region_l588_588211

axiom OA : ℝ := 4
axiom OB : ℝ := 16
axiom OC : ℝ := 12
axiom similarity (EA CB : ℝ) : EA / CB = OA / OB

theorem area_of_shaded_region (DE DC : ℝ) (h_DE : DE = OC - EA)
    (h_DC : DC = 12) (h_EA_CB : EA = 3) :
    (1 / 2) * DE * DC = 54 := by
  sorry

end area_of_shaded_region_l588_588211


namespace solution_set_for_inequality_l588_588780

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^2 - 2*x - 5

theorem solution_set_for_inequality :
  {x : ℝ | f x >= -2} = {x | -2 <= x ∧ x < 1 ∨ x >= 3} := sorry

end solution_set_for_inequality_l588_588780


namespace area_of_CDE_is_54_l588_588217

-- Define points O, A, B, C, D, and E using coordinates.
def point := (ℝ × ℝ)
def O : point := (0, 0)
def A : point := (4, 0)
def B : point := (16, 0)
def C : point := (16, 12)
def D : point := (4, 12)
def E : point := (4, 3)  -- Midpoint derived from problem's similarity conditions

-- Define the line segment lengths based on the points.
def length (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Define DE and DC
noncomputable def DE : ℝ := 9
noncomputable def DC : ℝ := 12

-- Area of triangle CDE
noncomputable def area_CDE : ℝ := (1 / 2) * DE * DC

-- Theorem statement
theorem area_of_CDE_is_54 : area_CDE = 54 := by
  sorry

end area_of_CDE_is_54_l588_588217


namespace inequality_ln_positive_l588_588858

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588858


namespace convex_polygon_side_condition_l588_588344

theorem convex_polygon_side_condition (n : ℕ) (h : n > 2) :
  (∀ (P : List Point), convex n P → 
    ∃ i, ¬ is_acute (internal_angle P i) ∧ ¬ is_acute (internal_angle P (i + 1) % n))
    ↔ n ≥ 7 :=
by
  sorry

-- Helper definitions that might be necessary for the theorem

def Point := (ℝ × ℝ)

def internal_angle (P : List Point) (i : ℕ) : ℝ := sorry -- appropriate definition for internal angle

def is_acute (θ : ℝ) : Prop := θ < (π / 2)

def convex (n : ℕ) (P : List Point) : Prop := sorry -- appropriate definition for convexity

end convex_polygon_side_condition_l588_588344


namespace pentagon_independent_properties_l588_588112

noncomputable def equiangular (pentagon : Polygon) : Prop :=
  ∀ i j : Fin 5, angle (pentagon.vertex i) (pentagon.vertex j) = 108

noncomputable def equilateral (pentagon : Polygon) : Prop :=
  ∀ i j : Fin 5, side_length (pentagon.vertex i) (pentagon.vertex j) = side_length (pentagon.vertex 0) (pentagon.vertex 1)

theorem pentagon_independent_properties : 
  ∃ (p1 p2 : Polygon), 
    equiangular p1 ∧ ¬equilateral p1 ∧ equilateral p2 ∧ ¬equiangular p2 := 
begin
  sorry
end

end pentagon_independent_properties_l588_588112


namespace e_exp_ax1_ax2_gt_two_l588_588788

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem e_exp_ax1_ax2_gt_two {a x1 x2 : ℝ} (h : a ≠ 0) (h1 : f a x1 = 0) (h2 : f a x2 = 0) (hx : x1 < x2) : 
  Real.exp (a * x1) + Real.exp (a * x2) > 2 :=
sorry

end e_exp_ax1_ax2_gt_two_l588_588788


namespace shaded_area_is_54_l588_588204

-- Define the coordinates of points O, A, B, C, D, E
structure Point where
  x : ℝ
  y : ℝ

-- Given points
def O := Point.mk 0 0
def A := Point.mk 4 0
def B := Point.mk 16 0
def C := Point.mk 16 12
def D := Point.mk 4 12
def E := Point.mk 4 3

-- Define the function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2) ^ (1/2)

-- Define similarity of triangles and calculate side lengths involved
def triangles_similarity (OA OB CB EA : ℝ) : Prop :=
  OA / OB = EA / CB

-- Define the condition
def condition : Prop := 
  triangles_similarity (distance O A) (distance O B) 12 (distance E A) ∧
  distance E A = 3 ∧
  distance D E = 9

-- Define the calculation of area of triangle given base and height
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- State that the area of triangle CDE is 54 cm²
def area_shaded_region : Prop :=
  triangle_area 9 12 = 54

-- Main theorem statement
theorem shaded_area_is_54 : condition → area_shaded_region := by
  sorry

end shaded_area_is_54_l588_588204


namespace new_shoes_last_for_two_years_l588_588632

theorem new_shoes_last_for_two_years :
  let cost_repair := 11.50
  let cost_new := 28.00
  let increase_factor := 1.2173913043478261
  (cost_new / ((increase_factor) * cost_repair)) ≠ 0 :=
by
  sorry

end new_shoes_last_for_two_years_l588_588632


namespace square_area_l588_588574

theorem square_area (side_length : ℕ) (h : side_length = 12) : side_length * side_length = 144 :=
by
  rw h
  norm_num
  sorry

end square_area_l588_588574


namespace balls_in_boxes_problem_l588_588806

theorem balls_in_boxes_problem:
  let num_balls := 5
  let num_boxes := 3
  (∀ (dist: List ℕ), dist.length = num_boxes ∧ dist.sum = num_balls →
    -- Number of distributions considering propositions from the proof steps
    dist = [5,0,0] ∨ dist = [4,1,0] ∨ dist = [3,2,0] ∨ dist = [2,2,1] ∨ dist = [3,1,1]) ↔
      -- Total number of ways is 5
      5 :=
by
  sorry

end balls_in_boxes_problem_l588_588806


namespace proof_fraction_of_sums_l588_588748

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Assuming the sequence is arithmetic-geometric
axiom arithmetic_geometric_seq (n : ℕ) : a (n + 1) = a n * 2 -- since q = 2 

-- Sum of the first n terms
axiom S_n_sum (n : ℕ) : S n = ∑ i in finset.range n, a i

-- Given condition
axiom condition_1 : a 3 - 4 * a 2 + 4 * a 1 = 0

-- Prove that the given problem condition results in the correct answer
theorem proof_fraction_of_sums : S 8 / S 4 = 17 := 
sorry

end proof_fraction_of_sums_l588_588748


namespace imaginary_part_z_l588_588078

open Complex

theorem imaginary_part_z : (im ((i - 1) / (i + 1))) = 1 :=
by
  -- The proof goes here, but it can be marked with sorry for now
  sorry

end imaginary_part_z_l588_588078


namespace problem_solution_l588_588717

noncomputable def factorial : ℕ → ℝ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def term (n : ℕ) : ℝ :=
  n / (factorial (n - 2) + factorial (n - 1) + factorial n)

noncomputable def sum_term : ℝ :=
  ∑ n in Finset.range 20 | (λ n, n + 3), term (n + 3)

theorem problem_solution : sum_term = (1 / 2) - (1 / factorial 22) := by
  sorry

end problem_solution_l588_588717


namespace range_of_m_min_x2_y2_range_k_l588_588744

noncomputable def circle_eq (m : ℝ) : (ℝ×ℝ) -> Prop :=
λ ⟨x, y⟩, x^2 + y^2 + 6*x - 8*y + m = 0

theorem range_of_m (P : ℝ×ℝ) (hP : P = (0, 4)) :
  (∀ m, ¬circle_eq m P) → 16 < m ∧ m < 25 :=
sorry

theorem min_x2_y2 (m : ℝ) (h1 : m = 24) : 
  let C := circle_eq m in
  ∃ (x y : ℝ), C (x, y) ∧ x^2 + y^2 = 24 :=
sorry

theorem range_k (m : ℝ) (h1 : m = 24) (k : ℝ) : 
  let C := circle_eq m in
  (∃ (x y : ℝ), C (x, y) ∧ y = 4 + k*x) → - (Real.sqrt 2) / 4 ≤ k ∧ k ≤ (Real.sqrt 2) / 4 :=
sorry

end range_of_m_min_x2_y2_range_k_l588_588744


namespace find_m_range_l588_588413

theorem find_m_range (m : ℝ) : 
  (∃ x : ℤ, 2 * (x : ℝ) - 1 ≤ 5 ∧ x - 1 ≥ m ∧ x ≤ 3) ∧ 
  (∃ y : ℤ, 2 * (y : ℝ) - 1 ≤ 5 ∧ y - 1 ≥ m ∧ y ≤ 3 ∧ x ≠ y) → 
  -1 < m ∧ m ≤ 0 := by
  sorry

end find_m_range_l588_588413


namespace scientific_notation_560000_l588_588286

theorem scientific_notation_560000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 560000 = a * 10 ^ n ∧ a = 5.6 ∧ n = 5 :=
by 
  sorry

end scientific_notation_560000_l588_588286


namespace bread_per_day_baguettes_per_day_croissants_per_day_l588_588952

-- Define the conditions
def loaves_per_hour : ℕ := 10
def hours_per_day : ℕ := 6
def baguettes_per_2hours : ℕ := 30
def croissants_per_75minutes : ℕ := 20

-- Conversion factors
def minutes_per_hour : ℕ := 60
def minutes_per_block : ℕ := 75
def blocks_per_75minutes : ℕ := 360 / 75

-- Proof statements
theorem bread_per_day :
  loaves_per_hour * hours_per_day = 60 := by sorry

theorem baguettes_per_day :
  (hours_per_day / 2) * baguettes_per_2hours = 90 := by sorry

theorem croissants_per_day :
  (blocks_per_75minutes * croissants_per_75minutes) = 80 := by sorry

end bread_per_day_baguettes_per_day_croissants_per_day_l588_588952


namespace sequence_values_l588_588054

theorem sequence_values (x y z : ℚ) :
  (∀ n : ℕ, x = 1 ∧ y = 9 / 8 ∧ z = 5 / 4) :=
by
  sorry

end sequence_values_l588_588054


namespace bounded_parabola_contains_at_most_two_integers_l588_588451

theorem bounded_parabola_contains_at_most_two_integers (a b c : ℝ) (h1 : |a| > 2) :
  ∀ (f : ℝ → ℝ), (f = λ x, a * x^2 + b * x + c) → 
  let S := {x : ℤ | -1 ≤ f x ∧ f x ≤ 1} in S.finite ∧ S.card ≤ 2 := 
begin
  intros f hf,
  sorry
end

end bounded_parabola_contains_at_most_two_integers_l588_588451


namespace log_pos_if_exp_diff_l588_588882

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588882


namespace probability_seg_not_less_than_one_l588_588640

variable (X : ℝ) (h_uniform : ∀ x, 0 ≤ x ∧ x ≤ 3 → (X.ProbabilityDensityFunction x) = 1/3)
noncomputable def prob_not_less_than_one : ℝ :=
  ∫ x in 1..2, X.ProbabilityDensityFunction x

theorem probability_seg_not_less_than_one :
  (prob_not_less_than_one X h_uniform) = 1 / 3 :=
sorry

end probability_seg_not_less_than_one_l588_588640


namespace triangle_side_lengths_and_angles_l588_588526

theorem triangle_side_lengths_and_angles
  (area: ℝ)
  (ha: area = 1470)
  (a b c : ℝ)
  (h_ratio: 13 * 35 = 91)
  (h_ratio_bc : b = 84)
  (h_ratio_c: c = 35)
  (β γ : ℝ)
  (h_beta: β = real.arctan (91 / 84) )
  (h_gamma: γ = 90 - β) :
  a = 91 ∧ β = 67 + 22 / 60 + 48 / 3600 ∧ γ = 22 + 37 / 60 + 12 / 3600 := 
sorry

end triangle_side_lengths_and_angles_l588_588526


namespace intersections_convex_polygon_l588_588990

open Set

-- Definitions to establish the problem context
def points (n : ℕ) := fin n → ℝ × ℝ

def L_shape_segments (n : ℕ) (A : ℝ × ℝ) (A' : ℝ × ℝ) (a : points n) (a' : points n) :=
  ∀ i : fin n, LineThrough (A, a i) ∧ LineThrough (A', a' i)

-- Given the conditions:
variables {n : ℕ}
variables {A A' : ℝ × ℝ}
variables {a a' : points n}

-- Intersections of lines
def intersections (A A' : ℝ × ℝ) (a a' : points n) : points n :=
  λ i, intersection_of_lines (A, a i) (A', a' i)

-- Prove that these intersections form a convex polygon
theorem intersections_convex_polygon 
  (segments : L_shape_segments n A A' a a')
  : isConvexPolygon (image (intersections A A' a a') univ) :=
sorry

end intersections_convex_polygon_l588_588990


namespace grid_inkblots_l588_588037

theorem grid_inkblots :
  ∃ (blots : fin 5 → fin 5 → Prop),
    (∑ i j, if blots i j then 1 else 0) = 7 ∧
    ∀ (r1 r2 c1 c2 : fin 5),
    r1 ≠ r2 → c1 ≠ c2 → 
    ∃ i j, blots i j ∧ i ≠ r1 ∧ i ≠ r2 ∧ j ≠ c1 ∧ j ≠ c2 :=
by sorry

end grid_inkblots_l588_588037


namespace incenter_bisects_angle_l588_588246

open EuclideanGeometry

variables {A B C D E F I L M T : Point}

-- Assume the necessary conditions about the geometric configuration
axiom incenter (I : Point) (ΔABC : Triangle) : incenter I ΔABC
axiom angle_bisectors (ΔABC : Triangle) (D E F : Point) : 
  angleBisector D ΔABC.sideBC ∧
  angleBisector E ΔABC.sideCA ∧
  angleBisector F ΔABC.sideAB
axiom intersection_circumcircle (E F : Point) (circumcircle : Circle) : 
  ∃ (L T : Point), L ∈ circumcircle ∧ T ∈ circumcircle ∧ lies_on_segment F L E
axiom midpointBC (ΔABC : Triangle) (M : Point) : midpoint M ΔABC.sideBC
axiom tangent (D T : Point) (incircle : Circle) : tangent DT incircle

-- Translate our proof problem now
theorem incenter_bisects_angle
  (h1 : incenter I (triangle A B C))
  (h2 : ∀ (P : Point), P ∈ {D, E, F} → angle_bisectors (triangle A B C) D E F)
  (h3 : ∃ (L T : Point), (L ∈ circumcircle (triangle A B C)) ∧ (T ∈ circumcircle (triangle A B C)) ∧ lies_on_segment F L E)
  (h4 : midpoint M (side BC (triangle A B C)))
  (h5 : tangent (D T) (incircle (triangle A B C))) :
  bisects_angle ILM T :=
by
  sorry

end incenter_bisects_angle_l588_588246


namespace max_contribution_l588_588018

theorem max_contribution (n : ℕ) (total : ℝ) (min_contribution : ℝ)
  (h1 : n = 12) (h2 : total = 20) (h3 : min_contribution = 1)
  (h4 : ∀ i : ℕ, i < n → min_contribution ≤ min_contribution) :
  ∃ max_contrib : ℝ, max_contrib = 9 :=
by
  sorry

end max_contribution_l588_588018


namespace LN_eq_MN_l588_588345

noncomputable def triangle (A B C : Point) : Triangle := sorry
noncomputable def median (A B C : Point) : Line := sorry
noncomputable def perpendicular_from_point (P : Point) (L : Line) : Line := sorry
noncomputable def intersection (L1 L2 : Line) : Point := sorry
noncomputable def distance (P1 P2 : Point) : ℝ := sorry

variable {A B C P L M N : Point}

theorem LN_eq_MN : 
  ∀ (ABC : Triangle) (P : Point),
  let L := intersection (perpendicular_from_point P (line A B)) (line A B),
  let M := intersection (perpendicular_from_point P (line A C)) (line A C),
  let E := midpoint A (midpoint B C),
  let N := intersection (perpendicular_from_point P (line A E)) (line A D) in
  distance L N = distance M N :=
sorry

end LN_eq_MN_l588_588345


namespace range_of_x_l588_588556

variable (x : ℝ)

-- Conditions used in the problem
def sqrt_condition : Prop := x + 2 ≥ 0
def non_zero_condition : Prop := x + 1 ≠ 0

-- The statement to be proven
theorem range_of_x : sqrt_condition x ∧ non_zero_condition x ↔ (x ≥ -2 ∧ x ≠ -1) :=
by
  sorry

end range_of_x_l588_588556


namespace exists_real_number_x_l588_588477

theorem exists_real_number_x (f : ℕ → ℕ) (h : ∀ i j : ℕ, 1 ≤ i ∧ 1 ≤ j ∧ i + j ≤ 1997 → f(i) + f(j) ≤ f(i + j) ∧ f(i + j) ≤ f(i) + f(j) + 1) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1997 → f(n) = Int.floor (n * x) :=
  sorry

end exists_real_number_x_l588_588477


namespace n_power_2020_plus_4_composite_l588_588110

theorem n_power_2020_plus_4_composite {n : ℕ} (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b := 
by
  sorry

end n_power_2020_plus_4_composite_l588_588110


namespace eighth_hexagonal_number_l588_588525

theorem eighth_hexagonal_number : (8 * (2 * 8 - 1)) = 120 :=
  by
  sorry

end eighth_hexagonal_number_l588_588525


namespace tangent_line_at_one_l588_588785

open Real

def f (x : ℝ) : ℝ := x - 4 * ln x

theorem tangent_line_at_one :
  ∃ m b, (∀ x, m * x + b = 3 * x + (f 1) - 4) ∧ (m = -3) :=
by{
  sorry
}

end tangent_line_at_one_l588_588785


namespace log_49_48_proof_l588_588367

variable (a b : ℝ)
variable (conditions : (1 / 7) ^ a = (1 / 3) ∧ Real.log 7 4 = b)

noncomputable def log_49_48_in_terms_of_a_b : ℝ :=
  if (1 / 7) ^ a = (1 / 3) ∧ Real.log 7 4 = b then
    (a + 2 * b) / 2
  else
    0

theorem log_49_48_proof : log_49_48_in_terms_of_a_b a b = Real.log 49 48 := by
  sorry

end log_49_48_proof_l588_588367


namespace number_of_monotone_functions_is_binom_l588_588147

variable (n : ℕ) (A : Finset ℕ) (B : Finset ℕ)
  (f : ℕ → ℕ)
  (hA : A = Finset.range (n + 1).filter (λ x, x > 0))
  (hB : B = Finset.range (2 * n + 1).filter (λ x, x > 0))

theorem number_of_monotone_functions_is_binom :
  (∃ f : ℕ → ℕ, InjOn f A ∧ MonotoneOn f A ∧ (∀ x ∈ A, f x ∈ B)) →
  (Finset.card {g : ℕ → ℕ | (∀ x ∈ A, g x ∈ B) ∧ MonotoneOn g A} = Nat.choose (2 * n) n) :=
sorry

end number_of_monotone_functions_is_binom_l588_588147


namespace total_shopping_cost_l588_588288

theorem total_shopping_cost 
  (sandwiches : ℕ := 3)
  (sandwich_cost : ℕ := 3)
  (water_bottle : ℕ := 1)
  (water_cost : ℕ := 2)
  : sandwiches * sandwich_cost + water_bottle * water_cost = 11 :=
by
  sorry

end total_shopping_cost_l588_588288


namespace parabola_focus_and_directrix_length_ab_l588_588389

theorem parabola_focus_and_directrix_length_ab (y x : ℝ) (L : ℝ → ℝ)
  (h_parabola : y^2 = 6 * x)
  (h_line_L : L x = x - 3 / 2)
  (h_L_passes_focus : ∃ F, F = (3 / 2, 0) ∧ L F.1 = F.2) :
  (∀ F, F = (3 / 2, 0) → L F.1 = F.2 ∧ focus parabola = F) ∧
  (∀ d, directrix parabola = -3 / 2) ∧
  (∀ A B, intersects parabola L A B → length_of_chord A B = 12) := 
by 
  sorry

end parabola_focus_and_directrix_length_ab_l588_588389


namespace volume_of_parallelepiped_l588_588530

noncomputable def sqrt_3 : ℝ := Real.sqrt 3

theorem volume_of_parallelepiped (a : ℝ) (θ : ℝ) (V : ℝ) 
(h1 : a = 2 * sqrt_3) (h2 : θ = Real.pi / 6) (h3 : V = 72) : 
V = a^2 * (a * Real.cot θ) := 
by
  sorry

end volume_of_parallelepiped_l588_588530


namespace not_prime_sum_of_squares_l588_588972

theorem not_prime_sum_of_squares 
  (a b c d : ℤ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h : a * b = c * d) : ¬ prime (a^2 + b^2 + c^2 + d^2) :=
by
  sorry

end not_prime_sum_of_squares_l588_588972


namespace circle_radius_zero_l588_588701

theorem circle_radius_zero (x y : ℝ) :
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → ∃ c : ℝ × ℝ, ∃ r : ℝ, (x - c.1)^2 + (y - c.2)^2 = r^2 ∧ r = 0 :=
by
  sorry

end circle_radius_zero_l588_588701


namespace max_sections_l588_588940

def rectangle : Type := sorry
def line_segment : Type := sorry
variable (l1 l2 l3 l4 l5 : line_segment)
variable (PQ : line_segment)

def intersects (l1 l2 : line_segment) : Prop := sorry -- defines intersection property

theorem max_sections (rect : rectangle) (PQ l1 l2 l3 l4 l5) :
  let lines := [PQ, l1, l2, l3, l4];
  ∀ l ∈ lines, intersects PQ l →
  intersects l1 l ∧ intersects l2 l ∧ intersects l3 l ∧ intersects l4 l →
  number_sections rect lines = 16 :=
sorry

end max_sections_l588_588940


namespace symmetry_center_of_tangent_l588_588565

theorem symmetry_center_of_tangent (k : ℤ) : 
  ∃ x, y = tan (π * x + π / 4) → (x, y) = ( (2*k-1)/4, 0 ) := 
begin
  sorry
end

end symmetry_center_of_tangent_l588_588565


namespace circumscribed_n_gon_l588_588264

theorem circumscribed_n_gon (n : ℕ) (h1 : n > 3) : 
  (n = 4 ∨ n > 5) → 
  ∃ (triangles : set (triangle ℝ)), 
    (∀ t ∈ triangles, ∃ t' ∈ triangles, t ≠ t' ∧ t ~ t') :=
by sorry

end circumscribed_n_gon_l588_588264


namespace sqrt_inequality_abc_pos_l588_588020

theorem sqrt_inequality_abc_pos (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  sqrt ((a + c) * (b + d)) ≥ sqrt (a * b) + sqrt (c * d) :=
sorry

end sqrt_inequality_abc_pos_l588_588020


namespace digit_6_or_7_count_in_base8_l588_588005

def counts_digit_in_base (n : ℕ) (base : ℕ) (digit : ℕ) : ℕ :=
  if h : base > digit then
    (Nat.digits base n).count (λ d, d = digit)
  else 0

def has_digit_6_or_7_in_base8 (n : ℕ) : Prop :=
  counts_digit_in_base n 8 6 > 0 ∨ counts_digit_in_base n 8 7 > 0

theorem digit_6_or_7_count_in_base8 :
  (Finset.range 256).filter has_digit_6_or_7_in_base8).card = 10 := by
  sorry

end digit_6_or_7_count_in_base8_l588_588005


namespace sqrt_sum_eq_pm3_l588_588739

theorem sqrt_sum_eq_pm3 (x y : ℝ) (h1 : x = 3 / 2) (h2 : y = sqrt (2 * x - 3) + sqrt (3 - 2 * x) + 5) :
  sqrt (x + y + 5 / 2) = 3 ∨ sqrt (x + y + 5 / 2) = -3 :=
by
  sorry

end sqrt_sum_eq_pm3_l588_588739


namespace value_of_a_parity_of_function_monotonicity_on_interval_l588_588779

noncomputable def func (x : ℝ) (a : ℝ) := x + a / x

theorem value_of_a : (func 1 a = 5) -> a = 4 :=
by 
  intro h,
  calc
    5 = func 1 a :- sorry

theorem parity_of_function : ∀ (x : ℝ), func (-x) 4 = - (func x 4) :=
by 
  intro x,
  calc
    func (-x) 4 = -x + 4 / -x :- sorry

theorem monotonicity_on_interval (x₁ x₂ : ℝ) (h₁ : 2 < x₁) (h₂ : x₁ < x₂) : func x₁ 4 < func x₂ 4 :=
by 
  sorry

end value_of_a_parity_of_function_monotonicity_on_interval_l588_588779


namespace min_a_dot_b_l588_588244

noncomputable def a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)
def k (k : ℝ) : Prop := k > 0

theorem min_a_dot_b {α β : ℝ} (h : k k) :
  let a := a α
  let b := b β
  (|k * a + b| - sqrt 3 * |a - k * b|) = 0 → 
  (a.1 * b.1 + a.2 * b.2) = 1 / 2 :=
sorry

end min_a_dot_b_l588_588244


namespace citizen_income_l588_588234

theorem citizen_income (tax_paid : ℝ) (base_income : ℝ) (base_rate excess_rate : ℝ) (income : ℝ) 
  (h1 : 0 < base_income) (h2 : base_rate * base_income = 4400) (h3 : tax_paid = 8000)
  (h4 : excess_rate = 0.20) (h5 : base_rate = 0.11)
  (h6 : tax_paid = base_rate * base_income + excess_rate * (income - base_income)) :
  income = 58000 :=
sorry

end citizen_income_l588_588234


namespace trig_shift_l588_588310

theorem trig_shift (x : ℝ) : ∃ k : ℝ, y = sin (2 * (x - k)) ↔ y = sin 2x - π/3 :=
sorry

end trig_shift_l588_588310


namespace length_of_segment_eq_ten_l588_588586

theorem length_of_segment_eq_ten (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  let y1 := real.cbrt 27 + 5,
      y2 := real.cbrt 27 - 5
  in abs (y1 - y2) = 10 := 
by
  sorry

end length_of_segment_eq_ten_l588_588586


namespace inequality_proof_l588_588504

theorem inequality_proof (a b : ℝ) (h : (a = 0 ∨ b = 0 ∨ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0))) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 :=
by
  sorry

end inequality_proof_l588_588504


namespace impossible_card_arrangement_l588_588391

-- Define the given conditions: 20 cards, each digit between 0 to 9 appearing twice, specific spacing conditions between identical digits.
theorem impossible_card_arrangement :
  let zeros := 2,
      ones := 3,
      twos := 4,
      threes := 5,
      fours := 6,
      fives := 7,
      sixes := 8,
      sevens := 9,
      eights := 10,
      nines := 11 in
  zeros + ones + twos + threes + fours + fives + sixes + sevens + eights + nines > 20 :=
by
  -- Summing the required spaces
  let total_space := 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11
  show total_space > 20, from
  calc
    total_space = 55 : by norm_num
    55 > 20 : by norm_num

end impossible_card_arrangement_l588_588391


namespace polygon_area_is_54_l588_588323

-- Define the coordinates corresponding to the vertices of the polygon
def vertices : List (ℝ × ℝ) := [(0,0), (3,0), (6,0), (6,3), (9,3), (9,6), (6,6), 
                                (6,9), (3,9), (3,6), (0,6), (0,3), (0,0)]

-- Define a function to calculate the area of a polygon given its vertices
noncomputable def polygon_area (verts : List (ℝ × ℝ)) : ℝ :=
  (1 / 2 : ℝ) * (List.sum (List.map (λ (i : ℕ),
    (verts.get! i).1 * (verts.get! ((i + 1) % verts.length)).2 -
    (verts.get! ((i + 1) % verts.length)).1 * (verts.get! i).2)
      (List.range verts.length)))

-- State the theorem that the area of the polygon is 54 square units
theorem polygon_area_is_54 : 
  polygon_area vertices = 54 :=
by
  sorry

end polygon_area_is_54_l588_588323


namespace sugar_needed_for_40_cookies_l588_588953

def num_cookies_per_cup_flour (a : ℕ) (b : ℕ) : ℕ := a / b

def cups_of_flour_needed (num_cookies : ℕ) (cookies_per_cup : ℕ) : ℕ := num_cookies / cookies_per_cup

def cups_of_sugar_needed (cups_flour : ℕ) (flour_to_sugar_ratio_num : ℕ) (flour_to_sugar_ratio_denom : ℕ) : ℚ := 
  (flour_to_sugar_ratio_denom * cups_flour : ℚ) / flour_to_sugar_ratio_num

theorem sugar_needed_for_40_cookies :
  let num_flour_to_make_24_cookies := 3
  let cookies := 24
  let ratio_num := 3
  let ratio_denom := 2
  num_cookies_per_cup_flour cookies num_flour_to_make_24_cookies = 8 →
  cups_of_flour_needed 40 8 = 5 →
  cups_of_sugar_needed 5 ratio_num ratio_denom = 10 / 3 :=
by 
  sorry

end sugar_needed_for_40_cookies_l588_588953


namespace fraction_correct_l588_588232

variables (A : ℝ) (J A S : ℝ)

/-- Define the tips for July -/
def tips_July : ℝ := 10 * A

/-- Define the tips for August -/
def tips_August : ℝ := 15 * A

/-- Define the tips for September -/
def tips_September : ℝ := 0.75 * (15 * A)

/-- Define the total tips for the months other than July and August -/
def total_other_months : ℝ := 8 * A

/-- Total tips for the period he worked -/
def total_tips : ℝ := total_other_months + tips_July + tips_August + tips_September

/-- Fraction of total tips for July, August, and September compared to the entire period -/
def fraction_tips : ℝ := (tips_July + tips_August + tips_September) / total_tips

theorem fraction_correct :
  fraction_tips = 36.25 / 44.25 :=
by
  sorry

end fraction_correct_l588_588232


namespace part_one_part_two_l588_588737

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part_one (x : ℝ) : f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
begin
  sorry
end

theorem part_two {a b : ℝ} (h_cond1 : 0 < b) (h_cond2 : b < 1/2) (h_cond3 : 1/2 < a)
  (h_eq : f a = 3 * f b) : ∃ m : ℤ, a^2 + b^2 > m ∧ m = 2 :=
begin
  sorry
end

end part_one_part_two_l588_588737


namespace dartboard_arrangements_l588_588653

theorem dartboard_arrangements : 
  ∃ (lists : Finset (Fin 5 → ℕ)), 
    (∀ l ∈ lists, (l.sum = 5 ∧ (∀ i j, i ≤ j → l j ≤ l i))) ∧ 
    lists.card = 7 :=
sorry

end dartboard_arrangements_l588_588653


namespace percentage_non_sugar_pie_l588_588274

theorem percentage_non_sugar_pie (total_weight sugar_weight : ℕ) 
  (h_total : total_weight = 200) 
  (h_sugar : sugar_weight = 50) : 
  ((total_weight - sugar_weight) * 100 / total_weight) = 75 := 
by 
  -- given conditions
  rw [h_total, h_sugar]
  -- simplify the expression 
  norm_num
  sorry

end percentage_non_sugar_pie_l588_588274


namespace triangle_is_isosceles_l588_588440

variable {Point : Type} [Field Point]

structure Triangle :=
(A B C : Point)

def midpoint (A B : Point) : Point := sorry -- Placeholder for the actual midpoint computation

def is_isosceles (T : Triangle) : Prop :=
(T.A = T.B) ∨ (T.B = T.C) ∨ (T.A = T.C)

axiom median (T : Triangle) (M : Point) : midpoint T.B T.C = M 

axiom angle_bisector (T : Triangle) (M : Point) : true -- Placeholder for the actual angle bisector property

theorem triangle_is_isosceles (T : Triangle) (M : Point) (h1 : midpoint T.B T.C = M) (h2 : angle_bisector T M) :
  is_isosceles T :=
sorry

end triangle_is_isosceles_l588_588440


namespace smallest_pythagorean_sum_square_l588_588014

theorem smallest_pythagorean_sum_square (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) :
  ∃ (k : ℤ), k = 4 ∧ (p + q + r)^2 ≥ k :=
by
  sorry

end smallest_pythagorean_sum_square_l588_588014


namespace area_of_enclosed_triangle_l588_588285

noncomputable def area_of_triangle (b h : ℝ) : ℝ :=
1 / 2 * b * h

noncomputable def base_length (x1 x2 : ℝ) : ℝ :=
x2 - x1

noncomputable def height (y1 y2 : ℝ) : ℝ :=
y1 - y2

-- Coordinates of the vertices from the solution:
def vertex1 : ℝ × ℝ := (1 / 2, 2)
def vertex2 : ℝ × ℝ := (4, 2)
def vertex3 : ℝ × ℝ := (6 / 5, 17 / 5)

-- Base along y = 2
def base : ℝ := base_length (vertex1.1) (vertex2.1)

-- Height from vertex3 to line y = 2
def height_triangle : ℝ := height (vertex3.2) 2

-- The area of the triangle
def area_triangle : ℝ := area_of_triangle base height_triangle

-- Statement to prove
theorem area_of_enclosed_triangle : area_triangle = 2.45 :=
sorry

end area_of_enclosed_triangle_l588_588285


namespace largest_angle_at_least_60_degrees_l588_588543

noncomputable def equilateral_triangle (ABC : Type) :=
  ∃ (A B C : ABC), ∀ (X Y : ABC), dist A B = dist B C ∧ dist B C = dist C A ∧
                   dist C A = dist A B

variable {P : ℝ × ℝ}

theorem largest_angle_at_least_60_degrees
  (ABC : Type) [metric_space ABC] [hilbert_space ABC]
  (A B C : ABC)
  (h1 : equilateral_triangle ABC)
  (C1 : midpoint A B)
  (A1 : midpoint B C)
  (B1 : midpoint C A)
  (P : ABC) :
  ∃ (largest_angle : ℝ), largest_angle ≥ 60 := 
sorry

end largest_angle_at_least_60_degrees_l588_588543


namespace binomial_log_inequality_l588_588119

theorem binomial_log_inequality (n : ℤ) :
  n * Real.log 2 ≤ Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ∧ 
  Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ≤ n * Real.log 4 :=
by sorry

end binomial_log_inequality_l588_588119


namespace shaded_area_of_triangle_CDE_l588_588197

-- Definitions of the points
noncomputable def O := (0, 0 : ℝ×ℝ)
noncomputable def A := (4, 0 : ℝ×ℝ)
noncomputable def B := (16, 0 : ℝ×ℝ)
noncomputable def C := (16, 12 : ℝ×ℝ)
noncomputable def D := (4, 12 : ℝ×ℝ)
noncomputable def E := (4, 3 : ℝ×ℝ)

-- Definition of the area calculation for the given triangle
theorem shaded_area_of_triangle_CDE : 
  let DE := 9 in
  let DC := 12 in
  (DE * DC) / 2 = 54 :=
by
  sorry

end shaded_area_of_triangle_CDE_l588_588197


namespace log_pos_if_exp_diff_l588_588877

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588877


namespace maximize_profit_l588_588262

noncomputable def R (x : ℝ) : ℝ := 
  if x ≤ 40 then
    40 * x - (1 / 2) * x^2
  else
    1500 - 25000 / x

noncomputable def cost (x : ℝ) : ℝ := 2 + 0.1 * x

noncomputable def f (x : ℝ) : ℝ := R x - cost x

theorem maximize_profit :
  ∃ x : ℝ, x = 50 ∧ f 50 = 300 := by
  sorry

end maximize_profit_l588_588262


namespace function_fixed_point_l588_588537

theorem function_fixed_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
by
  sorry

end function_fixed_point_l588_588537


namespace total_shopping_cost_l588_588287

theorem total_shopping_cost 
  (sandwiches : ℕ := 3)
  (sandwich_cost : ℕ := 3)
  (water_bottle : ℕ := 1)
  (water_cost : ℕ := 2)
  : sandwiches * sandwich_cost + water_bottle * water_cost = 11 :=
by
  sorry

end total_shopping_cost_l588_588287


namespace simplify_complex_expression_l588_588127

theorem simplify_complex_expression : 
  (2 * (5 + (complex.mk 0 1)) + (complex.mk 0 1) * ((-2) + -(complex.mk 0 1))) = 11 := by 
  sorry

end simplify_complex_expression_l588_588127


namespace art_of_passing_through_walls_l588_588028

theorem art_of_passing_through_walls (n : ℕ) :
  (2 * Real.sqrt (2 / 3) = Real.sqrt (2 * (2 / 3))) ∧
  (3 * Real.sqrt (3 / 8) = Real.sqrt (3 * (3 / 8))) ∧
  (4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15))) ∧
  (5 * Real.sqrt (5 / 24) = Real.sqrt (5 * (5 / 24))) →
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) →
  n = 63 :=
by
  sorry

end art_of_passing_through_walls_l588_588028


namespace correct_props_l588_588941

-- Condition definitions
def prop_1_condition (x y : ℝ) : Prop := (x > 0 ∧ y > 0 ∧ (x^2 + (y-4)^2) = ((x+2)^2 + y^2))
def prop_2_condition (y y1 z : ℝ) : Prop := (y * y1 > 0 ∧ (y1 - y)^2 + (1 - z)^2 = y^2 + z^2)
def prop_3_condition (x y z a b c : ℝ) : Prop := (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x/a + y/b + z/c = 1)
def prop_4_condition (x y y1 x2 : ℝ) : Prop := (x ∈ set.Icc 0 4 ∧ y ∈ set.Icc (-4) 4 ∧ y1 ∈ set.Icc (-4) 4 ∧ x2 ∈ set.Icc 0 4 ∧ ((y1 = y) ∧ (x2 = x) ∧ (x^2 + y^2 - 1 = 1 - x^2)))

-- Proposition definitions
def prop_1 (x y : ℝ) : Prop := ∃ x y, prop_1_condition x y → x/y + 1/(2 * y) = 2 * real.sqrt 2
def prop_2 (y y1 z : ℝ) : Prop := ∀ y y1 z, prop_2_condition y y1 z → (y^2 = 1 - 2 * z)
def prop_3 (x y z a b c : ℝ) : Prop := ∀ x y z a b c, prop_3_condition x y z a b c → (x / a + y / b + z / c = 1)
def prop_4 (x y y1 x2 : ℝ) : Prop := ∀ x y y1 x2, prop_4_condition x y y1 x2 → (y^2 - x^2 = 1)

-- Proof problem
theorem correct_props (x y y1 z a b c x2 : ℝ) :
  ¬ prop_1 x y ∧ prop_2 y y1 z ∧ prop_3 x y z a b c ∧ prop_4 x y y1 x2 :=
by
  split
  {
    intro h,
    sorry -- Proof for prop_1 being incorrect
  }
  {
    split
    {
      intro hy,
      sorry -- Proof for prop_2 being correct
    }
    {
      split
      {
        intro hz,
        sorry -- Proof for prop_3 being correct
      }
      {
        intro hx2,
        sorry -- Proof for prop_4 being correct
      }
    }
  }

end correct_props_l588_588941


namespace amanda_drawer_pulls_replacement_l588_588178

def drawer_pulls_cost (knob_count : ℕ) (knob_price : ℝ) (pull_price : ℝ) (total_cost : ℝ) : ℕ :=
  let knob_cost := knob_count * knob_price
  let remaining_cost := total_cost - knob_cost
  remaining_cost / pull_price

theorem amanda_drawer_pulls_replacement :
  drawer_pulls_cost 18 2.50 4.00 77 = 8 :=
by
  sorry

end amanda_drawer_pulls_replacement_l588_588178


namespace max_value_of_f_on_interval_l588_588787

noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x)) / (2 * Real.sin ((Real.pi / 2) - 2 * x))

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 6), f x = (Real.sqrt 3) / 2 := sorry

end max_value_of_f_on_interval_l588_588787


namespace initially_caught_and_tagged_fish_l588_588038

theorem initially_caught_and_tagged_fish (N T : ℕ) (hN : N = 800) (h_ratio : 2 / 40 = T / N) : T = 40 :=
by
  have hN : N = 800 := hN
  have h_ratio : 2 / 40 = T / 800 := by rw [hN] at h_ratio; exact h_ratio
  sorry

end initially_caught_and_tagged_fish_l588_588038


namespace rigid_motions_count_l588_588535

theorem rigid_motions_count (pattern : ℕ → ℕ → bool) (ℓ : ℕ → ℕ) : 
  (count (λ t, maps_to_itself pattern t) rigid_motions - 1) = 2 :=
by
  sorry

def rigid_motions : List (Transformation ℕ) :=
[rotation_180, translation_parallel, reflection_across_ℓ, reflection_perpendicular_ℓ, identity_transformation]

def maps_to_itself (pattern : ℕ → ℕ → bool) (t : Transformation ℕ) : Prop :=
-- Define the condition for a transformation to map the pattern onto itself
sorry

def rotation_180 : Transformation ℕ := 
-- Define rotation_180 here
sorry

def translation_parallel : Transformation ℕ := 
-- Define translation_parallel here
sorry

def reflection_across_ℓ : Transformation ℕ := 
-- Define reflection_across_ℓ here
sorry

def reflection_perpendicular_ℓ : Transformation ℕ := 
-- Define reflection_perpendicular_ℓ here
sorry

def identity_transformation : Transformation ℕ := 
-- Define identity_transformation here
sorry

end rigid_motions_count_l588_588535


namespace num_non_decreasing_codes_l588_588292

theorem num_non_decreasing_codes : 
  ∃ n : ℕ, n = 220 ∧ 
           ∀ (a b c : ℕ), 0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 9 → 
                          (∃ seq, seq = (a, b, c) ∧ seq ∈ ({ (a, b, c) : ℕ × ℕ × ℕ | 0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 9 }) 
                          ∧ (finset.card ({(a, b, c) : ℕ × ℕ × ℕ | 0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 9} : finset (ℕ × ℕ × ℕ)) = 220) :=
by
  -- The formal proof will go here
  sorry

end num_non_decreasing_codes_l588_588292


namespace parallel_lines_a_eq_neg1_l588_588393

-- Define the two lines
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a, 2, 6)
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (1, a-1, 3)

-- Define the condition that two lines are parallel
def lines_parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  l1.1 * l2.2 = l1.2 * l2.1

-- State the theorem to be proven
theorem parallel_lines_a_eq_neg1 (a : ℝ) :
  lines_parallel (line1 a) (line2 a) → a = -1 :=
by
  sorry

end parallel_lines_a_eq_neg1_l588_588393


namespace solution_to_system_of_equations_l588_588133

theorem solution_to_system_of_equations :
  ∃ x y : ℤ, 4 * x - 3 * y = 11 ∧ 2 * x + y = 13 ∧ x = 5 ∧ y = 3 :=
by
  sorry

end solution_to_system_of_equations_l588_588133


namespace inradius_is_correct_l588_588429

def isosceles_right_triangle (A B C : Type) :=
  ∃ (PQ PR QR : ℝ), PQ = 10 ∧ PR = 10 ∧ QR = 10 * Real.sqrt 2 ∧
  angle A B C = 90

def inradius_of_inscribed_circle {A B C : Type} (PQ PR QR : ℝ) : ℝ :=
  let area := (1/2) * PQ * PR in
  let semiperimeter := (PQ + PR + QR) / 2 in
  area / semiperimeter

theorem inradius_is_correct
  (A B C : Type)
  (PQ PR QR : ℝ)
  (h1 : PQ = 10)
  (h2 : PR = 10)
  (h3 : QR = 10 * Real.sqrt 2)
  (h_angle : angle A B C = 90) :
  inradius_of_inscribed_circle PQ PR QR = 10 - 5 * Real.sqrt 2 :=
by
  sorry

end inradius_is_correct_l588_588429


namespace log_pos_if_exp_diff_l588_588880

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588880


namespace martha_initial_apples_l588_588484

theorem martha_initial_apples :
  ∀ (jane_apples james_apples keep_apples more_to_give initial_apples : ℕ),
    jane_apples = 5 →
    james_apples = jane_apples + 2 →
    keep_apples = 4 →
    more_to_give = 4 →
    initial_apples = jane_apples + james_apples + keep_apples + more_to_give →
    initial_apples = 20 :=
by
  intros jane_apples james_apples keep_apples more_to_give initial_apples
  intro h_jane
  intro h_james
  intro h_keep
  intro h_more
  intro h_initial
  exact sorry

end martha_initial_apples_l588_588484


namespace triangle_sides_inequality_l588_588013

theorem triangle_sides_inequality
  {a b c : ℝ} (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0)
  (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  -- We would place the proof here if it were required
  sorry

end triangle_sides_inequality_l588_588013


namespace arithmetic_mean_solutions_l588_588745

noncomputable def tau (n : ℕ) : ℕ := n.divisors.card
noncomputable def phi (n : ℕ) : ℕ := n.totient

theorem arithmetic_mean_solutions (n : ℕ) (h : 0 < n) :
    (n = (tau n + phi n) / 2) ∨ (tau n = (n + phi n) / 2) ∨ (phi n = (n + tau n) / 2) ↔
    n = 1 ∨ n = 4 ∨ n = 6 ∨ n = 9 :=
by sorry

end arithmetic_mean_solutions_l588_588745


namespace simplify_expression_l588_588514

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end simplify_expression_l588_588514


namespace length_of_segment_eq_ten_l588_588587

theorem length_of_segment_eq_ten (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  let y1 := real.cbrt 27 + 5,
      y2 := real.cbrt 27 - 5
  in abs (y1 - y2) = 10 := 
by
  sorry

end length_of_segment_eq_ten_l588_588587


namespace inequality_implies_log_pos_l588_588821

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588821


namespace conversion_points_worth_two_l588_588947

theorem conversion_points_worth_two
  (touchdowns_per_game : ℕ := 4)
  (points_per_touchdown : ℕ := 6)
  (games_in_season : ℕ := 15)
  (total_touchdowns_scored : ℕ := touchdowns_per_game * games_in_season)
  (total_points_from_touchdowns : ℕ := total_touchdowns_scored * points_per_touchdown)
  (old_record_points : ℕ := 300)
  (points_above_record : ℕ := 72)
  (total_points_scored : ℕ := old_record_points + points_above_record)
  (conversions_scored : ℕ := 6)
  (total_points_from_conversions : ℕ := total_points_scored - total_points_from_touchdowns) :
  total_points_from_conversions / conversions_scored = 2 := by
sorry

end conversion_points_worth_two_l588_588947


namespace intersect_M_N_l588_588000

open Set

noncomputable def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }
noncomputable def N : Set ℝ := { y | ∃ x, y = 2^x }

theorem intersect_M_N : M ∩ N = { y | 0 < y ∧ y ≤ 2 } := 
by
  sorry

end intersect_M_N_l588_588000


namespace least_common_denominator_l588_588311

-- Define the list of numbers
def numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the least common multiple function
noncomputable def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Define the main theorem
theorem least_common_denominator : lcm_list numbers = 2520 := 
  by sorry

end least_common_denominator_l588_588311


namespace log_of_y_sub_x_plus_one_positive_l588_588906

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588906


namespace Josie_shopping_time_l588_588106

theorem Josie_shopping_time
  (wait_cart : ℕ := 3)
  (wait_employee : ℕ := 13)
  (wait_stocker : ℕ := 14)
  (wait_line : ℕ := 18)
  (total_shopping_trip : ℕ := 90) :
  total_shopping_trip - (wait_cart + wait_employee + wait_stocker + wait_line) = 42 :=
by
  -- Convert the total shopping trip time from hours to minutes
  have trip_minutes : total_shopping_trip = 90 := rfl
  -- Sum the waiting times
  have waiting_total : wait_cart + wait_employee + wait_stocker + wait_line = 48 := rfl
  -- Subtract the waiting time from the total trip time
  have shopping_time : total_shopping_trip - (wait_cart + wait_employee + wait_stocker + wait_line) = 90 - 48 := rfl
  -- Hence, Josie spent 42 minutes shopping
  rw [trip_minutes,waiting_total,shopping_time]
  exact rfl
  sorry

end Josie_shopping_time_l588_588106


namespace smallest_t_for_full_circle_l588_588148

def full_circle (θ t : ℝ) : Prop := 
  r = sin θ ∧ 0 ≤ θ ∧ θ ≤ t

theorem smallest_t_for_full_circle :
  ∃ t, (t ≤ 2 * Real.pi ∧ ∀ θ, 0 ≤ θ ∧ θ ≤ t → (full_circle θ t)) :=
begin
  sorry
end

end smallest_t_for_full_circle_l588_588148


namespace segment_length_abs_cubed_root_l588_588598

theorem segment_length_abs_cubed_root (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  ∃ a b : ℝ, a = 3 + 5 ∧ b = 3 - 5 ∧ (b - a).abs = 10 :=
by {
  have h1 : real.cbrt 27 = 3 := by norm_num,
  rw h1 at h,
  have h2 : |x - 3| = 5 := h,
  use [8, -2],
  split,
  { refl },
  { split,
    { refl },
    { norm_num } }
}

end segment_length_abs_cubed_root_l588_588598


namespace problem_l588_588814

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588814


namespace shaded_area_is_54_l588_588202

-- Define the coordinates of points O, A, B, C, D, E
structure Point where
  x : ℝ
  y : ℝ

-- Given points
def O := Point.mk 0 0
def A := Point.mk 4 0
def B := Point.mk 16 0
def C := Point.mk 16 12
def D := Point.mk 4 12
def E := Point.mk 4 3

-- Define the function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2) ^ (1/2)

-- Define similarity of triangles and calculate side lengths involved
def triangles_similarity (OA OB CB EA : ℝ) : Prop :=
  OA / OB = EA / CB

-- Define the condition
def condition : Prop := 
  triangles_similarity (distance O A) (distance O B) 12 (distance E A) ∧
  distance E A = 3 ∧
  distance D E = 9

-- Define the calculation of area of triangle given base and height
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- State that the area of triangle CDE is 54 cm²
def area_shaded_region : Prop :=
  triangle_area 9 12 = 54

-- Main theorem statement
theorem shaded_area_is_54 : condition → area_shaded_region := by
  sorry

end shaded_area_is_54_l588_588202


namespace sum_of_consecutive_integers_product_l588_588553

noncomputable def consecutive_integers_sum (n m k : ℤ) : ℤ :=
  n + m + k

theorem sum_of_consecutive_integers_product (n m k : ℤ)
  (h1 : n = m - 1)
  (h2 : k = m + 1)
  (h3 : n * m * k = 990) :
  consecutive_integers_sum n m k = 30 :=
by
  sorry

end sum_of_consecutive_integers_product_l588_588553


namespace shaded_area_of_triangle_CDE_l588_588195

-- Definitions of the points
noncomputable def O := (0, 0 : ℝ×ℝ)
noncomputable def A := (4, 0 : ℝ×ℝ)
noncomputable def B := (16, 0 : ℝ×ℝ)
noncomputable def C := (16, 12 : ℝ×ℝ)
noncomputable def D := (4, 12 : ℝ×ℝ)
noncomputable def E := (4, 3 : ℝ×ℝ)

-- Definition of the area calculation for the given triangle
theorem shaded_area_of_triangle_CDE : 
  let DE := 9 in
  let DC := 12 in
  (DE * DC) / 2 = 54 :=
by
  sorry

end shaded_area_of_triangle_CDE_l588_588195


namespace ross_breaths_per_minute_l588_588508

noncomputable def breaths_per_minute (total_air : ℝ) (air_per_breath : ℝ) (hours : ℝ) (minutes_per_hour : ℝ) : ℝ :=
  let total_breaths := total_air / air_per_breath in
  let total_minutes := hours * minutes_per_hour in
  total_breaths / total_minutes

theorem ross_breaths_per_minute :
  breaths_per_minute 13600 (5/9 : ℝ) 24 60 = 17 :=
by
  sorry

end ross_breaths_per_minute_l588_588508


namespace inequality_ln_positive_l588_588856

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588856


namespace impossible_L_sum_2005_l588_588034

-- Definition of the list of numbers filling the 9 by 9 grid
def grid : List (List ℕ) := 
  [
    [460, 461, 462, 463, 464, 465, 466, 467, 468],
    [469, 470, 471, 472, 473, 474, 475, 476, 477],
    [478, 479, 480, 481, 482, 483, 484, 485, 486],
    [487, 488, 489, 490, 491, 492, 493, 494, 495],
    [496, 497, 498, 499, 500, 501, 502, 503, 504],
    [505, 506, 507, 508, 509, 510, 511, 512, 513],
    [514, 515, 516, 517, 518, 519, 520, 521, 522],
    [523, 524, 525, 526, 527, 528, 529, 530, 531],
    [532, 533, 534, 535, 536, 537, 538, 539, 540]
  ]

-- The L-shaped piece can be represented by a list of relative positions
-- (row_offset, col_offset) from the top-left corner of the L-shape
def L_shapes : List (List (Int × Int)) := 
  [
    [(0, 0), (0, 1), (1, 0), (2, 0)], -- L-shape extending downwards
    [(0, 0), (0, 1), (1, 1), (2, 1)], -- L-shape extending downwards flipped
    [(0, 0), (1, 0), (1, 1), (1, 2)], -- L-shape extending to the right
    [(0, 1), (1, 0), (1, 1), (1, 2)]  -- L-shape extending to the right flipped
  ]

def sum_of_L_piece (grid : List (List ℕ)) (start_pos : Int × Int) (shape : List (Int × Int)) : ℕ :=
  shape.map (fun offset => 
    let r := start_pos.1 + offset.1
    let c := start_pos.2 + offset.2
    grid[r.toNat][c.toNat]
  ).sum

theorem impossible_L_sum_2005 : ∀ (row col : Nat), row < 9 → col < 9 →
  ∑ shape in L_shapes, sum_of_L_piece grid (row, col) shape ≠ 2005 :=
by
  sorry

end impossible_L_sum_2005_l588_588034


namespace log_y_minus_x_plus_1_pos_l588_588870

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588870


namespace sacks_filled_l588_588298

theorem sacks_filled (pieces_per_sack : ℕ) (total_pieces : ℕ) (h1 : pieces_per_sack = 20) (h2 : total_pieces = 80) : (total_pieces / pieces_per_sack) = 4 :=
by {
  sorry
}

end sacks_filled_l588_588298


namespace primes_div_order_l588_588083

theorem primes_div_order (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : q ∣ 3^p - 2^p) : p ∣ q - 1 :=
sorry

end primes_div_order_l588_588083


namespace probability_proof_l588_588088

variable (a b c : ℝ)
-- Probabilities adjusted by difficulty factors
def probability_xavier_success := (1 / 3) * a
def probability_yvonne_success := (1 / 2) * b
def probability_zelda_success := (5 / 8) * c

-- Probability that Zelda does not solve the problem
def probability_zelda_failure := 1 - probability_zelda_success c

-- Combined probability for Xavier and Yvonne to succeed and Zelda to fail
def combined_probability (a b c : ℝ) : ℝ :=
  probability_xavier_success a * probability_yvonne_success b * probability_zelda_failure c

theorem probability_proof :
  combined_probability a b c = (1 / 16) * a * b * c :=
by
  sorry

end probability_proof_l588_588088


namespace sum_of_integers_divisible_by_15_less_than_999_l588_588714

theorem sum_of_integers_divisible_by_15_less_than_999 : 
  ∑ k in finset.range 67, (15 * k) = 33165 := 
by
  sorry

end sum_of_integers_divisible_by_15_less_than_999_l588_588714


namespace PL_length_is_correct_l588_588179

-- Define the triangle and given conditions
variables (P Q R L : Type) [triangle : Triangle P Q R]
variable {PQ QR RP : ℝ}
variables (PQ = 6) (QR = 7) (RP = 8)

-- Define the circles \omega_1 and \omega_2 with tangency conditions
variables (ω₁ ω₂ : Circle)
variables (t₁ : ω₁.tangent_to RP P) (t₂ : ω₂.tangent_to PQ P)

-- Define intersection point L not equal to P
variables (L_intersect : ω₁ ∩ ω₂ = {P, L})

-- Length of PL to prove
noncomputable def PL_length : ℝ := sorry

-- Main theorem statement
theorem PL_length_is_correct : PL_length = 48 / 7 :=
by
  sorry

end PL_length_is_correct_l588_588179


namespace hexagon_side_length_l588_588108

theorem hexagon_side_length (d : ℝ) (h : d = 10) : 
  ∃ (a : ℝ), a = 20 * real.sqrt 3 / 3 ∧ d = a * real.sqrt 3 :=
by
  use 20 * real.sqrt 3 / 3
  split
  { refl }
  { rwa [h, eq_comm, mul_comm, mul_assoc, ← div_eq_mul_inv, div_mul_cancel, mul_one]
    apply real.sqrt_ne_zero.2
    norm_num }

end hexagon_side_length_l588_588108


namespace highest_student_id_in_sample_l588_588419

variable (n : ℕ) (start : ℕ) (interval : ℕ)

theorem highest_student_id_in_sample :
  start = 5 → n = 54 → interval = 9 → 6 = n / interval → start = 5 →
  5 + (interval * (6 - 1)) = 50 :=
by
  sorry

end highest_student_id_in_sample_l588_588419


namespace area_of_shaded_region_l588_588189

-- Define points F, G, H, I, J with their coordinates
def F := (0, 0)
def G := (4, 0)
def H := (16, 0)
def I := (16, 12)
def J := (4, 3)

-- Define the similarity condition
def similar_triangles_JFG_IHG : Prop :=
  (triangle.similar F G J) (triangle.similar H G I)

-- The lengths of the segments based on problem conditions
def length_HG := 12
def length_JG := 3
def length_IG := 9

-- Area calculation of triangle IJG
def area_IJG := (1/2 * length_IG * length_JG).toReal

-- Final proof statement
theorem area_of_shaded_region :
  similar_triangles_JFG_IHG →
  length_HG = 12 →
  length_JG = length_HG/4 →
  length_IG = length_HG - length_JG →
  real.floor (area_IJG + 0.5) = 14 :=
by
  intros h_sim h_HG h_JG h_IG
  sorry

end area_of_shaded_region_l588_588189


namespace problem_1_problem_2_l588_588981

namespace ProofProblems

def U : Set ℝ := {y | true}

def E : Set ℝ := {y | y > 2}

def F : Set ℝ := {y | ∃ (x : ℝ), (-1 < x ∧ x < 2 ∧ y = x^2 - 2*x)}

def complement (A : Set ℝ) : Set ℝ := {y | y ∉ A}

theorem problem_1 : 
  (complement E ∩ F) = {y | -1 ≤ y ∧ y ≤ 2} := 
  sorry

def G (a : ℝ) : Set ℝ := {y | ∃ (x : ℝ), (0 < x ∧ x < a ∧ y = Real.log x / Real.log 2)}

theorem problem_2 (a : ℝ) :
  (∀ y, (y ∈ G a → y < 3)) → a ≥ 8 :=
  sorry

end ProofProblems

end problem_1_problem_2_l588_588981


namespace diagonals_perpendicular_then_side_lengths_relation_l588_588111

variables {α : Type*} [inner_product_space ℝ α]

-- Define vectors representing sides of the quadrilateral
variables (a b c d : α)

-- Define the condition that the diagonals are perpendicular
def diagonals_perpendicular (a b : α) : Prop :=
  (a + b) ⬝ (b + (- (a + b + c))) = 0

-- The target theorem to prove that a^2 + c^2 = b^2 + d^2
theorem diagonals_perpendicular_then_side_lengths_relation
  (habcd : a + c = b + d)
  (hd : diagonals_perpendicular a b) :
  ∥a∥^2 + ∥c∥^2 = ∥b∥^2 + ∥d∥^2 :=
sorry -- proof goes here

end diagonals_perpendicular_then_side_lengths_relation_l588_588111


namespace limit_of_function_l588_588668

theorem limit_of_function:
  (tendsto (fun x => (exp (sin (2 * x)) - exp (tan (2 * x))) / (log (2 * x / π)))
     (nhds π/2) (nhds (-2 * π))) :=
by
  sorry

end limit_of_function_l588_588668


namespace sum_of_digits_of_sum_five_digit_palindromes_eq_45_l588_588635

-- Definitions for a five-digit palindrome and the conditions

def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
    n = 10001 * a + 1010 * b + 100 * c

def sum_five_digit_palindromes : ℕ :=
  ∑ n in (finset.filter is_five_digit_palindrome (finset.range 100000)), n

-- The main statement we're interested in proving
theorem sum_of_digits_of_sum_five_digit_palindromes_eq_45 :
  ∑ d in (int_to_digits (sum_five_digit_palindromes)), d = 45 := 
sorry

end sum_of_digits_of_sum_five_digit_palindromes_eq_45_l588_588635


namespace vets_recommend_yummy_dog_kibble_l588_588252

theorem vets_recommend_yummy_dog_kibble :
  (let total_vets := 1000
   let percentage_puppy_kibble := 20
   let vets_puppy_kibble := (percentage_puppy_kibble * total_vets) / 100
   let diff_yummy_puppy := 100
   let vets_yummy_kibble := vets_puppy_kibble + diff_yummy_puppy
   let percentage_yummy_kibble := (vets_yummy_kibble * 100) / total_vets
   percentage_yummy_kibble = 30) :=
by
  sorry

end vets_recommend_yummy_dog_kibble_l588_588252


namespace count_integers_343_use_4_or_5_base_7_l588_588006

-- Definitions based on conditions
def is_digit_in_base (n : ℕ) (b : ℕ) (d : ℕ → Prop) : Prop :=
  ∀ (i : ℕ), d (n / b^i % b)

def uses_digit_4_or_5 (n : ℕ) : Prop :=
  is_digit_in_base n 7 (λ d, d = 4 ∨ d = 5)

-- Main statement to prove
theorem count_integers_343_use_4_or_5_base_7 :
  ∃ (count : ℕ), count = 218 ∧ count = (∑ n in finset.range 343, ite (uses_digit_4_or_5 n) 1 0) :=
begin
  sorry
end

end count_integers_343_use_4_or_5_base_7_l588_588006


namespace set_intersection_complement_eq_l588_588793

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

noncomputable def complement (U B : Set ℕ) : Set ℕ := { x ∈ U | x ∉ B }

theorem set_intersection_complement_eq : (A ∩ (complement U B)) = {1, 3} := 
by 
  sorry

end set_intersection_complement_eq_l588_588793


namespace points_in_third_game_is_8_l588_588799

-- Define the conditions
def total_points : ℕ := 20
def points_first_game : ℕ := (1 / 2 : ℚ) * total_points
def points_second_game : ℕ := (1 / 10 : ℚ) * total_points
def points_third_game : ℕ := total_points - points_first_game - points_second_game

-- Prove that the points scored in the third game is 8
theorem points_in_third_game_is_8 : points_third_game = 8 := by
  -- Skipping the proof
  sorry

end points_in_third_game_is_8_l588_588799


namespace train_length_proof_l588_588171

-- Definitions of the speeds of the trains
def Speed_A : ℝ := 70 -- in km/h
def Speed_B : ℝ := 50 -- in km/h
def Speed_C : ℝ := 32 -- in km/h

-- Definitions for the times observed
def time_B : ℝ := 20 -- in seconds
def time_A : ℝ := 10 -- in seconds

-- Converting km/h to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Relative speed calculations in m/s
def Rel_Speed_B_C : ℝ := kmph_to_mps (Speed_B - Speed_C)
def Rel_Speed_A_C : ℝ := kmph_to_mps (Speed_A - Speed_C)

-- The lengths of the trains as observed
def Length_Train_B : ℝ := Rel_Speed_B_C * time_B
def Length_Train_A : ℝ := Rel_Speed_A_C * time_A

-- Main statement to prove
theorem train_length_proof : Length_Train_A ≈ 100 ∧ Length_Train_B ≈ 100 := 
by
  sorry

end train_length_proof_l588_588171


namespace dina_dolls_count_l588_588321

-- Define the conditions
variable (Ivy_dolls : ℕ)
variable (Collectors_Ivy_dolls : ℕ := 20)
variable (Dina_dolls : ℕ)

-- Condition: Ivy has 2/3 of her dolls as collectors editions
def collectors_edition_condition : Prop := (2 / 3 : ℝ) * Ivy_dolls = Collectors_Ivy_dolls

-- Condition: Dina has twice as many dolls as Ivy
def dina_ivy_dolls_relationship : Prop := Dina_dolls = 2 * Ivy_dolls

-- Theorem: Prove that Dina has 60 dolls
theorem dina_dolls_count : collectors_edition_condition Ivy_dolls ∧ dina_ivy_dolls_relationship Ivy_dolls Dina_dolls → Dina_dolls = 60 := by
  sorry

end dina_dolls_count_l588_588321


namespace relationship_of_rationals_l588_588758

theorem relationship_of_rationals (a b c : ℚ) (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a :=
by {
  sorry
}

end relationship_of_rationals_l588_588758


namespace AB_length_l588_588029

-- Variables for Triangle ABC
variables {A B C : Type} [triangle ABC] 

-- Define the conditions for the triangle
noncomputable def angle_A : ℝ := 90
noncomputable def tan_B : ℝ := 5 / 12
noncomputable def AC : ℝ := 65

theorem AB_length : ∀ (A B C : Type), 
  (angle_A = 90) → (tan_B = 5 / 12) → (AC = 65) → (∀ AB : ℝ, AB = 60) :=
by 
  sorry

end AB_length_l588_588029


namespace count_integers_in_range_l588_588802

theorem count_integers_in_range : 
  (setOf (λ x : ℤ, -6 ≤ 2 * x + 2 ∧ 2 * x + 2 ≤ 4)).toFinset.card = 6 := by
sorry

end count_integers_in_range_l588_588802


namespace quadratic_expression_value_l588_588548

noncomputable theory
open_locale classical

theorem quadratic_expression_value {a b c : ℝ}
  (h1 : a ≠ 0)
  (h2 : (-2) * a + (-1) * b + c = -2.5)
  (h3 : (-1) * a + (-1) * b + c = -5)
  (h4 : c = -2.5)
  (h5 : a + b + c = 5)
  (h6 : 4 * a + 2 * b + c = 17.5) :
  16 * a - 4 * b + c = 17.5 :=
sorry

end quadratic_expression_value_l588_588548


namespace min_moves_for_queens_swap_l588_588993

theorem min_moves_for_queens_swap :
  let black_queens_initial := [1, 1, 1, 1, 1, 1, 1, 1] -- All in the first rank
  let white_queens_initial := [8, 8, 8, 8, 8, 8, 8, 8] -- All in the last rank
  ∀ (black_positions white_positions : List Nat),
    -- Black and white queens move alternately 
    -- with one queen moving per turn, initially on above positions
    black_positions = black_queens_initial →
    white_positions = white_queens_initial →
    -- Prove that the minimum number of moves to switch the positions is 23.
    min_moves black_positions white_positions = 23 :=
sorry

end min_moves_for_queens_swap_l588_588993


namespace real_inequality_l588_588390

theorem real_inequality
  (a1 a2 a3 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (S : ℝ)
  (hS : S = a1 + a2 + a3)
  (h4 : ∀ i ∈ [a1, a2, a3], (i^2 / (i - 1) > S)) :
  (1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1) := 
by
  sorry

end real_inequality_l588_588390


namespace calculate_average_fish_caught_l588_588917

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l588_588917


namespace ppq_not_divisible_by_7_l588_588726

theorem ppq_not_divisible_by_7 (P Q : ℕ) (hP : P < 10) (hQ : Q < 10) :
  let PPPQQQ := 111000 * P + 111 * Q in
  (∃ P Q : ℕ, P < 10 ∧ Q < 10 ∧ ¬ 7 ∣ PPPQQQ) := by
  -- Note: we are stating that there exist some P and Q that make PPPQQQ not divisible by 7.
  sorry

end ppq_not_divisible_by_7_l588_588726


namespace side_length_CD_of_cyclic_pentagon_l588_588272

theorem side_length_CD_of_cyclic_pentagon (R : ℝ)
  (ABCDE : (points : List Point) × points.length = 5) -- Pentagon ABCDE
  (inscribed_in_circle : ∃ center : Point, ∀ p ∈ ABCDE.1, dist center p = R) 
  (angle_B : ∠ (ABCDE.1.nth 1) (ABCDE.1.nth 2) (ABCDE.1.nth 0) = 110) 
  (angle_E : ∠ (ABCDE.1.nth 4) (ABCDE.1.nth 3) (ABCDE.1.nth 0) = 100) :
  dist (ABCDE.1.nth 2) (ABCDE.1.nth 3) = R :=
by sorry

end side_length_CD_of_cyclic_pentagon_l588_588272


namespace segment_length_l588_588577

theorem segment_length (x : ℝ) (h : |x - (27^(1/3))| = 5) : ∃ a b : ℝ, a = 8 ∧ b = -2 ∧ |a - b| = 10 :=
by
  have hx1 : x = 27^(1/3) + 5 ∨ x = 27^(1/3) - 5 := abs_eq hx
  use [8, -2]
  split
  { calc 8 = 27^(1/3) + 5 := by sorry }
  split
  { calc -2 = 27^(1/3) - 5 := by sorry }
  { calc |8 - -2| = |8 + 2| := by sorry }

end segment_length_l588_588577


namespace range_of_a_l588_588383

noncomputable def is_above_xaxis (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x > 0

noncomputable def quadratic_function (a : ℝ) : ℝ → ℝ :=
  λ x, a * x ^ 2 + x + 5

theorem range_of_a (a : ℝ) :
  (is_above_xaxis (quadratic_function a)) → (a > 1 / 20) :=
by
  sorry

end range_of_a_l588_588383


namespace find_AX_l588_588052

variable (A B C X : Type) [IsReal]
variable (AC BC BX CX AX : ℝ)
variable (H1 : AC = 45)
variable (H2 : BC = 50)
variable (H3 : BX = 35)
variable (H4 : CX = 2/5 * BC)
variable (H5 : CX = 20)
variable (H6 : ∀ (a b : ℝ), AC / (AX + BX) ∧ CX / BX = a / b)

theorem find_AX : AX = 43.75 :=
by
  have : 45 * 35 = 20 * (AX + 35) := H5 ▸ H6.1
  have : 1575 = 20 * AX + 700 := calc
    45 * 35 = 1575 : sorry
    20 * (AX + 35) = 20 * AX + 700 : by ring
  have : AX = 43.75 := by
    sorry
  assumption

end find_AX_l588_588052


namespace solution_set_m_eq_0_solution_set_R_if_1_le_m_lt_9_l588_588792

-- Definitions of the problem conditions
def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-1)*x + 2

-- The first part of the problem: If m = 0, then the solution set of the inequality f(m, x) > 0 is (-2, 1)
theorem solution_set_m_eq_0 : 
  (∀ x : ℝ, f 0 x > 0 ↔ x ∈ set.Ioo (-2 : ℝ) 1) := 
sorry

-- The second part of the problem: If the solution set of f(m, x) > 0 is ℝ, then 1 ≤ m < 9
theorem solution_set_R_if_1_le_m_lt_9 (m : ℝ) : 
  (∀ x : ℝ, f m x > 0 ↔ true) ↔ 1 ≤ m ∧ m < 9 :=
sorry

end solution_set_m_eq_0_solution_set_R_if_1_le_m_lt_9_l588_588792


namespace segment_length_abs_eq_five_l588_588584

theorem segment_length_abs_eq_five : 
  (length : ℝ) (∀ x : ℝ, abs (x - (27 : ℝ)^(1 : ℝ) / (3 : ℝ)) = 5 → x = 8 ∨ x = -2) 
  → length = 10 := 
begin
  sorry
end

end segment_length_abs_eq_five_l588_588584


namespace cyclic_quadrilateral_ptolemy_l588_588502

theorem cyclic_quadrilateral_ptolemy 
  (a b c d : ℝ) 
  (h : a + b + c + d = Real.pi) :
  Real.sin (a + b) * Real.sin (b + c) = Real.sin a * Real.sin c + Real.sin b * Real.sin d :=
by
  sorry

end cyclic_quadrilateral_ptolemy_l588_588502


namespace neither_sufficient_nor_necessary_l588_588965

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem neither_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  ¬ ((q > 1) ↔ is_increasing_sequence a) :=
sorry

end neither_sufficient_nor_necessary_l588_588965


namespace sufficient_conditions_for_equation_l588_588314

theorem sufficient_conditions_for_equation 
  (a b c : ℤ) :
  (a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c) →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_conditions_for_equation_l588_588314


namespace inequality_for_pos_reals_equality_condition_l588_588474

open Real

theorem inequality_for_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / c + c / b = 4 * a / (a + b)) ↔ (a = b ∧ b = c) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

end inequality_for_pos_reals_equality_condition_l588_588474


namespace remainder_correct_l588_588339

open Polynomial

noncomputable def polynomial_remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem remainder_correct : polynomial_remainder (X^6 - 2*X^5 + X^4 - X^2 - 2*X + 1)
                                                  ((X^2 - 1)*(X - 2)*(X + 2))
                                                = 2*X^3 - 9*X^2 + 3*X + 2 :=
by
  sorry

end remainder_correct_l588_588339


namespace total_stairs_climbed_l588_588447
noncomputable def jonny_climbed : ℕ := 4872
noncomputable def julia_climbed : ℕ := Nat.floor (2 * Real.sqrt (jonny_climbed / 2) + 15)
noncomputable def sam_climbed : ℕ := Nat.floor (5 * Real.cbrt ((jonny_climbed + julia_climbed) / 3))

theorem total_stairs_climbed :
  jonny_climbed + julia_climbed + sam_climbed = 5045 :=
by
  sorry

end total_stairs_climbed_l588_588447


namespace sphere_volume_to_surface_area_l588_588379

noncomputable def sphere_surface_area (V : ℝ) : ℝ :=
  let R := real.cbrt (V / (4 / 3 * real.pi))
  4 * real.pi * R^2

theorem sphere_volume_to_surface_area {V : ℝ} (h : V = 4 * real.sqrt 3 * real.pi) : sphere_surface_area V = 12 * real.pi :=
  sorry

end sphere_volume_to_surface_area_l588_588379


namespace sequence_a_is_positive_integer_sequence_a_even_iff_sequence_a_odd_iff_l588_588085

def sequence_a (r : ℕ) : ℕ → ℕ
| 0     := 1
| (n+1) := (n * (sequence_a r n) + 2 * (n + 1)^(2 * r)) / (n + 2)

theorem sequence_a_is_positive_integer (r : ℕ) (n : ℕ) : 
  0 < sequence_a r n := sorry

theorem sequence_a_even_iff (r : ℕ) (n : ℕ) : 
  (sequence_a r (n+1) % 2 = 0) ↔ (n % 4 = 3 ∨ n % 4 = 0) := sorry

theorem sequence_a_odd_iff (r : ℕ) (n : ℕ) : 
  (sequence_a r (n+1) % 2 = 1) ↔ (n % 4 = 1 ∨ n % 4 = 2) := sorry

end sequence_a_is_positive_integer_sequence_a_even_iff_sequence_a_odd_iff_l588_588085


namespace percentage_markup_correct_l588_588155

noncomputable def selling_price : ℝ := 7967
noncomputable def cost_price : ℝ := 6425

def markup (sp cp : ℝ) : ℝ := sp - cp

def percentage_markup (m cp : ℝ) : ℝ := (m / cp) * 100

theorem percentage_markup_correct :
  percentage_markup (markup selling_price cost_price) cost_price ≈ 23.99 :=
by sorry

end percentage_markup_correct_l588_588155


namespace length_of_FD_l588_588049

/-- In a square of side length 8 cm, point E is located on side AD,
2 cm from A and 6 cm from D. Point F lies on side CD such that folding
the square so that C coincides with E creates a crease along GF. 
Prove that the length of segment FD is 7/4 cm. -/
theorem length_of_FD (x : ℝ) (h_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
    (h_AE : ∀ (A E : ℝ), A - E = 2) (h_ED : ∀ (E D : ℝ), E - D = 6)
    (h_pythagorean : ∀ (x : ℝ), (8 - x)^2 = x^2 + 6^2) : x = 7/4 :=
by
  sorry

end length_of_FD_l588_588049


namespace find_a_extremum_at_one_find_range_of_x_l588_588975

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / 3) * x^3 - (3 / 2) * x^2 + (a + 1) * x + 1

def f_derivative (a : ℝ) : ℝ → ℝ := λ x, a * x^2 - 3 * x + (a + 1)

theorem find_a_extremum_at_one :
  ∃ a : ℝ, f_derivative a 1 = 0 ∧ a = 1 :=
sorry

theorem find_range_of_x (a : ℝ) (h : 0 < a) :
  ∀ x : ℝ, f_derivative a x > x^2 - x - a + 1 → -2 ≤ x ∧ x ≤ 0 :=
sorry

end find_a_extremum_at_one_find_range_of_x_l588_588975


namespace minimum_obtuse_triangles_in_triangulation_of_2003gon_l588_588673

-- Defining the concepts of a polygon, a circle, and a triangulation
def polygon (n : ℕ) := { v : ℕ // 3 ≤ n }

def inscribed_circle (n : ℕ) (P : polygon n) := 
  ∃ C : ℝ × ℝ, ∀ v ∈ P, ∃ r : ℝ, (circle_equation C r v)

def triangulation (n : ℕ) (P : polygon n) := 
  ∀ t, t ∈ (triangularization P) → (obtuse t) ∨ (acute t ∨ right t)

-- Main statement we want to prove
theorem minimum_obtuse_triangles_in_triangulation_of_2003gon : 
  ∀ (P : polygon 2003), inscribed_circle 2003 P → ∃ k, k = 1999 ∧
  (∀ t ∈ (triangularization P), obtuse t → k = 1999) :=
by
  sorry

end minimum_obtuse_triangles_in_triangulation_of_2003gon_l588_588673


namespace income_on_fifth_day_l588_588261

-- Define the incomes for the first four days
def income_day1 := 600
def income_day2 := 250
def income_day3 := 450
def income_day4 := 400

-- Define the average income
def average_income := 500

-- Define the length of days
def days := 5

-- Define the total income for the 5 days
def total_income : ℕ := days * average_income

-- Define the total income for the first 4 days
def total_income_first4 := income_day1 + income_day2 + income_day3 + income_day4

-- Define the income on the fifth day
def income_day5 := total_income - total_income_first4

-- The theorem to prove the income of the fifth day is $800
theorem income_on_fifth_day : income_day5 = 800 := by
  -- proof is not required, so we leave the proof section with sorry
  sorry

end income_on_fifth_day_l588_588261


namespace sum_of_first_9_terms_is_27_l588_588051

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Definition for the geometric sequence
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Definition for the arithmetic sequence

axiom a_geo_seq : ∃ r : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n * r
axiom b_ari_seq : ∃ d : ℝ, ∀ n : ℕ, b_n (n + 1) = b_n n + d
axiom a5_eq_3 : 3 * a_n 5 - a_n 3 * a_n 7 = 0
axiom b5_eq_a5 : b_n 5 = a_n 5

noncomputable def S_9 := (1 / 2) * 9 * (b_n 1 + b_n 9)

theorem sum_of_first_9_terms_is_27 : S_9 = 27 := by
  sorry

end sum_of_first_9_terms_is_27_l588_588051


namespace bryson_new_shoes_l588_588657

-- Define the conditions as variables and constant values
def pairs_of_shoes : ℕ := 2 -- Number of pairs Bryson bought
def shoes_per_pair : ℕ := 2 -- Number of shoes per pair

-- Define the theorem to prove the question == answer
theorem bryson_new_shoes : pairs_of_shoes * shoes_per_pair = 4 :=
by
  sorry -- Proof placeholder

end bryson_new_shoes_l588_588657


namespace ratio_of_b_to_a_is_one_l588_588716

theorem ratio_of_b_to_a_is_one
  (n : ℕ)
  (a b : ℝ)
  (h1 : n = 4)
  (h2 : 0 < a)
  (h3 : 0 < b)
  (h4 : (a + b * complex.I)^4 = (a - b * complex.I)^4) :
  b / a = 1 :=
sorry

end ratio_of_b_to_a_is_one_l588_588716


namespace emani_money_l588_588683

def emani_has_30_more (E H : ℝ) : Prop := E = H + 30
def equal_share (E H : ℝ) : Prop := (E + H) / 2 = 135

theorem emani_money (E H : ℝ) (h1: emani_has_30_more E H) (h2: equal_share E H) : E = 150 :=
by
  sorry

end emani_money_l588_588683


namespace complement_of_P_relative_to_U_l588_588387

def U := {-1, 0, 1, 2, 3}
def P := {-1, 2, 3}

def complement (U P : Set ℤ) : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_of_P_relative_to_U :
  complement U P = {0, 1} :=
by
  sorry

end complement_of_P_relative_to_U_l588_588387


namespace inequality_proof_l588_588955

open EuclideanGeometry

noncomputable def point := sorry

variables (A B C I W1 W2 W3 : point) (r R : ℝ)

-- Conditions
axiom incenter (triangle : point → point → point → Prop) (I : point) (A B C : point) : Prop
axiom circumcircle_intersection (triangle : point → point → point → Prop) (I W : point) (l : point → Prop) : Prop
axiom inradius (I A B C : point) (r : ℝ) : Prop
axiom circumradius (A B C : point) (R : ℝ) : Prop

-- Proof problem statement
theorem inequality_proof :
  incenter (λ a b c, triangle a b c) I A B C →
  (∀ W, circumcircle_intersection (λ a b c, triangle a b c) I W (λ p, p ∈ [A, B, C])) →
  inradius I A B C r →
  circumradius A B C R →
  (dist I W1 + dist I W2 + dist I W3) ≥ 2R + real.sqrt (2R * r) :=
by sorry

end inequality_proof_l588_588955


namespace horse_revolutions_l588_588271

theorem horse_revolutions (R1 R2 : ℝ) (revs1 : ℕ) (distance1 distance2 : ℝ) :
  R1 = 30 → revs1 = 32 → R2 = 15 → 
  distance1 = 2 * π * R1 * revs1 →
  distance2 = 2 * π * R2 * 64 →
  distance1 = distance2 :=
by
  intros hR1 hRevs1 hR2 hDist1 hDist2
  rw [hR1, hRevs1, hR2] at hDist1 hDist2
  have h1 : distance1 = 1920 * π := by 
    rw [hR1, hRevs1]
    sorry
  have h2 : distance2 = 1920 * π := by 
    rw [hR2]
    sorry
  rw [h1, h2]
  exact rfl

end horse_revolutions_l588_588271


namespace A_pow_five_eq_rA_add_sI_l588_588462

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 1; 4, 3]

def I : Matrix (Fin 2) (Fin 2) ℚ :=
  1

theorem A_pow_five_eq_rA_add_sI :
  ∃ (r s : ℚ), (A^5) = r • A + s • I :=
sorry

end A_pow_five_eq_rA_add_sI_l588_588462


namespace initial_concentration_is_27_l588_588168

-- Define given conditions
variables (m m_c : ℝ) -- initial mass of solution and salt
variables (x : ℝ) -- initial percentage concentration of salt
variables (h1 : m_c = (x / 100) * m) -- initial concentration definition
variables (h2 : m > 0) (h3 : x > 0) -- non-zero positive mass and concentration

theorem initial_concentration_is_27 (h_evaporated : (m / 5) * 2 * (x / 100) = m_c) 
  (h_new_concentration : (x + 3) = (m_c * 100) / (9 * m / 10)) 
  : x = 27 :=
by
  sorry

end initial_concentration_is_27_l588_588168


namespace sequence_properties_l588_588357

/-- Given a sequence satisfying the given properties, 
    prove the properties of the sequence solution. -/
theorem sequence_properties 
  (a : ℕ → ℝ) (a_pos : ∀ n, a n > 0)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = 1 / sqrt (1 + 1 / (a n)^2)) :
  (∀ n, 1 / (a (n + 1))^2 - 1 / (a n)^2 = 1) ∧ 
  (∀ n, a n = 1 / sqrt n) ∧
  (∀ n, (finset.range n).sum (λ k, (a k)^2 * (a (k + 2))^2) = 1 / 2 * (3 / 2 - 1 / (n + 1) - 1 / (n + 2))) ∧ 
  (∀ n (t : ℕ), S n < t^2 - 3 * t - 13 / 4 → t ≥ 4) := sorry

end sequence_properties_l588_588357


namespace route_B_quicker_l588_588487

-- Define the conditions for Route A
def distance_A : ℝ := 8
def speed_A_non_construction : ℝ := 40
def speed_A_construction : ℝ := 20
def distance_A_non_construction : ℝ := 6
def distance_A_construction : ℝ := 2

-- Define the conditions for Route B
def distance_B : ℝ := 7
def speed_B_non_school : ℝ := 50
def speed_B_school : ℝ := 25
def distance_B_non_school : ℝ := 6
def distance_B_school : ℝ := 1

-- Calculate time for Route A
def time_A_non_construction : ℝ := distance_A_non_construction / speed_A_non_construction
def time_A_construction : ℝ := distance_A_construction / speed_A_construction
def total_time_A : ℝ := (time_A_non_construction + time_A_construction) * 60

-- Calculate time for Route B
def time_B_non_school : ℝ := distance_B_non_school / speed_B_non_school
def time_B_school : ℝ := distance_B_school / speed_B_school
def total_time_B : ℝ := (time_B_non_school + time_B_school) * 60

-- The main theorem to prove
theorem route_B_quicker : total_time_A - total_time_B = 5 + 2/5 := by
  sorry

end route_B_quicker_l588_588487


namespace union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l588_588091

def U : Set ℕ := { x | 1 ≤ x ∧ x < 9 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}
def C (S : Set ℕ) : Set ℕ := U \ S

theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6} := 
by {
  -- proof here
  sorry
}

theorem inter_A_B : A ∩ B = {3} := 
by {
  -- proof here
  sorry
}

theorem C_U_union_A_B : C (A ∪ B) = {7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_inter_A_B : C (A ∩ B) = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_A : C A = {4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_B : C B = {1, 2, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem union_C_U_A_C_U_B : C A ∪ C B = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem inter_C_U_A_C_U_B : C A ∩ C B = {7, 8} := 
by {
  -- proof here
  sorry
}

end union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l588_588091


namespace domain_of_myFunction_l588_588558

-- Define the function
def myFunction (x : ℝ) : ℝ := (x + 2) ^ (1 / 2) - (x + 1) ^ 0

-- State the domain constraints as a theorem
theorem domain_of_myFunction (x : ℝ) : 
  (x ≥ -2 ∧ x ≠ -1) →
  ∃ y : ℝ, y = myFunction x := 
sorry

end domain_of_myFunction_l588_588558


namespace redesigned_survey_customers_l588_588643

theorem redesigned_survey_customers :
  (7 / 70 : ℝ) + 0.04 ≈ 0.14 → (9 / x = 0.14) →  x ≈ 64 :=
begin
  sorry
end

end redesigned_survey_customers_l588_588643


namespace parabola_problem_l588_588355

open Classical

noncomputable def parabola_equation (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

/-- The proof statement -/
theorem parabola_problem :
  (∀ (x : ℝ), (x = 0 → parabola_equation 1 (-2) (-3) x = -3) ∧ 
              (x = -1 → parabola_equation 1 (-2) (-3) x = 0) ∧
              (x = 3 → parabola_equation 1 (-2) (-3) x = 0)) ∧
  (parabola_equation 1 (-2) (-3) = λ x, x^2 - 2*x - 3) ∧
  (∀ (x_1 x_2 : ℝ), x_1 < x_2 ∧ x_2 < 1 → parabola_equation 1 (-2) (-3) x_1 < parabola_equation 1 (-2) (-3) x_2) :=
by
  intros
  sorry

end parabola_problem_l588_588355


namespace ln_gt_sufficient_not_necessary_for_cube_l588_588606

theorem ln_gt_sufficient_not_necessary_for_cube (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (ln x > ln y) → (x^3 > y^3) ∧ ¬(x^3 > y^3 → ln x > ln y) :=
by
  sorry

end ln_gt_sufficient_not_necessary_for_cube_l588_588606


namespace sequence_property_l588_588376

theorem sequence_property (a : ℕ → ℝ) (f : ℕ → ℝ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ n, (a n) < (a (n + 1)))
  (h3 : ∀ n (m : ℝ), 0 ≤ m ∧ m < 1 → ∃ x1 x2, x1 ≠ x2 ∧ f n x1 = m ∧ f n x2 = m ∧ x1 ∈ set.Icc (a n) (a (n + 1)) ∧ x2 ∈ set.Icc (a n) (a (n + 1)))
  (h4 : ∀ n x, x ∈ set.Icc (a n) (a n + 1) → f n x = abs (sin ((1 / n) * (x - a n)))) : 
  ∀ n, a n = (n * (n - 1) / 2) * Real.pi :=
by
  sorry

end sequence_property_l588_588376


namespace systematic_sampling_40th_number_l588_588040

open Nat

theorem systematic_sampling_40th_number (N n : ℕ) (sample_size_eq : n = 50) (total_students_eq : N = 1000) (k_def : k = N / n) (first_number : ℕ) (first_number_eq : first_number = 15) : 
  first_number + k * 39 = 795 := by
  sorry

end systematic_sampling_40th_number_l588_588040


namespace rectangle_area_l588_588276

theorem rectangle_area (length diagonal : ℝ) (h1 : length = 16) (h2 : diagonal = 20) : ∃ area, area = 192 :=
by 
  -- Assume the width as a real number.
  let w := Real.sqrt (diagonal ^ 2 - length ^ 2)
  -- State that the width squared plus the length squared equals the diagonal squared.
  have hw : diagonal ^ 2 = length ^ 2 + w ^ 2 := sorry
  -- Verify that the computed width matches the expected value.
  have w_val : w = 12 := sorry
  -- Compute the area of the rectangle.
  let area := length * w
  -- Show that the calculated area matches the expected value of 192.
  have area_val : area = 192 := sorry
  -- Existential quantifier stating there exists an area that equals 192.
  exact ⟨area, area_val⟩

end rectangle_area_l588_588276


namespace count_possible_x_l588_588150

theorem count_possible_x (x : ℕ) (h1 : x + 15 > 40) (h2 : x + 40 > 15) (h3 : 15 + 40 > x) : 
  ∃ n : ℕ, n = 29 ∧ {y : ℕ | 25 < y ∧ y < 55}.card = n := 
by
  sorry

end count_possible_x_l588_588150


namespace shaded_area_is_54_l588_588203

-- Define the coordinates of points O, A, B, C, D, E
structure Point where
  x : ℝ
  y : ℝ

-- Given points
def O := Point.mk 0 0
def A := Point.mk 4 0
def B := Point.mk 16 0
def C := Point.mk 16 12
def D := Point.mk 4 12
def E := Point.mk 4 3

-- Define the function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2) ^ (1/2)

-- Define similarity of triangles and calculate side lengths involved
def triangles_similarity (OA OB CB EA : ℝ) : Prop :=
  OA / OB = EA / CB

-- Define the condition
def condition : Prop := 
  triangles_similarity (distance O A) (distance O B) 12 (distance E A) ∧
  distance E A = 3 ∧
  distance D E = 9

-- Define the calculation of area of triangle given base and height
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- State that the area of triangle CDE is 54 cm²
def area_shaded_region : Prop :=
  triangle_area 9 12 = 54

-- Main theorem statement
theorem shaded_area_is_54 : condition → area_shaded_region := by
  sorry

end shaded_area_is_54_l588_588203


namespace solution_l588_588759

def is_even_function {α β} [LinearOrder α] [TopologicalSpace α]
  [TopologicalSpace β] [TopologicalOrder α] [TopologicalOrder β]
  (f : α → β) := ∀ x, f x = f (-x)

def is_increasing_on {α β} [LinearOrder α] [Preorder β] [TopologicalSpace α]
  [TopologicalSpace β] (s : Set α) (f : α → β) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

noncomputable def problem_conditions (a : ℝ) (f : ℝ → ℝ) : Prop :=
  is_even_function f ∧ is_increasing_on (Set.Icc (-4 : ℝ) 0) f ∧ f a < f 3

theorem solution (a : ℝ) (f : ℝ → ℝ) (h : problem_conditions a f) : a ∈ Icc (-4 : ℝ) (-3) ∪ Icc 3 4 :=
sorry

end solution_l588_588759


namespace max_intersections_of_fifth_degree_polynomials_l588_588222

theorem max_intersections_of_fifth_degree_polynomials
    (p q : ℝ → ℝ)
    (hp : ∃ a b c d e f, p = λ x, 2*x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e)
    (hq : ∃ a b c d e f, q = λ x, 3*x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e)
    : ∃ m : ℕ, m ≤ 5 := 
sorry

end max_intersections_of_fifth_degree_polynomials_l588_588222


namespace exists_k_element_subset_l588_588471

noncomputable def num_primes_leq (n : ℕ) : ℕ := (List.filter Nat.Prime (List.rangeOf (2, n+1))).length 

theorem exists_k_element_subset
  (n : ℕ)
  (h₀ : n ≥ 3)
  (k := num_primes_leq n)
  (A : Finset ℕ)
  (hA₀ : A ⊆ Finset.range (n + 1) \ {0, 1})
  (hA₁ : A.card < k)
  (hA₂ : ∀ a1 a2 ∈ A, a1 ≠ a2 → ¬ (a1 ∣ a2) ∧ ¬ (a2 ∣ a1)) :
  ∃ (B : Finset ℕ), 
    B ⊆ Finset.range (n + 1) \ {0, 1} ∧ 
    B.card = k ∧ 
    A ⊆ B ∧ 
    ∀ b1 b2 ∈ B, b1 ≠ b2 → ¬ (b1 ∣ b2) ∧ ¬ (b2 ∣ b1) :=
sorry

end exists_k_element_subset_l588_588471


namespace permutations_of_six_museums_l588_588951

theorem permutations_of_six_museums : 
  ∀ (museums : Finset ℕ), 
  museums.card = 6 → 
  (museums.perm) = 720 := by
  sorry

end permutations_of_six_museums_l588_588951


namespace log_pos_given_ineq_l588_588895

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588895


namespace eta_eta_eta_of_10_l588_588958

def eta (m : ℕ) : ℕ := (Finset.filter (λ d, m % d = 0) (Finset.range (m + 1))).prod id

theorem eta_eta_eta_of_10 : eta (eta (eta 10)) = 10 ^ 450 := by
  sorry

end eta_eta_eta_of_10_l588_588958


namespace quadrant_of_angle_l588_588025

theorem quadrant_of_angle (θ : ℝ) (h1 : sin θ < 0) (h2 : cos θ < 0) : 
  π < θ ∧ θ < (3 * π) / 2 :=
sorry

end quadrant_of_angle_l588_588025


namespace log_y_minus_x_plus_1_gt_0_l588_588851

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588851


namespace inequality_ln_pos_l588_588836

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588836


namespace problem_l588_588613

theorem problem (a : ℝ) (h : a ≥ 0) :
  ((2 * (a + 1) + 2 * Real.sqrt (a^2 + 2 * a)) / (3 * a + 1 - 2 * Real.sqrt (a^2 + 2 * a)))^(1 / 2) 
  - (Real.sqrt (2 * a + 1) - Real.sqrt a)^(-1) * Real.sqrt (a + 2) 
  = (Real.sqrt a) / (Real.sqrt (2 * a + 1) - Real.sqrt a) :=
by
  sorry

end problem_l588_588613


namespace solve_abs_eq_zero_l588_588602

theorem solve_abs_eq_zero : ∃ x : ℝ, |5 * x - 3| = 0 ↔ x = 3 / 5 :=
by
  sorry

end solve_abs_eq_zero_l588_588602


namespace sum_of_distances_is_1106_over_5_l588_588970

-- Definitions of the problem setup
noncomputable def A := (0, 182 : ℝ × ℝ)
noncomputable def B := (-70, 0 : ℝ × ℝ)
noncomputable def C := (70, 0 : ℝ × ℝ)
noncomputable def X1 := (70 - 130 * sin (arctan (130/70)), 130 * cos (arctan (130/70)) : ℝ × ℝ)
noncomputable def perp_line (P Q : ℝ × ℝ) := (Q.2 - P.2, P.1 - Q.1)

-- Define the distance function
noncomputable def dist (P Q : ℝ × ℝ) := sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Compute initial distances
noncomputable def BX1 := dist B X1

-- Function to compute the next point
noncomputable def Xn (n : ℕ) : ℝ × ℝ :=
  match n % 2, n with
  | 1, _   => -- odd n: intersection with AB with the perpendicular to previous X
    let P := Xn (n - 1)
    let Q := Xn (n - 2)
    let l := perp_line Q P
    -- TODO: intersection of l with AB
    (???, ??? : ℝ × ℝ) -- placeholder
  | 0, _   => -- even n: intersection with AC with the perpendicular to previous X
    let P := Xn (n - 1)
    let Q := Xn (n - 2)
    let l := perp_line Q P
    -- TODO: intersection of l with AC
    (???, ??? : ℝ × ℝ) -- placeholder

-- Sum of distances in the sequence
noncomputable def distances_sum : ℕ → ℝ
| 0     => BX1
| (n+1) => distances_sum n + dist (Xn n) (Xn (n + 1))

-- Problem statement
theorem sum_of_distances_is_1106_over_5 : distances_sum n = 1106 / 5 :=
sorry

end sum_of_distances_is_1106_over_5_l588_588970


namespace fraction_equal_l588_588918

theorem fraction_equal {a b x : ℝ} (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + b) / (a - b) = (x + 1) / (x - 1) := 
by
  sorry

end fraction_equal_l588_588918


namespace inequality_implies_log_pos_l588_588826

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588826


namespace area_of_rectangle_is_2559_l588_588424

noncomputable def rectangle_area 
  (A B D : ℝ × ℝ)
  (A_coords : A = (15, -45))
  (B_coords : B = (505, 155))
  (D_x : D.1 = 17)
  (y : ℝ)
  (D_coords : D = (17, y)) : ℝ := 
  let AB := Math.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AD := Math.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  AB * AD

theorem area_of_rectangle_is_2559 (y : ℝ) : 
  let A : ℝ × ℝ := (15, -45)
  let B : ℝ × ℝ := (505, 155)
  let D : ℝ × ℝ := (17, y)
  rectangle_area A B D rfl rfl rfl y rfl = 2559 := 
sorry

end area_of_rectangle_is_2559_l588_588424


namespace num_ways_to_distribute_7_balls_in_4_boxes_l588_588007

def num_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  -- Implement the function to calculate the number of ways here, but we'll keep it as a placeholder for now.
  sorry

theorem num_ways_to_distribute_7_balls_in_4_boxes : 
  num_ways_to_distribute_balls 7 4 = 3 := 
sorry

end num_ways_to_distribute_7_balls_in_4_boxes_l588_588007


namespace concurrency_of_lines_l588_588964

theorem concurrency_of_lines 
  (Omega : Circle)
  (omega1 omega2 omega3 gamma1 gamma2 gamma3 : Circle)
  (O1 O2 O3 : Point)
  (T1 T2 T3 C1 C2 C3 S1 S2 S3 : Point)
  (h_tangent_omega : ∀ i j, i ≠ j → Tangent omega_i omega_j)
  (h_tangent_Omega : ∀ i, Tangent omega_i Omega)
  (h_tangent_gamma_Omega : ∀ i, Tangent gamma_i Omega)
  (h_tangent_gamma_omega: ∀ i, Tangent gamma_i omega_{i+1} ∧ Tangent gamma_i omega_{i+2})
  (h_Oi_centers: ∀ i, Center omega_i = O_i)
  (h_Ti_tangency: ∀ i, Tangency omega_i Omega T_i)
  (h_Ci_centers: ∀ i, Center gamma_i = C_i)
  (h_Si_tangency: ∀ i, Tangency gamma_i Omega S_i) :
  Concurrent (Line T1 C1) (Line T2 C2) (Line T3 C3) ∧ 
  Concurrent (Line O1 C1) (Line O2 C2) (Line O3 C3) ∧ 
  Concurrent (Line O1 S1) (Line O2 S2) (Line O3 S3) := sorry

end concurrency_of_lines_l588_588964


namespace find_number_l588_588160

theorem find_number : 
    let x := -0.2666874 in (0.8)^3 - (0.5)^3 / (0.8)^2 + x + (0.5)^2 = 0.3000000000000001 :=
by
    let a := (0.8)^3
    let b := (0.5)^3
    let c := (0.8)^2
    let d := (0.5)^2
    let x := -0.2666874
    have h1 : a - b / c + x + d = 0.3000000000000001
    exact h1
    sorry

end find_number_l588_588160


namespace symmedian_of_triangle_l588_588468

-- Conditions
variables {A B C : Point}
variables [hA : Circle A B C]
variables {D : Point}
variables [hD : TangentIntersection B C D]

-- Statement of the problem
theorem symmedian_of_triangle (A B C D : Point) [Circle A B C] [TangentIntersection B C D] :
  is_symmedian A B C D :=
sorry

end symmedian_of_triangle_l588_588468


namespace decreasing_intervals_max_value_g_tangent_points_l588_588090

-- Question 1
theorem decreasing_intervals (m : ℝ) (h₁ : m > 0) (hm : m = 1) :
  ∃ I1 I2 : set ℝ, I1 = set.Iio 0 ∧ I2 = set.Ioi (2 / 3) ∧
  ∀ x ∈ I1 ∪ I2, (∃ f : ℝ → ℝ, f x = -x^3 + m * x^2 - m ∧ 
  (∃ f' : ℝ → ℝ, f' x = -3 * x^2 + 2 * m * x ∧ f' x < 0)) :=
sorry

-- Question 2
theorem max_value_g (m : ℝ) (h₁ : m > 0) :
  ∃ g_max : ℝ, g_max = max (abs (-m + 4/27 * m^3)) m ∧
  (∀ x ∈ set.Icc 0 m, ∃ g : ℝ → ℝ, g x = |-(x^3) + m * (x^2) - m| ∧ 
  g x ≤ g_max) :=
sorry

-- Question 3
theorem tangent_points (m : ℝ) (h₁ : m > 0) :
  (∃ t ∈ set.Iic (0 : ℝ), ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
  (∀ f : ℝ → ℝ, f x1 = -x1^3 + m * x1^2 - m ∧ f x2 = -x2^3 + m * x2^2 - m) ∧ 
  (∀ f' : ℝ → ℝ, f' x1 = -3 * x1^2 + 2 * m * x1 ∧ 
  f' x2 = -3 * x2^2 + 2 * m * x2 ∧ 
  (f' x1 * (2 - x1) + f x1 = t) ∧ 
  (f' x2 * (2 - x2) + f x2 = t))) → 
  (0 < m ∧ m ≤ 8 / 3) ∨ (m ≥ 9 + 3 * real.sqrt 6) :=
sorry

end decreasing_intervals_max_value_g_tangent_points_l588_588090


namespace number_of_elements_in_T_l588_588968

def g (x : ℝ) : ℝ := (x + 8) / x

def seq_g : ℕ → (ℝ → ℝ)
| 1       := g
| (n + 1) := g ∘ seq_g n

def T := {x : ℝ | ∃ n : ℕ, n > 0 ∧ seq_g n x = x}

theorem number_of_elements_in_T : T.finite.to_finset.card = 2 := 
sorry

end number_of_elements_in_T_l588_588968


namespace diabolical_safe_solution_l588_588243

theorem diabolical_safe_solution (n c : ℕ) (d : ℕ → ℕ) :
  (∀ i, 0 ≤ d i ∧ d i < c) →
  ∃ (c_i : ℕ → ℕ), (∀ i, 0 ≤ c_i i) ∧ 
  (∀ turns, (∀ j, d (j + turns) % c = 0)) ↔ 
  ((n > 1 ∧ c > 1) ∧ ∃ p : ℕ, p.prime ∧ ∃ a b : ℕ, n = p^a ∧ c = p^b) :=
by
  sorry

end diabolical_safe_solution_l588_588243


namespace city_of_archimedes_schools_l588_588048

noncomputable def numberOfSchools : ℕ := 32

theorem city_of_archimedes_schools :
  ∃ n : ℕ, (∀ s : Set ℕ, s = {45, 68, 113} →
  (∀ x ∈ s, x > 1 → 4 * n = x + 1 → (2 * n ≤ x ∧ 2 * n + 1 ≥ x) ))
  ∧ n = numberOfSchools :=
sorry

end city_of_archimedes_schools_l588_588048


namespace calculate_average_fish_caught_l588_588915

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l588_588915


namespace arithmetic_seq_num_terms_l588_588747

theorem arithmetic_seq_num_terms (a1 : ℕ := 1) (S_odd S_even : ℕ) (n : ℕ) 
  (h1 : S_odd = 341) (h2 : S_even = 682) : 2 * n = 10 :=
by
  sorry

end arithmetic_seq_num_terms_l588_588747


namespace x_squared_interval_l588_588404

theorem x_squared_interval (x : ℝ) 
  (h : (∛(x + 12) - ∛(x - 12)) = 4) : 
  105 ≤ x^2 ∧ x^2 ≤ 110 :=
sorry

end x_squared_interval_l588_588404


namespace log_pos_given_ineq_l588_588889

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588889


namespace solve_equation_nat_numbers_l588_588516

theorem solve_equation_nat_numbers :
  ∃ (x y z : ℕ), (2 ^ x + 3 ^ y + 7 = z!) ∧ ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
sorry

end solve_equation_nat_numbers_l588_588516


namespace isosceles_triangle_exterior_angle_apex_angle_l588_588763

theorem isosceles_triangle_exterior_angle_apex_angle (T : Triangle) (h1 : IsIsoscelesTriangle T) (θ : ℝ) (h2 : θ = 100) :
  (ApexAngle T = 20 ∨ ApexAngle T = 80) := 
sorry

end isosceles_triangle_exterior_angle_apex_angle_l588_588763


namespace triangle_BC_length_eq_52_l588_588925

theorem triangle_BC_length_eq_52 
  (AB AC : ℕ) (B C X : Point) 
  (h45: AB = 95) 
  (h87: AC = 87) 
  (CircleCenteredA : ∃ r : ℕ, Circle r (Point.origin) intersects (line B C) at [B, X]) 
  (lengths_integer: ∀ l ∈ {BX, CX}, l ∈ ℤ) :
  distance B C = 52 :=
by sorry

end triangle_BC_length_eq_52_l588_588925


namespace intercept_form_l588_588221

theorem intercept_form (x y : ℝ) : 2 * x - 3 * y - 4 = 0 ↔ x / 2 + y / (-4/3) = 1 := sorry

end intercept_form_l588_588221


namespace dice_sum_probability_l588_588146

theorem dice_sum_probability :
  let die1 := {1, 1, 2, 3, 3, 5}
  let die2 := {1, 3, 5, 6, 7, 9}
  (∑ i in die1, ∑ j in die2, if (i + j = 6 ∨ i + j = 8 ∨ i + j = 10) then 1 else 0) / (die1.card * die2.card) = 1 / 3 :=
by
  sorry

end dice_sum_probability_l588_588146


namespace segment_angle_45_deg_l588_588797

theorem segment_angle_45_deg (A B C A1 B1 A2 B2 : Point) 
  (h_triangle : right_triangle A B C)
  (h_angle_bisector_A : angle_bisector A B C A1) 
  (h_angle_bisector_C : angle_bisector C A B B1)
  (h_perpendicular_A1 : perpendicular A1 A B A2)
  (h_perpendicular_B1 : perpendicular B1 B C B2) 
  : angle A2 C B2 = 45 :=
  sorry

end segment_angle_45_deg_l588_588797


namespace lineup_count_l588_588524

def total_players : ℕ := 15
def out_players : ℕ := 3  -- Alice, Max, and John
def lineup_size : ℕ := 6

-- Define the binomial coefficient in Lean
def binom (n k : ℕ) : ℕ :=
  if h : n ≥ k then
    Nat.choose n k
  else
    0

theorem lineup_count (total_players out_players lineup_size : ℕ) :
  let remaining_with_alice := total_players - out_players + 1 
  let remaining_without_alice := total_players - out_players + 1 
  let remaining_without_both := total_players - out_players 
  binom remaining_with_alice (lineup_size-1) + binom remaining_without_alice (lineup_size-1) + binom remaining_without_both lineup_size = 3498 :=
by
  sorry

end lineup_count_l588_588524


namespace segment_length_eq_ten_l588_588593

theorem segment_length_eq_ten :
  (abs (8 - (-2)) = 10) :=
by
  -- Given conditions
  have h1 : 8 = real.cbrt 27 + 5 := sorry
  have h2 : -2 = real.cbrt 27 - 5 := sorry
  
  -- Using the conditions to prove the length
  sorry

end segment_length_eq_ten_l588_588593


namespace cages_needed_l588_588273

def initial_puppies := 18.0
def bought_puppies := 3.0
def puppies_per_cage := 5.0

noncomputable def total_puppies := initial_puppies + bought_puppies

theorem cages_needed : Nat.ceil (total_puppies / puppies_per_cage) = 5 := by
  sorry

end cages_needed_l588_588273


namespace smallest_nonzero_magnitude_of_z_l588_588966

noncomputable theory
open Complex

def primitive_third_root_of_unity : ℂ :=
  exp (2 * π * I / 3)

theorem smallest_nonzero_magnitude_of_z (a b c z : ℂ) (ω : ℂ)
  (h_omega : ω = primitive_third_root_of_unity)
  (h_a : a = 1) (h_b : b = -ω) (h_c : c = -ω^2)
  (h_a_mag : abs a = 1) (h_b_mag : abs b = 1) (h_c_mag : abs c = 1)
  (h_eq : a * z^2 + b * z + c = 0) :
  ∃ z, abs z = 1/2 :=
sorry

end smallest_nonzero_magnitude_of_z_l588_588966


namespace inequality_ln_pos_l588_588835

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588835


namespace sandy_saved_last_year_percentage_l588_588449

theorem sandy_saved_last_year_percentage (S : ℝ) (P : ℝ) :
  (this_year_salary: ℝ) → (this_year_savings: ℝ) → 
  (this_year_saved_percentage: ℝ) → (saved_last_year_percentage: ℝ) → 
  this_year_salary = 1.1 * S → 
  this_year_saved_percentage = 6 →
  this_year_savings = (this_year_saved_percentage / 100) * this_year_salary →
  (this_year_savings / ((P / 100) * S)) = 0.66 →
  P = 10 :=
by
  -- The proof is to be filled in here.
  sorry

end sandy_saved_last_year_percentage_l588_588449


namespace value_of_a_minus_b_l588_588401

theorem value_of_a_minus_b (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : ab = 6) : a - b = 5 ∨ a - b = -5 := 
by
  sorry

end value_of_a_minus_b_l588_588401


namespace calculate_average_fish_caught_l588_588916

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l588_588916


namespace log_of_y_sub_x_plus_one_positive_l588_588901

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588901


namespace difference_in_roi_l588_588685

theorem difference_in_roi (E_investment : ℝ) (B_investment : ℝ) (E_rate : ℝ) (B_rate : ℝ) (years : ℕ) :
  E_investment = 300 → B_investment = 500 → E_rate = 0.15 → B_rate = 0.10 → years = 2 →
  (B_rate * B_investment * years) - (E_rate * E_investment * years) = 10 :=
by
  intros E_investment_eq B_investment_eq E_rate_eq B_rate_eq years_eq
  sorry

end difference_in_roi_l588_588685


namespace miquels_theorem_l588_588622

-- Define a triangle ABC with points D, E, F on sides BC, CA, and AB respectively
variables {A B C D E F : Type}

-- Assume we have a function that checks for collinearity of points
def is_on_side (X Y Z: Type) : Bool := sorry

-- Assume a function that returns the circumcircle of a triangle formed by given points
def circumcircle (X Y Z: Type) : Type := sorry 

-- Define the function that checks the intersection of circumcircles
def have_common_point (circ1 circ2 circ3: Type) : Bool := sorry

-- The theorem statement
theorem miquels_theorem (A B C D E F : Type) 
  (hD: is_on_side D B C) 
  (hE: is_on_side E C A) 
  (hF: is_on_side F A B) : 
  have_common_point (circumcircle A E F) (circumcircle B D F) (circumcircle C D E) :=
sorry

end miquels_theorem_l588_588622


namespace regular_polygons_intersections_l588_588115

theorem regular_polygons_intersections 
  (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = 9)
  (h3 : n3 = 10)
  (h4 : n4 = 11)
  (h_no_shared_vertices : ∀ (i j : ℕ), i ≠ j → ¬ (∃ x, x ∈ vertices i ∧ x ∈ vertices j))
  (h_no_three_sides_intersect : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ (∃ x, x ∈ sides i ∧ x ∈ sides j ∧ x ∈ sides k))
  : total_intersections = 1078 := 
by 
  sorry

end regular_polygons_intersections_l588_588115


namespace factorize_expression_l588_588691

variable (x y : ℝ)

theorem factorize_expression : 9 * x^2 * y - y = y * (3 * x + 1) * (3 * x - 1) := 
by
  sorry

end factorize_expression_l588_588691


namespace bianca_total_carrots_l588_588655

theorem bianca_total_carrots :
  let initial_carrots := 23
  let thrown_out_carrots := 10
  let picked_carrots_next_day := 47
  let remaining_carrots := initial_carrots - thrown_out_carrots
  let total_carrots := remaining_carrots + picked_carrots_next_day
  total_carrots = 60 :=
by
  6 sorry

end bianca_total_carrots_l588_588655


namespace sum_of_distances_parabola_circle_l588_588076

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_of_distances_parabola_circle :
  let focus := (0, 1/8 : ℝ)
  let p1 := (-14, 392)
  let p2 := (-1, 2)
  let p3 := (6.5, 84.5)
  let p4 := (8.5, 144.5)
  distance focus p1 + distance focus p2 + distance focus p3 + distance focus p4 = 23065.6875 :=
by
  sorry

end sum_of_distances_parabola_circle_l588_588076


namespace area_of_triangle_l588_588299

noncomputable def area_triangle_equivalence 
(radius : ℝ) (tangent_points : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (centers : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
let O1 := centers.1.1;
    O2 := centers.1.2;
    O3 := centers.2.1 in
let P1 := tangent_points.1.1;
    P2 := tangent_points.1.2;
    P3 := tangent_points.2.1 in
if (P1.1 = P2.1) && (P2.1 = P3.1) && (P3.1 = P1.1) && (P1.2 = P2.2) && (P2.2 = P3.2) && (P3.2 = P1.2) then
(√432) + (√918)
else
0

theorem area_of_triangle : 
  ∀ (ω1 ω2 ω3 : Circle) (P1 P2 P3 : Point) (O1 O2 O3 : Point),
  (radius ω1 = 5) ∧
  (radius ω2 = 5) ∧
  (radius ω3 = 5) ∧
  externally_tangent ω1 ω2 ∧
  externally_tangent ω2 ω3 ∧
  externally_tangent ω3 ω1 ∧
  lies_on_circle P1 ω1 ∧
  lies_on_circle P2 ω2 ∧
  lies_on_circle P3 ω3 ∧
  (dist P1 P2 = dist P2 P3) ∧
  (dist P2 P3 = dist P3 P1) ∧
  tangent_line P1 P2 ω1 ∧
  tangent_line P2 P3 ω2 ∧
  tangent_line P3 P1 ω3 ∧
  (dist O1 O2 = 10) ∧ 
  (dist O2 O3 = 10) ∧ 
  (dist O3 O1 = 10) →
  area_of_triangle (P1, P2, P3) = 1350 :=
begin
  sorry
end

end area_of_triangle_l588_588299


namespace fish_to_apples_l588_588423

variables (f l r a : ℝ)

theorem fish_to_apples (h1 : 3 * f = 2 * l) (h2 : l = 5 * r) (h3 : l = 3 * a) : f = 2 * a :=
by
  -- We assume the conditions as hypotheses and aim to prove the final statement
  sorry

end fish_to_apples_l588_588423


namespace log_y_minus_x_plus_1_gt_0_l588_588845

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588845


namespace prime_division_l588_588081

-- Definitions used in conditions
variables {p q : ℕ}

-- We assume p and q are prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- The problem states
theorem prime_division 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hdiv : divides q (3^p - 2^p)) 
  : p ∣ (q - 1) :=
sorry

end prime_division_l588_588081


namespace smallest_solution_l588_588710

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end smallest_solution_l588_588710


namespace will_bottles_last_l588_588611

theorem will_bottles_last {initial_bottles daily_consumed daily_shared daily_bought : ℝ} 
  (h_initial : initial_bottles = 28) 
  (h_daily_consumed : daily_consumed = 7) 
  (h_daily_shared : daily_shared = 3 / 2) 
  (h_daily_bought : daily_bought = 5 / 3) :
  ⌊initial_bottles / (daily_consumed + daily_shared - daily_bought)⌋ = 4 :=
by
  have net_change_per_day : ℝ := daily_consumed + daily_shared - daily_bought
  have number_of_days : ℝ := initial_bottles / net_change_per_day
  exact floor_eq_iff.mpr ⟨by linarith, by norm_cast; linarith⟩

end will_bottles_last_l588_588611


namespace munchausen_forest_l588_588050

theorem munchausen_forest (E B : ℕ) (h : B = 10 * E) : B > E := by sorry

end munchausen_forest_l588_588050


namespace inequality_ln_pos_l588_588830

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588830


namespace mike_eggs_basket_l588_588094

theorem mike_eggs_basket : ∃ k : ℕ, (30 % k = 0) ∧ (42 % k = 0) ∧ k ≥ 4 ∧ (30 / k) ≥ 3 ∧ (42 / k) ≥ 3 ∧ k = 6 := 
by
  -- skipping the proof
  sorry

end mike_eggs_basket_l588_588094


namespace roots_real_roots_equal_l588_588724

noncomputable def discriminant (a : ℝ) : ℝ :=
  let b := 4 * a
  let c := 2 * a^2 - 1 + 3 * a
  b^2 - 4 * 1 * c

theorem roots_real (a : ℝ) : discriminant a ≥ 0 ↔ a ≤ 1/2 ∨ a ≥ 1 := sorry

theorem roots_equal (a : ℝ) : discriminant a = 0 ↔ a = 1 ∨ a = 1/2 := sorry

end roots_real_roots_equal_l588_588724


namespace sum_of_first_9_terms_is_27_l588_588979

variable {ℕ : Type}
variable (a : ℕ → Int) -- Define the arithmetic sequence
variable (S : ℕ → Int) -- Define the sum of the first n terms

-- Define the arithmetic sequence
axiom a_3_equation : 2 * a 3 = 3 + a 1

-- Define the sum of the first n terms of the sequence
axiom S_n_sum : ∀ n, S n = n * (a 1 + a n) / 2

theorem sum_of_first_9_terms_is_27 : S 9 = 27 :=
by
  have h0 : 2 * a 3 = 3 + a 1 := a_3_equation
  have h1 : S 9 = 9 * (a 1 + a 9) / 2 := by sorry
  have h2 : S 9 = 9 * 3 := by sorry
  show S 9 = 27 from h2

end sum_of_first_9_terms_is_27_l588_588979


namespace dropped_test_score_l588_588239

theorem dropped_test_score (A B C D : ℕ) 
  (h1 : A + B + C + D = 280) 
  (h2 : A + B + C = 225) : 
  D = 55 := 
by sorry

end dropped_test_score_l588_588239


namespace composite_function_increasing_l588_588545

variable {F : ℝ → ℝ}

/-- An odd function is a function that satisfies f(-x) = -f(x) for all x. -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is strictly increasing on negative values if it satisfies the given conditions. -/
def strictly_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 < x2 → x2 < 0 → f x1 < f x2

/-- Combining properties of an odd function and strictly increasing for negative inputs:
  We need to prove that the composite function is strictly increasing for positive inputs. -/
theorem composite_function_increasing (hf_odd : odd_function F)
    (hf_strict_inc_neg : strictly_increasing_on_neg F)
    : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → F (F x1) < F (F x2) :=
  sorry

end composite_function_increasing_l588_588545


namespace tan_15_degree_l588_588248

theorem tan_15_degree : 
  let a := 45 * (Real.pi / 180)
  let b := 30 * (Real.pi / 180)
  Real.tan (a - b) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_15_degree_l588_588248


namespace union_sets_l588_588907

def M := {1, 2}
def N := {2, 3}

theorem union_sets : M ∪ N = {1, 2, 3} := by
  sorry

end union_sets_l588_588907


namespace hoseok_persimmons_l588_588950

/-- Jungkook picked 25 persimmons. If 3 times the persimmons that Hoseok picked is 4 less than the number of persimmons that Jungkook picked, how many persimmons did Hoseok pick? -/
theorem hoseok_persimmons (H : ℕ) (J : ℕ) (h1 : J = 25) (h2 : 3 * H = J - 4) : H = 7 :=
by {
  have h3 : 3 * H = 21, from h2.trans (by rw h1),
  have h4 : H = 21 / 3, from eq_div_of_mul_eq (three_ne_zero _) h3,
  exact nat.div_eq_of_eq_mul_right three_ne_zero h3
}

end hoseok_persimmons_l588_588950


namespace solve_for_x_l588_588130

variable x : ℝ
variable y : ℝ
variable z : ℝ

theorem solve_for_x (h₁ : y = -2.6) (h₂ : z = 4.3) (h₃ : 5 * x - 2 * y + 3.7 * z = 1.45) : x = -3.932 :=
begin
  sorry
end

end solve_for_x_l588_588130


namespace project_completion_days_l588_588630

theorem project_completion_days (A B total_days : ℝ) (hA : A = 10) (hB : B = 30) (h_total : total_days = 15) :
  let work_rate_A := 1 / A,
      work_rate_B := 1 / B,
      combined_work_rate := work_rate_A + work_rate_B,
      x := 10 in
  total_days * combined_work_rate - x * work_rate_A + x * work_rate_B = 1 → x = 10 :=
by {
  intros,
  sorry
}

end project_completion_days_l588_588630


namespace find_N_l588_588230

-- Define the problem parameters
def certain_value : ℝ := 0
def x : ℝ := 10

-- Define the main statement to be proved
theorem find_N (N : ℝ) : 3 * x = (N - x) + certain_value → N = 40 :=
  by sorry

end find_N_l588_588230


namespace probability_even_sum_is_correct_l588_588454

noncomputable def probability_even_sum (X : Finₓ 2012 → ℝ) : ℝ :=
  have h_uniform : ∀ n, 0 < X n ∧ X n ≤ 1 := sorry
  let sum := (Finₓ 2012).sum (λ n, ⌈log (2 ^ (n + 1)) (X n)⌉)
  if sum % 2 = 0 then 1 else 0

theorem probability_even_sum_is_correct :
  probability_even_sum = 2013 / 4025 :=
sorry

end probability_even_sum_is_correct_l588_588454


namespace mass_percentage_Cl_in_mixture_l588_588697

noncomputable def Na_molar_mass : ℝ := 22.99
noncomputable def Cl_molar_mass : ℝ := 35.45
noncomputable def O_molar_mass : ℝ := 16.00

noncomputable def NaClO_molar_mass : ℝ := Na_molar_mass + Cl_molar_mass + O_molar_mass
noncomputable def NaClO2_molar_mass : ℝ := Na_molar_mass + Cl_molar_mass + 2 * O_molar_mass

noncomputable def mass_Cl_in_NaClO (moles : ℕ) : ℝ := moles * Cl_molar_mass
noncomputable def mass_Cl_in_NaClO2 (moles : ℕ) : ℝ := moles * Cl_molar_mass

noncomputable def total_mass_NaClO (moles : ℕ) : ℝ := moles * NaClO_molar_mass
noncomputable def total_mass_NaClO2 (moles : ℕ) : ℝ := moles * NaClO2_molar_mass

theorem mass_percentage_Cl_in_mixture :
  let moles_NaClO := 3
      moles_NaClO2 := 2
      total_mass_Cl := mass_Cl_in_NaClO moles_NaClO + mass_Cl_in_NaClO2 moles_NaClO2
      total_mass_mixture := total_mass_NaClO moles_NaClO + total_mass_NaClO2 moles_NaClO2
  in (total_mass_Cl / total_mass_mixture) * 100 ≈ 43.85 :=
by
  -- Proof omitted
  sorry

end mass_percentage_Cl_in_mixture_l588_588697


namespace segment_length_abs_eq_five_l588_588580

theorem segment_length_abs_eq_five : 
  (length : ℝ) (∀ x : ℝ, abs (x - (27 : ℝ)^(1 : ℝ) / (3 : ℝ)) = 5 → x = 8 ∨ x = -2) 
  → length = 10 := 
begin
  sorry
end

end segment_length_abs_eq_five_l588_588580


namespace log_y_minus_x_plus_1_gt_0_l588_588850

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588850


namespace Dhoni_spent_40_percent_on_rent_l588_588316

noncomputable def rent_percentage_spent := 
  let x := 0.40 in -- 40%
  let dishwasher_expense := 0.80 * x in
  let total_spending := x + dishwasher_expense in
  total_spending = 0.72

theorem Dhoni_spent_40_percent_on_rent : rent_percentage_spent :=
by
  sorry

end Dhoni_spent_40_percent_on_rent_l588_588316


namespace no_infinite_non_constant_arithmetic_progression_with_powers_l588_588322

theorem no_infinite_non_constant_arithmetic_progression_with_powers (a b : ℕ) (b_ge_2 : b ≥ 2) : 
  ¬ ∃ (f : ℕ → ℕ) (d : ℕ), (∀ n : ℕ, f n = (a^(b + n*d)) ∧ b ≥ 2) := sorry

end no_infinite_non_constant_arithmetic_progression_with_powers_l588_588322


namespace minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l588_588712

theorem minimal_solution (x : ℝ) (h : x * |x| = 3 * x + 2) : -2 ≤ x :=
begin
  sorry
end

theorem x_eq_neg_two_is_solution : ( -2 : ℝ ) * |-2| = 3 * -2 + 2 :=
begin
  norm_num,
end

/-- The smallest value of x satisfying x|x| = 3x + 2 is -2 -/
theorem smallest_solution : ∃ x : ℝ, x * |x| = 3 * x + 2 ∧ ∀ y : ℝ, y * |y| = 3 * y + 2 → y ≥ x :=
begin
  use -2,
  split,
  { norm_num },
  { intro y,
    sorry }
end

end minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l588_588712


namespace ratio_supplementary_complementary_l588_588163

theorem ratio_supplementary_complementary (angle : ℝ) (h_angle: angle = 45) :
  let supplementary := 180 - angle,
      complementary := 90 - angle
  in supplementary / complementary = 3 := by
  sorry

end ratio_supplementary_complementary_l588_588163


namespace sqrt_product_eq_225_l588_588123

theorem sqrt_product_eq_225 : (Real.sqrt (5 * 3) * Real.sqrt (3 ^ 3 * 5 ^ 3) = 225) :=
by
  sorry

end sqrt_product_eq_225_l588_588123


namespace distance_between_points_of_tangency_l588_588441

noncomputable def distance_between_tangents (P Q R : Point) (angleQRP : ∠ Q R P = 60°)
(inscribed_circle_radius : ℝ) (tangent_circle_radius : ℝ) 
(inscribed_circle_radius = 2) (tangent_circle_radius = 3) : ℝ :=
√3

theorem distance_between_points_of_tangency (P Q R : Point) (angleQRP : ∠ Q R P = 60°)
(inscribed_circle_radius : ℝ) (tangent_circle_radius : ℝ)
(inscribed_circle_radius = 2) (tangent_circle_radius = 3) :
distance_between_tangents P Q R angleQRP inscribed_circle_radius tangent_circle_radius = √3 :=
sorry

end distance_between_points_of_tangency_l588_588441


namespace bus_people_final_count_l588_588631

theorem bus_people_final_count (initial_people : ℕ) (people_on : ℤ) (people_off : ℤ) :
  initial_people = 22 → people_on = 4 → people_off = -8 → initial_people + people_on + people_off = 18 :=
by
  intro h_initial h_on h_off
  rw [h_initial, h_on, h_off]
  norm_num

end bus_people_final_count_l588_588631


namespace distance_after_time_l588_588647

noncomputable def Adam_speed := 12 -- speed in mph
noncomputable def Simon_speed := 6 -- speed in mph
noncomputable def time_when_100_miles_apart := 100 / 15 -- hours

theorem distance_after_time (x : ℝ) : 
  (Adam_speed * x)^2 + (Simon_speed * x)^2 = 100^2 ->
  x = time_when_100_miles_apart := 
by
  sorry

end distance_after_time_l588_588647


namespace log_pos_given_ineq_l588_588886

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588886


namespace log_pos_given_ineq_l588_588888

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588888


namespace trajectory_equation_distance_ratio_is_constant_l588_588768

variable (θ : ℝ) (x y : ℝ) (P : (ℝ × ℝ)) (A : (ℝ × ℝ)) (M : (ℝ × ℝ))

/-- Define conditions given in the problem. -/
def isOnCurveC := P = (2 * Real.cos θ, 2 * Real.sin θ)
def isMidpointAP := M = ((2 * Real.cos θ + 2) / 2, (2 * Real.sin θ) / 2)
def pointA : A = (2, 0)
def pointE : (ℝ × ℝ) := (3 / 2, 0)
def pointF : (ℝ × ℝ) := (3, 0)

/-- Prove that the rectangular coordinate equation of point M is (x - 1)^2 + y^2 = 1. -/
theorem trajectory_equation (h1 : isOnCurveC θ P) (h2 : pointA A) (h3 : isMidpointAP θ M) :
  (M.1 - 1)^2 + M.2^2 = 1 := sorry

/-- Prove that the ratio of the distances from point M to point E and point F is a constant 1/2. -/
theorem distance_ratio_is_constant (h1 : isOnCurveC θ P) (h2 : pointA A) (h3 : isMidpointAP θ M) :
  ∀ E := pointE F := pointF, (Real.sqrt ((M.1 - E.1)^2 + M.2^2)) / 
    (Real.sqrt ((M.1 - F.1)^2 + M.2^2)) = 1 / 2 := sorry

end trajectory_equation_distance_ratio_is_constant_l588_588768


namespace maximum_value_of_f_on_domain_l588_588152

theorem maximum_value_of_f_on_domain : 
  ∀ (x : ℝ), x ∈ set.Icc (-2 : ℝ) (2 : ℝ) → (x ^ 2 + 2 * x) ≤ (2 ^ 2 + 2 * 2) := 
by 
  intro x hx 
  sorry

end maximum_value_of_f_on_domain_l588_588152


namespace inequality_ln_positive_l588_588857

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588857


namespace third_test_point_0_618_l588_588572

noncomputable def x1 : ℝ := 2 + 0.618 * (4 - 2)
noncomputable def x2 : ℝ := 4 - 0.618 * (4 - 2)
noncomputable def x3 : ℝ := 4 - 0.618 * (4 - x1)

theorem third_test_point_0_618 (interval : set ℝ) (x1 : ℝ) (x2 : ℝ) (x3 : ℝ) (h_interval : interval = set.Icc 2 4) (h_x1 : x1 = 2 + 0.618 * (4 - 2)) (h_x2 : x2 = 4 - 0.618 * (4 - 2)) (h_outcome : x1 > x2) :
  x3 = 4 - 0.618 * (4 - x1) := 
sorry  -- proof skipped as per instructions

end third_test_point_0_618_l588_588572


namespace total_colored_hangers_l588_588031

theorem total_colored_hangers (pink green : ℕ) :
  pink = 7 →
  green = 4 →
  let blue := green - 1 in
  let yellow := blue - 1 in
  pink + green + blue + yellow = 16 :=
by
  intros hp hg
  let blue := green - 1
  let yellow := blue - 1
  sorry

end total_colored_hangers_l588_588031


namespace segment_length_abs_eq_five_l588_588583

theorem segment_length_abs_eq_five : 
  (length : ℝ) (∀ x : ℝ, abs (x - (27 : ℝ)^(1 : ℝ) / (3 : ℝ)) = 5 → x = 8 ∨ x = -2) 
  → length = 10 := 
begin
  sorry
end

end segment_length_abs_eq_five_l588_588583


namespace max_value_of_f_l588_588452

noncomputable def f (x: ℝ) : ℝ := real.sqrt (x * (50 - x)) + real.sqrt (x * (2 - x))
def x0 := 25 / 13
def M := 10

theorem max_value_of_f : ∃ x0, ∃ M, (0 ≤ x0 ∧ x0 ≤ 2) ∧ (∀ x, 0 ≤ x ∧ x ≤ 2 → f x ≤ M) ∧ (f x0 = M) :=
by
  use 25 / 13
  use 10
  sorry

end max_value_of_f_l588_588452


namespace midpoints_of_AC_and_CD_l588_588053

-- Define the conditions
variables (A B C D M N : Point)
variables (area_ABD area_BCD area_ABC : ℝ)

-- The known area ratios
axiom ratio_areas : area_ABD / area_BCD = 3 / 4 ∧ area_ABC / area_BCD = 1 / 4
-- Points M and N on AC and CD satisfying the given condition
axiom M_on_AC : ∃ r : ℝ, 0 < r ∧ r < 1 ∧ r = AM / AC
axiom N_on_CD : ∃ r : ℝ, 0 < r ∧ r < 1 ∧ r = CN / CD
-- Points B, M, and N are collinear
axiom B_M_N_collinear : Collinear B M N

-- Prove that M and N are midpoints of AC and CD respectively
theorem midpoints_of_AC_and_CD :
  (AM = AC / 2) ∧ (CN = CD / 2) :=
sorry

end midpoints_of_AC_and_CD_l588_588053


namespace max_abs_x_l588_588473

theorem max_abs_x 
  (n : ℕ) (h_n : 2 ≤ n) 
  (x : ℕ → ℝ)
  (h_sum : (∑ i in Finset.range n, x i ^ 2) + (∑ i in Finset.range (n-1), x i * x (i + 1)) = 1) :
  ∃ (k : ℕ), k ∈ Finset.range n ∧ (∑ i in Finset.range n, (x i)^2 + (x i * (x (i + 1) * (i ≠ n-1)))) = 
  (Finset.max' (Finset.range n) (fun k => (1 ≤ k ≤ n)) ) ∧  
  ∃ k ≤ n, ∀ i, 1 ≤ i ≤ n → (x i)^2 ≤ (2 * k * ((n+1) - k))/ (n+1) :=
  sorry

end max_abs_x_l588_588473


namespace find_tan_x_l588_588072

-- Let x be a real number between 0 and π/2 such that the given condition holds.
variables {x : ℝ}

-- The given condition
def condition (x : ℝ) : Prop := (x > 0) ∧ (x < (π / 2)) ∧ ((sin x)^4 / 42 + (cos x)^4 / 75 = 1 / 117)

-- The goal is to prove that tan(x) = sqrt(14)/5
theorem find_tan_x (h : condition x) : tan x = (sqrt 14) / 5 :=
sorry

end find_tan_x_l588_588072


namespace eval_expression_l588_588689

noncomputable def i : ℂ := Complex.I

theorem eval_expression : i^8 + i^{18} + i^{-32} = 1 := by
  -- setting the conditions
  have h1 : i^2 = -1 := by sorry
  have h2 : i^4 = 1 := by sorry
  -- skipping the complex proof steps
  sorry

end eval_expression_l588_588689


namespace time_to_10th_floor_l588_588629

theorem time_to_10th_floor (seconds_even : Nat) (seconds_odd : Nat) : 
  seconds_even = 15 ∧ seconds_odd = 9 →
  (10 : Nat) = 10 → 
  (∑ i in (List.range 10).filter (λ n, n % 2 = 0), seconds_even + 1)
  + (∑ i in (List.range 10).filter (λ n, n % 2 = 1), seconds_odd + 1) = 2 * 60 := by
  intros h
  sorry

end time_to_10th_floor_l588_588629


namespace geo_properties_of_inverse_proportional_functions_l588_588002

open Real

def is_inverse_proportional (k : ℝ) (p : ℝ × ℝ) :=
  p.1 ≠ 0 ∧ p.2 = k / p.1

def is_perpendicular_to_x_axis (p : ℝ × ℝ) (c : ℝ × ℝ) :=
  c.1 = p.1 ∧ c.2 = 0

def is_perpendicular_to_y_axis (p : ℝ × ℝ) (d : ℝ × ℝ) :=
  d.1 = 0 ∧ d.2 = p.2

def intersects_func_y_equals_1_div_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, 1 / p.1)

def intersects_func_y_equals_1_div_x_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (1 / p.2, p.2)

def midpoint (a b m : ℝ × ℝ) :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

theorem geo_properties_of_inverse_proportional_functions (k : ℝ) (P C D A B : ℝ × ℝ)
  (h1 : is_inverse_proportional k P)
  (h2 : is_perpendicular_to_x_axis P C)
  (h3 : is_perpendicular_to_y_axis P D)
  (h4 : A = intersects_func_y_equals_1_div_x C)
  (h5 : B = intersects_func_y_equals_1_div_x_y D) :
  let S_triangle (O X Y : ℝ × ℝ) := 0.5 * (X.1 * Y.2 - X.2 * Y.1) in
  S_triangle (0, 0) C A = S_triangle (0, 0) B D ∧
  (P.2 * P.1) - (S_triangle (0, 0) C A + S_triangle (0, 0) B D) = k - 1 ∧
  (midpoint P C A -> midpoint P D B) :=
sorry

end geo_properties_of_inverse_proportional_functions_l588_588002


namespace hexagon_concurrent_midlines_l588_588415

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

-- Define the vertices of the hexagon
variables (A B C D E F : ℝ × ℝ)

-- Hypothesis: Vertices A, C, E lie on a straight line and are non-consecutive
axiom collinear_ACE : ∃ (l : ℝ → ℝ × ℝ), (A = l 0) ∧ (C = l 1) ∧ (E = l 2)
axiom non_consecutive_ACE : A ≠ C ∧ C ≠ E ∧ A ≠ E

-- Midpoints of opposite sides
def M := midpoint A B
def N := midpoint D E
def P := midpoint B C
def Q := midpoint E F
def R := midpoint C D
def S := midpoint F A

-- Lines connecting midpoints of opposite sides
def line_MQ := { x : ℝ × ℝ | ∃ t : ℝ, x = (M.1 + t * (Q.1 - M.1), M.2 + t * (Q.2 - M.2)) }
def line_NP := { x : ℝ × ℝ | ∃ t : ℝ, x = (N.1 + t * (P.1 - N.1), N.2 + t * (P.2 - N.2)) }
def line_RS := { x : ℝ × ℝ | ∃ t : ℝ, x = (R.1 + t * (S.1 - R.1), R.2 + t * (S.2 - R.2)) }

-- Theorem: The lines connecting the midpoints of the opposite sides of the hexagon intersect at a single point
theorem hexagon_concurrent_midlines (h : ∃ (A B C D E F : ℝ × ℝ), collinear_ACE ∧ non_consecutive_ACE) :
  ∃ (p : ℝ × ℝ), p ∈ line_MQ ∧ p ∈ line_NP ∧ p ∈ line_RS :=
sorry

end hexagon_concurrent_midlines_l588_588415


namespace parallelepiped_ratio_l588_588996

-- Definitions of vectors a, b, and c.
variables {V : Type*} [inner_product_space ℝ V]
variable {a b c : V}

-- Definitions of squared distances for T, M, V, U using a, b, c
def PT_sq := ∥a + b + c∥^2
def QM_sq := ∥a - b + c∥^2
def RV_sq := ∥-a + b + c∥^2
def SU_sq := ∥a + b - c∥^2

-- Definitions of squared distances for PQ, PR, PS
def PQ_sq := ∥b∥^2
def PR_sq := ∥c∥^2
def PS_sq := ∥a∥^2

-- The targeted ratio
def ratio := (PT_sq + QM_sq + RV_sq + SU_sq) / (PQ_sq + PR_sq + PS_sq)

-- The proof problem statement in Lean 4
theorem parallelepiped_ratio (h1 : a = a) (h2 : b = b) (h3 : c = c) :
  ratio = 4 := by 
  -- We input the assumptions to clarify constants (a, b, c) for the proof
  have ha := h1, have hb := h2, have hc := h3,
  sorry

end parallelepiped_ratio_l588_588996


namespace find_x_with_sxdx_is_96_l588_588720

def num_divisors (n : Nat) : Nat :=
  Nat.factors n |>.eraseDuplicates |>.length

def sum_divisors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ d => n % d = 0) |>.sum

theorem find_x_with_sxdx_is_96 (x : Nat) (hx_pos : 0 < x) :
  (sum_divisors x) * (num_divisors x) = 96 →
  x = 14 ∨ x = 15 ∨ x = 47 :=
sorry

end find_x_with_sxdx_is_96_l588_588720


namespace log_y_minus_x_plus_1_gt_0_l588_588849

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588849


namespace simplify_sqrt_expression_l588_588126

theorem simplify_sqrt_expression (a b : ℕ) (h_a : a = 5) (h_b : b = 3) :
  (sqrt (a * b) * sqrt ((b ^ 3) * (a ^ 3)) = 225) :=
by
  rw [h_a, h_b]
  sorry

end simplify_sqrt_expression_l588_588126


namespace problem1_l588_588789

variable {a t : ℝ} 
variable {x : ℝ} 
variable (f : ℝ → ℝ) (g : ℝ → ℝ)

def f (x : ℝ) := Real.log x / Real.log a
def g (x : ℝ) := Real.log (2 * x + t - 2) / Real.log a

theorem problem1 (ha : 0 < a ∧ a < 1) (hx : x ∈ Set.Icc (1/4) 2) 
  (h : ∀ x, x ∈ Set.Icc (1/4) 2 → 2 * f x ≥ g x) : t ≥ 2 := 
sorry

end problem1_l588_588789


namespace least_number_of_square_tiles_l588_588614

theorem least_number_of_square_tiles (length : ℕ) (breadth : ℕ) (gcd : ℕ) (area_room : ℕ) (area_tile : ℕ) (num_tiles : ℕ) :
  length = 544 → breadth = 374 → gcd = Nat.gcd length breadth → gcd = 2 →
  area_room = length * breadth → area_tile = gcd * gcd →
  num_tiles = area_room / area_tile → num_tiles = 50864 :=
by
  sorry

end least_number_of_square_tiles_l588_588614


namespace find_n_l588_588762

open Nat

theorem find_n (n : ℕ) (h : ∑ k in range (n + 1), k^3 = 441) : n = 6 :=
begin
  sorry
end

end find_n_l588_588762


namespace inequality_implies_log_pos_l588_588829

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588829


namespace correct_expression_l588_588620

theorem correct_expression :
  (∃ (x : ℝ), log 0.44 = x ∧ log 0.46 < x) ∨
  (∃ (y : ℝ), 1.01^3.4 = y ∧ y > 1.01^3.5) ∨
  (∃ (z : ℝ), 3.4^0.3 = z ∧ z < 3.5^0.3) ∨
  (∃ (a b : ℝ), log 7 6 = a ∧ log 6 7 = b ∧ a < b) := 
begin
  sorry
end

end correct_expression_l588_588620


namespace inequality_ln_pos_l588_588838

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588838


namespace walking_on_same_side_time_l588_588156

theorem walking_on_same_side_time
  (perimeter : ℕ) 
  (side_length : ℕ) 
  (start_distance : ℕ) 
  (speed_A : ℕ) 
  (speed_B : ℕ) 
  (time_to_same_side : ℕ) : 
  (perimeter = 2000) →
  (side_length = 400) →
  (start_distance = 800) →
  (speed_A = 50) →
  (speed_B = 46) →
  (time_to_same_side = 104) →
  ∃ t : ℕ, t = 104 :=
begin
  sorry
end

end walking_on_same_side_time_l588_588156


namespace log_y_minus_x_plus_1_gt_0_l588_588841

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588841


namespace log_y_minus_x_plus_1_gt_0_l588_588844

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588844


namespace sum_of_numbers_l588_588295

theorem sum_of_numbers :
  1357 + 7531 + 3175 + 5713 = 17776 :=
by
  sorry

end sum_of_numbers_l588_588295


namespace find_m_l588_588796

noncomputable def a_vector : ℝ × ℝ := (real.sqrt 3, 1)
noncomputable def b_vector (m : ℝ) : ℝ × ℝ := (m, 1)
noncomputable def angle := 2 * real.pi / 3

theorem find_m (m : ℝ) (h : real.angle a_vector (b_vector m) = angle) : m = -real.sqrt 3 :=
sorry

end find_m_l588_588796


namespace miner_distance_when_explosion_heard_l588_588637

-- Distance function for the miner (in feet)
def miner_distance (t : ℕ) : ℕ := 30 * t

-- Distance function for the sound after the explosion (in feet)
def sound_distance (t : ℕ) : ℕ := 1100 * (t - 45)

theorem miner_distance_when_explosion_heard :
  ∃ t : ℕ, miner_distance t / 3 = 463 ∧ miner_distance t = sound_distance t :=
sorry

end miner_distance_when_explosion_heard_l588_588637


namespace log_y_minus_x_plus_1_pos_l588_588863

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588863


namespace minimize_distances_l588_588330

structure Triangle (α : Type) :=
(A B C : α)

namespace Triangle

variables {α : Type} [MetricSpace α]

def distances (O : α) (t : Triangle α) : (ℝ, ℝ, ℝ) :=
let d₁ := dist O (line_proj t.A t.B t.C),
    d₂ := dist O (line_proj t.B t.C t.A),
    d₃ := dist O (line_proj t.C t.A t.B) in
  (d₁, d₂, d₃)

theorem minimize_distances (t : Triangle α) (O : α) :
  let (x, y, z) := distances O t in
  x / dist t.B t.C = y / dist t.C t.A = z / dist t.A t.B →
  ∀ P : α,
    let (x₁, y₁, z₁) := distances P t in
    x*x + y*y + z*z ≤ x₁*x₁ + y₁*y₁ + z₁*z₁ :=
sorry

end Triangle

end minimize_distances_l588_588330


namespace tangent_line_at_1_ln_inequality_l588_588380

-- Problem: Define the function f and prove related properties
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x + 1)

theorem tangent_line_at_1 :
  let y := f 1 in 
  (∀ x, y = (1/2) * (x - 1) → x - 2 * y - 1 = 0) :=
by
  sorry

theorem ln_inequality (n : ℕ) (h : 2 ≤ n) :
  Real.log n < ∑ i in Finset.range n, (1 / i) - (1 / 2) - (1 / (2 * n)) :=
by
  sorry

end tangent_line_at_1_ln_inequality_l588_588380


namespace total_cutlery_pieces_l588_588294

variable (knives := 30)
variable (forks := 45)
variable (teaspoons := 2.5 * knives : ℝ)
variable (additional_knives := (knives * (Real.sqrt 2) / 4) : ℝ)
variable (additional_forks := (forks * Real.sqrt (3 / 2)) : ℝ)
variable (additional_teaspoons := (teaspoons * (Real.sqrt 5) / 6) : ℝ)

noncomputable def total_knives := (knives + additional_knives)
noncomputable def total_forks := (forks + additional_forks)
noncomputable def total_teaspoons := (teaspoons + additional_teaspoons)
noncomputable def total_cutlery :=
  total_knives + total_forks + total_teaspoons

theorem total_cutlery_pieces :
  total_cutlery = 150 + (7.5 * Real.sqrt 2 + 45 * Real.sqrt (3 / 2) + 12.5 * Real.sqrt 5) := by
  sorry

end total_cutlery_pieces_l588_588294


namespace chessboard_pawns_adjacency_preservation_l588_588994

theorem chessboard_pawns_adjacency_preservation:
  (∀ (F : ℕ × ℕ → ℕ × ℕ) (n : ℕ),
    n > 0 ∧ 
    -- Conditions:
    (∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → F(i, j) == (i, j)) →
    (∀ i : ℕ, F(1, 1) = (1, 1) ∧ F(n, 1) = (n, 1)) ∧
    (∀ i j : ℕ, (abs ((i, j).fst - (F (i, j)).fst) + abs ((i, j).snd - (F (i, j)).snd)) = 1) →
    -- Conclusion:
    (∀ i j : ℕ, F(i, j) == (i, j))) :=
sorry

end chessboard_pawns_adjacency_preservation_l588_588994


namespace find_coordinates_find_slope_range_l588_588073

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the dot product condition for the first part
def dot_product_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  let PF1 := (x + Real.sqrt 3, y)
  let PF2 := (x - Real.sqrt 3, y)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = -5 / 4

-- First proof problem: coordinates of P
theorem find_coordinates (x y : ℝ) (hx1 : x > 0) (hy1 : y > 0) (h : ellipse x y) (h_dot : dot_product_condition (x, y)) :
  (x, y) = (1, Real.sqrt 3 / 2) := by
  sorry

-- Define the line passing through M(0, 2)
def line (k x : ℝ) : ℝ := k * x + 2

-- Define the conditions for part 2
def acute_angle_condition (k : ℝ) : Prop :=
  let discriminant := (16 * k)^2 - 4 * (1 + 4 * k^2) * 12
  let k_range1 := k^2 > 3 / 4
  let k_range2 := (4 * (4 - k^2)) / (1 + 4 * k^2) > 0
  k_range1 ∧ k_range2

-- Second proof problem: range of slope k
theorem find_slope_range (k : ℝ) (h : acute_angle_condition k) :
  k ∈ Set.Ioo (-(2 : ℝ)) (-(Real.sqrt 3) / 2) ∪ Set.Ioo ((Real.sqrt 3) / 2) (2 : ℝ) := by
  sorry

end find_coordinates_find_slope_range_l588_588073


namespace find_value_of_sum_of_constants_l588_588077

theorem find_value_of_sum_of_constants 
  (a : ℝ) (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} : ℝ) 
  (h : a + a_1 * (x + 2) + a_2 * (x + 2)^2 + a_3 * (x + 2)^3 + a_4 * (x + 2)^4 + a_5 * (x + 2)^5 + a_6 * (x + 2)^6 + a_7 * (x + 2)^7 + a_8 * (x + 2)^8 + a_9 * (x + 2)^9 + a_{10} * (x + 2)^{10} + a_{11} * (x + 2)^{11} + a_{12} * (x + 12)^{12} = (x^2 - 2 * x - 2)^6) : 
  2 * a_2 + 6 * a_3 + 12 * a_4 + 20 * a_5 + 30 * a_6 + 42 * a_7 + 56 * a_8 + 72 * a_9 + 90 * a_{10} + 110 * a_{11} + 132 * a_{12} = 492 := 
by 
  sorry

end find_value_of_sum_of_constants_l588_588077


namespace log_pos_given_ineq_l588_588893

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588893


namespace find_meeting_time_l588_588982

-- Define the context and the problem parameters
def lisa_speed : ℝ := 9  -- Lisa's speed in mph
def adam_speed : ℝ := 7  -- Adam's speed in mph
def initial_distance : ℝ := 6  -- Initial distance in miles

-- The time in minutes for Lisa to meet Adam
theorem find_meeting_time : (initial_distance / (lisa_speed + adam_speed)) * 60 = 22.5 := by
  -- The proof is omitted for this statement
  sorry

end find_meeting_time_l588_588982


namespace probability_mist_tiles_l588_588324

theorem probability_mist_tiles (total_tiles : ℕ) (s_count t_count i_count : ℕ) :
  total_tiles = 12 →
  s_count = 3 →
  t_count = 3 →
  i_count = 2 →
  (s_count + t_count + i_count) = 8 →
  (8 / total_tiles : ℚ) = 2 / 3 :=
by
  intros h_total_tiles h_s_count h_t_count h_i_count h_sum_tiles
  rw [h_total_tiles, h_sum_tiles]
  norm_num
  sorry

end probability_mist_tiles_l588_588324


namespace inequality_implies_log_pos_l588_588827

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588827


namespace at_least_sqrt_n_plus_one_diff_residues_l588_588458

theorem at_least_sqrt_n_plus_one_diff_residues {n : ℕ} (h : n ≥ 5) (a : Fin n → ℕ)
  (ha : (Finset.univ.image a).val = List.range n)
  : ∃ s : Finset ℕ, s.card ≥ ⌊Real.sqrt n⌋₊ + 1 ∧ ∀ (i : Fin n), (Finset.univ.image (λ i, ∑ j in Finset.range (i + 1), a j % n)).val.to_finset = s :=
sorry

end at_least_sqrt_n_plus_one_diff_residues_l588_588458


namespace part1_solution_set_l588_588734

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part1_solution_set : {x : ℝ | f x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end part1_solution_set_l588_588734


namespace sum_of_squares_diagonals_of_rhombus_l588_588411

theorem sum_of_squares_diagonals_of_rhombus (d1 d2 : ℝ) (h : (d1 / 2)^2 + (d2 / 2)^2 = 4) : d1^2 + d2^2 = 16 :=
sorry

end sum_of_squares_diagonals_of_rhombus_l588_588411


namespace boys_seen_l588_588807

theorem boys_seen (total_eyes : ℕ) (eyes_per_boy : ℕ) (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : total_eyes / eyes_per_boy = 23 := 
by 
  sorry

end boys_seen_l588_588807


namespace sin_690_l588_588659

-- Defining the known conditions as hypotheses:
axiom sin_periodic (x : ℝ) : Real.sin (x + 360) = Real.sin x
axiom sin_odd (x : ℝ) : Real.sin (-x) = - Real.sin x
axiom sin_thirty : Real.sin 30 = 1 / 2

theorem sin_690 : Real.sin 690 = -1 / 2 :=
by
  -- Proof would go here, but it is skipped with sorry.
  sorry

end sin_690_l588_588659


namespace Adam_total_shopping_cost_l588_588290

theorem Adam_total_shopping_cost :
  let sandwiches := 3
  let sandwich_cost := 3
  let water_cost := 2
  (sandwiches * sandwich_cost + water_cost) = 11 := 
by
  sorry

end Adam_total_shopping_cost_l588_588290


namespace overall_percentage_decrease_l588_588446

theorem overall_percentage_decrease (P x y : ℝ) (hP : P = 100) 
  (h : (P - (x / 100) * P) - (y / 100) * (P - (x / 100) * P) = 55) : 
  ((P - 55) / P) * 100 = 45 := 
by 
  sorry

end overall_percentage_decrease_l588_588446


namespace smallest_obtuse_triangles_l588_588670

def obtuseTrianglesInTriangulation (n : Nat) : Nat :=
  if n < 3 then 0 else (n - 2) - 2

theorem smallest_obtuse_triangles (n : Nat) (h : n = 2003) :
  obtuseTrianglesInTriangulation n = 1999 := by
  sorry

end smallest_obtuse_triangles_l588_588670


namespace probability_multiple_of_3_l588_588174

theorem probability_multiple_of_3:
  let tickets := Finset.range 20 |>.map (λ x => x + 1)
  let multiples_of_3 := tickets.filter (λ x => x % 3 = 0)
  let probability := (multiples_of_3.card : ℚ) / tickets.card
  probability = 3 / 10 := by
sorry

end probability_multiple_of_3_l588_588174


namespace inverse_of_B_cubed_l588_588909

theorem inverse_of_B_cubed
  (B_inv : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3, -1],
    ![0, 5]
  ]) :
  (B_inv ^ 3) = ![
    ![27, -49],
    ![0, 125]
  ] := 
by
  sorry

end inverse_of_B_cubed_l588_588909


namespace shaded_fraction_is_correct_l588_588642

-- Definitions based on the identified conditions
def initial_fraction_shaded : ℚ := 4 / 9
def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)
def infinite_series_fraction_shaded : ℚ := 4 / 9 * (4 / 3)

-- The theorem stating the problem
theorem shaded_fraction_is_correct :
  infinite_series_fraction_shaded = 16 / 27 :=
by
  sorry -- proof to be provided

end shaded_fraction_is_correct_l588_588642


namespace log_of_y_sub_x_plus_one_positive_l588_588899

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588899


namespace hydrogen_atoms_in_compound_l588_588265

theorem hydrogen_atoms_in_compound :
  let molecular_weight := 122
  let carbons := 7
  let oxygens := 2
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  let mass_C := carbons * atomic_weight_C
  let mass_O := oxygens * atomic_weight_O
  let total_mass_CO := mass_C + mass_O
  let mass_H := molecular_weight - total_mass_CO
  6 = Int.round (mass_H / atomic_weight_H) := by
    sorry

end hydrogen_atoms_in_compound_l588_588265


namespace find_triple_matrix_l588_588699

variable {R : Type*} [CommRing R]

theorem find_triple_matrix (a b c d x y z w : R) :
  (∀ a b c d : R, matrix.mul (matrix.vec_cons x (matrix.vec_cons y matrix.vec_empty)) 
                              (matrix.vec_cons (matrix.vec_cons a (matrix.vec_cons b matrix.vec_empty)) 
                                              (matrix.vec_cons (matrix.vec_cons c (matrix.vec_cons d matrix.vec_empty)) 
                                                              matrix.vec_empty)) 
                            = matrix.vec_cons (matrix.vec_cons (3 * a) (matrix.vec_cons (3 * b) matrix.vec_empty)) 
                                               (matrix.vec_cons (matrix.vec_cons (3 * c) (matrix.vec_cons (3 * d) matrix.vec_empty)) 
                                                               matrix.vec_empty)) 
  → (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 3) := by
  sorry

end find_triple_matrix_l588_588699


namespace segment_length_abs_cubed_root_l588_588599

theorem segment_length_abs_cubed_root (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  ∃ a b : ℝ, a = 3 + 5 ∧ b = 3 - 5 ∧ (b - a).abs = 10 :=
by {
  have h1 : real.cbrt 27 = 3 := by norm_num,
  rw h1 at h,
  have h2 : |x - 3| = 5 := h,
  use [8, -2],
  split,
  { refl },
  { split,
    { refl },
    { norm_num } }
}

end segment_length_abs_cubed_root_l588_588599


namespace largest_A_divisible_by_8_equal_quotient_remainder_l588_588603

theorem largest_A_divisible_by_8_equal_quotient_remainder :
  ∃ (A B C : ℕ), A = 8 * B + C ∧ B = C ∧ C < 8 ∧ A = 63 := by
  sorry

end largest_A_divisible_by_8_equal_quotient_remainder_l588_588603


namespace log_pos_if_exp_diff_l588_588876

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588876


namespace log_pos_if_exp_diff_l588_588884

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588884


namespace distance_between_parallel_lines_l588_588144

theorem distance_between_parallel_lines :
  let A := 3
  let B := 2
  let C1 := -1
  let C2 := 1 / 2
  let d := |C2 - C1| / Real.sqrt (A^2 + B^2)
  d = 3 / Real.sqrt 13 :=
by
  -- Proof goes here
  sorry

end distance_between_parallel_lines_l588_588144


namespace post_break_processing_orders_l588_588422

theorem post_break_processing_orders : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (k + 2)) = 1536 := 
by
  sorry

end post_break_processing_orders_l588_588422


namespace trapezium_distance_l588_588694

theorem trapezium_distance (h : ℝ) (a b A : ℝ) 
  (h_area : A = 95) (h_a : a = 20) (h_b : b = 18) :
  A = (1/2 * (a + b) * h) → h = 5 :=
by
  sorry

end trapezium_distance_l588_588694


namespace inequality_implies_log_pos_l588_588825

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588825


namespace original_quadratic_function_l588_588414

theorem original_quadratic_function (f : ℝ → ℝ) (h_vertex : f 0 = 0) (h_translated : ∃ a b c : ℝ, ∀ x : ℝ, f (x - b/2a) + (b^2 - 4*a*c)/(4*a) = 2*x^2 + x - 1) : 
  ∀ x : ℝ, f x = 2*x^2 := 
by sorry

end original_quadratic_function_l588_588414


namespace problem_l588_588812

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588812


namespace log_pos_if_exp_diff_l588_588879

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588879


namespace larry_wins_probability_l588_588448

theorem larry_wins_probability :
  (∃ rounds : ℕ, rounds ≤ 5 ∧
    ∀ (p_larry p_julius : ℝ), 
    p_larry = 3 / 4 ∧ 
    p_julius = 1 / 4 ∧ 
    let win_probability :=
      -- Probability Larry wins in the 1st round
      p_larry +
      -- Probability Larry wins in the 3rd round
      (1 - p_larry) * (1 - p_julius) * p_larry +
      -- Probability Larry wins in the 5th round
      (1 - p_larry) * (1 - p_julius) * (1 - p_larry) * (1 - p_julius) * p_larry
    in win_probability
  )
  = 825 / 1024 :=
sorry

end larry_wins_probability_l588_588448


namespace nancy_spelling_problems_l588_588343

structure NancyProblems where
  math_problems : ℝ
  rate : ℝ
  hours : ℝ
  total_problems : ℝ

noncomputable def calculate_spelling_problems (n : NancyProblems) : ℝ :=
  n.total_problems - n.math_problems

theorem nancy_spelling_problems :
  ∀ (n : NancyProblems), n.math_problems = 17.0 ∧ n.rate = 8.0 ∧ n.hours = 4.0 ∧ n.total_problems = 32.0 →
  calculate_spelling_problems n = 15.0 :=
by
  intros
  sorry

end nancy_spelling_problems_l588_588343


namespace solve_log_inequality_l588_588519

theorem solve_log_inequality (x : ℝ) :
  (log (x - 3) (x - 1) < 2) ↔ (x > 5 ∨ (3 < x ∧ x < 4)) :=
by
  sorry

end solve_log_inequality_l588_588519


namespace shaded_area_of_triangle_CDE_l588_588198

-- Definitions of the points
noncomputable def O := (0, 0 : ℝ×ℝ)
noncomputable def A := (4, 0 : ℝ×ℝ)
noncomputable def B := (16, 0 : ℝ×ℝ)
noncomputable def C := (16, 12 : ℝ×ℝ)
noncomputable def D := (4, 12 : ℝ×ℝ)
noncomputable def E := (4, 3 : ℝ×ℝ)

-- Definition of the area calculation for the given triangle
theorem shaded_area_of_triangle_CDE : 
  let DE := 9 in
  let DC := 12 in
  (DE * DC) / 2 = 54 :=
by
  sorry

end shaded_area_of_triangle_CDE_l588_588198


namespace coplanar_lines_parallel_l588_588621

theorem coplanar_lines_parallel (α : Plane) (m n : Line) 
  (h1 : m ⊆ α)
  (h2 : parallel n α)
  (h3 : coplanar m n) : 
  parallel m n :=
sorry

end coplanar_lines_parallel_l588_588621


namespace inequality_ln_positive_l588_588860

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588860


namespace probability_not_siblings_l588_588238

-- Definitions based on conditions
def people_in_room : ℕ := 7
def have_1_sibling : ℕ := 4
def have_2_siblings : ℕ := 3

-- Main statement to prove
theorem probability_not_siblings : 
  (have_1_sibling = 4 ∧ have_2_siblings = 3 ∧ people_in_room = 7) →
  (∃ p : ℚ, p = 16 / 21) :=
by
  intro h
  use 16 / 21
  sorry

end probability_not_siblings_l588_588238


namespace pablo_books_read_l588_588109

noncomputable def pages_per_book : ℕ := 150
noncomputable def cents_per_page : ℕ := 1
noncomputable def cost_of_candy : ℕ := 1500    -- $15 in cents
noncomputable def leftover_money : ℕ := 300    -- $3 in cents
noncomputable def total_money := cost_of_candy + leftover_money
noncomputable def earnings_per_book := pages_per_book * cents_per_page

theorem pablo_books_read : total_money / earnings_per_book = 12 := by
  sorry

end pablo_books_read_l588_588109


namespace option_A_option_B_option_D_l588_588742

-- Define the assumptions
variables {f : ℝ → ℝ}
axiom (h_even : ∀ x : ℝ, f (-x) = f x)
axiom (h_monotone_decreasing : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2)
axiom (h_f_neg1_zero : f (-1) = 0)

-- Prove option A: f(3) > f(4)
theorem option_A : f 3 > f 4 := 
by {
  have h_pos3 : 0 < 3 := by norm_num,
  have h_pos4 : 0 < 4 := by norm_num,
  have h_le : 3 < 4 := by norm_num,
  exact h_monotone_decreasing 3 4 h_pos3 h_le
}

-- Prove option B: If f(m-1) < f(2), then m < -1 or m > 3
theorem option_B (m : ℝ) : f (m-1) < f 2 → m < -1 ∨ m > 3 := 
by {
  intro h,
  cases lt_or_ge m 1 with h1 h2,
  { right,
    have h3 : (m - 1) < 0 := sub_neg_of_lt h1,
    have h5 : 0 < 2 := by norm_num,
    have h4 : f (m-1) ≥ f (-1) := le_of_not_lt (λ h_lt, (h_monotone_decreasing (m - 1) (-1 + 1) h3 (by linarith)).not_le (by linarith)),
    linarith [h4] },
  { left,
    linarith, }
}

-- Prove option D: There exists an upper bound for f(x)
theorem option_D : ∃ (m : ℝ), ∀ x : ℝ, f x ≤ m := 
by {
  use f 0,
  intro x,
  by_cases h_pos : 0 ≤ x,
  { have h_le : x = 0 ∨ 0 < x, from le_iff_eq_or_lt.mp h_pos,
    cases h_le with h_eq_zero h_gt_zero,
    { rw h_eq_zero },
    { exact (h_monotone_decreasing 0 x zero_lt_one h_gt_zero).le }},
  { have h_neg : -x > 0, from neg_pos.mpr (lt_of_not_ge h_pos),
    have h_symm : f x = f (-x), by rw h_even x,
    have h_le_neg : f x = f (-x), from h_even x,
    linarith [h_le_neg, (h_monotone_decreasing 0 (-x) zero_lt_one h_neg).le] }
}

end option_A_option_B_option_D_l588_588742


namespace log_y_minus_x_plus_1_pos_l588_588866

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588866


namespace log_pos_if_exp_diff_l588_588878

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588878


namespace evaluate_three_star_twostar_one_l588_588027

def operator_star (a b : ℕ) : ℕ :=
  a^b - b^a

theorem evaluate_three_star_twostar_one : operator_star 3 (operator_star 2 1) = 2 := 
  by
    sorry

end evaluate_three_star_twostar_one_l588_588027


namespace inequality_implies_log_pos_l588_588823

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588823


namespace maximum_monthly_profit_l588_588604

-- Let's set up our conditions

def selling_price := 25
def monthly_profit := 120
def cost_price := 20
def selling_price_threshold := 32
def relationship (x n : ℝ) := -10 * x + n

-- Define the value of n
def value_of_n : ℝ := 370

-- Profit function
def profit_function (x n : ℝ) : ℝ := (x - cost_price) * (relationship x n)

-- Define the condition for maximum profit where the selling price should be higher than 32
def max_profit_condition (n : ℝ) (x : ℝ) := x > selling_price_threshold

-- Define what the maximum profit should be
def max_profit := 160

-- The main theorem to be proven
theorem maximum_monthly_profit :
  (relationship selling_price value_of_n = monthly_profit) →
  max_profit_condition value_of_n 32 →
  profit_function 32 value_of_n = max_profit :=
by sorry

end maximum_monthly_profit_l588_588604


namespace james_travel_time_to_canada_l588_588064

def total_time : ℝ :=
  let time1 := 200 / 60
  let time2 := 120 / 50
  let time3 := 1.5
  let time4 := 250 / 65
  let rest_time := 1
  let stop_time := 0.5
  time1 + time2 + time3 + time4 + rest_time + stop_time

theorem james_travel_time_to_canada : total_time = 12.579 := 
sorry

end james_travel_time_to_canada_l588_588064


namespace number_of_subsets_l588_588001

open Set

theorem number_of_subsets {A B : Set ℕ} (hA : A = {0, 1, 2, 3}) (hB : B = {1, 2, 4}) :
  (C = A ∩ B) → (fintype.card (Set.Powerset C) = 4) :=
  by
   sorry

end number_of_subsets_l588_588001


namespace simplify_sqrt_expression_l588_588124

theorem simplify_sqrt_expression (a b : ℕ) (h_a : a = 5) (h_b : b = 3) :
  (sqrt (a * b) * sqrt ((b ^ 3) * (a ^ 3)) = 225) :=
by
  rw [h_a, h_b]
  sorry

end simplify_sqrt_expression_l588_588124


namespace actual_height_l588_588140

theorem actual_height (incorrect_avg : ℝ) (recorded_height : ℝ) (actual_avg : ℝ) (n : ℕ) :
  n = 20 →
  incorrect_avg = 175 →
  recorded_height = 151 →
  actual_avg = 174.25 →
  ∃ h : ℝ, h = 166 :=
by
  intros h20 havg hrec hact
  use 166
  sorry

end actual_height_l588_588140


namespace angle_PFQ_is_90_degrees_l588_588172

theorem angle_PFQ_is_90_degrees (p : ℝ) (hp : p > 0)
    (P : ℝ × ℝ) (A B : ℝ × ℝ)
    (hP_outside : ¬(∃ (x : ℝ), P = (x, real.sqrt (2 * p * x))))
    (hA : A.2 ^ 2 = 2 * p * A.1)
    (hB : B.2 ^ 2 = 2 * p * B.1)
    (Q : ℝ × ℝ)
    (hQ : Q ≠ (p / 2, 0))
    (circumcenter : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ))
    (hQ_def : Q = circumcenter P A B) :
  let F := (p / 2, 0) in
  let FP := (P.1 - F.1, P.2 - F.2) in
  let FQ := (Q.1 - F.1, Q.2 - F.2) in
  FP.1 * FQ.1 + FP.2 * FQ.2 = 0 := 
begin
  -- The proof goes here
  sorry
end

end angle_PFQ_is_90_degrees_l588_588172


namespace problem_statement_D_l588_588772

-- Definitions based on the problem conditions
variables {α β γ : Type} [plane α] [plane β] [plane γ]
variables {a b : Type} [line a] [line b]
variable  (P : point)

-- Hypotheses based on the given conditions
hypothesis non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
hypothesis non_coincident_lines : a ≠ b

-- Statement D: The target proof in Lean 4
theorem problem_statement_D
  (h1 : a ⊥ α)
  (h2 : intersects a b P) :
  ¬(b ⊥ α) :=
sorry

end problem_statement_D_l588_588772


namespace solve_quadratic_equation_l588_588131

theorem solve_quadratic_equation : 
  (∀ x : ℝ, 2 * x^2 - 3 * x = 1 - 2 * x ↔ (x = 1 ∨ x = -1/2)) :=
by {
  intro x,
  split,
  {
    intro h,
    let q := 2 * x^2 - 3 * x + 2 * x - 1,
    have h_sq_eq_0 : q = 0 := by linarith [h],
    have factorized_eq := ((quadratic_eq_roots_of_discriminant _ _ factorized_eq using_discriminant).spec x).mp h_sq_eq_0,
    cases factorized_eq,
    {
      rw factorized_eq,
      left, 
      refl
    },
    {
      rw factorized_eq,
      right,
      refl
    },
  },
  {
    intro h,
    cases h,
    {
      rw h,
      linarith
    },
    {
      rw h,
      linarith
    },
  },
}

end solve_quadratic_equation_l588_588131


namespace circle_good_point_existence_l588_588624

-- Definition of a good point
def is_good_point (circ : list ℤ) (i : ℕ) : Prop :=
  ∀ j k, j ≤ k → k < circ.length →
    circ.rotate i j k > 0

theorem circle_good_point_existence :
  ∀ (circ : list ℤ), circ.length = 1985 → (circ.count (-1) < 662) →
  ∃ i, is_good_point circ i :=
by
  intro circ
  intro h_length h_count
  sorry

end circle_good_point_existence_l588_588624


namespace distance_foci_l588_588722

-- Define the parameters for the ellipse
def center : ℝ × ℝ := (2, 3)
def a : ℝ := 8
def b : ℝ := 3

-- Define the distance between the foci
def distance_between_foci : ℝ := 2 * Real.sqrt (a^2 - b^2)

-- Statement that we need to prove
theorem distance_foci (h1 : a = 8) 
                      (h2 : b = 3) :
  distance_between_foci = 2 * Real.sqrt 55 := by
  -- Assume the conditions are given
  rw [h1, h2]
  -- Now simplify the main statement
  simp [distance_between_foci]
  sorry

end distance_foci_l588_588722


namespace log_y_minus_x_plus_1_gt_0_l588_588843

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588843


namespace inequality_ln_pos_l588_588834

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588834


namespace bus_stoppage_time_l588_588615

-- Define the speeds given in the conditions
def speed_without_stoppages : ℝ := 64 -- km/hr
def speed_with_stoppages : ℝ := 48 -- km/hr

-- Define the question as a Lean theorem statement
theorem bus_stoppage_time :
  ∃ (t : ℝ), t = 15 ∧ (1 - speed_with_stoppages / speed_without_stoppages) = t / 60 :=
by
  -- Definitions from the conditions
  let speed_difference := speed_without_stoppages - speed_with_stoppages
  let stoppage_fraction := speed_difference / speed_without_stoppages -- = 1/4

  -- Expected stoppage time in minutes
  use 15
  have h : speed_difference = 16, from rfl
  have h1 : speed_with_stoppages / speed_without_stoppages = 48 / 64, from rfl
  have h2 : 1 - 48 / 64 = 1 / 4, from by norm_num

  split
  {
    exact rfl
  }
  {
    exact h2
  }

end bus_stoppage_time_l588_588615


namespace smallest_positive_a_exists_l588_588679

theorem smallest_positive_a_exists :
  ∃ a : ℝ, 0 < a ∧ (∀ b : ℝ, 0 < b → (∃ n : ℤ, a ≤ n ∧ n ≤ 2016 * a) ↔ (∀ n : ℤ, a ≤ n ∧ n ≤ 2016 * a ↔ 2016)) ∧ a = 2017 / 2016 :=
sorry

end smallest_positive_a_exists_l588_588679


namespace number_of_true_propositions_l588_588776

def proposition1 : Prop :=
  ∀ (k : ℝ), |AB| = 5 → 
  (∃ l : ℝ, l = y - k * (x - 3) ∧ 
  ∀ A B : ℝ, 
  (\frac{x^2}{4} - \frac{y^2}{5} = 1) ∧ 
  (l = y = k * (x - 3)))

def proposition2 : Prop :=
  ∀ (O A B C P : ℝ),
  (∃ vOA vOB vOC : ℝ, vOA = OA ∧ vOB = OB ∧ vOC = OC) →
  (P = \frac{1}{6} * OA + \frac{1}{3} * OB + \frac{1}{2} * OC) →
  coplanar P A B C

def proposition3 : Prop :=
  ∀ (O A B C P : ℝ),
  (∃ vOA vOB vOC : ℝ, vOA = OA ∧ vOB = OB ∧ vOC = OC) →
  (P = 2 * OA - OB + 2 * OC) →
  ¬ coplanar P A B C

def proposition4 : Prop :=
  ∀ (θ ρ : ℝ),
  (θ = \frac{\pi}{3}) →
  (ρ = \frac{1}{1 - 2 * cos θ}) →
  ¬ ∃ (θ ρ : ℝ),  (θ = \frac{\pi}{3}) ∧ (ρ = \frac{1}{1 - 2 * cos θ})

theorem number_of_true_propositions : 
  (proposition1 ↔ False) ∧ 
  (proposition2 ↔ True) ∧ 
  (proposition3 ↔ True) ∧ 
  (proposition4 ↔ True) →
  (number_of_true_propositions = 3) := by
  sorry

end number_of_true_propositions_l588_588776


namespace find_smallest_k_l588_588721

noncomputable def minimal_k (m n : ℕ) (h1 : 2 ≤ m) (h2 : m < n) (h3 : Nat.gcd m n = 1) : ℤ :=
  (n - 1) * (m - 1) / 2 + m

theorem find_smallest_k (m n k : ℕ) (h1 : 2 ≤ m) (h2 : m < n) (h3 : Nat.gcd m n = 1) :
  (∀ I : Finset ℕ, I.card = m ∧ I ⊆ Finset.range (n+1) → 
    (Finset.sum (λ x => x) I > k → 
      ∃ a : ℕ → ℝ, (∀ i, i ∈ Finset.range (n+1) → a i ≤ a (i+1)) ∧ 
      (1/m : ℝ) * (Finset.sum a I) > (1/n : ℝ) * (Finset.sum a (Finset.range (n+1)))))
  ↔ k = minimal_k m n h1 h2 h3 := by sorry

end find_smallest_k_l588_588721


namespace club_truncator_probability_l588_588666

theorem club_truncator_probability :
  let P_more_wins := (2741 : ℚ) / (6561 : ℚ) in
  let m := 2741 in
  let n := 6561 in
  m + n = 9302 :=
sorry

end club_truncator_probability_l588_588666


namespace rainfall_difference_correct_l588_588062

def rainfall_difference (monday_rain : ℝ) (tuesday_rain : ℝ) : ℝ :=
  monday_rain - tuesday_rain

theorem rainfall_difference_correct : rainfall_difference 0.9 0.2 = 0.7 :=
by
  simp [rainfall_difference]
  sorry

end rainfall_difference_correct_l588_588062


namespace josie_shopping_time_l588_588104

theorem josie_shopping_time : 
  let wait_cart := 3
  let wait_employee := 13
  let wait_stocker := 14
  let wait_checkout := 18
  let total_trip_time := 90
  total_trip_time - (wait_cart + wait_employee + wait_stocker + wait_checkout) = 42 :=
by
  -- Conditions (definitions of waiting times and total trip time)
  let wait_cart := 3
  let wait_employee := 13
  let wait_stocker := 14
  let wait_checkout := 18
  let total_trip_time := 90

  -- Proof statement using the conditions
  calc
    total_trip_time - (wait_cart + wait_employee + wait_stocker + wait_checkout)
    = 90 - (3 + 13 + 14 + 18) : by rw [total_trip_time]
    ... = 90 - 48 : by norm_num
    ... = 42 : by norm_num

end josie_shopping_time_l588_588104


namespace parallel_lines_sufficient_but_not_necessary_l588_588003

theorem parallel_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = -3 → ∀ x y : ℝ, (x, y) ∈ {(x, y) | ax + y = 1} → (x, y) ∈ {(x, y) | 9x + ay = 1}) ∧
  (∃ a : ℝ, a ≠ -3 ∧ ∀ x y : ℝ, (x, y) ∈ {(x, y) | ax + y = 1} ∧ (x, y) ∈ {(x, y) | 9x + ay = 1}) :=
sorry

end parallel_lines_sufficient_but_not_necessary_l588_588003


namespace part1_solution_set_l588_588735

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part1_solution_set : {x : ℝ | f x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end part1_solution_set_l588_588735


namespace function_characteristics_l588_588522

def Quadrant1 (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f x > 0

def Quadrant2 (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ f x > 0

def IncreasingInFirstQuadrant (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2) → f x1 < f x2

theorem function_characteristics :
  ∀ f : ℝ → ℝ,
  Quadrant1 f ∧ Quadrant2 f ∧ IncreasingInFirstQuadrant f →
  f = (λ x, 2 * x + 3) :=
  by
    intros f h
    sorry

end function_characteristics_l588_588522


namespace binomial_expansion_third_term_l588_588142

theorem binomial_expansion_third_term 
  (n : ℕ) 
  (h : binomial_coefficient n 2 * (1/4:ℚ) = 7) : 
  n = 8 := 
sorry

end binomial_expansion_third_term_l588_588142


namespace equality_conditions_hold_l588_588417

-- Let's first define the trigonometric functions and the conditions given.
variables {A B C λ : ℝ}

-- angles of a triangle
axiom triangle_angles_sum : A + B + C = π

-- defining the cosine and sine functions
noncomputable def cos (x : ℝ) : ℝ := complex.cos x
noncomputable def sin (x : ℝ) : ℝ := complex.sin x

-- inequalities given in the problem statement
axiom inequality1 : cos A ^ 2 + λ * (sin (2 * B) + sin (2 * C)) ≤ 1 + λ ^ 2
axiom inequality2 : sin A ^ 2 + λ * (cos (2 * B) + cos (2 * C)) ≤ 1 + λ ^ 2

-- proving the equality conditions hold
theorem equality_conditions_hold :
  (0 < λ ∧ λ ≤ 1 ∧ B = C ∧ B = (π / 2) - (1 / 2) * arcsin λ) ↔
  (cos A ^ 2 + λ * (sin (2 * B) + sin (2 * C)) = 1 + λ ^ 2 ∧
   sin A ^ 2 + λ * (cos (2 * B) + cos (2 * C)) = 1 + λ ^ 2) :=
begin
  sorry
end

end equality_conditions_hold_l588_588417


namespace expression_divisible_by_11_l588_588617

theorem expression_divisible_by_11 (n : ℤ) : 11 ∣ (3^(2*n + 2) + 2^(6*n + 1)) :=
sorry

end expression_divisible_by_11_l588_588617


namespace vincent_bookstore_l588_588185

theorem vincent_bookstore :
  ∃ (F : ℝ), (∃ (L : ℝ), L = F / 2 ∧ 5 * (5 * F + 8 * L) = 180) → F = 4 :=
begin
  sorry

end vincent_bookstore_l588_588185


namespace inequality_ln_positive_l588_588854

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588854


namespace area_rhombus_is_720_l588_588661

variable (ABCD : Type*) [rhombus ABCD]
variables (R₁ R₂ R : ℝ)

noncomputable def area_rhombus : ℝ := sorry

theorem area_rhombus_is_720
  (R₁ : R₁ = 15) -- Circumradius of triangle ABD
  (R₂ : R₂ = 30) -- Circumradius of triangle ACD
  : area_rhombus ABCD = 720 := 
sorry

end area_rhombus_is_720_l588_588661


namespace heights_inequality_l588_588470

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h₁ : a ≤ b) (h₂ : b ≤ c) :
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) :=
by
  sorry

end heights_inequality_l588_588470


namespace origin_move_distance_l588_588267

-- Definitions based on conditions
def B : ℝ × ℝ := (3, 3)
def B' : ℝ × ℝ := (7, 10)
def O : ℝ × ℝ := (0, 0)
def radius_orig : ℝ := 3
def radius_new : ℝ := 5
def dilation_factor : ℝ := radius_new / radius_orig

-- Statement of the problem
theorem origin_move_distance :
  let k := dilation_factor,
      new_center := B',
      old_center := B,
      translation := (new_center.1 - k * old_center.1, new_center.2 - k * old_center.2) in
  ∥translation∥ = Real.sqrt 29 :=
sorry

end origin_move_distance_l588_588267


namespace segment_length_abs_eq_five_l588_588581

theorem segment_length_abs_eq_five : 
  (length : ℝ) (∀ x : ℝ, abs (x - (27 : ℝ)^(1 : ℝ) / (3 : ℝ)) = 5 → x = 8 ∨ x = -2) 
  → length = 10 := 
begin
  sorry
end

end segment_length_abs_eq_five_l588_588581


namespace shaded_area_of_triangle_CDE_l588_588199

-- Definitions of the points
noncomputable def O := (0, 0 : ℝ×ℝ)
noncomputable def A := (4, 0 : ℝ×ℝ)
noncomputable def B := (16, 0 : ℝ×ℝ)
noncomputable def C := (16, 12 : ℝ×ℝ)
noncomputable def D := (4, 12 : ℝ×ℝ)
noncomputable def E := (4, 3 : ℝ×ℝ)

-- Definition of the area calculation for the given triangle
theorem shaded_area_of_triangle_CDE : 
  let DE := 9 in
  let DC := 12 in
  (DE * DC) / 2 = 54 :=
by
  sorry

end shaded_area_of_triangle_CDE_l588_588199


namespace prime_division_l588_588082

-- Definitions used in conditions
variables {p q : ℕ}

-- We assume p and q are prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- The problem states
theorem prime_division 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hdiv : divides q (3^p - 2^p)) 
  : p ∣ (q - 1) :=
sorry

end prime_division_l588_588082


namespace indeterminate_equation_solution_l588_588503

theorem indeterminate_equation_solution (x y : ℝ) (n : ℕ) :
  (x^2 + (x + 1)^2 = y^2) ↔ 
  (x = 1/4 * ((1 + Real.sqrt 2)^(2*n + 1) + (1 - Real.sqrt 2)^(2*n + 1) - 2) ∧ 
   y = 1/(2 * Real.sqrt 2) * ((1 + Real.sqrt 2)^(2*n + 1) - (1 - Real.sqrt 2)^(2*n + 1))) := 
sorry

end indeterminate_equation_solution_l588_588503


namespace kitchen_length_l588_588985

-- Define the conditions as variables and constants
def width : ℝ := 16 -- width of the kitchen in feet
def height : ℝ := 10 -- height of the kitchen in feet
def paint_rate : ℝ := 40 -- square feet per hour Martha can paint
def total_hours : ℝ := 42 -- total hours taken to paint the kitchen
def total_area_painted : ℝ := total_hours * paint_rate
-- Each wall needs one coat of primer and two coats of paint (total 3 coats)
def actual_wall_area : ℝ := total_area_painted / 3

-- Define the final proof problem
theorem kitchen_length :
  (2 * (length * height) + 2 * (width * height) = actual_wall_area) →
  (length = 12) :=
by
  intro h
  sorry

end kitchen_length_l588_588985


namespace sum_of_inverses_mod_l588_588187

theorem sum_of_inverses_mod (h1 : 5⁻¹ % 17 = 7) (h2 : 5⁻³ % 17 = 3) : 
  (5⁻¹ + 5⁻³) % 17 = 10 := by 
  sorry

end sum_of_inverses_mod_l588_588187


namespace probability_difference_three_l588_588511

theorem probability_difference_three : 
    let s := {1, 2, 3, 4, 5, 6, 7, 8}
    let pair_count := 5
    let total_pairs := Nat.choose 8 2
    let probability := (pair_count : ℚ) / (total_pairs : ℚ)
    probability == 5 / 28 :=
by
    sorry

end probability_difference_three_l588_588511


namespace digit_with_specific_difference_l588_588928

def local_value (numeral : ℕ) (digit : ℕ) (place_value : ℕ) : ℕ :=
digit * place_value

def face_value (digit : ℕ) : ℕ :=
digit

def difference (numeral : ℕ) (digit : ℕ) (place_value : ℕ) : ℕ :=
local_value numeral digit place_value - face_value digit

def find_digit (numeral : ℕ) (target_difference : ℕ) : ℕ :=
  if difference numeral 7 1000 = target_difference then 7 else
  if difference numeral 6 100000 = target_difference then 6 else
  if difference numeral 5 10000 = target_difference then 5 else
  if difference numeral 0 1 = target_difference then 0 else
  if difference numeral 9 100 = target_difference then 9 else
  if difference numeral 3 1 = target_difference then 3 else 0 -- Adjust for other cases

theorem digit_with_specific_difference : find_digit 657903 6993 = 7 :=
by
  simp [find_digit, difference, local_value, face_value]
  simp [difference, local_value, face_value]
  rw [local_value, face_value, sub_eq_add_neg]
  rw [mul_comm]
  simp
  sorry -- proof steps

end digit_with_specific_difference_l588_588928


namespace sheela_deposit_percentage_l588_588512

theorem sheela_deposit_percentage (deposit : ℝ) (income : ℝ) (percentage : ℝ) :
  deposit = 4500 →
  income = 16071.42857142857 →
  percentage = (4500 / 16071.42857142857) * 100 →
  percentage = 28 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end sheela_deposit_percentage_l588_588512


namespace bricks_for_wall_l588_588236

theorem bricks_for_wall
  (wall_length : ℕ) (wall_height : ℕ) (wall_width : ℕ)
  (brick_length : ℕ) (brick_height : ℕ) (brick_width : ℕ)
  (L_eq : wall_length = 600) (H_eq : wall_height = 400) (W_eq : wall_width = 2050)
  (l_eq : brick_length = 30) (h_eq : brick_height = 12) (w_eq : brick_width = 10)
  : (wall_length * wall_height * wall_width) / (brick_length * brick_height * brick_width) = 136667 :=
by
  sorry

end bricks_for_wall_l588_588236


namespace log_of_y_sub_x_plus_one_positive_l588_588904

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588904


namespace inequality_ln_positive_l588_588862

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588862


namespace log_49_48_proof_l588_588366

variable (a b : ℝ)
variable (conditions : (1 / 7) ^ a = (1 / 3) ∧ Real.log 7 4 = b)

noncomputable def log_49_48_in_terms_of_a_b : ℝ :=
  if (1 / 7) ^ a = (1 / 3) ∧ Real.log 7 4 = b then
    (a + 2 * b) / 2
  else
    0

theorem log_49_48_proof : log_49_48_in_terms_of_a_b a b = Real.log 49 48 := by
  sorry

end log_49_48_proof_l588_588366


namespace smallest_value_l588_588407

theorem smallest_value (x : ℝ) (hx : 0 < x ∧ x < 1) :
  x^2 < x ∧
  x^2 < 2 * x ∧
  x^2 < sqrt x ∧
  x^2 < 1 / x :=
by
  sorry

end smallest_value_l588_588407


namespace part1_part2_l588_588491

-- Part 1: Definition of "consecutive roots quadratic equation"
def consecutive_roots (a b : ℤ) : Prop := a = b + 1 ∨ b = a + 1

-- Statement that for some k and constant term, the roots of the quadratic form consecutive roots
theorem part1 (k : ℤ) : consecutive_roots 7 8 → k = -15 → (∀ x : ℤ, x^2 + k * x + 56 = 0 → x = 7 ∨ x = 8) :=
by
  sorry

-- Part 2: Generalizing to the nth equation
theorem part2 (n : ℕ) : 
  (∀ x : ℤ, x^2 - (2 * n - 1) * x + n * (n - 1) = 0 → x = n ∨ x = n - 1) :=
by
  sorry

end part1_part2_l588_588491


namespace ball_travel_distance_l588_588258

noncomputable def total_distance_traveled (initial_height : ℕ) (bounce_fraction : ℚ) (hits : ℕ) : ℚ :=
  let descent := (0 : ℚ).add (initial_height) + (hits.nat_pred).sum (λ n, initial_height * (bounce_fraction ^ n))
  let ascent := (hits).sum (λ n, initial_height * (bounce_fraction ^ n))
  descent + ascent

theorem ball_travel_distance :
  total_distance_traveled 20 (2/3) 4 ≈ 80 :=
by
  sorry

end ball_travel_distance_l588_588258


namespace regular_polygon_sides_l588_588277

theorem regular_polygon_sides (n : ℕ) (h : ∀ i < n, (interior_angle_i : ℝ) = 150) :
  (n = 12) :=
by
  sorry

end regular_polygon_sides_l588_588277


namespace percentage_of_female_officers_on_duty_l588_588497

theorem percentage_of_female_officers_on_duty :
  ∀ (total_on_duty : ℕ) (half_on_duty : ℕ) (total_female_officers : ℕ), 
  total_on_duty = 204 → half_on_duty = total_on_duty / 2 → total_female_officers = 600 → 
  ((half_on_duty: ℚ) / total_female_officers) * 100 = 17 :=
by
  intro total_on_duty half_on_duty total_female_officers
  intros h1 h2 h3
  sorry

end percentage_of_female_officers_on_duty_l588_588497


namespace hyperbola_eccentricity_is_sqrt_5_l588_588350

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : ℝ :=
let c := a * Real.sqrt 5 in
  c / a

theorem hyperbola_eccentricity_is_sqrt_5 (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_PF1_PF2 : ∀ P F1 F2 : ℝ×ℝ, 2 * dist P F1 + dist P F2 = Real.sqrt 5 * dist F1 F2)
  (h_angle : ∀ P F1 F2 : ℝ×ℝ, angle P F1 F2 = π / 2) :
  hyperbola_eccentricity a b h_a_pos h_b_pos = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_is_sqrt_5_l588_588350


namespace problem_l588_588818

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588818


namespace ellipse_foci_coordinates_l588_588143

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
  (x^2 / 9 + y^2 / 5 = 1) →
  (x, y) = (2, 0) ∨ (x, y) = (-2, 0) :=
by
  sorry

end ellipse_foci_coordinates_l588_588143


namespace square_of_other_leg_l588_588539

-- Conditions
variable (a b c : ℝ)
variable (h₁ : c = a + 2)
variable (h₂ : a^2 + b^2 = c^2)

-- The theorem statement
theorem square_of_other_leg (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l588_588539


namespace exists_point_M_l588_588495

noncomputable theory
open_locale classical

structure Point :=
(x : ℝ) (y : ℝ)

def on_line (A B C : Point) : Prop :=
(A.y - B.y) * (A.y - C.y) = 0

def equal_view_angle (A B C D M: Point) : Prop :=
(∠ A M B = ∠ B M C) ∧ (∠ B M C = ∠ C M D)

theorem exists_point_M 
  (A B C D : Point)
  (h1 : on_line A B C)
  (h2 : on_line A C D)
  (h3 : A ≠ B ∧ B ≠ C ∧ C ≠ D) :
  ∃ M : Point, equal_view_angle A B C D M :=
sorry

end exists_point_M_l588_588495


namespace range_of_m_l588_588781

theorem range_of_m (m : ℝ) :
  (∃ x ∈ Icc (0 : ℝ) (π / 2), ((sin x + cos x) ^ 2 - 2 * cos x ^ 2 - m = 0)) → 
  m ∈ Icc (-1: ℝ) (real.sqrt 2) :=
by
  sorry

end range_of_m_l588_588781


namespace binomial_dist_p_value_l588_588769

theorem binomial_dist_p_value (n p : ℝ) (q : ℝ := 1 - p) 
  (h_X_binomial : ∃ (X : ℕ → ℝ), X follows binomial n p)
  (h_expectation : n * p = 30)
  (h_variance : n * p * q = 20) :
  p = 1 / 3 := 
by
  sorry

end binomial_dist_p_value_l588_588769


namespace triangle_ABC_angles_l588_588059

-- Given definitions and conditions
variables {A B C D I : Type}
variables [has_center_circle ABC]
variables [has_center_incircle BCD]
variables [is_bisector BD]
variables [circumcircle_center_equiv_incircle_center ABC BCD I]

-- The conjecture to prove
theorem triangle_ABC_angles
  (h1 : is_bisector BD)
  (h2 : circumcircle_center_equiv_incircle_center ABC BCD I) :
  angle A = angle B = 72 ∧ angle C = 36 :=
sorry

end triangle_ABC_angles_l588_588059


namespace prime_square_plus_two_is_prime_iff_l588_588723

theorem prime_square_plus_two_is_prime_iff (p : ℕ) (hp : Prime p) : Prime (p^2 + 2) ↔ p = 3 :=
sorry

end prime_square_plus_two_is_prime_iff_l588_588723


namespace radius_of_larger_ball_l588_588561

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

theorem radius_of_larger_ball (r_small : ℝ) (V_large : ℝ) :
    r_small = 2 →
    V_large = 4 * volume_of_sphere r_small →
    ∃ R : ℝ, volume_of_sphere R = V_large ∧
             R = 2 * Real.cbrt 4 :=
by
  intros h1 h2
  sorry

end radius_of_larger_ball_l588_588561


namespace triangle_angle_properties_l588_588044

theorem triangle_angle_properties
  (a b : ℕ)
  (h₁ : a = 45)
  (h₂ : b = 70) :
  ∃ (c : ℕ), a + b + c = 180 ∧ c = 65 ∧ max (max a b) c = 70 := by
  sorry

end triangle_angle_properties_l588_588044


namespace lines_parallel_a_eq_sqrt2_l588_588410

theorem lines_parallel_a_eq_sqrt2 (a : ℝ) (h1 : 1 ≠ 0) :
  (∀ a ≠ 0, ((- (1 / (2 * a))) = (- a / 2)) → a = Real.sqrt 2) :=
by
  sorry

end lines_parallel_a_eq_sqrt2_l588_588410


namespace units_digit_sum_l588_588715

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum
  (h1 : units_digit 13 = 3)
  (h2 : units_digit 41 = 1)
  (h3 : units_digit 27 = 7)
  (h4 : units_digit 34 = 4) :
  units_digit ((13 * 41) + (27 * 34)) = 1 :=
by
  sorry

end units_digit_sum_l588_588715


namespace log_49_48_in_terms_of_a_and_b_l588_588369

-- Define the constants and hypotheses
variable (a b : ℝ)
variable (h1 : a = Real.logb 7 3)
variable (h2 : b = Real.logb 7 4)

-- Define the statement to be proved
theorem log_49_48_in_terms_of_a_and_b (a b : ℝ) (h1 : a = Real.logb 7 3) (h2 : b = Real.logb 7 4) :
  Real.logb 49 48 = (a + 2 * b) / 2 :=
by
  sorry

end log_49_48_in_terms_of_a_and_b_l588_588369


namespace omega_value_sin_2theta_l588_588382

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x) + sqrt 3 * cos (ω * x)

theorem omega_value (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x : ℝ, f ω x = f ω (x + π)) : ω = 2 :=
by
  sorry

theorem sin_2theta (θ : ℝ) (hθ : θ ∈ set.Ioo 0 (π / 2)) 
  (h3 : f 2 ((θ/2) + (π / 12)) = 6/5) : sin (2 * θ) = 24 / 25 :=
by
  sorry

end omega_value_sin_2theta_l588_588382


namespace sqrt_product_eq_225_l588_588122

theorem sqrt_product_eq_225 : (Real.sqrt (5 * 3) * Real.sqrt (3 ^ 3 * 5 ^ 3) = 225) :=
by
  sorry

end sqrt_product_eq_225_l588_588122


namespace arithmetic_sequence_a10_l588_588770

theorem arithmetic_sequence_a10 :
  (∀ n : ℕ, n > 0 → ∃ d : ℝ, ∃ a : ℕ → ℝ, a 1 = 2 ∧ (∀ k : ℕ, k > 0 → a (k + 1) = a k + d) ∧
    (∀ m : ℕ, m > 0 → (a m)^2 / m) = (a 1)^2 / 1 + m * d) →
  a 10 = 20 := 
sorry

end arithmetic_sequence_a10_l588_588770


namespace granger_bought_12_cans_of_spam_l588_588394

theorem granger_bought_12_cans_of_spam : 
  ∀ (S : ℕ), 
    (3 * 5 + 4 * 2 + 3 * S = 59) → 
    (S = 12) := 
by
  intro S h
  sorry

end granger_bought_12_cans_of_spam_l588_588394


namespace problem_1_problem_2_l588_588381

theorem problem_1 (x : ℝ) : 
  f (x) = log (|x - 1| + |x - 4| + 2) → 
  (f(x) ≥ 3 ↔ (x ≤ -1/2 ∨ x ≥ 11/2)) := 
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, ∃ fx : ℝ, fx = log (|x - 1| + |x - 4| - a)) → 
  (a < 3) := 
sorry

end problem_1_problem_2_l588_588381


namespace system_of_equations_solution_l588_588132

theorem system_of_equations_solution (x y : ℤ) 
  (h1 : x^2 + x * y + y^2 = 37) 
  (h2 : x^4 + x^2 * y^2 + y^4 = 481) : 
  (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) ∨ (x = -3 ∧ y = -4) ∨ (x = -4 ∧ y = -3) := 
by sorry

end system_of_equations_solution_l588_588132


namespace reachable_iff_reachable_l588_588455

def reachable (A B : String) : Prop :=
  ∃ X : List (Char ⊕ Unit), A.toList = insert_arrows(A, X) ∧ B.toList = apply_arrows(A, X)

theorem reachable_iff_reachable {A B : String} : reachable A B ↔ reachable B A := sorry

end reachable_iff_reachable_l588_588455


namespace log_of_y_sub_x_plus_one_positive_l588_588905

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588905


namespace total_decorations_handed_out_l588_588729

theorem total_decorations_handed_out 
  (tinsel_per_box : ℕ) 
  (christmas_trees_per_box : ℕ) 
  (snow_globes_per_box : ℕ) 
  (families_received_boxes : ℕ) 
  (community_center_received_boxes : ℕ) 
  (total_decorations : ℕ) 
  (box_decorations : tinsel_per_box + christmas_trees_per_box + snow_globes_per_box)
  (family_decorations : families_received_boxes * box_decorations)
  (community_center_decorations : community_center_received_boxes * box_decorations) :
  tinsel_per_box = 4 →
  christmas_trees_per_box = 1 →
  snow_globes_per_box = 5 →
  families_received_boxes = 11 →
  community_center_received_boxes = 1 →
  total_decorations = family_decorations + community_center_decorations →
  total_decorations = 120 :=
by
  intros
  sorry

end total_decorations_handed_out_l588_588729


namespace john_exactly_three_green_marbles_l588_588069

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 7
def green_prob : ℚ := 8 / 15
def purple_prob : ℚ := 7 / 15
def binom_coeff : ℕ := Nat.choose 7 3 

-- Theorem Statement
theorem john_exactly_three_green_marbles :
  (binom_coeff : ℚ) * (green_prob^3 * purple_prob^4) = 8604112 / 15946875 :=
by
  sorry

end john_exactly_three_green_marbles_l588_588069


namespace ln_gt_sufficient_not_necessary_for_cube_l588_588605

theorem ln_gt_sufficient_not_necessary_for_cube (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (ln x > ln y) → (x^3 > y^3) ∧ ¬(x^3 > y^3 → ln x > ln y) :=
by
  sorry

end ln_gt_sufficient_not_necessary_for_cube_l588_588605


namespace trig_identity_l588_588377

theorem trig_identity 
  (α : ℝ) 
  (h_cos : cos α = 4 / 5) 
  (h_sin : sin α = 3 / 5) :
  (sin (π + α) + 2 * sin (π / 2 - α)) / (2 * cos (π - α)) = -5 / 8 := 
by sorry

end trig_identity_l588_588377


namespace definite_integral_evaluation_l588_588327

noncomputable def integral_geometric_circle : Real :=
  ∫ x in 0..3, sqrt (9 - x^2)

theorem definite_integral_evaluation :
  integral_geometric_circle = (9 / 4) * Real.pi := by
  sorry

end definite_integral_evaluation_l588_588327


namespace product_of_integers_l588_588564

-- Define the conditions as variables in Lean
variables {x y : ℤ}

-- State the main theorem/proof
theorem product_of_integers (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end product_of_integers_l588_588564


namespace plane_divides_pyramid_l588_588245

noncomputable def volume_of_parts (a h KL KK1: ℝ): ℝ × ℝ :=
  -- Define the pyramid and prism structure and the conditions
  let volume_total := (1/3) * (a^2) * h
  let volume_part1 := 512/15
  let volume_part2 := volume_total - volume_part1
  (⟨volume_part1, volume_part2⟩ : ℝ × ℝ)

theorem plane_divides_pyramid (a h KL KK1: ℝ) 
  (h₁ : a = 8 * Real.sqrt 2) 
  (h₂ : h = 4) 
  (h₃ : KL = 2) 
  (h₄ : KK1 = 1):
  volume_of_parts a h KL KK1 = (512/15, 2048/15) := 
by 
  sorry

end plane_divides_pyramid_l588_588245


namespace new_person_weight_l588_588241

theorem new_person_weight (initial_weight : ℕ) (weight_increase_per_person : ℕ) (number_of_people : ℕ) :
  initial_weight = 55 → weight_increase_per_person = 4 → number_of_people = 8 → 
  initial_weight + weight_increase_per_person * number_of_people = 87 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end new_person_weight_l588_588241


namespace shaded_area_is_54_l588_588200

-- Define the coordinates of points O, A, B, C, D, E
structure Point where
  x : ℝ
  y : ℝ

-- Given points
def O := Point.mk 0 0
def A := Point.mk 4 0
def B := Point.mk 16 0
def C := Point.mk 16 12
def D := Point.mk 4 12
def E := Point.mk 4 3

-- Define the function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2) ^ (1/2)

-- Define similarity of triangles and calculate side lengths involved
def triangles_similarity (OA OB CB EA : ℝ) : Prop :=
  OA / OB = EA / CB

-- Define the condition
def condition : Prop := 
  triangles_similarity (distance O A) (distance O B) 12 (distance E A) ∧
  distance E A = 3 ∧
  distance D E = 9

-- Define the calculation of area of triangle given base and height
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- State that the area of triangle CDE is 54 cm²
def area_shaded_region : Prop :=
  triangle_area 9 12 = 54

-- Main theorem statement
theorem shaded_area_is_54 : condition → area_shaded_region := by
  sorry

end shaded_area_is_54_l588_588200


namespace num_valid_sequences_l588_588961

def is_valid_sequence (b : List ℕ) : Prop :=
  b.length = 5 ∧
  (∀ i, 1 ≤ i ∧ i ≤ 4 → (b.get? i ≠ none ∧ b.get? (i+1) ≠ none ∧
  (b.nthLe i 0) + 2 ∈ b.take i ∧
  (b.nthLe i 0) - 2 ∈ b.take i))

noncomputable def valid_sequences : List (List ℕ) :=
  List.permutations [1, 3, 5, 7, 9] |>.filter is_valid_sequence

theorem num_valid_sequences : valid_sequences.length = 1 := sorry

end num_valid_sequences_l588_588961


namespace set_A_cardinality_l588_588231

statement1 : {x : ℝ | ∃ a b : ℝ, x = |a| / a + |b| / b} = {-2, 0, 2} := sorry

statement2 : ∀ a b c : ℝ, a + b + c = 0 ↔ (∀ x : ℝ, x = 1 → a * x^2 + b * x + c = 0) := sorry

statement3 : ∀ a b : ℝ, 1 < a → 1 < b → a ≠ b →
  (max (a^2 + b^2) (max (2 * a * b) (max (a + b) (2 * real.sqrt(a * b)))) = (a^2 + b^2)) := sorry

theorem set_A_cardinality : 
  let A := {a : ℤ | ∃ n : ℕ, n = 6 / (3 - a)}
  A.card = 4 := sorry

end set_A_cardinality_l588_588231


namespace log_y_minus_x_plus_1_pos_l588_588873

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588873


namespace remainder_problem_l588_588476

theorem remainder_problem (f y z : ℤ) (k m n : ℤ) 
  (h1 : f % 5 = 3) 
  (h2 : y % 5 = 4)
  (h3 : z % 7 = 6)
  (h4 : (f + y) % 15 = 7)
  : (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 :=
by
  sorry

end remainder_problem_l588_588476


namespace red_fraction_is_three_fifths_l588_588035

noncomputable def fraction_of_red_marbles (x : ℕ) : ℚ := 
  let blue_marbles := (2 / 3 : ℚ) * x
  let red_marbles := x - blue_marbles
  let new_red_marbles := 3 * red_marbles
  let new_total_marbles := blue_marbles + new_red_marbles
  new_red_marbles / new_total_marbles

theorem red_fraction_is_three_fifths (x : ℕ) (hx : x ≠ 0) : fraction_of_red_marbles x = 3 / 5 :=
by {
  sorry
}

end red_fraction_is_three_fifths_l588_588035


namespace inequality_ln_positive_l588_588852

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588852


namespace log_gt_suff_but_not_necc_for_cube_gt_l588_588608

theorem log_gt_suff_but_not_necc_for_cube_gt (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (ln x > ln y → x^3 > y^3) ∧ (¬(x^3 > y^3 → ln x > ln y)) :=
by
  sorry

end log_gt_suff_but_not_necc_for_cube_gt_l588_588608


namespace log_pos_given_ineq_l588_588887

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588887


namespace john_exactly_three_green_marbles_l588_588068

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 7
def green_prob : ℚ := 8 / 15
def purple_prob : ℚ := 7 / 15
def binom_coeff : ℕ := Nat.choose 7 3 

-- Theorem Statement
theorem john_exactly_three_green_marbles :
  (binom_coeff : ℚ) * (green_prob^3 * purple_prob^4) = 8604112 / 15946875 :=
by
  sorry

end john_exactly_three_green_marbles_l588_588068


namespace length_of_AB_l588_588935

theorem length_of_AB 
  {α β : ℝ} (A B C : Type*)
  [decidable_eq A] [decidable_eq B] [decidable_eq C]
  (AC BC AB : ℝ)
  (h_right: ∠C = 90 °)
  (h_tan: tan α = 3 / 2)
  (h_AC : AC = 12) 
  : AB = 2 * real.sqrt 117 := 
  sorry

end length_of_AB_l588_588935


namespace prove_fraction_identity_l588_588521

theorem prove_fraction_identity 
  (x y z : ℝ)
  (h1 : (x * z) / (x + y) + (y * z) / (y + z) + (x * y) / (z + x) = -18)
  (h2 : (z * y) / (x + y) + (z * x) / (y + z) + (y * x) / (z + x) = 20) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 20.5 := 
by
  sorry

end prove_fraction_identity_l588_588521


namespace log_y_minus_x_plus_1_gt_0_l588_588842

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588842


namespace valid_pictures_l588_588997

def valid_grid (grid : List (List (Option Nat))) : Prop :=
  ∀ (r c : Nat) (val : Nat),
    grid[r][c] = some val →
    val > 0 → val < 16 →
    (exists (r' c' : Nat), 
      ((abs (r' - r) = 1 ∧ c' = c) ∨ (r' = r ∧ abs (c' - c) = 1)) ∧ 
       grid[r'][c'] = some (val + 1))
    ∧ (exists (r' c' : Nat), 
      ((abs (r' - r) = 1 ∧ c' = c) ∨ (r' = r ∧ abs (c' - c) = 1)) ∧ 
       grid[r'][c'] = some (val - 1))

def picture1 := 
  [[none, none, some 11, none],
   [none, none, none, none],
   [some 5, none, none, none],
   [none, none, none, some 3]]

def picture2 := 
  [[none, some 2, none, none],
   [none, none, some 9, none],
   [none, some 7, none, none],
   [some 5, none, none, none]]

theorem valid_pictures : 
  valid_grid picture1 ∧ valid_grid picture2 :=
by
  sorry

end valid_pictures_l588_588997


namespace volume_of_solid_l588_588562

theorem volume_of_solid (a : ℝ) (b : ℝ) (upper_edge_length : ℝ)
  (edge_length : ℝ) (a_val : a = 3 * real.sqrt 2) (cond_b : b = 2 * a)
  (upper_edge_cond : upper_edge_length = 3 * a) (all_edge_cond : edge_length = a) :
  let volume := 2 * a^3 in
  volume = 108 * real.sqrt 2 := 
by
  sorry

end volume_of_solid_l588_588562


namespace functionalEquationSolution_l588_588089

noncomputable def f : ℝ → ℝ := sorry
axiom cond1 : ∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f(x * f(y)) * f(y) = f(x + y)
axiom cond2 : f 2 = 0
axiom cond3 : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0

theorem functionalEquationSolution :
    ∀ x : ℝ, 
    (x ≥ 2 → f x = 0) ∧ 
    (0 ≤ x ∧ x < 2 → f x = 2 / (2 - x)) := 
begin
    sorry
end

end functionalEquationSolution_l588_588089


namespace ellipse_and_parabola_proof_l588_588373

-- Define the ellipse and parabola constraints
theorem ellipse_and_parabola_proof :
  (center_origin : (0, 0)) →
  (eccentricity : 1 / 2) →
  (parabola_focus : (x^2 = 4 * sqrt 3 * y)) →
  ∃ (a b : ℝ), (a^2 = 4) ∧ (b^2 = 3) ∧
  (ellipse_equation : (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  (maximum_inscribed_circle_area : π * (3 / 4)^2 = 9 / 16 * π) ∧
  (line_equation : x = 1) :=
begin
  sorry
end

end ellipse_and_parabola_proof_l588_588373


namespace range_a_l588_588980

noncomputable def range_of_a : set ℝ :=
  {a | - (3 / 4) ≤ a ∧ a ≤ (2 / 3)}

theorem range_a :
  ∀ a : ℝ,
    (∀ x : ℝ, ∃ y : ℝ,
      let f' := - real.exp x - 1,
          g' := 3 * a - 2 * real.sin y in
        f' * g' = -1) ↔ a ∈ range_of_a :=
by
  sorry

end range_a_l588_588980


namespace Mila_hourly_rate_l588_588986

noncomputable def Agnes_hourly_rate : ℝ := 15
noncomputable def Agnes_weekly_hours : ℝ := 8
noncomputable def weeks_per_month : ℝ := 4
noncomputable def Mila_total_hours : ℝ := 48

-- Lean statement to prove the equivalent of the formalized problem
theorem Mila_hourly_rate :
  let Agnes_monthly_earnings := Agnes_hourly_rate * Agnes_weekly_hours * weeks_per_month in
  let Mila_monthly_earnings := Agnes_monthly_earnings in
  Mila_monthly_earnings / Mila_total_hours = 10 := by
  sorry

end Mila_hourly_rate_l588_588986


namespace polynomial_decomposition_l588_588302

theorem polynomial_decomposition :
  ∃ (ns : List ℕ), 
    (∀ n ∈ ns, 
      ∃ (a : ℕ → ℕ), 
        (∀ i, 1 ≤ i → i ≤ n → a i ≤ n ∧ Function.Injective a) ∧ 
        1976 * (List.sum (List.range (n+1)).map (λ x => x)) = (n * (n + 1)) / 2)
  ∧ ns = [0, 1, 3, 7, 15, 12, 18, 25, 37, 51, 75, 151, 246, 493, 987, 1975] :=
by
  sorry

end polynomial_decomposition_l588_588302


namespace inequality_ln_pos_l588_588831

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588831


namespace money_left_in_wallet_l588_588169

def initial_amount := 106
def spent_supermarket := 31
def spent_showroom := 49

theorem money_left_in_wallet : initial_amount - spent_supermarket - spent_showroom = 26 := by
  sorry

end money_left_in_wallet_l588_588169


namespace problem_statement_l588_588467

variable {A : Set ℕ} (hA : A.nonempty) 
variable {b : ℕ → ℕ} {c : ℕ → ℕ} 
variable {n : ℕ} (h1 : ∀ i < n, ∀ a ∈ A, b i * a + c i ∈ A)
variable (h2 : ∀ i j < n, i ≠ j → Disjoint (b i * A + c i) (b j * A + c j))

theorem problem_statement : (∑ i in Finset.range n, 1 / (b i : ℝ)) ≤ 1 := by
  sorry

end problem_statement_l588_588467


namespace inequality_ln_pos_l588_588837

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588837


namespace bike_cost_l588_588444

-- Defining the problem conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def leftover : ℚ := 20  -- 20 dollars left over
def quarter_value : ℚ := 0.25

-- Define the total quarters Jenn has
def total_quarters := jars * quarters_per_jar

-- Define the total amount of money from quarters
def total_money_quarters := total_quarters * quarter_value

-- Prove that the cost of the bike is $200
theorem bike_cost : total_money_quarters + leftover - 20 = 200 :=
sorry

end bike_cost_l588_588444


namespace third_stick_shorter_l588_588443

def length_first : ℕ := 3
def second_is_twice : ℕ := 2 * length_first
def total_length (T : ℕ) : Prop := length_first + second_is_twice + T = 14

theorem third_stick_shorter :
  ∃ T, total_length T → second_is_twice - T = 1 :=
by
  intros T hT
  have h1 := congr_arg (λ x, x - length_first - second_is_twice) hT
  sorry  -- Proof to be filled in

end third_stick_shorter_l588_588443


namespace quadratic_even_coefficient_l588_588999

theorem quadratic_even_coefficient (a b c : ℤ) (h_a_nonzero : a ≠ 0)
  (h_rational_roots : ∃ r s : ℚ, a * r * r + b * r + c = 0 ∧ a * s * s + b * s + c = 0) :
  (¬ (¬ even a ∧ ¬ even b ∧ ¬ even c)) :=
by
  sorry

end quadratic_even_coefficient_l588_588999


namespace function_domain_eq_l588_588145

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / x

theorem function_domain_eq : 
  {x : ℝ | x > -1 ∧ x ≠ 0} = (-1, 0) ∪ (0, ∞) :=
by
  sorry

end function_domain_eq_l588_588145


namespace not_possible_select_seven_distinct_weights_no_equal_subsets_l588_588349

theorem not_possible_select_seven_distinct_weights_no_equal_subsets :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 27 → s.card = 7 → ∃ (a b : Finset ℕ), a ≠ b ∧ a ⊆ s ∧ b ⊆ s ∧ a.sum id = b.sum id :=
by
  intro s hs hcard
  sorry

end not_possible_select_seven_distinct_weights_no_equal_subsets_l588_588349


namespace determine_g6_l588_588536

variable (g : ℕ → ℝ)

def functional_eq (g : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, m ≥ n → g(m + n) + 2 * g(m - n) = 2 * g(m) + g(n)

theorem determine_g6
  (h1 : g 1 = 1)
  (h2 : functional_eq g)
  (h3 : g 0 = 0) :
  g 6 = 0 :=
sorry

end determine_g6_l588_588536


namespace desks_built_by_carpenters_l588_588242

theorem desks_built_by_carpenters (h : 2 * 2.5 * r ≥ 2 * r) : 4 * 5 * r ≥ 8 * r :=
by
  sorry

end desks_built_by_carpenters_l588_588242


namespace stoker_bedtime_l588_588186

def time := ℕ -- Represent time in 24-hour format for simplicity

-- Conditions
def morning_wake_up_time : time := 8 * 60 + 30 -- 8:30 AM in minutes
def morning_coal_usage : ℕ := 5 -- kg
def evening_coal_usage : ℕ := 7 -- kg

-- Correct Answer
def bedtime : time := 22 * 60 + 30 -- 10:30 PM in minutes

-- Proof
theorem stoker_bedtime : 
  (forall usage : time, 
     usage = morning_wake_up_time + morning_coal_usage ->
     usage = evening_coal_usage) ->
  bedtime = 22 * 60 + 30 :=
by
  sorry

end stoker_bedtime_l588_588186


namespace complex_div_modulus_eq_l588_588478

theorem complex_div_modulus_eq (z : ℂ) (h : z = 1 + 2 * complex.I) : 
  (z^2 / complex.abs(z^2)) = - (3/5 : ℂ) + (4/5) * complex.I :=
by {
  rw h,
  sorry
}

end complex_div_modulus_eq_l588_588478


namespace mrs_heine_dogs_l588_588987

-- Define the number of biscuits per dog
def biscuits_per_dog : ℕ := 3

-- Define the total number of biscuits
def total_biscuits : ℕ := 6

-- Define the number of dogs
def number_of_dogs : ℕ := 2

-- Define the proof statement
theorem mrs_heine_dogs : total_biscuits / biscuits_per_dog = number_of_dogs :=
by
  sorry

end mrs_heine_dogs_l588_588987


namespace largest_number_is_b_l588_588609

noncomputable def a := 0.935
noncomputable def b := 0.9401
noncomputable def c := 0.9349
noncomputable def d := 0.9041
noncomputable def e := 0.9400

theorem largest_number_is_b : b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  -- proof can be filled in here
  sorry

end largest_number_is_b_l588_588609


namespace circumradius_eq_exradius_opposite_BC_l588_588733

-- Definitions of points and triangles
variable {A B C : Point}
variable (O I D : Point)
variable {α β γ : Angle}

-- Definitions of circumcenter, incenter, altitude, and collinearity
def is_circumcenter (O : Point) (A B C : Point) : Prop := sorry
def is_incenter (I : Point) (A B C : Point) : Prop := sorry
def is_altitude (A D B C : Point) : Prop := sorry
def collinear (O D I : Point) : Prop := sorry

-- Definitions of circumradius and exradius
def circumradius (A B C : Point) : ℝ := sorry
def exradius_opposite_BC (A B C : Point) : ℝ := sorry

-- Main theorem statement
theorem circumradius_eq_exradius_opposite_BC
  (h_circ : is_circumcenter O A B C)
  (h_incenter : is_incenter I A B C)
  (h_altitude : is_altitude A D B C)
  (h_collinear : collinear O D I) : 
  circumradius A B C = exradius_opposite_BC A B C :=
sorry

end circumradius_eq_exradius_opposite_BC_l588_588733


namespace max_intersections_circle_quadrilateral_l588_588600

def max_points_of_intersection (circle : Type) (quadrilateral : Type) (intersects : circle → set quadrilateral → ℕ) : ℕ :=
  (λ (c : circle) (q : set quadrilateral), intersects c q) circle quadrilateral

axiom circle_intersects_line_segment_at_most_two_points (line_segment : Type) (circle : Type) (intersects : circle → line_segment → ℕ) :
  ∀ (l : line_segment) (c : circle), intersects c l ≤ 2

axiom quadrilateral_has_four_line_segments (quadrilateral : Type) (line_segment : Type) (segments : set line_segment) :
  ∃ (l1 l2 l3 l4 : line_segment), segments = {l1, l2, l3, l4}

theorem max_intersections_circle_quadrilateral {circle quadrilateral line_segment : Type}
  (intersects : circle → set line_segment → ℕ)
  (segments : set line_segment) :
  ∃ (c : circle) (q : quadrilateral), max_points_of_intersection c q intersects = 8 :=
sorry

end max_intersections_circle_quadrilateral_l588_588600


namespace inequality_ln_positive_l588_588853

theorem inequality_ln_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_positive_l588_588853


namespace lim_n_to_infinity_fraction_l588_588665

noncomputable def limit_expression : ℝ := 
  real.limit (λ n: ℕ, (n: ℝ + 1) / (3 * n - 1)) sorry

theorem lim_n_to_infinity_fraction :
  limit_expression = 1 / 3 :=
sorry

end lim_n_to_infinity_fraction_l588_588665


namespace probability_of_multiple_of_3_is_3_over_10_l588_588175

def tickets := set.range (λ n, n + 1)

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def count_multiples_of_3 : ℕ :=
  (set.filter is_multiple_of_3 tickets).card

def total_tickets : ℕ := 20

def probability_multiple_of_3 : ℝ :=
  (count_multiples_of_3 : ℝ) / (total_tickets : ℝ)

theorem probability_of_multiple_of_3_is_3_over_10 :
  probability_multiple_of_3 = 3 / 10 := by
  sorry

end probability_of_multiple_of_3_is_3_over_10_l588_588175


namespace find_n_mod_l588_588696

theorem find_n_mod (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 5) : n ≡ -1723 [MOD 6] ↔ n = 5 :=
by
sorry

end find_n_mod_l588_588696


namespace find_xz_over_y_squared_l588_588719

variable {x y z : ℝ}

noncomputable def k : ℝ := 7

theorem find_xz_over_y_squared
    (h1 : x + k * y + 4 * z = 0)
    (h2 : 4 * x + k * y - 3 * z = 0)
    (h3 : x + 3 * y - 2 * z = 0)
    (h_nz : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
    (x * z) / (y ^ 2) = 26 / 9 :=
by sorry

end find_xz_over_y_squared_l588_588719


namespace triangle_perimeter_is_correct_l588_588418

noncomputable def perimeter_of_isosceles_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  a + b + c

theorem triangle_perimeter_is_correct :
  let A B C : ℝ := 15
  let hypotenuse : ℝ := A
  let leg : ℝ := 15 / Real.sqrt 2
  let perimeter : ℝ := perimeter_of_isosceles_right_triangle leg leg hypotenuse 
  perimeter = 15 + 15 * Real.sqrt 2 :=
  by
  sorry

end triangle_perimeter_is_correct_l588_588418


namespace max_a_2017_2018_ge_2017_l588_588960

def seq_a (a : ℕ → ℤ) (b : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ (∀ n, n ≥ 1 → 
  (b (n-1) = 1 → a (n+1) = a n * b n + a (n-1)) ∧ 
  (b (n-1) > 1 → a (n+1) = a n * b n - a (n-1)))

theorem max_a_2017_2018_ge_2017 (a : ℕ → ℤ) (b : ℕ → ℕ) (h : seq_a a b) :
  max (a 2017) (a 2018) ≥ 2017 :=
sorry

end max_a_2017_2018_ge_2017_l588_588960


namespace clock_broken_not_midnight_l588_588995

-- Defining the conditions and main proof statement
theorem clock_broken_not_midnight 
  (h_fast : ∀ t, hour_angle(t + 1) = hour_angle(t) + 60)
  (m_slow : ∀ t, minute_angle(t + 1/2) = minute_angle(t) + 6)
  (wakeup_time : ∃ t, hour_angle(t) = 180 ∧ minute_angle(t) = 0) :
  ¬ (∃ t0, t0 = 0 ∧ ∀ t, (t ≥ t0 → hour_angle(t) = 180 ∧ minute_angle(t) = 0)) := 
sorry

end clock_broken_not_midnight_l588_588995


namespace planted_area_ratio_l588_588634

noncomputable def ratio_of_planted_area_to_total_area : ℚ := 145 / 147

theorem planted_area_ratio (h : ∃ (S : ℚ), 
  (∃ (x y : ℚ), x * x + y * y ≤ S * S) ∧
  (∃ (a b : ℚ), 3 * a + 4 * b = 12 ∧ (3 * x + 4 * y - 12) / 5 = 2)) :
  ratio_of_planted_area_to_total_area = 145 / 147 :=
sorry

end planted_area_ratio_l588_588634


namespace find_ordered_pair_l588_588338

theorem find_ordered_pair :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 18 * m * n = 73 - 9 * m - 3 * n ∧ m = 4 ∧ n = 18 :=
by
  existsi 4
  existsi 18
  split
  . exact Nat.one_pos -- Proving 4 > 0
  split
  . exact Nat.succ_pos' 17 -- Proving 18 > 0
  split
  . calc
    18 * 4 * 18 = 18 * 72 : by ring
    ... = 73 - 36 - 54 : by norm_num
    ... = 73 - 90 : by norm_num
    ... = -17 : by norm_num
    ... = 73 - 9 * 4 - 3 * 18 : by ring
  split
  . refl
  . refl

end find_ordered_pair_l588_588338


namespace josie_shopping_time_l588_588105

theorem josie_shopping_time : 
  let wait_cart := 3
  let wait_employee := 13
  let wait_stocker := 14
  let wait_checkout := 18
  let total_trip_time := 90
  total_trip_time - (wait_cart + wait_employee + wait_stocker + wait_checkout) = 42 :=
by
  -- Conditions (definitions of waiting times and total trip time)
  let wait_cart := 3
  let wait_employee := 13
  let wait_stocker := 14
  let wait_checkout := 18
  let total_trip_time := 90

  -- Proof statement using the conditions
  calc
    total_trip_time - (wait_cart + wait_employee + wait_stocker + wait_checkout)
    = 90 - (3 + 13 + 14 + 18) : by rw [total_trip_time]
    ... = 90 - 48 : by norm_num
    ... = 42 : by norm_num

end josie_shopping_time_l588_588105


namespace vincent_total_cost_proof_l588_588493

-- Define all conditions
def monday_packs := 15
def monday_price_per_pack := 2.5
def monday_discount := 0.10

def tuesday_packs := 25
def tuesday_price_per_pack := 3.0
def tuesday_sales_tax := 0.05

def wednesday_packs := 30
def wednesday_price_per_pack := 3.5
def wednesday_discount := 0.15
def wednesday_sales_tax := 0.08

-- Define intermediate calculations
def monday_total := monday_packs * monday_price_per_pack
def monday_total_after_discount := monday_total - (monday_total * monday_discount)

def tuesday_total := tuesday_packs * tuesday_price_per_pack
def tuesday_total_after_tax := tuesday_total + (tuesday_total * tuesday_sales_tax)

def wednesday_total := wednesday_packs * wednesday_price_per_pack
def wednesday_total_after_discount := wednesday_total - (wednesday_total * wednesday_discount)
def wednesday_total_after_tax := wednesday_total_after_discount + (wednesday_total_after_discount * wednesday_sales_tax)

-- Define the statement to prove
theorem vincent_total_cost_proof : monday_total_after_discount + tuesday_total_after_tax + wednesday_total_after_tax = 208.89 := by
  sorry

end vincent_total_cost_proof_l588_588493


namespace difference_in_roi_l588_588684

theorem difference_in_roi (E_investment : ℝ) (B_investment : ℝ) (E_rate : ℝ) (B_rate : ℝ) (years : ℕ) :
  E_investment = 300 → B_investment = 500 → E_rate = 0.15 → B_rate = 0.10 → years = 2 →
  (B_rate * B_investment * years) - (E_rate * E_investment * years) = 10 :=
by
  intros E_investment_eq B_investment_eq E_rate_eq B_rate_eq years_eq
  sorry

end difference_in_roi_l588_588684


namespace solve_for_x_l588_588312

theorem solve_for_x (x : ℝ) (h : (x * real.sqrt (x^4))^(1/4) = 4) : x = 2^(8/3) :=
sorry

end solve_for_x_l588_588312


namespace radical_axis_perpendicular_l588_588249

noncomputable def radical_axis {α : Type*} [MetricSpace α] (O1 O2 : α) (R1 R2 : ℝ) (P : α) : Prop :=
  dist P O1 ^ 2 - R1 ^ 2 = dist P O2 ^ 2 - R2 ^ 2

theorem radical_axis_perpendicular {α : Type*} [MetricSpace α] (O1 O2 : α) (R1 R2 : ℝ) :
  ∃ l : LinearMap ℝ α ℝ, (∀ P : α, radical_axis O1 O2 R1 R2 P → l.to_fun (P - O1) = 0) ∧
                            ∀ P Q : α, l.to_fun (P - Q) = 0 → dist P Q ^ 2 = dist P O1 ^ 2 + dist Q O1 ^ 2 :=
sorry

end radical_axis_perpendicular_l588_588249


namespace product_of_four_integers_l588_588727

theorem product_of_four_integers (A B C D : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_pos_D : 0 < D)
  (h_sum : A + B + C + D = 36)
  (h_eq1 : A + 2 = B - 2)
  (h_eq2 : B - 2 = C * 2)
  (h_eq3 : C * 2 = D / 2) :
  A * B * C * D = 3840 :=
by
  sorry

end product_of_four_integers_l588_588727


namespace sum_of_21st_set_l588_588359

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

def first_element_of_set (n : ℕ) : ℕ := triangular_number n - n + 1

def sum_of_elements_in_set (n : ℕ) : ℕ := 
  n * ((first_element_of_set n + triangular_number n) / 2)

theorem sum_of_21st_set : sum_of_elements_in_set 21 = 4641 := by 
  sorry

end sum_of_21st_set_l588_588359


namespace avg_marks_second_class_l588_588527

theorem avg_marks_second_class
  (x : ℝ)
  (avg_class1 : ℝ)
  (avg_total : ℝ)
  (n1 n2 : ℕ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg_class1 = 30)
  (h4: avg_total = 48.75)
  (h5 : (n1 * avg_class1 + n2 * x) / (n1 + n2) = avg_total) :
  x = 60 := by
  sorry

end avg_marks_second_class_l588_588527


namespace nine_digit_palindromes_count_l588_588571

def digit_set : Finset ℕ := {6, 7, 8, 9}

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def nine_digit_positive_integer (n : ℕ) : Prop :=
  10^8 ≤ n ∧ n < 10^9

theorem nine_digit_palindromes_count :
  (Finset.filter (λ n, nine_digit_positive_integer n ∧ is_palindrome n ∧ (∀ d ∈ n.digits 10, d ∈ digit_set)) (Finset.range (10^9))).card = 1024 :=
by
  sorry

end nine_digit_palindromes_count_l588_588571


namespace area_of_original_plane_figure_l588_588924

-- Define the geometric properties
def is_isosceles_trapezoid (figure : Type) : Prop :=
  ∃ (top_base leg : ℝ), top_base = 1 ∧ leg = 1

def bottom_angle (angle : ℝ) : Prop :=
  angle = real.pi / 4

-- Define the specific conditions of the problem
variable (figure : Type) [is_isosceles_trapezoid figure]
variable (angle : ℝ) [bottom_angle angle]

-- Using the conditions to conclude the area
theorem area_of_original_plane_figure (b1 b2 height : ℝ)
    (h_top_base : b1 = 1) (h_leg : b2 = 1 + real.sqrt 2) (h_height : height = 2):
    ∃ (S : ℝ), S = 2 + real.sqrt 2 :=
by
  sorry

end area_of_original_plane_figure_l588_588924


namespace sum_of_radii_eq_12_l588_588633

noncomputable def radius_sum (r : ℝ) : ℝ := r + (6 - 2 * ℝ.sqrt 6)

theorem sum_of_radii_eq_12 :
  let r1 := 6 + 2 * ℝ.sqrt 6 in
  let r2 := 6 - 2 * ℝ.sqrt 6 in
  r1 + r2 = 12 :=
by 
  have : r1 + r2 = (6 + 2 * ℝ.sqrt 6) + (6 - 2 * ℝ.sqrt 6) := by rfl
  simp only [add_sub_cancel] at this
  exact this

end sum_of_radii_eq_12_l588_588633


namespace certain_number_value_l588_588409

variable {t b c x : ℕ}

theorem certain_number_value 
  (h1 : (t + b + c + 14 + x) / 5 = 12) 
  (h2 : (t + b + c + 29) / 4 = 15) : 
  x = 15 := 
by
  sorry

end certain_number_value_l588_588409


namespace problem_1_problem_2_problem_3_l588_588092

-- Given sequence {a_n} defined by initial condition and sum Sn
def a_n : ℕ → ℕ
def S_n : ℕ → ℕ
def b_n : ℕ → ℕ := λ n, a_n (n+1) - 2 * a_n n

-- Given conditions
axiom a1 : a_n 1 = 1
axiom S1 : ∀ n, S_n (n+1) = 4 * a_n n + 2

-- Problems to prove
theorem problem_1 : ∃ r : ℕ, ∀ n : ℕ, b_n (n+1) = r * b_n n := sorry

theorem problem_2 : ∃ d : ℕ, (∀ n : ℕ, (a_n (n+1)) / (2^(n+1)) - a_n n / (2^n) = d) ∧
                     ∃ a₀ : ℕ, ∀ n : ℕ, a_n n = (a₀ + d * (n-1)) * 2^n := sorry

theorem problem_3 : ∀ n : ℕ, S_n n = (3 * n - 4) * 2^(n-1) + 2 := sorry

end problem_1_problem_2_problem_3_l588_588092


namespace average_after_12th_innings_l588_588626

variable (runs_11 score_12 increase_avg : ℕ)
variable (A : ℕ)

theorem average_after_12th_innings
  (h1 : score_12 = 60)
  (h2 : increase_avg = 2)
  (h3 : 11 * A = runs_11)
  (h4 : (runs_11 + score_12) / 12 = A + increase_avg) :
  (A + 2 = 38) :=
by
  sorry

end average_after_12th_innings_l588_588626


namespace area_of_shaded_region_l588_588209

axiom OA : ℝ := 4
axiom OB : ℝ := 16
axiom OC : ℝ := 12
axiom similarity (EA CB : ℝ) : EA / CB = OA / OB

theorem area_of_shaded_region (DE DC : ℝ) (h_DE : DE = OC - EA)
    (h_DC : DC = 12) (h_EA_CB : EA = 3) :
    (1 / 2) * DE * DC = 54 := by
  sorry

end area_of_shaded_region_l588_588209


namespace inverse_h_l588_588967

-- Definitions from the problem conditions
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 5
def h (x : ℝ) : ℝ := f (g x)

-- Statement of the theorem for the inverse of h
theorem inverse_h : ∀ x : ℝ, h⁻¹ x = (x + 18) / 12 :=
sorry

end inverse_h_l588_588967


namespace find_number_l588_588532

-- Define the variables as nonnegative real numbers
variable (number w : ℝ)

-- Define the conditions of the problem
def condition1 := w = (number * 0.004) / 0.03
def condition2 := w ≈ 9.237333333333334

-- Theorem: Given the conditions, prove the value of the number is 69.28
theorem find_number (h₁ : condition1) (h₂ : condition2) : number = 69.28 := 
sorry

end find_number_l588_588532


namespace find_angle_D_l588_588930

variable (A B C D : ℝ)
variable (h1 : A + B = 180)
variable (h2 : C = D)
variable (h3 : C + 50 + 60 = 180)

theorem find_angle_D : D = 70 := by
  sorry

end find_angle_D_l588_588930


namespace log_pos_given_ineq_l588_588894

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588894


namespace triangle_angle_A_l588_588060

theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : a^2 - b^2 = sqrt 3 * b * c)
  (h2 : sin C = 2 * sqrt 3 * sin B) 
  (h3 : c = 2 * sqrt 3 * b) : A = 30 :=
  sorry

end triangle_angle_A_l588_588060


namespace determine_beta_l588_588392

-- Define a structure for angles in space
structure Angle where
  measure : ℝ

-- Define the conditions
def alpha : Angle := ⟨30⟩
def parallel_sides (a b : Angle) : Prop := true  -- Simplification for the example, should be defined properly for general case

-- The theorem to be proved
theorem determine_beta (α β : Angle) (h1 : α = Angle.mk 30) (h2 : parallel_sides α β) : β = Angle.mk 30 ∨ β = Angle.mk 150 := by
  sorry

end determine_beta_l588_588392


namespace adjacent_i_probability_l588_588275

theorem adjacent_i_probability :
  let total_arrangements := (10.factorial / 2.factorial : ℚ)
  let favorable_arrangements := (9.factorial : ℚ)
  favorable_arrangements / total_arrangements = 1 / 5 :=
by
  let total_arrangements := (10.factorial / 2.factorial : ℚ)
  let favorable_arrangements := (9.factorial : ℚ)
  sorry

end adjacent_i_probability_l588_588275


namespace complex_modulus_product_example_l588_588660

theorem complex_modulus_product_example :
  let z1 := Complex.mk 5 (-3)
  let z2 := Complex.mk 5 3
  abs z1 * abs z2 = 34 := by
  sorry

end complex_modulus_product_example_l588_588660


namespace shorter_piece_length_l588_588625

-- Definitions according to conditions in a)
variables (x : ℝ) (total_length : ℝ := 140)
variables (ratio : ℝ := 5 / 2)

-- Statement to be proved
theorem shorter_piece_length : x + ratio * x = total_length → x = 40 := 
by
  intros h
  sorry

end shorter_piece_length_l588_588625


namespace max_Xs_is_13_l588_588669

-- Define the conditions for the grid
def grid := fin (5 * 5) → Prop

-- Definition of the condition: no three 𝑋's in a row either vertically, horizontally, or diagonally
def no_three_in_row (g : grid) := 
  (∀ i, ∀ j, ¬((g (i * 5 + j) ∧ g (i * 5 + (j+1)) ∧ g (i * 5 + (j+2))))) ∧   -- horizontally
  (∀ i, ∀ j, ¬((g (i + j * 5) ∧ g ((i + 1) + j * 5) ∧ g ((i + 2) + j * 5)))) ∧ -- vertically
  (∀ i j, ¬((g (i * 5 + j) ∧ g ((i + 1) * 5 + (j+1)) ∧ g ((i + 2) * 5 + (j+2))))) ∧ -- main diagonal
  (∀ i j, ¬((g (i * 5 + j) ∧ g ((i + 1) * 5 + (j - 1)) ∧ g ((i + 2) * 5 + (j - 2)))))  -- anti-diagonal

noncomputable def max_Xs : ℕ := 
  if ∃ g : grid, no_three_in_row g ∧ (fin (5 * 5)).sum (λ i, if g i then 1 else 0) 
    > ∀ g' : grid, no_three_in_row g' → (fin (5 * 5)).sum (λ i, if g' i then 1 else 0) then
    (fin (5 * 5)).sum (λ i, if g i then 1 else 0)
  else 0

-- Proof problem in Lean statement
theorem max_Xs_is_13 : max_Xs = 13 := 
begin
  -- proof would be here
  sorry
end

end max_Xs_is_13_l588_588669


namespace minimum_additional_marbles_needed_l588_588093

def lisa_friends : ℕ := 14
def lisa_marbles : ℕ := 50

theorem minimum_additional_marbles_needed :
  let total_needed := (lisa_friends * (lisa_friends + 1)) / 2 in
  total_needed - lisa_marbles = 55 :=
by
  let total_needed := (14 * 15) / 2
  have : total_needed = 105 := by sorry
  have : 105 - 50 = 55 := by sorry
  exact rfl

end minimum_additional_marbles_needed_l588_588093


namespace problem_a_problem_b_l588_588370

noncomputable def f : ℝ → ℝ := sorry

variables {
  A1 : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y,
  A2 : ∀ x y : ℝ, x < y → 0 < x → 0 < y → f x < f y,
  A3 : 0 < (1 / 3 : ℝ),
  A4 : f (1 / 3) = -1
}

theorem problem_a :
  f 1 = 0 := sorry

theorem problem_b : 
  ∀ x : ℝ, f x - f (1 / (x - 2)) ≥ 2 → x > 1 + Real.sqrt 10 := sorry

end problem_a_problem_b_l588_588370


namespace projection_coordinates_l588_588385

variables (a b : ℝ × ℝ)
def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot := a.1 * b.1 + a.2 * b.2 in
  let mag2 := a.1^2 + a.2^2 in
  (dot / mag2 * a.1, dot / mag2 * a.2)

theorem projection_coordinates :
  proj (1, -2) (3, 4) = (-1, 2) :=
by
  -- Proof will be here
  sorry

end projection_coordinates_l588_588385


namespace inequality_ln_pos_l588_588839

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588839


namespace proof_incorrect_statement_B_l588_588278

def temperature := ℤ
def speed_of_sound := ℤ

def temp_data : List temperature := [-20, -10, 0, 10, 20, 30]
def sound_speed_data : List speed_of_sound := [318, 324, 330, 336, 342, 348]

def incorrect_statement_B : Prop := 
  ∀ t1 t2 (h : t1 < t2), 
  (t2 - t1) = 10 → 
  (sound_speed_data[temp_data.indexOf t2] - sound_speed_data[temp_data.indexOf t1]) ≠ -6

theorem proof_incorrect_statement_B : incorrect_statement_B :=
by sorry

end proof_incorrect_statement_B_l588_588278


namespace no_triangle_sides_conditions_l588_588114

noncomputable def smallest_b (a b : ℝ) [Fact (2 < a)] [Fact (a < b)] : ℝ :=
  (5 + Real.sqrt 17) / 4

theorem no_triangle_sides_conditions (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2))
  (h4 : ¬ (1/b + 1/a > 2 ∧ 1/b + 2 > 1/a ∧ 1/a + 2 > 1/b)) :
  b = smallest_b a b :=
by
  sorry

end no_triangle_sides_conditions_l588_588114


namespace angle_bisector_perpendicular_to_median_l588_588438

variables {α : Type*} [inner_product_space ℝ ℝ]

structure Triangle (α : Type*) := 
  (A B C : α)

structure Median (α : Type*) (T : Triangle α) := 
  (A1 : α)
  (med : A1 = (1 / 2) • (T.B + T.C))

structure AngleBisector (α : Type*) (T : Triangle α) := 
  (A2 : α)
  (angle_bis : T.B - T.A = T.C - T.A)

structure Parallel (α : Type*) (X Y Z : α) := 
  (parallel : ∃ (k : ℝ), X = k • Y)

structure Perpendicular (α : Type*) (X Y : α) := 
  (perp : inner_product_space ℝ ℝ X Y = 0)

variables {T : Triangle ℝ} 
           {M : Median ℝ T} 
           {A2 : AngleBisector ℝ T}
           {K : ℝ} -- Assume K is a point on AA1 implicitly as a parameter 

axiom Parallel_KA2_AC : Parallel ℝ (K • A2.angle_bis) (T.C - T.A)

theorem angle_bisector_perpendicular_to_median (T : Triangle ℝ) (M : Median ℝ T) (A2 : AngleBisector ℝ T) 
  (Parallel_KA2_AC : Parallel ℝ (K • A2.angle_bis) (T.C - T.A)) : 
  Perpendicular ℝ (A2.angle_bis) ((T.C - K • A2.angle_bis) - T.C) := 
sorry

end angle_bisector_perpendicular_to_median_l588_588438


namespace log_pos_given_ineq_l588_588892

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588892


namespace calculate_total_area_l588_588058

noncomputable def area_of_squares_on_XZ_and_XY (X Y Z : Type) [metric_space X] [has_dist X] 
  (XZ XY ZY : ℝ) (h1 : XZY ∈ X ∧ Y ∧ Z) (h2 : ∠XZY = π / 2) (h3 : dist ZY = 15) : Prop :=
  XZ^2 + XY^2 = 225

axiom right_triangle (XZY : Type) [metric_space XZY] (XZ XY ZY : ℝ) (h : ∠XZY = π / 2) : 
  XZ^2 + XY^2 = ZY^2

theorem calculate_total_area (H : right_triangle) (h3 : ZY = 15) : 
  area_of_squares_on_XZ_and_XY X Y Z XZ XY ZY ∠XZY (π / 2) 15 :=
sorry

end calculate_total_area_l588_588058


namespace min_value_of_M_l588_588959

theorem min_value_of_M
  (a b c : ℝ)
  (h1 : a + b + c = 12)
  (h2 : a + b > c)
  (h3 : b + c > a)
  (h4 : c + a > b) :
  ∃ (M : ℝ), M = 2.875 ∧ M = min
    (λ (a b c : ℝ), (a / (b + c - a)) + (4 * b / (c + a - b)) + (9 * c / (a + b - c))) :=
by
  sorry

end min_value_of_M_l588_588959


namespace degree_of_at_least_one_poly_ge_n_minus_1_l588_588549

theorem degree_of_at_least_one_poly_ge_n_minus_1
  (f g : ℤ[X])
  (A : ℕ → ℚ × ℚ)
  (n : ℕ)
  (h1 : ∀ k, 1 ≤ k ∧ k ≤ n → A k = (f.eval k, g.eval k))
  (h2 : regular_ngon (set.range A) n) :
  max f.natDegree g.natDegree ≥ n - 1 :=
sorry

end degree_of_at_least_one_poly_ge_n_minus_1_l588_588549


namespace area_of_shaded_region_l588_588188

-- Define points F, G, H, I, J with their coordinates
def F := (0, 0)
def G := (4, 0)
def H := (16, 0)
def I := (16, 12)
def J := (4, 3)

-- Define the similarity condition
def similar_triangles_JFG_IHG : Prop :=
  (triangle.similar F G J) (triangle.similar H G I)

-- The lengths of the segments based on problem conditions
def length_HG := 12
def length_JG := 3
def length_IG := 9

-- Area calculation of triangle IJG
def area_IJG := (1/2 * length_IG * length_JG).toReal

-- Final proof statement
theorem area_of_shaded_region :
  similar_triangles_JFG_IHG →
  length_HG = 12 →
  length_JG = length_HG/4 →
  length_IG = length_HG - length_JG →
  real.floor (area_IJG + 0.5) = 14 :=
by
  intros h_sim h_HG h_JG h_IG
  sorry

end area_of_shaded_region_l588_588188


namespace proof_original_slices_l588_588688

section

variable {Slices : Type} [Add Slices] [Mul Slices] [One Slices] [Zero Slices]

-- Andy ate 3 slices at two different points in time 
def slices_eaten_by_Andy : Slices := 3 + 3

-- Emma used 2 slices to make 1 piece of toast bread
def slices_per_toast : Slices := 2

-- Emma made 10 pieces of toast bread
def toasts_made : Slices := 10

-- After making toast, she had 1 slice left
def slices_left_after_toasting : Slices := 1

-- Calculating the total number of slices used for toast
def slices_used_for_toast : Slices := toasts_made * slices_per_toast

-- Total number of slices before Emma made the toast
def slices_before_toasting : Slices := slices_used_for_toast + slices_left_after_toasting

-- Calculating the original number of slices in the loaf
def original_loaf_slices : Slices := slices_before_toasting + slices_eaten_by_Andy

-- Proving the original number of slices in the loaf
theorem proof_original_slices : original_loaf_slices = 27 := by
  sorry

end

end proof_original_slices_l588_588688


namespace mandy_reading_ratio_l588_588983

theorem mandy_reading_ratio (x : ℕ) :
  (96 * x = 480) →
  (8 * x / 8 = 5) :=
by
  intros
  have hx : x = 5 := by
    rw [← (nat.div_eq_of_eq_mul_left (by decide) (by linarith))] at this
    exact this
  rw hx
  norm_num
  sorry

end mandy_reading_ratio_l588_588983


namespace shaded_area_of_triangle_CDE_l588_588194

-- Definitions of the points
noncomputable def O := (0, 0 : ℝ×ℝ)
noncomputable def A := (4, 0 : ℝ×ℝ)
noncomputable def B := (16, 0 : ℝ×ℝ)
noncomputable def C := (16, 12 : ℝ×ℝ)
noncomputable def D := (4, 12 : ℝ×ℝ)
noncomputable def E := (4, 3 : ℝ×ℝ)

-- Definition of the area calculation for the given triangle
theorem shaded_area_of_triangle_CDE : 
  let DE := 9 in
  let DC := 12 in
  (DE * DC) / 2 = 54 :=
by
  sorry

end shaded_area_of_triangle_CDE_l588_588194


namespace parallel_lines_m_l588_588738

theorem parallel_lines_m (m : ℝ) (h₁ : ∀ x y, mx + y + 1 = 0) (h₂ : ∀ x y, 9x + my + 2m + 3 = 0) :
  (∃ m, m = 3 ∧ (∀ x y, mx + y + 1 = 0) ∧ (∀ x y, 9x + my + 2m + 3 = 0) ∧ m ≠ 0) :=
sorry

end parallel_lines_m_l588_588738


namespace return_time_l588_588650

-- Define Annika's hike down the trail with given conditions
def hiking_rate : ℝ := 10 -- minutes per kilometer
def hiked_east_initial : ℝ := 2.5 -- kilometers
def hiked_east_total : ℝ := 3 -- kilometers

-- The theorem stating the total time required to return to the start is 35 minutes
theorem return_time : 
  (hiking_rate : ℝ) = 10 → 
  (hiked_east_initial : ℝ) = 2.5 → 
  (hiked_east_total : ℝ) = 3 →
  (total_return_time : ℝ) = (5 + 30) := 
by
  intros hr hei het
  have hr_eq : hr = hiking_rate, from rfl
  have hei_eq : hei = hiked_east_initial, from rfl
  have het_eq : het = hiked_east_total, from rfl
  have remaining_distance : ℝ := het - hei
  have remaining_time := remaining_distance * hr
  have return_time := het * hr
  have total_time := (remaining_time + return_time)
  have total_time_eq : total_time = 35, sorry
  exact total_time_eq

end return_time_l588_588650


namespace monotonicity_f_inequality_proof_l588_588976

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem monotonicity_f :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (x + ε)) ∧ (∀ x : ℝ, 1 < x → f (x - ε) > f x) := 
sorry

theorem inequality_proof (x : ℝ) (hx : 1 < x) :
  1 < (x - 1) / Real.log x ∧ (x - 1) / Real.log x < x :=
sorry

end monotonicity_f_inequality_proof_l588_588976


namespace ln_eq_mn_l588_588348

-- Definitions of the points and lines
variables {α : Type*} [metric_space α] 

/-- Represents a point outside the triangle -/
variable (P : α)

/-- Represents points A, B, C forming the triangle -/
variables (A B C : α)

/-- Represents the altitude from vertex A to side BC -/
variable (D : α)

/-- Represents the midpoint E of side BC -/
variable (E : α)

/-- Represents points L, M, and N on altitude AD due to perpendiculars from P to AB, AC, and AE respectively -/
variables (L M N : α)

/-- The main theorem stating that LN = MN -/
theorem ln_eq_mn : 
  (let 
    a := dist A B, 
    b := dist A C, 
    m := dist B C / 2, 
    d := dist A D, 
    e := dist D E,
    l := dist L N, 
    n := dist M N
  in (l = n)) := 
  sorry

end ln_eq_mn_l588_588348


namespace c_eq_a_l588_588991

section Sequences

variable (a : ℕ → ℕ)

-- Non-increasing sequence of positive integers and an additional 0
axiom pos_seq (n : ℕ) : a n > 0
axiom non_increasing (n m : ℕ) : n < m → a m ≤ a n
axiom last_zero : ∃ n, a (n + 1) = 0

-- Define b_k as the count of numbers in a greater than k
def b (k : ℕ) : ℕ := {i | a i > k}.to_finset.card

-- Define c_k in a similar manner
def c (k : ℕ) : ℕ := {i | b i > k}.to_finset.card

-- Prove that c_k = a_k for all k
theorem c_eq_a (k : ℕ) : c k = a k :=
by
  sorry

end Sequences

end c_eq_a_l588_588991


namespace incorrect_statements_about_functions_l588_588507

/-- Proof problem conditions and statements to be validated --/
theorem incorrect_statements_about_functions :
  let S1 := {x : ℝ | x ≤ 0}
  let R1 := {y : ℝ | 0 < y ∧ y ≤ 1}
  let S2 := {x : ℝ | x > 2}
  let R2 := {y : ℝ | 0 < y ∧ y < 1 / 2}
  let R3 := {y : ℝ | 0 ≤ y ∧ y ≤ 4}
  let S3 := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
  let R4 := {y : ℝ | y ≤ 3}
  let S4 := {x : ℝ | 0 < x ∧ x ≤ 8}
  (∀ x ∈ S1, 2^x ∈ R1) ∧
  (∀ x ∈ S2, 1/x ∈ R2) ∧
  (∃ x ∈ S3, x^2 ∈ R3) ∧
  (∀ x ∈ S4, log 2 x ∈ R4) :=
begin
  sorry
end

end incorrect_statements_about_functions_l588_588507


namespace log_of_y_sub_x_plus_one_positive_l588_588896

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588896


namespace roots_calculation_l588_588465
noncomputable theory

-- Definitions based on conditions
def quadratic_eq (x : ℝ) : ℝ := x^2 + 9 * x - 106

def roots := {
  p : ℝ,
  q : ℝ,
  h_distinct : p ≠ q,
  h_p : quadratic_eq p = 0,
  h_q : quadratic_eq q = 0
}

open roots

-- The proof goal
theorem roots_calculation (r : roots) : (r.p - 2) * (r.q - 2) = -84 := by
  sorry

end roots_calculation_l588_588465


namespace fraction_of_juniors_equals_seniors_l588_588428

theorem fraction_of_juniors_equals_seniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J * 7 = 4 * (J + S)) : J / S = 4 / 3 :=
sorry

end fraction_of_juniors_equals_seniors_l588_588428


namespace pigs_and_dogs_more_than_sheep_l588_588569

-- Define the number of pigs and sheep
def numberOfPigs : ℕ := 42
def numberOfSheep : ℕ := 48

-- Define the number of dogs such that it is the same as the number of pigs
def numberOfDogs : ℕ := numberOfPigs

-- Define the total number of pigs and dogs
def totalPigsAndDogs : ℕ := numberOfPigs + numberOfDogs

-- State the theorem about the difference between pigs and dogs and the number of sheep
theorem pigs_and_dogs_more_than_sheep :
  totalPigsAndDogs - numberOfSheep = 36 := 
sorry

end pigs_and_dogs_more_than_sheep_l588_588569


namespace symmetry_center_of_sine_function_l588_588538

theorem symmetry_center_of_sine_function :
  symmetry_center (λ x : Real, 2 * Real.sin (2 * x - π / 6)) = (π / 12, 0) :=
by
  sorry

end symmetry_center_of_sine_function_l588_588538


namespace segment_length_l588_588578

theorem segment_length (x : ℝ) (h : |x - (27^(1/3))| = 5) : ∃ a b : ℝ, a = 8 ∧ b = -2 ∧ |a - b| = 10 :=
by
  have hx1 : x = 27^(1/3) + 5 ∨ x = 27^(1/3) - 5 := abs_eq hx
  use [8, -2]
  split
  { calc 8 = 27^(1/3) + 5 := by sorry }
  split
  { calc -2 = 27^(1/3) - 5 := by sorry }
  { calc |8 - -2| = |8 + 2| := by sorry }

end segment_length_l588_588578


namespace parallelogram_area_zero_l588_588693

variables (A B C D : ℝ × ℝ × ℝ)
def is_parallelogram (A B C D : ℝ × ℝ × ℝ) : Prop :=
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) ∧
  ((B - A) × (D - A) = 0) ∧
  ((C - A) × (D - A) = 0)

noncomputable def area_of_parallelogram (A B C D : ℝ × ℝ × ℝ) [is_parallelogram A B C D] : ℝ :=
  (B - A).cross_product (D - A).norm

theorem parallelogram_area_zero :
  ∀ (A B C D : ℝ × ℝ × ℝ),
    A = (2, 4, 6) →
    B = (7, 9, 11) →
    C = (1, 3, 5) →
    D = (6, 8, 10) →
    area_of_parallelogram A B C D = 0 :=
by { sorry }

end parallelogram_area_zero_l588_588693


namespace observer_height_proof_l588_588291

noncomputable def height_observer (d m α β : ℝ) : ℝ :=
  let cot_alpha := 1 / Real.tan α
  let cot_beta := 1 / Real.tan β
  let u := (d * (m * cot_beta - d)) / (2 * d - m * (cot_beta - cot_alpha))
  20 + Real.sqrt (400 + u * m * cot_alpha - u^2)

theorem observer_height_proof :
  height_observer 290 40 (11.4 * Real.pi / 180) (4.7 * Real.pi / 180) = 52 := sorry

end observer_height_proof_l588_588291


namespace cost_of_new_game_l588_588116

theorem cost_of_new_game (initial_money : ℕ) (money_left : ℕ) (toy_cost : ℕ) (toy_count : ℕ)
  (h_initial : initial_money = 68) (h_toy_cost : toy_cost = 7) (h_toy_count : toy_count = 3) 
  (h_money_left : money_left = toy_count * toy_cost) :
  initial_money - money_left = 47 :=
by {
  sorry
}

end cost_of_new_game_l588_588116


namespace pure_imaginary_implies_a_eq_3_l588_588760

def complex_expression (a : ℝ) : ℂ :=
  a - 10 / (⟨3, -1⟩ : ℂ)

theorem pure_imaginary_implies_a_eq_3 (a : ℝ) (h : ∃ b : ℝ, complex_expression(a) = ⟨0, b⟩) : a = 3 :=
by
  sorry

end pure_imaginary_implies_a_eq_3_l588_588760


namespace red_squares_in_9x11_rectangle_l588_588649

theorem red_squares_in_9x11_rectangle :
  (∀ (R : ℕ → ℕ → Prop),
    (∀ i j, (i % 2 = 0 ∧ j % 3 = 0 ∨ i % 2 = 1 ∧ j % 3 = 2) → R i j) → 
    (forall a b, 
      (∀ i j, 0 ≤ i < 2 → 0 ≤ j < 3 → (R (a + i) (b + j) ↔ i * 3 + j = 1 ∨ i * 3 + j = 4)) → 
      (∃ res, (∀ i j, 0 ≤ i < 9 → 0 ≤ j < 11 → (R i j ↔ (i * 11 + j) ∈ res)) ∧ multiset.card res = 33))) :=
sorry

end red_squares_in_9x11_rectangle_l588_588649


namespace concyclic_points_l588_588500

variable {K L M O Q P : Type}
variable [MetricSpace K] [MetricSpace L] [MetricSpace M] [MetricSpace O] [MetricSpace Q] [MetricSpace P]
variable {circumcircle_KLM : Circumcircle K L M}
variable {circumcenter_O : Circumcenter K L M O}
variable {angle_L : Angle L O M}
variable {ray_LO : Ray L O}
variable {intersection_Q : ray_LO ∩ LineSegment K M = {Q}}
variable {circumcircle_QOM : Circumcircle Q O M}

def midpoint_of_arc (P : Type) (circ : Circumcircle Q O M) (A B : Type) : Prop :=
  ∃ M1 M2, midpoint M1 circ A B ∧ midpoint M2 circ A B ∧ M1 = M2 ∧ arc_containing M1 M2 = P

-- The proof statement:
theorem concyclic_points (K L M O Q P : Type) [MetricSpace K] [MetricSpace L] [MetricSpace M] [MetricSpace O] [MetricSpace Q] [MetricSpace P]
  (circumcenter_O : Circumcenter K L M O) 
  (angle_L : ∠ L O M = 30)
  (intersection_Q : (Ray L O) ∩ (LineSegment K M) = {Q})
  (midpoint_of_arc_P : midpoint_of_arc P (Circumcircle Q O M) O M) :
  Concyclic K L P Q :=
sorry

end concyclic_points_l588_588500


namespace parabola_point_comparison_l588_588743

theorem parabola_point_comparison (a : ℝ) (h : a > 0) (y1 y2 : ℝ) :
  let A := (-2, y1) in
  let B := (1, y2) in
  y1 = a * (-2)^2 ∧ y2 = a * (1)^2 → y1 > y2 :=
by
  intro h_eq
  let ⟨ha, hb⟩ := h_eq
  sorry

end parabola_point_comparison_l588_588743


namespace length_of_AC_in_isosceles_triangle_l588_588430

theorem length_of_AC_in_isosceles_triangle (AC BC : ℝ) (h_iso : ∃ A B C : ℝ, is_isosceles_triangle A B C)
  (h_base : BC = 8) (h_abs : |AC - BC| = 2) :
  AC = 10 ∨ AC = 6 :=
by
  sorry

end length_of_AC_in_isosceles_triangle_l588_588430


namespace log_y_minus_x_plus_1_pos_l588_588871

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588871


namespace rectangle_area_l588_588046

theorem rectangle_area (r : ℝ) 
  (h_w_pos : 0 < r) : 
  ∃ (w h : ℝ), 
    (h = r ∧ 
    ∃ A B C D : ℝ × ℝ, 
      let AC := (D.1 - A.1)^2 + (D.2 - A.2)^2 in
      let AN := (1 / 3) * AC in
      let N := (A.1 + (1 / 3)*(C.1 - A.1), A.2 + (1 / 3)*(C.2 - A.2)) in
      (w = sqrt((9/8) * r^2)) ∧
      (∃ O : ℝ × ℝ, 
        let P := (O.1, O.2 + r),
        let Q := (O.1 - r, O.2),
        let R := (O.1, O.2 - r) in
        (O.1 = D.1) ∧
        (O.2 = D.2) ∧
        (N.1^2 + N.2^2 = r^2) ∧
        (A.1, A.2) ∈ {P, Q, R}) ∧
    (w * h = ∃ a, (a = (2 * sqrt 2 / 3) * r^2)) :=
sorry

end rectangle_area_l588_588046


namespace shaded_area_of_triangle_CDE_l588_588196

-- Definitions of the points
noncomputable def O := (0, 0 : ℝ×ℝ)
noncomputable def A := (4, 0 : ℝ×ℝ)
noncomputable def B := (16, 0 : ℝ×ℝ)
noncomputable def C := (16, 12 : ℝ×ℝ)
noncomputable def D := (4, 12 : ℝ×ℝ)
noncomputable def E := (4, 3 : ℝ×ℝ)

-- Definition of the area calculation for the given triangle
theorem shaded_area_of_triangle_CDE : 
  let DE := 9 in
  let DC := 12 in
  (DE * DC) / 2 = 54 :=
by
  sorry

end shaded_area_of_triangle_CDE_l588_588196


namespace average_fish_per_person_l588_588914

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l588_588914


namespace average_speed_of_trip_l588_588268

theorem average_speed_of_trip :
  let total_distance := 50 -- in kilometers
  let distance1 := 25 -- in kilometers
  let speed1 := 66 -- in kilometers per hour
  let distance2 := 25 -- in kilometers
  let speed2 := 33 -- in kilometers per hour
  let time1 := distance1 / speed1 -- time taken for the first part
  let time2 := distance2 / speed2 -- time taken for the second part
  let total_time := time1 + time2 -- total time for the trip
  let average_speed := total_distance / total_time -- average speed of the trip
  average_speed = 44 := by
{
  sorry
}

end average_speed_of_trip_l588_588268


namespace average_mark_excluded_students_l588_588141

variables (N A E A_R A_E : ℕ)

theorem average_mark_excluded_students:
    N = 56 → A = 80 → E = 8 → A_R = 90 →
    N * A = E * A_E + (N - E) * A_R →
    A_E = 20 :=
by
  intros hN hA hE hAR hEquation
  rw [hN, hA, hE, hAR] at hEquation
  have h : 4480 = 8 * A_E + 4320 := hEquation
  sorry

end average_mark_excluded_students_l588_588141


namespace inequality_implies_log_pos_l588_588822

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588822


namespace triangle_angle_problem_l588_588362

theorem triangle_angle_problem
  {A B C P Q : Type}
  (isosceles_ABC : triangle A B C ∧ AB = AC ∧ AB < BC)
  (P_condition : ∃ P, ∉ segment B C ∧ BP = BA)
  (Q_condition : ∃ Q, ∉ A ∧ CQ = CA)
  (angle_QPC : ∃ x, angle Q P C = x) :
  angle P C Q = 180 - 4 * x :=
sorry

end triangle_angle_problem_l588_588362


namespace find_ab_value_l588_588921

noncomputable def complex_find_ab (z1 z2 : ℂ) : ℝ :=
let a := (2 : ℝ)
let b := (1 : ℝ)
in a + b

theorem find_ab_value (a b : ℝ) (z1 z2 : ℂ)
  (h1 : z1 = -1 + complex.I * a)
  (h2 : z2 = b + 2 * complex.I)
  (h3 : ∀ z : ℂ, z = z1 - z2 → z.im = 0)
  (h4 : ∀ z : ℂ, z = z1 * z2 → z.im = 0) :
  complex_find_ab z1 z2 = 3 :=
by sorry

end find_ab_value_l588_588921


namespace gcd_of_5_digit_permutations_l588_588334

open Int Nat

-- Define a function to generate all permutations of the list [1, 2, 3, 4, 5]
def permutations : List (List Int) := 
  ([1, 2, 3, 4, 5] : List Int).permutations

-- Convert each permutation to a five-digit number
def to_nat (l : List Int) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

-- Generate all five-digit numbers from permutations of [1, 2, 3, 4, 5]
def five_digit_numbers : List Nat := permutations.map to_nat

-- Predicate to check if a number is divisible by 3
def divisible_by_3 (n : Nat) : Prop := n % 3 = 0

theorem gcd_of_5_digit_permutations :
  Nat.gcd_list five_digit_numbers = 3 := by
  sorry

end gcd_of_5_digit_permutations_l588_588334


namespace log_y_minus_x_plus_1_pos_l588_588872

theorem log_y_minus_x_plus_1_pos (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_pos_l588_588872


namespace ratio_of_areas_eq_100_l588_588954

def T₁ (x y : ℝ) : Prop := log (3 + x^2 + y^2) / log 10 ≤ 3 + log (x + y) / log 10
def T₂ (x y : ℝ) : Prop := log (4 + x^2 + y^2) / log 10 ≤ 4 + log (x + y) / log 10

theorem ratio_of_areas_eq_100 :
  let area_T₁ := π * 499997 in
  let area_T₂ := π * 49996004 in
  area_T₂ / area_T₁ = 100 :=
by
  sorry

end ratio_of_areas_eq_100_l588_588954


namespace oranges_left_uneaten_l588_588492

variable (total_oranges : ℕ)
variable (half_oranges ripe_oranges unripe_oranges eaten_ripe_oranges eaten_unripe_oranges uneaten_ripe_oranges uneaten_unripe_oranges total_uneaten_oranges : ℕ)

axiom h1 : total_oranges = 96
axiom h2 : half_oranges = total_oranges / 2
axiom h3 : ripe_oranges = half_oranges
axiom h4 : unripe_oranges = half_oranges
axiom h5 : eaten_ripe_oranges = ripe_oranges / 4
axiom h6 : eaten_unripe_oranges = unripe_oranges / 8
axiom h7 : uneaten_ripe_oranges = ripe_oranges - eaten_ripe_oranges
axiom h8 : uneaten_unripe_oranges = unripe_oranges - eaten_unripe_oranges
axiom h9 : total_uneaten_oranges = uneaten_ripe_oranges + uneaten_unripe_oranges

theorem oranges_left_uneaten : total_uneaten_oranges = 78 := by
  sorry

end oranges_left_uneaten_l588_588492


namespace collinear_X_Y_C_l588_588170

-- Definitions of the circles and their centers
variables {S_A S_B S_C : Type} [sphere S_A] [sphere S_B] [sphere S_C]
variables {A B C : Type} [point A] [point B] [point C]
variables {C' B' A' : Type} [point C'] [point B'] [point A']

-- Definitions of the common tangents and their intersection
variables {ℓ_A ℓ_B : Type} [line ℓ_A] [line ℓ_B]
variables {X Y : Type} [point X] [point Y]

-- Specific conditions 
variables {angle_XBY : angle X B Y = 90} {angle_YAX : angle Y A X = 90}
variables {AC BC : distance A C = distance B C}

-- Problem Statement
theorem collinear_X_Y_C'_iff_AC_eq_BC
  (tangent_SA_SB : tangent S_A S_B C')
  (tangent_SA_SC : tangent S_A S_C B')
  (tangent_SB_SC : tangent S_B S_C A')
  (common_tangent_SA_SC : common_tangent ℓ_B S_A S_C B')
  (common_tangent_SB_SC : common_tangent ℓ_A S_B S_C A')
  (intersect_ell_X : intersect ℓ_A ℓ_B X)
  (right_angle_XBY : right_angle X B Y)
  (right_angle_YAX : right_angle Y A X) :
  (collinear X Y C') ↔ (distance A C = distance B C) :=
sorry

end collinear_X_Y_C_l588_588170


namespace six_letter_vowel_words_count_l588_588523

noncomputable def vowel_count_six_letter_words : Nat := 27^6

theorem six_letter_vowel_words_count :
  vowel_count_six_letter_words = 531441 :=
  by
    sorry

end six_letter_vowel_words_count_l588_588523


namespace find_f_minus_one_l588_588674

-- Given the function condition
def f (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) - 1) / (x^31 - 1)

theorem find_f_minus_one : f (-1) = 1 / 15 := 
by
  sorry

end find_f_minus_one_l588_588674


namespace largest_remainder_2018_l588_588284

def largest_remainder (n m : ℕ) : ℕ :=
  nat.find_greatest (λ r, ∃ q, n = q * m + r ∧ 0 ≤ r ∧ r < m) (n % m)

/-- The largest remainder when 2018 is divided by any integer from 1 to 1000 is 672. -/
theorem largest_remainder_2018 : 
  (∃ d (h : d ∈ finset.range 1000), largest_remainder 2018 (d + 1) = 672) :=
begin
  sorry
end

end largest_remainder_2018_l588_588284


namespace geographic_projection_l588_588501

theorem geographic_projection :
  ∃ (equator prime_meridian : straight_line)
    (meridians : list (straight_line))
    (parallels : list (concentric_circle)),
    -- equator as a straight line
    is_equator_line equator ∧
    -- prime meridian as a straight line perpendicular to the equator
    is_prime_meridian_line prime_meridian ∧
    perpendicular equator prime_meridian ∧
    -- 11 more meridians spaced 30° apart
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 11 → is_meridian_line (list.nth meridians i) i) ∧
    -- latitude circles as orthogonal concentric rings to the meridians
    (∀ (i : ℕ), is_parallel_circle (list.nth parallels i)) :=
by
  sorry

end geographic_projection_l588_588501


namespace simplify_fraction_l588_588329

theorem simplify_fraction:
  ((1/2 - 1/3) / (3/7 + 1/9)) * (1/4) = 21/272 :=
by
  sorry

end simplify_fraction_l588_588329


namespace derived_sequence_properties_l588_588746

noncomputable def is_derived_sequence (a b : ℕ → ℤ) (n : ℕ) :=
  b 1 = a n ∧ ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k-1) + a k - b (k-1)

theorem derived_sequence_properties
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (c : ℕ → ℤ)
  (n : ℕ)
  (h : n = 4)
  (h_b : is_derived_sequence a b n)
  (h_b_values : b 1 = 5 ∧ b 2 = -2 ∧ b 3 = 7 ∧ b 4 = 2) :
  (a 1 = 2 ∧ a 2 = 1 ∧ a 3 = 4 ∧ a 4 = 5) ∧
  (∀ n, (even n) → is_derived_sequence b c n ∧ c = a) :=
sorry

end derived_sequence_properties_l588_588746


namespace ball_fall_time_l588_588618

theorem ball_fall_time (h g : ℝ) (t : ℝ) : 
  h = 20 → g = 10 → h + 20 * (t - 2) - 5 * ((t - 2) ^ 2) = t * (20 - 10 * (t - 2)) → 
  t = Real.sqrt 8 := 
by
  intros h_eq g_eq motion_eq
  sorry

end ball_fall_time_l588_588618


namespace time_descend_hill_l588_588988

-- Definitions
def time_to_top : ℝ := 4
def avg_speed_whole_journey : ℝ := 3
def avg_speed_uphill : ℝ := 2.25

-- Theorem statement
theorem time_descend_hill (t : ℝ) 
  (h1 : time_to_top = 4) 
  (h2 : avg_speed_whole_journey = 3) 
  (h3 : avg_speed_uphill = 2.25) : 
  t = 2 := 
sorry

end time_descend_hill_l588_588988


namespace log_pos_if_exp_diff_l588_588881

theorem log_pos_if_exp_diff :
  ∀ (x y : ℝ), (2^x - 2^y < 3^(-x) - 3^(-y)) → (Real.log (y - x + 1) > 0) :=
by
  intros x y h
  sorry

end log_pos_if_exp_diff_l588_588881


namespace cost_of_items_l588_588658

theorem cost_of_items (x y z : ℝ)
  (h1 : 20 * x + 3 * y + 2 * z = 32)
  (h2 : 39 * x + 5 * y + 3 * z = 58) :
  5 * (x + y + z) = 30 := by
  sorry

end cost_of_items_l588_588658


namespace probability_of_multiple_of_3_is_3_over_10_l588_588176

def tickets := set.range (λ n, n + 1)

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def count_multiples_of_3 : ℕ :=
  (set.filter is_multiple_of_3 tickets).card

def total_tickets : ℕ := 20

def probability_multiple_of_3 : ℝ :=
  (count_multiples_of_3 : ℝ) / (total_tickets : ℝ)

theorem probability_of_multiple_of_3_is_3_over_10 :
  probability_multiple_of_3 = 3 / 10 := by
  sorry

end probability_of_multiple_of_3_is_3_over_10_l588_588176


namespace solve_equation_l588_588517

theorem solve_equation :
  ∃ (x y z : ℕ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 4 ∧ 
  (2^x + 3^y + 7 = nat.factorial z) ∧
  ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
by {
  sorry
}

end solve_equation_l588_588517


namespace tangents_intersection_circle_l588_588087

theorem tangents_intersection_circle 
  (O₁ O₂ : Point) 
  (r₁ r₂ : ℝ) 
  (A B C D : Point)
  (hO₁ : dist O₁ A = r₁ ∧ dist O₁ B = r₁)
  (hO₂ : dist O₂ C = r₂ ∧ dist O₂ D = r₂)
  (line : Line)
  (hline_A : A ∈ line)
  (hline_B : B ∈ line)
  (hline_C : C ∈ line)
  (hline_D : D ∈ line) :
  ∃ (circle : Circle) (O : Point), 
    OnCircle (circle, A) ∧
    OnCircle (circle, B) ∧
    OnCircle (circle, C) ∧
    OnCircle (circle, D) ∧
    Collinear O O₁ O₂ :=
by
  sorry

end tangents_intersection_circle_l588_588087


namespace cost_of_dozen_pens_before_discount_l588_588533

/-
  Given the combined cost of 3 pens and 5 pencils with a 10% discount is Rs. 200,
  and the ratio of the cost of one pen to one pencil before the discount is 5:1,
  prove that the cost of one dozen pens before the discount is approximately Rs. 666.60.
-/
theorem cost_of_dozen_pens_before_discount (cost_3pens_5pencils_discounted : ℝ)
  (ratio_pen_pencil : ℝ) :
  cost_3pens_5pencils_discounted = 200 →
  ratio_pen_pencil = 5 →
  ∃ (pen_cost : ℝ) (pencil_cost : ℝ),
    pen_cost / pencil_cost = 5 ∧
    (12 * pen_cost ≈ 666.60) :=
by
  intros h1 h2
  sorry

end cost_of_dozen_pens_before_discount_l588_588533


namespace sum_reciprocals_prime_factors_l588_588459

theorem sum_reciprocals_prime_factors :
  let B := {n : ℕ | ∀ p : ℕ, nat.prime p → p ∣ n → p ∈ {2, 3, 5, 7}} in 
  let sum_reciprocals := ∑' (n ∈ B), (1 : ℚ) / n in
  let ⟨p, q⟩ := num_denom sum_reciprocals in
  nat.gcd p q = 1 →
  p + q = 43 := 
by {
  let B := {n : ℕ | ∀ p : ℕ, prime p → p ∣ n → p ∈ {2, 3, 5, 7}},
  let sum_reciprocals := ∑' (n ∈ B), (1 : ℚ) / n,
  have num_denom_lemma : ∃ p q, sum_reciprocals = p / q ∧ nat.gcd p q = 1,
  { sorry },
  cases num_denom_lemma with p hp_q,
  exact hp_q.2.
}

end sum_reciprocals_prime_factors_l588_588459


namespace segment_length_eq_ten_l588_588591

theorem segment_length_eq_ten :
  (abs (8 - (-2)) = 10) :=
by
  -- Given conditions
  have h1 : 8 = real.cbrt 27 + 5 := sorry
  have h2 : -2 = real.cbrt 27 - 5 := sorry
  
  -- Using the conditions to prove the length
  sorry

end segment_length_eq_ten_l588_588591


namespace segment_length_eq_ten_l588_588594

theorem segment_length_eq_ten :
  (abs (8 - (-2)) = 10) :=
by
  -- Given conditions
  have h1 : 8 = real.cbrt 27 + 5 := sorry
  have h2 : -2 = real.cbrt 27 - 5 := sorry
  
  -- Using the conditions to prove the length
  sorry

end segment_length_eq_ten_l588_588594


namespace proof_complement_union_l588_588479

-- Definition of the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Definition of the subset A
def A : Finset ℕ := {0, 3, 4}

-- Definition of the subset B
def B : Finset ℕ := {1, 3}

-- Definition of the complement of A in U
def complement_A : Finset ℕ := U \ A

-- Definition of the union of the complement of A and B
def union_complement_A_B : Finset ℕ := complement_A ∪ B

-- Statement of the theorem
theorem proof_complement_union :
  union_complement_A_B = {1, 2, 3} :=
sorry

end proof_complement_union_l588_588479


namespace tangent_line_at_2_range_of_a_l588_588786

noncomputable def f (x : ℝ) : ℝ := x - log x - 1

theorem tangent_line_at_2: 
  ∃ m b, (f'(⟨2, sorry⟩) = m) ∧ (f 2 = 2 * m + b) ∧ (1 - log 2 = 1/2 * (2 - 2) + 2)  ∧ (x - 2 * x - log 4 = 0) := 
sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → f x ≥ a * x - 2) : 
  a ≤ 1 - 1 / exp 2 := 
sorry

end tangent_line_at_2_range_of_a_l588_588786


namespace total_students_in_school_l588_588237

theorem total_students_in_school (p : ℝ) (n_8 n_>8 T : ℝ)
  (h1 : p = 0.20)
  (h2 : n_8 = 60)
  (h3 : n_>8 = (2 / 3) * n_8)
  (h4 : 0.80 * T = n_8 + n_>8) :
  T = 125 :=
by
  sorry

end total_students_in_school_l588_588237


namespace time_of_A_l588_588927

/-- Define the length of the race and the distances by which A beats B --/
def race_length : ℝ := 280
def distance_beat : ℝ := 56
def time_beat : ℝ := 7

/-- Define the speeds of A and B, and A's time to complete the race --/
def V_a (T_a : ℝ) : ℝ := race_length / T_a
def V_b (T_a : ℝ) : ℝ := (race_length - distance_beat) / (T_a - time_beat)

/-- Main theorem stating A's time over the course --/
theorem time_of_A (T_a : ℝ) (h1 : V_a T_a = distance_beat / time_beat) : T_a = 35 :=
by
  sorry

end time_of_A_l588_588927


namespace PolynomialCoefficientSum_l588_588731

theorem PolynomialCoefficientSum :
  let a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7 : ℤ
  (h : (1 - 2*x)^7 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 + a_6*x^6 + a_7*x^7) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 :=
by
  -- Proof goes here
  sorry

end PolynomialCoefficientSum_l588_588731


namespace ROI_difference_l588_588687

-- Definitions based on the conditions
def Emma_investment : ℝ := 300
def Briana_investment : ℝ := 500
def Emma_yield : ℝ := 0.15
def Briana_yield : ℝ := 0.10
def years : ℕ := 2

-- The goal is to prove that the difference between their 2-year ROI is $10
theorem ROI_difference :
  let Emma_ROI := Emma_investment * Emma_yield * years
  let Briana_ROI := Briana_investment * Briana_yield * years
  (Briana_ROI - Emma_ROI) = 10 :=
by
  sorry

end ROI_difference_l588_588687


namespace car_truck_meet_distance_l588_588328

theorem car_truck_meet_distance
  (S x y : ℝ)
  (h_truck_meet : 18 = (0.75 * x + ((S - 0.75 * x) * x) / (x + y) - (S * x) / (x + y)))
  (h_speed : (x * y) / (x + y) = 24) :
  let k := ((1 / 3) * y * x) / (x + y)
  in k = 8 :=
by
  sorry

end car_truck_meet_distance_l588_588328


namespace sqrt_product_eq_225_l588_588121

theorem sqrt_product_eq_225 : (Real.sqrt (5 * 3) * Real.sqrt (3 ^ 3 * 5 ^ 3) = 225) :=
by
  sorry

end sqrt_product_eq_225_l588_588121


namespace ln_eq_mn_l588_588347

-- Definitions of the points and lines
variables {α : Type*} [metric_space α] 

/-- Represents a point outside the triangle -/
variable (P : α)

/-- Represents points A, B, C forming the triangle -/
variables (A B C : α)

/-- Represents the altitude from vertex A to side BC -/
variable (D : α)

/-- Represents the midpoint E of side BC -/
variable (E : α)

/-- Represents points L, M, and N on altitude AD due to perpendiculars from P to AB, AC, and AE respectively -/
variables (L M N : α)

/-- The main theorem stating that LN = MN -/
theorem ln_eq_mn : 
  (let 
    a := dist A B, 
    b := dist A C, 
    m := dist B C / 2, 
    d := dist A D, 
    e := dist D E,
    l := dist L N, 
    n := dist M N
  in (l = n)) := 
  sorry

end ln_eq_mn_l588_588347


namespace length_of_segment_eq_ten_l588_588585

theorem length_of_segment_eq_ten (x : ℝ) (h : |x - real.cbrt 27| = 5) : 
  let y1 := real.cbrt 27 + 5,
      y2 := real.cbrt 27 - 5
  in abs (y1 - y2) = 10 := 
by
  sorry

end length_of_segment_eq_ten_l588_588585


namespace mono_increasing_interval_l588_588544

noncomputable def y (x : ℝ) : ℝ := x - 2 * sin x

theorem mono_increasing_interval :
  set.Ioo (π / 3) (5 * π / 3) = { x : ℝ | 0 < x ∧ x < 2 * π ∧ 1 - 2 * cos x > 0 } :=
by
  sorry

end mono_increasing_interval_l588_588544


namespace primes_div_order_l588_588084

theorem primes_div_order (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : q ∣ 3^p - 2^p) : p ∣ q - 1 :=
sorry

end primes_div_order_l588_588084


namespace pizza_slices_correct_l588_588067

-- Definitions based on conditions
def john_slices : Nat := 3
def sam_slices : Nat := 2 * john_slices
def eaten_slices : Nat := john_slices + sam_slices
def remaining_slices : Nat := 3
def total_slices : Nat := eaten_slices + remaining_slices

-- The statement to be proven.
theorem pizza_slices_correct : total_slices = 12 := by
  sorry

end pizza_slices_correct_l588_588067


namespace lily_lottery_tickets_l588_588481

theorem lily_lottery_tickets (n : ℕ) :
  (∑ i in Finset.range (n + 1), (i + 1)) = 15 → n = 5 :=
by
  -- Placeholder, the actual proof will go here
  sorry

end lily_lottery_tickets_l588_588481


namespace area_of_shaded_region_l588_588191

-- Define points F, G, H, I, J with their coordinates
def F := (0, 0)
def G := (4, 0)
def H := (16, 0)
def I := (16, 12)
def J := (4, 3)

-- Define the similarity condition
def similar_triangles_JFG_IHG : Prop :=
  (triangle.similar F G J) (triangle.similar H G I)

-- The lengths of the segments based on problem conditions
def length_HG := 12
def length_JG := 3
def length_IG := 9

-- Area calculation of triangle IJG
def area_IJG := (1/2 * length_IG * length_JG).toReal

-- Final proof statement
theorem area_of_shaded_region :
  similar_triangles_JFG_IHG →
  length_HG = 12 →
  length_JG = length_HG/4 →
  length_IG = length_HG - length_JG →
  real.floor (area_IJG + 0.5) = 14 :=
by
  intros h_sim h_HG h_JG h_IG
  sorry

end area_of_shaded_region_l588_588191


namespace curve_equation_range_of_m_l588_588767

-- Definition of the curve C based on the given conditions
def curve_C (x y : ℝ) : Prop :=
  (y ≥ 0 ∧ x^2 = 4 * y) ∨ (y < 0 ∧ x = 0)

-- Definition of the distance difference condition
def distance_condition (x y : ℝ) : Prop :=
  real.sqrt (x^2 + (y - 1)^2) - abs y = 1

-- Proof problem to state the equivalence of the curve and the condition
theorem curve_equation (x y : ℝ) : distance_condition x y ↔ curve_C x y :=
sorry

-- Definition of the intersection and dot product condition
def dot_product_condition (k m x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = k*x1 + m ∧ y2 = k*x2 + m ∧ x1^2 = 4 * y1 ∧ x2^2 = 4 * y2 ∧
  -(4 * k^2) + (m - 1)^2 - 4 * m < 0

-- Proof problem to state the range for m given the dot product condition
theorem range_of_m (m : ℝ) (h : m > 0) : (∀ k : ℝ, ∃ x1 y1 x2 y2, dot_product_condition k m x1 y1 x2 y2) ↔ 3 - 2 * real.sqrt 2 < m ∧ m < 3 + 2 * real.sqrt 2 :=
sorry

end curve_equation_range_of_m_l588_588767


namespace max_nvst_cells_l588_588681

/--
Given a 10 x 10 grid where each cell is painted either black or white,
we define a cell to be "not in its place" if it has at least seven neighbors of a different color.
Neighbors are cells that share a common side or a common corner.

Theorem: 
The maximum number of cells on the board that can simultaneously be "not in their place" is 26.
-/
theorem max_nvst_cells (grid : Fin 10 → Fin 10 → Bool) : 
  ∃ (S : Fin 10 × Fin 10 → Prop), 
    ( ∀ i j, (S (i, j) → 
              (count_different_neighbors (grid i j) (find_neighbors i j grid) ≥ 7)) ) 
    ∧ (finset.card (finset.filter S finset.univ) = 26) :=
sorry

/--
Helper function: count the number of neighbors of a cell which have a different color.
-/
def count_different_neighbors (color : Bool) (neighbors : List (Fin 10 × Fin 10)) : Nat := 
  neighbors.countp (λ ⟨i, j⟩ => grid i j ≠ color)

/--
Helper function: find all neighbors of a cell at (i, j).
-/
def find_neighbors (i j : Fin 10) (grid : Fin 10 → Fin 10 → Bool) : List (Fin 10 × Fin 10) :=
  let neighbors := [
    (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
    (i, j - 1),             (i, j + 1),
    (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
  ]
  neighbors.filter (λ ⟨x, y⟩ => 0 ≤ x.val ∧ x.val < 10 ∧ 0 ≤ y.val ∧ y.val < 10)

end max_nvst_cells_l588_588681


namespace segment_length_increases_then_decreases_l588_588305

-- Definitions of lengths and points in triangle ABC
variable {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable {AB CD : ℝ}

-- Definition of the movement of the segment from A to D maintaining the properties
def segment_moving_from_A_to_D (AB CD : ℝ) : Prop :=
  ∀ t ∈ [0, 1], let L := (1 - t) * AB + t * CD in
  (t = 0 → L = AB) ∧ (t = 1 → L = CD) ∧
  (t < 1/2 → L < (1/2) * (AB + CD)) ∧
  (t > 1/2 → L > (1/2) * (AB + CD))

-- The main theorem to be proved
theorem segment_length_increases_then_decreases {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] (AB CD : ℝ) :
  AB < CD → segment_moving_from_A_to_D AB CD :=
by
  sorry

end segment_length_increases_then_decreases_l588_588305


namespace star_polygon_net_of_pyramid_l588_588498

theorem star_polygon_net_of_pyramid (R r : ℝ) (h : R > r) : R > 2 * r :=
by
  sorry

end star_polygon_net_of_pyramid_l588_588498


namespace Jake_apples_l588_588063

namespace PeachesAndApples

variable (Steven_apples : ℕ) (Jake_more_apples_than_Steven : ℕ)
variable (Steven_apples_val : Steven_apples = 8)
variable (Jake_more_apples_than_Steven_val : Jake_more_apples_than_Steven = 3)

theorem Jake_apples : (Steven_apples + Jake_more_apples_than_Steven) = 11 :=
by
  rw [Steven_apples_val, Jake_more_apples_than_Steven_val]
  sorry

end PeachesAndApples

end Jake_apples_l588_588063


namespace problem_statement_l588_588663

def is_valid_number (x y: ℕ) : Prop :=
  (2 * x + y) % 3 ≠ 0 ∧ 2 * x + y < 24 ∧ 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9

def count_valid_numbers : ℕ :=
  (List.range 9).bind (λ x, (List.range 10).filter_map (λ y, if is_valid_number x y then some 1 else none)).sum -- this counts valid (x,y) pairs

theorem problem_statement : count_valid_numbers = 72 := sorry

end problem_statement_l588_588663


namespace domain_of_v_l588_588219

def sqrt (x : ℝ) : ℝ := Real.sqrt x
def fourth_root (x : ℝ) : ℝ := x ^ (1/4 : ℝ)

def v (x : ℝ) : ℝ := sqrt (x - 5) + fourth_root (x - 4)

theorem domain_of_v :
  {x : ℝ | (x ≥ 5) ∧ (x ≥ 4)} = {x : ℝ | x ≥ 5} :=
by
  sorry

end domain_of_v_l588_588219


namespace inequality_implies_log_pos_l588_588819

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588819


namespace sum_of_solutions_eq_five_sixths_l588_588225

theorem sum_of_solutions_eq_five_sixths : 
  (∑ x in {x | (4 * x + 6) * (3 * x - 7) = 0}, x) = 5 / 6 := 
by
  sorry

end sum_of_solutions_eq_five_sixths_l588_588225


namespace log_y_minus_x_plus_1_gt_0_l588_588847

theorem log_y_minus_x_plus_1_gt_0 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  Real.log (y - x + 1) > 0 :=
sorry

end log_y_minus_x_plus_1_gt_0_l588_588847


namespace angle_bisector_lemma_l588_588944

theorem angle_bisector_lemma (XYZ : Type) (angle_XYZ : ℝ) (angle_XYW : ℝ) (angle_WYZ : ℝ) :
  angle_XYZ = 40 ∧ angle_XYW = 32 ∧ angle_XYZ = angle_WYZ →
  angle_WYZ = 16 :=
by
  intro h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  sorry

end angle_bisector_lemma_l588_588944


namespace train_traveled_marked_segment_l588_588563

structure Station (Label : Type) :=
  (label : Label)

structure Segment (Label : Type) :=
  (src dst : Station Label)

def traveled_segments {Label : Type} (start end : Station Label) (minutes : ℕ) : Prop :=
  ∃ marked_segment : Segment Label,  -- There exists a marked segment
  ∀ subway_network : list (Segment Label),  -- Considering all segments in the subway network
  train_path : list (Station Label),  -- The path taken by the train
  (train_path.head = start ∧ train_path.last = end ∧  -- The journey from start to end
   length train_path = minutes + 1 ∧  -- Total number of stations visited matches the minutes of travel
   marked_segment ∈ subway_network ∧  -- Marked segment is part of the network
   ∃ i, i + 1 < train_path.length ∧  -- Ensuring index within bounds
   train_path.nth i = marked_segment.src ∧ train_path.nth (i + 1) = marked_segment.dst)  -- Train passing through the marked segment

theorem train_traveled_marked_segment
  {Label : Type} (start end : Station Label) (minutes : ℕ) :
  start ≠ end ∧ minutes = 2016 → traveled_segments start end minutes :=
begin
  sorry,  -- Proof is omitted
end

end train_traveled_marked_segment_l588_588563


namespace first_ring_time_l588_588730

-- Define the properties of the clock
def rings_every_three_hours : Prop := ∀ n : ℕ, 3 * n < 24
def rings_eight_times_a_day : Prop := ∀ n : ℕ, n = 8 → 3 * n = 24

-- The theorem statement
theorem first_ring_time : rings_every_three_hours → rings_eight_times_a_day → (∀ n : ℕ, n = 1 → 3 * n = 3) := 
    sorry

end first_ring_time_l588_588730


namespace area_of_shaded_region_l588_588210

axiom OA : ℝ := 4
axiom OB : ℝ := 16
axiom OC : ℝ := 12
axiom similarity (EA CB : ℝ) : EA / CB = OA / OB

theorem area_of_shaded_region (DE DC : ℝ) (h_DE : DE = OC - EA)
    (h_DC : DC = 12) (h_EA_CB : EA = 3) :
    (1 / 2) * DE * DC = 54 := by
  sorry

end area_of_shaded_region_l588_588210


namespace stickers_left_after_giving_away_l588_588612

/-- Willie starts with 36 stickers and gives 7 to Emily. 
    We want to prove that Willie ends up with 29 stickers. -/
theorem stickers_left_after_giving_away (init_stickers : ℕ) (given_away : ℕ) (end_stickers : ℕ) : 
  init_stickers = 36 ∧ given_away = 7 → end_stickers = init_stickers - given_away → end_stickers = 29 :=
by
  intro h
  sorry

end stickers_left_after_giving_away_l588_588612


namespace segment_length_l588_588576

theorem segment_length (x : ℝ) (h : |x - (27^(1/3))| = 5) : ∃ a b : ℝ, a = 8 ∧ b = -2 ∧ |a - b| = 10 :=
by
  have hx1 : x = 27^(1/3) + 5 ∨ x = 27^(1/3) - 5 := abs_eq hx
  use [8, -2]
  split
  { calc 8 = 27^(1/3) + 5 := by sorry }
  split
  { calc -2 = 27^(1/3) - 5 := by sorry }
  { calc |8 - -2| = |8 + 2| := by sorry }

end segment_length_l588_588576


namespace problem_l588_588816

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588816


namespace number_of_apples_remaining_l588_588008

def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples_before_giving_away : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℕ := total_apples_before_giving_away / 5
def apples_remaining : ℕ := total_apples_before_giving_away - apples_given_to_son

theorem number_of_apples_remaining : apples_remaining = 12 :=
by
  sorry

end number_of_apples_remaining_l588_588008


namespace equivalent_daps_to_dups_l588_588405

theorem equivalent_daps_to_dups :
  (∃ (daps dops dips dups : Type) 
     (eq1 : 5 * dap = 4 * dop)
     (eq2 : 3 * dop = 9 * dip) 
     (eq3 : 2 * dip = 1 * dup), 
    36 * dup = 360 * dap) :=
begin
  sorry
end

end equivalent_daps_to_dups_l588_588405


namespace range_f_on_half_to_one_exists_a_for_g_lt_l588_588384

noncomputable def f (x : ℝ) : ℝ :=
  max (x^2 - 1) (2 * Real.log x)

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  max (x + Real.log x) (a * x^2 + x)

theorem range_f_on_half_to_one :
  Set.range (λ x : ℝ, f x) ∩ Set.Icc (-(3 : ℝ) / 4) 0 = Set.range (λ x : ℝ, f x)
  ∩ Set.Icc (-(3 : ℝ) / 4) 0 :=
sorry

theorem exists_a_for_g_lt :
  ∃ a ∈ Set.Ioc ((Real.log 2 - 1) / 4) 0, ∀ x > 1, g x a < (3 / 2) * x + 4 * a :=
sorry

end range_f_on_half_to_one_exists_a_for_g_lt_l588_588384


namespace semicircle_radius_in_right_triangle_PQR_l588_588431

def right_triangle (P Q R : Type*) [innermost P] [innermost Q] [innermost R] : Prop := 
  ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 15 ∧ b = 8

def semicircle_radius (P Q R : Type*) [innermost P] [innermost Q] [innermost R] : ℝ := sorry

theorem semicircle_radius_in_right_triangle_PQR (P Q R : Type*) [innermost P] [innermost Q] [innermost R] 
  (h : right_triangle P Q R) : semicircle_radius P Q R = 24 / 5 := sorry

end semicircle_radius_in_right_triangle_PQR_l588_588431


namespace avg_height_and_weight_of_class_l588_588420

-- Defining the given conditions
def num_students : ℕ := 70
def num_girls : ℕ := 40
def num_boys : ℕ := 30

def avg_height_30_girls : ℕ := 160
def avg_height_10_girls : ℕ := 156
def avg_height_15_boys_high : ℕ := 170
def avg_height_15_boys_low : ℕ := 160
def avg_weight_girls : ℕ := 55
def avg_weight_boys : ℕ := 60

-- Theorem stating the given question
theorem avg_height_and_weight_of_class :
  ∃ (avg_height avg_weight : ℚ),
    avg_height = (30 * 160 + 10 * 156 + 15 * 170 + 15 * 160) / num_students ∧
    avg_weight = (40 * 55 + 30 * 60) / num_students ∧
    avg_height = 161.57 ∧
    avg_weight = 57.14 :=
by
  -- include the solution steps here if required
  -- examples using appropriate constructs like ring, norm_num, etc.
  sorry

end avg_height_and_weight_of_class_l588_588420


namespace domain_of_myFunction_l588_588559

-- Define the function
def myFunction (x : ℝ) : ℝ := (x + 2) ^ (1 / 2) - (x + 1) ^ 0

-- State the domain constraints as a theorem
theorem domain_of_myFunction (x : ℝ) : 
  (x ≥ -2 ∧ x ≠ -1) →
  ∃ y : ℝ, y = myFunction x := 
sorry

end domain_of_myFunction_l588_588559


namespace greatest_prime_factor_of_power_difference_l588_588220

theorem greatest_prime_factor_of_power_difference (n : ℕ) 
  (h : ∀ p, prime p → p ∣ (4^n - 2^29) → p ≤ 31 ∧ ((p = 31) → (∀ q, prime q → q ∣ (4^n - 2^29) → p = q))) : 
  n = 17 :=
sorry

end greatest_prime_factor_of_power_difference_l588_588220


namespace arithmetic_progression_count_l588_588971

def S := Finset.range 2010

def is_arithmetic_progression (A : Finset ℕ) : Prop :=
  ∃ a b c, a < b ∧ b < c ∧ A = {a, b, c} ∧ b = (a + c) / 2

theorem arithmetic_progression_count :
  (Finset.filter (λ A, is_arithmetic_progression A) (Finset.powersetLen 3 S)).card = 1008016 :=
by sorry

end arithmetic_progression_count_l588_588971


namespace sqrt_domain_l588_588560

theorem sqrt_domain (x : ℝ) : x - 5 ≥ 0 ↔ x ≥ 5 :=
by sorry

end sqrt_domain_l588_588560


namespace angle_cosine_formula_l588_588794

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

noncomputable def angle_between_vectors (a b : V) : ℝ :=
inner_product_space.angle a b

theorem angle_cosine_formula
  (h : a = b + c) :
  real.cos (angle_between_vectors a b) =
    (∥a∥^2 + ∥b∥^2 - ∥c∥^2) / (2 * ∥a∥ * ∥b∥) :=
by sorry

end angle_cosine_formula_l588_588794


namespace smallest_n_div_by_2010_l588_588365

-- Given conditions
variables {α : Type*} [linear_ordered_ring α] 
variable (a : ℕ → α)
axiom a_all_integer : ∀ n, ∃ k : ℤ, a(n) = k
axiom a_2_odd : ∃ k : ℤ, a(2) = k ∧ odd k
axiom a_recurrence : ∀ n, n * (a(n+1) - a(n) + 3) = a(n+1) + a(n) + 3
axiom a_2009_divisible : ∃ k : ℤ, a(2009) = 2010 * k

-- Proof goal
theorem smallest_n_div_by_2010 : ∃ n > 1, a(n) = 2010 * (k : ℤ) ∧ ∀ m > 1, m < n → ¬(a(m) = 2010 * k) :=
by {
  sorry
}

end smallest_n_div_by_2010_l588_588365


namespace hotel_room_count_l588_588648

theorem hotel_room_count {total_lamps lamps_per_room : ℕ} (h_total_lamps : total_lamps = 147) (h_lamps_per_room : lamps_per_room = 7) : total_lamps / lamps_per_room = 21 := by
  -- We will insert this placeholder auto-proof, as the actual arithmetic proof isn't the focus.
  sorry

end hotel_room_count_l588_588648


namespace sin_pi_minus_alpha_l588_588371

noncomputable def alpha : ℝ := sorry -- alpha is to be defined

theorem sin_pi_minus_alpha (hα : 0 < α ∧ α < π/2)
                            (h₁ : sin (α + π/3) = 3/5) :
  sin (π - α) = (3 + 4 * real.sqrt 3) / 10 :=
sorry

end sin_pi_minus_alpha_l588_588371


namespace sum_first_15_odd_integers_l588_588224

theorem sum_first_15_odd_integers : 
  let first_term := 1 in
  let n_terms := 15 in
  let nth_odd (n : ℕ) := 2 * n - 1 in
  let last_term := nth_odd 15 in
  let sum_arithmetic_series (a l : ℕ) (n : ℕ) := n * (a + l) / 2 in
  sum_arithmetic_series first_term last_term n_terms = 225 :=
by
  sorry

end sum_first_15_odd_integers_l588_588224


namespace students_not_enrolled_in_any_section_l588_588165

/-- 
There are 100 students in total, 
37 of whom enrolled in the football section, 
40 of whom enrolled in the swimming section, 
and 15 students enrolled in both sections.
Prove that the number of students who did not enroll in any section is 38.
-/
theorem students_not_enrolled_in_any_section (total : ℕ) (football : ℕ) (swimming : ℕ) (both : ℕ)
  (h_total : total = 100) (h_football : football = 37) (h_swimming : swimming = 40) (h_both : both = 15) :
  (total - (football + swimming - both)) = 38 :=
by 
  rw [h_total, h_football, h_swimming, h_both]
  norm_num

end students_not_enrolled_in_any_section_l588_588165


namespace handshakes_of_9_boys_l588_588406

theorem handshakes_of_9_boys : 
  let n := 9 in let k := 2 in 
  nat.comb n k = 36 :=
by
  sorry

end handshakes_of_9_boys_l588_588406


namespace probability_same_time_alive_l588_588181

-- Define the conditions
def lives_in_last_1000_years (t: ℕ) : Prop := t ≤ 1000

-- Define the lifespan of mathematicians
def lifespan_A := 100
def lifespan_B := 80

-- Define the event living_overlap: probability calculation
noncomputable def living_overlap (born_A born_B : ℕ) : Prop :=
  lives_in_last_1000_years born_A ∧ lives_in_last_1000_years born_B ∧
  (born_B ≤ born_A + lifespan_A ∧ born_A ≤ born_B + lifespan_B)

-- The measurable space we integrate over
def total_area : ℕ := 1000000

-- The proof problem statement
theorem probability_same_time_alive :
  ∃ (overlapping_area : ℝ), 
  (overlapping_area / total_area = real_prob_of_living_overlap) :=
begin
  sorry
end

end probability_same_time_alive_l588_588181


namespace max_possible_value_l588_588469

noncomputable def max_product_of_distances {k : ℕ} (P : ℂ) (A : Fin k → ℂ) : ℝ :=
  ∏ i, complex.abs (P - A i)

theorem max_possible_value
  (k : ℕ) (h : 2 ≤ k) (A : Fin k → ℂ)
  (hA : ∀ i, complex.abs (A i) = 1)  -- vertices are on the unit circle
  (P : ℂ) (hP : complex.abs P ≤ 1)  -- P is on or inside the unit circle
  (hPolygon : ∀ i, A i = complex.exp ((2 * i * real.pi * complex.I) / k))  -- regular k-gon vertices
  : max_product_of_distances P A ≤ 2 :=
sorry

end max_possible_value_l588_588469


namespace sum_binom_cos_eq_l588_588457

theorem sum_binom_cos_eq (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (x : ℝ) :
  (∑ j in Finset.range (a + 1), Nat.choose a j * (2 * Real.cos ((2 * j - a : ℤ) * x))^b) =
  (∑ j in Finset.range (b + 1), Nat.choose b j * (2 * Real.cos ((2 * j - b : ℤ) * x))^a) :=
sorry

end sum_binom_cos_eq_l588_588457


namespace polynomial_has_one_positive_real_solution_l588_588396

-- Define the polynomial
def f (x : ℝ) : ℝ := x ^ 10 + 4 * x ^ 9 + 7 * x ^ 8 + 2023 * x ^ 7 - 2024 * x ^ 6

-- The proof problem statement
theorem polynomial_has_one_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ f x = 0 := by
  sorry

end polynomial_has_one_positive_real_solution_l588_588396


namespace circle_geometry_problem_l588_588263

theorem circle_geometry_problem 
  (O A B C D M : Type)
  [Circ : circle O]
  [Diameter : diameter O A B]
  (C_distinct : C ∈ O ∧ C ≠ A ∧ C ≠ B)
  (CD_perp_AB : ⊥line_segment C AB)
  (Intersects_at_D : ∃ D, intersects_perp C AB D)
  (OM_perp_BC : ⊥line_segment O BC)
  (Intersects_at_M : ∃ M, intersects_perp O BC M)
  (DB_eq_3OM : line_segment_len D B = 3 * line_segment_len O M) :
  angle_measure A B C = 30 := 
begin
  sorry
end

end circle_geometry_problem_l588_588263


namespace cos_angle_MJL_l588_588041

theorem cos_angle_MJL (JKL_angle : ∠J K L = 60)
                      (JNM_angle : ∠J N M = 30) :
  cos (∠M J L) = 5 / 8 := 
by 
  sorry

end cos_angle_MJL_l588_588041


namespace sum_of_trinomials_1_l588_588570

theorem sum_of_trinomials_1 (p q : ℝ) :
  (p + q = 0 ∨ p + q = 8) →
  (2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 2 ∨ 2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 18) :=
by sorry

end sum_of_trinomials_1_l588_588570


namespace simplify_sqrt_expression_l588_588125

theorem simplify_sqrt_expression (a b : ℕ) (h_a : a = 5) (h_b : b = 3) :
  (sqrt (a * b) * sqrt ((b ^ 3) * (a ^ 3)) = 225) :=
by
  rw [h_a, h_b]
  sorry

end simplify_sqrt_expression_l588_588125


namespace PQ_ge_sqrt3_div2_AB_plus_AC_l588_588360

-- Define the conditions of the problem
noncomputable def angle_A_eq_120 (A B C : Type) [normed_space ℝ A]
  (angle_A : angle A B C) : angle_A = 120 := sorry

noncomputable def points_on_sides (A B C K L : Type) [normed_space ℝ A] 
  [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ K] [normed_space ℝ L] 
  (on_AB : ∃ t ∈ segment ℝ A B, K = t) (on_AC : ∃ t ∈ segment ℝ A C, L = t) : 
  true := sorry

noncomputable def equilateral_triangles (B K P C L Q : Type) [normed_space ℝ B]
  [normed_space ℝ K] [normed_space ℝ P] [normed_space ℝ C] [normed_space ℝ L] 
  [normed_space ℝ Q] (equilateral_BKP : equilateral_triangle B K P)
  (equilateral_CLQ : equilateral_triangle C L Q) : true := sorry

-- The theorem to be proven
theorem PQ_ge_sqrt3_div2_AB_plus_AC {A B C K L P Q : Type} [normed_space ℝ A] 
  [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ K] [normed_space ℝ L] 
  [normed_space ℝ P] [normed_space ℝ Q] 
  (h1 : angle_A_eq_120 A B C)
  (h2 : points_on_sides A B C K L)
  (h3 : equilateral_triangles B K P C L Q) :
  dist P Q ≥ (sqrt 3 / 2) * (dist A B + dist A C) := 
sorry

end PQ_ge_sqrt3_div2_AB_plus_AC_l588_588360


namespace problem_l588_588813

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588813


namespace find_coordinates_of_C_l588_588945

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 2, y := 8 }
def M : Point := { x := 4, y := 11 }
def L : Point := { x := 6, y := 6 }

def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def isMidpoint (P A C : Point) : Prop :=
  P = midpoint A C

def coord_C (A M L : Point) : Point :=
  { x := 14, y := 2 }

theorem find_coordinates_of_C :
  isMidpoint M A (coord_C A M L) ∧ L.x = 6 ∧ L.y = 6 :=
by
  sorry

end find_coordinates_of_C_l588_588945


namespace mary_average_speed_l588_588486

-- Define the conditions
def time_uphill : ℕ := 45 -- in minutes
def distance_uphill : ℝ := 1.5 -- in km
def time_downhill : ℕ := 15 -- in minutes
def distance_downhill : ℝ := 1.5 -- in km
def time_school_to_library : ℕ := 15 -- in minutes
def distance_school_to_library : ℝ := 0.5 -- in km

-- Define the computed values to prove
def total_distance : ℝ := distance_uphill + distance_downhill + distance_school_to_library
def total_time_hours : ℝ := (time_uphill + time_downhill + time_school_to_library) / 60
def average_speed : ℝ := 3.5 / 1.25 -- Prove that this equals 2.8 km/hr

theorem mary_average_speed :
  average_speed = 2.8 :=
by
  sorry

end mary_average_speed_l588_588486


namespace polar_equation_of_line_l_and_distance_from_P_l588_588764

noncomputable def polar_equation_line_l {t : ℝ} (x y: ℝ) : Prop :=
  (x = 1/2 * t) ∧ (y = sqrt 3 / 2 * t)

noncomputable def polar_coordinates_point_P : ℝ × ℝ := 
  (2 * sqrt 15 / 3, 2 * π / 3)

theorem polar_equation_of_line_l_and_distance_from_P :
  ∀ {x y: ℝ}, polar_equation_line_l x y →
  let P := polar_coordinates_point_P in
  (θ = π / 3) ∧ (distance P.line_l = sqrt 5) := 
sorry

noncomputable def polar_equation_curve_C (ρ θ: ℝ) : Prop :=
  ρ^2 - 2 * ρ * cos θ - 2 = 0

noncomputable def intersection_points_MN_area :
  ∃ (ρ1 ρ2 : ℝ), 
    let P := polar_coordinates_point_P in
    polar_equation_curve_C ρ1 (π / 3) ∧ polar_equation_curve_C ρ2 (π / 3) ∧
    (ρ1 + ρ2 = 1) ∧ (ρ1 * ρ2 = -2) ∧
    (area_triangle P ρ1 ρ2 = 3 * sqrt 5 / 2) := 
sorry

end polar_equation_of_line_l_and_distance_from_P_l588_588764


namespace percentage_km_contributed_over_judson_l588_588070

theorem percentage_km_contributed_over_judson 
  (judson_contribution : ℝ)
  (kenny_contribution : ℝ)
  (camilo_contribution : ℝ)
  (total_cost : ℝ)
  (h_judson : judson_contribution = 500)
  (h_kenny : kenny_contribution > judson_contribution)
  (h_camilo : camilo_contribution = kenny_contribution + 200)
  (h_total : judson_contribution + kenny_contribution + camilo_contribution = total_cost) :
  (kenny_contribution - judson_contribution) / judson_contribution * 100 = 20 :=
by
  -- We will define the values according to the conditions
  have h_total_cost : total_cost = 1900 := h_total
  -- Define auxiliary values from conditions
  have h_judson_500 : judson_contribution = 500 := h_judson
  have h_kenny_ge_500 : kenny_contribution > 500 := h_kenny
  have ha_camilo_eq : camilo_contribution = kenny_contribution + 200 := h_camilo
  -- Now, need to prove the percentage
  sorry

end percentage_km_contributed_over_judson_l588_588070


namespace triangle_CE_squared_l588_588942

theorem triangle_CE_squared (A B C D E : Point) (h_ABC_acute : acute ∠A B C)
  (h1 : altitude A B C D) (h2 : AD = 4) (h3 : BD = 3) (h4 : CD = 2) 
  (h5 : BE = 5) (h6 : collinear A B E) (h7 : seg A B ⊆ seg A E ∩ seg B E) :
  CE^2 = 120 - 24 * sqrt(5) := by
  sorry

end triangle_CE_squared_l588_588942


namespace absolute_difference_of_integers_l588_588139

theorem absolute_difference_of_integers (x y : ℤ) (h1 : (x + y) / 2 = 15) (h2 : Int.sqrt (x * y) + 6 = 15) : |x - y| = 24 :=
  sorry

end absolute_difference_of_integers_l588_588139


namespace magnitude_conjugate_l588_588374

-- Define the point coordinates and the corresponding complex number
def point_coordinates : ℂ := -3 + 4 * Complex.I

-- State the theorem
theorem magnitude_conjugate (z : ℂ) (h : z = point_coordinates) : Complex.abs (Complex.conj z) = 5 :=
by
  -- The proof is not required hence we put sorry
  sorry

end magnitude_conjugate_l588_588374


namespace parametric_to_ordinary_l588_588154

theorem parametric_to_ordinary (α : ℝ) (x y : ℝ) 
  (h1 : x = sin (α / 2) + cos (α / 2))
  (h2 : y = sqrt (2 + sin α)) :
  y ^ 2 - x ^ 2 = 1 ∧ |x| ≤ sqrt 2 ∧ 1 ≤ y ∧ y ≤ sqrt 3 := 
by
  sorry

end parametric_to_ordinary_l588_588154


namespace persimmons_in_box_l588_588166

-- Defining the conditions
def apples (box : nat) : Prop := box = 3
def persimmons (box : nat) : Prop := box = 2

-- The theorem we need to prove
theorem persimmons_in_box (box : nat) (h1 : apples box) (h2 : persimmons box) : box = 2 :=
sorry

end persimmons_in_box_l588_588166


namespace new_person_weight_l588_588528

theorem new_person_weight (W : ℝ) :
  (∃ (W : ℝ), (390 - W + 70) / 4 = (390 - W) / 4 + 3 ∧ (390 - W + W) = 390) → 
  W = 58 :=
by
  sorry

end new_person_weight_l588_588528


namespace problem1_problem2_l588_588352

variable {m x : ℝ}
def vector_a := (x, -m)
def vector_b := ((m + 1) * x, x)

-- Proof problem (1)
theorem problem1 (h : m > 0) (h_ab : real.sqrt (x^2 + m^2) < real.sqrt (((m + 1)^2 * x^2) + x^2)) : (x < -m / (m + 1)) ∨ (x > m / (m + 1)) :=
sorry

-- Proof problem (2)
theorem problem2 (h_ab_dot : ∀ x : ℝ, (m + 1) * x * x - m * x > 1 - m) : m > 2 * real.sqrt 3 / 3 :=
sorry

end problem1_problem2_l588_588352


namespace women_reseating_l588_588490

-- Define the initial conditions and recursive relation for C
def C : ℕ → ℕ
| 1 => 1
| 2 => 2
| 3 => 6
| (n+1) => 2 * C n + 2 * C (n - 1) + C (n - 2)

theorem women_reseating : C 9 = 3086 := 
sorry

end women_reseating_l588_588490


namespace boat_distance_downstream_l588_588628

variables (speed_boat_still : ℕ) (speed_stream : ℕ) (time_downstream : ℕ) 

theorem boat_distance_downstream 
  (h1 : speed_boat_still = 24) 
  (h2 : speed_stream = 4) 
  (h3 : time_downstream = 2) : 
  let speed_downstream := speed_boat_still + speed_stream in
  let distance_downstream := speed_downstream * time_downstream in
  distance_downstream = 56 :=
by 
  rw [h1, h2, h3],
  have h4 : speed_downstream = 24 + 4 := rfl,
  rw h4,
  have h5 : distance_downstream = 28 * 2 := rfl,
  rw h5,
  exact eq.refl 56

end boat_distance_downstream_l588_588628


namespace emily_subtracts_79_to_get_39_squared_l588_588019

-- Define conditions
def squared_identity (n : ℕ) : ℕ :=
  (40 - n)^2

def computed_identity (n : ℕ) : ℕ :=
  40^2 - 2 * 40 * n + n^2

-- Math proof problem
theorem emily_subtracts_79_to_get_39_squared :
  squared_identity 1 = computed_identity 1 → 39^2 = 40^2 - 79 :=
by 
  intros h
  exact h
  sorry

end emily_subtracts_79_to_get_39_squared_l588_588019


namespace avg_words_per_hour_l588_588638

theorem avg_words_per_hour (words hours : ℝ) (h_words : words = 40000) (h_hours : hours = 80) :
  words / hours = 500 :=
by
  rw [h_words, h_hours]
  norm_num
  done

end avg_words_per_hour_l588_588638


namespace helen_choc_chip_yesterday_l588_588798

variable (total_cookies morning_cookies : ℕ)

theorem helen_choc_chip_yesterday :
  total_cookies = 1081 →
  morning_cookies = 554 →
  total_cookies - morning_cookies = 527 := by
  sorry

end helen_choc_chip_yesterday_l588_588798


namespace range_of_x_coordinate_of_point_P_on_curve_l588_588475

theorem range_of_x_coordinate_of_point_P_on_curve :
  let C (x : ℝ) := x^2 - 2 * x + 3
  let P (x : ℝ) := (x, C x)
  let slope_angle_range := (0 : ℝ) .. (Real.pi / 4)
  let slope_at_P (x : ℝ) := 2 * x - 2
  ∀ x, x ∈ [1, 3/2] ↔ slope_at_P x ∈ [0, 1] :=
by sorry

end range_of_x_coordinate_of_point_P_on_curve_l588_588475


namespace rounding_error_bounded_l588_588509

theorem rounding_error_bounded (n : ℕ) (a : Fin n → ℝ) (ha : ∀ i, a i > 0) :
  ∃ (r : Fin n → ℤ), (∑ i, (a i) - ∑ i, (r i : ℝ)) ≤ (n + 1) / 4 :=
by
  sorry

end rounding_error_bounded_l588_588509


namespace area_of_triangle_ABD_l588_588936

variables (AB CD : ℝ) (areaABCD : ℝ)

def is_parallelogram (ABCD : ℝ) := ABCD > 0
def twice_length (CD AB : ℝ) := CD = 2 * AB
def area_parallelogram := areaABCD = 24

theorem area_of_triangle_ABD 
  (h1 : is_parallelogram areaABCD)
  (h2 : twice_length CD AB)
  (h3 : area_parallelogram) :
  let A₁ := 8 
  in A₁ = areaABCD / 3 := 
  by
  -- proof to be provided
  sorry

end area_of_triangle_ABD_l588_588936


namespace tens_digit_of_factorial_difference_l588_588227

def factorial (n : ℕ) : ℕ :=
  (finset.range (n+1)).prod id

def modulo (a b : ℕ) : ℕ :=
  a % b

theorem tens_digit_of_factorial_difference :
  let thirteen_factorial := factorial 13
  let eighteen_factorial := factorial 18
  (modulo (eighteen_factorial - thirteen_factorial) 100) / 10 % 10 = 8 :=
by
  sorry

end tens_digit_of_factorial_difference_l588_588227


namespace polynomial_coeff_properties_l588_588353

theorem polynomial_coeff_properties :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 : ℤ,
  (∀ x : ℤ, (1 - 2 * x)^7 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) ∧
  a0 = 1 ∧
  (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = -1) ∧
  (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3^7)) :=
sorry

end polynomial_coeff_properties_l588_588353


namespace part_one_part_two_l588_588736

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part_one (x : ℝ) : f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
begin
  sorry
end

theorem part_two {a b : ℝ} (h_cond1 : 0 < b) (h_cond2 : b < 1/2) (h_cond3 : 1/2 < a)
  (h_eq : f a = 3 * f b) : ∃ m : ℤ, a^2 + b^2 > m ∧ m = 2 :=
begin
  sorry
end

end part_one_part_two_l588_588736


namespace find_p_l588_588627

-- Define the main variables and their types
variables {p : ℝ}

-- Define the probability equation given the conditions
def binomial_probability (p : ℝ) : ℝ :=
  28 * p^6 * (1-p)^2

-- Define the target probability value
def target_probability : ℝ := 125 / 256

-- Theorem stating the relationship between p and target_probability
theorem find_p : (binomial_probability p = target_probability) → p = 1 / 2 ∨ p = 1 / 4 ∨ p = 1 / 3 ∨ p = 1 / 8 ∨ p = 2 / 3 :=
by
  sorry

end find_p_l588_588627


namespace votes_difference_l588_588253

theorem votes_difference (total_votes john_votes : ℕ) (james_percentage : ℝ)
  (h_total_votes : total_votes = 1150) 
  (h_john_votes : john_votes = 150)
  (h_james_percentage : james_percentage = 0.7) :
  let remaining_votes := total_votes - john_votes in
  let james_votes := remaining_votes * james_percentage in
  let third_candidate_votes := remaining_votes - james_votes in
  third_candidate_votes - john_votes = 150 :=
by
  sorry

end votes_difference_l588_588253


namespace inequality_ln_pos_l588_588833

theorem inequality_ln_pos 
  (x y : ℝ) 
  (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
sorry

end inequality_ln_pos_l588_588833


namespace problem_l588_588809

-- Given condition: 2^x - 2^y < 3^(-x) - 3^(-y)
def inequality (x y : ℝ) : Prop := 2^x - 2^y < 3^(-x) - 3^(-y)

-- Statement to prove: ln(y - x + 1) > 0
theorem problem (x y : ℝ) (h : inequality x y) : Real.ln (y - x + 1) > 0 := 
sorry

end problem_l588_588809


namespace num_of_lines_through_3_points_eq_365_l588_588100

-- Define the ranges and properties of our marked points
def points : List (ℕ × ℕ) :=
  [(x, y) | x <- [0, 1, 2], y <- [0, 1, 2, ..., 26]]

-- Helper function for even check
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Helper function for odd check
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- Main theorem to be proven
theorem num_of_lines_through_3_points_eq_365 :
  ∃! l, ∃ a b c : ℕ,
  0 ≤ a ∧ a ≤ 26 ∧ 0 ≤ b ∧ b ≤ 26 ∧ 0 ≤ c ∧ c ≤ 26 ∧
  (is_even a ∧ is_even c ∨ is_odd a ∧ is_odd c) ∧
  b = (a + c) / 2 ∧
  l = 365 := sorry

end num_of_lines_through_3_points_eq_365_l588_588100


namespace smallest_solution_of_abs_eq_l588_588705

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end smallest_solution_of_abs_eq_l588_588705


namespace area_of_CDE_is_54_l588_588212

-- Define points O, A, B, C, D, and E using coordinates.
def point := (ℝ × ℝ)
def O : point := (0, 0)
def A : point := (4, 0)
def B : point := (16, 0)
def C : point := (16, 12)
def D : point := (4, 12)
def E : point := (4, 3)  -- Midpoint derived from problem's similarity conditions

-- Define the line segment lengths based on the points.
def length (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Define DE and DC
noncomputable def DE : ℝ := 9
noncomputable def DC : ℝ := 12

-- Area of triangle CDE
noncomputable def area_CDE : ℝ := (1 / 2) * DE * DC

-- Theorem statement
theorem area_of_CDE_is_54 : area_CDE = 54 := by
  sorry

end area_of_CDE_is_54_l588_588212


namespace time_between_2_and_3_right_angle_l588_588654

noncomputable def time_of_right_angle_between_hands (h m : ℝ) (a b : ℝ) (moved_minutes : ℝ) : Prop :=
  abs (a * moved_minutes - (h + b * moved_minutes )) = 90

theorem time_between_2_and_3_right_angle :
  time_of_right_angle_between_hands 60 0.5 6 300 / 11 :=
sorry

end time_between_2_and_3_right_angle_l588_588654


namespace hyperbola_eccentricity_l588_588791

/-- Given the hyperbola x^2 / a^2 - y^2 / b^2 = 1 (a > 0, b > 0), 
the parabola y^2 = 2p * x (p > 0), 
the intersection points A and B, 
the line passing through A and B passing through the focus of the parabola, 
and the length of the segment AB equalling the length of the conjugate axis of the hyperbola, 
prove that the eccentricity of the hyperbola is 3. -/
theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
  (eqn_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (eqn_parabola : ∀ x y, y^2 = 2 * p * x)
  (intersection_points : (x₁ y₁ x₂ y₂ : ℝ)
    (intersect_hyperbola : eqn_hyperbola x₁ y₁ ∧ eqn_hyperbola x₂ y₂)
    (intersect_parabola : eqn_parabola x₁ y₁ ∧ eqn_parabola x₂ y₂))
  (line_through_focus : (focus : ℝ) (hf : focus = p / 2)
    (line_AB_through_focus : ∃ m₁ m₂ : ℝ, y₂ - y₁ = m₁ * (x₂ - x₁) ∧ focus ∈ line_AB_through_focus))
  (length_AB : abs ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2) = 2 * b) :
  (let c := sqrt (a^2 + b^2) in c / a = 3) :=
by
  sorry

end hyperbola_eccentricity_l588_588791


namespace simplify_sum_of_squares_roots_l588_588664

theorem simplify_sum_of_squares_roots :
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 :=
by
  sorry

end simplify_sum_of_squares_roots_l588_588664


namespace count_lines_through_points_l588_588102

-- Define the conditions
def points_on_plane : list (ℤ × ℤ) :=
  [(x, y) | x in [0, 1, 2], y in list.range 27]

-- Definition for the line through exactly three points
def line_through_three_points (a b c : ℤ) : bool :=
  (2 * b = a + c)

#eval (let e := (list.range 14).map (λ n, 2 * n) in
       let o := (list.range 13).map (λ n, 2 * n + 1) in
       2 * (e.length * e.length + o.length * o.length))

def num_lines_through_three_points : ℕ :=
  (let even_vals := list.range' 0 14).map (λ n, 2 * n) in 
  let odd_vals := list.range' 0 13).map (λ n, 2 * n + 1) in 
  2 * (even_vals.length * even_vals.length + odd_vals.length * odd_vals.length)

theorem count_lines_through_points : num_lines_through_three_points = 365 :=
sorry

end count_lines_through_points_l588_588102


namespace simplify_expression_l588_588513

theorem simplify_expression : 5 * (18 / -9) * (24 / 36) = -(20 / 3) :=
by
  sorry

end simplify_expression_l588_588513


namespace smallest_solution_l588_588709

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end smallest_solution_l588_588709


namespace additional_interest_rate_correct_l588_588520

-- Define the given conditions
def P : ℝ := 8000
def A : ℝ := 10200
def T : ℝ := 3
def A' : ℝ := 10680

-- Additional interest earned
def interestAdditional := A' - A

-- Necessary values calculated from the definitions
def new_amount_with_interest := P * (1 + (T * (9.166 / 100)))
def Expected_A_new := 10200 + interestAdditional

theorem additional_interest_rate_correct :
  let additional_interest_rate := 2 in
  Expected_A_new = P * (1 + (T * ((9.166 + additional_interest_rate) / 100))) :=
by
  sorry

end additional_interest_rate_correct_l588_588520


namespace b_geometric_and_a_formula_sum_nb_n_l588_588977

open BigOperators

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Condition 1: Sequence sum formula
axiom Sum_S : ∀ n : ℕ, S n = 2 * a n - 3 * n

-- Condition 2: Definition of sequence b_n
def b_n (n : ℕ) : ℕ := a n + 3

-- Part 1: Prove b_n is a geometric sequence with a common ratio of 2 and find a_n
theorem b_geometric_and_a_formula : 
  (∀ n : ℕ, b_n (n + 1) = 2 * b_n n) ∧ (∀ n : ℕ, a n = 3 * 2 ^ n - 3) :=
by 
  sorry

-- Part 2: Sum of the first n terms of sequence {n * b_n}
theorem sum_nb_n :
  ∑ k in Finset.range (n + 1), k * b_n k = 3 * (n - 1) * 2 ^ (n + 1) + 6 :=
by 
  sorry

end b_geometric_and_a_formula_sum_nb_n_l588_588977


namespace eccentricity_of_ellipse_l588_588375

theorem eccentricity_of_ellipse (p a b c e : ℝ) (P Q : ℝ × ℝ) (h_positive_p : p > 0)
  (h_focus_parabola : P = (2 * c, c))
  (h_focus_ellipse : Q = (2 * c, c))
  (h_common_points : (P.1^2 = 2 * p * P.2) ∧ ((P.1^2 / b^2) + (P.2^2 / a^2) = 1) ∧ 
                      (Q.1^2 = 2 * p * Q.2) ∧ ((Q.1^2 / b^2) + (Q.2^2 / a^2) = 1))
  (h_focus_shared : p = 2 * c)
  (h_eccentricity : e = Math.sqrt (a^2 - b^2) / a)
  (h_solve_eccentricity : e = Math.sqrt 2 - 1) :
  e = Math.sqrt 2 - 1 :=
sorry

end eccentricity_of_ellipse_l588_588375


namespace number_of_lines_through_three_points_l588_588095

theorem number_of_lines_through_three_points : 
  (∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 26) →
  (∃ (count : ℕ), count = 365) :=
begin
  sorry
end

end number_of_lines_through_three_points_l588_588095


namespace area_of_triangle_l588_588251

-- Define the lines in terms of functions
def line1 (x : ℝ) : ℝ := 6
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define the points of intersection based on solving the equations of lines
def point1 : ℝ × ℝ := (4, line1 4)
def point2 : ℝ × ℝ := (-4, line1 (-4))
def point3 : ℝ × ℝ := (0, line2 0)

-- Translate problem to Lean proof statement
theorem area_of_triangle :
  let v1 := (4, 6)
  let v2 := (-4, 6)
  let v3 := (0, 2)
  let area := 1 / 2 * abs ((fst v1 * snd v2 + fst v2 * snd v3 + fst v3 * snd v1) -
                           (snd v1 * fst v2 + snd v2 * fst v3 + snd v3 * fst v1))
  area = 16 :=
sorry

end area_of_triangle_l588_588251


namespace even_integer_14_correct_l588_588342

def f (m : ℕ) := if m % 2 = 0 then ∏ i in (Finset.range ((m / 2) + 1)).filter (λ x, x > 0), 2 * i else 0

theorem even_integer_14_correct :
  (∀ m : ℕ, m % 2 = 0 ∧ 0 < m ∧ (∀ p : ℕ, prime p → (∀ k : ℕ, k ≤ m/2 → 2*k ≠ p) → p ≤ 13) → m ≤ 14) :=
sorry

end even_integer_14_correct_l588_588342


namespace gcd_m_n_l588_588079

noncomputable def m : ℕ := 5 * 11111111
noncomputable def n : ℕ := 111111111

theorem gcd_m_n : gcd m n = 11111111 := by
  sorry

end gcd_m_n_l588_588079


namespace company_bought_14_02_tons_l588_588266

noncomputable def gravel := 5.91
noncomputable def sand := 8.11
noncomputable def total_material := gravel + sand

theorem company_bought_14_02_tons : total_material = 14.02 :=
by 
  sorry

end company_bought_14_02_tons_l588_588266


namespace sampling_methods_used_l588_588039

-- Definitions based on problem conditions
def TotalHouseholds : Nat := 2000
def FarmerHouseholds : Nat := 1800
def WorkerHouseholds : Nat := 100
def IntellectualHouseholds : Nat := TotalHouseholds - FarmerHouseholds - WorkerHouseholds
def SampleSize : Nat := 40

-- The statement of the proof problem
theorem sampling_methods_used
  (N : Nat := TotalHouseholds)
  (F : Nat := FarmerHouseholds)
  (W : Nat := WorkerHouseholds)
  (I : Nat := IntellectualHouseholds)
  (S : Nat := SampleSize)
:
  (1 ∈ [1, 2, 3]) ∧ (2 ∈ [1, 2, 3]) ∧ (3 ∈ [1, 2, 3]) :=
by
  -- Add the proof here
  sorry

end sampling_methods_used_l588_588039


namespace intersection_proof_l588_588765

theorem intersection_proof :
  (∀ (θ ρ : ℝ), ρ * cos (θ + π / 6) = (sqrt 3 - 1) / 2 -> (sqrt 3 * ρ * cos θ - ρ * sin θ - sqrt 3 + 1 = 0)) ∧
  (∀ (θ ρ : ℝ), ρ * (1 - cos θ ^ 2) - 2 * cos θ = 0 -> (ρ * sin θ)^2 = 2 * ρ * cos θ) ∧
  (∀ t : ℝ, 3 * t ^ 2 - 4 * t - 16 = 0 -> |-sqrt(3)*(2+1/2*t)-(sqrt(3)/2*t)| ^ 2 + |sqrt(3)*(2+1/2*t)-(sqrt(3)/2*t)| ^ 2 = (2+1/2*t)^2 + (1/2*t)^2 - 4 * (2+1/2*t) * (1/2*t) = 112/9) :=
begin
  sorry
end

end intersection_proof_l588_588765


namespace total_colored_hangers_l588_588030

theorem total_colored_hangers (pink green : ℕ) :
  pink = 7 →
  green = 4 →
  let blue := green - 1 in
  let yellow := blue - 1 in
  pink + green + blue + yellow = 16 :=
by
  intros hp hg
  let blue := green - 1
  let yellow := blue - 1
  sorry

end total_colored_hangers_l588_588030


namespace car_rental_daily_rate_l588_588531

theorem car_rental_daily_rate :
  ∃ (x : ℕ), 
    let weekly_rate := 190 in
    let total_days := 11 in
    let total_cost := 310 in
    let additional_days := total_days - 7 in
    total_cost = weekly_rate + (x * additional_days) ∧ x = 30 :=
by
  let weekly_rate := 190
  let total_days := 11
  let total_cost := 310
  let additional_days := total_days - 7
  exists 30
  simp
  sorry

end car_rental_daily_rate_l588_588531


namespace values_of_a_l588_588777

def f : ℝ → ℝ :=
λ x, if x ≤ 0 then -x else x^2

theorem values_of_a (a : ℝ) (h : f a = 4) :
  a = -4 ∨ a = 2 :=
sorry

end values_of_a_l588_588777


namespace hyperbola_locus_l588_588540

noncomputable section

def is_hyperbola_locus (F1 F2 : Point) (d : ℝ) : Prop :=
  ∀ (P : Point), abs (dist P F1 - dist P F2) = d

theorem hyperbola_locus (F1 F2 : Point) (d : ℝ) (hF : dist F1 F2 > d) :
  is_hyperbola_locus F1 F2 d ↔ is_hyperbola F1 F2 d :=
by
  sorry

end hyperbola_locus_l588_588540


namespace part_a_part_b_l588_588304

-- Define the system of equations
def system_of_equations (x y z p : ℝ) :=
  x^2 - 3 * y + p = z ∧ y^2 - 3 * z + p = x ∧ z^2 - 3 * x + p = y

-- Part (a) proof problem statement
theorem part_a (p : ℝ) (hp : p ≥ 4) :
  (p > 4 → ¬ ∃ (x y z : ℝ), system_of_equations x y z p) ∧
  (p = 4 → ∀ (x y z : ℝ), system_of_equations x y z 4 → x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

-- Part (b) proof problem statement
theorem part_b (p : ℝ) (hp : 1 < p ∧ p < 4) :
  ∀ (x y z : ℝ), system_of_equations x y z p → x = y ∧ y = z :=
by sorry

end part_a_part_b_l588_588304


namespace shaded_area_is_54_l588_588205

-- Define the coordinates of points O, A, B, C, D, E
structure Point where
  x : ℝ
  y : ℝ

-- Given points
def O := Point.mk 0 0
def A := Point.mk 4 0
def B := Point.mk 16 0
def C := Point.mk 16 12
def D := Point.mk 4 12
def E := Point.mk 4 3

-- Define the function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2) ^ (1/2)

-- Define similarity of triangles and calculate side lengths involved
def triangles_similarity (OA OB CB EA : ℝ) : Prop :=
  OA / OB = EA / CB

-- Define the condition
def condition : Prop := 
  triangles_similarity (distance O A) (distance O B) 12 (distance E A) ∧
  distance E A = 3 ∧
  distance D E = 9

-- Define the calculation of area of triangle given base and height
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- State that the area of triangle CDE is 54 cm²
def area_shaded_region : Prop :=
  triangle_area 9 12 = 54

-- Main theorem statement
theorem shaded_area_is_54 : condition → area_shaded_region := by
  sorry

end shaded_area_is_54_l588_588205


namespace ratio_odd_even_divisors_l588_588973

-- Definitions based on the given problem conditions
def N : ℕ := 25 * 46 * 75 * 126

-- The statement we want to prove: the ratio of the sum of odd divisors to the sum of even divisors is 1:6
theorem ratio_odd_even_divisors (N = 25 * 46 * 75 * 126) :
  let sum_odd_divisors := (1 + 3 + 3^2 + 3^3) * (1 + 5 + 5^2 + 5^3 + 5^4) * (1 + 7) * (1 + 23),
      sum_all_divisors := (1 + 2 + 4) * sum_odd_divisors in
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors in
  sum_odd_divisors / sum_even_divisors = 1 / 6 := 
  sorry  -- Proof to be provided

end ratio_odd_even_divisors_l588_588973


namespace charles_whistles_l588_588118

theorem charles_whistles (S : ℕ) (C : ℕ) (h1 : S = 223) (h2 : S = C + 95) : C = 128 :=
by
  sorry

end charles_whistles_l588_588118


namespace line_relationship_parallel_or_skew_l588_588372

variables {Point Line Plane : Type}
variables (in_plane : Point → Plane → Prop) (on_line : Point → Line → Prop)
variables (parallel_line_plane : Line → Plane → Prop) (subset_line_plane : Line → Plane → Prop)
variables (a b : Line) (α : Plane)

-- Given conditions:
-- Line a is parallel to plane α
-- Line b is a subset of plane α
def parallel_to_plane := parallel_line_plane a α
def subset_of_plane := subset_line_plane b α

-- The positional relationship between line a and line b can be:
-- either parallel
-- or skew (which means no common points and not parallel)
def positional_relationship : Prop := 
  (∃ (parallel_lines : Line → Line → Prop), parallel_lines a b) ∨ 
  (¬(∃ (common_point : Point), on_line common_point a ∧ on_line common_point b) ∧ ¬(∃ (parallel_lines : Line → Line → Prop), parallel_lines a b))

-- The theorem statement
theorem line_relationship_parallel_or_skew 
  (h₁: parallel_to_plane) 
  (h₂: subset_of_plane) : positional_relationship :=
sorry

end line_relationship_parallel_or_skew_l588_588372


namespace log_of_y_sub_x_plus_one_positive_l588_588902

theorem log_of_y_sub_x_plus_one_positive (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  ln (y - x + 1) > 0 := 
by 
  sorry

end log_of_y_sub_x_plus_one_positive_l588_588902


namespace parallelogram_is_most_analogous_l588_588610

/-- 
  A parallelepiped is a three-dimensional figure formed by translating a parallelogram.
  We need to show that the figure most analogous to a parallelepiped, given the choices, is the Parallelogram.
--/
def analogous_to_parallelepiped (figure : Type) : Prop :=
  figure = D

constant Triangle : Type
constant Trapezoid : Type
constant Rectangle : Type
constant Parallelogram : Type

theorem parallelogram_is_most_analogous :
  analogous_to_parallelepiped Parallelogram :=
by
  sorry

end parallelogram_is_most_analogous_l588_588610


namespace solutions_in_nat_solutions_in_non_neg_int_l588_588805

-- Definitions for Part A
def nat_sol_count (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem solutions_in_nat (x1 x2 x3 : ℕ) : 
  (x1 > 0) → (x2 > 0) → (x3 > 0) → (x1 + x2 + x3 = 1000) → 
  nat_sol_count 997 3 = Nat.choose 999 2 := sorry

-- Definitions for Part B
theorem solutions_in_non_neg_int (x1 x2 x3 : ℕ) : 
  (x1 + x2 + x3 = 1000) → 
  nat_sol_count 1000 3 = Nat.choose 1002 2 := sorry

end solutions_in_nat_solutions_in_non_neg_int_l588_588805


namespace inequality_implies_log_pos_l588_588824

noncomputable def f (x : ℝ) : ℝ := 2^x - 3^(-x)

theorem inequality_implies_log_pos {x y : ℝ} (h : f(x) < f(y)) :
  log (y - x + 1) > 0 :=
by
  sorry

end inequality_implies_log_pos_l588_588824


namespace angle_BQC_is_90_l588_588439

-- Definitions (conditions)
variable (A B C P Q : Type) 
variable [is_triangle A B C] 
variable (H1 : ∠BAC = 40)
variable (H2 : ∠ABC = 70)
variable (H3 : on_circle_diameter P A) -- Circle with diameter AP
variable (H4 : intersects BP Q) -- BP intersects Q

-- Goal (question assertion)
theorem angle_BQC_is_90 
  (H1 : ∠BAC = 40) 
  (H2 : ∠ABC = 70) 
  (H3 : on_circle_diameter P A) 
  (H4 : intersects BP Q) 
  : ∠BQC = 90 :=
sorry

end angle_BQC_is_90_l588_588439


namespace DFGE_cyclic_l588_588461

open EuclideanGeometry

variable 
  (Γ : Circle)
  (B C D E F G : Point) -- Points B, C, D, E, F, G are on the circle
  (A : Point) -- A is the midpoint of the arc BC not containing B, C
  (h_mid_arc : is_midpoint_arc Γ A B C)
  (h_chord_AD : is_chord Γ A D)
  (h_chord_AE : is_chord Γ A E)
  (h_intersect_F : segment_intersect (A, D) (B, C) F)
  (h_intersect_G : segment_intersect (A, E) (B, C) G)

theorem DFGE_cyclic :
  cyclic_quadrilateral D F G E := by
  sorry -- Proof to be filled in

end DFGE_cyclic_l588_588461


namespace cubic_roots_product_l588_588774

theorem cubic_roots_product (x1 x2 x3 : ℝ) 
  (h1 : x1 + x2 + x3 = 1) 
  (h2 : x1 * x2 + x2 * x3 + x3 * x1 = -5) 
  (h3 : x1 * x2 * x3 = 1) 
  (h4 : ∀ x, (x - x1) * (x - x2) * (x - x3) = x^3 - x^2 - 5x - 1) :
  (x1^2 - 4 * x1 * x2 + x2^2) * 
  (x2^2 - 4 * x2 * x3 + x3^2) * 
  (x3^2 - 4 * x3 * x1 + x1^2) = 444 := 
by 
  sorry

end cubic_roots_product_l588_588774


namespace adam_and_simon_distance_75_miles_l588_588646

noncomputable def adam_biking_velocity := 10 / Real.sqrt 2 -- mph on both north and east direction
noncomputable def simon_biking_velocity := 7 -- mph south direction
noncomputable def displacement_adam (t : ℝ) := 5 * Real.sqrt 2 * t -- displacement south
def displacement_simon (t : ℝ) := simon_biking_velocity * t -- displacement south
def total_displacement (t : ℝ) := Real.sqrt ((displacement_adam t)^2 + (displacement_simon t)^2)
def time_to_reach_distance (d : ℝ) (t : ℝ) : Prop := total_displacement t = d

theorem adam_and_simon_distance_75_miles : ∃ t : ℝ, time_to_reach_distance 75 t ∧ t ≈ 7.54 :=
by
  sorry

end adam_and_simon_distance_75_miles_l588_588646


namespace ratio_m_M_l588_588456

open Real

variables {M m : ℝ}
variables (triangle_eq : is_equilateral_triangle ABC M)
variables (points_ab : set_eq_point_pairs [(E_1, AE_1, AB), (E_2, BE_2, AB)])
variables (points_bc : set_eq_point_pairs [(F_1, BF_1, BC), (F_2, CF_2, BC)])
variables (points_ac : set_eq_point_pairs [(G_1, CG_1, AC), (G_2, AG_2, AC)])
variable (m_eq : AE_1 = BE_2 = BF_1 = CF_2 = CG_1 = AG_2 = m)
variable (area_eq :
  area_hexagon E_1E_2F_1F_2G_1G_2 =
  area_triangle AE_1G_2 + area_triangle BF_1E_2 + area_triangle CG_1F_2)

theorem ratio_m_M :
  m / M = 1 / sqrt 6 :=
sorry

end ratio_m_M_l588_588456


namespace segment_length_l588_588575

theorem segment_length (x : ℝ) (h : |x - (27^(1/3))| = 5) : ∃ a b : ℝ, a = 8 ∧ b = -2 ∧ |a - b| = 10 :=
by
  have hx1 : x = 27^(1/3) + 5 ∨ x = 27^(1/3) - 5 := abs_eq hx
  use [8, -2]
  split
  { calc 8 = 27^(1/3) + 5 := by sorry }
  split
  { calc -2 = 27^(1/3) - 5 := by sorry }
  { calc |8 - -2| = |8 + 2| := by sorry }

end segment_length_l588_588575


namespace log_gt_suff_but_not_necc_for_cube_gt_l588_588607

theorem log_gt_suff_but_not_necc_for_cube_gt (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (ln x > ln y → x^3 > y^3) ∧ (¬(x^3 > y^3 → ln x > ln y)) :=
by
  sorry

end log_gt_suff_but_not_necc_for_cube_gt_l588_588607


namespace coin_collection_l588_588932

def initial_ratio (G S : ℕ) : Prop := G = S / 3
def new_ratio (G S : ℕ) (addedG : ℕ) : Prop := G + addedG = S / 2
def total_coins_after (G S addedG : ℕ) : ℕ := G + addedG + S

theorem coin_collection (G S : ℕ) (addedG : ℕ) 
  (h1 : initial_ratio G S) 
  (h2 : addedG = 15) 
  (h3 : new_ratio G S addedG) : 
  total_coins_after G S addedG = 135 := 
by {
  sorry
}

end coin_collection_l588_588932


namespace log_of_fraction_less_than_one_l588_588016

theorem log_of_fraction_less_than_one (a : ℝ) (h : 0 < a) : log a (2 / 5) < 1 ↔ (0 < a ∧ a < 2 / 5) ∨ a > 1 :=
  sorry

end log_of_fraction_less_than_one_l588_588016


namespace average_fish_per_person_l588_588913

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l588_588913


namespace profit_percent_calc_l588_588922

theorem profit_percent_calc (SP CP : ℝ) (h : CP = 0.25 * SP) : (SP - CP) / CP * 100 = 300 :=
by
  sorry

end profit_percent_calc_l588_588922


namespace apples_left_is_correct_l588_588010

-- Definitions for the conditions
def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℚ := 1 / 5 * total_apples
def apples_left : ℚ := total_apples - apples_given_to_son

-- The main statement to be proven
theorem apples_left_is_correct : apples_left = 12 := by
  sorry

end apples_left_is_correct_l588_588010


namespace apples_left_is_correct_l588_588011

-- Definitions for the conditions
def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℚ := 1 / 5 * total_apples
def apples_left : ℚ := total_apples - apples_given_to_son

-- The main statement to be proven
theorem apples_left_is_correct : apples_left = 12 := by
  sorry

end apples_left_is_correct_l588_588011


namespace log_identity_solution_l588_588908

theorem log_identity_solution (x : ℝ) : (log 3 x) * (log 4 3) = 4 -> x = 256 :=
by sorry

end log_identity_solution_l588_588908


namespace find_cd_sum_l588_588086

theorem find_cd_sum (x y z c d : ℝ) :
  (log 10 (x + y) = z) →
  (log 10 (x^2 + y^2) = z - 1) →
  (∀ x y z, x^5 + y^5 = c * 10^(5 * z) + d * 10^(3 * z)) →
  (c + d = 24) :=
by
  intros h1 h2 h3
  sorry

end find_cd_sum_l588_588086


namespace find_AC_length_l588_588929

open Real

-- Conditions
variables (O A D C B : Point)
variables (r : ℝ) (BO_len : ℝ) (angle_ABO : ℝ)
variables (dir_ABO : Angle O B A)

noncomputable def is_circle (O : Point) (A : Point) (r : ℝ) : Bool :=
  dist O A = r

noncomputable def is_diameter (O : Point) (A D : Point) : Bool :=
  dist A D = 2 * dist O A

noncomputable def is_chord (A C : Point) : Bool :=
  ∃ P : Point, is_circle O P r ∧ is_diameter O A D ∧ is_circle O C r

-- Specific angles & distances given
noncomputable def specific_conditions : Prop :=
  BO_len = 8 ∧
  angle_ABO = 45 ∧
  dir_ABO = 45

-- Proof statement
theorem find_AC_length
  (h_circle : is_circle O A r)
  (h_diameter : is_diameter O A D)
  (h_chord : is_chord A C)
  (h_BO_len : BO_len = 8)
  (h_angle_ABO : angle_ABO = 45)
  (h_dir_ABO : dir_ABO = 45)
  : dist A C = 8 :=
sorry

end find_AC_length_l588_588929


namespace dina_dolls_l588_588318

theorem dina_dolls (Ivy_collectors: ℕ) (h1: Ivy_collectors = 20) (h2: ∀ y : ℕ, 2 * y / 3 = Ivy_collectors) :
  ∃ x : ℕ, 2 * x = 60 :=
  sorry

end dina_dolls_l588_588318


namespace age_difference_between_brother_and_cousin_is_five_l588_588480

variable (Lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ)

-- Conditions
axiom lexie_age_def : Lexie_age = 8
axiom grandma_age_def : grandma_age = 68
axiom lexie_brother_condition : Lexie_age = brother_age + 6
axiom lexie_sister_condition : sister_age = 2 * Lexie_age
axiom uncle_grandma_condition : uncle_age = grandma_age - 12
axiom cousin_brother_condition : cousin_age = brother_age + 5

-- Goal
theorem age_difference_between_brother_and_cousin_is_five : 
  Lexie_age = 8 → grandma_age = 68 → brother_age = Lexie_age - 6 → cousin_age = brother_age + 5 → cousin_age - brother_age = 5 :=
by sorry

end age_difference_between_brother_and_cousin_is_five_l588_588480


namespace det_aA_bB_eq_zero_l588_588453

-- Definition for 3x3 zero matrix
def O3 : Matrix (Fin 3) (Fin 3) ℝ := 0

-- Assumptions
variables (A B : Matrix (Fin 3) (Fin 3) ℝ)
variable (ha : ∀ a b : ℝ, det (a • A + b • B) = 0)
variable (hconds : A * A + B * B = O3)

-- The proof statement
theorem det_aA_bB_eq_zero (a b : ℝ) : det (a • A + b • B) = 0 := by
  sorry

end det_aA_bB_eq_zero_l588_588453


namespace monthly_average_decrease_rate_l588_588926

-- Conditions
def january_production : Float := 1.6 * 10^6
def march_production : Float := 0.9 * 10^6
def rate_decrease : Float := 0.25

-- Proof Statement: we need to prove that the monthly average decrease rate x = 0.25 satisfies the given condition
theorem monthly_average_decrease_rate :
  january_production * (1 - rate_decrease) * (1 - rate_decrease) = march_production := by
  sorry

end monthly_average_decrease_rate_l588_588926


namespace max_m_value_l588_588435

theorem max_m_value {x y : ℝ} (hA : (3, 3) ∈ [{p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}])
  (crease1 : ∀ {x y : ℝ}, x - y + 1 = 0 → (3, 3) = (x, y))
  (crease2 : ∀ {x y : ℝ}, x + y - 7 = 0 → (3, 3) = (x, y))
  (h_angle : ∀ {P : ℝ × ℝ}, P ∈ [{q : ℝ × ℝ | (q.1 - 3)^2 + (q.2 - 4)^2 = 1}] →
    let M : ℝ × ℝ := (-m, 0), N : ℝ × ℝ := (m, 0)
    in ∠MPN = 90) : 
  ∃ m : ℝ, m = 6 := sorry

end max_m_value_l588_588435


namespace xyz_squared_sum_l588_588303

def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![[0, 3 * y, 2 * z],
   [2 * x, 2 * y, z],
   [x, -y, -z]]

def N_orthogonal (x y z : ℝ) : Prop :=
  (N x y z)ᵀ ⬝ (N x y z) = 1

theorem xyz_squared_sum (x y z : ℝ) (hN : N_orthogonal x y z) :
  x^2 + y^2 + z^2 = 46 / 105 :=
sorry

end xyz_squared_sum_l588_588303


namespace geometry_triangle_ratio_l588_588056

theorem geometry_triangle_ratio
    (A B C D N Q : Type)
    [Geometry A B C D N Q]
    (hAB : AB = 15)
    (hAC : AC = 13)
    (hAD_bisector : is_angle_bisector A D B C)
    (hN_on_AD : is_point_on_line N A D)
    (hAN_3ND : ratio AN ND = 3)
    (hQ_intersection : intersection Q AC BN) :
    exists m n : ℕ, 
    (nat_coprime m n) ∧
    ((CQ / QA = m / n) ∧ (m + n = 159)) :=
by
  sorry

end geometry_triangle_ratio_l588_588056


namespace length_of_BC_l588_588755

variable (A B C : Type) [inner_product_space ℝ A]
variable (AB AC BC : A)

variables (h1 : AB ⋅ AC = 0)
variables (h2 : ∥AB∥ = 3)
variables (h3 : ∥AC∥ = 2)

theorem length_of_BC : ∥BC∥ = Real.sqrt 13 := by
  sorry

end length_of_BC_l588_588755


namespace simplify_and_evaluate_correct_l588_588129

noncomputable def simplify_and_evaluate (x y : ℚ) : ℚ :=
  3 * (x^2 - 2 * x * y) - (3 * x^2 - 2 * y + 2 * (x * y + y))

theorem simplify_and_evaluate_correct : 
  simplify_and_evaluate (-1 / 2 : ℚ) (-3 : ℚ) = -12 := by
  sorry

end simplify_and_evaluate_correct_l588_588129


namespace segment_length_l588_588579

theorem segment_length (x : ℝ) (h : |x - (27^(1/3))| = 5) : ∃ a b : ℝ, a = 8 ∧ b = -2 ∧ |a - b| = 10 :=
by
  have hx1 : x = 27^(1/3) + 5 ∨ x = 27^(1/3) - 5 := abs_eq hx
  use [8, -2]
  split
  { calc 8 = 27^(1/3) + 5 := by sorry }
  split
  { calc -2 = 27^(1/3) - 5 := by sorry }
  { calc |8 - -2| = |8 + 2| := by sorry }

end segment_length_l588_588579


namespace log_pos_given_ineq_l588_588885

theorem log_pos_given_ineq (x y : ℝ) (h : 2^x - 2^y < 3^(-x) - 3^(-y)) : 
  log (y - x + 1) > 0 :=
by
  sorry

end log_pos_given_ineq_l588_588885
