import Mathlib

namespace NUMINAMATH_GPT_interest_rate_is_10_perc_l2198_219865

noncomputable def interest_rate (P : ℝ) (R : ℝ) (T : ℝ := 2) : ℝ := (P * R * T) / 100

theorem interest_rate_is_10_perc (P : ℝ) : 
  (interest_rate P 10) = P / 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_is_10_perc_l2198_219865


namespace NUMINAMATH_GPT_simplify_expression_l2198_219879

theorem simplify_expression (α : ℝ) :
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) - (2 * Real.cos (2 * α)^2 - 1)) /
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) + (2 * Real.cos (2 * α)^2 - 1)) = Real.tan (2 * α) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2198_219879


namespace NUMINAMATH_GPT_arithmetic_sequence_k_l2198_219849

theorem arithmetic_sequence_k :
  ∀ (a : ℕ → ℤ) (d : ℤ) (k : ℕ),
  d ≠ 0 →
  (∀ n : ℕ, a n = a 0 + n * d) →
  a 0 = 0 →
  a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 →
  k = 22 :=
by
  intros a d k hdnz h_arith h_a1_zero h_ak_sum
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_l2198_219849


namespace NUMINAMATH_GPT_intersection_points_l2198_219884

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := -f x
noncomputable def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  let a := 2
  let b := 1
  10 * a + b = 21 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l2198_219884


namespace NUMINAMATH_GPT_range_of_b_in_acute_triangle_l2198_219857

variable {a b c : ℝ}

theorem range_of_b_in_acute_triangle (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_acute : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2))
  (h_arith_seq : ∃ d : ℝ, 0 ≤ d ∧ a = b - d ∧ c = b + d)
  (h_sum_squares : a^2 + b^2 + c^2 = 21) :
  (2 * Real.sqrt 42) / 5 < b ∧ b ≤ Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_range_of_b_in_acute_triangle_l2198_219857


namespace NUMINAMATH_GPT_second_bucket_capacity_l2198_219861

-- Define the initial conditions as given in the problem.
def tank_capacity : ℕ := 48
def bucket1_capacity : ℕ := 4

-- Define the number of times the 4-liter bucket is used.
def bucket1_uses : ℕ := tank_capacity / bucket1_capacity

-- Define a condition related to bucket uses.
def buckets_use_relation (x : ℕ) : Prop :=
  bucket1_uses = (tank_capacity / x) - 4

-- Formulate the theorem that states the capacity of the second bucket.
theorem second_bucket_capacity (x : ℕ) (h : buckets_use_relation x) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_second_bucket_capacity_l2198_219861


namespace NUMINAMATH_GPT_intersection_A_B_l2198_219816

def A : Set ℝ := { x | x * Real.sqrt (x^2 - 4) ≥ 0 }
def B : Set ℝ := { x | |x - 1| + |x + 1| ≥ 2 }

theorem intersection_A_B : (A ∩ B) = ({-2} ∪ Set.Ici 2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2198_219816


namespace NUMINAMATH_GPT_truck_travel_distance_l2198_219869

variable (d1 d2 g1 g2 : ℝ)
variable (rate : ℝ)

-- Define the conditions
axiom condition1 : d1 = 300
axiom condition2 : g1 = 10
axiom condition3 : rate = d1 / g1
axiom condition4 : g2 = 15

-- Define the goal
theorem truck_travel_distance : d2 = rate * g2 := by
  -- axiom assumption placeholder
  exact sorry

end NUMINAMATH_GPT_truck_travel_distance_l2198_219869


namespace NUMINAMATH_GPT_exactly_one_wins_at_most_two_win_l2198_219854

def prob_A : ℚ := 4 / 5 
def prob_B : ℚ := 3 / 5 
def prob_C : ℚ := 7 / 10

theorem exactly_one_wins :
  (prob_A * (1 - prob_B) * (1 - prob_C) + 
   (1 - prob_A) * prob_B * (1 - prob_C) + 
   (1 - prob_A) * (1 - prob_B) * prob_C) = 47 / 250 := 
by sorry

theorem at_most_two_win :
  (1 - (prob_A * prob_B * prob_C)) = 83 / 125 :=
by sorry

end NUMINAMATH_GPT_exactly_one_wins_at_most_two_win_l2198_219854


namespace NUMINAMATH_GPT_verify_triangle_operation_l2198_219800

def triangle (a b c : ℕ) : ℕ := a^2 + b^2 + c^2

theorem verify_triangle_operation : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end NUMINAMATH_GPT_verify_triangle_operation_l2198_219800


namespace NUMINAMATH_GPT_problem_l2198_219808

def expr : ℤ := 7^2 - 4 * 5 + 2^2

theorem problem : expr = 33 := by
  sorry

end NUMINAMATH_GPT_problem_l2198_219808


namespace NUMINAMATH_GPT_sum_of_sequences_l2198_219896

-- Definition of the problem conditions
def seq1 := [2, 12, 22, 32, 42]
def seq2 := [10, 20, 30, 40, 50]
def sum_seq1 := 2 + 12 + 22 + 32 + 42
def sum_seq2 := 10 + 20 + 30 + 40 + 50

-- Lean statement of the problem
theorem sum_of_sequences :
  sum_seq1 + sum_seq2 = 260 := by
  sorry

end NUMINAMATH_GPT_sum_of_sequences_l2198_219896


namespace NUMINAMATH_GPT_arithmetic_square_root_of_nine_l2198_219843

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_nine_l2198_219843


namespace NUMINAMATH_GPT_sphere_volume_in_cone_l2198_219809

theorem sphere_volume_in_cone (d : ℝ) (r : ℝ) (π : ℝ) (V : ℝ) (h1 : d = 12) (h2 : r = d / 2) (h3 : V = (4 / 3) * π * r^3) :
  V = 288 * π :=
by 
  sorry

end NUMINAMATH_GPT_sphere_volume_in_cone_l2198_219809


namespace NUMINAMATH_GPT_complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l2198_219838

-- Definitions of the sets A and B
def set_A : Set ℝ := {y : ℝ | -1 < y ∧ y < 4}
def set_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 5}

-- Complement of B in the universal set U (ℝ)
def complement_B : Set ℝ := {y : ℝ | y ≤ 0 ∨ y ≥ 5}

theorem complement_B_def : (complement_B = {y : ℝ | y ≤ 0 ∨ y ≥ 5}) :=
by sorry

-- Union of A and B
def union_A_B : Set ℝ := {y : ℝ | -1 < y ∧ y < 5}

theorem union_A_B_def : (set_A ∪ set_B = union_A_B) :=
by sorry

-- Intersection of A and B
def intersection_A_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 4}

theorem intersection_A_B_def : (set_A ∩ set_B = intersection_A_B) :=
by sorry

-- Intersection of A and the complement of B
def intersection_A_complement_B : Set ℝ := {y : ℝ | -1 < y ∧ y ≤ 0}

theorem intersection_A_complement_B_def : (set_A ∩ complement_B = intersection_A_complement_B) :=
by sorry

-- Intersection of the complements of A and B
def complement_A : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 4} -- Derived from complement of A
def intersection_complements : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 5}

theorem intersection_complements_def : (complement_A ∩ complement_B = intersection_complements) :=
by sorry

end NUMINAMATH_GPT_complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l2198_219838


namespace NUMINAMATH_GPT_five_digit_integers_count_l2198_219813
open BigOperators

noncomputable def permutations_with_repetition (n : ℕ) (reps : List ℕ) : ℕ :=
  n.factorial / ((reps.map (λ x => x.factorial)).prod)

theorem five_digit_integers_count :
  permutations_with_repetition 5 [2, 2] = 30 :=
by
  sorry

end NUMINAMATH_GPT_five_digit_integers_count_l2198_219813


namespace NUMINAMATH_GPT_price_after_discount_eq_cost_price_l2198_219882

theorem price_after_discount_eq_cost_price (m : Real) :
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  price_after_discount = m :=
by
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  sorry

end NUMINAMATH_GPT_price_after_discount_eq_cost_price_l2198_219882


namespace NUMINAMATH_GPT_greatest_three_digit_number_condition_l2198_219835

theorem greatest_three_digit_number_condition :
  ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (n % 7 = 2) ∧ (n % 6 = 4) ∧ (n = 982) := 
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_number_condition_l2198_219835


namespace NUMINAMATH_GPT_linear_function_quadrants_l2198_219834

theorem linear_function_quadrants (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : ¬ ∃ x : ℝ, ∃ y : ℝ, x > 0 ∧ y < 0 ∧ y = b * x - a :=
sorry

end NUMINAMATH_GPT_linear_function_quadrants_l2198_219834


namespace NUMINAMATH_GPT_meaningful_iff_gt_3_l2198_219840

section meaningful_expression

variable (a : ℝ)

def is_meaningful (a : ℝ) : Prop :=
  (a > 3)

theorem meaningful_iff_gt_3 : (∃ b, b = (a + 3) / Real.sqrt (a - 3)) ↔ is_meaningful a :=
by
  sorry

end meaningful_expression

end NUMINAMATH_GPT_meaningful_iff_gt_3_l2198_219840


namespace NUMINAMATH_GPT_shopkeeper_sold_articles_l2198_219898

theorem shopkeeper_sold_articles (C : ℝ) (N : ℕ) 
  (h1 : (35 * C = N * C + (1/6) * (N * C))) : 
  N = 30 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_sold_articles_l2198_219898


namespace NUMINAMATH_GPT_max_strings_cut_volleyball_net_l2198_219823

-- Define the structure of a volleyball net with 10x20 cells where each cell is divided into 4 triangles.
structure VolleyballNet : Type where
  -- The dimensions of the volleyball net
  rows : ℕ
  cols : ℕ
  -- Number of nodes (vertices + centers)
  nodes : ℕ
  -- Maximum number of strings (edges) connecting neighboring nodes that can be cut without disconnecting the net
  max_cut_without_disconnection : ℕ

-- Define the specific volleyball net in question
def volleyball_net : VolleyballNet := 
  { rows := 10, 
    cols := 20, 
    nodes := (11 * 21) + (10 * 20), -- vertices + center nodes
    max_cut_without_disconnection := 800 
  }

-- The main theorem stating that we can cut these strings without the net falling apart
theorem max_strings_cut_volleyball_net (net : VolleyballNet) 
    (h_dim : net.rows = 10) 
    (h_dim2 : net.cols = 20) :
  net.max_cut_without_disconnection = 800 :=
sorry -- The proof is omitted

end NUMINAMATH_GPT_max_strings_cut_volleyball_net_l2198_219823


namespace NUMINAMATH_GPT_sequence_n_5_l2198_219868

theorem sequence_n_5 (a : ℤ) (n : ℕ → ℤ) 
  (h1 : ∀ i > 1, n i = 2 * n (i - 1) + a)
  (h2 : n 2 = 5)
  (h3 : n 8 = 257) : n 5 = 33 :=
by
  sorry

end NUMINAMATH_GPT_sequence_n_5_l2198_219868


namespace NUMINAMATH_GPT_maximum_a_value_l2198_219866

theorem maximum_a_value :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a + 1)*x^2 - (a + 1)*x + 2022 ∧ (a + 1)*x^2 - (a + 1)*x + 2022 ≤ 2022) →
  a ≤ 16175 := 
by {
  sorry
}

end NUMINAMATH_GPT_maximum_a_value_l2198_219866


namespace NUMINAMATH_GPT_div_by_13_l2198_219829

theorem div_by_13 (n : ℕ) (h : 0 < n) : 13 ∣ (4^(2*n - 1) + 3^(n + 1)) :=
by 
  sorry

end NUMINAMATH_GPT_div_by_13_l2198_219829


namespace NUMINAMATH_GPT_frog_reaches_top_l2198_219897

theorem frog_reaches_top (x : ℕ) (h1 : ∀ d ≤ x - 1, 3 * d + 5 ≥ 50) : x = 16 := by
  sorry

end NUMINAMATH_GPT_frog_reaches_top_l2198_219897


namespace NUMINAMATH_GPT_closed_under_all_operations_l2198_219885

structure sqrt2_num where
  re : ℚ
  im : ℚ

namespace sqrt2_num

def add (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re + y.re, x.im + y.im⟩

def subtract (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re - y.re, x.im - y.im⟩

def multiply (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re * y.re + 2 * x.im * y.im, x.re * y.im + x.im * y.re⟩

def divide (x y : sqrt2_num) : sqrt2_num :=
  let denom := y.re^2 - 2 * y.im^2
  ⟨(x.re * y.re - 2 * x.im * y.im) / denom, (x.im * y.re - x.re * y.im) / denom⟩

theorem closed_under_all_operations (a b c d : ℚ) :
  ∃ (e f : ℚ), 
    add ⟨a, b⟩ ⟨c, d⟩ = ⟨e, f⟩ ∧ 
    ∃ (g h : ℚ), 
    subtract ⟨a, b⟩ ⟨c, d⟩ = ⟨g, h⟩ ∧ 
    ∃ (i j : ℚ), 
    multiply ⟨a, b⟩ ⟨c, d⟩ = ⟨i, j⟩ ∧ 
    ∃ (k l : ℚ), 
    divide ⟨a, b⟩ ⟨c, d⟩ = ⟨k, l⟩ := by
  sorry

end sqrt2_num

end NUMINAMATH_GPT_closed_under_all_operations_l2198_219885


namespace NUMINAMATH_GPT_total_weight_correct_l2198_219856

def Marco_strawberry_weight : ℕ := 15
def Dad_strawberry_weight : ℕ := 22
def total_strawberry_weight : ℕ := Marco_strawberry_weight + Dad_strawberry_weight

theorem total_weight_correct :
  total_strawberry_weight = 37 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_correct_l2198_219856


namespace NUMINAMATH_GPT_find_integer_cube_sum_l2198_219864

-- Define the problem in Lean
theorem find_integer_cube_sum : ∃ n : ℤ, n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  use 6
  sorry

end NUMINAMATH_GPT_find_integer_cube_sum_l2198_219864


namespace NUMINAMATH_GPT_find_width_of_floor_l2198_219863

variable (w : ℝ) -- width of the floor

theorem find_width_of_floor (h1 : w - 4 > 0) (h2 : 10 - 4 > 0) 
                            (area_rug : (10 - 4) * (w - 4) = 24) : w = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_width_of_floor_l2198_219863


namespace NUMINAMATH_GPT_exists_number_divisible_by_5_pow_1000_with_no_zeros_l2198_219822

theorem exists_number_divisible_by_5_pow_1000_with_no_zeros :
  ∃ n : ℕ, (5 ^ 1000 ∣ n) ∧ (∀ d ∈ n.digits 10, d ≠ 0) := 
sorry

end NUMINAMATH_GPT_exists_number_divisible_by_5_pow_1000_with_no_zeros_l2198_219822


namespace NUMINAMATH_GPT_newspaper_cost_over_8_weeks_l2198_219805

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end NUMINAMATH_GPT_newspaper_cost_over_8_weeks_l2198_219805


namespace NUMINAMATH_GPT_quadratic_roots_range_l2198_219889

theorem quadratic_roots_range (k : ℝ) : (x^2 - 6*x + k = 0) → k < 9 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_range_l2198_219889


namespace NUMINAMATH_GPT_percentage_of_200_l2198_219831

theorem percentage_of_200 : ((1/4) / 100) * 200 = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_200_l2198_219831


namespace NUMINAMATH_GPT_water_formed_l2198_219887

theorem water_formed (n_HCl : ℕ) (n_CaCO3: ℕ) (n_H2O: ℕ) 
  (balance_eqn: ∀ (n : ℕ), 
    (2 * n_CaCO3) ≤ n_HCl ∧
    n_H2O = n_CaCO3 ):
  n_HCl = 4 ∧ n_CaCO3 = 2 → n_H2O = 2 :=
by
  intros h0
  obtain ⟨h1, h2⟩ := h0
  sorry

end NUMINAMATH_GPT_water_formed_l2198_219887


namespace NUMINAMATH_GPT_even_integer_operations_l2198_219886

theorem even_integer_operations (k : ℤ) (a : ℤ) (h : a = 2 * k) :
  (a * 5) % 2 = 0 ∧ (a ^ 2) % 2 = 0 ∧ (a ^ 3) % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_integer_operations_l2198_219886


namespace NUMINAMATH_GPT_cubic_polynomial_k_l2198_219802

noncomputable def h (x : ℝ) : ℝ := x^3 - x - 2

theorem cubic_polynomial_k (k : ℝ → ℝ)
  (hk : ∃ (B : ℝ), ∀ (x : ℝ), k x = B * (x - (root1 ^ 2)) * (x - (root2 ^ 2)) * (x - (root3 ^ 2)))
  (hroots : h (root1) = 0 ∧ h (root2) = 0 ∧ h (root3) = 0)
  (h_values : k 0 = 2) :
  k (-8) = -20 :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_k_l2198_219802


namespace NUMINAMATH_GPT_find_fourth_number_l2198_219833

theorem find_fourth_number (x : ℝ) (h : (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001) : x = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_number_l2198_219833


namespace NUMINAMATH_GPT_line_circle_intersect_l2198_219848

theorem line_circle_intersect {a : ℝ} :
  ∃ P : ℝ × ℝ, (P.1, P.2) = (-2, 0) ∧ (a * P.1 - P.2 + 2 * a = 0) ∧ (P.1^2 + P.2^2 < 9) :=
by
  use (-2, 0)
  sorry

end NUMINAMATH_GPT_line_circle_intersect_l2198_219848


namespace NUMINAMATH_GPT_triangles_form_even_square_l2198_219851

-- Given conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def triangle_area (b h : ℕ) : ℚ :=
  (b * h) / 2

-- Statement of the problem
theorem triangles_form_even_square (n : ℕ) :
  (∀ t : Fin n, is_right_triangle 3 4 5 ∧ triangle_area 3 4 = 6) →
  (∃ a : ℕ, a^2 = 6 * n) →
  Even n :=
by
  sorry

end NUMINAMATH_GPT_triangles_form_even_square_l2198_219851


namespace NUMINAMATH_GPT_desired_cost_per_pound_l2198_219862

/-- 
Let $p_1 = 8$, $w_1 = 25$, $p_2 = 5$, and $w_2 = 50$ represent the prices and weights of two types of candies.
Calculate the desired cost per pound $p_m$ of the mixture.
-/
theorem desired_cost_per_pound 
  (p1 : ℝ) (w1 : ℝ) (p2 : ℝ) (w2 : ℝ) (p_m : ℝ) 
  (h1 : p1 = 8) (h2 : w1 = 25) (h3 : p2 = 5) (h4 : w2 = 50) :
  p_m = (p1 * w1 + p2 * w2) / (w1 + w2) → p_m = 6 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_desired_cost_per_pound_l2198_219862


namespace NUMINAMATH_GPT_percentage_of_boys_l2198_219842

def ratio_boys_girls := 2 / 3
def ratio_teacher_students := 1 / 6
def total_people := 36

theorem percentage_of_boys : ∃ (n_student n_teacher n_boys n_girls : ℕ), 
  n_student + n_teacher = 35 ∧
  n_student * (1 + 1/6) = total_people ∧
  n_boys / n_student = ratio_boys_girls ∧
  n_teacher / n_student = ratio_teacher_students ∧
  ((n_boys : ℚ) / total_people) * 100 = 400 / 7 :=
sorry

end NUMINAMATH_GPT_percentage_of_boys_l2198_219842


namespace NUMINAMATH_GPT_hoseok_basketballs_l2198_219836

theorem hoseok_basketballs (v s b : ℕ) (h₁ : v = 40) (h₂ : s = v + 18) (h₃ : b = s - 23) : b = 35 := by
  sorry

end NUMINAMATH_GPT_hoseok_basketballs_l2198_219836


namespace NUMINAMATH_GPT_megan_seashells_l2198_219812

theorem megan_seashells (current_seashells desired_seashells diff_seashells : ℕ)
  (h1 : current_seashells = 307)
  (h2 : desired_seashells = 500)
  (h3 : diff_seashells = desired_seashells - current_seashells) :
  diff_seashells = 193 :=
by
  sorry

end NUMINAMATH_GPT_megan_seashells_l2198_219812


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l2198_219872

theorem min_value_of_a_plus_b 
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_eq : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l2198_219872


namespace NUMINAMATH_GPT_coeff_of_x_pow_4_in_expansion_l2198_219873

theorem coeff_of_x_pow_4_in_expansion : 
  (∃ c : ℤ, c = (-1)^3 * Nat.choose 8 3 ∧ c = -56) :=
by
  sorry

end NUMINAMATH_GPT_coeff_of_x_pow_4_in_expansion_l2198_219873


namespace NUMINAMATH_GPT_quadratic_symmetry_l2198_219819

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^2 + b * x + 1

theorem quadratic_symmetry 
  (a b x1 x2 : ℝ) 
  (h_quad : f x1 a b = f x2 a b) 
  (h_diff : x1 ≠ x2) 
  (h_nonzero : a ≠ 0) :
  f (x1 + x2) a b = 1 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_symmetry_l2198_219819


namespace NUMINAMATH_GPT_remaining_number_larger_than_4_l2198_219825

theorem remaining_number_larger_than_4 (m : ℕ) (h : 2 ≤ m) (a : ℚ) (b : ℚ) (h_sum_inv : (1 : ℚ) - 1 / (2 * m + 1 : ℚ) = 3 / 4 + 1 / b) :
  b > 4 :=
by sorry

end NUMINAMATH_GPT_remaining_number_larger_than_4_l2198_219825


namespace NUMINAMATH_GPT_prism_properties_sum_l2198_219883

/-- Prove that the sum of the number of edges, corners, and faces of a rectangular box (prism) with dimensions 2 by 3 by 4 is 26. -/
theorem prism_properties_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := 
by
  -- Provided conditions and definitions
  let edges := 12
  let corners := 8
  let faces := 6
  -- Summing up these values
  exact rfl

end NUMINAMATH_GPT_prism_properties_sum_l2198_219883


namespace NUMINAMATH_GPT_SavingsInequality_l2198_219875

theorem SavingsInequality (n : ℕ) : 52 + 15 * n > 70 + 12 * n := 
by sorry

end NUMINAMATH_GPT_SavingsInequality_l2198_219875


namespace NUMINAMATH_GPT_sum_of_number_and_square_eq_132_l2198_219855

theorem sum_of_number_and_square_eq_132 (x : ℝ) (h : x + x^2 = 132) : x = 11 ∨ x = -12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_square_eq_132_l2198_219855


namespace NUMINAMATH_GPT_total_shapes_proof_l2198_219847

def stars := 50
def stripes := 13

def circles : ℕ := (stars / 2) - 3
def squares : ℕ := (2 * stripes) + 6
def triangles : ℕ := (stars - stripes) * 2
def diamonds : ℕ := (stars + stripes) / 4

def total_shapes : ℕ := circles + squares + triangles + diamonds

theorem total_shapes_proof : total_shapes = 143 := by
  sorry

end NUMINAMATH_GPT_total_shapes_proof_l2198_219847


namespace NUMINAMATH_GPT_probability_heads_9_or_more_12_flips_l2198_219888

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end NUMINAMATH_GPT_probability_heads_9_or_more_12_flips_l2198_219888


namespace NUMINAMATH_GPT_total_chocolate_pieces_l2198_219827

def total_chocolates (boxes : ℕ) (per_box : ℕ) : ℕ :=
  boxes * per_box

theorem total_chocolate_pieces :
  total_chocolates 6 500 = 3000 :=
by
  sorry

end NUMINAMATH_GPT_total_chocolate_pieces_l2198_219827


namespace NUMINAMATH_GPT_trigonometric_identity_l2198_219817

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2198_219817


namespace NUMINAMATH_GPT_four_roots_sum_eq_neg8_l2198_219806

def op (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := op x 2

theorem four_roots_sum_eq_neg8 :
  ∃ (x1 x2 x3 x4 : ℝ), 
  (x1 ≠ -2) ∧ (x2 ≠ -2) ∧ (x3 ≠ -2) ∧ (x4 ≠ -2) ∧
  (f x1 = Real.log (abs (x1 + 2))) ∧ 
  (f x2 = Real.log (abs (x2 + 2))) ∧ 
  (f x3 = Real.log (abs (x3 + 2))) ∧ 
  (f x4 = Real.log (abs (x4 + 2))) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ 
  x2 ≠ x3 ∧ x2 ≠ x4 ∧ 
  x3 ≠ x4 ∧ 
  x1 + x2 + x3 + x4 = -8 :=
by 
  sorry

end NUMINAMATH_GPT_four_roots_sum_eq_neg8_l2198_219806


namespace NUMINAMATH_GPT_percent_calculation_l2198_219850

theorem percent_calculation (y : ℝ) : (0.3 * 0.7 * y - 0.1 * y) = 0.11 * y ∧ (0.11 * y / y * 100 = 11) := by
  sorry

end NUMINAMATH_GPT_percent_calculation_l2198_219850


namespace NUMINAMATH_GPT_pete_and_raymond_spent_together_l2198_219899

    def value_nickel : ℕ := 5
    def value_dime : ℕ := 10
    def value_quarter : ℕ := 25

    def pete_nickels_spent : ℕ := 4
    def pete_dimes_spent : ℕ := 3
    def pete_quarters_spent : ℕ := 2

    def raymond_initial : ℕ := 250
    def raymond_nickels_left : ℕ := 5
    def raymond_dimes_left : ℕ := 7
    def raymond_quarters_left : ℕ := 4
    
    def total_spent : ℕ := 155

    theorem pete_and_raymond_spent_together :
      (pete_nickels_spent * value_nickel + pete_dimes_spent * value_dime + pete_quarters_spent * value_quarter)
      + (raymond_initial - (raymond_nickels_left * value_nickel + raymond_dimes_left * value_dime + raymond_quarters_left * value_quarter))
      = total_spent :=
      by
        sorry
    
end NUMINAMATH_GPT_pete_and_raymond_spent_together_l2198_219899


namespace NUMINAMATH_GPT_initial_observations_l2198_219826

theorem initial_observations (n : ℕ) (S : ℕ) 
  (h1 : S / n = 11)
  (h2 : ∃ (new_obs : ℕ), (S + new_obs) / (n + 1) = 10 ∧ new_obs = 4):
  n = 6 := 
sorry

end NUMINAMATH_GPT_initial_observations_l2198_219826


namespace NUMINAMATH_GPT_general_term_arithmetic_sum_terms_geometric_l2198_219880

section ArithmeticSequence

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Conditions for Part 1
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  S 5 - S 2 = 195 ∧ d = -2 ∧
  ∀ n, S n = n * (a 1 + (n - 1) * (d / 2))

-- Prove the general term formula for the sequence {a_n}
theorem general_term_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) 
    (h : sum_arithmetic_sequence S a d) : 
    ∀ n, a n = -2 * n + 73 :=
sorry

end ArithmeticSequence


section GeometricSequence

variables {b : ℕ → ℝ} {n : ℕ} {T : ℕ → ℝ} {a : ℕ → ℝ}

-- Conditions for Part 2
def sum_geometric_sequence (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 13 ∧ b 2 = 65 ∧ a 4 = 65

-- Prove the sum of the first n terms for the sequence {b_n}
theorem sum_terms_geometric (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ)
    (h : sum_geometric_sequence b T a) : 
    ∀ n, T n = 13 * (5^n - 1) / 4 :=
sorry

end GeometricSequence

end NUMINAMATH_GPT_general_term_arithmetic_sum_terms_geometric_l2198_219880


namespace NUMINAMATH_GPT_joseph_drives_more_l2198_219815

-- Definitions for the problem
def v_j : ℝ := 50 -- Joseph's speed in mph
def t_j : ℝ := 2.5 -- Joseph's time in hours
def v_k : ℝ := 62 -- Kyle's speed in mph
def t_k : ℝ := 2 -- Kyle's time in hours

-- Prove that Joseph drives 1 more mile than Kyle
theorem joseph_drives_more : (v_j * t_j) - (v_k * t_k) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_joseph_drives_more_l2198_219815


namespace NUMINAMATH_GPT_find_number_l2198_219820

theorem find_number (x : ℝ) (h : (25 / 100) * x = 20 / 100 * 30) : x = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2198_219820


namespace NUMINAMATH_GPT_alcohol_to_water_ratio_l2198_219892

theorem alcohol_to_water_ratio (V p q : ℝ) (hV : V > 0) (hp : p > 0) (hq : q > 0) :
  let alcohol_first_jar := (p / (p + 1)) * V
  let water_first_jar   := (1 / (p + 1)) * V
  let alcohol_second_jar := (2 * q / (q + 1)) * V
  let water_second_jar   := (2 / (q + 1)) * V
  let total_alcohol := alcohol_first_jar + alcohol_second_jar
  let total_water := water_first_jar + water_second_jar
  (total_alcohol / total_water) = ((p * (q + 1) + 2 * p + 2 * q) / (q + 1 + 2 * p + 2)) :=
by
  sorry

end NUMINAMATH_GPT_alcohol_to_water_ratio_l2198_219892


namespace NUMINAMATH_GPT_max_value_of_f_l2198_219844

noncomputable def f (x : ℝ) := x^3 - 3 * x + 1

theorem max_value_of_f (h: ∃ x, f x = -1) : ∃ y, f y = 3 :=
by
  -- We'll later prove this with appropriate mathematical steps using Lean tactics
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2198_219844


namespace NUMINAMATH_GPT_equal_probability_of_selection_l2198_219811

-- Define a structure representing the scenario of the problem.
structure SamplingProblem :=
  (total_students : ℕ)
  (eliminated_students : ℕ)
  (remaining_students : ℕ)
  (selection_size : ℕ)
  (systematic_step : ℕ)

-- Instantiate the specific problem.
def problem_instance : SamplingProblem :=
  { total_students := 3001
  , eliminated_students := 1
  , remaining_students := 3000
  , selection_size := 50
  , systematic_step := 60 }

-- Define the main theorem to be proven.
theorem equal_probability_of_selection (prob : SamplingProblem) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ prob.remaining_students → 
  (prob.remaining_students - prob.systematic_step * ((i - 1) / prob.systematic_step) = i) :=
sorry

end NUMINAMATH_GPT_equal_probability_of_selection_l2198_219811


namespace NUMINAMATH_GPT_sum_series_eq_two_l2198_219859

theorem sum_series_eq_two : ∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1)) = 2 :=
sorry

end NUMINAMATH_GPT_sum_series_eq_two_l2198_219859


namespace NUMINAMATH_GPT_find_functional_solution_l2198_219893

theorem find_functional_solution (c : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) :
  ∀ x : ℝ, f x = x ^ 3 + c * x := by
  sorry

end NUMINAMATH_GPT_find_functional_solution_l2198_219893


namespace NUMINAMATH_GPT_part1_part2_l2198_219801
-- Import the entire Mathlib library for broader usage

-- Definition of the given vectors
def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

-- Part 1: Prove the dot product when x = -1 is 31
theorem part1 : (a.1 * (-1) + a.2 * (5)) = 31 := by
  sorry

-- Part 2: Prove the value of x when the vectors are parallel
theorem part2 : (4 : ℝ) / x = (7 : ℝ) / (x + 6) → x = 8 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2198_219801


namespace NUMINAMATH_GPT_hall_length_l2198_219860

theorem hall_length
  (breadth : ℝ) (stone_length_dm stone_width_dm : ℝ) (num_stones : ℕ) (L : ℝ)
  (h_breadth : breadth = 15)
  (h_stone_length : stone_length_dm = 6)
  (h_stone_width : stone_width_dm = 5)
  (h_num_stones : num_stones = 1800)
  (h_length : L = 36) :
  let stone_length := stone_length_dm / 10
  let stone_width := stone_width_dm / 10
  let stone_area := stone_length * stone_width
  let total_area := num_stones * stone_area
  total_area / breadth = L :=
by {
  sorry
}

end NUMINAMATH_GPT_hall_length_l2198_219860


namespace NUMINAMATH_GPT_area_of_largest_medallion_is_314_l2198_219876

noncomputable def largest_medallion_area_in_square (side: ℝ) (π: ℝ) : ℝ :=
  let diameter := side
  let radius := diameter / 2
  let area := π * radius^2
  area

theorem area_of_largest_medallion_is_314 :
  largest_medallion_area_in_square 20 3.14 = 314 := 
  sorry

end NUMINAMATH_GPT_area_of_largest_medallion_is_314_l2198_219876


namespace NUMINAMATH_GPT_beths_total_crayons_l2198_219804

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem beths_total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end NUMINAMATH_GPT_beths_total_crayons_l2198_219804


namespace NUMINAMATH_GPT_range_of_a_l2198_219878

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (e^x + 1) * (a * x + 2 * a - 2) < 2) → a < 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2198_219878


namespace NUMINAMATH_GPT_sum_in_base5_correct_l2198_219853

-- Define numbers in base 5
def n1 : ℕ := 231
def n2 : ℕ := 414
def n3 : ℕ := 123

-- Function to convert a number from base 5 to base 10
def base5_to_base10(n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100)
  d0 * 1 + d1 * 5 + d2 * 25

-- Convert the given numbers from base 5 to base 10
def n1_base10 : ℕ := base5_to_base10 n1
def n2_base10 : ℕ := base5_to_base10 n2
def n3_base10 : ℕ := base5_to_base10 n3

-- Base 10 sum
def sum_base10 : ℕ := n1_base10 + n2_base10 + n3_base10

-- Function to convert a number from base 10 to base 5
def base10_to_base5(n : ℕ) : ℕ :=
  let d0 := n % 5
  let d1 := (n / 5) % 5
  let d2 := (n / 25) % 5
  let d3 := (n / 125)
  d0 * 1 + d1 * 10 + d2 * 100 + d3 * 1000

-- Convert the sum from base 10 to base 5
def sum_base5 : ℕ := base10_to_base5 sum_base10

-- The theorem to prove the sum in base 5 is 1323_5
theorem sum_in_base5_correct : sum_base5 = 1323 := by
  -- Proof steps would go here, but we insert sorry to skip it
  sorry

end NUMINAMATH_GPT_sum_in_base5_correct_l2198_219853


namespace NUMINAMATH_GPT_largest_x_plus_y_l2198_219871

theorem largest_x_plus_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 18 / 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_x_plus_y_l2198_219871


namespace NUMINAMATH_GPT_range_of_m_l2198_219803

theorem range_of_m (m : ℝ) : (¬ ∃ x : ℝ, 4 ^ x + 2 ^ (x + 1) + m = 0) → m ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2198_219803


namespace NUMINAMATH_GPT_remainder_6n_mod_4_l2198_219814

theorem remainder_6n_mod_4 (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_6n_mod_4_l2198_219814


namespace NUMINAMATH_GPT_brittany_second_test_grade_l2198_219895

theorem brittany_second_test_grade
  (first_test_grade second_test_grade : ℕ) 
  (average_after_second : ℕ)
  (h1 : first_test_grade = 78)
  (h2 : average_after_second = 81) 
  (h3 : (first_test_grade + second_test_grade) / 2 = average_after_second) :
  second_test_grade = 84 :=
by
  sorry

end NUMINAMATH_GPT_brittany_second_test_grade_l2198_219895


namespace NUMINAMATH_GPT_product_mod_7_l2198_219839

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end NUMINAMATH_GPT_product_mod_7_l2198_219839


namespace NUMINAMATH_GPT_work_completion_time_l2198_219890

-- Define the constants for work rates and times
def W : ℚ := 1
def P_rate : ℚ := W / 20
def Q_rate : ℚ := W / 12
def initial_days : ℚ := 4

-- Define the amount of work done by P in the initial 4 days
def work_done_initial : ℚ := initial_days * P_rate

-- Define the remaining work after initial 4 days
def remaining_work : ℚ := W - work_done_initial

-- Define the combined work rate of P and Q
def combined_rate : ℚ := P_rate + Q_rate

-- Define the time taken to complete the remaining work
def remaining_days : ℚ := remaining_work / combined_rate

-- Define the total time taken to complete the work
def total_days : ℚ := initial_days + remaining_days

-- The theorem to prove
theorem work_completion_time :
  total_days = 10 := 
by
  -- these term can be the calculation steps
  sorry

end NUMINAMATH_GPT_work_completion_time_l2198_219890


namespace NUMINAMATH_GPT_alternating_sign_max_pos_l2198_219881

theorem alternating_sign_max_pos (x : ℕ → ℝ) 
  (h_nonzero : ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n ≠ 0)
  (h_condition : ∀ k, 1 ≤ k ∧ k ≤ 2022 → x k + (1 / x (k + 1)) < 0)
  (h_periodic : x 2023 = x 1) :
  ∃ m, m = 1011 ∧ ( ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n > 0 → n ≤ m ∧ m ≤ 2022 ) := 
sorry

end NUMINAMATH_GPT_alternating_sign_max_pos_l2198_219881


namespace NUMINAMATH_GPT_merchant_profit_percentage_l2198_219877

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : ((S - C) / C) * 100 = 50 := by
  -- Adding "by" to denote beginning of proof section
  sorry  -- Proof is skipped

end NUMINAMATH_GPT_merchant_profit_percentage_l2198_219877


namespace NUMINAMATH_GPT_smallest_value_between_0_and_1_l2198_219841

theorem smallest_value_between_0_and_1 (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3 * y ∧ y^3 < y^(1/3 : ℝ) ∧ y^3 < 1 ∧ y^3 < 1 / y :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_between_0_and_1_l2198_219841


namespace NUMINAMATH_GPT_root_range_of_quadratic_eq_l2198_219818

theorem root_range_of_quadratic_eq (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 < x2 ∧ x1^2 + k * x1 - k = 0 ∧ x2^2 + k * x2 - k = 0 ∧ 1 < x1 ∧ x1 < 2 ∧ 2 < x2 ∧ x2 < 3) ↔  (-9 / 2) < k ∧ k < -4 :=
by
  sorry

end NUMINAMATH_GPT_root_range_of_quadratic_eq_l2198_219818


namespace NUMINAMATH_GPT_complex_values_l2198_219828

open Complex

theorem complex_values (a b : ℝ) (i : ℂ) (h1 : i = Complex.I) (h2 : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by
  sorry

end NUMINAMATH_GPT_complex_values_l2198_219828


namespace NUMINAMATH_GPT_type_B_machine_time_l2198_219846

theorem type_B_machine_time :
  (2 * (1 / 5) + 3 * (1 / B) = 5 / 6) → B = 90 / 13 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_type_B_machine_time_l2198_219846


namespace NUMINAMATH_GPT_sara_spent_on_bought_movie_l2198_219824

-- Define the costs involved
def cost_ticket : ℝ := 10.62
def cost_rent : ℝ := 1.59
def total_spent : ℝ := 36.78

-- Define the quantity of tickets
def number_of_tickets : ℝ := 2

-- Define the total cost on tickets
def cost_on_tickets : ℝ := cost_ticket * number_of_tickets

-- Define the total cost on tickets and rented movie
def cost_on_tickets_and_rent : ℝ := cost_on_tickets + cost_rent

-- Define the total amount spent on buying the movie
def cost_bought_movie : ℝ := total_spent - cost_on_tickets_and_rent

-- The statement we need to prove
theorem sara_spent_on_bought_movie : cost_bought_movie = 13.95 :=
by
  sorry

end NUMINAMATH_GPT_sara_spent_on_bought_movie_l2198_219824


namespace NUMINAMATH_GPT_max_value_of_PQ_l2198_219891

noncomputable def maxDistance (P Q : ℝ × ℝ) : ℝ :=
  let dist (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
  let O1 : ℝ × ℝ := (0, 4)
  dist P Q

theorem max_value_of_PQ:
  ∀ (P Q : ℝ × ℝ),
    (P.1 ^ 2 + (P.2 - 4) ^ 2 = 1) →
    (Q.1 ^ 2 / 9 + Q.2 ^ 2 = 1) →
    maxDistance P Q ≤ 1 + 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_PQ_l2198_219891


namespace NUMINAMATH_GPT_largest_possible_d_l2198_219870

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 := 
sorry

end NUMINAMATH_GPT_largest_possible_d_l2198_219870


namespace NUMINAMATH_GPT_simplify_expression_l2198_219832

theorem simplify_expression :
  (2 * 6 / (12 * 14)) * (3 * 12 * 14 / (2 * 6 * 3)) * 2 = 2 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2198_219832


namespace NUMINAMATH_GPT_pairs_of_socks_now_l2198_219837

def initial_socks : Nat := 28
def socks_thrown_away : Nat := 4
def socks_bought : Nat := 36

theorem pairs_of_socks_now : (initial_socks - socks_thrown_away + socks_bought) / 2 = 30 := by
  sorry

end NUMINAMATH_GPT_pairs_of_socks_now_l2198_219837


namespace NUMINAMATH_GPT_smallest_positive_x_l2198_219821

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end NUMINAMATH_GPT_smallest_positive_x_l2198_219821


namespace NUMINAMATH_GPT_cost_of_one_each_l2198_219830

theorem cost_of_one_each (x y z : ℝ) (h1 : 3 * x + 7 * y + z = 24) (h2 : 4 * x + 10 * y + z = 33) :
  x + y + z = 6 :=
sorry

end NUMINAMATH_GPT_cost_of_one_each_l2198_219830


namespace NUMINAMATH_GPT_unanswered_questions_l2198_219845

variables (c w u : ℕ)

theorem unanswered_questions :
  (c + w + u = 50) ∧
  (6 * c + u = 120) ∧
  (3 * c - 2 * w = 45) →
  u = 37 :=
by {
  sorry
}

end NUMINAMATH_GPT_unanswered_questions_l2198_219845


namespace NUMINAMATH_GPT_reduced_price_per_kg_of_oil_l2198_219867

/-- The reduced price per kg of oil is approximately Rs. 48 -
given a 30% reduction in price and the ability to buy 5 kgs more
for Rs. 800. -/
theorem reduced_price_per_kg_of_oil
  (P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 800 / R = (800 / P) + 5) : 
  R = 48 :=
sorry

end NUMINAMATH_GPT_reduced_price_per_kg_of_oil_l2198_219867


namespace NUMINAMATH_GPT_range_of_m_l2198_219852

variable (f : Real → Real)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom decreasing_function : ∀ x y, x < y → -1 < x ∧ y < 1 → f x > f y
axiom domain : ∀ x, -1 < x ∧ x < 1 → true

-- The statement to be proved
theorem range_of_m (m : Real) : 
  f (1 - m) + f (1 - m^2) < 0 → 0 < m → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2198_219852


namespace NUMINAMATH_GPT_sweetsies_remainder_l2198_219874

-- Each definition used in Lean 4 statement should be directly from the conditions in a)
def number_of_sweetsies_in_one_bag (m : ℕ): Prop :=
  m % 8 = 5

theorem sweetsies_remainder (m : ℕ) (h : number_of_sweetsies_in_one_bag m) : 
  (4 * m) % 8 = 4 := by
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_sweetsies_remainder_l2198_219874


namespace NUMINAMATH_GPT_gcd_m_n_l2198_219810

   -- Define m and n according to the problem statement
   def m : ℕ := 33333333
   def n : ℕ := 666666666

   -- State the theorem we want to prove
   theorem gcd_m_n : Int.gcd m n = 3 := by
     -- put proof here
     sorry
   
end NUMINAMATH_GPT_gcd_m_n_l2198_219810


namespace NUMINAMATH_GPT_max_volume_of_open_top_box_l2198_219858

noncomputable def box_max_volume (x : ℝ) : ℝ :=
  (10 - 2 * x) * (16 - 2 * x) * x

theorem max_volume_of_open_top_box : ∃ x : ℝ, 0 < x ∧ x < 5 ∧ box_max_volume x = 144 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_of_open_top_box_l2198_219858


namespace NUMINAMATH_GPT_problem_solution_l2198_219894

def expr := 1 + 1 / (1 + 1 / (1 + 1))
def answer : ℚ := 5 / 3

theorem problem_solution : expr = answer :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2198_219894


namespace NUMINAMATH_GPT_four_digit_number_l2198_219807

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem four_digit_number (n : ℕ) (hn1 : 1000 ≤ n) (hn2 : n < 10000) (condition : n = 9 * (reverse_digits n)) :
  n = 9801 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_l2198_219807
