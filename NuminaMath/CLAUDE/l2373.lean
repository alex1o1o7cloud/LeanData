import Mathlib

namespace NUMINAMATH_CALUDE_base_3_312_property_l2373_237344

def base_3_representation (n : ℕ) : List ℕ :=
  sorry

def count_digit (l : List ℕ) (d : ℕ) : ℕ :=
  sorry

theorem base_3_312_property :
  let base_3_312 := base_3_representation 312
  let x := count_digit base_3_312 0
  let y := count_digit base_3_312 1
  let z := count_digit base_3_312 2
  z - y + x = 2 := by sorry

end NUMINAMATH_CALUDE_base_3_312_property_l2373_237344


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l2373_237336

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5*a + 3*b) (13*a + 8*b) = Int.gcd a b := by sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l2373_237336


namespace NUMINAMATH_CALUDE_smallest_common_multiple_tutors_smallest_group_l2373_237303

theorem smallest_common_multiple (n : ℕ) : n > 0 ∧ n % 14 = 0 ∧ n % 10 = 0 ∧ n % 15 = 0 → n ≥ 210 := by
  sorry

theorem tutors_smallest_group : ∃ (n : ℕ), n > 0 ∧ n % 14 = 0 ∧ n % 10 = 0 ∧ n % 15 = 0 ∧ n = 210 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_tutors_smallest_group_l2373_237303


namespace NUMINAMATH_CALUDE_factorization_equality_l2373_237332

theorem factorization_equality (a : ℝ) : (a + 1) * (a + 2) + 1/4 = (a + 3/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2373_237332


namespace NUMINAMATH_CALUDE_coins_in_first_stack_l2373_237322

theorem coins_in_first_stack (total : ℕ) (stack2 : ℕ) (h1 : total = 12) (h2 : stack2 = 8) :
  total - stack2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_coins_in_first_stack_l2373_237322


namespace NUMINAMATH_CALUDE_randy_store_spending_l2373_237349

/-- Proves that Randy spends $2 per store trip -/
theorem randy_store_spending (initial_amount : ℕ) (final_amount : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  final_amount = 104 →
  trips_per_month = 4 →
  months_per_year = 12 →
  (initial_amount - final_amount) / (trips_per_month * months_per_year) = 2 := by
  sorry

end NUMINAMATH_CALUDE_randy_store_spending_l2373_237349


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2373_237364

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → perpendicularPlanes α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2373_237364


namespace NUMINAMATH_CALUDE_unique_valid_number_l2373_237341

def is_valid_number (n : ℕ) : Prop :=
  ∃ (p q r s t u : ℕ),
    0 ≤ p ∧ p ≤ 9 ∧
    0 ≤ q ∧ q ≤ 9 ∧
    0 ≤ r ∧ r ≤ 9 ∧
    0 ≤ s ∧ s ≤ 9 ∧
    0 ≤ t ∧ t ≤ 9 ∧
    0 ≤ u ∧ u ≤ 9 ∧
    n = p * 10^7 + q * 10^6 + 7 * 10^5 + 8 * 10^4 + r * 10^3 + s * 10^2 + t * 10 + u ∧
    n % 17 = 0 ∧
    n % 19 = 0 ∧
    p + q + r + s = t + u

theorem unique_valid_number :
  ∃! n, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2373_237341


namespace NUMINAMATH_CALUDE_peach_basket_problem_l2373_237356

theorem peach_basket_problem (baskets : Nat) (total_peaches : Nat) (green_excess : Nat) :
  baskets = 2 →
  green_excess = 2 →
  total_peaches = 12 →
  ∃ red_peaches : Nat, red_peaches * baskets + (red_peaches + green_excess) * baskets = total_peaches ∧ red_peaches = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_peach_basket_problem_l2373_237356


namespace NUMINAMATH_CALUDE_no_consecutive_ones_eq_fib_l2373_237304

/-- The number of binary sequences of length n with no two consecutive 1s -/
def no_consecutive_ones (n : ℕ) : ℕ :=
  sorry

/-- The nth Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: The number of binary sequences of length n with no two consecutive 1s
    is equal to the (n+2)th Fibonacci number -/
theorem no_consecutive_ones_eq_fib (n : ℕ) : no_consecutive_ones n = fib (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_ones_eq_fib_l2373_237304


namespace NUMINAMATH_CALUDE_rectangle_area_l2373_237362

/-- Given a rectangle with width m centimeters and length 1 centimeter more than twice its width,
    its area is equal to 2m^2 + m square centimeters. -/
theorem rectangle_area (m : ℝ) : 
  let width := m
  let length := 2 * m + 1
  width * length = 2 * m^2 + m := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2373_237362


namespace NUMINAMATH_CALUDE_plane_graph_is_bipartite_plane_regions_two_colorable_l2373_237326

/-- A graph representing regions formed by lines dividing a plane -/
structure PlaneGraph where
  V : Type* -- Vertices (regions)
  E : V → V → Prop -- Edges (neighboring regions)

/-- Definition of a bipartite graph -/
def IsBipartite (G : PlaneGraph) : Prop :=
  ∃ (A B : Set G.V), (∀ v, v ∈ A ∨ v ∈ B) ∧ 
    (∀ u v, G.E u v → (u ∈ A ∧ v ∈ B) ∨ (u ∈ B ∧ v ∈ A))

/-- Theorem: The graph representing regions formed by lines dividing a plane is bipartite -/
theorem plane_graph_is_bipartite (G : PlaneGraph) : IsBipartite G := by
  sorry

/-- Corollary: Regions formed by lines dividing a plane can be colored with two colors -/
theorem plane_regions_two_colorable (G : PlaneGraph) : 
  ∃ (color : G.V → Bool), ∀ u v, G.E u v → color u ≠ color v := by
  sorry

end NUMINAMATH_CALUDE_plane_graph_is_bipartite_plane_regions_two_colorable_l2373_237326


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2373_237366

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2373_237366


namespace NUMINAMATH_CALUDE_function_equation_1_bijective_function_equation_2_neither_function_equation_3_neither_function_equation_4_neither_l2373_237391

-- 1. f(x+f(y))=2f(x)+y is bijective
theorem function_equation_1_bijective (f : ℝ → ℝ) :
  (∀ x y, f (x + f y) = 2 * f x + y) → Function.Bijective f :=
sorry

-- 2. f(f(x))=0 is neither injective nor surjective
theorem function_equation_2_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = 0) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 3. f(f(x))=sin(x) is neither injective nor surjective
theorem function_equation_3_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = Real.sin x) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 4. f(x+y)=f(x)f(y) is neither injective nor surjective
theorem function_equation_4_neither (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x * f y) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

end NUMINAMATH_CALUDE_function_equation_1_bijective_function_equation_2_neither_function_equation_3_neither_function_equation_4_neither_l2373_237391


namespace NUMINAMATH_CALUDE_complement_union_M_N_l2373_237321

def I : Set ℕ := {x | x ≤ 10}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 4, 6, 8, 10}

theorem complement_union_M_N :
  (M ∪ N)ᶜ = {0, 5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l2373_237321


namespace NUMINAMATH_CALUDE_total_items_eq_1900_l2373_237347

/-- The number of rows of pencils and crayons. -/
def num_rows : ℕ := 19

/-- The number of pencils in each row. -/
def pencils_per_row : ℕ := 57

/-- The number of crayons in each row. -/
def crayons_per_row : ℕ := 43

/-- The total number of pencils and crayons. -/
def total_items : ℕ := num_rows * (pencils_per_row + crayons_per_row)

theorem total_items_eq_1900 : total_items = 1900 := by
  sorry

end NUMINAMATH_CALUDE_total_items_eq_1900_l2373_237347


namespace NUMINAMATH_CALUDE_nine_a_value_l2373_237346

theorem nine_a_value (a b : ℚ) (eq1 : 8 * a + 3 * b = 0) (eq2 : a = b - 3) : 9 * a = -81 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_a_value_l2373_237346


namespace NUMINAMATH_CALUDE_largest_reciprocal_l2373_237329

def numbers : List ℚ := [1/4, 3/7, 2, 10, 2023]

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem largest_reciprocal :
  ∀ n ∈ numbers, reciprocal (1/4) ≥ reciprocal n :=
by sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l2373_237329


namespace NUMINAMATH_CALUDE_diagonal_length_after_triangle_removal_l2373_237393

/-- The diagonal length of a quadrilateral formed by removing two equal-area right triangles from opposite corners of a square --/
theorem diagonal_length_after_triangle_removal (s : ℝ) (A : ℝ) (h1 : s = 20) (h2 : A = 50) :
  let x := Real.sqrt (2 * A)
  let diagonal := Real.sqrt ((s - x)^2 + (s - x)^2)
  diagonal = 10 * Real.sqrt 2 := by
  sorry

#check diagonal_length_after_triangle_removal

end NUMINAMATH_CALUDE_diagonal_length_after_triangle_removal_l2373_237393


namespace NUMINAMATH_CALUDE_least_sum_m_n_l2373_237316

theorem least_sum_m_n (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^m.val = k * n^n.val)
  (h3 : ¬ ∃ k : ℕ, m = k * n) :
  ∀ p q : ℕ+, 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ k : ℕ, p^p.val = k * q^q.val) → 
    (¬ ∃ k : ℕ, p = k * q) → 
    m + n ≤ p + q :=
by
  sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l2373_237316


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2373_237354

theorem complex_expression_simplification :
  ∀ (i : ℂ), i^2 = -1 → 7 * (4 - 2*i) + 4*i * (6 - 3*i) = 40 + 10*i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2373_237354


namespace NUMINAMATH_CALUDE_digits_difference_in_base_d_l2373_237365

/-- Given two digits A and B in base d > 7, such that AB + AA = 172 in base d, prove A - B = 5 in base d -/
theorem digits_difference_in_base_d (d A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  (A * d + B) + (A * d + A) = 1 * d^2 + 7 * d + 2 →
  A - B = 5 := by
sorry

end NUMINAMATH_CALUDE_digits_difference_in_base_d_l2373_237365


namespace NUMINAMATH_CALUDE_triangle_tangent_determinant_l2373_237313

/-- Given angles A, B, C of a non-right triangle, the determinant of the matrix
    | tan²A  1      1     |
    | 1      tan²B  1     |
    | 1      1      tan²C |
    is equal to 2. -/
theorem triangle_tangent_determinant (A B C : Real) 
  (h : A + B + C = π) 
  (h_non_right : A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2) : 
  let M : Matrix (Fin 3) (Fin 3) Real := 
    !![Real.tan A ^ 2, 1, 1; 
       1, Real.tan B ^ 2, 1; 
       1, 1, Real.tan C ^ 2]
  Matrix.det M = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_determinant_l2373_237313


namespace NUMINAMATH_CALUDE_cos_arcsin_seven_twentyfifths_l2373_237314

theorem cos_arcsin_seven_twentyfifths : 
  Real.cos (Real.arcsin (7 / 25)) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_seven_twentyfifths_l2373_237314


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2373_237340

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + 4*x + 5) + Real.sqrt (x^2 - 8*x + 25) ≥ 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2373_237340


namespace NUMINAMATH_CALUDE_car_pricing_problem_l2373_237334

theorem car_pricing_problem (X : ℝ) (A : ℝ) : 
  X > 0 →
  0.8 * X * (1 + A / 100) = 1.2 * X →
  A = 50 := by
sorry

end NUMINAMATH_CALUDE_car_pricing_problem_l2373_237334


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_l2373_237357

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh : (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_l2373_237357


namespace NUMINAMATH_CALUDE_sin_transformations_l2373_237387

open Real

theorem sin_transformations (x : ℝ) :
  (∀ x, sin (2 * (x - π/6)) = sin (2*x - π/3)) ∧
  (∀ x, sin (2 * (x - π/3)) = sin (2*x - π/3)) ∧
  (∀ x, sin (2 * (x + 5*π/6)) = sin (2*x - π/3)) :=
by sorry

end NUMINAMATH_CALUDE_sin_transformations_l2373_237387


namespace NUMINAMATH_CALUDE_age_difference_l2373_237368

theorem age_difference (A B C : ℕ) (h : C = A - 13) : A + B - (B + C) = 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2373_237368


namespace NUMINAMATH_CALUDE_selection_problem_l2373_237388

theorem selection_problem (n_teachers : ℕ) (n_students : ℕ) : n_teachers = 4 → n_students = 5 →
  (Nat.choose n_teachers 1 * Nat.choose n_students 2 + 
   Nat.choose n_teachers 2 * Nat.choose n_students 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l2373_237388


namespace NUMINAMATH_CALUDE_mangoes_in_basket_l2373_237353

/-- The number of mangoes in a basket of fruits -/
def mangoes_count (total_fruits : ℕ) (pears : ℕ) (pawpaws : ℕ) (lemons : ℕ) : ℕ :=
  total_fruits - (pears + pawpaws + lemons + lemons)

theorem mangoes_in_basket :
  mangoes_count 58 10 12 9 = 18 :=
by sorry

end NUMINAMATH_CALUDE_mangoes_in_basket_l2373_237353


namespace NUMINAMATH_CALUDE_total_weight_loss_l2373_237397

def weight_loss_problem (seth_loss jerome_loss veronica_loss total_loss : ℝ) : Prop :=
  seth_loss = 17.5 ∧
  jerome_loss = 3 * seth_loss ∧
  veronica_loss = seth_loss + 1.5 ∧
  total_loss = seth_loss + jerome_loss + veronica_loss

theorem total_weight_loss :
  ∃ (seth_loss jerome_loss veronica_loss total_loss : ℝ),
    weight_loss_problem seth_loss jerome_loss veronica_loss total_loss ∧
    total_loss = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_loss_l2373_237397


namespace NUMINAMATH_CALUDE_triangle_inequality_l2373_237382

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(a + b > c ∧ b + c > a ∧ c + a > b) ↔ min a (min b c) + (a + b + c - max a (max b c) - min a (min b c)) ≤ max a (max b c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2373_237382


namespace NUMINAMATH_CALUDE_probability_estimate_l2373_237317

def is_hit (n : ℕ) : Bool :=
  n ≥ 3 ∧ n ≤ 9

def group_has_three_hits (group : List ℕ) : Bool :=
  (group.filter is_hit).length ≥ 3

def count_successful_groups (groups : List (List ℕ)) : ℕ :=
  (groups.filter group_has_three_hits).length

theorem probability_estimate (groups : List (List ℕ)) 
  (h1 : groups.length = 20) 
  (h2 : ∀ g ∈ groups, g.length = 4) 
  (h3 : ∀ g ∈ groups, ∀ n ∈ g, n ≥ 0 ∧ n ≤ 9) : 
  (count_successful_groups groups : ℚ) / groups.length = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_probability_estimate_l2373_237317


namespace NUMINAMATH_CALUDE_perfect_apples_l2373_237333

/-- Given a batch of apples, calculate the number of perfect apples -/
theorem perfect_apples (total : ℕ) (too_small : ℚ) (not_ripe : ℚ) 
  (h1 : total = 30) 
  (h2 : too_small = 1/6) 
  (h3 : not_ripe = 1/3) : 
  ↑total * (1 - (too_small + not_ripe)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_perfect_apples_l2373_237333


namespace NUMINAMATH_CALUDE_prob_both_3_l2373_237310

-- Define the number of sides for each die
def die1_sides : ℕ := 6
def die2_sides : ℕ := 7

-- Define the probability of rolling a 3 on each die
def prob_3_die1 : ℚ := 1 / die1_sides
def prob_3_die2 : ℚ := 1 / die2_sides

-- Theorem: The probability of rolling a 3 on both dice is 1/42
theorem prob_both_3 : prob_3_die1 * prob_3_die2 = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_3_l2373_237310


namespace NUMINAMATH_CALUDE_triangle_longest_side_l2373_237305

/-- Given a triangle with sides 8, y+5, and 3y+2, and perimeter 45, the longest side is 24.5 -/
theorem triangle_longest_side (y : ℝ) :
  8 + (y + 5) + (3 * y + 2) = 45 →
  max 8 (max (y + 5) (3 * y + 2)) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l2373_237305


namespace NUMINAMATH_CALUDE_factor_tree_problem_l2373_237360

/-- Factor tree problem -/
theorem factor_tree_problem (F G Y Z X : ℕ) : 
  F = 2 * 5 →
  G = 7 * 3 →
  Y = 7 * F →
  Z = 11 * G →
  X = Y * Z →
  X = 16170 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l2373_237360


namespace NUMINAMATH_CALUDE_range_of_m_l2373_237320

theorem range_of_m (x y m : ℝ) : 
  x > 0 → y > 0 → x + y = 3 → 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → 
    (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 5) → 
  -1 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2373_237320


namespace NUMINAMATH_CALUDE_onion_weight_problem_l2373_237392

/-- Given 40 onions weighing 7.68 kg, and 35 of these onions having an average weight of 190 grams,
    the average weight of the remaining 5 onions is 206 grams. -/
theorem onion_weight_problem (total_weight : Real) (remaining_avg : Real) :
  total_weight = 7.68 →
  remaining_avg = 190 →
  (total_weight * 1000 - 35 * remaining_avg) / 5 = 206 := by
sorry

end NUMINAMATH_CALUDE_onion_weight_problem_l2373_237392


namespace NUMINAMATH_CALUDE_expand_binomials_l2373_237335

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l2373_237335


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2373_237331

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 = 1 → (x = 1 ∨ x = -1)) ↔
  (∀ x : ℝ, (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2373_237331


namespace NUMINAMATH_CALUDE_min_value_expression_l2373_237376

theorem min_value_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∃ (x y z w : ℝ), x ≥ 0 ∧ y > 0 ∧ z > 0 ∧ w ≥ 0 ∧ y + z ≥ x + w ∧
    ∀ (a' d' b' c' : ℝ), a' ≥ 0 → d' ≥ 0 → b' > 0 → c' > 0 → b' + c' ≥ a' + d' →
      (b' / (c' + d')) + (c' / (a' + b')) ≥ (y / (z + w)) + (z / (x + y)) ∧
      (y / (z + w)) + (z / (x + y)) = Real.sqrt 2 - 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_min_value_expression_l2373_237376


namespace NUMINAMATH_CALUDE_total_feet_count_l2373_237359

/-- Given a total of 50 animals with 30 hens, prove that the total number of feet is 140. -/
theorem total_feet_count (total_animals : ℕ) (num_hens : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : 
  total_animals = 50 → 
  num_hens = 30 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  num_hens * hen_feet + (total_animals - num_hens) * cow_feet = 140 := by
sorry

end NUMINAMATH_CALUDE_total_feet_count_l2373_237359


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2373_237373

theorem solution_set_inequality (x : ℝ) :
  (Set.Iio (1/3 : ℝ)) = {x | Real.sqrt (x^2 - 2*x + 1) > 2*x} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2373_237373


namespace NUMINAMATH_CALUDE_max_altitude_product_l2373_237355

/-- 
Given a triangle ABC with base AB = 1 and altitude from C of length h,
this theorem states the maximum product of the three altitudes and the
triangle configuration that achieves it.
-/
theorem max_altitude_product (h : ℝ) (h_pos : h > 0) :
  let max_product := if h ≤ 1/2 then h^2 else h^3 / (h^2 + 1/4)
  let optimal_triangle := if h ≤ 1/2 then "right triangle at C" else "isosceles triangle with AC = BC"
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a * b * h ≤ max_product ∧
    (a * b * h = max_product ↔ 
      (h ≤ 1/2 ∧ c^2 = a^2 + b^2) ∨ 
      (h > 1/2 ∧ a = b))) :=
by sorry


end NUMINAMATH_CALUDE_max_altitude_product_l2373_237355


namespace NUMINAMATH_CALUDE_total_sightings_first_quarter_l2373_237398

/-- The total number of animal sightings in the first three months of the year. -/
def total_sightings (january_sightings : ℕ) : ℕ :=
  let february_sightings := 3 * january_sightings
  let march_sightings := february_sightings / 2
  january_sightings + february_sightings + march_sightings

/-- Theorem stating that the total number of animal sightings in the first three months is 143,
    given that there were 26 sightings in January. -/
theorem total_sightings_first_quarter (h : total_sightings 26 = 143) : total_sightings 26 = 143 := by
  sorry

end NUMINAMATH_CALUDE_total_sightings_first_quarter_l2373_237398


namespace NUMINAMATH_CALUDE_square_2007_position_l2373_237363

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DCBA

-- Define the transformations
def rotate180 (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

def reflectHorizontal (pos : SquarePosition) : SquarePosition := pos

-- Define the sequence of transformations
def transformSquare (n : Nat) : SquarePosition :=
  if n % 2 = 1 then
    rotate180 SquarePosition.ABCD
  else
    reflectHorizontal (rotate180 SquarePosition.ABCD)

-- State the theorem
theorem square_2007_position :
  transformSquare 2007 = SquarePosition.DCBA := by sorry

end NUMINAMATH_CALUDE_square_2007_position_l2373_237363


namespace NUMINAMATH_CALUDE_road_repair_equation_l2373_237319

theorem road_repair_equation (x : ℝ) (h : x > 0) : 
  (150 / x - 150 / (x + 5) = 5) ↔ 
  (∃ (original_days actual_days : ℝ), 
    original_days > 0 ∧ 
    actual_days > 0 ∧ 
    original_days = 150 / x ∧ 
    actual_days = 150 / (x + 5) ∧ 
    original_days - actual_days = 5) :=
by sorry

end NUMINAMATH_CALUDE_road_repair_equation_l2373_237319


namespace NUMINAMATH_CALUDE_rotation_equivalence_l2373_237371

def clockwise_rotation : ℝ := 480
def counterclockwise_rotation : ℝ := 240

theorem rotation_equivalence :
  ∀ y : ℝ,
  y < 360 →
  (clockwise_rotation % 360 = (360 - y) % 360) →
  y = counterclockwise_rotation :=
by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l2373_237371


namespace NUMINAMATH_CALUDE_first_number_value_l2373_237348

theorem first_number_value (a b c : ℕ) : 
  a + b + c = 500 → 
  (b = 200 ∨ c = 200 ∨ a = 200) → 
  b = 2 * c → 
  c = 100 → 
  a = 200 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l2373_237348


namespace NUMINAMATH_CALUDE_fraction_product_is_one_l2373_237311

theorem fraction_product_is_one :
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_is_one_l2373_237311


namespace NUMINAMATH_CALUDE_triangle_properties_l2373_237302

open Real

/-- Given a triangle ABC with angle C = 2π/3 and c² = 5a² + ab, prove the following:
    1. sin B / sin A = 2
    2. The maximum value of sin A * sin B is 1/4 -/
theorem triangle_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle : angle_C = 2 * π / 3)
  (h_side : c^2 = 5 * a^2 + a * b) :
  (sin angle_B / sin angle_A = 2) ∧
  (∀ x y : ℝ, 0 < x ∧ x < π / 3 → sin x * sin y ≤ 1 / 4) :=
by sorry


end NUMINAMATH_CALUDE_triangle_properties_l2373_237302


namespace NUMINAMATH_CALUDE_sin_plus_cos_for_point_l2373_237337

/-- Theorem: If the terminal side of angle α passes through point P(-4,3), then sin α + cos α = -1/5 -/
theorem sin_plus_cos_for_point (α : Real) : 
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) → 
  Real.sin α + Real.cos α = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_for_point_l2373_237337


namespace NUMINAMATH_CALUDE_parabola_shift_l2373_237390

/-- The equation of a parabola after horizontal and vertical shifts -/
def shifted_parabola (a b c : ℝ) (x y : ℝ) : Prop :=
  y = a * (x - b)^2 + c

theorem parabola_shift :
  ∀ (x y : ℝ),
  (y = 3 * x^2) →  -- Original parabola
  (shifted_parabola 3 1 (-2) x y)  -- Shifted parabola
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2373_237390


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l2373_237399

-- Define set A
def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ a + 2 }

-- Define set B
def B : Set ℝ := { x | |x - 4| < 1 }

-- Theorem statement
theorem set_inclusion_equivalence (a : ℝ) : A a ⊇ B ↔ 3 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l2373_237399


namespace NUMINAMATH_CALUDE_sum_black_eq_sum_white_l2373_237350

/-- Represents a frame in a multiplication table -/
structure Frame (m n : ℕ) :=
  (is_odd_m : Odd m)
  (is_odd_n : Odd n)

/-- The sum of numbers in black squares of the frame -/
def sum_black (f : Frame m n) : ℕ := sorry

/-- The sum of numbers in white squares of the frame -/
def sum_white (f : Frame m n) : ℕ := sorry

/-- Theorem stating that the sum of numbers in black squares equals the sum of numbers in white squares -/
theorem sum_black_eq_sum_white (m n : ℕ) (f : Frame m n) :
  sum_black f = sum_white f := by sorry

end NUMINAMATH_CALUDE_sum_black_eq_sum_white_l2373_237350


namespace NUMINAMATH_CALUDE_problem_solution_l2373_237328

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : 
  a^2 - b^2 = 32 ∧ a * b = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2373_237328


namespace NUMINAMATH_CALUDE_calculation_proof_l2373_237395

theorem calculation_proof : -2^2 - Real.sqrt 9 + (-5)^2 * (2/5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2373_237395


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2373_237386

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ z₁ z₂ : ℂ, z₁^2 + 5*z₁ + m = 0 ∧ z₂^2 + 5*z₂ + m = 0 ∧ Complex.abs (z₁ - z₂) = 3) → 
  (m = 4 ∨ m = 17/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2373_237386


namespace NUMINAMATH_CALUDE_complex_angle_pi_third_l2373_237367

theorem complex_angle_pi_third (z : ℂ) : 
  z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_angle_pi_third_l2373_237367


namespace NUMINAMATH_CALUDE_complex_addition_proof_l2373_237345

theorem complex_addition_proof : ∃ z : ℂ, 2 * (5 - 3*I) + z = 4 + 11*I :=
by
  use -6 + 17*I
  sorry

end NUMINAMATH_CALUDE_complex_addition_proof_l2373_237345


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l2373_237312

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2020 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l2373_237312


namespace NUMINAMATH_CALUDE_taras_rowing_speed_l2373_237394

/-- Tara's rowing problem -/
theorem taras_rowing_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_distance = 20) 
  (h2 : upstream_distance = 4) 
  (h3 : time = 2) 
  (h4 : current_speed = 2) :
  ∃ v : ℝ, 
    v + current_speed = downstream_distance / time ∧ 
    v - current_speed = upstream_distance / time ∧ 
    v = 8 := by
sorry

end NUMINAMATH_CALUDE_taras_rowing_speed_l2373_237394


namespace NUMINAMATH_CALUDE_min_value_theorem_l2373_237384

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) (hm : m > 0) (hn : n > 0) 
  (h_fixed_point : a^(2 - 2) = 1) 
  (h_linear : m * 2 + 4 * n = 1) : 
  1 / m + 2 / n ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2373_237384


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2373_237327

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x > 0 ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2373_237327


namespace NUMINAMATH_CALUDE_max_areas_is_9n_l2373_237339

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii : Fin (3 * n)
  secant_lines : Fin 2

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  9 * disk.n

/-- Theorem stating that the maximum number of non-overlapping areas is 9n -/
theorem max_areas_is_9n (disk : DividedDisk) :
  max_areas disk = 9 * disk.n :=
by sorry

end NUMINAMATH_CALUDE_max_areas_is_9n_l2373_237339


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l2373_237372

/-- Represents a participant's score in a math challenge --/
structure Score where
  points_scored : ℚ
  points_attempted : ℚ

/-- Calculates the success ratio of a score --/
def successRatio (s : Score) : ℚ := s.points_scored / s.points_attempted

/-- Represents the scores of a participant over two days --/
structure TwoDayScore where
  day1 : Score
  day2 : Score

/-- Calculates the overall success ratio for a two-day score --/
def overallSuccessRatio (s : TwoDayScore) : ℚ :=
  (s.day1.points_scored + s.day2.points_scored) / (s.day1.points_attempted + s.day2.points_attempted)

/-- Gamma's score for each day --/
def gammaScore : Score := { points_scored := 180, points_attempted := 300 }

/-- Delta's maximum possible two-day score --/
def deltaMaxScore : TwoDayScore := {
  day1 := { points_scored := 179, points_attempted := 299 },
  day2 := { points_scored := 180, points_attempted := 301 }
}

theorem delta_max_success_ratio :
  (∀ s : TwoDayScore,
    s.day1.points_attempted + s.day2.points_attempted = 600 ∧
    successRatio s.day1 < successRatio gammaScore ∧
    successRatio s.day2 < successRatio gammaScore) →
  (∀ s : TwoDayScore,
    s.day1.points_attempted + s.day2.points_attempted = 600 ∧
    successRatio s.day1 < successRatio gammaScore ∧
    successRatio s.day2 < successRatio gammaScore →
    overallSuccessRatio s ≤ overallSuccessRatio deltaMaxScore) :=
by sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l2373_237372


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l2373_237343

/-- The number of cards Nell initially had -/
def initial_cards : ℕ := 304

/-- The number of cards Nell has left -/
def remaining_cards : ℕ := 276

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem cards_given_to_jeff : cards_given = 28 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l2373_237343


namespace NUMINAMATH_CALUDE_sequence_properties_l2373_237318

def a (n : ℕ) : ℚ := (2/3)^(n-1) * ((2/3)^(n-1) - 1)

theorem sequence_properties :
  (∀ n : ℕ, a n ≤ a 1) ∧
  (∀ n : ℕ, a n ≥ a 3) ∧
  (∀ n : ℕ, n ≥ 3 → a n > a (n+1)) ∧
  (a 1 = 0) ∧
  (a 3 = -20/81) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2373_237318


namespace NUMINAMATH_CALUDE_michaels_truck_rental_cost_l2373_237385

/-- Calculates the total cost of renting a truck -/
def truckRentalCost (rentalFee : ℚ) (chargePerMile : ℚ) (milesDriven : ℕ) : ℚ :=
  rentalFee + chargePerMile * milesDriven

/-- Proves that the total cost for Michael's truck rental is $95.74 -/
theorem michaels_truck_rental_cost :
  truckRentalCost 20.99 0.25 299 = 95.74 := by
  sorry

#eval truckRentalCost 20.99 0.25 299

end NUMINAMATH_CALUDE_michaels_truck_rental_cost_l2373_237385


namespace NUMINAMATH_CALUDE_lime_bottom_implies_magenta_top_l2373_237308

-- Define the colors
inductive Color
| Purple
| Cyan
| Magenta
| Lime
| Silver
| Black

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define the property that all faces have different colors
def has_unique_colors (c : Cube) : Prop :=
  c.top.color ≠ c.bottom.color ∧
  c.top.color ≠ c.front.color ∧
  c.top.color ≠ c.back.color ∧
  c.top.color ≠ c.left.color ∧
  c.top.color ≠ c.right.color ∧
  c.bottom.color ≠ c.front.color ∧
  c.bottom.color ≠ c.back.color ∧
  c.bottom.color ≠ c.left.color ∧
  c.bottom.color ≠ c.right.color ∧
  c.front.color ≠ c.back.color ∧
  c.front.color ≠ c.left.color ∧
  c.front.color ≠ c.right.color ∧
  c.back.color ≠ c.left.color ∧
  c.back.color ≠ c.right.color ∧
  c.left.color ≠ c.right.color

-- Theorem statement
theorem lime_bottom_implies_magenta_top (c : Cube) 
  (h1 : has_unique_colors c) 
  (h2 : c.bottom.color = Color.Lime) : 
  c.top.color = Color.Magenta :=
sorry

end NUMINAMATH_CALUDE_lime_bottom_implies_magenta_top_l2373_237308


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2373_237375

theorem cyclic_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2373_237375


namespace NUMINAMATH_CALUDE_rabbits_count_l2373_237325

/-- Represents the number of rabbits and peacocks in a zoo. -/
structure ZooAnimals where
  rabbits : ℕ
  peacocks : ℕ

/-- The total number of heads in the zoo is 60. -/
def total_heads (zoo : ZooAnimals) : Prop :=
  zoo.rabbits + zoo.peacocks = 60

/-- The total number of legs in the zoo is 192. -/
def total_legs (zoo : ZooAnimals) : Prop :=
  4 * zoo.rabbits + 2 * zoo.peacocks = 192

/-- Theorem stating that given the conditions, the number of rabbits is 36. -/
theorem rabbits_count (zoo : ZooAnimals) 
  (h1 : total_heads zoo) (h2 : total_legs zoo) : zoo.rabbits = 36 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_count_l2373_237325


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2373_237315

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 4 ∧
  (∀ (y : ℝ), y * |y| = 3 * y + 4 → x ≤ y) ∧
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2373_237315


namespace NUMINAMATH_CALUDE_pond_water_after_evaporation_l2373_237377

/-- Calculates the remaining water in a pond after evaporation --/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem: The pond contains 205 gallons after 45 days --/
theorem pond_water_after_evaporation :
  remaining_water 250 1 45 = 205 := by
  sorry

end NUMINAMATH_CALUDE_pond_water_after_evaporation_l2373_237377


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l2373_237381

/-- Given a quadratic polynomial x^2 - sx + q with roots a and b, 
    where a + b = a^2 + b^2 = a^3 + b^3 = ... = a^2008 + b^2008,
    the maximum value of 1/a^2009 + 1/b^2009 is 2. -/
theorem max_reciprocal_sum (s q a b : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2008 → a^n + b^n = a + b) →
  a * b = q →
  a + b = s →
  x^2 - s*x + q = (x - a) * (x - b) →
  (∃ M : ℝ, ∀ s' q' a' b' : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2008 → a'^n + b'^n = a' + b') →
    a' * b' = q' →
    a' + b' = s' →
    x^2 - s'*x + q' = (x - a') * (x - b') →
    1 / a'^2009 + 1 / b'^2009 ≤ M) ∧
  1 / a^2009 + 1 / b^2009 = M →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l2373_237381


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2373_237324

theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2373_237324


namespace NUMINAMATH_CALUDE_only_vertical_angles_true_l2373_237351

-- Define the propositions
def vertical_angles_equal : Prop := ∀ (α β : ℝ), α = β → α = β
def corresponding_angles_equal : Prop := ∀ (α β : ℝ), α = β
def product_one_implies_one : Prop := ∀ (a b : ℝ), a * b = 1 → a = 1 ∨ b = 1
def square_root_of_four : Prop := ∀ (x : ℝ), x^2 = 4 → x = 2

-- Theorem stating that only vertical_angles_equal is true
theorem only_vertical_angles_true : 
  vertical_angles_equal ∧ 
  ¬corresponding_angles_equal ∧ 
  ¬product_one_implies_one ∧ 
  ¬square_root_of_four :=
sorry

end NUMINAMATH_CALUDE_only_vertical_angles_true_l2373_237351


namespace NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2373_237300

/-- The coefficient of x^2 in the expansion of (x - 2/x)^4 * (x - 2) -/
def coefficient_x_squared : ℤ := 16

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x_squared_proof :
  coefficient_x_squared = 
    (-(binomial 4 1 : ℤ) * 2) * (-2 : ℤ) := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2373_237300


namespace NUMINAMATH_CALUDE_multiple_problem_l2373_237309

theorem multiple_problem (m : ℝ) : 38 + m * 43 = 124 ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_problem_l2373_237309


namespace NUMINAMATH_CALUDE_carol_extra_chore_earnings_l2373_237358

/-- Proves that given the conditions, Carol earns $1.50 per extra chore -/
theorem carol_extra_chore_earnings
  (weekly_allowance : ℚ)
  (num_weeks : ℕ)
  (total_amount : ℚ)
  (avg_extra_chores : ℚ)
  (h1 : weekly_allowance = 20)
  (h2 : num_weeks = 10)
  (h3 : total_amount = 425)
  (h4 : avg_extra_chores = 15) :
  (total_amount - weekly_allowance * num_weeks) / (avg_extra_chores * num_weeks) = 3/2 :=
sorry

end NUMINAMATH_CALUDE_carol_extra_chore_earnings_l2373_237358


namespace NUMINAMATH_CALUDE_principal_calculation_l2373_237389

/-- Proves that given specific conditions, the principal amount is 900 --/
theorem principal_calculation (interest_rate : ℚ) (time : ℚ) (final_amount : ℚ) :
  interest_rate = 5 / 100 →
  time = 12 / 5 →
  final_amount = 1008 →
  final_amount = (1 + interest_rate * time) * 900 :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l2373_237389


namespace NUMINAMATH_CALUDE_divisible_by_2_3_5_7_under_300_l2373_237338

theorem divisible_by_2_3_5_7_under_300 : 
  ∃! n : ℕ, n > 0 ∧ n < 300 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_2_3_5_7_under_300_l2373_237338


namespace NUMINAMATH_CALUDE_dragons_volleyball_games_l2373_237307

theorem dragons_volleyball_games :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (initial_games * 55 / 100) →
    (initial_wins + 8) = ((initial_games + 12) * 60 / 100) →
    initial_games + 12 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_dragons_volleyball_games_l2373_237307


namespace NUMINAMATH_CALUDE_ratio_problem_l2373_237374

theorem ratio_problem (p q n : ℝ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2373_237374


namespace NUMINAMATH_CALUDE_select_shoes_count_l2373_237361

/-- The number of ways to select 4 shoes from 4 pairs of different shoes,
    with at least 2 shoes forming a pair -/
def select_shoes : ℕ :=
  Nat.choose 8 4 - 16

theorem select_shoes_count : select_shoes = 54 := by
  sorry

end NUMINAMATH_CALUDE_select_shoes_count_l2373_237361


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2373_237370

theorem rectangle_ratio (area : ℝ) (length : ℝ) (breadth : ℝ) :
  area = 6075 →
  length = 135 →
  area = length * breadth →
  length / breadth = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2373_237370


namespace NUMINAMATH_CALUDE_function_inequality_l2373_237378

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_ab : a > b ∧ b > 1) 
  (h_deriv : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f a + f b ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2373_237378


namespace NUMINAMATH_CALUDE_amount_increase_l2373_237383

theorem amount_increase (initial_amount : ℚ) : 
  (initial_amount * (9/8) * (9/8) = 4050) → initial_amount = 3200 := by
  sorry

end NUMINAMATH_CALUDE_amount_increase_l2373_237383


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2373_237380

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2373_237380


namespace NUMINAMATH_CALUDE_particle_probability_l2373_237396

def move_probability (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * move_probability (x-1) y +
       (1/3) * move_probability x (y-1) +
       (1/3) * move_probability (x-1) (y-1)

theorem particle_probability :
  move_probability 4 4 = 245 / 3^7 :=
sorry

end NUMINAMATH_CALUDE_particle_probability_l2373_237396


namespace NUMINAMATH_CALUDE_appetizer_cost_is_six_l2373_237330

/-- The cost of dinner for a group, including main meals, appetizers, tip, and rush order fee. --/
def dinner_cost (main_meal_cost : ℝ) (num_people : ℕ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_fee : ℝ) : ℝ :=
  let subtotal := main_meal_cost * num_people + appetizer_cost * num_appetizers
  subtotal + tip_rate * subtotal + rush_fee

/-- Theorem stating that the appetizer cost is $6.00 given the specified conditions. --/
theorem appetizer_cost_is_six :
  ∃ (appetizer_cost : ℝ),
    dinner_cost 12 4 2 appetizer_cost 0.2 5 = 77 ∧
    appetizer_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_appetizer_cost_is_six_l2373_237330


namespace NUMINAMATH_CALUDE_video_game_lives_l2373_237301

/-- Given an initial number of players, additional players, and total lives,
    calculate the number of lives per player. -/
def lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players + additional_players)

/-- Theorem: In the video game scenario, each player has 6 lives. -/
theorem video_game_lives : lives_per_player 2 2 24 = 6 := by
  sorry

#eval lives_per_player 2 2 24

end NUMINAMATH_CALUDE_video_game_lives_l2373_237301


namespace NUMINAMATH_CALUDE_box_volume_increase_l2373_237342

/-- 
Given a rectangular box with dimensions l, w, and h satisfying:
1. Volume is 5400 cubic inches
2. Surface area is 1920 square inches
3. Sum of edge lengths is 240 inches
Prove that increasing each dimension by 2 inches results in a volume of 7568 cubic inches
-/
theorem box_volume_increase (l w h : ℝ) 
  (hvolume : l * w * h = 5400)
  (harea : 2 * (l * w + w * h + h * l) = 1920)
  (hedge : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7568 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2373_237342


namespace NUMINAMATH_CALUDE_intersecting_rectangles_area_l2373_237379

/-- The total shaded area of two intersecting rectangles -/
theorem intersecting_rectangles_area (rect1_width rect1_height rect2_width rect2_height overlap_width overlap_height : ℕ) 
  (h1 : rect1_width = 4 ∧ rect1_height = 12)
  (h2 : rect2_width = 5 ∧ rect2_height = 7)
  (h3 : overlap_width = 4 ∧ overlap_height = 5) :
  rect1_width * rect1_height + rect2_width * rect2_height - overlap_width * overlap_height = 63 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_rectangles_area_l2373_237379


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_above_70_l2373_237352

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_sum_of_two_distinct_primes_above_70 :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p > 70 ∧ 
    q > 70 ∧ 
    p ≠ q ∧ 
    p + q = 144 ∧ 
    (∀ (r s : ℕ), is_prime r → is_prime s → r > 70 → s > 70 → r ≠ s → r + s ≥ 144) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_above_70_l2373_237352


namespace NUMINAMATH_CALUDE_capital_calculation_l2373_237369

/-- Calculates the capital of a business partner who joined later --/
def calculate_capital (x_capital y_capital : ℕ) (z_profit total_profit : ℕ) (z_join_month : ℕ) : ℕ :=
  let x_share := x_capital * 12
  let y_share := y_capital * 12
  let z_months := 12 - z_join_month
  let total_ratio := x_share + y_share
  ((z_profit * total_ratio) / (total_profit - z_profit)) / z_months

theorem capital_calculation (x_capital y_capital : ℕ) (z_profit total_profit : ℕ) (z_join_month : ℕ) :
  x_capital = 20000 →
  y_capital = 25000 →
  z_profit = 14000 →
  total_profit = 50000 →
  z_join_month = 5 →
  calculate_capital x_capital y_capital z_profit total_profit z_join_month = 30000 := by
  sorry

#eval calculate_capital 20000 25000 14000 50000 5

end NUMINAMATH_CALUDE_capital_calculation_l2373_237369


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2373_237306

theorem square_sum_theorem (x y : ℝ) (h1 : x - y = 5) (h2 : -x*y = 4) : x^2 + y^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2373_237306


namespace NUMINAMATH_CALUDE_intersection_implies_k_range_l2373_237323

/-- The line equation kx - y - k - 1 = 0 intersects the line segment MN,
    where M(2,1) and N(3,2) are the endpoints of the segment. -/
def intersects_segment (k : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    k * (2 + t) - (1 + t) - k - 1 = 0

/-- The theorem states that if the line intersects the segment MN,
    then k is in the range [3/2, 2]. -/
theorem intersection_implies_k_range :
  ∀ k : ℝ, intersects_segment k → 3/2 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_k_range_l2373_237323
