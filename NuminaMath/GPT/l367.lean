import Mathlib

namespace ordering_PQR_l367_367093

noncomputable def P := Real.sqrt 2
noncomputable def Q := Real.sqrt 7 - Real.sqrt 3
noncomputable def R := Real.sqrt 6 - Real.sqrt 2

theorem ordering_PQR : P > R ∧ R > Q := by
  sorry

end ordering_PQR_l367_367093


namespace value_of_E_l367_367888

variable {D E F : ℕ}

theorem value_of_E (h1 : D + E + F = 16) (h2 : F + D + 1 = 16) (h3 : E - 1 = D) : E = 1 :=
sorry

end value_of_E_l367_367888


namespace sum_of_possible_intersections_l367_367737

theorem sum_of_possible_intersections (n : ℕ) (h : n = 5) :
  let possible_values := {0, 1, 3, 4, 5, 6, 7, 8, 9, 10}
  ∑ x in possible_values, x = 53 :=
by
  have fact : finset.sum {0, 1, 3, 4, 5, 6, 7, 8, 9, 10} id = 53 := by norm_num
  exact fact

end sum_of_possible_intersections_l367_367737


namespace exactly_one_proposition_correct_l367_367798

def proposition1 : Prop :=
∀ α β γ : Plane, (α ⊥ γ ∧ β ⊥ γ) → (α ∥ β)

def proposition2 : Prop :=
∀ (l m : Line) (α : Plane), (l ≠ m ∧ l ⊥ α ∧ l ∥ m) → m ⊥ α

def proposition3 : Prop :=
∀ (α β : Plane) (m : Line), (m ∈ α) → (α ⊥ β ↔ m ⊥ β)

def proposition4 : Prop :=
∀ (a b : Line) (P : Point), (a ≠ b ∧ skew a b) → 
  ∃ (π : Plane), (P ∈ π ∧ π ⊥ a ∧ π ∥ b) ∨ (P ∈ π ∧ π ⊥ b ∧ π ∥ a)

theorem exactly_one_proposition_correct : 
  (prop_exist : ∃ (p1_correct : proposition1 ∨ proposition2 ∨ proposition3 ∨ proposition4), 
    (proposition1 → ¬proposition2) ∧ (proposition1 → ¬proposition3) ∧ 
    (proposition1 → ¬proposition4) ∧ (proposition2 → ¬proposition3) ∧ 
    (proposition2 → ¬proposition4) ∧ (proposition3 → ¬proposition4)) :=
by sorry

end exactly_one_proposition_correct_l367_367798


namespace concurrency_of_lines_l367_367417

variables {A B C H O M N P : Type*}
variables [geometry : Geometry A B C H O M N P]

/- Preconditions: -/
axiom orthocenter {A B C H : Type*} (h1 : is_orthocenter H A B C)
axiom circumcenter1 {A B C O : Type*} (h2 : is_circumcenter O A B C)
axiom distinct_points {A B C H O : Type*} (h3 : H ≠ A ∧ H ≠ B ∧ H ≠ C ∧ H ≠ O)
axiom circumcenter2 {H B C M : Type*} (h4 : is_circumcenter M H B C)
axiom circumcenter3 {H C A N : Type*} (h5 : is_circumcenter N H C A)
axiom circumcenter4 {H A B P : Type*} (h6 : is_circumcenter P H A B)

/- Theorem statement: -/
theorem concurrency_of_lines 
  (AM BN CP OH : Line*):
  concurrent_lines AM BN CP OH :=
sorry

end concurrency_of_lines_l367_367417


namespace ellipse_equation_line_fixed_point_l367_367421

-- Defines the ellipse problem
noncomputable def ellipse_foci (F₁ F₂ : (ℝ × ℝ)) : Prop :=
  F₁ = (-2, 0) ∧ F₂ = (2, 0)

noncomputable def line_pq_conditions (PQ : ℝ) : Prop :=
  PQ = 2 * Real.sqrt 2

noncomputable def slopes_conditions (k₁ k₂ : ℝ) : Prop :=
  k₁ + k₂ = 8

-- Proves that the equation of the ellipse given certain conditions
theorem ellipse_equation (F₁ F₂ : (ℝ × ℝ)) (PQ : ℝ) (h₁ : ellipse_foci F₁ F₂) (h₂ : line_pq_conditions PQ) :
  ∃ (a b : ℝ), a = 2 * Real.sqrt 2 ∧ b = 2 ∧ 
  (∀ x y : ℝ, (x^2 / 8) + (y^2 / 4) = 1) :=
sorry

-- Proves that line AB passes through the fixed point
theorem line_fixed_point (M : (ℝ × ℝ)) (k₁ k₂ : ℝ) (h₃ : slopes_conditions k₁ k₂) :
  ∀ A B : (ℝ × ℝ), ∃ (fixed_point : (ℝ × ℝ)), fixed_point = (-0.5, -2) ∧ 
  (line_through A B) :=
sorry

end ellipse_equation_line_fixed_point_l367_367421


namespace ravi_total_money_l367_367954

theorem ravi_total_money (nickels quarters dimes : ℕ) 
    (h1 : nickels = 6) 
    (h2 : quarters = nickels + 2) 
    (h3 : dimes = quarters + 4) : 
    (6 * 0.05 + quarters * 0.25 + dimes * 0.10) = 3.50 := 
by 
  sorry

end ravi_total_money_l367_367954


namespace units_digit_of_6_pow_5_l367_367594

theorem units_digit_of_6_pow_5 : (6^5 % 10) = 6 := 
by sorry

end units_digit_of_6_pow_5_l367_367594


namespace max_height_of_ball_l367_367234

def h (t : ℝ) : ℝ := -16 * t^2 + 80 * t + 21

theorem max_height_of_ball : ∃ t : ℝ, t = 2.5 ∧ h t = 121 :=
by
  use 2.5
  split
  · refl
  · sorry

end max_height_of_ball_l367_367234


namespace product_of_b_l367_367545

noncomputable def b_product : ℤ :=
  let y1 := 3
  let y2 := 8
  let x1 := 2
  let l := y2 - y1 -- Side length of the square
  let b₁ := x1 - l -- One possible value of b
  let b₂ := x1 + l -- Another possible value of b
  b₁ * b₂ -- Product of possible values of b

theorem product_of_b :
  b_product = -21 := by
  sorry

end product_of_b_l367_367545


namespace sum_adjacent_6_is_29_l367_367841

-- Define the grid and the placement of numbers 1 to 4
structure Grid :=
  (grid : Fin 3 → Fin 3 → Nat)
  (h_unique : ∀ i j, grid i j ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_distinct : Function.Injective (λ (i : Fin 3) (j : Fin 3), grid i j))
  (h_placement : grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧ grid 2 2 = 4)

-- Define the condition of the sum of numbers adjacent to 5 being 9
def sum_adjacent_5 (g : Grid) : Prop :=
  let (i, j) := (0, 1) in -- Position for number 5
  (g.grid (i.succ) j + g.grid (i.succ.pred) j + g.grid i (j.succ) + g.grid i (j.pred)) = 9

-- Define the main theorem
theorem sum_adjacent_6_is_29 (g : Grid) (h_sum_adj_5 : sum_adjacent_5 g) : 
  (g.grid 1 0 + g.grid 1 2 + g.grid 0 1 + g.grid 2 1 = 29) := sorry

end sum_adjacent_6_is_29_l367_367841


namespace pq_r_divisibility_l367_367441

noncomputable def f : Polynomial ℂ := X^3 + 2 * X^2 + 4 * X + 2
noncomputable def g (p q r : ℂ) : Polynomial ℂ := X^4 + 6 * X^3 + 8 * p * X^2 + 6 * q * X + r

theorem pq_r_divisibility (p q r : ℂ) (h : g p q r % f = 0) 
  (hp : p = 1.5) (hq : q = 3) (hr : r = 8) : 
  (p + q) * r = 36 := 
by
  simp only [hp, hq, hr]
  norm_num
  sorry

end pq_r_divisibility_l367_367441


namespace matrix_identity_l367_367911

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 4, 3]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  B^4 = -3 • B + 2 • I :=
by
  sorry

end matrix_identity_l367_367911


namespace curve_intersection_l367_367468

noncomputable def curve_C1 (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin α - cos α, 3 - 2 * sqrt 3 * sin α * cos α - 2 * (cos α)^2)

noncomputable def curve_C2 (ρ θ m : ℝ) : bool :=
  ρ * sin (θ - π / 4) = sqrt 2 / 2 * m

theorem curve_intersection (m : ℝ) : - (1 / 4) ≤ m ∧ m ≤ 6 :=
  sorry

end curve_intersection_l367_367468


namespace sum_adjacent_cells_of_6_is_29_l367_367849

theorem sum_adjacent_cells_of_6_is_29 (table : Fin 3 × Fin 3 → ℕ)
  (uniq : Function.Injective table)
  (range : ∀ x, 1 ≤ table x ∧ table x ≤ 9)
  (pos_1 : table ⟨0, 0⟩ = 1)
  (pos_2 : table ⟨2, 0⟩ = 2)
  (pos_3 : table ⟨0, 2⟩ = 3)
  (pos_4 : table ⟨2, 2⟩ = 4)
  (adj_5 : (∑ i in ({⟨1, 0⟩, ⟨1, 2⟩, ⟨0, 1⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 9) :
  (∑ i in ({⟨0, 1⟩, ⟨1, 0⟩, ⟨1, 2⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 29 :=
by
  sorry

end sum_adjacent_cells_of_6_is_29_l367_367849


namespace least_odd_prime_factor_2023_4_plus_1_l367_367724

theorem least_odd_prime_factor_2023_4_plus_1 :
  ∃ p : ℕ, prime p ∧ ((p ≡ 1 [MOD 8]) ∧ (2023^4 + 1) % p = 0) ∧
    (∀ q : ℕ, prime q ∧ ((q ≡ 1 [MOD 8]) ∧ (2023^4 + 1) % q = 0) → p ≤ q) :=
by
  sorry

end least_odd_prime_factor_2023_4_plus_1_l367_367724


namespace find_34th_number_in_sequence_l367_367089

-- The sequence definition with the repeats every 5th term.
def sequence : ℕ → ℕ
| 0     := 1
| (n+1) := if (n + 1) % 5 = 0 then (sequence n) else (sequence n + 2)

-- The problem statement.
theorem find_34th_number_in_sequence : sequence 33 = 55 := 
    sorry

end find_34th_number_in_sequence_l367_367089


namespace number_of_triangles_is_two_l367_367832

def lengths : List ℕ := [3, 4, 7, 9]

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def count_valid_triangles (lst : List ℕ) : ℕ :=
  let combinations := lst.combinations 3
  combinations.count (λ (triple : List ℕ), satisfies_triangle_inequality triple[0] triple[1] triple[2])

theorem number_of_triangles_is_two :
  count_valid_triangles lengths = 2 :=
by
  sorry

end number_of_triangles_is_two_l367_367832


namespace ratio_of_logs_l367_367144

theorem ratio_of_logs (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 18 ∧ Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_logs_l367_367144


namespace one_minus_repeating_decimal_l367_367325

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ := x

theorem one_minus_repeating_decimal:
  ∀ (x : ℚ), x = 1/3 → 1 - x = 2/3 :=
by
  sorry

end one_minus_repeating_decimal_l367_367325


namespace minimum_value_f_when_a_eq_1_l367_367424

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - 2 * a * Real.log x + (a - 2) * x

theorem minimum_value_f_when_a_eq_1 : ∃ x ∈ set.Ioi 0, f x 1 = -2 * Real.log 2 :=
begin
  use 2,
  split,
  { norm_num, },
  { sorry, }  -- Proving this requires differentiating and finding the minimum
end

end minimum_value_f_when_a_eq_1_l367_367424


namespace max_value_of_y_l367_367344

noncomputable def y (x : ℝ) : ℝ :=
  Real.cot (x + π / 4) + Real.tan (x - π / 6) + Real.sin (x + π / 3)

theorem max_value_of_y :
  ∃ x ∈ Icc (-π / 4) 0, y x = sqrt 3 / 2 :=
by
  sorry

end max_value_of_y_l367_367344


namespace smallest_odd_abundant_number_l367_367686

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n
def proper_divisors_sum (n : ℕ) : ℕ := (Finset.filter (λ d, d ≠ n) (Finset.divisors n)).sum id
def is_abundant (n : ℕ) : Prop := proper_divisors_sum n > n

theorem smallest_odd_abundant_number : ∃ n, is_odd n ∧ is_composite n ∧ is_abundant n ∧ ∀ m, is_odd m ∧ is_composite m ∧ is_abundant m → n ≤ m := sorry

end smallest_odd_abundant_number_l367_367686


namespace curve_equation_l367_367337

noncomputable def curve_passing_condition (x y : ℝ) : Prop :=
  (∃ (f : ℝ → ℝ), f 2 = 3 ∧ ∀ (t : ℝ), (f t) * t = 6 ∧ ((t ≠ 0 ∧ f t ≠ 0) → (t, f t) = (x, y)))

theorem curve_equation (x y : ℝ) (h1 : curve_passing_condition x y) : x * y = 6 :=
  sorry

end curve_equation_l367_367337


namespace vanya_third_visit_l367_367583

-- Define the conditions as a structure
structure VisitSchedule :=
  (pattern : ℕ → bool)  -- true: visit, false: no visit
  (days_in_month : ℕ)
  (visit_times : ℕ)

-- The actual problem stated in Lean
theorem vanya_third_visit (sched : VisitSchedule)
  (h_pattern : ∀ (n : ℕ), sched.pattern n ↔ (n % 7 = 3 ∨ n % 7 = 5))
  (h_visits : sched.visit_times = 10)
  (h_days_in_month : sched.days_in_month = 29)
  :
  let first_visit_next_month := 5
  let seq_wednesdays := [5, 12, 19, 26]
  let third_visit := seq_wednesdays.nth 2
  in third_visit = some 12 :=
by
  sorry

end vanya_third_visit_l367_367583


namespace sum_of_youngest_and_oldest_l367_367161

-- Let a1, a2, a3, a4 be the ages of Janet's 4 children arranged in non-decreasing order.
-- Given conditions:
variable (a₁ a₂ a₃ a₄ : ℕ)
variable (h_mean : (a₁ + a₂ + a₃ + a₄) / 4 = 10)
variable (h_median : (a₂ + a₃) / 2 = 7)

-- Proof problem:
theorem sum_of_youngest_and_oldest :
  a₁ + a₄ = 26 :=
sorry

end sum_of_youngest_and_oldest_l367_367161


namespace group_B_same_order_l367_367211

-- Definitions for the expressions in each group
def expr_A1 := 2 * 9 / 3
def expr_A2 := 2 + 9 * 3

def expr_B1 := 36 - 9 + 5
def expr_B2 := 36 / 6 * 5

def expr_C1 := 56 / 7 * 5
def expr_C2 := 56 + 7 * 5

-- Theorem stating that Group B expressions have the same order of operations
theorem group_B_same_order : (expr_B1 = expr_B2) := 
  sorry

end group_B_same_order_l367_367211


namespace solve_equation_l367_367979

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 := by
  sorry

end solve_equation_l367_367979


namespace four_digit_numbers_count_l367_367435

def num_four_digit_numbers := 5 * 5 * 8 * 7

theorem four_digit_numbers_count :
  (number of four-digit whole numbers such that the leftmost digit is odd, the second digit is even, and all four digits are different) = 1400 :=
by
  unfold num_four_digit_numbers
  sorry

end four_digit_numbers_count_l367_367435


namespace number_of_multiples_of_23_l367_367262

-- Define the sequence a(n, k)
def a (n k : ℕ) : ℕ := 2^(n - 1) * (n + 2 * k - 2)

-- Main theorem statement
theorem number_of_multiples_of_23 : 
  (finset.card (finset.filter (λ p : ℕ × ℕ, 23 ∣ a p.1 p.2) 
                               (finset.univ.image (λ n, (finset.range (101 - n)).image (λ k, (n, k)))))) = 9 :=
sorry

end number_of_multiples_of_23_l367_367262


namespace point_reflection_y_axis_l367_367151

open Point

def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem point_reflection_y_axis (A : Point) (hA : A = Point.mk (-1) 2) :
  reflect_y_axis A = Point.mk 1 2 :=
by
  rw [hA]
  simp [reflect_y_axis]
  sorry

end point_reflection_y_axis_l367_367151


namespace triangle_expression_negative_l367_367825

theorem triangle_expression_negative {a b c : ℝ} (habc : a > 0 ∧ b > 0 ∧ c > 0) (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a^2 + b^2 - c^2 - 2 * a * b < 0 :=
sorry

end triangle_expression_negative_l367_367825


namespace problem_statement_l367_367095

-- Definitions of lines and planes
variables {a β : Line} {b : Plane}

-- The conditions given in the problem
axiom a_ne_beta : a ≠ β
axiom a_ne_b : a ≠ b 

-- The proof statements that need to be shown as false
theorem problem_statement (h1: a ∥ a) (h2 : a ∥ β) : ¬ (a ∥ β) ∧ 
  (∀ h1 : a ⊥ a, ∀ h2 : b ∥ a, ¬ (a ⊥ b)) ∧ 
  (∀ h1 : a ⊥ a, ∀ h2 : b ∥ a ∧ b ⊆ β, ¬ (a ⊥ β)) ∧ 
  (∀ h1 : a ⊥ a, ∀ h2 : b ⊥ β, ∀ h3 : a ∥ β, ¬ (a ∥ b)) := by
  sorry

end problem_statement_l367_367095


namespace tan_C_right_triangle_trisected_l367_367927

theorem tan_C_right_triangle_trisected (A B C D E : Type)
  [right_triangle ABC B]
  (trisect: trisect_angle ABC B A D E)
  (h : DE/CE = 8/15) :
  tan (angle ACB) = 1 :=
sorry

end tan_C_right_triangle_trisected_l367_367927


namespace Sum_a_b_c_eq_zero_l367_367440

-- Definitions as per conditions
variables {a b c : ℝ}
def f (x : ℝ) : ℝ := a * x + b
def f_inv (x : ℝ) : ℝ := b * x + a + c

-- Proof statement
theorem Sum_a_b_c_eq_zero (h1 : ∀ x : ℝ, f (f_inv x) = x) : a + b + c = 0 :=
by
  sorry

end Sum_a_b_c_eq_zero_l367_367440


namespace scientific_notation_correct_l367_367623

def number_in_scientific_notation : ℝ := 1600000
def expected_scientific_notation : ℝ := 1.6 * 10^6

theorem scientific_notation_correct :
  number_in_scientific_notation = expected_scientific_notation := by
  sorry

end scientific_notation_correct_l367_367623


namespace perimeter_div_a_of_intersection_l367_367294

theorem perimeter_div_a_of_intersection (a : ℝ) (h : a > 0) :
  (let square_vertices := [(-a, -a), (a, -a), (-a, a), (a, a)],
       line_eq := λ x, -x / 3,
       intersection_points := [(a, -a / 3), (-a, a / 3)],
       distance := λ p1 p2, real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2),
       side1 := abs (a - a / 3),
       side2 := abs (-a + a / 3),
       side3 := abs (2 * a),
       diagonal := distance (a, -a / 3) (-a, a / 3),
       perimeter := 2 * side1 + side3 + diagonal)
  in perimeter / a = (8 + 2 * real.sqrt 10) / 3 := sorry

end perimeter_div_a_of_intersection_l367_367294


namespace least_three_digit_with_factors_correct_l367_367593

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end least_three_digit_with_factors_correct_l367_367593


namespace each_person_saves_per_month_l367_367086

theorem each_person_saves_per_month 
  (t : ℕ) (DP : ℕ) 
  (h_t : t = 3)
  (h_DP : DP = 108000):
  let months := t * 12 in
  let monthly_savings := DP / months in
  let each_person_savings := monthly_savings / 2 in
  each_person_savings = 1500 := sorry

end each_person_saves_per_month_l367_367086


namespace dice_sum_impossible_l367_367311

theorem dice_sum_impossible (a b c d : ℕ) (h1 : a * b * c * d = 216)
  (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
  (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) : 
  a + b + c + d ≠ 18 :=
sorry

end dice_sum_impossible_l367_367311


namespace min_width_l367_367928

theorem min_width (w : ℝ) (h : w * (w + 20) ≥ 150) : w ≥ 10 := by
  sorry

end min_width_l367_367928


namespace range_of_a_l367_367426

-- Define the function f
def f (a x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := -3 * x^2 + 2 * a * x - 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_prime a x ≤ 0) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end range_of_a_l367_367426


namespace measure_angle_and_geometric_locus_l367_367804

-- Definitions of the conditions
variables (h1 x14 x45 x12 s1 : Type) [HasAngle x12 s1]
variables (P1 P2 P4 P5 : Type)
variables (h1V h1'' : Type)
variables (axis : Type)

-- Given conditions as hypotheses
axiom h1_symmetry : h1 -- h1 is the line of symmetry
axiom transformation : (x14 * x45) -- Transformation according to x14 and x45
axiom maintain_symmetry : ∀ A B, (A = B → h1 = h1) -- h1 remains the line of symmetry
axiom symmetry_condition : h1V = h1'' -- h1^V is equivalent to h1''

-- Proof goal
theorem measure_angle_and_geometric_locus 
  (h_symmetry : h1_symmetry) 
  (trans : transformation) 
  (maintain_sym : maintain_symmetry)
  (sym_cond : symmetry_condition) :
  angle x12 s1 = 30 ∧ 
  (tangential_planes P1 P2 P4 P5 axis).
Proof
  sorry

end measure_angle_and_geometric_locus_l367_367804


namespace time_addition_and_sum_l367_367475

noncomputable def time_after_addition (hours_1 minutes_1 seconds_1 hours_2 minutes_2 seconds_2 : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_seconds := seconds_1 + seconds_2
  let extra_minutes := total_seconds / 60
  let result_seconds := total_seconds % 60
  let total_minutes := minutes_1 + minutes_2 + extra_minutes
  let extra_hours := total_minutes / 60
  let result_minutes := total_minutes % 60
  let total_hours := hours_1 + hours_2 + extra_hours
  let result_hours := total_hours % 12
  (result_hours, result_minutes, result_seconds)

theorem time_addition_and_sum :
  let current_hours := 3
  let current_minutes := 0
  let current_seconds := 0
  let add_hours := 300
  let add_minutes := 55
  let add_seconds := 30
  let (final_hours, final_minutes, final_seconds) := time_after_addition current_hours current_minutes current_seconds add_hours add_minutes add_seconds
  final_hours + final_minutes + final_seconds = 88 :=
by
  sorry

end time_addition_and_sum_l367_367475


namespace minimum_value_of_w_l367_367683

noncomputable def w (x y : ℝ) : ℝ := 3 * x ^ 2 + 3 * y ^ 2 + 9 * x - 6 * y + 27

theorem minimum_value_of_w : (∃ x y : ℝ, w x y = 20.25) := sorry

end minimum_value_of_w_l367_367683


namespace distinct_pawns_5x5_l367_367026

theorem distinct_pawns_5x5 : 
  ∃ n : ℕ, n = 14400 ∧ 
  (∃ (get_pos : Fin 5 → Fin 5), function.bijective get_pos) :=
begin
  sorry
end

end distinct_pawns_5x5_l367_367026


namespace height_of_prism_l367_367148

variables (a α H : ℝ)

theorem height_of_prism (ha : 0 < a) (hα : 0 < α ∧ α < π) :
  H = (a / 6) * real.sqrt (real.cot (α / 2)^2 + 3) :=
sorry

end height_of_prism_l367_367148


namespace carlos_singles_l367_367464

open Real

def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 6
def non_single_hits : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_single_hits

theorem carlos_singles :
  singles = 34 ∧ singles.toReal / total_hits.toReal * 100 = 75.56 := by
sorry

end carlos_singles_l367_367464


namespace f_monotonic_intervals_g_greater_than_4_3_l367_367429

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem f_monotonic_intervals :
  (∀ x < -1, ∀ y < -1, x < y → f x > f y) ∧ 
  (∀ x > -1, ∀ y > -1, x < y → f x < f y) :=
sorry

theorem g_greater_than_4_3 (x : ℝ) (h : x > 0) : g x > (4 / 3) :=
sorry

end f_monotonic_intervals_g_greater_than_4_3_l367_367429


namespace dot_product_AD_AB_correct_l367_367079

noncomputable def dot_product_AD_AB (A B C D : Point)
  (hD_midpoint : midpoint B C D)
  (hAB : dist A B = 2)
  (hBC : dist B C = 3)
  (hAC : dist A C = 4) : Float :=
  sorry

theorem dot_product_AD_AB_correct (A B C D : Point)
  (hD_midpoint : midpoint B C D)
  (hAB : dist A B = 2)
  (hBC : dist B C = 3)
  (hAC : dist A C = 4) :
  dot_product_AD_AB A B C D hD_midpoint hAB hBC hAC = 19 / 4 :=
by sorry

end dot_product_AD_AB_correct_l367_367079


namespace distance_from_point_P_to_base_line_l367_367565

-- Definitions based on the problem conditions
structure EquilateralTriangle :=
(side_length : ℝ)

-- Assumptions
def triangle := EquilateralTriangle.mk 1
def base_height (t : EquilateralTriangle) : ℝ := (t.side_length * Real.sqrt 3) / 2

-- Lean statement for the problem
theorem distance_from_point_P_to_base_line (t : EquilateralTriangle) (rotated_angle : ℝ) :
  rotated_angle = 60 → 
  base_height t = (Real.sqrt 3) / 2 :=
by
  -- We would normally provide the proof here, but it is omitted as per instructions.
  sorry

end distance_from_point_P_to_base_line_l367_367565


namespace cube_surface_area_l367_367209

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) : 6 * (edge_length * edge_length) = 2400 := by
  -- We state our theorem and assumptions here
  sorry

end cube_surface_area_l367_367209


namespace cos2θ_over_1_plus_sin2θ_l367_367808

theorem cos2θ_over_1_plus_sin2θ (x y : ℝ) (h₁ : x = 3) (h₂ : y = 4) :
  let θ := real.atan2 y x in
  (real.cos (2 * θ) / (1 + real.sin (2 * θ))) = -1 / 7 :=
by
  -- initial values from conditions
  let θ := real.atan2 y x
  have h₃ : x^2 + y^2 = 3^2 + 4^2 := by sorry -- proof steps are omitted
  have h₄ : real.sin θ = 4 / real.sqrt (3^2 + 4^2) := by sorry
  have h₅ : real.cos θ = 3 / real.sqrt (3^2 + 4^2) := by sorry
  have h₆ : 3^2 + 4^2 = 25 := by sorry
  specialize h₄ (3 : ℝ) (4 : ℝ) h₆
  specialize h₅ (3 : ℝ) (4 : ℝ) h₆
  have h₇ : real.sin θ = 4 / 5 := by sorry
  have h₈ : real.cos θ = 3 / 5 := by sorry
  -- using double-angle identities
  have h₉ : real.cos (2 * θ) = (real.cos θ)^2 - (real.sin θ)^2 := by sorry
  have h₁₀ : real.sin (2 * θ) = 2 * (real.sin θ) * (real.cos θ) := by sorry
  have h₁₁ : real.cos (2 * θ) = -7 / 25 := by sorry
  have h₁₂ : 1 + real.sin (2 * θ) = 49 / 25 := by sorry
  show (real.cos (2 * θ) / (1 + real.sin (2 * θ))) = -1 / 7
  from sorry

end cos2θ_over_1_plus_sin2θ_l367_367808


namespace num_equations_is_4_l367_367659

/-- Define the expressions -/
def exprs : List String := [
  "3x-5",
  "2a-3=0",
  "7>-3",
  "5-7=-2",
  "|x|=1",
  "2x^2+x=1"
]

/-- Define a predicate to check if an expression is an equation (contains '=') -/
def is_equation (e : String) : Bool := '=' ∈ e

/-- Calculate the number of equations in the list of expressions -/
def num_equations : Nat := exprs.countp is_equation

/-- The theorem statement proving the number of equations is 4 -/
theorem num_equations_is_4 : num_equations = 4 := by
  sorry

end num_equations_is_4_l367_367659


namespace initially_calculated_average_height_l367_367146

theorem initially_calculated_average_height
  (A : ℝ)
  (h1 : ∀ heights : List ℝ, heights.length = 35 → (heights.sum + (106 - 166) = heights.sum) → (heights.sum / 35) = 180) :
  A = 181.71 :=
sorry

end initially_calculated_average_height_l367_367146


namespace inequality_proof_l367_367782

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l367_367782


namespace train_crossing_time_l367_367012

theorem train_crossing_time :
  let length_of_train := 350
  let speed_in_kmph := 120 / (3600 / 1000 : ℚ) -- converting km/h to m/s
  let length_of_bridge := 980
  let total_distance := length_of_train + length_of_bridge
  let time := total_distance / speed_in_kmph
  (time ≈ 39.9) := by
sorry

end train_crossing_time_l367_367012


namespace simplify_expression_l367_367976

theorem simplify_expression :
  (sqrt 338 / sqrt 288 + sqrt 150 / sqrt 96 : ℚ) = 7 / 3 :=
by
  sorry

end simplify_expression_l367_367976


namespace sum_adjacent_cells_of_6_is_29_l367_367848

theorem sum_adjacent_cells_of_6_is_29 (table : Fin 3 × Fin 3 → ℕ)
  (uniq : Function.Injective table)
  (range : ∀ x, 1 ≤ table x ∧ table x ≤ 9)
  (pos_1 : table ⟨0, 0⟩ = 1)
  (pos_2 : table ⟨2, 0⟩ = 2)
  (pos_3 : table ⟨0, 2⟩ = 3)
  (pos_4 : table ⟨2, 2⟩ = 4)
  (adj_5 : (∑ i in ({⟨1, 0⟩, ⟨1, 2⟩, ⟨0, 1⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 9) :
  (∑ i in ({⟨0, 1⟩, ⟨1, 0⟩, ⟨1, 2⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 29 :=
by
  sorry

end sum_adjacent_cells_of_6_is_29_l367_367848


namespace Lyft_vs_Taxi_cost_difference_l367_367574

-- Define the given conditions
def Uber_cost : ℝ := 22
def Lyft_cost : ℝ := Uber_cost - 3
def Taxi_tip_percentage : ℝ := 0.20
def Taxi_total_cost_with_tip : ℝ := 18
def Taxi_original_cost : ℝ := Taxi_total_cost_with_tip / (1 + Taxi_tip_percentage)

-- Prove the final result
theorem Lyft_vs_Taxi_cost_difference : 
  Lyft_cost - Taxi_original_cost = 4 := by
  sorry

end Lyft_vs_Taxi_cost_difference_l367_367574


namespace area_of_region_bounded_by_y_eq_x_squared_and_y_eq_sqrt_x_l367_367531

noncomputable def area_between_curves : ℝ :=
  ∫ x in 0..1, (Real.sqrt x - x ^ 2)

theorem area_of_region_bounded_by_y_eq_x_squared_and_y_eq_sqrt_x :
  area_between_curves = 1 / 3 :=
by
  -- sorry allows us to skip the implementation of the proof.
  sorry

end area_of_region_bounded_by_y_eq_x_squared_and_y_eq_sqrt_x_l367_367531


namespace range_of_years_of_service_l367_367614

theorem range_of_years_of_service : 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  ∃ min max, (min ∈ years ∧ max ∈ years ∧ (max - min = 14)) :=
by 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  use 3, 17 
  sorry

end range_of_years_of_service_l367_367614


namespace triangle_area_example_l367_367834

noncomputable def area_triangle (BC AB : ℝ) (B : ℝ) : ℝ :=
  (1 / 2) * BC * AB * Real.sin B

theorem triangle_area_example
  (BC AB : ℝ) (B : ℝ)
  (hBC : BC = 2)
  (hAB : AB = 3)
  (hB : B = Real.pi / 3) :
  area_triangle BC AB B = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_example_l367_367834


namespace inequality_proof_l367_367784

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l367_367784


namespace arrange_desks_in_straight_line_l367_367596

theorem arrange_desks_in_straight_line
  (P1 P2 : Type)
  (is_point : P1 → Prop)
  (line : P2)
  (determine_line : P1 → P1 → P2)
  (front_desk back_desk middle_desk : P1) :
  is_point front_desk →
  is_point back_desk →
  is_point middle_desk →
  determine_line front_desk back_desk = line →
  (∃ p : P1, p = middle_desk → determine_line front_desk back_desk = line) →
  line = determine_line front_desk back_desk :=
by
  intros h1 h2 h3 h4 h5
  exact h4
  sorry

end arrange_desks_in_straight_line_l367_367596


namespace number_of_relatively_prime_integers_less_than_200_count_relatively_prime_to_15_or_24_less_than_200_l367_367816

theorem number_of_relatively_prime_integers_less_than_200 (n : ℕ) (h : n < 200) :
  (n % 3 ≠ 0 ∧ n % 5 ≠ 0) ∨ (n % 2 ≠ 0 ∧ n % 3 ≠ 0) := sorry

theorem count_relatively_prime_to_15_or_24_less_than_200 :
  finset.card (finset.filter (λ n, (n % 3 ≠ 0 ∧ n % 5 ≠ 0) ∨ (n % 2 ≠ 0 ∧ n % 3 ≠ 0)) (finset.range 200)) = 120 := sorry

end number_of_relatively_prime_integers_less_than_200_count_relatively_prime_to_15_or_24_less_than_200_l367_367816


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367970

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367970


namespace students_like_both_l367_367055

theorem students_like_both {total students_apple_pie students_chocolate_cake students_none students_at_least_one students_both : ℕ} 
  (h_total : total = 50)
  (h_apple : students_apple_pie = 22)
  (h_chocolate : students_chocolate_cake = 20)
  (h_none : students_none = 17)
  (h_least_one : students_at_least_one = total - students_none)
  (h_union : students_at_least_one = students_apple_pie + students_chocolate_cake - students_both) :
  students_both = 9 :=
by
  sorry

end students_like_both_l367_367055


namespace max_perfect_squares_pairwise_products_l367_367973

theorem max_perfect_squares_pairwise_products (a b : ℕ) (h_diff : a ≠ b) : 
  let products := [a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)] in
  let is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n in
  let S := products.countp is_perfect_square in
  S ≤ 1 :=
begin
  sorry
end

end max_perfect_squares_pairwise_products_l367_367973


namespace sum_term_S2018_l367_367071

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem sum_term_S2018 :
  ∃ (a S : ℕ → ℤ),
    arithmetic_sequence a ∧ 
    sum_first_n_terms a S ∧ 
    a 1 = -2018 ∧ 
    ((S 2015) / 2015 - (S 2013) / 2013 = 2) ∧ 
    S 2018 = -2018 
:= by
  sorry

end sum_term_S2018_l367_367071


namespace smallest_c_a_l367_367567

def factorial : ℕ → ℕ
| 0        := 1
| (n + 1)  := (n + 1) * factorial n

theorem smallest_c_a  (a b c : ℕ) (h1 : a * b * c = factorial 9) (h2 : a < b) (h3 : b < c) :
  c - a = 216 :=
  sorry

end smallest_c_a_l367_367567


namespace min_english_score_l367_367665

theorem min_english_score (A B : ℕ) (h_avg_AB : (A + B) / 2 = 90) : 
  ∀ E : ℕ, ((A + B + E) / 3 ≥ 92) ↔ E ≥ 96 := by
  sorry

end min_english_score_l367_367665


namespace original_savings_l367_367609

/-- Linda spent 3/4 of her savings on furniture and the rest on a TV costing $210. 
    What were her original savings? -/
theorem original_savings (S : ℝ) (h1 : S * (1/4) = 210) : S = 840 :=
by
  sorry

end original_savings_l367_367609


namespace determine_x_ohara_quadruple_l367_367306

-- Define the integers and the formula for the extended O'Hara quadruple
def extended_ohara_quadruple (a b c x : ℤ) : Prop :=
  real.sqrt a + real.sqrt b + c^2 = x

-- Prove the specific instance of the problem
theorem determine_x_ohara_quadruple :
  extended_ohara_quadruple 9 16 3 16 :=
by
  -- Here, we would provide the proof, which is omitted with sorry
  sorry

end determine_x_ohara_quadruple_l367_367306


namespace black_cars_count_l367_367557

-- Conditions
def red_cars : ℕ := 28
def ratio_red_black : ℚ := 3 / 8

-- Theorem statement
theorem black_cars_count :
  ∃ (black_cars : ℕ), black_cars = 75 ∧ (red_cars : ℚ) / (black_cars) = ratio_red_black :=
sorry

end black_cars_count_l367_367557


namespace polynomial_roots_count_4042_l367_367755

theorem polynomial_roots_count_4042 {P : ℝ → ℝ} (hP : ∀ x, (x - 2020) * P (x + 1) = (x + 2021) * P x) (h_nonzero : ¬ ∀ x, P x = 0) :
  ∃ n : ℕ, n = 4042 ∧ (P(x) = 0).nat_degree = n :=
begin
  sorry
end

end polynomial_roots_count_4042_l367_367755


namespace required_trips_l367_367656

def truckMaxWeight : ℝ := 4 -- Maximum weight the truck can carry in tons
def smallBoxWeight : ℝ := 0.5 -- Weight of a small box in tons
def mediumBoxWeight : ℝ := 1 -- Weight of a medium box in tons
def largeBoxWeight : ℝ := 1.5 -- Weight of a large box in tons

def smallBoxCount : ℝ := 280 -- Number of small boxes
def mediumBoxCount : ℝ := 350 -- Number of medium boxes
def largeBoxCount : ℝ := 241 -- Number of large boxes

def totalWeight : ℝ := (smallBoxCount * smallBoxWeight) + (mediumBoxCount * mediumBoxWeight) + (largeBoxCount * largeBoxWeight)

def tripsNeeded : ℝ := totalWeight / truckMaxWeight

def tripsRounded : ℝ := Real.ceil(tripsNeeded)

theorem required_trips : tripsRounded = 213 := by
  -- Proof goes here
  sorry

end required_trips_l367_367656


namespace chord_line_equation_l367_367796

theorem chord_line_equation (x y : ℝ) :
  (x^2 / 16 + y^2 / 9 = 1) ∧ (∃ (x1 y1 x2 y2 : ℝ), (x1^2 / 16 + y1^2 / 9 = 1) ∧ (x2^2 / 16 + y2^2 / 9 = 1) ∧ 
  ((x1 + x2) / 2 = 2) ∧ ((y1 + y2) / 2 = 3/2)) →
  3 * x + 4 * y - 12 = 0 :=
by
  intro h,
  sorry

end chord_line_equation_l367_367796


namespace sin_is_odd_l367_367084

def g (x : ℝ) : ℝ := Real.sin x

theorem sin_is_odd : ∀ x : ℝ, g(x) = -g(-x) := 
by
  intros x
  sorry

end sin_is_odd_l367_367084


namespace cylindrical_tin_volume_l367_367612

def volume_of_cylinder (d h : ℝ) : ℝ :=
  let r := d / 2
  in 3.14159 * r^2 * h

theorem cylindrical_tin_volume :
  volume_of_cylinder 14 2 ≈ 307.88 :=
by
  sorry

end cylindrical_tin_volume_l367_367612


namespace total_wet_surface_area_correct_l367_367639

namespace Cistern

-- Define the dimensions of the cistern and the depth of the water
def length : ℝ := 10
def width : ℝ := 8
def depth : ℝ := 1.5

-- Calculate the individual surface areas
def bottom_surface_area : ℝ := length * width
def longer_side_surface_area : ℝ := length * depth * 2
def shorter_side_surface_area : ℝ := width * depth * 2

-- The total wet surface area is the sum of all individual wet surface areas
def total_wet_surface_area : ℝ := 
  bottom_surface_area + longer_side_surface_area + shorter_side_surface_area

-- Prove that the total wet surface area is 134 m^2
theorem total_wet_surface_area_correct : 
  total_wet_surface_area = 134 := 
by sorry

end Cistern

end total_wet_surface_area_correct_l367_367639


namespace product_of_b_l367_367543

noncomputable def b_product : ℤ :=
  let y1 := 3
  let y2 := 8
  let x1 := 2
  let l := y2 - y1 -- Side length of the square
  let b₁ := x1 - l -- One possible value of b
  let b₂ := x1 + l -- Another possible value of b
  b₁ * b₂ -- Product of possible values of b

theorem product_of_b :
  b_product = -21 := by
  sorry

end product_of_b_l367_367543


namespace complex_problem_l367_367409

noncomputable def z (ζ : ℂ) : ℂ := ζ

theorem complex_problem (ζ : ℂ) (h : ζ + ζ⁻¹ = 2 * real.cos (real.pi / 36)) : 
  (ζ ^ 100 + ζ ^ (-100)) = -2 * real.cos (2 * real.pi / 9) :=
by sorry

end complex_problem_l367_367409


namespace snow_volume_l367_367477

-- Defining the dimensions of the driveway and depth of the snow
def length : ℝ := 30
def width : ℝ := 4
def depth : ℝ := 3 / 4

-- The goal is to prove that the volume of snow is 90 cubic feet
theorem snow_volume : (length * width * depth) = 90 := by
  sorry

end snow_volume_l367_367477


namespace equilateral_triangle_of_arith_geo_seq_l367_367896

def triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :=
  (α + β + γ = Real.pi) ∧
  (2 * β = α + γ) ∧
  (b^2 = a * c)

theorem equilateral_triangle_of_arith_geo_seq
  (A B C : ℝ) (a b c α β γ : ℝ)
  (h1 : triangle A B C a b c α β γ)
  : (a = c) ∧ (A = B) ∧ (B = C) ∧ (a = b) :=
  sorry

end equilateral_triangle_of_arith_geo_seq_l367_367896


namespace order_of_even_function_l367_367097

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem order_of_even_function {f : ℝ → ℝ}
  (h_even : is_even f)
  (h_mono_inc : is_monotonically_increasing_on_nonneg f) :
  f (-π) > f (3) ∧ f (3) > f (-2) :=
sorry

end order_of_even_function_l367_367097


namespace isosceles_triangle_angle_l367_367061

open Real

theorem isosceles_triangle_angle (A B C D E : Point) (n : ℝ) 
  (h_isosceles : A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ dist A B = dist A C)
  (h_ratio : dist B D / dist D A = n ∧ dist B E / dist E C = n)
  (h_perp : ∠ A E = ∠ D C + 90°) :
  ∠ BAC = arctan (2 * n + 1) :=
sorry

end isosceles_triangle_angle_l367_367061


namespace relatively_prime_count_200_l367_367815

theorem relatively_prime_count_200 (n : ℕ) (h : n = 200) : 
  {(k : ℕ) | k < 200 ∧ Nat.gcd k 15 = 1 ∧ Nat.gcd k 24 = 1}.card = 53 := by 
  sorry

end relatively_prime_count_200_l367_367815


namespace marble_pairs_proof_l367_367018

def marble_bag := Finset (Fin 8)
def jessica_bag := Finset (Fin 15)
def chosen_pair (m1 m2 : Fin 8) : Prop := m1 ≠ m2
def sum_equals (m1 m2 : Fin 8) (j : Fin 15) : Prop := m1.val + m2.val + 2 = j.val + 1

theorem marble_pairs_proof : 
  ∃ (pairs : Finset ((Fin 8) × (Fin 8))), 
  ∃ (jessica_marble : Finset (Fin 15)),
  pairs.card = 56 ∧ 
  ∀ (p ∈ pairs) (j ∈ jessica_marble), 
    chosen_pair p.1 p.2 ∧ sum_equals p.1 p.2 j := 
sorry

end marble_pairs_proof_l367_367018


namespace solve_for_x_l367_367611

theorem solve_for_x (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
sorry

end solve_for_x_l367_367611


namespace intersect_sphere_circle_l367_367341

-- Define the given sphere equation
def sphere (h k l R : ℝ) (x y z : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 + (z - l)^2 = R^2

-- Define the equation of a circle in the plane x = x0 parallel to the yz-plane
def circle_in_plane (x0 y0 z0 r : ℝ) (y z : ℝ) : Prop :=
  (y - y0)^2 + (z - z0)^2 = r^2

-- Define the intersecting circle from the sphere equation in the x = c plane
def intersecting_circle (h k l c R : ℝ) (y z : ℝ) : Prop :=
  (y - k)^2 + (z - l)^2 = R^2 - (h - c)^2

-- The main proof statement
theorem intersect_sphere_circle (h k l R c x0 y0 z0 r: ℝ) :
  ∀ y z, intersecting_circle h k l c R y z ↔ circle_in_plane x0 y0 z0 r y z :=
sorry

end intersect_sphere_circle_l367_367341


namespace solve_adult_tickets_l367_367260

theorem solve_adult_tickets (A C : ℕ) (h1 : 8 * A + 5 * C = 236) (h2 : A + C = 34) : A = 22 :=
sorry

end solve_adult_tickets_l367_367260


namespace total_area_of_two_squares_l367_367192

/-- Definition of a square with a given side length --/
structure Square (side_length : ℝ) :=
  (area : ℝ := side_length * side_length)

/-- The given problem conditions and definitions --/
def ABCD := Square 12
def IJKL := Square 12

/-- The point J as the lower right corner of IJKL is at the center of ABCD --/
-- Let's implicitly use this condition in our proof statement

/-- The theorem statement for the problem --/
theorem total_area_of_two_squares : (ABCD.area + IJKL.area - 36) = 252 :=
by
  sorry

end total_area_of_two_squares_l367_367192


namespace product_of_possible_b_l367_367548

theorem product_of_possible_b (b : ℤ) (h1 : y = 3) (h2 : y = 8) (h3 : x = 2)
  (h4 : (y = 3 ∧ y = 8 ∧ x = 2 ∧ (x = b ∨ x = b)) → forms_square y y x x) :
  b = 7 ∨ b = -3 → 7 * (-3) = -21 :=
by
  sorry

end product_of_possible_b_l367_367548


namespace number_of_chocolate_boxes_l367_367120

theorem number_of_chocolate_boxes
  (x y p : ℕ)
  (pieces_per_box : ℕ)
  (total_candies : ℕ)
  (h_y : y = 4)
  (h_pieces : pieces_per_box = 9)
  (h_total : total_candies = 90) :
  x = 6 :=
by
  -- Definitions of the conditions
  let caramel_candies := y * pieces_per_box
  let total_chocolate_candies := total_candies - caramel_candies
  let x := total_chocolate_candies / pieces_per_box
  
  -- Main theorem statement: x = 6
  sorry

end number_of_chocolate_boxes_l367_367120


namespace distinct_pawn_placements_on_chess_board_l367_367040

def numPawnPlacements : ℕ :=
  5! * 5!

theorem distinct_pawn_placements_on_chess_board :
  numPawnPlacements = 14400 := by
  sorry

end distinct_pawn_placements_on_chess_board_l367_367040


namespace collinear_vectors_l367_367401

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (OA OB OP : V) (m n : ℝ)

-- Given conditions
def non_collinear (OA OB : V) : Prop :=
  ∀ (t : ℝ), OA ≠ t • OB

def collinear_points (P A B : V) : Prop :=
  ∃ (t : ℝ), P - A = t • (B - A)

def linear_combination (OP OA OB : V) (m n : ℝ) : Prop :=
  OP = m • OA + n • OB

-- The theorem statement
theorem collinear_vectors (noncol : non_collinear OA OB)
  (collinearPAB : collinear_points OP OA OB)
  (lin_comb : linear_combination OP OA OB m n) :
  m = 2 ∧ n = -1 := by
sorry

end collinear_vectors_l367_367401


namespace range_of_reciprocals_l367_367829

theorem range_of_reciprocals (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) :
  ∃ c ∈ Set.Ici (9 : ℝ), (c = (1/a + 4/b)) :=
by
  sorry

end range_of_reciprocals_l367_367829


namespace shoe_matching_probability_l367_367142

open Rat

-- Each color has a distinct number of pairs.
def black_pairs : Nat := 6
def brown_pairs : Nat := 3
def gray_pairs : Nat := 2

-- Total pairs
def total_pairs : Nat := black_pairs + brown_pairs + gray_pairs

-- Total shoes
def total_shoes : Nat := 2 * total_pairs

-- Total counts for each color
def black_shoes : Nat := 2 * black_pairs
def brown_shoes : Nat := 2 * brown_pairs
def gray_shoes : Nat := 2 * gray_pairs

-- Probabilities for selecting shoes of the same color and opposite foot.
def prob_same_color_same_foot (color: Nat) (pairs: Nat) : ℚ :=
  (color.toRat / total_shoes.toRat) * ((pairs.toRat - 1) / (total_shoes.toRat - 1))

def probability : ℚ :=
  prob_same_color_same_foot black_shoes black_pairs +
  prob_same_color_same_foot brown_shoes brown_pairs +
  prob_same_color_same_foot gray_shoes gray_pairs

theorem shoe_matching_probability :
  probability = 7/33 := by
  -- The detailed proof would be filled here
  sorry

end shoe_matching_probability_l367_367142


namespace minimum_f_l367_367511

namespace Proof

def has_unique_solution (d e f : ℕ) (ineq : d < e ∧ e < f) : Prop :=
  ∃ x y : ℕ, (2 * x + y = 2010) ∧ (y = abs (x - d) + abs (x - e) + abs (x - f))

theorem minimum_f (d e f : ℕ) (h₁ : d < e) (h₂ : e < f) (h₃: has_unique_solution d e f (and.intro h₁ h₂)) : f = 1006 := 
sorry

end Proof

end minimum_f_l367_367511


namespace blue_paint_needed_l367_367940

noncomputable def blue_paint_cans (blue_ratio : ℕ) (green_ratio : ℕ) (total_cans : ℕ) : ℕ :=
  let fraction_blue := blue_ratio / (blue_ratio + green_ratio)
  (fraction_blue * total_cans).to_nat

theorem blue_paint_needed (blue_ratio green_ratio total_cans : ℕ)
  (h_blue_ratio : blue_ratio = 5)
  (h_green_ratio : green_ratio = 3)
  (h_total_cans : total_cans = 45) :
  blue_paint_cans blue_ratio green_ratio total_cans = 28 :=
by
  sorry

end blue_paint_needed_l367_367940


namespace symmetric_points_y_axis_l367_367392

theorem symmetric_points_y_axis (a b : ℝ) (h1 : ∀ a b : ℝ, y_symmetric a (-3) 4 b ↔ a = -4 ∧ b = -3) : a + b = -7 :=
by
  have h2 := h1 a b
  cases h2 with ha hb,
  rw [ha, hb],
  norm_num

end symmetric_points_y_axis_l367_367392


namespace find_vertex_P_l367_367082

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def midpoint (p1 p2 : Point3D) : Point3D :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2,
  z := (p1.z + p2.z) / 2 }

variables (Q R : Point3D)
variables (P : Point3D)
variables (D := { x := 2, y := 6, z := -2 } : Point3D)
variables (E := { x := 1, y := 5, z := -3 } : Point3D)
variables (F := { x := 3, y := 4, z := 5 } : Point3D)

-- The goal is to prove that the coordinates of P are (2, 3, 4) given the midpoints of QR, PR, and PQ.
theorem find_vertex_P
  (hD : midpoint Q R = D)
  (hE : midpoint P R = E)
  (hF : midpoint P Q = F) :
  P = { x := 2, y := 3, z := 4 } :=
by sorry

end find_vertex_P_l367_367082


namespace cube_fraction_inequality_l367_367786

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l367_367786


namespace probability_calculations_l367_367943

/-- Define the probability that player A hits at least one shot in two attempts given player A's shooting accuracy. -/
def prob_A_hits_at_least_one_shot (acc_A : ℚ) : ℚ :=
  1 - (1 - acc_A) * (1 - acc_A)

/-- Define the probability that together players A and B make exactly three shots -/
def prob_three_shots_combined (acc_A : ℚ) (acc_B : ℚ) : ℚ :=
  (2 * acc_A * (1 - acc_A) * acc_B * acc_B) + (2 * (1 - acc_B) * acc_A * acc_A * acc_B)

/-- Player A and B conditions and correctness of hitting target percentages calculations -/
theorem probability_calculations :
  let acc_A : ℚ := 1 / 2
  let acc_B : ℚ := 3 / 4
  prob_A_hits_at_least_one_shot acc_A = 3 / 4 ∧ prob_three_shots_combined acc_A acc_B = 3 / 8 :=
by
  -- Pseudo-code for the steps to perform calculations
  -- Calculation for part (I)
  calc
    prob_A_hits_at_least_one_shot 1 / 2 = 1 - (1 - 1/2) * (1 - 1/2) : by sorry
                                    ... = 1 - 1/4 : by sorry
                                    ... = 3/4 : by sorry
  -- Calculation for part (II)
  calc
    prob_three_shots_combined 1 / 2 (3 / 4) = (2 * 1/2 * (1 - 1/2) * (3/4) * (3/4)) + (2 * (1 - 3/4) * (1/2)^2 * (3/4)) : by sorry
                                           ... = 9/32 + 3/32 : by sorry
                                           ... = 3/8 : by sorry
  sorry

end probability_calculations_l367_367943


namespace sum_of_two_longest_altitudes_l367_367822

-- Define what it means for a triangle to have sides 7, 24, 25
def is_triangle_7_24_25 (a b c : ℝ) : Prop :=
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 7 ∧ b = 25 ∧ c = 24) ∨ (a = 24 ∧ b = 7 ∧ c = 25) ∨ 
  (a = 24 ∧ b = 25 ∧ c = 7) ∨ (a = 25 ∧ b = 7 ∧ c = 24) ∨ (a = 25 ∧ b = 24 ∧ c = 7)

-- Prove the sum of the two longest altitudes in such a triangle is 31
theorem sum_of_two_longest_altitudes (a b c : ℝ) (h : is_triangle_7_24_25 a b c) :
  let h_altitude (c : ℝ) := (a * b) / c in
  (a + b) - (a * b > c ∨ b * c > a ∨ c * a > b) → ℝ :=
by
  sorry

end sum_of_two_longest_altitudes_l367_367822


namespace ratio_of_areas_l367_367579

theorem ratio_of_areas (C1 C2 : ℝ) (h : (60 / 360) * C1 = (30 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l367_367579


namespace grill_run_time_l367_367242

def time_burn (coals : ℕ) (burn_rate : ℕ) (interval : ℕ) : ℚ :=
  (coals / burn_rate) * interval

theorem grill_run_time :
  let time_a1 := time_burn 60 15 20
  let time_a2 := time_burn 75 12 20
  let time_a3 := time_burn 45 15 20
  let time_b1 := time_burn 50 10 30
  let time_b2 := time_burn 70 8 30
  let time_b3 := time_burn 40 10 30
  let time_b4 := time_burn 80 8 30
  time_a1 + time_a2 + time_a3 + time_b1 + time_b2 + time_b3 + time_b4 = 1097.5 := sorry

end grill_run_time_l367_367242


namespace find_y_value_l367_367416

noncomputable def y_value (x y z : ℝ) : Prop :=
  (2 * y = x + z) ∧
  let r := -y / (x + 1) in
  (z = r * (-y)) ∧
  let s := y / x in
  (z + 2 = s * y) ∧
  (y = 12)

theorem find_y_value (x y z : ℝ) (h1 : 2 * y = x + z)
  (h2 : let r := -y / (x + 1) in z = r * (-y))
  (h3 : let s := y / x in z + 2 = s * y):
  y = 12 :=
sorry

end find_y_value_l367_367416


namespace find_a_value_l367_367502

theorem find_a_value (U : set ℕ) (A : set ℕ) (a : ℕ) (hU : U = {1, 3, 5, 7, 9}) (hA : A = {1, abs (a - 5), 9}) (hAU : (U \ A) = {5, 7}) :
  a = 2 ∨ a = 8 := 
sorry

end find_a_value_l367_367502


namespace employees_original_number_l367_367253

noncomputable def original_employees_approx (employees_remaining : ℝ) (reduction_percent : ℝ) : ℝ :=
  employees_remaining / (1 - reduction_percent)

theorem employees_original_number (employees_remaining : ℝ) (reduction_percent : ℝ) (original : ℝ) :
  employees_remaining = 462 → reduction_percent = 0.276 →
  abs (original_employees_approx employees_remaining reduction_percent - original) < 1 →
  original = 638 :=
by
  intros h_remaining h_reduction h_approx
  sorry

end employees_original_number_l367_367253


namespace no_integer_solution_l367_367136

theorem no_integer_solution : ∀ x : ℤ, sqrt (3 * x - 2) + sqrt (2 * x - 2) + sqrt (x - 1) ≠ 3 :=
by sorry

end no_integer_solution_l367_367136


namespace sequence_arith_to_sqrt_sum_arith_sum_arith_to_a2_eq_3a1_a2_eq_3a1_to_seq_arith_l367_367780

-- First combination
theorem sequence_arith_to_sqrt_sum_arith (a : ℕ → ℕ) (h1 : ∀ n, a n = a 1 + n * 2) (h2 : a 2 = 3 * a 1) : 
  ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ := sorry

-- Second combination
theorem sum_arith_to_a2_eq_3a1 (a : ℕ → ℕ) (h1 : ∀ n, a n = a 1 + n * 2) (h2 : ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ) : 
  a 2 = 3 * a 1 := sorry

-- Third combination
theorem a2_eq_3a1_to_seq_arith (a : ℕ → ℕ) (h1 : a 2 = 3 * a 1) (h2 : ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ) : 
  ∀ n, a n = a 1 + n * 2 := sorry

end sequence_arith_to_sqrt_sum_arith_sum_arith_to_a2_eq_3a1_a2_eq_3a1_to_seq_arith_l367_367780


namespace prove_cond_2_prove_cond_3_prove_cond_1_l367_367777

-- Definitions
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def seq_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := seq_sum n + a (n + 1)

def sqrt_seq (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  real.sqrt (S n)

-- Conditions
def cond_1 (a : ℕ → ℝ) : Prop := is_arithmetic a
def cond_2 (a : ℕ → ℝ) : Prop := is_arithmetic (λ n, sqrt_seq (seq_sum a) n)
def cond_3 (a : ℕ → ℝ) : Prop := a 1 * 3 = a 2

-- Proof Statements
theorem prove_cond_2 (a : ℕ → ℝ) (h1 : cond_1 a) (h3 : cond_3 a) : cond_2 a := sorry
theorem prove_cond_3 (a : ℕ → ℝ) (h1 : cond_1 a) (h2 : cond_2 a) : cond_3 a := sorry
theorem prove_cond_1 (a : ℕ → ℝ) (h2 : cond_2 a) (h3 : cond_3 a) : cond_1 a := sorry

end prove_cond_2_prove_cond_3_prove_cond_1_l367_367777


namespace max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l367_367427

noncomputable def y (x : ℝ) (a b : ℝ) : ℝ := (Real.cos x)^2 - a * (Real.sin x) + b

theorem max_min_conditions (a b : ℝ) :
  (∃ x : ℝ, y x a b = 0 ∧ (∀ x' : ℝ, y x' a b ≤ 0)) ∧ 
  (∃ x : ℝ, y x a b = -4 ∧ (∀ x' : ℝ, y x' a b ≥ -4)) ↔ 
  (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = -2) := sorry

theorem x_values_for_max_min_a2 (k : ℤ) :
  (∀ x, y x 2 (-2) = 0 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x 2 (-2)) = -4 ↔ x = Real.pi / 2 + 2 * Real.pi * k) := sorry

theorem x_values_for_max_min_aneg2 (k : ℤ) :
  (∀ x, y x (-2) (-2) = 0 ↔ x = Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x (-2) (-2)) = -4 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) := sorry

end max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l367_367427


namespace proof_of_x_and_velocity_l367_367375

variables (a T L R x : ℝ)

-- Given condition
def given_eq : Prop := (a * T) / (a * T - R) = (L + x) / x

-- Target statement to prove
def target_eq_x : Prop := x = a * T * (L / R) - L
def target_velocity : Prop := a * (L / R)

-- Main theorem to prove the equivalence
theorem proof_of_x_and_velocity (a T L R : ℝ) : given_eq a T L R x → target_eq_x a T L R x ∧ target_velocity a T L R =
  sorry

end proof_of_x_and_velocity_l367_367375


namespace joan_original_seashells_l367_367902

-- Definitions based on the conditions
def seashells_left : ℕ := 27
def seashells_given_away : ℕ := 43

-- Theorem statement
theorem joan_original_seashells : 
  seashells_left + seashells_given_away = 70 := 
by
  sorry

end joan_original_seashells_l367_367902


namespace solve_system_of_equations_l367_367983

theorem solve_system_of_equations :
    ∀ (x y : ℝ), 
    (x^3 * y + x * y^3 = 10) ∧ (x^4 + y^4 = 17) ↔
    (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1) :=
by
    sorry

end solve_system_of_equations_l367_367983


namespace original_price_l367_367512

theorem original_price (x : ℝ) (h1 : 0.95 * x * 1.40 = 1.33 * x) (h2 : 1.33 * x = 2 * x - 1352.06) : x = 2018 := sorry

end original_price_l367_367512


namespace bretschneider_theorem_l367_367123

theorem bretschneider_theorem
  (a b c d m n : ℝ)
  (A C : ℝ)
  (h1 : m = sqrt (a^2 + c^2 - 2 * a * c * cos A))
  (h2 : n = sqrt (b^2 + d^2 - 2 * b * d * cos C))
  : m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * cos (A + C) := 
sorry

end bretschneider_theorem_l367_367123


namespace value_of_v4_l367_367200

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem value_of_v4 :
  let x := -4 in
  let v0 := 3 in
  let v1 := v0 * x + 5 in
  let v2 := v1 * x + 6 in
  let v3 := v2 * x + 79 in
  let v4 := v3 * x - 8 in
  v4 = 220 := 
by
  sorry

end value_of_v4_l367_367200


namespace least_three_digit_with_factors_correct_l367_367591

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end least_three_digit_with_factors_correct_l367_367591


namespace calculation_l367_367555

noncomputable def distance_from_sphere_center_to_plane (S P Q R : Point) (r PQ QR RP : ℝ) : ℝ := 
  let a := PQ / 2
  let b := QR / 2
  let c := RP / 2
  let s := (PQ + QR + RP) / 2
  let K := Real.sqrt (s * (s - PQ) * (s - QR) * (s - RP))
  let R := (PQ * QR * RP) / (4 * K)
  Real.sqrt (r^2 - R^2)

theorem calculation 
  (P Q R S : Point) 
  (r : ℝ) 
  (PQ QR RP : ℝ)
  (h1 : PQ = 17)
  (h2 : QR = 18)
  (h3 : RP = 19)
  (h4 : r = 25) :
  distance_from_sphere_center_to_plane S P Q R r PQ QR RP = 35 * Real.sqrt 7 / 8 → 
  ∃ (x y z : ℕ), x + y + z = 50 ∧ (x.gcd z = 1) ∧ ¬ ∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ y := 
by {
  sorry
}

end calculation_l367_367555


namespace cube_fraction_inequality_l367_367788

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l367_367788


namespace total_yards_run_in_4_games_l367_367930

theorem total_yards_run_in_4_games (malik_ypg josiah_ypg darnell_avg : ℕ) (num_games : ℕ)
  (h1 : malik_ypg = 18) (h2 : josiah_ypg = 22) (h3 : darnell_avg = 11) (h4 : num_games = 4) :
  malik_ypg * num_games + josiah_ypg * num_games + darnell_avg * num_games = 204 := 
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_yards_run_in_4_games_l367_367930


namespace intersection_distance_to_pole_l367_367891

theorem intersection_distance_to_pole (rho theta : ℝ) (h1 : rho > 0) (h2 : rho = 2 * theta + 1) (h3 : rho * theta = 1) : rho = 2 :=
by
  -- We replace "sorry" with actual proof steps, if necessary.
  sorry

end intersection_distance_to_pole_l367_367891


namespace solve_for_n_l367_367978

theorem solve_for_n (n : ℝ) : 
  (0.05 * n + 0.06 * (30 + n)^2 = 45) ↔ 
  (n = -2.5833333333333335 ∨ n = -58.25) :=
sorry

end solve_for_n_l367_367978


namespace velocity_of_point_C_l367_367370

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l367_367370


namespace sin_2theta_value_l367_367745

theorem sin_2theta_value (θ : ℝ) (h : sin θ - cos θ = 1 / 3) : sin (2 * θ) = 8 / 9 :=
sorry

end sin_2theta_value_l367_367745


namespace convex_polygon_triangle_area_lt_one_l367_367641

theorem convex_polygon_triangle_area_lt_one (polygon : ConvexPolygon) (h_sides : polygon.sides = 1985) (h_perimeter : polygon.perimeter = 2800) :
  ∃ (A B C : polygon.vertex), triangle.area A B C < 1 := 
sorry

end convex_polygon_triangle_area_lt_one_l367_367641


namespace root_in_interval_l367_367450

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 0 :=
by
  sorry

end root_in_interval_l367_367450


namespace area_of_circle_l367_367443

theorem area_of_circle (r : ℝ) (h₁ : 4 * (1 / (2 * real.pi * r)) = 2 * r) : 
  real.pi * r^2 = 1 := 
sorry

end area_of_circle_l367_367443


namespace perfect_square_divisor_probability_l367_367644

theorem perfect_square_divisor_probability :
  let n := 12!
  let total_divisors := ∏ (e : ℕ) in [10, 5, 2, 1, 1], (e + 1)
  let perfect_square_divisors := ∏ (choices : ℕ) in [6, 3, 2, 1, 1], choices
  ∃ (m n : ℕ), Nat.Coprime m n ∧ n = (perfect_square_divisors * 792) ∧ (m + n = 23) :=
by
  sorry

end perfect_square_divisor_probability_l367_367644


namespace order_of_a_b_c_l367_367381

noncomputable def f (x : ℝ) : ℝ := 2*x + real.cos x

lemma increasing {x y : ℝ} (hx : x ∈ set.Ioo (-real.pi / 2) (real.pi / 2)) (hy : y ∈ set.Ioo (-real.pi / 2) (real.pi / 2)) (hxy : x < y) : f x < f y :=
by {
  have h1 : f' x = 2 - real.sin x,
  { sorry },
  have h2 : f' x > 0,
  { sorry },
  sorry
}

-- Main theorem
theorem order_of_a_b_c : 
  let a := f (-1),
      b := f (real.pi - 2),
      c := f (real.pi - 3) in
  a < c ∧ c < b :=
by {
  let a := f (-1),
  let b := f (real.pi - 2),
  let c := f (real.pi - 3),
  have ha : -1 ∈ set.Ioo (-real.pi / 2) (real.pi / 2), { sorry },
  have hb : (real.pi - 2) ∈ set.Ioo (-real.pi / 2) (real.pi / 2), { sorry },
  have hc : (real.pi - 3) ∈ set.Ioo (-real.pi / 2) (real.pi / 2), { sorry },
  have hab : -1 < (real.pi - 3) < (real.pi - 2), { sorry },
  have h1 : f (-1) < f (real.pi - 3), { apply increasing ha hc hab.left, sorry },
  have h2 : f (real.pi - 3) < f (real.pi - 2), { apply increasing hc hb hab.right, sorry },
  exact ⟨h1, h2⟩,
}

end order_of_a_b_c_l367_367381


namespace time_is_17_seconds_l367_367225

-- Define the lengths of the two trains
def length1 := 60 -- meters
def length2 := 280 -- meters

-- Define the speeds of the two trains in kmph
def speed1_kmph := 42 -- kmph
def speed2_kmph := 30 -- kmph

-- Convert speeds from kmph to m/s
def speed1_mps : Real := speed1_kmph * 1000 / 3600 -- m/s
def speed2_mps : Real := speed2_kmph * 1000 / 3600 -- m/s

-- Calculate the relative speed in m/s
def relative_speed_mps : Real := speed1_mps + speed2_mps -- m/s

-- Calculate the total length to be covered
def total_length : Nat := length1 + length2 -- meters

-- Calculate the time in seconds
def time : Real := total_length / relative_speed_mps -- seconds

-- Prove that the time for the two trains to be clear of each other is 17 seconds
theorem time_is_17_seconds : time = 17 := 
by
  sorry

end time_is_17_seconds_l367_367225


namespace minimum_value_sincos_fraction_l367_367345

theorem minimum_value_sincos_fraction : 
  ∀ x : ℝ, ∃ m : ℝ, (∀ y : ℝ, (sin y)^8 + (cos y)^8 + 1) / ((sin y)^6 + (cos y)^6 + 1) ≥ m ∧ 
  (sin x)^8 + (cos x)^8 + 1 / ((sin x)^6 + (cos x)^6 + 1) = 17 / 8 := sorry

end minimum_value_sincos_fraction_l367_367345


namespace probability_A_wins_championship_expectation_X_is_13_l367_367967

/-
Definitions corresponding to the conditions in the problem
-/
def prob_event1_A_win : ℝ := 0.5
def prob_event2_A_win : ℝ := 0.4
def prob_event3_A_win : ℝ := 0.8

def prob_event1_B_win : ℝ := 1 - prob_event1_A_win
def prob_event2_B_win : ℝ := 1 - prob_event2_A_win
def prob_event3_B_win : ℝ := 1 - prob_event3_A_win

/-
Proof problems corresponding to the questions and correct answers
-/

theorem probability_A_wins_championship : prob_event1_A_win * prob_event2_A_win * prob_event3_A_win
    + prob_event1_A_win * prob_event2_A_win * prob_event3_B_win
    + prob_event1_A_win * prob_event2_B_win * prob_event3_A_win 
    + prob_event1_B_win * prob_event2_A_win * prob_event3_A_win = 0.6 := 
sorry

noncomputable def X_distribution_table : list (ℝ × ℝ) := 
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

noncomputable def expected_value_X : ℝ := 
  ∑ x in X_distribution_table, x.1 * x.2

theorem expectation_X_is_13 : expected_value_X = 13 := sorry

end probability_A_wins_championship_expectation_X_is_13_l367_367967


namespace distance_pq_l367_367466

noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

#eval dist (⟨1 * real.cos (real.pi / 6), 1 * real.sin (real.pi / 6)⟩)
            (⟨2 * real.cos (real.pi / 2), 2 * real.sin (real.pi / 2)⟩)
-- This #eval statement evaluates the distance for illustration purposes. To hide it in a formal proof, just remove it.

theorem distance_pq : dist (⟨1 * real.cos (real.pi / 6), 1 * real.sin (real.pi / 6)⟩)
                           (⟨2 * real.cos (real.pi / 2), 2 * real.sin (real.pi / 2)⟩) = real.sqrt 3 :=
by
  sorry

end distance_pq_l367_367466


namespace proof_op_nabla_l367_367824

def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem proof_op_nabla :
  op_nabla (op_nabla (1/2) (1/3)) (1/4) = 9 / 11 := by
  sorry

end proof_op_nabla_l367_367824


namespace eval_expr_l367_367701

theorem eval_expr (a b : ℕ) (ha : a = 3) (hb : b = 4) : (a^b)^a - (b^a)^b = -16245775 := by
  sorry

end eval_expr_l367_367701


namespace area_EFGH_is_one_third_area_ABCD_l367_367150

variables {A B C D E F G H: Type}
variables [ConvexQuadrilateral A B C D]
variables (points_on_AB : DividesThreeEqualParts A B E F)
variables (points_on_CD : DividesThreeEqualParts C D G H)

theorem area_EFGH_is_one_third_area_ABCD 
  (h_AB : ∀ {A B E F}, DividesThreeEqualParts A B E F)
  (h_CD : ∀ {C D G H}, DividesThreeEqualParts C D G H) :
  area (quadrilateral E F G H) = (1/3) * area (quadrilateral A B C D) :=
by
  sorry

end area_EFGH_is_one_third_area_ABCD_l367_367150


namespace sum_of_squares_inequality_l367_367434

theorem sum_of_squares_inequality (n : ℕ) (h : n ≥ 3) (a : Fin n → ℝ) :
  ∑ i, a i ^ 2 ≥ (2 / (n - 1)) * ∑ (i j : Fin n) (h : i < j), a i * a j :=
sorry

end sum_of_squares_inequality_l367_367434


namespace other_discount_percentage_l367_367159

theorem other_discount_percentage :
  ∃ x : ℝ, 
  let price := 70 in
  let discounted_price_after_10 := price * 0.9 in
  let final_price := 61.11 in
  let discount := 1 - x / 100 in
  (discounted_price_after_10 * discount = final_price) → x = 3 := 
by
  sorry

end other_discount_percentage_l367_367159


namespace lucas_average_speed_l367_367215

theorem lucas_average_speed :
  ∀ (initial_reading final_reading : ℕ) (time_period : ℕ),
    initial_reading = 27472 →
    final_reading = 28482 →
    time_period = 4 →
    (final_reading - initial_reading) / time_period = 252.5 := 
begin
  sorry
end

end lucas_average_speed_l367_367215


namespace school_A_win_prob_expectation_X_is_13_l367_367960

-- Define the probabilities of school A winning individual events
def pA_event1 : ℝ := 0.5
def pA_event2 : ℝ := 0.4
def pA_event3 : ℝ := 0.8

-- Define the probability of school A winning the championship
def pA_win_championship : ℝ :=
  (pA_event1 * pA_event2 * pA_event3) +
  (pA_event1 * (1 - pA_event2) * pA_event3) +
  (pA_event1 * pA_event2 * (1 - pA_event3)) +
  ((1 - pA_event1) * pA_event2 * pA_event3)

-- Proof statement for the probability of school A winning the championship
theorem school_A_win_prob : pA_win_championship = 0.6 := sorry

-- Define the distribution and expectation for school B's total score
def X_prob : ℝ → ℝ
| 0  := (1 - pA_event1) * (1 - pA_event2) * (1 - pA_event3)
| 10 := pA_event1 * (1 - pA_event2) * (1 - pA_event3) +
        (1 - pA_event1) * pA_event2 * (1 - pA_event3) +
        (1 - pA_event1) * (1 - pA_event2) * pA_event3
| 20 := pA_event1 * pA_event2 * (1 - pA_event3) +
        pA_event1 * (1 - pA_event2) * pA_event3 +
        (1 - pA_event1) * pA_event2 * pA_event3
| 30 := pA_event1 * pA_event2 * pA_event3
| _  := 0

def expected_X : ℝ :=
  0 * X_prob 0 +
  10 * X_prob 10 +
  20 * X_prob 20 +
  30 * X_prob 30

-- Proof statement for the expectation of school B's total score
theorem expectation_X_is_13 : expected_X = 13 := sorry

end school_A_win_prob_expectation_X_is_13_l367_367960


namespace sum_adjacent_cells_of_6_is_29_l367_367846

theorem sum_adjacent_cells_of_6_is_29 (table : Fin 3 × Fin 3 → ℕ)
  (uniq : Function.Injective table)
  (range : ∀ x, 1 ≤ table x ∧ table x ≤ 9)
  (pos_1 : table ⟨0, 0⟩ = 1)
  (pos_2 : table ⟨2, 0⟩ = 2)
  (pos_3 : table ⟨0, 2⟩ = 3)
  (pos_4 : table ⟨2, 2⟩ = 4)
  (adj_5 : (∑ i in ({⟨1, 0⟩, ⟨1, 2⟩, ⟨0, 1⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 9) :
  (∑ i in ({⟨0, 1⟩, ⟨1, 0⟩, ⟨1, 2⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 29 :=
by
  sorry

end sum_adjacent_cells_of_6_is_29_l367_367846


namespace product_of_possible_b_l367_367547

theorem product_of_possible_b (b : ℤ) (h1 : y = 3) (h2 : y = 8) (h3 : x = 2)
  (h4 : (y = 3 ∧ y = 8 ∧ x = 2 ∧ (x = b ∨ x = b)) → forms_square y y x x) :
  b = 7 ∨ b = -3 → 7 * (-3) = -21 :=
by
  sorry

end product_of_possible_b_l367_367547


namespace math_proof_problem_l367_367423

-- Definition of the given function
def f (x : ℝ) := x * Real.exp x

-- Definitions of derivatives
noncomputable def f_0 (x : ℝ) := (deriv f) x
noncomputable def iterate_deriv (n : ℕ) : ℝ → ℝ := Nat.iterate deriv n f
noncomputable def f_n (n : ℕ) (x : ℝ) := (iterate_deriv n) x

-- Definitions of conditions
def condition1 : Prop := ∃ x, x = -1 ∧ f_0 x = 0
def condition3 : Prop := ∀ x, f_n 2015 x = x * Real.exp x + 2017 * Real.exp x

-- Lean statement encapsulating the above conditions
theorem math_proof_problem : condition1 ∧ condition3 := 
by
  sorry

end math_proof_problem_l367_367423


namespace sum_k_div_a_k_l367_367006

noncomputable def a : ℕ → ℝ
| 1       := 1
| (n + 1) := (n + 1) * a n / (2 * n + a n)

theorem sum_k_div_a_k : (∑ k in Finset.range 2017, (k + 1) / a (k + 1)) = 2^2018 - 2019 :=
by
  sorry

end sum_k_div_a_k_l367_367006


namespace solution_l367_367245

noncomputable def die1 : Finset ℕ := {1, 2, 3, 3, 4, 4}.toFinset
noncomputable def die2 : Finset ℕ := {2, 3, 5, 6, 7, 8}.toFinset

def probability_target_sum : ℚ :=
(let outcomes := (die1.product die2).toFinset in
 let favorable_outcomes := outcomes.filter (λ (x : ℕ × ℕ), (x.1 + x.2 = 6) ∨ (x.1 + x.2 = 8) ∨ (x.1 + x.2 = 10)) in
 favorable_outcomes.card.toRat / outcomes.card.toRat)

theorem solution : probability_target_sum = 11 / 36 :=
by sorry

end solution_l367_367245


namespace cookies_per_child_l367_367837

theorem cookies_per_child (total_cookies : ℕ) (adults : ℕ) (children : ℕ) (fraction_eaten_by_adults : ℚ) 
  (h1 : total_cookies = 120) (h2 : adults = 2) (h3 : children = 4) (h4 : fraction_eaten_by_adults = 1/3) :
  total_cookies * (1 - fraction_eaten_by_adults) / children = 20 := 
by
  sorry

end cookies_per_child_l367_367837


namespace masha_wins_game_l367_367116

/-- Masha and Rita are playing a game with three piles of candies. 
    Prove that Masha wins the game given the initial conditions. -/
theorem masha_wins_game :
  let piles := [10, 20, 30],
      total_moves := 10 - 1 + (20 - 1) + (30 - 1)
  in (total_moves % 2 = 1) → Masha_wins :=
by
  -- The definitions and conditions described above imply the necessary arithmetic
  intros
  sorry

end masha_wins_game_l367_367116


namespace marks_in_social_studies_l367_367131

-- Define Shekar's marks in various subjects.
def marks_in_mathematics : ℕ := 76
def marks_in_science : ℕ := 65
def marks_in_english : ℕ := 62
def marks_in_biology : ℕ := 85
def average_marks : ℕ := 74

-- Prove that Shekar's marks in social studies are 82.
theorem marks_in_social_studies (
  marks_in_mathematics = 76 ∧
  marks_in_science = 65 ∧
  marks_in_english = 62 ∧
  marks_in_biology = 85 ∧
  average_marks = 74
  ) : 
  (5 * average_marks) - (marks_in_mathematics + marks_in_science + marks_in_english + marks_in_biology) = 82 :=
by
  sorry

end marks_in_social_studies_l367_367131


namespace med_eq_mode_lt_mean_l367_367900

theorem med_eq_mode_lt_mean (data : List ℕ) (hd : data = [2, 5, 3, 5, 6, 3, 7, 3, 5, 2, 1, 6]) :
  let mean := (1 + 2 + 2 + 3 + 3 + 3 + 5 + 5 + 5 + 6 + 6 + 7) / 12 in
  let median := (3 + 3) / 2 in
  let mode := 3 in
  median = mode ∧ mode < mean := by
  sorry

end med_eq_mode_lt_mean_l367_367900


namespace range_of_m_l367_367771

noncomputable def f (x : ℝ) : ℝ :=
(x - 1/e)^2 + 1 / (x - 1/e)^2 

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ , f x + m * real.sqrt (f x + 2) ≥ 0) ↔ m ≥ -1 := 
sorry

end range_of_m_l367_367771


namespace distinct_balls_boxes_l367_367017

def count_distinct_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 7 ∧ boxes = 3 then 8 else 0

theorem distinct_balls_boxes :
  count_distinct_distributions 7 3 = 8 :=
by sorry

end distinct_balls_boxes_l367_367017


namespace test_line_equation_1_test_line_equation_2_l367_367754

noncomputable def line_equation_1 : Prop :=
  ∃ a b : ℝ, (M : ℝ × ℝ) = (-2, 1) ∧ A = (a, 0) ∧ B = (0, b) ∧
    (M.1 = (a + 0) / 2 ∧ M.2 = (0 + b) / 2) ∧
    (l : ℝ × ℝ → Prop) = (λ p, p.1 - 2 * p.2 + 4 = 0)

noncomputable def line_equation_2 : Prop :=
  ∃ a b : ℝ, (M : ℝ × ℝ) = (-2, 1) ∧ A = (a, 0) ∧ B = (0, b) ∧
    (M.1 = (2 * (0 - a / 3)) ∨ M.2 = (3 * (b - 1))) ∧
    (l : ℝ × ℝ → Prop) = (λ p, p.1 - 4 * p.2 + 6 = 0) ∨
    (l : ℝ × ℝ → Prop) = (λ p, p.1 + 4 * p.2 - 2 = 0)

-- Test cases
theorem test_line_equation_1 : line_equation_1 := by
  sorry

theorem test_line_equation_2 : line_equation_2 := by
  sorry

end test_line_equation_1_test_line_equation_2_l367_367754


namespace no_positive_integer_solutions_l367_367540

theorem no_positive_integer_solutions :
  ¬ ∃ (x1 x2 : ℕ), 903 * x1 + 731 * x2 = 1106 := by
  sorry

end no_positive_integer_solutions_l367_367540


namespace min_value_2x_4y_on_line_l367_367411

-- Define conditions and goal in Lean
theorem min_value_2x_4y_on_line (x y : ℝ) (h : x + 2 * y = 3) : 
  2^x + 4^y ≥ 4 * real.sqrt 2 :=
sorry

end min_value_2x_4y_on_line_l367_367411


namespace maximum_perimeter_proof_l367_367452

noncomputable def maximum_perimeter_of_triangle (O A B C : ℝ) (r : ℝ) 
  (h1: O = 0) 
  (h2: r = 1) 
  (h3: (B - O) * (C - O) = -1 / 2) 
  (h4: ∠BAC = 60) 
  : ℝ :=
3 * Real.sqrt 3

theorem maximum_perimeter_proof {O A B C : ℝ} 
  (r : ℝ) 
  (hO : O = 0) 
  (h_r : r = 1) 
  (h_dot_product : (B - O) * (C - O) = -1 / 2) 
  (h_angle_A : ∠BAC = 60) 
  : maximum_perimeter_of_triangle O A B C r hO h_r h_dot_product h_angle_A = 3 * Real.sqrt 3 :=
  by sorry

end maximum_perimeter_proof_l367_367452


namespace correct_function_is_f4_l367_367269

open Function

-- Define the functions as per the conditions
def f1 (x : ℝ) : ℝ := ln (x ^ 3)
def f2 (x : ℝ) : ℝ := -x ^ 2
def f3 (x : ℝ) : ℝ := -1 / x
def f4 (x : ℝ) : ℝ := x * abs x

-- Define being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define being an increasing function
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Prove that f4 is both odd and increasing
theorem correct_function_is_f4 : is_odd f4 ∧ is_increasing f4 := by {
  sorry
}

end correct_function_is_f4_l367_367269


namespace carpet_required_l367_367118

noncomputable def feet_to_yards (feet : ℝ) : ℝ := feet / 3

def bedroom_length_feet : ℝ := 15
def bedroom_width_feet : ℝ := 10
def closet_side_feet : ℝ := 6
def wastage_percentage : ℝ := 0.10

def bedroom_length_yards : ℝ := feet_to_yards bedroom_length_feet
def bedroom_width_yards : ℝ := feet_to_yards bedroom_width_feet
def closet_side_yards : ℝ := feet_to_yards closet_side_feet

def bedroom_area_yards : ℝ := bedroom_length_yards * bedroom_width_yards
def closet_area_yards : ℝ := closet_side_yards * closet_side_yards

def total_area_without_wastage_yards : ℝ := bedroom_area_yards + closet_area_yards
def total_area_with_wastage_yards : ℝ := total_area_without_wastage_yards * (1 + wastage_percentage)

theorem carpet_required : total_area_with_wastage_yards = 22.715 :=
sorry

end carpet_required_l367_367118


namespace complex_multiplication_l367_367405

-- Define the imaginary unit
def i : ℂ := 0 + 1 * complex.I

-- Condition that i^2 = -1
lemma i_squared_eq_neg_one : i * i = -1 := by
  rcases complex.I_def with ⟨_, h_real, h_im⟩,
  rw h_im,
  simp [complex.mul, complex.mul_re, complex.mul_im, complex.zero_re, complex.zero_im, 
       complex.one_re, complex.one_im, h_real, complex.zero_sub, complex.add_zero, complex.neg_zero, h_real],
  simp only [complex.of_real_zero, neg_zero, add_zero, sub_zero, zero_add, of_real_neg, zero_im, sub_zero],

-- Goal: (2+i)(3+i) = 5 + 5i
theorem complex_multiplication : (2 + i) * (3 + i) = 5 + 5 * i :=
by
  sorry

end complex_multiplication_l367_367405


namespace find_digital_photo_frames_l367_367130

theorem find_digital_photo_frames 
  (cost_camera : ℤ)
  (num_cameras : ℤ)
  (cost_frame : ℤ)
  (discount : ℚ)
  (total_paid : ℤ)
  (x : ℤ) : 
  (0.95 * (num_cameras * cost_camera + x * cost_frame) = total_paid) 
  → x = 3 := 
by 
  sorry

-- Define the constants according to the given conditions
def cost_camera : ℤ := 110
def num_cameras : ℤ := 2
def cost_frame : ℤ := 120
def discount : ℚ := 0.95
def total_paid : ℤ := 551

/-
  We pass the conditions to the theorem and assert that the specific x, which satisfies 
  the equation 0.95 * (num_cameras * cost_camera + x * cost_frame) = total_paid, equals 3.
-/
example : find_digital_photo_frames cost_camera num_cameras cost_frame discount total_paid 3 := 
by 
  sorry

end find_digital_photo_frames_l367_367130


namespace inscribed_circles_tangent_l367_367939

variable (a b c : ℝ) (A B C M : Type)

-- Define the side lengths of the triangle
def sideBC := a
def sideAC := b
def sideAB := c

-- Define the desired condition for point M on BC
def BM : ℝ := (a + c - b) / 2

theorem inscribed_circles_tangent (ABC : Triangle a b c) :
  ∃ M : Point, BM = (a + c - b) / 2 ∧ 
  tangent_point (inscribed_circle (Triangle A B M)) 
  = tangent_point (inscribed_circle (Triangle A C M)) 
:= by
  sorry

end inscribed_circles_tangent_l367_367939


namespace cost_of_one_pencil_and_one_pen_l367_367536

variables (x y : ℝ)

def eq1 := 4 * x + 3 * y = 3.70
def eq2 := 3 * x + 4 * y = 4.20

theorem cost_of_one_pencil_and_one_pen (h₁ : eq1 x y) (h₂ : eq2 x y) :
  x + y = 1.1286 :=
sorry

end cost_of_one_pencil_and_one_pen_l367_367536


namespace central_circle_unique_constant_sum_value_no_odd_in_middle_sides_sum_of_diagonal_equals_unique_solution_l367_367465

noncomputable def arrangement (numbers : List ℕ) := sorry

def condition (numbers : List ℕ) : Prop :=
  all sets of 3 consecutive numbers have the same sum sorry

theorem central_circle_unique (numbers : List ℕ) (k : ℕ) :
  condition numbers → (central_position numbers = k) → k = 7 :=
sorry

theorem constant_sum_value (numbers : List ℕ) (S : ℕ) :
  condition numbers → (sum_of_any_3_consecutive_numbers numbers = S) → S = 21 :=
sorry

theorem no_odd_in_middle_sides (numbers : List ℕ) :
  condition numbers → ¬ (∃ n, odd n ∧ middle_sides numbers = n) :=
sorry

theorem sum_of_diagonal_equals (numbers : List ℕ) :
  condition numbers → sum_diagonal numbers = 21 :=
sorry

theorem unique_solution (numbers : List ℕ) :
  condition numbers → 
  ∃! arrangement numbers, 
    (condition arrangement) :=
sorry

end central_circle_unique_constant_sum_value_no_odd_in_middle_sides_sum_of_diagonal_equals_unique_solution_l367_367465


namespace pieces_cut_from_rod_l367_367606

-- Defining the given conditions in Lean
def meters_to_centimeters (m : ℝ) : ℝ := m * 100
def pieces_from_rod (rod_length_cm piece_length_cm : ℝ) : ℝ := rod_length_cm / piece_length_cm

-- Stating the problem in Lean
theorem pieces_cut_from_rod : pieces_from_rod (meters_to_centimeters 42.5) 85 = 50 :=
by
  sorry

end pieces_cut_from_rod_l367_367606


namespace monk_same_height_l367_367631

noncomputable def f : ℝ → ℝ := 
  sorry -- Function representing monk's height during ascent from 6 AM to 6 PM

noncomputable def g : ℝ → ℝ := 
  sorry -- Function representing monk's height during descent from 6 AM to 2 PM following day

theorem monk_same_height (f_cont : continuous_on f (set.Icc 6 18)) 
                          (g_cont : continuous_on g (set.Icc 6 14)) : 
  ∃ t ∈ set.Icc (6 : ℝ) (14 : ℝ), f t = g t :=
begin
  -- Here we would apply the Intermediate Value Theorem to show that the function
  -- h(t) = f(t) - g(t) must have a zero within the interval [6 AM, 2 PM]
  sorry
end

end monk_same_height_l367_367631


namespace tangent_line_at_point_l367_367999

noncomputable def tangent_line_equation (x : ℝ) : Prop :=
  ∀ y : ℝ, y = x * (3 * Real.log x + 1) → (x = 1 ∧ y = 1) → y = 4 * x - 3

theorem tangent_line_at_point : tangent_line_equation 1 :=
sorry

end tangent_line_at_point_l367_367999


namespace students_not_in_biology_l367_367607

theorem students_not_in_biology (total_students : ℕ) (percent_in_biology : ℝ) (students_in_biology : ℕ) :
  total_students = 880 →
  percent_in_biology = 0.30 →
  students_in_biology = 0.30 * 880 →
  (total_students - students_in_biology) = 616 :=
by
  intros h1 h2 h3
  have h4 : total_students - students_in_biology = 880 - 264, by
    rw [h1, h3]
    norm_num
  exact h4
  sorry

end students_not_in_biology_l367_367607


namespace evaluate_expression_l367_367693

theorem evaluate_expression (a b : ℕ) (h₁ : a = 3) (h₂ : b = 4) : ((a^b)^a - (b^a)^b) = -16246775 :=
by
  rw [h₁, h₂]
  sorry

end evaluate_expression_l367_367693


namespace placement_of_pawns_l367_367035

-- Define the size of the chessboard and the total number of pawns
def board_size := 5
def total_pawns := 5

-- Define the problem statement
theorem placement_of_pawns : 
  (∑ (pawns : Finset (Fin total_pawns → Fin board_size)), 
    (∀ p1 p2 : Fin total_pawns, p1 ≠ p2 → pawns(p1) ≠ pawns(p2)) ∧ -- distinct positions
    (∀ i j : Fin total_pawns, i ≠ j → pawns(i) ≠ pawns(j)) ∧ -- no same row/column
    pawns.card = total_pawns) = 14400 :=
sorry

end placement_of_pawns_l367_367035


namespace number_of_moles_na2so4_l367_367347

theorem number_of_moles_na2so4 :
  (1 : ℕ) * H₂SO₄ + (2 : ℕ) * NaOH ⟶ (1 : ℕ) * Na₂SO₄ + (2 : ℕ) * H₂O →
  (1 * H₂SO₄ = 1) ∧ (2 * NaOH = 2) →
  number_of_moles_na2so4 = 1 := 
sorry

end number_of_moles_na2so4_l367_367347


namespace total_yards_run_l367_367932

theorem total_yards_run (Malik_yards_per_game : ℕ) (Josiah_yards_per_game : ℕ) (Darnell_yards_per_game : ℕ) (games : ℕ) 
  (hM : Malik_yards_per_game = 18) (hJ : Josiah_yards_per_game = 22) (hD : Darnell_yards_per_game = 11) (hG : games = 4) : 
  Malik_yards_per_game * games + Josiah_yards_per_game * games + Darnell_yards_per_game * games = 204 := by
  sorry

end total_yards_run_l367_367932


namespace sum_distinct_complex_numbers_l367_367763

theorem sum_distinct_complex_numbers (a b : ℂ) (h_distinct : a ≠ b) (h_nonzero : a * b ≠ 0)
  (h_set_eq : {a, b} = {a^2, b^2}) : a + b = -1 := by
  sorry

end sum_distinct_complex_numbers_l367_367763


namespace probability_sum_6_game_rule_not_fair_l367_367174

-- Definitions based on conditions
def balls : List ℕ := [1, 2, 3, 4, 5]

def outcomes : List (ℕ × ℕ) := (balls.product balls)

def sum_is_6 (x : ℕ × ℕ) : Prop := x.1 + x.2 = 6

def is_even (n : ℕ) : Prop := n % 2 = 0

def a_wins (x : ℕ × ℕ) : Prop := is_even (x.1 + x.2)

-- Probability of an event
def probability (event : List (ℕ × ℕ)) (total : List (ℕ × ℕ)) : ℚ :=
(event.length : ℚ) / (total.length : ℚ)

-- Necessary events
def event_sum_6 : List (ℕ × ℕ) := outcomes.filter sum_is_6

def event_a_wins : List (ℕ × ℕ) := outcomes.filter a_wins 

-- Proof statement 1: Probability that A wins and the sum is 6
theorem probability_sum_6 : probability event_sum_6 outcomes = 1 / 5 := sorry

-- Proof statement 2: Game rule fairness
theorem game_rule_not_fair : probability event_a_wins outcomes ≠ 1 / 2 := sorry

end probability_sum_6_game_rule_not_fair_l367_367174


namespace parallelogram_angle_B_l367_367885

theorem parallelogram_angle_B (A B C D : Type) [parallelogram A B C D] (hA : ∠A = 50) : ∠B = 130 :=
sorry

end parallelogram_angle_B_l367_367885


namespace placement_of_pawns_l367_367036

-- Define the size of the chessboard and the total number of pawns
def board_size := 5
def total_pawns := 5

-- Define the problem statement
theorem placement_of_pawns : 
  (∑ (pawns : Finset (Fin total_pawns → Fin board_size)), 
    (∀ p1 p2 : Fin total_pawns, p1 ≠ p2 → pawns(p1) ≠ pawns(p2)) ∧ -- distinct positions
    (∀ i j : Fin total_pawns, i ≠ j → pawns(i) ≠ pawns(j)) ∧ -- no same row/column
    pawns.card = total_pawns) = 14400 :=
sorry

end placement_of_pawns_l367_367036


namespace triangle_inequality_l367_367094

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l367_367094


namespace exists_disjoint_subsets_with_equal_sum_l367_367098

theorem exists_disjoint_subsets_with_equal_sum (A : Finset ℕ) (n : ℕ)
  (h1 : A.card = n)
  (h2 : A.sum id < 2^n - 1) :
  ∃ (B C: Finset ℕ), B ≠ ∅ ∧ C ≠ ∅ ∧ disjoint B C ∧ B.sum id = C.sum id := 
  sorry

end exists_disjoint_subsets_with_equal_sum_l367_367098


namespace john_trip_time_30_min_l367_367903

-- Definitions of the given conditions
variables {D : ℝ} -- Distance John traveled
variables {T : ℝ} -- Time John took
variable (T_john : ℝ) -- Time it took John (in hours)
variable (T_beth : ℝ) -- Time it took Beth (in hours)
variable (D_john : ℝ) -- Distance John traveled (in miles)
variable (D_beth : ℝ) -- Distance Beth traveled (in miles)

-- Given conditions
def john_speed := 40 -- John's speed in mph
def beth_speed := 30 -- Beth's speed in mph
def additional_distance := 5 -- Additional distance Beth traveled in miles
def additional_time := 1 / 3 -- Additional time Beth took in hours

-- Proving the time it took John to complete the trip is 30 minutes (0.5 hours)
theorem john_trip_time_30_min : 
  ∀ (T_john T_beth : ℝ), 
    T_john = (D) / john_speed →
    T_beth = (D + additional_distance) / beth_speed →
    (T_beth = T_john + additional_time) →
    T_john = 1 / 2 :=
by
  intro T_john T_beth
  sorry

end john_trip_time_30_min_l367_367903


namespace area_of_circle_below_line_l367_367721

noncomputable def area_below_line (r d : ℝ) : ℝ :=
  r^2 * real.arccos (d / r) - d * real.sqrt (r^2 - d^2)

theorem area_of_circle_below_line :
  let r := 8 in  -- radius of the circle
  let d := 2 in  -- vertical distance from the center to the line
  let area := area_below_line r d in
  (x - 3)^2 + (y - 10)^2 = r^2 →  -- circle's equation
  y = d + 8 →                    -- line's equation
  area ≈ 68.86 :=               -- area approximation
by
  sorry

end area_of_circle_below_line_l367_367721


namespace find_y_l367_367487

def oslash (a b : ℝ) : ℝ :=
  (sqrt (2 * a + b))^3

theorem find_y (y : ℝ) (h : oslash 9 y = 125) : y = 7 := by
  sorry

end find_y_l367_367487


namespace coefficient_x4_of_square_l367_367005

theorem coefficient_x4_of_square (q : Polynomial ℝ) (hq : q = Polynomial.X^5 - 4 * Polynomial.X^2 + 3) :
  (Polynomial.coeff (q * q) 4 = 16) :=
by {
  sorry
}

end coefficient_x4_of_square_l367_367005


namespace angle_BHC_l367_367080

noncomputable def triangle_ABC := 
  {A B C : Type*}
  [triangle_ABC : ∃ (α β γ : ℝ), α + β + γ = 180 ∧ α = 55 ∧ γ = 17]

structure Orthocenter (α β γ : ℝ) := 
  (H : Type*)

theorem angle_BHC (α β γ : ℝ) (H : Orthocenter α β γ) : 
  γ = 90 := 
sorry

end angle_BHC_l367_367080


namespace recipe_flour_requirement_l367_367111

def sugar_cups : ℕ := 9
def salt_cups : ℕ := 40
def flour_initial_cups : ℕ := 4
def additional_flour : ℕ := sugar_cups + 1
def total_flour_cups : ℕ := additional_flour

theorem recipe_flour_requirement : total_flour_cups = 10 := by
  sorry

end recipe_flour_requirement_l367_367111


namespace truck_distance_l367_367264

theorem truck_distance (d: ℕ) (g: ℕ) (eff: ℕ) (new_g: ℕ) (total_distance: ℕ)
  (h1: d = 300) (h2: g = 10) (h3: eff = d / g) (h4: new_g = 15) (h5: total_distance = eff * new_g):
  total_distance = 450 :=
sorry

end truck_distance_l367_367264


namespace ant_probability_l367_367271

-- Variables for the initial and target positions.
variables (start : ℕ × ℕ) (target : ℕ × ℕ)

-- Conditions: Starting at point A and moving for 6 minutes.
def initial_position := (0, 0)  -- Point labeled A
def target_position := (-2, 0)  -- Point labeled C
def time_steps := 6

-- Definition for movement, considering diagonal moves as well.
def neighbors (pos : ℕ × ℕ) : set (ℕ × ℕ) :=
  {(x + dx, y + dy) | dx, dy ∈ [-1, 0, 1], (dx ≠ 0 ∨ dy ≠ 0) ∧ (x, y) = pos}

-- Probability calculation.
-- This assumes each dot has an equal probability to be chosen next.
noncomputable def probability_of_reaching_target : ℝ :=
  begin
    /- Reasoning must be built here, but we assert the result for now -/
    exact 1 / 2,
  end

-- The theorem asserting the required probability equality.
theorem ant_probability :
  probability_of_reaching_target initial_position target_position time_steps = 1 / 2 :=
begin
  sorry -- Proof to be constructed
end

end ant_probability_l367_367271


namespace fraction_of_students_paired_l367_367880

theorem fraction_of_students_paired {t s : ℕ} 
  (h1 : t / 4 = s / 3) : 
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by sorry

end fraction_of_students_paired_l367_367880


namespace triangle_dot_product_l367_367076

/-- Let $ABC$ be a triangle with $D$ being the midpoint of $BC$.
    If $AB = 2$, $BC = 3$, and $AC = 4$, then the dot product 
    $\overrightarrow{AD} \cdot \overrightarrow{AB}$ is $\frac{19}{4}$.
-/
theorem triangle_dot_product (A B C D : Point) 
  (hD : midpoint D B C) 
  (hAB : dist A B = 2) 
  (hBC : dist B C = 3) 
  (hAC : dist A C = 4) : 
  (AD • AB) = 19 / 4 :=
sorry

-- Definitions to make the code syntactically correct
structure Point :=
(x : ℝ) (y : ℝ)

noncomputable def midpoint (D B C : Point) : Prop :=
D = ⟨(B.x + C.x)/2, (B.y + C.y)/2⟩

noncomputable def dist (P Q : Point) :=
real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

noncomputable def dot_product (v w : Vector) := sorry -- Placeholder for the actual implementation

end triangle_dot_product_l367_367076


namespace HB0_parallel_PQ_l367_367461

-- Definitions based on conditions
variables {A B C H O P Q B_0 : Type*}

-- The triangle is acute and not isosceles
axiom acute_not_isosceles (A B C : Type*) : ¬ isosceles A B C ∧ acute A B C

-- Definitions of points and their relationships
def altitude (A1 C1 : Type*) (triangle : Type*) : Prop := sorry

axiom AA1_CC1_altitudes (A1 C1 H : Type*) (triangle : Type*) : altitude A1 C1 triangle

axiom H_intersection_of_altitudes (H : Type*) : true

def circumcenter (O : Type*) (triangle : Type*) : Prop := sorry

axiom O_is_circumcenter (O : Type*) (triangle : Type*) : circumcenter O triangle

def midpoint (B0 A C : Type*) : Prop := sorry

axiom B0_is_midpoint_of_AC (B0 A C : Type*) : midpoint B0 A C

axiom BO_intersects_AC_at_P (B O P A C : Type*) : true

def line_intersection (Q BH A1C1 : Type*) : Prop := sorry

axiom BH_A1C1_intersection_at_Q (Q BH A1C1 : Type*) : line_intersection Q BH A1C1

-- Main statement to be proved in Lean 4
theorem HB0_parallel_PQ (A B C H O P Q B_0 A1 C1 : Type*)
  (h1 : acute_not_isosceles A B C) 
  (h2 : AA1_CC1_altitudes A1 C1 H (triangle A B C))
  (h3 : H_intersection_of_altitudes H) 
  (h4 : O_is_circumcenter O (triangle A B C))
  (h5 : B0_is_midpoint_of_AC B_0 A C) 
  (h6 : BO_intersects_AC_at_P B O P A C)
  (h7 : BH_A1C1_intersection_at_Q Q (line B H) (line A1 C1)) :
  parallel (line H B_0) (line P Q) :=
sorry

end HB0_parallel_PQ_l367_367461


namespace slope_of_line_l367_367729

theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (∃ m b : ℝ, y = m * x + b ∧ m = -4/7) :=
by
   intro h
   use -4/7, 28/7
   rw [mul_comm 7 y, ←sub_eq_iff_eq_add]
   simp [eq_sub_of_add_eq, h]
   split
   { sorry }
   { refl }

end slope_of_line_l367_367729


namespace tangent_line_at_point_l367_367996

noncomputable theory

open Real

def f (x : ℝ) : ℝ := x * (3 * log x + 1)

def f_deriv (x : ℝ) : ℝ := deriv f x

theorem tangent_line_at_point :
  f 1 = 1 ∧ f_deriv 1 = 4 → ∀ x : ℝ, (1 : ℝ) - 1 = 4 * (x - 1) → (4 : ℝ) * x - 3 = x * (3 * log x + 1) :=
by
  sorry

end tangent_line_at_point_l367_367996


namespace rhombus_perimeter_is_80_l367_367560

-- Definitions of the conditions
def rhombus_diagonals_ratio : Prop := ∃ (d1 d2 : ℝ), d1 / d2 = 3 / 4 ∧ d1 + d2 = 56

-- The goal is to prove that given the conditions, the perimeter of the rhombus is 80
theorem rhombus_perimeter_is_80 (h : rhombus_diagonals_ratio) : ∃ (p : ℝ), p = 80 :=
by
  sorry  -- The actual proof steps would go here

end rhombus_perimeter_is_80_l367_367560


namespace math_contest_students_l367_367986

theorem math_contest_students (n : ℝ) (h : n / 3 + n / 4 + n / 5 + 26 = n) : n = 120 :=
by {
    sorry
}

end math_contest_students_l367_367986


namespace a_seq_general_term_b_seq_sum_l367_367758

-- Definitions of the sequences and conditions
def a_seq (n : ℕ) : ℕ :=
if n = 1 then 1 else 2 * a_seq (n - 1) + 1

axiom a_4 : a_seq 4 = 15

-- Goal 1: Prove the general term formula for sequence {a_n}
theorem a_seq_general_term (n : ℕ) : a_seq n = 2^n - 1 :=
sorry

-- Definition and sum for sequence {b_n}
def b_seq (n : ℕ) : ℝ := n / (a_seq n + 1)
def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, b_seq (i + 1)

-- Goal 2: Prove the sum of the first n terms of sequence {b_n}
theorem b_seq_sum (n : ℕ) : S_n n = 2 - (n + 2) / 2^n :=
sorry

end a_seq_general_term_b_seq_sum_l367_367758


namespace percentage_of_muslim_boys_is_44_l367_367460

def total_boys : ℕ := 850
def percent_hindus : ℕ := 14
def percent_sikhs : ℕ := 10
def boys_other_communities : ℕ := 272

def count_hindus : ℕ := (percent_hindus * total_boys) / 100
def count_sikhs : ℕ := (percent_sikhs * total_boys) / 100
def count_hindus_and_sikhs : ℕ := count_hindus + count_sikhs
def count_muslims : ℕ := total_boys - count_hindus_and_sikhs - boys_other_communities
def percent_muslims : Real := (count_muslims * 100.0) / total_boys

theorem percentage_of_muslim_boys_is_44
  : percent_muslims = 44 := by
    sorry

end percentage_of_muslim_boys_is_44_l367_367460


namespace real_roots_exist_l367_367124

theorem real_roots_exist (a b c : ℝ) :
  ∃ x : ℝ, 3 * x^2 - 2 * (a + b + c) * x + (a * b + b * c + c * a) = 0 :=
begin
  let Δ := 4 * (a^2 + b^2 + c^2 - (a * b + b * c + c * a)),
  have h : Δ ≥ 0,
  { sorry },
  use (some value of x). -- Need to use some method to show an explicit x or properties such as the quadratic formula solution technique.
  sorry, -- Skipping the explicit proof, this line needs the actual proof implementation.
end

end real_roots_exist_l367_367124


namespace first_player_wins_with_min_prime_count_l367_367196

theorem first_player_wins_with_min_prime_count :
  ∃ n : ℕ, n = 3 ∧ ∃ primes : list ℕ,
    (∀ p ∈ primes, nat.prime p ∧ p ≤ 100) ∧
    (∀ p1 p2 : ℕ, p1 ∈ primes → p2 ∈ primes → p1 ≠ p2 → list.index_of p1 primes < list.index_of p2 primes → (p1 % 10 = p2 / 10)) ∧
    (primes.length = n) ∧
    (∀ p1 p2 : ℕ, p1 ∈ primes → p2 ∈ primes → p1 ≠ p2 → p1 % 10 = p2 / 10) :=
begin
  sorry
end

end first_player_wins_with_min_prime_count_l367_367196


namespace percentage_exploded_mid_year_l367_367252

-- Initial number of volcanoes
def V : ℕ := 200

-- Number of volcanoes exploded in first two months
def exploded_first_two_months : ℕ := 0.2 * V

-- Remaining volcanoes after first two months
def R1 : ℕ := V - exploded_first_two_months

-- Percentage explosion by mid-year
variable (x : ℝ)

-- Number of volcanoes exploded by mid-year
def exploded_mid_year := 0.01 * x * R1

-- Remaining volcanoes after mid-year
def R2 := R1 - exploded_mid_year

-- Number of volcanoes exploded by end of year
def exploded_end_year := 0.5 * R2

-- Remaining volcanoes at the end of the year
def intact_end_year : ℕ := 48

theorem percentage_exploded_mid_year :
  exploded_mid_year x + exploded_end_year x + exploded_first_two_months = V - intact_end_year →
  x = 40 := 
sorry

end percentage_exploded_mid_year_l367_367252


namespace point_in_third_quadrant_l367_367069

theorem point_in_third_quadrant (m : ℝ) : 
  (-1 < 0 ∧ -2 + m < 0) ↔ (m < 2) :=
by 
  sorry

end point_in_third_quadrant_l367_367069


namespace commission_8000_l367_367276

variable (C k : ℝ)

def commission_5000 (C k : ℝ) : Prop := C + 5000 * k = 110
def commission_11000 (C k : ℝ) : Prop := C + 11000 * k = 230

theorem commission_8000 
  (h1 : commission_5000 C k) 
  (h2 : commission_11000 C k)
  : C + 8000 * k = 170 :=
sorry

end commission_8000_l367_367276


namespace repeating_decimal_as_fraction_l367_367707

theorem repeating_decimal_as_fraction :
  let x := 0.36 + 0.0036 + 0.000036 + 0.00000036 + ∑' (n : ℕ), (0.0036 : ℝ) * ((1 / 100) ^ n)
  in x = (4 : ℝ) / (11 : ℝ) := 
by {
  let x : ℝ := ∑' (n : ℕ), (36 : ℝ) / ((10^2 : ℝ) * (10 ^ (2 * n))),
  have hx : x = (4 : ℝ) / (11 : ℝ), 
  sorry,
}

end repeating_decimal_as_fraction_l367_367707


namespace product_of_possible_b_l367_367546

theorem product_of_possible_b (b : ℤ) (h1 : y = 3) (h2 : y = 8) (h3 : x = 2)
  (h4 : (y = 3 ∧ y = 8 ∧ x = 2 ∧ (x = b ∨ x = b)) → forms_square y y x x) :
  b = 7 ∨ b = -3 → 7 * (-3) = -21 :=
by
  sorry

end product_of_possible_b_l367_367546


namespace time_to_15_feet_above_bottom_l367_367233

variable (R : ℝ) (T : ℝ) (h : ℝ → ℝ)

def ferris_wheel_height (t : ℝ) : ℝ := R * Real.cos((2 * Real.pi / T) * t) + R

theorem time_to_15_feet_above_bottom 
  (h_eq : ∀ t, h t = ferris_wheel_height R T t)
  (R_eq : R = 30) 
  (T_eq : T = 90) :
  ∃ t : ℝ, h t = 15 ∧ t = 30 := 
by
  sorry

end time_to_15_feet_above_bottom_l367_367233


namespace hexagon_area_l367_367472

theorem hexagon_area {R x y z : ℝ} (hR : R = 10) (hx : x + y + z = 45) :
  let X Y Z : ℝ := 10
  let XY'Z : ℝ := (5 * (x + y + z)) / 2
  in XY'Z = 112.5 :=
by sorry

end hexagon_area_l367_367472


namespace sum_adjacent_to_six_l367_367856

theorem sum_adjacent_to_six :
  ∀ (table : fin 3 × fin 3 → ℕ),
    (∀ i j, table i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∃! i j, table i j = 1) ∧
    (∃! i j, table i j = 2) ∧
    (∃! i j, table i j = 3) ∧
    (∃! i j, table i j = 4) ∧
    (∃! i j, table i j = 5) → 
    (∀ i j, 
      table i j = 5 → 
        let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                       (if i < 2 then table (i+1, j) else 0) + 
                       (if j > 0 then table (i, j-1) else 0) + 
                       (if j < 2 then table (i, j+1) else 0)
        in adj_sum = 9) →
    (∃ i j, table i j = 6 ∧
      let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                     (if i < 2 then table (i+1, j) else 0) + 
                     (if j > 0 then table (i, j-1) else 0) + 
                     (if j < 2 then table (i, j+1) else 0) 
      in adj_sum = 29) := sorry

end sum_adjacent_to_six_l367_367856


namespace exists_polynomial_l367_367256

def satisfies_condition (a : ℕ → ℝ) :=
  ∀ m : ℕ, m ≥ sufficiently_large → ∑ n in Finset.range (m + 1), a n * (-1)^n * (Nat.choose m n) = 0

def polynomial_exists (a : ℕ → ℝ) :=
  ∃ P : Polynomial ℝ, ∀ n : ℕ, a n = P.eval n

theorem exists_polynomial (a : ℕ → ℝ) :
  satisfies_condition a → polynomial_exists a :=
by
  intro h
  -- proof to be filled in
  sorry

end exists_polynomial_l367_367256


namespace smallest_x_for_convex_distortion_l367_367915

structure Hexagon (x : ℝ) :=
(vertices : fin 6 → ℝ × ℝ)
(side_length : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = x)

def is_distortion (H : Hexagon) (H' : Hexagon) : Prop :=
∀ i, dist (H.vertices i) (H'.vertices i) < 1

def is_convex (H : Hexagon) : Prop :=
∀ i j k, ccw (H.vertices i) (H.vertices j) (H.vertices k) ∨
         ccw (H.vertices j) (H.vertices k) (H.vertices i) ∨
         ccw (H.vertices k) (H.vertices i) (H.vertices j)

theorem smallest_x_for_convex_distortion (x : ℝ) :
  (∀ H H', H.side_length = x → is_distortion H H' → is_convex H') ↔ x ≥ 4 := 
by
  sorry

end smallest_x_for_convex_distortion_l367_367915


namespace sequence_2015_l367_367385

noncomputable def a : ℕ → ℚ
| 0     := 1 / 2
| (n+1) := 1 / (1 - a n)

theorem sequence_2015 : a 2014 = 2 := 
sorry

end sequence_2015_l367_367385


namespace uphill_integers_divisible_by_14_count_l367_367279

def isUphillInteger (n : ℕ) : Prop :=
  let digits := toDigits 10 n
  ∀ i, i < digits.length - 1 → digits.get (i + 1) > digits.get i

def isDivisibleBy14 (n : ℕ) : Prop :=
  n % 14 = 0

def countUphillIntegersDivisibleBy14 : ℕ :=
  (List.range (10^6)).count (λ n, isUphillInteger n ∧ isDivisibleBy14 n)

theorem uphill_integers_divisible_by_14_count :
  countUphillIntegersDivisibleBy14 = 2 := sorry

end uphill_integers_divisible_by_14_count_l367_367279


namespace cute_5_digit_integer_is_1_l367_367270

def is_cute (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5]
  ∃ (a b c d e : ℕ), 
  [a, b, c, d, e] = digits.permute
  ∧ a = e ∧ b = d
  ∧ (∀ k, k ∈ {1, 2, 3, 4, 5} → (digits.take k).foldl (λ x d, x * 10 + d) 0 % k = 0)

def count_cute_5_digit_integers : ℕ :=
  if h : ∃ n : ℕ, is_cute n then
    1
  else
    0

theorem cute_5_digit_integer_is_1 :
  count_cute_5_digit_integers = 1 := by
  sorry

end cute_5_digit_integer_is_1_l367_367270


namespace unique_real_solution_l367_367356

theorem unique_real_solution (k : ℝ) :
  (∀ x : ℝ, (3 * x + 4) + (x - 3) ^ 2 = 3 + k * x → x = (- b - sqrt d) / (2 * a) ∨ x = (- b + sqrt d) / (2 * a))
  ↔ (k = -3 + 2 * sqrt 10 ∨ k = -3 - 2 * sqrt 10) :=
by
  sorry

end unique_real_solution_l367_367356


namespace units_digit_of_7_power_exp_is_1_l367_367736

-- Define the periodicity of units digits of powers of 7
def units_digit_seq : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit_power_7 (n : ℕ) : ℕ :=
  units_digit_seq.get! (n % 4)

-- Define the exponent
def exp : ℕ := 8^5

-- Define the modular operation result
def exp_modulo : ℕ := exp % 4

-- Define the main statement
theorem units_digit_of_7_power_exp_is_1 :
  units_digit_power_7 exp = 1 :=
by
  simp [units_digit_power_7, units_digit_seq, exp, exp_modulo]
  sorry

end units_digit_of_7_power_exp_is_1_l367_367736


namespace quadrilateral_parallelogram_l367_367221

def divides_area (D : Diagonal) (Q : Quadrilateral) : Prop := 
  sorry

def is_parallelogram (Q : Quadrilateral) : Prop :=
  sorry

theorem quadrilateral_parallelogram (ABCD : Quadrilateral) 
  (h : ∀ D : Diagonal, divides_area(D, ABCD)) : is_parallelogram(ABCD) :=
sorry

end quadrilateral_parallelogram_l367_367221


namespace unique_solution_abs_eq_l367_367014

theorem unique_solution_abs_eq : 
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 :=
by
  use -5
  sorry

end unique_solution_abs_eq_l367_367014


namespace distance_between_DM_and_BN_l367_367467

open Real
open EuclideanSpace

noncomputable def distance_between_skew_lines (a : ℝ) : ℝ :=
  let B := (2 * a, -2 * a, 0)
  let D := (-2 * a, 2 * a, 0)
  let M := (-a, -a, sqrt 14 * a)
  let N := (a, a, sqrt 14 * a)
  let BN := (a - 2 * a, a + 2 * a, sqrt 14 * a - 0)
  let DM := (-a - (-2 * a), -a - 2 * a, sqrt 14 * a - 0)
  let n := (3, 1, 0)
  let MN := (2 * a, 2 * a, 0)
  abs ((2 * a * 3 + 2 * a * 1) / sqrt (3 ^ 2 + 1 ^ 2))

theorem distance_between_DM_and_BN (a : ℝ) : distance_between_skew_lines a = 4 * sqrt 10 * a / 5 :=
  sorry

end distance_between_DM_and_BN_l367_367467


namespace sum_adjacent_6_is_29_l367_367843

-- Define the grid and the placement of numbers 1 to 4
structure Grid :=
  (grid : Fin 3 → Fin 3 → Nat)
  (h_unique : ∀ i j, grid i j ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_distinct : Function.Injective (λ (i : Fin 3) (j : Fin 3), grid i j))
  (h_placement : grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧ grid 2 2 = 4)

-- Define the condition of the sum of numbers adjacent to 5 being 9
def sum_adjacent_5 (g : Grid) : Prop :=
  let (i, j) := (0, 1) in -- Position for number 5
  (g.grid (i.succ) j + g.grid (i.succ.pred) j + g.grid i (j.succ) + g.grid i (j.pred)) = 9

-- Define the main theorem
theorem sum_adjacent_6_is_29 (g : Grid) (h_sum_adj_5 : sum_adjacent_5 g) : 
  (g.grid 1 0 + g.grid 1 2 + g.grid 0 1 + g.grid 2 1 = 29) := sorry

end sum_adjacent_6_is_29_l367_367843


namespace find_real_numbers_l367_367715

theorem find_real_numbers (x y : ℝ) (h₁ : x + y = 3) (h₂ : x^5 + y^5 = 33) :
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end find_real_numbers_l367_367715


namespace amount_of_water_formed_l367_367304

noncomputable def na_hso3_moles : ℝ := 3
noncomputable def hcl_moles : ℝ := 4
def reaction_stoichiometry (na_hso3 hcl : ℝ) : ℝ := na_hso3 / 2

noncomputable def theoretical_yield_h2o (na_hso3 : ℝ) : ℝ :=
reaction_stoichiometry na_hso3 na_hso3 * 2

noncomputable def percent_yield : ℝ := 0.80

noncomputable def actual_yield_h2o (theoretical_yield : ℝ) : ℝ :=
percent_yield * theoretical_yield

theorem amount_of_water_formed :
  actual_yield_h2o (theoretical_yield_h2o na_hso3_moles) = 2.4 :=
sorry

end amount_of_water_formed_l367_367304


namespace sum_coefficients_odd_powers_l367_367072

-- Define the binomial expansion of (1 - x)^11
noncomputable def binom_expansion (x : ℝ) : ℝ :=
  (1 - x)^11

-- The theorem statement about the sum of coefficients of the terms with odd powers of x in the expansion
theorem sum_coefficients_odd_powers (x : ℝ) :
  (let series := binom_expansion x in ∑ k in finset.range 12, 
  if odd k then series.coeff (series.nat_degree - k) else 0) = -2^10 := sorry

end sum_coefficients_odd_powers_l367_367072


namespace triangle_area_l367_367223

def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area : heron_area 26 22 10 ≈ 107.76 :=
by
  unfold heron_area
  -- Semi-perimeter calculation
  have s : ℝ := (26 + 22 + 10) / 2
  have s_eq : s = 29 := by norm_num
  rw [s_eq]
  -- Heron's formula application
  have area_eq : real.sqrt (29 * (29 - 26) * (29 - 22) * (29 - 10)) = 107.76 := by norm_num
  exact area_eq

end triangle_area_l367_367223


namespace triangle_area_half_l367_367011

/-- Given a triangle ABC with |AC| > |BC|, point M lies on the angle bisector of ∠C,
and BM is perpendicular to the angle bisector, prove that the area of 
triangle AMC is half of the area of triangle ABC. -/
theorem triangle_area_half {A B C M : Type} [Triangle A B C]
  (hAC_gt_BC : |AC| > |BC|)
  (hM_on_bisector : ∃ bisector, M ∈ bisector ∧ is_angle_bisector bisector ∠C)
  (hBM_perpendicular : ∃ bisector, BM ⟂ bisector ∧ is_angle_bisector bisector ∠C) :
  area (triangle A M C) = 1 / 2 * area (triangle A B C) :=
sorry

end triangle_area_half_l367_367011


namespace spend_together_is_85_l367_367354

variable (B D : ℝ)

theorem spend_together_is_85 (h1 : D = 0.70 * B) (h2 : B = D + 15) : B + D = 85 := by
  sorry

end spend_together_is_85_l367_367354


namespace sum_of_adjacent_to_6_l367_367866

theorem sum_of_adjacent_to_6 :
  ∃ (grid : Fin 3 × Fin 3 → ℕ),
  (grid (0, 0) = 1 ∧ grid (0, 2) = 3 ∧ grid (2, 0) = 2 ∧ grid (2, 2) = 4 ∧
   ∀ i j, grid (i, j) ∈ finset.range 1 10 ∧ finset.univ.card = 9 ∧
   (grid (1, 0) + grid (1, 1) + grid (2, 1) = 9) ∧ 
   (grid (1, 1) = 6) ∧ 
   (sum_of_adjacent grid (1, 1) = 29))

where
  sum_of_adjacent (grid : Fin 3 × Fin 3 → ℕ) (x y : Fin 3 × Fin 3) : ℕ :=
  grid (x - 1, y) + grid (x + 1, y) + grid (x, y - 1) + grid (x, y + 1)
  sorry

end sum_of_adjacent_to_6_l367_367866


namespace smallest_integer_y_l367_367207

theorem smallest_integer_y (y : ℤ) (h : 7 - 5 * y < 22) : y ≥ -2 :=
by sorry

end smallest_integer_y_l367_367207


namespace isabella_jun_meetings_l367_367085

noncomputable def isabella_speed := 270 -- in m/min
noncomputable def isabella_radius := 50 -- in meters
noncomputable def isabella_circumference := 2 * Real.pi * isabella_radius -- circumference of the track

noncomputable def jun_speed := 330 -- in m/min
noncomputable def jun_radius := 60 -- in meters
noncomputable def jun_circumference := 2 * Real.pi * jun_radius -- circumference of the track

noncomputable def total_time := 40 -- in minutes

noncomputable def angular_speed (speed : ℝ) (circumference : ℝ) : ℝ :=
  (speed / circumference) * 2 * Real.pi

noncomputable def isabella_angular_speed := angular_speed isabella_speed isabella_circumference -- rad/min
noncomputable def jun_angular_speed := angular_speed jun_speed jun_circumference -- rad/min
noncomputable def relative_angular_speed := isabella_angular_speed + jun_angular_speed -- rad/min, because they're running in opposite directions

noncomputable def time_to_meet := 2 * Real.pi / relative_angular_speed -- time to meet once, in minutes

theorem isabella_jun_meetings : 
  let number_of_meetings := Real.floor (total_time / time_to_meet) 
  in number_of_meetings = 69 := sorry

end isabella_jun_meetings_l367_367085


namespace hyperbola_standard_equation_l367_367802

theorem hyperbola_standard_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h_real_axis : 2 * a = 4 * Real.sqrt 2) (h_eccentricity : a / Real.sqrt (a^2 + b^2) = Real.sqrt 6 / 2) :
    (a = 2 * Real.sqrt 2) ∧ (b = 2) → ∀ x y : ℝ, (x^2)/8 - (y^2)/4 = 1 :=
sorry

end hyperbola_standard_equation_l367_367802


namespace coefficient_x_squared_in_expansion_l367_367991

theorem coefficient_x_squared_in_expansion :
  (∃ f : ℝ[X], (3 * X - (1 / (X ^ 2 / 3))) ^ 2 = 9 * X^2 + f) :=
sorry

end coefficient_x_squared_in_expansion_l367_367991


namespace buying_beams_l367_367527

/-- Problem Statement:
Given:
1. The total money for beams is 6210 wen.
2. The transportation cost per beam is 3 wen.
3. Removing one beam means the remaining beams' total transportation cost equals the price of one beam.

Prove: 3 * (x - 1) = 6210 / x
-/
theorem buying_beams (x : ℕ) (h₁ : x > 0) (h₂ : 6210 % x = 0) :
  3 * (x - 1) = 6210 / x :=
sorry

end buying_beams_l367_367527


namespace company_employee_reduction_l367_367650

-- Definitions based on the conditions in the given problem.
def originalEmployees : ℝ := 208.04597701149424
def reductionPercentage : ℝ := 13 / 100

-- The Lean statement for proving the current number of employees.
theorem company_employee_reduction : originalEmployees - (reductionPercentage * originalEmployees) ≈ 181 := by
  sorry

end company_employee_reduction_l367_367650


namespace bananas_in_each_group_l367_367534

theorem bananas_in_each_group (total_bananas groups : ℕ) (h1 : total_bananas = 392) (h2 : groups = 196) :
    total_bananas / groups = 2 :=
by
  sorry

end bananas_in_each_group_l367_367534


namespace vegetarian_eaters_l367_367457

-- Define the conditions
theorem vegetarian_eaters : 
  ∀ (total family_size : ℕ) 
  (only_veg only_nonveg both_veg_nonveg eat_veg : ℕ), 
  family_size = 45 → 
  only_veg = 22 → 
  only_nonveg = 15 → 
  both_veg_nonveg = 8 → 
  eat_veg = only_veg + both_veg_nonveg → 
  eat_veg = 30 :=
by
  intros total family_size only_veg only_nonveg both_veg_nonveg eat_veg
  sorry

end vegetarian_eaters_l367_367457


namespace slope_of_AB_l367_367794

-- Definitions of the points A and B
def A : Prod Int Int := (0, 4)
def B : Prod Int Int := (1, 2)

-- Slope calculation definition
def slope (p1 p2 : Prod Int Int) : Int := (p2.2 - p1.2) / (p2.1 - p1.1)

-- The proof statement
theorem slope_of_AB : slope A B = -2 := by
  sorry

end slope_of_AB_l367_367794


namespace NorrisSavings_l367_367508

theorem NorrisSavings : 
  let saved_september := 29
  let saved_october := 25
  let saved_november := 31
  let saved_december := 35
  let saved_january := 40
  saved_september + saved_october + saved_november + saved_december + saved_january = 160 :=
by
  sorry

end NorrisSavings_l367_367508


namespace unit_stratified_sampling_l367_367636

theorem unit_stratified_sampling 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (selected_elderly : ℕ)
  (total : ℕ) (n : ℕ)
  (h1 : elderly = 27)
  (h2 : middle_aged = 54)
  (h3 : young = 81)
  (h4 : selected_elderly = 3)
  (h5 : total = elderly + middle_aged + young)
  (h6 : 3 / 27 = selected_elderly / elderly)
  (h7 : n / total = selected_elderly / elderly) : 
  n = 18 := 
by
  sorry

end unit_stratified_sampling_l367_367636


namespace placement_of_pawns_l367_367033

-- Define the size of the chessboard and the total number of pawns
def board_size := 5
def total_pawns := 5

-- Define the problem statement
theorem placement_of_pawns : 
  (∑ (pawns : Finset (Fin total_pawns → Fin board_size)), 
    (∀ p1 p2 : Fin total_pawns, p1 ≠ p2 → pawns(p1) ≠ pawns(p2)) ∧ -- distinct positions
    (∀ i j : Fin total_pawns, i ≠ j → pawns(i) ≠ pawns(j)) ∧ -- no same row/column
    pawns.card = total_pawns) = 14400 :=
sorry

end placement_of_pawns_l367_367033


namespace ramesh_profit_percentage_l367_367953

theorem ramesh_profit_percentage {LP : ℝ} 
  (h1 : LP * 0.80 = 12500)
  (h2 : LP * 0.20 + 125 + 250 = 12875)
  (selling_price : ℝ) 
  (h3 : selling_price = 18560) :
  (selling_price - LP) / LP * 100 ≈ 18.78 := 
by 
   have cost_price_without_discount : ℝ := LP 
   have profit : ℝ := selling_price - cost_price_without_discount
   have profit_percentage : ℝ := (profit / cost_price_without_discount) * 100
   sorry

end ramesh_profit_percentage_l367_367953


namespace regular_octoroll_into_circle_l367_367604

theorem regular_octoroll_into_circle (O C : Set Point) [regular_octagon O] [circle C] : 
  ∃ O' ⊆ O, center O' ∈ C := 
sorry

end regular_octoroll_into_circle_l367_367604


namespace problem_1_problem_2_l367_367672

theorem problem_1 : ((1 / 3 - 3 / 4 + 5 / 6) / (1 / 12)) = 5 := 
  sorry

theorem problem_2 : ((-1 : ℤ) ^ 2023 + |(1 : ℝ) - 0.5| * (-4 : ℝ) ^ 2) = 7 := 
  sorry

end problem_1_problem_2_l367_367672


namespace kids_playing_with_white_balls_l367_367059

theorem kids_playing_with_white_balls :
  ∀ (total kids_yellow kids_both : ℕ), total = 35 → kids_yellow = 28 → kids_both = 19 → 
  let W := total - kids_yellow + kids_both in
  W = 26 :=
by
  intros total kids_yellow kids_both h_total h_yellow h_both
  let W := total - kids_yellow + kids_both
  rw [h_total, h_yellow, h_both]
  sorry

end kids_playing_with_white_balls_l367_367059


namespace eval_expr_l367_367700

theorem eval_expr (a b : ℕ) (ha : a = 3) (hb : b = 4) : (a^b)^a - (b^a)^b = -16245775 := by
  sorry

end eval_expr_l367_367700


namespace solve_for_x_l367_367524

theorem solve_for_x : ∃ x : ℤ, 25 - (4 + 3) = 5 + x ∧ x = 13 :=
by {
  sorry
}

end solve_for_x_l367_367524


namespace main_theorem_1_main_theorem_2_l367_367797

-- Define the conditions of the ellipse
def ellipse_equation (x y a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Given conditions
def point_M := (sqrt 6, 1 : ℝ)
def eccentricity : ℝ := sqrt 2 / 2

def passes_through_M (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) : Prop :=
  ellipse_equation (sqrt 6) 1 a b h₁ h₂ h₃

def ellipse_problem_1 : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ 
    passes_through_M a b sorry sorry sorry ∧ 
    (sqrt ((a^2 - b^2)/(a^2))) = eccentricity ∧ 
    (a^2 = 8 ∧ b^2 = 4 ∧ ellipse_equation x y 8 4 sorry sorry sorry = (x^2 / 8) + (y^2 / 4) = 1)

-- Theorem for the second question
def point_P := (sqrt 6, 0 : ℝ)

def inner_product (P A B : ℝ × ℝ) : ℝ :=
  ((fst A - fst P) * (fst B - fst P)) + ((snd A - snd P) * (snd B - snd P))

def line_through_fixed_point (P A B : ℝ × ℝ) : Prop :=
  inner_product P A B = -2 →
  ∃ (fixed_point : ℝ × ℝ), 
    ∀ (A B : ℝ × ℝ), line_through fixed_point (fst A, snd A) ∧ line_through fixed_point (fst B, snd B)

def ellipse_problem_2 : Prop :=
  ∀ (A B : ℝ × ℝ), 
    (∃ k m : ℝ, snd A = k * fst A + m ∧ snd B = k * fst B + m) ∨ 
    (fst A = sqrt 6 / 3 ∧ snd A = sqrt 6 / 3 ∧ fst B = sqrt 6 / 3 ∧ snd B = - sqrt 6 / 3) →
  line_through_fixed_point point_P A B 

-- Main theorems
theorem main_theorem_1 : ellipse_problem_1 := sorry
theorem main_theorem_2 : ellipse_problem_2 := sorry

end main_theorem_1_main_theorem_2_l367_367797


namespace avg_of_second_largest_and_second_smallest_is_eight_l367_367391

theorem avg_of_second_largest_and_second_smallest_is_eight :
  ∀ (a b c d e : ℕ), 
  a + b + c + d + e = 40 → 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  ((d + b) / 2 : ℕ) = 8 := 
by
  intro a b c d e hsum horder
  /- the proof goes here, but we use sorry to skip it -/
  sorry

end avg_of_second_largest_and_second_smallest_is_eight_l367_367391


namespace sum_adjacent_to_6_is_29_l367_367852
-- Import the Mathlib library for the necessary tools and functions

/--
  In a 3x3 table filled with numbers from 1 to 9 such that each number appears exactly once, 
  with conditions: 
    * (1, 1) contains 1, (3, 1) contains 2, (1, 3) contains 3, (3, 3) contains 4,
    * The sum of the numbers in the cells adjacent to the cell containing 5 is 9,
  Prove that the sum of the numbers in the cells adjacent to the cell containing 6 is 29.
-/
theorem sum_adjacent_to_6_is_29 
  (table : Fin 3 → Fin 3 → Fin 9)
  (H_uniqueness : ∀ i j k l, (table i j = table k l) → (i = k ∧ j = l))
  (H_valid_entries : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)
  (H_initial_positions : table 0 0 = 1 ∧ table 2 0 = 2 ∧ table 0 2 = 3 ∧ table 2 2 = 4)
  (H_sum_adj_to_5 : ∃ (i j : Fin 3), table i j = 5 ∧ 
                      ((i > 0 ∧ table (i-1) j +
                       (i < 2 ∧ table (i+1) j) +
                       (j > 0 ∧ table i (j-1)) +
                       (j < 2 ∧ table i (j+1))) = 9)) :
  ∃ i j, table i j = 6 ∧
  (i > 0 ∧ table (i-1) j +
   (i < 2 ∧ table (i+1) j) +
   (j > 0 ∧ table i (j-1)) +
   (j < 2 ∧ table i (j+1))) = 29 := sorry

end sum_adjacent_to_6_is_29_l367_367852


namespace isosceles_triangle_BMC_l367_367537

variables (A B C D O M : Type)
variables [trapezoid ABCD] (circ1 : circle (triangle A B O)) (circ2 : circle (triangle C O D))
variables (H1 : ¬parallel A D C B) -- Not parallel to interpret that ABCD is trapezium
variables (H2 : intersection AC BD = O)
variables (H3 : points_on_circle circ1 A O B)
variables (H4 : points_on_circle circ2 C O D)
variables (H5 : intersects_circle_point circ1 circ2 M)
variables (H6 : M ∈ line A D)

theorem isosceles_triangle_BMC : is_isosceles_triangle B M C :=
sorry

end isosceles_triangle_BMC_l367_367537


namespace perpendicular_parallel_condition_l367_367827

variables {l m : Type*} [line l] [line m] [different_lines l m]
variable (α : Type*) [plane α]
variables [perpendicular_to_plane m α]

theorem perpendicular_parallel_condition (h1 : l ⊥ m) : 
  (l ∥ α) → (l ⊥ m) ∧ ¬ (l ∥ α → l ⊥ m) :=
begin
  sorry
end

end perpendicular_parallel_condition_l367_367827


namespace cleo_marbles_after_four_days_l367_367282

theorem cleo_marbles_after_four_days :
  ∃ (initial_marbles : ℕ) (taken_day2 fraction_day2 : ℚ) (taken_day3 fraction_day3 : ℚ) (portion_cleo portion_estela : ℚ),
  initial_marbles = 240 ∧ 
  fraction_day2 = 2/3 ∧ fraction_day3 = 3/5 ∧ portion_cleo = 7/8 ∧ portion_estela = 1/4 ∧ 
  let day2_taken := (fraction_day2 * initial_marbles).toNat in
  let remaining_after_day2 := initial_marbles - day2_taken in
  let share := (day2_taken / 3).toNat in 
  let remaining_marbles_after_day2 := remaining_after_day2 in 
  let day3_taken := (fraction_day3 * remaining_marbles_after_day2).toNat in
  let cleo_day3_share := (day3_taken / 2).toNat in
  let cleo_day4_share := (portion_cleo * cleo_day3_share).toNat in
  let estela_share := (portion_estela * cleo_day4_share).toNat in
  ∃ (cleo_final : ℕ), cleo_final = cleo_day4_share - estela_share ∧ cleo_final = 16 := 
by
  sorry

end cleo_marbles_after_four_days_l367_367282


namespace complex_pow_six_eq_eight_i_l367_367288

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end complex_pow_six_eq_eight_i_l367_367288


namespace x_n_squared_leq_2007_l367_367199

def recurrence (x y : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ y 0 = 2007 ∧
  ∀ n, x (n + 1) = x n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (y n + y (n + 1)) ∧
       y (n + 1) = y n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (x n + x (n + 1))

theorem x_n_squared_leq_2007 (x y : ℕ → ℝ) (h : recurrence x y) : ∀ n, x n ^ 2 ≤ 2007 :=
by sorry

end x_n_squared_leq_2007_l367_367199


namespace complex_conjugate_solution_l367_367412

theorem complex_conjugate_solution (z : ℂ) (h : (1 - 2 * complex.I) * z = 3 + complex.I) :
  complex.conj z = 1 - (7 / 5) * complex.I := 
sorry

end complex_conjugate_solution_l367_367412


namespace num_solutions_greater_l367_367515

def m (a b : ℤ) : ℕ := sorry -- Definition needs to ensure values of a and b conform to given conditions

theorem num_solutions_greater :
  (∑ a in (finset.Icc (-5000) 5000), ∑ b in (finset.Icc (-(5 * 10^9)) (5 * 10^9)), (m a b) * (m (-a) (-b)))
  >
  (∑ a in (finset.Icc (-5000) 5000), ∑ b in (finset.Icc (-(5 * 10^9)) (5 * 10^9)), (m a b) * (m (1 - a) (1 - b))) :=
sorry

end num_solutions_greater_l367_367515


namespace sequence_value_l367_367382

variables (a q : ℝ)
def a_n (n : ℕ) : ℝ := a * q^(n-1)
def b_n (n : ℕ) : ℝ := 1 + (finset.range n).sum (λ k, a_n a q (k + 1))
def c_n (n : ℕ) : ℝ := 2 + (finset.range n).sum (b_n a q)

theorem sequence_value (h1 : ∃ r, ∀ n, c_n a q n = r * (c_n a q (n - 1))) : a + q = 3 :=
by
  sorry

end sequence_value_l367_367382


namespace triangle_side_square_sum_ratio_four_l367_367497

theorem triangle_side_square_sum_ratio_four {a b c : ℝ} (A B C : ℝ × ℝ × ℝ)
  (midpointBC : (a, 0, 0) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2, (B.3 + C.3) / 2))
  (midpointCA : (0, b, 0) = ((C.1 + A.1) / 2, (C.2 + A.2) / 2, (C.3 + A.3) / 2))
  (midpointAB : (0, 0, c) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2))
  (perpendicular_BC_CA : B.1 * C.1 + B.2 * C.2 + B.3 * C.3 = 0)
  (perpendicular_CA_AB : C.1 * A.1 + C.2 * A.2 + C.3 * A.3 = 0)
  : (AB^2 + BC^2 + CA^2) / (a^2 + b^2 + c^2) = 4 := sorry

end triangle_side_square_sum_ratio_four_l367_367497


namespace mary_mother_is_jones_l367_367357

/-
Four mothers, each with a daughter, went shopping to buy rubber.
Each mother bought twice as many meters of rubber as her daughter.
Each of them bought as many meters as the cents she paid per meter.
Mrs. Jones spent 76 cents more than Mrs. White.
Nora bought 3 meters less rubber than Mrs. Brown.
Gladys bought 2 meters more rubber than Hilda, who spent 48 cents less than Mrs. Smith.
What is the name of Mary's mother?
-/

-- Definitions for the problem
variables (h g n m : ℕ) -- rubber bought by Hilda, Gladys, Nora and Mary respectively
variables (s_b j_b w_b smith_b : ℕ) -- rubber bought by Mrs. Brown, Mrs. Jones, Mrs. White, and Mrs. Smith respectively
variables (s_s j_s w_s smith_s : ℕ) -- cents spent per meter by Mrs. Brown, Mrs. Jones, Mrs. White, and Mrs. Smith respectively

-- Conditions from the problem
axiom hilda_rubber : h = 4
axiom gladys_rubber : g = 6
axiom nora_rubber : n = 9
axiom mary_rubber : m = 10

axiom doubled_rubber : s_s = 2 * h ∧ s_b = 2 * n ∧ w_b = 2 * g ∧ j_b = 2 * m
axiom hilda_cents : smith_s * h = smith_s  -- Hilda's mother is Mrs. Smith
axiom gladys_cents : w_s * g = w_s
axiom nora_cents : s_s * n = s_s
axiom mary_cents : j_s * m = j_s

axiom jones_more_cents : j_s = w_s + 76
axiom nora_less_rubber : n = s_b - 3
axiom gladys_more_rubber : g = h + 2
axiom hilda_less_cents : h = smith_b + 48 - smith_s

-- The theorem to prove
theorem mary_mother_is_jones : w_s * mary_rubber = 400 := sorry

end mary_mother_is_jones_l367_367357


namespace count_valid_mappings_l367_367109

def M : Set ℤ := {-2, 0, 1}
def N : Set ℤ := {1, 2, 3, 4, 5}

def valid_mapping (f : ℤ → ℤ) : Prop :=
  ∀ x ∈ M, (x + f x + x * f x) % 2 = 1

theorem count_valid_mappings : 
  {f : ℤ → ℤ // (∀ x ∈ M, f x ∈ N) ∧ valid_mapping f}.toFinset.card = 45 := 
sorry

end count_valid_mappings_l367_367109


namespace reduction_of_cycle_l367_367164

noncomputable def firstReductionPercentage (P : ℝ) (x : ℝ) : Prop :=
  P * (1 - (x / 100)) * 0.8 = 0.6 * P

theorem reduction_of_cycle (P x : ℝ) (hP : 0 < P) : firstReductionPercentage P x → x = 25 :=
by
  intros h
  unfold firstReductionPercentage at h
  sorry

end reduction_of_cycle_l367_367164


namespace solve_equation_l367_367977

theorem solve_equation :
  ∃ x : ℝ, (20 / (x^2 - 9) + 5 / (x - 3) = 2) ↔ (x = (5 + Real.sqrt 449) / 4 ∨ x = (5 - Real.sqrt 449) / 4) :=
begin
  sorry
end

end solve_equation_l367_367977


namespace angle_between_given_lines_l367_367171

noncomputable def slope_roots := 
  let a := 6
  let b := 1
  let c := -1
  let delta := b^2 - 4 * a * c
  ((-b + Real.sqrt delta) / (2 * a), (-b - Real.sqrt delta) / (2 * a))

def angle_between_lines (k1 k2 : ℝ) : ℝ :=
  Real.arctan (Abs (k1 + k2) / Abs (1 - k1 * k2))

theorem angle_between_given_lines :
  let k1 := slope_roots.1
  let k2 := slope_roots.2
  angle_between_lines k1 k2 = π / 4 :=
by
  sorry

end angle_between_given_lines_l367_367171


namespace candy_ticket_cost_l367_367598

theorem candy_ticket_cost
    (whack_a_mole_tickets : ℕ)
    (skee_ball_tickets : ℕ)
    (total_tickets : ℕ)
    (number_of_candies : ℕ)
    (ticket_cost_per_candy : ℕ) :
    whack_a_mole_tickets = 33 →
    skee_ball_tickets = 9 →
    total_tickets = whack_a_mole_tickets + skee_ball_tickets →
    number_of_candies = 7 →
    ticket_cost_per_candy = total_tickets / number_of_candies →
    ticket_cost_per_candy = 6 :=
by
  intros h_wh h_sb h_tot h_candies h_div
  rw [h_wh, h_sb] at h_tot
  rw add_comm at h_tot
  rw mul_comm at h_div
  sorry

end candy_ticket_cost_l367_367598


namespace triangle_ABC_right_l367_367767

-- Define the given points and the condition on the intersection with the parabola
def A : ℝ × ℝ := (1, 2)
def D : ℝ × ℝ := (5, -2)

-- Define the parabola as a set of points
def parabola : set (ℝ × ℝ) := {p | p.snd^2 = 4 * p.fst}

-- Define that B and C lie on the parabola and the line through D
def lineThroughD (B C : ℝ × ℝ) (m : ℝ) : Prop := 
  B ∈ parabola ∧ C ∈ parabola ∧ 
  (B.snd + 2 = m * (B.fst - 5)) ∧ 
  (C.snd + 2 = m * (C.fst - 5))

-- The main theorem 
theorem triangle_ABC_right (B C : ℝ × ℝ) (m : ℝ) 
  (hB : B ∈ parabola) (hC : C ∈ parabola)
  (h_line_B : B.snd + 2 = m * (B.fst - 5))
  (h_line_C : C.snd + 2 = m * (C.fst - 5)) :
  let AB := (B.fst - A.fst, B.snd - A.snd)
      AC := (C.fst - A.fst, C.snd - A.snd) in
  AB.fst * AC.fst + AB.snd * AC.snd = 0 :=
sorry

end triangle_ABC_right_l367_367767


namespace coin_problem_l367_367240

theorem coin_problem : ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5) ∧ n % 9 = 0 :=
by
  sorry

end coin_problem_l367_367240


namespace prove_cond_2_prove_cond_3_prove_cond_1_l367_367776

-- Definitions
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def seq_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := seq_sum n + a (n + 1)

def sqrt_seq (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  real.sqrt (S n)

-- Conditions
def cond_1 (a : ℕ → ℝ) : Prop := is_arithmetic a
def cond_2 (a : ℕ → ℝ) : Prop := is_arithmetic (λ n, sqrt_seq (seq_sum a) n)
def cond_3 (a : ℕ → ℝ) : Prop := a 1 * 3 = a 2

-- Proof Statements
theorem prove_cond_2 (a : ℕ → ℝ) (h1 : cond_1 a) (h3 : cond_3 a) : cond_2 a := sorry
theorem prove_cond_3 (a : ℕ → ℝ) (h1 : cond_1 a) (h2 : cond_2 a) : cond_3 a := sorry
theorem prove_cond_1 (a : ℕ → ℝ) (h2 : cond_2 a) (h3 : cond_3 a) : cond_1 a := sorry

end prove_cond_2_prove_cond_3_prove_cond_1_l367_367776


namespace prove_cond_2_prove_cond_3_prove_cond_1_l367_367775

-- Definitions
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def seq_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := seq_sum n + a (n + 1)

def sqrt_seq (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  real.sqrt (S n)

-- Conditions
def cond_1 (a : ℕ → ℝ) : Prop := is_arithmetic a
def cond_2 (a : ℕ → ℝ) : Prop := is_arithmetic (λ n, sqrt_seq (seq_sum a) n)
def cond_3 (a : ℕ → ℝ) : Prop := a 1 * 3 = a 2

-- Proof Statements
theorem prove_cond_2 (a : ℕ → ℝ) (h1 : cond_1 a) (h3 : cond_3 a) : cond_2 a := sorry
theorem prove_cond_3 (a : ℕ → ℝ) (h1 : cond_1 a) (h2 : cond_2 a) : cond_3 a := sorry
theorem prove_cond_1 (a : ℕ → ℝ) (h2 : cond_2 a) (h3 : cond_3 a) : cond_1 a := sorry

end prove_cond_2_prove_cond_3_prove_cond_1_l367_367775


namespace problem1_correct_problem2_correct_l367_367626

-- Definitions for problem 1
def A : ℝ × ℝ := (-2, 3)
def origin : ℝ × ℝ := (0, 0)

-- Definition of the slope of the line passing through origin and A
def slope := (A.2 - origin.2) / (A.1 - origin.1)

-- Given that line l is perpendicular to the line through origin and A
def line_l_is_perp : slope * slope = -1

-- Expected equation of line l based on the given problem
def line_l_eq (x y : ℝ) := 2 * x - 3 * y + 13 = 0

-- Definitions for problem 2
def A_triangle : ℝ × ℝ := (4, 0)
def B_triangle : ℝ × ℝ := (6, 7)
def C_triangle : ℝ × ℝ := (0, 3)

-- Slope of side AB
def slope_AB := (B_triangle.2 - A_triangle.2) / (B_triangle.1 - A_triangle.1)

-- The equation of the line representing the height from side AB
def height_line_eq (x y : ℝ) := 2 * x + 7 * y - 21 = 0

-- Statements to be proven
theorem problem1_correct : ∀ x y, (line_l_is_perp → line_l_eq x y) := by
  intro x y
  assume h
  sorry

theorem problem2_correct : ∀ x y, height_line_eq x y := by
  intro x y
  sorry

end problem1_correct_problem2_correct_l367_367626


namespace number_of_integer_solutions_l367_367681

theorem number_of_integer_solutions (y : ℤ) : 
  (∃ n : ℕ, n = 7) ↔ (∑ k in (finset.filter (λ y, 3 * y^2 + 17 * y + 14 ≤ 27) (finset.Icc (-6) 0)), (1 : ℕ)) = 7 :=
begin
  sorry
end

end number_of_integer_solutions_l367_367681


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367972

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367972


namespace num_machines_first_scenario_l367_367141

theorem num_machines_first_scenario (r : ℝ) (n : ℕ) :
  (∀ r, (2 : ℝ) * r * 24 = 1) →
  (∀ r, (n : ℝ) * r * 6 = 1) →
  n = 8 :=
by
  intros h1 h2
  sorry

end num_machines_first_scenario_l367_367141


namespace min_value_cos2_minus_sin2_l367_367346

theorem min_value_cos2_minus_sin2 : ∃ x : ℝ, (∀ y : ℝ, y = cos² (x/2) - sin² (x/2) implies y ≥ -1) ∧ 
  (∃ x : ℝ, y = cos² (x/2) - sin² (x/2) ∧ y = -1) :=
sorry

end min_value_cos2_minus_sin2_l367_367346


namespace sum_of_two_longest_altitudes_l367_367819

theorem sum_of_two_longest_altitudes (a b c : ℕ) (h : a^2 + b^2 = c^2) (h1: a = 7) (h2: b = 24) (h3: c = 25) : 
  (a + b = 31) :=
by {
  sorry
}

end sum_of_two_longest_altitudes_l367_367819


namespace complement_of_event_A_l367_367455

def event_A := ∃ (products : ℕ → bool), (∃ i j, i ≠ j ∧ products i = true ∧ products j = true)

def complement_event_A := ∃ (products : ℕ → bool), ¬ (∃ i j, i ≠ j ∧ products i = true ∧ products j = true)

theorem complement_of_event_A (batch_size : ℕ) (h : batch_size = 10) :
  (event_A ↔ ¬ event_A) ↔ complement_event_A := 
sorry

end complement_of_event_A_l367_367455


namespace find_pairs_l367_367600

-- Definitions for the conditions in the problem
def is_positive (x : ℝ) : Prop := x > 0

def equations (x y : ℝ) : Prop :=
  (Real.log (x^2 + y^2) / Real.log 10 = 2) ∧ 
  (Real.log x / Real.log 2 - 4 = Real.log 3 / Real.log 2 - Real.log y / Real.log 2)

-- Lean 4 Statement
theorem find_pairs (x y : ℝ) : 
  is_positive x ∧ is_positive y ∧ equations x y → (x, y) = (8, 6) ∨ (x, y) = (6, 8) :=
by
  sorry

end find_pairs_l367_367600


namespace min_value_function_l367_367204

theorem min_value_function : 
  ∃ x : ℝ, ∀ x : ℝ, y = (x^2 + 5) / (√(x^2 + 4)) → min y = 5/2 :=
sorry

end min_value_function_l367_367204


namespace total_yards_run_l367_367931

theorem total_yards_run (Malik_yards_per_game : ℕ) (Josiah_yards_per_game : ℕ) (Darnell_yards_per_game : ℕ) (games : ℕ) 
  (hM : Malik_yards_per_game = 18) (hJ : Josiah_yards_per_game = 22) (hD : Darnell_yards_per_game = 11) (hG : games = 4) : 
  Malik_yards_per_game * games + Josiah_yards_per_game * games + Darnell_yards_per_game * games = 204 := by
  sorry

end total_yards_run_l367_367931


namespace max_enclosed_fences_l367_367470

theorem max_enclosed_fences (n : ℕ) (h : n = 100) :
  ∃ f : ℕ, f = 199 ∧ 
    (∀ s t : set ℕ, s ⊆ finset.range n ∧ t ⊆ finset.range n ∧ s.nonempty ∧ t.nonempty → 
      s ≠ t → enclosure s ∧ enclosure t) :=
sorry

end max_enclosed_fences_l367_367470


namespace minimize_q_l367_367608

noncomputable def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

theorem minimize_q : ∃ x : ℝ, q x = 2 :=
by
  sorry

end minimize_q_l367_367608


namespace hyperbola_equation_is_l367_367924

noncomputable def equation_of_hyperbola (C : Type) : Prop :=
  ∀ (foci1 foci2 vertex : (ℝ × ℝ), equation : ℝ × ℝ → ℝ),
  foci1 = (-3, 0) →
  foci2 = (3, 0) →
  vertex = (2, 0) →
  equation = λ p, x, y. (x^2)/4 - (y^2)/5  = 1 →
  equation (vertex, foci1) = equation (foci2, vertex)

theorem hyperbola_equation_is :
  equation_of_hyperbola (λ (x y : ℝ), (x^2)/4 - (y^2)/5  = 1) := 
by
  intros foci1 foci2 vertex equation h1 h2 h3 h4
  sorry

end hyperbola_equation_is_l367_367924


namespace people_in_each_playgroup_l367_367563

theorem people_in_each_playgroup (girls boys parents playgroups : ℕ) (hg : girls = 14) (hb : boys = 11) (hp : parents = 50) (hpg : playgroups = 3) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end people_in_each_playgroup_l367_367563


namespace problem_statement_l367_367448

variables (P Q : Prop)

theorem problem_statement (h1 : ¬P) (h2 : ¬(P ∧ Q)) : ¬(P ∨ Q) :=
sorry

end problem_statement_l367_367448


namespace brian_shoes_l367_367088

theorem brian_shoes (J E B : ℕ) (h1 : J = E / 2) (h2 : E = 3 * B) (h3 : J + E + B = 121) : B = 22 :=
sorry

end brian_shoes_l367_367088


namespace general_term_sum_b_terms_lambda_range_l367_367000

noncomputable def S (n : ℕ) : ℚ := n * (n + 1) / 2

theorem general_term {n : ℕ} (hn : n ≥ 1) : a n = n := sorry

theorem sum_b_terms (n : ℕ) : T n = (n^2 + 3*n) / (2*(n+1)*(n+2)) := sorry

theorem lambda_range (λ : ℝ) (n : ℕ) (hn : n ≥ 1) : 
  (T n - λ * a n ≥ 3 * λ) ↔ λ ≤ 1 / 12 := sorry

end general_term_sum_b_terms_lambda_range_l367_367000


namespace intercept_sum_mod_7_l367_367291

theorem intercept_sum_mod_7 :
  ∃ (x_0 y_0 : ℤ), (2 * x_0 ≡ 3 * y_0 + 1 [ZMOD 7]) ∧ (0 ≤ x_0) ∧ (x_0 < 7) ∧ (0 ≤ y_0) ∧ (y_0 < 7) ∧ (x_0 + y_0 = 6) :=
by
  sorry

end intercept_sum_mod_7_l367_367291


namespace sum_adjacent_to_6_is_29_l367_367869

def in_grid (n : ℕ) (grid : ℕ → ℕ → ℕ) := ∃ i j, grid i j = n

def adjacent_sum (grid : ℕ → ℕ → ℕ) (i j : ℕ) : ℕ :=
  grid (i-1) j + grid (i+1) j + grid i (j-1) + grid i (j+1)

def grid := λ i j =>
  if i = 0 ∧ j = 0 then 1 else
  if i = 2 ∧ j = 0 then 2 else
  if i = 0 ∧ j = 2 then 3 else
  if i = 2 ∧ j = 2 then 4 else
  if i = 1 ∧ j = 1 then 6 else 0

lemma numbers_positions_adjacent_5 :
  grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧
  grid 2 2 = 4 ∧ 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else 0 in
  adjacent_sum grid 1 0 = 1 + 2 + 6 :=
by sorry

theorem sum_adjacent_to_6_is_29 : 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else
                     if i = 0 ∧ j = 1 then 7 else
                     if i = 2 ∧ j = 1 then 8 else
                     if i = 1 ∧ j = 2 then 9 else 0 in
  adjacent_sum grid 1 1 = 5 + 7 + 8 + 9 :=
by sorry

end sum_adjacent_to_6_is_29_l367_367869


namespace number_of_ways_to_place_pawns_l367_367031

theorem number_of_ways_to_place_pawns :
  let n := 5 in
  let number_of_placements := (n.factorial) in
  let number_of_permutations := (n.factorial) in
  number_of_placements * number_of_permutations = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l367_367031


namespace minimal_norm_l367_367486

open Real EuclideanSpace

variable (v : ℝ^2)

theorem minimal_norm (h : ‖v + ![4, -2]‖ = 10) : ‖v‖ = 10 - 2 * Real.sqrt 5 :=
sorry

end minimal_norm_l367_367486


namespace complement_union_example_l367_367926

open Set

variable (I : Set ℕ) (A : Set ℕ) (B : Set ℕ)

noncomputable def complement (U : Set ℕ) (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem complement_union_example
    (hI : I = {0, 1, 2, 3, 4})
    (hA : A = {0, 1, 2, 3})
    (hB : B = {2, 3, 4}) :
    (complement I A) ∪ (complement I B) = {0, 1, 4} := by
  sorry

end complement_union_example_l367_367926


namespace tan_alpha_value_expression_value_l367_367001

-- Part (1)
theorem tan_alpha_value (α : ℝ) (P : ℝ × ℝ) (hP1 : P = (-8, -6)) (hP2 : sin α = -3 / 5) : 
  tan α = 3 / 4 :=
sorry

-- Part (2)
theorem expression_value (α : ℝ) (h_tan : tan α = 3 / 4) : 
  (2 * cos (3 * π / 2 + α) + cos (-α)) / (sin (5 * π / 2 - α) - cos (π + α)) = 5 / 4 :=
sorry

end tan_alpha_value_expression_value_l367_367001


namespace binary_to_octal_equiv_l367_367299

theorem binary_to_octal_equiv :
  (binary_to_decimal(11011) = 27) →
  (decimal_to_octal(27) = 33) :=
by
  -- Proof will go here
  sorry

end binary_to_octal_equiv_l367_367299


namespace tom_found_total_money_in_dollars_l367_367186

def quarters := 25 * 0.25
def dimes := 15 * 0.10
def nickels := 12 * 0.05
def half_dollars := 7 * 0.50
def dollar_coins := 3 * 1.00
def pennies := 375 * 0.01
def total_money := quarters + dimes + nickels + half_dollars + dollar_coins + pennies

theorem tom_found_total_money_in_dollars : total_money = 18.60 := by
  unfold quarters dimes nickels half_dollars dollar_coins pennies total_money
  sorry

end tom_found_total_money_in_dollars_l367_367186


namespace count_valid_subsets_l367_367015

theorem count_valid_subsets :
  let S := {90, 94, 102, 135, 165, 174} in
  (finset.univ.filter (λ (T : finset ℕ), T.card = 3 ∧ T.sum % 5 = 0)).card = 2 :=
sorry

end count_valid_subsets_l367_367015


namespace infinite_product_value_l367_367687

theorem infinite_product_value :
  (∏ n in (range (∞)).filter (λ n, ∃ k, 2^k = n + 1), (3 : ℝ)^(1/2^(n + 1))) = (3 : ℝ) :=
by
  sorry

end infinite_product_value_l367_367687


namespace evaluate_expression_l367_367704

-- Define the expressions according to the given conditions
def expr1 : ℝ := Real.sqrt 5 * 5^(1 / 2 : ℝ)
def expr2 : ℝ := 20 / 4 * 3
def expr3 : ℝ := 9^(3 / 2 : ℝ)

-- The final expression we want to evaluate
def total_expr : ℝ := expr1 + expr2 - expr3

-- The theorem stating that the evaluated expression equals -7
theorem evaluate_expression : total_expr = -7 := by
  sorry

end evaluate_expression_l367_367704


namespace derivative_of_f_l367_367338

noncomputable def f (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_f :
  (deriv f) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end derivative_of_f_l367_367338


namespace repeating_decimal_to_fraction_l367_367318

noncomputable def x : ℚ := 3 + 56 / 99

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 3 + 56 / 99) : x = 353 / 99 := 
by 
  rw h
  exact (3 + 56 / 99 : ℚ)
  sorry

end repeating_decimal_to_fraction_l367_367318


namespace unspent_portion_l367_367519

theorem unspent_portion (G X : ℝ) (hG : 0 < G) (hX : 0 ≤ X) (hX_le_one : X ≤ 1) : 
  let platinum_limit := 2 * G
      gold_balance := X * G
      platinum_balance := (2 * G) / 7
      new_platinum_balance := platinum_balance + gold_balance
      unspent_balance := platinum_limit - new_platinum_balance
  in (unspent_balance / platinum_limit) = (12 - 7 * X) / 14 :=
by
  sorry

end unspent_portion_l367_367519


namespace range_of_m_l367_367799

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 2^x + 1 else 1 - Real.log (x) / Real.log 2

-- The problem is to find the range of m such that f(1 - m^2) > f(2m - 2). We assert the range of m as given in the correct answer.
theorem range_of_m : {m : ℝ | f (1 - m^2) > f (2 * m - 2)} = 
  {m : ℝ | -3 < m ∧ m < 1} ∪ {m : ℝ | m > 3 / 2} :=
sorry

end range_of_m_l367_367799


namespace cubic_vs_square_ratio_l367_367791

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l367_367791


namespace problem1_l367_367624

theorem problem1 (h1 : Real.tan (Real.pi / 3) = Real.sqrt 3)
  (h2 : (-2 : ℝ)⁻¹ = -0.5)
  (h3 : Real.sqrt (3 / 4) = Real.sqrt 3 / 2)
  (h4 : Real.cbrt 8 = 2)
  (h5 : abs (-0.5 * Real.sqrt 12) = Real.sqrt 3) :
  Real.tan (Real.pi / 3) * (-2 : ℝ)⁻¹ - (Real.sqrt (3 / 4) - Real.cbrt 8) + abs (-0.5 * Real.sqrt 12) = 2 :=
sorry

end problem1_l367_367624


namespace cubic_vs_square_ratio_l367_367790

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l367_367790


namespace one_minus_repeating_three_l367_367329

theorem one_minus_repeating_three : 1 - (0.\overline{3}) = 2 / 3 :=
by
  sorry

end one_minus_repeating_three_l367_367329


namespace average_page_count_correct_l367_367881

def classA_students := 15
def classA_details := [(6, 3), (4, 4), (5, 2)]

def classB_students := 20
def classB_details := [(8, 5), (6, 2), (6, 6)]

def classC_students := 25
def classC_details := [(10, 3), (5, 5), (7, 4), (3, 2)]

def total_pages (class_details : List (Nat × Nat)) := 
  class_details.foldl (λ acc x => acc + x.1 * x.2) 0

def average_page_count := 
  (total_pages classA_details + total_pages classB_details + total_pages classC_details) 
  / (classA_students + classB_students + classC_students) 

theorem average_page_count_correct :
  average_page_count ≈ 3.683 := 
by 
  sorry

end average_page_count_correct_l367_367881


namespace mul_table_mod_7_mul_table_mod_10_mul_table_mod_9_l367_367677

-- Multiplication table for multiplying by 0,1,2,3,4,5,6 in arithmetic modulo 7
theorem mul_table_mod_7 :
  (∀ n, 0 * n % 7 = 0) ∧
  (∀ n, 1 * n % 7 = n % 7) ∧
  (∀ n, 2 * n % 7 = [0, 2, 4, 6, 1, 3, 5][n % 7]) ∧
  (∀ n, 3 * n % 7 = [0, 3, 6, 2, 5, 1, 4][n % 7]) ∧
  (∀ n, 4 * n % 7 = [0, 4, 1, 5, 2, 6, 3][n % 7]) ∧
  (∀ n, 5 * n % 7 = [0, 5, 3, 1, 6, 4, 2][n % 7]) ∧
  (∀ n, 6 * n % 7 = [0, 6, 5, 4, 3, 2, 1][n % 7]) :=
by sorry

-- Multiplication table for multiplying by 2 and 5 in arithmetic modulo 10
theorem mul_table_mod_10 :
  (∀ n, 2 * n % 10 = [0, 2, 4, 6, 8, 0, 2, 4, 6, 8][n % 10]) ∧
  (∀ n, 5 * n % 10 = [0, 5, 0, 5, 0, 5, 0, 5, 0, 5][n % 10]) :=
by sorry

-- Multiplication table for multiplying by 3 in arithmetic modulo 9
theorem mul_table_mod_9 :
  (∀ n, 3 * n % 9 = [0, 3, 6, 0, 3, 6, 0, 3, 6][n % 9]) :=
by sorry

end mul_table_mod_7_mul_table_mod_10_mul_table_mod_9_l367_367677


namespace find_CD_squared_l367_367190

noncomputable def first_circle (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 25
noncomputable def second_circle (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem find_CD_squared : ∃ C D : ℝ × ℝ, 
  (first_circle C.1 C.2 ∧ second_circle C.1 C.2) ∧ 
  (first_circle D.1 D.2 ∧ second_circle D.1 D.2) ∧ 
  (C ≠ D) ∧ 
  ((D.1 - C.1)^2 + (D.2 - C.2)^2 = 50) :=
by
  sorry

end find_CD_squared_l367_367190


namespace sum_of_digits_math_problem_l367_367074

-- We define types for each distinct character.
variables (学 迎 春 杯 加油 油 吧 : ℕ)

-- Constraints from the problem.
def valid_digits : Prop :=
  ∀ x, x ∈ {学, 迎, 春, 杯, 加油, 油, 吧} → x < 10 ∧ x ≠ 7

def unique_digits : Prop :=
  ∀ x ∈ {学, 迎, 春, 杯, 加油, 油, 吧}, ∀ y ∈ {学, 迎, 春, 杯, 加油, 油, 吧}, x ≠ y → x ≠ y

-- Specific constraints.
def specific_constraints : Prop :=
  迎 ≠ 1 ∧ 春 ≠ 1 ∧ 杯 ≠ 1

-- The equation given in the problem.
def given_equation : Prop :=
  学 * 7 * 迎 * 春 * 杯 = 加油 * 10 + 油 * 10 + 吧

-- The sum of digits for 迎, 春, and 杯.
theorem sum_of_digits : Prop :=
  valid_digits ∧ unique_digits ∧ specific_constraints ∧ given_equation →
  迎 + 春 + 杯 = 18

-- We state the theorem we aim to prove.
theorem math_problem :
  sum_of_digits :=
sorry

end sum_of_digits_math_problem_l367_367074


namespace grocer_profit_l367_367249

theorem grocer_profit :
  let cost_per_pound := (0.50 / 3 : ℝ)
  let total_cost := 96 * cost_per_pound
  let selling_per_pound := (1.00 / 4 : ℝ)
  let total_selling := 96 * selling_per_pound
  let profit := total_selling - total_cost
  profit = 8.00 :=
by
  let cost_per_pound := (0.50 / 3 : ℝ)
  let total_cost := 96 * cost_per_pound
  let selling_per_pound := (1.00 / 4 : ℝ)
  let total_selling := 96 * selling_per_pound
  let profit := total_selling - total_cost
  have h1 : total_cost = 16.00 := sorry
  have h2 : total_selling = 24.00 := sorry
  have h3 : profit = 24.00 - 16.00 := sorry
  calc
    profit = 24.00 - 16.00 : h3
          ... = 8.00       : by linarith

end grocer_profit_l367_367249


namespace find_N_l367_367559

theorem find_N (a b c : ℤ) (N : ℤ)
  (h1 : a + b + c = 105)
  (h2 : a - 5 = N)
  (h3 : b + 10 = N)
  (h4 : 5 * c = N) : 
  N = 50 :=
by
  sorry

end find_N_l367_367559


namespace evaluate_tan_fraction_l367_367315

theorem evaluate_tan_fraction:
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_tan_fraction_l367_367315


namespace AGF_angle_40_l367_367895

-- Definitions of the problem
variables (A B C F G : Type) [t : Triangle (Parts := {Angle, Side})]
variables (angleACB : Angle) (angleCBA : Angle)
variables (is_parallel : IsParallel A F B C)
variables (is_midpoint : IsMidpoint G A C)

-- Conditions
def angle_ACB_60 : angleACB.measure = 60 := sorry
def angle_CBA_80 : angleCBA.measure = 80 := sorry
def AF_parallel_BC : IsParallel A F B C := sorry
def G_midpoint_AC : IsMidpoint G A C := sorry

-- Prove angle AGF = 40
theorem AGF_angle_40 : angle (A G F) = 40 :=
by
  apply angle_ACB_60
  apply angle_CBA_80
  apply AF_parallel_BC
  apply G_midpoint_AC
  sorry

end AGF_angle_40_l367_367895


namespace circle_area_from_diameter_l367_367945

def P : ℝ × ℝ := (1, 3)
def Q : ℝ × ℝ := (5, 8)

theorem circle_area_from_diameter :
  let d := Real.sqrt ((Q.fst - P.fst)^2 + (Q.snd - P.snd)^2),
      r := d / 2,
      A := Real.pi * r^2 in
  A = 41 * Real.pi / 4 :=
by
  sorry

end circle_area_from_diameter_l367_367945


namespace all_zero_points_l367_367380

theorem all_zero_points (points : Finset (ℝ × ℝ))
  (H1 : ¬ ∃ l : ℝ → ℝ, ∀ p ∈ points, p.2 = l p.1)
  (H2 : ∀ (l : ℝ → ℝ), ∀ p ∈ points, Finset.sum (points.filter (λ q, q.2 = l q.1)) (λ q, f q) = 0)
  (f : ℝ × ℝ → Int) 
  : ∀ p ∈ points, f p = 0 :=
begin
  sorry
end

end all_zero_points_l367_367380


namespace find_solutions_l367_367331

def solution_set : set (ℕ × ℕ × ℕ × ℕ) := 
  { ⟨1, 0, 0, 0⟩, ⟨3, 0, 0, 1⟩, ⟨1, 1, 1, 0⟩, ⟨2, 2, 1, 1⟩ }

theorem find_solutions :
  { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | 2^x * 3^y - 5^z * 7^w = 1 } = solution_set :=
sorry

end find_solutions_l367_367331


namespace inequality_proof_l367_367781

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l367_367781


namespace sum_abs_le_9_over_4_l367_367751

open Complex

theorem sum_abs_le_9_over_4 {n : ℕ} (a : Fin n → ℂ)
    (h : ∀ I : Finset (Fin n), I.Nonempty → abs (∏ j in I, (1 + a j) - 1) ≤ 1 / 2) :
    ∑ i, abs (a i) ≤ 9 / 4 :=
sorry

end sum_abs_le_9_over_4_l367_367751


namespace max_min_difference_eq_20_l367_367425

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_min_difference_eq_20 (a : ℝ) :
  let M := max ((f 0 a)) (max (f 1 a) (f 3 a))
  let N := min ((f 0 a)) (min (f 1 a) (f 3 a))
  in M - N = 20 := by
  sorry

end max_min_difference_eq_20_l367_367425


namespace ellipse_and_triangle_area_l367_367914

theorem ellipse_and_triangle_area :
  (∃ (a b : ℝ), (a > b) ∧ (b > 0) ∧ (∃ (c : ℝ), c = a / 2 ∧ 
  (a - c = 1) ∧ (b^2 = a^2 - c^2) ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 → min (dist (x, y) (a, 0)) = 1))) → 

  (∀ (P : ℝ × ℝ), P = (0, 2) ∧ (θ := 60) ∧ 
  (∀ A B : ℝ × ℝ, line_through_point_and_angle (0, 2) 60 ∩ ellipse (x^2 / 4 + y^2 / 3 = 1) = {A, B} →
  (area_of_triangle (0, 0) A B = 2 * sqrt 177 / 15))) :=
sorry

end ellipse_and_triangle_area_l367_367914


namespace isosceles_triangle_angle_difference_constant_l367_367389

theorem isosceles_triangle_angle_difference_constant
  (A B C M X T : Type*)
  [IsoscelesTriangle A B C]
  (hBC_midpoint : M = midpoint B C)
  (hX_on_arc : X ∈ arc (circle_circum ABM) MA)
  (hT_in_domain : T ∈ angle_domain B M A)
  (hTMX_right : ∠T M X = 90°)
  (hTX_equal_BX : T X = B X) :
  ∀ X, ∠M T B - ∠C T M = ∠M A B :=
sorry

end isosceles_triangle_angle_difference_constant_l367_367389


namespace extreme_points_of_f_range_of_a_for_f_le_g_l367_367493

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (1 / 2) * x^2 + a * x

noncomputable def g (x : ℝ) : ℝ :=
  Real.exp x + (3 / 2) * x^2

theorem extreme_points_of_f (a : ℝ) :
  (∃ (x1 x2 : ℝ), x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0)
    ↔ a < -2 :=
sorry

theorem range_of_a_for_f_le_g :
  (∀ x : ℝ, x > 0 → f x a ≤ g x) ↔ a ≤ Real.exp 1 + 1 :=
sorry

end extreme_points_of_f_range_of_a_for_f_le_g_l367_367493


namespace chess_competition_players_l367_367580

theorem chess_competition_players (J H : ℕ) (total_points : ℕ) (junior_points : ℕ) (high_school_points : ℕ → ℕ)
  (plays : ℕ → ℕ)
  (H_junior_points : junior_points = 8)
  (H_total_points : total_points = (J + H) * (J + H - 1) / 2)
  (H_total_points_contribution : total_points = junior_points + H * high_school_points H)
  (H_even_distribution : ∀ x : ℕ, 0 ≤ x ∧ x ≤ J → high_school_points H = x * (x - 1) / 2)
  (H_H_cases : H = 7 ∨ H = 9 ∨ H = 14) :
  H = 7 ∨ H = 14 :=
by
  have H_cases : H = 7 ∨ H = 14 :=
    by
      sorry
  exact H_cases

end chess_competition_players_l367_367580


namespace segment_lengths_l367_367874

-- Define the circle with radius 10
def radius : ℝ := 10

-- Define the perpendicular diameters AB and CD
def diameters_perpendicular (A B C D O K : ℝ) : Prop :=
  (dist O A = radius ∧ dist O B = radius) ∧
  (dist O C = radius ∧ dist O D = radius) ∧
  (dist A B = 2 * radius) ∧
  (dist C D = 2 * radius) ∧
  (∠ A O C = 90) -- Perpendicular diameters

-- Define the chord CH intersecting AB at K
def chord_intersection (C H A B K : ℝ) : Prop :=
  (dist C H = 12 ∧ is_on_line K C H ∧ is_on_line K A B)

-- The main theorem
theorem segment_lengths (A B C D O K H : ℝ) 
  (h1 : diameters_perpendicular A B C D O K) 
  (h2 : chord_intersection C H A B K) : 
  segment_length AB (2, 18) :=
sorry

end segment_lengths_l367_367874


namespace find_positive_integers_l367_367714

theorem find_positive_integers (n : ℕ) (h : 1 ≤ n ∧ n ≤ 2012) :
  ∃ (x : fin 2012 → ℕ), (∀ i j : fin 2012, i < j → x i < x j) ∧ 
                        (∑ i, (i.val + 1) / x (⟨i, sorry⟩ : fin 2012) = n) :=
sorry

end find_positive_integers_l367_367714


namespace chord_length_from_line_l367_367535

open Real

-- Define the circle equation and the center and radius
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 2)^2 = 4
def center := (0 : ℝ, -2 : ℝ)
def radius := 2

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = x

-- Define the distance from a point to a line
def distance_to_line (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

-- Define the distance from the center to the line y = x
def distance_center_to_line : ℝ :=
  distance_to_line center.fst center.snd 1 (-1) 0

-- Define the chord length formula given by the perpendicular distance from the center to the line
def chord_length (r d : ℝ) : ℝ := 2 * sqrt (r^2 - d^2)

theorem chord_length_from_line (x y : ℝ) (h : circle_eq x y) (h_line : line_eq x y) :
  chord_length radius distance_center_to_line = 2 * sqrt 2 :=
by
  -- We assume all conditions are valid
  -- The proof is replaced with sorry to establish the theorem statement correctly
  sorry

end chord_length_from_line_l367_367535


namespace perfect_squares_of_k_l367_367479

theorem perfect_squares_of_k (k : ℕ) (h : ∃ (a : ℕ), k * (k + 1) = 3 * a^2) : 
  ∃ (m n : ℕ), k = 3 * m^2 ∧ k + 1 = n^2 := 
sorry

end perfect_squares_of_k_l367_367479


namespace set_union_complement_eq_l367_367498

def P : Set ℝ := {x | x^2 - 4 * x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}
def R_complement_Q : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem set_union_complement_eq :
  P ∪ R_complement_Q = {x | x ≤ -2} ∪ {x | x ≥ 1} :=
by {
  sorry
}

end set_union_complement_eq_l367_367498


namespace problem_statement_l367_367418

variable {α : Type}

-- Define the sequences as arithmetic sequences
def is_arithmetic_sequence (seq : ℕ → α) [Add α] [One α] [HasNeg α] := 
  ∃ (a d : α), ∀ n, seq n = a + (n - 1) * d

-- Declare the sequences
variables (a_n b_n c_n : ℕ → ℝ)

-- Define the conditions
def conditions := 
  is_arithmetic_sequence a_n ∧
  is_arithmetic_sequence b_n ∧
  is_arithmetic_sequence c_n ∧
  (a_n 1 + b_n 1 + c_n 1 = 0) ∧ 
  (a_n 2 + b_n 2 + c_n 2 = 1)

-- Formalize the goal
theorem problem_statement (h : conditions) 
  : a_n 2015 + b_n 2015 + c_n 2015 = 2014 := sorry

end problem_statement_l367_367418


namespace multiples_of_eleven_ending_in_seven_l367_367013

theorem multiples_of_eleven_ending_in_seven (n : ℕ) : 
  (∀ k : ℕ, n > 0 ∧ n < 2000 ∧ (∃ m : ℕ, n = 11 * m) ∧ n % 10 = 7) → ∃ c : ℕ, c = 18 := 
by
  sorry

end multiples_of_eleven_ending_in_seven_l367_367013


namespace winning_candidate_votes_l367_367180

-- Definitions for conditions
def total_votes := 1256 + 7636 + 11632
def winning_percentage := 0.5666666666666664

-- Winning candidate's votes
def winning_votes := winning_percentage * total_votes

-- Statement to be proved
theorem winning_candidate_votes :
  winning_votes ≈ 11632 := 
sorry

end winning_candidate_votes_l367_367180


namespace evaluate_polynomial_l367_367768

variable {x y : ℚ}

theorem evaluate_polynomial (h : x - 2 * y - 3 = -5) : 2 * y - x = 2 :=
by
  sorry

end evaluate_polynomial_l367_367768


namespace trash_outside_classrooms_l367_367188

theorem trash_outside_classrooms (total_trash classroom_trash : ℕ) (h_total : total_trash = 1576) (h_classrooms : classroom_trash = 344) :
  total_trash - classroom_trash = 1232 :=
by
  rw [h_total, h_classrooms]
  exact Nat.sub_eq_of_eq_add' rfl

end trash_outside_classrooms_l367_367188


namespace rectangle_width_l367_367177

/-- Given the conditions:
    - length of a rectangle is 5.4 cm
    - area of the rectangle is 48.6 cm²
    Prove that the width of the rectangle is 9 cm.
-/
theorem rectangle_width (length width area : ℝ) 
  (h_length : length = 5.4) 
  (h_area : area = 48.6) 
  (h_area_eq : area = length * width) : 
  width = 9 := 
by
  sorry

end rectangle_width_l367_367177


namespace range_of_f_on_0_2_l367_367890

-- Definition of the new operation "⊕"
def op (a b : ℝ) : ℝ :=
  if a >= b then a else b^2

-- Definition of the function f(x)
def f (x : ℝ) : ℝ :=
  (op 1 x) * x

-- Statement of the problem
theorem range_of_f_on_0_2 : set.range (λ x, f x) = set.Icc 0 8 :=
sorry

end range_of_f_on_0_2_l367_367890


namespace dot_product_min_value_in_triangle_l367_367473

noncomputable def dot_product_min_value (a b c : ℝ) (angleA : ℝ) : ℝ :=
  b * c * Real.cos angleA

theorem dot_product_min_value_in_triangle (b c : ℝ) (hyp1 : 0 ≤ b) (hyp2 : 0 ≤ c) 
  (hyp3 : b^2 + c^2 + b * c = 16) (hyp4 : Real.cos (2 * Real.pi / 3) = -1 / 2) : 
  ∃ (p : ℝ), p = dot_product_min_value 4 b c (2 * Real.pi / 3) ∧ p = -8 / 3 :=
by
  sorry

end dot_product_min_value_in_triangle_l367_367473


namespace eval_expr_l367_367703

theorem eval_expr (a b : ℕ) (ha : a = 3) (hb : b = 4) : (a^b)^a - (b^a)^b = -16245775 := by
  sorry

end eval_expr_l367_367703


namespace find_x_range_l367_367350

theorem find_x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) (h3 : 2 * x - 5 > 0) : x > 5 / 2 :=
by
  sorry

end find_x_range_l367_367350


namespace velocity_of_point_C_l367_367369

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l367_367369


namespace exist_coprime_integers_l367_367738

theorem exist_coprime_integers:
  ∀ (a b p : ℤ), ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exist_coprime_integers_l367_367738


namespace cosine_of_B_in_triangle_l367_367453

theorem cosine_of_B_in_triangle :
  ∀ (a b : ℝ) (A : ℝ), a = 8 → b = 4 → A = π / 3 → (cos B = sqrt 13 / 4) :=
begin
  intros a b A ha hb hA,
  sorry
end

end cosine_of_B_in_triangle_l367_367453


namespace neg_q_implies_p_neg_q_is_sufficient_not_necessary_condition_l367_367390

variable (x : ℝ)

def p := x ≤ 1
def q := 1 / x < 1

theorem neg_q_implies_p : (0 ≤ x ∧ x ≤ 1) → p :=
by sorry

theorem neg_q_is_sufficient_not_necessary_condition : (∀ x, (0 ≤ x ∧ x ≤ 1) → p) ∧ ¬(∀ x, p → (0 ≤ x ∧ x ≤ 1)) :=
by sorry

end neg_q_implies_p_neg_q_is_sufficient_not_necessary_condition_l367_367390


namespace infinite_series_sum_l367_367667

theorem infinite_series_sum :
  let x := (1 / 1000)
  let S := ∑' n : ℕ, (n + 1) * (n + 1) * x^n
  S = (1002 / 1000) / (1 - x)^3 := 
by
  have x_pos : x > 0 := by norm_num
  have x_lt_one : x < 1 := by norm_num
  sorry

end infinite_series_sum_l367_367667


namespace episodes_per_season_l367_367090

theorem episodes_per_season
  (days_to_watch : ℕ)
  (episodes_per_day : ℕ)
  (seasons : ℕ) :
  days_to_watch = 10 →
  episodes_per_day = 6 →
  seasons = 4 →
  (episodes_per_day * days_to_watch) / seasons = 15 :=
by
  intros
  sorry

end episodes_per_season_l367_367090


namespace students_like_both_l367_367056

theorem students_like_both {total students_apple_pie students_chocolate_cake students_none students_at_least_one students_both : ℕ} 
  (h_total : total = 50)
  (h_apple : students_apple_pie = 22)
  (h_chocolate : students_chocolate_cake = 20)
  (h_none : students_none = 17)
  (h_least_one : students_at_least_one = total - students_none)
  (h_union : students_at_least_one = students_apple_pie + students_chocolate_cake - students_both) :
  students_both = 9 :=
by
  sorry

end students_like_both_l367_367056


namespace repeating_decimal_to_fraction_l367_367710

theorem repeating_decimal_to_fraction (x : ℚ) (hx : x = 0.363636...) : x = 4 / 11 := by
  sorry

end repeating_decimal_to_fraction_l367_367710


namespace intersection_of_A_and_B_l367_367481

-- Define the sets A and B
def A := {x : ℝ | real.log x / real.log 2 < 2}
def B := {x : ℝ | x^2 < 9}

-- Define the expected result for the intersection of A and B
def intersection := {x : ℝ | 0 < x ∧ x < 3}

-- The theorem statement
theorem intersection_of_A_and_B : (A ∩ B) = intersection :=
by sorry

end intersection_of_A_and_B_l367_367481


namespace find_a100_l367_367386

noncomputable def sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n ≥ 1, S n = ∑ i in finset.range n.succ, a i

noncomputable def sequence_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 2, a n = 3 * (S n) ^ 2 / (3 * (S n) - 2)

theorem find_a100 (a S : ℕ → ℝ)
  (h_sum : sequence_sum S a)
  (h_cond : sequence_condition a S) :
  a 100 = -3 / 87610 :=
sorry

end find_a100_l367_367386


namespace average_reading_days_l367_367312

def emery_days : ℕ := 20
def serena_days : ℕ := 5 * emery_days
def average_days (e s : ℕ) : ℕ := (e + s) / 2

theorem average_reading_days 
  (e s : ℕ) 
  (h1 : e = emery_days)
  (h2 : s = serena_days) :
  average_days e s = 60 :=
by
  rw [h1, h2, emery_days, serena_days]
  sorry

end average_reading_days_l367_367312


namespace number_of_soldiers_l367_367646

/-- A proof problem for determining the number of soldiers given specific conditions. -/
theorem number_of_soldiers (n B F : ℕ) (h1 : B = 6 * F) (h2 : B = 7 * (B - F)) : n = 98 := by
  -- Translate the condition of the problem
  have h3 : B = 42 := by
    rw [h1]
    sorry -- Calculation to show B = 42 given h1 and h2
  have h4 : F = 7 := by
    rw [h1] at h3
    sorry -- Calculation to show F = 7
  have h5 : n = B + F := by
    sorry -- Total soldiers is B + F
  rw [h3, h4] at h5
  exact h5

end number_of_soldiers_l367_367646


namespace tan_theta_eq_l367_367484

variables (k θ : ℝ)

-- Condition: k > 0
axiom k_pos : k > 0

-- Condition: k * cos θ = 12
axiom k_cos_theta : k * Real.cos θ = 12

-- Condition: k * sin θ = 5
axiom k_sin_theta : k * Real.sin θ = 5

-- To prove: tan θ = 5 / 12
theorem tan_theta_eq : Real.tan θ = 5 / 12 := by
  sorry

end tan_theta_eq_l367_367484


namespace sum_of_two_longest_altitudes_l367_367820

theorem sum_of_two_longest_altitudes (a b c : ℕ) (h : a^2 + b^2 = c^2) (h1: a = 7) (h2: b = 24) (h3: c = 25) : 
  (a + b = 31) :=
by {
  sorry
}

end sum_of_two_longest_altitudes_l367_367820


namespace beta_sum_l367_367676

noncomputable def Q (x : ℂ) : ℂ := (∑ i in finset.range 20, x^i)^2 - x^19

theorem beta_sum : 
  ∑ (k : ℕ) in (finset.range 6), 
  (let β := (k + 1 : ℕ) / (if k < 3 then 21 else 19) in 
   β) = 122 / 399 := 
begin
  sorry
end

end beta_sum_l367_367676


namespace removed_number_is_14_l367_367990

theorem removed_number_is_14 (A B C D : ℝ) 
  (h1 : (A + B + C + D) / 4 = 20)
  (h2 : ∃ x : ℝ, ((A + B + C + D - x) / 3 = 22)) :
  ∃ x, (A + B + C + D - (A + B + C + D - 66)) = 14 :=
by
  sorry

end removed_number_is_14_l367_367990


namespace further_flight_Gaeun_l367_367506

theorem further_flight_Gaeun :
  let nana_distance_m := 1.618
  let gaeun_distance_cm := 162.3
  let conversion_factor := 100
  let nana_distance_cm := nana_distance_m * conversion_factor
  gaeun_distance_cm > nana_distance_cm := 
  sorry

end further_flight_Gaeun_l367_367506


namespace repeating_decimal_356_fraction_l367_367316

noncomputable def repeating_decimal_356 := 3.0 + 56 / 99

theorem repeating_decimal_356_fraction : repeating_decimal_356 = 353 / 99 := by
  sorry

end repeating_decimal_356_fraction_l367_367316


namespace algebra_expression_value_l367_367439

theorem algebra_expression_value (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 11) : a^2 - a * b + b^2 = 67 :=
by
  sorry

end algebra_expression_value_l367_367439


namespace find_k_l367_367002

theorem find_k (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 12 = 0 → ∃ y : ℝ, y = x + 3 ∧ y^2 - k * y + 12 = 0) →
  k = 3 :=
sorry

end find_k_l367_367002


namespace ratio_of_areas_l367_367578

theorem ratio_of_areas (C1 C2 : ℝ) (h : (60 / 360) * C1 = (30 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l367_367578


namespace percentage_of_y_l367_367595

theorem percentage_of_y (y : ℝ) : (0.3 * 0.6 * y = 0.18 * y) :=
by {
  sorry
}

end percentage_of_y_l367_367595


namespace factorize_expression_l367_367320

-- The primary goal is to prove that -2xy^2 + 4xy - 2x = -2x(y - 1)^2
theorem factorize_expression (x y : ℝ) : 
  -2 * x * y^2 + 4 * x * y - 2 * x = -2 * x * (y - 1)^2 := 
by 
  sorry

end factorize_expression_l367_367320


namespace problem1_problem2_problem3_problem4_problem5_problem6_l367_367620

-- Proof Problem 1
theorem problem1 (AB A C D : ℝ) (h1 : AC = 1 / 7 * AB) (D_in_semicircle_on_CB : true) (D_perpendicular_to_A_on_AB : true):
  AD = Real.sqrt (7) := sorry

-- Proof Problem 2
theorem problem2 (f : ℝ → ℝ) (x: ℝ) (hx : f(x / (2 * x - 1)) + (x / (2 * x - 1)) * f(x) = 2) :
  f(x) = (4 * x - 2) / (x - 1) := sorry

-- Proof Problem 3
theorem problem3 (x y z A B C: ℤ) (h1: 2 * x - 2 * y + 3 * z - 2 = 1)
  (h2: 4 * z - x - 3 * y = 1)
  (h3: 5 * y - x - 7 * z + 7 = 1) (A_integers: true) (B_integers: true) (C_integers: true)
  (sum_ABC: A + B + C = 3) :
  (x = 1) ∧ (y = 2) ∧ (z = 2) := sorry

-- Proof Problem 4
theorem problem4 (x: ℝ) :
  ( (x < -1) ∨ (0 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x < 3) → 
    ((x - 1)^2 - 1) / ((x - 1) * (x + 2) * (x - 3)) ≤ 0 ) := sorry

-- Proof Problem 5
theorem problem5 (y: ℝ) (hy : 0 ≤ y ∧ y ≤ 1) :
  ∀ y, ((1 - y) * y^3).max = 27 / 256 := sorry

-- Proof Problem 6
theorem problem6 (r : ℝ) (h : r = 1) :
  r = 1 := sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l367_367620


namespace four_cells_same_color_rectangle_l367_367985

theorem four_cells_same_color_rectangle (color : Fin 3 → Fin 7 → Bool) :
  ∃ (r₁ r₂ r₃ r₄ : Fin 3) (c₁ c₂ c₃ c₄ : Fin 7), 
    r₁ ≠ r₂ ∧ r₃ ≠ r₄ ∧ c₁ ≠ c₂ ∧ c₃ ≠ c₄ ∧ 
    r₁ = r₃ ∧ r₂ = r₄ ∧ c₁ = c₃ ∧ c₂ = c₄ ∧
    color r₁ c₁ = color r₁ c₂ ∧ color r₂ c₁ = color r₂ c₂ := sorry

end four_cells_same_color_rectangle_l367_367985


namespace salary_increase_l367_367169

theorem salary_increase (S0 S3 : ℕ) (r : ℕ) : 
  S0 = 3000 ∧ S3 = 8232 ∧ (S0 * (1 + r / 100)^3 = S3) → r = 40 :=
by
  sorry

end salary_increase_l367_367169


namespace pairs_difference_divisible_by_7_l367_367360

open Set

theorem pairs_difference_divisible_by_7 :
  { (a, b) ∈ (Finset.product (Finset.range 16) (Finset.range 16)) |
    (|a - b| % 7 = 0) } =
  {(8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (15, 1)} :=
by {
  sorry
}

end pairs_difference_divisible_by_7_l367_367360


namespace possible_values_of_f_2023_l367_367104

noncomputable def f : ℝ → ℝ := sorry

theorem possible_values_of_f_2023 :
  (∀ x : ℝ, x ≠ 1 → f(x - f(x)) + f(x) = (x^2 - x + 1) / (x - 1)) →
  (∀ y : ℝ, y ≠ 0 ∧ y ≠ (2023 + 1 / 2022) → f 2023 = y) :=
by
  intros hcond hy
  sorry

end possible_values_of_f_2023_l367_367104


namespace trajectory_of_point_l367_367384

theorem trajectory_of_point (x y : ℝ) (h : real.sqrt ((x + 5)^2 + y^2) - real.sqrt ((x - 5)^2 + y^2) = 8) :
  (x^2 / 16) - (y^2 / 9) = 1 ∧ x > 0 := 
sorry

end trajectory_of_point_l367_367384


namespace max_value_of_f_l367_367549

noncomputable def f := λ x : ℝ, (-2 * x^2 + x - 3) / x

theorem max_value_of_f : ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≤ 1 - 2 * Real.sqrt 6) ∧ f x = 1 - 2 * Real.sqrt 6 :=
by
  sorry

end max_value_of_f_l367_367549


namespace correct_option_B_l367_367488

variables {Line Plane : Type}
variable [HasParallelLine Line Plane] [HasPerpendicularLine Line Plane]

-- Definitions representing the conditions:
variables (a b c : Line) (α β γ : Plane)

-- The final theorem statement:
theorem correct_option_B : (a ⊥ α) → (b ⊥ β) → (a ∥ b) → (α ∥ β) :=
sorry

end correct_option_B_l367_367488


namespace speed_of_point_C_l367_367363

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l367_367363


namespace find_angle_A_and_area_of_triangle_l367_367835

open Real
open Classical

noncomputable def A_in_triangle (a b c : ℝ) (A B C : ℝ) (h0 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : Prop :=
  c = a * cos B + b * sin A

noncomputable def S_in_triangle_given_a_eq_2_and_b_eq_c : Prop :=
  let a := 2 in
  ∃ (b c : ℝ), b = c ∧ (b ^ 2 = 4 + 2 * sqrt 2) ∧ (1 / 2 * b * c * sin (π / 4) = sqrt 2 + 1)

theorem find_angle_A_and_area_of_triangle (a b c A B : ℝ) (h : a = 2) (hb_eq_c : b = c) 
  (hA : A_in_triangle a b c A B (by simp [h])): 
  A = π / 4 ∧ S_in_triangle_given_a_eq_2_and_b_eq_c :=
by sorry

end find_angle_A_and_area_of_triangle_l367_367835


namespace school_A_win_prob_expectation_X_is_13_l367_367957

-- Define the probabilities of school A winning individual events
def pA_event1 : ℝ := 0.5
def pA_event2 : ℝ := 0.4
def pA_event3 : ℝ := 0.8

-- Define the probability of school A winning the championship
def pA_win_championship : ℝ :=
  (pA_event1 * pA_event2 * pA_event3) +
  (pA_event1 * (1 - pA_event2) * pA_event3) +
  (pA_event1 * pA_event2 * (1 - pA_event3)) +
  ((1 - pA_event1) * pA_event2 * pA_event3)

-- Proof statement for the probability of school A winning the championship
theorem school_A_win_prob : pA_win_championship = 0.6 := sorry

-- Define the distribution and expectation for school B's total score
def X_prob : ℝ → ℝ
| 0  := (1 - pA_event1) * (1 - pA_event2) * (1 - pA_event3)
| 10 := pA_event1 * (1 - pA_event2) * (1 - pA_event3) +
        (1 - pA_event1) * pA_event2 * (1 - pA_event3) +
        (1 - pA_event1) * (1 - pA_event2) * pA_event3
| 20 := pA_event1 * pA_event2 * (1 - pA_event3) +
        pA_event1 * (1 - pA_event2) * pA_event3 +
        (1 - pA_event1) * pA_event2 * pA_event3
| 30 := pA_event1 * pA_event2 * pA_event3
| _  := 0

def expected_X : ℝ :=
  0 * X_prob 0 +
  10 * X_prob 10 +
  20 * X_prob 20 +
  30 * X_prob 30

-- Proof statement for the expectation of school B's total score
theorem expectation_X_is_13 : expected_X = 13 := sorry

end school_A_win_prob_expectation_X_is_13_l367_367957


namespace probability_A_wins_championship_expectation_X_is_13_l367_367968

/-
Definitions corresponding to the conditions in the problem
-/
def prob_event1_A_win : ℝ := 0.5
def prob_event2_A_win : ℝ := 0.4
def prob_event3_A_win : ℝ := 0.8

def prob_event1_B_win : ℝ := 1 - prob_event1_A_win
def prob_event2_B_win : ℝ := 1 - prob_event2_A_win
def prob_event3_B_win : ℝ := 1 - prob_event3_A_win

/-
Proof problems corresponding to the questions and correct answers
-/

theorem probability_A_wins_championship : prob_event1_A_win * prob_event2_A_win * prob_event3_A_win
    + prob_event1_A_win * prob_event2_A_win * prob_event3_B_win
    + prob_event1_A_win * prob_event2_B_win * prob_event3_A_win 
    + prob_event1_B_win * prob_event2_A_win * prob_event3_A_win = 0.6 := 
sorry

noncomputable def X_distribution_table : list (ℝ × ℝ) := 
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

noncomputable def expected_value_X : ℝ := 
  ∑ x in X_distribution_table, x.1 * x.2

theorem expectation_X_is_13 : expected_value_X = 13 := sorry

end probability_A_wins_championship_expectation_X_is_13_l367_367968


namespace inequality_proof_l367_367513

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : 0 < x) : 
  1 + x^(n+1) ≥ (2*x)^n / (1 + x)^(n-1) := 
by
  sorry

end inequality_proof_l367_367513


namespace inequality_proof_l367_367489

noncomputable def triangle_inequality (a b c : ℝ) (n : ℕ+) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem inequality_proof (a b c : ℝ) (n : ℕ+) (h_triangle : triangle_inequality a b c n) :
  let s := 0.5 * (a + b + c) in
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ (2 / 3)^(n - 2) * s^(n - 1) :=
sorry

end inequality_proof_l367_367489


namespace water_formed_l367_367334

theorem water_formed
  (m_NaOH : ℕ) (m_HCl : ℕ) 
  (reaction_eq : m_NaOH = 1 ∧ m_HCl = 1)
  (mass_H₂O : ℕ) 
  (mass_water_formed : mass_H₂O = 18) :
  mass_water_formed = 18 ∧ m_NaOH = 1 → ∃ m_H₂O : ℕ, m_H₂O = 1 :=
by
  sorry

end water_formed_l367_367334


namespace part1_part2_l367_367004
noncomputable def f (x : ℝ) : ℝ := 2 * sin x * (cos x - cos (x + Real.pi / 2)) - 1

theorem part1 : f (Real.pi / 6) = (Real.sqrt 3) / 2 - 1 / 2 := by
  sorry

theorem part2 (m : ℝ) (h_max : ∃ x, 0 < x ∧ x < m ∧ ∀ y, 0 < y ∧ y < m → f y < f x)
              (h_no_min : ¬ ∃ x, 0 < x ∧ x < m ∧ ∀ y, 0 < y ∧ y < m → f x < f y) : 
  (3 * Real.pi / 8) < m ∧ m ≤ (7 * Real.pi / 8) := by
  sorry

end part1_part2_l367_367004


namespace speed_of_point_C_l367_367361

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l367_367361


namespace train_crossing_time_l367_367897

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def train_speed_kmph : ℝ := 85 -- km/h
noncomputable def kmph_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def crossing_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_crossing_time :
  crossing_time train_length train_speed_mps ≈ 10.59 :=
by
  sorry

end train_crossing_time_l367_367897


namespace probability_A_wins_championship_distribution_and_expectation_B_l367_367961

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l367_367961


namespace total_yards_run_in_4_games_l367_367929

theorem total_yards_run_in_4_games (malik_ypg josiah_ypg darnell_avg : ℕ) (num_games : ℕ)
  (h1 : malik_ypg = 18) (h2 : josiah_ypg = 22) (h3 : darnell_avg = 11) (h4 : num_games = 4) :
  malik_ypg * num_games + josiah_ypg * num_games + darnell_avg * num_games = 204 := 
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_yards_run_in_4_games_l367_367929


namespace solve_absolute_value_eq_l367_367716

theorem solve_absolute_value_eq (x : ℝ) : |x - 5| = 3 * x - 2 ↔ x = 7 / 4 :=
sorry

end solve_absolute_value_eq_l367_367716


namespace one_minus_repeat_three_l367_367321

theorem one_minus_repeat_three : 1 - (0.333333..<3̅) = 2 / 3 :=
by
  -- needs proof, currently left as sorry
  sorry

end one_minus_repeat_three_l367_367321


namespace triangle_is_right_triangle_l367_367471

-- Define angles A, B, and C such that they form a triangle
variables {A B C : ℝ}
-- Define the triangle condition
axiom triangle_ABC : A + B + C = 180
-- Define the sine condition given in the problem
axiom sine_condition : sin (A + B) = sin (A - B)

theorem triangle_is_right_triangle (triangle_ABC : A + B + C = 180) (sine_condition : sin (A + B) = sin (A - B)) : 
  ( A = 90 ) ∨ ( B = 90 ) ∨ ( C = 90 ) :=
sorry

end triangle_is_right_triangle_l367_367471


namespace smallest_area_l367_367480

noncomputable def area_of_triangle (A B C : ℝ × ℝ × ℝ) : ℝ :=
1 / 2 * ‖cross_product (B.1 - A.1, B.2 - A.2, B.3 - A.3) (C.1 - A.1, C.2 - A.2, C.3 - A.3)‖

noncomputable def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem smallest_area:
  let A := (-2,0,3)
  let B := (1,3,1)
  ∃ t : ℝ, area_of_triangle A B (t,2,0) = sqrt(142.3) / 2 := 
sorry

end smallest_area_l367_367480


namespace percent_increase_in_sales_l367_367653

theorem percent_increase_in_sales :
  let new := 416
  let old := 320
  (new - old) / old * 100 = 30 := by
  sorry

end percent_increase_in_sales_l367_367653


namespace Proof_PF_squared_eq_MF_prod_NF_Proof_angles_eq_l367_367621

variables {R : Type*} [LinearOrderedField R]

noncomputable def parabola_equation (p x : R) : R := (2 * p * x)
noncomputable def Focus (p : R) : (R × R) := ⟨p / 2, 0⟩
noncomputable def distance_squared (P Q : R × R) : R := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
noncomputable def tangent_points (x0 y0 p : R) : (R × R) := sorry 

theorem Proof_PF_squared_eq_MF_prod_NF (x0 y0 p : R) (P : R × R) (M N F : R × R) :
  P = (x0, y0) ∧ parabola_equation p M.1 = M.2^2 ∧ parabola_equation p N.1 = N.2^2 ∧ P ≠ M ∧ P ≠ N ∧ F = Focus p →
  distance_squared P F = distance_squared M F * distance_squared N F :=
sorry

theorem Proof_angles_eq (x0 y0 p : R) (P : R × R) (M N F : R × R) :
  P = (x0, y0) ∧ parabola_equation p M.1 = M.2^2 ∧ parabola_equation p N.1 = N.2^2 ∧ P ≠ M ∧ P ≠ N ∧ F = Focus p →
  ∠ P M F = ∠ F P N :=
sorry

end Proof_PF_squared_eq_MF_prod_NF_Proof_angles_eq_l367_367621


namespace total_number_of_cookies_l367_367216

open Nat -- Open the natural numbers namespace to work with natural number operations

def n_bags : Nat := 7
def cookies_per_bag : Nat := 2
def total_cookies : Nat := n_bags * cookies_per_bag

theorem total_number_of_cookies : total_cookies = 14 := by
  sorry

end total_number_of_cookies_l367_367216


namespace tickets_sold_second_half_l367_367145

-- Definitions from conditions
def total_tickets := 9570
def first_half_tickets := 3867

-- Theorem to prove the number of tickets sold in the second half of the season
theorem tickets_sold_second_half : total_tickets - first_half_tickets = 5703 :=
by sorry

end tickets_sold_second_half_l367_367145


namespace cos_240_l367_367305

theorem cos_240 :
  let θ := 240
  let θ_eq : θ = 180 + 60 := rfl
  (cos (real.pi * θ / 180)) = -1/2 :=
by {
  let h1 : cos (real.pi * (180 + 60) / 180) = - cos (real.pi * 60 / 180) := sorry,
  let h2 : cos (real.pi * 60 / 180) = 1/2 := sorry,
  rw θ_eq, rw h1, rw h2, norm_num
}

end cos_240_l367_367305


namespace fill_trough_time_l367_367274

noncomputable def time_to_fill (T_old T_new T_third : ℕ) : ℝ :=
  let rate_old := (1 : ℝ) / T_old
  let rate_new := (1 : ℝ) / T_new
  let rate_third := (1 : ℝ) / T_third
  let total_rate := rate_old + rate_new + rate_third
  1 / total_rate

theorem fill_trough_time:
  time_to_fill 600 200 400 = 1200 / 11 := 
by
  sorry

end fill_trough_time_l367_367274


namespace solution_x_3_over_4_l367_367217

noncomputable def main_equation (x : ℝ) : ℝ :=
  5^(-2 * real.log (3 - 4 * x^2) / real.log 0.04) + 1.5 * (real.log (4^x) / real.log (1/8)) 

theorem solution_x_3_over_4 :
  (∃ x : ℝ, -real.sqrt 3 / 2 < x ∧ x < real.sqrt 3 / 2 ∧ main_equation x = 0) ↔
  x = 3/4 :=
sorry

end solution_x_3_over_4_l367_367217


namespace angle_AYB_eq_2_angle_ADX_l367_367496

variables 
  (A B C D X Y : Type) 
  [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ D]
  [InnerProductSpace ℝ X]
  [InnerProductSpace ℝ Y]

-- Given conditions as definitions and assumptions
def convex_quadrilateral (A B C D : Type) : Prop := 
  ∃ (AB BC CD DA : ℝ), 
    AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0

def not_parallel (A B C D : Type) : Prop := 
  ∃ (m₁ m₂ : ℝ), 
    m₁ ≠ m₂

def specific_point_properties (A D X B C : Type) (angle_ADX angle_BCX angle_DAX angle_CBX : ℝ) : Prop := 
  angle_ADX = angle_BCX ∧ angle_ADX < 90 ∧
  angle_DAX = angle_CBX ∧ angle_DAX < 90

def intersection_perpendicular_bisectors (Y A B C D : Type) : Prop := 
  ∃ (P Q : Type), 
    P ≠ Q

-- The main theorem
theorem angle_AYB_eq_2_angle_ADX 
  (ABCD_convex : convex_quadrilateral A B C D)
  (AB_not_parallel_CD : not_parallel A B C D)
  (X_property : specific_point_properties A D X B C (∠ ADX) (∠ BCX) (∠ DAX) (∠ CBX))
  (Y_intersection: intersection_perpendicular_bisectors Y A B C D) :
  ∠ AYB = 2 * ∠ ADX :=
sorry  -- proof to be filled later 

end angle_AYB_eq_2_angle_ADX_l367_367496


namespace line_x_intercept_l367_367259

theorem line_x_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (1, -2)) (h2 : (x2, y2) = (5, 6)) :
  let m := (y2 - y1) / (x2 - x1) in
  let y_intercept := y1 - m * x1 in
  let x_intercept := - y_intercept / m in
  x_intercept = 2 :=
by
  sorry

end line_x_intercept_l367_367259


namespace sum_adjacent_to_6_is_29_l367_367868

def in_grid (n : ℕ) (grid : ℕ → ℕ → ℕ) := ∃ i j, grid i j = n

def adjacent_sum (grid : ℕ → ℕ → ℕ) (i j : ℕ) : ℕ :=
  grid (i-1) j + grid (i+1) j + grid i (j-1) + grid i (j+1)

def grid := λ i j =>
  if i = 0 ∧ j = 0 then 1 else
  if i = 2 ∧ j = 0 then 2 else
  if i = 0 ∧ j = 2 then 3 else
  if i = 2 ∧ j = 2 then 4 else
  if i = 1 ∧ j = 1 then 6 else 0

lemma numbers_positions_adjacent_5 :
  grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧
  grid 2 2 = 4 ∧ 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else 0 in
  adjacent_sum grid 1 0 = 1 + 2 + 6 :=
by sorry

theorem sum_adjacent_to_6_is_29 : 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else
                     if i = 0 ∧ j = 1 then 7 else
                     if i = 2 ∧ j = 1 then 8 else
                     if i = 1 ∧ j = 2 then 9 else 0 in
  adjacent_sum grid 1 1 = 5 + 7 + 8 + 9 :=
by sorry

end sum_adjacent_to_6_is_29_l367_367868


namespace repeating_decimal_356_fraction_l367_367317

noncomputable def repeating_decimal_356 := 3.0 + 56 / 99

theorem repeating_decimal_356_fraction : repeating_decimal_356 = 353 / 99 := by
  sorry

end repeating_decimal_356_fraction_l367_367317


namespace distinct_pawns_5x5_l367_367024

theorem distinct_pawns_5x5 : 
  ∃ n : ℕ, n = 14400 ∧ 
  (∃ (get_pos : Fin 5 → Fin 5), function.bijective get_pos) :=
begin
  sorry
end

end distinct_pawns_5x5_l367_367024


namespace least_11_heavy_three_digit_number_l367_367265

def is_11_heavy (n : ℕ) : Prop :=
  n % 11 > 7

theorem least_11_heavy_three_digit_number : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ is_11_heavy n ∧ ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ is_11_heavy m → n ≤ m :=
begin
  use 107,
  split,
  { exact nat.le_refl 107, },
  split,
  { exact nat.le_of_lt (nat.lt_step (nat.lt_step (nat.lt_step (nat.lt_step (nat.lt_step (nat.lt_step (nat.lt_step (nat.lt_step (nat.lt_step nat.one_lt_base))))))))) },
  split,
  { norm_num, },
  { 
    intros m hm,
    cases hm with h1 hm,
    cases hm with h2 hm,
    cases hm with h3 hm,
    
    by_contradiction h,
    push_neg at h,

    have : ∃ x : ℕ, 100 < x ∧ x ≤ 107 ∧ is_11_heavy x,
    {
      apply (nat.bounded_steps (λ (x : ℕ), is_11_heavy x) 99 7 8 12),
      exact dec_trivial,
    },

    cases this with x h4,
    cases con ⟨h4.fst, h4.snd.fst, h4.snd.snd⟩ (λ x hx, absurd hx h),

    exact False, 
  }
end

end least_11_heavy_three_digit_number_l367_367265


namespace sequence_arith_to_sqrt_sum_arith_sum_arith_to_a2_eq_3a1_a2_eq_3a1_to_seq_arith_l367_367779

-- First combination
theorem sequence_arith_to_sqrt_sum_arith (a : ℕ → ℕ) (h1 : ∀ n, a n = a 1 + n * 2) (h2 : a 2 = 3 * a 1) : 
  ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ := sorry

-- Second combination
theorem sum_arith_to_a2_eq_3a1 (a : ℕ → ℕ) (h1 : ∀ n, a n = a 1 + n * 2) (h2 : ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ) : 
  a 2 = 3 * a 1 := sorry

-- Third combination
theorem a2_eq_3a1_to_seq_arith (a : ℕ → ℕ) (h1 : a 2 = 3 * a 1) (h2 : ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ) : 
  ∀ n, a n = a 1 + n * 2 := sorry

end sequence_arith_to_sqrt_sum_arith_sum_arith_to_a2_eq_3a1_a2_eq_3a1_to_seq_arith_l367_367779


namespace one_minus_repeating_decimal_l367_367326

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ := x

theorem one_minus_repeating_decimal:
  ∀ (x : ℚ), x = 1/3 → 1 - x = 2/3 :=
by
  sorry

end one_minus_repeating_decimal_l367_367326


namespace polynomial_is_2y2_l367_367172

variables (x y : ℝ)

theorem polynomial_is_2y2 (P : ℝ → ℝ → ℝ) (h : P x y + (x^2 - y^2) = x^2 + y^2) : 
  P x y = 2 * y^2 :=
by
  sorry

end polynomial_is_2y2_l367_367172


namespace carnival_game_probability_l367_367904

/-- 
  Jolene and Tia are playing a game at a carnival. 
  There are five red balls numbered 5, 10, 15, 20, and 25.
  There are 25 green balls numbered 1 through 25. 
  Jolene randomly chooses one red ball. The carnival worker removes the green ball with the same number as the red ball chosen.
  Tia randomly chooses one of the 24 remaining green balls.
  Jolene and Tia win if the number on the ball chosen by Tia is a multiple of 3.
  Prove that the probability that Jolene and Tia will win the game is 13/40.
--/
theorem carnival_game_probability :
  let red_balls := {5, 10, 15, 20, 25},
      green_balls := finset.range (25 + 1),
      multiples_of_three := {3, 6, 9, 12, 15, 18, 21, 24}
  in
  ∃ (prob : ℚ), prob = 13 / 40 := sorry

end carnival_game_probability_l367_367904


namespace point_C_velocity_l367_367368

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l367_367368


namespace BN_greater_CN_l367_367887

-- Define the acute triangle ABC with given properties
variables (A B C D E F P Q R N M : Type)
variable [triangle : HasTriangle A B C]

-- Conditions as per the problem statement
def acute_triangle (h : acute_triangle (A, B, C)) : Prop :=
  (has_property_perpendicular AD BC D) ∧
  (has_property_perpendicular BE AC E) ∧
  (has_property_perpendicular CF AB F) ∧
  (line_intersects EF BC P) ∧
  (parallel_line_through_point D EF intersects AC Q intersects AB R) ∧
  (N_on_segment_BC A B C N) ∧
  (angle_sum_condition NQP NRP < 180)

// The statement of the theorem to prove in Lean
theorem BN_greater_CN (h : acute_triangle) : 
  B > C := 
sorry

end BN_greater_CN_l367_367887


namespace solution_of_equation_l367_367046

theorem solution_of_equation (m : ℝ) :
  (∃ x : ℝ, x = (4 - 3 * m) / 2 ∧ x > 0) ↔ m < 4 / 3 ∧ m ≠ 2 / 3 :=
by
  sorry

end solution_of_equation_l367_367046


namespace total_cost_price_is_correct_l367_367257

noncomputable def selling_price_before_discount (sp_after_discount : ℝ) (discount_rate : ℝ) : ℝ :=
  sp_after_discount / (1 - discount_rate)

noncomputable def cost_price_from_profit (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

noncomputable def cost_price_from_loss (selling_price : ℝ) (loss_rate : ℝ) : ℝ :=
  selling_price / (1 - loss_rate)

noncomputable def total_cost_price : ℝ :=
  let CP1 := cost_price_from_profit (selling_price_before_discount 600 0.05) 0.25
  let CP2 := cost_price_from_loss 800 0.20
  let CP3 := cost_price_from_profit 1000 0.30 - 50
  CP1 + CP2 + CP3

theorem total_cost_price_is_correct : total_cost_price = 2224.49 :=
  by
  sorry

end total_cost_price_is_correct_l367_367257


namespace power_greater_than_one_million_l367_367474

theorem power_greater_than_one_million (α β γ δ : ℝ) (ε ζ η : ℕ)
  (h1 : α = 1.01) (h2 : β = 1.001) (h3 : γ = 1.000001) 
  (h4 : δ = 1000000) 
  (h_eps : ε = 99999900) (h_zet : ζ = 999999000) (h_eta : η = 999999000000) :
  α^ε > δ ∧ β^ζ > δ ∧ γ^η > δ :=
by
  sorry

end power_greater_than_one_million_l367_367474


namespace domain_of_log_function_l367_367153

def f (x : ℝ) := log (x - 1)

theorem domain_of_log_function : ∀ x : ℝ, (f x).domain = set.Ioi 1 := by
  sorry

end domain_of_log_function_l367_367153


namespace quadratic_equation_m_l367_367438

theorem quadratic_equation_m (m : ℝ) (h1 : |m| + 1 = 2) (h2 : m + 1 ≠ 0) : m = 1 :=
sorry

end quadratic_equation_m_l367_367438


namespace class_situps_update_l367_367640

noncomputable def class_stats (x_situps : ℕ → ℕ) (n : ℕ) : ℝ × ℝ :=
  let avg := (∑ i in finset.range n, x_situps i) / n
  let variance := (∑ i in finset.range n, (x_situps i - avg) ^ 2) / n
  (avg, variance)

theorem class_situps_update (x_situps : ℕ → ℕ)
  (n : ℕ)
  (x_avg : ℝ)
  (x_var : ℝ)
  (x_new : ℕ) :
  (class_stats x_situps n = (x_avg, x_var)) →
  (x_new = x_avg) →
  (class_stats (fun i => if i = n then x_new else x_situps i) (n + 1) = (x_avg, x_var * (n / (n + 1)))) :=
begin
  sorry
end

end class_situps_update_l367_367640


namespace valid_votes_b_received_l367_367224

variable (V : ℕ) (V_valid: ℕ) (A_votes B_votes : ℕ)

-- Conditions
def condition1 : V = 7720 := by sorry
def condition2 : V_valid = 0.80 * V := by sorry
def condition3 : A_votes = B_votes + 0.15 * V := by sorry
def condition4 : A_votes + B_votes = V_valid := by sorry

-- Goal
theorem valid_votes_b_received : B_votes = 2509 := by
  have h1 : V = 7720 := condition1
  have h2 : V_valid = 0.80 * V := condition2
  have h3 : A_votes = B_votes + 0.15 * V := condition3
  have h4 : A_votes + B_votes = V_valid := condition4
  sorry -- proof goes here

end valid_votes_b_received_l367_367224


namespace two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l367_367627

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ 2^n - 1) ↔ ∃ k : ℕ, n = 3 * k :=
by sorry

theorem two_pow_n_plus_one_not_div_by_seven (n : ℕ) : n > 0 → ¬(7 ∣ 2^n + 1) :=
by sorry

end two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l367_367627


namespace cubic_vs_square_ratio_l367_367792

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l367_367792


namespace proof_of_x_and_velocity_l367_367373

variables (a T L R x : ℝ)

-- Given condition
def given_eq : Prop := (a * T) / (a * T - R) = (L + x) / x

-- Target statement to prove
def target_eq_x : Prop := x = a * T * (L / R) - L
def target_velocity : Prop := a * (L / R)

-- Main theorem to prove the equivalence
theorem proof_of_x_and_velocity (a T L R : ℝ) : given_eq a T L R x → target_eq_x a T L R x ∧ target_velocity a T L R =
  sorry

end proof_of_x_and_velocity_l367_367373


namespace relatively_prime_count_200_l367_367814

theorem relatively_prime_count_200 (n : ℕ) (h : n = 200) : 
  {(k : ℕ) | k < 200 ∧ Nat.gcd k 15 = 1 ∧ Nat.gcd k 24 = 1}.card = 53 := by 
  sorry

end relatively_prime_count_200_l367_367814


namespace find_f_6_12_l367_367156

namespace ProofProblem

def f (x y : ℤ) : ℤ :=
  if x = 0 ∨ y = 0 then 0 else f (x - 1) y + f x (y - 1) + x + y

theorem find_f_6_12 :
  f 6 12 = 77500 := by
  sorry

end ProofProblem

end find_f_6_12_l367_367156


namespace intersection_of_asymptotes_l367_367726

-- Define the function
def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

-- Define the point (3, 1)
def point_of_intersection : ℝ × ℝ := (3, 1)

theorem intersection_of_asymptotes :
    f.determine_vertical_asymptote = 3 ∧ 
    (f.determine_horizontal_asymptote (3) = 1)  →
    point_of_intersection = (3, 1) :=
by
  sorry

end intersection_of_asymptotes_l367_367726


namespace blue_tissue_length_exists_l367_367178

theorem blue_tissue_length_exists (B R : ℝ) (h1 : R = B + 12) (h2 : 2 * R = 3 * B) : B = 24 := 
by
  sorry

end blue_tissue_length_exists_l367_367178


namespace boys_ratio_total_students_l367_367456

theorem boys_ratio_total_students (p : ℝ) :
  (∀ p, p = 1 / 4 * (1 - p) -> p = 1 / 5) :=
by
  assume p : ℝ
  assume h : p = 1 / 4 * (1 - p)
  sorry

end boys_ratio_total_students_l367_367456


namespace max_independent_set_l367_367744

theorem max_independent_set (V : Type)
    [Fintype V] [DecidableEq V] [Nonempty V]
    (G : SimpleGraph V)
    (hV : Fintype.card V = 30)
    (hDegree : ∀ v : V, G.degree v ≤ 5)
    (hSubset : ∀ (S : Finset V), S.card = 5 → ∃ (a b : V), a ∈ S ∧ b ∈ S ∧ ¬ G.Adj a b) :
    ∃ (S : Finset V), S.card = 6 ∧ ∀ (v w : V), v ∈ S → w ∈ S → ¬ G.Adj v w :=
sorry

end max_independent_set_l367_367744


namespace count_positive_integers_satisfying_conditions_l367_367303

theorem count_positive_integers_satisfying_conditions :
  let condition1 (n : ℕ) := (169 * n) ^ 25 > n ^ 75
  let condition2 (n : ℕ) := n ^ 75 > 3 ^ 150
  ∃ (count : ℕ), count = 3 ∧ (∀ (n : ℕ), (condition1 n) ∧ (condition2 n) → 9 < n ∧ n < 13) :=
by
  sorry

end count_positive_integers_satisfying_conditions_l367_367303


namespace decreasing_power_function_l367_367307

theorem decreasing_power_function (m : ℝ) : 
  (m^2 - m - 1 ≠ 0) ∧ ∀ x > 0, ∃ (c : ℝ), c = (m^2 - m - 1) ∧ x ^ (m^2 + 2m - 3) ∧ (m = -1) := 
by
  sorry

end decreasing_power_function_l367_367307


namespace cube_root_neg_sixty_four_l367_367992

theorem cube_root_neg_sixty_four : ∃ x : ℝ, x^3 = -64 ∧ x = -4 :=
by
  use -4
  split
  · norm_num
  · rfl

end cube_root_neg_sixty_four_l367_367992


namespace find_real_a_l367_367404

theorem find_real_a (a : ℝ) : 
  (let i := Complex.I in 
  let z := ((i ^ 2 + a * i) / (1 + i) : ℂ) in 
  Im z = z) -> a = 1 
:= 
sorry

end find_real_a_l367_367404


namespace calculator_odd_probability_l367_367634

theorem calculator_odd_probability (n : ℕ) (h : n > 0) : 
  ∃ c, tendsto (λ n, (button_presses_probability n)) at_top (nhds c) ∧ c = 1/3 := 
sorry

end calculator_odd_probability_l367_367634


namespace triangle_area_l367_367836

theorem triangle_area {a b : ℝ} (h₁ : a = 3) (h₂ : b = 4) (h₃ : Real.sin (C : ℝ) = 1/2) :
  let area := (1 / 2) * a * b * (Real.sin C) 
  area = 3 := 
by
  rw [h₁, h₂, h₃]
  simp [Real.sin, mul_assoc]
  sorry

end triangle_area_l367_367836


namespace find_trajectory_l367_367383

-- Define the equations of circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the external tangency condition to C1 and internal tangency to C2
def tangent_to_C1 (M : ℝ × ℝ) (r : ℝ) : Prop := 
  let (x, y) := M in (x + 1)^2 + y^2 = (r + 1)^2

def tangent_to_C2 (M : ℝ × ℝ) (r : ℝ) : Prop := 
  let (x, y) := M in (x - 1)^2 + y^2 = (5 - r)^2

-- Define the equation of the trajectory of the center of circle M
def ellipse_trajectory (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 8) = 1

-- The theorem stating the condition
theorem find_trajectory (M : ℝ × ℝ) (r : ℝ) :
  (tangent_to_C1 M r) ∧ (tangent_to_C2 M r) → (ellipse_trajectory M.1 M.2) :=
sorry

end find_trajectory_l367_367383


namespace max_true_statements_l367_367105

theorem max_true_statements (y : ℝ) :
  (0 < y^3 ∧ y^3 < 2 → ∀ (y : ℝ),  y^3 > 2 → False) ∧
  ((-2 < y ∧ y < 0) → ∀ (y : ℝ), (0 < y ∧ y < 2) → False) →
  ∃ (s1 s2 : Prop), 
    ((0 < y^3 ∧ y^3 < 2) = s1 ∨ (y^3 > 2) = s1 ∨ (-2 < y ∧ y < 0) = s1 ∨ (0 < y ∧ y < 2) = s1 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s1) ∧
    ((0 < y^3 ∧ y^3 < 2) = s2 ∨ (y^3 > 2) = s2 ∨ (-2 < y ∧ y < 0) = s2 ∨ (0 < y ∧ y < 2) = s2 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s2) ∧ 
    (s1 ∧ s2) → 
    ∃ m : ℕ, m = 2 := 
sorry

end max_true_statements_l367_367105


namespace front_wheel_perimeter_l367_367553

theorem front_wheel_perimeter (P_b : ℕ) (revolutions_front: ℕ) (revolutions_back: ℕ) :
  P_b = 20 → revolutions_front = 240 → revolutions_back = 360 → 
  (revolutions_front * (30 : ℕ) = revolutions_back * P_b) → 
  30 = ((360 * P_b) / 240 : ℕ) := 
by
  intros P_b_eq revolutions_front_eq revolutions_back_eq distance_eq
  rw [P_b_eq, revolutions_front_eq, revolutions_back_eq] at distance_eq
  have h: 30 = ((360 * 20) / 240 : ℕ), by 
    calc
      30 = ((360 * 20) / 240 : ℕ) : by
        norm_num
  exact h

end front_wheel_perimeter_l367_367553


namespace find_a_value_l367_367503

theorem find_a_value (U : set ℕ) (A : set ℕ) (a : ℕ) (hU : U = {1, 3, 5, 7, 9}) (hA : A = {1, abs (a - 5), 9}) (hAU : (U \ A) = {5, 7}) :
  a = 2 ∨ a = 8 := 
sorry

end find_a_value_l367_367503


namespace final_number_of_cards_l367_367358

def initial_cards : ℕ := 26
def cards_given_to_mary : ℕ := 18
def cards_found_in_box : ℕ := 40
def cards_given_to_john : ℕ := 12
def cards_purchased_at_fleamarket : ℕ := 25

theorem final_number_of_cards :
  (initial_cards - cards_given_to_mary) + (cards_found_in_box - cards_given_to_john) + cards_purchased_at_fleamarket = 61 :=
by sorry

end final_number_of_cards_l367_367358


namespace placement_of_pawns_l367_367037

-- Define the size of the chessboard and the total number of pawns
def board_size := 5
def total_pawns := 5

-- Define the problem statement
theorem placement_of_pawns : 
  (∑ (pawns : Finset (Fin total_pawns → Fin board_size)), 
    (∀ p1 p2 : Fin total_pawns, p1 ≠ p2 → pawns(p1) ≠ pawns(p2)) ∧ -- distinct positions
    (∀ i j : Fin total_pawns, i ≠ j → pawns(i) ≠ pawns(j)) ∧ -- no same row/column
    pawns.card = total_pawns) = 14400 :=
sorry

end placement_of_pawns_l367_367037


namespace reject_null_hypothesis_l367_367743

noncomputable def test_null_hypothesis 
  (n m: ℕ) (x̄ ȳ: ℝ) (D_X D_Y: ℝ) (α: ℝ): Prop :=
  let standard_error := real.sqrt ((D_X / n) + (D_Y / m)) in
  let z_obs := (x̄ - ȳ) / standard_error in
  let critical_z := 2.58 in
  abs z_obs > critical_z

theorem reject_null_hypothesis
  (n m: ℕ) (x̄ ȳ: ℝ) (D_X D_Y: ℝ) (α: ℝ)
  (h1 : n = 40)
  (h2 : m = 50)
  (h3 : x̄ = 130)
  (h4 : ȳ = 140)
  (h5 : D_X = 80)
  (h6 : D_Y = 100)
  (h7 : α = 0.01) : test_null_hypothesis n m x̄ ȳ D_X D_Y α :=
by {
  rw [h1, h2, h3, h4, h5, h6, h7],
  sorry,
}

end reject_null_hypothesis_l367_367743


namespace male_students_tree_planting_l367_367309

theorem male_students_tree_planting (average_trees : ℕ) (female_trees : ℕ) 
    (male_trees : ℕ) : 
    (average_trees = 6) →
    (female_trees = 15) → 
    (1 / male_trees + 1 / female_trees = 1 / average_trees) → 
    male_trees = 10 :=
by
  intros h_avg h_fem h_eq
  sorry

end male_students_tree_planting_l367_367309


namespace cone_lateral_surface_area_l367_367449

theorem cone_lateral_surface_area (r h : ℝ) (r_eq : r = 1) (h_eq : h = Real.sqrt 3) :
  let l := Real.sqrt (r^2 + h^2)
  let A := Real.pi * r * l
  A = 2 * Real.pi :=
by
  have l_eq : l = 2 := by 
    rw [←r_eq, ←h_eq]
    simp [Real.sqrt_eq_rpow, Real.sqrt]
  rw [←l_eq, ←r_eq]
  simp [A]
  sorry

end cone_lateral_surface_area_l367_367449


namespace Hex_game_Zermelo_Alice_has_winning_strategy_l367_367293

-- Define the game board and winning conditions
structure HexagonalBoard (n : ℕ) :=
  (cells : set (ℕ × ℕ)) -- The cells of the board

-- Define the players
inductive Player
| Alice
| Bob

-- Determine if a path exists between two sets of cells
def connected (b : HexagonalBoard n) (S T : set (ℕ × ℕ)) : Prop :=
  sorry -- Define the connection predicate

-- Winning conditions for Alice and Bob
def AliceWins (b : HexagonalBoard n) : Prop :=
  connected b -- define north and south edges as sets of cells

def BobWins (b : HexagonalBoard n) : Prop :=
  connected b -- define east and west edges as sets of cells

-- Zermelo's Theorem application
theorem Hex_game_Zermelo (n : ℕ) (b : HexagonalBoard n) : 
  (∃ strategy_For_Alice : (ℕ × ℕ) → ℕ, AliceWins b) ∨
  (∃ strategy_For_Bob : (ℕ × ℕ) → ℕ, BobWins b) :=
sorry

-- Prove that Alice has a winning strategy
theorem Alice_has_winning_strategy (n : ℕ) (b : HexagonalBoard n) : 
  ∃ strategy_For_Alice : (ℕ × ℕ) → ℕ, AliceWins b :=
sorry

end Hex_game_Zermelo_Alice_has_winning_strategy_l367_367293


namespace exists_two_numbers_satisfy_inequality_l367_367514

theorem exists_two_numbers_satisfy_inequality (x1 x2 x3 x4 : ℕ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) :
  ∃ x y ∈ {x1, x2, x3, x4}, 0 ≤ (x - y) / (1 + x + y + 2 * x * y) ∧ (x - y) / (1 + x + y + 2 * x * y) < 2 - Real.sqrt 3 :=
sorry

end exists_two_numbers_satisfy_inequality_l367_367514


namespace simplify_trig_expression_l367_367132

theorem simplify_trig_expression (x : ℝ) :
  (sin (2 * x) + 1 + sin x - cos x) / (sin (2 * x) + 1 + sin x + cos x) = tan (x / 2) :=
sorry

end simplify_trig_expression_l367_367132


namespace coeff_inequality_l367_367907

-- Definitions based on the conditions
noncomputable def P (a : (Fin n → ℝ)) (X : ℝ) : ℝ := 
  ∑ i in Finset.range n, a i * X^i

def coefficients_bound (a : Fin n → ℝ) : Prop :=
  ∀ i, 0 ≤ a i ∧ a i ≤ a 0

-- The main theorem to prove
theorem coeff_inequality {a : Fin n → ℝ} (hb : coefficients_bound a) :
  let b := λ k : Fin (2*n + 1), ∑ i in Finset.range n, a i * a (n + 1 - i)
  in 4 * b (n + 1) ≤ (P a 1)^2 := 
sorry

end coeff_inequality_l367_367907


namespace remainder_of_sum_of_cubes_l367_367913

theorem remainder_of_sum_of_cubes {b : Fin 2023 → ℕ}
  (h1 : StrictMono b)
  (h2 : (∑ i, b i) = 2023 ^ 2023)
  : (∑ i, (b i) ^ 3) % 6 = 2 := 
sorry

end remainder_of_sum_of_cubes_l367_367913


namespace circumcircle_ABC_tangent_AD_l367_367630

-- Definitions
variables (Ω₁ Ω₂ : Circle)
variables (A D E F C B : Point)
variables (t₁ t₂ : Line)
variables [tangent t₁ A Ω₁] [tangent t₁ D Ω₂]
variables [parallel t₁ t₂] [tangent t₂ Ω₁] [intersect t₂ Ω₂ = {E, F}]
variables [C ∈ Ω₂] [separates EF D C] [intersect EF CD = {B}]

-- Theorem to prove
theorem circumcircle_ABC_tangent_AD :
  tangent (circumcircle ⟨A, B, C⟩) AD :=
begin
  sorry
end

end circumcircle_ABC_tangent_AD_l367_367630


namespace product_multiple_of_4_probability_l367_367377

-- Definitions and conditions
def eight_sided_die := {1, 2, 3, 4, 5, 6, 7, 8}
def roll_die (die : Set ℕ) := {x // x ∈ die}

noncomputable def probability_product_multiple_of_4 : ℚ :=
let outcomes := (roll_die eight_sided_die) × (roll_die eight_sided_die) in
let favorable_outcomes := {p | p.1 * p.2 % 4 = 0} in
(favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

-- Proof statement
theorem product_multiple_of_4_probability : 
  probability_product_multiple_of_4 = 7 / 16 :=
sorry

end product_multiple_of_4_probability_l367_367377


namespace least_possible_value_of_b_l367_367143

noncomputable def factors (n : ℕ) : ℕ :=
  (range (n + 1)).filter (λ d, n % d = 0).length

theorem least_possible_value_of_b 
  (a b : ℕ) (pos_a : 0 < a) (pos_b : 0 < b)
  (h_a_factors : factors a = 4)
  (h_b_factors : factors b = a)
  (h_divisible : b % a = 0) :
  b = 12 :=
sorry

end least_possible_value_of_b_l367_367143


namespace necessary_but_insufficient_for_extreme_value_at_l367_367688

variable {α : Type*} [TopologicalSpace α] [NormedAddCommGroup α] [NormedSpace ℝ α]
variable {β : Type*} [NormedAddCommGroup β] [NormedSpace ℝ β]
variable {f : α → β} {x₀ : α}

theorem necessary_but_insufficient_for_extreme_value_at {f : α → ℝ} {x₀ : α}
  (hf : ContinuousAt f x₀)
  (h0 : f x₀ = 0) :
  IsExtremum f x₀ ↔ (∃ x₁, IsExtremum f x₁ ∧ f x₁ = 0) :=
sorry

end necessary_but_insufficient_for_extreme_value_at_l367_367688


namespace infinite_expressible_terms_l367_367490

theorem infinite_expressible_terms
  (a : ℕ → ℕ)
  (h1 : ∀ n, a n < a (n + 1)) :
  ∃ f : ℕ → ℕ, (∀ n, a (f n) = (f n).succ * a 1 + (f n).succ.succ * a 2) ∧
    ∀ i j, i ≠ j → f i ≠ f j :=
by
  sorry

end infinite_expressible_terms_l367_367490


namespace line_equation_slope_45_distance_2sqrt2_l367_367722

theorem line_equation_slope_45_distance_2sqrt2 :
  ∃ b : ℝ, (b = 4 ∨ b = -4) ∧ ∀ x y : ℝ, (x - y + b = 0) ∧
  (1 : ℝ) = 1 ∧
  (real.sqrt ((1:ℝ) ^ 2 + (-1:ℝ) ^ 2) = real.sqrt 2 ∧
  (abs b / real.sqrt 2 = 2 * real.sqrt 2) :=
begin
  sorry

end line_equation_slope_45_distance_2sqrt2_l367_367722


namespace equal_five_digit_number_sets_l367_367622

def five_digit_numbers_not_div_5 : ℕ :=
  9 * 10^3 * 8

def five_digit_numbers_first_two_not_5 : ℕ :=
  8 * 9 * 10^3

theorem equal_five_digit_number_sets :
  five_digit_numbers_not_div_5 = five_digit_numbers_first_two_not_5 :=
by
  repeat { sorry }

end equal_five_digit_number_sets_l367_367622


namespace compute_pow_l367_367285

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end compute_pow_l367_367285


namespace min_w_value_l367_367684

open Real

noncomputable def w (x y : ℝ) : ℝ := 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27

theorem min_w_value : ∃ x y : ℝ, w x y = 81 / 4 :=
by
  use [-3/2, 1]
  dsimp [w]
  norm_num
  done

end min_w_value_l367_367684


namespace radha_profit_percentage_l367_367127

theorem radha_profit_percentage (SP CP : ℝ) (hSP : SP = 144) (hCP : CP = 90) :
  ((SP - CP) / CP) * 100 = 60 := by
  sorry

end radha_profit_percentage_l367_367127


namespace altitude_relation_l367_367298

-- Definitions based on conditions
def O_is_intersection_of_medians (ABC : Triangle) : Prop :=
  ∃ O : Point, is_intersection_of_medians O ABC

def altitude_OP_three_times_smaller (AOB ABC : Triangle) (O : Point) (h_op h_ck : ℝ) : Prop :=
  O_is_intersection_of_medians ABC ∧ h_op = h_ck / 3

-- Theorem statement
theorem altitude_relation
  (ABC AOB : Triangle)
  (O : Point)
  (h_ck : ℝ) : ∃ h_op : ℝ, altitude_OP_three_times_smaller AOB ABC O h_op h_ck :=
sorry

end altitude_relation_l367_367298


namespace cube_fraction_inequality_l367_367787

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l367_367787


namespace product_of_repeating_nine_and_nine_l367_367349

theorem product_of_repeating_nine_and_nine : (0.9 = 1) → 0.9 * 9 = 9 := by
  intro h
  have h1 : 0.9 = (1 : ℝ) := h
  rw h1
  norm_num
  sorry

end product_of_repeating_nine_and_nine_l367_367349


namespace train_cross_time_l367_367655

-- Define the conditions
def train_speed_kmhr := 52
def train_length_meters := 130

-- Conversion factor from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℕ) : ℕ := (speed_kmhr * 1000) / 3600

-- Speed of the train in m/s
def train_speed_ms := kmhr_to_ms train_speed_kmhr

-- Calculate time to cross the pole
def time_to_cross_pole (distance_m : ℕ) (speed_ms : ℕ) : ℕ := distance_m / speed_ms

-- The theorem to prove
theorem train_cross_time : time_to_cross_pole train_length_meters train_speed_ms = 9 := by sorry

end train_cross_time_l367_367655


namespace distinct_pawns_5x5_l367_367027

theorem distinct_pawns_5x5 : 
  ∃ n : ℕ, n = 14400 ∧ 
  (∃ (get_pos : Fin 5 → Fin 5), function.bijective get_pos) :=
begin
  sorry
end

end distinct_pawns_5x5_l367_367027


namespace placement_of_pawns_l367_367034

-- Define the size of the chessboard and the total number of pawns
def board_size := 5
def total_pawns := 5

-- Define the problem statement
theorem placement_of_pawns : 
  (∑ (pawns : Finset (Fin total_pawns → Fin board_size)), 
    (∀ p1 p2 : Fin total_pawns, p1 ≠ p2 → pawns(p1) ≠ pawns(p2)) ∧ -- distinct positions
    (∀ i j : Fin total_pawns, i ≠ j → pawns(i) ≠ pawns(j)) ∧ -- no same row/column
    pawns.card = total_pawns) = 14400 :=
sorry

end placement_of_pawns_l367_367034


namespace find_slope_l367_367731

-- Define the line equation
def line (x y : ℝ) : Prop :=
  4 * x + 7 * y = 28

-- Define the slope of the line to be proved
def slope (m : ℝ) : Prop :=
  m = -4 / 7

-- State the theorem
theorem find_slope : ∃ m : ℝ, slope m ∧ (∀ x y : ℝ, line x y → y = m * x + 4) :=
by
  sorry

end find_slope_l367_367731


namespace sum_adjacent_cells_of_6_is_29_l367_367845

theorem sum_adjacent_cells_of_6_is_29 (table : Fin 3 × Fin 3 → ℕ)
  (uniq : Function.Injective table)
  (range : ∀ x, 1 ≤ table x ∧ table x ≤ 9)
  (pos_1 : table ⟨0, 0⟩ = 1)
  (pos_2 : table ⟨2, 0⟩ = 2)
  (pos_3 : table ⟨0, 2⟩ = 3)
  (pos_4 : table ⟨2, 2⟩ = 4)
  (adj_5 : (∑ i in ({⟨1, 0⟩, ⟨1, 2⟩, ⟨0, 1⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 9) :
  (∑ i in ({⟨0, 1⟩, ⟨1, 0⟩, ⟨1, 2⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 29 :=
by
  sorry

end sum_adjacent_cells_of_6_is_29_l367_367845


namespace average_is_4_l367_367437

theorem average_is_4 (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := 
by 
  sorry 

end average_is_4_l367_367437


namespace carnations_percentage_l367_367248

variables {F P P_R P_C R_F R_R R_C : ℕ}

-- Conditions
def half_of_pink_flowers_are_roses : Prop := P_R = P / 2
def two_fifths_of_red_flowers_are_roses : Prop := R_R = 2 * R_F / 5
def seventy_percent_of_roses_are_pink : Prop := P = 7 * (P_R + R_R) / 10
def thirty_percent_of_flowers_are_red : Prop := R_F = 3 * F / 10

-- Claim to prove
def percentage_of_carnations_is_53 : Prop :=
  100 * (P_C + R_C) / F = 53

theorem carnations_percentage
  (h1 : half_of_pink_flowers_are_roses)
  (h2 : two_fifths_of_red_flowers_are_roses)
  (h3 : seventy_percent_of_roses_are_pink)
  (h4 : thirty_percent_of_flowers_are_red) :
  percentage_of_carnations_is_53 :=
by
  sorry

end carnations_percentage_l367_367248


namespace slope_of_line_l367_367735

theorem slope_of_line : ∀ (x y : ℝ), (4 * x + 7 * y = 28) → ∃ m b : ℝ, (-4 / 7 = m) ∧ (y = m * x + b) :=
by
  intros x y h
  use [-4 / 7, 4]
  split
  · refl
  sorry

end slope_of_line_l367_367735


namespace right_angle_triangle_XY_length_l367_367879

noncomputable def length_XY {X Y Z : Type} [MetricSpace X] [MetricSpace Y] [MetricSpace Z] 
  (triangleXYZ : IsRightTriangle X Y Z) 
  (cosZ : real.cos Z = 3 / 5) 
  (YZ : dist Y Z = 10) 
  : real :=
sqrt (YZ^2 - (cosZ * YZ)^2)

theorem right_angle_triangle_XY_length (XYZ : Triangle)
  (angleY : XYZ.angled_y = π / 2) 
  (cosZ : ∀ Z, real.cos XYZ.angled_z = 3 / 5) 
  (YZ : ∀ Y Z, dist Y Z = 10) 
  : dist XYZ.side_x Y = 8 :=
sorry

end right_angle_triangle_XY_length_l367_367879


namespace watch_cost_price_l367_367219

theorem watch_cost_price (SP_loss SP_gain CP : ℝ) 
  (h1 : SP_loss = 0.9 * CP) 
  (h2 : SP_gain = 1.04 * CP) 
  (h3 : SP_gain - SP_loss = 196) 
  : CP = 1400 := 
sorry

end watch_cost_price_l367_367219


namespace min_positive_period_f_l367_367551

noncomputable def f : ℝ → ℝ := λ x => (Real.sin x) ^ 2 - (Real.cos x) ^ 2

theorem min_positive_period_f : IsPeriodic f π :=
sorry

end min_positive_period_f_l367_367551


namespace distance_from_point_P_l367_367183

theorem distance_from_point_P (P A B C : Point) (h : ℝ) (d : ℝ) (area_ABC area_PBC area_ABP : ℝ) :
  P ∈ triangle A B C →
  Line.parallel (Line.mk P (Line.point_parallel_to P (line_through A B))) (Line.mk A B) →
  area_ABC = area_PBC + area_ABP →
  area_PBC = 3 * area_ABP →
  h = 2 →
  d = h - (3/4 * h) →
  d = 1/2 :=
by
  sorry

end distance_from_point_P_l367_367183


namespace proportion_solution_l367_367231

theorem proportion_solution : 
  (24 * 24 = 576) → (36 : ℝ) / (3 : ℝ) = (6912 : ℝ) / (24 * 24 : ℝ):=
by
  intro h
  calc
    (36 : ℝ) / (3 : ℝ) = 12 : by sorry
    ... = (6912 : ℝ) / (576 : ℝ) : by sorry

end proportion_solution_l367_367231


namespace sequence_general_formula_sequence_sum_l367_367759

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (hn : n > 0)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, S n = ∑ i in finset.range (n + 1), a i)
  (h3 : a 1 > 1)
  (h4 : ∀ n, 6 * S n = a n ^ 2 + 3 * a n + 2) :
  a n = 3 * n - 1 :=
sorry

theorem sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) (hn : n > 0)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a n = 3 * n - 1)
  (h3 : ∀ n, b n = (a n - 1) / 2 ^ n)
  (h4 : T n = ∑ i in finset.range (n + 1), b i) :
  T n = 4 - (3 * n + 4) / 2 ^ n :=
sorry

end sequence_general_formula_sequence_sum_l367_367759


namespace player_A_wins_l367_367495

def three_color_pattern {n : ℕ} (hn4 : n > 3) (hn3 : ¬(3 ∣ n)) : ∀ x y : ℕ, x < n → y < n → char :=
λ x y hxn hyn,
  match (x % 3, y % 3) with
  | (0, 0) | (1, 1) | (2, 2) => red
  | (0, 1) | (1, 2) | (2, 0) => green
  | (0, 2) | (1, 0) | (2, 1) => blue

theorem player_A_wins (n : ℕ) (hn4: n > 3) (hn3: ¬(3 ∣ n)):
  ∃ c : char, ∃ x y : ℕ, x < n ∧ y < n ∧ ∃ t : n' < n, (∀ x' y': ℕ, x' < n ∧ y' < n ∧ (((x' < x ∨ x' > x) ∧ (y' < y ∨ y' > y)) →
 (three_color_pattern hn4 hn3 x' y') ≠ c)) :=
sorry

end player_A_wins_l367_367495


namespace pyramid_volume_l367_367459

-- Definitions of a regular quadrilateral pyramid and its properties
structure Pyramid :=
  (slant_height : ℝ)
  (sphere_ratio : ℝ)
  (base_surface_area : ℝ)
  (height : ℝ)
  (volume : ℝ)
  (is_regular : Bool)
  (is_sphere_inscribed : Bool)

-- Example pyramid satisfying given conditions
def example_pyramid : Pyramid :=
{ slant_height := a,
  sphere_ratio := 1/8,
  base_surface_area := (2 * (4 * a / 5))^2,  -- Side length is 2b where b = 4a/5
  height := 3 * a / 5,  -- using the derived relation in the solution steps
  volume := 64 * a^3 / 125,  -- the calculated volume
  is_regular := true,
  is_sphere_inscribed := true }

-- The statement to be proved in Lean 4
theorem pyramid_volume (a : ℝ) : (example_pyramid.is_regular = true) ∧ (example_pyramid.is_sphere_inscribed = true) ∧ (example_pyramid.slant_height = a) ∧ (example_pyramid.sphere_ratio = 1 / 8) →
  example_pyramid.volume = 64 * a^3 / 125 :=
by
  sorry

end pyramid_volume_l367_367459


namespace unique_minimum_of_sum_of_powers_l367_367296

theorem unique_minimum_of_sum_of_powers (m : ℕ) (hm : 1 < m) :
  ∃! x : ℝ, x = (m + 1 : ℝ) / 2 ∧ 
  ∀ y : ℝ, (∑ k in Finset.range m, (y - k)^4) ≥ (∑ k in Finset.range m, ((m + 1 : ℝ) / 2 - k)^4) :=
sorry

end unique_minimum_of_sum_of_powers_l367_367296


namespace sequence_arith_to_sqrt_sum_arith_sum_arith_to_a2_eq_3a1_a2_eq_3a1_to_seq_arith_l367_367778

-- First combination
theorem sequence_arith_to_sqrt_sum_arith (a : ℕ → ℕ) (h1 : ∀ n, a n = a 1 + n * 2) (h2 : a 2 = 3 * a 1) : 
  ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ := sorry

-- Second combination
theorem sum_arith_to_a2_eq_3a1 (a : ℕ → ℕ) (h1 : ∀ n, a n = a 1 + n * 2) (h2 : ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ) : 
  a 2 = 3 * a 1 := sorry

-- Third combination
theorem a2_eq_3a1_to_seq_arith (a : ℕ → ℕ) (h1 : a 2 = 3 * a 1) (h2 : ∀ n, ((∑ i in range n, a i) : ℝ ^ (1 / 2)) = ℝ) : 
  ∀ n, a n = a 1 + n * 2 := sorry

end sequence_arith_to_sqrt_sum_arith_sum_arith_to_a2_eq_3a1_a2_eq_3a1_to_seq_arith_l367_367778


namespace part1_part2_part3_l367_367770

-- Definitions based on the conditions in the problem
variable {n : ℕ} (a b c p : ℕ → ℕ)

-- Given conditions
def cond_A : Prop := a 2 = 4 * b 1
def cond_B : Prop := ∀ n, S n = 2 * a n - 2
def cond_C : Prop := ∀ n, (n + 1) * b (n + 1) - (n + 2) * b n = (n + 1)^2 + (n + 1)
def cond_D : Prop :=
  ∀ n, c (2 * n - 1) = - (a (2 * n - 1) * b (2 * n - 1)) / 2 ∧
       c (2 * n) = (a (2 * n) * b (2 * n)) / 4
def pn_def : ℕ → ℕ := λ n, c (2 * n - 1) + c (2 * n)

-- Defining the conditions that are used for proving the main statement
axiom basic_conditions : cond_A ∧ cond_B ∧ cond_C ∧ cond_D

-- The three parts to be proved
theorem part1 (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : ∀ n, S n = 2 * a n - 2)
  (h2 : a 2 = 4 * b 1) : a n = 2^n := sorry

theorem part2 (b : ℕ → ℕ) (n : ℕ) (h : ∀ n, (n + 1) * b (n + 1) - (n + 2) * b n = (n + 1)^2 + (n + 1)) :
  ∀ n, ∃ d : ℕ, ∀ m, b m / m + d = b (m + n) / (m + n) := sorry

theorem part3 (c : ℕ → ℕ) (p : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ n, c (2 * n - 1) = - (a (2 * n - 1) * b (2 * n - 1)) / 2)
  (h2 : ∀ n, c (2 * n) = (a (2 * n) * b (2 * n)) / 4)
  (h3 : ∀ n, T n = pn_def p n + (4 * n - 1) * 4^(n - 1)) :
  T n = 3 * 4^0 + 7 * 4^1 + 11 * 4^2 + ∀ m, (4 * (m + 1) - 1) * 4^m := sorry

end part1_part2_part3_l367_367770


namespace compute_pow_l367_367284

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end compute_pow_l367_367284


namespace speed_of_point_C_l367_367362

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l367_367362


namespace L_shaped_region_perimeter_l367_367886

-- Define the necessary structures and assumptions
structure Region :=
  (right_angles : ∀ θ, θ ∈ angles → θ = 90°)
  (tick_marks : ∀ side, side ∈ tick_sides → length side = 2)
  (area : ℝ)

-- Define the L-shaped region with the given conditions
def L_shaped_region : Region :=
  { right_angles := sorry,
    tick_marks := sorry,
    area := 104 }

-- Statement to be proven
theorem L_shaped_region_perimeter : L_shaped_region.area = 104 → 
  ∀ region : Region, region = L_shaped_region → (some_perimeter_function region) = 58.67 :=
begin
  intros h region h_eq,
  sorry
end

end L_shaped_region_perimeter_l367_367886


namespace sum_adjacent_to_six_l367_367857

theorem sum_adjacent_to_six :
  ∀ (table : fin 3 × fin 3 → ℕ),
    (∀ i j, table i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∃! i j, table i j = 1) ∧
    (∃! i j, table i j = 2) ∧
    (∃! i j, table i j = 3) ∧
    (∃! i j, table i j = 4) ∧
    (∃! i j, table i j = 5) → 
    (∀ i j, 
      table i j = 5 → 
        let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                       (if i < 2 then table (i+1, j) else 0) + 
                       (if j > 0 then table (i, j-1) else 0) + 
                       (if j < 2 then table (i, j+1) else 0)
        in adj_sum = 9) →
    (∃ i j, table i j = 6 ∧
      let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                     (if i < 2 then table (i+1, j) else 0) + 
                     (if j > 0 then table (i, j-1) else 0) + 
                     (if j < 2 then table (i, j+1) else 0) 
      in adj_sum = 29) := sorry

end sum_adjacent_to_six_l367_367857


namespace parallelogram_AHSO_l367_367266

open EuclideanGeometry

-- Let ABC be an acute-angled triangle with orthocenter H
variables {A B C H : Point}
variable {△ABC : Triangle A B C}
variable [acute_angle : AcuteTriangle △ABC]
variable [orthocenter : Orthocenter H △ABC]

-- Let AE and BF be the altitudes of △ABC
variables {E F : Point}
variable [altitude_AE : Altitude A E △ABC]
variable [altitude_BF : Altitude B F △ABC]

-- Let AE' and BF' be the reflections of AE and BF in the angle bisectors of ∠A and ∠B respectively
variables {AE' BF' : Line}
variable [reflection_AE : ReflectedInAngleBisector AE A AE']
variable [reflection_BF : ReflectedInAngleBisector BF B BF']

-- Let O be the intersection of AE' and BF'
variable {O : Point}
variable [intersection_O : Intersection AE' BF' O]

-- Let M and N be the intersections of AE and AO with the circumcircle of △ABC, respectively
variables {M N : Point}
variable [circumcircle_M : OnCircumcircle M A E △ABC]
variable [circumcircle_N : OnCircumcircle N A O △ABC]

-- Let P be the intersection of BC and HN
variables {P : Point}
variable [intersection_P : Intersection (Line B C) (Line H N) P]

-- Let R be the intersection of BC and OM
variables {R : Point}
variable [intersection_R : Intersection (Line B C) (Line O M) R]

-- Let S be the intersection of HR and OP
variables {S : Point}
variable [intersection_S : Intersection (Line H R) (Line O P) S]

-- The goal is to show that AHSO is a parallelogram
theorem parallelogram_AHSO : Parallelogram A H S O := 
sorry

end parallelogram_AHSO_l367_367266


namespace find_z_l367_367160

-- Use noncomputable theory if necessary
noncomputable theory

-- Define the values used in the problem
def mean_3 (a b c : ℕ) : ℚ := (a + b + c) / 3
def mean_2 (a z : ℚ) : ℚ := (a + z) / 2

-- Define the theorem to prove
theorem find_z (z : ℚ) : 
  mean_3 8 14 24 = mean_2 16 z -> 
  z = 44 / 3 :=
by
  -- Proof is omitted
  sorry

end find_z_l367_367160


namespace largest_n_proof_l367_367343

def largest_n_less_than_50000_divisible_by_7 (n : ℕ) : Prop :=
  n < 50000 ∧ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36) % 7 = 0

theorem largest_n_proof : ∃ n, largest_n_less_than_50000_divisible_by_7 n ∧ ∀ m, largest_n_less_than_50000_divisible_by_7 m → m ≤ n := 
sorry

end largest_n_proof_l367_367343


namespace eugene_degree_is_four_l367_367541

-- Problem definition based on the given conditions
def albert_degrees : ℕ := 1
def bassim_degrees : ℕ := 2
def clara_degrees : ℕ := 3
def daniel_degrees : ℕ := 4

-- Target statement to prove given the conditions
theorem eugene_degree_is_four (albert_degrees = 1) (bassim_degrees = 2) (clara_degrees = 3) (daniel_degrees = 4) : (∃ eugene_degree : ℕ, eugene_degree = 4) := 
  sorry

end eugene_degree_is_four_l367_367541


namespace t_equals_2S_div_V_plus_V₀_l367_367043

noncomputable def compute_t (g a V V₀ S : ℝ) : ℝ :=
  let denom := V + V₀ in
  2 * S / denom

theorem t_equals_2S_div_V_plus_V₀
  (g a V V₀ t S : ℝ)
  (hv : V = (g + a) * t + V₀)
  (hs : S = (1/2) * (g + a) * t^2 + V₀ * t) :
    t = compute_t g a V V₀ S := by
  sorry

end t_equals_2S_div_V_plus_V₀_l367_367043


namespace repeating_decimal_to_fraction_l367_367319

noncomputable def x : ℚ := 3 + 56 / 99

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 3 + 56 / 99) : x = 353 / 99 := 
by 
  rw h
  exact (3 + 56 / 99 : ℚ)
  sorry

end repeating_decimal_to_fraction_l367_367319


namespace least_n_for_multiple_of_101_l367_367912

noncomputable def a : ℕ → ℕ
| 10 := 101
| n := if h : n > 10 then 101 * a (n - 1) + n else 0

theorem least_n_for_multiple_of_101 :
  ∃ (n : ℕ), n > 10 ∧ a n % 101 = 0 ∧ ∀ (m : ℕ), m > 10 → m < n → a m % 101 ≠ 0 :=
sorry

end least_n_for_multiple_of_101_l367_367912


namespace bug_position_2010_jumps_l367_367114

def jump (pos : ℕ) : ℕ :=
  if pos % 2 = 0 then (pos + 3) % 6 else (pos + 1) % 6

def bug_position_after_jumps (start_pos jumps : ℕ) : ℕ :=
  (List.iterate jump jumps start_pos) + 1

theorem bug_position_2010_jumps : bug_position_after_jumps 5 2010 = 2 :=
by
  sorry

end bug_position_2010_jumps_l367_367114


namespace sum_inequality_l367_367397

theorem sum_inequality (a : ℕ → ℝ) (n : ℕ) (h : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (h' : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → a i < a j) :
  (∑ i in finset.range n, 1 / (1 + a (i+1)))^2 ≤ 
  1 / (a 1) + ∑ i in finset.range (n - 1), 1 / (a (i + 2) - a (i + 1)) :=
sorry

end sum_inequality_l367_367397


namespace equivalence_of_series_l367_367478

section
variable {a b c : ℕ → ℝ}

-- Given the sequences of positive numbers (a_n) and (b_n)
variable (ha : ∀ n, 0 < a n)
variable (hb : ∀ n, 0 < b n)

-- There exists a sequence (c_n) of positive numbers such that the series converge
def seq_satisfies (c : ℕ → ℝ) : Prop :=
  (∀ n, 0 < c n) ∧ summable (λ n, a n / c n) ∧ summable (λ n, c n / b n)

-- The condition to show
theorem equivalence_of_series :
  (∃ c : ℕ → ℝ, seq_satisfies ha hb c) ↔ summable (λ n, real.sqrt (a n / b n)) :=
  sorry
end

end equivalence_of_series_l367_367478


namespace find_y_values_l367_367352

variable (x y : ℝ)

theorem find_y_values 
    (h1 : 3 * x^2 + 9 * x + 4 * y - 2 = 0)
    (h2 : 3 * x + 2 * y - 6 = 0) : 
    y^2 - 13 * y + 26 = 0 := by
  sorry

end find_y_values_l367_367352


namespace coin_difference_max_min_l367_367941

/--
Paul owes Paula 45 cents and has a pocket full of 5-cent coins, 10-cent coins, 
and 25-cent coins. Prove that the difference between the largest and the 
smallest number of coins he can use to pay her exactly is 6.
-/
theorem coin_difference_max_min :
  let value := 45
  let denominations := [5, 10, 25]
  let min_coins := 1 + 2  -- Use one 25-cent coin + two 10-cent coins
  let max_coins := 45 / 5 -- Use nine 5-cent coins
  max_coins - min_coins = 6 :=
begin
  -- Definitions of values, denominations, min_coins, and max_coins
  let value := 45,
  let denominations := [5, 10, 25],
  let min_coins := 1 + 2, -- 3
  let max_coins := 45 / 5, -- 9
  -- Prove the difference is 6
  show max_coins - min_coins = 6,
  from sorry
end

end coin_difference_max_min_l367_367941


namespace greatest_integer_func_l367_367705

noncomputable def pi_approx : ℝ := 3.14159

theorem greatest_integer_func : (⌊2 * pi_approx - 6⌋ : ℝ) = 0 := 
by
  sorry

end greatest_integer_func_l367_367705


namespace triangle_height_l367_367445

theorem triangle_height (area base : ℝ) (h : ℝ) (h_area : area = 46) (h_base : base = 10) 
  (h_formula : area = (base * h) / 2) : 
  h = 9.2 :=
by
  sorry

end triangle_height_l367_367445


namespace length_of_tube_l367_367237

theorem length_of_tube (h1 : ℝ) (mass_water : ℝ) (rho : ℝ) (doubled_pressure : Bool) (g : ℝ) :
  h1 = 1.5 → 
  mass_water = 1000 → 
  rho = 1000 → 
  g = 9.8 →
  doubled_pressure = true →
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  intros h1_val mass_water_val rho_val g_val doubled_pressure_val
  have : ∃ h2, 29400 = 1000 * g * (h1 + h2) := sorry
  use 1.5
  assumption_sid
  sorry
  
end length_of_tube_l367_367237


namespace prove_angle_BAC_eq_60_l367_367575

-- Definitions of the conditions
noncomputable def TriangleABC_Isosceles (A B C P : Point) : Prop :=
  (dist(A, B) = dist(B, C)) ∧ -- Triangle ABC is isosceles with AB = BC
  (on_line(P, line(A, C)) ∧ dist(A, P) = dist(P, C)) ∧ -- P is on AC with AP = PC
  (angle_bisector(line(B, P), line(P, C))) ∧ -- Angle bisector of ∠BPC passes through P
  (angle(B, P, C) = 60) -- ∠PBC = 60°

-- Proof goal 
theorem prove_angle_BAC_eq_60 (A B C P : Point) (hconds : TriangleABC_Isosceles A B C P) : angle(A, B, C) = 60 := 
  sorry -- satisfying the proof is left as an exercise.

end prove_angle_BAC_eq_60_l367_367575


namespace randy_initial_blocks_l367_367518

theorem randy_initial_blocks (x : ℕ) (used_blocks : ℕ) (left_blocks : ℕ) 
  (h1 : used_blocks = 36) (h2 : left_blocks = 23) (h3 : x = used_blocks + left_blocks) :
  x = 59 := by 
  sorry

end randy_initial_blocks_l367_367518


namespace value_of_f_prime_at_1_l367_367446

def f (x : ℝ) : ℝ := (1/3) * x ^ 3 - deriv f 1 * x ^ 2 - x

theorem value_of_f_prime_at_1 : deriv f 1 = 0 := by
    sorry

end value_of_f_prime_at_1_l367_367446


namespace probability_ace_of_spades_top_l367_367652

noncomputable def total_cards : ℕ := 52
noncomputable def desired_outcomes : ℕ := 1

theorem probability_ace_of_spades_top (h1 : total_cards = 52) (h2 : desired_outcomes = 1) : 
  (desired_outcomes : ℝ) / total_cards = (1 / 52 : ℝ) :=
by
  rw [h1, h2]
  norm_num
  sorry

end probability_ace_of_spades_top_l367_367652


namespace parabola_focus_directrix_l367_367045

noncomputable def parabola_condition (p : ℝ) : Prop :=
  let focus : ℝ × ℝ := (p / 2, 0)
  (focus.1 + focus.2 - 2 = 0) ∧ (y^2 = 2*p*x) ∧ (x = -p/2)

theorem parabola_focus_directrix (p : ℝ) :
  parabola_condition 4 ∧ x = -2 :=
by
  sorry

end parabola_focus_directrix_l367_367045


namespace quadrant_I_range_l367_367921

def x (c : ℝ) : ℝ := 13 / (2 * c + 1)
def y (c : ℝ) : ℝ := (8 - 10 * c) / (2 * c + 1)

theorem quadrant_I_range (c : ℝ) : 
  (x c > 0) ∧ (y c > 0) ↔ (-1 / 2 < c ∧ c < 4 / 5) :=
by
  sorry

end quadrant_I_range_l367_367921


namespace find_m_plus_n_l367_367923

theorem find_m_plus_n (b : ℝ) (h1 : -17 ≤ b ∧ b ≤ 17) :
  ∃ m n : ℕ, nat.coprime m n ∧ (∃ (h2 : n ≠ 0), (m : ℝ) / (n : ℝ) = 29 / 34 ∧ m + n = 63) :=
by sorry

end find_m_plus_n_l367_367923


namespace sum_of_scores_achieved_in_three_ways_l367_367458

theorem sum_of_scores_achieved_in_three_ways :
  ∃ (correct_sum : ℕ), correct_sum ∈ {88, 90, 92, 94, 96} ∧
  (∀ S ∈ (finset.range 101).filter (λ s : ℕ,
    (finset.univ.val.countp (λ (c : ℕ), c ≤ 25 ∧ S - 4 * c ≥ 0 ∧ c + (S - 4 * c) ≤ 25) = 3)), true) :=
sorry

end sum_of_scores_achieved_in_three_ways_l367_367458


namespace convex_pentagon_area_l367_367300

noncomputable def area_of_pentagon (FG GH HI IJ JF r : ℝ) : ℝ :=
  let s : ℝ := (FG + GH + HI + IJ + JF) / 2 in
  r * s

theorem convex_pentagon_area (FG GH HI IJ JF r : ℝ)
  (hFG : FG = 7) (hGH : GH = 8) (hHI : HI = 8)
  (hIJ : IJ = 8) (hJF : JF = 9) :
  area_of_pentagon FG GH HI IJ JF r = 20 * r :=
by
  unfold area_of_pentagon
  rw [hFG, hGH, hHI, hIJ, hJF]
  norm_num
  sorry -- Proof to be completed

end convex_pentagon_area_l367_367300


namespace perpendicular_bisector_midpoint_l367_367925

theorem perpendicular_bisector_midpoint :
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  3 * R.1 - 2 * R.2 = -15 :=
by
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  sorry

end perpendicular_bisector_midpoint_l367_367925


namespace sum_of_cosines_l367_367140

theorem sum_of_cosines :
  (∃ (a b c : ℕ), ∀ x : ℝ, 
      sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2 → 
      cos (a * x) * cos (b * x) * cos (c * x) = 0)  ∧ a + b + c = 14 :=
begin
  sorry
end

end sum_of_cosines_l367_367140


namespace proposition_with_false_converse_l367_367212

-- Define the propositions and their converses
def PropA : Prop :=
  ∀ (l1 l2 : Line), parallel l1 l2 → ∀ (angle1 angle2 : Angle), corresponding_angles l1 l2 angle1 angle2 → angle1 = angle2

def ConversePropA : Prop :=
  ∀ (l1 l2 : Line), (∀ (angle1 angle2 : Angle), corresponding_angles l1 l2 angle1 angle2 → angle1 = angle2) → parallel l1 l2

def PropB : Prop :=
  ∀ (angle1 angle2 : Angle), vertical_angles angle1 angle2 → angle1 = angle2

def ConversePropB : Prop :=
  ∀ (angle1 angle2 : Angle), angle1 = angle2 → vertical_angles angle1 angle2

def PropC : Prop :=
  ∀ (a b : ℝ), |a| = |b| → a = b

def ConversePropC : Prop :=
  ∀ (a b : ℝ), a = b → |a| = |b|

def PropD : Prop :=
  ∀ (p : Point) (a b : Point), (on_perpendicular_bisector p a b) → distance p a = distance p b

def ConversePropD : Prop :=
  ∀ (p : Point) (a b : Point), distance p a = distance p b → on_perpendicular_bisector p a b

-- Statement to be proved in Lean, which is the equivalent math proof problem
theorem proposition_with_false_converse : ¬ ConversePropB := sorry

end proposition_with_false_converse_l367_367212


namespace coeff_of_x2_is_15_l367_367073

noncomputable def coeff_of_x2_in_expansion : ℕ :=
  let expr := (x^2 - 2*x + 1)^3
  in (expr.coeff 2)

theorem coeff_of_x2_is_15 : coeff_of_x2_in_expansion = 15 := by
  sorry

end coeff_of_x2_is_15_l367_367073


namespace range_of_b_l367_367047

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → (5 < b ∧ b < 7) :=
sorry

end range_of_b_l367_367047


namespace side_length_25_l367_367617

noncomputable def side_length_of_land_plot (area: ℝ) : ℝ := 
  Real.sqrt area

theorem side_length_25 (area: ℝ) (h: area = Real.sqrt 625) : side_length_of_land_plot area = 25 := 
by {
  rw [side_length_of_land_plot, h],
  have : Real.sqrt (Real.sqrt 625) = 25,
  sorry
}

end side_length_25_l367_367617


namespace triangle_third_side_l367_367415

theorem triangle_third_side {x : ℕ} (h1 : 3 < x) (h2 : x < 7) (h3 : x % 2 = 1) : x = 5 := by
  sorry

end triangle_third_side_l367_367415


namespace sum_adjacent_to_six_l367_367858

theorem sum_adjacent_to_six :
  ∀ (table : fin 3 × fin 3 → ℕ),
    (∀ i j, table i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∃! i j, table i j = 1) ∧
    (∃! i j, table i j = 2) ∧
    (∃! i j, table i j = 3) ∧
    (∃! i j, table i j = 4) ∧
    (∃! i j, table i j = 5) → 
    (∀ i j, 
      table i j = 5 → 
        let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                       (if i < 2 then table (i+1, j) else 0) + 
                       (if j > 0 then table (i, j-1) else 0) + 
                       (if j < 2 then table (i, j+1) else 0)
        in adj_sum = 9) →
    (∃ i j, table i j = 6 ∧
      let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                     (if i < 2 then table (i+1, j) else 0) + 
                     (if j > 0 then table (i, j-1) else 0) + 
                     (if j < 2 then table (i, j+1) else 0) 
      in adj_sum = 29) := sorry

end sum_adjacent_to_six_l367_367858


namespace min_digits_decimal_l367_367277

theorem min_digits_decimal :
  let numerator := 987654321
  let power_of_2 := 2^30
  let power_of_5 := 5^6
  let power_of_3 := 3
  let fraction := numerator / (power_of_2 * power_of_5 * power_of_3)
  ∃ (digits : ℕ), digits = 30 ∧ (decimal_digits_needed fraction = digits) :=
sorry

end min_digits_decimal_l367_367277


namespace problem_four_points_l367_367121

-- Define the problem conditions and statement in Lean
theorem problem_four_points (A B C X Y Z D E F : Type*)
  (h_circle : is_on_circle A B C)
  (h_acute : is_acute_triangle A B C)
  (h_X_on_circle : is_on_circle X A B C)
  (h_Y_on_circle : is_on_circle Y A B C)
  (h_Z_on_circle : is_on_circle Z A B C)
  (h_AX_perp_BC : is_perpendicular A X B C D)
  (h_BY_perp_AC : is_perpendicular B Y A C E)
  (h_CZ_perp_AB : is_perpendicular C Z A B F) :
  ∑ (AX AD BY BE CZ CF : ℝ), (AX / AD) + (BY / BE) + (CZ / CF) = 4 := 
sorry

end problem_four_points_l367_367121


namespace number_of_ways_to_place_pawns_l367_367028

theorem number_of_ways_to_place_pawns :
  let n := 5 in
  let number_of_placements := (n.factorial) in
  let number_of_permutations := (n.factorial) in
  number_of_placements * number_of_permutations = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l367_367028


namespace crews_complete_job_l367_367566

-- Define the productivity rates for each crew
variables (x y z : ℝ)

-- Define the conditions derived from the problem
def condition1 : Prop := 1/(x + y) = 1/z - 3/5
def condition2 : Prop := 1/(x + z) = 1/y
def condition3 : Prop := 1/(y + z) = 2/(7 * x)

-- Target proof: the combined time for all three crews
def target_proof : Prop := 1/(x + y + z) = 4/3

-- Final Lean 4 statement combining all conditions and proof requirement
theorem crews_complete_job (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : target_proof x y z :=
sorry

end crews_complete_job_l367_367566


namespace tom_age_difference_l367_367572

/-- 
Tom Johnson's age is some years less than twice as old as his sister.
The sum of their ages is 14 years.
Tom's age is 9 years.
Prove that the number of years less Tom's age is than twice his sister's age is 1 year. 
-/ 
theorem tom_age_difference (T S : ℕ) 
  (h₁ : T = 9) 
  (h₂ : T + S = 14) : 
  2 * S - T = 1 := 
by 
  sorry

end tom_age_difference_l367_367572


namespace evaluate_expression_l367_367698

theorem evaluate_expression : 
  let a := 3
  let b := 4
  (a^b)^a - (b^a)^b = -16245775 := 
by 
  sorry

end evaluate_expression_l367_367698


namespace avg_remaining_students_l367_367057

variable (n : ℕ) (hn : n > 20) (havg_class : 10) (havg_group : 18)

theorem avg_remaining_students (hn : n > 20) : 
  (10 : ℚ) * n = 270 + (15 : ℚ) * 18 →  
  (havg_class * (n : ℚ)) = 270 + (b * (n - 15)) → 
  ((∀ i < 15, hi i > 15) -> 
  (b = (10 * n - 270) / (n - 15))) :=
begin
  sorry
end

end avg_remaining_students_l367_367057


namespace sum_adjacent_to_6_is_29_l367_367873

def in_grid (n : ℕ) (grid : ℕ → ℕ → ℕ) := ∃ i j, grid i j = n

def adjacent_sum (grid : ℕ → ℕ → ℕ) (i j : ℕ) : ℕ :=
  grid (i-1) j + grid (i+1) j + grid i (j-1) + grid i (j+1)

def grid := λ i j =>
  if i = 0 ∧ j = 0 then 1 else
  if i = 2 ∧ j = 0 then 2 else
  if i = 0 ∧ j = 2 then 3 else
  if i = 2 ∧ j = 2 then 4 else
  if i = 1 ∧ j = 1 then 6 else 0

lemma numbers_positions_adjacent_5 :
  grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧
  grid 2 2 = 4 ∧ 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else 0 in
  adjacent_sum grid 1 0 = 1 + 2 + 6 :=
by sorry

theorem sum_adjacent_to_6_is_29 : 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else
                     if i = 0 ∧ j = 1 then 7 else
                     if i = 2 ∧ j = 1 then 8 else
                     if i = 1 ∧ j = 2 then 9 else 0 in
  adjacent_sum grid 1 1 = 5 + 7 + 8 + 9 :=
by sorry

end sum_adjacent_to_6_is_29_l367_367873


namespace function_properties_l367_367155

noncomputable def is_odd_function {f : ℝ → ℝ} := ∀ x : ℝ, f (-x) = -f x
noncomputable def is_increasing_function {f : ℝ → ℝ} := ∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2

theorem function_properties {f : ℝ → ℝ} 
  (h1 : ∀ x1 x2 : ℝ, f (x1 + x2) = f (x1) + f (x2))
  (h2 : ∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ x2 → f x1 ≤ f x2) : 
  is_odd_function f ∧ is_increasing_function f := 
sorry

end function_properties_l367_367155


namespace sum_of_adjacent_to_6_l367_367867

theorem sum_of_adjacent_to_6 :
  ∃ (grid : Fin 3 × Fin 3 → ℕ),
  (grid (0, 0) = 1 ∧ grid (0, 2) = 3 ∧ grid (2, 0) = 2 ∧ grid (2, 2) = 4 ∧
   ∀ i j, grid (i, j) ∈ finset.range 1 10 ∧ finset.univ.card = 9 ∧
   (grid (1, 0) + grid (1, 1) + grid (2, 1) = 9) ∧ 
   (grid (1, 1) = 6) ∧ 
   (sum_of_adjacent grid (1, 1) = 29))

where
  sum_of_adjacent (grid : Fin 3 × Fin 3 → ℕ) (x y : Fin 3 × Fin 3) : ℕ :=
  grid (x - 1, y) + grid (x + 1, y) + grid (x, y - 1) + grid (x, y + 1)
  sorry

end sum_of_adjacent_to_6_l367_367867


namespace least_three_digit_multiple_l367_367589

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l367_367589


namespace find_d_l367_367764

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) : d = 2 :=
by
  sorry

end find_d_l367_367764


namespace angle_bisector_excircle_inequality_l367_367760

/-- Given a triangle ABC, A-excircle intersects the angle bisector of ∠A at two points D and E,
with D lying on the segment AE. Prove the inequality AD / AE ≤ (BC^2) / (DE^2). -/
theorem angle_bisector_excircle_inequality 
  (A B C D E: Point) 
  (p : ℝ)   -- semi perimeter
  (a : ℝ)   -- side length BC
  (r_a : ℝ) -- A-excircle radius
  (h_a : ℝ) -- altitude from A to BC
  (angle_bisector_intersect : is_angle_bisector (∠ A) (D E))
  (intersect_condition : segment_contains (A E) D)
  (triangle : is_triangle (A B C)) 
  : (length (A D) / length (A E)) ≤ (a^2 / (2 * r_a)^2) := 
sorry

end angle_bisector_excircle_inequality_l367_367760


namespace point_on_line_min_distance_to_line_l367_367463

-- Problem 1: Proving that point P is on line l
theorem point_on_line (P : ℝ × ℝ) (h_P : P = (4 * real.cos (real.pi / 2), 4 * real.sin (real.pi / 2))) : 
  (∃ x y : ℝ, P = (x, y) ∧ x - y + 4 = 0) :=
  sorry

-- Problem 2: Proving the minimum distance from point Q on curve C to line l
theorem min_distance_to_line (α : ℝ) (Q : ℝ × ℝ) (h_Q : Q = (sqrt 3 * real.cos α, real.sin α)) 
  (h_α_range : 0 ≤ α ∧ α < 2 * real.pi) :
  ∃ d : ℝ, d = sqrt 2 ∧ ∀ ε > 0, ∃ β : ℝ, 0 ≤ β ∧ β < 2 * real.pi ∧ 
  abs (Q.1 - Q.2 + 4) / sqrt (2) < d + ε :=
  sorry

end point_on_line_min_distance_to_line_l367_367463


namespace area_ratio_of_concentric_circles_l367_367577

theorem area_ratio_of_concentric_circles (C1 C2 : ℝ) 
  (h1 : (60 / 360) * C1 = (30 / 360) * C2) : (C1 / C2)^2 = 1 / 4 := 
by 
  have h : C1 / C2 = 1 / 2 := by
    field_simp [h1]
  rw [h]
  norm_num

end area_ratio_of_concentric_circles_l367_367577


namespace smallest_c_a_l367_367568

def factorial : ℕ → ℕ
| 0        := 1
| (n + 1)  := (n + 1) * factorial n

theorem smallest_c_a  (a b c : ℕ) (h1 : a * b * c = factorial 9) (h2 : a < b) (h3 : b < c) :
  c - a = 216 :=
  sorry

end smallest_c_a_l367_367568


namespace inequality_proof_l367_367773

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h : (∑ i, a i) = 1) :
    (∑ i, a i ^ 4 / (a i ^ 3 + a i ^ 2 * a ((i + 1) % n) + 
    a i * a ((i + 1) % n) ^ 2 + a ((i + 1) % n) ^ 3)) ≥ 1 / 4 :=
sorry

end inequality_proof_l367_367773


namespace proof_solution_l367_367987

noncomputable def proof_problem (x : ℝ) : Prop :=
  (⌈2 * x⌉₊ : ℝ) - (⌊2 * x⌋₊ : ℝ) = 0 → (⌈2 * x⌉₊ : ℝ) - 2 * x = 0

theorem proof_solution (x : ℝ) : proof_problem x :=
by
  sorry

end proof_solution_l367_367987


namespace range_of_function_l367_367379

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3) ^ 2 - 6 * (Real.log x / Real.log 3) + 6

theorem range_of_function : 
  (set.range (f : set.Icc 1 81 → ℝ)) = set.Icc (-3) 6 :=
sorry

end range_of_function_l367_367379


namespace MN_parallel_PQ_l367_367122

theorem MN_parallel_PQ (A B C D M N P Q : ℝ → ℝ) (h1 : is_square A B C D)
(AM BC CD : ∀ x y : ℝ → ℝ, intersect_at M x y)
(AN CD : ∀ x y : ℝ → ℝ, intersect_at N x y)
(angle_MAN : ∠ A M N = 45)
(circle : is_circular_through A B C D) : parallel MN PQ :=
by
  sorry

end MN_parallel_PQ_l367_367122


namespace problem1_problem2_l367_367669

-- Problem 1: Prove the expression equals 5
theorem problem1 : (1 : ℚ) * ((1/3 : ℚ) - (3/4) + (5/6)) / (1/12) = 5 := by
  sorry

-- Problem 2: Prove the expression equals 7
theorem problem2 : ((-1 : ℤ)^2023 + |(1 - 0.5 : ℚ)| * ((-4)^2)) = 7 := by
  sorry

end problem1_problem2_l367_367669


namespace inverse_function_l367_367157

noncomputable def f (x : ℝ) : ℝ := 2^x - 1

noncomputable def f_inv (x : ℝ) : ℝ := log (x + 1) / log 2

theorem inverse_function :
  ∀ y : ℝ, y > -1 → f_inv (f y) = y :=
by
  intro y
  intro hy
  have h1 : f y = 2^y - 1 := by simp [f]
  have h2 : f_inv (2^y - 1) = log ((2^y - 1) + 1) / log 2 := by simp [f_inv]
  rw [h1, h2]
  have h3 : log (2^y) / log 2 = y := by sorry -- logarithm properties
  rw h3
  assumption

end inverse_function_l367_367157


namespace triangle_area_percentage_approx_l367_367254

noncomputable def triangle_area_percentage (s : ℝ) : ℝ :=
  let square_area := s^2
  let triangle_area := (sqrt 3) / 4 * s^2
  let pentagon_area := square_area + triangle_area
  (triangle_area / pentagon_area) * 100

theorem triangle_area_percentage_approx (s : ℝ) (hs : s > 0) :
  triangle_area_percentage s ≈ 19.106 :=
by
  sorry

end triangle_area_percentage_approx_l367_367254


namespace number_of_points_l367_367070

/-- 
Given the coordinates of point P(a, b) where a ≠ b,
both a and b are elements of the set {1, 2, 3, 4, 5, 6},
and the distance from point P to the origin |OP| is greater than or equal to 5,
the number of such points P is 20.
-/
theorem number_of_points : 
  ∃ (points : Finset (ℕ × ℕ)), 
  points.card = 20 ∧ 
  (∀ (p : ℕ × ℕ), p ∈ points → (p.1 ≠ p.2) ∧ p.1 ∈ {1, 2, 3, 4, 5, 6} ∧ p.2 ∈ {1, 2, 3, 4, 5, 6} ∧ (p.1^2 + p.2^2 ≥ 25)) :=
by
  sorry

end number_of_points_l367_367070


namespace one_minus_repeating_three_l367_367327

theorem one_minus_repeating_three : 1 - (0.\overline{3}) = 2 / 3 :=
by
  sorry

end one_minus_repeating_three_l367_367327


namespace policeman_catches_thief_l367_367359

/-
  From a police station situated on a straight road infinite in both directions, a thief has stolen a police car.
  Its maximal speed equals 90% of the maximal speed of a police cruiser. When the theft is discovered some time
  later, a policeman starts to pursue the thief on a cruiser. However, the policeman does not know in which direction
  along the road the thief has gone, nor does he know how long ago the car has been stolen. The goal is to prove
  that it is possible for the policeman to catch the thief.
-/
theorem policeman_catches_thief (v : ℝ) (T₀ : ℝ) (o₀ : ℝ) :
  (0 < v) →
  (0 < T₀) →
  ∃ T p, T₀ ≤ T ∧ p ≤ v * T :=
sorry

end policeman_catches_thief_l367_367359


namespace symmetric_sum_l367_367395

theorem symmetric_sum (a b : ℤ) (h1 : a = -4) (h2 : b = -3) : a + b = -7 := by
  sorry

end symmetric_sum_l367_367395


namespace vector_addition_proof_l367_367810

def vector_add (a b : ℤ × ℤ) : ℤ × ℤ :=
  (a.1 + b.1, a.2 + b.2)

theorem vector_addition_proof :
  let a := (2, 0)
  let b := (-1, -2)
  vector_add a b = (1, -2) :=
by
  sorry

end vector_addition_proof_l367_367810


namespace evaluate_expression_l367_367101

def decimal_digits : ℕ → ℕ 
| 1 := 9 
| 2 := 8 
| 3 := 7 
| 4 := 6 
| 5 := 5 
| 6 := 1 
| 7 := 2 
| 8 := 3 
| _ := 0 -- Let's assume digits beyond 8 return 0 for simplicity

theorem evaluate_expression :
  ∃ (n : ℕ), n ∈ {39, 29, 41, 31} ∧ 
  5 * decimal_digits (decimal_digits (decimal_digits (decimal_digits 5))) 
  + 2 * decimal_digits (decimal_digits (decimal_digits (decimal_digits 8))) = n :=
by
  sorry

end evaluate_expression_l367_367101


namespace total_weight_of_paper_l367_367666

theorem total_weight_of_paper :
  let bunch_sheets := 4
  let bundle_sheets := 2
  let heap_sheets := 20
  let color_weight := 0.03
  let white_weight := 0.05
  let scrap_weight := 0.04
  let colored_sheets := 3 * bundle_sheets
  let white_sheets := 2 * bunch_sheets
  let scrap_sheets := 5 * heap_sheets
  let total_weight := colored_sheets * color_weight + white_sheets * white_weight + scrap_sheets * scrap_weight
  total_weight = 4.58 :=
by
  let bunch_sheets := 4
  let bundle_sheets := 2
  let heap_sheets := 20
  let color_weight := 0.03
  let white_weight := 0.05
  let scrap_weight := 0.04
  let colored_sheets := 3 * bundle_sheets
  let white_sheets := 2 * bunch_sheets
  let scrap_sheets := 5 * heap_sheets
  let total_weight := colored_sheets * color_weight + white_sheets * white_weight + scrap_sheets * scrap_weight
  have h1 : colored_sheets = 6 := rfl
  have h2 : white_sheets = 8 := rfl
  have h3 : scrap_sheets = 100 := rfl
  have h4 : total_weight = 6 * 0.03 + 8 * 0.05 + 100 * 0.04 := rfl
  have h5 : total_weight = 0.18 + 0.40 + 4.00 := by rw [h4]
  have h6 : total_weight = 4.58 := by rw [h5]
  exact h6

end total_weight_of_paper_l367_367666


namespace nth_term_sequence_l367_367170

theorem nth_term_sequence (n : ℕ) : 
  (∃ (a b : ℕ → ℕ), (∀ n, a n = n + 2) ∧ (∀ n, b n = 2 * n + 3)) → 
  (∃ (term : ℕ → ℚ), ∀ n, term n = (a n : ℚ) / (b n : ℚ) → term n = (n + 2) / (2 * n + 3)) :=
  by intro h; obtain ⟨a, b, ha, hb⟩ := h; exists (λ n, (a n : ℚ) / (b n : ℚ)); intro n; rw [ha, hb]; sorry

end nth_term_sequence_l367_367170


namespace point_C_velocity_l367_367365

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l367_367365


namespace repeating_decimal_to_fraction_l367_367709

theorem repeating_decimal_to_fraction (x : ℚ) (hx : x = 0.363636...) : x = 4 / 11 := by
  sorry

end repeating_decimal_to_fraction_l367_367709


namespace buy_beams_l367_367529

theorem buy_beams (C T x : ℕ) (hC : C = 6210) (hT : T = 3) (hx: x > 0):
  T * (x - 1) = C / x :=
by
  rw [hC, hT]
  sorry

end buy_beams_l367_367529


namespace one_minus_repeat_three_l367_367323

theorem one_minus_repeat_three : 1 - (0.333333..<3̅) = 2 / 3 :=
by
  -- needs proof, currently left as sorry
  sorry

end one_minus_repeat_three_l367_367323


namespace probability_A_wins_championship_distribution_and_expectation_B_l367_367964

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l367_367964


namespace simson_line_l367_367662

-- Define the conditions as Lean types and terms.
variables (O A B C P D E F : Type)

-- Define the geometric setup of the problem.
variable [InscribedTriangle : ∀ {A B C : Type}, A ∈ circle(O) ∧ B ∈ circle(O) ∧ C ∈ circle(O)]
variable [IsPointOnCircle : P ∈ circle(O)]
variable [Perpendiculars : ∀ {P A B C : Type}, orthogonal(P, AB) ∧ orthogonal(P, BC) ∧ orthogonal(P, CA)]
variable [FeetOfPerpendiculars : ∀ {P A B C : Type}, foot(P, AB) = D ∧ foot(P, BC) = E ∧ foot(P, CA) = F]

-- The statement to be proved.
theorem simson_line (InscribedTriangle) (IsPointOnCircle) (Perpendiculars) (FeetOfPerpendiculars) :
  collinear D E F :=
sorry

end simson_line_l367_367662


namespace sin_two_pi_zero_l367_367290

theorem sin_two_pi_zero : Real.sin (2 * Real.pi) = 0 :=
by 
  -- We assume the necessary periodicity and value properties of the sine function
  sorry

end sin_two_pi_zero_l367_367290


namespace winning_strategy_exists_l367_367197

def prime_numbers : List ℕ := [
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61,
  67, 71, 73, 79, 83, 89, 97
]

def valid_prime (n : ℕ) : Prop := n ∈ prime_numbers

def last_digit (n : ℕ) : ℕ := n % 10

def first_digit (n : ℕ) : ℕ := n / 10 ^ (n.toString.length - 1)

def can_continue (n m : ℕ) : Prop :=
  last_digit n = first_digit m

noncomputable def strategy (primes : List ℕ) : Bool :=
  primes = [19, 97, 79]

theorem winning_strategy_exists : ∃ primes : List ℕ, strategy primes = true ∧ length primes = 3 :=
by
  sorry

end winning_strategy_exists_l367_367197


namespace solve_for_n_l367_367351

theorem solve_for_n (n : ℚ) (h : (1 / (n + 2)) + (2 / (n + 2)) + (n / (n + 2)) = 3) : n = -3/2 := 
by
  sorry

end solve_for_n_l367_367351


namespace distinct_eight_numbers_prime_distinct_choices_l367_367228

-- Definitions
def eight_derived_numbers (a b c : ℕ) : set ℕ :=
{a + b + c, a + bc, b + ac, c + ab, (a + b) * c, (b + c) * a, (c + a) * b, a * b * c}

def d (n : ℕ) : ℕ := (n.divisors).card  -- Number of positive divisors of n

-- Theorem statements
theorem distinct_eight_numbers (a b c n : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_bound : n / 2 < a ∧ n / 2 < b ∧ n / 2 < c ∧ a ≤ n ∧ b ≤ n ∧ c ≤ n) :
  (eight_derived_numbers a b c).card = 8 :=
sorry

theorem prime_distinct_choices (p n : ℕ) (hp : p.prime) (h_n : n ≥ p ^ 2) :
  ∃ (b c : ℕ), b ≠ c ∧ (p + 1 ≤ b ∧ b ≤ n) ∧ (p + 1 ≤ c ∧ c ≤ n) 
  ∧ (eight_derived_numbers p b c).card < 8 ↔ (d (p - 1)) :=
sorry

end distinct_eight_numbers_prime_distinct_choices_l367_367228


namespace customer_difference_l367_367657

theorem customer_difference (initial_customers after_customers : ℕ) 
  (h1 : initial_customers = 19) (h2 : after_customers = 4) : 
  initial_customers - after_customers = 15 := 
by 
  rw [h1, h2]
  norm_num

end customer_difference_l367_367657


namespace max_visible_faces_sum_l367_367134

-- Definition for the conditions of the problem
def opposite_faces (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 5
  | 3 => 4
  | 4 => 3
  | 5 => 2
  | 6 => 1
  | _ => 0 -- this case shouldn't happen in our context

def sum_opposite_faces (n m : ℕ) : ℕ :=
  if m = opposite_faces n then
    n + m
  else
    0

-- The main definition to calculate the maximum possible sum for each die
def max_sum_die_exposed_faces (exposed_faces : ℕ) : ℕ :=
  match exposed_faces with
  | 2 => 7 -- since the only visible faces will be opposites, summing to 7
  | 3 => 6 + 7 -- we place 6 on one face, and opposite faces summing to 7
  | 4 => 5 + 6 + 7 -- we place 5 and 6 on some faces, and opposites summing to 7
  | 5 => 2 + 3 + 4 + 5 + 6 -- we hide 1 to maximize the sum
  | _ => 0 -- this case shouldn't happen in our problem
  end

-- The final proof statement.
theorem max_visible_faces_sum :
  max_sum_die_exposed_faces 5 + 
  2 * max_sum_die_exposed_faces 3 + 
  2 * max_sum_die_exposed_faces 4 + 
  max_sum_die_exposed_faces 2 = 89 := by
  sorry

end max_visible_faces_sum_l367_367134


namespace complex_number_problem_l367_367407

noncomputable def z (cos_value : ℂ) := by sorry

theorem complex_number_problem
  (z : ℂ)
  (hz : z + z⁻¹ = 2 * real.cos (5 * real.pi / 180)) :
  z^100 + z^(-100) = -1.92 :=
sorry

end complex_number_problem_l367_367407


namespace necessary_and_sufficient_condition_l367_367403

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (a > b) ↔ (a - 1/a > b - 1/b) :=
sorry

end necessary_and_sufficient_condition_l367_367403


namespace greatest_distance_between_circle_centers_l367_367189

-- Define the dimensions of the rectangle
def rectangle_length : ℝ := 20
def rectangle_width : ℝ := 15

-- Define the diameter of the circles
def circle_diameter : ℝ := 4

-- Define the problem as a theorem in Lean
theorem greatest_distance_between_circle_centers :
  ∃ d : ℝ, d = sqrt (16^2 + 11^2) ∧ d = sqrt 377 := 
begin
  use sqrt (16^2 + 11^2),
  split,
  {
    calc sqrt (16^2 + 11^2) = sqrt (256 + 121) : by rw [sq_16, sq_11]
                        ... = sqrt 377         : by refl,
  },
  {
    refl,
  }
end

-- auxiliary calculations
lemma sq_16 : 16^2 = 256 :=
by norm_num

lemma sq_11 : 11^2 = 121 :=
by norm_num

end greatest_distance_between_circle_centers_l367_367189


namespace cookies_left_for_Monica_l367_367935

noncomputable def total_cookies : ℕ := 400

noncomputable def father_ate : ℕ := 0.10 * total_cookies
noncomputable def mother_ate : ℕ := father_ate / 2
noncomputable def brother_ate : ℕ := mother_ate + 2
noncomputable def sister_ate : ℕ := 1.5 * brother_ate
noncomputable def aunt_ate : ℕ := 2 * father_ate
noncomputable def cousin_ate : ℕ := 0.80 * aunt_ate
noncomputable def grandmother_ate : ℕ := cousin_ate / 3

noncomputable def total_eaten : ℕ := father_ate + mother_ate + brother_ate + sister_ate + aunt_ate + cousin_ate + grandmother_ate

theorem cookies_left_for_Monica : total_cookies - total_eaten = 120 := by
  sorry

end cookies_left_for_Monica_l367_367935


namespace angle_between_bisectors_of_trihedral_angle_l367_367619

noncomputable def angle_between_bisectors_trihedral (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) : ℝ :=
  60

theorem angle_between_bisectors_of_trihedral_angle (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) :
  angle_between_bisectors_trihedral α β γ hα hβ hγ = 60 := 
sorry

end angle_between_bisectors_of_trihedral_angle_l367_367619


namespace quadrilateral_ABCD_AB_eq_p_plus_sqrt_q_l367_367065

theorem quadrilateral_ABCD_AB_eq_p_plus_sqrt_q (BC CD AD : ℝ) (angle_A angle_B : ℝ) (h1 : BC = 8)
  (h2 : CD = 12) (h3 : AD = 10) (h4 : angle_A = 60) (h5 : angle_B = 60) : 
  ∃ (p q : ℤ), AB = p + real.sqrt q ∧ p + q = 150 :=
by
  sorry

end quadrilateral_ABCD_AB_eq_p_plus_sqrt_q_l367_367065


namespace sale_price_for_45_percent_profit_l367_367556

theorem sale_price_for_45_percent_profit (C S_p S_l : ℝ) (h1 : S_l = 448) (h2 : 0.45 * C = 928) (h3 : S_p - C = C - S_l) : S_p = 3676.4444 :=
by {
  have hC : C = 928 / 0.45, 
  rw div_eq_inv_mul, 
  exact (eq_inv_mul_of_mul_eq h2),
  rw ← hC at h3,
  rw h1 at h3,
  linarith, 
  sorry
}

end sale_price_for_45_percent_profit_l367_367556


namespace identity_functions_l367_367826

theorem identity_functions (g : ℝ → ℝ) (h : g = fun x => real.cbrt (x ^ 3)) :
  ∀ x : ℝ, g x = x :=
by
  sorry

end identity_functions_l367_367826


namespace fraction_weevils_25_percent_l367_367087

-- Define the probabilities
def prob_good_milk : ℝ := 0.8
def prob_good_egg : ℝ := 0.4
def prob_all_good : ℝ := 0.24

-- The problem definition and statement
def fraction_weevils (F : ℝ) : Prop :=
  0.32 * (1 - F) = 0.24

theorem fraction_weevils_25_percent : fraction_weevils 0.25 :=
by sorry

end fraction_weevils_25_percent_l367_367087


namespace no_such_real_m_l367_367400

noncomputable def A : set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 4 }
noncomputable def B (m : ℝ) : set ℝ := { x : ℝ | 1 - m ≤ x ∧ x ≤ 3 * m - 2 }

theorem no_such_real_m (m : ℝ) (h : 1 < m) : ¬ (∀ x : ℝ, x ∈ A → x ∈ (B m)) :=
by 
  sorry

end no_such_real_m_l367_367400


namespace check_sufficient_condition_for_eq_l367_367308

theorem check_sufficient_condition_for_eq (a b c : ℤ) (h : a = c - 1 ∧ b = a - 1) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 1 := 
by
  sorry

end check_sufficient_condition_for_eq_l367_367308


namespace one_elephant_lake_empty_in_365_days_l367_367250

variables (C K V : ℝ)
variables (t : ℝ)

noncomputable def lake_empty_one_day (C K V : ℝ) := 183 * C = V + K
noncomputable def lake_empty_five_days (C K V : ℝ) := 185 * C = V + 5 * K

noncomputable def elephant_time (C K V t : ℝ) : Prop :=
  (t * C = V + t * K) → (t = 365)

theorem one_elephant_lake_empty_in_365_days (C K V t : ℝ) :
  (lake_empty_one_day C K V) →
  (lake_empty_five_days C K V) →
  (elephant_time C K V t) := by
  intros h1 h2 h3
  sorry

end one_elephant_lake_empty_in_365_days_l367_367250


namespace one_minus_repeating_decimal_l367_367324

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ := x

theorem one_minus_repeating_decimal:
  ∀ (x : ℚ), x = 1/3 → 1 - x = 2/3 :=
by
  sorry

end one_minus_repeating_decimal_l367_367324


namespace f_2011_value_l367_367909

open Set Function

-- Define the set B
def B : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

-- Define the function f : B -> ℝ with the given functional equation
axiom f : B → ℝ
axiom f_property : ∀ x : B, f x + f (2 - 1 / x) = Real.log (|x - 1|)

-- The proof goal
theorem f_2011_value : 
  f (2011 : ℚ) = Real.log (6031 / 4021) := 
sorry

end f_2011_value_l367_367909


namespace type_a_products_total_l367_367246

theorem type_a_products_total
  (A B C D : ℕ)
  (h_geometric_seq : ∃ r : ℕ, B = A * r ∧ C = B * r ∧ D = C * r)
  (h_total_3000 : A + B + C + D = 3000)
  (extracted_total_150 : 150)
  (extracted_bd_total_100 : 100)
  (h_extracted_correct : True)   -- Dummy placeholder for extraction operation conditions
  : A = 200 := sorry

end type_a_products_total_l367_367246


namespace effective_avg_percent_increase_l367_367875

noncomputable def initial_population : ℝ := 175000
noncomputable def final_population : ℝ := 297500
noncomputable def birth_rate : ℝ := 0.015
noncomputable def death_rate : ℝ := 0.01
noncomputable def immigration_rate : ℝ := 0.004
noncomputable def emigration_rate : ℝ := 0.003
noncomputable def years : ℝ := 10

theorem effective_avg_percent_increase :
  let net_growth_rate := birth_rate + immigration_rate - death_rate - emigration_rate in
  let cagr := (final_population / initial_population)^(1 / years) - 1 in
  cagr * 100 = 5.477 :=
by
  sorry

end effective_avg_percent_increase_l367_367875


namespace polygon_visibility_l367_367516

variables {V : Type*} [inner_product_space ℝ V] (points : list (V × V))

def visible_from (O : V) : Prop :=
∀ (P : V), P ∈ points → ∃ R ∈ points, ∀ Q ∈ segments O P, Q ∈ R

def side_fully_visible (P : V) : Prop :=
∃ A B : V, (A, B) ∈ points ∧ (∀ Q ∈ segments A B, Q = A ∨ Q = B ∨ (∃ O, visible_from O ∧ Q ∈ segments O P))

theorem polygon_visibility (O : V) :
  (visible_from O points) → (∀ X : V, ∃ (A B : V), (A, B) ∈ points ∧ side_fully_visible X) :=
sorry

end polygon_visibility_l367_367516


namespace values_of_m_l367_367019

theorem values_of_m (m : ℝ) : 
  (2 ∈ ({m - 1, 2*m, m^2 - 1} : set ℝ)) ↔ 
  (m = 3 ∨ m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
by
  sorry

end values_of_m_l367_367019


namespace noah_large_painting_price_l367_367507

-- Definition of conditions
def price_small_painting : ℕ := 30
def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def this_month_sales_total : ℕ := 1200
def this_month_sales_multiplier : ℕ := 2

-- Lean statement of the problem
theorem noah_large_painting_price : 
  ∃ (L : ℕ), this_month_sales_total = (this_month_sales_multiplier * last_month_large_sales * L + this_month_sales_multiplier * last_month_small_sales * price_small_painting) ∧ L = 60 := 
begin
  use 60,
  split,
  { calc  this_month_sales_total = 1200              : by rfl
                         ... = 2 * 8 * 60 + 2 * 4 * 30 : by sorry },
  { refl }
end

end noah_large_painting_price_l367_367507


namespace point_C_velocity_l367_367366

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l367_367366


namespace Sara_taller_than_Joe_l367_367520

noncomputable def Roy_height := 36

noncomputable def Joe_height := Roy_height + 3

noncomputable def Sara_height := 45

theorem Sara_taller_than_Joe : Sara_height - Joe_height = 6 :=
by
  sorry

end Sara_taller_than_Joe_l367_367520


namespace problem_l367_367801

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + (a - 2) * x

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1 / 3) * m * x^3 - m * x

theorem problem (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, x ∈ Ioi 0 → deriv (λ x, Real.log x + (a - 2) * x) x = 0 → a = 1)
  ∧ (∀ x : ℝ, x ∈ Ioi 0 → x < 1 → deriv (λ x, Real.log x - x) x > 0)
  ∧ (∀ x : ℝ, x ∈ Ioi 0 → x > 1 → deriv (λ x, Real.log x - x) x < 0)
  ∧ (f 1 1 = -1)
  ∧ (∀ x1 ∈ Ioo 1 2, ∃ x2 ∈ Ioo 1 2, f x1 1 = g x2 m)
  ∧ (range (λ x, f x 1) = set.Ioo (Real.log 2 - 2) -1)
  ∧ (range (λ x, g x m) = set.Ioo -(2 / 3 * m) (2 / 3 * m))
  ∧ (m ∈ set.Ici (3 - (3 / 2) * Real.log 2)) :=
by
  -- Proof goes here
  sorry

end problem_l367_367801


namespace derivative_at_minus_five_l367_367003

/-- Given the function f(x) = 1/x, the derivative f'(-5) is -1/25 --/
theorem derivative_at_minus_five : (deriv (λ x : ℝ, 1 / x) (-5) = -1 / 25) := by
  sorry

end derivative_at_minus_five_l367_367003


namespace range_of_a_l367_367428

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * log x

theorem range_of_a (a : ℝ) :
  (∀ t : ℝ, t ≥ 1 → f (2 * t - 1) a ≥ 2 * f t a - 3) ↔ a ≤ 2 :=
by
  sorry

end range_of_a_l367_367428


namespace a_n_formula_l367_367007

noncomputable def a : ℕ → ℝ
| 1       := 2
| (n + 1) := a n + log (1 + 1 / n)

theorem a_n_formula (n : ℕ) (h : n ≥ 1) : a n = 2 + log n := 
by
if h : n = 1 then 
  rw h
  simp
else
  sorry

end a_n_formula_l367_367007


namespace Exercise_l367_367748

-- Define the given conditions and what needs to be proved
variables {m n : Line} {α β : Plane}

-- State the theorem in Lean 4
theorem Exercise
  (h1 : different_lines m n)
  (h2 : non_coincident_planes α β)
  (h3 : perpendicular_to_plane m α)
  (h4 : not_parallel m n)
  (h5 : not_parallel n β) :
  perpendicular_to_plane α β :=
sorry

end Exercise_l367_367748


namespace calculate_smaller_sphere_radius_l367_367765

noncomputable def smaller_sphere_radius (r1 r2 r3 r4 : ℝ) : ℝ := 
  if h : r1 = 2 ∧ r2 = 2 ∧ r3 = 3 ∧ r4 = 3 then 
    6 / 11 
  else 
    0

theorem calculate_smaller_sphere_radius :
  smaller_sphere_radius 2 2 3 3 = 6 / 11 :=
by
  sorry

end calculate_smaller_sphere_radius_l367_367765


namespace daniel_original_noodles_l367_367680

-- Define the total number of noodles Daniel had originally
def original_noodles : ℕ := 81

-- Define the remaining noodles after giving 1/3 to William
def remaining_noodles (n : ℕ) : ℕ := (2 * n) / 3

-- State the theorem
theorem daniel_original_noodles (n : ℕ) (h : remaining_noodles n = 54) : n = original_noodles := by sorry

end daniel_original_noodles_l367_367680


namespace binomial_coeff_x_solution_l367_367823

theorem binomial_coeff_x_solution (x : ℕ) (hx : binomial 12 (x + 1) = binomial 12 (2 * x - 1)) : 
    x = 2 ∨ x = 4 :=
by
  -- Proof will be here. 
  sorry

end binomial_coeff_x_solution_l367_367823


namespace solve_equation_l367_367333

theorem solve_equation (x : ℝ) : (x ≈ 20.105 ∨ x ≈ 0.895) ↔ x^2 + 6 * x + 6 * x * sqrt(x + 2) = 24 :=
by
  -- Proof goes here
  sorry

end solve_equation_l367_367333


namespace cos2_add_3sin2_eq_2_l367_367016

theorem cos2_add_3sin2_eq_2 (x : ℝ) (hx : -20 < x ∧ x < 100) (h : Real.cos x ^ 2 + 3 * Real.sin x ^ 2 = 2) : 
  ∃ n : ℕ, n = 38 := 
sorry

end cos2_add_3sin2_eq_2_l367_367016


namespace card_placement_unique_sum_identification_l367_367230

open Finset

theorem card_placement_unique_sum_identification :
  let cards := range (100 + 1)  -- 100 cards numbered from 1 to 100
  in -- ways to place the cards into 3 boxes such that each pair sum identifies the third box
  ∃ (A B C : Finset ℕ), (A ∪ B ∪ C = cards) ∧ (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (B ∩ C = ∅) ∧ (0 < card A) ∧ (0 < card B) ∧ (0 < card C)
  ∧ (∀ (a ∈ A) (b ∈ B) (c ∈ C), finset.sum (A, B, C, a, b, c) = 12) :=
by sorry

end card_placement_unique_sum_identification_l367_367230


namespace simplify_trig_expression_l367_367133

theorem simplify_trig_expression (α : ℝ) : 
  (sin (α + π))^2 * cos (π + α) * cos (-α - 2 * π) / 
  (tan (π + α) * (sin (π / 2 + α))^3 * sin (-α - 2 * π)) = 1 :=
by
  sorry

end simplify_trig_expression_l367_367133


namespace abs_diff_base5_l367_367717

def base5_add (a b c : Nat) : Prop := (a + b + c) % 5

noncomputable def A : Nat := 3
noncomputable def B : Nat := 2

-- condition 1: A_5 + B_5 + 2_5 = 4_5
def condition1 : Prop := base5_add A B 2 = 4

-- condition 2: 1_5 + B_5 + 1_5 = 2_5 with carry
def condition2 : Prop := base5_add 1 B 1 = 2

-- condition 3: B + 3 = A + 1
def condition3 : Prop := B + 3 = A + 1

-- Final theorem to prove the absolute difference
theorem abs_diff_base5 :
  condition1 ∧ condition2 ∧ condition3 → | A - B | = 1 :=
by
  sorry -- Proof to be filled in by a theorem prover

end abs_diff_base5_l367_367717


namespace sailing_speed_l367_367649

noncomputable def min_average_speed (distance : ℝ) (admit_rate : ℝ) (admit_time : ℝ) 
  (sink_capacity : ℝ) (pump_rate : ℝ) : ℝ :=
  let admit_rate_per_hour := admit_rate / (admit_time / 60)
  let net_inflow_rate := admit_rate_per_hour - pump_rate
  let time_to_sink := sink_capacity / net_inflow_rate
  distance / time_to_sink

theorem sailing_speed (distance admit_rate admit_time sink_capacity pump_rate : ℝ) 
  (h_distance : distance = 150)
  (h_admit_rate : admit_rate = 13 / 3)
  (h_admit_time : admit_time = 5 / 2)
  (h_sink_capacity : sink_capacity = 180)
  (h_pump_rate : pump_rate = 9) :
  min_average_speed distance admit_rate admit_time sink_capacity pump_rate ≈ 79.15 :=
by
  sorry

end sailing_speed_l367_367649


namespace greater_number_l367_367615

theorem greater_number (x y : ℕ) (h1 : x * y = 2048) (h2 : x + y - (x - y) = 64) : x = 64 :=
by
  sorry

end greater_number_l367_367615


namespace infinite_solutions_l367_367908

noncomputable def exists_integers_r_s_t (n : ℕ) (h : n ≠ (nat.cbrt n) ^ 3) (a : ℝ) (b : ℝ) (c : ℝ) := 
  ∃ (r s t : ℤ), r ≠ 0 ∨ s ≠ 0 ∨ t ≠ 0 ∧ r * a + s * b + t * c = 0

theorem infinite_solutions :
  ∃ (f : ℕ → ℕ), (∀ m : ℕ, 0 < m → f m < f (m + 1)) ∧ (∀ m : ℕ, 
  let n := f m in 
  let a := real.cbrt n in
  let b := 1 / (a - ↑(int.floor a)) in
  let c := 1 / (b - ↑(int.floor b)) in
  exists_integers_r_s_t n (by 
    by_contra h, 
    exact nat.lt_irrefl n (nat.lt_of_le_of_lt (nat.cbrt_lt_self (nat.succ_pos n)) 
      (by { have := nat.cbrt_nonneg n, linarith }))) a b c) := sorry

end infinite_solutions_l367_367908


namespace cylindrical_to_rectangular_l367_367678

theorem cylindrical_to_rectangular :
  ∀ (r θ z : ℝ), r = 6 → θ = 7 * Real.pi / 4 → z = -2 →
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2 ∧ z = -2 :=
by
  intro r θ z hr hθ hz
  simp only [hr, hθ, hz]
  rw [Real.cos_eq_div, Real.sin_eq_div]
  have hcos : Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 := sorry
  have hsin : Real.sin (7 * Real.pi / 4) = - Real.sqrt 2 / 2 := sorry
  rw [hcos, hsin]
  split
  { ring }
  split
  { ring }
  { refl }

end cylindrical_to_rectangular_l367_367678


namespace half_angle_quadrant_l367_367107

theorem half_angle_quadrant
  (α : ℝ)
  (h1 : ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)
  (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
  ∃ k : ℤ, k * Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi * 3 / 4 ∧ Real.cos (α / 2) ≤ 0 := sorry

end half_angle_quadrant_l367_367107


namespace complex_quadrant_l367_367807

open Complex

theorem complex_quadrant
  (z : ℂ)
  (i_unit : ℂ = Complex.I)
  (M : Set ℂ := {1, z * (1 + Complex.I)})
  (N : Set ℂ := {3, 4})
  (union_condition : M ∪ N = {1, 2, 3, 4}) :
  z * (1 + Complex.I) = 2 → (z = 1 - Complex.I) ∧ (1 ≥ 0 ∧ -1 < 0) :=
by
  sorry

end complex_quadrant_l367_367807


namespace inequality_not_satisfied_integer_values_count_l367_367739

theorem inequality_not_satisfied_integer_values_count :
  ∃ (n : ℕ), n = 5 ∧ ∀ (x : ℤ), 3 * x^2 + 17 * x + 20 ≤ 25 → x ∈ [-4, -3, -2, -1, 0] :=
  sorry

end inequality_not_satisfied_integer_values_count_l367_367739


namespace determine_h_l367_367301

theorem determine_h (h : ℚ[X]) :
    (16 * X ^ 4 + 5 * X ^ 3 - 4 * X + 2 + h
    =
    -8 * X ^ 3 + 7 * X ^ 2 - 6 * X + 5)
    → h = -16 * X ^ 4 - 13 * X ^ 3 + 7 * X ^ 2 - 2 * X + 3 :=
by
  intro h_eqn
  sorry

end determine_h_l367_367301


namespace sum_adjacent_to_6_is_29_l367_367850
-- Import the Mathlib library for the necessary tools and functions

/--
  In a 3x3 table filled with numbers from 1 to 9 such that each number appears exactly once, 
  with conditions: 
    * (1, 1) contains 1, (3, 1) contains 2, (1, 3) contains 3, (3, 3) contains 4,
    * The sum of the numbers in the cells adjacent to the cell containing 5 is 9,
  Prove that the sum of the numbers in the cells adjacent to the cell containing 6 is 29.
-/
theorem sum_adjacent_to_6_is_29 
  (table : Fin 3 → Fin 3 → Fin 9)
  (H_uniqueness : ∀ i j k l, (table i j = table k l) → (i = k ∧ j = l))
  (H_valid_entries : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)
  (H_initial_positions : table 0 0 = 1 ∧ table 2 0 = 2 ∧ table 0 2 = 3 ∧ table 2 2 = 4)
  (H_sum_adj_to_5 : ∃ (i j : Fin 3), table i j = 5 ∧ 
                      ((i > 0 ∧ table (i-1) j +
                       (i < 2 ∧ table (i+1) j) +
                       (j > 0 ∧ table i (j-1)) +
                       (j < 2 ∧ table i (j+1))) = 9)) :
  ∃ i j, table i j = 6 ∧
  (i > 0 ∧ table (i-1) j +
   (i < 2 ∧ table (i+1) j) +
   (j > 0 ∧ table i (j-1)) +
   (j < 2 ∧ table i (j+1))) = 29 := sorry

end sum_adjacent_to_6_is_29_l367_367850


namespace sum_of_adjacent_to_6_l367_367863

theorem sum_of_adjacent_to_6 :
  ∃ (grid : Fin 3 × Fin 3 → ℕ),
  (grid (0, 0) = 1 ∧ grid (0, 2) = 3 ∧ grid (2, 0) = 2 ∧ grid (2, 2) = 4 ∧
   ∀ i j, grid (i, j) ∈ finset.range 1 10 ∧ finset.univ.card = 9 ∧
   (grid (1, 0) + grid (1, 1) + grid (2, 1) = 9) ∧ 
   (grid (1, 1) = 6) ∧ 
   (sum_of_adjacent grid (1, 1) = 29))

where
  sum_of_adjacent (grid : Fin 3 × Fin 3 → ℕ) (x y : Fin 3 × Fin 3) : ℕ :=
  grid (x - 1, y) + grid (x + 1, y) + grid (x, y - 1) + grid (x, y + 1)
  sorry

end sum_of_adjacent_to_6_l367_367863


namespace determine_intersection_l367_367499

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def S := {1, 2, 4, 5}
def T := {3, 4, 5, 7}
def complement_U_T := U \ T

theorem determine_intersection :
  S ∩ complement_U_T = {1, 2} :=
by
  sorry

end determine_intersection_l367_367499


namespace larger_number_l367_367613

theorem larger_number (HCF LCM a b : ℕ) (h_hcf : HCF = 28) (h_factors: 12 * 15 * HCF = LCM) (h_prod : a * b = HCF * LCM) :
  max a b = 180 :=
sorry

end larger_number_l367_367613


namespace parallelogram_area_l367_367102

def v : ℝ × ℝ := (7, 4)
def w : ℝ × ℝ := (2, -9)

def area_parallelogram (v w : ℝ × ℝ) : ℝ :=
  (v.1 * (2 * w.2) - v.2 * (2 * w.1)).abs

theorem parallelogram_area :
  area_parallelogram v w = 142 :=
by
  -- Proof omitted
  sorry

end parallelogram_area_l367_367102


namespace people_on_trolley_l367_367263

-- Given conditions
variable (X : ℕ)

def initial_people : ℕ := 10

def second_stop_people : ℕ := initial_people - 3 + 20

def third_stop_people : ℕ := second_stop_people - 18 + 2

def fourth_stop_people : ℕ := third_stop_people - 5 + X

-- Prove the current number of people on the trolley is 6 + X
theorem people_on_trolley (X : ℕ) : 
  fourth_stop_people X = 6 + X := 
by 
  unfold fourth_stop_people
  unfold third_stop_people
  unfold second_stop_people
  unfold initial_people
  sorry

end people_on_trolley_l367_367263


namespace liked_both_desserts_l367_367053

noncomputable def total_students : ℕ := 50
noncomputable def apple_pie_lovers : ℕ := 22
noncomputable def chocolate_cake_lovers : ℕ := 20
noncomputable def neither_dessert_lovers : ℕ := 17
noncomputable def both_desserts_lovers : ℕ := 9

theorem liked_both_desserts :
  (total_students - neither_dessert_lovers) + both_desserts_lovers = apple_pie_lovers + chocolate_cake_lovers - both_desserts_lovers :=
by
  sorry

end liked_both_desserts_l367_367053


namespace buying_beams_l367_367528

/-- Problem Statement:
Given:
1. The total money for beams is 6210 wen.
2. The transportation cost per beam is 3 wen.
3. Removing one beam means the remaining beams' total transportation cost equals the price of one beam.

Prove: 3 * (x - 1) = 6210 / x
-/
theorem buying_beams (x : ℕ) (h₁ : x > 0) (h₂ : 6210 % x = 0) :
  3 * (x - 1) = 6210 / x :=
sorry

end buying_beams_l367_367528


namespace solve_equation_l367_367981

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l367_367981


namespace solution_set_bx2_2ax_c3b_min_value_a2_c2_over_a_c_min_value_a_2b_4c_over_b_a_l367_367757

-- Problem 1
theorem solution_set_bx2_2ax_c3b (a b c : ℝ) (y : ℝ → ℝ)
  (h : y = λ x, a * x^2 + b * x + c)
  (h_cond : ∀ x, -3 < x ∧ x < 4 → y x > 0) :
  ∀ x, -3 < x ∧ x < 5 → b * x^2 + 2 * a * x - (c + 3 * b) < 0 :=
sorry

-- Problem 2
theorem min_value_a2_c2_over_a_c (a c : ℝ) 
  (y : ℝ → ℝ) 
  (h_eq : ∀ x, y x = a * x^2 + 2 * x + c)
  (h_cond1 : ∀ x, y x ≥ 0)
  (h_cond2 : a > c)
  (h_cond3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + c = 0) :
  ∃ min_val : ℝ,  min_val = 2 * real.sqrt 2 ∧
  (∀ r : ℝ, r = (a^2 + c^2) / (a - c) → r ≥ min_val) :=
sorry

-- Problem 3
theorem min_value_a_2b_4c_over_b_a (a b c : ℝ)
  (y : ℝ → ℝ)
  (h_eq : ∀ x, y x = a * x^2 + b * x + c)
  (h_cond1 : ∀ x, y x ≥ 0)
  (h_cond2 : a < b) :
  ∃ min_val : ℝ, min_val = 8 ∧
  (∀ r : ℝ, r = (a + 2 * b + 4 * c) / (b - a) → r ≥ min_val) :=
sorry

end solution_set_bx2_2ax_c3b_min_value_a2_c2_over_a_c_min_value_a_2b_4c_over_b_a_l367_367757


namespace buy_beams_l367_367530

theorem buy_beams (C T x : ℕ) (hC : C = 6210) (hT : T = 3) (hx: x > 0):
  T * (x - 1) = C / x :=
by
  rw [hC, hT]
  sorry

end buy_beams_l367_367530


namespace trigonometric_relationship_l367_367106

noncomputable def α : ℝ := Real.cos 4
noncomputable def b : ℝ := Real.cos (4 * Real.pi / 5)
noncomputable def c : ℝ := Real.sin (7 * Real.pi / 6)

theorem trigonometric_relationship : b < α ∧ α < c := 
by
  sorry

end trigonometric_relationship_l367_367106


namespace cube_fraction_inequality_l367_367785

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l367_367785


namespace tangent_line_at_point_l367_367997

noncomputable theory

open Real

def f (x : ℝ) : ℝ := x * (3 * log x + 1)

def f_deriv (x : ℝ) : ℝ := deriv f x

theorem tangent_line_at_point :
  f 1 = 1 ∧ f_deriv 1 = 4 → ∀ x : ℝ, (1 : ℝ) - 1 = 4 * (x - 1) → (4 : ℝ) * x - 3 = x * (3 * log x + 1) :=
by
  sorry

end tangent_line_at_point_l367_367997


namespace least_number_of_control_weights_is_4_l367_367179

/-- Given a set of control weights where each control weight is a non-integer number of grams.
    Any integer weight from 1 g to 40 g can be balanced by some of these weights when placed
    on one pan of the balance and the measured weight on the other pan. 

    Prove that the least number of control weights required is 4. -/
theorem least_number_of_control_weights_is_4 (weights : List ℕ) :
  (∀ (w : ℕ), 1 ≤ w ∧ w ≤ 40 → 
    ∃ (l r : List ℕ), (l.Sum - r.Sum) = w ∧ l.Length + r.Length ≤ 4) → (weights.length ≥ 4) :=
by
  sorry

end least_number_of_control_weights_is_4_l367_367179


namespace unspent_portion_proof_l367_367129

variable (G : ℝ)
variable (gold_limit : G = G)
variable (platinum_limit : 2 * G = 2 * G)
variable (gold_balance : ℝ := (1 / 3) * G)
variable (platinum_balance : ℝ := (1 / 2) * G)
variable (new_platinum_balance : ℝ := platinum_balance + gold_balance)
variable (unspent_portion : ℝ := (2 * G) - new_platinum_balance)

theorem unspent_portion_proof (G : ℝ) :
  G > 0 →
  gold_limit = G →
  platinum_limit = 2 * G →
  gold_balance = (1 / 3) * G →
  platinum_balance = (1 / 2) * G →
  new_platinum_balance = (5 / 6) * G →
  unspent_portion = (7 / 6) * G :=
by
  intros
  sorry

end unspent_portion_proof_l367_367129


namespace taxi_ride_cost_l367_367654

def baseFare : ℝ := 1.50
def costPerMile : ℝ := 0.25
def milesTraveled : ℕ := 5
def totalCost := baseFare + (costPerMile * milesTraveled)

/-- The cost of a 5-mile taxi ride is $2.75. -/
theorem taxi_ride_cost : totalCost = 2.75 := by
  sorry

end taxi_ride_cost_l367_367654


namespace corresponding_angles_not_always_equal_l367_367166

theorem corresponding_angles_not_always_equal :
  (∀ α β c : ℝ, (α = β ∧ ¬c = 0) → (∃ x1 x2 y : ℝ, α = x1 ∧ β = x2 ∧ x1 = y * c ∧ x2 = y * c)) → False :=
by
  sorry

end corresponding_angles_not_always_equal_l367_367166


namespace inscribed_square_proof_l367_367243

theorem inscribed_square_proof :
  (∃ (r : ℝ), 2 * π * r = 72 * π ∧ r = 36) ∧ 
  (∃ (s : ℝ), (2 * (36:ℝ))^2 = 2 * s ^ 2 ∧ s = 36 * Real.sqrt 2) :=
by
  sorry

end inscribed_square_proof_l367_367243


namespace set_difference_lt3_gt0_1_leq_x_leq_2_l367_367399

def A := {x : ℝ | |x| < 3}
def B := {x : ℝ | x^2 - 3 * x + 2 > 0}

theorem set_difference_lt3_gt0_1_leq_x_leq_2 : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end set_difference_lt3_gt0_1_leq_x_leq_2_l367_367399


namespace length_of_chord_l367_367494

theorem length_of_chord (k : ℝ) :
  let center := (1, 1) in
  let radius := 2 in
  let line := (k + 1) * 1 - k * 1 - 1 = 0 in
  if line then 2 * radius = 4 else false :=
begin
  sorry
end

end length_of_chord_l367_367494


namespace cost_of_each_trophy_is_correct_l367_367117

noncomputable def is_divisible_by_8 (n : ℕ) : Prop :=
  (n % 8) = 0

noncomputable def is_divisible_by_9 (n : ℕ) : Prop :=
  (n % 9) = 0

theorem cost_of_each_trophy_is_correct :
  ∃ a b : ℕ, 
  let total_cost := 1000 * a + 990 + b in
  let individual_cost := total_cost / 72 in
  is_divisible_by_8 (990 + b) ∧ 
  is_divisible_by_9 (a + 9 + 9 + 9 + b) ∧ 
  individual_cost = 11.11 := 
sorry

end cost_of_each_trophy_is_correct_l367_367117


namespace mean_minus_median_l367_367741

theorem mean_minus_median :
  let n0 := 3 in     -- number of students missing 0 days
  let n1 := 2 in     -- number of students missing 1 day
  let n2 := 4 in     -- number of students missing 2 days
  let n3 := 2 in     -- number of students missing 3 days
  let n4 := 1 in     -- number of students missing 4 days
  let n5 := 5 in     -- number of students missing 5 days
  let n6 := 1 in     -- number of students missing 6 days
  let total_students := 18 in
  let median := (2 + 2) / 2 in
  let mean := (n0 * 0 + n1 * 1 + n2 * 2 + n3 * 3 + n4 * 4 + n5 * 5 + n6 * 6) / total_students in
  mean - median = 5 / 6 :=
by {
  -- placeholder for proof steps
  sorry
}

end mean_minus_median_l367_367741


namespace eval_expr_l367_367702

theorem eval_expr (a b : ℕ) (ha : a = 3) (hb : b = 4) : (a^b)^a - (b^a)^b = -16245775 := by
  sorry

end eval_expr_l367_367702


namespace chocolate_oranges_initial_l367_367113

theorem chocolate_oranges_initial (p_c p_o G n_c x : ℕ) 
  (h_candy_bar_price : p_c = 5) 
  (h_orange_price : p_o = 10) 
  (h_goal : G = 1000) 
  (h_candy_bars_sold : n_c = 160) 
  (h_equation : G = p_o * x + p_c * n_c) : 
  x = 20 := 
by
  sorry

end chocolate_oranges_initial_l367_367113


namespace slices_in_loaf_initial_l367_367187

-- Define the total slices used from Monday to Friday
def slices_used_weekdays : Nat := 5 * 2

-- Define the total slices used on Saturday
def slices_used_saturday : Nat := 2 * 2

-- Define the total slices used in the week
def total_slices_used : Nat := slices_used_weekdays + slices_used_saturday

-- Define the slices left
def slices_left : Nat := 6

-- Prove the total slices Tony started with
theorem slices_in_loaf_initial :
  let slices := total_slices_used + slices_left
  slices = 20 :=
by
  sorry

end slices_in_loaf_initial_l367_367187


namespace tangent_line_at_point_l367_367995

noncomputable theory

open Real

def f (x : ℝ) : ℝ := x * (3 * log x + 1)

def f_deriv (x : ℝ) : ℝ := deriv f x

theorem tangent_line_at_point :
  f 1 = 1 ∧ f_deriv 1 = 4 → ∀ x : ℝ, (1 : ℝ) - 1 = 4 * (x - 1) → (4 : ℝ) * x - 3 = x * (3 * log x + 1) :=
by
  sorry

end tangent_line_at_point_l367_367995


namespace probability_A_wins_championship_expectation_X_is_13_l367_367966

/-
Definitions corresponding to the conditions in the problem
-/
def prob_event1_A_win : ℝ := 0.5
def prob_event2_A_win : ℝ := 0.4
def prob_event3_A_win : ℝ := 0.8

def prob_event1_B_win : ℝ := 1 - prob_event1_A_win
def prob_event2_B_win : ℝ := 1 - prob_event2_A_win
def prob_event3_B_win : ℝ := 1 - prob_event3_A_win

/-
Proof problems corresponding to the questions and correct answers
-/

theorem probability_A_wins_championship : prob_event1_A_win * prob_event2_A_win * prob_event3_A_win
    + prob_event1_A_win * prob_event2_A_win * prob_event3_B_win
    + prob_event1_A_win * prob_event2_B_win * prob_event3_A_win 
    + prob_event1_B_win * prob_event2_A_win * prob_event3_A_win = 0.6 := 
sorry

noncomputable def X_distribution_table : list (ℝ × ℝ) := 
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

noncomputable def expected_value_X : ℝ := 
  ∑ x in X_distribution_table, x.1 * x.2

theorem expectation_X_is_13 : expected_value_X = 13 := sorry

end probability_A_wins_championship_expectation_X_is_13_l367_367966


namespace solution_exists_l367_367712

def f (x : ℚ⁺) : ℚ⁺ := sorry
axiom f_property : ∀ x y : ℚ⁺, f (x * f y) = f x / y

noncomputable def prime1 (n : ℕ) : ℚ⁺ := sorry -- Define the n-th prime in the first set
noncomputable def prime2 (n : ℕ) : ℚ⁺ := sorry -- Define the n-th prime in the second set

axiom f_values : ∀ n : ℕ, f (prime1 n) = prime2 n ∧ f (prime2 n) = 1 / prime1 n

theorem solution_exists : ∃ f : ℚ⁺ → ℚ⁺, (∀ x y : ℚ⁺, f (x * f y) = f x / y) ∧
  (∀ n : ℕ, f (prime1 n) = prime2 n) ∧ (∀ n : ℕ, f (prime2 n) = 1 / prime1 n) := sorry

end solution_exists_l367_367712


namespace count_15000_safe_numbers_l367_367353

def is_psafe (p n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 0 → ¬ (abs (n - k * p) ≤ 3)

def count_safe_numbers (p8 p12 p15 upper_bound: ℕ) : ℕ :=
  { n : ℕ | n ≤ upper_bound ∧ is_psafe p8 n ∧ is_psafe p12 n ∧ is_psafe p15 n}.card

theorem count_15000_safe_numbers :
  count_safe_numbers 8 12 15 15000 = 2173 :=
sorry

end count_15000_safe_numbers_l367_367353


namespace repeating_decimal_as_fraction_l367_367708

theorem repeating_decimal_as_fraction :
  let x := 0.36 + 0.0036 + 0.000036 + 0.00000036 + ∑' (n : ℕ), (0.0036 : ℝ) * ((1 / 100) ^ n)
  in x = (4 : ℝ) / (11 : ℝ) := 
by {
  let x : ℝ := ∑' (n : ℕ), (36 : ℝ) / ((10^2 : ℝ) * (10 ^ (2 * n))),
  have hx : x = (4 : ℝ) / (11 : ℝ), 
  sorry,
}

end repeating_decimal_as_fraction_l367_367708


namespace sum_of_side_lengths_l367_367168

theorem sum_of_side_lengths (p q r : ℕ) (h : p = 8 ∧ q = 1 ∧ r = 5) 
    (area_ratio : 128 / 50 = 64 / 25) 
    (side_length_ratio : 8 / 5 = Real.sqrt (128 / 50)) :
    p + q + r = 14 := 
by 
  sorry

end sum_of_side_lengths_l367_367168


namespace account_initial_amount_l367_367281

variable {t : ℕ} {a : ℕ}

def initialAmount (t a : ℕ) : ℕ := a + t

theorem account_initial_amount : initialAmount 69 26935 = 27004 :=
by
  rw [initialAmount]
  sorry

end account_initial_amount_l367_367281


namespace area_of_enclosed_shape_l367_367719

noncomputable def area_enclosed_by_line_and_parabola : ℝ :=
  (∫ (x : ℝ) in -1..3, (2*x + 3)) - (∫ (x : ℝ) in -1..3, x^2)

theorem area_of_enclosed_shape :
  area_enclosed_by_line_and_parabola = 32 / 3 :=
by
  -- We need to prove the statement here
  sorry

end area_of_enclosed_shape_l367_367719


namespace prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l367_367244

-- Definitions for the conditions.

-- cube configuration
def num_dice : ℕ := 27
def num_visible_dice : ℕ := 26
def num_faces_per_die : ℕ := 6
def num_visible_faces : ℕ := 54

-- Given probabilities
def prob_six (face : ℕ) : ℚ := 1/6
def prob_not_six (face : ℕ) : ℚ := 5/6
def prob_not_one (face : ℕ) : ℚ := 5/6

-- Expected values given conditions
def expected_num_sixes : ℚ := 9
def expected_sum_faces : ℚ := 189
def expected_diff_digits : ℚ := 6 - (5^6) / (2 * 3^17)

-- Probabilities given conditions
def prob_25_sixes_on_surface : ℚ := (26 * 5) / (6^26)
def prob_at_least_one_one : ℚ := 1 - (5^6) / (2^2 * 3^18)

-- Lean statements for proof

theorem prob_of_25_sixes_on_surface :
  prob_25_sixes_on_surface = 31 / (2^13 * 3^18) := by
  sorry

theorem prob_of_at_least_one_one_on_surface :
  prob_at_least_one_one = 0.99998992 := by
  sorry

theorem expected_number_of_sixes_on_surface :
  expected_num_sixes = 9 := by
  sorry

theorem expected_sum_of_numbers_on_surface :
  expected_sum_faces = 189 := by
  sorry

theorem expected_value_of_diff_digits_on_surface :
  expected_diff_digits = 6 - (5^6) / (2 * 3^17) := by
  sorry

end prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l367_367244


namespace sum_of_numerator_and_denominator_of_cos_gamma_is_32_l367_367051

theorem sum_of_numerator_and_denominator_of_cos_gamma_is_32 
  (l1 l2 l3 : ℝ) (γ δ : ℝ) (r : ℝ)
  (h_l1 : l1 = 5)
  (h_l2 : l2 = 12)
  (h_l3 : l3 = 13)
  (h_angle1 : l1^2 = 2 * r^2 * (1 - cos γ))
  (h_angle2 : l2^2 = 2 * r^2 * (1 - cos δ))
  (h_angle3 : l3^2 = 2 * r^2 * (1 - cos (γ + δ)))
  (h_angle_sum_lt_pi : γ + δ < π)
  (h_cos_gamma_pos_rat : ∃ p q : ℕ, p / q = cos γ) :
  let ⟨p, q, h_cos_fraction⟩ := h_cos_gamma_pos_rat in 
  let sum := p + q in
  sum = 32 :=
sorry

end sum_of_numerator_and_denominator_of_cos_gamma_is_32_l367_367051


namespace distance_from_point_to_line_proof_l367_367339

open Real EuclideanSpace

-- Definitions for points and line
def p : EuclideanSpace ℝ (Fin 3) := ![0, 3, -1]
def a : EuclideanSpace ℝ (Fin 3) := ![1, -2, 0]
def b : EuclideanSpace ℝ (Fin 3) := ![3, 1, 4]

-- Definition for the direction vector
def d := b - a

-- Function to parameterize the line
def line (t : ℝ) : EuclideanSpace ℝ (Fin 3) := a + t • d

-- Function for vector from point p to point on the line
def v_minus_p (t : ℝ) : EuclideanSpace ℝ (Fin 3) := line t - p

-- Distance function
def distance_from_point_to_line : ℝ := 
  let t := (9 : ℝ) / 29 in
  dist (line t) p

theorem distance_from_point_to_line_proof:
  distance_from_point_to_line = sqrt 22058 / 29 := sorry

end distance_from_point_to_line_proof_l367_367339


namespace sum_adjacent_6_is_29_l367_367840

-- Define the grid and the placement of numbers 1 to 4
structure Grid :=
  (grid : Fin 3 → Fin 3 → Nat)
  (h_unique : ∀ i j, grid i j ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_distinct : Function.Injective (λ (i : Fin 3) (j : Fin 3), grid i j))
  (h_placement : grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧ grid 2 2 = 4)

-- Define the condition of the sum of numbers adjacent to 5 being 9
def sum_adjacent_5 (g : Grid) : Prop :=
  let (i, j) := (0, 1) in -- Position for number 5
  (g.grid (i.succ) j + g.grid (i.succ.pred) j + g.grid i (j.succ) + g.grid i (j.pred)) = 9

-- Define the main theorem
theorem sum_adjacent_6_is_29 (g : Grid) (h_sum_adj_5 : sum_adjacent_5 g) : 
  (g.grid 1 0 + g.grid 1 2 + g.grid 0 1 + g.grid 2 1 = 29) := sorry

end sum_adjacent_6_is_29_l367_367840


namespace range_inclination_angle_l367_367167

theorem range_inclination_angle (θ : ℝ) :
  let α := Real.arctan (x * Real.cos θ + 4) in
  (cos θ ∈ [-1, 1]) →
  (0 ≤ α ∧ α ≤ π/4) ∨ (3 * π / 4 ≤ α ∧ α < π) :=
sorry

end range_inclination_angle_l367_367167


namespace diagonal_passes_through_intersection_l367_367194

variables {A B C D M N P Q K O : Type*}
variables [parallelogram A B C D] [parallelogram M N P Q]
variables (intersection_O_ABCD : is_intersection_of_diagonals O A C B D)
variables (KQ_is_diagonal_MNPQ : is_diagonal K Q M N P Q)

theorem diagonal_passes_through_intersection :
  passes_through (diagonal KQ_is_diagonal_MNPQ) O :=
sorry

end diagonal_passes_through_intersection_l367_367194


namespace water_required_l367_367718

variable (BaO H2O BaOH₂ : Type)

-- Definitions for the reactants and products
def barium_oxide : BaO := sorry
def water : H2O := sorry
def barium_hydroxide : BaOH₂ := sorry

-- The balanced equation
def reacts_to : BaO × H2O → BaOH₂ := sorry

-- Conditions
def balanced_equation (x : BaO) (y : H2O) : reacts_to (x, y) = barium_hydroxide := sorry

theorem water_required (barium_oxide_moles : ℕ) (water_moles : ℕ) :
  (∀ (x : BaO) (y : H2O), balanced_equation x y) →
  barium_oxide_moles = 3 → water_moles = 3 :=
by
  intros Eq_Balanced BaO_Moles_Condition
  simp
  sorry

end water_required_l367_367718


namespace second_discount_is_20_percent_l367_367272

-- Definitions based on conditions
def normal_price : ℝ := 174.99999999999997
def first_discount_percentage : ℝ := 0.10
def final_price : ℝ := 126.0

-- Definition of the price after the first discount
def price_after_first_discount : ℝ := normal_price * (1 - first_discount_percentage)

-- Definition of the second discount
def second_discount_percentage : ℝ := (price_after_first_discount - final_price) / price_after_first_discount * 100

-- Statement to prove
theorem second_discount_is_20_percent : second_discount_percentage = 20 :=
by sorry

end second_discount_is_20_percent_l367_367272


namespace inequality_proof_l367_367783

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l367_367783


namespace range_of_a_l367_367422

def f (a: ℝ) (x: ℝ): ℝ :=
  if x ≤ 1 then (a + 3) * x - 5 else (2 * a) / x

theorem range_of_a (a: ℝ):
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ -2 ≤ a ∧ a < 0 :=
by
  sorry

end range_of_a_l367_367422


namespace solve_equation_l367_367525

theorem solve_equation : ∀ x : ℝ, ((x - 1) / 3 = 2 * x) → (x = -1 / 5) :=
by
  assume x h,
  sorry

end solve_equation_l367_367525


namespace problem_trajectory_l367_367108

section Trajectory

variable (l1 l2 l3 : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (N P : ℝ × ℝ)
variable (l : ℝ → ℝ)
variable (Q : ℝ × ℝ := (0, 2))

theorem problem_trajectory (h_l1 : l1.1^2 + l1.2^2 = 9)
                          (h_l2 : l2.1^2 + l2.2^2 = 9)
                          (h_l3 : l3.1^2 + l3.2^2 = 9)
                          (h_M : M.1^2 + M.2^2 = 9)
                          (h_perpendicular : N.1 = M.1 ∧ N.2 = 0)
                          (h_ratio : P.1 = N.1 ∧ P.2 = (2 * N.2 + M.2) / 3):
  (∃ C : ℝ → ℝ → Prop, ∀ x y, C x y ↔ (x^2 / 9 + y^2 / 4 = 1)) ∧
  (∃ F : ℝ × ℝ, F = (-4 / 3, -2) ∧
   (forall A B : ℝ × ℝ, l A.1 = A.2 → l B.1 = B.2 →
         (C A.1 A.2 ∧ C B.1 B.2) →
         (Q.2 - A.2) / (Q.1 - A.1) + (Q.2 - B.2) / (Q.1 - B.1) = 3 →
          -- Line l passing through the fixed point (Q)
         ((∃ k b, l = λ x, k * x + b ∧ b = -(4 * k) - 2) ∨ (A.1 = -4 / 3 ∧ B.1 = -4 / 3)))):
sorry

end Trajectory

end problem_trajectory_l367_367108


namespace log_sum_seven_l367_367876

-- Define the geometric sequence condition
variable {b : ℕ → ℝ}

-- Define that the sequence is geometric where each term is positive and \(b_7 \cdot b_8 = 3\)
axiom geometric_sequence (r : ℝ) (h_pos : ∀ n, b n > 0) (h_geom : ∀ n, b (n + 1) = r * b n):
  b 7 * b 8 = 3

-- The proof statement we want to prove
theorem log_sum_seven (r : ℝ) (h_pos : ∀ n, b n > 0) (h_geom : ∀ n, b (n + 1) = r * b n)
  (h_prod : b 7 * b 8 = 3) :
  (∑ n in finset.range 14, real.logb 3 (b (n + 1))) = 7 := 
sorry

end log_sum_seven_l367_367876


namespace evaluate_expression_l367_367699

theorem evaluate_expression : 
  let a := 3
  let b := 4
  (a^b)^a - (b^a)^b = -16245775 := 
by 
  sorry

end evaluate_expression_l367_367699


namespace find_brick_width_l367_367635

noncomputable def width_of_brick (V_wall : ℝ) (n_bricks : ℝ) : ℝ :=
  let V_brick_volume := 80 * W * 6
  in V_wall / (n_bricks * V_brick_volume)

theorem find_brick_width
  (V_wall : ℝ := 800 * 600 * 22.5)
  (n_bricks : ℝ := 2000) :
  width_of_brick V_wall n_bricks = 5.625 := sorry

end find_brick_width_l367_367635


namespace min_value_abc_l367_367947

theorem min_value_abc : 
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a^b % 10 = 4) ∧ (b^c % 10 = 2) ∧ (c^a % 10 = 9) ∧ 
    (a + b + c = 17) :=
  by {
    sorry
  }

end min_value_abc_l367_367947


namespace det_proj_matrix_is_zero_l367_367485

open Matrix

-- Define v as a 2x1 matrix
def v : Matrix (Fin 2) (Fin 1) ℚ := ![![3], ![2]]

-- Define the projection matrix Q
def Q : Matrix (Fin 2) (Fin 2) ℚ := 
  let vt := transpose v
  let vt_v := vt.mul v
  let v_vt := v.mul vt
  (1 / vt_v.getElem 0 0) • v_vt

-- Prove the determinant of Q is 0
theorem det_proj_matrix_is_zero : det Q = 0 :=
by
  sorry

end det_proj_matrix_is_zero_l367_367485


namespace quadrilateral_ABCD_pq_sum_l367_367067

noncomputable def AB_pq_sum : ℕ :=
  let p : ℕ := 9
  let q : ℕ := 141
  p + q

theorem quadrilateral_ABCD_pq_sum (BC CD AD : ℕ) (m_angle_A m_angle_B : ℕ) (hBC : BC = 8) (hCD : CD = 12) (hAD : AD = 10) (hAngleA : m_angle_A = 60) (hAngleB : m_angle_B = 60) : AB_pq_sum = 150 := by sorry

end quadrilateral_ABCD_pq_sum_l367_367067


namespace coefficient_x3_l367_367336

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3 (n k : ℕ) (x : ℤ) :
  let expTerm : ℤ := 1 - x + (1 / x^2017)
  let expansion := fun (k : ℕ) => binomial n k • ((1 - x)^(n - k) * (1 / x^2017)^k)
  (n = 9) → (k = 3) →
  (expansion k) = -84 :=
  by
    intros
    sorry

end coefficient_x3_l367_367336


namespace complex_number_quadrant_l367_367554

-- Definitions and conditions used in the proof problem
def z : ℂ := 4 / (1 + Complex.i)

-- A theorem that states the location of the complex number in the fourth quadrant
theorem complex_number_quadrant : z.re > 0 ∧ z.im < 0 :=
by
  -- This is where the proof steps would go if we were to solve it
  sorry

end complex_number_quadrant_l367_367554


namespace limit_problem_l367_367229

-- Define the function f(x)
def f (x : ℝ) : ℝ := (5 * x^2 - 51 * x + 10) / (x - 10)

-- Define the limit problem
theorem limit_problem (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - 10| ∧ |x - 10| < δ → |f x - 49| < ε :=
begin
  use ε / 5,
  split,
  { linarith, },
  { intros x hx,
    have : |f(x) - 49| = |5 * (x - 10)|, sorry,
    rw this,
    apply lt_of_mul_lt_mul_right _ (abs_nonneg _),
    rw abs_of_nonneg, { linarith, }, 
    exact le_of_lt (hx.2),
  }
end

end limit_problem_l367_367229


namespace min_sum_x_y_l367_367749

noncomputable def lg (x : ℝ) := Real.log10 x

theorem min_sum_x_y (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h_arith_seq : 2 * 2 = lg x + lg y) : x + y ≥ 200 := 
by {
  -- Proof skipped
  sorry
}

example (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h_arith_seq : 2 * 2 = lg x + lg y) : x + y = 200 ↔ x = 100 ∧ y = 100 := 
by {
  -- Proof skipped
  sorry
}


end min_sum_x_y_l367_367749


namespace rotation_result_l367_367562

-- Definition for the initial vector
def initial_vector : Matrix (Fin 3) (Fin 1) ℚ := ![![2], ![1], ![3]]

-- Definition for 90 degree rotation matrix about the x-axis
def rotation_x_90 : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, 0, 0], 
    ![0, 0, -1], 
    ![0, 1, 0]]

-- Definition for theta rotation matrix about the z-axis
def rotation_z_theta (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![Real.cos θ, -Real.sin θ, 0], 
    ![Real.sin θ, Real.cos θ, 0], 
    ![0, 0, 1]]

-- Final resulting vector after both rotations
noncomputable def resulting_vector (θ : ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  rotation_z_theta θ ⬝ (rotation_x_90 ⬝ initial_vector)

-- Theorem statement
theorem rotation_result (θ : ℝ) :
  resulting_vector θ = 
  ![![2 * Real.cos θ + 3 * Real.sin θ], 
    ![-2 * Real.sin θ + 3 * Real.cos θ], 
    ![1]] := sorry

end rotation_result_l367_367562


namespace probability_A_wins_championship_distribution_and_expectation_B_l367_367962

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l367_367962


namespace centroid_divides_in_ratio_l367_367949

-- Define the cyclic quadrilateral with vertices A, B, C, D
structure CyclicQuadrilateral (P Q R S : Type) :=
(circumcenter : Point)
(intersection_of_diagonals : Point)
(is_cyclic : ∃ O : Point, ∃ r : ℝ, ∀ (p : Point), p ∈ {P, Q, R, S} → dist p O = r)
(perpendicular_diagonals : ∃ H : Point, is_perpendicular (P ↔ R) (Q ↔ S))

-- Define the condition that the centroid divides OH in the ratio 1:2
def divides_in_ratio (G O H : Point) (num den : ℕ) :=
∃ m : ℝ, dist O G = m * num ∧ dist G H = m * den

-- Prove the translated problem
theorem centroid_divides_in_ratio 
  {A B C D O H G : Point} 
  (quad: CyclicQuadrilateral A B C D)
  (centroid_condition: centroid A B C D = G) :
  divides_in_ratio G O H 1 2 :=
sorry

end centroid_divides_in_ratio_l367_367949


namespace sum_of_adjacent_to_6_l367_367865

theorem sum_of_adjacent_to_6 :
  ∃ (grid : Fin 3 × Fin 3 → ℕ),
  (grid (0, 0) = 1 ∧ grid (0, 2) = 3 ∧ grid (2, 0) = 2 ∧ grid (2, 2) = 4 ∧
   ∀ i j, grid (i, j) ∈ finset.range 1 10 ∧ finset.univ.card = 9 ∧
   (grid (1, 0) + grid (1, 1) + grid (2, 1) = 9) ∧ 
   (grid (1, 1) = 6) ∧ 
   (sum_of_adjacent grid (1, 1) = 29))

where
  sum_of_adjacent (grid : Fin 3 × Fin 3 → ℕ) (x y : Fin 3 × Fin 3) : ℕ :=
  grid (x - 1, y) + grid (x + 1, y) + grid (x, y - 1) + grid (x, y + 1)
  sorry

end sum_of_adjacent_to_6_l367_367865


namespace average_exercise_days_is_4_36_l367_367877

noncomputable def average_exercise_days : ℕ :=
  let frequency := [(1, 1), (2, 3), (3, 2), (4, 6), (5, 8), (6, 3), (7, 2)]
  let total_exercise_days := frequency.foldl (λ acc val => acc + (val.1 * val.2)) 0
  let total_students := frequency.foldl (λ acc val => acc + val.2) 0
  total_exercise_days / total_students

theorem average_exercise_days_is_4_36 :
  average_exercise_days = 4.36 := by
  sorry

end average_exercise_days_is_4_36_l367_367877


namespace calculate_expression_l367_367668

theorem calculate_expression (m n : ℝ) : 9 * m^2 - (m - 2 * n)^2 = 4 * (2 * m - n) * (m + n) :=
by
  sorry

end calculate_expression_l367_367668


namespace minimize_potato_cost_l367_367706

def potatoes_distribution (x1 x2 x3 : ℚ) : Prop :=
  x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧
  x1 + x2 + x3 = 12 ∧
  x1 + 4 * x2 + 3 * x3 ≤ 40 ∧
  x1 ≤ 10 ∧ x2 ≤ 8 ∧ x3 ≤ 6 ∧
  4 * x1 + 3 * x2 + 1 * x3 = (74 / 3)

theorem minimize_potato_cost :
  ∃ x1 x2 x3 : ℚ, potatoes_distribution x1 x2 x3 ∧ x1 = (2/3) ∧ x2 = (16/3) ∧ x3 = 6 :=
by
  sorry

end minimize_potato_cost_l367_367706


namespace distance_after_15_minutes_l367_367539

def initial_distance : ℝ := 2.5
def hyosung_speed : ℝ := 0.08
def mimi_speed_hour : ℝ := 2.4
def mimi_speed_minute : ℝ := mimi_speed_hour / 60
def total_time_minutes : ℝ := 15
def total_distance_together : ℝ := (hyosung_speed + mimi_speed_minute) * total_time_minutes
def remaining_distance : ℝ := initial_distance - total_distance_together

theorem distance_after_15_minutes :
  remaining_distance = 0.7 :=
by
  /- 
  We are given:
  initial_distance : ℝ = 2.5
  hyosung_speed : ℝ = 0.08
  mimi_speed_hour : ℝ = 2.4
  To convert mimi_speed_hour to km per minute, we divide by 60:
  mimi_speed_minute = mimi_speed_hour / 60 = 2.4 / 60 = 0.04
  Together, their speed is hyosung_speed + mimi_speed_minute = 0.08 + 0.04 = 0.12 km/min.
  In 15 minutes they cover distance: 0.12 * 15 = 1.8 km.
  So, the remaining distance = initial_distance - 1.8 = 2.5 - 1.8 = 0.7 km.
  -/
  sorry

end distance_after_15_minutes_l367_367539


namespace solution_set_xf_gt_0_l367_367406

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom f_neg_two : f (-2) = 0
axiom positive_condition (x : ℝ) (hx : x > 0) : (x * (deriv f x) - f x) / (x^2) > 0

theorem solution_set_xf_gt_0 :
  {x : ℝ | x * f x > 0} = set.Ioo (-∞) (-2) ∪ set.Ioo 2 ∞ :=
sorry

end solution_set_xf_gt_0_l367_367406


namespace least_three_digit_multiple_of_3_4_9_is_108_l367_367587

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l367_367587


namespace find_five_digit_square_pairs_l367_367564

theorem find_five_digit_square_pairs :
  ∃ a b : ℕ, 10000 ≤ a ∧ a ≤ 99999 ∧
            10000 ≤ b ∧ b ≤ 99999 ∧
            (∀ i, a.digits 10 i + 1 = b.digits 10 i) ∧
            a = 115^2 ∧ b = 156^2 :=
by
  sorry

end find_five_digit_square_pairs_l367_367564


namespace find_x_l367_367830

theorem find_x : ∃ x : ℝ, (∀ a b : ℝ, a @ b = a^b / 2) ∧ (3 @ x = 4.5) -> x = 2 :=
  sorry

end find_x_l367_367830


namespace bisect_broken_line_l367_367052

open Real EuclideanGeometry

theorem bisect_broken_line
  {A B C M H : Point}
  (h1 : AB > AC)
  (h2 : M = midpoint_arc A B C)
  (h3 : perpendicular_from_midpoint M AB H) :
  bisects_broken_line H B A C :=
sorry

end bisect_broken_line_l367_367052


namespace baseball_team_total_runs_l367_367632

noncomputable def total_runs_by_opponents (team_scores : List ℕ) (opponents_scores : List ℕ) : ℕ :=
  List.sum opponents_scores

theorem baseball_team_total_runs :
  let team_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let lost_games := [1, 2, 3, 4, 5, 6, 7, 8] in
  let won_games := [9, 10, 11, 12, 13, 14, 15] in
  let opponents_scores_lost := List.map (λ x, x + 2) lost_games in
  let opponents_scores_won := List.map (λ x, x / 3) won_games in
  total_runs_by_opponents team_scores (opponents_scores_lost ++ opponents_scores_won) = 78 :=
by
  sorry

end baseball_team_total_runs_l367_367632


namespace complex_problem_l367_367410

noncomputable def z (ζ : ℂ) : ℂ := ζ

theorem complex_problem (ζ : ℂ) (h : ζ + ζ⁻¹ = 2 * real.cos (real.pi / 36)) : 
  (ζ ^ 100 + ζ ^ (-100)) = -2 * real.cos (2 * real.pi / 9) :=
by sorry

end complex_problem_l367_367410


namespace inverse_proportion_relationship_l367_367378

theorem inverse_proportion_relationship (k : ℝ) (y1 y2 y3 : ℝ) :
  y1 = (k^2 + 1) / -1 →
  y2 = (k^2 + 1) / 1 →
  y3 = (k^2 + 1) / 2 →
  y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end inverse_proportion_relationship_l367_367378


namespace two_digit_palindromes_l367_367818

theorem two_digit_palindromes:
  let is_odd := (λ n : ℕ, n % 2 = 1)
  let two_digit := finset.range 90 \ finset.range 10
  let reversed (n : ℕ) := (n % 10) * 10 + (n / 10)
  let is_palindrome (n : ℕ) := let s := n.digits 10 in s = s.reverse
  (finset.filter (λ n, 
    is_odd (n / 10) ∧ is_odd (n % 10) ∧ is_palindrome (n + reversed n)) 
    two_digit).card = 6 :=
by sorry

end two_digit_palindromes_l367_367818


namespace number_of_ways_to_place_pawns_l367_367029

theorem number_of_ways_to_place_pawns :
  let n := 5 in
  let number_of_placements := (n.factorial) in
  let number_of_permutations := (n.factorial) in
  number_of_placements * number_of_permutations = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l367_367029


namespace largest_divisor_of_Q_l367_367273

theorem largest_divisor_of_Q (Q : ℕ) (hQ : ∃ n ∈ {1, 2, 3, 4, 5, 6, 7, 8}, Q = 8! / n) : ∃ d : ℕ, d = 48 ∧ ∀ m : ℕ, (m ≤ 48 → m ∣ Q) :=
begin
  existsi (48 : ℕ),
  split,
  { refl },  -- d = 48

  sorry  -- Proof that any m dividing Q and less than or equal to 48 divides 48
end

end largest_divisor_of_Q_l367_367273


namespace shortest_part_length_l367_367232

theorem shortest_part_length (total_length : ℝ) (r1 r2 r3 : ℝ) (shortest_length : ℝ) :
  total_length = 196.85 → r1 = 3.6 → r2 = 8.4 → r3 = 12 → shortest_length = 29.5275 :=
by
  sorry

end shortest_part_length_l367_367232


namespace red_blue_arcs_equal_sum_l367_367388

theorem red_blue_arcs_equal_sum (n : ℕ) (A : Fin n → Point) (O O' : Point) (r r' : ℝ) :
  InscribedPolygon A O r →
  CircumscribedMidpoints A O' r' →
  (∃ red blue : Set (Arc O'),
    (is_partition red blue) ∧
    (∑ arc in red, arc.length = ∑ arc in blue, arc.length)) := sorry

end red_blue_arcs_equal_sum_l367_367388


namespace distinct_pawn_placements_on_chess_board_l367_367038

def numPawnPlacements : ℕ :=
  5! * 5!

theorem distinct_pawn_placements_on_chess_board :
  numPawnPlacements = 14400 := by
  sorry

end distinct_pawn_placements_on_chess_board_l367_367038


namespace max_guaranteed_score_l367_367633

-- Definitions based on the conditions
def blackboard := list (ℤ × ℤ)

noncomputable def satisfying_conditions (pairs : blackboard) : Prop :=
  ∃ (f : ℤ → bool), 
    (∀ k : ℤ, ((k, k) ∈ pairs → (f k ∨ f (-k))) ∧ ((-k, -k) ∈ pairs → (f k ∨ f (-k))))
    ∧ (∀ x : ℤ, f x → ¬ f (-x))
    ∧ list.length pairs = 68

-- Definition of the score based on the problem conditions
noncomputable def score (pairs : blackboard) (erased : ℤ → bool) : ℕ :=
  list.length (list.filter (λ pair, erased pair.fst ∨ erased pair.snd) pairs)

-- Main theorem statement
theorem max_guaranteed_score (pairs : blackboard) (h : satisfying_conditions pairs) : 
  ∃ erased : ℤ → bool, score pairs erased = 43 :=
sorry

end max_guaranteed_score_l367_367633


namespace min_w_value_l367_367685

open Real

noncomputable def w (x y : ℝ) : ℝ := 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27

theorem min_w_value : ∃ x y : ℝ, w x y = 81 / 4 :=
by
  use [-3/2, 1]
  dsimp [w]
  norm_num
  done

end min_w_value_l367_367685


namespace least_three_digit_multiple_of_3_4_9_is_108_l367_367586

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l367_367586


namespace find_value_of_a_l367_367501

variable (U : Set ℕ) (A : ℕ → Set ℕ)
variable (a : ℕ)
variable (complement_U : Set ℕ → Set ℕ)
variable (a_value : ℕ)

-- Definitions
def universal_set := U = {1, 3, 5, 7, 9}
def set_A := A a = {1, |a - 5|, 9}
def complement_of_A := complement_U (A a) = {5, 7}

-- Theorem statement
theorem find_value_of_a (hU : universal_set U) (hA : set_A A a) (hCA : complement_of_A complement_U A a) :
  a_value = 2 ∨ a_value = 8 := 
sorry

end find_value_of_a_l367_367501


namespace problem_statement_l367_367173

open BigOperators

noncomputable def number_of_situations_none_form_pair : ℕ :=
  (Nat.choose 10 4) * 2^4

noncomputable def number_of_situations_exactly_two_pairs : ℕ :=
  Nat.choose 10 2

noncomputable def number_of_situations_one_pair_and_two_non_pairs : ℕ :=
  (Nat.choose 10 1) * (Nat.choose 9 2) * 2^2

theorem problem_statement (shoes : Finset (Fin 20)) (h : shoes.card = 4) :
  number_of_situations_none_form_pair = 3360 ∧
  number_of_situations_exactly_two_pairs = 45 ∧
  number_of_situations_one_pair_and_two_non_pairs = 1440 := 
  by sorry

end problem_statement_l367_367173


namespace sara_marbles_left_l367_367955

-- Definitions
def m1 : ℕ := 10
def m2 : ℕ := 5
def m3 : ℕ := 7
def m4 : ℕ := 3

-- Statement
theorem sara_marbles_left : m1 + m2 - m3 - m4 = 5 := by
  simp [m1, m2, m3, m4]
  sorry

end sara_marbles_left_l367_367955


namespace race_analysis_l367_367182

structure Runner (A B C : Type) :=
  (finished_before : A → B → Prop)
  (head_to_head : list (A × B × C) → ℕ × ℕ × ℕ)

def race_sequence (races : list (Runner A B C)) : list A × list B × list C :=
  races.foldl
    (λ (acc : list A × list B × list C) (race : Runner A B C),
      (acc.1.append [race.A], acc.2.append [race.B], acc.3.append [race.C])) ([], [], [])

theorem race_analysis (races : list (Runner A B C)) :
  ∃ A B C : Type,
    (2 = List.countp (λ (race : A × B × C), race.1.finished_before A B) races ∧ 
     2 = List.countp (λ (race : A × B × C), race.2.finished_before B C) races ∧ 
     2 = List.countp (λ (race : A × B × C), race.3.finished_before C A) races) →
  ∃ (new_race : Runner A B C), 
     (3 = List.countp (λ (race : A × B × C), race.1.finished_before A B) (races ++ [new_race]) ∧ 
      3 = List.countp (λ (race : A × B × C), race.2.finished_before B C) (races ++ [new_race]) ∧ 
      3 = List.countp (λ (race : A × B × C), race.3.finished_before C A) (races ++ [new_race]))
:= sorry

end race_analysis_l367_367182


namespace largest_prime_factor_of_T_l367_367292

-- Conditions as Lean definitions
def cyclic_four_digit_sequence (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), (i < seq.length) →
  let a := seq.nth_le i (by linarith) / 1000;
      b := (seq.nth_le i (by linarith) / 100) % 10;
      c := (seq.nth_le i (by linarith) / 10) % 10;
      d := seq.nth_le i (by linarith) % 10 in
  let ni := (i + 1) % seq.length in
  seq.nth_le ni (by linarith) = b * 1000 + c * 100 + d * 10 + (seq.nth_le (ni + 1) % seq.length (by linarith) / 1000)

-- Statement to prove
theorem largest_prime_factor_of_T (seq : List ℕ) 
  (h1 : ∀ x, x ∈ seq → 1000 ≤ x ∧ x < 10000) 
  (h2 : cyclic_four_digit_sequence seq) :
  ∃ p : ℕ, prime p ∧ p = 101 ∧ p ∣ (seq.sum) :=
sorry

end largest_prime_factor_of_T_l367_367292


namespace number_of_ways_to_place_pawns_l367_367032

theorem number_of_ways_to_place_pawns :
  let n := 5 in
  let number_of_placements := (n.factorial) in
  let number_of_permutations := (n.factorial) in
  number_of_placements * number_of_permutations = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l367_367032


namespace dot_product_AD_AB_correct_l367_367078

noncomputable def dot_product_AD_AB (A B C D : Point)
  (hD_midpoint : midpoint B C D)
  (hAB : dist A B = 2)
  (hBC : dist B C = 3)
  (hAC : dist A C = 4) : Float :=
  sorry

theorem dot_product_AD_AB_correct (A B C D : Point)
  (hD_midpoint : midpoint B C D)
  (hAB : dist A B = 2)
  (hBC : dist B C = 3)
  (hAC : dist A C = 4) :
  dot_product_AD_AB A B C D hD_midpoint hAB hBC hAC = 19 / 4 :=
by sorry

end dot_product_AD_AB_correct_l367_367078


namespace stable_f_range_l367_367447

noncomputable def stable_function (g : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ (a b c : ℝ), a ∈ D → b ∈ D → c ∈ D → 
  let g_vals := [g a, g b, g c] in
  g_vals.sum > 2 * g_vals.max

noncomputable def f (x m : ℝ) : ℝ := (real.log x) / x + m

theorem stable_f_range {m : ℝ} :
  stable_function (λ x, f x m) (set.Icc (1 / real.exp 2) (real.exp 2)) ↔ m > 4 * (real.exp 2) + 1 / real.exp :=
by {
  sorry
}

end stable_f_range_l367_367447


namespace negation_correct_l367_367162

-- Define the conditions
def conditions (x y : Real) : Prop := x ≠ 0 ∨ y ≠ 0

-- Define the original statement's conclusion
def original_conclusion (x y : Real) : Prop := xy = 0

-- Negation of the original statement
def negation_statement (x y : Real) : Prop := (x ≠ 0 ∨ y ≠ 0) → xy ≠ 0

-- The statement that needs to be proven
theorem negation_correct (x y : Real) : negation_statement x y := sorry

end negation_correct_l367_367162


namespace monotonic_increasing_range_of_a_l367_367753

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * (2^x) - 1) / (2^x + 1)

theorem monotonic_increasing (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : 
  (∀ x : ℝ, x > 0 → f a ((Real.log2 x) * (Real.log2 (8 / x))) + f a (-a) < 0) ↔ (a > 9 / 4) :=
by
  sorry

end monotonic_increasing_range_of_a_l367_367753


namespace area_sum_equality_l367_367919

theorem area_sum_equality (A B C D K L M N O : Point)
  (hK : midpoint K A B) (hL : midpoint L B C) (hM : midpoint M C D) (hN : midpoint N D A)
  (hKM : line_through K M) (hLN : line_through L N)
  (hintersect : K M ∩ L N = {O}) :
  area AKON + area CLOM = area BKOL + area DNOM := 
sorry

end area_sum_equality_l367_367919


namespace sum_adjacent_6_is_29_l367_367842

-- Define the grid and the placement of numbers 1 to 4
structure Grid :=
  (grid : Fin 3 → Fin 3 → Nat)
  (h_unique : ∀ i j, grid i j ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_distinct : Function.Injective (λ (i : Fin 3) (j : Fin 3), grid i j))
  (h_placement : grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧ grid 2 2 = 4)

-- Define the condition of the sum of numbers adjacent to 5 being 9
def sum_adjacent_5 (g : Grid) : Prop :=
  let (i, j) := (0, 1) in -- Position for number 5
  (g.grid (i.succ) j + g.grid (i.succ.pred) j + g.grid i (j.succ) + g.grid i (j.pred)) = 9

-- Define the main theorem
theorem sum_adjacent_6_is_29 (g : Grid) (h_sum_adj_5 : sum_adjacent_5 g) : 
  (g.grid 1 0 + g.grid 1 2 + g.grid 0 1 + g.grid 2 1 = 29) := sorry

end sum_adjacent_6_is_29_l367_367842


namespace smallest_x_for_convex_distortion_l367_367916

structure Hexagon (x : ℝ) :=
(vertices : fin 6 → ℝ × ℝ)
(side_length : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = x)

def is_distortion (H : Hexagon) (H' : Hexagon) : Prop :=
∀ i, dist (H.vertices i) (H'.vertices i) < 1

def is_convex (H : Hexagon) : Prop :=
∀ i j k, ccw (H.vertices i) (H.vertices j) (H.vertices k) ∨
         ccw (H.vertices j) (H.vertices k) (H.vertices i) ∨
         ccw (H.vertices k) (H.vertices i) (H.vertices j)

theorem smallest_x_for_convex_distortion (x : ℝ) :
  (∀ H H', H.side_length = x → is_distortion H H' → is_convex H') ↔ x ≥ 4 := 
by
  sorry

end smallest_x_for_convex_distortion_l367_367916


namespace fixed_point_for_parabola_lines_l367_367812

theorem fixed_point_for_parabola_lines :
  ∀ (p a b : ℝ) (M M1 M2 : ℝ × ℝ),
  ab ≠ 0 → b^2 ≠ 2 * p * a →
  (M.1, M.2 = y^2 := 2 * p * x) →
  (M1 ≠ M ∧ M2 ≠ M) →
  ∃ fixed : ℝ × ℝ,
  fixed = (a, 2 * p * a / b) ∧
  (∀ M1 M2, line_through M1 M2 = line_through fixed)
  :=
sorry

end fixed_point_for_parabola_lines_l367_367812


namespace determine_possible_dimensions_l367_367651

theorem determine_possible_dimensions (l : ℕ) (n : ℕ) :
  (2 * n^2 = 8/9 * l^2) → 
  (l^2 < 2 * (n + 1)^2) → 
  (2 ∃ l' (l' : ℕ < 9),
    (l = 3 * l') ∧ 
    (n = 2 * l') ∧ 
    l ∈ { 3, 6, 9, 12, 15, 18, 21, 24 }) := 
begin
  sorry 
end

end determine_possible_dimensions_l367_367651


namespace eq_iff_squared_eq_l367_367889

theorem eq_iff_squared_eq (a b : ℝ) : a = b ↔ a^2 + b^2 = 2 * a * b :=
by
  sorry

end eq_iff_squared_eq_l367_367889


namespace smallest_x_for_convex_distortion_l367_367918

def is_distortion (H : list (ℝ × ℝ)) (H' : list (ℝ × ℝ)) : Prop :=
  H.length = 6 ∧ H'.length = 6 ∧
  ∀ i, 0 ≤ i ∧ i < 6 → dist (H.nth_le i sorry) (H'.nth_le i sorry) < 1

def is_convex (V : list (ℝ × ℝ)) : Prop :=
  ∀ i j k, 0 ≤ i ∧ i < 6 → 0 ≤ j ∧ j < 6 → 0 ≤ k ∧ k < 6 → j ≠ i → k ≠ i → k ≠ j →
  let v_i := V.nth_le i sorry; let v_j := V.nth_le j sorry; let v_k := V.nth_le k sorry in
  (v_k.2 - v_i.2) * (v_j.1 - v_i.1) ≠ (v_j.2 - v_i.2) * (v_k.1 - v_i.1)

def regular_hexagon (x : ℝ) : list (ℝ × ℝ) := [
  (0, 0), (x, 0), (1.5 * x, sqrt 3 / 2 * x), (x, sqrt 3 * x), (0, sqrt 3 * x), (-0.5 * x, sqrt 3 / 2 * x)
]

theorem smallest_x_for_convex_distortion :
  ∀ H : list (ℝ × ℝ), H = regular_hexagon 4 → ∀ H', is_distortion H H' → is_convex H' :=
begin
  intros H H_eq H' H_distortion,
  sorry
end

end smallest_x_for_convex_distortion_l367_367918


namespace part_a_part_b_l367_367922

variables {A B C D E F H M N P Q T K L : Point}
  (triangle_non_isosceles_acute : ∀ {A B C : Point}, is_non_isosceles_acute_triangle A B C)
  (altitudes : ∀ {A B C D E F : Point}, are_altitudes A B C D E F)
  (orthocenter : ∀ {A B C H : Point}, is_orthocenter A B C H)
  (DE_int_AD_at_M : ∀ {A D E M : Point}, intersects_at D E A D M)
  (DF_int_AD_at_N : ∀ {A D F N : Point}, intersects_at D F A D N)
  (NP_perp_AB : ∀ {N P A B : Point}, perpendicular N P A B)
  (MQ_perp_AC : ∀ {M Q A C : Point}, perpendicular M Q A C)
  (tangencyT: ∀ {A P Q E F T : Point}, tangency_point A P Q E F T)
  (DT_int_MN_at_K : ∀ {D T M N K : Point}, intersects_at D T M N K)
  (reflectionA_at_MN : ∀ {A M N L : Point}, reflection_over_line A M N L)

theorem part_a (h₁ : triangle_non_isosceles_acute A B C) (h₂ : altitudes A B C D E F)
    (h₃ : orthocenter A B C H) (h₄ : DE_int_AD_at_M D E A D M) (h₅ : DF_int_AD_at_N D F A D N)
    (h₆ : NP_perp_AB N P A B) (h₇ : MQ_perp_AC M Q A C) : 
    tangent EF (circumcircle A P Q) :=
sorry

theorem part_b (h₁ : triangle_non_isosceles_acute A B C) (h₂ : altitudes A B C D E F)
    (h₃ : orthocenter A B C H) (h₄ : DE_int_AD_at_M D E A D M) (h₅ : DF_int_AD_at_N D F A D N)
    (h₆ : NP_perp_AB N P A B) (h₇ : MQ_perp_AC M Q A C) (h₈: tangencyT A P Q E F T)
    (h₉: DT_int_MN_at_K D T M N K) (h₁₀: reflectionA_at_MN A M N L) : 
    concur MN EF (circumcircle D L K) :=
sorry

end part_a_part_b_l367_367922


namespace regular_octoroll_into_circle_l367_367603

theorem regular_octoroll_into_circle (O C : Set Point) [regular_octagon O] [circle C] : 
  ∃ O' ⊆ O, center O' ∈ C := 
sorry

end regular_octoroll_into_circle_l367_367603


namespace symmetric_sum_l367_367394

theorem symmetric_sum (a b : ℤ) (h1 : a = -4) (h2 : b = -3) : a + b = -7 := by
  sorry

end symmetric_sum_l367_367394


namespace muffins_baked_by_James_correct_l367_367661

noncomputable def muffins_baked_by_James (muffins_baked_by_Arthur : ℝ) (ratio : ℝ) : ℝ :=
  muffins_baked_by_Arthur / ratio

theorem muffins_baked_by_James_correct :
  muffins_baked_by_James 115.0 12.0 = 9.5833 :=
by
  -- Add the proof here
  sorry

end muffins_baked_by_James_correct_l367_367661


namespace equation_solution_l367_367135

noncomputable def solve_equation (x : ℝ) : Prop :=
  (1/4) * x^(1/2 * Real.log x / Real.log 2) = 2^(1/4 * (Real.log x / Real.log 2)^2)

theorem equation_solution (x : ℝ) (hx : 0 < x) : solve_equation x → (x = 2^(2*Real.sqrt 2) ∨ x = 2^(-2*Real.sqrt 2)) :=
  by
  intro h
  sorry

end equation_solution_l367_367135


namespace constant_sum_distances_l367_367663

variables {r d : ℝ}
variables {C D P O : EuclideanGeometry.Point ℝ} -- assuming EuclideanGeometry.Point type for points

/- Additional assumptions based on given conditions -/
axiom circle_center {O : EuclideanGeometry.Point ℝ} (r : ℝ) : EuclideanGeometry.circle O r
axiom points_on_diameter {A B : EuclideanGeometry.Point ℝ} (h1 : EuclideanGeometry.distance A B = 2 * r) :
  EuclideanGeometry.diameter O A B
axiom points_on_line_segment {C D : EuclideanGeometry.Point ℝ} (A B : EuclideanGeometry.Point ℝ) :
  EuclideanGeometry.on_line_segment A B C ∧ EuclideanGeometry.on_line_segment A B D
axiom equal_distances {O C D : EuclideanGeometry.Point ℝ} (d : ℝ) :
  EuclideanGeometry.distance O C = d ∧ EuclideanGeometry.distance O D = d
axiom P_on_circumference {P O : EuclideanGeometry.Point ℝ} (r : ℝ) :
  EuclideanGeometry.distance P O = r

/- The theorem to prove: -/
theorem constant_sum_distances 
  (h_circle : EuclideanGeometry.circle O r)
  (h_diameter : ∀ {A B : EuclideanGeometry.Point ℝ}, EuclideanGeometry.distance A B = 2 * r → EuclideanGeometry.diameter O A B)
  (h_line_segments : ∀ {A B : EuclideanGeometry.Point ℝ}, EuclideanGeometry.on_line_segment A B C ∧ EuclideanGeometry.on_line_segment A B D)
  (h_equal_distances : EuclideanGeometry.distance O C = d ∧ EuclideanGeometry.distance O D = d)
  (h_on_circumference : EuclideanGeometry.distance P O = r) :
  EuclideanGeometry.distance P C ^ 2 + EuclideanGeometry.distance P D ^ 2 = 2 * (r ^ 2 + d ^ 2) := 
sorry

end constant_sum_distances_l367_367663


namespace sum_adjacent_to_six_l367_367859

theorem sum_adjacent_to_six :
  ∀ (table : fin 3 × fin 3 → ℕ),
    (∀ i j, table i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∃! i j, table i j = 1) ∧
    (∃! i j, table i j = 2) ∧
    (∃! i j, table i j = 3) ∧
    (∃! i j, table i j = 4) ∧
    (∃! i j, table i j = 5) → 
    (∀ i j, 
      table i j = 5 → 
        let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                       (if i < 2 then table (i+1, j) else 0) + 
                       (if j > 0 then table (i, j-1) else 0) + 
                       (if j < 2 then table (i, j+1) else 0)
        in adj_sum = 9) →
    (∃ i j, table i j = 6 ∧
      let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                     (if i < 2 then table (i+1, j) else 0) + 
                     (if j > 0 then table (i, j-1) else 0) + 
                     (if j < 2 then table (i, j+1) else 0) 
      in adj_sum = 29) := sorry

end sum_adjacent_to_six_l367_367859


namespace sum_of_angles_S_and_R_l367_367944

-- Define the problem conditions
variables (E F R G H : Point)
variables (arc_FR arc_RG : Real)
variables (on_circle : Circle E F R G H)
variables (arc_FR_eq : arc_FR = 60)
variables (arc_RG_eq : arc_RG = 48)

-- Define the angles
def angle_S (arc_FG arc_EH : Real) : Real := (arc_FG - arc_EH) / 2
def angle_R (arc_EH : Real) : Real := arc_EH / 2

-- Lean Mathematical Proof Statement
theorem sum_of_angles_S_and_R :
  let arc_FG := arc_FR + arc_RG,
  let arc_EH := 108 in
  angle_S arc_FG arc_EH + angle_R arc_EH = 54 :=
by
  sorry

end sum_of_angles_S_and_R_l367_367944


namespace bill_sunday_miles_l367_367937

variable (B : ℕ)

-- Conditions
def miles_Bill_Saturday : ℕ := B
def miles_Bill_Sunday : ℕ := B + 4
def miles_Julia_Sunday : ℕ := 2 * (B + 4)
def total_miles : ℕ := miles_Bill_Saturday B + miles_Bill_Sunday B + miles_Julia_Sunday B

theorem bill_sunday_miles (h : total_miles B = 32) : miles_Bill_Sunday B = 9 := by
  sorry

end bill_sunday_miles_l367_367937


namespace sum_coordinates_of_k_l367_367831

theorem sum_coordinates_of_k :
  ∀ (f k : ℕ → ℕ), (f 4 = 8) → (∀ x, k x = (f x) ^ 3) → (4 + k 4) = 516 :=
by
  intros f k h1 h2
  sorry

end sum_coordinates_of_k_l367_367831


namespace fish_count_l367_367112

theorem fish_count (m k ak : ℕ) (mk : m = 7) (kk : k = 3 * m) (ak : ak = k - 15) :
  m + k + ak = 34 :=
by
  sorry

end fish_count_l367_367112


namespace odd_function_incorrect_statements_count_l367_367021

variable {α : Type*}

def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

def is_even_function (f : α → α) : Prop :=
  ∀ x, f (-x) = f x

theorem odd_function_incorrect_statements_count {f : ℝ → ℝ} (h : is_odd_function f) : 
  let s1 := ∀ x, |f x| = |f (-x)|,
      s2 := ∀ x, f x * f (-x) = f (-x) * f x,
      s3 := ∀ x, f x * f (-x) ≥ 0,
      s4 := ∀ x, f (-x) + |f x| = 0
  in (¬s3 ∧ ¬s4) ∧ s1 ∧ s2 := by
  sorry

end odd_function_incorrect_statements_count_l367_367021


namespace length_of_tube_l367_367238

theorem length_of_tube (h1 : ℝ) (mass_water : ℝ) (rho : ℝ) (doubled_pressure : Bool) (g : ℝ) :
  h1 = 1.5 → 
  mass_water = 1000 → 
  rho = 1000 → 
  g = 9.8 →
  doubled_pressure = true →
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  intros h1_val mass_water_val rho_val g_val doubled_pressure_val
  have : ∃ h2, 29400 = 1000 * g * (h1 + h2) := sorry
  use 1.5
  assumption_sid
  sorry
  
end length_of_tube_l367_367238


namespace probability_of_8_or_9_ring_l367_367165

theorem probability_of_8_or_9_ring (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  p9 + p8 = 0.5 :=
by
  sorry

end probability_of_8_or_9_ring_l367_367165


namespace evaluate_expression_l367_367695

theorem evaluate_expression (a b : ℕ) (h₁ : a = 3) (h₂ : b = 4) : ((a^b)^a - (b^a)^b) = -16246775 :=
by
  rw [h₁, h₂]
  sorry

end evaluate_expression_l367_367695


namespace vector_inequality_l367_367828

variable {V : Type*} [inner_product_space ℝ V]

theorem vector_inequality
  {a b : V}
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (h : ∥a + b∥ = ∥b∥) :
  ∥2 • b∥ > ∥a + 2 • b∥ :=
sorry

end vector_inequality_l367_367828


namespace unique_pair_l367_367332

theorem unique_pair (m n : ℕ) (h1 : m < n) (h2 : n ∣ m^2 + 1) (h3 : m ∣ n^2 + 1) : (m, n) = (1, 1) :=
sorry

end unique_pair_l367_367332


namespace adi_change_l367_367658

theorem adi_change : 
  let pencil := 0.35
  let notebook := 1.50
  let colored_pencils := 2.75
  let discount := 0.05
  let tax := 0.10
  let payment := 20.00
  let total_cost_before_discount := pencil + notebook + colored_pencils
  let discount_amount := discount * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let tax_amount := tax * total_cost_after_discount
  let total_cost := total_cost_after_discount + tax_amount
  let change := payment - total_cost
  change = 15.19 :=
by
  sorry

end adi_change_l367_367658


namespace sum_adjacent_to_6_is_29_l367_367872

def in_grid (n : ℕ) (grid : ℕ → ℕ → ℕ) := ∃ i j, grid i j = n

def adjacent_sum (grid : ℕ → ℕ → ℕ) (i j : ℕ) : ℕ :=
  grid (i-1) j + grid (i+1) j + grid i (j-1) + grid i (j+1)

def grid := λ i j =>
  if i = 0 ∧ j = 0 then 1 else
  if i = 2 ∧ j = 0 then 2 else
  if i = 0 ∧ j = 2 then 3 else
  if i = 2 ∧ j = 2 then 4 else
  if i = 1 ∧ j = 1 then 6 else 0

lemma numbers_positions_adjacent_5 :
  grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧
  grid 2 2 = 4 ∧ 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else 0 in
  adjacent_sum grid 1 0 = 1 + 2 + 6 :=
by sorry

theorem sum_adjacent_to_6_is_29 : 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else
                     if i = 0 ∧ j = 1 then 7 else
                     if i = 2 ∧ j = 1 then 8 else
                     if i = 1 ∧ j = 2 then 9 else 0 in
  adjacent_sum grid 1 1 = 5 + 7 + 8 + 9 :=
by sorry

end sum_adjacent_to_6_is_29_l367_367872


namespace trajectory_is_hyperbola_branch_l367_367746

def P_trajectory (x y : ℝ) : Prop :=
  abs ((sqrt ((x - 3)^2 + (y - 3)^2)) - (sqrt ((x + 3)^2 + (y - 3)^2))) = 4

theorem trajectory_is_hyperbola_branch :
  ∀ x y : ℝ, P_trajectory x y → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (y^2 / b^2 - (x - c)^2 / a^2 = 1 ∨ y^2 / b^2 - (x + c)^2 / a^2 = 1) :=
sorry

end trajectory_is_hyperbola_branch_l367_367746


namespace distinct_pawn_placements_on_chess_board_l367_367041

def numPawnPlacements : ℕ :=
  5! * 5!

theorem distinct_pawn_placements_on_chess_board :
  numPawnPlacements = 14400 := by
  sorry

end distinct_pawn_placements_on_chess_board_l367_367041


namespace find_slope_l367_367732

-- Define the line equation
def line (x y : ℝ) : Prop :=
  4 * x + 7 * y = 28

-- Define the slope of the line to be proved
def slope (m : ℝ) : Prop :=
  m = -4 / 7

-- State the theorem
theorem find_slope : ∃ m : ℝ, slope m ∧ (∀ x y : ℝ, line x y → y = m * x + 4) :=
by
  sorry

end find_slope_l367_367732


namespace parking_fee_for_2_hours_daytime_parking_fee_for_4_2_hours_daytime_parking_fee_for_leave_between_1930_and_2400_l367_367241

def parking_fee (hours : ℝ) (start_time : ℝ) : ℝ :=
  if start_time >= 7.5 ∧ start_time < 19.5 then
    if hours <= 2 then (⌈2 * hours⌉) * 2
    else if hours <= 9 then 4 + (⌈2 * (hours - 2)⌉) * 3
    else if hours <= 24 then min (4 + 42 + ⌈2 * (hours - 9)⌉) 60
    else 0  -- this case shouldn't occur within 24 hours
  else if start_time >= 19.5 ∨ start_time < 7.5 then
    if hours <= 2 then (⌈2 * hours⌉) * 1
    else min ((⌈2 * hours⌉)) * 1 15
  else 
    0 -- this case shouldn't occur

theorem parking_fee_for_2_hours_daytime : 
  parking_fee 2 7.5 = 8 :=
by sorry

theorem parking_fee_for_4_2_hours_daytime : 
  parking_fee 4.2 7.5 = 23 :=
by sorry

theorem parking_fee_for_leave_between_1930_and_2400 (x : ℕ) (h : 10 ≤ x ∧ x ≤ 13) : 
  parking_fee x 10.5 = (if x = 10 then 59 else 60) :=
by sorry

end parking_fee_for_2_hours_daytime_parking_fee_for_4_2_hours_daytime_parking_fee_for_leave_between_1930_and_2400_l367_367241


namespace bottles_needed_l367_367664

variable (guests : ℕ) (pctChampagne pctWine pctJuice : ℚ) 
variable (glassesPerGuestChampagne glassesPerGuestWine glassesPerGuestJuice : ℕ)
variable (servingsPerBottleChampagne servingsPerBottleWine servingsPerBottleJuice : ℕ)

-- Conditions
def totalGuests := 120
def percentageChampagne := 0.60
def percentageWine := 0.30
def percentageJuice := 0.10

def glassesPerGuestChampagne := 2
def glassesPerGuestWine := 1
def glassesPerGuestJuice := 1

def servingsPerBottleChampagne := 6
def servingsPerBottleWine := 5
def servingsPerBottleJuice := 4

-- Prove the required number of bottles
theorem bottles_needed
  (H : guests = totalGuests)
  (P1 : pctChampagne = percentageChampagne)
  (P2 : pctWine = percentageWine)
  (P3 : pctJuice = percentageJuice)
  (G1 : glassesPerGuestChampagne = 2)
  (G2 : glassesPerGuestWine = 1)
  (G3 : glassesPerGuestJuice = 1)
  (S1 : servingsPerBottleChampagne = 6)
  (S2 : servingsPerBottleWine = 5)
  (S3 : servingsPerBottleJuice = 4):
  let guestsChampagne := guests * pctChampagne
  let glassesChampagne := guestsChampagne * glassesPerGuestChampagne
  let bottlesChampagne := glassesChampagne / servingsPerBottleChampagne
  let guestsWine := guests * pctWine
  let glassesWine := guestsWine * glassesPerGuestWine
  let bottlesWine := (glassesWine / servingsPerBottleWine).ceil
  let guestsJuice := guests * pctJuice
  let glassesJuice := guestsJuice * glassesPerGuestJuice
  let bottlesJuice := glassesJuice / servingsPerBottleJuice in
  
  bottlesChampagne = 24 ∧ bottlesWine = 8 ∧ bottlesJuice = 3 := by
  sorry

end bottles_needed_l367_367664


namespace length_of_tube_l367_367235

/-- Prove that the length of the tube is 1.5 meters given the initial conditions -/
theorem length_of_tube (h1 : ℝ) (m_water : ℝ) (rho : ℝ) (g : ℝ) (p_ratio : ℝ) :
  h1 = 1.5 ∧ m_water = 1000 ∧ rho = 1000 ∧ g = 9.8 ∧ p_ratio = 2 → 
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  sorry

end length_of_tube_l367_367235


namespace octagon_center_inside_circle_pentagon_center_inside_circle_n_gon_center_inside_circle_l367_367602

-- Definition of regular polygon and flipping
structure RegularPolygon (n : ℕ) :=
  (sides_eq : ∀ i j, (0 ≤ i ∧ i < n) ∧ (0 ≤ j ∧ j < n) → SideLength i = SideLength j)
  (interior_angle : RealAngle := (n - 2) * 180 / n)

-- Function that checks if flipping a polygon results in its center being inside the circle
def can_center_be_inside_circle (n : ℕ) (polygon : RegularPolygon n) (circle_radius : ℝ) : Prop :=
  ∃ flips : (ℕ → ℕ), center_inside_circle polygon flips circle_radius

-- Specification of the problem for octagon
theorem octagon_center_inside_circle (circle_radius : ℝ) :
  can_center_be_inside_circle 8 (RegularPolygon.mk (λ i j _, rfl) (by norm_num)) circle_radius :=
sorry

-- Specification of the problem for pentagon
theorem pentagon_center_inside_circle (circle_radius : ℝ) :
  can_center_be_inside_circle 5 (RegularPolygon.mk (λ i j _, rfl) (by norm_num)) circle_radius :=
sorry

-- General case for which polygons have the center inside the circle after flipping
theorem n_gon_center_inside_circle (n : ℕ) (h : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6) (circle_radius : ℝ) :
  can_center_be_inside_circle n (RegularPolygon.mk (λ i j _, rfl) (by norm_num)) circle_radius :=
sorry

end octagon_center_inside_circle_pentagon_center_inside_circle_n_gon_center_inside_circle_l367_367602


namespace speed_of_point_C_l367_367364

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l367_367364


namespace trapezium_other_parallel_side_l367_367720

theorem trapezium_other_parallel_side (a : ℝ) (b d : ℝ) (area : ℝ) 
  (h1 : a = 18) (h2 : d = 15) (h3 : area = 285) : b = 20 :=
by
  sorry

end trapezium_other_parallel_side_l367_367720


namespace best_model_is_model4_l367_367893

-- Define the R^2 values for each model
def R_squared_model1 : ℝ := 0.25
def R_squared_model2 : ℝ := 0.80
def R_squared_model3 : ℝ := 0.50
def R_squared_model4 : ℝ := 0.98

-- Define the highest R^2 value and which model it belongs to
theorem best_model_is_model4 (R1 R2 R3 R4 : ℝ) (h1 : R1 = R_squared_model1) (h2 : R2 = R_squared_model2) (h3 : R3 = R_squared_model3) (h4 : R4 = R_squared_model4) : 
  (R4 = 0.98) ∧ (R4 > R1) ∧ (R4 > R2) ∧ (R4 > R3) :=
by
  sorry

end best_model_is_model4_l367_367893


namespace range_of_a_for_increasing_y_l367_367355

theorem range_of_a_for_increasing_y :
  ∀ (y : ℝ → ℝ) (a : ℝ), (y = λ x, 2 * x^2 - 4 * x - 1) → (∀ x > a, ∀ x' > x, y x < y x') ↔ a ≤ 1 :=
sorry

end range_of_a_for_increasing_y_l367_367355


namespace cube_angle_diagonals_l367_367550

theorem cube_angle_diagonals (q : ℝ) (h : q = 60) : 
  ∃ (d : String), d = "space diagonals" :=
by
  sorry

end cube_angle_diagonals_l367_367550


namespace complex_pow_six_eq_eight_i_l367_367287

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end complex_pow_six_eq_eight_i_l367_367287


namespace chord_dividing_angle_in_half_length_eq_4_l367_367050

theorem chord_dividing_angle_in_half_length_eq_4
  (AB BC : ℝ)
  (angle_ABC : ℝ)
  (h_AB : AB = sqrt 3)
  (h_BC : BC = 3 * sqrt 3)
  (h_angle_ABC : angle_ABC = 60) :
  ∃ chord_length : ℝ, 
    chord_length = 4 :=
by {
  sorry
}

end chord_dividing_angle_in_half_length_eq_4_l367_367050


namespace rickshaw_distance_l367_367643

theorem rickshaw_distance (km1_charge : ℝ) (rate_per_km : ℝ) (total_km : ℝ) (total_charge : ℝ) :
  km1_charge = 13.50 → rate_per_km = 2.50 → total_km = 13 → total_charge = 103.5 → (total_charge - km1_charge) / rate_per_km = 36 :=
by
  intro h1 h2 h3 h4
  -- We would fill in proof steps here, but skipping as required.
  sorry

end rickshaw_distance_l367_367643


namespace complement_of_A_in_U_l367_367008

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ 2}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_of_A_in_U :
  (U \ A) = complement_U_A :=
sorry

end complement_of_A_in_U_l367_367008


namespace complex_pow_six_eq_eight_i_l367_367286

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end complex_pow_six_eq_eight_i_l367_367286


namespace probability_adjacent_difference_l367_367523

noncomputable def probability_no_adjacent_same_rolls : ℚ :=
  (7 / 8) ^ 6

theorem probability_adjacent_difference :
  let num_people := 6
  let sides_of_die := 8
  ( ∀ i : ℕ, 0 ≤ i ∧ i < num_people -> (∃ x : ℕ, 1 ≤ x ∧ x ≤ sides_of_die)) →
  probability_no_adjacent_same_rolls = 117649 / 262144 := 
by 
  sorry

end probability_adjacent_difference_l367_367523


namespace arrangement_of_numbers_l367_367203

theorem arrangement_of_numbers (numbers : Finset ℕ) 
  (h1 : numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) 
  (h_sum : ∀ a b c d e f, a + b + c + d + e + f = 33)
  (h_group_sum : ∀ k1 k2 k3 k4, k1 + k2 + k3 + k4 = 26)
  : ∃ (n : ℕ), n = 2304 := by
  sorry

end arrangement_of_numbers_l367_367203


namespace liked_both_desserts_l367_367054

noncomputable def total_students : ℕ := 50
noncomputable def apple_pie_lovers : ℕ := 22
noncomputable def chocolate_cake_lovers : ℕ := 20
noncomputable def neither_dessert_lovers : ℕ := 17
noncomputable def both_desserts_lovers : ℕ := 9

theorem liked_both_desserts :
  (total_students - neither_dessert_lovers) + both_desserts_lovers = apple_pie_lovers + chocolate_cake_lovers - both_desserts_lovers :=
by
  sorry

end liked_both_desserts_l367_367054


namespace evaluate_expression_l367_367692

theorem evaluate_expression (a b : ℕ) (h₁ : a = 3) (h₂ : b = 4) : ((a^b)^a - (b^a)^b) = -16246775 :=
by
  rw [h₁, h₂]
  sorry

end evaluate_expression_l367_367692


namespace velocity_of_point_C_l367_367371

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l367_367371


namespace seating_arrangements_l367_367505

-- Definitions (conditions):
def family_members : Finset String := {"Mr. Lopez", "Mrs. Lopez", "Child1", "Child2"}

def is_driver (x : String) : Prop :=
  x = "Mr. Lopez" ∨ x = "Mrs. Lopez"

-- Main theorem:
theorem seating_arrangements : 
  ∃ (n : ℕ), n = 12 ∧ 
  (let driver_choices : Finset String := family_members.filter is_driver in
  let passenger_choices : Finset String := family_members \ driver_choices in
  let front_passenger_seat : Finset String := passenger_choices in
  let back_seat_arrangement : Finset (String × String) := (passenger_choices \ front_passenger_seat).product (passenger_choices \ front_passenger_seat) in
  driver_choices.card * front_passenger_seat.card * back_seat_arrangement.card = n) :=
begin
  have h1 : (family_members.filter is_driver).card = 2, -- 2 choices for the driver
    sorry,
  have h2 : (family_members.filter (λ x, ¬is_driver x)).card = 2, -- 2 children left
    sorry,
  have h3 : ∃ n, n = 12,
    use 12,
    split,
    exact rfl,
    sorry
end

end seating_arrangements_l367_367505


namespace symmetric_line_eq_l367_367340

theorem symmetric_line_eq (x y : ℝ) : (x - y = 0) → (x = 1) → (y = -x + 2) :=
by
  sorry

end symmetric_line_eq_l367_367340


namespace total_questions_attempted_l367_367462

theorem total_questions_attempted 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (total_questions : ℕ) (incorrect_answers : ℕ)
  (h_marks_per_correct : marks_per_correct = 4)
  (h_marks_lost_per_wrong : marks_lost_per_wrong = 1) 
  (h_total_marks : total_marks = 130) 
  (h_correct_answers : correct_answers = 36) 
  (h_score_eq : marks_per_correct * correct_answers - marks_lost_per_wrong * incorrect_answers = total_marks)
  (h_total_questions : total_questions = correct_answers + incorrect_answers) : 
  total_questions = 50 :=
by
  sorry

end total_questions_attempted_l367_367462


namespace william_tickets_l367_367599

theorem william_tickets (initial_tickets final_tickets : ℕ) (h1 : initial_tickets = 15) (h2 : final_tickets = 18) : 
  final_tickets - initial_tickets = 3 := 
by
  sorry

end william_tickets_l367_367599


namespace sum_adjacent_to_6_is_29_l367_367870

def in_grid (n : ℕ) (grid : ℕ → ℕ → ℕ) := ∃ i j, grid i j = n

def adjacent_sum (grid : ℕ → ℕ → ℕ) (i j : ℕ) : ℕ :=
  grid (i-1) j + grid (i+1) j + grid i (j-1) + grid i (j+1)

def grid := λ i j =>
  if i = 0 ∧ j = 0 then 1 else
  if i = 2 ∧ j = 0 then 2 else
  if i = 0 ∧ j = 2 then 3 else
  if i = 2 ∧ j = 2 then 4 else
  if i = 1 ∧ j = 1 then 6 else 0

lemma numbers_positions_adjacent_5 :
  grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧
  grid 2 2 = 4 ∧ 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else 0 in
  adjacent_sum grid 1 0 = 1 + 2 + 6 :=
by sorry

theorem sum_adjacent_to_6_is_29 : 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else
                     if i = 0 ∧ j = 1 then 7 else
                     if i = 2 ∧ j = 1 then 8 else
                     if i = 1 ∧ j = 2 then 9 else 0 in
  adjacent_sum grid 1 1 = 5 + 7 + 8 + 9 :=
by sorry

end sum_adjacent_to_6_is_29_l367_367870


namespace sum_of_two_longest_altitudes_l367_367821

-- Define what it means for a triangle to have sides 7, 24, 25
def is_triangle_7_24_25 (a b c : ℝ) : Prop :=
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 7 ∧ b = 25 ∧ c = 24) ∨ (a = 24 ∧ b = 7 ∧ c = 25) ∨ 
  (a = 24 ∧ b = 25 ∧ c = 7) ∨ (a = 25 ∧ b = 7 ∧ c = 24) ∨ (a = 25 ∧ b = 24 ∧ c = 7)

-- Prove the sum of the two longest altitudes in such a triangle is 31
theorem sum_of_two_longest_altitudes (a b c : ℝ) (h : is_triangle_7_24_25 a b c) :
  let h_altitude (c : ℝ) := (a * b) / c in
  (a + b) - (a * b > c ∨ b * c > a ∨ c * a > b) → ℝ :=
by
  sorry

end sum_of_two_longest_altitudes_l367_367821


namespace inequality_proof_l367_367103

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
by
  sorry

end inequality_proof_l367_367103


namespace angle_measure_of_Q_l367_367975

namespace RegularDecagon

def is_regular_decagon (D : Type) (A B C D E F G H I J : D) : Prop :=
  -- Definition for regular decagon
  sorry

def extended_sides_meet_at (A H E F Q : Type) : Prop :=
  -- Definition for the sides of regular decagon being extended to meet at point Q
  sorry

theorem angle_measure_of_Q {D : Type} (A B C D E F G H I J Q : D) 
    (h_regular_decagon : is_regular_decagon D A B C D E F G H I J)
    (h_extended_sides : extended_sides_meet_at A H E F Q) :
    angle_measure (A H E F Q) = 72 :=
sorry

end RegularDecagon

end angle_measure_of_Q_l367_367975


namespace largest_k_power_of_2_dividing_product_of_first_50_even_numbers_l367_367099

open Nat

theorem largest_k_power_of_2_dividing_product_of_first_50_even_numbers :
  let Q := (List.range (50 + 1)).map (λ n, 2 * n).prod in
  let k := (Q.factorization 2) in
  k = 97 :=
by
  sorry

end largest_k_power_of_2_dividing_product_of_first_50_even_numbers_l367_367099


namespace slope_of_line_l367_367727

theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (∃ m b : ℝ, y = m * x + b ∧ m = -4/7) :=
by
   intro h
   use -4/7, 28/7
   rw [mul_comm 7 y, ←sub_eq_iff_eq_add]
   simp [eq_sub_of_add_eq, h]
   split
   { sorry }
   { refl }

end slope_of_line_l367_367727


namespace shawn_password_possibilities_l367_367521

theorem shawn_password_possibilities : 
  ∃ n, n = (Nat.choose 4 2) ∧ n = 6 :=
by
  exists Nat.choose 4 2
  split
  any_goals sorry
  have : Nat.choose 4 2 = 6,
  from rfl
  exact this

end shawn_password_possibilities_l367_367521


namespace max_horizontal_segment_length_l367_367725

theorem max_horizontal_segment_length (y : ℝ → ℝ) (h : ∀ x, y x = x^3 - x) :
  ∃ a, (∀ x₁, y x₁ = y (x₁ + a)) ∧ a = 2 :=
by
  sorry

end max_horizontal_segment_length_l367_367725


namespace solve_inequality_l367_367984

theorem solve_inequality (x : ℝ) :
  (9 * x^2 + 27 * x - 40) / ((3 * x - 4) * (x + 5)) < 5 ↔
  (x ∈ Ioo (-6:ℝ) (-5) ∨ x ∈ Ioo (4/3:ℝ) (5/3)) :=
by
  sorry

end solve_inequality_l367_367984


namespace differential_equation_solution_l367_367898

theorem differential_equation_solution (C : ℝ) (x : ℝ) (y : ℝ) :
  (differentiable x) → (y' x = y' (x)) + x * y = x^2 → 
  ∃ C : ℝ, y = (x^3 / 4) + (C / x) := sorry

end differential_equation_solution_l367_367898


namespace four_points_away_l367_367420

noncomputable def center : ℝ × ℝ := (2, 2)
noncomputable def radius : ℝ := 2 * Real.sqrt 5
noncomputable def line (m : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = 2 * p.1 + m

theorem four_points_away (m : ℝ) :
  (∃ p1 p2 p3 p4 : ℝ × ℝ,
    (x - 2)^2 + (y - 2)^2 = 20 ∧
    (Real.abs (4 - 2 + m) / Real.sqrt (2^2 + 1) = Real.sqrt 5)) →
  -7 < m ∧ m < 3 :=
sorry

end four_points_away_l367_367420


namespace problem1_problem2_l367_367278

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem1 : sqrt 12 + sqrt 8 * sqrt 6 = 6 * sqrt 3 := by
  sorry

theorem problem2 : sqrt 12 + 1 / (sqrt 3 - sqrt 2) - sqrt 6 * sqrt 3 = 3 * sqrt 3 - 2 * sqrt 2 := by
  sorry

end problem1_problem2_l367_367278


namespace exactly_one_even_needs_assumption_l367_367948

open Nat

theorem exactly_one_even_needs_assumption 
  {a b c : ℕ} 
  (h : (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) ∧ (a % 2 = 0 → b % 2 = 1) ∧ (a % 2 = 0 → c % 2 = 1) ∧ (b % 2 = 0 → c % 2 = 1)) :
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) → (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) → (¬(a % 2 = 0 ∧ b % 2 = 0) ∧ ¬(b % 2 = 0 ∧ c % 2 = 0) ∧ ¬(a % 2 = 0 ∧ c % 2 = 0)) := 
by
  sorry

end exactly_one_even_needs_assumption_l367_367948


namespace sum_adjacent_to_6_is_29_l367_367854
-- Import the Mathlib library for the necessary tools and functions

/--
  In a 3x3 table filled with numbers from 1 to 9 such that each number appears exactly once, 
  with conditions: 
    * (1, 1) contains 1, (3, 1) contains 2, (1, 3) contains 3, (3, 3) contains 4,
    * The sum of the numbers in the cells adjacent to the cell containing 5 is 9,
  Prove that the sum of the numbers in the cells adjacent to the cell containing 6 is 29.
-/
theorem sum_adjacent_to_6_is_29 
  (table : Fin 3 → Fin 3 → Fin 9)
  (H_uniqueness : ∀ i j k l, (table i j = table k l) → (i = k ∧ j = l))
  (H_valid_entries : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)
  (H_initial_positions : table 0 0 = 1 ∧ table 2 0 = 2 ∧ table 0 2 = 3 ∧ table 2 2 = 4)
  (H_sum_adj_to_5 : ∃ (i j : Fin 3), table i j = 5 ∧ 
                      ((i > 0 ∧ table (i-1) j +
                       (i < 2 ∧ table (i+1) j) +
                       (j > 0 ∧ table i (j-1)) +
                       (j < 2 ∧ table i (j+1))) = 9)) :
  ∃ i j, table i j = 6 ∧
  (i > 0 ∧ table (i-1) j +
   (i < 2 ∧ table (i+1) j) +
   (j > 0 ∧ table i (j-1)) +
   (j < 2 ∧ table i (j+1))) = 29 := sorry

end sum_adjacent_to_6_is_29_l367_367854


namespace num_representable_integers_1200_l367_367302

def f (x : ℝ) : ℤ := ⌊3 * x⌋ + ⌊5 * x⌋ + ⌊7 * x⌋ + ⌊9 * x⌋

theorem num_representable_integers_1200 : ∃ n : ℕ, n = 720 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ 1200) → (∃ x : ℝ, f x = k) ↔ k ≤ n := 
sorry

end num_representable_integers_1200_l367_367302


namespace area_of_trapezoid_l367_367993

-- Define the trapezoid and the given conditions
variables {A B C D M : Type} -- Points representing vertices and intersection point of diagonals
-- Triangles ABM and CDM and their areas
variables (area_ABM area_CDM : ℝ) 
-- Heights from M in triangles ABM and CDM
variables (m1 m2 : ℝ)

-- Given conditions
def is_trapezoid (ABCD : Type) : Prop :=
  ∃ (AB CD : ℝ), area_ABM = 18 ∧ area_CDM = 50

-- Define the midline and height of the trapezoid
lemma trapezoid_area (ABCD : Type) [is_trapezoid ABCD] : ℝ :=
  let m1 := 3 / 5 * m2 in
  let H := m1 + m2 in
  let midline := 4 / 5 * 50 in
  midline * H

-- Proposition: the area of the trapezoid
theorem area_of_trapezoid (ABCD : Type) [is_trapezoid ABCD] : ∃ (T : ℝ), T = 128 :=
by
  let m2 := 50 / area_CDM in
  let m1 := 3 / 5 * m2 in
  let H := m1 + m2 in
  let midline := 4 / 5 * 50 in
  let T := midline * H in
  exact ⟨T, by norm_num⟩

end area_of_trapezoid_l367_367993


namespace triangle_ABC_right_l367_367766

noncomputable def pointA : ℝ × ℝ := (1, 2)
noncomputable def pointLine : ℝ × ℝ := (5, -2)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

theorem triangle_ABC_right :
  ∃ B C : ℝ × ℝ, parabola B.1 B.2 ∧ parabola C.1 C.2 ∧
  line_through pointLine B ∧ line_through pointLine C ∧
  is_right_triangle pointA B C :=
sorry

end triangle_ABC_right_l367_367766


namespace pants_cost_l367_367510

theorem pants_cost (starting_amount shirts_cost shirts_count amount_left money_after_shirts pants_cost : ℕ) 
    (h1 : starting_amount = 109)
    (h2 : shirts_cost = 11)
    (h3 : shirts_count = 2)
    (h4 : amount_left = 74)
    (h5 : money_after_shirts = starting_amount - shirts_cost * shirts_count)
    (h6 : pants_cost = money_after_shirts - amount_left) :
  pants_cost = 13 :=
by
  sorry

end pants_cost_l367_367510


namespace p_minus_q_value_l367_367542

theorem p_minus_q_value
  (a b c y1 : ℝ)
  (h1 : y1 = 5)
  (h2 : a = 5)
  (h3 : b = 2)
  (h4 : c = -2)
  (h5 : ∀ x, y1 = a * x ^ 2 + b * x + c → (x = 1 ∨ x = -1.4)) :
  ∃ p q : ℕ, Nat.coprime p q ∧ p - q = 476 :=
by
  sorry

end p_minus_q_value_l367_367542


namespace length_of_tube_l367_367236

/-- Prove that the length of the tube is 1.5 meters given the initial conditions -/
theorem length_of_tube (h1 : ℝ) (m_water : ℝ) (rho : ℝ) (g : ℝ) (p_ratio : ℝ) :
  h1 = 1.5 ∧ m_water = 1000 ∧ rho = 1000 ∧ g = 9.8 ∧ p_ratio = 2 → 
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  sorry

end length_of_tube_l367_367236


namespace matilda_fathers_chocolate_bars_l367_367934

/-- Matilda had 20 chocolate bars and shared them evenly amongst herself and her 4 sisters.
    When her father got home, he was upset that they did not put aside any chocolates for him.
    They felt bad, so they each gave up half of their chocolate bars for their father.
    Their father then gave 3 chocolate bars to their mother and ate some.
    Matilda's father had 5 chocolate bars left.
    Prove that Matilda's father ate 2 chocolate bars. -/
theorem matilda_fathers_chocolate_bars:
  ∀ (total_chocolates initial_people chocolates_per_person given_to_father chocolates_left chocolates_eaten: ℕ ),
    total_chocolates = 20 →
    initial_people = 5 →
    chocolates_per_person = total_chocolates / initial_people →
    given_to_father = (chocolates_per_person / 2) * initial_people →
    chocolates_left = given_to_father - 3 →
    chocolates_left - 5 = chocolates_eaten →
    chocolates_eaten = 2 :=
by
  intros
  sorry

end matilda_fathers_chocolate_bars_l367_367934


namespace translation_of_sin_2x_right_pi_over_3_l367_367573

-- Define the original function
def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the translated function
def g (x : ℝ) : ℝ := Real.sin (2 * x - 2 * Real.pi / 3)

-- Theorem statement
theorem translation_of_sin_2x_right_pi_over_3 :
    ∀ x : ℝ, g x = f (x - Real.pi / 3) :=
sorry

end translation_of_sin_2x_right_pi_over_3_l367_367573


namespace P_is_orthocenter_l367_367119

noncomputable def point := (ℝ, ℝ)
def dist (p1 p2 : point) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def triangle (A B C : point) := {a : point // a = A ∨ a = B ∨ a = C}
def is_orthocenter (P A B C : point) := ∀ (d : point), d ∈ {A, B, C} → ∃ (l : line), l ⊥ PA ∧ P ∈ l

theorem P_is_orthocenter (P A B C : point)
  (hPA : dist P A = 3)
  (hPB : dist P B = 5)
  (hPC : dist P C = 7)
  (h_max_area : ∀ A' B' C', let area_ABC := area A B C in
                  area_ABC ≥ area A' B' C') :
  is_orthocenter P A B C :=
sorry

end P_is_orthocenter_l367_367119


namespace smallest_value_36k_minus_5l_l367_367268

theorem smallest_value_36k_minus_5l (k l : ℕ) :
  ∃ k l, 0 < 36^k - 5^l ∧ (∀ k' l', (0 < 36^k' - 5^l' → 36^k - 5^l ≤ 36^k' - 5^l')) ∧ 36^k - 5^l = 11 :=
by sorry

end smallest_value_36k_minus_5l_l367_367268


namespace error_percentage_in_area_l367_367063

theorem error_percentage_in_area
  (L W : ℝ)          -- Actual length and width of the rectangle
  (hL' : ℝ)          -- Measured length with 8% excess
  (hW' : ℝ)          -- Measured width with 5% deficit
  (hL'_def : hL' = 1.08 * L)  -- Condition for length excess
  (hW'_def : hW' = 0.95 * W)  -- Condition for width deficit
  :
  ((hL' * hW' - L * W) / (L * W) * 100 = 2.6) := sorry

end error_percentage_in_area_l367_367063


namespace complex_number_in_second_quadrant_l367_367625

-- Definitions representing the conditions
def imaginary_unit : ℂ := complex.I

def z : ℂ := -1 + 3 * imaginary_unit

-- Definition to check the quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The main proof statement
theorem complex_number_in_second_quadrant : in_second_quadrant z :=
by
  sorry

end complex_number_in_second_quadrant_l367_367625


namespace sequence_add_l367_367310

theorem sequence_add (x y : ℝ) (h1 : x = 81 * (1 / 3)) (h2 : y = x * (1 / 3)) : x + y = 36 :=
sorry

end sequence_add_l367_367310


namespace triangle_HM_length_l367_367049

noncomputable def length_HM (A B C M H : Point) (AB BC CA : ℝ) (AM MB : ℝ) : ℝ :=
  -- definition ensures the lengths between points and relations
  sorry

theorem triangle_HM_length :
  ∀ (A B C M H : Point) (AB BC CA : ℝ), 
  AB = 20 → BC = 18 → CA = 22 → 
  (AM : ℝ) → (MB : ℝ) → 
  AM = 2*MB → AM + MB = AB → 
  -- Defining H as the foot of altitude from A to BC
  foot_of_altitude (A B C H) →
  ∃ (HM : ℝ), length_HM A B C M H AB BC CA AM MB = HM := 
by
  intros A B C M H AB BC CA hAB hBC hCA AM MB hAM2MB hAM_MB hfoot_of_altitude,
  sorry

end triangle_HM_length_l367_367049


namespace tetrahedrons_identical_probability_l367_367193

theorem tetrahedrons_identical_probability :
  ∃ p : ℚ, p = 33 / 26244 :=
begin
  use 33 / 26244,
  sorry
end

end tetrahedrons_identical_probability_l367_367193


namespace farthest_vertex_label_l367_367154

-- The vertices and their labeling
def cube_faces : List (List Nat) := [
  [1, 2, 5, 8],
  [3, 4, 6, 7],
  [2, 4, 5, 7],
  [1, 3, 6, 8],
  [2, 3, 7, 8],
  [1, 4, 5, 6]
]

-- Define the cube vertices labels
def vertices : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

-- Statement of the problem in Lean 4
theorem farthest_vertex_label (h : true) : 
  ∃ v : Nat, v ∈ vertices ∧ ∀ face ∈ cube_faces, v ∉ face → v = 6 := 
sorry

end farthest_vertex_label_l367_367154


namespace triangle_similarity_proof_l367_367833

open Classical

/-- In Δ ABC, given that DE is parallel to AB, CD = 3 cm, DA = 9 cm, CE = 5 cm, and BC is twice the length of AB, 
    prove that AB = 10 cm. -/
theorem triangle_similarity_proof:
  ∀ (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
    (DE AB : Set (segment A B)) (DE_parallel_AB : parallel DE AB)
    (CD DA CE AB_ BC : ℝ),
    CD = 3 →
    DA = 9 →
    CE = 5 →
    BC = 2 * AB →
    ∃ (AB : ℝ), AB = 10 :=
by
  sorry

end triangle_similarity_proof_l367_367833


namespace sum_adjacent_to_six_l367_367861

theorem sum_adjacent_to_six :
  ∀ (table : fin 3 × fin 3 → ℕ),
    (∀ i j, table i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∃! i j, table i j = 1) ∧
    (∃! i j, table i j = 2) ∧
    (∃! i j, table i j = 3) ∧
    (∃! i j, table i j = 4) ∧
    (∃! i j, table i j = 5) → 
    (∀ i j, 
      table i j = 5 → 
        let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                       (if i < 2 then table (i+1, j) else 0) + 
                       (if j > 0 then table (i, j-1) else 0) + 
                       (if j < 2 then table (i, j+1) else 0)
        in adj_sum = 9) →
    (∃ i j, table i j = 6 ∧
      let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                     (if i < 2 then table (i+1, j) else 0) + 
                     (if j > 0 then table (i, j-1) else 0) + 
                     (if j < 2 then table (i, j+1) else 0) 
      in adj_sum = 29) := sorry

end sum_adjacent_to_six_l367_367861


namespace ME_tangent_to_circumcircle_of_AEF_l367_367482

noncomputable def triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

noncomputable def angle_bisector_foot (B C : Point) (ABC_triangle : triangle B C A) : Point := sorry

noncomputable def midpoint (B C : Point) : Point := sorry

theorem ME_tangent_to_circumcircle_of_AEF
  (A B C : Point)
  (ABC_triangle : triangle A B C)
  (E := angle_bisector_foot B A ABC_triangle)
  (F := angle_bisector_foot C A ABC_triangle)
  (M := midpoint B C) :
  tangent (line M E) (circumcircle A E F) := sorry

end ME_tangent_to_circumcircle_of_AEF_l367_367482


namespace max_factors_of_bn_l367_367561

theorem max_factors_of_bn (b n : ℕ) (hb : 0 < b ∧ b ≤ 20) (hn : 0 < n ∧ n ≤ 20) : 
  ∃ b n, b ≤ 20 ∧ n ≤ 20 ∧ ∀ k, (factors_count (b^n) = k → k ≤ 81) :=
begin
  sorry
end

end max_factors_of_bn_l367_367561


namespace find_D_coordinates_l367_367946

open EuclideanGeometry

noncomputable def A : Point := (4, 10)
noncomputable def B : Point := (2, 2)
noncomputable def C : Point := (6, 4)
noncomputable def midpoint (P Q : Point) : Point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def is_square (M N O P : Point) : Prop :=
  dist M N = dist N O ∧ dist N O = dist O P ∧ dist O P = dist P M ∧
  dist_diag_eq (M, O) (N, P)

theorem find_D_coordinates (D : Point) :
  (∃ D, is_square (midpoint A B) (midpoint B C) (midpoint C D) (midpoint D A)) →
  D.1 + D.2 = 12 :=
sorry

end find_D_coordinates_l367_367946


namespace average_cost_per_book_l367_367239

/-- Given: 
  batch of 350 books,
  total cost of $15.30, 
  additional delivery fee of $9.25.
  Prove that the average cost per book in cents, rounded to the nearest whole number, is 7 cents.
-/
theorem average_cost_per_book (
  batch_size : ℕ := 350,
  cost_per_batch : ℝ := 15.30,
  delivery_fee : ℝ := 9.25
) : 
  let total_cost_cents := (cost_per_batch + delivery_fee) * 100
  in round (total_cost_cents / batch_size) = 7 :=
by sorry

end average_cost_per_book_l367_367239


namespace molecular_weight_of_4_moles_C6H6_l367_367205

noncomputable def atomic_weight_C : real := 12.01
noncomputable def atomic_weight_H : real := 1.008
def molecular_formula_benzene : list (string × nat) := [("C", 6), ("H", 6)]
def moles_benzene : real := 4

theorem molecular_weight_of_4_moles_C6H6 :
  (moles_benzene * (6 * atomic_weight_C + 6 * atomic_weight_H) = 312.432) :=
by
  sorry

end molecular_weight_of_4_moles_C6H6_l367_367205


namespace monthly_sales_fraction_l367_367091

theorem monthly_sales_fraction (V S_D T : ℝ) 
  (h1 : S_D = 6 * V) 
  (h2 : S_D = 0.35294117647058826 * T) 
  : V = (1 / 17) * T :=
sorry

end monthly_sales_fraction_l367_367091


namespace quarters_per_jar_l367_367899

/-- Jenn has 5 jars full of quarters. Each jar can hold a certain number of quarters.
    The bike costs 180 dollars, and she will have 20 dollars left over after buying it.
    Prove that each jar can hold 160 quarters. -/
theorem quarters_per_jar (num_jars : ℕ) (cost_bike : ℕ) (left_over : ℕ)
  (quarters_per_dollar : ℕ) (total_quarters : ℕ) (quarters_per_jar : ℕ) :
  num_jars = 5 → cost_bike = 180 → left_over = 20 → quarters_per_dollar = 4 →
  total_quarters = ((cost_bike + left_over) * quarters_per_dollar) →
  quarters_per_jar = (total_quarters / num_jars) →
  quarters_per_jar = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end quarters_per_jar_l367_367899


namespace middle_school_survey_population_sample_l367_367571

theorem middle_school_survey_population_sample (total_students : ℕ) (surveyed_students : ℕ) : 
  total_students = 32000 → surveyed_students = 1600 →
  (¬ (total_students = 32000)) ∧ (surveyed_students = 1600) ∧ (¬ (surveyed_students = total_students)) ∧
  (¬ (surveyed_students > total_students)) :=
by
  intro h1 h2
  split
  {
    intro h3
    contradiction
  }
  split
  {
    exact h2
  }
  split
  {
    intro h3
    contradiction
  }
  {
    intro h3
    contradiction
  }
  sorry

end middle_school_survey_population_sample_l367_367571


namespace most_stable_performance_l367_367181

structure Shooter :=
(average_score : ℝ)
(variance : ℝ)

def A := Shooter.mk 8.9 0.45
def B := Shooter.mk 8.9 0.42
def C := Shooter.mk 8.9 0.51

theorem most_stable_performance : 
  B.variance < A.variance ∧ B.variance < C.variance :=
by
  sorry

end most_stable_performance_l367_367181


namespace virus_probability_odd_l367_367674

theorem virus_probability_odd (m n : ℕ) (hm : 1 ≤ m ∧ m ≤ 7) (hn : 1 ≤ n ∧ n ≤ 9) :
  (nat.card {k | k ∈ finset.range(8) \ {0} ∧ k % 2 ≠ 0}.card * nat.card {k | k ∈ finset.range(10) \ {0} ∧ k % 2 ≠ 0}.card : ℚ) 
  / (finset.range(8) \ {0}).card * (finset.range(10) \ {0}).card = 20 / 63 := sorry

end virus_probability_odd_l367_367674


namespace expression_evaluation_l367_367314

theorem expression_evaluation :
  (\dfrac{\frac{1}{4} - \frac{1}{5}}{\frac{2}{5} - \frac{1}{4}} + \dfrac{\frac{1}{6}}{\frac{1}{3} - \frac{1}{4}} = \frac{7}{3}) :=
by
  sorry

end expression_evaluation_l367_367314


namespace new_person_weight_l367_367533

theorem new_person_weight (avg_weight_increase : ℝ) (old_weight new_weight : ℝ) (n : ℕ)
    (weight_increase_per_person : avg_weight_increase = 3.5)
    (number_of_persons : n = 8)
    (replaced_person_weight : old_weight = 62) :
    new_weight = 90 :=
by
  sorry

end new_person_weight_l367_367533


namespace total_years_l367_367185

variable (T D : ℕ)
variable (Tom_years : T = 50)
variable (Devin_years : D = 25 - 5)

theorem total_years (hT : T = 50) (hD : D = 25 - 5) : T + D = 70 := by
  sorry

end total_years_l367_367185


namespace Haleigh_can_make_3_candles_l367_367813

variable (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ)

def wax_leftover (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ) : ℝ := 
  n20 * w20 + n5 * w5 + n1 * w1 

theorem Haleigh_can_make_3_candles :
  ∀ (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ), 
  n20 = 5 →
  w20 = 2 →
  n5 = 5 →
  w5 = 0.5 →
  n1 = 25 →
  w1 = 0.1 →
  oz10 = 10 →
  (wax_leftover n20 n5 n1 w20 w5 w1 oz10) / 5 = 3 := 
by
  intros n20 n5 n1 w20 w5 w1 oz10 hn20 hw20 hn5 hw5 hn1 hw1 hoz10
  rw [hn20, hw20, hn5, hw5, hn1, hw1, hoz10]
  sorry

end Haleigh_can_make_3_candles_l367_367813


namespace triangle_side_length_l367_367081

noncomputable def length_AB (AC BC : ℝ) (angle_sum : ℝ) : ℝ := 
  if (AC = 1) ∧ (BC = 3) ∧ (angle_sum = 60) then 2 * Real.sqrt 13 else 0

theorem triangle_side_length (AC BC : ℝ) (angle_sum : ℝ) (hAC : AC = 1) (hBC : BC = 3) (hAngleSum : angle_sum = 60) : 
  length_AB AC BC angle_sum = 2 * Real.sqrt 13 :=
by 
  rw [length_AB]
  simp [hAC, hBC, hAngleSum]
  sorry

end triangle_side_length_l367_367081


namespace find_slope_l367_367730

-- Define the line equation
def line (x y : ℝ) : Prop :=
  4 * x + 7 * y = 28

-- Define the slope of the line to be proved
def slope (m : ℝ) : Prop :=
  m = -4 / 7

-- State the theorem
theorem find_slope : ∃ m : ℝ, slope m ∧ (∀ x y : ℝ, line x y → y = m * x + 4) :=
by
  sorry

end find_slope_l367_367730


namespace problem_l367_367906

variables {d : Line} {M N : Point} 
variables {α β γ δ : Circle} {A B C D : Point}
variables {ω : Circle}

-- Hypotheses
def conditions : Prop :=
  (Tangent α d) ∧ (Tangent β d) ∧ (Tangent γ d) ∧ (Tangent δ d) ∧
  (Tangent ω α) ∧ (Tangent ω β) ∧ (Tangent ω γ) ∧ (Tangent ω δ) ∧
  (ExternallyTangent α β M) ∧ (ExternallyTangent γ δ N) ∧
  (AC_on_same_half_plane : ∃ P : Point, P ∈ (half_plane A C d))

theorem problem (h : conditions) : 
  Concurrent_or_Parallel (AC_line A C) (BD_line B D) (MN_line M N) :=
sorry

end problem_l367_367906


namespace extra_pieces_correct_l367_367128

def pieces_per_package : ℕ := 7
def number_of_packages : ℕ := 5
def total_pieces : ℕ := 41

theorem extra_pieces_correct : total_pieces - (number_of_packages * pieces_per_package) = 6 :=
by
  sorry

end extra_pieces_correct_l367_367128


namespace nested_sqrt_solution_l367_367137

noncomputable def solve_nested_sqrt_eq (n m : ℕ) : Prop :=
  ∃ k : ℕ, k = 1964 ∧ (∀ i : ℕ, i < k → A_nesting(n, i + 1) = m) → (n, m) = (0, 0)

def A_nesting (x : ℕ) : ℕ → ℕ
| 0 := sqrt x
| (n + 1) := sqrt (x + A_nesting x n)

theorem nested_sqrt_solution : ∀ n m : ℕ, solve_nested_sqrt_eq n m :=
  sorry

end nested_sqrt_solution_l367_367137


namespace train_crosses_pole_in_expected_time_l367_367083

noncomputable def train_crossing_time
  (length : ℝ)
  (speed_kmh : ℝ)
  (conversion_factor : ℝ := 1000 / 3600) : ℝ :=
  let speed_ms := speed_kmh * conversion_factor
  in length / speed_ms

theorem train_crosses_pole_in_expected_time
  (length : ℝ := 140)
  (speed_kmh : ℝ := 210)
  (expected_time : ℝ := 2.4)
  (conversion_factor : ℝ := 1000 / 3600) :
  train_crossing_time length speed_kmh = expected_time :=
by
  sorry

end train_crosses_pole_in_expected_time_l367_367083


namespace imo_1971_q3_l367_367951

noncomputable def euler_totient (n : ℕ) : ℕ := sorry

theorem imo_1971_q3 :
    ∃ (n : ℕ → ℕ), (∀ i j : ℕ, i ≠ j → n i ≥ 2 ∧ n j ≥ 2 ∧ gcd (2 ^ (n i) - 3) (2 ^ (n j) - 3) = 1) :=
begin
  sorry
end

end imo_1971_q3_l367_367951


namespace probability_of_solution_in_set_l367_367618

def is_solution (x : ℝ) : Prop :=
  (x - 3 = 0) ∨ (x + 14 = 0) ∨ (2 * x + 5 = 0)

def numbers_set : set ℝ := 
  {-10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10}

def solutions_in_set : set ℝ :=
  {x | x ∈ numbers_set ∧ is_solution x}

lemma count_solutions : card solutions_in_set = 1 := by sorry

lemma total_numbers : card numbers_set = 12 := by sorry

theorem probability_of_solution_in_set :
  (card solutions_in_set : ℝ) / (card numbers_set : ℝ) = 1 / 12 :=
by
  rw [count_solutions, total_numbers]
  norm_num
  sorry

end probability_of_solution_in_set_l367_367618


namespace arithmetic_not_geometric_sequence_l367_367396

theorem arithmetic_not_geometric_sequence (a b c : ℝ) (h1 : 2 ^ a = 3) (h2 : 2 ^ b = 6) (h3 : 2 ^ c = 12) :
  (a + c = 2 * b) ∧ (a * c ≠ b ^ 2) :=
by
  sorry

end arithmetic_not_geometric_sequence_l367_367396


namespace n_times_s_l367_367492

noncomputable def f (a : ℝ) (f : ℝ → ℝ) := ∀ x y : ℝ, f((x - y)^2) = f(x)^2 - 2 * a * x * f(y) + a * y^2

theorem n_times_s (f : ℝ → ℝ) (a : ℝ) (h : f (a) f) (ha : 0 < a) : 
  let n := 2 
  let s := 2 * a + 1 
  have ns_eq : n * s = 4 * a + 2 := by sorry 

end n_times_s_l367_367492


namespace sum_leq_two_l367_367126

open Classical

theorem sum_leq_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 + b^3 = 2) : a + b ≤ 2 :=
by
  sorry

end sum_leq_two_l367_367126


namespace problem_ellipse_equation_problem_triangle_area_l367_367387

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = sqrt 2 ∧ b = 1 ∧
  let C := λ (x y : ℝ), (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 in
  C x y ↔ (x ^ 2) / 2 + y ^ 2 = 1

noncomputable def triangle_area : Prop :=
  let a := sqrt 2 in
  let b := 1 in
  let c := sqrt (a ^ 2 - b ^ 2) in
  let F1 := (-c, 0) in
  let F2 := (c, 0) in
  let l := λ x, sqrt 3 * (x + c) in
  ∃ (x1 x2 : ℝ), 
    ((x1^2)/ 2 + (l x1)^2 = 1) ∧ ((x2^2)/ 2 + (l x2)^2 = 1) ∧ 
    let M := (x1, l x1) in 
    let N := (x2, l x2) in
    let MN := real.sqrt (7 * x1 ^ 2 + 12 * x1 + 4) in
    let d := real.sqrt 3 / 2 in
    area_of_triangle ((0, 0), M, N) = (1/2) * MN * d ∧ 
    area_of_triangle ((0, 0), M, N) = 2 * real.sqrt 6 / 7

theorem problem_ellipse_equation : ellipse_equation :=
by {
  -- details of the proof would go here
  sorry
}

theorem problem_triangle_area : triangle_area :=
by {
  -- details of the proof would go here
  sorry
}

end problem_ellipse_equation_problem_triangle_area_l367_367387


namespace sum_adjacent_to_6_is_29_l367_367855
-- Import the Mathlib library for the necessary tools and functions

/--
  In a 3x3 table filled with numbers from 1 to 9 such that each number appears exactly once, 
  with conditions: 
    * (1, 1) contains 1, (3, 1) contains 2, (1, 3) contains 3, (3, 3) contains 4,
    * The sum of the numbers in the cells adjacent to the cell containing 5 is 9,
  Prove that the sum of the numbers in the cells adjacent to the cell containing 6 is 29.
-/
theorem sum_adjacent_to_6_is_29 
  (table : Fin 3 → Fin 3 → Fin 9)
  (H_uniqueness : ∀ i j k l, (table i j = table k l) → (i = k ∧ j = l))
  (H_valid_entries : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)
  (H_initial_positions : table 0 0 = 1 ∧ table 2 0 = 2 ∧ table 0 2 = 3 ∧ table 2 2 = 4)
  (H_sum_adj_to_5 : ∃ (i j : Fin 3), table i j = 5 ∧ 
                      ((i > 0 ∧ table (i-1) j +
                       (i < 2 ∧ table (i+1) j) +
                       (j > 0 ∧ table i (j-1)) +
                       (j < 2 ∧ table i (j+1))) = 9)) :
  ∃ i j, table i j = 6 ∧
  (i > 0 ∧ table (i-1) j +
   (i < 2 ∧ table (i+1) j) +
   (j > 0 ∧ table i (j-1)) +
   (j < 2 ∧ table i (j+1))) = 29 := sorry

end sum_adjacent_to_6_is_29_l367_367855


namespace sum_adjacent_cells_of_6_is_29_l367_367847

theorem sum_adjacent_cells_of_6_is_29 (table : Fin 3 × Fin 3 → ℕ)
  (uniq : Function.Injective table)
  (range : ∀ x, 1 ≤ table x ∧ table x ≤ 9)
  (pos_1 : table ⟨0, 0⟩ = 1)
  (pos_2 : table ⟨2, 0⟩ = 2)
  (pos_3 : table ⟨0, 2⟩ = 3)
  (pos_4 : table ⟨2, 2⟩ = 4)
  (adj_5 : (∑ i in ({⟨1, 0⟩, ⟨1, 2⟩, ⟨0, 1⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 9) :
  (∑ i in ({⟨0, 1⟩, ⟨1, 0⟩, ⟨1, 2⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 29 :=
by
  sorry

end sum_adjacent_cells_of_6_is_29_l367_367847


namespace least_three_digit_multiple_l367_367590

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l367_367590


namespace equilateral_triangle_side_length_l367_367176

theorem equilateral_triangle_side_length (s : ℕ) (h : 3 * s = 78) : s = 26 :=
begin
  sorry
end

end equilateral_triangle_side_length_l367_367176


namespace equal_area_of_aknmas_and_abc_l367_367884

-- Definitions for our problem context
variables {A B C L N K M : Type}
variables [fintype A] [fintype B] [fintype C]
variables {a b c : ℝ} -- lengths of triangle sides
variables (area : Type) [field area]

-- Assume our points are in ℝ^2 for simplicity
variables (triangle : (ℝ × ℝ) → ℝ)
variables (bisector : ℝ → ℝ × ℝ)
variables (circumcircle : ℝ → ℝ)
variables (perpendicular_from_L_to_AB : ℝ → ℝ × ℝ)
variables (perpendicular_from_L_to_AC : ℝ → ℝ × ℝ)
variables (acute_angle : ℝ)

-- The theorem stating the area equality of quadrilateral AKNM and triangle ABC
theorem equal_area_of_aknmas_and_abc 
  (h : acute_angle > 0)
  (h₂ : ∀ x, x > 0)
  (triangle_acute_angled : ∀ ⦃A B C⦄, acute_angle)
  (interior_bisector_meets_BC_L : bisector A = (B, C))
  (interior_bisector_meets_circumcircle_N : ∀ x, circumcircle A = bisector x)
  (perpendicular_drawn_to_AB_at_K : ∀ l, perpendicular_from_L_to_AB l = K)
  (perpendicular_drawn_to_AC_at_M : ∀ l, perpendicular_from_L_to_AC l = M) :
  area (quadrilateral_area A K N M) = area (triangle_area A B C) :=
begin
  sorry
end

end equal_area_of_aknmas_and_abc_l367_367884


namespace evaluate_expression_l367_367697

theorem evaluate_expression : 
  let a := 3
  let b := 4
  (a^b)^a - (b^a)^b = -16245775 := 
by 
  sorry

end evaluate_expression_l367_367697


namespace diameter_of_pool_l367_367660

-- Define the conditions and the expected result
def pool_volume : ℝ := 16964.600329384884
def pool_depth : ℝ := 6
def approximate_diameter : ℝ := 59.99488

-- Define the proof statement
theorem diameter_of_pool :
  2 * real.sqrt (pool_volume / (real.pi * pool_depth)) ≈ approximate_diameter := sorry

end diameter_of_pool_l367_367660


namespace solve_equation_l367_367982

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l367_367982


namespace smallest_x_for_convex_distortion_l367_367917

def is_distortion (H : list (ℝ × ℝ)) (H' : list (ℝ × ℝ)) : Prop :=
  H.length = 6 ∧ H'.length = 6 ∧
  ∀ i, 0 ≤ i ∧ i < 6 → dist (H.nth_le i sorry) (H'.nth_le i sorry) < 1

def is_convex (V : list (ℝ × ℝ)) : Prop :=
  ∀ i j k, 0 ≤ i ∧ i < 6 → 0 ≤ j ∧ j < 6 → 0 ≤ k ∧ k < 6 → j ≠ i → k ≠ i → k ≠ j →
  let v_i := V.nth_le i sorry; let v_j := V.nth_le j sorry; let v_k := V.nth_le k sorry in
  (v_k.2 - v_i.2) * (v_j.1 - v_i.1) ≠ (v_j.2 - v_i.2) * (v_k.1 - v_i.1)

def regular_hexagon (x : ℝ) : list (ℝ × ℝ) := [
  (0, 0), (x, 0), (1.5 * x, sqrt 3 / 2 * x), (x, sqrt 3 * x), (0, sqrt 3 * x), (-0.5 * x, sqrt 3 / 2 * x)
]

theorem smallest_x_for_convex_distortion :
  ∀ H : list (ℝ × ℝ), H = regular_hexagon 4 → ∀ H', is_distortion H H' → is_convex H' :=
begin
  intros H H_eq H' H_distortion,
  sorry
end

end smallest_x_for_convex_distortion_l367_367917


namespace maria_trip_distance_l367_367222

theorem maria_trip_distance (D : ℝ) 
  (h1 : D / 2 + D / 8 + 210 = D) 
  (h2 : D / 2 > 0) 
  (h3 : 210 > 0) : 
  D = 560 :=
by
  have h4 : (3 * D) / 8 = 210, from calc
    (3 * D) / 8 = D / 2 - (D / 8) + 210 - 210 : by ring
            ... = 210                          : by linarith [h1],

  sorry

end maria_trip_distance_l367_367222


namespace least_three_digit_with_factors_correct_l367_367592

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end least_three_digit_with_factors_correct_l367_367592


namespace slope_of_line_l367_367734

theorem slope_of_line : ∀ (x y : ℝ), (4 * x + 7 * y = 28) → ∃ m b : ℝ, (-4 / 7 = m) ∧ (y = m * x + b) :=
by
  intros x y h
  use [-4 / 7, 4]
  split
  · refl
  sorry

end slope_of_line_l367_367734


namespace volumes_not_equal_l367_367191

theorem volumes_not_equal :
  let r₀ := 5
  let h₀ := 7
  let r₁ := r₀ + 3
  let h₁ := h₀ + 4
  let Volume₁ := π * r₁^2 * h₀
  let Volume₂ := π * r₀^2 * h₁
  Volume₁ ≠ Volume₂ :=
by
  -- Definitions
  let r₀ := 5
  let h₀ := 7
  let r₁ := r₀ + 3
  let h₁ := h₀ + 4
  let Volume₁ := π * r₁^2 * h₀
  let Volume₂ := π * r₀^2 * h₁
  -- Proof
  have : Volume₁ = π * 8^2 * 7 := by 
    simp [Volume₁, r₁]
    sorry  -- placeholder for simp result
  have : Volume₂ = π * 5^2 * 11 := by 
    simp [Volume₂, h₁]
    sorry  -- placeholder for simp result
  have h1 : Volume₁ = 448 * π := by 
    simp 
    sorry  -- placeholder for simp calculation
  have h2 : Volume₂ = 275 * π := by 
    simp 
    sorry  -- placeholder for simp calculation
  show 448 * π ≠ 275 * π, from
    by linarith

end volumes_not_equal_l367_367191


namespace time_parents_called_l367_367582

-- Define the conditions
def vanya_mow_rate : ℝ := 1 / 5
def petya_mow_rate : ℝ := 1 / 6
def unmown_fraction : ℝ := 1 / 10
def initial_time : ℝ := 11

-- Define the statement of the theorem
theorem time_parents_called : 
  ∃ t : ℝ, 
    t - 1 ≥ 0 ∧ t - 2 ≥ 0 ∧
    vanya_mow_rate * (t - 2) + petya_mow_rate * (t - 1) = 1 - unmown_fraction ∧
    initial_time + t = 15 := sorry

end time_parents_called_l367_367582


namespace distance_to_other_focus_l367_367673

def hyperbola_eq (x y : ℝ) : Prop :=
  (y^2 / 64) - (x^2 / 16) = 1

def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def focus1 : ℝ × ℝ := (4, 0) -- assuming one of the foci at (4, 0) without loss of generality

def focus2 : ℝ × ℝ := (-4, 0) -- accordingly the other foci would be at (-4, 0)

noncomputable def point_on_hyperbola (x y : ℝ) (h : hyperbola_eq x y) : ℝ × ℝ :=
  (x, y)

theorem distance_to_other_focus {x y : ℝ} (h : hyperbola_eq x y)
  (d : ℝ) (h_d : distance (point_on_hyperbola x y h) focus1 = 4) :
  distance (point_on_hyperbola x y h) focus2 = 20 :=
sorry

end distance_to_other_focus_l367_367673


namespace interior_diagonals_sum_l367_367645

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 112)
  (h2 : 4 * (a + b + c) = 60) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := 
by 
  sorry

end interior_diagonals_sum_l367_367645


namespace part1_part2_l367_367762

noncomputable def ellipse_eq (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def problem1 (a b : ℝ) (h : a > b ∧ b > 0) :
  Prop := ∀ x₀ y₀ x₁ y₁ : ℝ,
  ((x₀^2 / a^2) + (y₀^2 / b^2) = 1) →
  ((x₁^2 / a^2) + (y₁^2 / b^2) = 1) →
  ((y₀ - y₁) / (x₀ - x₁)) * ((y₀ + y₁) / (x₀ + x₁)) = - (b^2 / a^2)

noncomputable def problem2 (a b : ℝ) (h : a > b ∧ b > 0)
  (AB MN : ℝ) : 
  Prop := 
  ∀ k₁ k₂ : ℝ, 
  (2 * a * AB = MN^2) →
  (AB = (2 * a * b^2 * (1 + k₁^2)) / (b^2 + a^2 * k₁^2)) →
  (MN = 2 * a * b * (√(1 + k₂^2)) / √(b^2 + a^2 * k₂^2)) →
  k₁^2 = k₂^2 ∨ (MN) ⧸ 0

theorem part1 (a b : ℝ) (h : a > b ∧ b > 0) : problem1 a b h := sorry

theorem part2 (a b : ℝ) (h : a > b ∧ b > 0) (AB MN : ℝ) : problem2 a b h AB MN := sorry

end part1_part2_l367_367762


namespace area_of_region_max_volume_of_solid_l367_367092

open Real

theorem area_of_region (a : ℝ) (h : 0 < a ∧ a < 1) :
  let A := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ (p.1 / a)^0.5 + (p.2 / (1 - a))^0.5 ≤ 1} in
  (let area := a * (1 - a) / 6
   in true) :=
  sorry

theorem max_volume_of_solid (a : ℝ) (h : 0 < a ∧ a < 1) :
  let V_max := (4 * π) / 405
  in
  (let eq_a := a = 1/3
    in true) :=
  sorry

end area_of_region_max_volume_of_solid_l367_367092


namespace sum_adjacent_to_six_l367_367860

theorem sum_adjacent_to_six :
  ∀ (table : fin 3 × fin 3 → ℕ),
    (∀ i j, table i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∃! i j, table i j = 1) ∧
    (∃! i j, table i j = 2) ∧
    (∃! i j, table i j = 3) ∧
    (∃! i j, table i j = 4) ∧
    (∃! i j, table i j = 5) → 
    (∀ i j, 
      table i j = 5 → 
        let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                       (if i < 2 then table (i+1, j) else 0) + 
                       (if j > 0 then table (i, j-1) else 0) + 
                       (if j < 2 then table (i, j+1) else 0)
        in adj_sum = 9) →
    (∃ i j, table i j = 6 ∧
      let adj_sum := (if i > 0 then table (i-1, j) else 0) + 
                     (if i < 2 then table (i+1, j) else 0) + 
                     (if j > 0 then table (i, j-1) else 0) + 
                     (if j < 2 then table (i, j+1) else 0) 
      in adj_sum = 29) := sorry

end sum_adjacent_to_six_l367_367860


namespace least_three_digit_multiple_l367_367588

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l367_367588


namespace problem_1_problem_2_l367_367671

theorem problem_1 : ((1 / 3 - 3 / 4 + 5 / 6) / (1 / 12)) = 5 := 
  sorry

theorem problem_2 : ((-1 : ℤ) ^ 2023 + |(1 : ℝ) - 0.5| * (-4 : ℝ) ^ 2) = 7 := 
  sorry

end problem_1_problem_2_l367_367671


namespace number_of_increasing_8_digit_numbers_mod_1000_l367_367910

theorem number_of_increasing_8_digit_numbers_mod_1000 : 
  let M := (Nat.choose 15 7)
  in  (M % 1000) = 435 :=
by
  let M := 6435
  show (M % 1000) = 435
  sorry

end number_of_increasing_8_digit_numbers_mod_1000_l367_367910


namespace cost_of_apples_is_2_l367_367711

variable (A : ℝ)

def cost_of_apples (A : ℝ) : ℝ := 5 * A
def cost_of_sugar (A : ℝ) : ℝ := 3 * (A - 1)
def cost_of_walnuts : ℝ := 0.5 * 6
def total_cost (A : ℝ) : ℝ := cost_of_apples A + cost_of_sugar A + cost_of_walnuts

theorem cost_of_apples_is_2 (A : ℝ) (h : total_cost A = 16) : A = 2 := 
by 
  sorry

end cost_of_apples_is_2_l367_367711


namespace cubic_vs_square_ratio_l367_367789

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l367_367789


namespace angle_between_a_c_pi_over_6_min_value_f_in_range_l367_367811

def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def b (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.cos x)
def c : ℝ × ℝ := (-1, 0)

def f (x : ℝ) : ℝ := 2 * ((a x).1 * (b x).1 + (a x).2 * (b x).2) + 1

theorem angle_between_a_c_pi_over_6 :
  ∀ x = (Real.pi / 6), let a_x = a x in
  let c_x = c in
  a_x.1 = (Real.sqrt 3) / 2 ∧ a_x.2 = 1 / 2 ∧
  (c_x.1 * a_x.1 + c_x.2 * a_x.2) / (Real.sqrt ((a_x.1)^2 + (a_x.2)^2) * Real.sqrt ((c_x.1)^2 + (c_x.2)^2)) = - (Real.sqrt 3) / 2 :=
sorry

theorem min_value_f_in_range :
  ∃ x, x ∈ Set.Icc (Real.pi / 2) (9 * Real.pi / 8) ∧ f x = -Real.sqrt 2 :=
sorry

end angle_between_a_c_pi_over_6_min_value_f_in_range_l367_367811


namespace right_angled_triangle_exists_l367_367214

theorem right_angled_triangle_exists :
  (¬ (2^2 + 3^2 = 4^2)) ∧
  ((3^2 + 4^2 = 5^2)) ∧
  (¬ (4^2 + 5^2 = 6^2)) ∧
  (¬ (5^2 + 6^2 = 7^2)) :=
by {
  -- Set A: 2, 3, 4
  have hA : ¬ (2^2 + 3^2 = 4^2),
  { simp, norm_num, },

  -- Set B: 3, 4, 5
  have hB : (3^2 + 4^2 = 5^2),
  { simp, norm_num, },

  -- Set C: 4, 5, 6
  have hC : ¬ (4^2 + 5^2 = 6^2),
  { simp, norm_num, },

  -- Set D: 5, 6, 7
  have hD : ¬ (5^2 + 6^2 = 7^2),
  { simp, norm_num, },

  exact ⟨hA, hB, hC, hD⟩,
}

end right_angled_triangle_exists_l367_367214


namespace value_of_a_l367_367803

def hyperbolaFociSharedEllipse : Prop :=
  ∃ a > 0, 
    (∃ c h k : ℝ, c = 3 ∧ (h, k) = (3, 0) ∨ (h, k) = (-3, 0)) ∧ 
    ∃ x y : ℝ, ((x^2) / 4) - ((y^2) / 5) = 1 ∧ ((x^2) / (a^2)) + ((y^2) / 16) = 1

theorem value_of_a : ∃ a > 0, hyperbolaFociSharedEllipse ∧ a = 5 :=
by
  sorry

end value_of_a_l367_367803


namespace perimeter_of_shape_l367_367648

-- Define the shape being described
def shape := { s : ℕ // s = 4 }

-- Define the total area of the shape
def total_area (s : shape) : ℕ := 196

def side_length_of_square (area : ℕ) : ℕ :=
  Int.sqrt (area / 4)

def perimeter (length : ℕ) : ℕ :=
  5 * length + 8 * length

theorem perimeter_of_shape :
  ∀ s : shape, perimeter (side_length_of_square (total_area s)) = 91 := by
  intro s
  sorry

end perimeter_of_shape_l367_367648


namespace school_A_win_prob_expectation_X_is_13_l367_367959

-- Define the probabilities of school A winning individual events
def pA_event1 : ℝ := 0.5
def pA_event2 : ℝ := 0.4
def pA_event3 : ℝ := 0.8

-- Define the probability of school A winning the championship
def pA_win_championship : ℝ :=
  (pA_event1 * pA_event2 * pA_event3) +
  (pA_event1 * (1 - pA_event2) * pA_event3) +
  (pA_event1 * pA_event2 * (1 - pA_event3)) +
  ((1 - pA_event1) * pA_event2 * pA_event3)

-- Proof statement for the probability of school A winning the championship
theorem school_A_win_prob : pA_win_championship = 0.6 := sorry

-- Define the distribution and expectation for school B's total score
def X_prob : ℝ → ℝ
| 0  := (1 - pA_event1) * (1 - pA_event2) * (1 - pA_event3)
| 10 := pA_event1 * (1 - pA_event2) * (1 - pA_event3) +
        (1 - pA_event1) * pA_event2 * (1 - pA_event3) +
        (1 - pA_event1) * (1 - pA_event2) * pA_event3
| 20 := pA_event1 * pA_event2 * (1 - pA_event3) +
        pA_event1 * (1 - pA_event2) * pA_event3 +
        (1 - pA_event1) * pA_event2 * pA_event3
| 30 := pA_event1 * pA_event2 * pA_event3
| _  := 0

def expected_X : ℝ :=
  0 * X_prob 0 +
  10 * X_prob 10 +
  20 * X_prob 20 +
  30 * X_prob 30

-- Proof statement for the expectation of school B's total score
theorem expectation_X_is_13 : expected_X = 13 := sorry

end school_A_win_prob_expectation_X_is_13_l367_367959


namespace school_A_win_prob_expectation_X_is_13_l367_367958

-- Define the probabilities of school A winning individual events
def pA_event1 : ℝ := 0.5
def pA_event2 : ℝ := 0.4
def pA_event3 : ℝ := 0.8

-- Define the probability of school A winning the championship
def pA_win_championship : ℝ :=
  (pA_event1 * pA_event2 * pA_event3) +
  (pA_event1 * (1 - pA_event2) * pA_event3) +
  (pA_event1 * pA_event2 * (1 - pA_event3)) +
  ((1 - pA_event1) * pA_event2 * pA_event3)

-- Proof statement for the probability of school A winning the championship
theorem school_A_win_prob : pA_win_championship = 0.6 := sorry

-- Define the distribution and expectation for school B's total score
def X_prob : ℝ → ℝ
| 0  := (1 - pA_event1) * (1 - pA_event2) * (1 - pA_event3)
| 10 := pA_event1 * (1 - pA_event2) * (1 - pA_event3) +
        (1 - pA_event1) * pA_event2 * (1 - pA_event3) +
        (1 - pA_event1) * (1 - pA_event2) * pA_event3
| 20 := pA_event1 * pA_event2 * (1 - pA_event3) +
        pA_event1 * (1 - pA_event2) * pA_event3 +
        (1 - pA_event1) * pA_event2 * pA_event3
| 30 := pA_event1 * pA_event2 * pA_event3
| _  := 0

def expected_X : ℝ :=
  0 * X_prob 0 +
  10 * X_prob 10 +
  20 * X_prob 20 +
  30 * X_prob 30

-- Proof statement for the expectation of school B's total score
theorem expectation_X_is_13 : expected_X = 13 := sorry

end school_A_win_prob_expectation_X_is_13_l367_367958


namespace ellipse_equation_fixed_points_l367_367158

-- Define the parameters of the ellipse
variables (a b t : ℝ)
-- Given conditions
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom eccentricity : b^2 + t^2 = a^2 ∧ t / a = 1 / 2

-- Define the problem
theorem ellipse_equation 
  (eccentricity_eq_half : eccentricity)
  : a = 2 * t ∧ b = sqrt(3) * t → (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)) := sorry 

-- Assuming conditions for the line and intersection points
variables (x1 y1 x2 y2 : ℝ)
variables (P Q : ℝ × ℝ)

axioms 
  (line_eq : ∀ y, x1 = t * y + 1)
  (intersections : x1^2 / 4 + y1^2 / 3 = 1 ∧ x2^2 / 4 + y2^2 / 3 = 1)
  (P_coords : P.fst = 4 ∧ P.snd = 6 * y1 / (x1 + 2))
  (Q_coords : Q.fst = 4 ∧ Q.snd = 6 * y2 / (x2 + 2))

-- Prove the circle with diameter PQ passes through fixed points
theorem fixed_points
  (circle_condition : ∀ (m n : ℝ), (4 - m)^2 + (P.snd - n) * (Q.snd - n) = 0) 
  : (P = (4, 6 * y1 / (x1 + 2)) ∧ Q = (4, 6 * y2 / (x2 + 2))) → ((m = 1 ∧ n = 0) ∨ (m = 7 ∧ n = 0)) := sorry

end ellipse_equation_fixed_points_l367_367158


namespace area_A_l367_367989

-- Given conditions
def triangle_area (A B C : ℝ) : Prop :=
  ∃ (α β γ : ℝ), α + β + γ = π ∧ sin α * sin β * sin γ ≠ 0 ∧
  A = AB * sin γ ∧ B = BC * sin α ∧ C = CA * sin β

-- Point definitions
def point_on_ray (A B A' : ℝ) : Prop := B ≤ A' ∧ A' = k * A

variables (A B C A' B' C' : ℝ) (k₁ k₂ k₃ : ℝ)
hypothesis
  h₁ : triangle_area A B C
  h₂ : point_on_ray A B B'
  h₃ : point_on_ray B C C'
  h₄ : point_on_ray C A A'
  h₅ : k₁ = 1 ∧ k₂ = 2 ∧ k₃ = 3

-- Proof target
theorem area_A'B'C' : triangle_area A' B' C' = 18 := by sorry

end area_A_l367_367989


namespace largest_palindrome_divisible_by_127_l367_367723

-- Definitions
def is_palindrome (n : Nat) : Prop :=
  let s := n.repr
  s = s.reverse

def largest_5_digit_palindrome_divisible_by_127 : Nat :=
  99399

-- Proof Problem Statement
theorem largest_palindrome_divisible_by_127 : ∀ n : Nat,
  is_palindrome n ∧ n % 127 = 0 → n ≤ 99999 → largest_5_digit_palindrome_divisible_by_127 = 99399 :=
by sorry

end largest_palindrome_divisible_by_127_l367_367723


namespace min_students_in_class_l367_367454

-- Define the conditions
variables (b g : ℕ) -- number of boys and girls
variable (h1 : 3 * b = 4 * (2 * g)) -- Equal number of boys and girls passed the test

-- Define the desired minimum number of students
def min_students : ℕ := 17

-- The theorem which asserts that the total number of students in the class is at least 17
theorem min_students_in_class (b g : ℕ) (h1 : 3 * b = 4 * (2 * g)) : (b + g) ≥ min_students := 
sorry

end min_students_in_class_l367_367454


namespace inequality_analysis_l367_367675

noncomputable def condition1 (p q r s t u : ℝ) := p^2 < s^2
noncomputable def condition2 (p q r s t u : ℝ) := q^2 < t^2
noncomputable def condition3 (p q r s t u : ℝ) := r^2 < u^2

theorem inequality_analysis (p q r s t u : ℝ) 
  (h1: condition1 p q r s t u) 
  (h2: condition2 p q r s t u) 
  (h3: condition3 p q r s t u) : 
  ¬ (pq + qr + rp > st + tu + us) ∧ (p^2q^2 + q^2r^2 + r^2p^2 < s^2t^2 + t^2u^2 + u^2s^2) :=
by
  sorry

end inequality_analysis_l367_367675


namespace number_of_routes_from_P_to_Q_is_3_l367_367581

-- Definitions of the nodes and paths
inductive Node
| P | Q | R | S | T | U | V
deriving DecidableEq, Repr

-- Definition of paths between nodes based on given conditions
def leads_to : Node → Node → Prop
| Node.P, Node.R => True
| Node.P, Node.S => True
| Node.R, Node.T => True
| Node.R, Node.U => True
| Node.S, Node.Q => True
| Node.T, Node.Q => True
| Node.U, Node.V => True
| Node.V, Node.Q => True
| _, _ => False

-- Proof statement: the number of different routes from P to Q
theorem number_of_routes_from_P_to_Q_is_3 : 
  ∃ (n : ℕ), n = 3 ∧ (∀ (route_count : ℕ), route_count = n → 
  ((leads_to Node.P Node.R ∧ leads_to Node.R Node.T ∧ leads_to Node.T Node.Q) ∨ 
   (leads_to Node.P Node.R ∧ leads_to Node.R Node.U ∧ leads_to Node.U Node.V ∧ leads_to Node.V Node.Q) ∨
   (leads_to Node.P Node.S ∧ leads_to Node.S Node.Q))) :=
by
  -- Placeholder proof
  sorry

end number_of_routes_from_P_to_Q_is_3_l367_367581


namespace find_distance_between_parallel_sides_l367_367335

noncomputable def distance_between_parallel_sides (area a b : ℝ) : ℝ :=
  2 * area / (a + b)

theorem find_distance_between_parallel_sides :
  distance_between_parallel_sides 304 20 18 = 16 :=
begin
  -- The proof is omitted as per the instructions
  sorry
end

end find_distance_between_parallel_sides_l367_367335


namespace six_people_six_chairs_l367_367062

/-- The number of ways six people can sit in a row of eight chairs if two specific chairs (specifically chairs 3 and 6) cannot be used simultaneously is 18720. -/
theorem six_people_six_chairs : 
  let chairs := 8
  let unusable_chair_pairs := [(3, 6)]
  ∀ (people : ℕ), people = 6 → 
  (∏ x in (finset.range chairs).powerset.filter (λ s, s.card = people ∧ 
    (unusable_chair_pairs.forall (λ (p : ℕ × ℕ), 
    ¬(s.member p.fst ∧ s.member p.snd)))), 
    people.factorial) = 18720 := 
by 
  intro chairs unusable_chair_pairs people h
  have h1: chairs = 8 := rfl
  have h2: unusable_chair_pairs = [(3, 6)] := rfl
  have h3: people = 6 := h
  sorry

end six_people_six_chairs_l367_367062


namespace triangle_side_lengths_l367_367637

theorem triangle_side_lengths (r : ℝ) (AC BC AB : ℝ) (y : ℝ) 
  (h1 : r = 3 * Real.sqrt 2)
  (h2 : AC = 5 * Real.sqrt y) 
  (h3 : BC = 13 * Real.sqrt y) 
  (h4 : AB = 10 * Real.sqrt y) : 
  r = 3 * Real.sqrt 2 → 
  (∃ (AC BC AB : ℝ), 
     AC = 5 * Real.sqrt (7) ∧ 
     BC = 13 * Real.sqrt (7) ∧ 
     AB = 10 * Real.sqrt (7)) :=
by
  sorry

end triangle_side_lengths_l367_367637


namespace length_of_train_l367_367261

-- Definitions of given conditions
def train_speed (kmh : ℤ) := 25
def man_speed (kmh : ℤ) := 2
def crossing_time (sec : ℤ) := 28

-- Relative speed calculation (in meters per second)
def relative_speed := (train_speed 1 + man_speed 1) * (5 / 18 : ℚ)

-- Distance calculation (in meters)
def distance_covered := relative_speed * (crossing_time 1 : ℚ)

-- The theorem statement: Length of the train equals distance covered in crossing time
theorem length_of_train : distance_covered = 210 := by
  sorry

end length_of_train_l367_367261


namespace range_of_dot_product_l367_367793

noncomputable def P (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ) 

def A : ℝ × ℝ := (-2, 0)

def AO : ℝ × ℝ := (2, 0)

def AP (θ : ℝ) : ℝ × ℝ :=
  let (px, py) := P θ
  (px + 2, py)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem range_of_dot_product :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi →
  2 ≤ dot_product AO (AP θ) ∧ dot_product AO (AP θ) ≤ 6 :=
by
  intros θ θ_range
  -- prove that the range is correct
  sorry

end range_of_dot_product_l367_367793


namespace remainder_equality_l367_367809

theorem remainder_equality 
  (Q Q' S S' E s s' : ℕ) 
  (Q_gt_Q' : Q > Q')
  (h1 : Q % E = S)
  (h2 : Q' % E = S')
  (h3 : (Q^2 * Q') % E = s)
  (h4 : (S^2 * S') % E = s') :
  s = s' :=
sorry

end remainder_equality_l367_367809


namespace number_of_ways_to_place_pawns_l367_367030

theorem number_of_ways_to_place_pawns :
  let n := 5 in
  let number_of_placements := (n.factorial) in
  let number_of_permutations := (n.factorial) in
  number_of_placements * number_of_permutations = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l367_367030


namespace first_player_wins_with_min_prime_count_l367_367195

theorem first_player_wins_with_min_prime_count :
  ∃ n : ℕ, n = 3 ∧ ∃ primes : list ℕ,
    (∀ p ∈ primes, nat.prime p ∧ p ≤ 100) ∧
    (∀ p1 p2 : ℕ, p1 ∈ primes → p2 ∈ primes → p1 ≠ p2 → list.index_of p1 primes < list.index_of p2 primes → (p1 % 10 = p2 / 10)) ∧
    (primes.length = n) ∧
    (∀ p1 p2 : ℕ, p1 ∈ primes → p2 ∈ primes → p1 ≠ p2 → p1 % 10 = p2 / 10) :=
begin
  sorry
end

end first_player_wins_with_min_prime_count_l367_367195


namespace pirate_problem_solution_l367_367942

def pirate_problem : Prop :=
  ∃ x : ℕ,
    (∑ i in finset.range (x + 1), i) = 5 * x ∧
    6 * x = 54

theorem pirate_problem_solution : pirate_problem :=
by
  sorry

end pirate_problem_solution_l367_367942


namespace uniform_b_interval_l367_367491

-- Define that b_1 is a uniform random variable on [0, 1]
def is_uniform_on_interval (b_1 : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → b_1 = x

-- Define the transformation b = 3(b_1 - 2)
def transform_b (b_1 b : ℝ) : Prop :=
  b = 3 * (b_1 - 2)

-- The main theorem stating the equivalence
theorem uniform_b_interval (b_1 b : ℝ) :
  is_uniform_on_interval b_1 → transform_b b_1 b → is_uniform_on_interval b (-6, -3) :=
by
  sorry

end uniform_b_interval_l367_367491


namespace height_of_tree_l367_367451

noncomputable def height_of_flagpole : ℝ := 4
noncomputable def shadow_of_flagpole : ℝ := 6
noncomputable def shadow_of_tree : ℝ := 12

theorem height_of_tree (h : height_of_flagpole / shadow_of_flagpole = x / shadow_of_tree) : x = 8 := by
  sorry

end height_of_tree_l367_367451


namespace equivalence_proof_l367_367689

-- Definitions for conditions
def P : Prop := -- The blue dragon on planet Gamma breathes fire (Definition placeholder)
def Q : Prop := -- The silver lion on planet Theta does not roar during storms (Definition placeholder)

-- The proof problem statement
theorem equivalence_proof (P Q : Prop) : 
  ((P → Q) ↔ (¬ Q → ¬ P)) ∧ ((P → Q) ↔ (¬ P ∨ Q)) :=
sorry

end equivalence_proof_l367_367689


namespace arrange_four_skycrapers_l367_367220

theorem arrange_four_skycrapers (
  A B C D: ℝ × ℝ 
) (h1 : ∃ t, t.1 + t.2 < 1 ∧ 0 < t.1 ∧ 0 < t.2 ∧ t.1 + t.2 > 0) : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ × ℝ),
    (x₁ = A ∨ x₁ = B ∨ x₁ = C ∨ x₁ = D) ∧
    (x₂ = A ∨ x₂ = B ∨ x₂ = C ∨ x₂ = D) ∧ x₂ ≠ x₁ ∧
    (x₃ = A ∨ x₃ = B ∨ x₃ = C ∨ x₃ = D) ∧ x₃ ≠ x₂ ∧ x₃ ≠ x₁ ∧
    (x₄ = A ∨ x₄ = B ∨ x₄ = C ∨ x₄ = D) ∧ x₄ ≠ x₃ ∧ x₄ ≠ x₂ ∧ x₄ ≠ x₁ := 
sorry

end arrange_four_skycrapers_l367_367220


namespace no_real_roots_f_of_f_x_eq_x_l367_367805

theorem no_real_roots_f_of_f_x_eq_x (a b c : ℝ) (h: (b - 1)^2 - 4 * a * c < 0) : 
  ¬(∃ x : ℝ, (a * (a * x^2 + b * x + c)^2 + b * (a * x^2 + b * x + c) + c = x)) := 
by
  sorry

end no_real_roots_f_of_f_x_eq_x_l367_367805


namespace distinct_pawns_5x5_l367_367025

theorem distinct_pawns_5x5 : 
  ∃ n : ℕ, n = 14400 ∧ 
  (∃ (get_pos : Fin 5 → Fin 5), function.bijective get_pos) :=
begin
  sorry
end

end distinct_pawns_5x5_l367_367025


namespace find_value_of_a_l367_367500

variable (U : Set ℕ) (A : ℕ → Set ℕ)
variable (a : ℕ)
variable (complement_U : Set ℕ → Set ℕ)
variable (a_value : ℕ)

-- Definitions
def universal_set := U = {1, 3, 5, 7, 9}
def set_A := A a = {1, |a - 5|, 9}
def complement_of_A := complement_U (A a) = {5, 7}

-- Theorem statement
theorem find_value_of_a (hU : universal_set U) (hA : set_A A a) (hCA : complement_of_A complement_U A a) :
  a_value = 2 ∨ a_value = 8 := 
sorry

end find_value_of_a_l367_367500


namespace solve_trig_eq_l367_367138

open Real

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (7 / 4 - 3 * cos (2 * x)) * abs (1 + 2 * cos (2 * x)) = sin x * (sin x + sin (5 * x)) →
  ∃ k : ℤ, x = (Int.natAbs k * (π / 6) + (k / 2) * π) ∨ x = -(Int.natAbs k * (π / 6) + (k / 2) * π) :=
begin
    sorry
end

end solve_trig_eq_l367_367138


namespace minimum_value_of_w_l367_367682

noncomputable def w (x y : ℝ) : ℝ := 3 * x ^ 2 + 3 * y ^ 2 + 9 * x - 6 * y + 27

theorem minimum_value_of_w : (∃ x y : ℝ, w x y = 20.25) := sorry

end minimum_value_of_w_l367_367682


namespace number_of_convex_quadrilaterals_l367_367938

theorem number_of_convex_quadrilaterals (points : Finset ℝ) (h : points.card = 12 ∧ ∀ p ∈ points, IsPointOnCircle p) :
  ∃ quadrilaterals, quadrilaterals.card = 495 :=
begin
  -- We need to prove that the number of convex quadrilaterals is exactly 495
  sorry
end

end number_of_convex_quadrilaterals_l367_367938


namespace monotonic_intervals_f_leq_zero_for_a_in_interval_l367_367096

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * real.log x - a * x^2 + (2 * a - 1) * x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x a) / x

theorem monotonic_intervals (a : ℝ) :
  if a ≤ 0 then 
    ∀ x : ℝ, 0 < x → g x a = real.log x - a * x + 2 * a - 1 → derivative (λ x, g x a) x ≥ 0
  else 
    ∀ x : ℝ, 0 < x → g x a = real.log x - a * x + 2 * a - 1 →
      (0 < x ∧ x < 1 / a → derivative (λ x, g x a) x ≥ 0) ∧ (x > 1 / a → derivative (λ x, g x a) x ≤ 0) :=
sorry

theorem f_leq_zero_for_a_in_interval (a : ℝ) (h : 1 / 2 < a ∧ a ≤ 1) :
   ∀ x : ℝ, 0 < x → f x a ≤ 0 :=
sorry

end monotonic_intervals_f_leq_zero_for_a_in_interval_l367_367096


namespace num_solutions_cos_eq_l367_367348

theorem num_solutions_cos_eq :
    (∃ (x : ℝ), -real.pi ≤ x ∧ x ≤ real.pi ∧ cos (4 * x) + (cos (3 * x)) ^ 2 + (cos (2 * x)) ^ 3 + (cos x) ^ 4 = 0) →
    10 :=
sorry

end num_solutions_cos_eq_l367_367348


namespace isosceles_triangle_ABC_l367_367432

open EuclideanGeometry

variables {A B C D M : Point}
variables (triangle_ABC : Triangle A B C)
variables (exterior_angle_bisector_ABC : ExteriorAngleBisector B C D)
variables (M_midpoint_BD : Midpoint M B D)

-- Define the problem as mentioned
axiom angle_BCD_60 : angle B C D = 60
axiom CD_double_AB : distance C D = 2 * distance A B
axiom M_is_midpoint : midpoint M B D

theorem isosceles_triangle_ABC (h1 : angle B C D = 60)
                              (h2 : distance C D = 2 * distance A B)
                              (h3 : midpoint M B D) :
                              isosceles_triangle A M C := sorry

end isosceles_triangle_ABC_l367_367432


namespace cost_of_4_stamps_l367_367628

theorem cost_of_4_stamps (cost_per_stamp : ℕ) (h : cost_per_stamp = 34) : 4 * cost_per_stamp = 136 :=
by
  sorry

end cost_of_4_stamps_l367_367628


namespace sum_adjacent_to_6_is_29_l367_367853
-- Import the Mathlib library for the necessary tools and functions

/--
  In a 3x3 table filled with numbers from 1 to 9 such that each number appears exactly once, 
  with conditions: 
    * (1, 1) contains 1, (3, 1) contains 2, (1, 3) contains 3, (3, 3) contains 4,
    * The sum of the numbers in the cells adjacent to the cell containing 5 is 9,
  Prove that the sum of the numbers in the cells adjacent to the cell containing 6 is 29.
-/
theorem sum_adjacent_to_6_is_29 
  (table : Fin 3 → Fin 3 → Fin 9)
  (H_uniqueness : ∀ i j k l, (table i j = table k l) → (i = k ∧ j = l))
  (H_valid_entries : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)
  (H_initial_positions : table 0 0 = 1 ∧ table 2 0 = 2 ∧ table 0 2 = 3 ∧ table 2 2 = 4)
  (H_sum_adj_to_5 : ∃ (i j : Fin 3), table i j = 5 ∧ 
                      ((i > 0 ∧ table (i-1) j +
                       (i < 2 ∧ table (i+1) j) +
                       (j > 0 ∧ table i (j-1)) +
                       (j < 2 ∧ table i (j+1))) = 9)) :
  ∃ i j, table i j = 6 ∧
  (i > 0 ∧ table (i-1) j +
   (i < 2 ∧ table (i+1) j) +
   (j > 0 ∧ table i (j-1)) +
   (j < 2 ∧ table i (j+1))) = 29 := sorry

end sum_adjacent_to_6_is_29_l367_367853


namespace triangle_area_inequality_l367_367894

theorem triangle_area_inequality 
  (a b c : ℝ) (λ μ ν : ℝ) (h1 : 0 < λ) (h2 : 0 < μ) (h3 : 0 < ν) 
  (h_triangle : ∃ A B C : Type, ∃ Δ : ℝ, Δ = (√((s * (s - a) * (s - b) * (s - c)))) / 2 ∧ s = (a + b + c) / 2) :
  (∃ Δ : ℝ, Δ ≤ (a * b * c) ^ (2/3) * (λ * μ + μ * ν + ν * λ) / (4 * real.sqrt 3 * (λ * μ * ν) ^ (2/3))) :=
by sorry

end triangle_area_inequality_l367_367894


namespace smallest_positive_integer_congruence_l367_367208

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 0 < x ∧ x < 17 ∧ (3 * x ≡ 14 [MOD 17]) := sorry

end smallest_positive_integer_congruence_l367_367208


namespace thirteenth_number_is_6358_l367_367201

theorem thirteenth_number_is_6358 : 
  ∃ (l : List ℕ), l = List.permutations [3, 5, 6, 8] ∧ l.nth 12 = some 6358 := sorry

end thirteenth_number_is_6358_l367_367201


namespace evaluate_expression_l367_367694

theorem evaluate_expression (a b : ℕ) (h₁ : a = 3) (h₂ : b = 4) : ((a^b)^a - (b^a)^b) = -16246775 :=
by
  rw [h₁, h₂]
  sorry

end evaluate_expression_l367_367694


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367969

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367969


namespace problem_correct_options_l367_367433

section orthogonal_and_projection

variables (a b : ℝ × ℝ)
def vec_a : ℝ × ℝ := (-1, 1)
def vec_b : ℝ × ℝ := (0, -2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  (dot_product a b / (magnitude a ^ 2)) • a

theorem problem_correct_options : 
  orthogonal (vec_a + vec_b) vec_a ∧ projection vec_a vec_b = (1, -1) :=
by
  sorry

end orthogonal_and_projection

end problem_correct_options_l367_367433


namespace factorization_left_to_right_l367_367210

-- Define the equation and its factorized form
def LHS : ℝ → ℝ := λ x, x^2 - 1
def RHS : ℝ → ℝ := λ x, (x + 1) * (x - 1)

-- The main theorem statement
theorem factorization_left_to_right : ∀ x : ℝ, LHS x = RHS x :=
by
  -- Proof to be provided later
  sorry

end factorization_left_to_right_l367_367210


namespace full_sets_and_leftovers_l367_367022

theorem full_sets_and_leftovers (total_hotdogs : ℕ) (set_size : ℕ) 
  (h1 : total_hotdogs = 25197625) (h2 : set_size = 5) :
  ∃ (full_sets leftovers : ℕ), full_sets = 5039525 ∧ leftovers = 0 ∧ total_hotdogs = set_size * full_sets + leftovers :=
by {
  use (5039525, 0),
  split,
  { exact rfl },
  split,
  { exact rfl },
  rw [h1, h2]
}

end full_sets_and_leftovers_l367_367022


namespace rectangle_diagonal_length_l367_367255

-- Given conditions
def length : ℝ := 16
def area : ℝ := 192

-- The width can be derived from the area and the length
def width : ℝ := area / length

-- Pythagorean theorem to find the diagonal
def diagonal_length : ℝ := real.sqrt (length^2 + width^2)

-- Statement to prove
theorem rectangle_diagonal_length : diagonal_length = 20 :=
by sorry

end rectangle_diagonal_length_l367_367255


namespace complex_number_problem_l367_367408

noncomputable def z (cos_value : ℂ) := by sorry

theorem complex_number_problem
  (z : ℂ)
  (hz : z + z⁻¹ = 2 * real.cos (5 * real.pi / 180)) :
  z^100 + z^(-100) = -1.92 :=
sorry

end complex_number_problem_l367_367408


namespace hexagon_area_b_l367_367247

theorem hexagon_area_b (ABCDEF : Hexagon) (AF G : Point) (a b : ℕ) :
  (∃ s, (s = 3) ∧
  (∀ (A B C D E F : Point), ABCDEF.has_equal_sides [A, B, C, D, E, F] s) ∧
  (G = midpoint A F) ∧
  (∀ (α : angle), ABCDEF.has_equal_angles α (120 : ℝ))
  ) →
  ∃ s : ℝ, (s = 27√3) ∧ b = 3 := 
by 
  sorry

end hexagon_area_b_l367_367247


namespace max_dist_2_minus_2i_l367_367752

open Complex

noncomputable def max_dist (z1 : ℂ) : ℝ :=
  Complex.abs 1 + Complex.abs z1

theorem max_dist_2_minus_2i :
  max_dist (2 - 2*I) = 1 + 2 * Real.sqrt 2 := by
  sorry

end max_dist_2_minus_2i_l367_367752


namespace constraint_condition_2000_yuan_wage_l367_367184

-- Definitions based on the given conditions
def wage_carpenter : ℕ := 50
def wage_bricklayer : ℕ := 40
def total_wage : ℕ := 2000

-- Let x be the number of carpenters and y be the number of bricklayers
variable (x y : ℕ)

-- The proof problem statement
theorem constraint_condition_2000_yuan_wage (x y : ℕ) : 
  wage_carpenter * x + wage_bricklayer * y = total_wage → 5 * x + 4 * y = 200 :=
by
  intro h
  -- Simplification step will be placed here
  sorry

end constraint_condition_2000_yuan_wage_l367_367184


namespace remainder_47_mod_288_is_23_mod_24_l367_367616

theorem remainder_47_mod_288_is_23_mod_24 (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := 
sorry

end remainder_47_mod_288_is_23_mod_24_l367_367616


namespace kamal_marks_in_mathematics_l367_367905

def kamal_marks_english : ℕ := 96
def kamal_marks_physics : ℕ := 82
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 79
def kamal_number_of_subjects : ℕ := 5

theorem kamal_marks_in_mathematics :
  let total_marks := kamal_average_marks * kamal_number_of_subjects
  let total_known_marks := kamal_marks_english + kamal_marks_physics + kamal_marks_chemistry + kamal_marks_biology
  total_marks - total_known_marks = 65 :=
by
  sorry

end kamal_marks_in_mathematics_l367_367905


namespace monster_perimeter_l367_367883

theorem monster_perimeter (r : ℝ) (θ : ℝ) (h1 : r = 1) (h2 : θ = 120) : 
  let full_circle := 2 * Real.pi * r,
      unshaded_arc := (θ / 360) * full_circle,
      shaded_arc := full_circle - unshaded_arc,
      perimeter := shaded_arc + 2 * r 
  in perimeter = (4 / 3) * Real.pi + 2 :=
by
  sorry

end monster_perimeter_l367_367883


namespace integral_values_solution_l367_367297

theorem integral_values_solution : ∃ (x y z : ℤ), z ^ x = y ^ (3 * x) ∧ 2 ^ z = 4 * 8 ^ x ∧ x + y + z = 20 ∧ x = 2 ∧ y = 2 ∧ z = 8 :=
by
  have h₁ : (8 : ℤ) ^ (2 : ℤ) = 2 ^ (3 * 2 + 2), sorry,
  have h₂ : (2 : ℤ) ^ (3 * 2 + 2) = 32, sorry,
  use (2, 2, 8),
  simp,
  split,
  { rw [pow_two, ← h₁, ← h₂], ring } -- proof for z ^ x = y ^ (3 * x)
  split,
  { rw [pow_two, ← h₁, ← h₂], ring } -- proof for 2 ^ z = 4 * 8 ^ x
  split,
  { norm_num }, -- proof for x + y + z = 20
  split, { refl }, -- proof for x = 2
  split, { refl }, -- proof for y = 2
  { refl } -- proof for z = 8

end integral_values_solution_l367_367297


namespace largest_k_power_of_2_dividing_product_of_first_50_even_numbers_l367_367100

open Nat

theorem largest_k_power_of_2_dividing_product_of_first_50_even_numbers :
  let Q := (List.range (50 + 1)).map (λ n, 2 * n).prod in
  let k := (Q.factorization 2) in
  k = 97 :=
by
  sorry

end largest_k_power_of_2_dividing_product_of_first_50_even_numbers_l367_367100


namespace winning_strategy_exists_l367_367198

def prime_numbers : List ℕ := [
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61,
  67, 71, 73, 79, 83, 89, 97
]

def valid_prime (n : ℕ) : Prop := n ∈ prime_numbers

def last_digit (n : ℕ) : ℕ := n % 10

def first_digit (n : ℕ) : ℕ := n / 10 ^ (n.toString.length - 1)

def can_continue (n m : ℕ) : Prop :=
  last_digit n = first_digit m

noncomputable def strategy (primes : List ℕ) : Bool :=
  primes = [19, 97, 79]

theorem winning_strategy_exists : ∃ primes : List ℕ, strategy primes = true ∧ length primes = 3 :=
by
  sorry

end winning_strategy_exists_l367_367198


namespace max_triangles_proof_l367_367068

noncomputable def max_triangles (points : Finset ℝ × ℝ × ℝ) (h_non_coplanar : ∀ s : Finset (ℝ × ℝ × ℝ), s.card = 4 → ¬AffinelyIndependent ℝ (s : Set (ℝ × ℝ × ℝ))) (h_no_tetrahedron : ¬∃ s : Finset (ℝ × ℝ × ℝ), s.card = 4 ∧ is_tetrahedron (s : Set (ℝ × ℝ × ℝ))) : ℕ :=
  27

theorem max_triangles_proof : ∀ (points : Finset (ℝ × ℝ × ℝ)),
  (∀ s : Finset (ℝ × ℝ × ℝ), s.card = 4 → ¬AffinelyIndependent ℝ (s : Set (ℝ × ℝ × ℝ))) →
  (¬∃ s : Finset (ℝ × ℝ × ℝ), s.card = 4 ∧ is_tetrahedron (s : Set (ℝ × ℝ × ℝ))) →
  max_triangles points sorry sorry = 27 :=
sorry

end max_triangles_proof_l367_367068


namespace digit_A_divisible_by_2_3_and_4_l367_367920

theorem digit_A_divisible_by_2_3_and_4:
  ∃ (A : ℕ), 0 ≤ A ∧ A < 10 ∧
  (∃ k : ℕ, 26372 * 10^3 + A * 10^2 + 2 * 10 + 1 = 2 * k) ∧
  (∃ k : ℕ, (2 + 6 + 3 + 7 + 2 + A + 2 + 1 = 3 * k)) ∧
  (∃ k : ℕ, 100 + 10+A + 2 = 4 * k) :=
begin
  use [4],
  split,
  { exact by norm_num },
  split,
  { exact by norm_num },
  split,
  { use 1318615, norm_num},
  split,
  { use 10, norm_num},
  { use 252, norm_num},
end

end digit_A_divisible_by_2_3_and_4_l367_367920


namespace segment_MN_length_l367_367115

-- Given AC = BC
variables {A B C E M N : Point} (h_AC_eq_BC : distance A C = distance B C)

-- Given point E on AB
variables (h_E_on_AB : collinear A B E)

-- Defined lengths AE and BE
variables (AE BE : ℝ) (h_dist_AE : distance A E = AE) (h_dist_BE : distance B E = BE)

-- Points M and N where circles inscribed in triangles touch CE
variables (CE : Segment) (h_M_on_CE : CE.contains M) (h_N_on_CE : CE.contains N)

-- Distance MN is to be found
def length_MN : ℝ :=
  (abs (BE - AE)) / 2

theorem segment_MN_length :
  distance M N = length_MN AE BE :=
sorry

end segment_MN_length_l367_367115


namespace quadrilateral_ABCD_pq_sum_l367_367066

noncomputable def AB_pq_sum : ℕ :=
  let p : ℕ := 9
  let q : ℕ := 141
  p + q

theorem quadrilateral_ABCD_pq_sum (BC CD AD : ℕ) (m_angle_A m_angle_B : ℕ) (hBC : BC = 8) (hCD : CD = 12) (hAD : AD = 10) (hAngleA : m_angle_A = 60) (hAngleB : m_angle_B = 60) : AB_pq_sum = 150 := by sorry

end quadrilateral_ABCD_pq_sum_l367_367066


namespace exists_distinct_naturals_divisible_sum_diff_l367_367690

theorem exists_distinct_naturals_divisible_sum_diff :
  ∃ (a : Fin 2013 → ℕ), (∀ i j : Fin 2013, i ≠ j → (a i + a j) % abs (a i - a j) = 0) := by
    sorry

end exists_distinct_naturals_divisible_sum_diff_l367_367690


namespace infinitely_many_not_expressed_as_sum_of_powers_l367_367952

-- Define the number of positive divisors \( d(a) \)
def numDivisors (n : ℕ) : ℕ :=
  (List.range n).count (λ k, n % (k + 1) = 0)

-- Main theorem statement
theorem infinitely_many_not_expressed_as_sum_of_powers :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ k ∈ S, ∀ a b : ℕ, k ≠ a ^ numDivisors a + b ^ numDivisors b :=
by
  -- Placeholder for the proof
  sorry

end infinitely_many_not_expressed_as_sum_of_powers_l367_367952


namespace alcohol_percentage_calculation_l367_367218

-- Define the conditions as hypothesis
variables (original_solution_volume : ℝ) (original_alcohol_percent : ℝ)
          (added_alcohol_volume : ℝ) (added_water_volume : ℝ)

-- Assume the given values in the problem
variables (h1 : original_solution_volume = 40) (h2 : original_alcohol_percent = 5)
          (h3 : added_alcohol_volume = 2.5) (h4 : added_water_volume = 7.5)

-- Define the proof goal
theorem alcohol_percentage_calculation :
  let original_alcohol_volume := original_solution_volume * (original_alcohol_percent / 100)
  let total_alcohol_volume := original_alcohol_volume + added_alcohol_volume
  let total_solution_volume := original_solution_volume + added_alcohol_volume + added_water_volume
  let new_alcohol_percent := (total_alcohol_volume / total_solution_volume) * 100
  new_alcohol_percent = 9 :=
by {
  sorry
}

end alcohol_percentage_calculation_l367_367218


namespace points_lie_on_parabola_l367_367740

theorem points_lie_on_parabola (t : ℝ) :
  ∃ (a b c : ℝ), ∀ t : ℝ, 
  let x := 3^t - 4 in
  let y := 9^t - 6 * 3^t - 2 in
  y = a * x^2 + b * x + c :=
begin
  use [1, 2, -6],
  sorry
end

end points_lie_on_parabola_l367_367740


namespace product_of_b_l367_367544

noncomputable def b_product : ℤ :=
  let y1 := 3
  let y2 := 8
  let x1 := 2
  let l := y2 - y1 -- Side length of the square
  let b₁ := x1 - l -- One possible value of b
  let b₂ := x1 + l -- Another possible value of b
  b₁ * b₂ -- Product of possible values of b

theorem product_of_b :
  b_product = -21 := by
  sorry

end product_of_b_l367_367544


namespace proof_of_x_and_velocity_l367_367376

variables (a T L R x : ℝ)

-- Given condition
def given_eq : Prop := (a * T) / (a * T - R) = (L + x) / x

-- Target statement to prove
def target_eq_x : Prop := x = a * T * (L / R) - L
def target_velocity : Prop := a * (L / R)

-- Main theorem to prove the equivalence
theorem proof_of_x_and_velocity (a T L R : ℝ) : given_eq a T L R x → target_eq_x a T L R x ∧ target_velocity a T L R =
  sorry

end proof_of_x_and_velocity_l367_367376


namespace range_of_a_l367_367750

theorem range_of_a {
  a : ℝ
} :
  (∀ x ∈ Set.Ici (2 : ℝ), (x^2 + (2 - a) * x + 4 - 2 * a) > 0) ↔ a < 3 :=
by
  sorry

end range_of_a_l367_367750


namespace min_z_value_l367_367020

theorem min_z_value (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4 ^ x + 2 ^ y) : z ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_z_value_l367_367020


namespace karan_borrowed_years_l367_367504

-- Define constants
def principal : ℝ := 5525.974025974026
def total_amount : ℝ := 8510
def rate_of_interest : ℝ := 6 / 100

-- Define a term representing the number of years
def number_of_years : ℝ := (total_amount - principal) * 100 / (principal * rate_of_interest)

-- The theorem statement that verifies the number of years Mr. Karan borrowed the money for is 9.
theorem karan_borrowed_years : number_of_years = 9 := by
  -- Simplify the expressions and the following calculations
  sorry

end karan_borrowed_years_l367_367504


namespace rectangle_area_l367_367892

/-- 
In the rectangle \(ABCD\), \(AD - AB = 9\) cm. The area of trapezoid \(ABCE\) is 5 times 
the area of triangle \(ADE\). The perimeter of triangle \(ADE\) is 68 cm less than the 
perimeter of trapezoid \(ABCE\). Prove that the area of the rectangle \(ABCD\) 
is 3060 square centimeters.
-/
theorem rectangle_area (AB AD : ℝ) (S_ABC : ℝ) (S_ADE : ℝ) (P_ADE : ℝ) (P_ABC : ℝ) :
  AD - AB = 9 →
  S_ABC = 5 * S_ADE →
  P_ADE = P_ABC - 68 →
  (AB * AD = 3060) :=
by
  sorry

end rectangle_area_l367_367892


namespace charles_nickels_l367_367280

theorem charles_nickels :
  ∀ (num_pennies num_cents penny_value nickel_value n : ℕ),
  num_pennies = 6 →
  num_cents = 21 →
  penny_value = 1 →
  nickel_value = 5 →
  (num_cents - num_pennies * penny_value) / nickel_value = n →
  n = 3 :=
by
  intros num_pennies num_cents penny_value nickel_value n hnum_pennies hnum_cents hpenny_value hnickel_value hn
  sorry

end charles_nickels_l367_367280


namespace loss_percentage_is_13_l367_367152

def cost_price : ℕ := 1500
def selling_price : ℕ := 1305
def loss : ℕ := cost_price - selling_price
def loss_percentage : ℚ := (loss : ℚ) / cost_price * 100

theorem loss_percentage_is_13 :
  loss_percentage = 13 := 
by
  sorry

end loss_percentage_is_13_l367_367152


namespace find_total_games_l367_367691

-- Define the initial conditions
def avg_points_per_game : ℕ := 26
def games_played : ℕ := 15
def goal_avg_points : ℕ := 30
def required_avg_remaining : ℕ := 42

-- Statement of the proof problem
theorem find_total_games (G : ℕ) :
  avg_points_per_game * games_played + required_avg_remaining * (G - games_played) = goal_avg_points * G →
  G = 20 :=
by sorry

end find_total_games_l367_367691


namespace minimum_squares_and_perimeter_l367_367679

theorem minimum_squares_and_perimeter 
  (length width : ℕ) 
  (h_length : length = 90) 
  (h_width : width = 42) 
  (h_gcd : Nat.gcd length width = 6) 
  : 
  ((length / Nat.gcd length width) * (width / Nat.gcd length width) = 105) ∧ 
  (105 * (4 * Nat.gcd length width) = 2520) := 
by 
  sorry

end minimum_squares_and_perimeter_l367_367679


namespace proof_of_x_and_velocity_l367_367374

variables (a T L R x : ℝ)

-- Given condition
def given_eq : Prop := (a * T) / (a * T - R) = (L + x) / x

-- Target statement to prove
def target_eq_x : Prop := x = a * T * (L / R) - L
def target_velocity : Prop := a * (L / R)

-- Main theorem to prove the equivalence
theorem proof_of_x_and_velocity (a T L R : ℝ) : given_eq a T L R x → target_eq_x a T L R x ∧ target_velocity a T L R =
  sorry

end proof_of_x_and_velocity_l367_367374


namespace smallest_possible_c_minus_a_l367_367570

theorem smallest_possible_c_minus_a :
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a * b * c = Nat.factorial 9 ∧ c - a = 216 := 
by
  sorry

end smallest_possible_c_minus_a_l367_367570


namespace intersection_complement_eq_l367_367010

-- Define the universal set as ℝ
def U := set ℝ

-- Define the first set A
def A : set ℝ := {-1, 0, 1, 5}

-- Define the second set B as per condition
def B : set ℝ := {x | x ^ 2 - x - 2 ≥ 0}

-- Define the complement of B in ℝ
def C_B : set ℝ := {x | -1 < x ∧ x < 2}

-- Define the expected intersection of A and C_B
def expected_intersection : set ℝ := {0, 1}

-- The theorem we need to prove
theorem intersection_complement_eq : A ∩ C_B = expected_intersection :=
by
  sorry

end intersection_complement_eq_l367_367010


namespace marbles_count_l367_367878

variables {g y : ℕ}

theorem marbles_count (h1 : (g - 1)/(g + y - 1) = 1/8)
                      (h2 : g/(g + y - 3) = 1/6) :
                      g + y = 9 :=
by
-- This is just setting up the statements we need to prove the theorem. The actual proof is to be completed.
sorry

end marbles_count_l367_367878


namespace seq_properties_and_sum_l367_367761

-- Defining the arithmetic sequence {a_n}
variables (a_n : ℕ → ℚ) (d : ℚ)
axiom h1 : a_n 3 + a_n 6 = -1/3
axiom h2 : a_n 1 * a_n 8 = -4/3
axiom h3 : a_n 1 > a_n 8
axiom arithmetic_seq : ∃ (a : ℚ) (d : ℚ), ∀ n, a_n n = a + (n - 1) * d

-- Define the value of a_n explicitly as per solution
def seq_a_n := λ n : ℕ, -1/3 * n + 4/3

-- Define sequence {b_n}
def b_n := λ n : ℕ, -n + 2

-- Define sequence {2^b_n}
def seq_2_bn := λ n : ℕ, 2 ^ b_n n

-- Proving sum of sequence {2^b_n} is 4
noncomputable def sum_seq2_bn : ℚ := 2 / (1 - 1/2)

theorem seq_properties_and_sum :
  (∀ n, a_n n = seq_a_n n) ∧ (∀ n, b_n n = seq_a_n (3 * n - 2)) ∧ 
  (sum_seq2_bn = 4) :=
by
  sorry

end seq_properties_and_sum_l367_367761


namespace house_ordering_count_l367_367202

def house_colors := ["green", "blue", "red", "purple"]

-- Define the conditions as predicates for the ordering of houses
def condition1 (order : List String) : Prop := 
  order.indexOf "green" < order.indexOf "blue"

def condition2 (order : List String) : Prop := 
  order.indexOf "red" < order.indexOf "purple"

def condition3 (order : List String) : Prop := 
  (order.indexOf "green" + 1 ≠ order.indexOf "red") ∧ (order.indexOf "red" + 1 ≠ order.indexOf "green")

-- Define the valid orders satisfying all conditions
def valid_orders (order : List String) : Prop := 
  condition1 order ∧ condition2 order ∧ condition3 order

-- State the theorem: the number of valid orderings under the given conditions is 3
theorem house_ordering_count : {order : List String // valid_orders order}.card = 3 :=
by
  sorry

end house_ordering_count_l367_367202


namespace minimum_shirts_to_save_money_l367_367267
noncomputable def Acme_cost (x : ℕ) : ℕ := 70 + 11 * x
noncomputable def Beta_cost (x : ℕ) : ℕ := 10 + 15 * x

theorem minimum_shirts_to_save_money : ∃ (x : ℕ), x = 16 ∧ Acme_cost x < Beta_cost x :=
by {
  use 16,
  split,
  { refl, },
  { sorry }
}

end minimum_shirts_to_save_money_l367_367267


namespace correct_time_l367_367988

theorem correct_time (x : ℕ) : 8 * x + 7 * x = 60 -> 7 * 60 / 15 = 32 :=
by
  intro h
  have h1: 15 * x = 60 := by linarith
  have h2: x = 60 / 15 := by linarith
  have h3: x = 4 := by linarith
  exact Nat.eq_div_of_mul_eq_mul_right (by norm_num) h3 (by norm_num)

end correct_time_l367_367988


namespace sphere_surface_area_of_tetrahedron_l367_367413

theorem sphere_surface_area_of_tetrahedron (a : ℝ) (h : a = 4) : 
  let r := a * (sqrt 6) / 2 in
  4 * π * r^2 = 24 * π :=
by
  sorry

end sphere_surface_area_of_tetrahedron_l367_367413


namespace sum_first_9_terms_l367_367402

variable {a_n : ℕ → ℤ} -- Sequence a_n

-- Definitions for the conditions:
def arithmetic_sequence (u : ℕ → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ n : ℕ, u n = a + n * d

def condition (u : ℕ → ℤ) : Prop :=
  3 * u 4 + u 8 = 36

-- Target statement that needs to be proved:
theorem sum_first_9_terms (u : ℕ → ℤ) (h_arith : arithmetic_sequence u) (h_cond : condition u) :
  ∑ n in finset.range 9, u n = 81 :=
begin
  -- Proof goes here
  sorry
end

end sum_first_9_terms_l367_367402


namespace general_term_a_sum_c_sequence_l367_367414

-- Definitions and conditions
def sequence_a (n : ℕ) : ℕ := if n = 1 then 1 else 7 * 4^(n-2)
def S (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms

-- Conditions
axiom a1 : sequence_a 1 = 1
axiom a2 : ∀ n, n ≥ 2 → a_n = 3 * S (n-1) + 4

-- Goal (I)
theorem general_term_a (n : ℕ) : sequence_a n = if n = 1 then 1 else 7 * 4^(n-2) :=
sorry

-- Definitions for part (II)
def sequence_b (n : ℕ) : ℝ := log (2 : ℝ) (sequence_a (n+2) / 7)
def sequence_c (n : ℕ) : ℝ := sequence_b n / 2^(n+1)

def T (n : ℕ) : ℝ := (finset.range n).sum (λ i, sequence_c (i + 1))

-- Goal (II)
theorem sum_c_sequence (n : ℕ) : T n = 2 - (n + 2) / 2^n :=
sorry

end general_term_a_sum_c_sequence_l367_367414


namespace evaluate_expression_l367_367696

theorem evaluate_expression : 
  let a := 3
  let b := 4
  (a^b)^a - (b^a)^b = -16245775 := 
by 
  sorry

end evaluate_expression_l367_367696


namespace sum_adjacent_6_is_29_l367_367838

-- Define the grid and the placement of numbers 1 to 4
structure Grid :=
  (grid : Fin 3 → Fin 3 → Nat)
  (h_unique : ∀ i j, grid i j ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_distinct : Function.Injective (λ (i : Fin 3) (j : Fin 3), grid i j))
  (h_placement : grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧ grid 2 2 = 4)

-- Define the condition of the sum of numbers adjacent to 5 being 9
def sum_adjacent_5 (g : Grid) : Prop :=
  let (i, j) := (0, 1) in -- Position for number 5
  (g.grid (i.succ) j + g.grid (i.succ.pred) j + g.grid i (j.succ) + g.grid i (j.pred)) = 9

-- Define the main theorem
theorem sum_adjacent_6_is_29 (g : Grid) (h_sum_adj_5 : sum_adjacent_5 g) : 
  (g.grid 1 0 + g.grid 1 2 + g.grid 0 1 + g.grid 2 1 = 29) := sorry

end sum_adjacent_6_is_29_l367_367838


namespace sum_adjacent_to_6_is_29_l367_367851
-- Import the Mathlib library for the necessary tools and functions

/--
  In a 3x3 table filled with numbers from 1 to 9 such that each number appears exactly once, 
  with conditions: 
    * (1, 1) contains 1, (3, 1) contains 2, (1, 3) contains 3, (3, 3) contains 4,
    * The sum of the numbers in the cells adjacent to the cell containing 5 is 9,
  Prove that the sum of the numbers in the cells adjacent to the cell containing 6 is 29.
-/
theorem sum_adjacent_to_6_is_29 
  (table : Fin 3 → Fin 3 → Fin 9)
  (H_uniqueness : ∀ i j k l, (table i j = table k l) → (i = k ∧ j = l))
  (H_valid_entries : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)
  (H_initial_positions : table 0 0 = 1 ∧ table 2 0 = 2 ∧ table 0 2 = 3 ∧ table 2 2 = 4)
  (H_sum_adj_to_5 : ∃ (i j : Fin 3), table i j = 5 ∧ 
                      ((i > 0 ∧ table (i-1) j +
                       (i < 2 ∧ table (i+1) j) +
                       (j > 0 ∧ table i (j-1)) +
                       (j < 2 ∧ table i (j+1))) = 9)) :
  ∃ i j, table i j = 6 ∧
  (i > 0 ∧ table (i-1) j +
   (i < 2 ∧ table (i+1) j) +
   (j > 0 ∧ table i (j-1)) +
   (j < 2 ∧ table i (j+1))) = 29 := sorry

end sum_adjacent_to_6_is_29_l367_367851


namespace point_C_velocity_l367_367367

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l367_367367


namespace largest_prime_divisor_of_84511_l367_367342

def base5_to_base10 : ℕ := 10201021 -- This base-5 number represented as decimal for easier manipulation

def num : ℕ := 1*5^7 + 0*5^6 + 2*5^5 + 0*5^4 + 1*5^3 + 0*5^2 + 2*5^1 + 1*5^0

theorem largest_prime_divisor_of_84511 :
  num = 84511 ∧ (∀ p, prime p → p ∣ 84511 → p ≤ 139) ∧ prime 139 ∧ 139 ∣ 84511 :=
begin
  sorry
end

end largest_prime_divisor_of_84511_l367_367342


namespace valid_proposition_l367_367597

-- Define propositions
def A : Prop := (∃ x : ℕ, x - 1 = 0)
def B : Prop := 2 + 3 = 8
def C : Prop := false  -- C is not a proposition
def D : Prop := false  -- D requires context and cannot be determined

-- The theorem states that B is the valid proposition among the choices.
theorem valid_proposition : B ∧ ¬A ∧ ¬C ∧ ¬D := 
by {
  split,
  { exact false.intro (by norm_num) },
  split,
  { exact not_exists.mpr (λ x : ℕ, by norm_num) },
  split,
  { exact false.intro (λ h : false, h) },
  { exact false.intro (λ h : false, h) }
}

end valid_proposition_l367_367597


namespace solve_equation_l367_367980

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 := by
  sorry

end solve_equation_l367_367980


namespace min_value_of_f_l367_367772

noncomputable def f (x : ℝ) : ℝ := x + 3 / (x + 1)

theorem min_value_of_f (x : ℝ) (h : x > 0) : f x ≥ 2 * Real.sqrt 3 - 1 :=
begin
  sorry
end

end min_value_of_f_l367_367772


namespace point_on_line_iff_sum_one_l367_367517

variables {A B C O : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup O]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ O]

-- Defining the vector operations and their properties
variables (OA OB OC : A → ℝ → Type)
variables {a b c o : A}

-- Defining the points
variables (pA pB pC pO : A)

-- Conditions & Definitions
variable (hne : pA ≠ pB)
variable (hnot : ¬(pO = pA ∨ pO = pB))

-- Vector definitions based on given condition in the problem
def vec_OA := \overrightarrow pO pA
def vec_OB := \overrightarrow pO pB
def vec_OC := \overrightarrow pO pC

-- Given vector expression
def vec_expr (x y : ℝ) : vec_OC = x • vec_OA + y • vec_OB

-- Theorem to prove that C belongs to line AB if and only if x + y = 1
theorem point_on_line_iff_sum_one (x y : ℝ) :
  (∃ z : ℝ, \overrightarrow pA pC = z • \overrightarrow pA pB) ↔ (x + y = 1) :=
sorry

end point_on_line_iff_sum_one_l367_367517


namespace remainder_mod_7_l367_367206

theorem remainder_mod_7 : (9^7 + 8^8 + 7^9) % 7 = 3 :=
by sorry

end remainder_mod_7_l367_367206


namespace pentagon_to_triangle_l367_367163

theorem pentagon_to_triangle : 
  ∃ (square_side : ℕ) (triangle_side : ℕ), square_side = 2 ∧ triangle_side = 2 ∧
  let pentagon_area := square_side^2 + 2 * (1/2 * triangle_side * triangle_side) in
  ∃ (new_triangle_side : ℕ), 
    pentagon_area = 1/2 * new_triangle_side^2 ∧ new_triangle_side = 4 :=
begin
  sorry
end

end pentagon_to_triangle_l367_367163


namespace shelby_stars_yesterday_l367_367974

-- Define the number of stars earned yesterday
def stars_yesterday : ℕ := sorry

-- Condition 1: In all, Shelby earned 7 gold stars
def stars_total : ℕ := 7

-- Condition 2: Today, she earned 3 more gold stars
def stars_today : ℕ := 3

-- The proof statement that combines the conditions 
-- and question to the correct answer
theorem shelby_stars_yesterday (y : ℕ) (h1 : y + stars_today = stars_total) : y = 4 := 
by
  -- Placeholder for the actual proof
  sorry

end shelby_stars_yesterday_l367_367974


namespace quadrilateral_ABCD_AB_eq_p_plus_sqrt_q_l367_367064

theorem quadrilateral_ABCD_AB_eq_p_plus_sqrt_q (BC CD AD : ℝ) (angle_A angle_B : ℝ) (h1 : BC = 8)
  (h2 : CD = 12) (h3 : AD = 10) (h4 : angle_A = 60) (h5 : angle_B = 60) : 
  ∃ (p q : ℤ), AB = p + real.sqrt q ∧ p + q = 150 :=
by
  sorry

end quadrilateral_ABCD_AB_eq_p_plus_sqrt_q_l367_367064


namespace min_distance_squared_l367_367769

theorem min_distance_squared (a b c d : ℝ) (e : ℝ) (h₀ : e = Real.exp 1) 
  (h₁ : (a - 2 * Real.exp a) / b = 1) (h₂ : (2 - c) / d = 1) :
  (a - c)^2 + (b - d)^2 = 8 := by
  sorry

end min_distance_squared_l367_367769


namespace radius_of_circumscribed_sphere_is_correct_l367_367147

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  √(5 * a^2 / 4 + 9 * a^2 / 64) / 8

theorem radius_of_circumscribed_sphere_is_correct (a : ℝ) : 
  radius_of_circumscribed_sphere(a) = (a * √89) / 8 :=
by
  sorry

end radius_of_circumscribed_sphere_is_correct_l367_367147


namespace smallest_possible_c_minus_a_l367_367569

theorem smallest_possible_c_minus_a :
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a * b * c = Nat.factorial 9 ∧ c - a = 216 := 
by
  sorry

end smallest_possible_c_minus_a_l367_367569


namespace inscribed_circle_equilateral_l367_367638

theorem inscribed_circle_equilateral (A B C A1 B1 C1 : Type)
  [is_triangle A B C]
  (inscribed : inscribed_circle A B C A1 B1 C1)
  (similar : similar_triangles (triangle A1 B1 C1) (triangle A B C)) :
  is_equilateral (triangle A B C) :=
sorry

end inscribed_circle_equilateral_l367_367638


namespace can_form_triangle_l367_367213

theorem can_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  (a = 7 ∧ b = 12 ∧ c = 17) → True :=
by
  sorry

end can_form_triangle_l367_367213


namespace tangent_line_at_point_l367_367998

noncomputable def tangent_line_equation (x : ℝ) : Prop :=
  ∀ y : ℝ, y = x * (3 * Real.log x + 1) → (x = 1 ∧ y = 1) → y = 4 * x - 3

theorem tangent_line_at_point : tangent_line_equation 1 :=
sorry

end tangent_line_at_point_l367_367998


namespace sum_adjacent_cells_of_6_is_29_l367_367844

theorem sum_adjacent_cells_of_6_is_29 (table : Fin 3 × Fin 3 → ℕ)
  (uniq : Function.Injective table)
  (range : ∀ x, 1 ≤ table x ∧ table x ≤ 9)
  (pos_1 : table ⟨0, 0⟩ = 1)
  (pos_2 : table ⟨2, 0⟩ = 2)
  (pos_3 : table ⟨0, 2⟩ = 3)
  (pos_4 : table ⟨2, 2⟩ = 4)
  (adj_5 : (∑ i in ({⟨1, 0⟩, ⟨1, 2⟩, ⟨0, 1⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 9) :
  (∑ i in ({⟨0, 1⟩, ⟨1, 0⟩, ⟨1, 2⟩, ⟨2, 1⟩} : Finset (Fin 3 × Fin 3)), table i) = 29 :=
by
  sorry

end sum_adjacent_cells_of_6_is_29_l367_367844


namespace distance_to_other_focus_l367_367442

-- Define the ellipse equation and relevant parameters
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 25) + y^2 = 1

-- Define the condition that point P is at a distance of 6 from one focus
def distance_from_focus_1 (P : ℝ × ℝ) (focus1 : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (fx, fy) := focus1
  (px - fx)^2 + (py - fy)^2 = 6^2

-- In this case, assume the focus1 to be at (c, 0)
def focus1 : ℝ × ℝ := (sqrt (25 - 1), 0)

-- The main theorem to prove
theorem distance_to_other_focus (P : ℝ × ℝ) :
  ellipse P.1 P.2 → distance_from_focus_1 P focus1 → 
  let focus2 := (-sqrt (25 - 1), 0)
  sqrt ((P.1 - focus2.1)^2 + (P.2 - focus2.2)^2) = 4 :=
by
  sorry

end distance_to_other_focus_l367_367442


namespace distinct_pawn_placements_on_chess_board_l367_367042

def numPawnPlacements : ℕ :=
  5! * 5!

theorem distinct_pawn_placements_on_chess_board :
  numPawnPlacements = 14400 := by
  sorry

end distinct_pawn_placements_on_chess_board_l367_367042


namespace cylinder_sphere_surface_area_ratio_l367_367044

theorem cylinder_sphere_surface_area_ratio 
  (d : ℝ) -- d represents the diameter of the sphere and the height of the cylinder
  (S1 S2 : ℝ) -- Surface areas of the cylinder and the sphere
  (r := d / 2) -- radius of the sphere
  (S1 := 6 * π * r ^ 2) -- surface area of the cylinder
  (S2 := 4 * π * r ^ 2) -- surface area of the sphere
  : S1 / S2 = 3 / 2 :=
  sorry

end cylinder_sphere_surface_area_ratio_l367_367044


namespace problem_solution_l367_367469

noncomputable theory

open_locale real

def rectangular_eq_circle (x y : ℝ) : Prop := (x - sqrt 3) ^ 2 + (y - 1) ^ 2 = 4

def polar_eq_line (θ : ℝ) : Prop := θ = π / 3

def polar_eq_circle (ρ θ : ℝ) : Prop := ρ = 4 * sin (θ + π / 3)

def area_triangle_C_M_N : ℝ := sqrt 3

theorem problem_solution :
  (∀ x y : ℝ, rectangular_eq_circle x y ↔ ∃ ρ θ, polar_eq_circle ρ θ) ∧ 
  (∃ M N : ℝ × ℝ, polar_eq_line (θ : ℝ) → (area_triangle_C_M_N = sqrt 3)) :=
sorry

end problem_solution_l367_367469


namespace intersection_M_N_l367_367419

variable (U : Set ℤ) (M N : Set ℤ)

def U_def : U = {-1, 0, 1, 2, 3, 4} := by sorry
def M_complement : compl U M = {-1, 1} := by sorry
def N_def : N = {0, 1, 2, 3} := by sorry

theorem intersection_M_N : M ∩ N = {0, 2, 3} := by
  have hU : U = {-1, 0, 1, 2, 3, 4} := U_def
  have hM : compl U M = {-1, 1} := M_complement
  have hN : N = {0, 1, 2, 3} := N_def
  sorry

end intersection_M_N_l367_367419


namespace fraction_subtraction_l367_367330

theorem fraction_subtraction : (9 / 23) - (5 / 69) = 22 / 69 :=
by
  sorry

end fraction_subtraction_l367_367330


namespace sum_adjacent_6_is_29_l367_367839

-- Define the grid and the placement of numbers 1 to 4
structure Grid :=
  (grid : Fin 3 → Fin 3 → Nat)
  (h_unique : ∀ i j, grid i j ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_distinct : Function.Injective (λ (i : Fin 3) (j : Fin 3), grid i j))
  (h_placement : grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧ grid 2 2 = 4)

-- Define the condition of the sum of numbers adjacent to 5 being 9
def sum_adjacent_5 (g : Grid) : Prop :=
  let (i, j) := (0, 1) in -- Position for number 5
  (g.grid (i.succ) j + g.grid (i.succ.pred) j + g.grid i (j.succ) + g.grid i (j.pred)) = 9

-- Define the main theorem
theorem sum_adjacent_6_is_29 (g : Grid) (h_sum_adj_5 : sum_adjacent_5 g) : 
  (g.grid 1 0 + g.grid 1 2 + g.grid 0 1 + g.grid 2 1 = 29) := sorry

end sum_adjacent_6_is_29_l367_367839


namespace sum_of_products_eq_sum_of_fractions_l367_367125

theorem sum_of_products_eq_sum_of_fractions :
  (∑ k in finset.range 2019, (2019 - k) * k) = (∑ k in finset.range 2018, (2019 - k) * (2019 - k - 1) / 2) := 
  sorry

end sum_of_products_eq_sum_of_fractions_l367_367125


namespace dogs_not_eat_either_l367_367058

-- Define the given conditions as constants or variables
constant total_dogs : ℕ -- Total number of dogs
constant dogs_like_watermelon : ℕ -- Number of dogs liking watermelon
constant dogs_like_salmon : ℕ -- Number of dogs liking salmon
constant dogs_like_both : ℕ -- Number of dogs liking both watermelon and salmon

-- Assign the given values
axiom h_total_dogs : total_dogs = 80
axiom h_dogs_like_watermelon : dogs_like_watermelon = 15
axiom h_dogs_like_salmon : dogs_like_salmon = 55
axiom h_dogs_like_both : dogs_like_both = 10

-- Prove the number of dogs that do not like either watermelon or salmon
theorem dogs_not_eat_either : total_dogs - (dogs_like_watermelon + dogs_like_salmon - dogs_like_both) = 20 :=
by
  rw [h_total_dogs, h_dogs_like_watermelon, h_dogs_like_salmon, h_dogs_like_both]
  rfl

end dogs_not_eat_either_l367_367058


namespace one_minus_repeat_three_l367_367322

theorem one_minus_repeat_three : 1 - (0.333333..<3̅) = 2 / 3 :=
by
  -- needs proof, currently left as sorry
  sorry

end one_minus_repeat_three_l367_367322


namespace cost_price_of_first_batch_min_selling_price_for_second_batch_l367_367509

-- Define the cost calculation for the first batch
def batch1_cost (x : ℝ) : ℝ := 1000 / x

-- Define the cost calculation for the second batch
def batch2_cost (x : ℝ) : ℝ := 2500 / (x + 0.5)

theorem cost_price_of_first_batch :
  ∃ x : ℝ, (batch1_cost x) * 2 = batch2_cost x ∧ x = 2 :=
begin
  sorry
end

-- Define the income from selling the first batch
def batch1_income (price_per_flower : ℝ) : ℝ := 1000 / 2 * (price_per_flower - 2)

-- Define the income from selling the second batch
def batch2_income (m : ℝ) : ℝ := 2500 / 2.5 * (m - 2.5)

-- Define the total profit condition
def total_profit (profit1 profit2 : ℝ) : Bool := profit1 + profit2 ≥ 1500

theorem min_selling_price_for_second_batch :
  ∀ m : ℝ, total_profit (batch1_income 3) (batch2_income m) = (m ≥ 3.5) :=
begin
  sorry
end

end cost_price_of_first_batch_min_selling_price_for_second_batch_l367_367509


namespace sin_double_angle_l367_367747

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 3 / 4) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end sin_double_angle_l367_367747


namespace max_x1_x2_square_is_18_l367_367774

noncomputable def max_x1_x2_square (k : ℝ) (h : ∃ (x1 x2 : ℝ), ∀ (x : ℝ), x^2 - (k - 2) * x + (k^2 + 3 * k + 5) = 0 → x = x1 ∨ x = x2) : ℝ :=
  let x1 := sorry
  let x2 := sorry
  x1^2 + x2^2

theorem max_x1_x2_square_is_18 : ∀ (k : ℝ), (∃ (x1 x2 : ℝ), ∀ (x : ℝ), x^2 - (k - 2) * x + (k^2 + 3 * k + 5) = 0 → x = x1 ∨ x = x2) → max_x1_x2_square k = 18 :=
  by sorry

end max_x1_x2_square_is_18_l367_367774


namespace probability_A_wins_championship_distribution_and_expectation_B_l367_367963

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l367_367963


namespace tangent_line_at_P_l367_367994

noncomputable def f (x : ℝ) : ℝ := 1 / x - Real.sqrt x
def point_P : ℝ × ℝ := (4, -7 / 4)
def tangent_line (x y : ℝ) : Prop := 5 * x + 16 * y + 8 = 0

theorem tangent_line_at_P :
  tangent_line point_P.1 (f point_P.1 + ((-1 / 16 - 1 / 4) * (point_P.1 - 4))) ↔
  tangent_line point_P.1 point_P.2 :=
sorry

end tangent_line_at_P_l367_367994


namespace probability_A_wins_championship_expectation_X_is_13_l367_367965

/-
Definitions corresponding to the conditions in the problem
-/
def prob_event1_A_win : ℝ := 0.5
def prob_event2_A_win : ℝ := 0.4
def prob_event3_A_win : ℝ := 0.8

def prob_event1_B_win : ℝ := 1 - prob_event1_A_win
def prob_event2_B_win : ℝ := 1 - prob_event2_A_win
def prob_event3_B_win : ℝ := 1 - prob_event3_A_win

/-
Proof problems corresponding to the questions and correct answers
-/

theorem probability_A_wins_championship : prob_event1_A_win * prob_event2_A_win * prob_event3_A_win
    + prob_event1_A_win * prob_event2_A_win * prob_event3_B_win
    + prob_event1_A_win * prob_event2_B_win * prob_event3_A_win 
    + prob_event1_B_win * prob_event2_A_win * prob_event3_A_win = 0.6 := 
sorry

noncomputable def X_distribution_table : list (ℝ × ℝ) := 
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

noncomputable def expected_value_X : ℝ := 
  ∑ x in X_distribution_table, x.1 * x.2

theorem expectation_X_is_13 : expected_value_X = 13 := sorry

end probability_A_wins_championship_expectation_X_is_13_l367_367965


namespace velocity_of_point_C_l367_367372

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l367_367372


namespace slope_of_line_l367_367733

theorem slope_of_line : ∀ (x y : ℝ), (4 * x + 7 * y = 28) → ∃ m b : ℝ, (-4 / 7 = m) ∧ (y = m * x + b) :=
by
  intros x y h
  use [-4 / 7, 4]
  split
  · refl
  sorry

end slope_of_line_l367_367733


namespace least_three_digit_multiple_of_3_4_9_is_108_l367_367585

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l367_367585


namespace sum_of_adjacent_to_6_l367_367864

theorem sum_of_adjacent_to_6 :
  ∃ (grid : Fin 3 × Fin 3 → ℕ),
  (grid (0, 0) = 1 ∧ grid (0, 2) = 3 ∧ grid (2, 0) = 2 ∧ grid (2, 2) = 4 ∧
   ∀ i j, grid (i, j) ∈ finset.range 1 10 ∧ finset.univ.card = 9 ∧
   (grid (1, 0) + grid (1, 1) + grid (2, 1) = 9) ∧ 
   (grid (1, 1) = 6) ∧ 
   (sum_of_adjacent grid (1, 1) = 29))

where
  sum_of_adjacent (grid : Fin 3 × Fin 3 → ℕ) (x y : Fin 3 × Fin 3) : ℕ :=
  grid (x - 1, y) + grid (x + 1, y) + grid (x, y - 1) + grid (x, y + 1)
  sorry

end sum_of_adjacent_to_6_l367_367864


namespace number_of_relatively_prime_integers_less_than_200_count_relatively_prime_to_15_or_24_less_than_200_l367_367817

theorem number_of_relatively_prime_integers_less_than_200 (n : ℕ) (h : n < 200) :
  (n % 3 ≠ 0 ∧ n % 5 ≠ 0) ∨ (n % 2 ≠ 0 ∧ n % 3 ≠ 0) := sorry

theorem count_relatively_prime_to_15_or_24_less_than_200 :
  finset.card (finset.filter (λ n, (n % 3 ≠ 0 ∧ n % 5 ≠ 0) ∨ (n % 2 ≠ 0 ∧ n % 3 ≠ 0)) (finset.range 200)) = 120 := sorry

end number_of_relatively_prime_integers_less_than_200_count_relatively_prime_to_15_or_24_less_than_200_l367_367817


namespace unique_positive_real_solution_l367_367436

def f (x : ℝ) := x^11 + 5 * x^10 + 20 * x^9 + 1000 * x^8 - 800 * x^7

theorem unique_positive_real_solution :
  ∃! (x : ℝ), 0 < x ∧ f x = 0 :=
sorry

end unique_positive_real_solution_l367_367436


namespace distinct_pawn_placements_on_chess_board_l367_367039

def numPawnPlacements : ℕ :=
  5! * 5!

theorem distinct_pawn_placements_on_chess_board :
  numPawnPlacements = 14400 := by
  sorry

end distinct_pawn_placements_on_chess_board_l367_367039


namespace diamonds_in_G8_l367_367882

noncomputable def diamonds_in_design : ℕ → ℕ
| 1     := 3
| (n+1) := diamonds_in_design n + (n + 2) * 4

theorem diamonds_in_G8 : diamonds_in_design 8 = 195 :=
by
  sorry

end diamonds_in_G8_l367_367882


namespace sum_of_arithmetic_sequence_l367_367795

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : a 5 + a 4 = 18) (hS_def : ∀ n, S n = n * (a 1 + a n) / 2) : S 8 = 72 := 
sorry

end sum_of_arithmetic_sequence_l367_367795


namespace P1Q1_parallel_AB_l367_367483

open EuclideanGeometry

variable {A B C C1 M P Q P1 Q1 : Point}

-- Conditions
axiom M_on_median_cc1 : OnLine M (Line.mk C C1)
axiom P_midpoint_ma : Midpoint P M A
axiom Q_midpoint_mb : Midpoint Q M B
axiom P1_intersect_c1p_ca : ∃ X, Collinear X C C1 ∧ Collinear X P P1
axiom Q1_intersect_c1q_cb : ∃ Y, Collinear Y C C1 ∧ Collinear Y Q Q1

-- Theorem to prove
theorem P1Q1_parallel_AB : Parallel (Line.mk P1 Q1) (Line.mk A B) :=
by sorry

end P1Q1_parallel_AB_l367_367483


namespace find_whole_wheat_pastry_flour_l367_367956

variable (x : ℕ) -- where x is the pounds of whole-wheat pastry flour Sarah already had

-- Conditions
def rye_flour := 5
def whole_wheat_bread_flour := 10
def chickpea_flour := 3
def total_flour := 20

-- Total flour bought
def total_flour_bought := rye_flour + whole_wheat_bread_flour + chickpea_flour

-- Proof statement
theorem find_whole_wheat_pastry_flour (h : total_flour = total_flour_bought + x) : x = 2 :=
by
  -- The proof is omitted
  sorry

end find_whole_wheat_pastry_flour_l367_367956


namespace slope_of_line_l367_367728

theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (∃ m b : ℝ, y = m * x + b ∧ m = -4/7) :=
by
   intro h
   use -4/7, 28/7
   rw [mul_comm 7 y, ←sub_eq_iff_eq_add]
   simp [eq_sub_of_add_eq, h]
   split
   { sorry }
   { refl }

end slope_of_line_l367_367728


namespace geometric_sum_l367_367075

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 6) (h3 : a 3 = -18) :
  a 1 + a 2 + a 3 + a 4 = 40 :=
sorry

end geometric_sum_l367_367075


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367971

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l367_367971


namespace find_m_values_l367_367806

theorem find_m_values (m : ℝ) :
  let A := {0, m, m^2 - 3 * m + 2} in 2 ∈ A ↔ m = 0 ∨ m = 2 ∨ m = 3 :=
by sorry

end find_m_values_l367_367806


namespace sum_of_cosines_l367_367313

theorem sum_of_cosines :
  (∑ k in Finset.range 19, Real.cos (↑(k + 1) * Real.pi / 10)) = -1 :=
sorry

end sum_of_cosines_l367_367313


namespace jo_thinking_greatest_integer_l367_367901

theorem jo_thinking_greatest_integer :
  ∃ n : ℕ, n < 150 ∧ 
           (∃ k : ℤ, n = 9 * k - 2) ∧ 
           (∃ m : ℤ, n = 11 * m - 4) ∧ 
           (∀ N : ℕ, (N < 150 ∧ 
                      (∃ K : ℤ, N = 9 * K - 2) ∧ 
                      (∃ M : ℤ, N = 11 * M - 4)) → N ≤ n) 
:= by
  sorry

end jo_thinking_greatest_integer_l367_367901


namespace all_tutors_work_together_in_90_days_l367_367933

theorem all_tutors_work_together_in_90_days :
  lcm 5 (lcm 6 (lcm 9 10)) = 90 := by
  sorry

end all_tutors_work_together_in_90_days_l367_367933


namespace value_of_b_range_of_a_l367_367800

/-- A function f defined as f(x) = x / ln x - a * x + b --/
def f (x a b : ℝ) : ℝ := x / real.log x - a * x + b

/-- Condition that the tangent line at (e, f(e)) is y = -a * x + 2 * e --/
def tangent_line_condition (a b : ℝ) : Prop :=
  2 * e = e - a * e + b

/-- Proving the value of b is e given the conditions --/
theorem value_of_b (a : ℝ) : tangent_line_condition a e :=
  sorry

/-- Property of existence of x in [e, e^2] such that f(x) <= 1/4 + e --/
def exists_x_in_interval (a b x : ℝ) : Prop :=
  x ∈ set.Icc e (e^2) ∧ f x a b ≤ 1 / 4 + e

/-- Proving that a is in the range [1/2 - 1/(4*e^2), +∞) given the condition --/
theorem range_of_a (b : ℝ) (h : ∃ x, exists_x_in_interval (1 / 2 - 1 / (4 * e^2)) b x) :
  ∀ a, a ≥ 1 / 2 - 1 / (4 * e^2) :=
  sorry

end value_of_b_range_of_a_l367_367800


namespace greatest_possible_number_of_students_who_neither_liked_swimming_nor_soccer_l367_367629

theorem greatest_possible_number_of_students_who_neither_liked_swimming_nor_soccer:
  (total_students = 60) →
  (students_liked_swimming = 33) →
  (students_liked_soccer = 36) →
  (24 ≤ total_students - (students_liked_swimming + students_liked_soccer)) :=
begin
  intros,
  sorry,
end

end greatest_possible_number_of_students_who_neither_liked_swimming_nor_soccer_l367_367629


namespace compute_pow_l367_367283

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end compute_pow_l367_367283


namespace distinct_pawns_5x5_l367_367023

theorem distinct_pawns_5x5 : 
  ∃ n : ℕ, n = 14400 ∧ 
  (∃ (get_pos : Fin 5 → Fin 5), function.bijective get_pos) :=
begin
  sorry
end

end distinct_pawns_5x5_l367_367023


namespace marissas_sunflower_height_l367_367110

def height_in_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

def sunflower_height (sister_height_inches : ℕ) (additional_height_inches : ℕ) : ℕ :=
  sister_height_inches + additional_height_inches

theorem marissas_sunflower_height :
  let sister_height := height_in_inches 4 3 in
  let additional_height := 21 in
  sunflower_height sister_height additional_height / 12 = 6 :=
by
  sorry

end marissas_sunflower_height_l367_367110


namespace shortest_path_A_to_B_l367_367275

noncomputable def shortest_path_length (A B : ℝ × ℝ) (is_white_area : ℝ × ℝ → Prop) : ℝ :=
  if is_white_area A ∧ is_white_area B 
  then 7 + 5 * Real.sqrt 2
  else 0  -- this case will not be used as per the conditions

theorem shortest_path_A_to_B (A B : ℝ × ℝ)
  (is_white_area : ℝ × ℝ → Prop)
  (avoid_shaded_area : ∀ x : ℝ × ℝ, ¬is_white_area x → False)
  : shortest_path_length A B is_white_area = 7 + 5 * Real.sqrt 2 :=
by
  sorry

end shortest_path_A_to_B_l367_367275


namespace fraction_white_surface_area_l367_367642

-- Definitions for conditions
def larger_cube_side : ℕ := 3
def smaller_cube_count : ℕ := 27
def white_cube_count : ℕ := 19
def black_cube_count : ℕ := 8
def black_corners : Nat := 8
def faces_per_cube : ℕ := 6
def exposed_faces_per_corner : ℕ := 3

-- Theorem statement for proving the fraction of the white surface area
theorem fraction_white_surface_area : (30 : ℚ) / 54 = 5 / 9 :=
by 
  -- Add the proof steps here if necessary
  sorry

end fraction_white_surface_area_l367_367642


namespace final_prices_l367_367258

noncomputable def hat_initial_price : ℝ := 15
noncomputable def hat_first_discount : ℝ := 0.20
noncomputable def hat_second_discount : ℝ := 0.40

noncomputable def gloves_initial_price : ℝ := 8
noncomputable def gloves_first_discount : ℝ := 0.25
noncomputable def gloves_second_discount : ℝ := 0.30

theorem final_prices :
  let hat_price_after_first_discount := hat_initial_price * (1 - hat_first_discount)
  let hat_final_price := hat_price_after_first_discount * (1 - hat_second_discount)
  let gloves_price_after_first_discount := gloves_initial_price * (1 - gloves_first_discount)
  let gloves_final_price := gloves_price_after_first_discount * (1 - gloves_second_discount)
  hat_final_price = 7.20 ∧ gloves_final_price = 4.20 :=
by
  sorry

end final_prices_l367_367258


namespace octagon_center_inside_circle_pentagon_center_inside_circle_n_gon_center_inside_circle_l367_367601

-- Definition of regular polygon and flipping
structure RegularPolygon (n : ℕ) :=
  (sides_eq : ∀ i j, (0 ≤ i ∧ i < n) ∧ (0 ≤ j ∧ j < n) → SideLength i = SideLength j)
  (interior_angle : RealAngle := (n - 2) * 180 / n)

-- Function that checks if flipping a polygon results in its center being inside the circle
def can_center_be_inside_circle (n : ℕ) (polygon : RegularPolygon n) (circle_radius : ℝ) : Prop :=
  ∃ flips : (ℕ → ℕ), center_inside_circle polygon flips circle_radius

-- Specification of the problem for octagon
theorem octagon_center_inside_circle (circle_radius : ℝ) :
  can_center_be_inside_circle 8 (RegularPolygon.mk (λ i j _, rfl) (by norm_num)) circle_radius :=
sorry

-- Specification of the problem for pentagon
theorem pentagon_center_inside_circle (circle_radius : ℝ) :
  can_center_be_inside_circle 5 (RegularPolygon.mk (λ i j _, rfl) (by norm_num)) circle_radius :=
sorry

-- General case for which polygons have the center inside the circle after flipping
theorem n_gon_center_inside_circle (n : ℕ) (h : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6) (circle_radius : ℝ) :
  can_center_be_inside_circle n (RegularPolygon.mk (λ i j _, rfl) (by norm_num)) circle_radius :=
sorry

end octagon_center_inside_circle_pentagon_center_inside_circle_n_gon_center_inside_circle_l367_367601


namespace problem1_problem2_l367_367670

-- Problem 1: Prove the expression equals 5
theorem problem1 : (1 : ℚ) * ((1/3 : ℚ) - (3/4) + (5/6)) / (1/12) = 5 := by
  sorry

-- Problem 2: Prove the expression equals 7
theorem problem2 : ((-1 : ℤ)^2023 + |(1 - 0.5 : ℚ)| * ((-4)^2)) = 7 := by
  sorry

end problem1_problem2_l367_367670


namespace sequence_ap_iff_odd_l367_367713

theorem sequence_ap_iff_odd (n : ℕ) (h : n > 2) :
  (∃ (a : Fin n → ℕ), (¬ ∀ i, a i = a 0) ∧ 
    (∃ d ≠ 0, ∀ i : Fin n, a i * a ((i : ℕ + 1) % (n : ℕ)) = 
      a (i : ℕ + 1) % (n : ℕ + 1) * a ((i : ℕ + 2) % (n : ℕ)) - d)) ↔ 
  (∃ k : ℕ, n = 2 * k + 1) :=
by
  sorry

end sequence_ap_iff_odd_l367_367713


namespace part1_part2_l367_367431

-- Conditions
def U := ℝ
def A : Set ℝ := {x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2}
def B (m : ℝ) : Set ℝ := {x | x ≤ 3 * m - 4 ∨ x ≥ 8 + m}
def complement_U (B : Set ℝ) : Set ℝ := {x | ¬(x ∈ B)}
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

-- Assertions
theorem part1 (m : ℝ) (h1 : m = 2) : intersection A (complement_U (B m)) = {x | 2 < x ∧ x < 4} :=
  sorry

theorem part2 (h : intersection A (complement_U (B m)) = ∅) : -4 ≤ m ∧ m ≤ 5 / 3 :=
  sorry

end part1_part2_l367_367431


namespace eccentricity_of_ellipse_ellipse_equation_range_of_k_l367_367295

-- Define the ellipse and its properties
variables a b c : ℝ 
variables (h_ab : a > b) (h_b0 : b > 0) 
variables (h_foci : c = b) (h_vertex : (c^2 - b^2 = 0))

-- Define the center of the circle and its condition
variables N : ℝ × ℝ := (0, 2)
variables (max_radius : ℝ := sqrt 26)
variables (circle_condition : ∀ x y, x^2 / (2 * b^2) + y^2 / b^2 = 1 → (x^2 + (y - 2)^2 ≤ 26))

-- Define the range for k
variables (k : ℝ)
variables (line_l : ¬(k = 0))

-- Prove the eccentricity
theorem eccentricity_of_ellipse : (b = c) → (a = sqrt 2 * c) → e = (c / a) :=
by
  intros
  rw [h_foci, mul_div_cancel, sqrt_eq_iff_sq_eq]
  -- Additional steps omitted
  sorry

-- Prove the equation of the ellipse given the conditions
theorem ellipse_equation : (b = 3) → (∀ x y, x^2 / 18 + y^2 / 9 = 1) :=
by 
  intros
  -- Additional steps omitted
  sorry

-- Prove the range of k for symmetric points
theorem range_of_k : (-∞ < k ∧ k < -1 / 2) ∨ (1 / 2 < k ∧ k < +∞) :=
by 
  intros
  -- Additional steps omitted
  sorry

end eccentricity_of_ellipse_ellipse_equation_range_of_k_l367_367295


namespace symmetric_points_y_axis_l367_367393

theorem symmetric_points_y_axis (a b : ℝ) (h1 : ∀ a b : ℝ, y_symmetric a (-3) 4 b ↔ a = -4 ∧ b = -3) : a + b = -7 :=
by
  have h2 := h1 a b
  cases h2 with ha hb,
  rw [ha, hb],
  norm_num

end symmetric_points_y_axis_l367_367393


namespace percent_less_than_l367_367048

-- Definitions based on the given conditions.
variable (y q w z : ℝ)
variable (h1 : w = 0.60 * q)
variable (h2 : q = 0.60 * y)
variable (h3 : z = 1.50 * w)

-- The theorem that the percentage by which z is less than y is 46%.
theorem percent_less_than (y q w z : ℝ) (h1 : w = 0.60 * q) (h2 : q = 0.60 * y) (h3 : z = 1.50 * w) :
  100 - (z / y * 100) = 46 :=
sorry

end percent_less_than_l367_367048


namespace triangle_dot_product_l367_367077

/-- Let $ABC$ be a triangle with $D$ being the midpoint of $BC$.
    If $AB = 2$, $BC = 3$, and $AC = 4$, then the dot product 
    $\overrightarrow{AD} \cdot \overrightarrow{AB}$ is $\frac{19}{4}$.
-/
theorem triangle_dot_product (A B C D : Point) 
  (hD : midpoint D B C) 
  (hAB : dist A B = 2) 
  (hBC : dist B C = 3) 
  (hAC : dist A C = 4) : 
  (AD • AB) = 19 / 4 :=
sorry

-- Definitions to make the code syntactically correct
structure Point :=
(x : ℝ) (y : ℝ)

noncomputable def midpoint (D B C : Point) : Prop :=
D = ⟨(B.x + C.x)/2, (B.y + C.y)/2⟩

noncomputable def dist (P Q : Point) :=
real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

noncomputable def dot_product (v w : Vector) := sorry -- Placeholder for the actual implementation

end triangle_dot_product_l367_367077


namespace plumber_salary_percentage_l367_367476

def salary_construction_worker : ℕ := 100
def salary_electrician : ℕ := 2 * salary_construction_worker
def total_salary_without_plumber : ℕ := 2 * salary_construction_worker + salary_electrician
def total_labor_cost : ℕ := 650
def salary_plumber : ℕ := total_labor_cost - total_salary_without_plumber
def percentage_salary_plumber_as_construction_worker (x y : ℕ) : ℕ := (x * 100) / y

theorem plumber_salary_percentage :
  percentage_salary_plumber_as_construction_worker salary_plumber salary_construction_worker = 250 :=
by 
  sorry

end plumber_salary_percentage_l367_367476


namespace solve_cos_2x_eq_cos_x_plus_sin_x_l367_367139

open Real

theorem solve_cos_2x_eq_cos_x_plus_sin_x :
  ∀ x : ℝ,
    (cos (2 * x) = cos x + sin x) ↔
    (∃ k : ℤ, x = k * π - π / 4) ∨ 
    (∃ k : ℤ, x = 2 * k * π) ∨
    (∃ k : ℤ, x = 2 * k * π - π / 2) := 
sorry

end solve_cos_2x_eq_cos_x_plus_sin_x_l367_367139


namespace cars_in_parking_lot_l367_367175

theorem cars_in_parking_lot (C : ℕ) (customers_per_car : ℕ) (total_purchases : ℕ) 
  (h1 : customers_per_car = 5)
  (h2 : total_purchases = 50)
  (h3 : C * customers_per_car = total_purchases) : 
  C = 10 := 
by
  sorry

end cars_in_parking_lot_l367_367175


namespace union_A_B_l367_367444

noncomputable def is_N_star (x : ℤ) : Prop :=
x > 0

def set_A : set ℤ := {x | is_N_star x ∧ -1 ≤ x ∧ x ≤ 2}
def set_B : set ℤ := {1, 2, 3}

theorem union_A_B :
  set_A ∪ set_B = {1, 2, 3} :=
by
  sorry

end union_A_B_l367_367444


namespace how_many_tuna_l367_367936

-- Definitions for conditions
variables (customers : ℕ) (weightPerTuna : ℕ) (weightPerCustomer : ℕ)
variables (unsatisfiedCustomers : ℕ)

-- Hypotheses based on the problem conditions
def conditions :=
  customers = 100 ∧
  weightPerTuna = 200 ∧
  weightPerCustomer = 25 ∧
  unsatisfiedCustomers = 20

-- Statement to prove how many tuna Mr. Ray needs
theorem how_many_tuna (h : conditions customers weightPerTuna weightPerCustomer unsatisfiedCustomers) : 
  ∃ n, n = 10 :=
by
  sorry

end how_many_tuna_l367_367936


namespace min_value_f_over_f_l367_367756

theorem min_value_f_over_f'_0 (a b c : ℝ) (h_f0 : f'(0) = b) (h_f_nonneg : ∀ x : ℝ, f(x) = a * x^2 + b * x + c → f(x) ≥ 0) (h_b_pos : b > 0) :
  (∃ a ≥ 0, a * 1^2 + b * 1 + c) / b = 1 :=
by
  sorry

end min_value_f_over_f_l367_367756


namespace number_of_even_subsets_l367_367950

open Finset

theorem number_of_even_subsets (n : ℕ) : 
  (card {s : Finset (Fin n) | s.card % 2 = 0} = 2^(n-1)) :=
by
  sorry

end number_of_even_subsets_l367_367950


namespace lighting_bulbs_in_four_moves_l367_367227

noncomputable def can_light_all_bulbs_in_four_moves : Prop :=
  ∃ (moves : List (Set (ℕ × ℕ))), moves.length = 4 ∧
  (∀ move ∈ moves, (move ⊆ set_of (λ p : ℕ × ℕ, p.1 < 4 ∧ p.2 < 5)) ∧ (move ≠ ∅)) ∧
  (⋃ move ∈ moves, move) = set_of (λ p : ℕ × ℕ, p.1 < 4 ∧ p.2 < 5)

theorem lighting_bulbs_in_four_moves : can_light_all_bulbs_in_four_moves :=
sorry

end lighting_bulbs_in_four_moves_l367_367227


namespace sum_adjacent_to_6_is_29_l367_367871

def in_grid (n : ℕ) (grid : ℕ → ℕ → ℕ) := ∃ i j, grid i j = n

def adjacent_sum (grid : ℕ → ℕ → ℕ) (i j : ℕ) : ℕ :=
  grid (i-1) j + grid (i+1) j + grid i (j-1) + grid i (j+1)

def grid := λ i j =>
  if i = 0 ∧ j = 0 then 1 else
  if i = 2 ∧ j = 0 then 2 else
  if i = 0 ∧ j = 2 then 3 else
  if i = 2 ∧ j = 2 then 4 else
  if i = 1 ∧ j = 1 then 6 else 0

lemma numbers_positions_adjacent_5 :
  grid 0 0 = 1 ∧ grid 2 0 = 2 ∧ grid 0 2 = 3 ∧
  grid 2 2 = 4 ∧ 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else 0 in
  adjacent_sum grid 1 0 = 1 + 2 + 6 :=
by sorry

theorem sum_adjacent_to_6_is_29 : 
  let grid := λ i j, if i = 0 ∧ j = 0 then 1 else
                     if i = 2 ∧ j = 0 then 2 else
                     if i = 0 ∧ j = 2 then 3 else
                     if i = 2 ∧ j = 2 then 4 else
                     if i = 1 ∧ j = 1 then 6 else
                     if i = 1 ∧ j = 0 then 5 else
                     if i = 0 ∧ j = 1 then 7 else
                     if i = 2 ∧ j = 1 then 8 else
                     if i = 1 ∧ j = 2 then 9 else 0 in
  adjacent_sum grid 1 1 = 5 + 7 + 8 + 9 :=
by sorry

end sum_adjacent_to_6_is_29_l367_367871


namespace area_ratio_of_concentric_circles_l367_367576

theorem area_ratio_of_concentric_circles (C1 C2 : ℝ) 
  (h1 : (60 / 360) * C1 = (30 / 360) * C2) : (C1 / C2)^2 = 1 / 4 := 
by 
  have h : C1 / C2 = 1 / 2 := by
    field_simp [h1]
  rw [h]
  norm_num

end area_ratio_of_concentric_circles_l367_367576


namespace original_price_sarees_l367_367558

theorem original_price_sarees (P : ℝ) (h : 0.85 * 0.80 * P = 272) : P = 400 :=
by
  sorry

end original_price_sarees_l367_367558


namespace negation_exists_implication_l367_367552

theorem negation_exists_implication (x : ℝ) : (¬ ∃ y > 0, y^2 - 2*y - 3 ≤ 0) ↔ ∀ y > 0, y^2 - 2*y - 3 > 0 :=
by
  sorry

end negation_exists_implication_l367_367552


namespace multiplication_cycles_l367_367226

theorem multiplication_cycles
  (p : ℕ) [Fact (Nat.Prime p)]
  (a : ZMod p) (h : a ≠ 0) :
  (∀ k : ℕ, ∃ n : ℕ, ((a : (ZMod p)) ^ k = a)) ∧
  (∀ b : ZMod p, b ≠ 0 → ((∃ k, b = a ^ k) → ∀ n m, (a ^ n = b) ↔ (a ^ m = b))) ∧
  (a ^ (p - 1) = 1) :=
sorry

end multiplication_cycles_l367_367226


namespace num_elements_intersection_l367_367398

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 3, 4}

theorem num_elements_intersection : Finset.card (A ∩ B) = 3 := by
  sorry

end num_elements_intersection_l367_367398


namespace number_of_students_test_paper_C_l367_367647

theorem number_of_students_test_paper_C (n_students : ℕ) (n_selected : ℕ)
  (numbers_assigned : ℕ → Prop)
  (arithmetic_seq_first : ℕ) (arithmetic_seq_diff : ℕ) :
  n_students = 800 → n_selected = 40 →
  (∀ x, numbers_assigned x ↔ 1 ≤ x ∧ x ≤ 800) →
  arithmetic_seq_first = 18 → arithmetic_seq_diff = 20 →
  (∃ n_paperA n_paperB n_paperC,
    n_paperA = (count_elements_in_range numbers_assigned 1 200) ∧ 
    n_paperB = (count_elements_in_range numbers_assigned 201 560) ∧ 
    n_paperC = (count_elements_in_range numbers_assigned 561 800) ∧
    n_paperC = 12)
  sorry

end number_of_students_test_paper_C_l367_367647


namespace sum_of_adjacent_to_6_l367_367862

theorem sum_of_adjacent_to_6 :
  ∃ (grid : Fin 3 × Fin 3 → ℕ),
  (grid (0, 0) = 1 ∧ grid (0, 2) = 3 ∧ grid (2, 0) = 2 ∧ grid (2, 2) = 4 ∧
   ∀ i j, grid (i, j) ∈ finset.range 1 10 ∧ finset.univ.card = 9 ∧
   (grid (1, 0) + grid (1, 1) + grid (2, 1) = 9) ∧ 
   (grid (1, 1) = 6) ∧ 
   (sum_of_adjacent grid (1, 1) = 29))

where
  sum_of_adjacent (grid : Fin 3 × Fin 3 → ℕ) (x y : Fin 3 × Fin 3) : ℕ :=
  grid (x - 1, y) + grid (x + 1, y) + grid (x, y - 1) + grid (x, y + 1)
  sorry

end sum_of_adjacent_to_6_l367_367862


namespace release_angle_for_max_range_l367_367526

-- Definitions based on given conditions
def sling_length : ℝ := 1
def rotation_frequency : ℝ := 3
def distance_to_goliath : ℝ := 200
def optimal_release_angle : ℝ := 45 * (π / 180)  -- converting degrees to radians

-- Prove that the required release angle to hit Goliath is 45 degrees
theorem release_angle_for_max_range:
  ∃ θ, θ = optimal_release_angle ∧ 
          (let v := 2 * π * sling_length * rotation_frequency in
          let vx := v * Real.cos θ in
          let time_of_flight := distance_to_goliath / vx in
          let vy := v * Real.sin θ in
          vy * time_of_flight - 0.5 * 9.81 * time_of_flight ^ 2 = 0) :=
sorry

end release_angle_for_max_range_l367_367526


namespace daughter_work_rate_daughter_work_time_l367_367251

variables (M D : ℚ)

def man_work_rate := 1 / 4
def combined_work_rate := 1 / 3

theorem daughter_work_rate :
  M = man_work_rate →
  M + D = combined_work_rate →
  D = 1 / 12 :=
by
  intros hM hCombined
  rw [hM] at hCombined
  linarith

theorem daughter_work_time :
  D = 1 / 12 →
  1 / D = 12 :=
by
  intros hD
  rw [hD]
  norm_num

end daughter_work_rate_daughter_work_time_l367_367251


namespace Q_share_of_profit_l367_367610

def P_investment : ℕ := 54000
def Q_investment : ℕ := 36000
def total_profit : ℕ := 18000

theorem Q_share_of_profit : Q_investment * total_profit / (P_investment + Q_investment) = 7200 := by
  sorry

end Q_share_of_profit_l367_367610


namespace percent_increase_is_25_l367_367605

-- Define the old and new charges conditions
def old_price : ℝ := 1
def old_transactions : ℕ := 5
def new_price : ℝ := 0.75
def new_transactions : ℕ := 3

-- Define the ratio function
def ratio (price : ℝ) (transactions : ℕ) : ℝ := price / transactions

-- Define the percentage increase function
def percentage_increase (old_ratio new_ratio : ℝ) : ℝ := ((new_ratio - old_ratio) / old_ratio) * 100

-- Define the ratios for old and new charges
def old_ratio := ratio old_price old_transactions
def new_ratio := ratio new_price new_transactions

-- State the theorem to prove that the percentage increase is 25%
theorem percent_increase_is_25 :
  percentage_increase old_ratio new_ratio = 25 := by
  sorry

end percent_increase_is_25_l367_367605


namespace binomial_not_divisible_by_11_count_l367_367289

/-- The number of positive integers n ≤ 1330 for which binomial(2 * n, n) is not divisible by 11 is 295. -/
theorem binomial_not_divisible_by_11_count : 
  { n : ℕ | n ≤ 1330 ∧ ∃ k : ℕ, binomial (2 * n) n = 11 * k + r } = 295 :=
by
  sorry

end binomial_not_divisible_by_11_count_l367_367289


namespace hyperbola_eccentricity_l367_367430

-- Definitions for the problem conditions
structure HyperbolaData where
  (a b c : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_focal_dist : 2 * c = 2 * a * real.sqrt 1 + (b^2 / a^2))

-- The main statement to prove the eccentricity
theorem hyperbola_eccentricity (data : HyperbolaData)
  (intersection_condition : ∃ x y, y = (real.sqrt 3 / 3) * (x + data.c) ∧ 
                                    x^2 / data.a^2 - y^2 / data.b^2 = 1)
  (angle_condition : ∃ P F1 F2, 
                      ∠P F2 F1 = 2 * ∠P F1 F2
                    ∧ |F1F2| = 2 * data.c 
                    ∧ P ∈ line y = (real.sqrt 3 / 3) * (x + data.c)) :
  data.c / data.a = real.sqrt 3 + 1 :=
by
  sorry

end hyperbola_eccentricity_l367_367430


namespace range_of_k_for_distinct_real_roots_l367_367742

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ k*x1^2 - 2*x1 - 1 = 0 ∧ k*x2^2 - 2*x2 - 1 = 0) → k > -1 ∧ k ≠ 0 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l367_367742


namespace rectangular_solid_diagonal_angles_l367_367060

theorem rectangular_solid_diagonal_angles (α β γ : ℝ) :
  ∀ (h : ∀ α β : ℝ, cos α ^ 2 + cos β ^ 2 = 1),
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 :=
by
  intro h
  -- Sorry to skip the proof
  sorry

end rectangular_solid_diagonal_angles_l367_367060


namespace actual_height_of_boy_is_236_l367_367532

-- Define the problem conditions
def average_height (n : ℕ) (avg : ℕ) := n * avg
def incorrect_total_height := average_height 35 180
def correct_total_height := average_height 35 178
def wrong_height := 166
def height_difference := incorrect_total_height - correct_total_height

-- Proving the actual height of the boy whose height was wrongly written
theorem actual_height_of_boy_is_236 : 
  wrong_height + height_difference = 236 := sorry

end actual_height_of_boy_is_236_l367_367532


namespace area_of_triangle_l367_367584

theorem area_of_triangle :
  let x1 := 0
  let y1 := 0
  let x2 := 2
  let y2 := 3
  let x3 := 6
  let y3 := 8
  1/2 * | x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) | = 17 := by
  sorry

end area_of_triangle_l367_367584


namespace coefficients_equal_l367_367149

theorem coefficients_equal (n : ℕ) (h : n ≥ 6) : 
  (n = 7) ↔ 
  (Nat.choose n 5 * 3 ^ 5 = Nat.choose n 6 * 3 ^ 6) := by
  sorry

end coefficients_equal_l367_367149


namespace part1_intersection_part2_range_of_a_l367_367009

noncomputable def setA (x a : ℝ) : Prop := abs (2 * x - a) ≥ 2
noncomputable def setB (x : ℝ) : Prop := 2 < 2 ^ x ∧ 2 ^ x < 8

theorem part1_intersection (a : ℝ) :
  a = 2 →
  (setA a x ∧ setB x) ↔ (-∞, 0] ∪ [2, +∞) ∧ (1, 3) :=
by sorry

theorem part2_range_of_a :
  (∃ x, setA x a ∧ setB x) →
  (∀ x, setB x → setA a x) ↔ (a ∈ (-∞, 0] ∪ [8, +∞)) :=
by sorry

end part1_intersection_part2_range_of_a_l367_367009


namespace simplify_exponentiation_l367_367522

-- Define the exponents and the base
variables (t : ℕ)

-- Define the expression and expected result
def expr := t^5 * t^2
def expected := t^7

-- State the proof goal
theorem simplify_exponentiation : expr = expected := 
by sorry

end simplify_exponentiation_l367_367522


namespace one_minus_repeating_three_l367_367328

theorem one_minus_repeating_three : 1 - (0.\overline{3}) = 2 / 3 :=
by
  sorry

end one_minus_repeating_three_l367_367328


namespace length_of_SV_l367_367538

noncomputable def length_SV (P Q R S W U V T : Type)
  (side : ℝ) (center : (P Q R S W)) (midU : U) (area_split : ℝ) 
  (base1 base2 height : ℝ) : ℝ :=
by
  let base1 := sorry -- Definition for SV
  let base2 := side / 2
  let height := side / 2
  -- Assert the area for one of the trapezium regions
  have area_SVTW := side^2 / 3
  have trapezium_area := (1 / 2) * (base1 + base2) * height
  exact (trapezium_area = area_SVTW) → base1 = (5/6)

theorem length_of_SV (P Q R S W U V T : Type)
  (side : ℝ) (center : (P = center Q R S W)) (midU : U = midpoint R S) 
  (area_split : side^2 / 3) : 
  ∃ (SV: ℝ), SV = (5 / 6) :=
by
  let base1 := sorry -- Definition for SV
  have h2 : base1 = (5/6) := length_SV (P Q R S W U V T) side center midU area_split base1 (side / 2) (side / 2)
  exact ⟨base1, h2⟩

end length_of_SV_l367_367538
