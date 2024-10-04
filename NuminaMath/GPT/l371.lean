import Mathlib

namespace nondecreasing_seq_count_l371_371530

-- Defining the requirements for the sequence
def is_nondecreasing (a : ℕ → ℕ) : Prop :=
  ∀ i j, i ≤ j → a i ≤ a j

-- Define number set condition
def valid_number_set (a : ℕ → ℕ) : Prop :=
  ∀ i, a i ∈ {n | 1 ≤ n ∧ n ≤ 9 }

-- The main theorem statement
theorem nondecreasing_seq_count :
  let S := {a : Fin 10 → ℕ | is_nondecreasing a ∧ valid_number_set a ∧ (∃ s t u, {a 0, a 1, a 2, a 3, a 4, a 5, a 6, a 7, a 8, a 9} ⊆ {s, t, u})} in
  Fintype.card S = 3357 := by
  sorry

end nondecreasing_seq_count_l371_371530


namespace trigonometric_identity_proof_l371_371534

variable (α : ℝ)

theorem trigonometric_identity_proof
  (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos (2 * α) + 2 * Real.sin (2 * α) = 1 :=
by
  sorry

end trigonometric_identity_proof_l371_371534


namespace three_digit_number_multiple_of_3_prob_l371_371750

def digits := {1, 3, 5, 7}

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def number_of_valid_three_digit_numbers : ℕ :=
  let combinations := { (a, b, c) // a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c };
  let valid_combinations := { x ∈ combinations | is_multiple_of_3 (x.val.1 + x.val.2 + x.val.3) };
  valid_combinations.card * 6 -- 6 permutations of each combination

def total_number_of_three_digit_numbers : ℕ := 24 -- 4 * 3 * 2

theorem three_digit_number_multiple_of_3_prob :
  (number_of_valid_three_digit_numbers : ℚ) / (total_number_of_three_digit_numbers : ℚ) = 1 / 4 :=
by sorry

end three_digit_number_multiple_of_3_prob_l371_371750


namespace ellipse_semimajor_axis_value_l371_371990

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371990


namespace cost_per_use_l371_371959

def cost : ℕ := 30
def uses_in_a_week : ℕ := 3
def weeks : ℕ := 2
def total_uses : ℕ := uses_in_a_week * weeks

theorem cost_per_use : cost / total_uses = 5 := by
  sorry

end cost_per_use_l371_371959


namespace cos_theta_tetrahedron_tetrahedron_edge_lengths_volume_ineq_tetrahedron_perpendicular_edges_l371_371931

-- Problem 1
theorem cos_theta_tetrahedron (m n p q u v : ℝ) (h_m : m > 0) (h_n : n > 0) (theta : ℝ) :
    let cosine := ((p^2 + q^2) - (u^2 + v^2)) / (2 * m * n)
    in Math.cos theta = cosine := 
sorry

-- Problem 2
theorem tetrahedron_edge_lengths_volume_ineq (a b c d e f V : ℝ) (hV : V > 0) :
    (a^3 + b^3 + c^3 + d^3 + e^3 + f^3) ≥ 36 * Real.sqrt 2 * V :=
sorry

-- Problem 3
theorem tetrahedron_perpendicular_edges (A B C D : ℝ) (AB CD AC BD : Bool)
  (h₁ : AB = true) (h₂ : CD = true) (h₃ : AC = true) :
    (BD = true → BC = true) :=
sorry

end cos_theta_tetrahedron_tetrahedron_edge_lengths_volume_ineq_tetrahedron_perpendicular_edges_l371_371931


namespace polynomial_sum_l371_371666

theorem polynomial_sum :
  let f := (x^3 + 9*x^2 + 26*x + 24) 
  let g := (x + 3)
  let A := 1
  let B := 6
  let C := 8
  let D := -3
  (y = f/g) → (A + B + C + D = 12) :=
by 
  sorry

end polynomial_sum_l371_371666


namespace max_height_of_projectile_l371_371395

def projectile_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_of_projectile : 
  ∃ t : ℝ, projectile_height t = 161 :=
sorry

end max_height_of_projectile_l371_371395


namespace max_minus_min_l371_371078

noncomputable def f (x : ℝ) := if x > 0 then (x - 1) ^ 2 else (x + 1) ^ 2

theorem max_minus_min (n m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ (-1 / 2) → n ≤ f x ∧ f x ≤ m) →
  m - n = 1 :=
by { sorry }

end max_minus_min_l371_371078


namespace area_of_rectangle_l371_371432

theorem area_of_rectangle (s : ℝ) (EFGH : Type) [is_rectangle EFGH] :
  (∃ (circle : Type) [is_tangent_to_sides circle EFGH] [has_radius circle s]
  [passes_through_midpoint_of_diagonal circle EFGH], 
  area EFGH = 2 * s ^ 2) :=
sorry

end area_of_rectangle_l371_371432


namespace not_divisible_by_5_l371_371638

theorem not_divisible_by_5 (n : ℤ) : ¬ (n^2 - 8) % 5 = 0 :=
by sorry

end not_divisible_by_5_l371_371638


namespace probability_picasso_consecutive_l371_371620

-- Given Conditions
def total_pieces : Nat := 12
def picasso_paintings : Nat := 4

-- Desired probability calculation
theorem probability_picasso_consecutive :
  (Nat.factorial (total_pieces - picasso_paintings + 1) * Nat.factorial picasso_paintings) / 
  Nat.factorial total_pieces = 1 / 55 :=
by
  sorry

end probability_picasso_consecutive_l371_371620


namespace parabola_symmetry_l371_371861

theorem parabola_symmetry (a h m : ℝ) (A_on_parabola : 4 = a * (-1 - 3)^2 + h) (B_on_parabola : 4 = a * (m - 3)^2 + h) : 
  m = 7 :=
by 
  sorry

end parabola_symmetry_l371_371861


namespace julia_kids_l371_371960

theorem julia_kids :
  ∃ m t w h : ℕ,
    t = 14 ∧
    w = t + Nat.ceil (0.25 * 14) ∧
    h = 2 * w - 4 ∧
    m = t + 8 ∧
    m = 22 ∧
    t = 14 ∧
    w = 18 ∧
    h = 32 :=
by
  use [22, 14, 18, 32]
  split; refl
  split;
  calc
    18 = 14 + 4 : by sorry⟩ -- rounding 3.5 up is considered directly in the given problem
  split;
  calc
    32 = 2 * 18 - 4 : by sorry
  rfl

end julia_kids_l371_371960


namespace book_count_l371_371623

theorem book_count : 
  let initial_books := 54 in
  let books_after_giving_away := initial_books - 16 in
  let books_after_receiving := books_after_giving_away + 23 in
  let books_after_trade := books_after_receiving - 12 + 9 in
  let final_books := books_after_trade + 35 in
  final_books = 93 :=
by
  sorry

end book_count_l371_371623


namespace jerry_cases_l371_371199

def records_per_shelf (records_per_shelf_capacity : ℕ) (shelf_fill_rate : ℕ) : ℕ := 
  (shelf_fill_rate * records_per_shelf_capacity) / 100

def records_per_case (shelves_per_case : ℕ) (records_per_shelf : ℕ) : ℕ :=
  shelves_per_case * records_per_shelf

def ridges_per_case (records_per_case : ℕ) (ridges_per_record : ℕ) : ℕ :=
  records_per_case * ridges_per_record

def cases (total_ridges : ℕ) (ridges_per_case : ℕ) : ℕ :=
  total_ridges / ridges_per_case

theorem jerry_cases 
  (cases : ℕ) (shelves_per_case records_per_shelf_capacity ridges_per_record shelf_fill_rate total_ridges : ℕ)
  (h1 : shelves_per_case = 3)
  (h2 : records_per_shelf_capacity = 20)
  (h3 : ridges_per_record = 60)
  (h4 : shelf_fill_rate = 60)
  (h5 : total_ridges = 8640)
  (h6 : cases = 4) : 
  cases = cases total_ridges 
    (ridges_per_case 
      (records_per_case shelves_per_case 
        (records_per_shelf records_per_shelf_capacity shelf_fill_rate)) 
      ridges_per_record) :=
by
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end jerry_cases_l371_371199


namespace num_possible_values_l371_371253

-- Definitions based on conditions
def A : Set ℕ := sorry
def B : Set ℕ := sorry
def C : Set ℕ := sorry

axiom h1 : |A| = 92
axiom h2 : |B| = 35
axiom h3 : |C| = 63
axiom h4 : |A ∩ B| = 16
axiom h5 : |A ∩ C| = 51
axiom h6 : |B ∩ C| = 19

theorem num_possible_values :
  let x := |A ∩ B ∩ C| in
  7 ≤ x ∧ x ≤ 16 → (finset.range (16 + 1 - 7)).card = 10 :=
begin
  intros,
  sorry
end

end num_possible_values_l371_371253


namespace balls_picked_at_random_l371_371368

theorem balls_picked_at_random (n : ℕ) 
  (red_balls : ℕ := 3) (blue_balls : ℕ := 2) (green_balls : ℕ := 4) 
  (total_balls : ℕ := red_balls + blue_balls + green_balls) 
  (prob_red : ℚ := 1 / 12) :
  n = 2 :=
by
  have : total_balls = 9 := by norm_num
  have fact : ∀ n, n.factorial = nat.factorial n := by norm_num
  have comb (n k : ℕ) := nat.choose n k
  have comb_3_2 : comb 3 2 = 3 := by norm_num [comb]
  have comb_9_n := (nat.choose 9 n).to_rat / (nat.factorial n * nat.factorial (9 - n)).to_rat
  
  have prob_eq : (comb 3 2) / (comb 9 n) = prob_red := sorry
  have comb_9_2 : comb 9 2 = 36 := by norm_num [comb]
  have solution : n = 2 := by linarith
  
  exact solution

end balls_picked_at_random_l371_371368


namespace series_fraction_equals_2021_l371_371807

theorem series_fraction_equals_2021 :
  (∑ k in Finset.range 2020, (2021 - (k+1) : ℝ) / (k+1)) / (∑ k in Finset.range 2020, (1 : ℝ) / (k+2)) = 2021 := 
by
  sorry

end series_fraction_equals_2021_l371_371807


namespace unit_vector_perpendicular_to_a_l371_371051

theorem unit_vector_perpendicular_to_a :
  ∃ (m n : ℝ), 2 * m + n = 0 ∧ m^2 + n^2 = 1 ∧ m = sqrt 5 / 5 ∧ n = -2 * sqrt 5 / 5 :=
by
  sorry

end unit_vector_perpendicular_to_a_l371_371051


namespace squares_are_equal_l371_371468

theorem squares_are_equal (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
    (h₄ : a * (b + c + d) = b * (a + c + d)) 
    (h₅ : a * (b + c + d) = c * (a + b + d)) 
    (h₆ : a * (b + c + d) = d * (a + b + c)) : 
    a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := 
by
  sorry

end squares_are_equal_l371_371468


namespace ice_cream_cost_l371_371758

theorem ice_cream_cost
  (num_pennies : ℕ) (num_nickels : ℕ) (num_dimes : ℕ) (num_quarters : ℕ) 
  (leftover_cents : ℤ) (num_family_members : ℕ)
  (h_pennies : num_pennies = 123)
  (h_nickels : num_nickels = 85)
  (h_dimes : num_dimes = 35)
  (h_quarters : num_quarters = 26)
  (h_leftover : leftover_cents = 48)
  (h_members : num_family_members = 5) :
  (123 * 0.01 + 85 * 0.05 + 35 * 0.1 + 26 * 0.25 - 0.48) / 5 = 3 :=
by
  sorry

end ice_cream_cost_l371_371758


namespace triangle_DGH_is_isosceles_l371_371596

open EuclideanGeometry

variables (A B C D G H : Point) (Aacute : angle A < π / 2)
variable (h_parallelogram : parallelogram A B C D)
variable (h_G_on_AB : G ≠ B ∧ on_line G A B)
variable (h_CG_eq_CB : dist C G = dist C B)
variable (h_H_on_BC : H ≠ B ∧ on_line H B C)
variable (h_AB_eq_AH : dist A B = dist A H)

theorem triangle_DGH_is_isosceles :
  isosceles_triangle D G H :=
by
  sorry

end triangle_DGH_is_isosceles_l371_371596


namespace FermatLittleTheoremExample_remainder_2_100_mod_101_l371_371831

theorem FermatLittleTheoremExample (a : ℤ) (p : ℕ) (hp_prime : Nat.Prime p) (ha_not_div_p : ¬ (p ∣ a)) : 
  a ^ (p - 1) % p = 1 :=
by sorry

theorem remainder_2_100_mod_101 :
  2 ^ 100 % 101 = 1 :=
by
  have hp := Nat.Prime.prime_101
  apply FermatLittleTheoremExample 2 101 hp
  simp
  sorry

end FermatLittleTheoremExample_remainder_2_100_mod_101_l371_371831


namespace totalFourOfAKindCombinations_l371_371377

noncomputable def numberOfFourOfAKindCombinations : Nat :=
  13 * 48

theorem totalFourOfAKindCombinations : numberOfFourOfAKindCombinations = 624 := by
  sorry

end totalFourOfAKindCombinations_l371_371377


namespace largest_rectangle_area_l371_371689

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l371_371689


namespace find_y_l371_371860

def P : ℝ × ℝ := (-2, 7)
def Q (y : ℝ) : ℝ × ℝ := (4, y)
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)
def expected_slope : ℝ := -3 / 2

theorem find_y (y : ℝ) (h : slope P (Q y) = expected_slope) : y = -2 := 
sorry

end find_y_l371_371860


namespace sum_third_and_seventh_is_22_l371_371794

def seq : ℕ → ℕ
| 1       := 1
| (n + 1) := (n + 1) ^ 2 + 2 * (n + 1) - n ^ 2 - 2 * n

theorem sum_third_and_seventh_is_22 : seq 3 + seq 7 = 22 := by
  sorry

end sum_third_and_seventh_is_22_l371_371794


namespace find_x_l371_371921

theorem find_x (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 :=
sorry

end find_x_l371_371921


namespace calculate_expression_l371_371424

theorem calculate_expression : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end calculate_expression_l371_371424


namespace exists_root_in_interval_l371_371275

noncomputable def f (x : ℝ) : ℝ := x^2 - 1/x - 1

theorem exists_root_in_interval : 
  ∃ k : ℕ, k = 1 ∧ ∃ c : ℝ, k < c ∧ c < k + 1 ∧ f c = 0 := 
by
  use 1
  use 1.5
  simp [f]
  sorry

end exists_root_in_interval_l371_371275


namespace largest_rectangle_area_l371_371687

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l371_371687


namespace radius_of_third_circle_l371_371299

theorem radius_of_third_circle (r1 r2 : ℝ) (h1 : r1 = 24) (h2 : r2 = 36) :
  ∃ r : ℝ, π * r^2 = π * (r2^2 - r1^2) ∧ r = 12 * real.sqrt 5 :=
by
  sorry

end radius_of_third_circle_l371_371299


namespace problem_part1_problem_part2_l371_371867

noncomputable def a_n (n : ℕ) : ℕ := 2 ^ (n - 1)
noncomputable def S_n (n : ℕ) : ℕ := (finset.range (n + 1)).sum a_n
noncomputable def arithmetic_sequence (S_n : ℕ → ℕ) (a : ℤ) : Prop := 
  ∀ n : ℕ, n > 0 → 2 * S_n n = 2 ^ (n + 1) + a

theorem problem_part1 :
  (∀ n > 0, 2 * S_n n = 2 ^ (n + 1) + (-2)) ∧
  (∀ n : ℕ, n > 0 → a_n n = 2 ^ (n - 1)) :=
by
  sorry

noncomputable def b_n (n : ℤ) : ℤ := 
  (1 - (-2) * n) * (Int.log2 (b_n n * b_n (n + 1)))

noncomputable def T_n (n : ℤ) : ℤ :=
  (finset.range (n + 1)).sum (λi, (1 / b_n i))

theorem problem_part2 :
  ∀ n : ℤ, n > 0 → 
    T_n n = n / (2 * n + 1) :=
by
  sorry

end problem_part1_problem_part2_l371_371867


namespace quadratic_rewrite_l371_371137

theorem quadratic_rewrite (a b c x : ℤ) :
  (16 * x^2 - 40 * x - 72 = a^2 * x^2 + 2 * a * b * x + b^2 + c) →
  (a = 4 ∨ a = -4) →
  (2 * a * b = -40) →
  ab = -20 := by
sorry

end quadratic_rewrite_l371_371137


namespace possible_values_of_p_l371_371968

theorem possible_values_of_p (p q s : ℕ) (hy : ℕ) :
  prime p ∧ prime q ∧ prime s ∧ 2 ^ s * q = p ^ hy - 1 ∧ hy > 1 ↔ p = 3 ∨ p = 5 := 
sorry

end possible_values_of_p_l371_371968


namespace percent_increase_is_correct_l371_371416

noncomputable def first_triangle_side_length : ℝ := 4
noncomputable def growth_factor : ℝ := 1.2

def fifth_triangle_side_length : ℝ :=
  first_triangle_side_length * growth_factor^4

def first_triangle_perimeter : ℝ :=
  3 * first_triangle_side_length

def fifth_triangle_perimeter : ℝ :=
  3 * fifth_triangle_side_length

def percent_increase_perimeter : ℝ :=
  ((fifth_triangle_perimeter - first_triangle_perimeter) / first_triangle_perimeter) * 100

theorem percent_increase_is_correct :
  percent_increase_perimeter = 107.36 :=
by
  unfold percent_increase_perimeter first_triangle_perimeter fifth_triangle_perimeter fifth_triangle_side_length growth_factor first_triangle_side_length
  sorry

end percent_increase_is_correct_l371_371416


namespace extremum_of_f_on_M_l371_371525
noncomputable def f (x : ℝ) : ℝ := -x^2 - 6 * x + 1

def M : set ℝ := {x | x^2 + 4 * x ≤ 0}

theorem extremum_of_f_on_M :
  ∃ minval maxval, (minval = 1 ∧ maxval = 10) ∧ 
  (∀ x ∈ M, f x ≥ minval ∧ f x ≤ maxval) :=
sorry

end extremum_of_f_on_M_l371_371525


namespace eight_pow_15_div_sixtyfour_pow_6_l371_371316

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l371_371316


namespace sum_of_solutions_l371_371832

theorem sum_of_solutions (x : ℝ) (h : x + 16 / x = 12) : x = 8 ∨ x = 4 → 8 + 4 = 12 := by
  sorry

end sum_of_solutions_l371_371832


namespace largest_rectangle_area_l371_371690

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l371_371690


namespace infinitely_many_n_exists_l371_371249

theorem infinitely_many_n_exists (n : ℕ) : 
  (∃ m : ℕ, n = (m^2 + m + 2)^2 + (m^2 + m + 2) + 3) ∧ 
  (∀ p, prime p → p ∣ (n^2 + 3) → ∃ k : ℕ, k^2 < n ∧ p ∣ (k^2 + 3)) :=
sorry

end infinitely_many_n_exists_l371_371249


namespace inequality_solutions_l371_371981

theorem inequality_solutions (p p' q q' : ℕ) (hp : p ≠ p') (hq : q ≠ q') (hp_pos : 0 < p) (hp'_pos : 0 < p') (hq_pos : 0 < q) (hq'_pos : 0 < q') :
  (-(q : ℚ) / p > -(q' : ℚ) / p') ↔ (q * p' < p * q') :=
by
  sorry

end inequality_solutions_l371_371981


namespace log_a_eq_9_div_16_l371_371087

theorem log_a_eq_9_div_16 (a : ℝ) (h1 : 0 < a) (h2 : a ^ a = (9 * a) ^ (8 * a)) : Real.logBase a (3 * a) = 9 / 16 := by
  sorry

end log_a_eq_9_div_16_l371_371087


namespace inverse_of_B_squared_l371_371153

def B_inv : Matrix (Fin 2) (Fin 2) ℚ := ![
  (![1, 4] : Fin 2 → ℚ),
  ([-2, -7] : Fin 2 → ℚ)
]

theorem inverse_of_B_squared :
  (B_inv * B_inv) = ![
  ([-7, -24] : Fin 2 → ℚ),
  ([12, 41] : Fin 2 → ℚ)
] := by
  sorry

end inverse_of_B_squared_l371_371153


namespace verify_normal_distribution_properties_l371_371545

noncomputable def normal_distribution_properties (μ : ℝ) (σ : ℝ) (X : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, X x = (1 / (σ * Mathlib.Data.Real.Basic.sqrt (2 * Mathlib.Data.Real.Basic.pi))) * 
    Mathlib.Analysis.SpecialFunctions.Exp.exp (-(x - μ)^2 / (2 * σ^2))) →

  (∀ σ1 σ2 : ℝ, σ1 > 0 → σ2 > 0 → (σ1 < σ2 →
    (∀ x : ℝ, X x = (1 / (σ1 * Mathlib.Data.Real.Basic.sqrt (2 * Mathlib.Data.Real.Basic.pi))) *
    Mathlib.Analysis.SpecialFunctions.Exp.exp (-(x - μ)^2 / (2 * σ1^2))) → (X x is taller and thinner)) ∧
    (σ1 > σ2 →
    (∀ x : ℝ, X x = (1 / (σ2 * Mathlib.Data.Real.Basic.sqrt (2 * Mathlib.Data.Real.Basic.pi))) *
    Mathlib.Analysis.SpecialFunctions.Exp.exp (-(x - μ)^2 / (2 * σ^2))) → (X x is shorter and fatter))
  )

-- Theorem to verify the property of the normal distribution curve
theorem verify_normal_distribution_properties (μ : ℝ) :
  ∀ σ : ℝ, σ > 0 →
  normal_distribution_properties μ σ (λ x, 
    (1 / (σ * Mathlib.Data.Real.Basic.sqrt (2 * Mathlib.Data.Real.Basic.pi))) * 
    Mathlib.Analysis.SpecialFunctions.Exp.exp (-(x - μ)^2 / (2 * σ^2))) := 
by
  sorry

end verify_normal_distribution_properties_l371_371545


namespace coefficient_x2_expansion_l371_371660

theorem coefficient_x2_expansion : 
  let c := (2 * x + 1) ^ 6 in
  (∃ n : ℕ, n = 2) → 
  (∃ k : ℕ, k = binomial 6 2 * 2 ^ 2) →
  k = 60 :=
sorry

end coefficient_x2_expansion_l371_371660


namespace area_of_triangle_BCD_l371_371988

variables (b c d x y z : ℝ)
variables (AB AC AD : ℝ)

-- Define the tetrahedron and edge lengths
def tetrahedron_ABCD : Prop := AB = 2 * b ∧ AC = 2 * c ∧ AD = 2 * d

-- Define positive conditions for edge lengths
def positive_edges : Prop := 0 < b ∧ 0 < c ∧ 0 < d

-- Define mutually perpendicular edges
def perpendicular_edges : Prop := true -- Assume given because proving perpendicularity needs vectors

-- Define areas of triangles
def areas : Prop := (2 * x = b * c) ∧ (2 * y = c * d) ∧ (2 * z = b * d)

-- Calculate area of triangle BCD
def area_BCD : Prop := ∀ (b c d x y z : ℝ), tetrahedron_ABCD b c d x y z ∧ positive_edges b c d ∧ perpendicular_edges → areas b c d x y z → 
  Real.sqrt (x^2 + y^2 + z^2) = sqrt (x^2 + y^2 + z^2)

theorem area_of_triangle_BCD {b c d x y z : ℝ} : 
  tetrahedron_ABCD b c d x y z ∧ positive_edges b c d ∧ perpendicular_edges → areas b c d x y z → 
  Real.sqrt (x^2 + y^2 + z^2) = sqrt (x^2 + y^2 + z^2) := 
  by sorry

end area_of_triangle_BCD_l371_371988


namespace remaining_amount_needed_l371_371704

def goal := 150
def earnings_from_3_families := 3 * 10
def earnings_from_15_families := 15 * 5
def total_earnings := earnings_from_3_families + earnings_from_15_families
def remaining_amount := goal - total_earnings

theorem remaining_amount_needed : remaining_amount = 45 := by
  sorry

end remaining_amount_needed_l371_371704


namespace intersection_A_B_l371_371494

def A : Set ℝ := { x | x > 2 }
def B : Set ℝ := { x | 2 ≤ 2^x ∧ 2^x ≤ 8 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := by
  sorry

end intersection_A_B_l371_371494


namespace find_original_number_l371_371392

-- Given definitions and conditions
def doubled_add_nine (x : ℝ) : ℝ := 2 * x + 9
def trebled (y : ℝ) : ℝ := 3 * y

-- The proof problem we need to solve
theorem find_original_number (x : ℝ) (h : trebled (doubled_add_nine x) = 69) : x = 7 := 
by sorry

end find_original_number_l371_371392


namespace trajectory_is_parabola_l371_371502

noncomputable def distance_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (a * P.1 + b * P.2 + c).abs / (a^2 + b^2).sqrt

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

def is_parabola_trajectory (P : ℝ × ℝ) (A : ℝ × ℝ) (d : ℝ) : Prop :=
  ∃ (focus : ℝ × ℝ) (directrix : ℝ × ℝ × ℝ), 
  (distance P focus = distance_point_to_line P directrix.1 directrix.2 directrix.3 + d)

theorem trajectory_is_parabola (P A : ℝ × ℝ) (d : ℝ) (h : distance_point_to_line P 0 1 (-2) = distance P A + d) : 
  is_parabola_trajectory P A d :=
by 
  sorry

end trajectory_is_parabola_l371_371502


namespace fraction_tammy_derek_catches_l371_371590

theorem fraction_tammy_derek_catches :
  let joe_catches := 23
  let derek_catches := (2 * joe_catches) - 4
  let tammy_catches := 30
  tammy_catches / derek_catches = 5 / 7 := 
by
  -- Definitions to represent the problem's conditions
  let joe_catches := 23
  let derek_catches := (2 * joe_catches) - 4
  let tammy_catches := 30
  have fraction := tammy_catches / derek_catches
  -- Expected result
  show fraction = 5 / 7 from sorry

end fraction_tammy_derek_catches_l371_371590


namespace largest_prime_2025_digits_divisible_by_60_l371_371220

theorem largest_prime_2025_digits_divisible_by_60 :
  ∃ k', k' > 0 ∧ q^2 - k' ≡ 0 [MOD 60] → k' = 9 :=
by
  -- let q be the largest prime with 2025 digits
  let q := largest_prime_with_2025_digits
  have h_prime : Prime q := sorry -- Assume q is proved to be prime
  have h_digits : number_of_digits q = 2025 := sorry -- Assume q has 2025 digits
  -- Assume the conditions related to the divisibility checks
  have h4 : 4 ∣ (q^2 - 1) := sorry
  have h3 : 3 ∣ (q^2 - 1) := sorry
  have h5 : 5 ∣ (q^2 - 9) := sorry
  -- Prove the smallest positive integer k' that satisfies the conditions is 9
  exact
    -- The existence of such k' and its value
    ⟨9, sorry, sorry⟩

end largest_prime_2025_digits_divisible_by_60_l371_371220


namespace power_division_identity_l371_371310

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l371_371310


namespace max_rectangle_area_l371_371695

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l371_371695


namespace correct_number_of_conclusions_l371_371683

variables {a b c m : ℝ}

-- Constants condition for non-zero a
axiom h₀ : a ≠ 0

-- Condition for parabola opening downwards
axiom h₁ : a < 0

-- Conditions for the points on parabola
axiom h₂ : -2 < m
axiom h₃ : m < -1

-- Points on the parabola (1, 0) and (m, 0)
axiom h₄ : a * (1:ℝ)^2 + b * (1:ℝ) + c = 0
axiom h₅ : a * (m:ℝ)^2 + b * (m:ℝ) + c = 0

-- Conclusions
def conclusion_1 : Prop := a * b * c > 0
def conclusion_2 : Prop := a - b + c > 0
def conclusion_3 : Prop := a * (m + 1) - b + c > 0
def conclusion_4 : Prop := (4 * a * c - b^2 > 4 * a)

-- Proof problem: Number of correct conclusions is 3
def number_of_correct_conclusions : ℝ := 3

theorem correct_number_of_conclusions : 
  ((conclusion_1 ∨ conclusion_1 = true) ∧ 
  (conclusion_2 ∨ conclusion_2 = true) ∧
  (conclusion_3 ∨ conclusion_3 = true) ∧
  (¬ conclusion_4)) →
  3 = number_of_correct_conclusions := 
by
  sorry

end correct_number_of_conclusions_l371_371683


namespace average_books_per_member_l371_371673

theorem average_books_per_member :
  let books := [1, 2, 3, 4, 5]
  let members := [4, 5, 6, 2, 7]
  let total_books := (books.zip members).foldl (λ acc pair => acc + pair.1 * pair.2) 0
  let total_members := members.foldl (λ acc m => acc + m) 0
  let avg_books := (total_books : ℚ) / total_members
  avg_books ≈ 3 := by
  sorry

end average_books_per_member_l371_371673


namespace petyas_square_is_larger_l371_371626

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l371_371626


namespace petyas_square_is_larger_l371_371625

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l371_371625


namespace intervals_of_monotonicity_range_of_m_l371_371227

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2 * x + 5
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - x - 2

theorem intervals_of_monotonicity : 
  (∀ x : ℝ, x ∈ Iio (-2 / 3) → f' x > 0) ∧ 
  (∀ x : ℝ, x ∈ Ioo (-2 / 3) 1 → f' x < 0) ∧ 
  (∀ x : ℝ, x ∈ Ioi 1 → f' x > 0) :=
sorry

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) (2 : ℝ) → f x < m) ↔ m > 7 :=
sorry

end intervals_of_monotonicity_range_of_m_l371_371227


namespace evaluate_neg64_to_7_over_3_l371_371015

theorem evaluate_neg64_to_7_over_3 (a : ℝ) (b : ℝ) (c : ℝ) 
  (h1 : a = -64) (h2 : b = (-4)) (h3 : c = (7/3)) :
  a ^ c = -65536 := 
by
  have h4 : (-64 : ℝ) = (-4) ^ 3 := by sorry
  have h5 : a = b^3 := by rw [h1, h2, h4]
  have h6 : a ^ c = (b^3) ^ (7/3) := by rw [←h5, h3]
  have h7 : (b^3)^c = b^(3*(7/3)) := by sorry
  have h8 : b^(3*(7/3)) = b^7 := by norm_num
  have h9 : b^7 = -65536 := by sorry
  rw [h6, h7, h8, h9]
  exact h9

end evaluate_neg64_to_7_over_3_l371_371015


namespace smallest_acute_angle_bisector_l371_371297

noncomputable def angle_bisector_slope (m1 m2 : ℝ) : ℝ :=
  ((m1 + m2 + real.sqrt(1 + m1^2 + m2^2)) / (1 - m1 * m2))

theorem smallest_acute_angle_bisector 
  (intersection : ∃ p : ℝ × ℝ, p = (1,1)) 
  (line1 line2 line3 : ℝ → ℝ) 
  (h_line1 : ∀ x, line1 x = x) 
  (h_line2 : ∀ x, line2 x = 3 * x) 
  (h_line3 : ∀ x, line3 x = -x) : 
  angle_bisector_slope 1 3 = 2 - (real.sqrt 11) / 2 := 
by 
  sorry

end smallest_acute_angle_bisector_l371_371297


namespace average_weight_increase_l371_371266

theorem average_weight_increase (A : ℝ) : 
  let original_total_weight := 8 * A in
  let new_total_weight := original_total_weight + (64 - 56) in
  let new_average_weight := new_total_weight / 8 in
  new_average_weight - A = 1 := 
by
  sorry

end average_weight_increase_l371_371266


namespace domain_of_sqrt_2cosx_plus_1_l371_371663

open Real

theorem domain_of_sqrt_2cosx_plus_1 :
  ∀ k : ℤ, ∀ x : ℝ, 2 k * π - (2 * π) / 3 ≤ x ∧ x ≤ 2 k * π + (2 * π) / 3 ↔ (2 * cos x + 1 ≥ 0) :=
by
  sorry

end domain_of_sqrt_2cosx_plus_1_l371_371663


namespace volume_of_solid_l371_371786

-- Definitions of the surfaces
def surface1 (x y : ℝ) : ℝ := 2 * x ^ 2 + 8 * y ^ 2
def surface2 : ℝ := 4

-- The volume of the solid bounded by the two surfaces
def volumeSolid : ℝ := 
  ∫ z in 0..surface2, (π * surface1 (sqrt (z / 2)) (sqrt (z / 8)) / 8)

theorem volume_of_solid :
  volumeSolid = 2 * π :=
by
  sorry

end volume_of_solid_l371_371786


namespace students_own_both_pets_l371_371928

theorem students_own_both_pets (total : ℕ) (dog : ℕ) (cat : ℕ) (at_least_one : total = 45 ∧ dog = 28 ∧ cat = 33) :
  ∃ both : ℕ, total = dog + cat - both ∧ both = 16 :=
by
  obtain ⟨ht, hd, hc⟩ := at_least_one
  use 16
  split
  · rw [ht, hd, hc]
    exact rfl
  · exact rfl

end students_own_both_pets_l371_371928


namespace ellipse_semimajor_axis_value_l371_371989

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371989


namespace monotonicity_of_f_f_extreme_points_sum_less_than_minus_3_l371_371887

def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - 2 * x + log x

theorem monotonicity_of_f (a : ℝ) (h : a ≠ 0) :
  (if a ≥ 1 / 2 then ∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → f a x ≤ f a y
   else ∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) 
     → (if x < (1 - sqrt (1 - 2 * a)) / (2 * a) then f a x ≤ f a y
        else if y > (1 + sqrt (1 - 2 * a)) / (2 * a) then f a x ≤ f a y
        else f a y ≤ f a x)) :=
sorry

theorem f_extreme_points_sum_less_than_minus_3 
  (a : ℝ) (h1 : 0 < a) (h2 : a < 1 / 2) (x1 x2 : ℝ)
  (hx1: x1 = (1 - sqrt (1 - 2 * a)) / (2 * a)) (hx2: x2 = (1 + sqrt (1 - 2 * a)) / (2 * a))
  (hx1x2 : f a x1 = a * x1 ^ 2 - 2 * x1 + log x1)
  (hx2x2 : f a x2 = a * x2 ^ 2 - 2 * x2 + log x2) :
  f a x1 + f a x2 < -3 :=
sorry

end monotonicity_of_f_f_extreme_points_sum_less_than_minus_3_l371_371887


namespace max_rectangle_area_l371_371696

variables {a b : ℝ}

theorem max_rectangle_area (h : 2 * a + 2 * b = 60) : a * b ≤ 225 :=
by 
  -- Proof to be filled in
  sorry

end max_rectangle_area_l371_371696


namespace power_division_l371_371330

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l371_371330


namespace distance_between_A_B_is_16_l371_371183

-- The given conditions are translated as definitions
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- The theorem stating the proof problem
theorem distance_between_A_B_is_16 :
  let A : ℝ × ℝ := (4, 8)
  let B : ℝ × ℝ := (4, -8)
  let d : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  d = 16 :=
by
  sorry

end distance_between_A_B_is_16_l371_371183


namespace find_c_minus_a_l371_371160

variable (a b c d e : ℝ)

-- Conditions
axiom avg_ab : (a + b) / 2 = 40
axiom avg_bc : (b + c) / 2 = 60
axiom avg_de : (d + e) / 2 = 80
axiom geom_mean : (a * b * d) = (b * c * e)

theorem find_c_minus_a : c - a = 40 := by
  sorry

end find_c_minus_a_l371_371160


namespace Mark_hours_left_l371_371780

theorem Mark_hours_left (sick_days vacation_days : ℕ) (hours_per_day : ℕ) 
  (h1 : sick_days = 10) (h2 : vacation_days = 10) (h3 : hours_per_day = 8) 
  (used_sick_days : ℕ) (used_vacation_days : ℕ) 
  (h4 : used_sick_days = sick_days / 2) (h5 : used_vacation_days = vacation_days / 2) 
  : (sick_days + vacation_days - used_sick_days - used_vacation_days) * hours_per_day = 80 :=
by
  sorry

end Mark_hours_left_l371_371780


namespace inequality_abc_l371_371054

theorem inequality_abc (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a / real.sqrt b + b / real.sqrt a ≥ real.sqrt a + real.sqrt b := 
sorry

end inequality_abc_l371_371054


namespace complex_identity_l371_371881

def ω : ℂ := -1/2 + (Real.sqrt 3)/2 * Complex.I

theorem complex_identity : 1 + ω = -1 / ω :=
by
  sorry

end complex_identity_l371_371881


namespace sequence_periodic_l371_371225

def odd (n : ℤ) : Prop :=
  n % 2 = 1

def X_sequence (X : ℕ → list ℕ) (n : ℕ) :=
  ∀ k i, X 0 = [1, 0, ... , 0, 1] ∧
         (X (k+1) !! i = if X k !! i = X k !! (i + 1 % n) then 0 else 1)
         ∧ X k !! (n + 1) = X k !! 1

theorem sequence_periodic (n m : ℕ) (X : ℕ → list ℕ) :
  odd n ∧ n ≥ 2 ∧ X_sequence X n ∧ X m = X 0 → n ∣ m :=
sorry

end sequence_periodic_l371_371225


namespace drum_oil_capacity_l371_371448

theorem drum_oil_capacity (C : ℝ) (hx : 0 < C) :
  let drumX_oil := C / 2
  let drumY_capacity := 2 * C
  let drumY_oil := drumY_capacity / 4
  let total_oil_drumY := drumX_oil + drumY_oil
  total_oil_drumY / drumY_capacity = 1 / 2 :=
by
  -- Define the amounts of oil
  let drumX_oil := C / 2
  let drumY_capacity := 2 * C
  let drumY_oil := drumY_capacity / 4
  let total_oil_drumY := drumX_oil + drumY_oil
  
  -- Perform the required calculations
  have h1 : total_oil_drumY = C := by
    simp [drumX_oil, drumY_oil, total_oil_drumY, drumY_capacity]
    rw [add_div_eq_mul_add_mul_div]
    simp
    exact C

  have h2 : drumY_capacity = 2 * C := by
    refl

  -- Conclude the proof
  rw [h1, h2]
  congr
  norm_num
  exact div_self (two_ne_zero : (2 : ℝ) ≠ 0)
  sorry -- skip further proof steps

end drum_oil_capacity_l371_371448


namespace total_family_members_l371_371200

variable (members_father_side : Nat) (percent_incr : Nat)
variable (members_mother_side := members_father_side + (members_father_side * percent_incr / 100))
variable (total_members := members_father_side + members_mother_side)

theorem total_family_members 
  (h1 : members_father_side = 10) 
  (h2 : percent_incr = 30) :
  total_members = 23 :=
by
  sorry

end total_family_members_l371_371200


namespace determine_divisors_l371_371967

theorem determine_divisors (n : ℕ) (h_pos : n > 0) (d : ℕ) (h_div : d ∣ 3 * n^2) (h_exists : ∃ k : ℤ, n^2 + d = k^2) : d = 3 * n^2 := 
sorry

end determine_divisors_l371_371967


namespace inequality_transformation_l371_371533

theorem inequality_transformation (x y : ℝ) (h : 2 * x - 5 < 2 * y - 5) : x < y := 
by 
  sorry

end inequality_transformation_l371_371533


namespace f_of_8_l371_371068

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom function_property : ∀ x : ℝ, f (x + 2) = -1 / f (x)

-- Statement to prove
theorem f_of_8 : f 8 = 0 :=
sorry

end f_of_8_l371_371068


namespace polygon_sides_l371_371912

theorem polygon_sides (a : ℝ) (n : ℕ) (h1 : a = 140) (h2 : 180 * (n-2) = n * a) : n = 9 := 
by sorry

end polygon_sides_l371_371912


namespace magnitude_of_linear_combination_l371_371500

variables (e1 e2 : ℝ^3)
-- Conditions:
-- e1 and e2 are unit vectors
axiom e1_unit : ∥e1∥ = 1
axiom e2_unit : ∥e2∥ = 1

-- The angle between e1 and e2 is 60 degrees
axiom angle_60 : e1 • e2 = 1 * 1 * (Real.cos (Real.pi / 3))

-- The theorem to prove
theorem magnitude_of_linear_combination : ∥2 • e1 + 3 • e2∥ = Real.sqrt 19 := by
  sorry

end magnitude_of_linear_combination_l371_371500


namespace largest_rectangle_area_l371_371686

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l371_371686


namespace total_length_of_sticks_l371_371586

-- Definitions of stick lengths based on the conditions
def length_first_stick : ℕ := 3
def length_second_stick : ℕ := 2 * length_first_stick
def length_third_stick : ℕ := length_second_stick - 1

-- Proof statement
theorem total_length_of_sticks : length_first_stick + length_second_stick + length_third_stick = 14 :=
by
  sorry

end total_length_of_sticks_l371_371586


namespace power_division_identity_l371_371311

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l371_371311


namespace identify_counterfeit_coin_in_two_weighings_l371_371951

theorem identify_counterfeit_coin_in_two_weighings (coins : Fin 4 → ℝ) (counterfeit : Fin 4 → Prop) :
  (∃ i, counterfeit i ∧ (∀ j, j ≠ i → coins j = coins (Fin.succ j))) →
  ∃ i, counterfeit i ∧ coins_in_two_weighings coins counterfeit i :=
by
  sorry

end identify_counterfeit_coin_in_two_weighings_l371_371951


namespace f_is_even_l371_371477
noncomputable def f : ℝ → ℝ := sorry

axiom f_equation (x y : ℝ) : f(x + y) + f(x - y) = 2 * f(x) * f(y)
axiom f_nonzero_at_zero : f(0) ≠ 0

theorem f_is_even : ∀ x : ℝ, f(-x) = f(x) := sorry

end f_is_even_l371_371477


namespace part1_monotonic_intervals_part2_monotonic_intervals_l371_371890

noncomputable def f (a x : ℝ) := a * x^2 + (2 * a - 1) * x - Real.log x

theorem part1_monotonic_intervals (a : ℝ) (x : ℝ) (h : a = 1 / 2) (hx : x > 0) :
  ((0 < x ∧ x < 1 → deriv (λ x, f a x) x < 0) ∧
   (x > 1 → deriv (λ x, f a x) x > 0) ∧
   (x = 1 → f a x = 1 / 2)) := sorry

theorem part2_monotonic_intervals (a : ℝ) (x : ℝ) (hx : x > 0) :
  ((a ≤ 0 → deriv (λ x, f a x) x < 0) ∧
   (a > 0 → ((0 < x ∧ x < 1 / (2 * a) → deriv (λ x, f a x) x < 0) ∧
    (x > 1 / (2 * a) → deriv (λ x, f a x) x > 0)))) := sorry

end part1_monotonic_intervals_part2_monotonic_intervals_l371_371890


namespace D_not_prob_dist_l371_371048

def is_prob_dist (l : List ℚ) : Prop :=
  l.sum = 1

-- Define the four lists
def A : List ℚ := [0, 1/2, 0, 0, 1/2]
def B : List ℚ := [0.1, 0.2, 0.3, 0.4]
def C (p : ℚ) : List ℚ := [p, 1 - p] -- 0 ≤ p ≤ 1 is implied in the problem representation
def D : List ℚ := (List.range 7).map (λ n, 1 / (n + 1) / (n + 2))

-- The theorem stating that D is not a probability distribution
theorem D_not_prob_dist : ¬(is_prob_dist D) :=
  sorry

end D_not_prob_dist_l371_371048


namespace minimum_portraits_l371_371247

theorem minimum_portraits (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) 
  (h1 : ∀ i, 1 ≤ b i ∧ b i ≤ 80) 
  (h2 : ∀ i, 1600 ≤ a i ∧ a i ≤ 2008) 
  (h3 : ∏ i in finset.range n, ((a i) + (b i)) / (a i) = 5 / 4) : 
  n = 5 :=
sorry

end minimum_portraits_l371_371247


namespace scientific_notation_example_l371_371406

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end scientific_notation_example_l371_371406


namespace population_increase_after_5_years_l371_371434

noncomputable def effective_growth_rate (growth_rate : ℚ) (migration_rate : ℚ) (additional_rate : ℚ) : ℚ :=
	growth_rate - migration_rate + additional_rate

noncomputable def population_increase (initial_growth_rate : ℚ) (years : ℕ) : ℚ :=
	(1 + initial_growth_rate) ^ years

theorem population_increase_after_5_years :
  let growth_rate_A := 0.12
  let migration_rate_A := 0.02
  let additional_rate_A := 0
  let growth_rate_B := 0.08
  let migration_rate_B := 0.03
  let additional_rate_B := 0
  let growth_rate_C := 0.10
  let migration_rate_C := 0.01
  let additional_rate_C := 0.01
  let years := 5

  let eff_growth_A := effective_growth_rate growth_rate_A migration_rate_A additional_rate_A
  let eff_growth_B := effective_growth_rate growth_rate_B migration_rate_B additional_rate_B
  let eff_growth_C := effective_growth_rate growth_rate_C migration_rate_C additional_rate_C

  population_increase eff_growth_A years = 1.61051 ∧
  population_increase eff_growth_B years = 1.27628 ∧
  population_increase eff_growth_C years = 1.61051 :=
by
  sorry

end population_increase_after_5_years_l371_371434


namespace area_of_triangle_TSP_l371_371759

theorem area_of_triangle_TSP :
  ∀ S T P : ℝ × ℝ,
  S = (2, 7) ∧
  T = (0, b) ∧ b ∈ ℝ ∧
  (∃ m₁ c₁, m₁ = 3 ∧ S.2 = m₁ * S.1 + c₁ ∧ P.2 = m₁ * P.1 + c₁) ∧
  (∃ m₂ c₂, m₂ = -1 ∧ S.2 = m₂ * S.1 + c₂ ∧ P.2 = m₂ * P.1 + c₂) ∧
  (∃ m₃ c₃, m₃ = -1/3 ∧ T.2 = m₃ * T.1 + c₃ ∧ P.2 = m₃ * P.1 + c₃) →
  let base := sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) in
  let height := abs (T.2 - P.2) in
  0.5 * base * height = 15 * sqrt 17 / 4 :=
by
  intro S T P S_def T_def T_real line1_def line2_def line3_def
  let base := sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2)
  let height := abs (T.2 - P.2)
  have h_area : 0.5 * base * height = 15 * sqrt 17 / 4 := sorry
  exact h_area

end area_of_triangle_TSP_l371_371759


namespace probability_sum_not_equal_8_l371_371426

theorem probability_sum_not_equal_8 :
  ∑ 'x y: ℕ, if (1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ x + y ≠ 8) then (1 : ℚ) else 0 = (31 : ℚ) / 36 :=
sorry

end probability_sum_not_equal_8_l371_371426


namespace count_special_integers_in_range_l371_371805

theorem count_special_integers_in_range :
  let is_special (n : ℕ) := (n >= 1000) ∧ (n < 3000) ∧ (n % 10 = (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10))
  (finset.filter is_special (finset.range 3000)).card = 109 :=
by
  sorry

end count_special_integers_in_range_l371_371805


namespace statement_1_statement_2_statement_3_statement_4_l371_371039

variables (a b c x0 : ℝ)
noncomputable def P (x : ℝ) : ℝ := a*x^2 + b*x + c

-- Statement ①
theorem statement_1 (h : a - b + c = 0) : P a b c (-1) = 0 := sorry

-- Statement ②
theorem statement_2 (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a*x1^2 + c = 0 ∧ a*x2^2 + c = 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 := sorry

-- Statement ③
theorem statement_3 (h : P a b c c = 0) : a*c + b + 1 = 0 := sorry

-- Statement ④
theorem statement_4 (h : P a b c x0 = 0) : b^2 - 4*a*c = (2*a*x0 + b)^2 := sorry

end statement_1_statement_2_statement_3_statement_4_l371_371039


namespace valid_three_digit_numbers_count_l371_371903

def is_prime_or_even (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def count_valid_numbers : ℕ :=
  (4 * 4) -- number of valid combinations for hundreds and tens digits

theorem valid_three_digit_numbers_count : count_valid_numbers = 16 :=
by 
  -- outline the structure of the proof here, but we use sorry to indicate the proof is not complete
  sorry

end valid_three_digit_numbers_count_l371_371903


namespace distance_from_point_to_y_axis_l371_371480

theorem distance_from_point_to_y_axis 
  (a x0 : ℝ) 
  (h1 : (λ x : ℝ, a * x ^ 2) x0 = 2)
  (h2 : real.sqrt ((x0 - 0) ^ 2 + (2 - 1 / (4 * a)) ^ 2) = 3) : 
  |x0| = 2 * real.sqrt 2 := 
sorry

end distance_from_point_to_y_axis_l371_371480


namespace part1_part2_l371_371071

variables {A B : ℝ} {a b c : ℝ × ℝ}

-- definitions according to conditions
def vector_a (A : ℝ) : ℝ × ℝ := (Real.sin A, Real.cos A)
def vector_b (B : ℝ) : ℝ × ℝ := (Real.sin B, Real.cos B)
def vector_c : ℝ × ℝ := (1, 1)

-- condition for parallel vectors
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- condition for perpendicular vectors
def are_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Part (1): Prove that if vector_a A is parallel to vector_c, then A = π/4
theorem part1 (h1 : are_parallel (vector_a A) vector_c) : A = π / 4 :=
sorry

-- Part (2): Prove that if (vector_a A - vector_b B) is perpendicular to vector_c, then ∠ABC is a right angle
theorem part2 (h2 : are_perpendicular (vector_a A - vector_b B) vector_c) : A + B = π / 2 :=
sorry

end part1_part2_l371_371071


namespace ellipse_semimajor_axis_value_l371_371997

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371997


namespace fifth_friend_paid_13_l371_371033

noncomputable def fifth_friend_payment (a b c d e : ℝ) : Prop :=
a = (1/3) * (b + c + d + e) ∧
b = (1/4) * (a + c + d + e) ∧
c = (1/5) * (a + b + d + e) ∧
a + b + c + d + e = 120 ∧
e = 13

theorem fifth_friend_paid_13 : 
  ∃ (a b c d e : ℝ), fifth_friend_payment a b c d e := 
sorry

end fifth_friend_paid_13_l371_371033


namespace impossible_to_select_seven_weights_l371_371042

-- Define the set of weights from 1 to 26
def weights : Finset ℕ := Finset.range (26 + 1)

-- Define the property for subsets
def unique_subset_sums (s : Finset ℕ) : Prop :=
  ∀ a b : Finset ℕ, a ≠ b → a ∈ s.powerset → b ∈ s.powerset → a.sum ≠ b.sum

theorem impossible_to_select_seven_weights :
  ∀ s : Finset ℕ, s ⊆ weights → s.card = 7 → ¬ unique_subset_sums s :=
by
  intro s Hsub Hcard
  sorry

end impossible_to_select_seven_weights_l371_371042


namespace area_of_quadrilateral_l371_371362

variables (A B C M N P : Point)
variables (area_ABC : ℝ)
variables (AM MB BN NC : ℝ)
variables (triangle_area : ∀ A B C : Point, ℝ)

-- Defining the given conditions
def conditions := 
  AM = 2 * MB ∧ BN = NC ∧ area_ABC = 30

-- Exactly stating the problem in Lean: Prove the area of quadrilateral MBNP is 7
theorem area_of_quadrilateral (h : conditions A B C M N area_ABC) :
  let P := intersection (line A N) (line C M) in
  quadrilateral_area M B N P = 7 :=
sorry

end area_of_quadrilateral_l371_371362


namespace max_angle_B_l371_371918

-- We define the necessary terms to state our problem
variables {A B C : Real} -- The angles of triangle ABC
variables {cot_A cot_B cot_C : Real} -- The cotangents of angles A, B, and C

-- The main theorem stating that given the conditions the maximum value of angle B is pi/3
theorem max_angle_B (h1 : cot_B = (cot_A + cot_C) / 2) (h2 : A + B + C = Real.pi) :
  B ≤ Real.pi / 3 := by
  sorry

end max_angle_B_l371_371918


namespace initial_weight_of_solution_Y_is_8_l371_371260

theorem initial_weight_of_solution_Y_is_8
  (W : ℝ)
  (hw1 : 0.25 * W = 0.20 * W + 0.4)
  (hw2 : W ≠ 0) : W = 8 :=
by
  sorry

end initial_weight_of_solution_Y_is_8_l371_371260


namespace PQ_parallel_to_AB_l371_371484

variable (ABC : Type) [EuclideanPlane ABC]
variables (A B C H P Q : ABC)
variables [EqTriangle (triangle.mk A B C) (triangle.mk B C A)]
variable (AB_eq_BC : dist A B = dist B C)
variable (angle_ABC_90 : angle B A C = 90)
variable (height_BH : is_height B H (line.mk A C))
variable (point_P : is_point_on_line P (line.mk A C) ∧ dist A P = dist A B)
variable (point_Q : is_point_on_line Q (line.mk B C) ∧ dist B Q = height_BH)

theorem PQ_parallel_to_AB (h : triangle_ABC A B C ∧ isosceles_right ABC A B C) :
  by sorry -- Proof logic goes here

end PQ_parallel_to_AB_l371_371484


namespace problem_solution_l371_371035

noncomputable def a (n : ℕ) : ℝ :=
  if n > 1 then 1 / Real.log 5005 / Real.log n else 0

def b : ℝ :=
  a 5 + a 6 + a 7 + a 8

def c : ℝ :=
  a 7 + a 8 + a 9 + a 10 + a 11

theorem problem_solution : b - c = Real.log 5005 (1 / 33) := 
  sorry

end problem_solution_l371_371035


namespace ellipse_properties_l371_371868

def center : ℝ × ℝ := (0, 0)
def foci_on_x_axis : Prop := ∀ (e : ℝ), e > 0 → (center.1 - e, 0) ∨ (center.1 + e, 0)
def line_through_focus (c : ℝ) : ℝ → ℝ := λ x, 1/2 * (x - c)
def collinear (v1 v2 : ℝ × ℝ) : Prop := ∃ (k : ℝ), v1.1 * k = v2.1 ∧ v1.2 * k = v2.2
def point_on_ellipse (a b x y : ℝ) := x^2/a^2 + y^2/b^2 = 1

theorem ellipse_properties 
  (c x1 x2 y1 y2 a b : ℝ)
  (hFoci: foci_on_x_axis)
  (hLine: line_through_focus c 0)
  (hIntersection: point_on_ellipse a b x1 y1 ∧ point_on_ellipse a b x2 y2)
  (hCollinear: collinear (x1 + x2, y1 + y2) (-3, 1)):
  let e := (b*sqrt 5)/a in 
  let p1 := (center, a, b, c) in
  eccentricity e = sqrt 30 / 6 ∧
  ∀ (λ μ : ℝ) (M : ℝ × ℝ), 
    M = (λ * x1 + μ * x2, λ * y1 + μ * y2) →
    point_on_ellipse a b M.1 M.2 →
    λ^2 + μ^2 = 1 :=
sorry

end ellipse_properties_l371_371868


namespace ko_eq_pl_l371_371241

open Classical

variables (A B C D K L M N O P : Type) [IsTrapezoid A B C D] 
  (extendsAD : Extend AD K) (extendsBC : Extend BC L)

variables (InterKL_AB_M : Intersect KL AB M) (InterKL_CD_N : Intersect KL CD N)
variables (InterKL_AC_O : Intersect KL AC O) (InterKL_BD_P : Intersect KL BD P)
variables (KM_eq_NL : KM = NL)

theorem ko_eq_pl (h₁ : IsTrapezoid A B C D) (h₂ : Extend A D K) (h₃ : Extend B C L)
(h₄ : Intersect KL AB M) (h₅ : Intersect KL CD N) (h₆ : Intersect KL AC O) 
(h₇ : Intersect KL BD P) (h₈ : KM = NL) : KO = PL := sorry

end ko_eq_pl_l371_371241


namespace angle_A_is_30_deg_l371_371165

theorem angle_A_is_30_deg (A B C : Type) [triangle : Triangle A B C] (b c a : ℝ)
  (h : b^2 + c^2 - a^2 = real.sqrt 3 * b * c) : ∠A = 30 :=
by sorry

end angle_A_is_30_deg_l371_371165


namespace smallest_n_for_invariant_digit_sum_l371_371211

def digit_sum (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem smallest_n_for_invariant_digit_sum :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (∀ k ∈ {1, 2, ..., n^2}.to_finset, digit_sum (k * n) = digit_sum n) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (∀ k ∈ {1, 2, ..., m^2}.to_finset, digit_sum (k * m) = digit_sum m) → n ≤ m) :=
  ⟨999, by sorry⟩

end smallest_n_for_invariant_digit_sum_l371_371211


namespace BrotherUpperLimit_l371_371552

variable (w : ℝ) -- Arun's weight
variable (b : ℝ) -- Upper limit of Arun's weight according to his brother's opinion

-- Conditions as per the problem
def ArunOpinion (w : ℝ) := 64 < w ∧ w < 72
def BrotherOpinion (w b : ℝ) := 60 < w ∧ w < b
def MotherOpinion (w : ℝ) := w ≤ 67

-- The average of probable weights
def AverageWeight (weights : Set ℝ) (avg : ℝ) := (∀ w ∈ weights, 64 < w ∧ w ≤ 67) ∧ avg = 66

-- The main theorem to be proven
theorem BrotherUpperLimit (hA : ArunOpinion w) (hB : BrotherOpinion w b) (hM : MotherOpinion w) (hAvg : AverageWeight {w | 64 < w ∧ w ≤ 67} 66) : b = 67 := by
  sorry

end BrotherUpperLimit_l371_371552


namespace values_of_a_and_b_l371_371840

def is_root (a b x : ℝ) : Prop := x^2 - 2*a*x + b = 0

noncomputable def A : Set ℝ := {-1, 1}
noncomputable def B (a b : ℝ) : Set ℝ := {x | is_root a b x}

theorem values_of_a_and_b (a b : ℝ) (h_nonempty : Set.Nonempty (B a b)) (h_union : A ∪ B a b = A) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1) :=
sorry

end values_of_a_and_b_l371_371840


namespace pow_div_eq_l371_371326

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l371_371326


namespace smallest_N_existence_l371_371679

theorem smallest_N_existence :
  ∃ N : ℕ, (∀ (a b : ℕ), a ∈ (Set.range 2016.succ) ∧ b ∈ (Set.range 2016.succ) ∧ a ≠ b → a * b ≤ N) ∧
            (∀ M : ℕ, (∀ (a b : ℕ), a ∈ (Set.range 2016.succ) ∧ b ∈ (Set.range 2016.succ) ∧ a ≠ b → a * b ≤ M) → N ≤ M) ∧
            N = 1017072 :=
by
  sorry

end smallest_N_existence_l371_371679


namespace determine_a_of_parallel_lines_l371_371009

theorem determine_a_of_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x ↔ y = 3 * x + a) →
  (∀ x y : ℝ, y - 2 = (a - 3) * x ↔ y = (a - 3) * x + 2) →
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x → y - 2 = (a - 3) * x → 3 = a - 3) →
  a = 6 :=
by
  sorry

end determine_a_of_parallel_lines_l371_371009


namespace city_with_greatest_percentage_change_l371_371562

/-- Population data in 1990 and 2000 for five cities -/
def population_data : List (String × ℕ × ℕ) :=
  [("P", 150000, 180000),
   ("Q", 200000, 210000),
   ("R", 120000, 144000),
   ("S", 180000, 171000),
   ("T", 160000, 160000)]

/-- Calculate the percentage change in population from 1990 to 2000 -/
def percentage_change (pop1990 pop2000 : ℕ) : ℚ :=
  ((pop2000 - pop1990) : ℚ) / (pop1990 : ℚ) * 100

/-- Prove that the city with the greatest percentage change is City R -/
theorem city_with_greatest_percentage_change : 
  let changes := population_data.map (λ data, (data.1, percentage_change data.2 data.3)) in
  ∃ city, city ∈ changes ∧ city.1 = "R" ∧ ∀ city', city' ∈ changes → city'.2 ≤ city.2 :=
by
  let changes := population_data.map (λ data, (data.1, percentage_change data.2 data.3))
  existsi ("R", percentage_change 120000 144000)
  split
  sorry

end city_with_greatest_percentage_change_l371_371562


namespace smallest_N_existence_l371_371678

theorem smallest_N_existence :
  ∃ N : ℕ, (∀ (a b : ℕ), a ∈ (Set.range 2016.succ) ∧ b ∈ (Set.range 2016.succ) ∧ a ≠ b → a * b ≤ N) ∧
            (∀ M : ℕ, (∀ (a b : ℕ), a ∈ (Set.range 2016.succ) ∧ b ∈ (Set.range 2016.succ) ∧ a ≠ b → a * b ≤ M) → N ≤ M) ∧
            N = 1017072 :=
by
  sorry

end smallest_N_existence_l371_371678


namespace find_g_l371_371979

-- Define a monic polynomial of degree 2
def is_monic_quadratic (g : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, g = λ x, x^2 + b * x + c

-- Given conditions
variables (g : ℝ → ℝ)
variable (h_monic : is_monic_quadratic g)
variable (h_g0 : g 0 = -3)
variable (h_g1 : g (-1) = 8)

-- Statement to be proven
theorem find_g : g = (λ x, x^2 - 10 * x - 3) :=
by {
  sorry
}

end find_g_l371_371979


namespace race_order_count_l371_371120

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end race_order_count_l371_371120


namespace eight_pow_15_div_sixtyfour_pow_6_l371_371318

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l371_371318


namespace combo_simplify_l371_371257

-- Definitions of combinations
def C (n k : ℕ) : ℕ := nat.choose n k

-- The theorem statement
theorem combo_simplify (n : ℕ) : C n (n-2) + C n 3 + C (n+1) 2 = C (n+2) 3 := by
  sorry

end combo_simplify_l371_371257


namespace hyperbola_standard_eq_ellipse_standard_eq_dot_product_range_l371_371478

-- Part 1
theorem hyperbola_standard_eq (x y : ℝ) (passes_through : ∀ x y, x = 5 ∧ y = 9 / 4) : 
  (∃ λ : ℝ, λ ≠ 0 ∧ (x^2 / 16 - y^2 / 9 = λ)) :=
sorry

-- Part 2
theorem ellipse_standard_eq {a b : ℝ} (focus_parabola : (4, 0)) (focus_hyperbola : (5, 0)) :
  (a = 5 ∧ b = 3 ∧ a > b > 0 → a^2 - b^2 = 9 → (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

-- Part 3
theorem dot_product_range (AP BP : ℝ × ℝ) (P : ℝ × ℝ) (C D : ℝ × ℝ)
  (P_on_segment : ∀ x y, 0 ≤ x ∧ x ≤ 5 ∧ y = -3 / 5 * x + 3)
  (AP_eq : ∀ x y, (x + 1, y)) (BP_eq : ∀ x y, (x - 1, y)) : 
  (∃ r : ℝ, (∃ x, 191 / 34 ≤ x ∧ x ≤ 24 ∧ r = AP.1^2 + AP.2^2 - 1)) :=
sorry

end hyperbola_standard_eq_ellipse_standard_eq_dot_product_range_l371_371478


namespace domain_of_f_maximal_l371_371791

noncomputable def f (x : ℝ) : ℝ := sorry

theorem domain_of_f_maximal :
  (∀ x, x ≠ 0 → (f x + f (1/x) = 3 * x)) →
  (∀ x, x ≠ 0 → (1 / x ∈ {x | x ≠ 0})) →
  (set_of (λ x, x ≠ 0 ∧ ∃ y, (y = 1 ∨ y = -1) ∧ x = y) = {-1, 1}) :=
by
  intros h1 h2
  ext x
  simp only [set.mem_set_of_eq, set.mem_insert_iff, set.mem_singleton_iff]
  split
  { intro hx
    cases hx with hx1 hx2
    exact hx2 }
  { intro hx
    cases hx
    { use (-1)
      exact ⟨h1, rfl⟩ }
    { use (1)
      exact ⟨h1, rfl⟩ } }
  sorry

end domain_of_f_maximal_l371_371791


namespace teacher_age_is_56_l371_371741

theorem teacher_age_is_56 (s t : ℝ) (h1 : s = 40 * 15) (h2 : s + t = 41 * 16) : t = 56 := by
  sorry

end teacher_age_is_56_l371_371741


namespace total_distance_run_l371_371581

-- Define the distances run each day based on the distance run on Monday (x)
def distance_on_monday (x : ℝ) := x
def distance_on_tuesday (x : ℝ) := 2 * x
def distance_on_wednesday (x : ℝ) := x
def distance_on_thursday (x : ℝ) := (1/2) * x
def distance_on_friday (x : ℝ) := x

-- Define the condition for the shortest distance
def shortest_distance_condition (x : ℝ) :=
  min (distance_on_monday x)
    (min (distance_on_tuesday x)
      (min (distance_on_wednesday x)
        (min (distance_on_thursday x) 
          (distance_on_friday x)))) = 5

-- State and prove the total distance run over the week
theorem total_distance_run (x : ℝ) (hx : shortest_distance_condition x) : 
  distance_on_monday x + distance_on_tuesday x + distance_on_wednesday x + distance_on_thursday x + distance_on_friday x = 55 :=
by
  sorry

end total_distance_run_l371_371581


namespace max_value_of_a_l371_371523

theorem max_value_of_a
  (a : ℝ)
  (f : ℝ → ℝ := λ x, a * x^2 - a * x + 1) :
  (∀ x ∈ Icc (0 : ℝ) 1, abs (f x) ≤ 1) → a ≤ 8 := sorry

end max_value_of_a_l371_371523


namespace smallest_k_l371_371030

theorem smallest_k (k : ℕ) : 
  (k > 0 ∧ (k*(k+1)*(2*k+1)/6) % 400 = 0) → k = 800 :=
by
  sorry

end smallest_k_l371_371030


namespace sum_of_squares_of_sines_l371_371463

theorem sum_of_squares_of_sines : 
  (finset.range 89).sum (λ n, (real.sin (real.pi * (n + 1) / 180))^2) = 44.5 :=
by
  sorry

end sum_of_squares_of_sines_l371_371463


namespace range_of_a_l371_371515

noncomputable def g (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x + 3

theorem range_of_a 
  (h_mono_inc : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → g a x1 ≤ g a x2)
  (h_nonneg : ∀ x : ℝ, -1 < x ∧ x < 1 → 0 ≤ g a x) :
  (-2 : ℝ) ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l371_371515


namespace polynomial_with_transformed_roots_l371_371984

noncomputable def polynomial_transform (f : ℕ → ℕ) (a b c : ℕ) : ℕ := sorry

theorem polynomial_with_transformed_roots (a b c : ℕ) :
  (polynomial_transform (λ x, x + 3) a b c) = 0 :=
begin
  sorry
end

end polynomial_with_transformed_roots_l371_371984


namespace three_digit_number_is_212_l371_371294

def smallest_prime : ℕ := 2
def pi_approx : ℝ := 3.14
def tens_digit : ℕ := Int.floor (0.3 + pi_approx * 13) % 10  -- The first digit after the decimal
def smallest_three_digit_number_divisible_by_17 : ℕ := 102

theorem three_digit_number_is_212 :
  ∃ n : ℕ, n = 212 ∧ n / 100 = smallest_prime ∧ (n % 100) / 10 = tens_digit ∧ n % 10 = smallest_three_digit_number_divisible_by_17 % 10 :=
by
  use 212
  split
  · rfl
  · split
    · simp [smallest_prime]
    · split
      · simp [tens_digit]
      · simp [smallest_three_digit_number_divisible_by_17]

end three_digit_number_is_212_l371_371294


namespace bobs_sisters_time_l371_371422

-- Define Bob's current time in seconds
def bobs_current_time_seconds : ℕ := (10 * 60) + 40

-- Define the required improvement percentage
def improvement_percentage : ℕ := 50

-- Calculate the improved time Bob needs to achieve
def improved_time_seconds : ℕ := (bobs_current_time_seconds * (100 - improvement_percentage)) / 100

-- Define the target time for the proof
def target_time_seconds : ℕ := 320

-- The theorem stating the equivalence of Bob's sister's time with the calculated time
theorem bobs_sisters_time (bobs_current_time_seconds = 640) (improvement_percentage = 50) :
    improved_time_seconds = target_time_seconds :=
by
  -- Proof is omitted according to the instructions
  sorry

end bobs_sisters_time_l371_371422


namespace number_satisfies_equation_l371_371817

theorem number_satisfies_equation :
  ∃ x : ℝ, (0.6667 * x - 10 = 0.25 * x) ∧ (x = 23.9936) :=
by
  sorry

end number_satisfies_equation_l371_371817


namespace pow_div_eq_l371_371327

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l371_371327


namespace ball_hits_ground_time_l371_371751

theorem ball_hits_ground_time (t : ℚ) :
  (-4.9 * (t : ℝ)^2 + 5 * (t : ℝ) + 10 = 0) → t = 10 / 7 :=
sorry

end ball_hits_ground_time_l371_371751


namespace part1_M_union_N_part1_N_complement_part2_P_subset_M_l371_371114

section Main

variable U : Set ℝ := {x | True}
variable M : Set ℝ := {x | (x + 4) * (x - 6) < 0}
variable N : Set ℝ := {x | x - 5 < 0}
variable P (t : ℝ) : Set ℝ := {x | |x| = t}

theorem part1_M_union_N : M ∪ N = {x | x < 6} := sorry

theorem part1_N_complement : (U \ N) = {x | x ≥ 5} := sorry

theorem part2_P_subset_M (t : ℝ) : (P t ⊆ M) ↔ t < 4 := sorry

end Main

end part1_M_union_N_part1_N_complement_part2_P_subset_M_l371_371114


namespace proof_op_l371_371232

def op (A B : ℕ) : ℕ := (A * B) / 2

theorem proof_op (a b c : ℕ) : op (op 4 6) 9 = 54 := by
  sorry

end proof_op_l371_371232


namespace greatest_two_digit_number_l371_371334

theorem greatest_two_digit_number (x y : ℕ) (h1 : x < y) (h2 : x * y = 12) : 10 * x + y = 34 :=
sorry

end greatest_two_digit_number_l371_371334


namespace length_of_BD_l371_371575

open Real

-- Variables and assumptions
variables (A B C D E : Type) [OrderedSemiring A]
variable [MetricSpace E] [Norm E]
variable [NormedEField E]
variables (a b c d e : A) (ac bc ab de bd : ℝ)

-- Conditions as definitions
def triangle_ABC (AC BC : ℝ) := (angle A C B = π / 2) ∧ (a = 7) ∧ (b = 24)
def points_on_AB_BC (D E : Type) := (D ∈ line_segment A B) ∧ (E ∈ line_segment B C)
def angle_BED := (angle B E D = π / 2)
def DE_length := (de = 10)

-- Problem statement
theorem length_of_BD :
  triangle_ABC a b ∧ points_on_AB_BC c d ∧ angle_BED ∧ DE_length d →
  bd = 250 / 7 :=
begin
  intros,
  sorry
end

end length_of_BD_l371_371575


namespace proportion_in_triangle_l371_371927

-- Definitions of the variables and conditions
variables {P Q R E : Point}
variables {p q r m n : ℝ}

-- Conditions
def angle_bisector_theorem (h : p = 2 * q) (h1 : m = q + q) (h2 : n = 2 * q) : Prop :=
  ∀ (p q r m n : ℝ), 
  (m / r) = (n / q) ∧ 
  (m + n = p) ∧
  (p = 2 * q)

-- The theorem to be proved
theorem proportion_in_triangle (h : p = 2 * q) (h1 : m / r = n / q) (h2 : m + n = p) : 
  (n / q = 2 * q / (r + q)) :=
by
  sorry

end proportion_in_triangle_l371_371927


namespace find_a_b_l371_371745

-- Given function
def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x^2

-- Definitions of the given conditions
def P : ℝ × ℝ := (-1, -2)
def perpendicular_line_slope := 3
def tangent_perpendicular_to_line (f' : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  f' P.1 = -1 / perpendicular_line_slope

-- Main statement 
theorem find_a_b (a b : ℝ) :
  (a = -1 ∧ b = -3) ∧ 
  (∀ m : ℝ, (∀ x y ∈ set.Icc 0 m, (f x a b ≤ f y a b) ∨ (f x a b ≥ f y a b)) ↔ m ∈ set.Ico (-2) 0) :=
sorry

end find_a_b_l371_371745


namespace binomial_sum_square_identity_l371_371986

open Nat

theorem binomial_sum_square_identity (n : ℕ) (k : ℕ) (h_k : k = (n - 1) / 2) : 
  ∑ r in Finset.range (k + 1), ((n - 2 * r) ^ 2 / n ^ 2) * (Nat.choose n r) ^ 2 = (1 / n) * Nat.choose (2 * n - 2) (n - 1) :=
by
  sorry

end binomial_sum_square_identity_l371_371986


namespace eq_y_as_x_l371_371776

theorem eq_y_as_x (y x : ℝ) : 
  (y = 2*x - 3*y) ∨ (x = 2 - 3*y) ∨ (-y = 2*x - 1) ∨ (y = x) → (y = x) :=
by
  sorry

end eq_y_as_x_l371_371776


namespace swimming_pool_volume_l371_371401

-- Definitions from the problem conditions
def pool_width : ℝ := 9
def pool_length : ℝ := 12
def shallow_depth : ℝ := 1
def deep_depth : ℝ := 4

-- Equivalent proof problem to show the volume of the swimming pool is 270 cubic meters.
theorem swimming_pool_volume : 
  let area_base := 0.5 * (shallow_depth + deep_depth) * pool_width in
  let volume_pool := area_base * pool_length in
  volume_pool = 270 :=
by 
  let area_base := 0.5 * (shallow_depth + deep_depth) * pool_width
  let volume_pool := area_base * pool_length
  have area_base_calc : area_base = 22.5 := by
    -- Proof that area_base calculation is correct
    sorry
  have volume_pool_calc : volume_pool = 270 := by
    -- Proof that volume_pool calculation is correct
    sorry
  exact volume_pool_calc

end swimming_pool_volume_l371_371401


namespace fraction_dad_roasted_l371_371958

theorem fraction_dad_roasted :
  ∀ (dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roast = joe_marshmallows / 2 →
    total_roast = 49 →
    dad_roast = total_roast - joe_roast →
    (dad_roast : ℚ) / (dad_marshmallows : ℚ) = 1 / 3 :=
by
  intros dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast
  intro h_dad_marshmallows
  intro h_joe_marshmallows
  intro h_joe_roast
  intro h_total_roast
  intro h_dad_roast
  sorry

end fraction_dad_roasted_l371_371958


namespace number_of_ways_to_choose_cards_l371_371148

-- Definitions based on the conditions in the problem
def standard_deck := 52
def number_of_suits := 4
def cards_in_hand := 3

-- The theorem we need to prove
theorem number_of_ways_to_choose_cards (deck : ℕ) (suits : ℕ) (hand : ℕ) :
  deck = standard_deck → suits = number_of_suits → hand = cards_in_hand →
  ∃ ways : ℕ, ways = 12168 :=
by
  intros deck_standard suits_four hand_three h1 h2 h3
  use 12168
  sorry

end number_of_ways_to_choose_cards_l371_371148


namespace find_cost_price_l371_371402

noncomputable def cost_price (CP : ℝ) : Prop :=
  let SP1 := CP * 0.88 in
  let SP2 := CP * 1.08 in
  SP2 = SP1 + 350

theorem find_cost_price : ∃ CP : ℝ, cost_price CP ∧ CP = 1750 := 
by
  use 1750
  unfold cost_price
  simp
  sorry

end find_cost_price_l371_371402


namespace probability_two_balls_same_box_l371_371282

theorem probability_two_balls_same_box :
  let total_balls := 3 in
  let total_boxes := 5 in
  let total_outcomes := total_boxes ^ total_balls in
  let favorable_outcomes := (nat.choose total_balls 2) * total_boxes * (total_boxes - 1) in
  (favorable_outcomes / total_outcomes : ℚ) = 12 / 25 :=
by
  -- sorry is used to skip the proof
  sorry

end probability_two_balls_same_box_l371_371282


namespace sally_saturday_sales_l371_371644

-- Define the problem's conditions
def saturday_sales (S : ℕ) :=
  let sunday_sales := 1.5 * S
  let monday_sales := 1.95 * S
  S + sunday_sales + monday_sales = 290

-- Prove that the number of boxes sold on Saturday is 65
theorem sally_saturday_sales : ∃ S : ℕ, saturday_sales S ∧ S = 65 :=
by
  exists 65
  have h₁ : 1.5 * 65 = 97.5 := by norm_num
  have h₂ : 1.95 * 65 = 126.75 := by norm_num
  have h₃ : 65 + 97.5 + 126.75 = 290 := by norm_num
  unfold saturday_sales
  rw h₁
  rw h₂
  rw h₃
  split
  · rfl
  · rfl
  sorry

end sally_saturday_sales_l371_371644


namespace count_convex_quad_diagonals_l371_371043

theorem count_convex_quad_diagonals (n : ℕ) (h : n ≥ 6) :
  let binom := λ (n k : ℕ), nat.choose n k 
  in (n / 4) * binom (n-5) 3 = binom (n-3) 4 - binom (n-5) 2 :=
by sorry

end count_convex_quad_diagonals_l371_371043


namespace igor_number_l371_371411

-- Define the initial lineup
def initial_lineup : List ℕ := [2, 9, 3, 11, 7, 10, 6, 8, 5, 4, 1]

-- Define the command criteria function
def command_criteria (lineup : List ℕ) : List ℕ :=
  lineup.filter (λ x, ∀ y ∈ List.neighbors lineup x, x < y)

noncomputable def remaining_after_Igor_left (lineup : List ℕ) : List ℕ :=
  (command_criteria ∘ command_criteria ∘ command_criteria ∘ command_criteria) lineup

-- The main theorem
theorem igor_number :
  ∃ n, n ∈ initial_lineup ∧ 
       List.length (remaining_after_Igor_left (initial_lineup.filter (λ x, x ≠ n))) = 3 ∧ 
       n = 5 :=
by
  sorry

end igor_number_l371_371411


namespace lemonade_glasses_from_fruit_l371_371196

noncomputable def lemons_per_glass : ℕ := 2
noncomputable def oranges_per_glass : ℕ := 1
noncomputable def total_lemons : ℕ := 18
noncomputable def total_oranges : ℕ := 10
noncomputable def grapefruits : ℕ := 6
noncomputable def lemons_per_grapefruit : ℕ := 2
noncomputable def oranges_per_grapefruit : ℕ := 1

theorem lemonade_glasses_from_fruit :
  (total_lemons / lemons_per_glass) = 9 →
  (total_oranges / oranges_per_glass) = 10 →
  min (total_lemons / lemons_per_glass) (total_oranges / oranges_per_glass) = 9 →
  (grapefruits * lemons_per_grapefruit = 12) →
  (grapefruits * oranges_per_grapefruit = 6) →
  (9 + grapefruits) = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lemonade_glasses_from_fruit_l371_371196


namespace power_division_l371_371333

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l371_371333


namespace find_n_in_mod_range_l371_371908

theorem find_n_in_mod_range (a b : ℤ) (n : ℤ) 
  (h1 : a ≡ 54 [MOD 53]) 
  (h2 : b ≡ 98 [MOD 53]) 
  (h3 : 150 ≤ n ∧ n ≤ 200) 
  (h4 : a - b ≡ n [MOD 53]) : 
  n = 168 :=
  sorry

end find_n_in_mod_range_l371_371908


namespace a_sq_minus_b_sq_l371_371117

noncomputable def vec_diff : ℤ × ℤ → ℤ := λ v,
  (v.1 * v.1 + v.2 * v.2)

theorem a_sq_minus_b_sq (a b : ℤ × ℤ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0))
  (h1 : a.1 + b.1 = -3 ∧ a.2 + b.2 = 6)
  (h2 : a.1 - b.1 = -3 ∧ a.2 - b.2 = 2) :
  vec_diff a - vec_diff b = 21 := 
sorry

end a_sq_minus_b_sq_l371_371117


namespace rojo_speed_l371_371643

theorem rojo_speed (R : ℝ) 
  (H : 32 = (R + 3) * 4) : R = 5 :=
sorry

end rojo_speed_l371_371643


namespace find_a_b_find_min_k_harmonic_ln_inequality_l371_371103

noncomputable def f (x : ℝ) (a b : ℝ) := a * x ^ 3 + b * x ^ 2

-- Definitions based on the problem conditions
axiom extreme_value_at_one (a b : ℝ) : f 1 a b = 1/6
axiom extreme_point_derivative (a b : ℝ) : (3 * a * 1 ^ 2 + 2 * b * 1) = 0

theorem find_a_b (a b : ℝ) : a = -1/3 ∧ b = 1/2 :=
sorry

-- Definitions for the inequality f'(x) ≤ k ln (x + 1)
noncomputable def f' (x : ℝ) (a b : ℝ) := 3 * a * x ^ 2 + 2 * b * x

axiom derivative_inequality (k : ℝ) (x : ℝ) (a b : ℝ) :
  x ∈ Set.Ici 0 → f' x a b ≤ k * Real.log (x + 1)

theorem find_min_k : 1 ≤ 1 :=
by
  -- Value derived from the problem's conclusion
  sorry

-- Series inequality problem
theorem harmonic_ln_inequality (n : ℕ) (hn : 0 < n) : 
  (∑ i in Finset.range (n + 1), 1 / (i + 1)) < Real.log (n + 1) + 2 :=
sorry

end find_a_b_find_min_k_harmonic_ln_inequality_l371_371103


namespace cost_to_paint_cube_l371_371359

theorem cost_to_paint_cube : 
  let cost_per_kg := 60 in
  let coverage_per_kg := 20 in
  let side_length := 10 in
  let surface_area := 6 * (side_length * side_length) in
  let paint_needed := surface_area / coverage_per_kg in
  let total_cost := paint_needed * cost_per_kg in
  total_cost = 1800 :=
by
  let cost_per_kg := 60
  let coverage_per_kg := 20
  let side_length := 10
  let surface_area := 6 * (side_length * side_length)
  let paint_needed := surface_area / coverage_per_kg
  let total_cost := paint_needed * cost_per_kg
  sorry

end cost_to_paint_cube_l371_371359


namespace number_satisfies_equation_l371_371816

theorem number_satisfies_equation :
  ∃ x : ℝ, (0.6667 * x - 10 = 0.25 * x) ∧ (x = 23.9936) :=
by
  sorry

end number_satisfies_equation_l371_371816


namespace power_function_property_l371_371897

theorem power_function_property (m : ℤ) (h1 : m^2 - 2 * m - 3 ≤ 0) (h2 : even (m^2 - 2 * m - 3)) :
  m = -1 ∨ m = 1 ∨ m = 3 := by sorry

end power_function_property_l371_371897


namespace median_unchanged_l371_371177

def donationAmountsOrig : List ℕ := [30, 50, 50, 60, 60]
def donationAmountsNew : List ℕ := [50, 50, 50, 60, 60]

theorem median_unchanged (dOrig dNew : List ℕ) :
  dOrig = [30, 50, 50, 60, 60] →
  dNew  = [50, 50, 50, 60, 60] →
  List.median dOrig = List.median dNew :=
by
  sorry

end median_unchanged_l371_371177


namespace leo_current_weight_l371_371910

theorem leo_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 140) : 
  L = 80 :=
by 
  sorry

end leo_current_weight_l371_371910


namespace parallelogram_area_correct_l371_371929

noncomputable def parallelogram_area (AB AC BD : ℝ) (H1 : AB = 4) (H2 : AC = 4) (H3 : DiagonalBisectsAngles BD) : ℝ :=
  8 * Real.sqrt 3

theorem parallelogram_area_correct :
  parallelogram_area 4 4 (DiagonalBisectsAngles.something) = 8 * Real.sqrt 3 :=
sorry

end parallelogram_area_correct_l371_371929


namespace find_m_l371_371418

-- Define the conditions
variables (a b r s m : ℝ)
variables (S1 S2 : ℝ)

-- Conditions
def first_term_first_series := a = 12
def second_term_first_series := a * r = 6
def first_term_second_series := b = 12
def second_term_second_series := 12 * s = 6 + m
def sum_relation := S2 = 3 * S1
def sum_first_series := S1 = a / (1 - r)
def sum_second_series := S2 = b / (1 - s)

-- Proof statement
theorem find_m (h1 : first_term_first_series a)
              (h2 : second_term_first_series a r)
              (h3 : first_term_second_series b)
              (h4 : second_term_second_series s m)
              (h5 : sum_relation S2 S1)
              (h6 : sum_first_series S1 a r)
              (h7 : sum_second_series S2 b s) : 
              m = 4 := 
sorry

end find_m_l371_371418


namespace calculate_expression_l371_371425

theorem calculate_expression :
  (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end calculate_expression_l371_371425


namespace sum_original_numbers_is_five_l371_371404

noncomputable def sum_original_numbers (a b c d : ℤ) : ℤ :=
  a + b + c + d

theorem sum_original_numbers_is_five (a b c d : ℤ) (hab : 10 * a + b = overline_ab) 
  (h : 100 * (10 * a + b) + 10 * c + 7 * d = 2024) : sum_original_numbers a b c d = 5 :=
sorry

end sum_original_numbers_is_five_l371_371404


namespace f_p_equal_2_l371_371465

def is_parallel (l1 l2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a1 b1 c1 a2 b2 c2 : ℝ), l1 = λ x y => a1 * x + b1 * y + c1 ∧ l2 = λ x y => a2 * x + b2 * y + c2 ∧ (a1 * b2 - a2 * b1 = 0)

def f_p (p : Prop) : ℕ :=
  if p ∧ (¬p ∨ (is_parallel ∧ ¬is_parallel)) then 2 else 0

theorem f_p_equal_2 (l1 l2 : ℝ → ℝ → ℝ) (h : is_parallel l1 l2) : f_p (is_parallel l1 l2) = 2 :=
by
  sorry

end f_p_equal_2_l371_371465


namespace no_solution_t_s_l371_371806

noncomputable def find_k : ℝ :=
by sorry

theorem no_solution_t_s (k : ℝ) : 
  (\forall t s : ℝ, (⟨1, 3⟩ : ℝ × ℝ) + t • ⟨5, -2⟩ ≠ (⟨-2, 4⟩ : ℝ × ℝ) + s • ⟨2, k⟩) ↔ k = -4/5 :=
by sorry

end no_solution_t_s_l371_371806


namespace find_amount_l371_371364

noncomputable def x : ℝ := 170
noncomputable def A : ℝ := 552.5

theorem find_amount (h: 0.65 * x = 0.20 * A) : A = 552.5 := by
  have := calc
    A = (0.65 * x) / 0.20 : by field_simp [h]
    ... = 110.5 / 0.20    : by norm_num
    ... = 552.5           : by norm_num
  exact this

end find_amount_l371_371364


namespace sin_cos_power_sum_l371_371214

theorem sin_cos_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := 
by
  sorry

end sin_cos_power_sum_l371_371214


namespace area_of_triangle_DEF_correct_l371_371190

noncomputable def area_of_triangle_DEF {D E F L : Type} [real D] [real E] [real F] [real L] : real :=
  let DE := 12
  let EL := 9
  let EF := 17
  let DL := real.sqrt(63)
  (1 / 2) * EF * DL

theorem area_of_triangle_DEF_correct :
  ∀ [triangle DEF] 
  (H1 : ∃ L : triangle DEF, L ∈ EF ∧ DL_altitude DEF)
  (H2 : DE = 12)
  (H3 : EL = 9)
  (H4 : EF = 17),
  area_of_triangle_DEF = 51 * (real.sqrt(7)) / 2 :=
by sorry

end area_of_triangle_DEF_correct_l371_371190


namespace find_number_l371_371711

theorem find_number (x : ℤ) (h : x + x^2 + 15 = 96) : x = -9 :=
sorry

end find_number_l371_371711


namespace probability_interval_l371_371735

open Set

def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def interval_probability {a b c d : ℝ} (h1: a < b) (h2: c < d)
    (h3: a ≤ c) (h4: d ≤ b) : ℝ :=
  interval_length c d / interval_length a b

theorem probability_interval (a b : ℝ) (h1 : 0 < 5) (h_interval : (3:ℝ) < (4:ℝ)) 
    (h_bounds1 : (0:ℝ) < (3:ℝ)) (h_bounds2 : (4:ℝ) < (5:ℝ)) :
    interval_probability (0:ℝ) (5:ℝ) (3:ℝ) (4:ℝ) h1 h_interval h_bounds1 h_bounds2 = (1/5:ℝ) := 
sorry

end probability_interval_l371_371735


namespace sum_of_roots_eq_zero_l371_371056

theorem sum_of_roots_eq_zero :
  ∀ (x : ℝ), x^2 - 7 * |x| + 6 = 0 → (∃ a b c d : ℝ, a + b + c + d = 0) :=
by
  sorry

end sum_of_roots_eq_zero_l371_371056


namespace common_root_unique_solution_l371_371838

theorem common_root_unique_solution
  (p : ℝ) (h : ∃ x, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) :
  p = 3 :=
by sorry

end common_root_unique_solution_l371_371838


namespace sequence_sum_value_l371_371482

-- Definition of the sequence
noncomputable def a_seq (n : ℕ) : ℕ := if n = 1 then 4 else 4 * n * n

-- Given condition on the sequence
def sequence_condition (n : ℕ) : Prop :=
  (∑ i in Finset.range n, Real.sqrt (a_seq (i + 1))) = n^2 + n

-- Statement to prove
theorem sequence_sum_value (n : ℕ) (h : sequence_condition n) :
  (∑ i in Finset.range n, a_seq (i + 1) / (i + 1)) = 2 * n^2 + 2 * n :=
sorry

end sequence_sum_value_l371_371482


namespace sum_of_six_primes_even_l371_371280

/-- If A, B, and C are positive integers such that A, B, C, A-B, A+B, and A+B+C are all prime numbers, 
    and B is specifically the prime number 2,
    then the sum of these six primes is even. -/
theorem sum_of_six_primes_even (A B C : ℕ) (hA : Prime A) (hB : Prime B) (hC : Prime C) 
    (h1 : Prime (A - B)) (h2 : Prime (A + B)) (h3 : Prime (A + B + C)) (hB_eq_two : B = 2) : 
    Even (A + B + C + (A - B) + (A + B) + (A + B + C)) :=
by
  sorry

end sum_of_six_primes_even_l371_371280


namespace right_triangle_sides_l371_371657

/-- Given a right triangle with area 2 * r^2 / 3 where r is the radius of a circle touching one leg,
the extension of the other leg, and the hypotenuse, the sides of the triangle are given by r, 4/3 * r, and 5/3 * r. -/
theorem right_triangle_sides (r : ℝ) (x y : ℝ)
  (h_area : (x * y) / 2 = 2 * r^2 / 3)
  (h_hypotenuse : (x^2 + y^2) = (2 * r + x - y)^2) :
  x = r ∧ y = 4 * r / 3 :=
sorry

end right_triangle_sides_l371_371657


namespace sum_of_g_49_l371_371223

noncomputable def f (x : ℝ) : ℝ := 4 * x ^ 2 + 1
noncomputable def g (y : ℝ) : ℝ := if y = 49 then (2 * sqrt 3) ^ 2 - 2 * sqrt 3 + 1 else y * y - y + 1

theorem sum_of_g_49 : g 49 + g 49 = 26 := by
  sorry

end sum_of_g_49_l371_371223


namespace sum_first_9_terms_l371_371181

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

def a_4 (a q : ℝ) : ℝ := geometric_sequence a q 4
def a_7 (a q : ℝ) : ℝ := geometric_sequence a q 7
def a_3 (a q : ℝ) : ℝ := geometric_sequence a q 3
def a_6 (a q : ℝ) : ℝ := geometric_sequence a q 6
def a_9 (a q : ℝ) : ℝ := geometric_sequence a q 9

axiom a1_a4_a7 (a q : ℝ) : a + a_4 a q + a_7 a q = 2
axiom a3_a6_a9 (a q : ℝ) : a_3 a q + a_6 a q + a_9 a q = 18

theorem sum_first_9_terms (a q : ℝ) :
  let a_2 := geometric_sequence a q 2,
      a_5 := geometric_sequence a q 5,
      a_8 := geometric_sequence a q 8,
      S_9 := a + a_4 a q + a_7 a q + a_2 + a_5 + a_8 + a_3 a q + a_6 a q + a_9 a q
  in S_9 = 14 ∨ S_9 = 26 :=
by
  sorry

end sum_first_9_terms_l371_371181


namespace algebraic_expression_value_l371_371652

theorem algebraic_expression_value (x : ℝ) (hx : x = Real.sqrt 7 + 1) :
  (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3)) = Real.sqrt 7 - 1 :=
by
  sorry

end algebraic_expression_value_l371_371652


namespace polynomial_remainder_l371_371730

theorem polynomial_remainder :
  let p := 3 * X^2 - 19 * X + 53
  let d := X - 3
  polynomial.divModByMonic p d = (q, 23)
:=
sorry

end polynomial_remainder_l371_371730


namespace jude_spends_75_percent_on_cars_l371_371592

theorem jude_spends_75_percent_on_cars :
  ∀ (total_bottle_caps cost_per_car cost_per_truck trucks_bought total_vehicles cars_bought cars_cost remaining_bottle_caps : ℕ),
  total_bottle_caps = 100 →
  cost_per_car = 5 →
  cost_per_truck = 6 →
  trucks_bought = 10 →
  total_vehicles = 16 →
  cars_bought = total_vehicles - trucks_bought →
  remaining_bottle_caps = total_bottle_caps - (trucks_bought * cost_per_truck) →
  cars_cost = cars_bought * cost_per_car →
  100 * cars_cost = 75 * remaining_bottle_caps :=
by
  intros total_bottle_caps cost_per_car cost_per_truck trucks_bought total_vehicles cars_bought cars_cost remaining_bottle_caps
  assume h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end jude_spends_75_percent_on_cars_l371_371592


namespace pascals_triangle_multiples_of_3_l371_371167

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def is_multiple_of_3 (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → binomial n k % 3 = 0

def count_multiples_of_3_rows (rows : ℕ) : ℕ :=
  Nat.iterate rows 0 (λ n acc, if is_multiple_of_3 n then acc + 1 else acc)

theorem pascals_triangle_multiples_of_3 :
  count_multiples_of_3_rows 40 = 6 :=
sorry

end pascals_triangle_multiples_of_3_l371_371167


namespace evaluate_star_l371_371037

variables (R : Type*) [Field R]

def star (a b c x y : R) := a * x + b * y + c

theorem evaluate_star {
  a b c : R} 
  (h1 : a + 2 * b + c = 9) 
  (h2 : -3 * a + 3 * b + c = 6)
  (h3 : b + c = 2) :
  star a b c (-2) 5 = 18 := 
sorry

end evaluate_star_l371_371037


namespace equal_segments_l371_371576

variables {A B C D E F : Type} [metric_space A]
variables {a b c : ℝ}
variables {AC BC : ℝ}

hypothesis {h1 : ∀ (A B C D E F : A), B ∈ (segment A C) → D ∈ (segment A B) → F ∈ (segment B C) → DF ∩ AC = E} 
hypothesis {h2 : b = c}

theorem equal_segments (b c : ℝ) : AC = BC → (CE = BD ↔ DF = EF) :=
begin
  intro h, sorry
end

end equal_segments_l371_371576


namespace paths_from_A_to_B_no_revisits_l371_371139

noncomputable def numPaths : ℕ :=
  16

theorem paths_from_A_to_B_no_revisits : numPaths = 16 :=
by
  sorry

end paths_from_A_to_B_no_revisits_l371_371139


namespace PR_eq_AD_l371_371940

theorem PR_eq_AD 
  (A B C D E F M N O P R : Type)
  [InTriangle A B C]
  (is_midpoint_D : Midpoint D B C)
  (is_midpoint_E : Midpoint E A B)
  (is_midpoint_F : Midpoint F C A)
  (M_meets_internal_bisector_ADB : Meets M (InternalBisector AD B))
  (N_meets_internal_bisector_ADC : Meets N (InternalBisector AD C))
  (O_intersection_AD_MN : O = IntersectionPoint (LineThrough A D) (LineThrough M N))
  (P_intersection_AB_FO : P = IntersectionPoint (LineThrough A B) (LineThrough F O))
  (R_intersection_AC_EO : R = IntersectionPoint (LineThrough A C) (LineThrough E O)) :
  PR = AD :=
by
  sorry

end PR_eq_AD_l371_371940


namespace regular_hexagon_shaded_area_l371_371396

theorem regular_hexagon_shaded_area (side_length : ℝ) (hexagon : regular_hexagon) 
  (h_side_length : hexagon.side_length = 12) : 
  ∃ area : ℝ, area = 72 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_shaded_area_l371_371396


namespace equal_distribution_l371_371740

theorem equal_distribution 
  (total_profit : ℕ) 
  (num_employees : ℕ) 
  (profit_kept_percent : ℕ) 
  (remaining_to_distribute : ℕ)
  (each_employee_gets : ℕ) :
  total_profit = 50 →
  num_employees = 9 →
  profit_kept_percent = 10 →
  remaining_to_distribute = total_profit - (total_profit * profit_kept_percent / 100) →
  each_employee_gets = remaining_to_distribute / num_employees →
  each_employee_gets = 5 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_distribution_l371_371740


namespace find_x_l371_371473

def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (4, x)

theorem find_x (x : ℝ) (h : ∃k : ℝ, b x = (k * a.1, k * a.2)) : x = 6 := 
by 
  sorry

end find_x_l371_371473


namespace proposition1_correct_proposition2_incorrect_proposition3_incorrect_proposition4_correct_l371_371094

-- Definitions for the conditions
variables (m l : Set Point) (α β : Set Point) (A : Point) (cond1 cond2 cond3 cond4 : Prop)

hypothesis cond1 : m ⊆ α ∧ {A} = l ∩ α ∧ A ∉ m
hypothesis cond2 : m ⊆ α ∧ l ⊆ β ∧ (m // l) -- assuming // represents parallelism
hypothesis cond3 : m ⊆ α ∧ l ⊆ α ∧ (m // β) ∧ (l // β) ∧ {A} = l ∩ m
hypothesis cond4 : {m} = α ∩ β ∧ (l // m) ∧ l ⊈ α ∧ l ⊈ β

-- Propositions correctness
theorem proposition1_correct : handle_proposition1_correct (=cond1),
theorem proposition2_incorrect : handle_proposition2_incorrect (=cond2),
theorem proposition3_incorrect : handle_proposition3_incorrect (=cond3),
theorem proposition4_correct : handle_proposition4_correct (=cond4),
sorry

end proposition1_correct_proposition2_incorrect_proposition3_incorrect_proposition4_correct_l371_371094


namespace count_visible_factor_numbers_in_range_l371_371436

def visible_factor_number (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  let non_zero_digits := digits.filter (λ d => d ≠ 0)
  ∀ d ∈ non_zero_digits, n % d = 0

theorem count_visible_factor_numbers_in_range : 
  (Finset.filter visible_factor_number (Finset.range 51).map (λ x, x + 200)).card = 31 :=
by
  sorry

end count_visible_factor_numbers_in_range_l371_371436


namespace team_overall_progress_is_89_l371_371236

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

def overall_progress (changes : List Int) : Int :=
  changes.sum

theorem team_overall_progress_is_89 :
  overall_progress yard_changes = 89 :=
by
  sorry

end team_overall_progress_is_89_l371_371236


namespace remainder_of_S_mod_1000_l371_371973

def digit_contribution (d pos : ℕ) : ℕ := (d * d) * pos

def sum_of_digits_with_no_repeats : ℕ :=
  let thousands := (16 + 25 + 36 + 49 + 64 + 81) * (9 * 8 * 7) * 1000
  let hundreds := (16 + 25 + 36 + 49 + 64 + 81) * (8 * 7 * 6) * 100
  let tens := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 10
  let units := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 1
  thousands + hundreds + tens + units

theorem remainder_of_S_mod_1000 : (sum_of_digits_with_no_repeats % 1000) = 220 :=
  by
  sorry

end remainder_of_S_mod_1000_l371_371973


namespace outer_circle_radius_l371_371293

theorem outer_circle_radius (r R : ℝ) (hr : r = 4)
  (radius_increase : ∀ R, R' = 1.5 * R)
  (radius_decrease : ∀ r, r' = 0.75 * r)
  (area_increase : ∀ (A1 A2 : ℝ), A2 = 3.6 * A1)
  (initial_area : ∀ A1, A1 = π * R^2 - π * r^2)
  (new_area : ∀ A2 R' r', A2 = π * R'^2 - π * r'^2) :
  R = 6 := sorry

end outer_circle_radius_l371_371293


namespace weight_replacement_proof_l371_371659

noncomputable def weight_of_replaced_person (increase_in_average_weight new_person_weight : ℝ) : ℝ :=
  new_person_weight - (5 * increase_in_average_weight)

theorem weight_replacement_proof (h1 : ∀ w : ℝ, increase_in_average_weight = 5.5) (h2 : new_person_weight = 95.5) :
  weight_of_replaced_person 5.5 95.5 = 68 := by
  sorry

end weight_replacement_proof_l371_371659


namespace scientific_notation_example_l371_371407

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end scientific_notation_example_l371_371407


namespace find_smaller_number_l371_371661

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  sorry

end find_smaller_number_l371_371661


namespace scientific_notation_of_284000000_l371_371409

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end scientific_notation_of_284000000_l371_371409


namespace lim_n_b_n_l371_371802

noncomputable def M (x : ℝ) : ℝ := x - (x^3) / 3

def iter_M (k : ℕ) (x : ℝ) : ℝ := Nat.recOn k x (fun _ y => M y)

def b_n (n : ℕ) : ℝ := iter_M n (20 / (n : ℝ))

open Filter

theorem lim_n_b_n : tendsto (fun (n : ℕ) => n * b_n n) atTop (𝓝 (60 / 61)) :=
sorry

end lim_n_b_n_l371_371802


namespace number_of_terminal_zeros_l371_371437

theorem number_of_terminal_zeros (a b c : ℕ) (h1 : a = 40) (h2 : b = 360) (h3 : c = 125) : 
  let prod := a * b * c in
  let zeros := 5 in
  nat.num_of_terminal_zeros prod = zeros :=
by
  sorry

end number_of_terminal_zeros_l371_371437


namespace jackson_volume_discount_l371_371584

-- Given conditions as parameters
def hotTubVolume := 40 -- gallons
def quartsPerGallon := 4 -- quarts per gallon
def bottleVolume := 1 -- quart per bottle
def bottleCost := 50 -- dollars per bottle
def totalSpent := 6400 -- dollars spent by Jackson

-- Calculation related definitions
def totalQuarts := hotTubVolume * quartsPerGallon
def totalBottles := totalQuarts / bottleVolume
def costWithoutDiscount := totalBottles * bottleCost
def discountAmount := costWithoutDiscount - totalSpent
def discountPercentage := (discountAmount / costWithoutDiscount) * 100

-- The proof problem
theorem jackson_volume_discount : discountPercentage = 20 :=
by
  sorry

end jackson_volume_discount_l371_371584


namespace correct_distance_l371_371089

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

structure Vector3D :=
  (i : ℝ) (j : ℝ) (k : ℝ)

def pointA : Point3D := ⟨-1, 2, 1⟩
def pointP : Point3D := ⟨1, 2, -2⟩
def normalVector : Vector3D := ⟨2, 0, 1⟩

def vectorBetweenPoints (A P : Point3D) : Vector3D :=
  ⟨P.x - A.x, P.y - A.y, P.z - A.z⟩

def dotProduct (v w : Vector3D) : ℝ :=
  v.i * w.i + v.j * w.j + v.k * w.k

def magnitude (v : Vector3D) : ℝ :=
  Real.sqrt (v.i * v.i + v.j * v.j + v.k * v.k)

def distanceFromPointToPlane (A P : Point3D) (normalVector : Vector3D) : ℝ :=
  (abs (dotProduct (vectorBetweenPoints A P) normalVector)) / (magnitude normalVector)

theorem correct_distance :
  distanceFromPointToPlane pointA pointP normalVector = (Real.sqrt 5) / 5 :=
by
  sorry

end correct_distance_l371_371089


namespace part_I_monotonicity_part_II_value_a_l371_371097

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (x - 1)

def is_monotonic_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x < f y

def is_monotonic_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

theorem part_I_monotonicity :
  (is_monotonic_increasing f {x | 2 < x}) ∧
  ((is_monotonic_decreasing f {x | x < 1}) ∧ (is_monotonic_decreasing f {x | 1 < x ∧ x < 2})) :=
by
  sorry

theorem part_II_value_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → (Real.exp x * (x - 2)) / ((x - 1)^2) ≥ a * (Real.exp x / (x - 1))) → a ∈ Set.Iic 0 :=
by
  sorry

end part_I_monotonicity_part_II_value_a_l371_371097


namespace race_permutations_l371_371127

theorem race_permutations (r1 r2 r3 r4 : Type) [decidable_eq r1] [decidable_eq r2] [decidable_eq r3] [decidable_eq r4] :
  fintype.card (finset.univ : finset {l : list r1 | l ~ [r1, r2, r3, r4]}) = 24 :=
by
  sorry

end race_permutations_l371_371127


namespace same_profit_and_loss_selling_price_l371_371283

theorem same_profit_and_loss_selling_price (CP SP : ℝ) (h₁ : CP = 49) (h₂ : (CP - 42) = (SP - CP)) : SP = 56 :=
by 
  sorry

end same_profit_and_loss_selling_price_l371_371283


namespace coefficient_of_x7_in_expansion_of_2_minus_x_power_10_l371_371455

open BigOperators

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def term_in_expansion (a b : ℝ) (n r : ℕ) : ℝ :=
  binomial_coeff n r * a^(n-r) * b^r

theorem coefficient_of_x7_in_expansion_of_2_minus_x_power_10 :
  ∀ (x : ℝ), coeff (term_in_expansion 2 (-x) 10 7) x = -960 :=
by
  intros
  sorry

end coefficient_of_x7_in_expansion_of_2_minus_x_power_10_l371_371455


namespace remaining_amount_needed_l371_371705

def goal := 150
def earnings_from_3_families := 3 * 10
def earnings_from_15_families := 15 * 5
def total_earnings := earnings_from_3_families + earnings_from_15_families
def remaining_amount := goal - total_earnings

theorem remaining_amount_needed : remaining_amount = 45 := by
  sorry

end remaining_amount_needed_l371_371705


namespace height_on_hypotenuse_l371_371504

theorem height_on_hypotenuse (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10)
  (right_triangle : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1/2 * c * h = 1/2 * a * b) ∧ (h = 4.8) :=
by
  use 4.8
  split
  sorry

end height_on_hypotenuse_l371_371504


namespace minimal_polynomial_roots_l371_371828

def is_minimal_polynomial (p : polynomial ℚ) (r : ℚ) : Prop :=
  p.eval r = 0 ∧ p.leading_coeff = 1 ∧ ∀ (q : polynomial ℚ), q.degree < p.degree → ¬ q.eval r = 0

noncomputable def polynomial_with_roots : polynomial ℚ := 
  (polynomial.X^2 - 4 * polynomial.X - 1) * (polynomial.X^2 - 6 * polynomial.X + 2)

theorem minimal_polynomial_roots :
  is_minimal_polynomial polynomial_with_roots (2 + real.sqrt 5) ∧
  is_minimal_polynomial polynomial_with_roots (3 + real.sqrt 7) :=
by
  -- sorry to skip the proof steps
  sorry

end minimal_polynomial_roots_l371_371828


namespace complete_square_eq_l371_371348

theorem complete_square_eq (x : ℝ) : x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  have : x^2 - 2 * x = 5 := by linarith
  have : x^2 - 2 * x + 1 = 6 := by linarith
  exact eq_of_sub_eq_zero (by linarith)

end complete_square_eq_l371_371348


namespace find_lambda_l371_371546

-- Definitions for the equations of the hyperbola and parabola, and the coinciding foci.
theorem find_lambda (λ : ℝ) (hλ : λ ≠ 0) 
  (H_hyperbola : ∃ (x y : ℝ), y^2 - 3 * x^2 = λ ∧ (x, y) = (2, 0))
  (H_parabola : ∃ (y : ℝ), y^2 = 8 * 2):
  λ = -3 :=
by
  sorry

end find_lambda_l371_371546


namespace number_of_true_propositions_l371_371489

open Classical

def p := ∅ ⊆ {0}
def q := ¬ ({1} ∈ {1, 2})

def p_or_q := p ∨ q
def p_and_q := p ∧ q
def not_p := ¬ p

theorem number_of_true_propositions : (p_or_q ∧ ¬ p_and_q ∧ ¬ not_p) = True :=
by
  sorry

end number_of_true_propositions_l371_371489


namespace boys_attending_school_dance_l371_371714

-- Define the parameters according to the conditions provided.
def total_attendees : ℕ := 100
def faculty_and_staff_percentage : ℝ := 0.10
def fraction_girls : ℝ := 2 / 3

-- Provide the statement to prove the number of boys attending the school dance
theorem boys_attending_school_dance :
  let faculty_and_staff := (faculty_and_staff_percentage * total_attendees).toNat in
  let students := total_attendees - faculty_and_staff in
  let girls := (fraction_girls * students).toNat in
  let boys := students - girls in
  boys = 30 := 
by
  sorry

end boys_attending_school_dance_l371_371714


namespace smallest_m_l371_371579

theorem smallest_m (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n - m / n = 2011 / 3) : m = 1120 :=
sorry

end smallest_m_l371_371579


namespace decimal_rep_irrational_l371_371969

-- Define the sequence a_n
def sequence_a : ℕ → ℕ
| 0     := 2
| (n+1) := int.floor (3 * sequence_a n / 2)

-- Define a function to convert the sequence to a decimal representation
noncomputable def decimal_representation : ℕ → ℕ → ℕ
| a 0     := a
| a (n+1) := a + 10^(-(n+1)) * sequence_a (n+1)

-- The main theorem statement
theorem decimal_rep_irrational : ¬ ∃ q : ℚ, ∃ n : ℕ, decimal_representation (sequence_a 0) n = ↑q :=
by sorry

end decimal_rep_irrational_l371_371969


namespace counterfeit_coin_identifiable_in_two_weighings_l371_371943

/-- One of the four coins is counterfeit and differs in weight from the real ones.
    We state that the counterfeit coin can be identified in 2 weighings. -/
theorem counterfeit_coin_identifiable_in_two_weighings :
  (∃ (coins : Fin 4 → ℕ), 
  (∃ i : Fin 4, ∀ j : Fin 4, j ≠ i → coins j = 1) ∧ (coins i ≠ 1) → 
  (∃ (w1 w2 : Fin 2 → Fin 4), 
   (w1 ≠ w2) ∧  ∀ b : Fin 2 → bool, weigh coins w1 w2 b = true) ) :=
sorry

end counterfeit_coin_identifiable_in_two_weighings_l371_371943


namespace prob_X_ge_one_l371_371161

noncomputable def normal_distribution (mean variance : ℝ) : PMF ℝ :=
sorry -- PMF definition for normal distribution

noncomputable def X : PMF ℝ := normal_distribution (-1) (σ^2)

axiom prob_given : pmf.probability (set.Icc (-3) (-1)) X = 0.4

theorem prob_X_ge_one :
  pmf.probability (set.Ici 1) X = 0.1 :=
sorry

end prob_X_ge_one_l371_371161


namespace length_of_AB_of_intersection_l371_371090

theorem length_of_AB_of_intersection (x y : ℝ) :
  (x^2 + y^2 + 4*x - 4*y - 10 = 0) → (2*x - y + 1 = 0) → 
  2 * (sqrt ((3*sqrt 2)^2 - (3 / sqrt 5)^2)) = 9 * sqrt 5 / 5 :=
by
  intro h_circle h_line
  sorry

end length_of_AB_of_intersection_l371_371090


namespace positive_integer_solutions_l371_371820

theorem positive_integer_solutions (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) :
  1 + 2^x + 2^(2*x+1) = y^n ↔ 
  (x = 4 ∧ y = 23 ∧ n = 2) ∨ (∃ t : ℕ, 0 < t ∧ x = t ∧ y = 1 + 2^t + 2^(2*t+1) ∧ n = 1) :=
sorry

end positive_integer_solutions_l371_371820


namespace petya_square_larger_l371_371630

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end petya_square_larger_l371_371630


namespace planes_parallel_from_skew_lines_l371_371219

-- Definitions of parallelism and skew lines
structure Line : Type :=
(point : ℝ × ℝ × ℝ)
(direction: ℝ × ℝ × ℝ)

structure Plane : Type :=
(point : ℝ × ℝ × ℝ)
(normal: ℝ × ℝ × ℝ)

def are_parallel_lines (l1 l2 : Line) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ l1.direction = (k • l2.direction)

def are_parallel_planes (p1 p2 : Plane) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ p1.normal = (k • p2.normal)

def are_skew_lines (l1 l2 : Line) : Prop := 
  ∀ p : Plane, ¬ (l1 ∈ p ∧ l2 ∈ p)

-- Given lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Conditions for the problem
axiom distinct_lines : m ≠ n
axiom non_coincident_planes : α ≠ β
axiom skew_lines : are_skew_lines m n
axiom m_parallel_α : are_parallel_lines m {point := α.point, direction := α.normal}
axiom m_parallel_β : are_parallel_lines m {point := β.point, direction := β.normal}
axiom n_parallel_α : are_parallel_lines n {point := α.point, direction := α.normal}
axiom n_parallel_β : are_parallel_lines n {point := β.point, direction := β.normal}

-- The statement to be proved
theorem planes_parallel_from_skew_lines : are_parallel_planes α β :=
  sorry

end planes_parallel_from_skew_lines_l371_371219


namespace no_integer_roots_l371_371978

-- Define the context of the problem
variables {a₀ a₁ a₂ : ℤ}  -- Generalized for simplicity, can extend to a_n

noncomputable theory

-- Define the polynomial f(x) with integer coefficients
def f (a₀ a₁ a₂ : ℤ) (x : ℤ) : ℤ :=
  a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃  -- Example for cubic polynomial

-- Lean statement for the problem
theorem no_integer_roots (a₀ a₁ a₂ a₃ : ℤ) (h_not_divisible : ∀ x, 0 ≤ x → x < 1993 → ¬ (1992 ∣ f a₀ a₁ a₂ a₃ x)) : 
  ∀ m : ℤ, ¬ (f a₀ a₁ a₂ a₃ m = 0) :=
sorry

end no_integer_roots_l371_371978


namespace marbles_left_mrs_hilt_marbles_left_l371_371619

-- Define the initial number of marbles
def initial_marbles : ℕ := 38

-- Define the number of marbles lost
def marbles_lost : ℕ := 15

-- Define the number of marbles given away
def marbles_given_away : ℕ := 6

-- Define the number of marbles found
def marbles_found : ℕ := 8

-- Use these definitions to calculate the total number of marbles left
theorem marbles_left : ℕ :=
  initial_marbles - marbles_lost - marbles_given_away + marbles_found

-- Prove that total number of marbles left is 25
theorem mrs_hilt_marbles_left : marbles_left = 25 := by 
  sorry

end marbles_left_mrs_hilt_marbles_left_l371_371619


namespace sequence_product_l371_371939

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = q * a n) (h₄ : a 4 = 2) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l371_371939


namespace number_of_possible_orders_l371_371130

def number_of_finishing_orders : ℕ := 4 * 3 * 2 * 1

theorem number_of_possible_orders : number_of_finishing_orders = 24 := 
by
  have h : number_of_finishing_orders = 24 := by norm_num
  exact h

end number_of_possible_orders_l371_371130


namespace goods_train_passing_time_l371_371768

noncomputable def time_to_pass (v_w v_g : ℝ) (d_g : ℝ) : ℝ :=
  let relative_speed_kmph := v_w + v_g
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  d_g / relative_speed_mps

theorem goods_train_passing_time :
  time_to_pass 25 142.986561075114 140 ≈ 10.787 :=
by
  sorry

end goods_train_passing_time_l371_371768


namespace min_value_of_fx_in_interval_l371_371007

theorem min_value_of_fx_in_interval :
  ∀ (f : ℝ → ℝ) (f_def : ∀ x, f x = x - Real.exp x) (a b : ℝ),
  a = 0 → b = 1 →
  (∀ x, 0 <= x ∧ x <= 1 → f 1 = 1 - Real.exp 1) →
  ∃ x ∈ set.Icc (0:ℝ) 1, f x = 1 - Real.exp 1 :=
by
  -- Proof not needed
  sorry

end min_value_of_fx_in_interval_l371_371007


namespace cassidy_posters_now_l371_371788

-- Definitions for conditions
def posters_four_years_ago : ℕ := 22
def posters_lost : ℕ := 7
def posters_exchanged_net_loss : ℕ := 1
def total_posters_now := posters_four_years_ago - posters_lost - posters_exchanged_net_loss

def posters_after_summer : ℕ := 44
def posters_added_this_summer : ℕ := 6

-- Conclusion to prove
theorem cassidy_posters_now : total_posters_now = posters_after_summer - posters_added_this_summer := by
  calc
    total_posters_now 
      = 22 - 7 - 1 : by rfl
    ... = 14 : by rfl
    ... = 44 - 6 : by norm_num
    ... = 38 : by norm_num

end cassidy_posters_now_l371_371788


namespace purchase_methods_count_l371_371303

theorem purchase_methods_count 
  (books : ℕ → ℕ) 
  (costs : list ℕ := [2, 5, 11])
  (total_money : ℕ := 40)
  (min_count : ℕ := 1) :
  (∃ (x y z : ℕ), x * costs.head + y * costs.nth 1 + z * costs.nth 2 = total_money ∧ x ≥ min_count ∧ y ≥ min_count ∧ z ≥ min_count ∧ number_of_ways costs total_money = 5) := 
sorry

end purchase_methods_count_l371_371303


namespace triangle_equal_sides_l371_371483

-- Given definitions
variables {A B C D E : Type*} [metric_space A] [metric_space B] [metric_space C]
noncomputable def ray (x y : Type*) [metric_space x] [metric_space y] : set (x × y) := sorry

noncomputable def bisector (angle : Type*) [metric_space angle] : set angle := sorry

variables {AB AC BD CE : ℝ}
variables {D E : Type*}

-- Given conditions
-- D on ray AC, E on ray AB
-- BD = CE
-- Intersecting point on the bisector of angle BAC

axiom D_on_ray_AC : ray A C
axiom E_on_ray_AB : ray A B
axiom equal_segments : BD = CE
axiom intersection_on_bisector : ∀ P, P ∈ bisector (A ∠ BAC) → P ∈ (BD ∩ CE)

-- Prove that AB = AC
theorem triangle_equal_sides (A B C D E : Type*) [metric_space A] [metric_space B] [metric_space C]
  (D_on_ray_AC : ray A C) (E_on_ray_AB : ray A B)
  (equal_segments : BD = CE)
  (intersection_on_bisector : ∀ P, P ∈ bisector (A ∠ BAC) → P ∈ (BD ∩ CE)) :
  AB = AC := 
sorry

end triangle_equal_sides_l371_371483


namespace meeting_point_time_l371_371360

def racing_magic_time : ℕ := 60  -- in seconds
def charging_bull_rounds_per_hour : ℕ := 40

noncomputable def charging_bull_time : ℕ := 3600 / charging_bull_rounds_per_hour

theorem meeting_point_time :
  let lcm := Nat.lcm racing_magic_time charging_bull_time in
  lcm / 60 = 3 :=
by
  -- Calculation and proof come here
  sorry

end meeting_point_time_l371_371360


namespace contractor_absent_days_l371_371353

theorem contractor_absent_days (x y : ℕ) 
  (h1 : x + y = 30) 
  (h2 : 25 * x - 7.5 * y = 685) : 
  y = 2 :=
sorry

end contractor_absent_days_l371_371353


namespace circumference_of_circle_with_inscribed_rectangle_l371_371749

theorem circumference_of_circle_with_inscribed_rectangle
  (w h : ℝ) (h_w : w = 9) (h_h : h = 12) :
  let d := Real.sqrt (w^2 + h^2) in
  let circ := Real.pi * d in
  circ = 15 * Real.pi :=
by
  -- Definitions directly translated from the conditions:
  -- w and h are the width and height of the rectangle respectively.
  -- The diagonal ('d') of the rectangle calculated using the Pythagorean theorem.
  -- The circumference ('circ') of the circle using the diameter.
  sorry

end circumference_of_circle_with_inscribed_rectangle_l371_371749


namespace bells_toll_together_l371_371783

theorem bells_toll_together (a b c d : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 11) (h4 : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 :=
by
  rw [h1, h2, h3, h4]
  sorry

end bells_toll_together_l371_371783


namespace real_roots_product_eq_2006_l371_371341

def polynomial : Polynomial ℚ := 
  let a := (1:ℚ)
  let b := (90:ℚ)
  let c := (2027:ℚ)
  let y := x^2 + a*x + b
  let eq1 := (y / 3 = sqrt(y + 28))
  have roots_eq : List ℚ := Roots(eq1).some -- Assuming real roots exist and can be computed
  let product_result : ℚ := 2006
  product_result

theorem real_roots_product_eq_2006 (x : ℚ) : 
  let y = x^2 + 90*x + 2027
  let eq := y / 3 = sqrt(y + 28)
  (Roots(eq).some.prod : ℚ) = 2006 := 
  sorry

end real_roots_product_eq_2006_l371_371341


namespace problem_solution_l371_371852

noncomputable def f (x : ℝ) : ℝ := sorry
def a : ℕ → ℝ
| 1        := -1
| (n + 1) := 2 * a n - 1

theorem problem_solution :
  (∀ x : ℝ, f (-x) = -f x) →  -- f(x) is odd function
  (∀ x : ℝ, f (3/2 - x) = f x) →  -- f(3/2 - x) = f(x)
  f (-2) = -3 →                  -- f(-2) = -3
  f (a 5) + f (a 6) = 3 := 
by
  intros h1 h2 h3
  sorry

end problem_solution_l371_371852


namespace sum_coordinates_C_is_3_l371_371933

-- Define points as a structure 
structure Point where
  x : ℝ
  y : ℝ

-- Define vertices of the parallelogram
def A : Point := ⟨2, 3⟩
def B : Point := ⟨5, 7⟩
def D : Point := ⟨11, -1⟩

-- Define a function to calculate the midpoint of two points
def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

-- Define points and the midpoint
def M_AD : Point := midpoint A D

-- Define the coordinates of the vertex C as expressions to solve
def C_x := (2 * M_AD.x) - B.x
def C_y := (2 * M_AD.y) - B.y

-- The final statement to prove
theorem sum_coordinates_C_is_3 : C_x + C_y = 3 := sorry

end sum_coordinates_C_is_3_l371_371933


namespace probability_of_red_card_l371_371291

theorem probability_of_red_card :
  let cards := ["red", "yellow", "blue", "green"],
      all_possible_pairs := [(cards.nth 0, cards.nth 1), (cards.nth 0, cards.nth 2), (cards.nth 0, cards.nth 3), (cards.nth 1, cards.nth 2), (cards.nth 1, cards.nth 3), (cards.nth 2, cards.nth 3)] in
  let red_card_pairs := [(cards.nth 0, cards.nth 1), (cards.nth 0, cards.nth 2), (cards.nth 0, cards.nth 3)] in
  (red_card_pairs.length : ℚ) / (all_possible_pairs.length : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_red_card_l371_371291


namespace intersection_area_l371_371719

theorem intersection_area (h : ℝ) (k : ℝ) (r : ℝ) (h1 : h = 2) (k1 : k = 2) (r1 : r = 2): 
  (∃ a b : ℝ, a = -4 ∧ b = 2  ∧ ∃ θ : ℝ, 0 ≤ θ ∧ θ = 2 * (Real.pi) - 4) :=
by 
  use [-4, 2, 2 * (Real.pi) - 4],
  sorry

end intersection_area_l371_371719


namespace hyperbola_area_l371_371611

theorem hyperbola_area (F₁ F₂ P : Type*)
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hyperbola_eq : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (angle_eq : ∀ F₁ F₂ P, ∠(F₁ P F₂) = π / 2) :
  area_of_triangle F₁ F₂ P = 1 := 
sorry

end hyperbola_area_l371_371611


namespace greatest_value_of_squares_l371_371601

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 170)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
sorry

end greatest_value_of_squares_l371_371601


namespace proof1_proof2_proof3_l371_371640

namespace Proof

def is_rational (x : ℚ) : Prop := true

variables {f : ℚ → ℚ → ℚ}

axiom h1 : ∀ x y z : ℚ, f(x * y, z) = f(x, z) * f(y, z)
axiom h2 : ∀ x y z : ℚ, f(z, x * y) = f(z, x) * f(z, y)
axiom h3 : ∀ x : ℚ, f(x, 1 - x) = 1

theorem proof1 : ∀ x : ℚ, f(x, x) = 1 := 
begin
  intro x,
  sorry
end

theorem proof2 : ∀ x : ℚ, f(x, -x) = 1 := 
begin
  intro x,
  sorry
end

theorem proof3 : ∀ x y : ℚ, f(x, y) * f(y, x) = 1 := 
begin
  intros x y,
  sorry
end

end Proof

end proof1_proof2_proof3_l371_371640


namespace evaluate_x_l371_371344

theorem evaluate_x (x : ℝ) :
  (∑ _ in Finset.range 6, (8 : ℝ) ^ 8) = (2 : ℝ) ^ x → x = 25 + Real.log 3 / Real.log 2 :=
by
  sorry

end evaluate_x_l371_371344


namespace kobayashi_calculation_result_l371_371961

theorem kobayashi_calculation_result (M : ℕ) (hM : M % 37 = M) :
  ((M : ℚ) / 37).round (6 : ℤ) = 9.648649 :=
by
  sorry

end kobayashi_calculation_result_l371_371961


namespace solve_for_a_l371_371999

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l371_371999


namespace initial_quadratic_example_l371_371572

theorem initial_quadratic_example :
  ∃ (p q : ℤ), 
    (x^2 + p*x + q = 0) ∧ 
    (∀ n : ℕ, n ≤ 4 → 
      let p_n := p + n
      let q_n := q + n
      ((λ x, x^2 + p_n*x + q_n = 0) 
        has integer roots)) := sorry


end initial_quadratic_example_l371_371572


namespace ferocious_creatures_creepy_crawlers_l371_371159

-- Definitions based on conditions
def All (A B : Type) := ∀ x : A, B x
def Some (A : Type) := ∃ x : A, True

variables {Creature : Type} -- Domain of discourse
variable (Alligator : Creature → Prop)
variable (Ferocious : Creature → Prop)
variable (CreepyCrawler : Creature → Prop)

theorem ferocious_creatures_creepy_crawlers 
  (H1 : ∀ x, Alligator x → Ferocious x) -- All alligators are ferocious creatures
  (H2 : ∃ x, CreepyCrawler x ∧ Alligator x) -- Some creepy crawlers are alligators
  : ∃ x, Ferocious x ∧ CreepyCrawler x := -- Some ferocious creatures are creepy crawlers
by
  sorry

end ferocious_creatures_creepy_crawlers_l371_371159


namespace pow_div_eq_l371_371328

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l371_371328


namespace debbie_packed_large_boxes_l371_371798

theorem debbie_packed_large_boxes (L : ℕ) : 
  (L * 5) + (8 * 3) + (5 * 2) = 44 → L = 2 :=
by
  intros h
  have h_total : L * 5 + 24 + 10 = 44, from h
  sorry

end debbie_packed_large_boxes_l371_371798


namespace solve_quadratic_and_cubic_eqns_l371_371304

-- Define the conditions as predicates
def eq1 (x : ℝ) : Prop := (x - 1)^2 = 4
def eq2 (x : ℝ) : Prop := (x - 2)^3 = -125

-- State the theorem
theorem solve_quadratic_and_cubic_eqns : 
  (∃ x : ℝ, eq1 x ∧ (x = 3 ∨ x = -1)) ∧ (∃ x : ℝ, eq2 x ∧ x = -3) :=
by
  sorry

end solve_quadratic_and_cubic_eqns_l371_371304


namespace Z_divisible_by_11_l371_371975

-- Define the integer Z as per the given conditions.
def Z (a b c : ℕ) [fact (0 < a)] : ℕ :=
  100000000 * a + 10000000 * b + 1000000 * c +
  100000 * a + 10000 * b + 1000 * c +
  100 * a + 10 * b + c

-- The main theorem stating that Z is divisible by 11.
theorem Z_divisible_by_11 (a b c : ℕ) [fact (0 < a)] : 
  Z a b c % 11 = 0 :=
by
  sorry

end Z_divisible_by_11_l371_371975


namespace probability_of_winning_more_than_4000_l371_371239

theorem probability_of_winning_more_than_4000 :
  let keys := {k1, k2, k3 : ℕ},
      boxes := {b1, b2, b3 : ℕ},
      money := [4, 400, 4000]
  in
  (count (λ (f: keys → boxes), f k1 = b1 ∧ f k2 = b2 ∧ f k3 = b3 ∧ money.sum > 4000) univ) /
  (count (λ (f: keys → boxes), true) univ) = 1 / 6 :=
sorry


end probability_of_winning_more_than_4000_l371_371239


namespace x_1000_bounds_l371_371987

noncomputable def seq : ℕ → ℝ
| 0       := 5
| (n + 1) := seq n + 1 / seq n

theorem x_1000_bounds : 45 < seq 1000 ∧ seq 1000 < 45.1 :=
by
  sorry

end x_1000_bounds_l371_371987


namespace difference_of_squares_l371_371547

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
sorry

end difference_of_squares_l371_371547


namespace modified_full_house_probability_l371_371306

def total_choices : ℕ := Nat.choose 52 6

def ways_rank1 : ℕ := 13
def ways_3_cards : ℕ := Nat.choose 4 3
def ways_rank2 : ℕ := 12
def ways_2_cards : ℕ := Nat.choose 4 2
def ways_additional_card : ℕ := 11 * 4

def ways_modified_full_house : ℕ := ways_rank1 * ways_3_cards * ways_rank2 * ways_2_cards * ways_additional_card

def probability_modified_full_house : ℚ := ways_modified_full_house / total_choices

theorem modified_full_house_probability : probability_modified_full_house = 24 / 2977 := 
by sorry

end modified_full_house_probability_l371_371306


namespace equal_segments_and_concurrent_bisectors_l371_371204

variables (A B C D P Q R : Point)
variables [CyclicQuadrilateral A B C D]
variables (feetPD : PerpendicularFoot D B C P)
variables (feetQD : PerpendicularFoot D C A Q)
variables (feetRD : PerpendicularFoot D A B R)

theorem equal_segments_and_concurrent_bisectors :
  (dist P Q = dist Q R) ↔ (concurrent (angleBisector A B C) (angleBisector A D C) lineAC) :=
sorry

end equal_segments_and_concurrent_bisectors_l371_371204


namespace geometric_prog_sum_a_n_l371_371084

noncomputable def f (x : ℝ) : ℝ := sorry -- Definition of f(x) based on the problem description

-- Assume all conditions provided in the problem
axiom func_eqn : ∀ x y : ℝ, f(x) = f(y) * f(x - y)
axiom f_one : f(1) = 8 / 9

-- Statement for the first proof: f(n) forms a geometric progression
theorem geometric_prog (n : ℕ) (h : n > 0) : 
  ∃ r : ℝ, ∀ n : ℕ, f(n) = r^n := sorry

-- Statement for the second proof: Sum of the sequence a_1 + a_2 + ... + a_n
theorem sum_a_n (n : ℕ) (h : n > 0) : 
  let a := λ n, (n + 1) * f(n)
  in (range n).sum (λ k, a (k + 1)) = (80 - ((512 - 64 * n) / 9) * (8 / 9)^(n - 1)) := 
sorry

end geometric_prog_sum_a_n_l371_371084


namespace Connie_correct_number_l371_371431

theorem Connie_correct_number (x : ℤ) (h : x + 2 = 80) : x - 2 = 76 := by
  sorry

end Connie_correct_number_l371_371431


namespace max_rectangle_area_l371_371693

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l371_371693


namespace jacob_initial_fish_count_l371_371421

theorem jacob_initial_fish_count : 
  ∃ J : ℕ, 
    (∀ A : ℕ, A = 7 * J) → 
    (A' = A - 23) → 
    (J + 26 = A' + 1) → 
    J = 8 := 
by 
  sorry

end jacob_initial_fish_count_l371_371421


namespace inequality_solution_interval_S_max_value_l371_371519

theorem inequality_solution_interval :
  (∀ x : ℝ, |x^2 - 3 * x - 4| < 2 * x + 2 ↔ 2 < x ∧ x < 6) :=
sorry

theorem S_max_value :
  let a := 2
  let b := 6
  let m, n : ℝ
  in
  (m ∈ Ioo (-1) 1) →
  (n ∈ Ioo (-1) 1) →
  (m * n = 1 / 3) →
  let S := (a : ℝ) / (m^2 - 1) + (b : ℝ) / (3 * (n^2 - 1))
  in S ≤ -6 :=
sorry

end inequality_solution_interval_S_max_value_l371_371519


namespace equation_of_ellipse_equation_of_line_through_ellipse_l371_371064

variables (a b x y : ℝ)

-- Conditions for the ellipse
def ellipse (a b x y : ℝ) : Prop := (a > 0 ∧ b > 0) ∧ 
  (2 * a * (1/2) = 1 ∧ 2 = a ∧ b = sqrt 3) ∧
  ((x / 2) ^ 2 + (y / sqrt 3) ^ 2 = 1)

-- Prove that the equation of the ellipse is correct
theorem equation_of_ellipse (h : ellipse a b x y) : 
  ellipse a b x y → (a = 2 ∧ b = sqrt 3 ∧ x/2^2 + y/sqrt 3^2 = 1) :=
by {
  intros,
  exact h.right.left
}

-- Conditions for the line intersection problem
variables (M : ℝ × ℝ) (l : ℝ × ℝ → ℝ)

def line_through_point (l : ℝ × ℝ → ℝ) (M : ℝ × ℝ) : Prop := 
  l = λ p, p.2 - (1/2) * p.1 + 1 ∨ l = λ p, p.2 + (1/2) * p.1 - 1

def line_intersects_ellipse (l : ℝ × ℝ → ℝ) 
  (a b x y : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
  (ellipse a b A.1 A.2 ∧ ellipse a b B.1 B.2 ∧ 
  l A = 0 ∧ l B = 0 ∧
  ((A.1, A.2) = (B.1, B.2) + (M.1, M.2) ∧ M.1 = 2 * B.1))

-- Prove that the equation of the line is correct
theorem equation_of_line_through_ellipse (h_ellipse : ellipse a b x y) 
  (h_line : line_through_point l M) 
  (h_intersection : line_intersects_ellipse l a b x y) : 
  (line_through_point l ⟨0, 1⟩) :=
by {
  sorry
}

end equation_of_ellipse_equation_of_line_through_ellipse_l371_371064


namespace pumps_empty_pool_in_80_minutes_l371_371736

-- Define the rates of the pumps
def rate_A : ℝ := 1 / 4 -- pool per hour
def rate_B : ℝ := 1 / 2 -- pool per hour

-- Define the combined rate of both pumps working together
def combined_rate : ℝ := rate_A + rate_B

-- Define the time it takes to empty the pool together in hours
def time_in_hours : ℝ := 1 / combined_rate

-- Convert the time to minutes
def time_in_minutes : ℝ := time_in_hours * 60

theorem pumps_empty_pool_in_80_minutes :
  time_in_minutes = 80 :=
by
  -- Calculation steps are omitted in the theorem statement
  sorry

end pumps_empty_pool_in_80_minutes_l371_371736


namespace find_principal_amount_indu_gave_bindu_l371_371578

theorem find_principal_amount_indu_gave_bindu :
  ∃ (P : ℝ), 
    let A1 := P * 1.1664,
        A2 := P * 1.08,
        loss := 8.000000000000227 in
    A1 - A2 = loss ∧ P ≈ 92.59 :=
by
  sorry

end find_principal_amount_indu_gave_bindu_l371_371578


namespace range_of_m_l371_371919

variable {R : Type} [LinearOrderedField R]

def discriminant (a b c : R) : R := b^2 - 4 * a * c

theorem range_of_m (m : R) :
  (discriminant (1:R) m (m + 3) > 0) ↔ (m < -2 ∨ m > 6) :=
by
  sorry

end range_of_m_l371_371919


namespace janice_earnings_after_taxes_l371_371957

theorem janice_earnings_after_taxes :
  let base_pay := 30
  let overtime_weekday := 10
  let overtime_weekend := 15
  let days_worked := 5
  let hours := [5, 6, 7, 5, 6] -- Monday, Tuesday, Wednesday, Thursday, Sunday (Sunday is the weekend)
  let tips := 15
  let basic_pay := days_worked * base_pay
  let overtime_pay := (1 * overtime_weekday + 2 * overtime_weekday + 1 * overtime_weekend)
  let total_earnings_before_taxes := basic_pay + overtime_pay + tips
  let tax_basic := basic_pay * 0.1
  let tax_ot_tips := (overtime_pay + tips) * 0.05
  let total_taxes := tax_basic + tax_ot_tips
  let total_earnings_after_taxes := total_earnings_before_taxes - total_taxes
  in total_earnings_after_taxes = 192 :=
by sorry

end janice_earnings_after_taxes_l371_371957


namespace theta_in_second_quadrant_l371_371911

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h_cos : 2 * cos θ < 0)
  (h_sin : sin (2 * θ) < 0) :
  π / 2 < θ ∧ θ < π :=
by
  sorry

end theta_in_second_quadrant_l371_371911


namespace rational_series_sum_l371_371915

theorem rational_series_sum : ∀ (a b : ℚ), |a * b - 2| + (1 - b) ^ 2 = 0 →
  (finset.range 2022).sum (λ k, (1 : ℚ) / (↑a + k) / (↑b + k)) = 2022 / 2023 :=
by
  sorry

end rational_series_sum_l371_371915


namespace minimum_distance_traveled_l371_371964

noncomputable def least_possible_distance (a b : ℕ) : Prop :=
  let P : ℝ × ℝ := (2/7, 1/4)
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 1)
  let dist := (x y : ℝ × ℝ) -> (Real.sqrt ((y.1 - x.1) ^ 2 + (y.2 - x.2) ^ 2))
  a = 17 ∧ b = 2 

theorem minimum_distance_traveled (a b : ℕ) : least_possible_distance a b → a + b = 19 := 
  sorry

end minimum_distance_traveled_l371_371964


namespace exists_nat_a_b_l371_371980

theorem exists_nat_a_b (n : ℕ) (hn : 0 < n) : 
∃ a b : ℕ, 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n :=
by
  -- The proof steps would be filled here.
  sorry

end exists_nat_a_b_l371_371980


namespace can_cut_figure_into_equal_parts_l371_371191

-- Define the conditions of being inscribed in a grid and the congruence of parts
def inscribed_in_grid (F : set (ℝ × ℝ)) : Prop := 
  -- assuming a function that checks if the figure is inscribed in a grid
  sorry

def congruent_parts (F : set (ℝ × ℝ)) (parts : list (set (ℝ × ℝ))) : Prop :=
  -- assuming a function that checks if the parts are congruent
  sorry

-- Define the core problem statement
theorem can_cut_figure_into_equal_parts (F : set (ℝ × ℝ)) :
  inscribed_in_grid F → 
  ∃ (parts : list (set (ℝ × ℝ))), congruent_parts F parts :=
sorry

end can_cut_figure_into_equal_parts_l371_371191


namespace x_y_divisible_by_7_l371_371920

theorem x_y_divisible_by_7
  (x y a b : ℤ)
  (hx : 3 * x + 4 * y = a ^ 2)
  (hy : 4 * x + 3 * y = b ^ 2)
  (hx_pos : x > 0) (hy_pos : y > 0) :
  7 ∣ x ∧ 7 ∣ y :=
by
  sorry

end x_y_divisible_by_7_l371_371920


namespace find_natural_numbers_l371_371018

theorem find_natural_numbers 
(m : ℕ) : m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ↔ (∏ i in Finset.range m, (2 * i + 1)! = (Nat.factorial ((m * (m + 1)) / 2))) :=
by
  sorry

end find_natural_numbers_l371_371018


namespace initial_marbles_count_l371_371553

theorem initial_marbles_count (marbles_per_boy : ℕ) (number_of_boys : ℕ) 
  (total_marbles_given : ℕ) : marbles_per_boy = 2 ∧ number_of_boys = 14 ∧ total_marbles_given = marbles_per_boy * number_of_boys → total_marbles_given = 28 := 
by 
  intros h 
  cases h with h1 h2 
  cases h2 with h3 h4 
  rw [h1, h3] at h4 
  change h4 to (2 * 14) = 28
  exact h4


end initial_marbles_count_l371_371553


namespace derivative_of_f_eq_f_deriv_l371_371784

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x - (Real.sin a) ^ x

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x * Real.log (Real.cos a) - (Real.sin a) ^ x * Real.log (Real.sin a)

theorem derivative_of_f_eq_f_deriv (a : ℝ) (h : 0 < a ∧ a < Real.pi / 2) :
  (deriv (f a)) = f_deriv a :=
by
  sorry

end derivative_of_f_eq_f_deriv_l371_371784


namespace solve_equation_l371_371654

theorem solve_equation (x : ℝ) (h1 : x + 1 ≠ 0) (h2 : 2 * x - 1 ≠ 0) :
  (2 / (x + 1) = 3 / (2 * x - 1)) ↔ (x = 5) := 
sorry

end solve_equation_l371_371654


namespace earnings_total_l371_371782

-- Define the earnings for each day based on given conditions
def Monday_earnings : ℝ := 0.20 * 10 * 3
def Tuesday_earnings : ℝ := 0.25 * 12 * 4
def Wednesday_earnings : ℝ := 0.10 * 15 * 5
def Thursday_earnings : ℝ := 0.15 * 8 * 6
def Friday_earnings : ℝ := 0.30 * 20 * 2

-- Compute total earnings over the five days
def total_earnings : ℝ :=
  Monday_earnings + Tuesday_earnings + Wednesday_earnings + Thursday_earnings + Friday_earnings

-- Lean statement to prove the total earnings
theorem earnings_total :
  total_earnings = 44.70 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end earnings_total_l371_371782


namespace inequality_of_areas_l371_371243

variable {A B C D K L M N : Type}
variable [MetricSpace K] [MetricSpace L] [MetricSpace M] [MetricSpace N] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable {AKN BKL CLM DMN ABCD : ℝ}

-- Define the areas of the triangles and the quadrilateral
def S1 := area A K N
def S2 := area B K L
def S3 := area C L M
def S4 := area D M N
def S_ABCD := area A B C D

-- Ensure these points are well-defined, eg. all are part of respective quadrilateral sides
variable (K_on_AB : K ∈ line_segment A B)
variable (L_on_BC : L ∈ line_segment B C)
variable (M_on_CD : M ∈ line_segment C D)
variable (N_on_DA : N ∈ line_segment D A)

theorem inequality_of_areas
  (AKN_nonneg : 0 ≤ S1) 
  (BKL_nonneg : 0 ≤ S2) 
  (CLM_nonneg : 0 ≤ S3) 
  (DMN_nonneg : 0 ≤ S4)
  (ABCD_nonneg : 0 ≤ S_ABCD) :
  (∛S1 + ∛S2 + ∛S3 + ∛S4) ≤ 2 * ∛S_ABCD :=
by
  sorry

end inequality_of_areas_l371_371243


namespace largest_prime_factor_of_4620_l371_371725

open Nat

theorem largest_prime_factor_of_4620 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4620 ∧ (∀ q : ℕ, (Nat.Prime q ∧ q ∣ 4620) → q ≤ p) :=
begin
  use 11,
  split,
  { apply Nat.prime_of_nat_prime, exact prime_11, },
  split,
  { apply divides_prime_factors, norm_num, },
  { intros q hq,
    apply le_trans (prime_le_magnitude hq.1),
    suffices : q ∣ 4620 ∧ q ∈ { 2, 5, 3, 7, 11 }, from this.elim (λ H h, H.symm ▸ Nat.le_of_eq (set.eq_of_mem_singleton h)),
    exact ⟨hq.2,Hq⟩,
    { apply and.intro, exact Nat.prime_divisors.mem_list.mp _, exact list.mem_cons_of_mem _, exact hq } 
  },
end

end largest_prime_factor_of_4620_l371_371725


namespace intersection_A_B_l371_371492

variable {α : Type*} [LinearOrderedField α]

def A : Set α := { x | x > 2 }
def B : Set α := { x | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := by
  sorry

end intersection_A_B_l371_371492


namespace find_c_exactly_two_common_points_l371_371105

theorem find_c_exactly_two_common_points (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^3 - 3*x1 + c = 0) ∧ (x2^3 - 3*x2 + c = 0)) ↔ (c = -2 ∨ c = 2) := 
sorry

end find_c_exactly_two_common_points_l371_371105


namespace percent_of_fair_hair_employees_l371_371624

variable (E F : ℕ)
variable (women_fair_hair_percent : ℝ) (women_among_fair_hair_percent : ℝ)

theorem percent_of_fair_hair_employees 
  (h1 : women_fair_hair_percent = 0.10) 
  (h2 : women_among_fair_hair_percent = 0.40)
  (h3 : 0.10 * E = 0.40 * F) :
  (F / E) = 0.25 := by
  sorry

end percent_of_fair_hair_employees_l371_371624


namespace longest_side_length_of_quadrilateral_l371_371001

-- Define the system of inequalities
def inFeasibleRegion (x y : ℝ) : Prop :=
  (x + 2 * y ≤ 4) ∧
  (3 * x + y ≥ 3) ∧
  (x ≥ 0) ∧
  (y ≥ 0)

-- The goal is to prove that the longest side length is 5
theorem longest_side_length_of_quadrilateral :
  ∃ a b c d : (ℝ × ℝ), inFeasibleRegion a.1 a.2 ∧
                  inFeasibleRegion b.1 b.2 ∧
                  inFeasibleRegion c.1 c.2 ∧
                  inFeasibleRegion d.1 d.2 ∧
                  -- For each side, specify the length condition (Euclidean distance)
                  max (dist a b) (max (dist b c) (max (dist c d) (dist d a))) = 5 :=
by sorry

end longest_side_length_of_quadrilateral_l371_371001


namespace program_choice_count_is_15_l371_371398

/--
A student must choose a program of four courses from a menu of courses consisting of English, Algebra, Geometry, History, Art, and Latin.
This program must contain English and at least one mathematics course. Prove that the number of ways to choose such a program is 15.
-/
theorem program_choice_count_is_15 :
  let courses := ['English, 'Algebra, 'Geometry, 'History, 'Art, 'Latin] in
  let mathematics_courses := ['Algebra, 'Geometry] in
  ∃ count : ℕ, count = 15 ∧ 
    ∀ (program : list string), program.length = 4 ∧ 'English ∈ program ∧ 
    ((∃ math_course, math_course ∈ mathematics_courses ∧ math_course ∈ program) ∨ 
    (∀ math_course, math_course ∈ mathematics_courses → math_course ∈ program)) →
    (∃ valid_programs : finset (list string), valid_programs.card = 15 ∧ (program ∈ valid_programs)) :=
begin
  sorry
end

end program_choice_count_is_15_l371_371398


namespace subgraph_exists_l371_371602

open GraphTheory

variables (G : Graph) (k : ℕ)

theorem subgraph_exists:
  (0 ≠ k) → (d(G) ≥ 4 * k) → 
  ∃ H : Graph, (H ≤ G) ∧ (is_k_plus_1_connected H k) ∧ (epsilon H > epsilon G - k) := 
by
  sorry

end subgraph_exists_l371_371602


namespace largest_number_among_options_l371_371982

noncomputable def x : ℝ := 10 ^ (-1999)

def option_A : ℝ := 3 + x
def option_B : ℝ := 3 - x
def option_C : ℝ := 3 * x
def option_D : ℝ := 3 / x
def option_E : ℝ := x / 3

theorem largest_number_among_options : 
  option_D > option_A ∧
  option_D > option_B ∧
  option_D > option_C ∧
  option_D > option_E := by
  sorry

end largest_number_among_options_l371_371982


namespace max_black_squares_l371_371599

/- 
  We define the conditions first:
  1. n is a natural number divisible by 3.
  2. Each (m, m) sub-table (where m > 1) has the number of black squares not more than the white squares.
  We need to prove that the maximum number of black squares in the n x n table is 4n^2 / 9.
-/

theorem max_black_squares (n : ℕ) (h : n % 3 = 0)
  (condition : ∀ (m : ℕ), 1 < m → m ≤ n → 
  ∀ (sub_table : matrix (fin m) (fin m) bool), 
  sub_table.black_squares ≤ sub_table.white_squares) : 
  max_black_squares (n x n) ≤ (4 * n^2) / 9 :=
by
  sorry

end max_black_squares_l371_371599


namespace race_permutations_l371_371125

theorem race_permutations (r1 r2 r3 r4 : Type) [decidable_eq r1] [decidable_eq r2] [decidable_eq r3] [decidable_eq r4] :
  fintype.card (finset.univ : finset {l : list r1 | l ~ [r1, r2, r3, r4]}) = 24 :=
by
  sorry

end race_permutations_l371_371125


namespace lambda_intersection_B_empty_l371_371526

open Set

theorem lambda_intersection_B_empty (a : ℝ) :
  let Λ := { p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ (x - 2) / (y - 3) = a + 1 }
  let B := { p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ (a^2 - 1) * y + (a - 1) * x = 0 }
  Disjoint Λ B ↔ a = -1 ∨ a = -5/3 :=
begin
  sorry
end

end lambda_intersection_B_empty_l371_371526


namespace parabola_properties_l371_371605

theorem parabola_properties :
  let V1 := (0 : ℝ, 0 : ℝ)
  let F1 := (0 : ℝ, 1 / 16 : ℝ)
  let Q_vertex := (0 : ℝ, 1 / 4 : ℝ)
  let Q_focus := (0 : ℝ, 9 / 32 : ℝ)
  (V2 = Q_vertex) ∧ (F2 = Q_focus) ∧ ((dist (F1) (Q_focus)) / (dist (V1) (Q_vertex)) = 7 / 8) :=
by
  sorry

end parabola_properties_l371_371605


namespace common_divisors_count_l371_371141

theorem common_divisors_count (a b : ℤ) (ha : a = 84) (hb : b = 90) :
  let common_divisors := {d ∈ (has_divisors_divisors_of a) | d ∈ (has_divisors_divisors_of b)}
  nat.card common_divisors = 8 :=
by
  have h84 : set.has_divisors_divisors_of 84 = {-84, -42, -28, -21, -14, -12, -7, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84},
      from sorry
  have h90 : set.has_divisors_divisors_of 90 = {-90, -45, -30, -18, -15, -10, -9, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90},
      from sorry
  let common_divisors := set.inter (set.has_divisors_divisors_of 84) (set.has_divisors_divisors_of 90)
  have : common_divisors = {-6, -3, -2, -1, 1, 2, 3, 6},
      from sorry
  have : set.card common_divisors = 8,
      from sorry
  exact this

end common_divisors_count_l371_371141


namespace no_preimage_iff_lt_one_l371_371520

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem no_preimage_iff_lt_one (k : ℝ) :
  (∀ x : ℝ, f x ≠ k) ↔ k < 1 := 
by
  sorry

end no_preimage_iff_lt_one_l371_371520


namespace range_of_c_value_of_c_given_perimeter_l371_371073

variables (a b c : ℝ)

-- Question 1: Proving the range of values for c
theorem range_of_c (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) :
  1 < c ∧ c < 6 :=
sorry

-- Question 2: Finding the value of c for a given perimeter
theorem value_of_c_given_perimeter (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) (h3 : a + b + c = 18) :
  c = 5 :=
sorry

end range_of_c_value_of_c_given_perimeter_l371_371073


namespace set_y_cardinality_l371_371164

-- Defining the sets and the symmetric difference operation
variables {α : Type*} [DecidableEq α]

def symm_diff (x y : set α) : set α := (x \ y) ∪ (y \ x)

-- The main theorem statement
theorem set_y_cardinality (x y : set ℤ)
  (hx : x.card = 14) 
  (h_common : (x ∩ y).card = 6) 
  (h_symm_diff : (symm_diff x y).card = 20) :
  y.card = 12 :=
  sorry

end set_y_cardinality_l371_371164


namespace roots_and_a_of_polynomial_l371_371093

theorem roots_and_a_of_polynomial :
  ∀ (a : ℤ), 
  (∀ x : ℤ, x^4 - 16*x^3 + (81 - 2*a)*x^2 + (16*a - 142)*x + a^2 - 21*a + 68 = 0 → 
  (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 7)) ↔ a = -4 :=
sorry

end roots_and_a_of_polynomial_l371_371093


namespace max_regions_divided_by_20_lines_max_finite_regions_divided_by_20_lines_l371_371902

noncomputable def number_of_regions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

noncomputable def number_of_finite_regions (n : ℕ) : ℕ :=
  number_of_regions(n) - 2 * n

theorem max_regions_divided_by_20_lines
  (h1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 20 → 1 ≤ j ∧ j ≤ 20 → i ≠ j → lines i ∩ lines j ≠ ∅)
  (h2 : ∀ i j k : ℕ, 1 ≤ i ∧ i ≤ 20 → 1 ≤ j ∧ j ≤ 20 → 1 ≤ k ∧ k ≤ 20 → i ≠ j ∧ i ≠ k ∧ j ≠ k → 
        lines i ∩ lines j ∩ lines k = ∅) :
  number_of_regions 20 = 211 :=
sorry

theorem max_finite_regions_divided_by_20_lines
  (h1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 20 → 1 ≤ j ∧ j ≤ 20 → i ≠ j → lines i ∩ lines j ≠ ∅)
  (h2 : ∀ i j k : ℕ, 1 ≤ i ∧ i ≤ 20 → 1 ≤ j ∧ j ≤ 20 → 1 ≤ k ∧ k ≤ 20 → i ≠ j ∧ i ≠ k ∧ j ≠ k → 
        lines i ∩ lines j ∩ lines k = ∅) :
  number_of_finite_regions 20 = 171 :=
sorry

end max_regions_divided_by_20_lines_max_finite_regions_divided_by_20_lines_l371_371902


namespace hyperbola_eccentricity_l371_371613

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
variables (c e : ℝ)

-- Define the eccentricy condition for hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity :
  -- Conditions regarding the hyperbola and the distances
  (∀ x y : ℝ, hyperbola a b x y → 
    (∃ x y : ℝ, y = (2 / 3) * c ∧ x = 2 * a + (2 / 3) * c ∧
    ((2 / 3) * c)^2 + (2 * a + (2 / 3) * c)^2 = 4 * c^2 ∧
    (7 * e^2 - 6 * e - 9 = 0))) →
  -- Proving that the eccentricity e is as given
  e = (3 + Real.sqrt 6) / 7 :=
sorry

end hyperbola_eccentricity_l371_371613


namespace sine_of_negative_90_degrees_l371_371789

theorem sine_of_negative_90_degrees : Real.sin (-(Real.pi / 2)) = -1 := 
sorry

end sine_of_negative_90_degrees_l371_371789


namespace general_term_of_arithmetic_seq_sum_of_c2035_b_n_product_l371_371067

noncomputable def a_n : ℕ+ → ℕ+
| 1 := 1
| n := n

noncomputable def c (n : ℕ+) : ℕ :=
if n = 1 then 4 else 2^(n : ℕ)

theorem general_term_of_arithmetic_seq : ∀ n, a_n n = n :=
sorry

theorem sum_of_c2035 : ∑ i in (finset.range 2015).map finset.range_succ, c i = 2^2016 :=
sorry

noncomputable def b_n (n : ℕ+) : ℚ := (a_n (n + 1) : ℚ) / (a_n n : ℚ)

theorem b_n_product : ∀ n, b_n n = b_n (n + 1) * b_n (n*(n + 2)) :=
sorry

end general_term_of_arithmetic_seq_sum_of_c2035_b_n_product_l371_371067


namespace min_n_gt_T10_plus_1013_l371_371864

-- Definitions used as conditions

def seq_term (i : ℕ) : ℝ := 1 + (1 / 2)^i

def T (n : ℕ) : ℝ := n + 1 - (1 / 2)^n

def T_10 : ℝ := T 10

-- Minimum value of n such that n > T_10 + 1013
theorem min_n_gt_T10_plus_1013 : ∃ n : ℕ, ( n > T_10 + 1013 ) ∧ ( ∀ m : ℕ, (m > T_10 + 1013) → n ≤ m ) :=
by
  sorry

end min_n_gt_T10_plus_1013_l371_371864


namespace solve_inequality_l371_371454

theorem solve_inequality (x : ℝ) : 
  (2 / (x + 2) + 2 / (x + 6) ≥ 3 / 4) ↔ x ∈ Icc (-6) (-2) ∪ Icc (-2) 2 := 
sorry

end solve_inequality_l371_371454


namespace power_division_identity_l371_371313

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l371_371313


namespace race_permutations_l371_371134

-- Define the problem conditions: four participants
def participants : Nat := 4

-- Define the factorial function for permutations
def factorial : Nat → Nat
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem: The number of different possible orders in which Harry, Ron, Neville, and Hermione can finish is 24
theorem race_permutations : factorial participants = 24 :=
by
  simp [participants, factorial]
  sorry

end race_permutations_l371_371134


namespace legos_set_cost_l371_371045

-- Definitions for the conditions
def cars_sold : ℕ := 3
def price_per_car : ℕ := 5
def total_earned : ℕ := 45

-- The statement to prove
theorem legos_set_cost :
  total_earned - (cars_sold * price_per_car) = 30 := by
  sorry

end legos_set_cost_l371_371045


namespace distance_FP_l371_371229

noncomputable def focus_of_parabola : Point := ⟨1, 0⟩

noncomputable def directrix_of_parabola : Line := ⟨-1, -1, 0⟩ -- x = -1

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_focus (m : ℝ) : Line := ⟨m, -1, m⟩ -- y = mx - m

def intersects_parabola (L : Line) (x y : ℝ) : Prop := (parabola x y) ∧ (L.eval x y = 0)

axiom FM_eq_3FP (F M P : Point) : 3 • (F - P) = M - F

theorem distance_FP (P : Point) (F : Point := focus_of_parabola) (L : Line) (M : Point) (hPQ : intersects_parabola L P ∧ intersects_parabola L Q) (hFM : L.eval F = 0)
  (hM : directrix_of_parabola M) : 
  (|| F - P || = 4 / 3) ∨ (|| F - P || = 8 / 3) :=
by
  sorry

end distance_FP_l371_371229


namespace proof_problem_l371_371862

-- Definitions of propositions p and q
def p : Prop :=
  ∀ x : ℝ, x ∈ Ioc (- (Real.pi / 2)) 0 → Real.sin x > x

def q : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 1
  
-- Formal statement of the problem
theorem proof_problem : p ∧ ¬q :=
by
  -- Proof omitted
  sorry

end proof_problem_l371_371862


namespace ellipse_standard_equation_given_conditions_k_value_given_conditions_l371_371486
 
def ellipse_equation (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

def slope_of_line_through_focus (c k : ℝ) : Prop := 
  ∀ (x y : ℝ), (y = k * x + c) ∧ ((x^2 + 2 * y^2 = 2 * c^2) → k = 1)

def length_of_AB (a k : ℝ) : Prop :=
  ∃ (c : ℝ), c > 0 ∧ a > b ∧ b > 0 ∧ (k = 1) ∧ (|8 / 3| = sqrt ((0 + 4 * c / 3) ^ 2 + (c + 1 / 3 * c)^2))

def ratio_of_distances (a b k : ℝ) : Prop := 
  ∀ (c : ℝ), c > 0 ∧ b = sqrt 2 / 2 ∧ a = 2 ∧ (slope_of_line_through_focus c k) ∧ (k < 0) ∧ 
    (|AF2| / |AF1| = 5) ∧ (|BF2| / |BF1| = 1 / 2)

theorem ellipse_standard_equation_given_conditions : 
  ∀ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b ∧ b = c ∧ (slope_of_line_through_focus c 1) ∧ 
  (|8 / 3| = length_of_AB a 1)) → ellipse_equation 2 sqrt(2) := by sorry

theorem k_value_given_conditions :
  ∀ (a b k : ℝ), (a > 0 ∧ b > 0 ∧ a > b ∧ b = sqrt 2 / 2 ∧ (slope_of_line_through_focus c k) ∧ 
  (k < 0) ∧ (|AF2| / |AF1| = 5) ∧ (|BF2| / |BF1| = 1 / 2)) → k = -sqrt(14) / 6 := by sorry

end ellipse_standard_equation_given_conditions_k_value_given_conditions_l371_371486


namespace most_likely_option_after_rolling_fair_die_l371_371347

theorem most_likely_option_after_rolling_fair_die :
  (let outcomes_A := {x | x < 4 ∧ 1 ≤ x ∧ x ≤ 6}
   let outcomes_B := {x | x > 4 ∧ 1 ≤ x ∧ x ≤ 6}
   let outcomes_C := {x | x > 5 ∧ 1 ≤ x ∧ x ≤ 6}
   let outcomes_D := {x | x < 5 ∧ 1 ≤ x ∧ x ≤ 6}
   let count_A := finset.card (finset.filter outcomes_A (finset.range 7))
   let count_B := finset.card (finset.filter outcomes_B (finset.range 7))
   let count_C := finset.card (finset.filter outcomes_C (finset.range 7))
   let count_D := finset.card (finset.filter outcomes_D (finset.range 7))
  in count_D > count_A ∧ count_D > count_B ∧ count_D > count_C) :=
sorry

end most_likely_option_after_rolling_fair_die_l371_371347


namespace option_B_option_C_option_D_l371_371101

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x + π / 3) * Real.sin x

theorem option_B :
  ∀ x, π / 4 < x ∧ x < π / 2 → (f x) is_strictly_decreasing_on (Ioo (π / 4) (π / 2)) :=
sorry

theorem option_C :
  let g := λ x, f (x - 5 * π / 12) + (sqrt 3 / 2)
  evenFunction (g x) :=
sorry

theorem option_D :
  f (-π / 12) = f (π / 4) :=
sorry

end option_B_option_C_option_D_l371_371101


namespace scientific_notation_of_284000000_l371_371408

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end scientific_notation_of_284000000_l371_371408


namespace quadratic_has_one_solution_l371_371162

theorem quadratic_has_one_solution (k : ℝ) : 
  ((k + 2) * x^2 + 2 * k * x + 1 = 0) → 
  (set.count {x : ℝ | (k + 2) * x^2 + 2 * k * x + 1 = 0} = 1) →
  (k = -2 ∨ k = 2 ∨ k = -1) :=
by
  intro h_eq h_count
  sorry

end quadratic_has_one_solution_l371_371162


namespace vector_dot_product_l371_371119

noncomputable def dot_product_with_difference (a b : ℝ × ℝ) : ℝ :=
  let mag_a := real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let mag_b := 3 -- given |b| = 3
  let cos_theta := -1 / 2 -- cos(120°) = -1/2
  let dot_ab := mag_a * mag_b * cos_theta
  (a.1 ^ 2 + a.2 ^ 2) - dot_ab

theorem vector_dot_product:
  let a := (1 : ℝ, real.sqrt 3) in
  dot_product_with_difference a (0, 0) = 7 :=
by sorry

end vector_dot_product_l371_371119


namespace mean_variance_transformation_l371_371086

theorem mean_variance_transformation 
  (n : ℕ)
  (x : Fin n → ℝ)
  (mean_x : (∑ i : Fin n, x i) / n = 5)
  (var_x : (∑ i : Fin n, (x i - 5) ^ 2) / n = 2) :
  let y := λ i : Fin n, 7 * x i - 2
  in let mean_y := (∑ i : Fin n, y i) / n
  in let var_y := (∑ i : Fin n, (y i - mean_y) ^ 2) / n
  in mean_y + var_y = 131 := by
  sorry

end mean_variance_transformation_l371_371086


namespace area_of_triangle_OAB_l371_371972

noncomputable def parabola (x : ℝ) : ℝ := sqrt (3 * x)

noncomputable def focus : ℝ × ℝ := (3 / 4, 0)

noncomputable def line_through_focus (x : ℝ) : ℝ := (sqrt 3 / 3) * (x - 3 / 4)

noncomputable def points_of_intersection : set (ℝ × ℝ) :=
  { (x, y) | y = parabola x ∧ y = line_through_focus x }

theorem area_of_triangle_OAB :
  let A := classical.some (points_of_intersection.some_spec.1)
  let B := classical.some (points_of_intersection.some_spec.2)
  ∃ (area : ℝ), area = 9 / 4 :=
sorry

end area_of_triangle_OAB_l371_371972


namespace hot_peppers_percentage_correct_l371_371591

def sunday_peppers : ℕ := 7
def monday_peppers : ℕ := 12
def tuesday_peppers : ℕ := 14
def wednesday_peppers : ℕ := 12
def thursday_peppers : ℕ := 5
def friday_peppers : ℕ := 18
def saturday_peppers : ℕ := 12
def non_hot_peppers : ℕ := 64

def total_peppers : ℕ := sunday_peppers + monday_peppers + tuesday_peppers + wednesday_peppers + thursday_peppers + friday_peppers + saturday_peppers
def hot_peppers : ℕ := total_peppers - non_hot_peppers
def hot_peppers_percentage : ℕ := (hot_peppers * 100) / total_peppers

theorem hot_peppers_percentage_correct : hot_peppers_percentage = 20 := 
by 
  sorry

end hot_peppers_percentage_correct_l371_371591


namespace weight_of_new_person_l371_371358

theorem weight_of_new_person (average_increase : ℝ) (original_weight : ℝ) (num_people : ℕ)
  (h : num_people = 8) (avg_increase_correct : average_increase = 2.5) (old_weight_correct : original_weight = 45) : 
  let total_increase := num_people * average_increase in
  let new_weight_difference := total_increase in
  let W := original_weight + new_weight_difference in
  W = 65 :=
by
  sorry

end weight_of_new_person_l371_371358


namespace quadratic_two_distinct_real_roots_l371_371507

theorem quadratic_two_distinct_real_roots (k : ℝ) :
    (∃ x : ℝ, 2 * k * x^2 + (8 * k + 1) * x + 8 * k = 0 ∧ 2 * k ≠ 0) →
    k > -1/16 ∧ k ≠ 0 :=
by
  intro h
  sorry

end quadratic_two_distinct_real_roots_l371_371507


namespace original_population_divisor_l371_371244

theorem original_population_divisor (a b c : ℕ) (ha : ∃ a, ∃ b, ∃ c, a^2 + 120 = b^2 ∧ b^2 + 80 = c^2) :
  7 ∣ a :=
by
  sorry

end original_population_divisor_l371_371244


namespace petya_square_larger_than_vasya_square_l371_371634

variable (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)

def petya_square_side (a b : ℝ) : ℝ := a * b / (a + b)

def vasya_square_side (a b : ℝ) : ℝ := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petya_square_larger_than_vasya_square
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  petya_square_side a b > vasya_square_side a b :=
by sorry

end petya_square_larger_than_vasya_square_l371_371634


namespace power_inequality_l371_371470

theorem power_inequality (m : ℝ) (h : m > 0) : (0.9^1.1)^m < (1.1^0.9)^m := 
sorry

end power_inequality_l371_371470


namespace initial_girls_percentage_l371_371367

noncomputable def initial_percentage_of_girls (initial_students : ℕ) (new_students : ℕ) (new_percentage_girls : ℚ) : ℚ :=
(let G := new_percentage_girls * (initial_students + new_students) in (G / initial_students) * 100)

theorem initial_girls_percentage (initial_students : ℕ) (new_students : ℕ) (new_percentage_girls : ℚ) :
  initial_students = 20 ->
  new_students = 5 ->
  new_percentage_girls = 0.32 ->
  initial_percentage_of_girls initial_students new_students new_percentage_girls = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end initial_girls_percentage_l371_371367


namespace quadrilateral_to_rhombus_l371_371210

variable {α : Type*} [OrderedField α] [AddGroup α] [AddGroup δ] [AddAction α δ]

-- Definitions to capture the conditions in Lean
def IntersectionPointOfDiagonals (A B C D O : α) : Prop :=
  -- define the condition that O is the intersection point of diagonals 
  ∃ AC BD : α, O = (AC + BD) / 2

def PerimetersEqual (A B C D O : α) : Prop :=
  -- define the condition that the perimeters of the triangles are equal
  A + B + O = C + D + O

def isRhombus (A B C D : α) : Prop :=
  -- define what it means for ABCD to be a rhombus
  A = B ∧ B = C ∧ C = D ∧ D = A

-- The statement of the problem
theorem quadrilateral_to_rhombus (A B C D O : α) 
  (h1 : IntersectionPointOfDiagonals A B C D O) 
  (h2 : PerimetersEqual A B C D O) : isRhombus A B C D := by
  -- Define the steps to prove ABCD is a rhombus given the conditions h1 and h2
  sorry

end quadrilateral_to_rhombus_l371_371210


namespace max_altitudes_equilateral_l371_371775

noncomputable def sum_bisectors (a b c : ℝ) :=
  -- Placeholder for the sum of the angle bisectors formula
  sorry

noncomputable def sum_altitudes (a b c : ℝ) :=
  -- Placeholder for the sum of the altitudes formula
  sorry

theorem max_altitudes_equilateral (a b c : ℝ) (sum_bis : ℝ) :
  sum_bisectors a b c = sum_bis →
  ∀ (x y z : ℝ),
  sum_bisectors x y z = sum_bis → 
  sum_altitudes a b c ≤ sum_altitudes x y z := 
begin
  -- Proof omitted
  sorry
end

end max_altitudes_equilateral_l371_371775


namespace smallest_area_is_6_sqrt_7_l371_371342

noncomputable def smallest_area_of_right_triangle_with_sides (a b: ℝ) (ha: a = 6) (hb: b = 8) : ℝ :=
  min ((1 / 2) * a * sqrt (b^2 - a^2)) ((1 / 2) * a * b)

theorem smallest_area_is_6_sqrt_7 (a b A: ℝ) (ha: a = 6) (hb: b = 8) (ha_right_triangle: a^2 + b^2 = c^2 ∨ b^2 > a^2):
  A = 6*sqrt 7 :=
by
  -- Proof omitted
  sorry

end smallest_area_is_6_sqrt_7_l371_371342


namespace three_color_tiling_l371_371747

theorem three_color_tiling (n : ℕ) (hn : n % 3 = 0) (tiled : ∀ (x y : ℕ), x < 3 * n → y < 3 * n → (x + y) % 2 = 0)
  : ∃ coloring : (ℕ × ℕ) → ℕ, 
      (∀ x y, x < 3 * n → y < 3 * n → coloring (x, y) ∈ {0, 1, 2})
      ∧ (∀ x y, x < 3 * n → y < 3 * n → ∀ dx dy, (dx, dy) ∈ {(0, 2), (2, 0), (-2, 0), (0, -2)} → ((x + dx) < 3 * n ∧ (x + dx) ≥ 0 ∧ (y + dy) < 3 * n ∧ (y + dy) ≥ 0) → (coloring (x, y) ≠ coloring (x + dx, y + dy)))
      ∧ (∀ c ∈ {0, 1, 2}, ∃ k, k = (3 * n)^2 / 6 ∧ ∀ x y, x < 3 * n → y < 3 * n → coloring (x, y) = c → k = (3 * n)^2 / 6) := 
sorry

end three_color_tiling_l371_371747


namespace find_root_of_polynomial_l371_371346

theorem find_root_of_polynomial (a c x : ℝ)
  (h1 : a + c = -3)
  (h2 : 64 * a + c = 60)
  (h3 : x = 2) :
  a * x^3 - 2 * x + c = 0 :=
by
  sorry

end find_root_of_polynomial_l371_371346


namespace median_unchanged_l371_371174

def original_donations : List ℕ := [30, 50, 50, 60, 60]

def modified_donations : List ℕ := [50, 50, 50, 60, 60]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted[(sorted.length / 2)]

theorem median_unchanged :
  median original_donations = median modified_donations :=
by
  sorry

end median_unchanged_l371_371174


namespace num_distinct_arrangements_mississippi_l371_371439

theorem num_distinct_arrangements_mississippi : 
  nat.factorial 11 / (nat.factorial 4 * nat.factorial 4 * nat.factorial 2) = 34650 :=
by
  -- skipping the proof with sorry
  sorry

end num_distinct_arrangements_mississippi_l371_371439


namespace propositions_3_and_4_correct_l371_371077

noncomputable theory
open_locale classical
open set

-- Define the context of lines and planes in terms of sets
variables {L : Type*} {P : Type*} [linear_order L] -- L denotes lines, P denotes planes

-- Define non-overlapping property
def non_overlapping (x y : set L) : Prop := x ∩ y = ∅

-- Define lines and planes
variable (m n l : set L)
variable (α β : set P)

-- Define necessary properties
def subset_plane (m : set L) (α : set P) : Prop := m ⊆ α
def parallel_plane (m : set L) (α : set P) : Prop := ∀ x ∈ m, ∀ y ∈ α, x ≠ y → (x - y) ∥ α
def parallel_lines (m n : set L) : Prop := ∀ x ∈ m, ∀ y ∈ n, (x - y) ∥ α 
def perpendicular_plane (m : set L) (α : set P) : Prop := ∀ x ∈ m, ∀ y ∈ α, x ≠ y → (x - y) ⟂ α
def perpendicular_lines (m n : set L) : Prop := ∀ x ∈ m, ∀ y ∈ n, (x - y) ⟂ α 
def skew_lines (m n : set L) : Prop := ¬ ∃ p, p ∈ m ∧ p ∈ n

-- Assumptions corresponding to conditions in the problem
variables (hyp1 : non_overlapping m n)
variables (hyp2 : non_overlapping α β)

-- Statements we want to prove
theorem propositions_3_and_4_correct (H3 : parallel_lines m n ∧ perpendicular_plane m α → perpendicular_plane n α)
(H4 : (skew_lines m n ∧ parallel_plane m α ∧ parallel_plane n α ∧ perpendicular_lines l m ∧ perpendicular_lines l n) → perpendicular_plane l α) :
  true := sorry

end propositions_3_and_4_correct_l371_371077


namespace sum_of_subset_products_l371_371234

def A : Set ℝ := {1/2, 1/7, 1/11, 1/13, 1/15, 1/32}

theorem sum_of_subset_products :
  ( ∏ t in A, (1 + t) ) - 1 = 14/65 :=
by
  sorry

end sum_of_subset_products_l371_371234


namespace walter_hushpuppies_per_guest_l371_371305

variables (guests hushpuppies_per_batch time_per_batch total_time : ℕ)

def batches (total_time time_per_batch : ℕ) : ℕ :=
  total_time / time_per_batch

def total_hushpuppies (batches hushpuppies_per_batch : ℕ) : ℕ :=
  batches * hushpuppies_per_batch

def hushpuppies_per_guest (total_hushpuppies guests : ℕ) : ℕ :=
  total_hushpuppies / guests

theorem walter_hushpuppies_per_guest :
  ∀ (guests hushpuppies_per_batch time_per_batch total_time : ℕ),
    guests = 20 →
    hushpuppies_per_batch = 10 →
    time_per_batch = 8 →
    total_time = 80 →
    hushpuppies_per_guest (total_hushpuppies (batches total_time time_per_batch) hushpuppies_per_batch) guests = 5 :=
by 
  intros _ _ _ _ h_guests h_hpb h_tpb h_tt
  sorry

end walter_hushpuppies_per_guest_l371_371305


namespace inverse_value_l371_371510

theorem inverse_value (f : ℝ → ℝ) (h₁ : ∀ x, f x = 3*x^4 + 6) : 
  ∃ x, f x = 150 ∧ x = real.root 4 48 :=
by
  sorry

end inverse_value_l371_371510


namespace find_unknown_number_l371_371811

theorem find_unknown_number (x : ℝ) (h : (8 / 100) * x = 96) : x = 1200 :=
by
  sorry

end find_unknown_number_l371_371811


namespace tan_theta_l371_371604

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![k, 0], ![0, k]]

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, Real.sin θ], ![- (Real.sin θ), Real.cos θ]]

def combined_matrix (k θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (rotation_matrix θ) ⬝ (dilation_matrix k)

theorem tan_theta (k θ : ℝ) (hk : k > 0)
  (H : combined_matrix k θ = ![![(-6 : ℝ), 2], ![(-2), (-6)]]) : 
  Real.tan θ = - (1 / 3) :=
by 
  sorry

end tan_theta_l371_371604


namespace value_of_m_l371_371276

theorem value_of_m (m : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∃ (k : ℝ), (2 * m - 1) * x ^ (m ^ 2) = k * x ^ n) → m = 1 :=
by
  sorry

end value_of_m_l371_371276


namespace largest_even_three_digit_number_l371_371825

theorem largest_even_three_digit_number (x : ℕ) :
  (∃ k : ℕ, x = 30 * k + 12 ∧ x < 1000 ∧ x % 2 = 0 ∧ x % 5 = 2 ∧ gcd (30) (gcd (x) (15)) = 3) → x = 972 := 
by
  intro h
  cases h with k hk
  have H : x = 30 * k + 12 := hk.1
  have h1 := hk.2.1
  have h2 := hk.2.2.1
  have h3 := hk.2.2.2.1
  have h4 := hk.2.2.2.2
  sorry

end largest_even_three_digit_number_l371_371825


namespace sum_Pi_travel_distance_l371_371208

theorem sum_Pi_travel_distance :
  let A := (0, 1)
  let B := (0, 0)
  let C := (1, 1)
  let D := (1, 0)
  let Q (i : ℕ) : ℚ × ℚ := ((2:ℚ) / 3 ^ (i + 1), 0)
  let P (i : ℕ) : ℚ × ℚ := (2 * Q i.1 / (3 * Q i.1 + 1), 2 * Q i.1 / (3 * Q i.1 + 1))
  (Finset.range ∞).sum (λ i, dist (P i.1) (P (i + 1).1)) = 2 :=
by
  sorry

end sum_Pi_travel_distance_l371_371208


namespace find_y_l371_371286

variable {x y : ℤ}
variables (h1 : y = 2 * x - 3) (h2 : x + y = 57)

theorem find_y : y = 37 :=
by {
    sorry
}

end find_y_l371_371286


namespace part_a_part_b_l371_371595

-- Define the sequence (a_n) and the parameters A and m
variable (A : ℝ)
variable (a : ℕ → ℝ)
variable (m : ℕ)

-- Conditions
axiom a_1 : a 1 = 1
axiom a_condition : ∀ n : ℕ, 1 < (a (n + 1)) / (a n) ∧ (a (n + 1)) / (a n) ≤ A

-- Part (a): Unique non-decreasing surjective function f: ℕ → ℕ such that 1 < A ^ f n / a n ≤ A
theorem part_a : ∃! f : ℕ → ℕ, (∀ n : ℕ, 1 < A ^ (f n) / a n ∧ A ^ (f n) / a n ≤ A) ∧ (NonDecreasing f) := sorry

-- Part (b): Existence of a real number C > 1 such that a_n ≥ C^n for all n under the given conditions
theorem part_b : (∀ n : ℕ, f n ≤ m) → (∃ C : ℝ, C > 1 ∧ ∀ n : ℕ, a n ≥ C^n) := sorry

end part_a_part_b_l371_371595


namespace circumcircle_tangent_to_AB_l371_371778

theorem circumcircle_tangent_to_AB (ABC : Triangle)
  (hABC : ABC.is_acute)
  (AB_lt_AC : ABC.side_length AB < ABC.side_length AC)
  (Ω : Circle)
  (h_inscribed : ABC.inscribed_in Ω)
  (M : Point)
  (hM : M = ABC.centroid)
  (AH : Line)
  (hAH : AH = ABC.altitude_from A)
  (A' : Point)
  (hA' : Line.ray M H ∩ Ω = A') :
  ABC.circumcircle_tangent A' H B AB :=
  sorry

end circumcircle_tangent_to_AB_l371_371778


namespace first_train_length_correct_l371_371301

noncomputable def length_of_first_train : ℝ :=
  let speed_first_train := 90 * 1000 / 3600  -- converting to m/s
  let speed_second_train := 72 * 1000 / 3600 -- converting to m/s
  let relative_speed := speed_first_train + speed_second_train
  let distance_apart := 630
  let length_second_train := 200
  let time_to_meet := 13.998880089592832
  let distance_covered := relative_speed * time_to_meet
  let total_distance := distance_apart
  let length_first_train := total_distance - length_second_train
  length_first_train

theorem first_train_length_correct :
  length_of_first_train = 430 :=
by
  -- Place for the proof steps
  sorry

end first_train_length_correct_l371_371301


namespace smallest_N_existence_l371_371677

theorem smallest_N_existence :
  ∃ N : ℕ, (∀ (a b : ℕ), a ∈ (Set.range 2016.succ) ∧ b ∈ (Set.range 2016.succ) ∧ a ≠ b → a * b ≤ N) ∧
            (∀ M : ℕ, (∀ (a b : ℕ), a ∈ (Set.range 2016.succ) ∧ b ∈ (Set.range 2016.succ) ∧ a ≠ b → a * b ≤ M) → N ≤ M) ∧
            N = 1017072 :=
by
  sorry

end smallest_N_existence_l371_371677


namespace value_of_a_l371_371152

theorem value_of_a (a : ℝ) (h : 4 ∈ {a, a^2 - 3 * a}) : a = -1 := sorry

end value_of_a_l371_371152


namespace exist_nat_start_digits_irrational_concat_powers_of_two_l371_371739

-- Part (a)
theorem exist_nat_start_digits (A : ℕ) : ∃ n : ℕ, (∃ m : ℤ, 10^m * A < 2^n ∧ 2^n < 10^m * (A + 1)) :=
by sorry

-- Part (b)
theorem irrational_concat_powers_of_two : irrational (real.of_pnat (nat.iterate (λ n, n * 2) 1) / (10^10)) :=
by sorry

end exist_nat_start_digits_irrational_concat_powers_of_two_l371_371739


namespace maximum_acute_angles_convex_polygon_l371_371728

theorem maximum_acute_angles_convex_polygon (n : ℕ) (h₀ : 3 ≤ n) :
    (∀ i : fin n, ∠i < 180) ∧ (∀ i : fin n, (interior_angle i < 90) → (i < 4)) :=
sorry

end maximum_acute_angles_convex_polygon_l371_371728


namespace irreducible_fraction_l371_371255

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l371_371255


namespace value_of_a7_minus_a8_l371_371179

variable {a : ℕ → ℤ} (d a₁ : ℤ)

-- Definition that this is an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

-- Given condition
def condition (a : ℕ → ℤ) : Prop :=
  a 2 + a 6 + a 8 + a 10 = 80

-- The goal to prove
theorem value_of_a7_minus_a8 (a : ℕ → ℤ) (h_arith : is_arithmetic_seq a a₁ d)
  (h_cond : condition a) : a 7 - a 8 = 8 :=
sorry

end value_of_a7_minus_a8_l371_371179


namespace probability_same_color_l371_371369

-- Definitions based on conditions
def num_green_balls : ℕ := 7
def num_white_balls : ℕ := 7
def total_balls : ℕ := num_green_balls + num_white_balls
noncomputable def total_combinations : ℚ := (nat.choose total_balls 2 : ℕ)
noncomputable def combinations_green : ℚ := (nat.choose num_green_balls 2 : ℕ)
noncomputable def combinations_white : ℚ := (nat.choose num_white_balls 2 : ℕ)

-- The statement to prove
theorem probability_same_color : (combinations_green + combinations_white) / total_combinations = 42 / 91 :=
by
  sorry

end probability_same_color_l371_371369


namespace greatest_difference_units_digit_l371_371273

theorem greatest_difference_units_digit :
  ∃ (u1 u2 : ℕ), 
    (u1 < 10 ∧ u2 < 10) ∧ 
    (725 * 10 + u1) % 4 = 0 ∧ 
    (725 * 10 + u2) % 4 = 0 ∧ 
    (u2 - u1) = 6 :=
begin
  have h1 : 725 % 4 = 1, by sorry,
  have h2 : ∀ u, ((725 * 10 + u) % 4 = 0) ↔ u % 4 = 3, from sorry,
  use [0, 6],
  split, {split; norm_num},
  split,
  { rw h2, norm_num },
  split,
  { rw h2, norm_num },
  exact rfl,
end

end greatest_difference_units_digit_l371_371273


namespace volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l371_371965

noncomputable def volume_tetrahedron (V A B C : Point) : ℝ := sorry

def is_interior_point (M V A B C : Point) : Prop := sorry -- Definition of an interior point

def is_barycenter (M V A B C : Point) : Prop := sorry -- Definition of a barycenter

def intersects_lines_planes (M V A B C A1 B1 C1 : Point) : Prop := sorry -- Definition of intersection points

def intersects_lines_sides (V A1 B1 C1 A B C A2 B2 C2 : Point) : Prop := sorry -- Definition of intersection points with sides

theorem volume_le_one_fourth_of_original (V A B C: Point) 
  (M : Point) (A1 B1 C1 A2 B2 C2 : Point) 
  (h_interior : is_interior_point M V A B C) 
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1) 
  (h_intersects_sides : intersects_lines_sides V A1 B1 C1 A B C A2 B2 C2) :
  volume_tetrahedron V A2 B2 C2 ≤ (1/4) * volume_tetrahedron V A B C :=
sorry

theorem volume_of_sub_tetrahedron (V A B C: Point) 
  (M V1 : Point) (A1 B1 C1 : Point)
  (h_barycenter : is_barycenter M V A B C)
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1)
  (h_point_V1 : intersects_something_to_find_V1) : 
  volume_tetrahedron V1 A1 B1 C1 = (1/4) * volume_tetrahedron V A B C :=
sorry

end volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l371_371965


namespace tetrahedron_volume_at_most_one_eighth_l371_371248

theorem tetrahedron_volume_at_most_one_eighth
  {A B C D : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (a b c d : ℝ) -- lengths of the edges
  (h : ∃ ab ad ac bc bd cd : ℝ, 
    (ab = dist A B ∧ ad = dist A D ∧ ac = dist A C ∧ bc = dist B C ∧ bd = dist B D ∧ cd = dist C D) ∧
    (ab ≤ 1 ∧ ad ≤ 1 ∧ ac ≤ 1 ∧ bc ≤ 1 ∧ bd ≤ 1 ∧ cd > 1)) :
  volume_tetrahedron A B C D ≤ 1 / 8 :=
begin
  -- Proof intention: use the conditions to show the volume constraint.
  sorry -- This is where the actual proof would be constructed.
end

end tetrahedron_volume_at_most_one_eighth_l371_371248


namespace pow_div_eq_l371_371324

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l371_371324


namespace gas_cost_l371_371834

theorem gas_cost (x : ℝ) (h₁ : 5 * (x / 5 - 9) = 8 * (x / 8)) : x = 120 :=
by
  sorry

end gas_cost_l371_371834


namespace trig_expr_eq_one_l371_371464

theorem trig_expr_eq_one (α : ℝ) :
  ( (sin (8 * α) + sin (9 * α) + sin (10 * α) + sin (11 * α)) / 
    (cos (8 * α) + cos (9 * α) + cos (10 * α) + cos (11 * α)) ) *
  ( (cos (8 * α) - cos (9 * α) - cos (10 * α) + cos (11 * α)) / 
    (sin (8 * α) - sin (9 * α) - sin (10 * α) + sin (11 * α)) ) = 1 := 
by 
  sorry

end trig_expr_eq_one_l371_371464


namespace interest_difference_is_three_l371_371267

-- Defining the conditions as constants
def P : ℝ := 1200
def R : ℝ := 0.10
def T : ℝ := 1
def n : ℝ := 2

-- Calculating simple interest
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

-- Calculating compound interest
def compound_interest (P R n T : ℝ) : ℝ :=
  P * (1 + R / n)^(n * T) - P

-- Theorem statement: The difference between CI and SI should be $3
theorem interest_difference_is_three : 
  let SI := simple_interest P R T;
  let CI := compound_interest P R n T;
  CI - SI = 3 := sorry

end interest_difference_is_three_l371_371267


namespace sum_2001_terms_l371_371855

-- Definition of the sequence with the given conditions.
def seq (n : ℕ) : ℤ

axiom recurrence_relation (n : ℕ) (h : n ≥ 3) : seq n = seq (n - 1) - seq (n - 2)

axiom sum_1492 : (Finset.range 1492).sum seq = 1985
axiom sum_1985 : (Finset.range 1985).sum seq = 1492

-- The goal to prove.
theorem sum_2001_terms : (Finset.range 2001).sum seq = 986 := by
  sorry

end sum_2001_terms_l371_371855


namespace pow_div_l371_371322

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l371_371322


namespace custom_op_computation_l371_371521

-- Definition of the custom operation %%
def custom_op (x y : ℕ) : ℕ := x * y - 3 * x - y

-- Statement to prove
theorem custom_op_computation : custom_op 6 4 - custom_op 4 6 = -4 :=
by
  sorry

end custom_op_computation_l371_371521


namespace diagonal_splits_odd_vertices_l371_371935

theorem diagonal_splits_odd_vertices (n : ℕ) (H : n^2 ≤ (2 * n + 2) * (2 * n + 1) / 2) :
  ∃ (x y : ℕ), x < y ∧ x ≤ 2 * n + 1 ∧ y ≤ 2 * n + 2 ∧ (y - x) % 2 = 0 :=
sorry

end diagonal_splits_odd_vertices_l371_371935


namespace angle_F_values_l371_371188

noncomputable def sin_f_values (D E : ℝ) (h1 : 2 * Real.sin D + 3 * Real.cos E = 3)
                                      (h2 : 3 * Real.sin E + 5 * Real.cos D = 4) : Set ℝ :=
{F | F = 14.5 ∨ F = 165.5}

theorem angle_F_values (D E : ℝ) (h1 : 2 * Real.sin D + 3 * Real.cos E = 3)
                                    (h2 : 3 * Real.sin E + 5 * Real.cos D = 4) :
  sin_f_values D E h1 h2 = {14.5, 165.5} :=
by simp [sin_f_values, h1, h2]; sorry

end angle_F_values_l371_371188


namespace winning_pair_probability_l371_371717

theorem winning_pair_probability :
  let deck : Finset (Fin 8) := {0,1,2,3,4,5,6,7}
  let winning_pair (a b : Fin 8) : Prop := 
    (a < 4 ∧ b < 4) ∨ (4 ≤ a ∧ 4 ≤ b) ∨ (a % 4 = b % 4)
  (deck.card = 8) →
  ∃ (w : Finset (Fin 8 × Fin 8)), 
    (∀ p ∈ w, winning_pair p.1 p.2) ∧
    w.card = (Nat.choose 2 8) * (4 + 12) / (Nat.choose 2 8) → 
    (w.card.to_real / (Nat.choose 2 8).to_real) = 4 / 7 :=
begin
  let deck := {0,1,2,3,4,5,6,7},
  let winning_pair a b := 
    (a < 4 ∧ b < 4) ∨ (4 ≤ a ∧ 4 ≤ b) ∨ (a % 4 = b % 4), 
  assume h : deck.card = 8,
  let w := deck.product deck.filter (λ p, p.1 < p.2)?.filter (λ p, winning_pair p.1 p.2),
  use w,
  split,
  { assume p hp, 
    exact hp.2 },
  { rw [Finset.card_eq_sum_ones, Finset.sum_filter],
    sorry }
end

end winning_pair_probability_l371_371717


namespace find_f_x_l371_371517

theorem find_f_x (p q a b : ℝ) (h : p ≠ 0) (hq : q ≠ 1) (hx : ∀ x, f (x + p) - q * f x ≥ a * x + b) :
  (∀ x, q > 0 → f x = (a * x) / (1 - q) + h x * q ^ x + b / (1 - q) - (a * p) / (1 - q) ^ 2 ∧ 
               (h (x + p) ≥ h x)) ∧ 
  (∀ x, q < 0 → f x = (a * x) / (1 - q) + h x * (-q) ^ x + b / (1 - q) - (a * p) / (1 - q) ^ 2 ∧ 
               h (x + p) ≥ - h x) :=
sorry

end find_f_x_l371_371517


namespace probability_one_boy_one_girl_l371_371252

-- Define the total number of students (5), the number of boys (3), and the number of girls (2).
def total_students : Nat := 5
def boys : Nat := 3
def girls : Nat := 2

-- Define the probability calculation in Lean.
noncomputable def select_2_students_prob : ℚ :=
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose boys 1 * Nat.choose girls 1
  favorable_combinations / total_combinations

-- The statement we need to prove is that this probability is 3/5
theorem probability_one_boy_one_girl : select_2_students_prob = 3 / 5 := sorry

end probability_one_boy_one_girl_l371_371252


namespace cyclic_sum_ineq_l371_371851

theorem cyclic_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) 
  ≥ (1 / 3) * (a + b + c) :=
by
  sorry

end cyclic_sum_ineq_l371_371851


namespace max_rectangle_area_l371_371694

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l371_371694


namespace solution_set_of_inequality_l371_371476

noncomputable def f (x : ℝ) : ℝ :=
  -- Assume the existence of a function satisfying the given properties
  sorry

theorem solution_set_of_inequality :
  (∀ x₁ x₂ : ℝ, f(x₁ + x₂) = f(x₁) + f(x₂)) →
  (∀ x : ℝ, x > 0 → f(x) > 0) →
  (f(1) = 1) →
  { x : ℝ | 2^(1 + f(x)) + 2^(1 - f(x)) + 2 * f(x^2) ≤ 7 } = set.Icc (-1 : ℝ) 1 :=
sorry

end solution_set_of_inequality_l371_371476


namespace find_number_l371_371158

theorem find_number:
  ∃ x : ℝ, (3/4 * x + 9 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 :=
by
  sorry

end find_number_l371_371158


namespace number_of_digits_filled_l371_371278

noncomputable def num_digits_filled (x : ℕ) : ℕ :=
  if (10 * x + 4) < 31 then 1 else 0

theorem number_of_digits_filled :
  (∑ x in finset.range 10, num_digits_filled x) = 2 :=
by
  sorry

end number_of_digits_filled_l371_371278


namespace limit_n_b_n_l371_371799

noncomputable def M (x : ℝ) : ℝ := x - x^3 / 3

noncomputable def b_n (n : ℕ) : ℝ :=
  let rec iterate n x :=
    match n with
    | 0     => x
    | n + 1 => iterate n (M x)
  iterate n (20 / n)

theorem limit_n_b_n : 
  tendsto (λ n : ℕ, n * b_n n) at_top (𝓝 3) := 
sorry

end limit_n_b_n_l371_371799


namespace exists_2020_integers_l371_371447
open Nat

theorem exists_2020_integers : ∃ (a : Fin 2020 → ℕ), 
  (∀ i j : Fin 2020, i ≠ j → (a i ≠ a j)) ∧ -- 2020 different positive integers
  (∀ i j : Fin 2020, |a i - a j| = gcd (a i) (a j)) := -- for any two numbers a and b among them, |a - b| = gcd(a, b)
sorry

end exists_2020_integers_l371_371447


namespace min_distance_origin_to_curve_l371_371481

theorem min_distance_origin_to_curve (x y : ℝ) (h : (x - 1) * (y - 1) = 2) :
  ∃ x y, dist (0, 0) (x, y) = 2 - sqrt 2 := sorry

end min_distance_origin_to_curve_l371_371481


namespace probability_30_to_50_l371_371701

noncomputable def xi_distribution : ProbabilityDistribution ℝ := 
  NormalDist.mk 40 σ

axiom P_xi_less_than_30 (σ : ℝ) : ∫ x in Iic 30, xi_distribution.to_density (Pdf) x = 0.2

theorem probability_30_to_50 (σ : ℝ) : ∫ x in Ioc 30 50, xi_distribution.to_density (Pdf) x = 0.6 :=
by sorry

end probability_30_to_50_l371_371701


namespace binary_equation_l371_371423

def bin_to_dec (b : List ℕ) : ℕ :=
  b.foldl (λ acc x, x + acc * 2) 0

def dec_to_bin (n : ℕ) : List ℕ :=
  if n = 0 then [0] else 
    let rec loop (n : ℕ) (acc : List ℕ) :=
      if n = 0 then acc else loop (n / 2) (n % 2 :: acc)
    loop n []

theorem binary_equation :
  let bin_110110 := bin_to_dec [1, 1, 0, 1, 1, 0]
  let bin_1010 := bin_to_dec [1, 0, 1, 0]
  let bin_100 := bin_to_dec [1, 0, 0]
  let result := (bin_110110 / bin_100) * bin_1010
  dec_to_bin result = [1, 0, 0, 0, 0, 0, 1, 0] :=
by {
  sorry
}

end binary_equation_l371_371423


namespace no_red_is_complementary_to_at_most_one_l371_371554

def ExactlyNoRedBall (balls : List Bool) : Prop :=
  (balls = [true, true]) -- representing two white balls

def AtMostOneWhiteBall (balls : List Bool) : Prop :=
  (balls = [false, false]) ∨ (balls = [false, true]) ∨ (balls = [true, false])
  -- no white balls, or one of the two possible one white ball cases

def events_are_complementary (balls : List Bool) : Prop :=
  ExactlyNoRedBall balls ↔ ¬AtMostOneWhiteBall balls

theorem no_red_is_complementary_to_at_most_one (balls : List Bool) (h : length balls = 2) :
  -- Given there are 2 balls drawn (length condition)
  (balls = [false, false]) ∨ (balls = [false, true]) ∨ (balls = [false, true]) ∨ 
  (balls = [true, true]) →
  -- Given the possible combinations of 2 balls from 2 red and 2 white:
  events_are_complementary balls :=
by
  -- According to conditions and provided solution, we state the equivalency
  sorry

end no_red_is_complementary_to_at_most_one_l371_371554


namespace picture_area_l371_371906

theorem picture_area (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (3 * x + 4) * (y + 3) - x * y = 54 → x * y = 6 :=
by
  intros h
  sorry

end picture_area_l371_371906


namespace total_distance_run_l371_371580

-- Define the distances run each day based on the distance run on Monday (x)
def distance_on_monday (x : ℝ) := x
def distance_on_tuesday (x : ℝ) := 2 * x
def distance_on_wednesday (x : ℝ) := x
def distance_on_thursday (x : ℝ) := (1/2) * x
def distance_on_friday (x : ℝ) := x

-- Define the condition for the shortest distance
def shortest_distance_condition (x : ℝ) :=
  min (distance_on_monday x)
    (min (distance_on_tuesday x)
      (min (distance_on_wednesday x)
        (min (distance_on_thursday x) 
          (distance_on_friday x)))) = 5

-- State and prove the total distance run over the week
theorem total_distance_run (x : ℝ) (hx : shortest_distance_condition x) : 
  distance_on_monday x + distance_on_tuesday x + distance_on_wednesday x + distance_on_thursday x + distance_on_friday x = 55 :=
by
  sorry

end total_distance_run_l371_371580


namespace balls_in_boxes_l371_371532

theorem balls_in_boxes :
  let balls := 5
  let boxes := 2
  (boxes^balls = 32) :=
by
  let balls := 5
  let boxes := 2
  have h : boxes^balls = 32 :=
    by
      rw [←nat.pow_eq] {output := 2, input := (5 : ℕ)}
      exact dec_trivial
  exact h

end balls_in_boxes_l371_371532


namespace seating_arrangements_exactly_two_adjacent_empty_seats_l371_371712

theorem seating_arrangements_exactly_two_adjacent_empty_seats : 
  (∃ (arrangements : ℕ), arrangements = 72) :=
by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_seats_l371_371712


namespace sphere_volume_from_surface_area_l371_371878

theorem sphere_volume_from_surface_area (S : ℝ) (V : ℝ) (R : ℝ) (h1 : S = 36 * Real.pi) (h2 : S = 4 * Real.pi * R ^ 2) (h3 : V = (4 / 3) * Real.pi * R ^ 3) : V = 36 * Real.pi :=
by
  sorry

end sphere_volume_from_surface_area_l371_371878


namespace max_notebooks_l371_371198

-- Define the conditions
def available_money : ℝ := 10
def cost_per_notebook : ℝ := 1.25

-- Define the statement to be proved
theorem max_notebooks : ∃ n : ℕ, n ≤ 8 ∧ available_money < cost_per_notebook * (n + 1) :=
begin
  sorry
end

end max_notebooks_l371_371198


namespace sum_first_100_terms_l371_371849

def a (n : ℕ) : ℚ := 2 / (n * (n + 1))

def S (m : ℕ) : ℚ := ∑ k in finset.range m, 2 * (1 / (k + 1) - 1 / (k + 2))

theorem sum_first_100_terms :
  S 100 = 200 / 101 :=
by sorry

end sum_first_100_terms_l371_371849


namespace range_of_a_l371_371518

noncomputable def sequence_term (n : ℕ) (a : ℝ) : ℝ :=
  if n ≤ 5 then n + 15/n else a * Real.log n - 1/4

def min_value (a : ℝ) : ℝ := 31 / 4

theorem range_of_a (a : ℝ) :
  (∀ n, sequence_term n a ≥ min_value a) → a ≥ 8 / Real.log 6 :=
by
  intros h
  -- Proof needed here
  sorry

end range_of_a_l371_371518


namespace dima_floor_l371_371443

-- Definitions of the constants from the problem statement
def nine_story_building := 9
def elevator_descend_time := 60 -- seconds
def journey_upstairs_time := 70 -- seconds
def elevator_speed := (λ n : ℕ, (n - 1) / 60)
def dima_walk_speed := (λ n : ℕ, (n - 1) / 120)

-- Define the main problem statement
theorem dima_floor :
  ∃ n : ℕ, 
    n ≤ nine_story_building ∧
    (∃ m : ℕ, m < n ∧
    (journey_upstairs_time =
      ((m - 1) / elevator_speed n +
       (n - m) / (dima_walk_speed n))) ∧
    n = 7) :=
sorry

end dima_floor_l371_371443


namespace basketball_prices_l371_371238

variable (A B : ℕ)

theorem basketball_prices (h1 : A = 2 * B - 48) 
                          (h2 : ∃ n : ℕ, n * A = 9600 ∧ n * B = 7200) :
  A = 96 ∧ B = 72 :=
by
  obtain ⟨n, hn1, hn2⟩ := h2
  have hB : B = 72 := by
    have hB_eq : 9600 * B = 7200 * (2 * B - 48) := by
      rw [hn1, hn2]
    field_simp at hB_eq
    linarith
  have hA : A = 96 := by
    rw [hB, h1]
    norm_num
  exact ⟨hA, hB⟩

end basketball_prices_l371_371238


namespace roots_eqn_l371_371495

/-- Given that \alpha and \beta are the roots of x^2 - 3x - 4 = 0, prove that 4\alpha^3 + 9\beta^2 = -72. -/
theorem roots_eqn (α β : ℝ) (h_root_α : is_root (Polynomial.mk [ -4, -3, 1 ]) α)
  (h_root_β : is_root (Polynomial.mk [ -4, -3, 1 ]) β) :
  4 * α ^ 3 + 9 * β ^ 2 = -72 :=
by
  sorry

end roots_eqn_l371_371495


namespace P_bounds_l371_371205

-- Define a function P(n) 
def P (n : ℕ) := 
  { f : ℝ → ℝ // ∃ (a b c ∈ finset.range n), 
    (∀ x, f x = a * x^2 + b * x + c) ∧ 
    (∀ x, f x = 0 → x ∈ ℤ) }.card

-- State the theorem
theorem P_bounds (n : ℕ) (h : 4 ≤ n) : n < P n ∧ P n < n^2 := 
by
  sorry

end P_bounds_l371_371205


namespace rectangle_similarity_l371_371983

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def is_congruent (A B : Rectangle) : Prop :=
  A.length = B.length ∧ A.width = B.width

def is_similar (A B : Rectangle) : Prop :=
  A.length / A.width = B.length / B.width

theorem rectangle_similarity (A B : Rectangle)
  (h1 : ∀ P, is_congruent P A → ∃ Q, is_similar Q B)
  : ∀ P, is_congruent P B → ∃ Q, is_similar Q A :=
by sorry

end rectangle_similarity_l371_371983


namespace constant_sequence_is_AP_and_GP_l371_371038

theorem constant_sequence_is_AP_and_GP (seq : ℕ → ℕ) (h : ∀ n, seq n = 7) :
  (∃ d, ∀ n, seq n = seq (n + 1) + d) ∧ (∃ r, ∀ n, seq (n + 1) = seq n * r) :=
by
  sorry

end constant_sequence_is_AP_and_GP_l371_371038


namespace can_cover_board_cannot_cover_board_one_corner_removed_cannot_cover_board_two_corners_removed_l371_371814

-- 1. Can Felicie cover her chessboard with dominoes without exceeding the board's boundaries?
theorem can_cover_board : (∃ (dominoes : ℕ), dominoes = 32 ∧ can_cover 8 8 2 1 dominoes := sorry

-- 2. Can she cover the remaining part of the chessboard after removing one corner, still without exceeding the boundaries?
theorem cannot_cover_board_one_corner_removed : ¬ ∃ (dominoes : ℕ), dominoes * 2 = 63 ∧ can_cover 8 7 2 1 dominoes := sorry

-- 3. Can she cover the remaining part of the chessboard after removing two opposite corners without exceeding the boundaries?
theorem cannot_cover_board_two_corners_removed :
  ¬ ∃ (dominoes : ℕ), same_color_square_removed 8 8 black 2 ∧ can_cover (8-1) (8-1) 2 1 dominoes := sorry


end can_cover_board_cannot_cover_board_one_corner_removed_cannot_cover_board_two_corners_removed_l371_371814


namespace license_plate_count_l371_371138

theorem license_plate_count :
  let letters := 
    let first_letter_choices := 26 in
    let second_letter_choices := 21 in
    let third_letter_choices := 5 in
    first_letter_choices * second_letter_choices * third_letter_choices
  let digits := 
    let odd_digit_choices := 5 in
    let even_digit_choices := 5 in
    odd_digit_choices * odd_digit_choices * even_digit_choices * even_digit_choices
  letters * digits = 1706250 :=
by
  let letters := 26 * 21 * 5
  let digits := 5 * 5 * 5 * 5
  show letters * digits = 1706250
  sorry

end license_plate_count_l371_371138


namespace seven_lines_regions_l371_371648

theorem seven_lines_regions (n : ℕ) (hn : n = 7) (h1 : ¬ ∃ l1 l2 : ℝ, l1 = l2) (h2 : ∀ l1 l2 l3 : ℝ, ¬ (l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ (l1 = l2 ∧ l2 = l3))) :
  ∃ R : ℕ, R = 29 :=
by
  sorry

end seven_lines_regions_l371_371648


namespace initial_gallons_pure_water_l371_371382

variable {W : ℝ}

theorem initial_gallons_pure_water (h : 0.25 * 66.67 = 0.10 * (W + 66.67)) : W = 100 := by
  have h1: 0.25 * 66.67 = 0.10 * W + 0.10 * 66.67 := by
    rw [mul_add]
    exact h
  have h2: 16.6675 = 0.10 * W + 6.667 := by
    exact h1
  have h3: 16.6675 - 6.667 = 0.10 * W := by
    linarith
  have h4: 10 = 0.10 * W := by
    exact h3
  have h5: W = 100 := by
    field_simplify at h4
    exact h4
  exact h5

end initial_gallons_pure_water_l371_371382


namespace geometric_progression_ratio_l371_371020

theorem geometric_progression_ratio (q : ℝ) (h : |q| < 1 ∧ ∀a : ℝ, a = 4 * (a * q / (1 - q) - a * q)) :
  q = 1 / 5 :=
by
  sorry

end geometric_progression_ratio_l371_371020


namespace solution_set_of_inequality_l371_371709

theorem solution_set_of_inequality (a : ℝ) (h1 : 2 * a - 3 < 0) (h2 : 1 - a < 0) : 1 < a ∧ a < 3 / 2 :=
by
  sorry

end solution_set_of_inequality_l371_371709


namespace find_n_l371_371539

theorem find_n (n : ℚ) : 8^4 = 16^n * 2 → n = 11 / 4 := by
  intro h
  -- Further proof steps to be implemented here
  sorry

end find_n_l371_371539


namespace wooden_box_width_l371_371769

theorem wooden_box_width (W : ℝ) : 
  (∀ (length height: ℝ), 
   length = 8 ∧ height = 6 ∧ 
   1_000_000 * (0.08 * 0.07 * 0.06) = length * W * height) →
  W = 7 :=
by
  intros,
  cases h with length_eq height_eq,
  sorry

end wooden_box_width_l371_371769


namespace most_likely_units_digit_sum_is_zero_l371_371662

theorem most_likely_units_digit_sum_is_zero :
  ∃ (units_digit : ℕ), 
  (∀ m n : ℕ, (1 ≤ m ∧ m ≤ 9) ∧ (1 ≤ n ∧ n ≤ 9) → 
    units_digit = (m + n) % 10) ∧ 
  units_digit = 0 :=
sorry

end most_likely_units_digit_sum_is_zero_l371_371662


namespace at_least_one_number_greater_than_16000_l371_371716

theorem at_least_one_number_greater_than_16000 
    (numbers : Fin 20 → ℕ) 
    (h_distinct : Function.Injective numbers)
    (h_square_product : ∀ i : Fin 19, ∃ k : ℕ, numbers i * numbers (i + 1) = k^2)
    (h_first : numbers 0 = 42) :
    ∃ i : Fin 20, numbers i > 16000 :=
by
  sorry

end at_least_one_number_greater_than_16000_l371_371716


namespace minimal_storing_capacity_required_l371_371380

theorem minimal_storing_capacity_required (k : ℕ) (h1 : k > 0)
    (bins : ℕ → ℕ → ℕ → Prop)
    (h_initial : bins 0 0 0)
    (h_laundry_generated : ∀ n, bins (10 * n) (10 * n) (10 * n))
    (h_heaviest_bin_emptied : ∀ n r b g, (r + b + g = 10 * n) → max r (max b g) + 10 * n - max r (max b g) = 10 * n)
    : ∀ (capacity : ℕ), capacity = 25 :=
sorry

end minimal_storing_capacity_required_l371_371380


namespace proof_problem_l371_371490

def f (x : ℝ) := (2^x - 1) / (2^x + 1)
def g (x : ℝ) := x^(1/3)

def p : Prop := ∀ x : ℝ, f (-x) = -f x
def q : Prop := ∀ x : ℝ, x = 0 → (derivative g x).is_none

theorem proof_problem : p ∧ q :=
by
  sorry

end proof_problem_l371_371490


namespace eval_composed_function_l371_371155

noncomputable def f (x : ℝ) := 3 * x^2 - 4
noncomputable def k (x : ℝ) := 5 * x^3 + 2

theorem eval_composed_function :
  f (k 2) = 5288 := 
by
  sorry

end eval_composed_function_l371_371155


namespace pencil_length_after_sharpening_l371_371941

def initial_length : ℕ := 50
def monday_sharpen : ℕ := 2
def tuesday_sharpen : ℕ := 3
def wednesday_sharpen : ℕ := 4
def thursday_sharpen : ℕ := 5

def total_sharpened : ℕ := monday_sharpen + tuesday_sharpen + wednesday_sharpen + thursday_sharpen

def final_length : ℕ := initial_length - total_sharpened

theorem pencil_length_after_sharpening : final_length = 36 := by
  -- Here would be the proof body
  sorry

end pencil_length_after_sharpening_l371_371941


namespace find_union_of_sets_l371_371112

-- Define the sets A and B in terms of a
def A (a : ℤ) : Set ℤ := { n | n = |a + 1| ∨ n = 3 ∨ n = 5 }
def B (a : ℤ) : Set ℤ := { n | n = 2 * a + 1 ∨ n = a^2 + 2 * a ∨ n = a^2 + 2 * a - 1 }

-- Given condition: A ∩ B = {2, 3}
def condition (a : ℤ) : Prop := A a ∩ B a = {2, 3}

-- The correct answer: A ∪ B = {-5, 2, 3, 5}
theorem find_union_of_sets (a : ℤ) (h : condition a) : A a ∪ B a = {-5, 2, 3, 5} :=
sorry

end find_union_of_sets_l371_371112


namespace student_l371_371399

-- Definitions based on conditions
def first_year_courses : ℕ := 5
def first_year_average : ℕ := 70
def second_year_courses : ℕ := 6
def total_courses : ℕ := 11
def overall_average : ℚ := 86
def total_points := first_year_courses * first_year_average + second_year_courses * (overall_average * total_courses - first_year_courses * first_year_average) / second_year_courses

-- Theorem stating the problem
theorem student's_average_grade_last_year : 
  total_points / second_year_courses = 99.333333333333333333 {
  sorry
}

end student_l371_371399


namespace car_cost_is_4640_l371_371197

variable (hourly_wage : ℝ) (weekly_hours : ℝ) (overtime_rate : ℝ)
variable (overtime_threshold : ℝ) (num_weeks : ℝ)

def regular_hours := min weekly_hours overtime_threshold
def overtime_hours := max 0 (weekly_hours - overtime_threshold)
def regular_pay := hourly_wage * regular_hours
def overtime_pay := hourly_wage * overtime_rate * overtime_hours
def weekly_pay := regular_pay + overtime_pay
def total_earnings := num_weeks * weekly_pay
def car_cost := total_earnings

theorem car_cost_is_4640 (h_wage : hourly_wage = 20) (h_weekly_hours : weekly_hours = 52) 
    (h_overtime_rate : overtime_rate = 1.5) (h_overtime_threshold : overtime_threshold = 40) 
    (h_num_weeks : num_weeks = 4) : car_cost hourly_wage weekly_hours overtime_rate overtime_threshold num_weeks = 4640 := 
by 
sorry

end car_cost_is_4640_l371_371197


namespace max_rectangle_area_l371_371699

variables {a b : ℝ}

theorem max_rectangle_area (h : 2 * a + 2 * b = 60) : a * b ≤ 225 :=
by 
  -- Proof to be filled in
  sorry

end max_rectangle_area_l371_371699


namespace calculate_value_l371_371419

variable {α : Type*} [Preorder α] {f : α → α}

-- Define the properties of the function f
def is_odd_function (f : α → α) : Prop := ∀ x, f (-x) = -f x

def is_increasing_on (f : α → α) (a b : α) : Prop := ∀ x y, a ≤ x → x ≤ b → a ≤ y → y ≤ b → x ≤ y → f x ≤ f y

def has_max_and_min_value (f : α → α) (a b max_val min_val : α) : Prop :=
  (∀ x, a ≤ x → x ≤ b → f x ≤ max_val) ∧ (max_val = 2) ∧ (∀ x, a ≤ x → x ≤ b → f x ≥ min_val) ∧ (min_val = -1)

-- The theorem statement
theorem calculate_value (f : α → α) (a b : α) (max_val min_val : α)
  (h1 : is_odd_function f) (h2 : is_increasing_on f a b) (h3 : has_max_and_min_value f a b max_val min_val)
  (a_val : a = 3) (b_val : b = 6) : (2 * f (-b) + f (-a)) = -3 :=
by
  -- Proof goes here
  sorry

end calculate_value_l371_371419


namespace common_divisors_count_l371_371142

theorem common_divisors_count (a b : ℤ) (ha : a = 84) (hb : b = 90) :
  let common_divisors := {d ∈ (has_divisors_divisors_of a) | d ∈ (has_divisors_divisors_of b)}
  nat.card common_divisors = 8 :=
by
  have h84 : set.has_divisors_divisors_of 84 = {-84, -42, -28, -21, -14, -12, -7, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84},
      from sorry
  have h90 : set.has_divisors_divisors_of 90 = {-90, -45, -30, -18, -15, -10, -9, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90},
      from sorry
  let common_divisors := set.inter (set.has_divisors_divisors_of 84) (set.has_divisors_divisors_of 90)
  have : common_divisors = {-6, -3, -2, -1, 1, 2, 3, 6},
      from sorry
  have : set.card common_divisors = 8,
      from sorry
  exact this

end common_divisors_count_l371_371142


namespace fred_earned_correctly_l371_371594

-- Assuming Fred's earnings from different sources
def fred_earned_newspapers := 16 -- dollars
def fred_earned_cars := 74 -- dollars

-- Total earnings over the weekend
def fred_earnings := fred_earned_newspapers + fred_earned_cars

-- Given condition that Fred earned 90 dollars over the weekend
def fred_earnings_given := 90 -- dollars

-- The theorem stating that Fred's total earnings match the given earnings
theorem fred_earned_correctly : fred_earnings = fred_earnings_given := by
  sorry

end fred_earned_correctly_l371_371594


namespace intervals_of_monotonicity_l371_371511

-- Define the function f(x) given a real number a
def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - (3 / 2) * a * x^2 + (2 * a^2 + a - 1) * x + 3

-- Define the derivative of the function f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := x^2 - 3 * a * x + 2 * a^2 + a - 1

-- Theorem stating the intervals of monotonicity for different values of a
theorem intervals_of_monotonicity (a : ℝ) :
  (a = 2 → (∀ x, 0 ≤ f_prime a x)) ∧
  (a < 2 → (∀ x, (x < 2 * a - 1 ∨ x > a + 1 → 0 < f_prime a x) ∧ (2 * a - 1 < x ∧ x < a + 1 → f_prime a x < 0))) ∧
  (a > 2 → (∀ x, (x < a + 1 ∨ x > 2 * a - 1 → 0 < f_prime a x) ∧ (a + 1 < x ∧ x < 2 * a - 1 → f_prime a x < 0))) :=
sorry

end intervals_of_monotonicity_l371_371511


namespace altitude_of_balloon_l371_371503

variables (temp_ground temp_balloon temp_drop_rate altitude_increment : ℝ)
variables (cond1 : temp_drop_rate = 3)
variables (cond2 : temp_ground = 7)
variables (cond3 : temp_balloon = -2)
variables (cond4 : altitude_increment = 500)

theorem altitude_of_balloon :
  let temp_difference := temp_ground - temp_balloon,
      num_increments := temp_difference / temp_drop_rate,
      altitude := num_increments * altitude_increment in
  altitude = 1500 :=
by
  sorry

end altitude_of_balloon_l371_371503


namespace dima_floor_l371_371444

-- Definitions of the constants from the problem statement
def nine_story_building := 9
def elevator_descend_time := 60 -- seconds
def journey_upstairs_time := 70 -- seconds
def elevator_speed := (λ n : ℕ, (n - 1) / 60)
def dima_walk_speed := (λ n : ℕ, (n - 1) / 120)

-- Define the main problem statement
theorem dima_floor :
  ∃ n : ℕ, 
    n ≤ nine_story_building ∧
    (∃ m : ℕ, m < n ∧
    (journey_upstairs_time =
      ((m - 1) / elevator_speed n +
       (n - m) / (dima_walk_speed n))) ∧
    n = 7) :=
sorry

end dima_floor_l371_371444


namespace ellipse_semimajor_axis_value_l371_371993

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371993


namespace sum_an_bn_div_8n_l371_371206

theorem sum_an_bn_div_8n 
  (a_n b_n : ℕ → ℝ) 
  (h : ∀ n : ℕ, (3 - 2 * complex.I)^n = a_n n + b_n n * complex.I) :
  (∑' n : ℕ, (a_n n * b_n n) / 8^n) = 4 / 5 :=
by
  sorry

end sum_an_bn_div_8n_l371_371206


namespace days_for_4c_men_to_lay_6b_bricks_l371_371151

variables {b f c : ℕ} (h₁ : 2 * b * c * days_working_rate = 3 * f * c)

theorem days_for_4c_men_to_lay_6b_bricks
(p : 4 * c * x * days_working_rate = 6 * b): 
(x = ⟦ \frac{b^2}{f} ⟧) :=
sorry

end days_for_4c_men_to_lay_6b_bricks_l371_371151


namespace min_exponent_sum_is_31_l371_371875

noncomputable def min_exponent_sum (A : ℕ) (α β γ : ℕ) : ℕ :=
  if h : (A = 2^α * 3^β * 5^γ) ∧ 
         ((A / 2) = (A / 2).sqrt ^ 2) ∧ 
         ((A / 3) = (A / 3).cbrt ^ 3) ∧ 
         ((A / 5) = (A / 5) ^ (5 / 5)) then 
    α + β + γ 
  else 
    0

-- Now the theorem we want to state:
theorem min_exponent_sum_is_31 (α β γ : ℕ) (A : ℕ) :
  (A = 2^α * 3^β * 5^γ) ∧ 
  ((A / 2) = (A / 2).sqrt ^ 2) ∧ 
  ((A / 3) = (A / 3).cbrt ^ 3) ∧ 
  ((A / 5) = (A / 5) ^ (5 / 5)) → 
  min_exponent_sum A α β γ = 31 :=
sorry

end min_exponent_sum_is_31_l371_371875


namespace solve_sqrt_equation_l371_371822

theorem solve_sqrt_equation (z : ℝ) : sqrt(8 + 3 * z) = 10 → z = 92 / 3 :=
by
  intro h
  sorry

end solve_sqrt_equation_l371_371822


namespace num_zeros_after_decimal_l371_371452

theorem num_zeros_after_decimal (n : ℕ) (h : n = 15) : 
    count_initial_zeros (1 / 100^n) 30 :=
by
  sorry

end num_zeros_after_decimal_l371_371452


namespace range_a_l371_371088

noncomputable def range_of_a (θ : ℝ) (a : ℝ) : Prop :=
  let P := (3 * a - 9, a + 2)
  (3 * a - 9) * (a + 2) ≤ 0 ∧ sin (2 * θ) ≤ 0

theorem range_a (θ : ℝ) (a : ℝ) (h1 : ∃ θ, (3 * a - 9 = cos θ) ∧ (a + 2 = sin θ)) 
  (h2 : sin (2 * θ) ≤ 0) : a ∈ Icc (-2 : ℝ) 3 := 
sorry

end range_a_l371_371088


namespace find_f_l371_371516

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x / (a * x + b)

theorem find_f (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f 2 a b = 1) (h₂ : ∃! x, f x a b = x) :
  f x (1/2) 1 = 2 * x / (x + 2) :=
by
  sorry

end find_f_l371_371516


namespace problem_1_problem_2_l371_371847

noncomputable def conditions (a b c : ℝ) (A B C : ℝ) :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ a * Real.cos C + √3 * a * Real.sin C - b - c = 0

theorem problem_1 (a b c A B C : ℝ) (h : conditions a b c A B C):
  A = 60 := 
sorry

theorem problem_2 (a b c A B C : ℝ) (h : conditions a b c A B C) (ha : a = 7):
  14 < b + c ∧ b + c ≤ 21 :=
sorry

end problem_1_problem_2_l371_371847


namespace prove_irrationality_of_sqrt_3_l371_371777

noncomputable def irrational_number : Prop :=
  ∃ (x : ℝ), (x = |√3|) ∧ irrational x

theorem prove_irrationality_of_sqrt_3 : irrational_number :=
by 
  sorry

end prove_irrationality_of_sqrt_3_l371_371777


namespace quadratic_inequality_solution_l371_371543

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end quadratic_inequality_solution_l371_371543


namespace race_permutations_l371_371132

-- Define the problem conditions: four participants
def participants : Nat := 4

-- Define the factorial function for permutations
def factorial : Nat → Nat
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem: The number of different possible orders in which Harry, Ron, Neville, and Hermione can finish is 24
theorem race_permutations : factorial participants = 24 :=
by
  simp [participants, factorial]
  sorry

end race_permutations_l371_371132


namespace set_complement_intersection_l371_371974

open Set

variable (U A B : Set ℕ)

theorem set_complement_intersection :
  U = {2, 3, 5, 7, 8} →
  A = {2, 8} →
  B = {3, 5, 8} →
  (U \ A) ∩ B = {3, 5} :=
by
  intros
  sorry

end set_complement_intersection_l371_371974


namespace measure_of_angle_A_l371_371550

theorem measure_of_angle_A 
  (a b c : ℝ)
  (h : ∃ m n : ℝ × ℝ, m = (b, c - a) ∧ n = (b - c, c + a) ∧ (m.fst * n.fst + m.snd * n.snd) = 0) :
  ∠A = 2 * π / 3 := 
sorry

end measure_of_angle_A_l371_371550


namespace find_number_l371_371819

theorem find_number :
  ∃ (x : ℝ), 0.6667 * x - 10 = 0.25 * x ∧ x ≈ 24 :=
by
  have h : ∃ (x : ℝ), 0.6667 * x - 10 = 0.25 * x := sorry
  cases h with x hx
  use x
  split
  · exact hx
  · linarith

end find_number_l371_371819


namespace school_band_fundraising_l371_371706

-- Definitions
def goal : Nat := 150
def earned_from_three_families : Nat := 10 * 3
def earned_from_fifteen_families : Nat := 5 * 15
def total_earned : Nat := earned_from_three_families + earned_from_fifteen_families
def needed_more : Nat := goal - total_earned

-- Theorem stating the problem in Lean 4
theorem school_band_fundraising : needed_more = 45 := by
  sorry

end school_band_fundraising_l371_371706


namespace find_a_b_inverse_M_l371_371057

section LinearTransformation

variables {R : Type*} [CommRing R]

-- Define the matrix M
def M (a b : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![1, a], ![b, 1]]

-- Define the curve C
def C (x y : R) := x^2 + 4*x*y + 2*y^2 = 1

-- Define the curve C'
def C' (x y : R) := x^2 - 2*y^2 = 1

-- Define the transformation condition
def transformed_curve (a b x y : R) := 
  let x' := x + a * y,
      y' := b * x + y
  in C' x' y'

-- Prove that under the matrix M, the curve C becomes C'
theorem find_a_b (a b : R) : M a b = ![![1, 2], ![0, 1]] :=
by
  -- The equations to match the transformed curve to C'
  have h1 : 1 - 2 * b^2 = 1 := by sorry,
  have h2 : 2 * a - 4 * b = 0 := by sorry,
  have h3 : a^2 - 2 = -2 := by sorry,
  -- Solving these gives us a = 2 and b = 0
  sorry

-- Prove the inverse of M
theorem inverse_M : (M 2 0)⁻¹ = ![![1, -2], ![0, 1]] :=
by
  -- Proving the inverse by calculating the determinant and inverse manually
  sorry

end LinearTransformation

end find_a_b_inverse_M_l371_371057


namespace find_temp_tuesday_l371_371290

def temperatures := List ℕ

variables (T : temperatures)
-- Define the temperatures for each day of the week.
def temp_sunday    : ℕ := 40
def temp_monday    : ℕ := 50
def temp_wednesday : ℕ := 36
def temp_thursday  : ℕ := 82
def temp_friday    : ℕ := 72
def temp_saturday  : ℕ := 26

-- Define the conditions.
variables (avg_temp total_days : ℕ)
def average_temperature (temps : temperatures) : ℕ :=
  (List.sum temps) / total_days

-- The conditions for the average temperature and total days
axiom avg_temp_53 : avg_temp = 53
axiom total_days_7 : total_days = 7

theorem find_temp_tuesday :
  T = [temp_sunday, temp_monday, temp_wednesday, temp_thursday, temp_friday, temp_saturday] -> 
  average_temperature (T ++ [temp_tuesday]) avg_temp total_days = 53 -> 
  temp_tuesday = 65 :=
sorry

end find_temp_tuesday_l371_371290


namespace max_projectile_height_l371_371762

theorem max_projectile_height :
  ∃ t_max: ℝ, t_max = 2.5 ∧ (∀ t: ℝ, -20 * t^2 + 100 * t + 30 ≤ -20 * t_max^2 + 100 * t_max + 30) :=
begin
  use 2.5,
  split,
  { refl },
  { intro t,
    have ht : -20 * t_max^2 + 100 * t_max + 30 = 155, from sorry, -- This follows from the specific computation
    have h_eq : -20 * t^2 + 100 * t + 30 = -20 * (t - 2.5)^2 + 155, from sorry, -- Completing the square as per the solution
    rw [h_eq],
    linarith [(t - 2.5)^2, ht],
  }
end

end max_projectile_height_l371_371762


namespace common_divisors_84_90_l371_371143

def divisors (n : ℕ) : Set ℤ :=
  { d : ℤ | d ∣ n }

def common_divisor_count (a b : ℕ) : ℕ :=
  (divisors a ∩ divisors b).toFinset.card

theorem common_divisors_84_90 : common_divisor_count 84 90 = 8 := by
  sorry

end common_divisors_84_90_l371_371143


namespace exists_infinite_subset_coprime_sum_product_l371_371803

open Nat

def isCoprime (a b : ℕ) : Prop := gcd a b = 1

theorem exists_infinite_subset_coprime_sum_product (n : ℕ) (h : n > 1) :
  ∃ (A : set ℕ), infinite A ∧ (∀ a b c d e f g h i j : ℕ,
    a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ e ∈ A ∧
    f ∈ A ∧ g ∈ A ∧ h ∈ A ∧ i ∈ A ∧ j ∈ A ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧
    f ≠ g ∧ g ≠ h ∧ h ≠ i ∧ i ≠ j → 
    isCoprime (a + b + c + d + e + f + g + h + i + j)
              (a * b * c * d * e * f * g * h * i * j)) := 
sorry

end exists_infinite_subset_coprime_sum_product_l371_371803


namespace increase_in_expenses_is_20_percent_l371_371386

noncomputable def man's_salary : ℝ := 6500
noncomputable def initial_savings : ℝ := 0.20 * man's_salary
noncomputable def new_savings : ℝ := 260
noncomputable def reduction_in_savings : ℝ := initial_savings - new_savings
noncomputable def initial_expenses : ℝ := 0.80 * man's_salary
noncomputable def increase_in_expenses_percentage : ℝ := (reduction_in_savings / initial_expenses) * 100

theorem increase_in_expenses_is_20_percent :
  increase_in_expenses_percentage = 20 := by
  sorry

end increase_in_expenses_is_20_percent_l371_371386


namespace least_positive_period_intervals_monotonically_decreasing_max_min_values_interval_l371_371885

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * Real.cos (2 * x - Real.pi / 4)

theorem least_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi :=
sorry

theorem intervals_monotonically_decreasing :
  ∀ k : ℤ, ∀ x : ℝ,
    k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 8 →
    ∃ x1 x2 : ℝ, 
      (x1 < x2) ∧ (f x1 ≥ f x) ∧ (f x2 ≤ f x) :=
sorry

theorem max_min_values_interval :
  ∀ x : ℝ,
    (-Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 2) →
     ((f x = sqrt 2 ∧ x = Real.pi / 8) ∨ (f x = -1 ∧ x = Real.pi / 2)) :=
sorry

end least_positive_period_intervals_monotonically_decreasing_max_min_values_interval_l371_371885


namespace negate_exists_l371_371277

theorem negate_exists : 
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end negate_exists_l371_371277


namespace number_of_possible_orders_l371_371131

def number_of_finishing_orders : ℕ := 4 * 3 * 2 * 1

theorem number_of_possible_orders : number_of_finishing_orders = 24 := 
by
  have h : number_of_finishing_orders = 24 := by norm_num
  exact h

end number_of_possible_orders_l371_371131


namespace hyperbola_standard_eq_l371_371080
open real

def hyperbola_eqn (λ :ℝ) : Prop := 
  ∀ x y : ℝ, y^2 - (1/4) * x^2 = λ

theorem hyperbola_standard_eq :
  (∃ λ : ℝ, hyperbola_eqn λ (4) (sqrt 3) ∧
            λ = -1) →
  (∃ a b : ℝ, b = 2 ∧ y^2 - (x^2 / a) = 1 :=
by
  sorry

end hyperbola_standard_eq_l371_371080


namespace ellipse_semimajor_axis_value_l371_371992

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371992


namespace inclination_angle_of_tangent_l371_371544

-- Given conditions
def line (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

def circle (ϕ : ℝ) : ℝ × ℝ :=
  (4 + 2 * Real.cos ϕ, 2 * Real.sin ϕ)

-- Proof statement
theorem inclination_angle_of_tangent
  (α : ℝ)
  (H : ∃ t ϕ : ℝ, line t α = circle ϕ) :
  α = π / 6 ∨ α = 5 * π / 6 :=
sorry

end inclination_angle_of_tangent_l371_371544


namespace coefficient_of_term_x7_in_expansion_l371_371569

theorem coefficient_of_term_x7_in_expansion:
  let general_term (r : ℕ) := (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r * (x : ℤ)^(12 - (5 * r) / 2)
  ∃ r : ℕ, 12 - (5 * r) / 2 = 7 ∧ (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r = 240 := 
sorry

end coefficient_of_term_x7_in_expansion_l371_371569


namespace unique_adorable_5_digit_integer_l371_371748

def is_adorable (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5] in
  -- check if n contains exactly these digits
  Multiset.ofList (Nat.digits 10 n) = Multiset.ofList digits ∧
  -- check divisibility for each prefix length k
  (∀ k in digits.indices, ((Nat.digits 10 n).take (k + 1)).foldl (λ a x, a * 10 + x) 0 % (k + 1) = 0) ∧
  -- check sum of digits is divisible by 5
  digits.sum % 5 = 0

theorem unique_adorable_5_digit_integer : ∃! n : ℕ, is_adorable n ∧ 10000 ≤ n ∧ n < 100000 :=
by
  sorry

end unique_adorable_5_digit_integer_l371_371748


namespace sequence_periodic_if_zero_l371_371060

theorem sequence_periodic_if_zero {a : ℝ} (x : ℕ → ℝ) 
   (h₀ : x 0 = 1) 
   (h₁ : x 1 = a) 
   (h₂ : x 2 = a) 
   (h_rec : ∀ n ≥ 2, x (n + 1) = 2 * x n * x (n - 1) - x (n - 2))
   (n : ℕ) (h_n : x n = 0) :
   ∃ p > 0, ∀ k, x (k + p) = x k :=
begin
  sorry
end

end sequence_periodic_if_zero_l371_371060


namespace proof_problem_l371_371866

axiom arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n+1) - a n {:=:} a (m+1) - a m

noncomputable def a_1 : ℝ := sorry
noncomputable def d : ℝ := sorry

lemma arithmetic_sequence_properties :
  ∀ n : ℕ, a n > 0 → d ≠ 0 → Σ (a : ℕ → ℝ) (n : ℕ), arithmetic_sequence a → 
  a 7 = a 0 + 7 * d ∧ a 3 = a 0 + 3 * d ∧ a 4 = a 0 + 4 * d := sorry

theorem proof_problem (a : ℕ → ℝ) (h_seq : arithmetic_sequence a) (h_pos : ∀ i, a i > 0) (h_d_ne_zero : d ≠ 0) :
  a 0 * a 7 < a 3 * a 4 := 
by 
  have h_eq_a8 : a 7 = a 0 + 7 * d := by sorry
  have h_eq_a4 : a 3 = a 0 + 3 * d := by sorry
  have h_eq_a5 : a 4 = a 0 + 4 * d := by sorry
  have h_prod_a1_a8 : a 0 * (a 0 + 7 * d) = a 0 ^ 2 + 7 * a 0 * d := by sorry
  have h_prod_a4_a5 : (a 0 + 3 * d) * (a 0 + 4 * d) = a 0 ^ 2 + 7 * a 0 * d + 12 * d ^ 2 := by sorry
  have h_12_d2 : 12 * d ^ 2 > 0 := by sorry
  exact lt_of_add_lt_add_right h_12_d2

end proof_problem_l371_371866


namespace car_meets_train_in_24_minutes_l371_371766

-- Definitions based on conditions
def speed_of_train := 5 / 3 -- meters per minute
def speed_of_car := 5 / 2 -- meters per minute
def initial_distance := 20 -- meters

-- The proposition to be proved
theorem car_meets_train_in_24_minutes : 
  let relative_speed := speed_of_car - speed_of_train in
  let time := initial_distance / relative_speed in
  time = 24 :=
by
  let speed_of_train := 5 / 3
  let speed_of_car := 5 / 2
  let initial_distance := 20
  let relative_speed := speed_of_car - speed_of_train
  let time := initial_distance / relative_speed
  show time = 24
  sorry

end car_meets_train_in_24_minutes_l371_371766


namespace sum_bi_l371_371528

noncomputable def b3 : ℕ := 1
noncomputable def b4 : ℕ := 3
noncomputable def b5 : ℕ := 3
noncomputable def b6 : ℕ := 5
noncomputable def b7 : ℕ := 6
noncomputable def b8 : ℕ := 7

theorem sum_bi :
  0 ≤ b3 ∧ b3 < 3 ∧
  0 ≤ b4 ∧ b4 < 4 ∧
  0 ≤ b5 ∧ b5 < 5 ∧
  0 ≤ b6 ∧ b6 < 6 ∧
  0 ≤ b7 ∧ b7 < 7 ∧
  0 ≤ b8 ∧ b8 < 8 ∧
  (11 / 9) = (b3 / 3!) + (b4 / 4!) + (b5 / 5!) + (b6 / 6!) + (b7 / 7!) + (b8 / 8!) →
  b3 + b4 + b5 + b6 + b7 + b8 = 25 := by
  sorry

end sum_bi_l371_371528


namespace evaluate_neg64_to_7_over_3_l371_371013

theorem evaluate_neg64_to_7_over_3 (a : ℝ) (b : ℝ) (c : ℝ) 
  (h1 : a = -64) (h2 : b = (-4)) (h3 : c = (7/3)) :
  a ^ c = -65536 := 
by
  have h4 : (-64 : ℝ) = (-4) ^ 3 := by sorry
  have h5 : a = b^3 := by rw [h1, h2, h4]
  have h6 : a ^ c = (b^3) ^ (7/3) := by rw [←h5, h3]
  have h7 : (b^3)^c = b^(3*(7/3)) := by sorry
  have h8 : b^(3*(7/3)) = b^7 := by norm_num
  have h9 : b^7 = -65536 := by sorry
  rw [h6, h7, h8, h9]
  exact h9

end evaluate_neg64_to_7_over_3_l371_371013


namespace roots_of_polynomial_l371_371461

noncomputable def polynomial := Polynomial.C 1 * Polynomial.X^4 - Polynomial.C 3 * Polynomial.X^3 + Polynomial.C 3 * Polynomial.X^2 - Polynomial.C 1 * Polynomial.X - Polynomial.C 6

theorem roots_of_polynomial : (Polynomial.eval (2: ℝ) polynomial = 0 ∧ Polynomial.eval (2: ℝ) polynomial.derivative = 0 ∧ Polynomial.eval (2: ℝ) polynomial.derivative.derivative = 0) 
∧ Polynomial.eval (1: ℝ) polynomial = 0 :=
sorry

end roots_of_polynomial_l371_371461


namespace final_marbles_l371_371414

-- Define the initial count of marbles and the events
def initial_marbles : ℕ := 57
def first_game_lost : ℕ := 18
def second_game_won : ℕ := 25
def third_game_lost : ℕ := 12
def fourth_game_won : ℕ := 15
def marbles_given_away : ℕ := 10
def marbles_received : ℕ := 8

-- Define a theorem to prove that Alvin ends up with 65 marbles
theorem final_marbles (initial_marbles : ℕ) (first_game_lost : ℕ) (second_game_won : ℕ) (third_game_lost : ℕ) (fourth_game_won : ℕ) (marbles_given_away : ℕ) (marbles_received : ℕ) :
  initial_marbles - first_game_lost + second_game_won - third_game_lost + fourth_game_won - marbles_given_away + marbles_received = 65 :=
  by {
  let final_marbles := initial_marbles - first_game_lost + second_game_won - third_game_lost + fourth_game_won - marbles_given_away + marbles_received,
  have h1 : final_marbles = 57 - 18 + 25 - 12 + 15 - 10 + 8, by sorry,
  have h2 : 57 - 18 + 25 - 12 + 15 - 10 + 8 = 65, by sorry,
  rw h2 at h1,
  exact h1,
  }

end final_marbles_l371_371414


namespace distinctArrangements_COOKIE_l371_371904

/-- 
  Prove that the number of distinct arrangements of the letters in the word "COOKIE" 
  is 360. The word "COOKIE" consists of the letters C, O, O, K, I, E, with 'O' repeating twice.
-/
theorem distinctArrangements_COOKIE : 
  let n := 6
  let o_count := 2
  n! / (o_count! * 1! * 1! * 1! * 1!) = 360 :=
by {
  sorry
}

end distinctArrangements_COOKIE_l371_371904


namespace solve_natural_a_l371_371040

theorem solve_natural_a (a : ℕ) : 
  (∃ n : ℕ, a^2 + a + 1589 = n^2) ↔ (a = 43 ∨ a = 28 ∨ a = 316 ∨ a = 1588) :=
sorry

end solve_natural_a_l371_371040


namespace repeated_projections_square_l371_371242

theorem repeated_projections_square :
  ∀ (A B C D M1 : Point),
    is_square A B C D →
    M1 ∈ line A B →
    let M2 := projection M1 (line B C) D in
    let M3 := projection M2 (line C D) A in
    let M4 := projection M3 (line D A) B in
    let M5 := projection M4 (line A B) C in
    let M6 := projection M5 (line B C) D in
    let M7 := projection M6 (line C D) A in
    let M8 := projection M7 (line D A) B in
    let M9 := projection M8 (line A B) C in
    let M10 := projection M9 (line B C) D in
    let M11 := projection M10 (line C D) A in
    let M12 := projection M11 (line D A) B in
    let M13 := projection M12 (line A B) C in
    M13 = M1 :=
by
  intros A B C D M1 is_square_ABCDE M1_on_AB
  sorry

end repeated_projections_square_l371_371242


namespace correct_division_l371_371737

theorem correct_division (x : ℝ) (h : 8 * x + 8 = 56) : x / 8 = 0.75 :=
by
  sorry

end correct_division_l371_371737


namespace find_a_l371_371667

theorem find_a (a b c : ℝ) (h1 : ∀ x, x = 2 → y = 5) (h2 : ∀ x, x = 3 → y = 7) :
  a = 2 :=
sorry

end find_a_l371_371667


namespace min_value_sum_square_reciprocal_l371_371877

theorem min_value_sum_square_reciprocal {n : ℕ} (hn : n > 0) (x : Fin n → ℝ) 
  (hx_nonzero : ∀ i, x i ≠ 0) (hx_sum : ∑ i, x i = 0) :
  let S := (∑ i, (x i)^2) * (∑ i, 1 / (x i)^2) in
  S ≥ (if even n then ↑n ^ 2 else ↑n ^ 2 * (↑n ^ 2 + 3) / (↑n ^ 2 - 1)) := 
by sorry

end min_value_sum_square_reciprocal_l371_371877


namespace twelve_percent_greater_than_80_l371_371356

theorem twelve_percent_greater_than_80 (x : ℝ) (h : x = 80 + 0.12 * 80) : x = 89.6 :=
by
  sorry

end twelve_percent_greater_than_80_l371_371356


namespace number_of_possible_measures_A_l371_371671

theorem number_of_possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧
  (A = 180 - B) ∧
  (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧
  (∃ A_list : List ℕ, A_list.length = 17 ∧ ∀ a ∈ A_list, a = 180 - ((A_list.toList.indexOf? a).getOrElse 0) ) :=
sorry

end number_of_possible_measures_A_l371_371671


namespace angle_AMC_120_l371_371622

-- Define the equilateral triangle and the points
variables {A B C L K M : Type} [ordered_comm_group A] [ordered_comm_group B]
[ordered_comm_group C] [ordered_comm_group L] [ordered_comm_group K] [ordered_comm_group M]

-- Define the condition of equilateral triangle and given points
variables (equilateral_ABC : IsEquilateral A B C)
(on_AB : L ∈ line_segment A B)
(on_BC : K ∈ line_segment B C)
(intersection_AK_CL : M = intersection (line A K) (line C L))

-- Define the area conditions
variables (equal_area_condition : area (triangle A M C) = area (quadrilateral L B K M))

-- Prove that the angle AMC is 120 degrees
theorem angle_AMC_120 :
  ∠ A M C = 120 :=
by
  sorry

end angle_AMC_120_l371_371622


namespace num_primes_with_digit_seven_as_last_under_100_l371_371531

/-- 
Theorem: The number of primes less than 100 that have 7 as the ones digit is 6.
-/
theorem num_primes_with_digit_seven_as_last_under_100 : 
  (finset.univ.filter (λ n : ℕ, n < 100 ∧ n % 10 = 7 ∧ nat.prime n)).card = 6 :=
  sorry

end num_primes_with_digit_seven_as_last_under_100_l371_371531


namespace correct_answer_is_C_l371_371734

def question_1 : Prop := true
def question_2 : Prop := false
def question_3 : Prop := false
def question_4 : Prop := true

theorem correct_answer_is_C : (question_1 ∧ question_4) ∧ ¬(question_2 ∨ question_3) := by
  split
  -- Prove ① and ④ are correct
  { split
    { exact trivial }
    { exact trivial } }
  -- Prove ② and ③ are not correct
  { split
    { intro h 
      contradiction }
    { intro h 
      contradiction } }

end correct_answer_is_C_l371_371734


namespace each_regular_tire_distance_used_l371_371752

-- Define the conditions of the problem
def total_distance_traveled : ℕ := 50000
def spare_tire_distance : ℕ := 2000
def regular_tires_count : ℕ := 4

-- Using these conditions, we will state the problem as a theorem
theorem each_regular_tire_distance_used : 
  (total_distance_traveled - spare_tire_distance) / regular_tires_count = 12000 :=
by
  sorry

end each_regular_tire_distance_used_l371_371752


namespace largest_two_digit_prime_factor_of_binomial_l371_371722

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), 10 ≤ p ∧ p < 100 ∧ p.prime ∧ p ∣ nat.binom 150 75 ∧ ∀ q, 10 ≤ q ∧ q < 100 ∧ q.prime ∧ q ∣ nat.binom 150 75 → q ≤ p :=
begin
  use 47,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { sorry },
  { sorry }
end

end largest_two_digit_prime_factor_of_binomial_l371_371722


namespace points_description_l371_371004

noncomputable def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_description (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x + y = 0) := 
by 
  sorry

end points_description_l371_371004


namespace geometric_sequence_sum_first_n_terms_l371_371574

def a : ℕ → ℝ
| 0     := 1 / 2
| (n+1) := (↑(n+1) / (2 * (↑n + 1))) * a n

theorem geometric_sequence (n : ℕ) :
  ∀ k : ℕ, (k > 0) → (a k / ↑k) = (1 / 2) * (a 1 / 1) := sorry

theorem sum_first_n_terms (n : ℕ) :
  (∑ i in Finset.range n, a i) = 2 - (n + 2) / 2^n := sorry

end geometric_sequence_sum_first_n_terms_l371_371574


namespace collinear_vectors_l371_371420

variable (a b : ℝ × ℝ × ℝ)
variable (c1 c2 : ℝ × ℝ × ℝ)

noncomputable def vector1 : ℝ × ℝ × ℝ := (1, -2, 5)
noncomputable def vector2 : ℝ × ℝ × ℝ := (3, -1, 0)

noncomputable def c1_def := (4 * fst vector1 - 2 * fst vector2, 
                             4 * snd vector1 - 2 * snd vector2, 
                             4 * trd vector1 - 2 * trd vector2)

noncomputable def c2_def := (fst vector2 - 2 * fst vector1, 
                             snd vector2 - 2 * snd vector1, 
                             trd vector2 - 2 * trd vector1)

theorem collinear_vectors : ∃ γ : ℝ, c1 = γ • c2 := 
by 
  intro vector1 vector2 
  intro c1_def c2_def
  sorry

end collinear_vectors_l371_371420


namespace pow_div_l371_371323

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l371_371323


namespace find_perpendicular_slope_value_l371_371872

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end find_perpendicular_slope_value_l371_371872


namespace range_of_a_inequality_ln_sum_l371_371886

-- Define the function as in the problem
def f (a x : ℝ) : ℝ := (1 - x) / (a * x) + log x

-- Define the derivative of the function
def f' (a x : ℝ) : ℝ := (a * x - 1) / (a * (x^2))

-- Problem 1: Prove the range of 'a'
theorem range_of_a (a : ℝ) (h_inc : ∀ x : ℝ, 1 ≤ x → f' a x ≥ 0) : 1 ≤ a :=
by
  sorry

-- Problem 2: Prove the inequality for a = 1 and n ≥ 2
theorem inequality_ln_sum (n : ℕ) (h_n : 2 ≤ n) :
  (∑ i in finset.range n, 1 / (i + 2 : ℝ)) < log n ∧ 
  log n < n + ∑ i in finset.range (n - 1), 1 / (i + 2 : ℝ) :=
by
  sorry

end range_of_a_inequality_ln_sum_l371_371886


namespace possible_values_of_a_l371_371863

variable (a : ℝ)

def A : set ℝ := {0, 1}
def B (a : ℝ) : set ℝ := {x | a * x^2 + x - 1 = 0}

theorem possible_values_of_a :
  A ⊇ B a ↔ (a = 0 ∨ a < -1 / 4) := by sorry

end possible_values_of_a_l371_371863


namespace ellipse_equation_ellipse_trajectory_midpoint_max_triangle_area_l371_371568

theorem ellipse_equation 
  (F1 F2 : ℝ × ℝ) 
  (min_dist_to_right_focus : ℝ) 
  (hx : F1 = (-2 * Real.sqrt 2, 0) ∧ F2 = (2 * Real.sqrt 2, 0)) 
  (hy : min_dist_to_right_focus = 3 - 2 * Real.sqrt 2) :
  ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (∀ x y : ℝ, ((x^2) / (3 * 3) + y^2 = 1)) := 
sorry

theorem ellipse_trajectory_midpoint 
  (line_slope : ℝ) 
  (F1 F2 : ℝ × ℝ) 
  (hx : F1 = (-2 * Real.sqrt 2, 0) ∧ F2 = (2 * Real.sqrt 2, 0)) 
  (hy : min_dist_to_right_focus = 3 - 2 * Real.sqrt 2) :
  ∃ a b u v : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ u < v ∧ (∀ x y : ℝ, ((x - 18 * y = 0) ∧ (- (18 * Real.sqrt 37) / 37 < x < (18 * Real.sqrt 37) / 37))) :=
sorry

theorem max_triangle_area 
  (F1 F2 : ℝ × ℝ) 
  (O : ℝ × ℝ) 
  (hx : F1 = (-2 * Real.sqrt 2, 0) ∧ F2 = (2 * Real.sqrt 2, 0) ∧ O = (0, 0)) 
  (hy : min_dist_to_right_focus = 3 - 2 * Real.sqrt 2) :
  ∃ max_area : ℝ, max_area = (3 / 2) := 
sorry

end ellipse_equation_ellipse_trajectory_midpoint_max_triangle_area_l371_371568


namespace find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l371_371100

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Prove that a = 2 given the slope condition at x = 0
theorem find_a (a : ℝ) (h : f_prime 0 a = -1) : a = 2 :=
by sorry

-- Characteristics of the function f(x)
theorem monotonic_intervals (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, (x ≤ Real.log 2 → f_prime x a ≤ 0) ∧ (x >= Real.log 2 → f_prime x a >= 0) :=
by sorry

-- Prove that e^x > x^2 + 1 when x > 0
theorem exp_gt_xsquare_plus_one (x : ℝ) (hx : x > 0) : Real.exp x > x^2 + 1 :=
by sorry

end find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l371_371100


namespace probability_three_tails_two_heads_five_coins_l371_371913

theorem probability_three_tails_two_heads_five_coins : 
  (∃ p : ℚ, p = 5 / 16 ∧ 
  ∀ (n : ℕ) (k : ℕ), (n = 5) ∧ (k = 3) → 
  let single_seq_prob := (1/2 : ℚ)^n in 
  let favorable_seqs := Nat.choose n k in
  p = favorable_seqs * single_seq_prob) :=
begin
  sorry
end

end probability_three_tails_two_heads_five_coins_l371_371913


namespace total_length_of_sticks_l371_371589

theorem total_length_of_sticks :
  ∃ (s1 s2 s3 : ℝ), s1 = 3 ∧ s2 = 2 * s1 ∧ s3 = s2 - 1 ∧ (s1 + s2 + s3 = 14) := by
  sorry

end total_length_of_sticks_l371_371589


namespace divisibility_condition_l371_371600

theorem divisibility_condition (n m : ℕ) 
  (f : Polynomial ℤ)
  (h1 : f.degree = n)
  (h2 : (∀ k : ℤ, m ∣ f.eval k))
  (h3 : Int.gcd (f.coeff 0) (Int.gcd (f.coeff 1) (Int.gcd (f.coeff 2) (Int.gcd (f.coeff 3) ... (Int.gcd (f.coeff n) m)))) = 1) 
  : m ∣ n! := 
sorry

end divisibility_condition_l371_371600


namespace ellipse_distance_l371_371240

noncomputable def ellipse_problem (x y : ℝ) (a : ℝ) :=
  ∃ P : ℝ × ℝ, let PF1 := 5 in -- distance from P to one focus
  (PF1 = 5 ∧ (x^2 / 25 + y^2 / 9 = 1) ∧ a = 5 ∧ 2 * a - PF1 = 5)

theorem ellipse_distance (x y : ℝ) (a : ℝ) (h : ellipse_problem x y a) :
  ∃ P : ℝ × ℝ, let PF2 := 2 * a - 5 in PF2 = 5 := sorry

end ellipse_distance_l371_371240


namespace coordinates_of_point_P_l371_371246

theorem coordinates_of_point_P :
  ∀ (P : ℝ × ℝ), (P.1, P.2) = -1 ∧ (P.2 = -Real.sqrt 3) :=
by
  sorry

end coordinates_of_point_P_l371_371246


namespace jill_arrives_earlier_by_30_minutes_l371_371193

theorem jill_arrives_earlier_by_30_minutes :
  ∀ (d : ℕ) (v_jill v_jack : ℕ),
  d = 2 →
  v_jill = 12 →
  v_jack = 3 →
  ((d / v_jack) * 60 - (d / v_jill) * 60) = 30 :=
by
  intros d v_jill v_jack hd hvjill hvjack
  sorry

end jill_arrives_earlier_by_30_minutes_l371_371193


namespace rectangle_2010_position_l371_371272

def transform_90 (s : String) : String :=
  s[3].to_string ++ s[0].to_string ++ s[1].to_string ++ s[2].to_string

def reflect_vertical (s : String) : String :=
  s[2].to_string ++ s[1].to_string ++ s[0].to_string ++ s[3].to_string

def reflect_horizontal (s : String) : String :=
  s[3].to_string ++ s[2].to_string ++ s[1].to_string ++ s[0].to_string

def transformation_sequence (n : Nat) : String :=
  match n % 4 with
  | 0 => "ABCD"
  | 1 => "DABC"
  | 2 => "CBAD"
  | 3 => "DCBA"
  | _ => panic! "unreachable"

theorem rectangle_2010_position : transformation_sequence 2010 = "DABC" :=
  sorry

end rectangle_2010_position_l371_371272


namespace domain_of_f_l371_371270

noncomputable def f (x : ℝ) : ℝ := Real.log (4 - x^2)

theorem domain_of_f :
  {x : ℝ | 4 - x^2 > 0} = set.Ioo (-2 : ℝ) (2 : ℝ) :=
by {
  sorry
}

end domain_of_f_l371_371270


namespace eccentricity_relationship_l371_371606

noncomputable def ellipse_eccentricity (e1 : ℝ) : Prop := e1 > 0 ∧ e1 < 1
noncomputable def hyperbola_eccentricity (e2 : ℝ) : Prop := e2 > 1

theorem eccentricity_relationship
  (e1 e2 : ℝ)
  (F1 F2 P : ℝ × ℝ)
  (common_foci : F1 ≠ F2)
  (common_point : P ≠ F1 ∧ P ≠ F2)
  (eq_focal_distance : dist P F1 + dist P F2 = dist F1 F2) :
  ellipse_eccentricity e1 →
  hyperbola_eccentricity e2 →
  (|e1 * e2| / sqrt (e1^2 + e2^2) = sqrt (2) / 2) :=
by
  intros h_e1 h_e2
  sorry

end eccentricity_relationship_l371_371606


namespace total_expenditure_of_Louis_l371_371615

def fabric_cost (yards price_per_yard : ℕ) : ℕ :=
  yards * price_per_yard

def thread_cost (spools price_per_spool : ℕ) : ℕ :=
  spools * price_per_spool

def total_cost (yards price_per_yard pattern_cost spools price_per_spool : ℕ) : ℕ :=
  fabric_cost yards price_per_yard + pattern_cost + thread_cost spools price_per_spool

theorem total_expenditure_of_Louis :
  total_cost 5 24 15 2 3 = 141 :=
by
  sorry

end total_expenditure_of_Louis_l371_371615


namespace cos_phi_half_l371_371743

theorem cos_phi_half (PQRS X Y : ℝ) [Square PQRS] (HX : midpoint X QR) (HY : midpoint Y RS) :
  ∃ ϕ : ℝ, cos ϕ = 1 / 2 := sorry

end cos_phi_half_l371_371743


namespace counterfeit_identifiable_in_two_weighings_l371_371946

-- Define the condition that one of four coins is counterfeit
def is_counterfeit (coins : Fin 4 → ℚ) (idx : Fin 4) : Prop :=
  ∃ real_weight counterfeit_weight : ℚ, real_weight ≠ counterfeit_weight ∧
  (∀ i : Fin 4, i ≠ idx → coins i = real_weight) ∧ coins idx = counterfeit_weight

-- Define the main theorem statement
theorem counterfeit_identifiable_in_two_weighings (coins : Fin 4 → ℚ) :
  (∃ idx : Fin 4, is_counterfeit coins idx) → ∃ idx : Fin 4, is_counterfeit coins idx ∧
  ∀ (balance : (Fin 4 → Prop) → ℤ → Prop), (∃ w1 w2 : Fin 4 → Prop, balance w1 = 0 ∨ balance w2 = 0 → idx) :=
sorry

end counterfeit_identifiable_in_two_weighings_l371_371946


namespace solve_system_l371_371261

theorem solve_system (x y z u : ℝ) :
  x^3 * y^2 * z = 2 ∧
  z^3 * u^2 * x = 32 ∧
  y^3 * z^2 * u = 8 ∧
  u^3 * x^2 * y = 8 →
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
  (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
  (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
  (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2) :=
sorry

end solve_system_l371_371261


namespace determine_ts_l371_371438

theorem determine_ts :
  ∃ t s : ℝ, 
  (⟨3, 1⟩ : ℝ × ℝ) + t • (⟨4, -6⟩) = (⟨0, 2⟩ : ℝ × ℝ) + s • (⟨-3, 5⟩) :=
by
  use 6, -9
  sorry

end determine_ts_l371_371438


namespace equation_of_line_bisecting_ab_length_of_ab_with_inclination_l371_371065

/-- Given an ellipse and point M, prove the equation of line l if AB is bisected by M. -/
theorem equation_of_line_bisecting_ab
  (x y : ℝ)
  (h_ellipse : x^2 / 24 + y^2 / 12 = 1)
  (M : ℝ × ℝ) (h_M : M = (3, 1))
  (A B : ℝ × ℝ) (l : ℝ → ℝ)
  (h_A : A.1^2 / 24 + A.2^2 / 12 = 1)
  (h_B : B.1^2 / 24 + B.2^2 / 12 = 1)
  (h_bisect : M = ((A.1 + B.1)/2, (A.2 + B.2)/2)) :
  (∀ t : ℝ, l t = 3 * t + 2 - 11) :=
sorry

/-- Given an ellipse, point M, and line l with an inclination π/4, prove the length of AB. -/
theorem length_of_ab_with_inclination
  (x y : ℝ)
  (h_ellipse : x^2 / 24 + y^2 / 12 = 1)
  (M : ℝ × ℝ) (h_M : M = (3, 1))
  (l : ℝ → ℝ) (h_inclination : ∀ t : ℝ, l t = t - 2)
  (A B : ℝ × ℝ) (h_A : A.1^2 / 24 + A.2^2 / 12 = 1)
  (h_B : B.1^2 / 24 + B.2^2 / 12 = 1) :
  (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (16/3) * sqrt 2) :=
sorry

end equation_of_line_bisecting_ab_length_of_ab_with_inclination_l371_371065


namespace calc_x2015_l371_371216

noncomputable def f (x a : ℝ) : ℝ := x / (a * (x + 2))

theorem calc_x2015 (a x x_0 : ℝ) (x_seq : ℕ → ℝ)
  (h_unique: ∀ x, f x a = x → x = 0) 
  (h_a_val: a = 1 / 2)
  (h_f_x0: f x_0 a = 1 / 1008)
  (h_seq: ∀ n, x_seq (n + 1) = f (x_seq n) a)
  (h_x0_val: x_seq 0 = x_0):
  x_seq 2015 = 1 / 2015 :=
by
  sorry

end calc_x2015_l371_371216


namespace geometric_sequence_common_ratio_l371_371558

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : a 5 = 16)
  (h_pos : ∀ n : ℕ, 0 < a n) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l371_371558


namespace find_f_107_5_l371_371082

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_107_5 (h1 : ∀ x : ℝ, f(x + 3) = - 1 / f(x))
                     (h2 : ∀ x : ℝ, f(-x) = f(x))
                     (h3 : ∀ x : ℝ, x < 0 → f(x) = 4 * x) :
  f(107.5) = 1 / 10 :=
sorry

end find_f_107_5_l371_371082


namespace three_lines_intersection_l371_371548

theorem three_lines_intersection :
  let line1 := {p : ℝ × ℝ | 4 * p.2 - 3 * p.1 = 2}
  let line2 := {p : ℝ × ℝ | p.1 + 3 * p.2 = 3}
  let line3 := {p : ℝ × ℝ | 8 * p.1 - 12 * p.2 = 9}
  (set.finite ((line1 ∩ line2) ∪ (line1 ∩ line3) ∪ (line2 ∩ line3)).to_finset) ∧
  (set.card ((line1 ∩ line2) ∪ (line1 ∩ line3) ∪ (line2 ∩ line3)).to_finset) = 3 :=
by
  sorry

end three_lines_intersection_l371_371548


namespace not_all_on_C_implies_exists_not_on_C_l371_371923

def F (x y : ℝ) : Prop := sorry  -- Define F according to specifics
def on_curve_C (x y : ℝ) : Prop := sorry -- Define what it means to be on curve C according to specifics

theorem not_all_on_C_implies_exists_not_on_C (h : ¬ ∀ x y : ℝ, F x y → on_curve_C x y) :
  ∃ x y : ℝ, F x y ∧ ¬ on_curve_C x y := sorry

end not_all_on_C_implies_exists_not_on_C_l371_371923


namespace total_length_of_sticks_l371_371588

theorem total_length_of_sticks :
  ∃ (s1 s2 s3 : ℝ), s1 = 3 ∧ s2 = 2 * s1 ∧ s3 = s2 - 1 ∧ (s1 + s2 + s3 = 14) := by
  sorry

end total_length_of_sticks_l371_371588


namespace smallest_N_l371_371674

theorem smallest_N (N : ℕ) :
  (∀ (f : Fin 1008 → (ℕ × ℕ)),
    (∀ i, (∃ x y, x ≠ y ∧ x, y ∈ (List.range 2016).map (Nat.succ) ∧ f i = (x, y)) ∧ ∀ i, (f i).fst ≤ (f i).snd ∧ (f i).fst * (f i).snd ≤ N)) ↔ N ≥ 1017072 :=
sorry

end smallest_N_l371_371674


namespace xy_series_16_l371_371808

noncomputable def series (x y : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * (x * y)^n

theorem xy_series_16 (x y : ℝ) (h_series : series x y = 16) (h_abs : |x * y| < 1) :
  (x = 3 / 4 ∧ (y = 1 ∨ y = -1)) :=
sorry

end xy_series_16_l371_371808


namespace right_triangle_30_60_90_hypotenuse_l371_371178

theorem right_triangle_30_60_90_hypotenuse
  (A B C : Type)
  [tri : Triangle A B C]
  (hC : ∠C = 90)
  (hA : ∠A = 30)
  (BC : SegmentLength B C = 5) :
  SegmentLength A B = 10 :=
sorry

end right_triangle_30_60_90_hypotenuse_l371_371178


namespace degrees_to_radians_150_l371_371435

theorem degrees_to_radians_150 :
  (150 : ℝ) * (Real.pi / 180) = (5 * Real.pi) / 6 :=
by
  sorry

end degrees_to_radians_150_l371_371435


namespace prod_simplify_l371_371259

theorem prod_simplify : 
    (∏ n in Ico 2 995, (3 * n + 3) / (3 * n)) = 994 := by
    sorry

end prod_simplify_l371_371259


namespace count_n_repetitive_permutations_l371_371021

theorem count_n_repetitive_permutations (n : ℕ) : 
  let N_n := (6^n - 2 * 5^n + 4^n) / 4
  in N_n = (6^n - 2 * 5^n + 4^n) / 4 := by
  sorry

end count_n_repetitive_permutations_l371_371021


namespace a_sq_minus_b_sq_l371_371116

noncomputable def vec_diff : ℤ × ℤ → ℤ := λ v,
  (v.1 * v.1 + v.2 * v.2)

theorem a_sq_minus_b_sq (a b : ℤ × ℤ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0))
  (h1 : a.1 + b.1 = -3 ∧ a.2 + b.2 = 6)
  (h2 : a.1 - b.1 = -3 ∧ a.2 - b.2 = 2) :
  vec_diff a - vec_diff b = 21 := 
sorry

end a_sq_minus_b_sq_l371_371116


namespace petyas_square_is_larger_l371_371627

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l371_371627


namespace snoring_heart_disease_related_l371_371182

-- Define the conditions given in the problem
def critical_value_95 := 3.841
def critical_value_99 := 6.635
def k_value := 20.87

-- Define what it means for events to be related given specific probabilities
def is_related_95 (k: ℝ) := k > critical_value_95
def is_related_99 (k: ℝ) := k > critical_value_99

-- State the main theorem to be proven
theorem snoring_heart_disease_related (k: ℝ) :
  k = k_value → is_related_99 k := 
sorry

end snoring_heart_disease_related_l371_371182


namespace lcm_4_8_9_10_l371_371338

theorem lcm_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 :=
by
  -- Definitions of the numbers (additional definitions from problem conditions)
  let four := 4 
  let eight := 8
  let nine := 9
  let ten := 10
  
  -- Prime factorizations:
  have h4 : Nat.prime_factors four = [2, 2],
    from rfl
  
  have h8 : Nat.prime_factors eight = [2, 2, 2],
    from rfl

  have h9 : Nat.prime_factors nine = [3, 3],
    from rfl

  have h10 : Nat.prime_factors ten = [2, 5],
    from rfl

  -- Least common multiple calculation
  let highest_2 := 2 ^ 3
  let highest_3 := 3 ^ 2
  let highest_5 := 5

  -- Multiply together
  let lcm := highest_2 * highest_3 * highest_5

  show Nat.lcm (Nat.lcm four eight) (Nat.lcm nine ten) = lcm
  sorry

end lcm_4_8_9_10_l371_371338


namespace integral_evaluation_l371_371055

-- Definitions for given conditions
variable {a : ℝ}
variable (h_positive : a > 0)
variable (h_constant_term : polynomial.constant_coeff (polynomial.X^{-1} * a - polynomial.X)^6 = 15)

-- The theorem statement we want to prove
theorem integral_evaluation :
  ∫ (x : ℝ) in -a..a, (sqrt (1 - x^2) + sin (2 * x)) = π / 2 :=
sorry

end integral_evaluation_l371_371055


namespace ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l371_371771

section

variable {a b c : ℝ}

-- Statement 1
theorem ac_le_bc_if_a_gt_b_and_c_le_zero (h1 : a > b) (h2 : c ≤ 0) : a * c ≤ b * c := 
  sorry

-- Statement 2
theorem a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero (h1 : a * c ^ 2 > b * c ^ 2) (h2 : b ≥ 0) : a ^ 2 > b ^ 2 := 
  sorry

-- Statement 3
theorem log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1 (h1 : a > b) (h2 : b > -1) : Real.log (a + 1) > Real.log (b + 1) := 
  sorry

-- Statement 4
theorem inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero (h1 : a > b) (h2 : a * b > 0) : 1 / a < 1 / b := 
  sorry

end

end ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l371_371771


namespace petya_square_larger_l371_371632

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end petya_square_larger_l371_371632


namespace common_divisors_84_90_l371_371144

def divisors (n : ℕ) : Set ℤ :=
  { d : ℤ | d ∣ n }

def common_divisor_count (a b : ℕ) : ℕ :=
  (divisors a ∩ divisors b).toFinset.card

theorem common_divisors_84_90 : common_divisor_count 84 90 = 8 := by
  sorry

end common_divisors_84_90_l371_371144


namespace smallest_N_l371_371675

theorem smallest_N (N : ℕ) :
  (∀ (f : Fin 1008 → (ℕ × ℕ)),
    (∀ i, (∃ x y, x ≠ y ∧ x, y ∈ (List.range 2016).map (Nat.succ) ∧ f i = (x, y)) ∧ ∀ i, (f i).fst ≤ (f i).snd ∧ (f i).fst * (f i).snd ≤ N)) ↔ N ≥ 1017072 :=
sorry

end smallest_N_l371_371675


namespace side_length_c_4_l371_371062

theorem side_length_c_4 (A : ℝ) (b S c : ℝ) 
  (hA : A = 120) (hb : b = 2) (hS : S = 2 * Real.sqrt 3) : 
  c = 4 :=
sorry

end side_length_c_4_l371_371062


namespace xiao_ming_zi_shi_probability_l371_371932

/--
In ancient China, a day and night were divided into twelve time periods, each called a "shi chen."
The correspondence between ancient and modern times (partially) is shown below. Xiao Ming and three
other students from the astronomy interest group will be conducting relay observations from tonight 
at 23:00 to tomorrow morning at 7:00. Each person will observe for two hours, and the order of observation 
is randomly determined by drawing lots. We need to prove that the probability of Xiao Ming observing during 
the "zi shi" period is 1/4.

| Ancient Time | Zi Shi    | Chou Shi | Yin Shi | Mao Shi |
|--------------|-----------|----------|---------|---------|
| Modern Time  | 23:00~1:00| 1:00~3:00| 3:00~5:00| 5:00~7:00|
-/
theorem xiao_ming_zi_shi_probability :
  let total_hours := 8
  let students := 4
  let hours_per_student := 2
  let zi_shi_duration := 2
  probability_of_zi_shi := zi_shi_duration / total_hours
  probability_of_zi_shi = (1 : ℚ) / 4 :=
by sorry

end xiao_ming_zi_shi_probability_l371_371932


namespace product_of_coefficients_is_negative_integer_l371_371284

theorem product_of_coefficients_is_negative_integer
  (a b c : ℤ)
  (habc_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (discriminant_positive : (b * b - 4 * a * c) > 0)
  (product_cond : a * b * c = (c / a)) :
  ∃ k : ℤ, k < 0 ∧ k = a * b * c :=
by
  sorry

end product_of_coefficients_is_negative_integer_l371_371284


namespace ratio_of_perimeters_l371_371764

theorem ratio_of_perimeters (r : ℝ) (h_r : r = 5) :
  let s1 := 2 * Real.sqrt (5),
      s2 := 5 * Real.sqrt (2),
      perimeter_s1 := 4 * s1,
      perimeter_s2 := 4 * s2
  in perimeter_s1 / perimeter_s2 = Real.sqrt (10) / 5 :=
by
  let s1 := 2 * Real.sqrt (5)
  let s2 := 5 * Real.sqrt (2)
  let perimeter_s1 := 4 * s1
  let perimeter_s2 := 4 * s2
  sorry

end ratio_of_perimeters_l371_371764


namespace b_seq_arithmetic_sum_c_seq_l371_371059

open Real Nat

-- Conditions from the problem:
def a_seq (n : ℕ) : ℝ := (1 / 4) ^ n

def b_seq (n : ℕ) : ℝ := 3 * log (1 / 4) (a_seq n) - 2

def c_seq (n : ℕ) : ℝ := a_seq n * b_seq n

-- Mathematically equivalent proof problem:

theorem b_seq_arithmetic :
  ∃ (d : ℝ), ∀ n : ℕ, b_seq (n + 1) - b_seq n = d := sorry

theorem sum_c_seq (n : ℕ) :
  ∑ k in range (n + 1), c_seq k = 2 / 3 - 4 * (3 * (n + 1) + 2) / 3 * (1 / 4) ^ (n + 2) := sorry

end b_seq_arithmetic_sum_c_seq_l371_371059


namespace AM_values_l371_371718

noncomputable def given_conditions : Prop :=
  let BM : ℝ := 2
  let AB : ℝ := 3
  let CM : ℝ := 9
  let EM : ℝ := 2
  let MD : ℝ := 2
  let MF : ℝ := 6
  let possible_AM : ℝ := 1 ∨ 4 in
  ∃ AM : ℝ, (AM = 1 ∨ AM = 4) ∧ (BM = 2 ∧ AB = 3 ∧ CM = 9 ∧ EM = 2 ∧ MD = 2 ∧ MF = 6)

theorem AM_values : given_conditions :=
  sorry

end AM_values_l371_371718


namespace exist_infinite_prime_pairs_l371_371256

open Nat

theorem exist_infinite_prime_pairs (p q: ℕ) [fact (Nat.prime p)] [fact (Nat.prime q)] : 
  (∃∞ (p q: ℕ),
    p ∣ (2 ^ (q - 1) - 1) ∧ q ∣ (2 ^ (p - 1) - 1)) :=
sorry

end exist_infinite_prime_pairs_l371_371256


namespace median_unchanged_l371_371175

def original_donations : List ℕ := [30, 50, 50, 60, 60]

def modified_donations : List ℕ := [50, 50, 50, 60, 60]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted[(sorted.length / 2)]

theorem median_unchanged :
  median original_donations = median modified_donations :=
by
  sorry

end median_unchanged_l371_371175


namespace gcd_consecutive_integers_gcd_positive_integers_l371_371598

theorem gcd_consecutive_integers (n : ℕ) (hn : n % 2 = 1) :
  ∀ (a b c d e f : ℕ), 
    {a, b, c, d, e, f} = {n, n + 1, n + 2, n + 3, n + 4, n + 5} →
  (∃ (x u y v z w : ℕ), 
    {x, u, y, v, z, w} = {a, b, c, d, e, f} ∧ 
    ∀ (x u y v z w : ℕ),
      gcd (x*u*w + y*u*w + z*u*v) (u*v*w) = 1 ↔ 
        gcd (x, u) = 1 ∧ gcd (y, v) = 1 ∧ gcd (z, w) = 1)
:= sorry

theorem gcd_positive_integers (n : ℕ) :
  gcd (n, 10) = 1 ∧ gcd (n + 1, 3) = 1 ∧ gcd (n + 2, 2) = 1 ↔
  ∃ (k : ℕ), n = 2 * k + 1 ∧ n % 5 ≠ 0 :=
sorry

end gcd_consecutive_integers_gcd_positive_integers_l371_371598


namespace find_original_number_l371_371389

variable (x : ℕ)

theorem find_original_number (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by
  sorry

end find_original_number_l371_371389


namespace part1_monotonic_intervals_and_extreme_value_part2_general_monotonicity_l371_371891

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a - 1) * x - Real.log x

theorem part1_monotonic_intervals_and_extreme_value (x : ℝ) (h : x > 0) :
  ∀ x, f (1/2) x = (1/2) * x^2 - Real.log x → 
  ((0 < x ∧ x < 1) → (differentiable_at ℝ (f (1/2)) x ∧ deriv (f (1/2)) x < 0)) ∧
  ((1 < x) → (differentiable_at ℝ (f (1/2)) x ∧ deriv (f (1/2)) x > 0)) ∧
  (f (1/2) 1 = 1/2) := sorry

theorem part2_general_monotonicity (a x : ℝ) (h : x > 0) :
  ∀ x, f a x = a * x^2 + (2 * a - 1) * x - Real.log x → 
  ((a ≤ 0) → (∀ x > 0, differentiable_at ℝ (f a) x ∧ deriv (f a) x < 0)) ∧
  ((a > 0) → ((0 < x ∧ x < 1 / (2 * a)) → (differentiable_at ℝ (f a) x ∧ deriv (f a) x < 0)) ∧
                ((x > 1 / (2 * a)) → (differentiable_at ℝ (f a) x ∧ deriv (f a) x > 0))) := sorry

end part1_monotonic_intervals_and_extreme_value_part2_general_monotonicity_l371_371891


namespace odd_function_f_question_1_f_f_neg_one_analytic_expression_f_l371_371076

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -x^2 - 4*x - 3 else
if x = 0 then 0 else x^2 - 4*x + 3

theorem odd_function_f (x : ℝ) : f(-x) = -f(x) := by
  sorry

theorem question_1_f_f_neg_one : f (f (-1)) = 0 := by
  have h1 : f 1 = 0 := by sorry
  have h2 : f (-1) = -f (1) := odd_function_f 1
  rw [h1, neg_zero] at h2
  show f (f (-1)) = 0,
  rw [h2],
  rw [f],
  sorry

theorem analytic_expression_f (x : ℝ) : f(x) =
  if x < 0 then -x^2 - 4*x - 3 else 
  if x = 0 then 0 else 
  x^2 - 4*x + 3 :=
by sorry

end odd_function_f_question_1_f_f_neg_one_analytic_expression_f_l371_371076


namespace percentage_profit_on_apples_l371_371355

def total_percentage_profit (total_weight : ℕ) (profit_percentage : ℝ) (portion1_percentage : ℝ) (portion2_percentage : ℝ) : ℝ := 
    let weight1 := portion1_percentage * total_weight
    let weight2 := portion2_percentage * total_weight
    let cp := 1  -- Assume cost price per kg is 1 for simplicity
    let sp1 := cp * (1 + profit_percentage)
    let sp2 := cp * (1 + profit_percentage)
    let total_cp := total_weight * cp
    let total_sp := weight1 * sp1 + weight2 * sp2
    let total_profit := total_sp - total_cp
    (total_profit / total_cp) * 100

-- Proof statement
theorem percentage_profit_on_apples : total_percentage_profit 280 0.20 0.40 0.60 = 20 :=
    sorry

end percentage_profit_on_apples_l371_371355


namespace eight_pow_15_div_sixtyfour_pow_6_l371_371314

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l371_371314


namespace prime_triples_l371_371453

open Nat

theorem prime_triples (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) :
    (p ∣ q^r + 1) → (q ∣ r^p + 1) → (r ∣ p^q + 1) → (p, q, r) = (2, 5, 3) ∨ (p, q, r) = (3, 2, 5) ∨ (p, q, r) = (5, 3, 2) :=
  by
  sorry

end prime_triples_l371_371453


namespace find_a_and_parity_f_is_increasing_l371_371099

theorem find_a_and_parity (a : ℝ) (h : f (1/2) = 2/5) 
  (h1 : ∀ x ∈ Ioi (-1) ∩ Iio (1), f x = a * x / (1 + x^2)) :
  a = 1 ∧ ∀ x ∈ Ioi (-1) ∩ Iio (1), f (-x) = -f x :=
by 
  sorry 

theorem f_is_increasing (a : ℝ) (h : f (1/2) = 2/5) 
  (h1 : ∀ x ∈ Ioi (-1) ∩ Iio (1), f x = x / (1 + x^2)) :
  ∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂ :=
by
  sorry

end find_a_and_parity_f_is_increasing_l371_371099


namespace projection_vector_satisfies_conditions_l371_371000

variable (v1 v2 : ℚ)

def line_l (t : ℚ) : ℚ × ℚ :=
(2 + 3 * t, 5 - 2 * t)

def line_m (s : ℚ) : ℚ × ℚ :=
(-2 + 3 * s, 7 - 2 * s)

theorem projection_vector_satisfies_conditions :
  3 * v1 + 2 * v2 = 6 ∧ 
  ∃ k : ℚ, v1 = k * 3 ∧ v2 = k * (-2) → 
  (v1, v2) = (18 / 5, -12 / 5) :=
by
  sorry

end projection_vector_satisfies_conditions_l371_371000


namespace tetrahedron_triangle_inequalities_l371_371642

noncomputable theory
open_locale classical

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

def tetrahedron (AB CD AC BD AD BC : ℝ) : Prop :=
  AB + CD < AC + BD + AD + BC ∧
  AC + BD < AB + CD + AD + BC ∧
  AD + BC < AB + CD + AC + BD

theorem tetrahedron_triangle_inequalities (AB CD AC BD AD BC : ℝ)
  (h1 : AB < AC + BC)
  (h2 : AB < AD + BD)
  (h3 : CD < AC + AD)
  (h4 : CD < BC + BD) :
  tetrahedron AB CD AC BD AD BC := by
  have h5 : AB + CD < (AC + BC) + (AD + BD),
  { linarith [h1, h2, h3, h4] },
  have h6 : AC + BD < (AB + CD) + (AD + BC),
  { linarith [h1, h2, h3, h4] },
  have h7 : AD + BC < (AB + CD) + (AC + BD),
  { linarith [h1, h2, h3, h4] },
  exact ⟨h5, h6, h7⟩

example : tetrahedron 1 2 3 4 5 6 :=
begin
  sorry
end

end tetrahedron_triangle_inequalities_l371_371642


namespace ellipse_semimajor_axis_value_l371_371995

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371995


namespace cupcakes_per_package_l371_371201

theorem cupcakes_per_package (total_cupcakes : ℕ) (eaten_cupcakes : ℕ) (packages : ℕ) (remaining_cupcakes := total_cupcakes - eaten_cupcakes) 
  (cupcakes_per_package := remaining_cupcakes / packages) :
  total_cupcakes = 18 → eaten_cupcakes = 8 → packages = 5 → cupcakes_per_package = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold remaining_cupcakes
  unfold cupcakes_per_package
  norm_num
  sorry -- proof to be completed

end cupcakes_per_package_l371_371201


namespace largest_prime_factor_of_4620_l371_371727

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / m → ¬ (m ∣ n)

def prime_factors (n : ℕ) : List ℕ :=
  -- assumes a well-defined function that generates the prime factor list
  -- this is a placeholder function for demonstrating purposes
  sorry

def largest_prime_factor (l : List ℕ) : ℕ :=
  l.foldr max 0

theorem largest_prime_factor_of_4620 : largest_prime_factor (prime_factors 4620) = 11 :=
by
  sorry

end largest_prime_factor_of_4620_l371_371727


namespace amount_received_by_Sam_l371_371645

noncomputable def final_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem amount_received_by_Sam 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hP : P = 12000) (hr : r = 0.10) (hn : n = 2) (ht : t = 1) :
  final_amount P r n t = 12607.50 :=
by
  sorry

end amount_received_by_Sam_l371_371645


namespace max_distance_proof_l371_371934

namespace ProofExample

open Real

-- Define the parametric form of line l
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (-4 + (sqrt 2 / 2) * t, -2 + (sqrt 2 / 2) * t)

-- Define the Cartesian form of line l
def line_cartesian (x y : ℝ) : Prop :=
  y = x + 2

-- Define the polar form of the curve C
def curve_polar (rho theta : ℝ) : Prop :=
  rho = 2 * cos theta

-- Define the Cartesian form of the curve C
def curve_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x = 0

-- Define the condition for the parallel line to line l where b = -1 ± sqrt(2)
def parallel_line (x y b : ℝ) : Prop :=
  y = x + b

-- Define the maximum distance from any point P on curve C to line l
def max_distance_to_line (x y : ℝ) : ℝ :=
  abs (2 - (-1 - sqrt 2)) / sqrt 2 + 1

theorem max_distance_proof :
  ∀ (x y : ℝ), curve_cartesian x y →
  max_distance_to_line x y = (3 * sqrt 2 / 2) + 1 :=
sorry

end ProofExample

end max_distance_proof_l371_371934


namespace trigonometric_simplification_l371_371363

theorem trigonometric_simplification :
  (let sin15 := (Real.sqrt 6 - Real.sqrt 2) / 4 in
   let cos15 := (Real.sqrt 6 + Real.sqrt 2) / 4 in
   Real.sqrt (sin15 ^ 4 + 4 * cos15 ^ 2) - Real.sqrt (cos15 ^ 4 + 4 * sin15 ^ 2)) = (1 / 2) * Real.sqrt 3 := 
by
  sorry

end trigonometric_simplification_l371_371363


namespace price_increase_40_percent_l371_371281

theorem price_increase_40_percent (P Q P_new : ℝ) (Q_decrease_percent effect_on_revenue : ℝ) :
  Q_decrease_percent = 0.2 →
  effect_on_revenue = 0.12 →
  P_new * (Q * (1 - Q_decrease_percent)) = P * Q * (1 + effect_on_revenue) →
  P_new = 1.4 * P :=
begin
  intros h1 h2 h3,
  calc
  P_new = (1 + effect_on_revenue) * P * Q / (Q * (1 - Q_decrease_percent)) : by { rw h3 }
        = (1 + 0.12) * P * Q / (Q * (1 - 0.2)) : by { rw [h1, h2] }
        = 1.12 * P / 0.8 : by
        { field_simp,
          ring }
        = 1.4 * P : by
          { exact_mod_cast h3 },
end

end price_increase_40_percent_l371_371281


namespace sum_of_valid_three_digit_numbers_l371_371343

/-- A three-digit number is defined by its hundreds, tens, and units digits. -/
def three_digit_number (a b c : ℕ) := 100 * a + 10 * b + c

/-- The sum of the digits of a three-digit number. -/
def digit_sum (a b c : ℕ) := a + b + c

/-- The product of the digits of a three-digit number. -/
def digit_product (a b c : ℕ) := a * b * c

/-- The sum of three-digit numbers that satisfy the given conditions. -/
theorem sum_of_valid_three_digit_numbers : 
  ∑ n in {n | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 
          digit_sum a b c ∣ three_digit_number a b c ∧
          digit_product a b c ∣ three_digit_number a b c}, n = 444 := 
by
  sorry

end sum_of_valid_three_digit_numbers_l371_371343


namespace find_min_max_K_l371_371307

-- Assume non-negativity for x, y, z
variables (x y z : ℝ)
-- Define the conditions of the system
def condition1 := 4 * x + y + 2 * z = 4
def condition2 := 3 * x + 6 * y - 2 * z = 6
-- Define the expression K
def K := 5 * x - 6 * y + 7 * z

theorem find_min_max_K (h1 : condition1) (h2 : condition2) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) : 
  -2 ≤ K ∧ K ≤ 14 := by
  sorry

end find_min_max_K_l371_371307


namespace intersecting_lines_ratio_l371_371551

theorem intersecting_lines_ratio
  (A B C D F Q : Type)
  [triangle A B C]
  [on_line A B D]
  [on_line B C F]
  [on_line A C Q]
  [intersection_point CF AD Q]
  (h1 : CD / DB = 2 / 1)
  (h2 : AF / FB = 2 / 3) :
  CQ / QF = 1 := 
by
  sorry

end intersecting_lines_ratio_l371_371551


namespace ellipse_semimajor_axis_value_l371_371994

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371994


namespace average_age_population_l371_371555

/-- 
In a certain population the ratio of the number of women to the number of men is 5 to 3. 
If the average (arithmetic mean) age of the women is 25 years and the average age of the men is 31 years,
then the average age of the population is 27.25 years.
-/
theorem average_age_population (k : ℕ) (h_pos : k > 0) :
  let women := 5 * k,
      men := 3 * k,
      total_population := women + men,
      total_age := (25 * women) + (31 * men)
  in (total_age / total_population : ℝ) = 27.25 :=
  sorry

end average_age_population_l371_371555


namespace possible_triangle_area_l371_371708

noncomputable def sqrt : ℝ → ℝ := Real.sqrt  -- define sqrt function as necessary

theorem possible_triangle_area (a b : ℕ) (h : (7 : ℕ) ∈ [a, b, 7]) (theta : ℝ) (h_theta : theta = π / 3) :
  let area_1 := (7^2 * sqrt 3) / 4,
      area_2 := (7 ⬝ a ⬝ sqrt 3) / 2,
      area_3 := (7 ⬝ b ⬝ sqrt 3) / 2 in
  area_1 = 21.22 ∨ area_2 = 24.25 ∨ area_3 = 18.16 :=
by sorry

end possible_triangle_area_l371_371708


namespace find_matrix_l371_371025

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M^3 - 3 * M^2 + 2 * M = ![![8, 16], ![4, 8]]) : 
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_l371_371025


namespace tangent_line_area_correct_l371_371656

-- Define the curve
def y (x : ℝ) : ℝ := (1 / 4) * x^2

-- Define the point of tangency
def pt : ℝ × ℝ := (2, 1)

-- Define the tangent line at point (2, 1)
def tangent_line (x : ℝ) : ℝ := x - 1

-- Define the area calculation for the triangle formed by the tangent line with the axes
def enclosed_area : ℝ := 1 / 2 * 1 * 1

-- State the theorem to be proved
theorem tangent_line_area_correct : enclosed_area = 1 / 2 := by
  sorry

end tangent_line_area_correct_l371_371656


namespace root_relationship_l371_371104

theorem root_relationship (m n a b : ℝ) 
  (h_eq : ∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) : a < m ∧ m < n ∧ n < b :=
by
  sorry

end root_relationship_l371_371104


namespace room_length_l371_371268

theorem room_length (L : ℕ) (h : 72 * L + 918 = 2718) : L = 25 := by
  sorry

end room_length_l371_371268


namespace solve_floor_equation_l371_371670

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem solve_floor_equation :
  (∃ x : ℝ, (floor ((x - 1) / 2))^2 + 2 * x + 2 = 0) → x = -3 :=
by
  sorry

end solve_floor_equation_l371_371670


namespace proof_problem_l371_371070

variable (x : ℝ)

-- Definitions of propositions
def p : Prop := ∀ x : ℝ, 2^x > 0
def q : Prop := ¬(∀ x : ℝ, x > 1 → x > 2)

-- Theorem statement
theorem proof_problem : p ∧ ¬q := by
  unfold p q
  have h_p : ∀ x : ℝ, 2^x > 0 := by {
    intro x,
    exact real.rpow_pos_of_pos (by norm_num) _
  }
  have h_q : ¬(¬(∀ x : ℝ, x > 1 → x > 2)) := by {
    push_neg,
    intro h,
    use [1.5, h 1.5 (by norm_num1), by norm_num1],
  }
  exact ⟨h_p, h_q⟩

end proof_problem_l371_371070


namespace jack_money_lost_l371_371954

noncomputable def jack_total_book_cost (books_per_month: ℕ) (months: ℕ) (price_per_book: ℕ) (discount_rate: ℝ) : ℕ :=
  let total_books := books_per_month * months
  let initial_cost := total_books * price_per_book
  let discount := discount_rate * (initial_cost : ℝ)
  initial_cost - discount.to_nat

theorem jack_money_lost (books_per_month: ℕ) (months: ℕ) (price_per_book: ℕ) (discount_rate: ℝ) (sell_amount: ℕ) : ℕ :=
  jack_total_book_cost books_per_month months price_per_book discount_rate - sell_amount

example : jack_money_lost 3 12 20 0.10 500 = 148 := by
  unfold jack_money_lost
  unfold jack_total_book_cost
  norm_num
  /- After unfolding and calculating intermediate steps it should result in:
     jack_money_lost 3 12 20 0.10 500 = 648 - 500 = 148 -/
  norm_num -- this verifies that indeed the final result is 148
  rfl

end jack_money_lost_l371_371954


namespace existence_of_lines_l371_371192

-- Given conditions in Lean

def k_list := List Nat

def sum_k (k : k_list) : Nat := k.foldr (Nat.add) 0
def sum_k_squared (k : k_list) : Nat := k.foldr (fun x acc => x * x + acc) 0
def sum_k_pairwise_prod (k : k_list) : Nat := 
    @List.foldl (Σi j : Nat, i < j) 
                (fun acc p => acc + k[p.fst] * k[p.snd]) 
                0 
                (List.sigma (Finset.range k.length).toList (Finset.range k.length).toList).filter(λ p => p.fst < p.snd)

theorem existence_of_lines : 
    ∃ k : k_list, 
        sum_k k = 100 ∧ 
        sum_k_pairwise_prod k = 2002 ∧ 
        sum_k_squared k = 5996 := 
    sorry

end existence_of_lines_l371_371192


namespace point_on_line_tangent_to_parabola_l371_371469

theorem point_on_line_tangent_to_parabola :
  ∃ (P : ℝ × ℝ), P = (-1/4 : ℝ, -1/4 : ℝ) ∧
  (∀ (x : ℝ), P.2 = x → P.2 = x) ∧
  ∀ (y : ℝ), P.1 = y → P.1 = y ∧ 
  ∀ (m : ℝ), (m^2 - 4*(m - 1)*P.1 = 0) → (m_1 m_2 = 4*P.1 → m_1 * m_2 = -1) :=
sorry

end point_on_line_tangent_to_parabola_l371_371469


namespace sum_of_coefficients_l371_371433

theorem sum_of_coefficients :
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ,
  (∀ x : ℤ, (2 - x)^7 = a₀ + a₁ * (1 + x)^2 + a₂ * (1 + x)^3 + a₃ * (1 + x)^4 + a₄ * (1 + x)^5 + a₅ * (1 + x)^6 + a₆ * (1 + x)^7 + a₇ * (1 + x)^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 129 := by sorry

end sum_of_coefficients_l371_371433


namespace AC_BD_skew_l371_371340

-- Let A, B, C, and D be points in ℝ^3 such that AB and CD are skew lines
variables {A B C D : ℝ × ℝ × ℝ}
def skew (l1 l2 : ℝ × ℝ × ℝ → Prop) : Prop := ¬ ∃ α : ℝ × ℝ × ℝ, ∀ p ∈ l1, ∀ q ∈ l2, p ∈ α ∧ q ∈ α ∧ α ≠ l1 ∩ l2

-- Definitions of lines AB and CD being skew
def line_AB : ℝ × ℝ × ℝ → Prop := λ p, ∃ t : ℝ, p = A + t • (B - A)
def line_CD : ℝ × ℝ × ℝ → Prop := λ p, ∃ t : ℝ, p = C + t • (D - C)

-- Definitions of lines AC and BD
def line_AC : ℝ × ℝ × ℝ → Prop := λ p, ∃ t : ℝ, p = A + t • (C - A)
def line_BD : ℝ × ℝ × ℝ → Prop := λ p, ∃ t : ℝ, p = B + t • (D - B)

-- Theorem statement: AC and BD are skew lines
theorem AC_BD_skew (hAB_skew : skew line_AB line_CD) : skew line_AC line_BD :=
by sorry

end AC_BD_skew_l371_371340


namespace speed_solution_dock_distance_solution_l371_371449

section Problem1
variables {x y : ℕ}

def speed_conditions : Prop :=
  9 * (x + y) = 270 ∧ 13.5 * (x - y) = 270

theorem speed_solution (h : speed_conditions) : x = 25 ∧ y = 5 :=
sorry
end Problem1

section Problem2
variables {a : ℕ} {x y : ℕ} (hx : x = 25) (hy : y = 5)

def dock_condition (a : ℕ) : Prop :=
  (a / (x + y)) = (270 - a) / (x - y)

theorem dock_distance_solution (h : dock_condition a) : a = 162 :=
sorry
end Problem2

end speed_solution_dock_distance_solution_l371_371449


namespace derivative_of_f_l371_371456

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((2 * x + 1) / Real.sqrt 2) + (2 * x + 1) / (4 * x^2 + 4 * x + 3)

theorem derivative_of_f (x : ℝ) : deriv f x = 8 / (4 * x^2 + 4 * x + 3)^2 :=
by
  -- Proof will be provided here
  sorry

end derivative_of_f_l371_371456


namespace problem1_problem2_part1_problem2_part2_l371_371880

-- Lean statement for Proof Problem 1
theorem problem1 (a : ℝ) :
  (∀ M : ℝ × ℝ, M = (1, a) → ∃! t : ℝ → ℝ, ∀ x y : ℝ, t x = y → x^2 + y^2 = 4) ↔ a = sqrt(3) := sorry

-- Lean statement for Proof Problem 2 part i
theorem problem2_part1 (a : ℝ) (OM d1 d2 : ℝ) :
  a = sqrt(2) → (∀ AC BD : ℝ, AC ⊥ BD → OM = sqrt(4 - d1^2) ∧ OM = sqrt(4 - d2^2)) →
  max_area (S_ABCD : ℝ) ↔ S_ABCD = 5 := sorry

-- Lean statement for Proof Problem 2 part ii
theorem problem2_part2 (a : ℝ) (d1 d2 : ℝ) :
  a = sqrt(2) → (∀ AC BD : ℝ, AC ⊥ BD → d1^2 + d2^2 = 3) →
  max_sum_lengths (sum_lengths : ℝ) ↔ sum_lengths = 2 * sqrt(10) := sorry

end problem1_problem2_part1_problem2_part2_l371_371880


namespace find_trajectory_find_line_eq_l371_371853

noncomputable def trajectoryEquation (M : ℝ × ℝ) (F : ℝ × ℝ) (d1 d2 k : ℝ) (S : ℝ) : Prop :=
  let x := M.1
  let y := M.2
  let F_x := F.1
  let F_y := F.2
  (F_x = 2 ∧ F_y = 0 ∧ d1 = real.sqrt ((x - 2) ^ 2 + y ^ 2) ∧ d2 = abs (x - 3) ∧ (d1 / d2) = (real.sqrt 6 / 3) ∧
   (trajectory := (x ^ 2 / 6) + (y ^ 2 / 2) = 1) ∧ 
   (lineEq := y = k * (x - 2)) ∧ k ≠ 0 ∧ S = real.sqrt 3)

theorem find_trajectory :
  ∃ M F d1 d2 k S, 
    trajectoryEquation M F d1 d2 k S →
    (M = (x, y)) ∧ ((x / 6) + (y / 2) = 1) :=
by admit

theorem find_line_eq :
  ∃ M F d1 d2 k S, 
    (trajectoryEquation M F d1 d2 k S) →
    ((y = k * (x - 2)) ∧ S = real.sqrt 3) →
    (lineEq = (x - y - 2 = 0) ∨ (x + y - 2 = 0)) :=
by admit

end find_trajectory_find_line_eq_l371_371853


namespace ant_crosses_same_edge_twice_in_same_direction_l371_371779

-- Definitions for the problem constraints
variable (V E : Type) [Fintype V] [Fintype E]
variables [DecidableEq V] [DecidableEq E] (dodecahedron : Graph V E)

-- Ant path definitions
variable (path : List E) (crosses_twice : ∀ e : E, 2 ∃ path.count (e = _))

-- Closed path constraint
axiom closed_path : list.last path = list.head path 

-- Never turning back constraint
axiom never_turn_back : ∀ (i : Nat) (h : i < (path.length - 1)), path.get ⟨i, h⟩ ≠ path.get ⟨i + 1, sorry⟩

-- Proof statement required
theorem ant_crosses_same_edge_twice_in_same_direction 
  (h_path: ∀ e : E, path.count e = 2 ) : 
  ∃ e : E, 2 ∃ path.count e = sorry :=
  
by 
  sorry

end ant_crosses_same_edge_twice_in_same_direction_l371_371779


namespace differentiable_limit_is_derivative_l371_371870

variable {α : Type*} [LinearOrderedField α] (f : α → α) (x0 : α)

theorem differentiable_limit_is_derivative (h_differentiable : DifferentiableAt α f x0) :
  (lim (↥) (f (x0 + h) - f x0) / h = f' x0) := sorry

end differentiable_limit_is_derivative_l371_371870


namespace line_circle_equilateral_l371_371893

theorem line_circle_equilateral (k : ℝ) 
  (hL : ∀ x y : ℝ, k * x + y - 3 = 0 → x^2 + y^2 = 3)
  (hE : ∀ x₁ y₁ x₂ y₂ : ℝ, k * x₁ + y₁ - 3 = 0 → x₁^2 + y₁^2 = 3 →
    k * x₂ + y₂ - 3 = 0 → x₂^2 + y₂^2 = 3 → x₁ ≠ x₂ ∨ y₁ ≠ y₂ → 
    dist (0, 0) (x₁, y₁) = dist (0, 0) (x₂, y₂) ∧ 
    dist (x₁, y₁) (x₂, y₂) = dist (0, 0) (x₁, y₁)) :
    k = sqrt 3 ∨ k = -sqrt 3 :=
sorry

end line_circle_equilateral_l371_371893


namespace mass_percentage_of_H_in_NH4_l371_371023

theorem mass_percentage_of_H_in_NH4 
  (mass_H : ℝ) (num_H_atoms : ℕ) (mass_N : ℝ) : 
  mass_H = 1.01 ∧ num_H_atoms = 4 ∧ mass_N = 14.01 → 
  (num_H_atoms * mass_H) / (mass_N + num_H_atoms * mass_H) * 100 = 22.38 := 
  by 
    -- Introduce the conditions
    intro h,
    obtain ⟨mass_H_eq, num_H_atoms_eq, mass_N_eq⟩ := h,
    -- Replace values with conditions
    rw [mass_H_eq, num_H_atoms_eq, mass_N_eq],
    -- Calculation step requires proof; we skip this with sorry
    sorry

end mass_percentage_of_H_in_NH4_l371_371023


namespace jackie_sleeping_hours_l371_371955

def hours_in_a_day : ℕ := 24
def work_hours : ℕ := 8
def exercise_hours : ℕ := 3
def free_time_hours : ℕ := 5
def accounted_hours : ℕ := work_hours + exercise_hours + free_time_hours

theorem jackie_sleeping_hours :
  hours_in_a_day - accounted_hours = 8 := by
  sorry

end jackie_sleeping_hours_l371_371955


namespace range_of_a_l371_371034

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, (2 * a + 1) * x + a - 2 > (2 * a + 1) * 0 + a - 2)
  (h2 : a - 2 < 0) : -1 / 2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l371_371034


namespace petya_square_larger_than_vasya_square_l371_371636

variable (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)

def petya_square_side (a b : ℝ) : ℝ := a * b / (a + b)

def vasya_square_side (a b : ℝ) : ℝ := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petya_square_larger_than_vasya_square
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  petya_square_side a b > vasya_square_side a b :=
by sorry

end petya_square_larger_than_vasya_square_l371_371636


namespace number_of_arrangements_l371_371365

/-- Define the set of people as a type -/
inductive People
  | A 
  | B 
  | C 
  | D 
  | E1 
  | E2 
  | E3

open People

/-- Define a condition where A and B must be adjacent -/
def adjacent (p1 p2 : People) (l : List People) : Prop :=
  ∃ (i : ℕ), p1 ∈ l ∧ p2 ∈ l ∧ l !! i = some p1 ∧ l !! (i + 1) = some p2 ∨ l !! i = some p2 ∧ l !! (i + 1) = some p1

/-- Define a condition where C and D must not be adjacent -/
def not_adjacent (p1 p2 : People) (l : List People) : Prop :=
  ¬ ∃ (i : ℕ), p1 ∈ l ∧ p2 ∈ l ∧ (l !! i = some p1 ∧ l !! (i + 1) = some p2 ∨ l !! i = some p2 ∧ l !! (i + 1) = some p1)

noncomputable def arrangement_count : ℕ :=
  960

theorem number_of_arrangements :
  ∃ (l : List People), l.permutations ∧ adjacent A B l ∧ not_adjacent C D l ∧ l.length = 7 :=
sorry

end number_of_arrangements_l371_371365


namespace Nell_initial_baseball_cards_l371_371237

def Nell_baseball_cards_initial (B A : ℕ) : Prop :=
  ∃ (g : ℕ), A = 18 + 55 ∧ B = 178 + 123 ∧ ∀ (A' B' : ℕ), (A' = 55) → (B' = 178) → (B' = A' + 123) ∧ (A = 55 + 18) ∧ (B = B' + 123) 

theorem Nell_initial_baseball_cards : Nell_baseball_cards_initial 301 73 :=
by
  existsi 55
  existsi 178
  repeat {split}
  assumption
  admit
  sorry -- To be completed

end Nell_initial_baseball_cards_l371_371237


namespace lcm_of_4_8_9_10_l371_371337

theorem lcm_of_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 := by
  sorry

end lcm_of_4_8_9_10_l371_371337


namespace all_nice_numbers_l371_371388

def is_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_and_third_equal (n : ℕ) : Prop :=
  let a := n / 1000 in
  let b := (n / 100) % 10 in
  let c := (n / 10) % 10 in
  let d := n % 10 in
  a = c

def second_and_fourth_equal (n : ℕ) : Prop :=
  let a := n / 1000 in
  let b := (n / 100) % 10 in
  let c := (n / 10) % 10 in
  let d := n % 10 in
  b = d

def product_divides_square (n : ℕ) : Prop :=
  let a := n / 1000 in
  let b := (n / 100) % 10 in
  let c := (n / 10) % 10 in
  let d := n % 10 in
  let product := a * b * c * d in
  product ∣ n * n

def is_nice_number (n : ℕ) : Prop :=
  is_4_digit_number n ∧
  first_and_third_equal n ∧
  second_and_fourth_equal n ∧
  product_divides_square n

def nice_numbers : List ℕ :=
  [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1212, 2424, 3636, 4848, 1515]

theorem all_nice_numbers : {n : ℕ | is_nice_number n} = nice_numbers.toFinset := 
by sorry

end all_nice_numbers_l371_371388


namespace circle_equation_tangent_to_y_axis_l371_371664

noncomputable def center : ℝ × ℝ := (-2, 3)

noncomputable def radius (center : ℝ × ℝ) : ℝ :=
  abs (center.fst)

theorem circle_equation_tangent_to_y_axis
  (center_x center_y : ℝ)
  (h_center : center = (center_x, center_y))
  (h_tangent : radius (center_x, center_y) = abs center_x) :
  ∃ (radius : ℝ), (center_x + 2)^2 + (center_y - 3)^2 = radius^2 :=
begin
  use 2, -- radius is 2
  rw [h_center],
  rw [radius],
  have : center_x = -2 := rfl,
  have : center_y = 3 := rfl,
  simp [this],
  sorry
end

end circle_equation_tangent_to_y_axis_l371_371664


namespace not_possible_to_fill_board_with_2012_l371_371351

theorem not_possible_to_fill_board_with_2012 :
  let board := matrix (fin 5) (fin 5) ℕ in
  (∀ i j : fin 5, board i j = 0) →
  (∀ steps : list (fin 5 × fin 5), 
    let new_board := steps.foldl (λ b (x, y), fun i j => 
      if (i = x ∧ (j = y ∨ j = y - 1 ∨ j = y + 1)) ∨ 
         (j = y ∧ (i = x ∨ i = x - 1 ∨ i = x + 1)) 
      then b i j + 1 else b i j) board in
      ∃ i j : fin 5, new_board i j ≠ 2012) :=
begin
  sorry
end

end not_possible_to_fill_board_with_2012_l371_371351


namespace identify_counterfeit_coin_in_two_weighings_l371_371950

theorem identify_counterfeit_coin_in_two_weighings (coins : Fin 4 → ℝ) (counterfeit : Fin 4 → Prop) :
  (∃ i, counterfeit i ∧ (∀ j, j ≠ i → coins j = coins (Fin.succ j))) →
  ∃ i, counterfeit i ∧ coins_in_two_weighings coins counterfeit i :=
by
  sorry

end identify_counterfeit_coin_in_two_weighings_l371_371950


namespace find_angles_of_isosceles_intersecting_rays_l371_371564

theorem find_angles_of_isosceles_intersecting_rays
    (A B C M : Type)
    [IsoscelesTriangle A B C]
    (vertex_angle : ∠BAC = 100)
    (ray_A_angle : ∠CAM = 30)
    (ray_B_angle : ∠CBM = 20)
    (intersection_point : M ∈ interior_triangle A B C) :
  ∠ACM = 20 ∧ ∠BCM = 80 :=
by
  sorry

end find_angles_of_isosceles_intersecting_rays_l371_371564


namespace radii_of_regular_tetrahedron_l371_371427

variable (a : ℝ)

-- Define the inscribed radius r and circumscribed radius R for a regular tetrahedron with edge length a
def inscribed_radius (a : ℝ) : ℝ := (a / 12) * Real.sqrt 6
def circumscribed_radius (a : ℝ) : ℝ := (a / 4) * Real.sqrt 6

theorem radii_of_regular_tetrahedron (a : ℝ) :
  (inscribed_radius a = (a / 12) * Real.sqrt 6) ∧ (circumscribed_radius a = (a / 4) * Real.sqrt 6) := by
  split
  · rfl
  · rfl

end radii_of_regular_tetrahedron_l371_371427


namespace point_not_on_transformed_plane_l371_371224

-- Definition of the given plane and point
def point_A : ℝ × ℝ × ℝ := (0, 1, -1)
def plane_a := {p : ℝ × ℝ × ℝ | 6 * p.1 - 5 * p.2 + 3 * p.3 = 4}
def k : ℝ := -3/4

-- Definition of the similarity transformation applied to the plane
def transformed_plane (p : ℝ × ℝ × ℝ) : Prop :=
  6 * p.1 - 5 * p.2 + 3 * p.3 = -k * 4

-- The theorem to be proved
theorem point_not_on_transformed_plane : ¬ transformed_plane point_A :=
by
  unfold transformed_plane point_A
  simp
  sorry

end point_not_on_transformed_plane_l371_371224


namespace ellipse_semimajor_axis_value_l371_371996

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371996


namespace tangent_line_equations_l371_371858

theorem tangent_line_equations (k b : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x, l x = k * x + b) ∧
    (∃ x₁, x₁^2 = k * x₁ + b) ∧ -- Tangency condition with C1: y = x²
    (∃ x₂, -(x₂ - 2)^2 = k * x₂ + b)) -- Tangency condition with C2: y = -(x-2)²
  → ((k = 0 ∧ b = 0) ∨ (k = 4 ∧ b = -4)) := sorry

end tangent_line_equations_l371_371858


namespace simplify_fraction_l371_371258

theorem simplify_fraction :
  (4 / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27)) = (Real.sqrt 3 / 12) := 
by
  -- Proof goes here
  sorry

end simplify_fraction_l371_371258


namespace sum_of_c_n_l371_371061

-- Define the sequence {b_n}
def b : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * b n + 3

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := (a n) / (b n + 3)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => c i)

-- Theorem to prove
theorem sum_of_c_n : ∀ (n : ℕ), T n = (3 / 2 : ℚ) - ((2 * n + 3) / 2^(n + 1)) :=
by
  sorry

end sum_of_c_n_l371_371061


namespace limit_n_b_n_l371_371800

noncomputable def M (x : ℝ) : ℝ := x - x^3 / 3

noncomputable def b_n (n : ℕ) : ℝ :=
  let rec iterate n x :=
    match n with
    | 0     => x
    | n + 1 => iterate n (M x)
  iterate n (20 / n)

theorem limit_n_b_n : 
  tendsto (λ n : ℕ, n * b_n n) at_top (𝓝 3) := 
sorry

end limit_n_b_n_l371_371800


namespace unique_positive_a_for_one_solution_l371_371827

theorem unique_positive_a_for_one_solution :
  ∃ (d : ℝ), d ≠ 0 ∧ (∀ a : ℝ, a > 0 → (∀ x : ℝ, x^2 + (a + 1/a) * x + d = 0 ↔ x^2 + (a + 1/a) * x + d = 0)) ∧ d = 1 := 
by
  sorry

end unique_positive_a_for_one_solution_l371_371827


namespace find_angle_C_find_sin_B_l371_371189

-- Define triangle and given conditions
variable {A B C: ℝ} {a b c: ℝ}
axiom triangle (A B C : ℝ) (a b c : ℝ) : a > 0 ∧ b > 0 ∧ c > 0

-- Given condition: cos B = (a / c) - (b / (2 * c))
axiom cos_B_given (A B C : ℝ) (a b c : ℝ) : triangle A B C a b c → cos B = (a / c) - (b / (2 * c))

-- Question 1: Prove C = pi / 3
theorem find_angle_C (A B C : ℝ) (a b c : ℝ) [h : triangle A B C a b c] : 
  cos B = (a / c) - (b / (2 * c)) → C = π / 3 :=
by 
  -- Proof goes here 
  sorry

-- Additional condition for part 2
axiom c_eq_2a (A B C : ℝ) (a b c : ℝ) : c = 2 * a

-- Question 2: Prove sin B = (sqrt 3 + sqrt 39) / 8
theorem find_sin_B (A B C : ℝ) (a b c : ℝ) [h : triangle A B C a b c] : 
  cos B = (a / c) - (b / (2 * c)) → c = 2 * a → 
  sin B = (Real.sqrt 3 + Real.sqrt 39) / 8 :=
by 
  -- Proof goes here 
  sorry

end find_angle_C_find_sin_B_l371_371189


namespace lim_n_b_n_l371_371801

noncomputable def M (x : ℝ) : ℝ := x - (x^3) / 3

def iter_M (k : ℕ) (x : ℝ) : ℝ := Nat.recOn k x (fun _ y => M y)

def b_n (n : ℕ) : ℝ := iter_M n (20 / (n : ℝ))

open Filter

theorem lim_n_b_n : tendsto (fun (n : ℕ) => n * b_n n) atTop (𝓝 (60 / 61)) :=
sorry

end lim_n_b_n_l371_371801


namespace inequality_proof_l371_371985

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a / (a + 2 * b)^(1/3) + b / (b + 2 * c)^(1/3) + c / (c + 2 * a)^(1/3)) ≥ 1 := 
by
  sorry

end inequality_proof_l371_371985


namespace f_properties_l371_371497

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then 2^x + x 
else if x = 0 then 0 
else -2^(-x) + x

theorem f_properties : 
  (∀ x : ℝ, x > 0 → f(x) = 2^x + x) ∧ 
  (∀ x : ℝ, x < 0 → f(x) = -2^(-x) + x) ∧ 
  f(0) = 0 ∧ 
  (∀ x : ℝ, f(-x) = -f(x)) :=
by
  sorry

end f_properties_l371_371497


namespace z_is_200_percent_of_x_l371_371536

theorem z_is_200_percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : y = 0.75 * x) :
  z = 2 * x :=
sorry

end z_is_200_percent_of_x_l371_371536


namespace greatest_possible_difference_l371_371792

theorem greatest_possible_difference (x y : ℚ) (h1 : 3 < x) (h2 : x < (3/2)^3) (h3 : (3/2)^3 < y) (h4 : y < 7) :
  ∃ (n : ℕ), (n = (y.to_int - x.to_int)) ∧ n = 2 :=
by {
  sorry
}

end greatest_possible_difference_l371_371792


namespace weightlifter_total_weight_l371_371403

theorem weightlifter_total_weight (h : 10 + 10 = 20) : 
  let total_weight := 10 + 10 in 
  total_weight = 20 := 
by 
  simp [h]
  sorry

end weightlifter_total_weight_l371_371403


namespace trajectory_of_midpoint_is_ellipse_l371_371092

theorem trajectory_of_midpoint_is_ellipse
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c^2 = a^2 - b^2)
  (M : ℝ → ℝ × ℝ)
  (hM : ∀ θ : ℝ, M θ = (a * Real.cos θ, b * Real.sin θ))
  (F1 : ℝ × ℝ)
  (hF1 : F1 = (-c, 0)) :
  ∃ P : ℝ → ℝ × ℝ, ∀ θ : ℝ, P θ = ((a * Real.cos θ - c) / 2, (b * Real.sin θ) / 2) ∧
                     ∀ x y : ℝ, (∃ θ : ℝ, P θ = (x, y)) ↔ ((2 * x + c) / a)^2 + (2 * y / b)^2 = 1 :=
begin
  sorry
end

end trajectory_of_midpoint_is_ellipse_l371_371092


namespace inequality_div_two_l371_371846

theorem inequality_div_two (a b : ℝ) (h : a > b) : (a / 2) > (b / 2) :=
sorry

end inequality_div_two_l371_371846


namespace distinct_b_l371_371621

-- Define the problem conditions and the question
theorem distinct_b (a : Fin 100 → ℕ) (h_unique : Function.Injective a) :
  let b := λ (i : Fin 100), a i + Nat.gcd (Finset.erase (Finset.univ) i).toList.sum (Finset.fold op ⊥[⊥] (Finset.erase (Finset.univ) i)) in
  (Finset.univ.image b).card = 99 := 
by
  sorry

end distinct_b_l371_371621


namespace power_division_l371_371331

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l371_371331


namespace reception_time_l371_371412

-- Definitions of conditions
def noon : ℕ := 12 * 60 -- define noon in minutes
def rabbit_walk_speed (v : ℕ) : Prop := v > 0
def rabbit_run_speed (v : ℕ) : Prop := 2 * v > 0
def distance (D : ℕ) : Prop := D > 0
def delay (minutes : ℕ) : Prop := minutes = 10

-- Definition of the problem
theorem reception_time (v D : ℕ) (h_v : rabbit_walk_speed v) (h_D : distance D) (h_delay : delay 10) :
  noon + (D / v) * 2 / 3 = 12 * 60 + 40 :=
by sorry

end reception_time_l371_371412


namespace distinct_new_cards_after_trades_l371_371617

def total_cards := 750
def third (n : ℕ) := n / 3
def fifth (n : ℕ) := n / 5
def half (n : ℕ) := n / 2

theorem distinct_new_cards_after_trades :
  let duplicate_cards := third total_cards in
  let new_cards_from_josh := fifth duplicate_cards in
  let remaining_new_cards_after_alex := new_cards_from_josh - third new_cards_from_josh in
  let final_new_cards := remaining_new_cards_after_alex - half remaining_new_cards_after_alex in
  final_new_cards = 17 :=
by
  sorry

end distinct_new_cards_after_trades_l371_371617


namespace standard_deviation_of_distribution_l371_371264

theorem standard_deviation_of_distribution (μ σ : ℝ) 
    (h₁ : μ = 15) (h₂ : μ - 2 * σ = 12) : σ = 1.5 := by
  sorry

end standard_deviation_of_distribution_l371_371264


namespace evaluate_f_f_neg2_l371_371274

def f (x : ℝ) : ℝ :=
  if x < 0 then (1 / 2) ^ x else x + 1

theorem evaluate_f_f_neg2 : f (f (-2)) = 5 := by
  sorry

end evaluate_f_f_neg2_l371_371274


namespace partI_partII_l371_371512

noncomputable def f (x : ℝ) := (sqrt 3 / 2) * sin (2 * x) - cos x ^ 2 - 1 / 2

theorem partI : ∀ k ∈ ℤ, interval_increasing f (Icc (-π/6 + k * π) (π/3 + k * π)) := 
sorry

theorem partII (A B C : ℝ) (a b c : ℝ) (hC : c = sqrt 3)
 (hfC : f C = 0) 
 (hCollinear : ∃ k : ℝ, (1, sin A) = k • (2, sin B)) :
 a = 1 ∧ b = 2 :=
begin
  sorry
end

end partI_partII_l371_371512


namespace no_integer_exists_with_square_ending_l371_371202

noncomputable def ends_with (x : ℕ) (lst : List ℕ) : Prop :=
  x % 10 ^ lst.length = lst.foldr (λ d acc, d + acc * 10) 0

theorem no_integer_exists_with_square_ending (a : List ℕ) (b : ℕ) (hb : b ∈ [0, 1, 4, 5, 6, 9]) :
  ¬ ∃ k : ℕ, ends_with (k * k) (a ++ [b]) :=
begin
  sorry
end

end no_integer_exists_with_square_ending_l371_371202


namespace seating_arrangements_l371_371566

theorem seating_arrangements :
  let cubs := 3
  let red_sox := 3
  let yankees := 2
  let dodgers := 4
  let total_athletes := cubs + red_sox + yankees + dodgers
  total_athletes = 12 →
  (∃! arrangements : ℕ, arrangements = (factorial 4) * (factorial cubs) * (factorial red_sox) * (factorial yankees) * (factorial dodgers) ∧ arrangements = 41472) := 
by {
  intros,
  sorry
}

end seating_arrangements_l371_371566


namespace largest_prime_factor_20_15_10_l371_371459

open Nat

theorem largest_prime_factor_20_15_10 : 
  ∃ p : ℕ, Prime p ∧ p = 1787 ∧ ∀ q : ℕ, Prime q → (q ∣ (20^4 + 15^3 - 10^5)) → q ≤ p := 
by
  sorry

end largest_prime_factor_20_15_10_l371_371459


namespace pascals_triangle_30_rows_l371_371901

theorem pascals_triangle_30_rows : 
  ∑ n in range 30, (n + 1) = 465 := 
by
  sorry

end pascals_triangle_30_rows_l371_371901


namespace find_x_l371_371922

theorem find_x (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 :=
sorry

end find_x_l371_371922


namespace xy_value_l371_371288

theorem xy_value : ∃ (x y : ℝ), (x - y = 2) ∧ (x + y - 1 = 3 * x + 2 * y - 4) ∧ (x * y = -5 / 9) :=
by
  -- Variables and system of equations
  let x : ℝ := 5 / 3
  let y : ℝ := -1 / 3
  use x, y
  split
  -- Prove the first equation
  {
    linarith -- This will show the first equation holds
  }
  split
  -- Prove the second equation
  {
    linarith -- This will show the second equation holds
  }
  -- Prove the product x*y
  {
    field_simp
    linarith
  }

end xy_value_l371_371288


namespace find_k_parallel_find_k_perpendicular_l371_371472

noncomputable def veca : (ℝ × ℝ) := (1, 2)
noncomputable def vecb : (ℝ × ℝ) := (-3, 2)

def is_parallel (u v : (ℝ × ℝ)) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ u = (k * v.1, k * v.2)

def is_perpendicular (u v : (ℝ × ℝ)) : Prop := 
  u.1 * v.1 + u.2 * v.2 = 0

def calc_vector (k : ℝ) (a b : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (k * a.1 + b.1, k * a.2 + b.2)

theorem find_k_parallel : 
  ∃ k : ℝ, is_parallel (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

theorem find_k_perpendicular :
  ∃ k : ℝ, k = 25 / 3 ∧ is_perpendicular (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

end find_k_parallel_find_k_perpendicular_l371_371472


namespace inequality_proof_l371_371221

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) (h₀ : ∀ i, 0 < x i) (h₁ : ∑ i, x i = 1) :
  (∑ i, Real.sqrt (x i)) * (∑ i, 1 / Real.sqrt (1 + x i)) ≤ n^2 / Real.sqrt (n + 1) :=
by
  sorry

end inequality_proof_l371_371221


namespace no_geometric_progression_11_12_13_l371_371787

theorem no_geometric_progression_11_12_13 :
  ∀ (b1 : ℝ) (q : ℝ) (k l n : ℕ), 
  (b1 * q ^ (k - 1) = 11) → 
  (b1 * q ^ (l - 1) = 12) → 
  (b1 * q ^ (n - 1) = 13) → 
  False :=
by
  intros b1 q k l n hk hl hn
  sorry

end no_geometric_progression_11_12_13_l371_371787


namespace nth_term_closed_form_arithmetic_sequence_l371_371109

open Nat

noncomputable def S (n : ℕ) : ℕ := 3 * n^2 + 4 * n
noncomputable def a (n : ℕ) : ℕ := if h : n > 0 then S n - S (n-1) else S n

theorem nth_term_closed_form (n : ℕ) (h : n > 0) : a n = 6 * n + 1 :=
by
  sorry

theorem arithmetic_sequence (n : ℕ) (h : n > 1) : a n - a (n - 1) = 6 :=
by
  sorry

end nth_term_closed_form_arithmetic_sequence_l371_371109


namespace M_property_l371_371024

open Matrix

def cross_product (u v : Vector3 ℝ) : Vector3 ℝ :=
  ⟨u.y * v.z - u.z * v.y,
   u.z * v.x - u.x * v.z,
   u.x * v.y - u.y * v.x⟩

def M : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, -7, -4],
    ![7, 0, -3],
    ![-4, 3, 0]
  ]

theorem M_property (v : Vector3 ℝ) : 
  M.mul_vec v = cross_product ⟨3, -4, 7⟩ v :=
by 
  sorry

end M_property_l371_371024


namespace find_apex_angle_identical_cones_l371_371296

-- Define the identical cones and the conditions
variables (A : Point)
variables (r R : ℝ) (α : ℝ)
variables (O O1 O2 O3 : Point)

-- Constants used
constant sqrt3 : ℝ := real.sqrt 3
constant apex_angle_fourth_cone : ℝ := 2 * real.pi / 3
constant apex_angle_identical_cones : ℝ := 2 * real.arccot ((4 + sqrt3) / 3)

-- Conditions in the problem
axiom identical_cones_touching_each_other_externally : 
  ∀ (C1 C2 C3 : Cone), touching_externally C1 C2 ∧ touching_externally C2 C3 ∧ touching_externally C3 C1
axiom identical_cones_touching_fourth_cone_internally : 
  ∀ (C1 C2 C3 : Cone) (C4 : Cone), touching_internally C1 C4 ∧ touching_internally C2 C4 ∧ touching_internally C3 C4 ∧ C4.apex_angle = apex_angle_fourth_cone 

-- Declare that we are solving for the apex angle of the identical cones
theorem find_apex_angle_identical_cones : 
  ∀ (C1 C2 C3 : Cone), (C1.apex_angle = 2 * α) →
  (C2.apex_angle = 2 * α) →
  (C3.apex_angle = 2 * α) →
  (2 * real.arccot ((4 + sqrt3) / 3) = 2 * α) :=
begin
  sorry
end

end find_apex_angle_identical_cones_l371_371296


namespace part1_monotonic_intervals_part2_monotonic_intervals_l371_371889

noncomputable def f (a x : ℝ) := a * x^2 + (2 * a - 1) * x - Real.log x

theorem part1_monotonic_intervals (a : ℝ) (x : ℝ) (h : a = 1 / 2) (hx : x > 0) :
  ((0 < x ∧ x < 1 → deriv (λ x, f a x) x < 0) ∧
   (x > 1 → deriv (λ x, f a x) x > 0) ∧
   (x = 1 → f a x = 1 / 2)) := sorry

theorem part2_monotonic_intervals (a : ℝ) (x : ℝ) (hx : x > 0) :
  ((a ≤ 0 → deriv (λ x, f a x) x < 0) ∧
   (a > 0 → ((0 < x ∧ x < 1 / (2 * a) → deriv (λ x, f a x) x < 0) ∧
    (x > 1 / (2 * a) → deriv (λ x, f a x) x > 0)))) := sorry

end part1_monotonic_intervals_part2_monotonic_intervals_l371_371889


namespace locus_circle_l371_371527

-- Define the points A and B on the plane
variables (a k : ℝ)

-- Define the locus condition
def locus_of_points (x y : ℝ) : Prop :=
  let AM := (x + a) ^ 2 + y ^ 2 in
  let BM := (x - a) ^ 2 + y ^ 2 in
  (AM / BM) = k^2

-- Prove that the locus is a circle with the specified center and radius
theorem locus_circle (x y : ℝ) :
  locus_of_points a k x y ↔
  (x + a * (k^2 + 1) / (k^2 - 1))^2 + y^2 = (2 * k * a / |1 - k^2|)^2 :=
sorry

end locus_circle_l371_371527


namespace smallest_pencil_count_l371_371731

theorem smallest_pencil_count : ∃ x : ℕ, (∀ y : ℕ, y > 0 ∧ (∀ z : ℕ, (x % 18 = 0 ∧ x % 35 = 0) ↔ (y % 18 = 0 ∧ y % 35 = 0) → x ≤ y) ∧ (x = 630)) :=
by
  existsi 630
  split
  { intro y hy
    cases hy with hy_pos h
    rw [← and_comm] at h
    rw [← and_comm] at h.1
    exact ⟨hy_pos, h⟩ }
  { rfl }
  sorry

end smallest_pencil_count_l371_371731


namespace find_lambda_l371_371233

-- Given conditions
def quadratic_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

def conditions (f : ℝ → ℝ) (g : ℝ → ℝ) (a b c : ℝ) : Prop :=
  quadratic_symmetric_about_y_axis f ∧
  a + b = 1 ∧
  (∀ x, f f x = (f x)^2 + 1) ∧
  f x = a * x^2 + b * x + c ∧
  (∀ (x y : ℝ), (y = f x) → y^2 + 1 = (f y)^2 + 1)

-- Proof problem definitions
def g (x : ℝ) : ℝ := x^4 + 2 * x^2 + 2

def F (x : ℝ) (λ : ℝ) : ℝ := g x - λ * (x^2 + 1)

-- Required statement
theorem find_lambda : ∃ (λ : ℝ), (λ = 3 ∧ 
  ∀ x, x < -real.sqrt 2 / 2 → (F x λ).derivative < 0) ∧ 
  ( ∀ x, -real.sqrt 2 / 2 < x ∧ x < 0 → (F x λ).derivative > 0) :=
sorry

end find_lambda_l371_371233


namespace power_division_identity_l371_371309

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l371_371309


namespace scientific_notation_of_284000000_l371_371410

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end scientific_notation_of_284000000_l371_371410


namespace max_rectangle_area_l371_371697

variables {a b : ℝ}

theorem max_rectangle_area (h : 2 * a + 2 * b = 60) : a * b ≤ 225 :=
by 
  -- Proof to be filled in
  sorry

end max_rectangle_area_l371_371697


namespace circle_intersection_range_l371_371540

noncomputable def circleIntersectionRange (r : ℝ) : Prop :=
  1 < r ∧ r < 11

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) :
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x + 3)^2 + (y - 4)^2 = 36) ↔ circleIntersectionRange r :=
by
  sorry

end circle_intersection_range_l371_371540


namespace distance_between_lines_l371_371393

-- Define the conditions
variables (parallel_paths_distance crosswalk_length lines_length area : ℝ)
def parallelogram_base_1 := 20
def parallelogram_height_1 := parallel_paths_distance
def parallelogram_base_2 := lines_length

-- The given values
def parallel_paths_distance := 60
def crosswalk_length := 20
def lines_length := 75
def area := parallelogram_base_1 * parallelogram_height_1

-- The theorem to prove the distance between the lines is 16 feet
theorem distance_between_lines : (area / parallelogram_base_2) = 16 := by sorry

end distance_between_lines_l371_371393


namespace race_permutations_l371_371126

theorem race_permutations (r1 r2 r3 r4 : Type) [decidable_eq r1] [decidable_eq r2] [decidable_eq r3] [decidable_eq r4] :
  fintype.card (finset.univ : finset {l : list r1 | l ~ [r1, r2, r3, r4]}) = 24 :=
by
  sorry

end race_permutations_l371_371126


namespace parametric_curve_length_is_2pi_l371_371022

def parametricCurveLength : ℝ :=
  let x := λ (t : ℝ), 2 * Real.sin t
  let y := λ (t : ℝ), 2 * Real.cos t
  IntervalIntegral (λ t : ℝ, Real.sqrt ((Real.deriv x t) ^ 2 + (Real.deriv y t) ^ 2)) 0 π

theorem parametric_curve_length_is_2pi :
  parametricCurveLength = 2 * Real.pi := by
  sorry

end parametric_curve_length_is_2pi_l371_371022


namespace not_possible_to_fill_board_with_2012_l371_371352

theorem not_possible_to_fill_board_with_2012 :
  let board := matrix (fin 5) (fin 5) ℕ in
  (∀ i j : fin 5, board i j = 0) →
  (∀ steps : list (fin 5 × fin 5), 
    let new_board := steps.foldl (λ b (x, y), fun i j => 
      if (i = x ∧ (j = y ∨ j = y - 1 ∨ j = y + 1)) ∨ 
         (j = y ∧ (i = x ∨ i = x - 1 ∨ i = x + 1)) 
      then b i j + 1 else b i j) board in
      ∃ i j : fin 5, new_board i j ≠ 2012) :=
begin
  sorry
end

end not_possible_to_fill_board_with_2012_l371_371352


namespace real_roots_of_quadratic_l371_371703

theorem real_roots_of_quadratic : ∃ x1 x2 : ℝ, (x^2 - 5*x + 6 = 0) ∧ (x1 = 2 ∧ x2 = 3) :=
begin
  sorry
end

end real_roots_of_quadratic_l371_371703


namespace pow_div_eq_l371_371325

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l371_371325


namespace even_sum_probability_l371_371118

variable (A : Finset ℕ := {11, 19, 44, 55, 72, 81})
variable (B : Finset ℕ := {1, 13, 24, 37, 46})

theorem even_sum_probability : 
  let even_count (s : Finset ℕ) := s.filter (λ x => x % 2 = 0).card
  let odd_count (s : Finset ℕ) := s.filter (λ x => x % 2 = 1).card
  even_count A = 2 → odd_count A = 4 →
  even_count B = 2 → odd_count B = 3 →
  (even_count A * even_count B + odd_count A * odd_count B) / (A.card * B.card) = (8 / 15 : ℚ) :=
by
  intro even_count odd_count h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end even_sum_probability_l371_371118


namespace general_term_l371_371524

noncomputable def a : ℕ → ℚ
| 0     := 0  -- Since we start at n = 1
| 1     := 1
| (n+1) := a n + 1/(n * (n+1))

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = (2 * n - 1) / n := by
  sorry

end general_term_l371_371524


namespace points_on_line_distance_l371_371824

noncomputable def point_on_line_distance (t : ℝ) : ℝ × ℝ :=
  let x : ℝ := 1 - real.sqrt 2 * t
  let y : ℝ := 2 + real.sqrt 2 * t
  (x, y)

theorem points_on_line_distance (x y t : ℝ) :
  (x = 1 - real.sqrt 2 * t) →
  (y = 2 + real.sqrt 2 * t) →
  (real.sqrt ((x - 1) ^ 2 + (y - 2) ^ 2) = 4 * real.sqrt 2) →
  ((x = -3 ∧ y = 6) ∨ (x = 5 ∧ y = -2)) :=
by
  sorry

end points_on_line_distance_l371_371824


namespace number_of_points_C_l371_371560

open Real EuclideanGeometry Mathlib

-- Definitions
def is_point_on_plane (A B C : Point ℝ) : Prop :=
  ∃ (x : ℝ) (y : ℝ), A = ⟨0, 0⟩ ∧ B = ⟨12, 0⟩ ∧ C = ⟨x, y⟩

def perimeter_of_triangle (A B C : Point ℝ) (perimeter : ℝ) : Prop :=
  dist A B + dist A C + dist B C = perimeter

def area_of_triangle (A B C : Point ℝ) (area : ℝ) : Prop :=
  abs (1 / 2 * SignedArea (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)) = area

-- Theorem to prove
theorem number_of_points_C (A B : Point ℝ) :
  is_point_on_plane A B C → 
  perimeter_of_triangle A B C 60 →
  area_of_triangle A B C 120 → 
  ∃ P : Finset (Point ℝ), P.card = 4 :=
sorry

end number_of_points_C_l371_371560


namespace smallest_odd_abundant_number_l371_371440

-- Define a function to calculate the sum of proper divisors
def sum_of_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d : ℕ, d < n) (Finset.divisors n)).sum

-- Define a predicate for an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define a predicate for an abundant number
def is_abundant (n : ℕ) : Prop := sum_of_proper_divisors n > n

-- Prove that the smallest odd abundant number is 135
theorem smallest_odd_abundant_number : ∃ n, is_odd n ∧ is_abundant n ∧ (∀ m, is_odd m ∧ is_abundant m → n ≤ m) ∧ n = 135 :=
by
  -- We assume this theorem and leave the detailed proof for another time
  sorry

end smallest_odd_abundant_number_l371_371440


namespace num_zeros_in_decimal_representation_l371_371899

theorem num_zeros_in_decimal_representation :
  let denom := 2^3 * 5^10
  let frac := (1 : ℚ) / denom
  ∃ n : ℕ, n = 7 ∧ (∃ (a : ℕ) (b : ℕ), frac = a / 10^b ∧ ∃ (k : ℕ), b = n + k + 3) :=
sorry

end num_zeros_in_decimal_representation_l371_371899


namespace race_permutations_l371_371124

theorem race_permutations (r1 r2 r3 r4 : Type) [decidable_eq r1] [decidable_eq r2] [decidable_eq r3] [decidable_eq r4] :
  fintype.card (finset.univ : finset {l : list r1 | l ~ [r1, r2, r3, r4]}) = 24 :=
by
  sorry

end race_permutations_l371_371124


namespace part1_monotonic_intervals_and_extreme_value_part2_general_monotonicity_l371_371892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a - 1) * x - Real.log x

theorem part1_monotonic_intervals_and_extreme_value (x : ℝ) (h : x > 0) :
  ∀ x, f (1/2) x = (1/2) * x^2 - Real.log x → 
  ((0 < x ∧ x < 1) → (differentiable_at ℝ (f (1/2)) x ∧ deriv (f (1/2)) x < 0)) ∧
  ((1 < x) → (differentiable_at ℝ (f (1/2)) x ∧ deriv (f (1/2)) x > 0)) ∧
  (f (1/2) 1 = 1/2) := sorry

theorem part2_general_monotonicity (a x : ℝ) (h : x > 0) :
  ∀ x, f a x = a * x^2 + (2 * a - 1) * x - Real.log x → 
  ((a ≤ 0) → (∀ x > 0, differentiable_at ℝ (f a) x ∧ deriv (f a) x < 0)) ∧
  ((a > 0) → ((0 < x ∧ x < 1 / (2 * a)) → (differentiable_at ℝ (f a) x ∧ deriv (f a) x < 0)) ∧
                ((x > 1 / (2 * a)) → (differentiable_at ℝ (f a) x ∧ deriv (f a) x > 0))) := sorry

end part1_monotonic_intervals_and_extreme_value_part2_general_monotonicity_l371_371892


namespace sum_and_product_of_arithmetic_sequence_l371_371063

theorem sum_and_product_of_arithmetic_sequence
    (a1 a2 a3 : ℤ)
    (d : ℤ)
    (n : ℕ)
    (h1 : a2 = a1 + d)
    (h2 : a3 = a1 + 2 * d)
    (sum_cond : a1 + a2 + a3 = -3)
    (prod_cond : a1 * a2 * a3 = 8)
    (geo_cond : a2^2 = a1 * a3) : 
    Σ_n : ℕ → ℚ, S_n n = if n = 1 then 4 else (3 * n^2 - 11 * n + 20) / 2 := by
  sorry

end sum_and_product_of_arithmetic_sequence_l371_371063


namespace work_together_l371_371371

theorem work_together (A_days B_days : ℕ) (hA : A_days = 8) (hB : B_days = 4)
  (A_work : ℚ := 1 / A_days)
  (B_work : ℚ := 1 / B_days) :
  (A_work + B_work = 3 / 8) :=
by
  rw [hA, hB]
  sorry

end work_together_l371_371371


namespace intersection_A_B_l371_371493

def A : Set ℝ := { x | x > 2 }
def B : Set ℝ := { x | 2 ≤ 2^x ∧ 2^x ≤ 8 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := by
  sorry

end intersection_A_B_l371_371493


namespace problem_1_proof_problem_2_proof_problem_3_proof_l371_371925

noncomputable def problem_1_condition := 
  (a b c A B C : Real)
  (AB AC BA BC : ℝ) 
  (abc_triangle : ∀ a b c : Real, ∃ A B C : Real),
  (∀ (a b c: Real), abc_triangle a b c),
  (∀ (A B : Real), ∃ a b : Real, A ≠ B) →
  (dot_product_eq : ∀ (AB AC : ℝ), AB * AC = 1)

theorem problem_1_proof (a b c A B C : Real) 
  (abc_triangle : ∀ a b c : Real, ∃ A B C : Real) 
  (AB AC BA BC : ℝ)
  (dot_product_eq : ∀ (AB AC : ℝ), AB * AC = 1) 
  (h1 : ∀ (a b c : Real), abc_triangle a b c)
  (h2 : ∀ (A B : Real), ∃ a b : Real, A ≠ B) : 
  A = B :=
sorry

theorem problem_2_proof (a b c A B : Real) 
  (abc_triangle : ∀ a b c : Real, ∃ A B : Real)
  (AB AC : ℝ)
  (dot_product_eq : ∀ (AB AC : ℝ), AB * AC = 1) 
  (h1 : ∀ (x y : ℝ), ∃ c : ℝ, √c = 2) : 
  c = sqrt(2) :=
sorry

theorem problem_3_proof (a b c A B : Real) 
  (abc_triangle : ∀ a b c : Real, ∃ A B : Real)
  (area : ℝ)
  (h_script : ∀ (x y z : Real), abs(x + y) = sqrt(6)) 
  (h1 : ∀ (a b c : Real), abc_triangle a b c) 
  (h2 : ∀ (A B : Real), ∃ c : ℝ, sqrt(c) = 2) : 
  area = sqrt(3) / 2 :=
sorry

end problem_1_proof_problem_2_proof_problem_3_proof_l371_371925


namespace large_circle_diameter_l371_371254

theorem large_circle_diameter (r : ℝ) (R : ℝ) (s : ℝ) (inner_radius : ℝ) 
  (heptagon_side_length : ℝ) (heptagon_radius : ℝ) (n : ℝ) : 
  r = 2 →
  n = 7 →
  heptagon_side_length = 2 * r →
  heptagon_radius = heptagon_side_length / (2 * real.sin (π / n)) →
  inner_radius = heptagon_radius →
  R = inner_radius + r →
  s = 2 * R →
  s = 2 * (heptagon_radius + r) →
  s = (4 / (real.sin (π / 7))) + 4 :=
by
  sorry

end large_circle_diameter_l371_371254


namespace opposite_sides_range_l371_371081

theorem opposite_sides_range (a : ℝ) :
  (3 * (-3) - 2 * (-1) - a) * (3 * 4 - 2 * (-6) - a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  simp
  sorry

end opposite_sides_range_l371_371081


namespace find_n_l371_371302

theorem find_n (a : ℝ) (x : ℝ) (y : ℝ) (h1 : 0 < a) (h2 : a * x + 0.6 * a * y = 5 / 10)
(h3 : 1.6 * a * x + 1.2 * a * y = 1 - 1 / 10) : 
∃ n : ℕ, n = 10 :=
by
  sorry

end find_n_l371_371302


namespace problem1_solution_problem2_solution_l371_371650

noncomputable def problem1 : ℝ :=
  2 * real.sqrt 3 * 31.5 * 612

theorem problem1_solution : problem1 = 6 :=
sorry

noncomputable def problem2 : ℝ :=
  (real.log 3 / real.log 4 - real.log 3 / real.log 8)
  * (real.log 2 / real.log 3 + real.log 2 / real.log 9)

theorem problem2_solution : problem2 = 0 :=
sorry

end problem1_solution_problem2_solution_l371_371650


namespace determine_h_l371_371005

noncomputable def h (x : ℝ) : ℝ :=
  -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3

theorem determine_h :
  (12*x^4 + 4*x^3 - 2*x + 3 + h x = 6*x^3 + 8*x^2 - 10*x + 6) ↔
  (h x = -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3) :=
by 
  sorry

end determine_h_l371_371005


namespace find_f_8_5_l371_371218

-- Conditions as definitions in Lean
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def segment_function (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- The main theorem to prove
theorem find_f_8_5 (f : ℝ → ℝ) (h1 : even_function f) (h2 : periodic_function f 3) (h3 : segment_function f)
: f 8.5 = 1.5 :=
sorry

end find_f_8_5_l371_371218


namespace blue_cards_in_box_l371_371563

theorem blue_cards_in_box (x : ℕ) (h : 0.6 = (x : ℝ) / (x + 8)) : x = 12 :=
sorry

end blue_cards_in_box_l371_371563


namespace race_order_count_l371_371122

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end race_order_count_l371_371122


namespace volume_is_correct_min_edge_length_is_correct_l371_371186

-- Define the properties of the regular pyramid
def regularPyramid (S M N P Q : Point) (H F E : Point) : Prop :=
  is_apex S M N P Q ∧
  midpoint H M N ∧
  midpoint F N P ∧
  lies_on E S H ∧
  dist S H = 3 ∧
  dist S E = 9 ∧  -- Given \( S E = \frac{9}{1} \)
  dist_from_point_to_line S E F = sqrt 5

-- Define the properties of the sphere and the tetrahedron
def sphereAndTetrahedra (S C D A B : Point) (E F : Point) : Prop :=
  sphere_radius S 1 ∧
  lies_on_line C D E F ∧
  touches_line A B S ∧
  edge_length_of_regular_tetrahedra A B C D ≤ min_edge_length

-- Define the volume of the pyramid
def volume_of_pyramid {S M N P Q : Point} : ℝ :=
  4 * sqrt 6

-- Define the minimum edge length of the tetrahedra
def min_edge_length {A B C D : Point} : ℝ :=
  (2 * sqrt 2 * (sqrt 7 - 1)) / 3

-- Prove that the volume of the pyramid is as calculated
theorem volume_is_correct {S M N P Q H F E : Point} 
  (h : regularPyramid S M N P Q H F E) : 
  volume_of_pyramid = 4 * sqrt 6 :=
by
  sorry

-- Prove that the minimum edge length of the tetrahedra is correct
theorem min_edge_length_is_correct {S C D A B E F : Point} 
  (h : sphereAndTetrahedra S C D A B E F) : 
  min_edge_length = (2 * sqrt 2 * (sqrt 7 - 1)) / 3 :=
by
  sorry

end volume_is_correct_min_edge_length_is_correct_l371_371186


namespace sum_B_k_eq_l371_371287

def A (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | (k + 1) => A k + 2 * k

def B (n : ℕ) : ℚ := 
  let soft_arccot (x: ℚ): ℚ := Real.arctan (1 / x)
  let cot (x: ℚ): ℚ := 1 / Real.tan x
  let arccot_A_n := soft_arccot (A n : ℚ)
  let arccot_A_succ := soft_arccot (A (n + 1) : ℚ)
  let arccot_A_succ_succ := soft_arccot (A (n + 2) : ℚ)
  cot (arccot_A_n + arccot_A_succ + arccot_A_succ_succ)

theorem sum_B_k_eq (n : ℕ) : 
  (∑ k in Finset.range n.succ, B k) = n * (n^2 + 3 * n - 1) / 9 := 
sorry

end sum_B_k_eq_l371_371287


namespace geometric_condition_l371_371289

def sum_first_n_terms (n : ℕ) (c : ℝ) : ℝ := 3^n - c

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a 1 * (a 1 ^ n)

theorem geometric_condition (a : ℕ → ℝ) (c : ℝ) (h : ∀ n, (∑ i in finset.range n, a i) = 3^n - c) :
  c = 1 ↔ is_geometric_sequence a :=
sorry

end geometric_condition_l371_371289


namespace XYZ_triangle_XY_length_l371_371826

noncomputable def length_XY (XY ZY : ℝ) (angle_XPY : ℝ) : Prop :=
  angle_XPY = 60 ∧ ZY = 10 → XY = 5

theorem XYZ_triangle_XY_length :
  ∃ (XY ZY : ℝ),
  length_XY XY ZY 60 :=
begin
  use [5, 10],
  intro h,
  simp at h,
  sorry  -- Proof omitted here
end

end XYZ_triangle_XY_length_l371_371826


namespace motorcyclist_average_speed_l371_371760

theorem motorcyclist_average_speed :
  ∀ (t : ℝ), 120 / t = 60 * 3 → 
  3 * t / 4 = 45 :=
by
  sorry

end motorcyclist_average_speed_l371_371760


namespace ratio_trapezoid_to_rectangle_area_l371_371556

theorem ratio_trapezoid_to_rectangle_area
  (O : Point) (b h : ℝ)
  (CD EF : Line)
  (G H J : Point)
  (collinear : Collinear {O, G, H, J})
  (vertical_translation : TranslatedVertical CD EF)
  (half_distance : distance J H = distance H G = 1/2 * distance CD EF) :
  let T := area (trapezoid CDFE)
  let R := area (rectangle ELHF)
  in (T / R) = (1/2 + sqrt 2 / 2) :=
sorry

end ratio_trapezoid_to_rectangle_area_l371_371556


namespace zero_ends_of_double_factorial_thirty_l371_371538

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n+1)   := (n + 1) * factorial n

-- Define the count of trailing zeros
def trailing_zeros (n: ℕ) : ℕ :=
  let factors_of x p := if x % p = 0 then factors_of (x / p) p + 1 else 0
  List.sum (List.map (λ x, factors_of x 5) (List.range' 1 n))

-- Problem statement
theorem zero_ends_of_double_factorial_thirty :
  trailing_zeros (2 * factorial 30) = 7 := 
sorry

end zero_ends_of_double_factorial_thirty_l371_371538


namespace triangle_DEF_perimeter_l371_371926

noncomputable def DE := 15
noncomputable def EF := 15
noncomputable def DF := 15 * Real.sqrt 2

theorem triangle_DEF_perimeter
  (DE := 15)
  (EF := 15)
  (DF := 15 * Real.sqrt 2)
  (angle_DEF := 90) :
  DE + EF + DF = 30 + 15 * Real.sqrt 2 := by
  sorry

end triangle_DEF_perimeter_l371_371926


namespace incorrect_statement_about_g_l371_371609

def g (x : ℝ) : ℝ := (x - 4) / (x + 3)

theorem incorrect_statement_about_g :
  ¬ (g (-4) = -2) := by
  -- Proof should be provided here
  sorry

end incorrect_statement_about_g_l371_371609


namespace minimum_value_l371_371187

noncomputable def point := (ℝ × ℝ × ℝ) -- assuming three-dimensional points

/-- A right triangular prism ABC-A1B1C1 where AB = 1, BC = sqrt(3), angle ABC = 90 degrees -/
structure RightTriangularPrism :=
  (A B C A1 B1 C1 : point)
  (AB_eq_1 : dist A B = 1)
  (BC_eq_sqrt_3 : dist B C = Real.sqrt 3)
  (C1C_eq_sqrt_3 : dist C1 C = Real.sqrt 3)
  (angle_ABC_eq_90 : ∠ A B C = 90)

def minimize_expression (prism : RightTriangularPrism) (P : point) : ℝ :=
  dist prism.A1 P + (1 / 2) * dist P prism.C

theorem minimum_value (prism : RightTriangularPrism) (P : point) :
  ∃ P, minimize_expression prism P = 5/2 :=
sorry

end minimum_value_l371_371187


namespace david_does_50_percent_l371_371357

/-- Proof that David does 50% of the job given that John and David together can finish the job in 1 hour and John can do it alone in 2 hours. -/
theorem david_does_50_percent (john_rate david_rate : ℝ) :
  (john_rate = 1 / 2) → (john_rate + david_rate = 1) → (david_rate / (john_rate + david_rate) * 100 = 50) :=
by
  intros h_john_rate h_combined_rate
  rw [h_john_rate, h_combined_rate]
  sorry

end david_does_50_percent_l371_371357


namespace line_rect_eq_curve_rect_eq_slope_of_line_l371_371184

-- Define the parametric line l
def parametric_line (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

-- Define the polar equation of curve C
def polar_curve (θ : ℝ) : ℝ :=
  (4 / (1 + 3 * Real.sin θ ^ 2)) ^ (1 / 2)

-- Prove that the rectangular coordinate equation of line l is y = tan(α)x
theorem line_rect_eq (t α : ℝ) : (parametric_line t α).snd = (Real.tan α) * (parametric_line t α).fst :=
  sorry

-- Prove that the rectangular coordinate equation of curve C is x^2 + 4y^2 = 4
theorem curve_rect_eq (x y : ℝ) (θ : ℝ) : x^2 + 4 * y^2 = 4 :=
  sorry

-- Prove that if there are exactly 2 points on curve C that are sqrt(2) units away from line l, the slope k = ±1
theorem slope_of_line (α : ℝ) : ∃ k : ℝ, k = ±1 :=
  sorry

end line_rect_eq_curve_rect_eq_slope_of_line_l371_371184


namespace unit_vector_perpendicular_to_a_l371_371050

theorem unit_vector_perpendicular_to_a :
  ∃ (m n : ℝ), 2 * m + n = 0 ∧ m^2 + n^2 = 1 ∧ m = sqrt 5 / 5 ∧ n = -2 * sqrt 5 / 5 :=
by
  sorry

end unit_vector_perpendicular_to_a_l371_371050


namespace APNQ_is_parallelogram_l371_371226

open Real EuclideanGeometry

variables {A B C O P Q N : Point}
variables (triangle : EuclideanGeometry.triangle A B C)

axiom not_right_angle_BAC : ∠BAC ≠ π / 2
axiom O_is_circumcenter : EuclideanGeometry.is_circumcenter O A B C
axiom Gamma_is_circumcircle_BOC : EuclideanGeometry.is_circumcircle O (EuclideanGeometry.triangle B O C)
axiom P_on_AB : EuclideanGeometry.on_circle_interior P Gamma_is_circumcircle_BOC ∧ P ≠ B
axiom Q_on_AC : EuclideanGeometry.on_circle_interior Q Gamma_is_circumcircle_BOC ∧ Q ≠ C
axiom ON_is_diameter : EuclideanGeometry.is_diameter O N Gamma_is_circumcircle_BOC

theorem APNQ_is_parallelogram :
  EuclideanGeometry.is_parallelogram (quadrilateral A P N Q) :=
sorry

end APNQ_is_parallelogram_l371_371226


namespace skimmed_milk_addition_l371_371417

theorem skimmed_milk_addition (x : ℚ) 
    (whole_milk : ℚ := 1) 
    (initial_cream_percentage : ℚ := 0.05) 
    (target_cream_percentage : ℚ := 0.04) :
    (initial_cream_percentage = 0.05 ∧ target_cream_percentage = 0.04 ∧ whole_milk = 1)
    → \( \frac{initial_cream_percentage}{whole_milk + x} = target_cream_percentage \)
    → x = \frac{1}{4} :=
by
 sorry

end skimmed_milk_addition_l371_371417


namespace intersection_is_empty_l371_371963

noncomputable theory

-- Definitions based on the conditions
def is_equilateral_triangle (ABC : Triangle) : Prop :=
  ABC.a = ABC.b ∧ ABC.b = ABC.c

def parallel_to_BC (line : Line) (BC : Line) : Prop :=
  -- Assume a definition that checks if a line is parallel to BC
  sorry

def divides_equal_area (lines : Set Line) (ABC : Triangle) (n : ℕ) : Prop :=
  ∃ ps : List Polygon, length ps = n ∧ (∀ p ∈ ps, area(ABC) / n = area(p)) ∧ divides_by_parallel_lines ps lines

def divides_equal_perimeter (lines : Set Line) (ABC : Triangle) (n : ℕ) : Prop :=
  ∃ ps : List Polygon, length ps = n ∧ (∀ p ∈ ps, perimeter(ABC) / n = perimeter(p)) ∧ divides_by_parallel_lines ps lines

def divides_by_parallel_lines (polygons : List Polygon) (lines : Set Line) : Prop :=
  -- Assume a definition that checks if the set of lines divides the triangle
  sorry

-- Intersection problem statement
theorem intersection_is_empty (ABC : Triangle) (n : ℕ) (linesA linesP : Set Line) :
  is_equilateral_triangle ABC →
  n ≥ 2 →
  divides_equal_area linesA ABC n →
  divides_equal_perimeter linesP ABC n →
  disjoint linesA linesP :=
by {
  sorry
}

end intersection_is_empty_l371_371963


namespace ellipse_semimajor_axis_value_l371_371991

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l371_371991


namespace find_xy_l371_371637

noncomputable theory
open_locale classical

variables (x y : ℝ)
variables (h1 : x > 0) (h2 : y > 0)
variables (h3 : x^2 + y^2 = 2)
variables (h4 : x^4 + y^4 = 7 / 4)

theorem find_xy : x * y = 3 * real.sqrt 2 / 4 :=
by sorry

end find_xy_l371_371637


namespace piglet_eats_more_than_half_of_chocolate_bar_l371_371245

/-!
Which of the friends will be able to eat more than half of the whole chocolate bar
regardless of the actions of the other?
Participants: Piglet, Winnie-the-Pooh
Conditions:
- 7 by 7 chocolate bar.
- Piglet takes 1x1 pieces.
- Winnie-the-Pooh takes 2x1 or 1x2 pieces.
- Piglet goes first.
- If there are no 2x1 or 1x2 pieces left before Winnie's turn, all remaining chocolate is given to Piglet.

Prove:
- Piglet will always eat more than half the chocolate bar.
-/

noncomputable def chocolate_bar_game : Prop :=
  let white_squares := 25 in
  let black_squares := 24 in
  sorry -- The detailed proof is omitted here.

theorem piglet_eats_more_than_half_of_chocolate_bar :
  chocolate_bar_game :=
sorry -- Proof omitted.

end piglet_eats_more_than_half_of_chocolate_bar_l371_371245


namespace point_on_inverse_proportion_graph_l371_371279

def inverse_proportion_function (x : ℝ) : ℝ := -4 / x

theorem point_on_inverse_proportion_graph : inverse_proportion_function (-2) = 2 :=
by
  -- Substitution and simplification can be left for the proof
  sorry

end point_on_inverse_proportion_graph_l371_371279


namespace solve_for_x_l371_371157

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 6 = 13) : x = 35.5 :=
by
  -- Proof steps would go here
  sorry

end solve_for_x_l371_371157


namespace true_proposition_l371_371665

-- Definitions based on the conditions in a)
def statement1 := ∀ (R : Type) [rect : rectangle R], ∃ (P : Type) [par : parallelogram P], P = connect_midpoints R
def statement2 := ∀ (Q : Type) [quad : quadrilateral Q], perpendicular_diagonals Q ∧ equal_diagonals Q → square Q
def statement3 := ∀ (Q : Type) [quad : quadrilateral Q], ∃ (P : Type) [par : parallelogram P], parallel_sides Q ∧ congruent_angles Q → parallelogram Q
def statement4 := ∀ (Q : Type) [quad : quadrilateral Q], ∃ (P : Type) [par : parallelogram P], parallel_sides Q ∧ congruent_sides Q → parallelogram Q

-- The theorem to be proved
theorem true_proposition : statement3 :=
by sorry

end true_proposition_l371_371665


namespace number_of_girls_l371_371561

theorem number_of_girls (boys girls : ℕ) (h1 : boys = 337) (h2 : girls = boys + 402) : girls = 739 := by
  sorry

end number_of_girls_l371_371561


namespace asymptotes_of_C2_l371_371845

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def C1 (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
noncomputable def C2 (x y : ℝ) : Prop := (y^2 / a^2 - x^2 / b^2 = 1)
noncomputable def ecc1 : ℝ := (Real.sqrt (a^2 - b^2)) / a
noncomputable def ecc2 : ℝ := (Real.sqrt (a^2 + b^2)) / a

theorem asymptotes_of_C2 :
  a > b → b > 0 → ecc1 * ecc2 = Real.sqrt 3 / 2 → by exact (∀ x y : ℝ, C2 x y → x = - Real.sqrt 2 * y ∨ x = Real.sqrt 2 * y) :=
sorry

end asymptotes_of_C2_l371_371845


namespace number_exceeds_by_25_l371_371354

theorem number_exceeds_by_25 {
  -- Let the number be denoted as 'x'
  (x : ℕ) (h: x = (3 / 8 : ℚ) * x + 25) : x = 40 := 
sorry

end number_exceeds_by_25_l371_371354


namespace find_four_numbers_l371_371859

theorem find_four_numbers 
    (a b c d : ℕ) 
    (h1 : b - a = c - b)  -- first three numbers form an arithmetic sequence
    (h2 : d / c = c / (b - a + b))  -- last three numbers form a geometric sequence
    (h3 : a + d = 16)  -- sum of first and last numbers is 16
    (h4 : b + (12 - b) = 12)  -- sum of the two middle numbers is 12
    : (a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16) :=
by
  -- Proof will be provided here
  sorry

end find_four_numbers_l371_371859


namespace bed_sheet_length_l371_371905

variables (length_piece : ℕ) (time_per_cut : ℕ) (total_time : ℕ)
variables (H1 : length_piece = 20) (H2 : time_per_cut = 5) (H3 : total_time = 245)

theorem bed_sheet_length : 
  let number_of_pieces := total_time / time_per_cut in
  let total_length_cm := number_of_pieces * length_piece in
  let total_length_m := total_length_cm / 100 in
  total_length_m = 9.8 :=
by
  sorry

end bed_sheet_length_l371_371905


namespace sharks_win_percentage_at_least_ninety_percent_l371_371263

theorem sharks_win_percentage_at_least_ninety_percent (N : ℕ) :
  let initial_games := 3
  let initial_shark_wins := 2
  let total_games := initial_games + N
  let total_shark_wins := initial_shark_wins + N
  total_shark_wins * 10 ≥ total_games * 9 ↔ N ≥ 7 :=
by
  intros
  sorry

end sharks_win_percentage_at_least_ninety_percent_l371_371263


namespace hours_to_destination_l371_371451

def num_people := 4
def water_per_person_per_hour := 1 / 2
def total_water_bottles_needed := 32

theorem hours_to_destination : 
  ∃ h : ℕ, (num_people * water_per_person_per_hour * 2 * h = total_water_bottles_needed) → h = 8 :=
by
  sorry

end hours_to_destination_l371_371451


namespace reciprocal_neg_three_l371_371285

theorem reciprocal_neg_three : ∃ r : ℚ, -3 * r = 1 ∧ r = -1/3 := 
by
  use -1/3
  split
  · exact eq.div_mul_cancel -3 (by norm_num)
  · rfl
# sorry -- this will skip the proof, that's why need additional imports if there are exceptions in Lean build environment.

end reciprocal_neg_three_l371_371285


namespace find_a_l371_371505

-- Define the given complex number z
def z (a : ℝ) : ℂ := (a + complex.i) / complex.i

-- Define the condition that the real and imaginary parts of z are equal
def parts_equal (a : ℝ) : Prop := (z a).re = (z a).im

-- State the theorem to prove that under the given conditions, a = -1
theorem find_a (a : ℝ) (h : parts_equal a) : a = -1 := 
by {
  sorry
}

end find_a_l371_371505


namespace ratio_diagonal_to_side_l371_371012

theorem ratio_diagonal_to_side (ABCDE : ConvexPentagon) 
  (h_parallel : ∀ diag side, parallel diag side) : 
  ∀ (diag side : ℝ), ratio diag side = 1 / 2 * (sqrt 5 + 1) :=
by
  sorry

end ratio_diagonal_to_side_l371_371012


namespace distinct_units_digits_of_cubes_l371_371140

theorem distinct_units_digits_of_cubes : 
  (∃ l : List ℕ, 
    (∀ n : ℕ, ∃ d ∈ l, n^3 % 10 = d) ∧ 
    l.nodup ∧ 
    l.length = 10) :=
by
  sorry

end distinct_units_digits_of_cubes_l371_371140


namespace solution_set_l371_371075

variable {ℝ : Type*} [LinearOrderedField ℝ]

-- Definition of odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : odd_function f)
variable (h_f3 : f 3 = 3)
variable (h_mono : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)

theorem solution_set :
  {x : ℝ | (x + 2) * f (x + 2) < 9} = {x : ℝ | x < 1} :=
sorry

end solution_set_l371_371075


namespace eight_pow_15_div_sixtyfour_pow_6_l371_371315

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l371_371315


namespace determine_proportionality_l371_371577

def neither_direct_nor_inverse_proportional (x y : ℝ) : Prop :=
  ∀ k : ℝ, (x / y ≠ k) ∧ (x * y ≠ k)

theorem determine_proportionality : 
  ∀ (x y : ℝ),
    (2 * x + y = 5 ∨ 2 * x + 3 * y = 15) →
    neither_direct_nor_inverse_proportional x y :=
begin
  intros x y h,
  cases h,
  -- Case (A): 2x + y = 5
  {
    sorry
  },
  -- Case (D): 2x + 3y = 15
  {
    sorry
  }
end

end determine_proportionality_l371_371577


namespace C1_rectangular_eq_intersection_points_polar_coords_l371_371937

-- Definition of the parametric equations for curve C₁
def C1_parametric (t : ℝ) := (3 + t * cos (π / 4), 1 - t * sin (π / 4))

-- Definition of the rectangular coordinate equation for C₁
def C1_rectangular : (ℝ × ℝ) → Prop 
| (x, y) := x + y - 4 = 0

-- Definition of the polar equation for curve C₂
def C2_polar (θ : ℝ) := 4 * cos θ

-- Definition of the rectangular coordinate equation for C₂ derived from the polar equation
def C2_rectangular : (ℝ × ℝ) → Prop 
| (x, y) := x^2 + y^2 - 4 * x = 0

-- Lean statement proving the rectangular coordinate equation of curve C₁
theorem C1_rectangular_eq : ∀ (t : ℝ), C1_rectangular (C1_parametric t) := by
  -- Proof is skipped
  sorry

-- Lean statement proving the intersection points of curves C₁ and C₂ in polar coordinates
theorem intersection_points:
  ∀ (x y : ℝ),
    C1_rectangular (x, y) → 
    C2_rectangular (x, y) → 
    ((x, y) = (2, 2) ∨ (x, y) = (4, 0)) :=
  by
    -- Proof is skipped
    sorry

-- Lean statement for converting rectangular coordinates to polar coordinates
noncomputable def to_polar (x y : ℝ) : ℝ × ℝ :=
  (real.sqrt (x * x + y * y), real.atan2 y x)

-- Lean statement proving the conversion of intersection points to polar coordinates
theorem polar_coords:
  (to_polar 2 2) = (2 * real.sqrt 2, π / 4) ∧ (to_polar 4 0) = (4, 0) := by
  -- Proof is skipped
  sorry

end C1_rectangular_eq_intersection_points_polar_coords_l371_371937


namespace circ_ABC1_passes_through_A_HM_l371_371857

variables {A B C B1 C1 B2 C2 : Point}
variable {ω : Circle}

-- Define the triangle and conditions
variable (ABC : Triangle A B C)
variable (B1_on_AB : Point_on_line_segment B1 ⟨A, B⟩)
variable (C1_on_AC : Point_on_line_segment C1 ⟨A, C⟩)
variable (B2_C2_on_BC : ∀ B2 C2, Spiral_similarity A B1 C1 C2 B2 → Point_on_line_segment B2 ⟨B, C⟩ ∧ Point_on_line_segment C2 ⟨B, C⟩ )
variable (circ_ABC1 : Circumcircle A B1 C1 ω)
variable (B1B2_C1C2_concur : ∃ D ∈ ω, D ≠ B1 ∧ D ≠ C1 ∧ Line_intersection B1 B2 C1 C2 D)

-- Define the fixed point as the A-Humpty Point of ΔABC
variable (A_HM : Point)
axiom A_HM_is_A_humpty : A_Humpty_point A_HM ABC

-- Proposition statement
theorem circ_ABC1_passes_through_A_HM :
  ∃! P ∈ ω, P ≠ A ∧ P = A_HM :=
sorry

end circ_ABC1_passes_through_A_HM_l371_371857


namespace remaining_balance_l371_371376

theorem remaining_balance
  (deposit : ℝ)
  (original_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (deposit_paid : deposit = 120)
  (deposit_fraction : 0.10 * original_price = deposit)
  (discount_fraction : discount_rate = 0.15)
  (tax_fraction : tax_rate = 0.07) :
  let discounted_price := original_price * (1 - discount_rate)
  let final_amount := discounted_price * (1 + tax_rate)
  let remaining_balance := final_amount - deposit
  in remaining_balance = 971.40 :=
by
  sorry

end remaining_balance_l371_371376


namespace floor_condition_x_l371_371821

theorem floor_condition_x (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 48) : 8 ≤ x ∧ x < 49 / 6 := 
by 
  sorry

end floor_condition_x_l371_371821


namespace ratio_of_curvilinear_triangles_l371_371938

theorem ratio_of_curvilinear_triangles
  (A B C D E F : Type)
  (α : ℝ)
  (hyp_right : ∠BCA = π / 2)
  (angle_alpha : ∠CAB = α)
  (arc_center_C : circle_center C A B)
  (touches_hypotenuse : arc_touches_hypotenuse D AB)
  (intersects_legs : arc_intersects_legs E AC F BC) :
  let R := circle_radius C in
  let area_curv_ADE := (1/2) * R^2 * (cot α - (π/2) + α) in
  let area_curv_BDF := (1/2) * R^2 * (tan α - α) in
  area_curv_ADE / area_curv_BDF = (cot α - (π/2) + α) / (tan α - α) :=
sorry

end ratio_of_curvilinear_triangles_l371_371938


namespace smallest_N_l371_371676

theorem smallest_N (N : ℕ) :
  (∀ (f : Fin 1008 → (ℕ × ℕ)),
    (∀ i, (∃ x y, x ≠ y ∧ x, y ∈ (List.range 2016).map (Nat.succ) ∧ f i = (x, y)) ∧ ∀ i, (f i).fst ≤ (f i).snd ∧ (f i).fst * (f i).snd ≤ N)) ↔ N ≥ 1017072 :=
sorry

end smallest_N_l371_371676


namespace octagon_area_l371_371721

def diagonal_of_square : ℝ := 10
def side_length_of_square (d : ℝ) : ℝ := d / Real.sqrt 2
def area_of_square (s : ℝ) : ℝ := s^2
def area_of_triangle (s : ℝ) : ℝ := (1 / 2) * s^2
def area_of_octagon (d : ℝ) : ℝ := area_of_square (side_length_of_square d) + 4 * area_of_triangle (side_length_of_square d)

theorem octagon_area (d : diagonal_of_square = 10) : area_of_octagon diagonal_of_square = 150 :=
by
  sorry

end octagon_area_l371_371721


namespace no_such_function_exists_l371_371809

theorem no_such_function_exists :
  ¬(∃ (f : ℕ → ℕ), ∀ n > 1, f(n) = f(f(n - 1)) + f(f(n + 1))) :=
sorry

end no_such_function_exists_l371_371809


namespace simplify_trig_expression_l371_371651

-- Define the conditions as hypotheses
variables (α : ℝ)
hypothesis h1 : cos ((5 / 2) * Real.pi - α) = sin α
hypothesis h2 : cos (-α) = cos α
hypothesis h3 : sin ((3 / 2) * Real.pi + α) = -cos α
hypothesis h4 : cos ((21 / 2) * Real.pi - α) = sin α

theorem simplify_trig_expression : 
  (cos ((5 / 2) * Real.pi - α) * cos (-α)) / (sin ((3 / 2) * Real.pi + α) * cos ((21 / 2) * Real.pi - α)) = -1 :=
by
  -- Proof will be provided here
  sorry

end simplify_trig_expression_l371_371651


namespace counterfeit_coin_identifiable_in_two_weighings_l371_371945

/-- One of the four coins is counterfeit and differs in weight from the real ones.
    We state that the counterfeit coin can be identified in 2 weighings. -/
theorem counterfeit_coin_identifiable_in_two_weighings :
  (∃ (coins : Fin 4 → ℕ), 
  (∃ i : Fin 4, ∀ j : Fin 4, j ≠ i → coins j = 1) ∧ (coins i ≠ 1) → 
  (∃ (w1 w2 : Fin 2 → Fin 4), 
   (w1 ≠ w2) ∧  ∀ b : Fin 2 → bool, weigh coins w1 w2 b = true) ) :=
sorry

end counterfeit_coin_identifiable_in_two_weighings_l371_371945


namespace pizza_slices_left_l371_371003

theorem pizza_slices_left (total_slices_per_pizza : ℕ) (hawaiian_slices_dean : ℕ) 
                          (hawaiian_slices_frank : ℕ) (cheese_slices_sammy : ℕ) : 
    total_slices_per_pizza = 12 → hawaiian_slices_dean = 12 / 2 → 
    hawaiian_slices_frank = 3 → cheese_slices_sammy = 12 / 3 → 
    (2 * total_slices_per_pizza - (hawaiian_slices_dean + hawaiian_slices_frank + cheese_slices_sammy)) = 11 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end pizza_slices_left_l371_371003


namespace selling_price_is_3_point_50_l371_371385

-- Define the cost of one magazine
def cost_per_magazine : ℝ := 3

-- Define the number of magazines bought
def number_of_magazines : ℕ := 10

-- Define the gain Jewel will make
def gain : ℝ := 5

-- Define the total cost of the magazines
def total_cost : ℝ := number_of_magazines * cost_per_magazine

-- Define the total amount received from selling the magazines
def total_amount_received : ℝ := total_cost + gain

-- Define the selling price per magazine
def price_per_magazine : ℝ := total_amount_received / number_of_magazines

-- Theorem to prove the selling price per magazine is $3.50
theorem selling_price_is_3_point_50 : price_per_magazine = 3.50 := by
  sorry

end selling_price_is_3_point_50_l371_371385


namespace triangle_must_be_equilateral_l371_371669

-- Given an incircle touching the sides at points A', B', and C', respectively
def incircle_touches (A B C A' B' C': Point) (triangleABC : Triangle A B C) :=
  touches (incircle triangleABC) (segment A A') ∧
  touches (incircle triangleABC) (segment B B') ∧
  touches (incircle triangleABC) (segment C C')

-- Given the condition that AA' = BB' = CC'
def equal_distances_from_vertices_to_tangency_points (A B C A' B' C': Point) := 
  dist A A' = dist B B' ∧ 
  dist B B' = dist C C'

-- Prove that triangle ABC must be equilateral
theorem triangle_must_be_equilateral
  (A B C A' B' C' : Point)
  (triangleABC : Triangle A B C)
  (h1: incircle_touches A B C A' B' C' triangleABC)
  (h2: equal_distances_from_vertices_to_tangency_points A B C A' B' C') :
  is_equilateral triangleABC :=
sorry

end triangle_must_be_equilateral_l371_371669


namespace mean_of_solutions_eq_neg_one_l371_371460

theorem mean_of_solutions_eq_neg_one :
  let S := {x : ℝ | x^3 + 3*x^2 - 4*x = 0} in
  S.nonempty → ∑ x in S.to_finset, x / S.to_finset.card = -1 :=
by
  sorry

end mean_of_solutions_eq_neg_one_l371_371460


namespace fraction_identity_l371_371842

theorem fraction_identity (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end fraction_identity_l371_371842


namespace convert_base_10_to_base_13_l371_371795

theorem convert_base_10_to_base_13 :
  ∀ (n : ℕ), n = 156 → 13 < n ∧ n < 169 ∧ 
    ((n / 13 = 12) ∧ (n % 13 = 0)) → 
    (show (12 : ∀(A B C : ℕ), 
    string.abbrev = "C") from sorry) :=
begin
  intros n h eq1 eq2 eq3 eq4,
  sorry
end

end convert_base_10_to_base_13_l371_371795


namespace center_of_circle_is_1_neg2_l371_371373

theorem center_of_circle_is_1_neg2
  (tangent1 tangent2 : ℝ → Prop)
  (center_on_line : ℝ → ℝ → Prop)
  (tangent1_def : ∀ x y, tangent1 y ↔ 4 * x - 3 * y = 30)
  (tangent2_def : ∀ x y, tangent2 y ↔ 4 * x - 3 * y = -10)
  (center_on_line_def : ∀ x y, center_on_line x y ↔ 2 * x + y = 0) :
  ∃ x y, (4 * x - 3 * y = 10) ∧ (2 * x + y = 0) ∧ x = 1 ∧ y = -2 :=
begin
  sorry
end

end center_of_circle_is_1_neg2_l371_371373


namespace nate_cooking_for_people_l371_371366

/-- Given that 8 jumbo scallops weigh one pound, scallops cost $24.00 per pound, Nate is pairing 2 scallops with a corn bisque per person, and he spends $48 on scallops. We want to prove that Nate is cooking for 8 people. -/
theorem nate_cooking_for_people :
  (8 : ℕ) = 8 →
  (24 : ℕ) = 24 →
  (2 : ℕ) = 2 →
  (48 : ℕ) = 48 →
  let scallops_per_pound := 8
  let cost_per_pound := 24
  let scallops_per_person := 2
  let money_spent := 48
  let pounds_of_scallops := money_spent / cost_per_pound
  let total_scallops := scallops_per_pound * pounds_of_scallops
  let people := total_scallops / scallops_per_person
  people = 8 :=
by
  sorry

end nate_cooking_for_people_l371_371366


namespace largest_angle_in_triangle_l371_371173

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A = 45) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : 
  max A (max B C) = 75 :=
by
  -- Since no proof is needed, we mark it as sorry
  sorry

end largest_angle_in_triangle_l371_371173


namespace minimum_value_in_interval_l371_371506

def cubic_function (a : ℝ) (x : ℝ) : ℝ := a * x^3 - (3 / 2) * x^2 + 2 * x + 1

theorem minimum_value_in_interval :
  ∃ (x_min : ℝ), 
    x_min ∈ (Set.Ioc (1 : ℝ) 3) ∧ 
    (∀ x ∈ (Set.Ioc (1 : ℝ) 3), cubic_function (1/3) x_min ≤ cubic_function (1/3) x) ∧ 
    cubic_function (1/3) x_min = (5 / 3) := 
sorry

end minimum_value_in_interval_l371_371506


namespace find_n_l371_371163

noncomputable def collinear_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), (k ≠ 0) ∧ (b = (k * a.1, k * a.2))

theorem find_n (n : ℝ) 
  (h_collinear : collinear_vectors (n, 1) (4, n)) : n = 2 ∨ n = -2 :=
begin
  sorry
end

end find_n_l371_371163


namespace pow_div_l371_371319

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l371_371319


namespace sales_relationship_l371_371172

theorem sales_relationship :
  (∀ x y: ℝ, ((x = 13 ∧ y = 25) ∨ (x = 18 ∧ y = 20)) → y = -x + 38) :=
by
  intros x y h
  cases h with h1 h2
  {
    cases h1 with hx hy
    rw hx at hy
    rw hy
    ring
  }
  {
    cases h2 with hx hy
    rw hx at hy
    rw hy
    ring
  }
  sorry

end sales_relationship_l371_371172


namespace magnitude_of_conjugate_l371_371917

theorem magnitude_of_conjugate : 
  ∀ (z : ℂ), z = (3 + 4 * complex.I) / (1 - 2 * complex.I) → 
  complex.abs (conj z) = 2 * real.sqrt 2 :=
by
  assume z,
  intro h,
  sorry

end magnitude_of_conjugate_l371_371917


namespace minimum_value_expression_is_neg27_l371_371026

noncomputable def minimum_value_expression : ℤ :=
  let expr (x y : ℝ) :=
    (√(2 * (1 + Real.cos (2 * x))) - √(36 - 4 * √5) * Real.sin x + 2) *
    (3 + 2 * √(10 - √5) * Real.cos y - Real.cos (2 * y))
  in
  Int.round (Inf (Set.range (λ (xy : ℝ × ℝ), expr xy.1 xy.2)))

theorem minimum_value_expression_is_neg27 :
  minimum_value_expression = -27 := 
by 
  sorry

end minimum_value_expression_is_neg27_l371_371026


namespace ivan_running_distance_l371_371582

theorem ivan_running_distance (x MondayDistance TuesdayDistance WednesdayDistance ThursdayDistance FridayDistance : ℝ) 
  (h1 : MondayDistance = x)
  (h2 : TuesdayDistance = 2 * x)
  (h3 : WednesdayDistance = x)
  (h4 : ThursdayDistance = (1 / 2) * x)
  (h5 : FridayDistance = x)
  (hShortest : ThursdayDistance = 5) :
  MondayDistance + TuesdayDistance + WednesdayDistance + ThursdayDistance + FridayDistance = 55 :=
by
  sorry

end ivan_running_distance_l371_371582


namespace x_intercept_of_line_l371_371185

open Real

theorem x_intercept_of_line : 
  ∃ x : ℝ, 
  (∃ m : ℝ, m = (3 - -5) / (10 - -6) ∧ (∀ y : ℝ, y = m * (x - 10) + 3)) ∧ 
  (∀ y : ℝ, y = 0 → x = 4) :=
sorry

end x_intercept_of_line_l371_371185


namespace solve_a_2000_l371_371397

-- Define a condition function to check if a sequence satisfies the problem's constraints
def is_sequence_valid (a : ℕ → ℕ) : Prop :=
  ∀ (m n : ℕ), (m > 0) → (n > 0) → (m ∣ n) → (m < n) → (a m ∣ a n) ∧ (a m < a n)

-- Define our sequence function based on the problem statement
def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We'll assume sequence is defined for positive integers only
  | _ => 2 ^ (nat.factors n).sum

-- The main proof statement to show a_{2000} is 128 for the valid sequence
theorem solve_a_2000 : ∀ a : ℕ → ℕ, is_sequence_valid a → a 2000 = 128 := by
  sorry

end solve_a_2000_l371_371397


namespace problem_l371_371603

def T := {n : ℤ | ∃ (k : ℤ), n = 4 * (2*k + 1)^2 + 13}

theorem problem :
  (∀ n ∈ T, ¬ 2 ∣ n) ∧ (∀ n ∈ T, ¬ 5 ∣ n) :=
by
  sorry

end problem_l371_371603


namespace valid_numbers_count_l371_371529

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 5 ∧ d < 10

def count_valid_numbers : ℕ :=
  let first_digit_choices := 8 -- from 1 to 9 excluding 5
  let second_digit_choices := 8 -- from the digits (0-9 excluding 5 and first digit)
  let third_digit_choices := 7 -- from the digits (0-9 excluding 5 and first two digits)
  let fourth_digit_choices := 6 -- from the digits (0-9 excluding 5 and first three digits)
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem valid_numbers_count : count_valid_numbers = 2688 :=
  by
  sorry

end valid_numbers_count_l371_371529


namespace number_of_possible_orders_l371_371128

def number_of_finishing_orders : ℕ := 4 * 3 * 2 * 1

theorem number_of_possible_orders : number_of_finishing_orders = 24 := 
by
  have h : number_of_finishing_orders = 24 := by norm_num
  exact h

end number_of_possible_orders_l371_371128


namespace number_of_integers_satisfying_inequality_l371_371804

theorem number_of_integers_satisfying_inequality :
  {m : ℤ | (m - 3) * (m + 5) < 0}.card = 7 :=
begin
  sorry
end

end number_of_integers_satisfying_inequality_l371_371804


namespace collinear_A_P_Q_l371_371498

open EuclideanGeometry

/-- Given in triangle ABC:
  1. ∠ABC = 70°
  2. ∠ACB = 30°
  
  Points P and Q inside the triangle such that:
  3. ∠QBC = ∠QCB = 10°
  4. ∠PBQ = ∠PCB = 20°

  We need to prove that points A, P, and Q are collinear. -/
theorem collinear_A_P_Q (A B C P Q : Point)
  (h_triangle : EuclideanTriangle A B C)
  (h_angle_ABC : ∠ B A C = 70)
  (h_angle_ACB : ∠ A C B = 30)
  (h_point_Q : point_inside_triangle Q A B C)
  (h_point_P : point_inside_triangle P A B C)
  (h_angle_QBC : ∠ Q B C = 10)
  (h_angle_QCB : ∠ Q C B = 10)
  (h_angle_PBQ : ∠ P B Q = 20)
  (h_angle_PCB : ∠ P C B = 20) :
  collinear {A, P, Q} :=
sorry

end collinear_A_P_Q_l371_371498


namespace shift_function_min_k_l371_371668

theorem shift_function_min_k:
  ∀ (f g : ℝ → ℝ), (∀ x, f x = sin x * cos x - sqrt 3 * cos x ^ 2) →
                   (∀ x, g x = sin (2 * x + π / 3) - sqrt 3 / 2) →
                   (∃ k > 0, ∀ x, f x = g (x - k)) →
                   ∃ k, k = π / 3 :=
by
  sorry

end shift_function_min_k_l371_371668


namespace initial_pipes_l371_371916

variables (x : ℕ)

-- Defining the conditions
def one_pipe_time := x -- time for 1 pipe to fill the tank in hours
def eight_pipes_time := 1 / 4 -- 15 minutes = 1/4 hour

-- Proving the number of pipes
theorem initial_pipes (h1 : eight_pipes_time * 8 = one_pipe_time) : x = 2 :=
by
  sorry

end initial_pipes_l371_371916


namespace race_permutations_l371_371135

-- Define the problem conditions: four participants
def participants : Nat := 4

-- Define the factorial function for permutations
def factorial : Nat → Nat
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem: The number of different possible orders in which Harry, Ron, Neville, and Hermione can finish is 24
theorem race_permutations : factorial participants = 24 :=
by
  simp [participants, factorial]
  sorry

end race_permutations_l371_371135


namespace coefficient_x6_in_expansion_l371_371019

open BigOperators

theorem coefficient_x6_in_expansion : 
  let expr := (x + 1)^6 * (∑ i in Finset.range 7, x^i) in
  (expr.coeff 6 = 64) :=
by
  let x := (Polynomial.X)
  sorry

end coefficient_x6_in_expansion_l371_371019


namespace Jake_weight_is_118_l371_371909

-- Define the current weights of Jake, his sister, and Mark
variable (J S M : ℕ)

-- Define the given conditions
axiom h1 : J - 12 = 2 * (S + 4)
axiom h2 : M = J + S + 50
axiom h3 : J + S + M = 385

theorem Jake_weight_is_118 : J = 118 :=
by
  sorry

end Jake_weight_is_118_l371_371909


namespace problem_stmt_l371_371936

-- Definitions of the known angles and conditions
def angle_AE_eq_50 : Prop := angle A E B = 50
def angle_B_eq_90 : Prop := angle B = 90
def isosceles_BED : Prop := angle B E D = angle B D E

-- Conclusion we want to prove
def angle_CDE : Prop := angle C D E = 42.5

-- The main statement combining the conditions and the desired result
theorem problem_stmt (h1 : angle_AE_eq_50) (h2 : angle_B_eq_90) (h3 : isosceles_BED) : angle_CDE :=
  sorry

end problem_stmt_l371_371936


namespace friday_13th_more_likely_l371_371641

-- Definitions for Gregorian Calendar system
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def days_in_year (y : ℕ) : ℕ :=
  if is_leap_year y then 366 else 365

def days_in_400_years : ℕ :=
  (303 * 365 + 97 * 366)

lemma days_mod_7_zero : days_in_400_years % 7 = 0 :=
  by
    have : days_in_400_years = 146097 := by sorry
    exact Nat.ModEq.subst_eq 7 146097 0 (by sorry)

-- Main theorem: The 13th day of the month is more likely to fall on a Friday.
theorem friday_13th_more_likely : 
  (∃ k: ℕ, k <= 6 ∧ 
    ∀ (d: ℕ), 
      (d < 7 → 
        (∃ n : ℕ, (days_in_400_years ≡ n [MOD 7]) ∧ 
          (d = 5 ↔ 
            (∃ m: ℕ, 
              (m >= 1 ∧ m <= 12) ∧ 
              (13 + (if m = 1 then d else ⟨(* simple month starting day logic here *)⟩ ) ≡ 5 [MOD 7]))))) :=
by sorry

end friday_13th_more_likely_l371_371641


namespace largest_prime_factor_of_4620_l371_371724

open Nat

theorem largest_prime_factor_of_4620 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4620 ∧ (∀ q : ℕ, (Nat.Prime q ∧ q ∣ 4620) → q ≤ p) :=
begin
  use 11,
  split,
  { apply Nat.prime_of_nat_prime, exact prime_11, },
  split,
  { apply divides_prime_factors, norm_num, },
  { intros q hq,
    apply le_trans (prime_le_magnitude hq.1),
    suffices : q ∣ 4620 ∧ q ∈ { 2, 5, 3, 7, 11 }, from this.elim (λ H h, H.symm ▸ Nat.le_of_eq (set.eq_of_mem_singleton h)),
    exact ⟨hq.2,Hq⟩,
    { apply and.intro, exact Nat.prime_divisors.mem_list.mp _, exact list.mem_cons_of_mem _, exact hq } 
  },
end

end largest_prime_factor_of_4620_l371_371724


namespace largest_consecutive_sum_105_l371_371335

noncomputable def max_n_consecutive_sum_105 : ℕ :=
  let S : ℕ → ℕ := λ n => n * (n + 1) / 2
  (argmax (λ n => S n ≤ 105) 0) -- In search of maximum n where sum of first n positive integers is ≤ 105

theorem largest_consecutive_sum_105 : 
  max_n_consecutive_sum_105 = 14 :=
sorry

end largest_consecutive_sum_105_l371_371335


namespace missing_angle_is_zero_l371_371761

theorem missing_angle_is_zero (n : ℕ) (x : ℕ)
  (sum_of_all_but_one : ∑ i in finset.range (n-1), 180 - 360/(n-1) = 3240)
  (sum_of_interior_angles : n ≥ 3 → (n : ℝ) * 180 - 360 = (n : ℝ) * (180 - 2)) :
  x = 0 :=
by
  sorry

end missing_angle_is_zero_l371_371761


namespace medians_formula_l371_371837

noncomputable def ma (a b c : ℝ) : ℝ := (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2))
noncomputable def mb (a b c : ℝ) : ℝ := (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2))
noncomputable def mc (a b c : ℝ) : ℝ := (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2))

theorem medians_formula (a b c : ℝ) :
  ma a b c = (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2)) ∧
  mb a b c = (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2)) ∧
  mc a b c = (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2)) :=
by sorry

end medians_formula_l371_371837


namespace identical_rows_l371_371856

theorem identical_rows (n : ℕ) (a b : fin n → ℝ)
  (h1 : ∀ i j, (i < j) → (a i < a j)) -- first row is strictly increasing
  (h2 : ∀ x, ∃ i, b i = x ∧ ∃ j, a j = x) -- second row is a rearrangement
  (h3 : ∀ i, a i + b i < a (i+1) + b (i+1)) -- third row is strictly increasing
  (h4 : a (fin.last n) + b (fin.last n - 1) < a (fin.last n - 1) + b (fin.last n)) -- special case strict increase condition
  : ∀ i, a i = b i := 
sorry

end identical_rows_l371_371856


namespace S_arithmetic_iff_a_constant_l371_371907

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (d : ℝ)

-- S_n is the sum of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℝ := ∑ i in range (n + 1), a i

-- statement of the problem
theorem S_arithmetic_iff_a_constant :
  (∃ d : ℝ, ∀ n : ℕ, S (n + 1) - S n = d) ↔ ∀ n : ℕ, a n = d :=
sorry

end S_arithmetic_iff_a_constant_l371_371907


namespace angle_ACB_ninety_degrees_l371_371894

theorem angle_ACB_ninety_degrees : 
  let line := λ p : ℝ × ℝ, p.1 - p.2 + 2 = 0
  let circle := λ p : ℝ × ℝ, (p.1 - 3)^2 + (p.2 - 3)^2 = 4
  let A B : ℝ × ℝ
  
  (line A) ∧ (circle A) ∧ (line B) ∧ (circle B) →
  ∃ C : ℝ × ℝ, (C = (3, 3)) →
  ∠ACB = 90 :=
by
  sorry

end angle_ACB_ninety_degrees_l371_371894


namespace power_division_identity_l371_371312

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l371_371312


namespace radius_of_circle_B_l371_371292

theorem radius_of_circle_B (diam_A : ℝ) (factor : ℝ) (r_A r_B : ℝ) 
  (h1 : diam_A = 80) 
  (h2 : r_A = diam_A / 2) 
  (h3 : r_A = factor * r_B) 
  (h4 : factor = 4) : r_B = 10 := 
by 
  sorry

end radius_of_circle_B_l371_371292


namespace liangliang_distance_to_school_l371_371614

theorem liangliang_distance_to_school :
  (∀ (t : ℕ), (40 * t = 50 * (t - 5)) → (40 * 25 = 1000)) :=
sorry

end liangliang_distance_to_school_l371_371614


namespace find_distinct_ordered_pairs_l371_371900

theorem find_distinct_ordered_pairs : 
  {xy : Nat × Nat // 0 < xy.1 ∧ 0 < xy.2 ∧ (1 / (xy.1 : ℚ) + 1 / (xy.2 : ℚ) = 1 / 5)}.card = 3 :=
by
  sorry

end find_distinct_ordered_pairs_l371_371900


namespace ascending_order_sqrt2_32_54_88_916_l371_371265

theorem ascending_order_sqrt2_32_54_88_916 :
  let sqrt2 : ℝ := real.sqrt 2
  let n32 : ℝ := 32
  let n54 : ℝ := 54
  let n88 : ℝ := 88
  let n916 : ℝ := 916
  ∀ (a b c d e : ℝ), 
    a = sqrt2 → b = n32 → c = n54 → d = n88 → e = n916 →
    (b, d, c, e, a) = (32, 88, 54, 916, sqrt2) := by
  sorry

end ascending_order_sqrt2_32_54_88_916_l371_371265


namespace problem_statement_l371_371883

def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

-- The problem statement in Lean.
theorem problem_statement : f (-1) + f 1 = 0 :=
by
  -- proof would go here...
  sorry

end problem_statement_l371_371883


namespace math_problem_l371_371713

def problem_statement : Prop :=
  let cond1 := ¬ (0 = ({0} : set ℕ))
  let cond2 := ∀ (a b c : ℕ), {a, b, c} = {c, b, a}
  let cond3 := ¬ ({4 ⊢ (x - 1)^2 * (x - 2) = 0} = {1, 1, 2} : set ℕ)
  let cond4 := ¬ (set.finite {x : ℝ | 4 < x ∧ x < 5})
  cond1 ∧ cond2 ∧ cond3 ∧ cond4 → (¬cond1 ∧ cond2 ∧ ¬cond3 ∧ ¬cond4)

theorem math_problem : problem_statement :=
by
  sorry

end math_problem_l371_371713


namespace tangent_lines_parallel_to_line_l371_371457

theorem tangent_lines_parallel_to_line (a : ℝ) (b : ℝ)
  (h1 : b = a^3 + a - 2)
  (h2 : 3 * a^2 + 1 = 4) :
  (b = 4 * a - 4 ∨ b = 4 * a) :=
sorry

end tangent_lines_parallel_to_line_l371_371457


namespace problem_lean_l371_371215

-- Define the function f(x)
def f (x : ℝ) : ℝ := 9^x / (9^x + 3)

-- The main Lean theorem statement
theorem problem_lean
  (S : ℝ)
  (hS : S = ∑ i in (finset.range 2015).filter (λ n, 0 < n ∧ n < 2015), f ((n : ℝ) / 2015)) :
  S = 1007 :=
sorry

end problem_lean_l371_371215


namespace number_of_paths_l371_371394

def move (a b : ℕ) : list (ℕ × ℕ) :=
by
  exact [(a+1, b), (a, b+1), (a+1, b+1)]

def valid_move (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
by
  exact ¬ (p2.1 ≠ p1.1 ∧ p2.2 ≠ p1.2 ∧ p3.1 = p1.1 ∧ p3.2 = p1.2)

def no_right_angles (path : list (ℕ × ℕ)) : Prop :=
by
  match path with
  | [] => true
  | [p] => true
  | [p1, p2] => true
  | p1 :: p2 :: p3 :: ps => valid_move p1 p2 p3 ∧ no_right_angles (p2 :: p3 :: ps)

def end_point (path : list (ℕ × ℕ)) : (ℕ × ℕ) := 
by
  match path with
  | [] => (0, 0)
  | ps => ps.reverse.head

-- Main statement
theorem number_of_paths : 
  ∃ paths : list (list (ℕ × ℕ)), 
    (∀ p ∈ paths, p.head = (0, 0) ∧ end_point p = (6, 6)) ∧ 
    (∀ p ∈ paths, no_right_angles p) ∧ 
    (paths.length = 179) := 
by 
  sorry

end number_of_paths_l371_371394


namespace alice_number_l371_371413

theorem alice_number (n : ℕ) 
  (h1 : 180 ∣ n) 
  (h2 : 75 ∣ n) 
  (h3 : 900 ≤ n) 
  (h4 : n ≤ 3000) : 
  n = 900 ∨ n = 1800 ∨ n = 2700 := 
by
  sorry

end alice_number_l371_371413


namespace multiples_of_15_between_17_and_158_l371_371147

theorem multiples_of_15_between_17_and_158 : 
  let first := 30
  let last := 150
  let step := 15
  Nat.succ ((last - first) / step) = 9 := 
by
  sorry

end multiples_of_15_between_17_and_158_l371_371147


namespace cannot_reach_goal_state_l371_371349

-- Define the size of the board
def board_size : ℕ := 5

-- Define the initial state of the board (all zeros)
def initial_board (i j : ℕ) (h1 : i > 0 ∧ i ≤ board_size) (h2 : j > 0 ∧ j ≤ board_size) : ℕ := 0

-- Define the update operation on the board
def update_board (board : ℕ → ℕ → ℕ) (i j : ℕ) :=
  λ x y, if (x = i ∧ y = j) ∨ (x = i + 1 ∧ y = j) ∨ (x = i - 1 ∧ y = j) ∨ (x = i ∧ y = j + 1) ∨ (x = i ∧ y = j - 1) then board x y + 1 else board x y

-- Define the target value we want in each cell
def target_value : ℕ := 2012

-- Define the goal state (all cells have the target value)
def goal_state (i j : ℕ) (h1 : i > 0 ∧ i ≤ board_size) (h2 : j > 0 ∧ j ≤ board_size) : ℕ := target_value

-- Now, we provide the main theorem we need to prove
theorem cannot_reach_goal_state :
  ¬(∃ (f : ℕ → ℕ → ℕ → ℕ), 
    (f 0 = initial_board) ∧
    (∀ t i j h1 h2, f (t+1) = update_board (f t i j)) ∧
    (∀ i j h1 h2, (f some_t i j) = goal_state i j)) := sorry

end cannot_reach_goal_state_l371_371349


namespace min_angle_in_annulus_l371_371573

theorem min_angle_in_annulus (R r : ℝ) (A B : ℂ) (O : ℂ) 
  (hR_nonneg : R ≥ 0.5) 
  (hR_ge_r : R ≥ r) 
  (hA_on_outer_circle : abs (A - O) = R)
  (hB_on_inner_circle : abs (B - O) ≤ R) 
  (hAB : abs (A - B) = 1) : 
  ∃ θ : ℝ, angle O A B = θ ∧ θ = 0 := 
sorry

end min_angle_in_annulus_l371_371573


namespace find_m_l371_371612

open Set

def A (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 2}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem find_m (m : ℝ) :
  (A m ∩ B = ∅ ∧ A m ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end find_m_l371_371612


namespace max_value_of_a_l371_371522

theorem max_value_of_a
  (a : ℝ)
  (f : ℝ → ℝ := λ x, a * x^2 - a * x + 1) :
  (∀ x ∈ Icc (0 : ℝ) 1, abs (f x) ≤ 1) → a ≤ 8 := sorry

end max_value_of_a_l371_371522


namespace find_original_number_l371_371390

variable (x : ℕ)

theorem find_original_number (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by
  sorry

end find_original_number_l371_371390


namespace solution_set_of_inequality_f_greater_2_range_of_t_l371_371231

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 2) - abs (x - 2)

-- Question 1: Prove that the solution set of the inequality f(x) > 2 is {x | x < -6 ∨ ⅔ < x < 2 ∨ x ≥ 2}
theorem solution_set_of_inequality_f_greater_2 :
  { x : ℝ | f x > 2 } = { x : ℝ | x < -6 ∨ (⅔ < x ∧ x < 2) ∨ x ≥ 2 } := sorry

-- Question 2: Prove that ∃ x ∈ ℝ, f(x) < 2 - (7 / 2) * t implies t ∈ (-∞, 3 / 2) ∪ (2, ∞)
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x < 2 - (7 / 2) * t) → (t < 3 / 2 ∨ t > 2) := sorry

end solution_set_of_inequality_f_greater_2_range_of_t_l371_371231


namespace count_abundant_less_than_50_l371_371770

def is_abundant (n : ℕ) : Prop :=
  ∑ i in (Finset.filter (λ d, d ≠ n ∧ n % d = 0) (Finset.range (n+1))), i > n

def count_abundant_numbers (k : ℕ) : ℕ :=
  (Finset.range k).filter (λ n, is_abundant n).card

theorem count_abundant_less_than_50 : count_abundant_numbers 50 = 9 :=
by
  sorry

end count_abundant_less_than_50_l371_371770


namespace angle_BOC_eq_angle_AOD_l371_371475

variable {A B C D E F P O : Type}
variable [ConvexQuadrilateral A B C D]
variable [IntersectsOppositeSides A B C D E F]
variable [DiagonalsIntersect A B C D P]
variable [PerpendicularLine P EF O]

theorem angle_BOC_eq_angle_AOD : ∠(B O C) = ∠(A O D) :=
sorry

end angle_BOC_eq_angle_AOD_l371_371475


namespace intersection_correct_l371_371110

def A (x : ℝ) : Prop := |x| > 4
def B (x : ℝ) : Prop := -2 < x ∧ x ≤ 6
def intersection (x : ℝ) : Prop := B x ∧ A x

theorem intersection_correct :
  ∀ x : ℝ, intersection x ↔ 4 < x ∧ x ≤ 6 := 
by
  sorry

end intersection_correct_l371_371110


namespace no_positive_integer_n_eqn_l371_371445

theorem no_positive_integer_n_eqn (n : ℕ) : (120^5 + 97^5 + 79^5 + 44^5 ≠ n^5) ∨ n = 144 :=
by
  -- Proof omitted for brevity
  sorry

end no_positive_integer_n_eqn_l371_371445


namespace conic_section_parabola_l371_371008

theorem conic_section_parabola (x y : ℝ) : 
  abs (x - 3) = sqrt ((y + 4)^2 + x^2) → "P" :=
by
  sorry

end conic_section_parabola_l371_371008


namespace f_1996x_eq_1996_f_x_l371_371222

theorem f_1996x_eq_1996_f_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by
  sorry

end f_1996x_eq_1996_f_x_l371_371222


namespace fraction_identity_l371_371841

theorem fraction_identity (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end fraction_identity_l371_371841


namespace percent_non_union_women_l371_371168

theorem percent_non_union_women
  (total_employees : ℕ)
  (percent_men : ℝ)
  (percent_unionized : ℝ)
  (percent_unionized_men : ℝ)
  (h_total_100 : total_employees = 100)
  (h_percent_men : percent_men = 0.48)
  (h_percent_unionized : percent_unionized = 0.60)
  (h_percent_unionized_men : percent_unionized_men = 0.70) :
  (34 / 40) * 100 = 85 :=
by 
  have h_men := (48:ℝ)
  have h_union_employees := (60:ℝ)
  have h_union_men := 0.70 * 60
  have h_non_union_men := 48 - h_union_men
  have h_non_union_employees := 100 - 60
  have h_non_union_women := h_non_union_employees - h_non_union_men
  calc (h_non_union_women / h_non_union_employees) * 100 = (34 / 40) * 100 : sorry

end percent_non_union_women_l371_371168


namespace convex_polygon_sum_of_squares_leq_four_l371_371375

theorem convex_polygon_sum_of_squares_leq_four (P : Polyhedron ℝ) (h_convex : P.IsConvex) (h_contained : P.IsContainedIn (Cube 1)) :
  (∑ i in P.sides, i.length^2) ≤ 4 := 
sorry

end convex_polygon_sum_of_squares_leq_four_l371_371375


namespace cannot_divide_m_l371_371467

/-
  A proof that for the real number m = 2009^3 - 2009, 
  the number 2007 does not divide m.
-/

theorem cannot_divide_m (m : ℤ) (h : m = 2009^3 - 2009) : ¬ (2007 ∣ m) := 
by sorry

end cannot_divide_m_l371_371467


namespace sum_coordinates_center_l371_371006

theorem sum_coordinates_center (x y : ℝ) :
  (∃ k : ℝ, (x - 5)^2 + (y + 2)^2 = k) →
  (x + y = 3) :=
by
  intros h,
  sorry

end sum_coordinates_center_l371_371006


namespace max_rectangle_area_l371_371698

variables {a b : ℝ}

theorem max_rectangle_area (h : 2 * a + 2 * b = 60) : a * b ≤ 225 :=
by 
  -- Proof to be filled in
  sorry

end max_rectangle_area_l371_371698


namespace average_age_mentors_l371_371658

variable (total_members : ℕ) (average_age_members : ℕ)
variable (num_girls : ℕ) (num_boys : ℕ) (num_mentors : ℕ)
variable (average_age_girls : ℕ) (average_age_boys : ℕ)

theorem average_age_mentors :
  total_members = 50 →
  average_age_members = 20 →
  num_girls = 25 →
  num_boys = 20 →
  num_mentors = 5 →
  average_age_girls = 18 →
  average_age_boys = 19 →
  let total_age_members := total_members * average_age_members in
  let total_age_girls := num_girls * average_age_girls in
  let total_age_boys := num_boys * average_age_boys in
  let total_age_mentors := total_age_members - total_age_girls - total_age_boys in
  let average_age_mentors := total_age_mentors / num_mentors in
  average_age_mentors = 34 := by
  sorry

end average_age_mentors_l371_371658


namespace proof_problem_1_proof_problem_2_l371_371499

noncomputable def condition1 (A: Point) : Prop := A.x = -1

noncomputable def condition2 (A: Point) (l₂: Line) : Prop := l₁ ⊥ l₂ ∧ A ∈ l₂

noncomputable def condition3 : Point := Point.mk 1 0 -- F(1,0)

noncomputable def condition4 (A F: Point) (l₂: Line) (P: Point) : Prop :=
  let bisector := perpendicularBisector A F
  P ∈ l₂ ∧ P ∈ bisector

axiom condition5 (PMN: Triangle) : inscribedCircle PMN = Circle.mk (Point.mk 0 0) 1

axiom condition6 (P F: Point) : slope (Line.mk P F) = k

axiom condition7 (MN: Length) : k^2 = λ * MN^2

theorem proof_problem_1 (P: Point) (A: Point) :
  condition1 A → condition2 A l₂ → condition3  → 
  condition4 A F l₂ P → 
  (P ∈ parabola Focus (Point.mk 1 0) Directrix (Line.mk -1 : Line) ↔ P.y^2 = 4 * P.x) :=
by
  intros
  sorry

theorem proof_problem_2 (x₀: ℝ) :
  (x₀ > 1) →
  0 < λ ∧ λ < 1/4 :=
by
  intros
  sorry

end proof_problem_1_proof_problem_2_l371_371499


namespace circle_area_through_trianglular_midpoint_l371_371715

-- Define the right triangle and its properties
def right_triangle (P Q R : Type) : Prop :=
  ∃ (PQ PR QR : ℝ), PQ = 6 ∧ PR = 8 ∧ QR = real.sqrt (PQ^2 + PR^2) ∧
  hypotenuse_midpoint P Q R

-- Define the midpoint of the hypotenuse
def hypotenuse_midpoint (P Q R M : Type) : Prop :=
  M = midpoint Q R

-- Prove the area of the circle passing through specific points
theorem circle_area_through_trianglular_midpoint (P Q R M : Type) 
(h : right_triangle P Q R) 
(hM : hypotenuse_midpoint P Q R M) :
  let radius := real.sqrt (6^2 + 8^2) / 2 in
  let area := real.pi * radius^2 in
  area = 25 * real.pi :=
by {
  -- Sorry to skip the proof
  sorry
}

end circle_area_through_trianglular_midpoint_l371_371715


namespace perpendicular_unit_vector_exists_l371_371053

theorem perpendicular_unit_vector_exists :
  ∃ (m n : ℝ), (2 * m + n = 0) ∧ (m^2 + n^2 = 1) ∧ (m = (Real.sqrt 5) / 5) ∧ (n = -(2 * (Real.sqrt 5)) / 5) :=
by
  sorry

end perpendicular_unit_vector_exists_l371_371053


namespace selection_plans_l371_371646

variables (A B C T1 T2 T3 T4 T5 T6 : Prop)

theorem selection_plans :
  ∀ (A_goes B_goes C_goes : bool),
  (A_goes = !B_goes) →
  (A_goes = C_goes) →
  -- Calculate the number of ways to select 3 teachers given the conditions
  let selection_cases := 
    (if A_goes then 3 else 4) in
  -- Calculate the permutations of those teachers
  let arrangements := 3! in
  -- Calculate the total number of selection plans
  selection_cases * arrangements = 42 :=
begin
  sorry
end

end selection_plans_l371_371646


namespace root_in_interval_l371_371415

def f (x : ℝ) := x^3 + 3 * x - 3

theorem root_in_interval : ∃ x ∈ Icc (0 : ℝ) 1, f x = 0 :=
by {
  have h1 : f 0 = -3 := by norm_num [f],
  have h2 : f 1 = 1 := by norm_num [f],
  have ivt := intermediate_value_Icc (0 : ℝ) 1 0 h1 h2 sorry,
  simp only [exists_prop, Icc_eq_empty, Icc_eq_empty_iff, not_le, le_of_lt] at ivt,
  exact ivt.left,
}

end root_in_interval_l371_371415


namespace largest_rectangle_area_l371_371684

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l371_371684


namespace min_length_MN_l371_371108

-- Define the parabola and the focus
def parabola (x y : ℝ) := y^2 = -6 * x

def focus := (-1.5, 0)

-- Minimum length calculation problem
theorem min_length_MN (M N : ℝ × ℝ) (k : ℝ) (hk : k ≠ 0) 
  (hM : parabola M.fst M.snd) (hN : parabola N.fst N.snd) 
  (hCond : M.1 - focus.1 = k * (N.1 - focus.1) ∧ 
           M.2 - focus.2 = k * (N.2 - focus.2)) :
  ∃ m : ℝ, m = 6 ∧ |M.1 - N.1| + |M.2 - N.2| = m :=
sorry

end min_length_MN_l371_371108


namespace compute_gf3_l371_371217

def f (x : ℝ) : ℝ := x^3 - 3
def g (x : ℝ) : ℝ := 2 * x^2 - x + 4

theorem compute_gf3 : g (f 3) = 1132 := 
by 
  sorry

end compute_gf3_l371_371217


namespace min_value_of_a_l371_371549

theorem min_value_of_a:
  (∃ x ∈ set.Icc (2:ℝ) (3:ℝ), (1 + a * x) / (x * 2 ^ x) ≥ 1) → a ≥ (7 / 2) :=
by
  sorry

end min_value_of_a_l371_371549


namespace ellipse_proof_slopes_proof_l371_371066

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1

noncomputable def slopes_product (k1 k2 : ℝ) : Prop :=
  4 * √2 * (1 + k1^2) / (1 + 2 * k1^2) + 4 * √2 * (1 + k2^2) / (1 + 2 * k2^2) = 6 * √2

theorem ellipse_proof :
  ∃ a b : ℝ, a = 2 * √2 ∧ b = 2 ∧ ellipse_equation a b :=
begin
  use [2 * √2, 2],
  split,
  { refl },
  split,
  { refl },
  { intros x y,
    calc (x^2)/(8) + (y^2)/(4) = 1 : sorry }
end

theorem slopes_proof (k1 k2 : ℝ) (h : slopes_product k1 k2) :
  k1 * k2 = ½ ∨ k1 * k2 = -½ :=
begin
  sorry
end

end ellipse_proof_slopes_proof_l371_371066


namespace percentage_wearing_blue_shirts_l371_371170

theorem percentage_wearing_blue_shirts (total_students : ℕ) (red_percentage green_percentage : ℕ) 
  (other_students : ℕ) (H1 : total_students = 900) (H2 : red_percentage = 28) 
  (H3 : green_percentage = 10) (H4 : other_students = 162) : 
  (44 : ℕ) = 100 - (red_percentage + green_percentage + (other_students * 100 / total_students)) :=
by
  sorry

end percentage_wearing_blue_shirts_l371_371170


namespace original_total_price_l371_371400

theorem original_total_price (total_selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (selling_price_with_profit : total_selling_price/2 = original_price * (1 + profit_percent))
  (selling_price_with_loss : total_selling_price/2 = original_price * (1 - loss_percent)) :
  (original_price / (1 + profit_percent) + original_price / (1 - loss_percent) = 1333 + 1 / 3) := 
by
  sorry

end original_total_price_l371_371400


namespace divide_circle_three_equal_areas_l371_371446

theorem divide_circle_three_equal_areas (OA : ℝ) (r1 r2 : ℝ) 
  (hr1 : r1 = (OA * Real.sqrt 3) / 3) 
  (hr2 : r2 = (OA * Real.sqrt 6) / 3) : 
  ∀ (r : ℝ), r = OA → 
  (∀ (A1 A2 A3 : ℝ), A1 = π * r1 ^ 2 ∧ A2 = π * (r2 ^ 2 - r1 ^ 2) ∧ A3 = π * (r ^ 2 - r2 ^ 2) →
  A1 = A2 ∧ A2 = A3) :=
by
  sorry

end divide_circle_three_equal_areas_l371_371446


namespace henry_earning_per_lawn_l371_371136

theorem henry_earning_per_lawn
    (total_lawns : ℕ) (forgotten_lawns : ℕ) (earned_dollars : ℕ)
    (h_total_lawns : total_lawns = 12)
    (h_forgotten_lawns : forgotten_lawns = 7)
    (h_earned_dollars : earned_dollars = 25) :
    earned_dollars / (total_lawns - forgotten_lawns) = 5 :=
by
    rw [h_total_lawns, h_forgotten_lawns, h_earned_dollars]
    norm_num
    sorry

end henry_earning_per_lawn_l371_371136


namespace problem1_problem2_l371_371888

def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 1/(2*a)

def g (a : ℝ) (x : ℝ) : ℝ := f a x + |2*x - 1|

theorem problem1 (a : ℝ) (m : ℝ) (h : a ≠ 0) : 
  (∀ x : ℝ, f a x - f a (x + m) ≤ 1) ↔ abs m ≤ 1 := 
by sorry

theorem problem2 (a : ℝ) (h : a < 1/2) : 
  (∃ x : ℝ, g a x = 0) ↔ -1/2 ≤ a ∧ a < 0 := 
by sorry

end problem1_problem2_l371_371888


namespace counterfeit_identifiable_in_two_weighings_l371_371948

-- Define the condition that one of four coins is counterfeit
def is_counterfeit (coins : Fin 4 → ℚ) (idx : Fin 4) : Prop :=
  ∃ real_weight counterfeit_weight : ℚ, real_weight ≠ counterfeit_weight ∧
  (∀ i : Fin 4, i ≠ idx → coins i = real_weight) ∧ coins idx = counterfeit_weight

-- Define the main theorem statement
theorem counterfeit_identifiable_in_two_weighings (coins : Fin 4 → ℚ) :
  (∃ idx : Fin 4, is_counterfeit coins idx) → ∃ idx : Fin 4, is_counterfeit coins idx ∧
  ∀ (balance : (Fin 4 → Prop) → ℤ → Prop), (∃ w1 w2 : Fin 4 → Prop, balance w1 = 0 ∨ balance w2 = 0 → idx) :=
sorry

end counterfeit_identifiable_in_two_weighings_l371_371948


namespace fred_earned_63_dollars_l371_371203

-- Definitions for the conditions
def initial_money_fred : ℕ := 23
def initial_money_jason : ℕ := 46
def money_per_car : ℕ := 5
def money_per_lawn : ℕ := 10
def money_per_dog : ℕ := 3
def total_money_after_chores : ℕ := 86
def cars_washed : ℕ := 4
def lawns_mowed : ℕ := 3
def dogs_walked : ℕ := 7

-- The equivalent proof problem in Lean
theorem fred_earned_63_dollars :
  (initial_money_fred + (cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = total_money_after_chores) → 
  ((cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = 63) :=
by
  sorry

end fred_earned_63_dollars_l371_371203


namespace PA_PB_value_l371_371647

theorem PA_PB_value (x y t θ : ℝ) (A B P : ℝ × ℝ) (hA : A.1 - A.2 - 2 = 0) (hB : B.1 - B.2 - 2 = 0)
  (hx : A.1 * B.1 = -32/7) (hy : A.1 + B.1 = 36/7) (hxA : A.1 = 2 + (Real.sqrt 2)/2 * t)
  (hyA : A.2 = (Real.sqrt 2)/2 * t) (hxB : B.1 = 4 * Real.cos θ) (hyB : B.2 = 2 * Real.sqrt 3 * Real.sin θ)
  (hP : P = (2, 0)) :
  |Real.dist P A * Real.dist P B| = 48/7 :=
sorry

end PA_PB_value_l371_371647


namespace evaluate_expression_l371_371428

-- Define the mathematical expressions using Lean's constructs
def expr1 : ℕ := 201 * 5 + 1220 - 2 * 3 * 5 * 7

-- State the theorem we aim to prove
theorem evaluate_expression : expr1 = 2015 := by
  sorry

end evaluate_expression_l371_371428


namespace general_term_arithmetic_sequence_l371_371865

theorem general_term_arithmetic_sequence (a_n : ℕ → ℚ) (d : ℚ) (h_seq : ∀ n, a_n n = a_n 0 + n * d)
  (h_geometric : (a_n 2)^2 = a_n 1 * a_n 6)
  (h_condition : 2 * a_n 0 + a_n 1 = 1)
  (h_d_nonzero : d ≠ 0) :
  ∀ n, a_n n = (5/3) - n := 
by
  sorry

end general_term_arithmetic_sequence_l371_371865


namespace mean_and_median_are_13_l371_371763

-- Define the arithmetic sequence and conditions
noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (a1 a2 a3 : ℝ) : Prop :=
a2 * a2 = a1 * a3

-- Given conditions
def sample_conditions (a : ℕ → ℝ) (d : ℝ) :=
(arithmetic_sequence a d) ∧
a 3 = 8 ∧
geometric_sequence (a 1) (a 3) (a 7) ∧
∃ d ≠ 0, true

-- Problem statement
theorem mean_and_median_are_13 (a : ℕ → ℝ) (d : ℝ) (h : sample_conditions a d) :
 (∑ i in finset.range 10, (a i) / 10) = 13 ∧ (a 5 + a 6) / 2 = 13 :=
by sorry

end mean_and_median_are_13_l371_371763


namespace power_division_l371_371332

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l371_371332


namespace sin_A_obtuse_triangle_l371_371571

theorem sin_A_obtuse_triangle (A B C : Triangle) (AB AC : ℝ) 
  (angle_B : ∠B = π/4) (AB_gt_AC : AB > AC) (O I : Point)
  (circumcenter_O : is_circumcenter O A B C)
  (incenter_I : is_incenter I A B C) 
  (sqrt2_OI_eq : sqrt 2 * dist O I = AB - AC):
  sin (angle A) = sqrt 2 / 2 ∨ sin (angle A) = sqrt (sqrt 2 - 1 / 2) :=
sorry

end sin_A_obtuse_triangle_l371_371571


namespace find_chair_price_correct_l371_371797

noncomputable def original_chair_price {table_price chair_price final_price : ℝ} (discount_rate : ℝ) : Prop :=
  (table_price = 55) → 
  (4 * chair_price + table_price - discount_rate * (4 * chair_price + table_price) = final_price) → 
  (final_price = 135) → 
  discount_rate = 0.15 → 
  chair_price ≈ 25.96

theorem find_chair_price_correct : original_chair_price 0.15 :=
by
  intro table_price chair_price final_price discount_rate
  sorry

end find_chair_price_correct_l371_371797


namespace find_number_l371_371818

theorem find_number :
  ∃ (x : ℝ), 0.6667 * x - 10 = 0.25 * x ∧ x ≈ 24 :=
by
  have h : ∃ (x : ℝ), 0.6667 * x - 10 = 0.25 * x := sorry
  cases h with x hx
  use x
  split
  · exact hx
  · linarith

end find_number_l371_371818


namespace race_order_count_l371_371123

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end race_order_count_l371_371123


namespace petya_square_larger_than_vasya_square_l371_371635

variable (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)

def petya_square_side (a b : ℝ) : ℝ := a * b / (a + b)

def vasya_square_side (a b : ℝ) : ℝ := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petya_square_larger_than_vasya_square
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  petya_square_side a b > vasya_square_side a b :=
by sorry

end petya_square_larger_than_vasya_square_l371_371635


namespace second_player_wins_with_optimal_play_l371_371300

/-- 
  Two players take turns drawing lines on a plane, where each line must be unique. 
  A player wins if their move causes the number of regions the plane is divided 
  into by the lines to be a multiple of 5 for the first time. 

  Prove that the second player will win with optimal play.
-/
theorem second_player_wins_with_optimal_play : 
  ∃ (optimal_strategy : (ℕ → list (nat × nat) → Prop)), 
  (∀ (turn : ℕ) (lines : list (nat × nat)),
    turn % 2 = 1 → optimal_strategy turn lines → ∃ (new_line : nat × nat), optimal_strategy (turn + 1) (new_line :: lines))
  ∧ (∀ lines, length lines % 5 = 0 → second_player_wins_with_optimal_play)
:= 
sorry

end second_player_wins_with_optimal_play_l371_371300


namespace counterfeit_coin_identifiable_in_two_weighings_l371_371944

/-- One of the four coins is counterfeit and differs in weight from the real ones.
    We state that the counterfeit coin can be identified in 2 weighings. -/
theorem counterfeit_coin_identifiable_in_two_weighings :
  (∃ (coins : Fin 4 → ℕ), 
  (∃ i : Fin 4, ∀ j : Fin 4, j ≠ i → coins j = 1) ∧ (coins i ≠ 1) → 
  (∃ (w1 w2 : Fin 2 → Fin 4), 
   (w1 ≠ w2) ∧  ∀ b : Fin 2 → bool, weigh coins w1 w2 b = true) ) :=
sorry

end counterfeit_coin_identifiable_in_two_weighings_l371_371944


namespace det_A_mod_2017_l371_371836

-- Define α(m, n)
def α (m n : ℕ) : ℕ :=
  (List.range (Nat.log2 m + 1)).count (λ k, (m / 2^k) % 2 = 1 ∧ (n / 2^k) % 2 = 1)

-- Define the matrix M
noncomputable def M (i j : ℕ) : ℤ :=
  (-1) ^ α (i - 1) (j - 1)

-- Define the modified entry for matrix M
def a (i j : ℕ) : ℤ :=
  (1 - 2^2017) * M (i-1) (j-1)

-- Define matrix A
noncomputable def A (i j : ℕ) : ℤ :=
  a i j

-- Define det and det mod 2017
noncomputable def detA : ℤ := Matrix.det (Matrix.tabulate (2^2017) (2^2017) A)

-- Proof statement
theorem det_A_mod_2017 : (detA % 2017) = 1382 :=
by
  sorry

end det_A_mod_2017_l371_371836


namespace probability_absolute_value_less_than_1_96_l371_371876

noncomputable def standardNormalDistribution : Type :=
  sorry   -- (this is a placeholder for the actual definition of the standard normal distribution)

axiom standard_normal_symmetry {ξ : standardNormalDistribution} :
  ∀ ξ, P(ξ > 1.96) = 0.025

axiom probability_less_than_neg_1_96 {ξ : standardNormalDistribution} :
  P(ξ < -1.96) = 0.025

theorem probability_absolute_value_less_than_1_96
  (ξ : standardNormalDistribution)
  (h1 : P(ξ < -1.96) = 0.025)
  (h2 : ∀ ξ, P(ξ > 1.96) = 0.025) :
  P(|ξ| < 1.96) = 0.95 :=
by
  sorry   -- This is where the actual proof would go.

end probability_absolute_value_less_than_1_96_l371_371876


namespace find_m_value_l371_371113

theorem find_m_value (m : ℝ) :
  let A := { x : ℝ | x^2 - 3 * x + 2 = 0 },
      C := { x : ℝ | x^2 - m * x + 2 = 0 } in
  A ∩ C = C → (m = 3 ∨ -2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) :=
by
  intro h
  sorry

end find_m_value_l371_371113


namespace Dima_floor_l371_371441

theorem Dima_floor (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 9)
  (h2 : 60 = (n - 1))
  (h3 : 70 = (n - 1) / (n - 1) * 60 + (n - n / 2) * 2 * 60)
  (h4 : ∀ m : ℕ, 1 ≤ m ∧ m ≤ 9 → (5 * n = 6 * m + 1) → (n = 7 ∧ m = 6)) :
  n = 7 :=
by
  sorry

end Dima_floor_l371_371441


namespace fraction_milk_in_mug1_is_one_fourth_l371_371195

-- Condition definitions
def initial_tea_mug1 := 6 -- ounces
def initial_milk_mug2 := 6 -- ounces
def tea_transferred_mug1_to_mug2 := initial_tea_mug1 / 3
def tea_remaining_mug1 := initial_tea_mug1 - tea_transferred_mug1_to_mug2
def total_liquid_mug2 := initial_milk_mug2 + tea_transferred_mug1_to_mug2
def portion_transferred_back := total_liquid_mug2 / 4
def tea_ratio_mug2 := tea_transferred_mug1_to_mug2 / total_liquid_mug2
def milk_ratio_mug2 := initial_milk_mug2 / total_liquid_mug2
def tea_transferred_back := portion_transferred_back * tea_ratio_mug2
def milk_transferred_back := portion_transferred_back * milk_ratio_mug2
def final_tea_mug1 := tea_remaining_mug1 + tea_transferred_back
def final_milk_mug1 := milk_transferred_back
def final_total_liquid_mug1 := final_tea_mug1 + final_milk_mug1

-- Lean statement of the problem
theorem fraction_milk_in_mug1_is_one_fourth :
  final_milk_mug1 / final_total_liquid_mug1 = 1 / 4 :=
by
  sorry

end fraction_milk_in_mug1_is_one_fourth_l371_371195


namespace smallest_possible_class_size_l371_371930

theorem smallest_possible_class_size : 
  ∀ n : ℕ, 5 * n + 2 > 40 → ∃ T : ℕ, T = 5 * 8 + 2 ∧ T = 42 :=
by
  intro n
  intro h
  use 42
  rw [← nat.succ_eq_add_one, nat.succ_mul, nat.mul_succ]
  sorry

end smallest_possible_class_size_l371_371930


namespace donation_to_second_home_l371_371262

-- Definitions of the conditions
def total_donation := 700.00
def first_home_donation := 245.00
def third_home_donation := 230.00

-- Define the unknown donation to the second home
noncomputable def second_home_donation := total_donation - first_home_donation - third_home_donation

-- The theorem to prove
theorem donation_to_second_home :
  second_home_donation = 225.00 :=
by sorry

end donation_to_second_home_l371_371262


namespace largest_rectangle_area_l371_371685

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l371_371685


namespace condition_is_sufficient_but_not_necessary_l371_371091

variable (P Q : Prop)

theorem condition_is_sufficient_but_not_necessary :
    (P → Q) ∧ ¬(Q → P) :=
sorry

end condition_is_sufficient_but_not_necessary_l371_371091


namespace value_of_z_l371_371032

theorem value_of_z (z y : ℝ) (h1 : (12)^3 * z^3 / 432 = y) (h2 : y = 864) : z = 6 :=
by
  sorry

end value_of_z_l371_371032


namespace largest_prime_factor_of_4620_l371_371726

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / m → ¬ (m ∣ n)

def prime_factors (n : ℕ) : List ℕ :=
  -- assumes a well-defined function that generates the prime factor list
  -- this is a placeholder function for demonstrating purposes
  sorry

def largest_prime_factor (l : List ℕ) : ℕ :=
  l.foldr max 0

theorem largest_prime_factor_of_4620 : largest_prime_factor (prime_factors 4620) = 11 :=
by
  sorry

end largest_prime_factor_of_4620_l371_371726


namespace trig_identity_part1_trig_identity_part2_l371_371744
-- Importing the entire Mathlib library for necessary support

-- Noncomputable context since we are dealing with real numbers and trigonometric functions
noncomputable theory

-- Part 1 Lean statement
theorem trig_identity_part1 (a : ℝ) (h1 : π / 2 < a ∧ a < π) (h2 : Real.sin (π - a) = 4 / 5) :
  (Real.sin (2 * π + a) * Real.tan (π - a) * Real.cos (-π - a)) / 
  (Real.sin (3 * π / 2 - a) * Real.cos (π / 2 + a)) = -4 / 3 := 
sorry

-- Part 2 Lean statement
theorem trig_identity_part2 (θ : ℝ) (h : ∃ x : ℝ, x = θ ∧ sin θ = -2 * cos θ) :
  (1 + Real.sin (2 * θ) - Real.cos (2 * θ)) / 
  (1 + Real.sin (2 * θ) + Real.cos (2 * θ)) = -2 := 
sorry

end trig_identity_part1_trig_identity_part2_l371_371744


namespace min_consecutive_bullseyes_to_win_l371_371430

noncomputable def chelsea_secures_victory
  (bullseye_points : ℕ)
  (other_scores : Set ℕ)
  (chelsea_lead : ℕ)
  (total_shots : ℕ)
  (minimum_score : ℕ)
  (opponent_max_points_per_shot : ℕ)
  : ℕ := 
let k := chelsea_lead + (total_shots / 2) * minimum_score in
let opponent_max_possible_score := k + 540 in
let chelsea_total_score (n : ℕ) := k + 10 * n + 3 * (total_shots / 2 - n) + 180 in
let win_condition := chelsea_total_score(52) > opponent_max_possible_score in
52

theorem min_consecutive_bullseyes_to_win : 
  chelsea_secures_victory 10 {0, 1, 3, 7} 60 120 3 10 = 52 
:= 
by
  -- Actual mathematical proof showing that Chelsea needs at least 52 bullseyes to guarantee victory
  sorry

end min_consecutive_bullseyes_to_win_l371_371430


namespace volume_of_pyramid_l371_371485

theorem volume_of_pyramid (S A B C H : ℝ×ℝ×ℝ) 
(base_eq_tri : ∃ x, (S, A, B, C) forms_equilateral_triangle x) 
(h_proj : orthocenter_of_triangle H S B C A) 
(dihedral_angle : ∡HABC = 30) 
(sa_length : sa_dist S A = 2 * √3) : 
  volume_of_pyramid S A B C = (9 * √3) / 4 := 
by sorry

end volume_of_pyramid_l371_371485


namespace initial_friends_count_l371_371295

variable (F : ℕ)
variable (players_quit : ℕ)
variable (lives_per_player : ℕ)
variable (total_remaining_lives : ℕ)

theorem initial_friends_count
  (h1 : players_quit = 7)
  (h2 : lives_per_player = 8)
  (h3 : total_remaining_lives = 72) :
  F = 16 :=
by
  have h4 : 8 * (F - 7) = 72 := by sorry   -- Derived from given conditions
  have : 8 * F - 56 = 72 := by sorry        -- Simplify equation
  have : 8 * F = 128 := by sorry           -- Add 56 to both sides
  have : F = 16 := by sorry                -- Divide both sides by 8
  exact this                               -- Final result

end initial_friends_count_l371_371295


namespace probability_product_multiple_of_15_l371_371924

/-- Define the set of numbers. -/
def S := {5, 6, 9, 10}

/-- Define what it means for a product to be a multiple of 15. -/
def multiple_of_15 (a b : ℕ) : Prop := (a * b) % 15 = 0

/-- Define the pairs without replacement from the set S. -/
def pairs : set (ℕ × ℕ) :=
  { (a, b) | a ∈ S ∧ b ∈ S ∧ a ≠ b }

/-- The probability that the product of two numbers randomly chosen without replacement from S is a multiple of 15 is 2/3. -/
theorem probability_product_multiple_of_15 : 
  (card {p ∈ pairs | multiple_of_15 p.1 p.2}).toReal / (card pairs).toReal = 2 / 3 :=
sorry

end probability_product_multiple_of_15_l371_371924


namespace even_dimension_needed_for_odd_neighbors_l371_371011

theorem even_dimension_needed_for_odd_neighbors (m n : ℕ) :
  (∀ i j, i < m → j < n →
    (∃ color : bool, ∃ adjacency_count : ℕ, (adjacency_count % 2 = 1) ∧
      -- Each cell (i, j) is either color true (black) or false (white) and
      -- adjacency_count is the count of its adjacent cells sharing the same color.
      ∃ neighbors : list (ℕ × ℕ), 
        (∀ (a b : ℕ), (a, b) ∈ neighbors → (a = i ∧ b = j ∨ b = j ∧ ((a = i - 1 ∨ a = i + 1) ∧ a < m ∧ a ≥ 0)) ∨
                        (a = i ∧ (b = j - 1 ∨ b = j + 1) ∧ b < n ∧ b ≥ 0)) ∧
        (∃ same_color_count : ℕ, same_color_count = neighbors.filter (λ x, x.snd = color).length))) →
  (even m ∨ even n) :=
by
  intro h
  sorry

end even_dimension_needed_for_odd_neighbors_l371_371011


namespace sample_size_l371_371372

theorem sample_size
  (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (prob : ℝ)
  (h_freshmen : freshmen = 400)
  (h_sophomores : sophomores = 320)
  (h_juniors : juniors = 280)
  (h_prob : prob = 0.2) :
  let total_students := freshmen + sophomores + juniors in
  let sample_size := total_students * prob in
  sample_size = 200 :=
by
  sorry

end sample_size_l371_371372


namespace polar_to_rectangular_coordinates_l371_371796

theorem polar_to_rectangular_coordinates (r θ : ℝ) (hr : r = 5) (hθ : θ = (3 * Real.pi) / 2) :
    (r * Real.cos θ, r * Real.sin θ) = (0, -5) :=
by
  rw [hr, hθ]
  simp [Real.cos, Real.sin]
  sorry

end polar_to_rectangular_coordinates_l371_371796


namespace pow_div_l371_371321

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l371_371321


namespace andy_coats_l371_371952

-- Define the initial number of minks Andy buys
def initial_minks : ℕ := 30

-- Define the number of babies each mink has
def babies_per_mink : ℕ := 6

-- Define the total initial minks including babies
def total_initial_minks : ℕ := initial_minks * babies_per_mink + initial_minks

-- Define the number of minks set free by activists
def minks_set_free : ℕ := total_initial_minks / 2

-- Define the number of minks remaining after half are set free
def remaining_minks : ℕ := total_initial_minks - minks_set_free

-- Define the number of mink skins needed for one coat
def mink_skins_per_coat : ℕ := 15

-- Define the number of coats Andy can make
def coats_andy_can_make : ℕ := remaining_minks / mink_skins_per_coat

-- The theorem to prove the number of coats Andy can make
theorem andy_coats : coats_andy_can_make = 7 := by
  sorry

end andy_coats_l371_371952


namespace sum_neg_one_powers_l371_371308

theorem sum_neg_one_powers : (∑ k in finset.range 2010, (-1)^(k + 1)) = 0 :=
by
  sorry

end sum_neg_one_powers_l371_371308


namespace inscribe_2n_gon_l371_371839

-- Define the problem conditions in Lean:

variables (n : ℕ) (n_gt_0 : n > 0)
-- n is a natural number, and it is greater than 0

def lines : list (ℝ → ℝ) := sorry -- Placeholder for the list of \(2n-1\) lines
def circle_center : ℝ × ℝ := (0, 0) -- Assuming the circle is centered at origin for simplification
def circle_radius : ℝ := 1 -- Assuming radius of the circle is 1 for simplification
def point_K : ℝ × ℝ := sorry -- Placeholder for point K inside the circle

-- The main theorem to prove

theorem inscribe_2n_gon 
  (lines : list (ℝ → ℝ)) -- \(2n-1\) lines given on the plane
  (h_len_lines : lines.length = 2 * n - 1) -- given lines are exactly \(2n-1\)
  (circle_center : ℝ × ℝ) -- the coordinates of the circle center
  (circle_radius : ℝ) --  the radius of circle
  (point_K : ℝ × ℝ) -- Point K inside the circle
  (h_inside : (point_K.1 - circle_center.1)^2 + (point_K.2 - circle_center.2)^2 < circle_radius^2): -- K inside the circle
  ∃ polygon : list (ℝ × ℝ), -- The polygon vertices on the plane
    (polygon.length = 2 * n) ∧ -- polygon is a \(2n\)-gon
    (∃ i, 1 ≤ i ∧ i ≤ 2 * n ∧ (polygon.nth (i % (2 * n)) = point_K ∨ polygon.nth ((i-1) % (2 * n)) = point_K)) ∧ -- one side passes through K
    ∀ j, (1 ≤ j ∧ j < 2 * n) → parallel_to lines (polygon.nth (j % (2 * n))) (polygon.nth ((j+1) % (2 * n))) -- sides are parallel to the lines
      parse sorry -- Placeholder for parsing parallel_to
  
-- Placeholder for the function that checks if two line segments are parallel to any of the given lines.
def parallel_to : list (ℝ → ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry


end inscribe_2n_gon_l371_371839


namespace maximum_value_is_sqrt2_plus_1_div_2_l371_371474

noncomputable def maximum_value (a b : ℝ) : ℝ :=
  1 / (a^2 + 1) + 1 / (b^2 + 1)

theorem maximum_value_is_sqrt2_plus_1_div_2 (a b : ℝ) 
  (h : a + b = 2) : 
  maximum_value a b ≤ (√2 + 1) / 2 :=
sorry

end maximum_value_is_sqrt2_plus_1_div_2_l371_371474


namespace fraction_simplest_sum_l371_371672

theorem fraction_simplest_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (3975 : ℚ) / 10000 = (a : ℚ) / b) 
  (simp : ∀ (c : ℕ), c ∣ a ∧ c ∣ b → c = 1) : a + b = 559 :=
sorry

end fraction_simplest_sum_l371_371672


namespace find_number_l371_371537

theorem find_number (X : ℝ) (h : 0.8 * X = 0.7 * 60.00000000000001 + 30) : X = 90.00000000000001 :=
sorry

end find_number_l371_371537


namespace hunter_cannot_see_rabbit_l371_371781

-- Define the necessary concepts and conditions
structure LatticeConfig where
  r : ℝ
  h : ℝ -- the distance from the hunter to the rabbit
  hunter_pos : ℤ × ℤ -- position of the hunter on the integer lattice
  ∀ (p : ℤ × ℤ), p ≠ hunter_pos → tree_trunk_radius = r

theorem hunter_cannot_see_rabbit (config : LatticeConfig) 
  (h_pos : config.h > 1 / config.r) : 
  sorry

end hunter_cannot_see_rabbit_l371_371781


namespace solve_for_a_l371_371998

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l371_371998


namespace total_length_of_sticks_l371_371587

-- Definitions of stick lengths based on the conditions
def length_first_stick : ℕ := 3
def length_second_stick : ℕ := 2 * length_first_stick
def length_third_stick : ℕ := length_second_stick - 1

-- Proof statement
theorem total_length_of_sticks : length_first_stick + length_second_stick + length_third_stick = 14 :=
by
  sorry

end total_length_of_sticks_l371_371587


namespace cricketers_total_score_l371_371756

def runs_from_boundaries (boundaries : ℕ) : ℕ := boundaries * 4
def runs_from_sixes (sixes : ℕ) : ℕ := sixes * 6
def runs_from_running (total_score : ℝ) (running_percentage : ℝ) : ℝ := total_score * running_percentage

def total_runs (boundaries : ℕ) (sixes : ℕ) (total_score : ℝ) (running_percentage : ℝ) : ℝ :=
runs_from_boundaries boundaries + runs_from_sixes sixes + runs_from_running total_score running_percentage

theorem cricketers_total_score :
  ∀ (boundaries sixes : ℕ) (running_percentage : ℝ), 
  boundaries = 12 → 
  sixes = 2 → 
  running_percentage = 0.5774647887323944 →
  let T := 60 / (1 - running_percentage) in 
  T = 142 :=
by
  intros boundaries sixes running_percentage hb hs hr
  simp [hb, hs, hr]
  have T_def: T = 60 / (1 - running_percentage) := rfl
  rw T_def
  linarith
-- Sorry

end cricketers_total_score_l371_371756


namespace total_number_of_houses_l371_371169

theorem total_number_of_houses (G P GP N T : ℕ) 
  (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) 
  (hT : T = G + P - GP + N) : T = 65 :=
by 
  -- entering the given values and assumptions
  rw [hG, hP, hGP, hN] at hT
  -- calculating and simplifying
  calc 
    T = 50 + 40 - 35 + 10 : by rw hT
    ... = 90 - 35 + 10 : by norm_num
    ... = 55 + 10 : by norm_num
    ... = 65 : by norm_num

end total_number_of_houses_l371_371169


namespace magnitude_of_BC_l371_371488

noncomputable def triangle_area (AB AC : ℝ) (sin_BAC : ℝ) : ℝ :=
  0.5 * AB * AC * sin_BAC

noncomputable def magnitude (AB AC : ℝ) (cos_BAC : ℝ) : ℝ :=
  Real.sqrt (AC^2 + AB^2 - 2 * AC * AB * cos_BAC)

theorem magnitude_of_BC
  (S_ABC : ℝ) (AB AC : ℝ) (dot_lt_zero : (AB * AC * Real.cos ∠ABC) < 0)
  (h_S_ABC : S_ABC = 15 * Real.sqrt 3 / 4)
  (h_AB : AB = 3)
  (h_AC : AC = 5) :
  magnitude AB AC (-1/2) = 7 :=
by
  have h_sin := (15 * Real.sqrt 3) / (4 * (Real.sin ∠ABC));
  have cos_BAC := -1/2;
  sorry

end magnitude_of_BC_l371_371488


namespace min_policemen_needed_l371_371374

-- Definitions of the problem parameters
def city_layout (n m : ℕ) := n > 0 ∧ m > 0

-- Function to calculate the minimum number of policemen
def min_policemen (n m : ℕ) : ℕ := (m - 1) * (n - 1)

-- The theorem to prove
theorem min_policemen_needed (n m : ℕ) (h : city_layout n m) : min_policemen n m = (m - 1) * (n - 1) :=
by
  unfold city_layout at h
  unfold min_policemen
  sorry

end min_policemen_needed_l371_371374


namespace james_fence_problem_l371_371194

theorem james_fence_problem (w : ℝ) (hw : 0 ≤ w) (h_area : w * (2 * w + 10) ≥ 120) : w = 5 :=
by
  sorry

end james_fence_problem_l371_371194


namespace a_eq_3_suff_not_nec_l371_371535

theorem a_eq_3_suff_not_nec (a : ℝ) : (a = 3 → a^2 = 9) ∧ (a^2 = 9 → ∃ b : ℝ, b = a ∧ (b = 3 ∨ b = -3)) :=
by
  sorry

end a_eq_3_suff_not_nec_l371_371535


namespace find_original_number_l371_371391

-- Given definitions and conditions
def doubled_add_nine (x : ℝ) : ℝ := 2 * x + 9
def trebled (y : ℝ) : ℝ := 3 * y

-- The proof problem we need to solve
theorem find_original_number (x : ℝ) (h : trebled (doubled_add_nine x) = 69) : x = 7 := 
by sorry

end find_original_number_l371_371391


namespace f_at_1_l371_371513

def f (x : ℝ) : ℝ := Real.logb 3 (8 * x + 1)

theorem f_at_1 : f 1 = 2 := 
by 
  -- Sorry to denote a missing proof
  sorry

end f_at_1_l371_371513


namespace part1_part2_l371_371471

-- Definitions as per the conditions
def A (a b : ℚ) := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) := - a^2 + (1/2) * a * b + 2 / 3

-- Part (1)
theorem part1 (a b : ℚ) (h1 : a = -1) (h2 : b = -2) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3 := 
by 
  sorry

-- Part (2)
theorem part2 (a : ℚ) : 
  (∀ a : ℚ, 4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3) → 
  b = 1/2 :=
by 
  sorry

end part1_part2_l371_371471


namespace solve_fx_eq_1_div_4_l371_371850

noncomputable def f : ℝ → ℝ
| x := if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

theorem solve_fx_eq_1_div_4 :
  ∃! x : ℝ, (x > 1) ∧ f x = 1 / 4 :=
by
  sorry

end solve_fx_eq_1_div_4_l371_371850


namespace problem_1_problem_2_l371_371607

def y (x : ℝ) : ℝ := x + sorry -- Complete the definition of y

def f (n : ℕ) (x : ℝ) : ℝ := (x ^ 10, sorry) -- Complete the definition of f based on the given criteria

theorem problem_1 (n : ℕ) (x : ℝ) (hn : n > 1) : 
  f (n + 1) x = y x * f n x - f (n - 1) x := 
sorry -- Proof goes here

theorem problem_2 (n : ℕ) (x : ℝ) : 
  ∀ n > 0, f n x = (y x) ^ n - sorry := -- Complete the inductive definition
sorry -- Proof goes here

end problem_1_problem_2_l371_371607


namespace bella_total_roses_l371_371772

-- Define the constants and conditions
def dozen := 12
def roses_from_parents := 2 * dozen
def friends := 10
def roses_per_friend := 2
def total_roses := roses_from_parents + (roses_per_friend * friends)

-- Prove that the total number of roses Bella received is 44
theorem bella_total_roses : total_roses = 44 := 
by
  sorry

end bella_total_roses_l371_371772


namespace prob_shooting_A_first_l371_371298

-- Define the probabilities
def prob_A_hits : ℝ := 0.4
def prob_A_misses : ℝ := 0.6
def prob_B_hits : ℝ := 0.6
def prob_B_misses : ℝ := 0.4

-- Define the overall problem
theorem prob_shooting_A_first (k : ℕ) (ξ : ℕ) (hξ : ξ = k) :
  ((prob_A_misses * prob_B_misses)^(k-1)) * (1 - (prob_A_misses * prob_B_misses)) = 0.24^(k-1) * 0.76 :=
by
  -- Placeholder for proof
  sorry

end prob_shooting_A_first_l371_371298


namespace range_of_m_l371_371095

noncomputable def f : ℝ → ℝ
  | x := if x ≤ 1 then 2^x + 1 else 1 - Real.log x / Real.log 2

theorem range_of_m (m : ℝ) : 
  f (1 - m^2) > f (2*m - 2) ↔ m ∈ set.Ioo (-3 : ℝ) 1 ∪ set.Ioi (3/2) :=
sorry

end range_of_m_l371_371095


namespace identify_counterfeit_coin_in_two_weighings_l371_371949

theorem identify_counterfeit_coin_in_two_weighings (coins : Fin 4 → ℝ) (counterfeit : Fin 4 → Prop) :
  (∃ i, counterfeit i ∧ (∀ j, j ≠ i → coins j = coins (Fin.succ j))) →
  ∃ i, counterfeit i ∧ coins_in_two_weighings coins counterfeit i :=
by
  sorry

end identify_counterfeit_coin_in_two_weighings_l371_371949


namespace find_a_l371_371106

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a / x

theorem find_a (a : ℝ) (h: ∀ x : ℝ, y = x^2 + a / x) (h1 : deriv (λ x, f x a) 1 = 2) :
  a = 0 :=
sorry

end find_a_l371_371106


namespace smallest_n_l371_371345

theorem smallest_n (n : ℕ) (h_n_gt_1 : n > 1) :
  (∀ n > 1, 
    (n ≡ 1 [MOD 2]) ∧
    (n ≡ 1 ∨ n ≡ -1 [MOD 3]) ∧
    (n ≡ 1 [MOD 4]) ∧
    (n ≡ 1 [MOD 5]) ∧
    (n ≡ 1 [MOD 6]) ∧
    (n ≡ 1 [MOD 7]) ∧
    (n ≡ 1 [MOD 8]) ∧
    (n ≡ 1 [MOD 9]) ∧
    (n ≡ 1 [MOD 10])) → n = 2521 :=
sorry

end smallest_n_l371_371345


namespace find_a_from_perpendicular_lines_l371_371873

theorem find_a_from_perpendicular_lines (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 := 
by 
  sorry

end find_a_from_perpendicular_lines_l371_371873


namespace blue_line_length_correct_l371_371017

def white_line_length : ℝ := 7.67
def difference_in_length : ℝ := 4.33
def blue_line_length : ℝ := 3.34

theorem blue_line_length_correct :
  white_line_length - difference_in_length = blue_line_length :=
by
  sorry

end blue_line_length_correct_l371_371017


namespace max_T_n_attained_at_n0_l371_371976

-- Define the geometric sequence with common ratio sqrt(2)
def geom_seq (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * (sqrt 2) ^ n

-- Define the sum of the first n terms of the geometric sequence
def S_n (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * (1 - (sqrt 2) ^ n) / (1 - sqrt 2)

-- Define T_n given S_n and the geometric sequence
def T_n (a₁ : ℝ) (n : ℕ) : ℝ := (17 * S_n a₁ n - S_n a₁ (2 * n)) / (geom_seq a₁ (n + 1))

-- Define the problem statement
theorem max_T_n_attained_at_n0 :
  ∀ (a₁ : ℝ), ∃ n_0 : ℕ, T_n a₁ n_0 = Real.sup (λ n, T_n a₁ n) ∧ n_0 = 4 :=
by
  sorry

end max_T_n_attained_at_n0_l371_371976


namespace minutkin_bedtime_l371_371235

def time_minutkin_goes_to_bed 
    (morning_time : ℕ) 
    (morning_turns : ℕ) 
    (night_turns : ℕ) 
    (morning_hours : ℕ) 
    (morning_minutes : ℕ)
    (hours_per_turn : ℕ) 
    (minutes_per_turn : ℕ) : Nat := 
    ((morning_hours * 60 + morning_minutes) - (night_turns * hours_per_turn * 60 + night_turns * minutes_per_turn)) % 1440 

theorem minutkin_bedtime : 
    time_minutkin_goes_to_bed 9 9 11 8 30 1 12 = 1290 :=
    sorry

end minutkin_bedtime_l371_371235


namespace total_fruits_eaten_l371_371593

theorem total_fruits_eaten (apples_last_night: ℕ) (banana_last_night: ℕ) (oranges_last_night: ℕ)
  (apples_today_more: ℕ) (banana_today_factor: ℕ) (oranges_today_factor: ℕ):
  apples_last_night = 3 →
  banana_last_night = 1 →
  oranges_last_night = 4 →
  apples_today_more = 4 →
  banana_today_factor = 10 →
  oranges_today_factor = 2 →
  ((apples_last_night + apples_today_more) + 
   (apples_today_more * banana_today_factor) + 
   (apples_today_more * oranges_today_factor)) + 
  (apples_last_night + banana_last_night + oranges_last_night) = 39 := 
begin
  sorry
end

end total_fruits_eaten_l371_371593


namespace sum_of_roots_eq_2007_l371_371031

noncomputable def P (x : ℝ) : ℝ := 
  (x - 1)^2009 + 2 * (x - 2)^2008 + 3 * (x - 3)^2007 + 
  ∑ i in finset.range 2006, (i + 4) * (x - (i + 4))^(2006 - i) + 
  2008 * (x - 2008)^2 + 2009 * (x - 2009)

theorem sum_of_roots_eq_2007 : 
  P(x) = 0 → (∑ root in (polynomial.roots P), root) = 2007 :=
sorry

end sum_of_roots_eq_2007_l371_371031


namespace weeding_planting_support_l371_371010

-- Definitions based on conditions
def initial_weeding := 31
def initial_planting := 18
def additional_support := 20

-- Let x be the number of people sent to support weeding.
variable (x : ℕ)

-- The equation to prove.
theorem weeding_planting_support :
  initial_weeding + x = 2 * (initial_planting + (additional_support - x)) :=
sorry

end weeding_planting_support_l371_371010


namespace matrix_vector_multiplication_l371_371213

variable (N : Matrix (Fin 2) (Fin 2) ℝ)

noncomputable def vec1 : Vector ℝ 2 := ![1, 3]
noncomputable def vec2 : Vector ℝ 2 := ![-2, 4]
noncomputable def vec3 : Vector ℝ 2 := ![3, 11]
noncomputable def result1 : Vector ℝ 2 := ![2, 5]
noncomputable def result2 : Vector ℝ 2 := ![3, 1]
noncomputable def result3 : Vector ℝ 2 := ![7.4, 17.2]

theorem matrix_vector_multiplication:
  N.mulVec vec1 = result1 →
  N.mulVec vec2 = result2 →
  N.mulVec vec3 = result3 :=
by
  intros h1 h2
  sorry

end matrix_vector_multiplication_l371_371213


namespace janes_mean_score_l371_371956

theorem janes_mean_score :
  let scores := [98, 97, 92, 85, 93, 88, 82]
  let sum_scores := scores.sum
  let count_scores := scores.length
  (sum_scores.toFloat / count_scores.toFloat).round = 90.71 :=
by
  let scores := [98, 97, 92, 85, 93, 88, 82]
  let sum_scores := scores.sum
  let count_scores := scores.length
  have mean_score := (sum_scores.toFloat / count_scores.toFloat).round
  have expected_mean := 90.71
  show mean_score = expected_mean from sorry

end janes_mean_score_l371_371956


namespace frac_sum_is_one_l371_371844

theorem frac_sum_is_one (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end frac_sum_is_one_l371_371844


namespace librarian_books_taken_l371_371251

theorem librarian_books_taken (total_books books_per_shelf number_of_shelves books_taken books_used : Nat) 
  (h1 : total_books = 14)
  (h2 : books_per_shelf = 3)
  (h3 : number_of_shelves = 4)
  (h4 : books_used = books_per_shelf * number_of_shelves)
  (h5 : books_taken = total_books - books_used) : 
  books_taken = 2 :=
by
  rw [h1, h2, h3] at h4,
  rw [h1, h4] at h5,
  norm_num at h5,
  exact h5

end librarian_books_taken_l371_371251


namespace eight_pow_15_div_sixtyfour_pow_6_l371_371317

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l371_371317


namespace power_mod_eight_l371_371729

theorem power_mod_eight :
  (5 : ℤ) ≡ 5 [MOD 8] → 
  (5^2 : ℤ) ≡ 1 [MOD 8] → 
  (5^1082 : ℤ) ≡ 1 [MOD 8]:=
by
  intros h1 h2
  sorry

end power_mod_eight_l371_371729


namespace integral_f_eq_34_l371_371047

noncomputable def f (x : ℝ) := if x ∈ [0, 1] then (1 / Real.pi) * Real.sqrt (1 - x^2) else 2 - x

theorem integral_f_eq_34 :
  ∫ x in (0 : ℝ)..2, f x = 3 / 4 :=
by
  sorry

end integral_f_eq_34_l371_371047


namespace volume_of_water_correct_submerged_surface_area_correct_l371_371757

noncomputable def cylindrical_tank_radius : ℝ := 5 -- feet
noncomputable def cylindrical_tank_height : ℝ := 10 -- feet
noncomputable def water_height : ℝ := 3 -- feet

/-- Calculate the volume of water in the tank given the radius, height, and water height. -/
def volume_of_water_in_tank (r h wh : ℝ) : ℝ :=
  let theta := 2 * Real.arccos(2 / r)
  let sector_area := theta / (2 * Real.pi) * (Real.pi * r ^ 2)
  let triangle_area := 2 * r * Real.sqrt (r ^ 2 - (2 : ℝ) ^ 2)
  h * (sector_area - triangle_area)

theorem volume_of_water_correct :
  volume_of_water_in_tank cylindrical_tank_radius cylindrical_tank_height water_height = 290.7 * Real.pi - 40 * Real.sqrt 6 :=
sorry

/-- Calculate the submerged surface area of the cylindrical side. -/
def submerged_surface_area (r h wh : ℝ) : ℝ :=
  h * (2 * r * Real.sin (Real.arccos (2 / r)))

theorem submerged_surface_area_correct :
  submerged_surface_area cylindrical_tank_radius cylindrical_tank_height water_height = 91.5 :=
sorry

end volume_of_water_correct_submerged_surface_area_correct_l371_371757


namespace Amanda_final_quiz_score_to_get_A_l371_371774

theorem Amanda_final_quiz_score_to_get_A (q_avg : ℚ) (q1 q2 q3 q4 : ℚ) (needed_avg : ℚ) :
  q1 = 92 → q2 = 92 → q3 = 92 → q4 = 92 →
  q_avg = (q1 + q2 + q3 + q4) / 4 →
  needed_avg = 90 →
  ∃ q5 : ℚ, (q1 + q2 + q3 + q4 + q5) / 5 ≥ needed_avg ∧ q5 ≥ 82 :=
by {
  intros hq1 hq2 hq3 hq4 hq_avg h_needed_avg,
  use 82,
  split,
  {
    apply h_needed_avg,
    sorry,
  },
  {
    sorry,
  }
}

end Amanda_final_quiz_score_to_get_A_l371_371774


namespace binomial_arithmetic_progression_rational_terms_l371_371882

open Nat

theorem binomial_arithmetic_progression (x : ℝ) (n : ℕ) (h1 : n < 15) (h2 : n > 0)
  (h3 : binomialCoeff n 8 + binomialCoeff n 10 = 2 * binomialCoeff n 9) : n = 14 :=
  sorry

theorem rational_terms (x : ℝ) (n : ℕ) (h : n = 14) :
  (∀ r, r ∈ [0, 6, 12] → (C 14 r * x ^ (7 - r / 6)).isRational) :=
  sorry

end binomial_arithmetic_progression_rational_terms_l371_371882


namespace label_possible_iff_even_l371_371079

open Finset

variable {B : Type*} (A : Fin B → Finset B) (n : ℕ)

def satisfies_conditions (A : Fin (2 * n + 1) → Finset B) (n : ℕ) : Prop :=
  (∀ i, (A i).card = 2 * n) ∧
  (∀ i j, i < j → (A i ∩ A j).card = 1) ∧
  (∀ b ∈ ⋃ i, A i, (filter (λ i, b ∈ A i) (range (2 * n + 1))).card ≥ 2)

theorem label_possible_iff_even
  (h : ∃ (n : ℕ) (A : Fin (2 * n + 1) → Finset B), 
    satisfies_conditions A n) :
  ∀ {n : ℕ}, (∃ (A : Fin (2 * n + 1) → Finset B), satisfies_conditions A n) → 
  (∃ (f : B → Fin 2), ∀ i, (A i).filter (λ b, f b = 0).card = n) ↔ Even n :=
sorry

end label_possible_iff_even_l371_371079


namespace perpendicular_unit_vector_exists_l371_371052

theorem perpendicular_unit_vector_exists :
  ∃ (m n : ℝ), (2 * m + n = 0) ∧ (m^2 + n^2 = 1) ∧ (m = (Real.sqrt 5) / 5) ∧ (n = -(2 * (Real.sqrt 5)) / 5) :=
by
  sorry

end perpendicular_unit_vector_exists_l371_371052


namespace count_int_values_not_satisfying_inequality_l371_371835

theorem count_int_values_not_satisfying_inequality :
  (finset.card (finset.filter (λ x : ℤ, 3 * x^2 + 11 * x + 15 ≤ 18) (finset.Icc (-3 : ℤ) 0))) = 3 :=
sorry

end count_int_values_not_satisfying_inequality_l371_371835


namespace school_band_fundraising_l371_371707

-- Definitions
def goal : Nat := 150
def earned_from_three_families : Nat := 10 * 3
def earned_from_fifteen_families : Nat := 5 * 15
def total_earned : Nat := earned_from_three_families + earned_from_fifteen_families
def needed_more : Nat := goal - total_earned

-- Theorem stating the problem in Lean 4
theorem school_band_fundraising : needed_more = 45 := by
  sorry

end school_band_fundraising_l371_371707


namespace constant_term_expansion_l371_371848

-- Define the condition
def a : ℝ := ∫ x in 0..2, (2 * x - 1)

-- State the main theorem
theorem constant_term_expansion (a = ∫ x in 0..2, (2 * x - 1)) : 
  constant_term (expand_binomial_power (x + a / x) 4) = 24 :=
sorry

end constant_term_expansion_l371_371848


namespace find_a_from_perpendicular_lines_l371_371874

theorem find_a_from_perpendicular_lines (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 := 
by 
  sorry

end find_a_from_perpendicular_lines_l371_371874


namespace right_triangle_perimeter_l371_371496

theorem right_triangle_perimeter (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : (1/2) * a * b = 4) (h3 : sqrt 2 = (a + b) / c) :
  a + b + c = 4 * sqrt 2 + 4 := 
sorry

end right_triangle_perimeter_l371_371496


namespace evaluate_expression_l371_371813

theorem evaluate_expression :
  let a := 17
  let b := 19
  let c := 23
  let numerator1 := 136 * (1 / b - 1 / c) + 361 * (1 / c - 1 / a) + 529 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  let numerator2 := 144 * (1 / b - 1 / c) + 400 * (1 / c - 1 / a) + 576 * (1 / a - 1 / b)
  (numerator1 / denominator) * (numerator2 / denominator) = 3481 := by
  sorry

end evaluate_expression_l371_371813


namespace miki_pear_juice_l371_371618

def total_pears : ℕ := 18
def total_oranges : ℕ := 10
def pear_juice_per_pear : ℚ := 10 / 2
def orange_juice_per_orange : ℚ := 12 / 3
def max_blend_volume : ℚ := 44

theorem miki_pear_juice : (total_oranges * orange_juice_per_orange = 40) ∧ (max_blend_volume - 40 = 4) → 
  ∃ p : ℚ, p * pear_juice_per_pear = 4 ∧ p = 0 :=
by
  sorry

end miki_pear_juice_l371_371618


namespace IvanushkaWins_l371_371180

structure DuelConditions where
  sources : Fin 10 → ℕ
  accessible : ∀ n, n < 9 → sources n < sources 9
  notAccessible : sources 9 > sources 9
  deadly : ∀ n, n < 10 → sources n + 1
  counteract : ∀ n m, n < m → sources n < sources m

theorem IvanushkaWins (cond : DuelConditions) :
    IvanushkaSurvives ∧ KoscheiDies :=
by
  -- Define Ivanushka survives condition
  def IvanushkaSurvives := 
    cond.counteract 0 9
  -- Define Koschei dies condition
  def KoscheiDies := 
    ¬(cond.deadly 0 9)
  sorry

end IvanushkaWins_l371_371180


namespace smallest_n_for_invariant_digit_sum_l371_371212

def digit_sum (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem smallest_n_for_invariant_digit_sum :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (∀ k ∈ {1, 2, ..., n^2}.to_finset, digit_sum (k * n) = digit_sum n) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (∀ k ∈ {1, 2, ..., m^2}.to_finset, digit_sum (k * m) = digit_sum m) → n ≤ m) :=
  ⟨999, by sorry⟩

end smallest_n_for_invariant_digit_sum_l371_371212


namespace contradictory_properties_l371_371046

structure Square (s : ℝ) := 
  (all_sides_equal : ∀ a b c d : ℝ, a = s ∧ b = s ∧ c = s ∧ d = s)
  (all_angles_right : ∀ A B C D : ℝ, A = 90 ∧ B = 90 ∧ C = 90 ∧ D = 90)
  (diagonals_equal : ∀ d1 d2 : ℝ, d1 = d2)
  (diagonals_perpendicular : ∀ (d1 d2 : ℝ), ∠(d1, d2) = 90)
  (diagonals_bisect_angles : ∀ (A : ℝ), A / 2 = 45)
  (diagonals_bisect_each_other : ∀ (d1 d2 : ℝ), mid(d1) = mid(d2))
  (diagonals_is_symmetry_axes : ∀ d : ℝ, d ∈ axes_of_symmetry)
  (inscribed_circle_radius : ℝ := s / 2)
  (circumscribed_circle_radius : ℝ := (s * sqrt 2) / 2)
  (center_coincides : center_of(inscribed_circle_radius) = center_of(circumscribed_circle_radius))
  (center_of_symmetry : center)
  (midlines_perpendicular : ∀ midline side : ℝ, midline ⊥ side)
  (area : ℝ := s^2)
  (perimeter : ℝ := 4 * s )
  (max_area_per_given_perimeter : ∀ P : ℝ, P = 4 * sqrt (s^2))
  (min_perimeter_per_given_area : ∀ A : ℝ, A = s^2 / 16)

structure Rectangle (a b : ℝ) := 
  (unequal_sides : a ≠ b)
  (all_angles_right : ∀ A B C D : ℝ, A = 90 ∧ B = 90 ∧ C = 90 ∧ D = 90)
  (diagonals_equal : ∀ d1 d2 : ℝ, d1 = d2)
  (diagonals_not_perpendicular : ∀ (d1 d2 : ℝ), ∠(d1, d2) ≠ 90)

structure Rhombus (s : ℝ) := 
  (all_sides_equal : ∀ a b c d : ℝ, a = s ∧ b = s ∧ c = s ∧ d = s)
  (diagonals_perpendicular : ∀ (d1 d2 : ℝ), ∠(d1, d2) = 90)
  (diagonals_not_equal : ∀ d1 d2 : ℝ, d1 ≠ d2)

theorem contradictory_properties {a b s : ℝ} (sq : Square s) (rect : Rectangle a b) (rhomb : Rhombus s) :
  (sq.diagonals_perpendicular → sq.diagonals_equal) →
  ((rect.diagonals_equal → ¬rect.diagonals_perpendicular) ∧ (rhomb.diagonals_perpendicular → ¬rhomb.diagonals_equal)) := 
sorry

end contradictory_properties_l371_371046


namespace range_x1_x2_x3_l371_371228

-- Define the piecewise function
def f : ℝ → ℝ 
| x := if x < 0 then -2*x - 2 else x^2 - 2*x - 1

-- Define the problem statement
theorem range_x1_x2_x3 (x1 x2 x3 : ℝ) (h1 : x1 ≤ x2 ∧ x2 ≤ x3) (h2 : f x1 = f x2 ∧ f x2 = f x3) :
  3/2 ≤ x1 + x2 + x3 ∧ x1 + x2 + x3 < 2 :=
sorry

end range_x1_x2_x3_l371_371228


namespace cannot_reach_goal_state_l371_371350

-- Define the size of the board
def board_size : ℕ := 5

-- Define the initial state of the board (all zeros)
def initial_board (i j : ℕ) (h1 : i > 0 ∧ i ≤ board_size) (h2 : j > 0 ∧ j ≤ board_size) : ℕ := 0

-- Define the update operation on the board
def update_board (board : ℕ → ℕ → ℕ) (i j : ℕ) :=
  λ x y, if (x = i ∧ y = j) ∨ (x = i + 1 ∧ y = j) ∨ (x = i - 1 ∧ y = j) ∨ (x = i ∧ y = j + 1) ∨ (x = i ∧ y = j - 1) then board x y + 1 else board x y

-- Define the target value we want in each cell
def target_value : ℕ := 2012

-- Define the goal state (all cells have the target value)
def goal_state (i j : ℕ) (h1 : i > 0 ∧ i ≤ board_size) (h2 : j > 0 ∧ j ≤ board_size) : ℕ := target_value

-- Now, we provide the main theorem we need to prove
theorem cannot_reach_goal_state :
  ¬(∃ (f : ℕ → ℕ → ℕ → ℕ), 
    (f 0 = initial_board) ∧
    (∀ t i j h1 h2, f (t+1) = update_board (f t i j)) ∧
    (∀ i j h1 h2, (f some_t i j) = goal_state i j)) := sorry

end cannot_reach_goal_state_l371_371350


namespace problem1_solution_problem2_solution_l371_371785

noncomputable def problem1 : ℝ := 2 * real.sqrt 3 * 6 * real.root (12 : ℝ) 6 * 3 * real.root ((3 / 2 : ℝ)) 3
theorem problem1_solution : problem1 = 6 := sorry
 
noncomputable def problem2 : ℝ := real.log10 14 - 2 * real.log10 (7 / 3) + real.log10 7 - real.log10 18
theorem problem2_solution : problem2 = 0 := sorry

end problem1_solution_problem2_solution_l371_371785


namespace decimal_to_binary_13_l371_371002

theorem decimal_to_binary_13 : nat.binary_repr 13 = "1101" :=
sorry

end decimal_to_binary_13_l371_371002


namespace sqrt_ceil_eq_sqrt_sqrt_l371_371942

theorem sqrt_ceil_eq_sqrt_sqrt (a : ℝ) (h : a > 1) : 
  (Int.floor (Real.sqrt (Int.floor (Real.sqrt a)))) = (Int.floor (Real.sqrt (Real.sqrt a))) :=
sorry

end sqrt_ceil_eq_sqrt_sqrt_l371_371942


namespace geometric_and_arithmetic_sequences_l371_371074

theorem geometric_and_arithmetic_sequences (a b c x y : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : 2 * x = a + b)
  (h3 : 2 * y = b + c) :
  (a / x + c / y) = 2 := 
by 
  sorry

end geometric_and_arithmetic_sequences_l371_371074


namespace rate_percent_is_correct_l371_371361

theorem rate_percent_is_correct (P SI T R : ℝ)
  (hP : P = 800) (hSI : SI = 144) (hT : T = 4) : 
  (SI = P * R * T / 100) → R = 4.5 :=
by 
  intro h
  rw [hP, hSI, hT] at h
  linarith

end rate_percent_is_correct_l371_371361


namespace root_interval_l371_371096

def f (x : ℝ) : ℝ := 2^x + (1/4) * x - 5

theorem root_interval (hf : ∀ x y : ℝ, x < y → f(x) < f(y)) :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f(x) = 0 :=
by
  have h1 : f 2 < 0 := by norm_num
  have h2 : f 3 > 0 := by norm_num
  sorry

end root_interval_l371_371096


namespace complex_sum_correct_l371_371970

noncomputable def complex_sequence_sum (a_n b_n : ℕ → ℝ) : ℝ :=
  ∑' n, a_n n * b_n n / 7^n

theorem complex_sum_correct :
  (∀ n: ℕ, (2 + (Complex.i : ℂ))^n = (Complex.ofReal (a_n n)) + (Complex.i * Complex.ofReal (b_n n))) →
  complex_sequence_sum a_n b_n = 7 / 16 :=
by 
  intro h
  sorry

end complex_sum_correct_l371_371970


namespace cos_B_val_sin_A_pi_over_6_val_l371_371166

noncomputable def triangle_properties (A B C a b c : ℝ) (h1 : ∀ x, x = 4*sqrt(2) ∨ x = a ∨ x = c)
  (h2 : ∀ angles sides, ∃ t ∈ angles, t = A ∨ t = B ∨ t = C ∧ 
    ∃ t1 ∈ sides, t1 = a ∨ t1 = b ∨ t1 = c ∧
    ¬(angles = [A, B, C] ∧ sides = [a, b, c]) )

noncomputable def triangle_properties_cos_B (a b c : ℝ) (h : (cos C) / (cos B) = (3 * a - c) / b) : ℝ :=
cos B

noncomputable def triangle_sin_A_pi_over_6 (a b c : ℝ) (h1 : b = 4 * sqrt 2)
  (h2 : a = c) (h3 : cos B = 1 / 3) : ℝ :=
sin (A + π / 6)

-- Questions to be proven
theorem cos_B_val (A B C a b c : ℝ)
  (h1 : ∀ x, x = 4*sqrt(2) ∨ x = a ∨ x = c)
  (h2 : ∀ angles sides, ∃ t ∈ angles, t = A ∨ t = B ∨ t = C ∧ 
    ∃ t1 ∈ sides, t1 = a ∨ t1 = b ∨ t1 = c ∧
    ¬(angles = [A, B, C] ∧ sides = [a, b, c]) )
  (h3 : (cos C) / (cos B) = (3 * a - c) / b)
  : triangle_properties_cos_B a b c h3 = 1 / 3 :=
sorry

theorem sin_A_pi_over_6_val (a b c : ℝ) (h1 : b = 4 * sqrt 2)
  (h2 : a = c) (h3 : cos B = 1 / 3)
  : triangle_sin_A_pi_over_6 a b c h1 h2 h3 = (3 * sqrt 2 + sqrt 3) / 6 :=
sorry

end cos_B_val_sin_A_pi_over_6_val_l371_371166


namespace largest_rectangle_area_l371_371691

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l371_371691


namespace correct_options_l371_371378

variable (P : ℕ → ℚ) -- P is a function from natural numbers to rationals 

-- Define the conditions
axiom P_recurrence : ∀ n ≥ 3, P n = (5/6) * P (n-1) + (5/36) * P (n-2)
axiom P2_value : P 2 = 35/36
axiom P_monotonicity : ∀ n ≥ 2, P n > P (n+1)

-- The theorem that gathers the conditions and correct options
theorem correct_options : 
  P 2 = 35/36 ∧ (∀ n ≥ 2, P n > P (n+1)) ∧ (∀ n ≥ 3, P n = (5/6) * P (n-1) + (5/36) * P (n-2)) :=
by
  split
  -- Subproof for P 2 = 35/36
  sorry

  split
  -- Subproof for ∀ n ≥ 2, P n > P (n+1)
  sorry

  -- Subproof for ∀ n ≥ 3, P n = (5/6) * P (n-1) + (5/36) * P (n-2)
  sorry

end correct_options_l371_371378


namespace sphere_distance_l371_371700

theorem sphere_distance (O A B C : Point) (r : ℝ) (h_r : r = 20) 
  (h_on_sphere : ∀ P ∈ {A, B, C}, dist O P = r)
  (AB BC CA : ℝ) (h_AB : AB = 13) (h_BC : BC = 14) (h_CA : CA = 15) 
  (volume_eq : 15 * ℝ.sqrt 95 / 8 = 20) : 
  let m := 15
      n := 95
      k := 8 in
  m + n + k = 118 := sorry

end sphere_distance_l371_371700


namespace Trees_distance_l371_371746

theorem Trees_distance
  (yard_length : ℝ)
  (number_of_trees : ℕ)
  (trees_at_each_end : Prop :=
    yard_length = 150 ∧ number_of_trees = 11 ∧ trees_at_each_end := true) :
    ∃ distance_between_trees : ℝ, 
      (distance_between_trees = yard_length / (number_of_trees - 1)) ∧ distance_between_trees = 15 :=
begin
  sorry
end

end Trees_distance_l371_371746


namespace race_order_count_l371_371121

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end race_order_count_l371_371121


namespace no_such_quadratic_exists_l371_371479

theorem no_such_quadratic_exists : ¬ ∃ (b c : ℝ), 
  (∀ x : ℝ, 6 * x ≤ 3 * x^2 + 3 ∧ 3 * x^2 + 3 ≤ x^2 + b * x + c) ∧
  (x^2 + b * x + c = 1) :=
by
  sorry

end no_such_quadratic_exists_l371_371479


namespace sum_of_five_consecutive_even_integers_l371_371732

theorem sum_of_five_consecutive_even_integers (a : ℤ) 
  (h : a + (a + 4) = 144) : a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370 := by
  sorry

end sum_of_five_consecutive_even_integers_l371_371732


namespace f_at_five_l371_371381

-- Define the function f with the property given in the condition
axiom f : ℝ → ℝ
axiom f_prop : ∀ x : ℝ, f (3 * x - 1) = x^2 + x + 1

-- Prove that f(5) = 7 given the properties above
theorem f_at_five : f 5 = 7 :=
by
  sorry

end f_at_five_l371_371381


namespace conditional_probability_eventB_given_eventA_l371_371379

-- Definitions for the events and the probability spaces
def eventA (x y : ℕ) : Prop := (x * y) % 2 = 0
def eventB (x y : ℕ) : Prop := x % 2 = 0 ∧ y % 2 = 0

-- Given conditions: 
-- 1. x and y represent the numbers facing up on the first and second roll of a fair six-sided die (1 through 6).
-- 2. Define events A and B as given:
--    A: (x * y) is even
--    B: Both (x and y) are even

-- Lean statement for proving the conditional probability
theorem conditional_probability_eventB_given_eventA :
  -- Statement proving that P(B | A) = 1 / 3
  sorry

end conditional_probability_eventB_given_eventA_l371_371379


namespace find_perpendicular_slope_value_l371_371871

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end find_perpendicular_slope_value_l371_371871


namespace simplify_expression_l371_371649

theorem simplify_expression :
  (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4 * 6 * a^5) = 720 * a^15 :=
by
  sorry

end simplify_expression_l371_371649


namespace temperature_decrease_l371_371565

-- Conditions
def current_temperature : ℝ := 84
def temperature_factor : ℝ := 3 / 4

-- Mathematical problem: Prove that the temperature decrease is 21 degrees
theorem temperature_decrease : 
  current_temperature - (temperature_factor * current_temperature) = 21 :=
by
  sorry

end temperature_decrease_l371_371565


namespace median_unchanged_l371_371176

def donationAmountsOrig : List ℕ := [30, 50, 50, 60, 60]
def donationAmountsNew : List ℕ := [50, 50, 50, 60, 60]

theorem median_unchanged (dOrig dNew : List ℕ) :
  dOrig = [30, 50, 50, 60, 60] →
  dNew  = [50, 50, 50, 60, 60] →
  List.median dOrig = List.median dNew :=
by
  sorry

end median_unchanged_l371_371176


namespace petya_square_larger_l371_371629

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end petya_square_larger_l371_371629


namespace decreasing_seq_condition_l371_371083

variable {a : ℕ → ℝ} (a1 : ℝ) (d : ℝ)
  (h_a1_pos : a1 > 0)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)

theorem decreasing_seq_condition :
  (∀ n ≥ 2, 4^(a1 * a (n + 1)) < 4^(a1 * a n)) ↔ d < 0 :=
by
  sorry

end decreasing_seq_condition_l371_371083


namespace pow_div_l371_371320

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l371_371320


namespace lcm_of_4_8_9_10_l371_371336

theorem lcm_of_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 := by
  sorry

end lcm_of_4_8_9_10_l371_371336


namespace similarity_ratio_range_l371_371830

noncomputable def similarity_ratio_interval (x y z p : ℝ) (h_sim : (x / y) = (y / z) ∧ (y / z) = (z / p)) : Prop :=
  let k := x / y in
  (k + k * k > 1) ∧ (k * k + 1 > k) ∧ (1 + k > k * k) ∧
  k ∈ Ioo ((sqrt 5 - 1) / 2) ((sqrt 5 + 1) / 2)

theorem similarity_ratio_range (x y z p : ℝ) (h_sim : (x / y) = (y / z) ∧ (y / z) = (z / p)) :
  (0 : ℤ) ≤ floor ((sqrt 5 - 1) / 2) ∧ ceil ((sqrt 5 + 1) / 2) ≤ (2 : ℤ) :=
sorry

end similarity_ratio_range_l371_371830


namespace range_of_exponential_shifted_l371_371702

theorem range_of_exponential_shifted:
  (set.range (λ x: ℝ, 2^x - 1)) = set.Ioi (-1) :=
begin
  sorry
end

end range_of_exponential_shifted_l371_371702


namespace trigonometric_expression_value_l371_371049

theorem trigonometric_expression_value (a : ℝ) (h1 : cos a = 2 / 3) (h2 : -π / 2 < a ∧ a < 0) :
  (tan (-a - π) * sin (2 * π + a) / (cos (-a) * tan (π + a))) = sqrt 5 / 2 :=
by sorry

end trigonometric_expression_value_l371_371049


namespace cheapest_book_price_l371_371041

-- Define the main sequence and properties
def price (n : ℕ) (c : ℝ) : ℝ := c + 3 * (n - 1)

-- Define all the conditions given in the problem
variable (c : ℝ)
variable (h1 : price 21 c = 70)
variable (h2 : price 41 c = price 20 c + price 21 c)

theorem cheapest_book_price
  (h1 : price 21 c = 70)
  (h2 : price 41 c = price 20 c + price 21 c) :
  price 1 c = 10 :=
begin
  -- The actual proof would go here
  sorry
end

end cheapest_book_price_l371_371041


namespace frac_sum_is_one_l371_371843

theorem frac_sum_is_one (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end frac_sum_is_one_l371_371843


namespace petya_square_larger_than_vasya_square_l371_371633

variable (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)

def petya_square_side (a b : ℝ) : ℝ := a * b / (a + b)

def vasya_square_side (a b : ℝ) : ℝ := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petya_square_larger_than_vasya_square
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  petya_square_side a b > vasya_square_side a b :=
by sorry

end petya_square_larger_than_vasya_square_l371_371633


namespace f_satisfies_conditions_t_min_value_omega_functions_l371_371854

-- Define f(x) based on given conditions
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

-- Question 1: Prove that f(x) satisfies the given conditions
theorem f_satisfies_conditions :
  (f 0 = 4) ∧
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ (f x₁ = 2 * x₁) ∧ (f x₂ = 2 * x₂) ∧ (x₁ = 1) ∧ (x₂ = 4)) :=
by
  have H0 : f 0 = 4 := rfl
  exists 1
  exists 4
  constructor
  { linarith }
  { constructor
    { norm_num }
    { constructor
      { norm_num }
      { constructor
        { norm_num }
        { norm_num } } } }

-- Question 2: Prove that t = √2 / 2 given the minimum condition for h(x)
def h (x t : ℝ) : ℝ := f x - (2 * t - 3) * x

theorem t_min_value (t : ℝ) :
  (∀ x : ℝ, x ∈ (set.Icc 0 1) → h x t ≥ 7 / 2) ↔ t = Real.sqrt 2 / 2 :=
sorry

-- Question 3: Prove the range of m such that f(x) and g(x) are Ω functions on [0, 3]
def g (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

theorem omega_functions (m : ℝ) :
  (f 0 - g 0 m ≥ 0) ∧
  (f 3 - g 3 m ≥ 0) ∧
  ((-5)^2 - 4 * (4 - m) > 0) ↔
  - (9 : ℝ) / 4 < m ∧ m ≤ -2 :=
sorry

end f_satisfies_conditions_t_min_value_omega_functions_l371_371854


namespace simplify_expression_l371_371653

theorem simplify_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 9 / Real.log 18 + 1)) = 7 / 4 := 
sorry

end simplify_expression_l371_371653


namespace parallelogram_FD_length_l371_371207

structure Parallelogram (A B C D E F : Type) :=
  (angle_ABC : ℝ)
  (AB : ℝ)
  (BC : ℝ)
  (DE : ℝ)
  (intersection_BE_AD : Prop)

def length_FD (A B C D E F : Type) [Parallelogram A B C D E F] : ℝ :=
  let ⟨angle_ABC, AB, BC, DE, _⟩ := Parallelogram.mk angle_ABC AB BC DE sorry in
  -- Using similar triangles properties in the solution to assign FD value
  BC / 4

theorem parallelogram_FD_length 
  {A B C D E F : Type} 
  [Parallelogram A B C D E F] 
  (h1 : Parallelogram A B C D E F) 
  (angle_ABC_eq : h1.angle_ABC = 150)
  (AB_eq : h1.AB = 20)
  (BC_eq : h1.BC = 15)
  (DE_eq : h1.DE = 5)
  (intersection: h1.intersection_BE_AD) :
  length_FD A B C D E F = 3.75 :=
by
  sorry

end parallelogram_FD_length_l371_371207


namespace range_of_a_l371_371514

def f (x : ℝ) : ℝ := (1 / 2) * (3 * log (x + 2) - log (x - 2))
def F (a x : ℝ) : ℝ := a * log (x - 1) - f x
def is_monotonic_increasing (F : ℝ → ℝ) : Prop := ∀ ⦃x y : ℝ⦄, x < y → F x ≤ F y

theorem range_of_a (a : ℝ) :
  (is_monotonic_increasing (F a)) ↔ (1 ≤ a) := sorry

end range_of_a_l371_371514


namespace petyas_square_is_larger_l371_371628

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l371_371628


namespace diameter_of_circle_with_area_4pi_l371_371754

theorem diameter_of_circle_with_area_4pi : ∀ (r : ℝ), π * r^2 = 4 * π → 2 * r = 4 :=
by
  assume r
  assume h : π * r^2 = 4 * π
  sorry

end diameter_of_circle_with_area_4pi_l371_371754


namespace nasrin_mean_speed_l371_371953

variable (time_to_camp_hours : ℝ)
variable (time_to_camp_minutes : ℝ)
variable (distance_to_camp_km : ℝ)

-- Given conditions
def time_to_camp_total_minutes := (time_to_camp_hours * 60) + time_to_camp_minutes
def return_trip_factor := 1/3
def time_for_return_trip := time_to_camp_total_minutes * return_trip_factor
def total_distance_km := 2 * distance_to_camp_km
def total_time_minutes := time_to_camp_total_minutes + time_for_return_trip
def total_time_hours := total_time_minutes / 60

-- Mean speed calculation
def mean_speed := total_distance_km / total_time_hours

-- Given specific values
def time_to_camp_hours_value : ℝ := 2
def time_to_camp_minutes_value : ℝ := 30
def distance_to_camp_km_value : ℝ := 4.5

-- Proof statement
theorem nasrin_mean_speed :
  mean_speed time_to_camp_hours_value time_to_camp_minutes_value distance_to_camp_km_value = 2.7 := 
sorry

end nasrin_mean_speed_l371_371953


namespace collinear_EFP_l371_371597

variables {A B C D E F O P : Type}
variables [affine_plane A]

-- Conditions
def quadrilateral (A B C D : A) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  ∃ O, O ∉ line_through A C ∧ O ∉ line_through B D ∧
  colinear A O C ∧ colinear B O D

def is_intersection (O : A) (A C B D : A) : Prop :=
  ∃ O, colinear A O C ∧ colinear B O D

def parallelogram (A O D E : A) : Prop :=
  ∃ E, mid_point A D E ∧ mid_point O E D

def is_structurally_intersecting (P : A) (l1 l2 : set A) : Prop :=
  ∃ P, colinear A B P ∧ colinear C D P

-- Prove
theorem collinear_EFP (A B C D O P E F : A) 
  (h1 : quadrilateral A B C D)
  (h2 : is_intersection O A C B D)
  (h3 : is_structurally_intersecting P (line_through A B) (line_through C D))
  (h4 : parallelogram A O D E)
  (h5 : parallelogram B O C F)
  : collinear E F P :=
sorry

end collinear_EFP_l371_371597


namespace largest_two_digit_prime_factor_of_binomial_l371_371723

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), 10 ≤ p ∧ p < 100 ∧ p.prime ∧ p ∣ nat.binom 150 75 ∧ ∀ q, 10 ≤ q ∧ q < 100 ∧ q.prime ∧ q ∣ nat.binom 150 75 → q ≤ p :=
begin
  use 47,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { sorry },
  { sorry }
end

end largest_two_digit_prime_factor_of_binomial_l371_371723


namespace triangle_45_45_90_l371_371815

theorem triangle_45_45_90 (XY XZ : ℝ) (h1 : XZ = 12 * Real.sqrt 2) 
    (h2 : ∠YXZ = 90) (h3 : ∠Z = 45) : 
    XY = 12 * Real.sqrt 2 := 
  sorry

end triangle_45_45_90_l371_371815


namespace exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l371_371069

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem exists_a_f_has_two_zeros (a : ℝ) :
  (0 < a ∧ a < 2) ∨ (-2 < a ∧ a < 0) → ∃ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧ f x₁ a = 0) ∧
  (0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧ f x₂ a = 0) ∧ x₁ ≠ x₂ := sorry

theorem range_of_a_for_f_eq_g :
  ∀ a : ℝ, a ∈ Set.Icc (-2 : ℝ) (3 : ℝ) →
  ∃ x₁ : ℝ, x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ f x₁ a = g 2 ∧
  ∃ x₂ : ℝ, x₂ ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧ f x₁ a = g x₂ := sorry

end exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l371_371069


namespace angles_sum_l371_371895

noncomputable def Point := (ℝ × ℝ)

-- Define all the points
def A := (0, 1) : Point
def B := (0, 0) : Point
def C := (1, 0) : Point
def D := (2, 0) : Point
def E := (3, 0) : Point
def F := (3, 1) : Point

-- Function to calculate the angle between two vectors
noncomputable def angle (u v : Point) : ℝ :=
  Real.arccos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2)))

-- Vectors between the points
def vector (P Q : Point) : Point := (Q.1 - P.1, Q.2 - P.2)

-- The angles between the lines originating from the points
noncomputable def ∠FBE := angle (vector F B) (vector B E)
noncomputable def ∠FCE := angle (vector F C) (vector C E)
noncomputable def ∠FDE := angle (vector F D) (vector D E)

-- The Lean statement
theorem angles_sum : ∠FBE + ∠FCE = ∠FDE := sorry

end angles_sum_l371_371895


namespace max_a_no_lattice_points_l371_371466

theorem max_a_no_lattice_points :
  ∀ (m : ℝ), (1 / 3) < m → m < (17 / 51) →
  ¬ ∃ (x : ℕ) (y : ℕ), 0 < x ∧ x ≤ 50 ∧ y = m * x + 3 := 
by
  sorry

end max_a_no_lattice_points_l371_371466


namespace parallel_vectors_l371_371115

variable {n : ℝ}
variable {e1 e2 : ℝ}
variable (a b : ℝ → ℝ)

def vec_a := (λ (t : ℝ), 3 * if t = 0 then 1 else 0 - 4 * if t = 1 then 1 else 0)
def vec_b := (λ (t : ℝ), (1 - n) * if t = 0 then 1 else 0 + 3 * n * if t = 1 then 1 else 0)

theorem parallel_vectors 
  (h : (∃ k : ℝ, ∀ t, vec_a t = k * vec_b t) ∨ (e1 = 0 ∨ e2 = 0)) : 
  n = -4/5 ∨ n ∈ set.univ :=
by {
  sorry
}

end parallel_vectors_l371_371115


namespace license_plates_count_l371_371150

theorem license_plates_count :
  let letters := 26
  ∧ let letters_or_digits := 36
  ∧ let same_as_second := 1
  ∧ let digits := 10
  in letters * letters_or_digits * same_as_second * digits = 9360 :=
by
  sorry

end license_plates_count_l371_371150


namespace assign_guests_to_rooms_l371_371370

-- Define the problem conditions
def seven_rooms := 7
def deluxe_room_min_guests := 2
def deluxe_room_max_guests := 3
def total_business_partners := 7

-- State the theorem to prove
theorem assign_guests_to_rooms : 
  ∃ (valid_assignments : ℕ), valid_assignments = 27720 := 
by {
  -- Define the satisfaction of all conditions here. This is a high-level outline.
  let rooms := seven_rooms,
  let min_guests := deluxe_room_min_guests,
  let max_guests := deluxe_room_max_guests,
  let partners := total_business_partners,

  -- Placeholder for the proof
  sorry
}

end assign_guests_to_rooms_l371_371370


namespace problem_statement_l371_371733

-- Defining the terms x, y, and d as per the problem conditions
def x : ℕ := 2351
def y : ℕ := 2250
def d : ℕ := 121

-- Stating the proof problem in Lean
theorem problem_statement : (x - y)^2 / d = 84 := by
  sorry

end problem_statement_l371_371733


namespace Dima_floor_l371_371442

theorem Dima_floor (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 9)
  (h2 : 60 = (n - 1))
  (h3 : 70 = (n - 1) / (n - 1) * 60 + (n - n / 2) * 2 * 60)
  (h4 : ∀ m : ℕ, 1 ≤ m ∧ m ≤ 9 → (5 * n = 6 * m + 1) → (n = 7 ∧ m = 6)) :
  n = 7 :=
by
  sorry

end Dima_floor_l371_371442


namespace power_division_l371_371329

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l371_371329


namespace proof_problem_l371_371209

variable (I : Set)
variable (S1 S2 S3 : Set)
variable [Nonempty I]
variable [Subset I S1]
variable [Subset I S2]
variable [Subset I S3]

theorem proof_problem 
  (h1 : S1 ∪ S2 ∪ S3 ≠ I) : C_I S1 ∩ C_I S2 ∩ C_I S3 = ∅ :=
by
  sorry

end proof_problem_l371_371209


namespace power_function_inequality_l371_371896

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^a

theorem power_function_inequality (x : ℝ) (a : ℝ) : (x > 1) → (f x a < x) ↔ (a < 1) :=
by
  sorry

end power_function_inequality_l371_371896


namespace polynomial_product_l371_371462

noncomputable def sum_of_coefficients (g h : ℤ) : ℤ := g + h

theorem polynomial_product (g h : ℤ) :
  (9 * d^3 - 5 * d^2 + g) * (4 * d^2 + h * d - 9) = 36 * d^5 - 11 * d^4 - 49 * d^3 + 45 * d^2 - 9 * d →
  sum_of_coefficients g h = 18 :=
by
  intro
  sorry

end polynomial_product_l371_371462


namespace petya_square_larger_l371_371631

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end petya_square_larger_l371_371631


namespace count_n_satisfying_condition_l371_371036

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_n_satisfying_condition : 
  ({n : ℕ | 0 < n ∧ is_prime (n^3 - 9 * n^2 + 25 * n - 15)}).card = 3 :=
by
  sorry

end count_n_satisfying_condition_l371_371036


namespace solution_6x_equation_l371_371028

def cos_six_x (x : ℝ) : ℝ := Real.cos (6 * x)
def cos_square_four_x (x : ℝ) : ℝ := (Real.cos (4 * x))^2
def cos_cube_three_x (x : ℝ) : ℝ := (Real.cos (3 * x))^3
def cos_four_two_x (x : ℝ) : ℝ := (Real.cos (2 * x))^4

theorem solution_6x_equation (x : ℝ) (hx : -Real.pi ≤ x ∧ x ≤ Real.pi) :
  cos_six_x x + cos_square_four_x x + cos_cube_three_x x + cos_four_two_x x = 0 →
  x = Real.pi / 2 ∨ x = -Real.pi / 2 :=
sorry

end solution_6x_equation_l371_371028


namespace find_a_find_b_l371_371884

section Problem1

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^4 - 4 * x^3 + a * x^2 - 1

-- Condition 1: f is monotonically increasing on [0, 1]
def f_increasing_on_interval_01 (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x ≤ y → f x a ≤ f y a

-- Condition 2: f is monotonically decreasing on [1, 2]
def f_decreasing_on_interval_12 (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 2 ∧ x ≤ y → f y a ≤ f x a

-- Proof of a part
theorem find_a : ∃ a, f_increasing_on_interval_01 a ∧ f_decreasing_on_interval_12 a ∧ a = 4 :=
  sorry

end Problem1

section Problem2

noncomputable def f_fixed (x : ℝ) : ℝ := x^4 - 4 * x^3 + 4 * x^2 - 1
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x^2 - 1

-- Condition for intersections
def intersect_at_two_points (b : ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f_fixed x1 = g x1 b ∧ f_fixed x2 = g x2 b

-- Proof of b part
theorem find_b : ∃ b, intersect_at_two_points b ∧ (b = 0 ∨ b = 4) :=
  sorry

end Problem2

end find_a_find_b_l371_371884


namespace complement_union_l371_371111

open Set

variable {R : Type} [LinearOrder R] [TopologicalSpace R] [OrderTopology R]

def A (x : R) : Prop := x < 1
def B (x : R) : Prop := x > 3

theorem complement_union (R_eq_real : R = ℝ) :
  (compl (A ∪ B) : Set R) = {x : R | 1 ≤ x ∧ x ≤ 3} :=
by 
  intro x
  simp [compl, union, A, B, not_or, not_lt, not_gt]
  sorry  

end complement_union_l371_371111


namespace circle_condition_l371_371271

theorem circle_condition (f : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 4*x + 6*y + f = 0) ↔ f < 13 :=
by
  sorry

end circle_condition_l371_371271


namespace cauchy_schwarz_am_hm_inequality_l371_371639

theorem cauchy_schwarz_am_hm_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 / (1 + real.sqrt x)^2) + (1 / (1 + real.sqrt y)^2) ≥ (2 / (x + y + 2)) :=
by
  sorry

end cauchy_schwarz_am_hm_inequality_l371_371639


namespace edward_pipe_use_l371_371450

theorem edward_pipe_use
  (initial_washers remaining_washers washers_per_bolt : ℕ)
  (pipe_per_bolt used_washers bolts : ℕ)
  (initial_washers = 20)
  (remaining_washers = 4)
  (washers_per_bolt = 2)
  (pipe_per_bolt = 5)
  (used_washers = initial_washers - remaining_washers)
  (bolts = used_washers / washers_per_bolt) :
  (bolts * pipe_per_bolt = 40) :=
sorry

end edward_pipe_use_l371_371450


namespace part_I_part_II_l371_371898

open Set

variable (a b : ℝ)

theorem part_I (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hB_sub_A : B ⊆ A) : a = -1 :=
  sorry

theorem part_II (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hA_inter_B_nonempty : A ∩ B ≠ ∅) : a^2 - b^2 + 2 * a = -1 :=
  sorry

end part_I_part_II_l371_371898


namespace intersection_line_image_l371_371058

-- Given definitions and conditions
variables (A B C D M : Type) 
variables (proj : A → A_1) (proj : B → B_1) (proj : C → C_1) (proj : D → D_1) (proj : M → M_1)
variables (AB CD : A → A → Prop)
variables (intersect : A → A → A)
variables (parallel : A → A → Prop)

-- Definitions based on problem
def planes_intersect_parallel (h1: parallel AB CD) : (M_1, parallel A_1B_1 C_1D_1) :=
sorry

def planes_intersect_at_E (h2: ∃ E, AB ∩ CD = E) : (M_1, line_through M_1 (proj E)) :=
sorry

-- Main theorem
theorem intersection_line_image (A B C D M : Type)
(proj : A → A_1) (proj : B → B_1) (proj : C → C_1) (proj : D → D_1) (proj : M → M_1)
(AB CD : A → A → Prop) (parallel : A → A → Prop)
(intersect : A → A → A) :
( ∀ h : parallel AB CD, planes_intersect_parallel h ∧ 
  ∀ h : ∃ E, AB ∩ CD = E, planes_intersect_at_E h ) :=
sorry

end intersection_line_image_l371_371058


namespace derivative_at_one_l371_371977

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : deriv f 1 = 2 * Real.exp 1 := by
sorry

end derivative_at_one_l371_371977


namespace factor_tree_value_l371_371559

theorem factor_tree_value :
  ∃ (A : ℕ),
    (let D := 5 * 2 in
     let E := 7 * 3 in
     let B := 3 * D in
     let C := 7 * E in
     A = B * C) ∧
    A = 4410 :=
begin
  use 4410,
  split,
  { let D := 5 * 2,
    let E := 7 * 3,
    let B := 3 * D,
    let C := 7 * E,
    exact rfl },
  { exact rfl }
end

end factor_tree_value_l371_371559


namespace increasing_interval_of_a_l371_371542

variable {f : ℝ → ℝ}
variable (a : ℝ)

def f := λ x : ℝ, - (1/3) * x^3 + (1/2) * x^2 + 2 * a * x

theorem increasing_interval_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 2 / 3 ∧ deriv f x > 0) ↔ a > -1 / 9 := 
begin
  sorry
end

end increasing_interval_of_a_l371_371542


namespace polynomial_evaluation_l371_371016

theorem polynomial_evaluation 
  (x : ℝ) 
  (h1 : x^2 - 3 * x - 10 = 0) 
  (h2 : x > 0) : 
  (x^4 - 3 * x^3 + 2 * x^2 + 5 * x - 7) = 318 :=
by
  sorry

end polynomial_evaluation_l371_371016


namespace intersection_A_B_l371_371491

variable {α : Type*} [LinearOrderedField α]

def A : Set α := { x | x > 2 }
def B : Set α := { x | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := by
  sorry

end intersection_A_B_l371_371491


namespace value_of_a_m_minus_2n_l371_371154

variable (a : ℝ) (m n : ℝ)

theorem value_of_a_m_minus_2n (h1 : a^m = 8) (h2 : a^n = 4) : a^(m - 2 * n) = 1 / 2 :=
by
  sorry

end value_of_a_m_minus_2n_l371_371154


namespace monotone_decreasing_f_max_k_ineq_exponential_inequality_l371_371098

-- Problem 1: Monotonicity of f(x)
theorem monotone_decreasing_f {x : ℝ} (h : 0 < x) : 
  let f := λ x : ℝ, (1 + log (x + 1)) / x 
  in monotone_decreasing_on f (set.Ioi 0) :=
sorry

-- Problem 2: Maximum k
theorem max_k_ineq {k : ℕ} (k_le3 : k ≤ 3) {x : ℝ} (h : 0 < x) : 
  let f := λ x : ℝ, (1 + log (x + 1)) / x 
  in f x > k / (x + 1) → k = 3 :=
sorry

-- Problem 3: Exponential Inequality
theorem exponential_inequality {n : ℕ} (h : 0 < n) :
  (list.prod (list.map (λ i, 1 + (i * (i + 1))) (list.range n)))
  > real.exp ((2 * n) - 3) :=
sorry

end monotone_decreasing_f_max_k_ineq_exponential_inequality_l371_371098


namespace smallest_N_for_pairs_l371_371681

theorem smallest_N_for_pairs :
  ∃ N : ℕ, (∀ (a b : ℕ), a ≠ b ∧ a ∈ {1, 2, ..., 2016} ∧ b ∈ {1, 2, ..., 2016} →
  ∃ (pairs : list (ℕ × ℕ)), (∀ (p : ℕ × ℕ) ∈ pairs, (p.1 * p.2 ≤ N)) ∧ N = 1017072) := by
  sorry

end smallest_N_for_pairs_l371_371681


namespace maximize_OM_ON_l371_371742

theorem maximize_OM_ON (a b : ℝ) : 
  sorry -- Here we would prove that given the structure of the problem, with fixed AC and BC, the angle ACB = 135 degrees to maximize OM + ON.

end maximize_OM_ON_l371_371742


namespace sine_transform_correct_l371_371509

def transform_sine_function (x : ℝ) : ℝ :=
  let f1 := sin (x + (Real.pi / 3))
  let f2 := sin ((1 / 2) * x + (Real.pi / 3))
  (1 / 2) * f2

theorem sine_transform_correct :
  ∀ x : ℝ, transform_sine_function x = (1 / 2) * sin ((1 / 2) * x + (Real.pi / 3)) :=
by
  intro x
  unfold transform_sine_function
  sorry

end sine_transform_correct_l371_371509


namespace problem_l371_371156

noncomputable def f : ℝ → ℝ := sorry

theorem problem (f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x)
                (h : ∀ x : ℝ, 0 < x → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3 / 2 :=
sorry

end problem_l371_371156


namespace counterfeit_identifiable_in_two_weighings_l371_371947

-- Define the condition that one of four coins is counterfeit
def is_counterfeit (coins : Fin 4 → ℚ) (idx : Fin 4) : Prop :=
  ∃ real_weight counterfeit_weight : ℚ, real_weight ≠ counterfeit_weight ∧
  (∀ i : Fin 4, i ≠ idx → coins i = real_weight) ∧ coins idx = counterfeit_weight

-- Define the main theorem statement
theorem counterfeit_identifiable_in_two_weighings (coins : Fin 4 → ℚ) :
  (∃ idx : Fin 4, is_counterfeit coins idx) → ∃ idx : Fin 4, is_counterfeit coins idx ∧
  ∀ (balance : (Fin 4 → Prop) → ℤ → Prop), (∃ w1 w2 : Fin 4 → Prop, balance w1 = 0 ∨ balance w2 = 0 → idx) :=
sorry

end counterfeit_identifiable_in_two_weighings_l371_371947


namespace parabola_line_results_l371_371869

def parabola_eqn : Prop := ∀ x y : ℝ, y^2 = -4 * x

def focus_F : ℝ × ℝ := (-1, 0)

def line_l : ℝ × ℝ → ℝ := λ p, p.2 - p.1 - 1 = 0

def intersects (l : ℝ × ℝ → ℝ) (p_eqn : Prop) (A B : ℝ × ℝ) : Prop :=
  l(A) = 0 ∧ l(B) = 0 ∧ p_eqn A.1 A.2 ∧ p_eqn B.1 B.2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem parabola_line_results (A B : ℝ × ℝ) (AF BF : ℝ) :
  parabola_eqn A.1 A.2 → parabola_eqn B.1 B.2 →
  line_l A = 0 → line_l B = 0 →
  intersects line_l parabola_eqn A B →
  |distance A B| = 8 ∧ 
  (1 / |AF| + 1 / |BF|) = 1 := 
sorry

end parabola_line_results_l371_371869


namespace continuous_functions_integral_limit_l371_371962

variable {f g : ℝ → ℝ}

theorem continuous_functions_integral_limit
  (hf : ∀ x, 0 ≤ x → x ≤ 1 → ContinuousAt f x)
  (hg : ∀ x, 0 ≤ x → x ≤ 1 → ContinuousAt g x)
  (hfg : ∀ x, (0:ℝ) ≤ x → x ≤ 1 → f x * g x ≥ 4 * x^2) :
  |(∫ x in 0..1, f x)| ≥ 1 ∨ |(∫ x in 0..1, g x)| ≥ 1 := 
sorry

end continuous_functions_integral_limit_l371_371962


namespace periodic_function_has_least_period_l371_371793

noncomputable def f : ℝ → ℝ := sorry -- Define the function f

theorem periodic_function_has_least_period :
  (∀ x : ℝ, f(x + 5) + f(x - 5) = f(x)) → ∃ p : ℕ, p > 0 ∧ (∀ x : ℝ, f(x) = f(x + p)) ∧ p = 30 :=
by
  intros h
  use 30
  split
  exact Nat.zero_lt_bit0 (by decide)
  split
  intros x
  -- Proof skipped
  sorry
  rfl

end periodic_function_has_least_period_l371_371793


namespace product_of_roots_l371_371829

theorem product_of_roots (a b c : ℤ) (h_eq : a = 24 ∧ b = 60 ∧ c = -600) :
  ∀ x : ℂ, (a * x^2 + b * x + c = 0) → (x * (-b - x) = -25) := sorry

end product_of_roots_l371_371829


namespace area_of_ABCD_l371_371171

-- Define the trapezoid and the areas of the specific triangles
variables (A B C D E : Type) [is_trapezoid A B C D E]
  (area_∆ABE : ℝ) (area_∆ADE : ℝ)

noncomputable def Area_of_trapezoid (A B C D E: Type) [is_trapezoid A B C D E]
  (area_∆ABE : ℝ) (area_∆ADE : ℝ) : ℝ :=
  let area_∆BCE := area_∆ADE in -- Given symmetry
  let area_∆CDE := (4/25) * area_∆ABE in -- From solution step
  area_∆ABE + area_∆ADE + area_∆BCE + area_∆CDE

theorem area_of_ABCD (A B C D E : Type) [is_trapezoid A B C D E]
  (area_∆ABE : ℝ) (area_∆ADE : ℝ) (h1 : area_∆ABE = 50) (h2 : area_∆ADE = 20) :
  Area_of_trapezoid A B C D E area_∆ABE area_∆ADE = 98 :=
by sorry

end area_of_ABCD_l371_371171


namespace incorrect_expression_l371_371971

variable {r s : ℕ}
variable {P Q : ℕ}
variable {D : ℚ}

-- Definitions and conditions according to the problem
def decimal_D_defined (D : ℚ) (P : ℕ) (Q : ℕ) (r : ℕ) (s : ℕ) : Prop :=
  D = P / (10^r : ℚ) + Q / (10^(r + 1) : ℚ) * (Sum (fun n : ℕ => 10^(-s * n) : ℕ → ℚ) n)

-- Statement to prove
theorem incorrect_expression (h : decimal_D_defined D P Q r s) : 
  10^r * (10^s + 1) * D ≠ Q * (P + 1) := 
sorry

end incorrect_expression_l371_371971


namespace lines_through_three_distinct_points_l371_371146

theorem lines_through_three_distinct_points : 
  ∃ n : ℕ, n = 54 ∧ (∀ (i j k : ℕ), 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ 1 ≤ k ∧ k ≤ 3 → 
  ∃ (a b c : ℤ), -- Direction vector (a, b, c)
  abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
  ((i + a > 0 ∧ i + a ≤ 3) ∧ (j + b > 0 ∧ j + b ≤ 3) ∧ (k + c > 0 ∧ k + c ≤ 3) ∧
  (i + 2 * a > 0 ∧ i + 2 * a ≤ 3) ∧ (j + 2 * b > 0 ∧ j + 2 * b ≤ 3) ∧ (k + 2 * c > 0 ∧ k + 2 * c ≤ 3))) := 
sorry

end lines_through_three_distinct_points_l371_371146


namespace evaluate_expression_l371_371072

variable {a b c : ℝ}

theorem evaluate_expression
  (h : a / (35 - a) + b / (75 - b) + c / (85 - c) = 5) :
  7 / (35 - a) + 15 / (75 - b) + 17 / (85 - c) = 8 / 5 := by
  sorry

end evaluate_expression_l371_371072


namespace largest_rectangle_area_l371_371688

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l371_371688


namespace minimum_value_expression_is_neg27_l371_371027

noncomputable def minimum_value_expression : ℤ :=
  let expr (x y : ℝ) :=
    (√(2 * (1 + Real.cos (2 * x))) - √(36 - 4 * √5) * Real.sin x + 2) *
    (3 + 2 * √(10 - √5) * Real.cos y - Real.cos (2 * y))
  in
  Int.round (Inf (Set.range (λ (xy : ℝ × ℝ), expr xy.1 xy.2)))

theorem minimum_value_expression_is_neg27 :
  minimum_value_expression = -27 := 
by 
  sorry

end minimum_value_expression_is_neg27_l371_371027


namespace find_value_of_k_l371_371383

theorem find_value_of_k (k : ℝ) 
  (hcollinear : collinear {p : ℝ × ℝ | (p = (-1, 6)) ∨ (p = (6, k)) ∨ (p = (20, 3))}) : 
  k = 5 := sorry

end find_value_of_k_l371_371383


namespace lcm_4_8_9_10_l371_371339

theorem lcm_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 :=
by
  -- Definitions of the numbers (additional definitions from problem conditions)
  let four := 4 
  let eight := 8
  let nine := 9
  let ten := 10
  
  -- Prime factorizations:
  have h4 : Nat.prime_factors four = [2, 2],
    from rfl
  
  have h8 : Nat.prime_factors eight = [2, 2, 2],
    from rfl

  have h9 : Nat.prime_factors nine = [3, 3],
    from rfl

  have h10 : Nat.prime_factors ten = [2, 5],
    from rfl

  -- Least common multiple calculation
  let highest_2 := 2 ^ 3
  let highest_3 := 3 ^ 2
  let highest_5 := 5

  -- Multiply together
  let lcm := highest_2 * highest_3 * highest_5

  show Nat.lcm (Nat.lcm four eight) (Nat.lcm nine ten) = lcm
  sorry

end lcm_4_8_9_10_l371_371339


namespace janice_purchases_l371_371585

theorem janice_purchases (a b c : ℕ) : 
  a + b + c = 50 ∧ 30 * a + 200 * b + 300 * c = 5000 → a = 10 :=
sorry

end janice_purchases_l371_371585


namespace ellipse_problem_l371_371501

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_problem
    (C : ℝ × ℝ)
    (h1 : (0, 0) = (0, 0))  -- The center of the ellipse is the origin.
    (h2 : ∃ B : ℝ × ℝ, B = (2, 0))  -- The major axis AB is on the x-axis and has length 4.
    (h3 : C ∈ {p : ℝ × ℝ | ellipse_equation 2 (real.sqrt (4/3)) p.1 p.2})  -- Point C is on the ellipse.
    (h4 : ∠ (2, 0) (1, 1) = π/4)  -- ∠ CBA = π/4.
    (h5 : dist (2,0) (0,0) = 2 * 2)  -- B is at (2,0)
    (h6 : dist (1,1) (2,0) = real.sqrt 2) :   -- BC = √2
  ellipse_equation 2 2
(ellipse_equation 2 (real.sqrt (4/3))) :=
begin
  sorry
end

end ellipse_problem_l371_371501


namespace math_problem_l371_371429

theorem math_problem : (2 + (2 / 3 : ℚ) + 6.3 - ((5 / 3 : ℚ) - (1 + (3 / 5 : ℚ)))) = 8.9 := 
by
  norm_num
-- If there's any simplification required, such as converting 6.3 to (63 / 10 : ℚ), it can be added.

end math_problem_l371_371429


namespace domain_g_l371_371541

noncomputable def f : ℝ → ℝ := sorry  -- f is a real-valued function

theorem domain_g:
  (∀ x, x ∈ [-2, 4] ↔ f x ∈ [-2, 4]) →  -- The domain of f(x) is [-2, 4]
  (∀ x, x ∈ [-2, 2] ↔ (f x + f (-x)) ∈ [-2, 2]) :=  -- The domain of g(x) = f(x) + f(-x) is [-2, 2]
by
  intros h
  sorry

end domain_g_l371_371541


namespace evaluate_complex_expression_l371_371812

noncomputable def expression := 
  Complex.mk (-1) (Real.sqrt 3) / 2

noncomputable def conjugate_expression := 
  Complex.mk (-1) (-Real.sqrt 3) / 2

theorem evaluate_complex_expression :
  (expression ^ 12 + conjugate_expression ^ 12) = 2 := by
  sorry

end evaluate_complex_expression_l371_371812


namespace remainder_23x_32_l371_371029

def poly := 3 * x^5 + 2 * x^4 - x^3 - 4 * x + 5
def div_poly := (x + 1) * (x + 3)
def rem_poly := 23 * x + 32

theorem remainder_23x_32 (q : ℤ[X]) : 
  poly = div_poly * q + rem_poly :=
sorry

end remainder_23x_32_l371_371029


namespace directional_derivative_correct_l371_371458

noncomputable def function_z (x y : ℝ) : ℝ :=
  x^2 - x * y + y^3

def partial_derivative_x (x y : ℝ) : ℝ :=
  2 * x - y

def partial_derivative_y (x y : ℝ) : ℝ :=
  -x + 3 * y^2

def point_A := (1 : ℝ, -1 : ℝ)
def vector_a := (3 : ℝ, -4 : ℝ)

def gradient_at_A : ℝ × ℝ :=
  (partial_derivative_x point_A.1 point_A.2, partial_derivative_y point_A.1 point_A.2)

def unit_vector_a : ℝ × ℝ :=
  let norm_a := real.sqrt (3^2 + (-4)^2) in
  (3 / norm_a, -4 / norm_a)

def directional_derivative_at_A : ℝ :=
  gradient_at_A.1 * unit_vector_a.1 + gradient_at_A.2 * unit_vector_a.2

theorem directional_derivative_correct :
  directional_derivative_at_A = 1 / 5 := by
  sorry

end directional_derivative_correct_l371_371458


namespace expected_yolks_correct_l371_371384

-- Define the conditions
def total_eggs : ℕ := 15
def double_yolk_eggs : ℕ := 5
def triple_yolk_eggs : ℕ := 3
def single_yolk_eggs : ℕ := total_eggs - double_yolk_eggs - triple_yolk_eggs
def extra_yolk_prob : ℝ := 0.10

-- Define the expected number of yolks calculation
noncomputable def expected_yolks : ℝ :=
  (single_yolk_eggs * 1) + 
  (double_yolk_eggs * 2) + 
  (triple_yolk_eggs * 3) + 
  (double_yolk_eggs * extra_yolk_prob) + 
  (triple_yolk_eggs * extra_yolk_prob)

-- State that the expected number of total yolks is 26.8
theorem expected_yolks_correct : expected_yolks = 26.8 := by
  -- solution would go here
  sorry

end expected_yolks_correct_l371_371384


namespace max_students_seated_in_classroom_l371_371557

theorem max_students_seated_in_classroom : 
  (let n := 8 in -- total number of rows
   let a₁ := 10 in -- number of desks in the first row
   let d := 2 in -- common difference
   let an := a₁ + (n - 1) * d in -- last term of the arithmetic sequence
   let Sn := n / 2 * (a₁ + an) in -- sum of the arithmetic sequence
   Sn = 136) :=
by sorry

end max_students_seated_in_classroom_l371_371557


namespace number_of_possible_orders_l371_371129

def number_of_finishing_orders : ℕ := 4 * 3 * 2 * 1

theorem number_of_possible_orders : number_of_finishing_orders = 24 := 
by
  have h : number_of_finishing_orders = 24 := by norm_num
  exact h

end number_of_possible_orders_l371_371129


namespace sales_function_profit_function_max_profit_value_l371_371753

namespace MaxProfit

-- Define the initial cost and selling price
def cost : ℕ := 30
def initial_price : ℕ := 50

-- Define initial weekly sales and rate of change in sales
def initial_sales : ℕ := 300
def change_rate : ℤ := -10

-- Define the maximum selling price limit
def max_increase_limit : ℕ := 20

-- Define the function relationship between y (weekly sales) and x (price increase)
def sales (x : ℕ) : ℕ := initial_sales - x * -change_rate

-- Define the function relationship between w (weekly profit) and x (price increase)
def profit (x : ℕ) : ℤ := (initial_price + x - cost) * (initial_sales - x * -change_rate)

-- Prove the properties
theorem sales_function (x : ℕ) (h : x ≤ max_increase_limit) : 
  sales x = 300 - 10 * x :=
by { sorry }

theorem profit_function (x : ℕ) :
  profit x = -10 * x^2 + 100 * x + 6000 :=
by { sorry }

theorem max_profit_value (x : ℕ) (hx : x = 5) : 
  profit x = 6250 :=
by { sorry }

end MaxProfit

end sales_function_profit_function_max_profit_value_l371_371753


namespace flower_percentages_l371_371616

def total_flowers : ℕ := 30
def red_flowers : ℕ := 7
def white_flowers : ℕ := 6
def blue_flowers : ℕ := 5
def yellow_flowers : ℕ := 4
def purple_flowers : ℕ := total_flowers - (red_flowers + white_flowers + blue_flowers + yellow_flowers)

def percentage (part total : ℕ) : ℝ := (part.to_real / total.to_real) * 100

theorem flower_percentages :
    percentage (red_flowers + white_flowers + blue_flowers) total_flowers = 60 ∧
    percentage purple_flowers total_flowers ≈ 26.67 ∧
    percentage yellow_flowers total_flowers ≈ 13.33 :=
by sorry

end flower_percentages_l371_371616


namespace monthly_salary_l371_371387

variables (S : ℝ) (savings : ℝ) (new_expenses : ℝ)

theorem monthly_salary (h1 : savings = 0.20 * S)
                      (h2 : new_expenses = 0.96 * S)
                      (h3 : S = 200 + new_expenses) :
                      S = 5000 :=
by
  sorry

end monthly_salary_l371_371387


namespace product_inequality_l371_371610

theorem product_inequality
  (n : ℕ)
  (x : Fin n → ℝ)
  (hx_pos : ∀ k, 0 < x k)
  (hx_sum : ∑ k, x k = 1) :
  (∏ k, (1 + x k) / (x k)) ≥ (∏ k, (n - x k) / (1 - x k)) :=
sorry

end product_inequality_l371_371610


namespace value_of_f_at_pi_over_12_l371_371102

noncomputable def f (x : ℝ) : ℝ := 
  cos^2 (π / 4 + x) - cos^2 (π / 4 - x)

theorem value_of_f_at_pi_over_12 : 
  f (π / 12) = -1 / 2 :=
sorry

end value_of_f_at_pi_over_12_l371_371102


namespace bees_population_reduction_l371_371755

theorem bees_population_reduction :
  ∀ (initial_population loss_per_day : ℕ),
  initial_population = 80000 → 
  loss_per_day = 1200 → 
  ∃ days : ℕ, initial_population - days * loss_per_day = initial_population / 4 ∧ days = 50 :=
by
  intros initial_population loss_per_day h_initial h_loss
  use 50
  sorry

end bees_population_reduction_l371_371755


namespace oreo_shop_l371_371773

theorem oreo_shop (alpha_oreos alpha_milks beta_oreos : ℕ) (h1 : alpha_oreos = 6) (h2 : alpha_milks = 4) (h3 : beta_oreos = 6) :
  let total_ways :=
    (Nat.choose (alpha_oreos + alpha_milks) 3) +
    (Nat.choose (alpha_oreos + alpha_milks) 2) * beta_oreos +
    (Nat.choose (alpha_oreos + alpha_milks) 1) * (Nat.choose beta_oreos 2 + beta_oreos) +
    (Nat.choose beta_oreos 3 + beta_oreos * (beta_oreos - 1) + beta_oreos) in
  total_ways = 656 :=
by
  intro total_ways
  sorry

end oreo_shop_l371_371773


namespace inequality_abc_l371_371250

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) : 
  a^2 * b^2 + b^2 * c^2 + a^2 * c^2 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end inequality_abc_l371_371250


namespace arrangements_not_adjacent_to_c_l371_371487

theorem arrangements_not_adjacent_to_c : 
  ∃ (arrangements : ℕ), arrangements = 36 ∧
  (∀ (lst : List Char), lst.permutes ['a', 'b', 'c', 'd', 'e'] → 
    (∀ i, (lst[i] = 'a' ∨ lst[i] = 'b') → (lst[i + 1] ≠ 'c' ∧ lst[i - 1] ≠ 'c'))) :=
sorry

end arrangements_not_adjacent_to_c_l371_371487


namespace find_equation_of_line_l371_371085

def point := (ℝ × ℝ)

def line := ℝ → ℝ → Prop

def intersects (l1 l2 : line) (p : point) : Prop :=
  l1 p.1 p.2 ∧ l2 p.1 p.2

def line_through (p : point) (m : ℝ) : line :=
  λ x y, y = m * (x - p.1) + p.2

theorem find_equation_of_line
  (l_intersect : line) 
  (slope : ℝ) 
  (p: point) 
  (intersect_lines: ∀ (x y : ℝ), 
    (x - y + 1 = 0) → (2 * x - y = 0) → x = 1 ∧ y = 2 :=
  (λ x y eq1 eq2, ⟨eq1▪1!.symm, eq2▪2!.symm⟩))
  (distance : ℝ)
  (parallel_lines: ∀ (x y : ℝ), 
    (3 * x - y + C = 0) → (|1 - m| / √(3^2 + 1^2) = distance) → 
      line = 3x - y = 0 ∨ line = 3x - y - 2 = 0 :
sorry :=
  by
  sorry

end find_equation_of_line_l371_371085


namespace geometric_sequence_problem_l371_371570

noncomputable def a : ℕ → ℝ := sorry

theorem geometric_sequence_problem :
  a 4 = 4 →
  a 8 = 8 →
  a 12 = 16 :=
by
  intros h4 h8
  sorry

end geometric_sequence_problem_l371_371570


namespace smallest_period_of_f_intervals_of_monotonic_increase_value_of_m_l371_371230

def vec_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
def vec_b (x m : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x) + m)
def f (x m : ℝ) : ℝ := let (a1, a2) := vec_a x; let (b1, b2) := vec_b x m; a1 * b1 + a2 * b2

theorem smallest_period_of_f (m : ℝ) : 
  (∀ T > 0, (∀ x, f x m = f (x + T) m) → T ≥ Real.pi) ∧ 
  (∃ T > 0, (∀ x, f x m = f (x + T) m) ∧ T = Real.pi) := 
sorry

theorem intervals_of_monotonic_increase (m : ℝ) : 
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi → x ∈ set.Icc 0 (Real.pi / 6) ∨ x ∈ set.Icc (2 * Real.pi / 3) Real.pi → f x m < f y m) := 
sorry

theorem value_of_m : 
  (∀ x ∈ set.Icc 0 (Real.pi / 6), f x m ≤ 4) → 
  f (Real.pi / 6) m = 4 → 
  m = 1 := 
sorry

end smallest_period_of_f_intervals_of_monotonic_increase_value_of_m_l371_371230


namespace max_rectangle_area_l371_371692

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l371_371692


namespace no_stromino_covering_of_5x5_board_l371_371765

-- Define the conditions
def isStromino (r : ℕ) (c : ℕ) : Prop := 
  (r = 3 ∧ c = 1) ∨ (r = 1 ∧ c = 3)

def is5x5Board (r c : ℕ) : Prop := 
  r = 5 ∧ c = 5

-- The main goal is to show this proposition
theorem no_stromino_covering_of_5x5_board : 
  ∀ (board_size : ℕ × ℕ),
    is5x5Board board_size.1 board_size.2 →
    ∀ (stromino_count : ℕ),
      stromino_count = 16 →
      (∀ (stromino : ℕ × ℕ), 
        isStromino stromino.1 stromino.2 →
        ∀ (cover : ℕ), 
          3 = cover) →
      ¬(∃ (cover_fn : ℕ × ℕ → ℕ), 
          (∀ (pos : ℕ × ℕ), pos.fst < 5 ∧ pos.snd < 5 →
            cover_fn pos = 1 ∨ cover_fn pos = 2) ∧
          (∀ (i : ℕ), i < 25 → 
            ∃ (stromino_pos : ℕ × ℕ), 
              stromino_pos.fst < 5 ∧ stromino_pos.snd < 5 ∧ 
              -- Each stromino must cover exactly 3 squares, 
              -- which implies that the covering function must work appropriately.
              (cover_fn (stromino_pos.fst, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 1, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 2, stromino_pos.snd) = 3 ∨
               cover_fn (stromino_pos.fst, stromino_pos.snd + 1) +
               cover_fn (stromino_pos.fst, stromino_pos.snd + 2) = 3))) :=
by sorry

end no_stromino_covering_of_5x5_board_l371_371765


namespace complex_star_angle_sum_correct_l371_371790

-- Definitions corresponding to the conditions
def complex_star_interior_angle_sum (n : ℕ) (h : n ≥ 7) : ℕ :=
  180 * (n - 4)

-- The theorem stating the problem
theorem complex_star_angle_sum_correct (n : ℕ) (h : n ≥ 7) :
  complex_star_interior_angle_sum n h = 180 * (n - 4) :=
sorry

end complex_star_angle_sum_correct_l371_371790


namespace batsman_average_after_12th_innings_l371_371738

theorem batsman_average_after_12th_innings (A : ℝ) (h_average_increase : 65 = 12 * (A + 2) - 11 * A) :
  A + 2 = 43 := 
by
  -- Let average_after_11 be the average after 11 innings. Use the given condition to solve for average_after_11.
  have h1 : 12 * (A + 2) - 11 * A = 43 :=
    by linarith
  rw h1
  sorry

end batsman_average_after_12th_innings_l371_371738


namespace odd_operations_l371_371107

theorem odd_operations (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ j : ℤ, b = 2 * j + 1) :
  (∃ k : ℤ, (a * b) = 2 * k + 1) ∧ (∃ m : ℤ, a^2 = 2 * m + 1) :=
by {
  sorry
}

end odd_operations_l371_371107


namespace sophie_perceived_height_in_mirror_l371_371655

noncomputable def inch_to_cm : ℝ := 2.5

noncomputable def sophie_height_in_inches : ℝ := 50

noncomputable def sophie_height_in_cm := sophie_height_in_inches * inch_to_cm

noncomputable def perceived_height := sophie_height_in_cm * 2

theorem sophie_perceived_height_in_mirror : perceived_height = 250 :=
by
  unfold perceived_height
  unfold sophie_height_in_cm
  unfold sophie_height_in_inches
  unfold inch_to_cm
  sorry

end sophie_perceived_height_in_mirror_l371_371655


namespace digit_9_never_appears_in_sequence_digit_at_100th_position_is_5_l371_371720

theorem digit_9_never_appears_in_sequence : 
  ¬ ∃ n : ℕ, get_digit(n) = 9 :=
sorry

theorem digit_at_100th_position_is_5 :
  get_digit(100) = 5 :=
sorry

end digit_9_never_appears_in_sequence_digit_at_100th_position_is_5_l371_371720


namespace countNumbersWithDigit3_l371_371149

-- Conditions: Define the range of numbers
def isInRange (n : Nat) : Prop := 200 ≤ n ∧ n ≤ 599

-- Define digit conditions
def containsDigit3 (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.contains 3

-- The main statement asserting the required property
theorem countNumbersWithDigit3 : 
  (Finset.filter (λ n => containsDigit3 n) (Finset.range' 200 400)).card = 157 :=
by 
  sorry

end countNumbersWithDigit3_l371_371149


namespace complex_poly_root_exists_l371_371608

noncomputable def polynomial_has_complex_root (P : Polynomial ℂ) : Prop :=
  ∃ z : ℂ, P.eval z = 0

theorem complex_poly_root_exists (P : Polynomial ℂ) : polynomial_has_complex_root P :=
sorry

end complex_poly_root_exists_l371_371608


namespace make_all_green_l371_371810

-- Define the nature of cuates
def cuates (m n : ℕ) : Prop := (m / n).natAbs.prime ∨ (n / m).natAbs.prime

-- Define the problem
theorem make_all_green :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 4027 → (n.color = green ∨ n.color = red)) →
  (∀ m n : ℕ, cuates m n → step_possible m n (m.color, n.color) (toggle m.color, toggle n.color)) →
  ∃ seq : list (ℕ × ℕ), (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2014 → (apply_steps seq k).color = green) :=
by
  sorry

end make_all_green_l371_371810


namespace simplified_expression_result_l371_371044

theorem simplified_expression_result :
  ((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by {
  sorry
}

end simplified_expression_result_l371_371044


namespace smallest_N_for_pairs_l371_371682

theorem smallest_N_for_pairs :
  ∃ N : ℕ, (∀ (a b : ℕ), a ≠ b ∧ a ∈ {1, 2, ..., 2016} ∧ b ∈ {1, 2, ..., 2016} →
  ∃ (pairs : list (ℕ × ℕ)), (∀ (p : ℕ × ℕ) ∈ pairs, (p.1 * p.2 ≤ N)) ∧ N = 1017072) := by
  sorry

end smallest_N_for_pairs_l371_371682


namespace find_m_l371_371879

-- Circle equation: x^2 + y^2 + 2x - 6y + 1 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 6 * y + 1 = 0

-- Line equation: x + m * y + 4 = 0
def line_eq (x y m : ℝ) : Prop := x + m * y + 4 = 0

-- Prove that the value of m such that the center of the circle lies on the line is -1
theorem find_m (m : ℝ) : 
  (∃ x y : ℝ, circle_eq x y ∧ (x, y) = (-1, 3) ∧ line_eq x y m) → m = -1 :=
by {
  sorry
}

end find_m_l371_371879


namespace num_ints_between_150_250_same_remainder_7_9_l371_371145

theorem num_ints_between_150_250_same_remainder_7_9 : 
  {n : ℤ | 150 < n ∧ n < 250 ∧ ∃ r : ℤ, (n ≡ r [MOD 7]) ∧ (n ≡ r [MOD 9])}.card = 7 := 
by
  sorry

end num_ints_between_150_250_same_remainder_7_9_l371_371145


namespace find_b_l371_371567

noncomputable def hyperbola_and_circle (b : ℝ) : Prop :=
  let c := Real.sqrt (1 + b^2) in
  let F1 := (-c, 0) in
  let F2 := (c, 0) in
  let equation_hyperbola (x y : ℝ) := x^2 - y^2 / b^2 = 1 in
  let equation_circle (x y : ℝ) := x^2 + y^2 = 1 in
  equation_hyperbola (F1.1) (F1.2) ∧ equation_hyperbola (F2.1) (F2.2) ∧ 
  (∀ (A B : ℝ × ℝ), tangent_through F1 equation_circle → A = (1, 1) ∧ B = (1, -1)) →
  (dist F2 (1, -1) = dist (1, -1) (1, 1))

theorem find_b : 
  (∃ b : ℝ, hyperbola_and_circle b) → b = 2 + Real.sqrt 7 :=
sorry

end find_b_l371_371567


namespace ivan_running_distance_l371_371583

theorem ivan_running_distance (x MondayDistance TuesdayDistance WednesdayDistance ThursdayDistance FridayDistance : ℝ) 
  (h1 : MondayDistance = x)
  (h2 : TuesdayDistance = 2 * x)
  (h3 : WednesdayDistance = x)
  (h4 : ThursdayDistance = (1 / 2) * x)
  (h5 : FridayDistance = x)
  (hShortest : ThursdayDistance = 5) :
  MondayDistance + TuesdayDistance + WednesdayDistance + ThursdayDistance + FridayDistance = 55 :=
by
  sorry

end ivan_running_distance_l371_371583


namespace sqrt_of_9_l371_371710

theorem sqrt_of_9 (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
by {
  sorry
}

end sqrt_of_9_l371_371710


namespace instantaneous_velocity_at_3_l371_371269

def displacement (t : ℝ) : ℝ := 4 - 2 * t + t^2

theorem instantaneous_velocity_at_3 :
  let velocity (t : ℝ) := deriv displacement t in 
  velocity 3 = 4 :=
by
  -- Proof goes here
  sorry

end instantaneous_velocity_at_3_l371_371269


namespace problem_l371_371508

open Real

theorem problem (x y : ℝ) (h1 : 3 * x + 2 * y = 8) (h2 : 2 * x + 3 * y = 11) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 2041 / 25 :=
sorry

end problem_l371_371508


namespace scientific_notation_example_l371_371405

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end scientific_notation_example_l371_371405


namespace multiple_of_totient_l371_371966

theorem multiple_of_totient (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a : ℕ), ∀ (i : ℕ), 0 ≤ i ∧ i ≤ n → m ∣ Nat.totient (a + i) :=
by
sorry

end multiple_of_totient_l371_371966


namespace smallest_N_for_pairs_l371_371680

theorem smallest_N_for_pairs :
  ∃ N : ℕ, (∀ (a b : ℕ), a ≠ b ∧ a ∈ {1, 2, ..., 2016} ∧ b ∈ {1, 2, ..., 2016} →
  ∃ (pairs : list (ℕ × ℕ)), (∀ (p : ℕ × ℕ) ∈ pairs, (p.1 * p.2 ≤ N)) ∧ N = 1017072) := by
  sorry

end smallest_N_for_pairs_l371_371680


namespace find_third_number_l371_371914

theorem find_third_number (n : ℕ) :
  let p := (125 * 243 * n) / 405 in
  1000 ≤ p ∧ p < 10000 → n = 14 :=
by
  sorry

end find_third_number_l371_371914


namespace abs_quadratic_eq_linear_iff_l371_371823

theorem abs_quadratic_eq_linear_iff (x : ℝ) : 
  (|x^2 - 5*x + 6| = x + 2) ↔ (x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) :=
by
  sorry

end abs_quadratic_eq_linear_iff_l371_371823


namespace problem1_solution_problem2_solution_l371_371833

-- Problem 1: 
theorem problem1_solution (x : ℝ) (h : 4 * x^2 = 9) : x = 3 / 2 ∨ x = - (3 / 2) := 
by sorry

-- Problem 2: 
theorem problem2_solution (x : ℝ) (h : (1 - 2 * x)^3 = 8) : x = - 1 / 2 := 
by sorry

end problem1_solution_problem2_solution_l371_371833


namespace tripod_height_floor_l371_371767

-- Define constants and variables
constant initial_leg_length : ℝ := 6
constant initial_height : ℝ := 5
constant broken_length : ℝ := 2
variable m : ℕ
variable n : ℕ
variable h : ℝ

-- Define the conditions as Lean code
def remains_leg_length : ℝ := initial_leg_length - broken_length
def new_h := remains_leg_length * (initial_height / initial_leg_length) -- Hypothetical formula

-- Main theorem statement
theorem tripod_height_floor (m n : ℕ) :
  (h = m / Real.sqrt ↑n) → (∀ p : ℕ, (p^2 ∣ n) → p = 1 ∨ p = n) → 
  (h = 384 / Real.sqrt 1945) → 
  (Real.toFloor (m + Real.sqrt n) = 428) :=
sorry

end tripod_height_floor_l371_371767


namespace race_permutations_l371_371133

-- Define the problem conditions: four participants
def participants : Nat := 4

-- Define the factorial function for permutations
def factorial : Nat → Nat
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem: The number of different possible orders in which Harry, Ron, Neville, and Hermione can finish is 24
theorem race_permutations : factorial participants = 24 :=
by
  simp [participants, factorial]
  sorry

end race_permutations_l371_371133


namespace evaluate_neg64_to_7_over_3_l371_371014

theorem evaluate_neg64_to_7_over_3 (a : ℝ) (b : ℝ) (c : ℝ) 
  (h1 : a = -64) (h2 : b = (-4)) (h3 : c = (7/3)) :
  a ^ c = -65536 := 
by
  have h4 : (-64 : ℝ) = (-4) ^ 3 := by sorry
  have h5 : a = b^3 := by rw [h1, h2, h4]
  have h6 : a ^ c = (b^3) ^ (7/3) := by rw [←h5, h3]
  have h7 : (b^3)^c = b^(3*(7/3)) := by sorry
  have h8 : b^(3*(7/3)) = b^7 := by norm_num
  have h9 : b^7 = -65536 := by sorry
  rw [h6, h7, h8, h9]
  exact h9

end evaluate_neg64_to_7_over_3_l371_371014
